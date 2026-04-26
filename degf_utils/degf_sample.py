import copy
import inspect
import re
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput


_QUANTIFIER_WORDS = {
    "a", "an", "the", "this", "that", "these", "those", "some", "many", "much", "few", "several",
    "all", "each", "every", "any", "both", "either", "neither", "another", "other", "two", "three",
    "four", "five", "six", "seven", "eight", "nine", "ten",
}
_COMMON_ADJECTIVES = {
    "red", "green", "blue", "yellow", "white", "black", "gray", "brown", "orange", "pink", "purple",
    "small", "large", "big", "tiny", "young", "old", "new", "wooden", "metal", "plastic", "glass",
}
_ADJECTIVE_SUFFIXES = ("y", "ive", "ous", "ful", "less", "al", "ic", "ish", "ary", "ate", "ed", "ing")
_NON_NOUN_STOPWORDS = {
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "have", "has", "had",
    "to", "for", "from", "with", "without", "by", "on", "in", "at", "into", "onto", "over", "under", "after",
    "before", "while", "because", "if", "then", "than", "as", "or", "but", "so", "very",
}

_DEFAULT_CLIP_MODEL_PATH = "/home/ciram25-liurp/models/clip-vit-base-patch32"


def _decode_single_token(tokenizer: Any, token_id: int) -> str:
    if tokenizer is None:
        return ""
    try:
        return tokenizer.decode([token_id], skip_special_tokens=True)
    except Exception:
        return ""


def _get_runtime_context(model: Any) -> Dict[str, Any]:
    ctx = getattr(model, "_degf_runtime_context", None)
    if not isinstance(ctx, dict):
        ctx = {}
        setattr(model, "_degf_runtime_context", ctx)
    return ctx


def _get_degf_option(model: Any, model_kwargs: Dict[str, Any], key: str, default: Any) -> Any:
    if key in model_kwargs:
        return model_kwargs[key]

    ctx = _get_runtime_context(model)
    if key in ctx:
        return ctx[key]

    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and hasattr(gen_cfg, key):
        return getattr(gen_cfg, key)

    return default


def _token_to_candidate_phrase(token_text: str) -> str:
    normalized = _normalize_piece(token_text)
    if not normalized:
        return ""
    cleaned = normalized.lower().strip("\"'`()[]{}")
    cleaned = cleaned.strip("\.,;:!?")
    if not cleaned or not any(ch.isalpha() for ch in cleaned):
        return ""
    return cleaned


def _compute_clip_grounding_risk_for_phrase(
    model: Any,
    model_kwargs: Dict[str, Any],
    phrase: str,
) -> Optional[float]:
    if not phrase:
        return None

    raw_image = _get_degf_option(model, model_kwargs, "degf_raw_image", None)
    if raw_image is None:
        return None

    runtime_ctx = _get_runtime_context(model)
    clip_bundle = _get_degf_option(model, model_kwargs, "degf_clip_bundle", None)

    if clip_bundle is None:
        clip_bundle = runtime_ctx.get("degf_clip_bundle_fallback")

    if clip_bundle is None:
        from degf_utils.clip_text_risk import load_clip_risk_bundle

        clip_model_path = _get_degf_option(model, model_kwargs, "degf_clip_model_path", _DEFAULT_CLIP_MODEL_PATH)
        clip_device = _get_degf_option(model, model_kwargs, "degf_clip_device", "cuda")
        clip_bundle = load_clip_risk_bundle(model_path=clip_model_path, device=clip_device)
        runtime_ctx["degf_clip_bundle_fallback"] = clip_bundle

    image_cache_key = id(raw_image)
    image_feature = runtime_ctx.get("degf_clip_image_feature", None)
    image_feature_key = runtime_ctx.get("degf_clip_image_feature_key", None)

    if image_feature is None or image_cache_key != image_feature_key:
        from degf_utils.clip_text_risk import prepare_clip_image_feature

        image_feature = prepare_clip_image_feature(clip_bundle, raw_image)
        runtime_ctx["degf_clip_image_feature"] = image_feature
        runtime_ctx["degf_clip_image_feature_key"] = image_cache_key

    from degf_utils.clip_text_risk import score_text_grounding_risk

    clip_prompt_template = _get_degf_option(model, model_kwargs, "degf_clip_prompt_template", "a photo of {}")
    return score_text_grounding_risk(
        bundle=clip_bundle,
        image_feature=image_feature,
        phrase=phrase,
        prompt_template=clip_prompt_template,
    )


def _compute_clip_grounding_risk_for_token_id(
    model: Any,
    model_kwargs: Dict[str, Any],
    tokenizer: Any,
    token_id: int,
) -> Optional[float]:
    token_text = _decode_single_token(tokenizer, token_id)
    phrase = _token_to_candidate_phrase(token_text)
    return _compute_clip_grounding_risk_for_phrase(model, model_kwargs, phrase)


def _estimate_top1_clip_grounding_risk(
    model: Any,
    model_kwargs: Dict[str, Any],
    tokenizer: Any,
    next_token_logits: torch.Tensor,
) -> Optional[float]:
    if tokenizer is None:
        return None
    if next_token_logits.ndim != 2 or next_token_logits.shape[0] != 1:
        return None
    top_token_id = int(torch.argmax(next_token_logits[0], dim=-1).item())
    return _compute_clip_grounding_risk_for_token_id(model, model_kwargs, tokenizer, top_token_id)


def _maybe_apply_sd_logit_intervention(
    model: Any,
    next_token_scores: torch.Tensor,
    model_kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, bool]:
    if not bool(_get_degf_option(model, model_kwargs, "degf_enable_sd_logit_intervention", False)):
        return next_token_scores, False

    if next_token_scores.ndim != 2 or next_token_scores.shape[0] != 1:
        return next_token_scores, False

    degf_tokenizer = _get_degf_option(model, model_kwargs, "degf_tokenizer", None)
    sd_pipe = _get_degf_option(model, model_kwargs, "degf_sd_pipe", None)
    raw_image = _get_degf_option(model, model_kwargs, "degf_raw_image", None)
    if degf_tokenizer is None or sd_pipe is None or raw_image is None:
        return next_token_scores, False

    topk = int(_get_degf_option(model, model_kwargs, "degf_sd_topk", 8))
    if topk <= 0:
        return next_token_scores, False
    topk = min(topk, int(next_token_scores.shape[-1]))

    sd_lambda = float(_get_degf_option(model, model_kwargs, "degf_sd_lambda", 1.0))
    sd_timestep = int(_get_degf_option(model, model_kwargs, "degf_sd_timestep", 500))
    sd_unet_batch = int(_get_degf_option(model, model_kwargs, "degf_sd_unet_batch_size", 0))
    sd_seed = _get_degf_option(model, model_kwargs, "degf_sd_seed", None)

    try:
        # Lazy import to avoid diffusers overhead when this feature is disabled.
        from degf_utils.sd_feedback_scoring import prepare_sd_feedback_context, score_prompts_with_sd_context

        runtime_ctx = _get_runtime_context(model)
        context = runtime_ctx.get("degf_sd_context")
        if context is None or int(context.get("noise_timestep", -1)) != sd_timestep:
            context = prepare_sd_feedback_context(
                pipe=sd_pipe,
                image=raw_image,
                noise_timestep=sd_timestep,
                seed=sd_seed,
            )
            runtime_ctx["degf_sd_context"] = context

        _, top_indices = torch.topk(next_token_scores[0], k=topk)

        prompt_to_token_ids: Dict[str, List[int]] = {}
        prefix_tokens = runtime_ctx.get("degf_sd_prefix_tokens", [])
        if not isinstance(prefix_tokens, list):
            prefix_tokens = []
        for token_id in top_indices.tolist():
            token_text = _decode_single_token(degf_tokenizer, int(token_id))
            role, token = _classify_token_role(token_text)
            if role != "noun" or not token:
                continue
            phrase_tokens = [t for t in prefix_tokens if t]
            phrase_tokens.append(token)
            phrase = " ".join(phrase_tokens).strip()
            if not phrase:
                continue
            prompt = f"a photo of {phrase}"
            if prompt not in prompt_to_token_ids:
                prompt_to_token_ids[prompt] = []
            prompt_to_token_ids[prompt].append(int(token_id))

        if len(prompt_to_token_ids) == 0:
            return next_token_scores, False

        feedback_rows = score_prompts_with_sd_context(
            pipe=sd_pipe,
            context=context,
            candidate_prompts=list(prompt_to_token_ids.keys()),
            unet_batch_size=sd_unet_batch,
        )

        adjusted_scores = next_token_scores.clone()
        for row in feedback_rows:
            prompt = row["prompt"]
            score = float(row["score"])
            for token_id in prompt_to_token_ids.get(prompt, []):
                adjusted_scores[0, token_id] = adjusted_scores[0, token_id] + sd_lambda * score

        return adjusted_scores, True
    except Exception as e:
        if bool(_get_degf_option(model, model_kwargs, "degf_debug_sd_logit_intervention", False)):
            print(f"[DeGF SD intervene] skipped due to error: {e}")
        return next_token_scores, False


def _normalize_piece(text: str) -> str:
    if not text:
        return ""
    t = text.replace("▁", " ").replace("Ġ", " ").replace("Ċ", " ")
    return t.strip()


def _classify_token_role(text: str) -> Tuple[str, str]:
    normalized = _normalize_piece(text)
    if not normalized:
        return "other", ""

    cleaned = normalized.lower().strip("\"'`()[]{}")
    if not cleaned:
        return "other", ""

    if re.fullmatch(r"[\.,;:!?]+", cleaned):
        return "boundary", cleaned

    alpha = cleaned.strip("\.,;:!?")
    if not alpha:
        return "boundary", cleaned

    if any(ch.isdigit() for ch in alpha) or alpha in _QUANTIFIER_WORDS:
        return "quant", alpha

    if alpha in _COMMON_ADJECTIVES or alpha.endswith(_ADJECTIVE_SUFFIXES):
        return "adj", alpha

    if alpha in _NON_NOUN_STOPWORDS:
        return "other", alpha

    if alpha.isalpha():
        return "noun", alpha

    return "other", alpha


@dataclass
class NounPhraseRiskCache:
    enabled: bool
    risk_threshold: float
    max_cache_size: int = 64
    prefix_tokens: List[str] = field(default_factory=list)
    noun_phrases: List[str] = field(default_factory=list)
    delayed_trigger: bool = False
    correction_next_step: bool = False

    def consume_correction_flag(self) -> bool:
        trigger = self.correction_next_step
        self.correction_next_step = False
        return trigger

    def observe_token(self, token_text: str, token_risk: Optional[float]) -> Tuple[str, str, bool]:
        role, token = _classify_token_role(token_text)
        if not self.enabled:
            return role, "", False

        risk_is_high = token_risk is not None and token_risk >= self.risk_threshold

        if role in ("adj", "quant"):
            if token:
                self.prefix_tokens.append(token)
            if risk_is_high:
                self.delayed_trigger = True
            return role, "", False

        if role == "noun":
            phrase_tokens = [t for t in self.prefix_tokens if t]
            if token:
                phrase_tokens.append(token)
            phrase = " ".join(phrase_tokens).strip() or token
            self.prefix_tokens.clear()

            if phrase:
                self.noun_phrases.append(phrase)
                if len(self.noun_phrases) > self.max_cache_size:
                    self.noun_phrases = self.noun_phrases[-self.max_cache_size:]

            should_trigger = bool(risk_is_high or self.delayed_trigger)
            self.delayed_trigger = False
            self.correction_next_step = should_trigger
            return role, phrase, should_trigger

        if role == "boundary":
            self.prefix_tokens.clear()
            self.delayed_trigger = False
            return role, "", False

        if token not in {"of", "and"}:
            self.prefix_tokens.clear()
            self.delayed_trigger = False
        return role, "", False


def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only

    # auto-regressive generation
    model_kwargs_pos = model_kwargs.copy()
    model_kwargs_neg = model_kwargs.copy()
    
    print("use_ritual = ", model_kwargs.get("use_ritual"))
    print("use_vcd = ", model_kwargs.get("use_vcd"))
    print("use_m3id = ", model_kwargs.get("use_m3id"))
    print("use_diffusion = ", model_kwargs.get("use_diffusion"))
    
    print("pos = ", model_kwargs.get("degf_alpha_pos"))
    print("neg = ", model_kwargs.get("degf_alpha_neg"))
    print("beta = ", model_kwargs.get("degf_beta"))    
    
    t=0
    risk_count = 0
    token_count = 0
    risk_list = []
    risk_threshold = float(_get_degf_option(self, model_kwargs, "degf_risk_threshold", 0.1))
    np_cache = NounPhraseRiskCache(
        enabled=bool(_get_degf_option(self, model_kwargs, "degf_enable_np_cache", False)),
        risk_threshold=risk_threshold,
        max_cache_size=int(_get_degf_option(self, model_kwargs, "degf_np_cache_size", 64)),
    )
    degf_tokenizer = _get_degf_option(self, model_kwargs, "degf_tokenizer", None)
    risk_metric = str(_get_degf_option(self, model_kwargs, "degf_risk_metric", "clip")).lower()
    use_clip_risk_prefilter = risk_metric == "clip"
    sd_feedback_enabled = bool(_get_degf_option(self, model_kwargs, "degf_enable_sd_logit_intervention", False))
    np_cache_active = np_cache.enabled and degf_tokenizer is not None and use_clip_risk_prefilter
    sd_np_cache = NounPhraseRiskCache(
        enabled=sd_feedback_enabled and degf_tokenizer is not None,
        risk_threshold=risk_threshold,
        max_cache_size=int(_get_degf_option(self, model_kwargs, "degf_np_cache_size", 64)),
    )
    
    while True:
        force_correction_this_step = np_cache.consume_correction_flag() if np_cache_active else False
        force_sd_intervention_this_step = sd_np_cache.delayed_trigger if sd_np_cache.enabled else False
        token_risk_value = None
        apply_sd_intervention_this_step = False

        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # image_size: 336x336

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions, # True
            output_hidden_states=output_hidden_states
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        
        ## For complementive & contrastive decoding
        use_ritual = model_kwargs.get("use_ritual")
        use_vcd = model_kwargs.get("use_vcd")
        use_m3id = model_kwargs.get("use_m3id")
        use_diffusion = model_kwargs.get("use_diffusion")
        
        if use_ritual or use_vcd or use_m3id or use_diffusion:
            next_token_logits_pos = next_token_logits
            next_token_logits_neg = next_token_logits

            if model_kwargs["images_pos"] is not None and use_ritual:
                model_inputs_pos = self.prepare_inputs_for_generation_pos(input_ids, **model_kwargs_pos)
                outputs_pos = self(
                    **model_inputs_pos,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                next_token_logits_pos = outputs_pos.logits[:, -1, :]

            elif model_kwargs["images_neg"] is not None and use_vcd:
                model_inputs_neg = self.prepare_inputs_for_generation_neg(input_ids, **model_kwargs_neg)
                outputs_neg = self(
                    **model_inputs_neg,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :]
            elif use_m3id:
                model_inputs_neg = self.prepare_inputs_for_generation_m3id(input_ids, **model_kwargs_neg)
                outputs_neg = self(
                    **model_inputs_neg,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :]
            elif model_kwargs["images_neg"] is not None and use_diffusion:
                model_inputs_neg = self.prepare_inputs_for_generation_neg(input_ids, **model_kwargs_neg)
                outputs_neg = self(
                    **model_inputs_neg,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :]
                
                
            degf_alpha_pos = model_kwargs.get("degf_alpha_pos") if model_kwargs.get("degf_alpha_pos") is not None else 3
            degf_alpha_neg = model_kwargs.get("degf_alpha_neg") if model_kwargs.get("degf_alpha_neg") is not None else 1
            degf_beta = model_kwargs.get("degf_beta") if model_kwargs.get("degf_beta") is not None else 0.1

            # set cutoff for Adaptive Plausibility Constraints
            cutoff = torch.log(torch.tensor(degf_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            
            if use_ritual:
                diffs = (next_token_logits + degf_alpha_pos * next_token_logits_pos)
            elif use_vcd:
                diffs = (1 + degf_alpha_neg) * next_token_logits - degf_alpha_neg * next_token_logits_neg
            elif use_m3id:
                gamma_t = torch.exp(torch.tensor(-0.02*t))
                diffs = next_token_logits + (next_token_logits - next_token_logits_neg)*(1-gamma_t)/gamma_t
                t += 1
            elif use_diffusion:
                if use_clip_risk_prefilter:
                    token_risk_value = _estimate_top1_clip_grounding_risk(
                        self,
                        model_kwargs,
                        degf_tokenizer,
                        next_token_logits,
                    )
                    if token_risk_value is None:
                        token_risk_value = 0.0
                    risk_list.append(format(token_risk_value, '.4f'))
                else:
                    token_risk_value = 0.0
                    risk_list.append("disabled")

                if not use_clip_risk_prefilter:
                    risk_count += 1
                    token_count += 1
                    if sd_feedback_enabled:
                        diffs = next_token_logits
                        apply_sd_intervention_this_step = True
                    else:
                        diffs = (1 + degf_alpha_neg) * next_token_logits - degf_alpha_neg * next_token_logits_neg
                elif np_cache_active:
                    # With NP cache enabled, correction is delayed until a noun (or noun phrase end) confirms the trigger.
                    if force_correction_this_step or force_sd_intervention_this_step:
                        risk_count += 1
                        token_count += 1
                        if sd_feedback_enabled:
                            diffs = next_token_logits
                            apply_sd_intervention_this_step = True
                        else:
                            diffs = (1 + degf_alpha_neg) * next_token_logits - degf_alpha_neg * next_token_logits_neg
                    else:
                        token_count += 1
                        diffs = next_token_logits if sd_feedback_enabled else next_token_logits + degf_alpha_pos * next_token_logits_neg
                else:
                    if token_risk_value < risk_threshold and not force_sd_intervention_this_step:
                        token_count += 1
                        diffs = next_token_logits if sd_feedback_enabled else next_token_logits + degf_alpha_pos * next_token_logits_neg
                    else:
                        risk_count += 1
                        token_count += 1
                        if sd_feedback_enabled:
                            diffs = next_token_logits
                            apply_sd_intervention_this_step = True
                        else:
                            diffs = (1 + degf_alpha_neg) * next_token_logits - degf_alpha_neg * next_token_logits_neg
            
            logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            logits = logits_processor(input_ids, logits)
            logits = logits_warper(input_ids, logits)

            next_token_scores = logits
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        if apply_sd_intervention_this_step:
            runtime_ctx = _get_runtime_context(self)
            runtime_ctx["degf_sd_prefix_tokens"] = list(sd_np_cache.prefix_tokens)
            next_token_scores, sd_intervened = _maybe_apply_sd_logit_intervention(self, next_token_scores, model_kwargs)
        else:
            sd_intervened = False
        if sd_intervened:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        if use_diffusion and sd_np_cache.enabled and next_tokens.numel() == 1:
            token_text = _decode_single_token(degf_tokenizer, int(next_tokens[0].item()))
            token_phrase = _token_to_candidate_phrase(token_text)
            selected_token_risk = _compute_clip_grounding_risk_for_phrase(self, model_kwargs, token_phrase)
            if selected_token_risk is None:
                selected_token_risk = token_risk_value
            sd_np_cache.observe_token(token_text, selected_token_risk)

        if use_diffusion and np_cache_active and next_tokens.numel() == 1:
            token_text = _decode_single_token(degf_tokenizer, int(next_tokens[0].item()))
            token_phrase = _token_to_candidate_phrase(token_text)
            cached_token_risk = _compute_clip_grounding_risk_for_phrase(self, model_kwargs, token_phrase)
            if cached_token_risk is None:
                cached_token_risk = token_risk_value
            _, phrase, triggered = np_cache.observe_token(token_text, cached_token_risk)
            if triggered and bool(_get_degf_option(self, model_kwargs, "degf_debug_np_cache", False)):
                print(f"[DeGF NP cache] trigger next-step correction for phrase: '{phrase}'")

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        if use_ritual:
            model_kwargs_pos = self._update_model_kwargs_for_generation(
                outputs_pos, model_kwargs_pos, is_encoder_decoder=self.config.is_encoder_decoder
            )
        if use_vcd or use_m3id or use_diffusion:
            model_kwargs_neg = self._update_model_kwargs_for_generation(
                outputs_neg, model_kwargs_neg, is_encoder_decoder=self.config.is_encoder_decoder
            )
            
        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    print("risk_count: ", risk_count, "| token_count: ", token_count)
    
    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids, risk_list
    

def greedy_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only

    # auto-regressive generation
    model_kwargs_pos = model_kwargs.copy()
    model_kwargs_neg = model_kwargs.copy()
    
    print("use_ritual = ", model_kwargs.get("use_ritual"))
    print("use_vcd = ", model_kwargs.get("use_vcd"))
    print("use_m3id = ", model_kwargs.get("use_m3id"))
    print("use_diffusion = ", model_kwargs.get("use_diffusion"))
    
    print("pos = ", model_kwargs.get("degf_alpha_pos"))
    print("neg = ", model_kwargs.get("degf_alpha_neg"))
    print("beta = ", model_kwargs.get("degf_beta"))    
    
    t=0
    risk_count = 0
    token_count = 0
    risk_list = []
    risk_threshold = float(_get_degf_option(self, model_kwargs, "degf_risk_threshold", 0.1))
    np_cache = NounPhraseRiskCache(
        enabled=bool(_get_degf_option(self, model_kwargs, "degf_enable_np_cache", False)),
        risk_threshold=risk_threshold,
        max_cache_size=int(_get_degf_option(self, model_kwargs, "degf_np_cache_size", 64)),
    )
    degf_tokenizer = _get_degf_option(self, model_kwargs, "degf_tokenizer", None)
    risk_metric = str(_get_degf_option(self, model_kwargs, "degf_risk_metric", "clip")).lower()
    use_clip_risk_prefilter = risk_metric == "clip"
    sd_feedback_enabled = bool(_get_degf_option(self, model_kwargs, "degf_enable_sd_logit_intervention", False))
    np_cache_active = np_cache.enabled and degf_tokenizer is not None and use_clip_risk_prefilter
    sd_np_cache = NounPhraseRiskCache(
        enabled=sd_feedback_enabled and degf_tokenizer is not None,
        risk_threshold=risk_threshold,
        max_cache_size=int(_get_degf_option(self, model_kwargs, "degf_np_cache_size", 64)),
    )
    
    while True:
        force_correction_this_step = np_cache.consume_correction_flag() if np_cache_active else False
        force_sd_intervention_this_step = sd_np_cache.delayed_trigger if sd_np_cache.enabled else False
        token_risk_value = None
        apply_sd_intervention_this_step = False

        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # image_size: 336x336

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions, # True
            output_hidden_states=output_hidden_states
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        
        ## For complementive & contrastive decoding
        use_ritual = model_kwargs.get("use_ritual")
        use_vcd = model_kwargs.get("use_vcd")
        use_m3id = model_kwargs.get("use_m3id")
        use_diffusion = model_kwargs.get("use_diffusion")
        
        if use_ritual or use_vcd or use_m3id or use_diffusion:
            next_token_logits_pos = next_token_logits
            next_token_logits_neg = next_token_logits

            if model_kwargs["images_pos"] is not None and use_ritual:
                model_inputs_pos = self.prepare_inputs_for_generation_pos(input_ids, **model_kwargs_pos)
                outputs_pos = self(
                    **model_inputs_pos,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                next_token_logits_pos = outputs_pos.logits[:, -1, :]

            elif model_kwargs["images_neg"] is not None and use_vcd:
                model_inputs_neg = self.prepare_inputs_for_generation_neg(input_ids, **model_kwargs_neg)
                outputs_neg = self(
                    **model_inputs_neg,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :]
            elif use_m3id:
                model_inputs_neg = self.prepare_inputs_for_generation_m3id(input_ids, **model_kwargs_neg)
                outputs_neg = self(
                    **model_inputs_neg,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :]
            elif model_kwargs["images_neg"] is not None and use_diffusion:
                model_inputs_neg = self.prepare_inputs_for_generation_neg(input_ids, **model_kwargs_neg)
                outputs_neg = self(
                    **model_inputs_neg,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :]
                
                
            degf_alpha_pos = model_kwargs.get("degf_alpha_pos") if model_kwargs.get("degf_alpha_pos") is not None else 3
            degf_alpha_neg = model_kwargs.get("degf_alpha_neg") if model_kwargs.get("degf_alpha_neg") is not None else 1
            degf_beta = model_kwargs.get("degf_beta") if model_kwargs.get("degf_beta") is not None else 0.1

            # set cutoff for Adaptive Plausibility Constraints
            cutoff = torch.log(torch.tensor(degf_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            
            if use_ritual:
                diffs = (next_token_logits + degf_alpha_pos * next_token_logits_pos)
            elif use_vcd:
                diffs = (1 + degf_alpha_neg) * next_token_logits - degf_alpha_neg * next_token_logits_neg
            elif use_m3id:
                gamma_t = torch.exp(torch.tensor(-0.02*t))
                diffs = next_token_logits + (next_token_logits - next_token_logits_neg)*(1-gamma_t)/gamma_t
                t += 1
            elif use_diffusion:
                # calculate kl divergence
                # kl = nn.functional.kl_div(nn.functional.log_softmax(next_token_logits_neg, dim=-1), nn.functional.softmax(next_token_logits, dim=-1), reduction='batchmean')
                # if kl < 0.5:
                #     diffs = next_token_logits + degf_alpha_pos * next_token_logits_pos
                # else:
                #     diffs = (1 + degf_alpha_neg) * next_token_logits - degf_alpha_neg * next_token_logits_neg
                # diffs = (1 + (kl + 0.5)) * next_token_logits - (kl + 0.5) * next_token_logits_neg
                if use_clip_risk_prefilter:
                    token_risk_value = _estimate_top1_clip_grounding_risk(
                        self,
                        model_kwargs,
                        degf_tokenizer,
                        next_token_logits,
                    )
                    if token_risk_value is None:
                        token_risk_value = 0.0
                    risk_list.append(format(token_risk_value, '.4f'))
                else:
                    token_risk_value = 0.0
                    risk_list.append("disabled")

                if not use_clip_risk_prefilter:
                    risk_count += 1
                    token_count += 1
                    if sd_feedback_enabled:
                        diffs = next_token_logits
                        apply_sd_intervention_this_step = True
                    else:
                        diffs = (1 + degf_alpha_neg) * next_token_logits - degf_alpha_neg * next_token_logits_neg
                elif np_cache_active:
                    # With NP cache enabled, correction is delayed until a noun (or noun phrase end) confirms the trigger.
                    if force_correction_this_step or force_sd_intervention_this_step:
                        risk_count += 1
                        token_count += 1
                        if sd_feedback_enabled:
                            diffs = next_token_logits
                            apply_sd_intervention_this_step = True
                        else:
                            diffs = (1 + degf_alpha_neg) * next_token_logits - degf_alpha_neg * next_token_logits_neg
                    else:
                        token_count += 1
                        diffs = next_token_logits if sd_feedback_enabled else next_token_logits + degf_alpha_pos * next_token_logits_neg
                else:
                    if token_risk_value < risk_threshold and not force_sd_intervention_this_step:
                        token_count += 1
                        diffs = next_token_logits if sd_feedback_enabled else next_token_logits + degf_alpha_pos * next_token_logits_neg
                    else:
                        risk_count += 1
                        token_count += 1
                        if sd_feedback_enabled:
                            diffs = next_token_logits
                            apply_sd_intervention_this_step = True
                        else:
                            diffs = (1 + degf_alpha_neg) * next_token_logits - degf_alpha_neg * next_token_logits_neg
            
            logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            ## degf_comments: apply temperature warping and top-k filtering in contrastive decoding
            logits = logits_processor(input_ids, logits)
            # logits = logits_warper(input_ids, logits)

            next_token_scores = logits
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            # next_token_scores = logits_warper(input_ids, next_token_scores)
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        if apply_sd_intervention_this_step:
            runtime_ctx = _get_runtime_context(self)
            runtime_ctx["degf_sd_prefix_tokens"] = list(sd_np_cache.prefix_tokens)
            next_token_scores, sd_intervened = _maybe_apply_sd_logit_intervention(self, next_token_scores, model_kwargs)
        else:
            sd_intervened = False
        if sd_intervened:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        if use_diffusion and sd_np_cache.enabled and next_tokens.numel() == 1:
            token_text = _decode_single_token(degf_tokenizer, int(next_tokens[0].item()))
            token_phrase = _token_to_candidate_phrase(token_text)
            selected_token_risk = _compute_clip_grounding_risk_for_phrase(self, model_kwargs, token_phrase)
            if selected_token_risk is None:
                selected_token_risk = token_risk_value
            sd_np_cache.observe_token(token_text, selected_token_risk)

        if use_diffusion and np_cache_active and next_tokens.numel() == 1:
            token_text = _decode_single_token(degf_tokenizer, int(next_tokens[0].item()))
            token_phrase = _token_to_candidate_phrase(token_text)
            cached_token_risk = _compute_clip_grounding_risk_for_phrase(self, model_kwargs, token_phrase)
            if cached_token_risk is None:
                cached_token_risk = token_risk_value
            _, phrase, triggered = np_cache.observe_token(token_text, cached_token_risk)
            if triggered and bool(_get_degf_option(self, model_kwargs, "degf_debug_np_cache", False)):
                print(f"[DeGF NP cache] trigger next-step correction for phrase: '{phrase}'")

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        if use_ritual:
            model_kwargs_pos = self._update_model_kwargs_for_generation(
                outputs_pos, model_kwargs_pos, is_encoder_decoder=self.config.is_encoder_decoder
            )
        if use_vcd or use_m3id or use_diffusion:
            model_kwargs_neg = self._update_model_kwargs_for_generation(
                outputs_neg, model_kwargs_neg, is_encoder_decoder=self.config.is_encoder_decoder
            )
            
        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    print("risk_count: ", risk_count, "| token_count: ", token_count)
    
    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids, risk_list

def evolve_degf_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample
    transformers.generation.GenerationMixin.greedy_search = greedy_search
