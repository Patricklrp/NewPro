import argparse
import json
import os
import random
import re
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "eval_bench"))
sys.path.append(os.path.join(ROOT, "experiments"))

from chair_loader import CHAIRDataset
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import Conversation, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def parse_args():
    parser = argparse.ArgumentParser(description="Profile grounding_low overhead on CHAIR samples")
    parser.add_argument("--model_path", type=str, default="/home/ciram25-liurp/models/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default="llava")
    parser.add_argument("--clip_model_path", type=str, default="/home/ciram25-liurp/models/clip-vit-base-patch32")
    parser.add_argument("--data_path", type=str, default="/home/ciram25-liurp/dataset/coco/val2014")
    parser.add_argument("--anno_path", type=str, default="/home/ciram25-liurp/dataset/coco/annotations/instances_val2014.json")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--output_json", type=str, default="chair_results/llava-v1.5-7b/exp_004_grounding_low_overhead_100.json")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt_input_ids(tokenizer, question: str):
    conv_out = Conversation(
        system=(
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions."
        ),
        roles=("USER", "ASSISTANT"),
        version="v1",
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.TWO,
        sep=" ",
        sep2="</s>",
    )
    qu_out = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv_out.append_message(conv_out.roles[0], qu_out)
    conv_out.append_message(conv_out.roles[1], None)
    prompt_out = conv_out.get_prompt()
    input_ids = tokenizer_image_token(prompt_out, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    stop_str = conv_out.sep if conv_out.sep_style != SeparatorStyle.TWO else conv_out.sep2
    return input_ids, stop_str


def extract_candidate_words(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", text.lower())
    # Deduplicate in-order to avoid repeated text encoding work for same sample.
    seen = set()
    uniq = []
    for w in words:
        if w not in seen:
            uniq.append(w)
            seen.add(w)
    return uniq


def compute_grounding_low_scores(
    raw_image,
    caption: str,
    clip_processor,
    clip_model,
    text_feat_cache: Dict[str, torch.Tensor],
    clip_device: torch.device,
) -> Dict[str, float]:
    image_inputs = clip_processor(images=raw_image, return_tensors="pt").to(clip_device)
    with torch.no_grad():
        image_feat = clip_model.get_image_features(**image_inputs)
    image_feat = image_feat / (image_feat.norm(dim=-1, keepdim=True) + 1e-12)
    image_feat = image_feat[0]

    scores = {}
    for word in extract_candidate_words(caption):
        if word not in text_feat_cache:
            txt_inputs = clip_processor(text=[f"a photo of {word}"], return_tensors="pt", padding=True).to(clip_device)
            with torch.no_grad():
                txt_feat = clip_model.get_text_features(**txt_inputs)
            txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-12)
            text_feat_cache[word] = txt_feat[0]
        sim = torch.dot(image_feat, text_feat_cache[word]).item()
        scores[word] = 1.0 - sim
    return scores


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    set_seed(args.seed)

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, None, model_name)
    model.eval()
    llava_device = next(model.parameters()).device

    clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_processor = AutoProcessor.from_pretrained(args.clip_model_path)
    clip_model = CLIPModel.from_pretrained(args.clip_model_path).to(clip_device)
    clip_model.eval()

    dataset = CHAIRDataset(
        data_path=args.data_path,
        anno_path=args.anno_path,
        trans=image_processor,
        model=args.model_base,
    )

    n = min(args.num_samples, len(dataset))
    question = "Please describe this image in detail."

    text_feat_cache: Dict[str, torch.Tensor] = {}
    generation_time_total = 0.0
    grounding_time_total = 0.0
    sample_records = []

    all_start = time.perf_counter()
    for i in tqdm(range(n), total=n, desc="Profile grounding_low"):
        sample = dataset[i]
        image_id = int(sample["image_id"])
        image_path = sample["image_path"]
        image_tensor = sample["image"]

        input_ids, stop_str = build_prompt_input_ids(tokenizer, question)
        input_ids = input_ids.to(llava_device)

        gen_start = time.perf_counter()
        with torch.inference_mode():
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().to(llava_device),
                    images_pos=None,
                    images_neg=None,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=None,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    use_ritual=False,
                    use_vcd=False,
                    use_m3id=False,
                    use_diffusion=False,
                    degf_alpha_pos=3.0,
                    degf_alpha_neg=1.0,
                    degf_beta=0.1,
                )
        output_ids = generated[0] if isinstance(generated, (tuple, list)) else generated
        input_token_len = input_ids.shape[1]
        caption = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
        if caption.endswith(stop_str):
            caption = caption[:-len(stop_str)].strip()
        gen_elapsed = time.perf_counter() - gen_start
        generation_time_total += gen_elapsed

        grounding_start = time.perf_counter()
        raw_image = Image.open(image_path).convert("RGB")
        grounding_scores = compute_grounding_low_scores(
            raw_image=raw_image,
            caption=caption,
            clip_processor=clip_processor,
            clip_model=clip_model,
            text_feat_cache=text_feat_cache,
            clip_device=clip_device,
        )
        grounding_elapsed = time.perf_counter() - grounding_start
        grounding_time_total += grounding_elapsed

        sample_records.append(
            {
                "idx": i,
                "image_id": image_id,
                "gen_time_sec": gen_elapsed,
                "grounding_time_sec": grounding_elapsed,
                "n_candidate_words": len(grounding_scores),
            }
        )

    total_elapsed = time.perf_counter() - all_start

    estimated_without_grounding = max(total_elapsed - grounding_time_total, 1e-9)
    overhead_abs = grounding_time_total
    overhead_ratio_in_total = grounding_time_total / max(total_elapsed, 1e-9)
    overhead_ratio_vs_baseline = grounding_time_total / estimated_without_grounding

    summary = {
        "num_samples": n,
        "total_time_sec_with_grounding": total_elapsed,
        "generation_time_sec": generation_time_total,
        "grounding_low_time_sec": grounding_time_total,
        "estimated_time_sec_without_grounding": estimated_without_grounding,
        "grounding_overhead_sec": overhead_abs,
        "grounding_overhead_ratio_of_total": overhead_ratio_in_total,
        "grounding_overhead_ratio_vs_baseline": overhead_ratio_vs_baseline,
        "avg_time_sec_per_sample_total": total_elapsed / max(n, 1),
        "avg_time_sec_per_sample_grounding": grounding_time_total / max(n, 1),
        "avg_time_sec_per_sample_generation": generation_time_total / max(n, 1),
        "text_feature_cache_size": len(text_feat_cache),
    }

    result = {
        "summary": summary,
        "samples": sample_records,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n===== Grounding Low Overhead Profile =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"saved_json: {args.output_json}")


if __name__ == "__main__":
    main()
