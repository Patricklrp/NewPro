import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


DEFAULT_SD15_PATH = "/home/ciram25-liurp/models/stable-diffusion-v1-5"


def load_sd_pipeline_for_feedback(
    model_path: str = DEFAULT_SD15_PATH,
    device: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16,
) -> StableDiffusionPipeline:
    """Load SD pipeline for latent feedback scoring.

    The pipeline is used as a container of VAE/tokenizer/text_encoder/UNet/scheduler,
    and does not perform pixel-space generation in this module.
    """
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.unet.eval()
    return pipe


def _infer_device_and_dtype(pipe: StableDiffusionPipeline) -> Tuple[torch.device, torch.dtype]:
    device = next(pipe.unet.parameters()).device
    dtype = next(pipe.unet.parameters()).dtype
    return device, dtype


def _normalize_image_size(width: int, height: int) -> Tuple[int, int]:
    # SD VAE requires spatial size divisible by 8.
    norm_w = max(8, int(math.floor(width / 8.0) * 8))
    norm_h = max(8, int(math.floor(height / 8.0) * 8))
    return norm_w, norm_h


def _pil_to_vae_input(image: Image.Image, width: int, height: int) -> torch.Tensor:
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    arr = np.array(image).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None]  # [1, 3, H, W]
    tensor = torch.from_numpy(arr)
    tensor = tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
    return tensor


def _encode_prompts(pipe: StableDiffusionPipeline, prompts: Sequence[str], device: torch.device) -> torch.Tensor:
    text_inputs = pipe.tokenizer(
        list(prompts),
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = None
    if "attention_mask" in text_inputs:
        attention_mask = text_inputs.attention_mask.to(device)

    with torch.no_grad():
        if attention_mask is None:
            text_embeds = pipe.text_encoder(input_ids)[0]
        else:
            text_embeds = pipe.text_encoder(input_ids, attention_mask=attention_mask)[0]
    return text_embeds


def prepare_sd_feedback_context(
    pipe: StableDiffusionPipeline,
    image: Image.Image,
    noise_timestep: int = 500,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Prepare reusable latent context for fast repeated scoring.

    Returns dict with zt / epsilon / timestep and metadata. This allows calling
    UNet many times with different text prompts while reusing the same visual base.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    device, dtype = _infer_device_and_dtype(pipe)
    width, height = _normalize_image_size(image.width, image.height)
    image_tensor = _pil_to_vae_input(image, width, height).to(device=device, dtype=dtype)

    with torch.no_grad():
        posterior = pipe.vae.encode(image_tensor).latent_dist
        z0 = posterior.sample()
        z0 = z0 * pipe.vae.config.scaling_factor

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    epsilon = torch.randn(z0.shape, generator=generator, device=device, dtype=dtype)
    timestep = torch.tensor([int(noise_timestep)], device=device, dtype=torch.long)
    zt = pipe.scheduler.add_noise(z0, epsilon, timestep)

    return {
        "zt": zt,
        "epsilon": epsilon,
        "timestep": timestep,
        "width": int(width),
        "height": int(height),
        "noise_timestep": int(noise_timestep),
    }


def score_prompts_with_sd_context(
    pipe: StableDiffusionPipeline,
    context: Dict[str, Any],
    candidate_prompts: Sequence[str],
    unet_batch_size: int = 0,
) -> List[Dict[str, Any]]:
    """Score prompts using precomputed context from prepare_sd_feedback_context."""
    if len(candidate_prompts) == 0:
        return []

    zt = context["zt"]
    epsilon = context["epsilon"]
    timestep = context["timestep"]
    device = zt.device

    k = len(candidate_prompts)
    if unet_batch_size <= 0:
        unet_batch_size = k

    results: List[Dict[str, Any]] = []
    for start in range(0, k, unet_batch_size):
        end = min(start + unet_batch_size, k)
        prompt_chunk = candidate_prompts[start:end]

        prompt_embeds = _encode_prompts(pipe, prompt_chunk, device=device)

        bsz = end - start
        zt_batch = zt.repeat(bsz, 1, 1, 1)
        noise_batch = epsilon.repeat(bsz, 1, 1, 1)
        timestep_batch = timestep.repeat(bsz)

        with torch.no_grad():
            pred_noise = pipe.unet(
                zt_batch,
                timestep_batch,
                encoder_hidden_states=prompt_embeds,
                return_dict=True,
            ).sample

        mse = ((pred_noise - noise_batch) ** 2).flatten(1).mean(dim=1)
        score = -mse

        for p, m, s in zip(prompt_chunk, mse.tolist(), score.tolist()):
            results.append(
                {
                    "prompt": p,
                    "mse": float(m),
                    "score": float(s),
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def score_prompts_via_single_step_diffusion_loss(
    pipe: StableDiffusionPipeline,
    image: Image.Image,
    candidate_prompts: Sequence[str],
    noise_timestep: int = 500,
    seed: Optional[int] = None,
    unet_batch_size: int = 0,
) -> List[Dict[str, Any]]:
    """Stage-3/4 path-A implementation.

    Stage 3:
    1) Encode original image to latent z0 via VAE.
    2) Add known Gaussian noise epsilon at a preset timestep t to get zt.
    3) Pair K replicated zt with K candidate prompts.
    4) Run a one-step batched UNet forward pass.

    Stage 4 Path A:
    - Compute per-prompt MSE between predicted noise and known epsilon.
      Smaller MSE => better image-text alignment.
    - Return score = -MSE (larger is better).
    """
    context = prepare_sd_feedback_context(
        pipe=pipe,
        image=image,
        noise_timestep=noise_timestep,
        seed=seed,
    )
    return score_prompts_with_sd_context(
        pipe=pipe,
        context=context,
        candidate_prompts=candidate_prompts,
        unet_batch_size=unet_batch_size,
    )


def build_candidate_prompts_from_phrase(
    noun_phrase: str,
    templates: Optional[Sequence[str]] = None,
) -> List[str]:
    """Utility for building K candidate prompts from one phrase.

    This is optional but convenient for Stage-3 input construction.
    """
    if templates is None:
        templates = [
            "a photo of {x}",
            "an image of {x}",
            "a realistic picture of {x}",
            "a detailed photo of {x}",
        ]
    x = noun_phrase.strip()
    return [t.format(x=x) for t in templates if "{x}" in t]
