from typing import Any, Dict

import torch
from transformers import AutoProcessor, CLIPModel


DEFAULT_CLIP_MODEL_PATH = "/home/ciram25-liurp/models/clip-vit-base-patch32"


def load_clip_risk_bundle(model_path: str = DEFAULT_CLIP_MODEL_PATH, device: str = "cuda") -> Dict[str, Any]:
    """Load CLIP model/processor once and keep them in a reusable bundle."""
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_path)
    clip_model = CLIPModel.from_pretrained(model_path).to(torch_device)
    clip_model.eval()
    return {
        "processor": processor,
        "clip_model": clip_model,
        "device": torch_device,
        "model_path": model_path,
        "text_feature_cache": {},
    }


def prepare_clip_image_feature(bundle: Dict[str, Any], image: Any) -> torch.Tensor:
    """Encode image to normalized CLIP feature."""
    processor = bundle["processor"]
    clip_model = bundle["clip_model"]
    device = bundle["device"]

    with torch.no_grad():
        image_inputs = processor(images=image, return_tensors="pt").to(device)
        image_feature = clip_model.get_image_features(**image_inputs)
    image_feature = image_feature / (image_feature.norm(dim=-1, keepdim=True) + 1e-12)
    return image_feature[0].detach().float().cpu()


def score_text_grounding_risk(
    bundle: Dict[str, Any],
    image_feature: torch.Tensor,
    phrase: str,
    prompt_template: str = "a photo of {}",
) -> float:
    """Compute grounding risk = 1 - cosine_similarity(image, text)."""
    normalized = phrase.strip().lower()
    if not normalized:
        raise ValueError("phrase must be non-empty")

    prompt = prompt_template.format(normalized)
    cache = bundle.setdefault("text_feature_cache", {})

    if prompt not in cache:
        processor = bundle["processor"]
        clip_model = bundle["clip_model"]
        device = bundle["device"]
        with torch.no_grad():
            text_inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
            text_feature = clip_model.get_text_features(**text_inputs)
        text_feature = text_feature / (text_feature.norm(dim=-1, keepdim=True) + 1e-12)
        cache[prompt] = text_feature[0].detach().float().cpu()

    text_feature = cache[prompt]
    similarity = float(torch.dot(image_feature, text_feature).item())
    return 1.0 - similarity
