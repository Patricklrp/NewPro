import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from nltk.tokenize import TreebankWordTokenizer
from transformers import AutoTokenizer


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "experiments"))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import Conversation, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


@dataclass
class GroupStats:
    name: str
    count: int
    mean: float
    var: float


def parse_args():
    parser = argparse.ArgumentParser(description="Entropy analysis for hallucinated vs non-hallucinated tokens on LLaVA outputs")
    parser.add_argument("--model_path", type=str, default="/home/ciram25-liurp/models/llava-v1.5-7b")
    parser.add_argument("--image_root", type=str, default="/home/ciram25-liurp/dataset/coco/val2014")
    parser.add_argument(
        "--chair_result_file",
        type=str,
        default="chair_results/llava-v1.5-7b/exp_002_result.jsonl",
        help="Path to CHAIR result JSON containing per-sentence hallucination annotations",
    )
    parser.add_argument("--question", type=str, default="Please describe this image in detail.")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all samples")
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="chair_results/llava-v1.5-7b/exp_002_entropy",
        help="Prefix for output files (.csv/.png/.json)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_chair_sentences(chair_result_file: str) -> List[Dict]:
    with open(chair_result_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["sentences"]


def build_prompt_input_ids(tokenizer, question: str) -> torch.Tensor:
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
    return input_ids


def image_path_from_id(image_root: str, image_id: int) -> str:
    return os.path.join(image_root, f"COCO_val2014_{image_id:012d}.jpg")


def get_word_spans(text_lower: str) -> List[Tuple[int, int]]:
    tokenizer = TreebankWordTokenizer()
    return list(tokenizer.span_tokenize(text_lower))


def build_hallucinated_char_mask(
    caption: str,
    hallucination_idxs: Sequence[int],
    hallucinated_words: Sequence[Sequence[str]],
) -> np.ndarray:
    text_lower = caption.lower()
    char_mask = np.zeros(len(text_lower), dtype=bool)
    word_spans = get_word_spans(text_lower)

    if len(hallucination_idxs) == 0:
        return char_mask

    for hidx, hallu in zip(hallucination_idxs, hallucinated_words):
        hallu_word = hallu[0] if isinstance(hallu, (list, tuple)) and len(hallu) > 0 else str(hallu)
        word_len = max(1, len(str(hallu_word).split()))
        for offset in range(word_len):
            widx = hidx + offset
            if 0 <= widx < len(word_spans):
                s, e = word_spans[widx]
                char_mask[s:e] = True
    return char_mask


def token_entropies_from_teacher_forcing(
    model,
    tokenizer_fast,
    image_tensor: torch.Tensor,
    prompt_ids: torch.Tensor,
    caption: str,
    device: torch.device,
) -> Tuple[np.ndarray, List[Tuple[int, int]], str]:
    enc = tokenizer_fast(
        caption,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    caption_ids = enc["input_ids"].to(device)
    offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"][0].tolist()]

    if caption_ids.shape[1] == 0:
        return np.array([], dtype=np.float64), [], ""

    prompt_ids = prompt_ids.to(device)
    all_ids = torch.cat([prompt_ids, caption_ids], dim=1)
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = model(
            input_ids=all_ids,
            images=image_tensor.unsqueeze(0).half().to(device),
            use_cache=False,
            return_dict=True,
        )

    logits = outputs.logits[0]
    entropies: List[float] = []
    for j in range(caption_ids.shape[1]):
        step_pos = prompt_len + j - 1
        if step_pos < 0:
            continue
        step_logits = logits[step_pos].float()
        logp = torch.log_softmax(step_logits, dim=-1)
        p = torch.exp(logp)
        h = (-(p * logp).sum() / math.log(2.0)).item()
        entropies.append(h)

    return np.array(entropies, dtype=np.float64), offsets, caption


def token_is_hallucinated(offset: Tuple[int, int], hallu_mask: np.ndarray) -> bool:
    s, e = offset
    if e <= s:
        return False
    s = max(0, min(s, len(hallu_mask)))
    e = max(0, min(e, len(hallu_mask)))
    if e <= s:
        return False
    return bool(hallu_mask[s:e].any())


def summarize(values: List[float], name: str) -> GroupStats:
    if len(values) == 0:
        return GroupStats(name=name, count=0, mean=float("nan"), var=float("nan"))
    arr = np.array(values, dtype=np.float64)
    return GroupStats(name=name, count=int(arr.size), mean=float(arr.mean()), var=float(arr.var(ddof=0)))


def save_table(stats: List[GroupStats], csv_path: str):
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("group,count,mean_entropy_bits,var_entropy_bits2\n")
        for s in stats:
            f.write(f"{s.name},{s.count},{s.mean:.8f},{s.var:.8f}\n")


def plot_gaussian_compare(hallu: GroupStats, non_hallu: GroupStats, png_path: str):
    plt.figure(figsize=(8, 5))

    valid_stats = [s for s in [hallu, non_hallu] if np.isfinite(s.mean) and np.isfinite(s.var)]
    if not valid_stats:
        raise RuntimeError("No valid statistics to plot")

    x_left = min(s.mean - 4.0 * math.sqrt(max(s.var, 1e-8)) for s in valid_stats)
    x_right = max(s.mean + 4.0 * math.sqrt(max(s.var, 1e-8)) for s in valid_stats)
    xs = np.linspace(x_left, x_right, 600)

    def normal_pdf(x, mu, var):
        var = max(var, 1e-8)
        return 1.0 / np.sqrt(2 * np.pi * var) * np.exp(-((x - mu) ** 2) / (2 * var))

    if np.isfinite(hallu.mean) and np.isfinite(hallu.var):
        plt.plot(xs, normal_pdf(xs, hallu.mean, hallu.var), label="Hallucinated token entropy", linewidth=2.0)
    if np.isfinite(non_hallu.mean) and np.isfinite(non_hallu.var):
        plt.plot(xs, normal_pdf(xs, non_hallu.mean, non_hallu.var), label="Non-hallucinated token entropy", linewidth=2.0)

    plt.xlabel("Shannon entropy (bits)")
    plt.ylabel("Normal PDF")
    plt.title("Gaussian Comparison of Token Entropy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer_model, model, image_processor, _ = load_pretrained_model(args.model_path, None, model_name)
    tokenizer_fast = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_device = next(model.parameters()).device
    if model_device.type == "cpu" and requested_device.type == "cuda":
        model = model.to(requested_device)
        model_device = requested_device
    model.eval()

    prompt_ids = build_prompt_input_ids(tokenizer_model, args.question)
    sentences = load_chair_sentences(args.chair_result_file)
    if args.max_samples > 0:
        sentences = sentences[: args.max_samples]

    hallu_entropies: List[float] = []
    non_hallu_entropies: List[float] = []
    sample_debug: List[Dict] = []

    for i, item in enumerate(sentences):
        image_id = int(item["image_id"])
        caption = item["caption"]
        hallu_idxs = item.get("hallucination_idxs", [])
        hallu_words = item.get("mscoco_hallucinated_words", [])

        img_path = image_path_from_id(args.image_root, image_id)
        if not os.path.exists(img_path):
            print(f"[Warning] image missing for image_id={image_id}: {img_path}")
            continue

        raw_image = Image.open(img_path).convert("RGB")
        image_tensor = image_processor.preprocess(raw_image, return_tensors="pt")["pixel_values"][0]

        entropies, offsets, used_caption = token_entropies_from_teacher_forcing(
            model=model,
            tokenizer_fast=tokenizer_fast,
            image_tensor=image_tensor,
            prompt_ids=prompt_ids,
            caption=caption,
            device=model_device,
        )

        hallu_mask = build_hallucinated_char_mask(used_caption, hallu_idxs, hallu_words)
        hallu_count = 0
        non_hallu_count = 0

        for ent, off in zip(entropies.tolist(), offsets):
            if token_is_hallucinated(off, hallu_mask):
                hallu_entropies.append(ent)
                hallu_count += 1
            else:
                non_hallu_entropies.append(ent)
                non_hallu_count += 1

        sample_debug.append(
            {
                "image_id": image_id,
                "n_tokens": len(offsets),
                "n_hallucinated_tokens": hallu_count,
                "n_non_hallucinated_tokens": non_hallu_count,
            }
        )
        print(
            f"[{i + 1}/{len(sentences)}] image_id={image_id} "
            f"tokens={len(offsets)} hallu={hallu_count} non_hallu={non_hallu_count}"
        )

    hallu_stats = summarize(hallu_entropies, "hallucinated")
    non_hallu_stats = summarize(non_hallu_entropies, "non_hallucinated")
    stats = [hallu_stats, non_hallu_stats]

    csv_path = args.output_prefix + "_table.csv"
    png_path = args.output_prefix + "_gaussian.png"
    json_path = args.output_prefix + "_debug.json"

    save_table(stats, csv_path)
    plot_gaussian_compare(hallu_stats, non_hallu_stats, png_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "stats": [s.__dict__ for s in stats],
                "samples": sample_debug,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n===== Entropy Summary =====")
    print("group,count,mean_entropy_bits,var_entropy_bits2")
    for s in stats:
        print(f"{s.name},{s.count},{s.mean:.8f},{s.var:.8f}")
    print(f"table_csv={csv_path}")
    print(f"gaussian_png={png_path}")
    print(f"debug_json={json_path}")


if __name__ == "__main__":
    main()
