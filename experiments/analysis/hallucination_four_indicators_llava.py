import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from transformers import AutoProcessor, CLIPModel


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 图文匹配度风险 as an initial hallucination screening indicator")
    parser.add_argument("--clip_model_path", type=str, default="/home/ciram25-liurp/models/clip-vit-base-patch32")
    parser.add_argument("--image_root", type=str, default="/home/ciram25-liurp/dataset/coco/val2014")
    parser.add_argument("--chair_result_file", type=str, default="chair_results/llava-v1.5-7b/exp_004_result.jsonl")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--clip_device", type=str, default="cuda")
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="chair_results/llava-v1.5-7b/exp_004_grounding_risk",
    )
    parser.add_argument(
        "--target_recall",
        type=float,
        default=0.99,
        help="Target recall for high-recall operating point (e.g., 0.99)",
    )
    parser.add_argument(
        "--target_recalls",
        type=str,
        default="",
        help="Comma-separated recall targets for curve output, e.g. 0.90,0.95,0.98,0.99",
    )
    return parser.parse_args()


def parse_target_recall_list(target_recalls: str, fallback_target: float) -> List[float]:
    if target_recalls and target_recalls.strip():
        vals = []
        for x in target_recalls.split(","):
            x = x.strip()
            if not x:
                continue
            v = float(x)
            if not (0.0 < v <= 1.0):
                raise ValueError(f"Invalid target recall value: {v}")
            vals.append(v)
        if len(vals) == 0:
            return [fallback_target]
        return sorted(set(vals))
    return [fallback_target]


def image_path_from_id(image_root: str, image_id: int) -> str:
    return os.path.join(image_root, f"COCO_val2014_{image_id:012d}.jpg")


def load_chair_sentences(chair_result_file: str) -> List[Dict]:
    with open(chair_result_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["sentences"]


def build_hallucinated_char_mask(
    caption: str,
    hallucination_idxs: Sequence[int],
    hallucinated_words: Sequence[Sequence[str]],
) -> np.ndarray:
    text_lower = caption.lower()
    char_mask = np.zeros(len(text_lower), dtype=bool)
    word_spans = list(TreebankWordTokenizer().span_tokenize(text_lower))
    if len(hallucination_idxs) == 0:
        return char_mask

    for hidx, hallu in zip(hallucination_idxs, hallucinated_words):
        hallu_word = hallu[0] if isinstance(hallu, (list, tuple)) and len(hallu) > 0 else str(hallu)
        n_words = max(1, len(str(hallu_word).split()))
        for off in range(n_words):
            widx = hidx + off
            if 0 <= widx < len(word_spans):
                s, e = word_spans[widx]
                char_mask[s:e] = True
    return char_mask


def safe_alpha_token(token: str) -> bool:
    if token is None:
        return False
    t = token.strip()
    if not t:
        return False
    return any(c.isalpha() for c in t)


def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


def compute_auc_roc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int32)
    pos = labels.sum()
    neg = labels.shape[0] - pos
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, labels.shape[0] + 1)
    sum_pos_ranks = ranks[labels == 1].sum()
    auc = (sum_pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def compute_auc_pr(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int32)
    pos = labels.sum()
    if pos == 0:
        return float("nan")

    order = np.argsort(-scores)
    y = labels[order]

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    recall = tp / max(pos, 1)
    precision = tp / np.maximum(tp + fp, 1)

    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return float(np.trapz(precision, recall))


def best_f1(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float, float, float]:
    labels = labels.astype(np.int32)
    order = np.argsort(-scores)
    y = labels[order]
    s = scores[order]

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    fn = y.sum() - tp

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    idx = int(np.argmax(f1))
    return float(f1[idx]), float(s[idx]), float(precision[idx]), float(recall[idx])


def best_precision_at_target_recall(
    labels: np.ndarray,
    scores: np.ndarray,
    target_recall: float,
) -> Tuple[float, float, float, float, float]:
    labels = labels.astype(np.int32)
    order = np.argsort(-scores)
    y = labels[order]
    s = scores[order]

    pos = int(y.sum())
    if pos == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    recall = tp / max(pos, 1)
    precision = tp / np.maximum(tp + fp, 1)

    mask = recall >= target_recall
    if not np.any(mask):
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    candidate_idx = np.where(mask)[0]
    candidate_precisions = precision[candidate_idx]
    best_local = int(np.argmax(candidate_precisions))
    idx = int(candidate_idx[best_local])
    pred_pos = float(tp[idx] + fp[idx])
    total = float(labels.shape[0])
    pred_non_hallu_ratio = 1.0 - (pred_pos / max(total, 1.0))
    return float(precision[idx]), float(recall[idx]), float(s[idx]), pred_non_hallu_ratio, pred_pos / max(total, 1.0)


def plot_non_hallu_ratio_vs_recall_targets(curve_rows: List[Dict], target_values: List[float], png_path: str):
    plt.figure(figsize=(7.5, 5))
    rows = sorted(curve_rows, key=lambda r: r["target_recall"])
    xs = [r["target_recall"] for r in rows]
    ys = [r["non_hallucinated_ratio_at_target_recall"] for r in rows]
    plt.plot(xs, ys, marker="o", linewidth=2, label="图文匹配度风险")

    plt.xticks(target_values)
    plt.xlabel("Target Recall")
    plt.ylabel("Predicted Non-Hallucinated Ratio")
    plt.title("Non-Hallucinated Ratio Under Different Recall Targets")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    target_values = parse_target_recall_list(args.target_recalls, args.target_recall)

    sentences = load_chair_sentences(args.chair_result_file)
    if args.max_samples > 0:
        sentences = sentences[: args.max_samples]

    clip_device = torch.device(args.clip_device if torch.cuda.is_available() else "cpu")
    clip_processor = AutoProcessor.from_pretrained(args.clip_model_path)
    clip_model = CLIPModel.from_pretrained(args.clip_model_path).to(clip_device)
    clip_model.eval()

    lemmatizer = WordNetLemmatizer()
    word_tokenizer = TreebankWordTokenizer()

    text_feat_cache: Dict[str, torch.Tensor] = {}

    def get_text_feature(text: str) -> torch.Tensor:
        key = text.strip().lower()
        if key not in text_feat_cache:
            txt_in = clip_processor(text=[f"a photo of {key}"], return_tensors="pt", padding=True).to(clip_device)
            with torch.no_grad():
                tfeat = clip_model.get_text_features(**txt_in)
            tfeat = tfeat / (tfeat.norm(dim=-1, keepdim=True) + 1e-12)
            text_feat_cache[key] = tfeat[0].detach().cpu()
        return text_feat_cache[key]

    labels: List[int] = []
    grounding_risk_scores: List[float] = []

    for i, item in enumerate(sentences):
        image_id = int(item["image_id"])
        caption = item["caption"]
        hallu_idxs = item.get("hallucination_idxs", [])
        hallu_words = item.get("mscoco_hallucinated_words", [])

        img_path = image_path_from_id(args.image_root, image_id)
        if not os.path.exists(img_path):
            print(f"[Warning] image missing: {img_path}")
            continue

        raw_image = Image.open(img_path).convert("RGB")
        hallu_char_mask = build_hallucinated_char_mask(caption, hallu_idxs, hallu_words)
        word_spans = list(word_tokenizer.span_tokenize(caption.lower()))

        sentence_object_set = set()
        sentence_object_set.update([w.lower() for w in item.get("mscoco_gt_words", [])])
        sentence_object_set.update([w.lower() for w in item.get("mscoco_generated_words", [])])

        clip_img_in = clip_processor(images=raw_image, return_tensors="pt").to(clip_device)
        with torch.no_grad():
            img_feat = clip_model.get_image_features(**clip_img_in)
        img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-12)
        img_feat_cpu = img_feat[0].detach().cpu()

        for ws, we in word_spans:
            word = caption[ws:we].strip()
            if not safe_alpha_token(word):
                continue

            lemma = lemmatizer.lemmatize(word.lower())
            is_hallu = int(hallu_char_mask[ws:we].any())
            # Keep object-like scope while retaining all positive labels.
            if not (lemma in sentence_object_set or is_hallu == 1):
                continue

            txt_feat = get_text_feature(word)
            sim = float(torch.dot(img_feat_cpu, txt_feat).item())

            labels.append(is_hallu)
            grounding_risk_scores.append(1.0 - sim)

        if (i + 1) % 25 == 0 or (i + 1) == len(sentences):
            print(f"[{i + 1}/{len(sentences)}] processed, collected_words={len(labels)}")

    labels_arr = np.array(labels, dtype=np.int32)
    scores = np.array(grounding_risk_scores, dtype=np.float64)
    metric_name = "图文匹配度风险"

    auc_roc = compute_auc_roc(labels_arr, scores)
    auc_pr = compute_auc_pr(labels_arr, scores)
    f1, thr, precision, recall = best_f1(labels_arr, scores)
    p_hr, r_hr, thr_hr, nh_hr, ph_hr = best_precision_at_target_recall(labels_arr, scores, args.target_recall)

    result_rows = [
        {
            "metric": metric_name,
            "auroc": auc_roc,
            "auprc": auc_pr,
            "best_f1": f1,
            "best_threshold": thr,
            "best_precision": precision,
            "best_recall": recall,
            "target_recall": args.target_recall,
            "precision_at_target_recall": p_hr,
            "achieved_recall_at_target": r_hr,
            "threshold_at_target_recall": thr_hr,
            "non_hallucinated_ratio_at_target_recall": nh_hr,
            "predicted_hallucinated_ratio_at_target_recall": ph_hr,
        }
    ]

    curve_rows = []
    for target in target_values:
        p_t, r_t, thr_t, nh_t, ph_t = best_precision_at_target_recall(labels_arr, scores, target)
        curve_rows.append(
            {
                "metric": metric_name,
                "target_recall": target,
                "precision_at_target_recall": p_t,
                "achieved_recall_at_target": r_t,
                "threshold_at_target_recall": thr_t,
                "non_hallucinated_ratio_at_target_recall": nh_t,
                "predicted_hallucinated_ratio_at_target_recall": ph_t,
            }
        )

    csv_path = args.output_prefix + "_effectiveness.csv"
    json_path = args.output_prefix + "_effectiveness.json"
    curve_csv_path = args.output_prefix + "_non_hallucinated_ratio_targets.csv"
    curve_png_path = args.output_prefix + "_non_hallucinated_ratio_targets.png"

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "auroc",
                "auprc",
                "best_f1",
                "best_threshold",
                "best_precision",
                "best_recall",
                "target_recall",
                "precision_at_target_recall",
                "achieved_recall_at_target",
                "threshold_at_target_recall",
                "non_hallucinated_ratio_at_target_recall",
                "predicted_hallucinated_ratio_at_target_recall",
            ],
        )
        w.writeheader()
        for row in result_rows:
            w.writerow(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_words": int(labels_arr.shape[0]),
                "n_hallucinated": int(labels_arr.sum()),
                "n_non_hallucinated": int(labels_arr.shape[0] - labels_arr.sum()),
                "results": result_rows,
                "curve_results": curve_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(curve_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "target_recall",
                "precision_at_target_recall",
                "achieved_recall_at_target",
                "threshold_at_target_recall",
                "non_hallucinated_ratio_at_target_recall",
                "predicted_hallucinated_ratio_at_target_recall",
            ],
        )
        w.writeheader()
        for row in sorted(curve_rows, key=lambda x: (x["metric"], x["target_recall"])):
            w.writerow(row)

    plot_non_hallu_ratio_vs_recall_targets(curve_rows, target_values, curve_png_path)

    print("\n===== 图文匹配度风险 Effectiveness =====")
    print(f"n_words={labels_arr.shape[0]}, n_hallucinated={labels_arr.sum()}, n_non_hallucinated={labels_arr.shape[0]-labels_arr.sum()}")
    for row in result_rows:
        print(
            f"{row['metric']}: "
            f"AUROC={row['auroc']:.6f}, "
            f"AUPRC={row['auprc']:.6f}, "
            f"BestF1={row['best_f1']:.6f}, "
            f"thr={row['best_threshold']:.6f}, "
            f"P={row['best_precision']:.6f}, "
            f"R={row['best_recall']:.6f}, "
            f"P@R>={row['target_recall']:.2f}={row['precision_at_target_recall']:.6f}, "
            f"R@that={row['achieved_recall_at_target']:.6f}, "
            f"thr@that={row['threshold_at_target_recall']:.6f}, "
            f"non_hallu_ratio@that={row['non_hallucinated_ratio_at_target_recall']:.6f}"
        )
    print(f"csv={csv_path}")
    print(f"json={json_path}")
    print(f"curve_csv={curve_csv_path}")
    print(f"curve_png={curve_png_path}")


if __name__ == "__main__":
    main()
