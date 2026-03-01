import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import dist_util
from utils.logger import create_logger
from pope_loader import POPEDataSet
from torchvision import transforms as T


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="POPE evaluation on Qwen-VL.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default="qwen-vl")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--pope_path", type=str, required=True)
    parser.add_argument("--log_path", type=str, default="./logs")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--use_ritual", type=str2bool, default=False)
    parser.add_argument("--use_vcd", type=str2bool, default=False)
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    parser.add_argument("--use_diffusion", type=str2bool, default=False)

    parser.add_argument("--degf_alpha_pos", type=float, default=3)
    parser.add_argument("--degf_alpha_neg", type=float, default=1)
    parser.add_argument("--degf_beta", type=float, default=0.1)

    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--experiment_index", type=int, default=0)

    args = parser.parse_args()
    return args


def print_acc(pred_list, label_list, logger):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    if TP + FP == 0:
        precision = 0
    else:
        precision = float(TP) / float(TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = float(TP) / float(TP + FN)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)

    logger.info('TP\tFP\tTN\tFN\t')
    logger.info(f"{TP}\t{FP}\t{TN}\t{FN}")

    return acc, precision, recall, f1, yes_ratio


def recorder(out, pred_list):
    neg_words = ["No", "not", "no", "NO"]
    for line in out.split('\n'):
        line = line.replace('.', '').replace(',', '')
        words = line.split(' ')
        if any(word in neg_words for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
        break
    return pred_list


def main():
    args = parse_args()
    dist_util.setup_dist(args)

    if dist.get_rank() == 0:
        os.makedirs(args.log_path, exist_ok=True)
        model_string_name = args.model_path.split("/")[-1]
        experiment_dir = f"{args.log_path}/{model_string_name}/{args.degf_alpha_pos}_{args.degf_alpha_neg}_{args.degf_beta}/{args.experiment_index}"
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    if args.use_ritual or args.use_vcd or args.use_m3id or args.use_diffusion:
        logger.warning("Qwen-VL evaluation currently runs baseline generation only; DeGF options are ignored.")

    device = dist_util.device()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if use_cuda else None,
    )
    torch.set_default_dtype(default_dtype)
    if not use_cuda:
        model.to(device)
    if not hasattr(model.generation_config, "chat_format"):
        model.generation_config.chat_format = "chatml"
    if not hasattr(model.generation_config, "max_window_size"):
        model.generation_config.max_window_size = getattr(model.config, "max_position_embeddings", 2048)
    model.eval()

    trans = T.ToTensor()
    pope_dataset = POPEDataSet(args.pope_path, args.data_path, trans, "qwen-vl")
    pope_loader = DataLoader(
        pope_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info("Start eval...")
    pred_list, label_list = [], []

    for _, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        qs = data["query"][0]
        label = data["label"]
        image_path = data["image_path"][0]

        label_list = label_list + list(label)

        query = tokenizer.from_list_format([
            {"image": image_path},
            {"text": qs},
        ])

        with torch.inference_mode():
            response, _ = model.chat(
                tokenizer,
                query,
                history=None,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        pred_list = recorder(response, pred_list)

        logger.info("[VQA]")
        logger.info(f"V: {image_path}")
        logger.info(f"Q: {qs}")
        logger.info(f"A: {response}")
        if label == 1:
            logger.info("GT: Yes")
        elif label == 0:
            logger.info("GT: No")

        acc, precision, recall, f1, yes_ratio = print_acc(pred_list, label_list, logger)
        logger.info(
            f"acc: {acc*100:.2f}, precision: {precision*100:.2f}, recall: {recall*100:.2f}, f1: {f1*100:.2f}, yes_ratio: {yes_ratio*100:.2f}"
        )
        logger.info("=" * 50)

        np.save(f"{experiment_dir}/pred_list.npy", pred_list)
        np.save(f"{experiment_dir}/label_list.npy", label_list)

    if len(pred_list) != 0:
        logger.info(vars(args))
        acc, precision, recall, f1, yes_ratio = print_acc(pred_list, label_list, logger)
        logger.info(
            f"acc: {acc*100:.2f}, precision: {precision*100:.2f}, recall: {recall*100:.2f}, f1: {f1*100:.2f}, yes_ratio: {yes_ratio*100:.2f}"
        )
        np.save(f"{experiment_dir}/pred_list.npy", pred_list)


if __name__ == "__main__":
    main()
