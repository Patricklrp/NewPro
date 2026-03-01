import os
import sys
import json
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import dist_util
from utils.logger import create_logger
from chair_loader import CHAIRDataset
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
    parser = argparse.ArgumentParser(description="CHAIR evaluation on Qwen-VL.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default="qwen-vl")

    parser.add_argument("--data_path", type=str, required=True, help="COCO val2014 image path")
    parser.add_argument("--anno_path", type=str, required=True, help="COCO instances_val2014.json")
    parser.add_argument("--log_path", type=str, default="./logs/chair")
    parser.add_argument("--out_path", type=str, default="./chair_results/qwen-vl")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--use_ritual", type=str2bool, default=False)
    parser.add_argument("--use_vcd", type=str2bool, default=False)
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    parser.add_argument("--use_diffusion", type=str2bool, default=False)

    parser.add_argument("--degf_alpha_pos", type=float, default=3)
    parser.add_argument("--degf_alpha_neg", type=float, default=1)
    parser.add_argument("--degf_beta", type=float, default=0.1)

    parser.add_argument("--num_eval_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--experiment_index", type=int, default=0)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dist_util.setup_dist(args)

    if dist.get_rank() == 0:
        os.makedirs(args.log_path, exist_ok=True)
        os.makedirs(args.out_path, exist_ok=True)
        model_string_name = args.model_path.split("/")[-1]
        experiment_dir = f"{args.log_path}/{model_string_name}/{args.experiment_index}"
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
    chair_dataset = CHAIRDataset(args.data_path, args.anno_path, trans, "qwen-vl")
    chair_loader = DataLoader(
        chair_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info("Start eval...")
    for batch_id, data in tqdm(enumerate(chair_loader), total=args.num_eval_samples):
        if batch_id == args.num_eval_samples:
            break

        img_id = data["image_id"]
        image_path = data["image_path"][0]

        qs = "Please describe this image in detail."
        query = tokenizer.from_list_format([
            {"image": image_path},
            {"text": qs},
        ])

        with torch.inference_mode():
            outputs, _ = model.chat(
                tokenizer,
                query,
                history=None,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=1.0,
            )

        logger.info("[Caption]")
        logger.info(f"V: {image_path}")
        logger.info(f"Q: {qs}")
        logger.info(f"A: {outputs}")
        logger.info("=" * 50)

        img_save = {"image_id": img_id.item(), "caption": outputs}
        with open(os.path.join(args.out_path, f"exp_{args.experiment_index:03d}.jsonl"), "a") as f:
            json.dump(img_save, f)
            f.write('\n')

    logger.info(vars(args))


if __name__ == "__main__":
    main()
