import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision import transforms as T
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import dist_util
from utils.logger import create_logger

from pope_loader import POPEDataSet
from degf_utils.image_generation import get_image_generation_pipeline, generate_image_stable_diffusion

torch.multiprocessing.set_sharing_strategy('file_system')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="POPE evaluation on Qwen-VL.")
    parser.add_argument("--model_path", type=str, default="/home/ciram25-liurp/models/Qwen-VL-Chat")
    parser.add_argument("--model_base", type=str, default="qwen-vl")
    
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--data_path", type=str, default="/mnt/server18_hard0/jhjang/LVLM/crg/data/coco/val2014")
    parser.add_argument("--pope_path", type=str, default="/mnt/server8_hard1/donguk/rips2024/experiments/data/POPE/coco/coco_pope_random.json")
    parser.add_argument("--log_path", type=str, default="/mnt/server16_hard0/sangmin/code/neurips2024/logs/pope")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--use_ritual", type=str2bool, default=False)

    parser.add_argument("--use_vcd", type=str2bool, default=False)
    parser.add_argument("--noise_step", type=int, default=500)
    
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    parser.add_argument("--use_diffusion", type=str2bool, default=False)
    
    parser.add_argument("--degf_alpha_pos", type=float, default=3)
    parser.add_argument("--degf_alpha_neg", type=float, default=1)
    parser.add_argument("--degf_beta", type=float, default=0.1)
    
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--experiment_index", type=int, default=0)

    args = parser.parse_args()
    return args


def log_experiment_args(logger, args):
    logger.info("===== Experiment Parameters =====")
    for key, value in sorted(vars(args).items()):
        logger.info(f"{key}: {value}")
    logger.info("=================================")


def print_acc(pred_list, label_list, logger):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    # unknown_ratio = pred_list.count(2) / len(pred_list)

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

    logger.info('TP\tFP\tTN\tFN\t')
    logger.info('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

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
        f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)

    return acc, precision, recall, f1, yes_ratio

def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out.split('\n'):

        line = line.replace('.', '')
        line = line.replace(',', '')
        words = line.split(' ')

        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
        break
    
    return pred_list


def _is_stop_words(generated_ids, stop_words_ids):
    for stop_ids in stop_words_ids:
        if len(stop_ids) == 0:
            continue
        if len(generated_ids) >= len(stop_ids) and generated_ids[-len(stop_ids):] == stop_ids:
            return True
    return False


def qwen_chat_with_diffusion_fusion(
    tokenizer,
    model,
    query_pos,
    query_neg,
    max_new_tokens,
    degf_alpha_pos,
    degf_alpha_neg,
    degf_beta,
    make_context_fn,
    decode_tokens_fn,
    get_stop_words_ids_fn,
):
    generation_config = model.generation_config
    max_window_size = getattr(generation_config, "max_window_size", 2048)
    chat_format = generation_config.chat_format

    raw_text_pos, context_tokens_pos = make_context_fn(
        tokenizer,
        query_pos,
        history=[],
        system="You are a helpful assistant.",
        max_window_size=max_window_size,
        chat_format=chat_format,
    )

    _, context_tokens_neg = make_context_fn(
        tokenizer,
        query_neg,
        history=[],
        system="You are a helpful assistant.",
        max_window_size=max_window_size,
        chat_format=chat_format,
    )

    stop_words_ids = get_stop_words_ids_fn(chat_format, tokenizer)
    eos_token_id = generation_config.eos_token_id

    input_ids_pos = torch.tensor([context_tokens_pos], device=model.device)
    input_ids_neg = torch.tensor([context_tokens_neg], device=model.device)

    generated_ids = []
    js_list = []
    beta = max(float(degf_beta), 1e-8)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs_pos = model(input_ids=input_ids_pos, return_dict=True)
            outputs_neg = model(input_ids=input_ids_neg, return_dict=True)

            next_token_logits = outputs_pos.logits[:, -1, :]
            next_token_logits_neg = outputs_neg.logits[:, -1, :]

            m = 0.5 * (F.softmax(next_token_logits, dim=-1) + F.softmax(next_token_logits_neg, dim=-1))
            js = 0.5 * F.kl_div(F.log_softmax(next_token_logits, dim=-1), m, reduction='batchmean') + \
                 0.5 * F.kl_div(F.log_softmax(next_token_logits_neg, dim=-1), m, reduction='batchmean')
            js_list.append(format(js.item(), '.4f'))

            if js < 0.1:
                diffs = next_token_logits + float(degf_alpha_pos) * next_token_logits_neg
            else:
                diffs = (1 + float(degf_alpha_neg)) * next_token_logits - float(degf_alpha_neg) * next_token_logits_neg

            cutoff = torch.log(torch.tensor(beta, device=next_token_logits.device, dtype=next_token_logits.dtype)) + \
                     next_token_logits.max(dim=-1, keepdim=True).values
            fused_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            next_token = torch.argmax(fused_logits, dim=-1)
            next_token_id = int(next_token.item())
            generated_ids.append(next_token_id)

            next_token_tensor = next_token.unsqueeze(-1)
            input_ids_pos = torch.cat([input_ids_pos, next_token_tensor], dim=-1)
            input_ids_neg = torch.cat([input_ids_neg, next_token_tensor], dim=-1)

            if eos_token_id is not None and next_token_id == eos_token_id:
                break
            if _is_stop_words(generated_ids, stop_words_ids):
                break

    response = decode_tokens_fn(
        input_ids_pos[0],
        tokenizer,
        raw_text_len=len(raw_text_pos),
        context_length=len(context_tokens_pos),
        chat_format=chat_format,
        verbose=False,
        errors='replace'
    )

    return response, js_list

def main():
    args = parse_args()
    dist_util.setup_dist(args)
    device = dist_util.device()
    torch.manual_seed(args.seed)
    experiment_dir = None
    
    if dist.get_rank() == 0:
        os.makedirs(args.log_path, exist_ok=True)
        model_string_name = args.model_path.split("/")[-1]
        experiment_dir = f"{args.log_path}/{model_string_name}/{args.degf_alpha_pos}_{args.degf_alpha_neg}_{args.degf_beta}/{args.experiment_index}"
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        log_experiment_args(logger, args)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    logger.info('Initializing Model')

    if args.use_ritual or args.use_vcd or args.use_m3id:
        logger.warning("Qwen-VL evaluation currently ignores use_ritual/use_vcd/use_m3id.")

    if args.use_diffusion and not torch.cuda.is_available():
        logger.warning("use_diffusion=True requires CUDA for stable diffusion. Falling back to baseline.")
        args.use_diffusion = False

    make_context_fn = None
    decode_tokens_fn = None
    get_stop_words_ids_fn = None
    if args.use_diffusion:
        if os.path.isdir(args.model_path) and args.model_path not in sys.path:
            sys.path.append(args.model_path)
        try:
            from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids
            make_context_fn = make_context
            decode_tokens_fn = decode_tokens
            get_stop_words_ids_fn = get_stop_words_ids
        except Exception as exc:
            logger.warning(f"Failed to import qwen_generation_utils ({exc}); falling back to baseline.")
            args.use_diffusion = False

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

    pope_dataset = POPEDataSet(
        pope_path=args.pope_path, 
        data_path=args.data_path,
        trans=trans,
        model='qwen-vl'
    )
    pope_loader = DataLoader(
        pope_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True
    )

    logger.info("Start eval...")
    pred_list, label_list = [], []

    pipe = None
    generated_dir = None
    if args.use_diffusion:
        logger.info("Initializing diffusion image generation pipeline...")
        pipe = get_image_generation_pipeline()
        if experiment_dir is not None:
            generated_dir = os.path.join(experiment_dir, "generated_images")
            os.makedirs(generated_dir, exist_ok=True)

    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        qs = data["query"][0]
        label = int(data["label"][0].item())
        image_path = data["image_path"][0]
        label_list.append(label)

        vqa_image_path = image_path

        if args.use_diffusion:
            desc_prompt = qs + " Briefly describe relevant details."
            desc_query = tokenizer.from_list_format([
                {"image": image_path},
                {"text": desc_prompt},
            ])

            logger.info("[Diffusion] Generating description")
            with torch.inference_mode():
                description, _ = model.chat(
                    tokenizer,
                    query=desc_query,
                    history=None,
                    do_sample=False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=128,
                )
            description = description.strip()

            logger.info(f"V: {image_path}")
            logger.info(f"Q: {desc_prompt}")
            logger.info(f"D: {description}")

            generated_image = generate_image_stable_diffusion(pipe, description)

            if generated_dir is not None:
                gen_path = os.path.join(generated_dir, f"{batch_id:06d}.png")
            else:
                gen_path = os.path.join(os.path.dirname(image_path), f"tmp_gen_{batch_id:06d}.png")

            generated_image.save(gen_path)
            vqa_image_path = image_path

            logger.info(f"GEN: {gen_path}")
            logger.info(f"DIFF-FUSION: logits fusion with base={image_path}, neg={gen_path}, degf_beta={args.degf_beta}")

        prompt = qs + " Please answer with yes or no only."
        query = tokenizer.from_list_format([
            {"image": vqa_image_path},
            {"text": prompt},
        ])

        logger.info(f"[VQA] use_diffusion={args.use_diffusion}")
        with torch.inference_mode():
            if args.use_diffusion:
                query_neg = tokenizer.from_list_format([
                    {"image": gen_path},
                    {"text": prompt},
                ])
                outputs, js_list = qwen_chat_with_diffusion_fusion(
                    tokenizer=tokenizer,
                    model=model,
                    query_pos=query,
                    query_neg=query_neg,
                    max_new_tokens=args.max_new_tokens,
                    degf_alpha_pos=args.degf_alpha_pos,
                    degf_alpha_neg=args.degf_alpha_neg,
                    degf_beta=args.degf_beta,
                    make_context_fn=make_context_fn,
                    decode_tokens_fn=decode_tokens_fn,
                    get_stop_words_ids_fn=get_stop_words_ids_fn,
                )
            else:
                outputs, _ = model.chat(
                    tokenizer,
                    query=query,
                    history=None,
                    do_sample=False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                )
        outputs = outputs.strip()
        pred_list = recorder(outputs, pred_list)
        
        logger.info(f"V: {vqa_image_path}")
        logger.info(f"Q: {qs}")
        logger.info(f"A: {outputs}")
        if args.use_diffusion:
            logger.info(" ".join(js_list))

        if label == 1: logger.info(f"GT: Yes")
        elif label == 0: logger.info(f"GT: No")

        acc, precision, recall, f1, yes_ratio = print_acc(pred_list, label_list, logger)
        acc = round(acc*100,2)
        precision = round(precision*100,2)
        recall = round(recall*100,2)
        f1 = round(f1*100,2)
        yes_ratio = round(yes_ratio*100,2)
        logger.info(
            f"acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}, yes_ratio: {yes_ratio}"
        )
        
        logger.info(f"="*50)
        if experiment_dir is not None:
            np.save(f"{experiment_dir}/pred_list.npy", pred_list)
            np.save(f"{experiment_dir}/label_list.npy", label_list)

    if len(pred_list) != 0:
        logger.info(vars(args))
        acc, precision, recall, f1, yes_ratio = print_acc(pred_list, label_list, logger)
        
        acc = round(acc*100,2)
        precision = round(precision*100,2)
        recall = round(recall*100,2)
        f1 = round(f1*100,2)
        yes_ratio = round(yes_ratio*100,2)
        
        logger.info(
            f"acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}, yes_ratio: {yes_ratio}"
        )

        if experiment_dir is not None:
            np.save(f"{experiment_dir}/pred_list.npy", pred_list)
            np.save(f"{experiment_dir}/label_list.npy", label_list)

if __name__ == "__main__":
    main()
