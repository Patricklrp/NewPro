#!/bin/bash

seed=42
dataset_name="coco" # coco | aokvqa | gqa
type="adversarial" # random | popular | adversarial


# llava
model="llava"
model_root="/home/ciram25-liurp/models"
data_root="/home/ciram25-liurp/dataset"
model_path="${model_root}/llava-v1.5-7b"

# instructblip
# model="instructblip"
# model_path="${model_root}/instructblip-vicuna-7b"

pope_path="${data_root}/POPE/${dataset_name}/${dataset_name}_pope_${type}.json"
data_path="${data_root}/coco/val2014"

# data_path="/data/ce/data/gqa/images"

log_path="./logs"

use_ritual=False
use_vcd=False
use_m3id=False
use_diffusion=True

degf_alpha_pos=3.0
degf_alpha_neg=1.0
degf_beta=0.25

experiment_index=3

#####################################
# Run single experiment
#####################################
export CUDA_VISIBLE_DEVICES=1s
master_port=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PY
)
torchrun --nnodes=1 --nproc_per_node=1 --master_port ${master_port} eval_bench/pope_eval_${model}.py \
--seed ${seed} \
--model_path ${model_path} \
--model_base ${model} \
--pope_path ${pope_path} \
--data_path ${data_path} \
--log_path ${log_path} \
--use_ritual ${use_ritual} \
--use_vcd ${use_vcd} \
--use_m3id ${use_m3id} \
--use_diffusion ${use_diffusion} \
--degf_alpha_pos ${degf_alpha_pos} \
--degf_alpha_neg ${degf_alpha_neg} \
--degf_beta ${degf_beta} \
--experiment_index ${experiment_index}
