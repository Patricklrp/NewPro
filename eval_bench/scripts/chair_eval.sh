#!/bin/bash

seed=42 # 12 # 42

# llava | instructblip
model="instructblip"
model_root="/home/ciram25-liurp/models"
data_root="/home/ciram25-liurp/dataset"

if [ "${model}" = "llava" ]; then
	model_path="${model_root}/llava-v1.5-7b"
elif [ "${model}" = "instructblip" ]; then
	model_path="${model_root}/instructblip-vicuna-7b"
else
	echo "Unsupported model: ${model}" && exit 1
fi

coco_path="${data_root}/coco"
img_path="${coco_path}/val2014"
anno_path="${coco_path}/annotations/instances_val2014.json"

if [ ! -d "${img_path}" ]; then
	echo "COCO image path not found: ${img_path}" && exit 1
fi
if [ ! -f "${anno_path}" ]; then
	echo "COCO annotation not found: ${anno_path}" && exit 1
fi
if [ ! -d "${model_path}" ]; then
	echo "Model path not found: ${model_path}" && exit 1
fi
log_path="./logs/chair"
out_path="./chair_results/${model}"

use_ritual=False
use_vcd=False
use_m3id=False
use_diffusion=True

degf_alpha_pos=3.0
degf_alpha_neg=1.0
degf_beta=0.1

experiment_index=1

# Standard CHAIR evaluation uses 500 sampled images.
num_eval_samples=500
max_new_tokens=64

#####################################
# Run experiment
#####################################
export CUDA_VISIBLE_DEVICES=2
master_port=$(python3 - <<'PY'
import socket
s = socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PY
)

TORCHRUN_CMD="torchrun"
PYTHON_CMD="python3"
if [ "${CONDA_DEFAULT_ENV}" != "DeGF" ]; then
	if command -v conda >/dev/null 2>&1; then
		TORCHRUN_CMD="conda run -n DeGF torchrun"
		PYTHON_CMD="conda run -n DeGF python3"
	fi
fi

${TORCHRUN_CMD} --nnodes=1 --nproc_per_node=1 --master_port ${master_port} eval_bench/chair_eval_${model}.py \
--seed ${seed} \
--model_path ${model_path} \
--model_base ${model} \
--data_path ${img_path} \
--anno_path ${anno_path} \
--log_path ${log_path} \
--out_path ${out_path} \
--use_ritual ${use_ritual} \
--use_vcd ${use_vcd} \
--use_m3id ${use_m3id} \
--use_diffusion ${use_diffusion} \
--degf_alpha_pos ${degf_alpha_pos} \
--degf_alpha_neg ${degf_alpha_neg} \
--degf_beta ${degf_beta} \
--num_eval_samples ${num_eval_samples} \
--max_new_tokens ${max_new_tokens} \
--experiment_index ${experiment_index}

#####################################
# Run evaluation
#####################################
eval_index=$(printf "%03d" "${experiment_index}")
cap_json_path="${out_path}/exp_${eval_index}.jsonl"
echo ${cap_json_path}
${PYTHON_CMD} eval_bench/chair.py \
--cap_file ${cap_json_path} \
--coco_path ${coco_path}/annotations \
--save_path ${out_path}/exp_${eval_index}_result.jsonl \
--image_id_key image_id \
--caption_key caption