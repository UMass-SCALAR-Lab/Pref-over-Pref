#!/bin/bash
# Supervised fine-tuning on UltraFeedback (stage 1: produces the SFT policy that DPO starts from).
#
#   bash scripts/sft.sh --model_name_or_path meta-llama/Llama-3.2-3B --output_dir runs/sft
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-3B"
OUTPUT_DIR="${ROOT_DIR}/runs/sft"
TRAIN_DATASETS="ultrafeedback-sft"
EPOCHS=3
MAX_LENGTH=2048
PER_DEVICE_TRAIN_BATCH_SIZE=2
PER_DEVICE_EVAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=2e-5
ZERO_STAGE=3
OFFLOAD=none
LOG_RUN_NAME=sft
SEED=42
MAX_STEPS=""        # smoke tests: stop after this many micro-steps
parse_flags "$@"
prepare_output_dir

deepspeed --master_port "$(pick_master_port)" \
	--module pop_rlhf.finetune \
	--train_datasets "${TRAIN_DATASETS}" \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
	--max_length "${MAX_LENGTH}" \
	--trust_remote_code True \
	--epochs "${EPOCHS}" \
	--per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
	--per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
	--gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
	--gradient_checkpointing \
	--learning_rate "${LEARNING_RATE}" \
	--lr_scheduler_type cosine \
	--lr_warmup_ratio 0.03 \
	--weight_decay 0.0 \
	--seed "${SEED}" \
	--output_dir "${OUTPUT_DIR}" \
	--log_type wandb \
	--log_project PoP-SFT \
	--log_run_name "${LOG_RUN_NAME}" \
	--zero_stage "${ZERO_STAGE}" \
	--offload "${OFFLOAD}" \
	${MAX_STEPS:+--max_steps "${MAX_STEPS}"} \
	--bf16 True \
	--tf32 True
