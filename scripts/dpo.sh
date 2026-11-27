#!/bin/bash
# Preference optimisation from an SFT checkpoint. One script, one flag per paper variant:
#
#   --variant vanilla            Vanilla DPO
#   --variant margin-c           DPO-margin-c  (constant margin; set --margin, paper uses 1)
#   --variant margin-gt          DPO-margin-gt (margin = ground-truth UltraFeedback score gap)
#   --variant margin-gt-scaled   DPO-margin-gt-scaled (score gap as a loss weight, Scaled BT)
#   --variant pop-iter           DPO-PoP-iter   (ours)
#   --variant pop-random         DPO-PoP-random (ours)
#
#   bash scripts/dpo.sh --variant pop-random --model_name_or_path runs/sft/pytorch_model.bin --output_dir runs/pop-random
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

VARIANT=""
MODEL_NAME_OR_PATH=""
REF_MODEL_NAME_OR_PATH=""
OUTPUT_DIR=""
TRAIN_DATASETS=""   # override the variant's default dataset (e.g. a k-sweep or noise-ablation set)
EVAL_DATASETS=""
MARGIN=1.0          # only used by margin-c
EPOCHS=1
LEARNING_RATE=1e-6
SCALE_COEFF=0.1     # beta
MAX_LENGTH=2048
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
MARGIN_MIN=0.0      # PoP only: clip range of the weak-pair margin
MARGIN_MAX=10.0
TAU=0.99            # PoP only: Polyak coefficient of the target model
ZERO_STAGE=3
OFFLOAD=none
LOG_RUN_NAME=""
SEED=42
MAX_STEPS=""        # smoke tests: stop after this many micro-steps
parse_flags "$@"
[[ -n "${VARIANT}" && -n "${MODEL_NAME_OR_PATH}" ]] || { echo "--variant and --model_name_or_path are required" >&2; exit 1; }
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/runs/${VARIANT}-seed${SEED}}"
LOG_RUN_NAME="${LOG_RUN_NAME:-${VARIANT}-seed${SEED}}"

EXTRA=()
case "${VARIANT}" in
	vanilla)
		MODULE=pop_rlhf.algorithms.dpo
		TRAIN_DATASETS="${TRAIN_DATASETS:-ultrafeedback-pref-nomargin/train}"
		EVAL_DATASETS="${EVAL_DATASETS:-ultrafeedback-pref-nomargin/test}"
		EXTRA+=(--margin 0 --scaled_loss False) ;;
	margin-c)
		MODULE=pop_rlhf.algorithms.dpo
		TRAIN_DATASETS="${TRAIN_DATASETS:-ultrafeedback-pref-nomargin/train}"
		EVAL_DATASETS="${EVAL_DATASETS:-ultrafeedback-pref-nomargin/test}"
		EXTRA+=(--margin "${MARGIN}" --scaled_loss False) ;;
	margin-gt)
		MODULE=pop_rlhf.algorithms.dpo
		TRAIN_DATASETS="${TRAIN_DATASETS:-ultrafeedback-pref/train}"
		EVAL_DATASETS="${EVAL_DATASETS:-ultrafeedback-pref/test}"
		EXTRA+=(--scaled_loss False) ;;
	margin-gt-scaled)
		MODULE=pop_rlhf.algorithms.dpo
		TRAIN_DATASETS="${TRAIN_DATASETS:-ultrafeedback-pref/train}"
		EVAL_DATASETS="${EVAL_DATASETS:-ultrafeedback-pref/test}"
		EXTRA+=(--scaled_loss True) ;;
	pop-iter|pop-random)
		MODULE=pop_rlhf.algorithms.dpo_pop_fixed_ema
		[[ "${VARIANT}" == pop-iter ]] && DEFAULT_TRAIN="UF-PoP-Iter/train" || DEFAULT_TRAIN="UF-PoP-Random/train"
		TRAIN_DATASETS="${TRAIN_DATASETS:-${DEFAULT_TRAIN}}"
		EVAL_DATASETS="${EVAL_DATASETS:-ultrafeedback-pref/test}"
		EXTRA+=(--margin_min "${MARGIN_MIN}" --margin_max "${MARGIN_MAX}" --tau "${TAU}") ;;
	*) echo "Unknown --variant '${VARIANT}'" >&2; exit 1 ;;
esac
prepare_output_dir

deepspeed --master_port "$(pick_master_port)" \
	--module "${MODULE}" \
	--train_datasets "${TRAIN_DATASETS}" \
	--eval_datasets "${EVAL_DATASETS}" \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
	--ref_model_name_or_path "${REF_MODEL_NAME_OR_PATH}" \
	--max_length "${MAX_LENGTH}" \
	--trust_remote_code True \
	--epochs "${EPOCHS}" \
	--per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
	--per_device_eval_batch_size 2 \
	--gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
	--gradient_checkpointing \
	--learning_rate "${LEARNING_RATE}" \
	--lr_scheduler_type cosine \
	--lr_warmup_ratio 0.03 \
	--weight_decay 0.05 \
	--scale_coeff "${SCALE_COEFF}" \
	--seed "${SEED}" \
	--output_dir "${OUTPUT_DIR}" \
	--log_type wandb \
	--log_project PoP-DPO \
	--log_run_name "${LOG_RUN_NAME}" \
	--save_interval 300000 \
	--zero_stage "${ZERO_STAGE}" \
	--offload "${OFFLOAD}" \
	--max_grad_norm 5.0 \
	${MAX_STEPS:+--max_steps "${MAX_STEPS}"} \
	--bf16 True \
	--tf32 True \
	"${EXTRA[@]}"
