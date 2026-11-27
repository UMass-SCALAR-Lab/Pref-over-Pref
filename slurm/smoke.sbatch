#!/bin/bash
# End-to-end smoke test of the pop-rlhf pipeline on 1 GPU with a tiny budget (~8 micro-steps/stage):
#   SFT -> vanilla DPO -> DPO-margin-gt-scaled -> DPO-PoP-iter -> DPO-PoP-random -> vLLM generation.
# Submit explicitly:  sbatch slurm/smoke.sbatch   (needs data/pop/{iter,random}/train, see data/README.md)
# Everything is read from local caches (offline). Heavy checkpoints go to node-local temp space and are
# removed at the end; small results are copied to logs/smoke-<jobid>/.
#SBATCH --job-name=pop-smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --partition=gpu-preempt-highvram
#SBATCH --exclude=gpu010   # job 2312456 hit 'CUDA device busy or unavailable' on this node
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
hostname; date; nvidia-smi -L
nvidia-smi --query-gpu=name,compute_mode,memory.used,memory.total --format=csv
module load cuda/12.1 gcc/9.4.0
module list
source "$(conda info --base)/etc/profile.d/conda.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export POP_DATA_ROOT="${SLURM_SUBMIT_DIR}/data"

BASE=meta-llama/Llama-3.2-1B-Instruct
WORK="$(mktemp -d "${TMPDIR:-/tmp}/pop-smoke.XXXXXX")"
RESULTS="${SLURM_SUBMIT_DIR}/logs/smoke-${SLURM_JOB_ID:-local}"
mkdir -p "${RESULTS}"
trap 'cp -r "${WORK}/gens" "${RESULTS}/" 2>/dev/null; rm -rf "${WORK}"' EXIT
echo "WORK=${WORK}"

PASS=(); FAIL=()
step() { # step <name> <cmd...>
	local name="$1"; shift
	echo; echo "=================== STEP: ${name} ($(date +%T)) ==================="
	if "$@"; then PASS+=("${name}"); else FAIL+=("${name}"); echo "!!! STEP FAILED: ${name}"; fi
}

SUB=0.002   # fraction of each dataset
COMMON=(--max_length 512 --max_steps 8 --gradient_accumulation_steps 2)

conda activate safe-rlhf
which python && python --version
# Fail fast (distinct exit code) if the allocated GPU is unusable, so this is not mistaken for a code bug.
python -c "import torch; x = torch.ones(8, device='cuda'); print('CUDA OK:', torch.cuda.get_device_name(0), float((x + 1).sum()))" \
	|| { echo '!!! GPU SANITY CHECK FAILED on' "$(hostname)"; exit 2; }
step sft bash scripts/sft.sh --model_name_or_path "${BASE}" --output_dir "${WORK}/sft" \
	--train_datasets "ultrafeedback-sft:${SUB}" --epochs 1 "${COMMON[@]}"
SFT="${WORK}/sft/pytorch_model.bin"
ls -la "${WORK}/sft" "${SFT}" 2>&1 | head -30

if [[ -f "${SFT}/config.json" ]]; then
	step dpo-vanilla bash scripts/dpo.sh --variant vanilla --model_name_or_path "${SFT}" \
		--output_dir "${WORK}/vanilla" --train_datasets "ultrafeedback-pref-nomargin/train:${SUB}" \
		--eval_datasets "ultrafeedback-pref-nomargin/test:${SUB}" "${COMMON[@]}"
	step dpo-margin-gt-scaled bash scripts/dpo.sh --variant margin-gt-scaled --model_name_or_path "${SFT}" \
		--output_dir "${WORK}/gt-scaled" --train_datasets "ultrafeedback-pref/train:${SUB}" \
		--eval_datasets "ultrafeedback-pref/test:${SUB}" "${COMMON[@]}"
	step dpo-pop-iter bash scripts/dpo.sh --variant pop-iter --model_name_or_path "${SFT}" \
		--output_dir "${WORK}/pop-iter" --train_datasets "UF-PoP-Iter/train:${SUB}" \
		--eval_datasets "ultrafeedback-pref/test:${SUB}" "${COMMON[@]}"
	step dpo-pop-random bash scripts/dpo.sh --variant pop-random --model_name_or_path "${SFT}" \
		--output_dir "${WORK}/pop-random" --train_datasets "UF-PoP-Random/train:${SUB}" \
		--eval_datasets "ultrafeedback-pref/test:${SUB}" "${COMMON[@]}"
else
	echo "SFT output has no loadable checkpoint -> skipping DPO stages"; FAIL+=("dpo-stages-skipped")
fi

conda activate "${POP_EVAL_ENV:-pop_submit}"
which python && python --version
if [[ -f "${WORK}/pop-random/pytorch_model.bin/config.json" ]]; then
	step generate python eval/generate_completions.py --model_path "${WORK}/pop-random/pytorch_model.bin" \
		--tokenizer "${BASE}" --out_dir "${WORK}/gens" --debug_small_data
	ls -la "${WORK}/gens" 2>&1 | head
else
	echo "no PoP checkpoint -> skipping generation"; FAIL+=("generate-skipped")
fi

echo; echo "=================== SUMMARY ==================="
echo "PASS: ${PASS[*]:-none}"
echo "FAIL: ${FAIL[*]:-none}"
date
[[ ${#FAIL[@]} -eq 0 ]]
