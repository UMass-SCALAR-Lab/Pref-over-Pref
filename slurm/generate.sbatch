#!/bin/bash
# Generate completions on UltraFeedback test prompts with vLLM (1 GPU). Submit explicitly:
#   MODEL=runs/pop-random-seed0 TOKENIZER=meta-llama/Llama-3.2-3B-Instruct \
#   OUT=gens/POP-Random-k-2/seed-0 sbatch slurm/generate.sbatch
# MODEL must be a HF-format dir (config.json + tokenizer files copied in next to the weights).
# Add TASK=alpaca for AlpacaEval prompts, SMOKE=1 for a 10-prompt smoke test.
#SBATCH --job-name=pop-gen
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=superpod-a100
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
hostname; date
module list
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${POP_EVAL_ENV:-pop_submit}"
which python && python --version
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
: "${MODEL:?}" "${OUT:?}"

python eval/generate_completions.py \
	--model_path "${MODEL}" \
	--tokenizer "${TOKENIZER:-${MODEL}}" \
	--task "${TASK:-uf}" \
	--out_dir "${OUT}" \
	${SMOKE:+--debug_small_data}
