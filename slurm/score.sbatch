#!/bin/bash
# Score generations with the UltraRM-13b judge (1 GPU, bf16 ~26GB). Submit explicitly:
#   GENS=gens/POP-Random-k-2/seed-0 SCORES=scores/POP-Random-k-2/seed-0 sbatch slurm/score.sbatch
# UltraRM must be in the HF cache (or reachable); on offline nodes pre-download on a login node
# and export HF_HUB_OFFLINE=1. Afterwards (CPU, any node):
#   python eval/score_ultrarm.py compare --scores_dir scores --baseline DPO-margin-0 --out results/gen.json
#SBATCH --job-name=pop-score
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
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
: "${GENS:?}" "${SCORES:?}"

accelerate launch --num_processes 1 eval/score_ultrarm.py score \
	--generation_dir "${GENS}" \
	--out_dir "${SCORES}"
