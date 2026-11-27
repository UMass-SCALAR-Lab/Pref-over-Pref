#!/bin/bash
# SFT stage. Submit explicitly: BASE_MODEL=meta-llama/Llama-3.2-3B sbatch slurm/sft.sbatch
#SBATCH --job-name=pop-sft
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=superpod-a100
#SBATCH --mem=240G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
hostname; date
module load cuda/12.1 gcc/9.4.0
module list
# shellcheck disable=SC1090
source "${POP_ENV_ACTIVATE:?set POP_ENV_ACTIVATE to the path of your env activate script, or activate it before sbatch}"
which python && python --version

bash scripts/sft.sh --model_name_or_path "${BASE_MODEL:-meta-llama/Llama-3.2-3B}" \
	--output_dir "${OUT_ROOT:-runs}/sft"
