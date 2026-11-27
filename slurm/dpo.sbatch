#!/bin/bash
# One DPO / DPO-PoP run on a single node. Submit explicitly, e.g.
#   VARIANT=pop-random SFT_CKPT=/path/to/sft/pytorch_model.bin SEED=0 sbatch slurm/dpo.sbatch
#SBATCH --job-name=pop-dpo
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=2-00:00:00
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

: "${VARIANT:?}" "${SFT_CKPT:?}"
SEED="${SEED:-42}"
bash scripts/dpo.sh --variant "${VARIANT}" --model_name_or_path "${SFT_CKPT}" --seed "${SEED}" \
	--output_dir "${OUT_ROOT:-runs}/${VARIANT}-seed${SEED}"
