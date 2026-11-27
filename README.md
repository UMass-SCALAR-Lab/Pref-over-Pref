# Adaptive Margin RLHF via Preference over Preferences (PoP)

Code for [*Adaptive Margin RLHF via Preference over Preferences*](https://arxiv.org/abs/2509.22851)
(Chittepu, Singhal, Durrett, Niekum).

Instead of scalar scores, an annotator says which of two preferences is *stronger*. A PoP record is
`((x_s, y_s+, y_s-), (x_w, y_w+, y_w-))`. **DPO-PoP** trains the strong pair with a DPO loss whose margin is the
(detached, clipped) implicit margin of the *weak* pair under a Polyak-averaged copy of the policy. Compared to DPO
it costs one extra no-grad forward pass and one extra model copy.

Built on [PKU-Alignment/safe-rlhf](https://github.com/PKU-Alignment/safe-rlhf) (Apache-2.0, see `NOTICE`).

## Contents

| Path | What it is |
|---|---|
| `pop_rlhf/finetune/` | SFT |
| `pop_rlhf/algorithms/dpo/` | DPO baselines: vanilla, constant margin, ground-truth margin, scaled ground-truth margin |
| `pop_rlhf/algorithms/dpo_pop_fixed_ema/` | DPO-PoP (the loss is in `trainer.py`) |
| `pop_rlhf/datasets/raw/` | UltraFeedback and PoP datasets |
| `scripts/`, `slurm/` | launchers (`sft.sh`, `dpo.sh`) and SLURM templates |
| `data_gen/` | build PoP datasets from UltraFeedback ([docs](data_gen/README.md)) |
| `eval/` | generation, UltraRM judging, AlpacaEval, RewardBench, tables ([docs](eval/README.md)) |

## Setup

```bash
pip install -e .            # or: pip install -r requirements.txt
```
Training needs GPUs and [DeepSpeed](https://www.deepspeed.ai/). Generation (`eval/`) additionally needs vLLM.

## Train

```bash
# 1. SFT on UltraFeedback
bash scripts/sft.sh --model_name_or_path meta-llama/Llama-3.2-3B --output_dir runs/sft

# 2. Preference optimisation from the SFT checkpoint (weights are in <output_dir>/pytorch_model.bin)
bash scripts/dpo.sh --variant pop-random --model_name_or_path runs/sft/pytorch_model.bin --seed 0
```

`--variant` selects the method:

| Variant | Method |
|---|---|
| `vanilla` | DPO |
| `margin-c` | DPO with a constant margin (`--margin`) |
| `margin-gt` | DPO with the ground-truth UltraFeedback score gap as margin |
| `margin-gt-scaled` | DPO with the score gap as a loss weight |
| `pop-iter`, `pop-random` | DPO-PoP with PoP-iter / PoP-random data |

Useful options: `--output_dir`, `--seed`, `--train_datasets`, `--scale_coeff` (beta, 0.1), `--learning_rate` (1e-6),
`--epochs` (1), `--max_length` (2048), `--zero_stage` (3); PoP only: `--margin_min`, `--margin_max` (clip range of
the weak-pair margin, 0 and 10) and `--tau` (Polyak coefficient, 0.99). `--max_steps N` stops after `N` steps
(quick checks). Both scripts print every option with its default at the top.

On SLURM, `slurm/sft.sbatch` and `slurm/dpo.sbatch` wrap these scripts (settings via environment variables, see the
file headers), and `slurm/smoke.sbatch` runs every stage for a few steps as an end-to-end check.

## Data

`--train_datasets` accepts: `ultrafeedback-sft`, `ultrafeedback-pref/{train,test}`,
`ultrafeedback-pref-nomargin/{train,test}`, `UF-PoP-Iter/train`, `UF-PoP-Random/train`, `UF-PoP-4k/train`,
`UF-PoP-{Iter,Random}-k{1,2,4,8,16}/train` and `UF-PoP-{Iter,Random}-corrupt-{0.1,0.3,0.5,0.7,0.9}/train`.
Append `:<fraction>` to use a subset, e.g. `UF-PoP-Random/train:0.1`.

`UF-PoP-Iter/train` and `UF-PoP-Random/train` are downloaded automatically from the Hugging Face Hub
([`yaswanthchittepu/pop-uf`](https://huggingface.co/datasets/yaswanthchittepu/pop-uf)). To use local copies, or to build
the other PoP sets with `data_gen/`, point `POP_DATA_ROOT` at the data directory (default `./data`, see
[`data/README.md`](data/README.md)). On nodes without internet, download the datasets and models beforehand and set
`HF_HUB_OFFLINE=1`.

## Evaluate

See [`eval/README.md`](eval/README.md): implicit-margin metrics, vLLM generation, UltraRM win rate and median
advantage, AlpacaEval 2.0, RewardBench, and table/plot scripts (`slurm/generate.sbatch`, `slurm/score.sbatch`).

## Tests

`python -m pytest tests` (CPU only).

## Citation

```bibtex
@article{chittepu2025adaptive,
  title   = {Adaptive Margin RLHF via Preference over Preferences},
  author  = {Chittepu, Yaswanth and Singhal, Prasann and Durrett, Greg and Niekum, Scott},
  journal = {arXiv preprint arXiv:2509.22851},
  year    = {2025}
}
```
