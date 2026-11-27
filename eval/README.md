# Evaluation and tables

SLURM wrappers for generation and scoring (Q2.1, Q2.2): `slurm/generate.sbatch` and `slurm/score.sbatch` (see their headers).

Scripts for the paper's evaluation protocol. Only `margins_metrics.py`, `make_tables.py`
and `score_ultrarm.py compare` are CPU/numpy-only; everything else needs GPUs, model
downloads or API calls and must be run inside a SLURM allocation (not on a login node).
Credentials come from the environment only: `HF_TOKEN` (gated Llama / UltraRM weights),
`OPENAI_API_KEY` (AlpacaEval annotator). Set `HF_HOME` for caches.

Directory convention used below: `<root>/<method>/<seed>/...`, e.g. method
`DPO-margin-0` (Vanilla DPO) or `POP-Random-k-2`, seeds `seed-0`, `seed-2`, `seed-42`
Seeds are sorted by trailing integer and paired with the baseline's seeds **by position**.

## Q1 discriminative

1. Dump margins (per trained model; GPUs, multi-process via deepspeed/torchrun as for training):
   `python -m pop_rlhf.algorithms.dpo.main <same args as training> --eval_margins_only --save_npz_dir <margins>/<method> --save_npz_file <seed>.npz`
   producing `pref` (implicit margins) and `gt` (ground-truth margins) arrays.
2. `python eval/margins_metrics.py --margins_dir <margins> --out results/margins.json`
   Per file: accuracy `mean(pref >= 0)`, Pearson, Spearman (scipy), ECE (10 bins,
   `sigmoid(pref)` vs `gt >= 0`); mean ± population std (`np.std`) over files.
3. RewardBench: `python eval/rewardbench_wrapper.py run --model <policy> --ref_model <sft> --output_dir <rb>/<method>/<seed>`
   (calls the public `rewardbench` CLI), then
   `python eval/rewardbench_wrapper.py collect --rb_dir <rb> --out results/rewardbench.json`
   (Overall = mean of Chat, Chat Hard, Safety, Reasoning).

## Q2 generative

1. Generate (UltraFeedback test prompts, vLLM):
   `python eval/generate_completions.py --model_path <run>/pytorch_model.bin --tokenizer meta-llama/Llama-3.2-3B-Instruct --out_dir <gens>/<method>/<seed>`
   (`<run>/pytorch_model.bin` holds the weights; trained checkpoints already contain `config.json` and the tokenizer files).
2. Score with UltraRM (GPU): `accelerate launch eval/score_ultrarm.py score --generation_dir <gens>/<method>/<seed> --out_dir <scores>/<method>/<seed>`
3. Compare (CPU): `python eval/score_ultrarm.py compare --scores_dir <scores> --baseline DPO-margin-0 --out results/gen.json`
   Win rate `mean(r > r_base)` and median advantage `median(r - r_base)` per seed; mean ± population std over seeds.
4. AlpacaEval 2.0: `python eval/generate_completions.py --task alpaca ... --out_dir <alpaca>/<method>/<seed>`, then
   `python eval/alpacaeval_wrapper.py run --alpaca_dir <alpaca> [--dry_run]` and
   `python eval/alpacaeval_wrapper.py aggregate --alpaca_dir <alpaca> --out results/alpaca.json`
   (reads `<seed>/<annotators_config>/leaderboard.csv`; LC and raw win rate, avg length; mean ± sample std).

## Tables and plots

`python eval/make_tables.py --margins results/margins.json --gen results/gen.json --alpaca results/alpaca.json --rewardbench results/rewardbench.json [--methods ... --labels dir=Name ...] --out results/table1.md`

`python eval/plots.py {margin-hist | pop-gaps | noise} --help` (matplotlib imported lazily).
`noise` is the `plot_prf` figure (defaults are the values from the paper's logs).

## Evaluation settings

- Generation: vLLM, `n=1, temperature=0.8, top_p=1.0, top_k=50, repetition_penalty=1.3, presence/frequency_penalty=0, min_tokens=0, ignore_eos=False, max_tokens=1024`; dataset `HuggingFaceH4/ultrafeedback_binarized` split `test_prefs`, rows with an empty chosen/rejected answer filtered first.
- UF prompt: `BEGINNING OF CONVERSATION: USER: Prompt: {prompt}\n### Answer:  ASSISTANT:` (the double wrap and the double space are intentional: UltraRM scoring strips this wrapper again). AlpacaEval prompt: `BEGINNING OF CONVERSATION: USER: {instruction} ASSISTANT:`, generator name `<tokenizer>-<last component of out_dir>`.
- UltraRM: `openbmb/UltraRM-13b`, bf16, text `Human: {prompt}\nAssistant: {completion}` (chat wrapper stripped again), `max_seq_length=2048`, batch 8, pad = eos, reward read at the last non-pad token.
- AlpacaEval annotator: `alpaca_eval_gpt4_turbo_fn`.
- All "±" are population std (`np.std`) except AlpacaEval (sample std, ddof=1).

## Notes

- Accuracy in `margins_metrics.py` is `mean(pref >= 0)`; the trainer's own `test/accuracy` uses strict `chosen > rejected`.
- `rewardbench_wrapper.py` calls the public `rewardbench` CLI; check its flags and result format against your installed version.
- To include an extra run in a table, put its npz files / generations in another seed directory.
