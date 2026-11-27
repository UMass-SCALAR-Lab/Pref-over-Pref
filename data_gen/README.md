# data_gen: PoP dataset construction

Builds PoP datasets from UltraFeedback (`HuggingFaceH4/ultrafeedback_binarized`, split `train_prefs`). The ground-truth
margin of a preference is the UltraFeedback score gap `score_chosen - score_rejected`; it decides which preference of a
pair is the stronger one. Outputs are `datasets` directories under `--data_root` (default `$POP_DATA_ROOT`, else `./data`)
with the columns read by `pop_rlhf/datasets/raw/pop.py`:

`pref_prompt, pref_chosen, pref_rejected` (stronger), `unpref_prompt, unpref_chosen, unpref_rejected` (weaker),
`pref_score_chosen, pref_score_rejected, unpref_score_chosen, unpref_score_rejected`, `pref_gap, unpref_gap, pop_gap`.
The scores and gaps are not used for training.

Run from the repo root. Requires `datasets`; `annotate_pop_llm.py` also needs `langchain`, `langchain-openai`, `tqdm`
and `OPENAI_API_KEY`.

| Dataset | Command |
|---|---|
| `pop/iter/train` (PoP-iter, k=2) | `python data_gen/build_pop.py --mode iter` |
| `pop/random/train` (PoP-random, k=2) | `python data_gen/build_pop.py --mode random` |
| `pop/iter/test` | `python data_gen/build_pop.py --mode iter --split test_prefs` |
| `pop/iter_4k/train` | `python data_gen/build_pop.py --mode iter --subset_size 4000 --k 1 --out pop/iter_4k/train` |
| `pop/{iter,random}_k{K}/train` | `python data_gen/build_pop.py --mode iter --k 8` (default `--out` is `pop/{mode}_k{k}/train` for k != 2) |
| `pop/{iter,random}_noise-uniform-{p}/train` | `python data_gen/add_label_noise.py --kind iter --mode uniform` (swaps a fraction `p` of the records; reads `pop/{kind}/train`) |
| `pop/{iter,random}_noise-bt-temp-{T}/train` | `python data_gen/add_label_noise.py --kind iter --mode bt` (Bradley-Terry noise, temperature `T`) |
| `pop/random_llm_annotated/size_{N}` | `python data_gen/annotate_pop_llm.py --num_prefs 5000 --k 2` |

Options of `build_pop.py`: `--k` (PoP size is `k` times the number of preferences), `--seed` (default 42),
`--pop_margin` (minimum gap difference between the two preferences, default 1.0), `--subset_size`, `--out`.

## How pairs and labels are made

- **PoP-iter**: preferences are sorted by gap. In round-robin passes each preference is paired with a random weaker one
  (gap difference >= `--pop_margin`), without repeating a pair, until `k * |D_pref|` pairs exist.
- **PoP-random**: two preferences are drawn uniformly at random and kept if their gaps differ by at least
  `--pop_margin`; the one with the larger gap is the stronger preference.
- **LLM-annotated**: random pairs of preferences are shown to the LLM (default `gpt-4.1`, override with `--model`)
  twice, once in each A/B order. The label is kept only if both answers agree; disagreements and "cannot determine"
  answers count as ties and are discarded. Stats of the run are written to `openai_query_stats.txt`.
- Empty or identical chosen/rejected responses are filtered out of the training split.

`pop_to_pref.py` extracts the stronger preference of every PoP record into a plain preference dataset.

## Label noise

`add_label_noise.py` swaps the stronger and weaker preference of some records, which makes their label wrong.
Output datasets keep all columns and add a 0/1 `flipped` column. For swapped records all `pref_*` / `unpref_*` columns
are swapped; `pop_gap` is left unchanged.

- `--mode uniform`: each record is swapped independently with probability `p` (`--flip_probs`).
- `--mode bt`: each record is swapped with probability `sigmoid(pop_gap / T)` (`--temperatures`), so records with a
  larger margin difference are swapped more often, and a larger `T` moves the swap probability towards 0.5.
