# Data

PoP datasets are stored as Hugging Face `datasets` directories under `$POP_DATA_ROOT/pop/` (default `./data/pop/`).
A record holds a stronger preference (`pref_*`) and a weaker one (`unpref_*`); see `../data_gen/README.md` for the columns.

| Path | Registry name | Rows |
|---|---|---|
| `pop/iter/train` | `UF-PoP-Iter/train` | 123,374 |
| `pop/random/train` | `UF-PoP-Random/train` | 121,260 |

These two are hosted at <https://huggingface.co/datasets/yaswanthchittepu/pop-uf> (configs `iter` and `random`) and are
downloaded automatically when no local copy exists. To save a local copy:

```python
from datasets import load_dataset
load_dataset("yaswanthchittepu/pop-uf", "iter", split="train").save_to_disk("data/pop/iter/train")
load_dataset("yaswanthchittepu/pop-uf", "random", split="train").save_to_disk("data/pop/random/train")
```

The other sets (`pop/iter_4k/train`, `pop/{iter,random}_k{K}/train`, `pop/{iter,random}_noise-uniform-{p}/train`, `pop/{iter,random}_noise-bt-temp-{T}/train`) are built
with the scripts in `data_gen/`.
