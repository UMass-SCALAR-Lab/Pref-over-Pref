"""Preference-over-Preference (PoP) datasets, loaded from disk.

Each PoP record pairs a *stronger* preference with a *weaker* one
(``pref_*`` = stronger, ``unpref_*`` = weaker). Datasets are produced by the scripts in
``data_gen/`` and stored with ``datasets.Dataset.save_to_disk`` under ``$POP_DATA_ROOT``
(default: ``./data``). Registered names:

    UF-PoP-Iter/train                     PoP-iter  (each preference vs. k weaker ones)
    UF-PoP-Random/train                   PoP-random (random pairs ordered by margin)
    UF-PoP-{Iter,Random}-k{1,2,4,8,16}/train   scale-of-PoP ablation
    UF-PoP-{Iter,Random}-corrupt-{p}/train     label-noise ablation (p in 0.1,0.3,0.5,0.7,0.9)
"""

from __future__ import annotations

import os
from typing import ClassVar

from datasets import load_dataset, load_from_disk

from pop_rlhf.datasets.base import RawDataset, RawSamplePoP


__all__ = ['UF_PoP_Dataset', 'POP_DATASETS']

_FIELDS = (
    'pref_prompt', 'pref_chosen', 'pref_rejected',
    'unpref_prompt', 'unpref_chosen', 'unpref_rejected',
    'pref_score_chosen', 'pref_score_rejected',
    'unpref_score_chosen', 'unpref_score_rejected',
    'pref_gap', 'unpref_gap', 'pop_gap',
)  # fmt: skip


HUB_REPO = 'yaswanthchittepu/pop-uf'  # https://huggingface.co/datasets/yaswanthchittepu/pop-uf


def data_root() -> str:
    return os.environ.get('POP_DATA_ROOT', './data')


class UF_PoP_Dataset(RawDataset):
    """A PoP dataset, read from disk at ``$POP_DATA_ROOT/<REL_PATH>`` if present there, else (for the
    datasets published on the Hub) downloaded from ``HUB_REPO`` / ``HUB_CONFIG``.

    On clusters whose compute nodes have no internet, pre-download on a login node
    (``huggingface-cli download yaswanthchittepu/pop-uf --repo-type dataset``) and run with
    ``HF_HUB_OFFLINE=1``, or save a local copy (see ``data/README.md``).
    """

    REL_PATH: ClassVar[str]
    HUB_CONFIG: ClassVar[str | None] = None

    def __init__(self, path: str | None = None) -> None:
        local = path or os.path.join(data_root(), self.REL_PATH)
        if os.path.isdir(local):
            self.data = load_from_disk(local)
        elif self.HUB_CONFIG is not None:
            self.data = load_dataset(HUB_REPO, self.HUB_CONFIG, split='train')
        else:
            raise FileNotFoundError(
                f'{self.NAME}: expected a dataset at {local}. Build it with data_gen/ '
                'or set POP_DATA_ROOT.',
            )
        # Drop degenerate records where either pair has identical responses.
        self.data = self.data.filter(
            lambda b: [
                c != r and uc != ur
                for c, r, uc, ur in zip(
                    b['pref_chosen'], b['pref_rejected'], b['unpref_chosen'], b['unpref_rejected'],
                )
            ],
            batched=True,
            load_from_cache_file=False,
            keep_in_memory=True,  # don't write cache-*.arrow files next to the (possibly read-only) data
        )

    def __getitem__(self, index: int) -> RawSamplePoP:
        data = self.data[index]
        return RawSamplePoP(**{k: data[k] for k in _FIELDS})

    def __len__(self) -> int:
        return len(self.data)


POP_DATASETS: dict[str, type[UF_PoP_Dataset]] = {}


def _register(name: str, rel_path: str, hub_config: str | None = None) -> None:
    cls = type(
        f'PoP[{name}]',
        (UF_PoP_Dataset,),
        {'NAME': name, 'REL_PATH': rel_path, 'HUB_CONFIG': hub_config, '__module__': __name__},
    )
    POP_DATASETS[name] = cls  # keep a strong ref: RawDataset's registry is a WeakValueDictionary


# Main datasets (k = 2 by default).
_register('UF-PoP-Iter/train', 'pop/iter/train', hub_config='iter')
_register('UF-PoP-Random/train', 'pop/random/train', hub_config='random')
# Smaller (4k-preference) PoP-iter set used in the small-data setting.
_register('UF-PoP-4k/train', 'pop/iter_4k/train')

for _kind in ('iter', 'random'):
    _tag = _kind.capitalize()
    for _k in (1, 2, 4, 8, 16):  # PoP scale |D_PoP| = k |D_pref|
        _register(f'UF-PoP-{_tag}-k{_k}/train', f'pop/{_kind}_k{_k}/train')
    for _p in ('0.1', '0.3', '0.5', '0.7', '0.9'):  # fraction of flipped PoP labels
        _register(f'UF-PoP-{_tag}-corrupt-{_p}/train', f'pop/{_kind}_corrupt-{_p}/train')

