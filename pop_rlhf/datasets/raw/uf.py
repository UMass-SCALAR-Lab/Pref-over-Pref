"""UltraFeedback datasets: SFT split and (binarized) preference splits.

The preference datasets come in two flavours:

* ``ultrafeedback-pref/{train,test}``          -- keeps ``score_chosen``/``score_rejected`` so the
  ground-truth margin is available (used by DPO-margin-gt / DPO-margin-gt-scaled).
* ``ultrafeedback-pref-nomargin/{train,test}`` -- scores dropped (Vanilla DPO / DPO-margin-c).
"""

from __future__ import annotations

import glob
import os
from typing import ClassVar

from datasets import load_dataset

from pop_rlhf.datasets.base import RawDataset, RawSample


__all__ = [
    'UF_SFT_Dataset',
    'UF_Pref_Dataset',
    'UF_Pref_No_Margin_Dataset',
]

HF_PATH = 'HuggingFaceH4/ultrafeedback_binarized'


def _load_uf_split(path: str | None, split: str):
    """Load an UltraFeedback split.

    Falls back to the cached Hub parquet files when ``load_dataset`` cannot be used, e.g. on compute
    nodes without internet whose processed-datasets cache was written by an incompatible
    ``datasets`` version.
    """
    repo = path or HF_PATH
    try:
        return load_dataset(repo, split=split)
    except Exception as ex:  # noqa: BLE001
        if os.path.isdir(repo):
            raise
        from huggingface_hub import snapshot_download

        kwargs = {'repo_type': 'dataset', 'allow_patterns': [f'data/{split}-*.parquet']}
        try:
            root = snapshot_download(repo, local_files_only=True, **kwargs)  # offline-safe first
        except Exception:  # noqa: BLE001  not in the local hub cache: download it (needs network)
            try:
                root = snapshot_download(repo, **kwargs)
            except Exception as ex2:
                raise RuntimeError(
                    f'Could not load {repo}:{split} with `datasets` ({ex!r}), and the parquet files are '
                    'neither in the local Hugging Face cache nor downloadable (offline?). Pre-download '
                    f'them on a node with internet: `huggingface-cli download {repo} --repo-type dataset`.',
                ) from ex2
        files = sorted(glob.glob(os.path.join(root, 'data', f'{split}-*.parquet')))
        if not files:
            raise FileNotFoundError(f'No parquet files for {repo}:{split} under {root}') from ex
        return load_dataset('parquet', data_files=files, split='train')


class UF_SFT_Dataset(RawDataset):
    NAME: str = 'ultrafeedback-sft'

    def __init__(self, path: str | None = None) -> None:
        self.data = _load_uf_split(path, 'train_sft')
        self.data = self.data.filter(lambda x: len(x['messages'][-1]['content']) > 0)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(input=data['prompt'], answer=data['messages'][-1]['content'])

    def __len__(self) -> int:
        return len(self.data)


def _load_valid_pairs(path: str | None, split: str):
    """Load a binarized split and drop empty / identical chosen-rejected pairs."""
    data = _load_uf_split(path, split)
    data = data.filter(
        lambda x: len(x['chosen'][-1]['content']) > 0 and len(x['rejected'][-1]['content']) > 0,
    )
    return data.filter(
        lambda b: [c[-1]['content'] != r[-1]['content'] for c, r in zip(b['chosen'], b['rejected'])],
        batched=True,
    )


class UF_Pref_Dataset(RawDataset):
    """Binarized preferences *with* ground-truth scores."""

    SPLIT: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = _load_valid_pairs(path, self.SPLIT)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['prompt'],
            answer=data['chosen'][-1]['content'],
            other_answer=data['rejected'][-1]['content'],
            better=True,
            score_chosen=data['score_chosen'],
            score_rejected=data['score_rejected'],
        )

    def __len__(self) -> int:
        return len(self.data)


class UF_Pref_No_Margin_Dataset(UF_Pref_Dataset):
    """Binarized preferences with scores removed (no margin information)."""

    def __init__(self, path: str | None = None) -> None:
        self.data = _load_valid_pairs(path, self.SPLIT).remove_columns(
            ['score_chosen', 'score_rejected'],
        )

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['prompt'],
            answer=data['chosen'][-1]['content'],
            other_answer=data['rejected'][-1]['content'],
            better=True,
        )


class UF_Pref_Train_Dataset(UF_Pref_Dataset):
    NAME: str = 'ultrafeedback-pref/train'
    SPLIT: str = 'train_prefs'


class UF_Pref_Test_Dataset(UF_Pref_Dataset):
    NAME: str = 'ultrafeedback-pref/test'
    SPLIT: str = 'test_prefs'


class UF_Pref_No_Margin_Train_Dataset(UF_Pref_No_Margin_Dataset):
    NAME: str = 'ultrafeedback-pref-nomargin/train'
    SPLIT: str = 'train_prefs'


class UF_Pref_No_Margin_Test_Dataset(UF_Pref_No_Margin_Dataset):
    NAME: str = 'ultrafeedback-pref-nomargin/test'
    SPLIT: str = 'test_prefs'
