"""Cheap, CPU-only sanity checks (no models, no data on disk)."""
import torch

from pop_rlhf.algorithms.dpo_pop_fixed_ema.trainer import DPO_PoP_Fixed_EMA_Trainer
from pop_rlhf.datasets.base import RawDataset


def test_registry_contains_paper_datasets():
    names = set(RawDataset._RawDataset__REGISTRY.keys())
    for name in (
        'ultrafeedback-sft',
        'ultrafeedback-pref/train',
        'ultrafeedback-pref-nomargin/test',
        'UF-PoP-Iter/train',
        'UF-PoP-Random/train',
        'UF-PoP-Random-k16/train',
        'UF-PoP-Iter-corrupt-0.5/train',
    ):
        assert name in names, name


def test_polyak_update():
    p = torch.nn.Parameter(torch.zeros(3))
    DPO_PoP_Fixed_EMA_Trainer._ema_update(p, torch.ones(3), tau=0.99)
    assert torch.allclose(p.data, torch.full((3,), 0.01))


def test_hub_fallback(monkeypatch, tmp_path):
    from pop_rlhf.datasets.raw import pop as pop_mod

    calls = []

    class Fake:
        column_names = []

        def filter(self, *a, **k):
            return self

    monkeypatch.setattr(pop_mod, 'load_dataset', lambda *a, **k: calls.append((a, k)) or Fake())
    monkeypatch.setenv('POP_DATA_ROOT', str(tmp_path))  # empty -> falls back to the Hub
    RawDataset.load('UF-PoP-Random/train')
    assert calls[0][0] == ('yaswanthchittepu/pop-uf', 'random') and calls[0][1]['split'] == 'train'
    try:
        RawDataset.load('UF-PoP-Random-k4/train')  # not on the Hub, not on disk
        raise AssertionError('expected FileNotFoundError')
    except FileNotFoundError:
        pass
