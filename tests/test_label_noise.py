"""CPU-only checks for data_gen/add_label_noise.py."""
import os
import sys

import numpy as np
from datasets import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_gen'))
import add_label_noise as noise  # noqa: E402


def _toy(n=6):
    d = {f'{w}_{f}': [f'{w}{f}{i}' for i in range(n)] for w in ('pref', 'unpref') for f in ('prompt', 'chosen', 'rejected')}
    for w, base in (('pref', 8.0), ('unpref', 2.0)):
        d[f'{w}_score_chosen'] = [base + 1.0] * n
        d[f'{w}_score_rejected'] = [1.0] * n
        d[f'{w}_gap'] = [base] * n
    d['pop_gap'] = [6.0] * n
    return Dataset.from_dict(d)


def test_bt_swap_probability_is_sigmoid_of_gap_over_temperature():
    gaps = np.array([0.0, 1.0, 4.0, 9.0])
    p = noise.bt_swap_probability(gaps, 1.0)
    assert abs(p[0] - 0.5) < 1e-12                  # equal margins: coin flip
    assert np.all(np.diff(p) > 0) and p[3] > 0.999  # swap probability grows with the margin difference
    assert np.allclose(p, 1 / (1 + np.exp(-gaps)))
    assert np.all(noise.bt_swap_probability(gaps[1:], 16.0) < p[1:])  # larger T -> closer to 0.5


def test_swap_keeps_columns_consistent():
    ds = _toy()
    out = noise.add_noise(ds, [True, False, True, False, False, False])
    assert out.column_names[:len(ds.column_names)] == ds.column_names and 'flipped' in out.column_names
    assert out['flipped'] == [1, 0, 1, 0, 0, 0]
    r0, r1 = out[0], out[1]
    assert r0['pref_chosen'] == 'unprefchosen0' and r0['unpref_chosen'] == 'prefchosen0'
    assert r0['pref_gap'] == 2.0 and r0['unpref_gap'] == 8.0 and r0['pop_gap'] == 6.0   # pop_gap unchanged
    assert r1['pref_chosen'] == 'prefchosen1' and r1['pop_gap'] == 6.0                  # untouched
