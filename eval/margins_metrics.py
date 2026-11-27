#!/usr/bin/env python3
"""Q1 discriminative analysis from the margin npz files (numpy/scipy only).

Input layout (written by ``pop_rlhf.algorithms.dpo.main --eval_margins_only``,
i.e. ``SupervisedTrainer.eval_margins``)::

    <margins_dir>/<method>/<anything>.npz      # one npz per seed
        pref : DPO implicit reward margin  beta*[(log pi/pi_ref)(chosen) - (...)(rejected)]
        gt   : ground-truth (UltraFeedback score_chosen - score_rejected) margin

Per file: accuracy = mean(pref >= 0), Pearson and Spearman between ``pref`` and
``gt``, and ECE (10 bins; probs = sigmoid(pref), labels = gt >= 0). Reported as
mean +/- std (population std, as in the paper scripts) over the files of a method.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from _common import mean_std, save_json

METRICS = ('acc', 'pearson', 'spearman', 'ece')


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error for binary labels."""
    probs = probs.reshape(-1)
    labels = labels.reshape(-1).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (probs >= lo) & (probs <= hi) if i == 0 else (probs > lo) & (probs <= hi)
        if not np.any(in_bin):
            continue
        ece += np.abs(np.mean(probs[in_bin]) - np.mean(labels[in_bin])) * np.mean(in_bin)
    return float(ece)


def file_metrics(path: str) -> dict:
    from scipy.stats import pearsonr, spearmanr

    data = np.load(path)
    gt, pref = data['gt'], data['pref']
    probs = 1 / (1 + np.exp(-pref))
    return {
        'acc': float((pref >= 0).mean()),
        'pearson': float(pearsonr(gt, pref)[0]),
        'spearman': float(spearmanr(gt, pref)[0]),
        'ece': compute_ece(probs, (gt >= 0).astype(float), n_bins=10),
    }


def collect(margins_dir: str, skip=()) -> dict:
    skip = {s.lower() for s in skip}
    results = {}
    for method in sorted(os.listdir(margins_dir)):
        mdir = os.path.join(margins_dir, method)
        if not os.path.isdir(mdir) or method.lower() in skip:
            continue
        files = sorted(f for f in os.listdir(mdir) if f.endswith('.npz'))
        if not files:
            continue
        per_file = {f: file_metrics(os.path.join(mdir, f)) for f in files}
        results[method] = {m: mean_std([v[m] for v in per_file.values()]) for m in METRICS}
        results[method]['files'] = files
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--margins_dir', required=True, help='Directory with one sub-directory per method, each holding one .npz per seed.')
    p.add_argument('--skip', nargs='*', default=[], help='Method directory names to skip (case-insensitive).')
    p.add_argument('--out', default=None, help='Optional JSON output (consumed by make_tables.py).')
    args = p.parse_args()

    results = collect(args.margins_dir, args.skip)
    print(f"{'method':<32}{'n':>3}  {'acc':>15}  {'pearson':>15}  {'spearman':>15}  {'ece':>15}")
    for method, r in results.items():
        cells = '  '.join(f"{r[m]['mean']:.4f} ± {r[m]['std']:.4f}" for m in METRICS)
        print(f"{method:<32}{r['acc']['n']:>3}  {cells}")
    if args.out:
        save_json(results, args.out)


if __name__ == '__main__':
    main()
