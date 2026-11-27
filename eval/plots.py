#!/usr/bin/env python3
"""Paper figures. matplotlib is imported lazily inside each command.

``margin-hist``  histograms of implicit (``pref``) vs ground-truth (``gt``) margins from the
                 npz files of ``margins_metrics.py`` (density, 16 bins).
``pop-gaps``     histograms of the PoP-dataset gaps (pref_gap, unpref_gap, pop_gap) for the
                 Iter and Random preference-over-preference datasets, next to the
                 ground-truth UltraFeedback margin (score_chosen - score_rejected on train_prefs).
                 The PoP datasets are HF datasets on disk (``load_from_disk``) with columns
                 pref_gap / unpref_gap / pop_gap.
``noise``        the plot_prf figure: win rate and median advantage vs labeling-noise level
                 for PoP-Iter / PoP-Random. Defaults to the values from the paper's logs;
                 override with ``--data`` (JSON with keys noise, median_adv_iter,
                 median_adv_random, win_rate_iter, win_rate_random).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

BINS = 16

# Values extracted from the paper's evaluation logs (noise level -> metric).
NOISE_DEFAULT = {
    'noise': [0.0, 0.1, 0.3, 0.5],
    'median_adv_iter': [0.4008, 0.23046875, 0.3046875, 0.1875],
    'median_adv_random': [0.6625, 0.609375, 0.46875, 0.25],
    'win_rate_iter': [0.58, 0.54, 0.56, 0.54],
    'win_rate_random': [0.62, 0.60, 0.58, 0.56],
}


def margin_hist(args) -> None:
    import matplotlib.pyplot as plt

    methods = [m for m in sorted(os.listdir(args.margins_dir)) if os.path.isdir(os.path.join(args.margins_dir, m))]
    fig, axs = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.5), squeeze=False, sharey=True)
    for ax, method in zip(axs[0], methods):
        files = sorted(f for f in os.listdir(os.path.join(args.margins_dir, method)) if f.endswith('.npz'))
        data = np.load(os.path.join(args.margins_dir, method, files[0]))
        ax.hist(data['gt'], bins=BINS, alpha=0.5, density=True, label='ground truth')
        ax.hist(data['pref'], bins=BINS, alpha=0.5, density=True, label='implicit (DPO)')
        ax.set_title(method)
        ax.set_xlabel('Margin')
    axs[0][0].set_ylabel('Density')
    axs[0][0].legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)


def pop_gaps(args) -> None:
    import matplotlib.pyplot as plt
    from datasets import load_dataset, load_from_disk

    pop = [(load_from_disk(args.pop_iter_dir), 'iter'), (load_from_disk(args.pop_random_dir), 'random')]
    orig = load_dataset(args.uf_dataset, split='train_prefs', cache_dir=args.cache_dir)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key in ((axs[0, 0], 'pref_gap'), (axs[0, 1], 'unpref_gap'), (axs[1, 0], 'pop_gap')):
        for d, label in pop:
            ax.hist(d[key], bins=BINS, alpha=0.25, density=True, label=label)
        ax.set_title(key)
        ax.legend()
    gt = np.array(orig['score_chosen']) - np.array(orig['score_rejected'])
    axs[1, 1].hist(gt, bins=BINS, alpha=0.5, density=True, color='red')
    axs[1, 1].set_title('gt margins-UF dataset')
    for ax in axs.flat:
        ax.set_xlabel('Margin')
        ax.set_ylabel('Density')
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)


def noise(args) -> None:
    import json

    import matplotlib.pyplot as plt

    d = NOISE_DEFAULT
    if args.data:
        with open(args.data) as f:
            d = json.load(f)
    x = np.array(d['noise'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    panels = (('Win Rate', d['win_rate_iter'], d['win_rate_random']),
              ('Median Advantage', d['median_adv_iter'], d['median_adv_random']))
    for ax, (name, it, rnd) in zip(axes, panels):
        ax.plot(x, it, marker='o', label='Iter', linewidth=2)
        ax.plot(x, rnd, marker='o', label='Random', linewidth=2)
        ax.set_title(f'{name} vs Labeling noise level', fontsize=14)
        ax.set_xlabel('Labeling noise level', fontsize=12)
        ax.set_ylabel(name, fontsize=12)
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def add(name, fn, default_out):
        s = sub.add_parser(name)
        s.add_argument('--out', default=default_out)
        s.add_argument('--dpi', type=int, default=300)
        s.set_defaults(fn=fn)
        return s

    add('margin-hist', margin_hist, 'margin_hist.png').add_argument('--margins_dir', required=True)
    g = add('pop-gaps', pop_gaps, 'hist_pop_all.png')
    g.add_argument('--pop_iter_dir', required=True)
    g.add_argument('--pop_random_dir', required=True)
    g.add_argument('--uf_dataset', default='HuggingFaceH4/ultrafeedback_binarized')
    g.add_argument('--cache_dir', default=os.environ.get('HF_HOME'))
    add('noise', noise, 'pop_vs_noise_gen.png').add_argument('--data', default=None)
    args = p.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
