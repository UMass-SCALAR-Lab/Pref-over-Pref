"""Label-noise ablations for PoP datasets: swap the stronger and weaker preference of some records.

A record whose two preferences are swapped has a wrong label. Two noise settings:

  --mode uniform  Every record is swapped independently with probability p
                  (--flip_probs, default 0.1 0.3 0.5 0.7 0.9).
                  Reads pop/{kind}/train, writes pop/{kind}_noise-uniform-{p}/train.

  --mode bt       Bradley-Terry noise with temperature T (--temperatures, default 1 2 4 8 16).
                  Every record is swapped with probability sigmoid(pop_gap / T), where
                  pop_gap = pref_gap - unpref_gap >= 0 is the ground-truth margin difference of the
                  record. Records with a larger pop_gap are therefore swapped more often, and as T grows the
                  swap probability moves towards 0.5.
                  Reads pop/{kind}/train, writes pop/{kind}_noise-bt-temp-{T}/train.

Output datasets have the same columns as the input plus a 0/1 ``flipped`` column. For a swapped record all
``pref_*`` / ``unpref_*`` columns are swapped; ``pop_gap`` is left unchanged (it stays the ground-truth margin
difference of the original, correctly ordered record).

Example:
    python data_gen/add_label_noise.py --kind random --mode bt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from datasets import load_from_disk


PAIR_FIELDS = (
    'prompt', 'chosen', 'rejected', 'score_chosen', 'score_rejected', 'gap',
)  # each exists as pref_<f> and unpref_<f>


def swap_preferences(example: dict) -> dict:
    """Swap the stronger and weaker preference of a record (a wrong label)."""
    for f in PAIR_FIELDS:
        example[f'pref_{f}'], example[f'unpref_{f}'] = example[f'unpref_{f}'], example[f'pref_{f}']
    return example  # pop_gap is intentionally left unchanged


def bt_swap_probability(pop_gap, temperature: float):
    """Probability of swapping a record in --mode bt: sigmoid(pop_gap / T)."""
    x = np.clip(np.asarray(pop_gap, dtype=np.float64) / temperature, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def add_noise(dataset, swap_mask):
    """Return ``dataset`` with the records where ``swap_mask`` is True swapped and a ``flipped`` column."""
    mask = [bool(m) for m in swap_mask]
    indexed = dataset.add_column('flipped', [int(m) for m in mask])
    return indexed.map(
        lambda ex: swap_preferences(ex) if ex['flipped'] else ex,
        load_from_cache_file=False,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kind', choices=['iter', 'random'], required=True)
    ap.add_argument('--mode', choices=['uniform', 'bt'], default='uniform')
    ap.add_argument('--flip_probs', type=float, nargs='+', default=[0.1, 0.3, 0.5, 0.7, 0.9],
                    help='swap probabilities (--mode uniform)')
    ap.add_argument('--temperatures', type=float, nargs='+', default=[1, 2, 4, 8, 16],
                    help='BT temperatures (--mode bt)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data_root', default=os.environ.get('POP_DATA_ROOT', './data'))
    ap.add_argument('--src', default=None, help='source dir relative to --data_root; default pop/{kind}/train')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dataset = load_from_disk(os.path.join(args.data_root, args.src or f'pop/{args.kind}/train'))
    pop_gap = np.array(dataset['pop_gap'])

    if args.mode == 'uniform':
        settings = [(p, f'uniform-{p:g}', np.full(len(dataset), p)) for p in args.flip_probs]
    else:
        settings = [(t, f'bt-temp-{t:g}', bt_swap_probability(pop_gap, t)) for t in args.temperatures]

    for value, name, swap_prob in settings:
        noisy = add_noise(dataset, rng.random(len(dataset)) < swap_prob)
        out = os.path.join(args.data_root, f'pop/{args.kind}_noise-{name}/train')
        os.makedirs(out, exist_ok=True)
        noisy.save_to_disk(out)
        print(f'{args.mode} noise ({value:g}): swapped {sum(noisy["flipped"])}/{len(noisy)} records -> {out}')


if __name__ == '__main__':
    main()
