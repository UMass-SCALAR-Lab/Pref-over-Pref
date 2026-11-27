"""Label-noise ablations for PoP datasets.

  --mode flip (default)  Swap pref_* / unpref_* with probability p (0.1, 0.3, 0.5, 0.7, 0.9).
                         All columns are swapped except ``pop_gap``, which is left as the
                         ground truth; a ``flipped`` 0/1 column is added.
                         Reads  pop/{kind}/train, writes pop/{kind}_corrupt-{p}/train.
  --mode bt              Bradley-Terry label noise: for each temperature T the label is
                         drawn as Bernoulli(sigmoid(pop_gap / T)); label 1 swaps the pair.
                         Writes pop/{kind}_bt-temp-{T}/train with only the 6 prompt/chosen/
                         rejected columns (score and gap columns are not kept).

Example:
    python data_gen/corrupt_pop.py --kind random
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from datasets import Dataset, load_from_disk


def flip_labels_with_probability(example, p):
    flipped = 0
    if np.random.rand() < p:
        example['pref_prompt'], example['unpref_prompt'] = example['unpref_prompt'], example['pref_prompt']
        example['pref_chosen'], example['unpref_chosen'] = example['unpref_chosen'], example['pref_chosen']
        example['pref_rejected'], example['unpref_rejected'] = example['unpref_rejected'], example['pref_rejected']
        example['pref_score_chosen'], example['unpref_score_chosen'] = (
            example['unpref_score_chosen'], example['pref_score_chosen'])
        example['pref_score_rejected'], example['unpref_score_rejected'] = (
            example['unpref_score_rejected'], example['pref_score_rejected'])
        example['pref_gap'], example['unpref_gap'] = example['unpref_gap'], example['pref_gap']
        # pop_gap is intentionally left unchanged (ground truth).
        flipped = 1
    example['flipped'] = flipped
    return example


def get_bt_perturbed_dataset(df, temperature):
    logits = np.array(df['pop_gap']) / temperature
    probs = 1 / (1 + np.exp(-logits))
    labels = np.random.binomial(1, probs)
    out = {k: [] for k in (
        'pref_prompt', 'unpref_prompt', 'pref_chosen', 'unpref_chosen',
        'pref_rejected', 'unpref_rejected',
    )}
    for i, label in enumerate(labels):
        a, b = ('pref', 'unpref') if label == 0 else ('unpref', 'pref')
        out['pref_prompt'].append(df[f'{a}_prompt'][i])
        out['unpref_prompt'].append(df[f'{b}_prompt'][i])
        out['pref_chosen'].append(df[f'{a}_chosen'][i])
        out['unpref_chosen'].append(df[f'{b}_chosen'][i])
        out['pref_rejected'].append(df[f'{a}_rejected'][i])
        out['unpref_rejected'].append(df[f'{b}_rejected'][i])
    return Dataset.from_dict(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kind', choices=['iter', 'random'], required=True)
    ap.add_argument('--mode', choices=['flip', 'bt'], default='flip')
    ap.add_argument('--probs', type=float, nargs='+', default=[0.1, 0.3, 0.5, 0.7, 0.9],
                    help='flip probabilities (--mode flip)')
    ap.add_argument('--temperatures', type=float, nargs='+', default=[1, 2, 4, 8, 16],
                    help='BT temperatures (--mode bt)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data_root', default=os.environ.get('POP_DATA_ROOT', './data'))
    ap.add_argument('--src', default=None, help='source dir relative to --data_root; default pop/{kind}/train')
    args = ap.parse_args()

    np.random.seed(args.seed)
    src = os.path.join(args.data_root, args.src or f'pop/{args.kind}/train')
    dataset = load_from_disk(src)

    if args.mode == 'flip':
        for p in args.probs:
            corrupted = dataset.map(lambda ex: flip_labels_with_probability(ex, p), load_from_cache_file=False)
            out = os.path.join(args.data_root, f'pop/{args.kind}_corrupt-{p}/train')
            os.makedirs(out, exist_ok=True)
            corrupted.save_to_disk(out)
            print(f'Dataset with corruption probability {p} saved to {out}')
    else:
        for temp in args.temperatures:
            perturbed = get_bt_perturbed_dataset(dataset, temp)
            out = os.path.join(args.data_root, f'pop/{args.kind}_bt-temp-{temp:g}/train')
            os.makedirs(out, exist_ok=True)
            perturbed.save_to_disk(out)
            print(f'BT-perturbed dataset (temperature {temp:g}) saved to {out}')


if __name__ == '__main__':
    main()
