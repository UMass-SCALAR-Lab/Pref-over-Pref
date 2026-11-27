"""Build a PoP (preference-over-preference) dataset from UltraFeedback.

A PoP record is (stronger preference, weaker preference); ``pref_*`` columns hold the
stronger pair (larger score gap = ``score_chosen - score_rejected``).

  --mode iter    PoP-iter: every preference (sorted by gap) is compared against k
                 weaker ones (gap difference >= --pop_margin), sampled without
                 replacement in round-robin passes over the preferences.
  --mode random  PoP-random: random pairs of preferences, kept when their gaps differ
                 by >= --pop_margin and ordered by gap.

|D_PoP| = k * |D_pref|.

Examples (from the repo root; $POP_DATA_ROOT or ./data is the data root):
    python data_gen/build_pop.py --mode iter   --split train_prefs
    python data_gen/build_pop.py --mode iter   --split test_prefs  --out pop/iter/test
    python data_gen/build_pop.py --mode random
    python data_gen/build_pop.py --mode iter   --subset_size 4000 --k 1 --out pop/iter_4k/train
    python data_gen/build_pop.py --mode random --k 8
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict

from datasets import Dataset, load_dataset


def binary_search(arr, val):
    """``arr`` is in descending order. Return the largest index i with arr[i] <= val, or -1."""
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] > val:
            lo = mid + 1
        else:
            hi = mid
    return lo if arr[lo] <= val else -1


def add_sorted_gap(pref_data):
    pref_data = pref_data.map(
        lambda batch: {
            'gap': [sc - sr for sc, sr in zip(batch['score_chosen'], batch['score_rejected'])],
        },
        batched=True,
    )
    return pref_data.sort('gap', reverse=True)


def sample_pairs_iter(pref_data, pop_dsize, pop_margin):
    """PoP-iter index pairs (stronger, weaker) over ``pref_data`` sorted by descending gap."""
    n = pref_data.num_rows
    gaps = pref_data['gap']

    # Groups of equal gap: group_idxs[g] is the first row of group g.
    group = []
    prev_gap = float('inf')
    for i in range(n):
        if prev_gap != gaps[i]:
            prev_gap = gaps[i]
            group.append((i, prev_gap))
    group_idxs, group_gaps = zip(*group)

    # Lowest row index that may be paired with each group (gap <= group gap - margin).
    valid_start_idxs = []
    for i in range(len(group_idxs)):
        valid_idx = binary_search(group_gaps, group_gaps[i] - pop_margin)
        valid_start_idxs.append(group_idxs[valid_idx] if valid_idx >= 0 else -1)

    total_possible_pairs = 0
    for i in range(len(group_idxs)):
        if valid_start_idxs[i] == -1:
            break
        num_rows_group = (group_idxs[i + 1] if i < len(group_idxs) - 1 else n) - group_idxs[i]
        total_possible_pairs += num_rows_group * (n - valid_start_idxs[i])
    print(f'Max number of pairs that can be created from the dataset: {total_possible_pairs}')

    pairs_list, pairs_set = [], set()
    idx_pop_cnt = defaultdict(int)
    iters = 0
    cnt, break_flag = 0, False
    max_iters = (n - valid_start_idxs[0]) if (valid_start_idxs and valid_start_idxs[0] > -1) else 0
    while iters < max_iters:
        cache_ix = 1
        for rix in range(n):
            if cnt == pop_dsize:
                break_flag = True
                break
            if not (cache_ix == len(group_idxs) or rix < group_idxs[cache_ix]):
                cache_ix += 1
            if valid_start_idxs[cache_ix - 1] == -1:
                break
            max_possible_pairs_rix = n - valid_start_idxs[cache_ix - 1]
            if idx_pop_cnt[rix] == max_possible_pairs_rix:
                break
            pair_idx = random.randint(valid_start_idxs[cache_ix - 1], n - 1)
            while (rix, pair_idx) in pairs_set:
                pair_idx = random.randint(valid_start_idxs[cache_ix - 1], n - 1)
            pairs_list.append((rix, pair_idx))
            pairs_set.add((rix, pair_idx))
            idx_pop_cnt[rix] += 1
            cnt += 1
        if break_flag:
            break
        iters += 1

    print(f'Created {len(pairs_list)}/{pop_dsize} preference over preference pairs')
    if len(pairs_list) < pop_dsize:
        print(f'Unable to pair {pop_dsize - len(pairs_list)} preferences', flush=True)
    return pairs_list


def sample_pairs_random(pref_data, pop_dsize, pop_margin, max_attempts=1_000_000):
    """PoP-random index pairs (stronger, weaker) over ``pref_data`` sorted by descending gap."""
    n = pref_data.num_rows
    gaps = pref_data['gap']
    pairs_list, pairs_set = [], set()
    attempts = 0
    while len(pairs_list) < pop_dsize and attempts < max_attempts:
        i, j = random.sample(range(n), 2)
        if (i, j) in pairs_set:
            attempts += 1
            continue
        gap_i, gap_j = gaps[i], gaps[j]
        if abs(gap_i - gap_j) >= pop_margin:
            p1, p2 = (i, j) if gap_i > gap_j else (j, i)
            pairs_list.append((p1, p2))
            pairs_set.add((p1, p2))
        attempts += 1

    print(f'Created {len(pairs_list)}/{pop_dsize} preference over preference pairs')
    if len(pairs_list) < pop_dsize:
        print(
            f'Unable to pair {pop_dsize - len(pairs_list)} preferences '
            'due to margin constraint or sampling limit.',
        )
    return pairs_list


def pairs_to_dataset(pref_data, pairs_list):
    pop = defaultdict(list)
    for p1, p2 in pairs_list:
        r1, r2 = pref_data[p1], pref_data[p2]
        pop['pref_prompt'].append(r1['prompt'])
        pop['unpref_prompt'].append(r2['prompt'])
        pop['pref_chosen'].append(r1['chosen'][-1]['content'])
        pop['unpref_chosen'].append(r2['chosen'][-1]['content'])
        pop['pref_rejected'].append(r1['rejected'][-1]['content'])
        pop['unpref_rejected'].append(r2['rejected'][-1]['content'])
        pop['pref_score_chosen'].append(r1['score_chosen'])
        pop['unpref_score_chosen'].append(r2['score_chosen'])
        pop['pref_score_rejected'].append(r1['score_rejected'])
        pop['unpref_score_rejected'].append(r2['score_rejected'])
        pop['pref_gap'].append(r1['score_chosen'] - r1['score_rejected'])
        pop['unpref_gap'].append(r2['score_chosen'] - r2['score_rejected'])
        pop['pop_gap'].append(pop['pref_gap'][-1] - pop['unpref_gap'][-1])
    return Dataset.from_dict(pop)


def generate_pop_data(pref_data, pop_dsize, mode, pop_margin=1.0):
    pref_data = add_sorted_gap(pref_data)
    if mode == 'iter':
        pairs = sample_pairs_iter(pref_data, pop_dsize, pop_margin)
    else:
        pairs = sample_pairs_random(pref_data, pop_dsize, pop_margin)
    return pairs_to_dataset(pref_data, pairs)


def load_uf_prefs(split, dataset_name, cache_dir, subset_size):
    data = load_dataset(dataset_name, cache_dir=cache_dir)[split]
    if split == 'train_prefs':
        # Drop empty / identical responses (train split only, as in the original scripts).
        data = data.filter(
            lambda x: len(x['chosen'][-1]['content']) > 0 and len(x['rejected'][-1]['content']) > 0
        )
        data = data.filter(
            lambda b: [c[-1]['content'] != r[-1]['content'] for c, r in zip(b['chosen'], b['rejected'])],
            batched=True,
        )
    if subset_size is not None:
        data = data.shuffle(seed=42)
        data = data.select(range(min(subset_size, data.num_rows)))
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['iter', 'random'], required=True)
    ap.add_argument('--k', type=int, default=2, help='|D_PoP| = k * |D_pref| (default 2)')
    ap.add_argument('--seed', type=int, default=42, help='seed for the pair sampler')
    ap.add_argument('--pop_margin', type=float, default=1.0, help='minimum gap difference')
    ap.add_argument('--split', default='train_prefs', choices=['train_prefs', 'test_prefs'])
    ap.add_argument('--subset_size', type=int, default=None,
                    help='shuffle (seed 42) and keep this many preferences, e.g. 4000 for iter_4k')
    ap.add_argument('--dataset_name', default='HuggingFaceH4/ultrafeedback_binarized')
    ap.add_argument('--cache_dir', default=None)
    ap.add_argument('--data_root', default=os.environ.get('POP_DATA_ROOT', './data'))
    ap.add_argument('--out', default=None,
                    help='output dir relative to --data_root; default pop/{mode}/{train|test} '
                         '(pop/{mode}_k{k}/train when k != 2)')
    args = ap.parse_args()

    if args.out is None:
        if args.split == 'test_prefs':
            args.out = f'pop/{args.mode}/test'
        elif args.k == 2:
            args.out = f'pop/{args.mode}/train'
        else:
            args.out = f'pop/{args.mode}_k{args.k}/train'
    out_path = os.path.join(args.data_root, args.out)

    random.seed(args.seed)
    pref_data = load_uf_prefs(args.split, args.dataset_name, args.cache_dir, args.subset_size)
    pop_dsize = args.k * pref_data.num_rows
    print(f'{pref_data.num_rows} preferences -> target {pop_dsize} PoP pairs ({args.mode}, k={args.k})')

    pop = generate_pop_data(pref_data, pop_dsize, args.mode, pop_margin=args.pop_margin)
    os.makedirs(out_path, exist_ok=True)
    pop.save_to_disk(out_path)
    print(f'Saved {len(pop)} records to {out_path}')


if __name__ == '__main__':
    main()
