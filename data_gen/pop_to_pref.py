"""Derive a plain preference dataset from the stronger pairs (``pref_*``) of a PoP dataset.

Used in the LLM-annotated ablation to train a preference-only baseline on the same
preferences that appear in the PoP data.

Example:
    python data_gen/pop_to_pref.py --src pop/random_llm_annotated/size_5000 \
        --out prefs_from_pop/random_llm_annotated/size_5000
"""

from __future__ import annotations

import argparse
import os

from datasets import Dataset, load_from_disk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='PoP dataset dir, relative to --data_root')
    ap.add_argument('--out', required=True, help='output dir, relative to --data_root')
    ap.add_argument('--data_root', default=os.environ.get('POP_DATA_ROOT', './data'))
    args = ap.parse_args()

    pop = load_from_disk(os.path.join(args.data_root, args.src))
    prefs = Dataset.from_dict({
        'prompt': pop['pref_prompt'],
        'chosen': pop['pref_chosen'],
        'rejected': pop['pref_rejected'],
        'score_chosen': pop['pref_score_chosen'],
        'score_rejected': pop['pref_score_rejected'],
    })
    out = os.path.join(args.data_root, args.out)
    os.makedirs(out, exist_ok=True)
    prefs.save_to_disk(out)
    print(f'Saved {len(prefs)} preferences to {out}')


if __name__ == '__main__':
    main()
