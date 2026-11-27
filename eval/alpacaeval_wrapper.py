#!/usr/bin/env python3
"""Thin wrapper around the public ``alpaca_eval`` CLI (AlpacaEval 2.0, LC + raw win rate).

``run``        call ``alpaca_eval --model_outputs <file> --annotators_config <cfg>`` for every
               <alpaca_dir>/<method>/<seed>/model_outputs_alpaca.json (written by
               ``generate_completions.py --task alpaca``). Needs OPENAI_API_KEY in the
               environment (never put it in files) and network access; the annotator
               calls cost money, so use ``--dry_run`` first.
``aggregate``  read the resulting <seed>/<cfg>/leaderboard.csv files and write
               mean/std/count (std is pandas' sample std, ddof=1, as in the paper's
               parser) of length_controlled_winrate, win_rate and avg_length per method.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

from _common import list_subdirs, save_json

DEFAULT_CONFIG = 'alpaca_eval_gpt4_turbo_fn'
OUTPUT_NAME = 'model_outputs_alpaca.json'
COLUMNS = ('length_controlled_winrate', 'win_rate', 'avg_length')


def run(args) -> None:
    if not args.dry_run and not os.environ.get('OPENAI_API_KEY'):
        sys.exit('OPENAI_API_KEY is not set in the environment.')
    for method in list_subdirs(args.alpaca_dir):
        if args.methods and method not in args.methods:
            continue
        for seed in list_subdirs(os.path.join(args.alpaca_dir, method)):
            outputs = os.path.join(args.alpaca_dir, method, seed, OUTPUT_NAME)
            if not os.path.exists(outputs):
                continue
            cmd = ['alpaca_eval', '--model_outputs', outputs, '--annotators_config', args.annotators_config]
            print(' '.join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True)


def _read_row(path: str, generator: str):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) > 1:  # leaderboard may also list reference models; keep ours
        rows = [r for r in rows if r.get('') == generator] or rows[:1]
    return rows[0]


def aggregate(args) -> None:
    import numpy as np

    results = {}
    for method in list_subdirs(args.alpaca_dir):
        vals = {c: [] for c in COLUMNS}
        for seed in list_subdirs(os.path.join(args.alpaca_dir, method)):
            seed_dir = os.path.join(args.alpaca_dir, method, seed)
            lb = os.path.join(seed_dir, args.annotators_config, 'leaderboard.csv')
            if not os.path.exists(lb):
                continue
            with open(os.path.join(seed_dir, OUTPUT_NAME)) as f:
                generator = json.load(f)[0]['generator']
            row = _read_row(lb, generator)
            for c in COLUMNS:
                vals[c].append(float(row[c]))
        if vals[COLUMNS[0]]:
            results[method] = {
                c: {'mean': float(np.mean(v)),
                    'std': float(np.std(v, ddof=1)) if len(v) > 1 else float('nan'),
                    'count': len(v), 'values': v}
                for c, v in vals.items()
            }
    for method, r in results.items():
        print(f"{method:<32}LC {r['length_controlled_winrate']['mean']:.2f} ± {r['length_controlled_winrate']['std']:.2f}  "
              f"WR {r['win_rate']['mean']:.2f} ± {r['win_rate']['std']:.2f}  (n={r['win_rate']['count']})")
    if args.out:
        save_json(results, args.out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name, fn in (('run', run), ('aggregate', aggregate)):
        s = sub.add_parser(name)
        s.add_argument('--alpaca_dir', required=True, help='<alpaca_dir>/<method>/<seed>/model_outputs_alpaca.json')
        s.add_argument('--annotators_config', default=DEFAULT_CONFIG)
        s.set_defaults(fn=fn)
        if name == 'run':
            s.add_argument('--methods', nargs='*', default=None)
            s.add_argument('--dry_run', action='store_true', help='Print the commands only.')
        else:
            s.add_argument('--out', default=None, help='JSON output (consumed by make_tables.py).')
    args = p.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
