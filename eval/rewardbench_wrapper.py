#!/usr/bin/env python3
"""Thin wrapper around the public ``rewardbench`` package (allenai/reward-bench).

``run``      call the ``rewardbench`` CLI for a DPO-trained policy: its implicit reward
             is scored against the reference (SFT) model via ``--ref_model``. Needs GPU;
             run inside a SLURM allocation. HF_TOKEN is read from the environment by
             huggingface_hub (gated Llama weights); nothing is passed on the command line.
``collect``  gather per-seed section scores into one JSON (mean/std over seeds) with
             Chat, Chat Hard, Safety, Reasoning and Overall (mean of the four sections,
             the RewardBench leaderboard convention).

Input of ``collect``: <rb_dir>/<method>/<seed>.json, each holding either the section
scores ``{"Chat": .., "Chat Hard": .., "Safety": .., "Reasoning": ..}`` or per-subset
accuracies under ``"results_grouped"`` / ``"extra_results"`` (grouped into sections with
``rewardbench``'s own ``calculate_scores_per_section``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

from _common import mean_std, save_json

SECTIONS = ('Chat', 'Chat Hard', 'Safety', 'Reasoning')


def run(args) -> None:
    cmd = ['rewardbench', f'--model={args.model}', f'--ref_model={args.ref_model}',
           f'--batch_size={args.batch_size}', f'--output_dir={args.output_dir}',
           '--dataset=allenai/reward-bench', '--split=filtered']
    if args.tokenizer:
        cmd.append(f'--tokenizer={args.tokenizer}')
    print(' '.join(cmd), flush=True)
    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)
        subprocess.run(cmd, check=True)


def sections_from_file(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    lower = {k.lower(): v for k, v in data.items()}
    if all(s.lower() in lower for s in SECTIONS):
        sec = {s: float(lower[s.lower()]) for s in SECTIONS}
    else:
        from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
        from rewardbench.utils import calculate_scores_per_section

        grouped = data.get('results_grouped') or data['extra_results']
        sec = {k: float(v) for k, v in calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, grouped).items()}
        sec = {s: sec[s] for s in SECTIONS}
    sec['Overall'] = sum(sec[s] for s in SECTIONS) / len(SECTIONS)
    return sec


def collect(args) -> None:
    results = {}
    for method in sorted(os.listdir(args.rb_dir)):
        mdir = os.path.join(args.rb_dir, method)
        if not os.path.isdir(mdir):
            continue
        per_seed = [sections_from_file(os.path.join(mdir, f)) for f in sorted(os.listdir(mdir)) if f.endswith('.json')]
        if per_seed:
            results[method] = {k: mean_std([r[k] for r in per_seed]) for k in (*SECTIONS, 'Overall')}
    for method, r in results.items():
        print(f'{method:<32}' + '  '.join(f"{k} {v['mean']:.4f}±{v['std']:.4f}" for k, v in r.items()))
    if args.out:
        save_json(results, args.out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--model', required=True, help='Trained DPO policy (HF id or local dir with config + tokenizer).')
    r.add_argument('--ref_model', required=True, help='Reference (SFT) model used for the implicit reward.')
    r.add_argument('--tokenizer', default=None)
    r.add_argument('--batch_size', type=int, default=8)
    r.add_argument('--output_dir', required=True)
    r.add_argument('--dry_run', action='store_true')
    r.set_defaults(fn=run)
    c = sub.add_parser('collect')
    c.add_argument('--rb_dir', required=True)
    c.add_argument('--out', default=None)
    c.set_defaults(fn=collect)
    args = p.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
