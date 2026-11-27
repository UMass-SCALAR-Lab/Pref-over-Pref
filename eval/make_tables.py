#!/usr/bin/env python3
"""Aggregate the JSON outputs of the eval scripts into a Table-1-style markdown table.

Inputs (all optional; columns appear only for the files given)
  --margins     margins_metrics.py --out      -> UF Acc, Pearson, Spearman
  --gen         score_ultrarm.py compare --out -> Win rate, Median advantage (vs Vanilla DPO)
  --alpaca      alpacaeval_wrapper.py aggregate --out -> AlpacaEval 2.0 LC / WR
  --rewardbench rewardbench_wrapper.py collect --out  -> Chat, Chat Hard, Safety, Reasoning, Overall

Every cell is mean±std over seeds. Rows are matched by method directory name
(case-insensitive); use ``--methods`` to choose row order and ``--labels`` for display names.
"""

from __future__ import annotations

import argparse

from _common import load_json

DEFAULT_LABELS = {
    'dpo-margin-0': 'Vanilla DPO',
    'dpo-margin-1': 'DPO (margin=1)',
    'dpo-margin-gt': 'DPO (GT margin)',
    'dpo-margin-gt-scaled': 'DPO (scaled GT margin)',
    'pop-iter': 'PoP (Iter)',
    'pop-random': 'PoP (Random)',
    'pop-random-k-2': 'PoP (Random, k=2)',
}


def _cell(entry, digits, scale=1.0):
    if not entry:
        return '-'
    return f"{entry['mean'] * scale:.{digits}f}±{entry['std'] * scale:.{digits}f}"


def build(args) -> str:
    sources = {}
    columns = []  # (header, source key, getter, digits, scale)
    if args.margins:
        sources['margins'] = {k.lower(): v for k, v in load_json(args.margins).items()}
        columns += [('UF Acc', 'margins', lambda r: r.get('acc'), 3, 1),
                    ('Pearson', 'margins', lambda r: r.get('pearson'), 3, 1),
                    ('Spearman', 'margins', lambda r: r.get('spearman'), 3, 1)]
    if args.gen:
        sources['gen'] = {k.lower(): v for k, v in load_json(args.gen)['results'].items()}
        columns += [('Win rate', 'gen', lambda r: r.get('winrate'), 3, 1),
                    ('Median adv.', 'gen', lambda r: r.get('median_adv'), 3, 1)]
    if args.alpaca:
        sources['alpaca'] = {k.lower(): v for k, v in load_json(args.alpaca).items()}
        columns += [('AlpacaEval LC', 'alpaca', lambda r: r.get('length_controlled_winrate'), 2, 1),
                    ('AlpacaEval WR', 'alpaca', lambda r: r.get('win_rate'), 2, 1)]
    if args.rewardbench:
        sources['rb'] = {k.lower(): v for k, v in load_json(args.rewardbench).items()}
        columns += [(h, 'rb', (lambda h: lambda r: r.get(h))(h), 3, 1)
                    for h in ('Chat', 'Chat Hard', 'Safety', 'Reasoning', 'Overall')]
    if not columns:
        raise SystemExit('Provide at least one of --margins/--gen/--alpaca/--rewardbench.')

    labels = {**DEFAULT_LABELS}
    for item in args.labels or []:
        key, _, val = item.partition('=')
        labels[key.lower()] = val
    methods = [m.lower() for m in args.methods] if args.methods else \
        sorted({k for s in sources.values() for k in s})

    lines = ['| Method | ' + ' | '.join(c[0] for c in columns) + ' |',
             '|---|' + '---|' * len(columns)]
    for m in methods:
        cells = [_cell(col[2](sources[col[1]].get(m, {})), col[3], col[4]) for col in columns]
        lines.append(f'| {labels.get(m, m)} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--margins')
    p.add_argument('--gen')
    p.add_argument('--alpaca')
    p.add_argument('--rewardbench')
    p.add_argument('--methods', nargs='*', default=None, help='Method dir names, in row order.')
    p.add_argument('--labels', nargs='*', default=None, help='dir_name=Display name pairs.')
    p.add_argument('--out', default=None, help='Write markdown here as well as printing it.')
    args = p.parse_args()
    table = build(args)
    print(table)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(table + '\n')


if __name__ == '__main__':
    main()
