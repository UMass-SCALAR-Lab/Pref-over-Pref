#!/usr/bin/env python3
"""Score generations with UltraRM-13b and compare against a baseline (Q2).

Two sub-commands.

``score``    (GPU; launch with ``accelerate launch`` inside a SLURM allocation)
    Reads a HF dataset of generations (columns ``prompt``, ``completion``, as written
    by ``generate_completions.py``), scores ``Human: {prompt}\\nAssistant: {completion}``
    with openbmb/UltraRM-13b, and saves the dataset with a ``reward`` column to
    ``--out_dir`` (plus ``rewards.npy``).

``compare``  (CPU, numpy only)
    For every method directory, pairs each seed with the baseline's seed (sorted by
    trailing integer of the directory name, matched by *position*) and reports the
    win rate  mean(r_method > r_baseline)  and the median advantage
    median(r_method - r_baseline), then mean +/- std (population) over seeds.

Layout::

    <scores_dir>/<method>/<seed>/{rewards.npy | HF dataset with 'reward'}
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from _common import list_subdirs, mean_std, save_json, sort_seeds

CHAT_PREFIX = 'BEGINNING OF CONVERSATION: USER: '
CHAT_SUFFIX = 'ASSISTANT:'


def score(args) -> None:
    import torch
    from accelerate import Accelerator
    from datasets import load_from_disk
    from torch import nn
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, PreTrainedModel
    from transformers.data.data_collator import DataCollatorForLanguageModeling

    class LlamaRewardModel(PreTrainedModel):
        """UltraRM head (as in the openbmb/UltraRM-13b model card)."""

        config_class = LlamaConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = LlamaModel(config)
            self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=False)

        def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                    past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                    output_attentions=None, output_hidden_states=None, return_dict=None):
            hidden = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                past_key_values=past_key_values, inputs_embeds=inputs_embeds)[0]
            rewards = self.regression_head(hidden).squeeze(-1)
            ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1, 1)
            return torch.gather(rewards, 1, ends)

    token = os.environ.get('HF_TOKEN')
    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = LlamaTokenizer.from_pretrained(args.rm_model_path, cache_dir=args.cache_dir, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    rm_model = LlamaRewardModel.from_pretrained(
        args.rm_model_path, cache_dir=args.cache_dir, torch_dtype=torch.bfloat16,
        device_map={'': Accelerator().local_process_index}, token=token,
    )
    rm_model.eval()
    accelerator = Accelerator()

    def tokenize(element):
        out = tokenizer(element['prompt_n_completion'], add_special_tokens=True, truncation=True,
                        padding=False, max_length=args.max_seq_length,
                        return_overflowing_tokens=False, return_length=False)
        return {'input_ids': out['input_ids'], 'attention_mask': out['attention_mask']}

    rm_model = accelerator.prepare(rm_model)
    gens = load_from_disk(args.generation_dir)
    # Strip the chat wrapper added by generate_completions.py and re-wrap in the UltraRM template.
    s, e = len(CHAT_PREFIX), len(CHAT_SUFFIX)
    gens = gens.map(
        lambda b: {'prompt_n_completion': [f'Human: {p[s:-e]}\nAssistant: {c}'
                                           for p, c in zip(b['prompt'], b['completion'])]},
        batched=True,
    )
    tokenized = gens.map(tokenize, batched=True, remove_columns=gens.column_names)
    loader = accelerator.prepare(DataLoader(
        tokenized, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        num_workers=0, pin_memory=False, persistent_workers=False,
    ))
    rewards = []
    for batch in tqdm(loader):
        with torch.no_grad():
            rew = accelerator.gather_for_metrics(rm_model(**batch))
            rewards.append(rew.flatten())
    rewards = torch.cat(rewards, dim=0)
    values = [r.item() for r in rewards]
    if accelerator.is_main_process:
        gens.add_column('reward', values).save_to_disk(args.out_dir)
        np.save(os.path.join(args.out_dir, 'rewards.npy'), np.asarray(values, dtype=np.float64))


def load_rewards(path: str) -> np.ndarray:
    npy = os.path.join(path, 'rewards.npy')
    if os.path.exists(npy):
        return np.load(npy)
    from datasets import load_from_disk

    return np.array(load_from_disk(path)['reward'])


def compare(args) -> None:
    base_root = os.path.join(args.scores_dir, args.baseline)
    base_seeds = sort_seeds(list_subdirs(base_root))
    base_rewards = [load_rewards(os.path.join(base_root, s)) for s in base_seeds]
    methods = [m for m in list_subdirs(args.scores_dir) if m != args.baseline]
    if args.methods:
        methods = [m for m in methods if m in args.methods]

    results = {}
    print(f'Baseline: {args.baseline} (seeds: {base_seeds})\n')
    print(f"{'method':<32}{'seed':<14}{'winrate':>9}{'median_adv':>12}")
    for method in methods:
        seeds = sort_seeds(list_subdirs(os.path.join(args.scores_dir, method)))
        if len(seeds) != len(base_seeds):
            print(f'WARNING: {method} has {len(seeds)} seeds vs {len(base_seeds)} baseline seeds; pairing the first {min(len(seeds), len(base_seeds))}.')
        per_seed = {}
        for seed, ref in zip(seeds, base_rewards):
            r = load_rewards(os.path.join(args.scores_dir, method, seed))
            adv = r - ref
            per_seed[seed] = {
                'winrate': float(np.mean(r > ref)),
                'median_adv': float(np.median(adv)),
                'mean_adv': float(np.mean(adv)),
                'std_adv': float(np.std(adv)),
                'iqr_adv': float(np.percentile(adv, 75) - np.percentile(adv, 25)),
            }
            print(f"{method:<32}{seed:<14}{per_seed[seed]['winrate']:>9.4f}{per_seed[seed]['median_adv']:>12.4f}")
        results[method] = {
            'winrate': mean_std([v['winrate'] for v in per_seed.values()]),
            'median_adv': mean_std([v['median_adv'] for v in per_seed.values()]),
            'per_seed': per_seed,
        }

    print(f"\n{'method':<32}{'WR mean':>9}{'WR std':>8}{'Adv mean':>10}{'Adv std':>9}  n")
    for method, r in results.items():
        print(f"{method:<32}{r['winrate']['mean']:>9.4f}{r['winrate']['std']:>8.4f}"
              f"{r['median_adv']['mean']:>10.4f}{r['median_adv']['std']:>9.4f}  {r['winrate']['n']}")
    if args.out:
        save_json({'baseline': args.baseline, 'results': results}, args.out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('score', help='Score one generation directory with UltraRM (GPU).')
    s.add_argument('--generation_dir', required=True)
    s.add_argument('--out_dir', required=True)
    s.add_argument('--rm_model_path', default='openbmb/UltraRM-13b')
    s.add_argument('--cache_dir', default=os.environ.get('HF_HOME'))
    s.add_argument('--max_seq_length', type=int, default=2048)
    s.add_argument('--eval_batch_size', type=int, default=8)
    s.set_defaults(fn=score)

    c = sub.add_parser('compare', help='Win rate / median advantage vs a baseline over seeds (CPU).')
    c.add_argument('--scores_dir', required=True, help='<scores_dir>/<method>/<seed>/')
    c.add_argument('--baseline', default='DPO-margin-0', help='Method directory of Vanilla DPO.')
    c.add_argument('--methods', nargs='*', default=None, help='Restrict to these methods.')
    c.add_argument('--out', default=None, help='JSON output (consumed by make_tables.py).')
    c.set_defaults(fn=compare)

    args = p.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
