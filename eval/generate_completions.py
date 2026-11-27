#!/usr/bin/env python3
"""Generate completions with vLLM for the Q2 generative evaluation.

Tasks
  uf      UltraFeedback ``test_prefs`` prompts (default). Saves a HF dataset
          (columns ``prompt``, ``completion``) to ``--out_dir`` for ``score_ultrarm.py``.
  alpaca  AlpacaEval (tatsu-lab/alpaca_eval) instructions. Writes
          ``<out_dir>/model_outputs_alpaca.json`` for ``alpacaeval_wrapper.py``.

Prompt format and sampling parameters are those of the paper's original scripts.
Requires GPU + vllm; run it inside a SLURM allocation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

CHAT_PREFIX = 'BEGINNING OF CONVERSATION: USER: '
CHAT_SUFFIX = ' ASSISTANT:'


def uf_prompts(dataset_path: str, split: str):
    from datasets import load_dataset

    dataset = load_dataset(dataset_path)
    # Same filter as the original: drop rows with an empty chosen / rejected answer.
    dataset = dataset.filter(
        lambda x: len(x['chosen'][-1]['content']) > 0 and len(x['rejected'][-1]['content']) > 0
    )
    data = dataset[split]
    data = data.map(
        lambda b: {'prompt': [f'Prompt: {p}\n### Answer: ' for p in b['prompt']]},
        batched=True, remove_columns=data.column_names,
    )
    # NOTE: the original wraps the already-wrapped prompt, giving
    # "BEGINNING OF CONVERSATION: USER: Prompt: {p}\n### Answer:  ASSISTANT:".
    # Kept verbatim; score_ultrarm.py strips exactly this wrapper again.
    return data.map(
        lambda b: {'prompt': [f'{CHAT_PREFIX}{p}{CHAT_SUFFIX}' for p in b['prompt']]},
        batched=True,
    )


def alpaca_prompts():
    from datasets import load_dataset

    data = load_dataset('tatsu-lab/alpaca_eval', 'alpaca_eval', trust_remote_code=True)['eval']
    return data.map(
        lambda b: {'prompt': [f'{CHAT_PREFIX}{p}{CHAT_SUFFIX}' for p in b['instruction']]},
        batched=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_path', '-m', required=True, help='HF hub id or local model directory (e.g. <run>/pytorch_model.bin with config + tokenizer copied in).')
    p.add_argument('--tokenizer', default=None, help='Tokenizer id/path (default: model_path). Also used in the AlpacaEval generator name.')
    p.add_argument('--task', choices=['uf', 'alpaca'], default='uf')
    p.add_argument('--dataset_path', default='HuggingFaceH4/ultrafeedback_binarized')
    p.add_argument('--dataset_split', default='test_prefs')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--max_completion_length', type=int, default=1024)
    p.add_argument('--cache_dir', default=os.environ.get('HF_HOME'), help='Model download dir (default: $HF_HOME).')
    p.add_argument('--debug_small_data', action='store_true', help='Use only the first 10 prompts.')
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from datasets import Dataset

    os.makedirs(args.out_dir, exist_ok=True)
    data = alpaca_prompts() if args.task == 'alpaca' else uf_prompts(args.dataset_path, args.dataset_split)
    if args.debug_small_data:
        data = data.select(range(10))

    # Sampling parameters preserved from the original scripts.
    sampling_params = SamplingParams(
        n=1,
        presence_penalty=0,
        frequency_penalty=0,
        repetition_penalty=1.3,
        temperature=0.8,
        top_p=1.0,
        top_k=50,
        ignore_eos=False,
        max_tokens=args.max_completion_length,
        min_tokens=0,
        detokenize=True,
    )

    model = LLM(model=args.model_path, tokenizer=args.tokenizer, download_dir=args.cache_dir)
    start = time.time()
    outputs = model.generate(data['prompt'], sampling_params)
    print(f'Time taken to generate completions: {(time.time() - start) / 60.0:.2f} minutes', flush=True)

    if args.task == 'alpaca':
        exp = os.path.basename(os.path.normpath(args.out_dir))
        generator = f'{args.tokenizer}-{exp}'
        res = [
            {'instruction': data[i]['instruction'], 'output': o.outputs[0].text, 'generator': generator}
            for i, o in enumerate(outputs)
        ]
        with open(os.path.join(args.out_dir, 'model_outputs_alpaca.json'), 'w') as f:
            json.dump(res, f, indent=2)
    else:
        gen = defaultdict(list)
        for o in outputs:
            gen['prompt'].append(o.prompt)
            gen['completion'].append(o.outputs[0].text)
        Dataset.from_dict(gen).save_to_disk(args.out_dir)


if __name__ == '__main__':
    main()
