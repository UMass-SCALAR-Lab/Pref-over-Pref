"""LLM-annotated PoP-random: an LLM (GPT-4.1 by default) decides which of two
preferences is stronger.

Random pairs (i, j) of UltraFeedback preferences are labelled by querying the LLM TWICE:
once with i as Preference A and j as B, and once with the order swapped (to cancel position
bias). The second answer is mapped back to the (i, j) order. If both answers agree on a
winner, that is the label; if they disagree, or either answer is 'C' / unparsable, the pair
is a tie and is discarded. Ground-truth margin agreement is logged for monitoring only.

Needs OPENAI_API_KEY in the environment (and optionally OPENAI_MODEL). Calls the paid API.

Example:
    python data_gen/annotate_pop_llm.py --num_prefs 5000 --k 2
Writes size_{N} checkpoints under $POP_DATA_ROOT/pop/random_llm_annotated/ at every
multiple of the preference count (5000, 10000), or at --checkpoint_sizes if given.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict

import numpy as np
from datasets import Dataset, load_dataset
from langchain.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tqdm.auto import tqdm


SYSTEM_PROMPT = """You are an expert evaluator of preference pairs. Your task is to determine which of two preferences is stronger.

A preference includes:
- A prompt
- A chosen response (better response)
- A rejected response (worse response)

A preference is considered STRONGER if there is a large discrepancy between the chosen and rejected responses — meaning the chosen response is clearly better.

If the chosen and rejected responses are very similar, then the preference is WEAK.

Example:

Preference A:
Prompt: Write a short poem about dogs?
Chosen: Soft eyes gleam in morning light,
    Paws that dance with pure delight.
    Loyal hearts that never tire,
    Warmth beside the evening fire.

    Through every storm, through every song,
    Their love is steady, fierce, and strong.
    No richer gift the world could send—
    Than one true dog, a faithful friend.
Rejected: Silent paws on midnight floors,
    Whiskers brush through shadowed doors.
    A flick of tail, a regal stare—
    Mystery hums upon the air.

    Sunlit naps and moonlit schemes,
    Guardians of our secret dreams.
    Independent, soft, and sly—
    Little moons with tiger eyes.

Preference B:
Prompt: Which of dog or cat has two hands?
Chosen: Neither dogs nor cats have hands, they both have four paws.
Rejected: This is an invalid question.

Analysis: Preference A is the stronger preference. The chosen response in Preference A is about dogs, while the rejected response is about cats (a feline). This is a clear and large discrepancy. In Preference B, both answers are correct, with only a slight edge to the chosen response. Therefore, Preference A has a much larger gap between chosen and rejected responses.

OUTPUT FORMAT:
The first line must ONLY contain: A, B, or C
- A if Preference A is stronger
- B if Preference B is stronger  
- C if you cannot determine which is stronger or if there is a tie

Second line: Provide a short, concise explanation for your choice.

IMPORTANT: Avoid position bias. Do not let the order of presentation or length of responses influence your evaluation. Be objective."""

USER_PROMPT = """Evaluate the following two preferences and determine which one is stronger.

Preference A:
Prompt: {PROMPT_A}
Chosen: {CHOSEN_RESPONSE_A}
Rejected: {REJECTED_RESPONSE_A}

Preference B:
Prompt: {PROMPT_B}
Chosen: {CHOSEN_RESPONSE_B}
Rejected: {REJECTED_RESPONSE_B}

Which preference is stronger? Remember: First line should be A, B, or C only."""


def _query_labels(llm, contents, query_info):
    """Send one user message per element of ``contents``; return 0 (A), 1 (B) or -1 (C / invalid)."""
    messages = [[SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=c)] for c in contents]
    with get_openai_callback() as cb:
        responses = llm.batch(messages)
        query_info['total_tokens'] += cb.total_tokens
        query_info['prompt_tokens'] += cb.prompt_tokens
        query_info['prompt_tokens_cached'] += cb.prompt_tokens_cached
        query_info['completion_tokens'] += cb.completion_tokens
        query_info['total_cost'] += cb.total_cost
        query_info['successful_requests'] += cb.successful_requests

    labels = []
    for response in responses:
        text = response.content.strip().split('\n')[0].strip().upper()
        labels.append({'A': 0, 'B': 1}.get(text, -1))
    return labels


def llm_labeler_gpt(llm, batch):
    """Label a batch of pairs (i, j) by querying both presentation orders.

    Returns (labels, usage stats); label 0 => i stronger, 1 => j stronger, -1 => tie
    (the two orders disagree, or either answer was C / invalid)."""
    query_info = defaultdict(float)
    labels_ij = _query_labels(llm, [b['message']['content'] for b in batch], query_info)
    labels_ji = _query_labels(llm, [b['message_flip']['content'] for b in batch], query_info)

    all_labels = []
    for lab_ij, lab_ji in zip(labels_ij, labels_ji):
        if lab_ji in (0, 1):
            lab_ji = 1 - lab_ji  # back to the original (i, j) ordering
        all_labels.append(lab_ij if (lab_ij in (0, 1) and lab_ij == lab_ji) else -1)
    return all_labels, query_info


def write_stats(path, stats):
    with open(path, 'w') as f:
        for k, v in stats.items():
            f.write(f'{k}: {v}\n')


def generate_pop_data_batched_labels(
    pref_data,
    pop_dsize,
    labeler,
    label_batch_size=1,
    max_attempts=50_000,
    candidate_pool=128,
    seed=42,
    data_dump_path='./pop_data',
    checkpoint_sizes=None,
):
    rng = random.Random(seed)
    total_query_info = defaultdict(float)

    pref_data = pref_data.map(
        lambda batch: {'gap': [sc - sr for sc, sr in zip(batch['score_chosen'], batch['score_rejected'])]},
        batched=True,
    )
    n = pref_data.num_rows

    # Intermediate datasets are saved when the pair count hits these sizes.
    if checkpoint_sizes is None:
        checkpoint_sizes = {m * n for m in range(1, pop_dsize // n + 1)} if n > 0 else set()
    checkpoint_sizes = set(checkpoint_sizes)

    def build_pop_dataset_from_pairs(pairs):
        pop = defaultdict(list)
        for p1, p2 in pairs:
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
            pop['pref_gap'].append(r1['gap'])
            pop['unpref_gap'].append(r2['gap'])
            pop['pop_gap'].append(r1['gap'] - r2['gap'])
        return Dataset.from_dict(pop)

    def row(k):
        r = pref_data[int(k)]
        return {
            'prompt': r['prompt'],
            'chosen': r['chosen'][-1]['content'],
            'rejected': r['rejected'][-1]['content'],
            'gap': r['gap'],
        }

    pairs_list, pairs_set = [], set()
    attempts = 0
    pbar = tqdm(total=pop_dsize, desc='Building PoP pairs', unit='pair')

    while len(pairs_list) < pop_dsize and attempts < max_attempts:
        prev_pairs_len = len(pairs_list)
        cand = []
        for _ in range(candidate_pool):
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            if (i, j) in pairs_set:
                continue
            cand.append((i, j))

        if not cand:
            attempts += candidate_pool
            continue

        batch_payload = []
        for i, j in cand:
            ri, rj = row(i), row(j)
            fmt_ij = USER_PROMPT.format(
                PROMPT_A=ri['prompt'], CHOSEN_RESPONSE_A=ri['chosen'], REJECTED_RESPONSE_A=ri['rejected'],
                PROMPT_B=rj['prompt'], CHOSEN_RESPONSE_B=rj['chosen'], REJECTED_RESPONSE_B=rj['rejected'],
            )
            fmt_ji = USER_PROMPT.format(
                PROMPT_A=rj['prompt'], CHOSEN_RESPONSE_A=rj['chosen'], REJECTED_RESPONSE_A=rj['rejected'],
                PROMPT_B=ri['prompt'], CHOSEN_RESPONSE_B=ri['chosen'], REJECTED_RESPONSE_B=ri['rejected'],
            )
            batch_payload.append({
                'i': i,
                'j': j,
                'message': {'role': 'user', 'content': fmt_ij},
                'message_flip': {'role': 'user', 'content': fmt_ji},
                'gap_i': ri['gap'],
                'gap_j': rj['gap'],
            })

        labels = []
        for start in range(0, len(batch_payload), label_batch_size):
            sub_batch = batch_payload[start:start + label_batch_size]
            sub_labels, sub_query_info = labeler(sub_batch)
            labels.extend(sub_labels)
            for k, v in sub_query_info.items():
                total_query_info[k] += v
        # Monitoring only: agreement with the ground-truth margin ordering.
        true_labels = [0 if (b['gap_i'] - b['gap_j'] > 0) else 1 for b in batch_payload]
        labels_arr, true_arr = np.array(labels), np.array(true_labels)
        non_tie = labels_arr != -1
        total_query_info['correct_labeling_wrt_margins'] += (true_arr == labels_arr).sum().item()
        total_query_info['num_ties'] += (~non_tie).sum().item()
        total_query_info['non_tie_correct_wrt_margins'] += (true_arr[non_tie] == labels_arr[non_tie]).sum().item()
        total_query_info['total_labeling_attempts'] += len(batch_payload)

        for (i, j), lab in zip(cand, labels):
            if len(pairs_list) >= pop_dsize:
                break
            if lab == 0:
                new_pair = (i, j)
            elif lab == 1:
                new_pair = (j, i)
            else:
                continue  # tie / invalid: discarded

            pairs_list.append(new_pair)
            pairs_set.add(new_pair)

            if len(pairs_list) in checkpoint_sizes:
                checkpoint_dir = os.path.join(data_dump_path, f'size_{len(pairs_list)}')
                os.makedirs(checkpoint_dir, exist_ok=True)
                print(f'Saving intermediate PoP dataset with {len(pairs_list)} pairs to {checkpoint_dir}')
                build_pop_dataset_from_pairs(pairs_list).save_to_disk(checkpoint_dir)

        snapshot_dir = os.path.join(data_dump_path, 'running_snapshot')
        os.makedirs(snapshot_dir, exist_ok=True)
        if len(pairs_list) > 0:
            np.save(os.path.join(snapshot_dir, 'pairs_list.npy'), np.array(pairs_list, dtype=np.int32))
        stats = {**total_query_info}
        total_attempts = float(stats.get('total_labeling_attempts', 0.0))
        if total_attempts > 0:
            decided = total_attempts - stats.get('num_ties', 0.0)
            if decided > 0:  # agreement with ground-truth margins among non-tie labels
                stats['accuracy'] = stats.get('non_tie_correct_wrt_margins', 0.0) / decided * 100.0
        write_stats(os.path.join(snapshot_dir, 'openai_query_stats_running.txt'), stats)

        new_pairs = len(pairs_list) - prev_pairs_len
        if new_pairs > 0:
            pbar.update(new_pairs)
        attempts += len(cand)

    pbar.close()
    print(f'Created {len(pairs_list)}/{pop_dsize} preference-over-preference pairs')
    if len(pairs_list) < pop_dsize:
        print('Unable to reach target size by the attempt limit; add more attempts or increase candidate_pool.')

    pop_dataset_hf = build_pop_dataset_from_pairs(pairs_list)
    final_dir = os.path.join(data_dump_path, f'size_{len(pairs_list)}')
    os.makedirs(final_dir, exist_ok=True)
    pop_dataset_hf.save_to_disk(final_dir)

    stats = {**total_query_info}
    if stats.get('total_labeling_attempts', 0.0) > 0:
        decided = stats['total_labeling_attempts'] - stats.get('num_ties', 0.0)
        if decided > 0:
            stats['accuracy'] = stats.get('non_tie_correct_wrt_margins', 0.0) / decided * 100
    print('Total query info: ', stats)
    write_stats(os.path.join(data_dump_path, 'openai_query_stats.txt'), stats)
    return pop_dataset_hf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--num_prefs', type=int, default=5000, help='preferences to sample from (shuffle seed 1234)')
    ap.add_argument('--k', type=float, default=2, help='target |D_PoP| = k * num_prefs (default 2)')
    ap.add_argument('--seed', type=int, default=42, help='seed for candidate-pair sampling')
    ap.add_argument('--model', default=os.environ.get('OPENAI_MODEL', 'gpt-4.1'))
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--max_tokens', type=int, default=256)
    ap.add_argument('--label_batch_size', type=int, default=1)
    ap.add_argument('--candidate_pool', type=int, default=128)
    ap.add_argument('--max_attempts', type=int, default=50_000)
    ap.add_argument('--checkpoint_sizes', type=int, nargs='+', default=None)
    ap.add_argument('--dataset_name', default='HuggingFaceH4/ultrafeedback_binarized')
    ap.add_argument('--cache_dir', default=None)
    ap.add_argument('--data_root', default=os.environ.get('POP_DATA_ROOT', './data'))
    ap.add_argument('--out', default='pop/random_llm_annotated', help='dir relative to --data_root')
    args = ap.parse_args()

    if not os.environ.get('OPENAI_API_KEY'):
        raise SystemExit('Set OPENAI_API_KEY in the environment.')

    train_data = load_dataset(args.dataset_name, cache_dir=args.cache_dir)['train_prefs']
    train_data = train_data.filter(
        lambda x: len(x['chosen'][-1]['content']) > 0 and len(x['rejected'][-1]['content']) > 0
    )
    train_data = train_data.filter(
        lambda b: [c[-1]['content'] != r[-1]['content'] for c, r in zip(b['chosen'], b['rejected'])],
        batched=True,
    )
    train_data = train_data.shuffle(seed=1234)
    train_data = train_data.select(range(min(args.num_prefs, train_data.num_rows)))

    print(f'Using model: {args.model}')
    llm = ChatOpenAI(model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)

    random.seed(args.seed)
    np.random.seed(args.seed)
    out_path = os.path.join(args.data_root, args.out)
    os.makedirs(out_path, exist_ok=True)
    pop_dsize = int(args.k * train_data.num_rows)
    generate_pop_data_batched_labels(
        train_data,
        pop_dsize,
        lambda b: llm_labeler_gpt(llm, b),
        label_batch_size=args.label_batch_size,
        max_attempts=args.max_attempts,
        candidate_pool=args.candidate_pool,
        seed=args.seed,
        data_dump_path=out_path,
        checkpoint_sizes=args.checkpoint_sizes,
    )


if __name__ == '__main__':
    main()
