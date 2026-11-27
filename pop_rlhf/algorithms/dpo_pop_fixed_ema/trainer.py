# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""DPO-PoP: DPO with an adaptive, preference-over-preference margin.

Each training record is a pair of preferences ``(strong, weak)``. The strong pair gets the DPO
loss with a margin equal to the (detached, clipped) implicit reward margin of the *weak* pair,
measured under a Polyak-averaged target copy of the policy:

    L = -log sigmoid( beta * [ (log-ratio_chosen - log-ratio_rejected)_strong(pi_theta)
                               - clip_[margin_min, margin_max]( (log-ratio_chosen - log-ratio_rejected)_weak(pi_target) ) ] )

where ``beta = --scale_coeff`` and the log-ratios are w.r.t. the frozen reference policy. Note that
the clip is applied to the *unscaled* log-ratio difference (before multiplying by beta).
Compared to vanilla DPO this costs one extra no-grad forward pass (weak pair, target model) and
one extra model copy in memory.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import deepspeed
import numpy as np
import torch
import torch.nn.functional as F
from deepspeed.runtime.zero.partition_parameters import GatheredParameters, ZeroParamStatus
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from pop_rlhf.datasets import PreferenceDataset, PrefOverPrefDataset
from pop_rlhf.models import load_pretrained_models
from pop_rlhf.trainers import SupervisedTrainerEMA
from pop_rlhf.utils import gather_log_probabilities, get_all_reduce_mean, to_device


class DPO_PoP_Fixed_EMA_Trainer(SupervisedTrainerEMA):
    TRAINING_TYPE = 'dpo-pop-fixed-ema'
    DATASET_TYPE = PrefOverPrefDataset
    EVAL_DATASET_TYPE = PreferenceDataset

    model: deepspeed.DeepSpeedEngine  # policy pi_theta (trained)
    reference_model: deepspeed.DeepSpeedEngine  # frozen pi_ref
    model_ema: deepspeed.DeepSpeedEngine  # Polyak-averaged target pi_theta_hat (no grad)

    ds_train_config: dict[str, Any]
    ds_eval_config: dict[str, Any]

    def __init__(
        self,
        args: argparse.Namespace,
        ds_train_config: dict[str, Any],
        ds_eval_config: dict[str, Any],
    ) -> None:
        self.args = args
        self.ds_train_config = ds_train_config
        self.ds_eval_config = ds_eval_config
        self.scale_coeff = args.scale_coeff  # beta
        super().__init__(args, ds_train_config)

    def init_models(self) -> None:
        """Initialize policy, reference, and target (EMA) models and the tokenizer."""
        if self.ds_train_config is not None and self.ds_train_config['zero_optimization']['stage'] == 3:
            self.dstchf_train = HfDeepSpeedConfig(self.ds_train_config)
        if self.ds_eval_config is not None and self.ds_eval_config['zero_optimization']['stage'] == 3:
            self.dsechf_eval = HfDeepSpeedConfig(self.ds_eval_config)

        def load(path: str):
            return load_pretrained_models(
                path,
                model_max_length=self.args.max_length,
                padding_side='left',
                auto_model_type=AutoModelForCausalLM,
                trust_remote_code=self.args.trust_remote_code,
            )

        self.model, self.tokenizer = load(self.args.model_name_or_path)
        self.reference_model, _ = load(self.args.ref_model_name_or_path or self.args.model_name_or_path)
        self.model_ema, _ = load(self.args.model_name_or_path)

    def init_engines(self) -> None:
        super().init_engines()
        self.reference_model, *_ = deepspeed.initialize(
            model=self.reference_model,
            config=self.ds_eval_config,
        )
        self.model_ema, *_ = deepspeed.initialize(
            model=self.model_ema,
            config=self.ds_eval_config,
        )

    @staticmethod
    def compute_log_probs(
        model: AutoModelForCausalLM,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Per-token log probabilities of the given sequences."""
        logits = model(input_ids, attention_mask=attention_mask).logits
        return gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])

    def loss(  # pylint: disable=too-many-locals
        self,
        pref_chosen_input_ids,
        pref_chosen_attention_mask,
        pref_rejected_input_ids,
        pref_rejected_attention_mask,
        unpref_chosen_input_ids,
        unpref_chosen_attention_mask,
        unpref_rejected_input_ids,
        unpref_rejected_attention_mask,
        **kwargs,
    ):
        """DPO-PoP loss. ``pref_*`` is the stronger preference, ``unpref_*`` the weaker one."""
        batch_size = pref_chosen_input_ids.size(0)
        assert (
            batch_size
            == pref_rejected_input_ids.size(0)
            == unpref_chosen_input_ids.size(0)
            == unpref_rejected_input_ids.size(0)
        ), 'batch size mismatch!'

        # Strong pair under the policy (with grad).
        pref_chosen_log_probs, pref_rejected_log_probs = self.compute_log_probs(
            self.model.module,
            input_ids=torch.cat([pref_chosen_input_ids, pref_rejected_input_ids], dim=0),
            attention_mask=torch.cat([pref_chosen_attention_mask, pref_rejected_attention_mask], dim=0),
        ).chunk(chunks=2, dim=0)

        with torch.no_grad():
            # Weak pair under the Polyak-averaged target model.
            unpref_chosen_log_probs, unpref_rejected_log_probs = self.compute_log_probs(
                self.model_ema.module,
                input_ids=torch.cat([unpref_chosen_input_ids, unpref_rejected_input_ids], dim=0),
                attention_mask=torch.cat([unpref_chosen_attention_mask, unpref_rejected_attention_mask], dim=0),
            ).chunk(chunks=2, dim=0)

            # Both pairs under the frozen reference model.
            (
                ref_pref_chosen_log_probs,
                ref_pref_rejected_log_probs,
                ref_unpref_chosen_log_probs,
                ref_unpref_rejected_log_probs,
            ) = self.compute_log_probs(
                self.reference_model.module,
                input_ids=torch.cat(
                    [pref_chosen_input_ids, pref_rejected_input_ids, unpref_chosen_input_ids, unpref_rejected_input_ids],
                    dim=0,
                ),
                attention_mask=torch.cat(
                    [
                        pref_chosen_attention_mask,
                        pref_rejected_attention_mask,
                        unpref_chosen_attention_mask,
                        unpref_rejected_attention_mask,
                    ],
                    dim=0,
                ),
            ).chunk(chunks=4, dim=0)

        def end_index(mask: torch.Tensor) -> int:
            return mask.nonzero()[-1].squeeze().item()

        def pair_log_ratio(i, chosen_ids, rejected_ids, chosen_mask, rejected_mask, lp_c, lp_r, ref_c, ref_r):
            """Log-ratio of chosen/rejected w.r.t. the reference, summed from the first differing token."""
            diverge = (chosen_ids[i] != rejected_ids[i]).nonzero()[0].squeeze().item()
            c_slice = slice(diverge, end_index(chosen_mask[i]) + 1)
            r_slice = slice(diverge, end_index(rejected_mask[i]) + 1)
            chosen = lp_c[i, c_slice].sum(dim=-1) - ref_c[i, c_slice].sum(dim=-1)
            rejected = lp_r[i, r_slice].sum(dim=-1) - ref_r[i, r_slice].sum(dim=-1)
            return chosen, rejected

        losses = []
        rewards = {k: [] for k in ('pref_chosen', 'pref_rejected', 'unpref_chosen', 'unpref_rejected')}
        for i in range(batch_size):
            # Skip degenerate records (identical chosen/rejected).
            if torch.equal(pref_chosen_input_ids[i], pref_rejected_input_ids[i]):
                continue
            if torch.equal(unpref_chosen_input_ids[i], unpref_rejected_input_ids[i]):
                continue

            pref_c, pref_r = pair_log_ratio(
                i, pref_chosen_input_ids, pref_rejected_input_ids,
                pref_chosen_attention_mask, pref_rejected_attention_mask,
                pref_chosen_log_probs, pref_rejected_log_probs,
                ref_pref_chosen_log_probs, ref_pref_rejected_log_probs,
            )  # fmt: skip
            unpref_c, unpref_r = pair_log_ratio(
                i, unpref_chosen_input_ids, unpref_rejected_input_ids,
                unpref_chosen_attention_mask, unpref_rejected_attention_mask,
                unpref_chosen_log_probs, unpref_rejected_log_probs,
                ref_unpref_chosen_log_probs, ref_unpref_rejected_log_probs,
            )  # fmt: skip

            # Margin from the weak pair: no gradient (target model is under no_grad), clipped.
            weak_margin = torch.clamp(
                unpref_c - unpref_r,
                min=self.args.margin_min,
                max=self.args.margin_max,
            )
            losses.append(-F.logsigmoid(self.scale_coeff * ((pref_c - pref_r) - weak_margin)))

            rewards['pref_chosen'].append(self.scale_coeff * pref_c.detach())
            rewards['pref_rejected'].append(self.scale_coeff * pref_r.detach())
            rewards['unpref_chosen'].append(self.scale_coeff * unpref_c.detach())
            rewards['unpref_rejected'].append(self.scale_coeff * unpref_r.detach())

        loss = torch.stack(losses).mean()
        pref_chosen_reward = torch.stack(rewards['pref_chosen'])  # size = (B,)
        pref_rejected_reward = torch.stack(rewards['pref_rejected'])
        unpref_chosen_reward = torch.stack(rewards['unpref_chosen'])
        unpref_rejected_reward = torch.stack(rewards['unpref_rejected'])
        pref_reward_margin = pref_chosen_reward - pref_rejected_reward
        unpref_reward_margin = unpref_chosen_reward - unpref_rejected_reward

        return {
            'loss': loss,
            'pref_chosen_sample_reward': pref_chosen_reward,
            'pref_rejected_sample_reward': pref_rejected_reward,
            'unpref_chosen_sample_reward': unpref_chosen_reward,
            'unpref_rejected_sample_reward': unpref_rejected_reward,
            'accuracy': (pref_chosen_reward > pref_rejected_reward).float().mean(),
            'pref_reward_margin': pref_reward_margin,
            'unpref_reward_margin': unpref_reward_margin,
            'pop_margin': pref_reward_margin - unpref_reward_margin,
        }

    def train_step(
        self,
        pref_chosen_input_ids,
        pref_chosen_attention_mask,
        pref_rejected_input_ids,
        pref_rejected_attention_mask,
        unpref_chosen_input_ids,
        unpref_chosen_attention_mask,
        unpref_rejected_input_ids,
        unpref_rejected_attention_mask,
        **_,
    ):
        """Perform a single training step and update the target model."""
        loss_dict = self.loss(
            pref_chosen_input_ids=pref_chosen_input_ids,
            pref_chosen_attention_mask=pref_chosen_attention_mask,
            pref_rejected_input_ids=pref_rejected_input_ids,
            pref_rejected_attention_mask=pref_rejected_attention_mask,
            unpref_chosen_input_ids=unpref_chosen_input_ids,
            unpref_chosen_attention_mask=unpref_chosen_attention_mask,
            unpref_rejected_input_ids=unpref_rejected_input_ids,
            unpref_rejected_attention_mask=unpref_rejected_attention_mask,
        )
        loss = loss_dict['loss']
        self.model.backward(loss)
        self.model.step()

        if self.global_step % self.args.gradient_accumulation_steps == 0:
            self.sync_model_to_ema()

        with torch.no_grad():
            metrics = {
                'train/loss': get_all_reduce_mean(loss),
                'train/accuracy': get_all_reduce_mean(loss_dict['accuracy']),
            }
            for key in (
                'pref_chosen_sample_reward',
                'pref_rejected_sample_reward',
                'unpref_chosen_sample_reward',
                'unpref_rejected_sample_reward',
                'pref_reward_margin',
                'unpref_reward_margin',
                'pop_margin',
            ):
                metrics[f'train/{key}'] = get_all_reduce_mean(loss_dict[key].mean())

        out = {k: v.item() for k, v in metrics.items()}
        out['train/lr'] = self.model.optimizer.param_groups[0]['lr']
        return out

    @torch.no_grad()
    def sync_model_to_ema(self):
        """Polyak update of the target model: theta_hat <- tau * theta_hat + (1 - tau) * theta."""
        source = self.model.module
        target = self.model_ema.module
        tau = self.args.tau

        def full_data(p: torch.nn.Parameter) -> torch.Tensor:
            # ZeRO-3: parameters may be partitioned; gather before reading.
            if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                with GatheredParameters(p, modifier_rank=None):
                    return p.data.clone()
            return p.data.clone()

        for (_, param), (_, ema_param) in zip(source.named_parameters(), target.named_parameters()):
            src_data = full_data(param)
            if hasattr(ema_param, 'ds_id') and ema_param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                with GatheredParameters(ema_param, modifier_rank=None):
                    self._ema_update(ema_param, src_data, tau)
            else:
                self._ema_update(ema_param, src_data, tau)
            del src_data

    @staticmethod
    def _ema_update(ema_param: torch.nn.Parameter, src_data: torch.Tensor, tau: float) -> None:
        if ema_param.data.shape != src_data.shape:
            raise ValueError(f'Shape mismatch: {src_data.shape} vs {ema_param.data.shape}')
        ema_param.data.mul_(tau).add_(src_data, alpha=1 - tau)

    @torch.no_grad()
    def eval_test(self, max_batches: int = 1000, save_dir: str | None = None):
        """Evaluate the PoP loss and margins on (up to ``max_batches`` of) the training loader.

        If ``save_dir`` is given, the per-sample results are written to ``<save_dir>/eval_test_results.npz``.
        """
        keys = (
            'pref_chosen_sample_reward',
            'pref_rejected_sample_reward',
            'unpref_chosen_sample_reward',
            'unpref_rejected_sample_reward',
            'pref_reward_margin',
            'unpref_reward_margin',
            'pop_margin',
        )
        scalars: dict[str, list[float]] = {'loss': [], 'accuracy': []}
        arrays: dict[str, list[np.ndarray]] = {k: [] for k in (*keys, 'ground_pref_gap', 'ground_unpref_gap', 'ground_pop_gap')}

        self.model.eval()
        for i, batch in enumerate(self.train_dataloader):
            if i == max_batches:
                break
            batch = to_device(batch, self.args.device)
            loss_dict = self.loss(**batch)
            scalars['loss'].append(loss_dict['loss'].float().item())
            scalars['accuracy'].append(loss_dict['accuracy'].float().item())
            for k in keys:
                arrays[k].append(loss_dict[k].float().cpu().numpy())
            for gap, name in (('pref_gap', 'ground_pref_gap'), ('unpref_gap', 'ground_unpref_gap'), ('pop_gap', 'ground_pop_gap')):
                if gap in batch:
                    value = batch[gap]
                    arrays[name].append(value.float().cpu().numpy() if torch.is_tensor(value) else np.array(value))

        results = {k: np.mean(v) for k, v in scalars.items()}
        results.update({k: np.concatenate(v, axis=0) for k, v in arrays.items() if v})

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'eval_test_results.npz')
            np.savez(save_path, **results)
            print(f'Saved eval_test results to {save_path}')
        return results
