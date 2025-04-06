"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import torch.nn.functional as F
from utils import metric_synchronize_between_processes


def train_step(
    cfg,
    base_model,
    train_model,
    optimizer,
    scheduler,
    accelerator,
    batch,
    logger,
    metrics_dic,
    concept_extractor,
):

    # compute loss
    with accelerator.accumulate(base_model):

        sae_inputs, input_ids, question_len = torch.stack(batch["sae_inputs"]), torch.stack(batch["input_ids"]), torch.stack(batch["prompt_len"])

        concept_labels = concept_extractor(input_ids=sae_inputs)
        concept_labels = batch["concept_labels"]
        top_k_count = batch['topk']
   

        base_model.train()
        loss, concept_logit = train_model(
            input_ids=batch["input_ids"],
            question_length = batch["q_len"],
            top_k = top_k_count
        )

        loss_concept = torch.tensor(0.0).to(base_model.device)
        for t in range(cfg.pre_token):
            concept_logit = concept_logit[t]
            concept_labels = concept_labels[t]
            for i in range(top_k_count[t]):
                loss_concept += (
                    1
                    / top_k_count[t]
                    * F.cross_entropy(
                        concept_logit.view(-1, concept_logit.size(-1)),
                        concept_labels[:, i].contiguous().view(-1),
                    )
                )

        metrics_dic["loss"].append(loss.item())
        metrics_dic["loss_concept"].append(loss_concept.item())

        loss_total = loss + cfg.lambda_sae * loss_concept
        accelerator.backward(loss_total)

        if accelerator.sync_gradients:
            # clip gradient when using sync gradients
            grad_norm = accelerator.clip_grad_norm_(
                base_model.parameters(), cfg.grad_clip_thresh
            )
            metrics_dic["grad_norm"].append(grad_norm)

            # log metrics when using sync gradients (i.e., actual gradient update)
            if cfg.global_step % cfg.log_step_freq == 0:
                metric_synchronize_between_processes(
                    metrics_dic, accelerator
                )  # sync metrics across processes
                log_metrics = {
                    "train": {f"{k}": np.mean(v) for k, v in metrics_dic.items()},
                    "lr": optimizer.param_groups[0]["lr"],
                }
                logger.wandb_log(log_metrics, step=cfg.global_step)
                for k, v in metrics_dic.items():
                    logger.log(f"Step {cfg.global_step} Train {k}: {np.mean(v)}")

                metrics_dic.clear()
            cfg.global_step += 1

        optimizer.step()
        if accelerator.sync_gradients:
            scheduler.step()
        optimizer.zero_grad()
