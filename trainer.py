import inspect
import math
import os
import time
from collections import defaultdict
from functools import partial
from test import evaluate_acc

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler
from tqdm import tqdm
from test import evaluate_acc

# def accelerate(cfg, accelerator, base_llm, optimizer, train_loader, val_loader):
#     base_llm, optimizer, train_loader = accelerator.prepare(base_llm, optimizer, train_loader)

#     if cfg.use_torch_compile:
#         base_llm = torch.compile(base_llm)
#     return base_llm, optimizer, train_loader, val_loader


def trainer(
  cfg,
  train_process,
  base_model,
  train_model, 
  train_loader,
  val_loader,
  logger,
  accelerator,
  concept_extractor = None      
):
    main_process = accelerator.is_main_process

    model_params = list(base_model.parameters())

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters

    if fused_available:
        logger.log('We use fuesd AdamW to train')
    
    optimizer = torch.optim.AdamW(
        model_params,
        lr = cfg.lr,
        weight_decay=cfg.weight_decay,
        eps=cfg.eps,
        fused=fused_available,
    )
    scheduler = get_scheduler(
        name=cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_ratio * cfg.train_steps,
        num_training_steps=cfg.train_steps,
        scheduler_specific_kwargs={"min_lr": cfg.min_lr_rate * cfg.lr},
    )

    # base_model, optimizer, train_loader, val_loader = accelerate(cfg, accelerator, base_model, optimizer, train_loader, val_loader)
    
    # optimizer, train_loader = accelerator.prepare(optimizer, train_loader)
    # val_loader = accelerator.prepare(val_loader)

    kwargs = {}

    if concept_extractor is not None:
        kwargs['concept_extractor'] = concept_extractor
    
    logger.log_dirname(f'Training Start')
    metrics_dic = defaultdict(lambda: [])


    if cfg.use_torch_compile:
        logger.log(f"Using torch compile... after first ppl evaluation it may take sometime to run...")

    for epoch in tqdm(range(cfg.n_epochs), desc="Training Progress"):
        
        for steps, batch in enumerate(train_loader):
            if (cfg.global_step != 0 and cfg.global_step % cfg.save_step_freq == 0 and steps % cfg.grad_acc_steps == 0):
                accelerator.wait_for_everyone()
                base_model_origin = base_model._orig_mod if cfg.use_torch_compile else base_model
                unwrapped_model = accelerator.unwrap_model(base_model_origin)
                unwrapped_model.save_pretrained(
                    os.path.join(cfg.output_dir, f"step_{cfg.global_step}"),
                    is_main_process=main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(base_model_origin, unwrap=False),
                )

                if main_process:
                    torch.save(
                        optimizer.state_dict(),
                        os.path.join(
                            cfg.output_dir, f"step_{cfg.global_step}", "optimizer.pt"
                        )
                    )

                    torch.save(
                        scheduler.state_dict(),
                        os.path.join(
                            logger.logdir, f"step_{cfg.global_step}", "scheduler.pt"
                        ),
                    )
                accelerator.wait_for_everyone()
            
            if (
                cfg.global_step % cfg.eval_step_freq == 0
                and steps % cfg.grad_acc_steps == 0
                and val_loader is not None
            ):
                logger.log('*' * 50)
                logger.log('Start accuracy test!')
                stime = time.time()
                acc = evaluate_acc(cfg, base_model, accelerator, val_loader, eval_limit=cfg.test_sample)                
                logger.log(f"Eval time for acc: {time.time()-stime:.2f}s")
                logger.wandb_log(acc, step=cfg.global_step)
                logger.log(f"Step {cfg.global_step}: Eval acc: {acc}")
                logger.log(f"End evaluation")
                logger.log(f"*" * 50)
                torch.cuda.empty_cache()
            train_process(
                cfg, 
                base_model,
                train_model,
                optimizer,
                scheduler,
                accelerator,
                batch,
                logger,
                metrics_dic,
                concept_extractor
            )
            if cfg.global_step >= cfg.train_steps:
                    break
        if cfg.global_step >= cfg.train_steps:
            break

        accelerator.wait_for_everyone()
        base_model_origin = base_model._orig_mod if cfg.use_torch_compile else base_model
        unwrapped_model = accelerator.unwrap_model(base_model_origin)
        unwrapped_model.save_pretrained(
            os.path.join(cfg.output_dir, f"step_{cfg.global_step}"),
            is_main_process=main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(base_model_origin, unwrap=False),
        )
        if main_process:
            torch.save(
                optimizer.state_dict(), os.path.join(logger.logdir, f"last", "optimizer.pt")
            )
            torch.save(
                scheduler.state_dict(), os.path.join(logger.logdir, f"last", "scheduler.pt")
            )



