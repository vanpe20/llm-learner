# import openai
import argparse
import pandas as pd
import json
# from litellm import completion, Client
from math_equivalence import is_equiv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
from math_equivalence import is_equiv




def evaluate_acc(cfg, base_model, accelerator, val_dataloader, eval_limit=None):
    device = accelerator.device

    if eval_limit is not None:
        eval_limit = eval_limit // accelerator.num_processes
    
    ans_list = []
    outputs = []
    num_total = 0
    base_model.eval()
    for batch in val_dataloader:
        with torch.no_grad():
            # inputs = base_model.to_tokens(batch['input_ids'])
            # device = base_model.device
            batch['input_ids'] = batch['input_ids'].to(device)
            outputs = base_model.module.generate(batch['input_ids'], max_new_tokens = cfg.test_max_length)
            string_ans = is_equiv(outputs, batch['answer'])
            ans_list.append(string_ans)

            num_total += batch["input_ids"].shape[0]
            if eval_limit is not None and num_total >= eval_limit:
                break
    acc = sum(ans_list) / cfg.test_sample
    return acc
