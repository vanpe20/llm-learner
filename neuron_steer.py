import os
import torch
import torch.nn as nn
import transformer_lens
from sae_lens import SAE
import json
from torch.utils.data import DataLoader, Dataset
from torch.nn import DataParallel
from torch.nn import CrossEntropyLoss
from transformer_lens import ActivationCache
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from huggingface_hub import login
from tqdm import tqdm
import numpy as np
from huggingface_hub import hf_hub_download
from math_equivalence import is_equiv
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from reasoning_concept import TransformerLensSAE

global_pos = 0
length = 0

def generate_with_steering(model, prompt, positions, values, layer_num=31):
    """
    使用单个steering vector进行生成
    """
    # def steering_hook(activations, hook):
    #     # print(activations.shape)
    #     # print(ssd)
    #     global global_pos, length
    #     if global_pos == 0:
    #         length = activations.shape[1]

    #     global_pos +=1
    #     # print(current_pos)
    #     # print(ssd)
        
    #     # 只在生成的前几个token进行steering
    #     if global_pos == length:
    #         # current_state = activations[:, -1, :]
    #         latent = model.sae.encode(activations)
    #         for pos, val in zip(positions, values):
    #             latent[:, -1, :] = 0.0
    #             latent[:, -1 , pos] = val

    #         activations = model.sae.decode(latent)
        
    #     return activations

    # model.base_model.reset_hooks()
    # hook_point_name = f"blocks.{layer_num}.hook_resid_post"
    # model.base_model.add_hook(hook_point_name, steering_hook)
    
    # 生成文本
    tokens = model.base_model.to_tokens(prompt)
    # print(tokens.shape)
    output = model.base_model.generate(tokens, max_new_tokens=1000)

    output = model.tokenizer.decode(output[0], skip_special_tokens=True)

    return output




if __name__ == '__main__':


    # model = transformer_lens.HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8b-Instruct")
    # sae, cfg_dict, sparsity = SAE.from_pretrained(
    #     release = "llama_scope_lxr_8x", 
    #     sae_id = "l31r_8x",)

    Model = TransformerLensSAE(31)
    path = '/research/projects/trans_llm/Zeru_Shi/prm800k/prm800k/math_splits/test.jsonl'

    with open(path, 'r') as file:
        total_lines = sum(1 for _ in file)
    
    total = 0
    ans_1 = []
    ans_2 = []
    with open(path, 'r') as file:
        for i, line in tqdm(enumerate(file), total=total_lines, desc="Reading file"):

            item = json.loads(line.strip())
            # prompt = f"""
            #     Please reason step by step, and put your final answer within \boxed{{}}
                
            #     Question: {item['problem']} 
                
            #     Solution: {item['solution']}
            #     """
            # raw_label, raw_value, ex_label, ex_value = Model.single_latent(prompt)

            _prompt = f"""
                Please reason step by step, and put your final answer within \boxed{{}}
                
                Question: {item['problem']} 
                """
            # output = generate_with_steering(Model, _prompt, raw_label, raw_value)
            output = generate_with_steering(Model, _prompt, None, None)
            # output1 = generate_with_steering(Model, _prompt, ex_label, ex_value)
            string1 = is_equiv(output, item['answer'])
            # string2 = is_equiv(output1, item['answer'])
            ans_1.append(string1)
            # ans_2.append(string2)
            
            total +=1
            if total == 100:
                break
    print(f'With raw_SAE answer: {sum(ans_1)/total}')
    print(f'With ex_SAE answer: {sum(ans_2)/total}')
