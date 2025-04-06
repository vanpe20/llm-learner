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
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from huggingface_hub import login
from tqdm import tqdm
import numpy as np
from huggingface_hub import hf_hub_download
# from math_equivalence import is_equiv
from accelerate import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import os
import copy
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

removed_colume = ['solution', 'answer', 'problem_type', 'question_type', 'problem_is_valid', 'solution_is_valid', 'source', 'synthetic', 'finish_reasons', 'prompt_tokens', 'api_metadata']


class TransformerLensSAE(nn.Module):
    def __init__(self, layer_index ,base_model = None , tokenizer = None, model_name = 'meta-llama/Llama-3.1-8b-Instruct', release = 'llama_scope_lxr_8x', max_length = 2000, pre_token = 20):
        super().__init__()
        self.layer_index = layer_index
        # self.base_model = base_model
        self.base_model = transformer_lens.HookedTransformer.from_pretrained(model_name, device_map = 'auto')
        self.base_model.eval()
        

        # self.tokenizer = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        self.sae, _ , _= SAE.from_pretrained(
            release=release,
            sae_id=f"l{self.layer_index}r_8x")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sae = self.sae.to(device)

        self.transformer_lens_loc = f"blocks.{layer_index}.hook_resid_post"

        self.pre_token = pre_token
        self.current_pos = 0


    def get_latent_activation(self, tokens, new_act = None):
        self.base_model.reset_hooks()
        
        act_cache = []


        def forward_cache_hook(act, hook):
            cache = {}
            if new_act is not None:
                cache[hook.name] = new_act.detach()
                act_cache.append(cache)
                return new_act 
            # cache[hook.name] = act.detach()
            cache[hook.name] = act.detach()
            act_cache.append(cache)

        self.base_model.add_hook(self.transformer_lens_loc, forward_cache_hook, "fwd")

        
        grad_cache_ = []

        def backward_cache_hook(act, hook):
            grad_cache = {}
            # grad_cache[hook.name] = act.detach()
            grad_cache[hook.name] = act.detach()
            grad_cache_.append(grad_cache)

        self.base_model.add_hook(self.transformer_lens_loc, backward_cache_hook, "bwd")
        
        i = 0
        logits = self.base_model(tokens) 
        probs = torch.softmax(logits[:, -1, :], dim=-1)  
        next_token = torch.argmax(probs, dim=-1)
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=-1)
        act_cache = []

        #这里是输入数据形状是[bs, max_length]
        while i < self.pre_token:
            logits = self.base_model(tokens) 
            loss = self.compute_loss(logits, tokens)
            loss.backward()
            probs = torch.softmax(logits[:, -1, :], dim=-1)  
            # next_token = torch.multinomial(probs, 1, True).T
            next_token = torch.argmax(probs, dim=-1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=-1)

            i +=1
            del next_token, probs, loss
        torch.cuda.empty_cache()

        self.base_model.reset_hooks()
        return (
            ActivationCache(act_cache, self.base_model),
            ActivationCache(grad_cache_, self.base_model),
        )
        
    def compute_loss(self, logits, labels):
        # labels = labels.to(self.device)
        # Shift so that tokens < n predict n

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return loss
    def attribute_score(self, inputs, ans = None, labels = None):
        if labels is None:
            labels = inputs
        
        act, grad = self.get_latent_activation(inputs)

        # act = self.get_latent_activation(inputs, labels)
        if act is None:
            return None

        attributes = []
        latents = []
        for x, grad_x in zip(act, grad):
        # x = act[self.transformer_lens_loc]
        # grad_x = grad[self.transformer_lens_loc]
            activations = self.sae.encode(x[self.transformer_lens_loc])
            # latents.append(activations)
            w_dec = self.sae.W_dec.T
            w_dec = w_dec.to(torch.float32)
            attribute = torch.matmul(grad_x[self.transformer_lens_loc], w_dec) * activations
            attributes.append(attribute)

        return attributes


    def data_gen(self, path):
        with open(path, 'r') as file:
            total_lines = sum(1 for _ in file)

        with open(path, 'r') as file, open('/research/projects/trans_llm/Zeru_Shi/SAE-learner/train_data_1.jsonl', 'a') as out_file:
            for i, line in tqdm(enumerate(file), total=total_lines, desc="Reading file"):
                if i < 214:
                    continue
                try:
                    item = json.loads(line.strip())

                    if 'Solution 1' in item['solution']:
                        parts = item['solution'].split("Solution 1\n", 1)
                        if len(parts) > 1:
                            solution = parts[1].split("\n", 1)[0].strip()
                    else:
                        solution = item['solution']

                    if item['answer'] in solution:
                        solution = solution.replace(item['answer'], '?')

                    prompt = f"""
                    Please reason step by step, and put your final answer within \boxed{{}}
                    
                    Question: {item['problem']} 
                    
                    Solution: {solution}
                    """

                    tokens = self.base_model.to_tokens(prompt)
                    attributes = self.attribute_score(tokens)

                    concept_list = []
                    top_k_list = []

                    for attribute in attributes:
                        attribute = attribute[:, -2, :]

                        concept_labels = torch.nonzero(attribute[0], as_tuple=True)[0]

                        if len(concept_labels) > 15:
                            nonzero_value = attribute[0][concept_labels] 
                            sorted_indices = nonzero_value.argsort()
                            concept_labels = concept_labels[sorted_indices].tolist()[0:15]
                            top_k_count = 15
                        else:
                            concept_labels = concept_labels.tolist()
                            if isinstance(concept_labels, int):
                                top_k_count = 1
                                concept_labels = [concept_labels]
                            else:
                                top_k_count = len(concept_labels)

                        concept_list.append(concept_labels)
                        top_k_list.append(top_k_count)

                    item['concept_labels'] = concept_list
                    item['topk'] = top_k_list
                    item['q_len'] = item['api_metadata'][0]['prompt_tokens']

                    for key in removed_colume:
                        item.pop(key, None)

                    json.dump(item, out_file)
                    out_file.write("\n")

                except Exception as e:
                    print(f"Skipping item {i}")
                    continue 


    def forward(self, input_ids, labels = None):
        if labels is None:
            labels = input_ids
        attributes= self.attribute_score(input_ids)
        return attributes
    


classes = ['Algebra', 'Prealgebra', 'Counting & Probability', 'Precalculus', 'Geometry', 'Number Theory', 'Intermediate Algebra']

if __name__ == '__main__':

    model = TransformerLensSAE(31)
    model.data_gen('/research/projects/trans_llm/Zeru_Shi/SAE-learner/traindata.jsonl')
    print('Data Finish')





