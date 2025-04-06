import os
import random
import json

from accelerate import PartialState
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
import warnings

removed_colmue = ['problem', 'generations', 'answer', 'problem_type', 'question_type', 'problem_is_valid', 'solution_is_valid', 'source', 'synthetic', 'finish_reasons', 'api_metadata']
dataset_text_field = ['sae_inputs', 'input_ids']
val_column = ['problem', 'solution', 'answer', 'subject', 'level', 'unique_id']


def pack_examples(examples: dict[str, list[list]], seq_length: int) -> dict[str, list[list]]:

    # Join  all the values into a single list
    examples = {k: sum(v, []) for k, v in examples.items()}
    # Split the values into chunks of size seq_length
    examples = {k: [v[i : i + seq_length] for i in range(0, len(v), seq_length)] for k, v in examples.items()}
    # examples = {k: torch.tensor(v) for k, v in examples.items()}
    return examples


def get_max_token(path):

    max_answer = float('-inf')
    with open(path, 'r', encoding='utf-8') as input_file:
        for i, line in enumerate(input_file):
            sample = json.loads(line.strip())
            max_answer = max(max_answer, sample['q_len'])

    return max_answer



def get_train_dataloader(cfg, tokenizer,  accelerator):

    train_dataset = get_train_dataset(cfg, tokenizer)
    train_dataset.set_format(type='torch', columns=['sae_inputs', 'input_ids'])


    batch_size = cfg.global_batch_size // cfg.world_size

    cfg.n_epochs = (cfg.train_steps * cfg.global_batch_size * cfg.grad_acc_steps // len(train_dataset) + 1)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle= False,
        num_workers = cfg.num_works,
        batch_size = batch_size,
        pin_memory=True,
        persistent_workers = True
    )

    return accelerator.prepare(train_dataloader)


def get_train_dataset(cfg, tokenizer):
    dataset = load_dataset('json', data_files = cfg.data_path)

    column_names = dataset.column_names
    is_processed = 'sae_inputs' in column_names and 'input_ids' in column_names and 'question_len' in column_names

    map_kwargs = {}

    if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
        map_kwargs["num_proc"] = cfg.dataset_num_proc
    
    with PartialState().local_main_process_first():
        if is_processed:
            warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                    UserWarning,
                )
        else:
            if isinstance(dataset, Dataset): 
                map_kwargs["desc"] = f"Applying formatting function to dataset"
        
        if 'problem' in dataset["train"].column_names and 'solution' in dataset["train"].column_names and 'generations' in dataset["train"].column_names:
                
                def concat_prompt_completion(example):
                    train_input = example['problem'] + example['generations'] + tokenizer.eos_token
                    sae_input = example['problem'] + example['solution'] + tokenizer.eos_token
                    q_len = example['api_metadata'][0]['prompt_token']

                    return {'input_ids': train_input, 'sae_inputs': sae_input, 'q_len': q_len}
            
                dataset = dataset["train"].map(concat_prompt_completion, remove_columns = removed_colmue)


                if not is_processed:
                    if isinstance(dataset, Dataset): 
                        map_kwargs["desc"] = "Tokenizing dataset"
                    
                    def tokenize(example):
                        
                        input_token = tokenizer(example['input_ids'])['input_ids']
                        sae_token = tokenizer(example['sae_inputs'])['input_ids']

                        # pre_token = token[:example['q_len'] + cfg.pre_token] + tokenizer.eos_token * (max_len - example['q_len'] + cfg.pre_token)
                        
                        return {'input_ids': input_token, 'sae_inputs': sae_token}
                    
                    dataset = dataset.map(
                            tokenize,
                            **map_kwargs,
                        )
        if cfg.pack:

            # def align_length(example):
            #     if isinstance(example['sae_inputs'], list):
            #         train_input = example['sae_inputs']
            #     else:
            #         train_input = example['input_ids'].tolist()

            #     question_length = example['q_len']

            #     max_length = max(len(sae_input), len(train_input))
            #     question_tensor = torch.tensor([question_length] * max_length, dtype=torch.long)
                
            #     pad_token_id = tokenizer.pad_token_id 
            #     sae_input += [pad_token_id] * (max_length - len(sae_input))
            #     train_input += [pad_token_id] * (max_length - len(train_input))

            #     # sae_input = torch.tensor(sae_input)
            #     # train_input = torch.tensor(train_input)

                
            #     return {'sae_inputs': sae_input, 'input_ids': train_input, 'q_len': question_tensor}

            # dataset = dataset.map(align_length)

            if cfg.max_seq_length is None:
                raise ValueError("When packing is enabled, `max_seq_length` can't be `None`.")
            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = "Packing dataset"

            dataset = dataset.select_columns(["input_ids"])
            dataset = dataset.map(
                    pack_examples, batched=True, fn_kwargs={"seq_length": cfg.max_seq_length}, **map_kwargs
                )
        elif cfg.max_seq_length is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Truncating dataset"

                def truncate(example, max_seq_length):
                    return {key: example[key][:max_seq_length] for key in ["input_ids", "attention_mask"]}

                dataset = dataset.map(
                    truncate,
                    fn_kwargs={"max_seq_length": cfg.max_seq_length},
                    **map_kwargs,
                )
    return dataset




def get_val_dataset(cfg, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset('json', data_files = cfg.val_path)

    map_kwargs = {}

    if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
        map_kwargs["num_proc"] = cfg.dataset_num_proc
    
    column_names = dataset.column_names
    is_processed = 'input' in column_names and 'answer' in column_names
    

    with PartialState().local_main_process_first():

        if is_processed:
            warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                    UserWarning,
                )
        else:
            if isinstance(dataset, Dataset): 
                map_kwargs["desc"] = f"Applying formatting function to dataset"
            
            def concat_prompt_completion(example):
                prompt = 'Please reason step by step, and put your final answer within \\boxed{{}}. ' + example['problem']
                answer = example['answer']

                return {'input_ids': prompt, 'answer': answer}
        
            dataset = dataset["train"].map(concat_prompt_completion, remove_columns = val_column)


            # max_length = max(len(item) for item in dataset["input_ids"])

            if cfg.pack:
                map_kwargs["desc"] = f"Format the data"
                def tokenize(example):
                    inputs = example["input_ids"]

                    padded_inputs = tokenizer(
                        inputs,
                        truncation=True,  
                        return_tensors="pt"  
                    )
                    answers = example["answer"]

                    return {"input_ids": padded_inputs["input_ids"][0],  "answer": answers}

                dataset = dataset.map(tokenize)
                max_length = max(len(item) for item in dataset["input_ids"])

                def pad_to_max_length(example):
                    pad_token_id = tokenizer.pad_token_id 
                    inputs = example['input_ids'] + [pad_token_id] * (max_length - len(example['input_ids']))


                    return {"input_ids": inputs}

                dataset = dataset.map(pad_to_max_length)
        
            return dataset


def get_val_dataloader(cfg, tokenizer, accelerator):
    val_dataset = get_val_dataset(cfg, tokenizer)
    val_dataset.set_format(type='torch', columns=['input_ids'])

    batch_size = cfg.batch_size_eval // cfg.world_size
    

    val_dataloader = DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=batch_size,  
        pin_memory=True
    )

    return val_dataloader










