import os

import torch
import transformer_lens
from models.reasoning_concept import TransformerLensSAE
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from models.sae_model import SAE_Model
from transformers.utils import is_liger_kernel_available, is_peft_available

if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM

def get_base_llm(cfg, accelerator):

    model_init_kwargs = {}
    # config = AutoConfig.from_pretrained(cfg.model_name)

    torch_dtype = cfg.torch_dtype
    if torch_dtype == 'auto':
        pass
    elif isinstance(torch_dtype, str):
        model_init_kwargs['dtype'] = torch_dtype
    else:
        raise ValueError(
                "Invalid `torch_dtype` passed to `SFTConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
    
    if cfg.gradient_checkpointing:
        model_init_kwargs["use_cache"] = False
    
    model_init_kwargs["attn_implementation"] = cfg.attn_implementation


    model = transformer_lens.HookedTransformer.from_pretrained(cfg.model_name, **model_init_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    model = accelerator.prepare(model)

    if cfg.use_torch_compile:
        model = torch.compile(model)

    # if cfg.use_liger:
    #     if not is_liger_kernel_available():
    #         raise ImportError("Please install Liger-kernel for use_liger=True")
    #     model = AutoLigerKernelForCausalLM.from_pretrained(cfg.model_name, **model_init_kwargs)
    # else:
        # model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_init_kwargs)
    

   

    return model, tokenizer

def get_concept_extractor(cfg, accelator, base_llm, tokenizer):

    concept_extractor = TransformerLensSAE(cfg.laryer_index,
                                        base_llm, 
                                        model_name=cfg.model_name, 
                                        tokenizer=tokenizer,
                                        release = cfg.sae_model, 
                                        max_length = cfg.max_length,
                                        pre_token=cfg.pre_token)

    # ddp_local_rank = int(os.environ["LOCAL_RANK"])
    # local_device = f"cuda:{ddp_local_rank}"
    # concept_extractor = concept_extractor.to(local_device)
    # concept_extractor.base_model = concept_extractor.base_model.to(local_device)
    # concept_extractor.autoencoder = concept_extractor.sae.to(local_device)

    # if cfg.use_torch_compile and concept_extractor is not None:
    #     concept_extractor = torch.compile(concept_extractor)

    return concept_extractor

def get_train_model(cfg, model, concept_extractor):

    base_model = SAE_Model(cfg, model, sae=concept_extractor.sae, tokenizer= concept_extractor.tokenizer)

    return base_model