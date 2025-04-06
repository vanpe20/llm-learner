import os
import hydra
import torch
import torch._dynamo
from accelerate import Accelerator
import wandb
import torch.distributed as dist
from utils import Logger, set_random_seed
from models.reasoning_concept import TransformerLensSAE
from dataset.data_loader import get_val_dataloader, get_train_dataloader
from omegaconf import OmegaConf
from models import get_base_llm, get_concept_extractor, get_train_model
from train_process import train_step
from trainer import trainer
import torch

torch.cuda.empty_cache()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


wandb.init(project="llama-3.1-learner")


@hydra.main(config_path='conf', config_name='config', version_base='1.3.2')
def main(cfg):
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_acc_steps,
        log_with = 'wandb',
        project_dir="./logs")

    accelerator.wait_for_everyone()

    num_gpus = dist.get_world_size()
    cfg.distributed = num_gpus > 1
    cfg.world_size = num_gpus

    set_random_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if cfg.use_torch_compile:
        torch._dynamo.config.cache_size_limit = cfg.dynamo_size_limit

    
    base_model, tokenizer= get_base_llm(cfg, accelerator)

    concept_extractor = get_concept_extractor(cfg, accelerator, base_model, tokenizer)
    
    train_model = get_train_model(cfg, base_model, concept_extractor)
    


    train_loader = get_train_dataloader(cfg, tokenizer, accelerator)
    # train_loader = None

    val_loader = None
    # val_loader = get_val_dataloader(cfg, tokenizer, accelerator)

    print('Dataset create successfully!!!!!!!!')
    fname = f'{cfg.model_name}_{cfg.seed}'
    wandb_name = f'{cfg.model_name}_{cfg.seed}'

    logger = Logger(
        fname,
        cfg,
        main_process=accelerator.is_main_process,
        use_wandb=cfg.wandb_log,
        wandb_name=wandb_name,
        log_path=cfg.log_path,
    )
    logger.log(OmegaConf.to_yaml(cfg))
    
    trainer(
        cfg,
        train_step,
        base_model,
        train_model,
        train_loader,
        val_loader,
        logger,
        accelerator,
        concept_extractor,
    )

    logger.close_writer()



if __name__ == '__main__':
    main()
