
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F



def LN(
    x: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std


class SAE_Model(nn.Module):
    def __init__(self, cfg, model, sae, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.sae = sae
        self.tokenizer = tokenizer
        self.base_model = model
        self.current_len = 0
        self.pos_index = 0

    def compute_loss(self, logits, labels = None):
        if labels is None:
            labels = logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return loss


    def forward(self, input_ids, question_length, top_k_count):    
        bs = input_ids.shape[0]
        stored_concept_logits = torch.zeros(self.cfg.pre_token, bs, self.cfg.act_dim, device=input_ids.device)
        if self.current_len != question_length[0]:
            self.current_len = question_length[0]

        def steering_hook(activations, hook):
            nonlocal stored_concept_logits
            # 只在生成的前几个token进行steering
            current_token = input_ids[:, self.pos_index % self.cfg.max_seq_length]

            if question_length[self.pos_index % self.cfg.max_seq_length] != self.current_len:
                self.pos_index = 0
                self.current_len = question_length[self.pos_index % self.cfg.max_seq_length]

            question_id = self.current_len

            shift_id = question_id + self.cfg.pre_token
            
            if current_token < question_id and  self.equal_before == 0:
                self.pos_index +=1
                return activations
            else:
                if self.pos_index < shift_id:
                    activation = activations[:, -1, :]

                    concept_logits = self.sae.encode(activation)
                    step_idx = self.pos_index - self.shift_id
                    stored_concept_logits[step_idx, :, :] = concept_logits

                    top_k = top_k_count[self.pos_index - self.shift_id].item()
      
                    topk_indices = torch.topk(concept_logits, k=top_k, dim=-1)[1]

                    # topk_indices_t = torch.stack(topk_indices_list, dim=0) 
                    mask = torch.zeros_like(concept_logits, dtype=torch.bool)
                    mask.scatter_(-1, topk_indices, True)

                    concept_logit_act = mask * concept_logits
                    continuous_concept = self.sae.decode(concept_logit_act)

                    sim = F.cosine_similarity(activation, continuous_concept.unsqueeze(0), dim=1)

                    scale = 160 * torch.sigmoid(sim)
                    activations[:, -1, :] += scale * (continuous_concept)

                    self.pos_index += 1
                    return activations
                else:
                    self.pos_index +=1
                    return activations


                
        self.base_model.reset_hooks()
        hook_point_name = f"blocks.{self.cfg.layer_num}.hook_resid_post"
        self.base_model.add_hook(hook_point_name, steering_hook)
        
        logits = self.base_model(input_ids)
        loss = self.compute_loss(logits)

        # 生成文本
        # outputs = self.base_model.generate(input_ids, max_new_tokens=self.cfg.max_token)

        return loss, stored_concept_logits
