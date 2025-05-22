import torch
import math
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    # Qwen2Attention as _Qwen2Attention,
    # Qwen2MLP as _Qwen2MLP,
    Qwen2DecoderLayer as _Qwen2DecoderLayer,
    Qwen2Model,
    # Qwen2ForCausalLM as _Qwen2ForCausalLM,
    Qwen2ForCausalLM,
    Qwen2Config
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config as QwenConfig

# @torch.jit.script
def qwen2_loss_func(input, target):
    lm_logits, labels = input, target
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

class Qwen2Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_embeddings=config.vocab_size
        # config.vocab_size = 151936
        print(f"[DEBUG] config.pad_token_id: {config.pad_token_id}")
        print(f"[DEBUG] config.vocab_size: {config.vocab_size}")
        self.embed_tokens = nn.Embedding(config.vocab_size, self.embed_dim, config.pad_token_id)

        
    def forward(self, input_ids):
        device = input_ids.device
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        
        inputs_embeds = self.embed_tokens(input_ids)
        return inputs_embeds

class Qwen2DecoderLayer(_Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int, use_checkpoint=True):
        super().__init__(config, layer_idx)
        self.use_checkpoint = use_checkpoint
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

    def _attn_res(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x, _ = self.self_attn(
            hidden_states=x,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            position_embeddings=(cos, sin),
        )
        return x + residual

    def _mlp_res(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return x + residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]

        # seq_len = x.shape[1]
        # position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)  # shape: [1, seq_len]
        cos, sin = self.rotary_emb(x, position_ids)
        # cos, sin = self.rotary_emb(position_ids)  # 

        if self.use_checkpoint:
            x.requires_grad_(True)
            x = checkpoint(self._attn_res, x, cos, sin)
        else:
            x = self._attn_res(x, cos, sin)

        if self.use_checkpoint:
            x.requires_grad_(True)
            x = checkpoint(self._mlp_res, x)
        else:
            x = self._mlp_res(x)

        return x


class Qwen2LMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, x):
        x = self.norm(x)
        x = self.lm_head(x)
        return x
