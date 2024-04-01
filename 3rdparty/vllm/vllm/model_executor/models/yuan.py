# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, List, Optional, Tuple
import copy
import torch
from torch import nn
from .configuration_yuan import YuanConfig
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm,LlamaRotaryEmbedding
from vllm.model_executor.input_metadata import InputMetadata
#from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
#from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput
from vllm._C import ops
from vllm._C import cache_ops

KVCache = Tuple[torch.Tensor, torch.Tensor]
LFCache = Tuple[torch.Tensor, torch.Tensor]

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class YuanRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

class LocalizedFiltering(torch.nn.Module):
    """
    Mega's Exponential Moving Average layer, largely left unmodified from the original repo with the exception of
    variable names and moving away from the stateful representation of incremental decoding state. See
    "https://arxiv.org/abs/2209.10655" for more details.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.embed_dim = hidden_size
        self.lf_conv2d_group = 1
        self.lf_conv2d_num_pad = 0
        self.conv1 = torch.nn.Conv2d(self.embed_dim, self.embed_dim // 2, (2, 1), stride=(1, 1), padding=(self.lf_conv2d_num_pad, 0), groups=self.lf_conv2d_group)
        self.conv2 = torch.nn.Conv2d(self.embed_dim // 2, self.embed_dim, (2, 1), stride=(1, 1), padding=(self.lf_conv2d_num_pad, 0), groups=self.lf_conv2d_group)
        #self.output_layernorm = RMSNorm(self.embed_dim)
        self.output_layernorm = LlamaRMSNorm(self.embed_dim)

    def forward(self, inputs, lf1_cache, lf2_cache):
       inputs = inputs.permute([1, 0, 2])
       residual = inputs
       old_shape = inputs.view(inputs.shape[0], inputs.shape[1], inputs.shape[2]).shape
       new_shape = inputs.view(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2]).shape

       inputs = inputs.view(new_shape).permute([2, 3, 0, 1])
       inputs = torch.cat([lf1_cache, inputs], dim=2)
       output1 = self.conv1(inputs)
       output1 = torch.cat([lf2_cache, output1], dim=2)
       output2 = self.conv2(output1).permute([2, 3, 0, 1])
       output2 = output2.view(old_shape)
       
       assert list(output2.shape) == list(residual.shape), f'{output2.shape}, {residual.shape}'
       output3 = output2 + residual
       lf_output = self.output_layernorm(output3)
       lf_output = lf_output.permute([1, 0, 2])

       lf1_cache = inputs[:, :, -1:, :].contiguous()
       lf2_cache = output1[:, :, -1:, :].contiguous()
       return lf_output, lf1_cache, lf2_cache


class YuanMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.up_proj = ColumnParallelLinear(hidden_size,
                                            intermediate_size,
                                            bias=False,
                                            linear_method=linear_method)
        self.gate_proj= ColumnParallelLinear(hidden_size,
                                            intermediate_size,
                                            bias=False,
                                            linear_method=linear_method)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           linear_method=linear_method)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        x1, _ = self.up_proj(x)
        x3 = self.act_fn(x1)
        x2, _ = self.gate_proj(x)
        x, _ = self.down_proj(x2 * x3)
        return x


class YuanAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rope_theta: float = 10000,
        num_kv_heads=None,
        head_size=None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        self.head_size = hidden_size // num_heads if head_size is None else head_size
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.eps = 1e-6
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            linear_method=linear_method,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            linear_method=linear_method,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )
        
        self.lf_gate = LocalizedFiltering(self.hidden_size)
        self.rotary_emb = YuanRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        
        self.attn = PagedAttention(self.num_heads,
                                   self.head_dim,
                                   self.scaling,
                                   num_kv_heads=self.num_kv_heads)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        lf_cache: LFCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        lf1_cache, lf2_cache = lf_cache
        k_cache, v_cache = kv_cache
        v, _ = self.v_proj(hidden_states)
        v = v.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        if input_metadata.is_prompt:
            lf1_cache_shape = (bsz, self.total_num_kv_heads * self.head_dim, 1, 1)
            lf2_cache_shape = (bsz, self.total_num_kv_heads * self.head_dim // 2, 1, 1)
            lf1 = torch.zeros(lf1_cache_shape, dtype=torch.bfloat16, device='cuda')
            lf2 = torch.zeros(lf2_cache_shape, dtype=torch.bfloat16, device='cuda')
            hidden_states, lf1, lf2 = self.lf_gate(hidden_states, lf1, lf2)
            # print('='*80)
            # print(lf1.stride(0), lf2.stride(0), lf1.shape, lf2.shape)
            # print(lf1)
        else:
            lf1 = lf1_cache[:bsz, :, :, :]
            lf2 = lf2_cache[:bsz, :, :, :]
            hidden_states, lf1, lf2 = self.lf_gate(hidden_states, lf1, lf2)
        if lf1_cache is not None and lf2_cache is not None:
            cache_ops.lf_reshape_and_cache(
                lf1,
                lf2,
                lf1_cache,
                lf2_cache,
                input_metadata.slot_mapping.flatten(),
            )

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        qk = torch.cat([q, k], dim=-1)
        qk = qk.view(qk.shape[0], qk.shape[1], self.num_heads, int(qk.shape[-1] // self.num_heads))
        (q, k) = torch.chunk(qk, 2, dim=-1)
        q = q.permute([0, 2, 1, 3])
        k = k.permute([0, 2, 1, 3])
        
        cos, sin = self.rotary_emb(v, seq_len=1000)
        q, k = apply_rotary_pos_emb(q, k , cos, sin, positions)
        q = q.transpose(1, 2).contiguous().view(bsz, -1, self.num_heads * self.head_dim)
        k = k.transpose(1, 2).contiguous().view(bsz, -1, self.num_heads * self.head_dim)
        v = v.transpose(1, 2).contiguous().view(bsz, -1, self.num_heads * self.head_dim)
        
        
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class YuanDecoderLayer(nn.Module):

    def __init__(
        self,
        config: YuanConfig,
        #layer_id,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        #self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.self_attn = YuanAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            linear_method=linear_method,
            num_kv_heads=config.num_attention_heads,
        )
        self.mlp = YuanMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
                linear_method=linear_method,
        )
        # self.input_layernorm = RMSNorm(config.hidden_size,
        #                                eps=config.rms_norm_eps)
        # self.post_attention_layernorm = RMSNorm(config.hidden_size,
        #                                         eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        lf_cache: LFCache,
        input_metadata: InputMetadata,
        # residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # if residual is None:
        #     residual = hidden_states
        #     hidden_states = self.input_layernorm(hidden_states)
        # else: 
        #     hidden_states, residual = self.input_layernorm(
        #         hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            lf_cache=lf_cache,
            input_metadata=input_metadata,
        )
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        #hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states #, residual


class YuanModel(nn.Module):

    def __init__(
        self,
        config: YuanConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            YuanDecoderLayer(config, linear_method)
            for _ in range(config.num_hidden_layers)
        ])
        #self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        lf_caches: List[LFCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        #for i, (layer, lf1_past, lf2_past) in enumerate(zip(self.layers, lf1_cache_params.past_lf, lf2_cache_params.past_lf,)):
        for i, layer in enumerate(self.layers):
            # if input_metadata.is_prompt:
            #     k_cache, v_cache = kv_caches[i]
            #     lf1_cache, lf2_cache = lf_caches[i]
            # hidden_states, residual = layer(
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                lf_caches[i],
                input_metadata,
                # residual,
            )
        #hidden_states, _ = self.norm(hidden_states, residual)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class YuanForCausalLM(nn.Module):

    def __init__(
        self,
        config: YuanConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = YuanModel(config, linear_method)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        lf_caches: List[LFCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches, lf_caches, input_metadata)
        return hidden_states 

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        '''stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("q_proj", "q_proj", "q"),
            ("k_proj", "k_proj", "k"),
            ("v_proj", "v_proj", "v"),
            ("up_proj", "up_proj", 0),
            ("gate_proj", "gate_proj", 1),
            ("down_proj", "down_proj", 2),
            ("lf_gate", "lf_gate", "lf"),
        ]'''
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            '''for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:'''
                # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)

