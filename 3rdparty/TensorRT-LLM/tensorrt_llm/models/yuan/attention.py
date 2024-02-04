# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import List, Optional

import numpy as np
import tensorrt as trt

from ..._common import default_net, precision
from ..._utils import numpy_fp32_to_bf16, trt_dtype_to_np, int32_array, fp32_array
from ...functional import (AttentionMaskType, PositionEmbeddingType, chunk, transpose, arange,
                          RotaryScalingType, Tensor, bert_attention, cast, clip,index_select, constant, select,
                          concat, constant, embedding, expand_dims, expand_mask,
                          generate_alibi_biases, generate_alibi_slopes,
                          gpt_attention, matmul, repeat_interleave, round,
                          shape, slice, softmax, split, unsqueeze, view, where)
from ...module import Module
from ...parameter import Parameter
from ...quantization import QuantMode
from ...quantization.functional import dequantize, quantize
from ...quantization.layers import FP8Linear, FP8RowLinear
from ...layers.linear import ColumnLinear, RowLinear
from ...layers.lora import Lora, LoraRuntimeParams
from ...layers.conv import Conv2d
from ...layers.normalization import RmsNorm 
import pdb
class RopeEmbeddingUtils:

    @staticmethod
    def create_sinusoidal_positions(num_pos: int,
                                    dim: int,
                                    theta: float = 10000.0,
                                    dtype=np.float32):
        inv_freq = 1.0 / (theta**(np.arange(0, dim, 2) / dim)).astype(dtype)
        sinusoid_inp = np.einsum("i , j -> i j",
                                 np.arange(num_pos, dtype=dtype),
                                 inv_freq,
                                 dtype=dtype)
        concat = np.concatenate((np.sin(sinusoid_inp), np.cos(sinusoid_inp)),
                                axis=1)
        return np.expand_dims(concat, axis=0).astype(np.float32)
    
    @staticmethod
    def create_yuan_sinusoidal_positions(num_pos: int,
                                    dim: int,
                                    base: float = 10000.0,
                                    dtype=np.float32):
        inv_freq = 1.0 / (base**(np.arange(0, dim, 2) / dim)).astype(dtype)
        t = np.arange(num_pos, dtype=dtype)
        freqs = np.einsum("i , j -> i j", t, inv_freq, dtype=dtype)
        emb = np.concatenate((freqs, freqs), axis=-1)
        #cos = np.expand_dims(np.cos(emb), axis=0).astype(np.float32)
        #sin = np.expand_dims(np.sin(emb), axis=0).astype(np.float32)
        
        cos = np.cos(emb)
        sin = np.sin(emb)
        return cos, sin

    @staticmethod
    def rotate_every_two(tensor: Tensor) -> Tensor:
        assert tensor.ndim() == 4

        shape_tensor = concat([
            shape(tensor, i) / 2 if i == (tensor.ndim() -
                                          1) else shape(tensor, i)
            for i in range(tensor.ndim())
        ])
        x1 = slice(tensor, [0, 0, 0, 0], shape_tensor, [1, 1, 1, 2])
        x2 = slice(tensor, [0, 0, 0, 1], shape_tensor, [1, 1, 1, 2])
        x1 = expand_dims(x1, 4)
        x2 = expand_dims(x2, 4)
        zero = constant(
            np.ascontiguousarray(np.zeros([1],
                                          dtype=trt_dtype_to_np(x2.dtype))))
        x2 = zero - x2
        x = concat([x2, x1], 4)
        return view(
            x, concat([shape(x, 0),
                       shape(x, 1),
                       shape(x, 2),
                       shape(x, 3) * 2]))

    @staticmethod
    def rotate_half(tensor: Tensor) -> Tensor:
        # [bs, num_attention_kv_heads, seqlen, attention_head_size]
        assert tensor.ndim() == 4
        shape_tensor = concat([
            shape(tensor, i) / 2 if i == (tensor.ndim() -
                                          1) else shape(tensor, i)
            for i in range(tensor.ndim())
        ])
        last_dim = shape(tensor, tensor.ndim() - 1) / 2
        x1 = slice(tensor, [0, 0, 0, 0], shape_tensor, [1, 1, 1, 1])
        x2 = slice(tensor, concat([0, 0, 0, last_dim]), shape_tensor,
                   [1, 1, 1, 1])
        zero = constant(
            np.ascontiguousarray(np.zeros([1],
                                          dtype=trt_dtype_to_np(x2.dtype))))
        x2 = zero - x2
        x = concat([x2, x1], 3)
        return x

    @staticmethod
    def rotate_yuan_half(tensor: Tensor) -> Tensor:
        # [bs, num_attention_kv_heads, seqlen, attention_head_size]
        assert tensor.ndim() == 4
        shape_tensor = concat([
            shape(tensor, i) / 2 if i == (tensor.ndim() -
                                          1) else shape(tensor, i)
            for i in range(tensor.ndim())
        ])
        last_dim = shape(tensor, tensor.ndim() - 1) / 2
        x1 = slice(tensor, [0, 0, 0, 0], shape_tensor, [1, 1, 1, 1])
        x2 = slice(tensor, concat([0, 0, 0, last_dim]), shape_tensor,
                   [1, 1, 1, 1])
        zero = constant(
            np.ascontiguousarray(np.zeros([1],
                                          dtype=trt_dtype_to_np(x2.dtype))))
        x2 = zero - x2
        x = concat([x2, x1], tensor.ndim() - 1)
        return x
    
    @staticmethod
    def apply_rotary_pos_emb(
        tensor: Tensor,
        position_embedding: List[Tensor] = None,
        pos_emb_type: PositionEmbeddingType = PositionEmbeddingType.rope_gptj
    ) -> Tensor:

        rotate_func = None
        if pos_emb_type == PositionEmbeddingType.rope_gpt_neox:
            assert len(position_embedding) == 2
            cos, sin = position_embedding
            sin = expand_dims(sin, 2)
            cos = expand_dims(cos, 2)
            sin = concat([sin, sin], 3)
            cos = concat([cos, cos], 3)
            rotate_func = RopeEmbeddingUtils.rotate_half
        elif pos_emb_type == PositionEmbeddingType.rope_gptj:
            assert len(position_embedding) == 2
            cos, sin = position_embedding
            sin = expand_dims(sin, 2)
            cos = expand_dims(cos, 2)
            sin = repeat_interleave(sin, 2, 3)
            cos = repeat_interleave(cos, 2, 3)
            rotate_func = RopeEmbeddingUtils.rotate_every_two
        elif pos_emb_type == PositionEmbeddingType.chatglm:
            assert len(position_embedding) == 4
            cos0, cos1, sin0, sin1 = position_embedding
            if default_net().strongly_typed and tensor.dtype != cos0.dtype:
                tensor = cast(tensor, cos0.dtype)
            shape_tensor = concat([
                shape(tensor, i) / 2 if i == (tensor.ndim() -
                                              1) else shape(tensor, i)
                for i in range(tensor.ndim())
            ])
            last_dim = shape(tensor, tensor.ndim() - 1) / 2
            x_part0 = slice(tensor, [0, 0, 0, 0], shape_tensor, [1, 1, 1, 1])
            x_part1 = slice(tensor, concat([0, 0, 0, last_dim]), shape_tensor,
                            [1, 1, 1, 1])

            y_part0 = (x_part0 *
                       cos0) + (RopeEmbeddingUtils.rotate_half(x_part0) * sin0)
            y_part1 = (x_part1 *
                       cos1) + (RopeEmbeddingUtils.rotate_half(x_part1) * sin1)

            result = concat([y_part0, y_part1], dim=3)
            return result.view(shape(tensor))

        else:
            raise ValueError('The PositionEmbeddingType is not RoPE')
        return (tensor * cos) + (rotate_func(tensor) * sin)
    
    @staticmethod
    def apply_rotary_pos_emb_yuan(
        q: Tensor,
        k: Tensor,
        position_ids: Tensor,
        max_position_embeddings,
        num_attention_heads,
        attention_head_size
    ) -> Tensor:
        cos, sin = RopeEmbeddingUtils.create_yuan_sinusoidal_positions(max_position_embeddings, attention_head_size)
        cos = constant(cos)
        sin = constant(sin)
        cos = embedding(position_ids, cos)
        sin = embedding(position_ids, sin)

        cos_sin_shape = concat([
            shape(cos, 0),
            1,
            shape(cos, 1),
            attention_head_size,
        ])
        cos = cos.view(cos_sin_shape)
        sin = sin.view(cos_sin_shape)
        q_embed = (q * cos) + (RopeEmbeddingUtils.rotate_half(q) * sin)
        k_embed = (k * cos) + (RopeEmbeddingUtils.rotate_half(k) * sin)
        new_shape = concat([
            shape(q_embed, 0), 
            shape(q_embed, 2), 
            shape(q_embed, 1) * shape(q_embed, 3)
        ])
        output_q_embed = q_embed.permute([0, 2, 1, 3])
        output_k_embed = k_embed.permute([0, 2, 1, 3])
        output_q_embed = output_q_embed.view(new_shape)
        output_k_embed = output_k_embed.view(new_shape)
        return output_q_embed, output_k_embed, cos, sin, q_embed, k_embed

    @staticmethod
    def apply_rotary_pos_emb_chatglm(qkv, position_embedding,
                                     num_attention_heads, attention_head_size,
                                     max_position_embeddings,
                                     rotary_embedding_scale,
                                     remove_input_padding) -> Tensor:

        half_head_size = attention_head_size // 2
        input = qkv[0] if isinstance(qkv, list) else qkv
        input_shape = shape(input)
        batch_size = 1 if remove_input_padding else shape(input, 0)
        seqlen = shape(input, 0 if remove_input_padding else 1)
        if isinstance(qkv, list):
            query, key, value = qkv
        else:
            qkv = qkv.view(
                concat([
                    batch_size,
                    seqlen,
                    num_attention_heads,
                    3,
                    attention_head_size,
                ]))
            query, key, value = split(qkv, 1, dim=3)
        q_shape = concat([
            batch_size,
            seqlen,
            num_attention_heads,
            attention_head_size,
        ])
        query = query.view(q_shape)
        key = key.view(q_shape)
        value = value.view(q_shape)

        embedding_weight = RopeEmbeddingUtils.create_sinusoidal_positions(
            max_position_embeddings, half_head_size)
        embedding_weight /= rotary_embedding_scale
        embedding_weight = np.split(embedding_weight.squeeze(0), 2, axis=1)
        embedding_weight = np.concatenate(
            [
                embedding_weight[0],
                embedding_weight[0],
                embedding_weight[1],
                embedding_weight[1],
            ],
            axis=1,
        )

        if remove_input_padding:
            position_embedding = unsqueeze(position_embedding, 0)

        embedding_weight = constant(embedding_weight)
        position_embedding = embedding(position_embedding, embedding_weight)
        position_embedding, block_embedding = split(
            position_embedding,
            1,
            dim=1,
        )
        sin0, cos0 = split(position_embedding, half_head_size, dim=3)
        sin1, cos1 = split(block_embedding, half_head_size, dim=3)

        new_shape = concat([
            batch_size,
            seqlen,
            1,
            half_head_size,
        ])
        position_embedding = [
            tensor.view(new_shape) for tensor in [cos0, cos1, sin0, sin1]
        ]

        query = RopeEmbeddingUtils.apply_rotary_pos_emb(
            tensor=query,
            position_embedding=position_embedding,
            pos_emb_type=PositionEmbeddingType.chatglm)
        key = RopeEmbeddingUtils.apply_rotary_pos_emb(
            tensor=key,
            position_embedding=position_embedding,
            pos_emb_type=PositionEmbeddingType.chatglm)

        if default_net().strongly_typed:
            if query.dtype != value.dtype:
                query = cast(query, value.dtype)
            if key.dtype != value.dtype:
                key = cast(key, value.dtype)

        if isinstance(qkv, list):
            qkv = [
                query.view(input_shape),
                key.view(input_shape),
                value.view(input_shape),
            ]
        else:
            qkv = concat([query, key, value], dim=2)
            qkv = qkv.view(input_shape)

        return qkv

class KeyValueCacheParams:

    def __init__(self,
                 past_key_value: List[Tensor] = None,
                 host_past_key_value_lengths: Tensor = None,
                 host_max_attention_window_sizes: List[Tensor] = None,
                 kv_cache_block_pointers: List[Tensor] = None,
                 host_kv_cache_block_pointers: List[Tensor] = None,
                 cache_indirection: Tensor = None,
                 past_key_value_length: Tensor = None):
        self.past_key_value = past_key_value
        self.host_past_key_value_lengths = host_past_key_value_lengths
        self.host_max_attention_window_sizes = host_max_attention_window_sizes
        self.kv_cache_block_pointers = kv_cache_block_pointers
        self.host_kv_cache_block_pointers = host_kv_cache_block_pointers
        self.cache_indirection = cache_indirection
        # self.past_key_value_length = past_key_value_length

    def get_first_past_key_value(self):
        if self.past_key_value is None:
            return None
        return self.past_key_value[0]

    def get_first_kv_cache_block_pointers(self):
        if self.kv_cache_block_pointers is None:
            return None
        return self.kv_cache_block_pointers[0]

    def get_first_host_kv_cache_block_pointers(self):
        if self.host_kv_cache_block_pointers is None:
            return None
        return self.host_kv_cache_block_pointers[0]

    def fill_none_tensor_list(self, list_size):
        if self.past_key_value is None:
            self.past_key_value = tuple([None] * list_size)
        if self.host_max_attention_window_sizes is None:
            self.host_max_attention_window_sizes = tuple([None] * list_size)

    def is_valid(self, gpt_attention_plugin):
        if gpt_attention_plugin:
            if self.host_past_key_value_lengths is None:
                return False
            if self.host_max_attention_window_sizes is None:
                return False
            if self.cache_indirection is None:
                return False

        return True

class LFCacheParams:

    def __init__(self,
            past_lf: List[Tensor] = None):
        self.past_lf = past_lf
    def get_first_past_lf(self):
        return self.past_lf[0]

class LocalizedFiltering(Module):
    """
    Mega's Exponential Moving Average layer, largely left unmodified from the original repo with the exception of
    variable names and moving away from the stateful representation of incremental decoding state. See
    "https://arxiv.org/abs/2209.10655" for more details.
    """

    def __init__(self, hidden_size, rms_norm_eps, dtype):
        super().__init__()
        self.embed_dim = hidden_size
        self.lf_conv2d_group = 1
        self.lf_conv2d_num_pad = 0
        self.dtype = dtype
        self.conv1 = Conv2d(self.embed_dim, self.embed_dim // 2, (2, 1), stride=(1, 1), padding=(self.lf_conv2d_num_pad, 0), groups=self.lf_conv2d_group)
        self.conv2 = Conv2d(self.embed_dim // 2, self.embed_dim, (2, 1), stride=(1, 1), padding=(self.lf_conv2d_num_pad, 0), groups=self.lf_conv2d_group)
        self.output_layernorm = RmsNorm(self.embed_dim, rms_norm_eps, self.dtype)

    def forward(self, inputs, past_lf1, past_lf2):
        inputs = inputs.permute([1, 0, 2])
        residual = inputs
        old_shape = concat([
                shape(inputs, 0),
                shape(inputs, 1),
                shape(inputs, 2),
            ])

        new_shape = concat([
                shape(inputs, 0),
                1,
                shape(inputs, 1),
                shape(inputs, 2),
            ])
        inputs = inputs.view(new_shape).permute([2, 3, 0, 1])
        inputs = concat([past_lf1, inputs], dim=2)
        output1 = self.conv1(inputs, 1)
        output1 = concat([past_lf2, output1], dim=2)
        output2 = self.conv2(output1, 1).permute([2,3,0,1])
        output2 = output2.view(old_shape)
        assert list(output2.shape) == list(residual.shape), f'{output2.shape}, {residual.shape}'
        output3 = output2 + residual
        lf_output = self.output_layernorm(output3)
        lf_output = lf_output.permute([1, 0, 2])
        return lf_output, inputs, output1

class YuanAttention(Module):

    def __init__(self,
        hidden_size,
        num_attention_heads,
        num_kv_heads=None,
        max_position_embeddings=1024,
        num_layers=1,
        apply_query_key_layer_scaling=False,
        attention_head_size=None,
        attention_mask_type=AttentionMaskType.padding,
        bias=False,
        dtype=None,
        rms_norm_eps=1e-6,
        position_embedding_type=PositionEmbeddingType.learned_absolute,
        rotary_embedding_base=10000.0,
        rotary_embedding_scaling=None,
        rotary_embedding_percentage=1.0,
        tp_group=None,
        tp_size=1,
        tp_rank=0,
        quant_mode: QuantMode = QuantMode(0),
        q_scaling=1.0,
        cross_attention=False,
        relative_attention=False,
        max_distance=0,
        num_buckets=0,
        instance_id: int = 0,
        dense_bias=None):
        super().__init__()

        self.cross_attention = cross_attention
        self.attention_mask_type = attention_mask_type
        self.attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
        assert num_attention_heads % tp_size == 0, "num_attention_heads must be divisible by tp_size"
        self.num_attention_heads = num_attention_heads // tp_size
        self.num_attention_kv_heads = (num_kv_heads + tp_size - 1) // tp_size if num_kv_heads is not None else self.num_attention_heads
        self.hidden_size = hidden_size // tp_size
        self.max_position_embeddings = max_position_embeddings
        self.bias = bias
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.dtype = dtype
        if dense_bias is None:
            dense_bias = bias
        self.num_layers = num_layers
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.q_scaling = q_scaling
        if self.apply_query_key_layer_scaling:

            self.norm_factor *= self.num_layers
            self.q_scaling *= self.num_layers
        # Whether to scale ALiBi bias. Mathematically, it's equivalent to
        # normalizing QK after adding bias.
        #   - False, inv_sqrt_Dh * Q*K^T + alibi_bias
        #   - True,  inv_sqrt_Dh * Q*K^T + inv_sqrt_Dh * alibi_bias
        self.scale_alibi_bias = position_embedding_type == PositionEmbeddingType.alibi_with_scale
        self.position_embedding_type = position_embedding_type
        self.relative_attention = relative_attention
        self.max_distance = max_distance
        self.rotary_embedding_base = rotary_embedding_base
        self.rotary_embedding_scale_type = RotaryScalingType.none
        self.rotary_embedding_scale = 1.0
        if rotary_embedding_scaling is not None:
            assert rotary_embedding_scaling["type"] in ["linear", "dynamic"]
            self.rotary_embedding_scale_type = RotaryScalingType.linear if rotary_embedding_scaling[
                "type"] == "linear" else RotaryScalingType.dynamic
            self.rotary_embedding_scale = rotary_embedding_scaling["factor"]
            assert self.rotary_embedding_scale > 1.0
        self.embed_positions = None
        self.rotary_enabled = False
        self.rotary_embedding_dim = 0
        if self.position_embedding_type.is_rope():
            self.rotary_embedding_dim = int(self.attention_head_size *
                                            rotary_embedding_percentage)
            self.rotary_enabled = True
            self.embed_positions = RopeEmbeddingUtils.create_sinusoidal_positions(
                self.max_position_embeddings,
                self.rotary_embedding_dim,
            )

        self.quant_mode = quant_mode
        self.use_int8_kv_cache = self.quant_mode.has_int8_kv_cache()
        if self.quant_mode.has_kv_cache_quant():
            self.kv_orig_quant_scale = Parameter(shape=(1, ), dtype='float32')
            self.kv_quant_orig_scale = Parameter(shape=(1, ), dtype='float32')

        self.lf_gate = LocalizedFiltering(self.hidden_size * tp_size, rms_norm_eps, self.dtype)
        # The output feature size is therefore (h/tp + 2*kvh/tp) * d, where h is num_heads,
        # d is head_size, kvh is the num_kv_heads and tp is tensor_parallel_size.
        # In ColumnLinear op, the output dim is calculated by (h + 2*kvh) * d / tp,
        # which matches the desired output size (h/tp + 2*kvh/tp) * d after splitting
        self.use_fp8_qdq = self.quant_mode.has_fp8_qdq()
        if self.use_fp8_qdq:
            self.q = FP8Linear(
                hidden_size,
                tp_size * self.num_attention_heads * self.attention_head_size,
                bias=bias,
                dtype=dtype,
                tp_group=tp_group,
                tp_size=tp_size,
                gather_output=False)
            self.k = FP8Linear(
                hidden_size,
                (tp_size * self.num_attention_kv_heads * self.attention_head_size),
                bias=bias,
                dtype=dtype,
                tp_group=tp_group,
                tp_size=tp_size,
                gather_output=False)
            self.v = FP8Linear(
                hidden_size,
                (tp_size * self.num_attention_kv_heads * self.attention_head_size),
                bias=bias,
                dtype=dtype,
                tp_group=tp_group,
                tp_size=tp_size,
                gather_output=False)
            self.dense = FP8RowLinear(hidden_size,
                                      hidden_size,
                                      bias=dense_bias,
                                      dtype=dtype,
                                      tp_group=tp_group,
                                      tp_size=tp_size,
                                      instance_id=instance_id)
        else:
            # out dim is not necessarily hidden_size + kv specific size (in MQA/GQA), but num_heads * heads_size
            # example: d_model != num_heads * head_size in Flan-T5
            self.q = ColumnLinear(
                hidden_size,
                tp_size * self.num_attention_heads * self.attention_head_size,
                bias=bias,
                dtype=dtype,
                tp_group=tp_group,
                tp_size=tp_size,
                gather_output=False)
            self.k = ColumnLinear(
                hidden_size,
                (tp_size * self.num_attention_kv_heads * self.attention_head_size),
                bias=bias,
                dtype=dtype,
                tp_group=tp_group,
                tp_size=tp_size,
                gather_output=False)
            self.v = ColumnLinear(
                hidden_size,
                (tp_size * self.num_attention_kv_heads * self.attention_head_size),
                bias=bias,
                dtype=dtype,
                tp_group=tp_group,
                tp_size=tp_size,
                gather_output=False)
            self.dense = RowLinear(tp_size * self.num_attention_heads *
                                   self.attention_head_size,
                                   hidden_size,
                                   bias=dense_bias,
                                   dtype=dtype,
                                   tp_group=tp_group,
                                   tp_size=tp_size,
                                   instance_id=instance_id)

        # per-layer relative attention table
        if relative_attention:
            self.rel_attn_table = Parameter(shape=(num_attention_heads //
                                                   tp_size, num_buckets),
                                            dtype=dtype)

        # self.qkv_lora = Lora(
        #     in_hidden_size=hidden_size,
        #     out_hidden_sizes=[
        #         self.num_attention_heads * self.attention_head_size,
        #         self.num_attention_kv_heads * self.attention_head_size,
        #         self.num_attention_kv_heads * self.attention_head_size
        #     ],
        #     max_low_rank=min(
        #         hidden_size,
        #         self.num_attention_heads * self.attention_head_size,
        #         self.num_attention_kv_heads * self.attention_head_size),
        # )

    def forward(self,
                hidden_states: Tensor,
                position_ids=None,
                attention_mask=None,
                use_cache=False,
                lf1_cache_params=None,
                lf2_cache_params=None,
                kv_cache_params=None,
                attention_params=None,
                encoder_output: Optional[Tensor] = None,
                workspace=None,
                position_embedding=None,
                norm_before_bmm1=False,
                lora_layer_params=None):

        assert isinstance(hidden_states, Tensor)

        alibi_slopes = None
        # LFA
        past_lf1 = lf1_cache_params.get_first_past_lf()
        past_lf2 = lf2_cache_params.get_first_past_lf()

        q_lora_params = None
        k_lora_params = None
        v_lora_params = None
        # if lora_layer_params is not None:
        #     q_lora_params = lora_layer_params.get_runtime_params(0, "attn_q")
        #     k_lora_params = lora_layer_params.get_runtime_params(0, "attn_k")
        #     v_lora_params = lora_layer_params.get_runtime_params(0, "attn_v")
        value_states = self.v(hidden_states)#.view(concat([shape(hidden_states, 0), shape(hidden_states, 1), self.num_attention_kv_heads, self.attention_head_size])).permute([0, 2, 1, 3])
        hidden_states, past_lf1, past_lf2 = self.lf_gate(hidden_states, past_lf1, past_lf2)
        index = constant(int32_array([-1]))
        past_lf1 = index_select(past_lf1, 2, index)
        past_lf2 = index_select(past_lf2, 2, index)
        query_states = self.q(hidden_states)
        key_states = self.k(hidden_states)

        paged_kv_cache = default_net().plugin_config.paged_kv_cache

        qk_states = concat([query_states, key_states], dim=2)
        qk_states = qk_states.view(concat([shape(qk_states,0), shape(qk_states, 1), self.num_attention_heads, 2 * self.attention_head_size]))
        qk_states = chunk(qk_states, 2, dim=3)
        query_states = qk_states[0].permute([0, 2, 1, 3])
        key_states = qk_states[1].permute([0, 2, 1, 3])
        query_states,key_states, cos, sin, q_embed, k_embed = RopeEmbeddingUtils.apply_rotary_pos_emb_yuan(query_states, key_states, position_ids, self.max_position_embeddings, self.num_attention_heads, self.attention_head_size)

        past_key_value = None if kv_cache_params is None else kv_cache_params.get_first_past_key_value()
        cross_qkv = None

        assert paged_kv_cache == False
        def transpose_for_scores(x,
                                 rotary: bool = False,
                                 is_kv: bool = False):
            _num_attention_heads = self.num_attention_kv_heads if is_kv else self.num_attention_heads
            new_x_shape = concat([
                shape(x, 0),
                shape(x, 1), _num_attention_heads, self.attention_head_size
            ])
            # bs, nh, seq, hs
            return x.view(new_x_shape).permute([0, 2, 1, 3])

        kv_size = self.attention_head_size * self.num_attention_kv_heads

        query_states = transpose_for_scores(query_states)
        key_states = transpose_for_scores(key_states, is_kv=True)
        value_states = transpose_for_scores(value_states, is_kv=True)

        if past_key_value is not None:
            def dequantize_tensor(x, scale):
                # Cast from int8 to dtype
                casted_x = cast(x, self.dtype)
                return casted_x * scale

            if self.use_int8_kv_cache:
                past_key_value = dequantize_tensor(past_key_value, self.kv_quant_orig_scale.value)

            if self.use_fp8_qdq and self.quant_mode.has_kv_cache_quant():
                past_key_value = dequantize(past_key_value, self.kv_quant_orig_scale.value)

            # past_key_value [bs, 2, num_heads, max_seq_len, head_dim]
            past_key, past_value = split(past_key_value, 1, dim=1)

            key_shape = concat([
                shape(past_key, 0),
                shape(past_key, 2),
                shape(past_key, 3),
                shape(past_key, 4)
            ])
            past_key = past_key.view(key_shape, zero_is_placeholder=False)
            past_value = past_value.view(key_shape, zero_is_placeholder=False)

            key_states = concat([past_key, key_states], dim=2)#.cast(self.dtype)
            value_states = concat([past_value, value_states], dim=2)#.cast(self.dtype)
        
        key_inflated_shape = concat([
            shape(key_states, 0), 1,
            shape(key_states, 1),
            shape(key_states, 2),
            shape(key_states, 3)
        ])
        inflated_key = key_states.view(key_inflated_shape, zero_is_placeholder=False)
        inflated_value = value_states.view(key_inflated_shape, zero_is_placeholder=False)
        past_key_value = concat([inflated_key, inflated_value], dim=1).cast(self.dtype)

        if self.use_int8_kv_cache:

            def quantize_tensor(x, scale):
                scaled = x * scale
                rounded = round(scaled)
                clipped = clip(rounded, -128, 127)
                quantized = cast(clipped, 'int8')
                return quantized

            past_key_value = quantize_tensor(
                past_key_value, self.kv_orig_quant_scale.value)

        if self.use_fp8_qdq and self.quant_mode.has_kv_cache_quant():
            past_key_value = quantize(past_key_value,
                                      self.kv_orig_quant_scale.value,
                                      dtype='fp8')

        # MQA broadcast
        if self.num_attention_heads // self.num_attention_kv_heads > 1:
            key_states = repeat_interleave(key_states, self.num_attention_heads // self.num_attention_kv_heads, 1)
            value_states = repeat_interleave(value_states, self.num_attention_heads // self.num_attention_kv_heads, 1)
        key_length = shape(key_states, 2)

        query_length = shape(query_states, 2)
        #starts = concat([0, 0, 0, 0 ])  #key_length - query_length, 0])
        starts = concat([0, 0, key_length - query_length, 0])
        sizes = concat([1, 1, query_length, key_length])
        select_buf = np.expand_dims(
            np.tril(
                np.ones((self.max_position_embeddings,
                         self.max_position_embeddings))).astype(bool),
            (0, 1))

        select_buf = np.logical_not(select_buf)
        mask_buf = np.zeros_like(select_buf, np.float32)
        mask_buf[select_buf] = float('-inf')
        buffer = constant(mask_buf)
        generated_mask = slice(buffer, starts, sizes)

        key_states = key_states.permute([0, 1, 3, 2])

        with precision('float32'):
            attention_scores = matmul(cast(query_states, 'float32'), cast(key_states, 'float32'))
            if not norm_before_bmm1:
                attention_scores = attention_scores / self.norm_factor
            attention_scores = attention_scores + generated_mask

        attention_probs = softmax(attention_scores, dim=-1)
        if default_net().strongly_typed and (attention_probs.dtype != value_states.dtype):
            attention_probs = cast(attention_probs, value_states.dtype)

        context = matmul(attention_probs, value_states).permute([0, 2, 1, 3])
        context = context.view(concat([shape(context, 0), shape(context, 1), self.hidden_size]))

        dense_lora_params = None
        context = self.dense(context, workspace, lora_runtime_params=dense_lora_params)
        return (context, past_key_value, past_lf1, past_lf2)
