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
import configparser
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from safetensors import safe_open

import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import (str_dtype_to_np, str_dtype_to_torch,
                                 torch_to_numpy)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import YuanForCausalLM
from tensorrt_llm.models.quantized.quant import get_dummy_quant_scales
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime.lora_manager import LoraConfig


def get_scaling_factors(
    model_path: Union[str, Path],
    num_layers: int,
    quant_mode: Optional[QuantMode] = None,
) -> Optional[Dict[str, List[int]]]:
    """ Get the scaling factors for LLaMA model

    Returns a dictionary of scaling factors for the selected layers of the
    LLaMA model.

    Args:
        model_path (str): Path to the quantized LLaMA model
        layers (list): List of layers to get the scaling factors for. If None,
            all layers are selected.

    Returns:
        dict: Dictionary of scaling factors for the selected layers of the
        LLaMA model.

        example:

        {
            'qkv_act': qkv_act_scale,
            'qkv_weights': qkv_weights_scale,
            'qkv_output' : qkv_outputs_scale,
            'dense_act': dense_act_scale,
            'dense_weights': dense_weights_scale,
            'fc_act': fc_act_scale,
            'fc_weights': fc_weights_scale,
            'gate_act': gate_act_scale,
            'gate_weights': gate_weights_scale,
            'proj_act': proj_act_scale,
            'proj_weights': proj_weights_scale,
        }
    """

    if model_path is None:
        logger.warning(f"--quantized_fp8_model_path not specified. "
                       f"Initialize quantization scales automatically.")
        return get_dummy_quant_scales(num_layers)
    weight_dict = np.load(model_path)

    # yapf: disable
    scaling_factor = {
        'qkv_act': [],
        'qkv_weights': [],
        'qkv_output': [],
        'dense_act': [],
        'dense_weights': [],
        'fc_act': [],
        'fc_weights': [],
        'gate_act': [],
        'gate_weights': [],
        'proj_act': [],
        'proj_weights': [],
    }

    for layer in range(num_layers):
        scaling_factor['qkv_act'].append(max(
            weight_dict[f'_np:layers:{layer}:attention:qkv:q:activation_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:k:activation_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:v:activation_scaling_factor'].item()
        ))
        scaling_factor['qkv_weights'].append(max(
            weight_dict[f'_np:layers:{layer}:attention:qkv:q:weights_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:k:weights_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:v:weights_scaling_factor'].item()
        ))
        if quant_mode is not None and quant_mode.has_fp8_kv_cache():
            # Not calibrarting KV cache.
            scaling_factor['qkv_output'].append(1.0)
        scaling_factor['dense_act'].append(weight_dict[f'_np:layers:{layer}:attention:dense:activation_scaling_factor'].item())
        scaling_factor['dense_weights'].append(weight_dict[f'_np:layers:{layer}:attention:dense:weights_scaling_factor'].item())
        scaling_factor['fc_act'].append(weight_dict[f'_np:layers:{layer}:mlp:fc:activation_scaling_factor'].item())
        scaling_factor['fc_weights'].append(weight_dict[f'_np:layers:{layer}:mlp:fc:weights_scaling_factor'].item())
        scaling_factor['gate_act'].append(weight_dict[f'_np:layers:{layer}:mlp:gate:activation_scaling_factor'].item())
        scaling_factor['gate_weights'].append(weight_dict[f'_np:layers:{layer}:mlp:gate:weights_scaling_factor'].item())
        scaling_factor['proj_act'].append(weight_dict[f'_np:layers:{layer}:mlp:proj:activation_scaling_factor'].item())
        scaling_factor['proj_weights'].append(weight_dict[f'_np:layers:{layer}:mlp:proj:weights_scaling_factor'].item())
    # yapf: enable
    for k, v in scaling_factor.items():
        assert len(v) == num_layers, \
            f'Expect scaling factor {k} of length {num_layers}, got {len(v)}'

    return scaling_factor


def gen_suffix(rank, use_smooth_quant, quant_per_channel):
    suffix = f"{rank}.bin"
    if use_smooth_quant:
        sq_prefix = "int8."
        if quant_per_channel:
            sq_prefix += "col."
        suffix = sq_prefix + suffix
    return suffix


def extract_layer_idx(name):
    ss = name.split('.')
    for s in ss:
        if s.isdigit():
            return s
    return None


def split(v: Union[np.ndarray, torch.Tensor],
          tp_size: int,
          tp_rank: int,
          dim=0):
    if tp_size == 1:
        return v
    assert len(v.shape) > 1 or dim == 0
    if isinstance(v, np.ndarray):
        return np.ascontiguousarray(
            np.split(v, tp_size, axis=dim)[tp_rank].copy())
    else:
        assert v.shape[dim] % tp_size == 0, \
            'Unable to split: shape={v.shape} (dim={dim}) tp_size={tp_size}.'
        split_size = v.shape[dim] // tp_size
        return v.split(split_size, dim=dim)[tp_rank].clone().detach()


def dup_kv_weight(v, num_head, tp_size):
    assert tp_size % num_head == 0
    reps = tp_size // num_head
    head_size = v.shape[0] // num_head
    v = v.reshape(num_head, head_size,
                  -1)[:, None, :, :].expand(num_head, reps, head_size,
                                            v.shape[1])
    return v.reshape(num_head * reps * head_size, -1).clone().detach()


def parse_ft_config(ini_file):
    gpt_config = configparser.ConfigParser()
    gpt_config.read(ini_file)

    n_embd = gpt_config.getint('llama', 'hidden_size')
    n_head = gpt_config.getint('llama', 'num_attention_heads')
    n_layer = gpt_config.getint('llama', 'num_hidden_layers')
    n_positions = gpt_config.getint('llama', 'max_position_embeddings')
    vocab_size = gpt_config.getint('llama', 'vocab_size')
    hidden_act = gpt_config.get('llama', 'hidden_act')
    inter_size = gpt_config.getint('llama', 'intermediate_size', fallback=None)
    n_kv_head = gpt_config.getint('llama', 'num_key_value_heads', fallback=None)

    if inter_size is None:
        inter_size = 4 * n_embd

    return n_embd, n_head, n_layer, n_positions, vocab_size, hidden_act, inter_size, n_kv_head


def load_from_hf_yuan(tensorrt_llm_yuan: tensorrt_llm.models.YuanForCausalLM,
                       hf_yuan,
                       mapping=Mapping(),
                       dtype='float32',
                       use_gemm_woq_plugin=True,
                       lora_config=LoraConfig()):
    tensorrt_llm.logger.info('Loading weights from HF LLaMA...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_yuan, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()
    num_kv_heads = tensorrt_llm_yuan.num_kv_heads
    mha_mode = (num_kv_heads == tensorrt_llm_yuan.num_heads)

    model_params = dict(hf_yuan.named_parameters())
    # concatenate, duplicate and reshape q, k, v -> qkv
    for l in range(hf_yuan.config.num_hidden_layers):
        prefix = f'model.layers.{l}.self_attn.'
        q_weight = model_params[prefix + 'q_proj.weight']
        k_weight = model_params[prefix + 'k_proj.weight']
        v_weight = model_params[prefix + 'v_proj.weight']
        if not mha_mode:
            head_size = tensorrt_llm_yuan.hidden_size // tensorrt_llm_yuan.num_heads
            if num_kv_heads < mapping.tp_size:
                # duplicate the KV heads up to tensor_parallel
                k_weight = dup_kv_weight(k_weight, num_kv_heads,
                                         mapping.tp_size)
                v_weight = dup_kv_weight(v_weight, num_kv_heads,
                                         mapping.tp_size)
            assert (k_weight.shape[0] % (mapping.tp_size * head_size)) == 0
            assert (v_weight.shape[0] % (mapping.tp_size * head_size)) == 0
            qkv_weight = [q_weight, k_weight, v_weight]
        else:
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

        model_params[prefix + 'qkv_proj.weight'] = qkv_weight
    torch_dtype = str_dtype_to_torch(dtype)
    layers_per_pipeline_stage = hf_yuan.config.num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))

    vocab_size = hf_yuan.config.vocab_size
    for k, v in model_params.items():
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
        if 'model.embed_tokens.weight' in k:
            if lora_config.is_valid and lora_config.embedding_weight is not None:
                v = torch_to_numpy(
                    lora_config.embedding_weight.to(torch_dtype).detach().cpu())
            if hf_yuan.config.tie_word_embeddings:
                # lm_head.weight has the same weights as embedding
                if mapping.is_last_pp_rank():
                    tensorrt_llm_yuan.lm_head.weight.value = np.ascontiguousarray(
                        split(v, mapping.tp_size, mapping.tp_rank))
            if tensorrt_llm_yuan.use_parallel_embedding:
                v = split(v, mapping.tp_size, mapping.tp_rank,
                          tensorrt_llm_yuan.embedding_sharding_dim)
            if mapping.is_first_pp_rank():
                tensorrt_llm_yuan.vocab_embedding.weight.value = v
        elif 'model.norm.weight' in k:
            if mapping.is_last_pp_rank():
                tensorrt_llm_yuan.ln_f.weight.value = v
        elif 'lm_head.weight' in k:
            if mapping.is_last_pp_rank():
                if lora_config.is_valid and lora_config.lm_head_weight is not None:
                    v = torch_to_numpy(
                        lora_config.lm_head_weight.to(
                            torch_dtype).detach().cpu())
                    vocab_size = v.shape[0]
                if vocab_size % mapping.tp_size != 0:
                    # padding
                    vocab_size_padded = tensorrt_llm_yuan.lm_head.out_features * mapping.tp_size
                    pad_width = vocab_size_padded - vocab_size
                    v = np.pad(v, ((0, pad_width), (0, 0)),
                               'constant',
                               constant_values=0)
                tensorrt_llm_yuan.lm_head.weight.value = np.ascontiguousarray(
                    split(v, mapping.tp_size, mapping.tp_rank))
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None or int(layer_idx) not in layers_range:
                continue
            idx = int(layer_idx) - mapping.pp_rank * layers_per_pipeline_stage
            if 'input_layernorm.weight' in k:
                tensorrt_llm_yuan.layers[idx].input_layernorm.weight.value = v
            elif 'post_attention_layernorm.weight' in k:
                dst = tensorrt_llm_yuan.layers[idx].post_layernorm.weight
                dst.value = v
            elif 'self_attn.q_proj.weight' in k:
                dst = tensorrt_llm_yuan.layers[idx].attention.q.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            torch.tensor(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        dst.value = torch.tensor(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_yuan.layers[
                        idx].attention.q.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = split_v
            elif 'self_attn.k_proj.weight' in k:
                dst = tensorrt_llm_yuan.layers[idx].attention.k.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            torch.tensor(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        dst.value = torch.tensor(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_yuan.layers[
                        idx].attention.k.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = split_v
            elif 'self_attn.v_proj.weight' in k:
                dst = tensorrt_llm_yuan.layers[idx].attention.v.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            torch.tensor(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        dst.value = torch.tensor(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_yuan.layers[
                        idx].attention.v.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = split_v
            elif 'self_attn.o_proj.weight' in k:
                dst = tensorrt_llm_yuan.layers[idx].attention.dense.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        dst.value = torch.tensor(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_yuan.layers[
                        idx].attention.dense.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.up_proj.weight' in k:
                dst = tensorrt_llm_yuan.layers[idx].mlp.fc.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            torch.tensor(v), plugin_weight_only_quant_type)

                    if not use_gemm_woq_plugin:
                        dst.value = torch.tensor(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_yuan.layers[
                        idx].mlp.gate.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.down_proj.weight' in k:
                dst = tensorrt_llm_yuan.layers[idx].mlp.proj.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            torch.tensor(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        dst.value = torch.tensor(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_yuan.layers[
                        idx].mlp.proj.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.gate_proj.weight' in k:
                dst = tensorrt_llm_yuan.layers[idx].mlp.gate.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            torch.tensor(v), plugin_weight_only_quant_type)

                    if not use_gemm_woq_plugin:
                        dst.value = torch.tensor(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_yuan.layers[
                        idx].mlp.fc.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'self_attn.lf_gate.conv1.weight' in k:
                tensorrt_llm_yuan.layers[idx].attention.lf_gate.conv1.weight.value = v
            elif 'self_attn.lf_gate.conv1.bias' in k:
                tensorrt_llm_yuan.layers[idx].attention.lf_gate.conv1.bias.value = v
            elif 'self_attn.lf_gate.conv2.weight' in k:
                tensorrt_llm_yuan.layers[idx].attention.lf_gate.conv2.weight.value = v
            elif 'self_attn.lf_gate.conv2.bias' in k:
                tensorrt_llm_yuan.layers[idx].attention.lf_gate.conv2.bias.value = v
            elif 'self_attn.lf_gate.output_layernorm.weight' in k:
                tensorrt_llm_yuan.layers[idx].attention.lf_gate.output_layernorm.weight.value = v
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')


class QkvWeightHelper:
    """ A helper utility for loading QKV weights from sharded files. """

    def __init__(self, model: tensorrt_llm.models.LLaMAForCausalLM):
        self.hidden_size = model.hidden_size
        self.num_heads = model.num_heads
        self.num_kv_heads = model.num_kv_heads
        self.tp_size = model.mapping.tp_size
        self.tp_rank = model.mapping.tp_rank
        self.is_mha = self.num_heads == self.num_kv_heads
        self._qkv_weights = {}

    @staticmethod
    def is_qkv_weight(name):
        for k in ['q_proj', 'k_proj', 'v_proj']:
            if 'self_attn' in name and k in name:
                return True
        return False

    def add_weight(self, i: int, name: str, weight: torch.Tensor):
        if 'q_proj' in name:
            tag = 'q'
        elif 'k_proj' in name:
            tag = 'k'
        elif 'v_proj' in name:
            tag = 'v'
        else:
            raise ValueError(f'Got an unexpected parameter of name {name}')
        if i not in self._qkv_weights:
            self._qkv_weights[i] = {}
        self._qkv_weights[i][tag] = weight

    def is_qkv_prepared(self, layer_id):
        if layer_id not in self._qkv_weights:
            return False
        weights = self._qkv_weights[layer_id]
        return 'q' in weights and 'k' in weights and 'v' in weights

    def split_qkv_weights(self, layer_id):
        if not self.is_qkv_prepared(layer_id):
            return None
        weights = self._qkv_weights.pop(layer_id)  # to prevent memory leak.
        q, k, v = (torch.tensor(weights[t]) for t in ['q', 'k', 'v'])

        if not self.is_mha:
            head_size = self.hidden_size // self.num_heads
            if self.num_kv_heads < self.tp_size:
                # duplicate the KV heads up to tensor_parallel
                k = dup_kv_weight(k, self.num_kv_heads, self.tp_size)
                v = dup_kv_weight(v, self.num_kv_heads, self.tp_size)
            assert k.shape[0] % (self.tp_size * head_size) == 0
            assert v.shape[0] % (self.tp_size * head_size) == 0
            wq = split(q, self.tp_size, self.tp_rank)
            wk = split(k, self.tp_size, self.tp_rank)
            wv = split(v, self.tp_size, self.tp_rank)
            fused_qkv = torch.cat((wq, wk, wv), dim=0)
        else:
            qkv = torch.cat([q, k, v], dim=0)
            qkv = qkv.reshape(3, q.shape[0], q.shape[1])
            fused_qkv = split(qkv, self.tp_size, self.tp_rank, dim=1)
            fused_qkv = fused_qkv.reshape(3 * (q.shape[0] // self.tp_size),
                                          q.shape[1])
        return fused_qkv



def load_from_gptq_yuan(tensorrt_llm_yuan,
                         quant_ckpt_path,
                         mapping=Mapping(),
                         dtype="float16",
                         bin_model_dir=None):
    tensorrt_llm.logger.info(
        'Loading weights from groupwise GPTQ LLaMA safetensors...')
    tik = time.time()

    gptq_yuan = safe_open(quant_ckpt_path, framework="pt", device=0)
    gptq_prefix = "model."
    gptq_suffix_list = [".qweight", ".qzeros", ".scales"]
    gptq_key_list = [
        "embed_tokens.weight",  # vocab_embedding
        "lm_head.weight",  # lm_head
        "norm.weight",  # ln_f
        "self_attn.",  # attention.qkv
        "_proj",  # qkv suffix
        "self_attn.o_proj",  # attention.dense
        "mlp.up_proj",  # mlp.gate
        "mlp.down_proj",  # mlp.proj
        "mlp.gate_proj",  # mlp.fc
        "input_layernorm.weight",  # input_layernorm
        "post_attention_layernorm.weight",  # post_layernorm
    ]
    split_sym = "."

    packer = torch.ops.fastertransformer.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm
    torch_dtype = str_dtype_to_torch(dtype)

    def load(key, no_prefix=0):
        if no_prefix:
            return gptq_yuan.get_tensor(key)
        else:
            return gptq_yuan.get_tensor(gptq_prefix + key)

    def torch_split(v, dim):
        if v.shape[dim] % mapping.tp_size != 0:
            tensorrt_llm.logger.error(
                "Current weight shape is invalid for mapping.tp_size=" +
                str(mapping.tp_size))
            assert False, "Invalid TP size"
        return v.split(v.shape[dim] // mapping.tp_size,
                       dim=dim)[mapping.tp_rank]

    def unpack_int32_into_int8(w_packed):
        # Unpack inputs packed in int32/float32 into uint4 and store them in int8 format
        w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
        w_unpacked = torch.zeros(w_packed_int4x2.shape[0],
                                 w_packed_int4x2.shape[1] * 2,
                                 dtype=torch.int8)
        w_unpacked[:, ::2] = w_packed_int4x2 % 16
        w_unpacked[:, 1::2] = w_packed_int4x2 // 16
        return w_unpacked.contiguous()

    def process_and_assign_weight(mOp, v, tp_dim=-1):
        if tp_dim == -1:
            qweight_int32, qzeros_int32, scales_fp16 = [
                item.cpu() for item in v
            ]
        else:
            qweight_int32, qzeros_int32, scales_fp16 = [
                torch_split(item, tp_dim).cpu() for item in v
            ]

        USE_UINT4_INPUT = 1  # Set to true if checkpoint store UINT4 weights
        USE_GPTQ_FOR_LLAMA = 1  # GPTQ-for-LLaMA added 1 to zeros

        qweight_unpacked_int8 = unpack_int32_into_int8(
            qweight_int32.T).T.contiguous() - 8
        qweight_interleaved = preprocessor(packer(qweight_unpacked_int8),
                                           torch.quint4x2).view(torch.int8)
        # zeros = zeros * scales
        qzeros_unpacked_int32 = unpack_int32_into_int8(qzeros_int32)
        if not USE_UINT4_INPUT:
            # Correcting UINT4 values back to INT4 order
            mask_negative = qzeros_unpacked_int32[qzeros_unpacked_int32 < 0]
            mask_positive = qzeros_unpacked_int32[qzeros_unpacked_int32 >= 0]
            qzeros_unpacked_int32 = qzeros_unpacked_int32 + 16 * mask_negative - 16 * mask_positive
        zeros_x_scales_fp16 = (-qzeros_unpacked_int32 + 8 * USE_UINT4_INPUT -
                               USE_GPTQ_FOR_LLAMA) * scales_fp16
        zeros_x_scales_fp16 = zeros_x_scales_fp16.half()

        # return processed interleaved weight, original scales and zeros * scales
        mOp.qweight.value = qweight_interleaved.cpu().numpy()
        mOp.scale.value = scales_fp16.cpu().numpy()
        mOp.zero.value = zeros_x_scales_fp16.cpu().numpy()

    # Load weights from GPTQ checkpoint into TRT-LLM module
    # 1. vocab_embedding
    v = load(gptq_key_list[0])
    if mapping.is_first_pp_rank():
        tensorrt_llm_yuan.vocab_embedding.weight.value = v.to(
            torch_dtype).cpu().numpy()

    # 2. lm_head
    v = load(gptq_key_list[1], "no_prefix")
    if mapping.is_last_pp_rank():
        tensorrt_llm_yuan.lm_head.weight.value = torch_split(
            v, 0).to(torch_dtype).cpu().numpy()

    # 3. ln_f
    v = load(gptq_key_list[2])
    if mapping.is_last_pp_rank():
        tensorrt_llm_yuan.ln_f.weight.value = v.to(torch_dtype).cpu().numpy()

    # 4. Weights inside each layer
    num_hidden_layers = tensorrt_llm_yuan.num_layers
    layers_per_pipeline_stage = num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))

    for l in layers_range:
        layer_idx = l - mapping.pp_rank * layers_per_pipeline_stage
        prefix = "layers" + split_sym + str(layer_idx) + split_sym
        tensorrt_llm.logger.info(f'Process weights in layer: {layer_idx}')
        layer = tensorrt_llm_yuan.layers[layer_idx]

        # 4.1 attention.qkv
        qkv_weight_list = []
        for suf in gptq_suffix_list:
            qkv_list = []
            for comp in ["q", "k", "v"]:
                comp_part = load(prefix + gptq_key_list[3] + comp +
                                 gptq_key_list[4] + suf)
                comp_part = torch_split(comp_part, 1)
                qkv_list.append(comp_part)
            qkv_weight_list.append(torch.cat(qkv_list, dim=1))

        process_and_assign_weight(layer.attention.qkv, qkv_weight_list)

        # 4.2 attention.dense
        v = [load(prefix + gptq_key_list[5] + suf) for suf in gptq_suffix_list]
        process_and_assign_weight(layer.attention.dense, v, 0)

        # 4.3 mlp.gate
        v = [load(prefix + gptq_key_list[6] + suf) for suf in gptq_suffix_list]
        process_and_assign_weight(layer.mlp.gate, v, 1)

        # 4.4 mlp.proj
        v = [load(prefix + gptq_key_list[7] + suf) for suf in gptq_suffix_list]
        process_and_assign_weight(layer.mlp.proj, v, 0)

        # 4.5 mlp.fc
        v = [load(prefix + gptq_key_list[8] + suf) for suf in gptq_suffix_list]
        process_and_assign_weight(layer.mlp.fc, v, 1)

        # 4.6 input_layernorm
        v = load(prefix + gptq_key_list[9])
        layer.input_layernorm.weight.value = v.to(torch_dtype).cpu().numpy()

        # 4.7 post_layernorm
        v = load(prefix + gptq_key_list[10])
        layer.post_layernorm.weight.value = v.to(torch_dtype).cpu().numpy()

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return


def load_from_awq_yuan(tensorrt_llm_yuan: YuanForCausalLM,
                        quant_ckpt_path,
                        mapping=Mapping(),
                        dtype="float16",
                        bin_model_dir=None):
    tensorrt_llm.logger.info(
        'Loading weights from groupwise AWQ LLaMA checkpoint...')
    tik = time.time()

    if quant_ckpt_path.endswith(".pt"):
        awq_yuan = torch.load(quant_ckpt_path)
        awq_prefix = "model."
        awq_suffix_list = [
            ".weight",
            ".weight_quantizer._amax",
            ".input_quantizer._pre_quant_scale",
        ]
        awq_key_list = [
            "embed_tokens.weight",  # vocab_embedding
            "lm_head",  # lm_head
            "norm.weight",  # ln_f
            "self_attn.",  # attention.qkv
            "_proj",  # qkv suffix
            "self_attn.o_proj",  # attention.dense
            "mlp.up_proj",  # mlp.gate
            "mlp.down_proj",  # mlp.proj
            "mlp.gate_proj",  # mlp.fc
            "input_layernorm.weight",  # input_layernorm
            "post_attention_layernorm.weight",  # post_layernorm
        ]
        split_sym = "."

        def load(key):
            if "lm_head" in key:
                v = awq_yuan[key]
            else:
                v = awq_yuan[awq_prefix + key]
            return v

        group_size = load("layers.0.self_attn.o_proj.weight").numel() // load(
            "layers.0.self_attn.o_proj.weight_quantizer._amax").numel()
    elif quant_ckpt_path.endswith(".npz"):
        awq_yuan = np.load(quant_ckpt_path)
        awq_prefix = "_np:"
        awq_suffix_list = [
            ":weight",
            ":weights_scaling_factor",
            ":prequant_scaling_factor",
        ]
        awq_key_list = [
            "vocab_embedding:weight",  # vocab_embedding
            "lm_head",  # lm_head
            "final_layernorm:weight",  # ln_f
            "attention:qkv:",  # attention.qkv
            "",  # qkv suffix
            "attention:dense",  # attention.dense
            "mlp:gate",  # mlp.gate
            "mlp:proj",  # mlp.proj
            "mlp:fc",  # mlp.fc
            "input_layernorm:weight",  # input_layernorm
            "post_layernorm:weight",  # post_layernorm
        ]
        split_sym = ":"

        def load(key):
            v = torch.from_numpy(awq_yuan[awq_prefix + key])
            if "weights_scaling_factor" in key:
                v *= 7  # For AMMO *.npz checkpoints
            return v

        group_size = load("layers:0:attention:dense:weight").numel() // load(
            "layers:0:attention:dense:weights_scaling_factor").numel()
    else:
        assert False, "Unsupported AWQ quantized checkpoint format"

    quant_mode = getattr(tensorrt_llm_yuan, 'quant_mode', QuantMode(0))
    # Int8 KV cache
    use_int8_kv_cache = quant_mode.has_int8_kv_cache()

    packer = torch.ops.fastertransformer.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm
    torch_dtype = str_dtype_to_torch(dtype)

    def fromfile(dir_path, name, shape=None, dtype=None):
        p = dir_path + '/' + name
        if Path(p).exists():
            t = np.fromfile(p, dtype=dtype)
            if shape is not None:
                t = t.reshape(shape)
            return t
        return None

    def torch_split(v, dim):
        if v.shape[dim] % mapping.tp_size != 0:
            tensorrt_llm.logger.error(
                "Current weight shape is invalid for mapping.tp_size=" +
                str(mapping.tp_size))
            assert False, "Invalid TP size"
        return v.split(v.shape[dim] // mapping.tp_size,
                       dim=dim)[mapping.tp_rank]

    def AWQ_quantize_pack_preprocess(weight, scale):
        weight /= scale.repeat_interleave(group_size, dim=0)
        qweight_int8 = torch.clamp(torch.round(weight.cuda()).char(), -8, 7)
        int4_weight = preprocessor(packer(qweight_int8.cpu()), torch.quint4x2)
        return int4_weight.view(torch.int8).cpu().numpy()

    def process_and_assign_weight(mOp, v, tp_dim=0):
        weight = v[0].T.contiguous()
        [k, n] = weight.shape
        weight = torch_split(weight, tp_dim)
        amax = v[1].reshape((n, k // group_size)).T.contiguous()
        amax = torch_split(amax, tp_dim)
        pre_quant_scale = v[2].reshape((1, k))
        if tp_dim == 0:
            pre_quant_scale = torch_split(pre_quant_scale, 1)
        scale = amax / 8.0
        mOp.qweight.value = AWQ_quantize_pack_preprocess(weight, scale)
        mOp.scale.value = scale.to(torch_dtype).cpu().numpy()
        mOp.pre_quant_scale.value = pre_quant_scale.to(
            torch_dtype).cpu().numpy()

    def reSmooth_and_get_scale(weight, pre_quant_scale, avg_pre_quant_scale):
        # deSmooth and reSmooth
        [k, n] = weight.shape
        if quant_ckpt_path.endswith("pt"):
            # NPZ files are already re-smoothed
            weight *= pre_quant_scale.repeat((n, 1)).transpose(1,
                                                               0).contiguous()
            weight /= avg_pre_quant_scale.repeat(
                (n, 1)).transpose(1, 0).contiguous()

        # Get scale
        weight_t = weight.T.contiguous()
        weight_t = weight_t.reshape(n, k // group_size, group_size)
        weight_t = torch.abs(weight_t.reshape(-1, group_size))
        amax, idx = weight_t.max(1)
        amax = amax.reshape(n, k // group_size).T.contiguous()
        scale = amax / 8
        return weight, scale

    def process_and_assign_qkv_weight(prefix, mOp):
        q_weight = load(prefix + "q" + awq_key_list[4] +
                        awq_suffix_list[0]).T.contiguous()
        k_weight = load(prefix + "k" + awq_key_list[4] +
                        awq_suffix_list[0]).T.contiguous()
        v_weight = load(prefix + "v" + awq_key_list[4] +
                        awq_suffix_list[0]).T.contiguous()
        dim_k = q_weight.shape[0]
        q_weight = torch_split(q_weight, 1)
        k_weight = torch_split(k_weight, 1)
        v_weight = torch_split(v_weight, 1)
        q_pre_quant_scale = load(prefix + "q" + awq_key_list[4] +
                                 awq_suffix_list[2]).reshape((1, dim_k))
        k_pre_quant_scale = load(prefix + "k" + awq_key_list[4] +
                                 awq_suffix_list[2]).reshape((1, dim_k))
        v_pre_quant_scale = load(prefix + "v" + awq_key_list[4] +
                                 awq_suffix_list[2]).reshape((1, dim_k))
        qkv_pre_quant_scale = (q_pre_quant_scale + k_pre_quant_scale +
                               v_pre_quant_scale) / 3.0
        q_weight, q_scale = reSmooth_and_get_scale(q_weight, q_pre_quant_scale,
                                                   qkv_pre_quant_scale)
        k_weight, k_scale = reSmooth_and_get_scale(k_weight, k_pre_quant_scale,
                                                   qkv_pre_quant_scale)
        v_weight, v_scale = reSmooth_and_get_scale(v_weight, v_pre_quant_scale,
                                                   qkv_pre_quant_scale)
        qkv_weights = torch.cat((q_weight, k_weight, v_weight), dim=1)
        qkv_scale = torch.cat((q_scale, k_scale, v_scale), dim=1)

        mOp.pre_quant_scale.value = qkv_pre_quant_scale.to(
            torch_dtype).cpu().numpy()
        mOp.qweight.value = AWQ_quantize_pack_preprocess(qkv_weights, qkv_scale)
        mOp.scale.value = qkv_scale.to(torch_dtype).cpu().numpy()

    # Load weights from AWQ checkpoint into TRT-LLM module
    # 1. vocab_embedding
    v = load(awq_key_list[0])
    # TRT-LLM requires vocab_size to be multiple of 64 for successful GEMM
    if v.shape[0] % 64 != 0:
        v = torch.nn.functional.pad(v, [0, 0, 0, 64 - v.shape[0] % 64])
    if mapping.is_first_pp_rank():
        tensorrt_llm_yuan.vocab_embedding.weight.value = v.to(
            torch_dtype).cpu().numpy()

    # 2. lm_head
    v = [load(awq_key_list[1] + suf) for suf in awq_suffix_list]
    if v[0].shape[0] % 64 != 0:
        v[0] = torch.nn.functional.pad(v[0], [0, 0, 0, 64 - v[0].shape[0] % 64])
        scale_align = 64 * (v[0].shape[1] // group_size)
        v[1] = v[1].reshape(-1)
        v[1] = torch.nn.functional.pad(
            v[1], [0, scale_align - v[1].shape[0] % scale_align], value=1)
    if mapping.is_last_pp_rank():
        process_and_assign_weight(tensorrt_llm_yuan.lm_head, v, 1)

    # 3. ln_f
    v = load(awq_key_list[2])
    if mapping.is_last_pp_rank():
        tensorrt_llm_yuan.ln_f.weight.value = v.to(torch_dtype).cpu().numpy()

    # 4. Weights inside each layer
    num_hidden_layers = tensorrt_llm_yuan.num_layers
    layers_per_pipeline_stage = num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))

    for l in layers_range:
        layer_idx = l - mapping.pp_rank * layers_per_pipeline_stage
        prefix = "layers" + split_sym + str(layer_idx) + split_sym
        tensorrt_llm.logger.info(f'Process weights in layer: {layer_idx}')
        layer = tensorrt_llm_yuan.layers[layer_idx]

        # 4.1 attention.qkv
        process_and_assign_qkv_weight(prefix + awq_key_list[3],
                                      layer.attention.qkv)

        # 4.2 attention.dense
        v = [load(prefix + awq_key_list[5] + suf) for suf in awq_suffix_list]
        process_and_assign_weight(layer.attention.dense, v, 0)

        # 4.3 mlp.gate
        v = [load(prefix + awq_key_list[6] + suf) for suf in awq_suffix_list]
        process_and_assign_weight(layer.mlp.gate, v, 1)

        # 4.4 mlp.proj
        v = [load(prefix + awq_key_list[7] + suf) for suf in awq_suffix_list]
        process_and_assign_weight(layer.mlp.proj, v, 0)

        # 4.5 mlp.fc
        v = [load(prefix + awq_key_list[8] + suf) for suf in awq_suffix_list]
        process_and_assign_weight(layer.mlp.fc, v, 1)

        # 4.6 input_layernorm
        v = load(prefix + awq_key_list[9])
        layer.input_layernorm.weight.value = v.to(torch_dtype).cpu().numpy()

        # 4.7 post_layernorm
        v = load(prefix + awq_key_list[10])
        layer.post_layernorm.weight.value = v.to(torch_dtype).cpu().numpy()

        # 4.8 attention.kv_quant_orig_scale / kv_quant_orig_scale
        if use_int8_kv_cache:
            assert bin_model_dir, "You must pass --bin_model_dir to tell TRT-LLM where to look for scales of INT8 kv cache."
            t = fromfile(
                bin_model_dir, 'model.layers.' + str(layer_idx) +
                '.attention.query_key_value.scale_y_quant_orig.bin', [1],
                np.float32)
            assert t is not None, f"{bin_model_dir} does not contain model.layers.{layer_idx}.attention.query_key_value.scale_y_quant_orig.bin"
            layer.attention.kv_orig_quant_scale.value = 1.0 / t
            layer.attention.kv_quant_orig_scale.value = t

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
