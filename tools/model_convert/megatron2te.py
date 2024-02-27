# Copyright (c) 2023 Alibaba PAI Team.
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

import argparse
import random
import os
import re
import sys
import types
import numpy as np
import torch
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

from transformers import AutoTokenizer, GPT2Config, LlamaConfig, AutoConfig


def add_args(parser):
    parser.add_argument("--megatron-path", type=str, default=None, help="Base directory of Megatron repository")

    parser.add_argument(
        "--convert_checkpoint_from_transformers_to_te",
        action="store_true",
        help=(
            "If True, convert a Transformers checkpoint to a Te checkpoint. "
            "If False, convert a Te checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the converted checkpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="model name",
    )


    parser.add_argument(
        "--target_tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The tensor model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_pipeline_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The pipeline model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )

    parser.add_argument(
        "--target_params_dtype",
        type=str,
        default="fp32",
        help=(
            "The dtype of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--make_vocab_size_divisible_by",
        type=int,
        default=128,
        help=(
            "Pad the vocab size to be divisible by this value. "
            "This is added for computational efficieny reasons. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )

    parser.add_argument(
        '--extra_num_vocabs',
        type=int,
        default=0,
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help=(
            "The name of the pre-trained tokenizer to save. "
            "If not None, the tokenizer will be saved. "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )

    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="10GB",
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )

    return parser


def get_element_from_dict_by_path(d, path):
    """
    Get element from dictionary by path. If element is not present, recursively add empty dictionaries.
    Args:
        d (dict): the dictionary to get the element from
        path (list): the path to the element which is delimited by "."
    """
    path = path.split(".")
    for k in path:
        if k not in d:
            d[k] = {}
        d = d[k]
    return d

def get_megatron_sharded_states(args, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.
    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        sub_dir_name = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}"
        checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir_name))[0]
        checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dicts.append(state_dict)
    return tp_state_dicts


def convert_checkpoint_from_megatron_to_te(args):
    os.makedirs(args.save_path, exist_ok=True)
    sub_dirs = os.listdir(args.load_path)
    possible_sub_dirs = ["mp_rank_00", "mp_rank_01", "mp_rank_02", "mp_rank_03", 
            "mp_rank_04","mp_rank_05", "mp_rank_06", "mp_rank_07",]
    for sub_dir in possible_sub_dirs:
        if sub_dir in sub_dirs:
            rank0_checkpoint_name = [i for i in os.listdir(os.path.join(args.load_path, sub_dir)) if 'rng' in i][0]
            rank0_checkpoint_path = os.path.join(args.load_path, sub_dir, rank0_checkpoint_name)
            break
            
    print(f"Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}")
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")
    megatron_args = state_dict.get("args", None)

    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint insead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )
    tracker_filepath = os.path.join(args.save_path, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as f:
        f.write("release")

    # create `release` dir in args.load_path
    release_dir = os.path.join(args.save_path, "release")
    os.makedirs(release_dir, exist_ok=True)

    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32


    output_state_dict = {}

    tp_size = megatron_args.tensor_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size

    # The regex to extract layer names.
    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z0-9_.]+)")

    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, 0)

    # Convert.
    # print("Converting")
    for tp_rank in range(tp_size):
        for key, val in tp_state_dicts[0]["model"]["language_model"]["encoder"].items():
    
            # Match the name.
            # print(key)
            m = layer_re.match(key)
            # Stop if that's not a layer
            if m is None:
                continue
    
            # The index of the layer.
            layer_idx = int(m.group(1))
            # The name of the operation.
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)
            layer_name = f"layers.{layer_idx}"
            # print(layer_name, op_name, weight_or_bias)
    
            if weight_or_bias == '_extra_state':
                continue
            else:
                params = val.to(dtype)
    
    
            if op_name == "mlp.dense_h_to_4h":
                out_name = ".mlp.fc1_weight"
                output_state_dict[layer_name + out_name] = params.to(dtype).clone()
            elif op_name == "mlp.dense_4h_to_h":
                out_name = ".mlp.fc2_weight"
                output_state_dict[layer_name + out_name] = params.to(dtype).clone()
            else:
                output_state_dict[layer_name + '.' + op_name + '.' + weight_or_bias] = params.to(dtype).clone()
        # print(output_state_dict)
        print("Converting final layernorm")
        params = get_element_from_dict_by_path(tp_state_dicts[0], "model.language_model.encoder")
        output_state_dict["final_layernorm.weight"] = params["final_layernorm.weight"].to(dtype).clone()


    tp_state_dicts[tp_rank]["model"]["language_model"]["encoder"] = output_state_dict

    for tp_rank in range(args.target_tensor_model_parallel_size):
        checkpoint_dir = (
            f"mp_rank_{tp_rank:02d}"
            if args.target_pipeline_model_parallel_size == 1
            else f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
        )

        checkpoint_name = "model_optim_rng.pt"
        checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        torch.save(tp_state_dicts[tp_rank], checkpoint_path)

def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    convert_checkpoint_from_megatron_to_te(args)


if __name__ == "__main__":
    main()
