# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Merge tensor parallel partitions."""

import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module

from megatron.checkpointing import load_checkpoint, save_checkpoint, _load_base_checkpoint, get_distributed_optimizer_checkpoint_name
from megatron.checkpointing import ensure_directory_exists
from megatron.checkpointing import get_checkpoint_name
from megatron.checkpointing import get_checkpoint_version
from megatron.checkpointing import get_checkpoint_tracker_filename
from megatron.global_vars import set_global_variables, get_args
from megatron.global_vars import rebuild_tokenizer
from megatron.initialize import initialize_megatron
from megatron.arguments import (parse_args, validate_args)
from megatron.core import mpu
from megatron import update_num_microbatches
from megatron.core import mpu, tensor_parallel
from megatron.global_vars import get_args
from megatron.utils import (unwrap_model, print_rank_0)
from megatron.checkpointing import _load_base_checkpoint
from megatron.optimizer import get_megatron_optimizer, get_param_groups
from megatron.training import  get_optimizer_param_scheduler
from megatron.checkpointing import load_checkpoint
from copy import deepcopy
from tqdm import tqdm
from pretrain_yuan import model_provider

def get_model():
    args = get_args()

    pre_process = True if mpu.is_pipeline_first_stage() else False
    post_process = True if mpu.is_pipeline_last_stage() else False
    model = model_provider(pre_process, post_process)
    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    model = [LocalDDP(model_module,
                      args.accumulate_allreduce_grads_in_fp32,
                      args.use_contiguous_buffers_in_local_ddp)
             for model_module in model]
    return model

def get_parallel_checkpoint_name(path):

    tracker_filename = get_checkpoint_tracker_filename(path)
    iteration = 0
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        iteration = int(metastring)
    assert iteration > 0
    checkpoint_name = get_checkpoint_name(path, iteration)

    return checkpoint_name, iteration

def get_mp_merge_args(parser):
    """Provide extra arguments required for merging."""
    group = parser.add_argument_group(title='mp merge')

    group.add_argument('--model-type',type=str,help='Type of the model.')
    group.add_argument('--target-tensor-model-parallel-size', type=int, default=2,
                       help='Degree of pipeline model parallelism in output model.')
    group.add_argument('--target-pipeline-model-parallel-size', type=int, default=1,
                       help='Degree of pipeline model parallelism in output model.')
    group.add_argument('--with-distributed-optimizer', action='store_true',
                       help='Use distributed optimizer during split ckpt.')
    group.add_argument('--pipeline-generate-layer', type=str, default=None, help='This parameter controls which layers only convert the parameter.')
    group.add_argument('--tensor-generate-layer', type=str, default=None, help='This parameter controls which layers only convert the parameter.')

    return parser

def main():
    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    os.environ["WORLD_SIZE"] = "{}".format(2**5)

    # Args
    args = parse_args(extra_args_provider=get_mp_merge_args, ignore_unknown_args=True)
    validate_args(args)
    set_global_variables(args)
    args = get_args()

    args.orig_tensor_model_parallel_size = args.tensor_model_parallel_size
    args.orig_pipeline_model_parallel_size = args.pipeline_model_parallel_size
    args.orig_transformer_pipeline_model_parallel_size = args.transformer_pipeline_model_parallel_size

    args.target_transformer_pipeline_model_parallel_size = (
        args.target_pipeline_model_parallel_size - 1
        if args.standalone_embedding_stage else
        args.target_pipeline_model_parallel_size
    )

    print('\n merging tensor parallel partitions ...')
    print(' > orig number of partitions: {}'.format(args.orig_tensor_model_parallel_size))
    print(' > checkpoint path: {}'.format(args.load))
    print(' > model parameters:')
    print('    number of layers ................ {}'.format(args.num_layers))
    print('    hidden size ..................... {}'.format(args.hidden_size))
    print('    number of attention heads ....... {}'.format(args.num_attention_heads))
    if args.position_embedding_type != 'rope':
        print('    maximum position embeddings ..... {}'.format(args.max_position_embeddings))

    # Build and load partitions.
    partitions = []
    tokenizer = rebuild_tokenizer(args)
    sub_tensor_parallel_size = args.orig_tensor_model_parallel_size // args.target_tensor_model_parallel_size
    for pp_rank in range(args.target_pipeline_model_parallel_size):
        for tp_rank in range(args.target_tensor_model_parallel_size):
            print('processing {} {}'.format(pp_rank,tp_rank))
            # set orig pp_rank and tp_rank
            args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
            args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size
            args.transformer_pipeline_model_parallel_size = args.target_transformer_pipeline_model_parallel_size
            mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
            mpu.set_tensor_model_parallel_rank(tp_rank)
            mpu.set_pipeline_model_parallel_world_size(args.pipeline_model_parallel_size)
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            # build orig model
            model_ = get_model()
            model = unwrap_model(model_)
            param_groups = get_param_groups(model_, None, None, 1.0)

            total_sub_models = []
            sub_state_dicts = []
            total_sub_optim_state_dicts = []
            print('loading sub ckpts')
            for sub_tp_rank in tqdm(range(sub_tensor_parallel_size)):
                
                args.tensor_model_parallel_size = args.orig_tensor_model_parallel_size
                args.pipeline_model_parallel_size = args.orig_pipeline_model_parallel_size
                args.transformer_pipeline_model_parallel_size = args.orig_transformer_pipeline_model_parallel_size
                mpu.set_tensor_model_parallel_world_size(args.orig_tensor_model_parallel_size)
                mpu.set_tensor_model_parallel_rank(tp_rank * sub_tensor_parallel_size + sub_tp_rank)
                mpu.set_pipeline_model_parallel_world_size(args.pipeline_model_parallel_size)
                mpu.set_pipeline_model_parallel_rank(pp_rank)

                # load ckpt of submodel
                sub_state_dict, sub_checkpoint_name, release = _load_base_checkpoint(args.load, rank0=False)
                sub_model_ = get_model()
                sub_model = unwrap_model(sub_model_)
                sub_param_groups = get_param_groups(sub_model_, None, None, 1.0)
                sub_state_dicts.append(sub_state_dict)

                # Load orig Model.
                if len(sub_model) == 1:
                    sub_model[0].load_state_dict(sub_state_dict['model'], strict=True)
                else:
                    for i in range(len(model)):
                        mpu.set_virtual_pipeline_model_parallel_rank(i)
                        sub_model[i].load_state_dict(sub_state_dict['model%d' % i], strict=True)
                total_sub_models.append(sub_model_)
                total_numel = 0
                for name, param in sub_model[0].named_parameters():
                    total_numel += param.numel()
                if not args.no_load_optim and args.use_distributed_optimizer:
                    sub_optim_checkpoint_name = get_distributed_optimizer_checkpoint_name(sub_checkpoint_name)
                    sub_optim_state_dict = torch.load(sub_optim_checkpoint_name, map_location='cpu')
                    assert total_numel == sub_optim_state_dict[0][torch.float32]['param'].shape[0]
                    total_sub_optim_state_dicts.append(sub_optim_state_dict)


            # modify weight in sub_state_dict
            state_dict = deepcopy(sub_state_dict)
            print('processing weight......')
            for name, param in model_[0].named_parameters():
                if param.tensor_model_parallel:
                    #chunk_size = param.shape[param.partition_dim] // sub_tensor_parallel_size
                    if 'dense_h_to_4h' in name:
                        sub_chunks0 = []
                        sub_chunks1 = []
                        for sub_model_ in total_sub_models:
                            for sub_name, sub_param in sub_model_[0].named_parameters():
                                if name == sub_name:
                                    sub_chunk_size = sub_param.shape[-1]//2
                                    sub_chunk0 = torch.split(sub_param.data, sub_chunk_size, dim=-1)[0].clone().detach()
                                    sub_chunk1 = torch.split(sub_param.data, sub_chunk_size, dim=-1)[1].clone().detach()
                                    sub_chunks0.append(sub_chunk0)
                                    sub_chunks1.append(sub_chunk1)
                        chunk0 = torch.cat(sub_chunks0, dim=param.partition_dim).clone().detach()
                        chunk1 = torch.cat(sub_chunks1, dim=param.partition_dim).clone().detach()
                        param.data.copy_(torch.cat([chunk0, chunk1], dim=-1).clone().detach())
                    else:
                        sub_params = []
                        for sub_model_ in total_sub_models:
                            for sub_name, sub_param in sub_model_[0].named_parameters():
                                if name == sub_name:
                                    sub_params.append(sub_param.data)
                        param.data.copy_(torch.cat(sub_params, dim=param.partition_dim).clone().detach())
                else:
                    for sub_model_ in total_sub_models:
                        for sub_name, sub_param in sub_model_[0].named_parameters():
                            if name == sub_name:
                                param.data.copy_(sub_param.data.clone().detach())
            
            
            state_dict['model'] = model_[0].state_dict_for_save_checkpoint()

            if not args.no_load_optim and not args.use_distributed_optimizer:
                
                state_dict['optimizer']['optimizer']['param_groups'][0]['params'] = list(range(len(param_groups[0]['params'])))
                state_dict['optimizer']['optimizer']['param_groups'][1]['params'] = [i + len(param_groups[0]['params']) for i in range(len(param_groups[1]['params']))]
                acc_count = 0
                for i, pg in enumerate(param_groups):
                    start = acc_count
                    for j, param in enumerate(pg['params']):
                        if param.tensor_model_parallel:
                            if state_dict['model']['language_model']['encoder']['layers.0.mlp.dense_h_to_4h.weight'].shape == param.shape:
                                exp_avgs_chunk0 = []
                                exp_avg_sqs_chunk0 = []
                                fp32_from_fp16_params_chunk0 = []
                                exp_avgs_chunk1 = []
                                exp_avg_sqs_chunk1 = []
                                fp32_from_fp16_params_chunk1 = []
                                for sub_state_dict in sub_state_dicts:
                                    chunk_size = param.shape[-1]//2
                                    fp32_from_fp16_params_chunk0.append(torch.split(sub_state_dict['optimizer']['fp32_from_fp16_params'][i][j], chunk_size, dim=-1)[0].clone().detach())
                                    fp32_from_fp16_params_chunk1.append(torch.split(sub_state_dict['optimizer']['fp32_from_fp16_params'][i][j], chunk_size, dim=-1)[1].clone().detach())
                                    exp_avgs_chunk0.append(torch.split(state_dict['optimizer']['optimizer']['state'][acc_count]['exp_avg'], chunk_size, dim=-1)[0].clone().detach())
                                    exp_avgs_chunk1.append(torch.split(state_dict['optimizer']['optimizer']['state'][acc_count]['exp_avg'], chunk_size, dim=-1)[1].clone().detach())
                                    exp_avg_sqs_chunk0.append(torch.split(state_dict['optimizer']['optimizer']['state'][acc_count]['exp_avg_sq'], chunk_size, dim=-1)[0].clone().detach())
                                    exp_avg_sqs_chunk1.append(torch.split(state_dict['optimizer']['optimizer']['state'][acc_count]['exp_avg_sq'], chunk_size, dim=-1)[1].clone().detach())

                                fp32_from_fp16_params_chunk0 = torch.cat(fp32_from_fp16_params_chunk0, dim = param.partition_dim)
                                fp32_from_fp16_params_chunk1 = torch.cat(fp32_from_fp16_params_chunk1, dim = param.partition_dim)
                                exp_avgs_chunk0 = torch.cat(exp_avgs_chunk0, dim = param.partition_dim)
                                exp_avgs_chunk1 = torch.cat(exp_avgs_chunk1, dim = param.partition_dim)
                                exp_avg_sqs_chunk0 = torch.cat(exp_avg_sqs_chunk0, dim = param.partition_dim)
                                exp_avg_sqs_chunk1 = torch.cat(exp_avg_sqs_chunk1, dim = param.partition_dim)

                                fp32_from_fp16_params = torch.cat([fp32_from_fp16_params_chunk0, fp32_from_fp16_params_chunk1], dim = -1)
                                exp_avgs = torch.cat([exp_avgs_chunk0, exp_avgs_chunk1], dim = -1)
                                exp_avg_sqs = torch.cat([exp_avg_sqs_chunk0, exp_avg_sqs_chunk1], dim = -1)
                                assert param.shape == fp32_from_fp16_params.shape, 'fp32_from_fp16_prams, {}, {}'.format(param.shape, fp32_from_fp16_params.shape)
                                assert param.shape == exp_avgs.shape, 'exp_avgs, {}, {}'.format(param.shape, exp_avgs.shape)
                                assert param.shape == exp_avg_sqs.shape, 'exp_avg_sqs, {}, {}'.format(param.shape, exp_avg_sqs.shape)
                                state_dict['optimizer']['fp32_from_fp16_params'][i][j] = fp32_from_fp16_params
                                state_dict['optimizer']['optimizer']['state'][acc_count]['exp_avg'] = exp_avgs
                                state_dict['optimizer']['optimizer']['state'][acc_count]['exp_avg_sq'] = exp_avg_sqs

                            else:
                                exp_avgs = []
                                exp_avg_sqs = []
                                fp32_from_fp16_params = []
                                for sub_state_dict in sub_state_dicts:
                                    fp32_from_fp16_params.append(sub_state_dict['optimizer']['fp32_from_fp16_params'][i][j].detach().clone())
                                    exp_avgs.append(sub_state_dict['optimizer']['optimizer']['state'][acc_count]['exp_avg'].detach().clone())
                                    exp_avg_sqs.append(sub_state_dict['optimizer']['optimizer']['state'][acc_count]['exp_avg_sq'].detach().clone())
                                fp32_from_fp16_params = torch.cat(fp32_from_fp16_params, dim = param.partition_dim)
                                exp_avgs = torch.cat(exp_avgs, dim = param.partition_dim)
                                exp_avg_sqs = torch.cat(exp_avg_sqs, dim = param.partition_dim)
                                assert param.shape == fp32_from_fp16_params.shape, 'fp32_from_fp16_prams, {}, {}'.format(param.shape, fp32_from_fp16_params.shape)
                                assert param.shape == exp_avgs.shape, 'exp_avgs, {}, {}'.format(param.shape, exp_avgs.shape)
                                assert param.shape == exp_avg_sqs.shape, 'exp_avg_sqs, {}, {}'.format(param.shape, exp_avg_sqs.shape)
                                state_dict['optimizer']['fp32_from_fp16_params'][i][j] = fp32_from_fp16_params
                                state_dict['optimizer']['optimizer']['state'][acc_count]['exp_avg'] = exp_avgs
                                state_dict['optimizer']['optimizer']['state'][acc_count]['exp_avg_sq'] = exp_avg_sqs
                        else:
                            for sub_state_dict in sub_state_dicts:
                                assert state_dict['optimizer']['fp32_from_fp16_params'][i][j].shape == sub_state_dict['optimizer']['fp32_from_fp16_params'][i][j].shape
                                assert state_dict['optimizer']['optimizer']['state'][acc_count]['exp_avg'].shape == sub_state_dict['optimizer']['optimizer']['state'][acc_count]['exp_avg'].shape
                                assert state_dict['optimizer']['optimizer']['state'][acc_count]['exp_avg_sq'].shape == sub_state_dict['optimizer']['optimizer']['state'][acc_count]['exp_avg_sq'].shape
                                state_dict['optimizer']['fp32_from_fp16_params'][i][j] = sub_state_dict['optimizer']['fp32_from_fp16_params'][i][j].detach().clone()
                                state_dict['optimizer']['optimizer']['state'][acc_count]['exp_avg'] = sub_state_dict['optimizer']['optimizer']['state'][acc_count]['exp_avg'].detach().clone()
                                state_dict['optimizer']['optimizer']['state'][acc_count]['exp_avg_sq'] = sub_state_dict['optimizer']['optimizer']['state'][acc_count]['exp_avg_sq'].detach().clone()
                                break
                        acc_count += 1
            state_dict['args'].tensor_model_parallel_size = args.target_tensor_model_parallel_size
            args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
            mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
            mpu.set_tensor_model_parallel_rank(tp_rank)
            # output state dict ckpt file
            iteration = sub_state_dicts[0]['iteration']
            checkpoint_name = get_checkpoint_name(args.save, iteration)
            ensure_directory_exists(checkpoint_name)
            print('saving to ', checkpoint_name)
            torch.save(state_dict, checkpoint_name)
            # writing txt file
            if not torch.distributed.is_initialized() \
               or torch.distributed.get_rank() == 0:
                tracker_filename = get_checkpoint_tracker_filename(args.save)
                with open(tracker_filename, 'w') as f:
                    f.write(str(iteration))
            if not args.no_load_optim and args.use_distributed_optimizer:
                optim_checkpoint_name = get_distributed_optimizer_checkpoint_name(checkpoint_name)
                optim_state_dict = deepcopy(total_sub_optim_state_dicts[0])
                optim_state_dict[0][torch.float32]['param'] = None
                optim_state_dict[0][torch.float32]['exp_avg'] = None
                optim_state_dict[0][torch.float32]['exp_avg_sq'] = None
                sub_offset = 0
                for param_group in param_groups:
                    for p in param_group['params']:
                        if p.tensor_model_parallel:
                            if state_dict['model']['language_model']['encoder']['layers.0.mlp.dense_h_to_4h.weight'].shape == p.shape:
                                exp_avgs_chunk0 = []
                                exp_avg_sqs_chunk0 = []
                                params_chunk0 = []
                                exp_avgs_chunk1 = []
                                exp_avg_sqs_chunk1 = []
                                params_chunk1 = []
                                for sub_optim_state_dicts in total_sub_optim_state_dicts:
                                    chunk_size = p.numel() // sub_tensor_parallel_size
                                    sub_chunk_size = p.numel() // sub_tensor_parallel_size // 2
                                    sub_shape = list(p.shape)
                                    sub_shape[p.partition_dim] = p.shape[partition_dim] // sub_tensor_parallel_size

                                    param_tensor = sub_optim_state_dicts[0][torch.float32]['param'][sub_offset: sub_offset+chunk_size].clone().detach().reshape(sub_shape)
                                    exp_avg_tensor = sub_optim_state_dicts[0][torch.float32]['exp_avg'][sub_offset: sub_offset+chunk_size].clone().detach().reshape(sub_shape)
                                    exp_avg_sq_tensor = sub_optim_state_dicts[0][torch.float32]['exp_avg_sq'][sub_offset: sub_offset+chunk_size].clone().detach().reshape(sub_shape)
                                    sub_chunk_size = sub_shape[-1] // 2
                                    params_chunk0.append(torch.split(param_tensor, sub_chunk_size, dim=-1)[0].detach().clone())
                                    params_chunk1.append(torch.split(param_tensor, sub_chunk_size, dim=-1)[1].detach().clone())

                                    exp_avgs_chunk0.append(torch.split(exp_avg_tensor, sub_chunk_size, dim=-1)[0].detach().clone())
                                    exp_avgs_chunk1.append(torch.split(exp_avg_tensor, sub_chunk_size, dim=-1)[1].detach().clone())

                                    exp_avg_sqs_chunk0.append(torch.split(exp_avg_sq_tensor, sub_chunk_size, dim=-1)[0].detach().clone())
                                    exp_avg_sqs_chunk1.append(torch.split(exp_avg_sq_tensor, sub_chunk_size, dim=-1)[1].detach().clone())
                                params_chunk0 = torch.cat(params_chunk0, dim=p.partition_dim).detach().clone()
                                params_chunk1 = torch.cat(params_chunk1, dim=p.partition_dim).detach().clone()
                                param_tensor = torch.cat([params_chunk0, params_chunk1], dim=-1).detach.clone()

                                exp_avg_chunk0 = torch.cat(exp_avg_chunk0, dim=p.partition_dim).detach().clone()
                                exp_avg_chunk1 = torch.cat(exp_avg_chunk1, dim=p.partition_dim).detach().clone()
                                exp_avg_tensor = torch.cat([exp_avg_chunk0, exp_avg_chunk1], dim=-1).detach().clone()

                                exp_avg_sq_chunk0 = torch.cat(exp_avg_sq_chunk0, dim=p.partition_dim).detach().clone()
                                exp_avg_sq_chunk1 = torch.cat(exp_avg_sq_chunk1, dim=p.partition_dim).detach().clone()
                                exp_avg_sq_tensor = torch.cat([exp_avg_sq_chunk0, exp_avg_sq_chunk1], dim=-1).detach().clone()
                                assert p.shape == param_tensor.shape and p.shape == exp_avg_tensor.shape and p.shape == exp_avg_sq.shape
                                if optim_state_dict[0][torch.float32]['param'] == None:
                                    optim_state_dict[0][torch.float32]['param'] = param_tensor.flatten()
                                    optim_state_dict[0][torch.float32]['exp_avg'] = exp_avg_tensor.flatten()
                                    optim_state_dict[0][torch.float32]['exp_avg_sq'] = exp_avg_sq_tensor.flatten()
                                else:
                                    optim_state_dict[0][torch.float32]['param'] = torch.cat([optim_state_dict[0][torch.float32]['param'], param_tensor.flatten()])
                                    optim_state_dict[0][torch.float32]['exp_avg'] = torch.cat([optim_state_dict[0][torch.float32]['exp_avg'], exp_avg_tensor.flatten()])
                                    optim_state_dict[0][torch.float32]['exp_avg_sq'] = torch.cat([optim_state_dict[0][torch.float32]['exp_avg_sq'], exp_avg_sq_tensor.flatten()])
                                sub_offset += chunk_size
                            else:
                                param_tensors = []
                                exp_avg_tensors = []
                                exp_avg_sq_tensors = []
                                chunk_size = p.numel() //  sub_tensor_parallel_size
                                sub_shape = list(p.shape)
                                sub_shape[p.partition_dim] = p.shape[p.partition_dim] // sub_tensor_parallel_size
                                for sub_optim_state_dicts in total_sub_optim_state_dicts:
                                    param_tensors.append(sub_optim_state_dicts[0][torch.float32]['param'][sub_offset: sub_offset+chunk_size].clone().detach().reshape(sub_shape))
                                    exp_avg_tensors.append(sub_optim_state_dicts[0][torch.float32]['exp_avg'][sub_offset: sub_offset+chunk_size].clone().detach().reshape(sub_shape))
                                    exp_avg_sq_tensors.append(sub_optim_state_dicts[0][torch.float32]['exp_avg_sq'][sub_offset: sub_offset+chunk_size].clone().detach().reshape(sub_shape))
                                param_tensor = torch.cat(param_tensors, dim=p.partition_dim).flatten()
                                exp_avg_tensor = torch.cat(exp_avg_tensors, dim=p.partition_dim).flatten()
                                exp_avg_sq_tensor = torch.cat(exp_avg_sq_tensors, dim=p.partition_dim).flatten()
                                if optim_state_dict[0][torch.float32]['param'] == None:
                                    optim_state_dict[0][torch.float32]['param'] = param_tensor
                                    optim_state_dict[0][torch.float32]['exp_avg'] = exp_avg_tensor
                                    optim_state_dict[0][torch.float32]['exp_avg_sq'] = exp_avg_sq_tensor
                                else:
                                    optim_state_dict[0][torch.float32]['param'] = torch.cat([optim_state_dict[0][torch.float32]['param'], param_tensor])
                                    optim_state_dict[0][torch.float32]['exp_avg'] = torch.cat([optim_state_dict[0][torch.float32]['exp_avg'], exp_avg_tensor])
                                    optim_state_dict[0][torch.float32]['exp_avg_sq'] = torch.cat([optim_state_dict[0][torch.float32]['exp_avg_sq'], exp_avg_sq_tensor])
                                sub_offset += chunk_size
                        else:
                            param_tensor = total_sub_optim_state_dicts[0][0][torch.float32]['param'][sub_offset: sub_offset+p.numel()].clone().detach()
                            exp_avg_tensor = total_sub_optim_state_dicts[0][0][torch.float32]['exp_avg'][sub_offset: sub_offset+p.numel()].clone().detach()
                            exp_avg_sq_tensor = total_sub_optim_state_dicts[0][0][torch.float32]['exp_avg_sq'][sub_offset: sub_offset+p.numel()].clone().detach()
                            if optim_state_dict[0][torch.float32]['param'] == None:
                                optim_state_dict[0][torch.float32]['param'] = param_tensor
                                optim_state_dict[0][torch.float32]['exp_avg'] = exp_avg_tensor
                                optim_state_dict[0][torch.float32]['exp_avg_sq'] = exp_avg_sq_tensor
                            else:
                                optim_state_dict[0][torch.float32]['param'] = torch.cat([optim_state_dict[0][torch.float32]['param'], param_tensor])
                                optim_state_dict[0][torch.float32]['exp_avg'] = torch.cat([optim_state_dict[0][torch.float32]['exp_avg'], exp_avg_tensor])
                                optim_state_dict[0][torch.float32]['exp_avg_sq'] = torch.cat([optim_state_dict[0][torch.float32]['exp_avg_sq'], exp_avg_sq_tensor])
                            
                            sub_offset += p.numel()
                optim_checkpoint_name = get_distributed_optimizer_checkpoint_name(checkpoint_name)
                ensure_directory_exists(optim_checkpoint_name)
                torch.save(optim_state_dict, optim_checkpoint_name)
    print('done :-)')

if __name__ == '__main__':
    main()


