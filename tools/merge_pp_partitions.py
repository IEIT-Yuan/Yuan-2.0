import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module

from megatron.checkpointing import load_checkpoint, save_checkpoint, _load_base_checkpoint, \
    get_distributed_optimizer_checkpoint_name
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
from megatron.training import get_optimizer_param_scheduler
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


def get_blocks(pp_rank,target_pipeline_model_parallel_blocks,pipeline_model_parallel_blocks):
    pp_size = len(target_pipeline_model_parallel_blocks)
    blocks_start = sum(target_pipeline_model_parallel_blocks[:pp_rank])
    blocks_end = blocks_start+target_pipeline_model_parallel_blocks[pp_rank]-1

    orig_blocks = [ sum(pipeline_model_parallel_blocks[:i+1]) for i in range(len(pipeline_model_parallel_blocks))]
    

    def check_pos(block_num,arr):
        for i in range(len(arr)-1):
            if i==0 and block_num < arr[i]:
                return (i,block_num)
            elif i>0 and block_num < arr[i]:
                return (i,block_num-arr[i-1])
            elif block_num == arr[i]:
                return (i+1,0)
            elif block_num>arr[i] and block_num<arr[i+1]:
                return (i+1,block_num-arr[i])
    blocks_start_orig,num_start = check_pos(blocks_start,orig_blocks)
    blocks_end_orig,num_end = check_pos(blocks_end,orig_blocks)

    return blocks_start_orig,num_start,blocks_end_orig,num_end,orig_blocks

def load_orig_ckpt(orig_pp_rank,tp_rank,args):
    args.tensor_model_parallel_size = args.orig_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.orig_pipeline_model_parallel_size
    args.transformer_pipeline_model_parallel_size = args.orig_transformer_pipeline_model_parallel_size
    mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
    mpu.set_tensor_model_parallel_rank(tp_rank)
    mpu.set_pipeline_model_parallel_world_size(args.pipeline_model_parallel_size)
    mpu.set_pipeline_model_parallel_rank(orig_pp_rank)

    model_ = get_model()
    model = unwrap_model(model_)

    state_dict, checkpoint_name, release = _load_base_checkpoint(args.load, rank0=False)

    optim_state_dict = None
    if not args.no_load_optim and args.use_distributed_optimizer:
        optim_checkpoint_name=get_distributed_optimizer_checkpoint_name(checkpoint_name)
        optim_state_dict=torch.load(optim_checkpoint_name,map_location='cpu')

    if len(model) == 1:
        model[0].load_state_dict(state_dict['model'], strict=True)
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            model[i].load_state_dict(state_dict['model%d' % i], strict=True)
    total_numel = 0
    for name,param in model[0].named_parameters():
        total_numel +=  param.numel()
    
    param_groups = get_param_groups(model_, None, None, 1.0)
    return state_dict,optim_state_dict,param_groups

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
    group.add_argument('--target-pipeline-model-parallel-blocks',type=str, default=None,
                        help='The number of transformer layers specified by the user for each pipeline stage in output model.')
    group.add_argument('--tensor-generate-layer',type=str, default=None, help='This parameter controls which layers only convert the parameter.')
    group.add_argument('--pipeline-generate-layer', type=str, default=None, help='This parameter controls which layers only convert the parameter.')

    return parser

def main():
    os.environ["WORLD_SIZE"] = '{}'.format(2 ** 5)

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
    print('\n merging pipeline parallel partitions ...')
    print(' > orig number of partitions: {}'.format(args.orig_pipeline_model_parallel_size))
    print(' > checkpoint path: {}'.format(args.load))
    print(' > model parameters:')
    print('    number of layers ................ {}'.format(args.num_layers))
    print('    hidden size ..................... {}'.format(args.hidden_size))
    print('    number of attention heads ....... {}'.format(args.num_attention_heads))
    if args.position_embedding_type != 'rope':
        print('    maximum position embeddings ..... {}'.format(args.max_position_embeddings))

    #build and load partitions
    target_pipeline_model_parallel_blocks = [int(x) for x in args.target_pipeline_model_parallel_blocks.split(',')]
    pipeline_model_parallel_blocks = [int(x) for x in args.pipeline_model_parallel_blocks.split(',')]
    tensor_generate_layer_index = [int(x) for x in args.tensor_generate_layer.split(',')]
    sizes = []
    for tp_rank in tensor_generate_layer_index:
        for pp_rank in range(args.target_pipeline_model_parallel_size):
            print('processing pp_rank {},tp_rank  {}'.format(pp_rank,tp_rank))
            block_start_orig,num_start,block_end_orig,num_end,orig_blocks = get_blocks(pp_rank,target_pipeline_model_parallel_blocks,pipeline_model_parallel_blocks)
            state_dicts=[]
            optim_state_dicts=[]
            param_groupss=[]
            for orig_pp_rank in range(block_start_orig,block_end_orig+1):
                
                state_dict,optim_state_dict,param_groups = load_orig_ckpt(orig_pp_rank,tp_rank,args)
                state_dicts.append(state_dict)
                optim_state_dicts.append(optim_state_dict)
                param_groupss.append(param_groups)


            args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
            args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size
            args.transformer_pipeline_model_parallel_size = args.target_transformer_pipeline_model_parallel_size
            mpu.set_tensor_model_parallel_world_size(args.target_tensor_model_parallel_size)
            mpu.set_pipeline_model_parallel_world_size(args.target_pipeline_model_parallel_size)
            mpu.set_tensor_model_parallel_rank(tp_rank)
            mpu.set_pipeline_model_parallel_rank(pp_rank)

            model_ = get_model()
            model = unwrap_model(model_)
            new_state_dict = {}
            new_param_groups = get_param_groups(model_,None,None,1.0)

            if pp_rank == args.target_pipeline_model_parallel_size-1:
                new_state_dict = deepcopy(state_dicts[-1])
                if pp_rank == 0:
                    new_state_dict['model']['language_model']['embedding'] = state_dicts[0]['model']['language_model']['embedding']
            else:
                new_state_dict = deepcopy(state_dicts[0])
            
            if block_start_orig == block_end_orig:
                for key in state_dicts[0]['model']['language_model']['encoder']:
                    if key =='final_layernorm.weight':
                        continue
                    if int(key.split('.')[1])<num_start or int(key.split('.')[1])>num_end:
                        continue
                    name = 'layers.'+ str(int(key.split('.')[1])-num_start)+'.'+'.'.join(key.split('.')[2:])
            
                    new_state_dict['model']['language_model']['encoder'][name] =  state_dicts[0]['model']['language_model']['encoder'][key]

                for key in new_state_dict['model']['language_model']['encoder']:
                    if key == 'final_layernorm.weight':
                        continue
                    elif int(key.split('.')[1])<target_pipeline_model_parallel_blocks[pp_rank]:
                        continue
                    else:
                        del new_state_dict['model']['language_model']['encoder'][key]
            else:
                consumed_layernum = 0
                for i in range(len(state_dicts)):
                    if i == 0:
                         for key in state_dicts[i]['model']['language_model']['encoder']:
                            if int(key.split('.')[1])<num_start:
                                continue
                            name = 'layers.'+ str(int(key.split('.')[1])-num_start)+'.'+'.'.join(key.split('.')[2:])
                            
                            new_state_dict['model']['language_model']['encoder'][name] =  state_dicts[i]['model']['language_model']['encoder'][key]
                         consumed_layernum +=pipeline_model_parallel_blocks[block_start_orig]-num_start
                    elif i==len(state_dicts)-1:
                        for key in state_dicts[i]['model']['language_model']['encoder']:
                            if key =='final_layernorm.weight':
                                continue
                            if int(key.split('.')[1])>num_end:
                                continue
                            name = 'layers.'+ str(int(key.split('.')[1])+consumed_layernum)+'.'+'.'.join(key.split('.')[2:])
                            new_state_dict['model']['language_model']['encoder'][name] =  state_dicts[i]['model']['language_model']['encoder'][key]

                    else:
                        for key in state_dicts[i]['model']['language_model']['encoder']:
                            name = 'layers.'+ str(int(key.split('.')[1])+consumed_layernum)+'.'+'.'.join(key.split('.')[2:])
                            new_state_dict['model']['language_model']['encoder'][name] =  state_dicts[i]['model']['language_model']['encoder'][key]
                        consumed_layernum += pipeline_model_parallel_blocks[block_start_orig+i]


            new_state_dict['args'].tensor_model_parallel_size = args.target_tensor_model_parallel_size
            #output state dict ckpt file
            iteration = state_dict['iteration']
            new_checkpoint_name = get_checkpoint_name(args.save, iteration)
            ensure_directory_exists(new_checkpoint_name)
            new_state_dict['args'].pipeline_model_parallel_size = args.target_pipeline_model_parallel_size
            torch.save(new_state_dict,new_checkpoint_name)

            #writing txt file
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                tracker_filename = get_checkpoint_tracker_filename(args.save)
                with open(tracker_filename,'w') as f:
                    f.write(str(iteration))

            if not args.no_load_optim and args.use_distributed_optimizer:

                new_optim_state_dict = deepcopy(optim_state_dicts[0])
                new_optim_state_dict[0][torch.float32]['param'] = None
                new_optim_state_dict[0][torch.float32]['exp_avg'] = None
                new_optim_state_dict[0][torch.float32]['exp_avg_sq'] = None

                increment = 0
                offset = 0

                if pp_rank == 0:
                    increment = 1
                    p = param_groupss[0][0]['params'][0]
                    new_optim_state_dict[0][torch.float32]['param'] = optim_state_dicts[0][0][torch.float32]['param'][offset:offset+p.numel()].clone().detach()
                    new_optim_state_dict[0][torch.float32]['exp_avg'] = optim_state_dicts[0][0][torch.float32]['exp_avg'][offset:offset+p.numel()].clone().detach()
                    new_optim_state_dict[0][torch.float32]['exp_avg_sq'] = optim_state_dicts[0][0][torch.float32]['exp_avg_sq'][offset:offset+p.numel()].clone().detach()

                    for i in range(len(param_groupss[0])):
                        sizes.append(len(param_groupss[0][i]['params'])//pipeline_model_parallel_blocks[0])

                for i in range(len(param_groupss[0])):
                    for j in range(len(optim_state_dicts)):
                        offset = 0
                        if j ==0:
                            if i == 0:
                                start_ind = num_start*sizes[i]+increment
                                if block_start_orig == block_end_orig:
                                    end_ind = increment+(num_end+1)*sizes[i]
                                else:
                                    end_ind = len(param_groupss[j][i]['params'])
                            else:
                                start_ind = num_start*sizes[i]
                                if block_start_orig == block_end_orig:
                                    end_ind = increment+(num_end+1)*sizes[i]
                                else:
                                    end_ind = len(param_groupss[j][i]['params'])

                        elif j== len(optim_state_dicts)-1:
                            start_ind = 0
                            end_ind = (num_end+1)*sizes[i]
                        else:
                            start_ind = 0
                            end_ind = len(param_groupss[j][i]['params'])

                        for ind in range(start_ind):
                            p = param_groupss[j][i]['params'][ind]
                            offset += p.numel()
                        if j==len(optim_state_dicts)-1 and pp_rank == args.target_pipeline_model_parallel_size-1:
                            end_ind = len(param_groupss[j][i]['params'])
                        for k in range(start_ind,end_ind):
                            p = param_groupss[j][i]['params'][k]
                            param_tensor = optim_state_dicts[j][0][torch.float32]['param'][offset:offset+p.numel()].clone().detach().flatten()
                            exp_avg_tensor = optim_state_dicts[j][0][torch.float32]['exp_avg'][offset:offset+p.numel()].clone().detach().flatten()
                            exp_avg_sq_tensor = optim_state_dicts[j][0][torch.float32]['exp_avg_sq'][offset:offset+p.numel()].clone().detach().flatten()
                            offset += p.numel()
                            

                            if new_optim_state_dict[0][torch.float32]['param'] == None:
                                new_optim_state_dict[0][torch.float32]['param'] = param_tensor
                                new_optim_state_dict[0][torch.float32]['exp_avg'] = exp_avg_tensor
                                new_optim_state_dict[0][torch.float32]['exp_avg_sq'] = exp_avg_sq_tensor
                            else:
                    
                                new_optim_state_dict[0][torch.float32]['param'] = torch.cat([new_optim_state_dict[0][torch.float32]['param'],param_tensor])
                                new_optim_state_dict[0][torch.float32]['exp_avg'] = torch.cat([new_optim_state_dict[0][torch.float32]['exp_avg'],exp_avg_tensor])
                                new_optim_state_dict[0][torch.float32]['exp_avg_sq'] = torch.cat([new_optim_state_dict[0][torch.float32]['exp_avg_sq'],exp_avg_sq_tensor])


                new_optim_checkpoint_name = get_distributed_optimizer_checkpoint_name(new_checkpoint_name)
                ensure_directory_exists(new_optim_checkpoint_name)
                torch.save(new_optim_state_dict, new_optim_checkpoint_name)
    print('done:)')




if __name__ == '__main__':
    main()

