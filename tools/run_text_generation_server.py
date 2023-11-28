# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate Yuan"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import socket
from megatron import get_args
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron, _set_random_seed
from megatron.model import YuanModel
from megatron.training import get_model
from megatron.arguments import core_transformer_config_from_args
from megatron.text_generation_server import MegatronServer
from megatron.text_generation import generate_and_post_process
from megatron.text_generation import beam_search_and_post_process
from megatron.core.parallel_state import get_tensor_model_parallel_group
import torch
from megatron import get_tokenizer
import time
import math


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    config = core_transformer_config_from_args(get_args())

    print_rank_0('building Yuan model ...')
    model = YuanModel(config, num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process)

    return model


def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument('--top_p_decay', type=float, default=0.0)
    group.add_argument('--top_p_bound', type=float, default=0.0)
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')

    group.add_argument('--min_length', type=int, default=0)
    group.add_argument('--random_seed', type=int, default=1234)
    group.add_argument('--beam_width', type=int, default=None)
    group.add_argument('--length_penalty', type=int, default=1)
    group.add_argument('--prevent_newline_after_colon', type=bool, default=False)

    return parser


if __name__ == "__main__":
    t = time.time()
    seed = int(1000 * (math.ceil(t) - t))
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'YuanTokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()
    _set_random_seed(abs(seed))  #add random seed

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for text generation.")
    args.exit_on_missing_checkpoint = True
    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    # add a new pad token
    tokenizer = get_tokenizer()
    tokenizer.add_eos_token = False
    tokenizer.add_bos_token = False
    tokenizer.eod = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    stop_token = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    # pdb.set_trace()
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    torch.distributed.barrier()

    #  new add
    model.eval()

    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        server = MegatronServer(model)
        server.run("0.0.0.0", port=int(os.environ.get("PORT", 8900)))

    while True:
        choice = torch.cuda.LongTensor(1)
        torch.distributed.broadcast(choice, 0)
        if choice[0].item() == 0:
            try:
                generate_and_post_process(model)
            except ValueError as ve:
                pass
        elif choice[0].item() == 1:
            try:
                beam_search_and_post_process(model)
            except ValueError as ve:
                pass
