# coding=utf-8
import argparse
import torch, transformers
import math
from tqdm import tqdm
import sys, os, pdb, re, json
from abc import ABC
from torch.utils.data import Dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
sys.path.append('./')
from megatron import get_args
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.arguments import core_transformer_config_from_args
from megatron.text_generation_server import MegatronServer
from megatron.text_generation import generate_and_post_process
from megatron.text_generation import beam_search_and_post_process
import torch
from megatron import get_tokenizer


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    config = core_transformer_config_from_args(get_args())
    print_rank_0('building GPT model ...')
    model = GPTModel(config, num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process)
    return model


def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument('--max_len', type=int, default=1024)
    group.add_argument('--tokens_to_gen', type=int, default=20)
    group.add_argument('--datapath', type=str, default=None)
    group.add_argument('--output_path', type=str, default=None)
    group.add_argument('--num_samples_per_task', type=int, default=10)
    group.add_argument('--top_k', type=int, default=0)
    group.add_argument('--top_p', type=float, default=0.95)
    group.add_argument('--top_p_decay', type=float, default=0.0)
    group.add_argument('--top_p_bound', type=float, default=0.0)
    group.add_argument('--temp', type=float, default=0.5)
    group.add_argument('--min_length', type=int, default=0)
    group.add_argument('--random_seed', type=int, default=1234)
    group.add_argument('--beam_width', type=int, default=None)
    group.add_argument('--voting', action='store_true')
    group.add_argument('--length_penalty', type=int, default=1)
    group.add_argument('--total_rank', type=int, default=1)
    group.add_argument('--cur_rank', type=int, default=0)
    group.add_argument('--prevent_newline_after_colon', type=bool, default=False)
    return parser

class TruthfulQADataset(ABC, Dataset):
    def __init__(self, data_path):
        args = get_args()
        self.querys = []
        self.keys = []
        self.answers = []
        with open(data_path, 'r') as f:
            lines = f.readlines()
            idx = 0
            for i, l in enumerate(lines):
                if i % args.total_rank != args.cur_rank:
                    continue
                data = eval(l)
                query = data['query']
                prompt = '以上哪个选项是正确的？如果选项A正确则回复“正确答案是A选项”，如果选项B正确则回复“正确答案是B选项”，其余选项也是如此。'
                alphabeta = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                choice = ''
                for ii, key in enumerate(data.keys()):
                    if key == 'query':
                        continue
                    choice = choice + alphabeta[ii-1] + '. ' + data[key]
                query = query + '\n' + choice + '\n' + prompt + '<sep>'
                self.answers.append(data['ans1'])
                self.keys.append(idx)
                self.querys.append(query)
                idx += 1

    def __len__(self):
        return len(self.querys)

    def __getitem__(self, idx):
        try:
            key = self.keys[idx]
            query = self.querys[key]
            ans = self.answers[key]
        except Exception as e:
            print(e, idx, key, len(self.querys))
            exit()
        return {'task_id':key, 'query':query, 'answer': ans}

def main():
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'YuanTokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()
    tokenizer = get_tokenizer()
    tokenizer.add_tokens(['<sep>','<pad>','<mask>','<predict>','<FIM_SUFFIX>','<FIM_PREFIX>','<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens = True)
    dataset = TruthfulQADataset(args.datapath)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=mpu.get_data_parallel_rank(), num_replicas = mpu.get_data_parallel_world_size(), shuffle=False, drop_last=False)
    data_loader = torch.utils.data.DataLoader(dataset,
            batch_size=args.micro_batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=2)
    model = get_model(model_provider, wrap_with_ddp=False)
    if args.load is not None:
        _ = load_checkpoint(model, None, None)
    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    tokenizer = get_tokenizer()
    tokenizer.add_eos_token = False
    tokenizer.add_bos_token = False
    tokenizer.eod = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    stop_token = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    torch.distributed.barrier()

    model.eval()
    if args.fp16:
        model = model.half()
    elif args.bf16:
        model = model.bfloat16()
    else:
        model = model.float()
    model.cuda()
    torch.distributed.barrier()
    if torch.distributed.get_rank()==0 and not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    with torch.no_grad():
        if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
            fp = open(os.path.join(args.output_path, 'samples_{}.jsonl'.format(args.cur_rank)), 'w')
        data_iter = tqdm(enumerate(data_loader), total=len(data_loader)) if torch.distributed.get_rank()==0 else enumerate(data_loader)
        for i, batch in data_iter:
            sample_iter = range(args.num_samples_per_task)
            for j in sample_iter:
                inputs = []
                for q in batch['query']:
                    inputs.append(q)
                tokens = tokenizer(inputs, return_tensors='pt', padding=True).input_ids[:,:-1].to(torch.cuda.current_device())
                if len(tokens[0]) + args.tokens_to_gen >= args.max_position_embeddings:
                    tokens = tokens[:, -(args.max_position_embeddings - args.tokens_to_gen - 10):]
                if args.beam_width is not None:
                    response, response_seg, response_scores = \
                        beam_search_and_post_process(
                        model,
                        prompts=inputs,
                        tokens_to_generate=(args.max_len - len(tokens)),
                        beam_size = args.beam_width,
                        add_BOS=False,
                        stop_token=stop_token,
                        num_return_gen=args.beam_width,
                        length_penalty=args.length_penalty,
                        prevent_newline_after_colon=args.prevent_newline_after_colon)
                else:
                    args.random_seed += 1
                    response, response_seg, response_logprobs, _ = \
                        generate_and_post_process(
                        model,
                        prompts=inputs,
                        tokens_to_generate=args.tokens_to_gen,
                        return_output_log_probs=False,
                        top_k_sampling=args.top_k,
                        top_p_sampling=args.top_p,
                        top_p_decay=args.top_p_decay,
                        top_p_bound=args.top_p_bound,
                        temperature=args.temp,
                        add_BOS=False,
                        stop_on_eol=False,
                        prevent_newline_after_colon=args.prevent_newline_after_colon,
                        random_seed=args.random_seed)
                if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
                    new_sample = response
                    for k, x in enumerate(new_sample):
                        query = batch['query'][k]
                        r_ans = batch['answer'][k]
                        gen_ans = x
                        result = {'query': query, 'ans': r_ans, 'gen_ans':gen_ans}
                        fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                        fp.flush()
        if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_first_stage:
            fp.close()
    torch.distributed.barrier()

if __name__ == '__main__':
    main()
