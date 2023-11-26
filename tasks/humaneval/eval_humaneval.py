import argparse
from human_eval.data import write_jsonl, read_problems
import torch, transformers
import math
import pdb
from tqdm import tqdm
import sys, os, pdb, re, json
from abc import ABC
from torch.utils.data import Dataset

sys.path.append('./')
from megatron import get_args
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.arguments import core_transformer_config_from_args
from megatron.text_generation import generate_and_post_process
from megatron.text_generation import beam_search_and_post_process
import torch
from transformers import LlamaTokenizer
from megatron import get_tokenizer
import re
from typing import Optional
import json

def parse_code_block(string: str, lang: str) -> Optional[str]:
    code_pattern = fr"```{lang}\n(.*?)\n```"
    match = re.search(code_pattern, string, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return parse_first_func(string, lang)


def parse_first_func(code: str, lang: str) -> Optional[str]:
    assert lang == "python", "Only python is supported for now. TODO: Rust"
    code_lines = code.split("\n")
    def_i = 0
    last_i = 0
    for i, line in enumerate(code_lines):
        if line.startswith("def "):
            if def_i == 0:
                def_i = i
            else:
                break
        if line == "" and def_i != 0:
            last_i = i
            break

    if last_i == 0:
        last_i = len(code_lines) - 1

    if def_i == 0:
        return None

    return "\n".join(code_lines[def_i:last_i+1])


def get_textbookprompt(task_id, prompt_lines):

    prompt = [line['prompt'] for line in prompt_lines if line['task_id'] == task_id][0]

    return prompt


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    config = core_transformer_config_from_args(get_args())

    print_rank_0('building GPT model ...')
    model = GPTModel(config, num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process)
    return model


def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument('--max_len', type=int, default=1024)
    group.add_argument('--human_eval_datapath', type=str, default='<Specify path>')
    group.add_argument('--textprompts_datapath', type=str, default='<Specify path>')
    group.add_argument('--output_path', type=str, default='<Specify path>')
    group.add_argument('--num_samples_per_task', type=int, default=1)
    group.add_argument('--top_k', type=int, default=0)
    group.add_argument('--top_p', type=float, default=0.95)
    group.add_argument('--top_p_decay', type=float, default=0.0)
    group.add_argument('--top_p_bound', type=float, default=0.0)
    group.add_argument('--temp', type=float, default=1)
    group.add_argument('--min_length', type=int, default=0)
    group.add_argument('--random_seed', type=int, default=1234)
    group.add_argument('--beam_width', type=int, default=None)
    group.add_argument('--length_penalty', type=int, default=1)
    group.add_argument('--eos_id', type=int, default=28956)
    group.add_argument('--prevent_newline_after_colon', type=bool, default=False)
    return parser

class HumanEvalDataset(ABC, Dataset):
    def __init__(self, data_path):
        self.problems = read_problems(data_path)
        self.keys = list(self.problems.keys())

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        try:
            key = self.keys[idx]
            sample = self.problems[key]
        except Exception as e:
            print(e, idx, len(self.problems))
            exit()
        return {'task_id':key, 'sample':sample}


def main():
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'YuanTokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()

    textprompts_path = args.textprompts_datapath
    textprompts_lines = open(textprompts_path,'r',encoding='utf-8').readlines()
    textprompts_lines = [json.loads(line) for line in textprompts_lines]

    dataset = HumanEvalDataset(args.human_eval_datapath)
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

    samples = []
    with torch.no_grad():
        with open(os.path.join(args.output_path, f'samples_{args.rank}.jsonl'), 'w') as fp:
            data_iter = tqdm(enumerate(data_loader), total=len(data_loader)) if torch.distributed.get_rank()==0 else enumerate(data_loader)
            for i, batch in data_iter:
                sample_iter = tqdm(range(args.num_samples_per_task), total=args.num_samples_per_task) if torch.distributed.get_rank()==0 else  range(args.num_samples_per_task)
                for j in sample_iter:
                    new_prompt = get_textbookprompt(task_id=batch['task_id'][0], prompt_lines=textprompts_lines)
                    if new_prompt is not None:
                        batch['sample']['prompt'] = [new_prompt]
                    tokens = tokenizer(batch['sample']['prompt'], return_tensors='pt', padding=True).input_ids[:,:-1].to(torch.cuda.current_device())
                    if args.beam_width is not None:
                        response, response_seg, response_scores = \
                            beam_search_and_post_process(
                            model,
                            prompts=batch['sample']['prompt'],
                            tokens_to_generate=(args.max_len - len(tokens)),
                            beam_size = args.beam_width,
                            add_BOS=False,
                            stop_token=stop_token,
                            num_return_gen=args.beam_width,  # Returning whole beam
                            length_penalty=args.length_penalty,
                            prevent_newline_after_colon=args.prevent_newline_after_colon
                            )
                    else:
                        response, response_seg, response_logprobs, _ = \
                            generate_and_post_process(
                            model,
                            prompts=batch['sample']['prompt'],
                            tokens_to_generate=(args.max_len - len(tokens)),
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

                        if response[0][-5:]=='<eod>':
                            if response[0][0]==' ':
                                response = [response[0][1:-5]]
                            else:
                                response = [response[0][0:-5]]
                        if new_prompt is not None:
                            new_sample = [response[0][response[0].find("```python")+9:]]
                        else:
                            new_sample = response
                        print('\n\n')
                        print(response[0])
                        print('-----------------------------------------------------------')
                        for k, x in enumerate(new_sample):
                            x = x.replace('\nFIX = """\nAdd more test cases.\n"""','')
                            x = x.replace('\n\n\ndef','\ndef')
                            x = x.replace('\n    \n\n    ','\n    \n    ')
                            x = x.replace('\n\n    ','\n    ')
                            x = x.replace('\n\n\n','\n\n')
                            x = x.replace('"""\n        ','"""\n    ')
                            x = x.replace('\n```python','')
                            if x.count('"""') >= 2:
                                x = x.replace(x[x.find('"""'):x.find('"""',x.find('"""')+1)+3],'')
                            elif x.count("'''") >= 2:
                                x = x.replace(x[x.find("'''"):x.find("'''",x.find("'''")+1)+3],'')
                            x_func = parse_code_block(x, "python")
                            if x_func is not None:
                                x_func = x[0:x.find(x_func[0:10])]+x_func
                            else:
                                x_func = x
                            if x_func.count('\ndef') > 1 and x_func.count('\n```')<1 and x_func.count('\n#')<1 and x_func.count('\nif')<1:
                                x_func = x_func.replace(x_func[x_func.find('\ndef',x_func.find('\ndef')+1):],'\n')
                            if x_func.find('\nprint')>0:
                                x_func = x_func[0:x_func.find('\nprint')]
                            if x_func.find('\n# 示例')>0:
                                x_func = x_func[0:x_func.find('\n# 示例')]
                            if x_func.find('\n```')>0:
                                x_func = x_func[0:x_func.find('\n```')]
                            if x_func.find('\n# 测试')>0:
                                x_func = x_func[0:x_func.find('\n# 测试')]
                            if x_func.find('\n# 单元测试')>0:
                                x_func = x_func[0:x_func.find('\n# 单元测试')]
                            if x_func.find('\n#')>0:
                                x_func = x_func[0:x_func.find('\n#')]
                            if x_func.find('\nif')>0:
                                x_func = x_func[0:x_func.find('\nif')]
                            print(x_func)
                            x = dict(task_id=batch['task_id'][k], completion=x_func)
                            fp.write(json.dumps(x) + "\n")
                            fp.flush()
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        with open(os.path.join(args.output_path, f'samples.jsonl'), 'w') as fp:
            total_dict = {}
            for rank in range(args.world_size):
                with open(os.path.join(args.output_path, f'samples_{rank}.jsonl'), 'r') as fin:
                    for line in fin.readlines():
                        data_dict = json.loads(line)
                        task_id = data_dict['task_id']
                        if task_id not in total_dict:
                            total_dict[task_id] = [data_dict]
                        else:
                            if len(total_dict[task_id]) >= args.num_samples_per_task:
                                continue
                            total_dict[task_id].append(data_dict)
            for key in total_dict:
                 for i in range(args.num_samples_per_task):
                     fp.write(json.dumps(total_dict[key][i]) + '\n')

if __name__ == '__main__':
    main()
