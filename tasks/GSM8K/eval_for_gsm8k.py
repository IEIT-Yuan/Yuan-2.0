import sys
import os
import torch
from abc import ABC
from tqdm import tqdm
from torch.utils.data import Dataset

sys.path.append('./')
from megatron import get_args
from megatron.core import mpu
from megatron import get_tokenizer
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.arguments import core_transformer_config_from_args
from megatron.text_generation import generate_and_post_process
from megatron.text_generation import beam_search_and_post_process


def model_provider(pre_process=True, post_process=True):
    config = core_transformer_config_from_args(get_args())
    model = GPTModel(config, num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process)
    return model

def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument('--max_len', type=int, default=1024)
    group.add_argument('--model_config_path', type=str, default='./')
    group.add_argument('--math_datapath', type=str, default='./')
    group.add_argument('--output_path', type=str, default='./')
    group.add_argument('--num_samples_per_task', type=int, default=10)
    group.add_argument('--top_k', type=int, default=0)
    group.add_argument('--top_p', type=float, default=0.95)
    group.add_argument('--top_p_decay', type=float, default=0.0)
    group.add_argument('--top_p_bound', type=float, default=0.0)
    group.add_argument('--temp', type=float, default=0.5)
    group.add_argument('--min_length', type=int, default=0)
    group.add_argument('--random_seed', type=int, default=1234)
    group.add_argument('--beam_width', type=int, default=None)
    group.add_argument('--length_penalty', type=int, default=1)
    group.add_argument('--prevent_newline_after_colon', type=bool, default=False)
    return parser

def clean_tab(msg_text):
    __sep_note = "<n>"
    msg_text = msg_text.replace("\n", __sep_note)
    msg_text = msg_text.replace(__sep_note + __sep_note, __sep_note)
    msg_text = msg_text.replace(__sep_note + __sep_note, __sep_note)
    msg_text = msg_text.replace(__sep_note + __sep_note, __sep_note)
    return msg_text

class EvalDataset(ABC, Dataset):
    def __init__(self, data_path):
        self.problems = []
        self.keys = []
        self.answers = []

        with open(data_path, 'r') as f:
            lines = f.readlines()
            for ii, line in enumerate(lines):
                line = line.strip()
                gsm8k_prompt = "详细分析并求解以下数学问题。\n"
                index = line.find('[SEP]')
                line = gsm8k_prompt + line[:index] + '<sep>'
                self.problems.append(line)
                self.keys.append(ii)
                self.answers.append('')

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
    dataset = EvalDataset(args.math_datapath)
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
        os.makedirs(args.output_path)

    with torch.no_grad():
        data_iter = tqdm(enumerate(data_loader), total=len(data_loader)) if torch.distributed.get_rank()==0 else enumerate(data_loader)
        for i, batch in data_iter:
            sample_iter = tqdm(range(args.num_samples_per_task), total=args.num_samples_per_task) if torch.distributed.get_rank()==0 else  range(args.num_samples_per_task)
            for j in sample_iter:
                def inference_once(top_k=None, top_p=None, temp=None, seed=None):
                    tokens = tokenizer(batch['sample'], return_tensors='pt', padding=True).input_ids[:,:-1].to(torch.cuda.current_device())
                    if args.beam_width is not None:
                        response, response_seg, response_scores = \
                            beam_search_and_post_process(
                            model,
                            prompts=batch['sample'],
                            tokens_to_generate=(args.max_len - len(tokens)),
                            beam_size = args.beam_width,
                            add_BOS=False,
                            stop_token=stop_token,
                            num_return_gen=args.beam_width,
                            length_penalty=args.length_penalty,
                            prevent_newline_after_colon=args.prevent_newline_after_colon
                            )
                    else:
                        response, response_seg, response_logprobs, _ = \
                            generate_and_post_process(
                            model,
                            prompts=batch['sample'],
                            tokens_to_generate=(args.max_len - len(tokens)),
                            return_output_log_probs=False,
                            top_k_sampling=top_k,
                            top_p_sampling=top_p,
                            top_p_decay=args.top_p_decay,
                            top_p_bound=args.top_p_bound,
                            temperature=temp,
                            add_BOS=False,
                            stop_on_eol=False,
                            prevent_newline_after_colon=args.prevent_newline_after_colon,
                            random_seed=seed)

                    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
                        if response[0][0]==' ':
                            response = [response[0][1:-5]]
                        else:
                            response = [response[0][0:-5]]
                        new_sample = response
                        print('\n' + response[0])

                        with open(os.path.join(args.output_path, f'samples_{args.rank}.jsonl'), 'a', encoding='utf-8') as fp:
                            for _, x in enumerate(new_sample):
                                res = x.strip()
                                res = res.replace('<pad>', '')
                                res = res.replace('<eod>', '')
                                res = res.replace('<sep>', '[SEP]')
                                res = clean_tab(res)
                                record_res = res.strip() + '\n'
                                fp.write(record_res)
                inference_once(top_k=args.top_k, top_p=args.top_p, temp=args.temp, seed=args.random_seed)
              
    torch.distributed.barrier()


if __name__ == '__main__':
    main()
