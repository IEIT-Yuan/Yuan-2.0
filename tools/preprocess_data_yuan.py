#-*- coding : utf-8-*-
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

"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import sys
import re

sys.path.append('./')
import time
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from transformers import LlamaTokenizer
from megatron.data import indexed_dataset

random.seed(233)
np.random.seed(233)
g = torch.manual_seed(233)
torch.cuda.manual_seed_all(233)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = LlamaTokenizer.from_pretrained(self.args.tokenizer_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>',clean_up_tokenization_spaces=True)
        Encoder.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens = True)

        Encoder.splitter = IdentitySplitter()

    def encode(self, line):

        if line=='\n':
            return None, 0

        if line[:3] == '<n>':
            data = line[3:].strip().replace('\r','').replace('\n','')
        else:
            data = line.strip().replace('\r','').replace('\n','')
        data = data.replace("<n>","\n").replace('▃', '')
        data = data.replace('[SEP]','<sep>')
        if re.sub(r'(\n)+','',data) == '':
            return None, 0
        if not self.args.sentence_splitter:
            doc_ids = Encoder.tokenizer.encode(data)
            doc_ids.append(Encoder.tokenizer.eos_token_id)
        else:
            data = data.split(self.args.sentence_splitter)
            data = [item.strip() for item in data]
            data = [ item for item in data if item]
            doc_ids = [ Encoder.tokenizer.encode(item) for item in data]
            doc_ids[-1].append(Encoder.tokenizer.eos_token_id)
        return doc_ids, len(line)

    def random_spans_noise_mask(self, length, noisy_density=0.15, mean_noise_span_length=10.0):
        num_noise_tokens = round(length * noisy_density)
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        def random_segment(seq_length, num_segment):
            x = (torch.arange(seq_length - 1) < (num_segment - 1)).long()
            a = torch.randperm(seq_length - 1, generator=g)
            x = x[a]
            x = F.pad(x, [1, 0])
            segment_id = torch.cumsum(x, dim=0)
            segment_lengths = torch.zeros(num_segment, dtype=torch.long).scatter_add_(0, segment_id, torch.ones(seq_length, dtype=torch.long))

            return segment_lengths

        noise_span_lengths = random_segment(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = random_segment(num_nonnoise_tokens, num_noise_spans)
        interleaved_span_lengths = torch.stack([nonnoise_span_lengths, noise_span_lengths], dim=1).view(num_noise_spans * 2)
        span_start_ends = torch.cumsum(interleaved_span_lengths, dim=0).view(-1, 2)
        return span_start_ends.tolist()


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', default="<Specify path>", type=str, help='Path to input TXT')
    group.add_argument('--data-idx', default=None, type=str, help='the idx split')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer_path', default="../tokenizer", type=str, help='Path of tokenizer')

    group = parser.add_argument_group(title='output data')
    group.add_argument("--output_path", default="<Specify path>", type=str)
    group.add_argument('--output_prefix', default="yuan", type=str,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset_impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=36,
                       help='Number of worker processes to launch')
    group.add_argument('--log_interval', type=int, default=10000,
                       help='Interval between progress updates')
    group.add_argument('--sentence_splitter',type=bool, default=False)
    group.add_argument('--mean_noise_span_length', type=int, default=3)

    args = parser.parse_args()
    args.keep_empty = False

    args.rank = 0
    args.make_vocab_size_divisible_by = 128

    return args

def getfiles(path,ex_str='.py'):
    # 获取文件名
    file_list=[]
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if(len(ex_str)>0 and ex_str in name ):
                continue
            file_list.append(os.path.join(root,name))
    return file_list

def main():
    args = get_args()

    startup_start = time.time()
    fin_list=getfiles(args.input)
    output_dir = Path(args.output_path)
    if not output_dir.is_dir():
        os.makedirs(args.output_path)
    if isinstance(args.data_idx.split('-'), list) and len(args.data_idx.split('-'))>1:
        idx = [int(s) for s in args.data_idx.split('-')]
        idx = list(range(idx[0],idx[1]))
    else:
        idx = [int(args.data_idx)]
    print(idx,'/', len(fin_list))

    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>',clean_up_tokenization_spaces=True)
    tokenizer.add_tokens(['<sep>','<pad>','<mask>','<predict>','<FIM_SUFFIX>','<FIM_PREFIX>','<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens = True)

    for i in idx:

        fin_path = fin_list[i]
        print("Opening", fin_path)
        if args.sentence_splitter:
            level = "sentence"
        else:
            level = "document"

        fin_name = fin_path.split('/')[-1]
        context_bin_file = os.path.join(args.output_path, "{}_{}_context.bin".format(fin_name, level))
        context_idx_file = os.path.join(args.output_path,  "{}_{}_context.idx".format(fin_name, level))

        if not os.path.exists(fin_path)  or os.path.exists(context_bin_file):
            continue
        elif not os.path.exists(context_idx_file) and os.path.exists(context_bin_file):
            os.remove(context_bin_file)
        fin = open(fin_path, 'r', encoding='utf-8',errors='ignore')

        encoder = Encoder(args)
        pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)

        # use the tokenizer to encode the sentences
        encoded_docs = pool.imap_unordered(encoder.encode, fin, 30)

        if args.sentence_splitter:
            level = "sentence"
        else:
            level = "document"

        print(f"Vocab size: {tokenizer.vocab_size}")
        print(f"Output prefix: {args.output_prefix}")

        fin_name = fin_path.split('/')[-1]
        context_bin_file = os.path.join(args.output_path, "{}_{}_context.bin".format(fin_name, level))
        context_idx_file = os.path.join(args.output_path,  "{}_{}_context.idx".format(fin_name, level))

        if os.path.exists(context_idx_file):
            continue

        builder_context = indexed_dataset.make_builder(context_bin_file, impl=args.dataset_impl)

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)
        print("tokenizer vocab size:", tokenizer.vocab_size)
        total_tokens=0
        for i, (no_noise_tokens, bytes_processed) in enumerate(encoded_docs, start=1):
            if no_noise_tokens is None :
                continue
            total_tokens+=len(no_noise_tokens)
            total_bytes_processed += bytes_processed
            if level == "document":
                builder_context.add_item(torch.IntTensor(no_noise_tokens))
            if level == "sentence":
                for key, sentence in enumerate(no_noise_tokens):
                    if len(sentence) == 0:
                        continue
                    builder_context.add_item(torch.IntTensor(sentence))
                builder_context.end_document()

            if i % args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {i} documents",
                      f"({i/elapsed} docs/s, {mbs} MB/s).",
                      file=sys.stderr)

        builder_context.finalize(context_idx_file)
        print("Total time to used:", time.time() - startup_start)
        pool.close()
        print("total tokens: ",total_tokens )

if __name__ == '__main__':
    main()
