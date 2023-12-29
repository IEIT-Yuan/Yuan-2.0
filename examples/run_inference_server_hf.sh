#!/bin/bash

HF_PATH=<Specify path>   #本地地址，已下载模型存放地址
#HF_PATH="IEITYuan/Yuan2-2B-hf"  #从huggingface加载模型

ARGS="
    --do_sample false \
    --max_length 8192 \
    --max-position-embeddings 8192 \
    --num_beams 1 \
    --bf16 \
    --temperature 1 \
    --top_k 1
"

CUDA_VISIBLE_DEVICES=0 PORT=8000 python tools/run_text_generation_server_hf.py   \
       $GPT_ARGS \
       --load $HF_PATH
