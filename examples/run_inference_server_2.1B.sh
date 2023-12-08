#!/bin/bash

# Runs the "Yuan-2.1B" parameter model
NNODES=1
MASTER_PORT=12308

TOKENIZER_MODEL_PATH=./tokenizer
CHECKPOINT_PATH=<Specify path>


GPT_ARGS="
    --micro-batch-size 1 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 2048 \
    --use-lf-gate \
    --lf-conv2d-group 1 \
    --lf-conv2d-num-pad 0 \
    --position-embedding-type rope \
    --no-embedding-dropout \
    --flash-attn-drop 0.0 \
    --attention-dropout 0 \
    --fim-rate 0.5 \
    --hidden-dropout 0 \
    --norm-dtype RMSNorm \
    --disable-bias-linear \
    --reset-position-ids \
    --swiglu \
    --num-attention-heads 32 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --no-async-tensor-model-parallel-allreduce \
    --bf16 \
    --temperature 1 \
    --top_p 0 \
    --top_k 5 \
    --seed $RANDOM
"


CUDA_VISIBLE_DEVICES=0 PORT=8000 NCCL_TIMEOUT=36000000 torchrun --nproc_per_node 1 --master_addr localhost --node_rank 0 --nnodes 1 --master_port $MASTER_PORT tools/run_text_generation_server.py  \
       $GPT_ARGS \
       --tokenizer-type "YuanTokenizer" \
       --inference-server \
       --tokenizer-model-path $TOKENIZER_MODEL_PATH \
       --distributed-backend nccl \
       --load $CHECKPOINT_PATH 

