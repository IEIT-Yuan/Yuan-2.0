#!/bin/bash

# Runs the "Yuan-102B" parameter model
export CUDA_DEVICE_MAX_CONNECTIONS=1
GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6074
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

if [ "$TEMP" == "" ]; then
    TEMP=1
fi
if [ "$TOP_P" == "" ]; then
    TOP_P=0.0
fi
if [ "$TOP_K" == "" ]; then
    TOP_K=1
fi

TOKENIZER_MODEL_PATH=<Specify path to file>
CHECKPOINT_PATH=<Specify path>

GPT_ARGS="
    --micro-batch-size 1 \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --num-layers 42 \
    --distributed-timeout-minutes 120 \
    --hidden-size 8192 \
    --use-lf-gate \
    --lf-conv2d-group 1 \
    --lf-conv2d-num-pad 0 \
    --position-embedding-type rope \
    --no-embedding-dropout \
    --use-flash-attn \
    --flash-attn-drop 0.0 \
    --attention-dropout 0 \
    --fim-rate 0.0 \
    --hidden-dropout 0 \
    --norm-dtype RMSNorm \
    --disable-bias-linear \
    --reset-position-ids \
    --swiglu \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --bf16 \
    --temperature $TEMP \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --seed $RANDOM
"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=8000 NCCL_IB_TIMEOUT=22 NCCL_TIMEOUT=60000000000 torchrun $DISTRIBUTED_ARGS tools/run_text_generation_server.py \
       $GPT_ARGS \
       --tokenizer-type "YuanTokenizer" \
       --inference-server \
       --tokenizer-model-path $TOKENIZER_MODEL_PATH \
       --distributed-backend nccl \
       --load $CHECKPOINT_PATH
