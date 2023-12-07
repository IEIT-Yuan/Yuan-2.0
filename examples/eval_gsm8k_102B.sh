#!/bin/bash

# Runs the "Yuan-102B" parameter model inference

GPUS_PER_NODE=8
MAX_LENGTH=1024
MASTER_PORT=6000
MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

if [ "$TEMP" == "" ]; then
    TEMP=0
fi
if [ "$TOP_P" == "" ]; then
    TOP_P=0.0
fi
if [ "$TOP_K" == "" ]; then
    TOP_K=1
fi

CHECKPOINT_PATH=<Specify path>
TOKENIZER_MODEL_PATH=<Specify path>
MATH_DATA=<Specify path>
OUTPUT_PATH=<Specify path>

GPT_ARGS="
    --micro-batch-size 1 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --num-layers 84 \
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
    --no-async-tensor-model-parallel-allreduce \
    --bf16 \
    --temp $TEMP \
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

torchrun $DISTRIBUTED_ARGS tasks/GSM8K/eval_for_gsm8k.py \
       $GPT_ARGS \
       --tokenizer-type "YuanTokenizer" \
       --tokenizer-model-path $TOKENIZER_MODEL_PATH \
       --distributed-backend nccl \
       --num_samples_per_task 1 \
       --max_len $MAX_LENGTH \
       --output_path $OUTPUT_PATH \
       --math_datapath $MATH_DATA \
       --load $CHECKPOINT_PATH
