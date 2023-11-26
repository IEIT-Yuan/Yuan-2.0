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
    TEMP=1
fi
if [ "$TOP_P" == "" ]; then
    TOP_P=0
fi
if [ "$TOP_K" == "" ]; then
    TOP_K=1
fi

CHECKPOINT_PATH=<Specify path>
TOKENIZER_MODEL_PATH=<Specify path>
TruthfulQA_DATA=<Specify path>
OUTPUT_PATH=<Specify path>
TOKENS_TO_GEN=15
mkdir -p $OUTPUT_PATH

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
GPT_ARGS="
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --timing-log-level 2 \
    --num-layers 84 \
    --hidden-size 8192 \
    --use-lf-gate \
    --lf-conv2d-group 1 \
    --lf-conv2d-num-pad 0 \
    --position-embedding-type rope \
    --no-embedding-dropout \
    --flash-attn-drop 0.1 \
    --fim-rate 0 \
    --fim-spm-rate 0 \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --norm-dtype RMSNorm \
    --disable-bias-linear \
    --reset-position-ids \
    --use-flash-attn \
    --swiglu \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --max-position-embeddings 8192 \
    --no-async-tensor-model-parallel-allreduce \
    --bf16 \
    --temp $TEMP \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --seed $RANDOM
"
torchrun $DISTRIBUTED_ARGS ./tasks/TruthfulQA/eval_truthfulqa.py \
    $GPT_ARGS \
    --datapath $TruthfulQA_DATA \
    --tokenizer-type "YuanTokenizer" \
    --tokenizer-model-path $TOKENIZER_MODEL_PATH \
    --distributed-backend nccl \
    --num_samples_per_task 1 \
    --tokens_to_gen $TOKENS_TO_GEN \
    --max_len $MAX_LENGTH \
    --output_path $OUTPUT_PATH \
    --load $CHECKPOINT_PATH

