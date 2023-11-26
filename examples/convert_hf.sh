#!/bin/bash

# Runs the "2.1B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=<Specify path>
CHECKPOINT_PATH_SAVE=<Specify path>
LOG_PATH=<Specify path>
TOKENIZERPATH=<Specify path>
TENSORBOARD_PATH=<Specify path>

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --num-layers 24 \
    --hidden-size 2048 \
    --use-lf-gate \
    --lf-conv2d-group 1 \
    --position-embedding-type rope \
    --no-embedding-dropout \
    --flash-attn-drop 0.1 \
    --fim-rate 0.0 \
    --fim-spm-rate 0.5 \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --norm-dtype RMSNorm \
    --disable-bias-linear \
    --reset-position-ids \
    --use-flash-attn \
    --swiglu \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --num-attention-heads 32 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --micro-batch-size 2 \
    --global-batch-size 512 \
    --lr 0.0002 \
    --train-iters 16384 \
    --lr-decay-iters 16384 \
    --lr-decay-style cosine \
    --min-lr 2.0e-5 \
    --weight-decay 1e-1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --lr-warmup-iters 100 \
    --clip-grad 1.0 \
    --bf16
"
DATA_ARGS="
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 10,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --vocab-file $VOCAB_FILE \
    --save-interval 10000 \
    --eval-interval 1000000 \
    --eval-iters 10
"
LOG_ARGS="
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-log-interval 1 \
    --tensorboard-queue-size 1000 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-memory-to-tensorboard \
    --log-world-size-to-tensorboard
"


torchrun $DISTRIBUTED_ARGS tools/convert_hf.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $LOG_ARGS \
    --tokenizer-type "YuanTokenizer" \
    --tokenizer-model-path $TOKENIZERPATH \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH_SAVE \
    --load $CHECKPOINT_PATH

