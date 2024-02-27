#!/bin/bash

# Runs the "Yuan-2.1B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

#CHECKPOINT_PATH=/home/lx/Yuan-2.0/yuan2-2b-hf-to-megatron-fp8
CHECKPOINT_PATH=/home/lx/Yuan-2.0_fp8/2B_te
DATA_PATH=/home/lx/Yuan-2.0/data/train_sft/train_sample_2W.txt_document_context
TOKENIZER_MODEL_PATH=/home/lx/Yuan-2.0/tokenizer/tokenizer.model
TENSORBOARD_PATH=/home/lx/Yuan-2.0/tensorboard

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
    --timing-log-level 2 \
    --num-workers 2 \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 32 \
    --use-lf-gate \
    --lf-conv2d-group 1 \
    --lf-conv2d-num-pad 1 \
    --position-embedding-type rope \
    --no-embedding-dropout \
    --flash-attn-drop 0.1 \
    --fim-rate 0.5 \
    --fim-spm-rate 0.5 \
    --norm-dtype RMSNorm \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --disable-bias-linear \
    --reset-position-ids \
    --use-flash-attn \
    --swiglu \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 1 \
    --global-batch-size 384 \
    --lr 0.00001 \
    --train-iters 10 \
    --lr-decay-iters 10 \
    --lr-decay-style cosine \
    --min-lr 1.0e-6 \
    --weight-decay 1e-1 \
    --lr-warmup-iters 2 \
    --clip-grad 1.0 \
    --recompute-method uniform \
    --recompute-granularity full \
    --recompute-num-layers 1 \
    --bf16 \
    --sft-stage \
    --override-opt-param-scheduler \
    --train-reset \
    --finetune \
    --fp8-e4m3 \
    --transformer-impl transformer_engine \
    --fp8-amax-compute-algo max     
"
#--untie-embeddings-and-output-weights \
#    --no-load-args


DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type YuanTokenizer \
    --tokenizer-model-path $TOKENIZER_MODEL_PATH \
    --data-impl mmap \
    --split 10,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 100 \
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

torchrun $DISTRIBUTED_ARGS pretrain_yuan.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $LOG_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH

