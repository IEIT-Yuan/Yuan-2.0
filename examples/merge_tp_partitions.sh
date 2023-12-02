#!/bin/bash

#merge checkpoint along the tensor


LOAD_CHECKPOINT_PATH=<Specify the loaded ckpt path>
SAVE_CHECKPOINT_PATH=<Specify the stored ckpt path >
TOKENIZER_MODEL_PATH=<Specify tokenizer model path>
export CUDA_DEVICE_MAX_CONNECTIONS=1

python tools/merge_tp_partitions.py \
    --tensor-model-parallel-size 2 \
    --target-tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 4 \
    --target-pipeline-model-parallel-size 4 \
    --memorybuffer-device None \
    --tokenizer-type YuanTokenizer \
    --tokenizer-model-path $TOKENIZER_MODEL_PATH \
    --num-layers 42 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --use-lf-gate \
    --lf-conv2d-group 1 \
    --lf-conv2d-num-pad 1 \
    --position-embedding-type rope \
    --no-embedding-dropout \
    --flash-attn-drop 0.1 \
    --fim-rate 0.5 \
    --fim-spm-rate 0.5 \
    --attention-dropout 0 \
    --norm-dtype RMSNorm \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --disable-bias-linear \
    --reset-position-ids \
    --use-flash-attn \
    --swiglu \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --bf16 \
    --DDP-impl local \
    --use-cpu-initialization \
    --micro-batch-size 1 \
    --save-interval 1 \
    --recompute-method block \
    --recompute-granularity full \
    --recompute-num-layers 1 \
    --load $LOAD_CHECKPOINT_PATH \
    --save $SAVE_CHECKPOINT_PATH \
    --micro-batch-size 1 \
    --global-batch-size 1152 \
    --lr 0.00009 \
    --train-iters 63578 \
    --lr-decay-iters 63578 \
    --lr-decay-style cosine \
    --min-lr 1.8e-5 \
    --weight-decay 1e-1 \
    --no-load-optim \
    --use-distributed-optimizer
    
du -sh $SAVE_CHECKPOINT_PATH
