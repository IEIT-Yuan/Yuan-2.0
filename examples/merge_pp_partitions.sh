#!/bin/bash

#merge checkpoint along the pipeline


LOAD_CHECKPOINT_PATH=<Specify the loaded ckpt path>
SAVE_CHECKPOINT_PATH=<Specify the stored ckpt path>
TOKENIZER_MODEL_PATH=<Specify tokenizer model path>
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PATH=/opt/conda/bin/:$PATH

python tools/merge_pp_partitions.py \
    --model-type GPT \
    --tokenizer-model-path $TOKENIZER_MODEL_PATH \
    --tensor-model-parallel-size 8 \
    --target-tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 16 \
    --target-pipeline-model-parallel-size 1 \
    --pipeline-model-parallel-method block \
    --pipeline-model-parallel-blocks  2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,2 \
    --target-pipeline-model-parallel-blocks 42 \
    --tensor-generate-layer 0,1 \
    --tokenizer-type YuanTokenizer \
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
    --hidden-dropout 0 \
    --norm-dtype RMSNorm \
    --disable-bias-linear \
    --reset-position-ids \
    --use-flash-attn \
    --swiglu \
    --DDP-impl local \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --bf16 \
    --save-interval 1 \
    --recompute-method block \
    --recompute-granularity full \
    --recompute-num-layers 1 \
    --load $LOAD_CHECKPOINT_PATH \
    --save $SAVE_CHECKPOINT_PATH \
    --micro-batch-size 1 \
    --global-batch-size 1152 \
    --use-distributed-optimizer \
    --lr 0.00009 \
    --train-iters 63578 \
    --lr-decay-iters 63578 \
    --lr-decay-style cosine \
    --min-lr 0.9e-5 \
    --weight-decay 1e-1 \
    --no-load-optim \
    --use-distributed-optimizer \
    --use-cpu-initialization \
    --memorybuffer-device None \
    --data-impl mmap
du -sh $SAVE_CHECKPOINT_PATH

