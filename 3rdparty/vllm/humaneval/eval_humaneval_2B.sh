#!/bin/bash

# Runs the "Yuan-2B" parameter model inference


if [ "$NODE_RANK" == "" ]; then
    NODE_RANK=0
fi
if [ "$MASTER_ADDR" == "" ]; then
    MASTER_ADDR=localhost
fi
if [ "$NNODES" == "" ]; then
    NNODES=1
fi
if [ "$NUM_GPUS" == "" ]; then
    NUM_GPUS=1
fi
if [ "$TEMP" == "" ]; then
    TEMP=1
fi
if [ "$TOP_P" == "" ]; then
    TOP_P=0.0
fi
if [ "$TOP_K" == "" ]; then
    TOP_K=1
fi
if [ "$DATASET" == "" ]; then
    DATASET=HumanEval.jsonl.gz
fi

WORLD_SIZE=$(($NUM_GPUS*$NNODES))
if [ "$CASE_NAME" == "" ]; then
    CASE_NAME=test-2B-again
fi
#MASTER_PORT=12342
#export CUDA_VISIBLE_DEVICES=0

TOKENIZER_MODEL_PATH=./tokenizer
#CHECKPOINT_PATH=/mnt/beegfs/wangshenling/train/llamacase_2B_sft_20231125/ckpt/epoch7_lf
LOG_PATH=./logs/${CASE_NAME}
OUTPUT_PATH=./output/${CASE_NAME}
#PROMPT=HumanEval-textprompts.jsonl
MAX_LENGTH=512

mkdir -p $LOG_PATH
mkdir -p $OUTPUT_PATH


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
    --use-flash-attn \
    --flash-attn-drop 0.0 \
    --attention-dropout 0 \
    --fim-rate 0.0 \
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
    --temp $TEMP \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --seed $RANDOM
"

torchrun --nproc_per_node $NUM_GPUS --master_addr $MASTER_ADDR --node_rank $NODE_RANK --nnodes $NNODES --master_port $MASTER_PORT tasks/humaneval/eval_humaneval_2B.py \
       $GPT_ARGS \
       --tokenizer-type "YuanTokenizer" \
       --tokenizer-model-path $TOKENIZER_MODEL_PATH \
       --human_eval_datapath ./datasets/HUMANEVAL/${DATASET} \
       --textprompts_datapath ./datasets/HUMANEVAL/${PROMPT} \
       --distributed-backend nccl \
       --num_samples_per_task 1 \
       --max_len $MAX_LENGTH \
       --output_path $OUTPUT_PATH \
       --load $CHECKPOINT_PATH 2>&1 | tee ${LOG_PATH}/eval_${CASE_NAME}.log
evaluate_functional_correctness -p datasets/HUMANEVAL/${DATASET}  ${OUTPUT_PATH}/samples.jsonl 2>&1 | tee ${OUTPUT_PATH}/result.txt

