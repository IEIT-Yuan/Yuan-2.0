#!/bin/bash

# Runs the "Yuan-51B" parameter model inference


GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6042
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MAX_LENGTH=512

if [ "$TEMP" == "" ]; then
    TEMP=1
fi
if [ "$TOP_P" == "" ]; then
    TOP_P=0.0
fi
if [ "$TOP_K" == "" ]; then
    TOP_K=1
fi
if [ "$CASE_NAME" == "" ]; then
    CASE_NAME=test-51B
fi

TOKENIZER_MODEL_PATH=./tokenizer
CHECKPOINT_PATH=<Specify CHECKPOINT_PATH>
DATASET=HumanEval.jsonl.gz
PROMPT=HumanEval-textprompts.jsonl
LOG_PATH=./logs/${CASE_NAME}
OUTPUT_PATH=./output/${CASE_NAME}

mkdir -p $LOG_PATH
mkdir -p $OUTPUT_PATH

GPT_ARGS="
    --micro-batch-size 1 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --num-layers 42 \
    --hidden-size 8192 \
    --use-lf-gate \
    --use-flash-attn \
    --lf-conv2d-group 1 \
    --lf-conv2d-num-pad 0 \
    --position-embedding-type rope \
    --no-embedding-dropout \
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


torchrun $DISTRIBUTED_ARGS tasks/humaneval/eval_humaneval.py \
       $GPT_ARGS \
       --tokenizer-type "YuanTokenizer" \
       --tokenizer-model-path $TOKENIZER_MODEL_PATH \
       --human_eval_datapath ./datasets/HUMANEVAL/${DATASET} \
       --textprompts_datapath ./datasets/HUMANEVAL/${PROMPT} \
       --distributed-backend nccl \
       --num_samples_per_task 1 \
       --max_len $MAX_LENGTH \
       --output_path $OUTPUT_PATH \
       --load $CHECKPOINT_PATH 2>&1 | tee ${LOG_PATH}/eval_${NODE_RANK}_${CASE_NAME}.log
evaluate_functional_correctness -p datasets/HUMANEVAL/${DATASET}  ${OUTPUT_PATH}/samples.jsonl 2>&1 | tee ${OUTPUT_PATH}/result.txt

