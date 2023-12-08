#!/bin/bash


#Convert 51B ckpt with 16-way pipeline and 1-way tensor to 1-way pipeline and 4-way tensor.

LOAD_CHECKPOINT_PATH=<Specify the loaded ckpt path>
SAVE_SPLITED_CHECKPOINT_PATH=<Specify the stored splited ckpt path>
SAVE_CHECKPOINT_PATH=<Specify the final stored ckpt path>
TOKENIZER_MODEL_PATH=./tokenizer

bash ./examples/split_tp_partitions_51B.sh $LOAD_CHECKPOINT_PATH $SAVE_SPLITED_CHECKPOINT_PATH $TOKENIZER_MODEL_PATH
bash ./examples/merge_pp_partitions_51B.sh $SAVE_SPLITED_CHECKPOINT_PATH $SAVE_CHECKPOINT_PATH $TOKENIZER_MODEL_PATH
