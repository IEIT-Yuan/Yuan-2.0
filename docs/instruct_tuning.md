# Yuan2.0 Supervised Finetuning

## Introduction

This document provides instructions for supervised finetuning (SFT) of Yuan2.0.


## Usage

An example script to run Yuan-102B SFT is:

```shell
bash examples/pretrain_yuan2.0_102B_sft.sh
```

### Arguments setting

Before running the script, the relevant arguments should be set correctly.

Firstly,  make any desired modifications including setting the environment variables for `CHECKPOINT_PATH`, `DATA_PATH`,  `TOKENIZER_MODEL_PATH ` and `TENSORBOARD_PATH`.

`--train-reset` allows you to begin your training iters from 0.
`--sft-stage` is highly recommended to be on since it control the calculate of loss mask during SFT.
`--override-opt-param-scheduler` allows you to set your own scheduler.

If the dataset path is:

```
/path/dataset.bin
```

The `DATA_PATH` can be set :

```shell
DATA_PATH='1 /path/dataset'
```

For dataset preprocesss please refer to [documentation]().

Further command line arguments are described in the source file [`arguments.py`](./megatron/arguments) and [REAMME.md](https://github.com/NVIDIA/Megatron-LM/blob/main/README.md)


