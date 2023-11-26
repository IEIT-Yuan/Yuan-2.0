# Yuan2.0 Pretraining

## Introduction

This document provides instructions for pretraining model of Yuan2.0.

Three models are provided, the main parameters are as follows:

|      | Layer number | Hidden size | Attention head |
| :--: | :----------: | :---------: | :------------: |
| 2.1B |      24      |    2048     |       32       |
| 51B  |      42      |    8192     |       64       |
| 102B |      84      |    8192     |       64       |

## Usage

The  scripts describe three models in Yuan2.0:

- 2.1B : [`pretrain_yuan2.0_2.1B.sh`](../examples/pretrain_yuan2.0_2.1B.sh)

- 51B : [`pretrain_yuan2.0_51B.sh`](../examples/pretrain_yuan2.0_51B.sh)

- 102B : [`pretrain_yuan2.0_102B.sh`](../examples/pretrain_yuan2.0_102B.sh)

### Example

An example script to run Yuan-2.1B pretraining is:

```shell
bash examples/pretrain_yuan2.0_2.1B.sh
```

### Arguments setting

Before running the script, the relevant arguments should be set correctly.

Firstly,  make any desired modifications including setting the environment variables for `CHECKPOINT_PATH`, `DATA_PATH`,  `TOKENIZER_MODEL_PATH ` and `TENSORBOARD_PATH`.

If the dataset path is:

```bash
/path/dataset.bin
```

The `DATA_PATH` can be set :

```shell
#DATA_PATH='weight dataset_path'
DATA_PATH='1 /path/dataset'
```

The dataset  preprocess can see documentation [here]().

A simple and efficient three-dimensional model-parallel approach can be controlled by `--tensor-model-parallel-size` and `--pipeline-model-parallel-size ` flag.  If the `--pipeline-model-parallel-method` flag is set to `block`, the number of transformer layers shoule be specified by the `--pipeline-model-parallel-blocks` for each pipeline stage.

The Localized Filtering-based Attention(LFA) can be activated by the '`--use-lf-gate` flag. And the `--lf-conv2d-num-pad` flag shoule be set to `1` for training and `0` for inference.

The `--use-distributed-optimizer` and `--recompute-method` can control the use of memory during Training.

Further command line arguments are described in the source file [`arguments.py`](../megatron/arguments) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/main/README.md)

