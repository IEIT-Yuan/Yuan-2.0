# Yuan2.0 of FP8

## Introduction
该文档是使用`FP8`精度进行`Yuan2.0`的预训练及指令微调的说明文档


### Pretraining

An example script to run Yuan-2.1B pretraining of FP8 is:
```shell
bash examples/pretrain_yuan2.0_2.1B_fp8.sh
```
### Pretraining Arguments setting

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

### Instruct_tuning

转换`megatron`权重为`transformer_engine`对应的权重,以`Yuan-2.1B`为例:
```shell
bash tools/model_convert/megatron2te_convertor.sh /PATH/Yuan-2.0/ /PATH/2B/iter_0000001 /PATH/2B_TE 1 1 yuan2-2b 0
```

修改`load`模型时的参数,`strict`设置为`False`,原因在于`Transformer engine`中多了`_extra_state`是用来存`fp8`训练的`scale`和`history`的，这些在加载的时候会出现冲突
```
megatron/checkpointing.py 555行
model[0].load_state_dict(state_dict['model'], strict=False)
``` 

An example script to run Yuan-2.1B SFT is:

```shell
bash examples/pretrain_yuan2.0_2.1B_sft_fp8.sh
```
### Instruct_tuning Arguments setting

Before running the script, the relevant arguments should be set correctly.

Firstly,  make any desired modifications including setting the environment variables for `CHECKPOINT_PATH`, `DATA_PATH`,  `TOKENIZER_MODEL_PATH ` and `TENSORBOARD_PATH`.

`--train-reset` allows you to begin your training iters from 0.
`--sft-stage` is highly recommended to be on since it control the calculate of loss mask during SFT.
`--override-opt-param-scheduler` allows you to set your own scheduler.
`--finetune` load model for finetuning. do not load optimizer or rng state from checkpoint and set iters to 0. Assumed when loading a release checkpoint.

If the dataset path is:

```
/path/dataset.bin
```

The `DATA_PATH` can be set :

```shell
DATA_PATH='1 /path/dataset'
``` 
