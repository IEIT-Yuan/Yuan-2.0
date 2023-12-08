# checkpoint_process


## Introduction

The provided 51B ckpt was trained with 16-way pipeline parallelism and 1-way tensor parallelism. The provided 102B ckpt was trained with 32-way pipeline parallelism and 1-way tensor parallelism. 

To efficiently utilize multiple devices in the distributed training process, we provide scripts for merging/spliting tensor and merging pipeline, which can be found in the  **`examples`** directory.

**`examples/split_tp_partitions.sh`**: Split  checkpoint along the tensor.

**`examples/merge_tp_partitions.sh`**: Merge  checkpoint along the tensor.

**`examples/merge_pp_partitions.sh`**: Merge  checkpoint along the pipeline.

The variables in the code should be set as follows:

|Variable name	|Description	|
|--------------------------|----------------------------------------|
|`LOAD_CHECKPOINT_PATH`|the path that loads the checkpoint to be splited/merged|
|`SAVE_CHECKPOINT_PATH`|the storage path of the splited/merged checkpoint|
|`SAVE_SPLITED_CHECKPOINT_PATH`|the middle storage path of the converted checkpoint|
|`TOKENIZER_MODEL_PATH`|the path of tokenizer model|
|`--tensor-model-parallel-size`|the original tensor model parallel size|
|`--pipeline-model-parallel-size`|the original pipeline model parallel size|
|`--target-tensor-model-parallel-size`|the target tensor model parallel size|
|`--target-pipeline-model-parallel-size`|the target pipeline model parallel size|
|`--pipeline-model-parallel-blocks`|the number of transformer layers specified by the user for each pipeline stage|
|`--target-pipeline-model-parallel-blocks`|the number of transformer layers specified by the user for each pipeline stage in output model|
|`--process-checkpoint`|the parameter sets device=None when processing checkpoint|
|`--pipeline-generate-layer`|the parameter controls which-way pipeline only convert the parameter.|
|`--tensor-generate-layer`|the parameter controls which-way tensor only convert the parameter.|

## Usage

Run the following command to split checkpoint along tensor:
```bash
bash examples/split_tp_partitions.sh
```
Run the following command to merge checkpoint along tensor:
```bash
bash examples/merge_tp_partitions.sh
```
Run the following command to merge checkpoint along pipeline:
```bash
bash examples/merge_pp_partitions.sh
```
The scirpt for converting the 51B ckpt with 16-way pipeline and 1-way tensor to 4-way tensor and 1-way pipeline is provided: 
```
bash examples/ckpt_partitions_51B.sh
```
The scirpt for converting the 102B ckpt with 32-way pipeline and 1-way tensor to 8-way tensor and 1-way pipeline is provided:
```
bash examples/ckpt_partitions_102B.sh
```



There is no fixed order for splitting tensor and merging pipeline. It is generally suggested to split tensor first and then merge pipeline.
If you want to define the parameters for splitting and merging yourself, you can follow the steps below (take 51Bckpt as an example):
>**step1 bash examples/split\_tp\_partitions.sh**
--tensor-model-parallel-size  1
--target-model-parallel-size  8 
--pipeline-model-parallel-size 16
--target-pipeline-model-parallel-size  16
--pipeline-model-parallel-blocks 2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,2
--pipeline-generate-layer 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
**step2 bash examples/merge\_pp\_partitions.sh**
--tensor-model-parallel-size 8
--target-model-parallel-size 8
--pipeline-model-parallel-size 16
--target-model-parallel-size 2
--pipeline-model-parallel-blocks 2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,2
--target-pipeline-model-parallel-blocks 10,32
--tensor-generate-layer 0,1,2,3,4,5,6,7

When runing step2 after step1, the splited ckpt obtained by step1 need to be as LOAD_CHECKPOINT_PATH in script2. It should be noted that `tensor-model-parallel-size` and `pipeline-model-parallel-size` need to be the same as the number of tensor ways and parallel ways in the loaded checkpoint.
 

## Notice

### --pipeline-generate-layer and --tensor-generate-layer

They can used to control which layers only convert the parameter. If all layers are converted, specify all-way pipelines or all-way tensors(for example, 4-way tensor: 0,1,2,3. 1-way tensor:0. 1-way pipeline:0, 8-way pipeline:0,1,2,3,4,5,6,7). If only convert the layers in pipeline stage 0,1, specify the parameter `--pipeline-generate-layer` as 0,1. If only convert the layers in tensor 3,4,5,6, specify the parameter `--tensor-generate-layer` as 3,4,5,6.

### --pipeline-model-parallel-blocks and --target-pipeline-model-parallel-blocks

`--pipeline-model-parallel-blocks` specifys the number of transformer layers for each pipeline stage, and the length of this parameter needs to equal with pipeline-model-parallel-size. `--target-pipeline-model-parallel-blocks` specify the number of transformer layers for each pipeline stage in output model and its length needs to equal with target-pipeline-model-parallel-size.

### WORLD\_SIZE setting

When procesing 51B/102B ckpt, if *'AssertionError: num of args.pipeline\_model\_parallel\_blocks must eq args.pipeline\_model\_parallel\_size'* occurs, it may be because tensor\_model\_parallel\_size * pipeline\_model\_parallel\_size > world\_size. Modifying **os.environ["WORLD\_SIZE"] in scirpt merge\_pp\_partitions.py/split\_tp\_partitions.py** can solve this problem. It is recommended to set it to **256** as it is large enough to cover most cases.
