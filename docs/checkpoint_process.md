# checkpoint_process


## Introduction

To efficiently utilize multiple devices in the distributed training process, we provide scripts for merging/spliting tensor and merging pipeline, which can be found in the  **`examples`** directory.

**`examples/split_tp_partitions.sh`**: Split  checkpoint along the tensor.

**`examples/merge_tp_partitions.sh`**: Merge  checkpoint along the tensor.

**`examples/merge_pp_partitions.sh`**: Merge  checkpoint along the pipeline.

The variables in the code should be set as follows:

|Variable name	|Description	|
|--------------------------|----------------------------------------|
|`LOAD_CHECKPOINT_PATH`|the path that loads the checkpoint to be splited/merged|
|`SAVE_CHECKPOINT_PATH`|the storage path of the splited/merged checkpoint|
|`TOKENIZER_MODEL`|the tokenizer model or path|
|`--tensor-model-parallel-size`|the original tensor model parallel size|
|`--pipeline-model-parallel-size`|the original pipeline model parallel size|
|`--target-tensor-model-parallel-size`|the target tensor model parallel size|
|`--target-pipeline-model-parallel-size`|the target pipeline model parallel size|
|`--pipeline-model-parallel-blocks`|The number of transformer layers specified by the user for each pipeline stage|
|`--target-pipeline-model-parallel-blocks`|The number of transformer layers specified by the user for each pipeline stage in output model|
|`--process-checkpoint`|the parameter sets device=None when processing checkpoint|

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
**--pipeline-generate-layer--** and **--tensor-generate-layer--**  controls which layers only convert the parameter. For example, only the layers in the pipeline 0,1,2,3 are convert:
```
--pipeline-generate-layer-- 0,1,2,3
```

The provided 51B ckpt was trained with 16 pipeline parallelism and 1 tensor parallelism. The provided 102B ckpt was trained with 32 pipeline parallelism and 1 tensor parallelism. The parameters need to be modified when using the script. 


## Notice

When procesing 51B/102B ckpt, if 'AssertionError: num of args.pipeline\_model\_parallel\_blocks must eq args.pipeline\_model\_parallel\_size' occurs, it may be because tensor\_model\_parallel\_size * pipeline\_model\_parallel\_size > world\_size. Modifying os.environ["WORLD\_SIZE"] in scirpt merge\_pp\_partitions.py/split\_tp\_partitions.py can solve this problem. 
