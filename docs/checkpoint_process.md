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