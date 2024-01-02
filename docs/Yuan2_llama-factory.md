# 基于LLaMA Factory实现源2.0微调

[llama-factory](https://github.com/hiyouga/LLaMA-Factory)是开源社区一款非常优秀的专门针对LLM做微调的框架。该项目支持业界主流的
开源模型做微调，如chatGLM、llama、faclon、mistral、Qwen、yuan2.0等；通过使用huggingface [transformers](https://github.com/huggingface/transformers)支持基于deepspeed/fsdp的多节点多卡
并行训练；支持SFT、RLHF、DPO等LLM微调过程；下述介绍使用llama-factory微调yuan2.0模型的流程。

## 准备微调环境

- docker pull nvcr.io/nvidia/pytorch:23.08-py3
- docker run -v HOST_WORK_PATH:/workspace/ --ipc=host  --gpus all  --shm-size='64g' -it  nvcr.io/nvidia/pytorch:23.08-py3 /bin/bash
- git  clone  https://github.com/hiyouga/LLaMA-Factory.git  
- cd  LLaMA-Factory
- pip  config  set  global.index-url  https://pypi.tuna.tsinghua.edu.cn/simple
- pip  install  -r  requirements.txt  
- pip  install  transformers_stream_generator  deepspeed 

## 准备模型及数据
- 获取[yuan2.0](https://github.com/IEIT-Yuan/Yuan-2.0?tab=readme-ov-file#%E6%A8%A1%E5%9E%8B)  huggingface模型文件： 
- 准备数据：<br />（1）自定义数据集时，使用[llama-factory](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README_zh.md)所要求的数据格式，如使用下述json格式的文件；将准备好的数据集信息(文件路径、文件哈希值等)
添加进入llama-factory中的dataset_info.json。<br />（2）使用llama-factory中已有的数据集做微调，直接指定数据集名称不需要做其它修改，如--dataset alpca_en代表使用alpaca-data 52K 数据集。
```
[
    {
        "instruction":  "我们如何在日常生活中减少用水？",
        "input":  "",
        "output":  "1.  使用节水装置，如节水淋浴喷头和水龙头。  \n2.  使用水箱或水桶收集家庭废水，例如洗碗和洗浴。  \n3.  在社区中提高节水意识。  \n4.  检查水管和灌溉系统的漏水情况，并及时修复它们。  \n5.  洗澡时间缩短，使用低流量淋浴头节约用水。  \n6.  收集雨水，用于园艺或其他非饮用目的。  \n7.  刷牙或擦手时关掉水龙头。  \n8.  减少浇水草坪的时间。  \n9.  尽可能多地重复使用灰水（来自洗衣机、浴室水槽和淋浴的水）。  \n10.  只购买能源效率高的洗碗机和洗衣机。\n<eod>  "
    },
    {
	...
	}
]
```
- 使用llama-factory 中定制的yuan模板

```
--template yuan
```
## 源2.0全量微调
```
deepspeed --num_gpus=8 src/train_bash.py \
        --stage sft \
        --model_name_or_path path-to-yuan-hf-model \
        --do_train \
        --dataset alpaca_en \
        --finetuning_type full  \
        --output_dir yuan2_2B_full_fintuning_checkpoint\
        --overwrite_cache \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4  \
        --gradient_accumulation_steps 4  \
        --preprocessing_num_workers 16 \
        --lr_scheduler_type cosine \
        --logging_steps 10    \
        --save_steps 10000   \
        --learning_rate 5e-5   \
        --max_grad_norm 0.5     \
        --num_train_epochs 3   \
        --evaluation_strategy no  \
        --bf16 \
        --deepspeed ./zero2_ds_woloading.json \
        --template yuan \
        --overwrite_output_dir     \
        --cutoff_len 2048\
        --sft_packing   \
        --gradient_checkpointing True 
```
 
> --stage 可以选择训练模型的方式，如sft代表有监督微调，rm代表奖励模型训练，dpo代表直接偏好优化等；<br />--finetuning_type 可以指定微调的类型如lora、full、freeze等；<br />--sft_packing  将多条样本
进行拼接到固定长度，加速模型处理速度；<br />其余参数可以参考[llama-factory源码](https://github.com/hiyouga/LLaMA-Factory/tree/main/src/llmtuner/hparams)理解
- zero2 config文件参考
```
{
  "zero_optimization": {
     "stage": 2,
     "allgather_partitions": true,
     "allgather_bucket_size": 5e8,
     "reduce_scatter": true,
     "reduce_bucket_size": 5e8,
     "overlap_comm": false,
     "contiguous_gradients": true
  },
   "bf16": {
   "enabled": "auto",
   "loss_scale": 0,
   "initial_scale_power": 16,
   "loss_scale_window": 1000,
   "hysteresis": 2,
   "min_loss_scale": 1
   },
    "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null
  },
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true
}
```

## lora及Qlora高效微调
<br />对大模型进行全量微调是一件昂贵的事情，我们可以使用高效微调的方法，通过给大模型添加额外参数，对新添加的参数进行微调进而改进大模型性能，如[lora](https://arxiv.org/abs/2106.09685)、[Qlora](https://arxiv.org/abs/2305.14314)高效微调方案。
<br />lora本质上是一种重参数化方法，通过在参数矩阵添加旁支，来微调大模型性能。lora通过只在部分权重矩阵上添加旁支，来降低计算量；通过只更新旁支矩阵的参数，降低显存占用及并行通信量。
<br />Qlora在lora的基础上将模型权重量化为4bit，并将scale参数再进行一次量化（double quant），以达到显存进一步节省的目的。需要注意的是Qlora相比于lora一般会添加更多的旁支矩阵，其并不能加速计算，反而会有效率上的损失。
<br />使用llama-factory可以通过添加如下参数非常方便对yuan2.0模型进行基于lora和Qlora的高效微调。
- finetuning_type  选择full做全量微调，选择lora做高效微调
- 当选择lora时,--lora_target可指定"q_proj",  "k_proj",  "v_proj",  "o_proj",  "gate_proj", "up_proj",  "down_proj"
- 当指定--quantization_bit  4或8 可以启动Qlora微调


## 实测数据参考
| 微调方案     |     序列长度      |    Model       |  精度：加载/计算    |   GPU | bs:micro/global |显存占用(1*GPU)|微调耗时                 |
| ------------| ----------------- | -------------  | ------------------ | ------|---------------- | ------ | ----------------------------- |
|  ds_zero2_full| 2048            |  Yuan-2 2B     | bf16/bf16          | 8*L40s| 4/128           |19G     |0.15h                          |
|  ds_zero3_lora| 2048            |  Yuan-2 51B    | bf16/bf16          | 8*L40s| 1/128           |34G     |3.54h                          |
|  ds_zero3_lora| 2048            |  Yuan-2 102B   | bf16/bf16          | 8*L40s| 1/128           |47G     |7h                             |
|  Qlora        | 1024            |  Yuan-2 2B     | int4/bf16          | 1*L40s| 1/16            |4.5G    |1h                             |
|  Qlora        | 1024            |  Yuan-2 51B    | int4/bf16          | 1*L40s| 1/16            |40G     |22h                            |

>以上测试使用20K条samples微调3个epoch，原始的单条sample转换为tokens平均长度为167，packing到1024或2048

## 硬件资源评估
|Method |精度   |yuan2-2B |yuan2-51B  |yuan2-102B|
| ------| ----- |-------- | --------- | -------- |
|Full  |16 bit|40GB|1000GB|2000GB|
|lora  |16 bit|7GB|120GB |230GB  |
|Qlora |4 bit|5GB|40GB  |80GB   |
