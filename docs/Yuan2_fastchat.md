# 基于Fastchat实现源2.0的微调和推理部署

[FastChat](https://github.com/lm-sys/FastChat)是一个用于训练、部署和评估基于LLM(大型语言模型)的聊天机器人的开放平台。通过使用huggingface [transformers](https://github.com/huggingface/transformers)支持LLM基于deepspeed/fsdp的多节点多卡微调；下述介绍使用FastChat微调yuan2.0模型的流程。

## 准备微调环境

- docker pull nvcr.io/nvidia/pytorch:23.08-py3
- docker run -v HOST_WORK_PATH:/workspace/ --ipc=host  --gpus all -p host-port:container-port --shm-size='64g' -it  nvcr.io/nvidia/pytorch:23.08-py3 /bin/bash
- git  clone  https://github.com/lm-sys/FastChat.git
- cd  FastChat
- pip  config  set  global.index-url  https://pypi.tuna.tsinghua.edu.cn/simple
- pip install -e  ".[model_worker,webui,train]"
- pip install deepspeed “bitsandbytes>=0.39.0” “transformers==4.31.0” plotly openai

## 准备模型及数据
- 获取[yuan2.0](https://github.com/IEIT-Yuan/Yuan-2.0?tab=readme-ov-file#hugging-face%E7%89%88%E6%9C%AC-)  huggingface模型文件： 
- 准备数据：FastChat为聊天机器人训练及服务做支持，因此其所需要的标准数据集为多轮及单轮对话数据集。<br />(1)自定义数据集时，使用[fastchat](https://github.com/lm-sys/FastChat/blob/main/data/dummy_conversation.json)所要求的数据格式，如使用下述json格式的文件定义单论或多轮对话数据集。<br />(2)使用已有的指令数据集改造为单轮对话，可以使用alpaca-data[英文](https://github.com/tatsu-lab/stanford_alpaca?tab=readme-ov-file#data-release)或[中文](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/alpaca_data_zh_51k.json)数据集进行对应格式的改造。
<br />(3)使用开源的多轮对话数据集，如BELLE项目开源的用户与助手的多轮对话数据集[bella-0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)。

```
#multi turns example
[
   [
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "human",
        "value": "Who are you?"
      },
      {
        "from": "gpt",
        "value": "I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS)."
      },
      {
        "from": "human",
        "value": "Have a nice day!"
      },
      {
        "from": "gpt",
        "value": "You too!"
      }
    ]
  },
  {
    "id": "identity_1",
    "conversations": [
      {
        "from": "human",
        "value": "Who are you?"
      },
      {
        "from": "gpt",
        "value": "My name is Vicuna, and I'm a language model developed by Large Model Systems Organization (LMSYS)."
      }
    ]
  },
]
# single turn example
[
  {
    "id": "1",
    "conversations": [
      {
        "from": "human",
        "value": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGive three tips for staying healthy.\n\n### Response:"
      },
      {
        "from": "gpt",
        "value": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
      }
    ]
  },
  {
    "id": "2",
    "conversations": [
      {
        "from": "human",
        "value": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat are the three primary colors?\n\n### Response:"
      },
      {
        "from": "gpt",
        "value": "The three primary colors are red, blue, and yellow."
      }
    ]
  },
```
## 使用fastchat 中定制的yuan2.0训练脚本文件

```
在fastchat/train/train_mem.py脚本中
- from fastchat.train.train import train
+ from fastchat.train.train_yuan2 import train

在fastchat/train/train_lora.py脚本中
-from fastchat.train.train import (
-    DataArguments,
-    ModelArguments,
-    make_supervised_data_module,
-)

+from fastchat.train.train_yuan2 import (
+    DataArguments,
+    ModelArguments,
+    make_supervised_data_module,
+)

将fastchat/train/train_yuan2.py脚本中的special tokenizer复制到train_lora.py
+tokenizer.add_tokens(
+        [
+            "<eod>",
+            "<sep>",
+            "<pad>",
+           "<mask>",
+           "<predict>",
+           "<FIM_SUFFIX>",
+          "<FIM_PREFIX>",
+          "<FIM_MIDDLE>",
+          "<commit_before>",
+          "<commit_msg>",
+          "<commit_after>",
+          "<jupyter_start>",
+          "<jupyter_text>",
+          "<jupyter_code>",
+          "<jupyter_output>",
+          "<empty_output>",
+      ],
+      special_tokens=True,
+  )

```
> fastchat中添加的yuan2_template相关信息，以下内容无需修改，开发者如有特殊需求可调整或改变如下相关模板信息
```
#yuan template infomation

fastchat/conversation.py脚本，包含yuan2.0 chat定制的模板信息

# Yuan2.0 chat template
# source: https://huggingface.co/IEITYuan/Yuan2-2B-Janus-hf/blob/main/tokenizer_config.json#L6
register_conv_template(
    Conversation(
        name="yuan2",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.YUAN2,
        sep="<sep>",
        sep2="\n",
        stop_token_ids=[
            77185,
        ],  # "<eod>"
        stop_str="<eod>",
    )
)
fastchat/model/model_adapter.py脚本， 包含yuan2.0 chat模型及tokenizer加载时的函数

class Yuan2Adapter(BaseModelAdapter):
    """The model adapter for Yuan2.0"""

    def match(self, model_path: str):
        return "yuan2" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        # from_pretrained_kwargs["torch_dtype"] = torch.bfloat16
        tokenizer = LlamaTokenizer.from_pretrained(
            model_path,
            add_eos_token=False,
            add_bos_token=False,
            eos_token='<eod>',
            eod_token='<eod>',
            sep_token='<sep>',
            revision = revision,
        )
        tokenizer.add_tokens(
            ['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>', '<commit_before>',
             '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>',
             '<jupyter_output>', '<empty_output>'], special_tokens=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # device_map='auto',
            trust_remote_code=True,
            **from_pretrained_kwargs
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("yuan2")

fastchat/model/model_yuan2.py脚本，包含yuan2.0 chat模型生成内容时的默认设置

```
## 源2.0全量微调
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=20001 fastchat/train/train_mem.py \
        --model_name_or_path  path-to-huggingface-models \
        --trust_remote_code True\
        --data_path ./data/alpaca_data_zh_conversion.json \
        --bf16 True \
        --output_dir ./test_yuan2b_full \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 1200 \
        --save_total_limit 10 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 1024 \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --deepspeed playground/zero2_ds_woloading.json \
        --efficient_loss False \
        --split_example_loss True \
        --last_response_loss False \
```
 
> <br />--model_max_length 可以指定微调时单个样本最大长度；<br /><br />--efficient_loss,--split_example_loss,--last_response_loss,代表了三种不同的面对多轮对话的loss计算方式。(1) efficient_loss代表计算聊天助手回答部分的loss；(2) last_response_loss代表只计算最后一轮聊天助手回答部分的loss；(3) split_example_loss代表将多轮对话拆分成多组样本，计算每组样本中最后一轮聊天助手内容部分的loss。选择时有且仅有一个为True，其余为False。 <br /><br />其余参数可以参考[fastchat源码](https://github.com/lm-sys/FastChat)及[transformers](https://github.com/huggingface/transformers)理解
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
<br />使用fastchat可以通过如下方式非常方便对yuan2.0模型进行基于lora和Qlora的高效微调。
- 使用train_lora.py脚本，torchrun --nproc_per_node=8 --master_port=XXXX fastchat/train/train_lora.py .....
- 使用--lora_target_modules指定模型添加的lora模块，可以指定"q_proj",  "k_proj",  "v_proj",  "o_proj",  "gate_proj", "up_proj",  "down_proj"中的一个或多个，默认使用"q_proj", "v_proj"
- 使用--lora_r指定lora矩阵的秩
- 当指定--q_lora (True or False)指定是否使用Qlora进行高效微调<br />
- 高效微调在进行多轮对话微调时loss计算方式与全量微调一致，可以使用yuan2.0定义的三种不同方式中的一种   
高效微调参考脚本如下：
```
CUDA_VISIBLE_DEVICES=0 python  fastchat/train/train_lora.py \
        --model_name_or_path  hf-to-yuan-path \
        --trust_remote_code True\
        --data_path ./data/alpaca-data-conversation.json \
        --bf16 True \
        --output_dir ./checkpoints_yuan2_2b_lora \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 1200 \
        --save_total_limit 10 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --model_max_length 512 \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --q_lora True \
        --efficient_loss False \
        --split_example_loss True \
        --last_response_loss False \


```



## 微调实测数据参考
| 微调方案     |     序列长度      |    Model       |  精度：加载/计算    |   GPU | bs:micro/global |显存占用(1*GPU)|epoch耗时|
| ------------| ----------------- | -------------  | ------------------ | ------|---------------- | ------ | -----|
|  ds_zero2_full| 2048            |  Yuan-2 2B     | bf16/bf16          | 8*L20| 1/128           |16G     |1.68h|
|  ds_zero3_lora| 2048            |  Yuan-2 51B    | bf16/bf16          | 8*L20| 1/128           |43G     |23h|
|  ds_zero3_lora| 2048            |  Yuan-2 102B   | bf16/bf16          | 8*L20| 1/128           |45G     |47h|
|  ds_zero2_full| 1024            |  Yuan-2 2B     | bf16/bf16          | 8*L20| 1/128           |15G     |1.3h|
|  ds_zero3_lora| 1024            |  Yuan-2 51B    | bf16/bf16          | 8*L20| 1/128           |43G     |18h|
|  ds_zero3_lora| 1024            |  Yuan-2 102B   | bf16/bf16          | 8*L20| 1/128           |42G     |40h|
|  Qlora        | 1024            |  Yuan-2 2B     | int4/bf16          | 1*L20| 1/16            |4.5G    |3.4h|

>以上测试使用52K条alpaca-samples，改造为单轮对话数据；epoch耗时为微调单个epoch的时间

## 微调部署及使用

基于yuan2.0微调完成的chat模型，使用fastchat可以非常方便的进行服务部署及使用。

-  命令行方式
```angular2html
使用N个GPU部署chat模型
python3 -m fastchat.serve.cli --model PATH-TO_CHATMODELS --num-gpus N
```
- WebGUI方式
```angular2html
python3 -m fastchat.serve.controller  --host 0.0.0.0 &
python3 -m fastchat.serve.model_worker --model-path PATH-TO_CHATMODELS --host 0.0.0.0 &
#--gpus 0,1,2,3 --num-gpus 4 指定使用4个GPU加载模型进行推理
python3 -m fastchat.serve.gradio_web_server --host 0.0.0.0 --port 映射的IP端口号
```
- OpenAI-Compatible RESTful APIs

关于安装`fastchat`及相关依赖，可以执行:
```shell
pip3 install "fschat[model_worker,webui]"
pip3 install transformers==4.36.2 einops==0.7.0 gradio==3.50.2 gradio_client==0.6.1 pydantic==1.10.13
```

在正确安装完`fastchat`之后，可以参考[fastchat openai api启动脚本](../examples/fastchat_openai_server_engine.sh), 修改脚本里HOST、PORT、MODEL_PATH等内容：
```shell
CONTROLLER_HOST="0.0.0.0"
CONTROLLER_PORT=8503

MODEL_WORKER_HOST="0.0.0.0"
MODEL_WORKER_PORT=8504

API_SERVER_HOST="0.0.0.0"
API_SERVER_PORT=8505

MODEL_PATH="/mnt/models/Yuan2-2B-Mars-hf/"
```
启动完毕后，验证：
```shell
# cURL或者浏览器访问 http://<api_server_host>:<api_server_port>/v1/models 确保结果中有一个类似的模型：
{
    "object": "list",
    "data": [
        {
            "id": "yuan2",
            "object": "model",
            "created": 1713955516,
            "owned_by": "fastchat",
            "root": "yuan2",
            "parent": null,
            "permission": [
                {
                    "id": "modelperm-KT7CstuH8yLHFWWiFzVpkd",
                    "object": "model_permission",
                    "created": 1713955516,
                    "allow_create_engine": false,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": true,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ]
        }
    ]
}
```
使用`openai`客户端调用:
```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://<api_server_host>:<api_server_port>/v1",
)

completion = client.chat.completions.create(
    model="yuan2",
    messages=[
        {"role": "system", "content": "你是一个私人助手，能帮我解决很多问题。"},
        {"role": "user", "content": "你好!"}
    ]
)

print(completion.choices[0].message)

# output
# ChatCompletionMessage(content='你好！很高兴为你提供帮助。请问有什么我可以为你做的吗？', role='assistant', function_call=None, tool_calls=None)
```
>我们可以在[langchain](https://github.com/langchain-ai/langchain)中[使用OpenAI-Compatible RESTful APIs完成基于LLM的应用构建。](https://github.com/lm-sys/FastChat/blob/main/docs/langchain_integration.md)
