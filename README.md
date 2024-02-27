
# 源2.0

[Read this in English.](./README-EN.md)

📔 更为详细的使用信息，可以参考：[源2.0 论文](https://arxiv.org/ftp/arxiv/papers/2311/2311.15786.pdf)



## 目录

- [源2.0](#源20)
  - [目录](#目录)
  - [持续更新🔥🔥🔥](#持续更新)
  - [介绍](#介绍)
  - [源大模型共训计划](#源大模型共训计划)
  - [快速启动](#快速启动)
    - [环境配置](#环境配置)
    - [数据预处理](#数据预处理)
    - [预训练](#预训练)
    - [模型微调](#模型微调)
    - [模型](#模型)
    - [Hugging Face版本 ](#hugging-face版本-)
    - [原始版本 ](#原始版本-)
  - [评测结果](#评测结果)
  - [代码调用](#代码调用)
  - [源2.0 + 源Chat部署](#源20--源chat部署)
    - [linux部署](#linux部署)
    - [Windows部署](#windows部署)
      - [🔘 GPU部署](#-gpu部署)
      - [🔘 CPU部署](#-cpu部署)
  - [联系我们](#联系我们)
  - [招聘公告](#招聘公告)


<!-- markdown-toc end -->




## 持续更新🔥🔥🔥
* [2024-02-27] [增加用FP8精度训练和微调源2.0 2B模型](./docs/FP8.md),详请参见本页中的章节
* [2024-02-04] [增加用 TensorRT-LLM & Triton Server 部署2B模型](https://github.com/inspurMJX/Yuan-2.0/blob/main/3rdparty/TensorRT-LLM/README_Yuan.md),详请参见本页中的章节
* [2024-01-24] [源2.0适配FastChat框架](https://github.com/lm-sys/FastChat/blob/main/docs/model_support.md)，支持最新[对话模板](https://huggingface.co/IEITYuan/Yuan2-2B-Janus-hf/blob/main/tokenizer_config.json#L6)：FastChat是一个用于训练、部署和评估基于大型语言模型的开放平台。用户可以基于FastChat框架更快、更灵活地使用源2.0大模型。
* [2024-01-13] [新版 2B 模型发布：Yuan2-2B-Janus-hf](https://huggingface.co/IEITYuan/Yuan2-2B-Janus-hf)：**从评测结果上来看，新版本的Yuan2-2B模型在推理、代码、对话等领域，均取得了显著的成果。准确率方面，HumanEval的评测结果从54.9%提升至61.5%，GSM8K的评测结果从66.6%提升至70.2% 。**
* [2024-01-04] [使用 源Chat(YuanChat) 搭建对话应用](https://github.com/IEIT-Yuan/YuanChat/tree/main):源Chat 是Yuan-2.0 项目的一部分, 作为Yuan-2.0的一个客户端应用. 源Chat 提供了一种简单的交互方式，可以让用户很轻松的使用 Yuan-2.0, 用户可以很方便的进行测试以及使用。
* [2024-01-02] [增加 Hugging Face 版本模型下载链接](https://github.com/IEIT-Yuan/Yuan-2.0?tab=readme-ov-file#hugging-face%E7%89%88%E6%9C%AC-),详情参见本页中的章节。





## 介绍

源2.0 是浪潮信息发布的新一代基础语言大模型。我们开源了全部的3个模型源2.0-102B，源2.0-51B和源2.0-2B。并且我们提供了预训练，微调，推理服务的相关脚本，以供研发人员做进一步的开发。源2.0是在源1.0的基础上，利用更多样的高质量预训练数据和指令微调数据集，令模型在语义、数学、推理、代码、知识等不同方面具备更强的理解能力。

-----

对本仓库源码的使用遵循开源许可协议 **Apache 2.0**。

源2.0模型支持商用，不需要申请授权，请您了解并遵循[《源2.0模型许可协议》](./LICENSE-Yuan)，勿将开源模型和代码及基于开源项目产生的衍生物用于任何可能给国家和社会带来危害的用途以及用于任何未经过安全评估和备案的服务。

尽管模型在训练时我们已采取措施尽力确保数据的合规性和准确性，但模型参数量巨大且受概率随机性因素影响，我们无法保证输出内容的准确性，且模型易被输入指令所误导，本项目不承担开源模型和代码导致的数据安全、舆情风险或发生任何模型被误导、滥用、传播、不当利用而产生的风险和责任。**您将对通过使用、复制、分发和修改模型等方式利用该开源项目所产生的风险与后果，独自承担全部责任。**

## 源大模型共训计划

我们希望开源的模型更符合开发者应用需求，为此我们推出源大模型共训计划，开发者提出自己的应用或场景的需求，由我们来准备训练数据并对源大模型进行增强训练，训练后的模型依然在社区开源。

每月六日我们会收集前一月开发者提出的具体需求，经过评审后列入当月模型训练计划，训练完成后的模型在当月月末就会更新到开源社区。开发者只需要提出需求，由我们来进行数据准备、模型训练并开源。请开发者在issue的“源大模型共训计划”问题下提出具体需求，提出需求的具体格式无要求，只需要说清楚具体的应用场景、对大模型的能力需求以及给出输入输出的说明。

以下是提出需求的一些示例（几条示例，能够反应场景的典型特性即可）：

1. 场景需求：能够基于业务场景生成相关内容，对场景的描述。
 输入：用户问题，输出：正确的答案。

2. 场景需求：我想让大模型能够阅读一个领域下的多篇论文，给出这些论文的综述，当前领域研究的热点以及未解决的问题，从而辅助学术研究。
输入为：一个领域下的多篇论文，输出为：综述研究报告，研究热点总结，未解决问题总结。

## 快速启动 
详细启动文档可参考[快速启动](Quickstart.md).

### 环境配置

我们建议使用有我们提供的最新的docker[镜像文件](https://hub.docker.com/r/yuanmodel/yuan2.0).

我们可以通过下面命令启动容器：

```bash
docker pull yuanmodel/yuan2.0:V1-base
docker run --gpus all --privileged --ulimit stack=68719476736 --shm-size=1000G -itd -v /path/to/yuan_2.0:/workspace/yuan_2.0 -v /path/to/dataset:/workspace/dataset -v /path/to/checkpoints:/workspace/checkpoints --name your_name yuanmodel/yuan2.0:V1-base
docker exec -it your_name bash
```




### 数据预处理

我们提供了数据预处理的脚本，参考[数据预处理说明文档](./docs/data_process.md).

### 预训练

我们提供了用于预训练的文档和 [`example`](./examples)的脚本，具体使用方法可以参考[预训练说明文档](./docs/pretrain.md).



### 模型微调

请参考指令微调 [源2.0 指令微调示例](./docs/instruct_tuning.md)。

请注意，不同的微调脚本对应的模型并不相同，请根据需要选择对应的模型。

支持使用[llama-factory进行指令微调](./docs/Yuan2_llama-factory.md)。

支持使用[fastchat进行多轮对话的微调](./docs/Yuan2_fastchat.md)。

### 模型

源2.0 是浪潮信息发布的新一代基础语言大模型。我们开源了全部的3个模型：源2.0-102B、源2.0-51B、源2.0-2B。提供预训练、微调、推理服务的相关脚本，以供研发人员做进一步开发。源2.0是在源1.0的基础上，利用更多样的高质量预训练数据和指令微调数据集，令模型在语义、数学、推理、代码、知识等不同方面具备更强的理解能力。


🥇🥇🥇 **<font color=#FFC125 >我们提供了源2.0的模型文件，可以通过以下链接进行下载：</font>**


### <font color=#FFC125 >Hugging Face版本 </font> 



|    模型     | 序列长度  |                                                                                                                                                                                       下载链接                                                                                                                                                                                        |
| :----------: | :------: |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| 源2.0-102B-hf |    4K    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2.0-102B-hf/summary) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-102B-hf) \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-102B-hf)  \|  [百度网盘](https://pan.baidu.com/s/1O4GkPSTPu5nwHk4v9byt7A?pwd=pq74#list/path=%2F) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-102B-hf) |
| 源2.0-51B-hf  |    4K    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2.0-51B-hf/summary)  \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2.0-51B-hf)  \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-51B-hf)   \| [百度网盘](https://pan.baidu.com/s/1-qw30ZuyrMfraFtkLgDg0A?pwd=v2nd#list/path=%2F) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-51B-hf) |
|  源2.0-2B-hf  |    8K    |  [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2.0-2B-hf/summary)   \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-2B-hf)   \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-2B-hf)    \| [百度网盘](https://pan.baidu.com/s/1nt-03OAnjtZwhiVywj3xGw?pwd=nqef#list/path=%2F) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-2B-hf)   |
|  源2.0-2B-Janux-hf <sup><font color="#FFFF00">*New*</font><br /></sup> |    8K    |  [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-2B-Janus-hf/files)   \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-2B-Janus-hf) \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-2B-Janus-hf)  \| [百度网盘](https://pan.baidu.com/s/1f7l-rSVlYAij33htR51TEg?pwd=hkep ) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-2B-Janus-hf)                                 |





### <font color=#FFC125 >原始版本 </font> 


|    模型     | 序列长度  |                                                                                                                                                                           下载链接                                                                                                                                                                           |
| :----------: | :------: |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| 源2.0-102B |    4K    |                             [ModelScope](https://www.modelscope.cn/models/YuanLLM/Yuan2.0-102B/summary)  \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2.0-102B)  \|  [百度网盘](https://pan.baidu.com/s/1Tb9W6hEWS4bMkaE3p5s1fw?pwd=xrfo) \| [WiseModel](https://wisemodel.cn/models/IEIT-Yuan/Yuan2.0-102B)                              |
| 源2.0-51B  |    4K    |                               [ModelScope](https://www.modelscope.cn/models/YuanLLM/Yuan2.0-51B/summary)  \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2.0-51B)   \| [百度网盘](https://pan.baidu.com/s/1bOypWMepdh9GFK_hHXVQbQ?pwd=1uw3) \| [WiseModel](https://wisemodel.cn/models/IEIT-Yuan/Yuan2.0-51B)                               |
|  源2.0-2B  |    8K    |                               [ModelScope](https://www.modelscope.cn/models/YuanLLM/Yuan2.0-2B/summary)   \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2.0-2B)    \| [百度网盘](https://pan.baidu.com/s/1Xj8Mi2tPwuuVu7Cb0tCbtw?pwd=qxpa) \| [WiseModel](https://wisemodel.cn/models/IEIT-Yuan/Yuan2.0-2B)                                |
|  源2.0-2B-Janux <sup><font color="#FFFF00">*New*</font><br /></sup> |    8K    |                             [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-2B-Janus/files)   \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-2B-hf)  \| [百度网盘](https://pan.baidu.com/s/1hCHI9LwxborXWABaShwl4w?pwd=sdyq) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-2B-Janus)                             |



源2.0-2B模型支持的序列长度为8192个tokens，源2.0-51B和源2.0-102B模型支持的序列长度为4096个tokens，可以根据用户设备的内存大小设置 `--max-position-embeddings` 和 `--seq-length` 的值。



## 评测结果

我们提供了[HumanEval](./docs/eval_humaneval.md)，[AGIEval-GK-Math](./docs/eval_agieval_math_cn.md)，[GSM8K](./docs/eval_gsm8k_cn.md)和[TruthfulQA](./docs/eval_TruthfulQA.md)的评估脚本，以方便大家复现我们的评测结果。在4个典型任务上，我们在论文中给出了源2.0不同尺寸模型的精度。

| Model             | GSM8K   | AGIEval-GK-Math-QA     | AGIEval-GK-Math-Cloze     | HumanEval | TurthfulQA |
| ----------------- | :----:  | :------------: | :---------------: | :-------: | ---------- |
|  GPT-4            |  92%    |     47.0%      |       16.1%       |   86.6%   |     59%    |
|  ChatGPT         | 68.6%\* |     36.5%      |        7.3%       |  66.5%\*  |     34%\*  |
|  Llama2           | 56.8%   |       -        |         -         |   29.9%   |       -    |
| 源2.0-102B      | 76.6%   |     38.7%      |       13.5%       |   67.1%   |     58%    |
| 源2.0-102B-SC   | 86.2%   |     45.5%      |       15.2%       |   77.4%   |       -    |

\* 使用与源2.0完全相同的输入数据对ChatGPT进行测试，时间2023年11月

## 代码调用 

考虑到推理服务的效率，源2.0-51B和源2.0-102B模型在启动推理服务之前，需要将模型转换成只有张量并行的模型文件。可以参考[文档](./docs/checkpoint_process.md)

可以通过调用推理服务，向推理服务发送请求实现模型的调用，[源2.0 推理服务](./docs/inference_server.md)

详细启动推理服务的流程可以参考 [Yuan2_inference_guide文档](./docs/Yuan2_inference_guide_cn.md)

可以使用[replicate.com/ieit-yuan](https://replicate.com/ieit-yuan)进行yuan2.0的线上api调用 ，具体操作方式参考replicate的官方文档。在LangChain和llamaIndex中使用replicate的教程可参考：https://python.langchain.com/docs/integrations/providers/replicate 和 https://docs.llamaindex.ai/en/stable/api_reference/llms/replicate.html。


## 源2.0 + 源Chat部署

使用 [源Chat（YuanChat）](https://github.com/IEIT-Yuan/YuanChat) 可以快速构建基于源2.0大模型的对话应用，源Chat 提供了一种简单的交互方式，支持在linux部署和Windows 操作系统上的便捷部署。


### linux部署


**Step 1:** 根据 [源2.0 推理服务](./docs/inference_server_cn.md)，获取推理服务的 request url：`http://127.0.0.1:8000` ，支持ckpt和HuggingFace两种模型方式部署

**Step 2:** 根据 [源Chat部署文档](https://github.com/IEIT-Yuan/YuanChat/blob/main/README.md) 完成源Chat的部署

**Step 3:** 在浏览器中访问链接：http://localhost:5050，验证是否部署正确


### Windows部署
#### 🔘 GPU部署
**Step 1:** 根据 [源2.0 推理服务](./docs/inference_server_cn.md)，获取推理服务的 request url：`http://127.0.0.1:8000` ，支持ckpt和HuggingFace两种模型方式部署

**Step 2:** 根据 [源Chat部署文档](https://github.com/IEIT-Yuan/YuanChat/blob/main/README.md) 完成源Chat的部署

**Step 3:** 在浏览器中访问链接：http://localhost:5050，验证是否部署正确

#### 🔘 CPU部署
仅支持HuggingFace模型方式部署

**Step 1:** 通过修改HuggingFace模型配置文件手动关闭flash_atten，具体如下：将[config_cpu.json](https://huggingface.co/IEITYuan/Yuan2-2B-hf/blob/main/config_cpu.json) 内容替代[config.json](https://huggingface.co/IEITYuan/Yuan2-2B-hf/blob/main/config.json), 将[yuan_hf_model_cpu.py](https://huggingface.co/IEITYuan/Yuan2-2B-hf/blob/main/yuan_hf_model_cpu.py) 内容替代[yuan_hf_model.py](https://huggingface.co/IEITYuan/Yuan2-2B-hf/blob/main/yuan_hf_model.py)

**Step 2:** 根据 [Hugging Face 模型推理api部署](https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/docs/inference_server_cn.md#huggingface%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86api%E9%83%A8%E7%BD%B2) ，获取推理服务的 request url：`http://127.0.0.1:8000` 

**Step 3:** 根据 [源Chat部署文档](https://github.com/IEIT-Yuan/YuanChat/blob/main/README.md) 完成源Chat的部署

**Step 4:** 在浏览器中访问链接：`http://localhost:5050`，验证是否部署正确

详细部署方案可以参考 [源2.0](https://github.com/IEIT-Yuan/Yuan-2.0/tree/main) 与 [源Chat](https://github.com/IEIT-Yuan/YuanChat/) 

## TensorRT-LLM推理服务部署
性能测试

我们比较了Yuan2.0-2B的trt_llm模型和原始的megatron模型进行的推理速度

max_output_len=300, prompt="写一篇春游作文<sep>"

| Batch_size |  Megatron(推理速度:token/s)   |   trt-llm-engine_2B(推理速度:token/s)  |   性能提升(倍)          
| :---------: |:----------------------------:|:--------------------------------------:|:------------:|
| 1 | 29 | 124 | 4.35 |
| 4 | 114| 477 | 4.17 |
| 8 | 229 | 880 | 3.85 |
| 16 | 432| 1888 | 4.37 |
| 32 | 842 | 3326 | 3.95 |
| 64 | 1684| 6724 | 3.99 |

详细部署方案可以参考[TensorRT-LLM Yuan](./3rdparty/TensorRT-LLM/README_Yuan.md)

## 源2.0 + FP8
性能测试

我们使用不同的数据类型对`2B`模型的预训练和微调分别进行测试，如下是测试结果，使用`FP8`相较于`BF16`有`30%`的性能提升。

|    times/step     | BF16  |     FP8     |
| :----------: | :------: | :-----------: |
| pretrain |    16.61    | 12.77| 
| instruct_tuning |    16.37    | 12.83|   

详细方案可以参考[Yuan_FP8](./docs/FP8.md)

## 联系我们
1.给我们发邮件：air_service@ieisystem.com

2.加入开发者微信群：
扫码关注“源AI看世界”公众号，发送消息 **“入群”** 获取开发者技术交流群二维码。
  ![Image text](https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/images/%E6%BA%90%E5%85%AC%E4%BC%97%E5%8F%B7%E4%BA%8C%E7%BB%B4%E7%A0%81.png)

## 招聘公告

我们正在招聘大模型框架研发、推理性能优化、开源社区运营方向相关专家。

请申请者将个人简历发送至邮箱(wushaohua@ieisystem.com)，并注明邮件主题”源项目团队应聘简历-个人名字”。
