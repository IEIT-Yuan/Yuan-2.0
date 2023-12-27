# 源2.0

[Read this in English.](./README-EN.md)

📔 更为详细的使用信息，可以参考：[源2.0 论文](https://arxiv.org/ftp/arxiv/papers/2311/2311.15786.pdf)



## 目录

- [源2.0](#源20)
  - [目录](#目录)
  - [介绍](#介绍)
  - [源大模型共训计划](#源大模型共训计划)
  - [快速启动](#快速启动)
    - [环境配置](#环境配置)
    - [数据预处理](#数据预处理)
    - [预训练](#预训练)
    - [模型微调](#模型微调)
    - [模型](#模型)
  - [评测结果](#评测结果)

  - [代码调用](#代码调用)


<!-- markdown-toc end -->

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

### 模型

我们提供了源2.0有监督微调的模型文件。模型文件可以通过下面的链接下载得到：

|    Model     | 序列长度  |         Download Link         |
| :----------: | :------: | :---------------------------: |
| 源2.0-102B |    4K    | [ModelScope](https://www.modelscope.cn/models/YuanLLM/Yuan2.0-102B/files) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2.0-102B-hf) \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2.0-102B)  \|  [百度网盘](https://pan.baidu.com/s/1Tb9W6hEWS4bMkaE3p5s1fw?pwd=xrfo) \| [WiseModel](https://wisemodel.cn/models/IEIT-Yuan/Yuan2.0-102B)|
| 源2.0-51B  |    4K    | [ModelScope](https://www.modelscope.cn/models/YuanLLM/Yuan2.0-51B/files)  \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2.0-51B-hf)  \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2.0-51B)   \| [百度网盘](https://pan.baidu.com/s/1bOypWMepdh9GFK_hHXVQbQ?pwd=1uw3) \| [WiseModel](https://wisemodel.cn/models/IEIT-Yuan/Yuan2.0-51B)|
|  源2.0-2B  |    8K    | [ModelScope](https://www.modelscope.cn/models/YuanLLM/Yuan2.0-2B/files)   \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2.0-2B-hf)   \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2.0-2B)    \| [百度网盘](https://pan.baidu.com/s/1Xj8Mi2tPwuuVu7Cb0tCbtw?pwd=qxpa) \| [WiseModel](https://wisemodel.cn/models/IEIT-Yuan/Yuan2.0-2B) |

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

详细启动推理服务的流程可以参考[Yuan2_inference_guide文档](./docs/Yuan2_inference_guide_cn.md)

## 联系我们
1.给我们发邮件：air_service@ieisystem.com

2.加入开发者微信群：
扫码关注“源AI看世界”公众号，发送消息 **“入群”** 获取开发者技术交流群二维码。
  ![Image text](https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/images/%E6%BA%90%E5%85%AC%E4%BC%97%E5%8F%B7%E4%BA%8C%E7%BB%B4%E7%A0%81.png)

## 招聘公告

我们正在招聘大模型框架研发、推理性能优化、开源社区运营方向相关专家。

请申请者将个人简历发送至邮箱(wushaohua@ieisystem.com)，并注明邮件主题”源项目团队应聘简历-个人名字”。
