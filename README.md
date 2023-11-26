# Yuan2.0

[Read this in English.](./README-EN.md)

📔 更为详细的使用信息，可以参考：[Yuan2.0 论文](./docs/Yuan2.0_paper.pdf)



## 目录

- [Yuan2.0](#yuan20)
  - [目录](#目录)
  - [介绍](#介绍)
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

Yuan2.0 是浪潮信息发布的新一代预训练模型。我们开源了全部的3个模型Yuan2.0-102B，Yuan2.0-51B和Yuan2.0-2B。并且我们提供了预训练，微调，推理服务的相关脚本，以供研发人员做进一步的开发。Yuan2.0是在Yuan1.0的基础上，利用更多样的高质量预训练数据和指令微调数据集，令模型在语义、数学、推理、代码、知识等不同具备更强的理解能力。

-----

对本仓库源码的使用遵循开源许可协议 **[Apache 2.0](https://github.com/baichuan-inc/Baichuan-7B/blob/main/LICENSE)**。

源2.0模型支持商用，不需要申请授权，请您了解并遵循[《源2.0模型许可协议》](./LICENSE-YUAN)，勿将开源模型和代码及基于开源项目产生的衍生物用于任何可能给国家和社会带来危害的用途以及用于任何未经过安全评估和备案的服务。

尽管模型在训练时我们已采取措施尽力确保数据的合规性和准确性，但模型参数量巨大且受概率随机性因素影响，我们无法保证输出内容的准确性，且模型易被输入指令所误导，本项目不承担开源模型和代码导致的数据安全、舆情风险或发生任何模型被误导、滥用、传播、不当利用而产生的风险和责任。**您将对通过使用、复制、分发和修改模型等方式利用该开源项目所产生的风险与后果，独自承担全部责任。**

## 快速启动 

### 环境配置

我们建议使用有我们提供的最新的docker[镜像文件](https://pan.baidu.com/s/1IKjYqlf2kAPQzGsA6EdMCA?pwd=hopd).

我们可以通过下面命令启动容器：

```bash
docker load < ./yuan_v2.0.tar
docker run --gpus all -it -v /path/to/yuan_2.0:/workspace/yuan_2.0 -v /path/to/dataset:/workspace/dataset -v /path/to/checkpoints:/workspace/checkpoints yuan_v2.0:latest
```



### 数据预处理

我们提供了数据预处理的脚本，参考[数据预处理说明文档](./docs/data_process.md).

### 预训练

我们提供了用于预训练的文档和 [`example`](./examples)的脚本，具体使用方法可以参考[预训练说明文档](./docs/pretrain.md).



### 模型微调

请参考指令微调 [Yuan2.0 指令微调示例](./docs/instruct_tuning.md)。

请注意，不同的微调脚本对应的模型并不相同，请根据需要选择对应的模型。

### 模型

我们提供了Yuan2.0有监督微调的模型文件。模型文件可以通过下面的链接下载得到：

|    Model     | 序列长度  |         Download Link         |
| :----------: | :------: | :---------------------------: |
| Yuan2.0-102B |    4K    | [here](https://pan.baidu.com/s/1Tb9W6hEWS4bMkaE3p5s1fw?pwd=xrfo) |
| Yuan2.0-51B  |    4K    | [here](https://pan.baidu.com/s/1bOypWMepdh9GFK_hHXVQbQ?pwd=1uw3) |
|  Yuan2.0-2B  |    8K    | [here](https://pan.baidu.com/s/1Xj8Mi2tPwuuVu7Cb0tCbtw?pwd=qxpa) |

Yuan2.0-2B模型支持的序列长度为8192个tokens，Yuan2.0-51B和Yuan2.0-102B模型支持的序列长度为4096个tokens，可以根据用户设备的内存大小设置 `--max-position-embeddings` 和 `--seq-length` 的值。



## 评测结果

我们提供了[HumanEval](./docs/eval_humaneval.md)，[AGIEval-Math](./docs/eval_agieval_math.md)，[GSM-8K](./docs/eval_gsm8k.md)和[TruthfulQA](./docs/eval_truthfulqa.md)的评估脚本。在4个典型任务上，我们用Yuan-2.0不同版本模型上进行了性能测试。

| Model             | GSM8K   | AGIEval-GK-Math-QA     | AGIEval-GK-Math-Cloze     | HumanEval | TurthfulQA |
| ----------------- | :----:  | :------------: | :---------------: | :-------: | ---------- |
|  GPT-4            |  92%    |     47.0%      |       16.1%       |   86.6%   |     59%    |
|  Chat-GPT         | 68.6%\* |     36.5%      |        7.3%       |  66.8%\*  |     34%\*  |
|  Llama2           | 56.8%   |       -        |         -         |   29.9%   |       -    |
| Yuan2.0-102B      | 76.6%   |     38.7%      |       13.5%       |   67.1%   |       -    |
| Yuan2.0-102B-SC   |   -     |       -        |        -          |   77.4%   |       -    |

\* 使用与源2.0完全相同的输入数据对ChatGPT进行测试，时间2023年11月

## 代码调用 

考虑到推理服务的效率，Yuan2.0-51B和Yuan2.0-102B模型在启动推理服务之前，需要将模型转换成只有张量并行的模型文件。可以参考[文档](./docs/checkpoint_process.md)

可以通过调用推理服务，向推理服务发送请求实现模型的调用，[Yuan2.0 推理服务](./docs/run_inference_server.md)



