# Yuan2.0

[中文文档请点这里](./README.md)

📔  For more detailed usage information, please refer to [Yuan2.0 Paper](https://arxiv.org/ftp/arxiv/papers/2311/2311.15786.pdf)



## Table of Contents
- [Yuan2.0](#yuan20)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Quick Start](#quick-start)
    - [Environment Config](#environment-config)
    - [Data preprocess](#data-preprocess)
    - [Pretrain](#pretrain)
    - [Model Fine-tuning](#model-fine-tuning)
    - [Models](#models)
    - [Evaluation](#evaluation)
  - [Inference Service](#inference-service)


<!-- markdown-toc end -->

## Introduction

Yuan2.0 is a new generation Fundamental Large Language Model developed by IEIT System. We have published all three models, Yuan 2.0-102B, Yuan 2.0-51B, and Yuan 2.0-2B. And we provide relevant scripts for pretraining, fine-tuning, and inference services for other developers. Yuan2.0 is based on Yuan1.0, utilizing a wider range of high-quality pre training data and instruction fine-tuning datasets to enhance the model's understanding of semantics, mathematics, reasoning, code, knowledge, and other aspects.

---

The use of the source code in this repository requires compliance with the open source license agreement **Apache 2.0**.
The Yuan2.0 model supports commercial use and does not require authorization. Please understand and comply with the [《Yuan 2.0 Model License Agreement》](./LICENSE-Yuan). Do not use the open source model and code, as well as derivatives generated from open source projects, for any purposes that may cause harm to the country and society, or for any services that have not undergone security assessment and filing.
Although we have taken measures to ensure the compliance and accuracy of the data during training, the model has a huge number of parameters and is affected by probability and randomness factors. We cannot guarantee the accuracy of the output content, and the model is easily misled by input instructions. This project does not assume any data security, public opinion risks, or any model misleading, abusing, spreading caused by open-source models and code Risks and responsibilities arising from improper utilization  **You will be solely responsible for the risks and consequences arising from the use, copying, distribution, and modification of the model in this open source project**




## Quick Start 

See detail documentation here [Quickstart](Quickstart.md).

### Environment Config

We strongly recommend using the latest release of docker images we provided [here](https://hub.docker.com/r/yuanmodel/yuan2.0).

You can launch an instance of the Yuan 2.0 container with the following Docker commands:

```bash
docker load < ./yuan_v2.0.tar
docker run --gpus all -it --rm -v /path/to/yuan_2.0:/workspace/yuan_2.0 -v /path/to/dataset:/workspace/dataset -v /path/to/checkpoints:/workspace/checkpoints yuan_v2.0:latest
```



### Data preprocess

We have provided the data preprocess script. See documentation [here](./docs/data_process.md).

### Pretrain

We've provided several scripts for pretraining in the [`example`](./examples). The details can be seen from documentation [here](./docs/pretrain.md).

### Model Fine-tuning

We also have provided the supervised fine-tuning script. See documentation [here](./docs/checkpoint_process.md).

### Models

🥇🥇🥇 **<font color=#FFC125 >We have provided Yuan2.0 supervised-finetuned checkpoints. The checkpoint files of the models through the following links:</font>**


### <font color=#FFC125 >Hugging Face Version </font> 



|    Model     | Seq Len  |         Download Link         |
| :----------: | :------: | :---------------------------: |
| Yuan2.0-102B-hf |    4K    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2.0-102B-hf/summary) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-102B-hf) \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-102B-hf)  \|  [百度网盘](https://pan.baidu.com/s/1O4GkPSTPu5nwHk4v9byt7A?pwd=pq74#list/path=%2F) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-102B-hf)|
| Yuan2.0-51B-hf  |    4K    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2.0-51B-hf/summary)  \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2.0-51B-hf)  \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-51B-hf)   \| [百度网盘](https://pan.baidu.com/s/1-qw30ZuyrMfraFtkLgDg0A?pwd=v2nd#list/path=%2F) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-51B-hf)|
|  Yuan2.0-2B-hf  |    8K    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2.0-2B-hf/summary)   \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-2B-hf)   \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-2B-hf)    \| [百度网盘](https://pan.baidu.com/s/1nt-03OAnjtZwhiVywj3xGw?pwd=nqef#list/path=%2F) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-2B-hf) |



### <font color=#FFC125 >Origin Version </font> 


|    Model     | Seq Len  |         Download Link         |
| :----------: | :------: | :---------------------------: |
| Yuan2.0-102B |    4K    | [ModelScope](https://www.modelscope.cn/models/YuanLLM/Yuan2.0-102B/summary)  \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2.0-102B)  \|  [百度网盘](https://pan.baidu.com/s/1Tb9W6hEWS4bMkaE3p5s1fw?pwd=xrfo) \| [WiseModel](https://wisemodel.cn/models/IEIT-Yuan/Yuan2.0-102B)|
| Yuan2.0-51B  |    4K    | [ModelScope](https://www.modelscope.cn/models/YuanLLM/Yuan2.0-51B/summary)  \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2.0-51B)   \| [百度网盘](https://pan.baidu.com/s/1bOypWMepdh9GFK_hHXVQbQ?pwd=1uw3) \| [WiseModel](https://wisemodel.cn/models/IEIT-Yuan/Yuan2.0-51B)|
|  Yuan2.0-2B  |    8K    | [ModelScope](https://www.modelscope.cn/models/YuanLLM/Yuan2.0-2B/summary)   \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2.0-2B)    \| [百度网盘](https://pan.baidu.com/s/1Xj8Mi2tPwuuVu7Cb0tCbtw?pwd=qxpa) \| [WiseModel](https://wisemodel.cn/models/IEIT-Yuan/Yuan2.0-2B) |


Yuan2.0-2B model support sequence length up to 8192 tokens,  Yuan2.0-51B and Yuan2.0-102B models support sequence length up to 4096 tokens, you and set `--max-position-embeddings` and `--seq-length` values according to your device memory.

### Evaluation

We provide evaluation scripts for [HumanEval](./docs/eval_humaneval.md)，[AGIEval-Math](./docs/eval_agieval_math.md)，[GSM-8K](./docs/eval_gsm_8k.md) and [TruthfulQA](./docs/eval_TruthfulQA.md) for users to reproduce ours numbers. We conducted performance tests on different size of Yuan 2.0 models, which can be found in our paper.

| Model             | GSM8K   | AGIEval-GK-Math-QA     | AGIEval-GK-Math-Cloze     | HumanEval | TurthfulQA |
| ----------------- | :----:  | :------------: | :---------------: | :-------: | ---------- |
|  GPT-4            |  92%    |     47.0%      |       16.1%       |   86.6%   |     59%    |
|  Chat-GPT         | 68.6%\* |     36.5%      |        7.3%       |  66.5%\*  |     34%\*  |
|  Llama2           | 56.8%   |       -        |         -         |   29.9%   |       -    |
| Yuan2.0-102B      | 76.6%   |     38.7%      |       13.5%       |   67.1%   |     58%    |
| Yuan2.0-102B-SC   | 86.2%   |     45.5%      |       15.2%       |   77.4%   |       -    |

\* Evaluate ChatGPT using exactly the same input data as Yuan 2.0 in November 2023

## Inference Service

For the inference efficiency, the Yuan2.0-51B and Yuan2.0-102B models need to be converted into model files with only tensor parallelism before starting the inference service. The details can be seen from documentation [here](./docs/checkpoint_process.md).

You can call the model by calling the inference service and sending a request to it. The details can be seen from documentation [here](./docs/inference_server.md).
