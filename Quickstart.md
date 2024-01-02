# Quickstart
## 0. 须知
开始上手之前，请确保运行环境（[docker镜像](https://hub.docker.com/r/yuanmodel/yuan2.0)）、[源码](https://github.com/IEIT-Yuan/Yuan-2.0/)、[模型](https://huggingface.co/IEITYuan/Yuan2-2B-hf)已就绪。

若想再容器外面访问对应模型的推理接口，可以再docker run 的时候额外增加`--network=host` 参数。
## 1. 推理
### 1.1 推理运行
不同参数的模型推理脚本对应于：[run_inference_server_2.1B.sh](./examples/run_inference_server_2.1B.sh)、[run_inference_server_51B.sh](./examples/run_inference_server_51B.sh)、[run_inference_server_102B.sh](./examples/run_inference_server_102B.sh)

对于2B模型，修改脚本中的参数：`CHECKPOINT_PATH`，替换对应的2B模型路径，直接执行即可，注意脚本执行路径。
```bash
bash +x examples/run_inference_server_2.1B.sh
```
<mark>**注意**<mark>：51B和102B模型，需要进行转换，

51B详情参见文档[Yuan2_inference_guide_cn.md](./docs/Yuan2_inference_guide_cn.md#font-colorffc125-源20-51b-模型-font-).
102B详情参见文档[Yuan2_inference_guide_cn.md](./docs/Yuan2_inference_guide_cn.md#font-colorffc125-源20-102b-模型font-).
### 1.2 推理结果验证
方式一：

- 使用Curl进行测试

```
#如下命令返回Unicode编码
curl http://YourIP:YourPort/yuan -X PUT   \
--header 'Content-Type: application/json' \
--data '{"ques_list":[{"id":"000","ques":"请帮忙作一首诗，主题是冬至"}], "tokens_to_generate":500, "top_k":5}'

#如下命令返回原始形式
echo -en "$(curl -s  http://127.0.0.1:8000/yuan -X PUT  --header 'Content-Type: application/json' --data '{"ques_list":[{"id":"000","ques":"作一首词 ，主题是冬至"}], "tokens_to_generate":500, "top_k":5}')"
```
方式二：

- 使用第三方工具，比如Postman，注意请求方法是`PUT`方法，header是`application/json`,body体是json格式，如：
```
{
    "ques_list":[
        {
            "id":"000",
            "ques":"请帮忙作一首诗，主题是冬至"
        }
    ],
    "tokens_to_generate":500,
    "top_k":5
}
```
方式三：
- 使用Python进行测试，我们提供了一个示例代码来测试API调用的性能，运行前注意将代码中 `ip`和`port` 根据api部署情况进行修改。

```bash
python tools/start_inference_server_api.py
```
## 2. 微调
### 2.1 数据集准备
数据格式以及数据处理参考文档：[data_process_cn.md](./docs/data_process_cn.md)

### 2.2 微调运行
不同参数的模型问题脚本对应于：[pretrain_yuan2.0_2.1B_sft.sh](./examples/pretrain_yuan2.0_2.1B_sft.sh)、[pretrain_yuan2.0_51B_sft.sh](./examples/pretrain_yuan2.0_51B_sft.sh)、[pretrain_yuan2.0_102B_sft.sh](./examples/pretrain_yuan2.0_102B_sft.sh)

对于2B模型，修改脚本中的参数：`CHECKPOINT_PATH`，替换对应的2B模型路径；修改脚本中的参数：`DATA_PATH`，替换对应的微调数据集路径；修改脚本中的参数：`TOKENIZER_MODEL_PATH`，替换对应的tokenizer路径，默认是`./tokenizer`；修改脚本中的参数：`TENSORBOARD_PATH`，该参数是输出路径，真实存在即可；直接执行，注意脚本执行路径。
```bash
bash +x examples/pretrain_yuan2.0_2.1B_sft.sh
```
<mark>**注意**<mark>：51B和102B模型，不需要进行转换，直接加载即可。

## 3. 评测结果复现
我们提供了[HumanEval](./docs/eval_humaneval.md)，[AGIEval-GK-Math](./docs/eval_agieval_math_cn.md)，[GSM8K](./docs/eval_gsm8k_cn.md)和[TruthfulQA](./docs/eval_TruthfulQA.md)的评估脚本，以方便大家复现我们的评测结果。在4个典型任务上，我们在论文中给出了源2.0不同尺寸模型的精度。

<mark>**注意**<mark>：51B和102B模型，需要进行转换，转换脚本参见[ckpt_partitions_51B.sh](./examples/ckpt_partitions_51B.sh)、[ckpt_partitions_102B.sh](./examples/ckpt_partitions_102B.sh)