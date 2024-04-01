# 基于vLLM的Yuan 2.0推理服务部署

## 配置vLLM环境
vLLM环境配置主要分为以下两步，拉取Nvidia官方的镜像创建docker，以及安装vllm运行环境
### Step 1. 拉取Nvidia提供的pytorch镜像

```bash
# 拉取官方镜像
docker pull nvcr.io/nvidia/pytorch:23.07-py3

# 创建容器, 可以用"-v"挂载至你的本地目录
docker run --gpus all -itd --network=host  -v your_filepath --cap-add=IPC_LOCK --device=/devinfiniband --privileged 
--name your_dockername --ulimit core=0 --ulimit memlock=1 --ulimit stack=68719476736 --shm-size=1000G 
nvcr.io/nvidia/pytorch:23.07-py3

# 进入容器
docker exec -it your_dockername bash
```
### Step 2. 安装vLLM运行环境

```bash
# 进入你的工作目录
cd /your_workspace

# 拉取我们的项目
git clone https://github.com/IEIT-Yuan/Yuan-2.0.git

# 进入vLLM项目
cd Yuan-2.0/3rdparty/vllm

# 安装依赖
pip install -r requirements.txt

# 安装vllm
pip install -e .
```

## Yuan2.0-2B模型基于vLLM的推理和部署

以下是如何使用vLLM推理框架对Yuan2.0-2B模型进行推理和部署的示例

### Step 1. 准备Yuan2.0-2B的hf模型
下载Yuan2.0-2B hugging face模型，参考地址：https://huggingface.co/IEITYuan/Yuan2-2B-hf

将下载好的Yuan2.0-2B模型的ckpt移动至你的本地目录下(本案例中的路径如下：/temp_data/LLM_test/Tensorrt-llm-yuan/yuan2B_Janus)
### Step 2. 基于Yuan2.0-2B的vllm推理
#### Option1:单个prompt推理
```bash
# 编辑test_yuan_1prompt.py
vim test_yuan_1prompt.py

# 1.修改LLM模型路径(必选)
# 2.修改prompts提示词内容(可选)
# 3.修改sampling_params参数(可选)
'''
prompts = ["如果你是一个算法工程师，让你写一个大模型相关的规划，你应该怎么写？"]
sampling_params = SamplingParams(max_tokens=300, temperature=1, top_p=0, top_k=1, min_p=0.0, length_penalty=1.0, repetition_penalty=1.0, stop="<eod>", )

llm = LLM(model="/temp_data/LLM_test/Tensorrt-llm-yuan/yuan2B_Janus", trust_remote_code=True)
'''
```
以脚本中prompt为例进行测试的预期输出结果如下:
```
Prompt: '如果你是一个算法工程师，让你写一个大模型相关的规划，你应该怎么写？', Generated text: '<sep> 作为一个算法工程师，编写一个大模型相关的规划需要考虑以下几个方面：\n1. 目标设定：明确你的规划的目标是什么。是为了提高模型的精度和准确性，还是为了实现特定的应用场景或其他目标？\n2. 数据收集和预处理：确定如何处理大量的数据并对其进行预处理。这包括数据清洗、去噪、特征选择、特征提取等步骤。\n3. 算法选择和设计：根据目标和数据特征，选择合适的算法进行建模。例如，可以使用监督学习算法（如支持向量机、决策树等）或无监督学习算法（如聚类、异常检测等）。\n4. 模型参数和优化：确定模型的参数和超参数，以优化模型的性能。可以使用学习率、正则化等技术来调整模型的行为。\n5. 训练和测试：使用一部分数据来训练模型，然后使用另一部分数据来评估模型的性能。可以使用交叉验证等技术来对模型进行评估。\n6. 模型部署和效果预测：当模型性能达到预期要求时，可以考虑将其部署到实际场景中，以实现自动化的预测和决策。\n7. 监测和优化：定期监测模型的性能，并根据监测结果进行优化。可以使用指标分析、可视化等方法来评估模型的效果。\n8. 安全和合规性：确保模型在安全和合规方面的考虑，包括数据隐私保护、模型'
```
#### Option 2. 多个prompt推理
```bash
# 编辑test_yuan_3prompt.py
vim test_yuan_3prompt.py

# 1.修改LLM模型路径(必选)
# 2.修改prompts提示词内容或者增加/减少prompts(可选)
# 3.修改sampling_params参数(可选)
'''
prompts = [
        "如果你是一个算法工程师，让你写一个大模型相关的规划，你应该怎么写？",
        "写一些 VR 和健身结合的方案",
        "世界上最高的山是什么",
    ]

sampling_params = SamplingParams(max_tokens=256, temperature=0.8, top_p=0.95, stop="<eod>")

llm = LLM(model="/temp_data/LLM_test/Tensorrt-llm-yuan/yuan2B_Janus", trust_remote_code=True)
'''
# 注意：用多个prompt进行推理时，可能由于补padding的操作，和用单个prompt推理时结果不一样
```

### Step 3. 基于vllm.entrypoints.api_server部署Yuan2.0-2B
基于api_server部署Yuan2.0-2B的步骤包括推理服务的发起和调用。其中调用vllm.entrypoints.api_server推理服务有以下两种方式：第一种是通过命令行直接调用；第二种方式是通过运行脚本批量调用。
```bash
# 发起服务，--model后修改为您的ckpt路径
python -m vllm.entrypoints.api_server --model=/temp_data/LLM_test/Tensorrt-llm-yuan/yuan2B_Janus/ --trust-remote-code

# 发起服务后，显示如下：
INFO 03-27 08:35:47 llm_engine.py:73] Initializing an LLM engine with config: model='/temp_data/LLM_test/Tensorrt-llm-yuan/yuan2B_Janus/', tokenizer='/temp_data/LLM_test/Tensorrt-llm-yuan/yuan2B_Janus/', tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=auto, tensor_parallel_size=1, quantization=None, enforce_eager=False, seed=0)
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
INFO 03-27 08:35:53 llm_engine.py:230] # GPU blocks: 11227, # CPU blocks: 780
INFO 03-27 08:35:55 model_runner.py:425] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 03-27 08:35:55 model_runner.py:429] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode.
INFO 03-27 08:36:03 model_runner.py:471] Graph capturing finished in 8 secs.
INFO:     Started server process [1628]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
#### Option 1. 基于命令行调用服务
```bash
# 使用命令行调用服务的指令如下：
curl http://localhost:8000/generate -d '{"prompt": "如果你是一个算法工程师，让你写一个大模型相关的规划，你应该怎么写？", "use_beam_search": false,  "n": 1, "temperature": 1, "top_p": 0, "top_k": 1,  "max_tokens":256, "stop": "<eod>"}'

# 预期输出结果如下：
{"text":["如果你是一个算法工程师，让你写一个大模型相关的规划，你应该怎么写？<sep> 作为一个算法工程师，编写一个大模型相关的规划需要考虑以下几个方面：\n1. 目标设定：明确你的规划的目标是什么。是为了提高模型的精度和准确性，还是为了实现特定的应用场景或其他目标？\n2. 数据收集和预处理：确定如何处理大量的数据并对其进行预处理。这包括数据清洗、去噪、特征选择、特征提取等步骤。\n3. 算法选择和设计：根据目标和数据特征，选择合适的算法进行建模。例如，可以使用监督学习算法（如支持向量机、决策树等）或无监督学习算法（如聚类、异常检测等）。\n4. 模型参数和优化：确定模型的参数和超参数，以优化模型的性能。可以使用学习率、正则化等技术来调整模型的行为。\n5. 训练和测试：使用一部分数据来训练模型，然后使用另一部分数据来评估模型的性能。可以使用交叉验证等技术来对模型进行评估。\n6. 模型部署和效果预测：当模型性能达到预期要求时，可以考虑将其部署到实际场景中，以实现自动化的预测和决策。\n7. 监测和优化：定期监测模型的性能"]}

# 服务发起端输出结果显示如下：
INFO 03-27 08:38:50 async_llm_engine.py:377] Received request 1d6a3ed293ac40e2ace25813130ad4bf: prompt: '如果你是一个算法工程师，让你写一个大模型相关的规划，你应该怎么写？', sampling params: SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1, top_p=0, top_k=1, min_p=0.0, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=['<eod>'], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=256, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True), prompt token ids: None.
INFO 03-27 08:38:50 llm_engine.py:655] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%
INFO 03-27 08:38:54 async_llm_engine.py:111] Finished request 1d6a3ed293ac40e2ace25813130ad4bf.
INFO:     127.0.0.1:50013 - "POST /generate HTTP/1.1" 200 OK
```
#### Option 2. 基于命令脚本调用服务
调用openai.api_server的相关脚本为yuan_api_server.py，内容如下
```bash
import requests
import json

outputs = []
with open('/mnt/Yuan-2.0/3rdparty/vllm/humaneval/human-eval-gpt4-translation-fixed5.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        prompt = data.get('prompt')
        raw_json_data = {
                "prompt": prompt,
                "logprobs": 1,
                "max_tokens": 256,
                "temperature": 1,
                "use_beam_search": False,
                "top_p": 0,
                "top_k": 1,
                "stop": "<eod>",
                }
        json_data = json.dumps(raw_json_data)
        headers = {
                "Content-Type": "application/json",
                }
        response = requests.post(f'http://localhost:8000/generate',
                             data=json_data,
                             headers=headers)
        output = response.text
        output = json.loads(output)
        output = output['text']
        outputs.append(output[0])
print(outputs)   #可以选择打印输出还是储存到新的jsonl文件
...

# 示例中是读取的中文版本humaneval测试集，通过批量调用推理服务并将结果保存在对应的jsonl文件中
# 您可以代码中读取的jsonl文件路径替换为您的路径
# 或者在此代码基础上进行修改，例如手动传入多个prompts以批量调用api_server进行推理
```
修改完成后运行以下命令脚本调用推理服务即可
```bash
python yuan_api_server.py
```
### Step 4. 基于vllm.entrypoints.openai.api_server部署Yuan2.0-2B
基于openai的api_server部署Yuan2.0-2B的步骤和step 3的步骤类似，发起服务和调用服务的方式如下：

发起服务命令：
```bash
python -m vllm.entrypoints.openai.api_server --model=/temp_data/LLM_test/Tensorrt-llm-yuan/yuan2B_Janus/ --trust-remote-code
```
调用服务命令：
```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "/temp_data/LLM_test/Tensorrt-llm-yuan/yuan2B_Janus/", "prompt": "如果你是一个算法工程师，让你写一个大模型相关的规划，你应该怎么写？", "max_tokens": 300, "temperature": 1, "top_p": 0, "top_k": 1, "stop": "<eod>"}'
```
调用服务脚本如下：
```bash
import requests
import json

outputs = []
with open('/mnt/Yuan-2.0/3rdparty/vllm/humaneval/human-eval-gpt4-translation-fixed5.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        prompt = data.get('prompt')
        raw_json_data = {
                "model": "/temp_data/LLM_test/Tensorrt-llm-yuan/yuan2B_Janus/",
                "prompt": prompt,
                "max_tokens": 256,
                "temperature": 1,
                "use_beam_search": False,
                "top_p": 0,
                "top_k": 1,
                "stop": "<eod>",
                }
        json_data = json.dumps(raw_json_data, ensure_ascii=True)
        headers = {
                "Content-Type": "application/json",
                }
        response = requests.post(f'http://localhost:8000/v1/completions',
                             data=json_data,
                             headers=headers)
        output = response.text
        output = json.loads(output)
        output0 = output["choices"][0]['text']
        outputs.append(output0)
...
# 此脚本您需要修改"model"后的ckpt路径，其他修改方式和yuan_api_server.py一致
```
