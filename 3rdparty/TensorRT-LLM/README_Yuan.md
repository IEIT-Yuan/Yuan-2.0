# 基于TensorRT-LLM的Yuan 2.0推理服务部署

## 配置TensorRT-LLM和tensorrtllm_backend环境
目前存在几种访问 TensorRT-LLM 和 tensorrtllm_backend 的方法。对于 TensorRT-LLM 的 0.7.1-release 或更高版本，Yuan2.0 端到端部署的推荐方法是从[Triton NGC page](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)开始。
## Option 1: 运行我们提供的 Docker 容器
为了简化使用TensorRT-LLM和tensorrtllm_backend部署Yuan2.0的过程，我们提供了一个容器。其中包含创建、运行TensorRT-LLM模型和启动
tritonserver所需的一切环境和依赖。这个容器是构建自nvcr.io/nvidia/tritonserver:23.10-py3镜像，并按照tensorrtllm_backend仓库中的
Dockerfile一步步构建而成。
参考地址：https://hub.docker.com/repository/docker/zhaoxudong01/trt_llm_yuan/tags?page=1&ordering=last_updated
```bash
# 拉取我们提供的镜像
docker pull zhaoxudong01/trt_llm_yuan:v1.0
# 创建容器
docker run --gpus all -itd --network=host  -v your_filepath --cap-add=IPC_LOCK --device=/devinfiniband --privileged 
--name your_dockername --ulimit core=0 --ulimit memlock=1 --ulimit stack=68719476736 --shm-size=1000G 
zhaoxudong01/trt_llm_yuan:v1.0

# 接下来按照构建Yuan2.0 trt_llm_engine步骤操作即可
```

我们同时提供了该镜像的[百度网盘下载地址链接](https://pan.baidu.com/s/1l7y-dVUyJziJz-HCs4QfTA?pwd=a4fd)

## Option 2: 通过容器step by step构建
### Step 1. 拉取Nvidia镜像
从 23.10 基础容器开始，这个镜像包含 Triton 推理服务器，支持 Tensorflow、PyTorch、TensorRT、ONNX 和 OpenVINO 模型。
```bash
# 拉取官方镜像
docker pull nvcr.io/nvidia/tritonserver:23.10-py3

# 创建容器, 可以用"-v"挂载至你的本地目录
docker run --gpus all -itd --network=host  -v your_filepath --cap-add=IPC_LOCK --device=/devinfiniband --privileged 
--name your_dockername --ulimit core=0 --ulimit memlock=1 --ulimit stack=68719476736 --shm-size=1000G 
nvcr.io/nvidia/tritonserver:23.10-py3
# 进入容器
docker exec -it your_dockername bash
```
### Step 2. 通过tensorrtllm_backend的Dockerfile安装依赖和编译环境

在docker中，你可以按照下面描述的步骤配置你的环境。


```bash
# 进入你的工作目录
cd your_workspace

# 你可以在Yuan-2.0/3rdparty/tensorrtllm_backend/dockerfile/Dockerfile.trt_llm_backend目录下先预览这个文件，然后按照以下步骤
# 安装依赖并编译tensorrt_llm和triton_backend的环境step by step
cat Yuan-2.0/3rdparty/tensorrtllm_backend/dockerfile/Dockerfile.trt_llm_backend

ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver
ARG BASE_TAG=23.10-py3
......

# Step1: 进入tensorrtllm_backend目录，更新apt-get
cd Yuan-2.0/3rdparty/tensorrtllm_backend
apt-get update && apt-get install -y --no-install-recommends rapidjson-dev python-is-python3

# Step2: 在tensorrtllm_backend目录下，安装tensorrtllm_backend依赖
pip3 install -r requirements.txt --extra-index-url https://pypi.ngc.nvidia.com

# Step3: 在tensorrtllm_backend目录下，移除原有的tensorrt
apt-get remove --purge -y tensorrt*
pip uninstall -y tensorrt

# Step4: 在tensorrtllm_backend目录下，安装tensorrt
cp Yuan-2.0/3rdparty/TensorRT-LLM/docker/common/install_tensorrt.sh ./
bash install_tensorrt.sh
rm install_tensorrt.sh

# Step5: 在tensorrtllm_backend目录下，安装相关库(mpi4py、polygraphy、cmake、pytorch)
cp Yuan-2.0/3rdparty/TensorRT-LLM/docker/common/install_polygraphy.sh ./
bash install_polygraphy.sh
cp Yuan-2.0/3rdparty/TensorRT-LLM/docker/common/install_cmake.sh ./
bash install_cmake.sh
cp Yuan-2.0/3rdparty/TensorRT-LLM/docker/common/install_mpi4py.sh ./
bash install_mpi4py.sh
cp Yuan-2.0/3rdparty/TensorRT-LLM/docker/common/install_pytorch.sh ./
bash install_pytorch.sh
rm install_*
 
# Step6: 导出环境变量
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:${LD_LIBRARY_PATH}
export TRT_ROOT=/usr/local/tensorrt
export PATH="/usr/local/cmake/bin:${PATH}"

# Step7: 安装tensorrt_llm
cd Yuan-2.0 #进入Yuan-2.0目录下
git submodule init
git sunmodule update #更新submodules
cd 3rdparty/tensorrtllm_backend/  #回到tensorrtllm_backend目录
mkdir app # 在tensorrtllm_backend目录下新建文件夹
cd app
cp -r Yuan-2.0/3rdparty/TensorRT-LLM ./ # 将整个TensorRT-LLM文件拷贝至tensorrtllm_backend/app下
cd TensorRT-LLM # 进入tensorrtllm_backend/app/TensorRT-LLM目录
python3 scripts/build_wheel.py --trt_root="${TRT_ROOT}" -i -c # 安装tensorrt_llm


# Step8: 安装tensorrtllm_backend
cd .. # 回到app目录下
cp ../inflight_batcher_llm ./ # 复制此inflight_batcher_llm文件夹至app下 
mv TensorRT-LLM tensorrt_llm # 将TensorRT-LLM重命名为tensorrt_llm
cd inflight_batcher_llm
bash scripts/build.sh # 构建tensorrtllm_backend

# Step9: 安装tensorrt_llm wheel
cd Yuan-2.0/3rdparty/TensorRT_LLM/build
pip3 install tensorrt_llm-0.7.1-cp310-cp310-linux_x86_64.whl

# Step10: 拷贝相关文件至/opt/tritonserver/backend/tensorrtllm目录下
mkdir /opt/tritonserver/backends/tensorrtllm
cd ../../../ # 回到/tensorrtllm_backend目录
cp /app/tensorrt_llm/cpp/build/tensorrt_llm/libtensorrt_llm.so /opt/tritonserver/backend/tensorrtllm
cp /app/tensorrt_llm/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so* /opt/tritonserver/backend/tensorrtllm
cp /app/tensorrt_llm/tensorrt_llm/libs/ /opt/tritonserver/backend/tensorrtllm
cp /app/inflight_batcher_llm/build/libtriton_tensorrtllm.so /opt/tritonserver/backend/tensorrtllm

# 此时，您已完成运行TensorRT-LLM&tensorrtllm_backend时全部环境的配置
# 请确保在继续下一步之前已正确完成前一步骤
```

## 构建Yuan2.0 trt_llm_engine并启动tritonserver服务

以下是如何使用TensorRT-LLM构建Yuan2.0-2B trt_llm模型的示例。

### 准备Yuan2.0-2B的hf模型
下载Yuan2.0-2B hugging face模型，参考地址：https://huggingface.co/IEITYuan/Yuan2-2B-hf

### 构建Yuan2.0 trt_llm_engine

```bash
# 运行TensorRT-LLM/examples/yuan目录下build.py脚本，将--model_dir和--engine_dir替换为你的目录
python3 build.py --max_batch_size=4 --model_dir=<model_path> --dtype=float16  
--use_gemm_plugin=float16 --output_dir=<engine_path>

# 用TensorRT-LLM/examples/yuan目录下run.py脚本，测试Yuan2.0 trt_llm_engine
python3 run.py --max_output_len=100 --tokenizer_dir=<tokenizer_path>
--output_dir=<engine_path>
--input_text="写一篇春游作文<sep>"
```
当Yuan2.0 trt_llm_engine构建成功，预期输出结果:
```
Input [Text 0]: "写一篇春游作文<sep>"
Output [Text 0 Beam 0]: "亲爱的同学们：
春天到了，大地复苏，草木变绿，花儿绽放。这是一个美好的季节，也是我们展开青春梦想的舞台。今天，我们前往公园，感受春天的温暖。
早晨，阳光洒在大地上，给空气涂抹了一层斑斓的彩色，缤纷的色彩像云霞一样美丽。鸟儿开始欢快地舞蹈，漫过天空，唤醒大地。我们沐浴在温暖的阳光下，感受到来自大自然的恩赐。
漫步在小径上，我们"
```
### 复制模型文件
```bash
# 进入tensorrtllm_backend目录
cd Yuan-2.0/3rdparty/tensorrtllm_backend

# 拷贝模型文件
cp -r all_models/gpt triton_model_repo_yuan
```
### 修改tensorrtllm_backend下模型参数

在这个示例中，'triton_mode_repo_yuan' 目录中有四个模型将被使用：
- "preprocessing": 此模型用于进行编码，即从提示（字符串）转换为 input_ids（整数列表）
- "tensorrt_llm": 此模型是您的 TensorRT-LLM 模型的封装，用于推理。
- "postprocessing": 此模型用于解码，即将输出 ID（整数列表）转换为输出字符串。
- "ensemble": 此模型用于将上述三个模型连接在一起
preprocessing -> tensorrt_llm -> postprocessing

以下表格显示了在部署前需要修改的字段：

*triton_model_repo/preprocessing/config.pbtxt*

| Name |                                 Description                                 
| :----------------------: |:---------------------------------------------------------------------------:|
| `tokenizer_dir` | 模型的tokenizer路径. 在本案例中, 路径设置为 `/temp_data/LLM_test/Tensorrt-llm-yuan/yuan2B` |
| `tokenizer_type` |        模型的tokenizer类型，支持`t5`, `auto`, `llama` and `yuan`。 在此案例中，应当设置为`yuan`        |

*triton_model_repo/tensorrt_llm/config.pbtxt*

| Name |                                               Description                                               
| :----------------------: |:-------------------------------------------------------------------------------------------------------:|
| `gpt_model_path` | 部署TensorRT-LLM引擎的路径. 在此案例中, 路径应设置为`/temp_data/LLM_test/Tensorrt-llm-yuan/yuan2B/trt_engines/fp16/1-gpu` |

*triton_model_repo/postprocessing/config.pbtxt*

| Name |                                 Description                                 
| :----------------------: |:---------------------------------------------------------------------------:|
| `tokenizer_dir` | 模型的tokenizer路径. 在此案例中, 路径设置为 `/temp_data/LLM_test/Tensorrt-llm-yuan/yuan2B` |
| `tokenizer_type` |        模型的tokenizer类型，支持`t5`, `auto`, `llama` and `yuan`。 在此案例中，应当设置为`yuan`        |

### 启动Tritonserver服务

您可以使用以下命令启动 Triton 服务器：

```bash
# 进入tensorrtllm_backend目录下
cd Yuan-2.0/3rdparty/tensorrtllm_backend
# --world_size设置为1，目前暂不支持多卡(后续会更新)
python3 scripts/launch_triton_server.py --world_size=1 --model_repo=triton_model_repo_yuan
```

成功部署后，服务器会产生类似以下的日志：
```
I0919 14:52:10.475738 293 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
I0919 14:52:10.475968 293 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
I0919 14:52:10.517138 293 http_server.cc:187] Started Metrics Service at 0.0.0.0:8002
```

### 使用 Triton 的 generate 端点查询服务器

从 Triton 23.10 版本开始，您可以使用 Triton 的[generate endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md)，通过以下常规格式的 curl 命令在客户端环境/容器中查询服务器：

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "写一篇春游作文<sep>", "max_tokens": 100, "bad_words": "", "stop_words": "", "pad_id": 77185, "end_id": 77185}'
```

在这个例子中使用的模型情况下，您可以将 MODEL_NAME 替换为`ensemble`。查看 ensemble 模型的 config.pbtxt 文件，您会发现生成此模型的响应需要6个参数：

- "text_input": 输入文本以生成响应
- "max_tokens": 所请求的输出标记数
- "bad_words": 一个坏词列表（可以为空）
- "stop_words": 一个停止词列表（可以为空）
- "pad_id": 设置此索引为77185
- "eod_id": 设置此索引为77185

```bash
# 结束任务后用以下命令停止tritonserver
pkill tritonserver
```

## 性能测试

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

## 局限性
到目前为止，使用TensorRT-LLM构建的trt-llm-Yuan2.0引擎时：
- 暂不支持'gpt_attention'，'remove_input_padding'和'inflight_batching'
- 只支持Yuan2.0-2B单GPU部署。
