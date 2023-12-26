# 快速推理指引

本脚本介绍了102B模型和51B模型快速使用指引，主要包括ckpt转换以及推理服务使用

## 102B模型：

### step1：

首先需要转换ckpt，我们提供的102B的模型文件是32路流水并行-1路张量并行（32pp，1tp）的模型文件，为了提高推理效率，需要将32路流水并行的模型文件转换为8路张量并行的模型文件（适用于80GB GPU），转换流程是：

（32路流水-1路张量）->（32路流水-8路张量）->（1路流水-8路张量）

我们提供了自动转换脚本，可以依次执行完上述流程，使用方式如下：

```
1. vim examples/ckpt_partitions_102B.sh

2. 修改如下三个环境变量（LOAD_CHECKPOINT_PATH，SAVE_SPLITED_CHECKPOINT_PATH，SAVE_CHECKPOINT_PATH）：

LOAD_CHECKPOINT_PATH 表示Yuan2.0开源的原始32路流水并行的模型文件路径，需要路径下面包含latest_checkpointed_iteration.txt这个文件（这个文件会在开源的模型文件中），一个示例如下：
LOAD_CHECKPOINT_PATH=/mnt/102B

SAVE_SPLITED_CHECKPOINT_PATH 表示转换产生的中间文件路径，主要作用于（32路流水-1路张量）->（32路流水-8路张量）产生的结果文件，等脚本完成转换之后可以删除该文件，一个示例如下：
SAVE_SPLITED_CHECKPOINT_PATH=./ckpt-102B-mid

SAVE_CHECKPOINT_PATH 表示最终产生的8路张量并行的ckpt，主要作用于（32路流水-8路张量）->（1路流水-8路张量）产生的结果文件，一个示例如下：
SAVE_CHECKPOINT_PATH=./ckpt-102B-8tp

如果在megatron主目录下面运行脚本，可以不修改TOKENIZER_MODEL_PATH=./tokenizer（因为github主目录下面包含tokenizer），否则需要指定tokenizer路径

3. bash examples/ckpt_partitions_102B.sh
```

在上述步骤完成后，会在`SAVE_CHECKPOINT_PATH`指定的目录下面生成一个8路张量并行的ckpt，可以用于推理服务。

### step2：

```
1. 启动推理服务，需要在脚本examples/run_inference_server_102B.sh中修改环境变量CHECKPOINT_PATH：

vim examples/run_inference_server_102B.sh

修改环境变量CHECKPOINT_PATH为step1中SAVE_CHECKPOINT_PATH指定的路径，比如以上示例中SAVE_CHECKPOINT_PATH=./ckpt-102B-8tp，那么需要指定examples/run_inference_server_102B.sh中CHECKPOINT_PATH=./ckpt-102B-8tp

2. 启动推理服务（需要8张80GB GPU卡）：
#注意，程序默认的端口号为8000，如果8000被占用，需要修改examples/run_inference_server_102B.sh中的环境变量PORT为实际使用的端口号
bash examples/run_inference_server_102B.sh

等待程序完成cpkt的加载并出现如下信息后，可以执行下一步调用推理服务：
  successfully loaded checkpoint from ./ckpt-102B-8tp at iteration 1

 * Serving Flask app 'megatron.text_generation_server'
 * Debug mode: off
   WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8000
 * Running on http://127.0.0.1:8000

3. 在相同docker中调用推理服务：
#注意，程序默认的端口号为8000，如果8000被占用，需要修改tools/start_inference_server_api.py中的request_url="http://127.0.0.1:8000/yuan"
中的端口号为实际使用的端口号
python tools/start_inference_server_api.py

如果运行成功会返回推理结果
```


## 51B模型：

### step1：

首先需要转换ckpt，我们提供的51B的模型文件是16路流水并行-1路张量并行（16pp，1tp）的模型文件，为了提高推理效率，需要将16路流水并行的模型文件转换为4路张量并行的模型文件（适用于80GB GPU），转换流程是：

（16路流水-1路张量）->（16路流水-4路张量）->（1路流水-4路张量）

我们提供了自动转换脚本，可以依次执行完上述流程，使用方式如下：

```
1. vim examples/ckpt_partitions_51B.sh

2. 修改如下三个环境变量（LOAD_CHECKPOINT_PATH，SAVE_SPLITED_CHECKPOINT_PATH，SAVE_CHECKPOINT_PATH）：

LOAD_CHECKPOINT_PATH 表示Yuan2.0开源的原始16路流水并行的模型文件路径，需要路径下面包含latest_checkpointed_iteration.txt这个文件（这个文件会在开源的模型文件中），一个示例如下：
LOAD_CHECKPOINT_PATH=/mnt/51B

SAVE_SPLITED_CHECKPOINT_PATH 表示转换产生的中间文件路径，主要作用于（16路流水-1路张量）->（16路流水-4路张量）产生的结果文件，等脚本完成转换之后可以删除该文件，一个示例如下：
SAVE_SPLITED_CHECKPOINT_PATH=./ckpt-51B-mid

SAVE_CHECKPOINT_PATH 表示最终产生的4路张量并行的ckpt，主要作用于（16路流水-4路张量）->（1路流水-4路张量）产生的结果文件，一个示例如下：
SAVE_CHECKPOINT_PATH=./ckpt-51B-4tp

如果在megatron主目录下面运行脚本，可以不修改TOKENIZER_MODEL_PATH=./tokenizer（因为github主目录下面包含tokenizer），否则需要指定tokenizer路径

3. bash examples/ckpt_partitions_51B.sh
```

在上述步骤完成后，会在`SAVE_CHECKPOINT_PATH`指定的目录下面生成一个4路张量并行的ckpt，可以用于推理服务。

### step2：

```
1. 启动推理服务，需要在脚本examples/run_inference_server_51B.sh中修改环境变量CHECKPOINT_PATH：

vim examples/run_inference_server_51B.sh

修改环境变量CHECKPOINT_PATH为step1中SAVE_CHECKPOINT_PATH指定的路径，比如以上示例中SAVE_CHECKPOINT_PATH=./ckpt-51B-4tp，那么需要指定examples/run_inference_server_51B.sh中CHECKPOINT_PATH=./ckpt-51B-4tp

2. 启动推理服务（需要4张80GB GPU卡）：
#注意，程序默认的端口号为8000，如果8000被占用，需要修改examples/run_inference_server_51B.sh中的环境变量PORT为实际使用的端口号
bash examples/run_inference_server_51B.sh

等待程序完成cpkt的加载并出现如下信息后，可以执行下一步调用推理服务：
  successfully loaded checkpoint from ./ckpt-51B-4tp at iteration 1

 * Serving Flask app 'megatron.text_generation_server'
 * Debug mode: off
   WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8000
 * Running on http://127.0.0.1:8000

3. 在相同docker中调用推理服务：
#注意，程序默认的端口号为8000，如果8000被占用，需要修改tools/start_inference_server_api.py中的request_url="http://127.0.0.1:8000/yuan"
中的端口号为实际使用的端口号
python tools/start_inference_server_api.py

如果运行成功会返回推理结果
```
