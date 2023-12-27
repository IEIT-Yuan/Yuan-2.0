



### <font color=#FFC125 >源2.0-102B 模型</font> 

-----
**本脚本为102B模型的快速使用指引，主要包括ckpt转换以及推理服务使用**
### <strong>🔘 步骤 1</strong> 

首先需要转换ckpt，我们提供的102B的模型文件是32路流水并行-1路张量并行（32pp，1tp）的模型文件，为了提高推理效率，需要将32路流水并行的模型文件转换为8路张量并行的模型文件（适用于80GB GPU），转换流程是：

> （32路流水-1路张量）->（32路流水-8路张量）->（1路流水-8路张量）


我们提供了自动转换脚本，可以依次执行完转换流程，使用方式如下：


**<font color=#FFFFF0 >A. 查看以下脚本： </font>**
****
```sh
vim examples/ckpt_partitions_102B.sh 
```

**<font color=#FFFFF0 >B. 修改环境变量： </font>**  


- **<font color=#66CD00 >变量 1 </font>**：`LOAD_CHECKPOINT_PATH`，原始32路流水并行的模型文件路径，需要路径下面包含‘latest_checkpointed_iteration.txt’这个文件（文件在开源的模型文件中）


```sh
  LOAD_CHECKPOINT_PATH=/mnt/102B  # 变量1设置示例
 ```


-  **<font color=#66CD00 >变量 2 </font>**：`SAVE_SPLITED_CHECKPOINT_PATH`，表示转换产生的中间文件路径，主要作用于（32路流水-1路张量）->（32路流水-8路张量）产生的结果文件，等脚本完成转换之后可以删除该文件


```sh
  SAVE_SPLITED_CHECKPOINT_PATH=./ckpt-102B-mid # 变量2设置示例
```


- **<font color=#66CD00 >变量 3 </font>**：`SAVE_CHECKPOINT_PATH`，表示最终产生的8路张量并行的ckpt，主要作用于（32路流水-8路张量）->（1路流水-8路张量）产生的结果文件

```sh
  SAVE_CHECKPOINT_PATH=./ckpt-102B-8tp # 变量3设置示例
```


如果在megatron主目录下面运行脚本，可以不修改TOKENIZER_MODEL_PATH=./tokenizer（因为github主目录下面包含tokenizer），否则需要指定tokenizer路径


**<font color=#FFFFF0 >C. 执行以下脚本： </font>**

```sh
bash examples/ckpt_partitions_102B.sh
```

在上述步骤完成后，会在 `SAVE_CHECKPOINT_PATH` 指定的目录下面生成一个8路张量并行的ckpt，可以用于推理服务。



### <strong>🔘 步骤 2</strong> 

**<font color=#FFFFF0 >A. 修改环境变量： </font>**

启动推理服务，需要在脚本 examples/run_inference_server_102B.sh 中修改环境变量CHECKPOINT_PATH：

```sh
vim examples/run_inference_server_102B.sh
```
修改环境变量 `CHECKPOINT_PATH` 为 **🔘 步骤 1** 中 `SAVE_CHECKPOINT_PATH` 指定的路径，例如：

```sh
SAVE_CHECKPOINT_PATH=./ckpt-102B-8tp
```
需要指定 examples/run_inference_server_102B.sh 中 `CHECKPOINT_PATH`，例如：

```sh
CHECKPOINT_PATH=./ckpt-102B-8tp
```

**<font color=#FFFFF0 >B. 启动推理服务： </font>**



> **<font color=yellow >[注意] </font>**
> <br />程序默认的端口号为 `8000` ，如果 `8000` 被占用，需要修改 examples/run_inference_server_102B.sh 中的环境变量 `PORT` 为实际使用的端口号

```sh
bash examples/run_inference_server_102B.sh
```


等待程序完成cpkt的加载并出现如下信息后，可以执行下一步调用推理服务：
> successfully loaded checkpoint from ./ckpt-102B-8tp at iteration 1
> <br />Serving Flask app 'megatron.text_generation_server'
> <br />Debug mode: off
> <br />WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
> <br />Running on all addresses (0.0.0.0)
> <br />Running on http://127.0.0.1:8000 
> <br />Running on http://127.0.0.1:8000



**<font color=#FFFFF0 >C. 调用推理服务： </font>**

在相同docker中调用推理服务：

> **<font color=yellow >[注意] </font>**
> <br />程序默认的端口号为 `8000` ，如果 `8000` 被占用，请在 tools/start_inference_server_api.py 中替换`request_url="http://127.0.0.1:8000/yuan"` 的实际端口；


```sh
python tools/start_inference_server_api.py
```
运行成功后，会返回推理结果；




### <font color=#FFC125 >源2.0-51B 模型 </font> 

-----
**本脚本为51B模型的快速使用指引，主要包括ckpt转换以及推理服务使用**

### <strong>🔘 步骤 1</strong> 

首先需要转换ckpt，我们提供的51B的模型文件是16路流水并行-1路张量并行（16pp，1tp）的模型文件，为了提高推理效率，需要将16路流水并行的模型文件转换为4路张量并行的模型文件（适用于80GB GPU），转换流程是：

> （16路流水-1路张量）->（16路流水-4路张量）->（1路流水-4路张量）


我们提供了自动转换脚本，可以依次执行完转换流程，使用方式如下：


**<font color=#FFFFF0 >A. 查看以下脚本： </font>**
****
```sh
vim examples/ckpt_partitions_51B.sh
```



**<font color=#FFFFF0 >B. 修改环境变量： </font>**  


- **<font color=#66CD00 >变量 1 </font>**：`LOAD_CHECKPOINT_PATH`，原始16路流水并行的模型文件路径，需要路径下面包含latest_checkpointed_iteration.txt这个文件（这个文件会在开源的模型文件中）



```sh
  LOAD_CHECKPOINT_PATH=/mnt/51B # 变量1设置示例
 ```


-  **<font color=#66CD00 >变量 2 </font>**：`SAVE_SPLITED_CHECKPOINT_PATH`，表示转换产生的中间文件路径，主要作用于（16路流水-1路张量）->（16路流水-4路张量）产生的结果文件，等脚本完成转换之后可以删除该文件


```sh
  SAVE_SPLITED_CHECKPOINT_PATH=./ckpt-51B-mid # 变量2设置示例
```


- **<font color=#66CD00 >变量 3 </font>**：`SAVE_CHECKPOINT_PATH`，表示最终产生的4路张量并行的ckpt，主要作用于（16路流水-4路张量）->（1路流水-4路张量）产生的结果文件

```sh
  SAVE_CHECKPOINT_PATH=./ckpt-51B-4tp # 变量3设置示例
```

如果在megatron主目录下面运行脚本，可以不修改TOKENIZER_MODEL_PATH=./tokenizer（因为github主目录下面包含tokenizer），否则需要指定tokenizer路径



**<font color=#FFFFF0 >C. 执行以下脚本： </font>**

```sh
bash examples/ckpt_partitions_51B.sh
```

在上述步骤完成后，会在 `SAVE_CHECKPOINT_PATH` 指定的目录下面生成一个4路张量并行的ckpt，可以用于推理服务。



### <strong>🔘 步骤 2</strong> 

**<font color=#FFFFF0 >A. 修改环境变量： </font>**

启动推理服务，需要在脚本 examples/run_inference_server_51B.sh 中修改环境变量CHECKPOINT_PATH：

```sh
vim examples/run_inference_server_51B.sh
```
修改环境变量 `CHECKPOINT_PATH` 为 **🔘 步骤 1** 中 `SAVE_CHECKPOINT_PATH` 指定的路径，例如：

```sh
SAVE_CHECKPOINT_PATH=./ckpt-51B-4tp
```
需要指定 examples/run_inference_server_51B.sh 中 `CHECKPOINT_PATH`，例如：

```sh
CHECKPOINT_PATH=./ckpt-51B-4tp
```
**<font color=#FFFFF0 >B. 启动推理服务： </font>**



> **<font color=yellow >[注意] </font>**
> <br />程序默认的端口号为 `8000` ，如果 `8000` 被占用，需要修改 examples/run_inference_server_51B.sh 中的环境变量 `PORT` 为实际使用的端口号

```sh
bash examples/run_inference_server_51B.sh
```


等待程序完成cpkt的加载并出现如下信息后，可以执行下一步调用推理服务：
> successfully loaded checkpoint from ./ckpt-51B-4tp at iteration 1
> <br />Serving Flask app 'megatron.text_generation_server'
> <br />Debug mode: off
> <br />WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
> <br />Running on all addresses (0.0.0.0)
> <br />Running on http://127.0.0.1:8000 
> <br />Running on http://127.0.0.1:8000



**<font color=#FFFFF0 >C. 调用推理服务： </font>**

在相同docker中调用推理服务：

> **<font color=yellow >[注意] </font>**
> <br />程序默认的端口号为 `8000` ，如果 `8000` 被占用，请在 tools/start_inference_server_api.py 中替换`request_url="http://127.0.0.1:8000/yuan"` 的实际端口；


```sh
python tools/start_inference_server_api.py
```
运行成功后，会返回推理结果；
