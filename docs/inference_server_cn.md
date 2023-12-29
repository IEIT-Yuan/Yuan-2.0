# Yuan2.0 推理 API 部署

  - [ckpt模型推理API部署](#ckpt模型推理API部署)
  - [HuggingFace模型推理API部署](#HuggingFace模型推理API部署)
  - [API部署效果测试](#API部署效果测试)
  

## ckpt模型推理API部署
-  可以通过如下步骤进行部署：

   第一步，修改脚本文件 examples/run_inference_server_~~x~~B.sh

    	`TOKENIZER_MODEL_PATH` 表示TOKENIZER相关文件存放路径；
    	`CHECKPOINT_PATH` 表示模型相关文件存放路径；
    	`GPUS_PER_NODE` 表示使用该节点GPU卡数目，该数目应与模型张量并行路数保持一致；
    	`CUDA_VISIBLE_DEVICES` 表示使用的GPU编号，不同编号之间用逗号隔开，编号数目和应与`GPUS_PER_NODE`保持一致；
    	`PORT` 表示服务使用端口号，一个服务占用一个端口号，用户可根据实际情况自行修改；
  
   第二步，运行仓库中的脚本进行部署：

   ```bash
   #2.1B模型服务启动命令
   bash examples/run_inference_server_2.1B.sh
   
   #51B模型服务启动命令
   bash examples/run_inference_server_51B.sh
   
   #102B模型服务启动命令
   bash examples/run_inference_server_102B.sh
   ```

## HuggingFace模型推理API部署
- 可以通过如下步骤进行部署

   第一步，修改脚本文件 examples/run_inference_server_hf.sh

    	`HF_PATH` 表示HuggingFace模型相关文件存放路径；
    	`CUDA_VISIBLE_DEVICES` 表示使用的GPU编号，不同编号之间用逗号隔开；
    	`PORT` 表示服务使用端口号，一个服务占用一个端口号，用户可根据实际情况自行修改；
  
   第二步，运行仓库中的脚本进行部署：

   ```bash
   bash examples/run_inference_server_hf.sh
   ```
   
- 需要特别注意：若在Windows/CPU中运行，需要手动关闭flash_atten，需要按以下方式修改HuggingFace模型文件代码
   ```
   修改 config.json中"use_flash_attention"为 false；
   注释掉 yuan_hf_model.py中第35、36行；
   修改yuan_hf_model.py中第271行为 inference_hidden_states_memory = torch.empty(bsz, 2, hidden_states.shape[2], dtype=hidden_states.dtype)
   ```


## API部署效果测试

- 使用Python进行测试

我们还编写了一个示例代码来测试API调用的性能，运行前注意将代码中 `ip`和`port` 根据api部署情况进行修改。

```bash
python tools/start_inference_server_api.py
```

- 使用Curl进行测试

```
#如下命令返回Unicode编码
curl http://127.0.0.1:8000/yuan -X PUT   \
--header 'Content-Type: application/json' \
--data '{"ques_list":[{"id":"000","ques":"请帮忙作一首诗，主题是冬至"}], "tokens_to_generate":500, "top_k":5}'

#如下命令返回原始形式
echo -en "$(curl -s  http://127.0.0.1:8000/yuan -X PUT  --header 'Content-Type: application/json' --data '{"ques_list":[{"id":"000","ques":"作一首词 ，主题是冬至"}], "tokens_to_generate":500, "top_k":5}')"
```

