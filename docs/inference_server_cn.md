# Yuan2.0 推理 API 部署

-  可以通过如下步骤进行部署：

   第一步，通过 `vim examples/run_inference_server_2.1B.sh` 修改 `TOKENIZER_MODEL_PATH、CHECKPOINT_PATH`两个变量为实际存放文件路径；`CUDA_VISIBLE_DEVICES`表示使用GPU编号，`PORT`表示服务使用端口号，用户可根据实际情况自行修改。

   第二步，运行仓库中的脚本进行部署：

```bash
bash examples/run_inference_server_2.1B.sh
```

- 使用Python进行测试

我们还编写了一个示例代码来测试API调用的性能，运行前注意将代码中 `ip`和`port` 根据api部署情况进行修改。

```bash
python tools/start_inference_server_api.py
```

- 使用Curl进行测试

```
curl http://127.0.0.1:8000/yuan -X PUT   \
--header 'Content-Type: application/json' \
--data '{"ques_list":[{"id":"000","ques":"请帮忙作一首诗，主题是冬至"}], "tokens_to_generate":500, "top_k":5}'
```

