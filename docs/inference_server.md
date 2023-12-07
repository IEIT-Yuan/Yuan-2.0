# Yuan2.0 Inference-Server

## Introduction

This document provides instructions for inference-server of Yuan2.0.

## Usage

- First step，modify the script file

   	`TOKENIZER_MODEL_PATH` indicates the storage path for TOKENIZER related files；
	`CHECKPOINT_PATH` indicates the storage path for model related files；
    `GPUS_PER_NODE` indicates the number of GPU cards used for this node, this number should be consistent with the number of parallel paths for model tensors；
    `CUDA_VISIBLE_DEVICES` indicates the GPU number used, the number of used numbers should be consistent with `GPUS_PER_NODE` ；
    `PORT` indicates the port number used by the service, one service occupies one port number, the user can modify it according to the actual situation.
  
- Second step,  run the script in the warehouse for deployment
```bash
#2.1B deployment command
bash examples/run_inference_server_2.1B.sh

#51B deployment command
bash examples/run_inference_server_51B.sh

#102B deployment command
bash examples/run_inference_server_102B.sh
```


- Testing with Python

Also, we have written a sample code to test the performance of the API calls. Before running, make sure to modify the 'ip' and 'port' in the code according to the API deployment situation.

```bash
python tools/start_inference_server_api.py
```

- Testing with Curl

```
#return the Unicode encoding
curl http://127.0.0.1:8000/yuan -X PUT   \
--header 'Content-Type: application/json' \
--data '{"ques_list":[{"id":"000","ques":"请帮忙作一首诗，主题是冬至"}], "tokens_to_generate":500, "top_k":5}'

# return the original form
echo -en "$(curl -s  http://127.0.0.1:8000/yuan -X PUT  --header 'Content-Type: application/json' --data '{"ques_list":[{"id":"000","ques":"作一首词 ，主题是冬至"}], "tokens_to_generate":500, "top_k":5}')"
```

