# Yuan2.0 Inference-Server

## Introduction

This document provides instructions for inference-server of Yuan2.0.

## Usage

- An example script to run Yuan2.0 api deployment script is:

  notice： `vim examples/run_inference_server_2.1B.sh` modify `TOKENIZER_MODEL_PATH、CHECKPOINT_PATH`are the actual path to store the model files.

  `CUDA_VISIBLE_DEVICES` indicates the use of GPU numbers, and `PORT` indicates the port number used by the service. Users can modify it according to their actual situation.

```bash
bash examples/run_inference_server_2.1B.sh
```

- Testing with Python

Also, we have written a sample code to test the performance of the API calls. Before running, make sure to modify the 'ip' and 'port' in the code according to the API deployment situation.

```bash
python tools/start_inference_server_api.py
```

- Testing with Cur

```
curl http://127.0.0.1:8000/yuan -X PUT   \
--header 'Content-Type: application/json' \
--data '{"ques_list":[{"id":"000","ques":"请帮忙作一首诗，主题是冬至"}], "tokens_to_generate":500, "top_k":5}'
```

