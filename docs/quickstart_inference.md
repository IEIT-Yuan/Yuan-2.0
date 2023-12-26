# Quick Start Inference

This script describes the inference guide of the 102B model and the 51B model, including the ckpt conversion and the use of the inference service.

## 102B：

### step1：

Firstly, you need to convert the ckpt.

The parallelism method of the 102B-models is 32-pipeline-parallelism and 1-tensor--parallelism(32pp, 1tp). In order to improve the parallelism efficiency of the inference, you need to convert parallelism method of the 102B-models from  (32pp, 1tp) to (1pp, 8tp). (Apply to 80GB-GPU)

The conversion process is as follows:

(32pp, 1tp) -> (32pp, 8tp) -> (1pp, 8tp)

We provide an automatic conversion script that can be used as follows:

```
1. vim examples/ckpt_partitions_102B.sh

2. Set three environment variables: LOAD_CHECKPOINT_PATH, SAVE_SPLITED_CHECKPOINT_PATH, SAVE_CHECKPOINT_PATH:

LOAD_CHECKPOINT_PATH: The path to the base 102B-model(32pp, 1tp), this path needs to contain the 'latest_checkpointed_iteration.txt' file. An example is shown below:

LOAD_CHECKPOINT_PATH=/mnt/102B

SAVE_SPLITED_CHECKPOINT_PATH: The path to the temporary 102B-model(32pp, 8tp), which can be removed when all conversions are done. An example is shown below:

SAVE_SPLITED_CHECKPOINT_PATH=./ckpt-102B-mid

SAVE_CHECKPOINT_PATH: The path to the resulting 102B-model(1pp, 8tp). An example is shown below:

SAVE_CHECKPOINT_PATH=./ckpt-102B-8tp

If you run the script in the Yuan home directory, you can use the path: TOKENIZER_MODEL_PATH=./tokenizer (because the Yuan home directory contains the tokenizer), otherwise you need to specify the tokenizer path.

3. bash examples/ckpt_partitions_102B.sh
```

After the above steps are completed, an 8-way tensor parallel ckpt will be generated in the directory specified by `SAVE_CHECKPOINT_PATH`, which can be used for inference services.

### step2：

```
1. Set environment variable 'CHECKPOINT_PATH' in script 'examples/run_inference_server_102B.sh'.

vim examples/run_inference_server_102B.sh

Set environment variable 'CHECKPOINT_PATH' to 'SAVE_CHECKPOINT_PATH' specified in step-1. For example, if in step-1 SAVE_CHECKPOINT_PATH=./ckpt-102B-8tp, you should set CHECKPOINT_PATH=./ckpt-102B-8tp in script examples/run_inference_server_102B.sh


2. Start the inference service（Requires 8 x 80GB-GPU）：

#The default port number of the script is 8000, if 8000 is occupied, you need to change the environment variable 'PORT' in examples/run_inference_server_102B.sh to the used port number.

bash examples/run_inference_server_102B.sh

After the program finishes loading the cpkt and the following information appears, you can perform the next step to call the inference service:

  successfully loaded checkpoint from ./ckpt-102B-8tp at iteration 1

 * Serving Flask app 'megatron.text_generation_server'
 * Debug mode: off
   WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8000
 * Running on http://127.0.0.1:8000

3. Use the inference service in the same docker：

#The default port number of the script is 8000, if 8000 is occupied, you need to change 'request_url="http://127.0.0.1:8000/yuan"' in script 'tools/start_inference_server_api.py' to the used port number.

python tools/start_inference_server_api.py

If the inference service runs successfully, the inference result will be returned
```


## 51B：

### step1：

Firstly, you need to convert the ckpt.

The parallelism method of the 51B-models is 16-pipeline-parallelism and 1-tensor--parallelism(16pp, 1tp). In order to improve the parallelism efficiency of the inference, you need to convert parallelism method of the 51B-models from  (16pp, 1tp) to (1pp, 4tp). (Apply to 80GB-GPU)

The conversion process is as follows:

(16pp, 1tp) -> (16pp, 4tp) -> (1pp, 4tp)

We provide an automatic conversion script that can be used as follows:

```
1. vim examples/ckpt_partitions_51B.sh

2. Set three environment variables: LOAD_CHECKPOINT_PATH, SAVE_SPLITED_CHECKPOINT_PATH, SAVE_CHECKPOINT_PATH:

LOAD_CHECKPOINT_PATH: The path to the base 51B-model(16pp, 1tp), this path needs to contain the 'latest_checkpointed_iteration.txt' file. An example is shown below:

LOAD_CHECKPOINT_PATH=/mnt/51B

SAVE_SPLITED_CHECKPOINT_PATH: The path to the temporary 51B-model(16pp, 4tp), which can be removed when all conversions are done. An example is shown below:

SAVE_SPLITED_CHECKPOINT_PATH=./ckpt-51B-mid

SAVE_CHECKPOINT_PATH: The path to the resulting 51B-model(1pp, 4tp). An example is shown below:

SAVE_CHECKPOINT_PATH=./ckpt-51B-4tp

If you run the script in the Yuan home directory, you can use the path: TOKENIZER_MODEL_PATH=./tokenizer (because the Yuan home directory contains the tokenizer), otherwise you need to specify the tokenizer path.

3. bash examples/ckpt_partitions_51B.sh
```

After the above steps are completed, an 4-way tensor parallel ckpt will be generated in the directory specified by `SAVE_CHECKPOINT_PATH`, which can be used for inference services.

### step2：

```
1. Set environment variable 'CHECKPOINT_PATH' in script 'examples/run_inference_server_51B.sh'.

vim examples/run_inference_server_51B.sh

Set environment variable 'CHECKPOINT_PATH' to 'SAVE_CHECKPOINT_PATH' specified in step-1. For example, if in step-1 SAVE_CHECKPOINT_PATH=./ckpt-51B-4tp, you should set CHECKPOINT_PATH=./ckpt-51B-4tp in script examples/run_inference_server_51B.sh

2. Start the inference service（Requires 4 x 80GB-GPU）：

#The default port number of the script is 8000, if 8000 is occupied, you need to change the environment variable 'PORT' in examples/run_inference_server_51B.sh to the used port number.

bash examples/run_inference_server_51B.sh

After the program finishes loading the cpkt and the following information appears, you can perform the next step to call the inference service:

  successfully loaded checkpoint from ./ckpt-51B-4tp at iteration 1

 * Serving Flask app 'megatron.text_generation_server'
 * Debug mode: off
   WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8000
 * Running on http://127.0.0.1:8000

3. Use the inference service in the same docker：

#The default port number of the script is 8000, if 8000 is occupied, you need to change 'request_url="http://127.0.0.1:8000/yuan"' in script 'tools/start_inference_server_api.py' to the used port number.

python tools/start_inference_server_api.py

If the inference service runs successfully, the inference result will be returned
```
