#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""Sample Generate Yuan"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from transformers import AutoModelForCausalLM, LlamaTokenizer
from transformers import BitsAndBytesConfig
import torch
import argparse
import time
import datetime
import math
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import threading
lock = threading.Lock()

# 注意：若在Windows中运行，提示flash_atten未安装时，需要按以下方式修改代码
# 修改 config.json中"use_flash_attention"为 false；
# 注释掉 yuan_hf_model.py中第35、36行；
# 修改yuan_hf_model.py中第271行为 inference_hidden_states_memory = torch.empty(bsz, 2, hidden_states.shape[2], dtype=hidden_states.dtype)


def text_generate_args():
    """
    # max_length：生成文本的最大长度；
    # min_length：生成文本的最小长度；
    # do_sample=False 来使用贪心采样，设置 do_sample=True 和 temperature=1.0 来使用随机采样；
    # 设置 do_sample=True、top_k=K 和 temperature=1.0 来使用 Top-K 采样；
    # num_beams：Beam Search 算法中的 beam 宽度，用于控制生成结果的多样性，设置 num_beams=K 来使用 Beam Search 算法；
    # temperature：用于控制生成结果的多样性，值越高生成的文本越多样化，设置 temperature=T 来调整温度。
    """

    parser = argparse.ArgumentParser(description='Yuan text generation')
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=1.0, help='Sampling temperature.')
    parser.add_argument("--top_k", type=int, default=0, help='Top k sampling.')
    parser.add_argument("--max_length", type=int, default=1024, help='Size of the output generated text.')
    parser.add_argument('--min_length', type=int, default=0)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--load', type=str, default="IEITYuan/Yuan2-2B-hf", help='Directory containing a model checkpoint.')
    parser.add_argument('--gpu', type=str, default="0", help='the num of gpu you used')
    args = parser.parse_args()

    return args


def success_ques(outputs, message="success"):
    basic = {"flag": True,
                  "errCode": "0",
                  "errMessage": message,
                  "exceptionMsg": "",
                  "resData": outputs}
    print(outputs)
    return basic

def fail_exception(err_message, err_code):
    basic = {"flag": False,  # True、False
                  "errCode": err_code,
                  "successMessage": "",
                  "exceptionMsg": err_message,  # if error，return info
                  "resData": {}}
    print(err_code, err_message)
    return basic


class MegatronGenerate(Resource):
    def __init__(self, model ,tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.AS_ERROR_INPUT_CODE = 1
        self.AS_ERROR_GENERATE_CODE = 2

    def put(self):
        args = self.args
        print('=========================================')
        print("request IP: " + str(request.remote_addr))
        print("current time: ", datetime.datetime.now())
        print(request.get_json())

        if not "ques_list" in request.get_json():
            e = "ques_list argument required"
            return jsonify(fail_exception(str(e), self.AS_ERROR_INPUT_CODE))

        ques_list = request.get_json()["ques_list"]
        if not isinstance(ques_list, list):
            e = "ques_list is not a list"
            return jsonify(fail_exception(str(e), self.AS_ERROR_INPUT_CODE))

        if len(ques_list) == 0:
            e = "ques_list is empty"
            return jsonify(fail_exception(str(e), self.AS_ERROR_INPUT_CODE))

        if len(ques_list) > 1:
            e = "Maximum number of ques_list is 1"
            return jsonify(fail_exception(str(e), self.AS_ERROR_INPUT_CODE))

        tokens_to_generate = args.max_length  # Choosing hopefully sane default. Full sequence is slow
        # tokens_to_generate = (args.max_len - len(tokens))
        if "tokens_to_generate" in request.get_json():
            tokens_to_generate = request.get_json()["tokens_to_generate"]
            if not isinstance(tokens_to_generate, int) or tokens_to_generate < 1:
                e = "tokens_to_generate must be an integer and greater than 0"
                return jsonify(fail_exception(str(e), self.AS_ERROR_INPUT_CODE))

        temperature = args.temperature
        if "temperature" in request.get_json():
            temperature = request.get_json()["temperature"]
            if not (type(temperature) == int or type(temperature) == float):
                e = "the type of temperature must be int or float"
                return jsonify(fail_exception(str(e), self.AS_ERROR_INPUT_CODE))

            if not (0.0 < temperature <= 100.0):
                e = "temperature must be a positive number less than or equal to 100.0"
                return jsonify(fail_exception(str(e), self.AS_ERROR_INPUT_CODE))

        top_k = args.top_k
        if "top_k" in request.get_json():
            top_k = request.get_json()["top_k"]
            if not (type(top_k) == int):
                e = "the type of top_k must be int"
                return jsonify(fail_exception(str(e), self.AS_ERROR_INPUT_CODE))
            if not (0 <= top_k <= 1000):
                e = "top_k must be greater than or equal to 0"
                return jsonify(fail_exception(str(e), self.AS_ERROR_INPUT_CODE))

        num_beams = args.num_beams
        if "num_beams" in request.get_json():
            num_beams = request.get_json()["num_beams"]
            if not isinstance(num_beams, int):
                return "random_seed must be integer"
            if num_beams < 0:
                return "random_seed must be a positive integer"

        do_sample = args.do_sample
        if "do_sample" in request.get_json():
            do_sample = request.get_json()["do_sample"]
            if not isinstance(do_sample, bool):
                return "do_sample must be a boolean value"

        try:
            with lock:  # Need to get lock to keep multiple threads from hitting code
                print('input:' + str(ques_list[0]["ques"]))
                inputs = tokenizer(ques_list[0]["ques"], return_tensors="pt")["input_ids"].to(self.args.device)

                response_raw = model.generate(inputs, do_sample=do_sample, top_k=top_k, temperature=temperature, num_beams=num_beams, max_length=tokens_to_generate)
                response = tokenizer.decode(response_raw[0])
                print('output:' + response)

                res = response.strip()
                res = res.split('<sep>')[-1].replace('</s>', '')
                outputs = [{"id": '0', "ans": res}]
                resData = {'output': outputs}
                print("infer finished time: ", datetime.datetime.now())
                return jsonify(success_ques(resData))
        except Exception as e:
            return jsonify(fail_exception(str(e), self.AS_ERROR_GENERATE_CODE))


class YuanServer(object):
    def __init__(self, model, tokenizer, args):
        self.app = Flask(__name__, static_url_path='')
        api = Api(self.app)
        api.add_resource(MegatronGenerate, '/yuan', resource_class_args=[model, tokenizer, args])

    def run(self, url, port):
        self.app.run(url, threaded=True, debug=False, port=int(os.environ.get("PORT", port)))


if __name__ == "__main__":
    t = time.time()
    seed = int(1000 * (math.ceil(t) - t))
    args = text_generate_args()
    args.device = torch.device("cuda:{0}".format(args.gpu) if torch.cuda.is_available() else 'cpu')  # 设备

    print("Creat tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(args.load)
    tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>', '<commit_before>',
         '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>', '<jupyter_output>', '<empty_output>'], special_tokens=True)

    print("Creat model...")

    if torch.cuda.is_available():
        # gpu 推理代码
        device_map = 'auto'
    else:
        # cpu 推理代码
        device_map = 'cpu'

    # # gpu 量化推理代码
    # # Our 4-bit configuration to load the LLM with less GPU memory
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,  # 4-bit quantization
    #     bnb_4bit_quant_type='nf4',  # Normalized float 4
    #     bnb_4bit_use_double_quant=True,  # Second quantization after the first
    #     bnb_4bit_compute_dtype=torch.bfloat16  # Computation type
    # )
    # model = AutoModelForCausalLM.from_pretrained(args.load, device_map=device_map, quantization_config=bnb_config, trust_remote_code=True).eval()

    model = AutoModelForCausalLM.from_pretrained(args.load, device_map=device_map, trust_remote_code=True).eval()

    server = YuanServer(model, tokenizer, args)
    server.run("0.0.0.0", port=int(os.environ.get("PORT", 8000)))

