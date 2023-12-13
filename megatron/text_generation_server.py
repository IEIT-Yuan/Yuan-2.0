# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import datetime
import torch
import json
import threading
from flask import Flask, request, jsonify, current_app
from flask_restful import Resource, Api
from megatron import get_args
from megatron.text_generation import generate_and_post_process
from megatron.text_generation import beam_search_and_post_process
import os
from megatron import get_tokenizer

GENERATE_NUM = 0
BEAM_NUM = 1
lock = threading.Lock()


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
    def __init__(self, model):
        self.model = model
        self.AS_ERROR_INPUT_CODE=1
        self.AS_ERROR_GENERATE_CODE=2
        # self.tokenizer = get_tokenizer()

    @staticmethod
    def send_do_generate():
        choice = torch.cuda.LongTensor([GENERATE_NUM])
        torch.distributed.broadcast(choice, 0)
     
    @staticmethod
    def send_do_beam_search():
        choice = torch.cuda.LongTensor([BEAM_NUM])
        torch.distributed.broadcast(choice, 0)
    
    def put(self):
        args = get_args()
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
        
        if len(ques_list) > 128:
            e = "Maximum number of ques_list is 128"
            return jsonify(fail_exception(str(e), self.AS_ERROR_INPUT_CODE))
        
        tokens_to_generate = args.out_seq_length  # Choosing hopefully sane default.  Full sequence is slow
        # tokens_to_generate = (args.max_len - len(tokens))
        if "tokens_to_generate" in request.get_json():
            tokens_to_generate = request.get_json()["tokens_to_generate"]
            if not isinstance(tokens_to_generate, int) or tokens_to_generate < 1:
                e = "tokens_to_generate must be an integer and greater than 0"
                return jsonify(fail_exception(str(e), self.AS_ERROR_INPUT_CODE))

        logprobs = False
        if "logprobs" in request.get_json():
            logprobs = request.get_json()["logprobs"]
            if not isinstance(logprobs, bool):
                return "logprobs must be a boolean value"
        
        if tokens_to_generate == 0 and not logprobs:
            return "tokens_to_generate=0 implies logprobs should be True"
        
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
                e = "top_k must be greater than or equal to 1"
                return jsonify(fail_exception(str(e), self.AS_ERROR_INPUT_CODE))
        
        top_p = args.top_p
        if "top_p" in request.get_json():
            top_p = request.get_json()["top_p"]
            if not (type(top_p) == int or type(top_p) == float):
                e = "the type of top_p must be int or float"
                return jsonify(fail_exception(str(e), self.AS_ERROR_INPUT_CODE))
            if not (0.0 <= top_p <= 1.0):
                e = "top_p must be a positive number and must be less than or equal to 1.0"
                return jsonify(fail_exception(str(e), self.AS_ERROR_INPUT_CODE))
        
        top_p_decay = args.top_p_decay
        if "top_p_decay" in request.get_json():
            top_p_decay = request.get_json()["top_p_decay"]
            if not (type(top_p_decay) == float):
                return "top_p_decay must be a positive float less than or equal to 1.0"
            if top_p == 0.0:
                return "top_p_decay cannot be set without top_p"
            if not (0 <= top_p_decay <= 1.0):
                return "top_p_decay must be less than or equal to 1.0"
        
        top_p_bound = args.top_p_bound
        if "top_p_bound" in request.get_json():
            top_p_bound = request.get_json()["top_p_bound"]
            if not (type(top_p_bound) == float):
                return "top_p_bound must be a positive float less than or equal to top_p"
            if top_p == 0.0:
                return "top_p_bound cannot be set without top_p"
            if not (0.0 < top_p_bound <= top_p):
                return "top_p_bound must be greater than 0 and less than top_p"
        
        add_BOS = False
        if "add_BOS" in request.get_json():
            add_BOS = request.get_json()["add_BOS"]
            if not isinstance(add_BOS, bool):
                return "add_BOS must be a boolean value"
        
        # if any([len(prompt) == 0 for prompt in prompts]) and not add_BOS:
        #     return "Empty prompts require add_BOS=true"

        stop_on_double_eol = False
        if "stop_on_double_eol" in request.get_json():
            stop_on_double_eol = request.get_json()["stop_on_double_eol"]
            if not isinstance(stop_on_double_eol, bool):
                return "stop_on_double_eol must be a boolean value"
        
        stop_on_eol = False
        if "stop_on_eol" in request.get_json():
            stop_on_eol = request.get_json()["stop_on_eol"]
            if not isinstance(stop_on_eol, bool):
                return "stop_on_eol must be a boolean value"

        prevent_newline_after_colon = args.prevent_newline_after_colon
        if "prevent_newline_after_colon" in request.get_json():
            prevent_newline_after_colon = request.get_json()["prevent_newline_after_colon"]
            if not isinstance(prevent_newline_after_colon, bool):
                return "prevent_newline_after_colon must be a boolean value"

        random_seed = args.random_seed
        if "random_seed" in request.get_json():
            random_seed = request.get_json()["random_seed"]
            if not isinstance(random_seed, int):
                return "random_seed must be integer"
            if random_seed < 0: 
                return "random_seed must be a positive integer"

        no_log = False
        if "no_log" in request.get_json():
            no_log = request.get_json()["no_log"]
            if not isinstance(no_log, bool):
                return "no_log must be a boolean value"
        
        beam_width = None
        if "beam_width" in request.get_json():
            beam_width = request.get_json()["beam_width"]
            if not isinstance(beam_width, int):
                return "beam_width must be integer"
            if beam_width < 1:
                return "beam_width must be an integer > 1"
            if len(ques_list) > 1:
                return "When doing beam_search, batch size must be 1"

        stop_token=50256
        if "stop_token" in request.get_json():
            stop_token = request.get_json()["stop_token"]
            if not isinstance(stop_token, int):
                return "stop_token must be an integer"
        
        length_penalty = 1 
        if "length_penalty" in request.get_json():
            length_penalty = request.get_json()["length_penalty"]
            if not isinstance(length_penalty, float):
                return "length_penalty must be a float"

        try:
            with lock:  # Need to get lock to keep multiple threads from hitting code

                # input
                prompts = []
                ids = []
                for ques in ques_list:
                    if not "ques" in ques:
                        e = "ques argument must be in ques_list"
                        return jsonify(fail_exception(str(e), self.AS_ERROR_INPUT_CODE))
                    if "id" in ques:
                        ids.append(ques['id'])
                    else:
                        ids.append('0')
                    prompts.append(ques['ques'])

                if any([len(prompt) == 0 for prompt in prompts]):
                    e = "Cannot exist empty prompt in input ques_list"
                    return jsonify(fail_exception(str(e), self.AS_ERROR_INPUT_CODE))


                context_tokens = [get_tokenizer().tokenize(s) for s in prompts]
                max_len = max([len(tokens) for tokens in context_tokens])
                if max_len + tokens_to_generate > args.seq_length:
                    e = "the sum of input and output seqence lengths shoud be less than {}".format(args.seq_length)
                    return jsonify(fail_exception(str(e), self.AS_ERROR_INPUT_CODE))

                resData = {}
                outputs = []
                if beam_width is not None:
                    MegatronGenerate.send_do_beam_search()  # Tell other ranks we're doing beam_search
                    response, response_seg, response_scores = \
                        beam_search_and_post_process(
                        self.model,
                        prompts=prompts,
                        tokens_to_generate=tokens_to_generate,
                        beam_size = beam_width,
                        add_BOS=add_BOS,
                        stop_token=stop_token,
                        num_return_gen=beam_width,  # Returning whole beam
                        length_penalty=length_penalty,
                        prevent_newline_after_colon=prevent_newline_after_colon
                        )
                    
                else:
                    MegatronGenerate.send_do_generate()  # Tell other ranks we're doing generate
                    print('input:' + str(prompts))
                    response, response_seg, response_logprobs, _ = \
                        generate_and_post_process(
                        self.model,
                        prompts=prompts,
                        tokens_to_generate=tokens_to_generate,
                        return_output_log_probs=logprobs,
                        top_k_sampling=top_k,
                        top_p_sampling=top_p,
                        top_p_decay=top_p_decay,
                        top_p_bound=top_p_bound,
                        temperature=temperature,
                        add_BOS=add_BOS,
                        use_eod_token_for_early_termination=True,
                        stop_on_double_eol=stop_on_double_eol,
                        stop_on_eol=stop_on_eol,
                        prevent_newline_after_colon=prevent_newline_after_colon,
                        random_seed=random_seed)
                    print(response)
                    
                for i, rid in enumerate(ids):
                    res = response[i].strip()
                    res = res[0:res.find('<eod>')]
                    res = res.split('<sep>')[-1]
                    outputs.append({"id": str(rid), "ans": res})
                resData['output'] = outputs
                print("infer finished time: ", datetime.datetime.now())
                return jsonify(success_ques(resData))
        except Exception as e:
            return jsonify(fail_exception(str(e), self.AS_ERROR_GENERATE_CODE))
        

class MegatronServer(object):
    def __init__(self, model):
        self.app = Flask(__name__, static_url_path='')
        api = Api(self.app)
        api.add_resource(MegatronGenerate, '/yuan', resource_class_args=[model])
        
    def run(self, url, port):
        self.app.run(url, threaded=True, debug=False, port=int(os.environ.get("PORT", port)))


