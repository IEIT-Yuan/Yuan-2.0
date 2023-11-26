#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
端到端推理请求示例：
请求方式：post请求
请求url：http://ip:port/yuan
请求参数：
{
    "ques_list":[{"id":"000","ques":"你能否提供一些改善健康生活方式的具体建议？"}],
    "tokens_to_generate":30,
    "temperature":1.0,
    "top_p":0.0,
    "top_k":5
}

正确返回：
{
    "errCode": "0",
    "errMessage": "success",
    "exceptionMsg": "",
    "flag": true,
    "resData": {
        "output": [
            {
                "ans": " 当然可以。改善健康生活方式有很多方法，以下是一些具体的建议：\n1. 健康饮食：增加蔬菜、水果、全谷物",
                "id": "000"
            }
        ]
    }
}

"""

import json
import os
import sys
dir_path=os.getcwd()
sys.path.append(os.getcwd())
import requests
import time


def rest_post(url, data, timeout, show_error=False):
    '''Call rest post method'''
    # 参考https://www.cnblogs.com/lly-lcf/p/13876823.html
    try:
        response = requests.put(url, json=data, timeout=timeout)
        res = json.loads(response.text)
        return res["resData"]["output"]
    except Exception as exception:
        if show_error:
            print(exception)
        return None


if __name__ == '__main__':

    request_data={
        "ques_list":[{"id":"000","ques":"你能否提供一些改善健康生活方式的具体建议？"}],
        "tokens_to_generate":3000,
        "temperature":1.0,
        "top_p":0.0,
        "top_k":5
    }
    request_url="http://ip:port/yuan"
    result = rest_post(request_url, request_data, 30, show_error=False)
    print(result)
    print("end")

    pass
