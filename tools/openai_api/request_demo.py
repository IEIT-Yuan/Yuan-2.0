"""
This script is an example of using the OpenAI API to create various interactions with a Yuan2.0 model.
"""

from openai import OpenAI

import os
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

base_url = "http://10.51.24.212:8051/v1/"
client = OpenAI(
    api_key="EMPTY",
    base_url=base_url
)

def function_chat():
    messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model="yuan2",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    if response:
        content = response.choices[0].message.content
        print(content)
    else:
        print("Error:", response.status_code)


def simple_chat(use_stream=True):
    messages = [
        {
            "role": "system",
            "content": "你是浪潮信息研发的大语言模型。",
        },
        {
            "role": "user",
            "content": "你好，请给我写一首诗，主题是春节。"
        }
    ]
    response = client.chat.completions.create(
        model="Yuan2-2B-Janus".lower(),
        messages=messages,
        stream=use_stream,
        max_tokens=1024,
        temperature=1.0,
        seed=1234,
        # repeat_penalty=1.2,
        top_p=0.9)
    if response:
        if use_stream:
            for chunk in response:
                print(chunk.choices[0].delta.content)
        else:
            content = response.choices[0].message.content
            print(content)
    else:
        print("Error:", response.status_code)


def embedding():
    response = client.embeddings.create(
        model="bge-large-zh-1.5",
        input=["你好，写一个春节晚会致辞，100字左右。"],
    )
    embeddings = response.data[0].embedding
    print(f"embeddings length: {len(embeddings)}")


if __name__ == "__main__":
    simple_chat(use_stream=False)
    simple_chat(use_stream=True)
    embedding()
    function_chat()
