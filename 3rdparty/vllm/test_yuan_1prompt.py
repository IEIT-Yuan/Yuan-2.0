from vllm import LLM, SamplingParams
import time
import os

prompts = ["如果你是一个算法工程师，让你写一个大模型相关的规划，你应该怎么写？"]
sampling_params = SamplingParams(max_tokens=300, temperature=1, top_p=0, top_k=1, min_p=0.0, length_penalty=1.0, repetition_penalty=1.0, stop="<eod>", )

llm = LLM(model="/temp_data/LLM_test/Tensorrt-llm-yuan/yuan2B_Janus", trust_remote_code=True)

start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
end_time = time.time()
total_tokens = 0
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    num_tokens = len(generated_text)
    total_tokens += num_tokens
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("inference_time:", (end_time - start_time))
#print("total_tokens:", total_tokens)

