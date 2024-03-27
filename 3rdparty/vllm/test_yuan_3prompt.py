from vllm import LLM, SamplingParams
import time
import os

prompts = [
        "大学生写简历有哪些思路建议和注意事项",
        "请以人工智能四个字写一首诗",
        "世界上最高的山是什么",
    ]

sampling_params = SamplingParams(max_tokens=256, temperature=0.8, top_p=0.95, stop="<eod>")

llm = LLM(model="/temp_data/LLM_test/Tensorrt-llm-yuan/yuan2B_Janus", trust_remote_code=True)

start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
end_time = time.time()

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("inference_time:", (end_time - start_time))
