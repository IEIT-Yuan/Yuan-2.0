from vllm import LLM, SamplingParams
import time
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
prompts = ["写一篇春游作文<sep>"]
#    "Hello, my name is",
#    "The president of the United States is",
#    "The capital of France is",
#    "The future of AI is",

sampling_params = SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)

llm = LLM(model="/temp_data/LLM_test/Tensorrt-llm-yuan/yuan2B_Janus", trust_remote_code=True)

start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
end_time = time.time()

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("inference_time:", (end_time - start_time))
