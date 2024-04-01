from vllm import LLM, SamplingParams
import time
import os
import json

sampling_params = SamplingParams(max_tokens=512, temperature=1, top_p=0, top_k=1, stop="<eod>" )

llm = LLM(model="/temp_data/LLM_test/Tensorrt-llm-yuan/yuan2B_Janus", trust_remote_code=True)

results = []
with open('/mnt/vllm_20240319/humaneval/human-eval-gpt4-translation-fixed5.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            prompts = data.get('prompt')
            outputs = llm.generate(prompts, sampling_params)
            results.append(outputs[0])
            

generated_data = [] 
with open('/mnt/vllm_20240319/humaneval/samples.jsonl', 'r', encoding='utf-8') as file:
    for line, result in zip(file, results):
        data = json.loads(line)
        task_id = data.get('task_id')
        completion = data.get('completion')
        generated_text = result.outputs[0].text
        data['completion'] = generated_text
        generated_data.append({'task_id': task_id, 'completion': generated_text})


with open('/mnt/vllm_yuan_HE.jsonl', 'w', encoding='utf-8') as file:
    for data in generated_data:
        json.dump(data, file, ensure_ascii=False)
        file.write('\n')
