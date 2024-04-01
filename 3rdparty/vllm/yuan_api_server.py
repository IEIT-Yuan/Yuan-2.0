import requests
import json

outputs = [] 
with open('/mnt/Yuan-2.0/3rdparty/vllm/humaneval/human-eval-gpt4-translation-fixed5.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        prompt = data.get('prompt')
        raw_json_data = {
                "prompt": prompt,
                "logprobs": 1,
                "max_tokens": 256,
                "temperature": 1,
                "use_beam_search": False,
                "top_p": 0,
                "top_k": 1,
                "stop": "<eod>",
                }
        json_data = json.dumps(raw_json_data)
        headers = {
                "Content-Type": "application/json",
                }
        response = requests.post(f'http://localhost:8000/generate',
                             data=json_data,
                             headers=headers)
        output = response.text
        output = json.loads(output)
        output = output['text']
        outputs.append(output[0])
print(outputs)
'''
generated_data = []
with open('/mnt/vllm_20240319/humaneval/samples.jsonl', 'r', encoding='utf-8') as file:
    for line, output in zip(file, outputs):
        data = json.loads(line)
        task_id = data.get('task_id')
        completion = data.get('completion')
        data['completion'] = output
        generated_data.append({'task_id': task_id, 'completion': output})

with open('/mnt/vllm_yuan_api.jsonl', 'w', encoding='utf-8') as file:
    for data in generated_data:
        json.dump(data, file, ensure_ascii=False)
        file.write('\n')

'''
