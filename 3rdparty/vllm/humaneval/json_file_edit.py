import json

prompts = []
with open('/mnt/vllm_20240319/humaneval/human-eval-gpt4-translation-fixed5.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        prompt = data.get('prompt')
        prompts.append(prompt)

with open('/mnt/vllm_20240319/HumanEval.jsonl', 'r', encoding='utf-8') as file:
    all_data = file.readlines()

with open("/mnt/output_file.jsonl", "a", encoding='utf-8') as out_file:
    for prompt, data in zip(prompts, all_data):
        data_dict = json.loads(data)
        data_dict["prompt"] = prompt
        json.dump(data_dict, out_file)
        out_file.write('\n')
