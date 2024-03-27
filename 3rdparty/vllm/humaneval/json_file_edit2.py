import json

with open('/mnt/vllm_20240319/humaneval/human-eval-gpt4-translation-fixed5.jsonl', 'r', encoding='utf-8') as file:
    prompts = [json.loads(line)['prompt'] for line in file]

with open('/mnt/vllm_20240319/HumanEval.jsonl', 'r', encoding='utf-8') as file:
    with open("/mnt/output_file.jsonl", "a") as out_file:
        for prompt, line in zip(prompts, file):
            data = json.loads(line)
            data["prompt"] = prompt
            json.dump(data, out_file)
            out_file.write('\n')
