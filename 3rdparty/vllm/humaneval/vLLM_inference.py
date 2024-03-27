import json 

prompts = []
with open("/mnt/vllm_20240319/humaneval/human-eval-gpt4-translation-fixed5.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        prompt = data.get("prompt")
        index = prompt.find("代码如下：\n```")
        prompt = prompt[index:]
        prompts.append(prompt[16:])
print(prompts)
#print(len(prompYuan2B_vllm_gents))
import pdb
#pdb.set_trace()
with open("/mnt/vllm_yuan_HE.jsonl", "r", encoding="utf-8") as input_file:
    with open("/mnt/vllm_yuan_HE_Mid.jsonl", "w", encoding="utf-8") as output_file:
        for line, prompt_ in zip(input_file, prompts):
            data = json.loads(line)
            completion = data.get("completion")
            new_completion = prompt_ + completion
            data["completion"] = new_completion
            json.dump(data, output_file, ensure_ascii=False)
            output_file.write('\n')
        
