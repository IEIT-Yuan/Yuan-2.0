import json 
results = []
with open("/mnt/hf_output.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        index = data.find("代码如下")
        data = data[index:]
        results.append(data[16:])
print(results)
with open("/mnt/vllm_yuan_top_p0.jsonl", "r", encoding="utf-8") as input_file:
    with open("/mnt/yuan_hf.jsonl", "w", encoding="utf-8") as output_file:
        for line, result in zip(input_file, results):
            data = json.loads(line)
            data["completion"] = result
            json.dump(data, output_file, ensure_ascii=False)
            output_file.write('\n')
        
