from transformers import AutoModel, LlamaTokenizer
import torch
import time

device = 'cuda'
model_dir = "/temp_data/LLM_test/Tensorrt-llm-yuan/yuan2B_Janus"

model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device)

tokenizer = LlamaTokenizer.from_pretrained(model_dir, trust_remote_code=True)
tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

input_text = ["Write a short story about spring.<sep>"]

input_id = tokenizer.encode(input_text, return_tensors='pt').to(device)
start_time = time.time()
with torch.no_grad():
    output = model.generate(input_id, do_sample=True, temperature=0.8, top_p=0.95, max_length=512).to(device)
end_time = time.time()
print(tokenizer.decode(output[0]))
print("inference_time:", (end_time - start_time))

