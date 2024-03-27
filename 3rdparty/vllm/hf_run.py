import torch, transformers
import sys, os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from transformers import AutoTokenizer,LlamaTokenizer
from yuan_hf_model import YuanForCausalLM

print("Creat tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained('/temp_data/LLM_test/yuan_model', trust_remote_code=True, add_eos_token=False, add_bos_token=False)
tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

print("Creat model...")
model = YuanForCausalLM.from_pretrained('/temp_data/LLM_test/Tensorrt-llm-yuan/yuan2B_Janus',torch_dtype=torch.float16,trust_remote_code=True)
#print(model)
inputs = tokenizer("写一篇春游作文<sep> ", return_tensors="pt")["input_ids"].to("cuda:0")

input_ids = tokenizer.encode("Write a short story about spring.<sep>", add_special_tokens=False, truncation=True, max_length=923)
outputs = model.generate(inputs,do_sample=False,max_length=100)
print(tokenizer.decode(outputs[0]))
