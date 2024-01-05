import torch
import sys, os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from transformers import AutoModelForCausalLM, LlamaTokenizer
from transformers import BitsAndBytesConfig
from datetime import datetime


# Our 4-bit configuration to load the LLM with less GPU memory
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_quant_type='nf4',  # Normalized float 4
    bnb_4bit_use_double_quant=True,  # Second quantization after the first
    bnb_4bit_compute_dtype=torch.bfloat16  # Computation type
)

print("Creat tokenizer...")
path = "your_model_hf_path"
tokenizer = LlamaTokenizer.from_pretrained(path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

print("Creat model...")
model = AutoModelForCausalLM.from_pretrained(path,device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True).eval()  #gpu 代码
# model = AutoModelForCausalLM.from_pretrained(path, device_map='auto', quantization_config=bnb_config, trust_remote_code=True).eval()  #gpu 量化代码

Time1 = datetime.now()
inputs = tokenizer("请问目前最先进的机器学习算法有哪些？", return_tensors="pt")["input_ids"].to("cuda:0")
outputs = model.generate(inputs,do_sample=False,max_length=100)
print(tokenizer.decode(outputs[0]))
print(datetime.now() - Time1)
