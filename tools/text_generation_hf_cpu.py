
# 注意：CPU版本需要手工关掉 Flash attention，预先下载模型文件到本地后，步骤如下：
# （1）修改 config.json中"use_flash_attention"为 false；
# （2）注释掉 yuan_hf_model.py中第35、36行；修改yuan_hf_model.py中第271行为 inference_hidden_states_memory = torch.empty(bsz, 2, hidden_states.shape[2], dtype=hidden_states.dtype)

import sys, os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from transformers import AutoModelForCausalLM, LlamaTokenizer
from datetime import datetime

print("Creat tokenizer...")
path = "your_model_hf_path"
tokenizer = LlamaTokenizer.from_pretrained(path)
tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

print("Creat model...")
model = AutoModelForCausalLM.from_pretrained(path, device_map="cpu", trust_remote_code=True).eval()  #cpu 代码

Time1 = datetime.now()
inputs = tokenizer("请问目前最先进的机器学习算法有哪些？", return_tensors="pt")["input_ids"].to("cpu")

# max_length：生成文本的最大长度；
# min_length：生成文本的最小长度；
# do_sample=False 来使用贪心采样，设置 do_sample=True 和 temperature=1.0 来使用随机采样；
# 设置 do_sample=True、top_k=K 和 temperature=1.0 来使用 Top-K 采样；
# num_beams：Beam Search 算法中的 beam 宽度，用于控制生成结果的多样性，设置 num_beams=K 来使用 Beam Search 算法；
# temperature：用于控制生成结果的多样性，值越高生成的文本越多样化，设置 temperature=T 来调整温度。
outputs = model.generate(inputs, do_sample=True, top_k=5, max_length=100)
print(tokenizer.decode(outputs[0]))
print(datetime.now() - Time1)


