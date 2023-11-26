### Introduction

To fine-tuning or inference with huggingface model formats, we provide scripts for convert checkpoint , which can be found in the  '**examples**' directory. 

**examples/convert_hf.sh.** Running the program can convert yuan checkpoint format to huggingface checkpoint format.The parallel strategy of yuan2.0 and huggingface is inconsistent. When model parallel and pipeline-parallel models exist in the model, the model needs to be converted to a serial model and then the model conversion is carried out. For details, see '**docs/checkpoint_process.md**'

In addition, we have provided relevant documents of huggingface,we can find the file in '**tools/huggingface**' directory. 
