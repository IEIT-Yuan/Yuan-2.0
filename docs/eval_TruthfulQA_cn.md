# eval_truthfulqa

## 数据集
TruthfulQA英文数据集路径：**datasets/TruthfulQA/TruthfulQA_EN.json.** 
TruthfulQA原始英文数据集链接：https://github.com/sylinrl/TruthfulQA/tree/main/data
TruthfulQA的测试集一共包括 817 个问题.

TruthfulQA翻译后的中文数据集路径：**datasets/TruthfulQA/TruthfulQA_CN.json.** 
我们使用的是MC1得分的测试集，翻译过程使用的是gpt-3.5。
在TruthfulQA_CN.json文件里面, 'query' 表示问题, 'ans1' 表示第一个答案, 'ans2' 表示第二个答案, 以此类推。

## 评估方法

### 简介
使用如下脚本可以得到Truthful_QA_CN的评估结果：
**examples/eval_truthfulqa_102B.sh.** 

脚本中需要修改的参数如下：
| 参数名称               | 参数描述          |
| ------------------- | --------------------------------------------- |
| `CHECKPOINT_PATH`    | 需要评估的模型路径.       |
| `TOKENIZER_MODEL_PATH`    | tokenizer路径                  |
| `TruthfulQA_DATA`    | 评测数据集路径（即TruthfulQA数据集的路径）.                  |
| `OUTPUT_PATH`    | 评测结果保存的路径.                  |

### 使用方法

使用下述命令可以评估模型在TruthfulQA任务上的MC1得分：
```
bash -x examples/eval_truthfulqa_102B.sh
```

### 结果
评测结果保存在$OUTPUT_PATH路径下，评测结果中的'gen_ans'键值表示的内容包括："<sep>"之前的内容是问题和各个选项，"<sep>"之后的内容是模型选择的答案。

## 精度
### 简介

TruthfulQA-CN任务中MC1精度得分值可以通过运行以下脚本得到：
**tasks/TruthfulQA/score_truthfulqa.py.** 

脚本中的参数"sys.argv[1]"表示的是之前评测结果保存的路径（即上述的$OUTPUT_PATH）

### 使用方法
运行以下命令可以得到模型在测试集上的MC1得分
```
python score_truthfulqa.py <specify Path>
```
### 结果
MC1得分结果将呈现为“MC1 acc: 0.xx”。

