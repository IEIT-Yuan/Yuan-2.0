# eval_agieval_math

## 数据集
**`dataset/AGIEval-Math/mathqa.txt`.** [AGI-eval](https://github.com/ruixiangcui/AGIEval)评测集中的GAOKAO-mathqa数据集，共包含351个数学问题。

**`dataset/AGIEval-Math/mathcloze.txt`.** AGI-eval评测集中的GAOKAO-mathcloze数据集，共包含118个问题。

其中，“[SEP]”之前的内容为原始问题，“[SEP]”之后的内容是该问题的标准答案。

## 评测

### 说明
**`examples/eval_mathqa_102B.sh`.** 运行该程序即可获得模型在mathqa数据集上的推理结果（以Yuan2.0-102B模型为例）。

**`examples/eval_mathcloze_102B.sh`.** 运行该程序即可获得模型在mathcloze数据集上的推理结果（以Yuan2.0-102B模型为例）。

代码中的变量设置如下：

| 变量名              | 解释          |
| ------------------- | --------------------------------------------- |
| `CHECKPOINT_PATH`      | 待评测checkpoint的路径 |
| `TOKENIZER_MODEL_PATH` | tokenizer的路径          |
| `MATH_DATA`    | 待测试数据集的路径         |
| `OUTPUT_PATH`    | 推理结果的保存路径         |

### 运行

运行以下命令获得推理结果（以Yuan2.0-102B模型为例）：
```
bash -x examples/eval_mathqa_102B.sh
bash -x examples/eval_mathcloze_102B.sh
```
### 结果
评测结果将保存在 `$OUTPUT_PATH`中。其中，“[SEP]”之前的内容为原始问题，“[SEP]”之后的内容是模型对该问题的解析。

## 准确率
### 说明
**`tasks/AGIEval-Math/score_mathqa.py`.** 运行该程序即可获得mathqa评测结果的准确率。

代码中的变量设置如下：

| 变量名称           | 说明        |
| ------------------- | --------------------------------------------- |
| `origin_file_path`  | 测试集的保存路径        |
| `eval_file_path`    | 评测结果文件的保存路径 |
| `txt_eval_res_dir`  | 准确率评判结果的保存路径，以"true"结尾的文件中为正确结果，以"false"结尾的文件中包含错误结果。 |

### 运行
执行以下命令以评估模型在测试集上的准确率：
```
python score_mathqa.py
```
### 结果
“Number of correct answers”和“Number of incorrect answers”分别表示回答正确答案数和回答错误答案数，“accuracy”表示准确率。Yuan2.0-102B模型结果表明在mathqa数据集上的准确率为38.7%，在mathcloze数据集上的准确率为38.7%。

