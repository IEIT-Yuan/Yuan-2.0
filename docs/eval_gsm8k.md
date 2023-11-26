# eval_gms8k

## Dataset
**`datasets/GSM8K/gsm8k_En.txt`.** The original English version of the [gsm8k](https://github.com/openai/grade-school-math) test set containing 1,319 questions.

**`datasets/GSM8K/gsm8k_Cn.txt`.** The Chinese version of the gsm8k test set obtained by translation with the aid of gpt-4 model.

In the text, the content before `[SEP]` is the question, and the content after `[SEP]` is the standard answer to that question.

## Evaluation

### Introduction
**`examples/eval_gms8k_102B.sh`.** The evaluation results for gsm8k-Cn could be obtained by running this program. 

The variables in the code should be set as follows: 

| Variable name               | Description          |
| ------------------- | --------------------------------------------- |
| `CHECKPOINT_PATH`    | the path that saves the checkpoint to be evaluated.       |
| `TOKENIZER_MODEL_PATH`    | the path that saves the tokenizer.                  |
| `MATH_DATA`    | the path that saves the evaluation set.                  |
| `OUTPUT_PATH`    | the path that saves the evaluation results.                  |

### Usage

Run the following command to evaluate the model's performance on the test set:
```
bash -x examples/eval_gms8k_102B.sh
```

### Result
The evaluation result will be saved in the path of `$OUTPUT_PATH`. In the text, the content before `[SEP]` is the question, and the content after `[SEP]` is the answer of our model to that question.

## Accuracy
### Introduction
**`tasks/GSM8K/score_gsm8k.py`.** The accuracy of evaluation results for gsm8k-Cn could be obtained by running this program.

The path variables in the code should be set as follows: 

| Variable name               | Description          |
| ------------------- | --------------------------------------------- |
| `origin_file_path`  | Path of evaluation set file.                 |
| `eval_file_path`    | Path for saving the evaluation result file.                  |
| `txt_eval_res_dir`  | Path for storing distinguished results. Files ending with _true contain correctly results, while those ending in _false contain incorrectly results. |

### Usage
Run the following command to evaluate the model's performance on the test set:
```
python score_gsm8k.py
```
### Result
"Number of correct answers" and "Number of incorrect answers" respectively represent the number of correct answers and the number of incorrect answers, while "accuracy" indicates the accuracy rate. The result shows that the accuracy on gsm8k_Cn evaluation set is 76.6%.

