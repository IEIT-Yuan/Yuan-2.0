# eval_agieval_math

## Dataset
**`dataset/AGIEval-Math/mathqa.txt`.** The mathqa set in [AGI-eval](https://github.com/ruixiangcui/AGIEval) contains 351 questions.

**`dataset/AGIEval-Math/mathcloze.txt`.** The mathcloze set in AGI-eval contains 118 questions.

In the text, the content before `[SEP]` is the question, and the content after `[SEP]` is the standard answer to that question.

## Evaluation

### Introduction
**`examples/eval_mathqa_102B.sh`.** The evaluation results for mathqa could be obtained by running this program. 

**`examples/eval_mathcloze_102B.sh`.** The evaluation results for mathcloze could be obtained by running this program. 

The variables in the code should be set as follows: 

| Variable name               | Description          |
| ------------------- | --------------------------------------------- |
| `CHECKPOINT_PATH`    | the path that saves the checkpoint to be evaluated.                  |
| `TOKENIZER_MODEL_PATH`    | the path that saves the tokenizer model.                  |
| `MATH_DATA`    | the path that saves the evaluation set.                  |
| `OUTPUT_PATH`    | the path that saves the evaluation results.                  |

### Usage

Run the following command to evaluate the model's performance on the test set:
```
bash -x examples/eval_mathqa_102B.sh
bash -x examples/eval_mathcloze_102B.sh
```
### Result
The evaluation result will be saved in the path of `$OUTPUT_PATH`. In the text, the content before `[SEP]` is the question, and the content after `[SEP]` is the answer of our model to that question.

## Accuracy
### Introduction
**`tasks/AGIEval-Math/score_mathqa.py`.** The accuracy of evaluation results for mathqa could be obtained by running this program.

The path variables in the code should be set as follows: 

| Variable name               | Description          |
| ------------------- | --------------------------------------------- |
| `origin_file_path`  | Path of evaluation set file.             |
| `eval_file_path`    | Path for saving the evaluation result file.        |
| `txt_eval_res_dir`  | Path for storing distinguished results. Files ending with _true contain correctly results, while those ending in _false contain incorrectly results. |

### Usage
Run the following command to evaluate the model's performance on the test set: 
```
python score_mathqa.py
```
### Result
"Number of correct answers" and "Number of incorrect answers" respectively represent the number of correct answers and the number of incorrect answers, while "accuracy" indicates the accuracy rate. The result shows that the accuracy on mathqa evaluation set is 38.7% and the accuracy on mathcloze evaluation set is 13.5%.
