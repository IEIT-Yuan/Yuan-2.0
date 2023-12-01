# eval_truthfulqa

## Dataset
**datasets/TruthfulQA/TruthfulQA_EN.json.** The original English version of the [TruthfulQA](https://github.com/sylinrl/TruthfulQA/tree/main/data) test set containing 817 questions.

**datasets/TruthfulQA/TruthfulQA_CN.json.** The Chinese version of the TruthfulQA test set for MC1, which is obtained by translation with the aid of gpt-4 model.

In TruthfulQA_CN.json, 'query' denotes the question, 'ans1' denotes the first option, 'ans2' denotes the second option, and so on.

## Evaluation

### Introduction
**examples/eval_truthfulqa_102B.sh.** The evaluation results for TruthfulQA_CN could be obtained by running this program. 

The variables in the code should be set as follows: 

| Variable name               | Description          |
| ------------------- | --------------------------------------------- |
| `CHECKPOINT_PATH`    | the path that saves the checkpoint to be evaluated.       |
| `TOKENIZER_MODEL_PATH`    | the path that saves the tokenizer.                  |
| `TruthfulQA_DATA`    | the path that saves the evaluation set.                  |
| `OUTPUT_PATH`    | the path that saves the evaluation results.                  |

### Usage

Run the following command to evaluate the model's performance on MC1 test set:
```
bash -x examples/eval_truthfulqa_102B.sh
```

### Result
The evaluation result will be saved in the path of $OUTPUT_PATH. For the key of 'gen_ans', the content before <sep> is the question and options, and the content after <sep> is the model choice.

## Accuracy
### Introduction
**tasks/TruthfulQA/score_truthfulqa.py.** The MC1 accuracy of evaluation results for TruthfulQA-CN could be obtained by running this program.

The path variable in the code "sys.argv[1]" denotes the evaluation results. 

### Usage
Run the following command to evaluate the model's performance on the test set:
```
python score_truthfulqa.py <specify Path>
```
### Result
The results show as "MC1 acc: 0.xx".

