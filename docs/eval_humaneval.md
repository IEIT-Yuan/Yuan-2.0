# eval\_humaneval

## Dataset

**`datasets/HUMANEVAL/HumanEval.jsonl.gz`** The original English version of the [HumanEval](https://github.com/openai/human-eval "HumanEval") dataset containing 164 questions.

**`datasets/HUMANEVAL/HumanEval-textprompts.jsonl`** The Chinese version of the HumanEval dataset obtained by translation with the aid of gpt-4 model.

## Evaluation

### Introduction

**`examples/eval_humaneval_102B.sh`.** The evaluation results for Chinese HumanEval could be obtained by running this program.

The variables in the code should be set as follows:

| Variable name          | Description                                         |
| ---------------------- | --------------------------------------------------- |
| `CHECKPOINT_PATH`      | the path that saves the checkpoint to be evaluated. |
| `TOKENIZER_MODEL_PATH` | the path that saves the tokenizer.                  |
| `LOG_PATH`             | the path that saves the evaluation log.             |
| `OUTPUT_PATH`          | the path that saves the evaluation results.         |

### Requirement

Make sure HumanEval program is installed befere running the HumanEval evaluation on Yuan2.0 checkpoint.&#x20;

```bash
$ git clone https://github.com/openai/human-eval
$ pip install -e human-eval
```

After HumanEval program is installed, we shall go to this script,

```bash
/usr/local/lib/python3.10/dist-packages/human_eval-1.0-py3.10.egg/human_eval/execution.py
```

and make the following change on `check_program` variable in `check_correctness` function, to ensure there is no duplicate function signature in generated codes.

```python
check_program = (
    #problem["prompt"] +
    completion + "\n" +
    problem["test"] + "\n" +
    f"check({problem['entry_point']})"
)
```

### Usage

Run the following command to evaluate the model's performance on the dataset:

```
bash examples/eval_humaneval_102B.sh
```

### Results

The evaluation results will be gathered in `samples.jsonl` in `$OUTPUT\_PATH`. After the generation of all the tasks done, the `evaluate_functional_correctness` function of HumanEval would automatically evaluate the results and return the accuracy.
