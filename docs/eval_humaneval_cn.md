# eval\_humaneval

## 评测数据集

**datasets/HUMANEVAL/HumanEval.jsonl.gz** 英文原版的 [HumanEval](https://github.com/openai/human-eval "HumanEval") 评测数据集包含164 道问题。

**datasets/HUMANEVAL/HumanEval-textprompts.jsonl** 借助 gpt-4 翻译获得的中文版 HumanEval 数据集。

## 评测

### 简介

**examples/eval\_humaneval\_2B.sh.** 通过运行该程序，可以获得2B 模型在中文 HumanEval 评测数据集的评估结果。

**examples/eval\_humaneval\_51B.sh.** 通过运行该程序，可以获得51B 模型在中文 HumanEval 评测数据集的评估结果。

**examples/eval\_humaneval\_102B.sh.** 通过运行该程序，可以获得102B 模型在中文 HumanEval 评测数据集的评估结果。

在运行评测程序之前，你仅需在 bash 脚本中指定以下 checkpoint\_path参数，其他必要的路径已经设置好了。如果要评测自己合并的checkpoint，请务必将 bash 脚本中的 `--tensor-model-parallel-size` 参数改为新合并checkpoint上的张量并行数：

| 参数名称                           | 参数描述                  |
| ------------------------------ | --------------------- |
| `CHECKPOINT_PATH`              | 待评测的checkpoint的保存路径。  |
| `--tensor-model-parallel-size` | 待评测的checkpoint的张量并行数。 |

### 环境要求

在 Yuan2.0 checkpoint上运行 HumanEval 评测之前，确保已安装 HumanEval 程序。

```text
$ git clone https://github.com/openai/human-eval
$ pip install -e human-eval
```

安装好HumanEval后，请移步至此脚本：

```text
/usr/local/lib/python3.10/dist-packages/human_eval-1.0-py3.10.egg/human_eval/execution.py
```

并对 "check\_correctness "函数中的 "check\_program "变量作如下修改，以确保生成的代码中没有重复的函数签名。

```text
check_program = (
    #problem["prompt"] +
    completion + "\n" +
    problem["test"] + "\n" +
    f"check({problem['entry_point']})"
)

```

此外，如果您是第一次使用 HumanEval ，必须删除 "check\_program "中多余的 "#"，就在 "exec(check\_program, exec\_globals) "行之前。

```text
# WARNING
# This program exists to execute untrusted model-generated code. Although
# it is highly unlikely that model-generated code will do something overtly
# malicious in response to this test suite, model-generated code may act
# destructively due to a lack of model capability or alignment.
# Users are strongly encouraged to sandbox this evaluation suite so that it
# does not perform destructive actions on their host or network. For more
# information on how OpenAI sandboxes its code, see the accompanying paper.
# Once you have read this disclaimer and taken appropriate precautions,
# uncomment the following line and proceed at your own risk:
                         exec(check_program, exec_globals)
                result.append("passed")
            except TimeoutException:
                result.append("timed out")
            except BaseException as e:
                result.append(f"failed: {e}")

```



### 使用

运行以下命令分别评测 2B、51B 和 102B 模型在 HumanEval 数据集上的表现。运行 bash 脚本前，应将目录更改为 "Yuan-2.0 "主目录，并且只需在 bash 脚本中指定存放 checkpoint的路径，其他路径（如 tokenizer 和 HumanEval 数据集）已在 bash 脚本中设置好。



在HumanEval数据集上评测102B模型：

```text
cd <Specify Path>/Yuan-2.0/
bash examples/eval_humaneval_102B.sh
```

在HumanEval数据集上评测51B模型：

```text
cd <Specify Path>/Yuan-2.0/
bash examples/eval_humaneval_51B.sh
```

在HumanEval数据集上评测2B模型：

```text
cd <Specify Path>/Yuan-2.0/
bash examples/eval_humaneval_2B.sh
```

### 结果

评测结果将收集在 \$OUTPUT\_PATH 中的 samples.jsonl 中。生成所有任务后，HumanEval 的 "evaluate\_functional\_correctness "函数将自动评测结果并返回准确度。

