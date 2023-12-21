# data\_process

## Introduction

Since Yuan2.0 runs under the Megatron framework, the text corpus needs to be transformed into token ids and stored in binary files before training. We provide **preprocess\_data\_yuan.py**, a script which helps to efficiently transform texts into token ids, and which is specifically designed for preprocessing Chinese corpus. The script can be found in the 'tools' directory.

The main variables in the code should be set as follows:

| Variable name      | Description                                                                                                                                                                                                                                         |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--input`          | The path where you store the training datasets, the datasets should be stored in .txt files. Note: even there is only one .txt file needs to be processed, the path should be where you put the .txt (i.e. the folder), not the path for the .txt.  |
| `--data-idx`       | This sets up the indices for the training dataset. If there is just one dataset to convert, the --data-idx should be set to '0'. If there are multiple training datasets, set it as '0-n', where n is the number of training datasets.              |
| `--tokenizer_path` | The path to import tokenizer files.                                                                                                                                                                                                                 |
| `--output_path`    | The path where to store the preprocessed dataset, one .idx file and one .bin file would be created for each dataset.                                                                                                                                |



## Dataset

Samples in dataset should be seperated with '\n', and within each sample, the '\n' should be replaced with '\<n>', therefore each line in the dataset is a single sample. And the program would replace the '\<n>' back to '\n' during preprocessing.&#x20;

For the datasets used to finetune Yuan2.0, you shall put a '\<sep>' between the instruction and the response.&#x20;

The following is an example of samples in finetune dataset:

```text
John买了3件衬衫，每件售价为20美元。此外，他还需要支付所有商品的10%税款。他总共支付了多少钱？<sep>John购买的3件衬衫的总价为3 \times 20 = 60美元。<n>所有商品的税款为总价的10%，即60 \times 0.1 = 6美元。<n>因此，John总共支付的钱数为60 + 6 = 66美元。
每年，Dani作为赢得亚马逊季度最佳买家的奖励，会得到4个一对裤子（每对裤子2条）。如果初始时他有50条裤子，计算出5年后他将拥有多少条裤子。<sep>每年Dani会得到4 \times 2 = 8条裤子，因此5年后他将得到8 \times 5 = 40条裤子。<n>那么，5年后他总共拥有的裤子数量为初始时的50条加上5年内得到的40条，即50 + 40 = 90条裤子。<n>因此，5年后他将拥有90条裤子。
```



## Usage

Run the following command to initiate data processing.

```text
python ./tools/preprocess_data_yuan.py --input '<Specify path>' --data-idx '0-42' --tokenizer_path './tokenizer' --output_path '<Specify path>'
```

If a dataset has already been processed, i.e. its .idx file and .bin file have been existed in the '—output\_path', the program would skip this dataset.&#x20;

