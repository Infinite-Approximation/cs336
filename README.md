# CS336 Spring 2025 Assignment 1: Basics

## 检查是否通过全部测试

```sh
uv run pytest
```

这个命令会自动安装对应的环境，然后进行测试。
可能会存在train_bpe_test这个样例没通过，这是因为如果你的cpu核数过大，那么创建多进程的时候会花费很多时间，经测试，128核的时候会不通过该测试，16核可以通过测试。如果无法通过测试，请修改 `/root/cs336/assignment1-basics/cs336_basics/tokenizer/tokenizer.py` 中的 `train_bpe` 函数，调整对应的核数。

## Tokenizer
实验相关代码均在 `cs336_basics/tokenizer` 目录下。

### 进行tokenizer数据(TinyStories data)的获取
``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

cd ..
```

运行 
```sh
python cs336_basics/tokenizer/run_tinystories_experiments.py
```
训练结束会得到两个文件，一个是 `cs336_basics/tokenizer/res/openwebtxt_vocab.json`，另一个是 `cs336_basics/tokenizer/res/tinystories_merges.txt`。

## model
### 获取训练和评估数据
运行
```sh
./cs336_basics/train_script/data_prepare.sh
```
我们会得到训练数据 `data/TinyStoriesV2-GPT4-train_tokens.npy`，
以及评估数据 `data/TinyStoriesV2-GPT4-valid_tokens.npy`

### 开始训练
运行
```sh
./cs336_basics/train_script/train_demo.sh
```
我们可以在cpu上训练一个约9.31M的模型。

### 测试效果
运行
```sh
python cs336_basics/decode.py
```
我们就可以得到我们的LLM输出的结果了！
