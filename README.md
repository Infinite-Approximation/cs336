# CS336 Spring 2025 Assignment 1: Basics

## 检查是否通过全部测试

```sh
uv run pytest
```

这个命令会自动安装对应的环境，然后进行测试。
可能会存在train_bpe_test这个样例没通过，这是因为如果你的cpu核数过大，那么创建多进程的时候会花费很多时间，经测试，128核的时候会不通过该测试，16核可以通过测试。如果无法通过测试，请修改 `/root/cs336/assignment1-basics/cs336_basics/tokenizer/tokenizer.py` 中的 `train_bpe` 函数，调整对应的核数。

## Tokenizer
实验相关代码均在 `cs336_basics/tokenizer` 目录下。

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

cd ..
```

