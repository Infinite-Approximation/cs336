import time
import json
import os
import psutil
from cs336_basics.tokenizer.tokenizer import train_bpe


def run_experiment():
    # 1. 路径和参数配置
    # 注意：请根据你的实际路径修改数据集位置
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    print(f"开始训练 BPE 分词器...")
    print(f"数据集: {input_path}")
    print(f"目标词表大小: {vocab_size}")

    # 2. 测量开始时间与内存
    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / (1024**3)  # GB

    # 3. 执行训练
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    # 4. 测量结束时间与内存
    end_time = time.time()
    end_mem = process.memory_info().rss / (1024**3)  # GB
    duration_mins = (end_time - start_time) / 60

    print("-" * 30)
    print(f"训练完成！")
    peak_mem = max(start_mem, end_mem)
    print(f"耗时: {duration_mins:.2f} 分钟")
    print(f"内存占用: 峰值约 {peak_mem:.2f} GB")

    # 5. 找到最长的token
    id, longest_token = max(vocab.items(), key=lambda x: len(x[1]))
    print(f"最长的 token ID: {id}, 内容: {longest_token}, 长度: {len(longest_token)} 字节")
    # 保存实验指标
    stats = {
        "dataset": input_path,
        "vocab_size": vocab_size,
        "duration_mins": round(duration_mins, 2),
        "peak_memory_gb": round(peak_mem, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("cs336_basics/tokenizer/res/tinystories_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    print(f"性能指标已保存至: cs336_basics/tokenizer/res/tinystories_stats.json")

    # Serialize the resulting vocabulary and merges to disk for further inspection.
    serializable_vocab = {id: list(b) for id, b in vocab.items()}
    with open("cs336_basics/tokenizer/res/tinystories_vocab.json", "w") as f:
        json.dump(serializable_vocab, f)

    # 保存合并规则 (存为文本，每行一对)
    # 我们用十六进制表示字节，防止特殊字符导致文件不可读
    with open("cs336_basics/tokenizer/res/tinystories_merges.txt", "w") as f:
        for m1, m2 in merges:
            f.write(f"{m1.hex()} {m2.hex()}\n")

    print("保存成功：tinystories_vocab.json, tinystories_merges.txt")


if __name__ == "__main__":
    run_experiment()
