"""
使用 cProfile 对 BPE 训练进行性能分析
"""
import cProfile
import pstats
from pstats import SortKey
from cs336_basics.tokenizer.tokenizer import train_bpe

if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 4096
    special_tokens = ["<|endoftext|>"]

    print(f"Profiling BPE training with cProfile...")
    print(f"Dataset: {input_path}")
    print(f"Vocab size: {vocab_size}")
    print("-" * 80)

    # 运行 profiler
    profiler = cProfile.Profile()
    profiler.enable()

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    profiler.disable()

    # 创建统计对象
    stats = pstats.Stats(profiler)

    # 按累计时间排序，显示前 20 个最耗时的函数
    print("\n" + "=" * 80)
    print("Top 20 functions by cumulative time:")
    print("=" * 80)
    stats.sort_stats(SortKey.CUMULATIVE).print_stats(20)

    # 按函数自身时间排序
    print("\n" + "=" * 80)
    print("Top 20 functions by total time (excluding sub-calls):")
    print("=" * 80)
    stats.sort_stats(SortKey.TIME).print_stats(20)

    # 保存详细报告到文件
    stats.dump_stats("profile_results.prof")
    print("\n" + "=" * 80)
    print("Detailed profile saved to: profile_results.prof")
    print("You can view it with: python -m pstats profile_results.prof")
    print("=" * 80)
