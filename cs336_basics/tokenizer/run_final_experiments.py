import os
import time
from cs336_basics.tokenizer.tokenizer import Tokenizer


def get_compression_ratio(tokenizer, documents):
    total_bytes = 0
    total_tokens = 0
    for doc in documents:
        total_bytes += len(doc.encode("utf-8"))
        tokens = tokenizer.encode(doc)
        total_tokens += len(tokens)
    return total_bytes / total_tokens if total_tokens > 0 else 0


def run_final_experiments():
    res_dir = (
        "/home/jkd/online_course_learning/cs336/assignment1-basics/cs336_basics/res"
    )
    ts_vocab = os.path.join(res_dir, "tinystories_vocab.json")
    ts_merges = os.path.join(res_dir, "tinystories_merges.txt")
    owt_vocab = os.path.join(res_dir, "openwebtxt_vocab.json")
    owt_merges = os.path.join(res_dir, "openwebtxt_merges.txt")

    special_tokens = ["<|endoftext|>"]

    print("Loading tokenizers...")
    ts_tokenizer = Tokenizer.from_files(ts_vocab, ts_merges, special_tokens)
    owt_tokenizer = Tokenizer.from_files(owt_vocab, owt_merges, special_tokens)

    # Load sample documents
    def sample_docs(path, n=10):
        docs = []
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            # Split by <|endoftext|> and remove empty ones
            raw_docs = [d for d in content.split("<|endoftext|>") if d.strip()]
            docs = raw_docs[:n]
        return docs

    ts_val_path = "/home/jkd/online_course_learning/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    owt_val_path = (
        "/home/jkd/online_course_learning/cs336/assignment1-basics/data/owt_valid.txt"
    )

    ts_docs = sample_docs(ts_val_path)
    owt_docs = sample_docs(owt_val_path)

    print("\n--- (a) Compression Ratios ---")
    ts_ratio = get_compression_ratio(ts_tokenizer, ts_docs)
    owt_ratio = get_compression_ratio(owt_tokenizer, owt_docs)
    print(f"TinyStories (10K vocab) on TinyStories docs: {ts_ratio:.4f} bytes/token")
    print(f"OpenWebText (32K vocab) on OpenWebText docs: {owt_ratio:.4f} bytes/token")

    print("\n--- (b) Cross-Domain Analysis ---")
    cross_ratio = get_compression_ratio(ts_tokenizer, owt_docs)
    print(f"TinyStories (10K vocab) on OpenWebText docs: {cross_ratio:.4f} bytes/token")

    print("\n--- (c) Throughput Estimation ---")
    # Using OWT tokenizer on a larger chunk of OWT data
    large_sample = sample_docs(owt_val_path, n=100)
    total_text = "".join(large_sample)
    start_time = time.time()
    _ = owt_tokenizer.encode(total_text)
    end_time = time.time()

    duration = end_time - start_time
    bytes_processed = len(total_text.encode("utf-8"))
    throughput = bytes_processed / duration  # bytes/sec
    print(f"Throughput: {throughput:.2f} bytes/sec")

    pile_size_gb = 825
    pile_size_bytes = pile_size_gb * (1024**3)
    hrs_to_tokenize = (pile_size_bytes / throughput) / 3600
    print(f"Time to tokenize Pile ({pile_size_gb}GB): {hrs_to_tokenize:.2f} hours")


if __name__ == "__main__":
    run_final_experiments()
