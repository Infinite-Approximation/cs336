import argparse
import numpy as np
import numpy.typing as npt
import torch

from cs336_basics.tokenizer.tokenizer import Tokenizer

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample language modeling input sequences and their corresponding labels from the dataset.
    
    Args:
        dataset: 1D numpy array of integer token IDs in the dataset
        batch_size: Number of sequences to sample
        context_length: Length of each sampled sequence
        device: PyTorch device string (e.g., 'cpu' or 'cuda:0')
        
    Returns:
        Tuple of two torch.LongTensors of shape (batch_size, context_length):
        - First tensor: sampled input sequences
        - Second tensor: corresponding next-token targets
    """
    max_start_idx = len(dataset) - context_length
    # 随机采样
    start_indices = np.random.randint(0, max_start_idx, batch_size)
    x = torch.zeros((batch_size, context_length), device=device, dtype=torch.long)
    y = torch.zeros((batch_size, context_length), device=device, dtype=torch.long)
    for i in range(batch_size):
        # 随机选取一个开始位置
        start_idx = start_indices[i]
        x[i] = torch.from_numpy(dataset[start_idx: start_idx + context_length].copy())
        y[i] = torch.from_numpy(dataset[start_idx + 1: start_idx + context_length + 1].copy())
    return x, y


def prepare_data(
    vocab_file: str,
    merges_file: str,
    input_file: str,
    output_file: str
) -> None:
    tokenizer = Tokenizer.from_files(vocab_file, merges_file, special_tokens=['<|endoftext|>'])
    # Read input file
    print(f"Reading input file: {input_file}")
    all_ids = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)

    # Save as numpy array
    print(f"Saving {len(all_ids)} tokens to {output_file}")
    np.save(output_file, np.array(all_ids, dtype=np.uint32))
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--vocab", type=str, required=True, help="Vocabulary JSON file")
    parser.add_argument("--merges", type=str, required=True, help="Merges text file")
    parser.add_argument("--input", type=str, required=True, help="Input text file")
    parser.add_argument("--output", type=str, required=True, help="Output .npy file")
    
    args = parser.parse_args()
    
    prepare_data(
        vocab_file=args.vocab,
        merges_file=args.merges,
        input_file=args.input,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()