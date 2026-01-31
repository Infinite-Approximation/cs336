import torch
import torch.nn as nn
from typing import Optional
from cs336_basics.model.attention import softmax
from cs336_basics.tokenizer.tokenizer import Tokenizer

torch.set_float32_matmul_precision("high")

def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """
    Generate text completions from a language model using nucleus (top-p) sampling.

    Args:
        model: The language model to use for generation (e.g., TransformerLM).
        input_ids: Input tensor of token IDs, shape (batch_size, seq_len).
        max_new_tokens: Maximum number of new tokens to generate.
        eos_token_id: The token ID that signals the end of generation.
        temperature: Temperature for softmax scaling. Higher values make output more random.
                    Set to 1.0 for no scaling, 0.0 for greedy decoding.
        top_p: Nucleus sampling threshold. Only sample from the smallest set of tokens
               whose cumulative probability exceeds top_p. Set to 1.0 to disable.
        device: Device to run generation on.

    Returns:
        Generated token IDs tensor of shape (batch_size, seq_len + generated_len).
    """
    input_ids = torch.tensor(input_ids).to(device).long()
    # 如果是一维的，就修改为 (1, seq_len)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    for _ in range(max_new_tokens):
        logits = model(input_ids)
        # 只需要最后一个时间步
        next_token_logits = logits[:, -1, :]
        if temperature == 0.0:  # 贪婪采样
            probs_greedy = softmax(next_token_logits)
            next_token = torch.argmax(probs_greedy, dim=-1, keepdim=True)
        else:
            next_token_logits /= temperature
            probs = softmax(next_token_logits)  # shape: (1, vocab_size)
            if top_p < 1.0:  # 才用top-p采样
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                # 需要注意可能第一个概率就是大于top_p的
                sorted_indices_to_remove[..., 0] = 0

                sorted_probs[sorted_indices_to_remove] = 0.0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

                # 采样
                next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
                next_token = torch.gather(sorted_indices, -1, next_token_idx)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
        if (next_token == eos_token_id).any():
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return input_ids


if __name__ == "__main__":
    import argparse
    from cs336_basics.model.transformer_lm import TransformerLM
    from cs336_basics.checkpoint import load_checkpoint

    parser = argparse.ArgumentParser(description="Generate text from a trained model")
    parser.add_argument(
        "--vocab_file", type=str, default="cs336_basics/tokenizer/res/tinystories_vocab.json"
    )
    parser.add_argument(
        "--merge_file", type=str, default="cs336_basics/tokenizer/res/tinystories_merges.txt"
    )

    parser.add_argument(
        "--checkpoint", type=str, default='checkpoints/tinystories_17m_bs_128_lr_6.00e-03/checkpoint_final.pt', help="Path to model checkpoint"
    )
    parser.add_argument(
        "--prompt", type=str, default="Once upon a time", help="Input prompt text"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--eos_token_id",
        type=int,
        default=256,
        help="End-of-sequence token ID",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.9, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Nucleus sampling threshold"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # Model hyperparameters (should match the trained model)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--norm_eps", type=float, default=1e-5)

    args = parser.parse_args()

    # Create model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        norm_eps=args.norm_eps,
        device=args.device,
        dtype=torch.float32,
    )
    model = torch.compile(model)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tokenizer = Tokenizer.from_files(args.vocab_file, args.merge_file, special_tokens=['<|endoftext|>'])
    input_ids = tokenizer.encode(args.prompt)
    generated_ids = generate(model, input_ids, args.max_new_tokens, eos_token_id=256,
                            temperature=args.temperature, top_p=args.top_p, device=args.device)
    print("Generated IDs:", generated_ids[0].tolist())
    output_text = tokenizer.decode(generated_ids[0].tolist())
    print("Prompt:")
    print(args.prompt)
    print("Generated Text:")
    print(output_text)
