"""Training script for language model."""

import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
import math
import wandb

# Enable TensorFloat32 for better performance on Ampere GPUs
torch.set_float32_matmul_precision("high")


from tqdm import tqdm
from cs336_basics.model.transformer_lm import TransformerLM
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.data import get_batch
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.cross_entropy import cross_entropy


def train(
    # Model hyperparameters
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    norm_eps: float,
    dtype: torch.dtype,
    # Optimizer hyperparameters
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    eps: float,
    max_grad_norm: float,
    # Training hyperparameters
    batch_size: int,
    max_iters: int,
    warmup_iters: int,
    # Data paths
    train_data_path: str | os.PathLike,
    val_data_path: str | os.PathLike,
    # Checkpointing
    checkpoint_dir: str | os.PathLike,
    save_interval: int,
    # Logging
    log_interval: int,
    eval_interval: int,
    eval_iters: int,
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    # Optional: resume from checkpoint
    resume_from: str | os.PathLike | None = None,
    # WandB
    use_wandb: bool = False,
    wandb_project: str = "transformer-lm",
    wandb_run_name: str | None = None,
    use_pre_rmsnorm: bool = True,
    use_post_rmsnorm: bool = False,
    use_rope: bool = True,
    use_SiLU: bool = False,
) -> None:
    """
    Main training loop for language model.
    """
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
        )

    train_data = np.load(train_data_path, mmap_mode="r")
    val_data = np.load(val_data_path, mmap_mode="r")
    model = TransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
        norm_eps,
        device,
        dtype,
        use_SiLU
    )
    model = torch.compile(model)
    optimizer = AdamW(
        model.parameters(), learning_rate, (beta1, beta2), eps, weight_decay
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    start_time = time.time()
    for iter in tqdm(range(max_iters)):
        x, y = get_batch(train_data, batch_size, context_length, device)
        logits = model(input_ids=x, positions=None, use_pre_rmsnorm=use_pre_rmsnorm, 
                       use_post_rmsnorm=use_post_rmsnorm, use_rope=use_rope)
        loss = cross_entropy(logits, y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), max_grad_norm)

        # 计算当前步的学习率
        lr = get_lr_cosine_schedule(
            it=iter,
            max_learning_rate=learning_rate,
            min_learning_rate=learning_rate * 0.1,  # 常见的做法是衰减到最大值的 10%
            warmup_iters=warmup_iters,
            cosine_cycle_iters=max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        # 记录训练指标
        if iter % log_interval == 0:
            log_dict = {
                "train_loss": loss.item(),
                "learning_rate": lr,
                "wallclock_time": time.time() - start_time,
            }
            if use_wandb:
                wandb.log(log_dict, step=iter)
            else:
                print(f"Step {iter}: Loss={loss.item():.4f}, LR={lr:.6f}")

        # 定期在评估集上面进行评估
        if iter % eval_interval == 0:
            val_loss = evaluate(
                model, val_data, batch_size, context_length, eval_iters, device
            )
            if use_wandb:
                wandb.log(
                    {"val_loss": val_loss, "perplexity": math.exp(val_loss)}, step=iter
                )

        # 保存checkpoint
        if iter % save_interval == 0:
            save_path = os.path.join(checkpoint_dir, f"checkpoint_{iter}.pt")
            save_checkpoint(model, optimizer, iter, save_path)

    if use_wandb:
        wandb.finish()

    save_path = os.path.join(checkpoint_dir, f"checkpoint_final.pt")
    save_checkpoint(model, optimizer, max_iters, save_path)


def evaluate(
    model: nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    device: str,
) -> float:
    """
    Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        dataset: Dataset to evaluate on (memory-mapped array)
        batch_size: Batch size for evaluation
        context_length: Context length
        eval_iters: Number of batches to evaluate on
        device: Device to evaluate on

    Returns:
        Average loss over evaluation batches
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for iter in range(eval_iters):
            x, y = get_batch(dataset, batch_size, context_length, device)
            logits = model(x)
            loss = cross_entropy(logits, y)
            total_loss += loss.item()
    avg_loss = total_loss / eval_iters
    perplexity = math.exp(avg_loss)

    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")

    model.train()
    return avg_loss


def main():
    """Parse command-line arguments and run training."""
    parser = argparse.ArgumentParser(description="Train a Transformer language model")

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--norm_eps", type=float, default=1e-5)

    # Optimizer hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--warmup_iters", type=int, default=1000)

    # Data paths
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--save_interval", type=int, default=1000)

    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=1)

    # Device and Data type
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )

    # Resume
    parser.add_argument("--resume_from", type=str, default=None)

    # WandB
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="transformer-lm", help="WandB project name"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="WandB run name"
    )

    # 消融实验
    parser.add_argument('--no_pre_rmsnorm', action='store_false', dest='use_pre_rmsnorm', default=True)
    parser.add_argument('--use_post_rmsnorm', action='store_true', default=False)
    parser.add_argument('--no_rope', action='store_false', dest='use_rope', default=True)
    parser.add_argument('--use_SiLU', action='store_true', default=False)

    args = parser.parse_args()
    # 转换 dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    train(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        norm_eps=args.norm_eps,
        dtype=dtype,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        warmup_iters=args.warmup_iters,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        device=args.device,
        resume_from=args.resume_from,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        use_pre_rmsnorm=args.use_pre_rmsnorm,
        use_post_rmsnorm=args.use_post_rmsnorm,
        use_rope=args.use_rope,
        use_SiLU=args.use_SiLU,
    )


if __name__ == "__main__":
    main()
