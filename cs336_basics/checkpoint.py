import os
from typing import BinaryIO, IO
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Save model and optimizer states to a checkpoint file.
    
    Args:
        model: PyTorch model to save
        optimizer: PyTorch optimizer to save
        iteration: Current training iteration number
        out: Output path or file-like object to save checkpoint to
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)
    


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load model and optimizer states from a checkpoint file.
    
    Args:
        src: Source path or file-like object to load checkpoint from
        model: PyTorch model to restore state into
        optimizer: PyTorch optimizer to restore state into
        
    Returns:
        The iteration number that was saved in the checkpoint
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']