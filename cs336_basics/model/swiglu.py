import torch
import torch.nn as nn
import math


def silu(x):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        """
        Construct the SwiGLU position-wise feed-forward network.

        Args:
            d_model: Hidden dimension of the model
            d_ff: Inner feed-forward dimension (≈ 8/3 * d_model, typically multiple of 64)
            device: Device to store parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        if d_ff is None:
            d_ff = math.ceil((8 / 3 * d_model) / 64) * 64
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))

        # Xavier / Glorot initialization
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w3)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the SwiGLU feed-forward network to input of shape (..., d_model).
        Returns tensor of shape (..., d_model).
        """
        return (silu(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T

class SiLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        """
        SiLU position-wise feed-forward network (2 matrices).
        
        Args:
            d_model: Hidden dimension of the model
            d_ff: Inner feed-forward dimension
            device: Device to store parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))

        # Xavier / Glorot initialization
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SiLU FFN: silu(x @ W1.T) @ W2.T
        
        Args:
            x: Input of shape (..., d_model)
        
        Returns:
            Output of shape (..., d_model)
        """
        return silu(x @ self.w1.T) @ self.w2.T