import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module.

        Args:
            d_model: Hidden dimension of the model
            eps: Epsilon value for numerical stability
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        torch.nn.init.ones_(self.g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., d_model) and return a tensor of the same shape.
        """
        origin_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        rms = (torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps) ** 0.5
        normalized_x = x / rms * self.g
        return normalized_x.to(dtype=origin_dtype)