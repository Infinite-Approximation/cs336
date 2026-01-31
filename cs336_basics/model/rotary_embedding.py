import torch
import torch.nn as nn
from einops import einsum

class RotaryPositionalEmbeddingNaive(nn.Module):
    """
    这是最朴素的实现，也就是x中的元素两两旋转，
    通过 x[0::2] 取出来的是实部，通过 x[1::2] 取出来的是虚部
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ):
        """
        Construct the RoPE module and create buffers if needed.

        Args:
            theta: Θ value for the RoPE
            d_k: Dimension of query and key vectors
            max_seq_len: Maximum sequence length that will be inputted
            device: Device to store the buffer on
        """
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 计算频率: 1 / (theta^(2i/d_k)) for i in [0, 1, ..., d_k/2-1]
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float) / d_k)
        )
        positions = torch.arange(0, max_seq_len, device=device, dtype=torch.float)
        freq = einsum(positions, inv_freq, "m, i->m i")
        cos = freq.cos()
        sin = freq.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process an input tensor and return a tensor of the same shape.

        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Token positions of shape (..., seq_len) specifying
                           the token positions of x along the sequence dimension

        Returns:
            Tensor of the same shape as x with RoPE applied
        """
        x = x.clone()  # 复制一份，避免修改原始输入
        cos_sin = self.cos_sin_cache[token_positions]
        cos, sin = torch.chunk(cos_sin, chunks=2, dim=-1)
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        x[..., 0::2] = y0
        x[..., 1::2] = y1
        return x


class RotaryPositionalEmbeddingInLLama(nn.Module):
    """
    这是LLama的实现，也就是将x的前半部分看成是实部，将后半部分看作是虚部
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ):
        """
        Construct the RoPE module and create buffers if needed.

        Args:
            theta: Θ value for the RoPE
            d_k: Dimension of query and key vectors
            max_seq_len: Maximum sequence length that will be inputted
            device: Device to store the buffer on
        """
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 计算频率: 1 / (theta^(2i/d_k)) for i in [0, 1, ..., d_k/2-1]
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float) / d_k)
        )
        positions = torch.arange(0, max_seq_len, device=device, dtype=torch.float)
        freq = einsum(positions, inv_freq, "m, i->m i")
        cos = freq.cos()
        sin = freq.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process an input tensor and return a tensor of the same shape.

        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Token positions of shape (..., seq_len) specifying
                           the token positions of x along the sequence dimension

        Returns:
            Tensor of the same shape as x with RoPE applied
        """
        cos_sin = self.cos_sin_cache[token_positions]
        cos, sin = torch.chunk(cos_sin, chunks=2, dim=-1)
        x0, x1 = torch.chunk(x, chunks=2, dim=-1)
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.cat((y0, y1), dim=-1)
