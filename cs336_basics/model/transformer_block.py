import torch
import torch.nn as nn
from .attention import MultiHeadSelfAttention
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        norm_eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        use_SiLU: bool = False,
    ):
        """
        Initialize the Transformer block.

        Args:
            d_model: Dimensionality of the Transformer block inputs/outputs.
            num_heads: Number of heads to use in multi-head self-attention.
            d_ff: Dimensionality of the position-wise feed-forward inner layer.
            dropout: Dropout probability.
            norm_eps: Epsilon for RMSNorm.
            device: Device to place parameters on.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.ln1 = RMSNorm(d_model, norm_eps, device, dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, device, dtype, theta, max_seq_len)
        self.ln2 = RMSNorm(d_model, norm_eps, device, dtype)
        if use_SiLU:
            from .swiglu import SiLU
            self.mlp = SiLU(d_model, d_ff, device, dtype)
        else:
            self.mlp = SwiGLU(d_model, d_ff, device, dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
        use_pre_rmsnorm: bool = True,
        use_post_rmsnorm: bool = False,
        use_rope: bool = True,
    ) -> torch.Tensor:
        """
        Apply the Transformer block.

        Args:
            hidden_states: Input tensor of shape (..., seq_len, d_model)
            positions: Token positions of shape (..., seq_len)

        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        # Pre-norm Attention block
        residual = hidden_states.clone()
        if use_pre_rmsnorm:
            hidden_states = self.ln1(hidden_states)
        hidden_states = self.attn(hidden_states, positions, use_rope)
        hidden_states = residual + hidden_states
        if use_post_rmsnorm:
            hidden_states = self.ln1(hidden_states)
        # Pre-norm FFN block
        residual = hidden_states.clone()
        if use_pre_rmsnorm:
            hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        if use_post_rmsnorm:
            hidden_states = self.ln2(hidden_states)
        return hidden_states