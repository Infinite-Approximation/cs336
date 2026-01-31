import torch
import torch.nn as nn
from einops import einsum
from einops import rearrange


def softmax(x: torch.Tensor, dim: int = -1):
    max_ele, _ = torch.max(x, dim=dim, keepdim=True)
    smaller_x = x - max_ele
    exp_x = torch.exp(smaller_x)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.

    Args:
        Q: torch.Tensor Query tensor of shape (..., q, d_k)
        K: torch.Tensor Key tensor of shape (..., k, d_k)
        V: torch.Tensor Value tensor of shape (..., k, d_v)
        mask: torch.Tensor | None Mask tensor of shape (q, k)

    Returns:
        torch.Tensor: The output tensor of shape (..., q, d_v)
    """
    d_k = Q.shape[-1]
    attention_score = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / d_k**0.5
    if mask is not None:
        attention_score.masked_fill_(mask == 0, float("-inf"))
    attn_weight = softmax(attention_score, dim=-1)
    output = einsum(attn_weight, V, "... q k, ... k d_v -> ... q d_v")
    return output


class MultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention module.

    Following Vaswani et al., 2017, sets d_k = d_v = d_model / num_heads.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        theta: float | None = None,
        max_seq_len: int | None = None,
    ):
        """
        Initialize multi-head self-attention.

        Args:
            d_model: Dimensionality of the transformer block inputs/outputs
            num_heads: Number of attention heads
            device: Device to place parameters on
            dtype: Data type for parameters
            theta: Optional RoPE theta parameter. If provided with max_seq_len, creates RoPE module
            max_seq_len: Optional maximum sequence length for RoPE
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # head dimension
        self.max_seq_len = max_seq_len
        # Create RoPE module if theta and max_seq_len are provided
        if theta is not None and max_seq_len is not None:
            from cs336_basics.model.rotary_embedding import RotaryPositionalEmbeddingNaive
            self.rope = RotaryPositionalEmbeddingNaive(theta, self.d_k, max_seq_len, device)
        else:
            self.rope = None

        # Initialize Q, K, V projection weights
        self.qkv_proj = nn.Parameter(
            torch.empty(3 * d_model, d_model, device=device, dtype=dtype) # output_size * input_size
        )

        # Initialize output projection weight
        self.o_proj = nn.Parameter(
            torch.empty(d_model, d_model, device=device, dtype=dtype)
        )

        # Xavier / Glorot initialization
        nn.init.xavier_uniform_(self.qkv_proj)
        nn.init.xavier_uniform_(self.o_proj)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        use_rope: bool = True,
    ) -> torch.Tensor:
        """
        Apply multi-head self-attention.

        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            mask: Optional causal mask of shape (seq_len, seq_len)
            token_positions: Optional token positions for RoPE of shape (..., seq_len)

        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        # 1. Project to Q, K, V
        qkv = einsum(
            x, self.qkv_proj, "... seq_len d_model, qkv d_model -> ... seq_len qkv"
        )
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # 2. Split into multiple heads
        q_heads = rearrange(
            q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads
        )
        k_heads = rearrange(
            k, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads
        )
        v_heads = rearrange(
            v, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads
        )

        seq_len = x.shape[-2]
        if token_positions is None:
           token_positions = torch.arange(seq_len, device=x.device)
        # Apply RoPE if available
        if self.rope is not None and use_rope:
            q_heads = self.rope(q_heads, token_positions)
            k_heads = self.rope(k_heads, token_positions)
        
        # 3. Apply scaled_dot_product_attention
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        attn_output = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)
        # 4. Concatenate heads
        concat_output = rearrange(
            attn_output, "... h seq_len d_k -> ... seq_len (h d_k)", h=self.num_heads
        )
        # 5. Apply output projection
        output = einsum(
            concat_output,
            self.o_proj,
            "... seq_len d_in, d_out d_in -> ... seq_len d_out",
        )
        return output