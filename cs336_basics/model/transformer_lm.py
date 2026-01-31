import torch
import torch.nn as nn
from .embedding import Embedding
from .transformer_block import TransformerBlock
from .rmsnorm import RMSNorm
from .linear import Linear
from .attention import softmax

class TransformerLM(nn.Module):
    """
    Transformer Language Model.

    Stacks multiple Transformer blocks and applies them to token embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        norm_eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        use_SiLU: bool = False,
    ):
        """
        Initialize the Transformer Language Model.

        Args:
            vocab_size: The size of the vocabulary (number of unique tokens).
            context_length: The maximum context length (max sequence length).
            d_model: The dimensionality of the model embeddings and sublayer outputs.
            num_layers: The number of Transformer blocks to stack.
            num_heads: Number of heads to use in multi-headed attention.
            d_ff: Dimensionality of the feed-forward inner layer.
            rope_theta: The RoPE Θ parameter.
            norm_eps: Epsilon for RMSNorm layers.
            device: Device to place parameters on.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.embed_tokens = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, norm_eps, device, dtype, use_SiLU) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(d_model, norm_eps, device, dtype)
        self.output_embed = Linear(d_model, vocab_size, device, dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        use_pre_rmsnorm: bool = True,
        use_post_rmsnorm: bool = False,
        use_rope: bool = True,
    ) -> torch.Tensor:
        """
        Apply the Transformer LM to input token IDs.

        Args:
            input_ids: Input tensor of token IDs, shape (batch_size, seq_len).

        Returns:
            Output logits of shape (batch_size, seq_len, vocab_size).
        """
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, use_pre_rmsnorm, use_post_rmsnorm, use_rope)
        if use_pre_rmsnorm:
            hidden_states = self.norm(hidden_states)
        logits = self.output_embed(hidden_states)
        return logits