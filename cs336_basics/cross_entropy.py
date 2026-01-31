from jaxtyping import Float, Int
from torch import Tensor
import torch



def cross_entropy(
    logits: Float[Tensor, "... vocab_size"], targets: Int[Tensor, "..."]
) -> Float[Tensor, ""]:
    """
    Compute the cross entropy loss.

    This function computes the cross entropy ℓᵢ = -log(softmax(oᵢ)[xᵢ₊₁]) for predicted
    logits (oᵢ) and targets (xᵢ₊₁).

    The function should handle the following:
    - Subtract the largest element for numerical stability.
    - Cancel out log and exp whenever possible.
    - Handle any additional batch dimensions and return the average across the batch.
      As with section 3.3, we assume batch-like dimensions always come first, before
      the vocabulary size dimension.

    Args:
        logits: Float[Tensor, "... vocab_size"]
            Predicted logits, where the last dimension is the vocabulary size.
        targets: Int[Tensor, "..."]
            Target labels.

    Returns:
        Float[Tensor, ""]
            The average cross-entropy loss across the batch.
    """
    # 数值稳定性 - 减去最大值
    # 在vocab_size维度(最后一维)上找最大值
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
    logits_shifted = logits - max_logits

    log_sum_exp = torch.log(torch.sum(torch.exp(logits_shifted), dim=-1))
    # 这里需要取出 o_i[x_{i+1}]，由于是logits的shape是 b*s*v, 而targets的shape是 b*s，
    # 我们需要展平
    targets_shape = targets.shape
    logits_flat = logits.view(-1, logits.shape[-1])
    target_flat = targets.view(-1, 1)
    target_logits = torch.gather(logits_flat, 1, target_flat)
    target_logits = target_logits.view(targets_shape) # 还原成 b*s
    
    # 相加
    loss = -target_logits + max_logits.squeeze(-1) + log_sum_exp
    return loss.mean()
