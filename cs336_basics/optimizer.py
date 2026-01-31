from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import matplotlib.pyplot as plt


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get(
                    "t", 0
                )  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (α)
        betas: Coefficients (β1, β2) for computing running averages of gradient and its square
        eps: Term (ε) added to denominator for numerical stability
        weight_decay: Weight decay coefficient (λ)
    """

    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                g = p.grad
                t = state.get('t', 1)
                m  = state.get('m', torch.zeros_like(p))
                v = state.get('v', torch.zeros_like(p))
                # 更新m, v
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * g**2
                state['t'] = t + 1
                state['m'] = m
                state['v'] = v

                alpha_t = lr * (1 - beta2**t)**0.5 / (1 - beta1**t)
                p.data -= alpha_t * m / (torch.sqrt(v)+eps)
                p.data -= lr * weight_decay * p.data

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Compute learning rate with cosine annealing schedule and linear warmup.
    
    Args:
        it: Current iteration number
        max_learning_rate: Maximum learning rate (α_max)
        min_learning_rate: Minimum learning rate (α_min)
        warmup_iters: Number of warmup iterations (T_w)
        cosine_cycle_iters: Number of cosine annealing iterations (T_c)
    
    Returns:
        Learning rate at the given iteration
    """
    import math
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif warmup_iters <= it and it <= cosine_cycle_iters:
        return min_learning_rate + 1/2 * (1 + math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate
    
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """
    Clip the combined gradients of parameters to have L2 norm at most max_l2_norm.
    
    Args:
        parameters: Collection of trainable parameters
        max_l2_norm: Maximum L2-norm for the gradients
        
    The gradients of the parameters (parameter.grad) are modified in-place.
    """
    # 1. 计算所有梯度的总 L2 norm
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            total_norm += torch.sum(param.grad ** 2)
    total_norm = torch.sqrt(total_norm)
    
    # 2. 如果总norm超过max_l2_norm，那么就裁剪
    clip_coef = max_l2_norm / (total_norm + eps)
    if total_norm >= max_l2_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad.mul_(clip_coef)

# Example usage:
if __name__ == "__main__":
    lrs = [1e1, 1e2, 1e3]
    losses_per_lr = {}

    for lr in lrs:
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)
        losses = []

        for t in range(10):
            opt.zero_grad()  # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean()  # Compute a scalar loss value.
            losses.append(loss.cpu().item())
            print(f"LR={lr}, Iteration {t}: Loss={loss.cpu().item():.6f}")
            loss.backward()  # Run backward pass, which computes gradients.
            opt.step()  # Run optimizer step.

        losses_per_lr[lr] = losses

    # Plot the loss trends
    plt.figure(figsize=(10, 6))
    for lr, losses in losses_per_lr.items():
        plt.plot(range(10), losses, marker="o", label=f"LR={lr}")

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Trends for Different Learning Rates")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")  # Use log scale for better visualization
    plt.savefig("loss_trends.png")
    plt.show()
    print("Plot saved as 'loss_trends.png'")
