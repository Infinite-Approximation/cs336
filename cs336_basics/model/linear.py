import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Construct a linear transformation module.
        
        Args:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        # Initialize weight matrix with shape (out_features, in_features)
        # This matches PyTorch convention where weights are stored as (d_out, d_in)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        
        # Calculate standard deviation according to the formula: sigma = sqrt(2 / (din + dout))
        std = (2 / (in_features + out_features)) ** 0.5
        
        # Apply truncated normal initialization: N(0, sigma^2) truncated at [-3sigma, 3sigma]
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        
        Args:
            x: torch.Tensor Input tensor
            
        Returns:
            torch.Tensor: The transformed output
        """
        # x: (..., in_features), weight: (out_features, in_features)
        # x @ weight.T: (..., in_features) @ (in_features, out_features) = (..., out_features)
        return x @ self.weight.T