import torch
from torch import Tensor, nn
from torch.nn.functional import gelu

from .probe import Probe


class MlpProbe(Probe):
    """Two-layer perceptron probe with GELU activation."""

    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(num_classes, num_features, device, dtype)

        # Same expansion ratio as Vaswani et al. (2017)
        hidden_dim = 4 * num_features
        output_dim = num_classes if num_classes > 2 else 1

        self.inner = nn.Linear(num_features, hidden_dim, device=device, dtype=dtype)
        self.outer = nn.Linear(hidden_dim, output_dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.outer(gelu(self.inner(x))).squeeze(-1)

    def reset_parameters(self):
        """Reset parameters, if necessary, before fitting a new model."""
        self.inner.reset_parameters()
        self.outer.reset_parameters()
