import torch
from torch import Tensor

from .probe import Probe


class LinearProbe(Probe):
    """Linear probe trained with cross-entropy loss."""

    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(num_features, num_classes, device, dtype)

        self.linear = torch.nn.Linear(
            num_features,
            num_classes if num_classes > 2 else 1,
            device=device,
            dtype=dtype,
        )
        self.linear.bias.data.zero_()
        self.linear.weight.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x).squeeze(-1)
