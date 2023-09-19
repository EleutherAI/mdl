import torch
from torch import Tensor, nn, optim

from .probe import Probe


class QuadraticProbe(Probe):
    """Probe of the form `y_i = x.T @ A @ x + b.T @ x + c`."""

    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(num_features, num_classes, device, dtype)

        self.bilinear = nn.Bilinear(
            num_features,
            num_features,
            num_classes,
            device=device,
            dtype=dtype,
        )
        # nn.Bilinear already has a bias term, so we don't need to add one here
        self.linear = nn.Linear(
            num_features,
            num_classes,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def build_optimizer(self) -> optim.Optimizer:
        return optim.AdamW(self.parameters())

    def forward(self, x: Tensor) -> Tensor:
        return self.bilinear(x, x) + self.linear(x)
