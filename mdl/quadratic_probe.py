import torch
from torch import Tensor, nn, optim
from mup import MuAdam
from schedulefree import AdamWScheduleFree

from .probe import Probe


class QuadraticProbe(Probe):
    """Probe of the form `y_i = x.T @ A @ x + b.T @ x + c`."""
    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        learning_rate: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        schedule_free: bool = False,
        mup: bool = False,
    ):
        super().__init__(num_features, num_classes, device, dtype)

        self.learning_rate = learning_rate
        self.betas = betas
        self.schedule_free = schedule_free
        self.mup = mup

        self.norm = nn.BatchNorm1d(num_classes, device=device, dtype=dtype)
        self.bilinear = nn.Bilinear(
            num_features,
            num_features,
            num_classes,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.linear = nn.Linear(
            num_features,
            num_classes,
            device=device,
            dtype=dtype,
        )

    def build_optimizer(self) -> optim.Optimizer:
        opt_cls = AdamWScheduleFree if self.schedule_free else optim.AdamW
        if self.mup:
            return MuAdam(self.parameters(), opt_cls, lr=self.learning_rate, betas=self.betas)
        return opt_cls(self.parameters(), lr=self.learning_rate, betas=self.betas)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.bilinear(x, x)) + self.linear(x)
