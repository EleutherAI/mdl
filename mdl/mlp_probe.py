from functools import partial
from itertools import pairwise

import torch
from torch import Tensor, nn, optim

from .probe import Probe


class MlpProbe(Probe):
    """Multi-layer perceptron probe with GELU activation."""

    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        num_layers: int = 2,
    ):
        super().__init__(num_features, num_classes, device, dtype)
        self.num_layers = num_layers

        # Same expansion ratio as Vaswani et al. (2017)
        hidden_dim = (
            4 * num_features if num_layers == 2 else round(num_features * 4 / 3)
        )
        output_dim = num_classes if num_classes > 2 else 1
        sizes = [num_features] + [hidden_dim] * (num_layers - 1) + [output_dim]

        self.net = nn.Sequential()
        for in_dim, out_dim in pairwise(sizes):
            self.net.append(
                nn.Linear(in_dim, out_dim, device=device, dtype=dtype),
            )
            self.net.append(nn.GELU())

        self.net.pop(-1)  # Remove last activation

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)

    def build_optimizer(self) -> optim.Optimizer:
        if self.num_layers > 1:
            return optim.AdamW(self.parameters())
        else:
            # Use Nesterov SGD for linear probes. The problem is convex and there's
            # really no need to use an adaptive learning rate. We can set the fixed
            # LR considerably higher and this seems to help with convergence.
            return optim.SGD(
                self.parameters(),
                # Learning rate of 0.1 with momentum 0.9 is "really" an LR of unity in
                # PyTorch's parametrization; see https://youtu.be/k8fTYJPd3_I
                lr=0.1,
                momentum=0.9,
                # Nesterov seems to be strictly better than regular momentum
                nesterov=True,
                # Use same weight decay as AdamW above
                weight_decay=0.01,
            )


# Convenience alias
LinearProbe = partial(MlpProbe, num_layers=1)
