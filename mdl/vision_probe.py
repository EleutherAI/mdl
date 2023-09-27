import torch
import torchvision as tv
from torch import Tensor, nn, optim

from .probe import Probe


class VisionProbe(Probe):
    """Probe based on a TorchVision model. Defaults to ResNet-18."""

    def __init__(
        self,
        num_classes: int = 2,
        learning_rate: float = 0.005,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        model: str = "resnet18",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        num_features: int = 3,  # Unused
        pretrained: bool = False,
    ):
        super().__init__(num_features, num_classes, device, dtype)

        if not pretrained:
            net = tv.models.get_model(model, num_classes=num_classes)
        else:
            net = tv.models.resnet18(pretrained=pretrained)
            net.fc = nn.Linear(net.fc.in_features, num_classes)

        self.net = net.to(device=device, dtype=dtype)  # type: ignore
        if model == "resnet18":
            self.net.conv1 = torch.nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
                device=device,
                dtype=dtype,
            )
            self.net.maxpool = torch.nn.Identity(device=device, dtype=dtype)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.norm = tv.transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        )
        if model == "resnet18" and not pretrained:
            net.conv1 = nn.Conv2d(
                3,
                64,
                3,
                stride=1,
                padding="same",
                bias=False,
                device=device,
                dtype=dtype,
            )
            net.maxpool = nn.Identity()

    def build_optimizer(self) -> optim.Optimizer:
        return optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(self.norm(x))
