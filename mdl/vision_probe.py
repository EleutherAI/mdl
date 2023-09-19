import torch
import torchvision as tv
from torch import Tensor, optim

from .probe import Probe


class VisionProbe(Probe):
    """Probe based on a TorchVision model. Defaults to ResNet-18."""

    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        model: str = "resnet18",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(num_features, num_classes, device, dtype)

        net = tv.models.get_model(model, num_classes=num_classes)
        self.net = net.to(device=device, dtype=dtype)  # type: ignore
        self.norm = tv.transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        )
        self.train_augmentor = tv.transforms.Compose(
            [
                tv.transforms.RandomCrop(28),  # TODO: Make this configurable
            ]
        )
        self.test_augmentor = tv.transforms.Compose(
            [
                tv.transforms.CenterCrop(28),
            ]
        )

    def build_optimizer(self) -> optim.Optimizer:
        return optim.SGD(self.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)

    def augment_data(self, x: Tensor) -> Tensor:
        return self.train_augmentor(x) if self.training else self.test_augmentor(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(self.norm(x))
