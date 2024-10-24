import torch
import torchvision as tv
from torch import Tensor, nn, optim
from transformers import ViTConfig, ViTForImageClassification, ConvNextV2Config, ConvNextV2ForImageClassification

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


class ViTProbe(Probe):
    def __init__(
            self, 
            # Unused
            num_features: int, 
            hidden_size: int,
            num_classes: int = 2, 
            num_layers: int = 2,
            device: str | torch.device | None = None, 
            dtype: torch.dtype | None = None
    ):
        super().__init__(num_features, num_classes, device, dtype)

        cfg = ViTConfig(
            image_size=32,
            num_channels=3,
            patch_size=1,
            num_labels=num_classes,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            intermediate_size=256,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
        )
        self.net = ViTForImageClassification(cfg).to(device)

    def build_optimizer(self):
        return torch.optim.SGD(
            self.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, 3, 32, 32)
        return self.net(x).logits


class ConvNextProbe(Probe):
    def __init__(
            self,
            # Unused
            num_features: int, 
            num_classes: int = 2, 
            num_layers: int = 2,
            hidden_size: int = 2,
            device: str | torch.device | None = None, 
            dtype: torch.dtype | None = None
        ):
        super().__init__(num_features, num_classes, device, dtype)
        
        depths = [2, 2, 6, 2]

        # Double hidden size at each additional layer
        hidden_sizes = [hidden_size] + [hidden_size * 2 ** i for i in range(1, num_layers)]
        
        cfg = ConvNextV2Config(
                image_size=32,
                num_channels=3,
                depths=depths,
                drop_path_rate=0.1,
                hidden_sizes=hidden_sizes,
                num_labels=num_classes,
                # The default of 4 x 4 patches shrinks the image too aggressively for
                # low-resolution images like CIFAR-10
                patch_size=1,
            )
        self.net = ConvNextV2ForImageClassification(cfg).to(device)
    
    def build_optimizer(self):
        return torch.optim.SGD(
            self.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4
        )

    def forward(self, x):
        x = x.reshape(-1, 3, 32, 32)
        return self.net(x).logits
