import torch
import torchvision as tv
from torch import Tensor, nn, optim
from transformers import (
    ConvNextV2Config, ConvNextV2ForImageClassification, 
    SwinForImageClassification, SwinConfig
)
from mup import MuReadout, MuAdam, MuSGD, load_base_shapes, set_base_shapes
from schedulefree import AdamWScheduleFree

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
        *,
        num_features: int = 3,  # Unused
        pretrained: bool = False,
        base_shapes_path: str | None = None,
        **kwargs
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

        self.mup = base_shapes_path is not None
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
        opt_cls = MuSGD if self.mup else optim.SGD
        return opt_cls(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(self.norm(x))


class ConvNextProbe(Probe):
    def __init__(
            self,
            num_features: int, 
            num_classes: int = 2, 
            num_layers: int = 2,
            hidden_size: int = 2,
            device: str | torch.device | None = None, 
            dtype: torch.dtype | None = None,
            *,
            learning_rate: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            schedule_free: bool = False,
            base_shapes_path: str | None = None,
            **kwargs
        ):
        assert num_features == 3 * 32 * 32
        super().__init__(num_features, num_classes, device, dtype)

        self.learning_rate = learning_rate
        self.betas = betas
        self.schedule_free = schedule_free
        self.mup = base_shapes_path is not None
        
        depths = [1, 1, 3, 1]
        depths = [depth * num_layers for depth in depths]

        hidden_sizes = [hidden_size] + [hidden_size * 2 ** i for i in range(1, 4)]
        
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

        self.net = ConvNextV2ForImageClassification(cfg).to(device=device, dtype=dtype)

        # Configure MuP
        self.net.classifier = MuReadout(
            self.net.classifier.in_features,
            self.net.classifier.out_features,
            device=device,
            dtype=dtype,
            readout_zero_init=True
        )

        if base_shapes_path:
            base_shapes = load_base_shapes(base_shapes_path)
            set_base_shapes(self, base_shapes)

    
    def build_optimizer(self):
        opt_cls = AdamWScheduleFree if self.schedule_free else optim.AdamW
        if self.mup:
            return MuAdam(self.parameters(), opt_cls, lr=self.learning_rate, betas=self.betas)
        return opt_cls(self.parameters(), lr=self.learning_rate, betas=self.betas)

    def forward(self, x):
        return self.net(x).logits


class SwinProbe(Probe):
    def __init__(
            self,
            num_features: int, 
            num_classes: int = 2, 
            num_layers: int = 2,
            hidden_size: int = 2,
            device: str | torch.device | None = None, 
            dtype: torch.dtype | None = None,
            *,
            learning_rate: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            schedule_free: bool = False,
            base_shapes_path: str | None = None,
        ):
        assert num_features == 3 * 32 * 32
        super().__init__(num_features, num_classes, device, dtype)

        self.learning_rate = learning_rate
        self.betas = betas
        self.schedule_free = schedule_free
        self.mup = base_shapes_path is not None

        # depths=[1, 2, 1] seen in a gist somewhere
        depths = [1, 1, 2]
        depths = [depth * num_layers for depth in depths]

        # num_heads=[2, 2, 4] seen in a gist somewhere
        num_heads = [1, 1, 2]
        num_heads = [num_head * num_layers for num_head in num_heads]

        hidden_sizes = [num_heads[0] * hidden_size * 2**i for i in range(3)] 
        
        cfg = SwinConfig(
                image_size=32,
                num_channels=3,
                depths=depths,
                drop_path_rate=0.1,
                hidden_sizes=hidden_sizes,
                num_labels=num_classes,
                embed_dim=num_heads[0] * 4, # Can scale this and the hidden_sizes * 4 arbitrarily
                num_heads=num_heads,
                # The default of 4 x 4 patches shrinks the image too aggressively for
                # low-resolution images like CIFAR-10
                patch_size=2,
                window_size=2,
            )

        self.net = SwinForImageClassification(cfg).to(device=device, dtype=dtype)

        # Configure MuP
        self.net.classifier = MuReadout(
            self.net.classifier.in_features,
            self.net.classifier.out_features,
            device=device,
            dtype=dtype,
            readout_zero_init=True
        )
        
        if base_shapes_path:
            base_shapes = load_base_shapes(base_shapes_path)
            set_base_shapes(self, base_shapes)

    def build_optimizer(self):
        opt_cls = AdamWScheduleFree if self.schedule_free else optim.AdamW
        if self.mup:
            return MuAdam(self.parameters(), opt_cls, lr=self.learning_rate, betas=self.betas)
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=self.betas)

    def forward(self, x):
        return self.net(x).logits