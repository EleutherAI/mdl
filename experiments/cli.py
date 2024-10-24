from argparse import ArgumentParser
from itertools import pairwise
from typing import Callable, Sized
from pathlib import Path

import math
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as tm
import torchvision as tv
from concept_erasure import LeaceFitter, OracleEraser, OracleFitter, QuadraticFitter, LeaceEraser, QuadraticEraser
from torch.utils.data import Subset
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor, nn
from torch.utils.data import Dataset, random_split
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm
from pytorch_lightning.callbacks import Callback
import lovely_tensors as lt
from transformers import ConvNextV2ForImageClassification, ConvNextV2Config
from transformers import ViTConfig, ViTForImageClassification
lt.monkey_patch()

class Log2CheckpointCallback(Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.last_saved_step = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step
        if global_step == 0:
            return

        if self.is_power_of_two(global_step) and global_step > self.last_saved_step:
            filename: str = f"{self.filepath}/step={global_step:07d}.ckpt"
            trainer.save_checkpoint(filename)
            self.last_saved_step = global_step

    @staticmethod
    def is_power_of_two(n):
        return (n & (n-1) == 0) and n != 0

torch.set_default_tensor_type(torch.DoubleTensor)

class Mlp(pl.LightningModule):
    def __init__(self, k, h=128):
        super().__init__()
        self.save_hyperparameters()

        self.build_net()
        self.train_acc = tm.Accuracy("multiclass", num_classes=k)
        self.val_acc = tm.Accuracy("multiclass", num_classes=k)
        self.test_acc = tm.Accuracy("multiclass", num_classes=k)
    
    def build_net(self):
        k, h = self.hparams['k'], self.hparams['h']
        self.net = torch.nn.Sequential(
            torch.nn.Linear(32 * 32 * 3, h),
            torch.nn.ReLU(),
            torch.nn.Linear(h, h),
            torch.nn.ReLU(),
            torch.nn.Linear(h, k),
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
    
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)

        self.train_acc(y_hat, y)
        self.log(
            "train_acc", self.train_acc, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)

        self.val_acc(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)

        self.test_acc(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())


class ResMlp(Mlp):    
    def build_net(self):
        sizes = [3 * 32 * 32] + [self.hparams['h']] + [self.hparams['k']]

        self.net = nn.Sequential(
            *[
                MlpBlock(in_dim, out_dim, device=self.device, dtype=self.dtype)
                for in_dim, out_dim in pairwise(sizes)
            ]
        )
        # self.reset_parameters()
    
    # def reset_parameters(self):
    #     # ResNet initialization
    #     for m in self.net.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    
    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4
        )


class ResNet(Mlp):
    def build_net(self):
        self.net = tv.models.resnet18(pretrained=False, num_classes=self.hparams['k'])

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4
        )


class ViT(Mlp):
    def build_net(self):
        cfg = ViTConfig(
            image_size=32,
            num_channels=3,
            patch_size=1,
            num_labels=self.hparams['k'],
            hidden_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            intermediate_size=128,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
        )
        self.net = ViTForImageClassification(cfg)
    
    # def configure_optimizers(self):
    #     return torch.optim.SGD(
    #         self.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4
    #     )

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        return self.net(x).logits


class ConvNext(Mlp):
    def build_net(self):
        depths = [2, 2, 6, 2]
        hidden_sizes = [40, 80, 160, 320]

        cfg = ConvNextV2Config(
                image_size=32,
                num_channels=3,
                depths=depths,
                drop_path_rate=0.1,
                hidden_sizes=hidden_sizes,
                num_labels=self.hparams['k'],
                # The default of 4 x 4 patches shrinks the image too aggressively for
                # low-resolution images like CIFAR-10
                patch_size=1,
            )
        self.net = ConvNextV2ForImageClassification(cfg)
    
    # def configure_optimizers(self):
    #     return torch.optim.SGD(
    #         self.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4
    #     )

    def forward(self, x):
        x = x.reshape(-1, 3, 32, 32)
        return self.net(x).logits


class MlpBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        self.linear1 = nn.Linear(
            in_features, out_features, bias=False, device=device, dtype=dtype
        )
        self.linear2 = nn.Linear(
            out_features, out_features, bias=False, device=device, dtype=dtype
        )
        self.bn1 = nn.BatchNorm1d(out_features, device=device, dtype=dtype)
        self.bn2 = nn.BatchNorm1d(out_features, device=device, dtype=dtype)
        self.downsample = (
            nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
            if in_features != out_features
            else None
        )

    def forward(self, x):
        identity = x

        out = self.linear1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)

        out = self.linear2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = nn.functional.relu(out)

        return out


class LeacedDataset(Dataset):
    """Wrapper for a dataset of (X, Z) pairs that erases Z from X"""
    def __init__(
        self,
        inner: Dataset[tuple[Tensor, ...]],
        eraser: Callable,
        transform: Callable[[Tensor], Tensor] = lambda x: x,
    ):
        # Pylance actually keeps track of the intersection type
        assert isinstance(inner, Sized), "inner dataset must be sized"
        assert len(inner) > 0, "inner dataset must be non-empty"

        self.dataset = inner
        self.eraser = eraser
        self.transform = transform

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x, z = self.dataset[idx]

        # Erase BEFORE transforming
        if isinstance(self.eraser, LeaceEraser):
            x_erased = self.eraser(x.flatten())
            x = x_erased.reshape(x.shape)
        elif isinstance(self.eraser, OracleEraser):
            x_erased = self.eraser(x.flatten(), torch.tensor(z).unsqueeze(0))
            x = x_erased.reshape(x.shape)
        else:
            z_tensor = torch.tensor(z)
            if z_tensor.ndim == 0:
                z_tensor = z_tensor.unsqueeze(0)
            x = self.eraser(x.unsqueeze(0), z_tensor)
        return self.transform(x), z

        #             z_tensor = torch.tensor(data=z).type_as(x).to(torch.int64)
        #     if z_tensor.ndim == 0:
        #         z_tensor = z_tensor.unsqueeze(0)
        #     x = self.eraser(x.unsqueeze(0), z_tensor)
        # return self.transform(x), z

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("--eraser", type=str, choices=("none", "leace", "oleace", "qleace"))
    parser.add_argument("--net", type=str, choices=("mlp", "resmlp", "resnet", "convnext", "vit"))
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Split the "train" set into train and validation
    nontest = CIFAR10(
        "/home/lucia/cifar10", download=True, transform=tv.transforms.ToTensor()
    )
    train, val = random_split(nontest, [0.9, 0.1])

    # Test set is entirely separate
    test = CIFAR10(
        "/home/lucia/cifar10-test",
        download=True,
        train=False,
        transform=tv.transforms.ToTensor(),
    )
    k = 10  # Number of classes
    final = nn.Identity() if args.net == "resnet" else nn.Flatten(0)
    train_trf = tv.transforms.Compose([
        # tv.transforms.RandomHorizontalFlip(), # Increases norm of class covariance diff
        # tv.transforms.RandomCrop(32, padding=4),
        final,
    ])
    if args.eraser != "none":
        cls = {
            "leace": LeaceFitter,
            "oleace": OracleFitter,
            "qleace": QuadraticFitter,
        }[args.eraser]

        fitter = cls(3 * 32 * 32, k, dtype=torch.float64, device=device, shrinkage=False)
        state_path = Path("erasers_cache") / f"cifar10_state.pth"
        state_path.parent.mkdir(exist_ok=True)
        state = {} if not state_path.exists() else torch.load(state_path)
        
        if args.eraser in state:
            eraser = state[args.eraser]
        else:
            for x, y in tqdm(train):
                y = torch.as_tensor(y).view(1)
                if args.eraser != "qleace":
                    y = F.one_hot(y, k)

                fitter.update(x.view(1, -1).to(device), y.to(device))

            eraser = fitter.eraser.to("cpu")

            state[args.eraser] = eraser
            torch.save(state, state_path)
    else:
        eraser = lambda x, y: x
    train = LeacedDataset(train, eraser, transform=train_trf)
    val = LeacedDataset(val, eraser, transform=final)
    test = LeacedDataset(test, eraser, transform=final)

    if args.debug:
        train = Subset(train, range(10_000))
    
    # Create the data module
    dm = pl.LightningDataModule.from_datasets(train, val, test, batch_size=128, num_workers=8)
    
    model_cls = {
        "mlp": Mlp,
        "resmlp": ResMlp,
        "resnet": ResNet,
        "vit": ViT,
        "convnext": ConvNext,
    }[args.net]
    model = model_cls(k, h=args.width)

    trainer = pl.Trainer(
        callbacks=[
            # EarlyStopping(monitor="val_loss", patience=5),
            Log2CheckpointCallback(filepath=f'checkpoints-{args.width}')
        ],
        logger=WandbLogger(name=args.name, project="mdl", entity="eleutherai") if not "test" in args.name else None,
        max_epochs=200,
        precision=64
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)