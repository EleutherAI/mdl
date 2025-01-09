from pathlib import Path
from typing import TypeVar, Type, Any, cast, Literal
from dataclasses import dataclass
from simple_parsing import ArgumentParser
import lovely_tensors as lt
import os
import pickle
import json

import wandb
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import torchvision.utils as vutils
from torch import Tensor
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2.functional import to_tensor
from datasets import load_dataset, DatasetDict, load_from_disk
from mup import make_base_shapes
from concept_erasure.quadratic import QuadraticFitter
from concept_erasure.leace import LeaceFitter
from concept_erasure.alf_qleace import AlfQLeaceFitter

from mdl.lenet_probe import LeNetProbe
from mdl.mlp_probe import ResMlpProbe, MlpProbe, LinearProbe
from mdl.sweep import Sweep
from mdl.vision_probe import ConvNextProbe, VisionProbe, SwinProbe
from mdl.resnet_probe import ResNetProbe


@dataclass
class Args:
    # General settings
    name: str = ""
    out: str = "results"

    # Dataset options
    dataset: Literal["cifar10", "mnist", "cifarnet", "fake-cifar10", "fake-cifarnet"] = "cifar10"
    eraser: Literal["control", "leace", "oleace", "qleace", "alf_qleace"] = "control"
    method: Literal["leace", "orth", "none"] = "leace"
    shrinkage: bool = False
    normalize: bool = False
    post_erase_normalize: bool = False
    alf_qleace_target: float = 0.9

    # Model architecture
    net: Literal["mlp", "resmlp", "resnet", "convnext", "linear", "vision", "swin", "lenet"] = (
        "mlp"
    )
    act: Literal["relu", "gelu", "swiglu"] = "relu"

    # Model dimensions
    width: int = 128
    depth: int = 2
    mup_width: int | None = None  # Width of the base model used to tune the initial LR
    mup_depth: int | None = None  # Depth of the base model used to tune the initial LR

    # Training parameters
    lr: float = 1e-3
    b1: float = 0.9
    num_seeds: int = 5
    max_epochs: int = 30_000
    early_stop_epochs: int = 100
    schedulefree: bool = False

    # Runtime flags
    debug: bool = False
    nocache: bool = False
    nowritecache: bool = False
    save: bool = False
    overwrite: bool = False
    trial: bool = False  # Run a single trial with all data
    wandb_run_id: str | None = None


T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)


def get_cifarnet():
    cache_dir = 'data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "cifar_processed.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def map_fn(ex):
        return {"input_ids": to_tensor(ex["img"]), "label": ex["label"]}

    data = assert_type(DatasetDict, load_dataset("EleutherAI/cifarnet"))

    nontest = data["train"].map(function=map_fn)
    nontest.set_format(type="torch", columns=["input_ids", "label"])

    X = assert_type(Tensor, nontest["input_ids"])
    Y = assert_type(Tensor, nontest["label"])

    # Shuffle deterministically
    rng = torch.Generator(device=X.device).manual_seed(42)
    perm = torch.randperm(len(X), generator=rng, device=X.device)
    X, Y = X[perm], Y[perm]

    # Get number of classes
    k = int(Y.max()) + 1

    # Split train and validation
    val_size = 1024
    X_train, X_val = X[:-val_size], X[-val_size:]
    Y_train, Y_val = Y[:-val_size], Y[-val_size:]

    with open(cache_path, "wb") as f:
        pickle.dump((X_train, Y_train, X_val, Y_val, k, X, Y), f)

    return X_train, Y_train, X_val, Y_val, k, X, Y


def get_cifar10(device: str | torch.device):
    nontest = CIFAR10("data/cache/cifar10", download=True)
    images, labels = zip(*nontest)

    X = torch.stack(list(map(to_tensor, images))).to(device)
    Y = torch.tensor(labels).to(device)

    # Shuffle deterministically
    rng = torch.Generator(device=X.device).manual_seed(42)
    perm = torch.randperm(len(X), generator=rng, device=X.device)
    X, Y = X[perm], Y[perm]

    k = int(Y.max()) + 1

    # Split train and validation
    val_size = 1024
    X_train, X_val = X[:-val_size], X[-val_size:]
    Y_train, Y_val = Y[:-val_size], Y[-val_size:]

    return X_train, Y_train, X_val, Y_val, k, X, Y


def get_fake_cifarnet():
    train = load_dataset("EleutherAI/erased-cifarnet", split="train")
    X = torch.stack([to_tensor(img) for img in train["image"]])
    Y = torch.tensor(train["label"])

    # Shuffle deterministically
    rng = torch.Generator(device=X.device).manual_seed(42)
    perm = torch.randperm(len(X), generator=rng, device=X.device)
    X, Y = X[perm], Y[perm]

    k = int(Y.max()) + 1

    # Split train and validation
    val_size = 1024
    X_train, X_val = X[:-val_size], X[-val_size:]
    Y_train, Y_val = Y[:-val_size], Y[-val_size:]

    return X_train, Y_train, X_val, Y_val, k, X, Y


def get_fake_cifar10():
    train = load_dataset("EleutherAI/erased-cifar10", split="train")
    X = torch.stack([to_tensor(img) for img in train["image"]])
    Y = torch.tensor(train["label"])

    # Shuffle deterministically
    rng = torch.Generator(device=X.device).manual_seed(42)
    perm = torch.randperm(len(X), generator=rng, device=X.device)
    X, Y = X[perm], Y[perm]

    k = int(Y.max()) + 1

    # Split train and validation
    val_size = 1024
    X_train, X_val = X[:-val_size], X[-val_size:]
    Y_train, Y_val = Y[:-val_size], Y[-val_size:]

    return X_train, Y_train, X_val, Y_val, k, X, Y

def normalize_dataset(
    X: Tensor, X_train: Tensor, X_val: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    eps = torch.finfo(X_train.dtype).eps
    X_flat = X_train.reshape(X_train.shape[0], -1)

    mean = X_flat.mean(dim=0, keepdim=True)
    scaling = torch.std(X_flat, dim=0) + eps

    def normalize_data(data: Tensor) -> Tensor:
        data_flat = data.reshape(data.shape[0], -1)
        data_centered = data_flat - mean
        data_normalized = data_centered / scaling
        return data_normalized.reshape(data.shape)

    X = normalize_data(X)
    X_train = normalize_data(X_train)
    X_val = normalize_data(X_val)

    return X, X_train, X_val


class IdentityEraser:
    def __init__(self):
        pass

    def __call__(self, x: Tensor) -> Tensor:
        return x

    def to(self, device: str | torch.device) -> "IdentityEraser":
        return self


def load_eraser(
    args: Args,
    device: str | torch.device,
    fit_device: str | torch.device,
    dtype: torch.dtype,
    method: str,
    shrinkage: bool,
    alf_qleace_target: float | None,
    X_train: Tensor,
    Y_train: Tensor,
    nowritecache: bool,
):
    state_path = Path("data") / "erasers_cache" / f"{args.dataset}_{dtype}_state.pth"
    state_path.parent.mkdir(exist_ok=True)
    state = {} if not state_path.exists() else torch.load(state_path)

    if args.eraser not in state or args.nocache:
        if args.eraser == "control":
            state[args.eraser] = IdentityEraser()
        else:
            cls = {
                "leace": LeaceFitter,
                "qleace": QuadraticFitter,
                "alf_qleace": AlfQLeaceFitter,
            }[args.eraser]

            if args.eraser == "leace":
                fitter = cls(num_features, k, dtype=dtype, device=device, method=method, shrinkage=shrinkage)
            elif args.eraser == "alf_qleace":
                fitter = cls(num_features, k, dtype=dtype, device=device, method=method, shrinkage=shrinkage, target_erasure=alf_qleace_target)
            else:
                fitter = cls(num_features, k, dtype=dtype, device=device)

            Y_tensor = (
                F.one_hot(Y_train, k)
                if args.eraser != "qleace"
                else Y_train
            ).to(device)
            X_tensor = X_train.flatten(1).to(device).to(dtype)
            fitter.update(X_tensor, Y_tensor)
            fitter = fitter.to(fit_device)
            
            state[args.eraser] = fitter.eraser
        if not nowritecache:
            torch.save(state, state_path)

    return state[args.eraser]



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lt.monkey_patch()
    Path("data").mkdir(exist_ok=True)
    dtype = torch.float32

    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    # Initialize directories
    mup_path = Path("data/mup")
    mup_path.mkdir(exist_ok=True)

    data_path = Path(
        f"/mnt/ssd-1/lucia/{args.out}"
        if not args.debug
        else f"/mnt/ssd-1/lucia/debug-{args.out}"
    )
    data_path.mkdir(exist_ok=True, parents=True)

    seed_path = Path(f"data/{args.out}-seeds")
    seed_path.mkdir(exist_ok=True, parents=True)

    # Get dataset
    (X_train, Y_train, X_val, Y_val, k, X, Y) = {
        "cifar10": get_cifar10(device),
        "cifarnet": get_cifarnet(),
        "fake-cifar10": get_fake_cifar10(),
        "fake-cifarnet": get_fake_cifarnet(),
    }[args.dataset]

    if args.normalize:
        assert args.eraser == "control"
        X, X_train, X_val, = normalize_dataset(X, X_train, X_val)

    num_features = X.shape[1] * X.shape[2] * X.shape[3]

    # Fit eraser on dataset and save to cache
    state_path = Path("data") / "erasers_cache" / f"{args.dataset}_state.pth"
    state_path.parent.mkdir(exist_ok=True)
    state = {} if not state_path.exists() else torch.load(state_path)

    eraser = load_eraser(
        args,
        "cpu", # device if args.eraser != "leace" else "cpu",
        device if args.dataset != "cifarnet" else "cpu",
        dtype if args.eraser != "leace" else torch.float64,
        args.method,
        args.shrinkage,
        args.alf_qleace_target,
        X_train,
        Y_train,
        args.nowritecache,
    ).to(device)

    if args.eraser != "control":
        (Path('data') / 'saved_images').mkdir(exist_ok=True)
        images = eraser.to("cpu")(X_train[:5].flatten(1)).reshape_as(X_train[:5])
        [vutils.save_image(images[i], f'saved_images/image_{i}_90%_{args.dataset}_{args.eraser}.png', normalize=True) for i in range(5)]

    image_size = X.shape[-1]

    model_cls = {
        "mlp": MlpProbe,
        "resmlp": ResMlpProbe,
        "resnet": ResNetProbe,
        "convnext": ConvNextProbe,
        "linear": LinearProbe,
        "vision": VisionProbe,
        "swin": SwinProbe,
        "lenet": LeNetProbe,
    }[args.net]

    probe_kwargs = {}
    if args.net == "lenet":
        with open(f'data/lenet_configs_{image_size}.json', 'r') as f: 
            lenet_params = json.load(f)[f"{args.depth}_{args.width}"]

        probe_kwargs['conv_hidden_sizes'] = lenet_params['conv_hidden_sizes']
        probe_kwargs['fc_hidden_sizes'] = lenet_params['fc_hidden_sizes']

    # Prepare hyperparameter scaling factors and base shapes
    base_model = model_cls(
        num_classes=k,
        num_features=num_features,
        num_layers=args.depth,  # mup depth unsupported
        hidden_size=args.mup_width if args.mup_width else args.width,
        **probe_kwargs
    )
    delta_model = model_cls(
        num_classes=k,
        num_features=num_features,
        num_layers=args.depth,
        hidden_size=args.width,
        **probe_kwargs
    )

    base_shapes_path = (
        mup_path / f"mup-{args.net}-{args.width}-{args.depth}-{args.mup_width}.bsh"
    )
    make_base_shapes(base_model, delta_model, savefile=str(base_shapes_path))

    if args.mup_depth:
        if model_cls == MlpProbe:
            # Depth-wise scaling for MLPs from https://arxiv.org/pdf/2305.07810
            args.lr = args.lr * (args.mup_depth / args.depth) ** (3 / 2)
        else:
            # More conservative scaling for vision models
            args.lr = args.lr * (args.mup_depth / args.depth) ** (1 / 2)

    # Define flattening, augmentations, and eraser transform
    flatten = {
        "mlp": True,
        "resmlp": True,
        "resnet": False,
        "convnext": False,
        "linear": True,
        "vision": False,
        "swin": False,
        "lenet": False,
    }

    padding = round(image_size * 0.125)

    augment = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.view(-1, X.shape[1], X.shape[2], X.shape[3])),
            transforms.RandomCrop(image_size, padding),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x.flatten(1)),
        ]
        if flatten[args.net]
        else [
            transforms.RandomCrop(image_size, padding),
            transforms.RandomHorizontalFlip(),
        ]
    )

    # If LEACE, scale normalization can use the covariance of the vanilla data
    # If ALF-QLEACE, scale normalization must use the covariance of the erased data
    # I will gather these and hard code
    if args.post_erase_normalize:
        if args.eraser == "leace" or args.eraser == "control":
            std = X_train.flatten(1).std(dim=0).to(device)
        elif args.eraser == "alf_qleace":
            std = eraser.to("cpu")(X_train.flatten(1)).std(dim=0).to(device)
        else:
            print("Not implemented")
    else:
        std = torch.tensor(1.0).to(device)

    def erase_transform(x: Tensor, y: Tensor):
        x_erased = (
            eraser(x.flatten(1), y) if args.eraser == "qleace" else eraser(x.flatten(1))
        )

        if args.post_erase_normalize:
            x_erased = x_erased / std

        return x_erased if flatten[args.net] else x_erased.reshape_as(x)

    if args.post_erase_normalize:
        X_val = X_val.flatten(1) / X_train.flatten(1).std(dim=0).to(X_val.device)

    # Collect MDL data
    probe_kwargs = dict(
        num_layers=args.depth,
        hidden_size=args.width,
        learning_rate=args.lr,
        schedule_free=args.schedulefree,
        betas=(args.b1, 0.999),
        base_shapes_path=base_shapes_path,
    )
    if model_cls == LeNetProbe:
        probe_kwargs['conv_hidden_sizes'] = lenet_params['conv_hidden_sizes']
        probe_kwargs['fc_hidden_sizes'] = lenet_params['fc_hidden_sizes']
    if model_cls == MlpProbe:
        probe_kwargs["activation"] = args.act
    if args.trial:
        # These are otherwise passed into the sweep
        probe_kwargs["num_classes"] = k
        probe_kwargs["num_features"] = num_features
        probe_kwargs["dtype"] = dtype
        probe_kwargs["device"] = device

    results = []
    for seed in range(args.num_seeds):
        wandb_name = f'{args.eraser} {args.name} w={args.width} d={args.depth} s={seed} {args.net} act={args.act} lr={args.lr:.7f} b1={args.b1} n={args.normalize} es={args.early_stop_epochs} d={args.dataset}'

        seed_file = (
            seed_path
            / f"{args.net}_{args.act}_h={args.width}_d={args.depth}_{args.eraser}_{args.name}_{seed}_{args.dataset}.pth"
        )
        if not args.overwrite and seed_file.exists():
            results.append(torch.load(seed_file))
            continue

        run = (
            wandb.init(
                project="mdl",
                id=args.wandb_run_id if args.wandb_run_id else None,
                entity="eleutherai",
                name=wandb_name,
                config={"eraser": args.eraser, **vars(args)},
                reinit=args.wandb_run_id is None,
            )
            if not args.debug
            else None
        )

        if args.trial:
            # Run a single trial with a large dataset
            model_cls(**probe_kwargs).fit(
                X_train[:len(X_train)//2].to(device),
                Y_train[:len(Y_train)//2].to(device),
                x_val=X_val.to(device),
                y_val=Y_val.to(device),
                seed=0,
                transform=erase_transform,
                augment=augment,
                max_epochs=args.max_epochs,
                early_stop_epochs=args.max_epochs,
                logger=run,
            )
            wandb.finish()
            exit(0)

        sweep = Sweep(
            num_features,
            k,
            device=device,
            dtype=dtype,
            num_chunks=10,
            logger=run,
            probe_cls=model_cls,
            ckpt_every=10,
            probe_kwargs=probe_kwargs,
        )
        results.append(
            sweep.run(
                X,
                Y,
                seed=seed,
                transform=erase_transform,
                augment=augment,
                reduce_lr_on_plateau=False,
                max_epochs=args.max_epochs,
                early_stop_epochs=args.early_stop_epochs,
            )
        )

        if not args.debug:
            torch.save(results, seed_file)

        try:
            wandb.finish()
        except Exception as e:
            print("Caught exception: ", e)
            pass

    # Save results
    torch.save(
        results,
        data_path
        / f"{args.net}_{args.act}_h={args.width}_d={args.depth}_{args.eraser}_{args.name}_{args.dataset}.pth",
    )
