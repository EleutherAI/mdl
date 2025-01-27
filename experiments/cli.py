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
from torchvision.transforms.v2.functional import to_dtype, to_image
from datasets import load_dataset, DatasetDict, load_from_disk
from mup import make_base_shapes
from concept_erasure.quadratic import QuadraticFitter
from concept_erasure.leace import LeaceFitter
from concept_erasure.alf_qleace import AlfQLeaceFitter
from concept_erasure.re import RandomEraser

from mdl.lenet_probe import LeNetProbe
from mdl.mlp_probe import ResMlpProbe, MlpProbe, LinearProbe
from mdl.sweep import Sweep
from mdl.vision_probe import ConvNextProbe, VisionProbe, SwinProbe
from mdl.resnet_probe import ResNetProbe

torch.set_float32_matmul_precision('high')

@dataclass
class Args:
    # General settings
    name: str = ""
    out: str = "results"

    # Dataset options
    dataset: Literal["cifar10", "cifarnet", "fake-cifar10", 
                     "fake-cifarnet", "svhn", "fake-svhn", "fake-leace-cifar10",
                     "fake-leace-cifarnet", "fake-leace-svhn",
                    ] = "cifar10"
    eraser: Literal["control", "leace", "oleace", "qleace", "alf_qleace", "random"] = "control"
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

    # Model dimensions for simple models
    width: int = 128
    depth: int = 2
    mup_width: int | None = None  # Width of the base model used to tune the initial LR
    mup_depth: int | None = None  # Depth of the base model used to tune the initial LR
    
    # Model dimensions for SOTA vision architectures
    arch: Literal["atto", "femto", "pico", "nano", "tiny"] = "atto"
    mup_arch: Literal["atto", "femto", "pico", "nano", "tiny"] = "atto"

    # Training parameters
    lr: float = 1e-3
    b1: float = 0.9
    num_seeds: int = 5
    max_epochs: int = 30_000
    early_stop_epochs: int = 100

    # Runtime flags
    debug: bool = False
    nocache: bool = False
    nowritecache: bool = False
    save: bool = False
    overwrite: bool = False
    trial: bool = False  # Run a single trial with all data


T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)


def get_cifarnet(shuffle=True):
    cache_dir = 'data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"cifar_processed{'_unshuffled' if not shuffle else ''}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)


    def map_fn(ex):
        return {
            "input_ids": to_dtype(to_image(ex["img"]), dtype=torch.float32, scale=True), 
            "label": ex["label"]
        }

    data = assert_type(DatasetDict, load_dataset("EleutherAI/cifarnet"))

    nontest = data["train"].map(function=map_fn)
    nontest.set_format(type="torch", columns=["input_ids", "label"])

    X = assert_type(Tensor, nontest["input_ids"])
    Y = assert_type(Tensor, nontest["label"])

    if shuffle:
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


def get_cifar10(device: str | torch.device = 'cuda', shuffle=True):
    nontest = CIFAR10("data/cache/cifar10", download=True)
    images, labels = zip(*nontest)

   
    X = torch.stack([
        to_dtype(to_image(item), dtype=torch.float32, scale=True) 
        for item in images
    ]).to(device)

    Y = torch.tensor(labels).to(device)

    # Shuffle deterministically
    if shuffle:
        rng = torch.Generator(device=X.device).manual_seed(42)
        perm = torch.randperm(len(X), generator=rng, device=X.device)
        X, Y = X[perm], Y[perm]

    k = int(Y.max()) + 1

    # Split train and validation
    val_size = 1024
    X_train, X_val = X[:-val_size], X[-val_size:]
    Y_train, Y_val = Y[:-val_size], Y[-val_size:]

    return X_train, Y_train, X_val, Y_val, k, X, Y


def get_fake_leace_cifar10(shuffle=True):
    train = load_from_disk("data/leace-and-quadratic-iterative-erasure-cifar10/train")
    val = load_from_disk("data/leace-and-quadratic-iterative-erasure-cifar10/val")
    train.set_format(type="torch", columns=["image", "label"])
    val.set_format(type="torch", columns=["image", "label"])

    X_train = train["image"]
    Y_train = train["label"]
    X_val = val["image"]
    Y_val = val["label"]

    X = X_train
    Y = Y_train
    k = int(Y_train.max()) + 1

    if shuffle:
        rng = torch.Generator(device=X_train.device).manual_seed(42)
        perm = torch.randperm(len(X_train), generator=rng, device=X_train.device)
        X_train, Y_train = X_train[perm], Y_train[perm]
        perm = torch.randperm(len(X_val), generator=rng, device=X_val.device)
        X_val, Y_val = X_val[perm], Y_val[perm]

    return X_train, Y_train, X_val, Y_val, k, X, Y

def get_fake_leace_cifarnet(shuffle=True):
    train = load_from_disk("data/leace-and-quadratic-iterative-erasure-cifarnet/train")
    val = load_from_disk("data/leace-and-quadratic-iterative-erasure-cifarnet/val")
    train.set_format(type="torch", columns=["image", "label"])
    val.set_format(type="torch", columns=["image", "label"])

    X_train = train["image"]
    Y_train = train["label"]
    X_val = val["image"]
    Y_val = val["label"]

    X = X_train
    Y = Y_train
    k = int(Y_train.max()) + 1

    if shuffle:
        rng = torch.Generator(device=X_train.device).manual_seed(42)
        perm = torch.randperm(len(X_train), generator=rng, device=X_train.device)
        X_train, Y_train = X_train[perm], Y_train[perm]
        perm = torch.randperm(len(X_val), generator=rng, device=X_val.device)
        X_val, Y_val = X_val[perm], Y_val[perm]

    return X_train, Y_train, X_val, Y_val, k, X, Y

def get_fake_leace_svhn(shuffle=True):
    train = load_from_disk("data/leace-and-quadratic-iterative-erasure-svhn/train")
    val = load_from_disk("data/leace-and-quadratic-iterative-erasure-svhn/val")
    train.set_format(type="torch", columns=["image", "label"])
    val.set_format(type="torch", columns=["image", "label"])

    X_train = train["image"]
    Y_train = train["label"]
    X_val = val["image"]
    Y_val = val["label"]

    X = X_train
    Y = Y_train
    k = int(Y_train.max()) + 1

    if shuffle:
        rng = torch.Generator(device=X_train.device).manual_seed(42)
        perm = torch.randperm(len(X_train), generator=rng, device=X_train.device)
        X_train, Y_train = X_train[perm], Y_train[perm]
        perm = torch.randperm(len(X_val), generator=rng, device=X_val.device)
        X_val, Y_val = X_val[perm], Y_val[perm]

    return X_train, Y_train, X_val, Y_val, k, X, Y



def get_fake_cifarnet(shuffle=True):
    train = load_dataset("EleutherAI/erased-cifarnet", split="train")
    X = torch.stack([to_dtype(to_image(img), dtype=torch.float32, scale=True) for img in train["image"]]) # type: ignore
    Y = torch.tensor(train["label"])

    if shuffle:
        rng = torch.Generator(device=X.device).manual_seed(42)
        perm = torch.randperm(len(X), generator=rng, device=X.device)
        X, Y = X[perm], Y[perm]

    k = int(Y.max()) + 1

    # Split train and validation
    val_size = 1024
    X_train, X_val = X[:-val_size], X[-val_size:]
    Y_train, Y_val = Y[:-val_size], Y[-val_size:]

    return X_train, Y_train, X_val, Y_val, k, X, Y


def get_fake_cifar10(shuffle=True):
    train = load_dataset("EleutherAI/erased-cifar10", split="train")
    X = torch.stack([to_dtype(to_image(img), dtype=torch.float32, scale=True) for img in train["image"]])
    Y = torch.tensor(train["label"])

    if shuffle:
        rng = torch.Generator(device=X.device).manual_seed(42)
        perm = torch.randperm(len(X), generator=rng, device=X.device)
        X, Y = X[perm], Y[perm]

    k = int(Y.max()) + 1

    # Split train and validation
    val_size = 1024
    X_train, X_val = X[:-val_size], X[-val_size:]
    Y_train, Y_val = Y[:-val_size], Y[-val_size:]

    return X_train, Y_train, X_val, Y_val, k, X, Y


def get_svhn(device, shuffle=True):
    data = load_dataset("ufldl-stanford/svhn", 'cropped_digits', split='train')
    X = torch.stack([to_dtype(to_image(img), dtype=torch.float32, scale=True) for img in data["image"]])
    Y = torch.tensor(data["label"])

    if shuffle:
        rng = torch.Generator(device=X.device).manual_seed(42)
        perm = torch.randperm(len(X), generator=rng, device=X.device)
        X, Y = X[perm], Y[perm]

    k = int(Y.max()) + 1

    # Split train and validation
    val_size = 1024
    X_train, X_val = X[:-val_size], X[-val_size:]
    Y_train, Y_val = Y[:-val_size], Y[-val_size:]

    return X_train, Y_train, X_val, Y_val, k, X, Y


def get_fake_svhn(shuffle=True):
    data = load_dataset("EleutherAI/erased-svhn", split="train")
    X = torch.stack([to_dtype(to_image(img), dtype=torch.float32, scale=True) for img in data["image"]])
    Y = torch.tensor(data["label"])

    if shuffle:
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


def get_cache_key(dataset_str, eraser_str, dtype, method, shrinkage, alf_qleace_target, random_erase_dims):
    if eraser_str == 'alf_qleace':
        return f"{eraser_str}_{dataset_str}_{dtype}_{method}_{shrinkage}_{alf_qleace_target}"
    elif eraser_str == 'leace':
        return f"{eraser_str}_{dataset_str}_{dtype}_{method}_{shrinkage}"
    elif eraser_str == 'qleace':
        return f"{eraser_str}_{dataset_str}_{dtype}"
    elif eraser_str == 'control':
        return f"{eraser_str}"
    elif eraser_str == 'random':
        return f"{eraser_str}_{dataset_str}_{random_erase_dims}"
    else:
        raise ValueError(f"Unknown eraser: {eraser_str}")
        

def load_eraser(
    eraser_str: str,
    dataset_str: str,
    dtype: torch.dtype,
    method: str,
    shrinkage: bool,
    alf_qleace_target: float | None,
    X_train: Tensor,
    Y_train: Tensor,
    num_features: int,
    k: int,
    nowritecache: bool,
    nocache: bool = False,
    device: str | torch.device = "cpu",
    fit_device: str | torch.device = "cpu",
    random_erase_dims=300
):
    state_path = Path("data") / "erasers_cache" / "state.pth"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {} if not state_path.exists() else torch.load(state_path, weights_only=False)

    cache_key = get_cache_key(dataset_str, eraser_str, dtype, method, shrinkage, alf_qleace_target, random_erase_dims)

    if cache_key not in state or nocache:
        if eraser_str == "control":
            state[cache_key] = IdentityEraser()
        elif eraser_str == "random":
            state[cache_key] = RandomEraser(X_train.flatten(1).shape[1], erase_dims=random_erase_dims)
        else:
            if eraser_str == "leace":
                fitter = LeaceFitter(num_features, k, dtype=dtype, device=device, method=method, shrinkage=shrinkage)
            elif eraser_str == "alf_qleace":
                fitter = AlfQLeaceFitter(num_features, k, dtype=dtype, device=device, method=method, shrinkage=shrinkage, target_erasure=alf_qleace_target)
            else:
                fitter = QuadraticFitter(num_features, k, dtype=dtype, device=device)

            Y_tensor = (
                F.one_hot(Y_train, k)
                if eraser_str != "qleace"
                else Y_train
            ).to(device)
            X_tensor = X_train.flatten(1).to(device).to(dtype)
            fitter.update(X_tensor, Y_tensor)
            fitter = fitter.to(fit_device)
            
            state[cache_key] = fitter.eraser

        if not nowritecache:
            torch.save(state, state_path)

    return state[cache_key]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lt.monkey_patch()
    Path("data").mkdir(exist_ok=True)
    dtype = torch.bfloat16

    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    # Initialize directories
    mup_path = Path("data/mup")
    mup_path.mkdir(exist_ok=True)

    data_path = Path(
        f"{args.out}"
        if not args.debug
        else f"debug-{args.out}"
    )
    data_path.mkdir(exist_ok=True, parents=True)

    seed_path = Path(f"{args.out}-seeds")
    seed_path.mkdir(exist_ok=True, parents=True)

    # Get dataset
    (X_train, Y_train, X_val, Y_val, k, X, Y) = {
        "cifar10": get_cifar10,
        "cifarnet": get_cifarnet,
        # "fake-cifar10": get_fake_cifar10,
        # "fake-cifarnet": get_fake_cifarnet,
        "svhn": get_svhn,
        # "fake-svhn": get_fake_svhn,
        "fake-leace-cifar10": get_fake_leace_cifar10,
        "fake-leace-cifarnet": get_fake_leace_cifarnet,
        "fake-leace-svhn": get_fake_leace_svhn,
    }[args.dataset]()
    X_train = X_train.to(dtype)
    X_val = X_val.to(dtype)
    X = X.to(dtype)

    if args.normalize:
        assert args.eraser == "control"
        X, X_train, X_val, = normalize_dataset(X, X_train, X_val)

    num_features = X.shape[1] * X.shape[2] * X.shape[3]
    
    # Get eraser
    eraser = load_eraser(
        args.eraser,
        args.dataset,
        dtype if args.eraser != "leace" else torch.float64,
        args.method,
        args.shrinkage,
        args.alf_qleace_target,
        X_train,
        Y_train,
        num_features,
        k,
        args.nowritecache,
        args.nocache,
        "cpu",
        device if args.dataset != "cifarnet" else "cpu",
    ).to(device)

    # Get model
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
        arch=args.mup_arch if args.mup_arch else args.arch,
        **probe_kwargs
    )
    delta_model = model_cls(
        num_classes=k,
        num_features=num_features,
        num_layers=args.depth,
        hidden_size=args.width,
        arch=args.arch,
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
    }[args.net]

    padding = round(image_size * 0.125)

    augment = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.view(-1, X.shape[1], X.shape[2], X.shape[3])),
            transforms.RandomCrop(image_size, padding),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x.flatten(1)),
        ]
        if flatten
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

        return x_erased if flatten else x_erased.reshape_as(x)

    if args.post_erase_normalize:
        X_val = X_val.flatten(1) / X_train.flatten(1).std(dim=0).to(X_val.device)

    # Collect MDL data
    # TODO this can probably be cleaned up
    probe_kwargs = dict(
        num_layers=args.depth,
        hidden_size=args.width,
        learning_rate=args.lr,
        schedule_free=True,
        betas=(args.b1, 0.999),
        base_shapes_path=base_shapes_path,
    )
    if model_cls == LeNetProbe:
        probe_kwargs['conv_hidden_sizes'] = lenet_params['conv_hidden_sizes']
        probe_kwargs['fc_hidden_sizes'] = lenet_params['fc_hidden_sizes']
    if model_cls == MlpProbe:
        probe_kwargs["activation"] = args.act
    if model_cls == SwinProbe or model_cls == ConvNextProbe:
        probe_kwargs["arch"] = args.arch
    if args.trial:
        # These are otherwise passed into the sweep
        probe_kwargs["num_classes"] = k
        probe_kwargs["num_features"] = num_features
        probe_kwargs["dtype"] = dtype
        probe_kwargs["device"] = device

    results = []

    size_str = f'a={args.arch}' if args.net == "convnext" or args.net == "swin" else f'h={args.width}_d={args.depth}'

    for seed in range(args.num_seeds):
        wandb_name = f'{args.eraser} {args.name} {size_str.replace("_", " ")} s={seed} {args.net} act={args.act} lr={args.lr:.7f} b1={args.b1} n={args.normalize} es={args.early_stop_epochs} d={args.dataset}'

        seed_file = (
            seed_path
            / f"{args.net}_{args.act}_{size_str}_{args.eraser}_{args.name}_{seed}_{args.dataset}.pth"
        )
        if seed_file.exists(): # not args.overwrite and
            try:
                results.append(torch.load(seed_file))
                continue
            except:
                pass

        run = (
            wandb.init(
                project="mdl",
                id=None,
                entity="eleutherai",
                name=wandb_name,
                config={"eraser": args.eraser, **vars(args)},
                reinit=True,
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
            ckpt_every=None,
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
            try:
                torch.save(results, seed_file)
            except Exception as e:
                print("Caught exception: ", e)
                pass

        try:
            wandb.finish()
        except Exception as e:
            print("Caught exception: ", e)
            pass

    # Save results
    torch.save(
        results,
        data_path
        / f"{args.net}_{args.act}_{size_str}_{args.eraser}_{args.name}_{args.dataset}.pth",
    )
