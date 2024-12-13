from argparse import ArgumentParser
from pathlib import Path
from functools import partial
from typing import Any

from datasets import load_dataset
import wandb
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2.functional import to_tensor
from concept_erasure.quadratic import QuadraticFitter
from concept_erasure.leace import LeaceFitter
from concept_erasure.alf_qleace import AlfQLeaceFitter
from torch import Tensor
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm
import lovely_tensors as lt
from mup import make_base_shapes

from mdl.mlp_probe import ResMlpProbe, MlpProbe, LinearProbe
from mdl.sweep import Sweep
from mdl.vision_probe import ConvNextProbe, VisionProbe, SwinProbe
from mdl.resnet_probe import ResNetProbe

lt.monkey_patch()


def get_mnist():
    train_dataset: HfDataset = load_dataset("mnist", split='train') # type: ignore

    def map_fn(ex):
        return {
            'input_ids': transforms.ToTensor()(ex['image']),
            'label': ex['label']
        }

    train_dataset = train_dataset.map(
        function=map_fn,
        remove_columns=['image'],
        new_fingerprint='transformed_mnist', # type: ignore
        keep_in_memory=True # type: ignore
    )
    train_dataset = train_dataset.with_format('torch')
    train_dataset.set_format(type='torch', columns=['input_ids', 'label'])

    print("Final columns:", train_dataset.column_names)

    # Calculate mean and std of pixel values
    input_ids = assert_type(Tensor, train_dataset['input_ids'])
    mean = input_ids.mean().item()
    std = input_ids.std().item()
    def normalize(image):
        transform = transforms.Compose([
            transforms.Normalize((mean,), (std,))
        ])
        return transform(image)


    test_dataset: HfDataset = load_dataset('mnist', split='test') # type: ignore

    test_dataset = test_dataset.map(
        function=map_fn,
        remove_columns=['image'],
        new_fingerprint='transformed_mnist', # type: ignore
        keep_in_memory=True # type: ignore
    )
    test_dataset.set_format(type='torch', columns=['input_ids', 'label'])

    test_dataset = test_dataset.map(
        lambda example: {'input_ids': normalize(example['input_ids'])},
        new_fingerprint='transformed_mnist'
    )

    return test_dataset



def get_cifarnet(device="cpu"):
    nontest = load_dataset("EleutherAI/cifarnet", split='train') # type: ignore
    def map_fn(ex):
        return {
            'input_ids': transforms.ToTensor()(ex['img']),
            'label': ex['label']
        }

    nontest: HfDataset = nontest.map(function=map_fn) # type: ignore
    nontest.set_format(type='torch', columns=['input_ids', 'label'])

    X: Tensor = nontest['input_ids'].to(device) # type: ignore
    Y: Tensor = nontest['label'].to(device) # type: ignore

    # Shuffle deterministically
    rng = torch.Generator(device=X.device).manual_seed(42)
    perm = torch.randperm(len(X), generator=rng, device=X.device)
    X, Y = X[perm], Y[perm]

    k = int(Y.max()) + 1

    # Split train and validation
    val_size = 1024

    X_train, X_val = X[:-val_size], X[-val_size:]
    Y_train, Y_val = Y[:-val_size], Y[-val_size:]

    # Test set is entirely separate
    test = load_dataset("EleutherAI/cifarnet", split='test') # type: ignore
    test = test.map(map_fn)
    test.set_format(type='torch', columns=['input_ids', 'label'])

    X_test: Tensor = test['input_ids'].to(device) # type: ignore
    Y_test: Tensor = test['label'].to(device) # type: ignore

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, k, X, Y


def get_cifar10(normalize: bool = False):
    nontest = CIFAR10("/home/lucia/cifar10", download=True)

    images, labels = zip(*nontest)
    X: Tensor = torch.stack(list(map(to_tensor, images))).to(device)
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

    # Test set is entirely separate
    test = CIFAR10(root="/home/lucia/cifar10-test", train=False, download=True)
    test_images, test_labels = zip(*test)
    X_test: Tensor = torch.stack(list(map(to_tensor, test_images))).to(device)
    Y_test = torch.tensor(test_labels).to(device)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, k, X, Y


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--eraser", type=str, choices=("control", "leace", "oleace", "qleace", "qleace2"), default="control")
    parser.add_argument("--net", type=str, choices=("mlp", "resmlp", "resnet", "convnext", "linear", "vision", "swin"), default="mlp")
    parser.add_argument("--mup_width", type=int, help="Width of the base model used to tune the initial LR")
    parser.add_argument("--mup_depth", type=int, help="Depth of the base model used to tune the initial LR")
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=30_000)
    parser.add_argument("--early_stop_epochs", type=int, default=100)
    parser.add_argument("--schedulefree", action="store_true")
    parser.add_argument("--dataset", type=str, choices=("cifar10", "mnist", "cifarnet"), default="cifar10")
    parser.add_argument("--act", type=str, choices=("relu", "gelu", "swiglu"), default="relu")
    parser.add_argument("--muon", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--nocache", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--trial", action="store_true", help="Run a single trial with all data")
    args = parser.parse_args()

    (X_train, Y_train, X_val, Y_val, X_test, Y_test, k, X, Y) = get_cifar10(device)
    if args.dataset == "cifar10":
        (X_train, Y_train, X_val, Y_val, X_test, Y_test, k, X, Y) = get_cifar10(normalize=args.normalize)
    elif args.dataset == "mnist":
        (X_train, Y_train, X_val, Y_val, X_test, Y_test, k, X, Y) = get_mnist()
    elif args.dataset == "cifarnet":
        (X_train, Y_train, X_val, Y_val, X_test, Y_test, k, X, Y) = get_cifarnet()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    num_features = X.shape[1] * X.shape[2] * X.shape[3]

    def normalize(X, X_train, X_val, X_test):
        X_flat = X.reshape(X.shape[0], -1)
        
        mean = X_flat.mean(dim=0, keepdim=True)
        X_centered = X_flat - mean
        
        cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)
        
        scaling = torch.sqrt(torch.diagonal(cov))
        scaling = torch.where(scaling > 0, scaling, torch.ones_like(scaling))
        
        def normalize_data(data: Tensor) -> Tensor:
            data_flat = data.reshape(data.shape[0], -1)
            data_centered = data_flat - mean
            data_normalized = data_centered / scaling
            return data_normalized.reshape(data.shape)
        
        X = normalize_data(X)
        X_train = normalize_data(X_train)
        X_val = normalize_data(X_val)
        X_test = normalize_data(X_test)

        return X, X_train, X_val, X_test

    # Populate eraser cache using training data
    state_path = Path("erasers_cache") / f"{args.dataset}_state_2.pth"
    state_path.parent.mkdir(exist_ok=True)
    state = {} if not state_path.exists() else torch.load(state_path)

    def normalize(X, X_train, X_val, X_test):
        X_flat = X.reshape(X.shape[0], -1)
        
        mean = X_flat.mean(dim=0, keepdim=True)
        X_centered = X_flat - mean
        
        cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)
        
        scaling = torch.sqrt(torch.diagonal(cov))
        scaling = torch.where(scaling > 0, scaling, torch.ones_like(scaling))
        
        def normalize_data(data: Tensor) -> Tensor:
            data_flat = data.reshape(data.shape[0], -1)
            data_centered = data_flat - mean
            data_normalized = data_centered / scaling
            return data_normalized.reshape(data.shape)
        
        X = normalize_data(X)
        X_train = normalize_data(X_train)
        X_val = normalize_data(X_val)
        X_test = normalize_data(X_test)

        return X, X_train, X_val, X_test

    if args.eraser != "control" and (args.eraser not in state or args.nocache):
        cls = {
            "leace": LeaceFitter,
            "qleace": QuadraticFitter,
            "qleace2": AlfQLeaceFitter,
        }[args.eraser]

        fitter = cls(
            num_features, k, dtype=torch.float32, device=device, shrinkage=True
        )
        for x, y in tqdm(zip(X_train, Y_train)):
            y = torch.as_tensor(y).view(1)
            if args.eraser != "qleace":
                y = F.one_hot(y, k)

            fitter.update(x.view(1, -1).to(device), y.to(device))

        state[args.eraser] = fitter.eraser
        torch.save(state, state_path)

    model_cls = {
        "mlp": MlpProbe,
        "resmlp": ResMlpProbe,
        "resnet": ResNetProbe,
        "convnext": ConvNextProbe,
        "linear": LinearProbe,
        "vision": VisionProbe,
        "swin": SwinProbe,
    }[args.net]

    flatten = {
        "mlp": True,
        "resmlp": True,
        "resnet": False,
        "convnext": False,
        "linear": True,
        "vision": False,
        "swin": False,
    }

    image_size = X.shape[-1]
    padding = round(image_size * 0.125)

    if flatten[args.net]:

        def reshape(x):
            "reshape tensor to CxHxW"
            return x.view(-1, X.shape[1], X.shape[2], X.shape[3])

        augment = transforms.Compose(
            [
                transforms.Lambda(reshape),
                transforms.RandomCrop(image_size, padding=padding),
                transforms.RandomHorizontalFlip(),
                transforms.Lambda(lambda x: x.flatten(1)),
            ]
        )

        def none_transform(x, y):
            return x.flatten(1)

    else:
        augment = transforms.Compose(
            [
                transforms.RandomCrop(image_size, padding=padding),
                transforms.RandomHorizontalFlip(),
            ]
        )

        def none_transform(x, y):
            return x

    if args.eraser == "leace" or args.eraser == "qleace2":

        def erase(x: Tensor, y: Tensor, eraser):
            x_erased = eraser(x.flatten(1))
            return x_erased if flatten[args.net] else x_erased.reshape_as(x)

    else:

        def erase(x: Tensor, y: Tensor, eraser):
            x_erased = eraser(x.flatten(1), y)
            return x_erased if flatten[args.net] else x_erased.reshape_as(x)

    transform = (
        partial(erase, eraser=state[args.eraser].to(device))
        if args.eraser != "control"
        else none_transform
    )

    # TODO Lucia normalize eraserd data - currently only supports control run
    if args.normalize:
        X, X_train, X_val, X_test = normalize(X, X_train, X_val, X_test)


    # TODO Lucia normalize eraserd data - currently only supports control run
    if args.normalize:
        X, X_train, X_val, X_test = normalize(X, X_train, X_val, X_test)

    base_model = model_cls(
        num_classes=k,
        num_features=num_features,
        num_layers=args.depth, # mup depth unsupported
        hidden_size=args.mup_width if args.mup_width else args.width,
    )
    delta_model = model_cls(
        num_classes=k,
        num_features=num_features,
        num_layers=args.depth,
        hidden_size=args.width,
    )
    base_shapes_path = f"mup-{args.net}-{args.width}-{args.depth}-{args.mup_width}.bsh"
    make_base_shapes(base_model, delta_model, savefile=base_shapes_path)

    if args.mup_depth and args.depth != args.mup_depth:
        if model_cls == MlpProbe:
            # Implement depth-wise scaling from https://arxiv.org/pdf/2305.07810
            args.lr = args.lr * (args.mup_depth / args.depth) ** (3/2)
        else:
            # Implement more conservative scaling for vision models
            args.lr = args.lr * (args.mup_depth / args.depth) ** (1/2)

    seed_path = Path(
        f"/mnt/ssd-1/lucia/{args.out}-seeds"
        if not args.debug
        else f"/mnt/ssd-1/lucia/debug-{args.out}-seeds"
    )
    seed_path.mkdir(exist_ok=True, parents=True)

    results = []
    for seed in range(args.num_seeds):
        wandb_name = f'{args.eraser} {args.name} w={args.width} d={args.depth} s={seed} {args.net} act={args.act} lr={args.lr:.3f} b1={args.b1} n={args.normalize} es={args.early_stop_epochs} es={args.early_stop_epochs}'

        run = (
            wandb.init(
                project="mdl",
                entity="eleutherai",
                name=wandb_name,
                config={"eraser": args.eraser, **vars(args)},
                reinit=True
            ) 
            if not args.debug
            else None
        )

        if args.trial:
            probe_kwargs: dict[str, Any] = dict(
                num_classes=k,
                num_features=num_features,
                num_layers=args.depth,
                hidden_size=args.width,
                device=device,
                learning_rate=args.lr,
                schedule_free=args.schedulefree,
                betas=(args.b1, 0.999),
                base_shapes_path=base_shapes_path,
                dtype=torch.float32,
            )
            if args.muon:
                probe_kwargs['muon'] = True

            if model_cls == MlpProbe:
                probe_kwargs['activation'] = args.act

            probe = model_cls(**probe_kwargs)
            probe.fit(
                X_train,
                Y_train,
                x_val=X_val,
                y_val=Y_val,
                seed=0,
                transform=transform,
                augment=augment,
                max_epochs=args.max_epochs,
                early_stop_epochs=args.early_stop_epochs,
                logger=run
            )
            wandb.finish()
            exit(0)

        # name = (
        #     None
        #     if not args.save
        #     else f"{args.net}_h={args.width}_d={args.depth}_{'_'.join(args.erasers)}_{args.name}"
        # )
        
        probe_kwargs = dict(
            num_layers=args.depth,
            hidden_size=args.width,
            learning_rate=args.lr,
            schedule_free=args.schedulefree,
            betas=(args.b1, 0.999),
            base_shapes_path=base_shapes_path,
        )
        if args.muon:
                probe_kwargs['muon'] = True
        if model_cls == MlpProbe:
            probe_kwargs['activation'] = args.act


        sweep = Sweep(
            num_features,
            k,
            device=device,
            dtype=torch.float32,
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
                transform=transform,
                augment=augment,
                reduce_lr_on_plateau=False,
                max_epochs=args.max_epochs,
                early_stop_epochs=args.early_stop_epochs,
            )
        )
        torch.save(results, seed_path / f"{args.net}_{args.act}_h={args.width}_d={args.depth}_{args.eraser}_{args.name}_{seed}.pth")

        try:
            wandb.finish()
        except Exception as e:
            print("Caught exception: ", e)
            pass


    data_path = Path(
        f"/mnt/ssd-1/lucia/{args.out}"
        if not args.debug
        else f"/mnt/ssd-1/lucia/debug-{args.out}"
    )
    data_path.mkdir(exist_ok=True, parents=True)
    
    torch.save(
        results,
        data_path
        / f"{args.net}_{args.act}_h={args.width}_d={args.depth}_{args.eraser}_{args.name}.pth",
    )
