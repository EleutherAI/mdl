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
    parser.add_argument("--overwrite", action="store_true")
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

    if args.eraser != "control" and (args.eraser not in state or args.nocache):
        cls = {
            "leace": LeaceFitter,
            "qleace": QuadraticFitter,
            "qleace2": AlfQLeaceFitter,
        }[args.eraser]

        dtype = torch.bfloat16 if args.dataset == "cifarnet" else torch.float32
        if args.eraser == "qleace2":
            dtype = torch.float32

        fitter = cls(
            num_features, k, dtype=dtype, device=device, shrinkage=True
        )

        for x, y in tqdm(zip(X_train, Y_train)):
            y = torch.as_tensor(y).view(1)
            if args.eraser != "qleace":
                y = F.one_hot(y, k)

            fitter.update(x.view(1, -1).to(device).to(dtype), y.to(device))

        if args.dataset == "cifarnet":
            fitter = fitter.to("cpu")
        eraser = fitter.eraser

        state[args.eraser] = fitter.eraser

    eraser = state[args.eraser]

    erased = eraser(X_train.reshape(len(X_train), -1)).reshape(X_train.shape)
    class_means = [erased[Y_train == c].mean(0) for c in Y_train.unique()]
    print("Class means after qleace2 erasure:", class_means)

    universal_mean = torch.stack(class_means).mean(0)

    max_mean_diff = torch.stack([
        (universal_mean - other_mean).flatten().norm()
        for other_mean in class_means
    ]).max()
    max_pixel_diff = torch.stack([
        (universal_mean - other_mean).flatten().abs().max()
        for other_mean in class_means
    ]).max()
    print("Max difference norm", max_mean_diff)
    print("Max pixel difference", max_pixel_diff)

    # universal_covariance = torch.cov(erased.flatten(1).T)
    # print("Universal covariance:", universal_covariance)

    class_covariances = [erased[Y_train == c].flatten(1).T.cov() for c in Y_train.unique()]
    max_diff_between_any_two_covariances = torch.stack([
        (class_covariances[i] - class_covariances[j]).flatten().norm()
        for i in range(len(class_covariances))
        for j in range(i + 1, len(class_covariances))
    ]).max()
    max_pixel_diff_between_any_two_covariances = torch.stack([
        (class_covariances[i] - class_covariances[j]).flatten().abs().max()
        for i in range(len(class_covariances))
        for j in range(i + 1, len(class_covariances))
    ]).max()
    # max_covariance_diff = torch.stack([
    #     (universal_covariance - other_covariance).flatten().norm()
    #     for other_covariance in class_covariances
    # ]).max()

    # Print covariance traces
    print("Max covariance difference norm", max_diff_between_any_two_covariances)
    print("Max covariance difference pixel", max_pixel_diff_between_any_two_covariances)

    leace_eraser = state["leace"]
    leace_erased: Tensor = leace_eraser.to(X_train.device)(X_train.reshape(len(X_train), -1)).reshape(X_train.shape)

    leace_class_means = [leace_erased[Y_train == c].mean(0) for c in Y_train.unique()]
    leace_universal_mean = torch.stack(leace_class_means).mean(0)
    leace_max_mean_diff = torch.stack([
        (leace_universal_mean - other_mean).flatten().norm()
        for other_mean in leace_class_means
    ]).max()
    leace_max_pixel_diff = torch.stack([
        (leace_universal_mean - other_mean).flatten().abs().max()
        for other_mean in leace_class_means
    ]).max()
    print("Max LEACE difference norm", leace_max_mean_diff)
    print("Max LEACE pixel difference", leace_max_pixel_diff)

    # leace_universal_covariance = torch.cov(leace_erased.flatten(1).T)
    # print("LEACE universal covariance:", leace_universal_covariance)

    leace_class_covariances = [leace_erased[Y_train == c].flatten(1).T.cov() for c in Y_train.unique()]
    # leace_max_covariance_diff = torch.stack([
    #     (leace_universal_covariance - other_covariance).flatten().norm()
    #     for other_covariance in leace_class_covariances
    # ]).max()
    leace_max_diff_between_any_two_covariances = torch.stack([
        (leace_class_covariances[i] - leace_class_covariances[j]).flatten().norm()
        for i in range(len(leace_class_covariances))
        for j in range(i + 1, len(leace_class_covariances))
    ]).max()
    leace_max_pixel_diff_between_any_two_covariances = torch.stack([
        (leace_class_covariances[i] - leace_class_covariances[j]).flatten().abs().max()
        for i in range(len(leace_class_covariances))
        for j in range(i + 1, len(leace_class_covariances))
    ]).max()

    print("Max LEACE covariance difference norm", leace_max_diff_between_any_two_covariances)
    print("Max LEACE covariance difference pixel", leace_max_pixel_diff_between_any_two_covariances)

    breakpoint()

    # Cov traces
    
    # 190.42 CIFAR-10
    print(f"Unerased Cov trace: {X_train.flatten(1).T.cov().trace().item():.2f}") 
    # 135.79 CIFAR-10
    print(f"LEACE Cov trace: {leace_erased.flatten(1).T.cov().trace().item():.2f}") 
    # 24.79 CIFAR-10
    print(f"QLEACE2 Cov trace: {erased.flatten(1).T.cov().trace().item():.2f}") 
    

    # Test the leace in alfqleace

    # delta = X_train.flatten(1) - eraser.bias if eraser.bias is not None else X_train.flatten(1)
    # first_step_erased = X_train.flatten(1) - (delta @ eraser.proj_right.mH) @ eraser.proj_left.mH
    # first_step_erased = first_step_erased.reshape(X_train.shape)

    # first_step_class_means = [first_step_erased[Y_train == c].mean(0) for c in Y_train.unique()]
    # first_step_universal_mean = torch.stack(first_step_class_means).mean(0)
    # first_step_max_mean_diff = torch.stack([
    #     (first_step_universal_mean - other_mean).flatten().norm()
    #     for other_mean in first_step_class_means
    # ]).max()
    # first_step_max_pixel_diff = torch.stack([
    #     (first_step_universal_mean - other_mean).flatten().abs().max()
    #     for other_mean in first_step_class_means
    # ]).max()
    # print("Max first step difference norm", first_step_max_mean_diff)
    # print("Max first step pixel difference", first_step_max_pixel_diff)

    # v = eraser.alf_qleace_vecs
    # second_step_erased = first_step_erased.flatten(1) - (v @ first_step_erased.flatten(1).mH).mH @ v
    # second_step_erased = second_step_erased.reshape(X_train.shape)

    # second_step_class_means = [second_step_erased[Y_train == c].mean(0) for c in Y_train.unique()]
    # second_step_universal_mean = torch.stack(second_step_class_means).mean(0)
    # second_step_max_mean_diff = torch.stack([
    #     (second_step_universal_mean - other_mean).flatten().norm()
    #     for other_mean in second_step_class_means
    # ]).max()
    # print("Max second step difference norm", second_step_max_mean_diff)
    # second_step_max_pixel_diff = torch.stack([
    #     (second_step_universal_mean - other_mean).flatten().abs().max()
    #     for other_mean in second_step_class_means
    # ]).max()
    # print("Max second step pixel difference", second_step_max_pixel_diff)

