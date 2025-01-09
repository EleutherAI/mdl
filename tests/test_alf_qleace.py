from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
from concept_erasure.quadratic import QuadraticFitter
from concept_erasure.leace import LeaceFitter
from concept_erasure.alf_qleace import AlfQLeaceFitter
from torch import Tensor
from tqdm.auto import tqdm
import lovely_tensors as lt

from experiments.cli import get_cifar10, get_cifarnet


if __name__ == "__main__":
    lt.monkey_patch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = ArgumentParser()
    parser.add_argument("--eraser", type=str, choices=("control", "leace", "oleace", "qleace", "alf_qleace"), default="control")
    parser.add_argument("--dataset", type=str, choices=("cifar10", "cifarnet"), default="cifar10")
    parser.add_argument("--nocache", action="store_true")
    args = parser.parse_args()

    (X_train, Y_train, X_val, Y_val, X_test, Y_test, k, X, Y) = {
        "cifar10": get_cifar10(),
        "cifarnet": get_cifarnet(),
    }[args.dataset]

    num_features = X.shape[1] * X.shape[2] * X.shape[3]

    # Populate eraser cache using training data
    state_path = Path("data") / "erasers_cache" / f"{args.dataset}_state.pth"
    state_path.parent.mkdir(exist_ok=True)
    state = {} if not state_path.exists() else torch.load(state_path)

    if args.eraser != "control" and (args.eraser not in state or args.nocache):
        cls = {
            "leace": LeaceFitter,
            "qleace": QuadraticFitter,
            "alf_qleace": AlfQLeaceFitter,
        }[args.eraser]

        dtype = torch.bfloat16 if args.dataset == "cifarnet" else torch.float32
        if args.eraser == "alf_qleace":
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
    print("Class means after alf_qleace erasure:", class_means)

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
    print(f"ALF-QLEACE Cov trace: {erased.flatten(1).T.cov().trace().item():.2f}") 
    