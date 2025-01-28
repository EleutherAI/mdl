import argparse
import json
import os
import random
from typing import Callable, Literal

import numpy as np
import torch
import torchvision as tv
from concept_erasure import QuadraticFitter
from sklearn.metrics import accuracy_score
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_tensor

from mdl import MlpProbe, QuadraticProbe, VisionProbe
from mdl.probe import Probe

NUM_CLASSES = 10
IMAGE_SIZE = 32


def fit_linear_editor(X: torch.Tensor, Z: torch.Tensor, num_classes: int):
    """A linear editor is just a translation between class conditional means."""
    N, D = X.shape
    assert Z.shape == (N,)

    translation_maps = torch.zeros(
        (num_classes, num_classes, D), device=X.device
    )  # i -> j
    conditional_means = torch.zeros((num_classes, D), device=X.device)
    for i in range(num_classes):
        X_i = X[Z == i]
        conditional_means[i] = X_i.mean(dim=0)

    for i in range(num_classes):
        for j in range(num_classes):
            translation_maps[i, j] = conditional_means[j] - conditional_means[i]

    def editor(
        X_eval: torch.Tensor, source_z: torch.Tensor, target_z: torch.Tensor
    ) -> torch.Tensor:
        assert X_eval.shape[0] == len(source_z) == len(target_z)
        assert X_eval.shape[1] == D
        assert source_z.max() < num_classes
        assert target_z.max() < num_classes

        device, dtype = X_eval.device, X_eval.dtype

        X_eval_target = (
            X_eval + translation_maps.to(device).to(dtype)[source_z, target_z]
        )
        return X_eval_target

    return editor


def get_train_test_data(
    download_dir: str = "/mnt/ssd-1/alexm/cifar10",
    test_size: int | None = None,
    train_size: int | None = None,
    flatten: bool = False,
    device="cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_data = CIFAR10(root=download_dir, download=True)
    test_data = CIFAR10(root=download_dir, train=False, download=True)
    X_train, Y_train = prepare_data(
        train_data, size=train_size, flatten=flatten, device=device
    )
    X_test, Y_test = prepare_data(
        test_data, size=test_size, flatten=flatten, device=device
    )
    print("Train+val size:", len(X_train))
    print("Test size:", len(X_test))
    return X_train, X_test, Y_train, Y_test


def prepare_data(
    data: CIFAR10, size: int | None = None, flatten: bool = False, device="cuda"
):
    images, labels = zip(*data)

    X = torch.stack(list(map(to_tensor, images))).to(device)  # n x c x w x h
    Y = torch.tensor(labels).to(X.device)

    # Shuffle deterministically
    rng = torch.Generator(device=X.device).manual_seed(42)
    perm = torch.randperm(len(X), generator=rng, device=X.device)
    X, Y = X[perm][:size], Y[perm][:size]

    if flatten:
        X = X.view(X.shape[0], -1)  # n x d

    return X, Y


def train_model(
    cls: type[Probe], X_train: torch.Tensor, Y_train: torch.Tensor
) -> Probe:
    if cls == VisionProbe:
        model = cls(
            num_classes=NUM_CLASSES,
            device=X_train.device,
            dtype=torch.float32,
        )
    else:
        model = cls(
            X_train.shape[1],
            num_classes=NUM_CLASSES,
            device=X_train.device,
            dtype=torch.float32,
        )
    model.fit(
        X_train,
        Y_train,
        verbose=True,
        max_epochs=100,
        early_stop_epochs=4,
        reduce_lr_on_plateau=False,
    )
    return model


def get_editor(
    kind: Literal["linear", "quadratic"],
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    return_fitter=False,
):
    # We need to flatten the data (in the VisionProbe case it's not already flat)
    X_train = X_train.view(X_train.shape[0], -1).cpu().double()
    Y_train = Y_train.cpu()
    if kind == "quadratic":
        fitter_cls = QuadraticFitter
        fitter = fitter_cls.fit(X_train, Y_train)
        fitter_editor = fitter.editor()

        def editor(
            X_eval: torch.Tensor, source_z: torch.Tensor, target_z: torch.Tensor
        ) -> torch.Tensor:
            assert X_eval.shape[0] == len(source_z) == len(target_z)
            assert X_eval.shape[1] == X_train.shape[1]
            assert source_z.max() < NUM_CLASSES
            assert target_z.max() < NUM_CLASSES

            X_eval = X_eval.to(X_train.device).to(X_train.dtype)
            X_eval_target = X_eval.clone()
            target_z = target_z.to(X_train.device)
            source_z = source_z.to(X_train.device)

            for i in range(NUM_CLASSES):
                X_eval_target[target_z == i] = fitter_editor(
                    X_eval[target_z == i], source_z[target_z == i].cpu(), i
                ).to(X_eval.device)
            return X_eval_target

        if return_fitter:
            return editor, fitter
        return editor
    else:
        return fit_linear_editor(X_train, Y_train, num_classes=NUM_CLASSES)


def evaluate_model(
    model: Probe,
    X_test: torch.Tensor = None,
    Y_test: torch.Tensor = None,
    editor: Callable = None,
) -> torch.Tensor:
    model.eval()

    def get_logits(x, batch_size=64):
        x_batches = x.to(model.dtype).split(batch_size)
        logits = torch.cat([model(x_batch) for x_batch in x_batches])
        return logits

    def eval_metric(logits, y, metric: Literal["loss", "top1"] = "top1"):
        if metric == "loss":
            return model.loss_fn(logits, y).item()
        else:
            return float(accuracy_score(y.cpu(), logits.argmax(dim=1).cpu()))

    X_test_original = X_test.clone()
    Y_test_original = Y_test.clone()
    device = X_test.device
    # 9x the data by making a copy of each row and editing the
    # concept to be each of the 9 other classes
    X_test = X_test.repeat(NUM_CLASSES, *([1] * (X_test.ndim - 1)))
    Y_test = Y_test.repeat(NUM_CLASSES)
    Y_target = (
        torch.arange(NUM_CLASSES, dtype=Y_test.dtype)
        .repeat_interleave(len(X_test) // NUM_CLASSES)
        .to(device)
    )

    # we must flatten the data before passing it to the fitter
    X_test_flat = X_test.view(X_test.shape[0], -1)
    X_test = (
        editor(X_test_flat, source_z=Y_test, target_z=Y_target)
        .view(X_test.shape)
        .to(device)
    )
    Y_test = Y_test.to(device)
    logits = get_logits(X_test)
    logits_without_edit = get_logits(X_test_original)
    results = dict()
    for metric in ["loss", "top1"]:
        results[metric] = eval_metric(logits_without_edit, Y_test_original, metric)

        for eval_against in ["source", "target"]:
            Y_eval = Y_test if eval_against == "source" else Y_target

            # Make metric matrix
            mat = np.zeros((NUM_CLASSES, NUM_CLASSES))  # source -> target
            for source in range(NUM_CLASSES):
                for target in range(NUM_CLASSES):
                    mask = (Y_test == source) & (Y_target == target)
                    mat[source, target] = eval_metric(
                        logits[mask], Y_eval[mask], metric
                    )
            results[f"{metric}_matrix_against_{eval_against}"] = mat.tolist()

            # Evaluate on all non-id edits
            mask = Y_target != Y_test
            results[f"{metric}_against_{eval_against}_edited"] = eval_metric(
                logits[mask], Y_eval[mask], metric
            )
    return results


def main(args):
    padding = round(IMAGE_SIZE * 0.125)

    augmentor = tv.transforms.Compose(
        [
            tv.transforms.RandomCrop(IMAGE_SIZE, padding=padding),
            tv.transforms.RandomHorizontalFlip(),
        ]
    )

    results = []
    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        model_configs = [
            (MlpProbe, dict(num_layers=3)),
            (VisionProbe, dict(augmentor=augmentor)),
            (QuadraticProbe, dict()),
        ]
        editing_modes = ["quadratic", "linear"]
        for cls, cfg in model_configs:
            cfg_str = f"(num_layers={cfg['num_layers']})" if cls == MlpProbe else ""
            print(f"Training {cls.__name__} {cfg_str}...")
            X_train, X_test, Y_train, Y_test = get_train_test_data(
                train_size=args.train_size,
                test_size=args.test_size,
                flatten=cls != VisionProbe,
                download_dir=args.download_dir,
                device=args.device,
            )
            model = train_model(cls, X_train, Y_train)

            for editing_mode in editing_modes:
                print(f"Evaluating {cls.__name__} with {editing_mode} editor...")

                if cls == VisionProbe and args.augment_before_edit:
                    X_train_for_edit = model.augment_data(X_train)
                else:
                    X_train_for_edit = X_train
                editor = get_editor(editing_mode, X_train_for_edit, Y_train)
                with torch.no_grad():
                    eval_result = evaluate_model(model, X_test, Y_test, editor)
                results.append(
                    {
                        "model": cls.__name__ + cfg_str,
                        "editing_mode": editing_mode,
                        "n_test": int(X_test.shape[0]),
                        "n_train": int(X_train.shape[0]),
                        "seed": seed,
                        **eval_result,
                    }
                )
                for k, v in results[-1].items():
                    print(f"{k}: {v}")
                print()

    with open(os.path.join(args.out_dir, "CIFAR_editing_results.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--download-dir", type=str, default="/mnt/ssd-1/alexm/cifar10")
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=".")  # "../data/"
    parser.add_argument("--augment-before-edit", action="store_true", default=True)

    args = parser.parse_args()
    main(args)
