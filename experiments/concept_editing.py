import json
import random
from typing import Callable, Literal

import numpy as np
import torch
from concept_erasure import QuadraticFitter
from sklearn.metrics import accuracy_score
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_tensor

from mdl import MlpProbe, QuadraticProbe, VisionProbe
from mdl.probe import Probe

NUM_CLASSES = 10


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
    test_size: int = 1024,
    total_size: int | None = None,
    flatten: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = CIFAR10(root=download_dir, download=True)
    images, labels = zip(*data)

    X = torch.stack(list(map(to_tensor, images))).cuda()  # n x c x w x h
    Y = torch.tensor(labels).cuda()

    # Shuffle deterministically
    rng = torch.Generator(device=X.device).manual_seed(42)
    perm = torch.randperm(len(X), generator=rng, device=X.device)
    X, Y = X[perm][:total_size], Y[perm][:total_size]

    if flatten:
        X = X.view(X.shape[0], -1)  # n x d

    X_train, X_test = X[:-test_size], X[-test_size:]
    Y_train, Y_test = Y[:-test_size], Y[-test_size:]
    return X_train, X_test, Y_train, Y_test


def train_model(
    cls: type[Probe], X_train: torch.Tensor, Y_train: torch.Tensor
) -> Probe:
    model = cls(
        X_train.shape[1],
        num_classes=NUM_CLASSES,
        device=X_train.device,
        dtype=torch.bfloat16,
    )
    model.fit(X_train, Y_train, verbose=True, max_epochs=200, early_stop_epochs=20)
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
    eval_against_target: bool = True,
    compute_by_class: bool = False,
    metric: Literal["loss", "top1"] = "loss",
) -> torch.Tensor:
    def eval_metric(x, y):
        return (
            model.loss(x, y)
            if metric == "loss"
            else accuracy_score(
                y.clone().cpu(), model(x.to(model.dtype)).argmax(dim=1).cpu()
            )
        )

    device = X_test.device
    # 10x the data by making a copy of each row and editing the
    # concept to be each of the 10 classes
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
    Y_eval = Y_target if eval_against_target else Y_test

    if compute_by_class:
        losses = torch.zeros(
            (NUM_CLASSES, NUM_CLASSES), device=device
        )  # source -> target
        for source in range(NUM_CLASSES):
            for target in range(NUM_CLASSES):
                mask = (Y_test == source) & (Y_target == target)
                losses[source, target] = eval_metric(X_test[mask], Y_eval[mask])
        return losses
    return eval_metric(X_test, Y_eval)


def main():
    # TODO: use multiple seeds
    results = []
    model_configs = [
        (MlpProbe, dict(num_layers=2)),
        (MlpProbe, dict(num_layers=6)),
        (VisionProbe, dict()),
        (QuadraticProbe, dict()),
    ]
    editing_modes = ["quadratic", "linear"]
    for cls, cfg in model_configs:
        cfg_str = f"(num_layers={cfg['num_layers']})" if cls == MlpProbe else ""
        print(f"Training {cls.__name__} {cfg_str}...")
        X_train, X_test, Y_train, Y_test = get_train_test_data(
            total_size=None, test_size=1024, flatten=cls != VisionProbe
        )
        model = train_model(cls, X_train, Y_train)
        for editing_mode in editing_modes:
            print(f"Evaluating {cls.__name__} with {editing_mode} editor...")
            editor = get_editor(editing_mode, X_train, Y_train)
            with torch.no_grad():
                model.eval()

                loss_edited = evaluate_model(
                    model, X_test, Y_test, editor=editor, eval_against_target=True
                )
                loss_against_source_edited = evaluate_model(
                    model, X_test, Y_test, editor=editor, eval_against_target=False
                )
                loss_matrix_against_source = evaluate_model(
                    model, X_test, Y_test, editor=editor, compute_by_class=True
                )
                loss_matrix_against_target = evaluate_model(
                    model,
                    X_test,
                    Y_test,
                    editor=editor,
                    eval_against_target=True,
                    compute_by_class=True,
                )
                loss = model.loss(X_test, Y_test)

                acc_edited = evaluate_model(
                    model,
                    X_test,
                    Y_test,
                    editor=editor,
                    eval_against_target=True,
                    metric="top1",
                )
                acc_against_source_edited = evaluate_model(
                    model,
                    X_test,
                    Y_test,
                    editor=editor,
                    eval_against_target=False,
                    metric="top1",
                )
                acc_matrix_against_source = evaluate_model(
                    model,
                    X_test,
                    Y_test,
                    editor=editor,
                    compute_by_class=True,
                    metric="top1",
                )
                acc_matrix_against_target = evaluate_model(
                    model,
                    X_test,
                    Y_test,
                    editor=editor,
                    eval_against_target=True,
                    compute_by_class=True,
                    metric="top1",
                )
                acc = accuracy_score(
                    Y_test.cpu(), model(X_test.to(model.dtype)).argmax(dim=1).cpu()
                )
            results.append(
                {
                    "model": cls.__name__ + cfg_str,
                    "editing_mode": editing_mode,
                    "loss": float(loss),
                    "edited_loss_against_source": float(loss_against_source_edited),
                    "edited_loss_against_target": float(loss_edited),
                    "loss_matrix_against_source": loss_matrix_against_source.cpu()
                    .numpy()
                    .tolist(),
                    "loss_matrix_against_target": loss_matrix_against_target.cpu()
                    .numpy()
                    .tolist(),
                    "top1": float(acc),
                    "edited_top1_against_source": float(acc_against_source_edited),
                    "edited_top1_against_target": float(acc_edited),
                    "top1_matrix_against_source": acc_matrix_against_source.cpu()
                    .numpy()
                    .tolist(),
                    "top1_matrix_against_target": acc_matrix_against_target.cpu()
                    .numpy()
                    .tolist(),
                    "n_test": int(X_test.shape[0]),
                    "n_train": int(X_train.shape[0]),
                }
            )
            print(f"Loss without editing: {loss}")
            print(f"Loss against target with editing: {loss_edited}")
            print(f"Loss against source with editing: {loss_against_source_edited}")
            print(f"Loss matrix against source: {loss_matrix_against_source}")
            print(f"Loss matrix against target: {loss_matrix_against_target}")
            print(f"Top-1 without editing: {acc}")
            print(f"Top-1 against target with editing: {acc_edited}")
            print(f"Top-1 against source with editing: {acc_against_source_edited}")
            print(f"Top-1 matrix against source: {acc_matrix_against_source}")
            print(f"Top-1 matrix against target: {acc_matrix_against_target}")
            print()

    with open("../data/CIFAR_editing_results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    main()
