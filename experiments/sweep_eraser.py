import subprocess
from argparse import ArgumentParser
from pathlib import Path
from glob import glob


def run_training(
    width: int | None,
    depth: int | None,
    arch: str | None,
    eraser: str,
    lr: float,
    b1: float,
    mup_width: int | None,
    mup_depth: int | None,
    mup_arch: str | None,
    args,
):
    cmd = [
        "CUDA_VISIBLE_DEVICES={args.device}",
        "python",
        "-m",
        "experiments.cli",
        "--name",
        f"{args.out}",
        "--eraser",
        f"{eraser}",
        "--out",
        f"{args.out}",
        "--net",
        args.net,
        "--lr",
        f"{lr}",
        "--b1",
        f"{b1}",
        "--act",
        f"{args.act}",
        "--dataset",
        f"{args.dataset}",
    ]

    if arch is not None:
        cmd.extend(["--arch", f"{arch}", "--mup_arch", f"{mup_arch}"])
    else:
        cmd.extend(
            [
                "--width",
                f"{width}",
                "--depth",
                f"{depth}",
                "--mup_width",
                f"{mup_width}",
                "--mup_depth",
                f"{mup_depth}",
            ]
        )

    if args.normalize:
        cmd.append("--normalize")
    if args.nocache:
        cmd.append("--nocache")
    if args.overwrite:
        cmd.append("--overwrite")

    print("\nLaunching training...")
    print("Command:", " ".join(cmd))

    try:
        process = subprocess.run(" ".join(cmd), shell=True, check=True, text=True)
        print("process exit code", process.returncode)
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        # print("Continuing to train...")
        exit(0)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--net", type=str, default="convnext")
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--width", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--nocache", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=(
            "mnist",
            "cifarnet",
            "cifar10",
            "fake-cifar10",
            "fake-cifarnet",
            "svhn",
            "fake-svhn",
        ),
        default="cifar10",
    )
    parser.add_argument(
        "--erasers", nargs="+", default=["control", "qleace", "leace", "alf_qleace"]
    )  # "random"
    parser.add_argument(
        "--act", type=str, choices=("relu", "gelu", "swiglu"), default="relu"
    )
    return parser.parse_args()


# Most to least powerful
sweep_params = {
    "lenet": {
        "lr": {
            "control": 5e-4,  # Guessing
            "leace": 5e-4,  # Guessing
            "qleace": 5e-4,  # Guessing
            "alf_qleace": 5e-4,  # Guessing
        },
        "b1": {
            "control": 0.95,  # Guessing
            "leace": 0.95,  # Guessing
            "qleace": 0.95,  # Guessing
            "alf_qleace": 0.95,  # Guessing
        },
        "mup_width": 128,
        "mup_depth": 2,
        # These will be converted from the MLP of this size to a parameter-matched LeNet
        "widths": [64, 128, 256, 512, 1024, 2048],
        "depths": [1, 2, 3, 4, 6, 8],
    },
    "mlp": {
        "svhn": {
            "lr": {
                "control": 1e-4,
                "leace": 1e-4,
                "qleace": 1e-4,
                "alf_qleace": 1e-4,  # guessing
                "random": 1e-4,
            },
            "b1": {
                "control": 0.95,  # was 0.99 for cifar10
                "leace": 0.95,
                "qleace": 0.95,
                "alf_qleace": 0.95,  # guessing
                "random": 0.95,
            },
        },
        "lr": {
            "control": 5e-4,
            "leace": 5e-4,
            "qleace": 5e-4,
            "alf_qleace": 5e-4,  # guessing
            "random": 5e-4,
        },
        "b1": {
            "control": 0.95,  # was 0.99 for cifar10
            "leace": 0.95,
            "qleace": 0.95,
            "alf_qleace": 0.95,  # guessing
            "random": 0.95,
        },
        "mup_width": 128,
        "mup_depth": 2,
        "widths": [64, 128, 256, 512, 1024, 2048],
        "depths": [1, 2, 3, 4, 6, 8],  # #  Loses coherence at 16, 1 breaks probe
    },
    "convnext": {
        "lr": {
            "control": 5e-5,
            "leace": 1e-4,
            "qleace": 1e-3,
            "alf_qleace": 1e-3,
        },
        "b1": {
            "control": 0.9,
            "leace": 0.9,
            "qleace": 0.9,
            "alf_qleace": 0.9,
        },
        "archs": ["atto", "femto", "pico", "nano", "tiny"],
        "mup_arch": "atto",
    },
    "swin": {
        "lr": {
            "control": 1e-3,
            "leace": 1e-3,
            "qleace": 1e-3,
            "alf_qleace": 1e-3,
        },
        "b1": {
            "control": 0.9,
            "leace": 0.9,
            "qleace": 0.9,
            "alf_qleace": 0.9,
        },
        "archs": ["atto", "femto", "pico", "nano", "tiny"],
        "mup_arch": "atto",
    },
    "resmlp": {
        "mup_width": 128,
        "mup_depth": 2,
        "widths": [64, 128, 256, 512, 1024, 2048],
        "depths": [1, 2, 3, 4, 6, 8],
        "lr": {
            "control": 5e-4,
            "leace": 5e-4,
            "qleace": 5e-4,
            "alf_qleace": 5e-4,  # guessing
        },
        "b1": {
            "control": 0.99,
            "leace": 0.95,
            "qleace": 0.95,
            "alf_qleace": 0.95,  # guessing
        },
    },
    "skipmlp": {
        "mup_width": 128,
        "mup_depth": 2,
        "widths": [64, 128, 256, 512, 1024, 2048],
        "depths": [1, 2, 3, 4, 6, 8],
        "lr": {
            "control": 5e-4,
            "leace": 5e-4,
            "qleace": 5e-4,
            "alf_qleace": 5e-4,  # guessing
        },
        "b1": {
            "control": 0.99,
            "leace": 0.95,
            "qleace": 0.95,
            "alf_qleace": 0.95,  # guessing
        },
    },
}


def artifact_exists(
    width: int | None, depth: int | None, arch: str | None, eraser: str, args
):
    assert not (width is None and arch is None)

    if arch is not None:
        patterns = [
            f"{args.net}_{args.act}_arch={arch}_{eraser}_*_d={args.dataset}.pth",
            f"{args.net}_{args.act}_arch={arch}_{eraser}_*_{args.dataset}.pth",
        ]
    else:
        patterns = [
            f"{args.net}_{args.act}_h={width}_d={depth}_{eraser}_*_d={args.dataset}.pth",
            f"{args.net}_{args.act}_h={width}_d={depth}_{eraser}_*_{args.dataset}.pth",
        ]

    for pattern in patterns:
        full_pattern = str(Path(f"{args.out}") / pattern)
        if glob(full_pattern):
            return True

    return False


def main():
    args = parse_args()

    params = sweep_params[args.net]

    for eraser in args.erasers:
        if args.dataset in sweep_params[args.net]:
            lr = sweep_params[args.net][args.dataset]["lr"][eraser]
            b1 = sweep_params[args.net][args.dataset]["b1"][eraser]
            print("Using dataset specific lr and b1")
        else:
            lr = sweep_params[args.net]["lr"][eraser]
            b1 = sweep_params[args.net]["b1"][eraser]

        if args.net in ["swin", "convnext"]:
            for arch in params["archs"][args.start :]:
                if args.overwrite or not artifact_exists(
                    None, None, arch, eraser, args
                ):
                    run_training(
                        None,
                        None,
                        arch,
                        eraser,
                        lr,
                        b1,
                        None,
                        None,
                        params["mup_arch"],
                        args,
                    )
        else:
            if args.width:
                for width in params["widths"][args.start :]:
                    if args.overwrite or not artifact_exists(
                        width, params["mup_depth"], None, eraser, args
                    ):
                        run_training(
                            width,
                            params["mup_depth"],
                            None,
                            eraser,
                            lr,
                            b1,
                            params["mup_width"],
                            params["mup_depth"],
                            None,
                            args,
                        )

            if args.depth:
                for depth in params["depths"][args.start :]:
                    if args.overwrite or not artifact_exists(
                        params["mup_width"], depth, None, eraser, args
                    ):
                        run_training(
                            params["mup_width"],
                            depth,
                            None,
                            eraser,
                            lr,
                            b1,
                            params["mup_width"],
                            params["mup_depth"],
                            None,
                            args,
                        )


if __name__ == "__main__":
    main()
