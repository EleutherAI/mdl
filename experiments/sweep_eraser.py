import subprocess
from argparse import ArgumentParser
from pathlib import Path

def run_training(
        width: int, depth: int, eraser: str, lr: float, b1: float, mup_width: int, mup_depth: int, args
    ):
    cmd = [
        f"CUDA_VISIBLE_DEVICES={args.device}",
        "python", "-m", "experiments.cli",
        "--name", f"{args.out}",
        "--width", f"{width}",
        "--depth", f"{depth}",
        "--eraser", f"{eraser}",
        "--out", f"{args.out}",
        "--net", args.net,
        "--lr", f"{lr}",
        "--b1", f"{b1}",
        "--schedulefree",
        "--mup_width", f"{mup_width}",
        "--mup_depth", f"{mup_depth}",
        "--act", f"{args.act}",
        "--dataset", f"{args.dataset}",
    ]
    if args.normalize:
        cmd.append("--normalize")
    if args.nocache:
        cmd.append("--nocache")
    print(f"\nLaunching training...")
    print("Command:", " ".join(cmd))
    
    try:
        process = subprocess.run(
            " ".join(cmd),
            shell=True,
            check=True,
            text=True
        )
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
    parser.add_argument("--dataset", type=str, choices=("mnist", "cifarnet", "cifar10"), default="cifar10")
    parser.add_argument("--erasers", nargs="+", default=["control", "qleace", "leace", "qleace2"])
    parser.add_argument("--act", type=str, choices=("relu", "gelu", "swiglu"), default="relu")
    return parser.parse_args()

# Most to least powerful
sweep_params = {
    'convnext': {
        'lr': {
            'control': 5e-5,
            'leace': 1e-4,
            'qleace': 1e-3,
        },
        'b1': {
            'control': 0.9,
            'leace': 0.9,
            'qleace': 0.9,
        },
        # Width specifies the first stage; at each additional stage the width is doubled
        'mup_width': 40,
        'mup_depth': 2,
        'widths': [40, 48, 64], # 80, 96
        'depths': [2, 3, 4]
    },
    'swin': {
        'lr': {
            'control': 1e-3,
            'leace': 1e-3,
            'qleace': 1e-3,
        },
        'b1': {
            'control': 0.9,
            'leace': 0.9,
            'qleace': 0.9,
        },
        'mup_width': 32,
        'mup_depth': 2,
        'widths': [32, 64, 128], # 256, 512
        'depths': [2, 4, 8]
    },
    # 'resnet': {
    #     'mup_width': 4,
    #     'mup_depth': 2,
    #     'widths': [2, 4, 8], # num channels doubled after each layer
    #     'depths': [2, 4, 8] #  16
    # },
    'resmlp': {
        'mup_width': 128,
        'mup_depth': 2,
        'widths': [128, 256, 512, 1024],
        'depths': [2, 4, 8],
        'lr': {
            'control': 5e-4,
            'leace': 5e-4,
            'qleace': 5e-4,
            'qleace2': 5e-4, # guessing
        },
        'b1': {
            'control': 0.99,
            'leace': 0.95,
            'qleace': 0.95,
            'qleace2': 0.95, # guessing
        },
    },
    'mlp': {
        'lr': {
            'control': 5e-4,
            'leace': 5e-4,
            'qleace': 5e-4,
            'qleace2': 5e-4, # guessing
        },
        'b1': {
            'control': 0.99,
            'leace': 0.95,
            'qleace': 0.95,
            'qleace2': 0.95, # guessing
        },
        'mup_width': 128,
        'mup_depth': 2,
        'widths': [64, 128, 256, 512, 1024, 2048],
        'depths': [1, 2, 3, 4, 6, 8] # #  Loses coherence at 16, 1 breaks probe
    },
    # 'linear': {
    #     # Unused
    #     'mup_width': 0,
    #     'mup_depth': 0,
    #     'widths': [0],
    #     'depths': [0]
    # },
}

def artifact_exists(width, depth, net, eraser, out, act, args):
    artifact_name = f"{net}_{act}_h={width}_d={depth}_{eraser}_{out}.pth"
    or_names = [
        f"{net}_{act}_h={width}_d={depth}_{eraser}_24-11-19.pth",
        # f"{net}_{act}_h={width}_d={depth}_{eraser}_results.pth",
    ]

    if args.normalize:
        artifact_name = f"{net}_{act}_h={width}_d={depth}_{eraser}_n={args.normalize}_{out}.pth"
        or_name = f"{net}_{act}_h={width}_d={depth}_{eraser}_n={args.normalize}_24-11-19.pth"
        if (Path(f"/mnt/ssd-1/lucia/{out}") / or_name).exists():
            return True
        return (Path(f"/mnt/ssd-1/lucia/{out}") / artifact_name).exists()
    
    for or_name in or_names:
        if (Path(f"/mnt/ssd-1/lucia/{out}") / or_name).exists():
            return True
    return (Path(f"/mnt/ssd-1/lucia/{out}") / artifact_name).exists()


def main():
    args = parse_args()

    widths = sweep_params[args.net]['widths']
    depths = sweep_params[args.net]['depths']
    mup_width = sweep_params[args.net]['mup_width']
    mup_depth = sweep_params[args.net]['mup_depth']

    for eraser in args.erasers:
        lr = sweep_params[args.net]['lr'][eraser]
        b1 = sweep_params[args.net]['b1'][eraser]

        if args.width:
            for width in widths[args.start:]:
                if args.overwrite or not artifact_exists(width, mup_depth, args.net, eraser, args.out, args.act, args):
                    run_training(
                        width, mup_depth, eraser, lr, b1, mup_width, mup_depth, args
                    )

        if args.depth:
            for depth in depths[args.start:]:
                if args.overwrite or not artifact_exists(mup_width, depth, args.net, eraser, args.out, args.act, args):
                    run_training(
                        mup_width, depth, eraser, lr, b1, mup_width, mup_depth, args
                    )

if __name__ == "__main__":
    main()