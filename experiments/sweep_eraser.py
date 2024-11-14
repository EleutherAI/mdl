import subprocess
from argparse import ArgumentParser
from pathlib import Path

def run_training(width: int, depth: int, net: str, eraser: str, device: int, out: str):
    cmd = [
        f"CUDA_VISIBLE_DEVICES={device}",
        "python", "-m", "experiments.cli",
        "--name", f"{out}",
        "--width", f"{width}",
        "--depth", f"{depth}",
        "--eraser", f"{eraser}",
        "--out", f"{out}",
        "--net", net,
    ]
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
        print("Continuing to train...")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--net", type=str, default="convnext")
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--erasers", nargs="+", default=["none", "qleace", "leace"])
    return parser.parse_args()

# Most to least powerful
sweep_params = {
    'convnext': {
        # Width specifies the first stage; at each additional stage the width is doubled
        'widths': [40, 48, 64], # 80, 96
        'depths': [2, 3, 4]
    },
    'swin': {
        'widths': [32, 64, 128], # 256, 512
        'depths': [2, 4, 8]
    },
    'resnet': {
        'widths': [2, 4, 8], # num channels doubled after each layer
        'depths': [2, 4, 8] #  16
    },
    'resmlp': {
        'widths': [128, 256, 512, 1024],
        'depths': [2, 4, 8]
    },
    'mlp': {
        'widths': [64, 128, 256, 512, 1024, 2048], # 4096, 8192
        'depths': [2, 4, 8] # 16
    },
    'linear': {
        # Unused
        'widths': [0],
        'depths': [0]
    },
}

def artifact_exists(width, depth, net, eraser, out):
    artifact_name = f"{net}_h={width}_d={depth}_{eraser}_sweep_all_epochs.pth"
    return (Path(f"/mnt/ssd-1/lucia/{out}") / artifact_name).exists()


def main():
    args = parse_args()

    widths = sweep_params[args.net]['widths']
    depths = sweep_params[args.net]['depths']
    
    for eraser in args.erasers:
        for width in widths:
            if not artifact_exists(width, depths[0], args.net, eraser, args.out):
                run_training(width, depths[0], args.net, eraser, args.device, args.out)

        for depth in depths:
            if not artifact_exists(widths[0], depth, args.net, eraser, args.out):
                run_training(widths[0], depth, args.net, eraser, args.device, args.out)

if __name__ == "__main__":
    main()