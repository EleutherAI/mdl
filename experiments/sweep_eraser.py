import subprocess
from argparse import ArgumentParser


def run_training(width: int, depth: int, net: str, eraser: str, device: int):
    cmd = [
        f"CUDA_VISIBLE_DEVICES={device}",
        "python", "-m", "experiments.cli",
        "--name sweep",
        "--width", f"{width}",
        "--depth", f"{depth}",
        "--eraser", f"{eraser}",
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
    parser.add_argument("--net", type=str, default=0)
    return parser.parse_args()

def main():
    args = parse_args()

    if args.net == "convnext":
        for width in [8, 16, 32, 64]:
            for depth in [2, 4, 8]:
                for eraser in ["none", "qleace", "leace"]:
                    run_training(width, depth, "convnext", eraser, args.device)

    elif args.net == "resmlp":
        for width in [128, 256, 512, 1024]:
            for depth in [2, 4, 8]:
                for eraser in ["none", "qleace", "leace"]:
                    run_training(width, depth, "resmlp", eraser, args.device)

    if args.net == "mlp":
        for width in [128, 256, 512, 1024]:
            for depth in [2, 4, 8]:
                for eraser in ["none", "qleace", "leace"]:
                    run_training(width, depth, "mlp", eraser, args.device)

    if args.net == "vit":
        for width in [128, 256, 512, 1024]:
            for depth in [2, 4, 8]:
                for eraser in ["none", "qleace", "leace"]:
                    run_training(width, depth, "vit", eraser, args.device)

    if args.net == "linear":
        for width in [1]: # Unused
            for depth in [1]: # Unused
                for eraser in ["none", "qleace", "leace"]:
                    run_training(width, depth, "linear", eraser, args.device)
    
    if args.net == "resnet":
        for width in [2, 4, 8]:
            for depth in [2, 4, 8]:
                for eraser in ["none", "qleace", "leace"]:
                    run_training(width, depth, "resnet", eraser, args.device)

if __name__ == "__main__":
    main()