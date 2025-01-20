from experiments.sweep_eraser import sweep_params

# for depth in sweep_params['mlp']['depths']: 
#     for width in sweep_params['mlp']['widths']: 
#         print(mlp_parameter_count(depth, width))

from argparse import ArgumentParser
from pathlib import Path

import wandb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from experiments.sweep_eraser import sweep_params
from experiments.scrape_wandb import scrape_data, DISPLAY_NAMES

import plotly.io as pio

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469


# We want to figure out whether the gap between LeNet and parameter-matched MLP is greater over an erased
# dataset than over a vanilla dataset. We will do this by retrieving the final loss and MDL metrics for all four
# specifications using the plotting helper fns.

def analyze_conv_gain(df: pd.DataFrame, out: Path):
    """Create plots for each network and activation function combination, with different erasers as lines on the same plot."""
    out.mkdir(exist_ok=True)

    all_dfs = ["cifar10", "cifarnet", "fake-cifar10", "fake-cifarnet"]

    # Colors for different erasers
    colors = px.colors.qualitative.Set1
    ordered_datasets = ["CIFAR-10", "CIFARNet"]

    reference_width = sweep_params["mlp"]["mup_width"]
    reference_depth = sweep_params["mlp"]["mup_depth"]
    width_depths = [
        (width, reference_depth) for width in sweep_params["mlp"]["widths"]
    ]
    widths_depth = [
        (reference_width, depth) for depth in sweep_params["mlp"]["depths"]
    ]

    # Create separate plot for each activation function
    def interleave(list1, list2) -> list:
        from itertools import chain
        return list(chain.from_iterable(zip(list1, list2))) + list1[len(list2):] + list2[len(list1):]

    net = DISPLAY_NAMES["mlp"]
    act = DISPLAY_NAMES["relu"]

    num_rows = max(len(width_depths), len(widths_depth))

    for col, item in enumerate([width_depths, widths_depth], 1):
        for row, (width, depth) in enumerate(item, 1):

            for dataset in ["CIFAR-10", "CIFARNet"]:
                unerased_mlp_df = df[
                    (df["dataset"] == dataset) &
                    (df["net"] == "MLP") &
                    (df["eraser"] == "Control") & 
                    (df["act"] == act) & 
                    (df['width'] == width) & 
                    (df['depth'] == depth)
                ]
                leaced_mlp_df = df[
                    (df["dataset"] == dataset) &
                    (df["net"] == "MLP") &
                    (df["eraser"] == "leace") & 
                    (df["act"] == act) & 
                    (df['width'] == width) & 
                    (df['depth'] == depth)
                ]
                second_erased_mlp_df = df[
                    (df["dataset"] == f"Erased {dataset}") &
                    (df["net"] == "MLP") &
                    (df["eraser"] == "Control") & 
                    (df["act"] == act) & 
                    (df['width'] == width) & 
                    (df['depth'] == depth)
                ]

                unerased_lenet_df = df[
                    (df["dataset"] == dataset) &
                    (df["net"] == "LeNet") &
                    (df["eraser"] == "Control") & 
                    (df["act"] == act) & 
                    (df['width'] == width) & 
                    (df['depth'] == depth)
                ]
                leaced_lenet_df = df[
                    (df["dataset"] == dataset) &
                    (df["net"] == "LeNet") &
                    (df["eraser"] == "leace") & 
                    (df["act"] == act) & 
                    (df['width'] == width) & 
                    (df['depth'] == depth)
                ]
                second_erased_lenet_df = df[
                    (df["dataset"] == f"Erased {dataset}") &
                    (df["net"] == "LeNet") &
                    (df["eraser"] == "Control") & 
                    (df["act"] == act) & 
                    (df['width'] == width) & 
                    (df['depth'] == depth)
                ]

                unerased_mean_mlp_data = unerased_mlp_df.groupby(["step"])["loss"].agg(["mean", "std"]).reset_index()
                unerased_mean_lenet_data = unerased_lenet_df.groupby(["step"])["loss"].agg(["mean", "std"]).reset_index()

                leaced_mean_mlp_data = leaced_mlp_df.groupby(["step"])["loss"].agg(["mean", "std"]).reset_index()
                leaced_mean_lenet_data = leaced_lenet_df.groupby(["step"])["loss"].agg(["mean", "std"]).reset_index()

                second_erased_mean_mlp_data = second_erased_mlp_df.groupby(["step"])["loss"].agg(["mean", "std"]).reset_index()
                second_erased_mean_lenet_data = second_erased_lenet_df.groupby(["step"])["loss"].agg(["mean", "std"]).reset_index()

                mlp_unerased_loss = unerased_mean_mlp_data["mean"].iloc[-1]
                lenet_unerased_loss = unerased_mean_lenet_data["mean"].iloc[-1]

                mlp_leaced_loss = leaced_mean_mlp_data["mean"].iloc[-1]
                lenet_leaced_loss = leaced_mean_lenet_data["mean"].iloc[-1]

                letnet_second_erased_loss = second_erased_mean_lenet_data["mean"].iloc[-1]
                mlp_second_erased_loss = second_erased_mean_mlp_data["mean"].iloc[-1]

                # all_dfs = ["cifar10", "cifarnet", "fake-cifar10", "fake-cifarnet"]
                if depth == 2:
                # if width in [128, 256, 512] and depth < 6:
                    print("width = ", width, "depth = ", depth)

                    def ratio_diff(lenet_unerased_loss, lenet_erased_loss, mlp_unerased_loss, mlp_erased_loss):
                        unerased_conv_gain = lenet_unerased_loss / mlp_unerased_loss
                        erased_conv_gain = lenet_erased_loss / mlp_erased_loss
                        # print(f"Conv gain: {unerased_conv_gain:.2f} -> {erased_conv_gain:.2f} ({erased_conv_gain / unerased_conv_gain:.2f}x)")
                        return erased_conv_gain  - unerased_conv_gain

                    print("Leaced gain diff: ", ratio_diff(lenet_unerased_loss, lenet_leaced_loss, mlp_unerased_loss, mlp_leaced_loss))
                    print("Second erased gain diff: ", ratio_diff(lenet_unerased_loss, letnet_second_erased_loss, mlp_unerased_loss, mlp_second_erased_loss))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--out", type=str, default="images/sweep_plots")
    parser.add_argument("--data", type=str, default="loss_curve.csv")
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    data_path = Path('data')
    
    data = data_path / f'conv_gain_{args.tag + "_" if args.tag else ""}{args.data}'
    out = data_path / args.out

    dfs = []
    for dataset in ["cifar10", "cifarnet", "fake-cifar10", "fake-cifarnet"]:
        # scrape_data(data, dataset, args.tag)
        dfs.append(pd.read_csv(data))
    df = pd.concat(dfs)
    
    analyze_conv_gain(df, out)

