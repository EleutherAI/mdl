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

# We have all of vanilla cifar and cifarnet with MLPs
# We have all of erased cifar and cifarnet with MLPs
# We have like one erased CIFAR10 run with LeNet at best (running now)
# We have no erased cifarnet runs with LeNet (running now)
# We have no vanilla CIFAR10 runs with LeNet (running now)
# We have no vanilla cifarnet runs with LeNet (running now)

# We should write code in advance to get these results as they come in

def analyze_conv_gain(df: pd.DataFrame, out: Path, tag: str):
    """Create plots for each network and activation function combination, with different erasers as lines on the same plot."""
    out.mkdir(exist_ok=True)

    all_dfs = ["cifar10", "cifarnet", "fake-cifar10", "fake-cifarnet"]

    # Colors for different erasers
    colors = px.colors.qualitative.Set1
    ordered_datasets = ["CIFAR-10", "CIFARNet"]

    reference_width = sweep_params["MLP"]["mup_width"]
    reference_depth = sweep_params["MLP"]["mup_depth"]
    width_depths = [
        (width, reference_depth) for width in sweep_params["MLP"]["widths"]
    ]
    widths_depth = [
        (reference_width, depth) for depth in sweep_params["MLP"]["depths"]
    ]

    # Create separate plot for each activation function
    def interleave(list1, list2) -> list:
        from itertools import chain
        return list(chain.from_iterable(zip(list1, list2))) + list1[len(list2):] + list2[len(list1):]

    net = DISPLAY_NAMES["mlp"]
    act = DISPLAY_NAMES["relu"]

    num_rows = max(len(width_depths), len(widths_depth))
    # fig = make_subplots(
    #     rows=num_rows,
    #     cols=2,
    #     subplot_titles=[f"Width={w}, Depth={d}" for w, d in interleave(width_depths, widths_depth)],
    #     vertical_spacing=0.03,
    #     row_heights=[400] * num_rows,
    # )

    # fig.update_layout(
    #     title=f"Loss over 5 seeds ({net}, {act})",
    #     height=280 * num_rows,
    #     width=1200,
    #     showlegend=True,
    #     legend=dict(
    #         title="Eraser type",
    #         yanchor="top",
    #         y=0.99,
    #         xanchor="right",
    #         x=0.99,
    #     ),
    # )

    # # Match y-axes across subplots
    # fig.update_yaxes(matches="y1")

    for col, item in enumerate([width_depths, widths_depth], 1):
        for row, (width, depth) in enumerate(item, 1):
            # fig.update_yaxes(title_text="Loss (bits per sample)", row=row, col=col)

            # if row == len(width_depths):
            #     fig.update_xaxes(title_text="Epoch", row=row, col=col)
            # else:
            #     fig.update_xaxes(showticklabels=False, row=row, col=col)


            for dataset in ["CIFAR-10", "CIFARNet"]:
                unerased_mlp_df = df[
                    (df["dataset"] == dataset) &
                    (df["net"] == "MLP") &
                    (df["eraser"] == "Control") & 
                    (df["act"] == act) & 
                    (df['width'] == width) & 
                    (df['depth'] == depth)
                ]
                erased_mlp_df = df[
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
                erased_lenet_df = df[
                    (df["dataset"] == f"Erased {dataset}") &
                    (df["net"] == "LeNet") &
                    (df["eraser"] == "Control") & 
                    (df["act"] == act) & 
                    (df['width'] == width) & 
                    (df['depth'] == depth)
                ]


                breakpoint()

                unerased_mean_mlp_data = unerased_mlp_df.groupby(["step"])["loss"].agg(["mean", "std"]).reset_index()
                unerased_mean_lenet_data = unerased_lenet_df.groupby(["step"])["loss"].agg(["mean", "std"]).reset_index()

                erased_mean_mlp_data = erased_mlp_df.groupby(["step"])["loss"].agg(["mean", "std"]).reset_index()
                erased_mean_lenet_data = erased_lenet_df.groupby(["step"])["loss"].agg(["mean", "std"]).reset_index()

                mlp_unerased_loss = unerased_mean_mlp_data["mean"].iloc[-1]
                lenet_unerased_loss = unerased_mean_lenet_data["mean"].iloc[-1]

                letnet_erased_loss = erased_mean_lenet_data["mean"].iloc[-1]
                mlp_erased_loss = erased_mean_mlp_data["mean"].iloc[-1]

                # all_dfs = ["cifar10", "cifarnet", "fake-cifar10", "fake-cifarnet"]

                print("width = ", width, "depth = ", depth)

                def ratio_diff(lenet_unerased_loss, lenet_erased_loss, mlp_unerased_loss, mlp_erased_loss):
                    unerased_conv_gain = lenet_unerased_loss / mlp_unerased_loss
                    erased_conv_gain = lenet_erased_loss / mlp_erased_loss
                    print(f"Conv gain: {unerased_conv_gain:.2f} -> {erased_conv_gain:.2f} ({erased_conv_gain / unerased_conv_gain:.2f}x)")
                    return erased_conv_gain  - unerased_conv_gain

                print("Erased gain diff: ", ratio_diff(lenet_unerased_loss, letnet_erased_loss, mlp_unerased_loss, mlp_erased_loss))

            # fig.update_xaxes(
            #     type="log",
            #     row=row,
            #     col=col,
            #     tickvals=[
            #         2**i
            #         for i in range(
            #             int(np.log2(min(net_df["step"]))),
            #             int(np.log2(max(net_df["step"]))) + 1,
            #         )
            #     ],
            #     ticktext=[
            #         f"2<sup>{i}</sup>"
            #         for i in range(
            #             int(np.log2(min(net_df["step"]))),
            #             int(np.log2(max(net_df["step"]))) + 1,
            #         )
            #     ],
            # )

            # # Plot all erasers for this configuration
            # for eraser_idx, eraser in enumerate(ordered_datasets):
            #     # TODO use difference between erased and real data points
            #     data = df[
            #         (df["eraser"] == eraser) & 
            #         (df["act"] == act) & 
            #         (df["net"] == net) & 
            #         (df['width'] == width) & 
            #         (df['depth'] == depth)
            #     ]

            #     mean_data = data.groupby(["step"])["loss"].agg(["mean", "std"]).reset_index()

            #     # Plot individual runs as scattered points
            #     fig.add_trace(
            #         go.Scatter(
            #             x=data["step"],
            #             y=data["loss"],
            #             mode="markers",
            #             marker=dict(color=colors[eraser_idx], size=5, opacity=0.3),
            #             name=f"{eraser} (seeds)",
            #             showlegend=False,
            #             legendgroup=eraser,
            #         ),
            #         row=row,
            #         col=col,
            #     )

            #     # Plot mean as a line
            #     fig.add_trace(
            #         go.Scatter(
            #             x=mean_data["step"],
            #             y=mean_data["mean"],
            #             mode="lines+markers",
            #             line=dict(width=2),
            #             name=f"{eraser}",
            #             legendgroup=eraser,
            #             showlegend=row == 1 and col == 1,
            #             marker=dict(color=colors[eraser_idx]),
            #         ),
            #         row=row,
            #         col=col,
            #     )

        # fig.write_image(out / f"{net}_{act}_{dataset}{'_' + tag if tag else ''}_loss.pdf", format="pdf")

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
    
    analyze_conv_gain(df, out, args.tag)

