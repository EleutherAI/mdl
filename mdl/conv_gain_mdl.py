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
from experiments.plot_mdl import load_sweep_data

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

    

    # Colors for different erasers
    # colors = px.colors.qualitative.Set1
    # all_dfs = ["cifar10", "cifarnet", "fake-cifar10", "fake-cifarnet"]
    # ordered_datasets = ["CIFAR-10", "CIFARNet"]

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


            # df[(df["dataset"] == "fake-cifar10")] # & (df["net"] == "mlp")

            # df[(df["dataset"] == dataset) & (df["net_id"] == "mlp")] #  & (df["eraser"] == "control")


            for dataset in ["cifar10"]: #  "cifarnet"
                # import json
                # if dataset == "cifar10":
                #     with open('data/lenet_configs_32.json', 'r') as f:
                #         hyperparams = json.load(f)
                # else:
                #     with open('data/lenet_configs_64.json', 'r') as f:
                #         hyperparams = json.load(f)

                    
                unerased_mlp_df = df[
                    (df["dataset"] == dataset) &
                    (df["net_id"] == "mlp") &
                    (df["eraser"] == "Control") & 
                    (df["act"] == act) & 
                    (df['width'] == width) & 
                    (df['depth'] == depth)
                ]
                erased_mlp_df = df[
                    (df["dataset"] == f"fake-{dataset}") &
                    (df["net_id"] == "mlp") &
                    (df["eraser"] == "Control") & 
                    (df["act"] == act) & 
                    (df['width'] == width) & 
                    (df['depth'] == depth)
                ]

                unerased_lenet_df = df[
                    (df["dataset"] == dataset) &
                    (df["net_id"] == "lenet") &
                    (df["eraser"] == "Control") & 
                    (df["act"] == act) & 
                    (df['width'] == width) & 
                    (df['depth'] == depth)
                ]
                erased_lenet_df = df[
                    (df["dataset"] == f"fake-{dataset}") &
                    (df["net_id"] == "lenet") &
                    (df["eraser"] == "Control") & 
                    (df["act"] == act) & 
                    (df['width'] == width) & 
                    (df['depth'] == depth)
                ]

                if unerased_lenet_df.empty or erased_lenet_df.empty or unerased_mlp_df.empty or erased_mlp_df.empty:
                    print(width, depth, dataset)
                    print(unerased_lenet_df.empty, erased_lenet_df.empty, unerased_mlp_df.empty, erased_mlp_df.empty)
                    continue

                unerased_mean_mlp_mdl = unerased_mlp_df['mdl'].mean()
                unerased_mean_lenet_mdl = unerased_lenet_df['mdl'].mean()

                erased_mean_mlp_mdl = erased_mlp_df['mdl'].mean()
                erased_mean_lenet_mdl = erased_lenet_df['mdl'].mean()

                # all_dfs = ["cifar10", "cifarnet", "fake-cifar10", "fake-cifarnet"]

                print("width = ", width, "depth = ", depth)

                def ratio_diff(lenet_unerased_loss, lenet_erased_loss, mlp_unerased_loss, mlp_erased_loss):
                    # unerased_conv_gain = lenet_unerased_loss / mlp_unerased_loss
                    # erased_conv_gain = lenet_erased_loss / mlp_erased_loss
                    print(f"lenet_unerased_loss {lenet_unerased_loss}, lenet_erased_loss {lenet_erased_loss}, mlp_unerased_loss {mlp_unerased_loss} mlp_erased_loss {mlp_erased_loss}")
                    # print(f"Conv gain: {unerased_conv_gain:.2f} -> {erased_conv_gain:.2f} ({erased_conv_gain / unerased_conv_gain:.2f}x)")
                    print(f"Differences, unerased ", mlp_unerased_loss - lenet_unerased_loss, "difference, erased:", mlp_erased_loss - lenet_erased_loss)
                    print("Difference of differences", (mlp_erased_loss - lenet_erased_loss) - (mlp_unerased_loss - lenet_unerased_loss)) # how much loss you lose switching to lenet on unerased - how much loss you loss switching to lenet on erased
                    # return erased_conv_gain  - unerased_conv_gain
                    return (mlp_erased_loss - lenet_erased_loss) - (mlp_unerased_loss - lenet_unerased_loss)

                print("Erased gain diff: ", ratio_diff(unerased_mean_lenet_mdl, erased_mean_lenet_mdl, unerased_mean_mlp_mdl, erased_mean_mlp_mdl))

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
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/mnt/ssd-1/lucia/24-11-21"),
        help="Path to the directory containing .pth files.",
    )
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    data_path = Path('data')
    
    out = data_path / args.out

    df = load_sweep_data(args.data)
    
    analyze_conv_gain(df, out, args.tag)

