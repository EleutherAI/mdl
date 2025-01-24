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


def plot_data(df: pd.DataFrame, out: Path, dataset: str, tag: str):
    """Create plots for each network and activation function combination, with different erasers as lines on the same plot."""
    out.mkdir(exist_ok=True)

    df = df[df["dataset"] == dataset]

    # Colors for different erasers
    colors = px.colors.qualitative.Plotly
    ordered_erasers = ["Control", "LEACE", "QLEACE", "ALF-QLEACE"]

    for net_id in df["net_id"].unique():
        net = DISPLAY_NAMES[net_id]
        
        # Get each activation function used to train this network
        net_acts = df[df["net"] == net]["act"].unique()

        reference_width = sweep_params[net_id]["mup_width"]
        reference_depth = sweep_params[net_id]["mup_depth"]
        width_depths = [
            (width, reference_depth) for width in sweep_params[net_id]["widths"]
        ]
        widths_depth = [
            (reference_width, depth) for depth in sweep_params[net_id]["depths"]
        ]

        # Create separate plot for each activation function
        def interleave(list1, list2) -> list:
            from itertools import chain
            return list(chain.from_iterable(zip(list1, list2))) + list1[len(list2):] + list2[len(list1):]


        for act in net_acts:
            num_rows = max(len(width_depths), len(widths_depth))
            fig = make_subplots(
                rows=num_rows,
                cols=2,
                subplot_titles=[f"Width={w}, Depth={d}" for w, d in interleave(width_depths, widths_depth)],
                vertical_spacing=0.03,
                row_heights=[400] * num_rows,
            )

            fig.update_layout(
                title=f"Loss over 5 seeds ({net}, {act})",
                height=280 * num_rows,
                width=1200,
                showlegend=True,
                legend=dict(
                    title="Eraser type",
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                ),
            )

            # Match y-axes across subplots
            fig.update_yaxes(matches="y1")

            for col, item in enumerate([width_depths, widths_depth], 1):
                for row, (width, depth) in enumerate(item, 1):
                    fig.update_yaxes(title_text="Loss (bits per sample)", row=row, col=col)

                    if row == len(width_depths):
                        fig.update_xaxes(title_text="Epoch", row=row, col=col)
                    else:
                        fig.update_xaxes(showticklabels=False, row=row, col=col)

                    net_df = df[df["net"] == net]
                    fig.update_xaxes(
                        type="log",
                        row=row,
                        col=col,
                        tickvals=[
                            2**i
                            for i in range(
                                int(np.log2(min(net_df["step"]))),
                                int(np.log2(max(net_df["step"]))) + 1,
                            )
                        ],
                        ticktext=[
                            f"2<sup>{i}</sup>"
                            for i in range(
                                int(np.log2(min(net_df["step"]))),
                                int(np.log2(max(net_df["step"]))) + 1,
                            )
                        ],
                    )

                    # Plot all erasers for this configuration
                    for eraser_idx, eraser in enumerate(ordered_erasers):
                        data = df[
                            (df["eraser"] == eraser) & 
                            (df["act"] == act) & 
                            (df["net"] == net) & 
                            (df['width'] == width) & 
                            (df['depth'] == depth)
                        ]
                        data = data.sort_values("step")

                        mean_data = data.groupby(["step"])["loss"].agg(["mean", "std"]).reset_index()
                        mean_data = mean_data.sort_values("step")

                        # Plot individual runs as transparent lines
                        for seed in data["seed"].unique():
                            seed_data = data[data["seed"] == seed]
                            fig.add_trace(
                                go.Scatter(
                                    x=seed_data["step"],
                                    y=seed_data["loss"],
                                    mode="lines",
                                    marker=dict(color=colors[eraser_idx], size=5),
                                    opacity=0.3,
                                    name=f"{eraser} (seeds)",
                                    showlegend=False,
                                    legendgroup=eraser,
                                ),
                                row=row,
                                col=col,
                            )
                        # fig.add_trace(
                        #     go.Scatter(
                        #         x=data["step"],
                        #         y=data["loss"],
                        #         mode="lines+markers",
                        #         marker=dict(color=colors[eraser_idx], size=5, opacity=0.3),
                        #         name=f"{eraser} (seeds)",
                        #         showlegend=False,
                        #         legendgroup=eraser,
                        #     ),
                        #     row=row,
                        #     col=col,
                        # )

                        # Plot mean as a line
                        fig.add_trace(
                            go.Scatter(
                                x=mean_data["step"],
                                y=mean_data["mean"],
                                mode="lines+markers",
                                line=dict(width=2),
                                name=f"{eraser}",
                                legendgroup=eraser,
                                showlegend=row == 1 and col == 1,
                                marker=dict(color=colors[eraser_idx]),
                            ),
                            row=row,
                            col=col,
                        )

            # Save plot for this activation function
            fig.write_image(out / f"{net}_{act}_{dataset}{'_' + tag if tag else ''}_loss.pdf", format="pdf")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--out", type=str, default="images/sweep_plots")
    parser.add_argument("--data", type=str, default="loss_curve.csv")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--scrape", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    data_path = Path('data')
    
    data = data_path / f'{args.tag + "_" if args.tag else ""}{args.data}'
    out = data_path / args.out

    if args.scrape:
        scrape_data(data, args.dataset, args.tag)

    df = pd.read_csv(data)
    plot_data(df, out, args.dataset, args.tag)