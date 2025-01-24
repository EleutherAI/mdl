
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.io as pio
import plotly.express as px


from experiments.plot.plot_mdl import DISPLAY_NAMES, load_sweep_data
from experiments.sweep_eraser import sweep_params


def create_plots(df: pd.DataFrame, output_dir: Path, dataset: str):
    output_dir.mkdir(exist_ok=True, parents=True)
    colors = px.colors.qualitative.Plotly

    ordered_erasers = [
        e
        for e in ["QLEACE", "Iterative Erasure", "ALF-QLEACE", "LEACE", "Control",]
        if e in df["eraser"].unique()
    ]

    df = df[df["dataset"] == dataset].sort_values(["depth", "width"])
    # unique_nets = df["net_id"].unique()

    if dataset == "cifar10":
        unique_nets = ['mlp', 'resmlp', 'lenet', 'swin', 'convnext']
    else:
        unique_nets = ['mlp', 'resmlp', 'lenet']

    n_rows = len(unique_nets)

    titles = [DISPLAY_NAMES[net_id] for net_id in unique_nets]
    titles = [(title, "") for title in titles]
    
    # flatten into a single list
    titles = [item for sublist in titles for item in sublist]

    # Create subplot grid
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        horizontal_spacing=0.04,
        vertical_spacing=0.04,
        subplot_titles=titles,
        
    )

    for i in range(len(fig.layout.annotations)):
        # Skip empty titles (if you only want to move non-empty ones)
        if fig.layout.annotations[i].text != "":
            # To center the title over its subplot:
            fig.layout.annotations[i].x = 0.5  # 0.5 is center
            fig.layout.annotations[i].y += 0.006
            
            # If you want to ensure the title stays anchored to this position:
            fig.layout.annotations[i].xanchor = 'center'

    # Update overall layout
    fig.update_layout(
        height=300 * n_rows,
        width=1200,
        showlegend=True,
        legend=dict(
            title="Eraser type",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.95,
            bgcolor="rgba(255, 255, 255, 0.9)" 
        ),
    )

    # Process each network
    for row_idx, net_id in enumerate(unique_nets, 1):
        net = DISPLAY_NAMES[net_id]
        reference_width = sweep_params[net_id]["mup_width"]
        reference_depth = sweep_params[net_id]["mup_depth"]

        # Set up y-axes
        for col in [1, 2]:
            fig.update_yaxes(
                title_text="MDL (bits per sample)" if col == 1 else "",
                showticklabels=True,
                row=row_idx,
                col=col,
            )

        # Set up x-axes (only for bottom row)
        if row_idx == n_rows:
            for col, param in enumerate(["depth", "width"], 1):
                param_data = df[df["net_id"] == net_id][param].unique()
                if len(param_data) == 0:
                    continue

                # Configure ticks for this specific subplot
                tick_vals = [2**i for i in range(
                    int(np.log2(min(param_data))),
                    int(np.log2(max(param_data))) + 1,
                )]
                tick_text = [f"2<sup>{i}</sup>" for i in range(
                    int(np.log2(min(param_data))),
                    int(np.log2(max(param_data))) + 1,
                )]

                fig.update_xaxes(
                    type="log",
                    tickvals=tick_vals,
                    ticktext=tick_text if row_idx == n_rows else None,  # Only show labels on bottom row
                    showticklabels=(row_idx == n_rows),  # Only show labels on bottom row
                    title_text=param.title() if row_idx == n_rows else "",  # Only show title on bottom row
                    row=row_idx,
                    col=col,
                )

        net_df = df[(df["net"] == net) & (df["act"] == "ReLU")]

        # Plot data for each eraser type
        for eraser_idx, eraser in enumerate(ordered_erasers):
            data = net_df[net_df["eraser"] == eraser]
            if data.empty:
                continue

            mean_data = (
                data.groupby(["width", "depth"])["mdl"]
                .agg(["mean", "std"])
                .reset_index()
            )

            # Plot depth sweep (fixed width)
            depth_data = data[data["width"] == reference_width]
            mean_depth = mean_data[mean_data["width"] == reference_width]

            # Plot width sweep (fixed depth)
            width_data = data[data["depth"] == reference_depth]
            mean_width = mean_data[mean_data["depth"] == reference_depth]

            for col, (param_data, mean_param) in enumerate(
                [
                    (depth_data, mean_depth),
                    (width_data, mean_width),
                ],
                1,
            ):
                param = "depth" if col == 1 else "width"

                # Plot individual points
                fig.add_trace(
                    go.Scatter(
                        x=param_data[param],
                        y=param_data["mdl"],
                        mode="markers",
                        marker=dict(color=colors[eraser_idx], size=5, opacity=0.3),
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col,
                )

                # Plot mean line (only show legend for first row)
                fig.add_trace(
                    go.Scatter(
                        x=mean_param[param],
                        y=mean_param["mean"],
                        mode="lines+markers",
                        line=dict(width=2),
                        marker=dict(color=colors[eraser_idx]),
                        name=eraser,
                        showlegend=(row_idx == 1 and col == 1),
                    ),
                    row=row_idx,
                    col=col,
                )

    fig.write_image(output_dir / f"all_models_MDL_{dataset}.pdf", format="pdf")



def main():
    parser = ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/mnt/ssd-1/lucia/24-11-21"))
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--out", type=Path, default=Path("data/images/sweep_plots"))
    args = parser.parse_args()

    if args.dataset == "cifar10":
        assert "cifarnet" not in args.data.name

    print("Loading disk data into dataframe...")
    df = load_sweep_data(args.data)

    print("Creating plots...")
    create_plots(df, args.out, args.dataset)


if __name__ == "__main__":
    main()
