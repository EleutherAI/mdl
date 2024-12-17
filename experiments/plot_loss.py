from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import json

import wandb
from wandb.apis.public import Run
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from experiments.sweep_eraser import sweep_params
from experiments.plot_mdl import DISPLAY_NAMES


def parse_run_params(run: Run) -> dict | None:
    """Parse run name parts into parameters."""
    
    parts: list[str] = run.name.split(' ')
    
    try:
        eraser, _, width_str, depth_str, seed_str, net = parts[:6]
        
        remaining_params = parts[6:] # unfortunately the order of these varies
        
        param_dict: dict[str, Any] = {
            'act': DISPLAY_NAMES['relu']
        }
        for param in remaining_params:
            if param.startswith('b1='):
                param_dict['b1'] = float(param.split('=')[1])
            elif param.startswith('lr='):
                param_dict['lr'] = float(param.split('=')[1])
            elif param.startswith('act='):
                param_dict['act'] = DISPLAY_NAMES[param.split('=')[1]]
            
        param_dict.update({
            'net_id': net,
            'seed': int(seed_str.split('=')[1]),
            'width': int(width_str.split('=')[1]),
            'depth': int(depth_str.split('=')[1]),
            'eraser': DISPLAY_NAMES[eraser],
            'net': DISPLAY_NAMES[net],
            # 'date': run.created_at
        })
        return param_dict
    except:
        return None

def parse_dataset(run: Run) -> str:
    """Parse dataset from run name."""
    try:
        with run.file('wandb-metadata.json').download(replace=True) as f:
            metadata = json.load(f)
        args = metadata['args']
    except:
        print(list(run.files()))
        return ''
    if not args:
        return ''

    if '24-11-21' not in run.name and '24-11-19' not in run.name:
        print(str(args))
    
    return 'cifarnet' if 'cifarnet' in str(args) else 'cifar10'


def scrape_data(filename: Path, dataset_str: str, tag: str):
    api = wandb.Api()
    runs = api.runs("eleutherai/mdl")

    latest_runs = {}
    for run in runs:
        if tag and tag not in run.name:
            continue
        if not tag:
            if '24-11-21' not in run.name and '24-11-19' not in run.name:
                if dataset_str == 'cifarnet' or 'resmlp' in run.name:
                    if not 'result' in run.name and not 'cifarnet' in run.name:
                        continue
                else:
                    continue
        dataset = parse_dataset(run)
        if dataset != dataset_str:
            continue

        params = parse_run_params(run)
        if not params:
            continue
        
        params['dataset'] = dataset_str
        
        param_key = tuple(sorted(params.items()))

        if param_key not in latest_runs or run.created_at > latest_runs[param_key].created_at:
            latest_runs[param_key] = run

    data = []
    for param_key, run in latest_runs.items():
        try:
            params = dict(param_key)
            # params = parse_run_params(run)
            # if not params:
            #     continue

            history = list(run.scan_history())
            if not history:
                print(f"No loss data found for run {run.name}")
                continue

            log2_max = int(history[-1]['_step']).bit_length()
            steps = [2 ** i for i in range(log2_max)]

            run_data = []
            for row in history:
                if row['_step'] in steps:
                    entry = {
                        **params,
                        'loss': row['val/loss'],
                        'step': row['_step'],
                        'run': run.name
                    }
                    run_data.append(entry)

            data.extend(run_data)

        except Exception as e:
            print(f"Error processing run {run.name}: {e}")
            continue

    pd.DataFrame(data).to_csv(filename, index=False)
    print(f"Saved loss curve to {filename}")

def plot_data(df: pd.DataFrame, out: Path):
    """Create plots for each network and eraser type with a line for each activation function.
    Seed data is plotted as markers and mean data as lines."""

    out.mkdir(exist_ok=True)

    colors = px.colors.qualitative.Set1

    ordered_erasers = ["Control", "LEACE", "QLEACE"]

    df = df.sort_values(["depth", "width"])

    for net_id in df["net_id"].unique():
        net = DISPLAY_NAMES[net_id]
        
        ordered_acts = ["ReLU", "GELU", "SwiGLU"] if net == "MLP" else ["ReLU"]

        reference_width = sweep_params[net_id]["mup_width"]
        reference_depth = sweep_params[net_id]["mup_depth"]
        net_depths = sweep_params[net_id]["depths"]
        net_widths = sweep_params[net_id]["widths"]
        width_depth_pairs = [
            (width, reference_depth) for width in net_widths
        ] + [
            (reference_width, depth) for depth in net_depths
        ]

        fig = make_subplots(
            rows=len(width_depth_pairs),
            cols=len(ordered_erasers),
            subplot_titles=ordered_erasers,
            vertical_spacing=0.01,
            horizontal_spacing=0.05,
        )
        fig.update_layout(
            title=f"Loss over 5 seeds ({net})",
            height=280 * len(width_depth_pairs),
            width=1200,
            showlegend=False,
        )

        fig.update_yaxes(matches="y1")

        for col, eraser in enumerate(ordered_erasers, 1):
            for row, (width, depth) in enumerate(width_depth_pairs, 1):
                # Update axis labels
                if col == 2:  # Only add once per row
                    fig.add_annotation(
                        text=f"Width={width}, Depth={depth}",
                        xref="paper",
                        yref="paper",
                        x=0.5,  # Position to the left of the plots
                        y=(1 - (row - 0.5) / len(width_depth_pairs)) + 0.05,  # Position above the row
                        showarrow=False,
                        font=dict(size=12),
                    )

                fig.update_yaxes(title_text="Loss (bits per sample)", row=row, col=1)
                fig.update_yaxes(showticklabels=False, row=row, col=2)
                fig.update_yaxes(showticklabels=False, row=row, col=3)

                if row == len(width_depth_pairs):
                    fig.update_xaxes(title_text="Step", row=row, col=col)
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

                # Plot data
                for act_idx, act in enumerate(ordered_acts):
                    data = df[
                        (df["eraser"] == eraser) & 
                        (df["act"] == act) & 
                        (df["net"] == net) & 
                        (df['width'] == width) & 
                        (df['depth'] == depth)
                    ]
                    mean_data = data.groupby(["step"])["loss"].agg(["mean", "std"]).reset_index()

                    fig.add_trace(
                        go.Scatter(
                            x=data["step"],
                            y=data["loss"],
                            mode="markers",
                            marker=dict(color=colors[act_idx], size=5, opacity=0.3),
                            name=act,
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=mean_data["step"],
                            y=mean_data["mean"],
                            mode="lines+markers",
                            line=dict(width=2),
                            name=act,
                            showlegend=row == 1 and col == 3,
                            marker=dict(color=colors[act_idx]),
                        ),
                        row=row,
                        col=col,
                    )

                    # Add legend for multiple activation functions
                    if len(ordered_acts) > 1:
                        fig.update_layout(
                            showlegend=True,
                            legend=dict(
                                title="Activation function",
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.09,
                            ),
                        )

        fig.write_image(out / f"{net}_loss.pdf", format="pdf")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--out", type=str, default="data/images/sweep_plots")
    parser.add_argument("--data", type=str, default="loss_curve.csv")
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    data, out = Path(f'{args.tag + "_" if args.tag else ""}{args.data}'), Path(args.out)

    scrape_data(data, args.dataset, args.tag)

    df = pd.read_csv(data)

    plot_data(df, out)
