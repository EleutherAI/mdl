from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from experiments.sweep_eraser import sweep_params
from experiments.plot.plot_mdl import load_sweep_data


def analyze_conv_gain(df: pd.DataFrame, out: Path, tag: str):
    """Create plots showing how convolutional advantage varies with width/depth."""
    out.mkdir(exist_ok=True)

    reference_width = sweep_params["mlp"]["mup_width"]
    reference_depth = sweep_params["mlp"]["mup_depth"]
    width_depths = [(width, reference_depth) for width in sweep_params["mlp"]["widths"]]
    widths_depth = [(reference_width, depth) for depth in sweep_params["mlp"]["depths"]]

    def diff_of_diffs(
        lenet_unerased_metric, 
        lenet_erased_metric, 
        mlp_unerased_metric, 
        mlp_erased_metric,
        lenet_unerased_std,
        lenet_erased_std,
        mlp_unerased_std,
        mlp_erased_std,
        lenet_unerased_n,
        lenet_erased_n,
        mlp_unerased_n,
        mlp_erased_n,
    ) -> tuple[float, float]:
        """The difference in the extent to which adding a convolution changes the metric on an erased and an unerased dataset"""
        did = (mlp_erased_metric - lenet_erased_metric) - (
            mlp_unerased_metric - lenet_unerased_metric
        )

        se_erased = np.sqrt(
            lenet_erased_std ** 2 / lenet_erased_n
            + mlp_erased_std ** 2 / mlp_erased_n
        )
        se_unerased = np.sqrt(
            lenet_unerased_std ** 2 / lenet_unerased_n
            + mlp_unerased_std ** 2 / mlp_unerased_n
        )

        se = np.sqrt(
            se_erased ** 2 + se_unerased ** 2
        )

        return did, se

    # Lists to store results
    width_results = []
    depth_results = []

    for dataset in ["cifar10"]: # "cifarnet"
        # Width sweep
        for width, depth in width_depths:
            data = {"lenet": {}, "mlp": {}}
            for net in ["lenet", "mlp"]:
                for eraser in ["Control", "LEACE", "QLEACE", "ALF-QLEACE"]:
                    data[net][eraser] = {}

                    data[net][eraser]["df"] = df[
                        (df["dataset"] == dataset)
                        & (df["net_id"] == net)
                        & (df["eraser"] == eraser)
                        & (df["act"] == "ReLU")
                        & (df["width"] == width)
                        & (df["depth"] == depth)
                    ]
                    data[net][eraser]["mean"] = data[net][eraser]["df"]["mdl"].mean()
                    data[net][eraser]["std"] = data[net][eraser]["df"]["mdl"].std()
                    data[net][eraser]["n"] = 10 # len(data[net][eraser]["df"]["mdl"])

                # Add fake dataset eraser under Iterative Erasure
                data[net]["Iterative Erasure"] = {}
                data[net]["Iterative Erasure"]["df"] = df[
                    (df["dataset"] == f"fake-{dataset}")
                    & (df["net_id"] == net)
                    & (df["eraser"] == "Control")
                    & (df["act"] == "ReLU")
                    & (df["width"] == width)
                    & (df["depth"] == depth)
                ]
                data[net]["Iterative Erasure"]["mean"] = data[net]["Iterative Erasure"][
                    "df"
                ]["mdl"].mean()
                data[net]["Iterative Erasure"]["std"] = data[net]["Iterative Erasure"][
                    "df"
                ]["mdl"].std()
                data[net]["Iterative Erasure"]["n"] = 10 # len(
                    # data[net]["Iterative Erasure"]["df"]["mdl"]
                # )


            for eraser in ["Control", "LEACE", "QLEACE", "ALF-QLEACE"]:
                if data['mlp'][eraser]["n"] == 0 or data['lenet'][eraser]["n"] == 0:
                    print(f"Skipping {eraser} {width} {depth} because n=0")
                    print('mlp', data['mlp'][eraser]["n"], 'lenet', data['lenet'][eraser]["n"])
                    continue

                data['mlp'][eraser]["did"], data['mlp'][eraser]["se"] = diff_of_diffs(
                    data['lenet']["Control"]["mean"],
                    data['lenet'][eraser]["mean"],
                    data['mlp']["Control"]["mean"],
                    data['mlp'][eraser]["mean"],
                    data['lenet']["Control"]["std"],
                    data['lenet'][eraser]["std"],
                    data['mlp']["Control"]["std"],
                    data['mlp'][eraser]["std"],
                    data['lenet']["Control"]["n"],
                    data['lenet'][eraser]["n"],
                    data['mlp']["Control"]["n"],
                    data['mlp'][eraser]["n"],
                )

            data['mlp']["Iterative Erasure"]["did"], data['mlp']["Iterative Erasure"]["se"] = diff_of_diffs(
                data['lenet']["Control"]["mean"],
                data['lenet']["Iterative Erasure"]["mean"],
                data['mlp']["Control"]["mean"],
                data['mlp']["Iterative Erasure"]["mean"],
                data['lenet']["Control"]["std"],
                data['lenet']["Iterative Erasure"]["std"],
                data['mlp']["Control"]["std"],
                data['mlp']["Iterative Erasure"]["std"],
                data['lenet']["Control"]["n"],
                data['lenet']["Iterative Erasure"]["n"],
                data['mlp']["Control"]["n"],
                data['mlp']["Iterative Erasure"]["n"],
            )

            width_results.append({
                "width": width,
                "leace_did": data["mlp"]["LEACE"]["did"],
                "qleace_did": data["mlp"]["QLEACE"]["did"],
                "iterative_erasure_did": data["mlp"]["Iterative Erasure"]["did"],
                "alf_qleace_did": data["mlp"]["ALF-QLEACE"]["did"],
                "leace_se": data["mlp"]["LEACE"]["se"],
                "qleace_se": data["mlp"]["QLEACE"]["se"],
                "iterative_erasure_se": data["mlp"]["Iterative Erasure"]["se"],
                "alf_qleace_se": data["mlp"]["ALF-QLEACE"]["se"],
            })

        # Depth sweep
        for width, depth in widths_depth:
            data = {"lenet": {}, "mlp": {}}

            for net in ["lenet", "mlp"]:
                for eraser in ["Control", "LEACE", "QLEACE", "ALF-QLEACE"]:
                    data[net][eraser] = {}
                    data[net][eraser]["df"] = df[
                        (df["dataset"] == dataset)
                        & (df["net_id"] == net)
                        & (df["eraser"] == eraser)
                        & (df["act"] == "ReLU")
                        & (df["width"] == width)
                        & (df["depth"] == depth)
                    ]
                    data[net][eraser]["mean"] = data[net][eraser]["df"]["mdl"].mean()
                    data[net][eraser]["std"] = data[net][eraser]["df"]["mdl"].std()
                    data[net][eraser]["n"] = 10 # len(data[net][eraser]["df"]["mdl"])

                # Add fake dataset eraser under Iterative Erasure
                data[net]["Iterative Erasure"] = {}
                data[net]["Iterative Erasure"]["df"] = df[
                    (df["dataset"] == f"fake-{dataset}")
                    & (df["net_id"] == net)
                    & (df["eraser"] == "Control")
                    & (df["act"] == "ReLU")
                    & (df["width"] == width)
                    & (df["depth"] == depth)
                ]
                data[net]["Iterative Erasure"]["mean"] = data[net]["Iterative Erasure"][
                    "df"
                ]["mdl"].mean()
                data[net]["Iterative Erasure"]["std"] = data[net]["Iterative Erasure"][
                    "df"
                ]["mdl"].std()
                data[net]["Iterative Erasure"]["n"] = 10 # len(
                    # data[net]["Iterative Erasure"]["df"]["mdl"]
                # )

            for eraser in ["Control", "LEACE", "QLEACE", "ALF-QLEACE"]:
                if data['mlp'][eraser]["n"] == 0:
                    print(f"Skipping mlp {eraser} {width} {depth} because n=0")
                    continue
                if data['lenet'][eraser]["n"] == 0:
                    print(f"Skipping lenet {eraser} {width} {depth} because n=0")
                    continue
                
                data['mlp'][eraser]["did"], data['mlp'][eraser]["se"] = diff_of_diffs(
                    data['lenet']["Control"]["mean"],
                    data['lenet'][eraser]["mean"],
                    data['mlp']["Control"]["mean"],
                    data['mlp'][eraser]["mean"],
                    data['lenet']["Control"]["std"],
                    data['lenet'][eraser]["std"],
                    data['mlp']["Control"]["std"],
                    data['mlp'][eraser]["std"],
                    data['lenet']["Control"]["n"],
                    data['lenet'][eraser]["n"],
                    data['mlp']["Control"]["n"],
                    data['mlp'][eraser]["n"],
                )

            data['mlp']["Iterative Erasure"]["did"], data['mlp']["Iterative Erasure"]["se"] = diff_of_diffs(
                data['lenet']["Control"]["mean"],
                data['lenet']["Iterative Erasure"]["mean"],
                data['mlp']["Control"]["mean"],
                data['mlp']["Iterative Erasure"]["mean"],
                data['lenet']["Control"]["std"],
                data['lenet']["Iterative Erasure"]["std"],
                data['mlp']["Control"]["std"],
                data['mlp']["Iterative Erasure"]["std"],
                data['lenet']["Control"]["n"],
                data['lenet']["Iterative Erasure"]["n"],
                data['mlp']["Control"]["n"],
                data['mlp']["Iterative Erasure"]["n"],
            )
            
            depth_results.append(
                {
                    "depth": depth,
                    "leace_did": data["mlp"]["LEACE"]["did"] if 'did' in data["mlp"]["LEACE"] else None,
                    "qleace_did": data["mlp"]["QLEACE"]["did"] if 'did' in data["mlp"]["QLEACE"] else None,
                    "iterative_erasure_did": data["mlp"]["Iterative Erasure"][
                        "did"
                    ] if 'did' in data["mlp"]["Iterative Erasure"] else None,
                    "alf_qleace_did": data["mlp"]["ALF-QLEACE"]["did"],
                    "leace_se": data["mlp"]["LEACE"]["se"] if 'se' in data["mlp"]["LEACE"] else None,
                    "qleace_se": data["mlp"]["QLEACE"]["se"] if 'se' in data["mlp"]["QLEACE"] else None,
                    "iterative_erasure_se": data["mlp"][
                        "Iterative Erasure"
                    ]["se"] if 'se' in data["mlp"]["Iterative Erasure"] else None,
                    "alf_qleace_se": data["mlp"]["ALF-QLEACE"][
                        "se"
                    ],
                }
            )

        width_df = pd.DataFrame(width_results)
        depth_df = pd.DataFrame(depth_results)

        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.04)

        # Width subplot (left)
        for method_idx, (method, name) in enumerate([
            ("qleace", "QLEACE"), ("iterative_erasure", "Iterative Erasure"), 
            ("alf_qleace", "ALF-QLEACE"),("leace", "LEACE")
        ]):
            width_df = width_df.sort_values('width')

            fig.add_trace(
                go.Scatter(
                    x=width_df["width"],
                    y=width_df[f"{method}_did"],
                    line_color=px.colors.qualitative.Plotly[method_idx],
                    name=name
                ),
                row=1,
                col=1,
            )

            x = width_df["width"].tolist() + width_df["width"].tolist()[::-1]
            y = (width_df[f"{method}_did"] + width_df[f"{method}_se"]).tolist() + (width_df[f"{method}_did"] - width_df[f"{method}_se"]).tolist()[::-1]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    fill='toself',
                    fillcolor=px.colors.qualitative.Plotly[method_idx],
                    opacity=0.1,
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name=f"{name} error band",
                ),
                row=1,
                col=1,
            )

        # Depth subplot (right)  
        for method_idx, (method, name) in enumerate([
            ("qleace", "QLEACE"), ("iterative_erasure", "Iterative Erasure"), 
            ("alf_qleace", "ALF-QLEACE"),("leace", "LEACE")
        ]):
            depth_df = depth_df.sort_values('depth')

            fig.add_trace(
                go.Scatter(
                    x=depth_df["depth"],
                    y=depth_df[f"{method}_did"],
                    line_color=px.colors.qualitative.Plotly[method_idx],
                    name=name,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

            x = depth_df["depth"].tolist() + depth_df["depth"].tolist()[::-1]
            y = (depth_df[f"{method}_did"] + depth_df[f"{method}_se"]).tolist() + (depth_df[f"{method}_did"] - depth_df[f"{method}_se"]).tolist()[::-1]            
            
            # Error bands
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    fill='toself',
                    fillcolor=px.colors.qualitative.Plotly[method_idx],
                    opacity=0.1,
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name=f"{name} error band"
                ),
                row=1,
                col=2
            )

     
        # Update layout
        fig.update_layout(
            height=350,
            width=1000,
            legend=dict(
                yanchor="middle",
                y=0.66,
                xanchor="left",
                x=0.55,  # Position legend just outside the right edge of plots
                bgcolor='rgba(255,255,255,0.7)'
            ),
            margin=dict(l=20, r=20, t=30, b=50),
        )

        fig.update_xaxes(
            title_text="Width",
            type="log",
            tickvals=[
                2**i
                for i in range(
                    int(np.log2(min(width_df["width"]))),
                    int(np.log2(max(width_df["width"]))) + 1,
                )
            ],
            ticktext=[
                f"2<sup>{i}</sup>"
                for i in range(
                    int(np.log2(min(width_df["width"]))),
                    int(np.log2(max(width_df["width"]))) + 1,
                )
            ],
            row=1,
            col=1,
        )
        fig.update_xaxes(
            title_text="Depth",
            type="log",
            tickvals=[
                2**i
                for i in range(
                    int(np.log2(min(depth_df["depth"]))),
                    int(np.log2(max(depth_df["depth"]))) + 1,
                )
            ],
            ticktext=[
                f"2<sup>{i}</sup>"
                for i in range(
                    int(np.log2(min(depth_df["depth"]))),
                    int(np.log2(max(depth_df["depth"]))) + 1,
                )
            ],
            row=1,
            col=2,
        )

        fig.update_yaxes(
            title_text="Difference in differences",
            range=[
                0,
                max(width_df["leace_did"].max(), width_df["qleace_did"].max()) * 1.1,
            ],
            row=1,
            col=1,
        )

        fig.update_yaxes(
            range=[
                0,
                max(width_df["leace_did"].max(), width_df["qleace_did"].max()) * 1.1,
            ],
            showticklabels=False,
            row=1,
            col=2,
        )

        fig.write_image(out / f"combined_did{'_' + tag if tag else ''}_{dataset}.pdf")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/mnt/ssd-1/lucia/24-11-21"),
        help="Path to the directory containing .pth files.",
    )
    parser.add_argument("--out", type=Path, default="data/images/sweep_plots")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    df = load_sweep_data(args.data)

    analyze_conv_gain(df, args.out, args.tag)
