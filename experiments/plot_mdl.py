import pandas as pd
from pathlib import Path
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path


def load_sweep_data(data_path: Path) -> pd.DataFrame:
    """Load and parse sweep data files into a DataFrame."""
    records = []
    
    for file in data_path.glob("*.pth"):
        # Parse filename
        net_type, width, depth, eraser, _ = file.stem.split('_')
        width = int(width.split('=')[1])
        depth = int(depth.split('=')[1])
        
        # Load data and create records
        data = torch.load(file)
        for eraser_name, mdls in data.items():
            for seed, mdl_result in enumerate(mdls):
                records.append({
                    'model': net_type,
                    'width': width,
                    'depth': depth,
                    'eraser': eraser_name,
                    'seed': seed,
                    'mdl': mdl_result.mdl,
                    'ce_curve': mdl_result.ce_curve,
                    'sample_sizes': mdl_result.sample_sizes,
                    'total_trials': mdl_result.total_trials
                })

    return pd.DataFrame(records)


def create_plots(df: pd.DataFrame, output_dir: Path):
    """Create plots for each network type."""
    output_dir.mkdir(exist_ok=True)

    colors = px.colors.qualitative.Set1

    eraser_types = df['eraser'].unique()

    mean_df = df.groupby(['model', 'width', 'depth', 'eraser'])['mdl'].agg(['mean', 'std']).reset_index()

    for net_type in df['model'].unique():
        fig = make_subplots(
            rows=len(eraser_types), cols=2,
            subplot_titles=[f"{eraser.upper()} - Depth-wise" 
                            if i % 2 == 1 
                            else f"{eraser.upper()} - Width-wise" 
                            for eraser in eraser_types for i in range(2)
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
            row_heights=[400] * len(eraser_types)
        )
        fig.update_layout(
            title=f"{net_type} Network Analysis",
            height=300 * len(eraser_types),
            width=1200,
            showlegend=False,
        )

        net_mean_df = mean_df[mean_df['model'] == net_type]
        net_df = df[df['model'] == net_type]

        widths = sorted(net_mean_df['width'].unique())
        depths = sorted(net_mean_df['depth'].unique())
        reference_width = widths[0]
        reference_depth = depths[0]
        
        for row, eraser in enumerate(eraser_types, 1):
            fig.update_xaxes(title_text="Depth", row=row, col=1)
            fig.update_xaxes(title_text="Width", row=row, col=2)
            fig.update_yaxes(title_text="MDL Score", row=row, col=1)
            fig.update_yaxes(title_text="MDL Score", row=row, col=2)

            eraser_mean_df = net_mean_df[net_mean_df['eraser'] == eraser]
            eraser_df = net_df[net_df['eraser'] == eraser]
            if eraser_mean_df.empty:
                breakpoint()
                continue

            mean_depth = eraser_mean_df[eraser_mean_df['width'] == reference_width]
            depth_seeds = eraser_df[eraser_df['width'] == reference_width]

            fig.add_trace(
                go.Scatter(
                    x=depth_seeds['depth'],
                    y=depth_seeds['mdl'],
                    mode='markers',
                    marker=dict(color=colors[0], size=5, opacity=0.3),
                    name=f'{eraser} (seeds)',
                    showlegend=False,
                ),
                row=row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=mean_depth['depth'],
                    y=mean_depth['mean'],
                    mode='lines+markers',
                    name=f'{eraser} (mean)',
                    line=dict(color=colors[0], width=2),
                ),
                row=row, col=1
            )

            # Width Analysis (right column)
            width_data = eraser_mean_df[eraser_mean_df['depth'] == reference_depth]
            width_seeds = eraser_df[eraser_df['depth'] == reference_depth]
            
            fig.add_trace(
                go.Scatter(
                    x=width_seeds['width'],
                    y=width_seeds['mdl'],
                    mode='markers',
                    marker=dict(color=colors[1], size=5, opacity=0.3),
                    name=f'{eraser} (seeds)',
                    showlegend=False
                ),
                row=row, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=width_data['width'],
                    y=width_data['mean'],
                    mode='lines+markers',
                    name=f'{eraser} (mean)',
                    line=dict(color=colors[1], width=2),
                ),
                row=row, col=2
            )

        fig.write_image(output_dir / f"{net_type}_mdl_analysis.pdf", format='pdf')
    
def main():
    data_path = Path("/mnt/ssd-1/lucia/results")
    output_dir = Path("data/images/sweep_plots")
    
    print("Loading disk data into dataframe...")
    df = load_sweep_data(data_path)

    print("Creating plots...")
    create_plots(df, output_dir)

if __name__ == "__main__":
    main()