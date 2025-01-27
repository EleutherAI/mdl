from pathlib import Path
from typing import Any
import json

import wandb
from wandb.apis.public import Run
import pandas as pd

DISPLAY_NAMES = {
    # nets
    "mlp": "MLP",
    "convnext": "ConvNeXt",
    "swin": "Swin",
    "resmlp": "ResMLP",
    "lenet": "LeNet",
    # erasers
    "control": "Control",
    "leace": "LEACE",
    "qleace": "QLEACE",
    "qleace2": "ALF-QLEACE",
    "alf_qleace": "ALF-QLEACE",
    "alf-qleace": "ALF-QLEACE",
    # activation functions
    "relu": "ReLU",
    "gelu": "GELU",
    "swiglu": "SwiGLU",
    # datasets
    "cifar-10": "CIFAR-10",
    "cifarnet": "CIFARNet",
    "fake-cifar10": "Erased CIFAR-10",
    "fake-cifarnet": "Erased CIFARNet",
}


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

    str_args = ' '.join(args)
    if '24-11-21' not in run.name and '24-11-19' not in run.name:
        print(str_args)

    if 'fake-leace-cifar10' in str_args:
        return 'fake-leace-cifar10'
    elif 'fake-leace-cifarnet' in str_args:
        return 'fake-leace-cifarnet'
    elif 'fake-leace-svhn' in str_args:
        return 'fake-leace-svhn'
    elif 'fake-cifar10' in str_args:
        return 'fake-cifar10'
    elif 'fake-cifarnet' in str_args:
        return 'fake-cifarnet'
    elif 'cifarnet' in str_args:
        return 'cifarnet'
    elif 'svhn' in str_args:
        return 'svhn'
    return 'cifar10' # Some runs have no dataset tagged


def scrape_data(filename: Path):
    api = wandb.Api(timeout=1000)
    runs = api.runs("eleutherai/mdl")

    latest_runs = {}
    for run in runs:
        if '24-11-21' not in run.name and '24-11-19' not in run.name and 'results' not in run.name:
            # if dataset_str == 'cifarnet' or 'resmlp' in run.name:
            #     if not 'result' in run.name and not 'cifarnet' in run.name:
            #         continue
            # else:
            continue

        dataset = parse_dataset(run)
        # if dataset != dataset_str:
            # continue

        params = parse_run_params(run)
        if not params:
            continue
        
        params['dataset'] = dataset
        
        param_key = tuple(sorted(params.items()))

        if param_key not in latest_runs or run.created_at > latest_runs[param_key].created_at:
            latest_runs[param_key] = run

    data = []
    for param_key, run in latest_runs.items():
        try:
            params = dict(param_key)

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


