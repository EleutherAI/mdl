import json
import itertools

from experiments.sweep_eraser import sweep_params
from mdl.lenet_probe import LeNetConfig


def mlp_parameter_count(num_layers, hidden_size, input_size = 3072, num_classes = 10):
    count = input_size * hidden_size + hidden_size # Input W+b

    for _ in range(num_layers - 1):
        count += hidden_size * hidden_size + hidden_size # Hidden W+b

    count += hidden_size * num_classes + num_classes # Output W+b

    return count


def lenet_parameter_count(cfg):
    def count_conv_params(in_channels, out_channels, kernel_size):
        return (in_channels * kernel_size * kernel_size * out_channels) + out_channels

    def count_linear_params(in_features, out_features):
        return (in_features * out_features) + out_features

    def conv_output_size(size, kernel_size): 
        """feature map sizes after each convolution and pooling"""
        return (size - kernel_size) + 1

    # Extract config values
    conv_hidden_size_1, conv_hidden_size_2 = cfg.conv_hidden_sizes
    fc_hidden_size_1, fc_hidden_size_2 = cfg.fc_hidden_sizes
    kernel_sizes = cfg.kernel_sizes
    
    # First conv + MaxPool
    feature_map_size = conv_output_size(cfg.image_size, kernel_sizes[0])
    feature_map_size = feature_map_size // 2
    
    # Second conv + MaxPool
    feature_map_size = conv_output_size(feature_map_size, kernel_sizes[1])
    feature_map_size = feature_map_size // 2
    
    # Count parameters for each layer
    conv1_params = count_conv_params(cfg.num_channels, conv_hidden_size_1, kernel_sizes[0])
    conv2_params = count_conv_params(conv_hidden_size_1, conv_hidden_size_2, kernel_sizes[1])
    
    flattened_size = conv_hidden_size_2 * feature_map_size * feature_map_size
    fc1_params = count_linear_params(flattened_size, fc_hidden_size_1)
    fc2_params = count_linear_params(fc_hidden_size_1, fc_hidden_size_2)
    fc3_params = count_linear_params(fc_hidden_size_2, cfg.num_labels)
    
    return conv1_params + conv2_params + fc1_params + fc2_params + fc3_params


def find_closest_config(target_params: int, image_size: int) -> tuple[list[int], list[int], int]:
    possible_conv_sizes = [32, 40, 48, 64, 128]
    possible_fc_sizes = [64, 128, 256, 512, 1024, 2048]
    kernel_sizes = [5, 5]
    num_channels = 3
    num_labels = 10
    
    best_diff = float('inf')
    best_config = None
    
    for conv1, conv2 in itertools.product(possible_conv_sizes, repeat=2):
        for fc1, fc2 in itertools.product(possible_fc_sizes, repeat=2):
            cfg = LeNetConfig(
                image_size=image_size,
                num_channels=num_channels,
                conv_hidden_sizes=[conv1, conv2],
                fc_hidden_sizes=[fc1, fc2],
                kernel_sizes=kernel_sizes,
                num_labels=num_labels
            )
            
            params = lenet_parameter_count(cfg)
            diff = abs(params - target_params)
            
            if diff < best_diff:
                best_diff = diff
                best_config = ([conv1, conv2], [fc1, fc2], params)
    
    if best_config is None:
        raise ValueError("No valid configuration found")
    
    return best_config


def main():    
    # Generate configurations for both image sizes
    widths, depths = sweep_params['mlp']['widths'], sweep_params['mlp']['depths']

    configs_32 = {}
    configs_64 = {}

    for depth in depths:
        for width in widths:
            mlp_params = mlp_parameter_count(depth, width, input_size=3072)  # 32x32x3
            conv_32, fc_32, actual_32 = find_closest_config(mlp_params, 32)
            
            mlp_params_64 = mlp_parameter_count(depth, width, input_size=12288)  # 64x64x3
            conv_64, fc_64, actual_64 = find_closest_config(mlp_params_64, 64)
            
            configs_32[f"{depth}_{width}"] = {
                'conv_hidden_sizes': conv_32,
                'fc_hidden_sizes': fc_32,
                'params': actual_32,
                'target_params': mlp_params
            }
            
            configs_64[f"{depth}_{width}"] = {
                'conv_hidden_sizes': conv_64,
                'fc_hidden_sizes': fc_64,
                'params': actual_64,
                'target_params': mlp_params_64
            }

    with open('data/lenet_configs_32.json', 'w') as f:
        json.dump(configs_32, f, indent=2)

    print(configs_64)
    with open('data/lenet_configs_64.json', 'w') as f:
        json.dump(configs_64, f, indent=2)


if __name__ == "__main__":
    main()