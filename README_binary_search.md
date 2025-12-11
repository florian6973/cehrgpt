# Binary Search for Optimal Model Depth

This directory contains scripts to perform binary search for the optimal number of layers (depth) to match a target parameter count for different width values (hidden sizes).

## Scripts Overview

### 1. `binary_search_depth.py`
Basic binary search script that tests all width values (multiples of 2*n_heads) for a single target parameter count.

### 2. `binary_search_depth_optimized.py`
Optimized version that:
- Uses parameter estimation to reduce the number of model runs
- Supports multiple target parameter counts
- Includes calibration for better estimation accuracy
- Provides detailed summaries and tables

### 3. `run_binary_search_single_width.py`
Simplified script for testing a single width value, useful for debugging and quick tests.

## Usage

### Basic Usage (Single Target Parameter Count)

```bash
# Run binary search for all width values with target of 100M parameters
python binary_search_depth.py --target_params 100000000 --cuda_visible_devices 1

# Run with custom settings
python binary_search_depth.py \
    --target_params 50000000 \
    --num_heads 12 \
    --min_layers 1 \
    --max_layers 50 \
    --timeout_seconds 180 \
    --cuda_visible_devices 0
```

### Optimized Usage (Multiple Target Parameter Counts)

```bash
# Test multiple target parameter counts
python binary_search_depth_optimized.py \
    --target_params 50000000 100000000 200000000 \
    --cuda_visible_devices 1

# Skip calibration for faster runs (uses default estimation)
python binary_search_depth_optimized.py \
    --target_params 100000000 \
    --skip_calibration \
    --cuda_visible_devices 1
```

### Single Width Testing

```bash
# Test a specific width value
python run_binary_search_single_width.py \
    --width 768 \
    --target_params 100000000 \
    --cuda_visible_devices 1

# Test with custom parameters
python run_binary_search_single_width.py \
    --width 1024 \
    --target_params 50000000 \
    --num_heads 16 \
    --min_layers 5 \
    --max_layers 30 \
    --cuda_visible_devices 0
```

## Width Values

The scripts automatically generate width values that are multiples of `2 * num_heads`:
- With `num_heads=8`: [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320]
- With `num_heads=12`: [24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480]

## Output Files

### JSON Results
- `binary_search_results_target_<target_params>.json` - Results for single target
- `binary_search_results_optimized.json` - Results for multiple targets
- `binary_search_width_<width>_target_<target_params>.json` - Single width results

### Example Output Structure
```json
{
  "width": 768,
  "optimal_layers": 12,
  "actual_params": 98765432,
  "target_params": 100000000,
  "difference": 1234568,
  "percentage_diff": 1.23
}
```

## Algorithm Details

### Binary Search Process
1. **Estimation**: Calculate estimated layers needed for target parameters
2. **Initial Test**: Run model with estimated layers
3. **Binary Search**: Narrow down the optimal layer count
4. **Fine-tuning**: Test nearby configurations for best match

### Parameter Estimation
The scripts use a standard transformer parameter calculation:
- **Embeddings**: `vocab_size * hidden_size + max_position_embeddings * hidden_size`
- **Per Layer**: 
  - Attention: `4 * hidden_size * hidden_size`
  - Feed-forward: `2 * hidden_size * (4 * hidden_size)`
  - Layer Norm: `4 * hidden_size`

### Optimization Features
- **Caching**: Avoid re-running tested configurations
- **Calibration**: Adjust estimation based on actual model runs
- **Early Termination**: Stop when exact match is found
- **Error Handling**: Gracefully handle failed model runs

## Configuration Options

### Model Parameters
- `--num_heads`: Number of attention heads (default: 8)
- `--max_position_embeddings`: Maximum sequence length (default: 2048)
- `--max_tokens_per_batch`: Batch size in tokens (default: 2048)

### Search Parameters
- `--min_layers`: Minimum layers to try (default: 1)
- `--max_layers`: Maximum layers to try (default: 100)
- `--timeout_seconds`: Timeout for each model run (default: 120)

### Hardware
- `--cuda_visible_devices`: GPU device to use (default: "1")
- `--memory_fraction`: GPU memory fraction (default: 1.0)

## Example Results

### Summary Table Output
```
====================================================================================================
SUMMARY TABLE
====================================================================================================
Target Params   Width  Layers  Actual Params   Diff         % Diff  
----------------------------------------------------------------------------------------------------
100,000,000     768    12      98,765,432      1,234,568    1.23%   
100,000,000     1024   8       99,876,543      123,457      0.12%   
100,000,000     1280   6       100,123,456     123,456      0.12%   
```

### Best Configurations Found
- **Width 768**: 12 layers → 98.8M parameters (1.2% diff)
- **Width 1024**: 8 layers → 99.9M parameters (0.1% diff)
- **Width 1280**: 6 layers → 100.1M parameters (0.1% diff)

## Tips for Efficient Usage

1. **Start Small**: Test with a single width first using `run_binary_search_single_width.py`
2. **Use Calibration**: Let the optimized script calibrate parameter estimation
3. **Adjust Timeouts**: Increase timeout for larger models
4. **Monitor GPU**: Use GPU monitoring to ensure sufficient memory
5. **Check Results**: Review JSON outputs for detailed analysis

## Troubleshooting

### Common Issues
- **Model Run Fails**: Check GPU memory and reduce batch size
- **Timeout Errors**: Increase `--timeout_seconds`
- **Poor Estimation**: Use calibration (don't skip with `--skip_calibration`)
- **Memory Issues**: Reduce `--max_tokens_per_batch` or use smaller width values

### Debug Mode
For debugging, use the single width script with verbose output:
```bash
python run_binary_search_single_width.py --width 768 --target_params 100000000 --timeout_seconds 300
```
