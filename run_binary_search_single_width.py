#!/usr/bin/env python3
"""
Script to run binary search for optimal depth (number of layers) 
to match a target parameter count for a single width value.

Usage:
    python run_binary_search_single_width.py --width 768 --target_params 100000000
"""

import os
import argparse
import subprocess
import time
import threading
import sys
from typing import Dict, List, Tuple, Optional
import json

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    print("Warning: pynvml not available. Install with: pip install nvidia-ml-py")
    NVML_AVAILABLE = False

def get_gpu_memory_usage(gpu_id: int = 0) -> Dict[str, int]:
    """Get current GPU memory usage in MB."""
    if not NVML_AVAILABLE:
        return {"used": 0, "total": 0, "free": 0}
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "used": info.used // 1024 // 1024,  # Convert to MB
            "total": info.total // 1024 // 1024,
            "free": info.free // 1024 // 1024
        }
    except Exception as e:
        print(f"Error getting GPU memory info: {e}")
        return {"used": 0, "total": 0, "free": 0}

def monitor_vram_async(gpu_id: int = 0, interval: float = 1.0):
    """Start VRAM monitoring in a separate thread and return the thread."""
    measurements = []
    
    def monitor():
        while True:
            try:
                memory_info = get_gpu_memory_usage(gpu_id)
                memory_info["timestamp"] = time.time()
                measurements.append(memory_info)
                time.sleep(interval)
            except KeyboardInterrupt:
                break
    
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return thread, measurements

def run_and_capture(command, timeout_seconds=60):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        text=True,
        shell=True,
        executable="/bin/bash",
    )

    output = []
    start_time = time.time()

    try:
        while True:
            # Check for timeout
            if time.time() - start_time > timeout_seconds:
                print(f"\nProcess timed out after {timeout_seconds} seconds. Terminating...")
                process.terminate()
                process.wait(timeout=5)  # Give it 5 seconds to terminate gracefully
                if process.poll() is None:
                    process.kill()  # Force kill if still running
                return -5, ''.join(output)
            
            # Try to read with a short timeout to allow checking the main timeout
            try:
                char = process.stdout.read(1)  # Read one character at a time
                if char == '' and process.poll() is not None:
                    break
                if char:
                    sys.stdout.write(char)     # Print to console immediately
                    sys.stdout.flush()
                    output.append(char)
                    if " 10/" in ''.join(output):
                        print(f"Process killed after 10 steps. Terminating...")
                        process.terminate()
                        process.wait(timeout=5)  # Give it 5 seconds to terminate gracefully
                        if process.poll() is None:
                            process.kill()  # Force kill if still running
                        return -5, ''.join(output)
            except (IOError, OSError):
                # Handle case where process has ended
                if process.poll() is not None:
                    break
                time.sleep(0.1)  # Small delay before retrying
                
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Terminating...")
        process.terminate()
        process.wait(timeout=5)
        if process.poll() is None:
            process.kill()

    process.wait()
    return process.returncode, ''.join(output)

def start_monitoring(gpu_id: int = 0, interval: float = 1.0):
    if NVML_AVAILABLE:
        print(f"Starting VRAM monitoring for GPU {gpu_id} using pynvml...")
        monitor_thread, measurements = monitor_vram_async(gpu_id=gpu_id, interval=interval)
    else:
        print(f"Starting VRAM monitoring for GPU {gpu_id} using nvidia-smi...")
        monitor_thread, measurements = monitor_vram_async(gpu_id=gpu_id, interval=interval)
    time.sleep(1)  # Give monitoring a moment to start
    return monitor_thread, measurements

def get_n_parameters(result_text: str):
    txt = "Number of parameters:"
    n_parameters = result_text.strip().split(txt)[1].split(',')[0]
    print(f"Number of parameters: {n_parameters}")
    return int(n_parameters)

def estimate_parameters(hidden_size: int, num_layers: int, num_heads: int, vocab_size: int = 32000, max_position_embeddings: int = 2048) -> int:
    """
    Estimate the number of parameters for a transformer model.
    This is based on the standard transformer architecture used in most models.
    """
    # Embedding parameters
    embedding_params = vocab_size * hidden_size + max_position_embeddings * hidden_size
    
    # Per layer parameters
    # Self-attention: Q, K, V projections + output projection
    attention_params = 4 * hidden_size * hidden_size
    
    # Feed-forward network (typically 4x hidden_size)
    ffn_params = 2 * hidden_size * (4 * hidden_size)
    
    # Layer normalization parameters (2 per layer for attention and ffn)
    ln_params = 4 * hidden_size
    
    # Total per layer
    layer_params = attention_params + ffn_params + ln_params
    
    # Total model parameters
    total_params = embedding_params + (num_layers * layer_params)
    
    return total_params

def estimate_layers_for_target_params(hidden_size: int, target_params: int, num_heads: int, vocab_size: int = 32000, max_position_embeddings: int = 2048) -> int:
    """
    Estimate the number of layers needed to achieve a target parameter count.
    """
    # Calculate embedding parameters
    embedding_params = vocab_size * hidden_size + max_position_embeddings * hidden_size
    
    # Calculate parameters per layer
    attention_params = 4 * hidden_size * hidden_size
    ffn_params = 2 * hidden_size * (4 * hidden_size)
    ln_params = 4 * hidden_size
    layer_params = attention_params + ffn_params + ln_params
    
    # Calculate layers needed
    remaining_params = target_params - embedding_params
    if remaining_params <= 0:
        return 1
    
    estimated_layers = max(1, int(remaining_params / layer_params))
    return estimated_layers

def run_model_config(args, num_layers: int, hidden_size: int) -> int:
    """Run a model configuration and return the number of parameters."""
    print(f"Testing configuration: {num_layers} layers, {hidden_size} hidden size")
    
    cmd = f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} python -u -m cehrgpt.runners.hf_cehrgpt_pretrain_runner   --model_name_or_path {args.model_name_or_path}_test   --tokenizer_name_or_path {args.tokenizer_name_or_path}   --output_dir {args.output_dir}   --data_folder {args.data_folder}   --dataset_prepared_path {args.dataset_prepared_path}   --do_train true --seed 42   --dataloader_num_workers 16 --dataloader_prefetch_factor 8   --hidden_size {hidden_size} --num_hidden_layers {num_layers} --max_position_embeddings {args.max_position_embeddings} --evaluation_strategy epoch --save_strategy epoch   --sample_packing --max_tokens_per_batch {args.max_tokens_per_batch}   --warmup_steps 500 --weight_decay 0.01   --num_train_epochs 50 --learning_rate 0.0002   --use_early_stopping --early_stopping_threshold 0.001 --load_best_model_at_end True --n_head {args.num_heads} --memory_fraction {args.memory_fraction}"

    print(cmd)
    monitor_thread, measurements = start_monitoring(gpu_id=int(args.cuda_visible_devices), interval=2.0)
    return_code, result_text = run_and_capture(cmd, timeout_seconds=args.timeout_seconds)
    print(f"Return code: {return_code}")
    
    if return_code == 0:
        n_parameters = get_n_parameters(result_text)
        return n_parameters
    else:
        print(f"Model run failed with return code {return_code}")
        return -1

def binary_search_depth(args, hidden_size: int, target_params: int, min_layers: int = 1, max_layers: int = 100) -> Tuple[Optional[int], Optional[int]]:
    """
    Perform binary search to find the optimal number of layers for a given target parameter count.
    
    Args:
        args: Command line arguments
        hidden_size: Hidden size of the model
        target_params: Target number of parameters
        min_layers: Minimum number of layers to try
        max_layers: Maximum number of layers to try
    
    Returns:
        Tuple of (optimal_layers, actual_params)
    """
    print(f"\n=== Binary search for hidden_size={hidden_size}, target_params={target_params:,} ===")
    
    best_layers = None
    best_params = None
    min_diff = float('inf')
    
    # Cache for already tested configurations
    tested_configs = {}
    
    # Start with estimated layers
    estimated_layers = estimate_layers_for_target_params(hidden_size, target_params, args.num_heads, 
                                                        vocab_size=32000, max_position_embeddings=args.max_position_embeddings)
    estimated_layers = max(min_layers, min(max_layers, estimated_layers))
    
    print(f"Starting with estimated layers: {estimated_layers}")
    
    # Test the estimated configuration first
    actual_params = run_model_config(args, estimated_layers, hidden_size)
    print(f"Estimated layers: {estimated_layers}, Actual params: {actual_params}")
    input()
    tested_configs[estimated_layers] = actual_params
    
    if actual_params != -1:
        diff = abs(actual_params - target_params)
        min_diff = diff
        best_layers = estimated_layers
        best_params = actual_params
        print(f"Initial test: {estimated_layers} layers, {actual_params:,} parameters (diff: {diff:,})")
    
    # Binary search around the estimated value
    left = min_layers
    right = max_layers
    
    # Adjust search bounds based on initial result
    if actual_params != -1:
        if actual_params < target_params:
            left = estimated_layers + 1
        else:
            right = estimated_layers - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        # Skip if already tested
        if mid in tested_configs:
            actual_params = tested_configs[mid]
        else:
            actual_params = run_model_config(args, mid, hidden_size)
            tested_configs[mid] = actual_params
        
        if actual_params == -1:
            # Model run failed, try fewer layers
            right = mid - 1
            continue
        
        print(f"Layers: {mid}, Parameters: {actual_params:,}, Target: {target_params:,}")
        
        # Check if this is the best match so far
        diff = abs(actual_params - target_params)
        if diff < min_diff:
            min_diff = diff
            best_layers = mid
            best_params = actual_params
        
        if actual_params < target_params:
            # Need more layers
            left = mid + 1
        elif actual_params > target_params:
            # Need fewer layers
            right = mid - 1
        else:
            # Exact match found
            best_layers = mid
            best_params = actual_params
            break
    
    # Try a few more configurations around the best found
    if best_layers is not None:
        for offset in [-2, -1, 1, 2]:
            test_layers = best_layers + offset
            if test_layers >= min_layers and test_layers <= max_layers and test_layers not in tested_configs:
                actual_params = run_model_config(args, test_layers, hidden_size)
                tested_configs[test_layers] = actual_params
                
                if actual_params != -1:
                    diff = abs(actual_params - target_params)
                    if diff < min_diff:
                        min_diff = diff
                        best_layers = test_layers
                        best_params = actual_params
                        print(f"Better match found: {test_layers} layers, {actual_params:,} parameters")
    
    if best_layers is not None:
        print(f"Best configuration: {best_layers} layers, {best_params:,} parameters (diff: {min_diff:,})")
    else:
        print("No valid configuration found")
    
    return best_layers, best_params

# Configuration
CEHR_GPT_MODEL_DIR = "/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/cehrgpt_mia/model_dir_scaling_test"
CEHR_GPT_DATA_DIR = "/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/cehrgpt_flo/"

parser = argparse.ArgumentParser(description="Binary search for optimal depth to match target parameter count")
parser.add_argument("--width", type=int, required=True, help="Hidden size (width) of the model")
parser.add_argument("--target_params", type=int, required=True, help="Target number of parameters")
parser.add_argument("--model_name_or_path", type=str, default=CEHR_GPT_MODEL_DIR)
parser.add_argument("--tokenizer_name_or_path", type=str, default=CEHR_GPT_MODEL_DIR)
parser.add_argument("--output_dir", type=str, default=CEHR_GPT_MODEL_DIR)
parser.add_argument("--data_folder", type=str, default=CEHR_GPT_DATA_DIR + "/patient_sequence/train")
parser.add_argument("--dataset_prepared_path", type=str, default=CEHR_GPT_DATA_DIR + "/dataset_prepared")
parser.add_argument("--cuda_visible_devices", "-c", type=str, default="1")
parser.add_argument("--num_heads", "-nh", type=int, default=8)
parser.add_argument("--max_position_embeddings", "-mpe", type=int, default=2048)
parser.add_argument("--max_tokens_per_batch", "-mtb", type=int, default=2048)
parser.add_argument("--timeout_seconds", "-t", type=int, default=120, help="Timeout in seconds for each training run")
parser.add_argument("--min_layers", type=int, default=1, help="Minimum number of layers to try")
parser.add_argument("--max_layers", type=int, default=100, help="Maximum number of layers to try")

args = parser.parse_args()

# Set default values for args
args.memory_fraction = 1.0
args.max_position_embeddings_multiplier = 1
args.max_tokens_per_batch_multiplier = 1
args.max_position_embeddings = args.max_position_embeddings * args.max_position_embeddings_multiplier
args.max_tokens_per_batch = args.max_tokens_per_batch * args.max_tokens_per_batch_multiplier

# Get GPU ID from cuda_visible_devices
gpu_id = int(args.cuda_visible_devices) if args.cuda_visible_devices.isdigit() else 0

print(f"Starting binary search for width={args.width}, target_params={args.target_params:,}")

# Run binary search
optimal_layers, actual_params = binary_search_depth(
    args, 
    args.width, 
    args.target_params, 
    args.min_layers, 
    args.max_layers
)

# Create result
if optimal_layers is not None:
    result = {
        "width": args.width,
        "optimal_layers": optimal_layers,
        "actual_params": actual_params,
        "target_params": args.target_params,
        "difference": abs(actual_params - args.target_params),
        "percentage_diff": abs(actual_params - args.target_params) / args.target_params * 100
    }
    print(f"\nSUCCESS: Width {args.width} -> {optimal_layers} layers, {actual_params:,} parameters")
else:
    result = {
        "width": args.width,
        "optimal_layers": None,
        "actual_params": None,
        "target_params": args.target_params,
        "error": "Could not find valid configuration"
    }
    print(f"\nFAILED: Could not find valid configuration for width {args.width}")

# Save result to JSON file
output_filename = f"binary_search_width_{args.width}_target_{args.target_params:,}.json"
with open(output_filename, "w") as f:
    json.dump(result, f, indent=2)

print(f"Result saved to {output_filename}")
