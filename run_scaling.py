import os
import argparse
import subprocess
import pprint
import time
import threading
import sys
from typing import Dict, List
import numpy as np
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

def monitor_vram(gpu_id: int = 0, interval: float = 1.0) -> List[Dict]:
    """Monitor VRAM usage and return list of measurements."""
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
    
    return measurements

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

def get_gpu_memory_usage_nvidia_smi(gpu_id: int = 0) -> Dict[str, int]:
    """Get GPU memory usage using nvidia-smi command."""
    try:
        cmd = f"nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i {gpu_id}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout.strip()
            used, total = map(int, output.split(', '))
            return {
                "used": used,
                "total": total,
                "free": total - used
            }
    except Exception as e:
        print(f"Error getting GPU memory info via nvidia-smi: {e}")
    
    return {"used": 0, "total": 0, "free": 0}

def monitor_vram_nvidia_smi_async(gpu_id: int = 0, interval: float = 1.0):
    """Start VRAM monitoring using nvidia-smi in a separate thread."""
    measurements = []
    
    def monitor():
        while True:
            try:
                memory_info = get_gpu_memory_usage_nvidia_smi(gpu_id)
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
        # env={"PATH": os.environ["PATH"]}
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
        monitor_thread, measurements = monitor_vram_nvidia_smi_async(gpu_id=gpu_id, interval=interval)
    time.sleep(1)  # Give monitoring a moment to start
    return monitor_thread, measurements

def get_n_parameters(result_text: str):
    txt = "Number of parameters:"
    n_parameters = result_text.strip().split(txt)[1].split(',')[0]
    print(f"Number of parameters: {n_parameters}")
    return n_parameters


# CEHR_GPT_MODEL_DIR = "/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/cehrgpt_mia/model_dir_mia_20"
CEHR_GPT_MODEL_DIR = "/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/cehrgpt_mia/model_dir_scaling_test"

# CEHR_GPT_MODEL_DIR = "/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/cehrgpt_mia/model_dir_test"

CEHR_GPT_DATA_DIR = "/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/cehrgpt_flo/"


parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default=CEHR_GPT_MODEL_DIR)
parser.add_argument("--tokenizer_name_or_path", type=str, default=CEHR_GPT_MODEL_DIR)
parser.add_argument("--output_dir", type=str, default=CEHR_GPT_MODEL_DIR)
parser.add_argument("--data_folder", type=str, default=CEHR_GPT_DATA_DIR + "/patient_sequence/train")
parser.add_argument("--dataset_prepared_path", type=str, default=CEHR_GPT_DATA_DIR + "/dataset_prepared")
parser.add_argument("--cuda_visible_devices", "-c", type=str, default="1")
parser.add_argument("--hidden_size", "-hs", type=int, default=768)
parser.add_argument("--hidden_size_multiplier", "-hsm", type=int, default=1)
parser.add_argument("--num_hidden_layers", "-nl", type=int, default=14)
parser.add_argument("--num_heads", "-nh", type=int, default=16)#12)
parser.add_argument("--max_position_embeddings", "-mpe", type=int, default=2048)
parser.add_argument("--max_position_embeddings_multiplier", "-mpem", type=int, default=1)
# parser.add_argument("--max_position_embeddings_multiplier", "-mpem", type=int, default=8)
parser.add_argument("--max_tokens_per_batch", "-mtb", type=int, default=2048)
# parser.add_argument("--max_tokens_per_batch_multiplier", "-mtbm", type=int, default=8)
parser.add_argument("--max_tokens_per_batch_multiplier", "-mtbm", type=int, default=1)

parser.add_argument("--timeout_seconds", "-t", type=int, default=120, help="Timeout in seconds for each training run")

# parser.add_argument("--timeout_seconds", "-t", type=int, default=120, help="Timeout in seconds for each training run")
# parser.add_argument("--memory_fraction", "-mf", type=float, default=1.0)

args = parser.parse_args()

# sizes = {}
# vram_stats = {}

data_stats = {}

hs = args.hidden_size
mpe = args.max_position_embeddings
mtpb = args.max_tokens_per_batch

# Get GPU ID from cuda_visible_devices
gpu_id = int(args.cuda_visible_devices) if args.cuda_visible_devices.isdigit() else 0

# for (nh, hsm) in [(10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6)]:
# for (nh, hsm) in [(4, 2), (8, 2), (12, 2), (16, 2), (20, 2), (24, 2), (28, 2), (32, 2), (36, 2)]:
# for (nh, hsm) in [(4, 2), (8, 2), (12, 2), (16, 2), (20, 2), (24, 2), (28, 2)]:
# for (nh, hsm) in [(28, 2), (36, 2), (44, 2), (52, 2), (60, 2), (68, 2), (76, 2), (84, 2), (92, 2), (100, 2)]:

# for (nh, hsm) in [(8, 2)]:
for (nh, hsm) in [(14, 1), (14, 2), (14, 4), (20, 1), (20, 2), (20, 4), (28, 1), (28, 2), (28, 4), (32, 1), (32, 2), (32, 4), (28, 5), (44, 4)]:

# for (nh, hsm) in [(4, 2)]:
    print(f"Running for {nh} layers and {hsm} hidden size multiplier")
    args.num_hidden_layers = nh
    args.hidden_size_multiplier = hsm

    args.hidden_size = hs * args.hidden_size_multiplier
    args.max_position_embeddings = mpe * args.max_position_embeddings_multiplier
    args.max_tokens_per_batch = mtpb * args.max_tokens_per_batch_multiplier

    high = 1.0
    low = 0.1
    last_vram_stats = None
    while high - low > 0.1:
        mid = (high + low) / 2
        args.memory_fraction = mid
        print(f"Memory fraction: {args.memory_fraction}")
        # run_and_capture(cmd, timeout_seconds=args.timeout_seconds)
        cmd = f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} python -u -m cehrgpt.runners.hf_cehrgpt_pretrain_runner   --model_name_or_path {args.model_name_or_path}_test   --tokenizer_name_or_path {args.tokenizer_name_or_path}   --output_dir {args.output_dir}   --data_folder {args.data_folder}   --dataset_prepared_path {args.dataset_prepared_path}   --do_train true --seed 42   --dataloader_num_workers 16 --dataloader_prefetch_factor 8   --hidden_size {args.hidden_size} --num_hidden_layers {args.num_hidden_layers} --max_position_embeddings {args.max_position_embeddings} --evaluation_strategy epoch --save_strategy epoch   --sample_packing --max_tokens_per_batch {args.max_tokens_per_batch}   --warmup_steps 500 --weight_decay 0.01   --num_train_epochs 50 --learning_rate 0.0002   --use_early_stopping --early_stopping_threshold 0.001 --load_best_model_at_end True --n_head {args.num_heads} --memory_fraction {args.memory_fraction}"

        print(cmd)
        monitor_thread, measurements = start_monitoring(gpu_id=gpu_id, interval=2.0)
        return_code, result_text = run_and_capture(cmd, timeout_seconds=args.timeout_seconds)
        print(f"Return code: {return_code}")
        n_parameters = get_n_parameters(result_text)

        if return_code == 1:
            low = mid
            if last_vram_stats is None:
                last_vram_stats = {
                    "max_used_mb": np.nan,
                    "avg_used_mb": np.nan,
                    "total_mb": np.nan,
                    "measurements": [],
                    "n_parameters": n_parameters,
                    'memory_fraction_low': low,
                    'memory_fraction_high': high,
                }
            else:
                last_vram_stats['memory_fraction_low'] = low
                last_vram_stats['memory_fraction_high'] = high
        else:
            high = mid
            monitor_thread.join(timeout=1)
        
            if measurements:
                # Calculate VRAM statistics
                used_memory = [m["used"] for m in measurements]
                max_used = max(used_memory) if used_memory else 0
                avg_used = sum(used_memory) / len(used_memory) if used_memory else 0
                
                last_vram_stats = {
                    "max_used_mb": max_used,
                    "avg_used_mb": avg_used,
                    "total_mb": measurements[0]["total"] if measurements else 0,
                    "measurements": measurements,
                    "n_parameters": n_parameters,
                    'memory_fraction_low': low,
                    'memory_fraction_high': high,
                }
                
                print(f"VRAM Usage - Max: {max_used}MB, Avg: {avg_used:.1f}MB")

        # save results when works,

    data_stats[(nh, hsm)] = last_vram_stats

    # no early stopping
    # cmd = f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} python -u -m cehrgpt.runners.hf_cehrgpt_pretrain_runner   --model_name_or_path {args.model_name_or_path}   --tokenizer_name_or_path {args.tokenizer_name_or_path}   --output_dir {args.output_dir}   --data_folder {args.data_folder}   --dataset_prepared_path {args.dataset_prepared_path}   --do_train true --seed 42   --dataloader_num_workers 16 --dataloader_prefetch_factor 8   --hidden_size {args.hidden_size} --num_hidden_layers {args.num_hidden_layers} --max_position_embeddings {args.max_position_embeddings} --evaluation_strategy epoch --save_strategy epoch   --sample_packing --max_tokens_per_batch {args.max_tokens_per_batch}   --warmup_steps 500 --weight_decay 0.01   --num_train_epochs 20 --learning_rate 0.0002 --load_best_model_at_end True --n_head {args.num_heads} --use_early_stopping False"

    # cmd = f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} python -u -m cehrgpt.runners.hf_cehrgpt_pretrain_runner   --model_name_or_path {args.model_name_or_path}_test   --tokenizer_name_or_path {args.tokenizer_name_or_path}   --output_dir {args.output_dir}   --data_folder {args.data_folder}   --dataset_prepared_path {args.dataset_prepared_path}   --do_train true --seed 42   --dataloader_num_workers 16 --dataloader_prefetch_factor 8   --hidden_size {args.hidden_size} --num_hidden_layers {args.num_hidden_layers} --max_position_embeddings {args.max_position_embeddings} --evaluation_strategy epoch --save_strategy epoch   --sample_packing --max_tokens_per_batch {args.max_tokens_per_batch}   --warmup_steps 500 --weight_decay 0.01   --num_train_epochs 50 --learning_rate 0.0002   --use_early_stopping --early_stopping_threshold 0.001 --load_best_model_at_end True --n_head {args.num_heads} --model_size_only True"

    # CUDA_VISIBLE_DEVICES=1 python -u -m cehrgpt.runners.hf_cehrgpt_pretrain_runner   --model_name_or_path $CEHR_GPT_MODEL_DIR   --tokenizer_name_or_path $CEHR_GPT_MODEL_DIR   --output_dir $CEHR_GPT_MODEL_DIR   --data_folder "$CEHR_GPT_DATA_DIR/patient_sequence/train"   --dataset_prepared_path "$CEHR_GPT_DATA_DIR/dataset_prepared"   --do_train true --seed 42   --dataloader_num_workers 16 --dataloader_prefetch_factor 8   --hidden_size 768 --num_hidden_layers 14 --max_position_embeddings 4096^C --evaluation_strategy epoch --save_strategy epoch   --sample_packing --max_tokens_per_batch 16384   --warmup_steps 500 --weight_decay 0.01   --num_train_epochs 50 --learning_rate 0.0002   --use_early_stopping --early_stopping_threshold 0.001 --load_bes
    # t_model_at_end True

    # args.memory_fraction = high # high always works
    # run_and_capture(cmd, timeout_seconds=args.timeout_seconds)
    # cmd = f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} python -u -m cehrgpt.runners.hf_cehrgpt_pretrain_runner   --model_name_or_path {args.model_name_or_path}_test   --tokenizer_name_or_path {args.tokenizer_name_or_path}   --output_dir {args.output_dir}   --data_folder {args.data_folder}   --dataset_prepared_path {args.dataset_prepared_path}   --do_train true --seed 42   --dataloader_num_workers 16 --dataloader_prefetch_factor 8   --hidden_size {args.hidden_size} --num_hidden_layers {args.num_hidden_layers} --max_position_embeddings {args.max_position_embeddings} --evaluation_strategy epoch --save_strategy epoch   --sample_packing --max_tokens_per_batch {args.max_tokens_per_batch}   --warmup_steps 500 --weight_decay 0.01   --num_train_epochs 50 --learning_rate 0.0002   --use_early_stopping --early_stopping_threshold 0.001 --load_best_model_at_end True --n_head {args.num_heads} --memory_fraction {args.memory_fraction}"


print("\n" + "="*50)


if data_stats:
    print("\n" + "="*50)
    print("VRAM USAGE RESULTS:")
    print("="*50)
    for (nh, hsm), stats in data_stats.items():
        print(f"Layers: {nh}, Hidden Size Multiplier: {hsm}, model size: {stats['n_parameters']}")
        print(f"  Max VRAM Used: {stats['max_used_mb']}MB")
        print(f"  Avg VRAM Used: {stats['avg_used_mb']:.1f}MB")
        print(f"  Total VRAM: {stats['total_mb']}MB")
        print(f"  Utilization: {stats['max_used_mb']/stats['total_mb']*100:.1f}%" if stats['total_mb'] > 0 else "  Utilization: N/A")
        print()

# add next steps script to run cehrgpt on the rest
import pandas as pd

# Convert sizes dict to proper DataFrame format
# sizes_df = pd.DataFrame(list(sizes.items()), columns=['config', 'parameters'])
# sizes_df.to_csv("sizes.csv", index=False)

# Convert vram_stats dict to proper DataFrame format
vram_data = []
for (nh, hsm), stats in data_stats.items():
    vram_data.append({
        'layers': nh,
        'hidden_size_multiplier': hsm,
        'max_used_mb': stats['max_used_mb'],
        'avg_used_mb': stats['avg_used_mb'],
        'total_mb': stats['total_mb'],
        'n_parameters': stats['n_parameters'],
        'memory_fraction_low': stats['memory_fraction_low'],
        'memory_fraction_high': stats['memory_fraction_high'],
    })
vram_df = pd.DataFrame(vram_data)
vram_df.to_csv("data_size_stats.csv", index=False)