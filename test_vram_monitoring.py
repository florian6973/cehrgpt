#!/usr/bin/env python3
"""
Test script for VRAM monitoring functionality.
"""

import time
import subprocess
import threading
from typing import Dict, List

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    print("✓ pynvml available")
except ImportError:
    print("✗ pynvml not available")
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

def test_vram_monitoring():
    """Test VRAM monitoring functionality."""
    print("Testing VRAM monitoring...")
    
    # Test current VRAM usage
    print("\n1. Current VRAM usage:")
    if NVML_AVAILABLE:
        memory = get_gpu_memory_usage(0)
        print(f"   pynvml: {memory['used']}MB used, {memory['total']}MB total")
    
    memory_smi = get_gpu_memory_usage_nvidia_smi(0)
    print(f"   nvidia-smi: {memory_smi['used']}MB used, {memory_smi['total']}MB total")
    
    # Test monitoring for 5 seconds
    print("\n2. Monitoring VRAM for 5 seconds...")
    
    if NVML_AVAILABLE:
        print("   Using pynvml...")
        monitor_thread, measurements = monitor_vram_async(gpu_id=0, interval=1.0)
    else:
        print("   Using nvidia-smi...")
        monitor_thread, measurements = monitor_vram_nvidia_smi_async(gpu_id=0, interval=1.0)
    
    time.sleep(5)
    monitor_thread.join(timeout=1)
    
    if measurements:
        print(f"   Collected {len(measurements)} measurements")
        used_memory = [m["used"] for m in measurements]
        max_used = max(used_memory)
        avg_used = sum(used_memory) / len(used_memory)
        print(f"   Max VRAM: {max_used}MB")
        print(f"   Avg VRAM: {avg_used:.1f}MB")
    else:
        print("   No measurements collected")

if __name__ == "__main__":
    test_vram_monitoring() 