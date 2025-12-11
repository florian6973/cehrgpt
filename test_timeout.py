#!/usr/bin/env python3
"""
Test script for timeout functionality.
"""

import time
import subprocess
import sys

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
                return -1, ''.join(output)
            
            # Try to read with a short timeout to allow checking the main timeout
            try:
                char = process.stdout.read(1)  # Read one character at a time
                if char == '' and process.poll() is not None:
                    break
                if char:
                    sys.stdout.write(char)     # Print to console immediately
                    sys.stdout.flush()
                    output.append(char)
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

def test_timeout():
    """Test timeout functionality."""
    print("Testing timeout functionality...")
    
    # Test 1: Short command that should complete
    print("\n1. Testing short command (should complete):")
    return_code, output = run_and_capture("echo 'Hello World'", timeout_seconds=10)
    print(f"Return code: {return_code}")
    print(f"Output: {output.strip()}")
    
    # Test 2: Long command that should timeout
    print("\n2. Testing long command (should timeout after 3 seconds):")
    return_code, output = run_and_capture("sleep 10", timeout_seconds=3)
    print(f"Return code: {return_code}")
    print(f"Output: {output.strip()}")
    
    # Test 3: Command that produces output
    print("\n3. Testing command with output (should timeout):")
    return_code, output = run_and_capture("for i in {1..100}; do echo 'Line $i'; sleep 0.1; done", timeout_seconds=3)
    print(f"Return code: {return_code}")
    print(f"Output length: {len(output)} characters")

if __name__ == "__main__":
    test_timeout() 