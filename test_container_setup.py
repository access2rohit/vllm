#!/usr/bin/env python3
"""
Container setup validation script for LoRA optimization testing.

This script validates that the container environment is properly configured
for testing the LoRA optimization implementations.
"""

import sys
import os
import subprocess
import torch
from pathlib import Path

def check_gpu_setup():
    """Validate GPU setup and CUDA availability."""
    print("=== GPU Setup Validation ===")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
    
    assert torch.cuda.is_available(), "CUDA must be available for testing"
    assert torch.cuda.device_count() > 0, "At least one GPU required"
    print("✅ GPU setup validated")
    print()

def check_vllm_installation():
    """Check if vLLM is properly installed."""
    print("=== vLLM Installation Check ===")
    
    try:
        import vllm
        print(f"vLLM version: {vllm.__version__}")
        
        # Check if our modifications are present
        from vllm.lora.worker_manager import RequestPipelineManager
        print("✅ RequestPipelineManager found - request pipelining available")
        
        from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
        manager_methods = [method for method in dir(LRUCacheWorkerLoRAManager) 
                          if 'prefetch' in method.lower()]
        if manager_methods:
            print(f"✅ Prefetch methods found: {manager_methods}")
        else:
            print("⚠️  No prefetch methods found - may need to rebuild")
            
    except ImportError as e:
        print(f"❌ vLLM import failed: {e}")
        print("Please install vLLM with: pip install -e .")
        sys.exit(1)
    
    print("✅ vLLM installation validated")
    print()

def check_lora_adapter():
    """Check if the LoRA adapter is available."""
    print("=== LoRA Adapter Check ===")
    
    lora_path = "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"
    
    if not os.path.exists(lora_path):
        print(f"❌ LoRA adapter not found at: {lora_path}")
        print("Please ensure the LoRA adapter is mounted correctly")
        sys.exit(1)
    
    # Check for required files
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    for file in required_files:
        file_path = os.path.join(lora_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024**2  # MB
            print(f"✅ {file}: {size:.1f} MB")
        else:
            print(f"❌ Missing required file: {file}")
            sys.exit(1)
    
    print("✅ LoRA adapter validated")
    print()

def check_workspace_structure():
    """Validate workspace directory structure."""
    print("=== Workspace Structure Check ===")
    
    expected_dirs = [
        "/workspace/vllm_src",
        "/workspace/profiling_code", 
        "/workspace/vllm_profiler",
        "/workspace/profiles"
    ]
    
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"⚠️  {dir_path} not found")
    
    # Check if we're in the right directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    if "/workspace/vllm_src" in current_dir or "/fsx/srivrohi/aws/multi-lora/vllm_src" in current_dir:
        print("✅ Working in correct directory")
    else:
        print("⚠️  Consider changing to /workspace/vllm_src")
    
    print()

def check_conda_environment():
    """Check conda environment setup."""
    print("=== Conda Environment Check ===")
    
    # Check if conda is available
    try:
        result = subprocess.run(["conda", "--version"], capture_output=True, text=True)
        print(f"Conda version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("⚠️  Conda not found - may need to run 'conda init' first")
        return
    
    # Check current environment
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "base")
    print(f"Current conda environment: {conda_env}")
    
    if conda_env == "vllm":
        print("✅ Using vllm conda environment")
    else:
        print("⚠️  Not using vllm environment - run 'conda activate vllm'")
    
    print()

def main():
    """Run all validation checks."""
    print("🚀 Container Setup Validation for LoRA Optimization Testing")
    print("=" * 60)
    
    try:
        check_gpu_setup()
        check_conda_environment()
        check_workspace_structure()
        check_lora_adapter()
        check_vllm_installation()
        
        print("🎉 All validation checks passed!")
        print("Ready to run LoRA optimization benchmarks.")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()