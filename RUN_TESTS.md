# LoRA Optimization Testing Guide

This guide provides step-by-step instructions for testing the LoRA request-level pipelining optimization in the Docker container.

## 🐳 Container Setup

### 1. Launch Container
```bash
docker run -it --gpus all --rm --shm-size=64g \
  -v /fsx/srivrohi/aws/multi-lora/vllm_src:/workspace/vllm_src \
  -v /fsx/srivrohi/aws/vllm_profiler:/workspace/vllm_profiler \
  -v /fsx/srivrohi/aws/multi-lora/profiling_code:/workspace/profiling_code \
  -v /opt/dlami/nvme/hf_cache:/fsx/srivrohi/hf_cache \
  -v /fsx/srivrohi/aws/profiles:/workspace/profiles \
  -p 8000:8000 -p 6006:6006 \
  vllm-build:cu128 bash
```

### 2. Validate Container Environment
```bash
# Check GPU availability
python3 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# Expected output:
# 2.8.0+cu128 (or similar)
# 12.8 (or similar)
# True
# 1 (or more)
```

### 3. Setup Conda Environment
```bash
conda init
source ~/.bashrc
conda deactivate && conda activate vllm
```

### 4. Navigate to Source Directory
```bash
cd /workspace/vllm_src
# or
cd /fsx/srivrohi/aws/multi-lora/vllm_src
```

### 5. Switch to Pipelining Branch
```bash
git checkout feature/request-level-pipelining
```

### 6. Rebuild vLLM with Optimizations
```bash
pip install -e .
```

## 🧪 Testing Sequence

### Step 1: Container Validation
```bash
python3 test_container_setup.py
```
**Expected**: All validation checks should pass ✅

### Step 2: Quick Functionality Test
```bash
python3 quick_test_pipelining.py
```
**Expected**: 
- Basic LoRA functionality works
- Pipelining components are available
- Multi-LoRA switching shows potential improvements

### Step 3: Comprehensive Benchmark
```bash
python3 benchmark_lora_pipelining.py
```
**Expected Results**:
- Single request: Similar performance (±5%)
- Multi-request (5): 30-60% improvement
- Stress test (10): 50-80% improvement

### Step 4: Detailed Profiling (Optional)
```bash
python3 profile_lora_optimization.py
```
**Expected**: Detailed NVTX markers for Nsight Systems analysis

## 📊 Expected Performance Improvements

| Scenario | Expected Improvement | Metric |
|----------|---------------------|---------|
| Single Request | 0-5% | Latency |
| 5 Multi-Requests | 30-60% | Total time |
| 10 Multi-Requests | 50-80% | Total time |
| Throughput | 3-7x | Requests/second |

## 🔍 Troubleshooting

### Issue: GPU Not Available
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### Issue: vLLM Import Errors
```bash
# Reinstall vLLM
pip uninstall vllm -y
pip install -e .

# Check installation
python3 -c "import vllm; print(vllm.__version__)"
```

### Issue: LoRA Adapter Not Found
```bash
# Check LoRA path
ls -la /fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter/

# Expected files:
# adapter_config.json
# adapter_model.safetensors
```

### Issue: Out of Memory
```bash
# Reduce GPU memory utilization in test scripts
# Edit benchmark files and change gpu_memory_utilization from 0.8 to 0.6
```

### Issue: Pipelining Not Working
```bash
# Verify branch
git branch
# Should show: * feature/request-level-pipelining

# Check implementation
python3 -c "from vllm.lora.worker_manager import RequestPipelineManager; print('Pipelining available')"
```

## 📈 Analyzing Results

### Benchmark Results
- Look for `lora_pipelining_benchmark_results.json`
- Key metrics: `improvement_pct`, `throughput_improvement`
- Subsequent request times should be much faster than first request

### Profiling Results
- Look for `lora_optimization_profile.json`
- Memory usage should be reasonable
- NVTX markers show pipelining patterns

### Success Criteria
✅ **Minimum Success**: 20% improvement in multi-request scenarios
🎯 **Target Success**: 50%+ improvement in multi-request scenarios
🚀 **Excellent Success**: 3x+ throughput improvement

## 🎯 Next Steps After Testing

1. **If tests pass**: Ready for production evaluation
2. **If improvements are modest**: Consider combining with inter-layer parallelization
3. **If tests fail**: Check implementation and container setup

## 📝 Reporting Results

Please share:
1. Output from `test_container_setup.py`
2. Results from `benchmark_lora_pipelining.py`
3. Any error messages or unexpected behavior
4. GPU model and memory information

## 🔧 Advanced Testing

### Custom Benchmark Parameters
```bash
# Edit benchmark_lora_pipelining.py to adjust:
# - Number of requests
# - Model parameters
# - LoRA configurations
# - Memory utilization
```

### Nsight Systems Profiling
```bash
# Install Nsight Systems in container (if needed)
# Run with profiling:
nsys profile -o lora_optimization python3 profile_lora_optimization.py
```

This testing suite will validate that the request-level pipelining optimization is working correctly and measure the actual performance improvements in your environment.