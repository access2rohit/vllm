#!/usr/bin/env python3
"""
Debug baseline comparison with detailed logging to understand what's happening.
"""

import os
import time
import logging
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Enable debug logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_with_logging(prefetch_enabled: bool, test_name: str):
    """Test with detailed logging."""
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"Prefetch enabled: {prefetch_enabled}")
    print(f"{'='*60}")
    
    # Set environment
    if prefetch_enabled:
        os.environ.pop('DISABLE_LORA_PREFETCH', None)
    else:
        os.environ['DISABLE_LORA_PREFETCH'] = '1'
    
    # Create LLM
    llm = LLM(
        model="openai/gpt-oss-20b",
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_num_seqs=1,
        gpu_memory_utilization=0.3,
        enable_lora=True,
        max_loras=2,
        max_cpu_loras=4,
        max_lora_rank=128,
    )
    
    # Define LoRA requests
    lora_requests = [
        LoRARequest("test_1", 1, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
        LoRARequest("test_2", 2, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
    ]
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
    
    # Warmup
    print("🔥 Warmup...")
    llm.generate(["Warmup 1"], sampling_params, lora_request=lora_requests[0])
    llm.generate(["Warmup 2"], sampling_params, lora_request=lora_requests[1])
    
    # Test
    print("📊 Testing...")
    
    print(f"\n🔄 First request (LoRA {lora_requests[0].lora_int_id})...")
    start_time = time.time()
    outputs1 = llm.generate(["Test question 1"], sampling_params, lora_request=lora_requests[0])
    first_time = time.time() - start_time
    print(f"✅ First request completed: {first_time:.3f}s")
    
    print(f"\n🔄 Second request (LoRA {lora_requests[1].lora_int_id})...")
    start_time = time.time()
    outputs2 = llm.generate(["Test question 2"], sampling_params, lora_request=lora_requests[1])
    second_time = time.time() - start_time
    print(f"✅ Second request completed: {second_time:.3f}s")
    
    improvement = ((first_time - second_time) / first_time) * 100
    print(f"\n📈 Results for {test_name}:")
    print(f"  First request:  {first_time:.3f}s")
    print(f"  Second request: {second_time:.3f}s")
    print(f"  Improvement:    {improvement:+.1f}%")
    
    return {
        "test_name": test_name,
        "first_time": first_time,
        "second_time": second_time,
        "improvement": improvement
    }

def main():
    """Run debug tests with logging."""
    print("🔬 Debug Baseline Comparison with Logging")
    print("=" * 60)
    
    # Test baseline
    baseline_result = test_with_logging(False, "BASELINE (No Optimization)")
    
    print("\n" + "="*60)
    print("Waiting 5 seconds between tests...")
    time.sleep(5)
    
    # Test optimized
    optimized_result = test_with_logging(True, "OPTIMIZED (With Prefetching)")
    
    # Compare
    print(f"\n{'='*60}")
    print("🎯 COMPARISON")
    print(f"{'='*60}")
    
    baseline_second = baseline_result["second_time"]
    optimized_second = optimized_result["second_time"]
    true_improvement = ((baseline_second - optimized_second) / baseline_second) * 100
    
    print(f"Baseline second request:  {baseline_second:.3f}s")
    print(f"Optimized second request: {optimized_second:.3f}s")
    print(f"TRUE improvement:         {true_improvement:+.1f}%")
    
    if abs(true_improvement) < 1:
        print("❌ No significant difference - optimization not working")
        print("\n🔍 DIAGNOSIS:")
        print("   - Optimization code exists but is never triggered")
        print("   - prefetch_next_adapter method is not called automatically")
        print("   - Need to integrate with vLLM engine request processing")
    elif true_improvement > 5:
        print("✅ Optimization is working!")
    else:
        print("⚠️  Small improvement detected")

if __name__ == "__main__":
    main()