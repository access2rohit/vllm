#!/usr/bin/env python3
"""
Test script that actually measures LoRA loading performance.
This test uses MORE than max_loras to force eviction and reloading.
"""

import os
import time
import logging
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Enable detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_with_eviction(prefetch_enabled: bool, test_name: str):
    """Test with cache eviction to force actual LoRA loading."""
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"Prefetch enabled: {prefetch_enabled}")
    print(f"{'='*60}")
    
    # Set environment
    if prefetch_enabled:
        os.environ.pop('DISABLE_LORA_PREFETCH', None)
    else:
        os.environ['DISABLE_LORA_PREFETCH'] = '1'
    
    # Create LLM with max_loras=2 (so we can force eviction with 3 LoRAs)
    llm = LLM(
        model="openai/gpt-oss-20b",
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_num_seqs=1,
        gpu_memory_utilization=0.3,
        enable_lora=True,
        max_loras=2,  # Only 2 LoRAs can be in GPU cache
        max_cpu_loras=4,
        max_lora_rank=128,
    )
    
    # Define 3 LoRA requests (more than max_loras to force eviction)
    lora_requests = [
        LoRARequest("lora_1", 1, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
        LoRARequest("lora_2", 2, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
        LoRARequest("lora_3", 3, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
    ]
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
    
    print("📊 Testing with cache eviction...")
    print("Using 3 LoRAs with max_loras=2 to force reloading\n")
    
    times = []
    
    # Cycle through LoRAs: 1 -> 2 -> 3 -> 1 -> 2 -> 3
    # This forces eviction and reloading
    for i, lora_idx in enumerate([0, 1, 2, 0, 1, 2]):
        lora_req = lora_requests[lora_idx]
        print(f"Request {i+1}: Using LoRA {lora_req.lora_int_id}")
        
        start_time = time.time()
        outputs = llm.generate([f"Question {i+1}?"], sampling_params, lora_request=lora_req)
        request_time = time.time() - start_time
        times.append(request_time)
        
        print(f"  Completed in {request_time:.3f}s")
    
    # Calculate average time for requests 4-6 (after warmup)
    avg_time = sum(times[3:]) / 3
    
    print(f"\n📈 Results for {test_name}:")
    print(f"  Request times: {[f'{t:.3f}s' for t in times]}")
    print(f"  Average (requests 4-6): {avg_time:.3f}s")
    
    return {
        "test_name": test_name,
        "times": times,
        "avg_time": avg_time
    }

def main():
    """Run test with cache eviction."""
    print("🔬 LoRA Loading Performance Test (With Cache Eviction)")
    print("=" * 60)
    print("This test uses 3 LoRAs with max_loras=2")
    print("This forces eviction and reloading to measure true performance")
    print("=" * 60)
    
    # Test baseline
    baseline_result = test_with_eviction(False, "BASELINE (No Optimization)")
    
    print("\n" + "="*60)
    print("Waiting 5 seconds between tests...")
    time.sleep(5)
    
    # Test optimized
    optimized_result = test_with_eviction(True, "OPTIMIZED (With Prefetching)")
    
    # Compare
    print(f"\n{'='*60}")
    print("🎯 COMPARISON")
    print(f"{'='*60}")
    
    baseline_avg = baseline_result["avg_time"]
    optimized_avg = optimized_result["avg_time"]
    improvement = ((baseline_avg - optimized_avg) / baseline_avg) * 100
    
    print(f"Baseline average:  {baseline_avg:.3f}s")
    print(f"Optimized average: {optimized_avg:.3f}s")
    print(f"TRUE improvement:  {improvement:+.1f}%")
    print(f"Time saved:        {baseline_avg - optimized_avg:.3f}s per request")
    
    if improvement > 10:
        print("\n✅ Optimization is working effectively!")
    elif improvement > 5:
        print("\n✅ Moderate improvement detected")
    elif improvement > 0:
        print("\n⚠️  Small improvement detected")
    else:
        print("\n❌ No improvement or regression")
        print("\n🔍 Possible reasons:")
        print("   - Prefetch not completing before LoRA is needed")
        print("   - LoRA loading time is too short relative to inference")
        print("   - Cache behavior different than expected")

if __name__ == "__main__":
    main()
