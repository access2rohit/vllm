#!/usr/bin/env python3
"""
Test TTFT (Time To First Token) improvement in worst-case scenario.
Forces adapter eviction on every request by setting max_loras=1.
This simulates a production scenario with many adapters but limited GPU memory.
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

def test_ttft_worst_case(prefetch_enabled: bool, test_name: str, num_requests: int = 10):
    """Test TTFT with forced adapter eviction (max_loras=1, max_cpu_loras=1)."""
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"Prefetch enabled: {prefetch_enabled}")
    print(f"Configuration: max_loras=1, max_cpu_loras=10")
    print(f"Optimizing: CPU→GPU transfer (not disk loading)")
    print(f"{'='*70}")
    
    # Set environment
    if prefetch_enabled:
        os.environ.pop('DISABLE_LORA_PREFETCH', None)
    else:
        os.environ['DISABLE_LORA_PREFETCH'] = '1'
    
    # Create LLM with realistic production config:
    # - max_loras=1: Only 1 LoRA in GPU (worst case for GPU memory)
    # - max_cpu_loras=10: All LoRAs in CPU (CPU memory is abundant)
    # This tests CPU→GPU transfer optimization
    llm = LLM(
        model="openai/gpt-oss-20b",
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_num_seqs=1,  # Process one request at a time
        gpu_memory_utilization=0.3,
        enable_lora=True,
        max_loras=1,  # Only 1 LoRA in GPU cache - FORCES GPU EVICTION
        max_cpu_loras=10,  # All LoRAs in CPU cache - NO DISK RELOAD
        max_lora_rank=128,
    )
    
    # Define multiple LoRAs (more than cache can hold)
    lora_requests = [
        LoRARequest("lora_1", 1, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
        LoRARequest("lora_2", 2, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
        LoRARequest("lora_3", 3, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
        LoRARequest("lora_4", 4, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
    ]
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
    
    print(f"📊 Testing {num_requests} requests with {len(lora_requests)} different LoRAs")
    print(f"   Each request forces GPU eviction (max_loras=1)")
    print(f"   All LoRAs cached in CPU (max_cpu_loras=10)")
    print(f"   Measuring CPU→GPU transfer time impact on TTFT")
    print()
    
    # NO WARMUP - Start cold to measure real loading time
    
    request_times = []
    
    # Prepare all requests upfront
    all_prompts = []
    all_lora_reqs = []
    for i in range(num_requests):
        lora_idx = i % len(lora_requests)
        all_prompts.append(f"Question {i+1}: What is the capital of France?")
        all_lora_reqs.append(lora_requests[lora_idx])
    
    print("Submitting all requests at once (batched mode)")
    print("This allows the optimizer to see upcoming LoRA requests")
    print()
    
    # Submit ALL requests at once - this is the key!
    # The engine can now see all upcoming LoRA requests and prefetch accordingly
    start_time = time.time()
    outputs = llm.generate(
        all_prompts,
        sampling_params,
        lora_request=all_lora_reqs
    )
    total_time = time.time() - start_time
    
    # Calculate per-request average
    avg_time = total_time / num_requests
    
    print(f"All {num_requests} requests completed in {total_time:.2f}s")
    print(f"Average time per request: {avg_time:.3f}s")
    
    print(f"\n📈 Results for {test_name}:")
    print(f"  Total time:    {total_time:.2f}s")
    print(f"  Avg per request: {avg_time:.3f}s")
    print(f"  Throughput:    {num_requests/total_time:.2f} req/s")
    
    return {
        "test_name": test_name,
        "total_time": total_time,
        "avg_time": avg_time,
        "num_requests": num_requests
    }

def main():
    """Run TTFT worst-case test."""
    print("🔬 TTFT Worst-Case Scenario Test")
    print("=" * 70)
    print("Configuration:")
    print("  - max_loras=1 (only 1 LoRA in GPU)")
    print("  - max_cpu_loras=10 (all LoRAs in CPU)")
    print("  - No warmup (cold start)")
    print("  - Cycling through 4 different LoRAs")
    print("  - Measures CPU→GPU transfer impact on TTFT")
    print()
    print("This simulates:")
    print("  - Production with 1000s of adapters")
    print("  - GPU can only hold 1 adapter at a time")
    print("  - CPU has all adapters cached (abundant memory)")
    print("  - Every request requires CPU→GPU transfer")
    print("  - Optimization: Prefetch next adapter to GPU while processing current")
    print("=" * 70)
    
    num_requests = 10
    
    # Test baseline
    print("\n🚫 Testing BASELINE (No Optimization)")
    baseline_result = test_ttft_worst_case(False, "BASELINE", num_requests)
    
    print("\n" + "="*70)
    print("Waiting 5 seconds between tests...")
    time.sleep(5)
    
    # Test optimized
    print("\n🚀 Testing OPTIMIZED (With Prefetching)")
    optimized_result = test_ttft_worst_case(True, "OPTIMIZED", num_requests)
    
    # Compare
    print(f"\n{'='*70}")
    print("🎯 COMPARISON - TTFT IMPROVEMENT")
    print(f"{'='*70}")
    
    baseline_total = baseline_result["total_time"]
    optimized_total = optimized_result["total_time"]
    total_improvement = ((baseline_total - optimized_total) / baseline_total) * 100
    
    baseline_avg = baseline_result["avg_time"]
    optimized_avg = optimized_result["avg_time"]
    avg_improvement = ((baseline_avg - optimized_avg) / baseline_avg) * 100
    
    print(f"\nTotal Time:")
    print(f"  Baseline:   {baseline_total:.2f}s")
    print(f"  Optimized:  {optimized_total:.2f}s")
    print(f"  Improvement: {total_improvement:+.1f}%")
    print(f"  Time saved:  {baseline_total - optimized_total:.2f}s")
    
    print(f"\nAverage Per Request:")
    print(f"  Baseline:   {baseline_avg:.3f}s")
    print(f"  Optimized:  {optimized_avg:.3f}s")
    print(f"  Improvement: {avg_improvement:+.1f}%")
    
    print(f"\nThroughput:")
    baseline_throughput = num_requests / baseline_total
    optimized_throughput = num_requests / optimized_total
    throughput_improvement = ((optimized_throughput - baseline_throughput) / baseline_throughput) * 100
    print(f"  Baseline:   {baseline_throughput:.2f} req/s")
    print(f"  Optimized:  {optimized_throughput:.2f} req/s")
    print(f"  Improvement: {throughput_improvement:+.1f}%")
    
    if total_improvement > 20:
        print("\n✅ EXCELLENT! Significant TTFT improvement in worst-case scenario!")
        print("   The optimization is highly effective when adapters must be reloaded")
    elif total_improvement > 10:
        print("\n✅ GOOD! Moderate TTFT improvement detected")
    elif total_improvement > 5:
        print("\n✅ Small but measurable improvement")
    elif total_improvement > 0:
        print("\n⚠️  Minimal improvement")
    else:
        print("\n❌ No improvement detected")
        print("\n🔍 Possible reasons:")
        print("   - Prefetch not completing before adapter is needed")
        print("   - Adapter loading still too fast relative to inference")
        print("   - Check logs for prefetch trigger messages")

if __name__ == "__main__":
    main()
