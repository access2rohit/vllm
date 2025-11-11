#!/usr/bin/env python3
"""
Test script for batched/high-throughput scenarios.
This simulates a realistic production workload with multiple concurrent requests.
"""

import os
import time
import logging
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import List, Tuple

# Enable detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_batched_throughput(prefetch_enabled: bool, test_name: str, num_requests: int = 20):
    """Test throughput with batched requests using multiple LoRAs."""
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"Prefetch enabled: {prefetch_enabled}")
    print(f"Number of requests: {num_requests}")
    print(f"{'='*70}")
    
    # Set environment
    if prefetch_enabled:
        os.environ.pop('DISABLE_LORA_PREFETCH', None)
    else:
        os.environ['DISABLE_LORA_PREFETCH'] = '1'
    
    # Create LLM with batching support
    llm = LLM(
        model="openai/gpt-oss-20b",
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_num_seqs=4,  # Allow batching of up to 4 requests
        gpu_memory_utilization=0.3,
        enable_lora=True,
        max_loras=3,  # Support 3 LoRAs in cache
        max_cpu_loras=6,
        max_lora_rank=128,
    )
    
    # Define 4 different LoRAs
    lora_requests = [
        LoRARequest("lora_1", 1, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
        LoRARequest("lora_2", 2, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
        LoRARequest("lora_3", 3, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
        LoRARequest("lora_4", 4, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
    ]
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
    
    # Create a workload that cycles through LoRAs
    # This simulates a multi-tenant scenario where different users
    # use different LoRAs
    prompts_and_loras: List[Tuple[str, LoRARequest]] = []
    for i in range(num_requests):
        lora_idx = i % len(lora_requests)
        prompts_and_loras.append((
            f"Question {i+1}: What is the meaning of life?",
            lora_requests[lora_idx]
        ))
    
    print(f"📊 Processing {num_requests} requests with {len(lora_requests)} different LoRAs")
    print(f"   Batching enabled (max_num_seqs=4)")
    print(f"   LoRA pattern: {[i % len(lora_requests) + 1 for i in range(min(12, num_requests))]}")
    print()
    
    # Process all requests and measure total time
    start_time = time.time()
    
    # Submit all requests at once (simulating high load)
    all_prompts = [p for p, _ in prompts_and_loras]
    all_lora_requests = [l for _, l in prompts_and_loras]
    
    outputs = llm.generate(
        all_prompts,
        sampling_params,
        lora_request=all_lora_requests
    )
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    throughput = num_requests / total_time
    avg_latency = total_time / num_requests
    
    print(f"\n📈 Results for {test_name}:")
    print(f"  Total time:        {total_time:.2f}s")
    print(f"  Throughput:        {throughput:.2f} requests/second")
    print(f"  Avg latency:       {avg_latency:.3f}s per request")
    print(f"  Total requests:    {num_requests}")
    
    return {
        "test_name": test_name,
        "total_time": total_time,
        "throughput": throughput,
        "avg_latency": avg_latency,
        "num_requests": num_requests
    }

def main():
    """Run batched throughput test."""
    print("🔬 Batched/High-Throughput LoRA Performance Test")
    print("=" * 70)
    print("This test simulates a production workload with:")
    print("  - Multiple concurrent requests (batching)")
    print("  - Multiple LoRAs (multi-tenant scenario)")
    print("  - High request rate (requests coming faster than serving)")
    print("=" * 70)
    
    num_requests = 20
    
    # Test baseline
    print("\n🚫 Testing BASELINE (No Optimization)")
    baseline_result = test_batched_throughput(False, "BASELINE", num_requests)
    
    print("\n" + "="*70)
    print("Waiting 5 seconds between tests...")
    time.sleep(5)
    
    # Test optimized
    print("\n🚀 Testing OPTIMIZED (With Prefetching)")
    optimized_result = test_batched_throughput(True, "OPTIMIZED", num_requests)
    
    # Compare
    print(f"\n{'='*70}")
    print("🎯 COMPARISON")
    print(f"{'='*70}")
    
    baseline_throughput = baseline_result["throughput"]
    optimized_throughput = optimized_result["throughput"]
    throughput_improvement = ((optimized_throughput - baseline_throughput) / baseline_throughput) * 100
    
    baseline_latency = baseline_result["avg_latency"]
    optimized_latency = optimized_result["avg_latency"]
    latency_improvement = ((baseline_latency - optimized_latency) / baseline_latency) * 100
    
    print(f"\nThroughput:")
    print(f"  Baseline:   {baseline_throughput:.2f} req/s")
    print(f"  Optimized:  {optimized_throughput:.2f} req/s")
    print(f"  Improvement: {throughput_improvement:+.1f}%")
    
    print(f"\nAverage Latency:")
    print(f"  Baseline:   {baseline_latency:.3f}s")
    print(f"  Optimized:  {optimized_latency:.3f}s")
    print(f"  Improvement: {latency_improvement:+.1f}%")
    
    print(f"\nTotal Time:")
    print(f"  Baseline:   {baseline_result['total_time']:.2f}s")
    print(f"  Optimized:  {optimized_result['total_time']:.2f}s")
    print(f"  Time saved: {baseline_result['total_time'] - optimized_result['total_time']:.2f}s")
    
    if throughput_improvement > 10:
        print("\n✅ Significant throughput improvement!")
        print("   The optimization is working well in batched scenarios")
    elif throughput_improvement > 5:
        print("\n✅ Moderate throughput improvement")
    elif throughput_improvement > 0:
        print("\n⚠️  Small throughput improvement")
    else:
        print("\n❌ No improvement or regression")
        print("\n🔍 In batched scenarios, the optimization should help because:")
        print("   - Multiple requests in queue = can look ahead")
        print("   - Longer processing time = prefetch has time to complete")
        print("   - If no improvement, check if LoRAs are being cached")

if __name__ == "__main__":
    main()
