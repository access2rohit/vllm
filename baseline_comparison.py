#!/usr/bin/env python3
"""
Baseline comparison script to test LoRA optimization against unoptimized baseline.
Includes proper warmup for both scenarios.
"""

import os
import time
import json
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def run_lora_test(prefetch_enabled: bool, test_name: str, num_warmup: int = 2, num_test: int = 3) -> Dict[str, Any]:
    """Run LoRA test with proper warmup and multiple measurements."""
    print(f"\n{'='*60}")
    print(f"Running {test_name}")
    print(f"Prefetch enabled: {prefetch_enabled}")
    print(f"Warmup runs: {num_warmup}, Test runs: {num_test}")
    print(f"{'='*60}")
    
    # Set environment variable for this test
    if prefetch_enabled:
        os.environ.pop('DISABLE_LORA_PREFETCH', None)  # Remove if exists
    else:
        os.environ['DISABLE_LORA_PREFETCH'] = '1'
    
    try:
        # Create LLM instance
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
        
        # Define LoRA requests (different IDs, same path for controlled testing)
        lora_requests = [
            LoRARequest("test_1", 1, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
            LoRARequest("test_2", 2, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
        ]
        
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
        
        # Warmup phase
        print(f"🔥 Warmup phase ({num_warmup} runs)...")
        for i in range(num_warmup):
            print(f"  Warmup {i+1}/{num_warmup}")
            # First request
            llm.generate(["Warmup question 1?"], sampling_params, lora_request=lora_requests[0])
            # Second request  
            llm.generate(["Warmup question 2?"], sampling_params, lora_request=lora_requests[1])
        
        # Test phase
        print(f"📊 Test phase ({num_test} runs)...")
        first_times = []
        second_times = []
        
        for i in range(num_test):
            print(f"  Test run {i+1}/{num_test}")
            
            # First request (baseline measurement)
            start_time = time.time()
            outputs1 = llm.generate([f"Test question {i+1} part 1?"], sampling_params, lora_request=lora_requests[0])
            first_time = time.time() - start_time
            first_times.append(first_time)
            print(f"    First request: {first_time:.3f}s")
            
            # Second request (optimization target)
            start_time = time.time()
            outputs2 = llm.generate([f"Test question {i+1} part 2?"], sampling_params, lora_request=lora_requests[1])
            second_time = time.time() - start_time
            second_times.append(second_time)
            print(f"    Second request: {second_time:.3f}s")
        
        # Calculate statistics
        avg_first = sum(first_times) / len(first_times)
        avg_second = sum(second_times) / len(second_times)
        improvement = ((avg_first - avg_second) / avg_first) * 100
        
        return {
            "test_name": test_name,
            "prefetch_enabled": prefetch_enabled,
            "success": True,
            "num_warmup": num_warmup,
            "num_test": num_test,
            "first_times": first_times,
            "second_times": second_times,
            "avg_first_time": avg_first,
            "avg_second_time": avg_second,
            "improvement_percent": improvement,
            "min_first": min(first_times),
            "max_first": max(first_times),
            "min_second": min(second_times),
            "max_second": max(second_times)
        }
        
    except Exception as e:
        return {
            "test_name": test_name,
            "prefetch_enabled": prefetch_enabled,
            "success": False,
            "error": str(e)
        }

def main():
    """Run baseline comparison tests with proper warmup."""
    print("🔬 LoRA Optimization Baseline Comparison with Warmup")
    print("=" * 70)
    
    results = []
    
    # Test 1: Baseline (optimization disabled)
    print("\n🚫 Testing BASELINE (No Optimization)")
    baseline_result = run_lora_test(
        prefetch_enabled=False,
        test_name="BASELINE (No Optimization)",
        num_warmup=2,
        num_test=3
    )
    results.append(baseline_result)
    
    # Small delay between tests to ensure clean state
    print("\n⏳ Waiting 5 seconds between tests...")
    time.sleep(5)
    
    # Test 2: Optimized (optimization enabled)
    print("\n🚀 Testing OPTIMIZED (With Prefetching)")
    optimized_result = run_lora_test(
        prefetch_enabled=True,
        test_name="OPTIMIZED (With Prefetching)",
        num_warmup=2,
        num_test=3
    )
    results.append(optimized_result)
    
    # Analysis
    print(f"\n{'='*70}")
    print("📊 DETAILED COMPARISON RESULTS")
    print(f"{'='*70}")
    
    for result in results:
        if result["success"]:
            print(f"\n{result['test_name']}:")
            print(f"  Warmup runs:        {result['num_warmup']}")
            print(f"  Test runs:          {result['num_test']}")
            print(f"  Avg first request:  {result['avg_first_time']:.3f}s (range: {result['min_first']:.3f}-{result['max_first']:.3f}s)")
            print(f"  Avg second request: {result['avg_second_time']:.3f}s (range: {result['min_second']:.3f}-{result['max_second']:.3f}s)")
            print(f"  Internal improvement: {result['improvement_percent']:+.1f}%")
            print(f"  Individual times:")
            for i, (first, second) in enumerate(zip(result['first_times'], result['second_times'])):
                print(f"    Run {i+1}: {first:.3f}s → {second:.3f}s")
        else:
            print(f"\n{result['test_name']}: FAILED")
            if "error" in result:
                print(f"  Error: {result['error']}")
    
    # Calculate TRUE optimization effectiveness
    if baseline_result["success"] and optimized_result["success"]:
        baseline_avg_second = baseline_result["avg_second_time"]
        optimized_avg_second = optimized_result["avg_second_time"]
        
        true_improvement = ((baseline_avg_second - optimized_avg_second) / baseline_avg_second) * 100
        
        print(f"\n🎯 TRUE OPTIMIZATION EFFECTIVENESS:")
        print(f"{'='*50}")
        print(f"  Baseline avg second request:  {baseline_avg_second:.3f}s")
        print(f"  Optimized avg second request: {optimized_avg_second:.3f}s")
        print(f"  TRUE improvement:             {true_improvement:+.1f}%")
        print(f"  Absolute time saved:          {baseline_avg_second - optimized_avg_second:.3f}s")
        
        # Statistical significance check (simple)
        baseline_times = baseline_result["second_times"]
        optimized_times = optimized_result["second_times"]
        baseline_min = min(baseline_times)
        optimized_max = max(optimized_times)
        
        if optimized_max < baseline_min:
            print(f"  📈 Strong evidence: All optimized times < all baseline times")
        elif true_improvement > 10:
            print(f"  ✅ Significant improvement detected!")
        elif true_improvement > 5:
            print(f"  ✅ Moderate improvement detected")
        elif true_improvement > 0:
            print(f"  ⚠️  Small improvement detected")
        else:
            print(f"  ❌ No improvement or regression detected")
            
        # Throughput analysis
        baseline_throughput = 1.0 / baseline_avg_second
        optimized_throughput = 1.0 / optimized_avg_second
        throughput_improvement = ((optimized_throughput - baseline_throughput) / baseline_throughput) * 100
        
        print(f"\n📈 THROUGHPUT ANALYSIS:")
        print(f"  Baseline throughput:  {baseline_throughput:.2f} requests/second")
        print(f"  Optimized throughput: {optimized_throughput:.2f} requests/second")
        print(f"  Throughput improvement: {throughput_improvement:+.1f}%")
        
    else:
        print(f"\n❌ Could not compare - one or both tests failed")
    
    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"baseline_comparison_results_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {filename}")
    
    return results

if __name__ == "__main__":
    main()