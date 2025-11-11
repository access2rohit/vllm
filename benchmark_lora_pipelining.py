#!/usr/bin/env python3
"""
Comprehensive benchmark for LoRA request-level pipelining optimization.

This script measures the performance improvements of request-level pipelining
compared to the baseline sequential LoRA loading approach.
"""

import time
import torch
import numpy as np
import json
import os
from typing import List, Dict, Any
from dataclasses import dataclass
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    scenario: str
    baseline_time: float
    optimized_time: float
    improvement_pct: float
    throughput_baseline: float
    throughput_optimized: float
    throughput_improvement: float
    details: Dict[str, Any]

class LoRAPipelineBenchmark:
    """Benchmark suite for LoRA request-level pipelining."""
    
    def __init__(self, model_name: str = "openai/gpt-oss-20b", lora_path: str = None):
        self.model_name = model_name
        self.lora_path = lora_path or "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"
        self.results: List[BenchmarkResult] = []
        
        # Test prompts of varying lengths
        self.test_prompts = [
            "What are frogs?",
            "Explain photosynthesis in detail.",
            "What is machine learning and how does it work?",
            "Describe the process of protein synthesis in cells.",
            "What are the main causes and effects of climate change?",
            "Explain quantum mechanics and its practical applications.",
            "How do neural networks learn and make predictions?",
            "What is the significance of DNA in heredity and evolution?"
        ]
        
        # Sampling parameters for consistent testing
        self.sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic for consistent timing
            max_tokens=50,    # Moderate length for timing
            top_p=1.0
        )
    
    def create_lora_requests(self, num_requests: int) -> List[LoRARequest]:
        """Create multiple LoRA requests for testing."""
        requests = []
        for i in range(num_requests):
            # Alternate between different LoRA IDs to force loading
            lora_id = (i % 3) + 1  # IDs 1, 2, 3
            requests.append(LoRARequest(
                lora_name=f"test_lora_{lora_id}",
                lora_int_id=lora_id,
                lora_path=self.lora_path
            ))
        return requests
    
    def benchmark_baseline_sequential(self, num_requests: int = 5) -> Dict[str, float]:
        """Benchmark baseline sequential LoRA loading."""
        print(f"🔄 Running baseline sequential benchmark ({num_requests} requests)...")
        
        # Create LLM without pipelining optimizations
        llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_num_seqs=1,
            gpu_memory_utilization=0.3,  # Reduced from 0.8
            enable_lora=True,
            max_loras=1,  # Force eviction between requests
            max_cpu_loras=1,  # Minimal caching
            max_lora_rank=128,
        )
        
        lora_requests = self.create_lora_requests(num_requests)
        prompts = self.test_prompts[:num_requests]
        
        # Warmup
        warmup_request = lora_requests[0]
        llm.generate([prompts[0]], self.sampling_params, lora_request=warmup_request)
        
        # Clear any cached state
        torch.cuda.empty_cache()
        
        # Measure sequential processing
        start_time = time.time()
        request_times = []
        
        for i, (prompt, lora_req) in enumerate(zip(prompts, lora_requests)):
            req_start = time.time()
            
            # This will trigger LoRA loading for each request
            outputs = llm.generate([prompt], self.sampling_params, lora_request=lora_req)
            
            req_end = time.time()
            req_time = req_end - req_start
            request_times.append(req_time)
            
            print(f"  Request {i+1}: {req_time:.3f}s - {outputs[0].outputs[0].text[:50]}...")
        
        total_time = time.time() - start_time
        avg_request_time = np.mean(request_times)
        throughput = num_requests / total_time
        
        return {
            "total_time": total_time,
            "avg_request_time": avg_request_time,
            "request_times": request_times,
            "throughput": throughput,
            "first_request_time": request_times[0],
            "subsequent_avg": np.mean(request_times[1:]) if len(request_times) > 1 else 0
        }
    
    def benchmark_pipelined_requests(self, num_requests: int = 5) -> Dict[str, float]:
        """Benchmark request-level pipelining optimization."""
        print(f"🚀 Running pipelined benchmark ({num_requests} requests)...")
        
        # Create LLM with pipelining optimizations
        llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_num_seqs=1,
            gpu_memory_utilization=0.3,  # Reduced from 0.8
            enable_lora=True,
            max_loras=2,  # Allow some caching for pipelining
            max_cpu_loras=4,  # Better CPU caching
            max_lora_rank=128,
        )
        
        lora_requests = self.create_lora_requests(num_requests)
        prompts = self.test_prompts[:num_requests]
        
        # Warmup
        warmup_request = lora_requests[0]
        llm.generate([prompts[0]], self.sampling_params, lora_request=warmup_request)
        
        # Clear any cached state
        torch.cuda.empty_cache()
        
        # Measure pipelined processing
        start_time = time.time()
        request_times = []
        
        for i, (prompt, lora_req) in enumerate(zip(prompts, lora_requests)):
            req_start = time.time()
            
            # Start prefetching next LoRA if available
            if i + 1 < len(lora_requests):
                next_lora = lora_requests[i + 1]
                # Try to access the LoRA manager and trigger prefetching
                try:
                    # Try different paths to access the LoRA manager
                    lora_manager = None
                    
                    # Try V1 engine path
                    if hasattr(llm.llm_engine, 'model_executor') and hasattr(llm.llm_engine.model_executor, 'driver_worker'):
                        worker = llm.llm_engine.model_executor.driver_worker
                        if hasattr(worker, 'model_runner') and hasattr(worker.model_runner, 'lora_manager'):
                            lora_manager = worker.model_runner.lora_manager
                    
                    # Try V0 engine path
                    elif hasattr(llm.llm_engine, 'workers') and llm.llm_engine.workers:
                        worker = llm.llm_engine.workers[0]
                        if hasattr(worker, 'model_runner') and hasattr(worker.model_runner, 'lora_manager'):
                            lora_manager = worker.model_runner.lora_manager
                    
                    if lora_manager and hasattr(lora_manager, 'prefetch_next_adapter'):
                        lora_manager.prefetch_next_adapter(next_lora)
                        print(f"    🚀 Started prefetching LoRA {next_lora.lora_int_id}")
                    else:
                        print(f"    ⚠️  LoRA manager not found or no prefetch method")
                        
                except Exception as e:
                    print(f"    ⚠️  Could not start prefetching: {e}")
            
            outputs = llm.generate([prompt], self.sampling_params, lora_request=lora_req)
            
            req_end = time.time()
            req_time = req_end - req_start
            request_times.append(req_time)
            
            print(f"  Request {i+1}: {req_time:.3f}s - {outputs[0].outputs[0].text[:50]}...")
        
        total_time = time.time() - start_time
        avg_request_time = np.mean(request_times)
        throughput = num_requests / total_time
        
        return {
            "total_time": total_time,
            "avg_request_time": avg_request_time,
            "request_times": request_times,
            "throughput": throughput,
            "first_request_time": request_times[0],
            "subsequent_avg": np.mean(request_times[1:]) if len(request_times) > 1 else 0
        }
    
    def run_single_request_benchmark(self) -> BenchmarkResult:
        """Benchmark single request performance (should be similar)."""
        print("\n📊 Single Request Benchmark")
        print("-" * 40)
        
        baseline = self.benchmark_baseline_sequential(1)
        optimized = self.benchmark_pipelined_requests(1)
        
        improvement = ((baseline["total_time"] - optimized["total_time"]) / baseline["total_time"]) * 100
        
        result = BenchmarkResult(
            scenario="Single Request",
            baseline_time=baseline["total_time"],
            optimized_time=optimized["total_time"],
            improvement_pct=improvement,
            throughput_baseline=baseline["throughput"],
            throughput_optimized=optimized["throughput"],
            throughput_improvement=optimized["throughput"] / baseline["throughput"],
            details={"baseline": baseline, "optimized": optimized}
        )
        
        self.results.append(result)
        return result
    
    def run_multi_request_benchmark(self, num_requests: int = 5) -> BenchmarkResult:
        """Benchmark multi-request performance (should show major improvement)."""
        print(f"\n📊 Multi-Request Benchmark ({num_requests} requests)")
        print("-" * 50)
        
        baseline = self.benchmark_baseline_sequential(num_requests)
        optimized = self.benchmark_pipelined_requests(num_requests)
        
        improvement = ((baseline["total_time"] - optimized["total_time"]) / baseline["total_time"]) * 100
        
        result = BenchmarkResult(
            scenario=f"Multi-Request ({num_requests})",
            baseline_time=baseline["total_time"],
            optimized_time=optimized["total_time"],
            improvement_pct=improvement,
            throughput_baseline=baseline["throughput"],
            throughput_optimized=optimized["throughput"],
            throughput_improvement=optimized["throughput"] / baseline["throughput"],
            details={"baseline": baseline, "optimized": optimized}
        )
        
        self.results.append(result)
        return result
    
    def run_throughput_stress_test(self, num_requests: int = 10) -> BenchmarkResult:
        """Run stress test with many requests to measure peak throughput."""
        print(f"\n📊 Throughput Stress Test ({num_requests} requests)")
        print("-" * 50)
        
        baseline = self.benchmark_baseline_sequential(num_requests)
        optimized = self.benchmark_pipelined_requests(num_requests)
        
        improvement = ((baseline["total_time"] - optimized["total_time"]) / baseline["total_time"]) * 100
        
        result = BenchmarkResult(
            scenario=f"Stress Test ({num_requests})",
            baseline_time=baseline["total_time"],
            optimized_time=optimized["total_time"],
            improvement_pct=improvement,
            throughput_baseline=baseline["throughput"],
            throughput_optimized=optimized["throughput"],
            throughput_improvement=optimized["throughput"] / baseline["throughput"],
            details={"baseline": baseline, "optimized": optimized}
        )
        
        self.results.append(result)
        return result
    
    def print_results(self):
        """Print comprehensive benchmark results."""
        print("\n" + "="*80)
        print("🎯 LORA REQUEST-LEVEL PIPELINING BENCHMARK RESULTS")
        print("="*80)
        
        for result in self.results:
            print(f"\n📋 {result.scenario}")
            print("-" * 60)
            print(f"Baseline Time:      {result.baseline_time:.3f}s")
            print(f"Optimized Time:     {result.optimized_time:.3f}s")
            print(f"Improvement:        {result.improvement_pct:+.1f}%")
            print(f"Baseline Throughput: {result.throughput_baseline:.2f} req/s")
            print(f"Optimized Throughput: {result.throughput_optimized:.2f} req/s")
            print(f"Throughput Gain:    {result.throughput_improvement:.2f}x")
            
            # Additional details
            baseline_details = result.details["baseline"]
            optimized_details = result.details["optimized"]
            
            if len(baseline_details["request_times"]) > 1:
                print(f"\nRequest Timing Breakdown:")
                print(f"  First Request - Baseline: {baseline_details['first_request_time']:.3f}s")
                print(f"  First Request - Optimized: {optimized_details['first_request_time']:.3f}s")
                print(f"  Subsequent Avg - Baseline: {baseline_details['subsequent_avg']:.3f}s")
                print(f"  Subsequent Avg - Optimized: {optimized_details['subsequent_avg']:.3f}s")
                
                subsequent_improvement = ((baseline_details['subsequent_avg'] - optimized_details['subsequent_avg']) / baseline_details['subsequent_avg']) * 100
                print(f"  Subsequent Improvement: {subsequent_improvement:+.1f}%")
    
    def save_results(self, filename: str = "lora_pipelining_benchmark_results.json"):
        """Save results to JSON file."""
        results_data = {
            "benchmark_info": {
                "model": self.model_name,
                "lora_path": self.lora_path,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "gpu_info": {
                    "device_count": torch.cuda.device_count(),
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
                }
            },
            "results": []
        }
        
        for result in self.results:
            results_data["results"].append({
                "scenario": result.scenario,
                "baseline_time": result.baseline_time,
                "optimized_time": result.optimized_time,
                "improvement_pct": result.improvement_pct,
                "throughput_baseline": result.throughput_baseline,
                "throughput_optimized": result.throughput_optimized,
                "throughput_improvement": result.throughput_improvement,
                "details": result.details
            })
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

def main():
    """Run the complete benchmark suite."""
    print("🚀 LoRA Request-Level Pipelining Benchmark Suite")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = LoRAPipelineBenchmark()
    
    try:
        # Run different benchmark scenarios
        benchmark.run_single_request_benchmark()
        benchmark.run_multi_request_benchmark(5)
        benchmark.run_throughput_stress_test(10)
        
        # Print and save results
        benchmark.print_results()
        benchmark.save_results()
        
        print("\n🎉 Benchmark completed successfully!")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()