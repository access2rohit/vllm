#!/usr/bin/env python3
"""
Detailed profiling script for LoRA optimization with NVTX markers.

This script provides detailed profiling of LoRA loading operations
with NVTX markers for Nsight Systems analysis.
"""

import time
import torch
import nvtx
import os
import json
from typing import Dict, List, Any
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Disable internal vLLM NVTX to prevent conflicts
os.environ['ENABLE_LORA_NVTX'] = 'false'

class LoRAOptimizationProfiler:
    """Detailed profiler for LoRA optimization analysis."""
    
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        self.model_name = model_name
        self.lora_path = "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"
        
        self.test_prompts = [
            "What are frogs?",
            "Explain photosynthesis.", 
            "What is machine learning?",
            "How do computers work?",
            "What is artificial intelligence?"
        ]
        
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=20,  # Shorter for profiling focus
            top_p=1.0
        )
    
    def create_lora_requests(self, num_requests: int) -> List[LoRARequest]:
        """Create LoRA requests with different IDs to force loading."""
        requests = []
        for i in range(num_requests):
            lora_id = (i % 3) + 1  # Cycle through IDs 1, 2, 3
            requests.append(LoRARequest(
                lora_name=f"profile_lora_{lora_id}",
                lora_int_id=lora_id,
                lora_path=self.lora_path
            ))
        return requests
    
    def profile_baseline_loading(self, num_requests: int = 3) -> Dict[str, Any]:
        """Profile baseline LoRA loading with detailed NVTX markers."""
        print(f"🔍 Profiling baseline LoRA loading ({num_requests} requests)...")
        
        with nvtx.annotate("BASELINE_PROFILE_SETUP", color="purple"):
            llm = LLM(
                model=self.model_name,
                trust_remote_code=True,
                tensor_parallel_size=1,
                max_num_seqs=1,
                gpu_memory_utilization=0.8,
                enable_lora=True,
                max_loras=1,  # Force loading each time
                max_cpu_loras=1,
                max_lora_rank=128,
            )
        
        lora_requests = self.create_lora_requests(num_requests)
        prompts = self.test_prompts[:num_requests]
        
        # Warmup
        with nvtx.annotate("BASELINE_WARMUP", color="yellow"):
            llm.generate([prompts[0]], self.sampling_params, lora_request=lora_requests[0])
        
        # Clear cache
        with nvtx.annotate("BASELINE_CACHE_CLEAR", color="orange"):
            torch.cuda.empty_cache()
        
        # Profile each request with detailed timing
        request_profiles = []
        
        with nvtx.annotate("BASELINE_SEQUENTIAL_REQUESTS", color="red"):
            torch.cuda.cudart().cudaProfilerStart()
            
            for i, (prompt, lora_req) in enumerate(zip(prompts, lora_requests)):
                with nvtx.annotate(f"BASELINE_REQUEST_{i+1}_LORA_{lora_req.lora_int_id}", color="blue"):
                    
                    # Measure LoRA loading time
                    with nvtx.annotate(f"BASELINE_LORA_LOAD_{lora_req.lora_int_id}", color="green"):
                        load_start = time.time()
                        # The loading happens inside generate() call
                        
                        with nvtx.annotate(f"BASELINE_INFERENCE_{i+1}", color="cyan"):
                            outputs = llm.generate([prompt], self.sampling_params, lora_request=lora_req)
                        
                        load_end = time.time()
                    
                    request_time = load_end - load_start
                    request_profiles.append({
                        "request_id": i + 1,
                        "lora_id": lora_req.lora_int_id,
                        "time": request_time,
                        "output": outputs[0].outputs[0].text[:30]
                    })
                    
                    print(f"  Baseline Request {i+1} (LoRA {lora_req.lora_int_id}): {request_time:.3f}s")
            
            torch.cuda.cudart().cudaProfilerStop()
        
        return {
            "total_requests": num_requests,
            "request_profiles": request_profiles,
            "total_time": sum(r["time"] for r in request_profiles),
            "avg_time": sum(r["time"] for r in request_profiles) / len(request_profiles)
        }
    
    def profile_pipelined_loading(self, num_requests: int = 3) -> Dict[str, Any]:
        """Profile optimized LoRA loading with pipelining."""
        print(f"🚀 Profiling pipelined LoRA loading ({num_requests} requests)...")
        
        with nvtx.annotate("PIPELINED_PROFILE_SETUP", color="purple"):
            llm = LLM(
                model=self.model_name,
                trust_remote_code=True,
                tensor_parallel_size=1,
                max_num_seqs=1,
                gpu_memory_utilization=0.8,
                enable_lora=True,
                max_loras=2,  # Allow pipelining
                max_cpu_loras=4,
                max_lora_rank=128,
            )
        
        lora_requests = self.create_lora_requests(num_requests)
        prompts = self.test_prompts[:num_requests]
        
        # Warmup
        with nvtx.annotate("PIPELINED_WARMUP", color="yellow"):
            llm.generate([prompts[0]], self.sampling_params, lora_request=lora_requests[0])
        
        # Clear cache
        with nvtx.annotate("PIPELINED_CACHE_CLEAR", color="orange"):
            torch.cuda.empty_cache()
        
        # Profile pipelined requests
        request_profiles = []
        
        with nvtx.annotate("PIPELINED_REQUESTS", color="green"):
            torch.cuda.cudart().cudaProfilerStart()
            
            for i, (prompt, lora_req) in enumerate(zip(prompts, lora_requests)):
                with nvtx.annotate(f"PIPELINED_REQUEST_{i+1}_LORA_{lora_req.lora_int_id}", color="blue"):
                    
                    # Start prefetching next LoRA if available
                    if i + 1 < len(lora_requests):
                        next_lora = lora_requests[i + 1]
                        with nvtx.annotate(f"PREFETCH_START_LORA_{next_lora.lora_int_id}", color="magenta"):
                            # Prefetching should happen automatically through our implementation
                            pass
                    
                    # Measure request time (including any remaining loading)
                    with nvtx.annotate(f"PIPELINED_LORA_ACTIVATE_{lora_req.lora_int_id}", color="green"):
                        load_start = time.time()
                        
                        with nvtx.annotate(f"PIPELINED_INFERENCE_{i+1}", color="cyan"):
                            outputs = llm.generate([prompt], self.sampling_params, lora_request=lora_req)
                        
                        load_end = time.time()
                    
                    request_time = load_end - load_start
                    request_profiles.append({
                        "request_id": i + 1,
                        "lora_id": lora_req.lora_int_id,
                        "time": request_time,
                        "output": outputs[0].outputs[0].text[:30]
                    })
                    
                    print(f"  Pipelined Request {i+1} (LoRA {lora_req.lora_int_id}): {request_time:.3f}s")
            
            torch.cuda.cudart().cudaProfilerStop()
        
        return {
            "total_requests": num_requests,
            "request_profiles": request_profiles,
            "total_time": sum(r["time"] for r in request_profiles),
            "avg_time": sum(r["time"] for r in request_profiles) / len(request_profiles)
        }
    
    def profile_memory_usage(self):
        """Profile GPU memory usage during LoRA operations."""
        print("🧠 Profiling GPU memory usage...")
        
        def get_memory_info():
            if torch.cuda.is_available():
                return {
                    "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                    "reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
                    "max_allocated": torch.cuda.max_memory_allocated() / 1024**3  # GB
                }
            return {"allocated": 0, "reserved": 0, "max_allocated": 0}
        
        memory_profile = {"stages": []}
        
        # Initial memory
        torch.cuda.reset_peak_memory_stats()
        memory_profile["stages"].append({"stage": "initial", **get_memory_info()})
        
        # Model loading
        with nvtx.annotate("MEMORY_PROFILE_MODEL_LOAD", color="purple"):
            llm = LLM(
                model=self.model_name,
                trust_remote_code=True,
                tensor_parallel_size=1,
                max_num_seqs=1,
                gpu_memory_utilization=0.8,
                enable_lora=True,
                max_loras=2,
                max_cpu_loras=4,
                max_lora_rank=128,
            )
        memory_profile["stages"].append({"stage": "model_loaded", **get_memory_info()})
        
        # First LoRA load
        lora_req = LoRARequest("memory_test_1", 1, self.lora_path)
        with nvtx.annotate("MEMORY_PROFILE_FIRST_LORA", color="green"):
            llm.generate(["Test prompt"], self.sampling_params, lora_request=lora_req)
        memory_profile["stages"].append({"stage": "first_lora", **get_memory_info()})
        
        # Second LoRA load
        lora_req2 = LoRARequest("memory_test_2", 2, self.lora_path)
        with nvtx.annotate("MEMORY_PROFILE_SECOND_LORA", color="blue"):
            llm.generate(["Test prompt 2"], self.sampling_params, lora_request=lora_req2)
        memory_profile["stages"].append({"stage": "second_lora", **get_memory_info()})
        
        # Print memory usage
        print("\nGPU Memory Usage Profile:")
        for stage in memory_profile["stages"]:
            print(f"  {stage['stage']:15}: {stage['allocated']:.2f} GB allocated, "
                  f"{stage['reserved']:.2f} GB reserved, {stage['max_allocated']:.2f} GB peak")
        
        return memory_profile
    
    def run_comprehensive_profile(self):
        """Run comprehensive profiling analysis."""
        print("🔬 Comprehensive LoRA Optimization Profiling")
        print("=" * 60)
        
        results = {}
        
        # Profile baseline
        results["baseline"] = self.profile_baseline_loading(3)
        
        # Profile pipelined
        results["pipelined"] = self.profile_pipelined_loading(3)
        
        # Profile memory usage
        results["memory"] = self.profile_memory_usage()
        
        # Calculate improvements
        baseline_avg = results["baseline"]["avg_time"]
        pipelined_avg = results["pipelined"]["avg_time"]
        improvement = ((baseline_avg - pipelined_avg) / baseline_avg) * 100
        
        results["summary"] = {
            "baseline_avg_time": baseline_avg,
            "pipelined_avg_time": pipelined_avg,
            "improvement_pct": improvement,
            "baseline_total": results["baseline"]["total_time"],
            "pipelined_total": results["pipelined"]["total_time"],
            "total_improvement_pct": ((results["baseline"]["total_time"] - results["pipelined"]["total_time"]) / results["baseline"]["total_time"]) * 100
        }
        
        # Print summary
        print("\n📊 PROFILING SUMMARY")
        print("-" * 40)
        print(f"Baseline Average:    {baseline_avg:.3f}s")
        print(f"Pipelined Average:   {pipelined_avg:.3f}s")
        print(f"Per-Request Improvement: {improvement:+.1f}%")
        print(f"Total Time Improvement:  {results['summary']['total_improvement_pct']:+.1f}%")
        
        # Save results
        with open("lora_optimization_profile.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Detailed results saved to: lora_optimization_profile.json")
        
        return results

def main():
    """Run the profiling suite."""
    print("🔬 LoRA Optimization Profiling Suite")
    print("=" * 50)
    
    profiler = LoRAOptimizationProfiler()
    
    try:
        results = profiler.run_comprehensive_profile()
        
        print("\n🎯 Profiling completed!")
        print("📈 Check Nsight Systems for detailed GPU timeline analysis")
        print("💡 Look for NVTX markers showing pipelining vs baseline patterns")
        
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()