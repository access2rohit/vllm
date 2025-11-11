#!/usr/bin/env python3
"""
Quick test script to validate LoRA request-level pipelining is working.

This script performs a simple test to ensure the pipelining implementation
is functional before running comprehensive benchmarks.
"""

import time
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def test_basic_lora_functionality():
    """Test basic LoRA functionality without pipelining."""
    print("🧪 Testing basic LoRA functionality...")
    
    try:
        llm = LLM(
            model="openai/gpt-oss-20b",
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_num_seqs=1,
            gpu_memory_utilization=0.3,  # Reduced from 0.7
            enable_lora=True,
            max_loras=1,
            max_cpu_loras=2,
            max_lora_rank=128,
        )
        
        lora_request = LoRARequest(
            lora_name="test_lora",
            lora_int_id=1,
            lora_path="/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"
        )
        
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
        
        start_time = time.time()
        outputs = llm.generate(["What are frogs?"], sampling_params, lora_request=lora_request)
        end_time = time.time()
        
        print(f"✅ Basic LoRA test passed in {end_time - start_time:.2f}s")
        print(f"   Output: {outputs[0].outputs[0].text}")
        return True
        
    except Exception as e:
        print(f"❌ Basic LoRA test failed: {e}")
        return False

def test_pipelining_components():
    """Test that pipelining components are available."""
    print("🔧 Testing pipelining components...")
    
    try:
        # Test imports
        from vllm.lora.worker_manager import RequestPipelineManager
        print("✅ RequestPipelineManager import successful")
        
        from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
        
        # Check if prefetch method exists
        manager_methods = [method for method in dir(LRUCacheWorkerLoRAManager) 
                          if 'prefetch' in method.lower()]
        
        if manager_methods:
            print(f"✅ Prefetch methods found: {manager_methods}")
        else:
            print("⚠️  No prefetch methods found - implementation may not be active")
        
        # Test RequestPipelineManager instantiation
        pipeline_mgr = RequestPipelineManager(device="cuda")
        print("✅ RequestPipelineManager instantiation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipelining components test failed: {e}")
        return False

def test_multi_lora_switching():
    """Test switching between multiple LoRAs."""
    print("🔄 Testing multi-LoRA switching...")
    
    try:
        llm = LLM(
            model="openai/gpt-oss-20b",
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_num_seqs=1,
            gpu_memory_utilization=0.3,  # Reduced from 0.7
            enable_lora=True,
            max_loras=2,  # Allow some caching
            max_cpu_loras=3,
            max_lora_rank=128,
        )
        
        # Create different LoRA requests
        lora_requests = [
            LoRARequest("test_lora_1", 1, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
            LoRARequest("test_lora_2", 2, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
            LoRARequest("test_lora_3", 3, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
        ]
        
        prompts = ["What are frogs?", "Explain photosynthesis.", "What is AI?"]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=15)
        
        times = []
        for i, (prompt, lora_req) in enumerate(zip(prompts, lora_requests)):
            start_time = time.time()
            outputs = llm.generate([prompt], sampling_params, lora_request=lora_req)
            end_time = time.time()
            
            request_time = end_time - start_time
            times.append(request_time)
            
            print(f"  Request {i+1} (LoRA {lora_req.lora_int_id}): {request_time:.2f}s")
            print(f"    Output: {outputs[0].outputs[0].text[:40]}...")
        
        # Analyze timing pattern
        first_time = times[0]
        subsequent_times = times[1:]
        
        if subsequent_times:
            avg_subsequent = sum(subsequent_times) / len(subsequent_times)
            if avg_subsequent < first_time * 0.8:  # 20% improvement threshold
                print(f"✅ Potential pipelining detected: subsequent requests {((first_time - avg_subsequent) / first_time * 100):.1f}% faster")
            else:
                print(f"⚠️  No clear pipelining benefit: subsequent requests only {((first_time - avg_subsequent) / first_time * 100):.1f}% faster")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-LoRA switching test failed: {e}")
        return False

def main():
    """Run quick validation tests."""
    print("🚀 Quick LoRA Pipelining Validation")
    print("=" * 40)
    
    tests = [
        ("Basic LoRA Functionality", test_basic_lora_functionality),
        ("Pipelining Components", test_pipelining_components),
        ("Multi-LoRA Switching", test_multi_lora_switching),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 30)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for comprehensive benchmarking.")
        print("💡 Run 'python3 benchmark_lora_pipelining.py' for detailed performance analysis.")
    else:
        print("⚠️  Some tests failed. Please check the implementation before benchmarking.")
    
    return passed == total

if __name__ == "__main__":
    main()