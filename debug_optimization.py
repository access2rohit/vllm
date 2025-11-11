#!/usr/bin/env python3
"""
Debug script to test if the optimization works when manually triggered.
"""

import os
import time
import logging
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Enable debug logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_manual_prefetch():
    """Test the optimization by manually triggering prefetch."""
    print("🔬 Testing Manual Prefetch Triggering")
    print("=" * 50)
    
    # Ensure optimization is enabled
    os.environ.pop('DISABLE_LORA_PREFETCH', None)
    
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
    
    # Try to find LoRA manager
    lora_manager = None
    try:
        # Try V1 engine path through engine_core
        if hasattr(llm.llm_engine, 'engine_core'):
            engine_core = llm.llm_engine.engine_core
            if hasattr(engine_core, 'core_engines') and engine_core.core_engines:
                print("⚠️  V1 engine detected - LoRA manager in separate process")
                print("   Cannot access LoRA manager directly for manual prefetch")
                print("   This explains why optimization isn't working!")
                return
    except Exception as e:
        print(f"Error accessing engine: {e}")
    
    if not lora_manager:
        print("❌ Could not access LoRA manager")
        print("🔍 This confirms the issue: optimization exists but can't be triggered")
        return
    
    print("✅ Found LoRA manager, testing manual prefetch...")
    
    # Test 1: First request (baseline)
    print("\n📊 Test 1: First request (no prefetch possible)")
    start_time = time.time()
    outputs1 = llm.generate(["What are frogs?"], sampling_params, lora_request=lora_requests[0])
    first_time = time.time() - start_time
    print(f"First request time: {first_time:.3f}s")
    
    # Manually trigger prefetch for second request
    print(f"\n🚀 Manually triggering prefetch for LoRA {lora_requests[1].lora_int_id}")
    try:
        lora_manager.prefetch_next_adapter(lora_requests[1])
        print("✅ Prefetch triggered")
        
        # Wait a bit for prefetch to work
        time.sleep(1.0)
        
    except Exception as e:
        print(f"❌ Prefetch failed: {e}")
    
    # Test 2: Second request (should benefit from prefetch)
    print("\n📊 Test 2: Second request (with manual prefetch)")
    start_time = time.time()
    outputs2 = llm.generate(["Explain photosynthesis."], sampling_params, lora_request=lora_requests[1])
    second_time = time.time() - start_time
    print(f"Second request time: {second_time:.3f}s")
    
    # Analysis
    improvement = ((first_time - second_time) / first_time) * 100
    print(f"\n📈 Results:")
    print(f"  First request:  {first_time:.3f}s")
    print(f"  Second request: {second_time:.3f}s")
    print(f"  Improvement:    {improvement:+.1f}%")
    
    if improvement > 10:
        print("✅ Manual prefetch is working!")
    else:
        print("❌ Manual prefetch not effective")

if __name__ == "__main__":
    test_manual_prefetch()