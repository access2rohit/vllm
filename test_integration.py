#!/usr/bin/env python3
"""
Test script to verify the LoRA optimization integration is working.
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

def test_integration():
    """Test that the optimization is properly integrated."""
    print("🔬 Testing LoRA Optimization Integration")
    print("=" * 60)
    
    # Ensure optimization is enabled
    os.environ.pop('DISABLE_LORA_PREFETCH', None)
    
    # Create LLM
    print("Creating LLM...")
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
    
    # Define LoRA requests (different IDs to force loading)
    lora_requests = [
        LoRARequest("test_1", 1, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
        LoRARequest("test_2", 2, "/fsx/srivrohi/hf_cache/tat-qa/gpt-oss-20b/moe-all-eager-0-rank-128/lora_adapter"),
    ]
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
    
    print("\n📊 Running test...")
    print("Look for log messages like:")
    print("  - '🚀 Triggering prefetch for LoRA X'")
    print("  - '🔄 start_prefetch called'")
    print("  - '✅ Prefetch completed'")
    print("  - '⚡ LoRA X loaded from prefetch cache'")
    print()
    
    # First request
    print(f"Request 1: Using LoRA {lora_requests[0].lora_int_id}")
    start_time = time.time()
    outputs1 = llm.generate(["What are frogs?"], sampling_params, lora_request=lora_requests[0])
    first_time = time.time() - start_time
    print(f"✅ Completed in {first_time:.3f}s\n")
    
    # Second request (should trigger prefetch and benefit from it)
    print(f"Request 2: Using LoRA {lora_requests[1].lora_int_id}")
    start_time = time.time()
    outputs2 = llm.generate(["Explain photosynthesis."], sampling_params, lora_request=lora_requests[1])
    second_time = time.time() - start_time
    print(f"✅ Completed in {second_time:.3f}s\n")
    
    # Analysis
    improvement = ((first_time - second_time) / first_time) * 100
    print(f"📈 Results:")
    print(f"  First request:  {first_time:.3f}s")
    print(f"  Second request: {second_time:.3f}s")
    print(f"  Improvement:    {improvement:+.1f}%")
    
    if improvement > 10:
        print("\n✅ Optimization appears to be working!")
        print("   Check the logs above for prefetch messages")
    else:
        print("\n⚠️  No significant improvement detected")
        print("   Check if prefetch messages appeared in the logs")

if __name__ == "__main__":
    test_integration()