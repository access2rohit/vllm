# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the B3 AST kernel-discovery tool."""

from tools.tuned_config.discover_kernels import discover_triton_kernels


def test_finds_known_kernels():
    kernels = discover_triton_kernels("vllm")

    # A well-known attention kernel.
    assert (
        "vllm.v1.attention.ops.triton_unified_attention:kernel_unified_attention"
        in kernels
    )
    # The 4 @triton.jit kernels backing the tunable loaders.
    assert (
        "vllm.model_executor.layers.fused_moe.fused_moe:fused_moe_kernel" in kernels
    )
    assert (
        "vllm.model_executor.layers.quantization.utils.fp8_utils:"
        "_w8a8_triton_block_scaled_mm" in kernels
    )
    assert (
        "vllm.model_executor.layers.quantization.utils.int8_utils:"
        "_w8a8_block_int8_matmul" in kernels
    )
    assert (
        "vllm.model_executor.layers.mamba.ops.mamba_ssm:"
        "_selective_scan_update_kernel" in kernels
    )


def test_count_in_sane_band():
    # We measured 272 @triton.jit decorators across 115 files; the function-id
    # count (some files define multiple) lands higher. Allow a wide band so the
    # test is robust to upstream churn but still catches a broken walk.
    kernels = discover_triton_kernels("vllm")
    assert len(kernels) >= 200, f"only found {len(kernels)} kernels"


def test_ids_are_module_qualified():
    kernels = discover_triton_kernels("vllm")
    # Every id must be "<dotted.module>:<func>" so same-named kernels in
    # different modules do not collide.
    for kid in kernels:
        assert ":" in kid, kid
        mod, func = kid.split(":", 1)
        assert mod.startswith("vllm"), kid
        assert func and "." not in func, kid
