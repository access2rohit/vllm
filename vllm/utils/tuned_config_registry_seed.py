# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Seed the tunable-kernel registry with vLLM's 4 shipped tunable kernels.

Importing this module registers the MoE expert GEMM, block-FP8 GEMM,
block-INT8 GEMM, and Mamba selective_state_update kernels into
``TUNABLE_KERNEL_REGISTRY``. In production this would be imported at package
init; tests import it explicitly to obtain a complete manifest.

Each ``key_fn`` reproduces the EXACT filename f-string used by the
corresponding loader today (see the referenced module:line), so the registry's
view of a kernel's config filename matches what the kernel actually searches
for. Notably the INT8 template is UNSPACED (``block_shape=[{n},{k}]`` with no
space after the comma) -- the spaced variant is the ~14-month silent-drift bug
the helper work fixed, and encoding the correct form here lets the B2/B3
round-trip test guard it generically.

The packaged config directories are computed the same way the loaders compute
them (``<module dir>/configs[/...]``) so a manifest consumer can locate the
shipped configs.
"""

import vllm.model_executor.layers.fused_moe.fused_moe as _moe
import vllm.model_executor.layers.mamba.ops.mamba_ssm as _ssm
import vllm.model_executor.layers.quantization.utils.fp8_utils as _fp8
import vllm.model_executor.layers.quantization.utils.int8_utils as _int8
from vllm.platforms import current_platform
from vllm.utils.tuned_config_registry import (
    Applicability,
    TunableKernelSpec,
    register_tunable_kernel,
)


def _device_name() -> str:
    """vLLM device name with spaces replaced by underscores (as every loader
    does before building its filename)."""
    return current_platform.get_device_name().replace(" ", "_")


# ---- key_fn builders: each mirrors the loader's exact filename f-string -----


def _moe_key_fn(
    E: int, N: int, dtype: str | None, block_shape: list[int] | None = None
) -> str:
    # Mirrors fused_moe.get_config_file_name (fused_moe.py).
    return _moe.get_config_file_name(E, N, dtype, block_shape)


def _fp8_key_fn(N: int, K: int, block_n: int, block_k: int) -> str:
    # Mirrors fp8_utils.get_w8a8_block_fp8_configs (fp8_utils.py:818).
    device_name = _device_name()
    return (
        f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8,"
        f"block_shape=[{block_n},{block_k}].json"
    )


def _int8_key_fn(N: int, K: int, block_n: int, block_k: int) -> str:
    # Mirrors int8_utils.get_w8a8_block_int8_configs (int8_utils.py:376).
    # UNSPACED block_shape -- the correct form after the silent-drift fix.
    device_name = _device_name()
    return (
        f"N={N},K={K},device_name={device_name},dtype=int8_w8a8,"
        f"block_shape=[{block_n},{block_k}].json"
    )


def _ssu_key_fn(headdim: int, dstate: int, cache_dtype: str) -> str:
    # Mirrors mamba_ssm.get_ssm_configs (mamba_ssm.py): canonicalize cache
    # dtype, then build via the loader's own filename helper.
    cache_dtype = _ssm._canonical_cache_dtype(cache_dtype)
    return _ssm.get_ssm_config_file_name(headdim, dstate, cache_dtype, _device_name())


_SEED_SPECS = (
    TunableKernelSpec(
        name="fused_moe",
        applicability=Applicability.COMPUTE_HEAVY_INPATH,
        key_fn=_moe_key_fn,
        packaged_dir=_moe._CONFIGS_DIR,
    ),
    TunableKernelSpec(
        name="w8a8_block_fp8",
        applicability=Applicability.COMPUTE_HEAVY_INPATH,
        key_fn=_fp8_key_fn,
        packaged_dir=_fp8._CONFIGS_DIR,
    ),
    TunableKernelSpec(
        name="w8a8_block_int8",
        applicability=Applicability.COMPUTE_HEAVY_INPATH,
        key_fn=_int8_key_fn,
        packaged_dir=_int8._CONFIGS_DIR,
    ),
    TunableKernelSpec(
        name="selective_state_update",
        applicability=Applicability.COMPUTE_HEAVY_INPATH,
        key_fn=_ssu_key_fn,
        packaged_dir=_ssm._CONFIGS_DIR,
    ),
)


def seed_registry() -> None:
    """Register the 4 shipped tunable kernels. Idempotent: skips names already
    present so re-import (or a prior explicit registration) does not raise."""
    from vllm.utils.tuned_config_registry import TUNABLE_KERNEL_REGISTRY

    for spec in _SEED_SPECS:
        if spec.name not in TUNABLE_KERNEL_REGISTRY:
            register_tunable_kernel(spec)


# Populate on import.
seed_registry()
