# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the tunable-kernel registry (vllm/utils/tuned_config_registry.py).

Pure metadata + file IO; no GPU/torch kernels required.
"""

import dataclasses
import json

import pytest

from vllm.utils.tuned_config_registry import (
    TUNABLE_KERNEL_REGISTRY,
    Applicability,
    TunableKernelSpec,
    get_tunable_kernels,
    get_tuned_config_for,
    register_tunable_kernel,
    tunable_triton_kernel,
)

ENV = "VLLM_TUNED_CONFIG_FOLDER"


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot/restore the global registry so tests do not leak into each
    other (or into the real seeded entries)."""
    saved = dict(TUNABLE_KERNEL_REGISTRY)
    TUNABLE_KERNEL_REGISTRY.clear()
    try:
        yield
    finally:
        TUNABLE_KERNEL_REGISTRY.clear()
        TUNABLE_KERNEL_REGISTRY.update(saved)


# ----------------------------- Phase 1: registry core ----------------------


def test_register_and_get():
    spec_b = TunableKernelSpec(
        name="b_kernel",
        applicability=Applicability.OPTIONAL,
        key_fn=lambda: "b.json",
        packaged_dir="/pkg/b",
    )
    spec_a = TunableKernelSpec(
        name="a_kernel",
        applicability=Applicability.COMPUTE_HEAVY_INPATH,
        key_fn=lambda: "a.json",
        packaged_dir="/pkg/a",
    )
    # Register out of order; get_tunable_kernels must sort by name.
    register_tunable_kernel(spec_b)
    register_tunable_kernel(spec_a)

    got = get_tunable_kernels()
    assert [s.name for s in got] == ["a_kernel", "b_kernel"]
    assert got[0] is spec_a and got[1] is spec_b


def test_duplicate_name_raises():
    spec = TunableKernelSpec(
        name="dup",
        applicability=Applicability.OPTIONAL,
        key_fn=lambda: "x.json",
        packaged_dir="/pkg",
    )
    register_tunable_kernel(spec)
    with pytest.raises(ValueError, match="already registered"):
        register_tunable_kernel(spec)


def test_spec_is_frozen():
    spec = TunableKernelSpec(
        name="frozen",
        applicability=Applicability.OPTIONAL,
        key_fn=lambda: "x.json",
        packaged_dir="/pkg",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "mutated"  # type: ignore[misc]


# ----------------------------- Phase 2: decorator --------------------------


def test_decorator_builds_and_loads(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path))

    def key_fn(n, k):
        return f"N={n},K={k}.json"

    @tunable_triton_kernel(
        name="fake_kernel",
        key_fn=key_fn,
        packaged_dir="/nonexistent/pkg",
        applicability=Applicability.COMPUTE_HEAVY_INPATH,
    )
    def get_fake_configs(n, k):  # body is replaced by the decorator
        raise AssertionError("original body must not run")

    # The sentinel file lives under the env override folder at key_fn's name.
    (tmp_path / "N=16,K=32.json").write_text(json.dumps({"1": {"BLOCK": 64}}))

    # Decorated getter builds the filename, loads, and normalizes to int keys.
    assert get_fake_configs(16, 32) == {1: {"BLOCK": 64}}

    # Tag attached and registered.
    spec = getattr(get_fake_configs, "__vllm_tunable__")
    assert isinstance(spec, TunableKernelSpec)
    assert spec.name == "fake_kernel"
    assert spec.key_fn is key_fn
    assert TUNABLE_KERNEL_REGISTRY["fake_kernel"] is spec


def test_decorator_returns_none_on_miss(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path))

    @tunable_triton_kernel(
        name="miss_kernel",
        key_fn=lambda n: f"N={n}.json",
        packaged_dir="/nonexistent/pkg",
    )
    def get_miss_configs(n):
        raise AssertionError("original body must not run")

    # No file written anywhere -> heuristic-fallback signal (None).
    assert get_miss_configs(99) is None


def test_decorator_preserves_lru_cache(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path))

    @tunable_triton_kernel(
        name="cached_kernel",
        key_fn=lambda n: f"N={n}.json",
        packaged_dir="/nonexistent/pkg",
    )
    def get_cached_configs(n):
        raise AssertionError("original body must not run")

    assert get_cached_configs(1) is None
    # First call missed -> cached None. Writing the file now should NOT change
    # the cached result until cache_clear, proving lru_cache is in effect.
    (tmp_path / "N=1.json").write_text(json.dumps({"1": {"BLOCK": 8}}))
    assert get_cached_configs(1) is None
    get_cached_configs.cache_clear()
    assert get_cached_configs(1) == {1: {"BLOCK": 8}}


# ----------------------------- Phase 3: seed the 4 real kernels ------------


def test_real_kernels_registered():
    # Importing the seed module registers the 4 shipped tunable kernels.
    from vllm.utils.tuned_config_registry_seed import seed_registry

    seed_registry()  # idempotent; the autouse fixture cleared the registry

    expected = {
        "fused_moe",
        "w8a8_block_fp8",
        "w8a8_block_int8",
        "selective_state_update",
    }
    by_name = {s.name: s for s in get_tunable_kernels()}
    assert expected <= set(by_name)

    for name in expected:
        spec = by_name[name]
        assert spec.applicability is Applicability.COMPUTE_HEAVY_INPATH
        # tuner_script is intentionally absent from the spec (D2): the tuner
        # linkage lives only in getter docstrings, not in registry metadata.
        assert not hasattr(spec, "tuner_script")


def test_real_kernel_key_fns_produce_valid_filenames():
    from vllm.utils.tuned_config_registry_seed import seed_registry

    seed_registry()
    by_name = {s.name: s for s in get_tunable_kernels()}

    # Representative args per kernel -> the loader's real filename f-string.
    samples = {
        "fused_moe": ((8, 14336, "fp8_w8a8"), {}),
        "w8a8_block_fp8": ((1536, 7168, 128, 128), {}),
        "w8a8_block_int8": ((1536, 7168, 128, 128), {}),
        "selective_state_update": ((64, 128, "float16"), {}),
    }
    for name, (args, kwargs) in samples.items():
        fname = by_name[name].key_fn(*args, **kwargs)
        assert fname.endswith(".json"), fname
        assert "device_name=" in fname, fname


def test_int8_key_fn_is_unspaced():
    # The motivating bug: a stray space in the INT8 block_shape made the
    # filename miss all 39 shipped configs. The registry must encode the
    # corrected, UNSPACED form.
    from vllm.utils.tuned_config_registry_seed import seed_registry

    seed_registry()
    by_name = {s.name: s for s in get_tunable_kernels()}
    fname = by_name["w8a8_block_int8"].key_fn(1536, 7168, 128, 128)
    assert "block_shape=[128,128]" in fname
    assert "block_shape=[128, 128]" not in fname


# ----------------------------- generic get_tuned_config_for ----------------


def test_get_tuned_config_for_nearest_M(monkeypatch, tmp_path):
    # Use a real registered kernel + its real key_fn end-to-end. The sentinel
    # config file is written at the EXACT filename the spec's key_fn produces,
    # under the VLLM_TUNED_CONFIG_FOLDER override so packaged_dir is bypassed.
    from vllm.utils.tuned_config_registry_seed import seed_registry

    seed_registry()
    monkeypatch.setenv(ENV, str(tmp_path))

    shape = dict(N=1536, K=7168, block_n=128, block_k=128)
    spec = TUNABLE_KERNEL_REGISTRY["w8a8_block_int8"]
    fname = spec.key_fn(**shape)
    (tmp_path / fname).write_text(
        json.dumps({"1": {"BLOCK_SIZE_M": 4}, "4096": {"BLOCK_SIZE_M": 64}})
    )

    # M=8 is nearest the "1" entry; M=3000 is nearest the "4096" entry.
    assert get_tuned_config_for("w8a8_block_int8", M=8, **shape) == {
        "BLOCK_SIZE_M": 4
    }
    assert get_tuned_config_for("w8a8_block_int8", M=3000, **shape) == {
        "BLOCK_SIZE_M": 64
    }


def test_get_tuned_config_for_miss_returns_none(monkeypatch, tmp_path):
    # No file present anywhere under the override -> resolve fails -> None.
    from vllm.utils.tuned_config_registry_seed import seed_registry

    seed_registry()
    monkeypatch.setenv(ENV, str(tmp_path))

    result = get_tuned_config_for(
        "w8a8_block_int8", M=8, N=1536, K=7168, block_n=128, block_k=128
    )
    assert result is None


def test_get_tuned_config_for_unknown_name_raises():
    # An unregistered name must fail loudly (KeyError), not silently return
    # None -- a typo'd kernel name is a programming error, not a config miss.
    with pytest.raises(KeyError):
        get_tuned_config_for("not_a_real_kernel", M=8, N=1, K=1)
