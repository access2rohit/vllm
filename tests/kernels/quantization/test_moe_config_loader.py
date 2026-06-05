# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config-loading tests for the fused-MoE tuned-config loader.

Pure file IO (path resolution + JSON parsing); no GPU kernels are launched.
Verifies that get_moe_configs honors VLLM_TUNED_CONFIG_FOLDER (env override),
reads the packaged configs dir, preserves the VLLM_BATCH_INVARIANT guard, and
that the int-key/triton_version normalization is behavior-preserving.

The expected JSON filename is reproduced via the loader's own
get_config_file_name so the tests match whatever GPU the suite runs on.
"""

import json

import pytest

from vllm.model_executor.layers.fused_moe import fused_moe

ENV = "VLLM_TUNED_CONFIG_FOLDER"

# Arbitrary MoE shape args; only used to build a matching filename.
E, N, DTYPE = 8, 14336, "fp8_w8a8"
BLOCK_N, BLOCK_K = 128, 128

SENTINEL = {1: {"BLOCK_SIZE_M": 64, "marker": "sentinel"}}


def _moe_file_name() -> str:
    block_shape = [BLOCK_N, BLOCK_K]
    return fused_moe.get_config_file_name(E, N, DTYPE, block_shape)


def _write(dir_path, file_name, payload):
    (dir_path / file_name).write_text(json.dumps(payload))


@pytest.fixture(autouse=True)
def _clear_cache():
    fused_moe.get_moe_configs.cache_clear()
    yield
    fused_moe.get_moe_configs.cache_clear()


def test_moe_packaged_path_loads(monkeypatch, tmp_path):
    """NO-BREAK: env unset, a config in _CONFIGS_DIR loads unchanged -- pins the
    pre-migration packaged read behavior."""
    monkeypatch.delenv(ENV, raising=False)
    _write(tmp_path, _moe_file_name(), {"1": SENTINEL[1]})
    monkeypatch.setattr(fused_moe, "_CONFIGS_DIR", str(tmp_path))
    assert fused_moe.get_moe_configs(E, N, DTYPE, BLOCK_N, BLOCK_K) == (SENTINEL)


def test_moe_env_override(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    pkg = tmp_path / "pkg"
    env_dir.mkdir()
    pkg.mkdir()
    _write(env_dir, _moe_file_name(), {"1": {"src": "env"}})
    _write(pkg, _moe_file_name(), {"1": {"src": "pkg"}})
    monkeypatch.setenv(ENV, str(env_dir))
    monkeypatch.setattr(fused_moe, "_CONFIGS_DIR", str(pkg))
    assert fused_moe.get_moe_configs(E, N, DTYPE, BLOCK_N, BLOCK_K) == {
        1: {"src": "env"}
    }


def test_moe_batch_invariant_returns_none(monkeypatch, tmp_path):
    """The VLLM_BATCH_INVARIANT early-return guard must be preserved: even with
    a matching packaged config present, the loader returns None."""
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    _write(tmp_path, _moe_file_name(), {"1": SENTINEL[1]})
    monkeypatch.setattr(fused_moe, "_CONFIGS_DIR", str(tmp_path))
    assert fused_moe.get_moe_configs(E, N, DTYPE, BLOCK_N, BLOCK_K) is None


def test_moe_isdigit_filter_is_noop(monkeypatch, tmp_path):
    """An int-keyed config carrying a triton_version key loads identically to
    one without it (the key is popped, digit filter keeps the int keys)."""
    monkeypatch.delenv(ENV, raising=False)
    _write(
        tmp_path,
        _moe_file_name(),
        {
            "1": {"BLOCK_SIZE_M": 64},
            "16": {"BLOCK_SIZE_M": 128},
            "triton_version": "3.6.0",
        },
    )
    monkeypatch.setattr(fused_moe, "_CONFIGS_DIR", str(tmp_path))
    assert fused_moe.get_moe_configs(E, N, DTYPE, BLOCK_N, BLOCK_K) == {
        1: {"BLOCK_SIZE_M": 64},
        16: {"BLOCK_SIZE_M": 128},
    }
