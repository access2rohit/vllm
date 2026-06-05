# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config-loading tests for the block FP8 / INT8 GEMM tuned-config loaders.

These exercise pure file IO (path resolution + JSON parsing) -- no GPU kernels
are launched -- so unlike tests/kernels/quantization/test_block_fp8.py there is
no module-level capability skip. They verify that get_w8a8_block_fp8_configs and
get_w8a8_block_int8_configs honor VLLM_TUNED_CONFIG_FOLDER (env override),
fall back to the packaged configs dir, and return None when nothing matches.

The expected JSON filename is reproduced from the loader's exact f-string using
the *real* device name of the running host, so the tests match whatever GPU the
suite happens to run on.
"""

import json

import pytest

from vllm.model_executor.layers.quantization.utils import (
    fp8_utils,
    int8_utils,
)
from vllm.platforms import current_platform

ENV = "VLLM_TUNED_CONFIG_FOLDER"

# Shape args shared by the filename builders below. Values are arbitrary; the
# tests only require that the filename the loader builds matches the file we
# write under the sentinel.
N, K, BLOCK_N, BLOCK_K = 1536, 7168, 128, 128

SENTINEL = {1: {"BLOCK_SIZE_M": 64, "marker": "sentinel"}}


def _device_name() -> str:
    return current_platform.get_device_name().replace(" ", "_")


def _fp8_file_name() -> str:
    dev = _device_name()
    return (
        f"N={N},K={K},device_name={dev},dtype=fp8_w8a8,"
        f"block_shape=[{BLOCK_N},{BLOCK_K}].json"
    )


def _int8_file_name() -> str:
    # NOTE: no space after the comma in block_shape -- matches the 39 shipped
    # int8 config files. The pre-fix loader built "[128, 128]" (with a space),
    # which never matched any file. test_int8_packaged_path_loads is the proof.
    dev = _device_name()
    return (
        f"N={N},K={K},device_name={dev},dtype=int8_w8a8,"
        f"block_shape=[{BLOCK_N},{BLOCK_K}].json"
    )


def _write(dir_path, file_name, payload):
    (dir_path / file_name).write_text(json.dumps(payload))


@pytest.fixture(autouse=True)
def _clear_caches():
    # Both loaders are @functools.lru_cache-decorated; clear before and after
    # each test so env/_CONFIGS_DIR monkeypatches actually take effect.
    fp8_utils.get_w8a8_block_fp8_configs.cache_clear()
    int8_utils.get_w8a8_block_int8_configs.cache_clear()
    yield
    fp8_utils.get_w8a8_block_fp8_configs.cache_clear()
    int8_utils.get_w8a8_block_int8_configs.cache_clear()


# =============================== FP8 ======================================


def test_fp8_packaged_path_loads(monkeypatch, tmp_path):
    """NO-BREAK: with env unset, a config in _CONFIGS_DIR loads unchanged --
    pins the pre-change packaged read behavior."""
    monkeypatch.delenv(ENV, raising=False)
    _write(tmp_path, _fp8_file_name(), {"1": SENTINEL[1]})
    monkeypatch.setattr(fp8_utils, "_CONFIGS_DIR", str(tmp_path))
    assert fp8_utils.get_w8a8_block_fp8_configs(N, K, BLOCK_N, BLOCK_K) == (SENTINEL)


def test_fp8_env_override(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    pkg = tmp_path / "pkg"
    env_dir.mkdir()
    pkg.mkdir()
    _write(env_dir, _fp8_file_name(), {"1": {"src": "env"}})
    _write(pkg, _fp8_file_name(), {"1": {"src": "pkg"}})
    monkeypatch.setenv(ENV, str(env_dir))
    monkeypatch.setattr(fp8_utils, "_CONFIGS_DIR", str(pkg))
    assert fp8_utils.get_w8a8_block_fp8_configs(N, K, BLOCK_N, BLOCK_K) == {
        1: {"src": "env"}
    }


def test_fp8_env_miss_falls_back_to_packaged(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    pkg = tmp_path / "pkg"
    env_dir.mkdir()
    pkg.mkdir()
    _write(pkg, _fp8_file_name(), {"1": SENTINEL[1]})
    monkeypatch.setenv(ENV, str(env_dir))
    monkeypatch.setattr(fp8_utils, "_CONFIGS_DIR", str(pkg))
    assert fp8_utils.get_w8a8_block_fp8_configs(N, K, BLOCK_N, BLOCK_K) == (SENTINEL)


def test_fp8_both_miss_returns_none(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    pkg = tmp_path / "pkg"
    env_dir.mkdir()
    pkg.mkdir()
    monkeypatch.setenv(ENV, str(env_dir))
    monkeypatch.setattr(fp8_utils, "_CONFIGS_DIR", str(pkg))
    assert fp8_utils.get_w8a8_block_fp8_configs(N, K, BLOCK_N, BLOCK_K) is None


def test_fp8_non_dict_returns_none(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV, raising=False)
    (tmp_path / _fp8_file_name()).write_text(json.dumps([1, 2, 3]))
    monkeypatch.setattr(fp8_utils, "_CONFIGS_DIR", str(tmp_path))
    assert fp8_utils.get_w8a8_block_fp8_configs(N, K, BLOCK_N, BLOCK_K) is None


# =============================== INT8 =====================================


def test_int8_packaged_path_loads(monkeypatch, tmp_path):
    """PROVES THE BUGFIX: the sentinel is written under the UNSPACED filename
    (block_shape=[128,128]). The pre-fix loader built "[128, 128]" with a stray
    space and never matched -- so this test is red before the int8_utils fix and
    green after. It is also the no-break proof for the packaged read."""
    monkeypatch.delenv(ENV, raising=False)
    _write(tmp_path, _int8_file_name(), {"1": SENTINEL[1]})
    monkeypatch.setattr(int8_utils, "_CONFIGS_DIR", str(tmp_path))
    assert int8_utils.get_w8a8_block_int8_configs(N, K, BLOCK_N, BLOCK_K) == SENTINEL


def test_int8_env_override(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    pkg = tmp_path / "pkg"
    env_dir.mkdir()
    pkg.mkdir()
    _write(env_dir, _int8_file_name(), {"1": {"src": "env"}})
    _write(pkg, _int8_file_name(), {"1": {"src": "pkg"}})
    monkeypatch.setenv(ENV, str(env_dir))
    monkeypatch.setattr(int8_utils, "_CONFIGS_DIR", str(pkg))
    assert int8_utils.get_w8a8_block_int8_configs(N, K, BLOCK_N, BLOCK_K) == {
        1: {"src": "env"}
    }


def test_int8_env_miss_falls_back_to_packaged(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    pkg = tmp_path / "pkg"
    env_dir.mkdir()
    pkg.mkdir()
    _write(pkg, _int8_file_name(), {"1": SENTINEL[1]})
    monkeypatch.setenv(ENV, str(env_dir))
    monkeypatch.setattr(int8_utils, "_CONFIGS_DIR", str(pkg))
    assert int8_utils.get_w8a8_block_int8_configs(N, K, BLOCK_N, BLOCK_K) == SENTINEL


def test_int8_both_miss_returns_none(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    pkg = tmp_path / "pkg"
    env_dir.mkdir()
    pkg.mkdir()
    monkeypatch.setenv(ENV, str(env_dir))
    monkeypatch.setattr(int8_utils, "_CONFIGS_DIR", str(pkg))
    assert int8_utils.get_w8a8_block_int8_configs(N, K, BLOCK_N, BLOCK_K) is None


def test_int8_non_dict_returns_none(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV, raising=False)
    (tmp_path / _int8_file_name()).write_text(json.dumps([1, 2, 3]))
    monkeypatch.setattr(int8_utils, "_CONFIGS_DIR", str(tmp_path))
    assert int8_utils.get_w8a8_block_int8_configs(N, K, BLOCK_N, BLOCK_K) is None
