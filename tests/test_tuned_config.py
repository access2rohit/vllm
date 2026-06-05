# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the shared tuned-config helper (vllm/utils/tuned_config.py).

Pure file IO + string joins; no GPU, no torch kernels required.
"""

import json

import pytest

from vllm.utils.tuned_config import (
    load_tuned_config,
    resolve_tuned_config_path,
    tuned_config_search_paths,
)

ENV = "VLLM_TUNED_CONFIG_FOLDER"


# ----------------------------- search paths --------------------------------


def test_search_paths_env_unset_returns_packaged_only(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert tuned_config_search_paths("f.json", "/pkg") == ["/pkg/f.json"]


def test_search_paths_env_set_returns_env_then_packaged(monkeypatch):
    monkeypatch.setenv(ENV, "/env")
    assert tuned_config_search_paths("f.json", "/pkg") == [
        "/env/f.json",
        "/pkg/f.json",
    ]


def test_search_paths_no_packaged_dir_returns_env_only(monkeypatch):
    monkeypatch.setenv(ENV, "/env")
    assert tuned_config_search_paths("f.json", None) == ["/env/f.json"]


def test_search_paths_env_unset_and_no_packaged_returns_empty(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert tuned_config_search_paths("f.json", None) == []


# ----------------------------- resolve -------------------------------------


def test_resolve_returns_first_existing(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV, raising=False)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "f.json").write_text("{}")
    assert resolve_tuned_config_path("f.json", str(pkg)) == str(
        pkg / "f.json"
    )


def test_resolve_env_shadows_packaged(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    pkg = tmp_path / "pkg"
    env_dir.mkdir()
    pkg.mkdir()
    (env_dir / "f.json").write_text("{}")
    (pkg / "f.json").write_text("{}")
    monkeypatch.setenv(ENV, str(env_dir))
    assert resolve_tuned_config_path("f.json", str(pkg)) == str(
        env_dir / "f.json"
    )


def test_resolve_env_miss_falls_back_to_packaged(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    pkg = tmp_path / "pkg"
    env_dir.mkdir()
    pkg.mkdir()
    (pkg / "f.json").write_text("{}")
    monkeypatch.setenv(ENV, str(env_dir))
    assert resolve_tuned_config_path("f.json", str(pkg)) == str(
        pkg / "f.json"
    )


def test_resolve_none_when_nothing_exists(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV, raising=False)
    assert resolve_tuned_config_path("missing.json", str(tmp_path)) is None


# ----------------------------- load ----------------------------------------


def test_load_dict_returns_int_keyed_config(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV, raising=False)
    (tmp_path / "f.json").write_text(
        json.dumps({"1": {"BLOCK": 64}, "8": {"BLOCK": 128}})
    )
    out = load_tuned_config("f.json", str(tmp_path))
    assert out == {1: {"BLOCK": 64}, 8: {"BLOCK": 128}}
    assert all(isinstance(k, int) for k in out)


def test_load_pops_triton_version_and_filters_non_digit(
    monkeypatch, tmp_path
):
    monkeypatch.delenv(ENV, raising=False)
    (tmp_path / "f.json").write_text(
        json.dumps(
            {
                "1": {"BLOCK": 64},
                "triton_version": "3.5.1",
                "not_a_number": {"BLOCK": 0},
            }
        )
    )
    out = load_tuned_config("f.json", str(tmp_path))
    assert out == {1: {"BLOCK": 64}}


def test_load_list_returns_none(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV, raising=False)
    (tmp_path / "f.json").write_text(json.dumps([1, 2, 3]))
    assert load_tuned_config("f.json", str(tmp_path)) is None


def test_load_non_dict_scalar_returns_none(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV, raising=False)
    (tmp_path / "f.json").write_text(json.dumps(42))
    assert load_tuned_config("f.json", str(tmp_path)) is None


def test_load_missing_returns_none(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV, raising=False)
    assert load_tuned_config("missing.json", str(tmp_path)) is None


def test_load_malformed_json_raises(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV, raising=False)
    (tmp_path / "f.json").write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        load_tuned_config("f.json", str(tmp_path))


def test_load_env_override_wins(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    pkg = tmp_path / "pkg"
    env_dir.mkdir()
    pkg.mkdir()
    (env_dir / "f.json").write_text(json.dumps({"1": {"src": "env"}}))
    (pkg / "f.json").write_text(json.dumps({"1": {"src": "pkg"}}))
    monkeypatch.setenv(ENV, str(env_dir))
    assert load_tuned_config("f.json", str(pkg)) == {1: {"src": "env"}}
