# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared resolution + loading for offline-tuned Triton kernel config files.

All tuned-config loaders (fused MoE, block FP8/INT8 GEMM, Mamba SSU, ...) follow
the same contract: a JSON file mapping batch size -> kernel config, found either
in the user-supplied ``VLLM_TUNED_CONFIG_FOLDER`` (highest priority, enables
hot-loading new configs into an existing container via env var + restart) or in
the configs shipped with vLLM. This module is the single place that precedence
lives so every loader -- current and future -- behaves identically.
"""

import json
import os
from typing import Any

import vllm.envs as envs


def tuned_config_search_paths(file_name: str, packaged_dir: str | None) -> list[str]:
    """Ordered candidate paths for ``file_name``: the user override folder
    (``VLLM_TUNED_CONFIG_FOLDER``) first if set, then ``packaged_dir`` if not
    None. Pure string joins -- no filesystem access."""
    paths: list[str] = []
    user_folder = envs.VLLM_TUNED_CONFIG_FOLDER
    if user_folder is not None:
        paths.append(os.path.join(user_folder, file_name))
    if packaged_dir is not None:
        paths.append(os.path.join(packaged_dir, file_name))
    return paths


def resolve_tuned_config_path(file_name: str, packaged_dir: str | None) -> str | None:
    """First search path that exists on disk, else None."""
    for path in tuned_config_search_paths(file_name, packaged_dir):
        if os.path.exists(path):
            return path
    return None


def load_tuned_config(
    file_name: str, packaged_dir: str | None
) -> dict[int, Any] | None:
    """Resolve, open, and normalize a tuned-config file to ``{int_batch: config}``.

    Returns None if no file matched or the JSON is not a dict. Drops the
    ``triton_version`` metadata key and keeps only integer-string keys. Does NOT
    log -- callers emit their own kernel-specific info/warning messages. A
    malformed JSON file raises (same as the existing inline loaders)."""
    path = resolve_tuned_config_path(file_name, packaged_dir)
    if path is None:
        return None
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return None
    raw.pop("triton_version", None)
    return {int(k): v for k, v in raw.items() if k.isdigit()}
