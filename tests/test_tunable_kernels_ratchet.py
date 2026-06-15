# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B3 enforcement: completeness ratchet + reasoned allowlist.

A newly added ``@triton.jit`` kernel must satisfy ONE of:
  (a) be backed by a registered tunable loader (TUNABLE_KERNEL_IMPLS), OR
  (b) appear in tools/tuned_config/exempt_kernels.py with a one-line reason.

Pre-existing kernels were seeded into the allowlist once (the ratchet
baseline), so the check is green today and only bites NEW kernels -- no mass
migration. The allowlist-with-reason is the de-coercion mechanism: CI never
says "you must tune", only "tune OR justify". This file also runs the B2
round-trip correctness check on the registered kernels.
"""

import json

import pytest

from tools.tuned_config.discover_kernels import discover_triton_kernels
from tools.tuned_config.exempt_kernels import EXEMPT_KERNELS, TUNABLE_KERNEL_IMPLS
from vllm.utils.tuned_config import load_tuned_config
from vllm.utils.tuned_config_registry import get_tunable_kernels
from vllm.utils.tuned_config_registry_seed import seed_registry

ENV = "VLLM_TUNED_CONFIG_FOLDER"

SAMPLE_ARGS = {
    "fused_moe": ((8, 14336, "fp8_w8a8"), {"block_shape": [128, 128]}),
    "w8a8_block_fp8": ((1536, 7168, 128, 128), {}),
    "w8a8_block_int8": ((1536, 7168, 128, 128), {}),
    "selective_state_update": ((64, 128, "float16"), {}),
}


@pytest.fixture(autouse=True, scope="module")
def _seed():
    seed_registry()


# ----------------------------- the ratchet ---------------------------------


def test_ratchet_complete():
    discovered = discover_triton_kernels("vllm")
    covered = set(TUNABLE_KERNEL_IMPLS.values()) | set(EXEMPT_KERNELS)
    missing = discovered - covered
    assert missing == set(), (
        f"New @triton.jit kernel(s) {sorted(missing)} must either register a "
        f"tuned-config loader (see vllm/utils/tuned_config_registry.py) OR be "
        f"added to tools/tuned_config/exempt_kernels.py with a reason "
        f"(e.g. 'memory-bound elementwise; tuning <2%')."
    )


def test_no_exempt_kernel_is_also_tunable():
    # A kernel cannot be both tunable and exempt.
    overlap = set(TUNABLE_KERNEL_IMPLS.values()) & set(EXEMPT_KERNELS)
    assert overlap == set(), f"kernels both tunable and exempt: {sorted(overlap)}"


def test_no_empty_exemption_reasons():
    bad = []
    for kid, reason in EXEMPT_KERNELS.items():
        r = (reason or "").strip()
        if not r or r.upper() == "TODO" or r.upper().startswith("TODO"):
            bad.append(kid)
    assert bad == [], (
        f"Exemption reasons must be substantive (no empty / 'TODO' "
        f"rubber-stamps): {bad}"
    )


def test_allowlist_baseline_matches_discovery():
    # The seeded baseline must exactly partition discovered kernels into
    # tunable-backing vs exempt, with no leftovers -- proving the ratchet
    # starts green and only future kernels can break it.
    discovered = discover_triton_kernels("vllm")
    impls = set(TUNABLE_KERNEL_IMPLS.values())
    assert impls <= discovered
    assert set(EXEMPT_KERNELS) <= discovered
    assert discovered == impls | set(EXEMPT_KERNELS)


# ----------------------------- B2 round-trip on registered kernels ----------


def test_registered_kernels_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path))
    specs = {s.name: s for s in get_tunable_kernels() if s.name in SAMPLE_ARGS}
    assert set(specs) == set(SAMPLE_ARGS)

    for name, spec in specs.items():
        args, kwargs = SAMPLE_ARGS[name]
        file_name = spec.key_fn(*args, **kwargs)
        (tmp_path / file_name).write_text(
            json.dumps({"1": {"BLOCK_SIZE_M": 64, "kernel": name}})
        )
        loaded = load_tuned_config(file_name, "/nonexistent/packaged")
        assert loaded == {1: {"BLOCK_SIZE_M": 64, "kernel": name}}, (
            f"{name}: key_fn produced {file_name!r} but loader could not find "
            f"it -- filename drift (cf. the INT8 stray-space bug)."
        )
