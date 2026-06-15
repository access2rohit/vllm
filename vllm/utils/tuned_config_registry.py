# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Central registry of tunable Triton kernels (RFC mechanism option A3).

This module is the single, importable source of truth for "which Triton
kernels in vLLM expose an offline-tuned-config path, and how." Each tunable
kernel registers a :class:`TunableKernelSpec` describing its config-filename
builder (``key_fn``), the packaged config directory, an ``applicability`` tag,
and (optionally) the tuner script that generates its configs.

The registry is populated at import time -- either via the
:func:`tunable_triton_kernel` decorator placed on a kernel's "configs getter",
or via an explicit :func:`register_tunable_kernel` call. To get a complete
manifest, an importer must first import every module that registers a kernel
(in production this happens at package init; in tests we import the seed module
``vllm.utils.tuned_config_registry_seed`` explicitly). This mirrors the
existing model/attention-backend registration pattern in vLLM.

The registry is purely additive: wrapping a getter with the decorator does not
change the configs it returns at runtime; it only records metadata and routes
the filename-build + load + fallback through the shared
``vllm/utils/tuned_config.py`` helper.
"""

import enum
import functools
from collections.abc import Callable
from dataclasses import dataclass

from vllm.utils.tuned_config import load_tuned_config


class Applicability(enum.Enum):
    """How relevant offline tuning is for a kernel.

    - ``COMPUTE_HEAVY_INPATH``: GEMM-like, on the per-token hot path; tuning
      yields real speedups (MoE expert GEMMs, block-quant GEMMs, SSM update).
    - ``OPTIONAL``: tuning may help in some shapes but is not load-bearing.
    - ``EXPERIMENTAL``: a tuning path exists but is not yet validated/shipped.
    """

    COMPUTE_HEAVY_INPATH = "compute_heavy_inpath"
    OPTIONAL = "optional"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True)
class TunableKernelSpec:
    """Immutable metadata describing one tunable Triton kernel.

    ``key_fn`` maps the kernel's shape/dtype arguments to the JSON config
    filename that :func:`load_tuned_config` searches for. ``packaged_dir`` is
    the directory shipped with vLLM that holds those configs. ``tuner_script``
    points at the benchmark script that regenerates the configs (or None).
    """

    name: str
    applicability: Applicability
    key_fn: Callable[..., str]
    packaged_dir: str
    tuner_script: str | None = None


# The single source of truth, keyed by kernel name. Populated at import time.
TUNABLE_KERNEL_REGISTRY: dict[str, TunableKernelSpec] = {}


def register_tunable_kernel(spec: TunableKernelSpec) -> None:
    """Insert ``spec`` into the registry. Raise on a duplicate name so two
    kernels cannot silently shadow each other."""
    if spec.name in TUNABLE_KERNEL_REGISTRY:
        raise ValueError(
            f"Tunable kernel {spec.name!r} is already registered; kernel names "
            f"must be unique across the vLLM package."
        )
    TUNABLE_KERNEL_REGISTRY[spec.name] = spec


def get_tunable_kernels() -> list[TunableKernelSpec]:
    """Return all registered specs, sorted by name (stable for tests/manifests).

    Note: completeness depends on the relevant kernel modules (or the seed
    module) having been imported first.
    """
    return [TUNABLE_KERNEL_REGISTRY[name] for name in sorted(TUNABLE_KERNEL_REGISTRY)]


def get_tuned_config_for(name: str, M: int, **shape_kwargs) -> dict | None:
    """Resolve + select a single tuned kernel config by registered name.

    This is the GENERIC, name-based front-end that lets a kernel use the tuned
    path without writing its own getter ("minimal form"). It performs two
    stages and returns the chosen config dict (or None):

    - Stage 1 (resolve): build the config filename via the registered
      ``spec.key_fn(**shape_kwargs)`` and load it through
      :func:`load_tuned_config` against ``spec.packaged_dir`` (honoring the
      ``VLLM_TUNED_CONFIG_FOLDER`` override). This yields the full
      ``{batch_M: config}`` dict, or None if no file matched.
    - Stage 2 (select): pick the entry whose batch key is nearest to the
      runtime ``M``.

    Feeding the chosen knobs into ``kernel[grid]`` (stage 3) stays at the
    caller and is intentionally out of scope here.

    Raises ``KeyError`` if ``name`` is not a registered tunable kernel.
    """
    spec = TUNABLE_KERNEL_REGISTRY[name]
    configs = load_tuned_config(spec.key_fn(**shape_kwargs), spec.packaged_dir)
    if not configs:
        return None
    return configs[min(configs, key=lambda x: abs(x - M))]


def tunable_triton_kernel(
    name: str,
    key_fn: Callable[..., str],
    packaged_dir: str,
    applicability: Applicability = Applicability.COMPUTE_HEAVY_INPATH,
    tuner_script: str | None = None,
):
    """Decorator turning a "configs getter" into a tunable kernel front-end.

    The decorated function's body is *replaced*: calling the wrapper builds the
    filename via ``key_fn(*args, **kwargs)``, runs it through
    :func:`load_tuned_config` against ``packaged_dir`` (honoring the
    ``VLLM_TUNED_CONFIG_FOLDER`` override), and returns the loaded dict or None
    (the caller's heuristic-fallback signal). A :class:`TunableKernelSpec` is
    built, attached to the wrapper as ``__vllm_tunable__``, and registered in
    ``TUNABLE_KERNEL_REGISTRY`` at decoration (import) time.

    The wrapper is wrapped in ``functools.lru_cache`` so repeated lookups for
    the same shape are cheap (matching the existing getters, which were
    ``lru_cache``d). A ``.cache_clear`` passthrough is therefore available.
    """

    spec = TunableKernelSpec(
        name=name,
        applicability=applicability,
        key_fn=key_fn,
        packaged_dir=packaged_dir,
        tuner_script=tuner_script,
    )

    def decorator(func):
        @functools.lru_cache
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            file_name = key_fn(*args, **kwargs)
            return load_tuned_config(file_name, packaged_dir)

        wrapper.__vllm_tunable__ = spec
        register_tunable_kernel(spec)
        return wrapper

    return decorator
