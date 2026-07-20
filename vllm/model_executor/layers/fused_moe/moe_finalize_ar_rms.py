# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OP-301: fused MoE-finalize + MNNVL oneshot AllReduce + residual + RMSNorm.

This module carries:

1. The JIT build of the CUDA extension in ``csrc/moe_finalize_ar_rms/`` (a
   MoE-finalize prologue variant of FlashInfer's MNNVL oneshot Lamport
   allreduce kernel, fp32 accumulation end to end).
2. A capture side-channel that lets the opaque
   ``vllm::moe_forward_shared_no_finalize`` custom op receive the unfinalized
   MoE tensors (``expert_weights``, ``expanded_idx_to_permuted_idx``) from the
   TRT-LLM NVFP4 monolithic experts kernel, which is invoked several call
   frames below the custom op boundary.
3. The two opaque custom ops used by the
   ``MoeFinalizeARRMSFusionPass`` graph rewrite
   (vllm/compilation/passes/fusion/moe_finalize_ar_rms_fusion.py):

   - ``vllm::moe_forward_shared_no_finalize``: same signature as
     ``vllm::moe_forward_shared`` but returns
     ``(shared_output, gemm2_output, expert_weights,
     expanded_idx_to_permuted_idx)`` with the MoE finalize skipped
     (``do_finalize=False`` on the FlashInfer TRT-LLM fp4 block-scale MoE).
   - ``vllm::moe_finalize_ar_rms_fused``: consumes those tensors plus the
     residual/gamma and performs finalize + routed scaling + shared-expert
     add + TP allreduce + residual + RMSNorm in one kernel, returning
     ``(norm_out, residual_out)``. ``mutates_args=[]`` — all outputs are
     freshly allocated (Lamport workspace mutation is internal state, as
     with the existing flashinfer allreduce fusion ops).

Gating: the graph rewrite is enabled by ``VLLM_MOE_FINALIZE_AR_RMS_FUSION``
and restricted via ``is_applicable_for_range`` to compile ranges with
``end <= 16`` (the oneshot regime). Non-rewritten graphs never call these
ops, and the default ``do_finalize=True`` path is untouched.
"""

import contextlib
import functools
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# Oneshot regime bound; must match MoeFinalizeARRMSFusionPass gating.
MOE_FINALIZE_AR_RMS_MAX_TOKENS = 16


# ---------------------------------------------------------------------------
# JIT extension
# ---------------------------------------------------------------------------


def _csrc_dir() -> Path:
    # vllm/model_executor/layers/fused_moe/ -> repo root / csrc
    return Path(__file__).resolve().parents[4] / "csrc" / "moe_finalize_ar_rms"


@functools.cache
def _load_extension():
    """Build and load the moe_finalize_ar_rms torch extension via JIT.

    Compiles against the FlashInfer header tree shipped inside the installed
    flashinfer package (the kernel reuses the MNNVL Lamport utilities from
    ``flashinfer/comm/trtllm_mnnvl_allreduce.cuh``).
    """
    import flashinfer.jit.env as fi_env
    from torch.utils.cpp_extension import load

    csrc = _csrc_dir()
    fi_include = Path(fi_env.FLASHINFER_INCLUDE_DIR)
    spdlog_include = fi_include.parent / "spdlog" / "include"

    module = load(
        name="vllm_moe_finalize_ar_rms",
        sources=[str(csrc / "moe_finalize_ar_rms_binding.cu")],
        extra_include_paths=[
            str(csrc),
            str(fi_include),
            str(spdlog_include),
        ],
        extra_cuda_cflags=[
            "-O3",
            "--std=c++17",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_100a,code=sm_100a",
            # torch.utils.cpp_extension injects -D__CUDA_NO_HALF_*__ by
            # default; the FlashInfer headers require implicit half/bf16
            # conversions, so undo them (same as FlashInfer's own JIT flags).
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        verbose=False,
    )
    logger.info_once("Loaded moe_finalize_ar_rms JIT extension")
    return module


# ---------------------------------------------------------------------------
# Capture side-channel (opaque-op boundary -> deep experts kernel)
# ---------------------------------------------------------------------------


@dataclass
class _NoFinalizeCapture:
    active: bool = False
    expert_weights: torch.Tensor | None = None
    expanded_idx_to_permuted_idx: torch.Tensor | None = None
    filled: bool = field(default=False)


_capture = _NoFinalizeCapture()


def capture_active() -> bool:
    """True while vllm::moe_forward_shared_no_finalize is executing.

    Checked by TrtLlmNvFp4ExpertsMonolithic.apply() to select
    ``do_finalize=False`` and stash the routing tensors.
    """
    return _capture.active


def stash_unfinalized(
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
) -> None:
    """Called by the experts kernel while a no-finalize capture is active."""
    assert _capture.active, (
        "stash_unfinalized called outside a no-finalize capture window"
    )
    _capture.expert_weights = expert_weights
    _capture.expanded_idx_to_permuted_idx = expanded_idx_to_permuted_idx
    _capture.filled = True


@contextlib.contextmanager
def _no_finalize_window():
    _capture.active = True
    _capture.expert_weights = None
    _capture.expanded_idx_to_permuted_idx = None
    _capture.filled = False
    try:
        yield _capture
    finally:
        _capture.active = False
        _capture.expert_weights = None
        _capture.expanded_idx_to_permuted_idx = None
        _capture.filled = False


# ---------------------------------------------------------------------------
# Custom op 1: moe_forward_shared_no_finalize
# ---------------------------------------------------------------------------

# Same layer_name typing dance as vllm/.../runner/moe_runner.py: on
# torch >= 2.11 layer_name is a hoisted LayerName opaque object; on older
# versions a plain str. Importing moe_runner here is safe (it does not import
# this module).
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (  # noqa: E402
    _layer_name_type,
)


def _moe_forward_shared_no_finalize(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
        _resolve_layer_name,
        get_layer_from_name,
    )

    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    debug_compare = bool(os.environ.get("VLLM_MOE_FINALIZE_AR_RMS_DEBUG_COMPARE"))
    if debug_compare:
        dbg_hidden = hidden_states.clone()
        dbg_router = router_logits.clone()
        dbg_shared_in = (
            shared_experts_input.clone() if shared_experts_input is not None else None
        )
    with _no_finalize_window() as cap:
        result = layer._forward_impl(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
        )
        assert isinstance(result, tuple), (
            "moe_forward_shared_no_finalize requires a shared-experts MoE layer"
        )
        shared_output, gemm2_output = result
        assert cap.filled, (
            "The experts kernel did not stash unfinalized MoE tensors. "
            "moe_forward_shared_no_finalize is only valid when the layer "
            "uses the TRT-LLM NVFP4 monolithic experts kernel with "
            "no-finalize support."
        )
        expert_weights = cap.expert_weights
        expanded_idx = cap.expanded_idx_to_permuted_idx
    assert expert_weights is not None and expanded_idx is not None

    if debug_compare:
        # In-situ contract check: finalize_reference(no-finalize outputs)
        # must match the production do_finalize=True path on identical inputs.
        # IMPORTANT: compute my_fused BEFORE the reference call — the returned
        # tensors are dlpack views into flashinfer's persistent workspace and
        # a second op invocation clobbers them.
        my_fused = _finalize_reference(
            gemm2_output,
            expert_weights,
            expanded_idx,
            torch.zeros_like(shared_output),
            1.0,
        ).clone()
        # Snapshot the dlpack-backed outputs BEFORE the reference invocation
        # (it reuses flashinfer's workspace and clobbers them).
        dbg_gemm2 = gemm2_output.cpu().clone()
        dbg_ew = expert_weights.cpu().clone()
        dbg_idx = expanded_idx.cpu().clone()
        ref_result = layer._forward_impl(
            dbg_hidden, dbg_router, dbg_shared_in, input_ids
        )
        ref_shared, ref_fused = ref_result
        fused_rel = (
            (my_fused.float() - ref_fused.float()).norm()
            / ref_fused.float().norm().clamp_min(1e-6)
        ).item()
        per_tok = (
            (my_fused.float() - ref_fused.float()).norm(dim=-1)
            / ref_fused.float().norm(dim=-1).clamp_min(1e-6)
        )
        shared_rel = (
            (shared_output.float() - ref_shared.float()).norm()
            / ref_shared.float().norm().clamp_min(1e-6)
        ).item()
        worst = per_tok.max().item()
        if fused_rel > 5e-3 or shared_rel > 5e-3 or worst > 5e-2:
            dump_dir = os.environ.get("VLLM_MOE_FINALIZE_AR_RMS_DEBUG_DUMP")
            if dump_dir:
                import torch.distributed as dist

                rank = dist.get_rank() if dist.is_initialized() else 0
                path = Path(dump_dir) / f"mismatch_rank{rank}.pt"
                if not path.exists():
                    Path(dump_dir).mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "layer": str(_resolve_layer_name(layer_name)),
                            "gemm2": dbg_gemm2,
                            "ew": dbg_ew,
                            "idx": dbg_idx,
                            "ref_fused": ref_fused.cpu(),
                            "my_fused": my_fused.cpu(),
                            "shared": shared_output.cpu(),
                            "hidden_in": dbg_hidden.cpu(),
                            "router_in": dbg_router.cpu(),
                        },
                        path,
                    )
            logger.warning(
                "[moe_finalize debug_compare] MISMATCH layer=%s ntok=%d "
                "fused_rel=%.5f worst_tok_rel=%.5f shared_rel=%.5f "
                "per_tok=%s gemm2=%s ew=%s idx_head=%s",
                _resolve_layer_name(layer_name),
                shared_output.shape[0],
                fused_rel,
                worst,
                shared_rel,
                [round(x, 4) for x in per_tok.tolist()[:32]],
                tuple(gemm2_output.shape),
                tuple(expert_weights.shape),
                expanded_idx.flatten()[:16].tolist(),
            )
        else:
            logger.debug(
                "[moe_finalize debug_compare] ok layer=%s ntok=%d "
                "fused_rel=%.5f",
                _resolve_layer_name(layer_name),
                shared_output.shape[0],
                fused_rel,
            )

    return shared_output, gemm2_output, expert_weights, expanded_idx


def _moe_forward_shared_no_finalize_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # The gemm2_output row count and index/scale sizes are data-dependent
    # (TRT-LLM-gen pads the permuted dimension to its tile size), so expose
    # them as unbacked symints. They are consumed only by the opaque
    # vllm::moe_finalize_ar_rms_fused op, whose fake impl does not inspect
    # these shapes.
    ctx = torch.library.get_ctx()
    hidden = hidden_states.shape[-1]
    num_permuted = ctx.new_dynamic_size()
    num_expanded = ctx.new_dynamic_size()
    if shared_experts_input is not None:
        shared_out = torch.empty_like(shared_experts_input)
    else:
        shared_out = torch.empty_like(hidden_states)
    gemm2_output = hidden_states.new_empty((num_permuted, hidden))
    expert_weights = hidden_states.new_empty(
        (hidden_states.shape[0], num_expanded), dtype=torch.bfloat16
    )
    expanded_idx = hidden_states.new_empty(
        (hidden_states.shape[0] * num_expanded,), dtype=torch.int32
    )
    return shared_out, gemm2_output, expert_weights, expanded_idx


direct_register_custom_op(
    op_name="moe_forward_shared_no_finalize",
    op_func=_moe_forward_shared_no_finalize,
    fake_impl=_moe_forward_shared_no_finalize_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


# ---------------------------------------------------------------------------
# Custom op 2: moe_finalize_ar_rms_fused
# ---------------------------------------------------------------------------


def _finalize_reference(
    gemm2_output: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    shared_expert_output: torch.Tensor,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """Pure-torch MoE finalize (fp32 accum), used by the eager fallback."""
    num_tokens, hidden = shared_expert_output.shape
    idx = expanded_idx_to_permuted_idx.view(num_tokens, -1).to(torch.long)
    top_k = idx.shape[1]
    valid = idx >= 0
    safe_idx = torch.where(valid, idx, torch.zeros_like(idx))
    rows = gemm2_output.float()[safe_idx.view(-1)].view(num_tokens, top_k, hidden)
    scales = expert_weights.float().view(num_tokens, top_k, 1)
    scales = torch.where(valid.unsqueeze(-1), scales, torch.zeros_like(scales))
    acc = (rows * scales).sum(dim=1) * routed_scaling_factor
    acc = acc + shared_expert_output.float()
    return acc.to(shared_expert_output.dtype)


def _moe_finalize_ar_rms_fused(
    gemm2_output: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    shared_expert_output: torch.Tensor,
    residual: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    routed_scaling_factor: float,
    world_size: int,
    max_token_num: int,
    launch_with_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        get_fi_ar_workspace,
    )
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

    num_tokens, hidden = shared_expert_output.shape
    norm_out = torch.empty_like(shared_expert_output)
    residual_out = torch.empty_like(residual)

    workspace = get_fi_ar_workspace(
        world_size=world_size,
        rank=get_tensor_model_parallel_rank(),
        max_token_num=max_token_num,
        hidden_dim=hidden,
        dtype=shared_expert_output.dtype,
        group=get_tp_group().device_group,
    )

    use_kernel = (
        workspace is not None
        and getattr(workspace, "backend", None) == "mnnvl"
        and shared_expert_output.dtype == torch.bfloat16
        and num_tokens <= MOE_FINALIZE_AR_RMS_MAX_TOKENS
        and world_size in (2, 4, 8)
        # Debug escape hatch: force the unfused reference chain to isolate
        # numeric issues to the CUDA kernel vs the graph rewrite.
        and not os.environ.get("VLLM_MOE_FINALIZE_AR_RMS_FORCE_FALLBACK")
    )

    if use_kernel:
        module = _load_extension()
        module.moe_finalize_ar_rms_fused(
            gemm2_output.contiguous(),
            expert_weights.contiguous(),
            expanded_idx_to_permuted_idx.contiguous(),
            shared_expert_output.contiguous(),
            float(routed_scaling_factor),
            residual.contiguous(),
            rms_gamma.contiguous(),
            float(rms_eps),
            workspace.mc_ptr,
            workspace.uc_ptrs_dev,
            workspace.buffer_flags,
            world_size,
            workspace.rank,
            launch_with_pdl,
            norm_out,
            residual_out,
        )
        return norm_out, residual_out

    # Fallback (no mnnvl workspace / unsupported config): unfused chain with
    # identical semantics. Keeps the rewritten graph correct everywhere the
    # fused kernel cannot run.
    logger.warning_once(
        "moe_finalize_ar_rms_fused falling back to the unfused chain "
        "(mnnvl workspace unavailable or unsupported config)."
    )
    combined = _finalize_reference(
        gemm2_output,
        expert_weights,
        expanded_idx_to_permuted_idx,
        shared_expert_output,
        routed_scaling_factor,
    )
    reduced = tensor_model_parallel_all_reduce(combined)
    pre_norm = (reduced.float() + residual.float()).to(residual.dtype)
    residual_out.copy_(pre_norm)
    x = pre_norm.float()
    rrms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + rms_eps)
    norm_out.copy_(((x * rrms) * rms_gamma.float()).to(norm_out.dtype))
    return norm_out, residual_out


def _moe_finalize_ar_rms_fused_fake(
    gemm2_output: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    shared_expert_output: torch.Tensor,
    residual: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    routed_scaling_factor: float,
    world_size: int,
    max_token_num: int,
    launch_with_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(shared_expert_output), torch.empty_like(residual)


direct_register_custom_op(
    op_name="moe_finalize_ar_rms_fused",
    op_func=_moe_finalize_ar_rms_fused,
    mutates_args=[],
    fake_impl=_moe_finalize_ar_rms_fused_fake,
)


def moe_finalize_ar_rms_fusion_enabled() -> bool:
    return bool(envs.VLLM_MOE_FINALIZE_AR_RMS_FUSION)
