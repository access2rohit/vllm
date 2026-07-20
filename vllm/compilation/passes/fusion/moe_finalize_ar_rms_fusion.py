# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OP-301: MoE-finalize -> add_mul -> AllReduce -> RMSNorm fusion pass.

Rewrites the per-MoE-layer region (see
kernel_opt_artifacts .../rounds/1/debate/round_2/op301_fx_region_dump.txt):

    moe_forward_shared = torch.ops.vllm.moe_forward_shared(...)
    shared = moe_forward_shared[0]
    fused  = moe_forward_shared[1]
    fused *= routed_scaling_factor            # 2.827 for Kimi-K2.6
    add    = shared + fused
    ar     = torch.ops.vllm.all_reduce(add, group_name=...)
    view   = ar.view(num_tokens, hidden)      # optional
    rms    = torch.ops.vllm_ir.fused_add_rms_norm[.maybe_inplace](
                 view, residual, weight, eps, ...)
    norm_out, residual_out = rms[0], rms[1]

into:

    nf = torch.ops.vllm.moe_forward_shared_no_finalize(...)
    shared, gemm2_output, expert_weights, expanded_idx = nf[0..3]
    fused_out = torch.ops.vllm.moe_finalize_ar_rms_fused(
        gemm2_output, expert_weights, expanded_idx, shared,
        residual, weight, eps, routed_scaling_factor,
        world_size, max_token_num, launch_with_pdl)
    norm_out, residual_out = fused_out[0], fused_out[1]

The fused op runs the MoE finalize (fp32 top-k combine), routed scaling,
shared-expert add, MNNVL oneshot Lamport TP-allreduce, residual add and
RMSNorm in a single kernel (csrc/moe_finalize_ar_rms/).

Because ``moe_forward_shared`` is an opaque custom op with a non-tensor
``layer_name`` argument, the torch pattern-matcher (``pm.register_replacement``)
cannot trace this region; the pass matches nodes manually instead.

Gating (all required):
- ``VLLM_MOE_FINALIZE_AR_RMS_FUSION=1``
- flashinfer.comm available and mnnvl allreduce workspace initialized
- ``is_applicable_for_range``: compile_range.end <= 16 (oneshot regime);
  the BS32 (twoshot) graph is never modified.
- no match -> graph unchanged.
"""

import operator
from typing import Any

import torch
import torch.fx as fx

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger

from ..vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)

# Oneshot regime bound (tokens). Must match
# vllm/model_executor/layers/fused_moe/moe_finalize_ar_rms.py.
MOE_FINALIZE_AR_RMS_MAX_TOKENS = 16

_MOE_FORWARD_SHARED_OPS: tuple[Any, ...] = ()
_MOE_FORWARD_SHARED_NO_FINALIZE_OP: Any = None
_MOE_FINALIZE_AR_RMS_FUSED_OP: Any = None


def _init_ops() -> bool:
    """Resolve op handles lazily (registration order safe)."""
    global _MOE_FORWARD_SHARED_OPS
    global _MOE_FORWARD_SHARED_NO_FINALIZE_OP
    global _MOE_FINALIZE_AR_RMS_FUSED_OP
    if _MOE_FINALIZE_AR_RMS_FUSED_OP is not None:
        return True
    try:
        # Importing registers the custom ops.
        import vllm.model_executor.layers.fused_moe.moe_finalize_ar_rms  # noqa: F401

        _MOE_FORWARD_SHARED_OPS = (
            torch.ops.vllm.moe_forward_shared.default,
            torch.ops.vllm.moe_forward_shared,
        )
        _MOE_FORWARD_SHARED_NO_FINALIZE_OP = (
            torch.ops.vllm.moe_forward_shared_no_finalize.default
        )
        _MOE_FINALIZE_AR_RMS_FUSED_OP = torch.ops.vllm.moe_finalize_ar_rms_fused.default
        return True
    except (ImportError, AttributeError):
        logger.warning_once(
            "moe_finalize_ar_rms fusion ops unavailable; pass disabled."
        )
        return False


_MUL_TARGETS = frozenset(
    {
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul_.Tensor,
        operator.mul,
        operator.imul,
    }
)
_ADD_TARGETS = frozenset(
    {
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add_.Tensor,
        operator.add,
        operator.iadd,
    }
)
_VIEW_TARGETS = frozenset(
    {
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
    }
)


def _is_getitem(node: fx.Node, idx: int | None = None) -> bool:
    if node.op != "call_function" or node.target is not operator.getitem:
        return False
    return idx is None or node.args[1] == idx


def _single_user(node: fx.Node) -> fx.Node | None:
    users = list(node.users)
    return users[0] if len(users) == 1 else None


def _rms_norm_targets() -> tuple[Any, ...]:
    targets: list[Any] = []
    op = getattr(torch.ops, "vllm_ir", None)
    if op is not None and hasattr(op, "fused_add_rms_norm"):
        packet = op.fused_add_rms_norm
        targets.append(packet.default)
        if hasattr(packet, "maybe_inplace"):
            targets.append(packet.maybe_inplace)
    return tuple(targets)


class _MatchedRegion:
    """Nodes of one matched finalize->add_mul->AR->RMSNorm chain."""

    def __init__(self) -> None:
        self.moe_node: fx.Node | None = None
        self.shared_getitem: fx.Node | None = None
        self.fused_getitem: fx.Node | None = None
        self.mul_node: fx.Node | None = None
        self.routed_scaling_factor: float = 1.0
        self.add_node: fx.Node | None = None
        self.all_reduce_node: fx.Node | None = None
        self.view_node: fx.Node | None = None
        self.rms_node: fx.Node | None = None
        self.residual_arg: fx.Node | None = None
        self.weight_arg: Any = None
        self.eps: float = 1e-5
        self.norm_getitem: fx.Node | None = None
        self.residual_getitem: fx.Node | None = None


def _match_region(moe_node: fx.Node, rms_targets: tuple[Any, ...]) -> _MatchedRegion | None:
    m = _MatchedRegion()
    m.moe_node = moe_node

    # getitem fan-out of the (shared, fused) tuple
    for user in moe_node.users:
        if _is_getitem(user, 0):
            m.shared_getitem = user
        elif _is_getitem(user, 1):
            m.fused_getitem = user
    if m.shared_getitem is None or m.fused_getitem is None:
        return None

    # fused *= routed_scaling_factor (scalar)
    mul = _single_user(m.fused_getitem)
    if mul is None or mul.target not in _MUL_TARGETS:
        return None
    scalar = None
    other = None
    for a in mul.args:
        if isinstance(a, (int, float)):
            scalar = float(a)
        elif isinstance(a, fx.Node) and a is not m.fused_getitem:
            other = a
    if scalar is None or other is not None:
        return None
    m.mul_node = mul
    m.routed_scaling_factor = scalar

    # shared + scaled_fused
    add = _single_user(mul)
    if add is None or add.target not in _ADD_TARGETS:
        return None
    add_operands = [a for a in add.args if isinstance(a, fx.Node)]
    if len(add_operands) != 2 or m.shared_getitem not in add_operands:
        return None
    # The shared getitem must have no other users than the add (the fused op
    # re-emits it internally; leaving another consumer would keep the old
    # moe_forward_shared alive).
    if any(u is not add for u in m.shared_getitem.users):
        return None
    m.add_node = add

    # tensor-parallel all-reduce
    ar = _single_user(add)
    if ar is None or ar.op != "call_function":
        return None
    ar_target_name = str(ar.target)
    if "all_reduce" not in ar_target_name or "vllm" not in ar_target_name:
        return None
    m.all_reduce_node = ar

    # optional view between AR and the RMSNorm input
    nxt = _single_user(ar)
    if nxt is not None and nxt.target in _VIEW_TARGETS:
        m.view_node = nxt
        nxt = _single_user(nxt)
    if nxt is None or nxt.target not in rms_targets:
        return None
    m.rms_node = nxt

    # fused_add_rms_norm(x, residual, weight, eps, ...)
    rms_args = list(m.rms_node.args)
    if len(rms_args) < 4:
        return None
    if not isinstance(rms_args[1], fx.Node):
        return None
    m.residual_arg = rms_args[1]
    m.weight_arg = rms_args[2]
    if not isinstance(rms_args[3], (int, float)):
        return None
    m.eps = float(rms_args[3])

    for user in m.rms_node.users:
        if _is_getitem(user, 0):
            m.norm_getitem = user
        elif _is_getitem(user, 1):
            m.residual_getitem = user
    if m.norm_getitem is None:
        return None
    return m


class MoeFinalizeARRMSFusionPass(VllmInductorPass):
    """Fuses MoE finalize + shared add + routed scale + TP AR + RMSNorm."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.disabled = True
        self.matched_count = 0

        if not envs.VLLM_MOE_FINALIZE_AR_RMS_FUSION:
            return
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size not in (2, 4, 8):
            logger.warning_once(
                "moe_finalize_ar_rms fusion disabled: tp_size=%d not in {2,4,8}",
                self.tp_size,
            )
            return
        if config.model_config is None:
            return
        if not _init_ops():
            return

        # Require the mnnvl flashinfer allreduce workspace (the fused kernel
        # reuses its Lamport buffers). Sized identically to AllReduceFusionPass
        # so both consumers share one workspace.
        try:
            from vllm.distributed.device_communicators.flashinfer_all_reduce import (
                get_fi_ar_workspace,
            )
        except ImportError:
            logger.warning_once(
                "flashinfer unavailable; moe_finalize_ar_rms fusion disabled."
            )
            return

        self.hidden_dim = config.model_config.get_hidden_size()
        max_size = config.compilation_config.pass_config.flashinfer_max_size(
            self.tp_size
        )
        if max_size is None:
            return
        element_size = torch.tensor([], dtype=self.model_dtype).element_size()
        self.max_token_num = max_size // (self.hidden_dim * element_size)
        self.max_token_num = min(
            self.max_token_num, config.scheduler_config.max_num_batched_tokens
        )

        workspace = get_fi_ar_workspace(
            world_size=self.tp_size,
            rank=get_tensor_model_parallel_rank(),
            max_token_num=self.max_token_num,
            hidden_dim=self.hidden_dim,
            dtype=self.model_dtype,
            group=get_tp_group().device_group,
        )
        if workspace is None or getattr(workspace, "backend", None) != "mnnvl":
            logger.warning_once(
                "moe_finalize_ar_rms fusion disabled: mnnvl allreduce "
                "workspace unavailable (backend=%s).",
                getattr(workspace, "backend", None),
            )
            return

        self.disabled = False

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        if self.disabled:
            return False
        # Oneshot regime only. The twoshot (BS32) graph must be byte-identical
        # to baseline.
        return bool(
            compile_range.end
            <= min(MOE_FINALIZE_AR_RMS_MAX_TOKENS, self.max_token_num)
        )

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        if self.disabled:
            return
        self.matched_count = self.rewrite(graph)
        logger.debug(
            "MoeFinalizeARRMSFusionPass replaced %s regions", self.matched_count
        )

    def rewrite(self, graph: fx.Graph) -> int:
        """Match and rewrite all finalize->add_mul->AR->RMSNorm regions.

        Returns the number of regions rewritten. Non-matching graphs are
        left untouched.
        """
        if not _init_ops():
            return 0
        rms_targets = _rms_norm_targets()
        if not rms_targets:
            return 0

        matches: list[_MatchedRegion] = []
        num_candidates = 0
        for node in graph.nodes:
            if node.op == "call_function" and node.target in _MOE_FORWARD_SHARED_OPS:
                num_candidates += 1
                m = _match_region(node, rms_targets)
                if m is not None:
                    matches.append(m)

        if num_candidates and not matches and logger.isEnabledFor(10):
            # Diagnostic: dump the user chain of the first candidate so a
            # mismatch against the expected finalize->mul->add->AR->RMS shape
            # is visible in DEBUG logs.
            for node in graph.nodes:
                if (
                    node.op == "call_function"
                    and node.target in _MOE_FORWARD_SHARED_OPS
                ):
                    chain: list[str] = []
                    cur: fx.Node | None = node
                    for _ in range(10):
                        if cur is None:
                            break
                        users = list(cur.users)
                        chain.append(
                            f"{cur.op}:{cur.target} users="
                            + str([f"{u.op}:{u.target}(args1={u.args[1] if len(u.args) > 1 else None})" for u in users])
                        )
                        cur = users[0] if len(users) == 1 else (users[-1] if users else None)
                    logger.debug(
                        "moe_finalize_ar_rms no-match diagnostic (candidates=%d, "
                        "rms_targets=%s):\n%s",
                        num_candidates,
                        rms_targets,
                        "\n".join(chain),
                    )
                    break

        for m in matches:
            self._apply(graph, m)

        if matches:
            graph.eliminate_dead_code()
        return len(matches)

    def _apply(self, graph: fx.Graph, m: _MatchedRegion) -> None:
        assert m.moe_node is not None and m.rms_node is not None
        with graph.inserting_before(m.moe_node):
            nf = graph.call_function(
                _MOE_FORWARD_SHARED_NO_FINALIZE_OP,
                args=tuple(m.moe_node.args),
                kwargs=dict(m.moe_node.kwargs),
            )
            shared = graph.call_function(operator.getitem, args=(nf, 0))
            gemm2 = graph.call_function(operator.getitem, args=(nf, 1))
            expert_w = graph.call_function(operator.getitem, args=(nf, 2))
            expanded_idx = graph.call_function(operator.getitem, args=(nf, 3))

        with graph.inserting_before(m.rms_node):
            fused = graph.call_function(
                _MOE_FINALIZE_AR_RMS_FUSED_OP,
                args=(
                    gemm2,
                    expert_w,
                    expanded_idx,
                    shared,
                    m.residual_arg,
                    m.weight_arg,
                    m.eps,
                    m.routed_scaling_factor,
                    self.tp_size,
                    self.max_token_num,
                    True,  # launch_with_pdl
                ),
            )
            norm_out = graph.call_function(operator.getitem, args=(fused, 0))
            residual_out = graph.call_function(operator.getitem, args=(fused, 1))

        assert m.norm_getitem is not None
        m.norm_getitem.replace_all_uses_with(norm_out)
        if m.residual_getitem is not None:
            m.residual_getitem.replace_all_uses_with(residual_out)
        # Detach the old chain; eliminate_dead_code removes it (the old
        # moe_forward_shared / all_reduce nodes become dead once the rms
        # getitems are rewired).
        graph.erase_node(m.norm_getitem)
        if m.residual_getitem is not None:
            graph.erase_node(m.residual_getitem)
        graph.erase_node(m.rms_node)
