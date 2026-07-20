# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from enum import IntEnum

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    aux_stream,
    current_stream,
)
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
)

logger = init_logger(__name__)

# Stash for the routed-activation quant issued on the shared-experts aux
# stream (VLLM_SHARED_STREAM_QUANT_OFFLOAD). Indexed by DBO ubatch id,
# mirroring SharedExperts._output; slot 0 is used when DBO is disabled.
# Each entry is (quant_input_ref, a1q, a1q_scale, ready_event) and is
# single-shot: it is populated by maybe_sync_shared_experts_stream and
# consumed (or discarded) by the next _quantize_input call on this thread.
_AUX_STREAM_QUANT_STASH: list[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.cuda.Event] | None
] = [None, None]

# Dedicated stream for the below-MIN_TOKENS routed-activation quant offload
# (VLLM_SHARED_STREAM_QUANT_OFFLOAD_BS1_DEDICATED). Module-level singleton
# mirroring vllm.utils.torch_utils.aux_stream(): a single process-wide
# stream avoids a per-layer stream explosion and keeps profiling sane.
_quant_offload_stream: torch.cuda.Stream | None = None


def quant_offload_stream() -> torch.cuda.Stream | None:
    """Ensures the dedicated quant-offload stream is initialized only once."""
    global _quant_offload_stream

    if _quant_offload_stream is None and current_platform.is_cuda_alike():
        _quant_offload_stream = torch.cuda.Stream()

    return _quant_offload_stream


def consume_aux_stream_quant_stash(
    a1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """Return the aux-stream-quantized (a1q, a1q_scale) for `a1` if a stash
    entry exists and was produced from exactly this tensor, else None.

    The identity check (`is`, not equality) guarantees we only ever skip the
    main-stream quant when the aux stream quantized the very same tensor
    object; any intervening transform (router-weight scaling, dispatch,
    padding) produces a new tensor and falls through to the baseline path.
    """
    idx = dbo_current_ubatch_id()
    entry = _AUX_STREAM_QUANT_STASH[idx]
    if entry is None:
        return None
    # Single-shot: always clear, even on identity mismatch, so a stale
    # entry can never leak into a later layer / step.
    _AUX_STREAM_QUANT_STASH[idx] = None
    input_ref, a1q, a1q_scale, event = entry
    if input_ref is not a1:
        return None
    logger.info_once(
        "VLLM_SHARED_STREAM_QUANT_OFFLOAD: aux-stream quant stash adopted by "
        "_quantize_input (main-stream quant skipped)"
    )
    stream = current_stream()
    stream.wait_event(event)
    # a1q/a1q_scale were allocated on the aux stream; mark their use on the
    # consuming (main) stream so the caching allocator cannot reuse their
    # blocks early.
    a1q.record_stream(stream)
    if a1q_scale is not None:
        a1q_scale.record_stream(stream)
    return a1q, a1q_scale


class SharedExpertsOrder(IntEnum):
    # No shared experts.
    NONE = (0,)

    # No overlap - defensively called before MK.
    NO_OVERLAP = (1,)

    # Overlapped with dispatch/combine in DP/EP - called by the MK.
    MK_INTERNAL_OVERLAPPED = (2,)

    # Overlapped with the gate, router, experts in aux stream.
    MULTI_STREAM_OVERLAPPED = (3,)


class SharedExperts(torch.nn.Module):
    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        enable_dbo: bool,
        mk_can_overlap_shared_experts: Callable[[], bool],
    ):
        super().__init__()

        # The SharedExperts need to handle DBO since they can be called from
        # an MK's finalize method.  We keep a list of outputs indexed by current
        # DBO ubatch id to handle this case.  If DBO is not enabled, the
        # index is always 0 and the second output list element is ignored.
        self.enable_dbo = enable_dbo
        self._output: list[torch.Tensor | None] = [None, None]
        self._layer = layer
        self._moe_config = moe_config

        self._mk_can_overlap_shared_experts = mk_can_overlap_shared_experts

        # Allow disabling of the separate shared experts stream for
        # debug purposes.
        # TODO: Remove this after more extensive testings with TP/DP
        # and other execution modes
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            logger.debug_once("Disabling MoE shared_experts cuda stream")
            self._stream = None
        else:
            # TODO(rob): enable shared expert overlap with non-cuda-alike.
            # aux_stream() returns None on non-cuda-alike platforms.
            self._stream = aux_stream()
            if self._stream is not None:
                logger.debug_once("Enabled separate cuda stream for MoE shared_experts")

    # TODO(bnell): Hack for elastic_ep. Get rid of this
    def _set_moe_config(self, new_moe_config: FusedMoEConfig):
        self.moe_config = new_moe_config

    @property
    def _disable_shared_experts_overlap(self) -> bool:
        # Disable shared expert overlap if:
        #   - we are using eplb with non-default backend, because of correctness issues
        #   - we are using flashinfer with DP, since there nothing to gain
        parallel_config = self._moe_config.moe_parallel_config
        return (
            parallel_config.enable_eplb
            and parallel_config.all2all_backend != "allgather_reducescatter"
        ) or parallel_config.use_fi_nvl_two_sided_kernels

    def _determine_shared_experts_order(
        self,
        hidden_states: torch.Tensor,
    ) -> SharedExpertsOrder:
        if self._disable_shared_experts_overlap:
            return SharedExpertsOrder.NO_OVERLAP

        if self._mk_can_overlap_shared_experts():
            return SharedExpertsOrder.MK_INTERNAL_OVERLAPPED

        should_run_shared_in_aux_stream = (
            current_platform.is_cuda()
            and self._stream is not None
            and hidden_states.shape[0]
            <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
        )

        if should_run_shared_in_aux_stream:
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        else:
            return SharedExpertsOrder.NO_OVERLAP

    def maybe_sync_shared_experts_stream(
        self,
        shared_experts_input: torch.Tensor,
        offload_quant_config: FusedMoEQuantConfig | None = None,
    ):
        experts_order = self._determine_shared_experts_order(shared_experts_input)

        if experts_order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            assert self._stream is not None

            # Record that the clone will be used by shared_experts_stream
            # to avoid gc issue from deallocation of hidden_states_clone
            # For more details: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html # noqa: E501
            # NOTE: We don't need shared_output.record_stream(current_stream())
            # because we synch the streams before using shared_output.
            shared_experts_input.record_stream(self._stream)

            # Three-way placement dispatch for the routed-activation quant.
            # Per-BS gate: below MIN_TOKENS the quant at the head of the aux
            # burst delays the shared-expert MLP enough that the aux stream
            # starts gating the fork/join (measured E2E regression at BS1 on
            # B200 TP8); at/above it the aux slack fully hides the quant.
            # Shape-based branches like the token threshold in
            # _determine_shared_experts_order: num_tokens is fixed per
            # captured CUDA graph, so this is capture-safe.
            offload_at_aux_head = (
                offload_quant_config is not None
                and shared_experts_input.shape[0]
                >= envs.VLLM_SHARED_STREAM_QUANT_OFFLOAD_MIN_TOKENS
            )
            quant_stream = (
                quant_offload_stream()
                if (
                    offload_quant_config is not None
                    and not offload_at_aux_head
                    and envs.VLLM_SHARED_STREAM_QUANT_OFFLOAD_BS1_DEDICATED
                )
                else None
            )

            if quant_stream is not None:
                # Below MIN_TOKENS with the BS1 flag on: fork BOTH side
                # streams from one event recorded on main and run the quant
                # concurrently on the dedicated stream. The aux burst is not
                # touched (unlike the head placement, which delayed it and
                # regressed BS1); the consumer adopts the result through the
                # stash's completion event exactly as in the head placement.
                logger.info_once(
                    "VLLM_SHARED_STREAM_QUANT_OFFLOAD_BS1_DEDICATED: "
                    "routed-activation quant placed on the DEDICATED side "
                    "stream for num_tokens=%d (< MIN_TOKENS=%d); aux burst "
                    "untouched",
                    shared_experts_input.shape[0],
                    envs.VLLM_SHARED_STREAM_QUANT_OFFLOAD_MIN_TOKENS,
                )
                fork_event = torch.cuda.Event()
                fork_event.record(current_stream())
                self._stream.wait_event(fork_event)
                quant_stream.wait_event(fork_event)
                shared_experts_input.record_stream(quant_stream)
                self._enqueue_aux_stream_quant(
                    shared_experts_input, offload_quant_config, stream=quant_stream
                )
            else:
                # Shipped fork, byte-identical for num_tokens >= MIN_TOKENS
                # (and whenever the BS1 dedicated-stream flag is off).
                # Mark sync start point for the aux stream since we will
                # run in parallel with router/gate.
                self._stream.wait_stream(current_stream())

                if offload_at_aux_head:
                    self._enqueue_aux_stream_quant(
                        shared_experts_input, offload_quant_config
                    )

    def _enqueue_aux_stream_quant(
        self,
        x: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        stream: torch.cuda.Stream | None = None,
    ):
        """Issue the routed-activation quant on a side stream
        (VLLM_SHARED_STREAM_QUANT_OFFLOAD).

        By default (`stream=None`) the quant runs at the head of the
        aux-stream burst, in the aux stream's slack ahead of the
        shared-expert MLP. With an explicit `stream` (the dedicated
        quant-offload stream, below MIN_TOKENS) it runs there instead,
        concurrent with the untouched aux burst. Either way the stream has
        already been synchronized with the main stream at the fork, so `x`
        (the pre-routing hidden states) is ready, and a recorded event lets
        the main-stream consumer (`_quantize_input` in the prepare step)
        adopt the result instead of re-quantizing serially.
        """
        # Deferred import: utils pulls in quantization helper modules that
        # must not become import-time deps of this (widely imported) module.
        from vllm.model_executor.layers.fused_moe.utils import (
            moe_kernel_quantize_input,
        )

        if stream is None:
            stream = self._stream

        input_sf = (
            quant_config.a1_gscale
            if quant_config.use_nvfp4_w4a4
            else quant_config.a1_scale
        )
        with torch.cuda.stream(stream):
            a1q, a1q_scale = moe_kernel_quantize_input(
                x,
                input_sf,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
                is_scale_swizzled=quant_config.is_scale_swizzled,
                mx_alignment=quant_config.mx_alignment,
            )
            event = torch.cuda.Event()
            event.record(stream)
        # Index by ubatch id unconditionally (0 when DBO is inactive) so the
        # producer and consumer (consume_aux_stream_quant_stash) always agree.
        _AUX_STREAM_QUANT_STASH[dbo_current_ubatch_id()] = (x, a1q, a1q_scale, event)

    def _run_in_aux_stream(
        self,
        shared_experts_input: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: assert that maybe_sync_shared_experts_stream has been called.

        # Run shared experts in parallel on a separate stream.
        with torch.cuda.stream(self._stream):
            output = self._layer(shared_experts_input)
        current_stream().wait_stream(self._stream)

        return output

    @property
    def _output_idx(self) -> int:
        return dbo_current_ubatch_id() if self.enable_dbo else 0

    @property
    def output(self) -> torch.Tensor:
        assert self._output[self._output_idx] is not None
        output = self._output[self._output_idx]
        self._output[self._output_idx] = None
        return output

    def forward(
        self,
        shared_experts_input: torch.Tensor,
        order: SharedExpertsOrder,
    ):
        experts_order = self._determine_shared_experts_order(shared_experts_input)

        if order != experts_order:
            return None

        assert self._output[self._output_idx] is None

        if order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            self._output[self._output_idx] = self._run_in_aux_stream(
                shared_experts_input
            )
        else:
            self._output[self._output_idx] = self._layer(shared_experts_input)

        assert self._output[self._output_idx] is not None
