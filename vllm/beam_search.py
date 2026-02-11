# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.logprobs import Logprob
from vllm.lora.request import LoRARequest

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalDataDict


@dataclass
class BeamSearchSequence:
    """A sequence for beam search.
    It keeps track of the tokens and the log probability of the sequence.
    The text field is optional and will only be filled when the sequence is
    about to be returned to the user.
    """

    # The tokens include the prompt.
    tokens: list[int]
    logprobs: list[dict[int, Logprob]]
    lora_request: LoRARequest | None = None
    cum_logprob: float = 0.0
    text: str | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    multi_modal_data: "MultiModalDataDict | None" = None
    mm_processor_kwargs: dict[str, Any] | None = None


@dataclass
class BeamSearchOutput:
    """The output of beam search.
    It contains the list of the best beam search sequences.
    The length of the list is equal to the beam width.
    """

    sequences: list[BeamSearchSequence]


class BeamSearchInstance:
    def __init__(
        self,
        prompt_tokens: list[int],
        lora_request: LoRARequest | None = None,
        logprobs: list[dict[int, Logprob]] | None = None,
        **kwargs,
    ):
        self.beams: list[BeamSearchSequence] = [
            BeamSearchSequence(
                tokens=prompt_tokens,
                logprobs=[] if logprobs is None else list(logprobs),
                lora_request=lora_request,
                **kwargs,
            )
        ]
        self.completed: list[BeamSearchSequence] = []


def get_beam_search_score(
    tokens: list[int],
    cumulative_logprob: float,
    eos_token_id: int,
    length_penalty: float = 1.0,
) -> float:
    """Calculate the beam search score with length penalty.

    Adapted from

    https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
    """
    seq_len = len(tokens)
    if tokens[-1] == eos_token_id:
        seq_len -= 1

    return cumulative_logprob / (seq_len**length_penalty)


def create_sort_beams_key_function(eos_token_id: int, length_penalty: float):
    def sort_beams_key(x: BeamSearchSequence) -> float:
        return get_beam_search_score(
            x.tokens, x.cum_logprob, eos_token_id, length_penalty
        )

    return sort_beams_key


# ============================================================
# Tier 3: GPU-Resident Beam Search Components
# ============================================================


@dataclass
class BeamSearchConfig:
    """Immutable configuration for GPU-resident beam search.
    Sent from the engine to the worker once per beam search request.
    """

    beam_width: int
    max_tokens: int
    eos_token_id: int
    length_penalty: float = 1.0
    ignore_eos: bool = False
    temperature: float = 0.0
    prompt_token_ids: list[int] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.prompt_token_ids is None:
            self.prompt_token_ids = []


@dataclass
class BeamSearchState:
    """Mutable GPU-resident state for beam search.
    All tensors live on GPU throughout the beam search loop.
    """

    # [beam_width, prompt_len + max_tokens], int64 — full token sequences
    token_ids: torch.Tensor
    # [beam_width], float32 — cumulative log probabilities
    cum_logprobs: torch.Tensor
    # Number of currently active (non-completed) beams
    num_active: int
    # Completed beams (moved to CPU as they finish)
    completed: list[BeamSearchSequence]

    @staticmethod
    def initialize(
        prompt_token_ids: list[int],
        beam_width: int,
        max_tokens: int,
        device: torch.device,
    ) -> "BeamSearchState":
        """Create initial beam state from prompt tokens."""
        prompt_len = len(prompt_token_ids)
        total_len = prompt_len + max_tokens

        # All beams start with the same prompt
        token_ids = torch.zeros(beam_width, total_len, dtype=torch.long, device=device)
        prompt_tensor = torch.tensor(prompt_token_ids, dtype=torch.long, device=device)
        # Fill prompt into all beam rows
        token_ids[:, :prompt_len] = prompt_tensor.unsqueeze(0)

        cum_logprobs = torch.zeros(beam_width, dtype=torch.float32, device=device)

        return BeamSearchState(
            token_ids=token_ids,
            cum_logprobs=cum_logprobs,
            num_active=1,  # Start with 1 beam, expand after first step
            completed=[],
        )


@torch.no_grad()
def gpu_beam_select(
    logits: torch.Tensor,
    cum_logprobs: torch.Tensor,
    beam_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure GPU beam selection. 4 ops, ~0.02ms on A100.

    Args:
        logits: [num_beams, vocab_size] raw logits from model
        cum_logprobs: [num_beams] current cumulative log probabilities
        beam_width: number of beams to keep

    Returns:
        parent_indices: [2 * beam_width] which old beam each candidate came from
        token_ids: [2 * beam_width] selected token for each candidate
        new_cum_logprobs: [2 * beam_width] updated cumulative logprobs
    """
    # 1. log_softmax -> logprobs
    logprobs = torch.log_softmax(logits.float(), dim=-1)

    # 2. broadcast-add cumulative logprobs
    combined = logprobs + cum_logprobs.unsqueeze(1)

    # 3. flatten + topk (2x candidates like HuggingFace)
    flat = combined.reshape(-1)
    vocab_size = logits.shape[-1]
    k = min(2 * beam_width, flat.shape[0])
    topk_values, topk_indices = torch.topk(flat, k)

    # 4. decode parent beam and token from flat index
    parent_indices = topk_indices // vocab_size
    token_ids = topk_indices % vocab_size

    return parent_indices, token_ids, topk_values
