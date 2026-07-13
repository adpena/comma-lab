# SPDX-License-Identifier: MIT
"""Typed, research-only policy for exact frozen-SegNet forward transfer.

The policy deliberately contains no selected thread count.  Selection is a
runtime measurement over the finite, host-derived candidate set and is keyed
by the forward signature.  A stale result therefore cannot silently transfer
to another Torch build, model, shape, or CPU topology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, get_args

ExactForwardStrategy = Literal[
    "eager_nchw_autograd",
    "eager_channels_last_autograd",
]


@dataclass(frozen=True)
class SegNetExactForwardTransferPolicy:
    """Compile the fail-closed measurement contract for task #456.

    ``1`` below is not a tuned optimum: it is the cardinality lower bound for
    a CPU worker set.  The winning cardinality is absent from the DSL and must
    be measured for the exact forward signature named by the receipt.
    """

    physical_core_count: int
    torch_default_intraop_threads: int
    batch_size: int
    channels: int
    height: int
    width: int
    logical_core_count: int | None = None
    strategy: ExactForwardStrategy = "eager_nchw_autograd"
    verdict_pair_cardinality: Literal[600] = 600
    matched_sign_alpha: float = 0.01
    matched_sign_alpha_provenance: Literal[
        "OPERATOR_SEALED_TRANSFER_V4_FALSE_POSITIVE_BUDGET"
    ] = "OPERATOR_SEALED_TRANSFER_V4_FALSE_POSITIVE_BUDGET"
    checkpoint_interval_pairs: int = 25
    checkpoint_interval_provenance: Literal[
        "ASSUMED_RECOVERY_ENVELOPE_MAX_25_PAIR_RECOMPUTE"
    ] = "ASSUMED_RECOVERY_ENVELOPE_MAX_25_PAIR_RECOMPUTE"
    exactness_metric: Literal["hard_argmax_bytes"] = "hard_argmax_bytes"
    fallback: Literal["torch_default_intraop_threads"] = "torch_default_intraop_threads"
    research_only: Literal[True] = True
    score_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    pointer_moved: Literal[False] = False
    contest_cpu_measured: Literal[False] = False
    mps_used: Literal[False] = False
    cuda_used: Literal[False] = False

    def __post_init__(self) -> None:
        positive = (
            self.physical_core_count,
            self.torch_default_intraop_threads,
            self.batch_size,
            self.channels,
            self.height,
            self.width,
            self.checkpoint_interval_pairs,
        )
        if any(not isinstance(value, int) or value < 1 for value in positive):
            raise ValueError("core, thread, and forward-shape values must be positive integers")
        if not self.research_only or self.score_claim or self.promotion_eligible:
            raise ValueError("exact-forward transfer remains research-only and non-promotable")
        if self.pointer_moved or self.contest_cpu_measured or self.mps_used or self.cuda_used:
            raise ValueError("exact-forward transfer cannot escalate advisory authority")
        if self.logical_core_count is not None and (
            not isinstance(self.logical_core_count, int) or self.logical_core_count < 1
        ):
            raise ValueError("logical_core_count must be a positive integer when present")
        if self.verdict_pair_cardinality != 600:
            raise ValueError("the operator-sealed transfer verdict requires all 600 real pairs")
        if not (0.0 < self.matched_sign_alpha < 0.5):
            raise ValueError("matched_sign_alpha must lie in (0, 0.5)")

    @classmethod
    def supported_strategies(cls) -> tuple[str, ...]:
        """Expose the typed strategy domain to the runtime probe."""

        return tuple(get_args(ExactForwardStrategy))

    @property
    def forward_work_elements(self) -> int:
        """Shape-derived work signature; not a latency or FLOP claim."""

        return self.batch_size * self.channels * self.height * self.width

    @property
    def effective_thread_ceiling(self) -> int:
        """Fail within the intersection of physical and Torch runtime capacity."""

        observed = [self.physical_core_count, self.torch_default_intraop_threads]
        if self.logical_core_count is not None:
            observed.append(self.logical_core_count)
        return min(observed)

    @property
    def abba_stage_order(self) -> tuple[str, ...]:
        """Process-static terminal order; each stage has a fresh replay child."""

        return ("baseline_rep0", "selected_rep0", "selected_rep1", "baseline_rep1")

    @property
    def candidate_threads(self) -> tuple[int, ...]:
        """Finite host-derived tournament; no winning count is encoded here."""

        return tuple(range(1, self.effective_thread_ceiling + 1))

    @property
    def heuristic_canary_count(self) -> int:
        """Bounded logarithmic screen size; explicitly not an exactness law."""

        return max(1, math.ceil(math.log2(self.effective_thread_ceiling + 1)))

    def compile_measurement_contract(self) -> dict[str, object]:
        return {
            "mode": "exact_frozen_segnet_forward_transfer",
            "strategy": self.strategy,
            "forward_shape_nchw": [
                self.batch_size,
                self.channels,
                self.height,
                self.width,
            ],
            "forward_work_elements": self.forward_work_elements,
            "physical_core_count": self.physical_core_count,
            "logical_core_count": self.logical_core_count,
            "torch_default_intraop_threads": self.torch_default_intraop_threads,
            "effective_thread_ceiling": self.effective_thread_ceiling,
            "candidate_threads": list(self.candidate_threads),
            "canary_count": self.heuristic_canary_count,
            "canary_count_authority": "ASSUMED_HEURISTIC_SCREEN_ONLY",
            "verdict_pair_cardinality": self.verdict_pair_cardinality,
            "matched_sign_alpha": self.matched_sign_alpha,
            "matched_sign_alpha_provenance": self.matched_sign_alpha_provenance,
            "checkpoint_interval_pairs": self.checkpoint_interval_pairs,
            "checkpoint_interval_provenance": self.checkpoint_interval_provenance,
            "process_lifecycle": {
                "method": "fresh_child_process_static_threads",
                "parent_canary_is_terminal_evidence": False,
                "stage_order": list(self.abba_stage_order),
                "measurement_children_per_stage": 1,
                "independent_replay_children_per_stage": 1,
                "bind_intraop_before_model_load": True,
                "bind_interop_before_model_load": True,
                "mid_pass_thread_mutation_forbidden": True,
                "four_way_measurement_sequence_sha_equality_required": True,
                "measurement_replay_sequence_sha_equality_required": True,
                "n600_only_go": True,
            },
            "selection_law": (
                "argmin unpaired heuristic real-canary median forward time over the host-derived "
                "finite candidate set, conditioned on zero hard-argmax flips; the selected arm "
                "then requires a separate full-cardinality matched admission measurement"
            ),
            "admission_predicate": {
                "argmax_flip_count": 0,
                "reference_candidate_argmax_sha256_equal": True,
                "input_gradient_graph_required": True,
                "runtime_fingerprint_exact_match_on_resume": True,
                "static_binding_before_after_equal": True,
                "independent_full_replay_per_pass": True,
                "matched_sign_test_pvalue_lte": self.matched_sign_alpha,
            },
            "fallback": self.fallback,
            "research_only": self.research_only,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "pointer_moved": self.pointer_moved,
            "contest_cpu_measured": self.contest_cpu_measured,
            "mps_used": self.mps_used,
            "cuda_used": self.cuda_used,
            "authority_axis": "[macOS-CPU advisory; training-forward timing only]",
            "live_trainer_argv": [],
        }


__all__ = ["ExactForwardStrategy", "SegNetExactForwardTransferPolicy"]
