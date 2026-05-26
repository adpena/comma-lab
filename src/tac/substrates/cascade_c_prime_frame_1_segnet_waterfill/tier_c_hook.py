# SPDX-License-Identifier: MIT
"""Tier-C MDL ablation prober hook for cascade_c_prime_frame_1_segnet_waterfill.

Per CLAUDE.md FORBIDDEN_PATTERNS "Forbidden predicted_band-from-random-init-
Tier-C-density (the phantom-predicted-band trap)" + Catalog #324
``check_no_predicted_band_without_post_training_tier_c_validation``: the
substrate's predicted ΔS band MUST be validated via post-training Tier-C
density re-measurement on the LANDED ARCHIVE (≥1 epoch trained), NOT on
random-init weights.

THIS hook provides the canonical interface for the post-training Tier-C
validation step. Per the per-substrate symposium PROCEED_WITH_REVISIONS verdict
(revision #4): post-training Tier-C density re-measurement is the operator-
routable gate that promotes the synthesis -0.058820 prediction from
CARGO-CULTED (Contrarian + Atick dissent flagged 10-30× literature
overestimate) to HARD-EARNED-EMPIRICALLY-VERIFIED.

**Status at L0 scaffold**: ``FORMALIZATION_PENDING`` per Catalog #344 canonical
equations registry discipline. The hook EMITS the canonical command for
``tools/mdl_scorer_conditional_ablation.py --tier c`` invocation but does NOT
invoke the tool itself (paired-CUDA dispatch operator-decision-required per
CLAUDE.md "Executing actions with care").

## Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Tier-C tool invocation | CANONICAL (`tools/mdl_scorer_conditional_ablation.py`) | sister landings: NSCS06 / Z6 / C6 IBPS / DP1 all route through this canonical CLI |
| Verdict classification | CANONICAL (within-class / across-class / indeterminate; mirror Catalog #227) | per `tac.canonical_equations.is_residual_hybrid_context` sister |
| Substrate-specific consumer | UNIQUE (this module) | per-substrate canonical-equation-anchor proposal context binding |
| Predicted band validation | CANONICAL (Catalog #324 + `PredictedBandWithValidation`) | reuses canonical sister surface |

## 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map: N/A
- hook #2 Pareto constraint: ACTIVE (Tier-C verdict ratifies/refutes the
  predicted Pareto band; within-class → no class-shift bonus per Catalog #227)
- hook #3 bit-allocator: N/A
- hook #4 cathedral autopilot dispatch: ACTIVE (autopilot ranker can consume
  Tier-C verdict via `tac.canonical_equations` registry per Catalog #344)
- hook #5 continual-learning posterior: ACTIVE (Tier-C verdict feeds posterior
  per `tac.council_continual_learning.append_council_anchor`)
- hook #6 probe-disambiguator: ACTIVE PRIMARY (Tier-C MDL ablation IS the
  canonical disambiguator between within-class refinement vs class-shift)

## NO_SUPERSESSION_NEEDED:adds_new_hook_module_does_not_supersede_existing_canonical_tier_c_tool_per_Catalog_110_113_APPEND_ONLY_HISTORICAL_PROVENANCE
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .substrate_contract import CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT

__all__ = [
    "DEFAULT_TIER_C_TOOL_PATH",
    "FORMALIZATION_PENDING_VERDICT",
    "TierCAblationHookVerdict",
    "TierCAblationProbeRequest",
    "build_tier_c_ablation_probe_request",
    "classify_tier_c_density_verdict",
]


DEFAULT_TIER_C_TOOL_PATH: str = "tools/mdl_scorer_conditional_ablation.py"
"""Canonical Tier-C MDL ablation tool path; sister landings consume this."""


FORMALIZATION_PENDING_VERDICT: str = "FORMALIZATION_PENDING"
"""Per Catalog #344 sister discipline: scaffold-time verdict before paired-CUDA."""


# Tier-C density verdict bands per Catalog #227 sister gate.
_TIER_C_WITHIN_CLASS_THRESHOLD: float = 0.70
"""Density ≥ 0.70 = within-class (sister of A1/PR106 saturated archives)."""

_TIER_C_ACROSS_CLASS_THRESHOLD: float = 0.30
"""Density ≤ 0.30 = across-class (sister of C6 MDL-IBPS / DP1 first-anchor)."""


@dataclass(frozen=True)
class TierCAblationProbeRequest:
    """Canonical request envelope for `tools/mdl_scorer_conditional_ablation.py`.

    Per CLAUDE.md "Forbidden score claims": this is a REQUEST envelope; the
    tool itself emits the empirical verdict. The hook builds the canonical CLI
    invocation + Tier-C aggregation JSON path expected by sister consumers.
    """

    substrate_id: str
    archive_sha256: str
    """Post-training archive sha256 (per HARD-EARNED Catalog #324 verdict;
    NEVER random-init or pre-training-archive sha)."""
    archive_path: Path
    """Path to landed contest archive (e.g.
    `experiments/results/.../submission_dir/archive.zip`)."""
    output_json_path: Path
    """Where the Tier-C tool will write `*_mdl_ablation.json` per the canonical
    schema with `tier_c` rows."""
    canonical_equation_proposal: str = (
        "atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1"
    )
    """Per Cascade C' canonical equation #344 anchor proposal memo
    (`feedback_cascade_c_prime_canonical_equation_344_anchor_proposal_20260526.md`)."""

    def canonical_cli_invocation(self) -> tuple[str, ...]:
        """Return the canonical CLI args for `tools/mdl_scorer_conditional_ablation.py`.

        Per CLAUDE.md "Operator gates must be wired and used": this returns the
        explicit invocation tuple operators can review BEFORE firing.
        """
        return (
            ".venv/bin/python",
            DEFAULT_TIER_C_TOOL_PATH,
            "--archive",
            str(self.archive_path),
            "--archive-sha256",
            self.archive_sha256,
            "--substrate-id",
            self.substrate_id,
            "--output-json",
            str(self.output_json_path),
            "--tier",
            "c",
        )


@dataclass(frozen=True)
class TierCAblationHookVerdict:
    """MLX-LOCAL hook verdict per Catalog #324 + #344.

    Per Catalog #287/#323 canonical Provenance: scaffold-time verdict is
    FORMALIZATION_PENDING; paired-CUDA verdict promotes via the canonical
    helper. Score-axis fields ALWAYS carry non-promotable provenance.
    """

    substrate_id: str
    formalization_status: str
    """One of: FORMALIZATION_PENDING / VALIDATED_POST_TRAINING / FALSIFIED."""
    probe_request: Optional[TierCAblationProbeRequest] = None
    tier_c_density_estimate: Optional[float] = None
    """Post-training Tier-C density [0.0, 1.0]; None at scaffold-time."""
    substrate_class_verdict: Optional[str] = None
    """One of: within_class / across_class / indeterminate; None pre-empirical."""
    operator_routable_note: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def axis_tag(self) -> str:
        if self.formalization_status == FORMALIZATION_PENDING_VERDICT:
            return "[macOS-MLX research-signal]"
        return "[contest-CUDA]"

    @property
    def is_pending(self) -> bool:
        return self.formalization_status == FORMALIZATION_PENDING_VERDICT


def classify_tier_c_density_verdict(density: float) -> str:
    """Classify Tier-C density into within_class / across_class / indeterminate.

    Per Catalog #227 sister bands (within-class density ≥ 0.70 floors at
    -0.005; across-class density ≤ 0.30 adds -0.01 to -0.03 reward).
    """
    if not (0.0 <= density <= 1.0):
        raise ValueError(f"density must be in [0.0, 1.0]; got {density}")
    if density >= _TIER_C_WITHIN_CLASS_THRESHOLD:
        return "within_class"
    if density <= _TIER_C_ACROSS_CLASS_THRESHOLD:
        return "across_class"
    return "indeterminate"


def build_tier_c_ablation_probe_request(
    *,
    archive_sha256: str,
    archive_path: Path,
    output_dir: Path,
) -> TierCAblationHookVerdict:
    """Build canonical Tier-C MDL ablation probe request for this substrate.

    Per CLAUDE.md FORBIDDEN_PATTERNS "phantom-predicted-band trap" + Catalog
    #324: the request demands a POST-TRAINING archive sha256 (verified via the
    canonical helper's `__post_init__` if the archive path is inspected); the
    hook does NOT invoke the tool itself per CLAUDE.md "Executing actions with
    care" (PAID dispatch operator-decision-required per per-substrate symposium
    revision #3).

    Args:
        archive_sha256: post-training archive sha256 (≥1 epoch trained).
        archive_path: path to landed contest archive.zip.
        output_dir: directory for tool's JSON output emission.

    Returns:
        TierCAblationHookVerdict with FORMALIZATION_PENDING status + canonical
        CLI invocation embedded.
    """
    substrate_id = CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT.id
    output_json = output_dir / f"{substrate_id}_tier_c_mdl_ablation.json"
    request = TierCAblationProbeRequest(
        substrate_id=substrate_id,
        archive_sha256=archive_sha256,
        archive_path=archive_path,
        output_json_path=output_json,
    )
    return TierCAblationHookVerdict(
        substrate_id=substrate_id,
        formalization_status=FORMALIZATION_PENDING_VERDICT,
        probe_request=request,
        operator_routable_note=(
            "Tier-C MDL ablation probe canonical CLI built; operator-routable "
            "via per-substrate symposium revision #4 (`tools/mdl_scorer_"
            "conditional_ablation.py --tier c` on landed paired-CUDA archive). "
            "Promotion of `atick_redlich_asymmetric_scorer_channel_lagrangian_"
            "routing_savings_v1` canonical equation #344 anchor to "
            "VALIDATED_POST_TRAINING gates this probe verdict."
        ),
        provenance={
            "canonical_equation_proposal": (
                "atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1"
            ),
            "canonical_equation_status": "FORMALIZATION_PENDING",
            "per_substrate_symposium_revision": "revision_4_tier_c_post_training",
            "score_claim": False,
            "promotion_eligible": False,
            "axis_tag": "[macOS-MLX research-signal]",
            "evidence_grade": "request-envelope-not-empirical",
        },
    )
