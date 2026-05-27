# SPDX-License-Identifier: MIT
"""Cathedral consumer for the MLX per-pair master-gradient HEURISTIC PRIOR.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
De-orphans `tac.master_gradient_mlx_extractor` /
`tools/extract_master_gradient_mlx.py` per operator directive 2026-05-27
*"Shouldn't that be automated and wired and integrated into a pipeline rather
than orphaned tool?"* + the 7th AUTOMATED+COMPOUNDING+OPTIMAL standing
directive + CLAUDE.md "Results must become system intelligence".

DISTINCT from the sister `master_gradient_per_pair_consumer` package: that
sister reads the PyTorch-AUTHORITY surface (`master_gradient_anchors.jsonl`
via `tac.master_gradient_consumers.load_per_pair_gradient_from_anchor`,
which routes through Catalog #327 `is_authoritative_axis_anchor`). THIS
consumer reads the MLX HEURISTIC-PRIOR surface — the
`.omx/state/mlx_research_signal_manifest.jsonl` NON-PROMOTABLE rows the MLX
extractor lands. Those rows REFUSE `master_gradient_anchors.jsonl` authority
by construction (per `mlx_master_gradient_anchor_blockers`:
`source_runtime_full_frame_parity_missing` +
`canonical_archive_byte_domain_mapping_missing` +
`per_weight_or_per_byte_projector_missing`).

TIER-A OBSERVABILITY-ONLY per Catalog #341 + #357: every return value carries
`predicted_delta_adjustment=0.0` + `promotable=False` + `axis_tag="[predicted]"`
so the MLX heuristic prior CANNOT leak into a score signal. The signal is a
probe-ranking heuristic prior for the closed-form PREDICTION sweep, NOT a
contest-axis score claim (Catalog #192/#127/#323; macOS-MLX research-signal).

RIGOR-GATING (explicit per the de-orphan landing): consumption trust level is
gated by the rigor-review verdict (`master_gradient_analysis_rigor_signal_review_*`).
If that review finds the per-tensor-FD-via-MLX-oracle output is a
uniform-mantissa-projection ARTIFACT rather than a genuine heuristic prior,
this consumer REMAINS Tier-A observability-only and a PyTorch-autograd
authority cross-check (`tools/extract_master_gradient.py`) gates any
promotion. The consumer is SAFE regardless of the rigor verdict because it
NEVER promotes the signal to a score adjustment. It does NOT bake the
"class-shift is the only lever" conclusion into the ranker as fact.

Hook numbers per Catalog #125 6-hook wire-in:
- Hook #1 SENSITIVITY_MAP (per-pair heuristic-prior feeds the 5D canvas)
- Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH (this consumer)
- Hook #5 CONTINUAL_LEARNING_POSTERIOR (new MLX manifest rows / canonical
  equation `mlx_per_pair_master_gradient_per_byte_fd_v1` recalibration)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "mlx_per_pair_master_gradient_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    The MLX manifest rows are persisted by the canonical CLI / pipeline
    (`tac.master_gradient_mlx_pipeline`) under fcntl lock per Catalog
    #131/#138. This consumer is STATELESS (reads the manifest fresh per
    candidate) so the hook is a no-op by design. Per CLAUDE.md "Apples-to-apples
    evidence discipline" + Catalog #192: this hook does NOT promote the
    macOS-MLX research-signal to contest-grade.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — Tier-A observability-only cathedral annotation.

    Looks up the latest MLX per-pair HEURISTIC-PRIOR artifact for the
    candidate's archive via
    `tac.master_gradient_mlx_pipeline.resolve_latest_mlx_artifact_for_sha`
    + reads the operating-point from the canonical manifest. Returns a
    `[predicted]` observability annotation citing the artifact's presence +
    operating point. NO score adjustment per Catalog #341.

    The annotation surfaces presence + operating-point only; it NEVER returns
    raw byte tensors (Catalog #318) and NEVER promotes the heuristic prior to
    authority. Downstream the 5D canvas populator consumes the artifact via
    its canonical loader path.
    """
    archive_sha256 = (
        candidate.get("archive_sha256") if isinstance(candidate, Mapping) else None
    )

    base_absent = {
        "predicted_delta_adjustment": 0.0,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "evidence_grade": "macOS-MLX research-signal",
    }

    if not isinstance(archive_sha256, str) or len(archive_sha256) < 8:
        return {
            **base_absent,
            "rationale": (
                "MLX per-pair heuristic-prior surface available; candidate "
                "carries no archive_sha256 so no artifact lookup attempted "
                "[predicted]"
            ),
        }

    try:
        from tac.master_gradient_mlx_pipeline import (
            resolve_latest_mlx_artifact_for_sha,
        )
    except (ImportError, ModuleNotFoundError):
        return {
            **base_absent,
            "rationale": (
                "tac.master_gradient_mlx_pipeline import unavailable; MLX "
                "per-pair heuristic-prior surface absent [predicted]"
            ),
        }

    try:
        artifact_path = resolve_latest_mlx_artifact_for_sha(archive_sha256)
    except (FileNotFoundError, ValueError, OSError):
        artifact_path = None

    if not artifact_path:
        return {
            **base_absent,
            "rationale": (
                f"no MLX per-pair heuristic-prior artifact for archive "
                f"{archive_sha256[:12]}; auto-trigger pipeline schedules "
                f"extraction on frontier change [predicted]"
            ),
        }

    # Surface the operating point from the manifest (presence + op only;
    # NEVER raw byte tensors per Catalog #318).
    op_summary = _operating_point_summary(archive_sha256)

    return {
        "predicted_delta_adjustment": 0.0,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "evidence_grade": "macOS-MLX research-signal",
        "rationale": (
            f"MLX per-pair HEURISTIC PRIOR available for archive "
            f"{archive_sha256[:12]}: artifact={artifact_path}; "
            f"{op_summary}; macOS-MLX research-signal (NON-PROMOTABLE per "
            f"Catalog #192/#127/#323; refuses master_gradient_anchors "
            f"authority). Trust level gated by the rigor-review verdict "
            f"(master_gradient_analysis_rigor_signal_review_*); consumed as a "
            f"Tier-A heuristic prior for the 5D canvas / closed-form prediction "
            f"sweep ONLY, never promoted to a score adjustment [predicted]"
        ),
    }


def _operating_point_summary(archive_sha256: str) -> str:
    """Read the latest operating-point for the sha from the canonical manifest."""
    try:
        import json

        from tac.master_gradient_mlx_pipeline import (
            MLX_RESEARCH_SIGNAL_MANIFEST_PATH,
        )
    except (ImportError, ModuleNotFoundError):
        return "operating_point=unavailable"
    manifest = MLX_RESEARCH_SIGNAL_MANIFEST_PATH
    if not manifest.exists():
        return "operating_point=unavailable"
    op: dict[str, Any] | None = None
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and obj.get("archive_sha256") == archive_sha256:
            candidate_op = obj.get("operating_point")
            if isinstance(candidate_op, dict):
                op = candidate_op
    if not op:
        return "operating_point=unavailable"
    d_seg = op.get("d_seg")
    d_pose = op.get("d_pose")
    return f"operating_point(d_seg={d_seg}, d_pose={d_pose})"
