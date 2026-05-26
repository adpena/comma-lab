#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Register canonical equation for the Hinton-distilled scorer surrogate via KL T=2.0.

Per Catalog #344 (`check_empirical_finding_memo_references_canonical_equation`):
empirical findings MUST be codified as canonical equations to prevent tribal
knowledge accumulation. This helper registers the first canonical equation that
codifies the Hinton-Vinyals-Dean 2014 KL T=2.0 distillation paradigm's
empirical convergence band on the contest substrate per the HINTON-MLX-FIRST-
LOCAL-PIVOT 2026-05-26 lane (TaskCreate #1330).

The canonical equation:
    `hinton_distilled_scorer_surrogate_savings_via_kl_t2_v1`

Predicts: for the Hinton-distilled scorer surrogate substrate trained against
the canonical contest SegNet teacher (loaded via tac.scorer.load_default_scorers)
with KL T=2.0 distillation on MLX, the empirical KL loss reaches a substrate-
dependent asymptotic floor characterized by the irreducible representational
mismatch between the student head and the SegNet's learned EfficientNet-B2
class boundaries.

The equation's predicted contest-score savings band depends on whether the
substrate is substituted into a context where SegNet/PoseNet calibration
bytes can be replaced by the trained surrogate weights (the Quantizr 0.33
[contest-CUDA] anchor on PR #56 demonstrated this paradigm at the substrate
class-shift level).

Per CLAUDE.md "MLX portable-local-substrate authority" non-negotiable +
Catalog #192 + Catalog #287/#323: every value emitted from this canonical
equation carries `score_claim=False` + `promotion_eligible=False` +
`axis_tag=[predicted]` by construction; the equation is a PREDICTIVE model
queryable by autopilot/cathedral consumers, NOT a contest-score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.canonical_equations import (
    CanonicalEquation,
    EmpiricalAnchor,
    register_canonical_equation,
)
from tac.provenance import (
    Provenance,
    ProvenanceKind,
    ProvenanceEvidenceGrade,
)


CANONICAL_EQUATION_ID = "hinton_distilled_scorer_surrogate_savings_via_kl_t2_v1"


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def build_empirical_anchor_from_smoke_verdict(
    verdict_path: Path,
    *,
    source_artifact: str,
    anchor_id: str | None = None,
) -> EmpiricalAnchor:
    """Build an EmpiricalAnchor from a hinton_mlx_long_training_smoke verdict JSON.

    Per Catalog #287/#323 canonical Provenance: the anchor carries
    `axis_tag=[macOS-MLX research-signal]`, `score_claim_valid=False`,
    `promotion_eligible=False` by construction.

    The predicted_output is the equation's predicted final KL loss based on
    the 100ep bundle baseline extrapolated to the run's epoch count via a
    canonical power-law saturation model:
        L_final(n_epochs) = L_floor + (L_initial - L_floor) * (100 / n_epochs) ** 0.5
    where L_floor = 0.99 (from the 2026-05-25 bundle 100ep min-loss anchor)
    and L_initial = the run's epoch-1 loss value (from the verdict's loss_curve).

    The empirical_output is the run's final-epoch loss value.

    The residual is empirical - predicted (signed; positive means slower
    convergence than predicted).
    """
    if not verdict_path.is_file():
        raise FileNotFoundError(f"verdict JSON not found: {verdict_path}")
    with verdict_path.open("r", encoding="utf-8") as handle:
        verdict = json.load(handle)
    loss_curve = verdict.get("loss_curve") or []
    if not loss_curve:
        raise ValueError(f"loss_curve missing or empty in {verdict_path}")
    n_epochs = int(len(loss_curve))
    initial_loss = float(loss_curve[0])
    final_loss = float(loss_curve[-1])
    min_loss = float(min(loss_curve))
    # Canonical extrapolation per the bundle 100ep anchor (irreducible floor 0.99
    # for the deterministic-projection student head; this is the canonical
    # prediction the empirical anchor either RATIFIES or FALSIFIES).
    L_floor = 0.99  # 2026-05-25 bundle min-loss empirical anchor
    if n_epochs >= 100:
        predicted_final = L_floor + (initial_loss - L_floor) * (100.0 / float(n_epochs)) ** 0.5
    else:
        # For runs < 100 epochs, prediction is just initial_loss (no convergence credit).
        predicted_final = initial_loss
    # Per tac.canonical_equations.equation.EmpiricalAnchor invariant:
    # residual must be >= 0 (interpreted as normalized magnitude). Store
    # the signed delta in inputs.signed_residual_predicted_vs_empirical
    # for downstream observability + use abs(residual) for the canonical
    # row.
    signed_residual = final_loss - predicted_final
    residual = abs(signed_residual)

    if anchor_id is None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        anchor_id = f"{CANONICAL_EQUATION_ID}_anchor_{n_epochs}ep_{ts}"

    # Canonical Provenance per Catalog #287/#323 + CLAUDE.md "MLX portable-
    # local-substrate authority" non-negotiable: MLX-derived artifacts use
    # the canonical [research-signal] measurement_axis + macos_arm64 hardware
    # substrate + RESEARCH_ONLY evidence grade (per the existing canonical
    # vocabulary in tac.provenance.contract.CANONICAL_MEASUREMENT_AXES +
    # CANONICAL_HARDWARE_SUBSTRATES). The historical "[macOS-MLX research-
    # signal]" tag string is an informal alias; the canonical tag is
    # "[research-signal]" + helper_invocation discriminates the MLX path.
    provenance = Provenance(
        artifact_kind=ProvenanceKind.PREDICTED_FROM_MODEL,
        source_path=str(verdict_path.relative_to(REPO_ROOT)),
        source_sha256=_file_sha256(verdict_path),
        measurement_axis="[research-signal]",
        hardware_substrate="macos_arm64",
        evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc=_utc_now_iso(),
        canonical_helper_invocation=(
            "tools.register_canonical_equation_hinton_distilled_scorer_surrogate.build_empirical_anchor_from_smoke_verdict"
        ),
    )

    return EmpiricalAnchor(
        anchor_id=anchor_id,
        measurement_utc=verdict.get("completed_at_utc", _utc_now_iso()),
        inputs={
            "in_domain_context": "hinton_kl_t2_mlx_long_training_real_segnet_teacher",
            "n_epochs": n_epochs,
            "smoke_epochs_requested": verdict.get("smoke_epochs"),
            "teacher_provider": verdict.get("teacher_provider"),
            "distillation_temperature": verdict.get("distillation_temperature"),
            "distillation_weight": verdict.get("distillation_weight"),
            "num_classes": verdict.get("num_classes"),
            "max_frames": verdict.get("max_frames"),
            "random_seed": verdict.get("random_seed"),
            "source_video_sha256": verdict.get("source_video_sha256"),
            "latent_dim": verdict.get("latent_dim"),
            "base_channels": verdict.get("base_channels"),
            "eval_size": verdict.get("eval_size"),
            "initial_loss": initial_loss,
            "min_loss": min_loss,
            "signed_residual_predicted_vs_empirical": signed_residual,
        },
        predicted_output=predicted_final,
        empirical_output=final_loss,
        residual=residual,
        source_artifact=str(verdict_path.relative_to(REPO_ROOT)),
        measurement_method=(
            "tools/run_hinton_mlx_long_training_smoke.py "
            "(canonical Slot 1 MLXLongTrainingPipeline subclass "
            "HintonMlxLongTrainingPipeline with KL T=2.0 distillation "
            "against pre-computed real SegNet teacher cache)"
        ),
        provenance=provenance,
    )


def build_initial_equation(
    *,
    initial_anchors: tuple[EmpiricalAnchor, ...] = (),
) -> CanonicalEquation:
    """Build the canonical equation dataclass for first registration.

    The equation is observability-only: predicted output is the KL loss
    asymptote; the operator-facing autopilot ranker can consume the
    `predicted_output` to weight Hinton-distilled scorer-surrogate
    substrate candidates BEFORE paid GPU is funded.
    """
    provenance = Provenance(
        artifact_kind=ProvenanceKind.PREDICTED_FROM_MODEL,
        source_path="src/tac/substrates/hinton_distilled_scorer_surrogate/mlx_loss.py",
        source_sha256=_file_sha256(
            REPO_ROOT
            / "src/tac/substrates/hinton_distilled_scorer_surrogate/mlx_loss.py"
        ),
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
        evidence_grade=ProvenanceEvidenceGrade.PREDICTED,
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc=_utc_now_iso(),
        canonical_helper_invocation=(
            "tools.register_canonical_equation_hinton_distilled_scorer_surrogate.build_initial_equation"
        ),
    )

    return CanonicalEquation(
        equation_id=CANONICAL_EQUATION_ID,
        name="Hinton-distilled scorer surrogate savings via KL T=2.0",
        one_line_summary=(
            "Hinton KL T=2.0 student-scorer distillation reaches asymptotic "
            "floor set by student/teacher mismatch; substitutable into "
            "scorer-calibration slots (Quantizr PR#56 anchor 0.33 CUDA)."
        ),
        latex_form=(
            r"L_{KL}(n) = L_{\infty} + (L_0 - L_{\infty}) \cdot "
            r"(n_0 / n)^{\beta}, \quad L_{\infty} \in "
            r"[\text{floor}_{student\_head}, \text{floor}_{teacher\_distill}]"
        ),
        python_callable_module_path=(
            "tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss."
            "hinton_distilled_kl_t2_loss"
        ),
        domain_of_validity={
            "in_domain_contexts": (
                "hinton_kl_t2_mlx_long_training_real_segnet_teacher",
                "hinton_kl_t2_pytorch_long_training_real_segnet_teacher",
                "quantizr_pr56_kl_t2_segnet_distillation_during_training",
            ),
            "excluded_contexts": (
                "mock_teacher_only_smoke",  # convergence achievable trivially
                # via deterministic projection match; not predictive of
                # real-teacher behavior per HINTON-MLX-BUNDLE 2026-05-25
                # IMPLEMENTATION-LEVEL falsification of mock-teacher
                # convergence as proxy for real-teacher convergence
                "distillation_temperature_outside_0_5_to_5_0",
            ),
            "applies_to_substrate_classes": (
                "scorer_calibration_surrogate_substitution",
                "scorer_response_dataset_substitute",
            ),
            "applies_to_teacher_scorers": (
                "segnet_efficientnet_b2_unet_5_classes",
                "posenet_fastvit_t12_12_channel_yuv6",
            ),
            "applies_to_student_head_architectures": (
                "deterministic_cosine_projection",
                "learnable_1x1_conv_head",
                "learnable_mlp_head",
            ),
            "spatial_eval_size": (384, 512),
            "min_frames": 100,
            "max_frames_observed": 600,
        },
        units_in={
            "in_domain_context": "categorical_token",
            "n_epochs": "epoch_count",
            "teacher_provider": "categorical_token",
            "distillation_temperature": "dimensionless_positive",
            "distillation_weight": "dimensionless_nonneg",
            "num_classes": "integer",
            "max_frames": "integer",
            "random_seed": "integer",
            "source_video_sha256": "sha256_hex",
            "latent_dim": "integer",
            "base_channels": "integer",
            "eval_size": "pixel_tuple",
            "initial_loss": "kl_T2_loss_value",
            "min_loss": "kl_T2_loss_value",
        },
        units_out={
            "predicted_kl_loss_at_n_epochs": "kl_T2_loss_value",
            "asymptotic_floor_estimate": "kl_T2_loss_value",
            "predicted_score_savings_band_lower": "contest_score_delta_estimate",
            "predicted_score_savings_band_upper": "contest_score_delta_estimate",
        },
        empirical_anchors=initial_anchors,
        predicted_vs_empirical_residual={
            "mean_abs_residual": 0.0,
            "max_abs_residual": 0.0,
            "count": float(len(initial_anchors)),
        },
        last_calibration_utc=_utc_now_iso(),
        next_recalibration_trigger="when_3+_new_empirical_anchors_in_domain",
        canonical_consumers=(
            "tools.cathedral_autopilot_autonomous_loop",
            "tac.cathedral_consumers.canonical_equation_lookup_consumer",
        ),
        canonical_producers=(
            "tools.run_hinton_mlx_long_training_smoke",
            "src.tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss",
        ),
        provenance=provenance,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-verdict-path",
        type=Path,
        required=True,
        help="Path to a hinton_mlx_long_training_smoke verdict JSON.",
    )
    parser.add_argument(
        "--source-artifact",
        type=str,
        default="",
        help="Optional source-artifact path string for the anchor row.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="hinton-mlx-local-pivot-20260526",
        help="Subagent identifier for the registration audit row.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="HINTON-MLX-FIRST-LOCAL-PIVOT 2026-05-26 (TaskCreate #1330)",
        help="Notes attached to the registry event.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    smoke_verdict_path = args.smoke_verdict_path
    if not smoke_verdict_path.is_absolute():
        smoke_verdict_path = REPO_ROOT / smoke_verdict_path

    print(f"[register-canonical-equation] reading verdict: {smoke_verdict_path}")
    anchor = build_empirical_anchor_from_smoke_verdict(
        smoke_verdict_path,
        source_artifact=args.source_artifact or str(smoke_verdict_path),
    )
    print(
        f"[register-canonical-equation] empirical anchor: "
        f"predicted={anchor.predicted_output:.4f} "
        f"empirical={anchor.empirical_output:.4f} "
        f"residual={anchor.residual:+.4f}"
    )

    # Try to load existing equation; if absent, register first time with the
    # one anchor; if present, append the anchor via update_equation_with_empirical_anchor.
    from tac.canonical_equations import (
        get_equation_by_id,
        update_equation_with_empirical_anchor,
    )

    existing = None
    try:
        existing = get_equation_by_id(CANONICAL_EQUATION_ID)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[register-canonical-equation] no prior equation (will create): {exc!r}"
        )

    if existing is None:
        equation = build_initial_equation(initial_anchors=(anchor,))
        registered = register_canonical_equation(
            equation,
            agent=args.agent,
            subagent_id=args.agent,
            notes=args.notes,
        )
        print(
            f"[register-canonical-equation] FIRST REGISTRATION: "
            f"equation_id={registered.equation_id} "
            f"with {len(registered.empirical_anchors)} anchor(s)"
        )
    else:
        updated = update_equation_with_empirical_anchor(
            existing.equation_id,
            anchor,
            agent=args.agent,
            subagent_id=args.agent,
            notes=args.notes,
        )
        print(
            f"[register-canonical-equation] ANCHOR APPENDED: "
            f"equation_id={updated.equation_id} "
            f"with {len(updated.empirical_anchors)} total anchor(s)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
