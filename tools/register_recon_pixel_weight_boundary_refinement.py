# SPDX-License-Identifier: MIT
"""Register the HONEST WAVE-5E boundary-refinement anchor on the canonical equation
``detector_informed_recon_weight_d_seg_savings_v1`` (Catalog #344 + #307).

WHAT THE A/B FOUND (read from the REAL ab_output.json — NO fabricated numbers):
  The full-grid-SegNet-saliency recon_pixel_weight does NOT beat uniform on a
  DENSE per-pixel recon-weight optimization (sal == uni d_seg to the noise floor,
  at both a generous and a binding L_inf budget). The A/B artifact records score
  and d_seg fields only; this registrar does NOT infer or claim correction-
  concentration statistics that are not present in the artifact.

  This REFINES the sister #1587 equation: the lever is SURFACE-SPECIFIC. It is
  CONTEST_RELEVANT at the sparse-delta KEEP-OR-DROP allocation surface (#1587,
  ``pack_sparse_delta`` hard byte budget — the weight decides WHICH pixels survive),
  but NOT at the dense continuous recon-weight surface (the weight only reorders
  convergence; the dense correction fixes the argmax-flipping pixels regardless).

Per CLAUDE.md "Forbidden premature KILL": this is a boundary REFINEMENT (an
IMPLEMENTATION-LEVEL falsification per Catalog #307 of the dense-recon-weight
APPLICATION), NOT a kill of the lever's paradigm. The equation stays valid at the
allocation surface; the new anchor records where it does NOT apply.

NON-PROMOTABLE per Catalog #192/#341/#127/#323 ([macOS-CPU advisory]).
FAIL-CLOSED: refuses if the ab_output.json is missing or malformed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.canonical_equations import (  # noqa: E402
    EmpiricalAnchor,
    update_equation_with_empirical_anchor,
)
from tac.provenance import build_provenance_for_macos_cpu_advisory  # noqa: E402

EQUATION_ID = "detector_informed_recon_weight_d_seg_savings_v1"


def _build_boundary_refinement_anchor(
    *,
    row: dict[str, Any],
    now: str,
    source_artifact: str,
    provenance: Any,
) -> EmpiricalAnchor:
    """Build one honest dense-surface boundary-refinement anchor.

    The transferred model claim from #1587 is directional: detector-informed
    saliency should produce a positive d_seg margin versus uniform. For this
    dense recon-weight surface the empirical margin is non-positive, so the
    residual is a sign/boundary miss (1.0), not a perfect numeric residual.
    """
    d_uni = float(row["d_seg_uniform"])
    d_sal = float(row["d_seg_full_grid_saliency"])
    empirical_margin = d_uni - d_sal  # >0 would mean saliency helps
    empirical_sign = "positive" if empirical_margin > 0.0 else "non_positive"
    margin_sign_match = empirical_sign == "positive"
    return EmpiricalAnchor(
        anchor_id=f"recon_weight_dense_palette{row['n_levels']}",
        measurement_utc=now,
        inputs={
            "surface": "dense_per_pixel_recon_weight_optimization",
            "transferred_from_surface": "sparse_delta_allocation_ranking",
            "operating_point": f"palette_n_levels_{row['n_levels']}",
            "baseline_d_seg": float(row["baseline_d_seg"]),
            "saliency_nonzero_fraction": float(row["saliency_nonzero_fraction"]),
            "verdict": str(row["verdict"]),
        },
        predicted_output={
            "margin_saliency_vs_uniform_sign": "positive",
            "residual_type": "sign_boundary_miss",
        },
        empirical_output={
            "d_seg_uniform": d_uni,
            "d_seg_full_grid_saliency": d_sal,
            "d_seg_s_uniward_texture": float(row["d_seg_s_uniward_texture"]),
            "margin_saliency_vs_uniform": empirical_margin,
            "margin_saliency_vs_uniform_sign": empirical_sign,
            "margin_sign_match": margin_sign_match,
            "saliency_beats_uniform": bool(row["saliency_beats_uniform"]),
        },
        residual=0.0 if margin_sign_match else 1.0,
        source_artifact=source_artifact,
        measurement_method=("real_segnet_d_seg_argmax_flip_rate_on_recon_weighted_correction"),
        provenance=provenance,
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ab-output",
        default="experiments/results/recon_pixel_weight_real_render_ab_20260531/ab_output.json",
    )
    ap.add_argument(
        "--subagent-id",
        default="recon_pixel_weight_canonical_channel_20260531",
    )
    args = ap.parse_args()

    ab_path = REPO_ROOT / args.ab_output
    if not ab_path.is_file():
        print(f"[register] FAIL-CLOSED: ab output not found at {ab_path}", file=sys.stderr)
        return 2
    ab = json.loads(ab_path.read_text())
    if ab.get("schema") != "recon_pixel_weight_real_render_ab_v1":
        print(f"[register] FAIL-CLOSED: unexpected schema {ab.get('schema')!r}", file=sys.stderr)
        return 2

    per_palette = ab.get("per_palette", {})
    if not per_palette:
        print("[register] FAIL-CLOSED: no per_palette results", file=sys.stderr)
        return 2

    now = datetime.now(UTC).isoformat()
    import hashlib

    ab_sha = hashlib.sha256(ab_path.read_bytes()).hexdigest()
    prov = build_provenance_for_macos_cpu_advisory(
        archive_sha256=ab_sha,
        source_path=str(ab_path.relative_to(REPO_ROOT)),
        captured_at_utc=now,
    )

    # One anchor per palette operating point. The transferred MODEL CLAIM (from
    # #1587) is directional: "saliency reduces d_seg over uniform". At this dense
    # recon-weight surface the empirical sign is non-positive, so residual records
    # a sign/boundary miss instead of laundering the no-transfer result as 0.0.
    appended = 0
    for _key, row in per_palette.items():
        anchor = _build_boundary_refinement_anchor(
            row=row,
            now=now,
            source_artifact=str(ab_path.relative_to(REPO_ROOT)),
            provenance=prov,
        )
        update_equation_with_empirical_anchor(
            EQUATION_ID,
            anchor,
            agent="claude",
            subagent_id=args.subagent_id,
            notes=(
                "WAVE-5E boundary refinement (Catalog #307 IMPLEMENTATION-LEVEL): "
                "the detector-informed recon weight is SURFACE-SPECIFIC — "
                "CONTEST_RELEVANT at the sparse-delta allocation surface (#1587) but "
                "NOT at the dense recon-weight surface (saliency == uniform d_seg). "
                "The dense A/B artifact records d_seg evidence only; this anchor "
                "does not infer correction-concentration statistics."
            ),
        )
        appended += 1
        print(
            f"[register] appended anchor recon_weight_dense_palette{row['n_levels']}: "
            f"empirical_margin={anchor.empirical_output['margin_saliency_vs_uniform']:+.6f} "
            f"residual={anchor.residual:+.6f} "
            f"verdict={row['verdict']}",
            flush=True,
        )

    print(
        f"[register] appended {appended} boundary-refinement anchors to "
        f"{EQUATION_ID} (surface-specific: lever does NOT transfer to dense "
        f"recon-weight surface).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
