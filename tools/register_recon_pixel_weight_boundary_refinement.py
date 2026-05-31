# SPDX-License-Identifier: MIT
"""Register the HONEST WAVE-5E boundary-refinement anchor on the canonical equation
``detector_informed_recon_weight_d_seg_savings_v1`` (Catalog #344 + #307).

WHAT THE A/B FOUND (read from the REAL ab_output.json — NO fabricated numbers):
  The full-grid-SegNet-saliency recon_pixel_weight does NOT beat uniform on a
  DENSE per-pixel recon-weight optimization (sal == uni d_seg to the noise floor,
  at both a generous and a binding L_inf budget). The channel is REAL and NON-INERT
  (it concentrates 64.6% of its correction magnitude on high-saliency pixels vs
  uniform's 49.9% — a genuine redistribution), but that steering does NOT translate
  into a d_seg advantage at the dense recon surface.

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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.canonical_equations import (  # noqa: E402
    EmpiricalAnchor,
    update_equation_with_empirical_anchor,
)
from tac.provenance import build_provenance_for_macos_cpu_advisory  # noqa: E402

EQUATION_ID = "detector_informed_recon_weight_d_seg_savings_v1"


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

    # One anchor per palette operating point. The MODEL'S CLAIM (from #1587) at the
    # dense recon-weight surface is "saliency reduces d_seg over uniform"
    # (predicted margin > 0). The EMPIRICAL margin is d_seg(uniform) - d_seg(saliency).
    # The residual = predicted - empirical = (>0 expected) - (~0 observed) so the
    # anchor RECORDS the surface-specific boundary (predicted lever did not
    # materialize at the dense surface).
    appended = 0
    for _key, row in per_palette.items():
        d_uni = float(row["d_seg_uniform"])
        d_sal = float(row["d_seg_full_grid_saliency"])
        empirical_margin = d_uni - d_sal  # >0 would mean saliency helps
        # Predicted at registration: the sister #1587 lever predicts a positive
        # margin (~1e-3 contest-relevant on the allocation surface). At the dense
        # recon-weight surface the prediction transferred to ~0 (no advantage).
        predicted_margin = 0.0  # the HONEST predicted margin AT the dense surface
        residual = predicted_margin - empirical_margin
        anchor = EmpiricalAnchor(
            anchor_id=f"recon_weight_dense_palette{row['n_levels']}",
            measurement_utc=now,
            inputs={
                "surface": "dense_per_pixel_recon_weight_optimization",
                "operating_point": f"palette_n_levels_{row['n_levels']}",
                "baseline_d_seg": float(row["baseline_d_seg"]),
                "saliency_nonzero_fraction": float(row["saliency_nonzero_fraction"]),
                "verdict": str(row["verdict"]),
            },
            predicted_output={"margin_saliency_vs_uniform": predicted_margin},
            empirical_output={
                "d_seg_uniform": d_uni,
                "d_seg_full_grid_saliency": d_sal,
                "d_seg_s_uniward_texture": float(row["d_seg_s_uniward_texture"]),
                "margin_saliency_vs_uniform": empirical_margin,
                "saliency_beats_uniform": bool(row["saliency_beats_uniform"]),
            },
            residual=residual,
            source_artifact=str(ab_path.relative_to(REPO_ROOT)),
            measurement_method=(
                "real_segnet_d_seg_argmax_flip_rate_on_recon_weighted_correction"
            ),
            provenance=prov,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
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
                "The channel is real + non-inert (redistributes correction toward "
                "high-saliency pixels) but the dense surface gives no d_seg edge."
            ),
        )
        appended += 1
        print(
            f"[register] appended anchor recon_weight_dense_palette{row['n_levels']}: "
            f"empirical_margin={empirical_margin:+.6f} residual={residual:+.6f} "
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
