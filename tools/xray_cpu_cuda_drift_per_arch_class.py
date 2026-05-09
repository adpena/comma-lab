#!/usr/bin/env python3
"""Predict CPU score from CUDA score per architecture-class drift profile.

WHEN TO USE: when a CUDA-only auth eval result has landed and you need to
project the CPU-axis (medal-band) score before deciding whether to spend on
a `[contest-CPU GHA Linux x86_64]` dispatch. The HNeRV cluster registry has
calibrated R_pose=5.04, R_seg=1.17, gap=0.0329; non-HNeRV classes are
uncalibrated-default with 4× wider bands.

WHAT IT REVEALS:
  - architecture class the archive maps to (e.g. ``hnerv_ft_microcodec``)
  - confidence label (``calibrated`` / ``uncalibrated_default``)
  - predicted CPU score (point + 1-σ band)
  - decoder-vs-network attribution of the observed drift
  - whether the predicted CPU score is INSIDE / BORDERLINE / OUTSIDE
    the medal band (PR102 silver 0.19538; PR101 gold 0.19284)
  - actionable next step (dispatch / hold / drop)

Operationally answers: "this archive scores 0.229 [contest-CUDA]; is it
worth $0.06 + 5min on a GHA CPU eval to confirm a 0.196 medal-band CPU
result, or will it land OUTSIDE the medal band?"

NOT a score claim. Predicted bands tagged
``[predicted; learning-layer registry posterior]``. Diagnostic only.

Output:
  experiments/results/xray_cpu_cuda_drift_per_arch_class_<timestamp>/
    drift_prediction.json
    drift_prediction.md
    rebuild_command.txt

Usage:
  .venv/bin/python tools/xray_cpu_cuda_drift_per_arch_class.py \
      --archive experiments/results/track4_sg_a1_t178000_20260509/archive.zip \
      --cuda-score 0.22933 \
      [--label pr107_apogee]
      [--medal-floor 0.19538]
      [--medal-tolerance 0.005]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "xray_cpu_cuda_drift_per_arch_class_v1"
TOOL = "tools/xray_cpu_cuda_drift_per_arch_class.py"

# Default medal-band thresholds from CLAUDE.md / dossier.
DEFAULT_MEDAL_FLOOR = 0.19538  # PR102 silver
DEFAULT_MEDAL_TOLERANCE = 0.005  # 0.005 above silver still counts as borderline


def predict_cpu_band(
    archive_path: Path | None,
    cuda_score: float,
    *,
    metadata: dict | None = None,
) -> dict:
    """Wrapper around the registry's ``predict_cpu_score`` per archive.

    Imports lazily so this CLI doesn't pull torch into every invocation.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from tac.optimization.cuda_cpu_axis_profile_registry import (  # noqa: E402
        bootstrap_registry_from_hnerv_anchors,
        classify_archive_into_profile,
    )

    registry = bootstrap_registry_from_hnerv_anchors()
    arch_class = classify_archive_into_profile(
        archive_path=str(archive_path) if archive_path else None,
        archive_metadata=metadata,
    )
    profile = registry.get(arch_class)
    if profile is None:
        # Class not seeded; create a minimal uncalibrated entry on the fly
        # using HNeRV defaults (this matches the registry's fallback policy).
        from tac.optimization.cuda_cpu_axis_profile_registry import (  # noqa: E402
            ArchitectureProfile,
        )
        profile = ArchitectureProfile(
            architecture_class=arch_class,
            decoder_class="unknown",
            notes="uncalibrated_default fallback",
        )
    pred = profile.predict_cpu_score(cuda_score=cuda_score)
    pred["architecture_class"] = arch_class
    pred["n_anchors"] = profile.n_anchors
    pred["r_pose_mean"] = float(profile.r_pose_mean)
    pred["r_seg_mean"] = float(profile.r_seg_mean)
    return pred


def medal_band_verdict(
    predicted_cpu_score: float,
    predicted_cpu_score_high: float,
    predicted_cpu_score_low: float,
    *,
    medal_floor: float,
    medal_tolerance: float,
) -> dict:
    """Classify the predicted CPU score into a medal-band verdict.

    INSIDE   = point estimate ≤ medal_floor
    BORDERLINE = point estimate ≤ medal_floor + medal_tolerance
    OUTSIDE  = point estimate > medal_floor + medal_tolerance
    UNCERTAIN = band straddles the medal_floor boundary by > medal_tolerance
    """
    inside_floor = float(medal_floor)
    borderline_max = float(medal_floor + medal_tolerance)
    band_half = (predicted_cpu_score_high - predicted_cpu_score_low) / 2.0

    if predicted_cpu_score <= inside_floor:
        if predicted_cpu_score_high > borderline_max:
            return {
                "verdict": "INSIDE_with_uncertainty",
                "rationale": (
                    f"point {predicted_cpu_score:.5f} ≤ medal_floor "
                    f"{inside_floor:.5f} but band high "
                    f"{predicted_cpu_score_high:.5f} > "
                    f"borderline {borderline_max:.5f}"
                ),
            }
        return {
            "verdict": "INSIDE",
            "rationale": (
                f"point {predicted_cpu_score:.5f} ≤ medal_floor "
                f"{inside_floor:.5f}; band confident"
            ),
        }
    if predicted_cpu_score <= borderline_max:
        return {
            "verdict": "BORDERLINE",
            "rationale": (
                f"point {predicted_cpu_score:.5f} ∈ (medal_floor "
                f"{inside_floor:.5f}, borderline {borderline_max:.5f}]"
            ),
        }
    if predicted_cpu_score_low <= borderline_max:
        return {
            "verdict": "UNCERTAIN",
            "rationale": (
                f"point {predicted_cpu_score:.5f} > borderline "
                f"{borderline_max:.5f} but band low "
                f"{predicted_cpu_score_low:.5f} ≤ borderline"
            ),
        }
    return {
        "verdict": "OUTSIDE",
        "rationale": (
            f"point {predicted_cpu_score:.5f} > borderline "
            f"{borderline_max:.5f}; band confident"
        ),
    }


def actionable_next_step(verdict: str, n_anchors: int) -> str:
    """Return the recommended next action based on verdict + calibration."""
    if verdict == "INSIDE":
        return (
            "DISPATCH CPU eval — high probability of medal-band CPU score; "
            "$0.06 GHA spend justified."
        )
    if verdict in {"INSIDE_with_uncertainty", "BORDERLINE"}:
        return (
            "DISPATCH CPU eval with caveat — borderline; result may flip "
            "either way. Worth the $0.06 GHA spend if no cheaper signal "
            "available."
        )
    if verdict == "UNCERTAIN":
        if n_anchors < 3:
            return (
                "HOLD — drift profile is uncalibrated for this architecture "
                "class. Consider building a [contest-CPU] anchor for class "
                "calibration before spending on this archive."
            )
        return (
            "DISPATCH CPU eval — calibrated band straddles medal floor; "
            "the empirical CPU result will resolve which side."
        )
    if verdict == "OUTSIDE":
        return (
            "DROP — predicted CPU score > medal+tolerance with confident "
            "band. Do NOT spend on CPU eval; iterate on the candidate "
            "instead."
        )
    return "REVIEW — verdict label unrecognized"


def render_markdown(report: dict, regen_header: str) -> str:
    p = report["prediction"]
    v = report["medal_verdict"]
    lines = [regen_header, ""]
    lines.append("# CPU-vs-CUDA drift prediction (per-architecture-class)")
    lines.append("")
    lines.append(
        f"_Schema_: `{report['schema_version']}` · _Generated_: "
        f"`{report['generated_at_utc']}`"
    )
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- archive: `{report['archive_path']}`")
    lines.append(f"- label: `{report['label']}`")
    lines.append(f"- CUDA score: **{report['cuda_score']:.5f}**")
    lines.append(f"- medal floor: {report['medal_floor']:.5f}")
    lines.append(f"- medal tolerance: ±{report['medal_tolerance']:.5f}")
    lines.append("")
    lines.append("## Architecture class")
    lines.append("")
    lines.append(f"- class: **`{p['architecture_class']}`**")
    lines.append(f"- confidence: `{p['confidence_label']}`")
    lines.append(f"- n_anchors: {p['n_anchors']}")
    lines.append(f"- R_pose: {p['r_pose_mean']:.3f}")
    lines.append(f"- R_seg: {p['r_seg_mean']:.3f}")
    lines.append("")
    lines.append("## Predicted CPU score")
    lines.append("")
    lines.append(
        f"- point: **{p['predicted_cpu_score']:.5f}** "
        f"(band: [{p['predicted_cpu_score_low']:.5f}, "
        f"{p['predicted_cpu_score_high']:.5f}])"
    )
    lines.append(f"- gap used: {p['score_gap_used']:.5f} ± {p['score_gap_band_half']:.5f}")
    lines.append(f"- evidence: `{p['evidence_grade']}`")
    lines.append("")
    lines.append("## Medal-band verdict")
    lines.append("")
    lines.append(f"- **{v['verdict']}**")
    lines.append(f"- rationale: {v['rationale']}")
    lines.append("")
    lines.append("## Recommended next step")
    lines.append("")
    lines.append(report["recommended_next_step"])
    lines.append("")
    lines.append(
        "_Tag_: `[diagnostic: cpu-vs-cuda drift prediction]` + "
        "`[predicted; learning-layer registry posterior]`. NOT a score "
        "claim. Per CLAUDE.md \"Submission auth eval — BOTH CPU AND CUDA\", "
        "shippable archives still require a `[contest-CPU GHA Linux "
        "x86_64]` empirical CPU eval before any medal claim."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CPU-vs-CUDA drift prediction per arch class. Diagnostic only."
    )
    parser.add_argument("--archive", type=Path, default=None,
                        help="Archive ZIP to classify (optional if --metadata-json provided)")
    parser.add_argument("--metadata-json", type=Path, default=None,
                        help="Pre-computed archive metadata for classification")
    parser.add_argument("--cuda-score", type=float, required=True,
                        help="CUDA score from upstream/evaluate.py [contest-CUDA]")
    parser.add_argument("--label", default="unlabeled")
    parser.add_argument("--medal-floor", type=float, default=DEFAULT_MEDAL_FLOOR)
    parser.add_argument("--medal-tolerance", type=float, default=DEFAULT_MEDAL_TOLERANCE)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.archive is None and args.metadata_json is None:
        print("ERROR: provide --archive OR --metadata-json", file=sys.stderr)
        return 2
    if args.archive is not None and not args.archive.exists():
        print(f"ERROR: archive not found: {args.archive}", file=sys.stderr)
        return 2

    metadata: dict | None = None
    if args.metadata_json is not None:
        try:
            metadata = json.loads(args.metadata_json.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"ERROR loading metadata JSON: {e}", file=sys.stderr)
            return 2

    try:
        prediction = predict_cpu_band(
            archive_path=args.archive,
            cuda_score=args.cuda_score,
            metadata=metadata,
        )
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: prediction failed: {e}", file=sys.stderr)
        return 2

    verdict = medal_band_verdict(
        predicted_cpu_score=prediction["predicted_cpu_score"],
        predicted_cpu_score_high=prediction["predicted_cpu_score_high"],
        predicted_cpu_score_low=prediction["predicted_cpu_score_low"],
        medal_floor=args.medal_floor,
        medal_tolerance=args.medal_tolerance,
    )

    next_step = actionable_next_step(verdict["verdict"], prediction["n_anchors"])

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or (
        REPO_ROOT
        / "experiments"
        / "results"
        / f"xray_cpu_cuda_drift_per_arch_class_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    state_basis = json.dumps(
        {
            "label": args.label,
            "cuda_score": args.cuda_score,
            "arch_class": prediction["architecture_class"],
        },
        sort_keys=True,
    )
    state_hash = hashlib.sha256(state_basis.encode()).hexdigest()[:16]

    report = {
        "schema_version": SCHEMA,
        "tool": TOOL,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "from_state_hash": state_hash,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "predicted_learning_layer_registry_posterior",
        "label": args.label,
        "archive_path": str(args.archive) if args.archive else None,
        "cuda_score": args.cuda_score,
        "medal_floor": args.medal_floor,
        "medal_tolerance": args.medal_tolerance,
        "prediction": prediction,
        "medal_verdict": verdict,
        "recommended_next_step": next_step,
    }
    out_json = out_dir / "drift_prediction.json"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True))

    regen = (
        f"<!-- generated_at: {report['generated_at_utc']}, "
        f"from_state_hash: {report['from_state_hash']} -->"
    )
    out_md = out_dir / "drift_prediction.md"
    out_md.write_text(render_markdown(report, regen))

    parts = [".venv/bin/python tools/xray_cpu_cuda_drift_per_arch_class.py"]
    if args.archive:
        parts.append(f"--archive {args.archive}")
    if args.metadata_json:
        parts.append(f"--metadata-json {args.metadata_json}")
    parts.append(f"--cuda-score {args.cuda_score}")
    parts.append(f"--label {args.label}")
    parts.append(f"--medal-floor {args.medal_floor}")
    parts.append(f"--medal-tolerance {args.medal_tolerance}")
    (out_dir / "rebuild_command.txt").write_text(" \\\n  ".join(parts) + "\n")

    print(f"[xray-drift] wrote {out_json}")
    print(f"[xray-drift] arch={prediction['architecture_class']} "
          f"n_anchors={prediction['n_anchors']} "
          f"predicted_cpu={prediction['predicted_cpu_score']:.5f} "
          f"verdict={verdict['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
