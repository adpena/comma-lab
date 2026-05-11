#!/usr/bin/env python3
"""Loader drift xray — PyAV (CPU) vs DALI (CUDA) decoded RGB (handoff P5 D3).

Per CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval — BOTH CPU AND
CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE": this tool NEVER produces a score
claim. It quantifies how much of the device-axis drift comes from the
**ground-truth loader path** (PyAV/FFmpeg vs DALI/NVDEC) BEFORE PoseNet/SegNet
sees a single byte.

This is the bridge between the substrate-class-boundary hypothesis (A1 vs
PR106 device-axis flips) and the layer-drift xrays in
`cpu_cuda_xray_segnet_layer_drift.py` / `cpu_cuda_xray_posenet_layer_drift.py`:

- If PyAV-vs-DALI raw-RGB delta is LARGE (e.g., max-abs > 0 LSB on many pixels),
  loader drift contributes to the observed device-axis score drift directly.
- If PyAV-vs-DALI raw-RGB delta is SMALL/ZERO, the entire 5× pose ratio
  originates downstream (preprocess / scorer-forward / threshold geometry).

This tool is a thin synthesis wrapper around the canonical
`tools/probe_eval_loader_drift.py` (which already implements the full 2x2
diagnostic cell matrix in `INTENDED_CELL_SPECS`). Its job:

1. Always run the locally-available cells (PyAV/CPU on macOS / Linux) and emit
   shared-input tensor custody files for downstream consumption by the
   SegNet/PoseNet layer-drift xrays.
2. Plan (but not dispatch) the CUDA+DALI cells per CLAUDE.md cross-agent-dispatch
   coordination. Operator dispatches via Modal/Vast.ai on Linux x86_64 GPU.
3. Synthesize whether loader drift dominates, equals, or is dominated by scorer
   drift, given empirical anchor ratios from the device-axis matrix.

OUTPUT:
  experiments/results/cpu_cuda_xray_loader_drift_<UTC>/
    decode_drift.json          PyAV decode custody + cells emitted + plan
    decode_drift.md            human review surface
    shared_inputs/             eval_loader_shared_input_tensor.v1 *.pt files
    rebuild_command.txt

NOT A SCORE CLAIM. Output is tagged `[diagnostic-not-score]`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = REPO_ROOT / "upstream"

SCHEMA = "cpu_cuda_xray_loader_drift.v1"
TOOL = "tools/cpu_cuda_xray_loader_drift.py"
NON_PROMOTABLE_FIELDS: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
}


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _invoke_probe_eval_loader_drift(
    *,
    output_dir: Path,
    upstream_dir: Path,
    batch_size: int,
    video_limit: int,
    max_batches: int,
) -> tuple[Path, dict[str, Any]]:
    """Invoke the canonical probe_eval_loader_drift.py. The probe handles the
    "shared input tensor" custody for us — we just orchestrate the call and
    the synthesis layer.
    """
    shared_dir = output_dir / "shared_inputs"
    json_out = output_dir / "probe_eval_loader_drift_report.json"
    cmd = [
        str(REPO_ROOT / ".venv/bin/python"),
        str(REPO_ROOT / "tools/probe_eval_loader_drift.py"),
        "--video-names-file",
        str(upstream_dir / "public_test_video_names.txt"),
        "--data-dir",
        str(upstream_dir / "videos"),
        "--batch-size",
        str(batch_size),
        "--video-limit",
        str(video_limit),
        "--max-batches",
        str(max_batches),
        "--save-shared-input-dir",
        str(shared_dir),
        "--json-out",
        str(json_out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # probe returns 2 when CUDA/DALI is unavailable but PyAV cells succeed
    # (which is the macOS case). Return code 0 means the full 2x2 ran.
    # Return code 3 means a probe runtime error.
    if result.returncode == 3:
        raise RuntimeError(
            f"probe_eval_loader_drift.py runtime error:\n"
            f"stdout-tail: {result.stdout[-1000:]}\n"
            f"stderr-tail: {result.stderr[-1000:]}"
        )
    if not json_out.exists():
        raise RuntimeError(
            f"probe did not produce report at {json_out}; rc={result.returncode}"
        )
    payload = json.loads(json_out.read_text())
    return json_out, payload


def _synthesize_loader_drift_attribution(
    probe_report: dict[str, Any],
) -> dict[str, Any]:
    """Convert probe rows into a mechanism-attribution summary."""
    comparison_available = probe_report.get("comparison_available")
    rows = probe_report.get("comparison_rows") or []
    if not comparison_available or not rows:
        return {
            "loader_drift_measured": False,
            "loader_drift_unavailable_reason": probe_report.get(
                "comparison_unavailable_reason"
            ),
            "loader_drift_unavailable_class": probe_report.get(
                "comparison_unavailable_class"
            ),
            "interpretation": (
                "Loader-drift cell (DALI/NVDEC vs PyAV/FFmpeg) NOT MEASURED on "
                "this run because CUDA/DALI was unavailable on the local host. "
                "The macOS-side PyAV shared-input tensors are still written to "
                "disk for paired CPU/CUDA xray consumption once a Linux x86_64 "
                "GPU dispatch completes."
            ),
        }
    # Aggregate raw RGB drift across all rows / batches
    max_abs_lsbs: list[float] = []
    mean_abs_lsbs: list[float] = []
    rms_abs_lsbs: list[float] = []
    nonzero_fractions: list[float] = []
    shape_matches: list[bool] = []
    for row in rows:
        comp = row.get("comparison") or {}
        if comp.get("shape_match"):
            shape_matches.append(True)
            max_abs_lsbs.append(float(comp.get("max_abs_lsb", 0.0)))
            mean_abs_lsbs.append(float(comp.get("mean_abs_lsb", 0.0)))
            rms_abs_lsbs.append(float(comp.get("rms_abs_lsb", 0.0)))
            nonzero_fractions.append(float(comp.get("nonzero_fraction", 0.0)))
        else:
            shape_matches.append(False)
    if not max_abs_lsbs:
        return {
            "loader_drift_measured": False,
            "loader_drift_unavailable_reason": "no shape-matched rows",
        }
    overall_max = max(max_abs_lsbs)
    overall_mean = sum(mean_abs_lsbs) / len(mean_abs_lsbs)
    overall_rms = (sum(r * r for r in rms_abs_lsbs) / len(rms_abs_lsbs)) ** 0.5
    overall_nonzero = sum(nonzero_fractions) / len(nonzero_fractions)
    # Interpretation per CLAUDE.md device-axis-matrix mechanism reading.
    # uint8 RGB: a max-abs of 1 LSB is the resolution of the decoder; > 1 LSB
    # means PyAV and DALI/NVDEC do not produce byte-identical decoded frames.
    if overall_max == 0.0:
        loader_class = "byte_identical"
        loader_interpretation = (
            "PyAV and DALI/NVDEC produce byte-identical decoded RGB. Loader "
            "drift contributes ZERO to device-axis score drift. All observed "
            "5× pose ratio in A1 / 0.2× in PR106 r2 originates downstream of "
            "the decode (preprocess YUV6 / scorer-forward kernels / threshold "
            "geometry / score functional)."
        )
    elif overall_max <= 1.0:
        loader_class = "single_lsb_drift"
        loader_interpretation = (
            "PyAV and DALI/NVDEC differ by at most 1 LSB per pixel. This is "
            "decoder rounding-mode drift, not a kernel difference. Whether it "
            "propagates to a 5× pose drift depends on scorer Lipschitz "
            "constant; the layer-drift xrays test this."
        )
    elif overall_max <= 8.0:
        loader_class = "small_multi_lsb_drift"
        loader_interpretation = (
            "PyAV and DALI/NVDEC differ by a few LSBs per pixel. Probably "
            "colorspace conversion / chroma subsampling / interpolation drift. "
            "Could account for a fraction of the observed pose-axis drift."
        )
    else:
        loader_class = "large_drift"
        loader_interpretation = (
            f"PyAV and DALI/NVDEC differ by max {overall_max:.1f} LSB per "
            "pixel. This is a significant decoder-path difference and likely "
            "a dominant contributor to the observed device-axis drift."
        )
    return {
        "loader_drift_measured": True,
        "loader_class": loader_class,
        "max_abs_lsb_across_batches": overall_max,
        "mean_abs_lsb_across_batches": overall_mean,
        "rms_abs_lsb_across_batches": overall_rms,
        "nonzero_fraction_mean": overall_nonzero,
        "num_batches_compared": len(max_abs_lsbs),
        "all_shape_matches": all(shape_matches),
        "interpretation": loader_interpretation,
    }


def _list_shared_input_artifacts(shared_dir: Path) -> list[dict[str, Any]]:
    """Index the shared_inputs/ directory for downstream consumption."""
    rows: list[dict[str, Any]] = []
    if not shared_dir.exists():
        return rows
    for path in sorted(shared_dir.glob("*.pt")):
        rows.append(
            {
                "path": str(path),
                "filename": path.name,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
                "schema": "eval_loader_shared_input_tensor.v1",
            }
        )
    return rows


def _dispatch_plan_for_cuda_dali_cells() -> dict[str, Any]:
    """Emit the canonical Linux x86_64 GPU dispatch contract for the 4 cells
    that require CUDA/DALI (cuda_dali, cuda_av_shared_input, cpu_dali).
    """
    return {
        "schema": "cpu_cuda_xray_loader_drift_dispatch_plan.v1",
        "purpose": (
            "Run the CUDA/DALI cells of the 2x2 loader-drift matrix on a Linux "
            "x86_64 GPU. The local probe_eval_loader_drift.py invocation in "
            "this tool already wrote the PyAV (cpu_av) cells; the remote job "
            "completes the four-cell discriminator."
        ),
        "dispatch_target": "Modal CUDA / Vast.ai 4090 / Lightning T4",
        "lane_id_claim_template": "lane_cpu_cuda_xray_p5_landing_loader_dali_capture",
        "claim_command": [
            ".venv/bin/python",
            "tools/claim_lane_dispatch.py",
            "claim",
            "--lane-id",
            "lane_cpu_cuda_xray_p5_landing_loader_dali_capture",
            "--platform",
            "modal",
            "--status",
            "diagnostic_loader_drift_dali",
            "--notes",
            "P5 loader-drift DALI/NVDEC vs PyAV xray; score_claim=false",
        ],
        "remote_command": [
            ".venv/bin/python",
            "tools/probe_eval_loader_drift.py",
            "--run-forward-cells",
            "--save-shared-input-dir",
            "${REMOTE_OUTPUT_DIR}/shared_inputs",
            "--json-out",
            "${REMOTE_OUTPUT_DIR}/probe_eval_loader_drift_report.json",
        ],
        "estimated_modal_cost_usd_per_run": 0.05,
        "evidence_grade": "diagnostic_not_score",
        "interpretation": (
            "Diagnostic-only Linux x86_64 GPU dispatch. NEVER produces a score "
            "claim and CANNOT promote, kill, or rank any lane."
        ),
        **NON_PROMOTABLE_FIELDS,
    }


def _build_report(
    *,
    probe_report_path: Path,
    probe_report: dict[str, Any],
    shared_input_artifacts: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    attribution = _synthesize_loader_drift_attribution(probe_report)
    plan = _dispatch_plan_for_cuda_dali_cells() if not attribution.get(
        "loader_drift_measured"
    ) else None
    state_basis = json.dumps(
        {
            "label": label,
            "probe_sha256": _sha256_file(probe_report_path),
            "loader_drift_measured": attribution.get("loader_drift_measured"),
        },
        sort_keys=True,
    )
    state_hash = hashlib.sha256(state_basis.encode()).hexdigest()[:16]
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "generated_at_utc": now,
        "from_state_hash": state_hash,
        "label": label,
        "tag": "[diagnostic-not-score]",
        "evidence_grade": "diagnostic_not_score",
        "probe_report_path": str(probe_report_path),
        "probe_report_sha256": _sha256_file(probe_report_path),
        "probe_comparison_available": probe_report.get("comparison_available"),
        "probe_comparison_unavailable_class": probe_report.get(
            "comparison_unavailable_class"
        ),
        "probe_environment": probe_report.get("environment", {}),
        "loader_drift_attribution": attribution,
        "shared_input_artifacts": shared_input_artifacts,
        "shared_input_artifact_count": len(shared_input_artifacts),
        "dispatch_plan_for_cuda_dali": plan,
        **NON_PROMOTABLE_FIELDS,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(
        f"<!-- generated_at: {report['generated_at_utc']}, "
        f"from_state_hash: {report['from_state_hash']} -->"
    )
    lines.append("")
    lines.append("# Loader drift xray — PyAV vs DALI/NVDEC (P5 D3)")
    lines.append("")
    lines.append(
        f"_Schema_: `{report['schema']}` · _Tag_: `{report['tag']}` · "
        f"_Label_: `{report['label']}`"
    )
    lines.append("")
    lines.append("## Probe custody")
    lines.append("")
    lines.append(f"- probe report: `{report['probe_report_path']}`")
    lines.append(f"  - sha256: `{report['probe_report_sha256']}`")
    lines.append(
        f"- shared-input tensors emitted: {report['shared_input_artifact_count']}"
    )
    attribution = report["loader_drift_attribution"]
    lines.append("")
    lines.append("## Loader drift attribution")
    lines.append("")
    if attribution.get("loader_drift_measured"):
        lines.append(f"- **loader class**: `{attribution['loader_class']}`")
        lines.append(
            f"- max-abs LSB across batches: {attribution['max_abs_lsb_across_batches']:.4f}"
        )
        lines.append(
            f"- mean-abs LSB: {attribution['mean_abs_lsb_across_batches']:.4f}"
        )
        lines.append(f"- rms-abs LSB: {attribution['rms_abs_lsb_across_batches']:.4f}")
        lines.append(
            f"- nonzero pixel fraction (mean): "
            f"{attribution['nonzero_fraction_mean']:.4f}"
        )
        lines.append(f"- num batches compared: {attribution['num_batches_compared']}")
        lines.append("")
        lines.append(f"_Interpretation_: {attribution['interpretation']}")
    else:
        lines.append("- **loader drift NOT MEASURED on this run**")
        lines.append(
            f"- unavailable class: `{attribution.get('loader_drift_unavailable_class')}`"
        )
        lines.append(
            f"- unavailable reason: {attribution.get('loader_drift_unavailable_reason')}"
        )
        lines.append("")
        lines.append(f"_Interpretation_: {attribution.get('interpretation')}")
    lines.append("")
    if report.get("dispatch_plan_for_cuda_dali"):
        plan = report["dispatch_plan_for_cuda_dali"]
        lines.append("## Dispatch plan for CUDA/DALI cells")
        lines.append("")
        lines.append(f"- target: `{plan['dispatch_target']}`")
        lines.append(f"- lane id: `{plan['lane_id_claim_template']}`")
        lines.append(f"- estimated cost: ${plan['estimated_modal_cost_usd_per_run']}")
        lines.append("")
        lines.append("```bash")
        lines.append(" ".join(plan["remote_command"]))
        lines.append("```")
        lines.append("")
        lines.append(
            "Run the canonical probe on Linux x86_64 GPU after operator "
            "approval; per CLAUDE.md cross-agent-dispatch coordination, claim "
            "the lane BEFORE dispatching."
        )
    lines.append("")
    lines.append("## Shared-input tensors emitted")
    lines.append("")
    if report["shared_input_artifacts"]:
        lines.append("| Filename | size (KB) | sha256 |")
        lines.append("|---|---:|---|")
        for row in report["shared_input_artifacts"]:
            kb = row["size_bytes"] / 1024
            lines.append(f"| `{row['filename']}` | {kb:.1f} | `{row['sha256'][:16]}…` |")
    else:
        lines.append("(none)")
    lines.append("")
    lines.append(
        "These artifacts are consumable by "
        "`tools/cpu_cuda_xray_segnet_layer_drift.py --shared-input-tensor ...` "
        "and `tools/cpu_cuda_xray_posenet_layer_drift.py --shared-input-tensor "
        "...` for paired layer-drift xrays."
    )
    lines.append("")
    lines.append(
        "Per CLAUDE.md `forbidden_mps_derived_strategic_decision`, this xray is "
        "a mechanism-attribution diagnostic ONLY. No score claim, no kill verdict."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-dir", type=Path, default=UPSTREAM_DIR,
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Probe batch size (default 4).",
    )
    parser.add_argument(
        "--video-limit", type=int, default=1,
        help="Number of videos to sample (default 1).",
    )
    parser.add_argument(
        "--max-batches", type=int, default=1,
        help="Number of probe batches to compare (default 1).",
    )
    parser.add_argument(
        "--label", default="loader_drift_p5_xray",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Default: experiments/results/cpu_cuda_xray_loader_drift_<UTC>",
    )
    args = parser.parse_args(argv)

    timestamp = _utc_stamp()
    output_dir = args.output_dir or (
        REPO_ROOT
        / "experiments"
        / "results"
        / f"cpu_cuda_xray_loader_drift_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_report_path, probe_report = _invoke_probe_eval_loader_drift(
        output_dir=output_dir,
        upstream_dir=args.upstream_dir,
        batch_size=args.batch_size,
        video_limit=args.video_limit,
        max_batches=args.max_batches,
    )
    shared_input_artifacts = _list_shared_input_artifacts(output_dir / "shared_inputs")
    report = _build_report(
        probe_report_path=probe_report_path,
        probe_report=probe_report,
        shared_input_artifacts=shared_input_artifacts,
        label=args.label,
    )
    (output_dir / "decode_drift.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (output_dir / "decode_drift.md").write_text(_render_markdown(report))

    parts = [
        ".venv/bin/python tools/cpu_cuda_xray_loader_drift.py",
        f"--batch-size {args.batch_size}",
        f"--video-limit {args.video_limit}",
        f"--max-batches {args.max_batches}",
        f"--label {args.label}",
    ]
    (output_dir / "rebuild_command.txt").write_text(" \\\n  ".join(parts) + "\n")

    print(f"[xray-loader] wrote {output_dir / 'decode_drift.json'}")
    attribution = report["loader_drift_attribution"]
    if attribution.get("loader_drift_measured"):
        print(
            f"[xray-loader] loader_class={attribution['loader_class']}; "
            f"max_abs_lsb={attribution['max_abs_lsb_across_batches']:.4f}"
        )
    else:
        print(
            f"[xray-loader] loader drift NOT measured on this host; "
            f"shared-input tensors emitted: {report['shared_input_artifact_count']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
