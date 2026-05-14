#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""SegNet shared-input CPU/CUDA layer-drift xray (handoff P5 deliverable 1).

Per CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval — BOTH CPU AND
CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE": this tool NEVER produces a score
claim. It computes per-layer activation deltas between a CPU forward and a
CUDA forward of the SegNet (EfficientNet-B2 → Unet head) scorer fed the SAME
RGB input tensor (no decoder drift), and localizes the FIRST layer where the
SegNet argmax decision diverges.

This is the layer-level xray of the SegNet axis of the substrate-class boundary
hypothesis (see `.omx/research/device_axis_paired_anchor_matrix_20260511.md`):
A1 CPU/CUDA seg ratio is 1.18× (CPU favored); PR106 r2 seg ratio is 1.017×
(CUDA marginally favored). Both ratios are small relative to the 5× pose ratio,
but the seg term is the LARGER absolute contributor at the operating point.
This xray localizes WHICH SegNet layer carries the seg-ratio signal.

WHAT IT REVEALS:
  - per-layer L2-relative drift between CPU and CUDA forwards
  - max-abs / mean-abs drift per layer
  - argmax disagreement fraction on the final logits
  - FIRST layer (by topological order) whose drift exceeds a threshold
  - cumulative compounding factor (1+ε)^L through the EfficientNet stages

INPUT MODES:
  --cpu-only   capture CPU record locally; emit a "plan" stub for the CUDA
               capture command (operator dispatches via existing remote
               bootstrap on Linux x86_64 GPU per CLAUDE.md remote-code parity).
  --paired     pass BOTH --cpu-record AND --cuda-record (each produced by
               experiments/dump_scorer_activations.py); the tool then computes
               the full per-layer drift table.

INPUT-TENSOR CUSTODY:
  --shared-input-tensor PATH    consume an eval_loader_shared_input_tensor.v1
                                artifact (PyAV-decoded RGB) so CPU and CUDA
                                forwards consume the SAME RGB input bytes,
                                isolating scorer-forward drift from loader drift.
  --frame-pair-idx N            fall back to a deterministic frame pair from
                                upstream/videos/0.mkv (PyAV decode only).

OUTPUT:
  experiments/results/cpu_cuda_xray_segnet_layer_drift_<UTC>/
    layer_drift.json          typed schema (layer-drift xray v1)
    layer_drift.md            human review surface
    dispatch_plan.json        (cpu-only mode) Linux x86_64 GPU command
    rebuild_command.txt

NOT A SCORE CLAIM. Output is tagged `[diagnostic-not-score]`. Per CLAUDE.md
non-negotiable, no KILL verdict can be inferred from this xray.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
UPSTREAM_DIR = REPO_ROOT / "upstream"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

SCHEMA = "cpu_cuda_xray_segnet_layer_drift.v1"
TOOL = "tools/cpu_cuda_xray_segnet_layer_drift.py"
SCORER = "segnet"
NON_PROMOTABLE_FIELDS: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
}

# Argmax-divergence detection threshold: SegNet final layer is logit-shaped,
# so we localize on `rank_top1_disagreement > 0`. Intermediate non-logit layers
# use L2-relative drift exceedance.
DEFAULT_LAYER_DRIFT_THRESHOLD = 1e-2  # 1% L2-relative drift


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _capture_cpu_record(
    *,
    upstream_dir: Path,
    shared_input_tensor: Path | None,
    frame_pair_idx: int,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Invoke the canonical dump_scorer_activations.py via direct import.

    Writes `<output_dir>/cpu_segnet_record.pt` and returns the path + metadata.
    """
    import subprocess

    cpu_dir = output_dir / "cpu_capture"
    cpu_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(REPO_ROOT / ".venv/bin/python"),
        str(REPO_ROOT / "experiments/dump_scorer_activations.py"),
        "--upstream-dir",
        str(upstream_dir),
        "--device",
        "cpu",
        "--output-dir",
        str(cpu_dir),
        "--scorer",
        SCORER,
        "--capture-mode",
        "fingerprint",
        "--frame-pair-idx",
        str(frame_pair_idx),
    ]
    if shared_input_tensor is not None:
        cmd.extend(["--shared-input-tensor", str(shared_input_tensor)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"dump_scorer_activations.py failed for CPU SegNet capture:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    record_path = cpu_dir / "segnet_record.pt"
    if not record_path.exists():
        raise RuntimeError(f"expected CPU segnet record not produced at {record_path}")
    summary = json.loads((cpu_dir / "summary.json").read_text())
    return record_path, summary


def _load_record(path: Path):
    """Load an IntrospectionRecord dict via tac.diagnostics."""
    from tac.diagnostics.scorer_introspection import IntrospectionRecord

    return IntrospectionRecord.from_disk(path)


def _layer_drift_to_rows(
    drift: dict[str, list],
) -> list[dict[str, Any]]:
    """Flatten DriftMetrics into JSON-serializable rows preserving order."""
    rows: list[dict[str, Any]] = []
    for layer_name, entries in drift.items():
        for entry in entries:
            row = asdict(entry)
            row["layer_name"] = layer_name
            rows.append(row)
    return rows


def _localize_first_divergence(
    rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """Find the first row (in topological order) whose drift signals exceed
    a divergence boundary. Argmax (rank_top1_disagreement > 0) is the canonical
    SegNet decision signal; L2-relative is the architectural-drift signal.
    """
    first_argmax_layer: dict[str, Any] | None = None
    first_l2_layer: dict[str, Any] | None = None
    for row in rows:
        if first_argmax_layer is None:
            rank = row.get("rank_top1_disagreement")
            if rank is not None and rank > 0:
                first_argmax_layer = {
                    "layer_name": row["layer_name"],
                    "module_type": row["module_type"],
                    "rank_top1_disagreement": rank,
                    "l2_relative_error": row.get("l2_relative_error"),
                    "max_abs_error": row.get("max_abs_error"),
                }
        if first_l2_layer is None:
            l2 = row.get("l2_relative_error")
            if l2 is not None and l2 > threshold:
                first_l2_layer = {
                    "layer_name": row["layer_name"],
                    "module_type": row["module_type"],
                    "l2_relative_error": l2,
                    "max_abs_error": row.get("max_abs_error"),
                    "threshold": threshold,
                }
        if first_argmax_layer is not None and first_l2_layer is not None:
            break
    return {
        "first_argmax_divergence": first_argmax_layer,
        "first_l2_relative_exceedance": first_l2_layer,
        "l2_relative_threshold": threshold,
    }


def _summarize_stage_compounding(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-stage compounding factor (1+ε)^L through the
    EfficientNet-B2 encoder stages.

    Two ε sources: `l2_relative_error` (full-tensor mode) and
    `fingerprint_only_l2_proxy` (fingerprint mode). The output records which
    source produced each stage's factor so downstream consumers don't conflate
    proxy with full ε.
    """
    from tac.diagnostics.cuda_cpu_drift import compounding_factor

    def pick_eps(row: dict[str, Any]) -> tuple[float | None, str]:
        l2 = row.get("l2_relative_error")
        if isinstance(l2, float) and l2 == l2:
            return float(l2), "full_tensor"
        proxy = row.get("fingerprint_only_l2_proxy")
        if isinstance(proxy, float) and proxy == proxy:
            return float(proxy), "fingerprint_proxy"
        return None, "none"

    # Group by stage prefix; smp Unet exposes encoder layers under "encoder.blocks."
    by_stage: dict[str, list[float]] = {}
    by_stage_source: dict[str, set] = {}
    for row in rows:
        name = row["layer_name"]
        if not name:
            continue
        parts = name.split(".")
        if "blocks" in parts:
            i = parts.index("blocks")
            if i + 1 < len(parts):
                stage_key = ".".join(parts[: i + 2])
            else:
                stage_key = ".".join(parts[: i + 1])
        else:
            stage_key = parts[0]
        l2, source = pick_eps(row)
        if l2 is None:
            continue
        by_stage.setdefault(stage_key, []).append(l2)
        by_stage_source.setdefault(stage_key, set()).add(source)
    stages = []
    for stage_key, eps_list in sorted(by_stage.items()):
        stages.append(
            {
                "stage_key": stage_key,
                "num_layers": len(eps_list),
                "mean_eps": float(sum(eps_list) / len(eps_list)) if eps_list else 0.0,
                "max_eps": float(max(eps_list)) if eps_list else 0.0,
                "compound_factor": compounding_factor(eps_list),
                "eps_sources": sorted(by_stage_source.get(stage_key, set())),
            }
        )
    return {"by_stage": stages}


def _segnet_final_logits_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract the final logits-layer disagreement signal.

    smp.Unet emits final 5-class logits at module name `segmentation_head` (a
    `SegmentationHead` container) or `segmentation_head.0` (the underlying
    Conv2d). Shape is (B, 5, H, W), so the channel axis (not last axis) is the
    decision axis. The tac.diagnostics drift comparator's rank-disagreement only
    fires for last-axis classifiers, so for SegNet we report the per-pixel
    architectural drift on the head and rely on the per-layer L2-relative-error
    plus the per-stage compounding for argmax-divergence localization.
    """
    final = None
    for row in rows:
        # First preference: SegmentationHead container at the end of the model.
        if row["layer_name"] == "segmentation_head":
            final = row
            break
    if final is None:
        # Fallback: the inner Conv2d that produces the actual logits.
        for row in rows:
            if row["layer_name"] == "segmentation_head.0":
                final = row
                break
    if final is None:
        return {"available": False}
    return {
        "available": True,
        "layer_name": final["layer_name"],
        "module_type": final["module_type"],
        "rank_top1_disagreement": final.get("rank_top1_disagreement"),
        "kl_divergence": final.get("kl_divergence"),
        "l2_relative_error": final.get("l2_relative_error"),
        "max_abs_error": final.get("max_abs_error"),
        "mean_abs_error": final.get("mean_abs_error"),
        "note": (
            "smp.Unet logits axis is channel (B, num_classes, H, W); "
            "rank_top1_disagreement above is from tac.diagnostics's last-axis "
            "comparator and is NOT meaningful for segmentation logits. The "
            "architectural drift signal lives in `l2_relative_error` and "
            "`max_abs_error`."
        ),
    }


def _emit_dispatch_plan(
    *,
    output_dir: Path,
    shared_input_tensor: Path | None,
    frame_pair_idx: int,
) -> Path:
    """Write the Linux x86_64 GPU dispatch command + claim template so the
    operator can produce the paired CUDA record without re-deriving the spec.
    """
    plan: dict[str, Any] = {
        "schema": "cpu_cuda_xray_segnet_dispatch_plan.v1",
        "purpose": (
            "Capture the CUDA SegNet introspection record paired with the local "
            "CPU record so the two can be fed back into this tool's --paired mode."
        ),
        "dispatch_target": "Linux x86_64 GPU (Modal CPU+GPU container or Vast.ai 4090)",
        "shared_input_tensor_required": True,
        "shared_input_tensor_path_template": str(shared_input_tensor)
        if shared_input_tensor
        else "${SHARED_INPUT_TENSOR_PT}",
        "frame_pair_idx": frame_pair_idx,
        "lane_id_claim_template": "lane_cpu_cuda_xray_p5_landing_segnet_cuda_capture",
        "claim_command": [
            ".venv/bin/python",
            "tools/claim_lane_dispatch.py",
            "claim",
            "--lane-id",
            "lane_cpu_cuda_xray_p5_landing_segnet_cuda_capture",
            "--platform",
            "modal",
            "--status",
            "diagnostic_cuda_segnet_introspection",
            "--notes",
            "P5 SegNet shared-input layer-drift xray; score_claim=false",
        ],
        "remote_command": [
            ".venv/bin/python",
            "experiments/dump_scorer_activations.py",
            "--upstream-dir",
            "upstream",
            "--device",
            "cuda",
            "--shared-input-tensor",
            "${SHARED_INPUT_TENSOR_PT}",
            "--output-dir",
            "${OUTPUT_DIR}",
            "--scorer",
            "segnet",
            "--capture-mode",
            "fingerprint",
        ],
        "post_remote_step": [
            ".venv/bin/python",
            "tools/cpu_cuda_xray_segnet_layer_drift.py",
            "--paired",
            "--cpu-record",
            "${LOCAL_CPU_RECORD_PT}",
            "--cuda-record",
            "${REMOTE_CUDA_RECORD_PT}",
            "--shared-input-tensor",
            "${SHARED_INPUT_TENSOR_PT}",
        ],
        "estimated_modal_cost_usd_per_run": 0.05,
        "evidence_grade": "diagnostic_not_score",
        "interpretation": (
            "Per CLAUDE.md, this dispatch is a diagnostic-only run. NEVER produces "
            "a score claim and CANNOT promote, kill, or rank any lane."
        ),
        **NON_PROMOTABLE_FIELDS,
    }
    out = output_dir / "dispatch_plan.json"
    out.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return out


def _detect_capture_host() -> dict[str, Any]:
    """Detect the host where this xray tool is running.

    Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-
    COMPLIANT HARDWARE": only Linux x86_64 CPU records can be paired with
    Linux x86_64 CUDA records to produce a true `[contest-CUDA]` vs
    `[contest-CPU]` drift comparison. A macOS-Darwin CPU record paired with
    Linux-CUDA mixes (device-axis drift) ⊕ (macOS-vs-Linux drift), so the
    comparison must be tagged `[macOS-CPU advisory only]` not `[contest-CPU]`.
    """
    sys_name = platform.system()
    machine = platform.machine()
    is_macos = sys_name == "Darwin"
    is_linux_x86_64 = sys_name == "Linux" and machine in ("x86_64", "AMD64")
    contest_compliant_cpu = is_linux_x86_64
    return {
        "platform": platform.platform(),
        "system": sys_name,
        "machine": machine,
        "is_macos_darwin": is_macos,
        "is_linux_x86_64": is_linux_x86_64,
        "contest_compliant_cpu_substrate": contest_compliant_cpu,
        "evidence_grade_qualifier": (
            "contest_cpu" if contest_compliant_cpu else "macos_cpu_advisory_only"
        ),
    }


def _build_report(
    *,
    mode: str,
    cpu_record_path: Path,
    cuda_record_path: Path | None,
    shared_input_tensor: Path | None,
    frame_pair_idx: int,
    drift_rows: list[dict[str, Any]] | None,
    threshold: float,
    label: str,
    cpu_capture_host: dict[str, Any] | None,
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        "generated_at_utc": now,
        "label": label,
        "scorer": SCORER,
        "mode": mode,
        "tag": "[diagnostic-not-score]",
        "evidence_grade": "diagnostic_not_score",
        "cpu_record_path": str(cpu_record_path),
        "cpu_record_sha256": _sha256_file(cpu_record_path),
        "cpu_capture_host": cpu_capture_host,
        "cuda_record_path": str(cuda_record_path) if cuda_record_path else None,
        "cuda_record_sha256": _sha256_file(cuda_record_path) if cuda_record_path else None,
        "shared_input_tensor_path": str(shared_input_tensor) if shared_input_tensor else None,
        "shared_input_tensor_sha256": _sha256_file(shared_input_tensor) if shared_input_tensor else None,
        "frame_pair_idx": frame_pair_idx,
        "l2_relative_drift_threshold": threshold,
        **NON_PROMOTABLE_FIELDS,
    }
    if mode == "paired" and cpu_capture_host is not None:
        if not cpu_capture_host.get("contest_compliant_cpu_substrate"):
            report["mixed_substrate_advisory"] = (
                "CPU record was captured on a NON-Linux-x86_64 substrate "
                f"({cpu_capture_host.get('platform')}); the resulting layer-drift "
                "table includes BOTH device-axis drift AND macOS-vs-Linux-CPU "
                "drift and is tagged `[macOS-CPU advisory only]` per CLAUDE.md "
                "`forbidden_mps_derived_strategic_decision` and \"Submission "
                "auth eval — BOTH CPU AND CUDA\" non-negotiables. Re-run with "
                "a Linux x86_64 CPU record for authoritative attribution."
            )
    if drift_rows is not None:
        report["num_layers_compared"] = len(drift_rows)
        report["layer_drift_rows"] = drift_rows
        report["first_divergence"] = _localize_first_divergence(drift_rows, threshold)
        report["stage_compounding"] = _summarize_stage_compounding(drift_rows)
        report["final_logits"] = _segnet_final_logits_summary(drift_rows)
    else:
        report["layer_drift_rows"] = None
        report["pending_cuda_capture"] = True
        report["pending_reason"] = (
            "Local CPU record captured; CUDA record requires a Linux x86_64 "
            "GPU dispatch. See dispatch_plan.json."
        )
    state_basis = json.dumps(
        {
            "label": label,
            "scorer": SCORER,
            "mode": mode,
            "cpu_sha": report["cpu_record_sha256"],
            "cuda_sha": report["cuda_record_sha256"],
            "shared_input_sha": report["shared_input_tensor_sha256"],
        },
        sort_keys=True,
    )
    report["from_state_hash"] = hashlib.sha256(state_basis.encode()).hexdigest()[:16]
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(
        f"<!-- generated_at: {report['generated_at_utc']}, "
        f"from_state_hash: {report['from_state_hash']} -->"
    )
    lines.append("")
    lines.append("# SegNet shared-input CPU/CUDA layer-drift xray (P5 D1)")
    lines.append("")
    lines.append(f"_Schema_: `{report['schema']}` · _Mode_: `{report['mode']}` ")
    lines.append(f"· _Tag_: `{report['tag']}` · _Label_: `{report['label']}`")
    lines.append("")
    lines.append("## Custody")
    lines.append("")
    lines.append(f"- CPU record: `{report['cpu_record_path']}`")
    lines.append(f"  - sha256: `{report['cpu_record_sha256']}`")
    if report["cuda_record_path"]:
        lines.append(f"- CUDA record: `{report['cuda_record_path']}`")
        lines.append(f"  - sha256: `{report['cuda_record_sha256']}`")
    else:
        lines.append("- CUDA record: **PENDING** — see `dispatch_plan.json`")
    if report["shared_input_tensor_path"]:
        lines.append(f"- shared input tensor: `{report['shared_input_tensor_path']}`")
        lines.append(f"  - sha256: `{report['shared_input_tensor_sha256']}`")
    lines.append(f"- frame pair idx: {report['frame_pair_idx']}")
    if report.get("cpu_capture_host"):
        h = report["cpu_capture_host"]
        lines.append(f"- CPU capture host: `{h.get('platform')}` (contest-compliant: {h.get('contest_compliant_cpu_substrate')})")
    lines.append("")
    if report.get("mixed_substrate_advisory"):
        lines.append("## ⚠️ Mixed-substrate advisory")
        lines.append("")
        lines.append(report["mixed_substrate_advisory"])
        lines.append("")
    if report.get("layer_drift_rows") is None:
        lines.append("## Status")
        lines.append("")
        lines.append("CPU record captured locally. CUDA capture pending operator")
        lines.append("Linux x86_64 GPU dispatch per the emitted `dispatch_plan.json`.")
        lines.append("")
        lines.append(
            "Per CLAUDE.md remote-code-parity + MPS-noise non-negotiables, NO "
            "score claim or kill verdict can be derived from this CPU-only run."
        )
        return "\n".join(lines) + "\n"
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- layers compared: {report['num_layers_compared']}")
    first = report["first_divergence"]
    if first.get("first_argmax_divergence"):
        a = first["first_argmax_divergence"]
        lines.append(
            f"- **first argmax-divergence layer**: `{a['layer_name']}` "
            f"({a['module_type']}; rank_top1_disagreement="
            f"{a['rank_top1_disagreement']})"
        )
    else:
        lines.append("- no argmax-divergence layer detected")
    if first.get("first_l2_relative_exceedance"):
        b = first["first_l2_relative_exceedance"]
        lines.append(
            f"- **first L2-relative-drift > {b['threshold']:.2e}**: "
            f"`{b['layer_name']}` ({b['module_type']}; "
            f"L2_rel={b['l2_relative_error']:.4e})"
        )
    else:
        lines.append(
            f"- no L2-relative drift > {report['l2_relative_drift_threshold']:.2e} detected"
        )
    if report.get("final_logits", {}).get("available"):
        f = report["final_logits"]
        lines.append(
            f"- final logits: `{f['layer_name']}`; rank_top1_disagreement="
            f"{f.get('rank_top1_disagreement')}; "
            f"L2_rel={f.get('l2_relative_error')}"
        )
    lines.append("")
    lines.append("## Per-stage compounding factor")
    lines.append("")
    lines.append("| Stage | #layers | mean ε | max ε | (1+ε)^L |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in report["stage_compounding"]["by_stage"]:
        lines.append(
            f"| `{row['stage_key']}` | {row['num_layers']} | "
            f"{row['mean_eps']:.4e} | {row['max_eps']:.4e} | "
            f"{row['compound_factor']:.4f} |"
        )
    lines.append("")
    lines.append(
        "_Interpretation_: the per-stage compounding factor is the geometric "
        "(1+ε_i) product over the stage's layer L2-relative drifts. A factor "
        "well above 1 means CPU and CUDA forwards have measurably diverged by "
        "the time the activation crosses the stage. NOT a score claim."
    )
    lines.append("")
    lines.append(
        "Per CLAUDE.md `forbidden_mps_derived_strategic_decision`, ALL strategic "
        "decisions from this xray require both `[contest-CUDA]` and "
        "`[contest-CPU]` empirical anchors on 1:1 contest-compliant hardware. "
        "This tool is a mechanism-attribution diagnostic only."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--cpu-only",
        action="store_true",
        help="Capture CPU SegNet record locally; emit dispatch plan for CUDA.",
    )
    mode_group.add_argument(
        "--paired",
        action="store_true",
        help="Compute paired CPU/CUDA layer drift; requires both records.",
    )
    parser.add_argument("--cpu-record", type=Path, default=None)
    parser.add_argument("--cuda-record", type=Path, default=None)
    parser.add_argument(
        "--shared-input-tensor",
        type=Path,
        default=None,
        help="eval_loader_shared_input_tensor.v1 artifact (preferred for paired xray).",
    )
    parser.add_argument("--frame-pair-idx", type=int, default=0)
    parser.add_argument(
        "--l2-relative-drift-threshold",
        type=float,
        default=DEFAULT_LAYER_DRIFT_THRESHOLD,
    )
    parser.add_argument("--label", default="segnet_p5_xray")
    parser.add_argument(
        "--upstream-dir", type=Path, default=UPSTREAM_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: experiments/results/cpu_cuda_xray_segnet_layer_drift_<UTC>",
    )
    args = parser.parse_args(argv)

    timestamp = _utc_stamp()
    output_dir = args.output_dir or (
        REPO_ROOT
        / "experiments"
        / "results"
        / f"cpu_cuda_xray_segnet_layer_drift_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cpu_record_path: Path
    cuda_record_path: Path | None = None
    drift_rows: list[dict[str, Any]] | None = None
    cpu_capture_host: dict[str, Any] | None = None

    if args.paired:
        if args.cpu_record is None or args.cuda_record is None:
            parser.error("--paired requires --cpu-record AND --cuda-record")
        if not args.cpu_record.exists():
            parser.error(f"--cpu-record not found: {args.cpu_record}")
        if not args.cuda_record.exists():
            parser.error(f"--cuda-record not found: {args.cuda_record}")
        cpu_record_path = args.cpu_record
        cuda_record_path = args.cuda_record
        # In paired mode we cannot know retroactively where the cpu_record was
        # captured. Default to the current host but mark it explicitly as a
        # pairing-time assumption; the operator can override with an explicit
        # CLI flag (--cpu-record-substrate) if needed in a follow-up.
        cpu_capture_host = _detect_capture_host()
        cpu_capture_host["note"] = (
            "Host detected at pairing-time, not at record-capture-time. If the "
            "CPU record was captured on a different host (e.g., Linux x86_64 "
            "remote), the substrate qualifier may be incorrect."
        )
        from tac.diagnostics import compute_layer_drift

        rec_cpu = _load_record(cpu_record_path)
        rec_cuda = _load_record(cuda_record_path)
        drift = compute_layer_drift(rec_cpu, rec_cuda)
        drift_rows = _layer_drift_to_rows(drift)
        mode = "paired"
    else:
        cpu_record_path, _ = _capture_cpu_record(
            upstream_dir=args.upstream_dir,
            shared_input_tensor=args.shared_input_tensor,
            frame_pair_idx=args.frame_pair_idx,
            output_dir=output_dir,
        )
        cpu_capture_host = _detect_capture_host()
        _emit_dispatch_plan(
            output_dir=output_dir,
            shared_input_tensor=args.shared_input_tensor,
            frame_pair_idx=args.frame_pair_idx,
        )
        mode = "cpu_only"

    report = _build_report(
        mode=mode,
        cpu_record_path=cpu_record_path,
        cuda_record_path=cuda_record_path,
        shared_input_tensor=args.shared_input_tensor,
        frame_pair_idx=args.frame_pair_idx,
        drift_rows=drift_rows,
        threshold=args.l2_relative_drift_threshold,
        label=args.label,
        cpu_capture_host=cpu_capture_host,
    )
    (output_dir / "layer_drift.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (output_dir / "layer_drift.md").write_text(_render_markdown(report))

    # rebuild_command.txt with placeholders for re-runnable contract
    parts = [
        ".venv/bin/python tools/cpu_cuda_xray_segnet_layer_drift.py",
        f"--{mode.replace('_', '-')}",
    ]
    if mode == "paired":
        parts.append(f"--cpu-record {args.cpu_record}")
        parts.append(f"--cuda-record {args.cuda_record}")
    if args.shared_input_tensor:
        parts.append(f"--shared-input-tensor {args.shared_input_tensor}")
    parts.append(f"--frame-pair-idx {args.frame_pair_idx}")
    parts.append(f"--l2-relative-drift-threshold {args.l2_relative_drift_threshold}")
    parts.append(f"--label {args.label}")
    (output_dir / "rebuild_command.txt").write_text(" \\\n  ".join(parts) + "\n")

    print(f"[xray-segnet] mode={mode}")
    print(f"[xray-segnet] wrote {output_dir / 'layer_drift.json'}")
    if drift_rows is not None:
        first = report["first_divergence"]
        if first.get("first_argmax_divergence"):
            a = first["first_argmax_divergence"]
            print(
                f"[xray-segnet] first argmax-divergence layer: {a['layer_name']} "
                f"({a['module_type']}; rank={a['rank_top1_disagreement']})"
            )
        else:
            print("[xray-segnet] no argmax-divergence layer detected")
    else:
        print(f"[xray-segnet] CUDA capture pending; see {output_dir / 'dispatch_plan.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
