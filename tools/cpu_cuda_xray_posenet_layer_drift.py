#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PoseNet shared-input CPU/CUDA layer-drift xray (handoff P5 deliverable 2).

Per CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval — BOTH CPU AND
CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE": this tool NEVER produces a score
claim. It computes per-layer activation deltas between a CPU forward and a
CUDA forward of the PoseNet (FastViT-T12 backbone → Hydra pose head) scorer
fed the SAME YUV6 input tensor (no decoder drift), and localizes WHERE the
pose-axis drift originates.

This is the SISTER of `cpu_cuda_xray_segnet_layer_drift.py` for the PoseNet
half of the substrate-class boundary hypothesis (see
`.omx/research/device_axis_paired_anchor_matrix_20260511.md`):

- A1 (CPU favored): pose CUDA/CPU ratio = 5.18×
- PR106 r2 (CUDA favored): pose CUDA/CPU ratio = 0.197× (inverse direction!)

Both ratios are ~5× away from unity, while seg ratios are ~1× both ways.
**Pose is the high-leverage device-axis driver.** This tool localizes WHICH
FastViT-T12 RepMixer block or Hydra head layer carries the 5× drift signal.

WHAT IT REVEALS:
  - per-layer L2-relative drift between CPU and CUDA forwards
  - max-abs / mean-abs drift per layer
  - cumulative compounding factor (1+ε)^L through FastViT stages
  - which Hydra head linear layer produces the diverging pose output
  - empirical test of "12 RepMixerBlocks × ε ≈ 0.14 → 4.8× compound" hypothesis

INPUT MODES (sister of SegNet tool):
  --cpu-only   capture CPU record locally; emit dispatch plan for CUDA.
  --paired     pass --cpu-record AND --cuda-record for full drift table.

OUTPUT:
  experiments/results/cpu_cuda_xray_posenet_layer_drift_<UTC>/
    layer_drift.json    typed schema (layer-drift xray v1)
    layer_drift.md      human review surface
    dispatch_plan.json  (cpu-only mode) Linux x86_64 GPU command
    rebuild_command.txt

NOT A SCORE CLAIM. Output tagged `[diagnostic-not-score]`.
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

SCHEMA = "cpu_cuda_xray_posenet_layer_drift.v1"
TOOL = "tools/cpu_cuda_xray_posenet_layer_drift.py"
SCORER = "posenet"
NON_PROMOTABLE_FIELDS: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
}

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
            f"dump_scorer_activations.py failed for CPU PoseNet capture:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    record_path = cpu_dir / "posenet_record.pt"
    if not record_path.exists():
        raise RuntimeError(f"expected CPU posenet record not produced at {record_path}")
    summary = json.loads((cpu_dir / "summary.json").read_text())
    return record_path, summary


def _load_record(path: Path):
    from tac.diagnostics.scorer_introspection import IntrospectionRecord

    return IntrospectionRecord.from_disk(path)


def _layer_drift_to_rows(drift: dict[str, list]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_name, entries in drift.items():
        for entry in entries:
            row = asdict(entry)
            row["layer_name"] = layer_name
            rows.append(row)
    return rows


def _localize_first_divergence(
    rows: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    """For PoseNet, the canonical decision signal is the final 6-dim pose-MSE
    contribution per `modules.PoseNet.compute_distortion`. We track:
    1. First L2-relative-drift exceedance (per-layer architectural drift)
    2. Pose head Hydra-output drift (final-output proxy)
    """
    first_l2_layer: dict[str, Any] | None = None
    hydra_pose_head_layer: dict[str, Any] | None = None
    for row in rows:
        if first_l2_layer is None:
            l2 = row.get("l2_relative_error")
            if l2 is not None and isinstance(l2, float) and l2 == l2 and l2 > threshold:
                first_l2_layer = {
                    "layer_name": row["layer_name"],
                    "module_type": row["module_type"],
                    "l2_relative_error": l2,
                    "max_abs_error": row.get("max_abs_error"),
                    "threshold": threshold,
                }
        # The pose-head final linear is at "hydra.final_layer.pose"
        if row["layer_name"] == "hydra.final_layer.pose":
            hydra_pose_head_layer = {
                "layer_name": row["layer_name"],
                "module_type": row["module_type"],
                "l2_relative_error": row.get("l2_relative_error"),
                "max_abs_error": row.get("max_abs_error"),
                "mean_abs_error": row.get("mean_abs_error"),
                "kl_divergence": row.get("kl_divergence"),
                "rank_top1_disagreement": row.get("rank_top1_disagreement"),
            }
    return {
        "first_l2_relative_exceedance": first_l2_layer,
        "hydra_pose_head_layer": hydra_pose_head_layer,
        "l2_relative_threshold": threshold,
    }


def _fastvit_compounding(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-stage compounding factor (1+ε)^L through FastViT-T12.

    FastViT-T12 layout: stem → stages.0..3, each containing N RepMixer/Attention
    blocks at `vision.stages.{i}.blocks.{j}`. The "12 RepMixer blocks ε^L = 5×"
    hypothesis maps to the per-block L2-relative drift at this granularity.

    Two ε sources:
    - `l2_relative_error` (full-tensor mode, the architecturally correct ε).
    - `fingerprint_only_l2_proxy` (fingerprint mode, an L2-norm proxy that is
      NOT the same as full ε but tracks gross magnitude drift). Used as a
      fallback when full tensors weren't captured. The output records which
      source produced each compound factor so downstream consumers don't
      conflate proxy with full ε.
    """
    from tac.diagnostics.cuda_cpu_drift import compounding_factor

    def pick_eps(row: dict[str, Any]) -> tuple[float | None, str]:
        l2 = row.get("l2_relative_error")
        if isinstance(l2, float) and l2 == l2:  # not NaN
            return float(l2), "full_tensor"
        proxy = row.get("fingerprint_only_l2_proxy")
        if isinstance(proxy, float) and proxy == proxy:
            return float(proxy), "fingerprint_proxy"
        return None, "none"

    # Bucket by FastViT block (vision.stages.{i}.blocks.{j})
    by_block: dict[str, list[float]] = {}
    by_block_eps_source: dict[str, set] = {}
    by_stage: dict[str, list[float]] = {}
    by_stage_eps_source: dict[str, set] = {}
    other: list[float] = []
    for row in rows:
        name = row["layer_name"]
        l2, source = pick_eps(row)
        if l2 is None:
            continue
        if "vision.stages" in name and "blocks" in name:
            parts = name.split(".")
            # vision.stages.{i}.blocks.{j} OR longer (...sub-modules)
            try:
                i_idx = parts.index("stages") + 1
                j_idx = parts.index("blocks") + 1
                stage_key = ".".join(parts[: i_idx + 1])
                # The block itself is registered at depth j_idx + 1
                if len(parts) == j_idx + 1:
                    block_key = ".".join(parts[: j_idx + 1])
                    by_block.setdefault(block_key, []).append(l2)
                    by_block_eps_source.setdefault(block_key, set()).add(source)
                by_stage.setdefault(stage_key, []).append(l2)
                by_stage_eps_source.setdefault(stage_key, set()).add(source)
            except (ValueError, IndexError):
                other.append(l2)
        elif name.startswith("hydra"):
            by_stage.setdefault("hydra", []).append(l2)
            by_stage_eps_source.setdefault("hydra", set()).add(source)
        elif name.startswith("summarizer"):
            by_stage.setdefault("summarizer", []).append(l2)
            by_stage_eps_source.setdefault("summarizer", set()).add(source)
        elif name.startswith("vision.stem"):
            by_stage.setdefault("vision.stem", []).append(l2)
            by_stage_eps_source.setdefault("vision.stem", set()).add(source)
        else:
            other.append(l2)
    by_stage_rows = []
    for stage_key, eps_list in sorted(by_stage.items()):
        by_stage_rows.append(
            {
                "stage_key": stage_key,
                "num_layers": len(eps_list),
                "mean_eps": float(sum(eps_list) / len(eps_list)) if eps_list else 0.0,
                "max_eps": float(max(eps_list)) if eps_list else 0.0,
                "compound_factor": compounding_factor(eps_list),
                "eps_sources": sorted(by_stage_eps_source.get(stage_key, set())),
            }
        )
    by_block_rows = []
    for block_key, eps_list in sorted(by_block.items()):
        by_block_rows.append(
            {
                "block_key": block_key,
                "num_layers": len(eps_list),
                "mean_eps": float(sum(eps_list) / len(eps_list)) if eps_list else 0.0,
                "max_eps": float(max(eps_list)) if eps_list else 0.0,
                "compound_factor": compounding_factor(eps_list),
                "eps_sources": sorted(by_block_eps_source.get(block_key, set())),
            }
        )
    # The whole-FastViT product across all FastViT blocks (the "12 blocks × ε" test):
    fastvit_all: list[float] = []
    fastvit_sources: set = set()
    for row in rows:
        name = row["layer_name"]
        if not ("vision.stages" in name and "blocks" in name):
            continue
        l2, source = pick_eps(row)
        if l2 is None:
            continue
        fastvit_all.append(l2)
        fastvit_sources.add(source)
    return {
        "by_stage": by_stage_rows,
        "by_fastvit_block": by_block_rows,
        "fastvit_all_blocks": {
            "num_blocks_total": len(fastvit_all),
            "mean_eps": float(sum(fastvit_all) / len(fastvit_all)) if fastvit_all else 0.0,
            "max_eps": float(max(fastvit_all)) if fastvit_all else 0.0,
            "compound_factor": compounding_factor(fastvit_all),
            "eps_sources": sorted(fastvit_sources),
        },
        "other_layers_count": len(other),
    }


def _emit_dispatch_plan(
    *,
    output_dir: Path,
    shared_input_tensor: Path | None,
    frame_pair_idx: int,
) -> Path:
    plan: dict[str, Any] = {
        "schema": "cpu_cuda_xray_posenet_dispatch_plan.v1",
        "purpose": (
            "Capture the CUDA PoseNet introspection record paired with the "
            "local CPU record. PoseNet is the high-leverage device-axis driver."
        ),
        "dispatch_target": "Linux x86_64 GPU (Modal CPU+GPU container or Vast.ai 4090)",
        "shared_input_tensor_required": True,
        "shared_input_tensor_path_template": str(shared_input_tensor)
        if shared_input_tensor
        else "${SHARED_INPUT_TENSOR_PT}",
        "frame_pair_idx": frame_pair_idx,
        "lane_id_claim_template": "lane_cpu_cuda_xray_p5_landing_posenet_cuda_capture",
        "claim_command": [
            ".venv/bin/python",
            "tools/claim_lane_dispatch.py",
            "claim",
            "--lane-id",
            "lane_cpu_cuda_xray_p5_landing_posenet_cuda_capture",
            "--platform",
            "modal",
            "--status",
            "diagnostic_cuda_posenet_introspection",
            "--notes",
            "P5 PoseNet shared-input layer-drift xray; score_claim=false",
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
            "posenet",
            "--capture-mode",
            "fingerprint",
        ],
        "post_remote_step": [
            ".venv/bin/python",
            "tools/cpu_cuda_xray_posenet_layer_drift.py",
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
    """Sister of cpu_cuda_xray_segnet_layer_drift._detect_capture_host. Per
    CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-
    COMPLIANT HARDWARE", paired drift on a macOS-CPU record is mixed-substrate."""
    sys_name = platform.system()
    machine = platform.machine()
    is_macos = sys_name == "Darwin"
    is_linux_x86_64 = sys_name == "Linux" and machine in ("x86_64", "AMD64")
    return {
        "platform": platform.platform(),
        "system": sys_name,
        "machine": machine,
        "is_macos_darwin": is_macos,
        "is_linux_x86_64": is_linux_x86_64,
        "contest_compliant_cpu_substrate": is_linux_x86_64,
        "evidence_grade_qualifier": (
            "contest_cpu" if is_linux_x86_64 else "macos_cpu_advisory_only"
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
                "`forbidden_mps_derived_strategic_decision`."
            )
    if drift_rows is not None:
        report["num_layers_compared"] = len(drift_rows)
        report["layer_drift_rows"] = drift_rows
        report["first_divergence"] = _localize_first_divergence(drift_rows, threshold)
        report["fastvit_compounding"] = _fastvit_compounding(drift_rows)
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
    lines.append("# PoseNet shared-input CPU/CUDA layer-drift xray (P5 D2)")
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
    if first.get("hydra_pose_head_layer"):
        h = first["hydra_pose_head_layer"]
        lines.append(
            f"- **Hydra pose head output**: `{h['layer_name']}`; "
            f"L2_rel={h.get('l2_relative_error')}; "
            f"max_abs={h.get('max_abs_error')}"
        )
    lines.append("")
    fc = report["fastvit_compounding"]
    lines.append("## FastViT block-level compounding (the 5× pose hypothesis)")
    lines.append("")
    fb = fc["fastvit_all_blocks"]
    lines.append(
        f"- **all FastViT blocks**: n_layers={fb['num_blocks_total']}; "
        f"mean ε={fb['mean_eps']:.4e}; max ε={fb['max_eps']:.4e}; "
        f"**(1+ε)^L = {fb['compound_factor']:.4f}×**"
    )
    lines.append("")
    lines.append("### Per FastViT stage")
    lines.append("")
    lines.append("| Stage | #layers | mean ε | max ε | (1+ε)^L |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in fc["by_stage"]:
        lines.append(
            f"| `{row['stage_key']}` | {row['num_layers']} | "
            f"{row['mean_eps']:.4e} | {row['max_eps']:.4e} | "
            f"{row['compound_factor']:.4f} |"
        )
    lines.append("")
    lines.append("### Per FastViT block (RepMixer/Attention)")
    lines.append("")
    lines.append("| Block | #layers | mean ε | max ε | (1+ε)^L |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in fc["by_fastvit_block"]:
        lines.append(
            f"| `{row['block_key']}` | {row['num_layers']} | "
            f"{row['mean_eps']:.4e} | {row['max_eps']:.4e} | "
            f"{row['compound_factor']:.4f} |"
        )
    lines.append("")
    lines.append(
        "_Interpretation_: a per-block compound factor close to the empirical "
        "5× pose ratio (A1) or 0.2× (PR106 r2) is evidence the device-axis drift "
        "originates from per-block FastViT numerics. A flat factor means the "
        "drift comes from somewhere else (loader, preprocess, Hydra head, or "
        "score-functional non-linearity)."
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
    mode_group.add_argument("--cpu-only", action="store_true")
    mode_group.add_argument("--paired", action="store_true")
    parser.add_argument("--cpu-record", type=Path, default=None)
    parser.add_argument("--cuda-record", type=Path, default=None)
    parser.add_argument("--shared-input-tensor", type=Path, default=None)
    parser.add_argument("--frame-pair-idx", type=int, default=0)
    parser.add_argument(
        "--l2-relative-drift-threshold", type=float, default=DEFAULT_LAYER_DRIFT_THRESHOLD,
    )
    parser.add_argument("--label", default="posenet_p5_xray")
    parser.add_argument("--upstream-dir", type=Path, default=UPSTREAM_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    timestamp = _utc_stamp()
    output_dir = args.output_dir or (
        REPO_ROOT
        / "experiments"
        / "results"
        / f"cpu_cuda_xray_posenet_layer_drift_{timestamp}"
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

    parts = [
        ".venv/bin/python tools/cpu_cuda_xray_posenet_layer_drift.py",
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

    print(f"[xray-posenet] mode={mode}")
    print(f"[xray-posenet] wrote {output_dir / 'layer_drift.json'}")
    if drift_rows is not None:
        fb = report["fastvit_compounding"]["fastvit_all_blocks"]
        print(
            f"[xray-posenet] FastViT blocks compound factor: "
            f"{fb['compound_factor']:.4f}× (n={fb['num_blocks_total']}, "
            f"mean ε={fb['mean_eps']:.4e})"
        )
        first = report["first_divergence"]
        if first.get("first_l2_relative_exceedance"):
            b = first["first_l2_relative_exceedance"]
            print(
                f"[xray-posenet] first L2-drift > threshold: {b['layer_name']} "
                f"(L2_rel={b['l2_relative_error']:.4e})"
            )
    else:
        print(f"[xray-posenet] CUDA capture pending; see {output_dir / 'dispatch_plan.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
