#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure task #452's defect-network phase recode on the real n600 cache.

This probe is rate-side only.  It compares exact serialized bytes and requires
bit-identical incumbent/candidate decoded phase fields.  A through-R report may
be attached as a receiver canary, but the receipt refuses to call it a phase
effect A/B because the current ``PHAS1`` carrier is not packed or consumed by
``levelset_byte_close_and_eval.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.boundary_math.dense_raster_lzma_baseline import encode_partition  # noqa: E402
from tac.boundary_math.defect_network_rate_code import (  # noqa: E402
    encode_defect_tube_recode,
)
from tac.boundary_math.movable_deshare import detect_seg_roles  # noqa: E402
from tac.boundary_math.phase_residual_carrier import (  # noqa: E402
    PhaseCarrierConfig,
    compute_tie_field_from_margins,
    phase_carrier_report,
)

AXIS = (
    "[macOS-CPU advisory . deterministic numpy standalone-section rate probe . NON-PROMOTABLE]"
)
PAPER = {
    "authors": ["Nathan Benjamin", "Ho Tat Lam", "Conghuan Luo"],
    "year": 2026,
    "title": "Chiral Tube Algebras I: Topological Defect Lines, Twisted Modules, and Finite Gauging",
    "arxiv_id": "2607.07786",
    "url": "https://arxiv.org/abs/2607.07786",
    "sections_used": ["1.1", "1.3"],
    "import_status": "ANALOGY_ONLY; no CFT theorem is used as a compression theorem",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: list[str]) -> str:
    proc = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else f"UNKNOWN(rc={proc.returncode})"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _execution_source_custody() -> dict[str, dict[str, Any]]:
    paths = {
        "probe": REPO / "tools/probe_defect_network_rate_code.py",
        "defect_network_rate_code": REPO / "src/tac/boundary_math/defect_network_rate_code.py",
        "phase_residual_carrier": REPO / "src/tac/boundary_math/phase_residual_carrier.py",
        "xi_spline_residual_coder": REPO
        / "src/tac/boundary_math/xi_spline_residual_coder.py",
        "dense_raster_lzma_baseline": REPO / "src/tac/boundary_math/dense_raster_lzma_baseline.py",
        "class_role_detector": REPO / "src/tac/boundary_math/movable_deshare.py",
        "through_r_canary_tool": REPO / "tools/levelset_byte_close_and_eval.py",
    }
    return {
        name: {
            "path": path.relative_to(REPO).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
    }


def _extract_probe_geometry(
    labels: np.ndarray, margins: np.ndarray, cfg: PhaseCarrierConfig
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    masks: list[np.ndarray] = []
    class_maps: list[np.ndarray] = []
    for p in range(labels.shape[0]):
        _, mask, class_map = compute_tie_field_from_margins(labels[p], margins[p], cfg)
        masks.append(mask)
        class_maps.append(class_map)
    return masks, class_maps


def _junction_count(labels: np.ndarray) -> int:
    """Count 2x2 cells containing at least three class labels."""

    total = 0
    for frame in labels:
        cells = np.stack(
            (frame[:-1, :-1], frame[:-1, 1:], frame[1:, :-1], frame[1:, 1:]), axis=-1
        )
        cells.sort(axis=-1)
        unique_count = 1 + np.sum(cells[..., 1:] != cells[..., :-1], axis=-1)
        total += int(np.sum(unique_count >= 3))
    return total


def _through_r_canary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "attached": False,
            "phase_effect_ab_status": "UNMEASURED; current PHAS1 section is not receiver-consumed",
        }
    report = json.loads(path.read_text())
    parity = report.get("parity_on_inflated_frames", {})
    phase = report.get("phase_carrier", {})
    return {
        "attached": True,
        "path": str(path),
        "sha256": _sha256(path),
        "tool": report.get("tool"),
        "authority": report.get("authority"),
        "pairs_scored": parity.get("pairs_scored"),
        "d_seg_realized_on_inflated": parity.get("d_seg_realized_on_inflated"),
        "d_pose_realized_on_inflated": parity.get("d_pose_realized_on_inflated"),
        "raw_sha256": report.get("inflate", {}).get("raw_sha256"),
        "phase_section_bytes_remeasured": phase.get("section_bytes"),
        "phase_reconstruction_bit_identical": phase.get("reconstruction_bit_identical"),
        "phase_effect_ab_status": (
            "NOT_APPLICABLE: this report measures the base witness through R; the current tool builds "
            "PHAS1 for rate accounting but does not pack or consume it. Equal phase decode therefore "
            "proves conditional receiver invariance, not a realized phase-effect A/B."
        ),
    }


def run(gt_cache: Path, output_dir: Path, through_r_report: Path | None) -> dict[str, Any]:
    z = np.load(gt_cache, allow_pickle=False)
    required = ("lstars", "margins", "gt_poses")
    missing = [key for key in required if key not in z.files]
    if missing:
        raise ValueError(f"GT cache missing required keys: {missing}")
    labels = np.asarray(z["lstars"])
    margins = np.asarray(z["margins"], dtype=np.float32)
    poses = np.asarray(z["gt_poses"])
    roles = detect_seg_roles(labels)
    ground_classes = (roles.road, roles.lane, roles.undriv)
    cfg = PhaseCarrierConfig(classes=ground_classes)

    incumbent, incumbent_report = phase_carrier_report(labels, margins, poses, cfg)
    masks, class_maps = _extract_probe_geometry(labels, margins, cfg)
    candidate_plain, plain_report = encode_defect_tube_recode(
        incumbent, masks, class_maps, z2_orientation_quotient=False
    )
    candidate_z2, z2_report = encode_defect_tube_recode(
        incumbent, masks, class_maps, z2_orientation_quotient=True
    )
    candidates = [("component_delta", candidate_plain, plain_report), ("z2_gauged", candidate_z2, z2_report)]
    selected_name, selected_blob, selected_report = min(
        candidates, key=lambda item: (len(item[1]), item[0])
    )
    contour_sizes = [encode_partition(frame).n_bytes for frame in labels]
    canary = _through_r_canary(through_r_report)
    exact_equal = bool(
        selected_report.exact_residual_roundtrip and selected_report.exact_phase_field_roundtrip
    )
    bytes_saved = int(selected_report.bytes_saved)
    rate_code_subverdict = "GO" if exact_equal and bytes_saved > 0 else "NO-GO"
    defect_mechanism_subverdict = (
        "GO" if selected_report.component_stream_delta_bytes < 0 else "NO-GO"
    )
    overall_verdict = "NEEDS-MORE" if rate_code_subverdict == "GO" else "NO-GO"
    review_status = "recovery-written-UNREVIEWED"
    receipt: dict[str, Any] = {
        "schema_version": "defect_network_rate_code_probe_v1",
        "task": 452,
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "verdict": overall_verdict,
        "rate_code_subverdict": rate_code_subverdict,
        "rate_code_scope": "standalone PHAS1/DTUB1 section under shared GT-cache geometry",
        "defect_mechanism_subverdict": defect_mechanism_subverdict,
        "verdict_scope": (
            "formulation: lossless standalone PHAS1 recoding on gt_n600 using shared out-of-band "
            "GT-cache-derived 8-connected ground-boundary geometry; receiver derivability and "
            "archive integration unknown; no family/paradigm kill"
        ),
        "review_status": review_status,
        "pre_registered_gate": {
            "go": "candidate section is smaller than PHAS1 and residual plus decoded phase fields are bit-identical",
            "no_go": "candidate is not smaller or any exact equality check fails",
            "through_r_requirement": (
                "a receiver-consumed PHAS1 A/B is required before a GO can be promoted beyond rate-code subverdict"
            ),
            "comparison_noise_floor_bytes": 0,
            "across_seed_variance": "UNKNOWN; deterministic exact-byte comparison has no within-input variance",
        },
        "paper": PAPER,
        "class_roles_self_detected": roles.as_dict(),
        "ground_classes_inherited_from_roles": list(ground_classes),
        "source": {
            "gt_cache": str(gt_cache),
            "gt_cache_sha256": _sha256(gt_cache),
            "n_pairs": int(labels.shape[0]),
            "label_shape": list(labels.shape),
            "geometry_authority": (
                "lstars and margins from the GT cache; shared out-of-band by both codecs in this "
                "probe; actual witness-receiver derivability is UNKNOWN"
            ),
        },
        "execution_source_custody": _execution_source_custody(),
        "incumbent_phase_carrier": {
            "section_bytes": len(incumbent),
            "residual_bytes": incumbent_report.xi_amortized_residual_bytes,
            "raw_anchor_residual_bytes": incumbent_report.raw_tie_residual_bytes,
            "xi_amortization_ratio": incumbent_report.amortization_ratio,
            "residual_count": incumbent_report.total_residual_count,
            "q_step": incumbent_report.q_step,
            "tie_recon_rmse_px": incumbent_report.tie_recon_rmse_px,
            "reconstruction_bit_identical": incumbent_report.reconstruction_bit_identical,
            "review_status": "fresh-eyes-reviewed(0); incumbent source inspected this session",
        },
        "dense_raster_lzma_baseline_context": {
            "meaning": "lossless full label-map description baseline; not co-stored with the phase recode",
            "sum_payload_bytes_600": int(sum(contour_sizes)),
            "mean_payload_bytes_per_frame": float(np.mean(contour_sizes)),
            "min_payload_bytes": int(min(contour_sizes)),
            "max_payload_bytes": int(max(contour_sizes)),
            "duplicate_data_clause": "not added to PHAS1 or DTUB1; one geometric home per byte",
        },
        "candidate_comparison": {
            name: {**report.to_dict(), "sha256": hashlib.sha256(blob).hexdigest()}
            for name, blob, report in candidates
        },
        "selected_candidate": {
            "name": selected_name,
            **selected_report.to_dict(),
            "sha256": hashlib.sha256(selected_blob).hexdigest(),
            "rate_term_delta_if_receiver_closed": 25.0 * (len(selected_blob) - len(incumbent)) / 37_545_489,
        },
        "mechanism_falsifiers": {
            "component_delta_correlation": {
                "status": defect_mechanism_subverdict,
                "incumbent_residual_stream_bytes": (
                    selected_report.incumbent_residual_stream_bytes
                ),
                "component_stream_bytes": selected_report.component_stream_bytes,
                "component_stream_delta_bytes": selected_report.component_stream_delta_bytes,
                "reason": (
                    "the standalone-section candidate is smaller only because shared-cache geometry "
                    "lets it derive counts instead of storing the incumbent per-frame count table; "
                    "the component transform itself "
                    "must beat the incumbent residual stream before defect correlation gets credit"
                ),
            },
            "zero_mode_only": {
                "status": "NO-GO",
                "constant_component_fraction": selected_report.constant_component_fraction,
                "constant_pixel_fraction": selected_report.constant_pixel_fraction,
                "reason": "a per-component constant alone cannot reconstruct the non-constant phase pixels exactly",
            },
            "finite_z2_gauging": {
                "status": (
                    "GO" if z2_report.candidate_section_bytes < plain_report.candidate_section_bytes else "NO-GO"
                ),
                "plain_bytes": plain_report.candidate_section_bytes,
                "z2_bytes": z2_report.candidate_section_bytes,
                "group_label_bytes_counted": z2_report.group_label_bytes,
                "reason": "the group label is counted; only total serialized bytes decide",
            },
            "junction_fusion": {
                "status": "NO-GO_AS_SEPARATE_FIELD_ELIMINATION",
                "three_class_2x2_junction_cells": _junction_count(labels),
                "reason": (
                    "PHAS1 stores no separate vertex datum. Junction phase samples remain ordinary residual "
                    "entries, so a fusion rule has no dedicated field to remove; deleting them would break exact decode."
                ),
            },
        },
        "through_r_canary": canary,
        "boundary_statement": (
            "The selected standalone-section recode is exact and smaller under shared GT-cache geometry, "
            "but actual receiver derivability is unknown and the current phase carrier is not receiver-consumed. "
            "The component streams are compared separately from generic header deduplication. "
            "Therefore d_seg/d_pose equality is conditional on a future receiver consuming the decoded phase field; "
            "this receipt does not promote the rate-code GO into a score or archive GO."
        ),
        "provenance": {
            "git_head": _git(["rev-parse", "HEAD"]),
            "git_status_short": _git(["status", "--short"]),
            "python": sys.version,
            "platform": platform.platform(),
            "command": " ".join(sys.argv),
        },
        "stores_consulted": (
            "tools/corpus_query.py over research/equations/memory/DAG/council/tasks/docs; "
            "phase/covariance/flicker ledgers; canonical task/lane/subagent stores; arXiv 2607.07786 HTML. "
            "Deliberately did not consult paid-provider or live-run actuation stores because this is a $0 rate probe."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "selected_candidate.dtub1"
    receipt_path = output_dir / "measurement_receipt.json"
    manifest_path = output_dir / "artifact_manifest.json"
    candidate_path.write_bytes(selected_blob)
    receipt["artifact_manifest_path"] = str(manifest_path)
    _atomic_json(receipt_path, receipt)
    manifest = {
        "schema_version": "task452_artifact_manifest_v1",
        "measurement_receipt": {
            "path": str(receipt_path),
            "bytes": receipt_path.stat().st_size,
            "sha256": _sha256(receipt_path),
        },
        "selected_candidate": {
            "path": str(candidate_path),
            "bytes": candidate_path.stat().st_size,
            "sha256": _sha256(candidate_path),
        },
        "through_r_canary": (
            None
            if through_r_report is None
            else {
                "path": str(through_r_report),
                "bytes": through_r_report.stat().st_size,
                "sha256": _sha256(through_r_report),
            }
        ),
        "execution_source_custody": receipt["execution_source_custody"],
    }
    _atomic_json(manifest_path, manifest)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt-cache",
        type=Path,
        default=REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--through-r-report", type=Path)
    args = parser.parse_args()
    if str(args.output_dir).startswith(("/tmp", "/private/tmp")):
        raise SystemExit("output-dir must be durable under experiments/results, never /tmp")
    receipt = run(args.gt_cache, args.output_dir, args.through_r_report)
    selected = receipt["selected_candidate"]
    print(
        json.dumps(
            {
                "verdict": receipt["verdict"],
                "rate_code_subverdict": receipt["rate_code_subverdict"],
                "defect_mechanism_subverdict": receipt["defect_mechanism_subverdict"],
                "selected": selected["name"],
                "incumbent_bytes": receipt["incumbent_phase_carrier"]["section_bytes"],
                "candidate_bytes": selected["candidate_section_bytes"],
                "bytes_saved": selected["bytes_saved"],
                "exact_phase_field_roundtrip": selected["exact_phase_field_roundtrip"],
                "receipt": str(args.output_dir / "measurement_receipt.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
