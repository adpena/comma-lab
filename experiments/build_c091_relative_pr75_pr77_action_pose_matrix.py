#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a C091-relative local matrix for PR75/PR77 action-pose candidates.

The tool does not generate or dispatch remote jobs. It validates existing
single-member archive candidates, recomputes SHA/byte custody, and records the
component-improvement break-even needed to cross 0.314 against the C091 PR75
public replay anchor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/c091_relative_pr75_pr77_action_pose_20260503_codex"
)
TOOL = "experiments/build_c091_relative_pr75_pr77_action_pose_matrix.py"
SCHEMA = "c091_relative_pr75_pr77_action_pose_candidate_matrix_v1"
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
SUB314_TARGET = 0.314
C091 = {
    "archive_bytes": 276_481,
    "archive_sha256": "03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746",
    "score": 0.31516575028285976,
    "score_component": 0.060804000000000004 + 0.07026450028285976,
}


@dataclass(frozen=True)
class CandidateSeed:
    candidate_id: str
    archive_path: Path
    manifest_path: Path | None
    family: str
    source_streams: dict[str, str]
    exact_dispatch_readiness: str
    local_readiness: str
    component_basin_prior: str
    rank_notes: list[str]
    top3_rank: int | None


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _safe_zip_name(name: str) -> str:
    parts = Path(name).parts
    if (
        not name
        or name.startswith("/")
        or ".." in parts
        or len(parts) != 1
        or name.startswith(".")
        or name.startswith("__MACOSX/")
        or name.startswith("._")
    ):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return name


def _archive_profile(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = zf.infolist()
        names = [_safe_zip_name(info.filename) for info in infos if not info.is_dir()]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        members = [
            {
                "compress_size": info.compress_size,
                "compress_type": info.compress_type,
                "date_time": list(info.date_time),
                "external_attr": info.external_attr,
                "filename": info.filename,
                "file_size": info.file_size,
            }
            for info in infos
        ]
    single_stored_p = (
        len(members) == 1
        and members[0]["filename"] == "p"
        and int(members[0]["compress_type"]) == zipfile.ZIP_STORED
    )
    return {
        "archive_bytes": path.stat().st_size,
        "duplicate_names": duplicate_names,
        "member_count": len(members),
        "members": members,
        "single_stored_member_p": single_stored_p,
        "status": "passed" if single_stored_p and not duplicate_names else "failed",
    }


def _break_even(archive_bytes: int) -> dict[str, Any]:
    delta_bytes = archive_bytes - int(C091["archive_bytes"])
    rate_delta = delta_bytes * RATE_SCORE_PER_BYTE
    score_if_components_unchanged = float(C091["score"]) + rate_delta
    required = max(0.0, score_if_components_unchanged - SUB314_TARGET)
    return {
        "anchor": "c091_pr75_public_replay",
        "archive_delta_bytes_vs_c091": delta_bytes,
        "rate_score_delta_vs_c091": rate_delta,
        "score_if_components_unchanged": score_if_components_unchanged,
        "sub314_component_score_improvement_needed": required,
        "sub314_equivalent_bytes_needed_after_candidate": (
            math.ceil(required / RATE_SCORE_PER_BYTE) if required > 0 else 0
        ),
    }


def _load_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def _manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "noop",
        "noop_status",
        "policy",
        "wire_format",
        "selected_record_count",
        "selected_delta_seg_trace_sum",
        "selected_delta_pose_trace_sum",
        "selected_delta_combined_trace_sum",
        "changed_action_id_record_count",
        "pair_tile_duplicate_record_count",
        "encoded_action_brotli_bytes",
        "encoded_action_codec",
        "runtime_action_raw_bytes",
    ):
        if key in manifest:
            out[key] = manifest[key]
    guard = manifest.get("action_selection_guard")
    if isinstance(guard, dict):
        out["action_selection_guard"] = {
            key: guard.get(key)
            for key in (
                "changed_action_id_record_count",
                "custom_delta_record_count",
                "exact_duplicate_record_count",
                "pair_tile_duplicate_record_count",
                "record_count",
                "transformed_record_count",
            )
        }
    dispatch_safety = manifest.get("dispatch_safety")
    if isinstance(dispatch_safety, dict):
        out["source_manifest_dispatch_safety"] = dispatch_safety
    return out


def _exact_eval_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    component_score = float(data["score_seg_contribution"]) + float(
        data["score_pose_contribution"]
    )
    return {
        "archive_bytes": int(data["archive_size_bytes"]),
        "archive_sha256": data["provenance"]["archive_sha256"],
        "avg_posenet_dist": float(data["avg_posenet_dist"]),
        "avg_segnet_dist": float(data["avg_segnet_dist"]),
        "component_score": component_score,
        "component_score_delta_vs_c091": component_score - float(C091["score_component"]),
        "path": _rel(path),
        "score_recomputed_from_components": float(data["score_recomputed_from_components"]),
    }


def _component_evidence() -> dict[str, Any]:
    paths = {
        "c091_anchor": REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/contest_auth_eval.json",
        "pr77_public_replay": REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_pr77_tile_delta_public_replay_t4_20260503T1116Z/contest_auth_eval.json",
        "pr75_top40_p3_t4": REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top40_p3_t4_20260503T0440Z/contest_auth_eval.json",
        "pr75_top25_ampminus1_p3_t4": REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top25_ampminus1_p3_t4_20260503T0520Z/contest_auth_eval.json",
        "pr75_minp_public_actions_only_t4": REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_pr75_minp_public_actions_only_t4_20260503T1101Z/contest_auth_eval.json",
    }
    return {
        label: _exact_eval_summary(path)
        for label, path in paths.items()
        if path.exists()
    }


def _candidate_seeds() -> list[CandidateSeed]:
    return [
        CandidateSeed(
            candidate_id="pr77_actions_pr75mask_renderer_c089pose_fixedslice",
            archive_path=REPO_ROOT
            / "experiments/results/pr77_action_pose_mixed_container_20260503_codex/pr77_actions_pr75mask_renderer_c089pose_fixedslice/archive.zip",
            manifest_path=REPO_ROOT
            / "experiments/results/pr77_action_pose_mixed_container_20260503_codex/pr77_actions_pr75mask_renderer_c089pose_fixedslice/manifest.json",
            family="pr77_action_pose_mixed_container",
            source_streams={
                "actions": "pr77_public_action_delta_source_order",
                "mask": "c091_pr75_public",
                "pose": "c089_qp1_lossless_resweep",
                "renderer": "c091_pr75_public",
            },
            exact_dispatch_readiness="ready_after_lane_claim; currently queued separately as exact_eval_pr77_action_pose_fixedslice_t4_20260503T114254Z",
            local_readiness="runtime_parse_validation_passed_in_source_manifest",
            component_basin_prior="PR77 public exact replay was component-negative vs C091; this row changes pose packing too, so exact eval is required but sub-0.314 is not supported by local math.",
            rank_notes=[
                "best existing PR77 mixed byte count",
                "preserves PR77 source action order",
                "needs 1599 byte-equivalent component gain after rate saving",
            ],
            top3_rank=1,
        ),
        CandidateSeed(
            candidate_id="c067_pr75_actions_pose_safe_positive_ampminus1_p6",
            archive_path=REPO_ROOT
            / "experiments/results/pr75_qp1_nextwave_sub314_20260503_worker/action_compiler_candidates/c067_pr75_actions_pose_safe_positive_ampminus1_p6/archive.zip",
            manifest_path=REPO_ROOT
            / "experiments/results/pr75_qp1_nextwave_sub314_20260503_worker/action_compiler_candidates/c067_pr75_actions_pose_safe_positive_ampminus1_p6/manifest.json",
            family="pr75_action_policy_p6",
            source_streams={
                "actions": "PR75 records filtered to positive pose-safe atoms, ampminus1 transform",
                "mask": "C067/C089 baseline source",
                "pose": "C067/C089 QP1 source",
                "renderer": "C067/C089 renderer source",
            },
            exact_dispatch_readiness="ready_after_lane_claim_no_remote_dispatch_from_this_tool",
            local_readiness="manifest guard passed: changed actions, no exact duplicates, no pair/tile duplicates",
            component_basin_prior="real action-id changes with positive local trace sums, but trace gain is far below the C091 sub-0.314 break-even.",
            rank_notes=[
                "lowest-byte real action-policy candidate in the local PR75/QP1 nextwave matrix",
                "28 changed action ids and positive pose trace",
                "not enough modeled component gain for sub-0.314",
            ],
            top3_rank=2,
        ),
        CandidateSeed(
            candidate_id="c067_pr75_actions_positive_poseharm_ampminus1_p6",
            archive_path=REPO_ROOT
            / "experiments/results/pr75_qp1_nextwave_sub314_20260503_worker/action_compiler_candidates/c067_pr75_actions_positive_poseharm_ampminus1_p6/archive.zip",
            manifest_path=REPO_ROOT
            / "experiments/results/pr75_qp1_nextwave_sub314_20260503_worker/action_compiler_candidates/c067_pr75_actions_positive_poseharm_ampminus1_p6/manifest.json",
            family="pr75_action_policy_p6",
            source_streams={
                "actions": "PR75 positive atoms with pose-harm ampminus1 shrink",
                "mask": "C067/C089 baseline source",
                "pose": "C067/C089 QP1 source",
                "renderer": "C067/C089 renderer source",
            },
            exact_dispatch_readiness="ready_after_lane_claim_no_remote_dispatch_from_this_tool",
            local_readiness="manifest guard passed: changed actions, no exact duplicates, no pair/tile duplicates",
            component_basin_prior="larger selected SegNet trace than the pose-safe row, but exact PR75 action probes have not shown sub-0.314-scale gains.",
            rank_notes=[
                "60 selected records and 31 changed action ids",
                "larger selected trace sum than pose-safe row",
                "slightly worse byte count than the best PR77 mixed candidate",
            ],
            top3_rank=3,
        ),
        CandidateSeed(
            candidate_id="pr77_actions_c089mask_pr75renderer_c089pose_p3",
            archive_path=REPO_ROOT
            / "experiments/results/pr77_action_pose_mixed_container_20260503_codex/pr77_actions_c089mask_pr75renderer_c089pose_p3/archive.zip",
            manifest_path=REPO_ROOT
            / "experiments/results/pr77_action_pose_mixed_container_20260503_codex/pr77_actions_c089mask_pr75renderer_c089pose_p3/manifest.json",
            family="pr77_action_pose_mixed_container",
            source_streams={
                "actions": "pr77_public_action_delta_source_order",
                "mask": "c089_lossless_resweep",
                "pose": "c089_qp1_lossless_resweep",
                "renderer": "c091_pr75_public",
            },
            exact_dispatch_readiness="fallback_ready_after_lane_claim_if_fixedslice_candidate_fails",
            local_readiness="runtime_parse_validation_passed_in_source_manifest",
            component_basin_prior="same PR77 action basin as the queued fixed-slice candidate; use only as parser-risk fallback.",
            rank_notes=[
                "self-describing P3 fallback",
                "3 bytes larger than fixed-slice row",
                "not an independent component experiment",
            ],
            top3_rank=None,
        ),
        CandidateSeed(
            candidate_id="pr77_actions_sorted_c089mask_pr75renderer_c089pose_p6_probe",
            archive_path=REPO_ROOT
            / "experiments/results/pr77_action_pose_mixed_container_20260503_codex/pr77_actions_sorted_c089mask_pr75renderer_c089pose_p6_probe/archive.zip",
            manifest_path=REPO_ROOT
            / "experiments/results/pr77_action_pose_mixed_container_20260503_codex/pr77_actions_sorted_c089mask_pr75renderer_c089pose_p6_probe/manifest.json",
            family="pr77_action_order_probe",
            source_streams={
                "actions": "pr77_public_actions_sorted_for_p6_probe",
                "mask": "c089_lossless_resweep",
                "pose": "c089_qp1_lossless_resweep",
                "renderer": "c091_pr75_public",
            },
            exact_dispatch_readiness="not_ready; requires local action-order raw-output parity proof",
            local_readiness="fail_closed_before_exact_eval",
            component_basin_prior="changes action record order; not dispatchable until parity proves runtime output equivalence.",
            rank_notes=[
                "P6 byte probe only",
                "decoded action order changes",
                "do not dispatch from this matrix",
            ],
            top3_rank=None,
        ),
    ]


def build_matrix(*, output_dir: Path) -> dict[str, Any]:
    component_evidence = _component_evidence()
    rows: list[dict[str, Any]] = []
    for seed in _candidate_seeds():
        if not seed.archive_path.exists():
            raise FileNotFoundError(seed.archive_path)
        archive_bytes = seed.archive_path.stat().st_size
        archive_sha = _sha256_file(seed.archive_path)
        manifest = _load_json(seed.manifest_path)
        break_even = _break_even(archive_bytes)
        matrix_row = {
            "archive_bytes": archive_bytes,
            "archive_path": _rel(seed.archive_path),
            "archive_sha256": archive_sha,
            "break_even_vs_c091": break_even,
            "candidate_id": seed.candidate_id,
            "component_basin_prior": seed.component_basin_prior,
            "evidence_grade": "empirical_byte_screen_and_prior_exact_eval_context",
            "exact_dispatch_readiness": seed.exact_dispatch_readiness,
            "family": seed.family,
            "local_readiness": seed.local_readiness,
            "manifest_path": _rel(seed.manifest_path) if seed.manifest_path else None,
            "no_remote_gpu_dispatch_performed": True,
            "rank_notes": seed.rank_notes,
            "score_claim": False,
            "source_manifest_summary": _manifest_summary(manifest)
            if isinstance(manifest, dict)
            else {},
            "source_streams": seed.source_streams,
            "sub314_dispatch_worthy": False,
            "top3_exact_eval_candidate_rank": seed.top3_rank,
            "zip_profile": _archive_profile(seed.archive_path),
        }
        if matrix_row["zip_profile"]["status"] != "passed":
            matrix_row["exact_dispatch_readiness"] = "not_ready; archive profile failed"
            matrix_row["local_readiness"] = "failed_archive_profile"
        rows.append(matrix_row)
    rows = sorted(
        rows,
        key=lambda row: (
            row["top3_exact_eval_candidate_rank"] is None,
            row["top3_exact_eval_candidate_rank"] or 999,
            row["archive_bytes"],
            row["candidate_id"],
        ),
    )
    summary = {
        "anchor": {
            "archive_bytes": C091["archive_bytes"],
            "archive_sha256": C091["archive_sha256"],
            "score": C091["score"],
            "score_component": C091["score_component"],
        },
        "candidate_count": len(rows),
        "candidates": rows,
        "component_evidence": component_evidence,
        "finding": {
            "no_new_sub314_dispatch_worthy_candidate": True,
            "mathematical_blocker": (
                "Best local rows save only 152-164 bytes versus C091, worth about "
                "0.00010 score. Crossing 0.314 still needs about 0.00106 "
                "component-score improvement, while observed PR77 replay and PR75 "
                "action exact probes are component-negative or far smaller."
            ),
        },
        "promotion_eligible": False,
        "schema": SCHEMA,
        "score_claim": False,
        "sub314_target": SUB314_TARGET,
        "tool": TOOL,
    }
    _write_json(output_dir / "candidate_matrix.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_matrix(output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "candidate_count": summary["candidate_count"],
                "matrix_path": str((args.output_dir / "candidate_matrix.json").resolve()),
                "no_new_sub314_dispatch_worthy_candidate": summary["finding"][
                    "no_new_sub314_dispatch_worthy_candidate"
                ],
                "top3": [
                    {
                        "archive_bytes": row["archive_bytes"],
                        "archive_sha256": row["archive_sha256"],
                        "candidate_id": row["candidate_id"],
                        "rank": row["top3_exact_eval_candidate_rank"],
                    }
                    for row in summary["candidates"]
                    if row["top3_exact_eval_candidate_rank"] is not None
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
