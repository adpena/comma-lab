# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

import tac.optimization.l5_staircase_v2 as l5_v2
from tac.exact_eval_custody import contest_score
from tac.optimization.l5_staircase_v2 import (
    L5_V2_ASYMPTOTIC_CANDIDATE_SURFACE_ARTIFACT_PATH,
    L5_V2_ASYMPTOTIC_CANDIDATE_SURFACE_REPORT_PATH,
    L5_V2_PACKETIR_SECTION_ENTROPY_EVIDENCE_SCHEMA,
    L5_V2_PACKETIR_SECTION_ENTROPY_MATRIX_ARTIFACT_PATH,
    L5_V2_PACKETIR_SECTION_ENTROPY_MATRIX_ARTIFACT_SHA256,
    L5_V2_PACKETIR_STACK_EVIDENCE_SCHEMA,
    L5_V2_PR106_STACK_CELL_CANDIDATES_SCHEMA,
    PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_SHA256,
    PREDICTED_DELTA_BAND,
    SUBJECT_ID,
    L5V2GateEvidence,
    l5_v2_architecture_lock_packet,
    l5_v2_canonical_sideinfo_gate_evidence,
    l5_v2_dispatch_readiness,
    l5_v2_packetir_section_entropy_evidence_payload,
    l5_v2_packetir_stack_evidence_payload,
    l5_v2_pr106_stack_cell_candidates,
    l5_v2_prediction_band_payload,
    l5_v2_prediction_band_status,
    l5_v2_prediction_band_verdict,
    l5_v2_required_gates,
    l5_v2_research_basis_ids,
    l5_v2_staircase_steps,
    l5_v2_tt5l_campaign_readiness,
    render_l5_v2_architecture_lock_packet_markdown,
    render_l5_v2_asymptotic_candidate_surface_markdown,
)
from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES,
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS,
    L5V2_SIDEINFO_EFFECT_CURVE_SCHEMA,
)
from tac.optimization.l5_v2_probe_disambiguator import (
    L5V2_CANDIDATES,
    L5V2_PROBE_SCHEMA,
    L5V2_PROBE_TOOL_PATH,
    evaluate_l5_v2_probe,
    observation_from_mapping,
)
from tac.optimization.substrate_composition_matrix import (
    build_composition_matrix,
    per_substrate_pareto_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _sha(seed: int) -> str:
    return f"{seed:064x}"[-64:]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recursive_string_contains(value: object, needle: str) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, dict):
        return any(_recursive_string_contains(v, needle) for v in value.values())
    if isinstance(value, list):
        return any(_recursive_string_contains(v, needle) for v in value)
    return False


def _write_aggregate_manifest(path: Path, *, aggregate_sha: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"aggregate_sha256": aggregate_sha}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _file_sha256(path)


def _write_tt5l_archive_zip(
    path: Path,
    *,
    nonzero_side_info: bool,
    num_pairs: int = 600,
) -> None:
    from tac.substrates.time_traveler_l5_autonomy.architecture import (
        TimeTravelerConfig,
        TimeTravelerSubstrate,
    )
    from tac.substrates.time_traveler_l5_autonomy.archive import pack_archive

    torch.manual_seed(0)
    cfg = TimeTravelerConfig(
        hidden_dim=8,
        num_hidden_layers=1,
        num_pairs=num_pairs,
        output_height=8,
        output_width=8,
        per_pair_side_info_bytes=45,
    )
    substrate = TimeTravelerSubstrate(cfg)
    side_info = np.zeros((cfg.num_pairs, cfg.per_pair_side_info_bytes), dtype=np.int8)
    if nonzero_side_info:
        side_info[0, 0] = 1
    bin_bytes = pack_archive(
        world_model_state_dict=substrate.state_dict(),
        per_pair_side_info=side_info,
        meta={
            "int8_scale": 64.0,
            "first_omega": cfg.first_omega,
            "hidden_omega": cfg.hidden_omega,
            "coord_feature_freqs": cfg.coord_feature_freqs,
            "coord_dim": cfg.coord_dim,
            "markov_transition_band": cfg.markov_transition_band,
        },
        num_pairs=cfg.num_pairs,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
        foveation_grid_h=cfg.foveation_grid_h,
        foveation_grid_w=cfg.foveation_grid_w,
        pose_dim=cfg.pose_dim,
        per_pair_bytes=cfg.per_pair_side_info_bytes,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", bin_bytes)


def _write_standard_tt5l_materialized_work_unit_plan(
    path: Path,
    *,
    archive_rel: str,
    archive_sha: str,
    archive_bytes: int,
    runtime_rel: str,
) -> None:
    pair_group_id = "pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda"
    cpu_lane = "lane_l5_v2_measure_tt5l_autonomy_paired_exact_contest_cpu"
    cuda_lane = "lane_l5_v2_measure_tt5l_autonomy_paired_exact_contest_cuda"
    run_id = "l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_random_lsb"
    output_root = "experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact"
    cpu_output = f"{output_root}/modal_auth_eval_cpu/{run_id}_cpu"
    cuda_output = f"{output_root}/modal_auth_eval/{run_id}_cuda"
    path.write_text(
        json.dumps(
            {
                "schema": "modal_paired_auth_eval_dispatch_plan_v2",
                "pair_group_id": pair_group_id,
                "run_id": run_id,
                "output_root": output_root,
                "score_claim": False,
                "promotion_eligible": False,
                "archive": {
                    "path": archive_rel,
                    "sha256": archive_sha,
                    "bytes": archive_bytes,
                },
                "runtime": {
                    "submission_dir": runtime_rel,
                    "expected_runtime_tree_sha256_by_axis": {
                        "contest_cpu": "b" * 64,
                        "contest_cuda": "c" * 64,
                    },
                    "expected_runtime_content_tree_sha256_by_axis": {
                        "contest_cpu": "d" * 64,
                        "contest_cuda": "d" * 64,
                    },
                },
                "required_axes": ["contest_cpu", "contest_cuda"],
                "lanes": {
                    "contest_cpu": cpu_lane,
                    "contest_cuda": cuda_lane,
                },
                "outputs": {
                    "contest_cpu": cpu_output,
                    "contest_cuda": cuda_output,
                },
                "commands": {
                    "contest_cpu": [
                        ".venv/bin/modal",
                        "run",
                        "--detach",
                        "experiments/modal_auth_eval_cpu.py",
                        "--archive",
                        archive_rel,
                        "--expected-archive-sha256",
                        archive_sha,
                        "--submission-dir",
                        runtime_rel,
                        "--pair-group-id",
                        pair_group_id,
                        "--lane-id",
                        cpu_lane,
                        "--instance-job-id",
                        f"{run_id}_cpu",
                        "--output-dir",
                        cpu_output,
                        "--expected-runtime-tree-sha256",
                        "b" * 64,
                        "--provider-detach-ack",
                    ],
                    "contest_cuda": [
                        ".venv/bin/modal",
                        "run",
                        "--detach",
                        "experiments/modal_auth_eval.py",
                        "--archive",
                        archive_rel,
                        "--expected-archive-sha256",
                        archive_sha,
                        "--submission-dir",
                        runtime_rel,
                        "--pair-group-id",
                        pair_group_id,
                        "--lane-id",
                        cuda_lane,
                        "--instance-job-id",
                        f"{run_id}_cuda",
                        "--output-dir",
                        cuda_output,
                        "--expected-runtime-tree-sha256",
                        "c" * 64,
                        "--provider-detach-ack",
                    ],
                },
                "axes_skipped_due_to_existing_anchor": {
                    "contest_cpu": False,
                    "contest_cuda": False,
                },
                "existing_anchors_reused": {
                    "contest_cpu": None,
                    "contest_cuda": None,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _axis_rows(*, anchor_type: str | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for axis in ("contest_cpu", "contest_cuda"):
        archive_bytes = 1
        row: dict[str, object] = {
            "axis": axis,
            "archive_sha256": _sha(11),
            "runtime_tree_sha256": _sha(12),
            "runtime_content_tree_sha256": _sha(13),
            "inflate_device": "cpu" if axis == "contest_cpu" else "cuda",
            "eval_device": "cpu" if axis == "contest_cpu" else "cuda",
            "component_deltas": {
                "seg_dist_delta": 0.0,
                "pose_dist_delta": 0.0,
                "score_delta": 0.0,
            },
        }
        if anchor_type is not None:
            row["anchor_type"] = anchor_type
            row["score_claim"] = False
            row["evidence_grade"] = (
                "contest_paired_exact"
                if anchor_type == "exact"
                else "paired_diagnostic"
            )
            if anchor_type == "diagnostic":
                row["diagnostic_reason"] = "diagnostic anchor, not promotion evidence"
            if anchor_type == "exact":
                row["score"] = 25.0 * archive_bytes / 37_545_489
                row["seg_dist"] = 0.0
                row["pose_dist"] = 0.0
                row["archive_bytes"] = archive_bytes
                row["n_samples"] = 600
                row["hardware"] = (
                    "linux-x86_64" if axis == "contest_cpu" else "modal-t4"
                )
                row["auth_eval_command"] = f"contest_auth_eval --axis {axis}"
                row["artifact_path"] = (
                    "experiments/results/time_traveler_l5_v2/"
                    f"{axis}_anchor.json"
                )
                row["log_path"] = (
                    "experiments/results/time_traveler_l5_v2/"
                    f"{axis}_anchor.log"
                )
                row["inflated_outputs_manifest_path"] = (
                    "experiments/results/time_traveler_l5_v2/"
                    f"{axis}_inflated_outputs_manifest.json"
                )
                row["inflated_raw_output_aggregate_sha256"] = (
                    _sha(31) if axis == "contest_cpu" else _sha(32)
                )
        rows.append(row)
    return rows


def _probe_observation_payload(
    *,
    candidate_id: str,
    delta: float,
    repo_root: Path,
    seed: int,
) -> dict[str, object]:
    archive_bytes = 1
    archive_sha = _sha(200 + seed)
    runtime_sha = _sha(300 + seed)
    artifact_path = (
        "experiments/results/time_traveler_l5_v2/"
        f"{candidate_id}_probe_artifact.json"
    )
    artifact_file = repo_root / artifact_path
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_file.write_text(
        json.dumps({"candidate_id": candidate_id, "delta": delta}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    axis_evidence: list[dict[str, object]] = []
    for axis in ("contest_cpu", "contest_cuda"):
        log_path = (
            "experiments/results/time_traveler_l5_v2/"
            f"{candidate_id}_{axis}_probe.log"
        )
        log_file = repo_root / log_path
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(f"{candidate_id} {axis}\n", encoding="utf-8")
        axis_artifact_path = (
            "experiments/results/time_traveler_l5_v2/"
            f"{candidate_id}_{axis}_probe.json"
        )
        axis_artifact_file = repo_root / axis_artifact_path
        axis_artifact_file.write_text(
            json.dumps(
                {
                    "axis": axis,
                    "candidate_id": candidate_id,
                    "score_delta": delta,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raw_output_aggregate_sha = _sha(400 + seed * 10 + len(axis))
        manifest_path = (
            "experiments/results/time_traveler_l5_v2/"
            f"{candidate_id}_{axis}_inflated_outputs_manifest.json"
        )
        manifest_file = repo_root / manifest_path
        manifest_file.write_text(
            json.dumps(
                {
                    "schema": "contest_auth_eval_inflated_output_manifest_v1",
                    "aggregate_sha256": raw_output_aggregate_sha,
                    "raw_file_count": 1,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        axis_evidence.append(
            {
                "axis": axis,
                "archive_sha256": archive_sha,
                "runtime_tree_sha256": runtime_sha,
                "score": 25.0 * archive_bytes / 37_545_489,
                "seg_dist": 0.0,
                "pose_dist": 0.0,
                "archive_bytes": archive_bytes,
                "n_samples": 600,
                "hardware": "linux-x86_64" if axis == "contest_cpu" else "modal-t4",
                "inflate_device": "cpu" if axis == "contest_cpu" else "cuda",
                "eval_device": "cpu" if axis == "contest_cpu" else "cuda",
                "auth_eval_command": (
                    f"contest_auth_eval --axis {axis} "
                    f"--device {'cpu' if axis == 'contest_cpu' else 'cuda'}"
                ),
                "log_path": log_path,
                "artifact_path": axis_artifact_path,
                "artifact_sha256": _file_sha256(axis_artifact_file),
                "inflated_outputs_manifest_path": manifest_path,
                "inflated_outputs_manifest_sha256": _file_sha256(manifest_file),
                "raw_output_aggregate_sha256": raw_output_aggregate_sha,
                "score_delta": delta,
            }
        )
    return {
        "candidate_id": candidate_id,
        "predicted_or_measured_delta": delta,
        "evidence_grade": "contest_paired_exact",
        "exact_axes": ["contest_cpu", "contest_cuda"],
        "artifact_path": artifact_path,
        "artifact_sha256": _file_sha256(artifact_file),
        "predicate_id": f"l5_v2_probe_{candidate_id}_predicate",
        "predicate_passed": True,
        "archive_sha256": archive_sha,
        "runtime_tree_sha256": runtime_sha,
        "axis_evidence": axis_evidence,
        "sideinfo_consumed": True,
        "byte_closed_archive": True,
    }


def _probe_disambiguator_payload(repo_root: Path) -> dict[str, object]:
    deltas = {
        "c1_world_model_foveation": -0.010,
        "z5_predictive_coding_world_model": -0.020,
        "time_traveler_l5_autonomy": -0.030,
    }
    observations = [
        _probe_observation_payload(
            candidate_id=candidate_id,
            delta=deltas[candidate_id],
            repo_root=repo_root,
            seed=index,
        )
        for index, candidate_id in enumerate(L5V2_CANDIDATES)
    ]
    verdict = evaluate_l5_v2_probe(
        tuple(observation_from_mapping(row) for row in observations),
        repo_root=repo_root,
    )
    return {
        "schema": L5V2_PROBE_SCHEMA,
        "tool_path": L5V2_PROBE_TOOL_PATH,
        "candidate_ids": list(L5V2_CANDIDATES),
        "paired_exact_axes_required": True,
        "observations": observations,
        "verdict": verdict,
        "verdict_sha256": _canonical_json_sha256(verdict),
    }


def _gate_artifact_payload(gate_id: str, *, repo_root: Path | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "gate_id": gate_id,
        "predicate_id": f"l5_v2_{gate_id}_predicate",
        "passed": True,
    }
    if gate_id == "byte_closed_temporal_sideinfo_consumption":
        payload["proof_scope"] = "contest_full_frame_consumption_proof"
        payload["byte_mutation_proof"] = {
            "section": "temporal_sideinfo",
            "parser_consumed_bytes": True,
            "output_changed": True,
            "raw_output_shape_compatible": True,
            "non_target_sections_identical": True,
            "non_target_payload_sections_identical": True,
            "allowed_header_delta": {
                "allowed": True,
                "changed_fields": [],
                "allowed_changed_fields": ["side_len"],
            },
            "section_hashes": {
                "tt5l_header": {
                    "target_section": False,
                    "baseline_sha256": _sha(50),
                    "mutated_sha256": _sha(50),
                    "identical": True,
                    "allowed_delta": {
                        "allowed": True,
                        "changed_fields": [],
                        "allowed_changed_fields": ["side_len"],
                    },
                },
                "world_model_blob": {
                    "target_section": False,
                    "baseline_sha256": _sha(51),
                    "mutated_sha256": _sha(51),
                    "identical": True,
                },
                "ac_state_blob": {
                    "target_section": False,
                    "baseline_sha256": _sha(52),
                    "mutated_sha256": _sha(52),
                    "identical": True,
                },
                "meta_blob": {
                    "target_section": False,
                    "baseline_sha256": _sha(53),
                    "mutated_sha256": _sha(53),
                    "identical": True,
                },
                "per_pair_side_info_blob": {
                    "target_section": True,
                    "baseline_sha256": _sha(54),
                    "mutated_sha256": _sha(55),
                    "identical": False,
                },
            },
            "section_offset": 1024,
            "section_nbytes": 16,
            "section_sha256": _sha(24),
            "baseline_archive_sha256": _sha(25),
            "mutated_archive_sha256": _sha(26),
            "runtime_tree_sha256": _sha(27),
            "mutated_byte_offsets": [1024, 1031],
            "baseline_inflate_sha256": _sha(21),
            "mutated_inflate_sha256": _sha(22),
            "inflate_command": (
                "submissions/time_traveler_l5_autonomy/inflate.sh "
                "archive_dir output_dir file_list.txt"
            ),
            "inflated_outputs_manifest_path": (
                "experiments/results/time_traveler_l5_v2/"
                "byte_mutation_inflated_outputs_manifest.json"
            ),
            "inflated_raw_output_aggregate_sha256": _sha(23),
            "n_pairs_hashed": 600,
            "total_frames": 1200,
            "raw_output_frame_nbytes": 874 * 1164 * 3,
            "file_list_sha256": _sha(28),
            "baseline_raw_output_aggregate_sha256": _sha(29),
            "mutated_raw_output_aggregate_sha256": _sha(30),
            "inflate_provenance_required": True,
            "inflate_provenance_valid": True,
            "inflate_provenance_blockers": [],
            "inflate_provenance": {
                "baseline": {
                    "schema": "tt5l_inflate_provenance_v1",
                    "archive_sha256": _sha(25),
                    "runtime_tree_sha256": _sha(27),
                    "file_list_sha256": _sha(28),
                    "output_aggregate_sha256": _sha(29),
                    "command": (
                        ".venv/bin/python "
                        "src/tac/substrates/time_traveler_l5_autonomy/inflate.py "
                        "archive_dir output_dir file_list.txt"
                    ),
                    "exit_code": 0,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "mutated": {
                    "schema": "tt5l_inflate_provenance_v1",
                    "archive_sha256": _sha(26),
                    "runtime_tree_sha256": _sha(27),
                    "file_list_sha256": _sha(28),
                    "output_aggregate_sha256": _sha(30),
                    "command": (
                        ".venv/bin/python "
                        "src/tac/substrates/time_traveler_l5_autonomy/inflate.py "
                        "archive_dir output_dir file_list.txt"
                    ),
                    "exit_code": 0,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
            },
        }
        if repo_root is not None:
            proof = payload["byte_mutation_proof"]
            assert isinstance(proof, dict)
            provenance = proof["inflate_provenance"]
            assert isinstance(provenance, dict)
            log_root = repo_root / "experiments" / "results" / "time_traveler_l5_v2"
            log_root.mkdir(parents=True, exist_ok=True)
            for label in ("baseline", "mutated"):
                log_path = log_root / f"{label}_inflate.log"
                log_path.write_text(
                    f"{label} TT5L inflate completed with exit_code=0\n",
                    encoding="utf-8",
                )
                entry = provenance[label]
                assert isinstance(entry, dict)
                entry["log_path"] = str(log_path.relative_to(repo_root))
                entry["log_sha256"] = _file_sha256(log_path)
                entry["log_bytes"] = log_path.stat().st_size
    elif gate_id == "c1_z5_tt5l_probe_disambiguator":
        assert repo_root is not None
        payload["probe_disambiguator"] = _probe_disambiguator_payload(repo_root)
    elif gate_id == "paired_cpu_cuda_axis_plan":
        payload["paired_axis_plan"] = _axis_rows()
    elif gate_id == "exact_anchor_or_diagnostic_pair":
        payload["anchor_pair"] = _axis_rows(anchor_type="exact")
    return payload


def _valid_gate_evidence(repo_root: Path) -> dict[str, L5V2GateEvidence]:
    out: dict[str, L5V2GateEvidence] = {}
    artifact_root = repo_root / "experiments" / "results" / "time_traveler_l5_v2"
    artifact_root.mkdir(parents=True, exist_ok=True)
    for gate in l5_v2_required_gates():
        artifact_path = artifact_root / f"{gate.gate_id}.json"
        payload = _gate_artifact_payload(gate.gate_id, repo_root=repo_root)
        if gate.gate_id == "byte_closed_temporal_sideinfo_consumption":
            proof = payload.get("byte_mutation_proof")
            assert isinstance(proof, dict)
            manifest_path = repo_root / str(proof["inflated_outputs_manifest_path"])
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                        {
                            "raw_output_aggregate_sha256": proof[
                                "inflated_raw_output_aggregate_sha256"
                            ],
                            "n_pairs_hashed": proof["n_pairs_hashed"],
                            "total_frames": proof["total_frames"],
                            "raw_output_frame_nbytes": proof[
                                "raw_output_frame_nbytes"
                            ],
                            "file_list_sha256": proof["file_list_sha256"],
                            "baseline_raw_output_aggregate_sha256": proof[
                                "baseline_raw_output_aggregate_sha256"
                            ],
                            "mutated_raw_output_aggregate_sha256": proof[
                                "mutated_raw_output_aggregate_sha256"
                            ],
                        },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        rows = payload.get("anchor_pair")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for key in ("artifact_path", "log_path", "inflated_outputs_manifest_path"):
                    custody_path = repo_root / str(row[key])
                    custody_path.parent.mkdir(parents=True, exist_ok=True)
                    if key == "inflated_outputs_manifest_path":
                        custody_path.write_text(
                            json.dumps(
                                {
                                    "aggregate_sha256": row[
                                        "inflated_raw_output_aggregate_sha256"
                                    ],
                                },
                                sort_keys=True,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                    else:
                        custody_path.write_text(f"{key}\n", encoding="utf-8")
        artifact_path.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        out[gate.gate_id] = L5V2GateEvidence(
            gate_id=gate.gate_id,
            artifact_path=str(artifact_path.relative_to(repo_root)),
            artifact_sha256=_file_sha256(artifact_path),
            predicate_id=f"l5_v2_{gate.gate_id}_predicate",
            predicate_passed=True,
            evidence_grade="contest_artifact",
        )
    return out


def _write_contest_sideinfo_proof_artifact(repo_root: Path) -> Path:
    payload = _gate_artifact_payload(
        "byte_closed_temporal_sideinfo_consumption",
        repo_root=repo_root,
    )
    proof = payload["byte_mutation_proof"]
    assert isinstance(proof, dict)
    manifest_path = repo_root / str(proof["inflated_outputs_manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "raw_output_aggregate_sha256": proof[
                    "inflated_raw_output_aggregate_sha256"
                ],
                "n_pairs_hashed": proof["n_pairs_hashed"],
                "total_frames": proof["total_frames"],
                "raw_output_frame_nbytes": proof["raw_output_frame_nbytes"],
                "file_list_sha256": proof["file_list_sha256"],
                "baseline_raw_output_aggregate_sha256": proof[
                    "baseline_raw_output_aggregate_sha256"
                ],
                "mutated_raw_output_aggregate_sha256": proof[
                    "mutated_raw_output_aggregate_sha256"
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_path = (
        repo_root / l5_v2.TT5L_CONTEST_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_PATH
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return artifact_path


def _write_committed_contest_sideinfo_proof_artifact(repo_root: Path) -> Path:
    payload = _gate_artifact_payload(
        "byte_closed_temporal_sideinfo_consumption",
        repo_root=repo_root,
    )
    proof = payload["byte_mutation_proof"]
    assert isinstance(proof, dict)
    manifest_path = repo_root / ".omx/research/test_tt5l_contest_sideinfo_manifest.json"
    proof["inflated_outputs_manifest_path"] = str(
        manifest_path.relative_to(repo_root)
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "raw_output_aggregate_sha256": proof[
                    "inflated_raw_output_aggregate_sha256"
                ],
                "n_pairs_hashed": proof["n_pairs_hashed"],
                "total_frames": proof["total_frames"],
                "raw_output_frame_nbytes": proof["raw_output_frame_nbytes"],
                "file_list_sha256": proof["file_list_sha256"],
                "baseline_raw_output_aggregate_sha256": proof[
                    "baseline_raw_output_aggregate_sha256"
                ],
                "mutated_raw_output_aggregate_sha256": proof[
                    "mutated_raw_output_aggregate_sha256"
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_path = (
        repo_root / l5_v2.TT5L_CONTEST_SIDEINFO_COMMITTED_PROOF_ARTIFACT_PATH
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return artifact_path


def _valid_gate_evidence_payloads(repo_root: Path) -> dict[str, dict[str, object]]:
    return {
        gate_id: gate_evidence.__dict__.copy()
        for gate_id, gate_evidence in _valid_gate_evidence(repo_root).items()
    }


def _write_tt5l_move_level_feasibility_artifact(repo_root: Path) -> Path:
    source_root = Path(__file__).resolve().parent.parent.parent.parent
    builder_tool_path = repo_root / l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_TOOL_PATH
    builder_tool_path.parent.mkdir(parents=True, exist_ok=True)
    builder_tool_path.write_text(
        (
            source_root / l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_TOOL_PATH
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    proof_tool_relpath = "tools/prove_tt5l_move_level_feasibility.py"
    proof_tool_path = repo_root / proof_tool_relpath
    proof_tool_path.parent.mkdir(parents=True, exist_ok=True)
    proof_tool_path.write_text(
        (source_root / proof_tool_relpath).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    proof_artifact_path = (
        repo_root
        / "experiments"
        / "results"
        / "time_traveler_l5_v2"
        / "tt5l_move_level_solver_proof.json"
    )
    proof_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    proof_payload = {
        "schema": "tt5l_move_level_solver_proof_fixture_v1",
        "subject_id": l5_v2.TT5L_DYKSTRA_SUBSTRATE_ID,
        "predicate_id": l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_PREDICATE_ID,
        "predicate_passed": True,
        "move_level_constraint_proof": True,
        "residual_max": 0.0,
        "residual_tolerance": 1e-9,
        "constraint_set_ids": sorted(l5_v2.TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    dykstra_artifact_path = repo_root / l5_v2.TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH
    mechanism_records = [
        {
            "constraint_id": constraint_id,
            "passed": True,
            "residual": 0.0,
            "details": {"fixture": True},
        }
        for constraint_id in sorted(l5_v2.TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS)
    ]
    artifact_path = repo_root / l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    proof_payload["score_axis_sanity_artifact_sha256"] = _file_sha256(
        dykstra_artifact_path
    )
    proof_payload["generated_by_tool"] = proof_tool_relpath
    proof_payload["tool_sha256"] = _file_sha256(proof_tool_path)
    proof_payload["mechanism_records"] = mechanism_records
    proof_payload["witness_variables"] = {
        constraint_id: {"fixture": True}
        for constraint_id in sorted(l5_v2.TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS)
    }
    proof_artifact_path.write_text(
        json.dumps(proof_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_path.write_text(
        json.dumps(
            {
                "schema": l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_SCHEMA,
                "subject_id": l5_v2.TT5L_DYKSTRA_SUBSTRATE_ID,
                "predicate_id": l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_PREDICATE_ID,
                "predicate_passed": True,
                "move_level_constraint_proof": True,
                "residual_max": 0.0,
                "residual_tolerance": 1e-9,
                "constraint_set_ids": sorted(
                    l5_v2.TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS
                ),
                "constraint_set_count": len(
                    l5_v2.TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS
                ),
                "proof_artifact_path": str(proof_artifact_path.relative_to(repo_root)),
                "proof_artifact_sha256": _file_sha256(proof_artifact_path),
                "score_axis_sanity_artifact_path": str(
                    dykstra_artifact_path.relative_to(repo_root)
                ),
                "score_axis_sanity_artifact_sha256": _file_sha256(
                    dykstra_artifact_path
                ),
                "generated_by_tool": l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_TOOL_PATH,
                "tool_sha256": _file_sha256(builder_tool_path),
                "generated_at_utc": "2026-05-16T00:00:00+00:00",
                "command_argv": [
                    ".venv/bin/python",
                    "experiments/solve_tt5l_move_level_constraints.py",
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_path


def _write_tt5l_first_anchor_timing_smoke_artifact(repo_root: Path) -> Path:
    result_artifact_path = (
        repo_root
        / "experiments"
        / "results"
        / "time_traveler_l5_v2"
        / "tt5l_timing_smoke_result.json"
    )
    result_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    result_artifact_path.write_text(
        json.dumps(
            {
                "schema": "tt5l_timing_smoke_result_v1",
                "lane_id": l5_v2.LANE_ID,
                "smoke_passed": True,
                "score_claim": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_path = repo_root / l5_v2.TT5L_FIRST_ANCHOR_TIMING_SMOKE_ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "schema": l5_v2.TT5L_FIRST_ANCHOR_TIMING_SMOKE_SCHEMA,
                "lane_id": l5_v2.LANE_ID,
                "predicate_id": (
                    l5_v2.TT5L_FIRST_ANCHOR_TIMING_SMOKE_PREDICATE_ID
                ),
                "predicate_passed": True,
                "required_axes": ["contest_cpu", "contest_cuda"],
                "provider": "modal",
                "hardware": "A100",
                "provider_call_id": "fc-test-tt5l",
                "command_argv": [
                    ".venv/bin/python",
                    "experiments/train_substrate_time_traveler_l5_autonomy.py",
                    "--timing-smoke",
                ],
                "elapsed_seconds": 123.0,
                "seconds_per_epoch": 12.3,
                "result_artifact_path": str(
                    result_artifact_path.relative_to(repo_root)
                ),
                "result_artifact_sha256": _file_sha256(result_artifact_path),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_path


def _tt5l_sideinfo_liveness(variant: str) -> dict[str, object]:
    nonzero = 0 if variant in {"zero", "ablated"} else 600
    return {
        "checked": True,
        "shape": [600, 45],
        "total_values": 27_000,
        "nonzero_values": nonzero,
        "nonzero_fraction": nonzero / 27_000,
    }


def _tt5l_sideinfo_cell_score(variant: str, variant_idx: int) -> tuple[float, float, int, float]:
    seg_dist = 0.001 if variant == "trained" else 0.002
    pose_dist = 0.0
    archive_bytes = 1000 + variant_idx
    return (
        seg_dist,
        pose_dist,
        archive_bytes,
        contest_score(seg_dist, pose_dist, archive_bytes),
    )


def _tt5l_sideinfo_cell_artifacts(
    repo_root: Path,
    *,
    axis: str,
    variant: str,
    variant_idx: int,
    axis_idx: int,
) -> dict[str, str]:
    base = repo_root / "experiments" / "results" / axis / variant
    artifact = base / "contest_auth_eval.json"
    log = base / "auth_eval.log"
    manifest = base / "inflated_outputs_manifest.json"
    raw_sha = _sha(90_000 + variant_idx * 10 + axis_idx)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{}\n", encoding="utf-8")
    log.write_text("ok\n", encoding="utf-8")
    return {
        "artifact_path": str(artifact.relative_to(repo_root)),
        "artifact_sha256": _file_sha256(artifact),
        "log_path": str(log.relative_to(repo_root)),
        "inflated_outputs_manifest_path": str(manifest.relative_to(repo_root)),
        "inflated_outputs_manifest_sha256": _write_aggregate_manifest(
            manifest,
            aggregate_sha=raw_sha,
        ),
        "raw_output_aggregate_sha256": raw_sha,
    }


def _tt5l_sideinfo_effect_curve_cells(repo_root: Path) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for variant_idx, variant in enumerate(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS):
        archive_sha = _sha(50_000 + variant_idx)
        runtime_content_sha = _sha(60_000 + variant_idx)
        for axis_idx, axis in enumerate(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES):
            seg_dist, pose_dist, archive_bytes, score = _tt5l_sideinfo_cell_score(
                variant,
                variant_idx,
            )
            is_cuda = axis == "contest_cuda"
            cells.append(
                {
                    "axis": axis,
                    "variant": variant,
                    "archive_sha256": archive_sha,
                    "archive_bytes": archive_bytes,
                    "runtime_tree_sha256": _sha(70_000 + variant_idx * 10 + axis_idx),
                    "runtime_content_tree_sha256": runtime_content_sha,
                    "hardware": (
                        "NVIDIA T4 CUDA GPU"
                        if is_cuda
                        else "Ubuntu x86_64 CPU"
                    ),
                    "inflate_device": "auto" if is_cuda else "cpu",
                    "eval_device": "cuda" if is_cuda else "cpu",
                    "auth_eval_command": (
                        "contest_auth_eval --device cuda"
                        if is_cuda
                        else "contest_auth_eval --device cpu"
                    ),
                    **_tt5l_sideinfo_cell_artifacts(
                        repo_root,
                        axis=axis,
                        variant=variant,
                        variant_idx=variant_idx,
                        axis_idx=axis_idx,
                    ),
                    "n_samples": 600,
                    "seg_dist": seg_dist,
                    "pose_dist": pose_dist,
                    "score": score,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "blockers": [],
                    "sideinfo_liveness": _tt5l_sideinfo_liveness(variant),
                }
            )
    return cells


def _write_tt5l_sideinfo_effect_curve_artifact(repo_root: Path) -> Path:
    artifact_path = repo_root / l5_v2.TT5L_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "schema": L5V2_SIDEINFO_EFFECT_CURVE_SCHEMA,
                "measurement_id": "measure_tt5l_sideinfo_effect_curve",
                "predicate_id": "tt5l_paired_sideinfo_effect_curve_v1",
                "predicate_passed": True,
                "required_axes": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES),
                "required_variants": list(
                    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "effect_blockers": [],
                "axis_effects": {
                    axis: {
                        "trained_score": 0.1,
                        "best_control_variant": "zero",
                        "best_control_score": 0.2,
                        "delta_vs_best_control": 0.1,
                        "trained_beats_or_ties_best_control": True,
                    }
                    for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES
                },
                "observed_cells": _tt5l_sideinfo_effect_curve_cells(repo_root),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_path


def _write_tt5l_partial_sideinfo_effect_curve_artifact(repo_root: Path) -> Path:
    artifact_path = repo_root / l5_v2.TT5L_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    cell_artifacts = _tt5l_sideinfo_cell_artifacts(
        repo_root,
        axis="contest_cuda",
        variant="trained",
        variant_idx=3,
        axis_idx=1,
    )
    artifact_path.write_text(
        json.dumps(
            {
                "schema": L5V2_SIDEINFO_EFFECT_CURVE_SCHEMA,
                "measurement_id": "measure_tt5l_sideinfo_effect_curve",
                "predicate_id": "tt5l_paired_sideinfo_effect_curve_v1",
                "predicate_passed": False,
                "required_axes": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES),
                "required_variants": list(
                    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "effect_blockers": [
                    "trained_not_best_or_tied:contest_cpu",
                    "trained_not_best_or_tied:contest_cuda",
                ],
                "contract_blockers": [
                    "tt5l_sideinfo_effect_curve_predicate_not_passed",
                    (
                        "tt5l_sideinfo_effect_curve_cells_missing:"
                        "contest_cpu/trained,contest_cuda/zero"
                    ),
                    (
                        "tt5l_sideinfo_effect_curve_cell_sideinfo_nonzero_missing:"
                        "contest_cuda:trained"
                    ),
                ],
                "axis_effects": {
                    "contest_cpu": {
                        "trained_score": None,
                        "best_control_variant": "",
                        "best_control_score": None,
                        "delta_vs_best_control": None,
                        "trained_beats_or_ties_best_control": False,
                    },
                    "contest_cuda": {
                        "trained_score": 3.9007398365396795,
                        "best_control_variant": "",
                        "best_control_score": None,
                        "delta_vs_best_control": None,
                        "trained_beats_or_ties_best_control": False,
                    },
                },
                "observed_cells": [
                    {
                        "axis": "contest_cuda",
                        "variant": "trained",
                        "archive_sha256": _sha(50_003),
                        "archive_bytes": 34_603,
                        "runtime_tree_sha256": _sha(70_031),
                        "runtime_content_tree_sha256": _sha(60_003),
                        "hardware": "NVIDIA T4 CUDA GPU",
                        "inflate_device": "auto",
                        "eval_device": "cuda",
                        "auth_eval_command": "contest_auth_eval --device cuda",
                        **cell_artifacts,
                        "n_samples": 600,
                        "seg_dist": 0.02515214,
                        "pose_dist": 0.18563657,
                        "score": 3.9007398365396795,
                        "score_claim": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                        "blockers": [],
                        "sideinfo_liveness": {
                            "checked": True,
                            "shape": [600, 45],
                            "total_values": 27_000,
                            "nonzero_values": 0,
                            "nonzero_fraction": 0.0,
                            "total_pairs": 600,
                            "nonzero_pair_count": 0,
                        },
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_path


def _write_tt5l_sideinfo_effect_curve_dispatch_plan(repo_root: Path) -> Path:
    artifact_path = (
        repo_root / l5_v2.TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_ARTIFACT_PATH
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    work_units = []
    for idx, variant in enumerate(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS):
        archive_sha = _sha(80_000 + idx)
        work_units.append(
            {
                "variant": variant,
                "work_unit_id": f"measure_tt5l_sideinfo_effect_curve__{variant}",
                "lane_id": f"lane_l5_v2_tt5l_sideinfo_effect_curve_{variant}",
                "pair_group_id": (
                    f"pair_l5_v2_tt5l_sideinfo_effect_curve_{variant}_{archive_sha[:12]}"
                ),
                "archive": {
                    "path": f"experiments/results/{variant}/archive.zip",
                    "sha256": archive_sha,
                    "bytes": 34_000 + idx,
                },
                "required_axes": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES),
                "required_cells": [
                    {"axis": axis, "variant": variant}
                    for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES
                ],
                "ready_for_operator_dispatch": True,
                "ready_for_provider_dispatch": False,
                "dispatch_blockers": [],
                "score_claim_blockers": [
                    "requires_paired_cpu_cuda_exact_eval_for_sideinfo_effect_curve"
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
            }
        )
    artifact_path.write_text(
        json.dumps(
            {
                "schema": l5_v2.TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_SCHEMA,
                "plan_id": "l5_v2_tt5l_sideinfo_effect_curve_dispatch_test",
                "measurement_id": "measure_tt5l_sideinfo_effect_curve",
                "required_axes": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES),
                "required_variants": list(
                    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
                ),
                "work_unit_count": len(work_units),
                "ready_work_unit_count": len(work_units),
                "work_units": work_units,
                "ready_for_operator_dispatch": True,
                "ready_for_provider_dispatch": False,
                "dispatch_attempted": False,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_path


def _tt5l_lightning_paired_axis_plan_cell(
    *,
    variant: str,
    axis: str,
    seed: int,
) -> dict[str, object]:
    required_device = "cpu" if axis == "contest_cpu" else "cuda"
    role = f"exact_{required_device}_eval"
    return {
        "variant": variant,
        "axis": axis,
        "role": role,
        "required_device": required_device,
        "archive_sha256": _sha(100 + seed),
        "command_sha256": _sha(200 + seed),
        "state_sha256": _sha(300 + seed),
        "dry_run_stdout_sha256": _sha(400 + seed),
        "dry_run_stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "ready_for_operator_dispatch": True,
        "ready_for_provider_dispatch": False,
        "invariants": {
            "status_dry_run": True,
            "role_matches_axis": True,
            "adjudication_required_device_matches_axis": True,
            "command_contains_expected_device": True,
            "command_omits_opposite_device": True,
            "cuda_require_marker_matches_axis": True,
            "queue_metadata_axis_matches_cell": True,
            "stderr_empty": True,
        },
        "spec": {
            "role": role,
            "adjudication": {"required_device": required_device},
        },
    }


def _tt5l_lightning_paired_axis_plan_payload() -> dict[str, object]:
    cells: list[dict[str, object]] = []
    seed = 0
    for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS:
        for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES:
            seed += 1
            cells.append(
                _tt5l_lightning_paired_axis_plan_cell(
                    variant=variant,
                    axis=axis,
                    seed=seed,
                )
            )
    return {
        "schema": (
            l5_v2.TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_SCHEMA
        ),
        "source_commit": _sha(1),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_provider_dispatch": False,
        "dispatch_attempted": False,
        "all_cells_dry_run_ready": True,
        "blockers": [
            "dry_run_only_no_provider_job_launched",
            "requires_lightning_identity_and_workspace_preflight_before_submit",
        ],
        "cells": cells,
    }


def _write_tt5l_lightning_paired_axis_plan_artifact(repo_root: Path) -> Path:
    artifact_path = (
        repo_root
        / l5_v2.TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(_tt5l_lightning_paired_axis_plan_payload(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact_path


def _write_tt5l_dykstra_artifact(
    repo_root: Path,
    *,
    write_move_level: bool = True,
) -> Path:
    tool_path = repo_root / l5_v2.TT5L_DYKSTRA_FEASIBILITY_TOOL_PATH
    tool_path.parent.mkdir(parents=True, exist_ok=True)
    tool_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    artifact_path = repo_root / l5_v2.TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "schema": l5_v2.TT5L_DYKSTRA_FEASIBILITY_SCHEMA,
                "predicate_id": l5_v2.TT5L_DYKSTRA_FEASIBILITY_PREDICATE_ID,
                "generated_by_tool": (
                    l5_v2.TT5L_DYKSTRA_FEASIBILITY_GENERATED_BY_TOOL
                ),
                "generated_at_utc": "2026-05-16T00:00:00+00:00",
                "command_argv": [
                    l5_v2.TT5L_DYKSTRA_FEASIBILITY_GENERATED_BY_TOOL,
                    "--tt5l-five-move-polytope",
                ],
                "tool_sha256": _file_sha256(tool_path),
                "substrate_id": l5_v2.TT5L_DYKSTRA_SUBSTRATE_ID,
                "verdict": "FEASIBLE",
                "tested_score_axis_band": [0.150, 0.170],
                "input_band_role": "planning_band_not_score_or_rank_authority",
                "archive_size_bytes": 34_603,
                "rate_contribution": 0.05,
                "seg_budget": 0.001,
                "pose_budget": 0.011,
                "feasibility_band_lo": 0.150,
                "feasibility_band_hi": 0.170,
                "feasibility_rationale": (
                    "test fixture: TT5L retired additive band intersects "
                    "planning-only Dykstra feasibility interval"
                ),
                "blocker_axis": None,
                "dykstra_iteration_count": 1,
                "score_formula": l5_v2.TT5L_DYKSTRA_SCORE_FORMULA,
                "contest_seg_multiplier": 100.0,
                "constraint_set_ids": sorted(
                    l5_v2.TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS
                ),
                "constraint_set_count": len(
                    l5_v2.TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS
                ),
                "polytope_projection_kind": l5_v2.TT5L_DYKSTRA_PROJECTION_KIND,
                "feasibility_scope": l5_v2.TT5L_DYKSTRA_FEASIBILITY_SCOPE,
                "verdict_authority_scope": (
                    l5_v2.TT5L_DYKSTRA_VERDICT_AUTHORITY_SCOPE
                ),
                "move_level_constraint_proof": False,
                "projection_limitations": (
                    "test fixture: score-axis projection only; not score authority"
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if write_move_level:
        _write_tt5l_move_level_feasibility_artifact(repo_root)
    return artifact_path


def test_l5_v2_research_basis_is_explicit_and_canonical() -> None:
    ids = l5_v2_research_basis_ids()

    assert ids == (
        "rao_ballard_1999",
        "friston_free_energy_2010",
        "dreamerv3_2023",
        "ha_schmidhuber_world_models_2018",
        "vjepa2_2025",
        "vjepa2_1_dense_2026",
        "teconerv_2026",
        "neural_dsc_2021",
        "dcvc_2021",
        "fvc_2021",
        "scale_space_flow_2020",
        "video_interpolation_codec_2018",
        "pnvc_2025",
        "dcvc_rt_2025",
        "unified_intra_inter_nvc_2025",
        "glvc_2025",
        "gnvc_vd_2025",
        "snerv_spectra_2025",
        "metanerv_2025",
        "c3_neural_compression_2024",
        "atick_redlich_1990",
        "wyner_ziv_1976",
        "slepian_wolf_1973",
        "lu_dvc_2019",
        "rissanen_mdl_1978",
        "mackay_itila_2003",
        "tishby_information_bottleneck_1999",
        "tishby_zaslavsky_2015",
        "balle_hyperprior_2018",
        "elic_2022",
        "checkerboard_context_2021",
        "rdp_tradeoff_2019",
        "coin_2021",
        "hnerv_2023",
    )


def test_l5_v2_asymptotic_pursuit_candidates_are_source_backed() -> None:
    payload = l5_v2.l5_v2_asymptotic_pursuit_candidates()

    assert payload["schema"] == (
        l5_v2.L5_V2_ASYMPTOTIC_PURSUIT_CANDIDATES_SCHEMA
    )
    assert payload["candidate_ids"] == [
        "z6_z7_z8_predictive_coding_world_models",
        "rudin_floor_interpretable_ml_substrate",
        "tishby_ib_pure_substrate",
    ]
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["ready_for_paid_dispatch"] is False
    assert len(payload["l5_v2_asymptotic_next_action_status"]) == 3

    rows = {row["candidate_id"]: row for row in payload["candidates"]}
    z6 = rows["z6_z7_z8_predictive_coding_world_models"]
    z6_status = z6["l5_v2_asymptotic_next_action_status"]
    assert z6["recommended_next_action_id"] == "build_z6_l1_scaffold_first"
    assert z6["ready_for_recommended_next_action"] is False
    assert z6["recommended_next_action_status"] == "completed_or_superseded"
    assert z6["effective_recommended_next_action_id"] == (
        "completed_or_superseded:build_z6_l1_scaffold_first"
    )
    assert z6["ready_for_l1_build"] is False
    assert z6["l1_scaffold_present"] is True
    assert z6["recommended_next_action_completed_or_superseded"] is True
    assert z6["local_ledger_present"] is True
    assert z6["lane_registry_registered"] is True
    assert z6["local_ledger_sha256"]
    assert z6["expected_first_artifacts_all_present"] is True
    assert z6["ready_for_l1_scaffold_dispatch"] is False
    assert z6["ready_for_l1_build_semantics"] == (
        "l1_scaffold_present_next_action_completed"
    )
    assert z6_status["schema"] == l5_v2.L5_V2_ASYMPTOTIC_NEXT_ACTION_STATUS_SCHEMA
    assert z6_status["candidate_id"] == z6["candidate_id"]
    assert z6_status["ledger_present"] is True
    assert z6_status["ledger_sha256"] == z6["local_ledger_sha256"]
    assert z6_status["lane_registry_registered"] is True
    assert z6_status["expected_first_artifacts_all_present"] is True
    assert z6_status["ready_for_l1_build_semantics"] == (
        "l1_scaffold_present_next_action_completed"
    )
    assert z6_status["next_prerequisite_status"]["status"] == (
        "completed_or_superseded"
    )
    assert z6_status["next_prerequisite_status"]["ready_for_l1_build"] is False
    assert z6["post_l1_proxy_evidence"]["schema"] == (
        l5_v2.Z6_POST_L1_PROXY_EVIDENCE_STATUS_SCHEMA
    )
    assert z6["post_l1_proxy_evidence"]["artifact_present"] is True
    assert z6["post_l1_proxy_evidence"]["artifact_valid"] is True
    assert z6["post_l1_proxy_evidence"]["verdict"] == (
        l5_v2.Z6_REAL_VIDEO_EGO_PROXY_SWEEP_FULL_FILM_VERDICT
    )
    assert (
        z6["post_l1_proxy_evidence"]["paired_control_initialization"]
        == "shared_modules_seed_order_matched_v2"
    )
    assert z6["post_l1_proxy_evidence"]["full_film_proxy_found"] is True
    assert z6["post_l1_proxy_evidence"]["posenet_proxy_tested"] is True
    assert z6["post_l1_proxy_evidence"]["semantic_ego_proxy_supported"] is False
    scorer_smoke = z6["post_l1_proxy_evidence"]["scorer_bearing_paired_smoke"]
    assert scorer_smoke["artifact_present"] is True
    assert scorer_smoke["schema"] == l5_v2.Z6_SCORER_BEARING_PAIRED_SMOKE_SCHEMA
    assert scorer_smoke["best_proxy_id"] == "random_control"
    assert scorer_smoke["semantic_scorer_proxy_supported"] is False
    assert scorer_smoke["score_claim"] is False
    assert scorer_smoke["promotion_eligible"] is False
    assert scorer_smoke["ready_for_paid_dispatch"] is False
    assert z6["post_l1_proxy_evidence"]["score_claim"] is False
    assert z6["post_l1_proxy_evidence"]["ready_for_paid_dispatch"] is False
    assert (
        z6["post_l1_proxy_evidence"]["allowed_to_spend_on_z6_full_film"]
        is False
    )
    assert z6["post_l1_recommended_next_action_status"] == (
        "scorer_bearing_probe_blocks_z6_v1_paid_dispatch"
    )
    assert z6["post_l1_recommended_next_action_id"] == (
        "z6_scorer_bearing_probe_blocks_full_film_paid_dispatch"
    )
    assert z6_status["post_l1_proxy_evidence"] == z6["post_l1_proxy_evidence"]
    assert z6_status["next_prerequisite_status"][
        "post_l1_recommended_next_action_status"
    ] == "scorer_bearing_probe_blocks_z6_v1_paid_dispatch"
    assert "time_traveler_l5_z6" in "\n".join(
        z6["expected_first_artifacts"]
    )
    assert l5_v2.Z6_IDENTITY_PREDICTOR_DISAMBIGUATOR_TOOL_PATH in z6[
        "expected_first_artifacts"
    ]
    assert "l1_scaffold_present_next_action_completed_or_superseded" in z6[
        "l1_build_blockers"
    ]
    assert "requires_z6_l1_scaffold_before_paid_dispatch" in z6["blockers"]
    assert (
        "requires_identity_predictor_disambiguator_result_before_paradigm_claim"
        in z6["blockers"]
    )
    assert (
        "z6_full_film_paid_dispatch_blocked_ego_proxy_semantics_not_hard_earned"
        in z6["blockers"]
    )
    assert (
        "z6_full_film_paid_dispatch_blocked_posenet_pose_proxy_not_best"
        in z6["blockers"]
    )
    assert (
        "z6_full_film_paid_dispatch_blocked_scorer_bearing_semantics_not_hard_earned"
        in z6["blockers"]
    )
    assert (
        "z6_full_film_paid_dispatch_blocked_posenet_pose_scorer_proxy_not_best"
        in z6["blockers"]
    )

    rudin = rows["rudin_floor_interpretable_ml_substrate"]
    assert rudin["ready_for_recommended_next_action"] is False
    assert rudin["recommended_next_action_status"] == "completed_or_superseded"
    assert rudin["ready_for_l1_build"] is False
    assert rudin["l1_scaffold_present"] is True
    assert "rudin_floor_interpretable_ml" in "\n".join(
        rudin["expected_first_artifacts"]
    )
    assert "l1_scaffold_present_next_action_completed_or_superseded" in rudin[
        "l1_build_blockers"
    ]
    assert rudin["post_l1_probe_evidence"]["schema"] == (
        l5_v2.RUDIN_POST_L1_PROBE_EVIDENCE_STATUS_SCHEMA
    )
    assert rudin["post_l1_probe_evidence"]["artifact_present"] is True
    assert rudin["post_l1_probe_evidence"]["artifact_valid"] is True
    assert rudin["post_l1_probe_evidence"]["verdict"] == (
        l5_v2.RUDIN_MEANINGFUL_INTERPRETABILITY_VERDICT
    )
    assert rudin["post_l1_probe_evidence"]["allowed_to_spend"] is False
    assert rudin["post_l1_recommended_next_action_status"] == (
        "proxy_positive_requires_ratification"
    )
    assert rudin["post_l1_recommended_next_action_id"] == (
        "run_rudin_t3_ratification_scorer_probe_before_paid_dispatch"
    )
    assert (
        "rudin_proxy_positive_requires_t3_ratification_and_scorer_probe"
        in rudin["blockers"]
    )

    tishby = rows["tishby_ib_pure_substrate"]
    assert tishby["ready_for_recommended_next_action"] is False
    assert tishby["recommended_next_action_status"] == "completed_or_superseded"
    assert tishby["ready_for_l1_build"] is False
    assert tishby["l1_scaffold_present"] is True
    assert "tishby_ib_pure" in "\n".join(tishby["expected_first_artifacts"])
    assert "l1_scaffold_present_next_action_completed_or_superseded" in tishby[
        "l1_build_blockers"
    ]
    assert tishby["post_l1_probe_evidence"]["schema"] == (
        l5_v2.TISHBY_POST_L1_PROBE_EVIDENCE_STATUS_SCHEMA
    )
    assert tishby["post_l1_probe_evidence"]["artifact_present"] is True
    assert tishby["post_l1_probe_evidence"]["artifact_valid"] is True
    assert tishby["post_l1_probe_evidence"]["d4_verdict"] == (
        l5_v2.TISHBY_D4_INDEPENDENT_VERDICT
    )
    assert tishby["post_l1_probe_evidence"]["vib_tractability_verdict"] == (
        l5_v2.TISHBY_VIB_TRACTABLE_VERDICT
    )
    assert tishby["post_l1_probe_evidence"]["allowed_to_spend"] is False
    assert tishby["post_l1_recommended_next_action_status"] == (
        "blocked_pending_conditioning_or_mine"
    )
    assert tishby["post_l1_recommended_next_action_id"] == (
        "advance_tishby_only_with_meaningful_conditioning_or_mine_beta_sweep"
    )
    assert (
        "tishby_path_vib_paid_dispatch_blocked_d4_independent_scorer_class_probe"
        in tishby["blockers"]
    )

    for row in rows.values():
        assert row["horizon_class"] == "asymptotic_pursuit"
        assert row["score_claim"] is False
        assert row["promotion_eligible"] is False
        assert row["rank_or_kill_eligible"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
        assert row["ready_for_paid_dispatch"] is False


def test_l5_v2_asymptotic_candidate_surface_artifact_tracks_live_payload() -> None:
    live = l5_v2.l5_v2_asymptotic_pursuit_candidates()
    artifact_path = REPO_ROOT / L5_V2_ASYMPTOTIC_CANDIDATE_SURFACE_ARTIFACT_PATH
    assert L5_V2_ASYMPTOTIC_CANDIDATE_SURFACE_REPORT_PATH.endswith(".md")
    assert artifact_path.is_file()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact["schema"] == live["schema"]
    assert artifact["candidate_ids"] == live["candidate_ids"]
    assert artifact["score_claim"] is False
    assert artifact["promotion_eligible"] is False
    assert artifact["ready_for_exact_eval_dispatch"] is False
    assert artifact["ready_for_paid_dispatch"] is False
    artifact_rows = {row["candidate_id"]: row for row in artifact["candidates"]}
    live_rows = {row["candidate_id"]: row for row in live["candidates"]}
    for candidate_id, live_row in live_rows.items():
        artifact_row = artifact_rows[candidate_id]
        for field in (
            "expected_first_artifacts",
            "expected_first_artifact_status",
            "expected_first_artifacts_all_present",
            "l1_scaffold_present",
            "recommended_next_action_status",
            "effective_recommended_next_action_id",
            "ready_for_recommended_next_action",
            "ready_for_l1_build",
            "ready_for_l1_scaffold_dispatch",
            "l1_build_blockers",
            "blockers",
            "post_l1_probe_evidence",
            "post_l1_proxy_evidence",
            "post_l1_recommended_next_action_status",
            "post_l1_recommended_next_action_id",
            "post_l1_recommended_next_action",
        ):
            assert artifact_row[field] == live_row[field]


def test_l5_v2_asymptotic_candidate_surface_markdown_reports_current_status() -> None:
    payload = l5_v2.l5_v2_asymptotic_pursuit_candidates()

    report = render_l5_v2_asymptotic_candidate_surface_markdown(payload)

    assert "# L5 v2 asymptotic candidate surface" in report
    assert "z6_z7_z8_predictive_coding_world_models" in report
    assert "expected_first_artifacts_all_present: `True`" in report
    assert "recommended_next_action_status: `completed_or_superseded`" in report
    assert (
        "post_l1_recommended_next_action_status: "
        "`scorer_bearing_probe_blocks_z6_v1_paid_dispatch`"
    ) in report
    assert "full_film_proxy_found_real_video_smoke" in report
    assert "MEANINGFUL_INTERPRETABILITY" in report
    assert "proxy_positive_requires_ratification" in report
    assert "blocked_pending_conditioning_or_mine" in report
    assert "d4=INDEPENDENT;vib=TRACTABLE" in report
    assert "score_claim: `false`" in report


def test_l5_v2_asymptotic_candidate_surface_cli_writes_json_and_markdown(
    tmp_path: Path,
) -> None:
    artifact_root = (
        REPO_ROOT
        / "experiments"
        / "results"
        / "time_traveler_l5_v2"
        / f"test_asymptotic_candidate_surface_{tmp_path.name}"
    )
    output_json = artifact_root / "surface.json"
    output_md = artifact_root / "surface.md"
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "tools/build_l5_v2_asymptotic_candidate_surface.py",
                "--output-json",
                str(output_json.relative_to(REPO_ROOT)),
                "--output-md",
                str(output_md.relative_to(REPO_ROOT)),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert output_json.is_file()
        assert output_md.is_file()
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert payload["candidate_count"] == 3
        assert payload["ready_for_paid_dispatch"] is False
        assert "score_claim=false" in proc.stdout
    finally:
        if artifact_root.exists():
            shutil.rmtree(artifact_root)


def test_l5_v2_asymptotic_pursuit_candidates_fail_closed_without_ledgers(
    tmp_path: Path,
) -> None:
    payload = l5_v2.l5_v2_asymptotic_pursuit_candidates(repo_root=tmp_path)

    assert payload["candidate_count"] == 3
    assert payload["ready_for_paid_dispatch"] is False
    assert all(row["local_ledger_present"] is False for row in payload["candidates"])
    assert all(
        row["lane_registry_registered"] is False for row in payload["candidates"]
    )
    assert all(
        row["ready_for_recommended_next_action"] is False
        for row in payload["candidates"]
    )
    assert all(row["ready_for_l1_build"] is False for row in payload["candidates"])
    assert (
        "l5_v2_asymptotic_pursuit_ledger_missing:"
        "z6_z7_z8_predictive_coding_world_models"
    ) in payload["blockers"]
    assert (
        "l5_v2_asymptotic_pursuit_lane_registry_missing:"
        "z6_z7_z8_predictive_coding_world_models:"
        "lane_time_traveler_l5_z6_z7_z8_predictive_coding_world_models_"
        "scoping_design_20260516"
    ) in payload["blockers"]
    z6 = next(
        row
        for row in payload["candidates"]
        if row["candidate_id"] == "z6_z7_z8_predictive_coding_world_models"
    )
    assert z6["post_l1_proxy_evidence"]["artifact_present"] is False
    assert z6["post_l1_proxy_evidence"]["artifact_valid"] is False
    assert "z6_real_video_ego_proxy_sweep_missing" in z6["blockers"]
    tishby = next(
        row
        for row in payload["candidates"]
        if row["candidate_id"] == "tishby_ib_pure_substrate"
    )
    assert tishby["post_l1_probe_evidence"]["artifact_present"] is False
    assert tishby["post_l1_probe_evidence"]["artifact_valid"] is False
    assert "tishby_d4_probe_missing" in tishby["blockers"]
    assert "tishby_vib_tractability_missing" in tishby["blockers"]
    rudin = next(
        row
        for row in payload["candidates"]
        if row["candidate_id"] == "rudin_floor_interpretable_ml_substrate"
    )
    assert rudin["post_l1_probe_evidence"]["artifact_present"] is False
    assert rudin["post_l1_probe_evidence"]["artifact_valid"] is False
    assert "rudin_proxy_disambiguator_missing" in rudin["blockers"]


def test_l5_v2_asymptotic_pursuit_candidates_require_registry_with_ledger(
    tmp_path: Path,
) -> None:
    ledger_rel = Path(
        ".omx/research/"
        "time_traveler_l5_z6_z7_z8_predictive_coding_world_models_"
        "asymptotic_pursuit_scoping_design_20260516.md"
    )
    ledger_path = tmp_path / ledger_rel
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text("# Z6 source ledger\n", encoding="utf-8")

    payload = l5_v2.l5_v2_asymptotic_pursuit_candidates(repo_root=tmp_path)
    rows = {row["candidate_id"]: row for row in payload["candidates"]}
    z6 = rows["z6_z7_z8_predictive_coding_world_models"]

    assert z6["local_ledger_present"] is True
    assert z6["local_ledger_sha256"]
    assert z6["lane_registry_registered"] is False
    assert z6["l5_v2_asymptotic_next_action_status"]["ledger_present"] is True
    assert z6["l5_v2_asymptotic_next_action_status"]["ledger_sha256"] == (
        z6["local_ledger_sha256"]
    )
    assert (
        z6["l5_v2_asymptotic_next_action_status"]["lane_registry_registered"]
        is False
    )
    assert z6["ready_for_recommended_next_action"] is False
    assert z6["ready_for_l1_build"] is False
    assert z6["ready_for_l1_scaffold_dispatch"] is False
    assert (
        "l5_v2_asymptotic_pursuit_lane_registry_missing:"
        "z6_z7_z8_predictive_coding_world_models:"
        "lane_time_traveler_l5_z6_z7_z8_predictive_coding_world_models_"
        "scoping_design_20260516"
    ) in z6["blockers"]
    assert z6["post_l1_proxy_evidence"]["artifact_present"] is False
    assert z6["post_l1_proxy_evidence"]["artifact_valid"] is False
    assert "z6_real_video_ego_proxy_sweep_missing" in z6["blockers"]
    assert "requires_l5_v2_asymptotic_pursuit_lane_registry_entry" in z6[
        "l1_build_blockers"
    ]


def test_l5_v2_staircase_steps_are_ordered_and_fail_closed() -> None:
    steps = l5_v2_staircase_steps()
    gate_ids = {gate.gate_id for gate in l5_v2_required_gates()}

    assert [step.step_id for step in steps] == [
        "l5v2_00_source_and_alias_custody",
        "l5v2_01_dykstra_score_axis_sanity",
        "l5v2_02_sideinfo_consumption_proof",
        "l5v2_03_probe_disambiguator",
        "l5v2_04_paired_axis_anchor",
        "l5v2_05_stack_of_stacks_candidate",
    ]
    for step in steps:
        assert step.research_basis_ids
        assert step.dispatch_allowed is False
        assert step.promotion_eligible is False
        assert set(step.required_gate_ids) <= gate_ids

    dykstra_step = next(
        step for step in steps if step.step_id == "l5v2_01_dykstra_score_axis_sanity"
    )
    assert l5_v2.TT5L_DYKSTRA_FEASIBILITY_TOOL_PATH in dykstra_step.deliverable_surface
    assert l5_v2.TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH in dykstra_step.deliverable_surface

    probe_step = next(step for step in steps if step.step_id == "l5v2_03_probe_disambiguator")
    assert probe_step.deliverable_surface == L5V2_PROBE_TOOL_PATH
    assert Path(L5V2_PROBE_TOOL_PATH).is_file()

    anchor_step = next(step for step in steps if step.step_id == "l5v2_04_paired_axis_anchor")
    assert (
        l5_v2.TT5L_FIRST_ANCHOR_TIMING_SMOKE_TOOL_PATH
        in anchor_step.deliverable_surface
    )
    assert (
        l5_v2.TT5L_FIRST_ANCHOR_TIMING_SMOKE_ARTIFACT_PATH
        in anchor_step.deliverable_surface
    )


def test_l5_v2_prediction_band_is_source_backed_but_rank_blocked() -> None:
    payload = l5_v2_prediction_band_payload()
    verdict = l5_v2_prediction_band_verdict()

    assert payload["subject_id"] == SUBJECT_ID
    assert (payload["low"], payload["high"]) == PREDICTED_DELTA_BAND
    assert payload["score_claim"] is False
    assert payload["planning_only"] is True
    assert payload["band_source"]["research_basis_ids"] == l5_v2_research_basis_ids()
    ledger_paths = set(payload["band_source"]["local_ledger_paths"])
    assert (
        "file:.omx/research/l5_v2_latest_source_basis_wirein_20260516_codex.md"
        in ledger_paths
    )
    assert (
        "file:.omx/research/tt5l_sideinfo_consumption_proof_20260516_codex.md"
        in ledger_paths
    )
    assert verdict["valid_for_dispatch_planning"] is True
    assert verdict["valid_for_rank_reward"] is False
    assert "prediction_band_baseline_missing" in verdict["blockers"]
    assert "prediction_band_empirical_anchor_missing" in verdict["blockers"]
    assert "prediction_band_research_basis_missing" not in verdict["blockers"]


def test_l5_v2_prediction_band_status_preserves_diagnostic_anchor_pair(
    tmp_path: Path,
) -> None:
    payload = _gate_artifact_payload(
        "exact_anchor_or_diagnostic_pair",
        repo_root=tmp_path,
    )
    payload["schema"] = "l5_v2_tt5l_paired_exact_anchor_pair_v1"
    payload["predicate_id"] = l5_v2.TT5L_PAIRED_EXACT_ANCHOR_PAIR_PREDICATE_ID
    payload["pair_group_id"] = "pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda"
    payload["classification"] = "paired_exact_measured_config_failure_non_promotional_anchor"
    payload["archive_sha256"] = _sha(11)
    payload["runtime_content_tree_sha256"] = _sha(13)
    payload["score_claim"] = False
    payload["promotion_eligible"] = False
    payload["rank_or_kill_eligible"] = False
    payload["ready_for_exact_eval_dispatch"] = False
    rows = payload["anchor_pair"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        row["promotion_eligible"] = False
        row["rank_or_kill_eligible"] = False
        row["ready_for_exact_eval_dispatch"] = False
        for key in ("artifact_path", "log_path", "inflated_outputs_manifest_path"):
            custody_path = tmp_path / str(row[key])
            custody_path.parent.mkdir(parents=True, exist_ok=True)
            if key == "inflated_outputs_manifest_path":
                custody_path.write_text(
                    json.dumps(
                        {"aggregate_sha256": row["inflated_raw_output_aggregate_sha256"]},
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                custody_path.write_text(
                    f"prediction band diagnostic anchor {row['axis']} {key}\n",
                    encoding="utf-8",
                )

    anchor_path = tmp_path / l5_v2.TT5L_PAIRED_EXACT_ANCHOR_PAIR_ARTIFACT_PATH
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    anchor_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    status = l5_v2_prediction_band_status(repo_root=tmp_path)
    diagnostic = status["diagnostic_anchor_pair_status"]
    packet = l5_v2_architecture_lock_packet(repo_root=tmp_path)
    report = render_l5_v2_architecture_lock_packet_markdown(packet)

    assert status["rank_reward_allowed"] is False
    assert "prediction_band_baseline_missing" in status["blockers"]
    assert "prediction_band_empirical_anchor_missing" in status["blockers"]
    assert status["diagnostic_anchor_preserved_but_not_rankable"] is True
    assert diagnostic["artifact_valid"] is True
    assert diagnostic["paired_axes"] == ["contest_cpu", "contest_cuda"]
    assert diagnostic["score_claim"] is False
    assert diagnostic["promotion_eligible"] is False
    assert diagnostic["rank_or_kill_eligible"] is False
    assert packet["prediction_band_status"]["diagnostic_anchor_pair_status"][
        "artifact_valid"
    ] is True
    assert "## Prediction Band" in report
    assert "- diagnostic_anchor_pair_valid: `True`" in report
    assert "paired_exact_measured_config_failure_non_promotional_anchor" in report


def test_l5_v2_packetir_stack_evidence_is_axis_labelled_and_nonpromotional() -> None:
    payload = l5_v2_packetir_stack_evidence_payload()

    assert payload["schema"] == L5_V2_PACKETIR_STACK_EVIDENCE_SCHEMA
    assert payload["source_matrix_artifact_sha256"] == (
        PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_SHA256
    )
    assert payload["source_candidate_count"] == 16
    assert payload["paired_candidate_count"] == 0
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["paired_candidates"] == []
    assert "l5_v2_packetir_no_runtime_bound_paired_exact_candidates" in payload[
        "blockers"
    ]


def test_l5_v2_packetir_matrix_sha_pin_matches_committed_artifact() -> None:
    matrix_path = Path(l5_v2.PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_PATH)

    assert matrix_path.is_file()
    assert _file_sha256(matrix_path) == PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_SHA256


def test_l5_v2_packetir_section_entropy_evidence_is_nonpromotional() -> None:
    payload = l5_v2_packetir_section_entropy_evidence_payload()

    assert payload["schema"] == L5_V2_PACKETIR_SECTION_ENTROPY_EVIDENCE_SCHEMA
    assert payload["source_matrix_artifact_sha256"] == (
        L5_V2_PACKETIR_SECTION_ENTROPY_MATRIX_ARTIFACT_SHA256
    )
    assert payload["profiled_candidate_count"] == 2
    assert payload["prototype_row_count"] == 12
    assert payload["rate_positive_prototype_row_count"] == 0
    assert payload["adaptive_prototype_row_count"] == 4
    assert payload["rate_positive_adaptive_prototype_row_count"] == 0
    assert payload["derived_prefix_adaptive_prototype_row_count"] == 4
    assert payload["rate_positive_derived_prefix_adaptive_prototype_row_count"] == 2
    assert payload["best_rate_positive_prototype"] is None
    assert payload["best_adaptive_prototype"]["delta_bytes_vs_source_section"] == 1
    assert payload["best_rate_positive_adaptive_prototype"] is None
    assert (
        payload["best_rate_positive_derived_prefix_adaptive_prototype"][
            "delta_bytes_vs_source_section"
        ]
        == -1
    )
    assert payload["best_charged_prototype"]["delta_bytes_vs_source_section"] == (
        58284.0
    )
    assert "l5_v2_packetir_static_context_recode_no_rate_positive_prototypes" in (
        payload["blockers"]
    )
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_l5_v2_packetir_section_entropy_matrix_sha_pin_matches_artifact() -> None:
    matrix_path = Path(L5_V2_PACKETIR_SECTION_ENTROPY_MATRIX_ARTIFACT_PATH)

    assert matrix_path.is_file()
    assert (
        _file_sha256(matrix_path)
        == L5_V2_PACKETIR_SECTION_ENTROPY_MATRIX_ARTIFACT_SHA256
    )


def test_l5_v2_packetir_section_entropy_evidence_fails_closed_without_matrix(
    tmp_path: Path,
) -> None:
    payload = l5_v2_packetir_section_entropy_evidence_payload(repo_root=tmp_path)

    assert payload["profiled_candidate_count"] == 0
    assert payload["prototype_row_count"] == 0
    assert payload["rate_positive_prototype_row_count"] == 0
    assert "l5_v2_packetir_section_entropy_matrix_artifact_missing" in payload[
        "blockers"
    ]
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_l5_v2_dispatch_readiness_surfaces_section_entropy_evidence() -> None:
    readiness = l5_v2_dispatch_readiness()
    entropy = readiness["packetir_section_entropy_evidence"]

    assert entropy["schema"] == L5_V2_PACKETIR_SECTION_ENTROPY_EVIDENCE_SCHEMA
    assert entropy["score_claim"] is False
    assert entropy["promotion_eligible"] is False
    assert entropy["ready_for_exact_eval_dispatch"] is False
    assert entropy["rate_positive_prototype_row_count"] == 0
    assert entropy["adaptive_prototype_row_count"] == 4
    assert entropy["best_adaptive_prototype"]["delta_bytes_vs_source_section"] == 1
    assert entropy["derived_prefix_adaptive_prototype_row_count"] == 4
    assert entropy["rate_positive_derived_prefix_adaptive_prototype_row_count"] == 2
    assert (
        entropy["best_rate_positive_derived_prefix_adaptive_prototype"][
            "delta_bytes_vs_source_section"
        ]
        == -1
    )


def test_l5_v2_dispatch_readiness_prioritizes_tt5l_campaign_action(
    tmp_path: Path,
) -> None:
    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]
    asymptotic = readiness["asymptotic_pursuit_candidates"]

    assert tt5l["schema"] == "l5_v2_tt5l_campaign_readiness_v1"
    assert asymptotic["schema"] == (
        l5_v2.L5_V2_ASYMPTOTIC_PURSUIT_CANDIDATES_SCHEMA
    )
    assert asymptotic["candidate_count"] == 3
    assert asymptotic["ready_for_exact_eval_dispatch"] is False
    assert tt5l["non_pr106_staircase_priority"] is True
    assert tt5l["packetir_is_optional_stack_evidence"] is True
    assert tt5l["score_claim"] is False
    assert tt5l["promotion_eligible"] is False
    assert tt5l["ready_for_exact_eval_dispatch"] is False
    assert tt5l["proof_tool_path"] == (
        "tools/build_tt5l_contest_sideinfo_consumption_proof.py"
    )
    assert tt5l["dykstra_feasibility_artifact_valid"] is False
    assert tt5l["first_anchor_timing_smoke_allowed"] is False
    assert tt5l["next_non_pr106_l5_action"]["action_id"] == (
        "run_tt5l_dykstra_score_axis_sanity"
    )
    assert tt5l["next_non_pr106_l5_action"]["phase"] == (
        "cargo_cult_unwind_feasibility"
    )
    assert l5_v2.TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH in tt5l[
        "next_non_pr106_l5_action"
    ]["expected_artifacts"]
    command_template = tt5l["next_non_pr106_l5_action"]["command_template"]
    assert "--tt5l-five-move-polytope" in command_template
    assert "<score_axis_lower_bound>" in command_template
    assert "<polytope_projected_lower_bound>" not in command_template
    assert "tt5l_dykstra_feasibility_artifact_missing" in tt5l["blockers"]
    assert "PR106" not in tt5l["next_non_pr106_l5_action"]["action_id"]


def test_l5_v2_tt5l_dykstra_artifact_unblocks_sideinfo_next_action(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)

    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["dykstra_feasibility_artifact_valid"] is True
    assert tt5l["dykstra_feasibility_status"]["archive_size_bytes"] == 34_603
    assert tt5l["dykstra_feasibility_status"]["tested_score_axis_band"] == [
        0.150,
        0.170,
    ]
    assert (
        tt5l["dykstra_feasibility_status"]["input_band_role"]
        == "planning_band_not_score_or_rank_authority"
    )
    assert tt5l["dykstra_feasibility_status"]["predicate_id"] == (
        l5_v2.TT5L_DYKSTRA_FEASIBILITY_PREDICATE_ID
    )
    assert tt5l["dykstra_feasibility_status"]["generated_by_tool"] == (
        l5_v2.TT5L_DYKSTRA_FEASIBILITY_GENERATED_BY_TOOL
    )
    assert tt5l["dykstra_feasibility_status"]["tool_sha256"] == _file_sha256(
        tmp_path / l5_v2.TT5L_DYKSTRA_FEASIBILITY_TOOL_PATH
    )
    assert tt5l["sideinfo_gate_evidence_valid"] is False
    assert tt5l["first_anchor_timing_smoke_allowed"] is False
    assert tt5l["next_non_pr106_l5_action"]["action_id"] == (
        "materialize_tt5l_contest_full_frame_sideinfo_consumption_proof"
    )
    assert "tt5l_dykstra_feasibility_artifact_missing" not in tt5l["blockers"]


def test_l5_v2_tt5l_dykstra_artifact_requires_tool_provenance(
    tmp_path: Path,
) -> None:
    artifact_path = _write_tt5l_dykstra_artifact(tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    for field in (
        "schema",
        "predicate_id",
        "generated_by_tool",
        "generated_at_utc",
        "command_argv",
        "tool_sha256",
    ):
        payload.pop(field)
    artifact_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["dykstra_feasibility_artifact_valid"] is False
    assert "tt5l_dykstra_feasibility_schema_missing_or_stale" in tt5l["blockers"]
    assert "tt5l_dykstra_feasibility_predicate_id_missing_or_stale" in tt5l[
        "blockers"
    ]
    assert "tt5l_dykstra_feasibility_generated_by_tool_missing_or_stale" in tt5l[
        "blockers"
    ]
    assert "tt5l_dykstra_feasibility_generated_at_utc_missing" in tt5l["blockers"]
    assert "tt5l_dykstra_feasibility_command_argv_missing" in tt5l["blockers"]
    assert "tt5l_dykstra_feasibility_tool_sha256_invalid" in tt5l["blockers"]


def test_l5_v2_score_axis_dykstra_does_not_unlock_without_move_level_proof(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path, write_move_level=False)
    evidence = _valid_gate_evidence_payloads(tmp_path)
    evidence.pop("c1_z5_tt5l_probe_disambiguator")
    evidence.pop("paired_cpu_cuda_axis_plan")
    evidence.pop("exact_anchor_or_diagnostic_pair")

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["dykstra_feasibility_artifact_valid"] is True
    assert tt5l["dykstra_score_axis_sanity_valid"] is True
    assert tt5l["move_level_feasibility_artifact_valid"] is False
    assert tt5l["sideinfo_gate_evidence_valid"] is True
    assert tt5l["sideinfo_effect_curve_allowed"] is False
    assert tt5l["first_anchor_timing_smoke_allowed"] is False
    assert "tt5l_move_level_feasibility_artifact_missing" in tt5l["blockers"]
    assert tt5l["next_non_pr106_l5_action"]["action_id"] == (
        "materialize_tt5l_move_level_feasibility_proof"
    )
    assert tt5l["next_non_pr106_l5_action"]["tool_path"] == (
        l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_TOOL_PATH
    )
    assert l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_TOOL_PATH in tt5l[
        "next_non_pr106_l5_action"
    ]["command_template"]
    assert "--proof-artifact" in tt5l["next_non_pr106_l5_action"][
        "command_template"
    ]
    assert tt5l["next_non_pr106_l5_action"][
        "score_axis_sanity_is_not_move_level_proof"
    ] is True


def test_l5_v2_tt5l_move_level_feasibility_requires_proof_hash_match(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    proof_artifact_path = (
        tmp_path
        / "experiments"
        / "results"
        / "time_traveler_l5_v2"
        / "tt5l_move_level_solver_proof.json"
    )
    proof_artifact_path.write_text(
        json.dumps({"schema": "tampered"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["dykstra_feasibility_artifact_valid"] is True
    assert tt5l["move_level_feasibility_artifact_valid"] is False
    assert "tt5l_move_level_feasibility_proof_artifact_sha256_mismatch" in tt5l[
        "blockers"
    ]
    assert tt5l["next_non_pr106_l5_action"]["action_id"] == (
        "materialize_tt5l_move_level_feasibility_proof"
    )


def test_l5_v2_tt5l_move_level_feasibility_rejects_handwritten_shape(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    artifact_path = tmp_path / l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_ARTIFACT_PATH
    artifact_path.write_text(
        json.dumps(
            {
                "schema": l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_SCHEMA,
                "subject_id": l5_v2.TT5L_DYKSTRA_SUBSTRATE_ID,
                "predicate_id": l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_PREDICATE_ID,
                "predicate_passed": True,
                "move_level_constraint_proof": True,
                "residual_max": 0.0,
                "residual_tolerance": 1e-9,
                "constraint_set_ids": sorted(
                    l5_v2.TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS
                ),
                "constraint_set_count": len(
                    l5_v2.TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["move_level_feasibility_artifact_valid"] is False
    assert "tt5l_move_level_feasibility_generated_by_tool_mismatch" in tt5l[
        "blockers"
    ]
    assert "tt5l_move_level_feasibility_tool_sha256_invalid" in tt5l["blockers"]
    assert "tt5l_move_level_feasibility_proof_artifact_path_missing" in tt5l[
        "blockers"
    ]
    assert tt5l["next_non_pr106_l5_action"]["action_id"] == (
        "materialize_tt5l_move_level_feasibility_proof"
    )


def test_l5_v2_tt5l_dykstra_artifact_rejects_empty_json(
    tmp_path: Path,
) -> None:
    tool_path = tmp_path / l5_v2.TT5L_DYKSTRA_FEASIBILITY_TOOL_PATH
    tool_path.parent.mkdir(parents=True, exist_ok=True)
    tool_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    artifact_path = tmp_path / l5_v2.TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("{}\n", encoding="utf-8")

    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["dykstra_feasibility_artifact_valid"] is False
    assert "tt5l_dykstra_feasibility_artifact_empty" in tt5l["blockers"]
    assert tt5l["next_non_pr106_l5_action"]["action_id"] == (
        "run_tt5l_dykstra_score_axis_sanity"
    )


def test_l5_v2_tt5l_dykstra_artifact_requires_archive_size_basis(
    tmp_path: Path,
) -> None:
    artifact_path = _write_tt5l_dykstra_artifact(tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload.pop("archive_size_bytes")
    artifact_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["dykstra_feasibility_artifact_valid"] is False
    assert "tt5l_dykstra_feasibility_archive_size_bytes_missing" in tt5l["blockers"]
    assert tt5l["next_non_pr106_l5_action"]["action_id"] == (
        "run_tt5l_dykstra_score_axis_sanity"
    )


def test_l5_v2_tt5l_dykstra_artifact_rejects_active_predicted_band_field(
    tmp_path: Path,
) -> None:
    artifact_path = _write_tt5l_dykstra_artifact(tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["predicted_band"] = [0.150, 0.170]
    artifact_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["dykstra_feasibility_artifact_valid"] is False
    assert "tt5l_dykstra_feasibility_active_predicted_band_field_present" in tt5l[
        "blockers"
    ]
    assert tt5l["next_non_pr106_l5_action"]["action_id"] == (
        "run_tt5l_dykstra_score_axis_sanity"
    )


def test_l5_v2_tt5l_dykstra_artifact_requires_non_authority_input_band_role(
    tmp_path: Path,
) -> None:
    artifact_path = _write_tt5l_dykstra_artifact(tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload.pop("input_band_role")
    artifact_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["dykstra_feasibility_artifact_valid"] is False
    assert "tt5l_dykstra_feasibility_input_band_role_missing_or_stale" in tt5l[
        "blockers"
    ]


def test_l5_v2_tt5l_dykstra_artifact_rejects_stale_scalar_projection(
    tmp_path: Path,
) -> None:
    artifact_path = _write_tt5l_dykstra_artifact(tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    for key in (
        "score_formula",
        "contest_seg_multiplier",
        "constraint_set_ids",
        "constraint_set_count",
        "polytope_projection_kind",
        "feasibility_scope",
        "verdict_authority_scope",
    ):
        payload.pop(key, None)
    artifact_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["dykstra_feasibility_artifact_valid"] is False
    assert "tt5l_dykstra_feasibility_score_formula_missing_or_stale" in tt5l[
        "blockers"
    ]
    assert "tt5l_dykstra_feasibility_five_move_constraints_missing" in tt5l[
        "blockers"
    ]
    assert "tt5l_dykstra_feasibility_constraint_set_ids_not_exact" in tt5l[
        "blockers"
    ]
    assert "tt5l_dykstra_feasibility_constraint_set_count_mismatch" in tt5l[
        "blockers"
    ]
    assert "tt5l_dykstra_feasibility_scope_missing_or_stale" in tt5l["blockers"]
    assert "tt5l_dykstra_feasibility_verdict_authority_scope_missing_or_stale" in tt5l[
        "blockers"
    ]


def test_l5_v2_tt5l_dykstra_artifact_rejects_extra_constraint_ids(
    tmp_path: Path,
) -> None:
    artifact_path = _write_tt5l_dykstra_artifact(tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["constraint_set_ids"] = [
        *payload["constraint_set_ids"],
        "cargo_cult_sixth_move",
    ]
    payload["constraint_set_count"] = len(payload["constraint_set_ids"])
    artifact_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["dykstra_feasibility_artifact_valid"] is False
    assert "tt5l_dykstra_feasibility_five_move_constraints_missing" not in tt5l[
        "blockers"
    ]
    assert "tt5l_dykstra_feasibility_constraint_set_ids_not_exact" in tt5l[
        "blockers"
    ]
    assert "tt5l_dykstra_feasibility_constraint_set_count_mismatch" in tt5l[
        "blockers"
    ]


def test_l5_v2_tt5l_dykstra_artifact_rejects_wrong_constraint_count(
    tmp_path: Path,
) -> None:
    artifact_path = _write_tt5l_dykstra_artifact(tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["constraint_set_count"] = len(
        l5_v2.TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS
    ) + 1
    artifact_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["dykstra_feasibility_artifact_valid"] is False
    assert "tt5l_dykstra_feasibility_constraint_set_ids_not_exact" not in tt5l[
        "blockers"
    ]
    assert "tt5l_dykstra_feasibility_constraint_set_count_mismatch" in tt5l[
        "blockers"
    ]


def test_l5_v2_tt5l_dykstra_artifact_rejects_move_level_proof_authority(
    tmp_path: Path,
) -> None:
    artifact_path = _write_tt5l_dykstra_artifact(tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["move_level_constraint_proof"] = True
    artifact_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["dykstra_feasibility_artifact_valid"] is False
    assert "tt5l_dykstra_feasibility_move_level_proof_not_false" in tt5l["blockers"]
    assert tt5l["first_anchor_timing_smoke_allowed"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("score_claim", "missing"),
        ("promotion_eligible", "false"),
        ("ready_for_exact_eval_dispatch", 0),
    ],
)
def test_l5_v2_tt5l_dykstra_artifact_requires_literal_false_authority_flags(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    artifact_path = _write_tt5l_dykstra_artifact(tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if value == "missing":
        payload.pop(field)
    else:
        payload[field] = value
    artifact_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["dykstra_feasibility_artifact_valid"] is False
    assert f"tt5l_dykstra_feasibility_{field}_not_false" in tt5l["blockers"]
    assert tt5l["first_anchor_timing_smoke_allowed"] is False


@pytest.mark.parametrize(
    ("verdict", "blocker"),
    [
        ("INDETERMINATE", "tt5l_dykstra_feasibility_verdict_indeterminate"),
        ("INFEASIBLE", "tt5l_dykstra_feasibility_verdict_infeasible"),
    ],
)
def test_l5_v2_tt5l_dykstra_artifact_only_feasible_unlocks(
    tmp_path: Path,
    verdict: str,
    blocker: str,
) -> None:
    artifact_path = _write_tt5l_dykstra_artifact(tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["verdict"] = verdict
    artifact_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["dykstra_feasibility_artifact_valid"] is False
    assert blocker in tt5l["blockers"]
    assert tt5l["first_anchor_timing_smoke_allowed"] is False
    assert tt5l["next_non_pr106_l5_action"]["action_id"] == (
        "run_tt5l_dykstra_score_axis_sanity"
    )


def test_tt5l_recipe_labels_retired_band_as_non_authority() -> None:
    recipe = Path(l5_v2.TT5L_MODAL_A100_DISPATCH_RECIPE_PATH).read_text(
        encoding="utf-8"
    )

    assert "Predicted contest-CPU score:" not in recipe
    assert "Retired additive contest-CPU band:" in recipe
    assert "NOT an active prediction" in recipe


def test_l5_v2_valid_gates_do_not_unlock_tt5l_timing_without_dykstra(
    tmp_path: Path,
) -> None:
    readiness = l5_v2_dispatch_readiness(
        gate_evidence=_valid_gate_evidence(tmp_path),
        repo_root=tmp_path,
    )
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["sideinfo_gate_evidence_valid"] is True
    assert tt5l["dykstra_feasibility_artifact_valid"] is False
    assert tt5l["first_anchor_timing_smoke_allowed"] is False
    assert tt5l["next_non_pr106_l5_action"]["action_id"] == (
        "run_tt5l_dykstra_score_axis_sanity"
    )


def test_l5_v2_tt5l_sideinfo_effect_curve_requires_dykstra_and_sideinfo_evidence(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    evidence = _valid_gate_evidence_payloads(tmp_path)
    evidence.pop("c1_z5_tt5l_probe_disambiguator")
    evidence.pop("paired_cpu_cuda_axis_plan")
    evidence.pop("exact_anchor_or_diagnostic_pair")

    readiness = l5_v2_dispatch_readiness(
        gate_evidence=evidence,
        repo_root=tmp_path,
    )
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["dykstra_feasibility_artifact_valid"] is True
    assert tt5l["sideinfo_gate_evidence_valid"] is True
    assert tt5l["sideinfo_effect_curve_allowed"] is True
    assert tt5l["first_anchor_timing_smoke_allowed"] is False


def test_l5_v2_tt5l_probe_action_advances_after_template_exists(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    evidence = _valid_gate_evidence(tmp_path)
    evidence.pop("c1_z5_tt5l_probe_disambiguator")
    template = tmp_path / l5_v2.TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("{}\n", encoding="utf-8")

    readiness = l5_v2_dispatch_readiness(
        gate_evidence=evidence,
        repo_root=tmp_path,
    )
    action = readiness["tt5l_campaign_readiness"]["next_non_pr106_l5_action"]

    assert action["action_id"] == "populate_and_evaluate_c1_z5_tt5l_probe_observations"
    assert action["probe_status"] == "observation_intake_required"
    assert "tools/audit_l5_v2_probe_observations.py" in action["command_template"]
    assert "l5_v2_probe_gate_artifact_20260516_codex.json" in action[
        "command_template"
    ]
    assert (
        action["measurement_schedule_tool_path"]
        == l5_v2.L5V2_MEASUREMENT_SCHEDULE_TOOL_PATH
    )
    assert "tools/build_l5_v2_lattice_measurement_schedule.py" in action[
        "measurement_schedule_command_template"
    ]
    assert l5_v2.TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH in action[
        "measurement_schedule_command_template"
    ]
    assert l5_v2.TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH in action[
        "expected_artifacts"
    ]
    assert l5_v2.TT5L_PROBE_OBSERVATION_INTAKE_REPORT_PATH in action[
        "expected_artifacts"
    ]
    assert action["measurement_schedule_expected_artifacts"] == [
        l5_v2.L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH,
        l5_v2.L5V2_MEASUREMENT_SCHEDULE_REPORT_PATH,
    ]
    assert action["score_claim"] is False
    assert "planning-only" in action["measurement_schedule_semantics"]


def test_l5_v2_tt5l_probe_action_uses_existing_fail_closed_intake(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    evidence = _valid_gate_evidence(tmp_path)
    evidence.pop("c1_z5_tt5l_probe_disambiguator")
    template = tmp_path / l5_v2.TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("{}\n", encoding="utf-8")
    intake_path = tmp_path / l5_v2.TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH
    intake_path.parent.mkdir(parents=True, exist_ok=True)
    intake_path.write_text(
        json.dumps(
            {
                "schema": "l5_v2_probe_observation_intake_v1",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "architecture_lock_allowed": False,
                "verdict": {
                    "evaluated_observations": [
                        {
                            "candidate_id": candidate_id,
                            "eligible_for_architecture_lock": False,
                            "blockers": [
                                f"l5_v2_probe_required_candidate_ineligible:{candidate_id}"
                            ],
                        }
                        for candidate_id in L5V2_CANDIDATES
                    ],
                    "architecture_lock_allowed": False,
                    "blockers": ["l5_v2_probe_no_eligible_candidate"],
                },
                "blockers": ["l5_v2_probe_no_eligible_candidate"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    readiness = l5_v2_dispatch_readiness(
        gate_evidence=evidence,
        repo_root=tmp_path,
    )
    action = readiness["tt5l_campaign_readiness"]["next_non_pr106_l5_action"]
    intake_status = readiness["tt5l_campaign_readiness"][
        "probe_observation_intake_status"
    ]

    assert action["action_id"] == "materialize_l5_v2_paired_probe_measurements"
    assert action["probe_status"] == "observation_intake_present_fail_closed"
    assert intake_status["artifact_valid_for_measurement_planning"] is True
    assert intake_status["evaluated_observation_count"] == len(L5V2_CANDIDATES)
    assert "command_template" not in action
    assert "tools/build_l5_v2_lattice_measurement_schedule.py" in action[
        "measurement_schedule_command_template"
    ]
    assert "tools/build_l5_v2_paired_measurement_dispatch_plan.py" in action[
        "paired_measurement_dispatch_plan_command_template"
    ]
    assert action["execution_order"] == [
        "build_l5_v2_lattice_measurement_schedule",
        "build_l5_v2_paired_measurement_dispatch_plan",
        "fill_each_work_unit_archive_runtime_sha_and_operator_execute_flag",
    ]
    assert action["score_claim"] is False
    assert action["ready_for_exact_eval_dispatch"] is False


def test_l5_v2_tt5l_probe_action_advances_after_work_unit_materialized(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    evidence = _valid_gate_evidence(tmp_path)
    evidence.pop("c1_z5_tt5l_probe_disambiguator")
    template = tmp_path / l5_v2.TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("{}\n", encoding="utf-8")
    intake_path = tmp_path / l5_v2.TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH
    intake_path.parent.mkdir(parents=True, exist_ok=True)
    intake_path.write_text(
        json.dumps(
            {
                "schema": "l5_v2_probe_observation_intake_v1",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "verdict": {
                    "evaluated_observations": [
                        {"candidate_id": candidate_id}
                        for candidate_id in L5V2_CANDIDATES
                    ]
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    work_unit_path = (
        tmp_path / l5_v2.TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_ARTIFACT_PATH
    )
    work_unit_path.parent.mkdir(parents=True, exist_ok=True)
    archive_rel = "experiments/results/tt5l/archive.zip"
    archive_path = tmp_path / archive_rel
    _write_tt5l_archive_zip(archive_path, nonzero_side_info=True)
    archive_sha = _file_sha256(archive_path)
    runtime_rel = "experiments/results/tt5l/runtime"
    runtime_dir = tmp_path / runtime_rel
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    _write_standard_tt5l_materialized_work_unit_plan(
        work_unit_path,
        archive_rel=archive_rel,
        archive_sha=archive_sha,
        archive_bytes=archive_path.stat().st_size,
        runtime_rel=runtime_rel,
    )

    readiness = l5_v2_dispatch_readiness(
        gate_evidence=evidence,
        repo_root=tmp_path,
    )
    action = readiness["tt5l_campaign_readiness"]["next_non_pr106_l5_action"]
    materialized = readiness["tt5l_campaign_readiness"][
        "materialized_tt5l_paired_work_unit_status"
    ]

    assert (
        action["action_id"]
        == "review_and_execute_l5_v2_tt5l_materialized_paired_measurement"
    )
    assert materialized["artifact_valid"] is True
    assert action["ready_for_operator_dispatch"] is True
    assert action["ready_for_provider_dispatch"] is False
    assert action["score_claim"] is False
    assert action["ready_for_exact_eval_dispatch"] is False
    assert materialized["tt5l_sideinfo_stats"]["nonzero_pair_count"] == 1
    assert materialized["tt5l_sideinfo_stats"]["all_zero_pair_count"] == 599
    assert materialized["tt5l_sideinfo_stats"]["liveness_warnings"] == [
        "tt5l_side_info_some_pairs_all_zero",
        "tt5l_side_info_at_most_one_nonzero_per_pair_on_average",
    ]
    assert "tools/dispatch_modal_paired_auth_eval.py" in action[
        "operator_execute_command_template_after_review"
    ]
    assert "--execute" in action["operator_execute_command_template_after_review"]
    assert "--skip-axis-if-promotable-anchor-exists" not in action[
        "operator_execute_command_template_after_review"
    ]
    assert "experiments/modal_auth_eval.py" not in action[
        "operator_execute_command_template_after_review"
    ]


def test_l5_v2_tt5l_materialized_work_unit_surfaces_modal_billing_blocker(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    evidence = _valid_gate_evidence(tmp_path)
    evidence.pop("c1_z5_tt5l_probe_disambiguator")
    template = tmp_path / l5_v2.TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("{}\n", encoding="utf-8")
    intake_path = tmp_path / l5_v2.TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH
    intake_path.parent.mkdir(parents=True, exist_ok=True)
    intake_path.write_text(
        json.dumps(
            {
                "schema": "l5_v2_probe_observation_intake_v1",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "verdict": {
                    "evaluated_observations": [
                        {"candidate_id": candidate_id}
                        for candidate_id in L5V2_CANDIDATES
                    ]
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    work_unit_path = (
        tmp_path / l5_v2.TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_ARTIFACT_PATH
    )
    work_unit_path.parent.mkdir(parents=True, exist_ok=True)
    archive_rel = "experiments/results/tt5l/archive.zip"
    archive_path = tmp_path / archive_rel
    _write_tt5l_archive_zip(archive_path, nonzero_side_info=True)
    archive_sha = _file_sha256(archive_path)
    runtime_rel = "experiments/results/tt5l/runtime"
    runtime_dir = tmp_path / runtime_rel
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    _write_standard_tt5l_materialized_work_unit_plan(
        work_unit_path,
        archive_rel=archive_rel,
        archive_sha=archive_sha,
        archive_bytes=archive_path.stat().st_size,
        runtime_rel=runtime_rel,
    )
    blocker_path = tmp_path / l5_v2.TT5L_MATERIALIZED_MODAL_PROVIDER_BLOCKER_ARTIFACT_PATH
    blocker_path.parent.mkdir(parents=True, exist_ok=True)
    blocker_path.write_text(
        json.dumps(
            {
                "schema": "l5_v2_tt5l_materialized_provider_blocker_v1",
                "provider": "modal",
                "failure_class": "modal_workspace_billing_cycle_spend_limit_reached",
                "archive_sha256": archive_sha,
                "pair_group_id": (
                    "pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda"
                ),
                "resolved": False,
                "score_claim": False,
                "promotion_eligible": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    readiness = l5_v2_dispatch_readiness(
        gate_evidence=evidence,
        repo_root=tmp_path,
    )
    tt5l = readiness["tt5l_campaign_readiness"]
    action = tt5l["next_non_pr106_l5_action"]
    provider_blocker = action["provider_blocker_status"]

    assert action["action_id"] == (
        "resolve_l5_v2_tt5l_modal_provider_blocker_or_dispatch_alternate_provider"
    )
    assert provider_blocker["active"] is True
    assert provider_blocker["failure_class"] == (
        "modal_workspace_billing_cycle_spend_limit_reached"
    )
    assert action["ready_for_operator_dispatch"] is False
    assert action["ready_for_alternate_provider_planning"] is True
    assert action["dispatch_attempted"] is True
    assert action["modal_execute_command_suppressed_until_provider_blocker_resolved"] is True
    assert "operator_execute_command_template_after_review" not in action
    assert (
        "operator_execute_command_template_after_review"
        not in action["materialized_work_unit_status"]
    )
    assert not _recursive_string_contains(action, "--execute")
    assert (
        "l5_v2_tt5l_modal_provider_blocker_active:"
        "modal_workspace_billing_cycle_spend_limit_reached"
        in tt5l["blockers"]
    )


def test_l5_v2_tt5l_materialized_work_unit_rejects_runtime_content_axis_mismatch(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    evidence = _valid_gate_evidence(tmp_path)
    evidence.pop("c1_z5_tt5l_probe_disambiguator")
    template = tmp_path / l5_v2.TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("{}\n", encoding="utf-8")
    intake_path = tmp_path / l5_v2.TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH
    intake_path.parent.mkdir(parents=True, exist_ok=True)
    intake_path.write_text(
        json.dumps(
            {
                "schema": "l5_v2_probe_observation_intake_v1",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "verdict": {
                    "evaluated_observations": [
                        {"candidate_id": candidate_id}
                        for candidate_id in L5V2_CANDIDATES
                    ]
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    work_unit_path = (
        tmp_path / l5_v2.TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_ARTIFACT_PATH
    )
    work_unit_path.parent.mkdir(parents=True, exist_ok=True)
    archive_rel = "experiments/results/tt5l/archive.zip"
    archive_path = tmp_path / archive_rel
    _write_tt5l_archive_zip(archive_path, nonzero_side_info=True)
    archive_sha = _file_sha256(archive_path)
    runtime_rel = "experiments/results/tt5l/runtime"
    runtime_dir = tmp_path / runtime_rel
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    _write_standard_tt5l_materialized_work_unit_plan(
        work_unit_path,
        archive_rel=archive_rel,
        archive_sha=archive_sha,
        archive_bytes=archive_path.stat().st_size,
        runtime_rel=runtime_rel,
    )
    payload = json.loads(work_unit_path.read_text(encoding="utf-8"))
    payload["runtime"]["expected_runtime_content_tree_sha256_by_axis"][
        "contest_cuda"
    ] = "e" * 64
    work_unit_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    readiness = l5_v2_dispatch_readiness(
        gate_evidence=evidence,
        repo_root=tmp_path,
    )
    tt5l = readiness["tt5l_campaign_readiness"]
    materialized = tt5l["materialized_tt5l_paired_work_unit_status"]

    assert materialized["artifact_valid"] is False
    assert (
        "l5_v2_tt5l_materialized_paired_work_unit_runtime_content_axis_mismatch"
        in materialized["blockers"]
    )
    assert tt5l["next_non_pr106_l5_action"]["action_id"] == (
        "materialize_l5_v2_paired_probe_measurements"
    )


def test_l5_v2_tt5l_stale_modal_blocker_blocks_execute_command(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    evidence = _valid_gate_evidence(tmp_path)
    evidence.pop("c1_z5_tt5l_probe_disambiguator")
    template = tmp_path / l5_v2.TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("{}\n", encoding="utf-8")
    intake_path = tmp_path / l5_v2.TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH
    intake_path.parent.mkdir(parents=True, exist_ok=True)
    intake_path.write_text(
        json.dumps(
            {
                "schema": "l5_v2_probe_observation_intake_v1",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "verdict": {
                    "evaluated_observations": [
                        {"candidate_id": candidate_id}
                        for candidate_id in L5V2_CANDIDATES
                    ]
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    work_unit_path = (
        tmp_path / l5_v2.TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_ARTIFACT_PATH
    )
    work_unit_path.parent.mkdir(parents=True, exist_ok=True)
    archive_rel = "experiments/results/tt5l/archive.zip"
    archive_path = tmp_path / archive_rel
    _write_tt5l_archive_zip(archive_path, nonzero_side_info=True)
    archive_sha = _file_sha256(archive_path)
    runtime_rel = "experiments/results/tt5l/runtime"
    runtime_dir = tmp_path / runtime_rel
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    _write_standard_tt5l_materialized_work_unit_plan(
        work_unit_path,
        archive_rel=archive_rel,
        archive_sha=archive_sha,
        archive_bytes=archive_path.stat().st_size,
        runtime_rel=runtime_rel,
    )
    blocker_path = tmp_path / l5_v2.TT5L_MATERIALIZED_MODAL_PROVIDER_BLOCKER_ARTIFACT_PATH
    blocker_path.parent.mkdir(parents=True, exist_ok=True)
    blocker_path.write_text(
        json.dumps(
            {
                "schema": "l5_v2_tt5l_materialized_provider_blocker_v1",
                "provider": "modal",
                "failure_class": "modal_workspace_billing_cycle_spend_limit_reached",
                "archive_sha256": "0" * 64,
                "pair_group_id": (
                    "pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda"
                ),
                "resolved": False,
                "score_claim": False,
                "promotion_eligible": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    readiness = l5_v2_dispatch_readiness(
        gate_evidence=evidence,
        repo_root=tmp_path,
    )
    tt5l = readiness["tt5l_campaign_readiness"]
    action = tt5l["next_non_pr106_l5_action"]
    provider_blocker = action["provider_blocker_status"]

    assert action["action_id"] == (
        "refresh_or_retire_l5_v2_tt5l_modal_provider_blocker"
    )
    assert provider_blocker["artifact_exists"] is True
    assert provider_blocker["artifact_valid"] is False
    assert provider_blocker["active"] is False
    assert action["ready_for_operator_dispatch"] is False
    assert action["ready_for_provider_dispatch"] is False
    assert "operator_execute_command_template_after_review" not in action
    assert (
        "operator_execute_command_template_after_review"
        not in action["materialized_work_unit_status"]
    )
    assert not _recursive_string_contains(action, "--execute")
    assert (
        "l5_v2_tt5l_modal_provider_blocker_archive_sha_mismatch"
        in provider_blocker["blockers"]
    )
    assert (
        "l5_v2_tt5l_modal_provider_blocker_archive_sha_mismatch"
        in tt5l["blockers"]
    )


def test_l5_v2_tt5l_modal_blocker_surfaces_lightning_alternate_plan(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    evidence = _valid_gate_evidence(tmp_path)
    evidence.pop("c1_z5_tt5l_probe_disambiguator")
    template = tmp_path / l5_v2.TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("{}\n", encoding="utf-8")
    intake_path = tmp_path / l5_v2.TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH
    intake_path.parent.mkdir(parents=True, exist_ok=True)
    intake_path.write_text(
        json.dumps(
            {
                "schema": "l5_v2_probe_observation_intake_v1",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "verdict": {
                    "evaluated_observations": [
                        {"candidate_id": candidate_id}
                        for candidate_id in L5V2_CANDIDATES
                    ]
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    work_unit_path = (
        tmp_path / l5_v2.TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_ARTIFACT_PATH
    )
    work_unit_path.parent.mkdir(parents=True, exist_ok=True)
    archive_rel = "experiments/results/tt5l/archive.zip"
    archive_path = tmp_path / archive_rel
    _write_tt5l_archive_zip(archive_path, nonzero_side_info=True)
    archive_sha = _file_sha256(archive_path)
    runtime_rel = "experiments/results/tt5l/runtime"
    runtime_dir = tmp_path / runtime_rel
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    _write_standard_tt5l_materialized_work_unit_plan(
        work_unit_path,
        archive_rel=archive_rel,
        archive_sha=archive_sha,
        archive_bytes=archive_path.stat().st_size,
        runtime_rel=runtime_rel,
    )
    blocker_path = tmp_path / l5_v2.TT5L_MATERIALIZED_MODAL_PROVIDER_BLOCKER_ARTIFACT_PATH
    blocker_path.parent.mkdir(parents=True, exist_ok=True)
    blocker_path.write_text(
        json.dumps(
            {
                "schema": "l5_v2_tt5l_materialized_provider_blocker_v1",
                "provider": "modal",
                "failure_class": "modal_workspace_billing_cycle_spend_limit_reached",
                "archive_sha256": archive_sha,
                "pair_group_id": (
                    "pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda"
                ),
                "resolved": False,
                "score_claim": False,
                "promotion_eligible": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    alternate_plan_path = (
        tmp_path
        / l5_v2.TT5L_MATERIALIZED_LIGHTNING_ALT_PROVIDER_PLAN_ARTIFACT_PATH
    )
    alternate_plan_path.parent.mkdir(parents=True, exist_ok=True)
    alternate_plan_path.write_text(
        json.dumps(
            {
                "schema": "l5_v2_tt5l_lightning_alternate_provider_plan_v1",
                "provider": "lightning",
                "archive_sha256": archive_sha,
                "pair_group_id": (
                    "pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda"
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "local_lightning_doctor": {
                    "status": "OK",
                    "local_supply_chain_ok": True,
                },
                "exact_eval_dry_run": {
                    "status": "DRY_RUN",
                    "job_name": "l5-v2-tt5l-random-lsb-cuda-20260517",
                    "command_sha256": "a" * 64,
                },
                "source_manifest_probe": {
                    "command": (
                        ".venv/bin/python scripts/lightning_repro_workspace.py "
                        f"--dry-run --artifact {archive_rel} "
                        f"--artifact {runtime_rel}"
                    ),
                    "failure_class": "lightning_remote_identity_missing",
                    "observed_at_utc": "2026-05-17T07:30:46Z",
                    "returncode": 1,
                    "status_basis": (
                        "rerun_after_archive_refresh_against_current_archive_and_runtime; "
                        "still blocked until remote identity is supplied"
                    ),
                    "stderr_or_stdout": (
                        "set --remote or LIGHTNING_SSH_TARGET before staging to Lightning"
                    ),
                },
                "execution_blockers": [
                    "missing_lightning_ssh_target",
                    "missing_lightning_teamspace",
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    readiness = l5_v2_dispatch_readiness(
        gate_evidence=evidence,
        repo_root=tmp_path,
    )
    tt5l = readiness["tt5l_campaign_readiness"]
    action = tt5l["next_non_pr106_l5_action"]
    alternate = action["alternate_provider_plan_status"]

    assert action["action_id"] == (
        "resolve_l5_v2_tt5l_modal_provider_blocker_or_dispatch_alternate_provider"
    )
    assert alternate["artifact_valid"] is True
    assert alternate["provider"] == "lightning"
    assert alternate["local_supply_chain_ok"] is True
    assert alternate["exact_eval_dry_run_ok"] is True
    assert alternate["source_manifest_probe_current"] is True
    assert alternate["execution_ready"] is False
    assert action["ready_for_alternate_provider_planning"] is True
    assert action["ready_for_alternate_provider_dispatch"] is False
    assert (
        "l5_v2_tt5l_lightning_alt_provider_blocked:missing_lightning_ssh_target"
        in tt5l["blockers"]
    )
    report = render_l5_v2_architecture_lock_packet_markdown(
        l5_v2_architecture_lock_packet(gate_evidence=evidence, repo_root=tmp_path)
    )
    assert "## Materialized TT5L Provider Routing" in report
    assert "- provider_blocker_active: `True`" in report
    assert (
        "- provider_blocker_failure_class: "
        "`modal_workspace_billing_cycle_spend_limit_reached`"
    ) in report
    assert "- modal_execute_suppressed_until_blocker_resolved: `True`" in report
    assert "- alternate_provider: `lightning`" in report
    assert "- alternate_artifact_valid: `True`" in report
    assert "- lightning_source_manifest_probe_current: `True`" in report
    assert "- lightning_execution_ready: `False`" in report
    assert "missing_lightning_ssh_target" in report


def test_l5_v2_tt5l_lightning_alternate_plan_rejects_stale_source_manifest_probe(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    evidence = _valid_gate_evidence(tmp_path)
    evidence.pop("c1_z5_tt5l_probe_disambiguator")
    template = tmp_path / l5_v2.TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("{}\n", encoding="utf-8")
    intake_path = tmp_path / l5_v2.TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH
    intake_path.parent.mkdir(parents=True, exist_ok=True)
    intake_path.write_text(
        json.dumps(
            {
                "schema": "l5_v2_probe_observation_intake_v1",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "verdict": {
                    "evaluated_observations": [
                        {"candidate_id": candidate_id}
                        for candidate_id in L5V2_CANDIDATES
                    ]
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    work_unit_path = (
        tmp_path / l5_v2.TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_ARTIFACT_PATH
    )
    work_unit_path.parent.mkdir(parents=True, exist_ok=True)
    archive_rel = "experiments/results/tt5l/archive.zip"
    archive_path = tmp_path / archive_rel
    _write_tt5l_archive_zip(archive_path, nonzero_side_info=True)
    archive_sha = _file_sha256(archive_path)
    runtime_rel = "experiments/results/tt5l/runtime"
    runtime_dir = tmp_path / runtime_rel
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    _write_standard_tt5l_materialized_work_unit_plan(
        work_unit_path,
        archive_rel=archive_rel,
        archive_sha=archive_sha,
        archive_bytes=archive_path.stat().st_size,
        runtime_rel=runtime_rel,
    )
    blocker_path = tmp_path / l5_v2.TT5L_MATERIALIZED_MODAL_PROVIDER_BLOCKER_ARTIFACT_PATH
    blocker_path.parent.mkdir(parents=True, exist_ok=True)
    blocker_path.write_text(
        json.dumps(
            {
                "schema": "l5_v2_tt5l_materialized_provider_blocker_v1",
                "provider": "modal",
                "failure_class": "modal_workspace_billing_cycle_spend_limit_reached",
                "archive_sha256": archive_sha,
                "pair_group_id": (
                    "pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda"
                ),
                "resolved": False,
                "score_claim": False,
                "promotion_eligible": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    alternate_plan_path = (
        tmp_path
        / l5_v2.TT5L_MATERIALIZED_LIGHTNING_ALT_PROVIDER_PLAN_ARTIFACT_PATH
    )
    alternate_plan_path.parent.mkdir(parents=True, exist_ok=True)
    alternate_plan_path.write_text(
        json.dumps(
            {
                "schema": "l5_v2_tt5l_lightning_alternate_provider_plan_v1",
                "provider": "lightning",
                "archive_sha256": archive_sha,
                "pair_group_id": (
                    "pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda"
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "local_lightning_doctor": {
                    "status": "OK",
                    "local_supply_chain_ok": True,
                },
                "exact_eval_dry_run": {
                    "status": "DRY_RUN",
                    "job_name": "l5-v2-tt5l-random-lsb-cuda-20260517",
                    "command_sha256": "a" * 64,
                },
                "source_manifest_probe": {
                    "command": (
                        ".venv/bin/python scripts/lightning_repro_workspace.py "
                        "--dry-run --artifact experiments/results/old/archive.zip "
                        f"--artifact {runtime_rel}"
                    ),
                    "failure_class": "lightning_remote_identity_missing",
                    "returncode": 1,
                    "status_basis": (
                        "not_rerun_after_archive_refresh; still blocked until "
                        "remote identity is supplied"
                    ),
                },
                "execution_blockers": [
                    "missing_lightning_ssh_target",
                    "source_manifest_not_staged",
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    readiness = l5_v2_dispatch_readiness(
        gate_evidence=evidence,
        repo_root=tmp_path,
    )
    alternate = readiness["tt5l_campaign_readiness"][
        "next_non_pr106_l5_action"
    ]["alternate_provider_plan_status"]

    assert alternate["artifact_valid"] is False
    assert alternate["source_manifest_probe_current"] is False
    assert (
        "l5_v2_tt5l_lightning_alt_provider_source_manifest_probe_observed_at_missing"
        in alternate["blockers"]
    )
    assert (
        "l5_v2_tt5l_lightning_alt_provider_source_manifest_archive_mismatch"
        in alternate["blockers"]
    )
    assert (
        "l5_v2_tt5l_lightning_alt_provider_source_manifest_probe_stale"
        in alternate["blockers"]
    )


def test_l5_v2_tt5l_materialized_work_unit_rejects_all_zero_sideinfo(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    evidence = _valid_gate_evidence(tmp_path)
    evidence.pop("c1_z5_tt5l_probe_disambiguator")
    template = tmp_path / l5_v2.TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("{}\n", encoding="utf-8")
    intake_path = tmp_path / l5_v2.TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH
    intake_path.parent.mkdir(parents=True, exist_ok=True)
    intake_path.write_text(
        json.dumps(
            {
                "schema": "l5_v2_probe_observation_intake_v1",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "verdict": {
                    "evaluated_observations": [
                        {"candidate_id": candidate_id}
                        for candidate_id in L5V2_CANDIDATES
                    ]
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    work_unit_path = (
        tmp_path / l5_v2.TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_ARTIFACT_PATH
    )
    work_unit_path.parent.mkdir(parents=True, exist_ok=True)
    archive_rel = "experiments/results/tt5l/archive.zip"
    archive_path = tmp_path / archive_rel
    _write_tt5l_archive_zip(archive_path, nonzero_side_info=False)
    archive_sha = _file_sha256(archive_path)
    runtime_rel = "experiments/results/tt5l/runtime"
    runtime_dir = tmp_path / runtime_rel
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    _write_standard_tt5l_materialized_work_unit_plan(
        work_unit_path,
        archive_rel=archive_rel,
        archive_sha=archive_sha,
        archive_bytes=archive_path.stat().st_size,
        runtime_rel=runtime_rel,
    )

    readiness = l5_v2_dispatch_readiness(
        gate_evidence=evidence,
        repo_root=tmp_path,
    )
    materialized = readiness["tt5l_campaign_readiness"][
        "materialized_tt5l_paired_work_unit_status"
    ]

    assert materialized["artifact_valid"] is False
    assert (
        "l5_v2_tt5l_materialized_paired_work_unit_tt5l_sideinfo_all_zero"
        in materialized["blockers"]
    )
    assert materialized["tt5l_sideinfo_stats"]["valid"] is True
    assert materialized["tt5l_sideinfo_stats"]["nonzero_values"] == 0
    assert materialized["tt5l_sideinfo_stats"]["nonzero_pair_count"] == 0
    assert materialized["tt5l_sideinfo_stats"]["all_zero_pair_count"] == 600
    assert materialized["tt5l_sideinfo_stats"]["section_liveness"]["checked"] is True
    assert readiness["tt5l_campaign_readiness"]["next_non_pr106_l5_action"][
        "action_id"
    ] == "materialize_l5_v2_paired_probe_measurements"


def test_l5_v2_tt5l_materialized_work_unit_rejects_noncontest_pair_count(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    evidence = _valid_gate_evidence(tmp_path)
    evidence.pop("c1_z5_tt5l_probe_disambiguator")
    template = tmp_path / l5_v2.TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("{}\n", encoding="utf-8")
    intake_path = tmp_path / l5_v2.TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH
    intake_path.parent.mkdir(parents=True, exist_ok=True)
    intake_path.write_text(
        json.dumps(
            {
                "schema": "l5_v2_probe_observation_intake_v1",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "verdict": {
                    "evaluated_observations": [
                        {"candidate_id": candidate_id}
                        for candidate_id in L5V2_CANDIDATES
                    ]
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    work_unit_path = (
        tmp_path / l5_v2.TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_ARTIFACT_PATH
    )
    work_unit_path.parent.mkdir(parents=True, exist_ok=True)
    archive_rel = "experiments/results/tt5l/archive.zip"
    archive_path = tmp_path / archive_rel
    _write_tt5l_archive_zip(archive_path, nonzero_side_info=True, num_pairs=2)
    archive_sha = _file_sha256(archive_path)
    runtime_rel = "experiments/results/tt5l/runtime"
    runtime_dir = tmp_path / runtime_rel
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    _write_standard_tt5l_materialized_work_unit_plan(
        work_unit_path,
        archive_rel=archive_rel,
        archive_sha=archive_sha,
        archive_bytes=archive_path.stat().st_size,
        runtime_rel=runtime_rel,
    )

    readiness = l5_v2_dispatch_readiness(
        gate_evidence=evidence,
        repo_root=tmp_path,
    )
    materialized = readiness["tt5l_campaign_readiness"][
        "materialized_tt5l_paired_work_unit_status"
    ]

    assert materialized["artifact_valid"] is False
    assert (
        "l5_v2_tt5l_materialized_paired_work_unit_tt5l_num_pairs_not_full_contest"
        in materialized["blockers"]
    )
    assert materialized["tt5l_sideinfo_stats"]["valid"] is True
    assert materialized["tt5l_sideinfo_stats"]["num_pairs"] == 2
    assert materialized["tt5l_sideinfo_stats"]["nonzero_values"] > 0
    assert readiness["tt5l_campaign_readiness"]["next_non_pr106_l5_action"][
        "action_id"
    ] == "materialize_l5_v2_paired_probe_measurements"


def test_l5_v2_tt5l_materialized_work_unit_rejects_weak_axis_commands(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    evidence = _valid_gate_evidence(tmp_path)
    evidence.pop("c1_z5_tt5l_probe_disambiguator")
    template = tmp_path / l5_v2.TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("{}\n", encoding="utf-8")
    intake_path = tmp_path / l5_v2.TT5L_PROBE_OBSERVATION_INTAKE_ARTIFACT_PATH
    intake_path.parent.mkdir(parents=True, exist_ok=True)
    intake_path.write_text(
        json.dumps(
            {
                "schema": "l5_v2_probe_observation_intake_v1",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "verdict": {
                    "evaluated_observations": [
                        {"candidate_id": candidate_id}
                        for candidate_id in L5V2_CANDIDATES
                    ]
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    archive_rel = "experiments/results/tt5l/archive.zip"
    archive_path = tmp_path / archive_rel
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(b"tt5l archive")
    runtime_rel = "experiments/results/tt5l/runtime"
    runtime_dir = tmp_path / runtime_rel
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    work_unit_path = (
        tmp_path / l5_v2.TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_ARTIFACT_PATH
    )
    work_unit_path.parent.mkdir(parents=True, exist_ok=True)
    work_unit_path.write_text(
        json.dumps(
            {
                "schema": "modal_paired_auth_eval_dispatch_plan_v2",
                "pair_group_id": "wrong_pair_group",
                "score_claim": False,
                "promotion_eligible": False,
                "archive": {
                    "path": archive_rel,
                    "sha256": _file_sha256(archive_path),
                    "bytes": archive_path.stat().st_size,
                },
                "runtime": {
                    "submission_dir": runtime_rel,
                    "expected_runtime_tree_sha256_by_axis": {
                        "contest_cpu": "b" * 64,
                        "contest_cuda": "c" * 64,
                    },
                    "expected_runtime_content_tree_sha256_by_axis": {
                        "contest_cpu": "d" * 64,
                        "contest_cuda": "d" * 64,
                    },
                },
                "required_axes": ["contest_cpu", "contest_cuda"],
                "lanes": {
                    "contest_cpu": "wrong_cpu_lane",
                    "contest_cuda": "wrong_cuda_lane",
                },
                "commands": {
                    "contest_cpu": ["modal", "run", "cpu"],
                    "contest_cuda": ["modal", "run", "cuda"],
                },
                "axes_skipped_due_to_existing_anchor": {
                    "contest_cpu": False,
                    "contest_cuda": False,
                },
                "existing_anchors_reused": {
                    "contest_cpu": None,
                    "contest_cuda": None,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    readiness = l5_v2_dispatch_readiness(
        gate_evidence=evidence,
        repo_root=tmp_path,
    )
    materialized = readiness["tt5l_campaign_readiness"][
        "materialized_tt5l_paired_work_unit_status"
    ]

    assert materialized["artifact_valid"] is False
    assert (
        "l5_v2_tt5l_materialized_paired_work_unit_pair_group_id_mismatch"
        in materialized["blockers"]
    )
    assert (
        "l5_v2_tt5l_materialized_paired_work_unit_axis_command_not_modal_run:"
        "contest_cpu"
        in materialized["blockers"]
    )
    assert (
        "l5_v2_tt5l_materialized_paired_work_unit_axis_command_wrong_wrapper:"
        "contest_cuda"
        in materialized["blockers"]
    )
    assert readiness["tt5l_campaign_readiness"]["next_non_pr106_l5_action"][
        "action_id"
    ] == "materialize_l5_v2_paired_probe_measurements"


def test_l5_v2_tt5l_readiness_surfaces_measurement_schedule_without_authority(
    tmp_path: Path,
) -> None:
    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert (
        tt5l["measurement_schedule_tool_path"]
        == l5_v2.L5V2_MEASUREMENT_SCHEDULE_TOOL_PATH
    )
    assert (
        tt5l["measurement_schedule_artifact_path"]
        == l5_v2.L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH
    )
    assert (
        tt5l["measurement_schedule_report_path"]
        == l5_v2.L5V2_MEASUREMENT_SCHEDULE_REPORT_PATH
    )
    assert tt5l["measurement_schedule_score_claim"] is False
    assert tt5l["measurement_schedule_promotion_eligible"] is False
    assert tt5l["measurement_schedule_ready_for_exact_eval_dispatch"] is False


def test_l5_v2_canonical_probe_gate_evidence_auto_consumes_valid_artifact(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / l5_v2.TT5L_PROBE_GATE_ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "probe_disambiguator": _probe_disambiguator_payload(tmp_path),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    evidence = l5_v2.l5_v2_canonical_probe_gate_evidence(repo_root=tmp_path)

    assert evidence is not None
    assert evidence.gate_id == "c1_z5_tt5l_probe_disambiguator"
    assert evidence.artifact_path == l5_v2.TT5L_PROBE_GATE_ARTIFACT_PATH
    assert evidence.predicate_passed is True


def test_l5_v2_canonical_probe_gate_evidence_skips_blocked_artifact(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / l5_v2.TT5L_PROBE_GATE_ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "probe_disambiguator": {
                    "schema": L5V2_PROBE_SCHEMA,
                    "tool_path": L5V2_PROBE_TOOL_PATH,
                    "candidate_ids": list(L5V2_CANDIDATES),
                    "paired_exact_axes_required": True,
                    "observations": [],
                    "verdict": evaluate_l5_v2_probe(()),
                    "verdict_sha256": _canonical_json_sha256(
                        evaluate_l5_v2_probe(())
                    ),
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    evidence = l5_v2.l5_v2_canonical_probe_gate_evidence(repo_root=tmp_path)
    status = l5_v2.l5_v2_probe_gate_artifact_status(repo_root=tmp_path)
    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    report = render_l5_v2_architecture_lock_packet_markdown(
        l5_v2_architecture_lock_packet(repo_root=tmp_path)
    )

    assert evidence is None
    assert status["artifact_exists"] is True
    assert status["artifact_valid"] is False
    assert status["architecture_lock_allowed"] is False
    assert "l5_v2_probe_observations_missing" in status["verdict_blockers"]
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "c1_z5_tt5l_probe_disambiguator:architecture_lock_allowed"
        in status["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "c1_z5_tt5l_probe_disambiguator:architecture_lock_allowed"
        in readiness["tt5l_campaign_readiness"]["blockers"]
    )
    assert "## Probe Gate Artifact" in report
    assert "- artifact_exists: `True`" in report
    assert "- artifact_valid: `False`" in report
    assert "l5_v2_probe_observations_missing" in report


def test_l5_v2_canonical_sideinfo_discovers_contest_full_frame_artifact(
    tmp_path: Path,
) -> None:
    artifact_path = _write_contest_sideinfo_proof_artifact(tmp_path)

    evidence = l5_v2_canonical_sideinfo_gate_evidence(repo_root=tmp_path)

    assert evidence is not None
    assert evidence.artifact_path == (
        l5_v2.TT5L_CONTEST_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_PATH
    )
    assert evidence.artifact_sha256 == _file_sha256(artifact_path)
    assert evidence.evidence_grade == "contest_full_frame_inflate_consumption_proof"


def test_l5_v2_canonical_sideinfo_prefers_committed_full_frame_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_artifact_path = _write_contest_sideinfo_proof_artifact(tmp_path)
    committed_artifact_path = _write_committed_contest_sideinfo_proof_artifact(
        tmp_path
    )
    monkeypatch.setattr(
        l5_v2,
        "TT5L_CONTEST_SIDEINFO_COMMITTED_PROOF_ARTIFACT_SHA256",
        _file_sha256(committed_artifact_path),
    )

    evidence = l5_v2_canonical_sideinfo_gate_evidence(repo_root=tmp_path)

    assert evidence is not None
    assert live_artifact_path.is_file()
    assert evidence.artifact_path == (
        l5_v2.TT5L_CONTEST_SIDEINFO_COMMITTED_PROOF_ARTIFACT_PATH
    )
    assert evidence.artifact_sha256 == _file_sha256(committed_artifact_path)
    assert evidence.evidence_grade == (
        "contest_full_frame_inflate_consumption_proof_committed_custody"
    )


def test_l5_v2_dispatch_readiness_auto_consumes_canonical_sideinfo_artifact(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    _write_contest_sideinfo_proof_artifact(tmp_path)

    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["dykstra_feasibility_artifact_valid"] is True
    assert tt5l["sideinfo_gate_evidence_valid"] is True
    assert tt5l["sideinfo_effect_curve_allowed"] is True
    assert tt5l["first_anchor_timing_smoke_allowed"] is False
    assert tt5l["next_non_pr106_l5_action"]["action_id"] == (
        "emit_c1_z5_tt5l_probe_template"
    )
    assert tt5l["next_non_pr106_l5_action"]["expected_artifacts"] == [
        l5_v2.TT5L_PROBE_DISAMBIGUATOR_TEMPLATE_PATH
    ]


def test_l5_v2_paired_axis_next_action_requires_terminal_claim_custody(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    evidence = _valid_gate_evidence_payloads(tmp_path)
    evidence.pop("paired_cpu_cuda_axis_plan")

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)
    action = readiness["tt5l_campaign_readiness"]["next_non_pr106_l5_action"]

    assert action["action_id"] == "prepare_tt5l_paired_cpu_cuda_axis_plan"
    assert action["claim_lane_before_dispatch"] is True
    assert action["terminal_claim_required"] is True
    assert action["paired_dispatch_tool"] == "tools/dispatch_modal_paired_auth_eval.py"
    assert action["preclaim_forbidden"] is True
    assert action["standalone_active_claim_command"] is None
    assert action["required_axes"] == ["contest_cpu", "contest_cuda"]
    assert action["per_axis_job_id_fields"] == {
        "contest_cpu": "contest_cpu_job_id",
        "contest_cuda": "contest_cuda_job_id",
    }
    assert "tools/recover_modal_auth_eval.py" in action["harvest_command_template"]
    assert "completed_paired_axis_plan" in action["terminal_claim_success_template"]
    assert "failed_paired_axis_plan" in action["terminal_claim_failure_template"]
    assert "tools/dispatch_modal_paired_auth_eval.py" in action["command_template"]
    assert "--pair-group-id" in action["command_template"]
    assert "[--execute only after operator approval]" in action["command_template"]
    assert "&&" not in action["command_template"]
    assert "claim_lane_dispatch.py claim" not in action["command_template"]


def test_l5_v2_tt5l_readiness_surfaces_current_lightning_paired_axis_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(l5_v2, "_git_head_commit", lambda repo_root: _sha(1))
    _write_tt5l_lightning_paired_axis_plan_artifact(tmp_path)

    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    status = readiness["tt5l_campaign_readiness"][
        "sideinfo_effect_curve_lightning_paired_axis_plan_status"
    ]

    assert status["artifact_exists"] is True
    assert status["artifact_valid"] is True
    assert status["provider"] == "lightning"
    assert status["cell_count"] == 10
    assert status["expected_cell_count"] == 10
    assert status["covered_axes"] == ["contest_cpu", "contest_cuda"]
    assert status["covered_variants"] == sorted(
        L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
    )
    assert status["source_commit"] == _sha(1)
    assert status["current_head_commit"] == _sha(1)
    assert status["source_commit_matches_head"] is True
    assert "src/tac/optimization/l5_staircase_v2.py" in status[
        "source_relevant_paths"
    ]
    assert "tools/build_l5_v2_architecture_lock_packet.py" in status[
        "source_relevant_paths"
    ]
    assert "src/tac/exact_eval_custody.py" in status["source_relevant_paths"]
    assert status["source_relevant_paths_match"] is True
    assert status["source_custody_valid"] is True
    assert status["source_custody_current_for_execution"] is True
    assert status["all_cells_dry_run_structurally_valid"] is True
    assert status["all_cells_dry_run_ready"] is True
    assert status["execution_ready"] is False
    assert status["score_claim"] is False
    assert status["promotion_eligible"] is False
    assert status["ready_for_exact_eval_dispatch"] is False
    assert status["ready_for_provider_dispatch"] is False
    assert status["dispatch_attempted"] is False
    assert (
        "l5_v2_tt5l_lightning_paired_axis_plan_dry_run_only_no_provider_job_launched"
        in status["blockers"]
    )
    assert (
        "l5_v2_tt5l_lightning_paired_axis_plan_blocked:"
        "requires_lightning_identity_and_workspace_preflight_before_submit"
        in status["blockers"]
    )
    assert readiness["tt5l_campaign_readiness"]["paired_axis_plan_evidence_valid"] is (
        False
    )

    packet = l5_v2_architecture_lock_packet(repo_root=tmp_path)
    report = render_l5_v2_architecture_lock_packet_markdown(packet)
    assert "## Lightning Paired-Axis Dry-Run Plan" in report
    assert (
        l5_v2.TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH
        in report
    )
    assert "- cells: `10`/`10`" in report
    assert f"- source_commit: `{_sha(1)}`" in report
    assert "- source_relevant_paths_match: `True`" in report
    assert "- source_custody_current_for_execution: `True`" in report
    assert "- all_cells_dry_run_ready: `True`" in report
    assert "- all_cells_dry_run_structurally_valid: `True`" in report
    assert "- execution_ready: `False`" in report
    assert "l5_v2_tt5l_lightning_paired_axis_plan_dry_run_only" in report


def test_l5_v2_tt5l_lightning_paired_axis_plan_status_allows_head_only_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(l5_v2, "_git_head_commit", lambda repo_root: _sha(2))
    monkeypatch.setattr(
        l5_v2,
        "_git_is_ancestor",
        lambda repo_root, ancestor, descendant: True,
    )
    monkeypatch.setattr(
        l5_v2,
        "_git_diff_name_only",
        lambda repo_root, base_commit, head_commit, paths: (),
    )
    _write_tt5l_lightning_paired_axis_plan_artifact(tmp_path)

    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    status = readiness["tt5l_campaign_readiness"][
        "sideinfo_effect_curve_lightning_paired_axis_plan_status"
    ]

    assert status["artifact_valid"] is True
    assert status["all_cells_dry_run_structurally_valid"] is True
    assert status["all_cells_dry_run_ready"] is True
    assert status["source_commit"] == _sha(1)
    assert status["current_head_commit"] == _sha(2)
    assert status["source_commit_matches_head"] is False
    assert status["source_commit_is_ancestor"] is True
    assert status["source_relevant_diff_paths"] == []
    assert status["source_relevant_paths_match"] is True
    assert status["source_custody_valid"] is True
    assert status["source_custody_current_for_execution"] is True
    assert (
        "l5_v2_tt5l_lightning_paired_axis_plan_source_relevant_paths_changed"
        not in status["execution_blockers"]
    )


def test_l5_v2_tt5l_lightning_paired_axis_plan_status_blocks_relevant_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_path = "src/tac/optimization/l5_staircase_v2.py"
    monkeypatch.setattr(l5_v2, "_git_head_commit", lambda repo_root: _sha(2))
    monkeypatch.setattr(
        l5_v2,
        "_git_is_ancestor",
        lambda repo_root, ancestor, descendant: True,
    )
    monkeypatch.setattr(
        l5_v2,
        "_git_diff_name_only",
        lambda repo_root, base_commit, head_commit, paths: (changed_path,),
    )
    _write_tt5l_lightning_paired_axis_plan_artifact(tmp_path)

    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    status = readiness["tt5l_campaign_readiness"][
        "sideinfo_effect_curve_lightning_paired_axis_plan_status"
    ]

    assert status["artifact_valid"] is True
    assert status["all_cells_dry_run_structurally_valid"] is True
    assert status["all_cells_dry_run_ready"] is False
    assert status["source_commit"] == _sha(1)
    assert status["current_head_commit"] == _sha(2)
    assert status["source_commit_matches_head"] is False
    assert status["source_relevant_diff_paths"] == [changed_path]
    assert status["source_relevant_paths_match"] is False
    assert status["source_custody_valid"] is False
    assert status["source_custody_current_for_execution"] is False
    assert (
        "l5_v2_tt5l_lightning_paired_axis_plan_source_relevant_paths_changed"
        in status["execution_blockers"]
    )
    assert (
        "l5_v2_tt5l_lightning_paired_axis_plan_source_relevant_paths_changed"
        in status["blockers"]
    )


def test_l5_v2_tt5l_lightning_paired_axis_plan_status_rejects_missing_axis_cell(
    tmp_path: Path,
) -> None:
    payload = _tt5l_lightning_paired_axis_plan_payload()
    cells = payload["cells"]
    assert isinstance(cells, list)
    payload["cells"] = [
        cell
        for cell in cells
        if not (cell["variant"] == "trained" and cell["axis"] == "contest_cuda")
    ]
    artifact_path = (
        tmp_path
        / l5_v2.TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    readiness = l5_v2_dispatch_readiness(repo_root=tmp_path)
    status = readiness["tt5l_campaign_readiness"][
        "sideinfo_effect_curve_lightning_paired_axis_plan_status"
    ]

    assert status["artifact_exists"] is True
    assert status["artifact_valid"] is False
    assert status["all_cells_dry_run_ready"] is False
    assert "l5_v2_tt5l_lightning_paired_axis_plan_cell_count_mismatch" in status[
        "blockers"
    ]
    assert (
        "l5_v2_tt5l_lightning_paired_axis_plan_missing_cells:"
        "trained:contest_cuda"
        in status["blockers"]
    )
    assert (
        "l5_v2_tt5l_lightning_paired_axis_plan_missing_cells:trained:contest_cuda"
        in readiness["tt5l_campaign_readiness"]["blockers"]
    )


def test_l5_v2_tt5l_architecture_lock_requires_sideinfo_effect_curve(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    evidence = _valid_gate_evidence_payloads(tmp_path)
    evidence.pop("exact_anchor_or_diagnostic_pair")

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["dykstra_feasibility_artifact_valid"] is True
    assert tt5l["sideinfo_gate_evidence_valid"] is True
    assert tt5l["probe_gate_evidence_valid"] is True
    assert tt5l["paired_axis_plan_evidence_valid"] is True
    assert tt5l["sideinfo_effect_curve_allowed"] is True
    assert tt5l["sideinfo_effect_curve_artifact_valid"] is False
    assert tt5l["sideinfo_effect_curve_status"]["artifact_valid"] is False
    assert tt5l["architecture_lock_allowed"] is False
    assert tt5l["first_anchor_timing_smoke_artifact_valid"] is False
    assert tt5l["first_anchor_timing_smoke_allowed"] is False
    assert "tt5l_sideinfo_effect_curve_artifact_missing" in tt5l["blockers"]
    assert tt5l["next_non_pr106_l5_action"]["action_id"] == (
        "measure_tt5l_sideinfo_effect_curve"
    )
    assert tt5l["next_non_pr106_l5_action"]["artifact_path"] == (
        l5_v2.TT5L_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH
    )
    assert tt5l["next_non_pr106_l5_action"]["sideinfo_effect_curve_tool_path"] == (
        "tools/build_l5_v2_sideinfo_effect_curve.py"
    )
    assert "build_l5_v2_sideinfo_effect_curve.py" in tt5l[
        "next_non_pr106_l5_action"
    ]["command_template"]
    assert (
        tt5l["next_non_pr106_l5_action"]["architecture_lock_blocker"]
        == "requires_paired_cpu_cuda_sideinfo_effect_curve_before_architecture_lock"
    )
    assert tt5l["next_non_pr106_l5_action"]["required_axes"] == [
        "contest_cpu",
        "contest_cuda",
    ]


def test_l5_v2_tt5l_sideinfo_effect_curve_status_preserves_partial_negative_evidence(
    tmp_path: Path,
) -> None:
    _write_tt5l_partial_sideinfo_effect_curve_artifact(tmp_path)

    status = l5_v2.tt5l_sideinfo_effect_curve_status(repo_root=tmp_path)

    assert status["artifact_exists"] is True
    assert status["artifact_present"] is True
    assert status["artifact_valid"] is False
    assert status["measurement_id"] == "measure_tt5l_sideinfo_effect_curve"
    assert status["predicate_passed"] is False
    assert status["observed_cell_count"] == 1
    assert "contest_cpu/trained" in status["missing_cells"]
    assert "contest_cuda/zero" in status["missing_cells"]
    assert status["effect_blockers"] == [
        "trained_not_best_or_tied:contest_cpu",
        "trained_not_best_or_tied:contest_cuda",
    ]
    observed = status["observed_cells"][0]
    assert observed["axis"] == "contest_cuda"
    assert observed["variant"] == "trained"
    assert observed["score"] == 3.9007398365396795
    assert observed["sideinfo_nonzero_fraction"] == 0.0
    assert observed["sideinfo_nonzero_values"] == 0
    assert observed["sideinfo_total_values"] == 27_000
    assert (
        "tt5l_sideinfo_effect_curve_cell_sideinfo_nonzero_missing:contest_cuda:trained"
        in status["blockers"]
    )


def test_l5_v2_architecture_lock_packet_renders_partial_sideinfo_evidence(
    tmp_path: Path,
) -> None:
    _write_tt5l_partial_sideinfo_effect_curve_artifact(tmp_path)

    packet = l5_v2_architecture_lock_packet(repo_root=tmp_path)
    report = render_l5_v2_architecture_lock_packet_markdown(packet)

    assert "## Sideinfo Effect Curve" in report
    assert "- artifact_valid: `False`" in report
    assert "- observed_cell_count: `1`" in report
    assert "contest_cuda/trained" in report
    assert "sideinfo_nonzero_fraction=`0.0`" in report
    assert "sideinfo_nonzero_values=`0`/`27000`" in report
    assert "contest_cpu/trained" in report


def test_l5_v2_tt5l_sideinfo_dispatch_plan_status_surfaces_work_queue(
    tmp_path: Path,
) -> None:
    _write_tt5l_sideinfo_effect_curve_dispatch_plan(tmp_path)

    tt5l = l5_v2_tt5l_campaign_readiness(repo_root=tmp_path)
    status = tt5l["sideinfo_effect_curve_dispatch_plan_status"]

    assert status["artifact_exists"] is True
    assert status["artifact_valid"] is True
    assert status["work_unit_count"] == 5
    assert status["ready_work_unit_count"] == 5
    assert status["ready_for_operator_dispatch"] is True
    assert status["ready_for_provider_dispatch"] is False
    assert status["score_claim"] is False
    assert status["promotion_eligible"] is False
    assert [unit["variant"] for unit in status["work_units"]] == list(
        L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
    )
    assert all(unit["archive_sha256"] for unit in status["work_units"])
    assert all(unit["pair_group_id"] for unit in status["work_units"])


def test_l5_v2_architecture_lock_packet_renders_sideinfo_dispatch_plan(
    tmp_path: Path,
) -> None:
    _write_tt5l_sideinfo_effect_curve_dispatch_plan(tmp_path)

    packet = l5_v2_architecture_lock_packet(repo_root=tmp_path)
    report = render_l5_v2_architecture_lock_packet_markdown(packet)

    assert "## TT5L Sideinfo Dispatch Plan" in report
    assert "- work_units: `5`/`5`" in report
    assert "- ready_for_provider_dispatch: `false`" in report
    assert "work_unit `random_lsb`" in report
    assert "pair_l5_v2_tt5l_sideinfo_effect_curve_random_lsb" in report


def test_l5_v2_tt5l_first_anchor_timing_requires_probe_and_paired_axis_plan(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    _write_tt5l_sideinfo_effect_curve_artifact(tmp_path)
    evidence = _valid_gate_evidence_payloads(tmp_path)
    evidence.pop("exact_anchor_or_diagnostic_pair")

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["dykstra_feasibility_artifact_valid"] is True
    assert tt5l["sideinfo_gate_evidence_valid"] is True
    assert tt5l["probe_gate_evidence_valid"] is True
    assert tt5l["paired_axis_plan_evidence_valid"] is True
    assert tt5l["sideinfo_effect_curve_allowed"] is True
    assert tt5l["sideinfo_effect_curve_artifact_valid"] is True
    assert tt5l["architecture_lock_allowed"] is False
    assert tt5l["first_anchor_timing_smoke_artifact_valid"] is False
    assert tt5l["first_anchor_timing_smoke_allowed"] is True
    assert "tt5l_first_anchor_timing_smoke_artifact_missing" in tt5l["blockers"]
    assert tt5l["next_non_pr106_l5_action"]["action_id"] == (
        "materialize_tt5l_first_anchor_timing_smoke_artifact"
    )
    assert tt5l["next_non_pr106_l5_action"]["required_axes"] == [
        "contest_cpu",
        "contest_cuda",
    ]


def test_l5_v2_tt5l_first_anchor_timing_requires_custody_artifact(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    _write_tt5l_sideinfo_effect_curve_artifact(tmp_path)
    _write_tt5l_first_anchor_timing_smoke_artifact(tmp_path)
    evidence = _valid_gate_evidence_payloads(tmp_path)
    evidence.pop("exact_anchor_or_diagnostic_pair")

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["first_anchor_timing_smoke_artifact_valid"] is True
    assert tt5l["first_anchor_timing_smoke_allowed"] is True
    assert tt5l["first_anchor_timing_smoke_status"]["provider_call_id"] == (
        "fc-test-tt5l"
    )
    assert tt5l["first_anchor_timing_smoke_status"]["seconds_per_epoch"] == 12.3
    assert tt5l["architecture_lock_allowed"] is False
    assert tt5l["anchor_pair_evidence_valid"] is False
    assert tt5l["next_non_pr106_l5_action"]["action_id"] == (
        "materialize_tt5l_exact_or_diagnostic_anchor_pair"
    )


def test_l5_v2_tt5l_first_anchor_timing_accepts_paired_axis_timings(
    tmp_path: Path,
) -> None:
    artifact_path = _write_tt5l_first_anchor_timing_smoke_artifact(tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    axis_timings = {}
    for axis, elapsed in (("contest_cpu", 293.7), ("contest_cuda", 53.2)):
        result_path = (
            tmp_path
            / "experiments"
            / "results"
            / "time_traveler_l5_v2"
            / f"{axis}_contest_auth_eval.json"
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "canonical_score": 3.9,
                    "contest_auth_eval_elapsed_seconds": elapsed,
                    "score_axis": axis,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        axis_timings[axis] = {
            "contest_auth_eval_artifact_path": str(result_path.relative_to(tmp_path)),
            "contest_auth_eval_artifact_sha256": _file_sha256(result_path),
            "contest_auth_eval_elapsed_seconds": elapsed,
            "score_axis": axis,
        }
    payload["axis_timings"] = axis_timings
    artifact_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    _write_tt5l_dykstra_artifact(tmp_path)
    _write_tt5l_sideinfo_effect_curve_artifact(tmp_path)
    evidence = _valid_gate_evidence_payloads(tmp_path)
    evidence.pop("exact_anchor_or_diagnostic_pair")

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]
    packet = l5_v2_architecture_lock_packet(gate_evidence=evidence, repo_root=tmp_path)
    report = render_l5_v2_architecture_lock_packet_markdown(packet)

    assert tt5l["first_anchor_timing_smoke_artifact_valid"] is True
    assert tt5l["first_anchor_timing_smoke_status"]["axis_timings"][
        "contest_cpu"
    ]["contest_auth_eval_elapsed_seconds"] == 293.7
    assert "## First Anchor Timing Smoke" in report
    assert "contest_cuda" in report


def test_l5_v2_tt5l_first_anchor_timing_rejects_mismatched_result_hash(
    tmp_path: Path,
) -> None:
    artifact_path = _write_tt5l_first_anchor_timing_smoke_artifact(tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["result_artifact_sha256"] = _sha(999)
    artifact_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    _write_tt5l_dykstra_artifact(tmp_path)
    _write_tt5l_sideinfo_effect_curve_artifact(tmp_path)
    evidence = _valid_gate_evidence_payloads(tmp_path)
    evidence.pop("exact_anchor_or_diagnostic_pair")

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)
    tt5l = readiness["tt5l_campaign_readiness"]

    assert tt5l["first_anchor_timing_smoke_artifact_valid"] is False
    assert (
        "tt5l_first_anchor_timing_smoke_result_artifact_sha256_mismatch"
        in tt5l["blockers"]
    )
    assert tt5l["next_non_pr106_l5_action"]["action_id"] == (
        "materialize_tt5l_first_anchor_timing_smoke_artifact"
    )


def test_l5_v2_architecture_lock_packet_requires_timing_and_anchor(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    _write_tt5l_sideinfo_effect_curve_artifact(tmp_path)
    evidence = _valid_gate_evidence_payloads(tmp_path)
    evidence.pop("exact_anchor_or_diagnostic_pair")

    packet = l5_v2_architecture_lock_packet(
        gate_evidence=evidence,
        repo_root=tmp_path,
    )

    assert packet["schema"] == l5_v2.L5_V2_ARCHITECTURE_LOCK_PACKET_SCHEMA
    assert packet["architecture_lock_allowed"] is False
    assert packet["readiness_architecture_lock_allowed"] is False
    assert packet["required_checks"]["sideinfo_effect_curve_artifact_valid"] is True
    assert packet["required_checks"]["first_anchor_timing_smoke_artifact_valid"] is False
    assert packet["required_checks"]["anchor_pair_evidence_valid"] is False
    assert "requires_tt5l_first_anchor_timing_smoke_artifact" in packet[
        "architecture_lock_blockers"
    ]
    assert "requires_exact_or_diagnostic_anchor_pair" in packet[
        "architecture_lock_blockers"
    ]
    assert packet["score_claim"] is False
    assert packet["promotion_eligible"] is False
    assert packet["ready_for_exact_eval_dispatch"] is False


def test_l5_v2_architecture_lock_packet_allows_only_after_full_custody(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)
    _write_tt5l_sideinfo_effect_curve_artifact(tmp_path)
    _write_tt5l_first_anchor_timing_smoke_artifact(tmp_path)
    packet = l5_v2_architecture_lock_packet(
        gate_evidence=_valid_gate_evidence_payloads(tmp_path),
        repo_root=tmp_path,
    )
    report = render_l5_v2_architecture_lock_packet_markdown(packet)

    assert packet["architecture_lock_allowed"] is True
    assert all(packet["required_checks"].values())
    assert packet["architecture_lock_blockers"] == []
    assert "architecture_lock_allowed: `True`" in report
    assert "- none" in report
    assert "score, rank, promotion" in report


def test_l5_v2_architecture_lock_packet_artifact_tracks_live_payload() -> None:
    live = l5_v2_architecture_lock_packet()
    artifact_path = REPO_ROOT / l5_v2.L5_V2_ARCHITECTURE_LOCK_PACKET_ARTIFACT_PATH
    report_path = REPO_ROOT / l5_v2.L5_V2_ARCHITECTURE_LOCK_PACKET_REPORT_PATH
    assert artifact_path.is_file()
    assert report_path.is_file()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    for field in (
        "schema",
        "subject_id",
        "campaign_id",
        "lane_id",
        "architecture_lock_allowed",
        "architecture_lock_blockers",
        "required_checks",
        "next_non_pr106_l5_action",
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
    ):
        assert artifact[field] == live[field]

    artifact_tt5l = artifact["tt5l_campaign_readiness"]
    live_tt5l = live["tt5l_campaign_readiness"]
    for field in (
        "next_non_pr106_l5_action",
        "dykstra_feasibility_artifact_valid",
        "move_level_feasibility_artifact_valid",
        "sideinfo_gate_evidence_valid",
        "probe_gate_evidence_valid",
        "paired_axis_plan_evidence_valid",
        "sideinfo_effect_curve_artifact_valid",
        "sideinfo_effect_curve_status",
        "materialized_tt5l_paired_work_unit_status",
        "blockers",
    ):
        assert artifact_tt5l[field] == live_tt5l[field]
    volatile_source_fields = {
        "current_head_commit",
        "source_commit_matches_head",
    }
    artifact_plan_status = artifact_tt5l[
        "sideinfo_effect_curve_lightning_paired_axis_plan_status"
    ].copy()
    live_plan_status = live_tt5l[
        "sideinfo_effect_curve_lightning_paired_axis_plan_status"
    ].copy()
    for field in volatile_source_fields:
        artifact_plan_status.pop(field, None)
        live_plan_status.pop(field, None)
    assert artifact_plan_status == live_plan_status


def test_l5_v2_architecture_lock_packet_cli_writes_no_lock_packet(
    tmp_path: Path,
) -> None:
    fake_repo_root = (
        REPO_ROOT
        / "experiments"
        / "results"
        / "time_traveler_l5_v2"
        / f"test_arch_lock_packet_{tmp_path.name}"
    )
    output_json = fake_repo_root / ".omx/research/architecture_lock_packet.json"
    output_md = fake_repo_root / ".omx/research/architecture_lock_packet.md"

    try:
        proc = subprocess.run(
            [
                sys.executable,
                "tools/build_l5_v2_architecture_lock_packet.py",
                "--repo-root",
                str(fake_repo_root),
                "--output-json",
                str(output_json.relative_to(fake_repo_root)),
                "--output-md",
                str(output_md.relative_to(fake_repo_root)),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert proc.returncode == 0, proc.stdout + proc.stderr
        packet = json.loads(output_json.read_text(encoding="utf-8"))
        report = output_md.read_text(encoding="utf-8")
        assert packet["architecture_lock_allowed"] is False
        assert "architecture_lock_allowed=false" in proc.stdout
        assert "score_claim=false" in proc.stdout
        assert "requires_all_l5_v2_gate_evidence_valid" in packet[
            "architecture_lock_blockers"
        ]
        assert "L5 v2 architecture lock packet" in report
    finally:
        if fake_repo_root.exists():
            shutil.rmtree(fake_repo_root)


def test_l5_v2_packetir_stack_evidence_fails_closed_without_matrix(
    tmp_path: Path,
) -> None:
    payload = l5_v2_packetir_stack_evidence_payload(repo_root=tmp_path)

    assert payload["paired_candidate_count"] == 0
    assert "l5_v2_packetir_matrix_artifact_missing" in payload["blockers"]
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_l5_v2_pr106_stack_cell_candidates_are_blocked_proposals() -> None:
    payload = l5_v2_pr106_stack_cell_candidates()

    assert payload["schema"] == L5_V2_PR106_STACK_CELL_CANDIDATES_SCHEMA
    assert payload["source_packetir_paired_candidate_count"] == 0
    assert payload["candidate_count"] == 0
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["candidates"] == []
    assert "l5_v2_packetir_no_runtime_bound_paired_exact_candidates" in payload[
        "blockers"
    ]
    assert "l5_v2_pr106_stack_cell_candidates_missing" in payload["blockers"]


def test_l5_v2_pr106_stack_cell_candidates_honor_top_k() -> None:
    payload = l5_v2_pr106_stack_cell_candidates(top_k=2)

    assert payload["candidate_count"] == 0
    assert payload["candidates"] == []
    assert payload["score_claim"] is False


def test_l5_v2_pr106_stack_cell_candidates_label_worst_axis_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix_relpath = Path(".omx/research/test_pr106_packetir_matrix.json")
    matrix_path = tmp_path / matrix_relpath
    matrix_path.parent.mkdir(parents=True)
    archive_bytes = 186327
    cpu_seg = 0.00063
    cpu_pose = 0.00016
    cuda_seg = 0.00064
    cuda_pose = 0.00003
    cpu_score = 100 * cpu_seg + math.sqrt(10 * cpu_pose) + 25 * archive_bytes / 37_545_489
    cuda_score = 100 * cuda_seg + math.sqrt(10 * cuda_pose) + 25 * archive_bytes / 37_545_489
    for relpath in (
        "experiments/results/cpu/result.json",
        "experiments/results/cpu/result.log",
        "experiments/results/cuda/result.json",
        "experiments/results/cuda/result.log",
    ):
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relpath}\n", encoding="utf-8")
    matrix = {
        "schema": "pr106_packetir_candidate_matrix_v1",
        "candidate_count": 1,
        "status_counts": {"paired_exact_measured": 1},
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rows": [
            {
                "candidate_id": "format_0x0c_exact_radix",
                "format_id": "0x0C",
                "notes": "exact-radix grammar test",
                "status": "paired_exact_measured",
                "archive_sha256": _sha(12),
                "archive_path": "submissions/pr106/archive.zip",
                "source_artifact_warnings": [],
                "runtime_consumption": {
                    "valid": True,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "path": "experiments/results/runtime_consumption.json",
                    "sha256": _sha(21),
                    "sidecar_kind": "exact_radix_dim_fixed_meta",
                    "runtime_dir": "submissions/pr106_runtime",
                    "runtime_content_tree_sha256": _sha(22),
                    "runtime_content_tree_sha256_source": (
                        "direct_runtime_consumption_manifest"
                    ),
                    "runtime_content_tree_sha256_matches_current_runtime_dir": True,
                    "current_modal_uploaded_runtime": {
                        "runtime_content_tree_sha256": _sha(22),
                        "runtime_tree_sha256": _sha(23),
                    },
                },
                "exact_axis_evidence": {
                    "contest_cpu": {
                        "valid": True,
                        "canonical_score": cpu_score,
                        "score": cpu_score,
                        "archive_sha256": _sha(12),
                        "archive_size_bytes": archive_bytes,
                        "archive_bytes": archive_bytes,
                        "avg_segnet_dist": cpu_seg,
                        "seg_dist": cpu_seg,
                        "avg_posenet_dist": cpu_pose,
                        "pose_dist": cpu_pose,
                        "axis": "contest_cpu",
                        "runtime_tree_sha256": _sha(23),
                        "n_samples": 600,
                        "hardware": "linux-x86_64 cpu",
                        "inflate_device": "cpu",
                        "eval_device": "cpu",
                        "auth_eval_command": "contest_auth_eval.py --device cpu",
                        "artifact_path": "experiments/results/cpu/result.json",
                        "log_path": "experiments/results/cpu/result.log",
                        "sha256": _sha(31),
                        "score_claim_in_source_artifact": False,
                        "promotion_eligible_in_source_artifact": False,
                    },
                    "contest_cuda": {
                        "valid": True,
                        "canonical_score": cuda_score,
                        "score": cuda_score,
                        "archive_sha256": _sha(12),
                        "archive_size_bytes": archive_bytes,
                        "archive_bytes": archive_bytes,
                        "avg_segnet_dist": cuda_seg,
                        "seg_dist": cuda_seg,
                        "avg_posenet_dist": cuda_pose,
                        "pose_dist": cuda_pose,
                        "axis": "contest_cuda",
                        "runtime_tree_sha256": _sha(23),
                        "n_samples": 600,
                        "hardware": "Tesla T4",
                        "inflate_device": "cuda",
                        "eval_device": "cuda",
                        "auth_eval_command": "contest_auth_eval.py --device cuda",
                        "artifact_path": "experiments/results/cuda/result.json",
                        "log_path": "experiments/results/cuda/result.log",
                        "sha256": _sha(32),
                        "score_claim_in_source_artifact": False,
                        "promotion_eligible_in_source_artifact": False,
                    },
                },
            }
        ],
    }
    matrix_path.write_text(
        json.dumps(matrix, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        l5_v2,
        "PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_PATH",
        str(matrix_relpath),
    )
    monkeypatch.setattr(
        l5_v2,
        "PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_SHA256",
        _file_sha256(matrix_path),
    )

    payload = l5_v2_pr106_stack_cell_candidates(repo_root=tmp_path)

    assert payload["candidate_count"] == 1
    candidate = payload["candidates"][0]
    assert candidate["source_axis_scores"] == {
        "contest_cpu": cpu_score,
        "contest_cuda": cuda_score,
    }
    assert candidate["source_worst_axis_score"] == max(cpu_score, cuda_score)
    assert "source_max_axis_score" not in candidate
    assert candidate["packetir_sidecar_kind"] == "exact_radix_dim_fixed_meta"
    assert candidate["packetir_notes"] == "exact-radix grammar test"
    assert candidate["packetir_source_artifact_warnings"] == []
    assert candidate["source_cpu_cuda_score_gap"] == pytest.approx(
        cpu_score - cuda_score
    )
    component_delta = candidate["source_cpu_minus_cuda_component_delta"]
    assert component_delta["canonical_score"] == pytest.approx(cpu_score - cuda_score)
    assert component_delta["avg_segnet_dist"] == pytest.approx(-0.00001)
    assert component_delta["avg_posenet_dist"] == pytest.approx(0.00013)
    assert candidate["source_runtime_dir"] == "submissions/pr106_runtime"
    assert candidate["source_runtime_content_tree_sha256"] == _sha(22)
    assert candidate["current_runtime_content_tree_sha256"] == _sha(22)
    assert (
        candidate["source_runtime_content_tree_sha256_matches_current_runtime_dir"]
        is True
    )
    assert candidate["score_claim"] is False


def test_l5_v2_packetir_paired_rows_revalidate_exact_eval_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix_relpath = Path(".omx/research/test_pr106_packetir_matrix_missing_samples.json")
    matrix_path = tmp_path / matrix_relpath
    matrix_path.parent.mkdir(parents=True)
    for relpath in (
        "experiments/results/cpu/result.json",
        "experiments/results/cpu/result.log",
        "experiments/results/cuda/result.json",
        "experiments/results/cuda/result.log",
    ):
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relpath}\n", encoding="utf-8")
    archive_bytes = 1
    score = 25 * archive_bytes / 37_545_489
    axis_rows = {}
    for axis in ("contest_cpu", "contest_cuda"):
        device = "cpu" if axis == "contest_cpu" else "cuda"
        axis_rows[axis] = {
            "valid": True,
            "axis": axis,
            "canonical_score": score,
            "score": score,
            "archive_sha256": _sha(12),
            "archive_size_bytes": archive_bytes,
            "archive_bytes": archive_bytes,
            "avg_segnet_dist": 0.0,
            "seg_dist": 0.0,
            "avg_posenet_dist": 0.0,
            "pose_dist": 0.0,
            "runtime_tree_sha256": _sha(23),
            "n_samples": 600,
            "hardware": "linux-x86_64 cpu" if device == "cpu" else "Tesla T4",
            "inflate_device": device,
            "eval_device": device,
            "auth_eval_command": f"contest_auth_eval.py --device {device}",
            "artifact_path": f"experiments/results/{device}/result.json",
            "log_path": f"experiments/results/{device}/result.log",
            "score_claim_in_source_artifact": False,
            "promotion_eligible_in_source_artifact": False,
        }
    axis_rows["contest_cpu"].pop("n_samples")
    matrix = {
        "schema": "pr106_packetir_candidate_matrix_v1",
        "candidate_count": 1,
        "status_counts": {"paired_exact_measured": 1},
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rows": [
            {
                "candidate_id": "format_0x0c_missing_samples",
                "format_id": "0x0C",
                "status": "paired_exact_measured",
                "archive_sha256": _sha(12),
                "archive_path": "submissions/pr106/archive.zip",
                "runtime_consumption": {
                    "valid": True,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "runtime_content_tree_sha256_matches_current_runtime_dir": True,
                    "current_modal_uploaded_runtime": {
                        "runtime_content_tree_sha256": _sha(22),
                        "runtime_tree_sha256": _sha(23),
                    },
                },
                "exact_axis_evidence": axis_rows,
            }
        ],
    }
    matrix_path.write_text(
        json.dumps(matrix, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        l5_v2,
        "PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_PATH",
        str(matrix_relpath),
    )
    monkeypatch.setattr(
        l5_v2,
        "PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_SHA256",
        _file_sha256(matrix_path),
    )

    payload = l5_v2_pr106_stack_cell_candidates(repo_root=tmp_path)

    assert payload["candidate_count"] == 0
    assert any(
        "l5_v2_packetir_matrix_paired_row_axis_blocked:"
        "format_0x0c_missing_samples:" in str(blocker)
        and "exact_eval:contest_cpu:n_samples_missing" in str(blocker)
        for blocker in payload["blockers"]
    )


def test_l5_v2_pr106_stack_cell_candidates_fail_closed_without_matrix(
    tmp_path: Path,
) -> None:
    payload = l5_v2_pr106_stack_cell_candidates(repo_root=tmp_path)

    assert payload["candidate_count"] == 0
    assert "l5_v2_packetir_matrix_artifact_missing" in payload["blockers"]
    assert "l5_v2_pr106_stack_cell_candidates_missing" in payload["blockers"]
    assert payload["promotion_eligible"] is False


def test_l5_v2_dispatch_readiness_requires_artifact_evidence_not_booleans(
    tmp_path: Path,
) -> None:
    blocked = l5_v2_dispatch_readiness(repo_root=tmp_path)
    all_gate_ids = {gate.gate_id for gate in l5_v2_required_gates()}
    boolean_only = l5_v2_dispatch_readiness(
        dict.fromkeys(all_gate_ids, True),
        repo_root=tmp_path,
    )

    assert blocked["ready_for_dispatch"] is False
    assert blocked["all_gate_claims_satisfied"] is False
    assert blocked["all_gate_evidence_valid"] is False
    assert blocked["promotion_eligible"] is False
    assert blocked["ready_for_exact_eval_dispatch"] is False
    assert blocked["rank_or_kill_eligible"] is False
    assert {
        "requires_byte_closed_temporal_sideinfo_consumption_proof",
        "requires_c1_z5_tt5l_probe_disambiguator_before_architecture_lock",
        "requires_paired_cpu_cuda_axis_plan_before_promotion",
        "requires_l5_v2_empirical_anchor",
    } <= set(blocked["blockers"])
    assert any(
        blocker.startswith("l5_v2_gate_evidence_missing:")
        for blocker in blocked["blockers"]
    )
    assert boolean_only["all_gate_claims_satisfied"] is False
    assert boolean_only["all_gate_evidence_valid"] is False
    assert boolean_only["ready_for_gate_probe_dispatch"] is False
    assert boolean_only["ready_for_score_or_rank_dispatch"] is False
    assert boolean_only["ready_for_dispatch"] is False
    assert boolean_only["promotion_eligible"] is False
    assert boolean_only["ready_for_exact_eval_dispatch"] is False
    assert boolean_only["rank_or_kill_eligible"] is False
    assert boolean_only["score_claim"] is False
    assert all(gate["claimed_satisfied"] is True for gate in boolean_only["gates"])
    assert all(
        gate["claimed_satisfied_without_artifact"] is True
        for gate in boolean_only["gates"]
    )
    assert all(gate["status"] == "required" for gate in boolean_only["gates"])
    assert all(gate["evidence_valid"] is False for gate in boolean_only["gates"])
    assert blocked["packetir_stack_evidence"]["paired_candidate_count"] == 0
    assert blocked["packetir_stack_evidence"]["score_claim"] is False
    assert blocked["pr106_stack_cell_candidates"]["candidate_count"] == 0
    assert blocked["pr106_stack_cell_candidates"]["promotion_eligible"] is False


def test_l5_v2_dispatch_readiness_auto_consumes_paired_axis_plan_from_anchor_artifact(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence(tmp_path)
    evidence.pop("paired_cpu_cuda_axis_plan")
    artifact_path = (
        tmp_path / l5_v2.TT5L_PAIRED_AXIS_PLAN_FROM_ANCHOR_ARTIFACT_PATH
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _gate_artifact_payload("paired_cpu_cuda_axis_plan", repo_root=tmp_path)
    payload.update(
        {
            "schema": "l5_v2_tt5l_paired_cpu_cuda_axis_plan_from_anchor_v1",
            "gate_id": "paired_cpu_cuda_axis_plan",
            "predicate_id": l5_v2.TT5L_PAIRED_AXIS_PLAN_FROM_ANCHOR_PREDICATE_ID,
            "passed": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    )
    artifact_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    canonical = l5_v2.l5_v2_canonical_paired_axis_plan_gate_evidence(
        repo_root=tmp_path
    )
    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert canonical is not None
    assert canonical.gate_id == "paired_cpu_cuda_axis_plan"
    assert canonical.artifact_sha256 == _file_sha256(artifact_path)
    assert readiness["tt5l_campaign_readiness"]["paired_axis_plan_evidence_valid"] is True
    assert (
        "l5_v2_gate_evidence_missing:paired_cpu_cuda_axis_plan"
        not in readiness["blockers"]
    )


def test_l5_v2_dispatch_readiness_accepts_valid_gate_evidence(tmp_path: Path) -> None:
    ready = l5_v2_dispatch_readiness(
        gate_evidence=_valid_gate_evidence(tmp_path),
        repo_root=tmp_path,
    )

    assert ready["all_gate_claims_satisfied"] is True
    assert ready["all_gate_evidence_valid"] is True
    assert ready["ready_for_gate_probe_dispatch"] is False
    assert ready["tt5l_cargo_cult_preconditions_valid"] is False
    assert "tt5l_cargo_cult_preconditions_not_gate_probe_ready" in ready["blockers"]
    assert ready["ready_for_score_or_rank_dispatch"] is False
    assert ready["ready_for_dispatch"] is False
    assert "prediction_band_not_dispatch_ready" in ready["blockers"]
    assert "prediction_band:prediction_band_baseline_missing" in ready["blockers"]
    assert "prediction_band:prediction_band_empirical_anchor_missing" in ready[
        "blockers"
    ]
    assert ready["prediction_band_rank_ready"] is False
    assert ready["promotion_eligible"] is False
    assert ready["ready_for_exact_eval_dispatch"] is False
    assert ready["rank_or_kill_eligible"] is False
    assert ready["score_claim"] is False
    assert all(gate["evidence_valid"] is True for gate in ready["gates"])


def test_l5_v2_dispatch_readiness_gate_probe_requires_tt5l_feasibility(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)

    ready = l5_v2_dispatch_readiness(
        gate_evidence=_valid_gate_evidence(tmp_path),
        repo_root=tmp_path,
    )

    assert ready["all_gate_evidence_valid"] is True
    assert ready["tt5l_cargo_cult_preconditions_valid"] is True
    assert ready["ready_for_gate_probe_dispatch"] is True
    assert "tt5l_cargo_cult_preconditions_not_gate_probe_ready" not in ready["blockers"]
    assert ready["ready_for_score_or_rank_dispatch"] is False
    assert ready["ready_for_dispatch"] is False


def test_l5_v2_valid_gates_do_not_unlock_blocked_prediction_band(
    tmp_path: Path,
) -> None:
    _write_tt5l_dykstra_artifact(tmp_path)

    readiness = l5_v2_dispatch_readiness(
        gate_evidence=_valid_gate_evidence(tmp_path),
        repo_root=tmp_path,
    )

    assert readiness["all_gate_evidence_valid"] is True
    assert readiness["ready_for_gate_probe_dispatch"] is True
    assert readiness["prediction_band_verdict"]["valid_for_rank_reward"] is False
    assert readiness["ready_for_score_or_rank_dispatch"] is False
    assert readiness["ready_for_dispatch"] is False
    assert "prediction_band_not_dispatch_ready" in readiness["blockers"]


@pytest.mark.parametrize(
    ("field", "value", "expected_blocker"),
    [
        ("artifact_path", "", "l5_v2_gate_artifact_path_missing:"),
        ("artifact_path", "/tmp/l5v2.json", "l5_v2_gate_artifact_path_transient:"),
        ("artifact_sha256", "not-a-sha", "l5_v2_gate_artifact_sha256_invalid:"),
        ("predicate_id", "", "l5_v2_gate_predicate_id_missing:"),
        ("predicate_passed", False, "l5_v2_gate_predicate_failed:"),
    ],
)
def test_l5_v2_dispatch_readiness_rejects_bad_gate_evidence(
    tmp_path: Path,
    field: str,
    value: object,
    expected_blocker: str,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    first_gate_id = next(iter(evidence))
    evidence[first_gate_id][field] = value

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_claims_satisfied"] is False
    assert readiness["all_gate_evidence_valid"] is False
    assert readiness["ready_for_dispatch"] is False
    assert f"{expected_blocker}{first_gate_id}" in readiness["blockers"]


@pytest.mark.parametrize("value", ["false", "yes", 1])
def test_l5_v2_dispatch_readiness_rejects_non_bool_gate_claims(
    value: object,
) -> None:
    first_gate_id = l5_v2_required_gates()[0].gate_id

    readiness = l5_v2_dispatch_readiness({first_gate_id: value})
    gate = next(row for row in readiness["gates"] if row["gate_id"] == first_gate_id)

    assert gate["claimed_satisfied"] is False
    assert f"l5_v2_gate_claim_non_bool:{first_gate_id}" in readiness["blockers"]
    assert f"l5_v2_gate_claim_non_bool:{first_gate_id}" in gate["claim_blockers"]
    assert readiness["ready_for_dispatch"] is False


@pytest.mark.parametrize("value", ["false", "yes", 1])
def test_l5_v2_dispatch_readiness_rejects_non_bool_gate_evidence_predicate(
    tmp_path: Path,
    value: object,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    first_gate_id = next(iter(evidence))
    evidence[first_gate_id]["predicate_passed"] = value

    with pytest.raises(ValueError, match="predicate_passed must be a literal JSON boolean"):
        l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)


def test_l5_v2_dispatch_readiness_rejects_non_object_gate_evidence(
    tmp_path: Path,
) -> None:
    evidence: dict[str, object] = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "byte_closed_temporal_sideinfo_consumption"
    evidence[gate_id] = True

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert f"l5_v2_gate_evidence_non_object:{gate_id}" in readiness["blockers"]


@pytest.mark.parametrize("value", ["false", "yes", 1])
def test_l5_v2_dispatch_readiness_rejects_non_bool_gate_artifact_predicate(
    tmp_path: Path,
    value: object,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    first_gate_id = next(iter(evidence))
    artifact_path = tmp_path / str(evidence[first_gate_id]["artifact_path"])
    artifact_path.write_text(
        json.dumps({"gate_id": first_gate_id, "passed": value}) + "\n",
        encoding="utf-8",
    )
    evidence[first_gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert readiness["ready_for_gate_probe_dispatch"] is False
    assert (
        f"l5_v2_gate_artifact_predicate_non_bool:{first_gate_id}:passed"
        in readiness["blockers"]
    )


def test_l5_v2_dispatch_readiness_rejects_predicate_only_gate_artifacts(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    for gate_id, row in evidence.items():
        artifact_path = tmp_path / str(row["artifact_path"])
        artifact_path.write_text(
            json.dumps({"gate_id": gate_id, "passed": True}) + "\n",
            encoding="utf-8",
        )
        row["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert readiness["ready_for_gate_probe_dispatch"] is False
    assert (
        "l5_v2_gate_artifact_semantics_missing:"
        "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_missing:"
        "c1_z5_tt5l_probe_disambiguator:probe_disambiguator"
        in readiness["blockers"]
    )
    assert any(
        str(blocker).startswith(
            "l5_v2_gate_artifact_semantics_missing:"
            "paired_cpu_cuda_axis_plan:paired_axis_plan:"
        )
        for blocker in readiness["blockers"]
    )
    assert any(
        str(blocker).startswith(
            "l5_v2_gate_artifact_semantics_missing:"
            "exact_anchor_or_diagnostic_pair:anchor_pair:"
        )
        for blocker in readiness["blockers"]
    )


def test_l5_v2_probe_gate_rejects_metadata_stub_without_probe_verdict(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "c1_z5_tt5l_probe_disambiguator"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    probe = payload["probe_disambiguator"]
    assert isinstance(probe, dict)
    probe.pop("verdict")
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert readiness["ready_for_gate_probe_dispatch"] is False
    assert (
        "l5_v2_gate_artifact_semantics_missing:"
        "c1_z5_tt5l_probe_disambiguator:probe_verdict"
        in readiness["blockers"]
    )


def test_l5_v2_probe_gate_rejects_blocked_or_incomplete_probe_verdict(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "c1_z5_tt5l_probe_disambiguator"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    probe = payload["probe_disambiguator"]
    assert isinstance(probe, dict)
    verdict = probe["verdict"]
    assert isinstance(verdict, dict)
    verdict["architecture_lock_allowed"] = False
    verdict["blockers"] = ["l5_v2_probe_required_candidate_ineligible:c1"]
    verdict["evaluated_observations"] = [
        row
        for row in verdict["evaluated_observations"]
        if row["candidate_id"] != "time_traveler_l5_autonomy"
    ]
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "c1_z5_tt5l_probe_disambiguator:architecture_lock_allowed"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "c1_z5_tt5l_probe_disambiguator:probe_blockers_nonempty"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_missing:"
        "c1_z5_tt5l_probe_disambiguator:eligible_observations:"
        "time_traveler_l5_autonomy"
        in readiness["blockers"]
    )


def test_l5_v2_probe_gate_recomputes_verdict_from_observations(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "c1_z5_tt5l_probe_disambiguator"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    probe = payload["probe_disambiguator"]
    assert isinstance(probe, dict)
    observations = probe["observations"]
    assert isinstance(observations, list)
    tt5l = next(
        row for row in observations if row["candidate_id"] == "time_traveler_l5_autonomy"
    )
    tt5l["predicted_or_measured_delta"] = -0.030
    for axis_row in tt5l["axis_evidence"]:
        axis_row["score_delta"] = -0.005
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "c1_z5_tt5l_probe_disambiguator:probe_verdict_sha256_mismatch"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "c1_z5_tt5l_probe_disambiguator:probe_verdict_recompute_mismatch"
        in readiness["blockers"]
    )


def test_l5_v2_probe_gate_rejects_selected_candidate_not_min_eligible_delta(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "c1_z5_tt5l_probe_disambiguator"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    probe = payload["probe_disambiguator"]
    assert isinstance(probe, dict)
    verdict = probe["verdict"]
    assert isinstance(verdict, dict)
    verdict["selected_candidate_id"] = "c1_world_model_foveation"
    verdict["selected_delta"] = -0.010
    probe["verdict_sha256"] = _canonical_json_sha256(verdict)
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "c1_z5_tt5l_probe_disambiguator:probe_verdict_sha256_mismatch"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "c1_z5_tt5l_probe_disambiguator:probe_verdict_recompute_mismatch"
        in readiness["blockers"]
    )


def test_l5_v2_dispatch_readiness_rejects_invalid_paired_axis_semantics(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "paired_cpu_cuda_axis_plan"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    rows = payload["paired_axis_plan"]
    assert isinstance(rows, list)
    cuda_row = next(row for row in rows if row["axis"] == "contest_cuda")
    cuda_row["archive_sha256"] = _sha(99)
    cuda_row["inflate_device"] = "cpu"
    cuda_row["component_deltas"] = {"seg_dist_delta": "0"}
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "paired_cpu_cuda_axis_plan:paired_axis_plan:archive_sha256"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "paired_cpu_cuda_axis_plan:paired_axis_plan:contest_cuda_inflate_device"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "paired_cpu_cuda_axis_plan:paired_axis_plan:contest_cuda:seg_dist_delta"
        in readiness["blockers"]
    )


def test_l5_v2_dispatch_readiness_allows_axis_specific_runtime_trees(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "paired_cpu_cuda_axis_plan"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    rows = payload["paired_axis_plan"]
    assert isinstance(rows, list)
    for row in rows:
        if row["axis"] == "contest_cpu":
            row["runtime_tree_sha256"] = _sha(31)
        else:
            row["runtime_tree_sha256"] = _sha(32)
        row["runtime_content_tree_sha256"] = _sha(33)
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "paired_cpu_cuda_axis_plan:paired_axis_plan:runtime_tree_sha256"
        not in readiness["blockers"]
    )
    assert readiness["all_gate_evidence_valid"] is True


def test_l5_v2_dispatch_readiness_rejects_runtime_content_tree_mismatch(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "paired_cpu_cuda_axis_plan"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    rows = payload["paired_axis_plan"]
    assert isinstance(rows, list)
    cuda_row = next(row for row in rows if row["axis"] == "contest_cuda")
    cuda_row["runtime_content_tree_sha256"] = _sha(44)
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "paired_cpu_cuda_axis_plan:paired_axis_plan:runtime_content_tree_sha256"
        in readiness["blockers"]
    )


def test_l5_v2_dispatch_readiness_rejects_duplicate_axis_rows(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "paired_cpu_cuda_axis_plan"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    rows = payload["paired_axis_plan"]
    assert isinstance(rows, list)
    rows.append(dict(next(row for row in rows if row["axis"] == "contest_cpu")))
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "paired_cpu_cuda_axis_plan:paired_axis_plan:duplicate_axis:contest_cpu"
        in readiness["blockers"]
    )


def test_l5_v2_dispatch_readiness_rejects_negated_cuda_axis_semantics(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "paired_cpu_cuda_axis_plan"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    rows = payload["paired_axis_plan"]
    assert isinstance(rows, list)
    cuda_row = next(row for row in rows if row["axis"] == "contest_cuda")
    cuda_row["inflate_device"] = "cpu-no-cuda"
    cuda_row["eval_device"] = "cuda-disabled"
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "paired_cpu_cuda_axis_plan:paired_axis_plan:contest_cuda_inflate_device"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "paired_cpu_cuda_axis_plan:paired_axis_plan:contest_cuda_eval_device"
        in readiness["blockers"]
    )


def test_l5_v2_dispatch_readiness_rejects_macos_cpu_axis_semantics(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "paired_cpu_cuda_axis_plan"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    rows = payload["paired_axis_plan"]
    assert isinstance(rows, list)
    cpu_row = next(row for row in rows if row["axis"] == "contest_cpu")
    cpu_row["inflate_device"] = "macos-apple-m2-cpu"
    cpu_row["eval_device"] = "cpu-mps"
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "paired_cpu_cuda_axis_plan:paired_axis_plan:contest_cpu_inflate_device"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "paired_cpu_cuda_axis_plan:paired_axis_plan:contest_cpu_eval_device"
        in readiness["blockers"]
    )


def test_l5_v2_dispatch_readiness_rejects_cpu_axis_cuda_device_leak(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "paired_cpu_cuda_axis_plan"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    rows = payload["paired_axis_plan"]
    assert isinstance(rows, list)
    cpu_row = next(row for row in rows if row["axis"] == "contest_cpu")
    cpu_row["inflate_device"] = "cpu cuda"
    cpu_row["eval_device"] = "linux-x86_64 gpu"
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "paired_cpu_cuda_axis_plan:paired_axis_plan:contest_cpu_inflate_device"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "paired_cpu_cuda_axis_plan:paired_axis_plan:contest_cpu_eval_device"
        in readiness["blockers"]
    )


def test_l5_v2_sideinfo_consumption_requires_full_frame_inflate_custody(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "byte_closed_temporal_sideinfo_consumption"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    proof = payload["byte_mutation_proof"]
    assert isinstance(proof, dict)
    proof.pop("inflated_outputs_manifest_path", None)
    proof.pop("inflated_raw_output_aggregate_sha256", None)
    proof["inflate_command"] = "python inflate.py"
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert (
        "l5_v2_gate_artifact_semantics_missing:"
        "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
        "inflated_outputs_manifest_path"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
        "inflated_raw_output_aggregate_sha256"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:inflate_command"
        in readiness["blockers"]
    )


def test_l5_v2_sideinfo_consumption_requires_log_bound_inflate_provenance(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "byte_closed_temporal_sideinfo_consumption"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    proof = payload["byte_mutation_proof"]
    assert isinstance(proof, dict)
    provenance = proof["inflate_provenance"]
    assert isinstance(provenance, dict)
    baseline = provenance["baseline"]
    assert isinstance(baseline, dict)
    baseline.pop("log_sha256")
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert (
        "l5_v2_gate_artifact_semantics_missing:"
        "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
        "inflate_provenance:baseline:log_sha256"
        in readiness["blockers"]
    )


def test_l5_v2_sideinfo_consumption_rejects_toy_manifest_scope(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "byte_closed_temporal_sideinfo_consumption"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    payload["proof_scope"] = "local_no_gpu_toy_tt5l_archive_parser_and_inflate_consumption_only"
    proof = payload["byte_mutation_proof"]
    assert isinstance(proof, dict)
    proof["n_pairs_hashed"] = 1
    proof["total_frames"] = 2
    proof.pop("file_list_sha256", None)
    proof.pop("baseline_raw_output_aggregate_sha256", None)
    manifest_path = tmp_path / str(proof["inflated_outputs_manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "raw_output_aggregate_sha256": proof[
                    "inflated_raw_output_aggregate_sha256"
                ],
                "n_pairs_hashed": 1,
                "total_frames": 2,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    blockers = readiness["blockers"]
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
        "proof_scope_not_contest_full_frame"
        in blockers
    )
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
        "n_pairs_hashed"
        in blockers
    )
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
        "total_frames"
        in blockers
    )


def test_l5_v2_sideinfo_consumption_binds_mutation_to_parsed_section_range(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "byte_closed_temporal_sideinfo_consumption"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    proof = payload["byte_mutation_proof"]
    assert isinstance(proof, dict)
    proof["mutated_byte_offsets"] = [0, 1024]
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
        "mutated_byte_offsets_outside_section"
        in readiness["blockers"]
    )


def test_l5_v2_sideinfo_consumption_requires_non_target_section_identity(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "byte_closed_temporal_sideinfo_consumption"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    proof = payload["byte_mutation_proof"]
    assert isinstance(proof, dict)
    proof["non_target_sections_identical"] = False
    proof["non_target_payload_sections_identical"] = False
    section_hashes = proof["section_hashes"]
    assert isinstance(section_hashes, dict)
    world = section_hashes["world_model_blob"]
    assert isinstance(world, dict)
    world["mutated_sha256"] = _sha(99)
    world["identical"] = False
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
        "non_target_payload_sections_identical"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
        "section_hashes:world_model_blob"
        in readiness["blockers"]
    )


def test_l5_v2_sideinfo_consumption_requires_archive_runtime_section_identity(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "byte_closed_temporal_sideinfo_consumption"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    proof = payload["byte_mutation_proof"]
    assert isinstance(proof, dict)
    proof.pop("section_sha256", None)
    proof.pop("runtime_tree_sha256", None)
    proof["mutated_archive_sha256"] = proof["baseline_archive_sha256"]
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:section_sha256"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
        "runtime_tree_sha256"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:archive_sha_pair"
        in readiness["blockers"]
    )


def test_l5_v2_sideinfo_consumption_binds_raw_aggregate_to_manifest(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "byte_closed_temporal_sideinfo_consumption"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    proof = payload["byte_mutation_proof"]
    assert isinstance(proof, dict)
    manifest_path = tmp_path / str(proof["inflated_outputs_manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"aggregate_sha256": _sha(99)}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "byte_closed_temporal_sideinfo_consumption:byte_mutation_proof:"
        "inflated_outputs_manifest_path:aggregate_sha256_mismatch"
        in readiness["blockers"]
    )


def test_l5_v2_dispatch_readiness_rejects_invalid_anchor_semantics(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "exact_anchor_or_diagnostic_pair"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    rows = payload["anchor_pair"]
    assert isinstance(rows, list)
    cuda_row = next(row for row in rows if row["axis"] == "contest_cuda")
    cuda_row["score_claim"] = True
    cuda_row["evidence_grade"] = "advisory_proxy"
    cpu_row = next(row for row in rows if row["axis"] == "contest_cpu")
    cpu_row["anchor_type"] = "diagnostic"
    cpu_row["evidence_grade"] = "contest_paired_exact"
    cpu_row.pop("diagnostic_reason", None)
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "exact_anchor_or_diagnostic_pair:anchor_pair:contest_cuda:score_claim"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "exact_anchor_or_diagnostic_pair:anchor_pair:contest_cuda:evidence_grade"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "exact_anchor_or_diagnostic_pair:anchor_pair:contest_cpu:evidence_grade"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_missing:"
        "exact_anchor_or_diagnostic_pair:anchor_pair:contest_cpu:diagnostic_reason"
        in readiness["blockers"]
    )


def test_l5_v2_dispatch_readiness_consumes_pair_anchor_artifact_over_stale_axis_rows(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    evidence.pop("exact_anchor_or_diagnostic_pair")
    gate_id = "exact_anchor_or_diagnostic_pair"
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    payload["schema"] = "l5_v2_tt5l_paired_exact_anchor_pair_v1"
    payload["predicate_id"] = l5_v2.TT5L_PAIRED_EXACT_ANCHOR_PAIR_PREDICATE_ID
    payload["pair_group_id"] = "pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda"
    payload["paired_axis_status"] = "paired_cpu_cuda_harvested"
    payload["score_claim"] = False
    payload["promotion_eligible"] = False
    rows = payload["anchor_pair"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        row["promotion_eligible"] = False
        row["ready_for_exact_eval_dispatch"] = False
        if row["axis"] == "contest_cpu":
            row["dispatch_blockers"] = ["contest_cuda_pending_for_internal_promotion"]
            row["exact_cuda_evidence"] = False
        else:
            row["exact_cuda_evidence"] = True
        for key in ("artifact_path", "log_path", "inflated_outputs_manifest_path"):
            custody_path = tmp_path / str(row[key])
            custody_path.parent.mkdir(parents=True, exist_ok=True)
            if key == "inflated_outputs_manifest_path":
                custody_path.write_text(
                    json.dumps(
                        {"aggregate_sha256": row["inflated_raw_output_aggregate_sha256"]},
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                custody_path.write_text(f"{gate_id} {row['axis']} {key}\n", encoding="utf-8")

    anchor_path = tmp_path / l5_v2.TT5L_PAIRED_EXACT_ANCHOR_PAIR_ARTIFACT_PATH
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    anchor_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)
    gate = next(
        gate for gate in readiness["gates"] if gate["gate_id"] == gate_id
    )

    assert gate["evidence_valid"] is True
    assert gate["evidence"]["artifact_path"] == (
        l5_v2.TT5L_PAIRED_EXACT_ANCHOR_PAIR_ARTIFACT_PATH
    )
    assert (
        readiness["tt5l_campaign_readiness"]["anchor_pair_evidence_valid"] is True
    )
    assert "requires_l5_v2_empirical_anchor" not in readiness["blockers"]
    assert readiness["score_claim"] is False
    assert readiness["promotion_eligible"] is False


def test_l5_v2_dispatch_readiness_requires_exact_anchor_eval_custody(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "exact_anchor_or_diagnostic_pair"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    rows = payload["anchor_pair"]
    assert isinstance(rows, list)
    cuda_row = next(row for row in rows if row["axis"] == "contest_cuda")
    cuda_row.pop("n_samples", None)
    cuda_row["log_path"] = "experiments/results/time_traveler_l5_v2/missing.log"
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert (
        "l5_v2_gate_artifact_semantics_missing:"
        "exact_anchor_or_diagnostic_pair:anchor_pair:contest_cuda:"
        "exact_eval:n_samples_missing"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_missing:"
        "exact_anchor_or_diagnostic_pair:anchor_pair:contest_cuda:"
        "exact_eval:log_path_file_missing"
        in readiness["blockers"]
    )


def test_l5_v2_exact_anchor_requires_inflated_output_manifest_custody(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "exact_anchor_or_diagnostic_pair"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    rows = payload["anchor_pair"]
    assert isinstance(rows, list)
    cuda_row = next(row for row in rows if row["axis"] == "contest_cuda")
    cuda_row.pop("inflated_outputs_manifest_path", None)
    cuda_row.pop("inflated_raw_output_aggregate_sha256", None)
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert (
        "l5_v2_gate_artifact_semantics_missing:"
        "exact_anchor_or_diagnostic_pair:anchor_pair:contest_cuda:"
        "inflated_outputs_manifest_path"
        in readiness["blockers"]
    )
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "exact_anchor_or_diagnostic_pair:anchor_pair:contest_cuda:"
        "inflated_raw_output_aggregate_sha256"
        in readiness["blockers"]
    )


def test_l5_v2_exact_anchor_binds_raw_aggregate_to_manifest(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "exact_anchor_or_diagnostic_pair"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id, repo_root=tmp_path)
    rows = payload["anchor_pair"]
    assert isinstance(rows, list)
    cuda_row = next(row for row in rows if row["axis"] == "contest_cuda")
    manifest_path = tmp_path / str(cuda_row["inflated_outputs_manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"aggregate_sha256": _sha(99)}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert (
        "l5_v2_gate_artifact_semantics_invalid:"
        "exact_anchor_or_diagnostic_pair:anchor_pair:contest_cuda:"
        "inflated_outputs_manifest_path:aggregate_sha256_mismatch"
        in readiness["blockers"]
    )


def test_l5_v2_dispatch_readiness_verifies_artifact_files_and_hashes(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    first_gate_id = next(iter(evidence))

    evidence[first_gate_id]["artifact_sha256"] = _sha(99)
    mismatched = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)
    assert mismatched["ready_for_dispatch"] is False
    assert f"l5_v2_gate_artifact_sha256_mismatch:{first_gate_id}" in mismatched[
        "blockers"
    ]

    evidence = _valid_gate_evidence_payloads(tmp_path)
    evidence[first_gate_id]["artifact_path"] = "experiments/results/missing_l5v2.json"
    missing = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)
    assert missing["ready_for_dispatch"] is False
    assert f"l5_v2_gate_artifact_file_missing:{first_gate_id}" in missing["blockers"]

    evidence = _valid_gate_evidence_payloads(tmp_path)
    outside = tmp_path.parent / "outside_l5v2.json"
    outside.write_text("outside\n", encoding="utf-8")
    evidence[first_gate_id]["artifact_path"] = str(outside)
    evidence[first_gate_id]["artifact_sha256"] = _file_sha256(outside)
    outside_repo = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)
    assert outside_repo["ready_for_dispatch"] is False
    assert f"l5_v2_gate_artifact_path_outside_repo:{first_gate_id}" in outside_repo[
        "blockers"
    ]


def test_l5_v2_dispatch_readiness_verifies_gate_artifact_identity(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    first_gate_id = next(iter(evidence))
    artifact_path = tmp_path / str(evidence[first_gate_id]["artifact_path"])
    artifact_path.write_text(
        '{"gate_id":"wrong_gate","passed":true}\n',
        encoding="utf-8",
    )
    evidence[first_gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert readiness["ready_for_gate_probe_dispatch"] is False
    assert (
        f"l5_v2_gate_artifact_gate_id_mismatch:{first_gate_id}:wrong_gate"
        in readiness["blockers"]
    )


def test_l5_v2_dispatch_readiness_requires_artifact_predicate_identity(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    first_gate_id = next(iter(evidence))
    artifact_path = tmp_path / str(evidence[first_gate_id]["artifact_path"])
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload.pop("predicate_id", None)
    artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    evidence[first_gate_id]["artifact_sha256"] = _file_sha256(artifact_path)

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence, repo_root=tmp_path)

    assert readiness["all_gate_evidence_valid"] is False
    assert readiness["ready_for_gate_probe_dispatch"] is False
    assert (
        f"l5_v2_gate_artifact_predicate_id_missing:{first_gate_id}"
        in readiness["blockers"]
    )


def test_l5_v2_prediction_band_flows_into_composition_row() -> None:
    rows = {
        row.substrate_id: row
        for row in per_substrate_pareto_rows(matrix=build_composition_matrix())
    }
    row = rows[SUBJECT_ID]

    assert row.prediction_band is not None
    assert row.prediction_band["band_id"] == "time_traveler_l5_v2_delta_prior_20260516"
    assert row.prediction_band_verdict is not None
    assert row.prediction_band_verdict["valid_for_rank_reward"] is False
    assert "prediction_band_baseline_missing" in row.dispatch_blockers
    assert "prediction_band_empirical_anchor_missing" in row.dispatch_blockers
