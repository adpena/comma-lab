# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

import tac.optimization.l5_staircase_v2 as l5_v2
from tac.optimization.l5_staircase_v2 import (
    L5_V2_PACKETIR_SECTION_ENTROPY_EVIDENCE_SCHEMA,
    L5_V2_PACKETIR_SECTION_ENTROPY_MATRIX_ARTIFACT_PATH,
    L5_V2_PACKETIR_SECTION_ENTROPY_MATRIX_ARTIFACT_SHA256,
    L5_V2_PACKETIR_STACK_EVIDENCE_SCHEMA,
    L5_V2_PR106_STACK_CELL_CANDIDATES_SCHEMA,
    PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_SHA256,
    PREDICTED_DELTA_BAND,
    SUBJECT_ID,
    L5V2GateEvidence,
    l5_v2_dispatch_readiness,
    l5_v2_packetir_section_entropy_evidence_payload,
    l5_v2_packetir_stack_evidence_payload,
    l5_v2_pr106_stack_cell_candidates,
    l5_v2_prediction_band_payload,
    l5_v2_prediction_band_verdict,
    l5_v2_required_gates,
    l5_v2_research_basis_ids,
    l5_v2_staircase_steps,
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


def _sha(seed: int) -> str:
    return f"{seed:064x}"[-64:]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
                row["hardware"] = "cpu" if axis == "contest_cpu" else "modal-t4"
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
            "file_list_sha256": _sha(28),
            "baseline_raw_output_aggregate_sha256": _sha(29),
            "mutated_raw_output_aggregate_sha256": _sha(30),
        }
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


def _valid_gate_evidence_payloads(repo_root: Path) -> dict[str, dict[str, object]]:
    return {
        gate_id: gate_evidence.__dict__.copy()
        for gate_id, gate_evidence in _valid_gate_evidence(repo_root).items()
    }


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


def test_l5_v2_staircase_steps_are_ordered_and_fail_closed() -> None:
    steps = l5_v2_staircase_steps()
    gate_ids = {gate.gate_id for gate in l5_v2_required_gates()}

    assert [step.step_id for step in steps] == [
        "l5v2_00_source_and_alias_custody",
        "l5v2_01_sideinfo_consumption_proof",
        "l5v2_02_probe_disambiguator",
        "l5v2_03_paired_axis_anchor",
        "l5v2_04_stack_of_stacks_candidate",
    ]
    for step in steps:
        assert step.research_basis_ids
        assert step.dispatch_allowed is False
        assert step.promotion_eligible is False
        assert set(step.required_gate_ids) <= gate_ids

    probe_step = next(step for step in steps if step.step_id == "l5v2_02_probe_disambiguator")
    assert probe_step.deliverable_surface == L5V2_PROBE_TOOL_PATH
    assert Path(L5V2_PROBE_TOOL_PATH).is_file()


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


def test_l5_v2_dispatch_readiness_requires_artifact_evidence_not_booleans() -> None:
    blocked = l5_v2_dispatch_readiness()
    all_gate_ids = {gate.gate_id for gate in l5_v2_required_gates()}
    boolean_only = l5_v2_dispatch_readiness(dict.fromkeys(all_gate_ids, True))

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


def test_l5_v2_dispatch_readiness_accepts_valid_gate_evidence(tmp_path: Path) -> None:
    ready = l5_v2_dispatch_readiness(
        gate_evidence=_valid_gate_evidence(tmp_path),
        repo_root=tmp_path,
    )

    assert ready["all_gate_claims_satisfied"] is True
    assert ready["all_gate_evidence_valid"] is True
    assert ready["ready_for_gate_probe_dispatch"] is True
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


def test_l5_v2_valid_gates_do_not_unlock_blocked_prediction_band(
    tmp_path: Path,
) -> None:
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
    payload = _gate_artifact_payload(gate_id)
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
    payload = _gate_artifact_payload(gate_id)
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
    payload = _gate_artifact_payload(gate_id)
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
    payload = _gate_artifact_payload(gate_id)
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
    payload = _gate_artifact_payload(gate_id)
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
    payload = _gate_artifact_payload(gate_id)
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
    payload = _gate_artifact_payload(gate_id)
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
    payload = _gate_artifact_payload(gate_id)
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


def test_l5_v2_sideinfo_consumption_rejects_toy_manifest_scope(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "byte_closed_temporal_sideinfo_consumption"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id)
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
    payload = _gate_artifact_payload(gate_id)
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


def test_l5_v2_sideinfo_consumption_requires_archive_runtime_section_identity(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "byte_closed_temporal_sideinfo_consumption"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id)
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
    payload = _gate_artifact_payload(gate_id)
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
    payload = _gate_artifact_payload(gate_id)
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


def test_l5_v2_dispatch_readiness_requires_exact_anchor_eval_custody(
    tmp_path: Path,
) -> None:
    evidence = _valid_gate_evidence_payloads(tmp_path)
    gate_id = "exact_anchor_or_diagnostic_pair"
    artifact_path = tmp_path / str(evidence[gate_id]["artifact_path"])
    payload = _gate_artifact_payload(gate_id)
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
    payload = _gate_artifact_payload(gate_id)
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
    payload = _gate_artifact_payload(gate_id)
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
