# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tac.optimization.l5_staircase_v2 import (
    PREDICTED_DELTA_BAND,
    SUBJECT_ID,
    L5V2GateEvidence,
    l5_v2_dispatch_readiness,
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
)
from tac.optimization.substrate_composition_matrix import (
    build_composition_matrix,
    per_substrate_pareto_rows,
)


def _sha(seed: int) -> str:
    return f"{seed:064x}"[-64:]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _axis_rows(*, anchor_type: str | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for axis in ("contest_cpu", "contest_cuda"):
        archive_bytes = 1
        row: dict[str, object] = {
            "axis": axis,
            "archive_sha256": _sha(11),
            "runtime_tree_sha256": _sha(12),
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


def _gate_artifact_payload(gate_id: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "gate_id": gate_id,
        "predicate_id": f"l5_v2_{gate_id}_predicate",
        "passed": True,
    }
    if gate_id == "byte_closed_temporal_sideinfo_consumption":
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
        }
    elif gate_id == "c1_z5_tt5l_probe_disambiguator":
        payload["probe_disambiguator"] = {
            "schema": L5V2_PROBE_SCHEMA,
            "tool_path": L5V2_PROBE_TOOL_PATH,
            "candidate_ids": list(L5V2_CANDIDATES),
            "paired_exact_axes_required": True,
            "verdict": {
                "schema": L5V2_PROBE_SCHEMA,
                "tool": L5V2_PROBE_TOOL_PATH,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "architecture_lock_allowed": True,
                "selected_candidate_id": "time_traveler_l5_autonomy",
                "required_candidates": list(L5V2_CANDIDATES),
                "required_exact_axes": ["contest_cpu", "contest_cuda"],
                "evaluated_observations": [
                    {
                        "candidate_id": candidate_id,
                        "eligible_for_architecture_lock": True,
                        "exact_axes": ["contest_cpu", "contest_cuda"],
                        "blockers": [],
                    }
                    for candidate_id in L5V2_CANDIDATES
                ],
                "blockers": [],
            },
        }
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
        payload = _gate_artifact_payload(gate.gate_id)
        if gate.gate_id == "byte_closed_temporal_sideinfo_consumption":
            proof = payload.get("byte_mutation_proof")
            assert isinstance(proof, dict)
            manifest_path = repo_root / str(proof["inflated_outputs_manifest_path"])
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text('{"raw_output_aggregate_sha256":"ok"}\n', encoding="utf-8")
        rows = payload.get("anchor_pair")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for key in ("artifact_path", "log_path", "inflated_outputs_manifest_path"):
                    custody_path = repo_root / str(row[key])
                    custody_path.parent.mkdir(parents=True, exist_ok=True)
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
        "lu_dvc_2019",
        "rissanen_mdl_1978",
        "mackay_itila_2003",
        "tishby_information_bottleneck_1999",
        "tishby_zaslavsky_2015",
        "balle_hyperprior_2018",
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
    assert verdict["valid_for_dispatch_planning"] is True
    assert verdict["valid_for_rank_reward"] is False
    assert "prediction_band_baseline_missing" in verdict["blockers"]
    assert "prediction_band_empirical_anchor_missing" in verdict["blockers"]
    assert "prediction_band_research_basis_missing" not in verdict["blockers"]


def test_l5_v2_dispatch_readiness_requires_artifact_evidence_not_booleans() -> None:
    blocked = l5_v2_dispatch_readiness()
    all_gate_ids = {gate.gate_id for gate in l5_v2_required_gates()}
    boolean_only = l5_v2_dispatch_readiness(dict.fromkeys(all_gate_ids, True))

    assert blocked["ready_for_dispatch"] is False
    assert blocked["all_gate_claims_satisfied"] is False
    assert blocked["all_gate_evidence_valid"] is False
    assert blocked["promotion_eligible"] is False
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
    assert boolean_only["score_claim"] is False
    assert all(gate["claimed_satisfied"] is True for gate in boolean_only["gates"])
    assert all(
        gate["claimed_satisfied_without_artifact"] is True
        for gate in boolean_only["gates"]
    )
    assert all(gate["status"] == "required" for gate in boolean_only["gates"])
    assert all(gate["evidence_valid"] is False for gate in boolean_only["gates"])


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
    payload = _gate_artifact_payload(gate_id)
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
    payload = _gate_artifact_payload(gate_id)
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
