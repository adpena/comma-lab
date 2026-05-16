# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.l5_v2_probe_disambiguator import (
    L5V2_CANDIDATES,
    L5V2_PROBE_SCHEMA,
    L5V2_PROBE_TOOL_PATH,
    L5V2ProbeObservation,
    build_probe_template,
    evaluate_l5_v2_probe,
    observation_from_mapping,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _axis_evidence(
    axis: str,
    *,
    repo_root: Path | None = None,
    score_delta: float = 0.0,
) -> dict[str, object]:
    archive_bytes = 1
    score = 25.0 * archive_bytes / 37_545_489
    log_path = f"experiments/results/l5_v2_probe/{axis}.log"
    if repo_root is not None:
        path = repo_root / log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{axis} exact-eval log\n", encoding="utf-8")
    return {
        "axis": axis,
        "archive_sha256": "a" * 64,
        "runtime_tree_sha256": "b" * 64,
        "score": score,
        "seg_dist": 0.0,
        "pose_dist": 0.0,
        "archive_bytes": archive_bytes,
        "n_samples": 600,
        "hardware": "gha-linux-x86_64" if axis == "contest_cpu" else "modal-t4",
        "inflate_device": "cpu" if axis == "contest_cpu" else "cuda",
        "eval_device": "cpu" if axis == "contest_cpu" else "cuda",
        "auth_eval_command": f"contest_auth_eval --axis {axis}",
        "log_path": log_path,
        "score_delta": score_delta,
    }


def _eligible(repo_root: Path, candidate_id: str, delta: float) -> L5V2ProbeObservation:
    artifact_path = repo_root / "experiments" / "results" / "l5_v2_probe" / f"{candidate_id}.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        f'{{"candidate_id":"{candidate_id}","delta":{delta}}}\n',
        encoding="utf-8",
    )
    return L5V2ProbeObservation(
        candidate_id=candidate_id,
        predicted_or_measured_delta=delta,
        evidence_grade="contest_cpu_cuda_paired_exact",
        exact_axes=("contest_cpu", "contest_cuda"),
        artifact_path=str(artifact_path.relative_to(repo_root)),
        artifact_sha256=_file_sha256(artifact_path),
        predicate_id=f"l5_v2_probe_{candidate_id}_predicate",
        predicate_passed=True,
        archive_sha256="a" * 64,
        runtime_tree_sha256="b" * 64,
        axis_evidence=tuple(
            _axis_evidence(axis, repo_root=repo_root, score_delta=delta)
            for axis in ("contest_cpu", "contest_cuda")
        ),
        sideinfo_consumed=True,
        byte_closed_archive=True,
    )


def test_l5_v2_probe_template_is_fail_closed_and_complete() -> None:
    template = build_probe_template()

    assert template["schema"] == L5V2_PROBE_SCHEMA
    assert template["score_claim"] is False
    assert template["promotion_eligible"] is False
    assert template["ready_for_exact_eval_dispatch"] is False
    assert [row["candidate_id"] for row in template["observations"]] == list(
        L5V2_CANDIDATES
    )
    for row in template["observations"]:
        assert row["artifact_path"] == ""
        assert row["artifact_sha256"] == ""
        assert row["predicate_id"] == ""
        assert row["predicate_passed"] is False
        assert [axis_row["axis"] for axis_row in row["axis_evidence"]] == [
            "contest_cpu",
            "contest_cuda",
        ]


def test_l5_v2_probe_fails_closed_without_observations() -> None:
    verdict = evaluate_l5_v2_probe(())

    assert verdict["architecture_lock_allowed"] is False
    assert verdict["selected_candidate_id"] is None
    assert "l5_v2_probe_observations_missing" in verdict["blockers"]
    assert "l5_v2_probe_candidate_coverage_incomplete" in verdict["blockers"]
    assert "l5_v2_probe_no_eligible_candidate" in verdict["blockers"]
    assert verdict["score_claim"] is False


def test_l5_v2_probe_selects_best_paired_exact_candidate(tmp_path: Path) -> None:
    verdict = evaluate_l5_v2_probe(
        (
            _eligible(tmp_path, "c1_world_model_foveation", -0.010),
            _eligible(tmp_path, "z5_predictive_coding_world_model", -0.020),
            _eligible(tmp_path, "time_traveler_l5_autonomy", -0.030),
        ),
        repo_root=tmp_path,
    )

    assert verdict["architecture_lock_allowed"] is True
    assert verdict["selected_candidate_id"] == "time_traveler_l5_autonomy"
    assert verdict["selected_delta"] == -0.030
    assert verdict["selected_delta_source"] == "paired_axis_score_delta"
    assert verdict["blockers"] == []


def test_l5_v2_probe_requires_eligible_evidence_for_every_candidate(
    tmp_path: Path,
) -> None:
    blocked_c1 = dataclasses.replace(
        _eligible(tmp_path, "c1_world_model_foveation", -0.010),
        axis_evidence=(),
    )
    blocked_z5 = dataclasses.replace(
        _eligible(tmp_path, "z5_predictive_coding_world_model", -0.020),
        predicate_passed=False,
    )

    verdict = evaluate_l5_v2_probe(
        (
            blocked_c1,
            blocked_z5,
            _eligible(tmp_path, "time_traveler_l5_autonomy", -0.030),
        ),
        repo_root=tmp_path,
    )

    assert verdict["architecture_lock_allowed"] is False
    assert verdict["selected_candidate_id"] == "time_traveler_l5_autonomy"
    assert (
        "l5_v2_probe_required_candidate_ineligible:c1_world_model_foveation"
        in verdict["blockers"]
    )
    assert (
        "l5_v2_probe_required_candidate_ineligible:z5_predictive_coding_world_model"
        in verdict["blockers"]
    )
    assert "l5_v2_probe_candidate_coverage_incomplete" not in verdict["blockers"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("predicate_passed", "false"),
        ("sideinfo_consumed", "yes"),
        ("byte_closed_archive", 1),
    ],
)
def test_l5_v2_probe_mapping_rejects_non_bool_authority_fields(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = dataclasses.asdict(
        _eligible(tmp_path, "time_traveler_l5_autonomy", -0.030)
    )
    payload[field] = value

    with pytest.raises(ValueError, match=f"{field} must be a literal JSON boolean"):
        observation_from_mapping(payload)


@pytest.mark.parametrize(
    ("field", "value", "expected_blocker"),
    [
        ("predicate_passed", "false", "l5_v2_probe_predicate_failed"),
        ("sideinfo_consumed", "yes", "l5_v2_probe_sideinfo_consumption_missing"),
        ("byte_closed_archive", 1, "l5_v2_probe_byte_closed_archive_missing"),
    ],
)
def test_l5_v2_probe_evaluation_rejects_non_bool_authority_fields(
    tmp_path: Path,
    field: str,
    value: object,
    expected_blocker: str,
) -> None:
    observation = dataclasses.replace(
        _eligible(tmp_path, "time_traveler_l5_autonomy", -0.030),
        **{field: value},
    )

    verdict = evaluate_l5_v2_probe((observation,), repo_root=tmp_path)
    row = verdict["evaluated_observations"][0]

    assert verdict["architecture_lock_allowed"] is False
    assert expected_blocker in row["blockers"]


def test_l5_v2_probe_blocks_proxy_or_unconsumed_observations(tmp_path: Path) -> None:
    proxy = dataclasses.replace(
        _eligible(tmp_path, "time_traveler_l5_autonomy", -0.050),
        evidence_grade="macos_cpu_advisory",
        exact_axes=("contest_cpu",),
        sideinfo_consumed=False,
    )

    verdict = evaluate_l5_v2_probe((proxy,), repo_root=tmp_path)
    row = verdict["evaluated_observations"][0]

    assert verdict["architecture_lock_allowed"] is False
    assert "l5_v2_probe_paired_exact_axes_missing" in row["blockers"]
    assert "l5_v2_probe_sideinfo_consumption_missing" in row["blockers"]
    assert "l5_v2_probe_contest_evidence_grade_missing" in row["blockers"]


def test_l5_v2_probe_blocks_string_only_paired_axes(tmp_path: Path) -> None:
    string_only = dataclasses.replace(
        _eligible(tmp_path, "time_traveler_l5_autonomy", -0.050),
        axis_evidence=(),
    )

    verdict = evaluate_l5_v2_probe((string_only,), repo_root=tmp_path)
    row = verdict["evaluated_observations"][0]

    assert verdict["architecture_lock_allowed"] is False
    assert "l5_v2_probe_axis_evidence_missing:contest_cpu" in row["blockers"]
    assert "l5_v2_probe_axis_evidence_missing:contest_cuda" in row["blockers"]


def test_l5_v2_probe_rejects_duplicate_axis_evidence_rows(tmp_path: Path) -> None:
    valid = _eligible(tmp_path, "time_traveler_l5_autonomy", -0.050)
    duplicate_cpu = dict(valid.axis_evidence[0])
    duplicate_cpu["score_delta"] = -0.500

    verdict = evaluate_l5_v2_probe(
        (
            dataclasses.replace(
                valid,
                axis_evidence=(*valid.axis_evidence, duplicate_cpu),
            ),
        ),
        repo_root=tmp_path,
    )
    row = verdict["evaluated_observations"][0]

    assert verdict["architecture_lock_allowed"] is False
    assert "l5_v2_probe_axis_evidence_duplicate:contest_cpu" in row["blockers"]


def test_l5_v2_probe_blocks_axis_evidence_without_formula_closure(
    tmp_path: Path,
) -> None:
    bad_axis_evidence = dict(
        _axis_evidence("contest_cuda", repo_root=tmp_path, score_delta=-0.050)
    )
    bad_axis_evidence["score"] = 0.123
    malformed = dataclasses.replace(
        _eligible(tmp_path, "time_traveler_l5_autonomy", -0.050),
        axis_evidence=(
            _axis_evidence("contest_cpu", repo_root=tmp_path, score_delta=-0.050),
            bad_axis_evidence,
        ),
    )

    verdict = evaluate_l5_v2_probe((malformed,), repo_root=tmp_path)
    row = verdict["evaluated_observations"][0]

    assert verdict["architecture_lock_allowed"] is False
    assert "l5_v2_probe_axis_score_formula_mismatch:contest_cuda" in row["blockers"]


def test_l5_v2_probe_blocks_macos_cpu_axis_evidence(tmp_path: Path) -> None:
    bad_cpu = dataclasses.replace(
        _eligible(tmp_path, "time_traveler_l5_autonomy", -0.050),
        axis_evidence=(
            dict(
                _axis_evidence(
                    "contest_cpu",
                    repo_root=tmp_path,
                    score_delta=-0.050,
                ),
                hardware="Apple Silicon macOS advisory host",
                inflate_device="macos-apple-m2-cpu",
                eval_device="cpu-mps",
                auth_eval_command="contest_auth_eval --axis contest_cpu --device mps",
            ),
            _axis_evidence("contest_cuda", repo_root=tmp_path, score_delta=-0.050),
        ),
    )

    verdict = evaluate_l5_v2_probe((bad_cpu,), repo_root=tmp_path)
    row = verdict["evaluated_observations"][0]

    assert verdict["architecture_lock_allowed"] is False
    assert "l5_v2_probe_axis_hardware_not_contest_cpu:contest_cpu" in row["blockers"]
    assert (
        "l5_v2_probe_axis_inflate_device_not_contest_cpu:contest_cpu"
        in row["blockers"]
    )
    assert (
        "l5_v2_probe_axis_eval_device_not_contest_cpu:contest_cpu"
        in row["blockers"]
    )
    assert (
        "l5_v2_probe_axis_auth_eval_command_not_contest_cpu:contest_cpu"
        in row["blockers"]
    )


def test_l5_v2_probe_rejects_delta_not_derived_from_paired_axis_scores(
    tmp_path: Path,
) -> None:
    freeform = _eligible(tmp_path, "time_traveler_l5_autonomy", -0.050)
    axis_evidence = [
        dict(row, score_delta=-0.010)
        for row in freeform.axis_evidence
    ]

    verdict = evaluate_l5_v2_probe(
        (dataclasses.replace(freeform, axis_evidence=tuple(axis_evidence)),),
        repo_root=tmp_path,
    )
    row = verdict["evaluated_observations"][0]

    assert verdict["architecture_lock_allowed"] is False
    assert row["paired_score_delta"] == -0.010
    assert "l5_v2_probe_delta_not_bound_to_axis_evidence" in row["blockers"]


def test_l5_v2_probe_selection_uses_paired_axis_score_delta(
    tmp_path: Path,
) -> None:
    # CPU/CUDA deltas may diverge slightly; architecture lock uses the
    # conservative paired value, not an unverified handwritten field.
    c1 = _eligible(tmp_path, "c1_world_model_foveation", -0.030)
    z5 = _eligible(tmp_path, "z5_predictive_coding_world_model", -0.020)
    tt5l = _eligible(tmp_path, "time_traveler_l5_autonomy", -0.010)
    c1_rows = [
        dict(row, score_delta=-0.040 if row["axis"] == "contest_cpu" else -0.030)
        for row in c1.axis_evidence
    ]

    verdict = evaluate_l5_v2_probe(
        (
            dataclasses.replace(c1, axis_evidence=tuple(c1_rows)),
            z5,
            tt5l,
        ),
        repo_root=tmp_path,
    )

    assert verdict["architecture_lock_allowed"] is True
    assert verdict["selected_candidate_id"] == "c1_world_model_foveation"
    assert verdict["selected_delta"] == -0.030


def test_l5_v2_probe_requires_durable_axis_log_files(tmp_path: Path) -> None:
    valid = _eligible(tmp_path, "time_traveler_l5_autonomy", -0.050)
    axis_evidence = [dict(row) for row in valid.axis_evidence]
    cuda_row = next(row for row in axis_evidence if row["axis"] == "contest_cuda")
    cuda_row["log_path"] = "experiments/results/l5_v2_probe/missing_cuda.log"

    verdict = evaluate_l5_v2_probe(
        (dataclasses.replace(valid, axis_evidence=tuple(axis_evidence)),),
        repo_root=tmp_path,
    )
    row = verdict["evaluated_observations"][0]

    assert verdict["architecture_lock_allowed"] is False
    assert "l5_v2_probe_axis_log_path_file_missing:contest_cuda" in row["blockers"]


def test_l5_v2_probe_rejects_transient_or_outside_axis_logs(tmp_path: Path) -> None:
    valid = _eligible(tmp_path, "time_traveler_l5_autonomy", -0.050)

    transient_rows = [dict(row) for row in valid.axis_evidence]
    next(row for row in transient_rows if row["axis"] == "contest_cuda")[
        "log_path"
    ] = "/tmp/l5v2_cuda.log"
    transient = evaluate_l5_v2_probe(
        (dataclasses.replace(valid, axis_evidence=tuple(transient_rows)),),
        repo_root=tmp_path,
    )
    assert "l5_v2_probe_axis_log_path_transient:contest_cuda" in transient[
        "evaluated_observations"
    ][0]["blockers"]

    outside_log = tmp_path.parent / "outside_l5v2_cuda.log"
    outside_log.write_text("outside\n", encoding="utf-8")
    outside_rows = [dict(row) for row in valid.axis_evidence]
    next(row for row in outside_rows if row["axis"] == "contest_cuda")[
        "log_path"
    ] = str(outside_log)
    outside = evaluate_l5_v2_probe(
        (dataclasses.replace(valid, axis_evidence=tuple(outside_rows)),),
        repo_root=tmp_path,
    )
    assert "l5_v2_probe_axis_log_path_outside_repo:contest_cuda" in outside[
        "evaluated_observations"
    ][0]["blockers"]


def test_l5_v2_probe_blocks_observations_without_artifact_predicate_custody() -> None:
    missing_custody = L5V2ProbeObservation(
        candidate_id="time_traveler_l5_autonomy",
        predicted_or_measured_delta=-0.050,
        evidence_grade="contest_cpu_cuda_paired_exact",
        exact_axes=("contest_cpu", "contest_cuda"),
        archive_sha256="a" * 64,
        runtime_tree_sha256="b" * 64,
        sideinfo_consumed=True,
        byte_closed_archive=True,
    )

    verdict = evaluate_l5_v2_probe((missing_custody,))
    row = verdict["evaluated_observations"][0]

    assert verdict["architecture_lock_allowed"] is False
    assert "l5_v2_probe_artifact_path_missing" in row["blockers"]
    assert "l5_v2_probe_artifact_sha_invalid" in row["blockers"]
    assert "l5_v2_probe_predicate_id_missing" in row["blockers"]
    assert "l5_v2_probe_predicate_failed" in row["blockers"]


def test_l5_v2_probe_rejects_transient_artifact_and_bad_hashes() -> None:
    bad = L5V2ProbeObservation(
        candidate_id="time_traveler_l5_autonomy",
        predicted_or_measured_delta=-0.050,
        evidence_grade="contest_cpu_cuda_paired_exact",
        exact_axes=("contest_cpu", "contest_cuda"),
        artifact_path="/tmp/l5v2_probe.json",
        artifact_sha256="c" * 64,
        predicate_id="tt5l_probe_predicate",
        predicate_passed=True,
        archive_sha256="not-sha",
        runtime_tree_sha256="also-not-sha",
        sideinfo_consumed=True,
        byte_closed_archive=True,
    )

    verdict = evaluate_l5_v2_probe((bad,))
    row = verdict["evaluated_observations"][0]

    assert verdict["architecture_lock_allowed"] is False
    assert "l5_v2_probe_artifact_path_transient" in row["blockers"]
    assert "l5_v2_probe_archive_sha_invalid" in row["blockers"]
    assert "l5_v2_probe_runtime_tree_sha_invalid" in row["blockers"]


def test_l5_v2_probe_verifies_artifact_files_and_hashes(tmp_path: Path) -> None:
    valid = _eligible(tmp_path, "time_traveler_l5_autonomy", -0.050)

    missing = dataclasses.replace(valid, artifact_path="experiments/results/missing.json")
    missing_verdict = evaluate_l5_v2_probe((missing,), repo_root=tmp_path)
    assert "l5_v2_probe_artifact_file_missing" in missing_verdict[
        "evaluated_observations"
    ][0]["blockers"]

    mismatched = dataclasses.replace(valid, artifact_sha256="d" * 64)
    mismatched_verdict = evaluate_l5_v2_probe((mismatched,), repo_root=tmp_path)
    assert "l5_v2_probe_artifact_sha_mismatch" in mismatched_verdict[
        "evaluated_observations"
    ][0]["blockers"]

    outside = tmp_path.parent / "outside_probe.json"
    outside.write_text("outside\n", encoding="utf-8")
    outside_repo = dataclasses.replace(
        valid,
        artifact_path=str(outside),
        artifact_sha256=_file_sha256(outside),
    )
    outside_verdict = evaluate_l5_v2_probe((outside_repo,), repo_root=tmp_path)
    assert "l5_v2_probe_artifact_path_outside_repo" in outside_verdict[
        "evaluated_observations"
    ][0]["blockers"]


def test_l5_v2_probe_cli_emits_template() -> None:
    result = subprocess.run(
        [sys.executable, L5V2_PROBE_TOOL_PATH, "--emit-template"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["schema"] == L5V2_PROBE_SCHEMA
    assert [row["candidate_id"] for row in payload["observations"]] == list(
        L5V2_CANDIDATES
    )
