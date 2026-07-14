from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

import tools.probe_whole_teacher_distilled_student as probe
from tac.witness_dsl.whole_teacher_distilled_student_policy import (
    AdmissionDecision,
    AdmissionState,
    EvidenceTier,
    StudentSize,
    WholeTeacherDistilledStudentPolicy,
)


def _fit_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=tmp_path,
        receipt_dir=tmp_path / "receipts",
        storage_plan=tmp_path / "storage.json",
        cache_manifest=tmp_path / "manifest.json",
        bundle_root=None,
        student_size="small",
        anchor_cadence=64,
        exact_costate_reuse_kmax=2,
        seed=455,
        fit_epochs=40,
        resume_from=None,
        mode="fit-measure",
    )


def test_fit_policy_mismatch_is_refused_before_measurement_admission(tmp_path: Path) -> None:
    sha = "a" * 64
    args = _fit_args(tmp_path)
    preflight = {
        "cache_preflight": {
            "custody": {"sha256": sha},
            "validated": {
                "manifest_sha256": sha,
                "teacher_source_custody": {
                    "custody_sha256": sha,
                    "helmert_basis_sha256": sha,
                    "post_r_input_surface_sha256": sha,
                },
            },
        }
    }
    result = {
        "schema": probe.FIT_RESULT_SCHEMA,
        "n_pairs": 600,
        "train_pairs": 480,
        "heldout_pairs": 120,
        "teacher_calls": 0,
        "backend": "mlx",
        "numerical_reference": "numpy_fp32",
        "measurement_axis": probe.AXIS,
        "student_size": "small",
        "measured_tier": "training_gradient",
        "student_anchor_cadence": None,
        "fit_epochs": 40,
        "fit_steps": 1,
        "authority": {
            "research_only": True,
            "means_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "teacher_regeneration": False,
            "synthetic_fallback": False,
            "cpu_fallback": False,
            "full_input_vjp_is_decisive": True,
            "boundary_input_vjp_is_diagnostic_only": True,
            "primary_teacher_fidelity_numerical_authority": "numpy_fp32",
            "mlx_outputs_used_for_primary_teacher_gate": False,
        },
        "cache_manifest_sha256": sha,
        "cache_manifest_file_sha256": sha,
        "teacher_source_custody_sha256": sha,
        "source_custody_sha256": sha,
        "quotient_basis_sha256": sha,
        "post_r_input_surface_sha256": sha,
        "fit_policy": {"deliberately": "wrong"},
        "hashes": {"policy_sha256": sha},
    }

    with pytest.raises(probe.ProbeError, match="optimizer/objective policy drifted"):
        probe._validate_fit_measurement_shape(args, preflight, result)


def test_partial_failure_never_fabricates_zero_progress(tmp_path: Path) -> None:
    args = _fit_args(tmp_path)

    empty = probe._partial_fit_observation(args)
    assert empty["n_pairs"] == "UNKNOWN_AFTER_PARTIAL_EXECUTION"
    assert empty["fit_steps"] == "UNKNOWN_AFTER_PARTIAL_EXECUTION"
    assert empty["teacher_calls"] == "UNKNOWN_AFTER_PARTIAL_EXECUTION"

    checkpoint = tmp_path / "stage_02_fit_epoch_step_000007.json"
    checkpoint.write_text(
        json.dumps(
            {
                "metadata": {
                    "optimizer_step": 7,
                    "stage": "fit_epoch",
                    "teacher_calls": 0,
                }
            }
        )
    )
    observed = probe._partial_fit_observation(args)
    assert observed["fit_steps"] == 7
    assert observed["teacher_calls"] == 0
    assert observed["n_pairs"] == "UNKNOWN_AFTER_PARTIAL_EXECUTION"
    assert observed["last_durable_stage"] == "fit_epoch"


@pytest.mark.parametrize(
    ("state", "admitted", "expected_disposition"),
    [
        (
            AdmissionState.BLOCKED_VJP_FIDELITY,
            False,
            "MEASUREMENT_COMPLETE_SCOPED_NO_GO",
        ),
        (
            AdmissionState.GO_READY_NOT_FIRED,
            True,
            "ADMISSION_PREPARED_NOT_FIRED",
        ),
    ],
)
def test_fit_completion_is_a_typed_admission_decision_not_generic_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: AdmissionState,
    admitted: bool,
    expected_disposition: str,
) -> None:
    args = _fit_args(tmp_path)
    contract = {"payload": {}, "sha256": "a" * 64}
    policy = WholeTeacherDistilledStudentPolicy(
        selected_size=StudentSize.SMALL,
        student_anchor_cadence=64,
        exact_costate_reuse_kmax=2,
    )
    decision = AdmissionDecision(
        state=state,
        tier=EvidenceTier.TRAINING_GRADIENT,
        admitted=admitted,
        reasons=("test-only typed disposition",),
    )

    monkeypatch.setattr(probe, "_run_contract", lambda _args, _preflight: contract)
    monkeypatch.setattr(
        probe,
        "_core_fit_entrypoint",
        lambda: (lambda **_kwargs: {"fit_steps": 12, "teacher_calls": 0}),
    )
    monkeypatch.setattr(
        probe,
        "_validate_fit_measurement_shape",
        lambda _args, _preflight, _result: {"hashes": {}},
    )
    monkeypatch.setattr(
        probe,
        "_materialize_best_parameter_blob",
        lambda _args, _contract, _result, _hashes: {"sha256": "b" * 64},
    )
    calls: list[dict[str, Any]] = []

    def fake_admission(
        _args: argparse.Namespace,
        _preflight: dict[str, Any],
        _result: dict[str, Any],
        _surfaces: dict[str, Any],
        _parameter_receipt: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any]]:
        calls.append({"called": True})
        return (
            policy,
            decision,
            {
                "evidence": {"test_only": True},
                "evidence_sha256": "c" * 64,
                "derived_economics": {"strict_pays": admitted},
                "deterministic_repeat": {"test_only": True},
                "teacher_timing_custody": {"test_only": True},
            },
        )

    monkeypatch.setattr(probe, "_build_admission_evidence", fake_admission)

    @contextmanager
    def fake_scratch(_output_dir: Path, *, run_contract_sha256: str):
        assert run_contract_sha256 == contract["sha256"]
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        yield scratch
        (tmp_path / "cleanup_manifest.json").write_text("{}")

    monkeypatch.setattr(probe, "_success_only_scratch", fake_scratch)

    result = probe._fit_measure(
        args,
        {"blockers": [], "storage_preflight": {}, "cache_preflight": {}},
    )

    assert calls == [{"called": True}]
    assert result["disposition"] == expected_disposition
    assert result["admission"]["admission_prepared"] is admitted
    assert result["admission"]["typed_admission_decision"]["state"] == state.value
    assert not (tmp_path / "stage_04_complete.json").exists()
    assert (tmp_path / "stage_04_admission_decision.json").is_file()


def _repeat_stream(*, scope: str, verification_field: str) -> dict[str, Any]:
    sha = "a" * 64
    return {
        "scope": scope,
        "pair_count": 600,
        "timed": False,
        "charged_student_timing_includes_repeat": False,
        "first_forward_sha256": sha,
        "second_forward_sha256": sha,
        "forward_equal": True,
        "first_input_vjp_sha256": sha,
        "second_input_vjp_sha256": sha,
        "input_vjp_equal": True,
        "first_combined_sha256": sha,
        "second_combined_sha256": sha,
        "combined_equal": True,
        verification_field: True,
    }


def _valid_deterministic_repeat_result() -> dict[str, Any]:
    mlx_advisory = _repeat_stream(
        scope="full_ordered_n600_mlx_advisory_forward_and_input_vjp_stream",
        verification_field="advisory_verified",
    )
    authority = _repeat_stream(
        scope="full_ordered_n600_numpy_fp32_authority_forward_and_input_vjp_stream",
        verification_field="authority_verified",
    )
    authority.update(
        {
            "numerical_authority": "numpy_fp32",
            "mlx_advisory": mlx_advisory,
            "all_required_streams_equal": True,
        }
    )
    return {"deterministic_repeat": authority, "deterministic_repeat_verified": True}


def test_deterministic_repeat_requires_uncharged_numpy_authority_and_mlx_advisory_streams() -> None:
    result = _valid_deterministic_repeat_result()
    _repeat, verified = probe._validate_deterministic_repeat(result)
    assert verified is True

    advisory = result["deterministic_repeat"]["mlx_advisory"]
    advisory["second_forward_sha256"] = "b" * 64
    advisory["forward_equal"] = False
    advisory["advisory_verified"] = False
    result["deterministic_repeat"]["all_required_streams_equal"] = False
    result["deterministic_repeat_verified"] = False
    _repeat, verified = probe._validate_deterministic_repeat(result)
    assert verified is False

    result = _valid_deterministic_repeat_result()
    result["deterministic_repeat"]["mlx_advisory"][
        "charged_student_timing_includes_repeat"
    ] = True
    with pytest.raises(probe.ProbeError, match="leaked into charged student timing"):
        probe._validate_deterministic_repeat(result)


@pytest.mark.parametrize(
    "metric_name",
    (
        "NumPy-primary forward",
        "NumPy-primary decisive VJP",
        "boundary VJP diagnostic",
        "MLX/NumPy forward parity",
        "MLX/NumPy VJP parity",
    ),
)
def test_nested_metric_assignment_id_must_match_outer_cache_assignment(metric_name: str) -> None:
    with pytest.raises(probe.ProbeError, match="assignment_id drifted from outer/cache custody"):
        probe._validate_nested_metric_assignment_ids(
            "cache-assignment-17",
            {metric_name: {"assignment_id": "swapped-assignment-18"}},
        )
