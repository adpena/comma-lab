from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

import tac.optimization.direct_description_minimizer as ddm
from tac.canonical_frontier_pointer import POINTER_SCHEMA_VERSION
from tac.optimization.direct_description_minimizer import (
    ChargedFreePartitionRowV1,
    CountedDescriptionStreamV1,
    DescriptionStepMetricTelemetryV1,
    DirectDescriptionError,
    DirectDescriptionOpsGrammarMinimizerV1,
    DirectDescriptionStageCheckpointV1,
    DirectDescriptionZV1,
    MeasurementRungRowV1,
    ToleranceAllocationNodeV1,
    build_direct_description_arg_parser,
    build_direct_description_owner,
    build_launch_readiness,
    compile_direct_description_archive,
    derive_ceil_minus_one_caps,
    load_stage_checkpoint,
    numpy_reference_rank,
    optimizer_admission_status,
    parse_direct_description_archive,
    prove_baseline_reexpression,
    rfc8785_canonicalize,
    seal_failure_receipt,
    storage_preflight,
    validate_receiver_rate_custody,
    verify_allocation_tree,
    verify_charged_free_partition,
    verify_completion_certificate,
    verify_measurement_ladder,
)
from tac.optimization.s4_archive_composer import (
    SectionBytes,
    build_payload_manifest,
    canonical_json_bytes,
)
from tac.witness_dsl.dynamic_frontier_target import load_dynamic_frontier_target

H = "1" * 64
GIT_SHA = "a" * 40
CLASSES = ("MyCar", "Undrivable", "Road", "Lane", "Movable")
AXES = (
    "class",
    "canonical_stratum",
    "temporal_segment_or_event_window",
    "class_pair_boundary",
    "frequency_or_scale",
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_jcs(path: Path, value: dict) -> str:
    payload = rfc8785_canonicalize(value) + b"\n"
    path.write_bytes(payload)
    return _sha(payload)


def _synthetic_z(*, empty_pose: bool = False) -> DirectDescriptionZV1:
    bodies = (
        SectionBytes("seed.ppcs", b"xi", "raw", 2),
        SectionBytes("base.pbase3", b"ground", "mixed", 12),
        SectionBytes(
            "causal.pcr3",
            b"" if empty_pose else b"pose",
            "raw",
            0 if empty_pose else 4,
        ),
        SectionBytes("events.pce3", b"event", "lzma1_raw_1MiB", 9),
        SectionBytes("components.pcomp3", b"exception", "zlib9", 18),
    )
    manifest_payload = canonical_json_bytes(build_payload_manifest(bodies, source_commit="0" * 40))
    manifest = SectionBytes(
        "manifest.json",
        manifest_payload,
        "raw",
        len(manifest_payload),
    )
    return DirectDescriptionZV1.from_s4_sections((manifest, *bodies))


def _archive_file(tmp_path: Path) -> tuple[Path, ddm.DirectArchiveBuildResult]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    result = compile_direct_description_archive(_synthetic_z())
    path = tmp_path / "archive.zip"
    path.write_bytes(result.archive)
    return path, result


def _target_receipt(tmp_path: Path, *, planning_only: bool = False) -> tuple[Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_bytes(b"# frozen evaluator test fixture\n")
    runtime = tmp_path / "scorer_runtime.bin"
    runtime.write_bytes(b"frozen-runtime-test-fixture")
    solved_d_seg = "0.00015195999999999999"
    solved_d_pose = "0.00010183999999999999"
    derivation = {
        "schema": "direct_description_solved_target_derivation.v1",
        "pairs": 600,
        "through_R": True,
        "exact_evaluator_called": True,
        "evaluator_sha256": _sha(evaluator.read_bytes()),
        "scorer_runtime_sha256": _sha(runtime.read_bytes()),
        "solved_d_seg": solved_d_seg,
        "solved_d_pose": solved_d_pose,
    }
    derivation_path = tmp_path / "target_derivation.json"
    derivation_sha = _write_jcs(derivation_path, derivation)
    value = {
        "schema": "direct_description_full_precision_target.v1",
        "authority": "official_frozen_evaluator_solved_target",
        "hardware_axis": "[contest-CPU]",
        "launch_config_admissible": not planning_only,
        "planning_only": planning_only,
        "pairs": 600,
        "solved_d_seg": solved_d_seg,
        "solved_d_pose": solved_d_pose,
        "evaluator_path": str(evaluator),
        "evaluator_sha256": _sha(evaluator.read_bytes()),
        "scorer_runtime_path": str(runtime),
        "scorer_runtime_sha256": _sha(runtime.read_bytes()),
        "target_derivation_receipt_path": str(derivation_path),
        "target_derivation_receipt_sha256": derivation_sha,
    }
    path = tmp_path / ("target_planning.json" if planning_only else "target.json")
    return path, _write_jcs(path, value)


def _dynamic_frontier_snapshot(repo: Path, *, score: float = 0.25):
    now = datetime.now(UTC).isoformat()
    entry = {
        "score": score,
        "rank": 1,
        "name": "synthetic-public-row",
        "pr_number": 9001,
        "pr_url": "https://invalid.example/synthetic",
    }
    payload = {
        "schema_version": POINTER_SCHEMA_VERSION,
        "our_local_frontier_contest_cpu": None,
        "our_local_frontier_contest_cuda": None,
        "submitted_pr_number_for_current_frontier": None,
        "upstream_leaderboard_snapshot": {
            "best_entry": dict(entry),
            "entries": [dict(entry)],
        },
        "upstream_leaderboard_snapshot_at_utc": now,
        "last_refreshed_utc": now,
        "auto_update_on_dispatch_completion": True,
        "pointer_refresh_command": "synthetic-fixture-do-not-run",
        "refresh_provenance": {"fixture": True},
        "effective_frontier": {
            "score": 0.001,
            "source": "forged-cache-must-not-steer",
            "axis": "forged",
        },
    }
    pointer = repo / ".omx/state/canonical_frontier_pointer.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(json.dumps(payload), encoding="utf-8")
    return load_dynamic_frontier_target(repo_root=repo, now_utc_iso=now), now


def _completion_certificate(tmp_path: Path, grammar_sha: str = H) -> tuple[dict, str, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    preregistration = {
        "mode": "stationarity_exhaustion",
        "grammar_manifest_sha256": grammar_sha,
        "candidate_interpretations": ["corrected_xi"],
        "pools": ["Road.boundary_codim1"],
        "pairs": 600,
        "stopping_rule": "all_pool_kkt_or_gate_excluded",
        "kkt_threshold_numerator": 0,
        "kkt_threshold_denominator": 1,
        "max_steps_per_pool": 100,
        "max_restarts": 2,
        "wallclock_budget_seconds": 3600,
        "hard_oracle_admission_rule": (
            "same_pool_realized_score_gain_gt_25_over_37545489_per_byte"
        ),
    }
    preregistration_sha = _sha(rfc8785_canonicalize(preregistration))
    gradient_terms = [
        {"coordinate": "Road.boundary_codim1.x", "numerator": 1, "denominator": 2},
        {"coordinate": "Road.boundary_codim1.x", "numerator": -1, "denominator": 2},
    ]
    terminal = {
        "schema": "direct_description_terminal_pool_evidence.v1",
        "grammar_manifest_sha256": grammar_sha,
        "preregistration_sha256": preregistration_sha,
        "pool": "Road.boundary_codim1",
        "interpretation": "corrected_xi",
        "pairs": 600,
        "status": "searched",
        "gradient_terms": gradient_terms,
    }
    terminal_path = tmp_path / "terminal.json"
    terminal_sha = _write_jcs(terminal_path, terminal)
    rows = [
        {
            "pool": "Road.boundary_codim1",
            "interpretation": "corrected_xi",
            "status": "searched",
            "gradient_terms": gradient_terms,
            "claimed_kkt_residual_numerator": 0,
            "claimed_kkt_residual_denominator": 1,
            "terminal_evidence_path": str(terminal_path),
            "terminal_evidence_sha256": terminal_sha,
        }
    ]
    rows_sha = _sha(rfc8785_canonicalize(rows))
    audit = {
        "schema": "direct_description_completion_independent_audit.v1",
        "preregistration_sha256": preregistration_sha,
        "grammar_manifest_sha256": grammar_sha,
        "rows_sha256": rows_sha,
        "pairs": 600,
        "source_bound_replay": True,
        "auditor_independent": True,
        "outcome": "VERIFIED_DECLARED_FORMULATION_COMPLETION",
    }
    audit_path = tmp_path / "completion_audit.json"
    audit_sha = _write_jcs(audit_path, audit)
    return (
        {
            "schema": "DirectGrammarCompletionCertificateV1",
            "preregistration": preregistration,
            "preregistration_sha256": preregistration_sha,
            "optimizer_health": "HEALTHY",
            "restart_exhausted": False,
            "budget_exhausted": False,
            "verdict_scope": "FORMULATION_DECLARED_ANALYTIC_OPS_GRAMMAR",
            "rows": rows,
            "rows_sha256": rows_sha,
            "independent_audit_receipt_path": str(audit_path),
            "independent_audit_receipt_sha256": audit_sha,
        },
        preregistration_sha,
        audit_sha,
    )


def _failure_body(tmp_path: Path) -> tuple[dict, str]:
    archive_path, archive = _archive_file(tmp_path)
    target_path, target_sha = _target_receipt(tmp_path)
    caps = derive_ceil_minus_one_caps(target_path, target_sha)
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_bytes(b"# frozen evaluator fixture\n")
    certificate, preregistration_sha, audit_sha = _completion_certificate(tmp_path / "completion")
    measured_d_seg = 0.001
    measured_d_pose = 0.00010184
    score = 100 * measured_d_seg + math.sqrt(10 * measured_d_pose) + 25 * len(archive.archive) / ddm.SOURCE_BYTES
    archive_sha = _sha(archive.archive)
    readiness = dict.fromkeys(
        (
            "grammar",
            "archive_parse_reencode",
            "charged_free",
            "quarantine",
            "deterministic_decode",
            "storage",
            "resume",
            "evaluator",
            "external_attestation",
            "live_owners",
            "pose_owner",
            "completion",
            "byte_cap",
            "receiver_boundary",
        ),
        True,
    )
    return (
        {
            "schema": "DirectGrammarReceiverReachabilityFailureReceiptV1",
            "verdict_token": "DIRECT_GRAMMAR_RECEIVER_REACHABILITY_FAILURE",
            "verdict_scope": "FORMULATION_DECLARED_ANALYTIC_OPS_GRAMMAR",
            "primary_spec_sha256": ddm.PRIMARY_SPEC_SHA256,
            "git_sha": GIT_SHA,
            "seed": 1234,
            "run_id": "fixture_n600",
            "hardware_axis": "fabricated-axis",
            "full_precision_target_receipt_path": str(target_path),
            "full_precision_target_receipt_sha256": target_sha,
            "solved_d_seg": caps["solved_d_seg"],
            "solved_d_pose": caps["solved_d_pose"],
            "pointer_cap_bytes": caps["pointer_cap_bytes"],
            "pointer_cap_formula": "ceil_minus_one",
            "strict_0_15_cap_bytes": caps["strict_0_15_cap_bytes"],
            "strict_cap_role": "stretch_only",
            "grammar_manifest_sha256": H,
            "grammar_manifest_path": str(tmp_path / "missing_grammar.json"),
            "live_owner_receipt_manifest_sha256": H,
            "live_owner_receipt_manifest_path": str(tmp_path / "missing_owners.json"),
            "archive_path": str(archive_path),
            "archive_sha256": archive_sha,
            "archive_bytes": len(archive.archive),
            "parseback_sha256": archive_sha,
            "canonical_reencode_sha256": archive_sha,
            "raw_decode_sha256_x2": [H, H],
            "raw_decode_paths_x2": [str(tmp_path / "missing_raw_1"), str(tmp_path / "missing_raw_2")],
            "evaluator_path": str(evaluator),
            "evaluator_sha256": _sha(evaluator.read_bytes()),
            "scorer_runtime_path": str(tmp_path / "missing_runtime"),
            "scorer_runtime_sha256": H,
            "measurement_receipt_path": str(tmp_path / "missing_measurement.json"),
            "measurement_receipt_sha256": H,
            "measured_d_seg": measured_d_seg,
            "measured_d_pose": measured_d_pose,
            "measured_score": score,
            "pairs": 600,
            "optimizer_health": "HEALTHY",
            "completion_certificate": certificate,
            "completion_preregistration_sha256": preregistration_sha,
            "completion_independent_audit_sha256": audit_sha,
            "failed_receiver_predicates": ["solved_seg_cell_tolerance"],
            "all_readiness_predicates": readiness,
            "readiness_evidence_manifest_path": str(tmp_path / "missing_readiness.json"),
            "readiness_evidence_manifest_sha256": H,
            "external_attestation_path": str(tmp_path / "missing_attestation.json"),
            "external_attestation_sha256": H,
        },
        preregistration_sha,
    )


def test_owner_is_locked_and_custody_is_attached_last() -> None:
    owner = DirectDescriptionOpsGrammarMinimizerV1()
    assert owner.seed == 1234
    assert owner.execution_allowed is False
    assert owner.solve_order.startswith("seg_cells_then_pose")
    with pytest.raises(ValidationError):
        DirectDescriptionOpsGrammarMinimizerV1(seed=True)
    with pytest.raises(ValidationError):
        DirectDescriptionOpsGrammarMinimizerV1(n600_required_for_admission=False)
    with pytest.raises(ValidationError):
        DirectDescriptionOpsGrammarMinimizerV1(unknown_field=1)

    bundle = build_direct_description_owner()
    assert bundle["custody_argv_byte_identical"] is True
    assert bundle["pre_custody_typed_config_hash"] != bundle["typed_config_hash"]
    assert bundle["program_manifest"]["typed_config_hash"] == bundle["typed_config_hash"]
    assert bundle["program_manifest"]["compile_target"] == "DirectDescriptionWitnessProgramV1"
    assert bundle["consumer_argv"] == bundle["custody_argv"]
    assert bundle["consumer_argv"][2] == "tools/run_direct_description_minimizer.py"
    assert bundle["custody_constants_manifest"]["--seed"]["equation_id"] == ("dsl_custodied_scalar_identity_v1")
    assert "resolved_at" not in bundle["custody_constants_manifest"]["--seed"]
    assert build_direct_description_owner()["dsl_compile_hash"] == bundle["dsl_compile_hash"]


def test_real_consumer_parser_accepts_only_registered_tokens() -> None:
    parser = build_direct_description_arg_parser()
    parsed = parser.parse_args(
        [
            "--owner-manifest",
            "owner.json",
            "--mode",
            "preflight",
            "--execution-allowed",
            "false",
        ]
    )
    assert parsed.mode == "preflight"
    with pytest.raises(SystemExit):
        parser.parse_args(["--invented-flag", "1"])


def test_rfc8785_golden_unicode_and_numeric_edges() -> None:
    assert (
        rfc8785_canonicalize([333333333.33333329, 1e30, 4.50, 2e-3, 1e-27])
        == b"[333333333.3333333,1e+30,4.5,0.002,1e-27]"
    )
    assert rfc8785_canonicalize({"\u20ac": "x", "\r": "y", "1": "z"}) == (b'{"\\r":"y","1":"z","\xe2\x82\xac":"x"}')
    assert rfc8785_canonicalize(-0.0) == b"0"
    assert rfc8785_canonicalize([1e-7, 1e-6, 1e20, 1e21]) == (
        b"[1e-7,0.000001,100000000000000000000,1e+21]"
    )
    with pytest.raises(DirectDescriptionError):
        rfc8785_canonicalize(float("nan"))
    with pytest.raises(DirectDescriptionError):
        rfc8785_canonicalize(2**53)
    with pytest.raises(DirectDescriptionError):
        rfc8785_canonicalize("\ud800")


def test_z_archive_compile_parse_reencode_and_ledger(tmp_path: Path) -> None:
    z = _synthetic_z()
    first = compile_direct_description_archive(z)
    second = compile_direct_description_archive(z)
    assert first.archive == second.archive
    assert first.member == second.member
    assert parse_direct_description_archive(first.archive).z == z
    path = tmp_path / "archive.zip"
    path.write_bytes(first.archive)
    assert parse_direct_description_archive(path).archive == first.archive
    custody = first.custody()
    assert custody["archive_bytes"] == len(first.archive)
    assert custody["stream_payload_bytes"] != custody["archive_bytes"]
    assert {row["stream"] for row in custody["stream_ledger"]} == set(ddm._STREAM_TO_SECTION)

    with pytest.raises(ValidationError):
        DirectDescriptionZV1.model_validate(
            {**z.model_dump(), "unknown": CountedDescriptionStreamV1(payload=b"x", codec="raw", decoded_bytes=1)}
        )
    mutated = first.archive + b"trailing"
    with pytest.raises(DirectDescriptionError):
        parse_direct_description_archive(mutated)


def test_empty_optional_pose_stream_is_typed_but_not_an_active_receipt() -> None:
    z = _synthetic_z(empty_pose=True)
    result = compile_direct_description_archive(z)
    row = next(item for item in result.custody()["stream_ledger"] if item["stream"] == "pose6_dxi_residuals")
    assert row["encoded_payload_bytes"] == 0
    assert row["semantic_status"] == "LEGACY_OPAQUE_SECTION_REEXPRESSION"
    assert result.custody()["receiver_consumption_verified"] is False


def test_primary_live_semantics_cannot_be_self_asserted() -> None:
    with pytest.raises(ValidationError):
        CountedDescriptionStreamV1(
            payload=b"unbound",
            codec="raw",
            decoded_bytes=7,
            semantic_status="PRIMARY_LIVE_OWNER",
        )


def test_primary_quarantine_cannot_be_bypassed_by_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, _ = _archive_file(tmp_path)
    hit = type("Hit", (), {"identifier": "quarantined-fixture"})()
    monkeypatch.setattr(ddm, "is_quarantined_archive_bytes", lambda _payload: [hit])
    monkeypatch.setenv("TAC_ARTIFACT_QUARANTINE_SIGNAL_ONLY", "receipt harvesting only")
    with pytest.raises(DirectDescriptionError, match=r"non-waivable|refuses quarantine"):
        parse_direct_description_archive(path)
    with pytest.raises(DirectDescriptionError, match="refuses quarantine"):
        parse_direct_description_archive(path.read_bytes())


def test_settled_baseline_reexpression_when_ssd_is_present() -> None:
    if not Path(ddm.S4_BASELINE_ARCHIVE).is_file():
        pytest.skip("settled SSD control is unavailable on this host")
    receipt = prove_baseline_reexpression()
    assert receipt["verdict"] == "PASS_BYTE_EXACT"
    assert receipt["source_archive_bytes"] == 451_191
    assert receipt["source_archive_sha256"] == ddm.S4_BASELINE_SHA256


def test_caps_are_sha_bound_dynamic_decimal_and_ceil_minus_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontier_repo = tmp_path / "frontier"
    snapshot, now = _dynamic_frontier_snapshot(frontier_repo)
    monkeypatch.setattr(ddm, "_DYNAMIC_TARGET_REPO_ROOT", frontier_repo)
    target, sha = _target_receipt(tmp_path)
    caps = derive_ceil_minus_one_caps(
        target, sha, frontier_snapshot=snapshot, now_utc_iso=now
    )
    nonrate = Decimal(100) * Decimal(caps["solved_d_seg"]) + (
        Decimal(10) * Decimal(caps["solved_d_pose"])
    ).sqrt()
    continuous = (
        (Decimal(str(snapshot.target_score)) - nonrate)
        * Decimal(ddm.SOURCE_BYTES)
        / Decimal(25)
    )
    expected_pointer_cap = int(continuous.to_integral_value(rounding=ROUND_CEILING)) - 1
    assert caps["pointer_cap_bytes"] == expected_pointer_cap
    assert caps["pointer_score"] == str(snapshot.target_score)
    assert caps["dynamic_frontier_target"]["pointer_sha256"] == snapshot.pointer_sha256
    assert caps["strict_0_15_cap_bytes"] == 154_524
    assert caps["strict_cap_role"] == "stretch_only"
    with pytest.raises(DirectDescriptionError, match="SHA-256 mismatch"):
        derive_ceil_minus_one_caps(
            target, "0" * 64, frontier_snapshot=snapshot, now_utc_iso=now
        )
    planning, planning_sha = _target_receipt(tmp_path, planning_only=True)
    with pytest.raises(DirectDescriptionError, match="planning"):
        derive_ceil_minus_one_caps(
            planning, planning_sha, frontier_snapshot=snapshot, now_utc_iso=now
        )
    bad = tmp_path / "float.json"
    bad_value = json.loads(target.read_text())
    bad_value["solved_d_seg"] = 0.00015196
    bad_value["solved_d_pose"] = 0.00010184
    bad_sha = _write_jcs(bad, bad_value)
    with pytest.raises(DirectDescriptionError, match="strings"):
        derive_ceil_minus_one_caps(
            bad, bad_sha, frontier_snapshot=snapshot, now_utc_iso=now
        )
    rounded = tmp_path / "rounded.json"
    rounded_value = json.loads(target.read_text())
    rounded_value["solved_d_seg"] = "0.00015196"
    rounded_value["solved_d_pose"] = "0.00010184"
    with pytest.raises(DirectDescriptionError, match="display-rounded"):
        derive_ceil_minus_one_caps(
            rounded,
            _write_jcs(rounded, rounded_value),
            frontier_snapshot=snapshot,
            now_utc_iso=now,
        )


def test_caps_refuse_forged_stale_and_path_swapped_frontier_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontier_repo = tmp_path / "canonical"
    snapshot, now = _dynamic_frontier_snapshot(frontier_repo)
    monkeypatch.setattr(ddm, "_DYNAMIC_TARGET_REPO_ROOT", frontier_repo)
    target, sha = _target_receipt(tmp_path / "target")

    with pytest.raises(DirectDescriptionError, match="changed after snapshot"):
        derive_ceil_minus_one_caps(
            target,
            sha,
            frontier_snapshot=replace(snapshot, target_score=0.001),
            now_utc_iso=now,
        )
    stale_time = (datetime.fromisoformat(now) - timedelta(hours=25)).isoformat()
    with pytest.raises(DirectDescriptionError, match="24-hour"):
        derive_ceil_minus_one_caps(
            target,
            sha,
            frontier_snapshot=replace(snapshot, last_refreshed_utc=stale_time),
            now_utc_iso=now,
        )
    swapped_repo = tmp_path / "swapped"
    swapped, _ = _dynamic_frontier_snapshot(swapped_repo, score=0.24)
    with pytest.raises(DirectDescriptionError, match="noncanonical pointer path"):
        derive_ceil_minus_one_caps(
            target, sha, frontier_snapshot=swapped, now_utc_iso=now
        )


def test_caps_refuse_pointer_refresh_during_receipt_custody_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontier_repo = tmp_path / "canonical"
    snapshot, now = _dynamic_frontier_snapshot(frontier_repo)
    monkeypatch.setattr(ddm, "_DYNAMIC_TARGET_REPO_ROOT", frontier_repo)
    target, sha = _target_receipt(tmp_path / "target")
    original_decode = ddm._duplicate_refusing_json
    replaced = False

    def replace_pointer_once(payload: bytes):
        nonlocal replaced
        decoded = original_decode(payload)
        if not replaced:
            replaced = True
            _dynamic_frontier_snapshot(frontier_repo, score=0.24)
        return decoded

    monkeypatch.setattr(ddm, "_duplicate_refusing_json", replace_pointer_once)
    with pytest.raises(DirectDescriptionError, match="changed after snapshot"):
        derive_ceil_minus_one_caps(
            target, sha, frontier_snapshot=snapshot, now_utc_iso=now
        )


def test_numpy_reference_is_seeded_deterministic_and_rejects_alias_types() -> None:
    rows = [
        {"candidate_id": "a", "d_seg": 0.001, "d_pose": 0.01, "archive_bytes": 100},
        {"candidate_id": "b", "d_seg": 0.001, "d_pose": 0.01, "archive_bytes": 100},
    ]
    assert numpy_reference_rank(rows) == numpy_reference_rank(rows)
    assert {row["candidate_id"] for row in numpy_reference_rank(rows)} == {"a", "b"}
    with pytest.raises(DirectDescriptionError):
        numpy_reference_rank([{**rows[0], "archive_bytes": True}])
    with pytest.raises(DirectDescriptionError):
        numpy_reference_rank([{**rows[0], "d_seg": float("inf")}])


def _allocation_tree() -> list[ToleranceAllocationNodeV1]:
    rows: list[ToleranceAllocationNodeV1] = []
    for class_name in CLASSES:
        strata = sorted(ddm._APPLICABLE_STRATA[class_name])
        chart = "corrected_xi" if class_name in {"Road", "Lane"} else "image_or_object"
        root_id = f"{class_name}_root"
        rows.append(
            ToleranceAllocationNodeV1(
                node_id=root_id,
                parent_id=None,
                axis="class",
                label=class_name,
                chart=chart,
                tolerance=Decimal("0.000800"),
                archive_bytes=len(strata),
                d_seg_contribution=Decimal("0.000001") * len(strata),
                marginal_gain_numerator=25,
                marginal_gain_denominator=37_545_489,
                quantization_floor=Decimal("0"),
                receipt_sha256=H,
            )
        )
        for stratum in strata:
            parent = root_id
            for index, axis in enumerate(AXES[1:], 1):
                node_id = f"{class_name}_{stratum}_{index}"
                label = stratum if axis == "canonical_stratum" else f"{axis}_fixture"
                rows.append(
                    ToleranceAllocationNodeV1(
                        node_id=node_id,
                        parent_id=parent,
                        axis=axis,
                        label=label,
                        chart=chart,
                        tolerance=Decimal("0.000800"),
                        archive_bytes=1,
                        d_seg_contribution=Decimal("0.000001"),
                        marginal_gain_numerator=25,
                        marginal_gain_denominator=37_545_489,
                        quantization_floor=Decimal("0"),
                        receipt_sha256=H,
                    )
                )
                parent = node_id
    return rows


def test_recursive_allocation_tree_exact_lambda_and_edge_refusals() -> None:
    result = verify_allocation_tree(_allocation_tree())
    assert result["structurally_valid"] is True
    assert result["verified"] is False
    assert result["node_count"] == 49
    assert result["common_lambda_exact"] == "25/37545489"
    assert result["maximum_absolute_marginal_residual_exact"] == "0/1"
    roots_only = [row for row in _allocation_tree() if row.axis == "class"]
    with pytest.raises(DirectDescriptionError, match="leaves must reach"):
        verify_allocation_tree(roots_only)
    broken = _allocation_tree()
    broken[1] = broken[1].model_copy(update={"archive_bytes": 2})
    with pytest.raises(DirectDescriptionError, match="reconcile"):
        verify_allocation_tree(broken)
    off_lambda = _allocation_tree()
    off_lambda[-1] = off_lambda[-1].model_copy(update={"marginal_gain_numerator": 0})
    with pytest.raises(DirectDescriptionError, match="quantization floor"):
        verify_allocation_tree(off_lambda)
    wrong_chart = _allocation_tree()
    road_index = next(index for index, row in enumerate(wrong_chart) if row.label == "Road")
    wrong_chart[road_index] = wrong_chart[road_index].model_copy(update={"chart": "monolithic_g1_bev"})
    with pytest.raises(DirectDescriptionError, match="coordinate chart"):
        verify_allocation_tree(wrong_chart)


def test_charged_free_partition_does_not_turn_nullity_into_bytes() -> None:
    rows = [
        ChargedFreePartitionRowV1(
            component=name,
            disposition="COUNTED",
            justification="video-derived complete-description residue",
            bytes_if_counted=1,
            receipt_sha256=H,
            video_derived=True,
            generic_decoder_logic_only=False,
        )
        for name in ddm._STREAM_TO_SECTION
    ]
    rows += [
        ChargedFreePartitionRowV1(
            component="tropical_argmax_generic_algorithm",
            disposition="FREE",
            justification="rule-118 generic decoder logic",
            receipt_sha256=H,
            video_derived=False,
            generic_decoder_logic_only=True,
        ),
        ChargedFreePartitionRowV1(
            component="resize_kernel_blind_subspace",
            disposition="NULL",
            justification="measured geometry only; no byte delta claimed",
            receipt_sha256=H,
            video_derived=False,
            generic_decoder_logic_only=False,
        ),
    ]
    result = verify_charged_free_partition(rows)
    assert result["counted_description_payload_bytes"] == 6
    assert result["counted_description_payload_is_archive_bytes"] is False
    assert result["null_byte_savings"] is None
    with pytest.raises(ValidationError):
        ChargedFreePartitionRowV1(
            component="hidden_video_table",
            disposition="FREE",
            justification="fake",
            receipt_sha256=H,
            video_derived=True,
            generic_decoder_logic_only=True,
        )


def test_dual_metric_telemetry_is_extensible_and_hard_oracle_separate() -> None:
    row = DescriptionStepMetricTelemetryV1(
        step=1,
        tolerance_rung="0.000800",
        euclidean_cosine=0.5,
        fisher_cosine=-0.5,
        relative_norm_ratio=1.2,
        additional_metrics={"hilbert_projective": 0.25},
        exact_secant_direction_sha256=H,
        hard_receiver_admitted=False,
    )
    assert row.euclidean_cosine * row.fisher_cosine < 0
    assert row.hard_receiver_admitted is False
    with pytest.raises(ValidationError):
        DescriptionStepMetricTelemetryV1(**{**row.model_dump(), "euclidean_cosine": 1.01})


def test_measurement_ladder_requires_all_cells_then_tube_rows(tmp_path: Path) -> None:
    path, archive = _archive_file(tmp_path)
    raw_a = tmp_path / "decode_a.raw"
    raw_b = tmp_path / "decode_b.raw"
    raw_a.write_bytes(b"deterministic-test-raw")
    raw_b.write_bytes(raw_a.read_bytes())
    raw_sha = _sha(raw_a.read_bytes())
    rows: list[MeasurementRungRowV1] = []
    for index, tolerance in enumerate(ddm.TOLERANCE_RUNG_TEXT):
        receipt = {
            "schema": "direct_description_n600_measurement.v1",
            "pairs": 600,
            "archive_bytes": len(archive.archive),
            "archive_sha256": _sha(archive.archive),
            "d_seg": 0.0001,
            "d_pose": 0.001,
            "generator_lineage": "fresh_direct_description_fixture",
            "through_R": True,
            "exact_evaluator_called": True,
            "seg_constraints_solved_first": True,
            "pose_solved_within_seg_feasible_polytope": True,
            "hardware_axis": "[contest-CPU]",
            "raw_decode_paths_x2": [str(raw_a), str(raw_b)],
            "raw_decode_sha256_x2": [raw_sha, raw_sha],
        }
        receipt_path = tmp_path / f"measurement_{index}.json"
        receipt_sha = _write_jcs(receipt_path, receipt)
        rows.append(
            MeasurementRungRowV1(
                tolerance=tolerance,
                archive_path=str(path),
                archive_bytes=len(archive.archive),
                archive_sha256=_sha(archive.archive),
                d_seg=0.0001,
                d_pose=0.001,
                generator_lineage="fresh_direct_description_fixture",
                receipt_path=str(receipt_path),
                receipt_sha256=receipt_sha,
            )
        )
    result = verify_measurement_ladder(rows)
    assert result["verified"] is False
    assert result["structurally_valid"] is True
    assert len(result["rows"]) == 4
    with pytest.raises(DirectDescriptionError):
        verify_measurement_ladder(rows[:-1])
    with pytest.raises(ValidationError):
        MeasurementRungRowV1(**{**rows[0].model_dump(), "d_seg": 1.0})
    aliased_receipt = dict(receipt)
    aliased_receipt["raw_decode_paths_x2"] = [str(raw_a), str(raw_a)]
    aliased_path = tmp_path / "measurement_aliased_decode.json"
    aliased_sha = _write_jcs(aliased_path, aliased_receipt)
    with pytest.raises(ValidationError, match="distinct paths"):
        MeasurementRungRowV1(
            **{
                **rows[0].model_dump(),
                "receipt_path": str(aliased_path),
                "receipt_sha256": aliased_sha,
            }
        )
    hardlink = tmp_path / "decode_a_hardlink.raw"
    os.link(raw_a, hardlink)
    hardlink_receipt = dict(receipt)
    hardlink_receipt["raw_decode_paths_x2"] = [str(raw_a), str(hardlink)]
    hardlink_path = tmp_path / "measurement_hardlink_decode.json"
    hardlink_sha = _write_jcs(hardlink_path, hardlink_receipt)
    with pytest.raises(ValidationError, match="distinct file identities"):
        MeasurementRungRowV1(
            **{
                **rows[0].model_dump(),
                "receipt_path": str(hardlink_path),
                "receipt_sha256": hardlink_sha,
            }
        )


def _checkpoint(tmp_path: Path, *, phase: str = "static_grammar") -> DirectDescriptionStageCheckpointV1:
    _path, archive = _archive_file(tmp_path)
    argv = ("python", "tools/run_direct_description_minimizer.py")
    config = {"seed": 1234, "dsl_compile_hash": H}
    return DirectDescriptionStageCheckpointV1(
        config_sha256=_sha(rfc8785_canonicalize(config)),
        dsl_compile_hash=H,
        run_id="fixture_run",
        stage_name="static_proof",
        stage_phase=phase,
        stage_index=0,
        epoch=0,
        global_step=0,
        next_pair=0,
        active_tolerance="0.000800",
        active_rate_rung=0,
        z_member_b64=base64.b64encode(archive.member).decode("ascii"),
        z_member_sha256=_sha(archive.member),
        entropy_state={"status": "initialized"},
        archive_compiler_state={"schema": "s4.v1"},
        optimizer_state={"step": 0},
        ema_shadow={"used": False},
        rng_state={"seed": 1234},
        allocation_pool_states={"status": "unsearched"},
        hard_oracle_receipt_chain=(),
        step_metric_telemetry=(),
        seg_constraints_solved_for_active_tolerance=False,
        pose_solved_within_seg_feasible_polytope=False,
        best_archive_b64=base64.b64encode(archive.archive).decode("ascii"),
        best_archive_sha256=_sha(archive.archive),
        best_archive_bytes=len(archive.archive),
        trigger_certificate_state={"status": "not_eligible"},
        config=config,
        argv=argv,
        argv_sha256=_sha("\0".join(argv).encode()),
    )


def test_checkpoint_roundtrip_atomic_no_clobber_and_solve_order(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    payload = checkpoint.to_bytes()
    assert DirectDescriptionStageCheckpointV1.from_bytes(payload) == checkpoint
    path = checkpoint.write_new(tmp_path / "checkpoints")
    assert path.name == checkpoint.filename()
    assert (
        load_stage_checkpoint(
            path,
            expected_config_sha256=checkpoint.config_sha256,
            expected_dsl_compile_hash=checkpoint.dsl_compile_hash,
            expected_argv=checkpoint.argv,
        )
        == checkpoint
    )
    with pytest.raises(DirectDescriptionError, match="already exists"):
        checkpoint.write_new(tmp_path / "checkpoints")
    with pytest.raises(DirectDescriptionError):
        DirectDescriptionStageCheckpointV1.from_bytes(payload[:-1] + b"x")
    with pytest.raises(ValidationError, match="cells-then-pose"):
        _checkpoint(tmp_path, phase="tolerance")


def test_storage_waterfall_admits_first_real_tier_and_refuses_capacity(tmp_path: Path) -> None:
    receipt = storage_preflight(0, reserve_bytes=0, tiers=(tmp_path / "missing", tmp_path))
    assert receipt["outcome"] == "ADMIT"
    assert receipt["selected_tier"] == str(tmp_path)
    refused = storage_preflight(10**30, reserve_bytes=0, tiers=(tmp_path,))
    assert refused["outcome"] == "REFUSE"
    with pytest.raises(DirectDescriptionError):
        storage_preflight(True, tiers=(tmp_path,))


def test_completion_certificate_is_source_bound_and_independently_recomputed(tmp_path: Path) -> None:
    certificate, preregistration_sha, audit_sha = _completion_certificate(tmp_path)
    result = verify_completion_certificate(
        certificate,
        expected_preregistration_sha256=preregistration_sha,
        expected_grammar_manifest_sha256=H,
        expected_independent_audit_sha256=audit_sha,
    )
    assert result["maximum_kkt_residual_exact"] == "0/1"
    assert (
        optimizer_admission_status(
            certificate,
            expected_preregistration_sha256=preregistration_sha,
            expected_grammar_manifest_sha256=H,
            expected_independent_audit_sha256=audit_sha,
        )
        == "COMPLETION_CERTIFIED"
    )
    forged = json.loads(json.dumps(certificate))
    forged["rows"][0]["claimed_kkt_residual_numerator"] = 1
    forged["rows_sha256"] = _sha(rfc8785_canonicalize(forged["rows"]))
    with pytest.raises(DirectDescriptionError, match="independently"):
        verify_completion_certificate(
            forged,
            expected_preregistration_sha256=preregistration_sha,
            expected_grammar_manifest_sha256=H,
            expected_independent_audit_sha256=audit_sha,
        )
    forged["budget_exhausted"] = True
    assert (
        optimizer_admission_status(
            forged,
            expected_preregistration_sha256=preregistration_sha,
            expected_grammar_manifest_sha256=H,
            expected_independent_audit_sha256=audit_sha,
        )
        == "OPTIMIZER_NO_ADMISSION"
    )


def test_completion_rejects_cross_coordinate_cancellation_and_empty_search(tmp_path: Path) -> None:
    certificate, preregistration_sha, audit_sha = _completion_certificate(tmp_path / "coordinates")
    forged = json.loads(json.dumps(certificate))
    forged["rows"][0]["gradient_terms"][1]["coordinate"] = "Road.boundary_codim1.y"
    forged["rows"][0]["claimed_kkt_residual_numerator"] = 1
    forged["rows"][0]["claimed_kkt_residual_denominator"] = 2
    terminal_path = Path(forged["rows"][0]["terminal_evidence_path"])
    terminal = json.loads(terminal_path.read_text())
    terminal["gradient_terms"] = forged["rows"][0]["gradient_terms"]
    forged["rows"][0]["terminal_evidence_sha256"] = _write_jcs(terminal_path, terminal)
    forged["rows_sha256"] = _sha(rfc8785_canonicalize(forged["rows"]))
    with pytest.raises(DirectDescriptionError, match="KKT threshold"):
        verify_completion_certificate(
            forged,
            expected_preregistration_sha256=preregistration_sha,
            expected_grammar_manifest_sha256=H,
            expected_independent_audit_sha256=audit_sha,
        )

    certificate, preregistration_sha, _ = _completion_certificate(tmp_path / "excluded")
    row = certificate["rows"][0]
    terminal = {
        "schema": "direct_description_terminal_pool_evidence.v1",
        "grammar_manifest_sha256": H,
        "preregistration_sha256": preregistration_sha,
        "pool": row["pool"],
        "interpretation": row["interpretation"],
        "pairs": 600,
        "status": "gate_excluded",
    }
    terminal_path = tmp_path / "excluded" / "terminal_gate.json"
    terminal_sha = _write_jcs(terminal_path, terminal)
    gate = {
        "schema": "direct_description_gate_exclusion.v1",
        "grammar_manifest_sha256": H,
        "preregistration_sha256": preregistration_sha,
        "pool": row["pool"],
        "interpretation": row["interpretation"],
        "pairs": 600,
        "outcome": "EXCLUDED_BY_PREREGISTERED_GATE",
        "terminal_evidence_sha256": terminal_sha,
    }
    gate_path = tmp_path / "excluded" / "gate.json"
    gate_sha = _write_jcs(gate_path, gate)
    certificate["rows"] = [
        {
            "pool": row["pool"],
            "interpretation": row["interpretation"],
            "status": "gate_excluded",
            "gradient_terms": [],
            "terminal_evidence_path": str(terminal_path),
            "terminal_evidence_sha256": terminal_sha,
            "gate_receipt_path": str(gate_path),
            "gate_receipt_sha256": gate_sha,
        }
    ]
    certificate["rows_sha256"] = _sha(rfc8785_canonicalize(certificate["rows"]))
    with pytest.raises(DirectDescriptionError, match="empty/all-gate-excluded"):
        verify_completion_certificate(
            certificate,
            expected_preregistration_sha256=preregistration_sha,
            expected_grammar_manifest_sha256=H,
            expected_independent_audit_sha256=certificate["independent_audit_receipt_sha256"],
        )


def test_failure_receipt_refuses_self_asserted_fixture_and_fabricated_axis(tmp_path: Path) -> None:
    body, preregistration_sha = _failure_body(tmp_path)
    with pytest.raises(DirectDescriptionError, match="contest authority axis"):
        seal_failure_receipt(body, expected_completion_preregistration_sha256=preregistration_sha)
    body["hardware_axis"] = "[contest-CPU]"
    with pytest.raises(DirectDescriptionError, match="receipt file cannot be inspected"):
        seal_failure_receipt(body, expected_completion_preregistration_sha256=preregistration_sha)


def test_unhealthy_optimizer_cannot_mint_failure_token(tmp_path: Path) -> None:
    body, preregistration_sha = _failure_body(tmp_path)
    body["hardware_axis"] = "[contest-CPU]"
    body["optimizer_health"] = "UNHEALTHY"
    with pytest.raises(DirectDescriptionError, match="unhealthy"):
        seal_failure_receipt(
            body,
            expected_completion_preregistration_sha256=preregistration_sha,
        )


def _receiver_rate_receipt(tmp_path: Path) -> Path:
    archive_path, archive = _archive_file(tmp_path)
    rows = []
    for index, name in enumerate(CLASSES):
        measured = len(archive.archive) if index == 0 else 0
        rows.append(
            {
                "class_name": name,
                "measured_unique_home_bytes": measured,
                "archive_byte_ranges": [[0, len(archive.archive)]] if index == 0 else [],
                "consumption_count": 1 if index == 0 else 0,
                "receiver_handler_sha256": H if index == 0 else None,
                "output_mutation_sha256": H if index == 0 else None,
            }
        )
    value = {
        "schema": "direct_description_receiver_rate_custody.v1",
        "candidate_role": "fresh_primary_candidate",
        "archive_path": str(archive_path),
        "archive_bytes": len(archive.archive),
        "archive_sha256": _sha(archive.archive),
        "parser_consumption_receipt_sha256": H,
        "n64_receipt_sha256": H,
        "n600_receipt_sha256": H,
        "n64_deterministic": True,
        "n600_measured": True,
        "exact_evaluator_called": False,
        "class_rows": rows,
        "description_dimension_bytes": dict.fromkeys(ddm._STREAM_TO_SECTION, 0),
        "dimension_bytes": dict.fromkeys(
            ("pixel", "class", "boundary", "frame", "pair", "epoch", "chroma", "scale", "frequency"), 0
        ),
    }
    path = tmp_path / "receiver_rate.json"
    path.write_bytes(rfc8785_canonicalize(value) + b"\n")
    return path


def test_receiver_rate_custody_refuses_nonlocal_deflate_attribution(tmp_path: Path) -> None:
    path = _receiver_rate_receipt(tmp_path)
    with pytest.raises(DirectDescriptionError, match="UNSUPPORTED_NONLOCAL_DEFLATE_ATTRIBUTION"):
        validate_receiver_rate_custody(path)
    value = json.loads(path.read_text())
    value["candidate_role"] = "control"
    path.write_text(json.dumps(value))
    with pytest.raises(DirectDescriptionError, match="controls"):
        validate_receiver_rate_custody(path)


def test_launch_ticket_remains_structurally_draft(tmp_path: Path) -> None:
    owner = build_direct_description_owner()
    storage = storage_preflight(0, reserve_bytes=0, tiers=(tmp_path,))
    ticket = build_launch_readiness(owner, storage_receipt=storage)
    assert ticket["launch_ready"] is False
    assert ticket["spawn_permitted"] is False
    assert "PRIMARY_SPEC_EXECUTION_ALLOWED_FALSE" in ticket["blockers"]
    assert "FRESH_V3_FAMILY_POSE_IN_OBJECTIVE_RUNG_ZERO_MISSING" in ticket["blockers"]
    assert "CANONICAL_TYPED_COMPILER_INTEGRATION_MISSING" in ticket["blockers"]
    assert "CANONICAL_RESUME_REGISTRY_AND_CHECKPOINT_CADENCE_NOT_IMPLEMENTED" in ticket["blockers"]
    forged = dict(owner)
    forged["dsl_compile_hash"] = "0" * 64
    with pytest.raises(DirectDescriptionError, match="compile hash"):
        build_launch_readiness(forged, storage_receipt=storage)


def test_v2_pose_receiver_roundtrip_unique_homes_and_noop_detector() -> None:
    config = ddm.DirectDescriptionOptimizerConfigV1()
    initial, _target = ddm.build_n64_custody_descriptions(config)
    built = ddm.compile_direct_description_archive_v2(initial)
    assert ddm.compile_direct_description_archive_v2(
        ddm.parse_direct_description_archive_v2(built.archive).z
    ).archive == built.archive
    received = ddm.receive_direct_description_archive_v2(built.archive)
    assert received.output.dtype.name == "uint8"
    assert received.output.shape == (64, 2, 8, 8, 3)
    assert received.custody["pose6_records_consumed"] == 64
    assert received.custody["pose6_scalar_residuals_consumed"] == 384
    homes = received.custody["unique_final_zip_homes"]
    assert len(homes) == 7
    assert sum(row["home_bytes"] for row in homes) == len(built.archive)
    assert received.custody["all_archive_bytes_have_one_home"] is True

    pose = initial.pose6_dxi_residuals.payload
    mutated = initial.replace_stream_byte("pose6_dxi_residuals", 0, pose[0] ^ 1)
    mutated_result = ddm.receive_direct_description_archive_v2(
        ddm.compile_direct_description_archive_v2(mutated).archive
    )
    assert mutated_result.output_sha256 != received.output_sha256

    noop = ddm.prove_v2_noop_detector(initial)
    assert noop["archive_bytes_checked"] == len(built.archive)
    assert noop["semantic_payload_bytes_checked"] == sum(ddm._V2_BODY_BYTES.values())
    assert noop["all_archive_bytes_read_or_output_effective"] is True
    assert noop["all_semantic_payload_bytes_output_effective"] is True


def test_real_optimizer_descends_checkpoints_and_resumes_bit_exact(tmp_path: Path) -> None:
    config = ddm.DirectDescriptionOptimizerConfigV1()
    argv = ("ddm-local-custody", config.dsl_compile_hash())
    run_a = ddm.run_direct_description_optimizer(
        config,
        checkpoint_directory=tmp_path / "a",
        semantic_argv=argv,
    )
    run_b = ddm.run_direct_description_optimizer(
        config,
        checkpoint_directory=tmp_path / "b",
        semantic_argv=argv,
    )
    partial = ddm.run_direct_description_optimizer(
        config,
        checkpoint_directory=tmp_path / "resume",
        semantic_argv=argv,
        stop_after_stage_index=0,
    )
    assert partial.complete is False
    resumed = ddm.run_direct_description_optimizer(
        config,
        checkpoint_directory=tmp_path / "resume",
        semantic_argv=argv,
        resume_from=partial.checkpoint_paths[-1],
    )
    assert run_a.complete is run_b.complete is resumed.complete is True
    assert run_a.final_archive == run_b.final_archive == resumed.final_archive
    assert (
        run_a.final_receiver.output_sha256
        == run_b.final_receiver.output_sha256
        == resumed.final_receiver.output_sha256
    )
    assert len(run_a.checkpoint_paths) == len(config.stages) == 3
    assert len({path.name for path in run_a.checkpoint_paths}) == 3
    assert all(path.is_file() for path in (*partial.checkpoint_paths, *resumed.checkpoint_paths))
    assert all(row["stage_role"] == "candidate_search" for row in run_a.stage_history)
    assert all(row["strict_descent"] is True for row in run_a.stage_history)
    assert all(
        all(count > 0 for count in row["coordinates_by_stream"].values())
        for row in run_a.stage_history
    )
    assert run_a.objective["joint_integer_debt"] < run_a.stage_history[0]["objective_before"][
        "joint_integer_debt"
    ]
    checkpoint = ddm.load_optimizer_checkpoint(
        partial.checkpoint_paths[-1],
        expected_config=config,
        expected_semantic_argv=argv,
    )
    assert checkpoint.next_stage_index == 1
    assert checkpoint.optimizer_state["candidate_evaluations"] > 0
    with pytest.raises(DirectDescriptionError, match="config differs"):
        ddm.load_optimizer_checkpoint(
            partial.checkpoint_paths[-1],
            expected_config=config.model_copy(update={"run_id": "different_run"}),
            expected_semantic_argv=argv,
        )
    tampered_path = tmp_path / "tampered_checkpoint.json"
    tampered = bytearray(partial.checkpoint_paths[-1].read_bytes())
    tampered[-1] ^= 1
    tampered_path.write_bytes(tampered)
    with pytest.raises(DirectDescriptionError):
        ddm.load_optimizer_checkpoint(
            tampered_path,
            expected_config=config,
            expected_semantic_argv=argv,
        )


def test_n64_custody_smoke_receipts_five_scoped_green_blockers(tmp_path: Path) -> None:
    config = ddm.DirectDescriptionOptimizerConfigV1()
    argv = ("ddm-local-custody", config.dsl_compile_hash())
    receipt, receipt_path = ddm.run_n64_deterministic_custody_smoke(
        config,
        output_directory=tmp_path / "smoke",
        semantic_argv=argv,
    )
    assert receipt_path.is_file()
    assert receipt["label"] == "[custody-smoke]"
    assert receipt["score_claim"] is False
    assert receipt["candidate_archive"] is False
    assert receipt["determinism"]["same_seed_final_archive_bit_identical"] is True
    assert receipt["resume"]["all_stage_checkpoints_preserved"] is True
    assert receipt["roundtrip"]["parse_reencode_archive_byte_exact"] is True
    register = receipt["blocker_register"]
    assert len(register) == 19
    assert sum(row["flipped_green"] for row in register) == 4
    canonical_resume = next(
        row
        for row in register
        if row["blocker"] == "CANONICAL_RESUME_REGISTRY_AND_CHECKPOINT_CADENCE"
    )
    assert canonical_resume["flipped_green"] is False
    coverage = receipt["stratified_fixture_coverage"]
    assert coverage["pair_count"] == len(coverage["pair_assignments"]) == 64
    assert coverage["all_applicable_combinations_covered"] is True
    assert all(
        row["verdict_scope"] == "local deterministic n64 custody apparatus only"
        for row in register
        if row["flipped_green"]
    )
