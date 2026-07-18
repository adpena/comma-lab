# SPDX-License-Identifier: MIT
"""Focused fixture tests for the governed frame-195 diagnostic."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tools.diagnose_v10_power_diagram_frame195 as diagnostic
import tools.harvest_v10_power_diagram_blocked_prefix as harvester
import tools.v10_power_diagram_blocked_evidence as evidence


def _target() -> object:
    weight = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0],
        ]
    )
    return evidence.affine_scores_to_power_target(
        weight,
        np.array([0.1, -0.1, 0.2, -0.2, 0.0]),
        adjacency=tuple((i, j) for i in range(5) for j in range(i + 1, 5)),
    )


def _parity() -> SimpleNamespace:
    return SimpleNamespace(
        authority_label="MEASURED_NUMERICAL_PARITY_NOT_BOUNDARY_THEOREM",
        sample_count=1,
        mismatch_count=0,
        sample_agreement=1.0,
        max_pair_score_error=1e-7,
        minimum_affine_winner_margin=2e-7,
        f32_tie_uncertain_count=0,
        exact_on_samples=True,
        boundary_exactness="NO_GENERAL_VERDICT_WITHIN_F32_TIE_UNCERTAINTY",
    )


def _receipt() -> dict:
    reproduction = diagnostic.analyze_pixel(
        logits=np.array([0.1, 0.2, 0.3, 0.4, 0.0]),
        quotient_point=np.array([0.25, -0.5, 0.75, -1.0]),
        target=_target(),
        cached_lstar=3,
        parity_receipt=_parity(),
    )
    return {
        "schema": diagnostic.SCHEMA,
        "status": diagnostic.STATUS,
        "authority": {
            "new_run_label": diagnostic.MEASURED_REPRODUCTION,
            "checkpoint_aggregate_label": diagnostic.MEASURED_PRESERVED_STATE,
            "one_frame_one_pixel_only": True,
            "n600_rerun": False,
            "through_r_authority": False,
            "receiver_authority": False,
            "score_authority": False,
            "promotion_eligible": False,
            "cleanup_performed": False,
            "ssd_write_performed": False,
        },
        "outer_wrapper_custody": {
            "label": "OPERATOR_SUPPLIED_NOT_INNER_RUNTIME_MEASUREMENT",
            "original_governed_run_limits_claimed": False,
        },
        "preserved_checkpoint_state": {
            "label": diagnostic.MEASURED_PRESERVED_STATE,
            "next_canonical_frame": 195,
            "sample_count": diagnostic.EXPECTED_PREFIX_SAMPLES,
            "power_target_mismatch_count": 1,
            "cpu_torch_forward_mismatch_count": 0,
        },
        "reproduction": reproduction,
    }


def test_exact_diagnostic_scope_constants_are_stable() -> None:
    assert diagnostic.FRAME_INDEX == 195
    assert diagnostic.PIXEL_Y == 214
    assert diagnostic.PIXEL_X == 112
    assert diagnostic.MEASURED_REPRODUCTION == "MEASURED_REPRODUCTION"
    assert diagnostic.MEASURED_PRESERVED_STATE == "MEASURED_PRESERVED_STATE"


def test_native_f32_power_and_pixel_analysis_are_deterministic_and_pure() -> None:
    point = np.array([0.25, -0.5, 0.75, -1.0], dtype=np.float32)
    logits = np.array([0.1, 0.2, 0.3, 0.4, 0.0], dtype=np.float32)
    point_before = point.copy()
    logits_before = logits.copy()
    first = diagnostic.analyze_pixel(
        logits=logits,
        quotient_point=point,
        target=_target(),
        cached_lstar=3,
        parity_receipt=_parity(),
    )
    second = diagnostic.analyze_pixel(
        logits=logits,
        quotient_point=point,
        target=_target(),
        cached_lstar=3,
        parity_receipt=_parity(),
    )
    assert first == second
    np.testing.assert_array_equal(point, point_before)
    np.testing.assert_array_equal(logits, logits_before)
    assert first["label"] == diagnostic.MEASURED_REPRODUCTION
    assert first["frame"] == 195
    assert len(first["generic_f64_power"]["scores"]) == 5
    assert len(first["native_f32_power"]["scores"]) == 5


def test_input_fingerprint_is_unchanged_and_mtime_drift_is_refused(tmp_path: Path) -> None:
    artifact = tmp_path / "immutable.bin"
    artifact.write_bytes(b"preserve")
    digest = diagnostic.sha256_file(artifact)
    verified = diagnostic._verify_file(artifact, expected_sha256=digest, role="fixture")
    diagnostic._assert_unchanged(verified, role="fixture")
    stat = artifact.stat()
    os.utime(artifact, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(RuntimeError, match="changed after immutable verification"):
        diagnostic._assert_unchanged(verified, role="fixture")
    assert artifact.read_bytes() == b"preserve"


def test_post_run_hash_check_refuses_same_size_same_mtime_content_masquerade(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "immutable.bin"
    artifact.write_bytes(b"original")
    digest = diagnostic.sha256_file(artifact)
    verified = diagnostic._verify_file(artifact, expected_sha256=digest, role="fixture")
    artifact.write_bytes(b"masquera")
    os.utime(artifact, ns=(artifact.stat().st_atime_ns, verified.mtime_ns))
    with pytest.raises(RuntimeError, match="content hash changed"):
        diagnostic._assert_unchanged(verified, role="fixture", verify_hash=True)


def test_diagnostic_refuses_every_nonresearch_output_class_before_reading_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "isolated_worktree"
    research = repo / ".omx/research"
    tools = repo / "tools"
    source = repo / "src"
    main_checkout = tmp_path / "main_checkout/.omx/research"
    ssd = tmp_path / "ssd/pact"
    transient = tmp_path / "transient"
    outside = tmp_path / "outside"
    for directory in (research, tools, source, main_checkout, ssd, transient, outside):
        directory.mkdir(parents=True)
    resolved_repo = repo.resolve()
    monkeypatch.setattr(diagnostic, "REPO_ROOT", resolved_repo)
    monkeypatch.setattr(diagnostic, "HARVEST_REPO_ROOT", resolved_repo)
    monkeypatch.setattr(harvester, "REPO_ROOT", resolved_repo)

    input_file = tmp_path / "input.bin"
    input_file.write_bytes(b"immutable")
    upstream_root = tmp_path / "upstream"
    upstream_root.mkdir()
    args = SimpleNamespace(
        checkpoint=input_file,
        feature_cache=input_file,
        gt_cache=input_file,
        upstream_root=upstream_root,
        historical_container=input_file,
        historical_manifest=input_file,
        current_tombstone=input_file,
        wrapper_launcher=input_file,
        wrapper_timeout_seconds=1,
        wrapper_memory_limit_mb=1,
        torch_threads=1,
        torch_interop_threads=1,
        output=tools / "forbidden.json",
    )
    escape = research / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    rejected = (
        tools / "forbidden.json",
        source / "forbidden.json",
        main_checkout / "forbidden.json",
        ssd / "forbidden.json",
        transient / "forbidden.json",
        research / "missing/forbidden.json",
        escape / "forbidden.json",
        research / "forbidden.txt",
    )
    for candidate in rejected:
        args.output = candidate
        with pytest.raises(ValueError, match=r"research tree|parent must already exist|\.json suffix"):
            diagnostic.run_diagnostic(args, exact_inner_argv=["diagnostic"])
    assert input_file.read_bytes() == b"immutable"
    assert not any(candidate.exists() for candidate in rejected)


@pytest.mark.parametrize("target_scope", ["inside", "outside"])
def test_run_diagnostic_rejects_raw_broken_output_symlink_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_scope: str,
) -> None:
    repo = tmp_path / "isolated_worktree"
    research = repo / ".omx/research"
    outside = tmp_path / "outside"
    research.mkdir(parents=True)
    outside.mkdir()
    target_root = research if target_scope == "inside" else outside
    missing_target = target_root / "nonexistent.json"
    output_link = research / "link.json"
    output_link.symlink_to(missing_target)
    resolved_repo = repo.resolve()
    monkeypatch.setattr(diagnostic, "REPO_ROOT", resolved_repo)
    monkeypatch.setattr(diagnostic, "HARVEST_REPO_ROOT", resolved_repo)
    monkeypatch.setattr(harvester, "REPO_ROOT", resolved_repo)
    monkeypatch.setattr(
        diagnostic,
        "_absolute",
        lambda *_args, **_kwargs: pytest.fail("diagnostic work began before raw output rejection"),
    )
    with pytest.raises(ValueError, match="must not be a symlink"):
        diagnostic.run_diagnostic(SimpleNamespace(output=output_link), exact_inner_argv=["diagnostic"])
    assert output_link.is_symlink()
    assert not missing_target.exists()


def test_every_verified_diagnostic_input_is_rehashed_post_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified_inputs = {
        role: object()
        for role in (
            "checkpoint",
            "feature_cache",
            "gt_cache",
            "segnet_model",
            "upstream_modules",
            "upstream_frame_utils",
            "historical_manifest",
            "historical_container",
            "current_tombstone",
            "power_diagram_witness",
            "factorized_features_loader",
            "wrapper_launcher",
        )
    }
    calls: list[tuple[str, object, bool]] = []
    monkeypatch.setattr(
        diagnostic,
        "_assert_unchanged",
        lambda verified, *, role, verify_hash: calls.append((role, verified, verify_hash)),
    )
    diagnostic._rehash_verified_inputs(verified_inputs)
    assert calls == [(role, verified, True) for role, verified in verified_inputs.items()]


@pytest.mark.parametrize(
    ("section", "field", "bad_value"),
    [
        ("authority", "new_run_label", "success"),
        ("authority", "checkpoint_aggregate_label", "reproduced"),
        ("authority", "n600_rerun", True),
        ("authority", "through_r_authority", True),
        ("authority", "receiver_authority", True),
        ("authority", "score_authority", True),
        ("authority", "cleanup_performed", True),
        ("authority", "ssd_write_performed", True),
        ("preserved_checkpoint_state", "sample_count", 196),
        ("reproduction", "frame", 194),
        ("outer_wrapper_custody", "original_governed_run_limits_claimed", True),
    ],
)
def test_receipt_authority_refuses_scope_or_custody_overclaim(section: str, field: str, bad_value: object) -> None:
    receipt = _receipt()
    diagnostic.validate_receipt_authority(receipt)
    corrupted = copy.deepcopy(receipt)
    corrupted[section][field] = bad_value
    with pytest.raises(ValueError, match=r"drift|must not"):
        diagnostic.validate_receipt_authority(corrupted)


def test_main_checks_governed_admission_before_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    sentinel = object()
    monkeypatch.setattr(diagnostic, "_parse_args", lambda _argv: sentinel)
    monkeypatch.setattr(
        diagnostic,
        "assert_governed_admission",
        lambda label: events.append(f"admission:{label}"),
    )
    monkeypatch.setattr(
        diagnostic,
        "run_diagnostic",
        lambda args, *, exact_inner_argv: events.append(f"run:{args is sentinel}:{bool(exact_inner_argv)}"),
    )
    assert diagnostic.main([]) == 0
    assert events == ["admission:diagnose_v10_power_diagram_frame195", "run:True:True"]


def test_diagnostic_source_has_no_cleanup_or_scratch_mutation_calls() -> None:
    source = Path(diagnostic.__file__).read_text(encoding="utf-8")
    for forbidden in ("unlink(", "rmtree(", "open_memmap(", "cleanup_certified_scratch("):
        assert forbidden not in source
