# SPDX-License-Identifier: MIT
"""Focused acceptance tests for the immutable v10 blocked-prefix harvester."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tools.harvest_v10_power_diagram_blocked_prefix as harvester
import tools.v10_power_diagram_blocked_evidence as evidence
from tools.harvest_v10_power_diagram_blocked_prefix import (
    AUTHORITY_LABEL,
    EXPECTED_BLOCKED_REASON,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_FEATURE_CACHE_SHA256,
    EXPECTED_HISTORICAL_CONTAINER_SHA256,
    EXPECTED_MEASUREMENT_TOOL_SHA256,
    EXPECTED_PREFIX_SAMPLES,
    NARROW_VERDICT,
    RECEIPT_STATUS,
    VerifiedFile,
    _rate_comparisons,
    atomic_write_json_no_overwrite,
    fit_prefix_target,
    open_and_validate_feature_cache,
    scan_prefix,
    validate_blocked_checkpoint,
    validate_receipt_authority,
)


def _touch(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path.resolve()


def _fixture_immutable_identity(
    *,
    custody: dict[str, object],
    expected_pairs: int,
    seg_hw: tuple[int, int],
    camera_hwc: tuple[int, int, int],
    n_classes: int,
    head_rank: int,
    ridge: float,
    torch_threads_requested: int,
    torch_threads_effective: int,
    torch_interop_threads_requested: int,
    torch_interop_threads_effective: int,
    implementation: dict[str, object],
) -> dict[str, object]:
    """Build a parser fixture locally; read-only production code cannot construct it."""
    return {
        "custody_derivation": evidence.CUSTODY_DERIVATION,
        "custody": custody,
        "geometry": {
            "expected_pairs": expected_pairs,
            "seg_hw": list(seg_hw),
            "camera_hwc": list(camera_hwc),
            "n_classes": n_classes,
            "head_rank": head_rank,
        },
        "config": {
            "ridge": float(ridge),
            "batch_size": 1,
            "device": "cpu",
            "dtype": "torch.float32",
            "deterministic_algorithms": True,
            "torch_threads_requested": torch_threads_requested,
            "torch_threads_effective": torch_threads_effective,
            "torch_interop_threads_requested": torch_interop_threads_requested,
            "torch_interop_threads_effective": torch_interop_threads_effective,
        },
        "implementation": implementation,
    }


def _fixture_checkpoint_payload(
    state: evidence.ExtractionState, immutable_identity: dict[str, object]
) -> dict[str, object]:
    """Build a test-only historical payload for immutable parser coverage."""
    return {
        "schema": evidence.CHECKPOINT_SCHEMA,
        "status": state.status,
        "next_canonical_frame": state.next_frame,
        "immutable_identity": immutable_identity,
        "statistics": {
            "gram": state.statistics.gram.tolist(),
            "rhs": state.statistics.rhs.tolist(),
            "label_counts": state.statistics.label_counts.tolist(),
            "sample_count": state.statistics.sample_count,
        },
        "adjacency": [list(edge) for edge in sorted(state.adjacency)],
        "positive_control": {
            "power_target_mismatch_count": state.positive_power_mismatches,
            "cpu_torch_forward_mismatch_count": state.positive_forward_mismatches,
        },
        "blocked_reason": state.blocked_reason,
        "updated_utc": "fixture",
    }


def _blocked_fixture(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    upstream = (tmp_path / "upstream").resolve()
    paths = {
        "upstream": upstream,
        "gt_cache": _touch(tmp_path / "gt_n600.npz", b"gt"),
        "segnet_model": _touch(upstream / "models" / "segnet.safetensors", b"model"),
        "upstream_modules": _touch(upstream / "modules.py", b"modules"),
        "upstream_frame_utils": _touch(upstream / "frame_utils.py", b"frames"),
        "measurement_tool": _touch(tmp_path / "measure.py", b"measurement"),
        "power_diagram_witness": _touch(tmp_path / "power.py", b"power"),
        "factorized_features_loader": _touch(tmp_path / "loader.py", b"loader"),
    }
    custody = {
        "gt_cache": {
            "path": str(paths["gt_cache"]),
            "bytes": paths["gt_cache"].stat().st_size,
            "sha256": evidence.PINNED_GT_CACHE_SHA256,
        },
        "segnet_model": {
            "path": str(paths["segnet_model"]),
            "bytes": paths["segnet_model"].stat().st_size,
            "sha256": evidence.PINNED_SEGNET_SHA256,
        },
        "upstream_modules": {
            "path": str(paths["upstream_modules"]),
            "bytes": paths["upstream_modules"].stat().st_size,
            "sha256": evidence.PINNED_MODULES_SHA256,
        },
        "upstream_frame_utils": {
            "path": str(paths["upstream_frame_utils"]),
            "bytes": paths["upstream_frame_utils"].stat().st_size,
            "sha256": evidence.PINNED_FRAME_UTILS_SHA256,
        },
    }
    identity = _fixture_immutable_identity(
        custody=custody,
        expected_pairs=600,
        seg_hw=(384, 512),
        camera_hwc=(874, 1164, 3),
        n_classes=5,
        head_rank=4,
        ridge=1e-6,
        torch_threads_requested=6,
        torch_threads_effective=6,
        torch_interop_threads_requested=18,
        torch_interop_threads_effective=18,
        implementation={
            "tool": {
                "path": str(paths["measurement_tool"]),
                "sha256": EXPECTED_MEASUREMENT_TOOL_SHA256,
            },
            "power_diagram_witness": {
                "path": str(paths["power_diagram_witness"]),
                "sha256": "a" * 64,
            },
            "factorized_features_loader": {
                "path": str(paths["factorized_features_loader"]),
                "sha256": "b" * 64,
            },
        },
    )
    statistics = evidence.StreamingRidgeSufficientStatistics(4, 5)
    statistics.sample_count = EXPECTED_PREFIX_SAMPLES
    statistics.label_counts[:] = [EXPECTED_PREFIX_SAMPLES, 0, 0, 0, 0]
    state = evidence.ExtractionState(
        next_frame=195,
        statistics=statistics,
        adjacency={(0, 1)},
        positive_power_mismatches=1,
        positive_forward_mismatches=0,
        status="blocked",
        blocked_reason=EXPECTED_BLOCKED_REASON,
    )
    return _fixture_checkpoint_payload(state, identity), paths


def _fake_verified(path: Path, *, expected_sha256: str, role: str) -> VerifiedFile:
    del role
    stat = path.stat()
    return VerifiedFile(
        path=path,
        bytes=stat.st_size,
        sha256=expected_sha256,
        device=stat.st_dev,
        inode=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )


def _target() -> object:
    weight = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=np.float64,
    )
    return evidence.affine_scores_to_power_target(
        weight, np.array([0.2, -0.1, -0.1]), adjacency=((0, 1), (0, 2), (1, 2))
    )


def _authority_receipt() -> dict:
    target = _target()
    generator = evidence.compression_accounting(target)
    mismatch_count = 7
    execution_sources = harvester._capture_current_execution_sources()
    return {
        "schema": harvester.SCHEMA,
        "status": RECEIPT_STATUS,
        "current_execution_custody": {
            "source_files": {role: verified.receipt_row() for role, verified in sorted(execution_sources.items())},
            "runtime": harvester._current_runtime_custody(),
        },
        "authority": {
            "evidence_label": AUTHORITY_LABEL,
            "posthoc_prefix_only": True,
            "feature_pullback_only": True,
            "n600_authority": False,
            "through_r_authority": False,
            "rgb_receiver_authority": False,
            "receiver_arithmetic_specified": False,
            "contest_score_authority": False,
            "promotion_eligible": False,
        },
        "positive_control_exposure": {
            "label": "MEASURED_PRESERVED_BLOCKED_STATE",
            "observed_first_frame": 0,
            "observed_last_frame": 195,
            "observed_frame_count": 196,
            "fit_excluded_frame": 195,
            "blocked_reason": EXPECTED_BLOCKED_REASON,
            "power_target_mismatch_count": 1,
            "cpu_torch_forward_mismatch_count": 0,
        },
        "prefix_measurement": {
            "label": AUTHORITY_LABEL,
            "scan": {
                "first_frame": 0,
                "last_frame": 194,
                "frame_count": 195,
                "sample_count": EXPECTED_PREFIX_SAMPLES,
                "mismatch_count": mismatch_count,
            },
            "streaming_statistics": {
                "sample_count": EXPECTED_PREFIX_SAMPLES,
            },
            "fitted_feature_pullback_mismatch": {
                "label": "MEASURED_ADVISORY_POSTHOC_PREFIX_FEATURE_PULLBACK",
                "numerator": mismatch_count,
                "denominator": EXPECTED_PREFIX_SAMPLES,
                "fraction": mismatch_count / EXPECTED_PREFIX_SAMPLES,
            },
        },
        "generator": generator,
        "rate_comparison": _rate_comparisons(generator["brotli_quality11"]["bytes"]),
        "verdict": {
            "narrow_verdict": NARROW_VERDICT,
            "family_open": True,
            "paradigm_open": True,
            "equivalent_rate_win_claimed": False,
            "factor_6_complete": False,
            "score_gap_closed": False,
            "score_pointer_move_authorized": False,
            "equation_registry_registration_authorized": False,
            "cleanup_performed": False,
        },
    }


def test_preserved_hashes_and_prefix_contract_are_exact() -> None:
    assert EXPECTED_MEASUREMENT_TOOL_SHA256 == ("be094a1540a94bf51aa98706b6d4515eec150bb569380f69b308ed66556cd7c9")
    assert EXPECTED_CHECKPOINT_SHA256 == ("58656d231af5c63b12b3594d8eeeeccf0b2d0f25c09154ef3ef6da759e1fce4b")
    assert EXPECTED_FEATURE_CACHE_SHA256 == ("59e96781aa1bac153bc8bb277cecdbd4b4e98fdfd41f50aa2294537b90390944")
    assert EXPECTED_PREFIX_SAMPLES == 195 * 384 * 512
    assert RECEIPT_STATUS == "BLOCKED_WITH_POSTHOC_PREFIX_DIAGNOSTIC"
    assert AUTHORITY_LABEL == "ADVISORY_POSTHOC_PREFIX_0_194_OF_600"
    assert NARROW_VERDICT == ("FROZEN_HEAD_FLOAT32_POWER_TARGET_POSITIVE_CONTROL_BLOCKED_AT_FRAME_195")


def test_regular_fixture_hash_drift_is_refused_without_large_artifacts(tmp_path: Path) -> None:
    artifact = _touch(tmp_path / "fixture.bin", b"small deterministic fixture")
    actual = harvester.sha256_file(artifact)
    verified = harvester._verify_file(artifact, expected_sha256=actual, role="fixture")
    assert verified.sha256 == actual
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        harvester._verify_file(artifact, expected_sha256="0" * 64, role="fixture")


def test_post_use_execution_source_mutation_is_caught_by_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harvester_source = _touch(tmp_path / "harvester.py", b"harvest1")
    helper_source = _touch(tmp_path / "helper.py", b"helper01")
    monkeypatch.setattr(
        harvester,
        "_current_execution_source_paths",
        lambda: {
            "harvester": harvester_source,
            "blocked_evidence_helper": helper_source,
        },
    )
    captured = harvester._capture_current_execution_sources()
    helper_stat = helper_source.stat()
    helper_source.write_bytes(b"helper02")
    os.utime(helper_source, ns=(helper_stat.st_atime_ns, captured["blocked_evidence_helper"].mtime_ns))
    with pytest.raises(RuntimeError, match="content hash changed"):
        harvester._assert_current_execution_sources_unchanged(captured)


def test_historical_container_manifest_and_current_tombstone_lineage_is_exact() -> None:
    container = (
        harvester.REPO_ROOT / ".omx/research/evidence/measure_v10_power_diagram_generator_byteclose_"
        "be094a1540a94bf51aa98706b6d4515eec150bb569380f69b308ed66556cd7c9.source.gz"
    ).resolve()
    manifest_path = (
        harvester.REPO_ROOT / ".omx/research/evidence/measure_v10_power_diagram_generator_byteclose_"
        "be094a1540a94bf51aa98706b6d4515eec150bb569380f69b308ed66556cd7c9.manifest.json"
    ).resolve()
    tombstone = (harvester.REPO_ROOT / "tools/measure_v10_power_diagram_generator_byteclose.py").resolve()
    manifest, verified = harvester.validate_historical_lineage(
        historical_container=container,
        historical_manifest=manifest_path,
        current_tombstone=tombstone,
    )
    compressed = container.read_bytes()
    original = gzip.decompress(compressed)
    assert not container.with_name(container.name.replace(".source.gz", ".source.txt")).exists()
    assert manifest["container_sha256"] == EXPECTED_HISTORICAL_CONTAINER_SHA256
    assert manifest["container_bytes"] == container.stat().st_size == 16_187
    assert hashlib.sha256(compressed).hexdigest() == EXPECTED_HISTORICAL_CONTAINER_SHA256
    assert manifest["decompressed_bytes"] == len(original) == 62_907
    assert manifest["decompressed_sha256"] == hashlib.sha256(original).hexdigest()
    assert manifest["decompressed_sha256"] == EXPECTED_MEASUREMENT_TOOL_SHA256
    assert compressed[:10].hex() == "1f8b0800000000000203"
    assert verified["historical_container"].sha256 == EXPECTED_HISTORICAL_CONTAINER_SHA256
    assert verified["current_tombstone"].sha256 == harvester.EXPECTED_TOMBSTONE_SHA256


def test_tombstone_masquerade_at_noncanonical_path_is_refused(tmp_path: Path) -> None:
    container = (
        harvester.REPO_ROOT / ".omx/research/evidence/measure_v10_power_diagram_generator_byteclose_"
        "be094a1540a94bf51aa98706b6d4515eec150bb569380f69b308ed66556cd7c9.source.gz"
    ).resolve()
    manifest_path = container.with_name(container.name.replace(".source.gz", ".manifest.json"))
    masquerade = tmp_path / "measure_v10_power_diagram_generator_byteclose.py"
    masquerade.write_bytes(
        (harvester.REPO_ROOT / "tools/measure_v10_power_diagram_generator_byteclose.py").read_bytes()
    )
    with pytest.raises(ValueError, match="canonical live tool path"):
        harvester.validate_historical_lineage(
            historical_container=container,
            historical_manifest=manifest_path,
            current_tombstone=masquerade.resolve(),
        )


def test_blocked_checkpoint_accepts_exact_state_without_large_artifact_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, paths = _blocked_fixture(tmp_path)
    monkeypatch.setattr(harvester, "_verify_file", _fake_verified)
    state, verified = validate_blocked_checkpoint(
        payload,
        upstream_root=paths["upstream"],
        gt_cache=paths["gt_cache"],
        historical_manifest={"historical_checkpoint_path": str(paths["measurement_tool"])},
        lineage_files={},
    )
    assert state.status == "blocked"
    assert state.next_frame == 195
    assert state.statistics.sample_count == EXPECTED_PREFIX_SAMPLES
    assert state.positive_power_mismatches == 1
    assert state.positive_forward_mismatches == 0
    assert payload["immutable_identity"]["implementation"]["tool"]["sha256"] == (EXPECTED_MEASUREMENT_TOOL_SHA256)


@pytest.mark.parametrize(
    ("drift", "match"),
    [
        ("tool_hash", "implementation hash drift"),
        ("status", "blocked status"),
        ("reason", "blocked reason drift"),
        ("next_frame", "sample counts disagree"),
        ("geometry", "immutable geometry drift"),
        ("power_count", "positive-control mismatch counts drift"),
    ],
)
def test_checkpoint_refuses_hash_state_and_geometry_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    match: str,
) -> None:
    payload, paths = _blocked_fixture(tmp_path)
    corrupted = copy.deepcopy(payload)
    if drift == "tool_hash":
        corrupted["immutable_identity"]["implementation"]["tool"]["sha256"] = "0" * 64
    elif drift == "status":
        corrupted["status"] = "extracting"
        corrupted["blocked_reason"] = None
        corrupted["positive_control"]["power_target_mismatch_count"] = 0
    elif drift == "reason":
        corrupted["blocked_reason"] = "different frame"
    elif drift == "next_frame":
        corrupted["next_canonical_frame"] = 194
    elif drift == "geometry":
        corrupted["immutable_identity"]["geometry"]["seg_hw"] = [383, 512]
    elif drift == "power_count":
        corrupted["positive_control"]["power_target_mismatch_count"] = 2
    monkeypatch.setattr(harvester, "_verify_file", _fake_verified)
    with pytest.raises(ValueError, match=match):
        validate_blocked_checkpoint(
            corrupted,
            upstream_root=paths["upstream"],
            gt_cache=paths["gt_cache"],
            historical_manifest={"historical_checkpoint_path": str(paths["measurement_tool"])},
            lineage_files={},
        )


def test_prefix_scan_uses_exact_order_and_denominator_without_touching_frame_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260718)
    features = np.zeros((600, 2, 3, 2), dtype=np.float32)
    features[:3] = rng.normal(size=(3, 2, 3, 2))
    labels = np.zeros((600, 2, 3), dtype=np.int64)
    target = _target()
    for frame in range(3):
        labels[frame] = harvester.power_assign(features[frame].reshape(-1, 2), target).reshape(2, 3)
    labels[3] = 99
    calls: list[float] = []
    real_assign = harvester.power_assign

    def recording_assign(points: np.ndarray, fitted: object) -> np.ndarray:
        calls.append(float(points[0, 0]))
        return real_assign(points, fitted)

    monkeypatch.setattr(harvester, "power_assign", recording_assign)
    scan = scan_prefix(
        features,
        labels,
        target,
        prefix_frames=3,
        seg_hw=(2, 3),
        head_rank=2,
        n_classes=3,
    )
    assert calls == [float(features[index, 0, 0, 0]) for index in range(3)]
    assert scan == {
        "first_frame": 0,
        "last_frame": 2,
        "frame_count": 3,
        "sample_count": 18,
        "mismatch_count": 0,
    }


def test_prefix_fit_matches_dense_ridge_fit_and_assignments() -> None:
    rng = np.random.default_rng(195)
    features = rng.normal(size=(83, 4))
    labels = np.argmax(features @ rng.normal(size=(4, 5)) + rng.normal(size=5), axis=1)
    state = evidence.ExtractionState(
        next_frame=1,
        statistics=evidence.StreamingRidgeSufficientStatistics(4, 5),
        adjacency={(i, j) for i in range(5) for j in range(i + 1, 5)},
    )
    state.statistics.update(features, labels)
    target = fit_prefix_target(state)

    design = np.concatenate((features, np.ones((features.shape[0], 1))), axis=1)
    desired = np.eye(5)[labels] - 1.0 / 5.0
    coefficients = np.linalg.solve(design.T @ design + 1e-6 * np.eye(5), design.T @ desired)
    dense_target = evidence.affine_scores_to_power_target(
        coefficients[:-1].T,
        coefficients[-1],
        adjacency=tuple(sorted(state.adjacency)),
    )
    np.testing.assert_array_equal(
        harvester.power_assign(features, target),
        harvester.power_assign(features, dense_target),
    )
    np.testing.assert_array_equal(target.sites, dense_target.sites)
    np.testing.assert_array_equal(target.weights, dense_target.weights)


def test_feature_cache_refuses_state_and_byte_geometry_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_path = _touch(tmp_path / evidence.FEATURE_CACHE_NAME, b"header")

    class FakeMemmap:
        shape = (600, 384, 512, 4)
        dtype = np.dtype("<f4")
        offset = 128
        nbytes = 600 * 384 * 512 * 4 * 4

    monkeypatch.setattr(harvester.np, "memmap", FakeMemmap)
    monkeypatch.setattr(harvester.np, "load", lambda *_args, **_kwargs: FakeMemmap())
    with pytest.raises(ValueError, match="byte geometry drift"):
        open_and_validate_feature_cache(cache_path)
    FakeMemmap.shape = (599, 384, 512, 4)
    with pytest.raises(ValueError, match="geometry drift"):
        open_and_validate_feature_cache(cache_path)


def test_pdw1_parseback_and_rate_rows_are_strictly_non_equivalent() -> None:
    target = _target()
    accounting = evidence.compression_accounting(target)
    raw = bytes.fromhex(accounting["pdw1_hex"])
    assert evidence.encode_pdw1(evidence.decode_pdw1(raw)) == raw
    assert accounting["strict_parseback_byte_identical"] is True
    assert "order0_arithmetic_lower_bound" not in accounting
    ideal = accounting["order0_ideal_entropy_estimate"]
    assert ideal["label"] == "DERIVED_OPTIMISTIC_ROUNDED_UP_IDEAL_ENTROPY_BYTES"
    assert ideal["assumptions"] == "empirical PMF free; no model/header/termination overhead"
    assert "lower_bound" not in ideal
    assert "estimated_bytes_ceiling" not in ideal
    comparisons = _rate_comparisons(accounting["brotli_quality11"]["bytes"])
    assert {row["reference_bytes"] for row in comparisons.values()} == {
        228_764,
        235_974,
        225_272,
    }
    assert all(
        row["label"] == "NON_EQUIVALENT_TARGET_PAYLOAD_VS_FULL_REALIZATION_REFERENCE"
        and row["equivalent_rate_comparison"] is False
        for row in comparisons.values()
    )


def test_atomic_output_refuses_overwrite_and_preserves_existing_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "worktree"
    research = repo / ".omx/research"
    research.mkdir(parents=True)
    monkeypatch.setattr(harvester, "REPO_ROOT", repo.resolve())
    output = research / "receipt.json"
    atomic_write_json_no_overwrite(output, {"first": True})
    first_bytes = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        atomic_write_json_no_overwrite(output, {"second": True})
    assert output.read_bytes() == first_bytes
    assert json.loads(first_bytes) == {"first": True}


def test_output_is_confined_to_existing_resolved_repo_research_json(
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
    monkeypatch.setattr(harvester, "REPO_ROOT", repo.resolve())

    accepted = research / "new_receipt.json"
    assert harvester.validate_durable_output(accepted.resolve(strict=False)) == accepted.resolve(strict=False)

    rejected = (
        tools / "receipt.json",
        source / "receipt.json",
        main_checkout / "receipt.json",
        ssd / "receipt.json",
        transient / "receipt.json",
        research / "wrong_suffix.txt",
    )
    for candidate in rejected:
        with pytest.raises(ValueError, match=r"research tree|\.json suffix"):
            harvester.validate_durable_output(candidate.resolve(strict=False))

    missing_parent = research / "missing/receipt.json"
    with pytest.raises(ValueError, match="parent must already exist"):
        atomic_write_json_no_overwrite(missing_parent, {"forbidden": True})
    assert not missing_parent.parent.exists()

    escape = research / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="research tree"):
        harvester.validate_durable_output(escape / "receipt.json")
    assert not (outside / "receipt.json").exists()


@pytest.mark.parametrize("target_scope", ["inside", "outside"])
def test_run_harvest_rejects_raw_broken_output_symlink_before_work(
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
    monkeypatch.setattr(harvester, "REPO_ROOT", repo.resolve())
    monkeypatch.setattr(
        harvester,
        "_capture_current_execution_sources",
        lambda: pytest.fail("harvest work began before raw output rejection"),
    )
    with pytest.raises(ValueError, match="must not be a symlink"):
        harvester.run_harvest(SimpleNamespace(output=output_link), exact_argv=["harvester"])
    assert output_link.is_symlink()
    assert not missing_target.exists()


@pytest.mark.parametrize(
    ("section", "field", "bad_value"),
    [
        ("authority", "n600_authority", True),
        ("authority", "through_r_authority", True),
        ("authority", "rgb_receiver_authority", True),
        ("authority", "receiver_arithmetic_specified", True),
        ("authority", "contest_score_authority", True),
        ("authority", "promotion_eligible", True),
        ("verdict", "equivalent_rate_win_claimed", True),
        ("verdict", "factor_6_complete", True),
        ("verdict", "score_gap_closed", True),
        ("verdict", "score_pointer_move_authorized", True),
        ("verdict", "equation_registry_registration_authorized", True),
    ],
)
def test_receipt_authority_validation_refuses_overclaim(section: str, field: str, bad_value: object) -> None:
    receipt = _authority_receipt()
    validate_receipt_authority(receipt)
    receipt[section][field] = bad_value
    with pytest.raises(ValueError):
        validate_receipt_authority(receipt)


@pytest.mark.parametrize("missing_key", [None, "source_files", "runtime"])
def test_receipt_validation_requires_current_execution_custody(missing_key: str | None) -> None:
    receipt = _authority_receipt()
    if missing_key is None:
        receipt.pop("current_execution_custody")
    else:
        receipt["current_execution_custody"].pop(missing_key)
    with pytest.raises(ValueError, match="current execution custody"):
        validate_receipt_authority(receipt)


@pytest.mark.parametrize(
    ("path", "bad_value", "match"),
    [
        (
            ("source_files", "harvester", "sha256"),
            "0" * 64,
            "harvester source custody drift",
        ),
        (
            ("source_files", "blocked_evidence_helper", "path"),
            "/forged/helper.py",
            "blocked_evidence_helper source path drift",
        ),
        (
            ("runtime", "numpy", "version"),
            "forged",
            "runtime custody drift",
        ),
        (
            ("runtime", "platform", "machine"),
            "forged",
            "runtime custody drift",
        ),
    ],
)
def test_receipt_validation_refuses_forged_execution_source_and_runtime_custody(
    path: tuple[str, ...],
    bad_value: object,
    match: str,
) -> None:
    receipt = _authority_receipt()
    cursor = receipt["current_execution_custody"]
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = bad_value
    with pytest.raises(ValueError, match=match):
        validate_receipt_authority(receipt)


def test_receipt_validation_refuses_blocked_diagnostic_custody_and_arithmetic_drift() -> None:
    mutations = [
        (("status",), "success"),
        (("authority", "evidence_label"), "n600"),
        (("positive_control_exposure", "label"), "MEASURED_SUCCESS"),
        (("positive_control_exposure", "observed_first_frame"), 1),
        (("positive_control_exposure", "observed_last_frame"), 194),
        (("positive_control_exposure", "observed_frame_count"), 195),
        (("positive_control_exposure", "fit_excluded_frame"), 194),
        (("positive_control_exposure", "blocked_reason"), "different blocker"),
        (("positive_control_exposure", "power_target_mismatch_count"), 0),
        (("positive_control_exposure", "cpu_torch_forward_mismatch_count"), 1),
        (("prefix_measurement", "streaming_statistics", "sample_count"), 1),
        (("prefix_measurement", "fitted_feature_pullback_mismatch", "numerator"), 6),
        (("prefix_measurement", "fitted_feature_pullback_mismatch", "fraction"), 0.0),
        (("verdict", "cleanup_performed"), True),
        (
            (
                "rate_comparison",
                "optimistic_shared_edge_mdl_contour",
                "posthoc_prefix_generator_brotli_minus_reference_bytes",
            ),
            0,
        ),
    ]
    for path, bad_value in mutations:
        receipt = _authority_receipt()
        validate_receipt_authority(receipt)
        cursor = receipt
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = bad_value
        with pytest.raises(ValueError, match=r"drift|must be|disagrees"):
            validate_receipt_authority(receipt)


def test_parser_requires_all_explicit_paths() -> None:
    with pytest.raises(SystemExit):
        harvester._parse_args([])
    args = harvester._parse_args(
        [
            "--checkpoint",
            "/checkpoint",
            "--feature-cache",
            "/cache",
            "--gt-cache",
            "/gt",
            "--upstream-root",
            "/upstream",
            "--historical-container",
            "/container",
            "--historical-manifest",
            "/manifest",
            "--current-tombstone",
            "/tombstone",
            "--output",
            "/receipt",
        ]
    )
    assert args.checkpoint == Path("/checkpoint")
    assert args.feature_cache == Path("/cache")
    assert args.gt_cache == Path("/gt")
    assert args.upstream_root == Path("/upstream")
    assert args.historical_container == Path("/container")
    assert args.historical_manifest == Path("/manifest")
    assert args.current_tombstone == Path("/tombstone")
    assert args.output == Path("/receipt")
