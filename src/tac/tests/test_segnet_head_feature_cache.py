from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tac.governed_profile_admission as governed_admission_module
import tac.witness_control.segnet_head_feature_cache as cache_module
import tools.extract_segnet_head_features_n600 as extractor_module
from tac.admission_guard import BYPASS_OVERRIDE_ENV, mark_admitted_env
from tac.governed_profile_admission import GovernedAdmissionError, attest_safe_run_parent
from tac.witness_control.segnet_head_feature_cache import (
    CERTIFICATION_NAME,
    COMPLETION_CONTROL_NAME,
    LIVE_LOGITS_NAME,
    PROGRESS_NAME,
    STAGING_SCRATCH_NAME,
    FeatureCacheError,
    SegnetHeadFeatureCache,
    build_immutable_identity,
    open_gt_f1_stored_memmap,
    validate_feature_cache,
    validate_live_logit_positive_control,
)
from tools.extract_segnet_head_features_n600 import (
    ExtractionError,
    _assert_real_governed_admission,
    _attest_exact_argv,
    _build_cache_identity,
    _canonical_rebuild_argv,
    _emit_extraction_receipt,
    _parse_args,
    _require_equal_source_snapshots,
    _require_exact_module_file,
    _require_local_test_scope,
    _source_bindings_with_admission,
    storage_preflight,
)
from tools.profile_v10_uint8_lattice_n600 import (
    EXPECTED_PAIRS,
    ProfilerError,
    _feature_binding,
    _validate_feature_for_request,
)


def _source(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _identity(*, config_value: int = 1, source_payload: bytes = b"source") -> dict[str, object]:
    return build_immutable_identity(
        source_files={"synthetic_source": _source("/synthetic/source", source_payload)},
        config={"batch_size": 1, "test_value": config_value},
        frame_count=2,
        live_slice_shape=(3, 2, 2),
        quotient_slice_shape=(2, 2, 2),
    )


def _preflight(root: Path) -> dict[str, object]:
    resolved = root.resolve()
    return {
        "waterfall_order": ["/Volumes/VertigoDataTier/pact", "/Volumes/APDataStore/pact"],
        "existing_approved_roots": [],
        "selected_root": str(resolved),
        "filesystem_anchor": str(resolved.parent),
        "free_bytes_before": 1_000_000,
        "required_free_bytes": 1,
        "allow_local_output_for_tests": True,
        "PASS": True,
    }


def _test_atomic_json(path: Path, payload: dict[str, object]) -> None:
    """Authorize the exact prior bytes for tests that deliberately rewrite custody."""

    expected_prior = (path.read_bytes(),) if path.exists() else ()
    cache_module.atomic_json(path, payload, expected_prior_payloads=expected_prior)


def _flat_tree_snapshot(root: Path) -> dict[str, tuple[int, int, int, int, bytes | None]]:
    result: dict[str, tuple[int, int, int, int, bytes | None]] = {}
    for path in sorted(root.iterdir(), key=lambda candidate: candidate.name):
        metadata = path.lstat()
        payload = path.read_bytes() if stat.S_ISREG(metadata.st_mode) else None
        result[path.name] = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_size,
            payload,
        )
    return result


def _populate(root: Path, identity: dict[str, object]) -> SegnetHeadFeatureCache:
    cache = SegnetHeadFeatureCache.create(
        root,
        identity=identity,
        rebuild_command=("python", "synthetic.py"),
        storage_preflight=_preflight(root),
    )
    for frame in range(2):
        live = np.arange(12, dtype=np.float32).reshape(3, 2, 2) + frame
        quotient = np.arange(8, dtype=np.float32).reshape(2, 2, 2) - frame
        cache.commit_frame(frame, live, quotient)
    fresh = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    cache.mark_complete(positive_frame=0, fresh_live_logits=fresh)
    return cache


def test_frame195_style_algebraic_disagreement_is_diagnostic_not_blocker() -> None:
    # Direct forward is bit-identical and selects class 0 everywhere.  The
    # algebraic/reduction-order view selects class 1 at one synthetic pixel.
    cached = np.array([[[1.0]], [[1.0 - np.float32(2**-22)]]], dtype=np.float32)
    algebraic = np.array([[1]], dtype=np.int64)
    result = validate_live_logit_positive_control(
        cached,
        cached.copy(),
        algebraic_argmax=algebraic,
    )
    assert result.bitwise_live_logits_equal is True
    assert result.algebraic_argmax_disagreements == 1
    assert result.diagnostic_only_algebraic_disagreement is True


def test_live_positive_control_is_bitwise_not_close() -> None:
    cached = np.array([[[1.0]], [[0.0]]], dtype=np.float32)
    fresh = cached.copy()
    fresh[0, 0, 0] = np.nextafter(fresh[0, 0, 0], np.float32(2.0))
    with pytest.raises(FeatureCacheError, match="differs in 1"):
        validate_live_logit_positive_control(cached, fresh)


def test_resume_is_byte_identical_to_fresh_and_refuses_identity_drift(tmp_path: Path) -> None:
    identity = _identity()
    first = _populate(tmp_path / "first", identity)
    second = _populate(tmp_path / "second", identity)
    assert (first.root / "live_logits.f32.npy").read_bytes() == (second.root / "live_logits.f32.npy").read_bytes()
    assert (first.root / "quotient_features.f32.npy").read_bytes() == (
        second.root / "quotient_features.f32.npy"
    ).read_bytes()
    resumed = SegnetHeadFeatureCache.resume(first.root, expected_identity=identity)
    assert resumed.next_frame == 2
    assert resumed.progress["status"] == "complete"
    with pytest.raises(FeatureCacheError, match="immutable identity"):
        SegnetHeadFeatureCache.resume(
            first.root,
            expected_identity=_identity(config_value=2),
        )
    with pytest.raises(FeatureCacheError, match="immutable identity"):
        SegnetHeadFeatureCache.resume(
            first.root,
            expected_identity=_identity(source_payload=b"changed"),
        )


def test_resume_refuses_changed_committed_slice_bytes(tmp_path: Path) -> None:
    cache = _populate(tmp_path / "cache", _identity())
    mapped = np.load(cache.root / "live_logits.f32.npy", mmap_mode="r+")
    mapped[0, 0, 0, 0] += np.float32(1.0)
    mapped.flush()
    del mapped
    with pytest.raises(FeatureCacheError, match="slice 0 hash mismatch"):
        validate_feature_cache(cache.root)


def test_terminal_control_rejects_refreshed_noncontrol_frame_chain(tmp_path: Path) -> None:
    """Refreshing local hashes cannot silently move a completed cache root.

    Frame 0 is the independent fresh-forward control in this fixture.  This
    regression changes frame 1, refreshes its local slice hash, recomputes its
    chain row, and moves the progress-chain head.  The separately structured
    completion control remains bound to the original terminal commitment and
    therefore rejects the coordinated local rewrite.
    """

    cache = _populate(tmp_path / "cache", _identity())
    completion_path = cache.root / COMPLETION_CONTROL_NAME
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    original_terminal = completion["terminal_frame_commitment_sha256"]

    mapped = np.load(cache.root / LIVE_LOGITS_NAME, mmap_mode="r+")
    mapped[1, 0, 0, 0] += np.float32(1.0)
    mapped.flush()
    replacement_live_hash = cache_module.sha256_float32_slice(mapped[1])
    del mapped

    progress_path = cache.root / PROGRESS_NAME
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    row = progress["committed_frames"][1]
    row["live_logits_sha256"] = replacement_live_hash
    row["frame_commitment_sha256"] = cache_module._frame_commitment_sha256(
        identity_sha256=progress["identity_sha256"],
        previous_frame_commitment_sha256=row["previous_frame_commitment_sha256"],
        frame=row["frame"],
        live_logits_sha256=row["live_logits_sha256"],
        quotient_features_sha256=row["quotient_features_sha256"],
        diagnostics=row["diagnostics"],
    )
    progress["frame_chain_head_sha256"] = row["frame_commitment_sha256"]
    assert progress["frame_chain_head_sha256"] != original_terminal
    _test_atomic_json(progress_path, progress)

    with pytest.raises(FeatureCacheError, match="terminal chain"):
        validate_feature_cache(cache.root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("previous_frame_commitment_sha256", "0" * 64, "predecessor commitment mismatch"),
        ("frame_chain_head_sha256", "0" * 64, "frame-chain head mismatch"),
    ],
)
def test_resume_refuses_broken_frame_chain_links(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    cache = _populate(tmp_path / "cache", _identity())
    progress_path = cache.root / PROGRESS_NAME
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if field == "previous_frame_commitment_sha256":
        progress["committed_frames"][1][field] = value
    else:
        progress[field] = value
    _test_atomic_json(progress_path, progress)
    with pytest.raises(FeatureCacheError, match=message):
        SegnetHeadFeatureCache.resume(cache.root, expected_identity=_identity())


def test_completion_control_write_is_crash_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cache"
    cache = SegnetHeadFeatureCache.create(
        root,
        identity=_identity(),
        rebuild_command=("python", "synthetic.py"),
        storage_preflight=_preflight(root),
    )
    for frame in range(2):
        cache.commit_frame(
            frame,
            np.arange(12, dtype=np.float32).reshape(3, 2, 2) + frame,
            np.arange(8, dtype=np.float32).reshape(2, 2, 2) - frame,
        )

    real_atomic_json = cache_module.atomic_json

    def interrupt_progress_flip(path: Path, payload: object, **kwargs: object) -> None:
        if path.name == PROGRESS_NAME and isinstance(payload, dict) and payload.get("status") == "complete":
            raise OSError("synthetic interruption after terminal control")
        real_atomic_json(path, payload, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cache_module, "atomic_json", interrupt_progress_flip)
    fresh = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    with pytest.raises(OSError, match="synthetic interruption"):
        cache.mark_complete(positive_frame=0, fresh_live_logits=fresh)

    prepared = validate_feature_cache(root)
    assert prepared.progress["status"] == "ready_for_completion_validation"
    assert (root / COMPLETION_CONTROL_NAME).is_file()

    monkeypatch.setattr(cache_module, "atomic_json", real_atomic_json)
    resumed = SegnetHeadFeatureCache.resume(root, expected_identity=_identity())
    resumed.mark_complete(positive_frame=0, fresh_live_logits=fresh)
    assert validate_feature_cache(root, require_complete=True).complete is True


def _attested_child_argv() -> list[str]:
    return [sys.executable, str(Path(extractor_module.__file__).resolve()), "--synthetic-child"]


def _install_parent_argv(
    monkeypatch: pytest.MonkeyPatch,
    child_argv: list[str],
    *,
    rss: int = 512,
    timeout: int = 60,
    extra_outer: tuple[str, ...] = (),
    parent_python: str | None = None,
    parent_pid: int = 4242,
) -> None:
    parent = [
        sys.executable if parent_python is None else parent_python,
        str(extractor_module.REPO_ROOT / "tools/safe_run.py"),
        "--rss-mb",
        str(rss),
        "--timeout",
        str(timeout),
        *extra_outer,
        "--",
        *child_argv,
    ]
    monkeypatch.setattr(governed_admission_module.os, "getppid", lambda: parent_pid)
    monkeypatch.setattr(
        governed_admission_module,
        "_read_process_argv",
        lambda pid: parent if pid == parent_pid else [],
    )


def test_production_extractor_refuses_unarmed_or_bypass_only_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _attested_child_argv()
    _install_parent_argv(monkeypatch, child)
    with pytest.raises(ExtractionError, match="direct safe_run parent custody"):
        _assert_real_governed_admission(
            allow_local_output_for_tests=False,
            exact_child_argv=child,
            rss_cap_mb=512,
            timeout_seconds=60,
            env={},
        )
    with pytest.raises(ExtractionError, match="direct safe_run parent custody"):
        _assert_real_governed_admission(
            allow_local_output_for_tests=False,
            exact_child_argv=child,
            rss_cap_mb=512,
            timeout_seconds=60,
            env={BYPASS_OVERRIDE_ENV: "reviewed raw exception"},
        )


def test_raw_governed_marker_is_insufficient_without_direct_safe_run_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _attested_child_argv()
    monkeypatch.setattr(governed_admission_module.os, "getppid", lambda: 4242)
    monkeypatch.setattr(
        governed_admission_module,
        "_read_process_argv",
        lambda _pid: [sys.executable, *child[1:]],
    )
    governed_env = mark_admitted_env({})
    with pytest.raises(ExtractionError, match="direct safe_run parent custody"):
        _assert_real_governed_admission(
            allow_local_output_for_tests=False,
            exact_child_argv=child,
            rss_cap_mb=512,
            timeout_seconds=60,
            env=governed_env,
        )
    with pytest.raises(ExtractionError, match="direct safe_run parent custody"):
        _assert_real_governed_admission(
            allow_local_output_for_tests=True,
            exact_child_argv=child,
            rss_cap_mb=512,
            timeout_seconds=60,
            env=governed_env,
        )


def test_direct_safe_run_parent_attestation_binds_child_caps_and_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _attested_child_argv()
    _install_parent_argv(monkeypatch, child)
    attestation = attest_safe_run_parent(
        exact_child_argv=child,
        rss_cap_mb=512,
        timeout_seconds=60,
        repo_root=extractor_module.REPO_ROOT,
        env=mark_admitted_env({}),
    )
    assert attestation["child_exact_argv"] == child
    assert attestation["parent_python_executable"] == str(Path(sys.executable).resolve())
    assert attestation["outer_resource_caps"] == {"rss_cap_mb": 512, "timeout_seconds": 60.0}
    assert attestation["completed_safe_run_status_receipt"] is None
    assert set(attestation["source_custody"]) == {
        "governed_profile_admission",
        "safe_run",
        "admission_guard",
    }


@pytest.mark.parametrize(
    ("extra_outer", "rss", "timeout", "message"),
    [
        (("--skip-admission-gate",), 512, 60, "escape is forbidden"),
        (("--admission-override-rationale", "operator said yes"), 512, 60, "escape is forbidden"),
        ((), 513, 60, "RSS cap differs"),
        ((), 512, 61, "timeout differs"),
    ],
)
def test_safe_run_parent_refuses_skip_override_and_mismatched_outer_caps(
    monkeypatch: pytest.MonkeyPatch,
    extra_outer: tuple[str, ...],
    rss: int,
    timeout: int,
    message: str,
) -> None:
    child = _attested_child_argv()
    _install_parent_argv(monkeypatch, child, rss=rss, timeout=timeout, extra_outer=extra_outer)
    with pytest.raises(GovernedAdmissionError, match=message):
        attest_safe_run_parent(
            exact_child_argv=child,
            rss_cap_mb=512,
            timeout_seconds=60,
            repo_root=extractor_module.REPO_ROOT,
            env=mark_admitted_env({}),
        )


def test_safe_run_parent_refuses_a_different_python_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _attested_child_argv()
    foreign_python = tmp_path / "foreign-python"
    foreign_python.write_bytes(b"not the child runtime")
    _install_parent_argv(monkeypatch, child, parent_python=str(foreign_python))
    with pytest.raises(GovernedAdmissionError, match="same exact Python runtime"):
        attest_safe_run_parent(
            exact_child_argv=child,
            rss_cap_mb=512,
            timeout_seconds=60,
            repo_root=extractor_module.REPO_ROOT,
            env=mark_admitted_env({}),
        )


def test_extractor_manifest_identity_is_stable_across_fresh_and_resume_invocations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh_child = _attested_child_argv()
    _install_parent_argv(monkeypatch, fresh_child, parent_pid=111)
    fresh_attestation = attest_safe_run_parent(
        exact_child_argv=fresh_child,
        rss_cap_mb=512,
        timeout_seconds=60,
        repo_root=extractor_module.REPO_ROOT,
        env=mark_admitted_env({}),
    )
    resume_child = [*fresh_child, "--resume"]
    _install_parent_argv(monkeypatch, resume_child, parent_pid=999)
    resume_attestation = attest_safe_run_parent(
        exact_child_argv=resume_child,
        rss_cap_mb=512,
        timeout_seconds=60,
        repo_root=extractor_module.REPO_ROOT,
        env=mark_admitted_env({}),
    )
    fresh_rebuild = _canonical_rebuild_argv(fresh_child, resume=False)
    resume_rebuild = _canonical_rebuild_argv(resume_child, resume=True)
    assert fresh_rebuild == resume_rebuild

    base_sources = {"synthetic_source": cache_module.source_file_row(Path(extractor_module.__file__))}
    fresh_sources = _source_bindings_with_admission(base_sources, fresh_attestation)
    resume_sources = _source_bindings_with_admission(base_sources, resume_attestation)
    assert fresh_sources == resume_sources
    args = SimpleNamespace(
        batch_size=1,
        chunk_frames=1,
        torch_threads=1,
        torch_interop_threads=1,
        rss_cap_mb=512,
        timeout_seconds=60,
    )
    fresh_identity = _build_cache_identity(
        args=args,
        source_bindings=fresh_sources,
        rebuild_argv=fresh_rebuild,
        camera_hw=(2, 2),
        seg_hw=(2, 2),
        frame_count=2,
    )
    resume_identity = _build_cache_identity(
        args=args,
        source_bindings=resume_sources,
        rebuild_argv=resume_rebuild,
        camera_hw=(2, 2),
        seg_hw=(2, 2),
        frame_count=2,
    )
    assert cache_module.canonical_json_bytes(fresh_identity) == cache_module.canonical_json_bytes(resume_identity)

    root = tmp_path / "cache"
    SegnetHeadFeatureCache.create(
        root,
        identity=fresh_identity,
        rebuild_command=fresh_rebuild,
        storage_preflight=_preflight(root),
    )
    resumed = SegnetHeadFeatureCache.resume(root, expected_identity=resume_identity)
    assert (
        resumed.progress["identity_sha256"]
        == hashlib.sha256(cache_module.canonical_json_bytes(fresh_identity)).hexdigest()
    )
    persisted = [
        json.loads((root / name).read_text(encoding="utf-8"))
        for name in (
            cache_module.MANIFEST_NAME,
            PROGRESS_NAME,
            CERTIFICATION_NAME,
        )
    ]
    persisted_json = json.dumps(persisted, sort_keys=True)
    for volatile_field in ("parent_pid", "parent_exact_argv", "child_exact_argv"):
        assert volatile_field not in persisted_json


def test_partial_resume_revalidates_prefix_and_finishes_identically(tmp_path: Path) -> None:
    identity = _identity()
    root = tmp_path / "resumed"
    cache = SegnetHeadFeatureCache.create(
        root,
        identity=identity,
        rebuild_command=("python", "synthetic.py"),
        storage_preflight=_preflight(root),
    )
    live0 = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    quotient0 = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    cache.commit_frame(0, live0, quotient0)
    del cache
    resumed = SegnetHeadFeatureCache.resume(root, expected_identity=identity)
    live1 = live0 + 1
    quotient1 = quotient0 - 1
    resumed.commit_frame(1, live1, quotient1)
    resumed.mark_complete(positive_frame=0, fresh_live_logits=live0.copy())
    fresh = _populate(tmp_path / "fresh", identity)
    assert (root / "live_logits.f32.npy").read_bytes() == (fresh.root / "live_logits.f32.npy").read_bytes()
    assert (root / "quotient_features.f32.npy").read_bytes() == (fresh.root / "quotient_features.f32.npy").read_bytes()


def test_gt_f1_reader_maps_only_stored_member(tmp_path: Path) -> None:
    path = tmp_path / "tiny.npz"
    values = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
    np.savez(path, gt_f1=values, bulky_decoy=np.zeros(1024, dtype=np.float32))
    mapped = open_gt_f1_stored_memmap(path)
    assert isinstance(mapped, np.memmap)
    np.testing.assert_array_equal(mapped, values)


def test_commit_rejects_nonfinite_values(tmp_path: Path) -> None:
    cache = SegnetHeadFeatureCache.create(
        tmp_path / "cache",
        identity=_identity(),
        rebuild_command=("python", "synthetic.py"),
        storage_preflight=_preflight(tmp_path / "cache"),
    )
    live = np.zeros((3, 2, 2), dtype=np.float32)
    live[0, 0, 0] = np.nan
    with pytest.raises(FeatureCacheError, match="non-finite"):
        cache.commit_frame(
            0,
            live,
            np.zeros((2, 2, 2), dtype=np.float32),
        )


def test_commit_rejects_nonintegral_frame_index_without_coercion(tmp_path: Path) -> None:
    cache = SegnetHeadFeatureCache.create(
        tmp_path / "cache",
        identity=_identity(),
        rebuild_command=("python", "synthetic.py"),
        storage_preflight=_preflight(tmp_path / "cache"),
    )
    with pytest.raises(FeatureCacheError, match="Python or NumPy integer"):
        cache.commit_frame(
            0.0,  # type: ignore[arg-type]
            np.zeros((3, 2, 2), dtype=np.float32),
            np.zeros((2, 2, 2), dtype=np.float32),
        )


def test_cache_creation_atomically_recovers_certified_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cache"
    staging = root.with_name(f".{root.name}{cache_module.STAGING_SUFFIX}")

    def interrupt_final_rename(source: Path, destination: Path) -> None:
        if Path(source) == staging and Path(destination) == root:
            raise OSError("synthetic interruption before final rename")

    monkeypatch.setattr(cache_module, "_MOVE_PATH_NOREPLACE_TEST_HOOK", interrupt_final_rename)
    with pytest.raises(FeatureCacheError, match="pre-move interruption"):
        SegnetHeadFeatureCache.create(
            root,
            identity=_identity(),
            rebuild_command=("python", "synthetic.py"),
            storage_preflight=_preflight(root),
        )
    assert not root.exists()
    assert staging.is_dir()
    scratch = json.loads((staging / STAGING_SCRATCH_NAME).read_text(encoding="utf-8"))
    assert scratch["rebuildable"] is True
    assert scratch["final_cache_root"] == str(root.resolve())
    # A bounded parse failure is rebuildable only because the exact scratch
    # identity and certification were durably written before the arrays.
    with (staging / LIVE_LOGITS_NAME).open("r+b") as handle:
        handle.truncate(32)

    monkeypatch.setattr(cache_module, "_MOVE_PATH_NOREPLACE_TEST_HOOK", None)
    recovered = SegnetHeadFeatureCache.create(
        root,
        identity=_identity(),
        rebuild_command=("python", "synthetic.py"),
        storage_preflight=_preflight(root),
    )
    assert recovered.next_frame == 0
    assert root.is_dir()
    assert not staging.exists()
    assert (root / STAGING_SCRATCH_NAME).is_file()
    retained = [path for path in tmp_path.iterdir() if path.name.startswith(cache_module.DIRECTORY_RETAINED_PREFIX)]
    receipts = [
        path for path in tmp_path.iterdir() if path.name.startswith(cache_module.DIRECTORY_RETENTION_RECEIPT_PREFIX)
    ]
    assert len(retained) == len(receipts) == 1
    receipt, measured = cache_module._validate_directory_retention_receipt(receipts[0])
    assert Path(receipt["retention_destination"]) == retained[0]
    assert measured is not None


def test_cache_final_publish_refuses_late_destination_and_retries_without_tree_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cache"
    staging = cache_module._creation_staging_path(root.resolve())
    late_custody = tmp_path / "late-cache-destination"
    identities: dict[str, int] = {}

    def introduce_late_destination(source: Path, destination: Path) -> None:
        if source != staging or destination != root:
            return
        identities["staging"] = source.stat().st_ino
        destination.mkdir()
        (destination / "foreign.bin").write_bytes(b"late-foreign-destination")
        identities["foreign"] = destination.stat().st_ino

    monkeypatch.setattr(cache_module, "_MOVE_PATH_NOREPLACE_TEST_HOOK", introduce_late_destination)
    with pytest.raises(FeatureCacheError, match="destination appeared"):
        SegnetHeadFeatureCache.create(
            root,
            identity=_identity(),
            rebuild_command=("python", "synthetic.py"),
            storage_preflight=_preflight(root),
        )
    assert root.stat().st_ino == identities["foreign"]
    assert (root / "foreign.bin").read_bytes() == b"late-foreign-destination"
    assert staging.stat().st_ino == identities["staging"]

    os.replace(root, late_custody)
    monkeypatch.setattr(cache_module, "_MOVE_PATH_NOREPLACE_TEST_HOOK", None)
    recovered = SegnetHeadFeatureCache.create(
        root,
        identity=_identity(),
        rebuild_command=("python", "synthetic.py"),
        storage_preflight=_preflight(root),
    )
    assert recovered.next_frame == 0
    assert root.stat().st_ino == identities["staging"]
    assert late_custody.stat().st_ino == identities["foreign"]
    assert (late_custody / "foreign.bin").read_bytes() == b"late-foreign-destination"


def test_cache_final_publish_rolls_back_substituted_staging_and_preserves_both_trees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cache"
    staging = cache_module._creation_staging_path(root.resolve())
    admitted_displaced = tmp_path / "admitted-staging-displaced"
    foreign_custody = tmp_path / "foreign-staging-custody"
    identities: dict[str, int] = {}

    def substitute_staging(source: Path, destination: Path) -> None:
        if source != staging or destination != root:
            return
        identities["admitted"] = source.stat().st_ino
        os.replace(source, admitted_displaced)
        source.mkdir()
        (source / "foreign.bin").write_bytes(b"substituted-staging")
        identities["foreign"] = source.stat().st_ino

    monkeypatch.setattr(cache_module, "_MOVE_PATH_NOREPLACE_TEST_HOOK", substitute_staging)
    with pytest.raises(FeatureCacheError, match="rollback uncertainty"):
        SegnetHeadFeatureCache.create(
            root,
            identity=_identity(),
            rebuild_command=("python", "synthetic.py"),
            storage_preflight=_preflight(root),
        )
    assert not root.exists()
    assert admitted_displaced.stat().st_ino == identities["admitted"]
    assert staging.stat().st_ino == identities["foreign"]
    assert (staging / "foreign.bin").read_bytes() == b"substituted-staging"

    os.replace(staging, foreign_custody)
    os.replace(admitted_displaced, staging)
    monkeypatch.setattr(cache_module, "_MOVE_PATH_NOREPLACE_TEST_HOOK", None)
    recovered = SegnetHeadFeatureCache.create(
        root,
        identity=_identity(),
        rebuild_command=("python", "synthetic.py"),
        storage_preflight=_preflight(root),
    )
    assert recovered.next_frame == 0
    assert root.stat().st_ino == identities["admitted"]
    assert foreign_custody.stat().st_ino == identities["foreign"]
    assert (foreign_custody / "foreign.bin").read_bytes() == b"substituted-staging"


def test_cache_create_preserves_and_refuses_existing_empty_root(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    root_inode = root.stat().st_ino
    before = tuple(tmp_path.iterdir())

    with pytest.raises(FeatureCacheError, match="refusing existing cache directory"):
        SegnetHeadFeatureCache.create(
            root,
            identity=_identity(),
            rebuild_command=("python", "synthetic.py"),
            storage_preflight=_preflight(root),
        )

    assert root.stat().st_ino == root_inode
    assert tuple(tmp_path.iterdir()) == before


def test_cache_create_retires_empty_staging_with_validated_receipt(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    staging = cache_module._creation_staging_path(root.resolve())
    staging.mkdir()
    staging_inode = staging.stat().st_ino

    cache = SegnetHeadFeatureCache.create(
        root,
        identity=_identity(),
        rebuild_command=("python", "synthetic.py"),
        storage_preflight=_preflight(root),
    )

    assert cache.next_frame == 0
    retained = [path for path in tmp_path.iterdir() if path.name.startswith(cache_module.DIRECTORY_RETAINED_PREFIX)]
    receipts = [
        path for path in tmp_path.iterdir() if path.name.startswith(cache_module.DIRECTORY_RETENTION_RECEIPT_PREFIX)
    ]
    assert len(retained) == len(receipts) == 1
    assert retained[0].stat().st_ino == staging_inode
    receipt, measured = cache_module._validate_directory_retention_receipt(receipts[0])
    assert receipt["reason"] == "EMPTY_INTERRUPTED_CACHE_STAGING"
    assert Path(receipt["retention_destination"]) == retained[0]
    assert measured is not None and measured.entry_count == 0


def test_cache_retention_late_destination_collision_blocks_until_lossless_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cache"
    staging = cache_module._creation_staging_path(root.resolve())
    staging.mkdir()
    staging_inode = staging.stat().st_ino
    collision: dict[str, Path | int] = {}

    def introduce_retention_collision(source: Path, destination: Path) -> None:
        if source != staging or not destination.name.startswith(cache_module.DIRECTORY_RETAINED_PREFIX):
            return
        destination.mkdir()
        (destination / "foreign.bin").write_bytes(b"late-retention-destination")
        collision["path"] = destination
        collision["inode"] = destination.stat().st_ino

    monkeypatch.setattr(cache_module, "_MOVE_PATH_NOREPLACE_TEST_HOOK", introduce_retention_collision)
    with pytest.raises(FeatureCacheError, match="after its receipt was published"):
        SegnetHeadFeatureCache.create(
            root,
            identity=_identity(),
            rebuild_command=("python", "synthetic.py"),
            storage_preflight=_preflight(root),
        )
    destination = collision["path"]
    assert isinstance(destination, Path)
    assert staging.stat().st_ino == staging_inode
    assert destination.stat().st_ino == collision["inode"]
    assert (destination / "foreign.bin").read_bytes() == b"late-retention-destination"
    assert not root.exists()
    receipts = [
        path for path in tmp_path.iterdir() if path.name.startswith(cache_module.DIRECTORY_RETENTION_RECEIPT_PREFIX)
    ]
    assert len(receipts) == 1

    monkeypatch.setattr(cache_module, "_MOVE_PATH_NOREPLACE_TEST_HOOK", None)
    before_retry = _flat_tree_snapshot(tmp_path)
    with pytest.raises(FeatureCacheError, match="exactly one source/destination tree"):
        SegnetHeadFeatureCache.create(
            root,
            identity=_identity(),
            rebuild_command=("python", "synthetic.py"),
            storage_preflight=_preflight(root),
        )
    assert _flat_tree_snapshot(tmp_path) == before_retry

    foreign_custody = tmp_path / "late-retention-foreign-custody"
    os.replace(destination, foreign_custody)
    cache = SegnetHeadFeatureCache.create(
        root,
        identity=_identity(),
        rebuild_command=("python", "synthetic.py"),
        storage_preflight=_preflight(root),
    )
    assert cache.next_frame == 0
    assert destination.stat().st_ino == staging_inode
    assert foreign_custody.stat().st_ino == collision["inode"]
    assert (foreign_custody / "foreign.bin").read_bytes() == b"late-retention-destination"


def test_cache_creation_name_limit_refuses_before_mutating_existing_empty_root(tmp_path: Path) -> None:
    name_max = os.pathconf(tmp_path, "PC_NAME_MAX")
    root = tmp_path / ("x" * (name_max - 1))
    root.mkdir()
    root_inode = root.stat().st_ino
    before_names = {path.name for path in tmp_path.iterdir()}

    with pytest.raises(FeatureCacheError, match=r"NAME_MAX=.*refusing before mutation"):
        SegnetHeadFeatureCache.create(
            root,
            identity=_identity(),
            rebuild_command=("python", "synthetic.py"),
            storage_preflight=_preflight(root),
        )

    assert root.stat().st_ino == root_inode
    assert {path.name for path in tmp_path.iterdir()} == before_names


def test_truncated_staging_is_preserved_when_identity_or_certification_drifts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cache"
    staging = root.with_name(f".{root.name}{cache_module.STAGING_SUFFIX}")

    def interrupt_final_rename(source: Path, destination: Path) -> None:
        if Path(source) == staging and Path(destination) == root:
            raise OSError("synthetic interruption")

    monkeypatch.setattr(cache_module, "_MOVE_PATH_NOREPLACE_TEST_HOOK", interrupt_final_rename)
    with pytest.raises(FeatureCacheError, match="pre-move interruption"):
        SegnetHeadFeatureCache.create(
            root,
            identity=_identity(),
            rebuild_command=("python", "synthetic.py"),
            storage_preflight=_preflight(root),
        )
    with (staging / LIVE_LOGITS_NAME).open("r+b") as handle:
        handle.truncate(16)
    monkeypatch.setattr(cache_module, "_MOVE_PATH_NOREPLACE_TEST_HOOK", None)

    with pytest.raises(FeatureCacheError, match="identity-drifted"):
        SegnetHeadFeatureCache.create(
            root,
            identity=_identity(config_value=2),
            rebuild_command=("python", "synthetic.py"),
            storage_preflight=_preflight(root),
        )
    assert staging.is_dir()
    certification_path = staging / CERTIFICATION_NAME
    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    certification["false_authority_flags"]["score_authority"] = True
    _test_atomic_json(certification_path, certification)
    with pytest.raises(FeatureCacheError, match="certification"):
        SegnetHeadFeatureCache.create(
            root,
            identity=_identity(),
            rebuild_command=("python", "synthetic.py"),
            storage_preflight=_preflight(root),
        )
    assert staging.is_dir()
    assert (staging / LIVE_LOGITS_NAME).stat().st_size == 16


def test_final_certification_tamper_is_refused(tmp_path: Path) -> None:
    cache = _populate(tmp_path / "cache", _identity())
    certification_path = cache.root / CERTIFICATION_NAME
    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    certification["cleanup_action_performed"] = True
    _test_atomic_json(certification_path, certification)
    with pytest.raises(FeatureCacheError, match="certification"):
        validate_feature_cache(cache.root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("PASS", 1, "did not pass"),
        ("free_bytes_before", False, "byte counts"),
        ("required_free_bytes", 1.0, "byte counts"),
        ("selected_root", "relative/cache", "path/type custody"),
        ("filesystem_anchor", "relative", "path/type custody"),
        ("waterfall_order", [], "path/type custody"),
        ("allow_local_output_for_tests", 1, "path/type custody"),
    ],
)
def test_storage_preflight_exact_schema_and_types_are_required(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    preflight = _preflight(tmp_path / "cache")
    preflight[field] = value
    with pytest.raises(FeatureCacheError, match=message):
        SegnetHeadFeatureCache.create(
            tmp_path / "cache",
            identity=_identity(),
            rebuild_command=("python", "synthetic.py"),
            storage_preflight=preflight,
        )


def test_storage_preflight_requires_capacity_and_production_membership(tmp_path: Path) -> None:
    insufficient = _preflight(tmp_path / "insufficient")
    insufficient["required_free_bytes"] = insufficient["free_bytes_before"] + 1
    with pytest.raises(FeatureCacheError, match="did not pass"):
        SegnetHeadFeatureCache.create(
            tmp_path / "insufficient",
            identity=_identity(),
            rebuild_command=("python", "synthetic.py"),
            storage_preflight=insufficient,
        )

    outside = _preflight(tmp_path / "outside")
    outside["allow_local_output_for_tests"] = False
    outside["existing_approved_roots"] = ["/Volumes/VertigoDataTier/pact"]
    with pytest.raises(FeatureCacheError, match="first existing production SSD root"):
        SegnetHeadFeatureCache.create(
            tmp_path / "outside",
            identity=_identity(),
            rebuild_command=("python", "synthetic.py"),
            storage_preflight=outside,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frame", 0.0, "contiguous prefix"),
        ("frame", False, "contiguous prefix"),
        ("unexpected", 1, "contiguous prefix"),
        ("live_logits_sha256", "A" * 64, "row schema"),
        ("quotient_features_sha256", "0" * 63, "row schema"),
        ("diagnostics", False, "row schema"),
        ("diagnostics", {"bad": float("inf")}, "canonical finite JSON"),
    ],
)
def test_committed_frame_rows_require_exact_types_hashes_and_finite_diagnostics(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    cache = _populate(tmp_path / "cache", _identity())
    progress_path = cache.root / PROGRESS_NAME
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["committed_frames"][0][field] = value
    if field == "diagnostics" and isinstance(value, dict) and value.get("bad") == float("inf"):
        with pytest.raises(FeatureCacheError, match=message):
            _test_atomic_json(progress_path, progress)
        return
    _test_atomic_json(progress_path, progress)
    with pytest.raises(FeatureCacheError, match=message):
        validate_feature_cache(cache.root)


def test_storage_preflight_must_match_exactly_between_scratch_and_certification(tmp_path: Path) -> None:
    cache = _populate(tmp_path / "cache", _identity())
    scratch_path = cache.root / STAGING_SCRATCH_NAME
    scratch = json.loads(scratch_path.read_text(encoding="utf-8"))
    scratch["storage_preflight"]["free_bytes_before"] += 1
    _test_atomic_json(scratch_path, scratch)
    with pytest.raises(FeatureCacheError, match="certification"):
        validate_feature_cache(cache.root)


def test_progress_status_cross_invariants_and_completion_frame_tamper_fail(tmp_path: Path) -> None:
    complete = _populate(tmp_path / "complete", _identity())
    complete_progress_path = complete.root / PROGRESS_NAME
    complete_progress = json.loads(complete_progress_path.read_text(encoding="utf-8"))
    complete_progress["status"] = "partial"
    complete_progress["completion_positive_control"] = None
    _test_atomic_json(complete_progress_path, complete_progress)
    with pytest.raises(FeatureCacheError, match="partial cache must stop"):
        validate_feature_cache(complete.root)

    partial = SegnetHeadFeatureCache.create(
        tmp_path / "partial",
        identity=_identity(),
        rebuild_command=("python", "synthetic.py"),
        storage_preflight=_preflight(tmp_path / "partial"),
    )
    partial.commit_frame(
        0,
        np.zeros((3, 2, 2), dtype=np.float32),
        np.zeros((2, 2, 2), dtype=np.float32),
    )
    partial_progress_path = partial.root / PROGRESS_NAME
    partial_progress = json.loads(partial_progress_path.read_text(encoding="utf-8"))
    partial_progress["status"] = "ready_for_completion_validation"
    _test_atomic_json(partial_progress_path, partial_progress)
    with pytest.raises(FeatureCacheError, match="completion-ready cache"):
        validate_feature_cache(partial.root)

    frame_tamper = _populate(tmp_path / "frame-tamper", _identity())
    frame_progress_path = frame_tamper.root / PROGRESS_NAME
    frame_progress = json.loads(frame_progress_path.read_text(encoding="utf-8"))
    frame_progress["completion_positive_control"]["frame"] = 0.0
    _test_atomic_json(frame_progress_path, frame_progress)
    with pytest.raises(FeatureCacheError, match="positive-control schema"):
        validate_feature_cache(frame_tamper.root)


@pytest.mark.parametrize("status", ["partial", "ready_for_completion_validation"])
def test_noncomplete_cache_statuses_require_null_completion_control(
    tmp_path: Path,
    status: str,
) -> None:
    root = tmp_path / status
    cache = SegnetHeadFeatureCache.create(
        root,
        identity=_identity(),
        rebuild_command=("python", "synthetic.py"),
        storage_preflight=_preflight(root),
    )
    if status == "ready_for_completion_validation":
        for frame in range(2):
            cache.commit_frame(
                frame,
                np.zeros((3, 2, 2), dtype=np.float32),
                np.zeros((2, 2, 2), dtype=np.float32),
            )
    progress_path = root / PROGRESS_NAME
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["completion_positive_control"] = {"forged": True}
    _test_atomic_json(progress_path, progress)
    with pytest.raises(FeatureCacheError, match="partial cache carries"):
        validate_feature_cache(root)


def test_mark_complete_rejects_nonintegral_control_frame_without_coercion(tmp_path: Path) -> None:
    cache = SegnetHeadFeatureCache.create(
        tmp_path / "cache",
        identity=_identity(),
        rebuild_command=("python", "synthetic.py"),
        storage_preflight=_preflight(tmp_path / "cache"),
    )
    for frame in range(2):
        cache.commit_frame(
            frame,
            np.zeros((3, 2, 2), dtype=np.float32),
            np.zeros((2, 2, 2), dtype=np.float32),
        )
    with pytest.raises(FeatureCacheError, match="Python or NumPy integer"):
        cache.mark_complete(
            positive_frame=0.0,  # type: ignore[arg-type]
            fresh_live_logits=np.zeros((3, 2, 2), dtype=np.float32),
        )


def test_validation_is_read_only_and_writer_resume_is_writable(tmp_path: Path) -> None:
    cache = _populate(tmp_path / "cache", _identity())
    validation = validate_feature_cache(cache.root)
    assert validation.live_logits.flags.writeable is False
    assert validation.quotient_features.flags.writeable is False
    writer = SegnetHeadFeatureCache.resume(cache.root, expected_identity=_identity())
    assert writer.live_logits.flags.writeable is True
    assert writer.quotient_features.flags.writeable is True


def test_completion_scan_is_frame_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _populate(tmp_path / "cache", _identity())
    original = np.isfinite
    observed_sizes: list[int] = []

    def recording_isfinite(value: object) -> np.ndarray:
        observed_sizes.append(np.asarray(value).size)
        return original(value)

    monkeypatch.setattr(np, "isfinite", recording_isfinite)
    validate_feature_cache(cache.root, require_complete=True)
    assert observed_sizes
    assert max(observed_sizes) <= max(
        int(np.prod(cache.live_logits.shape[1:])),
        int(np.prod(cache.quotient_features.shape[1:])),
    )


def test_full_request_refuses_ready_for_completion_validation(tmp_path: Path) -> None:
    cache = _populate(tmp_path / "cache", _identity())
    progress_path = cache.root / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["status"] = "ready_for_completion_validation"
    progress["completion_positive_control"] = None
    _test_atomic_json(progress_path, progress)
    with pytest.raises(FeatureCacheError, match="explicitly partial"):
        _validate_feature_for_request(cache.root, max_frames=EXPECTED_PAIRS)


def test_profile_cross_binds_gt_source_path_bytes_and_hash(tmp_path: Path) -> None:
    bound_gt = tmp_path / "bound.npz"
    other_gt = tmp_path / "other.npz"
    bound_gt.write_bytes(b"bound")
    other_gt.write_bytes(b"other")
    identity = build_immutable_identity(
        source_files={
            "gt_n600_npz": _source(str(bound_gt.resolve()), b"bound"),
            "extractor_tool": cache_module.source_file_row(Path(extractor_module.__file__)),
            "cache_module": cache_module.source_file_row(Path(cache_module.__file__)),
        },
        config={
            "authority_mode": "deterministic_cpu_float32_batch_one",
            "batch_size": 1,
            "runtime": {
                "python": extractor_module.platform.python_version(),
                "python_implementation": extractor_module.platform.python_implementation(),
                "python_executable": str(Path(sys.executable).resolve()),
                "torch": extractor_module.torch.__version__,
                "numpy": np.__version__,
                "platform": extractor_module.platform.platform(),
            },
            "determinism": {
                "torch_deterministic_algorithms": True,
                "torch_threads_effective": 1,
                "torch_interop_threads_effective": 1,
            },
        },
        frame_count=2,
        live_slice_shape=(3, 2, 2),
        quotient_slice_shape=(2, 2, 2),
    )
    cache = _populate(tmp_path / "cache", identity)
    validation = validate_feature_cache(cache.root)
    binding = _feature_binding(validation, bound_gt, prefix_frames=2)
    assert binding["gt_n600_npz"]["path"] == str(bound_gt.resolve())
    with pytest.raises(ProfilerError, match="does not match"):
        _feature_binding(validation, other_gt, prefix_frames=2)


def test_exact_imported_path_custody_refuses_same_root_sibling(tmp_path: Path) -> None:
    expected = tmp_path / "modules.py"
    sibling = tmp_path / "other.py"
    expected.write_text("# expected\n", encoding="utf-8")
    sibling.write_text("# sibling\n", encoding="utf-8")
    module = SimpleNamespace(__file__=str(sibling))
    with pytest.raises(ExtractionError, match="!= manifest source"):
        _require_exact_module_file(module, expected, role="modules")


def test_local_output_requires_pytest_and_refuses_full_n600() -> None:
    tiny = SimpleNamespace(allow_local_output_for_tests=True, max_frames=2)
    with pytest.raises(ExtractionError, match="actual pytest"):
        _require_local_test_scope(tiny, env={})
    _require_local_test_scope(tiny, env={"PYTEST_CURRENT_TEST": "cache.py::test"})
    full = SimpleNamespace(allow_local_output_for_tests=True, max_frames=600)
    with pytest.raises(ExtractionError, match="absolute prefix"):
        _require_local_test_scope(full, env={"PYTEST_CURRENT_TEST": "cache.py::test"})


def test_extractor_exact_argv_substitution_is_refused(tmp_path: Path) -> None:
    tokens = [
        "--gt-cache",
        str(tmp_path / "gt.npz"),
        "--upstream-root",
        str(tmp_path / "upstream"),
        "--output-root",
        str(tmp_path / "output"),
        "--max-frames",
        "2",
        "--rss-cap-mb",
        "512",
        "--timeout-seconds",
        "60",
        "--allow-local-output-for-tests",
    ]
    args = _parse_args(tokens)
    exact = [sys.executable, str(Path(extractor_module.__file__).resolve()), *tokens]
    assert _attest_exact_argv(args, exact) == exact
    substituted = list(exact)
    substituted[substituted.index("2")] = "1"
    with pytest.raises(ExtractionError, match="does not reproduce"):
        _attest_exact_argv(args, substituted)


def test_storage_waterfall_refuses_lower_tier_when_higher_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    high = tmp_path / "VertigoDataTier" / "pact"
    low = tmp_path / "APDataStore" / "pact"
    high.mkdir(parents=True)
    low.mkdir(parents=True)
    monkeypatch.setattr(extractor_module, "SSD_ROOTS", (high, low))
    with pytest.raises(ExtractionError, match="first existing SSD root"):
        storage_preflight(
            low / "cache",
            required_bytes=1,
            allow_local_output_for_tests=False,
        )
    receipt = storage_preflight(
        high / "cache",
        required_bytes=1,
        allow_local_output_for_tests=False,
    )
    assert receipt["selected_root"] == str((high / "cache").resolve())
    assert receipt["existing_approved_roots"] == [str(high.resolve()), str(low.resolve())]


def test_cache_root_symlink_and_custody_hardlinks_are_refused(tmp_path: Path) -> None:
    cache = _populate(tmp_path / "cache", _identity())
    alias = tmp_path / "cache-alias"
    alias.symlink_to(cache.root, target_is_directory=True)
    with pytest.raises(FeatureCacheError, match="non-symlink directory"):
        validate_feature_cache(alias)

    manifest = cache.root / cache_module.MANIFEST_NAME
    os.link(manifest, tmp_path / "manifest-hardlink.json")
    with pytest.raises(FeatureCacheError, match="exactly one hard link"):
        validate_feature_cache(cache.root)


def test_writable_resume_array_hardlink_is_refused(tmp_path: Path) -> None:
    cache = _populate(tmp_path / "cache", _identity())
    os.link(cache.root / LIVE_LOGITS_NAME, tmp_path / "live-hardlink.npy")
    with pytest.raises(FeatureCacheError, match="exactly one hard link"):
        SegnetHeadFeatureCache.resume(cache.root, expected_identity=_identity())


def test_unknown_staging_sentinel_blocks_cleanup_and_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cache"
    staging = root.with_name(f".{root.name}{cache_module.STAGING_SUFFIX}")

    def interrupt_final_rename(source: Path, destination: Path) -> None:
        if Path(source) == staging and Path(destination) == root:
            raise OSError("synthetic interruption")

    monkeypatch.setattr(cache_module, "_MOVE_PATH_NOREPLACE_TEST_HOOK", interrupt_final_rename)
    with pytest.raises(FeatureCacheError, match="pre-move interruption"):
        SegnetHeadFeatureCache.create(
            root,
            identity=_identity(),
            rebuild_command=("python", "synthetic.py"),
            storage_preflight=_preflight(root),
        )
    with (staging / LIVE_LOGITS_NAME).open("r+b") as handle:
        handle.truncate(16)
    sentinel = staging / "unknown.operator-bytes"
    sentinel.write_bytes(b"preserve me")
    monkeypatch.setattr(cache_module, "_MOVE_PATH_NOREPLACE_TEST_HOOK", None)

    with pytest.raises(FeatureCacheError, match=r"unknown=.*unknown.operator-bytes.*preserving bytes"):
        SegnetHeadFeatureCache.create(
            root,
            identity=_identity(),
            rebuild_command=("python", "synthetic.py"),
            storage_preflight=_preflight(root),
        )
    assert sentinel.read_bytes() == b"preserve me"
    assert (staging / LIVE_LOGITS_NAME).stat().st_size == 16


@pytest.mark.parametrize("complete", [False, True])
def test_stable_atomic_metadata_prepared_file_is_resumed_without_pid_names(
    tmp_path: Path,
    complete: bool,
) -> None:
    target = tmp_path / "progress.json"
    value = {"schema": "test", "value": 7}
    expected = cache_module.canonical_json_bytes(value) + b"\n"
    prepared = cache_module.atomic_prepared_path(target)
    prepared.write_bytes(expected if complete else expected[: max(1, len(expected) // 2)])

    _test_atomic_json(target, value)

    assert target.read_bytes() == expected
    assert not prepared.exists()
    assert ".tmp-" not in prepared.name


def test_stable_atomic_metadata_drift_and_read_failure_preserve_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "progress.json"
    prepared = cache_module.atomic_prepared_path(target)
    prepared.write_bytes(b"x" * len(cache_module.canonical_json_bytes({"value": 1}) + b"\n"))
    before = prepared.read_bytes()
    with pytest.raises(FeatureCacheError, match="payload drift"):
        _test_atomic_json(target, {"value": 1})
    assert prepared.read_bytes() == before
    assert not target.exists()

    real_read_bound = cache_module.read_bound_file

    def fail_prepared_read(path: Path, *, role: str) -> cache_module.BoundFileSnapshot:
        if path == prepared:
            raise FeatureCacheError(f"stable atomic prepared file cannot be read; preserving bytes: {path}")
        return real_read_bound(path, role=role)

    monkeypatch.setattr(cache_module, "read_bound_file", fail_prepared_read)
    with pytest.raises(FeatureCacheError, match=r"cannot be read.*preserving bytes"):
        _test_atomic_json(target, {"value": 1})
    monkeypatch.setattr(cache_module, "read_bound_file", real_read_bound)
    assert prepared.read_bytes() == before
    assert not target.exists()


@pytest.mark.parametrize("interrupted_name", [STAGING_SCRATCH_NAME, cache_module.MANIFEST_NAME])
def test_cache_creation_rebuilds_identity_matched_stable_metadata_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_name: str,
) -> None:
    root = tmp_path / "cache"
    staging = cache_module._creation_staging_path(root.resolve())
    target = staging / interrupted_name
    prepared = cache_module.atomic_prepared_path(target)

    def interrupt_metadata_commit(source: Path, destination: Path) -> None:
        if source == prepared and destination == target:
            raise OSError("synthetic metadata commit interruption")

    monkeypatch.setattr(cache_module, "_ATOMIC_COMMIT_TEST_HOOK", interrupt_metadata_commit)
    with pytest.raises(FeatureCacheError, match=r"pre-commit interruption.*preserving bytes"):
        SegnetHeadFeatureCache.create(
            root,
            identity=_identity(),
            rebuild_command=("python", "synthetic.py"),
            storage_preflight=_preflight(root),
        )
    assert prepared.is_file()
    monkeypatch.setattr(cache_module, "_ATOMIC_COMMIT_TEST_HOOK", None)

    cache = SegnetHeadFeatureCache.create(
        root,
        identity=_identity(),
        rebuild_command=("python", "synthetic.py"),
        storage_preflight=_preflight(root),
    )
    assert cache.next_frame == 0
    assert not staging.exists()
    assert not prepared.exists()


def test_final_cache_layout_blocks_runtime_atomic_scratch_without_transaction_provenance(tmp_path: Path) -> None:
    cache = _populate(tmp_path / "cache", _identity())
    prepared = cache_module.atomic_prepared_path(cache.root / PROGRESS_NAME)
    prepared.write_bytes(b"certified writer-owned prefix")
    with pytest.raises(FeatureCacheError, match="orphaned, duplicated, or role-ambiguous"):
        validate_feature_cache(cache.root, require_complete=True)
    assert prepared.read_bytes() == b"certified writer-owned prefix"


def test_final_cache_layout_still_blocks_legacy_pid_scratch(tmp_path: Path) -> None:
    cache = _populate(tmp_path / "cache", _identity())
    legacy = cache.root / f".{PROGRESS_NAME}.tmp-123"
    legacy.write_bytes(b"legacy")
    with pytest.raises(FeatureCacheError, match=r"unknown=.*tmp-123.*preserving bytes"):
        validate_feature_cache(cache.root)
    assert legacy.read_bytes() == b"legacy"


def test_extractor_receipt_is_recoverable_sibling_without_cache_layout_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    cache = _populate(tmp_path / "cache", identity)
    cache_entries = {path.name for path in cache.root.iterdir()}
    receipt = {
        "schema": "extract_segnet_head_features_n600_receipt.v1",
        "cache_root": str(cache.root),
        "canonical_fresh_rebuild_argv": ["python", "extract.py"],
        "storage_preflight": _preflight(cache.root),
        "completion_positive_control": cache.progress["completion_positive_control"],
        "authority": {"score_authority": False, "promotion_eligible": False},
    }
    receipt_path = cache_module.extractor_receipt_path(cache.root)
    assert receipt_path.parent == cache.root.parent
    assert receipt_path.parent != cache.root

    _emit_extraction_receipt(
        cache.root,
        receipt,
        identity=identity,
        require_complete=True,
    )
    expected = cache_module.canonical_json_bytes(receipt) + b"\n"
    prepared = cache_module.atomic_prepared_path(receipt_path)
    prepared.write_bytes(expected[: len(expected) // 2])
    before = {path.name: path.read_bytes() for path in receipt_path.parent.iterdir() if path.is_file()}
    with pytest.raises(FeatureCacheError, match="lacks one consistent committed transaction"):
        _emit_extraction_receipt(
            cache.root,
            receipt,
            identity=identity,
            require_complete=True,
        )
    after = {path.name: path.read_bytes() for path in receipt_path.parent.iterdir() if path.is_file()}
    assert after == before
    assert prepared.read_bytes() == expected[: len(expected) // 2]

    assert {path.name for path in cache.root.iterdir()} == cache_entries
    assert "receipt.json" not in cache_entries
    assert validate_feature_cache(cache.root, require_complete=True).complete
    assert SegnetHeadFeatureCache.resume(cache.root, expected_identity=identity).progress["status"] == "complete"


def test_frozen_source_snapshots_require_complete_byte_equality() -> None:
    stable = {
        role: _source(f"/source/{role}", role.encode())
        for role in (
            "executed_modules_py",
            "executed_frame_utils_py",
            "segnet_weights",
            "executed_tac_scorer_py",
            "executed_factorization_module_py",
        )
    }
    assert _require_equal_source_snapshots(stable, stable) == stable
    for role in stable:
        changed = json.loads(json.dumps(stable))
        changed[role]["sha256"] = "0" * 64
        with pytest.raises(ExtractionError, match=role):
            _require_equal_source_snapshots(stable, changed)


def test_bound_source_read_rejects_path_replacement_during_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    displaced = tmp_path / "displaced.bin"
    foreign = tmp_path / "foreign.bin"
    source.write_bytes(b"original-source")
    foreign.write_bytes(b"foreign-source")
    real_read = cache_module.os.read
    replaced = False

    def replace_after_read(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        payload = real_read(descriptor, count)
        if payload and not replaced:
            replaced = True
            cache_module.os.replace(source, displaced)
            cache_module.os.replace(foreign, source)
        return payload

    monkeypatch.setattr(cache_module.os, "read", replace_after_read)
    with pytest.raises(FeatureCacheError, match="pathname identity changed"):
        cache_module.source_file_row(source)
    assert displaced.read_bytes() == b"original-source"
    assert source.read_bytes() == b"foreign-source"


def test_prepared_prefix_generation_refuses_replaced_inode_at_authorization_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "progress.json"
    prepared = cache_module.atomic_prepared_path(target)
    expected = cache_module.canonical_json_bytes({"value": 1}) + b"\n"
    original_prefix = expected[:5]
    prepared.write_bytes(original_prefix)
    displaced = tmp_path / "original-prefix.bin"
    foreign = tmp_path / "foreign-prefix.bin"
    foreign.write_bytes(b"foreign")

    def replace_at_authorization(path: Path) -> None:
        assert path == prepared
        cache_module.os.replace(prepared, displaced)
        cache_module.os.replace(foreign, prepared)

    monkeypatch.setattr(cache_module, "_ATOMIC_PREFIX_AUTHORIZATION_TEST_HOOK", replace_at_authorization)
    with pytest.raises(FeatureCacheError, match="pathname changed at authorization boundary"):
        _test_atomic_json(target, {"value": 1})
    assert displaced.read_bytes() == original_prefix
    assert prepared.read_bytes() == b"foreign"
    assert not target.exists()


def test_atomic_commit_substitution_rolls_back_without_foreign_target_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "progress.json"
    old_target = b"old-authoritative-target\n"
    target.write_bytes(old_target)
    old_target_inode = target.stat().st_ino
    value = {"value": 17}
    expected = cache_module.canonical_json_bytes(value) + b"\n"
    prepared = cache_module.atomic_prepared_path(target)
    prepared.write_bytes(expected)
    displaced = tmp_path / "validated-source-displaced.json"
    foreign = tmp_path / "foreign-source.json"
    foreign_bytes = b"foreign-must-never-be-target\n"
    foreign.write_bytes(foreign_bytes)

    def substitute_immediately_before_exchange(source: Path, destination: Path) -> None:
        assert source == prepared
        assert destination == target
        cache_module.os.replace(source, displaced)
        cache_module.os.replace(foreign, source)

    monkeypatch.setattr(cache_module, "_ATOMIC_COMMIT_TEST_HOOK", substitute_immediately_before_exchange)
    with pytest.raises(FeatureCacheError, match=r"payload drift|final publication boundary"):
        _test_atomic_json(target, value)

    assert target.read_bytes() == old_target
    assert target.stat().st_ino == old_target_inode
    assert displaced.read_bytes() == expected
    assert prepared.read_bytes() == foreign_bytes
    assert not foreign.exists()


def test_retention_compresses_long_original_basename_and_roundtrips(tmp_path: Path) -> None:
    original = f".frame_0000.bin.intent-attempt-00000000-9618-{'5a37' * 16}"
    source = tmp_path / original
    source.write_bytes(b"")
    snapshot = cache_module.read_bound_file(source, role="long retained source")

    retained = cache_module.retain_bound_file(source, snapshot, role="long retained source")

    assert len(cache_module.os.fsencode(retained.name)) <= cache_module.ATOMIC_RETAINED_COMPONENT_MAX
    assert cache_module.retained_original_name(retained.name) == original
    assert cache_module.validate_retained_file(retained, role="long retained result") == snapshot
    assert not source.exists()


def test_retention_name_overflow_refuses_before_mutation(tmp_path: Path) -> None:
    original = "".join(hashlib.sha256(f"retention-name-{index}".encode()).hexdigest() for index in range(3))
    source = tmp_path / original
    payload = b"preserve-before-name-overflow"
    source.write_bytes(payload)
    snapshot = cache_module.read_bound_file(source, role="oversized retained source")

    with pytest.raises(FeatureCacheError, match="cannot fit canonical retention naming"):
        cache_module.retain_bound_file(source, snapshot, role="oversized retained source")

    assert source.read_bytes() == payload
    assert {path.name for path in tmp_path.iterdir()} == {original}


@pytest.mark.parametrize("existing_target", [False, True])
def test_atomic_commit_post_linearization_crash_retries_and_cleans_displaced_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_target: bool,
) -> None:
    target = tmp_path / "progress.json"
    old_target = b"old-target\n"
    if existing_target:
        target.write_bytes(old_target)
    value = {"value": 23}
    expected = cache_module.canonical_json_bytes(value) + b"\n"
    prepared = cache_module.atomic_prepared_path(target)
    prepared.write_bytes(expected)

    def cut_after_exchange(source: Path, destination: Path) -> None:
        assert source == prepared
        assert destination == target
        raise OSError("synthetic post-linearization cut")

    monkeypatch.setattr(cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", cut_after_exchange)
    with pytest.raises(FeatureCacheError, match="post-linearization interruption"):
        _test_atomic_json(target, value)
    assert target.read_bytes() == expected
    if existing_target:
        assert prepared.read_bytes() == old_target
    else:
        assert not prepared.exists()

    monkeypatch.setattr(cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", None)
    cache_module.atomic_json(
        target,
        value,
        expected_prior_payloads=((old_target,) if existing_target else ()),
    )
    assert target.read_bytes() == expected
    assert not prepared.exists()
    assert not cache_module._atomic_generation_paths(target)
    assert not cache_module._atomic_transaction_paths(target)
    retired_originals = {
        cache_module.retained_original_name(path.name)
        for path in tmp_path.iterdir()
        if cache_module.is_retained_name(path.name)
    }
    assert (prepared.name in retired_originals) is existing_target
    assert any(
        name.startswith(f".{target.name}{cache_module.ATOMIC_TRANSACTION_SUFFIX}-") for name in retired_originals
    )


@pytest.mark.parametrize(
    ("target_name", "existing_prior"),
    [
        ("recovery_manifest.json", False),
        ("profile_progress.json", True),
        ("profile_receipt.json", True),
        (PROGRESS_NAME, True),
        (CERTIFICATION_NAME, True),
        (COMPLETION_CONTROL_NAME, True),
    ],
)
@pytest.mark.parametrize("foreign_kind", ["arbitrary", "canonical_json"])
def test_post_exchange_foreign_scratch_blocks_every_consumer_byte_identically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
    existing_prior: bool,
    foreign_kind: str,
) -> None:
    target = tmp_path / target_name
    if existing_prior:
        target.write_bytes(cache_module.canonical_json_bytes({"state": "authorized-prior"}) + b"\n")
    desired = {"state": "desired", "target": target_name}
    prepared = cache_module.atomic_prepared_path(target)

    def cut_after_exchange(source: Path, destination: Path) -> None:
        assert source == prepared
        assert destination == target
        raise OSError("synthetic post-exchange cut")

    monkeypatch.setattr(cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", cut_after_exchange)
    with pytest.raises(FeatureCacheError, match="post-linearization interruption"):
        _test_atomic_json(target, desired)
    monkeypatch.setattr(cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", None)

    displaced = tmp_path / f"{target_name}.descriptor-admitted-prior"
    injected = tmp_path / f"{target_name}.foreign-injection"
    foreign_payload = (
        b"arbitrary post-exchange foreign bytes"
        if foreign_kind == "arbitrary"
        else cache_module.canonical_json_bytes({"state": "valid-but-foreign"}) + b"\n"
    )
    injected.write_bytes(foreign_payload)
    if existing_prior:
        os.replace(prepared, displaced)
        os.replace(injected, prepared)
    else:
        os.replace(injected, prepared)
    before = _flat_tree_snapshot(tmp_path)

    expected_error = "scratch differs from pre-exchange custody" if existing_prior else "contradictory post-publication"
    with pytest.raises(FeatureCacheError, match=expected_error):
        _test_atomic_json(target, desired)

    assert _flat_tree_snapshot(tmp_path) == before
    assert target.read_bytes() == cache_module.canonical_json_bytes(desired) + b"\n"
    assert prepared.read_bytes() == foreign_payload


@pytest.mark.parametrize(
    "target_name",
    [
        "recovery_manifest.json",
        "profile_progress.json",
        "profile_receipt.json",
        PROGRESS_NAME,
        CERTIFICATION_NAME,
        COMPLETION_CONTROL_NAME,
    ],
)
def test_atomic_consumer_refuses_unauthorized_existing_target_before_writer_mutation(
    tmp_path: Path,
    target_name: str,
) -> None:
    target = tmp_path / target_name
    target.write_bytes(cache_module.canonical_json_bytes({"state": "foreign-prior"}) + b"\n")
    before = _flat_tree_snapshot(tmp_path)

    with pytest.raises(FeatureCacheError, match="not a consumer-authorized exact state"):
        cache_module.atomic_json(target, {"state": "desired"})

    assert _flat_tree_snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "malformed_name",
    [
        ".progress.json.atomic-generation-not-a-generation",
        ".progress.json.atomic-transaction-0000000x",
    ],
)
def test_atomic_writer_refuses_malformed_evidence_before_any_mutation(
    tmp_path: Path,
    malformed_name: str,
) -> None:
    target = tmp_path / PROGRESS_NAME
    prior = cache_module.canonical_json_bytes({"state": "authorized-prior"}) + b"\n"
    target.write_bytes(prior)
    malformed = tmp_path / malformed_name
    malformed.write_bytes(b"operator-unknown atomic evidence")
    before = _flat_tree_snapshot(tmp_path)

    with pytest.raises(FeatureCacheError, match="malformed atomic"):
        cache_module.atomic_json(
            target,
            {"state": "desired"},
            expected_prior_payloads=(prior,),
        )

    assert _flat_tree_snapshot(tmp_path) == before


def test_fresh_atomic_prepublication_cut_is_reachable_without_target_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "recovery_manifest.json"
    prepared = cache_module.atomic_prepared_path(target)
    desired = {"state": "fresh-manifest"}

    def cut_before_exchange(_source: Path, _destination: Path) -> None:
        raise OSError("synthetic pre-exchange cut")

    monkeypatch.setattr(cache_module, "_ATOMIC_COMMIT_TEST_HOOK", cut_before_exchange)
    with pytest.raises(FeatureCacheError, match="pre-commit interruption"):
        cache_module.atomic_json(target, desired)

    assert not target.exists()
    assert prepared.read_bytes() == cache_module.canonical_json_bytes(desired) + b"\n"
    assert cache_module._atomic_transaction_paths(target)
    prepared_inode = prepared.stat().st_ino

    monkeypatch.setattr(cache_module, "_ATOMIC_COMMIT_TEST_HOOK", None)
    cache_module.atomic_json(target, desired)

    assert target.read_bytes() == cache_module.canonical_json_bytes(desired) + b"\n"
    assert target.stat().st_ino == prepared_inode
    assert not prepared.exists()
    assert not cache_module._atomic_generation_paths(target)
    assert not cache_module._atomic_transaction_paths(target)


def test_fresh_atomic_crash_after_desired_fsync_before_transaction_retries_inode_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "recovery_manifest.json"
    desired = {"state": "fresh-before-transaction"}
    expected = cache_module.canonical_json_bytes(desired) + b"\n"
    real_write_transaction = cache_module._write_atomic_transaction_generation

    def cut_before_transaction(_path: Path, _record: dict[str, object]) -> Path:
        raise FeatureCacheError("synthetic cut before durable transaction")

    monkeypatch.setattr(cache_module, "_write_atomic_transaction_generation", cut_before_transaction)
    with pytest.raises(FeatureCacheError, match="before durable transaction"):
        cache_module.atomic_json(target, desired)

    prepared = cache_module.atomic_prepared_path(target)
    prepared_inode = prepared.stat().st_ino
    assert prepared.read_bytes() == expected
    assert not target.exists()
    assert not cache_module._atomic_transaction_paths(target)

    monkeypatch.setattr(cache_module, "_write_atomic_transaction_generation", real_write_transaction)
    cache_module.atomic_json(target, desired)

    assert target.read_bytes() == expected
    assert target.stat().st_ino == prepared_inode
    assert not prepared.exists()


def test_fresh_atomic_partial_only_transaction_generation_refuses_without_rewriting_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "recovery_manifest.json"
    desired = {"state": "fresh-partial-transaction"}

    def cut_during_transaction(path: Path, record: dict[str, object]) -> Path:
        transaction = cache_module.atomic_transaction_path(path, 0)
        full_payload = cache_module.canonical_json_bytes(record) + b"\n"
        transaction.write_bytes(full_payload[: len(full_payload) // 2])
        raise FeatureCacheError("synthetic partial transaction cut")

    monkeypatch.setattr(cache_module, "_write_atomic_transaction_generation", cut_during_transaction)
    with pytest.raises(FeatureCacheError, match="partial transaction cut"):
        cache_module.atomic_json(target, desired)

    prepared = cache_module.atomic_prepared_path(target)
    partial = cache_module.atomic_transaction_path(target, 0)
    partial_bytes = partial.read_bytes()
    assert prepared.is_file()
    assert partial_bytes
    assert not target.exists()

    before = _flat_tree_snapshot(tmp_path)
    with pytest.raises(FeatureCacheError, match=r"partial|complete fresh authority"):
        cache_module.atomic_json(target, desired)
    assert _flat_tree_snapshot(tmp_path) == before
    assert partial.read_bytes() == partial_bytes
    assert not target.exists()


@pytest.mark.parametrize("late_payload_kind", ["foreign", "same-bytes-aba"])
def test_fresh_atomic_late_target_blocks_and_preserves_both_inodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    late_payload_kind: str,
) -> None:
    target = tmp_path / "recovery_manifest.json"
    desired = {"state": "fresh-late-target"}
    expected = cache_module.canonical_json_bytes(desired) + b"\n"
    late_payload = expected if late_payload_kind == "same-bytes-aba" else b"foreign-late-target\n"
    observed_source_inode = -1

    def install_late_target(source: Path, destination: Path) -> None:
        nonlocal observed_source_inode
        observed_source_inode = source.stat().st_ino
        assert destination == target
        destination.write_bytes(late_payload)

    monkeypatch.setattr(cache_module, "_ATOMIC_COMMIT_TEST_HOOK", install_late_target)
    with pytest.raises(FeatureCacheError, match="fresh atomic destination appeared"):
        cache_module.atomic_json(target, desired)

    prepared = cache_module.atomic_prepared_path(target)
    assert prepared.read_bytes() == expected
    assert prepared.stat().st_ino == observed_source_inode
    assert target.read_bytes() == late_payload
    assert target.stat().st_ino != observed_source_inode
    before = _flat_tree_snapshot(tmp_path)

    monkeypatch.setattr(cache_module, "_ATOMIC_COMMIT_TEST_HOOK", None)
    expected_error = "designated source inode" if late_payload_kind == "same-bytes-aba" else "consumer-authorized"
    with pytest.raises(FeatureCacheError, match=expected_error):
        cache_module.atomic_json(target, desired)
    assert _flat_tree_snapshot(tmp_path) == before


def test_fresh_atomic_complete_transaction_with_both_names_absent_blocks_byte_identically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "recovery_manifest.json"
    desired = {"state": "fresh-both-absent"}

    def cut_before_publication(_source: Path, _destination: Path) -> None:
        raise OSError("synthetic pre-publication cut")

    monkeypatch.setattr(cache_module, "_ATOMIC_COMMIT_TEST_HOOK", cut_before_publication)
    with pytest.raises(FeatureCacheError, match="pre-commit interruption"):
        cache_module.atomic_json(target, desired)
    monkeypatch.setattr(cache_module, "_ATOMIC_COMMIT_TEST_HOOK", None)

    prepared = cache_module.atomic_prepared_path(target)
    preserved = tmp_path / "operator-preserved-designated-source.json"
    os.replace(prepared, preserved)
    before = _flat_tree_snapshot(tmp_path)

    with pytest.raises(FeatureCacheError, match="lost its designated source"):
        cache_module.atomic_json(target, desired)

    assert _flat_tree_snapshot(tmp_path) == before
    assert preserved.read_bytes() == cache_module.canonical_json_bytes(desired) + b"\n"


def test_existing_atomic_retry_replays_current_exact_prior_before_any_retention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / PROGRESS_NAME
    prior_a = cache_module.canonical_json_bytes({"state": "prior-a"}) + b"\n"
    prior_b = cache_module.canonical_json_bytes({"state": "prior-b"}) + b"\n"
    desired = {"state": "desired"}
    target.write_bytes(prior_a)

    def cut_after_exchange(_source: Path, _destination: Path) -> None:
        raise OSError("synthetic post-exchange cut")

    monkeypatch.setattr(cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", cut_after_exchange)
    with pytest.raises(FeatureCacheError, match="post-linearization interruption"):
        cache_module.atomic_json(target, desired, expected_prior_payloads=(prior_a,))
    monkeypatch.setattr(cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", None)
    before = _flat_tree_snapshot(tmp_path)

    with pytest.raises(FeatureCacheError, match="not authorized by the current consumer"):
        cache_module.atomic_json(target, desired, expected_prior_payloads=(prior_b,))
    assert _flat_tree_snapshot(tmp_path) == before

    cache_module.atomic_json(target, desired, expected_prior_payloads=(prior_a,))
    assert target.read_bytes() == cache_module.canonical_json_bytes(desired) + b"\n"


def test_sequential_existing_atomic_updates_do_not_reauthorize_retired_history_root(
    tmp_path: Path,
) -> None:
    target = tmp_path / PROGRESS_NAME
    prior_a = cache_module.canonical_json_bytes({"state": "legacy-a"}) + b"\n"
    value_b = {"state": "atomic-b"}
    payload_b = cache_module.canonical_json_bytes(value_b) + b"\n"
    value_c = {"state": "atomic-c"}
    payload_c = cache_module.canonical_json_bytes(value_c) + b"\n"
    target.write_bytes(prior_a)

    cache_module.atomic_json(target, value_b, expected_prior_payloads=(prior_a,))
    assert target.read_bytes() == payload_b
    first_retained = cache_module.validate_atomic_transaction_custody(
        target,
        desired_payload=payload_b,
    )
    assert first_retained

    cache_module.atomic_json(target, value_c, expected_prior_payloads=(payload_b,))
    assert target.read_bytes() == payload_c
    final_retained = cache_module.validate_atomic_transaction_custody(
        target,
        desired_payload=payload_c,
    )
    assert first_retained < final_retained


def test_forged_canonical_transaction_cannot_self_authorize_foreign_prior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / PROGRESS_NAME
    prior_a = cache_module.canonical_json_bytes({"state": "prior-a"}) + b"\n"
    prior_b = cache_module.canonical_json_bytes({"state": "foreign-prior-b"}) + b"\n"
    desired = {"state": "desired"}
    target.write_bytes(prior_a)

    def cut_after_exchange(_source: Path, _destination: Path) -> None:
        raise OSError("synthetic post-exchange cut")

    monkeypatch.setattr(cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", cut_after_exchange)
    with pytest.raises(FeatureCacheError, match="post-linearization interruption"):
        cache_module.atomic_json(target, desired, expected_prior_payloads=(prior_a,))
    monkeypatch.setattr(cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", None)

    prepared = cache_module.atomic_prepared_path(target)
    admitted_prior = tmp_path / "descriptor-admitted-prior-a.json"
    os.replace(prepared, admitted_prior)
    prepared.write_bytes(prior_b)
    foreign_snapshot = cache_module.read_bound_file(prepared, role="test forged foreign prior")

    transaction = cache_module._atomic_transaction_paths(target)[0]
    original_transaction = tmp_path / "descriptor-admitted-transaction.json"
    os.replace(transaction, original_transaction)
    forged = json.loads(original_transaction.read_text(encoding="utf-8"))
    forged["prior_bytes"] = len(prior_b)
    forged["prior_sha256"] = hashlib.sha256(prior_b).hexdigest()
    forged["prior_file_identity"] = list(foreign_snapshot.file_identity)
    source_row = next(row for row in forged["admitted_scratch"] if row["role"] == "DISPLACED_PRIOR_SOURCE")
    source_row["bytes"] = len(prior_b)
    source_row["sha256"] = hashlib.sha256(prior_b).hexdigest()
    source_row["file_identity"] = list(foreign_snapshot.file_identity)
    transaction.write_bytes(cache_module.canonical_json_bytes(forged) + b"\n")
    before = _flat_tree_snapshot(tmp_path)

    with pytest.raises(FeatureCacheError, match="not authorized by the current consumer"):
        cache_module.atomic_json(target, desired, expected_prior_payloads=(prior_a,))

    assert _flat_tree_snapshot(tmp_path) == before
    assert admitted_prior.read_bytes() == prior_a


def test_atomic_retained_hash_valid_orphan_blocks_idempotent_target_byte_identically(
    tmp_path: Path,
) -> None:
    target = tmp_path / PROGRESS_NAME
    desired = {"state": "desired"}
    expected = cache_module.canonical_json_bytes(desired) + b"\n"
    cache_module.atomic_json(target, desired)

    orphan = cache_module.atomic_prepared_path(target)
    orphan.write_bytes(b"hash-valid but transaction-foreign retained payload")
    orphan_snapshot = cache_module.read_bound_file(orphan, role="test orphan atomic scratch")
    retained_orphan = cache_module.retain_bound_file(
        orphan,
        orphan_snapshot,
        role="test orphan atomic scratch",
    )
    before = _flat_tree_snapshot(tmp_path)

    with pytest.raises(FeatureCacheError, match="orphaned, duplicated, or role-ambiguous"):
        cache_module.atomic_json(target, desired)

    assert _flat_tree_snapshot(tmp_path) == before
    assert (
        cache_module.validate_retained_file(
            retained_orphan,
            role="test orphan retained result",
        ).payload
        != expected
    )


def test_atomic_retained_duplicate_complete_transaction_blocks_provenance(
    tmp_path: Path,
) -> None:
    target = tmp_path / PROGRESS_NAME
    desired = {"state": "desired"}
    expected = cache_module.canonical_json_bytes(desired) + b"\n"
    cache_module.atomic_json(target, desired)
    retained_transaction = next(
        path
        for path in tmp_path.iterdir()
        if cache_module.is_retained_name(path.name)
        and cache_module.retained_original_name(path.name).startswith(
            f".{target.name}{cache_module.ATOMIC_TRANSACTION_SUFFIX}-"
        )
    )
    transaction_original = cache_module.retained_original_name(retained_transaction.name)
    injected = tmp_path / transaction_original
    injected.write_bytes(retained_transaction.read_bytes())
    injected_snapshot = cache_module.read_bound_file(injected, role="test duplicate transaction")
    cache_module.retain_bound_file(injected, injected_snapshot, role="test duplicate transaction")
    before = _flat_tree_snapshot(tmp_path)

    with pytest.raises(FeatureCacheError, match="duplicate complete transaction evidence"):
        cache_module.validate_atomic_transaction_custody(target, desired_payload=expected)

    assert _flat_tree_snapshot(tmp_path) == before


@pytest.mark.parametrize("retirement_cut", ["before", "after"])
def test_fresh_atomic_transaction_retirement_cuts_retry_without_byte_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retirement_cut: str,
) -> None:
    target = tmp_path / "recovery_manifest.json"
    desired = {"state": f"retirement-cut-{retirement_cut}"}

    def cut_transaction_retirement(source: Path, _destination: Path) -> None:
        if cache_module.ATOMIC_TRANSACTION_RE.fullmatch(source.name) is not None:
            raise OSError(f"synthetic {retirement_cut} transaction retirement cut")

    hook_name = (
        "_MOVE_PATH_NOREPLACE_TEST_HOOK" if retirement_cut == "before" else "_MOVE_PATH_NOREPLACE_POST_MOVE_TEST_HOOK"
    )
    monkeypatch.setattr(cache_module, hook_name, cut_transaction_retirement)
    with pytest.raises(FeatureCacheError, match=r"interruption|post-move mismatch"):
        cache_module.atomic_json(target, desired)

    expected = cache_module.canonical_json_bytes(desired) + b"\n"
    assert target.read_bytes() == expected
    assert cache_module._atomic_transaction_paths(target)

    monkeypatch.setattr(cache_module, hook_name, None)
    cache_module.atomic_json(target, desired)

    assert target.read_bytes() == expected
    assert not cache_module._atomic_transaction_paths(target)
    cache_module.validate_atomic_transaction_custody(target, desired_payload=expected)


def test_atomic_extra_retained_generation_blocks_before_active_cleanup_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / PROGRESS_NAME
    prior = cache_module.canonical_json_bytes({"state": "prior"}) + b"\n"
    desired = {"state": "desired"}
    target.write_bytes(prior)

    def cut_after_exchange(_source: Path, _destination: Path) -> None:
        raise OSError("synthetic post-exchange cut")

    monkeypatch.setattr(cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", cut_after_exchange)
    with pytest.raises(FeatureCacheError, match="post-linearization interruption"):
        cache_module.atomic_json(target, desired, expected_prior_payloads=(prior,))
    monkeypatch.setattr(cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", None)

    extra = cache_module.atomic_generation_path(target, 99)
    extra.write_bytes(b"independently retained foreign generation")
    extra_snapshot = cache_module.read_bound_file(extra, role="test extra retained generation")
    cache_module.retain_bound_file(extra, extra_snapshot, role="test extra retained generation")
    before = _flat_tree_snapshot(tmp_path)

    with pytest.raises(FeatureCacheError, match="orphaned, duplicated, or role-ambiguous"):
        cache_module.atomic_json(target, desired, expected_prior_payloads=(prior,))

    assert _flat_tree_snapshot(tmp_path) == before
    assert cache_module._atomic_transaction_paths(target)


def test_fresh_atomic_contradictory_complete_record_blocks_before_namespace_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "recovery_manifest.json"
    desired = {"state": "contradictory-complete-record"}

    def cut_before_publication(_source: Path, _destination: Path) -> None:
        raise OSError("synthetic pre-publication cut")

    monkeypatch.setattr(cache_module, "_ATOMIC_COMMIT_TEST_HOOK", cut_before_publication)
    with pytest.raises(FeatureCacheError, match="pre-commit interruption"):
        cache_module.atomic_json(target, desired)
    monkeypatch.setattr(cache_module, "_ATOMIC_COMMIT_TEST_HOOK", None)

    first = cache_module._atomic_transaction_paths(target)[0]
    forged = json.loads(first.read_text(encoding="utf-8"))
    forged_identity = list(forged["designated_source_identity"])
    forged_identity[1] += 1
    forged["designated_source_identity"] = forged_identity
    source_row = next(row for row in forged["admitted_scratch"] if row["role"] == "DESIGNATED_DESIRED_SOURCE")
    source_row["file_identity"] = forged_identity
    second = cache_module.atomic_transaction_path(target, 1)
    second.write_bytes(cache_module.canonical_json_bytes(forged) + b"\n")
    before = _flat_tree_snapshot(tmp_path)

    with pytest.raises(FeatureCacheError, match="contradictory complete"):
        cache_module.atomic_json(target, desired)

    assert _flat_tree_snapshot(tmp_path) == before
    assert not target.exists()


@pytest.mark.parametrize("retain_exact_prior", [False, True])
def test_post_exchange_retained_scratch_requires_exact_displaced_prior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retain_exact_prior: bool,
) -> None:
    target = tmp_path / PROGRESS_NAME
    prior = cache_module.canonical_json_bytes({"state": "authorized-prior"}) + b"\n"
    target.write_bytes(prior)
    desired = {"state": "desired"}
    prepared = cache_module.atomic_prepared_path(target)

    def cut_after_exchange(_source: Path, _destination: Path) -> None:
        raise OSError("synthetic post-exchange cut")

    monkeypatch.setattr(cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", cut_after_exchange)
    with pytest.raises(FeatureCacheError, match="post-linearization interruption"):
        cache_module.atomic_json(target, desired, expected_prior_payloads=(prior,))
    monkeypatch.setattr(cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", None)

    if retain_exact_prior:
        retained_snapshot = cache_module.read_bound_file(prepared, role="test exact displaced prior")
    else:
        displaced = tmp_path / "descriptor-admitted-prior.json"
        os.replace(prepared, displaced)
        prepared.write_bytes(cache_module.canonical_json_bytes({"state": "foreign-prior"}) + b"\n")
        retained_snapshot = cache_module.read_bound_file(prepared, role="test foreign retained scratch")
    retained = cache_module.retain_bound_file(
        prepared,
        retained_snapshot,
        role="test post-exchange retained scratch",
    )
    before = _flat_tree_snapshot(tmp_path)

    if retain_exact_prior:
        foreign_authorization = cache_module.canonical_json_bytes({"state": "different-authorized-prior"}) + b"\n"
        with pytest.raises(FeatureCacheError, match="not authorized by the current consumer"):
            cache_module.atomic_json(
                target,
                desired,
                expected_prior_payloads=(foreign_authorization,),
            )
        assert _flat_tree_snapshot(tmp_path) == before
        cache_module.atomic_json(target, desired, expected_prior_payloads=(prior,))
        assert target.read_bytes() == cache_module.canonical_json_bytes(desired) + b"\n"
        assert not cache_module._atomic_transaction_paths(target)
    else:
        with pytest.raises(FeatureCacheError, match="one exact active/retained realization"):
            cache_module.atomic_json(target, desired, expected_prior_payloads=(prior,))
        assert _flat_tree_snapshot(tmp_path) == before
        assert cache_module.validate_retained_file(retained, role="test foreign retained result").payload != prior


def test_fix20_retired_unfinalized_exchange_requires_exact_prior_then_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / PROGRESS_NAME
    prior_a = cache_module.canonical_json_bytes({"state": "prior-a"}) + b"\n"
    desired = {"state": "desired-b"}
    payload_b = cache_module.canonical_json_bytes(desired) + b"\n"
    target.write_bytes(prior_a)

    def cut_after_exchange(_source: Path, _destination: Path) -> None:
        raise OSError("synthetic post-exchange cut")

    monkeypatch.setattr(cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", cut_after_exchange)
    with pytest.raises(FeatureCacheError, match="post-linearization interruption"):
        cache_module.atomic_json(target, desired, expected_prior_payloads=(prior_a,))
    monkeypatch.setattr(cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", None)

    prepared = cache_module.atomic_prepared_path(target)
    prepared_snapshot = cache_module.read_bound_file(prepared, role="test displaced prior retirement")
    cache_module.retain_bound_file(prepared, prepared_snapshot, role="test displaced prior retirement")
    transaction = cache_module._atomic_transaction_paths(target)[0]
    transaction_snapshot = cache_module.read_bound_file(transaction, role="test transaction retirement")
    cache_module.retain_bound_file(transaction, transaction_snapshot, role="test transaction retirement")
    assert target.read_bytes() == payload_b
    assert not cache_module._atomic_completion_paths(target)[0]

    before = _flat_tree_snapshot(tmp_path)
    with pytest.raises(FeatureCacheError, match="prior is not authorized by the current consumer"):
        cache_module.atomic_json(target, desired, expected_prior_payloads=(payload_b,))
    assert _flat_tree_snapshot(tmp_path) == before

    cache_module.atomic_json(target, desired, expected_prior_payloads=(prior_a,))
    completions, _generations = cache_module._atomic_completion_paths(target)
    assert len(completions) == 1
    cache_module.validate_atomic_transaction_custody(target, desired_payload=payload_b)


def test_fix20_unresolved_exchange_binds_consumer_authorization_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / PROGRESS_NAME
    prior = cache_module.canonical_json_bytes({"state": "authorized-prior"}) + b"\n"
    desired = {"state": "authorized-desired"}
    desired_payload = cache_module.canonical_json_bytes(desired) + b"\n"
    authorization = hashlib.sha256(b"exact external authorization").hexdigest()
    foreign_authorization = hashlib.sha256(b"foreign external authorization").hexdigest()
    target.write_bytes(prior)

    def cut_after_exchange(_source: Path, _destination: Path) -> None:
        raise OSError("synthetic authorization restart")

    monkeypatch.setattr(cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", cut_after_exchange)
    with pytest.raises(FeatureCacheError, match="post-linearization interruption"):
        cache_module.atomic_json(
            target,
            desired,
            expected_prior_payloads=(prior,),
            consumer_authorization_sha256=authorization,
        )
    monkeypatch.setattr(cache_module, "_ATOMIC_POST_EXCHANGE_TEST_HOOK", None)

    for supplied in (None, foreign_authorization):
        before = _flat_tree_snapshot(tmp_path)
        with pytest.raises(FeatureCacheError, match="authorization is missing or foreign"):
            cache_module.atomic_json(
                target,
                desired,
                expected_prior_payloads=(prior,),
                consumer_authorization_sha256=supplied,
            )
        assert _flat_tree_snapshot(tmp_path) == before

    cache_module.atomic_json(
        target,
        desired,
        expected_prior_payloads=(prior,),
        consumer_authorization_sha256=authorization,
    )
    cache_module.validate_atomic_transaction_custody(target, desired_payload=desired_payload)


@pytest.mark.parametrize("phase", ["before_completion_write", "after_completion_fsync"])
def test_fix20_completion_construction_cuts_are_non_authority_and_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    target = tmp_path / PROGRESS_NAME
    prior = cache_module.canonical_json_bytes({"state": "prior"}) + b"\n"
    desired = {"state": f"desired-{phase}"}
    desired_payload = cache_module.canonical_json_bytes(desired) + b"\n"
    target.write_bytes(prior)

    def cut_completion(observed_phase: str, _construction: Path) -> None:
        if observed_phase == phase:
            raise OSError(f"synthetic {phase} cut")

    monkeypatch.setattr(cache_module, "_ATOMIC_COMPLETION_TEST_HOOK", cut_completion)
    with pytest.raises(FeatureCacheError, match=r"completion.*interruption"):
        cache_module.atomic_json(target, desired, expected_prior_payloads=(prior,))
    assert target.read_bytes() == desired_payload
    assert cache_module._atomic_completion_paths(target)[1]

    monkeypatch.setattr(cache_module, "_ATOMIC_COMPLETION_TEST_HOOK", None)
    cache_module.atomic_json(target, desired, expected_prior_payloads=(prior,))
    cache_module.validate_atomic_transaction_custody(target, desired_payload=desired_payload)


@pytest.mark.parametrize(
    "field",
    [
        "transaction_sha256",
        "target_basename",
        "parent_identity",
        "desired_file_identity",
        "desired_sha256",
        "prior_file_identity",
        "prior_sha256",
        "admitted_rows_sha256",
    ],
)
def test_fix20_completion_field_corruption_refuses_byte_identically(
    tmp_path: Path,
    field: str,
) -> None:
    target = tmp_path / PROGRESS_NAME
    prior = cache_module.canonical_json_bytes({"state": "prior"}) + b"\n"
    desired = {"state": "desired"}
    desired_payload = cache_module.canonical_json_bytes(desired) + b"\n"
    target.write_bytes(prior)
    cache_module.atomic_json(target, desired, expected_prior_payloads=(prior,))
    completion = cache_module._atomic_completion_paths(target)[0][0]
    corrupted = json.loads(completion.read_text(encoding="utf-8"))
    if field in {"transaction_sha256", "desired_sha256", "prior_sha256", "admitted_rows_sha256"}:
        corrupted[field] = "0" * 64
    elif field == "target_basename":
        corrupted[field] = "foreign-target.json"
    else:
        corrupted[field][1] += 1
    completion.write_bytes(cache_module.canonical_json_bytes(corrupted) + b"\n")
    before = _flat_tree_snapshot(tmp_path)

    with pytest.raises(FeatureCacheError, match="completion"):
        cache_module.validate_atomic_transaction_custody(target, desired_payload=desired_payload)
    assert _flat_tree_snapshot(tmp_path) == before


@pytest.mark.parametrize("damage", ["missing", "partial", "duplicate"])
def test_fix20_missing_partial_or_duplicate_completion_refuses_byte_identically(
    tmp_path: Path,
    damage: str,
) -> None:
    target = tmp_path / PROGRESS_NAME
    prior = cache_module.canonical_json_bytes({"state": "prior"}) + b"\n"
    desired = {"state": "desired"}
    desired_payload = cache_module.canonical_json_bytes(desired) + b"\n"
    target.write_bytes(prior)
    cache_module.atomic_json(target, desired, expected_prior_payloads=(prior,))
    completion = cache_module._atomic_completion_paths(target)[0][0]
    completion_payload = completion.read_bytes()

    if damage == "missing":
        os.replace(completion, tmp_path / "operator-preserved-completion.json")
        expected_error = "prior is not authorized"
    elif damage == "partial":
        completion.write_bytes(completion_payload[: len(completion_payload) // 2])
        expected_error = "partial payload occupies atomic completion"
    else:
        preserved = tmp_path / "operator-preserved-completion.json"
        os.replace(completion, preserved)
        completion.write_bytes(preserved.read_bytes())
        duplicate_snapshot = cache_module.read_bound_file(completion, role="test duplicate completion")
        cache_module.retain_bound_file(completion, duplicate_snapshot, role="test duplicate completion")
        os.replace(preserved, completion)
        expected_error = "completion proof is duplicated"
    before = _flat_tree_snapshot(tmp_path)

    with pytest.raises(FeatureCacheError, match=expected_error):
        cache_module.validate_atomic_transaction_custody(target, desired_payload=desired_payload)
    assert _flat_tree_snapshot(tmp_path) == before


def test_fix20_cross_transaction_completion_reuse_refuses_byte_identically(tmp_path: Path) -> None:
    target = tmp_path / PROGRESS_NAME
    payload_a = cache_module.canonical_json_bytes({"state": "a"}) + b"\n"
    value_b = {"state": "b"}
    payload_b = cache_module.canonical_json_bytes(value_b) + b"\n"
    value_c = {"state": "c"}
    payload_c = cache_module.canonical_json_bytes(value_c) + b"\n"
    target.write_bytes(payload_a)
    cache_module.atomic_json(target, value_b, expected_prior_payloads=(payload_a,))
    cache_module.atomic_json(target, value_c, expected_prior_payloads=(payload_b,))
    completions, _generations = cache_module._atomic_completion_paths(target)
    assert len(completions) == 2
    by_transaction = {
        json.loads(completion.read_text(encoding="utf-8"))["transaction_sha256"]: completion
        for completion in completions
    }
    ordered = sorted(by_transaction)
    by_transaction[ordered[1]].write_bytes(by_transaction[ordered[0]].read_bytes())
    before = _flat_tree_snapshot(tmp_path)

    with pytest.raises(FeatureCacheError, match="completion filename digest mismatch"):
        cache_module.validate_atomic_transaction_custody(target, desired_payload=payload_c)
    assert _flat_tree_snapshot(tmp_path) == before


@pytest.mark.parametrize("orphan_role", ["prepared", "generation", "transaction"])
def test_fix20_absent_target_retained_atomic_orphan_refuses_before_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    orphan_role: str,
) -> None:
    target = tmp_path / "recovery_manifest.json"
    orphan = {
        "prepared": cache_module.atomic_prepared_path(target),
        "generation": cache_module.atomic_generation_path(target, 7),
        "transaction": cache_module.atomic_transaction_path(target, 7),
    }[orphan_role]
    orphan.write_bytes(f"retained-{orphan_role}-orphan".encode())
    orphan_snapshot = cache_module.read_bound_file(orphan, role="test retained atomic orphan")
    cache_module.retain_bound_file(orphan, orphan_snapshot, role="test retained atomic orphan")
    hook_calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        cache_module, "_ATOMIC_COMMIT_TEST_HOOK", lambda source, path: hook_calls.append((source, path))
    )
    before = _flat_tree_snapshot(tmp_path)

    with pytest.raises(FeatureCacheError, match="target-absent atomic namespace"):
        cache_module.atomic_json(target, {"state": "desired"})
    assert _flat_tree_snapshot(tmp_path) == before
    assert not hook_calls
    assert not target.exists()


def test_fix20_late_retained_sibling_blocks_at_final_publication_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "recovery_manifest.json"
    desired = {"state": "desired"}

    def inject_retained_sibling(_source: Path, destination: Path) -> None:
        sibling = cache_module.atomic_generation_path(destination, 77)
        sibling.write_bytes(b"late-retained-foreign-sibling")
        snapshot = cache_module.read_bound_file(sibling, role="test late retained sibling")
        cache_module.retain_bound_file(sibling, snapshot, role="test late retained sibling")

    monkeypatch.setattr(cache_module, "_ATOMIC_COMMIT_TEST_HOOK", inject_retained_sibling)
    with pytest.raises(FeatureCacheError, match="final publication boundary"):
        cache_module.atomic_json(target, desired)
    assert not target.exists()
    assert cache_module.atomic_prepared_path(target).is_file()
    assert cache_module._atomic_transaction_paths(target)


@pytest.mark.parametrize("target_state", ["fresh", "existing"])
@pytest.mark.parametrize(
    "late_role",
    ["generation", "transaction", "completion", "completion_generation"],
)
def test_fix21_late_active_role_refuses_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_state: str,
    late_role: str,
) -> None:
    target = tmp_path / "recovery_manifest.json"
    prior = cache_module.canonical_json_bytes({"state": "prior"}) + b"\n"
    prior_inode: int | None = None
    if target_state == "existing":
        target.write_bytes(prior)
        prior_inode = target.stat().st_ino

    def inject_active_role(_source: Path, destination: Path) -> None:
        transaction_digest = "f" * 64
        sibling = {
            "generation": cache_module.atomic_generation_path(destination, 77),
            "transaction": cache_module.atomic_transaction_path(destination, 77),
            "completion": cache_module.atomic_completion_path(destination, transaction_digest),
            "completion_generation": cache_module.atomic_completion_generation_path(
                destination,
                transaction_digest,
                77,
            ),
        }[late_role]
        sibling.write_bytes(f"late-active-{late_role}".encode())

    monkeypatch.setattr(cache_module, "_ATOMIC_COMMIT_TEST_HOOK", inject_active_role)
    with pytest.raises(FeatureCacheError):
        cache_module.atomic_json(
            target,
            {"state": "desired"},
            expected_prior_payloads=((prior,) if target_state == "existing" else ()),
        )

    if target_state == "fresh":
        assert not target.exists()
    else:
        assert target.read_bytes() == prior
        assert target.stat().st_ino == prior_inode
    assert any(f"late-active-{late_role}".encode() == path.read_bytes() for path in tmp_path.iterdir())


def test_fix21_role_injected_after_transaction_write_is_refused_before_first_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "recovery_manifest.json"
    real_write = cache_module._write_atomic_transaction_generation

    def write_then_inject(path: Path, record: dict[str, object]) -> Path:
        transaction = real_write(path, record)
        cache_module.atomic_generation_path(path, 91).write_bytes(b"between-record-and-boundary")
        return transaction

    monkeypatch.setattr(cache_module, "_write_atomic_transaction_generation", write_then_inject)
    with pytest.raises(FeatureCacheError):
        cache_module.atomic_json(target, {"state": "desired"})
    assert not target.exists()
    assert cache_module.atomic_generation_path(target, 91).read_bytes() == b"between-record-and-boundary"


@pytest.mark.parametrize("partial", [b"", b'{"schema"'])
def test_fix21_absent_target_partial_only_transaction_refuses_byte_identically(
    tmp_path: Path,
    partial: bytes,
) -> None:
    target = tmp_path / "recovery_manifest.json"
    transaction = cache_module.atomic_transaction_path(target, 0)
    transaction.write_bytes(partial)
    before = _flat_tree_snapshot(tmp_path)

    with pytest.raises(FeatureCacheError, match=r"partial|complete fresh authority"):
        cache_module.atomic_json(target, {"state": "desired"})

    assert _flat_tree_snapshot(tmp_path) == before
    assert not target.exists()


@pytest.mark.parametrize("source_state", ["missing", "substituted"])
def test_fix21_absent_target_complete_transaction_requires_exact_designated_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_state: str,
) -> None:
    target = tmp_path / "recovery_manifest.json"
    desired = {"state": "desired"}
    desired_payload = cache_module.canonical_json_bytes(desired) + b"\n"

    def cut_before_publication(_source: Path, _destination: Path) -> None:
        raise OSError("synthetic transaction/source custody cut")

    monkeypatch.setattr(cache_module, "_ATOMIC_COMMIT_TEST_HOOK", cut_before_publication)
    with pytest.raises(FeatureCacheError, match="pre-commit interruption"):
        cache_module.atomic_json(target, desired)
    monkeypatch.setattr(cache_module, "_ATOMIC_COMMIT_TEST_HOOK", None)

    prepared = cache_module.atomic_prepared_path(target)
    preserved = tmp_path / "preserved-designated-source"
    os.replace(prepared, preserved)
    if source_state == "substituted":
        prepared.write_bytes(desired_payload)
    before = _flat_tree_snapshot(tmp_path)

    with pytest.raises(FeatureCacheError, match=r"designated source|identity/content drift"):
        cache_module.atomic_json(target, desired)

    assert _flat_tree_snapshot(tmp_path) == before
    assert not target.exists()


@pytest.mark.parametrize("target_state", ["fresh", "existing"])
def test_fix22_empty_transaction_injected_after_prepublication_refuses_before_writer_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_state: str,
) -> None:
    target = tmp_path / "recovery_manifest.json"
    prior = cache_module.canonical_json_bytes({"state": "prior"}) + b"\n"
    if target_state == "existing":
        target.write_bytes(prior)
    real_validate = cache_module._validate_atomic_prepublication_namespace
    post_injection: dict[str, dict[str, tuple[int, int, int, int, bytes | None]]] = {}

    def validate_then_inject(*args: object, **kwargs: object) -> object:
        boundary = real_validate(*args, **kwargs)
        cache_module.atomic_transaction_path(target, 71).write_bytes(b"")
        post_injection["tree"] = _flat_tree_snapshot(tmp_path)
        return boundary

    monkeypatch.setattr(cache_module, "_validate_atomic_prepublication_namespace", validate_then_inject)
    with pytest.raises(FeatureCacheError, match=r"partial|prepublication"):
        cache_module.atomic_json(
            target,
            {"state": "desired"},
            expected_prior_payloads=((prior,) if target_state == "existing" else ()),
        )

    assert _flat_tree_snapshot(tmp_path) == post_injection["tree"]
    if target_state == "fresh":
        assert not target.exists()
    else:
        assert target.read_bytes() == prior
        assert target.stat().st_ino == post_injection["tree"][target.name][1]


def test_fix22_strict_prefix_transaction_injected_after_prepublication_is_not_adopted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "recovery_manifest.json"
    real_validate = cache_module._validate_atomic_prepublication_namespace
    post_injection: dict[str, dict[str, tuple[int, int, int, int, bytes | None]]] = {}

    def validate_then_inject(*args: object, **kwargs: object) -> object:
        boundary = real_validate(*args, **kwargs)
        cache_module.atomic_transaction_path(target, 72).write_bytes(b'{"schema"')
        post_injection["tree"] = _flat_tree_snapshot(tmp_path)
        return boundary

    monkeypatch.setattr(cache_module, "_validate_atomic_prepublication_namespace", validate_then_inject)
    with pytest.raises(FeatureCacheError, match=r"partial|prepublication"):
        cache_module.atomic_json(target, {"state": "desired"})

    assert _flat_tree_snapshot(tmp_path) == post_injection["tree"]
    assert not target.exists()


def test_fix22_active_scratch_injected_after_prepublication_is_not_adopted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "recovery_manifest.json"
    real_validate = cache_module._validate_atomic_prepublication_namespace
    post_injection: dict[str, dict[str, tuple[int, int, int, int, bytes | None]]] = {}

    def validate_then_inject(*args: object, **kwargs: object) -> object:
        boundary = real_validate(*args, **kwargs)
        cache_module.atomic_generation_path(target, 73).write_bytes(b"")
        post_injection["tree"] = _flat_tree_snapshot(tmp_path)
        return boundary

    monkeypatch.setattr(cache_module, "_validate_atomic_prepublication_namespace", validate_then_inject)
    with pytest.raises(FeatureCacheError, match="changed after prepublication"):
        cache_module.atomic_json(target, {"state": "desired"})

    assert _flat_tree_snapshot(tmp_path) == post_injection["tree"]
    assert not target.exists()


@pytest.mark.parametrize("injected_kind", ["empty", "strict-prefix"])
def test_fix22_transaction_injected_after_record_write_refuses_before_first_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injected_kind: str,
) -> None:
    target = tmp_path / "recovery_manifest.json"
    real_write = cache_module._write_atomic_transaction_generation
    post_injection: dict[str, dict[str, tuple[int, int, int, int, bytes | None]]] = {}

    def write_then_inject(path: Path, record: dict[str, object]) -> Path:
        transaction = real_write(path, record)
        complete_payload = cache_module.canonical_json_bytes(record) + b"\n"
        injected_payload = b"" if injected_kind == "empty" else complete_payload[: len(complete_payload) // 2]
        cache_module.atomic_transaction_path(path, 74).write_bytes(injected_payload)
        post_injection["tree"] = _flat_tree_snapshot(tmp_path)
        return transaction

    monkeypatch.setattr(cache_module, "_write_atomic_transaction_generation", write_then_inject)
    with pytest.raises(FeatureCacheError, match="outside authorized construction"):
        cache_module.atomic_json(target, {"state": "desired"})

    assert _flat_tree_snapshot(tmp_path) == post_injection["tree"]
    assert not target.exists()


def test_fix22_complete_fresh_restart_with_preexisting_strict_prefix_remains_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "recovery_manifest.json"
    desired = {"state": "resumable-complete-plus-prefix"}
    desired_payload = cache_module.canonical_json_bytes(desired) + b"\n"

    def cut_before_publication(_source: Path, _destination: Path) -> None:
        raise OSError("synthetic complete transaction restart")

    monkeypatch.setattr(cache_module, "_ATOMIC_COMMIT_TEST_HOOK", cut_before_publication)
    with pytest.raises(FeatureCacheError, match="pre-commit interruption"):
        cache_module.atomic_json(target, desired)
    monkeypatch.setattr(cache_module, "_ATOMIC_COMMIT_TEST_HOOK", None)

    prepared = cache_module.atomic_prepared_path(target)
    prepared_inode = prepared.stat().st_ino
    complete_transaction = cache_module._atomic_transaction_paths(target)[0]
    complete_payload = complete_transaction.read_bytes()
    prefix = cache_module.atomic_transaction_path(target, 75)
    prefix.write_bytes(complete_payload[: len(complete_payload) // 2])

    cache_module.atomic_json(target, desired)

    assert target.read_bytes() == desired_payload
    assert target.stat().st_ino == prepared_inode
    assert not cache_module._atomic_transaction_paths(target)
    assert not cache_module._atomic_generation_paths(target)
    assert not prepared.exists()


def test_source_file_row_rejects_final_symlink_parent_symlink_and_hardlink(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source-custody")

    final_alias = tmp_path / "final-alias.bin"
    final_alias.symlink_to(source)
    with pytest.raises(FeatureCacheError, match="non-symlink regular file"):
        cache_module.source_file_row(final_alias)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    parent_source = real_parent / "parent-source.bin"
    parent_source.write_bytes(b"parent-source")
    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(FeatureCacheError, match="traverses a symlink"):
        cache_module.source_file_row(parent_alias / parent_source.name)

    hardlink = tmp_path / "source-hardlink.bin"
    os.link(source, hardlink)
    with pytest.raises(FeatureCacheError, match="exactly one hard link"):
        cache_module.source_file_row(source)

    assert source.read_bytes() == b"source-custody"
    assert parent_source.read_bytes() == b"parent-source"


def test_source_file_row_rejects_same_inode_content_change_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    original = b"stable-source-content"
    source.write_bytes(original)
    real_read = cache_module.os.read
    changed = False

    def mutate_after_read(descriptor: int, count: int) -> bytes:
        nonlocal changed
        payload = real_read(descriptor, count)
        if payload and not changed:
            changed = True
            source.write_bytes(b"changed-source-bytes")
        return payload

    monkeypatch.setattr(cache_module.os, "read", mutate_after_read)
    with pytest.raises(FeatureCacheError, match="content identity changed"):
        cache_module.source_file_row(source)
    assert source.read_bytes() == b"changed-source-bytes"


class _TinySegNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segmentation_head = torch.nn.Sequential(torch.nn.Conv2d(3, 5, kernel_size=1, bias=True))

    def preprocess_input(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, 0] if value.ndim == 5 else value

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.segmentation_head(value)


def _tiny_segnet_payload(path: Path, *, offset: float = 0.0) -> tuple[_TinySegNet, bytes]:
    from safetensors.torch import save

    model = _TinySegNet()
    with torch.no_grad():
        weight = model.segmentation_head[0].weight
        bias = model.segmentation_head[0].bias
        weight.copy_(torch.arange(weight.numel(), dtype=torch.float32).reshape_as(weight) / 32.0 + offset)
        assert bias is not None
        bias.copy_(torch.arange(bias.numel(), dtype=torch.float32) / 16.0 + offset)
    payload = save(model.state_dict())
    path.write_bytes(payload)
    return model.eval(), payload


class _ExtractionStorageBoundaryReached(RuntimeError):
    pass


@pytest.mark.parametrize(
    "changed_role",
    [
        "gt_n600_npz",
        "segnet_weights",
        "executed_modules_py",
        "executed_frame_utils_py",
        "executed_tac_scorer_py",
        "executed_factorization_module_py",
        "extractor_tool",
        "cache_module",
        None,
    ],
)
def test_run_extraction_loader_window_refuses_each_source_role_before_storage_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_role: str | None,
) -> None:
    upstream = tmp_path / "upstream"
    (upstream / "models").mkdir(parents=True)
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "operator-sentinel.bin"
    sentinel.write_bytes(b"unchanged")
    role_paths = {
        "gt_n600_npz": tmp_path / "gt.npz",
        "segnet_weights": upstream / "models" / "segnet.safetensors",
        "executed_modules_py": upstream / "modules.py",
        "executed_frame_utils_py": upstream / "frame_utils.py",
        "executed_tac_scorer_py": tmp_path / "scorer.py",
        "executed_factorization_module_py": tmp_path / "factorization.py",
        "extractor_tool": tmp_path / "extractor.py",
        "cache_module": tmp_path / "cache.py",
    }
    for role, path in role_paths.items():
        path.write_bytes(f"stable-{role}\n".encode())

    monkeypatch.setattr(extractor_module, "EXPECTED_PAIRS", 2)
    monkeypatch.setattr(extractor_module, "EXPECTED_CAMERA_SHAPE", (2, 2, 2, 3))
    monkeypatch.setattr(
        extractor_module,
        "_source_bindings",
        lambda _upstream, _gt: extractor_module._FrozenScorerSnapshot(
            rows={role: cache_module.source_file_row(path) for role, path in role_paths.items()},
            segnet_payload=role_paths["segnet_weights"].read_bytes(),
        ),
    )
    monkeypatch.setattr(
        extractor_module,
        "_assert_real_governed_admission",
        lambda **kwargs: {
            "source_custody": {
                "governed_profile_admission": _source("/admission", b"a"),
                "safe_run": _source("/safe-run", b"s"),
                "admission_guard": _source("/guard", b"g"),
            }
        },
    )
    monkeypatch.setattr(extractor_module, "open_gt_f1_stored_memmap", lambda _path: np.zeros((2, 2, 2, 3), np.uint8))
    monkeypatch.setattr(extractor_module, "prepend_paths", lambda _path: None)
    monkeypatch.setattr(extractor_module, "_require_exact_module_file", lambda module, expected, role: expected)
    monkeypatch.setattr(extractor_module.torch, "set_num_threads", lambda _value: None)
    monkeypatch.setattr(extractor_module.torch, "set_num_interop_threads", lambda _value: None)
    monkeypatch.setattr(extractor_module.torch, "use_deterministic_algorithms", lambda _value: None)
    head = torch.nn.Conv2d(4, 5, 1)
    model = SimpleNamespace(segmentation_head=[head])
    target = SimpleNamespace(rank=4)
    factorization = SimpleNamespace(
        __file__=str(role_paths["executed_factorization_module_py"]),
        affine_head_to_power_diagram=lambda *args: SimpleNamespace(
            target=target,
            quotient_basis=np.eye(4, dtype=np.float64),
        ),
    )
    scorer_module = SimpleNamespace(
        __file__=str(role_paths["executed_tac_scorer_py"]),
        load_default_segnet=lambda *args, **kwargs: model,
    )
    mutated = False

    def controlled_loader(module_name: str, source_row: object, *, role: str) -> object:
        nonlocal mutated
        if changed_role is not None and not mutated:
            mutated = True
            role_paths[changed_role].write_bytes(role_paths[changed_role].read_bytes() + b"loader-drift")
        if module_name.endswith("power_diagram_witness"):
            return factorization
        if module_name == "frame_utils":
            return frame_utils
        if module_name == "modules":
            return modules
        return scorer_module

    monkeypatch.setattr(extractor_module, "_load_source_module_from_snapshot", controlled_loader)
    modules = SimpleNamespace(__file__=str(role_paths["executed_modules_py"]))
    frame_utils = SimpleNamespace(
        __file__=str(role_paths["executed_frame_utils_py"]),
        segnet_model_input_size=(2, 2),
    )
    monkeypatch.setitem(sys.modules, "modules", modules)
    monkeypatch.setitem(sys.modules, "frame_utils", frame_utils)
    monkeypatch.setattr(
        extractor_module,
        "_load_segnet_from_admitted_payload",
        lambda _modules, _payload: model,
    )
    storage_calls = 0

    def stop_at_storage(*args: object, **kwargs: object) -> object:
        nonlocal storage_calls
        storage_calls += 1
        raise _ExtractionStorageBoundaryReached

    monkeypatch.setattr(extractor_module, "storage_preflight", stop_at_storage)
    tokens = [
        "--gt-cache",
        str(role_paths["gt_n600_npz"]),
        "--upstream-root",
        str(upstream),
        "--output-root",
        str(output),
        "--max-frames",
        "1",
        "--rss-cap-mb",
        "512",
        "--timeout-seconds",
        "60",
        "--allow-local-output-for-tests",
    ]
    args = extractor_module._parse_args(tokens)
    exact = [sys.executable, str(Path(extractor_module.__file__).resolve()), *tokens]

    if changed_role is None:
        with pytest.raises(_ExtractionStorageBoundaryReached):
            extractor_module.run_extraction(args, exact_argv=exact)
        assert storage_calls == 1
    else:
        with pytest.raises(ExtractionError, match=changed_role):
            extractor_module.run_extraction(args, exact_argv=exact)
        assert storage_calls == 0
    assert {path.name for path in output.iterdir()} == {sentinel.name}
    assert sentinel.read_bytes() == b"unchanged"


def test_extractor_byte_loader_executes_admitted_payload_while_weight_path_is_foreign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from safetensors import torch as safetensors_torch

    weight_path = tmp_path / "segnet.safetensors"
    admitted_model, admitted_payload = _tiny_segnet_payload(weight_path)
    admitted_snapshot = cache_module.read_bound_file(weight_path, role="test admitted SegNet weights")
    admitted_row = {
        "path": str(weight_path),
        "bytes": len(admitted_snapshot.payload),
        "sha256": hashlib.sha256(admitted_snapshot.payload).hexdigest(),
    }
    canonical = _TinySegNet()
    canonical.load_state_dict(safetensors_torch.load_file(weight_path), strict=True)
    canonical.eval()

    displaced = tmp_path / "admitted-weights-displaced.safetensors"
    foreign = tmp_path / "foreign.safetensors"
    foreign_custody = tmp_path / "foreign-weight-custody.safetensors"
    _foreign_model, foreign_payload = _tiny_segnet_payload(foreign, offset=7.0)
    os.replace(weight_path, displaced)
    os.replace(foreign, weight_path)
    monkeypatch.setattr(
        safetensors_torch,
        "load_file",
        lambda *_args, **_kwargs: pytest.fail("pathname-based weight loading is forbidden"),
    )

    loaded = extractor_module._load_segnet_from_admitted_payload(
        SimpleNamespace(SegNet=_TinySegNet),
        admitted_snapshot.payload,
    )

    assert weight_path.read_bytes() == foreign_payload
    assert admitted_snapshot.payload == admitted_payload
    assert admitted_row["bytes"] == len(admitted_snapshot.payload)
    assert admitted_row["sha256"] == hashlib.sha256(admitted_snapshot.payload).hexdigest()
    assert not loaded.training
    assert all(parameter.device.type == "cpu" for parameter in loaded.parameters())
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    fixed = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    with torch.inference_mode():
        assert torch.equal(loaded(fixed), canonical(fixed))
        assert torch.equal(loaded(fixed), admitted_model(fixed))
    assert safetensors_torch.save(loaded.state_dict()) == admitted_payload

    os.replace(weight_path, foreign_custody)
    os.replace(displaced, weight_path)
    assert weight_path.read_bytes() == admitted_payload
    assert foreign_custody.read_bytes() == foreign_payload


def test_extractor_source_snapshot_binds_weight_row_to_exact_payload(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    (upstream / "models").mkdir(parents=True)
    (upstream / "modules.py").write_bytes(b"modules source\n")
    (upstream / "frame_utils.py").write_bytes(b"frame utils source\n")
    weights = upstream / "models" / "segnet.safetensors"
    _model, payload = _tiny_segnet_payload(weights)
    gt_cache = tmp_path / "gt.npz"
    gt_cache.write_bytes(b"gt cache bytes")

    snapshot = extractor_module._source_bindings(upstream, gt_cache)

    assert snapshot.segnet_payload == payload
    assert snapshot.rows["segnet_weights"] == {
        "path": str(weights),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    _foreign, foreign_payload = _tiny_segnet_payload(weights, offset=11.0)
    assert weights.read_bytes() == foreign_payload
    assert snapshot.segnet_payload == payload


def test_factorization_lazy_source_loader_executes_admitted_bytes_not_preimported_alias(tmp_path: Path) -> None:
    module_name = "tac.boundary_math.power_diagram_witness"
    source = tmp_path / "factorization.py"
    source.write_text("EXECUTED_MARKER = 'admitted-new-bytes'\n", encoding="utf-8")
    row = cache_module.source_file_row(source)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = SimpleNamespace(EXECUTED_MARKER="stale-preimported-bytes")  # type: ignore[assignment]
    try:
        loaded = extractor_module._load_source_module_from_snapshot(module_name, row, role="factorization")
        assert loaded.EXECUTED_MARKER == "admitted-new-bytes"  # type: ignore[attr-defined]
        assert cache_module.source_file_row(source) == row
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


def test_extractor_import_path_does_not_eagerly_execute_factorization_module() -> None:
    module_name = "tac.boundary_math.power_diagram_witness"
    script = (
        "import sys; import tools.extract_segnet_head_features_n600; "
        f"raise SystemExit(1 if {module_name!r} in sys.modules else 0)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=extractor_module.REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
