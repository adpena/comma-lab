from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "replay_ep725_xcodec_n600_equality.py"
SPEC = importlib.util.spec_from_file_location("g22_replay", TOOL)
assert SPEC is not None and SPEC.loader is not None
g22 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g22)


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
    return buffer.getvalue()


def _pointer_observation(raw: bytes) -> dict[str, object]:
    target, provenance = g22._competitive_target_identity(raw)
    return {
        "artifact": {"sha256": hashlib.sha256(raw).hexdigest()},
        "competitive_target": target,
        "provenance": provenance,
    }


def test_frozen_contract_binds_exact_counted_selected_bytes() -> None:
    assert g22.FROZEN["g20_archive_bytes"] == 81027
    assert g22.FROZEN["g20_archive_sha256"] == (
        "8e9c7ba0fdd1fc0fdff696c639821d6e64a3110bb8744f47ae0ab3d287cd70d8"
    )


def test_canonical_json_is_sorted_and_newline_terminated() -> None:
    assert g22._canonical_json({"z": 1, "a": 2}) == b'{"a":2,"z":1}\n'


def test_stable_read_accepts_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "regular.bin"
    path.write_bytes(b"custody")
    assert g22._stable_read(path) == b"custody"


def test_stable_read_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"custody")
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    with pytest.raises(g22.ReplayError, match="regular file"):
        g22._stable_read(link)


def test_file_row_reopens_bytes_and_rejects_hash_drift(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"exact")
    with pytest.raises(g22.ReplayError, match="SHA-256 drift"):
        g22._file_row(path, expected_sha256="0" * 64)


def test_file_row_rejects_size_drift(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"exact")
    with pytest.raises(g22.ReplayError, match="byte-size drift"):
        g22._file_row(path, expected_bytes=4)


def test_validate_preserved_row_requires_absolute_reopened_bytes(tmp_path: Path) -> None:
    path = tmp_path / "preserved.json"
    path.write_bytes(b"preserved")
    row = g22._file_row(path)
    g22._validate_preserved_row(row, "fixture")
    path.write_bytes(b"tampered!")
    with pytest.raises(g22.ReplayError, match="SHA-256 drift"):
        g22._validate_preserved_row(row, "fixture")


def test_write_once_is_idempotent_but_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    g22._write_once(path, b"first\n")
    g22._write_once(path, b"first\n")
    with pytest.raises(g22.ReplayError, match="append-only artifact differs"):
        g22._write_once(path, b"second\n")
    assert path.read_bytes() == b"first\n"


def test_extract_single_member_hashes_reopened_member_bytes() -> None:
    member, row = g22._extract_single_member(_zip_bytes({"0.bin": b"payload"}), label="fixture")
    assert member == b"payload"
    assert row["bytes"] == 7
    assert row["sha256"] == hashlib.sha256(b"payload").hexdigest()


@pytest.mark.parametrize(
    "members",
    [
        {"wrong.bin": b"payload"},
        {"0.bin": b"payload", "extra.bin": b"extra"},
    ],
)
def test_extract_single_member_rejects_archive_shape_drift(members: dict[str, bytes]) -> None:
    with pytest.raises(g22.ReplayError, match=r"exactly one regular 0\.bin"):
        g22._extract_single_member(_zip_bytes(members), label="fixture")


def test_partition_preserves_order_and_exact_population() -> None:
    rows = g22._partition(list(range(10)), 3)
    assert rows == [list(range(4)), list(range(4, 7)), list(range(7, 10))]
    assert [value for row in rows for value in row] == list(range(10))


def test_chunk_rows_bind_pair_ids_and_exact_byte_ranges() -> None:
    rows = g22._chunk_rows(pair_count=5, chunk_pairs=2, frame_bytes=9)
    assert rows == [
        {"index": 0, "pair_ids": [0, 1], "byte_offset": 0, "byte_length": 36},
        {"index": 1, "pair_ids": [2, 3], "byte_offset": 36, "byte_length": 36},
        {"index": 2, "pair_ids": [4], "byte_offset": 72, "byte_length": 18},
    ]


def test_compare_range_checks_every_byte_and_returns_hashes(tmp_path: Path) -> None:
    left = tmp_path / "left.raw"
    right = tmp_path / "right.raw"
    raw = bytes(range(64))
    left.write_bytes(raw)
    right.write_bytes(raw)
    left_hash, right_hash = g22._compare_range(left, right, 7, 41)
    assert left_hash == right_hash == hashlib.sha256(raw[7:48]).hexdigest()


def test_compare_range_reports_first_exact_byte_mismatch(tmp_path: Path) -> None:
    left = tmp_path / "left.raw"
    right = tmp_path / "right.raw"
    left.write_bytes(b"0123456789")
    right.write_bytes(b"012X456789")
    with pytest.raises(g22.ReplayError, match="raw byte 3"):
        g22._compare_range(left, right, 0, 10)


def test_preallocate_materializes_exact_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "witness.raw"
    g22._preallocate(path, 16384)
    metadata = path.stat()
    assert metadata.st_size == 16384
    assert metadata.st_blocks * 512 >= 16384
    g22._preallocate(path, 16384)


def test_preallocate_rejects_wrong_sized_resume_file(tmp_path: Path) -> None:
    path = tmp_path / "witness.raw"
    path.write_bytes(b"short")
    with pytest.raises(g22.ReplayError, match="partial or wrong-sized"):
        g22._preallocate(path, 100)


def test_run_root_rejects_local_tier_without_test_override(tmp_path: Path) -> None:
    with pytest.raises(g22.ReplayError, match="SSD tier"):
        g22._validate_run_root(tmp_path / "run", allow_non_ssd=False)


def test_partial_file_blocks_resume(tmp_path: Path) -> None:
    (tmp_path / "chunk.partial.1").write_bytes(b"orphan")
    with pytest.raises(g22.ReplayError, match="partial files"):
        g22._assert_no_partial_files(tmp_path)


def _checkpoint_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object], str]:
    source = tmp_path / "source.raw"
    selected = tmp_path / "selected.raw"
    raw = b"abcdefghijkl"
    source.write_bytes(raw)
    selected.write_bytes(raw)
    row: dict[str, object] = {
        "index": 0,
        "pair_ids": [0, 1],
        "byte_offset": 2,
        "byte_length": 8,
    }
    manifest_sha = "a" * 64
    range_sha = hashlib.sha256(raw[2:10]).hexdigest()
    checkpoint = {
        "schema": g22.SCHEMA_CHUNK,
        "manifest_sha256": manifest_sha,
        "chunk": row,
        "uint8_exact_equal": True,
        "source_range_sha256": range_sha,
        "selected_range_sha256": range_sha,
        "pointer_observation_after_chunk": g22._preserve_pointer_observation(
            tmp_path / "run",
            g22.FRONTIER_POINTER.read_bytes(),
        ),
    }
    path = tmp_path / "checkpoint.json"
    g22._write_once_json(path, checkpoint)
    return path, source, selected, row, manifest_sha


def test_checkpoint_resume_rehashes_bound_byte_range(tmp_path: Path) -> None:
    path, source, selected, row, manifest_sha = _checkpoint_fixture(tmp_path)
    loaded = g22._validate_checkpoint(
        path,
        row=row,
        manifest_sha256=manifest_sha,
        source_raw=source,
        selected_raw=selected,
    )
    assert loaded["uint8_exact_equal"] is True


def test_checkpoint_resume_rejects_drifted_output_bytes(tmp_path: Path) -> None:
    path, source, selected, row, manifest_sha = _checkpoint_fixture(tmp_path)
    selected.write_bytes(b"abXdefghijkl")
    with pytest.raises(g22.ReplayError, match="uint8 mismatch"):
        g22._validate_checkpoint(
            path,
            row=row,
            manifest_sha256=manifest_sha,
            source_raw=source,
            selected_raw=selected,
        )


def test_checkpoint_prefix_rejects_gap(tmp_path: Path) -> None:
    root = tmp_path / "run"
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "chunk-0001.json").write_text("{}")
    rows = [
        {"index": 0},
        {"index": 1},
    ]
    with pytest.raises(g22.ReplayError, match="immutable prefix"):
        g22._validate_checkpoint_prefix(root, rows)


def test_execute_refuses_without_review_before_contract_access(tmp_path: Path) -> None:
    args = g22._parser().parse_args(
        ["--resume-from", str(tmp_path / "run"), "--receipt", str(tmp_path / "receipt.json")]
    )
    with pytest.raises(g22.ReplayError, match="--execute-reviewed"):
        g22._execute(args)


def test_execute_refuses_full_run_without_explicit_n600_confirmation(tmp_path: Path) -> None:
    args = g22._parser().parse_args(
        [
            "--resume-from",
            str(tmp_path / "run"),
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--execute-reviewed",
            "--allow-non-ssd",
        ]
    )
    with pytest.raises(g22.ReplayError, match="--confirm-full-n600"):
        g22._execute(args)


def test_execute_refuses_nonzero_bounded_pair_start(tmp_path: Path) -> None:
    args = g22._parser().parse_args(
        [
            "--resume-from",
            str(tmp_path / "run"),
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--pair-start",
            "1",
            "--pair-count",
            "1",
            "--execute-reviewed",
            "--allow-non-ssd",
        ]
    )
    with pytest.raises(g22.ReplayError, match="ordered prefix"):
        g22._execute(args)


def test_cleanup_certificate_precedes_and_survives_raw_deletion(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    source = run_root / "scratch/source.raw"
    selected = run_root / "scratch/selected.raw"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"same")
    selected.write_bytes(b"same")
    manifest = run_root / "run_manifest.json"
    checkpoint = run_root / "checkpoints/chunk-0000.json"
    g22._write_once_json(manifest, {"schema": "manifest"})
    g22._write_once_json(checkpoint, {"schema": "checkpoint"})
    precleanup = {
        "archive_artifact": {"selected": {"bytes": 1, "sha256": "a" * 64}},
        "generic_decoder_runtime": {"bytes": 1, "sha256": "b" * 64},
        "decode_receipt": {
            "run_manifest": g22._file_row(manifest),
            "chunk_checkpoints": [g22._file_row(checkpoint)],
        },
        "output_witness": {
            "scratch_files": [g22._file_row(source), g22._file_row(selected)],
        },
        "reproducibility": {"exact_resume_command": "python replay --resume-from exact"},
    }
    precleanup_path = run_root / "receipts/decode_receipt.pre_cleanup.json"
    g22._write_once_json(precleanup_path, precleanup)
    cleanup = g22._cleanup_certified_raws(
        run_root=run_root,
        source_raw=source,
        selected_raw=selected,
        precleanup_path=precleanup_path,
        precleanup=precleanup,
    )
    assert not source.exists() and not selected.exists()
    certificate = json.loads(Path(cleanup["certificate"]["path"]).read_text())
    assert certificate["delete_authorized"] is True
    assert certificate["targets"][0]["sha256"] == hashlib.sha256(b"same").hexdigest()
    resumed = g22._cleanup_certified_raws(
        run_root=run_root,
        source_raw=source,
        selected_raw=selected,
        precleanup_path=precleanup_path,
        precleanup=precleanup,
    )
    assert resumed == cleanup


def test_worker_environment_forces_portable_fp64_and_single_thread() -> None:
    environment = g22._worker_environment()
    assert environment["INFLATE_FP32"] == "0"
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["OMP_NUM_THREADS"] == "1"


def test_parser_has_no_score_or_pointer_mutation_options() -> None:
    option_names = {
        option
        for action in g22._parser()._actions
        for option in action.option_strings
    }
    assert not any("score" in option for option in option_names)
    assert not any("candidate" in option for option in option_names)
    assert not any("pointer" in option for option in option_names)
    assert "--confirm-full-n600" in option_names


def test_metadata_only_pointer_refresh_does_not_invalidate_decode_equality() -> None:
    original = json.loads(g22.FRONTIER_POINTER.read_text())
    refreshed = copy.deepcopy(original)
    refreshed["last_refreshed_utc"] = "2026-07-26T16:00:00+00:00"
    refreshed["upstream_leaderboard_snapshot_at_utc"] = "2026-07-26T16:00:00+00:00"
    before_raw = g22._canonical_json(original)
    after_raw = g22._canonical_json(refreshed)
    change = g22._pointer_change(_pointer_observation(before_raw), _pointer_observation(after_raw))
    assert change == {
        "pointer_artifact_changed": True,
        "competitive_target_changed": False,
        "rebase_required_before_admission": False,
        "decode_equality_invalidated": False,
    }


def test_semantic_pointer_change_sets_rebase_without_invalidating_equality() -> None:
    original = json.loads(g22.FRONTIER_POINTER.read_text())
    changed = copy.deepcopy(original)
    replacement = {
        "name": "new-target",
        "pr_number": 999,
        "pr_url": "https://example.invalid/999",
        "rank": 1,
        "score": 0.16,
    }
    changed["upstream_leaderboard_snapshot"]["best_entry"] = replacement
    changed["upstream_leaderboard_snapshot"]["entries"][0] = replacement
    before_raw = g22._canonical_json(original)
    after_raw = g22._canonical_json(changed)
    change = g22._pointer_change(_pointer_observation(before_raw), _pointer_observation(after_raw))
    assert change["pointer_artifact_changed"] is True
    assert change["competitive_target_changed"] is True
    assert change["rebase_required_before_admission"] is True
    assert change["decode_equality_invalidated"] is False


def test_intermediate_semantic_change_is_retained_even_if_end_target_returns() -> None:
    original = json.loads(g22.FRONTIER_POINTER.read_text())
    changed = copy.deepcopy(original)
    changed["upstream_leaderboard_snapshot"]["best_entry"]["score"] = 0.16
    changed["upstream_leaderboard_snapshot"]["entries"][0]["score"] = 0.16
    start = _pointer_observation(g22._canonical_json(original))
    middle = _pointer_observation(g22._canonical_json(changed))
    aggregate = g22._aggregate_pointer_change(
        start,
        start,
        [{"pointer_observation_after_chunk": middle}],
    )
    assert aggregate["competitive_target_changed"] is True
    assert aggregate["rebase_required_before_admission"] is True
    assert aggregate["decode_equality_invalidated"] is False


def test_completed_receipt_validation_does_not_reopen_later_live_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    manifest_path = run_root / "run_manifest.json"
    manifest = {
        "config": {"pair_start": 0, "pair_count": 1, "chunk_pairs": 1, "workers": 1},
    }
    manifest_path.write_text(json.dumps(manifest))
    observation = {
        "artifact": {"sha256": "a" * 64},
        "competitive_target": {"target_score": 0.172},
    }
    receipt = {
        "schema": g22.SCHEMA_FINAL,
        "run_root": str(run_root),
        "cleanup": {"completed": True, "certificate": {}, "completion": {}},
        "reproducibility": {"tool": {"path": str(TOOL)}},
        "authority_pointer_status": {
            "pointer_start": observation,
            "pointer_end": observation,
            "pointer_artifact_changed": False,
            "competitive_target_changed": False,
            "rebase_required_before_admission": False,
            "decode_equality_invalidated": False,
            "decode_equality_independent_of_pointer_change": True,
            "candidate_claim": False,
            "score_claim": False,
        },
        "decode_receipt": {
            "run_manifest": {"path": str(manifest_path)},
            "chunk_checkpoints": [],
        },
        "precleanup_receipt": {},
        "output_witness": {"scratch_files": []},
        "archive_artifact": {"selected": {"counted_rate_bytes": 81027}},
        "resource_custody": {"storage_preflight": {"reserve_bytes": 7}},
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(g22, "_validate_frozen_contract", lambda: None)
    monkeypatch.setattr(g22, "_validate_preserved_row", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(g22, "_validate_pointer_observation", lambda *_args, **_kwargs: None)
    original_stable_read = g22._stable_read

    def forbid_live_pointer(path: Path) -> bytes:
        if path == g22.FRONTIER_POINTER:
            raise AssertionError("completed receipt must not depend on later live pointer bytes")
        return original_stable_read(path)

    monkeypatch.setattr(g22, "_stable_read", forbid_live_pointer)
    args = g22._parser().parse_args(
        [
            "--resume-from",
            str(run_root),
            "--receipt",
            str(receipt_path),
            "--pair-count",
            "1",
            "--chunk-pairs",
            "1",
            "--workers",
            "1",
            "--reserve-bytes",
            "7",
            "--execute-reviewed",
            "--allow-non-ssd",
        ]
    )
    assert g22._finalize_existing(receipt_path, run_root, args) == receipt
