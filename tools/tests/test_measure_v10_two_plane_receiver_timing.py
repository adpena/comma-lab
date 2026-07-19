# SPDX-License-Identifier: MIT
"""Behavioral tests for the C1 preparation/timing/composition tool."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import tools.measure_v10_two_plane_receiver_timing as measure
from tac.witness_dsl.v10_production_receiver import parse_packet


class _ImmediateFuture:
    def __init__(self, value: Any) -> None:
        self._value = value

    def result(self) -> Any:
        return self._value


class _SandboxInlineProcessPool:
    """Exercise the process-pool branch where named semaphores are sandboxed."""

    def __init__(
        self,
        *,
        max_workers: int,
        initializer: Any,
        initargs: tuple[Any, ...],
    ) -> None:
        self.max_workers = max_workers
        initializer(*initargs)

    def submit(self, function: Any, *args: Any) -> _ImmediateFuture:
        return _ImmediateFuture(function(*args))

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        assert wait is True
        assert cancel_futures is True


@pytest.fixture(autouse=True)
def _sandbox_process_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    import tac.witness_dsl.v10_two_plane_timing_receiver as receiver

    monkeypatch.setattr(receiver, "ProcessPoolExecutor", _SandboxInlineProcessPool)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))


def _cache(path: Path, pair_count: int = 6) -> Path:
    coordinates = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    frame0 = np.stack([coordinates + pair_id for pair_id in range(pair_count)])
    frame1 = np.stack([coordinates + pair_id + 11 for pair_id in range(pair_count)])
    labels = np.zeros((pair_count, 2, 2), dtype=np.uint8)
    poses = np.zeros((pair_count, 6), dtype=np.float32)
    np.savez(
        path,
        n_pairs=np.asarray(pair_count),
        gt_f0=frame0,
        gt_f1=frame1,
        lstars=labels,
        gt_poses=poses,
    )
    return path


def _prepare(tmp_path: Path, *, pair_count: int = 6) -> tuple[Path, dict[str, Any]]:
    cache = _cache(tmp_path / "fixture.npz", pair_count)
    root = tmp_path / "work"
    receipt = root / "prepare.json"
    row = measure.prepare_two_plane_archive(
        cache_path=cache,
        work_root=root,
        receipt_path=receipt,
        pair_count=pair_count,
        camera_hw=(4, 4),
        scorer_hw=(2, 2),
        expected_cache_sha256=None,
        expected_y0_sha256=None,
        expected_y1_sha256=None,
        requested_storage_bytes=1,
        test_only_small_fixture=True,
    )
    return receipt, dict(row)


def _oracle(_raw: Path, pair_ids: tuple[int, ...], **_kwargs: Any) -> dict[str, Any]:
    return {
        "receiver_arithmetic": "native_float32_cpu_torch",
        "law_id": measure.F32_LAW_ID,
        "input_contract": measure.HARD_ORACLE_INPUT_CONTRACT,
        "pairs": [
            {
                "pair_id": pair_id,
                "d_seg": 0.0,
                "d_pose": 0.0,
                "seg_mismatched_pixels": 0,
                "pose6": [0.0] * 6,
            }
            for pair_id in pair_ids
        ],
    }


def _timed_fixture(tmp_path: Path) -> dict[str, Any]:
    prepare_path, prepared = _prepare(tmp_path)
    receipt_paths: list[Path] = []
    rows: list[dict[str, Any]] = []
    for label, workers in (("serial", 1), ("parallel-a", 4), ("parallel-b", 4)):
        receipt = tmp_path / f"{label}.json"
        row = measure.run_one_inflate(
            prepare_receipt_path=prepare_path,
            output_dir=tmp_path / f"out-{label}",
            timing_receipt_path=receipt,
            workers=workers,
            test_only_small_fixture=True,
        )
        receipt_paths.append(receipt)
        rows.append(dict(row))
    return {
        "prepare_path": prepare_path,
        "prepared": prepared,
        "receipt_paths": receipt_paths,
        "rows": rows,
        "raw_bytes": 6 * 2 * 4 * 4 * 3,
        "numerator_values": 6 * 2 * 2 * 2 * 3,
    }


def test_exact_integer_rounding_matches_half_up() -> None:
    from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator

    operator = DisjointResizeOperator.build(camera_h=4, camera_w=4, scorer_h=2, scorer_w=2)
    frame = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    numerator, denominator = operator.apply_numerators(frame)
    expected = ((numerator + denominator // 2) // denominator).astype(np.uint8)
    assert np.array_equal(measure.exact_operator_round_u8(operator, frame), expected)


def test_prepare_chunks_are_write_once_and_resume_does_not_recompress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    original = measure.encode_predictor_residual

    def wrapped(*args: Any, **kwargs: Any) -> bytes:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(measure, "encode_predictor_residual", wrapped)
    prepare_path, first = _prepare(tmp_path)
    assert calls == 1
    second = measure.prepare_two_plane_archive(
        cache_path=tmp_path / "fixture.npz",
        work_root=tmp_path / "work",
        receipt_path=prepare_path,
        pair_count=6,
        camera_hw=(4, 4),
        scorer_hw=(2, 2),
        expected_cache_sha256=None,
        expected_y0_sha256=None,
        expected_y1_sha256=None,
        resume=True,
        requested_storage_bytes=1,
        test_only_small_fixture=True,
    )
    assert calls == 1
    assert second == first
    assert first["combined_without_recompression"] is True
    with __import__("zipfile").ZipFile(first["archive_path"]) as archive:
        parsed = parse_packet(archive.read("0.bin"))
    assert Path(first["contest_archive_dir"], "0.bin").read_bytes() == parsed.packet_bytes
    assert first["contest_adapter_bound"] is True
    adapter = Path(first["contest_adapter_path"]).read_text(encoding="utf-8")
    assert 'contest-inflate "$1" "$2" "$3"' in adapter
    assert first["archive_sha256"] in adapter
    assert parsed.header["pair_count"] == 6
    assert parsed.header["frame0_policy_id"] == "description-frame0.v1"
    assert first["y0_sha256"] != first["y1_sha256"]


def test_prepare_stop_resume_has_no_fake_final_receipt(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "fixture.npz", 13)
    root = tmp_path / "work"
    receipt = root / "prepare.json"
    partial = measure.prepare_two_plane_archive(
        cache_path=cache,
        work_root=root,
        receipt_path=receipt,
        pair_count=13,
        camera_hw=(4, 4),
        scorer_hw=(2, 2),
        expected_cache_sha256=None,
        expected_y0_sha256=None,
        expected_y1_sha256=None,
        stop_after_chunks=1,
        requested_storage_bytes=1,
        test_only_small_fixture=True,
    )
    assert partial["completed"] is False
    assert not receipt.exists()
    assert not (root / "archive.zip").exists()
    completed = measure.prepare_two_plane_archive(
        cache_path=cache,
        work_root=root,
        receipt_path=receipt,
        pair_count=13,
        camera_hw=(4, 4),
        scorer_hw=(2, 2),
        expected_cache_sha256=None,
        expected_y0_sha256=None,
        expected_y1_sha256=None,
        resume=True,
        requested_storage_bytes=1,
        test_only_small_fixture=True,
    )
    assert completed["completed"] is True
    assert receipt.is_file()


def test_full_prepare_refuses_noncanonical_storage_before_writes(tmp_path: Path) -> None:
    root = tmp_path / "not-the-canonical-ssd"
    with pytest.raises(measure.C1MeasurementError, match="exact canonical SSD"):
        measure.prepare_two_plane_archive(
            cache_path=tmp_path / "does-not-matter.npz",
            work_root=root,
        )
    assert not root.exists()


def test_prepare_resume_refuses_edited_completed_chunk(tmp_path: Path) -> None:
    prepare_path, _prepared = _prepare(tmp_path)
    y0_path = measure._chunk_paths(tmp_path / "work", 0)[0]
    payload = bytearray(y0_path.read_bytes())
    payload[0] ^= 1
    y0_path.write_bytes(payload)
    with pytest.raises(measure.C1MeasurementError, match="custody drifted"):
        measure.prepare_two_plane_archive(
            cache_path=tmp_path / "fixture.npz",
            work_root=tmp_path / "work",
            receipt_path=prepare_path,
            pair_count=6,
            camera_hw=(4, 4),
            scorer_hw=(2, 2),
            expected_cache_sha256=None,
            expected_y0_sha256=None,
            expected_y1_sha256=None,
            resume=True,
            requested_storage_bytes=1,
            test_only_small_fixture=True,
        )


def test_actual_serial_and_parallel_cli_seam_is_byte_identical(tmp_path: Path) -> None:
    fixture = _timed_fixture(tmp_path)
    serial, first, second = fixture["rows"]
    assert serial["execution"]["workers"] == 1
    assert first["execution"]["workers"] == second["execution"]["workers"] == 4
    for key in (
        "raw_sha256",
        "stage_tree_sha256",
        "plane0_tree_sha256",
        "chunk_tree_sha256",
        "output_tree_sha256",
    ):
        assert serial[key] == first[key] == second[key]
    for row in (serial, first, second):
        assert row["schema"] == measure.TOOL_TIMING_SCHEMA
        assert row["resumed_pairs"] == 0
        assert row["raw_relative_path"] == "0.raw"
        assert row["numerator_values_verified"] == fixture["numerator_values"]
        assert all(row["timing"][name] > 0 for name in measure.TIMING_COMPONENTS)
        assert row[measure.CALLER_WALL_FIELD] >= row["timing"]["total_seconds"]
        nested = row["receiver_receipt"]
        assert Path(nested["path"]).is_file()
        assert nested["sha256"] == measure._sha256_file(Path(nested["path"]))


def test_compose_requires_three_fresh_rows_and_emits_close_ticket(tmp_path: Path) -> None:
    fixture = _timed_fixture(tmp_path)
    output = tmp_path / "composed.json"
    receipt = measure.compose_timing_receipt(
        prepare_receipt_path=fixture["prepare_path"],
        serial_receipt_path=fixture["receipt_paths"][0],
        parallel_receipt_paths=fixture["receipt_paths"][1:],
        output_path=output,
        hard_oracle_pair_ids=tuple(range(6)),
        hard_oracle_runner=_oracle,
        expected_pair_count=6,
        expected_raw_bytes=fixture["raw_bytes"],
        expected_numerator_values=fixture["numerator_values"],
        test_only_small_fixture=True,
    )
    assert receipt["timing_verdict"]["verdict"] == "CLOSE -> MODAL_MEASUREMENT_OWED"
    assert receipt["calibration_authority_status"] == "BLOCKED_CANONICAL_FULL_EVALUATE_RECEIPT_VALIDATOR_OWED"
    ticket = receipt["modal_measurement_ticket"]
    assert ticket["status"] == "TEST_ONLY_UNFIRED"
    assert ticket["evaluation_scope"].startswith("full evaluate.sh")
    assert ticket["archive"]["sha256"] == fixture["prepared"]["archive_sha256"]
    assert ticket["dispatch_claim_ledger"] == ".omx/state/active_lane_dispatch_claims.md"
    assert ticket["call_id_ledger"] == ".omx/state/modal_call_id_ledger.jsonl"
    assert ticket["ready_for_operator_authorized_dispatch"] is False
    assert ticket["max_cost_usd"] == 20
    assert ticket["full_evaluate_command"] == [
        "bash",
        "upstream/evaluate.sh",
        "--submission-dir",
        "submissions/c1_two_plane_receiver_timing_20260719",
        "--video-names-file",
        "upstream/public_test_video_names.txt",
        "--device",
        "cpu",
    ]
    assert ticket["instance_classes"][0]["dispatch_ready"] is False
    assert ticket["instance_classes"][1]["dispatch_ready"] is False
    assert ticket["instance_classes"][1]["seconds"] is None
    assert ticket["inflate_entrypoint"]["subcommand"] == "contest-inflate"
    assert receipt["source_receipts"][0]["role"] == "prepare"
    assert receipt["source_receipts"][0]["sha256"] == measure._sha256_file(fixture["prepare_path"])
    assert receipt["hard_oracle"]["input_contract"] == measure.HARD_ORACLE_INPUT_CONTRACT
    assert receipt["sacred_donor_snapshot_after_measurement"] == {"test_only_not_consulted": True}
    assert receipt["full_official_evaluation_measured"] is False
    assert receipt["score_claim"] is False
    assert output.read_bytes() == _canonical(receipt)


def test_production_modal_ticket_binds_runtime_sources_and_exact_cpu_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive.zip"
    adapter = tmp_path / "inflate.sh"
    packet = b"packet"
    archive.write_bytes(measure._canonical_archive_bytes(packet))
    adapter.write_bytes(b"adapter")
    adapter.chmod(0o755)
    runtime = json.loads(json.dumps(measure._runtime_source_custody()))
    runtime["remote_checkout_reproduces_sources"] = True
    for row in runtime["sources"].values():
        row["head_blob_sha256"] = row["sha256"]
        row["head_blob_matches_worktree"] = True
    monkeypatch.setattr(measure, "_runtime_source_custody", lambda: runtime)
    ticket = measure._modal_ticket(
        archive_path=archive,
        archive_sha256=measure._sha256_file(archive),
        archive_bytes=archive.stat().st_size,
        packet_sha256=measure._sha256_bytes(packet),
        contest_adapter_path=adapter,
        contest_adapter_sha256=measure._sha256_file(adapter),
        contest_adapter_bytes=adapter.stat().st_size,
        contest_adapter_mode="0755",
        contest_adapter_workers=4,
        test_only_small_fixture=False,
    )
    assert ticket["status"] == "UNFIRED"
    assert ticket["ready_for_operator_authorized_dispatch"] is True
    assert ticket["pre_dispatch_claim_command"][1:3] == ["tools/claim_lane_dispatch.py", "claim"]
    runtime = ticket["runtime_custody"]
    assert ticket["required_remote_checkout_git_sha"] == runtime["git_sha"]
    assert set(runtime["sources"]) == {
        "measurement_tool",
        "timed_receiver",
        "production_receiver",
        "integer_solver",
    }
    for row in runtime["sources"].values():
        assert len(row["sha256"]) == 64
        assert (measure.REPO_ROOT / row["path"]).is_file()
    adapter_materialization = ticket["remote_submission_materialization"][1]
    assert adapter_materialization["mode"] == "0755"
    assert adapter_materialization["post_copy_chmod_argv"][0:2] == ["chmod", "0755"]
    with pytest.raises(measure.C1MeasurementError, match="archive custody drifted"):
        measure._modal_ticket(
            archive_path=archive,
            archive_sha256="0" * 64,
            archive_bytes=archive.stat().st_size,
            packet_sha256=measure._sha256_bytes(packet),
            contest_adapter_path=adapter,
            contest_adapter_sha256=measure._sha256_file(adapter),
            contest_adapter_bytes=adapter.stat().st_size,
            contest_adapter_mode="0755",
            contest_adapter_workers=4,
            test_only_small_fixture=False,
        )


def test_modal_ticket_refuses_unreproducible_remote_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive.zip"
    adapter = tmp_path / "inflate.sh"
    packet = b"packet"
    archive.write_bytes(measure._canonical_archive_bytes(packet))
    adapter.write_bytes(b"adapter")
    adapter.chmod(0o755)
    runtime = json.loads(json.dumps(measure._runtime_source_custody()))
    runtime["remote_checkout_reproduces_sources"] = False
    runtime["sources"]["measurement_tool"]["head_blob_sha256"] = None
    runtime["sources"]["measurement_tool"]["head_blob_matches_worktree"] = False
    monkeypatch.setattr(measure, "_runtime_source_custody", lambda: runtime)
    ticket = measure._modal_ticket(
        archive_path=archive,
        archive_sha256=measure._sha256_file(archive),
        archive_bytes=archive.stat().st_size,
        packet_sha256=measure._sha256_bytes(packet),
        contest_adapter_path=adapter,
        contest_adapter_sha256=measure._sha256_file(adapter),
        contest_adapter_bytes=adapter.stat().st_size,
        contest_adapter_mode="0755",
        contest_adapter_workers=4,
        test_only_small_fixture=False,
    )
    assert ticket["ready_for_operator_authorized_dispatch"] is False
    assert ticket["pre_dispatch_claim_command"] is None
    assert any("MAIN landing owed" in blocker for blocker in ticket["dispatch_blockers"])


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda row: row.update({"resumed_pairs": 1}), "fresh non-resumed"),
        (lambda row: row["timing"].pop("solve1_seconds"), "solve1_seconds"),
        (lambda row: row.update({"archive_sha256": "f" * 64}), "differs"),
        (lambda row: row.update({"raw_sha256": "e" * 64}), "raw output custody drifted"),
        (
            lambda row: row.update({"contest_budget_verdict": "PASS because local<1800"}),
            "claimed contest-budget authority",
        ),
        (
            lambda row: row.update({"timing_verdict": "CLEARLY_UNDER", "local_lt_1800": True}),
            "claimed contest-budget authority",
        ),
        (
            lambda row: row["source_hashes"].update({"timed_receiver_sha256": "f" * 64}),
            "source hash custody drifted",
        ),
        (lambda row: row["host"].pop("platform"), "host custody is malformed"),
        (lambda row: row["thread_environment"].update({"UNBOUND_THREAD_ENV": "1"}), "thread-environment"),
    ],
)
def test_compose_refuses_invalid_timing_rows(
    tmp_path: Path,
    mutation: Any,
    match: str,
) -> None:
    fixture = _timed_fixture(tmp_path)
    bad = json.loads(json.dumps(fixture["rows"][1]))
    nested_path = Path(bad["receiver_receipt"]["path"])
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    mutation(bad)
    mutation(nested)
    _write_json(nested_path, nested)
    bad["receiver_receipt"]["bytes"] = nested_path.stat().st_size
    bad["receiver_receipt"]["sha256"] = measure._sha256_file(nested_path)
    bad_path = tmp_path / "bad.json"
    _write_json(bad_path, bad)
    with pytest.raises(measure.C1MeasurementError, match=match):
        measure.compose_timing_receipt(
            prepare_receipt_path=fixture["prepare_path"],
            serial_receipt_path=fixture["receipt_paths"][0],
            parallel_receipt_paths=(bad_path, fixture["receipt_paths"][2]),
            output_path=tmp_path / "composed.json",
            hard_oracle_pair_ids=tuple(range(6)),
            hard_oracle_runner=_oracle,
            expected_pair_count=6,
            expected_raw_bytes=fixture["raw_bytes"],
            expected_numerator_values=fixture["numerator_values"],
            test_only_small_fixture=True,
        )


def test_compose_refuses_fewer_than_six_hard_oracle_pairs(tmp_path: Path) -> None:
    fixture = _timed_fixture(tmp_path)
    with pytest.raises(measure.C1MeasurementError, match="at least six"):
        measure.compose_timing_receipt(
            prepare_receipt_path=fixture["prepare_path"],
            serial_receipt_path=fixture["receipt_paths"][0],
            parallel_receipt_paths=fixture["receipt_paths"][1:],
            output_path=tmp_path / "composed.json",
            hard_oracle_pair_ids=tuple(range(5)),
            hard_oracle_runner=_oracle,
            expected_pair_count=6,
            expected_raw_bytes=fixture["raw_bytes"],
            expected_numerator_values=fixture["numerator_values"],
            test_only_small_fixture=True,
        )


def test_production_compose_forbids_injected_hard_oracle(tmp_path: Path) -> None:
    with pytest.raises(measure.C1MeasurementError, match="forbids injected"):
        measure.compose_timing_receipt(
            prepare_receipt_path=tmp_path / "prepare.json",
            serial_receipt_path=tmp_path / "serial.json",
            parallel_receipt_paths=(tmp_path / "parallel-1.json", tmp_path / "parallel-2.json"),
            output_path=tmp_path / "composed.json",
            hard_oracle_runner=_oracle,
        )


def test_production_compose_rejects_all_calibration_paths_until_inner_receipt_validator(
    tmp_path: Path,
) -> None:
    with pytest.raises(measure.C1MeasurementError, match="production calibration anchors are blocked"):
        measure.compose_timing_receipt(
            prepare_receipt_path=tmp_path / "prepare.json",
            serial_receipt_path=tmp_path / "serial.json",
            parallel_receipt_paths=(tmp_path / "parallel-1.json", tmp_path / "parallel-2.json"),
            output_path=tmp_path / "composed.json",
            calibration_anchor_paths=(tmp_path / "self-attested.json",),
        )


def test_compose_reopens_prepared_archive_before_adjudication(tmp_path: Path) -> None:
    fixture = _timed_fixture(tmp_path)
    archive_path = Path(fixture["prepared"]["archive_path"])
    archive_path.write_bytes(archive_path.read_bytes() + b"drift")
    with pytest.raises(measure.C1MeasurementError, match="archive bytes drifted"):
        measure.compose_timing_receipt(
            prepare_receipt_path=fixture["prepare_path"],
            serial_receipt_path=fixture["receipt_paths"][0],
            parallel_receipt_paths=fixture["receipt_paths"][1:],
            output_path=tmp_path / "composed.json",
            hard_oracle_pair_ids=tuple(range(6)),
            hard_oracle_runner=_oracle,
            expected_pair_count=6,
            expected_raw_bytes=fixture["raw_bytes"],
            expected_numerator_values=fixture["numerator_values"],
            test_only_small_fixture=True,
        )


def test_compose_reopens_bound_contest_adapter_before_ticket_readiness(tmp_path: Path) -> None:
    fixture = _timed_fixture(tmp_path)
    adapter_path = Path(fixture["prepared"]["contest_adapter_path"])
    adapter_path.write_bytes(adapter_path.read_bytes() + b"# drift\n")
    with pytest.raises(measure.C1MeasurementError, match="adapter custody drifted"):
        measure.compose_timing_receipt(
            prepare_receipt_path=fixture["prepare_path"],
            serial_receipt_path=fixture["receipt_paths"][0],
            parallel_receipt_paths=fixture["receipt_paths"][1:],
            output_path=tmp_path / "composed.json",
            hard_oracle_pair_ids=tuple(range(6)),
            hard_oracle_runner=_oracle,
            expected_pair_count=6,
            expected_raw_bytes=fixture["raw_bytes"],
            expected_numerator_values=fixture["numerator_values"],
            test_only_small_fixture=True,
        )


@pytest.mark.parametrize("bad_id", [0.0, "0", False])
def test_hard_oracle_pair_ids_are_not_coerced(tmp_path: Path, bad_id: Any) -> None:
    fixture = _timed_fixture(tmp_path)
    with pytest.raises(measure.C1MeasurementError, match="exact integer"):
        measure.compose_timing_receipt(
            prepare_receipt_path=fixture["prepare_path"],
            serial_receipt_path=fixture["receipt_paths"][0],
            parallel_receipt_paths=fixture["receipt_paths"][1:],
            output_path=tmp_path / "composed.json",
            hard_oracle_pair_ids=(bad_id, 1, 2, 3, 4, 5),
            hard_oracle_runner=_oracle,
            expected_pair_count=6,
            expected_raw_bytes=fixture["raw_bytes"],
            expected_numerator_values=fixture["numerator_values"],
            test_only_small_fixture=True,
        )


def test_hard_oracle_bulk_root_is_inside_prepared_root(tmp_path: Path) -> None:
    fixture = _timed_fixture(tmp_path)
    observed: dict[str, Path] = {}

    def oracle(raw_path: Path, pair_ids: tuple[int, ...], **kwargs: Any) -> dict[str, Any]:
        observed["raw_path"] = raw_path
        observed["output_root"] = kwargs["output_root"]
        return _oracle(raw_path, pair_ids, **kwargs)

    measure.compose_timing_receipt(
        prepare_receipt_path=fixture["prepare_path"],
        serial_receipt_path=fixture["receipt_paths"][0],
        parallel_receipt_paths=fixture["receipt_paths"][1:],
        output_path=tmp_path / "composed.json",
        hard_oracle_pair_ids=tuple(range(6)),
        hard_oracle_runner=oracle,
        expected_pair_count=6,
        expected_raw_bytes=fixture["raw_bytes"],
        expected_numerator_values=fixture["numerator_values"],
        test_only_small_fixture=True,
    )
    assert observed["output_root"] == Path(fixture["prepared"]["archive_dir"]) / "hard_oracle"


def test_prepare_mode_cannot_be_laundered_at_inflate(tmp_path: Path) -> None:
    prepare_path, _prepared = _prepare(tmp_path)
    with pytest.raises(measure.C1MeasurementError, match="test-only mode differs"):
        measure.run_one_inflate(
            prepare_receipt_path=prepare_path,
            output_dir=tmp_path / "out",
            timing_receipt_path=tmp_path / "timing.json",
            workers=1,
            test_only_small_fixture=False,
        )


def test_full_output_path_must_be_strictly_under_prepared_root(tmp_path: Path) -> None:
    with pytest.raises(measure.C1MeasurementError, match="must resolve under"):
        measure._strict_descendant(tmp_path / "outside", tmp_path / "prepared", "full receiver output_dir")
    with pytest.raises(measure.C1MeasurementError, match="strict descendant"):
        measure._strict_descendant(tmp_path / "prepared", tmp_path / "prepared", "full receiver output_dir")


def test_sacred_donor_is_resnapshotted_after_measurement(tmp_path: Path) -> None:
    donor = tmp_path / "donor"
    donor.mkdir()
    evidence = donor / "evidence.bin"
    evidence.write_bytes(b"settled")
    snapshot = measure._tree_snapshot(donor)
    prepared = {
        "sacred_donor_root": str(donor),
        "sacred_donor_snapshot_before": snapshot,
        "sacred_donor_snapshot_after": snapshot,
    }
    assert measure._revalidate_sacred_donor(prepared, test_only_small_fixture=False) == snapshot
    evidence.write_bytes(b"changed")
    with pytest.raises(measure.C1MeasurementError, match="changed after prepare"):
        measure._revalidate_sacred_donor(prepared, test_only_small_fixture=False)


def test_executable_contest_adapter_consumes_non_sibling_official_surface(tmp_path: Path) -> None:
    _prepare_path, prepared = _prepare(tmp_path)
    adapter = Path(prepared["contest_adapter_path"])
    assert adapter.stat().st_mode & 0o777 == 0o755
    output_root = tmp_path / "non-sibling-output-parent" / "decoded"
    environment = dict(os.environ)
    environment["PACT_REPO_ROOT"] = str(measure.REPO_ROOT)
    environment["PYTHON"] = str(measure.REPO_ROOT / ".venv/bin/python")
    environment["PATH"] = f"{measure.REPO_ROOT / '.venv/bin'}{os.pathsep}{environment.get('PATH', '')}"
    environment["PYTHONPATH"] = f"{measure.REPO_ROOT / 'src'}{os.pathsep}{measure.REPO_ROOT}"
    run = subprocess.run(
        [
            str(adapter),
            prepared["contest_archive_dir"],
            str(output_root),
            prepared["video_names_file"],
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result["completed"] is True
    assert result["workers"] == 1
    assert result["test_only_small_fixture"] is True
    assert result["archive_input_kind"] == "extracted_0_bin"
    assert (output_root / "0.raw").is_file()
    assert Path(result["timing_wrapper_path"]).parent == output_root.parent
    archive_path = Path(prepared["archive_path"])
    archive_path.write_bytes(archive_path.read_bytes() + b"noncanonical-trailer")
    refused = subprocess.run(
        [
            str(adapter),
            prepared["contest_archive_dir"],
            str(tmp_path / "second-non-sibling-parent" / "decoded"),
            prepared["video_names_file"],
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "archive/packet bytes differ" in refused.stderr


def test_calibration_is_measured_paired_only_and_three_way() -> None:
    assert (
        measure.derive_timing_verdict(
            100.0, [{"classification": "full_official_evaluation", "ratio_contest_to_local": 2.0}]
        )["verdict"]
        == "CLEARLY_UNDER"
    )
    assert (
        measure.derive_timing_verdict(
            1000.0,
            [
                {"classification": "full_official_evaluation", "ratio_contest_to_local": 1.0},
                {"classification": "full_official_evaluation", "ratio_contest_to_local": 2.0},
            ],
        )["verdict"]
        == "CLOSE -> MODAL_MEASUREMENT_OWED"
    )
    assert (
        measure.derive_timing_verdict(
            1000.0, [{"classification": "full_official_evaluation", "ratio_contest_to_local": 2.0}]
        )["verdict"]
        == "CLEARLY_OVER"
    )
    assert (
        measure.derive_timing_verdict(1.0, [{"classification": "inflate_only", "ratio_contest_to_local": 0.01}])[
            "verdict"
        ]
        == "CLOSE -> MODAL_MEASUREMENT_OWED"
    )


def test_calibration_anchor_refuses_invented_margin(tmp_path: Path) -> None:
    path = tmp_path / "anchor.json"
    _write_json(
        path,
        {
            "schema": measure.CALIBRATION_SCHEMA,
            "measured": True,
            "paired": True,
            "classification": "full_official_evaluation",
            "local_inflate_seconds": 10.0,
            "contest_seconds": 20.0,
            "local_host": "measured-host",
            "contest_instance_class": "T4",
            "archive_sha256": "a" * 64,
            "novel_empirical_margin_percent": 0.5,
        },
    )
    with pytest.raises(measure.C1MeasurementError, match="invented"):
        measure._load_calibration_anchors((path,))


def test_calibration_anchor_cannot_self_attest_without_paired_evidence(tmp_path: Path) -> None:
    path = tmp_path / "anchor.json"
    row = {
        "schema": measure.CALIBRATION_SCHEMA,
        "measured": True,
        "paired": True,
        "classification": "full_official_evaluation",
        "local_inflate_seconds": 10.0,
        "contest_seconds": 20.0,
        "local_host": "measured-host",
        "contest_instance_class": "cpu-4c-16g",
        "archive_sha256": "a" * 64,
    }
    _write_json(path, row)
    with pytest.raises(measure.C1MeasurementError, match="reopenable paired evidence"):
        measure._load_calibration_anchors((path,))

    local = tmp_path / "local-receipt.json"
    contest = tmp_path / "contest-receipt.json"
    _write_json(local, {"kind": "local", "archive_sha256": "a" * 64})
    _write_json(contest, {"kind": "full-evaluate", "archive_sha256": "a" * 64})
    row["local_evidence"] = {
        "path": str(local),
        "bytes": local.stat().st_size,
        "sha256": measure._sha256_file(local),
        "archive_sha256": "a" * 64,
    }
    row["contest_evidence"] = {
        "path": str(contest),
        "bytes": contest.stat().st_size,
        "sha256": measure._sha256_file(contest),
        "archive_sha256": "a" * 64,
        "classification": "full_official_evaluation",
    }
    _write_json(path, row)
    assert measure._load_calibration_anchors((path,))[0]["ratio_contest_to_local"] == 2.0


def test_mlx_unavailable_is_host_custody_not_parity(tmp_path: Path) -> None:
    prepare_path, _prepared = _prepare(tmp_path)
    output = tmp_path / "mlx-unavailable.json"
    row = measure.run_mlx_parity(
        prepare_receipt_path=prepare_path,
        output_path=output,
        pair_ids=tuple(range(6)),
        runtime_status_fn=lambda **_kwargs: {
            "runtime_installed": True,
            "metal_usable": False,
            "host_custody_refusal": "installed without usable Metal",
        },
    )
    assert row["status"] == "HOST_CUSTODY_REFUSAL"
    assert row["parity_measured"] is False
    assert row["contest_verdict_input"] is False
    assert row["score_claim"] is False


@pytest.mark.parametrize("bad_id", [0.0, "0", False])
def test_mlx_pair_ids_are_not_coerced(tmp_path: Path, bad_id: Any) -> None:
    prepare_path, _prepared = _prepare(tmp_path)
    with pytest.raises(measure.C1MeasurementError, match="exact integer"):
        measure.run_mlx_parity(
            prepare_receipt_path=prepare_path,
            output_path=tmp_path / "mlx.json",
            pair_ids=(bad_id, 1, 2, 3, 4, 5),
            runtime_status_fn=lambda **_kwargs: {
                "runtime_installed": False,
                "metal_usable": False,
                "host_custody_refusal": "unavailable",
            },
        )


@pytest.mark.parametrize("bad_dimension", [4.0, "4", False])
def test_prepared_geometry_is_not_coerced(tmp_path: Path, bad_dimension: Any) -> None:
    prepare_path, prepared = _prepare(tmp_path)
    prepared = dict(prepared)
    prepared["camera_hw"] = [bad_dimension, 4]
    _write_json(prepare_path, prepared)
    with pytest.raises(measure.C1MeasurementError, match="exact integer"):
        measure.run_one_inflate(
            prepare_receipt_path=prepare_path,
            output_dir=tmp_path / "out",
            timing_receipt_path=tmp_path / "timing.json",
            workers=1,
            test_only_small_fixture=True,
        )


def test_mlx_parity_row_remains_false_authority_and_preserves_divergence(tmp_path: Path) -> None:
    prepare_path, _prepared = _prepare(tmp_path)

    def parity(_operator: Any, _y0: Any, _y1: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "parity_passed": False,
            "divergences": [{"pair_id": 2, "plane_index": 1, "first_mismatch_flat_index": 7}],
            "score_claim": False,
            "promotion_eligible": False,
            "contest_timing_verdict_eligible": False,
        }

    row = measure.run_mlx_parity(
        prepare_receipt_path=prepare_path,
        output_path=tmp_path / "mlx-divergence.json",
        pair_ids=tuple(range(6)),
        mlx_module=object(),
        runtime_status_fn=lambda **_kwargs: {
            "runtime_installed": True,
            "metal_usable": True,
            "host_custody_refusal": None,
        },
        parity_fn=parity,
    )
    assert row["status"] == "INTEGER_OP_DIVERGENCE"
    assert row["divergences"][0]["plane_index"] == 1
    assert row["contest_budget_authority"] is False
    assert row["score_claim"] is False


def test_cuda_envelope_has_only_derived_work_counts() -> None:
    row = measure.cuda_workload_envelope()
    assert row == {
        "status": "DERIVED_UNMEASURED_CUDA_WORKLOAD",
        "output_bytes": 3_662_409_600,
        "numerator_values": 707_788_800,
        "uniform_tap_products": 2_831_155_200,
        "seconds": None,
        "timing_verdict": None,
        "contest_budget_authority": False,
        "score_claim": False,
        "promotion_eligible": False,
    }


def test_inflate_cli_requires_explicit_worker_count() -> None:
    with pytest.raises(SystemExit):
        measure._build_parser().parse_args(
            [
                "inflate",
                "--prepare-receipt",
                "prepare.json",
                "--output-dir",
                "out",
                "--receipt",
                "timing.json",
            ]
        )
