from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "fit_taskspace_r10_n600_maximum_inverse.py"
SPEC = importlib.util.spec_from_file_location("g32_r10_fitter", TOOL)
assert SPEC is not None and SPEC.loader is not None
g32 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g32)


def _args(**updates):
    values = {
        "resume_from": Path("/tmp/g32-test"),
        "video": Path("video.mkv"),
        "selected_archive": Path("selected.zip"),
        "runtime": Path("inflate.py"),
        "pair_count": 2,
        "chunk_pairs": 1,
        "seed": 0,
        "sample_stride": 8,
        "reserve_bytes": 0,
        "execute_reviewed": True,
        "governed_claim_job_id": None,
        "governed_claim_platform": None,
        "preserve_scratch": False,
        "allow_local_test_only": True,
        "internal_runtime_plan": None,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def test_frozen_contract_is_exact_g20_g22_g27_lineage() -> None:
    assert g32.FROZEN["source_video_sha256"] == ("2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9")
    assert g32.FROZEN["selected_archive_sha256"] == ("8e9c7ba0fdd1fc0fdff696c639821d6e64a3110bb8744f47ae0ab3d287cd70d8")
    assert g32.FROZEN["runtime_sha256"] == ("4b54d512565f7275c53f697a931dd087222a36a69495b6e536a6b65dede36224")
    assert g32.FROZEN["r10_module_sha256"] == ("13cd771d10c333a458c9977f8b21b916a4baf80b063bb4f849f001a6f660e11d")


def test_full_n600_requires_live_governed_claim_not_caller_booleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(g32.G32LaunchError, match="governed-claim"):
        g32.validate_execution_flags(_args(pair_count=600))
    governed_args = _args(pair_count=600, governed_claim_job_id="job-1", governed_claim_platform="local")
    g32.validate_execution_flags(governed_args)
    with pytest.raises(g32.G32LaunchError, match="TAC_GOVERNED_ADMISSION"):
        g32.require_governed_n600_execution(governed_args, environment={})
    ledger = tmp_path / "active_lane_dispatch_claims.md"
    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    ledger.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| {now} | codex:test | {g32.G32_LANE_ID} | local | job-1 | | active | bounded test |\n"
    )
    monkeypatch.setattr(g32, "CLAIM_LEDGER", ledger)
    context = g32.require_governed_n600_execution(
        governed_args,
        environment={g32.GOVERNED_MARKER_ENV: "1"},
    )
    assert context["verified"] is True
    assert context["claim_record_sha256"] == g32._sha256_bytes(g32._canonical_json(context["claim"]))
    with pytest.raises(g32.G32LaunchError, match="only valid"):
        g32.validate_execution_flags(_args(pair_count=2, governed_claim_job_id="job-1"))
    with pytest.raises(g32.G32LaunchError, match="execute-reviewed"):
        g32.validate_execution_flags(_args(execute_reviewed=False))


def test_internal_parser_does_not_require_launch_arguments(tmp_path: Path) -> None:
    args = g32.build_parser().parse_args(["--internal-runtime-plan", str(tmp_path / "plan.json")])
    assert args.internal_runtime_plan == tmp_path / "plan.json"
    with pytest.raises(g32.G32LaunchError, match="missing required"):
        g32.validate_execution_flags(args)


def test_run_root_requires_ssd_without_explicit_test_escape(tmp_path: Path) -> None:
    with pytest.raises(g32.G32LaunchError, match="SSD waterfall"):
        g32.validate_run_root(tmp_path / "run", allow_local_test_only=False)
    assert g32.validate_run_root(tmp_path / "run", allow_local_test_only=True).is_dir()


def test_immutable_stage_store_is_content_addressed_and_contiguous(tmp_path: Path) -> None:
    store = g32.ImmutableStageStore(tmp_path, "a" * 64)
    first = store.publish("000_custody", {"exact": 1})
    assert first == store.publish("000_custody", {"exact": 1})
    second = store.publish("010_selected_base", {"exact": 2})
    assert first.name.startswith("000_custody.")
    assert second.name.startswith("010_selected_base.")
    assert store.load_prefix() == {
        "000_custody": {"exact": 1},
        "010_selected_base": {"exact": 2},
    }
    with pytest.raises(g32.G32LaunchError, match="conflicting"):
        store.publish("000_custody", {"exact": 9})


def test_immutable_stage_store_refuses_gap(tmp_path: Path) -> None:
    store = g32.ImmutableStageStore(tmp_path, "b" * 64)
    store.publish("000_custody", {"exact": 1})
    with pytest.raises(g32.G32LaunchError, match="non-contiguous"):
        store.publish("020_pair_index", {"exact": 3})


def test_immutable_stage_store_refuses_record_and_predecessor_drift(tmp_path: Path) -> None:
    store = g32.ImmutableStageStore(tmp_path, "f" * 64)
    first = store.publish("000_custody", {"exact": 1})
    raw = json.loads(first.read_text())
    raw["payload"] = {"exact": 9}
    raw["payload_sha256"] = hashlib.sha256(g32._canonical_json(raw["payload"])).hexdigest()
    first.write_bytes(g32._canonical_json(raw))
    with pytest.raises(g32.G32LaunchError, match="filename/content digest"):
        store.load_prefix()

    other = g32.ImmutableStageStore(tmp_path / "other", "f" * 64)
    other.publish("000_custody", {"exact": 1})
    second = other.publish("010_selected_base", {"exact": 2})
    second_record = json.loads(second.read_text())
    second_record["predecessor_record_sha256"] = "0" * 64
    replacement_raw = g32._canonical_json(second_record)
    replacement = second.with_name(f"010_selected_base.{hashlib.sha256(replacement_raw).hexdigest()}.json")
    second.rename(replacement)
    replacement.write_bytes(replacement_raw)
    with pytest.raises(g32.G32LaunchError, match="predecessor"):
        other.load_prefix()


def test_chunk_store_reopens_and_hashes_exact_ranges(tmp_path: Path) -> None:
    raw = tmp_path / "pairs.raw"
    raw.write_bytes(b"0123456789")
    store = g32.ChunkStore(tmp_path, "c" * 64)
    store.publish("selected", 0, 1, raw, 2, 4, {"real": True})
    row = store.lookup("selected", 0, 1, raw)
    assert row is not None
    assert row["range_sha256"] == hashlib.sha256(b"2345").hexdigest()
    raw.write_bytes(b"01xxxx6789")
    with pytest.raises(g32.G32LaunchError, match="differs"):
        store.lookup("selected", 0, 1, raw)


def test_chunk_store_refuses_filename_and_coordinate_drift(tmp_path: Path) -> None:
    raw = tmp_path / "pairs.raw"
    raw.write_bytes(b"0123456789")
    store = g32.ChunkStore(tmp_path, "7" * 64)
    checkpoint = store.publish("selected", 0, 1, raw, 0, 5, {"exact": True})
    record = json.loads(checkpoint.read_text())
    record["telemetry"] = {"exact": False}
    checkpoint.write_bytes(g32._canonical_json(record))
    with pytest.raises(g32.G32LaunchError, match="filename/content digest"):
        store.lookup("selected", 0, 1, raw)

    other = g32.ChunkStore(tmp_path / "other", "7" * 64)
    coordinate_checkpoint = other.publish("selected", 0, 1, raw, 0, 5, {"exact": True})
    coordinate_record = json.loads(coordinate_checkpoint.read_text())
    coordinate_record["stop_pair"] = 2
    replacement_raw = g32._canonical_json(coordinate_record)
    replacement = coordinate_checkpoint.with_name(
        f"selected.0000-0001.{hashlib.sha256(replacement_raw).hexdigest()}.json"
    )
    coordinate_checkpoint.rename(replacement)
    replacement.write_bytes(replacement_raw)
    with pytest.raises(g32.G32LaunchError, match="coordinates"):
        other.lookup("selected", 0, 1, raw)


def test_fit_range_store_reopens_chain_and_refuses_mutation(tmp_path: Path) -> None:
    store = g32.FitRangeStore(tmp_path, "9" * 64)
    first = store.publish("050_base_feature", 0, 1, {"records": [1]})
    store.publish("050_base_feature", 1, 2, {"records": [2]})
    assert store.load_all()["050_base_feature"] == (
        {"first_pair": 0, "stop_pair": 1, "payload": {"records": [1]}},
        {"first_pair": 1, "stop_pair": 2, "payload": {"records": [2]}},
    )
    record = json.loads(first.read_text())
    record["payload"] = {"records": [99]}
    record["payload_sha256"] = hashlib.sha256(g32._canonical_json(record["payload"])).hexdigest()
    first.write_bytes(g32._canonical_json(record))
    with pytest.raises(g32.G32LaunchError, match="drifted"):
        store.load_stage("050_base_feature")

    other = g32.FitRangeStore(tmp_path / "other", "9" * 64)
    other.publish("040_xip2", 0, 1, {"pair_solves": [1]})
    second = other.publish("040_xip2", 1, 2, {"pair_solves": [2]})
    second_record = json.loads(second.read_text())
    second_record["predecessor_record_sha256"] = "0" * 64
    replacement_raw = g32._canonical_json(second_record)
    replacement = second.with_name(f"040_xip2.0001-0002.{hashlib.sha256(replacement_raw).hexdigest()}.json")
    second.rename(replacement)
    replacement.write_bytes(replacement_raw)
    with pytest.raises(g32.G32LaunchError, match="predecessor"):
        other.load_stage("040_xip2")


def test_selected_materialization_uses_one_process_for_tail_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(g32, "HEIGHT", 1)
    monkeypatch.setattr(g32, "WIDTH", 2)
    monkeypatch.setattr(g32, "CHANNELS", 3)
    pair_count = 4
    chunk_pairs = 2
    pair_bytes = 2 * g32.HEIGHT * g32.WIDTH * g32.CHANNELS
    raw = tmp_path / "selected.raw"
    runtime = tmp_path / "inflate.py"
    member = tmp_path / "member.bin"
    runtime.write_text("runtime")
    member.write_bytes(b"member")
    chunks = g32.ChunkStore(tmp_path, "8" * 64)
    calls: list[dict[str, object]] = []

    def fake_subprocess(command, *, environment=None):
        assert environment is not None
        plan = json.loads(Path(command[-1]).read_text())
        calls.append(plan)
        store = g32.ChunkStore(Path(plan["run_root"]), plan["binding_sha256"])
        for first in range(plan["first_pair"], plan["pair_count"], plan["chunk_pairs"]):
            stop = min(plan["pair_count"], first + plan["chunk_pairs"])
            payload = bytes([first + 1]) * ((stop - first) * pair_bytes)
            g32._write_range(Path(plan["output"]), first * pair_bytes, payload)
            store.publish(
                "selected",
                first,
                stop,
                Path(plan["output"]),
                first * pair_bytes,
                len(payload),
                {"one_setup_for_population_tail": True},
            )
        return {"setup_calls": 1, "first_pair": plan["first_pair"]}

    monkeypatch.setattr(g32, "_subprocess_json", fake_subprocess)
    g32.materialize_selected_base(
        raw,
        runtime,
        member,
        pair_count=pair_count,
        chunk_pairs=chunk_pairs,
        run_root=tmp_path,
        chunk_store=chunks,
    )
    assert len(calls) == 1
    assert calls[0]["first_pair"] == 0
    g32.materialize_selected_base(
        raw,
        runtime,
        member,
        pair_count=pair_count,
        chunk_pairs=chunk_pairs,
        run_root=tmp_path,
        chunk_store=chunks,
    )
    assert len(calls) == 1


def test_storage_preflight_includes_all_streaming_work_arrays(tmp_path: Path) -> None:
    row = g32.storage_preflight(tmp_path, pair_count=1, reserve_bytes=0)
    pair_bytes = 2 * g32.HEIGHT * g32.WIDTH * g32.CHANNELS
    assert row["required_bytes"] == 5 * pair_bytes + g32.HEIGHT * g32.WIDTH
    with pytest.raises(g32.G32LaunchError, match="storage preflight"):
        g32.storage_preflight(tmp_path, pair_count=1, reserve_bytes=row["free_bytes"] + 1)


def test_source_materialization_is_one_linear_process_and_range_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(g32, "HEIGHT", 2)
    monkeypatch.setattr(g32, "WIDTH", 3)
    monkeypatch.setattr(g32, "CHANNELS", 3)
    monkeypatch.setattr(g32.shutil, "which", lambda _name: "/fake/ffmpeg")
    pair_count = 3
    pair_bytes = 2 * g32.HEIGHT * g32.WIDTH * g32.CHANNELS
    source = bytes(range(pair_count * pair_bytes))
    commands: list[list[str]] = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            commands.append(list(command))
            self.stdout = io.BytesIO(source)
            self.returncode = None

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(g32.subprocess, "Popen", FakeProcess)
    raw = tmp_path / "source.raw"
    chunks = g32.ChunkStore(tmp_path, "d" * 64)
    g32.materialize_source_pairs(
        raw,
        tmp_path / "source.mkv",
        pair_count=pair_count,
        chunk_pairs=1,
        chunk_store=chunks,
    )
    assert raw.read_bytes() == source
    assert len(commands) == 1
    assert "select=" not in " ".join(commands[0])
    assert commands[0][commands[0].index("-frames:v") + 1] == "6"
    assert all(chunks.lookup("source", pair, pair + 1, raw) is not None for pair in range(pair_count))

    g32.materialize_source_pairs(
        raw,
        tmp_path / "source.mkv",
        pair_count=pair_count,
        chunk_pairs=1,
        chunk_store=chunks,
    )
    assert len(commands) == 1


def test_source_materialization_revalidates_published_prefix_in_single_resume_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(g32, "HEIGHT", 1)
    monkeypatch.setattr(g32, "WIDTH", 2)
    monkeypatch.setattr(g32, "CHANNELS", 3)
    monkeypatch.setattr(g32.shutil, "which", lambda _name: "/fake/ffmpeg")
    pair_bytes = 2 * g32.HEIGHT * g32.WIDTH * g32.CHANNELS
    source = bytes(range(2 * pair_bytes))
    raw = tmp_path / "source.raw"
    g32._preallocate(raw, len(source))
    raw.write_bytes(source[:pair_bytes] + bytes(pair_bytes))
    chunks = g32.ChunkStore(tmp_path, "e" * 64)
    chunks.publish("source", 0, 1, raw, 0, pair_bytes, {"seeded_prefix": True})

    class FakeProcess:
        def __init__(self, _command, **_kwargs):
            self.stdout = io.BytesIO(source)
            self.returncode = None

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(g32.subprocess, "Popen", FakeProcess)
    g32.materialize_source_pairs(
        raw,
        tmp_path / "source.mkv",
        pair_count=2,
        chunk_pairs=1,
        chunk_store=chunks,
    )
    assert raw.read_bytes() == source
    assert chunks.lookup("source", 1, 2, raw) is not None


def test_cleanup_certifies_before_preserve_or_delete(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    scratch = run_root / "scratch" / "rebuildable.bin"
    scratch.parent.mkdir(parents=True)
    scratch.write_bytes(b"rebuildable")
    custody = {
        "source_video": {"sha256": "1" * 64},
        "runtime": {"sha256": "2" * 64},
        "selected_archive": {"sha256": "3" * 64},
        "r10_receiver": {"sha256": "4" * 64},
        "g22_full_n600_receipt": {"sha256": "5" * 64},
    }
    preserved = g32.cleanup_certify_or_block(
        run_root,
        [scratch],
        rebuild_command=("python", "fit.py"),
        custody=custody,
        preserve_scratch=True,
    )
    assert scratch.exists()
    assert preserved["preserved_paths"] == [str(scratch)]
    removed = g32.cleanup_certify_or_block(
        run_root,
        [scratch],
        rebuild_command=("python", "fit.py"),
        custody=custody,
        preserve_scratch=False,
    )
    assert not scratch.exists()
    assert removed["removed_paths"] == [str(scratch)]
    assert g32._verify_retained_cleanup(run_root, removed) == removed


def test_cleanup_intent_recovers_crash_after_deletion_before_certificate(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    scratch = run_root / "scratch" / "rebuildable.bin"
    scratch.parent.mkdir(parents=True)
    scratch.write_bytes(b"rebuildable")
    custody = {
        "source_video": {"sha256": "1" * 64},
        "runtime": {"sha256": "2" * 64},
        "selected_archive": {"sha256": "3" * 64},
        "r10_receiver": {"sha256": "4" * 64},
        "g22_full_n600_receipt": {"sha256": "5" * 64},
    }
    g32.cleanup_certify_or_block(
        run_root,
        [scratch],
        rebuild_command=("python", "fit.py"),
        custody=custody,
        preserve_scratch=True,
    )
    scratch.unlink()
    recovered = g32.cleanup_certify_or_block(
        run_root,
        [scratch],
        rebuild_command=("python", "fit.py"),
        custody=custody,
        preserve_scratch=False,
    )
    assert recovered["removed_paths"] == [str(scratch)]
    assert g32._verify_retained_cleanup(run_root, recovered) == recovered


def test_complete_resume_reopens_exact_packet_wrapper_and_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(g32, "parse_r10_packet", lambda payload: payload)
    monkeypatch.setattr(g32, "serialize_r10_packet", lambda payload: payload)
    run_root = tmp_path / "run"
    artifacts = run_root / "artifacts"
    artifacts.mkdir(parents=True)
    packet = b"exact packet"
    packet_sha256 = hashlib.sha256(packet).hexdigest()
    packet_path = artifacts / f"r10_packet.{packet_sha256}.bin"
    packet_path.write_bytes(packet)
    wrapper_buffer = io.BytesIO()
    with zipfile.ZipFile(wrapper_buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("r10.packet", packet)
    wrapper = wrapper_buffer.getvalue()
    wrapper_sha256 = hashlib.sha256(wrapper).hexdigest()
    wrapper_path = artifacts / f"r10_packet_wrapper.{wrapper_sha256}.zip"
    wrapper_path.write_bytes(wrapper)
    receipt = {
        "packet": {"bytes": len(packet), "sha256": packet_sha256},
        "wrapper_zip": {"bytes": len(wrapper), "sha256": wrapper_sha256, "member": "r10.packet"},
        "artifact_paths": {"packet": str(packet_path), "wrapper_zip": str(wrapper_path)},
    }
    receipt_raw = g32._canonical_json(receipt)
    receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    receipt_path = artifacts / f"receipt.{receipt_sha256}.json"
    receipt_path.write_bytes(receipt_raw)
    stage = {
        "executed": True,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha256,
        "blocker": None,
    }
    reopened, *_paths = g32._verify_retained_result_artifacts(run_root, stage)
    assert reopened == receipt
    packet_path.write_bytes(b"drift")
    with pytest.raises(g32.G32LaunchError, match=r"filename|receipt"):
        g32._verify_retained_result_artifacts(run_root, stage)
