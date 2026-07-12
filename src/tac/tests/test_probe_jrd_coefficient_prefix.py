# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import brotli
import pytest

REPO = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO / "tools/probe_jrd_coefficient_prefix.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("probe_jrd_coefficient_prefix_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeLbc:
    _magic = b"LVLS1\x00"

    @classmethod
    def _io_pack(cls, manifest, base, code, pose, lane=None, pcar=None):
        chunks = [manifest, base, code, pose or b""]
        out = bytearray(cls._magic)
        for chunk in chunks:
            out += struct.pack("<I", len(chunk)) + chunk
        decoded = json.loads(manifest)
        if decoded.get("lane_render_band") is not None:
            assert lane is not None
            out += struct.pack("<I", len(lane)) + lane
        if decoded.get("pose_carrier") is not None:
            assert pcar is not None
            out += struct.pack("<I", len(pcar)) + pcar
        return bytes(out)

    @classmethod
    def _read_blob_bytes(cls, blob):
        assert blob.startswith(cls._magic)
        offset = len(cls._magic)
        chunks = []
        for _ in range(4):
            (size,) = struct.unpack_from("<I", blob, offset)
            offset += 4
            chunks.append(blob[offset : offset + size])
            offset += size
        manifest = json.loads(chunks[0])
        lane = None
        pcar = None
        if manifest.get("lane_render_band") is not None:
            (size,) = struct.unpack_from("<I", blob, offset)
            offset += 4
            lane = blob[offset : offset + size]
            offset += size
        if manifest.get("pose_carrier") is not None:
            (size,) = struct.unpack_from("<I", blob, offset)
            offset += 4
            pcar = blob[offset : offset + size]
            offset += size
        assert offset == len(blob)
        return manifest, chunks[1], chunks[2], chunks[3], lane, pcar


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    return _load_tool()


def _simple_blob(tool: ModuleType) -> tuple[bytes, bytes]:
    manifest = {
        "n_pairs": 3,
        "code_shape": [6, 2],
        "code_scale": 0.125,
    }
    code_raw = bytes(range(12))
    blob = FakeLbc._io_pack(
        json.dumps(manifest, separators=(",", ":")).encode(),
        brotli.compress(b"base", quality=11),
        brotli.compress(code_raw, quality=11),
        b"",
    )
    return blob, code_raw


def test_exact_pair_cap_preserves_code_prefix_and_scale(tool: ModuleType) -> None:
    blob, code_raw = _simple_blob(tool)
    capped, proof = tool.exact_pair_cap_blob(FakeLbc, blob, eval_pairs=1)
    manifest, _base, code_b, _pose, _lane, _pcar = FakeLbc._read_blob_bytes(capped)
    assert manifest["n_pairs"] == 1
    assert manifest["code_shape"] == [2, 2]
    assert manifest["code_scale"] == 0.125
    assert brotli.decompress(code_b) == code_raw[:4]
    assert proof["code_scale_unchanged"] is True
    assert proof["code_prefix_exact"] is True


def test_exact_pair_cap_rejects_non_pair_local_code_shape(tool: ModuleType) -> None:
    manifest = {"n_pairs": 3, "code_shape": [3, 4], "code_scale": 1.0}
    blob = FakeLbc._io_pack(
        json.dumps(manifest).encode(),
        brotli.compress(b"base"),
        brotli.compress(bytes(range(12))),
        b"",
    )
    with pytest.raises(ValueError, match=r"2\*n_pairs"):
        tool.exact_pair_cap_blob(FakeLbc, blob, eval_pairs=1)


def test_repack_blob_is_identity_for_unchanged_parts(tool: ModuleType) -> None:
    blob, _code_raw = _simple_blob(tool)
    parts = tool.parse_blob(FakeLbc, blob)
    assert tool.repack_blob(
        FakeLbc, parts, base_raw=parts.base_raw, code_raw=parts.code_raw
    ) == blob


def test_receipt_path_must_be_durable_results_path(tool: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="experiments/results"):
        tool.refuse_transient_evidence_path(tmp_path)
    with pytest.raises(ValueError, match="must not be under"):
        tool.refuse_transient_evidence_path(Path("/tmp/jrd"))


def test_receiver_cache_is_never_authority_even_when_mutated(
    tool: ModuleType, tmp_path: Path
) -> None:
    blob, _code_raw = _simple_blob(tool)

    class ReplayLbc(FakeLbc):
        CAMERA_H = 1
        CAMERA_W = 1
        calls = 0
        twr = SimpleNamespace(
            cpu_verdict_d_seg_batch=lambda *_args: [0.1],
            cpu_verdict_d_pose_batch=lambda *_args: [0.2],
        )

        @classmethod
        def assemble_packet(cls, _blob, packet_dir):
            packet_dir.mkdir(parents=True, exist_ok=True)
            archive = packet_dir / "archive.zip"
            archive.write_bytes(b"deterministic-archive")
            return archive, archive.stat().st_size

        @classmethod
        def run_inflate(cls, packet_dir, _pairs, _max_pairs):
            cls.calls += 1
            raw = packet_dir / "pair.raw"
            raw.write_bytes(bytes([1, 2, 3, 4, 5, 6]))
            return {"raw_path": str(raw)}

    out_dir = tmp_path / "receipt"
    (out_dir / "scratch").mkdir(parents=True)
    gt = SimpleNamespace(lstars=[object()], gt_poses=[object()])
    first = tool.measure_blob(
        ReplayLbc,
        full_blob=blob,
        label="first",
        run_fingerprint="f" * 64,
        out_dir=out_dir,
        gt=gt,
        segnet=object(),
        posenet=object(),
    )
    cache_path = next((out_dir / "receiver_cache").glob("*.json"))
    cache = json.loads(cache_path.read_text())
    cache["metrics"]["d_seg"] = 999.0
    tool.atomic_write_json(cache_path, cache)
    second = tool.measure_blob(
        ReplayLbc,
        full_blob=blob,
        label="repeat",
        run_fingerprint="f" * 64,
        out_dir=out_dir,
        gt=gt,
        segnet=object(),
        posenet=object(),
    )
    assert ReplayLbc.calls == 2
    assert first["raw_sha256_local_host"] == second["raw_sha256_local_host"]
    assert second["d_seg"] == 0.1
    assert second["receiver_cache"] == "receiver_replayed_cache_never_authoritative"


def test_run_fingerprint_binds_code_bytes_not_mutable_head(tool: ModuleType) -> None:
    payload = tool.run_fingerprint_payload()
    assert "tool_sha256" in payload
    assert "prefix_core_sha256" in payload
    assert "git_head_at_start" not in payload
    assert set(payload["executed_stack_sha256"]) == set(tool.EXECUTED_STACK_FILES)
    assert len(payload["shipped_inflate_source_sha256"]) == 64
    scorer_distributions = {
        "torch_distribution",
        "timm_distribution",
        "segmentation_models_pytorch_distribution",
        "einops_distribution",
        "safetensors_distribution",
        "torchvision_distribution",
        "av_distribution",
    }
    assert scorer_distributions <= payload["runtime"].keys()
    assert all(payload["runtime"][key] != "MISSING" for key in scorer_distributions)
    assert "libavcodec" in payload["runtime"]["av_library_versions"]
    assert "relevant_env" in payload["runtime"]
    assert (
        "src/tac/optimization/frame1_seg_repair_atoms.py"
        in payload["executed_stack_sha256"]
    )


def test_receipt_path_refuses_protected_live_run(tool: ModuleType) -> None:
    with pytest.raises(ValueError, match="protected live run"):
        tool.refuse_transient_evidence_path(tool.PROTECTED_LIVE_RUN / "jrd_probe")


def test_parent_command_capture_records_permission_denial_without_fabrication(
    tool: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def deny(*_args, **_kwargs):
        raise PermissionError(1, "Operation not permitted", "ps")

    monkeypatch.setattr(tool.subprocess, "run", deny)
    receipt = tool.process_command(123)
    assert receipt == {
        "pid_at_capture": 123,
        "command": None,
        "capture_status": "unavailable",
        "capture_error_class": "PermissionError",
        "capture_errno": 1,
        "review_status": "UNKNOWN_command_capture_denied",
    }


def test_receiver_failure_retains_scratch_for_certify_or_block(
    tool: ModuleType, tmp_path: Path
) -> None:
    blob, _code_raw = _simple_blob(tool)

    class FailingLbc(FakeLbc):
        @classmethod
        def assemble_packet(cls, _blob, packet_dir):
            packet_dir.mkdir(parents=True, exist_ok=True)
            archive = packet_dir / "archive.zip"
            archive.write_bytes(b"archive")
            return archive, archive.stat().st_size

        @classmethod
        def run_inflate(cls, packet_dir, _pairs, _max_pairs):
            (packet_dir / "partial.raw").write_bytes(b"failure evidence")
            raise RuntimeError("receiver boundary failed")

    out_dir = tmp_path / "receipt"
    (out_dir / "scratch").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="receiver boundary failed"):
        tool.measure_blob(
            FailingLbc,
            full_blob=blob,
            label="failure",
            run_fingerprint="c" * 64,
            out_dir=out_dir,
            gt=object(),
            segnet=object(),
            posenet=object(),
        )
    retained = list((out_dir / "scratch").glob("candidate_*"))
    assert len(retained) == 1
    assert (retained[0] / "cap_packet" / "partial.raw").read_bytes() == b"failure evidence"


def test_cleanup_retains_and_blocks_uncertified_crash_scratch(
    tool: ModuleType, tmp_path: Path
) -> None:
    orphan = tmp_path / "scratch" / "candidate_crash"
    orphan.mkdir(parents=True)
    (orphan / "pair.raw").write_bytes(b"not-certified")
    with pytest.raises(RuntimeError, match="retained"):
        tool.cleanup_scratch(tmp_path, run_fingerprint="a" * 64)
    assert (orphan / "pair.raw").read_bytes() == b"not-certified"
    blocker = json.loads((tmp_path / "scratch" / "orphan_blocker.json").read_text())
    assert blocker["blocker"] == "uncertified_crash_left_candidate_scratch"
    assert blocker["retained_artifacts"][0]["total_bytes"] == len(b"not-certified")


def test_custody_fails_closed_when_canonical_scorer_contract_fails(
    tool: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tool,
        "build_upstream_eval_contract",
        lambda **_kwargs: {"contract_valid": False, "blockers": ["model_sha256_mismatch"]},
    )
    with pytest.raises(RuntimeError, match="canonical upstream scorer"):
        tool.verify_custody()


def test_resume_registry_requires_exact_reverification_for_atomic_candidate(
    tool: ModuleType, tmp_path: Path
) -> None:
    fingerprint = "b" * 64
    initial = tool.restore_or_initialize_resume(tmp_path, run_fingerprint=fingerprint)
    assert initial["restored"] is False
    candidate = tmp_path / "candidates" / "row.json"
    tool.atomic_write_json(
        candidate,
        {
            "status": "complete",
            "run_fingerprint": fingerprint,
            "label": "candidate",
        },
    )
    resumed = tool.restore_or_initialize_resume(tmp_path, run_fingerprint=fingerprint)
    assert resumed["restored"] is True
    assert resumed["recovered"] is False
    assert resumed["recovery_pending"] is True
    assert resumed["registered_checkpoint_paths"] == []
    assert resumed["pending_checkpoint_paths"] == ["candidates/row.json"]
    assert resumed["count"] == 1
    assert (tmp_path / "resume" / "resume_recovery.json").is_file()


def test_resume_registry_refuses_mutated_registered_candidate(
    tool: ModuleType, tmp_path: Path
) -> None:
    fingerprint = "d" * 64
    tool.restore_or_initialize_resume(tmp_path, run_fingerprint=fingerprint)
    candidate = tmp_path / "candidates" / "row.json"
    tool.atomic_write_json(
        candidate,
        {
            "status": "complete",
            "run_fingerprint": fingerprint,
            "label": "candidate",
            "d_seg": 0.1,
        },
    )
    pending = tool.restore_or_initialize_resume(tmp_path, run_fingerprint=fingerprint)
    assert pending["recovery_pending"] is True
    tool.write_resume_state(
        tmp_path,
        run_fingerprint=fingerprint,
        stage="candidate_reverified",
        verified_checkpoint_paths={"candidates/row.json"},
    )
    tool.atomic_write_json(
        candidate,
        {
            "status": "complete",
            "run_fingerprint": fingerprint,
            "label": "candidate",
            "d_seg": 0.2,
        },
    )
    with pytest.raises(tool.ResumeIntegrityError, match="mutated or lost"):
        tool.restore_or_initialize_resume(tmp_path, run_fingerprint=fingerprint)


def test_run_loop_reuse_gate_refuses_unregistered_crash_extra(
    tool: ModuleType, tmp_path: Path
) -> None:
    fingerprint = "9" * 64
    out_dir = tmp_path / "run"
    checkpoint = out_dir / "candidates" / "candidate.json"
    tool.atomic_write_json(
        checkpoint,
        {
            "status": "complete",
            "run_fingerprint": fingerprint,
            "section": "a",
            "family": "uniform",
            "bits_removed": 1,
            "d_seg": 999.0,
        },
    )
    assert (
        tool.load_registered_checkpoint_or_none(
            checkpoint,
            out_dir=out_dir,
            verified_checkpoint_paths=set(),
            run_fingerprint=fingerprint,
            expected_fields={"section": "a", "family": "uniform", "bits_removed": 1},
        )
        is None
    )
    loaded = tool.load_registered_checkpoint_or_none(
        checkpoint,
        out_dir=out_dir,
        verified_checkpoint_paths={"candidates/candidate.json"},
        run_fingerprint=fingerprint,
        expected_fields={"section": "a", "family": "uniform", "bits_removed": 1},
    )
    assert loaded is not None and loaded["d_seg"] == 999.0


@pytest.mark.parametrize(
    ("d_seg", "d_pose", "archive_bytes", "accepted"),
    [
        (0.1, 0.2, 99, True),
        (0.100001, 0.0, 90, False),
        (0.0, 0.200001, 90, False),
        (0.1, 0.2, 100, False),
        (0.1, 0.2, 101, False),
    ],
)
def test_combined_step_gate_requires_both_component_safety_and_zip_shrinkage(
    tool: ModuleType,
    d_seg: float,
    d_pose: float,
    archive_bytes: int,
    accepted: bool,
) -> None:
    gate = tool.combined_step_gate(
        {"d_seg": d_seg, "d_pose": d_pose, "archive_zip_bytes": archive_bytes},
        baseline={"d_seg": 0.1, "d_pose": 0.2},
        current_archive_bytes=100,
    )
    assert gate["accepted"] is accepted
    assert gate["accepted"] is (
        gate["safe_vs_sealed_baseline"]
        and gate["improves_vs_current_combined_bytes"]
    )


def test_combined_step_gate_refuses_fractional_byte_counts(tool: ModuleType) -> None:
    with pytest.raises(ValueError, match="must be integers"):
        tool.combined_step_gate(
            {"d_seg": 0.1, "d_pose": 0.2, "archive_zip_bytes": 99.9},
            baseline={"d_seg": 0.1, "d_pose": 0.2},
            current_archive_bytes=100,
        )


def test_payload_inventory_finds_only_byte_closed_v9_lvls1(
    tool: ModuleType, tmp_path: Path
) -> None:
    packet = tmp_path / "experiments" / "v9_candidate"
    packet.mkdir(parents=True)
    with zipfile.ZipFile(packet / "archive.zip", "w") as archive:
        archive.writestr("0.bin", b"LVLS1\x00payload")
    (packet / "vehicle_provenance.json").write_text(
        json.dumps(
            {
                "vehicle_family": "v9",
                "payload_sha256": tool.sha256_file(packet / "archive.zip"),
                "byte_closed": True,
            }
        ),
        encoding="utf-8",
    )
    (packet / "levelset_witness_ema.npz").write_bytes(b"checkpoint")
    inventory = tool.build_payload_inventory((tmp_path,))
    assert inventory["eligible_count"] == 1
    assert {row["kind"] for row in inventory["records"]} == {
        "archive_zip",
        "checkpoint_not_byte_closed",
    }


def test_payload_inventory_path_token_is_advisory_only(
    tool: ModuleType, tmp_path: Path
) -> None:
    packet = tmp_path / "experiments" / "v9_candidate"
    packet.mkdir(parents=True)
    with zipfile.ZipFile(packet / "archive.zip", "w") as archive:
        archive.writestr("0.bin", b"LVLS1\x00payload")

    inventory = tool.build_payload_inventory((tmp_path,))
    assert inventory["eligible_count"] == 0
    assert inventory["unclassified_lvls1_count"] == 1
    assert inventory["records"][0]["path_family_marker"] == "v9"
    assert inventory["records"][0]["typed_vehicle_provenance"] is None


def test_payload_inventory_refuses_symlinked_vehicle_provenance(
    tool: ModuleType, tmp_path: Path
) -> None:
    packet = tmp_path / "experiments" / "v9_candidate"
    packet.mkdir(parents=True)
    with zipfile.ZipFile(packet / "archive.zip", "w") as archive:
        archive.writestr("0.bin", b"LVLS1\x00payload")
    external = tmp_path / "external_vehicle_provenance.json"
    external.write_text(
        json.dumps(
            {
                "vehicle_family": "v9",
                "payload_sha256": tool.sha256_file(packet / "archive.zip"),
                "byte_closed": True,
            }
        ),
        encoding="utf-8",
    )
    (packet / "vehicle_provenance.json").symlink_to(external)

    inventory = tool.build_payload_inventory((tmp_path / "experiments",))

    assert inventory["eligible_count"] == 0
    assert inventory["records"][0]["typed_vehicle_provenance"] is None


def test_payload_inventory_refuses_candidate_symlink_without_reading_target(
    tool: ModuleType, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    protected = tmp_path / "protected.bin"
    protected.write_bytes(b"LVLS1\x00secret")
    candidate = root / "0.bin"
    candidate.symlink_to(protected)

    inventory = tool.build_payload_inventory((root,))

    assert inventory["records"] == []
    assert inventory["skipped_symlink_candidates"] == [
        {
            "path": str(candidate.absolute()),
            "reason": "candidate symlink refused without resolving or reading target",
        }
    ]


def test_system_integration_executes_canonical_consumers_without_promotion(
    tool: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tac.canonical_task_status.loader as task_loader
    import tac.canonical_task_status.writer as task_writer
    import tac.probe_outcomes_ledger as probe_ledger

    class TaskRow:
        status = "in_progress"

        def to_json_obj(self):
            return {"task_id": "v9_jrd_coeff_prefix_probe_20260712", "status": self.status}

    blocked = TaskRow()
    blocked.status = "blocked"
    registered_probe_rows = []
    monkeypatch.setattr(task_loader, "latest_status_by_task_id", lambda *_args: TaskRow())
    monkeypatch.setattr(task_writer, "update_status", lambda *_args, **_kwargs: blocked)
    monkeypatch.setattr(probe_ledger, "query_by_probe_id", lambda *_args: [])

    def register_probe(**row):
        registered_probe_rows.append(row)
        return {**row, "written_at_utc": "test"}

    monkeypatch.setattr(probe_ledger, "register_probe_outcome", register_probe)

    results_root = REPO / "experiments/results"
    with tempfile.TemporaryDirectory(prefix="jrd_integration_test_", dir=results_root) as tmp:
        out_dir = Path(tmp)
        run_fingerprint = "e" * 64
        advisory_score = tool.compute_contest_score(0.1, 0.2, 2)
        receipt = {
            "schema": "jrd_coefficient_prefix_measurement.v1",
            "status": "complete",
            "run_fingerprint": run_fingerprint,
            "task_id": "v9_jrd_coeff_prefix_probe_20260712",
            "task_verdict": "NEEDS-MORE",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "fixture_verdict": "NO-GO",
            "baseline": {
                "status": "complete",
                "label": "positive_control_baseline",
                "run_fingerprint": run_fingerprint,
                "archive_zip_bytes": 2,
                "archive_zip_sha256": "a" * 64,
                "full_blob_sha256": "b" * 64,
                "raw_sha256_local_host": "c" * 64,
                "d_seg": 0.1,
                "d_pose": 0.2,
                "implied_score_advisory": advisory_score,
            },
            "selected": {
                "run_fingerprint": run_fingerprint,
                "archive_zip_bytes": 2,
                "archive_zip_sha256": "a" * 64,
                "full_blob_sha256": "b" * 64,
                "raw_sha256_local_host": "c" * 64,
                "d_seg": 0.1,
                "d_pose": 0.2,
                "implied_score_advisory": advisory_score,
            },
            "delta": {
                "archive_bytes_saved": 0,
                "d_seg": 0.0,
                "d_pose": 0.0,
                "implied_score_advisory": 0.0,
            },
            "payload_inventory": {
                "path": "v9_v8_payload_inventory.json",
                "sha256": "1" * 64,
                "eligible_count": 0,
                "unclassified_lvls1_count": 0,
            },
            "controls": {
                "positive_baseline": "controls/baseline.json",
                "positive_repeat": "controls/baseline_repeat.json",
                "negative_all_zero": "controls/all_zero_negative.json",
            },
            "boundaries": {
                "eligible_v9_v8_payload": False,
                "eval_pairs": 1,
                "upstream_evaluate_py_run": False,
                "contest_cpu_linux_x86_64": False,
                "contest_cuda": False,
            },
            "accepted_combined_steps": [],
            "rejected_combined_steps": [],
        }
        receipt["content_sha256"] = tool.sha256_bytes(
            tool.canonical_json_bytes(receipt)
        )
        tool.atomic_write_json(out_dir / "measurement_receipt.json", receipt)
        inventory = {
            "schema": "jrd_v9_v8_payload_inventory.v1",
            "records": [],
            "eligible_count": 0,
            "unclassified_lvls1_count": 0,
        }
        inventory["inventory_sha256"] = tool.sha256_bytes(
            tool.canonical_json_bytes(inventory)
        )
        receipt["payload_inventory"]["sha256"] = inventory["inventory_sha256"]
        receipt["content_sha256"] = tool.sha256_bytes(
            tool.canonical_json_bytes(
                {key: value for key, value in receipt.items() if key != "content_sha256"}
            )
        )
        tool.atomic_write_json(out_dir / "measurement_receipt.json", receipt)
        tool.atomic_write_json(out_dir / "v9_v8_payload_inventory.json", inventory)
        tool.atomic_write_json(out_dir / "controls/baseline.json", receipt["baseline"])
        tool.atomic_write_json(
            out_dir / "controls/baseline_repeat.json",
            {**receipt["baseline"], "label": "positive_control_baseline_repeat"},
        )
        tool.atomic_write_json(
            out_dir / "controls/all_zero_negative.json",
            {
                "status": "complete",
                "run_fingerprint": run_fingerprint,
                "raw_sha256_local_host": "9" * 64,
                "d_seg": 0.2,
                "d_pose": 0.3,
            },
        )
        response = {
            "schema": "jrd_section_precision_response_curves.v1",
            "status": "complete",
            "run_fingerprint": run_fingerprint,
            "baseline": {
                "archive_zip_bytes": 2,
                "archive_zip_sha256": "a" * 64,
                "d_seg": 0.1,
                "d_pose": 0.2,
            },
            "pareto_constraint": {
                "score_compensation_allowed": False,
                "d_seg_max": 0.1,
                "d_pose_max": 0.2,
            },
            "rows": [
                {
                    "section": section,
                    "family": family,
                    "bits_removed": bits_removed,
                    "archive_zip_bytes": 2,
                    "archive_zip_sha256": "a" * 64,
                    "d_seg": 0.1,
                    "d_pose": 0.2,
                    "section_coefficient_count": section_count,
                    "run_fingerprint": run_fingerprint,
                }
                for section, section_count in tool.sealed_response_sections().items()
                for family in tool.PREFIX_FAMILIES
                for bits_removed in range(1, tool.MAX_INT8_PREFIX_PLANES + 1)
            ],
        }
        response["content_sha256"] = tool.sha256_bytes(
            tool.canonical_json_bytes(response)
        )
        tool.atomic_write_json(out_dir / "section_precision_response_curves.json", response)
        allocator = {
            "schema": "jrd_prefix_allocator_planning_input.v1",
            "research_only": True,
            "promotion_eligible": False,
            "run_fingerprint": run_fingerprint,
            "control_law": "exact component guard",
            "proposed": [],
            "accepted": [],
            "rejected": [],
        }
        allocator["content_sha256"] = tool.sha256_bytes(
            tool.canonical_json_bytes(allocator)
        )
        tool.atomic_write_json(
            out_dir / "allocator_planning_input.json",
            allocator,
        )
        tool.atomic_write_json(
            out_dir / "probe_disambiguator_output.json",
            {
                "schema": "test",
                "tool": "test",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "autopilot_rows": [
                    {
                        "candidate_id": "candidate",
                        "family": "jrd",
                        "lane_class": "research_only",
                        "predicted_score_delta": 0.0,
                        "expected_information_gain": 0.0,
                        "estimated_dispatch_cost_usd": 0.0,
                        "blockers": ["research_only"],
                        "score_claim": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                        "dispatch_attempted": False,
                    }
                ],
            },
        )
        result = tool.integrate_system_intelligence(
            out_dir,
            run_fingerprint=run_fingerprint,
            posterior_candidate={
                "contest_result": {
                    "axis": "cpu",
                    "hardware_substrate": tool.local_cpu_axis()["hardware_substrate"],
                    "architecture_class": "jrd_coefficient_prefix_v75_fixture_pair0",
                    "score_value": advisory_score,
                    "evidence_tag": tool.local_cpu_axis()["evidence_tag"],
                    "archive_sha256": "a" * 64,
                    "archive_bytes": 2,
                    "cpu_seg": 0.1,
                    "cpu_pose": 0.2,
                }
            },
            probe_outcome_candidate={
                **tool.canonical_probe_outcome_fields(exact_bytes_saved=0),
                "recipe_path": "replaced",
                "evidence_path": "replaced",
            },
            task_hook_candidate={
                "task_id": "v9_jrd_coeff_prefix_probe_20260712",
                "status": "blocked",
                "test_status": "green",
                "blocker": "eligible_nonlive_v9_v8_payload_missing_or_unresolved",
                "actual_delta_s": None,
            },
        )

        assert result["continual_learning"]["posterior_update"]["accepted"] is False
        assert result["cathedral_autopilot"]["candidate_ids"] == ["candidate"]
        assert result["canonical_task_status"]["status"] == "blocked"
        assert registered_probe_rows[0]["evidence_path"].startswith(
            "experiments/results/jrd_integration_test_"
        )
        assert (out_dir / "component_sensitivity_manifest.json").is_file()
        (out_dir / "controls/all_zero_negative.json").unlink()
        with pytest.raises(RuntimeError, match="negative_all_zero is missing"):
            tool.load_measurement_receipt_for_integration(
                out_dir / "measurement_receipt.json",
                run_fingerprint=run_fingerprint,
            )


def test_content_addressed_integration_artifacts_refuse_empty_or_forged_payloads(
    tool: ModuleType, tmp_path: Path
) -> None:
    response_path = tmp_path / "response.json"
    tool.atomic_write_json(response_path, {})
    with pytest.raises(RuntimeError, match="schema mismatch"):
        tool.load_response_curves_for_integration(
            response_path,
            run_fingerprint="a" * 64,
            expected_sections={"code": 1},
        )

    allocator_path = tmp_path / "allocator.json"
    tool.atomic_write_json(
        allocator_path,
        {
            "schema": "jrd_prefix_allocator_planning_input.v1",
            "content_sha256": "f" * 64,
        },
    )
    with pytest.raises(RuntimeError, match="content hash mismatch"):
        tool.load_allocator_for_integration(
            allocator_path,
            run_fingerprint="a" * 64,
            response_curves={
                "baseline": {"archive_zip_bytes": 2, "d_seg": 0.1, "d_pose": 0.2},
                "rows": [],
            },
        )

    receipt_path = tmp_path / "measurement_receipt.json"
    tool.atomic_write_json(receipt_path, {})
    with pytest.raises(RuntimeError, match="schema mismatch"):
        tool.load_measurement_receipt_for_integration(
            receipt_path, run_fingerprint="a" * 64
        )


def test_integration_loaders_rederive_response_and_allocator_semantics(
    tool: ModuleType, tmp_path: Path
) -> None:
    fingerprint = "b" * 64
    response = {
        "schema": "jrd_section_precision_response_curves.v1",
        "status": "complete",
        "run_fingerprint": fingerprint,
        "baseline": {"archive_zip_bytes": 100, "d_seg": 0.1, "d_pose": 0.2},
        "pareto_constraint": {
            "score_compensation_allowed": False,
            "d_seg_max": 0.1,
            "d_pose_max": 0.2,
        },
        "rows": [
            {
                "section": "code",
                "family": "bogus",
                "bits_removed": 99,
                "archive_zip_bytes": -1,
                "section_coefficient_count": 1,
                "d_seg": "not-a-number",
                "d_pose": float("nan"),
                "run_fingerprint": fingerprint,
            }
        ],
    }
    response["content_sha256"] = tool.sha256_bytes(tool.canonical_json_bytes(response))
    response_path = tmp_path / "semantic_response.json"
    tool.atomic_write_json(response_path, response)
    with pytest.raises(RuntimeError, match="invalid prefix identity"):
        tool.load_response_curves_for_integration(
            response_path,
            run_fingerprint=fingerprint,
            expected_sections={"code": 1},
        )

    baseline = {"archive_zip_bytes": 100, "d_seg": 0.1, "d_pose": 0.2}
    choice = {
        "section": "code",
        "family": "uniform",
        "bits_removed": 1,
        "archive_bytes": 99,
        "d_seg": 0.1,
        "d_pose": 0.2,
        "archive_bytes_saved": 1,
        "raw_precision_bits_removed": 1,
    }
    allocator = {
        "schema": "jrd_prefix_allocator_planning_input.v1",
        "research_only": True,
        "promotion_eligible": False,
        "run_fingerprint": fingerprint,
        "proposed": [choice],
        "accepted": [
            {
                "label": "combined_step=0;section=code",
                "choice": choice,
                "accepted": True,
                "safe_vs_sealed_baseline": True,
                "improves_vs_current_combined_bytes": True,
                "current_combined_bytes_before": 100,
                "archive_zip_bytes": 999,
                "d_seg": 999.0,
                "d_pose": 999.0,
            }
        ],
        "rejected": [],
    }
    allocator["content_sha256"] = tool.sha256_bytes(
        tool.canonical_json_bytes(allocator)
    )
    allocator_path = tmp_path / "semantic_allocator.json"
    tool.atomic_write_json(allocator_path, allocator)
    with pytest.raises(RuntimeError, match="asserted a gate"):
        tool.load_allocator_for_integration(
            allocator_path,
            run_fingerprint=fingerprint,
            response_curves={
                "baseline": baseline,
                "rows": [
                    {
                        "section": "code",
                        "family": "uniform",
                        "bits_removed": 1,
                        "archive_zip_bytes": 99,
                        "d_seg": 0.1,
                        "d_pose": 0.2,
                        "section_coefficient_count": 1,
                    }
                ],
            },
        )

    safe_choice = {
        "section": "code",
        "family": "uniform",
        "bits_removed": 1,
        "archive_bytes": 999,
        "d_seg": 0.1,
        "d_pose": 0.2,
        "archive_bytes_saved": 1,
        "raw_precision_bits_removed": 1,
    }
    forged_chain = {
        "schema": "jrd_prefix_allocator_planning_input.v1",
        "research_only": True,
        "promotion_eligible": False,
        "run_fingerprint": fingerprint,
        "proposed": [safe_choice],
        "accepted": [
            {
                "label": "combined_step=0;section=code",
                "choice": safe_choice,
                "accepted": True,
                "safe_vs_sealed_baseline": True,
                "improves_vs_current_combined_bytes": True,
                "current_combined_bytes_before": 2000,
                "archive_zip_bytes": 999,
                "d_seg": 0.1,
                "d_pose": 0.2,
            }
        ],
        "rejected": [],
    }
    forged_chain["content_sha256"] = tool.sha256_bytes(
        tool.canonical_json_bytes(forged_chain)
    )
    forged_chain_path = tmp_path / "forged_chain.json"
    tool.atomic_write_json(forged_chain_path, forged_chain)
    with pytest.raises(RuntimeError, match="prior bytes"):
        tool.load_allocator_for_integration(
            forged_chain_path,
            run_fingerprint=fingerprint,
            response_curves={
                "baseline": {
                    "archive_zip_bytes": 1000,
                    "d_seg": 0.1,
                    "d_pose": 0.2,
                },
                "rows": [
                    {
                        "section": "code",
                        "family": "uniform",
                        "bits_removed": 1,
                        "archive_zip_bytes": 999,
                        "d_seg": 0.1,
                        "d_pose": 0.2,
                        "section_coefficient_count": 1,
                    }
                ],
            },
        )

    omitted = {
        "schema": "jrd_prefix_allocator_planning_input.v1",
        "research_only": True,
        "promotion_eligible": False,
        "run_fingerprint": fingerprint,
        "proposed": [],
        "accepted": [],
        "rejected": [],
    }
    omitted["content_sha256"] = tool.sha256_bytes(tool.canonical_json_bytes(omitted))
    omitted_path = tmp_path / "omitted_allocator.json"
    tool.atomic_write_json(omitted_path, omitted)
    with pytest.raises(RuntimeError, match="proposal list"):
        tool.load_allocator_for_integration(
            omitted_path,
            run_fingerprint=fingerprint,
            response_curves={
                "baseline": {"archive_zip_bytes": 100, "d_seg": 0.1, "d_pose": 0.2},
                "rows": [
                    {
                        "section": "code",
                        "family": "uniform",
                        "bits_removed": 1,
                        "archive_zip_bytes": 99,
                        "d_seg": 0.1,
                        "d_pose": 0.2,
                        "section_coefficient_count": 1,
                    }
                ],
            },
        )

    choice_b = {
        "section": "b",
        "family": "uniform",
        "bits_removed": 1,
        "archive_bytes": 80,
        "d_seg": 0.1,
        "d_pose": 0.2,
        "archive_bytes_saved": 20,
        "raw_precision_bits_removed": 1,
    }
    choice_a = {
        "section": "a",
        "family": "uniform",
        "bits_removed": 1,
        "archive_bytes": 90,
        "d_seg": 0.1,
        "d_pose": 0.2,
        "archive_bytes_saved": 10,
        "raw_precision_bits_removed": 1,
    }
    step_b = {
        "label": "combined_step=0;section=b",
        "choice": choice_b,
        "accepted": True,
        "safe_vs_sealed_baseline": True,
        "improves_vs_current_combined_bytes": True,
        "current_combined_bytes_before": 100,
        "archive_zip_bytes": 80,
        "d_seg": 0.1,
        "d_pose": 0.2,
    }
    step_a = {
        "label": "combined_step=1;section=a",
        "choice": choice_a,
        "accepted": True,
        "safe_vs_sealed_baseline": True,
        "improves_vs_current_combined_bytes": True,
        "current_combined_bytes_before": 80,
        "archive_zip_bytes": 70,
        "d_seg": 0.1,
        "d_pose": 0.2,
    }
    reversed_decisions = {
        "schema": "jrd_prefix_allocator_planning_input.v1",
        "research_only": True,
        "promotion_eligible": False,
        "run_fingerprint": fingerprint,
        "proposed": [choice_b, choice_a],
        "accepted": [step_a, step_b],
        "rejected": [],
    }
    reversed_decisions["content_sha256"] = tool.sha256_bytes(
        tool.canonical_json_bytes(reversed_decisions)
    )
    reversed_path = tmp_path / "reversed_decisions.json"
    tool.atomic_write_json(reversed_path, reversed_decisions)
    with pytest.raises(RuntimeError, match="sequence order"):
        tool.load_allocator_for_integration(
            reversed_path,
            run_fingerprint=fingerprint,
            response_curves={
                "baseline": {"archive_zip_bytes": 100, "d_seg": 0.1, "d_pose": 0.2},
                "rows": [
                    {
                        "section": "a",
                        "family": "uniform",
                        "bits_removed": 1,
                        "archive_zip_bytes": 90,
                        "d_seg": 0.1,
                        "d_pose": 0.2,
                        "section_coefficient_count": 1,
                    },
                    {
                        "section": "b",
                        "family": "uniform",
                        "bits_removed": 1,
                        "archive_zip_bytes": 80,
                        "d_seg": 0.1,
                        "d_pose": 0.2,
                        "section_coefficient_count": 1,
                    },
                ],
            },
        )

    with pytest.raises(RuntimeError, match="different sealed baselines"):
        tool.assert_cross_artifact_baseline_identity(
            {"baseline": {"archive_zip_sha256": "a" * 64, "archive_zip_bytes": 10, "d_seg": 0.1, "d_pose": 0.2}},
            {"baseline": {"archive_zip_sha256": "b" * 64, "archive_zip_bytes": 20, "d_seg": 0.1, "d_pose": 0.2}},
        )

    shell = {
        "schema": "jrd_coefficient_prefix_measurement.v1",
        "status": "complete",
        "run_fingerprint": fingerprint,
        "task_id": "v9_jrd_coeff_prefix_probe_20260712",
        "task_verdict": "NEEDS-MORE",
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "baseline": {},
        "selected": {},
        "delta": {},
        "payload_inventory": {},
        "controls": {},
        "boundaries": {},
        "accepted_combined_steps": [],
        "rejected_combined_steps": [],
    }
    shell["content_sha256"] = tool.sha256_bytes(tool.canonical_json_bytes(shell))
    shell_path = tmp_path / "shell_receipt.json"
    tool.atomic_write_json(shell_path, shell)
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        tool.load_measurement_receipt_for_integration(
            shell_path, run_fingerprint=fingerprint
        )

    incomplete = {
        "schema": "jrd_section_precision_response_curves.v1",
        "status": "complete",
        "run_fingerprint": fingerprint,
        "baseline": {"archive_zip_bytes": 100, "d_seg": 0.1, "d_pose": 0.2},
        "pareto_constraint": {
            "score_compensation_allowed": False,
            "d_seg_max": 0.1,
            "d_pose_max": 0.2,
        },
        "rows": [
            {
                "section": "code",
                "family": "uniform",
                "bits_removed": 1,
                "archive_zip_bytes": 99,
                "section_coefficient_count": 1,
                "d_seg": 0.1,
                "d_pose": 0.2,
                "run_fingerprint": fingerprint,
            }
        ],
    }
    incomplete["content_sha256"] = tool.sha256_bytes(
        tool.canonical_json_bytes(incomplete)
    )
    incomplete_path = tmp_path / "incomplete_response.json"
    tool.atomic_write_json(incomplete_path, incomplete)
    with pytest.raises(RuntimeError, match="invalid prefix identity"):
        tool.load_response_curves_for_integration(
            incomplete_path,
            run_fingerprint=fingerprint,
            expected_sections={"code": 2},
        )
    with pytest.raises(RuntimeError, match="every sealed section"):
        tool.load_response_curves_for_integration(
            incomplete_path,
            run_fingerprint=fingerprint,
            expected_sections={"code": 1},
        )


def test_existing_probe_outcome_refuses_stale_load_bearing_field(
    tool: ModuleType,
) -> None:
    candidate = {
        "probe_id": "probe",
        "verdict": "DEFER",
        "threshold": 1.0,
        "next_action": "rerun exact receiver",
    }
    tool.assert_existing_probe_outcome_matches(dict(candidate), candidate)
    stale = {**candidate, "next_action": "trust old result"}
    with pytest.raises(RuntimeError, match="next_action"):
        tool.assert_existing_probe_outcome_matches(stale, candidate)

    canonical = tool.canonical_probe_outcome_fields(exact_bytes_saved=0)
    incomplete = {"probe_id": canonical["probe_id"], "verdict": "DEFER"}
    with pytest.raises(RuntimeError, match="measured control law"):
        tool.assert_probe_outcome_control_law(incomplete, exact_bytes_saved=0)
    authority_override = {
        **canonical,
        "recipe_path": "receipt.json",
        "evidence_path": "receipt.json",
        "promotion_eligible": True,
        "ready_for_exact_eval_dispatch": True,
    }
    with pytest.raises(RuntimeError, match="measured control law"):
        tool.assert_probe_outcome_control_law(authority_override, exact_bytes_saved=0)

    with pytest.raises(RuntimeError, match="canonical formula"):
        tool.require_derived_advisory_score(
            {
                "d_seg": 0.1,
                "d_pose": 0.2,
                "archive_zip_bytes": 10,
                "implied_score_advisory": 999.0,
            },
            field="forged",
        )


def test_negative_control_cache_has_zero_authority(
    tool: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    negative_path = tmp_path / "all_zero_negative.json"
    tool.atomic_write_json(negative_path, {"d_seg": 999.0})
    calls = []

    def fake_measure(*_args, **_kwargs):
        calls.append(1)
        return {"status": "complete", "d_seg": 0.25, "d_pose": 0.5}

    monkeypatch.setattr(tool, "measure_blob", fake_measure)
    measured = tool.measure_negative_control(
        object(),
        negative_path=negative_path,
        zero_blob=b"zero",
        run_fingerprint="a" * 64,
        out_dir=tmp_path,
        gt=object(),
        segnet=object(),
        posenet=object(),
    )

    assert calls == [1]
    assert measured["d_seg"] == 0.25
    assert tool.load_json(negative_path)["d_seg"] == 0.25


@pytest.mark.parametrize(
    "negative",
    [
        {"raw_sha256_local_host": "base", "d_seg": 0.2, "d_pose": 0.3},
        {"raw_sha256_local_host": "changed", "d_seg": 0.1, "d_pose": 0.3},
        {"raw_sha256_local_host": "changed", "d_seg": 0.2, "d_pose": 0.2},
    ],
)
def test_negative_control_requires_raw_and_both_components_to_move(
    tool: ModuleType, negative: dict[str, object]
) -> None:
    baseline = {"raw_sha256_local_host": "base", "d_seg": 0.1, "d_pose": 0.2}
    with pytest.raises(RuntimeError, match="both scorer components"):
        tool.validate_negative_control(baseline, negative)
    tool.validate_negative_control(
        baseline,
        {"raw_sha256_local_host": "changed", "d_seg": 0.2, "d_pose": 0.3},
    )


def test_real_fixture_parse_repack_and_pair_scoped_code_boundary(tool: ModuleType) -> None:
    lbc = tool.load_byte_close_module()
    blob, _breakdown, _context = tool.prepare_baseline_blob(lbc)
    parts = tool.parse_blob(lbc, blob)
    assert tool.repack_blob(
        lbc, parts, base_raw=parts.base_raw, code_raw=parts.code_raw
    ) == blob
    sections = tool.coefficient_sections(
        parts.manifest,
        base_raw_len=len(parts.base_raw),
        code_raw_len=len(parts.code_raw),
        eval_pairs=1,
    )
    code = sections[-1]
    assert code.name == "code_scored_pair_prefix"
    assert code.shape == (2, 32)
    assert code.count == 64
    assert len(parts.code_raw) - code.count == 1_198 * 32


def test_checkpoint_identity_mismatch_is_rejected(tool: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    tool.atomic_write_json(
        path,
        {
            "status": "complete",
            "run_fingerprint": "c" * 64,
            "section": "a",
            "family": "uniform",
            "bits_removed": 1,
        },
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        tool.load_checked_checkpoint(
            path,
            run_fingerprint="c" * 64,
            expected_fields={"bits_removed": 2},
        )
