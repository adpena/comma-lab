from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import tools.realize_ddm_m7_relaxed_receiver as realization_module
from tac.canonical_frontier_pointer import POINTER_SCHEMA_VERSION
from tac.witness_dsl.dynamic_frontier_target import load_dynamic_frontier_target
from tools.realize_ddm_m7_relaxed_receiver import (
    CHECKPOINT_SCHEMA,
    EVIDENCE_AXIS,
    EXPECTED_CONSTANTS,
    ROUTING_CANDIDATE,
    ROUTING_NOT_CANDIDATE,
    ArchiveCustody,
    BatchMeasurement,
    RealizationRefusal,
    RealizeConfig,
    aggregate_pair_rows,
    build_final_receipt,
    load_contiguous_checkpoints,
    measure_batch_stream,
    parse_args,
    require_roundtrip_identity,
    reroute_final_receipt_against_current_frontier,
    validate_archive_contract,
    validate_config_paths,
    verify_final_receipt,
    verify_historical_final_receipt,
    write_batch_checkpoint,
)


def _config_mapping(tmp_path: Path) -> dict[str, object]:
    return {
        **EXPECTED_CONSTANTS,
        "candidate_archive": str((tmp_path / "candidate.zip").resolve()),
        "runtime_dir": str((tmp_path / "runtime").resolve()),
        "upstream_dir": str((tmp_path / "upstream").resolve()),
        "ssd_output_dir": str((tmp_path / "output").resolve()),
    }


def _config(tmp_path: Path) -> RealizeConfig:
    return RealizeConfig.from_mapping(_config_mapping(tmp_path))


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


def _materialize_required_paths(tmp_path: Path) -> None:
    (tmp_path / "candidate.zip").write_bytes(b"candidate")
    runtime = tmp_path / "runtime"
    upstream = tmp_path / "upstream"
    for relative in (
        "inflate.py",
        "inflate.sh",
        "src/codec.py",
        "src/codec_ctx.py",
        "src/codec_sidecar.py",
        "src/fec10_hybrid_decoder.py",
        "src/frame_selector.py",
        "src/model.py",
    ):
        path = runtime / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    for relative in (
        "evaluate.py",
        "frame_utils.py",
        "modules.py",
        "models/posenet.safetensors",
        "models/segnet.safetensors",
        "public_test_video_names.txt",
    ):
        path = upstream / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")


def _zip_payload(*, member_name: str = "x", stored: bool = True) -> bytes:
    source = b"CTXR" + bytes([1]) + b"\x00" * 9
    member = (
        b"FP11"
        + len(source).to_bytes(4, "little")
        + source
        + (0).to_bytes(2, "little")
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            member_name,
            member,
            compress_type=(
                zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
            ),
        )
    return output.getvalue()


def _synthetic_archive_config(
    tmp_path: Path,
    payload: bytes,
    *,
    member_name: str = "x",
) -> RealizeConfig:
    path = tmp_path / "candidate.zip"
    path.write_bytes(payload)
    return dataclasses.replace(
        _config(tmp_path),
        candidate_archive=str(path),
        expected_archive_bytes=len(payload),
        expected_archive_sha256=hashlib.sha256(payload).hexdigest(),
        expected_member_name=member_name,
    )


def _archive_custody() -> ArchiveCustody:
    return ArchiveCustody(
        path="/candidate.zip",
        archive_bytes=177169,
        archive_sha256=EXPECTED_CONSTANTS["expected_archive_sha256"],
        member_name="x",
        member_bytes=10,
        member_sha256="a" * 64,
        compression="ZIP_STORED",
        framing="FP11->CTXR->latent+sidecar->FECa_selector(+optional_DQS1)",
    )


def _identity_bundle() -> dict[str, object]:
    identity = {"schema": "test.identity.v1", "archive": "abc"}
    digest = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {"identity": identity, "identity_sha256": digest}


def _measurement(start: int, end: int, *, d_seg: float = 0.1) -> BatchMeasurement:
    return BatchMeasurement(
        pair_start=start,
        pair_end_exclusive=end,
        pair_rows=tuple(
            {"pair_id": pair_id, "d_seg": d_seg, "d_pose": 0.2}
            for pair_id in range(start, end)
        ),
        candidate_frames_sha256="1" * 64,
        gt_frames_sha256="2" * 64,
        candidate_shape=(end - start, 2, 1, 1, 3),
        gt_shape=(end - start, 2, 1, 1, 3),
    )


def _checkpoint_summaries(n_pairs: int = 600, batch_pairs: int = 16) -> list[dict[str, object]]:
    summaries = []
    for start in range(0, n_pairs, batch_pairs):
        end = min(start + batch_pairs, n_pairs)
        summaries.append(
            {
                "path": f"/ssd/batch_{start:04d}_{end - 1:04d}.json",
                "pair_start": start,
                "pair_end_exclusive": end,
                "checkpoint_sha256": f"{start // batch_pairs:064x}",
            }
        )
    return summaries


def _pair_rows(d_seg: float, d_pose: float) -> list[dict[str, float | int]]:
    return [
        {"pair_id": pair_id, "d_seg": d_seg, "d_pose": d_pose}
        for pair_id in range(600)
    ]


def _roundtrip() -> dict[str, object]:
    return {
        "latent_raw_roundtrip_byte_exact": True,
        "member_byte_exact": True,
        "archive_byte_exact": True,
        "archive_bytes": 177169,
        "archive_sha256": EXPECTED_CONSTANTS["expected_archive_sha256"],
    }


def test_typed_config_accepts_only_exact_schema_and_constants(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.schema == EXPECTED_CONSTANTS["schema"]
    assert config.canonical_sha256()

    unknown = _config_mapping(tmp_path)
    unknown["invented"] = True
    with pytest.raises(RealizationRefusal, match="unknown config keys"):
        RealizeConfig.from_mapping(unknown)

    wrong = _config_mapping(tmp_path)
    wrong["batch_pairs"] = 8
    with pytest.raises(RealizationRefusal, match="sealed value"):
        RealizeConfig.from_mapping(wrong)

    authority = _config_mapping(tmp_path)
    authority["score_claim"] = True
    with pytest.raises(RealizationRefusal, match="sealed value"):
        RealizeConfig.from_mapping(authority)

    stale_target = _config_mapping(tmp_path)
    stale_target["pointer_score"] = 0.001
    stale_target["fork_threshold"] = 0.001
    with pytest.raises(RealizationRefusal, match="unknown config keys"):
        RealizeConfig.from_mapping(stale_target)


def test_cli_exposes_only_required_config() -> None:
    assert parse_args(["--config", "config.json"]).config == Path("config.json")
    with pytest.raises(SystemExit):
        parse_args(["--config", "config.json", "--n-pairs", "1"])


def test_config_paths_refuse_local_output_and_missing_runtime(tmp_path: Path) -> None:
    _materialize_required_paths(tmp_path)
    config = _config(tmp_path)
    with pytest.raises(RealizationRefusal, match="allowed SSD tier"):
        validate_config_paths(config)

    (tmp_path / "runtime" / "src" / "model.py").unlink()
    with pytest.raises(RealizationRefusal, match="missing bound source/runtime"):
        validate_config_paths(config)


def test_zip_member_hash_and_compression_refusals(tmp_path: Path) -> None:
    valid_payload = _zip_payload()
    custody = validate_archive_contract(
        _synthetic_archive_config(tmp_path, valid_payload)
    )
    assert custody.member_name == "x"
    assert custody.compression == "ZIP_STORED"

    wrong_member = _zip_payload(member_name="not-x")
    with pytest.raises(RealizationRefusal, match="member"):
        validate_archive_contract(
            _synthetic_archive_config(
                tmp_path,
                wrong_member,
                member_name="x",
            )
        )

    compressed = _zip_payload(stored=False)
    with pytest.raises(RealizationRefusal, match="ZIP_STORED"):
        validate_archive_contract(
            _synthetic_archive_config(tmp_path, compressed)
        )

    config = _synthetic_archive_config(tmp_path, valid_payload)
    Path(config.candidate_archive).write_bytes(valid_payload + b"changed")
    with pytest.raises(RealizationRefusal, match=r"byte mismatch|SHA-256 mismatch"):
        validate_archive_contract(config)


def test_roundtrip_gate_requires_every_identity_leg() -> None:
    require_roundtrip_identity(_roundtrip(), archive_custody=_archive_custody())
    bad = _roundtrip()
    bad["member_byte_exact"] = False
    with pytest.raises(RealizationRefusal, match="member_byte_exact"):
        require_roundtrip_identity(bad, archive_custody=_archive_custody())


def test_checkpoint_is_content_hashed_and_immutable(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    identity = _identity_bundle()
    first = _measurement(0, 2)
    path = write_batch_checkpoint(
        checkpoint_dir=checkpoint_dir,
        identity_bundle=identity,
        measurement=first,
    )
    payload = json.loads(path.read_text())
    assert payload["schema"] == CHECKPOINT_SCHEMA
    assert len(payload["checkpoint_sha256"]) == 64
    write_batch_checkpoint(
        checkpoint_dir=checkpoint_dir,
        identity_bundle=identity,
        measurement=first,
    )
    with pytest.raises(RealizationRefusal, match="immutable JSON"):
        write_batch_checkpoint(
            checkpoint_dir=checkpoint_dir,
            identity_bundle=identity,
            measurement=_measurement(0, 2, d_seg=0.11),
        )


def test_resume_accepts_only_one_revalidated_contiguous_prefix(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    identity = _identity_bundle()
    for start in (0, 2):
        write_batch_checkpoint(
            checkpoint_dir=checkpoint_dir,
            identity_bundle=identity,
            measurement=_measurement(start, start + 2),
        )
    rows, summaries = load_contiguous_checkpoints(
        checkpoint_dir=checkpoint_dir,
        identity_bundle=identity,
        n_pairs=6,
        batch_pairs=2,
    )
    assert [row["pair_id"] for row in rows] == [0, 1, 2, 3]
    assert len(summaries) == 2

    duplicate = checkpoint_dir / "duplicate.json"
    duplicate.write_bytes(next(checkpoint_dir.glob("batch_0000*.json")).read_bytes())
    with pytest.raises(RealizationRefusal, match="contiguous prefix"):
        load_contiguous_checkpoints(
            checkpoint_dir=checkpoint_dir,
            identity_bundle=identity,
            n_pairs=6,
            batch_pairs=2,
        )


def test_resume_refuses_gap_stale_identity_and_content_hash(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    identity = _identity_bundle()
    write_batch_checkpoint(
        checkpoint_dir=checkpoint_dir,
        identity_bundle=identity,
        measurement=_measurement(2, 4),
    )
    with pytest.raises(RealizationRefusal, match="contiguous prefix"):
        load_contiguous_checkpoints(
            checkpoint_dir=checkpoint_dir,
            identity_bundle=identity,
            n_pairs=4,
            batch_pairs=2,
        )

    only = next(checkpoint_dir.glob("*.json"))
    payload = json.loads(only.read_text())
    payload["identity_sha256"] = "f" * 64
    only.write_text(json.dumps(payload))
    with pytest.raises(RealizationRefusal, match="checkpoint_sha256 mismatch"):
        load_contiguous_checkpoints(
            checkpoint_dir=checkpoint_dir,
            identity_bundle=identity,
            n_pairs=4,
            batch_pairs=2,
        )

    stale_dir = tmp_path / "stale"
    stale_identity = _identity_bundle()
    stale_identity["identity"] = {"schema": "other.identity.v1"}
    stale_identity["identity_sha256"] = hashlib.sha256(
        json.dumps(
            stale_identity["identity"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    write_batch_checkpoint(
        checkpoint_dir=stale_dir,
        identity_bundle=stale_identity,
        measurement=_measurement(0, 2),
    )
    with pytest.raises(RealizationRefusal, match="stale checkpoint identity"):
        load_contiguous_checkpoints(
            checkpoint_dir=stale_dir,
            identity_bundle=identity,
            n_pairs=2,
            batch_pairs=2,
        )


def test_duplicate_and_missing_pair_rows_refuse() -> None:
    with pytest.raises(RealizationRefusal, match="missing"):
        aggregate_pair_rows(
            [
                {"pair_id": 0, "d_seg": 0.0, "d_pose": 0.0},
                {"pair_id": 0, "d_seg": 0.0, "d_pose": 0.0},
            ],
            n_pairs=2,
        )
    with pytest.raises(RealizationRefusal, match="missing"):
        aggregate_pair_rows(
            [{"pair_id": 0, "d_seg": 0.0, "d_pose": 0.0}],
            n_pairs=2,
        )


def test_synthetic_scorer_free_batch_path() -> None:
    batches = [
        np.zeros((2, 2, 1, 1, 3), dtype=np.uint8),
        np.ones((2, 2, 1, 1, 3), dtype=np.uint8),
    ]
    completed: list[BatchMeasurement] = []

    def render(pair_ids: list[int]) -> np.ndarray:
        return np.full((len(pair_ids) * 2, 1, 1, 3), 7, dtype=np.uint8)

    def score(
        _gt: np.ndarray,
        _candidate: np.ndarray,
        pair_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(pair_ids, dtype=np.float64) / 100,
            np.asarray(pair_ids, dtype=np.float64) / 200,
        )

    rows = measure_batch_stream(
        batches,
        n_pairs=4,
        batch_pairs=2,
        resume_rows=[],
        render_batch=render,
        score_batch=score,
        on_completed_batch=completed.append,
    )
    assert [row["pair_id"] for row in rows] == [0, 1, 2, 3]
    assert [batch.pair_start for batch in completed] == [0, 2]
    assert all(len(batch.candidate_frames_sha256) == 64 for batch in completed)


@pytest.mark.parametrize(
    ("d_seg", "d_pose", "expected_routing"),
    [
        (0.0, 0.0, ROUTING_CANDIDATE),
        (0.002, 0.0, ROUTING_NOT_CANDIDATE),
    ],
)
def test_final_routing_and_false_authority_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    d_seg: float,
    d_pose: float,
    expected_routing: str,
) -> None:
    frontier_repo = tmp_path / "frontier"
    snapshot, now = _dynamic_frontier_snapshot(frontier_repo)
    monkeypatch.setattr(realization_module, "_DYNAMIC_TARGET_REPO_ROOT", frontier_repo)
    receipt = build_final_receipt(
        config=_config(tmp_path),
        identity_bundle=_identity_bundle(),
        archive=_archive_custody(),
        roundtrip=_roundtrip(),
        pair_rows=_pair_rows(d_seg, d_pose),
        checkpoints=_checkpoint_summaries(),
        frontier_snapshot=snapshot,
        now_utc_iso=now,
    )
    assert receipt["routing_label"] == expected_routing
    assert receipt["evidence_axis"] == EVIDENCE_AXIS
    assert receipt["score_claim"] is False
    assert receipt["promotion_eligible"] is False
    assert receipt["ready_for_exact_eval_dispatch"] is False
    assert receipt["dynamic_frontier_target"]["target_score"] == snapshot.target_score
    assert receipt["dynamic_frontier_target"]["pointer_sha256"] == snapshot.pointer_sha256
    assert receipt["comparisons"]["dynamic_frontier_target_score"] == snapshot.target_score
    assert (
        receipt["counterfactual_to_realized_score_gap"]["rate"] == 0.0
    )
    assert "different high-byte exact-C1 object" in (
        receipt["arithmetic_counterfactual"]["caveat"]
    )
    verify_final_receipt(
        receipt, frontier_snapshot=snapshot, now_utc_iso=now
    )


def test_receipt_verifier_refuses_authority_escalation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontier_repo = tmp_path / "frontier"
    snapshot, now = _dynamic_frontier_snapshot(frontier_repo)
    monkeypatch.setattr(realization_module, "_DYNAMIC_TARGET_REPO_ROOT", frontier_repo)
    receipt = build_final_receipt(
        config=_config(tmp_path),
        identity_bundle=_identity_bundle(),
        archive=_archive_custody(),
        roundtrip=_roundtrip(),
        pair_rows=_pair_rows(0.0, 0.0),
        checkpoints=_checkpoint_summaries(),
        frontier_snapshot=snapshot,
        now_utc_iso=now,
    )
    receipt["promotion_eligible"] = True
    with pytest.raises(
        RealizationRefusal,
        match=r"receipt_sha256 mismatch|authority field",
    ):
        verify_final_receipt(
            receipt, frontier_snapshot=snapshot, now_utc_iso=now
        )


def test_final_routing_refuses_forged_stale_and_path_swapped_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontier_repo = tmp_path / "canonical"
    snapshot, now = _dynamic_frontier_snapshot(frontier_repo)
    monkeypatch.setattr(realization_module, "_DYNAMIC_TARGET_REPO_ROOT", frontier_repo)
    kwargs = {
        "config": _config(tmp_path),
        "identity_bundle": _identity_bundle(),
        "archive": _archive_custody(),
        "roundtrip": _roundtrip(),
        "pair_rows": _pair_rows(0.0, 0.0),
        "checkpoints": _checkpoint_summaries(),
        "now_utc_iso": now,
    }
    with pytest.raises(RealizationRefusal, match="changed after snapshot"):
        build_final_receipt(
            **kwargs,
            frontier_snapshot=dataclasses.replace(snapshot, target_score=0.001),
        )
    stale_time = (datetime.fromisoformat(now) - timedelta(hours=25)).isoformat()
    with pytest.raises(RealizationRefusal, match="24-hour"):
        build_final_receipt(
            **kwargs,
            frontier_snapshot=dataclasses.replace(
                snapshot, last_refreshed_utc=stale_time
            ),
        )
    swapped_repo = tmp_path / "swapped"
    swapped, _ = _dynamic_frontier_snapshot(swapped_repo, score=0.24)
    with pytest.raises(RealizationRefusal, match="noncanonical pointer path"):
        build_final_receipt(**kwargs, frontier_snapshot=swapped)


def test_historical_receipt_survives_pointer_move_but_new_routing_reopens_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontier_repo = tmp_path / "frontier"
    historical, now = _dynamic_frontier_snapshot(frontier_repo, score=0.25)
    monkeypatch.setattr(realization_module, "_DYNAMIC_TARGET_REPO_ROOT", frontier_repo)
    receipt = build_final_receipt(
        config=_config(tmp_path),
        identity_bundle=_identity_bundle(),
        archive=_archive_custody(),
        roundtrip=_roundtrip(),
        pair_rows=_pair_rows(0.0, 0.0),
        checkpoints=_checkpoint_summaries(),
        frontier_snapshot=historical,
        now_utc_iso=now,
    )
    assert receipt["routing_label"] == ROUTING_CANDIDATE

    current, current_now = _dynamic_frontier_snapshot(frontier_repo, score=0.10)
    verify_historical_final_receipt(receipt)
    with pytest.raises(RealizationRefusal, match="does not bind the current"):
        verify_final_receipt(
            receipt,
            frontier_snapshot=current,
            now_utc_iso=current_now,
        )

    audit = reroute_final_receipt_against_current_frontier(
        receipt,
        frontier_snapshot=current,
        now_utc_iso=current_now,
    )
    assert audit["historical_target_score"] == 0.25
    assert audit["current_target_score"] == 0.10
    assert audit["pointer_moved"] is True
    assert audit["current_routing_label"] == ROUTING_NOT_CANDIDATE
    assert audit["score_claim"] is False
