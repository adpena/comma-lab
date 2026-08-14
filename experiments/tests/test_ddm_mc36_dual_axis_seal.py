from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments import ddm_js1b_modal_cuda_argmax_field_materializer as js1b
from experiments import ddm_mc36_dual_axis_seal as seal_module
from experiments import ddm_qs1_modal_t4_dual_axis as dispatcher


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _arrange_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    store = tmp_path / "winner"
    source_inputs = store / "fire_order/fire_inputs"
    source_inputs.mkdir(parents=True)
    archive = source_inputs / "candidate_archive.zip"
    runtime = source_inputs / "candidate_runtime.zip"
    archive.write_bytes(b"archive")
    runtime.write_bytes(b"runtime")
    archive_record = _record(archive)
    runtime_record = _record(runtime)
    source_order = {
        "schema": "ddm_mc36_dual_axis_fire_order.v1",
        "variant": "successor_drop532_pair105",
        "candidate_archive": archive_record,
        "candidate_runtime": runtime_record,
        "remote_dispatched": False,
    }
    (store / "SEALED_FIRE_ORDER.json").write_text(json.dumps(source_order))
    advisory = {
        "all_gates_passed": True,
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE",
        "selection_mode": "all changed pairs over n600",
        "seg": {"net_flip_gain": 37, "base_flips": 34_970},
        "rate": {
            "delta_bytes": 17,
            "base_archive_bytes": 186_252,
        },
        "pose": {
            "delta_dpose": -1.4632967835484165e-10,
            "base_dpose_recomputed": 0.0001474653494795297,
        },
        "projected_delta_s": -2.006473916964026e-5,
        "score_claim": False,
        "promotion_eligible": False,
    }
    (store / "LOCAL_ADVISORY_RECOUNT.json").write_text(json.dumps(advisory))
    compiled = {
        "runtime_tree": {
            "root": str(tmp_path / "runtime_root"),
            "file_count": 1,
            "tree_sha256": "tree",
            "files": [
                {
                    "relative_path": "inflate.py",
                    "bytes": 1,
                    "sha256": "digest",
                }
            ],
        }
    }
    (store / "COMPILED_ARCHIVE.json").write_text(json.dumps(compiled))

    monkeypatch.setattr(seal_module, "STORE", store)
    monkeypatch.setattr(seal_module, "SOURCE_FIRE_ORDER", store / "fire_order")
    monkeypatch.setattr(
        seal_module, "SOURCE_ADVISORY", store / "LOCAL_ADVISORY_RECOUNT.json"
    )
    monkeypatch.setattr(
        seal_module, "SOURCE_COMPILED", store / "COMPILED_ARCHIVE.json"
    )
    monkeypatch.setattr(seal_module, "OUTPUT", store / "dispatcher_conformant_seal")
    monkeypatch.setattr(seal_module, "EXPECTED_CANDIDATE_SHA", archive_record["sha256"])
    monkeypatch.setattr(seal_module, "EXPECTED_CANDIDATE_BYTES", archive_record["bytes"])
    monkeypatch.setattr(seal_module, "EXPECTED_RUNTIME_SHA", runtime_record["sha256"])
    monkeypatch.setattr(seal_module, "EXPECTED_RUNTIME_BYTES", runtime_record["bytes"])
    return store


def test_seal_round_trips_through_real_dispatcher_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _arrange_source(tmp_path, monkeypatch)

    order = seal_module.seal()

    request_path = Path(order["request"]["path"])
    payloads, request = dispatcher.load_sealed_inputs(
        request_path,
        store / "dispatcher_conformant_seal/fire_inputs",
        order["request"]["sha256"],
    )
    assert set(payloads) == {
        "candidate_archive.zip",
        "candidate_runtime.zip",
        "POSE_SCREEN_RESULT.json",
    }
    assert request["schema"] == "ddm_qs1_t4_dual_axis_request.v1"
    assert order["dispatcher_validation_passed"] is True


def test_seal_preserves_worker_pose_unknown_placeholder_law(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _arrange_source(tmp_path, monkeypatch)

    seal_module.seal()

    request = json.loads(
        (store / "dispatcher_conformant_seal/SEALED_REQUEST.json").read_text()
    )
    screen = json.loads(
        (store / "dispatcher_conformant_seal/fire_inputs/POSE_SCREEN_RESULT.json").read_text()
    )
    assert request["local_pose_delta"] == 0.0
    assert request["pose_unmeasured"] is True
    assert screen["local_pose_delta"] == 0.0
    assert screen["pose_unmeasured"] is True
    assert screen["local_advisory"] == {
        "delta_archive_bytes": 17,
        "delta_dpose": -1.4632967835484165e-10,
        "net_seg_flip_gain": 37,
        "projected_delta_s": -2.006473916964026e-5,
    }
    assert request["score_claim"] is False
    assert request["promotion_eligible"] is False


def test_seal_refuses_advisory_triple_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _arrange_source(tmp_path, monkeypatch)
    advisory_path = store / "LOCAL_ADVISORY_RECOUNT.json"
    advisory = json.loads(advisory_path.read_text())
    advisory["seg"]["net_flip_gain"] = 36
    advisory_path.write_text(json.dumps(advisory))

    with pytest.raises(seal_module.MC36DualAxisSealError, match="triple differs"):
        seal_module.seal()


def test_canonical_json_payload_is_the_hashed_dispatch_input() -> None:
    payload = js1b.canonical_json_bytes({"z": 1, "a": 2})
    assert payload == b'{"a":2,"z":1}\n'
