from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_js1_stage0_per_edge as js1
from experiments import ddm_js1b_cuda_argmax_field_materializer_worker as worker
from experiments import ddm_js1b_modal_cuda_argmax_field_materializer as dispatcher
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def _write_runtime(root: Path) -> None:
    (root / "runtime/__pycache__").mkdir(parents=True)
    (root / "cpr1").mkdir()
    (root / "inflate.sh").write_text("#!/bin/sh\n")
    (root / "inflate.py").write_text("pass\n")
    (root / "runtime/f26_inflate.py").write_text("pass\n")
    (root / "cpr1/inflate.py").write_text("pass\n")
    (root / "runtime/__pycache__/discard.pyc").write_bytes(b"generated")
    (root / "archive.zip").write_bytes(b"separate exact input")


def _write_field(path: Path, value: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    return js1.file_record(path)


def test_k2_arithmetic_has_reserve_and_30_minute_headroom() -> None:
    result = dispatcher.k_arithmetic()
    assert result["k_archives"] == 2
    assert result["decode_seconds"] == 932.0
    assert result["scorer_seconds"] == pytest.approx(118.215)
    assert result["projected_seconds_with_reserve"] == pytest.approx(1350.215)
    assert result["headroom_seconds"] == pytest.approx(449.785)
    assert result["fits_30_minutes"] is True
    assert result["expected_retained_payload_bytes_before_metadata"] == 22_783_709_312
    assert worker.EXPECTED_RETAINED_PAYLOAD_BYTES == 22_783_709_312
    assert result["epistemic_status"].startswith("DERIVED_FROM_MEASURED")


def test_run_id_cannot_escape_retained_volume(tmp_path: Path) -> None:
    with pytest.raises(dispatcher.JS1BError, match="safe path component"):
        dispatcher.prepare_request(
            cp135_archive=tmp_path / "missing_cp135.zip",
            cp135_runtime=tmp_path / "missing_cp135_runtime",
            t1r1_archive=tmp_path / "missing_t1r1.zip",
            t1r1_runtime=tmp_path / "missing_t1r1_runtime",
            c1_target=tmp_path / "missing_c1.npy",
            run_id="../escape",
            resume_from="../escape",
        )


def test_runtime_bundle_is_deterministic_and_keeps_exact_receiver(tmp_path: Path) -> None:
    runtime = tmp_path / "adapted_runtime"
    _write_runtime(runtime)
    first, first_manifest = dispatcher.build_runtime_bundle(runtime, label="test")
    second, second_manifest = dispatcher.build_runtime_bundle(runtime, label="test")
    assert first == second
    assert first_manifest["bundle"] == second_manifest["bundle"]
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        names = set(archive.namelist())
    assert {"inflate.sh", "inflate.py", "runtime/f26_inflate.py", "cpr1/inflate.py"} <= names
    assert "archive.zip" not in names
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)


def test_worker_adjudication_requires_both_exact_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker, "EXPECTED_CP135_FLIPS", 2)
    monkeypatch.setattr(worker, "EXPECTED_C1_TARGET_FLIPS", 1)
    fields = tmp_path / "retained/fields"
    gt = np.zeros((1, 2, 2), dtype=np.uint8)
    cp135 = gt.copy()
    cp135[0, 0, :] = 1
    t1r1 = gt.copy()
    t1r1[0, 0, 0] = 1
    c1 = t1r1.copy()
    for name, value in {
        "gt": gt,
        "cp135_base": cp135,
        "t1r1_c1_composed": t1r1,
        "c1_target": c1,
    }.items():
        _write_field(fields / f"{name}_argmax_n600.npy", value)
    admitted = worker.adjudicate_fields(tmp_path)
    assert admitted["status"] == "ADMITTED"
    assert admitted["flips_vs_gt"]["t1r1_c1_composed"] == 1
    c1[0, 1, 1] = 1
    _write_field(fields / "c1_target_argmax_n600.npy", c1)
    blocked = worker.adjudicate_fields(tmp_path)
    assert blocked["status"] == "BLOCKED_AXIS_MISMATCH"
    assert "stop" in blocked["disposition"]


def test_from_argmax_fields_accepts_only_bound_admitted_cuda_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(js1, "N", 1)
    monkeypatch.setattr(js1, "SEG_H", 2)
    monkeypatch.setattr(js1, "SEG_W", 2)
    monkeypatch.setattr(js1, "TERMINAL_BASE_FLIPS", 0)
    monkeypatch.setattr(js1, "C1_BATCH16_REFERENCE_FLIPS", 0)
    records = {}
    for name in ("gt", "cp135_base", "t1r1_c1_composed", "c1_target"):
        records[name] = _write_field(
            tmp_path / f"retained/fields/{name}_argmax_n600.npy",
            np.zeros((1, 2, 2), dtype=np.uint8),
        )
    receipt = {
        "schema": "ddm_js1b_cuda_argmax_field_materializer_result.v1",
        "execution_status": "COMPLETE",
        "status": "ADMITTED",
        "axis": (
            "[contest-CUDA T4 frozen-SegNet argmax fields, n600, batch=16] COMPONENT-ONLY"
        ),
        "batch_size": 16,
        "score_claim": False,
        "promotion_eligible": False,
        "axis_adjudication": {
            "admitted_for_js1_stage0": True,
            "cp135_control": {"observed_flips": js1.TERMINAL_BASE_FLIPS},
            "c1_target_control": {"observed_flips": js1.C1_BATCH16_REFERENCE_FLIPS},
            "flips_vs_gt": {
                "cp135_base": 0,
                "t1r1_c1_composed": 0,
                "c1_target": 0,
            },
        },
        "fields": records,
    }
    (tmp_path / "FINAL_RESULT.json").write_text(json.dumps(receipt))
    loaded = js1.load_cuda_argmax_bundle(tmp_path)
    assert set(loaded["fields"]) == {"gt", "cp135_base", "t1r1_c1_composed", "c1_target"}
    assert loaded["receipt"]["status"] == "ADMITTED"


def test_from_argmax_fields_stops_on_control_mismatch(tmp_path: Path) -> None:
    receipt = {
        "schema": "ddm_js1b_cuda_argmax_field_materializer_result.v1",
        "execution_status": "COMPLETE",
        "status": "BLOCKED_AXIS_MISMATCH",
        "axis": (
            "[contest-CUDA T4 frozen-SegNet argmax fields, n600, batch=16] COMPONENT-ONLY"
        ),
        "batch_size": 16,
        "score_claim": False,
        "promotion_eligible": False,
        "axis_adjudication": {
            "admitted_for_js1_stage0": False,
            "cp135_control": {"observed_flips": js1.TERMINAL_BASE_FLIPS + 1},
            "c1_target_control": {"observed_flips": js1.C1_BATCH16_REFERENCE_FLIPS},
        },
    }
    (tmp_path / "FINAL_RESULT.json").write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match=r"stop.*field/custody"):
        js1.load_cuda_argmax_bundle(tmp_path)


def test_dispatch_and_worker_sources_pin_governance_receiver_and_retention() -> None:
    dispatch_source = Path(dispatcher.__file__).read_text()
    worker_source = Path(worker.__file__).read_text()
    assert "assert_modal_single_flight" in dispatch_source
    assert "register_dispatched_call_id_fail_closed" in dispatch_source
    assert "retained_volume.commit()" in dispatch_source
    assert "--resume-from" in dispatch_source
    assert 'runtime_root / "inflate.sh"' in worker_source
    assert "DaliVideoDataset" in worker_source
    assert "TensorVideoDataset" in worker_source
    assert "seg_input.float32.npy" in worker_source
    assert "logits.float32.npy" in worker_source
    assert "stage_30_final.json" in worker_source


def test_js1b_files_pass_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=Path.cwd(),
        strict=False,
        roots=(
            "experiments/ddm_js1b_modal_cuda_argmax_field_materializer.py",
            "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py",
            "experiments/ddm_js1_stage0_per_edge.py",
            "experiments/tests/test_ddm_js1b_cuda_argmax_field_materializer.py",
        ),
    )
    assert findings == []
