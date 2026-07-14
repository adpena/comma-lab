# SPDX-License-Identifier: MIT
from __future__ import annotations

import copy
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from numpy.lib import format as npy_format

REPO = Path(__file__).resolve().parents[3]
PROBE_PATH = REPO / "tools/probe_ripo_fisher_trust_region_saved.py"
SPEC = importlib.util.spec_from_file_location("probe_ripo_fisher_trust_region_saved", PROBE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _header_bytes(shape: tuple[int, ...], dtype: str) -> bytes:
    stream = io.BytesIO()
    npy_format.write_array_header_1_0(
        stream,
        {
            "descr": np.dtype(dtype).str,
            "fortran_order": False,
            "shape": shape,
        },
    )
    return stream.getvalue()


def _header_only_full_grid_npz(path: Path) -> None:
    fields = {
        "probabilities": (probe.EXPECTED_FIELD_SHAPE, "<f4"),
        "proposed_logit_step": (probe.EXPECTED_FIELD_SHAPE, "<f4"),
        "target_classes": (probe.EXPECTED_TARGET_SHAPE, "<i2"),
        "pair_ids": ((probe.EXPECTED_PAIRS,), "<i4"),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for key, (shape, dtype) in fields.items():
            archive.writestr(f"{key}.npy", _header_bytes(shape, dtype))


def _small_npz(path: Path) -> None:
    rng = np.random.default_rng(20260714)
    logits = rng.normal(size=(600, 2, 5)).astype(np.float32)
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    np.savez(
        path,
        probabilities=probabilities,
        proposed_logit_step=np.zeros_like(probabilities),
        target_classes=np.zeros((600, 2), dtype=np.int16),
        pair_ids=np.arange(600, dtype=np.int32),
    )


def _complete_custody(tmp_path: Path, input_npz: Path) -> dict[str, object]:
    artifacts: dict[str, Path] = {}
    for name in ("source", "checkpoint", "segnet", "r_operator", "producer_source"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(f"custodied-{name}".encode())
        artifacts[name] = path
    headers, blockers = probe._inspect_npz_headers(input_npz)
    assert not blockers
    hashes = {
        f"{name}_sha256": probe._sha256(artifacts[name])
        for name in ("source", "checkpoint", "segnet", "r_operator")
    }
    producer = {
        "schema": probe.PRODUCER_SCHEMA,
        "complete": True,
        "output": {
            "path": str(input_npz),
            "sha256": probe._sha256(input_npz),
            "container": "monolithic_npz",
            "arrays": headers,
        },
        "source_binding": {
            **hashes,
            "actual_r_applied": True,
            "frozen_cpu_torch_segnet": True,
        },
        "producer_source": {
            "path": str(artifacts["producer_source"]),
            "sha256": probe._sha256(artifacts["producer_source"]),
        },
    }
    producer_path = tmp_path / "producer_receipt.json"
    _write_json(producer_path, producer)
    return {
        "schema": probe.CUSTODY_SCHEMA,
        "complete": True,
        "n_pairs": 600,
        "pair_indices": list(range(600)),
        "actual_r_applied": True,
        "frozen_cpu_torch_segnet": True,
        "probability_source": "frozen_cpu_torch_segnet_after_actual_r",
        "saved_arrays_sha256": probe._sha256(input_npz),
        "source_path": str(artifacts["source"]),
        "checkpoint_path": str(artifacts["checkpoint"]),
        "segnet_path": str(artifacts["segnet"]),
        "r_operator_path": str(artifacts["r_operator"]),
        **hashes,
        "producer_receipt_path": str(producer_path),
        "producer_receipt_sha256": probe._sha256(producer_path),
    }


def _run(tmp_path: Path, **overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "input_npz": tmp_path / "saved.npz",
        "custody_json": tmp_path / "custody.json",
        "output_dir": tmp_path / "out",
        "delta": 0.002,
        "delta_convention": "delta_kl",
        "mode": "local_directional",
        "tolerance": 1e-10,
    }
    arguments.update(overrides)
    return probe.run_probe(**arguments)


def test_missing_arrays_and_custody_emit_atomic_no_verdict(tmp_path: Path) -> None:
    receipt = _run(tmp_path)
    assert receipt["status"] == probe.NO_VERDICT
    assert receipt["formulation_id"] == "categorical_fisher_output_space_trust_region_v1"
    assert receipt["materialization_attempted"] is False
    assert receipt["pair_sharded_streaming_implemented"] is False
    assert receipt["score_claim"] is False
    assert receipt["pointer_moved"] is False
    assert probe.BOUNDEDNESS_BLOCKER in receipt["blockers"]
    output = tmp_path / "out/receipt.json"
    assert json.loads(output.read_text()) == receipt
    assert not list((tmp_path / "out").glob("*.tmp"))
    assert not list((tmp_path / "out").glob(".scratch.*"))


def test_exact_full_grid_headers_are_required() -> None:
    valid = {
        "probabilities": {"shape": list(probe.EXPECTED_FIELD_SHAPE), "dtype_kind": "f"},
        "proposed_logit_step": {"shape": list(probe.EXPECTED_FIELD_SHAPE), "dtype_kind": "f"},
        "target_classes": {"shape": list(probe.EXPECTED_TARGET_SHAPE), "dtype_kind": "i"},
        "pair_ids": {"shape": [600], "dtype_kind": "i"},
    }
    assert probe._shape_blockers(valid) == []
    invalid = copy.deepcopy(valid)
    invalid["probabilities"]["shape"] = [600, 2, 5]
    invalid["target_classes"]["shape"] = [600, 2]
    blockers = probe._shape_blockers(invalid)
    assert any("probabilities must have exact shape" in blocker for blocker in blockers)
    assert any("target_classes must have exact shape" in blocker for blocker in blockers)


def test_monolithic_full_grid_is_blocked_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_npz = tmp_path / "saved.npz"
    _header_only_full_grid_npz(input_npz)
    _write_json(tmp_path / "custody.json", _complete_custody(tmp_path, input_npz))

    def forbidden_load(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"numpy.load must not be called: {args}, {kwargs}")

    monkeypatch.setattr(np, "load", forbidden_load)
    receipt = _run(tmp_path)
    assert receipt["status"] == probe.NO_VERDICT
    assert receipt["blockers"] == [probe.BOUNDEDNESS_BLOCKER]
    assert receipt["materialization_attempted"] is False
    assert receipt["inspected_npz_headers"]["probabilities"]["shape"] == list(
        probe.EXPECTED_FIELD_SHAPE
    )
    assert "exact_kl_vs_quadratic" not in receipt


def test_small_dummy_n600_prefix_cannot_masquerade_as_real_full_grid(tmp_path: Path) -> None:
    input_npz = tmp_path / "saved.npz"
    _small_npz(input_npz)
    _write_json(tmp_path / "custody.json", _complete_custody(tmp_path, input_npz))
    receipt = _run(tmp_path)
    assert receipt["status"] == probe.NO_VERDICT
    assert any("probabilities must have exact shape" in blocker for blocker in receipt["blockers"])
    assert any("target_classes must have exact shape" in blocker for blocker in receipt["blockers"])
    assert receipt["materialization_attempted"] is False


def test_producer_receipt_must_exactly_bind_output_and_source(tmp_path: Path) -> None:
    input_npz = tmp_path / "saved.npz"
    _small_npz(input_npz)
    custody = _complete_custody(tmp_path, input_npz)
    producer_path = Path(str(custody["producer_receipt_path"]))
    producer = json.loads(producer_path.read_text())
    producer["output"]["sha256"] = "0" * 64
    _write_json(producer_path, producer)
    custody["producer_receipt_sha256"] = probe._sha256(producer_path)
    _write_json(tmp_path / "custody.json", custody)
    receipt = _run(tmp_path)
    assert any("producer.output.sha256" in blocker for blocker in receipt["blockers"])


def test_start_end_custody_change_emits_toctou_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_npz = tmp_path / "saved.npz"
    _small_npz(input_npz)
    _write_json(tmp_path / "custody.json", _complete_custody(tmp_path, input_npz))
    original = probe._snapshot
    calls = 0

    def changing_snapshot(paths: object) -> dict[str, dict[str, object]]:
        nonlocal calls
        calls += 1
        value = original(paths)
        if calls == 3:
            mutated = copy.deepcopy(value)
            first = next(iter(mutated))
            mutated[first]["sha256"] = "f" * 64
            return mutated
        return value

    monkeypatch.setattr(probe, "_snapshot", changing_snapshot)
    receipt = _run(tmp_path)
    assert probe.TOCTOU_BLOCKER in receipt["blockers"]
    assert receipt["toctou_changed_paths"]
    assert receipt["start_custody"] != receipt["end_custody"]


def test_fixed_global_head_tier_remains_custody_blocked(tmp_path: Path) -> None:
    input_npz = tmp_path / "saved.npz"
    _small_npz(input_npz)
    _write_json(tmp_path / "custody.json", _complete_custody(tmp_path, input_npz))
    receipt = _run(tmp_path, tier="fixed_global_head")
    assert receipt["status"] == probe.NO_VERDICT
    assert any("fixed_global_head" in blocker for blocker in receipt["blockers"])


def test_resume_is_byte_stable_and_changed_input_refuses_reuse(tmp_path: Path) -> None:
    input_npz = tmp_path / "saved.npz"
    _small_npz(input_npz)
    _write_json(tmp_path / "custody.json", _complete_custody(tmp_path, input_npz))
    first = _run(tmp_path)
    receipt_path = tmp_path / "out/receipt.json"
    first_bytes = receipt_path.read_bytes()
    assert _run(tmp_path) == first
    assert receipt_path.read_bytes() == first_bytes
    with pytest.raises(probe.ProbeError, match="different progress fingerprint"):
        _run(tmp_path, delta=0.003)
