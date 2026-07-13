from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _probe():
    repo = Path(__file__).resolve().parents[3]
    path = repo / "tools/probe_jacobian_drift_certificate.py"
    spec = importlib.util.spec_from_file_location("probe_jacobian_drift_certificate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_canaries_distinguish_fixed_q_jacobian_drift() -> None:
    controls = _probe()._canaries()
    assert controls["status"] == "PASS"
    assert controls["affine_fixed_q_jacobian_drift_norm"] == 0.0
    assert "changing CE adjoint" in controls["affine_scope"]


def test_probe_cosine_is_fp64_accumulated_and_bounded() -> None:
    import torch

    probe = _probe()
    left = torch.full((100_000,), 0.1, dtype=torch.float32)
    assert -1.0 <= probe._cosine(left, left) <= 1.0


def test_atomic_receipt_write_is_byte_stable(tmp_path: Path) -> None:
    probe = _probe()
    path = tmp_path / "receipt.json"
    payload = {"schema": probe.SCHEMA, "status": "MEASURED", "rows": [2, 1]}
    probe._atomic_write_json(path, payload)
    first = path.read_bytes()
    probe._atomic_write_json(path, json.loads(path.read_text()))
    assert path.read_bytes() == first


def test_source_bundle_binds_exact_task454_receipt_and_probe_bytes() -> None:
    bundle = _probe()._immutable_bundle()
    assert bundle["source_receipt"]["sha256"] == _probe().SOURCE_RECEIPT_SHA256
    assert "tools/probe_jacobian_drift_certificate.py" in bundle["source_bytes"]
    assert bundle["authority"]["score_claim"] is False
