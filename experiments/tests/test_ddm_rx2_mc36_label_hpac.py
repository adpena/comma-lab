from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "experiments/ddm_rx2_mc36_label_hpac.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("ddm_rx2_mc36_label_hpac", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conditional_entropy_uses_exact_count_denominator() -> None:
    runner = _load_runner()
    deterministic = np.array([[7, 0, 0, 0, 0], [0, 4, 0, 0, 0]], dtype=np.uint64)
    fair_binary = np.array([[5, 5, 0, 0, 0]], dtype=np.uint64)
    assert runner._conditional_entropy_bits(deterministic) == 0.0
    assert runner._conditional_entropy_bits(fair_binary) == 10.0


def test_count_tables_are_retained_and_byte_priced(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "WORK_ROOT", tmp_path / "work")
    counts = np.array([[4, 3, 2, 1, 0], [0, 1, 2, 3, 4]], dtype=np.uint64)
    row, artifacts = runner._serialize_count_table("fixture", counts)
    assert row["table_bytes"] > 0
    assert row["table_path"].endswith("fixture.u64le.br11")
    assert len(artifacts) == 3
    assert all(Path(artifact["path"]).is_file() for artifact in artifacts)
    raw = Path(artifacts[0]["path"]).read_bytes()
    assert np.frombuffer(raw, dtype="<u8").reshape(2, 5).tolist() == counts.tolist()


def test_stage0_authority_label_is_fail_closed() -> None:
    runner = _load_runner()
    assert "NON-KILL-AUTHORITY" in RUNNER_PATH.read_text(encoding="utf-8")
    assert runner.MC36_MODEL_BYTES + runner.MC36_TOKEN_BYTES == 186_073
