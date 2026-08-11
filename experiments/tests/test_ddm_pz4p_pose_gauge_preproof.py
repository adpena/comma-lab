from __future__ import annotations

import ast
import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "experiments/ddm_pz4p_pose_gauge_preproof.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("ddm_pz4p_under_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pz = load_runner()


@pytest.mark.parametrize("depth", [3, 9, 15])
def test_bit_pack_round_trip(depth: int) -> None:
    rng = np.random.default_rng(1234 + depth)
    values = rng.integers(0, 1 << depth, size=777, dtype=np.uint32)
    payload = pz.bit_pack_unsigned(values, depth)
    decoded = pz.bit_unpack_unsigned(payload, len(values), depth)
    assert np.array_equal(decoded, values)
    assert len(payload) == (len(values) * depth + 7) // 8


@pytest.mark.parametrize("cell_mode", list(pz.CELL_MODES))
def test_gauge_wire_round_trip(cell_mode: str) -> None:
    config = pz.GaugeConfig(rank=3, depth=9, cell_mode=cell_mode)
    _, per_rank, _, _, groups = pz.mode_geometry(config)
    rng = np.random.default_rng(7)
    qmax = (1 << (config.depth - 1)) - 1
    codes = rng.integers(-qmax, qmax + 1, size=(pz.N, config.rank), dtype=np.int32)
    scales = rng.uniform(1e-4, 2e-3, size=(groups, config.rank if per_rank else 1))
    compensation = rng.normal(size=(config.rank + 1, pz.POSE_DIMS))
    payload = pz.encode_gauge(config, codes, scales, compensation)
    decoded_config, outputs, parts = pz.decode_gauge(payload)
    assert decoded_config == config
    assert outputs.shape == (pz.N, pz.POSE_DIMS)
    assert outputs.dtype == np.float32
    assert np.array_equal(parts["codes"], codes.astype(np.int16))
    assert (
        pz.encode_gauge(
            decoded_config,
            parts["codes"].astype(np.int32),
            parts["scales"].astype(np.float64),
            parts["compensation"].astype(np.float64),
        )
        == payload
    )


def test_hard_qat_retains_best_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pz, "N", 24)
    monkeypatch.setattr(
        pz,
        "CELL_MODES",
        {
            "global": (24, False, 0),
            "per_rank": (24, True, 1),
            "block100_rank": (12, True, 2),
            "block50_rank": (8, True, 3),
            "block25_rank": (6, True, 4),
        },
    )
    rng = np.random.default_rng(99)
    reference = rng.normal(size=(24, pz.POSE_DIMS)).astype(np.float32)
    centered = reference.astype(np.float64) - reference.mean(axis=0)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    config = pz.GaugeConfig(rank=6, depth=9, cell_mode="block25_rank")
    coefficients = u[:, :6] * singular[:6]
    scales = pz.initial_scales(coefficients, config)
    initial_codes = pz.nearest_codes(coefficients, scales, config)
    _, _, initial_mse = pz.compensate(initial_codes, scales, config, reference)
    codes, learned_scales, _, _, learned_mse, history = pz.optimize_rounding(
        coefficients, scales, config, reference, tmp_path
    )
    assert learned_mse <= initial_mse
    assert codes.shape == coefficients.shape
    assert learned_scales.shape == scales.shape
    assert len(history) == pz.QAT_ROUNDS
    for row in history:
        pz.validate_record(row["checkpoint"])


def test_pareto_frontier_uses_exact_two_axis_dominance() -> None:
    rows = [
        {"candidate_id": "a", "rate_envelope_bytes": 100, "decoded_output_mse": 5.0},
        {"candidate_id": "b", "rate_envelope_bytes": 110, "decoded_output_mse": 4.0},
        {"candidate_id": "c", "rate_envelope_bytes": 120, "decoded_output_mse": 4.5},
        {"candidate_id": "d", "rate_envelope_bytes": 130, "decoded_output_mse": 2.0},
    ]
    assert [row["candidate_id"] for row in pz.pareto_frontier(rows)] == ["a", "b", "d"]


def test_runner_has_no_scorer_import_or_execution_surface() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    imported = set()
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    assert not any("posenet" in name.lower() or "segnet" in name.lower() for name in imported)
    assert "evaluate" not in called
    assert "load_scorers" not in called


def test_grid_is_full_depth_cell_rank_product() -> None:
    configs = pz.candidate_configs()
    assert len(configs) == len(pz.DEPTHS) * len(pz.CELL_MODES) * len(pz.RANKS)
    assert len({config.candidate_id for config in configs}) == len(configs)
    assert max(config.depth for config in configs) < 16


@pytest.mark.parametrize(
    ("rank", "depth", "cell_mode"),
    [(0, 9, "global"), (7, 9, "global"), (3, 0, "global"), (3, 16, "global"), (3, 9, "bogus")],
)
def test_gauge_config_rejects_noncanonical_geometry(rank: int, depth: int, cell_mode: str) -> None:
    with pytest.raises(pz.PreproofError):
        pz.GaugeConfig(rank=rank, depth=depth, cell_mode=cell_mode)


def test_decode_rejects_nonfinite_scale() -> None:
    config = pz.GaugeConfig(rank=1, depth=3, cell_mode="global")
    payload = bytearray(
        pz.encode_gauge(
            config,
            np.zeros((pz.N, 1), dtype=np.int32),
            np.ones((1, 1), dtype=np.float64),
            np.ones((2, pz.POSE_DIMS), dtype=np.float64),
        )
    )
    header_bytes = struct.calcsize("<4sBBBBHHHH")
    payload[header_bytes : header_bytes + 4] = struct.pack("<f", float("nan"))
    with pytest.raises(pz.PreproofError, match="invalid"):
        pz.decode_gauge(bytes(payload))


def test_retained_record_must_remain_inside_candidate_root(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    escaped = tmp_path / "escaped.bin"
    escaped.write_bytes(b"retained")
    with pytest.raises(pz.PreproofError, match="escaped"):
        pz.validate_record(pz.file_record(escaped), allowed_root=candidate_root)
