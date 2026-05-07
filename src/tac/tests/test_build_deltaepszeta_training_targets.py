"""Tests for ``tools.build_deltaepszeta_training_targets``."""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest


def _load_tool_module():
    """Import the script as a module via importlib (it's not a package member)."""
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    tool_path = repo_root / "tools" / "build_deltaepszeta_training_targets.py"
    spec = importlib.util.spec_from_file_location(
        "build_deltaepszeta_training_targets", tool_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load tool module at {tool_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_shannon_json(tmp_path: pathlib.Path) -> pathlib.Path:
    """Build a minimal Shannon analysis JSON for testing the builder."""
    payload = {
        "started_at_utc": "2026-05-07T18:00:00Z",
        "substrate": "synthetic-test-substrate",
        "evidence_grade": "[empirical]",
        "score_claim": False,
        "brotli_quality": 11,
        "n_quant": 127,
        "n_tensors": 3,
        "ac_tensor_indices": [0, 1, 2],
        "summary": {
            "total_n_symbols": 1500,
            "total_shannon_floor_h0_bytes": 1000,
            "total_shannon_floor_h2_bytes": 500,
            "total_brotli_bytes": 1100,
            "total_ac_bytes_pr103_indices_only": 1050,
            "ac_regression_total_bytes": 0,
            "ac_regression_tensors": [],
            "brotli_over_h0_ratio": 1.1,
            "brotli_over_h2_ratio": 2.2,
        },
        "per_tensor": [
            {
                "idx": 0, "name": "tensor_a", "n_symbols": 800, "scale": 0.01,
                "in_pr103_ac_set": True,
                "H0_bits": 6.0, "H1_bits": 4.5, "H2_bits": 3.0,
                "shannon_floor_h0_bytes": 600,
                "shannon_floor_h2_bytes": 300,
                "brotli_bytes": 650, "ac_bytes": 620,
                "bits_per_symbol_brotli": 6.5, "bits_per_symbol_ac": 6.2,
                "ac_minus_brotli_bytes": -30,
            },
            {
                "idx": 1, "name": "tensor_b", "n_symbols": 500, "scale": 0.02,
                "in_pr103_ac_set": True,
                "H0_bits": 5.0, "H1_bits": 4.0, "H2_bits": 2.0,
                "shannon_floor_h0_bytes": 313,
                "shannon_floor_h2_bytes": 125,
                "brotli_bytes": 350, "ac_bytes": 330,
                "bits_per_symbol_brotli": 5.6, "bits_per_symbol_ac": 5.3,
                "ac_minus_brotli_bytes": -20,
            },
            {
                "idx": 2, "name": "tensor_c", "n_symbols": 200, "scale": 0.03,
                "in_pr103_ac_set": False,
                "H0_bits": 8.0, "H1_bits": 7.5, "H2_bits": 7.0,
                "shannon_floor_h0_bytes": 200,
                "shannon_floor_h2_bytes": 175,
                "brotli_bytes": 210, "ac_bytes": None,
                "bits_per_symbol_brotli": 8.4, "bits_per_symbol_ac": None,
                "ac_minus_brotli_bytes": None,
            },
        ],
    }
    json_path = tmp_path / "synthetic_shannon.json"
    json_path.write_text(json.dumps(payload))
    return json_path


def test_build_targets_basic_shape(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    src = _synthetic_shannon_json(tmp_path)
    targets = mod.build_targets(src)
    assert targets["score_claim"] is False
    assert targets["evidence_grade"] == "[empirical]"
    assert targets["n_tensors"] == 3
    assert len(targets["per_tensor"]) == 3
    # All loss weights sum to 1 within numerical precision.
    total_weight = sum(r["loss_weight_normalized"] for r in targets["per_tensor"])
    assert abs(total_weight - 1.0) < 1e-9


def test_targets_sorted_by_prize_descending(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    src = _synthetic_shannon_json(tmp_path)
    targets = mod.build_targets(src)
    prizes = [r["prize_bytes"] for r in targets["per_tensor"]]
    assert prizes == sorted(prizes, reverse=True)


def test_headroom_non_negative(tmp_path: pathlib.Path) -> None:
    """Per the H₂ ≤ H₀ identity, headroom must be non-negative for every tensor."""
    mod = _load_tool_module()
    src = _synthetic_shannon_json(tmp_path)
    targets = mod.build_targets(src)
    for r in targets["per_tensor"]:
        assert r["headroom_bits"] >= 0.0, (
            f"tensor {r['name']} has negative headroom_bits "
            f"({r['headroom_bits']}); H₂ > H₀ is impossible"
        )


def test_prize_matches_headroom_times_symbols(tmp_path: pathlib.Path) -> None:
    """Sanity: prize_bytes = headroom_bits × n_symbols / 8."""
    mod = _load_tool_module()
    src = _synthetic_shannon_json(tmp_path)
    targets = mod.build_targets(src)
    for r in targets["per_tensor"]:
        expected = r["headroom_bits"] * r["n_symbols"] / 8.0
        assert abs(r["prize_bytes"] - expected) < 1e-6


def test_resolve_shannon_json_picks_newest(tmp_path: pathlib.Path) -> None:
    """When given a glob, the resolver picks the lexicographically last match
    (which is the newest by UTC-stamp convention)."""
    mod = _load_tool_module()
    a = tmp_path / "lane_per_tensor_shannon_pr106_20260101T000000Z"
    b = tmp_path / "lane_per_tensor_shannon_pr106_20261231T000000Z"
    a.mkdir()
    b.mkdir()
    (a / "per_tensor_shannon.json").write_text("{}")
    (b / "per_tensor_shannon.json").write_text("{}")
    glob_pattern = str(tmp_path / "lane_per_tensor_shannon_pr106_*/per_tensor_shannon.json")
    resolved = mod._resolve_shannon_json(glob_pattern)
    assert resolved.parent == b


def test_render_markdown_contains_score_disclaimer(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    src = _synthetic_shannon_json(tmp_path)
    targets = mod.build_targets(src)
    md = mod.render_markdown(targets)
    assert "Score claims" in md or "score_claim" in md.lower()
    assert "[empirical]" in md
    # Markdown must NOT make a [contest-CUDA] claim.
    assert "[contest-CUDA]" not in md


def test_resolve_shannon_json_raises_on_no_match(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    pattern = str(tmp_path / "no_such_path*.json")
    with pytest.raises(FileNotFoundError):
        mod._resolve_shannon_json(pattern)


def test_pr106_real_shannon_json_loads_cleanly(tmp_path: pathlib.Path) -> None:
    """End-to-end on the real PR106 Shannon analysis (if present).

    Skipped cleanly when the artifact isn't present (e.g., fresh checkout).
    """
    import glob
    paths = sorted(glob.glob(
        "experiments/results/lane_per_tensor_shannon_pr106_*/per_tensor_shannon.json"
    ))
    if not paths:
        pytest.skip("no PR106 Shannon analysis present")
    mod = _load_tool_module()
    targets = mod.build_targets(pathlib.Path(paths[-1]))
    # On real PR106 substrate, H₂/H₀ ≈ 0.531 per Path B finding.
    ratio = targets["summary"]["ratio_h2_over_h0"]
    assert 0.40 <= ratio <= 0.65, f"PR106 H₂/H₀ ratio out of band: {ratio}"
    # Total prize must be positive (H₂ < H₀ aggregate).
    assert targets["summary"]["total_prize_bytes"] > 0
