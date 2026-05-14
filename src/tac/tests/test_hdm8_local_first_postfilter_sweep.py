# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "run_hdm8_local_first_postfilter_sweep.py"
SCREEN_TOOL_PATH = REPO_ROOT / "tools" / "screen_hdm8_postfilter_sweep.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("hdm8_local_first_sweep_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["hdm8_local_first_sweep_test"] = module
    spec.loader.exec_module(module)
    return module


def _load_screen_module():
    spec = importlib.util.spec_from_file_location("hdm8_screen_sweep_test", SCREEN_TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["hdm8_screen_sweep_test"] = module
    spec.loader.exec_module(module)
    return module


def test_palette_is_first_frame_safe_by_default() -> None:
    mod = _load_module()

    modes = mod.build_mode_palette(profile="fast")

    assert modes[0] == "none"
    assert len(modes) == len(set(modes))
    assert any(mode.startswith("even_rgb_bias:") for mode in modes)
    assert any(mode.startswith("even_rgb_scale:") for mode in modes)
    assert any(mode.startswith("even_contrast:") for mode in modes)
    assert any(mode.startswith("even_gamma:") for mode in modes)
    assert any(mode.startswith("even_grain_chroma:") for mode in modes)
    assert any(mode.startswith("even_tile_chroma:") for mode in modes)
    assert not any(mode.startswith("odd_") for mode in modes)


def test_palette_can_opt_into_last_frame_risk() -> None:
    mod = _load_module()

    modes = mod.build_mode_palette(profile="fast", include_last_frame_risky=True)

    assert any(mode.startswith("odd_bias:") for mode in modes)


def test_select_modes_carries_none_top_k_and_margin() -> None:
    mod = _load_module()
    payload = {
        "modes": [
            {"mode": "none", "score_proxy": 0.20, "delta_vs_none": 0.0},
            {"mode": "bad", "score_proxy": 0.25, "delta_vs_none": 0.05},
            {"mode": "best", "score_proxy": 0.19, "delta_vs_none": -0.01},
            {"mode": "near", "score_proxy": 0.2002, "delta_vs_none": 0.0002},
        ]
    }

    modes = mod.select_modes_for_next_stage(
        payload,
        top_k=1,
        margin=0.00025,
        required_modes=["manual"],
    )

    assert modes == ["none", "manual", "best", "near"]


def test_cpu_guard_subset_is_representative_and_bounded() -> None:
    mod = _load_module()
    modes = mod.build_mode_palette(profile="broad")

    subset = mod.select_cpu_guard_modes(modes, max_modes=32)

    assert subset[0] == "none"
    assert len(subset) <= 32
    assert len(subset) == len(set(subset))
    assert "even_bias:1" in subset
    assert "even_rgb_bias:1,-0.5,-0.5" in subset
    assert "even_rgb_scale:1.02,0.99,0.99" in subset
    assert "even_contrast:0.06" in subset
    assert "even_grain_chroma:1" in subset
    assert "even_tile_chroma:3" in subset
    assert any(mode.startswith("even_translate:") for mode in subset)


def test_split_modes_for_shards_keeps_none_in_every_shard() -> None:
    mod = _load_module()

    shards = mod.split_modes_for_shards(["none", "a", "b", "c", "d"], shard_size=3)

    assert shards == [["none", "a", "b"], ["none", "c", "d"]]


def test_merge_shard_payloads_recomputes_global_delta(tmp_path: Path) -> None:
    mod = _load_module()
    payload_a = {
        "schema": "hdm8_postfilter_proxy_sweep_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "axis": "local-mps-proxy-prefix",
        "archive": "archive.zip",
        "archive_bytes": 100,
        "n_pairs": 2,
        "modes": [
            {"mode": "none", "score_proxy": 0.20},
            {"mode": "a", "score_proxy": 0.18},
        ],
    }
    payload_b = {
        **payload_a,
        "modes": [
            {"mode": "none", "score_proxy": 0.20},
            {"mode": "b", "score_proxy": 0.19},
        ],
    }

    merged = mod.merge_shard_payloads(
        [payload_a, payload_b],
        output_json=tmp_path / "merged.json",
        started_at=0,
        shard_manifest=[],
        max_observed_rss_kb=1024,
    )

    rows = {row["mode"]: row for row in merged["modes"]}
    assert list(rows) == ["none", "a", "b"]
    assert rows["a"]["delta_vs_none"] == pytest.approx(-0.02)
    assert rows["b"]["delta_vs_none"] == pytest.approx(-0.01)
    assert merged["best"]["mode"] == "a"
    assert merged["rss_guard"]["max_observed_rss_kb"] == 1024


def test_selector_rate_estimate_counts_bits() -> None:
    mod = _load_module()

    estimate = mod._charged_selector_rate_estimate(17, 600)

    assert estimate["bits_per_index"] == 5
    assert estimate["raw_selector_bytes_lower_bound"] == 375
    assert estimate["raw_selector_rate_score_lower_bound"] > 0


def test_screen_identifies_first_frame_safe_modes_for_segnet_reuse() -> None:
    screen = _load_screen_module()

    assert screen._mode_preserves_second_frame("none")
    assert screen._mode_preserves_second_frame("even_bias:1")
    assert screen._mode_preserves_second_frame(
        "even_rgb_bias:1,-0.5,-0.5+even_grain_chroma:1"
    )
    assert not screen._mode_preserves_second_frame("bias:1")
    assert not screen._mode_preserves_second_frame("odd_bias:1")
