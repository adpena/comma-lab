# SPDX-License-Identifier: MIT
"""Tests for tools/xray_per_tensor_saliency_heatmap.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import xray_per_tensor_saliency_heatmap as xs  # noqa: E402


def test_load_saliency_equal_uses_byte_map_keys():
    bm = {"t1": 100, "t2": 200}
    sal = xs.load_saliency(None, equal=True, byte_map=bm)
    assert sal == {"t1": 1.0, "t2": 1.0}


def test_load_saliency_from_json(tmp_path):
    p = tmp_path / "sal.json"
    p.write_text(json.dumps({"t1": 0.5, "t2": 1.5}))
    sal = xs.load_saliency(p, equal=False, byte_map={"t1": 1})
    assert sal == {"t1": 0.5, "t2": 1.5}


def test_load_saliency_rejects_non_dict(tmp_path):
    p = tmp_path / "sal.json"
    p.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        xs.load_saliency(p, equal=False, byte_map={"t1": 1})


def test_load_saliency_drops_non_float_values(tmp_path):
    p = tmp_path / "sal.json"
    p.write_text(json.dumps({"t1": 0.5, "t2": "junk", "t3": 1.0}))
    sal = xs.load_saliency(p, equal=False, byte_map={"t1": 1})
    assert "t2" not in sal
    assert sal["t1"] == 0.5
    assert sal["t3"] == 1.0


def test_load_saliency_no_source_raises(tmp_path):
    with pytest.raises(ValueError):
        xs.load_saliency(None, equal=False, byte_map={"t1": 1})


def test_load_byte_map_from_json(tmp_path):
    p = tmp_path / "bm.json"
    p.write_text(json.dumps({"t1": 100, "t2": 200, "t3": 0}))
    bm = xs.load_byte_map(p, None)
    # zero-byte entries dropped
    assert bm == {"t1": 100, "t2": 200}


def test_load_byte_map_from_archive_profile_parser_sections(tmp_path):
    p = tmp_path / "abp.json"
    p.write_text(json.dumps({
        "parser_sections": [
            {"name": "t1", "compressed_bytes": 50},
            {"name": "t2", "compressed_bytes": 100},
        ]
    }))
    bm = xs.load_byte_map(None, p)
    assert bm == {"t1": 50, "t2": 100}


def test_load_byte_map_from_archive_profile_sections_alt_keys(tmp_path):
    p = tmp_path / "abp.json"
    p.write_text(json.dumps({
        "sections": [
            {"section_name": "x", "bytes": 25},
            {"tensor_name": "y", "payload_bytes": 75},
        ]
    }))
    bm = xs.load_byte_map(None, p)
    assert bm == {"x": 25, "y": 75}


def test_load_byte_map_no_source_raises():
    with pytest.raises(ValueError):
        xs.load_byte_map(None, None)


def test_build_heatmap_basic_ranking():
    saliency = {"a": 1.0, "b": 1.0, "c": 1.0}
    bytes_ = {"a": 100, "b": 50, "c": 200}
    h = xs.build_heatmap(saliency, bytes_, bottom_n_percent=33.0)
    # spb: a=0.01, b=0.02, c=0.005 → c first (lowest spb)
    assert h["rows"][0]["tensor_name"] == "c"
    assert h["rows"][-1]["tensor_name"] == "b"
    assert h["coarsen_target_count"] == 1


def test_build_heatmap_drops_tensors_not_in_both():
    saliency = {"a": 1.0, "b": 1.0, "extra": 1.0}
    bytes_ = {"a": 100, "b": 50, "alt": 25}
    h = xs.build_heatmap(saliency, bytes_, bottom_n_percent=50.0)
    assert h["total_tensors_compared"] == 2
    assert "extra" in h["tensors_only_in_saliency"]
    assert "alt" in h["tensors_only_in_byte_map"]


def test_build_heatmap_zero_byte_tensor_yields_zero_spb():
    saliency = {"a": 1.0}
    bytes_ = {"a": 0}
    # load_byte_map() filters zero bytes, but build_heatmap should not crash
    # if called directly with a zero entry.
    h = xs.build_heatmap(saliency, bytes_, bottom_n_percent=50.0)
    assert h["rows"][0]["saliency_per_byte"] == 0.0


def test_build_heatmap_invalid_bottom_n_raises():
    with pytest.raises(ValueError):
        xs.build_heatmap({"a": 1.0}, {"a": 100}, bottom_n_percent=-1)
    with pytest.raises(ValueError):
        xs.build_heatmap({"a": 1.0}, {"a": 100}, bottom_n_percent=101)


def test_build_heatmap_assigns_rank_index():
    saliency = {f"t{i}": 1.0 for i in range(5)}
    bytes_ = {f"t{i}": 10 * (i + 1) for i in range(5)}
    h = xs.build_heatmap(saliency, bytes_, bottom_n_percent=20.0)
    ranks = [r["rank"] for r in h["rows"]]
    assert ranks == [0, 1, 2, 3, 4]


def test_build_heatmap_coarsen_priority_flag():
    saliency = {f"t{i}": 1.0 for i in range(10)}
    bytes_ = {f"t{i}": 100 - i * 5 for i in range(10)}
    h = xs.build_heatmap(saliency, bytes_, bottom_n_percent=30.0)
    n_priority = sum(1 for r in h["rows"] if r["coarsen_priority"])
    assert n_priority == 3
    # First 3 rows must be flagged
    assert all(h["rows"][i]["coarsen_priority"] for i in range(3))
    assert not h["rows"][3]["coarsen_priority"]


def test_build_heatmap_aggregates_total_bytes():
    saliency = {"a": 1.0, "b": 1.0}
    bytes_ = {"a": 100, "b": 200}
    h = xs.build_heatmap(saliency, bytes_, bottom_n_percent=50.0)
    assert h["total_bytes"] == 300


def test_main_writes_outputs(tmp_path):
    sal = tmp_path / "sal.json"
    sal.write_text(json.dumps({"a": 0.5, "b": 1.5}))
    bm = tmp_path / "bm.json"
    bm.write_text(json.dumps({"a": 100, "b": 200}))
    out = tmp_path / "out"
    rc = xs.main([
        "--saliency-json", str(sal),
        "--byte-map-json", str(bm),
        "--output-dir", str(out),
        "--label", "test_archive",
    ])
    assert rc == 0
    assert (out / "saliency_heatmap.json").exists()
    assert (out / "saliency_heatmap.md").exists()
    rep = json.loads((out / "saliency_heatmap.json").read_text())
    assert rep["score_claim"] is False
    assert rep["label"] == "test_archive"


def test_main_equal_saliency_path(tmp_path):
    bm = tmp_path / "bm.json"
    bm.write_text(json.dumps({"a": 100, "b": 200}))
    out = tmp_path / "out"
    rc = xs.main([
        "--saliency-equal",
        "--byte-map-json", str(bm),
        "--output-dir", str(out),
    ])
    assert rc == 0
    rep = json.loads((out / "saliency_heatmap.json").read_text())
    assert rep["saliency_source"] == "uniform_equal"


def test_main_archive_profile_path(tmp_path):
    abp = tmp_path / "abp.json"
    abp.write_text(json.dumps({
        "parser_sections": [
            {"name": "x", "compressed_bytes": 50},
            {"name": "y", "compressed_bytes": 25},
        ]
    }))
    sal = tmp_path / "sal.json"
    sal.write_text(json.dumps({"x": 1.0, "y": 1.0}))
    out = tmp_path / "out"
    rc = xs.main([
        "--saliency-json", str(sal),
        "--archive-byte-profile", str(abp),
        "--output-dir", str(out),
    ])
    assert rc == 0


def test_main_zero_overlap_returns_2(tmp_path):
    sal = tmp_path / "sal.json"
    sal.write_text(json.dumps({"a": 1.0}))
    bm = tmp_path / "bm.json"
    bm.write_text(json.dumps({"b": 100}))
    rc = xs.main([
        "--saliency-json", str(sal),
        "--byte-map-json", str(bm),
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 2


def test_main_missing_saliency_returns_2(tmp_path):
    bm = tmp_path / "bm.json"
    bm.write_text(json.dumps({"a": 100}))
    rc = xs.main([
        "--byte-map-json", str(bm),
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 2


def test_markdown_includes_diagnostic_tag(tmp_path):
    sal = tmp_path / "sal.json"
    sal.write_text(json.dumps({"a": 1.0}))
    bm = tmp_path / "bm.json"
    bm.write_text(json.dumps({"a": 100}))
    out = tmp_path / "out"
    xs.main([
        "--saliency-json", str(sal),
        "--byte-map-json", str(bm),
        "--output-dir", str(out),
    ])
    md = (out / "saliency_heatmap.md").read_text()
    assert "[diagnostic: per-tensor saliency heatmap]" in md
    assert "generated_at:" in md
