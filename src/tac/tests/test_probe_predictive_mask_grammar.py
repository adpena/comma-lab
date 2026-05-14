# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "probe_predictive_mask_grammar.py"
SPEC = importlib.util.spec_from_file_location("probe_predictive_mask_grammar", MODULE_PATH)
assert SPEC is not None
probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def _tiny_masks() -> np.ndarray:
    arr = np.zeros((6, 8, 12), dtype=np.uint8)
    arr[:, 3:, :] = 1
    arr[:, 5:, 2:10] = 2
    arr[1::2, 2:5, 4:8] = 3
    arr[4:, 1:3, 8:11] = 4
    return arr


def test_candidate_payloads_are_deterministic_and_finite() -> None:
    arr = _tiny_masks()

    first = probe.build_candidate_payloads(arr)
    second = probe.build_candidate_payloads(arr)

    assert 5 <= len(first) < 20
    assert [item.candidate_id for item in first] == [item.candidate_id for item in second]
    assert [item.payload for item in first] == [item.payload for item in second]
    assert {item.family for item in first} >= {
        "temporal_pair_delta",
        "class_boundary_maps",
        "low_rank_row_column_spans",
        "keyframe_residual_schedule",
        "anisotropic_foveal",
    }


def test_probe_report_has_byte_accounting_and_no_score_claim(tmp_path: Path) -> None:
    masks_path = tmp_path / "decoded_mask_array.npy"
    np.save(masks_path, _tiny_masks())

    report = probe.run_probe(
        probe.ProbeConfig(
            decoded_mask_array=masks_path,
            output_dir=tmp_path / "out",
            baseline_bytes=500,
            compressors=("zlib9", "bz2_9"),
        ),
        command=["unit-test"],
    )

    manifest_path = tmp_path / "out" / probe.REPORT_NAME
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text()) == report
    assert report["schema"] == "predictive_mask_grammar_byte_probe_v1"
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["cloud_jobs_dispatched"] is False
    assert report["evidence_grade"] == "empirical_byte_probe_only"
    assert "contest_auth_eval.py --device cuda" in report["canonical_score_source_required"]
    assert report["baseline"]["bytes"] == 500
    assert report["probe_config"]["candidate_count"] == len(report["candidate_table"])

    for candidate in report["candidate_table"]:
        assert candidate["score_claim"] is False
        assert candidate["promotion_eligible"] is False
        assert candidate["evidence_grade"] == "empirical_byte_probe_only"
        assert candidate["exact_evaluable_now"] is False
        assert candidate["raw_payload_size_bytes"] > 0
        assert len(candidate["raw_payload_sha256"]) == 64
        assert candidate["best_compression"]["compressed_size_bytes"] == min(
            item["compressed_size_bytes"] for item in candidate["compression"]
        )
        for compressed in candidate["compression"]:
            assert compressed["compressed_size_bytes"] > 0
            assert len(compressed["compressed_sha256"]) == 64
            assert compressed["delta_bytes_vs_baseline"] == compressed["compressed_size_bytes"] - 500


def test_probe_report_candidate_table_is_repeatable(tmp_path: Path) -> None:
    masks_path = tmp_path / "decoded_mask_array.npy"
    np.save(masks_path, _tiny_masks())
    config_a = probe.ProbeConfig(
        decoded_mask_array=masks_path,
        output_dir=tmp_path / "out-a",
        compressors=("zlib9",),
    )
    config_b = probe.ProbeConfig(
        decoded_mask_array=masks_path,
        output_dir=tmp_path / "out-b",
        compressors=("zlib9",),
    )

    report_a = probe.run_probe(config_a, command=["unit-test"])
    report_b = probe.run_probe(config_b, command=["unit-test"])

    assert report_a["source"] == report_b["source"]
    assert report_a["candidate_table"] == report_b["candidate_table"]
    assert report_a["best_candidate_by_compressed_size"] == report_b["best_candidate_by_compressed_size"]


def test_default_discovery_prefers_c063_decoded_mask_array(tmp_path: Path) -> None:
    decoded = (
        tmp_path
        / "experiments"
        / "results"
        / "c063_trace_weighted_mask_grammar_plan_20260502_codex"
        / "decoded_mask_array.npy"
    )
    decoded.parent.mkdir(parents=True)
    np.save(decoded, _tiny_masks())

    found = probe.discover_decoded_mask_array(repo_root=tmp_path)

    assert found is not None
    assert found.path == decoded
    assert "C-063" in found.reason


def test_discovered_baseline_uses_prior_charged_probe_manifest(tmp_path: Path) -> None:
    manifest = (
        tmp_path
        / "experiments"
        / "results"
        / "c063_trace_weighted_mask_grammar_plan_20260502_codex"
        / "cmg2_lossless_probe_charged_pr67_20260502T0950Z"
        / "cmg2_mask_codec_probe_manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"baseline": {"role": "charged_current_mask_stream", "bytes": 219472}}))

    baseline = probe.discover_baseline(repo_root=tmp_path)

    assert baseline is not None
    assert baseline.bytes == 219472
    assert baseline.role == "charged_current_mask_stream"
