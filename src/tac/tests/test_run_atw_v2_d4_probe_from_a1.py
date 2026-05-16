# SPDX-License-Identifier: MIT
"""Static contract tests for the ATW v2 D4 probe runner."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TOOL = REPO_ROOT / "tools" / "run_atw_v2_d4_probe_from_a1.py"


def _source() -> str:
    return TOOL.read_text()


def test_atw_v2_d4_probe_runner_is_repo_relative_and_axis_labelled() -> None:
    text = _source()
    assert "/Users/adpena" not in text
    assert "REPO_ROOT = Path(__file__).resolve().parents[1]" in text
    assert '" ".join([".venv/bin/python", _repo_rel(Path(__file__)), *sys.argv[1:]])' in text
    assert "atw_codec_v2" in text
    assert "[diagnostic-CPU; H(latent|scorer_class) probe]" in text
    assert '"score_claim": False' in text
    assert '"promotion_eligible": False' in text
    assert '"ready_for_exact_eval_dispatch": False' in text


def test_atw_v2_d4_probe_runner_uses_canonical_entropy_helper() -> None:
    tree = ast.parse(_source())
    names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "compute_h_latent_given_scorer_class" in names
    assert "eval" not in names
    assert "exec" not in names


def test_atw_v2_d4_probe_runner_reuses_tishby_class_artifact_with_hashes() -> None:
    text = _source()
    assert "per_pair_segnet_class.json" in text
    assert "class_json_sha256" in text
    assert "latent_u8_sha256" in text
    assert "class_stream_sha256" in text
    assert "a1_archive_zip_sha256" in text
    assert "a1_inner_member_sha256" in text


def test_atw_v2_d4_probe_runner_writes_durable_research_verdict() -> None:
    text = _source()
    assert ".omx\" / \"research\" / \"atw_codec_v2_d4_probe_verdict_20260516_codex.json" in text
    assert ".omx\" / \"research\" / \"atw_codec_v2_d4_probe_verdict_20260516_codex.md" in text
    assert ".omx\" / \"state\" / \"h_latent_given_scorer_class_atw_codec_v2.json" in text
    assert "defer_measured_a1_latent_class_conditioning_surface" in text
    assert "probe_passed_ready_for_variant_b_smoke_claim_lifecycle" in text
