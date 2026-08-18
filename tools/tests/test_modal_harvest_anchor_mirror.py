"""Tests for the harvest-time anchor mirror (round-11 F1(b) class cure).

The keep01 contest-CUDA row was complete on disk under an SSD ``--output-dir``
and INVISIBLE to ``tac.frontier_scan``, which globs only ``experiments/results``.
The pointer went stale and had to be repaired by hand. These tests pin the two
properties that make the automatic mirror actually work:

1. the mirror lands where the scanner globs, under a filename the glob matches;
2. the score is the recomputed-from-components value, never the rounded
   ``final_score`` -- with no fallback path that could reintroduce it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_POLLER = Path(__file__).resolve().parents[1] / "modal_harvest_poller.py"
_spec = importlib.util.spec_from_file_location("modal_harvest_poller", _POLLER)
assert _spec is not None and _spec.loader is not None
poller = importlib.util.module_from_spec(_spec)
sys.modules["modal_harvest_poller"] = poller
_spec.loader.exec_module(poller)


def _result(**overrides):
    base = {
        "final_score": 0.16,  # the ROUNDED field; must never be the anchor
        "score_recomputed_from_components": 0.1571619225142182,
        "avg_posenet_dist": 7.72e-06,
        "avg_segnet_dist": 0.00030135,
        "n_samples": 600,
        "expected_archive_sha256": "316d17f8" + "0" * 56,
        "expected_archive_size_bytes": 177576,
        "archive_size_bytes": 177576,
        "gpu_model": "Tesla T4",
        "gpu_t4_match": True,
        "score_axis": "contest_cuda",
        "evidence_grade": "contest-CUDA",
        "promotion_eligible": False,
    }
    base.update(overrides)
    return base


def test_mirror_score_is_recomputed_never_rounded(tmp_path):
    receipt = tmp_path / "MODAL_REMOTE_RESULT.json"
    receipt.write_text(json.dumps(_result()))

    payload, blocker = poller.build_anchor_mirror(
        _result(), lane_id="lane_x", source_receipt=receipt
    )

    assert blocker is None
    assert payload["score"] == 0.1571619225142182
    assert payload["score"] != 0.16
    assert "final_score" not in payload


def test_mirror_refuses_when_recomputed_score_absent(tmp_path):
    """No fallback to final_score: absent recomputed score means NO anchor."""
    receipt = tmp_path / "MODAL_REMOTE_RESULT.json"
    result = _result()
    del result["score_recomputed_from_components"]
    receipt.write_text(json.dumps(result))

    payload, blocker = poller.build_anchor_mirror(
        result, lane_id="lane_x", source_receipt=receipt
    )

    assert payload is None
    assert "score_recomputed_from_components" in blocker


def test_mirror_lands_where_frontier_scan_globs(tmp_path):
    """The mirror must satisfy experiments/results/*/contest_auth_eval*.json."""
    receipt = tmp_path / "MODAL_REMOTE_RESULT.json"
    receipt.write_text(json.dumps(_result()))

    path = poller.write_anchor_mirror(
        _result(),
        source_receipt=receipt,
        label="sa3_keep01_t4_row_r2",
        lane_id="ddm_sa3_keep01_composed_t4_row_r2",
        repo_root=tmp_path,
        out_dir=tmp_path,
    )

    assert path is not None
    matched = list((tmp_path / "experiments/results").glob("*/contest_auth_eval*.json"))
    assert matched == [path]


def test_mirror_is_seen_by_frontier_scan(tmp_path):
    """END-TO-END: the real scanner turns the mirror into a qualifying Anchor."""
    from tac.frontier_scan import load_experiments_results_anchors

    receipt = tmp_path / "MODAL_REMOTE_RESULT.json"
    receipt.write_text(json.dumps(_result()))
    poller.write_anchor_mirror(
        _result(),
        source_receipt=receipt,
        label="sa3_keep01_t4_row_r2",
        lane_id="ddm_sa3_keep01_composed_t4_row_r2",
        repo_root=tmp_path,
        out_dir=tmp_path,
    )

    anchors = load_experiments_results_anchors(tmp_path)

    assert len(anchors) == 1
    anchor = anchors[0]
    assert anchor.score == 0.1571619225142182
    assert anchor.canonical_axis() == "contest_cuda"
    assert anchor.hardware_substrate == "linux_x86_64_t4"
    assert anchor.is_qualifying()


def test_source_receipt_is_hash_pinned(tmp_path):
    import hashlib

    receipt = tmp_path / "MODAL_REMOTE_RESULT.json"
    receipt.write_text(json.dumps(_result()))
    expected = hashlib.sha256(receipt.read_bytes()).hexdigest()

    payload, _ = poller.build_anchor_mirror(
        _result(), lane_id=None, source_receipt=receipt
    )

    assert payload["source_receipt_sha256"] == expected
    assert payload["source_receipt"] == str(receipt)


def test_substrate_derived_from_gpu_model_when_no_t4_match():
    result = _result(gpu_t4_match=False, gpu_model="NVIDIA A10G")
    assert poller._canonical_substrate(result) == "linux_x86_64_a10g"


def test_substrate_none_refuses_and_leaves_a_marker(tmp_path):
    """Unknown hardware writes no anchor, but does write a discoverable blocker."""
    receipt = tmp_path / "MODAL_REMOTE_RESULT.json"
    result = _result(gpu_t4_match=False, gpu_model="Some Unlisted GPU")
    receipt.write_text(json.dumps(result))

    path = poller.write_anchor_mirror(
        result,
        source_receipt=receipt,
        label="x",
        lane_id=None,
        repo_root=tmp_path,
        out_dir=tmp_path,
    )

    assert path is None
    marker = json.loads((tmp_path / "MIRROR_UNWRITTEN.json").read_text())
    assert "no canonical hardware substrate" in marker["blocker"]
    assert not (tmp_path / "experiments/results").exists()


def test_label_is_sanitized_into_a_safe_filename(tmp_path):
    receipt = tmp_path / "MODAL_REMOTE_RESULT.json"
    receipt.write_text(json.dumps(_result()))

    path = poller.write_anchor_mirror(
        _result(),
        source_receipt=receipt,
        label="lane/../weird label",
        lane_id=None,
        repo_root=tmp_path,
        out_dir=tmp_path,
    )

    assert path is not None
    assert path.parent == tmp_path / poller.MIRROR_DIR_REL
    assert "/" not in path.name
    assert " " not in path.name
