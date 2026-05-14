# SPDX-License-Identifier: MIT
"""Tests for ``tools/probe_hdm8_postfilter_frame_parity_and_no_op.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PROBE_PATH = REPO / "tools/probe_hdm8_postfilter_frame_parity_and_no_op.py"
ARCHIVE_PATH = (
    REPO
    / "experiments/results/pr106_r2_hdm7_hlm2_hdm8_candidate_20260514_codex/"
    "exact_eval_static_release_surface/archive.zip"
)
RUNTIME_TEMPLATE = REPO / "submissions/hdm8_film_grain_sidecar"


def _load_probe():
    # Force fresh runtime imports (codec/model/pr101_grammar) so test ordering
    # doesn't bleed cached siblings from other test files.
    for name in ("codec", "model", "pr101_grammar"):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(
        "probe_hdm8_postfilter_frame_parity_and_no_op", PROBE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_modes_handles_rgb_bias_triplets() -> None:
    probe = _load_probe()
    assert probe._parse_modes("none,even_grain_chroma:1.0,even_grain_chroma:2.0") == [
        "none",
        "even_grain_chroma:1.0",
        "even_grain_chroma:2.0",
    ]
    assert probe._parse_modes("none,even_rgb_bias:2,-1,-1") == [
        "none",
        "even_rgb_bias:2,-1,-1",
    ]
    assert probe._parse_modes(
        "none,rgb_bias:1,2,3,even_bias:5"
    ) == ["none", "rgb_bias:1,2,3", "even_bias:5"]
    # Chained rgb_bias modes — both triplets must survive intact.
    assert probe._parse_modes(
        "none,even_rgb_bias:2,0,-2,even_rgb_bias:-2,1,1,even_bias:1"
    ) == [
        "none",
        "even_rgb_bias:2,0,-2",
        "even_rgb_bias:-2,1,1",
        "even_bias:1",
    ]


def test_parse_modes_rejects_empty_and_handles_whitespace() -> None:
    probe = _load_probe()
    assert probe._parse_modes("") == []
    assert probe._parse_modes("  none ,  even_grain:0.5  ") == ["none", "even_grain:0.5"]


def test_verify_frame_parity_none_pass() -> None:
    probe = _load_probe()
    shas = [f"sha{i}" for i in range(8)]
    out = probe._verify_frame_parity(
        baseline_frame_shas=shas,
        mode_frame_shas=shas,
        n_frames=8,
        mode="none",
    )
    assert out["passed"] is True
    assert out["parity_contract"] == "no_change_anywhere"


def test_verify_frame_parity_even_only_passes_when_odd_frames_unchanged() -> None:
    probe = _load_probe()
    baseline = [f"base_{i}" for i in range(8)]
    candidate = list(baseline)
    # Change only even indices (frame 0 of each pair).
    candidate[0] = "changed_0"
    candidate[2] = "changed_2"
    out = probe._verify_frame_parity(
        baseline_frame_shas=baseline,
        mode_frame_shas=candidate,
        n_frames=8,
        mode="even_grain_chroma:1.0",
    )
    assert out["passed"] is True
    assert out["parity_contract"] == "first_frame_only_segnet_null"
    assert out["n_odd_mismatches"] == 0
    assert out["n_even_mismatches"] == 2


def test_verify_frame_parity_even_only_fails_when_odd_frames_change() -> None:
    probe = _load_probe()
    baseline = [f"base_{i}" for i in range(8)]
    candidate = list(baseline)
    candidate[1] = "leaked_into_odd_1"  # frame 1 of pair 0 changed = SegNet sees it
    out = probe._verify_frame_parity(
        baseline_frame_shas=baseline,
        mode_frame_shas=candidate,
        n_frames=8,
        mode="even_grain_chroma:1.0",
    )
    assert out["passed"] is False
    assert out["n_odd_mismatches"] == 1
    assert out["first_odd_mismatch_idx"] == 1


def test_verify_no_op_passes_when_mode_changes_bytes() -> None:
    probe = _load_probe()
    baseline = bytes(b"\x00" * 100)
    candidate = bytes(b"\x01" + b"\x00" * 99)
    out = probe._verify_no_op_pixel_delta(
        baseline_bytes=baseline,
        mode_bytes=candidate,
        mode="even_bias:1",
    )
    assert out["passed"] is True
    assert out["n_bytes_changed"] == 1
    assert out["sum_abs_delta"] == 1
    assert out["max_abs_delta"] == 1


def test_verify_no_op_fails_when_non_none_mode_emits_zero_pixel_delta() -> None:
    probe = _load_probe()
    baseline = bytes(b"\x00" * 100)
    candidate = bytes(b"\x00" * 100)
    out = probe._verify_no_op_pixel_delta(
        baseline_bytes=baseline,
        mode_bytes=candidate,
        mode="even_grain_chroma:0.001",  # too-small amplitude under uint8 round
    )
    assert out["passed"] is False
    assert out["n_bytes_changed"] == 0


def test_verify_no_op_none_baseline_passes() -> None:
    probe = _load_probe()
    baseline = bytes(b"\x42" * 100)
    out = probe._verify_no_op_pixel_delta(
        baseline_bytes=baseline,
        mode_bytes=baseline,
        mode="none",
    )
    assert out["passed"] is True
    assert out["is_no_op_expected"] is True
    assert out["n_bytes_changed"] == 0


def test_verify_no_op_shape_mismatch_fails() -> None:
    probe = _load_probe()
    out = probe._verify_no_op_pixel_delta(
        baseline_bytes=b"\x00" * 100,
        mode_bytes=b"\x00" * 99,
        mode="any",
    )
    assert out["passed"] is False
    assert out["blocker"] == "shape_mismatch"


@pytest.mark.skipif(
    not ARCHIVE_PATH.exists() or not RUNTIME_TEMPLATE.exists(),
    reason="HDM8 archive/runtime template not present in the working tree",
)
def test_end_to_end_smoke_2_pairs_cpu(tmp_path: Path) -> None:
    """End-to-end CPU smoke: 2 pairs, none + a known-positive even_* mode.

    Validates the full inflate flow + frame-parity + no-op proofs against the
    pinned HDM8 archive shipped in the repo. ~10 second wall-clock on CPU.
    """
    probe = _load_probe()
    output = tmp_path / "proof.json"
    rc = probe.main(
        [
            "--archive",
            str(ARCHIVE_PATH),
            "--runtime-template",
            str(RUNTIME_TEMPLATE),
            "--modes",
            "none,even_grain_chroma:2.0",
            "--n-pairs",
            "2",
            "--device",
            "cpu",
            "--output-json",
            str(output),
        ]
    )
    assert rc == 0
    proof = json.loads(output.read_text())
    assert proof["all_modes_passed"] is True
    assert proof["blocker_modes"] == []
    assert proof["score_claim"] is False
    assert proof["promotion_eligible"] is False
    assert proof["axis"] == "local-cpu-proof"
    assert proof["n_pairs_decoded"] == 2
    assert proof["n_frames"] == 4
    # SegNet structural-nullspace contract holds empirically.
    even_grain = next(m for m in proof["modes"] if m["mode"] == "even_grain_chroma:2.0")
    assert even_grain["frame_parity_proof"]["n_odd_mismatches"] == 0
    assert even_grain["frame_parity_proof"]["passed"] is True
    assert even_grain["no_op_pixel_delta_proof"]["n_bytes_changed"] > 0


def test_main_rejects_modes_without_none_baseline_first(tmp_path: Path) -> None:
    probe = _load_probe()
    output = tmp_path / "proof.json"
    with pytest.raises(SystemExit) as exc_info:
        probe.main(
            [
                "--archive",
                str(ARCHIVE_PATH if ARCHIVE_PATH.exists() else tmp_path),
                "--runtime-template",
                str(RUNTIME_TEMPLATE),
                "--modes",
                "even_grain_chroma:1.0",
                "--n-pairs",
                "1",
                "--device",
                "cpu",
                "--output-json",
                str(output),
            ]
        )
    assert "must start with 'none'" in str(exc_info.value)


def test_main_rejects_invalid_modes(tmp_path: Path) -> None:
    if not RUNTIME_TEMPLATE.exists():
        pytest.skip("runtime template not present")
    probe = _load_probe()
    output = tmp_path / "proof.json"
    with pytest.raises(ValueError):
        probe.main(
            [
                "--archive",
                str(ARCHIVE_PATH if ARCHIVE_PATH.exists() else tmp_path),
                "--runtime-template",
                str(RUNTIME_TEMPLATE),
                "--modes",
                "none,bogus_mode:99",
                "--n-pairs",
                "1",
                "--device",
                "cpu",
                "--output-json",
                str(output),
            ]
        )
