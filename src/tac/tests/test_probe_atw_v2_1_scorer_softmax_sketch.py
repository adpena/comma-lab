# SPDX-License-Identifier: MIT
"""Tests for the ATW V2-1 scorer-softmax sketch probe."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import probe_atw_v2_1_scorer_softmax_sketch as probe  # noqa: E402


def _one_hot_softmax(n_pairs: int, n_regions: int) -> np.ndarray:
    out = np.zeros((n_pairs, n_regions, 5), dtype=np.float32)
    for pair_idx in range(n_pairs):
        cls = pair_idx % 2
        out[pair_idx, :, cls] = 1.0
    return out


def test_build_scorer_softmax_sketches_is_deterministic_low_cardinality() -> None:
    n_pairs = 12
    sketches = probe.build_scorer_softmax_sketches(
        {
            16: _one_hot_softmax(n_pairs, 16),
            256: _one_hot_softmax(n_pairs, 256),
        }
    )

    assert [row.variant_id for row in sketches] == [
        "global_mean_softmax_q3",
        "global_top2_margin_q5",
        "region16_entropy_anchor_q4",
        "region16_presence_confmask_q4",
        "region256_coarse_entropy_anchor_q4",
    ]
    assert all(len(row.per_pair_symbols) == n_pairs for row in sketches)
    assert max(len(set(row.per_pair_symbols)) for row in sketches) == 2


def test_compute_sketch_mi_verdict_detects_correlated_symbols() -> None:
    latent = bytes([0, 0, 0, 0, 255, 255, 255, 255])
    verdict = probe.compute_sketch_mi_verdict(
        latent_stream=latent,
        per_pair_symbols=[10, 20],
        symbols_per_pair=4,
    )

    assert verdict.verdict == "MEANINGFUL_CONDITIONING"
    assert verdict.mutual_information_bits >= 0.99
    assert verdict.num_unique_side_info_symbols == 2


def test_build_probe_payload_marks_diagnostic_meaningful_gate(tmp_path: Path) -> None:
    n_pairs = 12
    latent = bytearray()
    for pair_idx in range(n_pairs):
        latent.extend(([0] if pair_idx % 2 == 0 else [255]) * 4)

    payload = probe.build_probe_payload(
        latent_stream=bytes(latent),
        softmax_by_region_count={
            16: _one_hot_softmax(n_pairs, 16),
            256: _one_hot_softmax(n_pairs, 256),
        },
        output_dir=tmp_path,
        softmax_provenance={"fixture": "one_hot"},
    )
    rendered = probe.render_markdown(payload)

    assert payload["schema"] == "atw_v2_1_scorer_softmax_sketch_probe_v1"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_paid_dispatch"] is False
    assert payload["phase2_status"] == (
        "scorer_softmax_sketch_meaningful_requires_new_d4_and_wave_n_plus_1_council"
    )
    assert payload["best_actionable_variant"] is not None
    assert payload["best_actionable_variant"]["mutual_information_bits"] >= 0.99
    first = payload["variants"][0]
    assert first["packet_roundtrip_ok"] is True
    assert first["byte_budget_ok"] is True
    assert first["high_cardinality_bias_guard_triggered"] is False
    assert Path(tmp_path / "atw_v2_1_scorer_softmax_sketch_global_mean_softmax_q3.bin").is_file()
    assert "not raw scorer logits" in rendered


def test_high_cardinality_guard_blocks_dispatch_authority(tmp_path: Path) -> None:
    n_pairs = 12
    softmax16 = np.zeros((n_pairs, 16, 5), dtype=np.float32)
    softmax256 = np.zeros((n_pairs, 256, 5), dtype=np.float32)
    latent = bytearray()
    for pair_idx in range(n_pairs):
        cls = pair_idx % 5
        conf = 0.52 + 0.03 * pair_idx
        rest = (1.0 - conf) / 4.0
        softmax16[pair_idx, :, :] = rest
        softmax256[pair_idx, :, :] = rest
        softmax16[pair_idx, :, cls] = conf
        softmax256[pair_idx, :, cls] = conf
        latent.extend([cls] * 4)

    payload = probe.build_probe_payload(
        latent_stream=bytes(latent),
        softmax_by_region_count={16: softmax16, 256: softmax256},
        output_dir=tmp_path,
    )

    assert payload["score_claim"] is False
    assert payload["best_actionable_variant"] is None
    assert payload["phase2_status"] == "scorer_softmax_sketches_only_weak_or_biased_conditioning"
    assert any(row["high_cardinality_bias_guard_triggered"] for row in payload["variants"])
    assert {
        row["phase2_action"]
        for row in payload["variants"]
        if row["high_cardinality_bias_guard_triggered"]
    } == {"reject_as_plugin_mi_upper_bound_until_lower_cardinality_or_heldout_probe"}
