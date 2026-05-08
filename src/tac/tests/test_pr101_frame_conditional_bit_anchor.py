from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "pr101_frame_conditional_bit_anchor.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pr101_frame_conditional_bit_anchor", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bits_per_pair_to_q_bits_clips_to_uint8_precision_bounds() -> None:
    tool = _load_tool()

    out = tool._bits_per_pair_to_q_bits(
        np.array([0.0, 28.0, 112.0, 224.0, 280.0], dtype=np.float64),
        latent_dim=28,
    )

    np.testing.assert_allclose(out, np.array([1.0, 1.0, 4.0, 8.0, 8.0]))


def test_requantise_per_pair_reduces_only_low_precision_rows() -> None:
    tool = _load_tool()
    q = np.array(
        [
            [255, 128, 17, 1],
            [255, 128, 17, 1],
        ],
        dtype=np.uint8,
    )

    out = tool._requantise_per_pair(q, np.array([4.0, 8.0], dtype=np.float64))

    np.testing.assert_array_equal(out[0], np.array([240, 128, 16, 0], dtype=np.uint8))
    np.testing.assert_array_equal(out[1], q[1])


def test_frame_conditional_anchor_proxy_contract_is_non_promotable() -> None:
    tool = _load_tool()

    contract = tool._proxy_evidence_contract()

    assert contract["score_claim"] is False
    assert contract["byte_proxy_only"] is True
    assert contract["promotion_eligible"] is False
    assert contract["rank_or_kill_eligible"] is False
    assert contract["ready_for_exact_eval_dispatch"] is False
    assert "awaiting_per_frame_score_marginal" in contract["dispatch_blockers"]
