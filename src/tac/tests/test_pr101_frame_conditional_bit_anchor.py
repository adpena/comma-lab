from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

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
    assert contract["typed_sideinfo_wire_contract_landed"] is True
    assert (
        "per_pair_bit_width_schema_change_requires_inflate_path_update"
        in contract["cleared_blockers"]
    )
    assert (
        "per_pair_bit_width_schema_change_requires_inflate_path_update"
        not in contract["dispatch_blockers"]
    )
    assert (
        "frame_conditional_packet_runtime_patch_not_built"
        in contract["dispatch_blockers"]
    )


def test_uniform_anchor_row_keeps_stock_pr101_schema() -> None:
    tool = _load_tool()

    contract = tool._implicit_uniform_wire_contract()

    assert contract["score_claim"] is False
    assert contract["sideinfo_required"] is False
    assert contract["q_bits_sideinfo"]["bytes"] == 0
    assert contract["decoder_helper_consumes_sideinfo_bytes"] is False


def test_uniform_only_sweep_does_not_report_changed_charged_bits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()
    monkeypatch.setattr(tool, "PR101_DECODER_BLOB_LEN", 10)
    monkeypatch.setattr(tool, "PR101_LATENT_BLOB_LEN", 4)
    monkeypatch.setattr(tool, "PR101_N_PAIRS", 2)
    monkeypatch.setattr(tool, "PR101_LATENT_DIM", 4)
    monkeypatch.setattr(tool, "_read_pr101_archive_bytes", lambda _path: b"A" * 20)
    monkeypatch.setattr(
        tool,
        "_extract_pr101_latent_payload",
        lambda _archive: (
            np.zeros(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
            np.arange(8, dtype=np.uint8).reshape(2, 4),
        ),
    )
    monkeypatch.setattr(tool, "_encode_pr101_latent_stream", lambda *_args: b"LATN")
    monkeypatch.setattr(
        tool,
        "_build_per_pair_complexity",
        lambda *_args, **_kwargs: np.ones(2, dtype=np.float64),
    )

    manifest = tool._sweep_etas(
        tmp_path / "archive.zip",
        tmp_path / "0.mkv",
        [0.0],
        floor=0.5,
        cap=2.0,
        output_dir=tmp_path,
    )

    assert manifest["best_eta"] == 0.0
    assert manifest["score_affecting_payload_changed"] is False
    assert manifest["charged_bits_changed"] is False
    assert manifest["best_archive_delta_bytes"] == 0
    assert manifest["rows"][0]["sidechannel_overhead_bytes"] == 0
    assert manifest["rows"][0]["score_affecting_payload_changed"] is False
    assert manifest["rows"][0]["charged_bits_changed"] is False
