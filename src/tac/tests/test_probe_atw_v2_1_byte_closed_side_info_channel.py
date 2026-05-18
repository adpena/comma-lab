# SPDX-License-Identifier: MIT
"""Tests for the ATW V2-1 byte-closed side-info probe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[3]
    target = repo_root / "tools" / "probe_atw_v2_1_byte_closed_side_info_channel.py"
    module_name = "probe_atw_v2_1_sideinfo_test"
    spec = importlib.util.spec_from_file_location(module_name, target)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_side_info_packet_round_trips_large_sparse_symbols() -> None:
    mod = _load_module()
    values = [0, 3840, 3840, 123456789, 0, 7, 123456789] * 9
    packet = mod.encode_side_info_packet(values, reducer_name="per_region_histogram")
    name, decoded = mod.decode_side_info_packet(packet)

    assert name == "per_region_histogram"
    assert decoded == values
    assert packet.startswith(mod.SIDE_INFO_MAGIC)


def test_dictionary_packet_keeps_low_cardinality_channel_under_budget() -> None:
    mod = _load_module()
    values = [(i % 7) * 987654321 for i in range(600)]
    packet = mod.encode_side_info_packet(values, reducer_name="per_region_histogram")
    name, decoded = mod.decode_side_info_packet(packet)

    assert name == "per_region_histogram"
    assert decoded == values
    assert len(packet) < 512


def test_build_probe_payload_is_fail_closed_and_marks_meaningful_channel(tmp_path: Path) -> None:
    mod = _load_module()
    n_pairs = 20
    symbols_per_pair = 4
    per_pair = [i % 5 for i in range(n_pairs)]
    latent_stream = bytes(
        per_pair[pair_idx]
        for pair_idx in range(n_pairs)
        for _ in range(symbols_per_pair)
    )
    reducer_outputs = {
        "per_pixel_histogram": per_pair,
        "per_region_histogram": [value * 1000 for value in per_pair],
        "per_pair_class_2_fraction": [0] * n_pairs,
        "per_frame_argmax": [0] * n_pairs,
    }

    payload = mod.build_probe_payload(
        latent_stream=latent_stream,
        per_pair_by_reducer=reducer_outputs,
        output_dir=tmp_path,
        source_reducer_json=tmp_path / "reducers.json",
        budget_bytes=2048,
    )

    assert payload["schema"] == "atw_v2_1_byte_closed_side_info_probe_v1"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_paid_dispatch"] is False
    assert payload["phase2_status"] == (
        "byte_closed_meaningful_channel_requires_wave_n_plus_1_council"
    )
    per_pixel = next(
        c for c in payload["channels"] if c["reducer_name"] == "per_pixel_histogram"
    )
    assert per_pixel["packet_roundtrip_ok"] is True
    assert per_pixel["byte_budget_ok"] is True
    assert per_pixel["verdict"] == "MEANINGFUL_CONDITIONING"
    assert Path(tmp_path / "atw_v2_1_side_info_per_pixel_histogram.bin").is_file()


def test_over_budget_channels_do_not_become_byte_closed_authority(tmp_path: Path) -> None:
    mod = _load_module()
    n_pairs = 20
    symbols_per_pair = 4
    per_pair = [i % 5 for i in range(n_pairs)]
    latent_stream = bytes(
        per_pair[pair_idx]
        for pair_idx in range(n_pairs)
        for _ in range(symbols_per_pair)
    )
    reducer_outputs = {
        "per_pixel_histogram": per_pair,
        "per_region_histogram": [value * 1000 for value in per_pair],
        "per_pair_class_2_fraction": [0] * n_pairs,
        "per_frame_argmax": [0] * n_pairs,
    }

    payload = mod.build_probe_payload(
        latent_stream=latent_stream,
        per_pair_by_reducer=reducer_outputs,
        output_dir=tmp_path,
        source_reducer_json=tmp_path / "reducers.json",
        budget_bytes=1,
    )
    rendered = mod.render_markdown(payload)

    assert payload["best_byte_closed_channel"] is None
    assert payload["best_overall_channel"]["byte_budget_ok"] is False
    assert all(not channel["byte_budget_ok"] for channel in payload["channels"])
    assert {
        channel["phase2_action"] for channel in payload["channels"]
    } == {"reject_or_recode_side_info_payload_before_mi_interpretation"}
    assert "No byte-closed channel fit" in rendered
    assert "Best byte-closed channel" not in rendered
