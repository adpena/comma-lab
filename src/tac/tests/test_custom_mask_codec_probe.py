# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "custom_mask_codec_probe.py"
SPEC = importlib.util.spec_from_file_location("custom_mask_codec_probe", MODULE_PATH)
assert SPEC is not None
probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def test_cmcp_rle1_roundtrips_deterministic_synthetic_masks() -> None:
    shape = probe.MaskShape(frames=3, height=10, width=12, classes=5)
    symbols = probe.make_synthetic_masks(shape)

    encoded_a = probe.encode_mask_rle_bitpacked(symbols, shape)
    encoded_b = probe.encode_mask_rle_bitpacked(symbols, shape)
    decoded_shape, decoded = probe.decode_mask_rle_bitpacked(encoded_a)

    assert encoded_a == encoded_b
    assert decoded_shape == shape
    assert decoded == symbols
    assert len(encoded_a) < len(symbols)


def test_fixed_width_bitpacking_rejects_non_zero_padding() -> None:
    packed = bytearray(probe.pack_fixed_width([1], bits_per_symbol=3))
    packed[-1] |= 0b0000_0001

    with pytest.raises(ValueError, match="non-zero padding"):
        probe.unpack_fixed_width(bytes(packed), count=1, bits_per_symbol=3)


def test_decoder_rejects_crc_mismatch() -> None:
    shape = probe.MaskShape(frames=2, height=4, width=6, classes=5)
    payload = bytearray(probe.encode_mask_rle_bitpacked(probe.make_synthetic_masks(shape), shape))
    payload[-1] ^= 0x01

    with pytest.raises(ValueError, match="CRC mismatch"):
        probe.decode_mask_rle_bitpacked(bytes(payload))


def test_probe_report_is_non_promotable_and_deterministic() -> None:
    config = probe.ProbeConfig(frames=3, height=10, width=12, classes=5)

    report_a, payload_a = probe.build_probe_report(config=config, command=["unit-test"])
    report_b, payload_b = probe.build_probe_report(config=config, command=["unit-test"])

    assert payload_a == payload_b
    assert json.dumps(report_a, sort_keys=True) == json.dumps(report_b, sort_keys=True)
    assert report_a["schema"] == "custom_mask_codec_probe_v1"
    assert report_a["score_claim"] is False
    assert report_a["promotion_eligible"] is False
    assert report_a["evidence_grade"] == "empirical"
    assert report_a["local_probe_only"] is True
    assert report_a["scorer_network_loaded"] is False
    assert "contest_auth_eval.py --device cuda" in report_a["canonical_score_source_required"]
    assert report_a["codec"]["roundtrip_bit_exact"] is True
    assert report_a["codec"]["score_claim"] is False
    assert report_a["codec"]["promotion_eligible"] is False


def test_run_probe_writes_payload_and_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "probe"
    report = probe.run_probe(
        output_dir=output_dir,
        config=probe.ProbeConfig(frames=2, height=8, width=8, classes=5),
        command=["custom_mask_codec_probe.py", "--unit-test"],
    )

    report_path = output_dir / probe.REPORT_NAME
    payload_path = output_dir / probe.PAYLOAD_NAME

    assert report_path.exists()
    assert payload_path.exists()
    assert json.loads(report_path.read_text()) == report
    assert report["artifacts"]["payload"]["size_bytes"] == payload_path.stat().st_size
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False


def test_run_probe_requires_force_for_non_empty_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "probe"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep")

    with pytest.raises(FileExistsError, match="--force"):
        probe.run_probe(
            output_dir=output_dir,
            config=probe.ProbeConfig(),
            command=["unit-test"],
        )
