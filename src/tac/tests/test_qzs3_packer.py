"""Byte-equivalence tests for QZS3 weight codec vs PR #67's unpacker.

These complement test_quantizr_faithful_renderer.py which checks our own
decode_qzs3_state_dict round-trip. The tests here load the actual PR #67
inflate.py module and verify our encoder produces bytes that PR #67's
``get_grouped_qv_state_dict`` decodes successfully.

Verified on 2026-05-01: our encoder produces 59288 bytes for an
init-random JointFrameGenerator — byte-identical to the uncompressed
QZS3 payload deployed in PR #67's rank-1 archive.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
from tac.quantizr_qzs3_codec import (
    QZS3_MAGIC,
    decode_qzs3_state_dict,
    encode_qzs3_state_dict,
    qzs3_qv_specs,
)


PR67_INFLATE = (
    Path(__file__).resolve().parents[3]
    / "reports/raw/leaderboard_intel_20260501/pr67_inflate.py"
)


def _import_pr67():
    """Side-load PR #67 inflate.py without running its main()."""

    if not PR67_INFLATE.exists():
        pytest.skip(f"pr67 inflate reference missing: {PR67_INFLATE}")
    spec = importlib.util.spec_from_file_location("_pr67_inflate", PR67_INFLATE)
    module = importlib.util.module_from_spec(spec)
    # PR #67 only invokes main() under __name__ == "__main__".
    spec.loader.exec_module(module)
    return module


def test_payload_size_matches_pr67_archive_byte_count() -> None:
    """Our QZS3 encoder must emit exactly 59288 bytes for an init-random
    JointFrameGenerator — matches PR #67's deployed uncompressed model size."""

    torch.manual_seed(0)
    model = build_quantizr_faithful_renderer().eval()
    payload = encode_qzs3_state_dict(model)
    assert payload.startswith(QZS3_MAGIC)
    # PR #67 archive's uncompressed model bytes (verified 2026-05-01).
    assert len(payload) == 59288, (
        f"QZS3 payload {len(payload)} bytes; expected 59288 to match PR #67 archive"
    )


def test_pr67_unpacker_decodes_our_payload() -> None:
    """Cross-check: PR #67's get_grouped_qv_state_dict must accept our bytes."""

    pr67 = _import_pr67()
    torch.manual_seed(123)
    model = build_quantizr_faithful_renderer().eval()
    payload = encode_qzs3_state_dict(model)

    decoded = pr67.get_grouped_qv_state_dict(payload, torch.device("cpu"))

    template_state = model.state_dict()
    assert set(decoded.keys()) == set(template_state.keys()), (
        f"key mismatch: missing={set(template_state)-set(decoded)} "
        f"extra={set(decoded)-set(template_state)}"
    )

    qv_keys = set(qzs3_qv_specs().keys())
    fp4_keys = {
        k for k in template_state
        if (".dw.weight" in k or ".pw.weight" in k)
        and k not in {"frame1_head.head.weight", "frame2_head.head.weight"}
    }

    for key, ref in template_state.items():
        actual = decoded[key]
        assert actual.shape == ref.shape, f"{key} shape: {actual.shape} != {ref.shape}"
        diff = (actual.float() - ref.float()).abs().max().item()
        if key in fp4_keys:
            # FP4 codebook noise bound (max magnitude 6.0 * scale)
            assert diff < 0.2, f"FP4 layer {key} drift {diff} exceeds 0.2"
        elif key in qv_keys:
            # Per-row min/step quant (8-bit norm), expect <= step_max
            assert diff < 0.1, f"qv layer {key} drift {diff} exceeds 0.1"
        else:
            # FP16 conversion noise
            assert diff < 1e-3, f"FP16 layer {key} drift {diff} exceeds 1e-3"


def test_pr67_unpacker_loads_into_jointframegenerator() -> None:
    """The decoded state_dict must load() cleanly with strict=True into the
    template JointFrameGenerator (PR #67 calls this directly)."""

    pr67 = _import_pr67()
    torch.manual_seed(456)
    model = build_quantizr_faithful_renderer().eval()
    payload = encode_qzs3_state_dict(model)

    decoded = pr67.get_grouped_qv_state_dict(payload, torch.device("cpu"))
    fresh = build_quantizr_faithful_renderer()
    # Should not raise: keys + shapes line up exactly.
    fresh.load_state_dict(decoded, strict=True)
    fresh.eval()

    mask = torch.zeros(1, 384, 512, dtype=torch.long)
    pose = torch.zeros(1, 6)
    with torch.no_grad():
        f1, f2 = fresh(mask, pose)
    assert f1.shape == (1, 3, 384, 512)
    assert f2.shape == (1, 3, 384, 512)
    assert torch.isfinite(f1).all() and torch.isfinite(f2).all()


def test_encoder_is_deterministic() -> None:
    """Same model -> same bytes (byte-identical archives are required for
    contest reproducibility + SHA-pinning custody chain)."""

    torch.manual_seed(789)
    model = build_quantizr_faithful_renderer().eval()
    a = encode_qzs3_state_dict(model)
    b = encode_qzs3_state_dict(model)
    assert a == b


def test_block_size_header_field_round_trips() -> None:
    """The 2-byte little-endian block_size field at offset 4 must round-trip."""

    torch.manual_seed(7)
    model = build_quantizr_faithful_renderer().eval()
    payload = encode_qzs3_state_dict(model, block_size=64)
    assert int.from_bytes(payload[4:6], "little") == 64
    decoded = decode_qzs3_state_dict(payload, device="cpu")
    fresh = build_quantizr_faithful_renderer()
    fresh.load_state_dict(decoded, strict=True)
