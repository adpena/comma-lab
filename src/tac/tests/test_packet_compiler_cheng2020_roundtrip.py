"""Roundtrip + golden-vector tests for the CompressAI Cheng-2020 adapter.

Sister to ``src/tac/packet_compiler/cheng2020.py``. Catalog #91 accepts
this sibling test file as the paired roundtrip proof for the adapter.
"""

# ENCODE_INFLATE_ROUNDTRIP — pytest covers encode->serialize->deserialize->
# decode end-to-end through compressai.models.{Cheng2020Anchor,
# Cheng2020Attention}.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
compressai_module = pytest.importorskip("compressai")
from compressai.models import Cheng2020Anchor  # noqa: E402

from tac.packet_compiler import (  # noqa: E402
    CHENG2020_VERSION,
    MAGIC_CHENG2020,
    VARIANT_ANCHOR,
    VARIANT_ATTENTION,
    Cheng2020Error,
    Cheng2020Payload,
    decode_cheng2020,
    deserialize_cheng2020,
    encode_cheng2020,
    serialize_cheng2020,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent / "packet_compiler" / "golden_vectors"
)


def _build_updated_anchor(*, n: int = 64) -> Cheng2020Anchor:
    torch.manual_seed(20260512)
    model = Cheng2020Anchor(N=n)
    model.update(force=True)
    model.eval()
    return model


# ── Round-trip / behavior ───────────────────────────────────────────────────


def test_cheng2020_payload_is_frozen() -> None:
    payload = Cheng2020Payload(
        magic=MAGIC_CHENG2020,
        version=CHENG2020_VERSION,
        variant=VARIANT_ANCHOR,
        n_batch=1,
        shape=(1, 1),
        strings=((b"",), (b"",)),
    )
    with pytest.raises((AttributeError, TypeError)):
        payload.variant = 0xFF  # type: ignore[misc]


def test_cheng2020_anchor_roundtrip_single_batch() -> None:
    model = _build_updated_anchor()
    torch.manual_seed(20260512)
    x = torch.randn(1, 3, 64, 64)
    payload = encode_cheng2020(model, x)
    assert payload.magic == MAGIC_CHENG2020
    assert payload.version == CHENG2020_VERSION
    assert payload.variant == VARIANT_ANCHOR
    assert payload.n_batch == 1
    assert len(payload.strings) == 2  # (y, z)
    assert len(payload.strings[0]) == 1
    assert len(payload.strings[1]) == 1

    blob = serialize_cheng2020(payload)
    parsed = deserialize_cheng2020(blob)
    assert parsed == payload

    x_hat = decode_cheng2020(model, parsed)
    assert x_hat.shape == x.shape


def test_cheng2020_anchor_roundtrip_multi_batch() -> None:
    model = _build_updated_anchor()
    torch.manual_seed(20260512)
    x = torch.randn(2, 3, 64, 64)
    payload = encode_cheng2020(model, x)
    assert payload.n_batch == 2
    assert len(payload.strings[0]) == 2
    assert len(payload.strings[1]) == 2

    blob = serialize_cheng2020(payload)
    parsed = deserialize_cheng2020(blob)
    assert parsed == payload
    x_hat = decode_cheng2020(model, parsed)
    assert x_hat.shape == (2, 3, 64, 64)


def test_cheng2020_serialize_is_deterministic() -> None:
    model = _build_updated_anchor()
    torch.manual_seed(20260512)
    x = torch.randn(1, 3, 64, 64)
    payload_a = encode_cheng2020(model, x)
    payload_b = encode_cheng2020(model, x)
    assert serialize_cheng2020(payload_a) == serialize_cheng2020(payload_b)


# ── Variant-mismatch safety ─────────────────────────────────────────────────


def test_cheng2020_decode_refuses_variant_mismatch() -> None:
    """A payload tagged ANCHOR must not decode through an ATTENTION model."""
    model_anchor = _build_updated_anchor()
    torch.manual_seed(20260512)
    x = torch.randn(1, 3, 64, 64)
    payload = encode_cheng2020(model_anchor, x)
    # Hand-mutate to ATTENTION tag.
    mismatched = Cheng2020Payload(
        magic=payload.magic,
        version=payload.version,
        variant=VARIANT_ATTENTION,
        n_batch=payload.n_batch,
        shape=payload.shape,
        strings=payload.strings,
    )
    with pytest.raises(Cheng2020Error, match="variant mismatch"):
        decode_cheng2020(model_anchor, mismatched)


# ── Failure modes ───────────────────────────────────────────────────────────


def test_cheng2020_rejects_un_updated_model() -> None:
    model = Cheng2020Anchor(N=64)  # NO update() call
    model.eval()
    torch.manual_seed(20260512)
    x = torch.randn(1, 3, 64, 64)
    with pytest.raises(Cheng2020Error, match="no entropy-bottleneck CDF tables"):
        encode_cheng2020(model, x)


def test_cheng2020_rejects_unsupported_model_class() -> None:
    class FakeCheng(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.entropy_bottleneck = torch.nn.Module()
            self.gaussian_conditional = torch.nn.Module()
            self.entropy_bottleneck._quantized_cdf = torch.zeros(1, 1, dtype=torch.int32)
            self.gaussian_conditional._quantized_cdf = torch.zeros(1, 1, dtype=torch.int32)

    fake = FakeCheng()
    fake.eval()
    torch.manual_seed(20260512)
    x = torch.randn(1, 3, 64, 64)
    with pytest.raises(Cheng2020Error, match="unsupported model class"):
        encode_cheng2020(fake, x)


def test_cheng2020_rejects_non_tensor_input() -> None:
    model = _build_updated_anchor()
    with pytest.raises(Cheng2020Error, match="must be a torch.Tensor"):
        encode_cheng2020(model, [1, 2, 3])  # type: ignore[arg-type]


def test_cheng2020_rejects_3d_input() -> None:
    model = _build_updated_anchor()
    x = torch.randn(3, 64, 64)
    with pytest.raises(Cheng2020Error, match="must be 4-D"):
        encode_cheng2020(model, x)


def test_cheng2020_payload_validates_magic() -> None:
    with pytest.raises(Cheng2020Error, match="magic must be"):
        Cheng2020Payload(
            magic=b"BAD!",
            version=CHENG2020_VERSION,
            variant=VARIANT_ANCHOR,
            n_batch=1,
            shape=(1, 1),
            strings=((b"",), (b"",)),
        )


def test_cheng2020_payload_validates_version() -> None:
    with pytest.raises(Cheng2020Error, match="unsupported version"):
        Cheng2020Payload(
            magic=MAGIC_CHENG2020,
            version=0xFF,
            variant=VARIANT_ANCHOR,
            n_batch=1,
            shape=(1, 1),
            strings=((b"",), (b"",)),
        )


def test_cheng2020_payload_validates_unknown_variant() -> None:
    with pytest.raises(Cheng2020Error, match="unknown variant"):
        Cheng2020Payload(
            magic=MAGIC_CHENG2020,
            version=CHENG2020_VERSION,
            variant=0xFE,
            n_batch=1,
            shape=(1, 1),
            strings=((b"",), (b"",)),
        )


def test_cheng2020_payload_validates_substream_count() -> None:
    with pytest.raises(Cheng2020Error, match="2-tuple"):
        Cheng2020Payload(
            magic=MAGIC_CHENG2020,
            version=CHENG2020_VERSION,
            variant=VARIANT_ANCHOR,
            n_batch=1,
            shape=(1, 1),
            strings=((b"",), (b"",), (b"",)),  # type: ignore[arg-type]
        )


def test_cheng2020_payload_validates_per_batch_substream_count() -> None:
    with pytest.raises(Cheng2020Error, match="entries; expected 2"):
        Cheng2020Payload(
            magic=MAGIC_CHENG2020,
            version=CHENG2020_VERSION,
            variant=VARIANT_ANCHOR,
            n_batch=2,
            shape=(1, 1),
            strings=((b"a",), (b"b",)),
        )


def test_cheng2020_deserialize_rejects_short_blob() -> None:
    with pytest.raises(Cheng2020Error, match="too short"):
        deserialize_cheng2020(b"\x00" * 3)


def test_cheng2020_deserialize_rejects_bad_magic() -> None:
    blob = b"FAKE" + b"\x01\x01" + b"\x01\x00\x00\x00" + b"\x01\x00\x00\x00" + b"\x01\x00\x00\x00" + b"\x02"
    with pytest.raises(Cheng2020Error, match="bad magic"):
        deserialize_cheng2020(blob)


def test_cheng2020_deserialize_rejects_truncated_string_body() -> None:
    blob = (
        MAGIC_CHENG2020
        + b"\x01"
        + b"\x01"
        + b"\x01\x00\x00\x00"
        + b"\x01\x00\x00\x00"
        + b"\x01\x00\x00\x00"
        + b"\x02"
        + b"\x64\x00\x00\x00"  # 100-byte length prefix
    )
    with pytest.raises(Cheng2020Error, match="truncated"):
        deserialize_cheng2020(blob)


def test_cheng2020_deserialize_rejects_bad_substream_count_header() -> None:
    blob = (
        MAGIC_CHENG2020
        + b"\x01\x01"
        + b"\x01\x00\x00\x00"
        + b"\x01\x00\x00\x00"
        + b"\x01\x00\x00\x00"
        + b"\x03"
    )
    with pytest.raises(Cheng2020Error, match="n_substreams must be 2"):
        deserialize_cheng2020(blob)


# ── Golden vector ───────────────────────────────────────────────────────────


class TestCheng2020GoldenVector:
    """Pin a deterministic ``encode -> serialize`` SHA against a known seed.

    Recipe (Cheng2020Anchor — primary golden vector):
        torch.manual_seed(20260512)
        model = Cheng2020Anchor(N=64)
        model.update(force=True); model.eval()
        torch.manual_seed(20260512)
        x = torch.randn(1, 3, 64, 64)
        payload = encode_cheng2020(model, x)
        blob = serialize_cheng2020(payload)
    """

    def test_cheng2020_golden_vector(self) -> None:
        model = _build_updated_anchor(n=64)
        torch.manual_seed(20260512)
        x = torch.randn(1, 3, 64, 64)
        payload = encode_cheng2020(model, x)
        blob = serialize_cheng2020(payload)
        digest = hashlib.sha256(blob).hexdigest()
        golden = GOLDEN_DIR / "compressai_cheng2020_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "CompressAI Cheng-2020 wire SHA changed; "
                "delete + regenerate vector if intentional"
            )
            assert data["wire_len"] == len(blob)
            assert data["n_batch"] == payload.n_batch
            assert data["shape"] == list(payload.shape)
            assert data["variant"] == payload.variant
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(
                    {
                        "n_batch": payload.n_batch,
                        "schema": "compressai_cheng2020.v1",
                        "shape": list(payload.shape),
                        "sha256": digest,
                        "string_lens_y": [len(s) for s in payload.strings[0]],
                        "string_lens_z": [len(s) for s in payload.strings[1]],
                        "variant": payload.variant,
                        "wire_len": len(blob),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
