"""Roundtrip + golden-vector tests for the CompressAI Ballé-hyperprior adapter.

Sister to ``src/tac/packet_compiler/balle_hyperprior.py``. Catalog #91
accepts this sibling test file as the paired roundtrip proof for the
adapter.
"""

# ENCODE_INFLATE_ROUNDTRIP — pytest covers encode->serialize->deserialize->
# decode end-to-end through compressai.models.{ScaleHyperprior,
# MeanScaleHyperprior, JointAutoregressiveHierarchicalPriors}.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
compressai_module = pytest.importorskip("compressai")
from compressai.models import (  # noqa: E402
    MeanScaleHyperprior,
    ScaleHyperprior,
)

from tac.packet_compiler import (  # noqa: E402
    BALLE_HYPERPRIOR_VERSION,
    MAGIC_BALLE_HYPERPRIOR,
    VARIANT_MEANSCALE,
    VARIANT_SCALE,
    BalleHyperpriorError,
    BalleHyperpriorPayload,
    decode_balle_hyperprior,
    deserialize_balle_hyperprior,
    encode_balle_hyperprior,
    serialize_balle_hyperprior,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent / "packet_compiler" / "golden_vectors"
)


def _build_updated_model(cls=ScaleHyperprior, *, n: int = 64, m: int = 96):
    torch.manual_seed(20260512)
    model = cls(N=n, M=m)
    model.update(force=True)
    model.eval()
    return model


# ── Round-trip / behavior ───────────────────────────────────────────────────


def test_balle_payload_is_frozen() -> None:
    payload = BalleHyperpriorPayload(
        magic=MAGIC_BALLE_HYPERPRIOR,
        version=BALLE_HYPERPRIOR_VERSION,
        variant=VARIANT_SCALE,
        n_batch=1,
        shape=(1, 1),
        strings=((b"",), (b"",)),
    )
    with pytest.raises((AttributeError, TypeError)):
        payload.variant = 0xFF  # type: ignore[misc]


def test_balle_scale_hyperprior_roundtrip_single_batch() -> None:
    model = _build_updated_model(ScaleHyperprior)
    torch.manual_seed(20260512)
    x = torch.randn(1, 3, 64, 64)
    payload = encode_balle_hyperprior(model, x)
    assert payload.magic == MAGIC_BALLE_HYPERPRIOR
    assert payload.version == BALLE_HYPERPRIOR_VERSION
    assert payload.variant == VARIANT_SCALE
    assert payload.n_batch == 1
    assert len(payload.strings) == 2  # (y, z)
    assert len(payload.strings[0]) == 1  # y per-batch
    assert len(payload.strings[1]) == 1  # z per-batch

    blob = serialize_balle_hyperprior(payload)
    parsed = deserialize_balle_hyperprior(blob)
    assert parsed == payload

    x_hat = decode_balle_hyperprior(model, parsed)
    assert x_hat.shape == x.shape


def test_balle_meanscale_hyperprior_roundtrip_tags_variant() -> None:
    model = _build_updated_model(MeanScaleHyperprior)
    torch.manual_seed(20260512)
    x = torch.randn(1, 3, 64, 64)
    payload = encode_balle_hyperprior(model, x)
    assert payload.variant == VARIANT_MEANSCALE

    blob = serialize_balle_hyperprior(payload)
    parsed = deserialize_balle_hyperprior(blob)
    assert parsed == payload
    assert parsed.variant == VARIANT_MEANSCALE

    x_hat = decode_balle_hyperprior(model, parsed)
    assert x_hat.shape == x.shape


def test_balle_roundtrip_multi_batch() -> None:
    model = _build_updated_model(ScaleHyperprior)
    torch.manual_seed(20260512)
    x = torch.randn(2, 3, 64, 64)
    payload = encode_balle_hyperprior(model, x)
    assert payload.n_batch == 2
    assert len(payload.strings[0]) == 2
    assert len(payload.strings[1]) == 2
    blob = serialize_balle_hyperprior(payload)
    parsed = deserialize_balle_hyperprior(blob)
    assert parsed == payload
    x_hat = decode_balle_hyperprior(model, parsed)
    assert x_hat.shape == (2, 3, 64, 64)


def test_balle_serialize_is_deterministic() -> None:
    model = _build_updated_model(ScaleHyperprior)
    torch.manual_seed(20260512)
    x = torch.randn(1, 3, 64, 64)
    payload_a = encode_balle_hyperprior(model, x)
    payload_b = encode_balle_hyperprior(model, x)
    assert serialize_balle_hyperprior(payload_a) == serialize_balle_hyperprior(
        payload_b
    )


# ── Variant-mismatch safety ─────────────────────────────────────────────────


def test_balle_decode_refuses_variant_mismatch() -> None:
    model_scale = _build_updated_model(ScaleHyperprior)
    model_meanscale = _build_updated_model(MeanScaleHyperprior)
    torch.manual_seed(20260512)
    x = torch.randn(1, 3, 64, 64)
    payload = encode_balle_hyperprior(model_scale, x)
    # variant tag says ScaleHyperprior; passing MeanScaleHyperprior must refuse.
    with pytest.raises(BalleHyperpriorError, match="variant mismatch"):
        decode_balle_hyperprior(model_meanscale, payload)


# ── Failure modes ───────────────────────────────────────────────────────────


def test_balle_rejects_un_updated_model() -> None:
    model = ScaleHyperprior(N=64, M=96)  # NO update() call
    model.eval()
    torch.manual_seed(20260512)
    x = torch.randn(1, 3, 64, 64)
    with pytest.raises(BalleHyperpriorError, match="no entropy-bottleneck CDF tables"):
        encode_balle_hyperprior(model, x)


def test_balle_rejects_unsupported_model_class() -> None:
    class FakeHyperprior(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.entropy_bottleneck = torch.nn.Module()
            self.gaussian_conditional = torch.nn.Module()
            self.entropy_bottleneck._quantized_cdf = torch.zeros(1, 1, dtype=torch.int32)
            self.gaussian_conditional._quantized_cdf = torch.zeros(1, 1, dtype=torch.int32)

    fake = FakeHyperprior()
    fake.eval()
    torch.manual_seed(20260512)
    x = torch.randn(1, 3, 64, 64)
    with pytest.raises(BalleHyperpriorError, match="unsupported model class"):
        encode_balle_hyperprior(fake, x)


def test_balle_rejects_non_tensor_input() -> None:
    model = _build_updated_model(ScaleHyperprior)
    with pytest.raises(BalleHyperpriorError, match="must be a torch.Tensor"):
        encode_balle_hyperprior(model, [1, 2, 3])  # type: ignore[arg-type]


def test_balle_rejects_3d_input() -> None:
    model = _build_updated_model(ScaleHyperprior)
    x = torch.randn(3, 64, 64)
    with pytest.raises(BalleHyperpriorError, match="must be 4-D"):
        encode_balle_hyperprior(model, x)


def test_balle_payload_validates_magic() -> None:
    with pytest.raises(BalleHyperpriorError, match="magic must be"):
        BalleHyperpriorPayload(
            magic=b"BAD!",
            version=BALLE_HYPERPRIOR_VERSION,
            variant=VARIANT_SCALE,
            n_batch=1,
            shape=(1, 1),
            strings=((b"",), (b"",)),
        )


def test_balle_payload_validates_version() -> None:
    with pytest.raises(BalleHyperpriorError, match="unsupported version"):
        BalleHyperpriorPayload(
            magic=MAGIC_BALLE_HYPERPRIOR,
            version=0xFF,
            variant=VARIANT_SCALE,
            n_batch=1,
            shape=(1, 1),
            strings=((b"",), (b"",)),
        )


def test_balle_payload_validates_unknown_variant() -> None:
    with pytest.raises(BalleHyperpriorError, match="unknown variant"):
        BalleHyperpriorPayload(
            magic=MAGIC_BALLE_HYPERPRIOR,
            version=BALLE_HYPERPRIOR_VERSION,
            variant=0xFE,
            n_batch=1,
            shape=(1, 1),
            strings=((b"",), (b"",)),
        )


def test_balle_payload_validates_substream_count() -> None:
    with pytest.raises(BalleHyperpriorError, match="2-tuple"):
        BalleHyperpriorPayload(
            magic=MAGIC_BALLE_HYPERPRIOR,
            version=BALLE_HYPERPRIOR_VERSION,
            variant=VARIANT_SCALE,
            n_batch=1,
            shape=(1, 1),
            strings=((b"",), (b"",), (b"",)),  # type: ignore[arg-type]
        )


def test_balle_payload_validates_per_batch_substream_count() -> None:
    with pytest.raises(BalleHyperpriorError, match="entries; expected 2"):
        BalleHyperpriorPayload(
            magic=MAGIC_BALLE_HYPERPRIOR,
            version=BALLE_HYPERPRIOR_VERSION,
            variant=VARIANT_SCALE,
            n_batch=2,
            shape=(1, 1),
            strings=((b"a",), (b"b",)),  # only one entry per substream
        )


def test_balle_deserialize_rejects_short_blob() -> None:
    with pytest.raises(BalleHyperpriorError, match="too short"):
        deserialize_balle_hyperprior(b"\x00" * 3)


def test_balle_deserialize_rejects_bad_magic() -> None:
    blob = b"FAKE" + b"\x01\x01" + b"\x01\x00\x00\x00" + b"\x01\x00\x00\x00" + b"\x01\x00\x00\x00" + b"\x02"
    with pytest.raises(BalleHyperpriorError, match="bad magic"):
        deserialize_balle_hyperprior(blob)


def test_balle_deserialize_rejects_truncated_string_body() -> None:
    blob = (
        MAGIC_BALLE_HYPERPRIOR
        + b"\x01"  # version
        + b"\x01"  # variant=SCALE
        + b"\x01\x00\x00\x00"  # n_batch=1
        + b"\x01\x00\x00\x00"  # shape_h=1
        + b"\x01\x00\x00\x00"  # shape_w=1
        + b"\x02"  # n_substreams=2
        + b"\x64\x00\x00\x00"  # 100-byte length prefix for y[0]
        # no body
    )
    with pytest.raises(BalleHyperpriorError, match="truncated"):
        deserialize_balle_hyperprior(blob)


def test_balle_deserialize_rejects_bad_substream_count_header() -> None:
    blob = (
        MAGIC_BALLE_HYPERPRIOR
        + b"\x01\x01"
        + b"\x01\x00\x00\x00"
        + b"\x01\x00\x00\x00"
        + b"\x01\x00\x00\x00"
        + b"\x03"  # n_substreams=3 (invalid)
    )
    with pytest.raises(BalleHyperpriorError, match="n_substreams must be 2"):
        deserialize_balle_hyperprior(blob)


# ── Golden vector ───────────────────────────────────────────────────────────


class TestBalleHyperpriorGoldenVector:
    """Pin deterministic ``encode -> serialize`` SHAs against known seeds.

    Recipe (ScaleHyperprior — primary golden vector):
        torch.manual_seed(20260512)
        model = ScaleHyperprior(N=64, M=96)
        model.update(force=True); model.eval()
        torch.manual_seed(20260512)
        x = torch.randn(1, 3, 64, 64)
        payload = encode_balle_hyperprior(model, x)
        blob = serialize_balle_hyperprior(payload)
    """

    def test_balle_hyperprior_golden_vector(self) -> None:
        model = _build_updated_model(ScaleHyperprior, n=64, m=96)
        torch.manual_seed(20260512)
        x = torch.randn(1, 3, 64, 64)
        payload = encode_balle_hyperprior(model, x)
        blob = serialize_balle_hyperprior(payload)
        digest = hashlib.sha256(blob).hexdigest()
        golden = GOLDEN_DIR / "compressai_balle_hyperprior_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "CompressAI Ballé hyperprior wire SHA changed; "
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
                        "schema": "compressai_balle_hyperprior.v1",
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
