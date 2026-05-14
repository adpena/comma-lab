# SPDX-License-Identifier: MIT
"""Roundtrip + golden-vector tests for the CompressAI FactorizedPrior adapter.

Sister to ``src/tac/packet_compiler/factorized_prior.py``. Catalog #91
(``check_encoder_decoder_dequantization_roundtrip_tested``) accepts this
sibling test file as the paired roundtrip proof for the adapter.

ENCODE_INFLATE_ROUNDTRIP tag below makes the linkage explicit to the
scanner.
"""

# ENCODE_INFLATE_ROUNDTRIP — pytest covers encode->serialize->deserialize->
# decode end-to-end through compressai.models.FactorizedPrior.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
compressai_module = pytest.importorskip("compressai")
from compressai.models import FactorizedPrior  # noqa: E402

from tac.packet_compiler import (  # noqa: E402
    FACTORIZED_PRIOR_VERSION,
    MAGIC_FACTORIZED_PRIOR,
    FactorizedPriorError,
    FactorizedPriorPayload,
    decode_factorized_prior,
    deserialize_factorized_prior,
    encode_factorized_prior,
    serialize_factorized_prior,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent / "packet_compiler" / "golden_vectors"
)


def _build_updated_model(*, n: int = 64, m: int = 96) -> FactorizedPrior:
    torch.manual_seed(20260512)
    model = FactorizedPrior(N=n, M=m)
    model.update(force=True)
    model.eval()
    return model


# ── Round-trip / behavior ───────────────────────────────────────────────────


def test_factorized_prior_payload_is_frozen() -> None:
    payload = FactorizedPriorPayload(
        magic=MAGIC_FACTORIZED_PRIOR,
        version=FACTORIZED_PRIOR_VERSION,
        n_batch=1,
        shape=(4, 4),
        strings=(b"\x00",),
    )
    with pytest.raises((AttributeError, TypeError)):
        payload.version = 99  # type: ignore[misc]


def test_factorized_prior_roundtrip_single_batch() -> None:
    model = _build_updated_model()
    torch.manual_seed(20260512)
    x = torch.randn(1, 3, 64, 64)
    payload = encode_factorized_prior(model, x)
    assert payload.magic == MAGIC_FACTORIZED_PRIOR
    assert payload.version == FACTORIZED_PRIOR_VERSION
    assert payload.n_batch == 1
    assert payload.shape == (4, 4)
    assert len(payload.strings) == 1
    assert isinstance(payload.strings[0], bytes)
    assert len(payload.strings[0]) > 0

    blob = serialize_factorized_prior(payload)
    parsed = deserialize_factorized_prior(blob)
    assert parsed == payload

    x_hat = decode_factorized_prior(model, parsed)
    assert x_hat.shape == x.shape


def test_factorized_prior_roundtrip_multi_batch() -> None:
    model = _build_updated_model()
    torch.manual_seed(20260512)
    x = torch.randn(3, 3, 64, 64)
    payload = encode_factorized_prior(model, x)
    assert payload.n_batch == 3
    assert len(payload.strings) == 3
    blob = serialize_factorized_prior(payload)
    parsed = deserialize_factorized_prior(blob)
    assert parsed == payload
    x_hat = decode_factorized_prior(model, parsed)
    assert x_hat.shape == (3, 3, 64, 64)


def test_factorized_prior_serialize_is_deterministic() -> None:
    model = _build_updated_model()
    torch.manual_seed(20260512)
    x = torch.randn(1, 3, 64, 64)
    payload_a = encode_factorized_prior(model, x)
    payload_b = encode_factorized_prior(model, x)
    assert serialize_factorized_prior(payload_a) == serialize_factorized_prior(
        payload_b
    )


# ── Failure modes ───────────────────────────────────────────────────────────


def test_factorized_prior_rejects_un_updated_model() -> None:
    model = FactorizedPrior(N=64, M=96)  # NO update() call
    model.eval()
    torch.manual_seed(20260512)
    x = torch.randn(1, 3, 64, 64)
    with pytest.raises(FactorizedPriorError, match="no CDF tables"):
        encode_factorized_prior(model, x)


def test_factorized_prior_rejects_non_tensor_input() -> None:
    model = _build_updated_model()
    with pytest.raises(FactorizedPriorError, match="must be a torch.Tensor"):
        encode_factorized_prior(model, [1, 2, 3])  # type: ignore[arg-type]


def test_factorized_prior_rejects_3d_input() -> None:
    model = _build_updated_model()
    x = torch.randn(3, 64, 64)
    with pytest.raises(FactorizedPriorError, match="must be 4-D"):
        encode_factorized_prior(model, x)


def test_factorized_prior_rejects_non_multiple_of_16_input() -> None:
    model = _build_updated_model()
    x = torch.randn(1, 3, 60, 64)  # 60 not divisible by 16
    with pytest.raises(FactorizedPriorError, match="divisible by 16"):
        encode_factorized_prior(model, x)


def test_factorized_prior_payload_validates_magic() -> None:
    with pytest.raises(FactorizedPriorError, match="magic must be"):
        FactorizedPriorPayload(
            magic=b"BAD!",
            version=FACTORIZED_PRIOR_VERSION,
            n_batch=1,
            shape=(4, 4),
            strings=(b"",),
        )


def test_factorized_prior_payload_validates_version() -> None:
    with pytest.raises(FactorizedPriorError, match="unsupported version"):
        FactorizedPriorPayload(
            magic=MAGIC_FACTORIZED_PRIOR,
            version=0xFF,
            n_batch=1,
            shape=(4, 4),
            strings=(b"",),
        )


def test_factorized_prior_payload_validates_n_batch_matches_strings() -> None:
    with pytest.raises(FactorizedPriorError, match="!= len\\(strings\\)"):
        FactorizedPriorPayload(
            magic=MAGIC_FACTORIZED_PRIOR,
            version=FACTORIZED_PRIOR_VERSION,
            n_batch=2,
            shape=(4, 4),
            strings=(b"a",),
        )


def test_factorized_prior_payload_validates_shape_dims() -> None:
    with pytest.raises(FactorizedPriorError, match="shape must be"):
        FactorizedPriorPayload(
            magic=MAGIC_FACTORIZED_PRIOR,
            version=FACTORIZED_PRIOR_VERSION,
            n_batch=1,
            shape=(4,),  # type: ignore[arg-type]
            strings=(b"",),
        )


def test_factorized_prior_payload_validates_positive_shape_dim() -> None:
    with pytest.raises(FactorizedPriorError, match="positive int"):
        FactorizedPriorPayload(
            magic=MAGIC_FACTORIZED_PRIOR,
            version=FACTORIZED_PRIOR_VERSION,
            n_batch=1,
            shape=(0, 4),
            strings=(b"",),
        )


def test_factorized_prior_deserialize_rejects_short_blob() -> None:
    with pytest.raises(FactorizedPriorError, match="too short"):
        deserialize_factorized_prior(b"\x00" * 3)


def test_factorized_prior_deserialize_rejects_bad_magic() -> None:
    blob = b"FAKE" + b"\x01" + b"\x01\x00\x00\x00" + b"\x04\x00\x00\x00" + b"\x04\x00\x00\x00" + b"\x00\x00\x00\x00"
    with pytest.raises(FactorizedPriorError, match="bad magic"):
        deserialize_factorized_prior(blob)


def test_factorized_prior_deserialize_rejects_unsupported_version() -> None:
    blob = MAGIC_FACTORIZED_PRIOR + b"\xff" + b"\x01\x00\x00\x00" + b"\x04\x00\x00\x00" + b"\x04\x00\x00\x00" + b"\x00\x00\x00\x00"
    with pytest.raises(FactorizedPriorError, match="unsupported wire version"):
        deserialize_factorized_prior(blob)


def test_factorized_prior_deserialize_rejects_truncated_string_body() -> None:
    # Header claims 100 bytes of string body but provides 0.
    blob = (
        MAGIC_FACTORIZED_PRIOR
        + b"\x01"
        + b"\x01\x00\x00\x00"
        + b"\x04\x00\x00\x00"
        + b"\x04\x00\x00\x00"
        + b"\x64\x00\x00\x00"  # 100-byte length prefix
        # no body
    )
    with pytest.raises(FactorizedPriorError, match="truncated"):
        deserialize_factorized_prior(blob)


def test_factorized_prior_deserialize_rejects_trailing_bytes() -> None:
    blob = (
        MAGIC_FACTORIZED_PRIOR
        + b"\x01"
        + b"\x01\x00\x00\x00"
        + b"\x04\x00\x00\x00"
        + b"\x04\x00\x00\x00"
        + b"\x00\x00\x00\x00"  # zero-length string body
        + b"GARBAGE"  # trailing
    )
    with pytest.raises(FactorizedPriorError, match="trailing"):
        deserialize_factorized_prior(blob)


# ── Golden vector ───────────────────────────────────────────────────────────


class TestFactorizedPriorGoldenVector:
    """Pin a deterministic ``encode -> serialize`` SHA-256 against a known
    seed + model config so the Rust parity stub can verify byte-for-byte
    parity once a native port lands.

    Recipe:
        torch.manual_seed(20260512)
        model = FactorizedPrior(N=64, M=96)
        model.update(force=True)
        model.eval()
        torch.manual_seed(20260512)
        x = torch.randn(1, 3, 64, 64)
        payload = encode_factorized_prior(model, x)
        blob = serialize_factorized_prior(payload)
        sha256 = hashlib.sha256(blob).hexdigest()
    """

    def test_factorized_prior_golden_vector(self) -> None:
        model = _build_updated_model(n=64, m=96)
        torch.manual_seed(20260512)
        x = torch.randn(1, 3, 64, 64)
        payload = encode_factorized_prior(model, x)
        blob = serialize_factorized_prior(payload)
        digest = hashlib.sha256(blob).hexdigest()
        golden = GOLDEN_DIR / "compressai_factorized_prior_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "CompressAI FactorizedPrior wire SHA changed; "
                "delete + regenerate vector if intentional"
            )
            assert data["wire_len"] == len(blob), (
                "wire length mismatch — schema drift?"
            )
            assert data["n_batch"] == payload.n_batch
            assert data["shape"] == list(payload.shape)
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(
                    {
                        "n_batch": payload.n_batch,
                        "schema": "compressai_factorized_prior.v1",
                        "shape": list(payload.shape),
                        "sha256": digest,
                        "string_lens": [len(s) for s in payload.strings],
                        "wire_len": len(blob),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
