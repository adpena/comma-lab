"""Magic-codec inflate dispatcher (research adapter, ≤200 LOC).

Reads a ``<base>.bin`` archive member that may be (a) a raw PR106 r2 binary
or (b) a magic-codec envelope wrapping a PR106 r2 binary. Dispatches by the
4-byte envelope magic (``MAGC``) and falls through to PR106's canonical
inflate decoder for the actual scene reconstruction.

This file is INTENTIONALLY narrow and deliberately **not** a promotion-grade
contest runtime yet. It is a provenance-preserving adapter for magic-codec
research artifacts. Non-raw envelopes still require the repository ``tac``
package for primitive decoding (lazily-imported deeper inside the
``_decode_*_to_bytes`` helpers), so outputs produced through this wrapper
must keep ``ready_for_exact_eval_dispatch=False`` until a byte-closed
runtime packet copies or vendors the exact primitive decoders AND vendors
the sibling ``pr106_latent_sidecar_r2`` runtime in the same packet AND
proves full-frame inflate parity against the source runtime.

Per Catalog #295 (NSCS06 v5 bug-class anchor; commit ``0b50ceceb``): a
submission ``inflate.py`` MUST be self-contained on a clean Modal worker.
This file currently uses BOTH (a) lazy ``from tac.packet_compiler.*``
imports inside the SRL1/SAC1/centered-delta decoders AND (b) sibling
submission lookup via ``_delegate_to_pr106``. Both are intentional research
shortcuts gated by the docstring claim that this is NOT a promotion-grade
runtime; per audit Priority 4
(.omx/research/submission_inflate_pythonpath_shim_audit_20260516.md) the
acceptance path is the same-line ``# SUBMISSION_PYTHONPATH_SHIM_OK``
waiver applied at the function-scoped imports + a clear dispatch contract
that the magic-codec envelope variants require the sibling submission and
the operator's working tree to be present, which is acceptable for the LAB
dispatch flow (where both are mounted) and NOT acceptable for the
OSS-release path until the helpers are vendored.

Design scope:

* the magic-codec dispatch is ~80 LOC;
* the PR106 inflate decoder is delegated via stdlib ``importlib`` to keep
  the magic-codec runtime under the HNeRV parity lesson 4 budget;
* no scorer load, no MPS, no torch outside the delegated PR106 path.

CLAUDE.md compliance:

* strict-scorer-rule: no scorer-network load at inflate time;
* eval_roundtrip-irrelevant: this is the inflate path, not training;
* ≤200 LOC, research-runtime deps are numpy + repo tac + sibling PR106 runtime;
* deterministic-bytes: identical archive input → identical RGB output.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

# Envelope constants must mirror tac.packet_compiler.magic_codec exactly so
# the inflate runtime stays decoupled from the build-time tac install.
_ENVELOPE_MAGIC = b"MAGC"
_ENVELOPE_HEADER_LEN = 10  # 4 magic + 1 pid + 1 ver + 4 inner_len

# Primitive ids (0xF0-0xFF reserved namespace per CLAUDE.md format_id
# discipline; collision-checked by tests/test_packet_compiler_magic_codec.py).
_PRIMITIVE_RLE_OF_ZEROS = 0xF0
_PRIMITIVE_ARITHMETIC_COEFFICIENTS = 0xF1
_PRIMITIVE_CENTERED_DELTA_UINT8 = 0xF2
_PRIMITIVE_DELTA_VARINT_POSE = 0xF3
_PRIMITIVE_CATEGORICAL_STREAM = 0xF4
_PRIMITIVE_LOWPASS_LUMA_RESIDUAL = 0xF5


def _is_magic_codec_envelope(blob: bytes) -> bool:
    return len(blob) >= _ENVELOPE_HEADER_LEN and blob[:4] == _ENVELOPE_MAGIC


def _parse_envelope(blob: bytes) -> tuple[int, int, int]:
    """Return (primitive_id, version, inner_byte_count)."""
    if not _is_magic_codec_envelope(blob):
        raise ValueError(
            f"not a magic-codec envelope: header={blob[:4]!r}"
        )
    primitive_id = blob[4]
    version = blob[5]
    if version != 1:
        raise ValueError(
            f"unsupported magic-codec envelope version {version}"
        )
    (inner_len,) = struct.unpack_from("<I", blob, 6)
    if _ENVELOPE_HEADER_LEN + inner_len != len(blob):
        raise ValueError(
            f"envelope length mismatch: header+inner="
            f"{_ENVELOPE_HEADER_LEN + inner_len} != total={len(blob)}"
        )
    return int(primitive_id), int(version), int(inner_len)


def _decode_envelope_to_inner_bytes(blob: bytes) -> bytes:
    """Strip envelope + return dense byte stream (PR106 r2 expected form)."""
    primitive_id, _version, _inner_len = _parse_envelope(blob)
    inner = blob[_ENVELOPE_HEADER_LEN:]
    if primitive_id == _PRIMITIVE_RLE_OF_ZEROS:
        return _decode_rle_to_bytes(inner)
    if primitive_id == _PRIMITIVE_ARITHMETIC_COEFFICIENTS:
        return _decode_ac_to_bytes(inner)
    if primitive_id == _PRIMITIVE_CENTERED_DELTA_UINT8:
        return _decode_centered_delta_to_bytes(inner)
    raise ValueError(
        f"unsupported magic-codec primitive_id 0x{primitive_id:02X} for "
        "magic_codec_pr106_r2 (only RLE / AC / centered-delta produce "
        "dense byte streams)"
    )


def _decode_rle_to_bytes(inner: bytes) -> bytes:
    """Decode an SRL1 (sparse RLE) stream back to dense int8 bytes."""
    if inner[:4] != b"SRL1":
        raise ValueError(f"expected SRL1 magic; got {inner[:4]!r}")
    _, total_length, n_nonzero, dtype_code = struct.unpack_from("<4sIIB", inner, 0)
    pos = 13
    dtype = (np.int8, np.int16, np.int32)[int(dtype_code)]
    idx_len = 4 * n_nonzero
    val_len = np.dtype(dtype).itemsize * n_nonzero
    indices = np.frombuffer(inner, dtype="<u4", count=n_nonzero, offset=pos)
    pos += idx_len
    values = np.frombuffer(inner, dtype=dtype, count=n_nonzero, offset=pos)
    out = np.zeros(int(total_length), dtype=dtype)
    if n_nonzero > 0:
        out[indices] = values
    return out.tobytes()


def _decode_ac_to_bytes(inner: bytes) -> bytes:
    """Decode an SAC1 (arithmetic-coded) stream back to dense int8 bytes."""
    if inner[:4] != b"SAC1":
        raise ValueError(f"expected SAC1 magic; got {inner[:4]!r}")
    # Delegate to constriction-backed deserializer in the tac runtime.
    from tac.packet_compiler.sparse_packet_ir import (  # SUBMISSION_PYTHONPATH_SHIM_OK:magic-codec-pr106-r2-is-research-only-non-promotable-adapter-per-module-docstring-requires-sibling-pr106-latent-sidecar-r2-and-tac-packet-compiler-sparse-packet-ir-both-mounted-via-lab-dispatch-flow-not-oss-release-self-contained-per-catalog-295-audit-priority-4
        decode_arithmetic_coefficients,
        deserialize_arithmetic_coefficients,
    )

    stream = deserialize_arithmetic_coefficients(inner)
    values = decode_arithmetic_coefficients(stream)
    # PR106 r2 archive bytes are uint8; AC may have used int32 internally —
    # cast back to int8 (the original quantisation level).
    return values.astype(np.int8).tobytes()


def _decode_centered_delta_to_bytes(inner: bytes) -> bytes:
    """Decode the centered-delta envelope back to dense float32 → uint8 bytes."""
    n_rows, n_dims = struct.unpack_from("<II", inner, 0)
    pos = 8
    (lzma_len,) = struct.unpack_from("<I", inner, pos)
    pos += 4
    lzma_bytes = inner[pos : pos + lzma_len]
    from tac.packet_compiler.pr101_sidecar_grammar import decode_centered_delta_uint8  # SUBMISSION_PYTHONPATH_SHIM_OK:magic-codec-pr106-r2-is-research-only-non-promotable-adapter-per-module-docstring-requires-tac-packet-compiler-pr101-sidecar-grammar-mounted-via-lab-dispatch-flow-not-oss-release-self-contained-per-catalog-295-audit-priority-4

    values = decode_centered_delta_uint8(
        lzma_bytes, n_pairs=int(n_rows), n_dims=int(n_dims)
    )
    return values.astype(np.float32).tobytes()


def _delegate_to_pr106(src: Path, dst: Path) -> None:
    """Run the canonical PR106 r2 inflate decoder on (src, dst)."""
    pr106_root = src.parent.parent / "pr106_latent_sidecar_r2"
    if not pr106_root.exists():
        # Repo-relative fallback for in-place test layouts.
        pr106_root = Path(__file__).resolve().parent.parent / (
            "pr106_latent_sidecar_r2"
        )
    inflate_py = pr106_root / "inflate.py"
    if not inflate_py.exists():
        raise FileNotFoundError(
            f"PR106 r2 inflate.py not found at {inflate_py}; "
            "magic_codec_pr106_r2 requires the sibling submission"
        )
    spec = importlib.util.spec_from_file_location("pr106_r2_inflate", inflate_py)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "main"):
        mod.main([str(src), str(dst)])
    elif hasattr(mod, "inflate"):
        mod.inflate(str(src), str(dst))
    else:
        raise AttributeError(
            "PR106 r2 inflate module exposes neither main(argv) nor inflate(src, dst)"
        )


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) < 2:
        print("usage: inflate.py <src.bin> <dst.raw>", file=sys.stderr)
        sys.exit(2)
    src = Path(args[0])
    dst = Path(args[1])
    raw = src.read_bytes()
    if _is_magic_codec_envelope(raw):
        inner_bytes = _decode_envelope_to_inner_bytes(raw)
        # Write the unwrapped bytes to a temp sibling so PR106's inflate
        # runtime sees a normal .bin file. Use the dst directory so we
        # remain within the operator-supplied output tree (no /tmp).
        unwrapped = dst.parent / (src.stem + ".unwrapped.bin")
        unwrapped.write_bytes(inner_bytes)
        _delegate_to_pr106(unwrapped, dst)
        unwrapped.unlink()
    else:
        _delegate_to_pr106(src, dst)


if __name__ == "__main__":
    main()
