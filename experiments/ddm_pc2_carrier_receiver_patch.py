"""KW1 receiver patch: teach a shipped F26 receiver the wide Rice-``k`` block.

The shipped receiver hard-codes the packed CAP1 metadata block at 40 bytes in
three modules:

* ``runtime/rr5_arith_basis.py`` -- ``BASIS_OFFSET`` (= 102 + 40) locates the
  basis payload for the RR5 basis rider;
* ``runtime/dx2_cabac_coefficients.py`` -- ``packed_ks`` reads the twelve Rice
  parameters out of a block it requires to be exactly 40 bytes;
* ``runtime/residual_archive.py`` -- ``_packed_portion`` and
  ``_restore_packed_cap1_metadata`` frame and expand the block.

This module applies the smallest diff that makes all three carry the block
width as a parameter, defaulting to the shipped 40 so **an archive without the
KW1 reserved bit takes exactly the path it takes today, byte for byte**.  Every
patch is an exact string replacement anchored on shipped text and refuses
unless that text appears exactly once, so a receiver generation that has moved
fails closed rather than being patched somewhere unintended.

The KW1 format itself lives in ``tac.kw1_wide_rice_k``, which is copied into the
patched runtime tree so the decoder runs the same bytes the encoder does (the
RR5/DX2 single-source-of-truth precedent).
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KW1_SOURCE = REPO / "src" / "tac" / "kw1_wide_rice_k.py"


class ReceiverPatchError(RuntimeError):
    """A patch anchor is missing, ambiguous, or the tree is not a receiver."""


# --------------------------------------------------------------------------- #
# rr5_arith_basis.py
# --------------------------------------------------------------------------- #
_RR5_SPLIT_OLD = '''def split_carrier_body(body: bytes) -> dict[str, object]:
    """Split a decompressed (CK2-restored) carrier body into its exact fields."""
    if len(body) < BASIS_OFFSET:
        raise RiderError("carrier body is shorter than the packed CAP1 prefix")
    basis_bits = int.from_bytes(body[0:3], "little")
    residual_bits = int.from_bytes(body[3:6], "little")
    if not basis_bits or not residual_bits:
        raise RiderError("carrier bit counts must be nonzero")
    basis_bytes = (basis_bits + 7) // 8
    rice_bytes = (residual_bits + 7) // 8
    packed_portion = BASIS_OFFSET + basis_bytes + rice_bytes
    if len(body) < packed_portion:
        raise RiderError("carrier body is truncated against its own bit counts")
    return {
        "basis_bits": basis_bits,
        "residual_bits": residual_bits,
        "scales": body[BIT_COUNT_BYTES:PACKED_METADATA_OFFSET],
        "metadata": bytearray(body[PACKED_METADATA_OFFSET:BASIS_OFFSET]),
        "basis": body[BASIS_OFFSET : BASIS_OFFSET + basis_bytes],
        "rice": body[BASIS_OFFSET + basis_bytes : packed_portion],
        "body_tail": body[packed_portion:],
    }'''

_RR5_SPLIT_NEW = '''def split_carrier_body(
    body: bytes, *, metadata_bytes: int = PACKED_METADATA_BYTES
) -> dict[str, object]:
    """Split a decompressed (CK2-restored) carrier body into its exact fields.

    DDM_KW1_WIDE_RICE_K_V1: ``metadata_bytes`` is the packed CAP1 metadata block
    width.  It defaults to the shipped 40, so every existing caller and every
    archive without the KW1 reserved bit is byte-identical to today.
    """
    basis_offset = PACKED_METADATA_OFFSET + int(metadata_bytes)
    if len(body) < basis_offset:
        raise RiderError("carrier body is shorter than the packed CAP1 prefix")
    basis_bits = int.from_bytes(body[0:3], "little")
    residual_bits = int.from_bytes(body[3:6], "little")
    if not basis_bits or not residual_bits:
        raise RiderError("carrier bit counts must be nonzero")
    basis_bytes = (basis_bits + 7) // 8
    rice_bytes = (residual_bits + 7) // 8
    packed_portion = basis_offset + basis_bytes + rice_bytes
    if len(body) < packed_portion:
        raise RiderError("carrier body is truncated against its own bit counts")
    return {
        "basis_bits": basis_bits,
        "residual_bits": residual_bits,
        "scales": body[BIT_COUNT_BYTES:PACKED_METADATA_OFFSET],
        "metadata": bytearray(body[PACKED_METADATA_OFFSET:basis_offset]),
        "basis": body[basis_offset : basis_offset + basis_bytes],
        "rice": body[basis_offset + basis_bytes : packed_portion],
        "body_tail": body[packed_portion:],
    }'''

_RR5_ASSEMBLE_OLD = '''    metadata = bytes(fields["metadata"])  # type: ignore[index]
    if len(metadata) != PACKED_METADATA_BYTES:
        raise RiderError("packed metadata must stay 40 bytes")'''

_RR5_ASSEMBLE_NEW = '''    metadata = bytes(fields["metadata"])  # type: ignore[index]
    # DDM_KW1_WIDE_RICE_K_V1: 40 is the shipped block, 43 the wide-k block.
    if len(metadata) not in (PACKED_METADATA_BYTES, KW1_PACKED_METADATA_BYTES):
        raise RiderError("packed metadata must be 40 or 43 bytes")'''

_RR5_APPLY_OLD = '''    fields = split_carrier_body(body)
    lengths = packed_lengths(bytes(fields["metadata"]))  # type: ignore[arg-type]'''

_RR5_APPLY_NEW = '''    fields = split_carrier_body(body, metadata_bytes=metadata_bytes)
    lengths = packed_lengths(bytes(fields["metadata"]))  # type: ignore[arg-type]'''

_RR5_APPLY_SIG_OLD = "def apply_rider_to_carrier_body(body: bytes) -> dict[str, object]:"
_RR5_APPLY_SIG_NEW = (
    "def apply_rider_to_carrier_body(\n"
    "    body: bytes, *, metadata_bytes: int = PACKED_METADATA_BYTES\n"
    ") -> dict[str, object]:"
)

_RR5_RESTORE_SIG_OLD = "def restore_carrier_body(body: bytes) -> bytes:"
_RR5_RESTORE_SIG_NEW = (
    "def restore_carrier_body(\n"
    "    body: bytes, *, metadata_bytes: int = PACKED_METADATA_BYTES\n"
    ") -> bytes:"
)

_RR5_RESTORE_OLD = '''    fields = split_carrier_body(body)
    symbols = decode_basis_arith('''

_RR5_RESTORE_NEW = '''    fields = split_carrier_body(body, metadata_bytes=metadata_bytes)
    symbols = decode_basis_arith('''

_RR5_CONST_OLD = "BASIS_OFFSET = PACKED_METADATA_OFFSET + PACKED_METADATA_BYTES  # 142"
_RR5_CONST_NEW = (
    "BASIS_OFFSET = PACKED_METADATA_OFFSET + PACKED_METADATA_BYTES  # 142\n"
    "# DDM_KW1_WIDE_RICE_K_V1: the wide Rice-k block replaces k_base + the 12x1-bit\n"
    "# field with 12 absolute nibbles, so the block grows 40 -> 43 bytes.\n"
    "KW1_PACKED_METADATA_BYTES = 43"
)

# --------------------------------------------------------------------------- #
# dx2_cabac_coefficients.py
# --------------------------------------------------------------------------- #
_DX2_KS_OLD = '''def packed_ks(metadata: bytes) -> np.ndarray:
    """Read the 12 fixed Rice ``k`` values from the 40-byte packed metadata."""

    if len(metadata) != 40:
        raise CabacCoefficientError("packed CAP1 metadata must be exactly 40 bytes")
    base = metadata[37]
    deltas = unpack_unsigned(metadata[38:40], CARRIER_DIM, 1).astype(np.int64)
    return _validate_ks(base + deltas)'''

_DX2_KS_NEW = '''def packed_ks(metadata: bytes) -> np.ndarray:
    """Read the 12 fixed Rice ``k`` values from the packed metadata block.

    DDM_KW1_WIDE_RICE_K_V1: the form is identified by the block LENGTH -- 40 is
    the shipped ``k_base`` + 12x1-bit field, 43 is the wide 12x4-bit absolute
    field.  A 40-byte block takes exactly the path it takes today.
    """

    if len(metadata) == 40:
        base = metadata[37]
        deltas = unpack_unsigned(metadata[38:40], CARRIER_DIM, 1).astype(np.int64)
        return _validate_ks(base + deltas)
    if len(metadata) == 43:
        from .kw1_wide_rice_k import unpack_ks as _kw1_unpack_ks

        return _validate_ks(np.asarray(_kw1_unpack_ks(metadata), dtype=np.int64))
    raise CabacCoefficientError("packed CAP1 metadata must be 40 or 43 bytes")'''

_DX2_APPLY_SIG_OLD = (
    "def apply_cabac_to_carrier_body(body: bytes) -> dict[str, object]:"
)
_DX2_APPLY_SIG_NEW = (
    "def apply_cabac_to_carrier_body(\n"
    "    body: bytes, *, metadata_bytes: int = 40\n"
    ") -> dict[str, object]:"
)

_DX2_APPLY_OLD = '''    fields = split_carrier_body(body)
    parameters = packed_ks(bytes(fields["metadata"]))
    symbols = rice_decode('''

_DX2_APPLY_NEW = '''    fields = split_carrier_body(body, metadata_bytes=metadata_bytes)
    parameters = packed_ks(bytes(fields["metadata"]))
    symbols = rice_decode('''

_DX2_APPLY_TAIL_OLD = '''    candidate_body = assemble_carrier_body(candidate_fields)
    if restore_carrier_body(candidate_body) != body:'''

_DX2_APPLY_TAIL_NEW = '''    candidate_body = assemble_carrier_body(candidate_fields)
    if restore_carrier_body(candidate_body, metadata_bytes=metadata_bytes) != body:'''

_DX2_RESTORE_SIG_OLD = "def restore_carrier_body(body: bytes) -> bytes:"
_DX2_RESTORE_SIG_NEW = (
    "def restore_carrier_body(body: bytes, *, metadata_bytes: int = 40) -> bytes:"
)

_DX2_RESTORE_OLD = '''    fields = split_carrier_body(body)
    if int(fields["residual_bits"]) % 8:'''

_DX2_RESTORE_NEW = '''    fields = split_carrier_body(body, metadata_bytes=metadata_bytes)
    if int(fields["residual_bits"]) % 8:'''

# --------------------------------------------------------------------------- #
# residual_archive.py
# --------------------------------------------------------------------------- #
_RA_BITS_OLD = "SZ1_RESERVED_KNOWN_BITS = 0x7F"
_RA_BITS_NEW = (
    "# DDM_KW1_WIDE_RICE_K_V1: the packed CAP1 metadata block carries the twelve\n"
    "# Rice parameters as 4-bit absolute nibbles instead of k_base + a 1-bit field,\n"
    "# so the block is 43 bytes instead of 40.  Additive: an archive with this bit\n"
    "# clear takes exactly the path it takes today.\n"
    "KW1_RESERVED_WIDE_RICE_K = 0x80\n"
    "KW1_PACKED_METADATA_BYTES = 43\n"
    "SZ1_RESERVED_KNOWN_BITS = 0xFF"
)

_RA_RESTORE_OLD = '''def _restore_packed_cap1_metadata(packed: bytes) -> bytes:
    if len(packed) < 142:
        raise ResidualArchiveError("packed CAP1 section is truncated")
    factor_base = packed[102]
    factors = factor_base + _unpack_unsigned(packed[103:114], 12, 7)
    bias_codes = _unpack_unsigned(packed[114:123], 12, 6)
    biases = np.where(bias_codes >= 32, bias_codes - 64, bias_codes).astype(np.int8)
    lengths = _unpack_unsigned(packed[123:139], 32, 4).astype(np.uint8)
    k_base = packed[139]
    ks = (k_base + _unpack_unsigned(packed[140:142], 12, 1)).astype(np.uint8)
    if (
        np.any(factors > 512)
        or np.any(biases < -16)
        or np.any(biases > 16)
        or np.any(ks >= 12)
    ):
        raise ResidualArchiveError("packed CAP1 metadata exceeds canonical domains")
    result = (
        packed[:102]
        + factors.astype("<i2").tobytes()
        + biases.tobytes()
        + lengths.tobytes()
        + ks.tobytes()
        + packed[142:]
    )
    if len(result) != len(packed) + 40:
        raise ResidualArchiveError("packed CAP1 inverse produced the wrong length")
    return result'''

_RA_RESTORE_NEW = '''def _restore_packed_cap1_metadata(
    packed: bytes, *, metadata_bytes: int = 40
) -> bytes:
    """Expand the packed CAP1 metadata block into its canonical 80 bytes.

    DDM_KW1_WIDE_RICE_K_V1: ``metadata_bytes`` is the block width.  It defaults
    to the shipped 40, whose arithmetic below is unchanged; 43 selects the wide
    Rice-k block, which restores to the SAME canonical 80 bytes, so every stage
    downstream of this function is bit-identical between the two forms.
    """
    block_end = 102 + int(metadata_bytes)
    if len(packed) < block_end:
        raise ResidualArchiveError("packed CAP1 section is truncated")
    factor_base = packed[102]
    factors = factor_base + _unpack_unsigned(packed[103:114], 12, 7)
    bias_codes = _unpack_unsigned(packed[114:123], 12, 6)
    biases = np.where(bias_codes >= 32, bias_codes - 64, bias_codes).astype(np.int8)
    lengths = _unpack_unsigned(packed[123:139], 32, 4).astype(np.uint8)
    if int(metadata_bytes) == 40:
        k_base = packed[139]
        ks = (k_base + _unpack_unsigned(packed[140:142], 12, 1)).astype(np.uint8)
    elif int(metadata_bytes) == KW1_PACKED_METADATA_BYTES:
        from .kw1_wide_rice_k import unpack_ks as _kw1_unpack_ks

        ks = np.asarray(_kw1_unpack_ks(packed[102:block_end]), dtype=np.uint8)
    else:
        raise ResidualArchiveError("packed CAP1 metadata width is not a known form")
    if (
        np.any(factors > 512)
        or np.any(biases < -16)
        or np.any(biases > 16)
        or np.any(ks >= 12)
    ):
        raise ResidualArchiveError("packed CAP1 metadata exceeds canonical domains")
    result = (
        packed[:102]
        + factors.astype("<i2").tobytes()
        + biases.tobytes()
        + lengths.tobytes()
        + ks.tobytes()
        + packed[block_end:]
    )
    if len(result) != len(packed) + (80 - int(metadata_bytes)):
        raise ResidualArchiveError("packed CAP1 inverse produced the wrong length")
    return result'''

_RA_RIDERS_OLD = '''    if reserved & RR5_RESERVED_ARITH_BASIS:
        from .rr5_arith_basis import restore_carrier_body as restore_rr5_carrier_body

        carrier_body = restore_rr5_carrier_body(carrier_body)'''

_RA_RIDERS_NEW = '''    _kw1_metadata_bytes = (
        KW1_PACKED_METADATA_BYTES if reserved & KW1_RESERVED_WIDE_RICE_K else 40
    )
    if reserved & RR5_RESERVED_ARITH_BASIS:
        from .rr5_arith_basis import restore_carrier_body as restore_rr5_carrier_body

        carrier_body = restore_rr5_carrier_body(
            carrier_body, metadata_bytes=_kw1_metadata_bytes
        )'''

_RA_DX2_OLD = '''        from .dx2_cabac_coefficients import restore_carrier_body as restore_dx2_carrier_body

        carrier_body = restore_dx2_carrier_body(carrier_body)'''

_RA_DX2_NEW = '''        from .dx2_cabac_coefficients import restore_carrier_body as restore_dx2_carrier_body

        carrier_body = restore_dx2_carrier_body(
            carrier_body, metadata_bytes=_kw1_metadata_bytes
        )'''

_RA_PORTION_OLD = '''    _packed_portion = 102 + 40 + (
        (int.from_bytes(carrier_body[0:3], "little") + 7) // 8
    ) + ((int.from_bytes(carrier_body[3:6], "little") + 7) // 8)
    if len(carrier_body) >= _packed_portion + 9:
        carrier_body = (
            _restore_packed_cap1_metadata(carrier_body[:_packed_portion])
            + carrier_body[_packed_portion:]
        )'''

_RA_PORTION_NEW = '''    _packed_portion = 102 + _kw1_metadata_bytes + (
        (int.from_bytes(carrier_body[0:3], "little") + 7) // 8
    ) + ((int.from_bytes(carrier_body[3:6], "little") + 7) // 8)
    if len(carrier_body) >= _packed_portion + 9:
        carrier_body = (
            _restore_packed_cap1_metadata(
                carrier_body[:_packed_portion], metadata_bytes=_kw1_metadata_bytes
            )
            + carrier_body[_packed_portion:]
        )'''


PATCHES: dict[str, tuple[tuple[str, str], ...]] = {
    "rr5_arith_basis.py": (
        (_RR5_CONST_OLD, _RR5_CONST_NEW),
        (_RR5_SPLIT_OLD, _RR5_SPLIT_NEW),
        (_RR5_ASSEMBLE_OLD, _RR5_ASSEMBLE_NEW),
        (_RR5_APPLY_SIG_OLD, _RR5_APPLY_SIG_NEW),
        (_RR5_APPLY_OLD, _RR5_APPLY_NEW),
        (_RR5_RESTORE_SIG_OLD, _RR5_RESTORE_SIG_NEW),
        (_RR5_RESTORE_OLD, _RR5_RESTORE_NEW),
    ),
    "dx2_cabac_coefficients.py": (
        (_DX2_KS_OLD, _DX2_KS_NEW),
        (_DX2_APPLY_SIG_OLD, _DX2_APPLY_SIG_NEW),
        (_DX2_APPLY_OLD, _DX2_APPLY_NEW),
        (_DX2_APPLY_TAIL_OLD, _DX2_APPLY_TAIL_NEW),
        (_DX2_RESTORE_SIG_OLD, _DX2_RESTORE_SIG_NEW),
        (_DX2_RESTORE_OLD, _DX2_RESTORE_NEW),
    ),
    "residual_archive.py": (
        (_RA_BITS_OLD, _RA_BITS_NEW),
        (_RA_RESTORE_OLD, _RA_RESTORE_NEW),
        (_RA_RIDERS_OLD, _RA_RIDERS_NEW),
        (_RA_DX2_OLD, _RA_DX2_NEW),
        (_RA_PORTION_OLD, _RA_PORTION_NEW),
    ),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def apply_patches(runtime_dir: Path) -> dict[str, dict[str, object]]:
    """Patch a staged ``runtime/`` in place and copy the KW1 module in.

    Every anchor must appear EXACTLY ONCE in the file it targets; anything else
    refuses, so a receiver generation whose text has moved cannot be patched
    somewhere unintended.
    """
    runtime_dir = Path(runtime_dir)
    if not (runtime_dir / "residual_archive.py").is_file():
        raise ReceiverPatchError(f"{runtime_dir} is not a receiver runtime tree")
    report: dict[str, dict[str, object]] = {}
    for name, edits in PATCHES.items():
        path = runtime_dir / name
        if not path.is_file():
            raise ReceiverPatchError(f"receiver module missing: {path}")
        original = path.read_bytes()
        text = original.decode("utf-8")
        for index, (old, new) in enumerate(edits):
            occurrences = text.count(old)
            if occurrences != 1:
                raise ReceiverPatchError(
                    f"{name} patch {index} anchor appears {occurrences} times, "
                    "expected exactly 1"
                )
            text = text.replace(old, new, 1)
        patched = text.encode("utf-8")
        path.write_bytes(patched)
        report[name] = {
            "shipped_sha256": _sha256(original),
            "shipped_bytes": len(original),
            "patched_sha256": _sha256(patched),
            "patched_bytes": len(patched),
            "edits": len(edits),
        }
    kw1_bytes = KW1_SOURCE.read_bytes()
    shutil.copyfile(KW1_SOURCE, runtime_dir / "kw1_wide_rice_k.py")
    report["kw1_wide_rice_k.py"] = {
        "shipped_sha256": None,
        "shipped_bytes": 0,
        "patched_sha256": _sha256(kw1_bytes),
        "patched_bytes": len(kw1_bytes),
        "edits": 0,
        "source": str(KW1_SOURCE),
    }
    return report
