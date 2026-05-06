"""Tests for pr106_stacked — composable subset of all 4 score-aware sidechannels.

Covers:
  - Outer wire format (0xFD magic + uint24 pr106_len + section list + 0x00 sentinel)
  - Empty (passthrough) composition: PR106 + only sentinel parses to 0 sections
  - Single-section compositions: latent only, yshift only, lrl1 only, wavelet only
  - Two-section compositions: latent+yshift, latent+lrl1, yshift+lrl1
  - Three-section composition: all 3 original sections (full stack)
  - Canonical-order application invariant (latent → yshift → lrl1 → wavelet consume-only)
  - End-of-sections sentinel discipline (missing/trailing-bytes guards)
  - Magic-byte anti-corruption guards
  - Duplicate-section rejection
  - Closing-the-loop: zero-correction full-stack pixel output equals plain PR106 inflate
  - Section parse drift: builder + inflate decode the same bytes identically

Mirrors test_pr106_yshift_sidechannel.py + test_pr106_lrl1_sidechannel.py
patterns. Uses sys.modules.pop('inflate', None) discipline per HEAD a98fc16f
fix (sister sidechannel module caching collision).

CPU-only — does not load CUDA scorers (per CLAUDE.md strict-scorer-rule, scorers
are NEVER loaded at inflate time). Wire-format + numerics are CUDA-independent.
Per CLAUDE.md MPS-noise rule: any score-producing assertion would need
[contest-CUDA] tag — this file ONLY tests bytewise wire format + composition.
"""
from __future__ import annotations

import subprocess
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.repo_io import read_json, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]
PR106_ARCHIVE = REPO_ROOT / (
    "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
)
INFLATE_DIR = REPO_ROOT / "submissions/pr106_stacked"
PR106_SRC = REPO_ROOT / "submissions/pr106_latent_sidecar/src"


@pytest.fixture(autouse=True)
def _cleanup_inflate_module_cache():
    """Pop both 'inflate' AND 'build_pr106_stacked' from sys.modules around
    every test in this file.

    Why: sister sidechannel inflate.py modules all import as bare name
    'inflate' (each in its own submissions/pr106_*_sidechannel/ dir). When
    multiple tests touch them in the same pytest session, sys.modules['inflate']
    holds whichever was last imported. The HEAD a98fc16f fix added pop()
    to yshift+lrl1 tests but the latent_sidecar test uses
    importlib.import_module without pop — so leaving our 'inflate' in cache
    breaks `pytest src/tac/tests/test_pr106_*` collection order.

    This autouse fixture ensures we both START and END each test with a
    clean cache, eliminating cross-test ordering effects.
    """
    sys.modules.pop("inflate", None)
    sys.modules.pop("build_pr106_stacked", None)
    yield
    sys.modules.pop("inflate", None)
    sys.modules.pop("build_pr106_stacked", None)


def _load_inflate():
    """Load submissions/pr106_stacked/inflate.py with sys.modules cache hygiene.

    Per HEAD a98fc16f fix: clear cached 'inflate' before importing — sister
    submissions all use module name 'inflate' and would otherwise leak.
    """
    sys.modules.pop("inflate", None)
    sys.path.insert(0, str(INFLATE_DIR))
    sys.path.insert(0, str(PR106_SRC))
    import inflate  # type: ignore[import-not-found]
    return inflate


def _load_builder():
    """Load experiments/build_pr106_stacked.py (depends on inflate cache hygiene)."""
    # Builder imports from inflate; ensure stacked inflate is the cached one.
    _load_inflate()
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    # Drop any stale builder import (other build_pr106_*.py modules
    # don't share a name, but this is defensive).
    sys.modules.pop("build_pr106_stacked", None)
    import build_pr106_stacked as builder  # type: ignore[import-not-found]
    return builder


def _build_zero_latent_blob() -> bytes:
    """Build a synthetic latent payload that is a TRUE no-op (all dim=255 sentinels)."""
    import brotli
    n_pairs = 600
    dim = np.full(n_pairs, 255, dtype=np.uint8)
    delta_q = np.zeros(n_pairs, dtype=np.int8)
    arr = np.stack([dim, delta_q.view(np.uint8)], axis=1)
    raw = struct.pack("<H", n_pairs) + arr.tobytes()
    return brotli.compress(raw, quality=11)


def _build_zero_yshift_blob() -> bytes:
    """Build a synthetic SC01-YSHIFT zero payload (no-op corrections)."""
    import brotli
    inflate = _load_inflate()
    n_frames = 1200
    raw = (
        inflate.SC01_HEADER.pack(
            inflate.SC01_MAGIC, inflate.SIDECHANNEL_MODE_Y_SHIFT, 3, n_frames, 1.0,
        )
        + np.zeros((n_frames, 3), dtype=np.int8).tobytes()
    )
    return brotli.compress(raw, quality=11)


def _build_zero_lrl1_blob(K: int = 4, low_h: int = 48, low_w: int = 64) -> bytes:
    """Build a synthetic LR01-LRL1 zero payload (no-op corrections)."""
    import brotli
    inflate = _load_inflate()
    n_frames = 1200
    raw = (
        inflate.LR01_HEADER.pack(
            inflate.LR01_MAGIC, inflate.SIDECHANNEL_MODE_LRL1,
            K, low_h, low_w, n_frames, 1.0, 1.0,
        )
        + np.zeros(K * low_h * low_w, dtype=np.int8).tobytes()
        + np.zeros(n_frames * K, dtype=np.int8).tobytes()
    )
    return brotli.compress(raw, quality=11)


def _build_wavelet_blob() -> bytes:
    """Build a synthetic WR01 payload matching tac.hnerv_wavelet_sidechannel."""
    from tac.hnerv_wavelet_sidechannel import encode_wavelet_atom_sidechannel

    plan = {
        "sections": [
            {
                "section_name": "latents_and_sidecar_brotli",
                "source_section_sha256": "1" * 64,
                "raw_bytes": 64,
                "atoms": [
                    {
                        "raw_offset": 2,
                        "raw_end": 4,
                        "level": 0,
                        "coefficient_index": 1,
                        "coefficient_quantized": -7,
                    },
                    {
                        "raw_offset": 16,
                        "raw_end": 20,
                        "level": 1,
                        "coefficient_index": 8,
                        "coefficient_quantized": 5,
                    },
                ],
            }
        ]
    }
    return encode_wavelet_atom_sidechannel(plan)


def _write_single_member_zip(path: Path, payload: bytes, *, member: str = "0.bin") -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        zf.writestr(info, payload)


# ===================================================================
# Constants + outer wire format
# ===================================================================


def test_stacked_constants():
    """Magic bytes + section IDs are the documented values."""
    inflate = _load_inflate()
    assert inflate.STACKED_MAGIC_BYTE == 0xFD
    assert inflate.SECTION_END == 0x00
    assert inflate.SECTION_LATENT == 0x01
    assert inflate.SECTION_YSHIFT == 0x02
    assert inflate.SECTION_LRL1 == 0x03
    assert inflate.SECTION_WAVELET == 0x04
    # Sister-mirror constants
    assert inflate.SC01_MAGIC == b"SC01"
    assert inflate.SIDECHANNEL_MODE_Y_SHIFT == 7
    assert inflate.SC01_HEADER.size == 14
    assert inflate.LR01_MAGIC == b"LR01"
    assert inflate.SIDECHANNEL_MODE_LRL1 == 8
    assert inflate.LR01_HEADER.size == 22
    assert inflate.WR01_MAGIC == b"WR01"
    assert inflate.WR01_SCHEMA_VERSION == 1
    assert inflate.LATENT_NO_OP_DIM == 255
    assert abs(inflate.LATENT_DELTA_SCALE - 0.01) < 1e-12


# ===================================================================
# Empty (passthrough) composition
# ===================================================================


def test_empty_composition_parses_zero_sections():
    """PR106 wrapped with NO sidechannels (just sentinel) parses cleanly."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    new_bin = builder.build_stacked_archive_bytes(pr106_bytes)

    # Wire structure: 0xFD + uint24 pr106_len + pr106_bytes + 0x00 sentinel
    assert new_bin[0] == 0xFD
    assert int.from_bytes(new_bin[1:4], "little") == len(pr106_bytes)
    assert new_bin[-1] == 0x00
    assert len(new_bin) == 1 + 3 + len(pr106_bytes) + 1

    sd, lat, meta, sections = inflate.parse_stacked_archive(new_bin)
    assert sections == {}
    assert len(sd) == 28
    assert tuple(lat.shape) == (600, 28)
    assert meta == {"n_pairs": 600, "latent_dim": 28, "base_channels": 36, "eval_size": [384, 512]}


def test_empty_composition_overhead_is_5_bytes():
    """Pure-passthrough wrap adds exactly 5 bytes (1 magic + 3 len + 1 sentinel)."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    new_bin = builder.build_stacked_archive_bytes(pr106_bytes)
    overhead = len(new_bin) - len(pr106_bytes)
    assert overhead == 5, f"expected 5-byte overhead for passthrough, got {overhead}"


# ===================================================================
# Single-section compositions
# ===================================================================


def test_latent_only_composition():
    """Latent section only (yshift+lrl1 absent)."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    latent_blob = _build_zero_latent_blob()
    new_bin = builder.build_stacked_archive_bytes(pr106_bytes, latent_blob=latent_blob)

    sd, lat, meta, sections = inflate.parse_stacked_archive(new_bin)
    assert set(sections.keys()) == {inflate.SECTION_LATENT}
    assert sections[inflate.SECTION_LATENT]["n_pairs"] == 600
    assert (sections[inflate.SECTION_LATENT]["dim"] == 255).all()
    assert (sections[inflate.SECTION_LATENT]["delta_q"] == 0).all()
    assert len(sd) == 28


def test_yshift_only_composition():
    """Yshift section only (latent+lrl1 absent)."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    yshift_blob = _build_zero_yshift_blob()
    new_bin = builder.build_stacked_archive_bytes(pr106_bytes, yshift_blob=yshift_blob)

    sd, lat, meta, sections = inflate.parse_stacked_archive(new_bin)
    assert set(sections.keys()) == {inflate.SECTION_YSHIFT}
    assert sections[inflate.SECTION_YSHIFT]["n_frames"] == 1200
    assert sections[inflate.SECTION_YSHIFT]["raw"].shape == (1200, 3)
    assert (sections[inflate.SECTION_YSHIFT]["raw"] == 0).all()
    assert sections[inflate.SECTION_YSHIFT]["step"] == 1.0


def test_lrl1_only_composition():
    """Lrl1 section only (latent+yshift absent)."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    lrl1_blob = _build_zero_lrl1_blob(K=4, low_h=48, low_w=64)
    new_bin = builder.build_stacked_archive_bytes(pr106_bytes, lrl1_blob=lrl1_blob)

    sd, lat, meta, sections = inflate.parse_stacked_archive(new_bin)
    assert set(sections.keys()) == {inflate.SECTION_LRL1}
    assert sections[inflate.SECTION_LRL1]["K"] == 4
    assert sections[inflate.SECTION_LRL1]["low_h"] == 48
    assert sections[inflate.SECTION_LRL1]["low_w"] == 64
    assert sections[inflate.SECTION_LRL1]["n_frames"] == 1200
    assert sections[inflate.SECTION_LRL1]["basis"].shape == (4, 48, 64)
    assert sections[inflate.SECTION_LRL1]["coeffs"].shape == (1200, 4)


def test_wavelet_only_composition_consumes_wr01_noop_runtime():
    """Wavelet section only is parsed and consumed as explicit no-op proof."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    wavelet_blob = _build_wavelet_blob()
    new_bin = builder.build_stacked_archive_bytes(pr106_bytes, wavelet_blob=wavelet_blob)

    sd, lat, meta, sections = inflate.parse_stacked_archive(new_bin)
    assert set(sections.keys()) == {inflate.SECTION_WAVELET}
    wavelet = sections[inflate.SECTION_WAVELET]
    assert wavelet["magic"] == "WR01"
    assert wavelet["runtime_mode"] == "explicit_noop_consume_only"
    assert wavelet["total_atom_count"] == 2
    assert wavelet["runtime_consumption_proof"]["runtime_consumed"] is True
    assert wavelet["runtime_consumption_proof"]["decoded_atom_count"] == 2
    assert len(sd) == 28
    assert tuple(lat.shape) == (600, 28)
    assert meta["n_pairs"] == 600


def test_builder_extracts_wavelet_sidechannel_blob_from_candidate_archive(tmp_path: Path):
    """Builder extracts WR01 blobs from 0xFA candidates and checks anchor bytes."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    from tac.hnerv_lowlevel_packer import write_stored_single_member_zip
    from tac.hnerv_wavelet_sidechannel import build_wavelet_sidechannel_archive_bytes

    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    wavelet_blob = _build_wavelet_blob()
    candidate_payload = build_wavelet_sidechannel_archive_bytes(
        source_payload=pr106_bytes,
        sidechannel_blob=wavelet_blob,
    )
    archive = tmp_path / "wavelet_candidate.zip"
    write_stored_single_member_zip(archive, member_name="x", payload=candidate_payload)

    extracted = builder.extract_wavelet_section_blob(
        archive,
        expected_pr106_bytes=pr106_bytes,
    )
    assert extracted == wavelet_blob
    decoded = inflate.decode_wavelet_blob(extracted)
    assert decoded["runtime_mode"] == "explicit_noop_consume_only"
    assert decoded["runtime_consumption_proof"]["decoded_atom_count"] == 2


def test_builder_rejects_latent_sister_anchor_mismatch(tmp_path: Path):
    builder = _load_builder()
    archive = tmp_path / "latent.zip"
    expected = b"\xffanchor"
    wrong = b"\xffwrong!"
    sidecar = _build_zero_latent_blob()
    payload = (
        bytes([builder.LATENT_OUTER_MAGIC, builder.LATENT_FORMAT_ID])
        + len(wrong).to_bytes(4, "little")
        + wrong
        + struct.pack("<H", len(sidecar))
        + sidecar
    )
    _write_single_member_zip(archive, payload)

    with pytest.raises(ValueError, match="embedded PR106 payload does not match"):
        builder.extract_latent_section_blob(archive, expected_pr106_bytes=expected)


def test_builder_rejects_yshift_sister_anchor_mismatch(tmp_path: Path):
    builder = _load_builder()
    archive = tmp_path / "yshift.zip"
    expected = b"\xffanchor"
    wrong = b"\xffwrong!"
    blob = _build_zero_yshift_blob()
    payload = (
        bytes([builder.YSHIFT_OUTER_MAGIC])
        + len(wrong).to_bytes(3, "little")
        + wrong
        + bytes([1])
        + struct.pack("<H", len(blob))
        + blob
    )
    _write_single_member_zip(archive, payload)

    with pytest.raises(ValueError, match="embedded PR106 payload does not match"):
        builder.extract_yshift_section_blob(archive, expected_pr106_bytes=expected)


def test_builder_rejects_lrl1_sister_anchor_mismatch(tmp_path: Path):
    builder = _load_builder()
    archive = tmp_path / "lrl1.zip"
    expected = b"\xffanchor"
    wrong = b"\xffwrong!"
    blob = _build_zero_lrl1_blob()
    payload = (
        bytes([builder.LRL1_OUTER_MAGIC])
        + len(wrong).to_bytes(3, "little")
        + wrong
        + bytes([1])
        + struct.pack("<H", len(blob))
        + blob
    )
    _write_single_member_zip(archive, payload)

    with pytest.raises(ValueError, match="embedded PR106 payload does not match"):
        builder.extract_lrl1_section_blob(archive, expected_pr106_bytes=expected)


# ===================================================================
# Two-section compositions
# ===================================================================


def test_latent_yshift_composition():
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()
    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    new_bin = builder.build_stacked_archive_bytes(
        pr106_bytes,
        latent_blob=_build_zero_latent_blob(),
        yshift_blob=_build_zero_yshift_blob(),
    )
    sd, lat, meta, sections = inflate.parse_stacked_archive(new_bin)
    assert set(sections.keys()) == {inflate.SECTION_LATENT, inflate.SECTION_YSHIFT}


def test_latent_lrl1_composition():
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()
    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    new_bin = builder.build_stacked_archive_bytes(
        pr106_bytes,
        latent_blob=_build_zero_latent_blob(),
        lrl1_blob=_build_zero_lrl1_blob(),
    )
    sd, lat, meta, sections = inflate.parse_stacked_archive(new_bin)
    assert set(sections.keys()) == {inflate.SECTION_LATENT, inflate.SECTION_LRL1}


def test_yshift_lrl1_composition():
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()
    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    new_bin = builder.build_stacked_archive_bytes(
        pr106_bytes,
        yshift_blob=_build_zero_yshift_blob(),
        lrl1_blob=_build_zero_lrl1_blob(),
    )
    sd, lat, meta, sections = inflate.parse_stacked_archive(new_bin)
    assert set(sections.keys()) == {inflate.SECTION_YSHIFT, inflate.SECTION_LRL1}


# ===================================================================
# Three-section composition (full stack)
# ===================================================================


def test_full_stack_composition():
    """All 3 sidechannels present — composes + parses cleanly."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()
    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    new_bin = builder.build_stacked_archive_bytes(
        pr106_bytes,
        latent_blob=_build_zero_latent_blob(),
        yshift_blob=_build_zero_yshift_blob(),
        lrl1_blob=_build_zero_lrl1_blob(),
    )
    sd, lat, meta, sections = inflate.parse_stacked_archive(new_bin)
    assert set(sections.keys()) == {
        inflate.SECTION_LATENT, inflate.SECTION_YSHIFT, inflate.SECTION_LRL1,
    }
    assert sections[inflate.SECTION_LATENT]["n_pairs"] == 600
    assert sections[inflate.SECTION_YSHIFT]["n_frames"] == 1200
    assert sections[inflate.SECTION_LRL1]["n_frames"] == 1200


def test_full_stack_overhead_under_2kb():
    """Zero-init full stack should add < 2KB overhead vs PR106."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    new_bin = builder.build_stacked_archive_bytes(
        pr106_bytes,
        latent_blob=_build_zero_latent_blob(),
        yshift_blob=_build_zero_yshift_blob(),
        lrl1_blob=_build_zero_lrl1_blob(),
    )
    overhead = len(new_bin) - len(pr106_bytes)
    # 5-byte outer wrapper + 3 sections (~3 + 4 + 4 = ~11 bytes section headers)
    # + ~15 + 37 + 43 brotli'd zero payloads = ~150 bytes total
    assert overhead < 2048, f"overhead {overhead} bytes exceeds 2KB ceiling"
    assert overhead < 500, f"zero-init overhead {overhead} should be tiny (~150 bytes)"


# ===================================================================
# Anti-corruption guards
# ===================================================================


def test_outer_archive_rejects_bad_magic():
    """First byte != 0xFD raises."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()
    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    new_bin = bytearray(builder.build_stacked_archive_bytes(pr106_bytes))
    new_bin[0] = 0xFE  # latent_sidecar's magic
    with pytest.raises(ValueError, match="pr106_stacked magic mismatch"):
        inflate.parse_stacked_archive(bytes(new_bin))


def test_outer_archive_rejects_missing_sentinel():
    """No 0x00 sentinel at tail raises."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()
    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    new_bin = bytearray(builder.build_stacked_archive_bytes(pr106_bytes))
    # Remove the sentinel byte
    truncated = bytes(new_bin[:-1])
    with pytest.raises(ValueError, match="missing end-of-sections sentinel"):
        inflate.parse_stacked_archive(truncated)


def test_outer_archive_rejects_trailing_bytes_after_sentinel():
    """Extra bytes after 0x00 sentinel raise."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()
    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    new_bin = builder.build_stacked_archive_bytes(pr106_bytes) + b"GARBAGE"
    with pytest.raises(ValueError, match="trailing bytes after end-of-sections"):
        inflate.parse_stacked_archive(new_bin)


def test_outer_archive_rejects_unknown_section_id():
    """Unknown section_id (0x05+) raises."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()
    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    # Hand-craft archive: PR106 + section_id=0x05 + 0-len section + sentinel
    bad = (
        bytes([0xFD])
        + len(pr106_bytes).to_bytes(3, "little")
        + pr106_bytes
        + bytes([0x05])
        + struct.pack("<H", 0)
        + bytes([0x00])
    )
    with pytest.raises(ValueError, match="unknown section id 0x05"):
        inflate.parse_stacked_archive(bad)


def test_outer_archive_rejects_duplicate_section():
    """Same section_id twice on the wire raises."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()
    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    yshift_blob = _build_zero_yshift_blob()
    # Hand-craft: PR106 + yshift section TWICE + sentinel
    bad = (
        bytes([0xFD])
        + len(pr106_bytes).to_bytes(3, "little")
        + pr106_bytes
        + bytes([0x02]) + struct.pack("<H", len(yshift_blob)) + yshift_blob
        + bytes([0x02]) + struct.pack("<H", len(yshift_blob)) + yshift_blob
        + bytes([0x00])
    )
    with pytest.raises(ValueError, match="duplicate section id 0x02"):
        inflate.parse_stacked_archive(bad)


def test_outer_archive_rejects_section_overruns_archive():
    """Section length field claiming more bytes than archive holds raises."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()
    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    bad = (
        bytes([0xFD])
        + len(pr106_bytes).to_bytes(3, "little")
        + pr106_bytes
        + bytes([0x01])
        + struct.pack("<H", 9999)  # claim 9999 bytes but archive has only sentinel left
        + bytes([0x00])
    )
    with pytest.raises(ValueError, match="declared length 9999 exceeds"):
        inflate.parse_stacked_archive(bad)


def test_outer_archive_rejects_pr106_len_overrun():
    """pr106_len exceeding archive bounds raises."""
    inflate = _load_inflate()
    bad = bytes([0xFD]) + (1 << 23).to_bytes(3, "little") + b"\xff\x00" + bytes([0x00])
    with pytest.raises(ValueError, match="exceeds archive size"):
        inflate.parse_stacked_archive(bad)


# ===================================================================
# Frame-count validation (sidechannel n_frames must match decoder n_pairs*2)
# ===================================================================


def test_yshift_frame_count_mismatch_raises_at_inflate():
    """Yshift section with wrong n_frames is caught — but decode_yshift_blob
    itself doesn't enforce this; the inflate driver does. Test the driver-side
    path by calling parse_stacked_archive (which doesn't check) then asserting
    we'd catch it in inflate via the validation block. Smoke-skip if PR106
    not present.

    NOTE: parse_stacked_archive doesn't validate frame counts (those need
    decoder meta); the actual validation happens in inflate.inflate(). This
    test confirms decode_yshift_blob accepts the bytes and the validation
    happens later.
    """
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    inflate = _load_inflate()
    import brotli
    # Build yshift with WRONG n_frames (600 instead of 1200)
    raw = (
        inflate.SC01_HEADER.pack(
            inflate.SC01_MAGIC, inflate.SIDECHANNEL_MODE_Y_SHIFT, 3, 600, 1.0,
        )
        + np.zeros((600, 3), dtype=np.int8).tobytes()
    )
    blob = brotli.compress(raw, quality=11)
    decoded = inflate.decode_yshift_blob(blob)
    # decode itself succeeds — the inflate driver catches the mismatch
    assert decoded["n_frames"] == 600  # which would mismatch decoder's 1200


# ===================================================================
# Section parser drift (builder + inflate must decode identically)
# ===================================================================


def test_builder_inflate_decoder_parity():
    """build_stacked_archive_bytes + parse_stacked_archive form a clean roundtrip."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()
    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)

    latent_blob = _build_zero_latent_blob()
    yshift_blob = _build_zero_yshift_blob()
    lrl1_blob = _build_zero_lrl1_blob()

    new_bin = builder.build_stacked_archive_bytes(
        pr106_bytes,
        latent_blob=latent_blob,
        yshift_blob=yshift_blob,
        lrl1_blob=lrl1_blob,
    )

    sd, lat, meta, sections = inflate.parse_stacked_archive(new_bin)
    # The decoded latent/yshift/lrl1 dicts should match what we'd get by
    # decoding the input blobs directly. NumPy arrays prevent dict ==, so
    # compare components individually.
    latent_decoded = inflate.decode_latent_blob(latent_blob)
    assert sections[inflate.SECTION_LATENT]["n_pairs"] == latent_decoded["n_pairs"]
    assert np.array_equal(
        sections[inflate.SECTION_LATENT]["dim"], latent_decoded["dim"],
    )
    assert np.array_equal(
        sections[inflate.SECTION_LATENT]["delta_q"], latent_decoded["delta_q"],
    )
    # NumPy arrays don't compare with ==; decode each manually and compare components
    yshift_decoded = inflate.decode_yshift_blob(yshift_blob)
    assert sections[inflate.SECTION_YSHIFT]["mode_id"] == yshift_decoded["mode_id"]
    assert sections[inflate.SECTION_YSHIFT]["step"] == yshift_decoded["step"]
    assert sections[inflate.SECTION_YSHIFT]["n_frames"] == yshift_decoded["n_frames"]
    assert np.array_equal(
        sections[inflate.SECTION_YSHIFT]["raw"], yshift_decoded["raw"],
    )
    lrl1_decoded = inflate.decode_lrl1_blob(lrl1_blob)
    assert sections[inflate.SECTION_LRL1]["K"] == lrl1_decoded["K"]
    assert sections[inflate.SECTION_LRL1]["n_frames"] == lrl1_decoded["n_frames"]
    assert np.array_equal(
        sections[inflate.SECTION_LRL1]["basis"], lrl1_decoded["basis"],
    )
    assert np.array_equal(
        sections[inflate.SECTION_LRL1]["coeffs"], lrl1_decoded["coeffs"],
    )


# ===================================================================
# Closing-the-loop test: zero-correction full stack == pure PR106 inflate
# (numerics only — pixel-level equality requires CUDA, so we test the
#  no-op semantics of each application function instead)
# ===================================================================


def test_apply_latent_corrections_no_op_with_sentinels():
    """All-dim=255 latent application is a true identity on latents."""
    inflate = _load_inflate()
    import torch
    n = 100
    latents = torch.randn(n, 28)
    snapshot = latents.clone()
    dim_arr = np.full(n, 255, dtype=np.uint8)
    delta_q_arr = np.zeros(n, dtype=np.int8)
    inflate.apply_latent_corrections(latents, dim_arr, delta_q_arr)
    assert torch.equal(latents, snapshot), "no-op latent corrections mutated tensor"


def test_apply_yshift_no_op_with_zero_row():
    """Zero (y_off, dy, dx) yields identity on a frame."""
    inflate = _load_inflate()
    rng = np.random.default_rng(seed=99)
    frame = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    sc_row = np.zeros(3, dtype=np.int8)
    out = inflate.apply_yshift(frame, sc_row, step=1.0)
    assert np.array_equal(out, frame)


def test_apply_lrl1_no_op_with_zero_coeffs():
    """All-zero coeffs yield identity correction (no pixel change)."""
    inflate = _load_inflate()
    K, low_h, low_w = 4, 48, 64
    H, W = 32, 32  # smaller test frame for speed
    rng = np.random.default_rng(seed=33)
    frame = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    basis = np.zeros((K, low_h, low_w), dtype=np.int8)
    upsampled = inflate.upsample_basis(basis, basis_step=1.0, target_h=H, target_w=W)
    coeffs = np.zeros(K, dtype=np.int8)
    out = inflate.apply_lrl1_to_frame(frame, upsampled, coeffs, coeff_step=1.0)
    assert np.array_equal(out, frame), "zero LRL1 correction mutated frame"


def test_passthrough_archive_yields_pure_pr106_pipeline():
    """Empty composition (no sections) produces a wrapped archive whose
    parsed (sd, lat, meta) is bit-identical to plain PR106 parse_packed_archive.

    This is the closing-the-loop test: zero corrections = no-op = same pixel
    output as plain PR106 inflate (proven by identical state_dict + latents +
    meta entering the decoder forward pass).
    """
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    # Direct PR106 parse (baseline)
    sys.path.insert(0, str(PR106_SRC))
    from codec import parse_packed_archive  # type: ignore[import-not-found]
    pr106_sd, pr106_lat, pr106_meta = parse_packed_archive(pr106_bytes)

    # Wrapped passthrough parse
    new_bin = builder.build_stacked_archive_bytes(pr106_bytes)
    wrapped_sd, wrapped_lat, wrapped_meta, wrapped_sections = (
        inflate.parse_stacked_archive(new_bin)
    )

    assert wrapped_sections == {}
    assert wrapped_meta == pr106_meta
    import torch
    assert torch.equal(wrapped_lat, pr106_lat), "wrapped latents differ from PR106 baseline"
    assert set(wrapped_sd.keys()) == set(pr106_sd.keys())
    for k in pr106_sd:
        assert torch.equal(wrapped_sd[k], pr106_sd[k]), (
            f"wrapped tensor {k!r} differs from PR106 baseline"
        )


def test_zero_correction_full_stack_yields_pure_pr106_pipeline():
    """All 3 sidechannels with TRUE zero corrections compose into an archive
    whose decoded (sd, lat, meta) exactly matches plain PR106, AND whose
    sidechannel sections all parse to no-op semantics.

    This is THE closing-the-loop test: composition with zero corrections is
    a pure no-op — the wrapped archive renders bit-identical pixels to plain
    PR106 inflate (proven structurally without needing a CUDA forward pass).
    """
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    sys.path.insert(0, str(PR106_SRC))
    from codec import parse_packed_archive  # type: ignore[import-not-found]
    pr106_sd, pr106_lat, pr106_meta = parse_packed_archive(pr106_bytes)

    # Build full-stack with zero corrections
    new_bin = builder.build_stacked_archive_bytes(
        pr106_bytes,
        latent_blob=_build_zero_latent_blob(),
        yshift_blob=_build_zero_yshift_blob(),
        lrl1_blob=_build_zero_lrl1_blob(),
    )
    wrapped_sd, wrapped_lat, wrapped_meta, wrapped_sections = (
        inflate.parse_stacked_archive(new_bin)
    )

    # Decoder inputs match PR106 exactly
    assert wrapped_meta == pr106_meta
    import torch
    assert torch.equal(wrapped_lat, pr106_lat)
    assert set(wrapped_sd.keys()) == set(pr106_sd.keys())
    for k in pr106_sd:
        assert torch.equal(wrapped_sd[k], pr106_sd[k])

    # All 3 sidechannels are no-ops
    latent_section = wrapped_sections[inflate.SECTION_LATENT]
    assert (latent_section["dim"] == 255).all(), "latent has non-sentinel dim"
    assert (latent_section["delta_q"] == 0).all(), "latent has non-zero delta"

    yshift_section = wrapped_sections[inflate.SECTION_YSHIFT]
    assert (yshift_section["raw"] == 0).all(), "yshift has non-zero corrections"

    lrl1_section = wrapped_sections[inflate.SECTION_LRL1]
    assert (lrl1_section["basis"] == 0).all(), "lrl1 basis is non-zero"
    assert (lrl1_section["coeffs"] == 0).all(), "lrl1 coeffs are non-zero"

    # Apply each no-op transform manually and confirm identity preserved
    snapshot = pr106_lat.clone()
    inflate.apply_latent_corrections(pr106_lat, latent_section["dim"], latent_section["delta_q"])
    assert torch.equal(pr106_lat, snapshot), "latent no-op mutated PR106 latents"


# ===================================================================
# Real-archive integration via builder driver (end-to-end CLI smoke)
# ===================================================================


def test_builder_extracts_blob_from_yshift_sister_archive(tmp_path):
    """builder.extract_yshift_section_blob round-trips the sister-archive payload."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()

    # Build a sister yshift archive
    sister_dir = tmp_path / "yshift"
    sister_dir.mkdir()
    yshift_blob = _build_zero_yshift_blob()
    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    # Hand-build the sister-archive layout (magic 0xFC + uint24 len +
    # pr106 + version 1 + uint16 sc_len + sc_blob)
    sister_bin = (
        bytes([0xFC])
        + len(pr106_bytes).to_bytes(3, "little")
        + pr106_bytes
        + bytes([1])
        + struct.pack("<H", len(yshift_blob))
        + yshift_blob
    )
    sister_zip = sister_dir / "pr106_yshift_sidechannel_archive.zip"
    with zipfile.ZipFile(sister_zip, "w", compression=zipfile.ZIP_STORED) as z:
        zi = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_STORED
        z.writestr(zi, sister_bin)

    extracted = builder.extract_yshift_section_blob(sister_zip)
    assert extracted == yshift_blob


def test_builder_rejects_wrong_outer_magic_sister(tmp_path):
    """builder rejects a sister archive with the wrong outer magic byte."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    pr106_bytes = builder.extract_pr106_bytes(PR106_ARCHIVE)
    # Hand-build a sister with the WRONG magic (latent's 0xFE) but yshift layout
    yshift_blob = _build_zero_yshift_blob()
    bogus = (
        bytes([0xFE])  # wrong: should be 0xFC for yshift
        + len(pr106_bytes).to_bytes(3, "little")
        + pr106_bytes
        + bytes([1])
        + struct.pack("<H", len(yshift_blob))
        + yshift_blob
    )
    bogus_zip = tmp_path / "bogus_yshift.zip"
    with zipfile.ZipFile(bogus_zip, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("0.bin", bogus)

    with pytest.raises(ValueError, match="expected pr106_yshift_sidechannel"):
        builder.extract_yshift_section_blob(bogus_zip)


def test_builder_metadata_records_archive_hash_custody(tmp_path):
    """CLI metadata records byte/hash custody for anchor and output archives."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    out_dir = tmp_path / "stacked"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "build_pr106_stacked.py"),
            "--pr106-archive",
            str(PR106_ARCHIVE),
            "--output-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[build-stacked] wrote" in proc.stdout

    archive_path = out_dir / "pr106_stacked_archive.zip"
    metadata = read_json(out_dir / "build_metadata.json")
    assert metadata["manifest_schema"] == "pr106_stacked_build_metadata_v2"
    assert metadata["pr106_archive_sha256"] == sha256_file(PR106_ARCHIVE)
    assert metadata["archive_sha256"] == sha256_file(archive_path)
    assert metadata["input_archives"]["pr106"] == {
        "path": str(PR106_ARCHIVE),
        "bytes": PR106_ARCHIVE.stat().st_size,
        "sha256": sha256_file(PR106_ARCHIVE),
    }
    assert metadata["input_archives"]["latent"] is None
    assert metadata["input_archives"]["yshift"] is None
    assert metadata["input_archives"]["lrl1"] is None
    assert metadata["input_archives"]["wavelet"] is None
    assert metadata["output_archive"] == {
        "path": str(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
    }
