"""Tests for ``experiments/ddm_jg2_tail_reencode.py``.

These cover the parts that decide whether a REPORTED BYTE DELTA IS REAL: the RX1
section split (a wrong split silently mis-slices the coder stream), the archive
repack round trip, the edit-application accounting (``tokens_changed`` is the
denominator of the headline bits-per-token number), and the RC64 source pin.

They do NOT re-test the coder or the IHS1 model.  Those are the shipped runtime's
own code, imported rather than reimplemented, and the module proves their
inverse-ness empirically with its 600-frame control stage instead of by unit test.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "experiments" / "ddm_jg2_tail_reencode.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ddm_jg2_tail_reencode", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ddm_jg2_tail_reencode"] = module
    spec.loader.exec_module(module)
    return module


jg2 = _load_module()


def _member(hpac: bytes, semantic: bytes, carrier: bytes, tail: bytes) -> bytes:
    header = struct.pack(
        "<4sBBBBHHH", b"RX1M", 1, 2, 0, 6, len(hpac), len(semantic), len(carrier)
    )
    return header + hpac + semantic + carrier + tail


# --- RX1 section split ------------------------------------------------------


def test_split_member_slices_every_section_at_its_declared_length():
    member = _member(b"H" * 7, b"S" * 11, b"C" * 5, b"T" * 23)
    sections = jg2.split_member(member)
    assert sections["hpac"] == b"H" * 7
    assert sections["semantic"] == b"S" * 11
    assert sections["carrier"] == b"C" * 5
    assert sections["tail"] == b"T" * 23


def test_split_member_treats_everything_after_the_three_streams_as_tail():
    """The header counts three streams; the tail is defined as the remainder."""
    member = _member(b"H", b"S", b"C", b"")
    assert jg2.split_member(member)["tail"] == b""
    longer = _member(b"H", b"S", b"C", b"x" * 1000)
    assert len(jg2.split_member(longer)["tail"]) == 1000


def test_split_member_round_trips_through_join_member():
    member = _member(b"H" * 3, b"S" * 4, b"C" * 5, b"T" * 6)
    assert jg2.join_member(jg2.split_member(member)) == member


def test_join_member_ignores_extra_keys_added_by_prepare():
    """``_prepare`` annotates the dict with the tail's own sub-split."""
    sections = jg2.split_member(_member(b"H", b"S", b"C", b"T" * 10))
    sections["residual_compact"] = sections["tail"][:2]
    sections["token_stream"] = sections["tail"][2:]
    assert jg2.join_member(sections) == _member(b"H", b"S", b"C", b"T" * 10)


def test_split_member_refuses_a_foreign_magic():
    bad = struct.pack("<4sBBBBHHH", b"XXXX", 1, 2, 0, 6, 0, 0, 0)
    with pytest.raises(jg2.Jg2Error, match="magic"):
        jg2.split_member(bad)


def test_split_member_refuses_a_truncated_header():
    with pytest.raises(jg2.Jg2Error, match="too short"):
        jg2.split_member(b"RX1M")


def test_residual_compact_constant_matches_the_tail_arithmetic():
    """tail = 96 B fixed residual table + the RC64 stream. Drift here mis-slices."""
    assert (
        jg2.RESIDUAL_COMPACT_BYTES + jg2.SHIPPED_TOKEN_STREAM_BYTES
        == jg2.SHIPPED_TAIL_BYTES
    )


# --- archive container ------------------------------------------------------


def test_pack_archive_round_trips_the_member(tmp_path):
    member = _member(b"H", b"S", b"C", b"T" * 40)
    out = tmp_path / "a.zip"
    jg2.pack_archive(member, out)
    assert jg2.read_archive_member(out) == member


def test_pack_archive_stores_uncompressed_single_member(tmp_path):
    member = _member(b"H", b"S", b"C", bytes(4096))
    out = tmp_path / "a.zip"
    jg2.pack_archive(member, out)
    with zipfile.ZipFile(out) as archive:
        assert archive.namelist() == ["p"]
        assert archive.infolist()[0].compress_type == zipfile.ZIP_STORED


def test_pack_archive_is_deterministic_across_two_writes(tmp_path):
    """Byte-close needs the container itself to be reproducible."""
    member = _member(b"H", b"S", b"C", b"T" * 99)
    first, second = tmp_path / "1.zip", tmp_path / "2.zip"
    jg2.pack_archive(member, first)
    jg2.pack_archive(member, second)
    assert first.read_bytes() == second.read_bytes()


def test_read_archive_member_refuses_extra_members(tmp_path):
    out = tmp_path / "a.zip"
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("p", b"x")
        archive.writestr("q", b"y")
    with pytest.raises(jg2.Jg2Error, match="exactly member p"):
        jg2.read_archive_member(out)


# --- edit application (the denominator of bits-per-changed-token) -----------


def _tokens(value: int = 2) -> np.ndarray:
    return np.full((jg2.N_PAIRS, jg2.EVAL_H, jg2.EVAL_W), value, dtype=np.uint8)


def test_apply_edits_with_no_file_returns_an_unchanged_field():
    base = _tokens()
    field, report = jg2.apply_edits(base, None)
    assert report["tokens_changed"] == 0
    assert report["edited_pairs"] == []
    assert np.array_equal(field, base)


def test_apply_edits_counts_only_cells_that_actually_differ(tmp_path):
    base = _tokens()
    plane = np.array(base[7])
    plane[0, 0] = 3
    plane[5, 5] = 4
    path = tmp_path / "e.npz"
    np.savez(path, **{"7": plane})
    field, report = jg2.apply_edits(base, path)
    assert report["tokens_changed"] == 2
    assert report["edited_pairs"] == [7]
    assert field[7, 0, 0] == 3
    assert field[7, 5, 5] == 4


def test_apply_edits_writing_an_identical_plane_counts_zero(tmp_path):
    """A no-op edit must not inflate the denominator."""
    base = _tokens()
    path = tmp_path / "e.npz"
    np.savez(path, **{"3": np.array(base[3])})
    _, report = jg2.apply_edits(base, path)
    assert report["tokens_changed"] == 0
    assert report["edited_pairs"] == [3]


def test_apply_edits_does_not_mutate_the_source_field(tmp_path):
    base = _tokens()
    original = base.copy()
    plane = np.array(base[1])
    plane[2, 2] = 0
    path = tmp_path / "e.npz"
    np.savez(path, **{"1": plane})
    jg2.apply_edits(base, path)
    assert np.array_equal(base, original)


def test_apply_edits_refuses_a_wrong_shaped_plane(tmp_path):
    path = tmp_path / "e.npz"
    np.savez(path, **{"0": np.zeros((4, 4), dtype=np.uint8)})
    with pytest.raises(jg2.Jg2Error, match="shape"):
        jg2.apply_edits(_tokens(), path)


def test_apply_edits_refuses_a_token_outside_the_five_class_alphabet(tmp_path):
    """The coder's alphabet is 5; a stray 7 would be an unencodable symbol."""
    plane = np.zeros((jg2.EVAL_H, jg2.EVAL_W), dtype=np.uint8)
    plane[0, 0] = 7
    path = tmp_path / "e.npz"
    np.savez(path, **{"0": plane})
    with pytest.raises(jg2.Jg2Error, match="outside"):
        jg2.apply_edits(_tokens(), path)


def test_apply_edits_reports_sorted_pairs(tmp_path):
    base = _tokens()
    path = tmp_path / "e.npz"
    np.savez(path, **{"513": np.array(base[513]), "283": np.array(base[283])})
    _, report = jg2.apply_edits(base, path)
    assert report["edited_pairs"] == [283, 513]


# --- token field loading ----------------------------------------------------


def test_load_tokens_refuses_a_wrong_sized_field(tmp_path):
    path = tmp_path / "t.u8"
    path.write_bytes(b"\x00" * 100)
    with pytest.raises(jg2.Jg2Error, match="token field must be"):
        jg2.load_tokens(path)


# --- bit accounting ---------------------------------------------------------


def test_row_bits_of_a_certain_symbol_is_zero():
    rows = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    assert jg2._row_bits(rows, np.array([0])) == pytest.approx(0.0)


def test_row_bits_of_a_uniform_row_is_log2_of_the_alphabet():
    rows = np.full((4, 5), 0.2, dtype=np.float32)
    assert jg2._row_bits(rows, np.zeros(4, dtype=np.int64)) == pytest.approx(
        4 * np.log2(5), rel=1e-6
    )


def test_row_bits_is_additive_over_positions():
    rows = np.array([[0.5, 0.5, 0.0, 0.0, 0.0]] * 3, dtype=np.float32)
    assert jg2._row_bits(rows, np.array([0, 1, 0])) == pytest.approx(3.0)


# --- provenance pins --------------------------------------------------------


def test_rc64_base_sha_matches_the_ddm_rr2_pin():
    """Drifting off rr2's pinned encoder source would silently change the coder."""
    assert (
        jg2.RC64_BASE_SHA
        == "5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6"
    )


def test_pointer_constants_match_the_frontier_pointer():
    assert (
        jg2.POINTER_ARCHIVE_SHA
        == "7ce46fd7a845d5987903a0d85a56581961eb7716a55c38a7361e3b5ecae94b5f"
    )
    assert jg2.POINTER_ARCHIVE_BYTES == 176_420
    assert jg2.POINTER_MEMBER_BYTES == 176_320


def test_resolve_rc64_base_refuses_an_override_with_a_wrong_sha(tmp_path, monkeypatch):
    bogus = tmp_path / "rc64.c"
    bogus.write_bytes(b"int main(void){return 0;}")
    monkeypatch.setenv("TAC_JG2_RC64_SOURCE", str(bogus))
    with pytest.raises(jg2.Jg2Error, match="sha mismatch"):
        jg2.resolve_rc64_base(object(), tmp_path)


def test_score_per_archive_byte_is_the_contest_rate_derivative():
    assert pytest.approx(25.0 / 37_545_489) == jg2.S_PER_ARCHIVE_BYTE


def test_score_per_seg_cell_uses_the_full_argmax_lattice():
    assert pytest.approx(100.0 / 117_964_800) == jg2.S_PER_SEG_CELL


# --- CLI --------------------------------------------------------------------


def test_parser_requires_a_stage_and_a_store():
    parser = jg2.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_defaults_to_the_full_clip_and_a_resumable_cadence():
    args = jg2.build_parser().parse_args(["--stage", "control", "--store", "/tmp/x"])
    assert args.frames == jg2.N_PAIRS
    assert args.checkpoint_every == 25
    assert args.resume is False


def test_main_refuses_a_frame_count_outside_the_clip():
    with pytest.raises(SystemExit, match="--frames"):
        jg2.main(["--stage", "control", "--store", "/tmp/x", "--frames", "0"])
    with pytest.raises(SystemExit, match="--frames"):
        jg2.main(["--stage", "control", "--store", "/tmp/x", "--frames", "601"])


# --- atomic IO --------------------------------------------------------------


def test_atomic_write_leaves_no_partial_file(tmp_path):
    target = tmp_path / "x.bin"
    jg2.atomic_write(target, b"payload")
    assert target.read_bytes() == b"payload"
    assert not list(tmp_path.glob("*.partial"))


def test_file_fact_records_bytes_and_sha(tmp_path):
    target = tmp_path / "x.bin"
    jg2.atomic_write(target, b"abc")
    fact = jg2.file_fact(target)
    assert fact["bytes"] == 3
    assert fact["sha256"] == jg2.sha256_bytes(b"abc")
