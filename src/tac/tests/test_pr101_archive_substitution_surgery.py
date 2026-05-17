# SPDX-License-Identifier: MIT
"""Tests for ``tools.pr101_archive_substitution_surgery``.

Verifies:
  - PR101 archive layout split is correct on real archive
  - decoder_blob substitution preserves the latent + sidecar bytes exactly
  - replacement length validation rejects wrong-size blobs
  - byte-faithful roundtrip: substitute the SAME decoder_blob → output
    archive's inner blob is byte-identical to input
  - verify subcommand emits structured report
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import zipfile

import pytest


def _load_tool_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "pr101_archive_substitution_surgery.py"
    spec = importlib.util.spec_from_file_location(
        "pr101_archive_substitution_surgery", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_synthetic_pr101_archive(
    tmp_path: pathlib.Path,
    *,
    decoder_blob: bytes | None = None,
    latent_blob: bytes | None = None,
    sidecar_blob: bytes | None = None,
) -> pathlib.Path:
    """Build a PR101-shaped archive with caller-controlled blob contents.

    Defaults: deterministic byte patterns so substitution roundtrip can
    be verified byte-faithfully.
    """
    mod = _load_tool_module()
    decoder_blob = decoder_blob if decoder_blob is not None else b"\xa1" * mod.PR101_DECODER_BLOB_LEN
    latent_blob = latent_blob if latent_blob is not None else b"\xb2" * mod.PR101_LATENT_BLOB_LEN
    sidecar_blob = sidecar_blob if sidecar_blob is not None else b"\xc3" * 607

    archive_path = tmp_path / "synthetic_pr101.zip"
    info = zipfile.ZipInfo(filename=mod.PR101_INNER_MEMBER_NAME)
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(info, decoder_blob + latent_blob + sidecar_blob)
    return archive_path


# ---------------------------------------------------------------------------
# Layout splitting
# ---------------------------------------------------------------------------

def test_split_inner_blob_correct_lengths(tmp_path) -> None:
    mod = _load_tool_module()
    archive = _make_synthetic_pr101_archive(tmp_path)
    blob = mod._read_inner_blob(archive)
    decoder, latent, sidecar = mod._split_pr101_inner_blob(blob)
    assert len(decoder) == mod.PR101_DECODER_BLOB_LEN
    assert len(latent) == mod.PR101_LATENT_BLOB_LEN
    assert len(sidecar) == 607  # default test sidecar


def test_split_rejects_too_short_blob() -> None:
    mod = _load_tool_module()
    short_blob = b"\x00" * (mod.PR101_DECODER_BLOB_LEN - 100)
    with pytest.raises(ValueError, match="< required minimum"):
        mod._split_pr101_inner_blob(short_blob)


def test_split_rejects_a1_prefixed_no_dead_k_layout() -> None:
    mod = _load_tool_module()
    blob = (
        mod.A1_PREFIXED_DECODER_SECTION_TOTAL.to_bytes(4, "little")
        + b"\xa1" * mod.PR101_DECODER_BLOB_LEN
        + b"\xb2" * mod.PR101_LATENT_BLOB_LEN
        + b"\xc3" * 607
    )

    with pytest.raises(ValueError, match="A1 prefixed no-dead-K layout"):
        mod._split_pr101_inner_blob(blob)


def test_read_inner_blob_rejects_wrong_member_name(tmp_path) -> None:
    mod = _load_tool_module()
    archive = tmp_path / "wrong_name.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("not_x", b"\x00" * 200_000)
    with pytest.raises(ValueError, match=r"members .+ expected"):
        mod._read_inner_blob(archive)


# ---------------------------------------------------------------------------
# Substitution roundtrip
# ---------------------------------------------------------------------------

def test_substitute_roundtrip_preserves_latent_and_sidecar(tmp_path) -> None:
    """Substituting the SAME decoder_blob bytes back must produce an
    archive whose inner blob is byte-identical to the input. This proves
    the latent + sidecar sections survive the round trip exactly."""
    mod = _load_tool_module()
    decoder_blob = b"\xa1" * mod.PR101_DECODER_BLOB_LEN
    latent_blob = b"\xb2" * mod.PR101_LATENT_BLOB_LEN
    sidecar_blob = b"\xc3" * 607
    archive = _make_synthetic_pr101_archive(
        tmp_path,
        decoder_blob=decoder_blob,
        latent_blob=latent_blob,
        sidecar_blob=sidecar_blob,
    )
    out = tmp_path / "roundtrip.zip"
    report = mod.substitute_decoder_blob(
        input_archive=archive,
        replacement_decoder_blob=decoder_blob,
        output_archive=out,
    )
    output_inner = mod._read_inner_blob(out)
    assert output_inner == decoder_blob + latent_blob + sidecar_blob
    assert report.sha256_input_decoder_blob == report.sha256_replacement_decoder_blob


def test_substitute_with_different_blob_changes_decoder_only(tmp_path) -> None:
    """Substituting a DIFFERENT decoder_blob must:
      (a) change the decoder section to the new bytes
      (b) leave the latent section byte-identical
      (c) leave the sidecar section byte-identical
    """
    mod = _load_tool_module()
    archive = _make_synthetic_pr101_archive(tmp_path)
    new_decoder = b"\xff" * mod.PR101_DECODER_BLOB_LEN
    out = tmp_path / "subst.zip"
    mod.substitute_decoder_blob(
        input_archive=archive,
        replacement_decoder_blob=new_decoder,
        output_archive=out,
    )
    output_inner = mod._read_inner_blob(out)
    decoder, latent, sidecar = mod._split_pr101_inner_blob(output_inner)
    assert decoder == new_decoder
    assert latent == b"\xb2" * mod.PR101_LATENT_BLOB_LEN
    assert sidecar == b"\xc3" * 607


def test_substitute_latent_blob_changes_latent_only(tmp_path) -> None:
    mod = _load_tool_module()
    archive = _make_synthetic_pr101_archive(tmp_path)
    new_latent = b"\xee" * mod.PR101_LATENT_BLOB_LEN
    out = tmp_path / "latent_subst.zip"

    report = mod.substitute_latent_blob(
        input_archive=archive,
        replacement_latent_blob=new_latent,
        output_archive=out,
    )

    output_inner = mod._read_inner_blob(out)
    decoder, latent, sidecar = mod._split_pr101_inner_blob(output_inner)
    assert decoder == b"\xa1" * mod.PR101_DECODER_BLOB_LEN
    assert latent == new_latent
    assert sidecar == b"\xc3" * 607
    assert report.latent_blob_len == mod.PR101_LATENT_BLOB_LEN
    assert "latent_blob substituted" in report.notes[0]


def test_substitute_latent_blob_rejects_wrong_length(tmp_path) -> None:
    mod = _load_tool_module()
    archive = _make_synthetic_pr101_archive(tmp_path)

    with pytest.raises(ValueError, match=r"replacement_latent_blob length .+ != LATENT_BLOB_LEN"):
        mod.substitute_latent_blob(
            input_archive=archive,
            replacement_latent_blob=b"\xee" * (mod.PR101_LATENT_BLOB_LEN - 1),
            output_archive=tmp_path / "latent_bad.zip",
        )


def test_substitute_sidecar_blob_changes_tail_only_and_may_change_length(tmp_path) -> None:
    mod = _load_tool_module()
    archive = _make_synthetic_pr101_archive(tmp_path)
    new_sidecar = b"\x99" * 128
    out = tmp_path / "sidecar_subst.zip"

    report = mod.substitute_sidecar_blob(
        input_archive=archive,
        replacement_sidecar_blob=new_sidecar,
        output_archive=out,
    )

    output_inner = mod._read_inner_blob(out)
    decoder, latent, sidecar = mod._split_pr101_inner_blob(output_inner)
    assert decoder == b"\xa1" * mod.PR101_DECODER_BLOB_LEN
    assert latent == b"\xb2" * mod.PR101_LATENT_BLOB_LEN
    assert sidecar == new_sidecar
    assert report.sidecar_blob_len == len(new_sidecar)
    assert report.inner_member_output_size == (
        mod.PR101_DECODER_BLOB_LEN + mod.PR101_LATENT_BLOB_LEN + len(new_sidecar)
    )
    assert "sidecar_blob substituted" in report.notes[0]


def test_substitute_sidecar_blob_rejects_empty_replacement(tmp_path) -> None:
    mod = _load_tool_module()
    archive = _make_synthetic_pr101_archive(tmp_path)

    with pytest.raises(ValueError, match="replacement_sidecar_blob is empty"):
        mod.substitute_sidecar_blob(
            input_archive=archive,
            replacement_sidecar_blob=b"",
            output_archive=tmp_path / "sidecar_bad.zip",
        )


def test_substitute_rejects_wrong_length_replacement(tmp_path) -> None:
    """Substituting a wrong-length blob would corrupt PR101's inflate
    (latent_blob extraction uses a fixed offset). Defensive guard
    rejects it loudly."""
    mod = _load_tool_module()
    archive = _make_synthetic_pr101_archive(tmp_path)
    wrong_size = b"\xff" * (mod.PR101_DECODER_BLOB_LEN - 100)  # too short
    out = tmp_path / "should_not_write.zip"
    with pytest.raises(ValueError, match=r"length .+ != DECODER_BLOB_LEN"):
        mod.substitute_decoder_blob(
            input_archive=archive,
            replacement_decoder_blob=wrong_size,
            output_archive=out,
        )
    assert not out.exists()


# ---------------------------------------------------------------------------
# Verify subcommand
# ---------------------------------------------------------------------------

def test_verify_byte_layout_reports_structured_layout(tmp_path) -> None:
    mod = _load_tool_module()
    archive = _make_synthetic_pr101_archive(tmp_path)
    layout = mod.verify_byte_layout(archive)
    assert layout["decoder_blob"]["matches_expected"] is True
    assert layout["latent_blob"]["matches_expected"] is True
    assert layout["sidecar_blob"]["len"] == 607
    assert layout["inner_blob_size"] == (
        mod.PR101_DECODER_BLOB_LEN + mod.PR101_LATENT_BLOB_LEN + 607
    )


def test_verify_real_pr101_archive_when_present() -> None:
    """If the canonical PR101 intake archive is on disk, verify its layout
    matches the documented PR101 byte-offsets. Skipped when not present so
    the test suite stays portable."""
    mod = _load_tool_module()
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    canonical = (
        repo_root
        / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
    )
    if not canonical.is_file():
        pytest.skip("canonical PR101 intake not present")
    layout = mod.verify_byte_layout(canonical)
    assert layout["decoder_blob"]["matches_expected"] is True
    assert layout["latent_blob"]["matches_expected"] is True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_verify_emits_json(tmp_path, capsys) -> None:
    mod = _load_tool_module()
    archive = _make_synthetic_pr101_archive(tmp_path)
    rc = mod.main(["verify", "--archive", str(archive)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["decoder_blob"]["matches_expected"] is True


def test_substitute_latent_blob_preserves_decoder_and_sidecar(tmp_path) -> None:
    """Latent substitution: decoder + sidecar sections must be byte-faithfully
    preserved; only the latent slice changes."""
    mod = _load_tool_module()
    archive = _make_synthetic_pr101_archive(tmp_path)
    new_latent = b"\xee" * mod.PR101_LATENT_BLOB_LEN
    out = tmp_path / "latent_subst.zip"
    report = mod.substitute_latent_blob(
        input_archive=archive,
        replacement_latent_blob=new_latent,
        output_archive=out,
    )
    output_inner = mod._read_inner_blob(out)
    decoder, latent, sidecar = mod._split_pr101_inner_blob(output_inner)
    assert decoder == b"\xa1" * mod.PR101_DECODER_BLOB_LEN
    assert latent == new_latent
    assert sidecar == b"\xc3" * 607
    assert report.latent_blob_len == mod.PR101_LATENT_BLOB_LEN


def test_substitute_latent_rejects_wrong_length(tmp_path) -> None:
    mod = _load_tool_module()
    archive = _make_synthetic_pr101_archive(tmp_path)
    out = tmp_path / "should_not_write.zip"
    with pytest.raises(ValueError, match=r"length .+ != LATENT_BLOB_LEN"):
        mod.substitute_latent_blob(
            input_archive=archive,
            replacement_latent_blob=b"\xff" * (mod.PR101_LATENT_BLOB_LEN - 100),
            output_archive=out,
        )
    assert not out.exists()


def test_substitute_sidecar_preserves_decoder_and_latent(tmp_path) -> None:
    """Sidecar substitution: decoder + latent sections preserved; sidecar
    can be ANY non-empty length (it's the tail section). Verifies a
    different-length sidecar replaces correctly + bytes_delta reflects
    the size change."""
    mod = _load_tool_module()
    archive = _make_synthetic_pr101_archive(tmp_path)
    # Use a SHORTER sidecar (500 bytes vs default 607) to test variable length
    new_sidecar = b"\xdd" * 500
    out = tmp_path / "sidecar_subst.zip"
    report = mod.substitute_sidecar_blob(
        input_archive=archive,
        replacement_sidecar_blob=new_sidecar,
        output_archive=out,
    )
    output_inner = mod._read_inner_blob(out)
    decoder, latent, sidecar = mod._split_pr101_inner_blob(output_inner)
    assert decoder == b"\xa1" * mod.PR101_DECODER_BLOB_LEN
    assert latent == b"\xb2" * mod.PR101_LATENT_BLOB_LEN
    assert sidecar == new_sidecar
    assert report.sidecar_blob_len == 500
    # Sidecar shrinkage by 107 bytes → archive shrinks by ~107 (modulo ZIP overhead)
    assert report.bytes_delta < 0


def test_substitute_sidecar_rejects_empty(tmp_path) -> None:
    mod = _load_tool_module()
    archive = _make_synthetic_pr101_archive(tmp_path)
    out = tmp_path / "empty_sidecar.zip"
    with pytest.raises(ValueError, match="sidecar_blob is empty"):
        mod.substitute_sidecar_blob(
            input_archive=archive,
            replacement_sidecar_blob=b"",
            output_archive=out,
        )
    assert not out.exists()


def test_bug_hunter_substitute_decoder_with_too_long_blob(tmp_path) -> None:
    """Adversarial: pass a blob LONGER than DECODER_BLOB_LEN. Must reject
    (corruption guard). This catches a class of bugs where caller code
    accidentally wraps the blob in extra metadata bytes."""
    mod = _load_tool_module()
    archive = _make_synthetic_pr101_archive(tmp_path)
    too_long = b"\xff" * (mod.PR101_DECODER_BLOB_LEN + 100)
    out = tmp_path / "should_not_write.zip"
    with pytest.raises(ValueError, match=r"length .+ != DECODER_BLOB_LEN"):
        mod.substitute_decoder_blob(
            input_archive=archive,
            replacement_decoder_blob=too_long,
            output_archive=out,
        )
    assert not out.exists()


def test_bug_hunter_corrupt_input_archive(tmp_path) -> None:
    """Adversarial: input archive has the right structure but inner blob
    is too short to even contain the fixed decoder + latent sections.
    Must surface the error clearly."""
    mod = _load_tool_module()
    short_archive = tmp_path / "too_short.zip"
    info = zipfile.ZipInfo(filename=mod.PR101_INNER_MEMBER_NAME)
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(short_archive, "w") as zf:
        zf.writestr(info, b"\x00" * 1000)  # < DECODER + LATENT minimum
    out = tmp_path / "out.zip"
    with pytest.raises(ValueError, match=r"< required minimum"):
        mod.substitute_decoder_blob(
            input_archive=short_archive,
            replacement_decoder_blob=b"\xff" * mod.PR101_DECODER_BLOB_LEN,
            output_archive=out,
        )


def test_byte_faithful_roundtrip_through_real_pr101_codec(tmp_path) -> None:
    """End-to-end smoke: real PR101 archive → decode_decoder_compact →
    encode_decoder_compact (PR101 defaults) → SHA-faithful to original.

    This is the keystone test proving the cathedral's PR101 codec is
    bit-faithful with PR101's native codec on PR101's own substrate.
    Skipped when the canonical intake archive is not on disk so the
    test suite stays portable.
    """
    mod = _load_tool_module()
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    canonical = (
        repo_root
        / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
    )
    if not canonical.is_file():
        pytest.skip("canonical PR101 intake not present")

    import hashlib

    from tac.pr101_split_brotli_codec import decode_decoder_compact, encode_decoder_compact

    inner = mod._read_inner_blob(canonical)
    decoder_blob, _latent, _sidecar = mod._split_pr101_inner_blob(inner)
    state_dict = decode_decoder_compact(decoder_blob)
    re_encoded = encode_decoder_compact(state_dict, brotli_quality=11)
    assert re_encoded == decoder_blob, (
        "cathedral re-encode of PR101's own state_dict must produce "
        "byte-identical decoder_blob (PR101 defaults)"
    )
    # SHA cross-check (would catch silent length-only-equal but bytes-differ)
    assert hashlib.sha256(re_encoded).hexdigest() == hashlib.sha256(decoder_blob).hexdigest()

    # Substitute the re-encoded blob back through the surgery tool;
    # output archive must be a valid PR101 archive and the sliced
    # decoder section must SHA-equal the original.
    out = tmp_path / "roundtrip_pr101.zip"
    report = mod.substitute_decoder_blob(
        input_archive=canonical,
        replacement_decoder_blob=re_encoded,
        output_archive=out,
    )
    assert report.sha256_input_decoder_blob == report.sha256_replacement_decoder_blob
    assert report.inner_member_name == "x"
    assert report.sha256_input_latent_blob == report.sha256_output_latent_blob
    assert report.sha256_input_sidecar_blob == report.sha256_output_sidecar_blob
    out_inner = mod._read_inner_blob(out)
    out_dec, out_lat, out_side = mod._split_pr101_inner_blob(out_inner)
    assert out_dec == decoder_blob
    assert out_lat == _latent
    assert out_side == _sidecar


def test_real_pr101_lgwin_variant_materializes_same_length_different_decoder(
    tmp_path,
) -> None:
    """PR101 can carry a byte-different Brotli encoding of the same decoder raw
    tensor stream while preserving the fixed 162,164-byte decoder slice."""
    mod = _load_tool_module()
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    canonical = (
        repo_root
        / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
    )
    if not canonical.is_file():
        pytest.skip("canonical PR101 intake not present")

    from tac.pr101_split_brotli_codec import decode_decoder_compact, encode_decoder_compact

    inner = mod._read_inner_blob(canonical)
    decoder_blob, _latent, _sidecar = mod._split_pr101_inner_blob(inner)
    state_dict = decode_decoder_compact(decoder_blob)
    lgwin_blob = encode_decoder_compact(
        state_dict,
        brotli_quality=11,
        brotli_lgwin=18,
    )
    assert len(lgwin_blob) == mod.PR101_DECODER_BLOB_LEN
    assert lgwin_blob != decoder_blob
    # Decodes to the same tensor stream under the stock PR101 byte-map contract.
    assert encode_decoder_compact(decode_decoder_compact(lgwin_blob)) == decoder_blob

    out = tmp_path / "lgwin18_pr101.zip"
    report = mod.substitute_decoder_blob(
        input_archive=canonical,
        replacement_decoder_blob=lgwin_blob,
        output_archive=out,
    )
    assert report.output_size_bytes == report.input_size_bytes
    assert report.sha256_output_archive != report.sha256_input_archive
    assert report.sha256_replacement_decoder_blob != report.sha256_input_decoder_blob
    assert report.sha256_input_latent_blob == report.sha256_output_latent_blob
    assert report.sha256_input_sidecar_blob == report.sha256_output_sidecar_blob


def test_cli_substitute_writes_report(tmp_path, capsys) -> None:
    mod = _load_tool_module()
    archive = _make_synthetic_pr101_archive(tmp_path)
    blob_path = tmp_path / "blob.bin"
    blob_path.write_bytes(b"\xff" * mod.PR101_DECODER_BLOB_LEN)
    out_archive = tmp_path / "out.zip"
    report_path = tmp_path / "report.json"
    rc = mod.main([
        "substitute",
        "--input-archive", str(archive),
        "--replacement-decoder-blob", str(blob_path),
        "--output-archive", str(out_archive),
        "--report", str(report_path),
    ])
    assert rc == 0
    report = json.loads(report_path.read_text())
    assert report["decoder_blob_replacement_len"] == mod.PR101_DECODER_BLOB_LEN
    assert report["sha256_input_archive"] != report["sha256_output_archive"]
