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
