"""Lane B-alt 2026-04-27: build_baseline_archive --use-brotli wiring tests.

Verifies the brotli flag is exposed, threaded through to the archive build,
recorded in provenance, and that the resulting .br round-trips through the
inflate-side decompress helper. Real archive builds require CUDA + GT video,
so we test the seams via subprocess argparse + direct helper invocation.

Memory: project_lane_b_alt_brotli_measurement.md — local measurement
showed -0.023 score on a 296KB renderer at q=11. Inflate side already
auto-decompresses .br files via decompress_brotli_files_in_dir.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_baseline_archive.py"


def test_use_brotli_flag_exists_in_argparse() -> None:
    """The --use-brotli + --brotli-quality flags must be exposed via argparse.
    Caught by the dead-flag preflight (CLAUDE.md non-negotiable) but pinned
    here so a future refactor cannot silently drop them."""
    src = SCRIPT.read_text()
    assert '"--use-brotli"' in src, "--use-brotli flag missing from argparse"
    assert '"--brotli-quality"' in src, "--brotli-quality flag missing"


def test_use_brotli_help_text_documents_inflate_side_compat() -> None:
    """Help text must reference the inflate-side auto-decompression so
    operators don't worry about needing a custom inflate path."""
    src = SCRIPT.read_text()
    assert "auto-" in src.lower() and "decompress" in src.lower(), (
        "--use-brotli help text must explain inflate-side auto-decompress"
    )


def test_use_brotli_changes_arcname_to_renderer_bin_br() -> None:
    """When --use-brotli is set, the archive entry name must be
    renderer.bin.br (NOT renderer.bin) — this is what the inflate-side
    decompress_brotli_files_in_dir helper greps for."""
    src = SCRIPT.read_text()
    assert "renderer.bin.br" in src, (
        "Brotli arcname renderer.bin.br missing — inflate-side auto-"
        "decompress requires the .br extension"
    )
    # Verify the wiring uses dynamic arcname, not hardcoded.
    assert "renderer_arcname" in src, (
        "Brotli wiring must use a dynamic arcname variable, not hardcoded"
    )


def test_brotli_round_trip_preserves_renderer_bytes(tmp_path: Path) -> None:
    """End-to-end: compress with the same helper build_baseline_archive uses,
    then decompress with the same helper inflate uses, and assert the bytes
    match exactly. This is the contract the inflate side relies on."""
    from tac.submission_archive import (
        compress_file_brotli,
        decompress_brotli_files_in_dir,
    )
    src = tmp_path / "renderer.bin"
    # Use semi-compressible data so we get a real compression ratio (not
    # pathological all-zeros).
    payload = bytes(range(256)) * 1024  # 256KB structured data
    src.write_bytes(payload)
    br = tmp_path / "renderer.bin.br"
    compress_file_brotli(src, br, quality=11)
    assert br.stat().st_size < src.stat().st_size, (
        "Brotli should compress structured data"
    )

    # Move .br into a fake extracted-archive dir, run inflate-side helper.
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    shutil.copy(br, extract_dir / "renderer.bin.br")
    n = decompress_brotli_files_in_dir(extract_dir)
    assert n == 1
    out = extract_dir / "renderer.bin"
    assert out.exists(), "decompress_brotli_files_in_dir must produce renderer.bin"
    assert out.read_bytes() == payload, "round-trip byte mismatch"


def test_use_brotli_help_string_via_subprocess() -> None:
    """Subprocess --help must list --use-brotli (the canonical user-facing
    contract). Catches the case where the flag is registered but accidentally
    in a hidden arg group."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"--help exited {proc.returncode}: {proc.stderr}"
    assert "--use-brotli" in proc.stdout, (
        "--use-brotli not visible in --help output"
    )
    assert "--brotli-quality" in proc.stdout, (
        "--brotli-quality not visible in --help output"
    )


def test_provenance_records_brotli_settings() -> None:
    """The provenance JSON must record use_brotli + brotli_quality so any
    downstream auditor can verify the archive was built with the expected
    compression. Source-level pin (full archive build needs CUDA)."""
    src = SCRIPT.read_text()
    assert '"use_brotli": bool(args.use_brotli)' in src, (
        "provenance must record use_brotli flag"
    )
    assert '"brotli_quality"' in src, (
        "provenance must record brotli_quality (or None when not used)"
    )


def test_provenance_renderer_entry_records_compression() -> None:
    """The renderer.bin (or .br) component entry in provenance must record
    the compression method so a future re-run can detect drift if the
    build re-encoded with different settings."""
    src = SCRIPT.read_text()
    # Look for the compression metadata field.
    assert '"compression":' in src, (
        "renderer component entry must record compression method"
    )
    assert "brotli-q" in src, (
        "compression label must be parseable (e.g. 'brotli-q11' or 'none')"
    )
    assert "size_bytes_uncompressed" in src and "size_bytes_in_archive" in src, (
        "renderer component must distinguish raw vs in-archive size"
    )
