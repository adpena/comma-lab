# SPDX-License-Identifier: MIT
"""Tests for Check 91: Lane 20 (Ballé hyperprior) BHv1 wire-format integrity.

Per CLAUDE.md non-negotiables applied to the Lane 20 production codec at
``src/tac/balle_hyperprior_codec.py``:

1. ``encode_qints_full_balle`` MUST serialize hyper_decoder weights into
   side_info (otherwise inflate-side decode silently drifts because the
   FP16 weight roundtrip changes σ values — debugged 2026-04-30 Phase B).
2. ``encode_qints_balle_auto`` MUST keep the static_baseline_bytes guard
   so a regressing untrained codec is never shipped — the kill criterion
   from Phase A council review §3.

Reference: .omx/research/council_lane_20_balle_design_20260430.md
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.preflight import (  # noqa: E402
    MetaBugViolation,
    check_balle_hyperprior_includes_side_info_in_archive,
)


# ─────────────────────────────────────────────────────────────────────────────
# Real-codebase regression
# ─────────────────────────────────────────────────────────────────────────────


def test_real_codebase_passes_strict() -> None:
    """[regression] Real codebase has 0 Lane 20 BHv1 integrity violations."""
    v = check_balle_hyperprior_includes_side_info_in_archive(
        strict=False, verbose=False
    )
    assert v == [], f"Real codebase should be clean; got {len(v)} violations: {v}"


def test_strict_real_codebase_does_not_raise() -> None:
    """[regression] strict=True on real codebase passes without raising."""
    check_balle_hyperprior_includes_side_info_in_archive(strict=True, verbose=False)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic counter-examples
# ─────────────────────────────────────────────────────────────────────────────


def _write_synth(tmp: Path, codec_body: str) -> Path:
    """Skeleton repo with src/tac/balle_hyperprior_codec.py."""
    (tmp / "src" / "tac").mkdir(parents=True, exist_ok=True)
    (tmp / "src" / "tac" / "balle_hyperprior_codec.py").write_text(
        textwrap.dedent(codec_body).strip() + "\n"
    )
    return tmp


def test_missing_decoder_serialization_caught(tmp_path: Path) -> None:
    """Codec where encode_qints_full_balle does NOT serialize hyper_decoder
    weights into side_info → flagged."""
    repo = _write_synth(
        tmp_path,
        codec_body="""
            def encode_qints_full_balle(*, qints, num_symbols, offset, codec):
                # BUG: never calls _serialize_hyper_decoder
                # BUG: side_info has no decoder bytes → decode would drift
                return b""

            def encode_qints_balle_auto(*, qints, num_symbols, offset,
                                        num_chunks_lite, full_codec,
                                        static_baseline_bytes=None):
                if static_baseline_bytes is not None:
                    return (b"", "static_wins", {})
                return (b"X", "hotz_lite", {})
        """,
    )
    v = check_balle_hyperprior_includes_side_info_in_archive(
        strict=False, verbose=False, repo_root=repo,
    )
    assert any("_serialize_hyper_decoder" in line for line in v), (
        f"Expected serializer-missing violation; got {v}"
    )


def test_missing_side_info_write_caught(tmp_path: Path) -> None:
    """Codec where encode_qints_full_balle calls _serialize_hyper_decoder
    but never writes the bytes into side_info → flagged."""
    repo = _write_synth(
        tmp_path,
        codec_body="""
            def encode_qints_full_balle(*, qints, num_symbols, offset, codec):
                decoder_blob = _serialize_hyper_decoder(codec.hyper_decoder)
                # BUG: we serialized but never wrote it to side_info
                side_info = io.BytesIO()
                side_info.write(b"some other bytes")
                return side_info.getvalue()

            def encode_qints_balle_auto(*, qints, num_symbols, offset,
                                        num_chunks_lite, full_codec,
                                        static_baseline_bytes=None):
                return (b"", "static_wins", {})
        """,
    )
    v = check_balle_hyperprior_includes_side_info_in_archive(
        strict=False, verbose=False, repo_root=repo,
    )
    # The violation message describes that the decoder bytes must be
    # written into side_info — keyword cluster (case-sensitive code mention).
    assert any(
        "must be written into" in line and "side_info" in line for line in v
    ), f"Expected missing-side_info-write violation; got {v}"


def test_missing_static_baseline_guard_caught(tmp_path: Path) -> None:
    """Codec where encode_qints_balle_auto has no static_baseline_bytes
    parameter → flagged (would ship regressing codec)."""
    repo = _write_synth(
        tmp_path,
        codec_body="""
            def encode_qints_full_balle(*, qints, num_symbols, offset, codec):
                decoder_blob = _serialize_hyper_decoder(codec.hyper_decoder)
                side_info = io.BytesIO()
                side_info.write(decoder_blob)
                return side_info.getvalue()

            def encode_qints_balle_auto(*, qints, num_symbols, offset,
                                        num_chunks_lite, full_codec):
                # BUG: no static_baseline_bytes parameter; will ship
                # regressing untrained codec without the kill criterion.
                return (b"X", "hotz_lite", {})
        """,
    )
    v = check_balle_hyperprior_includes_side_info_in_archive(
        strict=False, verbose=False, repo_root=repo,
    )
    assert any("static_baseline_bytes" in line for line in v), (
        f"Expected missing-baseline-guard violation; got {v}"
    )


def test_missing_static_wins_sentinel_caught(tmp_path: Path) -> None:
    """Codec with static_baseline_bytes parameter but no static_wins
    sentinel return path → flagged."""
    repo = _write_synth(
        tmp_path,
        codec_body="""
            def encode_qints_full_balle(*, qints, num_symbols, offset, codec):
                decoder_blob = _serialize_hyper_decoder(codec.hyper_decoder)
                side_info = io.BytesIO()
                side_info.write(decoder_blob)
                return side_info.getvalue()

            def encode_qints_balle_auto(*, qints, num_symbols, offset,
                                        num_chunks_lite, full_codec,
                                        static_baseline_bytes=None):
                # BUG: param exists but never returns static_wins sentinel
                return (b"X", "hotz_lite", {"static_baseline_bytes": 0})
        """,
    )
    v = check_balle_hyperprior_includes_side_info_in_archive(
        strict=False, verbose=False, repo_root=repo,
    )
    assert any("static_wins" in line for line in v), (
        f"Expected missing-static_wins-sentinel violation; got {v}"
    )


def test_strict_raises_on_violations(tmp_path: Path) -> None:
    """strict=True raises MetaBugViolation when scan finds violations."""
    repo = _write_synth(
        tmp_path,
        codec_body="""
            def encode_qints_full_balle(*, qints, num_symbols, offset, codec):
                # missing both serialization AND side_info write
                return b""

            def encode_qints_balle_auto(*, qints, num_symbols, offset, num_chunks_lite, full_codec):
                # missing baseline guard
                return (b"X", "hotz_lite", {})
        """,
    )
    with pytest.raises(MetaBugViolation, match="LANE 20 BHv1 WIRE-FORMAT INTEGRITY"):
        check_balle_hyperprior_includes_side_info_in_archive(
            strict=True, verbose=False, repo_root=repo,
        )


def test_clean_synth_passes(tmp_path: Path) -> None:
    """A correctly-shaped synthetic codec module → 0 violations."""
    repo = _write_synth(
        tmp_path,
        codec_body="""
            def encode_qints_full_balle(*, qints, num_symbols, offset, codec):
                decoder_blob = _serialize_hyper_decoder(codec.hyper_decoder)
                side_info = io.BytesIO()
                side_info.write(decoder_blob)
                return side_info.getvalue()

            def encode_qints_balle_auto(*, qints, num_symbols, offset, num_chunks_lite, full_codec, static_baseline_bytes=None):
                if static_baseline_bytes is not None:
                    return (b"", "static_wins", {})
                return (b"X", "hotz_lite", {})
        """,
    )
    v = check_balle_hyperprior_includes_side_info_in_archive(
        strict=False, verbose=False, repo_root=repo,
    )
    assert v == [], f"Clean synth should pass; got {v}"


def test_no_balle_codec_no_violations(tmp_path: Path) -> None:
    """If the file does not exist, no violations are produced."""
    (tmp_path / "src" / "tac").mkdir(parents=True, exist_ok=True)
    v = check_balle_hyperprior_includes_side_info_in_archive(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert v == []
