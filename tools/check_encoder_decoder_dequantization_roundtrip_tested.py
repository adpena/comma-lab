#!/usr/bin/env python3
"""B1 — Encoder/decoder dequantization roundtrip test scanner.

Bug class: encoder uses ``(rounded / N) * scale`` while paired runtime decoder
uses ``rounded * scale`` (or any other quantization-arithmetic mismatch).
Real instance: ``tools/pr101_cross_paradigm_hstack_vstack_empirical.py`` had
``(rounded / 127) * scale`` while runtime decoder did ``rounded * scale`` —
phantom 137,531 B claim that re-runs at 137,469 B with the bug fixed.

Detection rule: any tool with both an encoder AND a paired inflate.py SHOULD
have a roundtrip test that ENCODES → INFLATES → COMPARES bytewise (mod
known quantization) for at least one sample. This scanner walks
``tools/build_*.py`` and ``tools/pr101_*_empirical.py`` plus
``experiments/build_*.py`` and looks for AST patterns suggesting:

  * The tool emits a quantization codec (presence of ``scale`` * ``rounded``
    arithmetic, OR explicit ``q = (x / scale).round()`` patterns), AND
  * The tool produces an archive (writes a ``.zip`` or ``archive.zip``).

A scanned tool is CLEAN when EITHER:
  * the file contains a comment marker
    ``# ROUNDTRIP_TESTED: <pytest path>`` linking to a pytest that actually
    runs encode → inflate → bytewise-compare on at least one sample, OR
  * a sibling pytest file ``src/tac/tests/test_<tool_basename>_roundtrip.py``
    or ``src/tac/tests/test_<tool_basename>.py`` exists and contains the
    string ``ENCODE_INFLATE_ROUNDTRIP`` or ``encode_inflate_roundtrip``, OR
  * the tool itself contains a ``ROUNDTRIP_SELF_TEST`` block (idiomatic
    self-test marker).

Memory ref: ``feedback_codex_adversarial_review_4_landings_20260508.md``.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

SCANNED_GLOBS: tuple[str, ...] = (
    "tools/pr101_*_empirical.py",
    "tools/build_admm_*.py",
    "experiments/build_admm_*.py",
    "experiments/build_apogee_*.py",
)

# Tool is exempt if it lacks any quantization arithmetic — pure orchestrators
# that don't quantize don't need encoder/decoder roundtrip coverage. We use a
# coarse heuristic: presence of `.round()` or `np.round` AND presence of
# `scale` or `* scale` as a code identifier.
QUANT_TRIGGER_RE = re.compile(
    r"\.round\(\)|\bnp\.round\b|round_to_int|round_half_to_even"
)
SCALE_TRIGGER_RE = re.compile(r"\bscale\b|/\s*\d+\s*\*|/\s*scale|\*\s*scale")
ARCHIVE_TRIGGER_RE = re.compile(
    r"archive\.zip|\.zip['\"]|ZipFile|zipfile\.ZipFile"
)

ROUNDTRIP_MARKER_RE = re.compile(
    r"#\s*ROUNDTRIP_TESTED\s*:\s*(\S+)|#\s*ROUNDTRIP_SELF_TEST",
    re.IGNORECASE,
)
ROUNDTRIP_SIBLING_TOKENS = (
    "ENCODE_INFLATE_ROUNDTRIP",
    "encode_inflate_roundtrip",
)


@dataclass
class Finding:
    rel_path: str
    reason: str


def _has_quant_arithmetic(text: str) -> bool:
    return bool(QUANT_TRIGGER_RE.search(text)) and bool(
        SCALE_TRIGGER_RE.search(text)
    )


def _has_archive_emit(text: str) -> bool:
    return bool(ARCHIVE_TRIGGER_RE.search(text))


def _has_inline_roundtrip_marker(text: str) -> bool:
    return bool(ROUNDTRIP_MARKER_RE.search(text))


def _has_sibling_pytest_with_roundtrip(repo: Path, tool_path: Path) -> bool:
    base = tool_path.stem
    candidates = [
        repo / "src" / "tac" / "tests" / f"test_{base}_roundtrip.py",
        repo / "src" / "tac" / "tests" / f"test_{base}.py",
        repo / "src" / "tac" / "tests" / "preflight" / f"test_{base}_roundtrip.py",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            t = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(token in t for token in ROUNDTRIP_SIBLING_TOKENS):
            return True
    return False


def scan(repo_root: Path | None = None) -> list[Finding]:
    repo = repo_root or REPO_ROOT_DEFAULT
    findings: list[Finding] = []
    for glob in SCANNED_GLOBS:
        for tool in sorted(repo.glob(glob)):
            if not tool.is_file():
                continue
            try:
                text = tool.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            # Skip files without quant arithmetic AND archive emit — they're
            # not encoder/decoder paired tools.
            if not (_has_quant_arithmetic(text) and _has_archive_emit(text)):
                continue
            if _has_inline_roundtrip_marker(text):
                continue
            if _has_sibling_pytest_with_roundtrip(repo, tool):
                continue
            rel = tool.relative_to(repo)
            findings.append(
                Finding(
                    rel_path=str(rel),
                    reason=(
                        "encoder+archive emit but no roundtrip-test marker "
                        "(`# ROUNDTRIP_TESTED:<pytest>` or sibling pytest "
                        "with `ENCODE_INFLATE_ROUNDTRIP` token). Bug class: "
                        "encoder/decoder dequantization arithmetic mismatch "
                        "(B1)."
                    ),
                )
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    findings = scan(repo)
    if findings:
        print(
            f"[B1-encoder-decoder-roundtrip] {len(findings)} violation(s):",
            file=sys.stderr,
        )
        for f in findings:
            print(f"  • {f.rel_path}: {f.reason}", file=sys.stderr)
        if args.strict:
            return 1
    else:
        print("[B1-encoder-decoder-roundtrip] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
