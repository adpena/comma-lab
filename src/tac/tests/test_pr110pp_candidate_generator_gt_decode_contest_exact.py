#!/usr/bin/env python
"""STRICT regression guard (tracked): pr110pp candidate-generator GT decode is contest-exact.

This is the DURABLE, version-controlled second landing of the two-landing bug discipline
(CLAUDE.md "Bugs must be permanently fixed AND self-protected against") for the GT-decode
apples-to-apples bug class (R3 §0,
``.omx/research/pr110pp_r3_onhost_selector_verdict_20260610.md``).

Bug class: the contest ground-truth decode is ``frame_utils.yuv420_to_rgb`` (BT.601
limited-range + bilinear chroma upsampling, "matches nvdec"), NOT PyAV
``frame.to_ndarray(format="rgb24")`` (libswscale, a DIFFERENT YUV->RGB conversion). The
rgb24 path inflated absolute pose ~100x (incumbent per-pair mean 3.31e-3 vs the contest
frontier avg_posenet_dist 2.943e-05) and manufactured 591/600 spurious "improvable" pairs
(vs the true 7) — a substrate-invalidating measurement bug that mis-ranks every per-mode /
per-region / multi-mode / decoder-axis candidate generated against the wrong GT.

The pr110pp candidate-generator family lives under the gitignored ``experiments/results/``
custody tree, so the in-tree fast copy of this guard lives next to the lib. THIS tracked
copy survives a fresh checkout and runs in CI: it SCANS the family on disk and refuses any
executable ``to_ndarray(format="rgb24")`` GT decode. When the ignored custody tree is
absent (fresh CI checkout with no experiments/results), the scan is vacuously satisfied
(the surfaces it would protect do not exist), so the guard SKIPS rather than failing —
this is the correct fail-open-on-absence behavior for a guard over an ignored custody tree.

Run: .venv/bin/python -m pytest src/tac/tests/test_pr110pp_candidate_generator_gt_decode_contest_exact.py -q
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for _ in range(12):
        if (p / "experiments").is_dir() and (p / "upstream").is_dir():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[3]


REPO_ROOT = _repo_root()

PR110PP_FAMILY_DIRS = [
    REPO_ROOT / "experiments/results/pr110pp_r2_nonmps_candidate_20260609/analysis",
    REPO_ROOT / "experiments/results/pr110pp_r3_onhost_mode_table_20260610",
]

_RGB24_DECODE_RE = re.compile(r"""to_ndarray\s*\(\s*format\s*=\s*['"]rgb24['"]""")
_YUV420_RE = re.compile(r"yuv420_to_rgb")


def _iter_family_py_files():
    for d in PR110PP_FAMILY_DIRS:
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.py")):
            if "__pycache__" in f.parts or f.name.startswith("test_"):
                continue
            yield f


def _strip_comments(src: str) -> str:
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


def _executable_rgb24_gt_sites(path: Path) -> list[int]:
    """Line numbers of executable rgb24 GT-decode calls (comments/docstrings excluded)."""
    import ast

    src = path.read_text()
    try:
        tree = ast.parse(src)
        docs = {
            n.value.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        }
    except SyntaxError:
        docs = set()
    code = _strip_comments(src)
    for d in docs:
        code = code.replace(d, "")
    return [i for i, ln in enumerate(code.splitlines(), 1) if _RGB24_DECODE_RE.search(ln)]


def _family_present() -> bool:
    return any(d.is_dir() for d in PR110PP_FAMILY_DIRS)


def test_no_executable_rgb24_gt_decode_in_pr110pp_family():
    if not _family_present():
        pytest.skip("pr110pp candidate-generator family absent (ignored custody tree)")
    offenders = {}
    for f in _iter_family_py_files():
        sites = _executable_rgb24_gt_sites(f)
        if sites:
            offenders[str(f.relative_to(REPO_ROOT))] = sites
    assert not offenders, (
        "rgb24 GT-decode bug re-introduced (R3 §0). Contest GT decode MUST be "
        f"frame_utils.yuv420_to_rgb, not PyAV to_ndarray(format='rgb24'). Offenders: {offenders}"
    )


def test_every_gt_decode_surface_references_contest_decode():
    if not _family_present():
        pytest.skip("pr110pp candidate-generator family absent (ignored custody tree)")
    gt_files = []
    for f in _iter_family_py_files():
        src = f.read_text()
        if (("GT_VIDEO" in src) or ("0.mkv" in src) or ("decode_gt" in src)) and (
            ("to_ndarray(" in src) or ("yuv420_to_rgb" in src)
        ):
            gt_files.append(f)
    # The R2 lib is the canonical GT-decode surface; the scanner must find it (non-vacuous).
    assert any(f.name == "render_and_score_lib.py" for f in gt_files), (
        f"scanner found no render_and_score_lib.py GT-decode surface; found: {[f.name for f in gt_files]}"
    )
    bad = {
        str(f.relative_to(REPO_ROOT)): {
            "executable_rgb24_sites": _executable_rgb24_gt_sites(f),
            "references_yuv420_to_rgb": bool(_YUV420_RE.search(f.read_text())),
        }
        for f in gt_files
        if _executable_rgb24_gt_sites(f) or not _YUV420_RE.search(f.read_text())
    }
    assert not bad, f"GT-decode surface(s) not contest-exact: {bad}"


if __name__ == "__main__":
    test_no_executable_rgb24_gt_decode_in_pr110pp_family()
    test_every_gt_decode_surface_references_contest_decode()
    print("ALL TRACKED GT-DECODE REGRESSION GUARDS PASSED")
