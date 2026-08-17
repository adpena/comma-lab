# SPDX-License-Identifier: MIT
"""Measurement-integrity foundation for witness d_seg / d_pose authority.

Operator paranoia 2026-06-26 (the re-audit re-founding,
``.omx/research/reaudit_refounding_and_md_decoupling_20260626.md``): 3 of 9
witness measurement surfaces are NON-EXACT and were silently quoted as
"d_seg SCORE". The deepmath-smoke + lever-B + byte-closer-advisory compute a
PROXY d_seg = the generator's OWN per-pixel argmax vs the GT SegNet argmax,
SKIPPING the contest's reconstruction operator R (bicubic-up -> uint8 @ camera
-> bilinear-down) AND the SegNet RE-segmentation of the rendered RGB frame.
R3's harness audit measured this proxy as ~170-350x optimistic vs the realized
quantity. An EMA-only read (decay 0.997) lags fast single-frame descent up to
78x. Both classes had been read as floors.

THE ONE TRUSTED REALIZED HARNESS (the trainer's verdict path, git 8cceaa072):

    MLX/numpy render (sub-camera RGB)
      -> _torch_R_to_camera_uint8        (torch bicubic up to CAMERA + uint8)
      -> cpu_verdict_d_seg / cpu_verdict_d_pose
           -> REAL CPU-torch SegNet.preprocess_input (bilinear -> 384x512)
           -> argmax-disagreement RATE vs GT argmax  == modules.py:112
           -> REAL CPU-torch PoseNet pose MSE         == modules.py:84

This module is the single source of truth for:
  * the realized-harness token vocabulary (so a gate can tell realized from proxy),
  * the canonical FEASIBILITY-ONLY tag every proxy d_seg surface MUST carry,
  * a LOUD fail-closed warning helper the proxy harnesses call at runtime,
  * the validation-ledger location proving the realized path == the contest oracle.

NO-FAKE: a "feasibility" number is a research upper-bound, NEVER a score. Only the
realized harness (validated to >=6 decimals vs ``upstream/evaluate.py`` via
``experiments/validate_realized_harness_vs_oracle.py``) may carry a witness
d_seg/d_pose SCORE, and only after byte-close + contest-CPU/CUDA exact eval.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

__all__ = [
    "CONTEST_FAITHFUL_R_FN",
    "FEASIBILITY_ONLY_MARKER",
    "FEASIBILITY_ONLY_TAG",
    "GIT_LOG_WC_COUNT_PATTERN",
    "GT_ARGMAX_TOKENS",
    "GT_ARTIFACT_LITERAL_PATTERN",
    "NON_FIELD_SUFFIXES",
    "PROXY_DSEG_SIGNATURE_TOKENS",
    "REALIZED_HARNESS_TOKENS",
    "STALE_SWAP_R_FN",
    "VALIDATION_LEDGER_REL",
    "assert_no_git_log_wc_count_consumers",
    "find_git_log_wc_count_consumers",
    "find_gt_artifact_literals",
    "validation_ledger_path",
    "warn_ema_only_dseg",
    "warn_feasibility_only_dseg",
    "warn_stale_swap_roundtrip",
]

# The canonical machine-greppable marker. Any harness that computes a PROXY
# (generator-argmax / no-R / no-SegNet-re-seg) or EMA-only d_seg MUST contain
# this token in its source AND tag the emitted number with FEASIBILITY_ONLY_TAG.
FEASIBILITY_ONLY_MARKER = "WITNESS_DSEG_FEASIBILITY_ONLY"

# The human-facing axis tag stamped onto a proxy/feasibility d_seg value.
FEASIBILITY_ONLY_TAG = "[feasibility-only — NOT realized, NOT a score]"

# Tokens whose presence in a file indicates the REALIZED contest-faithful path
# (render -> R -> real CPU-torch SegNet/PoseNet -> argmax-disagreement / pose MSE).
# A file that references ANY of these is treated as on the trusted authority path.
REALIZED_HARNESS_TOKENS = frozenset(
    {
        "_torch_R_to_camera_uint8",
        "apply_contest_faithful_roundtrip_nhwc",
        "cpu_verdict_d_seg",
        "cpu_verdict_d_pose",
        "compute_distortion",  # upstream DistortionNet / SegNet / PoseNet authority
        "DistortionNet",
    }
)

# Tokens that, together, indicate a generator-argmax PROXY d_seg (no R, no re-seg):
# the file argmaxes a *generated* class field and diffs it against the GT argmax.
PROXY_DSEG_SIGNATURE_TOKENS = frozenset(
    {
        "generator_argmax",
        "generatable",
        "argmax-disagreement",
        "score-native quantity",
        "score-native argmax",
    }
)

# Tokens naming the GT argmax target a proxy diffs against.
#
# ⚠ DETECTION VOCABULARY ONLY -- A NAME IS NOT AN IDENTITY, AND NEVER A LINEAGE.
# These four strings answer "does this source text talk about a GT argmax target?".  They do NOT
# answer "which ground truth does it read?", and no gate may treat a hit here as evidence that the
# bytes behind the name are the right ones.  ``ddm_gl1`` MEASURED six distinct files named
# ``gt_argmax_n600.npy``, and two of them -- identical in name AND byte count -- carry DIFFERENT
# decode lineages (DALI/nvdec vs PyAV).  ``ddm_gl2`` then measured that the artifact these tokens
# most often name in practice, ``gt_segnet_argmax.u8``, is PyAV lineage and differs from the DALI
# ``gt_argmax_n600.npy`` on 20,673 of 117,964,800 sites = 0.017525 S units.  So "the file is called
# gt_argmax" is worth exactly nothing about which of those two it is.
#
# The content-addressed answer lives in :mod:`tac.gt_lineage`, keyed by sha256:
# :func:`tac.gt_lineage.assert_gt_lineage` for a single read,
# :func:`tac.gt_lineage.assert_single_lineage` for one instrument's span, and
# :func:`tac.gt_lineage.assert_gt_population_registered` for the repo-wide population.
# Use :data:`GT_ARTIFACT_LITERAL_PATTERN` / :func:`find_gt_artifact_literals` below when you need
# the PATH a file reads -- a path resolves to bytes, which resolve to a sha256, which resolves to a
# lineage.  A bare name resolves to nothing.
GT_ARGMAX_TOKENS = frozenset(
    {
        "gt_argmax",
        "gt_segnet_argmax",
        "lstar",
        "lstars",
    }
)

# Suffixes that are DEFINITIONALLY not a ground-truth tensor field: source, docs, config, and
# receipts.  This is a DENY-list on purpose, and that polarity is the whole point.
#
# ``ddm_gl1``'s census used an ALLOW-list (``npy|npz|pt|pth``) at BOTH discovery legs, so its
# disk sweep of 66 files contained zero ``.u8`` and zero ``.f16`` and a whole second GT population
# -- headerless raw memmaps with 15 live readers -- was structurally invisible.  An allow-list
# fails OPEN on every container nobody thought of.  A deny-list fails CLOSED: a new raw suffix
# lands in the must-be-registered population automatically and the guard refuses until someone
# measures it.
NON_FIELD_SUFFIXES = frozenset(
    {
        ".py",
        ".json",
        ".md",
        ".txt",
        ".log",
        ".yaml",
        ".yml",
        ".csv",
        ".sh",
        ".toml",
        ".lookup",
        ".v1",
    }
)

# GT-artifact PATH literals, with NO extension allow-list.  Matches a quoted literal whose final
# component starts with ``gt_`` and carries any short suffix; the suffix filtering happens
# afterwards against :data:`NON_FIELD_SUFFIXES`, where it is visible and auditable, instead of
# being welded into the regex where a missing alternative silently deletes a population.
GT_ARTIFACT_LITERAL_PATTERN = re.compile(
    r"""["']([^"'\s]*\bgt_[A-Za-z0-9_.]*\.[A-Za-z0-9_]{1,6})["']"""
)

# The canonical contest-faithful R function and the deprecated SWAP twin.
CONTEST_FAITHFUL_R_FN = "apply_contest_faithful_roundtrip_nhwc"
STALE_SWAP_R_FN = "apply_eval_roundtrip_nhwc"

# Validation ledger (proof the realized harness == the contest oracle).
VALIDATION_LEDGER_REL = "reports/realized_harness_vs_oracle_validation.json"

# ddm_mi1 / task #885: formatted `git log` output is not a count authority.
# In at least one runner path (`rtk`) the pipe capped at 50 rows while
# `git rev-list --count` returned the true five-digit count. Treat the shell
# pipeline as a measurement-integrity hazard wherever it is newly introduced.
GIT_LOG_WC_COUNT_PATTERN = re.compile(
    r"\bgit\s+log\b[^\n|]*\\?\|\s*wc\s+-l\b",
    re.IGNORECASE,
)


def find_gt_artifact_literals(text: str) -> set[str]:
    """Return the ground-truth-artifact PATH literals in a source text.

    This is the content-addressable counterpart to :data:`GT_ARGMAX_TOKENS`.  A token tells you a
    file TALKS ABOUT ground truth; a path literal tells you WHICH BYTES it reads, and bytes have a
    sha256, and a sha256 has a recorded decode lineage.  Every consumer that needs to decide
    anything about ground truth -- rather than merely notice that ground truth is mentioned --
    should start here and finish in :mod:`tac.gt_lineage`.

    Suffixes in :data:`NON_FIELD_SUFFIXES` are dropped because a receipt, a script, or a config is
    not a tensor field.  Every OTHER suffix is kept, including ones nobody has seen yet: that is
    the deny-list polarity, and it is what makes a new raw container visible on the day it lands
    instead of on the day someone remembers to widen a regex.
    """
    out: set[str] = set()
    for m in GT_ARTIFACT_LITERAL_PATTERN.finditer(text):
        lit = m.group(1)
        if Path(lit).suffix.lower() in NON_FIELD_SUFFIXES:
            continue
        out.add(lit)
    return out


def find_git_log_wc_count_consumers(text: str, relpath: str = "<text>") -> list[str]:
    """Return lines that count commits via `git log ... | wc -l`.

    Use `git rev-list --count <range>` instead. This check is deliberately about
    the shell pipeline, not every possible Git invocation: it targets the exact
    instrument that silently undercounted in ddm_mi1's input receipt.
    """

    hits: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if GIT_LOG_WC_COUNT_PATTERN.search(line):
            hits.append(f"{relpath}:{lineno}: {line.strip()}")
    return hits


def assert_no_git_log_wc_count_consumers(text: str, relpath: str = "<text>") -> None:
    """Raise if `text` contains the non-authoritative git-log line-count idiom."""

    hits = find_git_log_wc_count_consumers(text, relpath)
    if hits:
        joined = "\n".join(hits)
        raise AssertionError(
            "non-authoritative commit count: use `git rev-list --count` instead of "
            f"`git log ... | wc -l`\n{joined}"
        )


def _banner(lines: list[str]) -> None:
    msg = "\n".join(lines)
    print(msg, file=sys.stderr, flush=True)


def warn_feasibility_only_dseg(
    harness_name: str,
    *,
    reason: str = "generator-argmax proxy: no R, no SegNet re-segmentation",
    proxy_optimism_note: str = "R3 measured this proxy ~170-350x optimistic vs realized",
) -> str:
    """Emit a LOUD fail-closed warning that ``harness_name``'s d_seg is feasibility-only.

    Returns the canonical :data:`FEASIBILITY_ONLY_TAG` so the caller can stamp it
    onto every emitted record. Call this at the TOP of any proxy harness's main()
    BEFORE any d_seg is printed.
    """

    tag = FEASIBILITY_ONLY_TAG
    _banner(
        [
            "=" * 78,
            f"  {FEASIBILITY_ONLY_MARKER}: {harness_name}",
            f"  This harness's d_seg is {tag}.",
            f"  WHY: {reason}.",
            f"  {proxy_optimism_note}.",
            "  The ONLY witness d_seg SCORE authority is the realized harness",
            "  (render -> _torch_R_to_camera_uint8 -> real CPU-torch SegNet argmax),",
            "  validated vs upstream/evaluate.py by",
            "  experiments/validate_realized_harness_vs_oracle.py.",
            "  Do NOT promote / rank / kill / claim a frontier from this number.",
            "=" * 78,
        ]
    )
    warnings.warn(
        f"{harness_name}: d_seg is {tag} ({reason}). NOT a contest score.",
        stacklevel=2,
    )
    return tag


def warn_ema_only_dseg(harness_name: str, *, ema_decay: float = 0.997) -> str:
    """LOUD warning: an EMA-shadow d_seg (no live) lags fast descent (up to 78x)."""

    tag = FEASIBILITY_ONLY_TAG
    _banner(
        [
            "=" * 78,
            f"  {FEASIBILITY_ONLY_MARKER}: {harness_name} (EMA-only read)",
            f"  EMA-shadow (decay {ema_decay}) LAGS fast single-frame descent up to 78x.",
            f"  This EMA d_seg is {tag} unless reported ALONGSIDE the live d_seg.",
            "  Report BOTH live AND EMA; recompute-S-from-components (never cached).",
            "=" * 78,
        ]
    )
    warnings.warn(
        f"{harness_name}: EMA-only d_seg is {tag} (EMA lags live up to 78x).",
        stacklevel=2,
    )
    return tag


def warn_stale_swap_roundtrip(caller: str) -> None:
    """LOUD deprecation: ``apply_eval_roundtrip_nhwc`` is the NON-contest-faithful SWAP twin."""

    _banner(
        [
            "=" * 78,
            f"  DEPRECATED R: {STALE_SWAP_R_FN} called by {caller}.",
            "  This is PR95's TRAINING recipe (uint8 @ SCORER res) — NOT contest-faithful.",
            f"  The contest stores uint8 @ CAMERA res. Use {CONTEST_FAITHFUL_R_FN}",
            "  for ANY d_seg/d_pose verdict (else d_seg is optimistic, NOT a score).",
            "=" * 78,
        ]
    )
    warnings.warn(
        f"{STALE_SWAP_R_FN} is NON-contest-faithful (training-recipe only); "
        f"use {CONTEST_FAITHFUL_R_FN} for any verdict. Caller: {caller}.",
        DeprecationWarning,
        stacklevel=2,
    )


def validation_ledger_path(repo_root: Path | str | None = None) -> Path:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent.parent
    return root / VALIDATION_LEDGER_REL
