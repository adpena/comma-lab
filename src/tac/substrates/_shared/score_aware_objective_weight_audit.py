# SPDX-License-Identifier: MIT
"""Canonical detector for the OBJECTIVE-STARVATION bug class (Catalog #384).

Source: operator NON-NEGOTIABLE 2026-06-09. Empirical anchor: SNeRV's faithful
official-MFU/HFR/TUB renderer at ep22399 scored ``avg_segnet_dist = 0.711``
with ``observed_segnet_distillation_weight = None`` — "score-aware in name,
recon-MSE in gradient" — because the shared MLX harness
(``snerv_inverse_steg_carrier/mlx_native_train_export.py``) declares
``segnet_distillation_weight: float = 0.0`` and ``pose_distillation_weight:
float = 0.0`` as defaults. A run that CLAIMS to be "score-aware" / "PR95-style"
/ "scorer-aware" but optimizes recon-MSE with the scorer objective weights at
0.0 is starving the only authority that moves the contest score.

This module is the reusable detection logic; the thin STRICT preflight wrapper
(``tac.preflight.check_score_aware_run_has_nonzero_scorer_objective_weights``)
delegates to :func:`audit_score_aware_objective_weights`. Per AGENTS.md
"TAC / comma-lab Boundary": real reusable detection logic lives in ``tac``;
``preflight.py`` carries only the thin orchestrator wrapper.

The rule (a falling-rule list, per CLAUDE.md "Preflight failure messages must
cite the rule chain"):

1. **Not a score-aware claimant** — the file does NOT contain a score-aware /
   PR95-style / scorer-aware claim token → SKIP (no obligation).
2. **Explicit opt-out** — the file declares ``score_aware=false`` /
   ``scoreaware=false`` (case-insensitive), OR is ``research_only`` /
   ``dispatch_enabled=false`` (a research/non-dispatch scaffold makes no
   trained-score-aware claim) → SKIP.
3. **Same-line waiver** — a
   ``# SCORE_AWARE_OBJECTIVE_WEIGHTS_OK:<rationale>`` waiver on the offending
   default line (placeholder ``<rationale>`` / ``<reason>`` literals rejected
   per Catalog #287) → SKIP.
4. **Explicit nonzero weights present** — the file sets a SegNet objective
   weight AND a PoseNet objective weight to an explicit nonzero value → OK.
5. **VIOLATION** — the file claims score-aware, has a SegNet/PoseNet objective
   weight defaulting to ``0.0`` (or ``None``), and provides no explicit nonzero
   value, opt-out, or waiver → the objective-starvation bug.

The SegNet objective weight family (any one defaulting to 0.0 triggers the
SegNet half): ``segnet_distillation_weight`` /
``segnet_direct_live_distillation_weight``. The PoseNet objective weight family:
``pose_distillation_weight`` / ``pose_direct_live_distillation_weight``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "POSE_OBJECTIVE_WEIGHT_NAMES",
    "SCORE_AWARE_CLAIM_TOKENS",
    "SEGNET_OBJECTIVE_WEIGHT_NAMES",
    "ScoreAwareWeightFinding",
    "audit_score_aware_objective_weights",
    "file_has_objective_starvation",
]

# Tokens that CLAIM a run is score-aware / PR95-style / scorer-aware.
SCORE_AWARE_CLAIM_TOKENS: tuple[str, ...] = (
    "score-aware",
    "score_aware",
    "scoreaware",
    "scorer-aware",
    "scorer_aware",
    "pr95-style",
    "pr95_style",
)

# SegNet objective-weight argument/field names (any one at 0.0/None triggers
# the SegNet half of the starvation).
SEGNET_OBJECTIVE_WEIGHT_NAMES: tuple[str, ...] = (
    "segnet_distillation_weight",
    "segnet_direct_live_distillation_weight",
)

# PoseNet objective-weight argument/field names.
POSE_OBJECTIVE_WEIGHT_NAMES: tuple[str, ...] = (
    "pose_distillation_weight",
    "pose_direct_live_distillation_weight",
)

_WAIVER_TOKEN = "SCORE_AWARE_OBJECTIVE_WEIGHTS_OK"
# Catalog #287: a placeholder rationale cannot self-waive.
_PLACEHOLDER_RATIONALES = frozenset(
    {"<rationale>", "<reason>", "rationale", "reason", "tbd", "todo", ""}
)

_CLAIM_RE = re.compile(
    "|".join(re.escape(tok) for tok in SCORE_AWARE_CLAIM_TOKENS), re.IGNORECASE
)
# Explicit opt-out: the file declares it is NOT score-aware OR is research-only.
_OPT_OUT_RE = re.compile(
    r"score_aware\s*=\s*[Ff]alse"
    r"|scoreaware\s*=\s*[Ff]alse"
    r"|score-aware\s*=\s*[Ff]alse"
    r"|research_only\s*=\s*[Tt]rue"
    r"|research_only\s*:\s*[Tt]rue"
    r"|dispatch_enabled\s*=\s*[Ff]alse"
    r"|dispatch_enabled\s*:\s*[Ff]alse"
)


def _weight_zero_default_re(name: str) -> re.Pattern[str]:
    """Match ``<name> ... = 0.0`` or ``= None`` (kwarg default OR assignment).

    Handles type-annotated kwargs (``name: float = 0.0``), plain assignments
    (``name = 0.0``), ``int | None = None`` unions, and YAML (``name: 0.0``).
    """
    return re.compile(
        rf"\b{re.escape(name)}\b\s*"
        rf"(?::[^=\n]*)?"  # optional type annotation
        rf"[:=]\s*"
        rf"(?:0\.0|0|None|none|null)\b"
    )


def _weight_nonzero_re(name: str) -> re.Pattern[str]:
    """Match ``<name> ... = <nonzero number>`` (an explicit nonzero objective)."""
    return re.compile(
        rf"\b{re.escape(name)}\b\s*"
        rf"(?::[^=\n]*)?"
        rf"[:=]\s*"
        rf"(?:"
        rf"[1-9]\d*(?:\.\d+)?"  # 1, 25, 100, 1.5
        rf"|0?\.0*[1-9]\d*"  # 0.5, .25, 0.001
        rf")"
    )


@dataclass(frozen=True)
class ScoreAwareWeightFinding:
    """A single objective-starvation finding for one file."""

    path: str
    """Repo-relative path of the offending file."""

    claim_line: int
    """1-based line where the score-aware claim was found."""

    claim_excerpt: str
    """The matched claim line (stripped)."""

    zero_weight_names: tuple[str, ...]
    """Objective-weight names found defaulting to 0.0/None."""

    missing_axes: tuple[str, ...]
    """Which axis families lack an explicit nonzero weight ('segnet' / 'pose')."""

    def message(self) -> str:
        return (
            f"{self.path}:{self.claim_line} OBJECTIVE-STARVATION: file claims "
            f"score-aware/PR95-style (\"{self.claim_excerpt[:70]}\") but the "
            f"{'/'.join(self.missing_axes)} objective weight(s) default to 0.0/None "
            f"({', '.join(self.zero_weight_names)}) with no explicit nonzero value. "
            f"Rule chain: claim_present AND not(opt_out) AND not(waiver) AND "
            f"not(explicit_nonzero) -> VIOLATION. Unwind: (a) set the SegNet AND "
            f"PoseNet distillation/objective weights to explicit NONZERO values; "
            f"(b) declare score_aware=false / scoreaware=false (or research_only "
            f"/ dispatch_enabled=false) if the run is intentionally recon-only; "
            f"(c) same-line # {_WAIVER_TOKEN}:<substantive-rationale> waiver."
        )


def _has_waiver_near(lines: list[str], line_idx0: int) -> bool:
    """True if a non-placeholder waiver is on this line or the 2 lines around it."""
    for j in range(max(0, line_idx0 - 1), min(len(lines), line_idx0 + 2)):
        line = lines[j]
        if _WAIVER_TOKEN not in line:
            continue
        after = line.split(_WAIVER_TOKEN, 1)[1].lstrip(": ").strip().lower()
        rationale = after.split("#")[0].strip()
        # Strip a trailing close-paren/quote noise.
        rationale = rationale.rstrip(")\"' ")
        if rationale and rationale not in _PLACEHOLDER_RATIONALES and len(rationale) >= 4:
            return True
    return False


def file_has_objective_starvation(
    source: str, rel_path: str
) -> ScoreAwareWeightFinding | None:
    """Return a finding if ``source`` is a score-aware claimant starving the objective.

    Implements the rule chain in the module docstring. Returns ``None`` when the
    file is not a claimant, opts out, sets explicit nonzero weights, or waives.
    """
    claim_match = _CLAIM_RE.search(source)
    if claim_match is None:
        return None  # rule 1: not a claimant

    if _OPT_OUT_RE.search(source):
        return None  # rule 2: explicit opt-out / research-only / non-dispatch

    lines = source.splitlines()

    # Which objective-weight names default to 0.0/None?
    zero_segnet: list[str] = []
    zero_pose: list[str] = []
    zero_line_idx: int | None = None
    for name in SEGNET_OBJECTIVE_WEIGHT_NAMES:
        m = _weight_zero_default_re(name).search(source)
        if m is not None:
            zero_segnet.append(name)
            if zero_line_idx is None:
                zero_line_idx = source[: m.start()].count("\n")
    for name in POSE_OBJECTIVE_WEIGHT_NAMES:
        m = _weight_zero_default_re(name).search(source)
        if m is not None:
            zero_pose.append(name)
            if zero_line_idx is None:
                zero_line_idx = source[: m.start()].count("\n")

    if not zero_segnet and not zero_pose:
        return None  # no 0.0/None objective-weight default -> nothing to starve

    # rule 4: explicit nonzero present for an axis cancels that axis.
    has_nonzero_segnet = any(
        _weight_nonzero_re(name).search(source) for name in SEGNET_OBJECTIVE_WEIGHT_NAMES
    )
    has_nonzero_pose = any(
        _weight_nonzero_re(name).search(source) for name in POSE_OBJECTIVE_WEIGHT_NAMES
    )

    missing_axes: list[str] = []
    if zero_segnet and not has_nonzero_segnet:
        missing_axes.append("segnet")
    if zero_pose and not has_nonzero_pose:
        missing_axes.append("pose")
    if not missing_axes:
        return None  # both axes have explicit nonzero values -> OK

    # rule 3: same-line waiver on the zero-default line.
    if zero_line_idx is not None and _has_waiver_near(lines, zero_line_idx):
        return None

    claim_line0 = source[: claim_match.start()].count("\n")
    excerpt = lines[claim_line0].strip() if claim_line0 < len(lines) else ""
    return ScoreAwareWeightFinding(
        path=rel_path,
        claim_line=claim_line0 + 1,
        claim_excerpt=excerpt,
        zero_weight_names=tuple(zero_segnet + zero_pose),
        missing_axes=tuple(missing_axes),
    )


def _default_scan_paths(repo: Path) -> list[Path]:
    """The canonical scan surface: substrate trainers + score-aware loss configs."""
    paths: list[Path] = []
    experiments = repo / "experiments"
    if experiments.is_dir():
        paths.extend(sorted(experiments.glob("train_substrate_*.py")))
    substrates = repo / "src" / "tac" / "substrates"
    if substrates.is_dir():
        paths.extend(sorted(substrates.rglob("score_aware_loss.py")))
        paths.extend(sorted(substrates.rglob("mlx_native_train_export.py")))
    return paths


def audit_score_aware_objective_weights(
    repo_root: str | Path,
    *,
    scan_paths: Iterable[Path] | None = None,
) -> list[ScoreAwareWeightFinding]:
    """Audit the canonical scan surface for objective-starvation; return findings.

    The scan surface (when ``scan_paths`` is not given): every
    ``experiments/train_substrate_*.py`` + every substrate
    ``score_aware_loss.py`` + every substrate ``mlx_native_train_export.py``
    (the shared MLX harness family that carried the SNeRV recon-MSE anchor).
    """
    repo = Path(repo_root)
    paths = list(scan_paths) if scan_paths is not None else _default_scan_paths(repo)
    findings: list[ScoreAwareWeightFinding] = []
    for path in paths:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            rel = str(path.relative_to(repo))
        except ValueError:
            rel = str(path)
        finding = file_has_objective_starvation(source, rel)
        if finding is not None:
            findings.append(finding)
    return findings
