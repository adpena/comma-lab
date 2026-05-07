#!/usr/bin/env python3
"""3-axis contest-score Pareto frontier.

Extends the 1-axis ``apogee_intN_pareto.py`` (rate-only with a categorical
risk proxy for distortion) to the full 3-axis (d_seg, d_pose, B) frontier
using the cathedral's canonical
:func:`tac.contest_rate_distortion_system.contest_score` formula.

Why 3 axes:

  S = 100 d_seg + sqrt(10 d_pose) + 25 B / 37_545_489

is a function of three independent contest-objective inputs. A 1D rate
Pareto only shows "smallest archive that doesn't blow up distortion" — but
at the new 0.19 leaderboard the dominant *marginal* axis is pose
(2.71x SegNet's marginal at PR106's d_pose ~3.4e-5). Search in
(d_seg, d_pose, B) space reveals candidates that trade rate for **pose**,
which is the new floor pressure. See the importance-flip threshold
``d_pose = 2.5e-4``: below it, marginal value flips to pose-dominated.

A candidate ``X`` is **3-axis Pareto-dominated** by ``Y`` iff:

  d_seg(Y) <= d_seg(X)  AND  d_pose(Y) <= d_pose(X)  AND  B(Y) <= B(X)
  with at least one strict.

The frontier is the set of non-dominated candidates. Among the frontier,
contest-score ``S`` ranks them under the contest's specific weighting; ties
on ``S`` are broken by lower B, then lower d_pose, then lower d_seg.

Input: a list of evidence-JSON paths, each containing
``{auth_eval: {seg_distortion: float, pose_distortion: float, ...},
   archive_bytes: int, archive_path: str, score: float}``
or a glob pattern matching that schema. The cathedral's evidence convention
(per ``auto_promote_contest_cuda.py``) emits these.

Usage::

    .venv/bin/python tools/contest_score_pareto_3axis.py \\
        --evidence-glob 'experiments/results/**/pre_submission_compliance.contest_final.json'

    .venv/bin/python tools/contest_score_pareto_3axis.py \\
        --evidence-paths a.json b.json c.json --json
"""
import argparse
import glob
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Re-export cathedral constants so this module is self-documenting + does not
# silently drift from the canonical formula.
sys.path.insert(0, str(REPO_ROOT / "src"))
try:
    from tac.contest_rate_distortion_system import (
        CONTEST_POSE_WEIGHT,
        CONTEST_RATE_WEIGHT,
        CONTEST_RAW_VIDEO_BYTES,
        CONTEST_SEG_WEIGHT,
        contest_score,
        contest_score_decomposition,
        importance_flip_threshold,
    )
    IMPORTANCE_FLIP_POSE_FLOOR = importance_flip_threshold()
except ImportError as exc:  # pragma: no cover - clear error if cathedral missing
    raise SystemExit(
        "tac.contest_rate_distortion_system not importable; this tool is part "
        "of the cathedral and requires the canonical contest_score formula. "
        f"Import error: {exc}"
    ) from None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    label: str
    d_seg: float
    d_pose: float
    archive_bytes: int
    archive_path: str | None = None
    archive_sha256: str | None = None
    score: float = 0.0
    decomposition: dict[str, float] = field(default_factory=dict)
    pareto_dominated_by: str | None = None  # label of dominator
    pareto_dominators_count: int = 0  # how many candidates dominate me
    score_rank: int = 0  # 1 = best contest score
    is_frontier: bool = False  # alias for pareto_dominated_by is None
    importance_flip_above: bool = False  # True iff d_pose > IMPORTANCE_FLIP_POSE_FLOOR


def dominates(a: Candidate, b: Candidate) -> bool:
    """Returns True iff candidate ``a`` 3-axis Pareto-dominates ``b``.

    Dominance requires component-wise <= on all 3 axes, with at least one strict <.
    """
    le_all = (
        a.d_seg <= b.d_seg
        and a.d_pose <= b.d_pose
        and a.archive_bytes <= b.archive_bytes
    )
    if not le_all:
        return False
    strict_any = (
        a.d_seg < b.d_seg
        or a.d_pose < b.d_pose
        or a.archive_bytes < b.archive_bytes
    )
    return strict_any


# ---------------------------------------------------------------------------
# Evidence loader
# ---------------------------------------------------------------------------


def _extract_d_seg(payload: dict[str, Any]) -> float | None:
    auth = payload.get("auth_eval", {}) or {}
    if isinstance(auth, dict):
        strict_formula = auth.get("strict_formula") if isinstance(auth.get("strict_formula"), dict) else {}
        record = auth.get("record") if isinstance(auth.get("record"), dict) else {}
        for container in (strict_formula, record):
            for key in ("avg_segnet_dist", "seg_distortion", "d_seg", "segnet_distortion"):
                v = container.get(key)
                if v is not None:
                    return float(v)
    for key in ("seg_distortion", "d_seg", "segnet_distortion"):
        v = auth.get(key)
        if v is None:
            v = payload.get(key)
        if v is not None:
            return float(v)
    return None


def _extract_d_pose(payload: dict[str, Any]) -> float | None:
    auth = payload.get("auth_eval", {}) or {}
    if isinstance(auth, dict):
        strict_formula = auth.get("strict_formula") if isinstance(auth.get("strict_formula"), dict) else {}
        record = auth.get("record") if isinstance(auth.get("record"), dict) else {}
        for container in (strict_formula, record):
            for key in ("avg_posenet_dist", "pose_distortion", "d_pose", "posenet_distortion"):
                v = container.get(key)
                if v is not None:
                    return float(v)
    for key in ("pose_distortion", "d_pose", "posenet_distortion"):
        v = auth.get(key)
        if v is None:
            v = payload.get(key)
        if v is not None:
            return float(v)
    return None


def _extract_archive_bytes(payload: dict[str, Any]) -> int | None:
    for key in ("archive_bytes", "archive_size_bytes", "bytes"):
        v = payload.get(key)
        if v is None:
            auth = payload.get("auth_eval", {}) or {}
            if isinstance(auth, dict):
                strict_formula = (
                    auth.get("strict_formula")
                    if isinstance(auth.get("strict_formula"), dict)
                    else {}
                )
                record = auth.get("record") if isinstance(auth.get("record"), dict) else {}
                v = strict_formula.get(key, record.get(key, auth.get(key)))
        if v is not None:
            return int(v)
    return None


def _extract_archive_path(payload: dict[str, Any], evidence_path: Path) -> str | None:
    auth = payload.get("auth_eval") if isinstance(payload.get("auth_eval"), dict) else {}
    anchor_proof = (
        auth.get("anchor_proof")
        if isinstance(auth.get("anchor_proof"), dict)
        else {}
    )
    anchor_archive = (
        anchor_proof.get("archive")
        if isinstance(anchor_proof.get("archive"), dict)
        else {}
    )
    anchor_path = anchor_archive.get("path")
    if anchor_path:
        return str(anchor_path)
    for key in ("archive_path", "archive"):
        v = payload.get(key)
        if v is not None:
            return str(v)
    sibling = evidence_path.parent / "archive.zip"
    if sibling.exists():
        return str(sibling)
    return None


def _extract_archive_sha(payload: dict[str, Any]) -> str | None:
    auth = payload.get("auth_eval", {}) or {}
    if isinstance(auth, dict):
        anchor_proof = (
            auth.get("anchor_proof")
            if isinstance(auth.get("anchor_proof"), dict)
            else {}
        )
        anchor_archive = (
            anchor_proof.get("archive")
            if isinstance(anchor_proof.get("archive"), dict)
            else {}
        )
        record = auth.get("record") if isinstance(auth.get("record"), dict) else {}
        for container in (anchor_archive, record):
            for key in ("archive_sha256", "sha256"):
                v = container.get(key)
                if v is not None:
                    return str(v)
    for key in ("archive_sha256", "sha256"):
        v = auth.get(key)
        if v is None:
            v = payload.get(key)
        if v is not None:
            return str(v)
    return None


def load_candidate_from_evidence(path: Path) -> Candidate | None:
    """Parse one evidence JSON into a Candidate. Returns None if d_seg / d_pose
    / archive_bytes is missing — these tools only act on contest-CUDA evidence
    that has all three axes populated."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    d_seg = _extract_d_seg(payload)
    d_pose = _extract_d_pose(payload)
    archive_bytes = _extract_archive_bytes(payload)
    if d_seg is None or d_pose is None or archive_bytes is None:
        return None
    archive_path = _extract_archive_path(payload, path)
    archive_sha = _extract_archive_sha(payload)
    label = path.parent.name or path.stem
    score = float(contest_score(
        seg_distortion=d_seg, pose_distortion=d_pose, archive_bytes=archive_bytes,
    ))
    decomp = contest_score_decomposition(
        seg_distortion=d_seg, pose_distortion=d_pose, archive_bytes=archive_bytes,
    )
    return Candidate(
        label=label,
        d_seg=d_seg,
        d_pose=d_pose,
        archive_bytes=archive_bytes,
        archive_path=archive_path,
        archive_sha256=archive_sha,
        score=score,
        decomposition=decomp,
        importance_flip_above=d_pose > IMPORTANCE_FLIP_POSE_FLOOR,
    )


def load_candidates(
    evidence_paths: list[Path],
    *,
    glob_pattern: str | None = None,
    repo_root: Path = REPO_ROOT,
) -> list[Candidate]:
    paths: list[Path] = list(evidence_paths)
    if glob_pattern:
        for p in glob.glob(str(repo_root / glob_pattern), recursive=True):
            paths.append(Path(p))
    seen: set[Path] = set()
    out: list[Candidate] = []
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        cand = load_candidate_from_evidence(rp)
        if cand is not None:
            out.append(cand)
    return out


# ---------------------------------------------------------------------------
# 3-axis Pareto
# ---------------------------------------------------------------------------


def compute_pareto(candidates: list[Candidate]) -> list[Candidate]:
    """In-place mark Pareto dominance + score rank. Returns the same list
    (after augmenting fields). Pure function on the candidates list besides
    the in-place augmentation."""
    n = len(candidates)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(candidates[j], candidates[i]):
                candidates[i].pareto_dominators_count += 1
                if candidates[i].pareto_dominated_by is None:
                    candidates[i].pareto_dominated_by = candidates[j].label
        candidates[i].is_frontier = candidates[i].pareto_dominated_by is None
    # Score rank: 1 = lowest contest score (best)
    sorted_by_score = sorted(
        candidates,
        key=lambda c: (c.score, c.archive_bytes, c.d_pose, c.d_seg),
    )
    for rank, c in enumerate(sorted_by_score, start=1):
        c.score_rank = rank
    return candidates


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_table(candidates: list[Candidate]) -> str:
    if not candidates:
        return "(no candidates with full (d_seg, d_pose, B) evidence found)"
    sorted_by_score = sorted(candidates, key=lambda c: c.score_rank)
    out: list[str] = []
    out.append(
        f"3-axis contest-score Pareto frontier "
        f"(formula: S = {CONTEST_SEG_WEIGHT}*d_seg + sqrt({CONTEST_POSE_WEIGHT}*d_pose) "
        f"+ {CONTEST_RATE_WEIGHT}*B/{CONTEST_RAW_VIDEO_BYTES})"
    )
    out.append(
        f"importance-flip threshold: d_pose > {IMPORTANCE_FLIP_POSE_FLOOR:.2e} "
        "→ pose-dominated marginal"
    )
    out.append("")
    header = (
        f"{'rank':>4}  {'label':<40}  {'score':>8}  "
        f"{'d_seg':>10}  {'d_pose':>11}  {'bytes':>8}  "
        f"{'frontier':<10}  {'dominator':<28}"
    )
    out.append(header)
    out.append("-" * len(header))
    for c in sorted_by_score:
        flag = "FRONTIER" if c.is_frontier else f"DOM x{c.pareto_dominators_count}"
        dominator = c.pareto_dominated_by or "-"
        if len(dominator) > 26:
            dominator = dominator[:23] + "..."
        label = c.label
        if len(label) > 38:
            label = label[:35] + "..."
        out.append(
            f"{c.score_rank:>4}  {label:<40}  {c.score:>8.5f}  "
            f"{c.d_seg:>10.6f}  {c.d_pose:>11.4e}  {c.archive_bytes:>8,}  "
            f"{flag:<10}  {dominator:<28}"
        )
    out.append("")
    n_frontier = sum(1 for c in candidates if c.is_frontier)
    out.append(
        f"{n_frontier} of {len(candidates)} candidates on the 3-axis Pareto frontier; "
        f"top score = {min(c.score for c in candidates):.5f} "
        f"({min(candidates, key=lambda c: c.score).label})"
    )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-paths",
        nargs="*",
        default=[],
        help="Explicit list of evidence JSON paths.",
    )
    parser.add_argument(
        "--evidence-glob",
        default=None,
        help="Glob pattern (relative to repo root) matching evidence JSONs. "
        "Example: 'experiments/results/**/pre_submission_compliance.contest_final.json'",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Machine-readable JSON output instead of human table.",
    )
    args = parser.parse_args(argv)

    paths = [Path(p) for p in args.evidence_paths]
    cands = load_candidates(paths, glob_pattern=args.evidence_glob)
    cands = compute_pareto(cands)

    if args.json:
        out = {
            "contest_formula": {
                "seg_weight": CONTEST_SEG_WEIGHT,
                "pose_weight": CONTEST_POSE_WEIGHT,
                "rate_weight": CONTEST_RATE_WEIGHT,
                "raw_video_bytes": CONTEST_RAW_VIDEO_BYTES,
            },
            "importance_flip_pose_floor": IMPORTANCE_FLIP_POSE_FLOOR,
            "n_candidates": len(cands),
            "n_frontier": sum(1 for c in cands if c.is_frontier),
            "candidates": [asdict(c) for c in cands],
        }
        print(json.dumps(out, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(_fmt_table(cands))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
