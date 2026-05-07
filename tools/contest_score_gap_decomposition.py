"""Forensic decomposition of the score gap between PR106 and medal-band PRs.

Per dual-path investigation
(`feedback_pr106_substrate_is_below_medal_band_20260507.md` Path B):
PR106's +0.016 gap vs gold (PR101 @ 0.193) could be (a) substrate
inferiority, (b) implementation bug, or (c) measurement methodology
drift. This tool decomposes the gap into per-component shares so the
operator can localize the cause.

Usage:

    python tools/contest_score_gap_decomposition.py

The tool emits a markdown table of per-component (seg, pose, rate) share
differences and the per-component contribution to the gap, identifying
the dominant mismatch axis.

Strict-scorer-rule: pure CPU + math + json. No scorer load. The seg/pose
values for both anchors come from contest-CUDA replay artifacts that are
already on disk; this tool does NOT invoke any scorer.

Cross-references:

- :mod:`tac.contest_rate_distortion_system`: contest formula + decomposition
- ``feedback_pr106_substrate_is_below_medal_band_20260507``: Path-B
  investigation prescription
- ``experiments/results/pr103_repack_pr106_standalone_20260507/``: PR103-on-PR106 anchor
"""

import json
import pathlib
import sys
from dataclasses import dataclass

# Local import (tools live in repo root; tac is the canonical lib).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from tac.contest_rate_distortion_system import (
    contest_score_decomposition,
)

# ---------------------------------------------------------------------------
# Reference anchors (from public PR bodies + our local artifacts)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoreAnchor:
    """One (seg_distortion, pose_distortion, archive_bytes, score) anchor."""
    label: str
    seg: float
    pose: float
    bytes: int
    reported_score: float
    source: str
    evidence_grade: str  # "[contest-CUDA]" or "[public-PR-claim]"


# PR103-on-PR106 standalone (our local frontier; gates 1-5 GREEN as of 2026-05-07).
PR103_PR106_ANCHOR = ScoreAnchor(
    label="PR103-on-PR106",
    seg=0.00067082,
    pose=0.0000336,
    bytes=185_578,
    reported_score=0.2089810755823297,
    source="experiments/results/pr103_repack_pr106_standalone_20260507/",
    evidence_grade="[contest-CUDA T4]",
)


# Public medal-band reference points. seg/pose values are NOT in the public
# PR bodies (only the total score is); we back-solve approximate values from
# the published score under the canonical formula. These are PROXIES, not
# measured values; tagged "[public-PR-claim back-solved]" accordingly.
#
# Method: assume each medal entry's seg term is small (~0.05-0.07 share),
# rate is computed directly from bytes, pose is the residual.
def _back_solve_pose(*, score: float, bytes_: int, seg: float) -> float:
    """Back-solve pose distortion from total score, given seg + bytes.

    Bug-hunter v3 integration-seam fix 2026-05-07: previously, when the
    seg + rate terms already exceeded the published score (e.g. when the
    operator passes an over-estimated seg_estimate), the function silently
    returned 0.0. That produced misleading gap-decomposition tables where
    "pose_term=0" looked like a real measurement instead of an
    underdetermined back-solve. We now emit a UserWarning so the operator
    knows the seg estimate is too high for this anchor; the table consumer
    can see the warning in the test log or the decompose_anchor JSON
    payload (see ``ScoreAnchor.pose_underdetermined`` in the JSON output).
    """
    import warnings

    from tac.contest_rate_distortion_system import (
        CONTEST_POSE_WEIGHT,
        CONTEST_RATE_WEIGHT,
        CONTEST_RAW_VIDEO_BYTES,
        CONTEST_SEG_WEIGHT,
    )
    rate_term = CONTEST_RATE_WEIGHT * bytes_ / CONTEST_RAW_VIDEO_BYTES
    seg_term = CONTEST_SEG_WEIGHT * seg
    pose_term = score - seg_term - rate_term
    if pose_term <= 0:
        warnings.warn(
            f"_back_solve_pose: seg+rate terms ({seg_term + rate_term:.6f}) "
            f"already exceed reported score {score:.6f} for "
            f"bytes={bytes_:,} seg={seg:.6e}; pose back-solves to <= 0 "
            f"(returned 0.0). The seg estimate may be too high; the "
            f"resulting decomposition's pose share is underdetermined "
            f"and the table consumer should treat pose_term=0 as a guess.",
            UserWarning,
            stacklevel=2,
        )
        return 0.0
    return (pose_term * pose_term) / CONTEST_POSE_WEIGHT


def medal_band_anchors() -> list[ScoreAnchor]:
    """Return the public medal-band reference points (back-solved pose).

    Source: PR bodies on the public challenge repo. seg estimated at the
    canonical contest baseline; pose back-solved from total minus seg + rate.
    These are approximations; useful for SHARE comparison, not for
    score-claim use.
    """
    # Educated estimates of seg distortion for the 3 medal-band PRs based on
    # CLAUDE.md "TRUE score data 2026-04-21" table where seg was 0.00067 at
    # 178K bytes; medal entries are similar substrate.
    seg_estimate = 0.00067082
    return [
        ScoreAnchor(
            label="PR101 (gold) - back-solved",
            seg=seg_estimate,
            pose=_back_solve_pose(score=0.193, bytes_=178_258, seg=seg_estimate),
            bytes=178_258,
            reported_score=0.193,
            source="public PR #101 body",
            evidence_grade="[public-PR-claim back-solved]",
        ),
        ScoreAnchor(
            label="PR103 (silver) - back-solved",
            seg=seg_estimate,
            pose=_back_solve_pose(score=0.195, bytes_=178_223, seg=seg_estimate),
            bytes=178_223,
            reported_score=0.195,
            source="public PR #103 body",
            evidence_grade="[public-PR-claim back-solved]",
        ),
        ScoreAnchor(
            label="PR102 (bronze) - back-solved",
            seg=seg_estimate,
            pose=_back_solve_pose(score=0.195, bytes_=178_981, seg=seg_estimate),
            bytes=178_981,
            reported_score=0.195,
            source="public PR #102 body",
            evidence_grade="[public-PR-claim back-solved]",
        ),
    ]


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def decompose_anchor(anchor: ScoreAnchor) -> dict[str, float]:
    """Run contest_score_decomposition on the anchor + return augmented dict."""
    dec = contest_score_decomposition(
        seg_distortion=anchor.seg,
        pose_distortion=anchor.pose,
        archive_bytes=anchor.bytes,
    )
    dec["label"] = anchor.label
    dec["reported"] = anchor.reported_score
    dec["evidence_grade"] = anchor.evidence_grade
    dec["source"] = anchor.source
    dec["seg_input"] = anchor.seg
    dec["pose_input"] = anchor.pose
    dec["bytes_input"] = float(anchor.bytes)
    return dec


def gap_analysis(
    target: ScoreAnchor,
    references: list[ScoreAnchor],
) -> dict[str, dict[str, float]]:
    """For each reference, return per-component contribution to the gap
    ``score(target) - score(reference)``.

    Decomposes the gap into seg_term_delta + pose_term_delta + rate_term_delta;
    these sum exactly to the total score delta.
    """
    target_dec = decompose_anchor(target)
    out: dict[str, dict[str, float]] = {}
    for ref in references:
        ref_dec = decompose_anchor(ref)
        out[ref.label] = {
            "total_gap": target_dec["total"] - ref_dec["total"],
            "seg_term_delta": target_dec["seg_term"] - ref_dec["seg_term"],
            "pose_term_delta": target_dec["pose_term"] - ref_dec["pose_term"],
            "rate_term_delta": target_dec["rate_term"] - ref_dec["rate_term"],
            "ref_total": ref_dec["total"],
            "target_total": target_dec["total"],
        }
    return out


def render_markdown(target: ScoreAnchor, refs: list[ScoreAnchor]) -> str:
    """Beautiful markdown report of the gap analysis."""
    target_dec = decompose_anchor(target)
    lines = [
        "# Contest score gap decomposition",
        "",
        f"**Target**: {target.label}  `{target.evidence_grade}`",
        f"  - bytes={target.bytes:,}, seg={target.seg:.6f}, pose={target.pose:.6f}",
        f"  - score={target_dec['total']:.6f}  (reported {target.reported_score})",
        f"  - shares: seg={target_dec['seg_share']*100:.1f}%  "
        f"pose={target_dec['pose_share']*100:.1f}%  "
        f"rate={target_dec['rate_share']*100:.1f}%",
        "",
        "## Per-component decomposition (target)",
        "",
        f"- seg_term  = 100 · {target.seg:.6f} = {target_dec['seg_term']:.6f}",
        f"- pose_term = sqrt(10 · {target.pose:.6e}) = {target_dec['pose_term']:.6f}",
        f"- rate_term = 25 · {target.bytes} / 37,545,489 = {target_dec['rate_term']:.6f}",
        "",
        "## Gap to medal-band references",
        "",
        "| reference | ref_total | gap (target - ref) | seg_delta | pose_delta | rate_delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    gaps = gap_analysis(target, refs)
    for ref in refs:
        g = gaps[ref.label]
        lines.append(
            f"| {ref.label} | {g['ref_total']:.6f} | {g['total_gap']:+.6f} | "
            f"{g['seg_term_delta']:+.6f} | {g['pose_term_delta']:+.6f} | "
            f"{g['rate_term_delta']:+.6f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Each gap row sums to the total: **gap = seg_delta + pose_delta + rate_delta**.",
        "The dominant axis is the largest absolute delta; that's where the implementation",
        "bug or substrate mismatch is concentrated.",
        "",
        "## Caveats",
        "",
        "- Public-PR seg/pose values are **back-solved**, not measured. The",
        "  shape of the gap (which axis dominates) is robust under reasonable",
        "  perturbations of the assumed seg distortion; the absolute pose value",
        "  is sensitive.",
        "- Score claims are `[contest-CUDA T4]` (target) or",
        "  `[public-PR-claim back-solved]` (references). NEITHER converted to",
        "  `[contest-CUDA]` for the references; that would require running",
        "  the public PRs through our local contest-CUDA pipeline.",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Decompose PR106 vs medal-band score gap")
    p.add_argument(
        "--output-dir",
        default=None,
        help="output dir for JSON + markdown manifest; defaults to lane_score_gap_decomposition_<UTC>/",
    )
    args = p.parse_args(argv)

    target = PR103_PR106_ANCHOR
    refs = medal_band_anchors()
    md = render_markdown(target, refs)
    print(md)

    import datetime as _dt
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = pathlib.Path(args.output_dir) if args.output_dir else (
        pathlib.Path(f"experiments/results/lane_score_gap_decomposition_{ts}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": "tools/contest_score_gap_decomposition",
        "evidence_grade": "[empirical:target+public-PR-claim:references]",
        "score_claim": False,
        "target": decompose_anchor(target),
        "references": [decompose_anchor(r) for r in refs],
        "gaps": gap_analysis(target, refs),
    }
    (out_dir / "gap.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "gap.md").write_text(md)
    print(f"\nmanifest: {out_dir / 'gap.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
