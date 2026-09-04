# SPDX-License-Identifier: MIT
"""Every consequence of an exact contest row, as a function of the harvest payload.

WHY THIS EXISTS (measured, 2026-09-04). Two exact contest-CUDA rows landed that day
and MAIN hand-executed roughly ten obligatory consequences for EACH of them: recompute
S from components, refresh the pointer, write the terminal claim row, regenerate
``reports/latest.md`` and ``.omx/state/current_focus.md``, register and mark the lane,
duplicate custody to the second SSD, write the pointer-move memo with re-derived
sub-0.12 arithmetic, rewrite the hot-state POINTER_LINE, push. Three of those hand
steps went wrong in one day:

* an archive sha256 was MISTYPED into the claim ledger and needed a correcting row;
* a runtime-tree sha was TRUNCATED to 12 hex, which still fails the compliance
  checker's ``[0-9a-fA-F]{64}`` binding;
* the rate-corner demand was mis-computed in the fs1 memo and repaired with ``sed``;
* the ``#316`` citation-surface gate failed twice on the way;
* two fs2 memo intermediates (the pose term, the rate term, and the gap) were typed
  from a different rounding than the score they sum to.

THE PRIOR-LAW PREDICTION this module executes: every one of those consequences is a
DETERMINISTIC FUNCTION of (harvest payload, seal, prior pointer). Nothing in the list
needs judgement — only the memo's PROSE does, and prose is what this module takes as
an input rather than inventing.

WHAT IS DELIBERATELY NOT AUTOMATED (the falsifier, named as the charter requires):

* the mechanism paragraph, the headline, and the "what this does not claim" text —
  those are claims about the world, not functions of the payload;
* the decision to publish anything (PR #140 stays operator-gated);
* the commit itself, which is offered as a printed serializer command by default so
  MAIN reviews the memo before it lands.

NUMBERS. Every number this module emits is COMPUTED from the payload. The score is
recomputed by ``tac.auth_eval_result.recompute_contest_score_from_payload`` — the ONE
canonical recompute, verified 2026-09-04 to reproduce ``score_recomputed_from_components``
bit-for-bit on both of that day's rows (the summation ORDER matters: seg + pose + rate
gives afr1 0.14797617125559104, while rate + seg + pose gives ...107, and only the
former matches the receipts). It is never read from ``final_score``, which is the
rounded 0.15 display CLAUDE.md forbids as a score.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tac.auth_eval_result import recompute_contest_score_from_payload

#: The contest's uncompressed reference size; the rate term's denominator.
CONTEST_UNCOMPRESSED_BYTES = 37_545_489
#: The live target (operator 2026-08-21, "Our new goal to surpass is 0.12").
SUB_TARGET = 0.12
#: Score units bought per archive byte removed. DERIVED: d(rate)/d(bytes) = 25/N.
EXCHANGE_S_PER_BYTE = 25.0 / CONTEST_UNCOMPRESSED_BYTES

# The evaluator's own report.txt. Its byte count carries THOUSANDS SEPARATORS
# ("Submission file size: 180,022 bytes"), which is the parse that has bitten this
# repo before -- a regex without the comma class silently reads 180 and every derived
# number downstream is wrong by three orders of magnitude.
_REPORT_PATTERNS: dict[str, re.Pattern[str]] = {
    "avg_posenet_dist": re.compile(r"Average PoseNet Distortion:\s*([0-9.eE+-]+)"),
    "avg_segnet_dist": re.compile(r"Average SegNet Distortion:\s*([0-9.eE+-]+)"),
    "archive_bytes": re.compile(r"Submission file size:\s*([0-9][0-9,]*)\s*bytes"),
    "uncompressed_bytes": re.compile(r"Original uncompressed size:\s*([0-9][0-9,]*)\s*bytes"),
    "n_samples": re.compile(r"Evaluation results over\s*([0-9][0-9,]*)\s*samples"),
}

_ORDINAL_WORDS = {
    1: "FIRST", 2: "SECOND", 3: "THIRD", 4: "FOURTH", 5: "FIFTH", 6: "SIXTH",
    7: "SEVENTH", 8: "EIGHTH", 9: "NINTH", 10: "TENTH", 11: "ELEVENTH",
    12: "TWELFTH", 13: "THIRTEENTH", 14: "FOURTEENTH", 15: "FIFTEENTH",
    16: "SIXTEENTH", 17: "SEVENTEENTH", 18: "EIGHTEENTH", 19: "NINETEENTH",
}
_TENS_WORDS = {2: "TWENTY", 3: "THIRTY", 4: "FORTY", 5: "FIFTY", 6: "SIXTY",
               7: "SEVENTY", 8: "EIGHTY", 9: "NINETY"}


def ordinal_word(n: int) -> str:
    """English ordinal in the memo-title register: 24 -> ``TWENTY-FOURTH``.

    Small and explicit on purpose: the pointer-move memos are titled by ordinal and a
    hand-typed one drifts (a memo numbered 24 twice is a ledger with a hole in it).
    """

    if n <= 0:
        raise ValueError(f"pointer moves are counted from 1; got {n}")
    if n in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[n]
    tens, ones = divmod(n, 10)
    if tens in _TENS_WORDS:
        if ones == 0:
            return _TENS_WORDS[tens][:-1] + "IETH" if _TENS_WORDS[tens].endswith("Y") else _TENS_WORDS[tens] + "TH"
        return f"{_TENS_WORDS[tens]}-{_ORDINAL_WORDS[ones]}"
    return f"{n}TH"


def parse_evaluator_report(text: str) -> dict[str, float]:
    """Parse ``upstream/evaluate.py``'s printed report into floats.

    Thousands separators are stripped, which is the whole reason this exists as a
    named function with a test rather than an inline regex at each call site.
    """

    out: dict[str, float] = {}
    for key, pattern in _REPORT_PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            continue
        out[key] = float(match.group(1).replace(",", ""))
    return out


@dataclass(frozen=True)
class ScoreRow:
    """One exact contest row, with every term recomputed from its components."""

    score: float
    rate_term: float
    seg_term: float
    pose_term: float
    d_seg: float
    d_pose: float
    archive_bytes: int
    archive_sha256: str
    runtime_tree_sha256: str | None
    axis: str
    n_samples: int | None
    lane_id: str | None
    call_id: str | None
    gpu_model: str | None
    elapsed_seconds: float | None
    evidence_grade: str | None
    passed: bool
    validation_errors: tuple[str, ...]

    @property
    def distortion(self) -> float:
        """The non-rate half of S: what the rate corner must hold fixed."""

        return self.seg_term + self.pose_term

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "rate_term": self.rate_term,
            "seg_term": self.seg_term,
            "pose_term": self.pose_term,
            "d_seg": self.d_seg,
            "d_pose": self.d_pose,
            "archive_bytes": self.archive_bytes,
            "archive_sha256": self.archive_sha256,
            "runtime_tree_sha256": self.runtime_tree_sha256,
            "axis": self.axis,
            "n_samples": self.n_samples,
            "lane_id": self.lane_id,
            "call_id": self.call_id,
            "gpu_model": self.gpu_model,
            "elapsed_seconds": self.elapsed_seconds,
            "evidence_grade": self.evidence_grade,
            "passed": self.passed,
            "validation_errors": list(self.validation_errors),
            "distortion": self.distortion,
        }


class HarvestRefusal(ValueError):
    """A harvest payload that must not become a pointer move."""


def score_row_from_harvest(
    payload: Mapping[str, Any],
    *,
    lane_id: str | None = None,
    call_id: str | None = None,
) -> ScoreRow:
    """Build the row, REFUSING anything the payload does not prove.

    Fail-closed on: a missing or non-numeric recomputed score, a recompute that
    disagrees with the receipt's own ``score_recomputed_from_components``, a missing
    archive sha, and a missing byte count. Every one of those is a silent-wrong-number
    path if defaulted, and a pointer move made from a wrong number is a false frontier.
    """

    recomputed = recompute_contest_score_from_payload(payload)
    if recomputed is None:
        raise HarvestRefusal(
            "cannot recompute S from components: the payload lacks avg_segnet_dist, "
            "avg_posenet_dist or a byte count. final_score is the ROUNDED display and "
            "is never a score (CLAUDE.md)."
        )
    stated = payload.get("score_recomputed_from_components")
    if (
        isinstance(stated, (int, float))
        and not isinstance(stated, bool)
        and float(stated) != float(recomputed)
    ):
        raise HarvestRefusal(
            "recomputed S disagrees with the receipt's own "
            f"score_recomputed_from_components: {recomputed!r} vs {stated!r}. "
            "One of the two is wrong and neither may become the pointer."
        )
    sha = payload.get("expected_archive_sha256") or payload.get("archive_sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise HarvestRefusal(f"payload carries no full 64-hex archive sha256 (got {sha!r})")
    raw_bytes = payload.get("archive_size_bytes") or payload.get("expected_archive_size_bytes")
    if not isinstance(raw_bytes, (int, float)) or int(raw_bytes) <= 0:
        raise HarvestRefusal(f"payload carries no positive archive_size_bytes (got {raw_bytes!r})")
    archive_bytes = int(raw_bytes)
    d_seg = float(payload["avg_segnet_dist"])
    d_pose = float(payload["avg_posenet_dist"])
    runtime_tree = payload.get("expected_runtime_tree_sha256") or payload.get("runtime_tree_sha256")
    errors = payload.get("validation_errors")
    return ScoreRow(
        score=float(recomputed),
        rate_term=25.0 * archive_bytes / CONTEST_UNCOMPRESSED_BYTES,
        seg_term=100.0 * d_seg,
        pose_term=math.sqrt(10.0 * d_pose),
        d_seg=d_seg,
        d_pose=d_pose,
        archive_bytes=archive_bytes,
        archive_sha256=sha,
        runtime_tree_sha256=runtime_tree if isinstance(runtime_tree, str) and runtime_tree else None,
        axis=str(payload.get("score_axis") or ""),
        n_samples=int(payload["n_samples"]) if payload.get("n_samples") is not None else None,
        lane_id=lane_id,
        call_id=call_id,
        gpu_model=payload.get("gpu_model") or payload.get("gpu"),
        elapsed_seconds=(
            float(payload["modal_elapsed_seconds"])
            if isinstance(payload.get("modal_elapsed_seconds"), (int, float))
            else None
        ),
        evidence_grade=payload.get("evidence_grade"),
        passed=payload.get("passed") is True,
        validation_errors=tuple(str(e) for e in errors) if isinstance(errors, list) else (),
    )


def cross_check_against_report(row: ScoreRow, report_text: str) -> list[str]:
    """Compare the row against the evaluator's OWN printed report; return mismatches.

    The JSON receipt and the printed report are two independent renderings of the same
    run. Agreeing is cheap; disagreeing means the row is not what the evaluator said,
    which is a stop condition, not a rounding note. The printed distortions are 3
    significant figures, so they are compared at that precision and the BYTES — printed
    exactly, with separators — are compared exactly.
    """

    parsed = parse_evaluator_report(report_text)
    problems: list[str] = []
    if not parsed:
        return ["evaluator report.txt did not parse: no recognisable result lines"]
    if "archive_bytes" in parsed and int(parsed["archive_bytes"]) != row.archive_bytes:
        problems.append(
            f"report bytes {int(parsed['archive_bytes'])} != receipt bytes {row.archive_bytes}"
        )
    if (
        "uncompressed_bytes" in parsed
        and int(parsed["uncompressed_bytes"]) != CONTEST_UNCOMPRESSED_BYTES
    ):
        problems.append(
            f"report uncompressed size {int(parsed['uncompressed_bytes'])} != "
            f"{CONTEST_UNCOMPRESSED_BYTES} (the rate denominator moved?)"
        )
    if (
        "n_samples" in parsed
        and row.n_samples is not None
        and int(parsed["n_samples"]) != row.n_samples
    ):
        problems.append(f"report n_samples {int(parsed['n_samples'])} != receipt {row.n_samples}")
    for key, actual in (("avg_segnet_dist", row.d_seg), ("avg_posenet_dist", row.d_pose)):
        if key not in parsed:
            continue
        printed = parsed[key]
        if printed == 0.0 and actual == 0.0:
            continue
        if actual == 0.0 or abs(printed - actual) > max(abs(actual) * 5e-3, 1e-12):
            problems.append(f"report {key} {printed!r} disagrees with receipt {actual!r}")
    return problems


@dataclass(frozen=True)
class TargetArithmetic:
    """The sub-0.12 arithmetic, RE-DERIVED at one pointer position.

    CLAUDE.md's own law: binding numbers expire and nobody re-derives them. So the
    memo never inherits the previous move's corner demands; they are recomputed here
    from this row and this target, every time.
    """

    target: float
    gap: float
    exchange_s_per_byte: float
    rate_corner_max_bytes: float
    rate_corner_demand_bytes: float
    distortion_corner_max: float
    distortion_corner_reduction_x: float
    zero_distortion_max_bytes: float
    zero_distortion_margin_bytes: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "gap": self.gap,
            "exchange_s_per_byte": self.exchange_s_per_byte,
            "rate_corner_max_bytes": self.rate_corner_max_bytes,
            "rate_corner_demand_bytes": self.rate_corner_demand_bytes,
            "distortion_corner_max": self.distortion_corner_max,
            "distortion_corner_reduction_x": self.distortion_corner_reduction_x,
            "zero_distortion_max_bytes": self.zero_distortion_max_bytes,
            "zero_distortion_margin_bytes": self.zero_distortion_margin_bytes,
        }


def target_arithmetic(row: ScoreRow, *, target: float = SUB_TARGET) -> TargetArithmetic:
    """Both corners of the distance to ``target`` from this exact row.

    RATE corner: hold the measured distortion, ask what archive size reaches the
    target. DISTORTION corner: hold the measured bytes, ask what distortion reaches
    it. They move in OPPOSITE directions on a bytes-for-distortion trade, which is why
    both are printed — quoting one alone has mis-stated the demand before.
    """

    distortion = row.distortion
    rate_max_bytes = (target - distortion) * CONTEST_UNCOMPRESSED_BYTES / 25.0
    distortion_max = target - row.rate_term
    zero_distortion_max = target * CONTEST_UNCOMPRESSED_BYTES / 25.0
    return TargetArithmetic(
        target=target,
        gap=row.score - target,
        exchange_s_per_byte=EXCHANGE_S_PER_BYTE,
        rate_corner_max_bytes=rate_max_bytes,
        rate_corner_demand_bytes=row.archive_bytes - rate_max_bytes,
        distortion_corner_max=distortion_max,
        distortion_corner_reduction_x=(
            distortion / distortion_max if distortion_max > 0 else float("inf")
        ),
        zero_distortion_max_bytes=zero_distortion_max,
        zero_distortion_margin_bytes=zero_distortion_max - row.archive_bytes,
    )


@dataclass
class PriorAnchor:
    """The pointer position this move is measured against."""

    label: str
    score: float | None
    archive_bytes: int | None
    d_seg: float | None
    d_pose: float | None
    lane_id: str | None = None
    archive_sha256: str | None = None

    @property
    def known(self) -> bool:
        return self.score is not None


@dataclass
class PacketPlan:
    """Everything one pointer move will write, computed before anything is written."""

    row: ScoreRow
    prior: PriorAnchor
    arithmetic: TargetArithmetic
    move_number: int
    beats_prior: bool
    report_cross_check: list[str] = field(default_factory=list)
    seal_checks: list[str] = field(default_factory=list)

    @property
    def delta_score(self) -> float | None:
        if self.prior.score is None:
            return None
        return self.row.score - self.prior.score

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "pointer_move_plan.v1",
            "move_number": self.move_number,
            "beats_prior": self.beats_prior,
            "row": self.row.to_dict(),
            "prior": {
                "label": self.prior.label,
                "score": self.prior.score,
                "archive_bytes": self.prior.archive_bytes,
                "d_seg": self.prior.d_seg,
                "d_pose": self.prior.d_pose,
                "lane_id": self.prior.lane_id,
                "archive_sha256": self.prior.archive_sha256,
            },
            "delta_score": self.delta_score,
            "arithmetic": self.arithmetic.to_dict(),
            "report_cross_check": list(self.report_cross_check),
            "seal_checks": list(self.seal_checks),
        }


def _fmt(value: float | None) -> str:
    return "—" if value is None else repr(value)


def render_memo(
    plan: PacketPlan,
    *,
    date_utc: str,
    axis_label: str,
    headline: str,
    mechanism: str,
    custody_lines: list[str],
    not_claimed: str,
    equations_leg: str,
    projection: dict[str, Any] | None = None,
    next_from_here: str = "",
) -> str:
    """Render the pointer-move memo. Every NUMBER comes from ``plan``.

    The prose arguments are the arm's claims about the world and are passed through
    verbatim; the tables, deltas, corners and margins are computed. That split is the
    point: hand-typed prose is a style, a hand-typed number is a defect.
    """

    row = plan.row
    arith = plan.arithmetic
    prior = plan.prior
    delta = plan.delta_score
    title = (
        f"# {ordinal_word(plan.move_number)} POINTER MOVE — S {row.score} @ "
        f"{row.archive_bytes:,} B [{axis_label}]: {headline} ({date_utc})"
    )
    lines = [title, "", "Tokens: `[no-triality] [p0-ledger-ok]`", "", "## The row (exact, authority)", ""]
    identity = [
        f"`upstream/evaluate.py`, {row.gpu_model or 'unknown hardware'}, "
        f"{row.n_samples if row.n_samples is not None else 'unknown'} samples, axis `{row.axis}`.",
    ]
    if row.call_id:
        identity.append(f"Modal call `{row.call_id}`.")
    if row.lane_id:
        identity.append(f"Lane `{row.lane_id}`.")
    if row.elapsed_seconds is not None:
        identity.append(f"Modal wall {row.elapsed_seconds:.1f} s.")
    identity.append(f"Archive sha `{row.archive_sha256}`, {row.archive_bytes:,} B.")
    if row.runtime_tree_sha256:
        identity.append(f"Runtime tree `{row.runtime_tree_sha256}`.")
    identity.append(
        f"`passed: {str(row.passed).lower()}`, `validation_errors: {list(row.validation_errors)}`."
    )
    lines += [" ".join(identity), ""]
    lines += [
        "Recomputed FROM COMPONENTS (#877 — never the rounded display):",
        "",
        "| term | value |",
        "|---|---|",
        f"| rate 25·{row.archive_bytes:,}/{CONTEST_UNCOMPRESSED_BYTES:,} | {row.rate_term} |",
        f"| seg 100·{row.d_seg} | {row.seg_term} |",
        f"| pose √(10·{row.d_pose}) | {row.pose_term} |",
        f"| **S** | **{row.score}** |",
        "",
        f"| | {prior.label} (prior pointer) | this move | delta |",
        "|---|---|---|---|",
        f"| S | {_fmt(prior.score)} | {row.score} | "
        f"**{_fmt(delta)}** |",
        f"| d_seg | {_fmt(prior.d_seg)} | {row.d_seg} | "
        f"{_fmt(None if prior.d_seg is None else row.d_seg - prior.d_seg)} |",
        f"| d_pose | {_fmt(prior.d_pose)} | {row.d_pose} | "
        f"{_fmt(None if prior.d_pose is None else row.d_pose - prior.d_pose)} |",
        f"| bytes | {'—' if prior.archive_bytes is None else format(prior.archive_bytes, ',')} | "
        f"{row.archive_bytes:,} | "
        f"{'—' if prior.archive_bytes is None else format(row.archive_bytes - prior.archive_bytes, '+,')} |",
        "",
    ]
    if projection:
        projected = projection.get("projected_score")
        if isinstance(projected, (int, float)):
            lines += [
                "## Projection fidelity",
                "",
                f"Projected {projected!r}; realized − projected = "
                f"{row.score - float(projected)!r}. "
                f"{projection.get('note', '')}".rstrip(),
                "",
            ]
    lines += ["## The mechanism", "", mechanism.strip(), ""]
    lines += [
        f"## Sub-{arith.target:g} arithmetic RE-DERIVED at this move "
        "(law: binding numbers expire at every pointer move)",
        "",
        f"gap {arith.gap}. Exchange 25/{CONTEST_UNCOMPRESSED_BYTES:,} = "
        f"{arith.exchange_s_per_byte} S/B.",
        "",
        f"- **RATE corner** at held distortion {row.distortion}: archive ≤ "
        f"{arith.rate_corner_max_bytes:,.1f} B → **{-arith.rate_corner_demand_bytes:,.1f} B**.",
        f"- **DISTORTION corner** at held bytes {row.archive_bytes:,}: distortion ≤ "
        f"{arith.distortion_corner_max:.5g} → **{arith.distortion_corner_reduction_x:.1f}× reduction**.",
        f"- Zero-distortion B_max {arith.zero_distortion_max_bytes:,.3f} B → the archive is "
        f"**{arith.zero_distortion_margin_bytes:,.3f} B "
        f"{'under' if arith.zero_distortion_margin_bytes >= 0 else 'over'}** the threshold at "
        "zero distortion.",
        "",
        "## Custody",
        "",
    ]
    lines += [f"- {line}" for line in custody_lines]
    lines += ["", "## What this does NOT claim", "", not_claimed.strip(), ""]
    if next_from_here.strip():
        lines += ["## Next from here", "", next_from_here.strip(), ""]
    lines += [
        f"Equations leg (`tac.canonical_equations`): {equations_leg.strip()}",
        "",
        f"Own-vehicle frontier: **S {row.score} @ {row.archive_bytes:,} B [{axis_label}]**, "
        f"archive sha `{row.archive_sha256[:8]}…{row.archive_sha256[-5:]}`.",
        "",
    ]
    return "\n".join(lines)


def render_pointer_line(plan: PacketPlan, *, axis_label: str, extra: str = "") -> str:
    """The hot-state POINTER_LINE body: the own-vehicle frontier, one block."""

    row = plan.row
    delta = plan.delta_score
    parts = [
        f"OWN-VEHICLE FRONTIER: S {row.score} @ {row.archive_bytes:,} B [{axis_label}]",
        f"archive sha {row.archive_sha256}",
    ]
    if row.runtime_tree_sha256:
        parts.append(f"runtime tree {row.runtime_tree_sha256}")
    if row.lane_id:
        parts.append(f"lane {row.lane_id}")
    if delta is not None:
        parts.append(f"move #{plan.move_number}: {delta:+.6e} vs {plan.prior.label}")
    arith = plan.arithmetic
    parts.append(
        f"sub-{arith.target:g} gap {arith.gap:.8f}; rate corner "
        f"{-arith.rate_corner_demand_bytes:,.1f} B at held distortion; distortion corner "
        f"{arith.distortion_corner_reduction_x:.1f}x at held bytes"
    )
    if extra.strip():
        parts.append(extra.strip())
    return "\n".join(f"- {p}" for p in parts)


def pointer_move_event(plan: PacketPlan, *, axis_label: str, memo_path: str, at_utc: str) -> dict[str, Any]:
    """One append-only row for ``.omx/state/pointer_move_events.jsonl``."""

    return {
        "schema": "pointer_move_event.v1",
        "at_utc": at_utc,
        "move_number": plan.move_number,
        "axis": plan.row.axis,
        "axis_label": axis_label,
        "score": plan.row.score,
        "prior_score": plan.prior.score,
        "delta_score": plan.delta_score,
        "archive_bytes": plan.row.archive_bytes,
        "archive_sha256": plan.row.archive_sha256,
        "runtime_tree_sha256": plan.row.runtime_tree_sha256,
        "lane_id": plan.row.lane_id,
        "call_id": plan.row.call_id,
        "memo": memo_path,
        "gap_to_target": plan.arithmetic.gap,
        "rate_corner_demand_bytes": plan.arithmetic.rate_corner_demand_bytes,
    }


def load_json(path: Path) -> dict[str, Any]:
    """Read one JSON object, refusing anything that is not a mapping."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise HarvestRefusal(f"{path}: expected a JSON object, got {type(data).__name__}")
    return data


__all__ = [
    "CONTEST_UNCOMPRESSED_BYTES",
    "EXCHANGE_S_PER_BYTE",
    "SUB_TARGET",
    "HarvestRefusal",
    "PacketPlan",
    "PriorAnchor",
    "ScoreRow",
    "TargetArithmetic",
    "cross_check_against_report",
    "load_json",
    "ordinal_word",
    "parse_evaluator_report",
    "pointer_move_event",
    "render_memo",
    "render_pointer_line",
    "score_row_from_harvest",
    "target_arithmetic",
]
