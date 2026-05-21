#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit HFV1 PR101 exact-eval candidates against the FEC6 rate hurdle.

This consumes ``hfv1_pr101_exact_eval_readiness.json`` and quantifies how much
SegNet/PoseNet component improvement a larger HFV1 archive must deliver before
it can beat the current FEC6/PR110 CPU-axis score. It is a rate-arithmetic
dispatch advisor only; it does not claim a score and does not run eval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any

CONTEST_DENOM_BYTES = Decimal("37545489")
RATE_MULTIPLIER = Decimal("25")
DEFAULT_BASELINE_ARCHIVE = Path(
    "experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/"
    "submission_dir/archive.zip"
)
DEFAULT_BASELINE_SCORE = Decimal("0.192051")


getcontext().prec = 50


@dataclass(frozen=True)
class RateHurdleRow:
    variant: str
    archive_zip_path: str
    archive_zip_bytes: int
    archive_zip_sha256: str
    baseline_archive_bytes: int
    baseline_archive_sha256: str
    archive_byte_delta_vs_baseline: int
    current_rate_term: str
    baseline_rate_term: str
    current_rate_penalty_vs_baseline: str
    baseline_score: str
    baseline_component_term: str
    max_candidate_component_term_to_tie: str
    required_component_gain_to_tie_current_bytes: str
    estimated_recoverable_zip_bytes: int
    estimated_floor_archive_bytes: int
    estimated_floor_rate_term: str
    estimated_floor_rate_penalty_vs_baseline: str
    required_component_gain_to_tie_after_estimated_floor: str
    dispatch_priority: str
    advisory: str
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


@dataclass(frozen=True)
class RateHurdleAudit:
    schema: str
    generated_at_utc: str
    readiness_json: str
    baseline_archive_path: str
    baseline_archive_bytes: int
    baseline_archive_sha256: str
    baseline_score: str
    baseline_rate_term: str
    baseline_component_term: str
    candidates: list[RateHurdleRow]
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rate_term(archive_bytes: int) -> Decimal:
    return RATE_MULTIPLIER * Decimal(archive_bytes) / CONTEST_DENOM_BYTES


def _fmt_decimal(value: Decimal) -> str:
    return format(value, "f")


def _latest_readiness_json(results_root: Path) -> Path:
    candidates = sorted(
        results_root.glob(
            "hfv1_pr101_exact_eval_readiness_*/hfv1_pr101_exact_eval_readiness.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no hfv1_pr101_exact_eval_readiness_*/hfv1_pr101_exact_eval_readiness.json under {results_root}"
        )
    return candidates[0]


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _safe_score(value: str) -> Decimal:
    text = str(value).strip()
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text):
        raise ValueError(f"baseline score must be a decimal literal: {value!r}")
    return Decimal(text)


def _dispatch_priority(
    *,
    variant: str,
    required_gain_current: Decimal,
    required_gain_floor: Decimal,
) -> tuple[str, str]:
    if required_gain_current <= Decimal("0"):
        return (
            "dispatch_now_rate_favorable",
            "candidate archive is not rate-disadvantaged against baseline",
        )
    if variant == "seed_top16_component_hardpairs":
        return (
            "dispatch_first_if_remote_slot_clean",
            "component-hardpair targeting is the only nontrivial HFV1 row with a plausible component-gain mechanism",
        )
    if required_gain_floor <= Decimal("0.003"):
        return (
            "recode_then_dispatch",
            "estimated member-floor recode would make the rate hurdle small enough to test cheaply",
        )
    return (
        "defer_until_component_or_recode_signal",
        "current archive needs a component gain too large to justify priority over the seed hardpair row",
    )


def build_audit(
    *,
    readiness_json: Path,
    baseline_archive: Path,
    baseline_score: Decimal,
) -> RateHurdleAudit:
    payload = json.loads(readiness_json.read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("readiness JSON has no list-valued candidates")
    if not baseline_archive.is_file():
        raise FileNotFoundError(f"baseline archive not found: {baseline_archive}")

    baseline_bytes = baseline_archive.stat().st_size
    baseline_sha = _sha256_file(baseline_archive)
    baseline_rate = _rate_term(baseline_bytes)
    baseline_component = baseline_score - baseline_rate
    rows: list[RateHurdleRow] = []
    for candidate in candidates:
        archive_bytes = int(candidate["archive_zip_bytes"])
        recoverable = int(candidate.get("estimated_recoverable_zip_bytes") or 0)
        current_rate = _rate_term(archive_bytes)
        current_penalty = current_rate - baseline_rate
        max_component_to_tie = baseline_score - current_rate
        required_gain_current = baseline_component - max_component_to_tie
        floor_bytes = max(0, archive_bytes - recoverable)
        floor_rate = _rate_term(floor_bytes)
        floor_penalty = floor_rate - baseline_rate
        max_component_floor = baseline_score - floor_rate
        required_gain_floor = baseline_component - max_component_floor
        priority, advisory = _dispatch_priority(
            variant=str(candidate["variant"]),
            required_gain_current=required_gain_current,
            required_gain_floor=required_gain_floor,
        )
        rows.append(
            RateHurdleRow(
                variant=str(candidate["variant"]),
                archive_zip_path=str(candidate["archive_zip_path"]),
                archive_zip_bytes=archive_bytes,
                archive_zip_sha256=str(candidate["archive_zip_sha256"]),
                baseline_archive_bytes=baseline_bytes,
                baseline_archive_sha256=baseline_sha,
                archive_byte_delta_vs_baseline=archive_bytes - baseline_bytes,
                current_rate_term=_fmt_decimal(current_rate),
                baseline_rate_term=_fmt_decimal(baseline_rate),
                current_rate_penalty_vs_baseline=_fmt_decimal(current_penalty),
                baseline_score=_fmt_decimal(baseline_score),
                baseline_component_term=_fmt_decimal(baseline_component),
                max_candidate_component_term_to_tie=_fmt_decimal(max_component_to_tie),
                required_component_gain_to_tie_current_bytes=_fmt_decimal(
                    required_gain_current
                ),
                estimated_recoverable_zip_bytes=recoverable,
                estimated_floor_archive_bytes=floor_bytes,
                estimated_floor_rate_term=_fmt_decimal(floor_rate),
                estimated_floor_rate_penalty_vs_baseline=_fmt_decimal(floor_penalty),
                required_component_gain_to_tie_after_estimated_floor=_fmt_decimal(
                    required_gain_floor
                ),
                dispatch_priority=priority,
                advisory=advisory,
            )
        )
    rows.sort(
        key=lambda row: (
            0 if row.variant == "seed_top16_component_hardpairs" else 1,
            Decimal(row.required_component_gain_to_tie_after_estimated_floor),
            Decimal(row.required_component_gain_to_tie_current_bytes),
            row.variant,
        )
    )
    return RateHurdleAudit(
        schema="hfv1_pr101_rate_hurdle_audit_v1",
        generated_at_utc=_utc_iso(),
        readiness_json=_repo_rel(readiness_json),
        baseline_archive_path=_repo_rel(baseline_archive),
        baseline_archive_bytes=baseline_bytes,
        baseline_archive_sha256=baseline_sha,
        baseline_score=_fmt_decimal(baseline_score),
        baseline_rate_term=_fmt_decimal(baseline_rate),
        baseline_component_term=_fmt_decimal(baseline_component),
        candidates=rows,
    )


def render_markdown(audit: RateHurdleAudit) -> str:
    lines = [
        "# HFV1 PR101 Rate Hurdle Audit",
        "",
        f"- Generated UTC: {audit.generated_at_utc}",
        f"- Readiness JSON: `{audit.readiness_json}`",
        f"- Baseline archive: `{audit.baseline_archive_path}`",
        f"- Baseline bytes: {audit.baseline_archive_bytes}",
        f"- Baseline score: {audit.baseline_score}",
        f"- Baseline rate term: {audit.baseline_rate_term}",
        f"- Baseline component term: {audit.baseline_component_term}",
        "- Score claim: false",
        "- Promotion eligible: false",
        "- Ready for exact eval dispatch: false",
        "",
        "## Hurdles",
        "",
        "| variant | bytes | byte delta | current rate penalty | required component gain now | est floor bytes | required gain after est floor | priority |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in audit.candidates:
        lines.append(
            "| "
            f"`{row.variant}` | "
            f"{row.archive_zip_bytes} | "
            f"{row.archive_byte_delta_vs_baseline} | "
            f"{Decimal(row.current_rate_penalty_vs_baseline):.12g} | "
            f"{Decimal(row.required_component_gain_to_tie_current_bytes):.12g} | "
            f"{row.estimated_floor_archive_bytes} | "
            f"{Decimal(row.required_component_gain_to_tie_after_estimated_floor):.12g} | "
            f"`{row.dispatch_priority}` |"
        )
    lines.extend(["", "## Advisory", ""])
    for row in audit.candidates:
        lines.append(f"- `{row.variant}`: {row.advisory}")
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-json", type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("experiments/results"))
    parser.add_argument("--baseline-archive", type=Path, default=DEFAULT_BASELINE_ARCHIVE)
    parser.add_argument("--baseline-score", default=str(DEFAULT_BASELINE_SCORE))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results") / f"hfv1_pr101_rate_hurdle_{_utc_stamp()}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    readiness_json = args.readiness_json or _latest_readiness_json(args.results_root)
    audit = build_audit(
        readiness_json=readiness_json,
        baseline_archive=args.baseline_archive,
        baseline_score=_safe_score(args.baseline_score),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n"
    (args.output_dir / "hfv1_pr101_rate_hurdle.json").write_text(
        payload,
        encoding="utf-8",
    )
    (args.output_dir / "hfv1_pr101_rate_hurdle.md").write_text(
        render_markdown(audit),
        encoding="utf-8",
    )
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
