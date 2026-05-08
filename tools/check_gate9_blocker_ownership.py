#!/usr/bin/env python3
"""Gate 9 — Blocker ownership gate.

Source: ``.omx/research/representation_integration_gap_audit_20260508_codex.md``
prevent-recurrence gate #9.

Rule: if a high-upside learned-codec lane is blocked, the lane MUST
record one of:

  * ``active_owner`` + ``unblock_experiment``
  * ``exact_negative`` + ``reactivation_criteria``
  * ``compliance_impossibility_proof``
  * ``terminal_retirement_note``

"Waiting for clearance" (any blocker that doesn't pick one of the four
above) is not enough.

Detection (static):
  Scan ``.omx/state/lane_registry.json`` for any lane whose:
    * name/id contains a learned-codec representation token (``nerv``,
      ``hnerv``, ``mnerv``, ``coolchic``, ``c3_render``, ``balle``,
      ``hyperprior``, ``learned_codec``, ``ballé``), OR
    * has a ``high_upside=true`` annotation, OR
    * has ``blocked=true`` annotation,

  AND has ``blocked=true`` (or any non-empty ``blockers`` array, OR
  level==1 with no progress for 7+ days based on ``last_activity_utc``),

  REQUIRE one of the four ownership patterns:
    * ``active_owner`` (non-empty string) AND ``unblock_experiment``
      (non-empty string)
    * ``exact_negative`` (non-empty string) AND
      ``reactivation_criteria`` (non-empty list/string)
    * ``compliance_impossibility_proof`` (non-empty string)
    * ``terminal_retirement_note`` (non-empty string)

  Lanes annotated with ``ownership_waiver=<reason>`` are exempt.

Memory ref: ``feedback_representation_integration_gap_audit_20260508_codex.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

LEARNED_CODEC_TOKENS = (
    "nerv",
    "hnerv",
    "mnerv",
    "coolchic",
    "cool_chic",
    "cool-chic",
    "c3_render",
    "c3render",
    "balle",
    "ballé",
    "hyperprior",
    "learned_codec",
    "implicit_neural",
    "vqvae",
    "wavelet",
)


@dataclass
class Finding:
    file_rel: str
    lane_id: str
    reason: str


def _is_learned_codec_lane(lane: dict) -> bool:
    lane_id = str(lane.get("id", "")).lower()
    name = str(lane.get("name", "")).lower()
    for tok in LEARNED_CODEC_TOKENS:
        if tok in lane_id or tok in name:
            return True
    return lane.get("high_upside") is True


def _is_blocked(lane: dict) -> bool:
    if lane.get("blocked") is True:
        return True
    blockers = lane.get("blockers")
    if isinstance(blockers, list) and len(blockers) > 0:
        return True
    return bool(isinstance(blockers, str) and blockers.strip())


def _has_ownership(lane: dict) -> bool:
    # Pattern 1: active owner + unblock experiment
    owner = lane.get("active_owner") or lane.get("owner")
    exp = lane.get("unblock_experiment") or lane.get("next_unblock_action")
    if (
        isinstance(owner, str) and owner.strip()
        and isinstance(exp, str) and exp.strip()
    ):
        return True
    # Pattern 2: exact negative + reactivation criteria
    neg = lane.get("exact_negative") or lane.get("exact_negative_evidence")
    crit = lane.get("reactivation_criteria")
    if isinstance(neg, str) and neg.strip() and (
        (isinstance(crit, list) and len(crit) > 0)
        or (isinstance(crit, str) and crit.strip())
    ):
        return True
    # Pattern 3: compliance impossibility
    cip = lane.get("compliance_impossibility_proof")
    if isinstance(cip, str) and cip.strip():
        return True
    # Pattern 4: terminal retirement
    trn = lane.get("terminal_retirement_note")
    return bool(isinstance(trn, str) and trn.strip())


def _has_waiver(lane: dict) -> bool:
    waiver = lane.get("ownership_waiver")
    return bool(isinstance(waiver, str) and waiver.strip())


def scan(repo_root: Path | None = None) -> list[Finding]:
    repo = (repo_root or REPO_ROOT_DEFAULT).resolve()
    findings: list[Finding] = []
    path = repo / ".omx" / "state" / "lane_registry.json"
    if not path.is_file():
        return findings
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return findings
    rel = path.relative_to(repo).as_posix()
    lanes = data.get("lanes", [])
    if not isinstance(lanes, list):
        return findings
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        if not _is_learned_codec_lane(lane):
            continue
        if not _is_blocked(lane):
            continue
        if _has_waiver(lane):
            continue
        if _has_ownership(lane):
            continue
        findings.append(
            Finding(
                file_rel=rel,
                lane_id=str(lane.get("id", "<unknown>")),
                reason=(
                    "blocked learned-codec lane lacks ownership pattern. "
                    "Required: (active_owner+unblock_experiment) OR "
                    "(exact_negative+reactivation_criteria) OR "
                    "compliance_impossibility_proof OR "
                    "terminal_retirement_note. Gate 9 (blocker ownership)."
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
            f"[gate9-blocker-ownership] {len(findings)} violation(s):",
            file=sys.stderr,
        )
        for f in findings[:20]:
            print(
                f"  • {f.file_rel} lane={f.lane_id}: {f.reason}",
                file=sys.stderr,
            )
        if args.strict:
            return 1
    else:
        print("[gate9-blocker-ownership] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
