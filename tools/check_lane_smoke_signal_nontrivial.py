#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PCC9 — Catch lanes promoted to L2 with zero-delta smoke evidence.

Council Q2 prescription (`feedback_grand_council_predictor_calibration_no_arbitrariness_20260505.md`).
The 2026-05-05 forensic auditor (Finding B) discovered 4 lanes
(`lane_pr106_latent_sidecar`, `_yshift_sidechannel`, `_lrl1_sidechannel`,
`_stacked`) marked L2 with `real_archive_empirical: true`, where the evidence
points to a CPU-smoke `build_metadata.json` containing `scorer_available: false`
and all-zero correction deltas. The wave_deploy script would have wasted
$0.80-1.36 evaluating these PR106-equivalent archives.

This scanner catches that pattern. A lane is flagged if all of:
  1. `real_archive_empirical.status == true` in the lane registry
  2. The evidence string references a path containing `_smoke_` or `_cpu_smoke_`
  3. That path's `build_metadata.json` has `diagnostics.delta_q_min == delta_q_max`
     AND `diagnostics.delta_q_mean == 0.0` (zero-signal smoke build)

Exit codes:
    0    no violations
    1    violations found (only when --strict)
    2    registry parse error

Usage:
    python tools/check_lane_smoke_signal_nontrivial.py [--strict] [--repo-root PATH]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ".omx/state/lane_registry.json"

# Smoke-build paths typically contain `_smoke_`, `_smoke/`, `_smoke.`, or
# end the directory name with `_smoke`. `_cpu_smoke` is a sibling pattern.
SMOKE_PATH_PATTERNS = (
    re.compile(r"_smoke(?:_|/|\.|$)"),
    re.compile(r"_cpu_smoke"),
)


@dataclass(frozen=True)
class Violation:
    lane_id: str
    evidence_path: str
    delta_q_min: float
    delta_q_max: float
    delta_q_mean: float
    scorer_available: bool
    reason: str


def _looks_like_smoke_path(s: str) -> bool:
    return any(p.search(s) for p in SMOKE_PATH_PATTERNS)


def _extract_path_from_evidence(evidence: str) -> Path | None:
    """Find a build_metadata.json reference in the evidence string."""
    if not evidence:
        return None
    # Strip surrounding decoration; evidence is free-form text.
    candidates = re.findall(r"experiments/results/[A-Za-z0-9_./-]+", evidence)
    for c in candidates:
        path_str = c.rstrip(".,;:)")
        if _looks_like_smoke_path(path_str):
            return Path(path_str)
    return None


def _extract_transient_path_from_evidence(evidence: str) -> str | None:
    """Detect transient /tmp/ paths used as 'evidence' (anti-pattern).

    Per user mandate 2026-05-05 ("we need to stop using /tmp by principle"),
    /tmp evidence is itself a violation: it doesn't survive a fresh checkout
    and any agent reading the registry on a different machine can't verify
    the artifact.
    """
    if not evidence:
        return None
    matches = re.findall(r"/tmp/[A-Za-z0-9_./-]+", evidence)
    if matches:
        return matches[0].rstrip(".,;:)")
    return None


def _smoke_dir_metadata(repo_root: Path, evidence_path: Path) -> dict | None:
    """Find build_metadata.json within or adjacent to the evidence path."""
    candidates = [
        repo_root / evidence_path / "build_metadata.json",
        repo_root / evidence_path,  # direct file path
    ]
    if str(evidence_path).endswith("build_metadata.json"):
        candidates.insert(0, repo_root / evidence_path)
    for c in candidates:
        if c.is_file():
            try:
                return json.loads(c.read_text())
            except (OSError, json.JSONDecodeError):
                continue
    # Try parent directory if evidence pointed at the dir itself
    parent_meta = repo_root / evidence_path.parent / "build_metadata.json"
    if parent_meta.is_file():
        try:
            return json.loads(parent_meta.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return None


def scan_registry(repo_root: Path) -> list[Violation]:
    """Return violations: lanes with real_archive_empirical:true backed by zero-delta smoke."""
    registry_path = repo_root / REGISTRY_PATH
    if not registry_path.is_file():
        raise FileNotFoundError(f"lane registry not found: {registry_path}")
    data = json.loads(registry_path.read_text())
    lanes = data.get("lanes", [])

    violations: list[Violation] = []
    for lane in lanes:
        gates = lane.get("gates", {})
        rae = gates.get("real_archive_empirical", {})
        if not rae.get("status"):
            continue
        evidence = rae.get("evidence", "")

        # Detection 0: transient /tmp evidence (user mandate "stop using /tmp")
        transient = _extract_transient_path_from_evidence(evidence)
        if transient is not None:
            violations.append(Violation(
                lane_id=lane.get("id", "?"),
                evidence_path=transient,
                delta_q_min=0.0, delta_q_max=0.0, delta_q_mean=0.0,
                scorer_available=False,
                reason=(
                    f"transient_tmp_evidence: lane evidence references {transient!r} which is /tmp/. "
                    "/tmp paths do not survive a fresh checkout and cannot be verified by other agents. "
                    "Per user mandate 2026-05-05 ('stop using /tmp by principle'), relocate the artifact "
                    "to experiments/results/<lane_id>_<timestamp>/ and update the lane registry evidence."
                ),
            ))
            continue

        smoke_path = _extract_path_from_evidence(evidence)
        if smoke_path is None:
            # Evidence doesn't reference a smoke directory; not in scope.
            continue

        meta = _smoke_dir_metadata(repo_root, smoke_path)
        if meta is None:
            # Evidence references smoke path but no build_metadata.json — flag with note.
            violations.append(Violation(
                lane_id=lane.get("id", "?"),
                evidence_path=str(smoke_path),
                delta_q_min=0.0, delta_q_max=0.0, delta_q_mean=0.0,
                scorer_available=False,
                reason="evidence_references_smoke_dir_but_no_build_metadata",
            ))
            continue

        # Schema 1 (latent_sidecar): diagnostics.delta_q_{min,max,mean}
        diagnostics = meta.get("diagnostics", {})
        delta_min = diagnostics.get("delta_q_min", float("nan"))
        delta_max = diagnostics.get("delta_q_max", float("nan"))
        delta_mean = diagnostics.get("delta_q_mean", float("nan"))
        scorer_available = diagnostics.get("scorer_available", False)

        zero_delta_dx = (delta_min == 0 and delta_max == 0 and delta_mean == 0.0)

        # Schema 2 (yshift/lrl1): search_mode == "zero" indicates all-zero corrections.
        search_mode = meta.get("search_mode", "")
        zero_search_mode = search_mode == "zero"

        # Schema 3 (any): tag contains "[advisory only]" + score_claim=false signals
        # the artifact is wire-format-only, not an empirical signal claim.
        tag = (meta.get("tag") or "").lower()
        score_claim = meta.get("score_claim", None)
        advisory_only = ("[advisory only]" in tag) or (score_claim is False)

        # Schema 4 (any): council_status contains "PROPOSAL" or "PRE_REGISTERED"
        council_status = (meta.get("council_status") or "").upper()
        proposal_only = "PROPOSAL" in council_status or "PRE_REGISTERED" in council_status

        if zero_delta_dx or zero_search_mode or (advisory_only and proposal_only):
            if zero_delta_dx:
                reason_kind = "zero_delta_smoke (diagnostics.delta_q_*)"
            elif zero_search_mode:
                reason_kind = "zero_search_mode (search_mode='zero')"
            else:
                reason_kind = "advisory_proposal_only (tag+score_claim+council_status)"
            violations.append(Violation(
                lane_id=lane.get("id", "?"),
                evidence_path=str(smoke_path),
                delta_q_min=delta_min if not isinstance(delta_min, float) or delta_min == delta_min else 0.0,
                delta_q_max=delta_max if not isinstance(delta_max, float) or delta_max == delta_max else 0.0,
                delta_q_mean=delta_mean if not isinstance(delta_mean, float) or delta_mean == delta_mean else 0.0,
                scorer_available=scorer_available,
                reason=(
                    f"{reason_kind}: archive carries no non-trivial signal "
                    "yet lane is marked real_archive_empirical=true. Archive is bit-equivalent "
                    "to baseline + overhead bytes; dispatching it via exact-eval would just "
                    "re-measure the baseline. Run CUDA Stage 3 refinement before promotion."
                ),
            ))

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--strict", action="store_true",
                        help="exit nonzero on any violation")
    parser.add_argument("--json", action="store_true", help="emit JSON for CI")
    args = parser.parse_args(argv)

    try:
        violations = scan_registry(args.repo_root)
    except FileNotFoundError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"FATAL: registry parse error: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({
            "schema": "lane_smoke_signal_nontrivial_v1",
            "violations": [
                {"lane_id": v.lane_id, "evidence_path": v.evidence_path,
                 "delta_q_min": v.delta_q_min, "delta_q_max": v.delta_q_max,
                 "delta_q_mean": v.delta_q_mean, "scorer_available": v.scorer_available,
                 "reason": v.reason}
                for v in violations
            ],
            "count": len(violations),
        }, indent=2))
    else:
        for v in violations:
            print(f"  [VIOLATION] {v.lane_id}", file=sys.stderr)
            print(f"    evidence_path: {v.evidence_path}", file=sys.stderr)
            print(f"    delta_q (min/max/mean): {v.delta_q_min}/{v.delta_q_max}/{v.delta_q_mean}", file=sys.stderr)
            print(f"    scorer_available: {v.scorer_available}", file=sys.stderr)
            print(f"    reason: {v.reason}", file=sys.stderr)
        if violations:
            print(f"\n[PCC9] {len(violations)} lanes promoted to L2 with zero-delta smoke evidence.", file=sys.stderr)
            print(  # noqa: T201
                "       Backfill via `tools/lane_maturity.py unmark <lane> --gate real_archive_empirical` "
                "or replace evidence with a non-trivial empirical artifact.",
                file=sys.stderr,
            )
        else:
            print("[PCC9] OK: 0 lanes promoted to L2 with zero-delta smoke evidence", file=sys.stderr)

    if violations and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
