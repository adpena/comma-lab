#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan + harvest cross-machine variance probes for the same archive bytes.

Closes A1 PR Council Round 1 finding F6 (CRITICAL — frontier margin
+7.6e-6 < observed cross-x86_64 noise floor 1.6e-5; the "frontier"
delta is within cross-machine noise).

This tool plans N runs of ``contest_auth_eval`` on the SAME archive bytes
across multiple CPU runner types (GHA Linux x86_64 + Modal CPU containers
+ Vast.ai CPU + Lightning CPU Studio), then computes the empirical noise
floor that any future frontier-claim margin must EXCEED to be a real
delta vs measurement noise.

CLAUDE.md non-negotiables honored:

- "DOES NOT auto-dispatch" — emits a typed dispatch plan + writes the
  plan metadata + provides a harvest CLI that the operator runs once
  the GHA workflow_dispatch / Modal jobs return. Real spend ~$0.10-0.50
  per run × 5 = ~$2.50 max; the operator triggers separately.
- "1:1 contest-compliant hardware" — only Linux x86_64 runner types
  permitted. macOS-ARM is NOT permitted because Catalog #192 says
  macOS-CPU is non-promotable advisory.
- "Apples-to-apples evidence discipline" — every plan row pins the
  same archive sha256 and same evaluate.py sha256 and same
  ``public_test_video_names.txt``.
- "Forbidden score claims" — the variance probe writes
  ``score_claim=False`` on every output row (the variance numbers are
  inputs to a noise-floor estimate, not score claims).
- "no_tmp_paths" — output goes to ``reports/`` (or operator-specified
  durable path).

Usage::

    # 1. Plan + write dispatch metadata (NO actual dispatch fires).
    .venv/bin/python tools/measure_cross_machine_variance.py plan \\
        --archive-path submissions_frontier/<label>/archive.zip \\
        --inflate-sh   submissions_frontier/<label>/inflate.sh \\
        --n-runs 5 \\
        --runners gha-ubuntu-latest,modal-cpu-shared,modal-cpu-isolated \\
        --output reports/cross_machine_variance_<utc>.json

    # 2. Operator triggers the GHA + Modal runs separately
    #    (commands printed by the planner).

    # 3. Harvest the returned eval JSONs and compute statistics.
    .venv/bin/python tools/measure_cross_machine_variance.py harvest \\
        --plan reports/cross_machine_variance_<utc>.json \\
        --eval-jsons run1.json,run2.json,run3.json,run4.json,run5.json \\
        --output reports/cross_machine_variance_<utc>.harvested.json

The harvested artifact reports per-runner mean ± std + 95% CI plus the
cross-runner variance, which the operator uses to set a lower bound on
the frontier-margin discipline gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


VARIANCE_PLAN_SCHEMA = "tac_cross_machine_variance_plan_v1"
VARIANCE_HARVEST_SCHEMA = "tac_cross_machine_variance_harvest_v1"


# ── Permitted runner types ────────────────────────────────────────────────
#
# Each runner is one of:
# - "gha-ubuntu-latest"      — GitHub Actions ubuntu-latest workflow_dispatch
# - "gha-ubuntu-24.04"       — same as above but pinned image
# - "modal-cpu-shared"       — Modal CPU container, shared
# - "modal-cpu-isolated"     — Modal CPU container, isolated VM
# - "vast-cpu-x86_64"        — Vast.ai CPU instance, x86_64 Ubuntu
# - "lightning-cpu-studio"   — Lightning CPU Studio
#
# macOS-ARM is forbidden per Catalog #192. CUDA runners are forbidden because
# the variance probe is for the CPU public-leaderboard axis specifically.

ALLOWED_RUNNERS = frozenset({
    "gha-ubuntu-latest",
    "gha-ubuntu-24.04",
    "modal-cpu-shared",
    "modal-cpu-isolated",
    "vast-cpu-x86_64",
    "lightning-cpu-studio",
})

FORBIDDEN_RUNNERS_REASONS = {
    "macos-arm64": "Catalog #192: macOS-CPU is advisory-only; non-promotable",
    "macos-cpu": "Catalog #192: macOS-CPU is advisory-only; non-promotable",
    "mps": "CLAUDE.md MPS auth eval is NOISE",
    "gha-cuda": "variance probe is for CPU public-leaderboard axis",
    "vast-gpu": "variance probe is for CPU public-leaderboard axis",
}


# Estimated cost-per-run (USD) for budget-cap accounting. Used for the
# cumulative-cost gate ONLY — not for the actual dispatch (which the
# operator triggers separately).
RUNNER_ESTIMATED_COST_USD = {
    "gha-ubuntu-latest": 0.0,  # GHA free tier covers public repos
    "gha-ubuntu-24.04": 0.0,
    "modal-cpu-shared": 0.10,
    "modal-cpu-isolated": 0.20,
    "vast-cpu-x86_64": 0.05,
    "lightning-cpu-studio": 0.10,
}


# ── Plan dataclasses ──────────────────────────────────────────────────────


@dataclass
class DispatchPlanRow:
    """One planned dispatch — runner type + run index + dispatch command stub."""

    runner: str
    run_index: int
    estimated_cost_usd: float
    dispatch_command_stub: str
    output_artifact_relpath: str


@dataclass
class CrossMachineVariancePlan:
    """Typed cross-machine variance probe plan."""

    schema: str
    archive_path: str
    archive_sha256: str
    archive_bytes: int
    inflate_sh_path: str
    n_runs_per_runner: int
    runners: list[str]
    plan_rows: list[DispatchPlanRow]
    estimated_total_cost_usd: float
    planned_at_utc: str
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "archive_path": self.archive_path,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "inflate_sh_path": self.inflate_sh_path,
            "n_runs_per_runner": self.n_runs_per_runner,
            "runners": list(self.runners),
            "plan_rows": [
                {
                    "runner": r.runner,
                    "run_index": r.run_index,
                    "estimated_cost_usd": r.estimated_cost_usd,
                    "dispatch_command_stub": r.dispatch_command_stub,
                    "output_artifact_relpath": r.output_artifact_relpath,
                }
                for r in self.plan_rows
            ],
            "estimated_total_cost_usd": self.estimated_total_cost_usd,
            "planned_at_utc": self.planned_at_utc,
            "notes": self.notes,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "claude_md_compliance_tags": [
                "f6_council_round_1_critical_remediation",
                "no_auto_dispatch",
                "1to1_contest_compliant_runners_only",
                "no_macos_cpu_authoritative",
                "no_score_claim_advanced_by_this_artifact",
                "apples_to_apples_paired_archive_sha256",
            ],
        }


# ── Filesystem helpers ────────────────────────────────────────────────────


def _sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


# ── Validation ────────────────────────────────────────────────────────────


class VariancePlanRefused(Exception):
    """Raised when plan inputs violate a contract."""

    def __init__(self, reason_class: str, message: str) -> None:
        super().__init__(message)
        self.reason_class = reason_class


def _normalize_runners(runners: Iterable[str]) -> list[str]:
    out: list[str] = []
    for raw in runners:
        r = raw.strip().lower()
        if not r:
            continue
        if r in FORBIDDEN_RUNNERS_REASONS:
            raise VariancePlanRefused(
                "forbidden_runner",
                f"runner {r!r} forbidden: {FORBIDDEN_RUNNERS_REASONS[r]}",
            )
        if r not in ALLOWED_RUNNERS:
            raise VariancePlanRefused(
                "unknown_runner",
                f"runner {r!r} not in allowed set: {sorted(ALLOWED_RUNNERS)}",
            )
        out.append(r)
    if not out:
        raise VariancePlanRefused(
            "no_runners",
            "must specify at least one runner",
        )
    return out


# ── Dispatch command stubs ────────────────────────────────────────────────


def _dispatch_command_stub(
    runner: str,
    *,
    archive_path: str,
    inflate_sh_path: str,
    output_relpath: str,
    archive_sha256: str,
) -> str:
    """Render a stub command the operator runs to actually fire one run.

    The stubs are intentionally NOT executed by this tool — they are
    suggestions the operator can copy/paste OR feed into a separate
    orchestrator. The variance probe never auto-dispatches.
    """
    sha_short = archive_sha256[:12]
    if runner.startswith("gha-"):
        return (
            f"# GHA workflow_dispatch ({runner}) for archive sha={sha_short}:\n"
            f"gh workflow run contest_cpu_eval.yml \\\n"
            f"    -f archive_path={archive_path} \\\n"
            f"    -f inflate_sh_path={inflate_sh_path} \\\n"
            f"    -f lane_id=cross_machine_variance_{sha_short}_{runner} \\\n"
            f"    --ref main"
        )
    if runner.startswith("modal-"):
        kind = "shared" if runner.endswith("shared") else "isolated"
        return (
            f"# Modal CPU ({kind}) for archive sha={sha_short}; verify wrapper flags before spend:\n"
            f".venv/bin/modal run --detach experiments/modal_auth_eval_cpu.py -- \\\n"
            f"    --archive {archive_path} \\\n"
            f"    --inflate-sh {inflate_sh_path} \\\n"
            f"    --output-dir {Path(output_relpath).parent.as_posix()} \\\n"
            f"    --lane-id cross_machine_variance_{sha_short}_{runner} \\\n"
            f"    --instance-job-id cross_machine_variance_{sha_short}_{runner} \\\n"
            f"    --detach --provider-detach-ack"
        )
    if runner.startswith("vast-"):
        return (
            f"# Vast.ai CPU x86_64 for archive sha={sha_short}:\n"
            f".venv/bin/python scripts/launch_lane_on_vastai.py \\\n"
            f"    --image ubuntu:24.04 \\\n"
            f"    --gpu-type none \\\n"
            f"    --cpu-only \\\n"
            f"    --bootstrap scripts/remote_archive_only_eval.sh \\\n"
            f"    --archive-path {archive_path} \\\n"
            f"    --inflate-sh {inflate_sh_path} \\\n"
            f"    --output {output_relpath} \\\n"
            f"    --expected-archive-sha256 {archive_sha256}"
        )
    if runner.startswith("lightning-"):
        return (
            f"# Lightning CPU Studio for archive sha={sha_short}:\n"
            f".venv/bin/python scripts/launch_lane_lightning.py \\\n"
            f"    --studio-kind cpu \\\n"
            f"    --archive-path {archive_path} \\\n"
            f"    --inflate-sh {inflate_sh_path} \\\n"
            f"    --output {output_relpath} \\\n"
            f"    --expected-archive-sha256 {archive_sha256}"
        )
    return f"# unknown runner {runner!r}; operator must build dispatch command"


# ── Plan builder ──────────────────────────────────────────────────────────


def build_plan(
    *,
    archive_path: Path,
    inflate_sh_path: Path,
    n_runs: int,
    runners: list[str],
    output_dir: str = "reports/cross_machine_variance_runs",
    notes: str = "",
) -> CrossMachineVariancePlan:
    """Build the variance-probe plan dict.

    No filesystem state is mutated besides reading the archive + inflate.sh.
    The plan is returned for the caller to write to disk.
    """
    if n_runs <= 0:
        raise VariancePlanRefused(
            "invalid_n_runs",
            f"n_runs must be > 0, got {n_runs}",
        )
    if n_runs < 3:
        # Mean +/- std + 95% CI requires >=3 samples for reasonable coverage.
        # Warn loudly but allow.
        print(
            f"WARN: n_runs={n_runs} < 3; statistics will be unreliable",
            file=sys.stderr,
        )
    if not archive_path.is_file():
        raise VariancePlanRefused(
            "archive_missing",
            f"archive not found: {archive_path}",
        )
    if not inflate_sh_path.is_file():
        raise VariancePlanRefused(
            "inflate_sh_missing",
            f"inflate.sh not found: {inflate_sh_path}",
        )

    normalized_runners = _normalize_runners(runners)

    archive_sha256, archive_bytes = _sha256_file(archive_path)
    archive_path_str = str(archive_path)
    inflate_sh_str = str(inflate_sh_path)

    rows: list[DispatchPlanRow] = []
    total_cost = 0.0
    for runner in normalized_runners:
        cost_per = RUNNER_ESTIMATED_COST_USD.get(runner, 0.10)
        for i in range(n_runs):
            output_relpath = (
                f"{output_dir}/{archive_sha256[:12]}/{runner}/run_{i + 1:03d}.json"
            )
            stub = _dispatch_command_stub(
                runner,
                archive_path=archive_path_str,
                inflate_sh_path=inflate_sh_str,
                output_relpath=output_relpath,
                archive_sha256=archive_sha256,
            )
            rows.append(
                DispatchPlanRow(
                    runner=runner,
                    run_index=i + 1,
                    estimated_cost_usd=cost_per,
                    dispatch_command_stub=stub,
                    output_artifact_relpath=output_relpath,
                )
            )
            total_cost += cost_per

    return CrossMachineVariancePlan(
        schema=VARIANCE_PLAN_SCHEMA,
        archive_path=archive_path_str,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        inflate_sh_path=inflate_sh_str,
        n_runs_per_runner=n_runs,
        runners=normalized_runners,
        plan_rows=rows,
        estimated_total_cost_usd=total_cost,
        planned_at_utc=datetime.now(timezone.utc).isoformat(),
        notes=notes,
    )


# ── Harvest + statistics ──────────────────────────────────────────────────


def _extract_score_value(payload: dict[str, Any]) -> float | None:
    for key in (
        "canonical_score_recomputed",
        "score_recomputed_from_components",
        "canonical_score",
        "score_value",
        "score",
    ):
        v = payload.get(key)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _extract_runner_label(payload: dict[str, Any]) -> str:
    """Heuristic runner label extracted from harvested eval JSON."""
    hardware = str(payload.get("hardware") or "").lower()
    if "github" in hardware or "gha" in hardware or "ubuntu" in hardware:
        return "gha-ubuntu-latest"
    if "modal" in hardware:
        if "isolated" in hardware:
            return "modal-cpu-isolated"
        return "modal-cpu-shared"
    if "vast" in hardware:
        return "vast-cpu-x86_64"
    if "lightning" in hardware:
        return "lightning-cpu-studio"
    return hardware or "unknown"


def _t_critical_95_two_sided(n: int) -> float:
    """Two-sided 95% t-critical for small samples (n=df+1).

    Hard-coded for n in 2..10; uses 1.96 (z) for n >= 30 and a
    conservative interpolation between.
    """
    table = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
        6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
        15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000,
    }
    if n <= 1:
        return float("inf")
    if n in table:
        return table[n]
    # Conservative bound via the largest table entry below n.
    keys = sorted(table.keys())
    for k in reversed(keys):
        if k <= n:
            return table[k]
    return 1.96


def compute_statistics(scores_by_runner: dict[str, list[float]]) -> dict[str, Any]:
    """Compute mean/std/CI plus within-runner and total cross-runner variance.

    Returns a dict with keys:

    - ``per_runner``: dict[runner -> {n, mean, std, ci_95_low, ci_95_high}]
    - ``cross_runner_pooled_within_std``: repeat noise inside each runner
    - ``cross_runner_total_std``: all harvested scores pooled together,
      including stable between-runner offsets
    - ``cross_runner_pooled_std``: backward-compatible alias for total std
    - ``cross_runner_grand_mean``
    - ``cross_runner_n_total``
    - ``empirical_noise_floor`` — the recommended lower-bound that any
      "frontier" margin must exceed to be a real delta vs measurement
      noise. Set to ``2 * cross_runner_total_std`` as a conservative
      ~95% confidence floor.
    """
    per_runner: dict[str, dict[str, Any]] = {}
    all_scores: list[float] = []
    pooled_within_runner_var_numerator = 0.0
    pooled_within_runner_var_denominator = 0
    for runner, vals in scores_by_runner.items():
        n = len(vals)
        if n == 0:
            per_runner[runner] = {
                "n": 0,
                "mean": None,
                "std": None,
                "ci_95_low": None,
                "ci_95_high": None,
            }
            continue
        mean = statistics.fmean(vals)
        if n == 1:
            std = 0.0
            ci_low, ci_high = mean, mean
        else:
            std = statistics.stdev(vals)
            t = _t_critical_95_two_sided(n)
            half = t * std / math.sqrt(n)
            ci_low, ci_high = mean - half, mean + half
            # Pooled within-runner variance (sum of squared deviations / (n-1)
            # weighted by (n-1) === sum of squared deviations).
            pooled_within_runner_var_numerator += sum((v - mean) ** 2 for v in vals)
            pooled_within_runner_var_denominator += n - 1
        per_runner[runner] = {
            "n": n,
            "mean": mean,
            "std": std,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
        }
        all_scores.extend(vals)

    if pooled_within_runner_var_denominator > 0:
        pooled_within_var = (
            pooled_within_runner_var_numerator
            / pooled_within_runner_var_denominator
        )
        pooled_std = math.sqrt(pooled_within_var)
    else:
        pooled_std = 0.0

    grand_mean = statistics.fmean(all_scores) if all_scores else None
    n_total = len(all_scores)
    total_std = statistics.stdev(all_scores) if n_total > 1 else 0.0

    return {
        "per_runner": per_runner,
        "cross_runner_pooled_within_std": pooled_std,
        "cross_runner_total_std": total_std,
        "cross_runner_pooled_std": total_std,
        "cross_runner_grand_mean": grand_mean,
        "cross_runner_n_total": n_total,
        "empirical_noise_floor": 2.0 * total_std,
        "empirical_noise_floor_definition": (
            "2 * cross_runner_total_std (~95% confidence band on a single "
            "future measurement vs the pooled mean, including stable "
            "between-runner offsets). Frontier-margin discipline gate: any "
            "claimed delta MUST exceed this floor to be a real delta vs "
            "cross-machine noise."
        ),
    }


def harvest_eval_json(
    eval_json_path: Path,
    expected_sha256: str,
    *,
    planned_runners: set[str] | None = None,
) -> dict[str, Any]:
    """Read an eval JSON and return ``(runner_label, score_value)`` payload row.

    Raises :class:`VariancePlanRefused` if the eval JSON references a
    different archive sha (apples-to-apples discipline).
    """
    if not eval_json_path.is_file():
        raise VariancePlanRefused(
            "eval_json_missing",
            f"eval JSON not found: {eval_json_path}",
        )
    payload = json.loads(eval_json_path.read_text(encoding="utf-8"))
    sha = str(payload.get("archive_sha256") or "").strip().lower()
    if sha != expected_sha256.lower():
        raise VariancePlanRefused(
            "archive_sha_mismatch",
            f"eval JSON sha={sha[:12]} does not match plan sha={expected_sha256[:12]}",
        )
    n_samples = payload.get("n_samples")
    if n_samples != 600:
        raise VariancePlanRefused(
            "sample_count_not_600",
            f"eval JSON n_samples={n_samples!r}; variance probes require n_samples=600",
        )
    provenance = payload.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    device = str(payload.get("device") or provenance.get("device") or "").lower()
    if device != "cpu":
        raise VariancePlanRefused(
            "non_cpu_eval_json",
            f"variance probe harvest requires CPU eval JSON, got device={device!r}",
        )
    lane_tag = str(payload.get("lane_tag") or payload.get("evidence_tag") or "")
    score_axis = str(payload.get("score_axis") or "").lower()
    evidence_semantics = str(payload.get("evidence_semantics") or "").lower()
    if not lane_tag.startswith("[contest-CPU]") and not lane_tag.startswith("[contest-CPU "):  # CUSTODY_VALIDATOR_OK: CPU variance harvester validates archive sha, sample count, device, score_axis, and evidence_semantics in this function.
        raise VariancePlanRefused(
            "non_contest_cpu_eval_json",
            f"eval JSON lane_tag={lane_tag!r}; expected [contest-CPU...]",
        )
    if score_axis != "contest_cpu":
        raise VariancePlanRefused(
            "non_contest_cpu_eval_json",
            f"eval JSON score_axis={score_axis!r}; expected contest_cpu",
        )
    if evidence_semantics and evidence_semantics != "public_leaderboard_cpu_reproduction":
        raise VariancePlanRefused(
            "non_contest_cpu_eval_json",
            f"eval JSON evidence_semantics={evidence_semantics!r}",
        )
    eligible = payload.get("cpu_leaderboard_reproduction_eligible")
    if eligible is not None and eligible is not True:
        raise VariancePlanRefused(
            "non_contest_cpu_eval_json",
            f"cpu_leaderboard_reproduction_eligible={eligible!r}",
        )
    if payload.get("score_claim") is True:
        raise VariancePlanRefused(
            "score_claim_must_be_false",
            "CPU variance probes must not carry score_claim=True",
        )
    score = _extract_score_value(payload)
    if score is None:
        raise VariancePlanRefused(
            "score_value_missing",
            f"eval JSON {eval_json_path} has no canonical_score* / score_value field",
        )
    runner = _extract_runner_label(payload)
    if runner == "unknown" or (planned_runners is not None and runner not in planned_runners):
        raise VariancePlanRefused(
            "runner_not_in_plan",
            f"eval JSON runner={runner!r} not in planned runners={sorted(planned_runners or [])}",
        )
    return {
        "eval_json_path": str(eval_json_path),
        "runner": runner,
        "score_value": float(score),
        "n_samples": n_samples,
        "score_axis": score_axis,
        "evidence_semantics": evidence_semantics,
    }


def harvest(
    *,
    plan: dict[str, Any],
    eval_json_paths: list[Path],
    notes: str = "",
) -> dict[str, Any]:
    """Harvest a list of eval JSONs against a plan and return the variance report."""
    expected_sha = str(plan.get("archive_sha256") or "")
    if not expected_sha:
        raise VariancePlanRefused(
            "plan_missing_archive_sha",
            "plan missing archive_sha256",
        )
    planned_runners = {
        str(r).strip()
        for r in plan.get("runners", [])
        if isinstance(r, str) and str(r).strip()
    }
    rows = [
        harvest_eval_json(p, expected_sha, planned_runners=planned_runners or None)
        for p in eval_json_paths
    ]
    scores_by_runner: dict[str, list[float]] = {}
    for r in rows:
        scores_by_runner.setdefault(r["runner"], []).append(r["score_value"])
    stats = compute_statistics(scores_by_runner)
    return {
        "schema": VARIANCE_HARVEST_SCHEMA,
        "plan_archive_sha256": expected_sha,
        "plan_archive_bytes": plan.get("archive_bytes"),
        "n_eval_jsons_harvested": len(rows),
        "harvested_rows": rows,
        "statistics": stats,
        "harvested_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "claude_md_compliance_tags": [
            "f6_council_round_1_critical_remediation",
            "no_score_claim_advanced_by_this_artifact",
            "1to1_contest_compliant_runners_only",
        ],
    }


# ── CLI ───────────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plan + harvest cross-machine variance probes for the same "
            "archive bytes. Closes A1 PR Council Round 1 finding F6 "
            "(CRITICAL — frontier margin within cross-machine noise)."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    plan_p = sub.add_parser("plan", help="Build a dispatch plan; NO auto-dispatch")
    plan_p.add_argument("--archive-path", required=True, type=Path)
    plan_p.add_argument("--inflate-sh", required=True, type=Path)
    plan_p.add_argument("--n-runs", required=True, type=int,
                        help="Number of runs PER runner (>=3 recommended)")
    plan_p.add_argument(
        "--runners", required=True,
        help="Comma-separated runner labels (e.g. gha-ubuntu-latest,modal-cpu-shared)",
    )
    plan_p.add_argument(
        "--output", required=True, type=Path,
        help="Where to write the plan JSON (under reports/ or .omx/)",
    )
    plan_p.add_argument("--output-dir", default="reports/cross_machine_variance_runs",
                        help="Subdir for harvested per-run output JSONs")
    plan_p.add_argument("--notes", default="")

    harvest_p = sub.add_parser(
        "harvest", help="Harvest returned eval JSONs against an existing plan"
    )
    harvest_p.add_argument("--plan", required=True, type=Path)
    harvest_p.add_argument(
        "--eval-jsons", required=True,
        help="Comma-separated paths to harvested eval JSONs",
    )
    harvest_p.add_argument("--output", required=True, type=Path)
    harvest_p.add_argument("--notes", default="")
    return p


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.cmd == "plan":
        try:
            plan = build_plan(
                archive_path=args.archive_path,
                inflate_sh_path=args.inflate_sh,
                n_runs=args.n_runs,
                runners=args.runners.split(","),
                output_dir=args.output_dir,
                notes=args.notes,
            )
        except VariancePlanRefused as exc:
            print(f"REFUSED [{exc.reason_class}]: {exc}", file=sys.stderr)
            return 2
        _write_json_atomic(plan.to_json(), args.output)
        print(f"PLAN WRITTEN: {args.output}")
        print(f"  archive.sha256:        {plan.archive_sha256}")
        print(f"  n_runs_per_runner:     {plan.n_runs_per_runner}")
        print(f"  runners:               {plan.runners}")
        print(f"  total_planned_runs:    {len(plan.plan_rows)}")
        print(f"  estimated_total_cost_usd: ${plan.estimated_total_cost_usd:.2f}")
        print()
        print("Operator: trigger each plan_row's dispatch_command_stub separately.")
        print("When all eval JSONs are returned, run `harvest` to compute statistics.")
        return 0

    if args.cmd == "harvest":
        try:
            plan = json.loads(args.plan.read_text(encoding="utf-8"))
            eval_paths = [Path(p.strip()) for p in args.eval_jsons.split(",") if p.strip()]
            report = harvest(
                plan=plan,
                eval_json_paths=eval_paths,
                notes=args.notes,
            )
        except VariancePlanRefused as exc:
            print(f"REFUSED [{exc.reason_class}]: {exc}", file=sys.stderr)
            return 2
        _write_json_atomic(report, args.output)
        stats = report["statistics"]
        print(f"HARVEST WRITTEN: {args.output}")
        print(f"  n_harvested:                {report['n_eval_jsons_harvested']}")
        print(f"  cross_runner_grand_mean:    {stats['cross_runner_grand_mean']}")
        print(f"  cross_runner_pooled_std:    {stats['cross_runner_pooled_std']}")
        print(f"  EMPIRICAL_NOISE_FLOOR:      {stats['empirical_noise_floor']}")
        print()
        print("Frontier-margin discipline: any claimed delta MUST exceed")
        print(f"the noise floor ({stats['empirical_noise_floor']}) to be a real")
        print("delta vs cross-machine noise per CLAUDE.md apples-to-apples discipline.")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
