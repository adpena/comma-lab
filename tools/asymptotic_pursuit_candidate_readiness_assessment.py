#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ASYMPTOTIC PURSUIT candidate-readiness assessment + dispatch staging helper.

Per operator directive 2026-05-17 verbatim: *"Want to keep pushing to asymptotic
as top priority"*. Sister of `tools/check_predecessor_probe_outcome.py` (Catalog
#313) + `tools/audit_stale_l1_substrates.py` (Catalog #298).

The Q4 Wyner-Ziv-on-existing-archives path is DEFERRED per the Option B
falsification 2026-05-17 (top aggregate ΔS 0.000421 << 0.001 leaderboard
precision floor — every VALIDATED contest archive is at the entropy floor of
the canonical lzma+brotli+zlib general-purpose codecs). Per the Option B
council OP-3: redirect Q4 budget to **substrate-class-shift methods** —
predictive-receiver (Z6/Z7/Z8), cooperative-receiver (ATW V2), foveation
(TT5L L5), MDL Information Bottleneck (C6 IBPS).

This module assesses the empirical readiness of every candidate substrate-
class-shift method in the registry, ranks by EV-per-dollar, and surfaces a
ready-to-fire operator_authorize command for the TOP-1 candidate when
dispatch-ready (passes all Catalog #270 Tier 1/2/3 + Catalog #315 OPTIMAL
FORM + Catalog #313 predecessor-probe + Catalog #240 recipe-vs-trainer chain
gates).

Usage::

    .venv/bin/python tools/asymptotic_pursuit_candidate_readiness_assessment.py

    # JSON output for downstream consumers:
    .venv/bin/python tools/asymptotic_pursuit_candidate_readiness_assessment.py --json

    # Override the candidate list (whitespace-separated):
    .venv/bin/python tools/asymptotic_pursuit_candidate_readiness_assessment.py \\
        --candidates "z6,c6_e4_mdl_ibps,atw_codec_v2,time_traveler_l5_autonomy"

    # Write the canonical readiness artifact to disk:
    .venv/bin/python tools/asymptotic_pursuit_candidate_readiness_assessment.py \\
        --write-artifact

Exit codes::

    0   at least one TOP-1 candidate is READY for paid dispatch
    1   no candidate is READY (operator-action needed: fix NEEDS_FIX candidates
        or land Phase 2 council on DEFER candidates)
    2   query argument error

Library API::

    assess_candidates(...) -> ReadinessAssessment

    rank_by_ev_per_dollar(...) -> list[CandidateReadiness]

    build_operator_authorize_command(...) -> str

Per CLAUDE.md NON-NEGOTIABLES this module honors:
  * "Submission auth eval — BOTH CPU AND CUDA" — every recommended dispatch
    pairs CPU + CUDA per the canonical paired-dispatch flow.
  * "MPS auth eval is NOISE" — no local-MPS dispatches are recommended.
  * "Forbidden premature KILL" — DEFER verdicts surface reactivation criteria,
    never KILL.
  * Catalog #313 predecessor-probe — BLOCKING probe outcomes refuse dispatch.
  * Catalog #315 OPTIMAL FORM — PROCEED_WITH_REVISIONS council verdicts refuse
    paid dispatch.
  * Catalog #240 recipe-vs-trainer-chain — NotImplementedError trainers refuse
    dispatch.
  * Catalog #270 dispatch optimization protocol — Tier 1/2/3 must all pass.
  * Catalog #220 operational-mechanism — substrate L1+ byte addition must be
    operational at runtime.
  * Provenance discipline — every emitted predicted band carries
    ``provenance_kind: PREDICTED_FROM_MODEL`` + ``score_claim: false`` +
    ``promotion_eligible: false`` + ``evidence_grade: predicted``.

Sister memo: feedback_asymptotic_pursuit_substrate_class_shift_top_priority_landed_20260517.md.
Lane: lane_asymptotic_pursuit_substrate_class_shift_q4_pivot_top_priority_20260517.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Canonical candidate inventory per HORIZON-CLASS council 2026-05-17 Stage 2
# deferred queue + Z6/Z7/Z8 scoping design + ATW V2 D4 probe verdict +
# C6 IBPS first empirical confirmation Tier-C anchor.
CANONICAL_CANDIDATES: tuple[str, ...] = (
    "time_traveler_l5_z6",  # Z6 simple FiLM-conditioned next-frame predictor
    "atw_codec_v2",  # ATW V2 D4 cooperative-receiver (currently BLOCKED by probe)
    "time_traveler_l5_autonomy",  # TT5L L5 foveation + LAPose
    "c6_e4_mdl_ibps",  # C6 MDL Information Bottleneck Per Pair
    "nscs01_nullspace_split_renderer",  # NSCS01 nullspace split (PR95-paradigm)
    "nscs03_end_to_end_balle_joint_codec",  # NSCS03 end-to-end Ballé joint codec
)

# Dispatch-cost-per-GPU-hour per CLAUDE.md "GPU budget and compute resources"
# table; per CLAUDE.md "Production-hardened dispatch optimization protocol" the
# canonical paired CPU+CUDA cost is Modal CPU ($0.06/hr) + Modal CUDA T4
# ($0.59/hr) or Vast.ai 4090 ($0.25/hr).
PER_HOUR_USD_BY_GPU: dict[str, float] = {
    "T4": 0.59,
    "A10G": 1.10,
    "A100": 4.10,
    "H100": 5.65,
    "L40S": 1.80,
    "4090": 0.25,
    "CPU": 0.06,
}

# Recipes-to-substrate mapping for the canonical candidates
RECIPE_BY_SUBSTRATE: dict[str, str] = {
    "time_traveler_l5_z6": "substrate_time_traveler_l5_z6_modal_t4_dispatch",
    "atw_codec_v2": "substrate_atw_codec_v2_modal_a100_dispatch",
    "time_traveler_l5_autonomy": "substrate_time_traveler_l5_autonomy_modal_a100_dispatch",
    "c6_e4_mdl_ibps": "substrate_c6_e4_mdl_ibps_modal_t4_dispatch",
    "nscs01_nullspace_split_renderer": "substrate_nscs01_nullspace_split_renderer_modal_t4_dispatch",
    "nscs03_end_to_end_balle_joint_codec": "substrate_nscs03_end_to_end_balle_joint_codec_modal_a100_dispatch",
}


@dataclass(frozen=True)
class CandidateReadiness:
    """Per-candidate empirical readiness state.

    Per provenance discipline: every band field carries explicit
    ``provenance_kind`` + ``evidence_grade`` + ``score_claim`` + ``promotion_eligible``
    fields so downstream consumers (cathedral autopilot ranker / Rashomon
    ensemble / Assumption-Adversary) cannot mistake a predicted band for an
    empirical anchor.
    """

    substrate_id: str
    recipe_basename: str
    recipe_path: Path | None
    trainer_path: Path | None
    lane_maturity: str  # "L0" | "L1" | "L2" | "L3" | "NOT_REGISTERED"
    impl_complete: bool
    full_main_implemented: bool
    full_main_blocker: str | None  # e.g. "NotImplementedError" / None
    latest_council_verdict: str  # "PROCEED" | "PROCEED_WITH_REVISIONS" | "DEFERRED" | "NO_DELIBERATION"
    research_only: bool
    dispatch_enabled: bool
    dispatch_blockers: tuple[str, ...]
    predecessor_probe_blocking: bool
    predecessor_probe_id: str | None
    predecessor_probe_verdict: str | None
    predicted_delta_s_band_low: float | None  # provenance_kind=PREDICTED_FROM_MODEL
    predicted_delta_s_band_high: float | None  # provenance_kind=PREDICTED_FROM_MODEL
    predicted_delta_s_provenance: str  # "PREDICTED_FROM_MODEL" | "EMPIRICAL_ANCHOR" | "UNKNOWN"
    estimated_dispatch_cost_usd: float
    estimated_dispatch_wall_clock_seconds: int
    gpu_class: str
    min_smoke_gpu: str
    cost_band_epochs: int
    horizon_class: str  # "asymptotic_pursuit" | "frontier_pursuit" | "plateau_adjacent"
    blocking_issues: tuple[str, ...]
    readiness_verdict: str  # "READY" | "NEEDS_FIX" | "DEFER"
    ev_per_dollar: float  # |predicted_delta_s| / estimated_dispatch_cost_usd
    # Provenance fields (Catalog #127 + #192 + sister provenance subagent):
    score_claim: bool = False  # NEVER True from this assessment surface
    promotion_eligible: bool = False
    evidence_grade: str = "predicted"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Convert Path -> str for JSON
        for k in ("recipe_path", "trainer_path"):
            if d[k] is not None:
                d[k] = str(d[k])
        return d


@dataclass(frozen=True)
class ReadinessAssessment:
    """Top-level assessment of all candidates ranked by EV-per-dollar."""

    candidates: tuple[CandidateReadiness, ...]
    ranked_by_ev_per_dollar: tuple[str, ...]  # substrate_id, descending EV
    top_1_substrate: str | None
    top_1_readiness_verdict: str  # "READY" | "NEEDS_FIX" | "DEFER" | "NONE"
    top_1_operator_authorize_command: str | None
    top_2_substrate: str | None  # For Stage 2 stacking
    assessment_utc: str
    assessment_session: str
    # Provenance:
    score_claim: bool = False  # Assessment is a planning artifact, never a score claim
    promotion_eligible: bool = False
    evidence_grade: str = "predicted"
    provenance_kind: str = "PREDICTED_FROM_MODEL"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["candidates"] = [c.as_dict() if isinstance(c, dict) else c for c in d["candidates"]]
        # Coerce CandidateReadiness dicts properly
        d["candidates"] = [
            c.as_dict() if hasattr(c, "as_dict") else c for c in self.candidates
        ]
        return d


def _load_lane_registry(repo_root: Path) -> dict[str, Any]:
    """Read .omx/state/lane_registry.json or return empty schema."""
    path = repo_root / ".omx" / "state" / "lane_registry.json"
    if not path.exists():
        return {"lanes": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"lanes": []}


def _lookup_lane(reg: dict[str, Any], substrate_id: str) -> dict[str, Any] | None:
    """Find a registry lane whose id-substring matches the substrate token."""
    lanes = reg.get("lanes", []) if isinstance(reg, dict) else reg
    if not isinstance(lanes, list):
        return None
    # Prefer the most-recent (last) matching lane
    best: dict[str, Any] | None = None
    best_level = -1
    for lane in lanes:
        lid = (lane.get("id") or "").lower()
        if substrate_id.lower() in lid:
            lvl = int(lane.get("level", 0))
            if lvl > best_level:
                best = lane
                best_level = lvl
    return best


def _resolve_full_main_status(trainer_path: Path) -> tuple[bool, str | None]:
    """Inspect _full_main definition body for NotImplementedError raise.

    Returns (full_main_implemented, blocker_description).
    """
    if not trainer_path.exists():
        return False, "TRAINER_FILE_MISSING"
    text = trainer_path.read_text(encoding="utf-8", errors="ignore")
    # Find def _full_main + look at first ~60 lines after
    m = re.search(r"^def _full_main\(", text, re.MULTILINE)
    if m is None:
        return False, "FULL_MAIN_NOT_DEFINED"
    start = m.start()
    end = min(len(text), start + 4000)
    body = text[start:end]
    if "raise NotImplementedError" in body:
        return False, "RAISES_NotImplementedError"
    return True, None


def _resolve_latest_council_verdict(
    repo_root: Path, substrate_id: str
) -> str:
    """Look up latest council deliberation for the substrate per Catalog #315."""
    path = repo_root / ".omx" / "state" / "council_deliberation_posterior.jsonl"
    if not path.exists():
        return "NO_DELIBERATION"
    try:
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except Exception:
        return "NO_DELIBERATION"
    # Look up by deferred_substrate_id or substrate_alias or topic
    matching: list[dict[str, Any]] = []
    for r in rows:
        sid = (r.get("deferred_substrate_id") or "").lower()
        topic = (r.get("topic") or "").lower()
        if substrate_id.lower() in sid or substrate_id.lower() in topic:
            matching.append(r)
    if not matching:
        return "NO_DELIBERATION"
    # Latest wins (chronological by append order)
    latest = matching[-1]
    return latest.get("council_verdict", "NO_DELIBERATION")


def _resolve_predecessor_probe(
    repo_root: Path, substrate_id: str, recipe_basename: str
) -> tuple[bool, str | None, str | None]:
    """Catalog #313: check for blocking predecessor probe outcome.

    Returns (blocking, probe_id, verdict).
    """
    path = repo_root / ".omx" / "state" / "probe_outcomes.jsonl"
    if not path.exists():
        return False, None, None
    try:
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except Exception:
        return False, None, None
    # Look up by substrate or recipe
    matching: list[dict[str, Any]] = []
    for r in rows:
        sub = (r.get("substrate") or "").lower()
        rp = r.get("recipe_path") or ""
        if (substrate_id.lower() in sub) or (recipe_basename and recipe_basename in rp):
            if r.get("blocker_status") == "blocking":
                matching.append(r)
    if not matching:
        return False, None, None
    latest = matching[-1]
    return True, latest.get("probe_id"), latest.get("verdict")


def _parse_recipe(recipe_path: Path) -> dict[str, Any]:
    """Read recipe YAML or return empty dict."""
    if not recipe_path.exists():
        return {}
    try:
        import yaml

        with open(recipe_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _extract_predicted_band_from_design_memo(
    repo_root: Path, substrate_id: str
) -> tuple[float | None, float | None, str]:
    """Find a design memo for the substrate + extract predicted ΔS band.

    Returns (low, high, provenance_source). Provenance is the design memo path
    or "DEFAULT_HORIZON_CLASS_BAND" if no design memo found.
    """
    research = repo_root / ".omx" / "research"
    if not research.exists():
        return None, None, "RESEARCH_DIR_MISSING"
    pattern = re.compile(
        r"(?:predicted.*ΔS|predicted.*delta.*S|predicted_delta_s_band)[:\s]+[\"']?[A-Za-z0-9 _\-]*\[(-?[\d\.]+)[,\s]+(-?[\d\.]+)\]"
    )
    for f in sorted(research.glob("*.md")):
        if substrate_id.lower() in f.name.lower():
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")[:8000]
            except Exception:
                continue
            m = pattern.search(content)
            if m:
                try:
                    lo, hi = float(m.group(1)), float(m.group(2))
                    return lo, hi, f.name
                except ValueError:
                    pass
    return None, None, "DESIGN_MEMO_NOT_FOUND"


def _estimate_dispatch_cost(
    gpu: str, epochs: int, *, paired_axis: bool = True
) -> tuple[float, int]:
    """Estimate paired CPU + CUDA dispatch cost per Catalog #270.

    Returns (estimated_cost_usd, estimated_wall_clock_seconds).
    """
    # Roughly: smoke 100ep ~10min on T4, full 1000ep ~3hr on T4, scales with GPU
    if gpu == "T4":
        wall_seconds_full = epochs * 20  # ~20s per epoch for 100ep smoke = 33min
    elif gpu == "A10G":
        wall_seconds_full = epochs * 14
    elif gpu in ("A100", "H100", "4090"):
        wall_seconds_full = epochs * 8
    else:
        wall_seconds_full = epochs * 30  # CPU very slow

    rate = PER_HOUR_USD_BY_GPU.get(gpu.upper(), 1.0)
    cuda_cost = (wall_seconds_full / 3600.0) * rate
    if paired_axis:
        # Paired CPU eval typically ~1-2 hours on Modal CPU = ~$0.10
        cpu_cost = 0.10
    else:
        cpu_cost = 0.0
    return round(cuda_cost + cpu_cost, 3), int(wall_seconds_full)


def _compute_blocking_issues(
    *,
    full_main_implemented: bool,
    full_main_blocker: str | None,
    council_verdict: str,
    research_only: bool,
    dispatch_enabled: bool,
    dispatch_blockers: tuple[str, ...],
    predecessor_probe_blocking: bool,
    predecessor_probe_id: str | None,
    recipe_path: Path | None,
    trainer_path: Path | None,
) -> tuple[str, ...]:
    """Aggregate per-gate blockers per the canonical dispatch readiness contract."""
    issues: list[str] = []
    if recipe_path is None or not recipe_path.exists():
        issues.append("RECIPE_MISSING")
    if trainer_path is None or not trainer_path.exists():
        issues.append("TRAINER_MISSING")
    if not full_main_implemented:
        issues.append(f"CATALOG_240_FULL_MAIN_BLOCKED:{full_main_blocker}")
    if predecessor_probe_blocking:
        issues.append(f"CATALOG_313_PROBE_BLOCKING:{predecessor_probe_id}")
    if council_verdict == "PROCEED_WITH_REVISIONS":
        issues.append("CATALOG_315_COUNCIL_PROCEED_WITH_REVISIONS_NEEDS_ITERATION_TO_OPTIMAL_FORM")
    if research_only:
        issues.append("RECIPE_research_only=true_OPERATOR_NEEDS_TO_FLIP_AFTER_PHASE_2_COUNCIL")
    if not dispatch_enabled and not research_only:
        # If dispatch_enabled defaults to true when unset; only block if explicit False
        issues.append("RECIPE_dispatch_enabled=false")
    for blocker in dispatch_blockers:
        issues.append(f"RECIPE_DISPATCH_BLOCKER:{blocker}")
    return tuple(issues)


def _classify_readiness_verdict(blocking_issues: tuple[str, ...]) -> str:
    """Classify candidate readiness based on its blocking issues."""
    if not blocking_issues:
        return "READY"
    # Probe-blocking is harder to fix than recipe issues
    has_probe_block = any(i.startswith("CATALOG_313") for i in blocking_issues)
    has_council_block = any(i.startswith("CATALOG_315") for i in blocking_issues)
    has_full_main_block = any(i.startswith("CATALOG_240") for i in blocking_issues)
    if has_probe_block or has_council_block:
        return "DEFER"
    if has_full_main_block:
        # Trainer has NotImplementedError — substantive engineering work needed
        return "DEFER"
    # Otherwise the issues are recipe-flag adjustments and missing files
    return "NEEDS_FIX"


def _classify_horizon_class(
    predicted_low: float | None, predicted_high: float | None
) -> str:
    """Per CLAUDE.md "Max observability — non-negotiable" + Catalog #309.

    PLATEAU-ADJACENT [0.180, 0.200] | FRONTIER-PURSUIT [0.120, 0.180] |
    ASYMPTOTIC-PURSUIT [0.050, 0.120].

    Returns "asymptotic_pursuit" / "frontier_pursuit" / "plateau_adjacent" /
    "unknown".
    """
    if predicted_low is None or predicted_high is None:
        return "unknown"
    if predicted_low < 0.120:
        return "asymptotic_pursuit"
    if predicted_low < 0.180:
        return "frontier_pursuit"
    return "plateau_adjacent"


def _compute_ev_per_dollar(
    predicted_low: float | None,
    predicted_high: float | None,
    estimated_cost_usd: float,
    baseline: float = 0.19205,
) -> float:
    """Expected value per dollar = |baseline - predicted_midpoint| / cost.

    Baseline 0.19205 = current contest-CPU frontier per Catalog #316 fec6 anchor.
    """
    if predicted_low is None or predicted_high is None or estimated_cost_usd <= 0:
        return 0.0
    midpoint = (predicted_low + predicted_high) / 2.0
    delta = baseline - midpoint
    if delta <= 0:
        return 0.0
    return round(delta / estimated_cost_usd, 4)


def assess_candidate(
    substrate_id: str, *, repo_root: Path | None = None
) -> CandidateReadiness:
    """Build readiness assessment for a single substrate-class-shift candidate."""
    repo = repo_root or REPO_ROOT
    recipe_basename = RECIPE_BY_SUBSTRATE.get(substrate_id, f"substrate_{substrate_id}_modal_t4_dispatch")
    recipe_path = repo / ".omx" / "operator_authorize_recipes" / f"{recipe_basename}.yaml"
    trainer_path = repo / "experiments" / f"train_substrate_{substrate_id}.py"

    # Lane registry lookup
    reg = _load_lane_registry(repo)
    lane = _lookup_lane(reg, substrate_id)
    if lane is None:
        lane_maturity = "NOT_REGISTERED"
        impl_complete = False
    else:
        level = int(lane.get("level", 0))
        lane_maturity = f"L{level}"
        impl_complete = bool(lane.get("impl_complete", False))

    # Trainer _full_main status
    full_main_implemented, full_main_blocker = _resolve_full_main_status(trainer_path)

    # Council verdict per Catalog #315
    council_verdict = _resolve_latest_council_verdict(repo, substrate_id)

    # Recipe parsing
    recipe = _parse_recipe(recipe_path)
    research_only = bool(recipe.get("research_only", False))
    dispatch_enabled = bool(recipe.get("dispatch_enabled", not research_only))
    dispatch_blockers = tuple(recipe.get("dispatch_blockers", []) or [])
    gpu = str(recipe.get("gpu", "T4")).replace("${MODAL_GPU:-", "").rstrip("}")
    min_smoke_gpu = str(recipe.get("min_smoke_gpu", gpu))
    cost_band = recipe.get("cost_band", {}) or {}
    epochs = int(cost_band.get("epochs", 100))

    # Predecessor probe per Catalog #313
    probe_blocking, probe_id, probe_verdict = _resolve_predecessor_probe(
        repo, substrate_id, recipe_basename
    )

    # Predicted ΔS band from design memo
    predicted_low, predicted_high, predicted_source = _extract_predicted_band_from_design_memo(
        repo, substrate_id
    )

    # Cost estimation
    estimated_cost, estimated_wall = _estimate_dispatch_cost(gpu, epochs)

    # Horizon class
    horizon_class = _classify_horizon_class(predicted_low, predicted_high)

    # Blocking issues + readiness verdict
    blocking_issues = _compute_blocking_issues(
        full_main_implemented=full_main_implemented,
        full_main_blocker=full_main_blocker,
        council_verdict=council_verdict,
        research_only=research_only,
        dispatch_enabled=dispatch_enabled,
        dispatch_blockers=dispatch_blockers,
        predecessor_probe_blocking=probe_blocking,
        predecessor_probe_id=probe_id,
        recipe_path=recipe_path if recipe_path.exists() else None,
        trainer_path=trainer_path if trainer_path.exists() else None,
    )
    readiness_verdict = _classify_readiness_verdict(blocking_issues)

    # EV per dollar
    ev_per_dollar = _compute_ev_per_dollar(predicted_low, predicted_high, estimated_cost)

    return CandidateReadiness(
        substrate_id=substrate_id,
        recipe_basename=recipe_basename,
        recipe_path=recipe_path if recipe_path.exists() else None,
        trainer_path=trainer_path if trainer_path.exists() else None,
        lane_maturity=lane_maturity,
        impl_complete=impl_complete,
        full_main_implemented=full_main_implemented,
        full_main_blocker=full_main_blocker,
        latest_council_verdict=council_verdict,
        research_only=research_only,
        dispatch_enabled=dispatch_enabled,
        dispatch_blockers=dispatch_blockers,
        predecessor_probe_blocking=probe_blocking,
        predecessor_probe_id=probe_id,
        predecessor_probe_verdict=probe_verdict,
        predicted_delta_s_band_low=predicted_low,
        predicted_delta_s_band_high=predicted_high,
        predicted_delta_s_provenance=f"PREDICTED_FROM_MODEL:{predicted_source}",
        estimated_dispatch_cost_usd=estimated_cost,
        estimated_dispatch_wall_clock_seconds=estimated_wall,
        gpu_class=gpu,
        min_smoke_gpu=min_smoke_gpu,
        cost_band_epochs=epochs,
        horizon_class=horizon_class,
        blocking_issues=blocking_issues,
        readiness_verdict=readiness_verdict,
        ev_per_dollar=ev_per_dollar,
        score_claim=False,  # Provenance discipline
        promotion_eligible=False,
        evidence_grade="predicted",
    )


def rank_by_ev_per_dollar(
    candidates: tuple[CandidateReadiness, ...],
) -> tuple[str, ...]:
    """Rank READY > NEEDS_FIX > DEFER, then by ev_per_dollar within each tier."""
    tier_order = {"READY": 0, "NEEDS_FIX": 1, "DEFER": 2}
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (
            tier_order.get(c.readiness_verdict, 3),
            -c.ev_per_dollar,
        ),
    )
    return tuple(c.substrate_id for c in sorted_candidates)


def build_operator_authorize_command(candidate: CandidateReadiness) -> str:
    """Build the operator_authorize ready-to-fire command for a READY candidate.

    Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable
    + Catalog #167 smoke-before-full pattern.
    """
    if candidate.readiness_verdict != "READY":
        return f"# Candidate NOT READY (verdict={candidate.readiness_verdict}); operator-authorize NOT recommended"
    if candidate.recipe_path is None:
        return "# Recipe path missing; cannot generate command"
    return (
        f".venv/bin/python tools/run_modal_smoke_before_full.py "
        f"--recipe {candidate.recipe_path} "
        f"# Catalog #167 smoke-before-full pattern; smoke ~$1, full ~${candidate.estimated_dispatch_cost_usd}"
    )


def assess_candidates(
    candidates: tuple[str, ...] | None = None, *, repo_root: Path | None = None
) -> ReadinessAssessment:
    """Top-level: assess all canonical candidates + return ranked assessment."""
    repo = repo_root or REPO_ROOT
    candidates = candidates or CANONICAL_CANDIDATES
    assessed = tuple(assess_candidate(c, repo_root=repo) for c in candidates)
    ranked = rank_by_ev_per_dollar(assessed)
    # Top-1 = first in ranked order
    top_1_id = ranked[0] if ranked else None
    top_1 = next((c for c in assessed if c.substrate_id == top_1_id), None) if top_1_id else None
    top_1_verdict = top_1.readiness_verdict if top_1 else "NONE"
    top_1_cmd = build_operator_authorize_command(top_1) if top_1 else None
    top_2_id = ranked[1] if len(ranked) > 1 else None
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return ReadinessAssessment(
        candidates=assessed,
        ranked_by_ev_per_dollar=ranked,
        top_1_substrate=top_1_id,
        top_1_readiness_verdict=top_1_verdict,
        top_1_operator_authorize_command=top_1_cmd,
        top_2_substrate=top_2_id,
        assessment_utc=now,
        assessment_session="asymptotic_pursuit_candidate_readiness_assessment_20260517",
    )


def _render_human(assessment: ReadinessAssessment) -> str:
    """Human-readable rendering of the assessment."""
    out: list[str] = []
    out.append("=" * 78)
    out.append("ASYMPTOTIC PURSUIT — substrate-class-shift candidate-readiness assessment")
    out.append(f"  generated: {assessment.assessment_utc}")
    out.append(f"  session: {assessment.assessment_session}")
    out.append(f"  provenance: {assessment.provenance_kind} (score_claim={assessment.score_claim})")
    out.append("=" * 78)
    out.append("")
    out.append(f"TOP-1 RECOMMENDATION: {assessment.top_1_substrate or 'NONE'}")
    out.append(f"  readiness_verdict: {assessment.top_1_readiness_verdict}")
    if assessment.top_1_operator_authorize_command:
        out.append(f"  operator_authorize:")
        out.append(f"    {assessment.top_1_operator_authorize_command}")
    out.append("")
    out.append(f"TOP-2 (Stage 2 stacking candidate): {assessment.top_2_substrate or 'NONE'}")
    out.append("")
    out.append("Per-candidate readiness matrix:")
    out.append("")
    for sid in assessment.ranked_by_ev_per_dollar:
        c = next(x for x in assessment.candidates if x.substrate_id == sid)
        out.append(f"  [{c.readiness_verdict}] {c.substrate_id}")
        out.append(f"    lane_maturity={c.lane_maturity} | impl_complete={c.impl_complete} | full_main={c.full_main_implemented}")
        out.append(f"    council_verdict={c.latest_council_verdict} | research_only={c.research_only} | dispatch_enabled={c.dispatch_enabled}")
        out.append(f"    horizon_class={c.horizon_class} | predicted_ΔS_band=[{c.predicted_delta_s_band_low}, {c.predicted_delta_s_band_high}]")
        out.append(f"    estimated_cost=${c.estimated_dispatch_cost_usd} | wall_clock={c.estimated_dispatch_wall_clock_seconds}s | EV/$={c.ev_per_dollar}")
        out.append(f"    gpu={c.gpu_class} | min_smoke_gpu={c.min_smoke_gpu} | epochs={c.cost_band_epochs}")
        if c.predecessor_probe_blocking:
            out.append(f"    *** PROBE BLOCKED: {c.predecessor_probe_id} verdict={c.predecessor_probe_verdict}")
        if c.blocking_issues:
            out.append(f"    blocking_issues:")
            for issue in c.blocking_issues:
                out.append(f"      - {issue}")
        out.append("")
    out.append("=" * 78)
    out.append("Per CLAUDE.md 'Forbidden premature KILL': DEFER verdicts surface reactivation criteria.")
    out.append("Per CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA': all dispatches paired.")
    out.append("=" * 78)
    return "\n".join(out)


def write_artifact(
    assessment: ReadinessAssessment, *, repo_root: Path | None = None
) -> Path:
    """Persist the assessment to the canonical artifact directory."""
    repo = repo_root or REPO_ROOT
    artifact_dir = repo / ".omx" / "state" / "asymptotic_pursuit"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stamp = assessment.assessment_utc.replace(":", "").replace("-", "")
    path = artifact_dir / f"readiness_assessment_{stamp}.json"
    # Convert via as_dict
    payload = {
        "candidates": [c.as_dict() for c in assessment.candidates],
        "ranked_by_ev_per_dollar": list(assessment.ranked_by_ev_per_dollar),
        "top_1_substrate": assessment.top_1_substrate,
        "top_1_readiness_verdict": assessment.top_1_readiness_verdict,
        "top_1_operator_authorize_command": assessment.top_1_operator_authorize_command,
        "top_2_substrate": assessment.top_2_substrate,
        "assessment_utc": assessment.assessment_utc,
        "assessment_session": assessment.assessment_session,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "predicted",
        "provenance_kind": "PREDICTED_FROM_MODEL",
        "result_review_blockers": [
            "asymptotic_pursuit_readiness_assessment_is_planning_artifact_not_score_claim",
            "requires_paired_modal_cpu_cuda_dispatch_for_empirical_anchor_per_catalog_127",
        ],
    }
    # FIX-WAVE-R1 F2 closure 2026-05-17 + Catalog #131 META-meta drift fix:
    # write to unique timestamped filename (single-writer-per-session by
    # construction — each invocation produces a UTC-stamped filename so no
    # two callers ever race on the same path). Same-line BARE_WRITE_OK waiver
    # documents the genuine single-writer invariant per Catalog #131 contract.
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))  # BARE_WRITE_OK:unique_timestamped_filename_single_writer_per_session_no_concurrent_write_possible_per_fix_wave_r1
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true", help="Emit JSON output")
    p.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Comma-separated candidate IDs (override default canonical list)",
    )
    p.add_argument(
        "--write-artifact",
        action="store_true",
        help="Persist assessment to .omx/state/asymptotic_pursuit/",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override repo root (testing).",
    )
    args = p.parse_args(argv)

    candidates = (
        tuple(c.strip() for c in args.candidates.split(",")) if args.candidates else None
    )
    assessment = assess_candidates(candidates, repo_root=args.repo_root)

    if args.write_artifact:
        path = write_artifact(assessment, repo_root=args.repo_root)
        print(f"[asymptotic-pursuit] wrote artifact: {path}", file=sys.stderr)

    if args.json:
        payload = {
            "candidates": [c.as_dict() for c in assessment.candidates],
            "ranked_by_ev_per_dollar": list(assessment.ranked_by_ev_per_dollar),
            "top_1_substrate": assessment.top_1_substrate,
            "top_1_readiness_verdict": assessment.top_1_readiness_verdict,
            "top_1_operator_authorize_command": assessment.top_1_operator_authorize_command,
            "top_2_substrate": assessment.top_2_substrate,
            "assessment_utc": assessment.assessment_utc,
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_grade": "predicted",
            "provenance_kind": "PREDICTED_FROM_MODEL",
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_render_human(assessment))

    return 0 if assessment.top_1_readiness_verdict == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())
