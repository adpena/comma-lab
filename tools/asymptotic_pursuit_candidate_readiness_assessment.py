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
    "z6_v2_candidate_4c_scorer_logit",  # Atick scorer-logit side-info branch
    "time_traveler_l5_z7_lstm_predictive_coding",  # Z7 GRU recurrent predictor; pre-build gated on Z6 4c
    "time_traveler_l5_z7_mamba2",  # Z7 Mamba-2 recurrent predictor scaffold; Wave-N+2 pivot branch
    "atw_codec_v2",  # ATW V2 D4 cooperative-receiver (currently BLOCKED by probe)
    "atw_codec_v2_1_faiss_ivf_pq",  # ATW V2-1 Faiss-PQ channel; diagnostic-only after weak/bias probe
    "time_traveler_l5_autonomy",  # TT5L L5 foveation + LAPose
    "c6_e4_mdl_ibps",  # C6 MDL Information Bottleneck Per Pair
    "dp1_pr101_composition",  # DP1 frame-prior x PR101/FEC6 dual-stacking surface
    "lane_17_imp",  # Frankle LTH / IMP cycle-0 resurrection candidate
    "nscs01_nullspace_split_renderer",  # NSCS01 nullspace split (PR95-paradigm)
    "nscs03_end_to_end_balle_joint_codec",  # NSCS03 end-to-end Ballé joint codec
)

Z6_CANDIDATE4C_EXACT_OUTCOME_BLOCKER = (
    "z7_dispatch_requires_z6_wave2_candidate_4c_paired_exact_eval_outcome"
)
Z6_CANDIDATE4C_DISAMBIGUATOR_PATH = (
    ".omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.json"
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
    "z6_v2_candidate_4c_scorer_logit": "substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch",
    "time_traveler_l5_z7_lstm_predictive_coding": "substrate_time_traveler_l5_z7_lstm_predictive_coding_modal_t4_dispatch",
    "time_traveler_l5_z7_mamba2": "substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch",
    "atw_codec_v2": "substrate_atw_codec_v2_modal_a100_dispatch",
    "atw_codec_v2_1_faiss_ivf_pq": "substrate_atw_v2_1_modal_t4_smoke_dispatch",
    "time_traveler_l5_autonomy": "substrate_time_traveler_l5_autonomy_modal_a100_dispatch",
    "c6_e4_mdl_ibps": "substrate_c6_e4_mdl_ibps_modal_t4_dispatch",
    "dp1_pr101_composition": "substrate_pr101_with_dp1_prior_modal_cpu_smoke_dispatch",
    "lane_17_imp": "lane_17_imp_cycle0_vastai_4090_timing_smoke_dispatch",
    "nscs01_nullspace_split_renderer": "substrate_nscs01_nullspace_split_renderer_modal_t4_dispatch",
    "nscs03_end_to_end_balle_joint_codec": "substrate_nscs03_end_to_end_balle_joint_codec_modal_a100_dispatch",
}

TRAINER_BY_SUBSTRATE: dict[str, str] = {
    "z6_v2_candidate_4c_scorer_logit": "train_substrate_time_traveler_l5_z6.py",
    "time_traveler_l5_z7_lstm_predictive_coding": "train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py",
    "time_traveler_l5_z7_mamba2": "train_substrate_time_traveler_l5_z7_mamba2.py",
    "atw_codec_v2_1_faiss_ivf_pq": "train_substrate_atw_v2_1.py",
    "dp1_pr101_composition": "train_substrate_pr101_with_dp1_prior_regularizer.py",
    "lane_17_imp": "train_imp_cycle.py",
}

SUBSTRATE_ALIASES: dict[str, tuple[str, ...]] = {
    "time_traveler_l5_z7_lstm_predictive_coding": (
        "z7_lstm_predictive_coding",
        "time_traveler_l5_z7",
        "lane_c2_z7_mature_predictive_receiver_l5_campaign",
    ),
    "time_traveler_l5_z7_mamba2": (
        "z7_mamba2",
        "lane_top5_2_z7_mamba2_scaffold_design_20260518",
    ),
    "dp1_pr101_composition": (
        "pr101_with_dp1_prior_regularizer",
        "lane_dp1_plus_fec6_dual_stacking_build_20260517",
        "substrate_pr101_with_dp1_prior_modal_cpu_smoke_dispatch",
        "dp1_plus_fec6_composition_modal_paired_dispatch",
    ),
    "lane_17_imp": (
        "lane_17_imp_10cycle",
        "lane_j_imp_iterative_magnitude_pruning",
        "lane_per_substrate_symposium_lane_17_imp_20260517",
    ),
    "atw_codec_v2_1_faiss_ivf_pq": (
        "atw_v2_1",
        "v2-1",
        "faiss-ivf-pq",
        "substrate_atw_v2_1_modal_t4_smoke_dispatch",
        "lane_top5_3_atw_v2_1_faiss_ivf_pq_scaffold_design_20260518",
    ),
}


def _substrate_lookup_terms(substrate_id: str) -> tuple[str, ...]:
    """Return canonical + alias tokens for ledger/registry lookups."""

    return (substrate_id, *SUBSTRATE_ALIASES.get(substrate_id, ()))


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
    dispatch_blocker_supersessions: tuple[str, ...] = ()
    local_identity_disambiguator_probe_path: str | None = None
    local_identity_disambiguator_probe_verdict: str | None = None
    local_identity_disambiguator_runtime_output_changed: bool | None = None
    local_identity_disambiguator_blockers: tuple[str, ...] = ()
    local_identity_disambiguator_custody: dict[str, Any] = field(default_factory=dict)
    predicted_band_kind: str | None = None
    predicted_band_axis: str | None = None
    predicted_band_validation_status: str | None = None
    predicted_band_metadata_blockers: tuple[str, ...] = ()
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
        d["predicted_score_band"] = (
            [self.predicted_delta_s_band_low, self.predicted_delta_s_band_high]
            if self.predicted_band_kind == "predicted_score_band"
            else None
        )
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


def _lane_gate_status(lane: dict[str, Any], gate: str) -> bool:
    """Return the canonical lane-maturity gate status."""
    gates = lane.get("gates", {})
    if not isinstance(gates, dict):
        return False
    gate_payload = gates.get(gate, {})
    if not isinstance(gate_payload, dict):
        return False
    return bool(gate_payload.get("status"))


def _lookup_lane(
    reg: dict[str, Any], substrate_id: str, *, recipe_lane_id: str | None = None
) -> dict[str, Any] | None:
    """Find a registry lane by exact recipe lane_id, then substrate token."""
    lanes = reg.get("lanes", []) if isinstance(reg, dict) else reg
    if not isinstance(lanes, list):
        return None
    if recipe_lane_id:
        for lane in lanes:
            if lane.get("id") == recipe_lane_id:
                return lane
    # Prefer the most-recent (last) matching lane across canonical + alias ids.
    terms = tuple(term.lower() for term in _substrate_lookup_terms(substrate_id))
    best: dict[str, Any] | None = None
    best_level = -1
    for lane in lanes:
        lid = (lane.get("id") or "").lower()
        if any(term in lid for term in terms):
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
    terms = tuple(term.lower() for term in _substrate_lookup_terms(substrate_id))
    matching: list[dict[str, Any]] = []
    for r in rows:
        sid = (r.get("deferred_substrate_id") or "").lower()
        topic = (r.get("topic") or "").lower()
        aliases = r.get("substrate_aliases", [])
        alias_text = " ".join(str(alias).lower() for alias in aliases if alias)
        if any(term in sid or term in topic or term in alias_text for term in terms):
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
    def _row_matches_candidate(row: dict[str, Any]) -> bool:
        sub = str(row.get("substrate") or "").lower().strip()
        recipe_path = str(row.get("recipe_path") or "")
        recipe_stem = Path(recipe_path).stem if recipe_path else ""

        if recipe_basename and recipe_stem == recipe_basename:
            return True

        # Avoid prefix bleed between canonical siblings such as
        # atw_codec_v2 and atw_codec_v2_1_faiss_ivf_pq.
        other_candidates = tuple(
            sid.lower() for sid in CANONICAL_CANDIDATES if sid != substrate_id
        )
        if any(sub == other or sub.startswith(f"{other}_") for other in other_candidates):
            return False

        return any(sub == term or sub.startswith(f"{term}_") for term in terms)

    # Look up by exact substrate-family or recipe basename, not loose substring.
    terms = tuple(term.lower() for term in _substrate_lookup_terms(substrate_id))
    matching: list[dict[str, Any]] = []
    for r in rows:
        if _row_matches_candidate(r):
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


def _recipe_targets_contest_exact_eval(recipe: dict[str, Any]) -> bool:
    """Return True when a recipe explicitly targets contest exact eval."""

    modes = recipe.get("target_modes", [])
    if not isinstance(modes, list):
        return False
    return "contest_exact_eval" in {str(mode).strip() for mode in modes}


def _false_authority_probe_blockers(
    payload: dict[str, Any], *, prefix: str
) -> list[str]:
    blockers: list[str] = []
    for field, expected in (
        ("research_only", True),
        ("score_claim", False),
        ("promotion_eligible", False),
        ("rank_or_kill_eligible", False),
        ("ready_for_exact_eval_dispatch", False),
        ("ready_for_paid_dispatch", False),
        ("paradigm_claim_allowed", False),
    ):
        if payload.get(field) is not expected:
            blockers.append(f"{prefix}_{field}_not_{str(expected).lower()}")
    return blockers


def _resolve_z7_temporal_disambiguator_probe(
    probe_path: Path, payload: dict[str, Any]
) -> tuple[str | None, str | None, bool | None, tuple[str, ...], dict[str, Any]]:
    """Validate a Z7 temporal-vs-static disambiguator plan/result payload."""

    blockers = _false_authority_probe_blockers(
        payload, prefix="z7_temporal_disambiguator_probe"
    )
    verdict = payload.get("verdict")
    if not isinstance(verdict, str) or not verdict:
        blockers.append("z7_temporal_disambiguator_probe_verdict_missing")
        verdict = None
    elif verdict.startswith("blocked_"):
        blockers.append(f"z7_temporal_disambiguator_probe_blocked_verdict:{verdict}")

    payload_blockers = payload.get("blockers")
    if verdict == "pending_paired_exact_eval_json":
        required_no_score = {
            "no_paired_exact_eval_json",
            "no_contest_cuda_pair",
            "not_score_authority",
        }
        if not isinstance(payload_blockers, list) or not required_no_score.issubset(
            {str(blocker) for blocker in payload_blockers}
        ):
            blockers.append(
                "z7_temporal_disambiguator_pending_exact_eval_blockers_missing"
            )
        decision_rule = payload.get("decision_rule")
        custody = {
            "decision_rule": decision_rule if isinstance(decision_rule, dict) else {},
            "required_inputs": list(payload.get("required_inputs") or []),
            "required_future_artifacts": list(
                payload.get("required_future_artifacts") or []
            ),
        }
        if custody["decision_rule"].get("same_archive_bytes_required") is not True:
            blockers.append(
                "z7_temporal_disambiguator_same_archive_bytes_rule_missing"
            )
        return str(probe_path), verdict, None, tuple(blockers), custody

    comparability = payload.get("comparability")
    if not isinstance(comparability, dict):
        blockers.append("z7_temporal_disambiguator_comparability_missing")
        comparability = {}
    else:
        for field in ("same_score_axis", "same_n_samples", "same_archive_bytes"):
            if comparability.get(field) is not True:
                blockers.append(f"z7_temporal_disambiguator_{field}_not_true")

    source_evals = payload.get("source_evals")
    custody: dict[str, Any] = {
        "comparability": dict(comparability),
        "deltas": dict(payload.get("deltas") or {}),
    }
    if not isinstance(source_evals, list) or len(source_evals) != 2:
        blockers.append("z7_temporal_disambiguator_source_evals_missing")
    else:
        custody["source_eval_paths"] = [row.get("path") for row in source_evals]
        custody["source_eval_json_sha256"] = [
            row.get("json_sha256") for row in source_evals
        ]
        custody["source_eval_archive_sha256"] = [
            row.get("archive_sha256") for row in source_evals
        ]
        custody["score_axis"] = source_evals[0].get("score_axis")
        custody["n_samples"] = source_evals[0].get("n_samples")

    if isinstance(payload_blockers, list) and payload_blockers:
        blockers.extend(
            f"z7_temporal_disambiguator_payload_blocker:{blocker}"
            for blocker in payload_blockers
        )

    return str(probe_path), verdict, None, tuple(blockers), custody


def _resolve_identity_disambiguator_probe(
    repo_root: Path, recipe: dict[str, Any]
) -> tuple[str | None, str | None, bool | None, tuple[str, ...], dict[str, Any]]:
    """Validate an optional recipe-wired local disambiguator probe.

    A local byte/output-consumption or exact-eval comparison probe is not score
    authority. It is a dispatch-readiness guard: if the recipe names one,
    missing output, authoritative-looking flags, or an invalid comparison must
    block paid launch before we burn the next run.
    """

    raw_path = recipe.get("identity_disambiguator_probe")
    if raw_path is None:
        return None, None, None, (), {}
    probe_path = repo_root / str(raw_path)
    probe_path = probe_path.resolve()
    try:
        probe_path.relative_to(repo_root.resolve())
    except ValueError:
        return str(probe_path), None, None, (
            "identity_disambiguator_probe_path_outside_repo",
        ), {}
    if not probe_path.exists():
        return str(probe_path), None, None, (
            "identity_disambiguator_probe_missing",
        ), {}
    try:
        payload = json.loads(probe_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return str(probe_path), None, None, (
            "identity_disambiguator_probe_json_unreadable",
        ), {}
    if not isinstance(payload, dict):
        return str(probe_path), None, None, (
            "identity_disambiguator_probe_not_json_object",
        ), {}

    blockers: list[str] = []
    schema = payload.get("schema")
    if schema == "z7_temporal_coherence_vs_static_capacity_disambiguator_v1":
        return _resolve_z7_temporal_disambiguator_probe(probe_path, payload)
    if schema != "z6_predictive_coding_vs_identity_disambiguator_v1":
        blockers.append("identity_disambiguator_probe_schema_unrecognized")
    verdict = payload.get("verdict")
    if not isinstance(verdict, str) or not verdict:
        blockers.append("identity_disambiguator_probe_verdict_missing")
        verdict = None
    elif verdict.startswith("blocked_"):
        blockers.append(f"identity_disambiguator_probe_blocked_verdict:{verdict}")

    blockers.extend(
        _false_authority_probe_blockers(
            payload, prefix="identity_disambiguator_probe"
        )
    )

    result_review = payload.get("result_review")
    if not isinstance(result_review, dict):
        blockers.append("identity_disambiguator_probe_result_review_missing")
        runtime_changed = None
    else:
        runtime_changed = result_review.get(
            "identity_predictor_switch_changes_inflate_output"
        )
    requires_runtime_change = bool(
        recipe.get("identity_disambiguator_probe_requires_runtime_output_changed", True)
    )
    if requires_runtime_change and runtime_changed is not True:
        blockers.append("identity_disambiguator_probe_runtime_output_not_changed")

    comparison = payload.get("inflate_output_comparison")
    custody: dict[str, Any] = {}
    if requires_runtime_change and not isinstance(comparison, dict):
        blockers.append("identity_disambiguator_probe_inflate_output_comparison_missing")
    elif isinstance(comparison, dict):
        custody["output_root"] = comparison.get("output_root")
        custody["total_byte_differences"] = comparison.get("total_byte_differences")
        full_tree = comparison.get("full_output_tree")
        if isinstance(full_tree, dict):
            custody["full_output_aggregate_sha256"] = full_tree.get(
                "aggregate_sha256"
            )
            custody["full_output_total_bytes"] = full_tree.get("total_bytes")
        identity_tree = comparison.get("identity_output_tree")
        if isinstance(identity_tree, dict):
            custody["identity_output_aggregate_sha256"] = identity_tree.get(
                "aggregate_sha256"
            )
            custody["identity_output_total_bytes"] = identity_tree.get("total_bytes")
        runtime_custody = comparison.get("runtime_custody")
        if not isinstance(runtime_custody, dict):
            runtime_custody = payload.get("runtime_custody")
        if isinstance(runtime_custody, dict):
            custody["runtime_custody_aggregate_sha256"] = runtime_custody.get(
                "aggregate_sha256"
            )
            custody["runtime_custody_file_count"] = runtime_custody.get(
                "file_count"
            )
            custody["runtime_custody_total_bytes"] = runtime_custody.get(
                "total_bytes"
            )
        elif requires_runtime_change:
            blockers.append("identity_disambiguator_probe_runtime_custody_missing")
        if comparison.get("score_claim") is not False:
            blockers.append(
                "identity_disambiguator_probe_inflate_output_score_claim_not_false"
            )
        if comparison.get("runtime_output_changed") is not True and requires_runtime_change:
            blockers.append(
                "identity_disambiguator_probe_inflate_output_runtime_output_not_changed"
            )
        if comparison.get("evidence_axis") != "[local-inflate-output advisory]":
            blockers.append(
                "identity_disambiguator_probe_inflate_output_axis_missing"
            )
        if (
            requires_runtime_change
            and not custody.get("runtime_custody_aggregate_sha256")
        ):
            blockers.append(
                "identity_disambiguator_probe_runtime_custody_sha256_missing"
            )
        if requires_runtime_change and not custody.get(
            "full_output_aggregate_sha256"
        ):
            blockers.append(
                "identity_disambiguator_probe_full_output_aggregate_sha256_missing"
            )
        if requires_runtime_change and not custody.get(
            "identity_output_aggregate_sha256"
        ):
            blockers.append(
                "identity_disambiguator_probe_identity_output_aggregate_sha256_missing"
            )

    payload_blockers = payload.get("blockers")
    if verdict == "pending_paired_exact_eval_json":
        required_no_score = {
            "no_paired_exact_eval_json",
            "no_contest_cpu_cuda_pair",
            "not_score_authority",
        }
        if not isinstance(payload_blockers, list) or not required_no_score.issubset(
            {str(blocker) for blocker in payload_blockers}
        ):
            blockers.append(
                "identity_disambiguator_probe_pending_exact_eval_blockers_missing"
            )

    return str(probe_path), verdict, (
        runtime_changed if isinstance(runtime_changed, bool) else None
    ), tuple(blockers), custody


def _z6_candidate4c_paired_exact_outcome_landed(repo_root: Path) -> bool:
    """Return True when the Z6 4c full-vs-identity exact-eval pair is closed.

    Z7 depends on this outcome only as a sequencing gate. Once both
    `[contest-CUDA]` and `[contest-CPU]` exact-eval axes have landed, the
    generic "requires Z6 4c outcome" blocker should disappear so the queue shows
    the real remaining Z7 work (trainer/package/council), not a solved
    predecessor dependency.
    """

    path = repo_root / Z6_CANDIDATE4C_DISAMBIGUATOR_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    exact = payload.get("exact_eval")
    if not isinstance(exact, dict):
        return False
    if exact.get("status") != "contest_cuda_and_cpu_recovered":
        return False
    if payload.get("score_claim") is not False:
        return False
    if payload.get("promotion_eligible") is not False:
        return False
    for axis in ("contest_cuda", "contest_cpu"):
        axis_payload = exact.get(axis)
        if not isinstance(axis_payload, dict):
            return False
        for mode in ("full", "identity"):
            row = axis_payload.get(mode)
            if not isinstance(row, dict):
                return False
            if row.get("evidence_grade") not in {"contest-CUDA", "contest-CPU"}:
                return False
            if not row.get("result_path"):
                return False
            if row.get("n_samples") != 600:
                return False
            try:
                float(row["score_recomputed_from_components"])
                float(row["avg_segnet_dist"])
                float(row["avg_posenet_dist"])
            except (KeyError, TypeError, ValueError):
                return False
    return True


def _effective_dispatch_blockers(
    repo_root: Path,
    substrate_id: str,
    dispatch_blockers: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Apply evidence-backed supersessions to recipe-declared blockers."""

    superseded: list[str] = []
    effective: list[str] = []
    z6_outcome_landed = (
        substrate_id == "time_traveler_l5_z7_lstm_predictive_coding"
        and _z6_candidate4c_paired_exact_outcome_landed(repo_root)
    )
    for blocker in dispatch_blockers:
        if blocker == Z6_CANDIDATE4C_EXACT_OUTCOME_BLOCKER and z6_outcome_landed:
            superseded.append(blocker)
            continue
        effective.append(blocker)
    return tuple(effective), tuple(superseded)


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
    terms = tuple(term.lower() for term in _substrate_lookup_terms(substrate_id))
    for f in sorted(research.glob("*.md")):
        if any(term in f.name.lower() for term in terms):
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


def _extract_predicted_band_from_recipe(
    recipe: dict[str, Any], recipe_basename: str
) -> tuple[float | None, float | None, str]:
    """Fallback recipe-band extraction with Catalog #324 false-authority guard.

    A recipe-level ``predicted_band`` is a planning prior, not score authority.
    If Catalog #324 has already marked the band as a falsified disabled recipe
    band, suppress it from EV/$ ranking so a stale random-init prediction cannot
    resurface as dispatch priority.
    """
    band = recipe.get("predicted_band")
    if not isinstance(band, (list, tuple)) or len(band) != 2:
        return None, None, "RECIPE_PREDICTED_BAND_MISSING"
    status = str(recipe.get("predicted_band_validation_status", "unvalidated"))
    criteria = str(recipe.get("predicted_band_reactivation_criteria", "")).lower()
    dispatch_enabled = bool(recipe.get("dispatch_enabled", True))
    if (
        status == "pending_post_training"
        and not dispatch_enabled
        and "falsified" in criteria
    ):
        return (
            None,
            None,
            f"{recipe_basename}.yaml:falsified_recipe_band_suppressed_by_catalog_324",
        )
    try:
        lo, hi = float(band[0]), float(band[1])
    except (TypeError, ValueError):
        return None, None, f"{recipe_basename}.yaml:malformed_predicted_band"
    return lo, hi, f"{recipe_basename}.yaml:predicted_band:{status}"


def _prediction_band_recipe_metadata(
    recipe: dict[str, Any],
    *,
    predicted_low: float | None,
    predicted_high: float | None,
) -> tuple[str | None, str | None, str | None, tuple[str, ...]]:
    """Return explicit recipe metadata for a numeric predicted score band.

    Historical recipes used ``predicted_band`` for predicted score targets while
    some downstream field names say ``delta``. That is survivable only when the
    recipe labels the band kind, evidence axis, and validation status. Missing
    labels are planning-surface blockers: keep the candidate visible, but do not
    let an unlabeled band contribute EV/$ rank pressure.
    """

    if predicted_low is None or predicted_high is None:
        return None, None, None, ()

    kind_raw = recipe.get("predicted_band_kind")
    axis_raw = recipe.get("predicted_band_axis")
    status_raw = recipe.get("predicted_band_validation_status")

    kind = str(kind_raw).strip() if kind_raw not in (None, "") else None
    axis = str(axis_raw).strip() if axis_raw not in (None, "") else None
    status = str(status_raw).strip() if status_raw not in (None, "") else None

    blockers: list[str] = []
    if kind is None:
        blockers.append("predicted_band_kind_missing")
    elif kind not in {"predicted_score_band"}:
        blockers.append(f"predicted_band_kind_unsupported:{kind}")

    if axis is None:
        blockers.append("predicted_band_axis_missing")
    elif axis not in {"contest-CUDA", "contest-CPU", "macOS-CPU advisory"}:
        blockers.append(f"predicted_band_axis_unrecognized:{axis}")

    if status is None:
        blockers.append("predicted_band_validation_status_missing")
    elif status in {"unvalidated", "unknown"}:
        blockers.append(f"predicted_band_validation_status_{status}")

    return kind, axis, status, tuple(blockers)


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
    lane_registered: bool = True,
    contest_exact_eval_targeted: bool = True,
    recipe_horizon_class: str | None = None,
    computed_horizon_class: str | None = None,
) -> tuple[str, ...]:
    """Aggregate per-gate blockers per the canonical dispatch readiness contract."""
    issues: list[str] = []
    if not lane_registered:
        issues.append("LANE_REGISTRY_NOT_REGISTERED")
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
    if dispatch_enabled and not research_only and not contest_exact_eval_targeted:
        issues.append("RECIPE_target_modes_missing_contest_exact_eval")
    if (
        dispatch_enabled
        and not research_only
        and recipe_horizon_class
        and computed_horizon_class
        and computed_horizon_class != "unknown"
        and str(recipe_horizon_class).strip() != computed_horizon_class
    ):
        issues.append(
            "RECIPE_horizon_class_mismatch:"
            f"{str(recipe_horizon_class).strip()}!={computed_horizon_class}"
        )
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
    if (
        "RECIPE_dispatch_enabled=false" in blocking_issues
        and any(
            issue
            == (
                "RECIPE_DISPATCH_BLOCKER:"
                "candidate4c_modal_training_recipe_is_diagnostic_only_exact_cuda_handoff_required"
            )
            for issue in blocking_issues
        )
    ):
        # Candidate 4c's Modal training surface is deliberately disabled after
        # the diagnostic-only split. Treat it like a measured handoff gate, not
        # a recipe typo that an operator should flip into a contest-CUDA launch.
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


def _recipe_session_budget_floor_usd(recipe_path: Path | None) -> float:
    """Return the recipe-declared paid session budget floor, if available."""

    if recipe_path is None or not recipe_path.exists():
        return 0.0
    recipe = _parse_recipe(recipe_path)
    cost_band = recipe.get("cost_band", {}) or {}
    if not isinstance(cost_band, dict):
        return 0.0
    values: list[float] = []
    for field in ("predicted_cost_usd", "hand_calibrated_fallback_p50_usd"):
        try:
            value = float(cost_band.get(field, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if value > 0.0:
            values.append(value)
    return max(values) if values else 0.0


def assess_candidate(
    substrate_id: str, *, repo_root: Path | None = None
) -> CandidateReadiness:
    """Build readiness assessment for a single substrate-class-shift candidate."""
    repo = repo_root or REPO_ROOT
    recipe_basename = RECIPE_BY_SUBSTRATE.get(substrate_id, f"substrate_{substrate_id}_modal_t4_dispatch")
    recipe_path = repo / ".omx" / "operator_authorize_recipes" / f"{recipe_basename}.yaml"
    trainer_filename = TRAINER_BY_SUBSTRATE.get(
        substrate_id, f"train_substrate_{substrate_id}.py"
    )
    trainer_path = repo / "experiments" / trainer_filename
    recipe = _parse_recipe(recipe_path)
    recipe_lane_id = recipe.get("lane_id")
    recipe_lane_id = str(recipe_lane_id) if recipe_lane_id else None

    # Lane registry lookup
    reg = _load_lane_registry(repo)
    lane = _lookup_lane(reg, substrate_id, recipe_lane_id=recipe_lane_id)
    if lane is None:
        lane_maturity = "NOT_REGISTERED"
        impl_complete = False
    else:
        level = int(lane.get("level", 0))
        lane_maturity = f"L{level}"
        impl_complete = _lane_gate_status(lane, "impl_complete")

    # Trainer _full_main status
    full_main_implemented, full_main_blocker = _resolve_full_main_status(trainer_path)
    if (
        not full_main_implemented
        and trainer_path.exists()
        and recipe.get("trainer_contract") == "legacy_remote_lane_script"
    ):
        full_main_implemented = True
        full_main_blocker = None

    # Council verdict per Catalog #315
    council_verdict = _resolve_latest_council_verdict(repo, substrate_id)

    research_only = bool(recipe.get("research_only", False))
    dispatch_enabled = bool(recipe.get("dispatch_enabled", not research_only))
    dispatch_blockers = tuple(recipe.get("dispatch_blockers", []) or [])
    dispatch_blockers, dispatch_blocker_supersessions = _effective_dispatch_blockers(
        repo, substrate_id, dispatch_blockers
    )
    (
        local_probe_path,
        local_probe_verdict,
        local_probe_runtime_changed,
        local_probe_blockers,
        local_probe_custody,
    ) = _resolve_identity_disambiguator_probe(repo, recipe)
    contest_exact_eval_targeted = _recipe_targets_contest_exact_eval(recipe)
    gpu = str(recipe.get("gpu", "T4")).replace("${MODAL_GPU:-", "").rstrip("}")
    min_smoke_gpu = str(recipe.get("min_smoke_gpu", gpu))
    cost_band = recipe.get("cost_band", {}) or {}
    epochs = int(cost_band.get("epochs", 100))

    # Predecessor probe per Catalog #313
    probe_blocking, probe_id, probe_verdict = _resolve_predecessor_probe(
        repo, substrate_id, recipe_basename
    )

    # Prefer explicit recipe metadata over loose design-memo regex extraction.
    # Several memos carry both score bands and delta-S bands; recipe
    # `predicted_band_kind` is the axis label that prevents false authority.
    recipe_has_predicted_band = (
        isinstance(recipe.get("predicted_band"), (list, tuple))
        and len(recipe.get("predicted_band", [])) == 2
    )
    if recipe_has_predicted_band:
        predicted_low, predicted_high, predicted_source = (
            _extract_predicted_band_from_recipe(recipe, recipe_basename)
        )
    elif recipe.get("predicted_score_target", "__missing__") is None:
        predicted_low, predicted_high, predicted_source = (
            None,
            None,
            f"{recipe_basename}.yaml:predicted_score_target_null",
        )
    else:
        predicted_low, predicted_high, predicted_source = _extract_predicted_band_from_design_memo(
            repo, substrate_id
        )
        if predicted_low is None or predicted_high is None:
            predicted_low, predicted_high, predicted_source = (
                _extract_predicted_band_from_recipe(recipe, recipe_basename)
            )
    (
        predicted_band_kind,
        predicted_band_axis,
        predicted_band_validation_status,
        predicted_band_metadata_blockers,
    ) = _prediction_band_recipe_metadata(
        recipe,
        predicted_low=predicted_low,
        predicted_high=predicted_high,
    )

    # Cost estimation
    estimated_cost, estimated_wall = _estimate_dispatch_cost(gpu, epochs)

    # Horizon class
    predicted_source_is_recipe = predicted_source.startswith(f"{recipe_basename}.yaml:")
    computed_horizon_class = (
        _classify_horizon_class(predicted_low, predicted_high)
        if predicted_band_kind == "predicted_score_band" or predicted_source_is_recipe
        else "unknown"
    )
    recipe_horizon_class = str(recipe.get("horizon_class") or "").strip() or None
    horizon_class = (
        computed_horizon_class
        if computed_horizon_class != "unknown"
        else recipe_horizon_class or "unknown"
    )

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
        lane_registered=lane is not None,
        contest_exact_eval_targeted=contest_exact_eval_targeted,
        recipe_horizon_class=recipe_horizon_class,
        computed_horizon_class=computed_horizon_class,
    )
    if local_probe_blockers:
        blocking_issues = (
            *blocking_issues,
            *(
                f"LOCAL_IDENTITY_DISAMBIGUATOR_BLOCKER:{blocker}"
                for blocker in local_probe_blockers
            ),
        )
    if predicted_band_metadata_blockers:
        blocking_issues = (
            *blocking_issues,
            *(
                f"PREDICTED_BAND_METADATA_BLOCKER:{blocker}"
                for blocker in predicted_band_metadata_blockers
            ),
        )
    readiness_verdict = _classify_readiness_verdict(blocking_issues)

    # EV per dollar
    ev_per_dollar = (
        0.0
        if predicted_band_metadata_blockers
        else _compute_ev_per_dollar(predicted_low, predicted_high, estimated_cost)
    )

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
        dispatch_blocker_supersessions=dispatch_blocker_supersessions,
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
        local_identity_disambiguator_probe_path=local_probe_path,
        local_identity_disambiguator_probe_verdict=local_probe_verdict,
        local_identity_disambiguator_runtime_output_changed=(
            local_probe_runtime_changed
        ),
        local_identity_disambiguator_blockers=local_probe_blockers,
        local_identity_disambiguator_custody=local_probe_custody,
        predicted_band_kind=predicted_band_kind,
        predicted_band_axis=predicted_band_axis,
        predicted_band_validation_status=predicted_band_validation_status,
        predicted_band_metadata_blockers=predicted_band_metadata_blockers,
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
    recipe_name = candidate.recipe_basename or candidate.recipe_path.stem
    queue_budget_usd = round(1.0 + candidate.estimated_dispatch_cost_usd, 3)
    recipe_budget_usd = _recipe_session_budget_floor_usd(candidate.recipe_path)
    session_budget_usd = max(queue_budget_usd, recipe_budget_usd)
    return (
        "OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 "
        f"OPERATOR_AUTHORIZE_SESSION_BUDGET_USD={session_budget_usd:.3f} "
        ".venv/bin/python tools/run_modal_smoke_before_full.py "
        f"--recipe {recipe_name} "
        f"--operator-handle codex:{candidate.substrate_id} "
        "# Catalog #167 smoke-before-full pattern; "
        f"budget_floor ~${session_budget_usd:.3f} "
        f"(queue_estimate=${queue_budget_usd:.3f}, recipe_floor=${recipe_budget_usd:.3f})"
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
        out.append(f"    predicted_band_kind={c.predicted_band_kind} | axis={c.predicted_band_axis} | validation_status={c.predicted_band_validation_status}")
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
