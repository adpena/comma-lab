# SPDX-License-Identifier: MIT
"""Typed, advisory DDM campaign SENSE/DECIDE state.

This module is deliberately pure and launch-free.  It folds the current
MS4D/MS7/RD1/J8E evidence into one state object and reserves a strict JSONL
intake for J8F realized verdicts.  Missing telemetry is a typed blocker, never
an invitation to substitute a proxy.

Equation / control authorities:

* ``ddm_366_dimension_completeness_contract_20260724.md`` owns the standing
  class-E telemetry rows.
* ``ddm_j8e_688_compile_receipt_20260724.json`` owns the event-engine contract.
* FEED-603 owns the plateau-type -> F1..F7 contingency map.
* measured RD1 endpoints own the aggregate scalarization controls; they do not
  create per-bucket byte prices.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist, fmean, stdev
from typing import Any

from tac.ddm_campaign_evidence_join import (
    METRIC_ID as EVIDENCE_METRIC_ID,
)
from tac.ddm_campaign_evidence_join import (
    SCHEMA as EVIDENCE_JOIN_SCHEMA,
)
from tac.ddm_campaign_evidence_join import (
    bucket_key as evidence_bucket_key,
)
from tac.ddm_campaign_evidence_join import (
    validate_campaign_evidence_join,
)
from tac.ddm_costate_law import RATE_BREAK_EVEN_SCORE_PER_BYTE
from tac.optimization.ddm_event_continuation import EVENT_MARK_SCHEMA

SCHEMA = "ddm_campaign_costate_state.v1"
VERDICT_SCHEMA = EVENT_MARK_SCHEMA
SENSE_ROW_SCHEMA = "ddm_campaign_sense_row.v1"
METRIC_ROW_SCHEMA = "ddm_campaign_metric_sense_row.v1"
DECISION_SCHEMA = "ddm_campaign_plateau_decision.v1"
DYNAMIC_POLICY_SCHEMA = "ddm_campaign_dynamic_policy.v1"
BLOCKER_SCHEMA = "ddm_campaign_blocker.v1"
ENHANCEMENT_SCHEMA = "ddm_costate_organ_enhancement.v1"
ENHANCEMENT_BACKTEST_SCHEMA = "ddm_costate_organ_enhancement_backtest.v1"
MATURITY = "_dev"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
G2F_SOURCE_SHA256 = "92d860ab35bba158e7fd817edf632d3e3e7fc90b05669402d537c26a6e09a88e"

# CO5 re-adjudication (2026-07-28): each held enhancement carries a NAMED producer gate so
# "off" is a tracked duty-to-measure queue, never a silent collective PREMISE_FALSIFIED. The
# gates are the concrete telemetry a CT1-v2 (or a fired-duty ledger) would emit to flip the
# backtest; the enhancement designs remain valid (RE_PREMISE), none are RETIRED.
#
# co7 re-check (2026-07-28, round 7): CT1's landed state is UNCHANGED (no CT1-v2 receipt
# exists) => all four gates KEEP RE_PREMISE; none fire. What DID change: tb1 sealed the T3
# long-burn full-confirm cadence (gate_every 10 -> >=2 n600 d_seg+bytes+wall-clock endpoints
# per burn window) and eg1 rehearsed the terminal pose finisher (the S-composition pose leg),
# so the compression-progress gate's producer is now PARTIALLY BUILT and awaiting the burn —
# recorded per gate in `producer_state` (a queue-state annotation, not a satisfaction claim).
_CO5_ENHANCEMENT_GATES: dict[str, dict[str, Any]] = {
    "pontryagin_bellman_transition_residual": {
        "named_gate": "CT1V2_EMITS_ORDERED_ADJACENT_CAMPAIGN_COSTATES_AND_TRANSITION_JACOBIANS",
        "producer": (
            "a CT1-v2 telemetry emitter surfacing ordered adjacent campaign costates + realized "
            "transition Jacobians (CT1 surfaces one exact endpoint + cadence, no costate/Jacobian)"
        ),
        "producer_state": "HYPOTHETICAL_EMITTER_NOT_BUILT",
        "unblocks": (),
    },
    "m34_per_state_dual_consistency": {
        "named_gate": "CT1V2_EMITS_PER_STATE_M34_AND_ORGAN_DUAL_WITH_MEASURED_UNCERTAINTY",
        "producer": (
            "a CT1-v2 emitter of per-state M34 + same-state organ dual + measured uncertainty band "
            "(CT1 emits neither dual)"
        ),
        "producer_state": "HYPOTHETICAL_EMITTER_NOT_BUILT",
        "unblocks": (),
    },
    "compression_progress_per_effort": {
        "named_gate": "CT1V2_SURFACES_TWO_PLUS_EXACT_N600_S_ENDPOINTS_WITH_WALL_CLOCK_AND_BYTE_IDENTITY",
        "producer": (
            "a CT1-v2 encoding >=2 exact n600 S-endpoints as full rows (CT1 ran 3 verdict steps "
            "[0,1,50] but surfaces only step 50 as a full S-row; the cumulative trace is explicitly "
            "ADVISORY_BATCH_LOCAL, not n600)"
        ),
        "producer_state": (
            "PARTIALLY_BUILT_AWAITING_BURN: the tb1 T3 sealed full-confirm cadence "
            "(gate_every 10) will emit >=2 n600 d_seg+bytes+wall-clock endpoints per burn "
            "window; the S composition still owes the pose leg (eg1 terminal pose finisher "
            "rehearsed, external governor required). tb1 T2 surfaced ONE full-confirm "
            "endpoint per arm (seg-only, pose stubbed) — not two same-campaign exact "
            "S-endpoints, so the gate stays HELD"
        ),
        "unblocks": ("regret_bounded_duty_allocation",),
    },
    "regret_bounded_duty_allocation": {
        "named_gate": "TYPED_FIRED_DUTY_HISTORY_LEDGER_PLUS_ACTIVE_COMPRESSION_PROGRESS_PER_EFFORT",
        "producer": (
            "a typed fired-duty history ledger + an active compression-progress-per-effort signal "
            "(downstream of the compression-progress gate; CT1's 12 geometry-cure events are a "
            "count, not typed duty identity/outcome history)"
        ),
        "producer_state": "DOWNSTREAM_OF_COMPRESSION_PROGRESS_GATE",
        "unblocks": (),
    },
}


def _co5_disposition(passed: bool, named_gate: str | None) -> tuple[str, str]:
    """Derive an enhancement's disposition from its backtest + named-gate availability.

    ACTIVATE if the premise is satisfied at consumption; RE_PREMISE (tracked, not silent) when a
    named producer could flip the backtest; RETIRE only when no producer can supply the evidence.
    """

    if passed:
        return "ACTIVATE_ADVISORY", "premise satisfied at consumption; advisory-only, actuation NONE"
    if named_gate:
        return (
            "RE_PREMISE",
            f"held on named producer gate {named_gate}; tracked in the duty-to-measure queue",
        )
    return "RETIRE", "no producer can supply the required evidence; design closed with recorded reason"


DIMENSION_CONTRACT = ".omx/research/ddm_366_dimension_completeness_contract_20260724.md"
J8F_GLOBS = (
    ".omx/research/ddm_j8f_*/ddm_j8f_event_marks.jsonl",
    ".omx/research/ddm_j8f_*/event_marks.jsonl",
)


@dataclass(frozen=True, slots=True)
class CampaignSource:
    name: str
    path: str
    schema: str | None
    horizon: str


SOURCES: tuple[CampaignSource, ...] = (
    CampaignSource(
        "dimension_contract",
        DIMENSION_CONTRACT,
        None,
        "until #366 dimension contract is superseded by a new committed contract",
    ),
    CampaignSource(
        "ms4d_metric_bundle",
        ".omx/research/ddm_ms4d_direct_metric_completion_20260724T155932Z/BUNDLE-COMPLETE.json",
        "ddm_metric_custody_bundle.v1",
        "until scorer, R operator, target cache, or 25-bucket membership changes",
    ),
    CampaignSource(
        "ms7_receiver_edges",
        ".omx/research/ddm_ms7_receiver_edges_and_25bucket_reach_20260724T172249Z/ddm_ms7_receiver_edges_receipt.json",
        "ddm_ms7_receiver_edges_receipt.v1",
        "one receiver-object/coder mutation; instance-scoped PF3 control",
    ),
    CampaignSource(
        "rd1_lambda_frontier",
        ".omx/research/ddm_rd1_lambda_continuation_frontier_20260724T011239Z/"
        "ddm_rd1_lambda_continuation_frontier_receipt_v5.json",
        "ddm_rd1_lambda_continuation_frontier_receipt.v4",
        "until candidate domain, scorer custody, or counted archive bytes change",
    ),
    CampaignSource(
        "ev1_campaign_evidence_join",
        ".omx/research/ddm_ev1_campaign_evidence_joins_20260724T191623Z/ddm_ev1_campaign_evidence_join_receipt.json",
        EVIDENCE_JOIN_SCHEMA,
        "until V19/RD1 endpoint bytes, receiver, scorer, G4, or metric custody changes",
    ),
    CampaignSource(
        "ct1_campaign_telemetry",
        ".omx/research/ddm_ct1_campaign_telemetry_encode_20260725T111500Z/r6_rehearsal_receipt.json",
        "ddm_r6_campaign_rehearsal_receipt.v1",
        "content-hash and source-tree valid until the stopped campaign artifacts change",
    ),
    CampaignSource(
        "co4_lambda_ranker",
        ".omx/research/ddm_co4_road_local_and_precision_20260724/ddm_co4_road_local_and_precision_receipt.json",
        "ddm_lambda_ranker_n600_refit.v1",
        "until G3, EV1, MS4D, PF2 assignment, G4, J8F, scorer-oracle, or fold-contract custody changes",
    ),
    CampaignSource(
        "j8e_event_contract",
        ".omx/research/ddm_j8e_688_compile_receipt_20260724.json",
        "ddm_witness_program_compile.v1",
        "until event graph semantic hash or compiled typed configuration changes",
    ),
    CampaignSource(
        "g2f_amplitude_curve",
        ".omx/research/g2f_bidirectional_amplitude_ladder_20260721T145157Z.md",
        None,
        "formulation-scoped; remeasure after basis, receiver, or scorer-state mutation",
    ),
    CampaignSource(
        "v17_realized_curve",
        ".omx/research/ddm_a1_bounded_collateral_realized_n64_20260723T031500Z/"
        "ddm_a1_bounded_collateral_realized_receipt.json",
        "ddm_a1_bounded_collateral_realized_receipt.v1",
        "instance-scoped; relinearize at every accepted operating point",
    ),
    CampaignSource(
        "v16_validity_failure",
        ".omx/research/ddm_v16_coupled_joint_solve_lane_fix_20260723T013500Z/ddm_v16_coupled_joint_solve_receipt.json",
        "ddm_v16_coupled_joint_solve_receipt.v1",
        "closed instance only; family remains open",
    ),
    CampaignSource(
        "feed603_plateau_forks",
        ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md",
        None,
        "until the F1-F7 contingency map is explicitly superseded",
    ),
)


CLASS_E_DIMENSIONS: tuple[tuple[str, str, str], ...] = (
    ("joint_null_energy", "squared_projected_update", "null/gauge energy"),
    ("seg_only_energy", "squared_projected_update", "null/gauge energy"),
    ("pose_only_energy", "squared_projected_update", "null/gauge energy"),
    ("joint_visible_energy", "squared_projected_update", "null/gauge energy"),
    ("projector_rejected_energy", "squared_update", "resize-kernel support / nullity"),
    ("temporal_flicker", "realized_delta_S", "temporal / flicker"),
    ("clip_stationarity", "dimensionless", "clip (n600 stationarity)"),
    ("delta_bytes_per_step", "counted_archive_bytes_per_step", "rate (archive bytes only)"),
    ("dribble_rate", "joint_null_energy_per_step", "null/gauge energy"),
)


PLATEAU_FORKS: tuple[dict[str, str], ...] = (
    {
        "plateau_type": "GRAMMAR_EXPRESSIBLE",
        "fork_id": "F1",
        "formulation": "EXTEND_THE_LIFT",
    },
    {
        "plateau_type": "PAINT_TEXTURE_BUCKET",
        "fork_id": "F2",
        "formulation": "TEMPLATE_PAINT_DESCENT",
    },
    {
        "plateau_type": "GOOD_DIRECTION_SLOW_CONVERGENCE",
        "fork_id": "F3",
        "formulation": "SOLVE_STEP_ALTERNATION",
    },
    {
        "plateau_type": "DESCRIPTION_COST_BOUND",
        "fork_id": "F4",
        "formulation": "PLANE_DESCENT_REDESCRIBE",
    },
    {
        "plateau_type": "GRAMMAR_UNREACHABLE",
        "fork_id": "F5",
        "formulation": "DIRECT_COUNTED_FIELD",
    },
    {
        "plateau_type": "ILL_CONDITIONED_SCORER_DEPTH",
        "fork_id": "F6",
        "formulation": "FEATURE_SPACE_RELAY",
    },
    {
        "plateau_type": "SEG_POSE_COUPLING_PATHOLOGY",
        "fork_id": "F7",
        "formulation": "LEXICOGRAPHIC_ALTERNATING",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def load_campaign_sources(
    repo_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load the fixed evidence set with schema and content-hash custody."""

    public: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for spec in SOURCES:
        path = repo_root / spec.path
        if not path.is_file():
            raise ValueError(f"{spec.name}: missing campaign authority {spec.path}")
        sha = _sha256(path)
        payload: dict[str, Any] = {}
        if spec.schema is not None:
            payload = _load_json(path)
            if payload.get("schema") != spec.schema:
                raise ValueError(f"{spec.name}: schema drift {payload.get('schema')!r} != {spec.schema!r}")
            if payload.get("score_claim") is not False:
                raise ValueError(f"{spec.name}: missing score_claim=false authority firewall")
        public[spec.name] = {
            "path": spec.path,
            "sha256": sha,
            "schema": spec.schema,
            "freshness_horizon": spec.horizon,
            "status": "CONTENT_HASH_VERIFIED",
        }
        payloads[spec.name] = payload
    return public, payloads


def _finite(value: Any, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name}: non-finite value")
    return out


def _exact_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name}: expected exact JSON integer")
    return value


def validate_realized_verdict(
    raw: Mapping[str, Any],
    *,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Validate one J8F verdict and derive only identities in matching units."""

    if raw.get("schema") != VERDICT_SCHEMA:
        raise ValueError(f"verdict schema drift: {raw.get('schema')!r}")
    telemetry = raw.get("telemetry")
    if not isinstance(telemetry, Mapping):
        raise ValueError("event mark telemetry must be a mapping")
    required = (
        "S_before",
        "S_after",
        "counted_bytes_before",
        "counted_bytes_after",
        "delta_S_per_wall_clock_hour",
        "plateau_type",
        "pose_gate_state",
        "noise_sample_delta_S",
        "noise_regime_id",
        "evaluator_band_low",
        "evaluator_band_high",
        "plateau_residual",
        *(name for name, _, _ in CLASS_E_DIMENSIONS),
    )
    missing = [name for name in required if name not in telemetry]
    if missing:
        raise ValueError("verdict missing required #366 telemetry: " + ",".join(missing))
    event_id = raw.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event mark event_id must be nonempty")
    if not isinstance(raw.get("accepted"), bool):
        raise ValueError("event mark accepted must be boolean")
    noise_regime_id = telemetry["noise_regime_id"]
    if not isinstance(noise_regime_id, str) or not noise_regime_id:
        raise ValueError("noise_regime_id must be nonempty")
    plateau_residual = telemetry["plateau_residual"]
    if not isinstance(plateau_residual, Mapping):
        raise ValueError("plateau_residual must be a mapping")
    residual_required = ("residual_type", "metric_id", "value", "units")
    residual_missing = [name for name in residual_required if name not in plateau_residual]
    if residual_missing:
        raise ValueError("plateau_residual misses " + ",".join(residual_missing))
    normalized_residual = {
        "residual_type": str(plateau_residual["residual_type"]),
        "metric_id": str(plateau_residual["metric_id"]),
        "value": _finite(plateau_residual["value"], "plateau_residual.value"),
        "units": str(plateau_residual["units"]),
    }
    if not all(normalized_residual[name] for name in ("residual_type", "metric_id", "units")):
        raise ValueError("plateau_residual string identities must be nonempty")
    before = _finite(telemetry["S_before"], "S_before")
    after = _finite(telemetry["S_after"], "S_after")
    b0 = _exact_int(telemetry["counted_bytes_before"], "counted_bytes_before")
    b1 = _exact_int(telemetry["counted_bytes_after"], "counted_bytes_after")
    band_low = _finite(telemetry["evaluator_band_low"], "evaluator_band_low")
    band_high = _finite(telemetry["evaluator_band_high"], "evaluator_band_high")
    if band_low > band_high:
        raise ValueError("evaluator band is reversed")
    dimensions = {name: _finite(telemetry[name], name) for name, _, _ in CLASS_E_DIMENSIONS}
    candidate_bands: list[dict[str, Any]] = []
    for row in telemetry.get("candidate_evaluator_bands") or []:
        if not isinstance(row, Mapping):
            raise ValueError("candidate_evaluator_bands rows must be mappings")
        candidate_id = row.get("candidate_id")
        band = row.get("evaluator_band")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or not isinstance(band, (list, tuple))
            or len(band) != 2
        ):
            raise ValueError("candidate evaluator-band row is incomplete")
        low = _finite(band[0], "candidate_evaluator_band_low")
        high = _finite(band[1], "candidate_evaluator_band_high")
        if low > high:
            raise ValueError("candidate evaluator band is reversed")
        candidate_bands.append(
            {
                "candidate_id": candidate_id,
                "delta_S": _finite(row.get("delta_S"), "candidate_delta_S"),
                "evaluator_band": [low, high],
            }
        )
    derived_delta_s = after - before
    derived_delta_bytes = b1 - b0
    if not math.isclose(
        dimensions["delta_bytes_per_step"],
        derived_delta_bytes,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("delta_bytes_per_step disagrees with exact counted-byte identity")
    out = {
        "schema": VERDICT_SCHEMA,
        "verdict_id": event_id,
        "accepted": raw["accepted"],
        "S_before": before,
        "S_after": after,
        "delta_S": derived_delta_s,
        "counted_bytes_before": b0,
        "counted_bytes_after": b1,
        "delta_bytes": derived_delta_bytes,
        "delta_S_per_wall_clock_hour": _finite(
            telemetry["delta_S_per_wall_clock_hour"],
            "delta_S_per_wall_clock_hour",
        ),
        "plateau_type": str(telemetry["plateau_type"]).upper(),
        "plateau_residual": normalized_residual,
        "pose_gate_state": str(telemetry["pose_gate_state"]),
        "noise_sample_delta_S": _finite(telemetry["noise_sample_delta_S"], "noise_sample_delta_S"),
        "noise_regime_id": noise_regime_id,
        "evaluator_band": [band_low, band_high],
        "dimensions": dimensions,
        "candidate_evaluator_bands": candidate_bands,
        "source": {
            "path": source_path,
            "sha256": source_sha256,
            "status": "CONTENT_HASH_VERIFIED",
        },
        "evidence_axis": str(telemetry.get("evidence_axis") or EVIDENCE_AXIS),
        "score_claim": False,
        "execution_allowed": False,
    }
    if "alarm_familywise_alpha" in telemetry:
        alpha = _finite(telemetry["alarm_familywise_alpha"], "alarm_familywise_alpha")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alarm_familywise_alpha must be in (0,1)")
        out["alarm_familywise_alpha"] = alpha
    return out


def discover_j8f_verdicts(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the newest future J8F verdict stream, if it exists."""

    matches = sorted({path for pattern in J8F_GLOBS for path in repo_root.glob(pattern)})
    if not matches:
        return [], {
            "available": False,
            "status": "AWAITING_J8F_VERDICT_STREAM",
            "globs": list(J8F_GLOBS),
        }
    path = matches[-1]
    sha = _sha256(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{lineno}: expected JSON object")
            rows.append(
                validate_realized_verdict(
                    payload,
                    source_path=str(path.relative_to(repo_root)),
                    source_sha256=sha,
                )
            )
    return rows, {
        "available": True,
        "status": "CONTENT_HASH_VERIFIED",
        "path": str(path.relative_to(repo_root)),
        "sha256": sha,
        "rows": len(rows),
        "freshness_horizon": "until any accepted operating-point, scorer, receiver, or coder change",
    }


def derive_noise_alarm(
    noise_samples: Sequence[float],
    *,
    familywise_alpha: float,
) -> dict[str, Any]:
    """Derive ``k*sigma`` using a family-wise normal quantile.

    There is intentionally no default alpha and no literal alarm threshold.
    The caller must supply the preregistered error budget.
    """

    samples = [_finite(value, "noise_sample") for value in noise_samples]
    if len(samples) < 2:
        return {
            "status": "AWAITING_MEASURED_NOISE_FLOOR",
            "threshold_abs_delta_S": None,
            "sample_count": len(samples),
        }
    alpha = _finite(familywise_alpha, "familywise_alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError("familywise_alpha must be in (0,1)")
    sigma = stdev(samples)
    k = NormalDist().inv_cdf(1.0 - alpha / (2.0 * len(samples)))
    return {
        "status": "DERIVED_FROM_MEASURED_NOISE_FLOOR",
        "sample_count": len(samples),
        "mean_delta_S": fmean(samples),
        "sigma_delta_S": sigma,
        "familywise_alpha": alpha,
        "k": k,
        "threshold_abs_delta_S": k * sigma,
        "law": "alarm_abs_delta_S = normal_quantile(1-alpha/(2*n)) * sample_stdev",
    }


def derive_top_k_from_evaluator_bands(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Keep every candidate whose evaluator band overlaps the best candidate.

    This makes K a property of measured evaluator uncertainty, not a tuned
    integer.  Lower ``delta_S`` is better.
    """

    if not candidates:
        return {"status": "AWAITING_EVALUATOR_BANDS", "top_k": None, "candidate_ids": []}
    normalized: list[tuple[str, float, float, float]] = []
    for row in candidates:
        band = row.get("evaluator_band")
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            raise ValueError("candidate evaluator_band must have two endpoints")
        low = _finite(band[0], "evaluator_band_low")
        high = _finite(band[1], "evaluator_band_high")
        if low > high:
            raise ValueError("candidate evaluator band is reversed")
        center = _finite(row.get("delta_S"), "delta_S")
        normalized.append((str(row.get("candidate_id")), center, low, high))
    best = min(normalized, key=lambda item: (item[1], item[0]))
    kept = sorted(item[0] for item in normalized if item[2] <= best[3] and item[3] >= best[2])
    return {
        "status": "DERIVED_FROM_EVALUATOR_BAND_OVERLAP",
        "top_k": len(kept),
        "candidate_ids": kept,
        "best_candidate_id": best[0],
        "best_evaluator_band": [best[2], best[3]],
        "law": "K = cardinality of candidate confidence bands intersecting the best band",
    }


def route_plateau(
    plateau_type: str | None,
    *,
    trigger_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Route a measured plateau type to the exact FEED-603 F1..F7 fork."""

    normalized = str(plateau_type or "").upper()
    for row in PLATEAU_FORKS:
        if row["plateau_type"] == normalized:
            return {
                "schema": DECISION_SCHEMA,
                "status": "PRE_REGISTERED_FORK_SELECTED_ADVISORY",
                **row,
                "trigger_evidence": (dict(trigger_evidence) if trigger_evidence is not None else None),
                "law_ref": "FEED-603-descent-formulation-contingency-map",
                "actuation": "NONE",
                "main_landing_review_required": True,
            }
    return {
        "schema": DECISION_SCHEMA,
        "status": (
            "AWAITING_MEASURED_PLATEAU"
            if not normalized or normalized in {"NONE", "DESCENDING"}
            else "BLOCKED_UNKNOWN_PLATEAU_TYPE"
        ),
        "plateau_type": normalized or None,
        "fork_id": None,
        "formulation": None,
        "trigger_evidence": (dict(trigger_evidence) if trigger_evidence is not None else None),
        "law_ref": "FEED-603-descent-formulation-contingency-map",
        "actuation": "NONE",
        "main_landing_review_required": True,
    }


def _standing_sense_rows(verdicts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    last = verdicts[-1] if verdicts else None
    rows: list[dict[str, Any]] = []
    for name, units, contract_dim in CLASS_E_DIMENSIONS:
        rows.append(
            {
                "schema": SENSE_ROW_SCHEMA,
                "row_id": name,
                "mechanism_class": "E_TELEMETRY",
                "dimension_contract_row": contract_dim,
                "units": units,
                "value": (last["dimensions"][name] if last else None),
                "status": "MEASURED_J8F_REALIZED" if last else "AWAITING_J8F_MEASUREMENT",
                "verdict_id": last["verdict_id"] if last else None,
                "staleness_lineage": last["source"]
                if last
                else {
                    "status": "NO_J8F_SOURCE_ROW",
                    "authority": DIMENSION_CONTRACT,
                },
                "score_claim": False,
            }
        )
    return rows


def _rd1_metric_rows(
    rd1: Mapping[str, Any],
    evidence_join: Mapping[str, Any],
    source: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    edges = {int(row["dual_index"]): row for row in evidence_join["rd1_evidence"]["edge_summaries"]}
    for raw in rd1.get("aggregate_scalarization_controls") or []:
        dual = int(raw["dual_index"])
        edge = edges[dual]
        delta_bytes = _exact_int(edge["delta_counted_bytes"], "ev1_delta_counted_bytes")
        delta_d = _finite(edge["joint_delta_D"], "ev1_joint_delta_D")
        marginal_reduction = -delta_d / delta_bytes
        rows.append(
            {
                "schema": METRIC_ROW_SCHEMA,
                "row_id": f"rd1_aggregate_dual_{dual}",
                "scope": "RESTRICTED_MEASURED_N600_DESCRIPTION_POOL",
                "constraint_group": raw["constraint_group"],
                "epistemic_status": ("DERIVED_FROM_EV1_FRESH_RECEIVER_CLOSED_N600_ENDPOINTS"),
                "lambda_distortion_reduction_per_byte": marginal_reduction,
                "lambda_score_per_byte": (RATE_BREAK_EVEN_SCORE_PER_BYTE - marginal_reduction),
                "rate_break_even_score_per_byte": RATE_BREAK_EVEN_SCORE_PER_BYTE,
                "left_candidate_id": edge["before_endpoint"],
                "right_candidate_id": edge["after_endpoint"],
                "delta_counted_bytes": delta_bytes,
                "delta_D_realized": delta_d,
                "status": ("DERIVED_FROM_EV1_FRESH_N600_ENDPOINTS_NONADDITIVE_CONTROL"),
                "source": dict(source),
                "score_claim": False,
            }
        )
    return rows


def _rd1_bucket_rows(
    rd1: Mapping[str, Any],
    evidence_join: Mapping[str, Any],
    *,
    rd1_source: Mapping[str, Any],
    evidence_source: Mapping[str, Any],
    ms4d_source: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expose every typed RD1 dimension dual without manufacturing a price."""

    evidence_by_key = {evidence_bucket_key(row): row for row in evidence_join["rd1_evidence"]["bucket_rows"]}
    rows: list[dict[str, Any]] = []
    for raw in rd1.get("duals") or []:
        key = (
            int(raw["dual_index"]),
            str(raw["stratum"]),
            str(raw["scorer_visibility"]),
            str(raw["g4_temporal_class"]),
        )
        evidence = evidence_by_key.get(key)
        if evidence is None:
            raise ValueError(f"EV1 evidence join misses RD1 typed key {key!r}")
        bytes_per_d = raw.get("lambda_bytes_per_D_dimension")
        actionable = bool(raw.get("actionable_for_train_decision"))
        if actionable and bytes_per_d is not None:
            bytes_per_d = _finite(bytes_per_d, "lambda_bytes_per_D_dimension")
            if bytes_per_d <= 0.0:
                raise ValueError("actionable dimension lambda_bytes_per_D must be positive")
            reduction_per_byte = 1.0 / bytes_per_d
            score_per_byte = RATE_BREAK_EVEN_SCORE_PER_BYTE - reduction_per_byte
        else:
            reduction_per_byte = None
            score_per_byte = None
        rows.append(
            {
                "schema": METRIC_ROW_SCHEMA,
                "row_id": (
                    f"rd1_dimension_dual_{int(raw['dual_index'])}_"
                    f"{raw['stratum']}_{raw['g4_temporal_class']}_"
                    f"{raw['scorer_visibility']}"
                ),
                "scope": "RD1_TYPED_DIMENSION_POSTSOLVE",
                "dual_index": int(raw["dual_index"]),
                "stratum": raw["stratum"],
                "g4_temporal_class": raw["g4_temporal_class"],
                "scorer_visibility": raw["scorer_visibility"],
                "delta_D_dimension": _finite(
                    evidence["delta_D_dimension"],
                    "ev1_delta_D_dimension",
                ),
                "delta_counted_bytes_dimension": int(evidence["delta_counted_bytes_dimension"]),
                "byte_home_scope": evidence["scope"],
                "byte_home_k": _finite(evidence["k"], "ev1_byte_home_k"),
                "byte_home_k_numerator": int(evidence["k_numerator"]),
                "byte_home_k_denominator": int(evidence["k_denominator"]),
                "amortized_bytes_per_frame": _finite(
                    evidence["amortized_bytes_per_frame"],
                    "ev1_amortized_bytes_per_frame",
                ),
                "byte_home_ranges": list(evidence["byte_home_ranges"]),
                "receiver_uint8_abs_step_histogram": list(evidence["receiver_uint8_abs_step_histogram"]),
                "receiver_changed_channel_values": int(evidence["receiver_changed_channel_values"]),
                "receiver_uint8_abs_step_sum": int(evidence["receiver_uint8_abs_step_sum"]),
                "evidence_status": ("MEASURED_RECEIVER_CLOSED_AMORTIZED_HOME_AND_HISTOGRAM"),
                "pricing_owner": evidence["pricing_owner"],
                "lambda_bytes_per_D": bytes_per_d,
                "lambda_distortion_reduction_per_byte": reduction_per_byte,
                "lambda_score_per_byte": score_per_byte,
                "actionable_for_train_decision": actionable,
                "status": ("EVIDENCE_MEASURED_PRICING_SOLVE_PENDING_MS2R" if not actionable else str(raw["status"])),
                "verdict_scope": (
                    "restricted n600 description-level continuation; null prices are not family negatives"
                ),
                "sources": {
                    "rd1_dimension_dual": dict(rd1_source),
                    "ev1_receiver_evidence": dict(evidence_source),
                    "ms4d_metric_custody": dict(ms4d_source),
                },
                "score_claim": False,
            }
        )
    return rows


def _scoped_trust_regions(
    payloads: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw_rows = payloads["v17_realized_curve"].get("validity_radius", {}).get("raw_rows", [])
    boundary = sorted(
        (row for row in raw_rows if row.get("basis") == "boundary_normal_2x2"),
        key=lambda row: int(row["lattice_quanta"]),
    )
    valid_prefix: list[int] = []
    for row in boundary:
        rho = row.get("rho")
        if rho is None or _finite(row["realized_reduction"], "realized_reduction") <= 0.0 or _finite(rho, "rho") <= 0.0:
            break
        valid_prefix.append(int(row["lattice_quanta"]))
    radius = max(valid_prefix) if valid_prefix else None
    ms7 = payloads["ms7_receiver_edges"]
    pf3_radius = ms7.get("pf3", {}).get("dynamic_quantum_calibration", {}).get("calibration", {}).get("validity_radius")
    g2f_source = dict(sources["g2f_amplitude_curve"])
    g2f_matches = g2f_source["sha256"] == G2F_SOURCE_SHA256
    return [
        {
            "scope": "G2F_BIDIRECTIONAL_PIXEL_AMPLITUDE_CURVE",
            "value": 1.0 if g2f_matches else None,
            "units": "native_pixel_amplitude",
            "status": ("MEASURED_FORMULATION_SCOPED_KNEE" if g2f_matches else "SOURCE_CHANGED_REDERIVE_KNEE"),
            "law_ref": "g2f_bidirectional_amplitude_ladder_measured_knee_v1",
            "provenance_rung": "MEASURED",
            "source": g2f_source,
            "expected_source_sha256": G2F_SOURCE_SHA256,
            "not_universal": True,
        },
        {
            "scope": "V16_COUPLED_JOINT_SOLVE_LANE_FIX_INSTANCE",
            "value": None,
            "units": "per_DOF_trust_box",
            "status": "EXCLUDED_LINEARIZATION_INVALID_FORMULATION_SCOPED",
            "law_ref": "ddm_v16_coupled_joint_solve_instance_validity_v1",
            "provenance_rung": "MEASURED_NEGATIVE_FORMULATION_SCOPED",
            "source": dict(sources["v16_validity_failure"]),
            "verdict_scope": ("one V16 coupled solve instance; does not close the validity-radius family"),
            "not_universal": True,
        },
        {
            "scope": "V17_BOUNDARY_NORMAL_2X2_MATCHED_PREFIX",
            "value": radius,
            "units": "lattice_quanta",
            "status": (
                "DERIVED_MAX_CONSECUTIVE_POSITIVE_REALIZED_RATIO"
                if radius is not None
                else "NO_POSITIVE_REALIZED_PREFIX"
            ),
            "law_ref": "ddm_v17_realized_validity_ratio_uint8_v1",
            "provenance_rung": "DERIVED_FROM_MEASURED",
            "source": dict(sources["v17_realized_curve"]),
            "not_universal": True,
        },
        {
            "scope": "MS7_PF3_CLASS_BIRTH_INSTANCE",
            "value": pf3_radius,
            "units": "signed_lattice_quanta",
            "status": "MEASURED_INSTANCE_ONLY",
            "law_ref": "dynamic_quantum_calibration_v1",
            "provenance_rung": "MEASURED",
            "source": dict(sources["ms7_receiver_edges"]),
            "not_universal": True,
        },
    ]


def _blocker(
    blocker_id: str,
    *,
    scope: str,
    evidence: Mapping[str, Any],
    required_evidence: Iterable[str],
) -> dict[str, Any]:
    return {
        "schema": BLOCKER_SCHEMA,
        "blocker_id": blocker_id,
        "status": "BLOCKED",
        "verdict_scope": scope,
        "evidence": dict(evidence),
        "required_evidence": list(required_evidence),
        "actuation": "NONE",
        "score_claim": False,
    }


def _lambda_ranker_state(
    payload: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and compact the N600 held-out ranker receipt for consumers."""

    for key, expected in (
        ("research_only", True),
        ("execution_allowed", False),
        ("actuation", "NONE"),
        ("score_claim", False),
        ("promotion_eligible", False),
        ("main_landing_review_required", True),
    ):
        if payload.get(key) != expected:
            raise ValueError(f"co4_lambda_ranker: authority firewall drift at {key}")
    content = dict(payload)
    content_sha = content.pop("content_sha256", None)
    if not isinstance(content_sha, str) or _canonical_sha(content) != content_sha:
        raise ValueError("co4_lambda_ranker: content_sha256 mismatch")

    population = payload.get("population") or {}
    if (
        population.get("required_pairs") != 600
        or population.get("joined_pairs") != 600
        or population.get("heldout_unit") != "source_pair_id"
    ):
        raise ValueError("co4_lambda_ranker: exact N600 pair-held-out contract drift")
    selected = payload.get("selected_model") or {}
    metrics = selected.get("metrics") or {}
    if metrics.get("heldout_only") is not True or metrics.get("n_pairs") != 600:
        raise ValueError("co4_lambda_ranker: selected metrics are not held-out N600")
    admission = payload.get("admission_gate") or {}
    if (
        admission.get("metric") != "concatenated_pair_out_of_fold_ndcg_at_4"
        or float(admission.get("threshold", math.nan)) != 0.75
        or float(admission.get("observed", math.nan)) != float(metrics.get("ndcg_at_4", math.nan))
        or bool(admission.get("passed")) != (float(metrics.get("ndcg_at_4", math.nan)) >= 0.75)
        or bool(admission.get("duty_ranking_upgrade_eligible")) != bool(admission.get("passed"))
    ):
        raise ValueError("co4_lambda_ranker: preregistered admission contract drift")

    road_gate = payload.get("road_local_gate") or {}
    if (
        float(road_gate.get("road_threshold_ndcg_at_4", math.nan)) != 0.60
        or float(road_gate.get("global_threshold_ndcg_at_4", math.nan)) != 0.75
        or (road_gate.get("road_observed") or {}).get("heldout_only") is not True
        or (road_gate.get("road_observed") or {}).get("n_pairs") != 288
        or road_gate.get("evaluation_slice_is_router_forbidden") is not True
    ):
        raise ValueError("co4_lambda_ranker: Road-local gate contract drift")

    checks = {str(row["check_id"]): dict(row) for row in payload.get("self_checks") or []}
    precision = checks.get("wallace_mml_pair_precision") or {}
    precision_value = precision.get("value") or {}
    pair_intervals = int(precision_value.get("pair_intervals", 0))
    if int(precision_value.get("required", 600)) != 600:
        raise ValueError("co4_lambda_ranker: Wallace/MML required-pair count drift")
    pair_rankings = list(payload.get("pair_rankings") or [])
    if len(pair_rankings) != 600:
        raise ValueError("co4_lambda_ranker: pair-ranking cardinality drift")
    if any(row.get("score_claim") is not False or row.get("actuation") != "NONE" for row in pair_rankings):
        raise ValueError("co4_lambda_ranker: pair-row authority firewall drift")
    direct_count = sum(row.get("precision_class") == "DIRECT" for row in pair_rankings)
    propagated_count = sum(row.get("precision_class") == "PROPAGATED" for row in pair_rankings)
    unranked_count = sum(row.get("precision_class") == "UNRANKED" for row in pair_rankings)
    if (
        direct_count != int(precision_value.get("direct", -1))
        or propagated_count != int(precision_value.get("propagated", -1))
        or unranked_count != int(precision_value.get("unranked", -1))
        or direct_count + propagated_count + unranked_count != 600
    ):
        raise ValueError("co4_lambda_ranker: precision-class counts drift")
    for row in pair_rankings:
        if row.get("precision_class") == "PROPAGATED" and not (
            float(row["precision_design_effect"]) > 1.0
            and float(row["fisher_standard_error"]) > float(row["nominal_fisher_standard_error"])
        ):
            raise ValueError("co4_lambda_ranker: propagated interval lacks assumption penalty")
        if row.get("pair_order_class") not in {
            "LEADER",
            "ORDERED",
            "TIED",
            "UNRANKED",
        }:
            raise ValueError("co4_lambda_ranker: pair-order class drift")
    decide_rows = list(payload.get("decide_rows") or [])
    if not decide_rows or any(
        not isinstance(row.get("rudin_explanation"), dict)
        or row.get("actuation") != "NONE"
        or row.get("score_claim") is not False
        for row in decide_rows
    ):
        raise ValueError("co4_lambda_ranker: DECIDE explanation/firewall drift")
    blocker_ids = list(payload.get("blocker_ids") or [])
    if payload.get("j8f_blocker_preserved") is not True or "BLOCKED_J8F_REALIZED_VERDICT_TELEMETRY" not in blocker_ids:
        raise ValueError("co4_lambda_ranker: J8F blocker was not preserved")
    g4_counts: dict[str, int] = {}
    g4_statuses: set[str] = set()
    for row in pair_rankings:
        label = str(row.get("g4_temporal_class"))
        g4_counts[label] = g4_counts.get(label, 0) + 1
        g4_statuses.add(str(row.get("g4_pair_class_status")))
    race = [
        {
            "candidate_id": row.get("candidate_id"),
            "status": row.get("status"),
            "learned_form_tag": row.get("learned_form_tag"),
            "metrics": row.get("metrics"),
        }
        for row in payload.get("model_race") or []
    ]
    top_pairs = sorted(
        pair_rankings,
        key=lambda row: (-float(row["prediction"]), int(row["pair_id"])),
    )[:8]
    oracle_counts = {
        name: dict((payload.get("source_lineage") or {})[name]["surface_counts"])
        for name in (
            "margin_fisher_oracle",
            "pf2_bucket_assignment_oracle",
            "pose_tube_oracle",
            "stationarity_oracle",
        )
    }
    return {
        "schema": payload["schema"],
        "status": "HELDOUT_ADMITTED" if admission["passed"] else "HELDOUT_REJECTED",
        "verdict_scope": payload.get("verdict_scope"),
        "source": dict(source),
        "content_sha256": content_sha,
        "population": dict(population),
        "selected_model": dict(selected),
        "admission_gate": dict(admission),
        "road_local_gate": dict(road_gate),
        "historical_comparison": dict(payload.get("historical_comparison") or {}),
        "model_race": race,
        "ranking_error_slices": list(payload.get("ranking_error_slices") or []),
        "innovations": dict(payload.get("innovations") or {}),
        "pair_precision": {
            "status": precision.get("status"),
            "pair_intervals": pair_intervals,
            "direct": direct_count,
            "propagated": propagated_count,
            "unranked": unranked_count,
            "required_pairs": 600,
            "complete": pair_intervals == 600,
            "unranked_precision_owed": 600 - pair_intervals,
            "pair_duty_ranking_status": (
                "ADVISORY_RANK_ELIGIBLE"
                if admission["passed"] and pair_intervals == 600
                else "BLOCKED_INCOMPLETE_FISHER_PRECISION"
            ),
        },
        "top_heldout_diagnostics_nonactionable": top_pairs,
        "g4_pair_class_surface": {
            "counts": g4_counts,
            "statuses": sorted(g4_statuses),
            "pairwise_class_separability_claimed": False,
        },
        "oracle_surface_counts": oracle_counts,
        "self_checks": list(checks.values()),
        "decide_rows": decide_rows,
        "rudin_explanation": dict(payload.get("rudin_explanation") or {}),
        "bandit_allocation": dict(payload.get("bandit_allocation") or {}),
        "blocker_ids": blocker_ids,
        "j8f_blocker_preserved": True,
        "actuation": "NONE",
        "score_claim": False,
    }


def _campaign_tree_fingerprint(root: Path) -> dict[str, Any]:
    """Recompute CT1's RFC-8785-compatible small-tree custody summary.

    CT1 serialized only strings and integers in this tree manifest, so the
    repository's canonical JSON helper is byte-equivalent to its RFC-8785
    encoder.  Symlinks and special files are refused because neither can be
    covered by the settled receipt.
    """

    if not root.is_dir():
        raise FileNotFoundError(f"CT1 source campaign is absent: {root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"CT1 source campaign contains a symlink: {path}")
        if not path.is_file():
            if path.is_dir():
                continue
            raise ValueError(f"CT1 source campaign contains a special file: {path}")
        stat = path.stat()
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": stat.st_size,
                "mtime_ns": str(stat.st_mtime_ns),
                "sha256": _sha256(path),
            }
        )
    return {
        "file_count": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "tree_sha256": _canonical_sha({"files": rows}),
    }


def _ct1_telemetry_rows(
    payload: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], Mapping[str, Any]]:
    """Validate CT1's six authority-separated campaign observability rows."""

    for key, expected in (
        ("research_only", True),
        ("execution_allowed", False),
        ("actuation", "NONE"),
        ("score_claim", False),
        ("main_landing_review_required", True),
    ):
        if payload.get(key) != expected:
            raise ValueError(f"ct1_campaign_telemetry: authority firewall drift at {key}")
    observability = payload.get("observability_digest")
    if not isinstance(observability, Mapping):
        raise ValueError("ct1_campaign_telemetry: observability_digest is absent")
    for key, expected in (
        ("schema", "ddm_campaign_run_observability.v1"),
        ("available", True),
        ("actuation", "NONE"),
        ("execution_allowed", False),
        ("score_claim", False),
        ("main_landing_review_required", True),
    ):
        if observability.get(key) != expected:
            raise ValueError(f"ct1_campaign_telemetry: observability drift at {key}")
    raw_rows = observability.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("ct1_campaign_telemetry: rows are absent")
    rows: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("ct1_campaign_telemetry: row is not a mapping")
        row = dict(raw)
        row_id = row.get("row_id")
        if (
            not isinstance(row_id, str)
            or not row_id
            or row.get("schema") != "ddm_campaign_observability_row.v1"
            or row.get("score_claim") is not False
        ):
            raise ValueError("ct1_campaign_telemetry: row identity/firewall drift")
        if row_id in rows:
            raise ValueError(f"ct1_campaign_telemetry: duplicate row {row_id}")
        rows[row_id] = row
    required = {
        "latest_exact_n600_verdict",
        "accepted_steps_and_cadence",
        "cumulative_batch_local_trace",
        "pose_finish_engagement_watch",
        "geometry_cure_events",
        "schedule_endpoint_eta",
    }
    if set(rows) != required:
        raise ValueError(
            "ct1_campaign_telemetry: exact settled row set drift: " + ",".join(sorted(set(rows) ^ required))
        )
    cadence = rows["accepted_steps_and_cadence"]
    measured_seconds = _finite(
        cadence.get("measured_seconds_per_step"),
        "ct1 measured_seconds_per_step",
    )
    if cadence.get("epistemic_status") != "MEASURED" or measured_seconds <= 0.0:
        raise ValueError("ct1_campaign_telemetry: measured cadence is invalid")
    local = rows["cumulative_batch_local_trace"]
    if local.get("epistemic_status") != "ADVISORY_BATCH_LOCAL" or local.get("not_n600_verdict") is not True:
        raise ValueError("ct1_campaign_telemetry: batch-local authority wall drift")
    exact = rows["latest_exact_n600_verdict"]
    if exact.get("epistemic_status") != "MEASURED":
        raise ValueError("ct1_campaign_telemetry: latest n600 row is not measured")
    return rows, observability


def _ct1_consumption_freshness(
    payload: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify CT1 receipt and source-tree lineage at the consumption boundary."""

    campaign = payload.get("source_campaign")
    observability = payload.get("observability_digest")
    if not isinstance(campaign, Mapping) or not isinstance(observability, Mapping):
        raise ValueError("ct1_campaign_telemetry: source campaign custody is absent")
    path_raw = campaign.get("path")
    expected = campaign.get("after")
    before = campaign.get("before")
    if (
        not isinstance(path_raw, str)
        or not path_raw
        or not isinstance(expected, Mapping)
        or not isinstance(before, Mapping)
    ):
        raise ValueError("ct1_campaign_telemetry: source tree identity is incomplete")
    expected_summary = {
        "file_count": _exact_int(expected.get("file_count"), "ct1 expected file_count"),
        "bytes": _exact_int(expected.get("bytes"), "ct1 expected bytes"),
        "tree_sha256": str(expected.get("tree_sha256")),
    }
    static_custody_ok = (
        campaign.get("bit_identical_before_after") is True
        and campaign.get("mutated_by_rehearsal") is False
        and dict(before) == dict(expected)
        and observability.get("run_dir") == path_raw
        and (observability.get("discovery") or {}).get("run_name_hardcoded") is False
    )
    observed: dict[str, Any] | None = None
    stale_reason: str | None = None
    try:
        observed = _campaign_tree_fingerprint(Path(path_raw))
        if observed != expected_summary:
            stale_reason = "SOURCE_TREE_CONTENT_HASH_OR_IDENTITY_CHANGED"
    except (FileNotFoundError, OSError, ValueError) as exc:
        stale_reason = f"{type(exc).__name__}: {exc}"
    if not static_custody_ok and stale_reason is None:
        stale_reason = "CT1_REHEARSAL_OR_DISCOVERY_CUSTODY_DRIFT"
    fresh = stale_reason is None
    return {
        "status": (
            "FRESH_CONTENT_HASH_VERIFIED_AT_CONSUMPTION"
            if fresh
            else "[stale-advisory] HELD_SOURCE_CUSTODY_NOT_CURRENT"
        ),
        "fresh": fresh,
        "tag": "[fresh]" if fresh else "[stale-advisory]",
        "reason": stale_reason,
        "receipt": {
            "path": source["path"],
            "sha256": source["sha256"],
            "status": source["status"],
        },
        "source_tree": {
            "path": path_raw,
            "expected": expected_summary,
            "observed": observed,
        },
        "freshness_horizon": source["freshness_horizon"],
    }


def _co5_enhancement_state(
    *,
    ct1_payload: Mapping[str, Any],
    ct1_source: Mapping[str, Any],
    evidence_join: Mapping[str, Any],
    evidence_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Backtest the four CO4 designs against CT1 x EV1 before activation."""

    rows, _observability = _ct1_telemetry_rows(ct1_payload)
    freshness = _ct1_consumption_freshness(ct1_payload, ct1_source)
    ev1_counts = validate_campaign_evidence_join(evidence_join)
    ev1_context = {
        "source": dict(evidence_source),
        "exact_pair_rows": ev1_counts["exact_pair_rows"],
        "required_pair_rows": ev1_counts["required_pair_rows"],
        "dimension_byte_homes": ev1_counts["dimension_byte_homes"],
        "shared_across_frame_byte_homes": ev1_counts["shared_across_frame_byte_homes"],
        "join_kind": "CROSS_SECTIONAL_N600_ENDPOINT_EVIDENCE_NOT_ORDERED_CAMPAIGN_TRANSITIONS",
    }
    cadence = rows["accepted_steps_and_cadence"]
    local = rows["cumulative_batch_local_trace"]
    pose_watch = rows["pose_finish_engagement_watch"]
    exact = rows["latest_exact_n600_verdict"]
    common = {
        "ct1_freshness": freshness,
        "ev1_evidence_join": ev1_context,
    }
    specs = (
        {
            "enhancement_id": "pontryagin_bellman_transition_residual",
            "layer": "SENSE",
            "spec": (
                "Compute lambda_t-(dL_t/dx_t+J_t^T lambda_(t+1)) only from "
                "ordered adjacent campaign costates and realized transitions."
            ),
            "telemetry_row_ids": [
                "latest_exact_n600_verdict",
                "accepted_steps_and_cadence",
            ],
            "required_evidence": [
                "ordered_adjacent_campaign_costates",
                "realized_transition_jacobian_terms",
            ],
            "backtest_reason": (
                "CT1 has one surfaced exact n600 endpoint plus cadence; its "
                f"pose watch cites {pose_watch['observed_exact_verdict_count']} exact verdicts "
                "but carries no adjacent costates or transition Jacobians. EV1's 600 pair "
                "endpoints are cross-sectional and cannot be ordered into a Bellman trajectory."
            ),
        },
        {
            "enhancement_id": "m34_per_state_dual_consistency",
            "layer": "SENSE",
            "spec": (
                "Compare organ and M34 duals only at the same campaign state, "
                "dimension, units, and measured uncertainty band."
            ),
            "telemetry_row_ids": ["latest_exact_n600_verdict"],
            "required_evidence": [
                "ct1_per_state_m34_dual",
                "same_state_organ_dual",
                "measured_dual_uncertainty_band",
            ],
            "backtest_reason": (
                "CT1 emits no per-state M34 or organ dual. EV1 closes 162 dimension "
                "byte homes but does not create the missing state-indexed dual or uncertainty."
            ),
        },
        {
            "enhancement_id": "compression_progress_per_effort",
            "layer": "SENSE",
            "spec": (
                "Measure receiver-realized campaign delta_S per wall-clock hour; "
                "never promote a batch-local d_seg/d_pose trace to exact n600 progress."
            ),
            "telemetry_row_ids": [
                "accepted_steps_and_cadence",
                "cumulative_batch_local_trace",
                "latest_exact_n600_verdict",
            ],
            "required_evidence": [
                "two_or_more_exact_n600_campaign_S_endpoints",
                "matching_wall_clock_interval",
                "counted_archive_byte_identity",
            ],
            "backtest_reason": (
                f"CT1 measures {cadence['measured_seconds_per_step']:.12g} s/step, but its "
                "only cumulative objective trace is explicitly ADVISORY_BATCH_LOCAL and "
                f"not n600={local['not_n600_verdict']!s}. The surfaced exact row is step "
                f"{exact['global_step']} only. EV1 cannot supply the missing temporal endpoint."
            ),
        },
        {
            "enhancement_id": "regret_bounded_duty_allocation",
            "layer": "DECIDE",
            "spec": (
                "Rank advisory duties by measured compression progress per effort "
                "plus an exploration bonus backed by fired-duty history."
            ),
            "telemetry_row_ids": [
                "accepted_steps_and_cadence",
                "geometry_cure_events",
            ],
            "required_evidence": [
                "active_compression_progress_per_effort",
                "typed_fired_duty_history",
                "preregistered_exploration_confidence",
            ],
            "backtest_reason": (
                "Compression progress per effort failed its authority backtest. CT1's "
                "12 geometry-cure receipts are an event count, not typed fired-duty "
                "identity/outcome history, and EV1 has no wall-clock allocation history."
            ),
        },
    )
    enhancement_rows: list[dict[str, Any]] = []
    for spec in specs:
        reason = spec["backtest_reason"]
        if not freshness["fresh"]:
            reason = f"{freshness['tag']} CT1 input is not current at consumption: {freshness['reason']}; {reason}"
        # Backtest fails closed for all four: CT1 (landed 2026-07-25, AFTER the original gate
        # string) DELIBERATELY declines to fabricate the campaign delta_S/hour and surfaces only
        # one full exact n600 S-endpoint, so the required matched authority genuinely does not
        # exist yet. The disposition is DERIVED from that fact + the named producer gate.
        passed = False
        gate = _CO5_ENHANCEMENT_GATES.get(spec["enhancement_id"], {})
        disposition, disposition_reason = _co5_disposition(passed, gate.get("named_gate"))
        enhancement_rows.append(
            {
                "schema": ENHANCEMENT_SCHEMA,
                "enhancement_id": spec["enhancement_id"],
                "layer": spec["layer"],
                "spec_recalled_from": ("CO4 self-check/DECIDE/bandit rows; CT1 authority separation"),
                "spec": spec["spec"],
                "telemetry_row_ids": spec["telemetry_row_ids"],
                "input_hash_lineage": common,
                "backtest": {
                    "schema": ENHANCEMENT_BACKTEST_SCHEMA,
                    "status": "FAILED_CLOSED_MISSING_MATCHED_AUTHORITY",
                    "passed": passed,
                    "reason": reason,
                    "required_evidence": spec["required_evidence"],
                    "verdict_scope": (
                        "CT1 stopped-v5 telemetry x EV1 N600 campaign evidence join; enhancement design remains open"
                    ),
                },
                "active": disposition == "ACTIVATE_ADVISORY",
                "disposition": disposition,
                "disposition_reason": disposition_reason,
                "named_gate": gate.get("named_gate"),
                "producer_duty": gate.get("producer"),
                "producer_state": gate.get("producer_state"),
                "unblocks": list(gate.get("unblocks") or ()),
                "status": (
                    "ACTIVE_ADVISORY"
                    if disposition == "ACTIVATE_ADVISORY"
                    else f"RE_PREMISED_{gate.get('named_gate', 'NO_GATE')}"
                    if disposition == "RE_PREMISE"
                    else "RETIRED"
                ),
                "actuation": "NONE",
                "score_claim": False,
                "main_landing_review_required": True,
            }
        )
    active_count = sum(row["active"] for row in enhancement_rows)
    re_premised = [row for row in enhancement_rows if row["disposition"] == "RE_PREMISE"]
    retired = [row for row in enhancement_rows if row["disposition"] == "RETIRE"]
    # Duty-to-measure queue: the RE_PREMISE producer gates, ranked by how many sibling
    # enhancements each unblocks (dependency-topology), then id — a tracked drainable queue.
    duty_to_measure = sorted(
        (
            {
                "enhancement_id": row["enhancement_id"],
                "layer": row["layer"],
                "named_gate": row["named_gate"],
                "producer_duty": row["producer_duty"],
                "producer_state": row["producer_state"],
                "unblocks": row["unblocks"],
                "actuation": "NONE",
            }
            for row in re_premised
        ),
        key=lambda r: (-len(r["unblocks"]), r["enhancement_id"]),
    )
    for rank, row in enumerate(duty_to_measure, 1):
        row["rank"] = rank
    if active_count:
        status = "ACTIVE_ADVISORY"
    elif re_premised:
        status = f"RE_PREMISED_TRACKED_QUEUE_{len(re_premised)}_NAMED_GATES"
    else:
        status = "ALL_ENHANCEMENTS_RETIRED"
    return {
        "schema": "ddm_costate_organ_enhancement_activation.v1",
        "status": status,
        "charter_claimed_gate": "CT1_CAMPAIGN_DELTA_S_PER_WALL_CLOCK_HOUR",
        "gate_satisfied": False,
        "re_adjudicated": (
            "2026-07-28: per-enhancement named-gate dispositions replace the silent collective "
            "PREMISE_FALSIFIED; CT1 landed 2026-07-25 but deliberately declines the campaign "
            "delta_S/hour (only 1 full exact n600 S-endpoint surfaced) so all 4 are genuinely "
            "unsatisfied -> RE_PREMISE with named producer gates, 0 retired. Off is a tracked queue."
        ),
        "re_checked": (
            "2026-07-28 (co7): CT1 landed state unchanged (no CT1-v2 receipt) -> all four KEEP "
            "RE_PREMISE, none fire. Producer-state delta: the compression-progress gate's producer "
            "is now PARTIALLY_BUILT_AWAITING_BURN (tb1 T3 sealed full-confirm cadence + eg1 "
            "terminal pose finisher); tb1 T2 surfaced one seg-only full-confirm endpoint per arm, "
            "which is not two same-campaign exact n600 S-endpoints. "
            "2026-07-28 (co8): pn1's S6 VOI table adjudicated against the compression-progress "
            "gate — it is decision analysis on typed ASSUMED priors with ZERO exact n600 "
            "S-endpoints, so it is NOT the named CT1-v2-class producer; the gate stays HELD "
            "(producer still PARTIALLY_BUILT_AWAITING_BURN). The organ's co8 duty-allocator "
            "waterfill ranks duties on DERIVED/ASSUMED value bands and is explicitly distinct "
            "from the gated regret allocator (which requires MEASURED compression-progress-"
            "per-effort + a typed fired-duty history); no gate fires."
        ),
        "source_freshness": freshness,
        "active_count": active_count,
        "re_premised_count": len(re_premised),
        "retired_count": len(retired),
        "held_count": len(enhancement_rows) - active_count,
        "total_count": len(enhancement_rows),
        "dispositions": {row["enhancement_id"]: row["disposition"] for row in enhancement_rows},
        "duty_to_measure": duty_to_measure,
        "rows": enhancement_rows,
        "actuation": "NONE",
        "score_claim": False,
        "main_landing_review_required": True,
    }


def campaign_consumer_view(state: Mapping[str, Any], consumer: str) -> dict[str, Any]:
    """Return a named view and prove it came from this exact state digest."""

    view = (state.get("consumers") or {}).get(consumer)
    if not isinstance(view, dict):
        raise KeyError(f"unknown DDM campaign consumer {consumer!r}")
    if view.get("state_digest") != state.get("state_digest"):
        raise ValueError(f"{consumer}: forked campaign-state digest")
    return dict(view)


def build_campaign_costate(
    *,
    repo_root: Path,
    verdicts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compose one hash-lineaged campaign state for every read-only consumer."""

    sources, payloads = load_campaign_sources(repo_root)
    persisted, j8f_source = discover_j8f_verdicts(repo_root)
    realized = list(verdicts) if verdicts is not None else persisted
    for row in realized:
        if row.get("schema") != VERDICT_SCHEMA or "dimensions" not in row:
            raise ValueError("build_campaign_costate requires validated campaign verdict rows")

    rd1 = payloads["rd1_lambda_frontier"]
    ms7 = payloads["ms7_receiver_edges"]
    evidence_join = payloads["ev1_campaign_evidence_join"]
    lambda_ranker = _lambda_ranker_state(
        payloads["co4_lambda_ranker"],
        sources["co4_lambda_ranker"],
    )
    evidence_counts = validate_campaign_evidence_join(evidence_join)
    enhancement_state = _co5_enhancement_state(
        ct1_payload=payloads["ct1_campaign_telemetry"],
        ct1_source=sources["ct1_campaign_telemetry"],
        evidence_join=evidence_join,
        evidence_source=sources["ev1_campaign_evidence_join"],
    )
    exact_pair_rows = evidence_counts["exact_pair_rows"]
    required_pair_rows = evidence_counts["required_pair_rows"]
    typed_dimension_rows = len(rd1.get("duals") or [])
    actionable_dimension_prices = sum(bool(row.get("actionable_for_train_decision")) for row in rd1.get("duals") or [])
    metric_rows = _rd1_metric_rows(
        rd1,
        evidence_join,
        sources["ev1_campaign_evidence_join"],
    )
    bucket_rows = _rd1_bucket_rows(
        rd1,
        evidence_join,
        rd1_source=sources["rd1_lambda_frontier"],
        evidence_source=sources["ev1_campaign_evidence_join"],
        ms4d_source=sources["ms4d_metric_bundle"],
    )
    measured_dimension_homes = evidence_counts["dimension_byte_homes"]
    measured_shared_homes = evidence_counts["shared_across_frame_byte_homes"]
    measured_per_frame_homes = evidence_counts["per_frame_byte_homes"]
    measured_receiver_histograms = evidence_counts["receiver_uint8_histograms"]
    sense_rows = _standing_sense_rows(realized)
    latest = realized[-1] if realized else None
    decision = route_plateau(
        latest.get("plateau_type") if latest else None,
        trigger_evidence=latest.get("plateau_residual") if latest else None,
    )
    trust = _scoped_trust_regions(payloads, sources)

    latest_noise_regime = latest.get("noise_regime_id") if latest else None
    noise_rows = [row for row in realized if row.get("noise_regime_id") == latest_noise_regime]
    noise_policy = (
        derive_noise_alarm(
            [row["noise_sample_delta_S"] for row in noise_rows],
            familywise_alpha=_finite(latest["alarm_familywise_alpha"], "alarm_familywise_alpha"),
        )
        if latest and "alarm_familywise_alpha" in latest
        else {
            "status": "AWAITING_PREREGISTERED_ALPHA_AND_MEASURED_NOISE_FLOOR",
            "threshold_abs_delta_S": None,
            "sample_count": len(noise_rows),
        }
    )
    noise_policy.update(
        {
            "law_ref": "ddm_campaign_familywise_noise_alarm_v1",
            "provenance_rung": (
                "DERIVED_FROM_MEASURED"
                if noise_policy["threshold_abs_delta_S"] is not None
                else "BLOCKED_MISSING_MEASUREMENT_OR_POLICY"
            ),
            "noise_regime_id": latest_noise_regime,
            "source_lineage": latest.get("source") if latest else None,
        }
    )
    candidate_bands = list(latest.get("candidate_evaluator_bands") or []) if latest else []
    top_k = derive_top_k_from_evaluator_bands(candidate_bands)
    top_k.update(
        {
            "law_ref": "ddm_campaign_evaluator_band_overlap_top_k_v1",
            "provenance_rung": (
                "DERIVED_FROM_MEASURED" if top_k["top_k"] is not None else "BLOCKED_MISSING_MEASUREMENT"
            ),
            "source_lineage": latest.get("source") if latest else None,
        }
    )

    blockers = [
        _blocker(
            "BLOCKED_J8F_REALIZED_VERDICT_TELEMETRY",
            scope="campaign SENSE; no inference from J8E compile or historical proxies",
            evidence={
                "verdict_rows": len(realized),
                "j8f_source_status": j8f_source["status"],
            },
            required_evidence=(
                "schema_valid_ddm_event_mark_v1_rows",
                "all_DDM_366_class_E_dimensions",
            ),
        ),
    ]
    if realized:
        blockers = [row for row in blockers if row["blocker_id"] != "BLOCKED_J8F_REALIZED_VERDICT_TELEMETRY"]
    ranker_admitted = bool(lambda_ranker["admission_gate"]["passed"])
    precision_owed = int(lambda_ranker["pair_precision"]["unranked_precision_owed"])
    if not ranker_admitted:
        blockers.append(
            _blocker(
                "BLOCKED_CO4_HELDOUT_NDCG_ADMISSION",
                scope=lambda_ranker["verdict_scope"],
                evidence={
                    "ndcg_at_4": lambda_ranker["admission_gate"]["observed"],
                    "threshold": lambda_ranker["admission_gate"]["threshold"],
                },
                required_evidence=("heldout_ndcg_at_4_gte_0.75",),
            )
        )
    if precision_owed:
        blockers.append(
            _blocker(
                f"BLOCKED_PAIR_LEVEL_MS4D_FISHER_PRECISION_{precision_owed}",
                scope=("pair-order confidence only; held-out aggregate ranker admission remains measured"),
                evidence={
                    "pair_intervals": lambda_ranker["pair_precision"]["pair_intervals"],
                    "required_pairs": 600,
                },
                required_evidence=(
                    "positive_direct_or_validated_PF2_support_propagated_MS4D_Fisher_for_every_N600_pair",
                ),
            )
        )
    for row in enhancement_state["rows"]:
        if not row["active"]:
            blockers.append(
                _blocker(
                    f"BLOCKED_CO5_{row['enhancement_id'].upper()}",
                    scope=row["backtest"]["verdict_scope"],
                    evidence={
                        "backtest_status": row["backtest"]["status"],
                        "reason": row["backtest"]["reason"],
                        "ct1_freshness": enhancement_state["source_freshness"]["status"],
                    },
                    required_evidence=row["backtest"]["required_evidence"],
                )
            )
    core = {
        "schema": SCHEMA,
        "available": True,
        "status": "CAMPAIGN_SENSE_ADVISORY",
        "maturity": MATURITY,
        "research_only": True,
        "execution_allowed": False,
        "actuation": "NONE",
        "score_claim": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
        "evidence_axis": EVIDENCE_AXIS,
        "source_lineage": {
            "sources": sources,
            "j8f_verdict_stream": j8f_source,
            "lineage_digest": _canonical_sha({name: row["sha256"] for name, row in sources.items()}),
        },
        "metric_state": {
            "scorer_metric": EVIDENCE_METRIC_ID,
            "aggregate_scalarization_controls": metric_rows,
            "bucket_exchange_rates": bucket_rows,
            "bucket_exchange_rate_status": (
                f"EVIDENCE_MEASURED_{measured_dimension_homes}_OF_"
                f"{typed_dimension_rows}; PRICING_PENDING_MS2R_"
                f"{actionable_dimension_prices}_OF_{typed_dimension_rows}_ACTIONABLE"
            ),
            "v19_receiver_closed_join_status": (f"MEASURED_{exact_pair_rows}_OF_{required_pair_rows}"),
            "rd1_dimension_evidence_status": (
                f"MEASURED_{measured_dimension_homes}_AMORTIZED_HOMES_"
                f"({measured_shared_homes}_SHARED,{measured_per_frame_homes}_PER_FRAME)_AND_"
                f"{measured_receiver_histograms}_UINT8_HISTOGRAMS"
            ),
            "ms4d_terminal_status": "UNREACHABLE_BY_COUNTED_COORDINATES",
            "ms4d_terminal_buckets": int(ms7["r0"]["row_count"]),
            "ms7_mass_paying_buckets": int(ms7["r0"]["mass_paying_row_count"]),
            "ms7_pf3_status": ("MEASURED_PRICED_CONTROL_NONADMISSIBLE_R0_AND_ERROR_CAP"),
            "lambda_ranker": lambda_ranker,
        },
        "dynamic_policy": {
            "schema": DYNAMIC_POLICY_SCHEMA,
            "trust_regions": trust,
            "noise_alarm": noise_policy,
            "top_k": top_k,
            "alarm_or_top_k_hardcoded": False,
            "sealed_source_bound_constants": [
                "G2F measured knee guarded by expected source SHA",
                "rate break-even imported from canonical DDM costate law",
            ],
        },
        "sense": {
            "contract": DIMENSION_CONTRACT,
            "standing_rows": sense_rows,
            "enhancement_rows": [row for row in enhancement_state["rows"] if row["layer"] == "SENSE"],
            "verdict_count": len(realized),
            "latest_verdict": latest,
        },
        "decide": {
            "plateau_route": decision,
            "pre_registered_forks": list(PLATEAU_FORKS),
            "lambda_ranker_decide_rows": lambda_ranker["decide_rows"],
            "enhancement_rows": [row for row in enhancement_state["rows"] if row["layer"] == "DECIDE"],
        },
        "enhancement_activation": enhancement_state,
        "blockers": blockers,
    }
    state_digest = _canonical_sha(core)
    waiting_rows = sum(row["value"] is None for row in sense_rows)
    duty_queue: list[dict[str, Any]] = []
    if decision["fork_id"] is not None:
        duty_queue.append(
            {
                "rank": 1,
                "duty": decision["formulation"],
                "fork_id": decision["fork_id"],
                "reason": f"measured plateau type {decision['plateau_type']}",
                "actuation": "NONE",
            }
        )
    else:
        if waiting_rows:
            duty_queue.append(
                {
                    "duty": "J8F_MEASURE_CLASS_E_TELEMETRY",
                    "reason": f"{waiting_rows}/{len(sense_rows)} standing SENSE rows await measurement",
                    "actuation": "NONE",
                }
            )
        if ranker_admitted and precision_owed:
            duty_queue.append(
                {
                    "duty": "CO4_LAMBDA_RANKER_FISHER_PRECISION_CLOSURE",
                    "reason": (
                        f"held-out NDCG@4={lambda_ranker['admission_gate']['observed']:.6g} "
                        "admits the ranker, but "
                        f"{precision_owed}/600 pair-level Fisher intervals remain owed"
                    ),
                    "actuation": "NONE",
                }
            )
        if actionable_dimension_prices < typed_dimension_rows:
            duty_queue.append(
                {
                    "duty": "MS2R_TOLERANCE_CAPPED_DIMENSION_PRICING",
                    "reason": (
                        f"EV1 measured {measured_dimension_homes}/{typed_dimension_rows} "
                        f"exclusive homes ({measured_shared_homes} shared across frames) "
                        "and receiver histograms; ms2r owns the "
                        f"{actionable_dimension_prices}/{typed_dimension_rows} priced solve"
                    ),
                    "actuation": "NONE",
                }
            )
        for rank, row in enumerate(duty_queue, 1):
            row["rank"] = rank
    activation_nag = {
        "status": "BLOCKING_DUTIES_OWED" if blockers else "CLEAR_ADVISORY",
        "standing_sense_rows": len(sense_rows),
        "unmeasured_sense_rows": waiting_rows,
        "blocker_ids": [row["blocker_id"] for row in blockers],
        "lambda_ranker_admission": lambda_ranker["admission_gate"],
        "lambda_ranker_pair_precision": lambda_ranker["pair_precision"],
        "enhancement_activation": enhancement_state,
        "next_duty": duty_queue[0] if duty_queue else None,
        "actuation": "NONE",
    }
    dashboard = {
        "ok": True,
        "schema": SCHEMA,
        "status": core["status"],
        "maturity": MATURITY,
        "verdict_count": len(realized),
        "plateau_route": decision,
        "sense_rows": sense_rows,
        "metric_rows": metric_rows,
        "bucket_rows": bucket_rows,
        "campaign_evidence": {
            "v19_receiver_closed_join_status": core["metric_state"]["v19_receiver_closed_join_status"],
            "rd1_dimension_evidence_status": core["metric_state"]["rd1_dimension_evidence_status"],
            "bucket_exchange_rate_status": core["metric_state"]["bucket_exchange_rate_status"],
        },
        "lambda_ranker": lambda_ranker,
        "enhancement_activation": enhancement_state,
        "blockers": blockers,
        "activation_nag": activation_nag,
        "actuation": "NONE",
    }
    digest = {
        "ok": True,
        "schema": SCHEMA,
        "status": core["status"],
        "verdict_count": len(realized),
        "plateau_route": decision,
        "duty_queue": duty_queue,
        "campaign_evidence": {
            "v19_receiver_closed_join_status": core["metric_state"]["v19_receiver_closed_join_status"],
            "rd1_dimension_evidence_status": core["metric_state"]["rd1_dimension_evidence_status"],
            "bucket_exchange_rate_status": core["metric_state"]["bucket_exchange_rate_status"],
        },
        "lambda_ranker": lambda_ranker,
        "enhancement_activation": enhancement_state,
        "activation_nag": activation_nag,
        "actuation": "NONE",
    }
    consumers = {
        "digest": digest,
        "dashboard": dashboard,
        "duty_queue": {
            "ok": True,
            "rows": duty_queue,
            "lambda_ranker": lambda_ranker,
            "enhancement_activation": enhancement_state,
            "activation_nag": activation_nag,
            "actuation": "NONE",
        },
        "activation_nag": {
            "ok": True,
            "lambda_ranker": lambda_ranker,
            "enhancement_activation": enhancement_state,
            **activation_nag,
        },
    }
    for view in consumers.values():
        view["state_digest"] = state_digest
    return {
        **core,
        "state_digest": state_digest,
        "consumers": consumers,
    }


__all__ = [
    "BLOCKER_SCHEMA",
    "CLASS_E_DIMENSIONS",
    "DECISION_SCHEMA",
    "DYNAMIC_POLICY_SCHEMA",
    "ENHANCEMENT_BACKTEST_SCHEMA",
    "ENHANCEMENT_SCHEMA",
    "J8F_GLOBS",
    "METRIC_ROW_SCHEMA",
    "PLATEAU_FORKS",
    "SCHEMA",
    "SENSE_ROW_SCHEMA",
    "VERDICT_SCHEMA",
    "build_campaign_costate",
    "campaign_consumer_view",
    "derive_noise_alarm",
    "derive_top_k_from_evaluator_bands",
    "discover_j8f_verdicts",
    "load_campaign_sources",
    "route_plateau",
    "validate_realized_verdict",
]
