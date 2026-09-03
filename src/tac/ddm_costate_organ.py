# SPDX-License-Identifier: MIT
"""Live DDM describe-line costate organ (advisory, no actuation).

This top-level module is the successor to the witness-training-era costate
digest.  Keeping the live implementation outside ``tac.witness_control`` means
its import path does not execute the legacy package initializer.  Its SENSE
surface is the latest schema-checked DDM receipt fleet: dv1, g3, g4, the
v19-family, and optional solver-member telemetry, with e1 and dv2 registered
as pending producers.  Every decision row carries the exact input hashes that
make it valid.

Authority firewall:

* local/macOS frozen-scorer rows are advisory and never move the pointer;
* no trainer, launcher, provider, or subprocess import exists here;
* the quarantined 20260717 witness run is not a source;
* missing receiver realization produces a zero admitted lambda and a
  re-derivation duty, never an optimistic substitute.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.ddm_campaign_costate import (
    build_campaign_costate,
    campaign_consumer_view,
)
from tac.ddm_costate_law import (
    EQUATION_ID,
    RATE_BREAK_EVEN_SCORE_PER_BYTE,
    SCHEDULER_EQUATION_ID,
    ddm_joint_costate,
    gauss_southwell_validity_score,
)
from tac.optimization.scorer_analytic_atlas import build_ddm_lambda_bundle
from tac.scorer_value_oracle import ScorerValueOracle

SCHEMA = "ddm_live_costate_organ.v1"
CHECKPOINT_SCHEMA = "ddm_live_costate_checkpoint.v1"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
MATURITY = "_dev"
LEGACY_AUTHORITY_OWED_ROWS = 115
REPO = Path(__file__).resolve().parents[2]

ROUND10_CURRENT_SOURCES = (
    ".omx/state/main_hot_state.md",
    ".omx/research/ddm_dcc1_decoder_causal_conditioning_verdict_20260901.md",
    ".omx/research/ddm_x012_crossing_ledger_20260901.md",
    ".omx/research/ddm_dds1_ceiling_readjudication_20260901.md",
    "src/tac/canonical_equations/decoder_causal_condition_transport_20260901.py",
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    glob: str
    schema: str | None
    required: bool
    horizon: str


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        "dv1",
        "ddm_dv1_description_vocabulary_n600_*/receipt.json",
        "ddm_description_vocabulary_receipt.v1",
        True,
        "content-hash valid until a newer dv1 receipt or receiver realization lands",
    ),
    SourceSpec(
        "g3",
        "ddm_g3_score_atlas_n600_*/ddm_g3_score_atlas_receipt.json",
        "ddm_g3_score_atlas_receipt.v1",
        True,
        "content-hash valid until g3 atlas, scorer custody, or reconstruction changes",
    ),
    SourceSpec(
        "g4",
        "ddm_g4_spatial_stationarity_n600_*/ddm_g4_spatial_stationarity_receipt.json",
        "ddm_g4_spatial_stationarity_receipt.v1",
        True,
        "content-hash valid until recurrence decomposition or receiver realization changes",
    ),
    SourceSpec(
        "v19",
        "ddm_v19_pure_priced_objective_*/ddm_v19_pure_priced_objective_receipt.json",
        "ddm_v19_pure_priced_objective_receipt.v1",
        True,
        "one downstream joint-stack mutation; remeasure before reuse",
    ),
    SourceSpec(
        "ev1",
        "ddm_ev1_campaign_evidence_joins_*/ddm_ev1_campaign_evidence_join_receipt.json",
        "ddm_ev1_campaign_evidence_join_receipt.v1",
        True,
        "until V19/RD1 endpoint bytes, receiver, scorer, G4, or metric custody changes",
    ),
    SourceSpec(
        "v19b",
        "ddm_v19b_joint_remeasure_stack_*/ddm_v19b_joint_remeasure_stack_receipt.json",
        "ddm_v19b_joint_remeasure_stack_receipt.v1",
        True,
        "one downstream joint-stack mutation; measured joint-survival is family-local",
    ),
    SourceSpec(
        "e1",
        "ddm_e1_runtime_exporter_n600_*/ddm_e1_runtime_export_receipt.json",
        "ddm_e1_runtime_export_receipt.v1",
        False,
        "content-hash valid until E1 packet/exporter inputs change",
    ),
    SourceSpec(
        "e2",
        "ddm_e2_pose_stream_and_doctrine_export_*/ddm_e2_runtime_export_receipt.json",
        "ddm_e2_runtime_export_receipt.v1",
        False,
        "content-hash valid until E2 packet/exporter inputs change",
    ),
    SourceSpec(
        "dv2",
        "ddm_dv2_*/receipt.json",
        "sdwl1.n600_measurement_receipt.v1",
        False,
        "content-hash valid until SDWL1 inventory, grammar, or source custody changes",
    ),
    SourceSpec(
        "ms1",
        "ddm_ms1_min_description_lattice_solve_*/receipt.json",
        "ddm_min_description_lattice_solve_receipt.v1",
        False,
        "content-hash valid until the exact member, conditioning expansion, coder, or frozen scorer custody changes",
    ),
)


@dataclass
class DdmCostateCheckpoint:
    """Small resumable state for advisory cycles.

    The state is source-derived and contains no model weights.  It implements
    the canonical ``Resumable`` protocol so callers can register it with
    :class:`tac.witness_control.resume_registry.ResumeRegistry`.
    """

    source_hashes: dict[str, str]
    completed_block_ids: list[str]
    cycle: int = 0

    def state_arrays(self, prefix: str) -> dict[str, Any]:
        import numpy as np

        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "source_hashes": dict(sorted(self.source_hashes.items())),
            "completed_block_ids": sorted(set(self.completed_block_ids)),
            "cycle": int(self.cycle),
        }
        return {prefix + "json": np.asarray([json.dumps(payload, sort_keys=True)])}

    def restore_from_cfg(self, prefix: str, cfg: dict) -> bool:
        raw = cfg.get(prefix + "json")
        if raw is None:
            return False
        if hasattr(raw, "item"):
            raw = raw.item()
        elif isinstance(raw, (list, tuple)):
            raw = raw[0]
        payload = json.loads(str(raw))
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("DDM costate checkpoint schema drift")
        self.source_hashes = {str(k): str(v) for k, v in payload["source_hashes"].items()}
        self.completed_block_ids = [str(v) for v in payload["completed_block_ids"]]
        self.cycle = int(payload["cycle"])
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CHECKPOINT_SCHEMA,
            "source_hashes": dict(sorted(self.source_hashes.items())),
            "completed_block_ids": sorted(set(self.completed_block_ids)),
            "cycle": int(self.cycle),
        }


def register_ddm_costate_checkpoint(registry: Any, checkpoint: DdmCostateCheckpoint) -> Any:
    """Register the advisory controller in the canonical resume registry."""

    return registry.register("ddm_live_costate_advisory", "__ddmcostate_", checkpoint)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _latest(research: Path, pattern: str) -> Path | None:
    paths = sorted(research.glob(pattern))
    return paths[-1] if paths else None


def _validate_receipt(spec: SourceSpec, path: Path, payload: Mapping[str, Any]) -> None:
    if spec.schema is not None and payload.get("schema") != spec.schema:
        raise ValueError(f"{spec.name}: schema drift: {payload.get('schema')!r} != {spec.schema!r}")
    for key, expected in (
        ("score_claim", False),
        ("execution_allowed", False),
        ("research_only", True),
        ("promotion_eligible", False),
    ):
        if key in payload and payload.get(key) is not expected:
            raise ValueError(f"{spec.name}: authority firewall drift at {key}")
    run_id = payload.get("run_id")
    if spec.required and (not isinstance(run_id, str) or not run_id):
        raise ValueError(f"{spec.name}: receipt has no run_id")
    if run_id is not None and path.parent.name != run_id:
        raise ValueError(f"{spec.name}: latest-run custody mismatch: directory={path.parent.name!r} receipt={run_id!r}")
    if not path.is_file():
        raise ValueError(f"{spec.name}: source is not a file")


def discover_sources(repo_root: Path = REPO) -> dict[str, dict[str, Any]]:
    """Select the latest receipt in each registered producer family."""

    research = repo_root / ".omx" / "research"
    out: dict[str, dict[str, Any]] = {}
    for spec in SOURCE_SPECS:
        path = _latest(research, spec.glob)
        if path is None:
            out[spec.name] = {
                "name": spec.name,
                "available": False,
                "required": spec.required,
                "horizon": spec.horizon,
                "reason": "NO_MATCHING_LIVE_RECEIPT",
            }
            continue
        payload = _load_json(path)
        _validate_receipt(spec, path, payload)
        out[spec.name] = {
            "name": spec.name,
            "available": True,
            "required": spec.required,
            "horizon": spec.horizon,
            "path": str(path.relative_to(repo_root)),
            "sha256": _sha256(path),
            "run_id": payload.get("run_id") or path.parent.name,
            "run_id_custody": (
                "RECEIPT_FIELD"
                if payload.get("run_id")
                else "DIRECTORY_ID_PLUS_SCHEMA_AND_CONTENT_HASH"
            ),
            "schema": payload.get("schema"),
            "payload": payload,
        }
    return out


def _verify_named_output(
    receipt: Mapping[str, Any],
    *,
    basename: str,
    fallback: Path | None = None,
) -> tuple[Path | None, str | None, str]:
    rows = list(receipt.get("outputs") or []) + list(receipt.get("compact_outputs") or [])
    for row in rows:
        if Path(str(row.get("path", ""))).name != basename:
            continue
        candidate = Path(str(row["path"]))
        if not candidate.is_file() and fallback is not None and fallback.is_file():
            candidate = fallback
        if not candidate.is_file():
            continue
        observed = _sha256(candidate)
        if observed != row.get("sha256"):
            return candidate, observed, "HASH_MISMATCH"
        if int(row.get("bytes", candidate.stat().st_size)) != candidate.stat().st_size:
            return candidate, observed, "BYTE_COUNT_MISMATCH"
        return candidate, observed, "VERIFIED"
    return None, None, "NOT_FOUND"


def _dv1_summary(repo_root: Path, source: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = source["payload"]
    receipt_path = repo_root / source["path"]
    path, digest, status = _verify_named_output(
        receipt, basename="summary.json", fallback=receipt_path.parent / "summary.json"
    )
    if status != "VERIFIED" or path is None:
        raise ValueError(f"dv1 summary custody failed: {status}")
    return _load_json(path), {
        "path": str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path),
        "sha256": digest,
        "status": status,
    }


def _g3_atlas(
    source: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    receipt = source["payload"]
    path, digest, status = _verify_named_output(receipt, basename="ddm_g3_score_atlas_n600.jsonl")
    if status != "VERIFIED" or path is None:
        return {}, {"status": status, "path": str(path) if path else None, "sha256": digest}
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["pair_index"])] = row
    return rows, {
        "status": "VERIFIED",
        "path": str(path),
        "sha256": digest,
        "rows": len(rows),
    }


def _number(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("non-finite numeric receipt value")
    return out


def _block_costates(v19b: Mapping[str, Any]) -> list[dict[str, Any]]:
    buckets = {row["candidate_id"]: row["buckets"] for row in v19b["c1_bucket_attribution_per_admitted_move"]}
    rows: list[dict[str, Any]] = []
    for measured in v19b["greedy_screen"]["per_move_joint_table"]:
        single = measured["single_step_v19_delta"]
        joint = measured["joint_incremental_delta"]
        exact_gap = max(0.0, -_number(single["joint_delta"]))
        observed = buckets.get(measured["candidate_id"], {})
        helpful = sum(int(row.get("helpful_flips", 0)) for row in observed.values())
        harmful = sum(int(row.get("harmful_flips", 0)) for row in observed.values())
        visibility = helpful / max(helpful + harmful, 1)
        byte_delta = int(joint["delta_archive_bytes"])
        byte_price = 1.0 / max(abs(byte_delta), 1)
        d2 = max(
            0.0,
            min(1.0, _number(measured["nonadditivity"]["survival_fraction"])),
        )
        value = ddm_joint_costate(exact_gap, visibility, 1.0, byte_price, d2)
        rows.append(
            {
                "block_id": measured["candidate_id"],
                "family": measured["family"],
                "move_index": int(measured["move_index"]),
                "exact_gap": exact_gap,
                "visibility": visibility,
                "described_fraction_proxy": visibility,
                "uint8_realizability": 1.0,
                "byte_delta": byte_delta,
                "byte_price": byte_price,
                "dual_tolerance_d2": d2,
                "lambda_d2": value,
                "validity_radius": d2,
                "validity_kind": "V19B_MEASURED_JOINT_SURVIVAL_PROXY",
                "realized_joint_gain": max(0.0, -_number(joint["joint_delta"])),
                "frees_bytes": byte_delta < 0,
            }
        )
    return rows


def _primitive_costates(
    dv1: Mapping[str, Any],
    g4: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expose the measured bytes -> reach -> realized-proxy chain.

    Cell-space reach is not receiver realization.  These rows therefore retain
    their measured upper-bound marginal while their admitted D2 lambda is zero.
    """

    rows: list[dict[str, Any]] = []
    for row in dv1["enriched_mdl_table"]:
        if row.get("axis") != "exact cached semantic cells":
            continue
        counted = int(row["counted_bytes"])
        described = _number(row.get("road_described_fraction", 0.0))
        cell_gain = max(0.0, _number(row.get("net_errors_closed_all_strata", 0.0)))
        rows.append(
            {
                "primitive_id": row["candidate"],
                "producer": "dv1",
                "measured_bytes": counted,
                "described_fraction": described,
                "cell_space_gain": cell_gain,
                "realized_proxy_delta_s": None,
                "cell_upper_bound_per_byte": cell_gain / max(counted, 1),
                "uint8_realizability": 0.0,
                "dual_tolerance_d2": 0.0,
                "lambda_d2": 0.0,
                "status": "RECEIVER_REALIZATION_OWED",
            }
        )
    for row in g4["summary"]["top5_amortization_opportunities"]:
        counted = int(row["byte_measurement"]["selected_bytes"])
        rows.append(
            {
                "primitive_id": row["opportunity_id"],
                "producer": "g4",
                "measured_bytes": counted,
                "described_fraction": _number(row["cell_space_delta_d_seg"]),
                "cell_space_gain": _number(row["cell_space_delta_seg_score"]),
                "realized_proxy_delta_s": row["receiver_realized_delta_d_seg"],
                "cell_upper_bound_per_byte": _number(row["seg_score_gain_per_selected_byte"]),
                "uint8_realizability": 0.0,
                "dual_tolerance_d2": 0.0,
                "lambda_d2": 0.0,
                "status": "RECEIVER_REALIZATION_OWED",
            }
        )
    rows.sort(key=lambda row: row["cell_upper_bound_per_byte"], reverse=True)
    return rows


def rank_scheduler_blocks(
    blocks: Sequence[Mapping[str, Any]],
    *,
    completed: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Rank the current dependency frontier.

    Dependency filtering enforces topology.  Within that frontier, byte-freeing
    precedes spending, coarse scales precede fine scales, then independent
    blocks use Gauss-Southwell ``|lambda| * validity_radius``.
    """

    done = set(completed)
    eligible: list[dict[str, Any]] = []
    for raw in blocks:
        row = dict(raw)
        deps = set(row.get("dependencies") or [])
        if row["block_id"] in done or not deps.issubset(done):
            continue
        row["gauss_southwell_validity"] = gauss_southwell_validity_score(
            _number(row.get("lambda_abs", 0.0)),
            _number(row.get("validity_radius", 0.0)),
        )
        eligible.append(row)
    eligible.sort(
        key=lambda row: (
            not bool(row.get("frees_bytes")),
            int(row.get("coarse_level", 0)),
            -row["gauss_southwell_validity"],
            str(row["block_id"]),
        )
    )
    for rank, row in enumerate(eligible, 1):
        row["rank"] = rank
    return eligible


def _scheduler(
    dv1: Mapping[str, Any],
    g4: Mapping[str, Any],
    blocks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    primitive = next(
        row for row in dv1["enriched_mdl_table"] if row["candidate"] == "persistent_level_set_ground_partition"
    )
    g4_top = g4["summary"]["top5_amortization_opportunities"][0]
    measured_block = max(blocks, key=lambda row: row["lambda_d2"])
    candidates = [
        {
            "block_id": "j_paint_dv1_persistent_ground",
            "dependencies": ["source_custody", "pair_site_lambda"],
            "coarse_level": 0,
            "frees_bytes": False,
            "lambda_abs": (
                _number(primitive["net_errors_closed_all_strata"]) / max(int(primitive["counted_bytes"]), 1)
            ),
            "validity_radius": 0.0,
            "lambda_status": "UPPER_BOUND_ONLY_UINT8_REALIZABILITY_OWED",
            "reason": "283-byte coarse primitive reaches 79.519% Road in cell space; J_paint is owed",
        },
        {
            "block_id": "j_paint_g4_movable_midband",
            "dependencies": ["source_custody", "pair_site_lambda"],
            "coarse_level": 1,
            "frees_bytes": False,
            "lambda_abs": _number(g4_top["seg_score_gain_per_selected_byte"]),
            "validity_radius": 0.0,
            "lambda_status": "UPPER_BOUND_ONLY_UINT8_REALIZABILITY_OWED",
            "reason": "highest measured g4 cell-space value per byte; receiver realization is null",
        },
        {
            "block_id": "r6_exact_receiver_rehearsal",
            "dependencies": [
                "source_custody",
                "pair_site_lambda",
                "j_paint_dv1_persistent_ground",
            ],
            "coarse_level": 2,
            "frees_bytes": False,
            "lambda_abs": 0.0,
            "validity_radius": 0.0,
            "lambda_status": "UNIDENTIFIABLE_UNTIL_J_PAINT",
            "reason": "R6 exact replay is downstream of receiver J_paint",
        },
        {
            "block_id": "ddm_iteration_curve_instrument",
            "dependencies": ["source_custody", "pair_site_lambda"],
            "coarse_level": 3,
            "frees_bytes": False,
            "lambda_abs": _number(measured_block["lambda_d2"]),
            "validity_radius": _number(measured_block["validity_radius"]),
            "lambda_status": "MEASURED_FAMILY_LOCAL",
            "reason": "replace invalid witness-era NCDE with DDM joint-recursion iteration curves",
        },
    ]
    completed = {"source_custody", "pair_site_lambda"}
    ranked = rank_scheduler_blocks(candidates, completed=completed)
    recommendation = ranked[0] if ranked else None
    return {
        "equation_id": SCHEDULER_EQUATION_ID,
        "order": [
            "dependency_topology",
            "freeing_before_spending",
            "coarse_to_fine",
            "gauss_southwell_abs_lambda_times_validity_radius",
        ],
        "completed": sorted(completed),
        "ranked_frontier": ranked,
        "next_block": recommendation,
        "note": (
            "The first two live rows have zero admitted lambda because uint8 realization is owed; "
            "their ordering is an instrumentation recommendation, not an actuation claim."
        ),
    }


def _legacy_duty_count() -> int | None:
    try:
        from tac.witness_dsl.activation_ledger import duty_to_measure_ranked

        return sum(1 for row in duty_to_measure_ranked() if row.get("in_duty_queue"))
    except Exception:
        return None


def _duties(scheduler: Mapping[str, Any]) -> dict[str, Any]:
    current = _legacy_duty_count()
    return {
        "legacy_authority_snapshot_rows_retained": LEGACY_AUTHORITY_OWED_ROWS,
        "current_legacy_rows_retained": current,
        "retention_status": (
            "AT_LEAST_115_RETAINED"
            if current is None or current >= LEGACY_AUTHORITY_OWED_ROWS
            else "RETENTION_REGRESSION"
        ),
        "legacy_disposition": "DOMINATED_STALE",
        "legacy_reason": (
            "witness-training telemetry and old run-local activations do not describe the live "
            "DDM block/resolution/pair/site recursion"
        ),
        "live_ranked": [
            {
                "rank": 1,
                "duty": "J_paint",
                "block_id": "j_paint_dv1_persistent_ground",
                "reason": scheduler["ranked_frontier"][0]["reason"],
            },
            {
                "rank": 2,
                "duty": "R6_rehearsal",
                "block_id": "r6_exact_receiver_rehearsal",
                "reason": "receiver-closed exact rehearsal after J_paint",
            },
            {
                "rank": 3,
                "duty": "DDM_iteration_curves",
                "block_id": "ddm_iteration_curve_instrument",
                "reason": "fit only on DDM recursion; witness NCDE r2=0.060 is invalid here",
            },
        ],
    }


# ── 2026-07-28 describe-line arc: evidence join + band-position SENSE law + refreshed duty ──
# Advisory SENSE/DECIDE extension (actuation NONE, score_claim False). Every row is
# content-hashed to a committed artifact so a fresh checkout reproduces it. This EXTENDS the
# organ's evidence surface (per the #247 de-orphan law) with the measured 07-28 arc rows; it
# never rebuilds the organ beside itself.
ARC_EVIDENCE_AXIS = "[macOS-CPU advisory]"  # coder-rate / frozen-scorer advisory; NOT a contest score
# The live base error rate the arc is descending: the latest exact n600 SegNet-argmax d_seg,
# read machine-readably from the committed CT1 verdict (a fraction of sites = the density the
# pp1 band lemma is defined over). fd1 (0.0702156745) and sc1/rp1 (0.070519) are sister witnesses.
LIVE_BASE_DSEG_RECEIPT = (
    ".omx/research/ddm_ct1_campaign_telemetry_encode_20260725T111500Z/r6_rehearsal_receipt.json"
)
# The tb1 burn ENDPOINT (the S-A live critical-path vehicle) — the CURRENT live base after the
# t3 lotto burn DESCENDED from the tr1 T2 full-confirm bases (0.0138) all the way to the endpoint.
# ng1 §2 row 10 (owed to co9): the digest's "corrections DEAD at every live parent" verdict was
# parented on the STALE pre-arc bases (W_joint 0.0705, tr1 0.0138/0.0141) — the burn endpoint
# 0.00389 IS the live base and it is INSIDE the rational band [rho_c, 1e-2]. Read machine-readably
# from the committed pfs1 D1 locked-evaluate.py receipt (real evaluator, real bytes; archive
# 624ffe57; the SegNet-distortion line of the pinned upstream evaluate.py report). The re-grade
# it triggers is DUE by band-entry BUT carries the QA03/QA04 white-jitter MEASURED-BREAK-EVEN prior
# (seg is a base-quality game; in-band != a promising correction lever).
LIVE_BURN_ENDPOINT_EVAL_RECEIPT = ".omx/research/ddm_pfs1_d1_eval_receipt_20260729.json"


@dataclass(frozen=True)
class ArcEvidenceSpec:
    finding_id: str
    artifact: str  # committed path relative to repo root
    headline: str  # the measured row (numbers)
    crux_status: str  # SETTLED | LIVE_CRUX | SOLVED | LAW | APPARATUS
    verdict_scope: str
    # co7 conditional-validity extension: typed precondition tags on negative/settled
    # verdict rows. Each tag: {precondition_id, kind, holds_when, invalidated_by}.
    # A verdict binds only while its preconditions hold; a substrate change that breaks
    # one surfaces a re-grade duty (never a silent stale verdict).
    preconditions: tuple[Mapping[str, str], ...] = ()


ARC_EVIDENCE_SPECS: tuple[ArcEvidenceSpec, ...] = (
    ArcEvidenceSpec(
        "fd1_zero_accept_window",
        ".omx/research/ddm_fd1_family_d_gn_description_engine_20260728.md",
        "6 family-d GN candidates, 0 accepted; slope 0.000%/step; realized n600 d_seg "
        "bit-identical 0.0702156745 (5/6); seal BLOCKED_ZERO_ACCEPT_WINDOW_CAPACITY_ROUTED; "
        "99.6% wall-clock is realized-acceptance pricing",
        "LIVE_CRUX",
        "fixed-capacity GN descent of the W_joint base is exhausted; realization moves to "
        "the fd1 capacity/parametrization ladder (Rung1 grow shared/cross-pair DOF + #383-dual "
        "pose-null; Rung2 token-grid + partition->pixel renderer <=64KB scorer-in-loop)",
        (
            {
                "precondition_id": "fd1_fixed_capacity_wjoint_parametrization",
                "kind": "parametrization",
                "holds_when": "the fixed-capacity W_joint family-d parametrization is the live descent vehicle",
                "invalidated_by": (
                    "a committed successor materializes the capacity ladder (tr1 renderer) "
                    "or re-adjudicates the zero-accept fork (fd2)"
                ),
            },
        ),
    ),
    ArcEvidenceSpec(
        "fd1_box_solve_s0_hold",
        ".omx/research/ddm_fd1_family_d_gn_description_engine_20260728.md",
        "S0 box-solve C1/C0 d_seg 1.077x (1.2492e-3 vs 1.1600e-3); cells-hold flip 3.757e-4 "
        "~= 3.630e-4 GT; margin gap 165x box / 166x GT; HOLD hardened",
        "SETTLED",
        "the cell-space box partition reproduces baseline and holds; not the open crux",
    ),
    ArcEvidenceSpec(
        "pp1_direct_partition_price",
        ".omx/research/ddm_pp1_direct_partition_pricing_20260728.md",
        "direct partition 173,616 B lossless (KT context-arith o8+prev5); composed explicit "
        "route S~0.189 > 0.172 bar; >=350KB falsifier NOT reached (priced, NOT dead)",
        "SETTLED",
        "explicit-partition compression priced; does not beat PR130's implicit ~177KB leg",
        (
            {
                "precondition_id": "pp1_wjoint_partition_source",
                "kind": "parent",
                "holds_when": (
                    "the partition source is the q1/W_joint cell partition and the "
                    "competitive bar is 0.172"
                ),
                "invalidated_by": "a cheaper partition source (tr1 token stream) or a bar move",
            },
        ),
    ),
    ArcEvidenceSpec(
        "pp1_band_lemma",
        ".omx/research/ddm_pp1_band_lemma_receipt_20260728.json",
        "correction-stream position band: water 1.2731 B/flip; rho_c 5.015e-4 measured coherent; "
        "rho_u 8.59e-4 derived; rational only for base error rho in ~[5e-4, 1e-2]",
        "LAW",
        "registered ddm_pp1_correction_stream_position_band_v1; the SENSE law the band-position "
        "modulator consumes",
    ),
    ArcEvidenceSpec(
        "rp1_cells_hold",
        ".omx/research/ddm_rp1_rangeA_cell_realized_probe_20260728.md",
        "flipped/held pre-round SegNet margin 166.5x (0.0337 vs 5.6136); C1 d_seg 3.63e-4 "
        "= 2.39x q1; C0 0 flips; CELLS HOLD",
        "SETTLED",
        "flips concentrate at near-zero margin; sc1-far = engine capacity, not a formulation break",
    ),
    ArcEvidenceSpec(
        "sp1_support_race",
        ".omx/research/ddm_sp1_contour_support_coder_20260728.md",
        "support contour 444,394 B vs LZMA 421,366 B @ rho=0.864% (contour 5.5% worse, 3.1x above "
        "the phantom 142KB floor); min lossy S=0.27999 alone exceeds 0.172 bar; FLOOR_DEAD",
        "SETTLED",
        "explicit-residual copy-base support family CLOSED; corrections at this base explode",
        (
            {
                "precondition_id": "sp1_copy_base_parent",
                "kind": "parent",
                "holds_when": "the W_joint copy-base at rho~0.86% is the live correction parent",
                "invalidated_by": "parent switch to the tr1 renderer output",
            },
            {
                "precondition_id": "sp1_band_regime_explode",
                "kind": "band_regime",
                "holds_when": "every live parent base sits above the band upper (regime explode)",
                "invalidated_by": "a live parent base entering the rational band [rho_c, 1e-2]",
            },
        ),
    ),
    ArcEvidenceSpec(
        "sc1_ep_rank1_pose",
        ".omx/research/ddm_sc1_seeded_scene_carrier_20260728.md",
        "e_p SVD energy frac[0]=0.9986 (rank-1); AR-int5 residual 2,039 B (~2KB, ~9% of PR130's "
        "23,054 B); pose leg feasibility-bounded ~2KB, NOT binding (DC-mean stored separately, ch1)",
        "SOLVED",
        "pose carrier feasibility-bounded ~2KB; not the binding constraint",
    ),
    ArcEvidenceSpec(
        "ch1_confound_pass",
        ".omx/research/ddm_ch1_recursive_confound_pass_20260728.md",
        "15 rows checked: 7 CLEAN / 8 SEAM-NAMED / 0 CONFIRMED-CONFOUND; S1 plumbing-reopener "
        "DECISIVELY REFUTED (pose deltas realized-measured)",
        "APPARATUS",
        "apparatus-validity metadata: the 07-28 stack carries zero confirmed confounds",
    ),
    # ── co7 (2026-07-28 late arc): fd2 disambiguation + tb1 renderer build + eg1 endgame ──
    ArcEvidenceSpec(
        "fd2_zero_accept_disambiguation",
        ".omx/research/ddm_fd2_posenull_gn_disambiguation_20260728.md",
        "typed fork verdict SEG_REALIZATION_GAP_AT_UINT8_DOMINANT: pose-veto cure REFUTED as "
        "the binding mechanism (0/6 seg-only accepts, MEASURED); realization-gap dominant, "
        "directionally-unfaithful at flip amplitude (per-block)",
        "SETTLED",
        "adjudicates the fd1 zero-accept fork; the crux transfers to the trained "
        "partition->pixel renderer line (tr1/tb1), not to a pose-veto cure",
    ),
    ArcEvidenceSpec(
        "tb1_t2_race_verdict",
        ".omx/research/ddm_tb1_renderer_build_20260728.md",
        "pre-registered n600 A2 race: LOTTO Pareto-dominates (full-confirm d_seg 0.013833 @ "
        "534,597 B vs plain 0.014088 @ 549,927 B; -1.8%/-2.8%; renderer stream 3,284 vs "
        "20,214 B = 6.2x structural); Lane-Betti-0 caveat recorded NOT promoted (plain leads "
        "nucleation 264 vs 164, erased 906 vs 916); plain checkpoint retained as fallback",
        "SETTLED",
        "INSTANCE — single seed, no noise floor; pre-registered rule applied as written; the "
        "Lane-pool lever race fires FIRST in the burn plan",
    ),
    ArcEvidenceSpec(
        "tb1_t3_sealed_ticket",
        ".omx/research/configs/ddm_tb1_t3_long_burn_lotto_20260728.json",
        "T3 long-burn LOTTO ticket SEALED READY_TO_FIRE_UNDER_STANDING_GO (fires from MAIN "
        "only): n600 x 400 ep / 480-min resumable windows, gate_every 10 full-confirm; "
        "ticket_hash 007d8eacf402c4fe..., code 17166ee9c4",
        "APPARATUS",
        "burn-fire is a heavy launch = operator-GO (CONTAINMENT); the organ surfaces "
        "readiness only",
    ),
    ArcEvidenceSpec(
        "eg1_e1_byteclose_rehearsal",
        ".omx/research/ddm_eg1_tr1_rehearsal_20260728.json",
        "TR1 four-section byte-close rehearsed on the stale T2 checkpoint: canonical packet "
        "504,249 B (tokens 499,587 + renderer 3,341 + selector 535 + pose stub 83, "
        "Brotli-Q11); exact parse/re-emit + closed cursor + trailing-byte refusal; NumPy/MLX "
        "camera-byte parity 0.9999793579 (>0.9997 gate); locked evaluate.sh interface pass "
        "PASS_INTERFACE_ONLY_NONCOMPARABLE",
        "APPARATUS",
        "the R6 exact-eval on-ramp is BUILT; stale-checkpoint rehearsal — no candidate, no "
        "score, partial outputs non-comparable",
    ),
    ArcEvidenceSpec(
        "eg1_e2_stop_policy",
        ".omx/research/ddm_eg1_policy_arithmetic_20260728.json",
        "typed stop/continue/handoff policy + exact corner economics: strict byte ceilings "
        "190,334 B (0.172) / 157,294 B (0.15) at (d_seg 3e-4, d_pose 2.33e-5); TR1-A 149 KB "
        "-> S 0.144177 is the only sub-bar corner; QDBS cost bound 5.76-7.00 h, gain null",
        "APPARATUS",
        "advisory policy; a handoff needs a same-parent conservative score-gain-rate "
        "dominance quote — never a trajectory-detector hit alone",
    ),
    ArcEvidenceSpec(
        "eg1_e3_qdbs_finisher",
        ".omx/research/ddm_eg1_qdbs_rehearsal_20260728.json",
        "FD2-QDBS hard terminal finisher rehearsed: 48 candidates + shared base on the stale "
        "368-coordinate fd2 checkpoint; best test-oracle delta -5.4347826087e-7 (mechanism "
        "canary only); production custody REQUIRES_EXTERNAL_GOVERNOR",
        "APPARATUS",
        "finisher BUILT; no production compiler/receiver/frozen-scorer/n600 payoff exercised",
    ),
    ArcEvidenceSpec(
        "eg1_e3_pose_finisher",
        ".omx/research/ddm_eg1_pose_gn_rehearsal_20260728.json",
        "terminal six-equation pose GN rehearsed from frozen composed uint8 frames: one-pair "
        "advisory PoseNet MSE 5.2816893588 -> 1.3504594700 with frame 1 byte-identical; "
        "107-byte terminal section; not a candidate payoff (parent constant omitted)",
        "APPARATUS",
        "pose-leg finisher BUILT; production gating self-attested -> external governor "
        "required",
    ),
)


def _arc_evidence_rows(repo_root: Path) -> list[dict[str, Any]]:
    """Content-hashed advisory rows for the committed 2026-07-28 describe-line arc."""

    rows: list[dict[str, Any]] = []
    for spec in ARC_EVIDENCE_SPECS:
        path = repo_root / spec.artifact
        if not path.is_file():
            rows.append(
                {
                    "finding_id": spec.finding_id,
                    "artifact": spec.artifact,
                    "available": False,
                    "reason": "COMMITTED_ARTIFACT_ABSENT",
                    "crux_status": spec.crux_status,
                    "actuation": "NONE",
                    "score_claim": False,
                }
            )
            continue
        row = {
            "finding_id": spec.finding_id,
            "artifact": spec.artifact,
            "available": True,
            "sha256": _sha256(path),
            "headline": spec.headline,
            "crux_status": spec.crux_status,
            "verdict_scope": spec.verdict_scope,
            "evidence_axis": ARC_EVIDENCE_AXIS,
            "actuation": "NONE",
            "score_claim": False,
        }
        if spec.preconditions:
            row["preconditions"] = [dict(pre) for pre in spec.preconditions]
        rows.append(row)
    return rows


def _band_position(repo_root: Path) -> dict[str, Any]:
    """Wire the REGISTERED band lemma into the organ: place the live base in the rho-band.

    Reads the live base error rate (d_seg = fraction of flipped sites) from the committed CT1
    exact-n600 verdict and evaluates the canonical ``position_cost_band`` equation. The regime
    is the correction-class duty modulator: below rho_c -> concede (correction pointless),
    in-band -> rational, above the band upper -> support cost explodes (lower the base first).
    """

    from tac.canonical_equations.ddm_pp1_correction_stream_position_band_20260728 import (
        BAND_UPPER_DENSITY,
        EQUATION_ID,
        MEASURED_COHERENT_CROSSING_DENSITY,
        WATER_B_PER_FLIP,
        position_cost_band,
    )

    path = repo_root / LIVE_BASE_DSEG_RECEIPT
    if not path.is_file():
        return {
            "available": False,
            "reason": "LIVE_BASE_DSEG_RECEIPT_ABSENT",
            "equation_id": EQUATION_ID,
            "actuation": "NONE",
            "score_claim": False,
        }
    payload = _load_json(path)
    rows = ((payload.get("observability_digest") or {}).get("rows")) or []
    verdict = next((r for r in rows if r.get("row_id") == "latest_exact_n600_verdict"), None)
    if not isinstance(verdict, Mapping) or "d_seg" not in verdict:
        return {
            "available": False,
            "reason": "LIVE_BASE_EXACT_N600_VERDICT_ABSENT",
            "equation_id": EQUATION_ID,
            "source": {"path": LIVE_BASE_DSEG_RECEIPT, "sha256": _sha256(path)},
            "actuation": "NONE",
            "score_claim": False,
        }
    base_d_seg = _number(verdict["d_seg"])
    if not (0.0 < base_d_seg < 1.0):
        # A perfect (0) or degenerate (>=1) base is outside the correction-band domain; do not
        # crash the DDM-LIVE section — report it gracefully so the digest stays fail-open.
        return {
            "available": False,
            "reason": f"LIVE_BASE_DSEG_OUT_OF_BAND_DOMAIN ({base_d_seg:g})",
            "base_d_seg": base_d_seg,
            "equation_id": EQUATION_ID,
            "source": {"path": LIVE_BASE_DSEG_RECEIPT, "sha256": _sha256(path)},
            "actuation": "NONE",
            "score_claim": False,
        }
    band = position_cost_band(base_d_seg)
    regime = str(band["regime"])
    if regime == "concede":
        correction_regime = "CONCEDE_SHIP_NO_CORRECTION_STREAM_BELOW_RHO_C"
        correction_multiplier = 0.0
    elif regime == "correct":
        correction_regime = "CORRECTION_RATIONAL_IN_BAND"
        correction_multiplier = 1.0
    else:  # explode
        correction_regime = "ABOVE_BAND_SUPPORT_COST_EXPLODES_LOWER_BASE_FIRST"
        correction_multiplier = 0.0
    return {
        "available": True,
        "equation_id": EQUATION_ID,
        "base_d_seg": base_d_seg,
        "band": dict(band),
        "regime": regime,
        "rho_c": MEASURED_COHERENT_CROSSING_DENSITY,
        "band_upper": BAND_UPPER_DENSITY,
        "water_b_per_flip": WATER_B_PER_FLIP,
        "correction_class_regime": correction_regime,
        "correction_duty_multiplier": correction_multiplier,
        "source": {
            "path": LIVE_BASE_DSEG_RECEIPT,
            "sha256": _sha256(path),
            "row_id": "latest_exact_n600_verdict",
            "verdict_source_path": verdict.get("source_path"),
            "axis": verdict.get("evidence_axis", ARC_EVIDENCE_AXIS),
        },
        "evidence_axis": ARC_EVIDENCE_AXIS,
        "actuation": "NONE",
        "score_claim": False,
    }


def _refreshed_duties(
    legacy_duties: Mapping[str, Any],
    arc_rows: Sequence[Mapping[str, Any]],
    band_position: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the duty head against the live 07-28 arc state (DERIVED, not hand-ordered).

    Elimination argument, each leg cited to a measured arc row:
    (1) band-position places the base above the rational correction band -> correction-class
        duties score 0 (sp1's 444KB support wall is the measured teeth);
    (2) fd1's zero-accept window seals fixed-capacity GN descent -> that duty is exhausted;
    (3) rp1 cells-hold (166x) + fd1 S0 box-hold settle the cell-space realization (J_paint) premise.
    The surviving lever is the capacity/parametrization ladder (fd1 Rung1/Rung2), which therefore
    becomes the refreshed head, displacing the pre-arc J_paint cell-space realization.
    """

    index = {row["finding_id"]: row for row in arc_rows if row.get("available")}
    zero_accept_sealed = "fd1_zero_accept_window" in index
    cells_hold = "rp1_cells_hold" in index
    box_hold = "fd1_box_solve_s0_hold" in index
    support_dead = "sp1_support_race" in index
    # co7: the ladder head is MATERIALIZED once tb1 lands (Rung2 built + raced + sealed);
    # the duty head then hands off to the committed endgame chain (see duties_endgame).
    ladder_materialized = "tb1_t3_sealed_ticket" in index and "tb1_t2_race_verdict" in index
    band_regime = str(band_position.get("regime")) if band_position.get("available") else None
    correction_multiplier = float(band_position.get("correction_duty_multiplier", 1.0))

    # Derived priority per duty kind (higher = more live).
    candidates: list[dict[str, Any]] = []

    if ladder_materialized:
        candidates.append(
            {
                "duty": "ENDGAME_CHAIN_HANDOFF",
                "kind": "endgame_chain",
                "priority": 3.0,
                "basis": (
                    "DERIVED: tb1 materialized the fd1 capacity ladder (Rung2 renderer "
                    "raced + T3 sealed) and eg1 built the byte-close/policy/terminal chain "
                    "=> the duty head moves to the endgame chain (duties_endgame): "
                    "B-verdict watch > burn-fire > first-gates > T1-validity > byte-close"
                ),
                "cites": [
                    fid
                    for fid in (
                        "tb1_t3_sealed_ticket",
                        "tb1_t2_race_verdict",
                        "eg1_e1_byteclose_rehearsal",
                        "fd2_zero_accept_disambiguation",
                    )
                    if fid in index
                ],
                "actuation": "NONE",
            }
        )
    ladder_live = bool(band_regime == "explode" and zero_accept_sealed and not ladder_materialized)
    candidates.append(
        {
            "duty": "FD1_REALIZATION_LADDER_MATERIALIZATION",
            "kind": "capacity_ladder",
            "priority": 0.25 if ladder_materialized else (2.0 if ladder_live else 1.0),
            "basis": (
                "MATERIALIZED by tb1 (kept for lineage; see ENDGAME_CHAIN_HANDOFF)"
                if ladder_materialized
                else (
                    "DERIVED: base above-band (correction explodes) + fd1 zero-accept "
                    "(fixed-capacity descent exhausted) => lower the base by growing capacity/"
                    "re-parametrizing (fd1 Rung1 shared/cross-pair DOF + #383-dual pose-null; "
                    "Rung2 token-grid partition->pixel renderer <=64KB scorer-in-loop)"
                )
            ),
            "cites": [
                fid
                for fid in ("pp1_band_lemma", "fd1_zero_accept_window", "rp1_cells_hold")
                if fid in index
            ],
            "actuation": "NONE",
        }
    )
    candidates.append(
        {
            "duty": "R6_rehearsal",
            "kind": "receiver_closed_rehearsal",
            "priority": 1.0,
            "basis": "receiver-closed exact rehearsal, downstream of materialization",
            "cites": [],
            "actuation": "NONE",
        }
    )
    candidates.append(
        {
            "duty": "DDM_iteration_curves",
            "kind": "instrument",
            "priority": 0.5,
            "basis": "fit only on DDM recursion; witness NCDE r2=0.060 invalid here",
            "cites": [],
            "actuation": "NONE",
        }
    )
    candidates.sort(key=lambda row: (-row["priority"], row["duty"]))
    for rank, row in enumerate(candidates, 1):
        row["rank"] = rank

    demoted: list[dict[str, Any]] = []
    if cells_hold and box_hold:
        demoted.append(
            {
                "duty": "J_paint (cell-space receiver realization)",
                "prior_rank": 1,
                "disposition": "SETTLED_CELL_SPACE_SUBSUMED_INTO_MATERIALIZATION",
                "basis": "rp1 cells-hold 166x + fd1 S0 box-hold settle the cell-space premise; "
                "the open crux is materializing/lowering the W_joint base, not cell-space reach",
                "cites": ["rp1_cells_hold", "fd1_box_solve_s0_hold"],
            }
        )
    if support_dead or correction_multiplier == 0.0:
        demoted.append(
            {
                "duty": "correction/support/explicit-residual class",
                "prior_rank": None,
                "disposition": (
                    "BAND_DEAD_BELOW_RHO_C_CONCEDE"
                    if band_regime == "concede"
                    else "BAND_DEAD_ABOVE_BAND_EXPLODE"
                ),
                "basis": (
                    f"band-position regime={band_regime}; correction_duty_multiplier="
                    f"{correction_multiplier:g}"
                    + ("; sp1 FLOOR_DEAD 444KB support wall" if support_dead else "")
                ),
                "cites": [fid for fid in ("pp1_band_lemma", "sp1_support_race") if fid in index],
            }
        )
    if ladder_materialized:
        demoted.append(
            {
                "duty": "FD1_REALIZATION_LADDER_MATERIALIZATION (as the head)",
                "prior_rank": 1,
                "disposition": "MATERIALIZED_BY_TB1_SEALED_TICKET",
                "basis": (
                    "tb1 raced Rung2 (tr1 token-grid + partition->pixel renderer) to a "
                    "pre-registered n600 Pareto verdict and sealed the T3 long-burn ticket; "
                    "fd2 adjudicated the fork (seg-realization-gap, not pose veto)"
                ),
                "cites": [
                    fid
                    for fid in (
                        "tb1_t2_race_verdict",
                        "tb1_t3_sealed_ticket",
                        "fd2_zero_accept_disambiguation",
                    )
                    if fid in index
                ],
            }
        )

    return {
        "schema": "ddm_live_costate_refreshed_duties.v1",
        "derivation": "band-position + 07-28 arc-evidence elimination; not hand-ordered",
        "band_regime": band_regime,
        "live_ranked": candidates,
        "demoted": demoted,
        "superseded_legacy_head": (
            legacy_duties["live_ranked"][0]["duty"] if legacy_duties.get("live_ranked") else None
        ),
        "legacy_live_ranked": list(legacy_duties.get("live_ranked") or []),
        "actuation": "NONE",
        "score_claim": False,
    }


# ── co7 (2026-07-28): pending producers · per-parent band placement · SENSE laws ·
# conditional-validity trigger · the endgame duty chain. Advisory SENSE/DECIDE only
# (actuation NONE, score_claim False); every number cites a committed content-hashed
# artifact; in-flight parallel arms are registered PENDING, never folded early.
PRECONDITION_SCHEMA = "ddm_costate_verdict_precondition.v1"
T3_SEALED_TICKET = ".omx/research/configs/ddm_tb1_t3_long_burn_lotto_20260728.json"


@dataclass(frozen=True)
class PendingProducerSpec:
    """A named in-flight producer whose numbers stay uncounted until committed.

    ``consumed_by`` names the organ round that folded the committed artifact into
    typed rows (co8: rv1); a committed-but-unfolded producer keeps the co7
    FOLD_ON_NEXT_ROUND status so the queue never silently absorbs a landing.
    """

    name: str
    glob: str  # scanned under .omx/research/
    expectation: str
    named_gate: str
    consumed_by: str | None = None


PENDING_PRODUCER_SPECS: tuple[PendingProducerSpec, ...] = (
    PendingProducerSpec(
        "lv1_token_stack_prices",
        "ddm_lv1_*",
        "token-stack pricing on the tb1 token stream (~531 KB): factorization/truncation "
        "ladder prices are admissible only WITH realized-validity rows; pools-law: truncation "
        "vs quantization candidates COMPETE, never sum. Charter-cited prices stay uncounted "
        "until the committed receipt lands (NO-FAKE #4).",
        "LV1_COMMITTED_RECEIPT_WITH_REALIZED_VALIDITY_ROWS",
    ),
    PendingProducerSpec(
        "rv1_conditional_validity_table",
        "ddm_rv1_*",
        "the full historical conditional-validity/precondition table sweep; this organ seeds "
        "arc-local precondition rows ONLY and consumes rv1's committed table when it lands "
        "(no duplicate sweep).",
        "RV1_COMMITTED_CONDITIONAL_VALIDITY_TABLE",
        consumed_by="co8_rv1_conditional_validity_table",
    ),
)


def _pending_producers(repo_root: Path) -> list[dict[str, Any]]:
    """Registered in-flight producers: tracked queue rows, never silent absences."""

    research = repo_root / ".omx" / "research"
    rows: list[dict[str, Any]] = []
    for spec in PENDING_PRODUCER_SPECS:
        matches = [p for p in sorted(research.glob(spec.glob)) if p.exists()]
        row: dict[str, Any] = {
            "producer": spec.name,
            "glob": spec.glob,
            "expectation": spec.expectation,
            "named_gate": spec.named_gate,
            "available": bool(matches),
            "actuation": "NONE",
            "score_claim": False,
        }
        if matches:
            latest = matches[-1]
            target = latest if latest.is_file() else next(
                (p for p in sorted(latest.rglob("*")) if p.is_file()), None
            )
            row["path"] = str(latest.relative_to(repo_root))
            if target is not None:
                row["sha256"] = _sha256(target)
            row["status"] = (
                f"COMMITTED_CONSUMED_BY_{spec.consumed_by.upper()}"
                if spec.consumed_by
                else "COMMITTED_ARTIFACT_PRESENT_FOLD_ON_NEXT_ROUND"
            )
        else:
            row["reason"] = "PENDING_COMMITTED_PRODUCER"
        rows.append(row)
    return rows


# ── co8 (2026-07-28): consume the COMMITTED rv1 conditional-validity table + the pn1
# pantheon-of-pantheons rows (VOI ranking · nu-pivot decision node · granularity-ladder law +
# ARMED race · S1 dress-rehearsal duties) + the allocator-law duty waterfill. Advisory
# SENSE/DECIDE only (actuation NONE, score_claim False); every row content-hashed to the
# committed memo it is folded from; ASSUMED priors stay labeled ASSUMED (NO-FAKE #8).
RV1_TABLE_MEMO = ".omx/research/ddm_rv1_conditional_validity_regrade_20260728.md"
PN1_MEMO = ".omx/research/ddm_pn1_pantheon_of_pantheons_completion_20260728.md"
# The single consolidated deferral queue ledger (gc7r 07-29; the organ's DECLARED
# consumed-evidence + recall source per its own frontmatter). Dated glob so the date rolling
# forward selects the newest ledger without a code change. co9 registers it as consumed evidence
# and scans it for the OWNERSHIP-ON-GATE-OPEN SENSE surface.
DEFERRAL_LEDGER_GLOB = "ddm_deferral_queue_ledger_*.md"


@dataclass(frozen=True)
class Rv1ReactivationSpec:
    """One rv1 re-grade row: a standing negative whose killing precondition changed."""

    row_id: str
    rank: int  # rv1 §1 leverage x cheapness ranking
    negatives: tuple[str, ...]  # the committed verdicts re-graded (quoted in the memo)
    precondition_changed: str  # YES | YES_CONDITIONAL | PARTIAL_RACE | PARTIAL
    measurement: str  # the NAMED reactivation measurement — nothing reactivates without it
    status: str  # rv1's OWN schedulability field, frozen at 2026-07-28 (never a live state)
    consumer: str
    # ── rx1 (2026-07-31): THE WRITE PATH. `reactivated` shipped as a hardcoded `False`
    # literal inside the fold's comprehension with no writer anywhere in the module, so a
    # row could never record its own discharge — a ledger field that structurally cannot
    # log the thing it names (sister of the default-off orphan class + NO-FAKE #2, and the
    # co8 test froze it by asserting the constant). It is now RESOLVED FROM COMMITTED
    # EVIDENCE ON DISK and fails closed to False when no artifact matches.
    result_glob: tuple[str, ...]  # committed RESULT artifact(s) == the measurement LANDED
    landed_disposition: str  # what the landed measurement FOUND (never "the lever paid")
    open_state: str  # duty_state when no result artifact exists
    open_reason: str  # one line: why it is still open + what would close it
    charter_glob: tuple[str, ...] = ()  # chartered-and-live pointer (NOT a result)


RV1_REACTIVATION_SPECS: tuple[Rv1ReactivationSpec, ...] = (
    Rv1ReactivationSpec(
        "R1_terminal_band_discrete_search",
        1,
        (
            "eu1_dr1_composed_candidate_admissible_false",
            "v19c_correction_saturation_instance",
            "mc_finisher_400_execution_enabled_false",
        ),
        "YES",
        "EU1-FD2-QDBS-N600 exactly as eu1 pre-specified (<=48 full-n600 verdicts + base) on "
        "the burn's final lotto checkpoint TOKEN lattice; #400 mc_finisher diagonal adapted "
        "to tr1 token/renderer tensors; unblock DR1's ordered pairwise redundancy audit",
        "POST_BURN",
        "eg1 finishers / post-burn wave",
        result_glob=("ddm_pb1_postburn_completion_2*.md",),
        landed_disposition=(
            "LANDED 2026-07-29 (pb1 §2 QDBS = 49 evals, honest-axis mode, commit 838b5adfbc; "
            "#400 diagonal EXPLICIT = §4 renderer leg + §5 dxi pose-polish leg). pb1's own "
            "owed-table row records it DONE. CAVEAT travels with it: the 0.05-0.07 S prior "
            "was witness-vehicle/foreign-parent, NOT a same-parent quote. Sibling still open: "
            "the FULL-POPULATION GN/CG seg solve (orphan QA03) never ran"
        ),
        open_state="OPEN_POST_BURN",
        open_reason="needs the burn endpoint + a governed n600 verdict budget",
    ),
    Rv1ReactivationSpec(
        "R7_token_stream_coder_race",
        2,
        (
            "ab1_rate_axis_dead_misscoped",
            "ab3_ac_coder_dominated_int8_weights",
            "ab4_temporal_delta_hurts_int8_weights",
            "bb12_hyperprior_falsified_pr101_symbols",
        ),
        "YES",
        "$0 coder race on the T2 lotto token stream: brotli/LZMA baselines vs pp1 "
        "context-arith (o8 spatial + prev-N temporal KT) vs a small learned prior; "
        "falsifier: generic ~= context ~= iid floor => the negative honestly re-scopes "
        "onto the token object",
        "MEASURABLE_NOW_T2_DUMPS",
        "burn rate axis (entropy-coded prior + boundary-gated c + (D,c,levels) waterfill)",
        result_glob=("ddm_r7_token_coder_race_2*.md",),
        landed_disposition=(
            "LANDED 2026-07-29 as a complete non-additive race (14 token-entropy arms incl. "
            "KT-prev1/CTW/rANS/Bayes-mix vs Brotli-Q11/LZMA/Huffman, every admitted row a "
            "materialized R7PL frame with exact parse-back). The stale coder negatives are "
            "CLEARED off the token object, but the race did NOT hand back free bytes at the "
            "endpoint: solve-project endpoint winner = SMEVR 557,238 B -> 562,174 B composed "
            "(zero-init T2 winner Bayes-base+Brotli-delta 360,743 B is a DIFFERENT lineage and "
            "does not transfer). Deficit +371,840 B to the 0.172 ceiling. "
            "OWED_EG1_INTEGRATION_NOT_AN_ARCHIVE_ROW"
        ),
        open_state="OPEN_MEASURABLE_NOW",
        open_reason="T2 token dumps exist; $0 CPU coder race, no scorer slot needed",
    ),
    Rv1ReactivationSpec(
        "R4_token_granularity_correction_probe",
        3,
        (
            "n51_defer_pending_new_carrier",
            "multiscale_defer_pending_decoder_axis",
            "g4_no_go_unchanged_literal_scope",
        ),
        "YES",
        "$0 single-token perturbation probe on the T2 lotto checkpoint: realized flip "
        "footprint (36-pair gate + full-confirm), collateral ratio vs #51's 2,823/467, "
        "B/flip of a token-coordinate entry vs the 1.2731 water",
        "MEASURABLE_NOW_T2_CHECKPOINTS",
        "feeds R1 QDBS proposal design + R2 pricing + the token-stream entropy race; "
        "doubles as granularity-ladder rung-2 instrument",
        result_glob=("ddm_gr1_granularity_rerace_2*.md",),
        landed_disposition=(
            "LANDED 2026-07-30 as gr1 — a SUPERSET of the named probe (archive-faithful "
            "re-quant -> real SMEVR bytes + realized n600 d_seg through the frozen CPU SegNet; "
            "gates PASS, baseline injection 0.0038892 vs evaluate.py 0.00389011, delta 1.9e-6). "
            "Its OWN pre-registered GO test FAILED at token granularity: every candidate worse "
            "on realized seg+rate, B/flip 0.04-0.51 ALL below the 1.273 water -> token-granular "
            "correction STRICTLY DOMINATED (scope INSTANCE/FORMULATION). But the same probe "
            "found the UNIT: the cell, not the token (SMEVR conditions on per-cell temporal "
            "mode). cell_drop50 = 359,221 B @ realized n600 d_seg 0.004310 -> seg+rate 0.6702 = "
            "-0.098 vs the 0.7685 reference, byte-closed a6398e44. That base is CONSUMED by the "
            "v4b/v4c/v4d composed candidate. Also overturned QA11 and dominated QA07"
        ),
        open_state="OPEN_MEASURABLE_NOW",
        open_reason="T2 lotto checkpoints exist; $0 perturbation probe, no new training",
    ),
    Rv1ReactivationSpec(
        "R2_correction_stream_band_repriced",
        4,
        (
            "n72_lever_d_defer_not_kill",
            "n110_lever_d_no_go_with_reactivation",
            "n280_flip_coder_no_go_n600",
            "finishing_kit_no_go_at_convergence",
        ),
        "YES_CONDITIONAL",
        "the moment a burn checkpoint's realized d_seg enters [5e-4, 1e-2]: price the REAL "
        "correction stream (fc1 label coder + coherent support coder) on its residual flip "
        "masks; GO iff measured coded B/err < 1.2731 AND composed S improves; native "
        "<=~5e-4 ships NO stream",
        "BAND_ENTRY_ARMED",
        "byte-close composition of the burn candidate (E4/WS1 exporter chain)",
        result_glob=("ddm_ea1_einsteinian_negative_audit_2*.md",),
        landed_disposition=(
            "TRIGGER FIRED then PRICED NO-GO. The burn endpoint realized d_seg 0.0038892 sits "
            "INSIDE the armed [5e-4, 1e-2] band, so the conditional opened exactly as "
            "pre-specified; the stream was then priced and REFUSED on its own GO test "
            "(co9: dS_seg -0.001582 = 1.15% of the -0.138 ceiling at 1.45 B/flip ~= the 1.27 "
            "water). ea1 generalizes it: at an in-band base the seg residual is WHITE, and "
            "post-hoc correction streams are now measured non-paying at BOTH base regimes "
            "(verdict_scope: FORMULATION -- post-hoc streams x this vehicle class). "
            "In-band != a promising lever; the seg descent lever is the burn, not a stream"
        ),
        open_state="OPEN_BAND_ENTRY_ARMED",
        open_reason="fires the moment a burn checkpoint's realized d_seg enters [5e-4, 1e-2]",
    ),
    Rv1ReactivationSpec(
        "R6_lane_channel_intraining_entrants",
        5,
        (
            "c14_render_band_posthoc_misscoped_high",
            "c15_dash_comb_posthoc_net_negative",
            "c12_lane_prior_dead_gate_tainted",
        ),
        "YES",
        "enter the three as RACED entrants in the sealed FIRST burn item (Lane pool race): "
        "band-ACTIVE-in-training arm + in-training dash-comb arm + fixed-gate lane-prior "
        "arm; readout = realized Lane Betti-0 nucleation + per-class d_seg at the A1 gates",
        "BURN_WINDOW_RACE_FIRST_ITEM",
        "burn Lane-pool race (plain checkpoint retained as nucleation fallback)",
        result_glob=("ddm_b4s_burn4_endpoint_*.md", "ddm_b4s_burn4_result_*.md"),
        landed_disposition=(
            "burn-4 endpoint measured: R6 GO iff endpoint n600 d_seg < the 0.00426407708 "
            "ep641 control, else R6 closes at INSTANCE"
        ),
        open_state="OPEN_IN_FLIGHT_CHARTERED",
        open_reason=(
            "the named race did NOT run in its own burn window: the sealed tb1 ticket fired a "
            "SINGLE arm at fixed --class-weight-lane 1.0, and the Lane-pool race lived only in "
            "the ticket's adjudication CAVEAT, never in its levers -- so none of the three "
            "entrants (band-ACTIVE / in-training dash-comb / fixed-gate lane-prior) ever "
            "entered. Independently re-found by fh1 (A6: 'DSL lever EXISTS, default 1.0 = "
            "never-fired') and re-chartered as the burn-4 S1 fire (class_weight_lane 1.0->1.3). "
            "Closes on the burn-4 endpoint n600 row"
        ),
        charter_glob=("ddm_b4s_burn4_charter_*.md",),
    ),
    Rv1ReactivationSpec(
        "R8_solve_init_tokens_distill_shape",
        6,
        # rv1 counts R8 as ONE parked family (KD/distill); the pose-tube wall is the
        # PRECONDITION that parked it (does not transfer: pose TERMINAL on tr1), not a
        # separately re-graded negative.
        ("n74_kd_family_parked_continue_pending_corrected",),
        "PARTIAL",
        "A/B zero-init vs solve-INIT tokens, same seed/schedule (the tb1 s14 owed item); "
        "teacher = the exact-solve object, never a banned-lineage teacher",
        "BURN_WINDOW_RACE",
        "burn config (lv1-B slot; pn1 S5 solve-ANCHOR rides as a third arm)",
        result_glob=("ddm_sc2_schedule_optimality_convocation_2*.md",),
        landed_disposition=(
            "LANDED 2026-07-28 (sc2 row 14) and ADOPTED -- the campaign's ONE measured init "
            "lever. The named A/B ran at matched epoch on n600 full-confirm: "
            "token_init_mode=solve_project 0.009839 vs zero-init 0.013833 = -28.9%. Teacher = "
            "the exact-solve object (no banned-lineage teacher). Now ON in every current config "
            "including the bc1 QA24 re-burn. v1/v2 formulations MEASURED inadmissible "
            "(verdict_scope: FORMULATION). Sibling still parked: from-birth KD "
            "(dw1-CLOSED for continuation; from-SCRATCH needs a separate charter)"
        ),
        open_state="OPEN_BURN_WINDOW_RACE",
        open_reason="needs a matched-seed/schedule A/B slot in a live burn config",
    ),
    Rv1ReactivationSpec(
        "R3_directional_conditioning_from_scratch_race",
        7,
        # rv1 counts R3 as ONE family negative (owed16v2 FORMULATION NO-GO); #502 is a
        # built-never-raced orphan folded into the same race, not a second negative.
        ("owed16v2_directional_family_no_go_formulation_scope",),
        "PARTIAL_RACE",
        "matched from-scratch ON/OFF A/B of an oriented boundary-tangent conditioning "
        "channel (honest label: oriented-Fourier) inside the burn's registered CLADE-ICPE "
        "race slot, same seed/schedule/floor, realized-through-R n600 readout",
        "BURN_WINDOW_RACE",
        "burn config lever race (falsifiers already registered)",
        result_glob=("ddm_r3_oriented_conditioning_race_*.md",),
        landed_disposition=(
            "matched from-scratch ON/OFF n600 A/B of the oriented boundary-tangent "
            "conditioning channel, realized-through-R"
        ),
        open_state="OPEN_LEVER_NOT_BUILT_ON_LIVE_VEHICLE",
        open_reason=(
            "NOT merely un-fired -- UNBUILT: the sealed tb1 ticket carries no CLADE-ICPE slot "
            "and no oriented/directional conditioning lever exists on the tr1 partition "
            "renderer (its only 'directional' surface is the sn1 Road<->Lane asymmetric loss "
            "weight, a different object). The vehicle changed under the row: the -48% "
            "directional evidence was the witness INR's, and CLAUDE.md already routes "
            "self-orient OFF in production pending a matched from-scratch A/B. Closing this "
            "requires a BUILD (channel + DSL lever) before any race, so it can never be a $0 "
            "row -- rank it against the axis weights, not against its rv1 rank"
        ),
    ),
    Rv1ReactivationSpec(
        "R5_step_hosc_head_conditional",
        8,
        ("hosc_fixed_beta_diverges_instance_formulation",),
        "PARTIAL",
        "ONLY IF the burn's Lane-nucleation channel stalls (N8 falsifier): race ONE "
        "step-native/annealed-beta head variant under the cured form (beta-anneal 1.0->4.0 "
        "+ siren-init) inside the (D,c,levels) window; otherwise leave closed",
        "CONDITIONAL_TRIGGER_ONLY",
        "burn lever race (conditional entry only)",
        result_glob=("ddm_r5_step_hosc_head_race_*.md",),
        landed_disposition=(
            "one step-native/annealed-beta head variant raced under the cured form inside the "
            "(D,c,levels) window"
        ),
        open_state="OPEN_NON_BINDING_ON_LIVE_VEHICLE",
        open_reason=(
            "the conditional never armed AND the family does not bind: fh1 records no "
            "sinusoidal layers anywhere on the tr1 conv renderer, so there is no periodic "
            "activation for an annealed-beta cure to act on, and the hard-state selection the "
            "step family chased is already supplied by uint8-STE + the realized A1 gate "
            "(RACED as tr1_token_quant_L16_round). The Lane-nucleation trigger it waited on is "
            "now owned by R6/burn-4. Leave closed; re-open only on a periodic-activation head"
        ),
    ),
)


@dataclass(frozen=True)
class Rv1NonReactivationSpec:
    """One rv1 honest non-reactivation: precondition NOT changed — stays closed."""

    row_id: str
    family: str
    disposition: str
    reason: str


RV1_NON_REACTIVATION_SPECS: tuple[Rv1NonReactivationSpec, ...] = (
    Rv1NonReactivationSpec(
        "X1_warp_predict_family",
        "warp-PREDICT",
        "STAYS_CLOSED_FAMILY_SCOPE",
        "resample-blur flip physics is scorer/vehicle-independent; copy-PREDICT stays locked",
    ),
    Rv1NonReactivationSpec(
        "X2_per_pixel_literal_correction",
        "per-pixel/per-flip literal form",
        "STAYS_CLOSED_SCORER_PHYSICS",
        "position floor + receptive-field collateral are frozen-scorer arithmetic; live "
        "descendants are the TOKEN form (R4) and the band-priced fallback (R2), never dense",
    ),
    Rv1NonReactivationSpec(
        "X3_flip_label_coding",
        "flip-LABEL coding",
        "SOLVED_TO_FLOOR_CONSUMED_AS_IS",
        "fc1 0.32454 b/flip, coder 0.05% off floor; a component R2 consumes, not a negative",
    ),
    Rv1NonReactivationSpec(
        "X4_island_homotopy_ladder_323",
        "island/homotopy ladder",
        "DISSOLVED_ON_TOKEN_VEHICLE",
        "token-grid component birth is a discrete change — no bifurcation/hysteresis/barrier; "
        "residual Lane-birth routes to the sealed Lane-pool race",
    ),
    Rv1NonReactivationSpec(
        "X5_chroma_sub2px",
        "chroma <2px",
        "RECLASSIFIED_POSE_SAFE_EXPLOIT",
        "never a NO-GO: pose-invisible NOT seg-invisible; transfers to seg-null chroma "
        "steering in the terminal pose solve (charter-framing correction recorded)",
    ),
    Rv1NonReactivationSpec(
        "X6_posthoc_stored_pose_family",
        "post-hoc/stored pose",
        "STAYS_DEAD_PRECONDITION_STRENGTHENED",
        "tr1 frames are seg-only-trained BY DESIGN (pose TERMINAL #383) — even less "
        "pose-shaped than the witness's; only the terminal JOINT solve crosses the wall",
    ),
    Rv1NonReactivationSpec(
        "X7_lane_dash_dictionary",
        "lane dash-dictionary",
        "STAYS_CLOSED_DOMINATED_BY_CONTEXT_CODER",
        "pp1: contour Lane field 219.5 KB >> context-arith Lane attribution 62.3 KB",
    ),
    Rv1NonReactivationSpec(
        "X8_explicit_contour_support_copy_base",
        "explicit contour-support stream",
        "OBVIATED_OR_BAND_PRICED_ONLY",
        "measured-dead on the copy base; on the burn carrier native <=5e-4 ships no stream; "
        "in-band re-entry only through R2 pricing",
    ),
    Rv1NonReactivationSpec(
        "X9_self_orient_prerequisite",
        "self-orient as launch prerequisite",
        "STAYS_FALSIFIED_AS_PREREQUISITE",
        "g114 falsified the prerequisite premise; R3 races the family as a conditioning "
        "channel only — no adopt-by-citation",
    ),
    Rv1NonReactivationSpec(
        "X10_kd74_literal_hnerv_student",
        "KD #74 literal HNeRV-student path",
        "NOT_REFUNDED_BANNED_LINEAGE_DESCENDANT_LIVE",
        "#74's own verdict was CONTINUE-pending (never a negative; inert loop was #75/#76, "
        "firewalled) — charter-framing correction recorded; live descendant is R8",
    ),
    Rv1NonReactivationSpec(
        "X11_eikonal_viscosity_era",
        "eikonal/viscosity era",
        "RETRACTED_NO_REACTIVATION_VENUE_HERE",
        "the retraction stands (guard-tainted measurements void) but no SDF/level-set object "
        "exists on the token/conv vehicle for an eikonal term to act on",
    ),
    Rv1NonReactivationSpec(
        "X12_witness_era_formulation_negatives",
        "witness-era formulation negatives",
        "SWEPT_REMAIN_CLOSED",
        "A-C2/C3/C5/C16 + A-E1..E7 correctly scoped; none of N1-N8 touches their killing "
        "preconditions — the honesty check that this sweep does NOT reactivate everything",
    ),
)


RV1_CHARTER_CORRECTIONS: tuple[str, ...] = (
    "X10: KD #74 was never a blocked negative — committed verdict CONTINUE-pending-funded-"
    "long-train; the inert loop was #75/#76 (fleet-level) and the trust ledger firewalls #74",
    "X5: chroma '<2px' was never a NO-GO — it is a pose-safety EXPLOIT (pose-invisible, "
    "not seg-invisible) that transfers to the new vehicle unchanged",
)


def _rv1_conditional_validity_table(repo_root: Path) -> dict[str, Any]:
    """Fold the COMMITTED rv1 re-grade table into typed advisory rows (co8, rung 1).

    co7 registered rv1 as a PENDING producer (table_owner) so the organ would never
    duplicate the sweep; rv1 landed on main (merge 6cf5454509) and this function is the
    named fold: 8 reactivation rows (each a PROPOSAL with a named measurement — nothing
    reactivates until its measurement lands) + 12 honest non-reactivations as tagged
    closed verdicts + 2 charter-framing corrections, all stamped with the committed
    memo's content hash.

    rx1 (2026-07-31) — THE WRITE PATH. `reactivated` shipped as a hardcoded `False` with no
    writer anywhere in the module, so the ledger could never record a discharge: by 07-31,
    FIVE of the eight named measurements had landed (R8 sc2 07-28 · R1 pb1 + R7 07-29 · R4
    gr1 + R2 ea1 07-30) and the SessionStart digest was still advertising the two
    MEASURABLE_NOW rows as free duties — an already-discharged duty, re-offered every
    session. Each row now resolves its own state from COMMITTED evidence on disk (content-
    hashed, fails closed to False when absent). `reactivated=True` means STRICTLY "the named
    measurement landed", never "the lever paid" — R2 and R4 landed as measured NO-GO /
    DOMINATED at their own pre-registered GO tests; the outcome is in `disposition`.
    """

    path = repo_root / RV1_TABLE_MEMO
    if not path.is_file():
        return {
            "available": False,
            "reason": "RV1_COMMITTED_TABLE_ABSENT",
            "actuation": "NONE",
            "score_claim": False,
        }
    sha = _sha256(path)
    source = {"path": RV1_TABLE_MEMO, "sha256": sha}
    research = repo_root / ".omx" / "research"

    def _matches(globs: tuple[str, ...]) -> list[dict[str, str]]:
        """Committed artifacts (path + content hash) for a row's evidence globs."""

        found: list[dict[str, str]] = []
        for pattern in globs:
            for hit in sorted(research.glob(pattern)):
                if hit.is_file():
                    found.append(
                        {"path": str(hit.relative_to(repo_root)), "sha256": _sha256(hit)}
                    )
        return found[:3]

    reactivations: list[dict[str, Any]] = []
    for spec in sorted(RV1_REACTIVATION_SPECS, key=lambda s: s.rank):
        result_hits = _matches(spec.result_glob)
        landed = bool(result_hits)
        # evidence_kind is derived from what was actually FOUND, never from what was
        # declared: a declared-but-unmatched charter glob must read NONE, not CHARTER.
        evidence = result_hits if landed else _matches(spec.charter_glob)
        row: dict[str, Any] = {
            "row_id": spec.row_id,
            "rank": spec.rank,
            "negatives": list(spec.negatives),
            "precondition_changed": spec.precondition_changed,
            "measurement": spec.measurement,
            "status": spec.status,
            "consumer": spec.consumer,
            # EVIDENCE-DERIVED, never asserted: True means strictly "the NAMED reactivation
            # measurement LANDED as a committed artifact" — NOT "the lever paid". The
            # outcome (including measured NO-GOs) lives in `disposition`.
            "reactivated": landed,
            "duty_state": "LANDED" if landed else spec.open_state,
            "disposition": spec.landed_disposition if landed else spec.open_reason,
            "evidence": evidence,
            "evidence_kind": ("RESULT" if landed else "CHARTER") if evidence else "NONE",
            "result_glob": list(spec.result_glob),
            "source": source,
            "actuation": "NONE",
            "score_claim": False,
        }
        reactivations.append(row)
    closed = [
        {
            "row_id": spec.row_id,
            "family": spec.family,
            "disposition": spec.disposition,
            "reason": spec.reason,
            "source": source,
        }
        for spec in RV1_NON_REACTIVATION_SPECS
    ]
    by_status: dict[str, int] = {}
    by_duty_state: dict[str, int] = {}
    for row in reactivations:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        by_duty_state[row["duty_state"]] = by_duty_state.get(row["duty_state"], 0) + 1
    landed_rows = [row for row in reactivations if row["reactivated"]]
    # rv1 §2b accounting: 20 distinct negatives re-graded across the 8 rows (R1:3 · R2:4 ·
    # R3:1 family · R4:3 · R5:1 · R6:3 · R7:4 · R8:1). Derived from the typed rows, never
    # asserted beside them.
    distinct_negatives = sum(len(row["negatives"]) for row in reactivations)
    return {
        "schema": "ddm_costate_rv1_table.v1",
        "available": True,
        "source": source,
        "reactivation_rows": reactivations,
        "non_reactivation_rows": closed,
        "charter_corrections": list(RV1_CHARTER_CORRECTIONS),
        "counts": {
            "reactivations": len(reactivations),
            "non_reactivations": len(closed),
            "distinct_negatives_regraded": distinct_negatives,
            "by_status": dict(sorted(by_status.items())),
            # rx1: the live discharge split. `by_status` is rv1's frozen 2026-07-28
            # schedulability field and is NOT a state — read `by_duty_state`.
            "landed": len(landed_rows),
            "open": len(reactivations) - len(landed_rows),
            "by_duty_state": dict(sorted(by_duty_state.items())),
        },
        "boundary": (
            "re-grades are PROPOSALS with named measurements; NOTHING reactivates without "
            "its measurement landing (rv1 memo boundary, quoted). rx1 amendment: `reactivated` "
            "is RESOLVED FROM COMMITTED EVIDENCE (fails closed to False) and means ONLY that "
            "the named measurement LANDED — never that the lever paid; several landed rows are "
            "measured NO-GOs, read each row's `disposition`"
        ),
        "actuation": "NONE",
        "score_claim": False,
    }


def _pn1_nodes(repo_root: Path, band_parents: Mapping[str, Any]) -> dict[str, Any]:
    """Fold the committed pn1 rows the organ consumes (co8, rung 2).

    Four typed nodes, all content-hashed to the pn1 memo: (a) the S6 VOI ranking as a
    SENSE input (S2+R7 outrank every schedulable action because they can RE-ROUTE the
    sealed burn's waterfill before capacity is spent); (b) the nu-pivot as a named
    DECISION NODE (G4 feasibility pivots at nu in ~[0.55, 0.75]); (c) the S1
    dress-rehearsal Stage-A/Stage-B duty rows; (d) the granularity-ladder ARMED race +
    the rebalance-event watch. Priors from pn1 s7 stay labeled ASSUMED.
    """

    path = repo_root / PN1_MEMO
    if not path.is_file():
        return {
            "available": False,
            "reason": "PN1_COMMITTED_MEMO_ABSENT",
            "actuation": "NONE",
            "score_claim": False,
        }
    source = {"path": PN1_MEMO, "sha256": _sha256(path)}
    any_in_band = bool(band_parents.get("any_parent_in_band"))
    voi = {
        "schema": "ddm_costate_voi_ranking.v1",
        "rule": "VOI = P(outcome changes a decision) x value of the changed decision / cost",
        "priors_label": "ASSUMED (pn1 s7 bands are labeled priors, not measurements)",
        "rows": [
            {
                "rank": 1,
                "measurement": "S2_NU_AUDIT_PLUS_R7_CODER_RACE",
                "cost": "$0, NOW (existing T2 lotto dump)",
                "voi_class": "REROUTING_PRE_BURN",
                "value": (
                    "the ONLY $0 measurements that can RE-ROUTE the sealed burn's "
                    "(D,c,levels) waterfill BEFORE 400 epochs are spent (L2 wide AND "
                    "decision-coupled: nu low => move D up / gate c)"
                ),
            },
            {
                "rank": 2,
                "measurement": "FIRST_BURN_GATES",
                "cost": "inside the burn (operator-GO pending)",
                "voi_class": "NOT_SCHEDULABLE_ARRIVES_WITH_BURN",
                "value": (
                    "most posterior mass (L1 is the wall: 0.0138 -> 5e-4 = 27.7x descent) "
                    "but not a CHOICE; VOI realized automatically when the burn fires"
                ),
            },
            {
                "rank": 3,
                "measurement": "S1_STAGE_A_LOCAL_FULL_N600",
                "cost": "$0, ~35-60 min",
                "voi_class": "CHAIN_VALIDITY",
                "value": (
                    "collapses L5 (chain/drift validity, ASSUMED prior 0.9-0.99) cheaply; "
                    "can only surface GOOD failures (deploy-parity bugs found before they "
                    "contaminate the burn's exact row)"
                ),
            },
            {
                "rank": 4,
                "measurement": "S1_STAGE_B_MODAL_CPU_FLIGHT",
                "cost": "est <$2 inside a <=$20 envelope (PAID; operator-GO)",
                "voi_class": "CALIBRATION_NON_DECAYING",
                "value": "calibrates delta_seg/delta_pose (L5 tail); quiet-slot; value non-decaying",
            },
            {
                "rank": 5,
                "measurement": "LV1_B_VERDICT_T1_VALIDITY",
                "cost": "fires inside burn windows regardless",
                "voi_class": "SPEED_NOT_FEASIBILITY",
                "value": "tunes descent speed + rate-stack order, not feasibility (lv1 PENDING)",
            },
        ],
        "attach_under": {
            "S2_NU_AUDIT_PLUS_R7_CODER_RACE": "FIRST_GATES (prep lane)",
            "S1_STAGE_A_LOCAL_FULL_N600": "BYTE_CLOSE_CHAIN_READY",
            "S1_STAGE_B_MODAL_CPU_FLIGHT": "BYTE_CLOSE_CHAIN_READY",
        },
        "no_new_heads": True,  # pn1 FEED-pn1f: slots under existing heads
    }
    nu_pivot = {
        "schema": "ddm_costate_decision_node.v1",
        "node_id": "nu_fiber_fraction_g4_feasibility_pivot",
        "status": "UNMEASURED_REROUTING_CLASS_PRE_BURN",
        "pivot_window": [0.55, 0.75],
        "feasibility_condition": (
            "(1 - nu) * h_vis <= 0.578 b/quantum (G4 130 KB at N=1,843,200 token quanta; "
            "current lotto stream 2.305 b/q)"
        ),
        "measurement": (
            "S2 token-nullspace audit on the EXISTING T2 lotto dump: |g| sensitivity map "
            "(1 MLX backward) + stratified +/-1-quantum hard-null verification (fd2 36-pair "
            "geometry) + null-snap rate readout, q in {25,50,70,80,90}, tolerance "
            "delta d_seg <= +2e-4"
        ),
        "decision_routed": (
            "burn (D,c,levels) waterfill: nu low => coder race alone cannot reach 130 KB at "
            "D16 — move D up / gate c on boundary cells (both registered levers); either "
            "direction verifies-or-falsifies tb1 design decision #3 (zero-init gauge)"
        ),
        "anchors": (
            "pp1 173.6 KB partition price (upper); eg1 ceilings => tokens <=184.8 KB @0.172 "
            "/ <=151.8 KB @0.15 after renderer+selector+pose sections"
        ),
    }
    s1 = {
        "schema": "ddm_costate_s1_rehearsal_duties.v1",
        "why": (
            "the own-vehicle chain checkpoint -> TR1 export -> archive.zip -> inflate -> "
            "full-n600 evaluate.py has NEVER run end-to-end; the first own-vehicle exact "
            "row validates every stage against the only authority (score quality irrelevant)"
        ),
        "stages": [
            {
                "duty": "S1_STAGE_A_LOCAL_FULL_N600",
                "status": "READY_QUIET_SLOT_ZERO_DOLLARS",
                "spec": (
                    "sh1 protocol on the eg1 rehearsal packet (504,736 B ZIP): resumable "
                    # deterministic-bytes acceptable: deliberate contest-CPU authority receipt (upstream evaluate.py --device cpu axis), not a CUDA-fallback path
                    "inflate -> upstream evaluate.py --device cpu, all 600 samples, "
                    "recompute S from components; drift row #1 = evaluator d_seg vs tb1 "
                    "full-confirm 0.013833 on the SAME weights, green band |delta| <= 5e-4; "
                    "OUTSIDE band => STOP, deploy-parity bug found (the rehearsal's value)"
                ),
                "cost": "$0",
                "actuation": "NONE",
            },
            {
                "duty": "S1_STAGE_B_MODAL_CPU_FLIGHT",
                "status": "OPERATOR_GO_PAID_DISPATCH",
                "spec": (
                    "ONE staged Modal CPU-axis flight via "
                    "tools/dispatch_modal_paired_auth_eval.py (argparse verified; lane claim "
                    "first; harvest within 24h); bare-venv bootstrap smoke + decode "
                    "wall/RSS telemetry; drift row #2 = per-component deltas => eg1 E2's "
                    "staging gate becomes CALIBRATED instead of assumed"
                ),
                "cost": "est <$2 inside a <=$20 envelope",
                "actuation": "NONE (the organ surfaces readiness; dispatch is operator-GO)",
            },
        ],
        "attach_under": "BYTE_CLOSE_CHAIN_READY",
    }
    granularity_race = {
        "schema": "ddm_costate_armed_race_duty.v1",
        "duty": "GRANULARITY_LADDER_RACE",
        "status": "DUE" if any_in_band else "ARMED_NOT_DUE_BAND_STILL_EXPLODE",
        "trigger": (
            "first burn checkpoint whose full-confirm realized d_seg enters the priced "
            "band [5e-4, 1e-2] (the sp1/pp1 armed-duty pattern)"
        ),
        "race": (
            "rungs 2-4 (token delta / mask-bit flip / conditioning delta) at matched bytes "
            "against the rung-5 TerminalSolve quote; eg1 E2's same-parent conservative "
            "score-gain-rate rule adjudicates; rung 0 (native <= rho_c) moots from below, "
            "rung 5 (basin->solve handoff) moots from above"
        ),
        "instrument": "rv1 R4's single-token probe doubles as rung-2's entry instrument",
        "actuation": "NONE",
    }
    rebalance_watch = {
        "schema": "ddm_costate_rebalance_watch.v1",
        "duty": "REBALANCE_EVENT_WATCH",
        "status": "STANDING_WATCH",
        "events": [
            "nu measured (S2) -> re-route the burn (D,c,levels) waterfill pre-capacity",
            "R7 coder-race verdict -> h_vis input to the S4 floor + coder choice",
            "band-entry [5e-4, 1e-2] -> granularity race + R2 pricing flip ARMED->DUE",
            "lv1 receipt or CT1-v2 telemetry lands -> T1 validity + CO5 gate re-check",
            "S1 Stage-A drift row outside |delta d_seg| <= 5e-4 -> deploy-parity STOP",
        ],
        "actuation": "NONE",
    }
    return {
        "schema": "ddm_costate_pn1_nodes.v1",
        "available": True,
        "source": source,
        "voi_ranking": voi,
        "nu_pivot": nu_pivot,
        "s1_rehearsal": s1,
        "granularity_race_duty": granularity_race,
        "rebalance_watch": rebalance_watch,
        "actuation": "NONE",
        "score_claim": False,
    }


def _duty_allocator_waterfill(
    pn1_nodes: Mapping[str, Any],
    rv1_table: Mapping[str, Any],
) -> dict[str, Any]:
    """The allocator law applied to the organ itself (co8, rung 3).

    Operator elevation 07-28: "capacity should flow to where it buys the most." The
    schedulable ($0/cheap) duty queue is ranked by an explicit marginal-value-per-effort
    waterfill WHERE PRICES EXIST; each row labels its pricing basis (DERIVED arithmetic
    vs ASSUMED pn1 prior); same-pool rows COMPETE, never sum (the registered
    non-additive pools law), so R7 races AFTER S2 on the shared token-rate axis (pn1
    row 2: race the coder on the null-snapped stream). This waterfill is ADVISORY and
    is NOT the gated CO5 regret allocator, which requires MEASURED
    compression-progress-per-effort + a typed fired-duty history (both still held).
    """

    if not (pn1_nodes.get("available") and rv1_table.get("available")):
        return {
            "available": False,
            "reason": "PN1_OR_RV1_SOURCE_ABSENT",
            "actuation": "NONE",
            "score_claim": False,
        }
    rows = [
        {
            "rank": 1,
            "duty": "S2_NU_AUDIT",
            "cost": "$0 (~minutes-hours, existing T2 lotto dump)",
            "schedulable": True,
            "pool": "token_rate_axis",
            "value": (
                "re-routes the binding token-rate leg (~530 -> <=130-150 KB target, "
                "~0.27 S-scale) BEFORE burn capacity is spent + emits nu for the G4 "
                "feasibility pivot + verifies/falsifies tb1 zero-init gauge"
            ),
            "pricing_basis": "DERIVED_ARITHMETIC (pn1 s4 byte arithmetic; decision-coupled)",
            "depends_on": None,
        },
        {
            "rank": 2,
            "duty": "R7_CODER_RACE",
            "cost": "$0 (T2 token dumps exist now)",
            "schedulable": True,
            "pool": "token_rate_axis",
            "value": (
                "measures h_vis under the best admissible coder (S4 floor input); pp1 "
                "precedent: context-arith beat generic 2-4x, temporal-as-context -33 KB"
            ),
            "pricing_basis": "DERIVED_PRECEDENT (pp1 measured sister-object transfer)",
            "depends_on": "S2_NU_AUDIT (race on the null-snapped stream; same pool => competes)",
        },
        {
            "rank": 3,
            "duty": "R4_TOKEN_PROBE",
            "cost": "$0 TODAY (T2 checkpoints exist)",
            "schedulable": True,
            "pool": "correction_family",
            "value": (
                "cheapest disambiguator of the whole correction family; unblocks R1 QDBS "
                "design + R2 pricing (2 downstream consumers) + rung-2 instrument"
            ),
            "pricing_basis": "ASSUMED_LEVERAGE (rv1 rank-3; unblock count is structural)",
            "depends_on": None,
        },
        {
            "rank": 4,
            "duty": "S1_STAGE_A",
            "cost": "$0 (~35-60 min)",
            "schedulable": True,
            "pool": "chain_validity",
            "value": "collapses the L5 validity leg (ASSUMED 0.9-0.99) + drift row #1",
            "pricing_basis": "ASSUMED_PRIOR_BAND (pn1 s7, labeled)",
            "depends_on": None,
        },
        {
            "rank": 5,
            "duty": "S1_STAGE_B",
            "cost": "est <$2 (PAID)",
            "schedulable": False,
            "pool": "chain_validity",
            "value": "delta-calibration of the staging gate; non-decaying",
            "pricing_basis": "ASSUMED_PRIOR_BAND (pn1 s7, labeled)",
            "depends_on": "operator-GO (paid dispatch; the organ cannot schedule it)",
        },
        {
            "rank": 6,
            "duty": "GRANULARITY_LADDER_RACE",
            "cost": "bounded burn windows",
            "schedulable": False,
            "pool": "correction_family",
            "value": "adjudicates rungs 2-4 vs the rung-5 solve quote at matched bytes",
            "pricing_basis": "CONTINGENT (armed on band-entry; no price until triggered)",
            "depends_on": "band-entry trigger (ARMED_NOT_DUE while every parent explodes)",
        },
    ]
    return {
        "schema": "ddm_costate_duty_allocator_waterfill.v1",
        "available": True,
        "sources": {
            "pn1": dict(pn1_nodes["source"]),
            "rv1": dict(rv1_table["source"]),
        },
        "law": (
            "rank schedulable duties by marginal value per effort where prices exist; "
            "unpriced rows fall back to dependency topology; same-pool rows COMPETE, "
            "never sum (non-additive pools law)"
        ),
        "rows": rows,
        "pools": {
            "token_rate_axis": "S2 precedes R7 (R7 races on S2's null-snapped stream)",
            "correction_family": "R4 is the entry instrument; the race stays armed",
            "chain_validity": "Stage A free/now; Stage B paid/operator-GO",
        },
        "co5_regret_allocator": (
            "GATED_RE_PREMISE — this waterfill uses DERIVED/ASSUMED value bands and is "
            "NOT the regret-bounded allocator (held on TYPED_FIRED_DUTY_HISTORY + "
            "measured compression-progress-per-effort; see CO5 duty-to-measure queue)"
        ),
        "actuation": "NONE",
        "score_claim": False,
    }


def consumed_evidence_registry() -> dict[str, Any]:
    """The organ's consumed-evidence surface, DERIVED from the live module constants.

    Consumed by ``tools/organ_freshness_gate.py`` (co8 rung 6): the freshness detector
    diffs merged campaign landings on main against exactly this registry, so organ
    rounds become event-driven instead of operator-driven. NO-FAKE: this is computed
    from the same structures ``build_live_ddm_costate`` reads at build time — never a
    manually-maintained sibling list.
    """

    from tac.ddm_campaign_costate import J8F_GLOBS
    from tac.ddm_campaign_costate import SOURCES as CAMPAIGN_SOURCES

    paths = {spec.artifact for spec in ARC_EVIDENCE_SPECS}
    paths.update(
        (
            LIVE_BASE_DSEG_RECEIPT,
            LIVE_BURN_ENDPOINT_EVAL_RECEIPT,
            T3_SEALED_TICKET,
            RV1_TABLE_MEMO,
            PN1_MEMO,
        )
    )
    paths.update(spec.path for spec in CAMPAIGN_SOURCES)
    paths.update(ROUND10_CURRENT_SOURCES)
    globs = [".omx/research/" + spec.glob for spec in SOURCE_SPECS]
    globs += [".omx/research/" + spec.glob for spec in PENDING_PRODUCER_SPECS]
    globs += [".omx/research/" + DEFERRAL_LEDGER_GLOB]  # QA37: the organ's declared consumed source
    globs += list(J8F_GLOBS)
    return {
        "schema": "ddm_costate_consumed_evidence_registry.v1",
        "paths": sorted(paths),
        "globs": sorted(set(globs)),
    }


def _round10_recursive_leverage(
    repo_root: Path,
    open_gate_ownership: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-adjudicate the stale 07-28 queue against the AFR1/DCC1 campaign.

    This is an advisory SENSE/DECIDE surface only.  It does not mutate the old
    ledger, dispatch a producer, or claim that a reparented duty has fired.
    Every input row remains visible with one typed disposition.
    """

    sources: list[dict[str, Any]] = []
    for relative in ROUND10_CURRENT_SOURCES:
        path = repo_root / relative
        sources.append(
            {
                "path": relative,
                "available": path.is_file(),
                "sha256": _sha256(path) if path.is_file() else None,
            }
        )

    # Old vehicle-specific work is not silently called complete.  It is either
    # retained as standing apparatus, explicitly superseded, or reparented to a
    # current DCC1 successor whose owner and consumer are named here.
    dispositions: dict[str, tuple[str, str, str, str]] = {
        "QA09": ("REPARENTED", "task #1378 CCS1", "/Volumes/APDataStore/pact/ddm_gmf1_fitted_crossgroup_gm/", "lossless order/context evidence belongs in the fixed-X causal-schedule rate rung"),
        "QA25": ("SUPERSEDED", "MAIN organ custodian", ".omx/state/main_hot_state.md", "the v10/TR1 specification is not the AFR1/DCC1 object or queue"),
        "QA40": ("REPARENTED", "future QBW1/QBMIX owner", "/Volumes/APDataStore/pact/ddm_no2_quotient_born_body/", "temporal innovation is admissible only inside decoder-causal quotient topology"),
        "QA47": ("REPARENTED", "RB1 renderer owner", "/Volumes/APDataStore/pact/ddm_rb1/", "pose steering is changed-object renderer work, not a fixed-field rate rung"),
        "QA48": ("REPARENTED", "RB1 renderer owner", "/Volumes/APDataStore/pact/ddm_rb1/", "plane/parallax belongs to the exact changed-object renderer"),
        "QA49": ("REPARENTED", "RB1 renderer owner", "/Volumes/APDataStore/pact/ddm_rb1/", "the dual-use Seg check consumes a realized changed-object renderer"),
        "QA54": ("REPARENTED", "RB1 renderer owner", "/Volumes/APDataStore/pact/ddm_rb1/", "photometric gain is renderer-local and cannot precede the changed object"),
        "QD04": ("RETAINED_APPARATUS", "MAIN lane-registry custodian", ".omx/state/main_hot_state.md", "registry reconciliation remains a standing apparatus duty independent of candidate rank"),
        "QD08": ("SUPERSEDED", "MAIN landing-review custodian", ".omx/state/main_hot_state.md", "the r7/eg1 production handoff is outside the current AFR1/DCC1 build chain"),
        "QD10": ("RETAINED_APPARATUS", "apparatus owner", ".omx/state/probe_outcomes.jsonl", "realized-d_pose and identity-canary requirements remain generally applicable"),
        "QD11": ("RETAINED_APPARATUS", "every active arm", ".omx/research/ddm_deferral_queue_ledger_20260729.md", "defer-at-source remains a process invariant; it is not a candidate duty"),
        "QD15": ("RETAINED_APPARATUS", "every tac consumer", "src/tac", "editable-install source identity remains a standing import-custody precondition"),
        "QE09": ("RETAINED_APPARATUS", "rule-118 compliance owner", ".omx/state/canonical_task_status.jsonl", "pretrained-receiver compliance is still required if that family is consumed"),
        "QA60": ("REPARENTED", "RB1 renderer owner", "/Volumes/APDataStore/pact/ddm_rb1/", "the static two-plane extension is a changed-renderer option"),
        "QA61": ("REPARENTED", "QBW1/QBMIX owner", "/Volumes/APDataStore/pact/ddm_no2_quotient_born_body/", "a rank-1 motion carrier must be counted inside a causal quotient object"),
        "QA68": ("REPARENTED", "RB1 renderer owner", "/Volumes/APDataStore/pact/ddm_rb1/", "per-pair experts are renderer options and require their own counted grammar"),
        "QA69": ("REPARENTED", "QBW1/QBMIX owner", "/Volumes/APDataStore/pact/ddm_no2_quotient_born_body/", "bit allocation must be realized inside the counted quotient object"),
        "QA70": ("REPARENTED", "task #1378 CCS1", "/Volumes/APDataStore/pact/ddm_gmf1_fitted_crossgroup_gm/", "minimum-entropy selection is admissible only through a receiver-causal rate model"),
        "QA72": ("SUPERSEDED", "MAIN organ custodian", ".omx/state/main_hot_state.md", "stage-attribution and old pose-floor framing do not own a current DCC1 successor"),
        "QA77": ("REPARENTED", "RB1 renderer owner", "/Volumes/APDataStore/pact/ddm_rb1/", "bilevel composed descent belongs to a changed object, not the fixed-X rate test"),
        "QA65": ("REPARENTED", "QBW1/QBMIX owner", "/Volumes/APDataStore/pact/ddm_no2_quotient_born_body/", "the offset lattice must be part of a decoder-causal counted object"),
        "QA66": ("SUPERSEDED", "MAIN organ custodian", ".omx/state/main_hot_state.md", "the historical rung-A live-base promise is not transferable to AFR1"),
        "QA79": ("REPARENTED", "RB1 renderer owner", "/Volumes/APDataStore/pact/ddm_rb1/", "interpolation order is a renderer-owned exact-R choice"),
        "QA81": ("REPARENTED", "QBW1/QBMIX owner", "/Volumes/APDataStore/pact/ddm_no2_quotient_born_body/", "a lane carrier must be represented inside causal topology and counted parse state"),
        "QA82": ("RETAINED_APPARATUS", "MAIN default-census custodian", ".omx/state/canonical_task_status.jsonl", "generic-default census remains a cross-arm process check"),
        "QA90": ("REPARENTED", "task #1378 CCS1", "/Volumes/APDataStore/pact/ddm_gmf1_fitted_crossgroup_gm/", "temporal token coherence is a candidate causal schedule context, not a separate scorer arm"),
        "QA91": ("REPARENTED", "future QX representation owner", "/Volumes/APDataStore/pact/ddm_qx4/", "erased-lane events may matter only in a decoder-QBT target-overwrite grammar"),
        "QA92": ("CLOSED_CURRENT_INSTANCE", "MAIN organ custodian", ".omx/research/ddm_x012_crossing_ledger_20260901.md", "its own row is instance-dead and the current Cross remains empty; no token-grammar retry"),
    }

    rows: list[dict[str, Any]] = []
    for legacy in open_gate_ownership.get("open_gate_unfired_rows") or []:
        row_id = str(legacy.get("row_id"))
        disposition = dispositions.get(row_id)
        if disposition is None:
            rows.append(
                {
                    **dict(legacy),
                    "round10_disposition": "UNRESOLVED_FAIL_CLOSED",
                    "current_owner": "MAIN organ custodian",
                    "consumer_store": ".omx/state/main_hot_state.md",
                    "reason": "no current-campaign join was proved",
                }
            )
            continue
        state, owner, consumer, reason = disposition
        rows.append(
            {
                **dict(legacy),
                "round10_disposition": state,
                "current_owner": owner,
                "consumer_store": consumer,
                "reason": reason,
            }
        )

    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["round10_disposition"])
        counts[key] = counts.get(key, 0) + 1

    return {
        "schema": "ddm_costate_recursive_leverage_round10.v1",
        "available": all(row["available"] for row in sources),
        "maturity": "_dev",
        "evidence_axis": "[source-inspected advisory]",
        "sources": sources,
        "legacy_queue_source": open_gate_ownership.get("source"),
        "legacy_open_denominator": int(open_gate_ownership.get("open_gate_count") or 0),
        "re_adjudicated_count": len(rows),
        "re_adjudications": rows,
        "disposition_counts": dict(sorted(counts.items())),
        "sense_laws": [
            "decoder_causal_condition_transport_v1",
            "same_basin_sharp_optimum_v1",
            "byte_distortion_cross_intersection_count_v1",
            "roundtrip_token_to_argmax_affine_v1",
        ],
        "duty_ranking": [
            {"rank": 1, "duty": "CCS1_FIXED_X_CAUSAL_GM_RATE_RUNG", "owner": "task #1378 CCS1", "status": "ACTIVE_OWNED"},
            {"rank": 2, "duty": "QX_QBT_TARGET_OVERWRITE_GRAMMAR", "owner": "future MAIN-assigned QX representation owner", "status": "FOLDED_BEHIND_RANK_1"},
            {"rank": 3, "duty": "QBW1_QBMIX_CAUSAL_QUOTIENT_RENDERER", "owner": "existing NO2 successor owner", "status": "FOLDED_BEHIND_NEARER_EXACT_ROWS"},
            {"rank": 4, "duty": "RB1_EXACT_CHANGED_OBJECT_RENDERER", "owner": "existing RB1 owner", "status": "STILL_ADMISSIBLE_EXISTING_OWNER"},
        ],
        "containment": "UNCHANGED",
        "actuation": "NONE",
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
    }


def _burn_endpoint_base(repo_root: Path) -> dict[str, Any] | None:
    """The tb1 burn-ENDPOINT base d_seg, MEASURED by locked evaluate.py on the shipped bytes.

    ng1 §2 row 10: this is the CURRENT live base of the S-A critical-path vehicle (the t3 lotto
    burn descended here from the tr1 T2 full-confirm 0.0138). Parsed from the pinned upstream
    evaluate.py report text ("Average SegNet Distortion: X") inside the committed pfs1 D1 eval
    receipt — the format is fixed by the pinned upstream snapshot, so the read is deterministic.
    Fail-open (None on any absence/parse failure): a SENSE input must never break the digest.
    """

    path = repo_root / LIVE_BURN_ENDPOINT_EVAL_RECEIPT
    if not path.is_file():
        return None
    try:
        payload = _load_json(path)
    except Exception:
        return None
    report = payload.get("report")
    if not isinstance(report, str):
        return None
    m = re.search(r"Average SegNet Distortion:\s*([0-9][0-9.eE+-]*)", report)
    if not m:
        return None
    try:
        d_seg = float(m.group(1))
    except ValueError:
        return None
    if not (0.0 < d_seg < 1.0):
        return None
    return {
        "d_seg": d_seg,
        "archive_sha256": payload.get("archive_sha256"),
        "evidence_axis": payload.get("evidence_axis"),
        "source_path": LIVE_BURN_ENDPOINT_EVAL_RECEIPT,
        "sha256": _sha256(path),
    }


def _band_position_parents(repo_root: Path, base_band: Mapping[str, Any]) -> dict[str, Any]:
    """Per-parent band placement: the LIVE burn endpoint + the pre-arc W_joint / tr1 T2 bases.

    The tr1 arm bases are read machine-readably from the committed sealed ticket
    (``adjudication.arithmetic.{plain,lotto}.full_dseg``); the tb1 burn ENDPOINT is read from the
    committed pfs1 D1 locked-evaluate.py receipt (``_burn_endpoint_base``). ng1 §2 row 10 (co9):
    the pre-arc parents (W_joint 0.0705, tr1 0.0138/0.0141) are STALE — the burn endpoint 0.00389
    is the live base and it is INSIDE the rational band, so ``any_parent_in_band`` is now True and
    the correction-class re-grade is DUE by band-entry. The re-grade carries the QA03/QA04
    white-jitter MEASURED-BREAK-EVEN prior (seg = base-quality; in-band != a promising lever).
    """

    from tac.canonical_equations.ddm_pp1_correction_stream_position_band_20260728 import (
        EQUATION_ID,
        position_cost_band,
    )

    rows: list[dict[str, Any]] = []
    endpoint = _burn_endpoint_base(repo_root)
    if endpoint is not None:
        base = endpoint["d_seg"]
        rows.append(
            {
                "parent": "tb1_burn_endpoint",
                "base_d_seg": base,
                "regime": str(position_cost_band(base)["regime"]),
                "vehicle": "S-A live critical path (t3 lotto endpoint; the CURRENT live base)",
                "measured_correction_value_at_base": (
                    "BREAK_EVEN — QA03 full-population GN/CG seg solve (+1,866 flips, ΔS_seg "
                    "−0.001582 = 1.15% of the −0.138 ceiling, 1.45 B/flip ≈ water 1.27) + QA04 "
                    "attack-search round-2 (+773 flips, ΔS_seg −0.000655) = 3rd/4th white-jitter "
                    "confirmations; in-band is RATIONAL by the band lemma but corrections are "
                    "MEASURED break-even at this base (seg is a base-quality game)"
                ),
                "source": {
                    "path": endpoint["source_path"],
                    "sha256": endpoint["sha256"],
                    "archive_sha256": endpoint["archive_sha256"],
                    "axis": endpoint["evidence_axis"],
                },
            }
        )
    if base_band.get("available"):
        rows.append(
            {
                "parent": "W_joint_describe_line",
                "base_d_seg": base_band["base_d_seg"],
                "regime": base_band["regime"],
                "vehicle": "pre-arc describe-line (S-E lineage; SUPERSEDED-PENDING by the tr1 pivot)",
                "source": dict(base_band["source"]),
            }
        )
    ticket_path = repo_root / T3_SEALED_TICKET
    ticket_row: dict[str, Any] = {"available": False, "reason": "SEALED_TICKET_ABSENT"}
    if ticket_path.is_file():
        ticket = _load_json(ticket_path)
        arithmetic = (ticket.get("adjudication") or {}).get("arithmetic") or {}
        sha = _sha256(ticket_path)
        for arm in ("plain", "lotto"):
            arm_row = arithmetic.get(arm)
            if not isinstance(arm_row, Mapping) or "full_dseg" not in arm_row:
                continue
            base = _number(arm_row["full_dseg"])
            in_domain = 0.0 < base < 1.0
            rows.append(
                {
                    "parent": f"tr1_{arm}_t2_full_confirm",
                    "base_d_seg": base,
                    "counted_bytes": arm_row.get("total_bytes"),
                    "regime": (
                        str(position_cost_band(base)["regime"]) if in_domain else "OUT_OF_DOMAIN"
                    ),
                    "source": {"path": T3_SEALED_TICKET, "sha256": sha},
                }
            )
        adjudication = ticket.get("adjudication") or {}
        ticket_row = {
            "available": True,
            "path": T3_SEALED_TICKET,
            "sha256": sha,
            "status": str(adjudication.get("status", "")),
            "winner_arm": adjudication.get("winner_arm"),
        }
    return {
        "schema": "ddm_costate_band_position_parents.v1",
        "equation_id": EQUATION_ID,
        "rows": rows,
        "any_parent_in_band": any(row.get("regime") == "correct" for row in rows),
        "sealed_ticket": ticket_row,
        "evidence_axis": ARC_EVIDENCE_AXIS,
        "actuation": "NONE",
        "score_claim": False,
    }


def _deferral_ledger_path(repo_root: Path) -> Path | None:
    """Newest committed DDM deferral queue ledger (dated glob; None if absent)."""
    hits = sorted((repo_root / ".omx" / "research").glob(DEFERRAL_LEDGER_GLOB))
    return hits[-1] if hits else None


def _deferral_ledger_source(repo_root: Path) -> dict[str, Any] | None:
    """Content-hashed source stamp for the deferral ledger (path + sha + frontmatter date)."""
    path = _deferral_ledger_path(repo_root)
    if path is None:
        return None
    date_utc = None
    try:
        for ln in path.read_text(errors="replace").splitlines()[:12]:
            m = re.match(r"\s*date_utc:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", ln)
            if m:
                date_utc = m.group(1)
                break
    except Exception:
        date_utc = None
    return {
        "path": str(path.relative_to(repo_root)),
        "sha256": _sha256(path),
        "date_utc": date_utc,
    }


def _ledger_age_days(date_utc: str | None) -> float | None:
    """Age in days of the ledger's as-of date (None if unparseable). Read-only, UTC."""
    if not date_utc:
        return None
    try:
        import datetime as _dt

        d = _dt.date.fromisoformat(date_utc)
        return max(0.0, (_dt.datetime.now(_dt.UTC).date() - d).days)
    except Exception:
        return None


# Gate-status column tokens that mean the gate is OPEN / actionable-now (vs CLOSED / HELD /
# BLOCKED / pre-arc). Matched with an ALPHANUMERIC-BOUNDARY regex, NEVER bare substring
# containment: "MET" is a substring of METHOD / PHOTOMETRIC / TELEMETRY / GEOMETRY / PARAMETER
# and "OPEN" of OPENPILOT / REOPEN, all of which occur in ledger prose. MEASURED on the live
# ledger 2026-07-31 (ddm_deferral_queue_ledger_20260729.md): bare containment flagged 29 rows
# gate-OPEN, boundary matching flags 28 — the extra was QA55 matching "MET" inside "METHOD=0"
# (a ZIP compression-method note), i.e. one phantom duty row in the controller's DECIDE queue.
# "_" counts as a boundary, so a hypothetical GATE_MET cell still matches (fail-safe direction:
# surfacing a row that may not be open is cheaper than hiding one that is).
# Sister bug class: task #829 substring-scan slot-holder.
_LEDGER_GATE_OPEN_TOKENS = ("OPEN", "MET", "FIRED MID-ARM", "GATE FIRED", "MEASURABLE_NOW")
_LEDGER_GATE_CLOSED_TOKENS = ("CLOSED", "BLOCKED", "PRE-ARC", "HELD", "STAGED")
# Row-status tokens that mean the item has NOT completed (an open-gate/no-completion item).
# Substring is CORRECT here: the only containments are OVERDUE->DUE and ORPHANED->ORPHAN, both
# of which SHOULD match. Do not "fix" this one to boundary matching without re-measuring.
_LEDGER_STATUS_UNFIRED = ("DUE", "ORPHAN")


def _ledger_token_re(tokens: tuple[str, ...]) -> re.Pattern[str]:
    """Alphanumeric-boundary alternation over literal ledger tokens (never bare containment)."""
    alternation = "|".join(re.escape(tok) for tok in tokens)
    return re.compile(rf"(?<![A-Z0-9])(?:{alternation})(?![A-Z0-9])")


_LEDGER_GATE_OPEN_RE = _ledger_token_re(_LEDGER_GATE_OPEN_TOKENS)
_LEDGER_GATE_CLOSED_RE = _ledger_token_re(_LEDGER_GATE_CLOSED_TOKENS)


def _open_gate_ownership_scan(repo_root: Path) -> dict[str, Any]:
    """OWNERSHIP-ON-GATE-OPEN SENSE surface (co9, new SENSE law).

    Operator 07-30 routing audit: 4 items had gates that OPENED with no owner-arm to FIRE them
    (QA05 owner-arm died; QA41 gate opened inside an arm with no duty; QA03/04/11 ran-but-unflipped
    before this ledger). This scan makes "gate OPEN + not yet FIRED" machine-surfaced, not
    audit-discovered: it parses the committed deferral ledger's markdown tables and flags every row
    whose gate is OPEN/actionable-now while its row-status is still DUE/ORPHAN (i.e. not FIRED),
    surfacing the OWNER (the consumer column) + the ledger age. A row with an EMPTY consumer is a
    genuine no-owner alarm. ADVISORY only (actuation NONE); fail-open (any error -> unavailable).

    NO-FAKE: derives entirely from the committed ledger cells; asserts nothing the ledger does not
    already state. Does not judge arm-liveness (not in the ledger) — it surfaces open-gate/no-fire
    so the operator/next-arm can route ownership; the routing-audit judgment stays the human's.
    """

    source = _deferral_ledger_source(repo_root)
    if source is None:
        return {
            "schema": "ddm_costate_open_gate_ownership.v1",
            "available": False,
            "reason": "DEFERRAL_LEDGER_ABSENT",
            "actuation": "NONE",
            "score_claim": False,
        }
    path = repo_root / source["path"]
    try:
        text = path.read_text(errors="replace")
    except Exception as exc:
        return {
            "schema": "ddm_costate_open_gate_ownership.v1",
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "actuation": "NONE",
            "score_claim": False,
        }
    age_days = _ledger_age_days(source.get("date_utc"))
    open_rows: list[dict[str, Any]] = []
    for ln in text.splitlines():
        if not ln.lstrip().startswith("| Q"):  # QA*/QD*/QE*/QF* data rows only
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        row_id = cells[0]
        status = cells[-1].upper()
        # gate status column: S-A/S-D tables put it 5th (index 4); S-E puts it 5th too. Use a
        # substring scan across all cells so a schema shift never silently drops a row.
        joined_upper = " ".join(cells).upper()
        status_unfired = any(tok in status for tok in _LEDGER_STATUS_UNFIRED) and "FIRED" not in status
        if not status_unfired:
            continue
        gate_open = bool(_LEDGER_GATE_OPEN_RE.search(joined_upper))
        # PROVABLY REDUNDANT, kept behaviour-identical and named so no reader mistakes it for a
        # veto: it can only be True when gate_open is False, and that case already `continue`s on
        # the first clause below. A CLOSED token does NOT override an OPEN token here.
        gate_closed_only = bool(_LEDGER_GATE_CLOSED_RE.search(joined_upper)) and not gate_open
        if not gate_open or gate_closed_only:
            continue
        consumer = cells[-2] if len(cells) >= 2 else ""
        no_owner = not consumer or consumer in ("—", "-", "?")
        open_rows.append(
            {
                "row_id": row_id,
                "row_status": status.split()[0] if status else status,
                "owner": consumer or "(NONE)",
                "no_owner_alarm": bool(no_owner),
                "item": cells[1][:80] if len(cells) > 1 else "",
                "age_days": age_days,
            }
        )
    # cn1/cn2 gate-opener (QA37 contract): the rc1 branch never landed, so QE03's cn1/cn2
    # consumption waves are HELD on an UNMERGED branch — a gate that OPENS from an off-main
    # landing, invisible to the on-main ledger scan. Encode it as a named gate-opener so the
    # unmerged-branch recall surface is not orphaned (deferral-scatter lesson).
    gate_openers = [
        {
            "gate_opener": "rc1_branch_landing",
            "opens": "QE03 (cn1 #726 + cn2 #727 consumption waves; recovered receipts + PDW1 fp32)",
            "state": "UNMERGED_BRANCH — rc1 never landed; HELD off-main",
            "recall_surface": "git branch -a --no-merged main + the QD14 standing unmerged-branch sweep",
            "actuation": "NONE",
        }
    ]
    no_owner = [r for r in open_rows if r["no_owner_alarm"]]
    return {
        "schema": "ddm_costate_open_gate_ownership.v1",
        "available": True,
        "source": source,
        "ledger_age_days": age_days,
        "open_gate_unfired_rows": open_rows,
        "open_gate_count": len(open_rows),
        "no_owner_alarm_count": len(no_owner),
        "no_owner_alarm_rows": [r["row_id"] for r in no_owner],
        "gate_openers": gate_openers,
        "note": (
            "DUE/ORPHAN rows whose gate is OPEN but which have NOT fired; OWNER = the ledger "
            "consumer cell; age = ledger as-of. Advisory routing surface, not an arm-liveness judge."
        ),
        "actuation": "NONE",
        "score_claim": False,
    }


def _sense_laws(
    arc_index: Mapping[str, Mapping[str, Any]],
    pn1_source: Mapping[str, Any] | None = None,
    ledger_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The 07-28 SENSE laws, anchored to committed artifacts.

    * ``ema_gate_basis_v1`` — instrument-validity precondition (the #85 class):
      gate basis is LIVE params before warmup W = 2/(1-d), EMA shadow after.
    * ``basin_solve_handoff_v1`` — operator-binding preference: on MEASURED
      basin-entry, hand descent to the solve/finisher chain; detectors PRIORITIZE a
      finisher quote (eg1 E2 typed policy), they never auto-stop training.
    * ``correction_granularity_ladder_v1`` (co8) — the pn1 s8 DERIVED law; its RACE
      stays a contingent ARMED duty (band-entry trigger), never a pre-burn spend.
    """

    tb1 = arc_index.get("tb1_t2_race_verdict")
    policy = arc_index.get("eg1_e2_stop_policy")
    gate_basis = {
        "law_id": "ema_gate_basis_v1",
        "statement": (
            "gate/instrument basis = LIVE params before the registered warmup boundary "
            "W = 2/(1-d) steps (d = EMA decay), EMA shadow after; every gate row records "
            "the basis it was measured on"
        ),
        "empirical_anchor": (
            "#85 EMA shadow-lag confound CAUGHT LIVE: the first tb1 T2 launch aborted at ep9 "
            "on a shadow-lag gate artifact (custody t2_n600_plain_aborted_emawarmup_gate_"
            "artifact/); the relaunch under this law ran 40/40 epochs with zero confound "
            "alarms and correctly classified the ep24 double-rebase as FIRST_GATE"
        ),
        "consequence": (
            "a gate/telemetry row without basis custody is #85-suspect and must not clear "
            "L3 verdict-clearance"
        ),
        "status": "ACTIVE_INSTRUMENT_VALIDITY_PRECONDITION",
        "code_commit": "17166ee9c4",
        "source": (
            {"path": tb1["artifact"], "sha256": tb1["sha256"]}
            if tb1 and tb1.get("available")
            else None
        ),
    }
    basin = {
        "law_id": "basin_solve_handoff_v1",
        "statement": (
            "on MEASURED basin-entry, hand descent off to the solve/finisher chain "
            "(solve-only preferable; the train-least doctrine); trajectory detectors "
            "PRIORITIZE a finisher quote, they never auto-stop training"
        ),
        "never_auto_stop": True,
        "detector_guards": {
            "#344_ncde": "shadow-only, actuation NONE; defaults are provisional forecast features",
            "#216_saddle_staircase": (
                "no calibrated classifier; a staircase prioritizes a QDBS quote only"
            ),
            "#475_grokking": (
                "negative scoped to a fixed 31-feature quadratic chart; no stage-advance "
                "authority"
            ),
        },
        "executable_form": (
            "eg1 E2 typed policy: MEASURE_FINISHER_QUOTE -> HANDOFF_* only on a same-parent "
            "conservative score-gain-rate dominance (finisher lower bound strictly above the "
            "training window's upper bound)"
        ),
        "live_evaluation": {
            "status": "ARMED_NO_TRIGGER",
            "reason": (
                "tb1 T2 measured still-descending trajectories at window end (no plateau); "
                "no basin-entry detector hit is committed on the live tr1 parent"
            ),
            "duty_modulation": "NONE — bounded training windows remain the head",
        },
        "doctrine_anchor": "train_least_surgical_kolmogorov_projection_realization_doctrine_20260716",
        "source": (
            {"path": policy["artifact"], "sha256": policy["sha256"]}
            if policy and policy.get("available")
            else None
        ),
    }
    ladder = {
        "law_id": "correction_granularity_ladder_v1",
        "kind": "DERIVED_LAW",
        "statement": (
            "for a frozen scorer whose error mass is REGION/topology-level, "
            "bytes-per-realized-flip improves monotonically with the COHERENCE of the "
            "edit bought per byte: collateral (#51: a local appearance discontinuity "
            "reads as a NEW boundary) falls as coherence rises while position/value cost "
            "per affected pixel falls as one coordinate steers a larger footprint"
        ),
        "rungs": (
            "0 native<=rho_c (no corrections exist) | 1 per-pixel (MEASURED DEAD: 1.525 "
            "B/flip floor + collateral) | 2 per-token region repaint (UNMEASURED; rv1 R4 "
            "probe = entry instrument) | 3 mask-bit flip | 4 conditioning delta | "
            "5 NO stream: extra TerminalSolve iterations on already-counted params "
            "(the basin->solve law applied to corrections; eg1 E3 executors built)"
        ),
        "dominance_hypothesis": (
            "pre-registered: bytes-per-realized-flip strictly improves up the ladder"
        ),
        "falsifier": (
            "any rung inversion at matched bytes through the realized flip gate replaces "
            "the ladder's ordering — the race result has authority, never the elegance"
        ),
        "race_posture": (
            "CONTINGENT, not a pre-burn spend: fires ONLY on band-entry [5e-4, 1e-2] "
            "(see granularity_race_duty in pn1_nodes; sp1/pp1 ARMED pattern)"
        ),
        "cross_links": (
            "g2g2 low-dim coefficient dominance; #669c deepest-layer assignment; "
            "pools law (rungs compete for ONE residual — race then waterfill, never stack)"
        ),
        "status": "REGISTERED_LAW_RACE_ARMED_NOT_DUE",
        "source": dict(pn1_source) if pn1_source else None,
    }
    # ── co9 measured laws, anchored to the committed deferral ledger (which cites the SSD
    #    receipts). Each is MEASURED on the [macOS-CPU advisory] axis; none is a contest score.
    ledger_src = dict(ledger_source) if ledger_source else None
    two_plane_pose = {
        "law_id": "posenet_far_field_photometrics_bidirectional_v1",
        "statement": (
            "PoseNet reads FAR-FIELD photometrics: any actuator that touches far-field content is "
            "priced on the POSE axis, in BOTH directions. FORWARD (encode it): a two-plane per-class "
            "warp (far cls2 -> pure-rotation K·R·K⁻¹ at s_t=0 · ground -> full homography · hood -> "
            "identity; masks = the shipped partition, rule-118 free) RECOVERS pose. REVERSE (drop it): "
            "freezing far-field content COSTS pose. Same mechanism, opposite sign."
        ),
        "empirical_anchor": (
            "FORWARD QA43: composed pose axis 1.4881 -> P0 1.2630 -> 0.9127 = −0.5754 S at ≤7.3 KB "
            "marginal (95/112 tail wins, 41 >10×); REVERSE QA06 Knee-A: sky/hood drops froze "
            "far-field -> +0.185 S pose (0.28002128) — the SAME two-plane physics IN REVERSE"
        ),
        "consequence": (
            "price every far-field-touching actuator (token drops, sky/hood edits, waterfill rungs) "
            "on the JOINT axis, not SegNet flips alone (the Knee-A INSTANCE reject was a pose-blind "
            "waterfill); pose successors QA43 stage-1b + QA44 photometric rungs ride this law"
        ),
        "epistemic_status": "MEASURED",
        "evidence_axis": ARC_EVIDENCE_AXIS,
        "ledger_rows": ["QA43", "QA06", "QA44"],
        "source": ledger_src,
    }
    sensitivity_spread = {
        "law_id": "token_sensitivity_spread_nu_pivot_v1",
        "statement": (
            "The token stream's per-quantum sensitivity is 35× spread (median 8.6e-5 / p99 3.0e-3) "
            "with 27% grads exact-zero, and ν=0.7 sits INSIDE the pivot band [0.55, 0.75] -> "
            "CONTINUOUS log-bit allocation (sensitivity-ordered waterfill) DOMINATES the 3-rung "
            "{L16,L8,L4} ladder; uniform null-snap is DOMINATED by the wr1 sensitivity-ordered curve."
        ),
        "empirical_anchor": (
            "QA11 S2 ν nullspace re-measure on the t3 FINAL ckpt: ν=0.7 in-band (old ν=0.0 = FORM "
            "ARTIFACT confirmed); hard-null deciles 0.56 -> 13.0; the generic-triple law's 2nd "
            "instrument (uniform null-snap dominated). CONFOUND CARRIED: prereg baseline d_seg "
            "0.013833 vs measured q=0 0.0038892 (3.6×) — the prereg number is QUARANTINED"
        ),
        "consequence": (
            "re-prices the QA07 (nested-quant ladder) + QA12 (token-LOTTO) token-RATE pool toward "
            "continuous log-bit allocation; the 3-rung ladder is dominated (QA07 amendment)"
        ),
        "epistemic_status": "MEASURED",
        "evidence_axis": ARC_EVIDENCE_AXIS,
        "ledger_rows": ["QA11", "QA07", "QA12"],
        "source": ledger_src,
    }
    white_jitter = {
        "law_id": "seg_is_base_quality_white_jitter_v1",
        "statement": (
            "SEG is a BASE-QUALITY game, not a corrections game: at the in-band burn base (0.00389), "
            "full-population seg solves and attack searches move flips at ≈ the region-merge water "
            "level (1.27 B/flip) — i.e. BREAK-EVEN. In-band by the band lemma is RATIONAL but NOT a "
            "promising correction lever; only a LOWER BASE (burn capacity) descends seg."
        ),
        "empirical_anchor": (
            "QA03 full-population GN/CG seg solve: +1,866 flips, ΔS_seg −0.001582 = 1.15% of the "
            "−0.138 ceiling, 1.45 B/flip ≈ water 1.27 (~break-even) = 3rd white-jitter confirmation; "
            "QA04 attack-search round-2: +773 flips, ΔS_seg −0.000655 @ 800 evals = 4th confirmation"
        ),
        "consequence": (
            "the correction-class re-grade is DUE by band-entry (ng1 §2 row 10 re-parent) BUT its "
            "MEASURED prior is BREAK-EVEN; corrections-class duties stay low-priority — the seg "
            "descent lever is the burn (lower the base), not any correction stream at this base"
        ),
        "epistemic_status": "MEASURED",
        "evidence_axis": ARC_EVIDENCE_AXIS,
        "ledger_rows": ["QA03", "QA04"],
        "source": ledger_src,
    }
    # ── ja1 (QA73) STANDING allocator law: the digest surfaces the top of the JOINT
    #    waterfill table, NOT axis-scoped duties. Every rung fires on its JOINT realized
    #    exchange rate (read from the committed table), never on axis identity. Advisory.
    joint_allocator = {
        "law_id": "joint_exchange_rate_allocator_v1",
        "statement": (
            "no rung fires on axis identity; every rung fires on its JOINT realized exchange rate "
            "read from the committed ja1 waterfill table. Pools are NON-ADDITIVE (same-pool levers "
            "COMPETE, never sum). At the v4c base (S 0.992972; seg 0.431 · pose 0.322 · rate 0.240) "
            "the axes are near-parity, so the axis-reflex (biggest-axis-first) is the dominant "
            "allocation error — the JOINT table overrides it."
        ),
        "allocation_surprise": (
            "seg is the LARGEST axis (0.431) but its BYTE pool is MEASURED-SATURATED at the "
            "cell_drop50 knee (gr1: restore +0.047S / drop-more +0.052S) AND seg corrections are "
            "BREAK-EVEN (co9 white-jitter). Every cheap LIVE byte lever is on the POSE axis; seg "
            "only moves via a HEAVY re-burn (QA24, a CAPACITY pool, PARALLEL not vs the byte budget)."
        ),
        "table_top": [
            "1 QA66 photometric per-pair rung-A: REALIZED-live-base −0.0134 S @ +150 B (READY, v4d)",
            "2 QA72a+QA54 information/precision: $0 stage-attribution (DERIVED content-limited) — run FIRST",
            "3 QA68 pose-content expert menu: DERIVED headroom in the 88-pair tail (90% of pose mass; UNBUILT)",
            "4 QA65 dim0 offset-finer: DERIVED-bounded-SMALL (pose content-limited, NOT storage — demoted)",
            "5 QA24 seg re-burn: ≤ −0.098 seg+rate LOWER BOUND (HEAVY, operator-GO, PARALLEL capacity pool)",
        ],
        "saturated_do_not_spend": [
            "token-cell bytes (gr1: knee, any move +0.05S)",
            "rate/lossless container (SMEVR floor + deflate consumed)",
            "seg corrections (co9 white-jitter break-even)",
        ],
        "order_of_operations": (
            "physical order motion→projection→photometric→uint8→coder; token_base change invalidates "
            "pose+photo+selector (re-solve ~1h+30min), pose change invalidates photo fit (~30min), so "
            "cheap pose rungs (v4d, no token-base touch) come FIRST; the seg re-burn (v5) LAST/PARALLEL."
        ),
        "epistemic_status": "MEASURED_ALLOCATOR",
        "evidence_axis": ARC_EVIDENCE_AXIS,
        "ledger_rows": ["QA73", "QA66", "QA65", "QA68", "QA72", "QA54", "QA24", "QA64"],
        "committed_table": ".omx/research/ddm_ja1_joint_waterfill_table_20260731.json",
        "committed_atlas": ".omx/research/ddm_ja1_joint_sensitivity_atlas_20260731.json",
        "committed_order_dag": ".omx/research/ddm_ja1_order_of_operations_dag_20260731.json",
        "source": ledger_src,
    }
    return {
        "schema": "ddm_costate_sense_laws.v1",
        "rows": [
            gate_basis,
            basin,
            ladder,
            two_plane_pose,
            sensitivity_spread,
            white_jitter,
            joint_allocator,
        ],
        "actuation": "NONE",
        "score_claim": False,
    }


def _conditional_validity_review(
    arc_rows: Sequence[Mapping[str, Any]],
    band_parents: Mapping[str, Any],
    *,
    rv1_table_available: bool = False,
) -> dict[str, Any]:
    """The substrate-change trigger: evaluate precondition tags against the live state.

    A negative verdict binds only while its preconditions hold. When a precondition
    breaks: a committed successor re-grades it in place (RE_GRADED_BY_COMMITTED_SUCCESSOR),
    otherwise a typed re-grade duty is surfaced (DUE, or ARMED while the band still
    explodes). The FULL historical sweep is owned by rv1: PENDING through co7,
    COMMITTED + CONSUMED as ``rv1_table`` since co8 — the arc-local rows here remain
    the live-arc seeds (no duplicate sweep either way).
    """

    index = {row["finding_id"]: row for row in arc_rows if row.get("available")}
    live_parent = (
        "tr1_lotto_sealed" if "tb1_t3_sealed_ticket" in index else "W_joint_describe_line"
    )
    any_in_band = bool(band_parents.get("any_parent_in_band"))
    successors: dict[str, list[str]] = {
        "fd1_fixed_capacity_wjoint_parametrization": [
            fid
            for fid in (
                "fd2_zero_accept_disambiguation",
                "tb1_t2_race_verdict",
                "tb1_t3_sealed_ticket",
            )
            if fid in index
        ],
    }
    rows: list[dict[str, Any]] = []
    duties: list[dict[str, Any]] = []
    for row in arc_rows:
        if not row.get("available"):
            continue
        for pre in row.get("preconditions") or ():
            kind = str(pre["kind"])
            if kind == "band_regime":
                status = "BROKEN" if any_in_band else "HOLDS"
            elif kind == "parent":
                status = "HOLDS" if live_parent == "W_joint_describe_line" else "BROKEN"
            elif kind == "parametrization":
                status = "BROKEN" if successors.get(str(pre["precondition_id"])) else "HOLDS"
            else:
                status = "UNEVALUATED"
            entry: dict[str, Any] = {
                "schema": PRECONDITION_SCHEMA,
                "finding_id": row["finding_id"],
                "precondition_id": pre["precondition_id"],
                "kind": kind,
                "holds_when": pre["holds_when"],
                "invalidated_by": pre["invalidated_by"],
                "status": status,
                "live_parent": live_parent,
            }
            if status == "BROKEN":
                named = successors.get(str(pre["precondition_id"])) or []
                if named:
                    entry["disposition"] = "RE_GRADED_BY_COMMITTED_SUCCESSOR"
                    entry["successors"] = named
                else:
                    entry["disposition"] = "REGRADE_CANDIDATE"
                    duties.append(
                        {
                            "duty": f"RE_GRADE_{row['finding_id'].upper()}",
                            "finding_id": row["finding_id"],
                            "precondition_id": pre["precondition_id"],
                            "reason": (
                                f"precondition {pre['precondition_id']} broken "
                                f"(live parent={live_parent}); re-grade the verdict on the "
                                "new substrate"
                            ),
                            "status": (
                                "DUE"
                                if any_in_band or kind == "band_regime"
                                else "ARMED_NOT_DUE_BAND_STILL_EXPLODE"
                            ),
                            "actuation": "NONE",
                        }
                    )
            rows.append(entry)
    return {
        "schema": "ddm_costate_conditional_validity.v1",
        "live_parent": live_parent,
        "any_parent_in_band": any_in_band,
        "rows": rows,
        "re_grade_duties": duties,
        "table_owner": (
            "rv1_conditional_validity_table (COMMITTED + CONSUMED co8 as rv1_table; "
            "arc-local rows here remain the live-arc seeds — no duplicate sweep)"
            if rv1_table_available
            else (
                "rv1_conditional_validity_table (pending producer; this organ seeds "
                "arc-local rows only — no duplicate sweep)"
            )
        ),
        "table_owner_state": "CONSUMED_CO8" if rv1_table_available else "PENDING",
        "actuation": "NONE",
        "score_claim": False,
    }


def _endgame_chain_duties(
    arc_index: Mapping[str, Mapping[str, Any]],
    pending_rows: Sequence[Mapping[str, Any]],
    band_parents: Mapping[str, Any],
) -> dict[str, Any]:
    """The 07-28-late endgame duty chain, each leg cited to a committed receipt.

    Chain: B-verdict watch -> burn-fire -> first-gates -> T1-validity-gate ->
    byte-close chain readiness (eg1's E1 as the R6 on-ramp). The organ is advisory:
    burn-fire is a heavy launch and stays operator-GO.
    """

    pending = {row["producer"]: row for row in pending_rows}
    lv1 = pending.get("lv1_token_stack_prices") or {}
    sealed = "tb1_t3_sealed_ticket" in arc_index

    def _cites(*fids: str) -> list[str]:
        return [fid for fid in fids if fid in arc_index]

    chain = [
        {
            "rank": 1,
            "duty": "B_VERDICT_WATCH",
            "status": "STANDING_WATCH",
            "basis": (
                "watch the burn's realized corner against the eg1 exact ceilings "
                "(190,334 B @0.172 / 157,294 B @0.15 at d_seg 3e-4, d_pose 2.33e-5; "
                "TR1-A 149 KB -> S 0.144177 is the only sub-bar corner)"
            ),
            "cites": _cites("eg1_e2_stop_policy"),
        },
        {
            "rank": 2,
            "duty": "T3_BURN_FIRE",
            "status": "READY_OPERATOR_GO" if sealed else "NOT_SEALED",
            "basis": (
                "sealed LOTTO ticket READY_TO_FIRE_UNDER_STANDING_GO; fires from MAIN only; "
                "heavy launch = operator-GO (CONTAINMENT) — the organ surfaces readiness only"
            ),
            "cites": _cites("tb1_t3_sealed_ticket", "tb1_t2_race_verdict"),
        },
        {
            "rank": 3,
            "duty": "FIRST_GATES",
            "status": "BLOCKED_ON_BURN_FIRE",
            "basis": (
                "first gate_every=10 full-confirms under ema_gate_basis_v1 (live before "
                "W=2/(1-d), shadow after); lane Betti-0 is a burn stage-exit facet; the "
                "Lane-pool lever race fires FIRST (tb1 caveat handling)"
            ),
            "cites": _cites("tb1_t2_race_verdict"),
            "law": "ema_gate_basis_v1",
        },
        {
            "rank": 4,
            "duty": "T1_VALIDITY_GATE",
            "status": (
                "PRODUCER_COMMITTED_FOLD_NEXT_ROUND"
                if lv1.get("available")
                else "PENDING_COMMITTED_PRODUCER_LV1"
            ),
            "basis": (
                "truncated token-stack prices are admissible only with realized-validity "
                "rows; pools-law: truncation vs quantization candidates COMPETE, never sum"
            ),
            "cites": [],
            "pending_producer": "lv1_token_stack_prices",
        },
        {
            "rank": 5,
            "duty": "BYTE_CLOSE_CHAIN_READY",
            "status": "BUILT_STANDING",
            "basis": (
                "eg1 E1 rehearsed the TR1 packet/receiver/locked-eval interface = the R6 "
                "exact-eval on-ramp; keep exporter identity bound to the burn's checkpoints"
            ),
            "cites": _cites("eg1_e1_byteclose_rehearsal"),
        },
    ]
    for row in chain:
        row["actuation"] = "NONE"
    head = next((row for row in chain if row["status"] == "READY_OPERATOR_GO"), chain[0])
    return {
        "schema": "ddm_costate_endgame_chain.v1",
        "derivation": (
            "committed-receipt chain (tb1 seal + eg1 E1/E2/E3); head = the first "
            "operator-actionable leg; corrections-class legs absent because every live "
            f"parent is above-band (any_parent_in_band={band_parents.get('any_parent_in_band')})"
        ),
        "chain": chain,
        "head_actionable": head["duty"],
        "actuation": "NONE",
        "score_claim": False,
    }


def _instrument_audit(backtest: Mapping[str, Any], v19b: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ncde": {
            "legacy_r2": 0.060,
            "status": "INVALID_FOR_LIVE_DDM",
            "reason": (
                "the old fit consumes witness epoch trajectories; v19b contains ten discrete, "
                "family-changing greedy moves rather than a controlled continuous path"
            ),
            "replacement": "persist per-cycle DDM objective/bytes/family/D2 iteration curves",
            "refit_gate": ">=8 same-family DDM cycles with hash-stable inputs and walk-forward r2>=0.5",
        },
        "factorized_adjoint": dict(backtest),
        "pose_gate": {
            "status": "RETIRED_NOT_APPLICABLE_TO_DDM_DISCRETE_SOLVE",
            "legacy_degenerate_guard": "PRESERVED_FAIL_TO_BANKED_R1",
            "reason": "sigma_min witness finisher gate is not a DDM describe-line control",
        },
        "maturity": {
            "current": MATURITY,
            "prod_criteria": [
                "all required source hashes closed and consumer-verified",
                "full 600-pair g3 pair/site lambda available without shared-byte duplication",
                "J_paint receiver realization and R6 exact rehearsal complete",
                "DDM iteration instrument passes walk-forward r2>=0.5 or remains explicitly retired",
                "three clean independent reviews and MAIN landing review",
                "contest-CPU/CUDA evidence kept separate from macOS advisory evidence",
            ],
            "met": False,
        },
        "v19b_curve": {
            "points": len(v19b["greedy_screen"]["per_move_joint_table"]),
            "final_joint_delta": v19b["n600"]["joint_delta_vs_v15_control"]["joint_delta"],
            "archive_delta_bytes": v19b["n600"]["joint_delta_vs_v15_control"]["delta_archive_bytes"],
        },
    }


def _source_public(source: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in source.items() if k != "payload"}


def build_live_ddm_costate(
    *,
    repo_root: Path = REPO,
    resume_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one read-only live DDM SENSE→COMPOSE→DECIDE cycle."""

    sources = discover_sources(repo_root)
    missing_required = [name for name, row in sources.items() if row["required"] and not row["available"]]
    if missing_required:
        return {
            "schema": SCHEMA,
            "available": False,
            "status": "MISSING_REQUIRED_LIVE_DDM_RECEIPTS",
            "missing_required": missing_required,
            "sources": {k: _source_public(v) for k, v in sources.items()},
            "actuation": "NONE",
            "score_claim": False,
        }

    dv1, dv1_custody = _dv1_summary(repo_root, sources["dv1"])
    g3_atlas, g3_bulk = _g3_atlas(sources["g3"])
    g4 = sources["g4"]["payload"]
    ev1 = sources["ev1"]["payload"]
    v19b = sources["v19b"]["payload"]
    scorer_value_oracle = ScorerValueOracle(repo_root)
    oracle_rate_row = scorer_value_oracle.bucket_assignments()
    hashes = {name: row["sha256"] for name, row in sources.items() if row["available"]}
    oracle_rate_sha = oracle_rate_row.lineage[0].observed_sha256
    if oracle_rate_sha is None:  # pragma: no cover - fail-closed read guarantees this
        raise ValueError("scorer-value oracle returned fresh rate data without SHA")
    hashes["scorer_value_oracle:rate"] = oracle_rate_sha
    hashes["dv1_summary"] = str(dv1_custody["sha256"])
    if g3_bulk.get("sha256"):
        hashes["g3_full_atlas"] = str(g3_bulk["sha256"])
    lambda_bundle = build_ddm_lambda_bundle(
        atlas=g3_atlas,
        evidence_join=ev1,
        source_hashes=hashes,
    )
    pairs = list(lambda_bundle["pair_rows"])
    sites = list(lambda_bundle["site_rows"])
    campaign = build_campaign_costate(
        repo_root=repo_root,
    )
    for name, row in campaign["source_lineage"]["sources"].items():
        hashes[f"campaign:{name}"] = str(row["sha256"])
    j8f_source = campaign["source_lineage"]["j8f_verdict_stream"]
    if j8f_source.get("sha256"):
        hashes["campaign:j8f_verdict_stream"] = str(j8f_source["sha256"])
    if resume_state is not None:
        prior = dict(resume_state.get("source_hashes") or {})
        if prior and prior != hashes:
            changed = sorted(set(prior) | set(hashes))
            changed = [name for name in changed if prior.get(name) != hashes.get(name)]
            raise ValueError("resume source hashes are stale; re-derive instead of restoring: " + ",".join(changed))

    backtest = dict(lambda_bundle["backtest"])
    blocks = _block_costates(v19b)
    primitives = _primitive_costates(dv1, g4)
    scheduler = _scheduler(dv1, g4, blocks)
    duties = _duties(scheduler)
    arc_evidence = _arc_evidence_rows(repo_root)
    band_position = _band_position(repo_root)
    band_parents = _band_position_parents(repo_root, band_position)
    arc_index = {row["finding_id"]: row for row in arc_evidence if row.get("available")}
    rv1_table = _rv1_conditional_validity_table(repo_root)
    pn1_nodes = _pn1_nodes(repo_root, band_parents)
    ledger_source = _deferral_ledger_source(repo_root)
    open_gate_ownership = _open_gate_ownership_scan(repo_root)
    round10 = _round10_recursive_leverage(repo_root, open_gate_ownership)
    sense_laws = _sense_laws(
        arc_index,
        pn1_source=pn1_nodes.get("source") if pn1_nodes.get("available") else None,
        ledger_source=ledger_source,
    )
    pending_producers = _pending_producers(repo_root)
    conditional_validity = _conditional_validity_review(
        arc_evidence,
        band_parents,
        rv1_table_available=bool(rv1_table.get("available")),
    )
    duty_allocator = _duty_allocator_waterfill(pn1_nodes, rv1_table)
    duties_endgame = _endgame_chain_duties(arc_index, pending_producers, band_parents)
    duties_refreshed = _refreshed_duties(duties, arc_evidence, band_position)
    instruments = _instrument_audit(backtest, v19b)

    spline = next(row for row in dv1["joint_compositions"] if row["candidate_id"] == "spline_plus_events")
    persistent = next(
        row
        for row in dv1["primitive_measurements"]
        if row["measurement"]["candidate_id"] == "persistent_level_set_ground_partition"
    )["measurement"]
    n600 = v19b["n600"]
    optional_missing = [
        spec.name
        for spec in SOURCE_SPECS
        if not spec.required and not sources[spec.name]["available"]
    ]
    queue = [
        {
            "producer": name,
            "reason": sources[name]["reason"],
            "command": ("python3 tools/ddm_costate_organ.py --json # rerun after the producer receipt lands"),
        }
        for name in optional_missing
    ]
    if g3_bulk["status"] != "VERIFIED":
        queue.insert(
            0,
            {
                "producer": "g3_full_atlas",
                "reason": g3_bulk["status"],
                "command": "restore/verify the SHA-pinned g3 JSONL on the SSD, then re-derive",
            },
        )
    if not pairs:
        queue.insert(
            0,
            {
                "producer": "pair_site_lambda",
                "reason": "no exact g3/EV1 pair join available",
                "command": "python3 tools/ddm_costate_organ.py --json",
            },
        )

    cycle = int((resume_state or {}).get("cycle", -1)) + 1
    checkpoint = DdmCostateCheckpoint(
        source_hashes=hashes,
        completed_block_ids=["source_custody", "pair_site_lambda"] if pairs else ["source_custody"],
        cycle=cycle,
    )
    return {
        "schema": SCHEMA,
        "available": True,
        "status": "LIVE_DDM_ADVISORY",
        "maturity": MATURITY,
        "research_only": True,
        "execution_allowed": False,
        "actuation": "NONE",
        "score_claim": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
        "evidence_axis": EVIDENCE_AXIS,
        "equation_id": EQUATION_ID,
        "rate_break_even_score_per_byte": RATE_BREAK_EVEN_SCORE_PER_BYTE,
        "live": {
            "reach": {
                "candidate": spline["candidate_id"],
                "road_described_fraction": spline["road_described_fraction"],
                "persistent_road_described_fraction": persistent["road_described_fraction"],
                "scope": "semantic cells only; receiver realization owed",
            },
            "box": {
                "inside_c1_byte_box": spline["inside_c1_byte_box"],
                "counted_bytes": spline["counted_bytes"],
                "new_joint_section_bytes": spline["new_joint_section_bytes"],
            },
            "fleet": {
                "present": sum(row["available"] for row in sources.values()),
                "registered": len(sources),
                "missing_optional": optional_missing,
                "v19b_accepted_moves": v19b["accepted_move_count"],
                "v19b_final_archive_bytes": n600["measurement"]["archive_bytes"],
                "v19b_joint_delta": n600["joint_delta_vs_v15_control"]["joint_delta"],
                "scorer_value_oracle_coverage": scorer_value_oracle.coverage_report(
                    verify=False
                )["counts"],
            },
        },
        "sources": {name: _source_public(row) for name, row in sources.items()},
        "source_custody": {
            "dv1_summary": dv1_custody,
            "g3_full_atlas": g3_bulk,
            "input_hashes": dict(sorted(hashes.items())),
            "scorer_value_oracle_rate_row": oracle_rate_row.to_dict(
                include_value=False
            ),
            "selection_rule": "latest run-id timestamp per schema-registered producer family",
            "quarantined_20260717_run_consulted": False,
        },
        "lambda": {
            "producer": lambda_bundle["producer"],
            "producer_schema": lambda_bundle["schema"],
            "producer_status": lambda_bundle["status"],
            "producer_content_sha256": lambda_bundle["content_sha256"],
            "missing_exact_pair_lambda_count": lambda_bundle[
                "missing_exact_pair_lambda_count"
            ],
            "unconsumed_missing_pairs_counted_inert": lambda_bundle[
                "unconsumed_missing_pairs_counted_inert"
            ],
            "law": ("lambda_D2=exact_gap*visibility*uint8_realizability*byte_price*dual_tolerance_D2"),
            "pair_rows": pairs,
            "site_rows": sites,
            "primitive_rows": primitives,
            "block_rows": blocks,
            "backtest": backtest,
            "shared_rate_custody": (
                "EV1 measures one exact V19 global candidate-delta home; per-pair byte "
                "allocation remains null and is never multiplied across pairs. G3 "
                "baseline allocated bytes price pair lambda."
            ),
        },
        "scheduler": scheduler,
        "duties": duties,
        "duties_refreshed": duties_refreshed,
        "arc_evidence": {
            "schema": "ddm_live_costate_arc_evidence.v1",
            "arc": "2026-07-28 describe-line arc (fd1/pp1/rp1/sp1/sc1/ch1)",
            "rows": arc_evidence,
            "present": sum(1 for row in arc_evidence if row.get("available")),
            "registered": len(arc_evidence),
            "evidence_axis": ARC_EVIDENCE_AXIS,
            "actuation": "NONE",
            "score_claim": False,
        },
        "band_position": band_position,
        "band_position_parents": band_parents,
        "sense_laws": sense_laws,
        "open_gate_ownership": open_gate_ownership,
        "round10": round10,
        "pending_producers": pending_producers,
        "conditional_validity": conditional_validity,
        "rv1_table": rv1_table,
        "pn1_nodes": pn1_nodes,
        "duty_allocator": duty_allocator,
        "duties_endgame": duties_endgame,
        "instruments": instruments,
        "campaign": campaign,
        "staleness": {
            "policy": "consumer verifies content hashes; J-of-J radius when emitted, SLA otherwise",
            "source_horizons": {name: row["horizon"] for name, row in sources.items()},
            "j_of_j_radius_available": False,
            "family_local_validity_proxy": v19b["nonadditivity"]["survival_fraction"],
            "rederivation_queue": queue,
            "rederive_command": (
                "python3 tools/ddm_costate_organ.py --write-receipt "
                ".omx/research/ddm_costate_organ_elevation2_20260723/"
                "ddm_costate_organ_elevation2_receipt.json"
            ),
        },
        "resume_state": checkpoint.to_dict(),
        "legacy": {
            "costate_organ_v2_v3": "DOMINATED_STALE",
            "witness_training_rows": "RETAINED_AS_HISTORY_NOT_RANK_ELIGIBLE",
            "reason": "live campaign state is DDM describe-line plus joint recursion",
        },
    }


def digest_lines(report: Mapping[str, Any]) -> list[str]:
    """Render the compact live DDM section used by ``tools/costate_digest.py``."""

    if not report.get("available"):
        return ["DDM-LIVE unavailable: " + ",".join(report.get("missing_required") or ["source error"])]
    live = report["live"]
    reach = live["reach"]["road_described_fraction"]
    box = "IN" if live["box"]["inside_c1_byte_box"] else "OUT"
    fleet = live["fleet"]
    backtest = report["lambda"]["backtest"]
    next_block = report["scheduler"]["next_block"]
    duties = report["duties"]
    refreshed = report.get("duties_refreshed") or {}
    band_pos = report.get("band_position") or {}
    arc = report.get("arc_evidence") or {}
    stale_q = report["staleness"]["rederivation_queue"]
    campaign = report["campaign"]
    campaign_digest = campaign_consumer_view(campaign, "digest")
    campaign_nag = campaign_digest["activation_nag"]
    campaign_evidence = campaign_digest["campaign_evidence"]
    lambda_ranker = campaign_digest["lambda_ranker"]
    lambda_model = lambda_ranker["selected_model"]
    lambda_metrics = lambda_model["metrics"]
    enhancements = campaign_digest["enhancement_activation"]
    plateau = campaign_digest["plateau_route"]
    campaign_next = campaign_nag.get("next_duty") or {}
    lines = [
        (
            f"DDM-LIVE reach={100.0 * reach:.3f}% Road[semantic-cell] box={box} "
            f"fleet={fleet['present']}/{fleet['registered']} "
            f"v19b={fleet['v19b_accepted_moves']} moves "
            f"deltaS={fleet['v19b_joint_delta']:.6f} {EVIDENCE_AXIS}"
        ),
        (
            "DDM-lambda: "
            f"pair/site={len(report['lambda']['pair_rows'])}/"
            f"{len(report['lambda']['site_rows'])} "
            f"heldout={lambda_model['candidate_id']} "
            f"rho={lambda_metrics['spearman_rho']:.3f} "
            f"NDCG@4={lambda_metrics['ndcg_at_4']:.3f}; "
            f"legacy-control rho={backtest['spearman_rho']:.3f} "
            f"NDCG@4={backtest['ndcg_at_4']:.3f}; "
            "shared candidate bytes=MEASURED_EXACT_GLOBAL_HOME"
        ),
        (
            f"DDM-next[pre-arc-scheduler]: {next_block['block_id']} rank={next_block['rank']} "
            f"mode={next_block['lambda_status']} "
            f"GS={next_block['gauss_southwell_validity']:.6g}"
        ),
        (
            "DDM-duty[07-28-refreshed]: "
            + " > ".join(row["duty"] for row in refreshed.get("live_ranked") or [])
            + (
                f"; superseded pre-arc head {refreshed['superseded_legacy_head']} "
                f"({refreshed['demoted'][0]['disposition']})"
                if refreshed.get("demoted")
                else ""
            )
        )
        if refreshed.get("live_ranked")
        else (
            "DDM-duty: "
            + " > ".join(row["duty"] for row in duties["live_ranked"])
            + f"; legacy={duties['current_legacy_rows_retained']} retained DOMINATED"
        ),
        (
            (
                f"DDM-band[pre-arc describe-line base]: d_seg={band_pos['base_d_seg']:.6g} "
                f"regime={band_pos['regime'].upper()} (rho_c={band_pos['rho_c']:.3g}, "
                f"upper={band_pos['band_upper']:.3g}) -> corrections "
                f"{'RATIONAL' if band_pos['correction_duty_multiplier'] else 'DEAD'}: "
                f"{band_pos['correction_class_regime']} "
                "(the LIVE burn endpoint is in DDM-parents, not here)"
            )
            if band_pos.get("available")
            else f"DDM-band: unavailable ({band_pos.get('reason', 'no live base')})"
        ),
        (
            f"DDM-arc[07-28]: {arc.get('present', 0)}/{arc.get('registered', 0)} rows "
            + " ".join(
                f"{row['finding_id']}={row['crux_status']}"
                for row in (arc.get("rows") or [])
                if row.get("available")
            )
        ),
        (
            f"DDM-staleness: hashes={len(report['source_custody']['input_hashes'])} "
            f"rederive-queue={len(stale_q)} maturity={report['maturity']} "
            "actuation=NONE MAIN-review=REQUIRED"
        ),
    ]
    round10 = report.get("round10") or {}
    if round10.get("available"):
        ranking = round10.get("duty_ranking") or []
        lines.append(
            "DDM-round10[current AFR1/DCC1]: "
            f"re-adjudicated={round10.get('re_adjudicated_count', 0)}/"
            f"{round10.get('legacy_open_denominator', 0)} "
            "head="
            + (str(ranking[0]["duty"]) if ranking else "NONE")
            + " maturity=_dev containment=UNCHANGED actuation=NONE"
        )
    # ── co7 lines: endgame chain · SENSE laws · per-parent band · conditional validity ·
    # pending producers. Advisory; every row cites committed content-hashed artifacts.
    chain = report.get("duties_endgame") or {}
    if chain.get("chain"):
        lines.append(
            "DDM-chain[endgame]: "
            + " > ".join(f"{row['duty']}({row['status'].lower()})" for row in chain["chain"])
            + f"; head={chain['head_actionable']} actuation=NONE"
        )
    laws = report.get("sense_laws") or {}
    law_rows = {row["law_id"]: row for row in laws.get("rows") or []}
    if law_rows:
        gate_law = law_rows.get("ema_gate_basis_v1") or {}
        basin_law = law_rows.get("basin_solve_handoff_v1") or {}
        lines.append(
            "DDM-laws: gate-basis=ACTIVE(W=2/(1-d), #85-fixed) "
            f"basin-solve={((basin_law.get('live_evaluation') or {}).get('status', 'ABSENT'))} "
            f"(never-auto-stop={basin_law.get('never_auto_stop')}; "
            f"gate-basis-commit={gate_law.get('code_commit', 'ABSENT')})"
        )
        co9_laws = [
            law_rows.get(lid)
            for lid in (
                "posenet_far_field_photometrics_bidirectional_v1",
                "token_sensitivity_spread_nu_pivot_v1",
                "seg_is_base_quality_white_jitter_v1",
            )
            if law_rows.get(lid)
        ]
        if co9_laws:
            lines.append(
                "DDM-laws[co9 MEASURED]: "
                "pose-far-field=BIDIRECTIONAL (QA43 −0.575S recover / QA06 +0.185S freeze) · "
                "token-sensitivity=35×-spread ν=0.7-in-band -> continuous-log-bit DOMINATES 3-rung "
                "(QA07/12 re-priced) · seg=BASE-QUALITY white-jitter (QA03/04 corrections BREAK-EVEN "
                "at in-band base) [macOS-CPU advisory; not a score]"
            )
        allocator = law_rows.get("joint_exchange_rate_allocator_v1")
        if allocator:
            top = (allocator.get("table_top") or [])[:3]
            lines.append(
                "DDM-joint[ja1 allocator, QA73]: fire on JOINT exchange rate not axis identity; "
                "TOP=" + " | ".join(top)
                + " ; SURPRISE=seg is largest axis but byte-SATURATED at knee -> cheap byte levers "
                "are POSE; seg moves only via HEAVY re-burn (QA24, PARALLEL). "
                f"table={allocator.get('committed_table')} [macOS-CPU advisory; not a score; actuation NONE]"
            )
    parents = report.get("band_position_parents") or {}
    if parents.get("rows"):
        lines.append(
            "DDM-parents: "
            + " · ".join(
                f"{row['parent']}={row['base_d_seg']:.6g}:{str(row['regime']).upper()}"
                for row in parents["rows"]
            )
            + (
                " -> corrections DEAD at every live parent"
                if not parents.get("any_parent_in_band")
                else (
                    " -> tb1_burn_endpoint IN-BAND: correction-class re-grade DUE by band-entry "
                    "(ng1 §2 row 10 re-parent) BUT QA03/QA04 white-jitter = MEASURED BREAK-EVEN "
                    "at this base (seg=base-quality; lower the base, not correct it)"
                )
            )
        )
    validity = report.get("conditional_validity") or {}
    if validity.get("rows") is not None:
        v_rows = validity.get("rows") or []
        broken = [row for row in v_rows if row.get("status") == "BROKEN"]
        regraded = [
            row for row in broken if row.get("disposition") == "RE_GRADED_BY_COMMITTED_SUCCESSOR"
        ]
        v_duties = validity.get("re_grade_duties") or []
        due = sum(1 for row in v_duties if row.get("status") == "DUE")
        owner_state = (
            "consumed-co8"
            if validity.get("table_owner_state") == "CONSUMED_CO8"
            else "pending"
        )
        lines.append(
            f"DDM-validity: preconditions={len(v_rows)} broken={len(broken)} "
            f"re-graded={len(regraded)} re-grade-duties={len(v_duties)} (due={due}) "
            f"live-parent={validity.get('live_parent')} table-owner=rv1[{owner_state}]"
        )
    pending = report.get("pending_producers") or []
    if pending:
        lines.append(
            "DDM-pending: "
            + " ".join(
                f"{row['producer']}="
                + ("COMMITTED" if row.get("available") else "PENDING")
                for row in pending
            )
            + "; numbers uncounted until committed (NO-FAKE)"
        )
    # ── co9 line: OWNERSHIP-ON-GATE-OPEN — DUE/ORPHAN rows whose gate is OPEN but unfired,
    #    surfaced with OWNER + age so open-gate/no-owner is machine-surfaced, not audit-discovered.
    owners = report.get("open_gate_ownership") or {}
    if owners.get("available") and owners.get("open_gate_count"):
        rows = owners.get("open_gate_unfired_rows") or []
        top = ", ".join(
            f"{r['row_id']}(owner={r['owner'][:22]}{'⚠NO-OWNER' if r['no_owner_alarm'] else ''})"
            for r in rows[:6]
        )
        more = owners["open_gate_count"] - min(6, len(rows))
        openers = owners.get("gate_openers") or []
        opener_txt = (
            " | gate-openers: " + "; ".join(f"{o['gate_opener']}->{o['opens'][:40]}" for o in openers)
            if openers
            else ""
        )
        age = owners.get("ledger_age_days")
        age_txt = f"{age:.0f}d" if isinstance(age, (int, float)) else "?"
        lines.append(
            f"DDM-owners[gate-open/unfired]: {owners['open_gate_count']} rows "
            f"(no-owner-alarm={owners['no_owner_alarm_count']}; ledger age {age_txt}): {top}"
            + (f" (+{more})" if more > 0 else "")
            + opener_txt
            + " [advisory routing surface; actuation NONE]"
        )
    # ── co8 lines: rv1 table fold · pn1 VOI/nu-pivot/S1 nodes · allocator waterfill.
    rv1 = report.get("rv1_table") or {}
    if rv1.get("available"):
        counts = rv1["counts"]
        # rx1: report the LIVE discharge split, not rv1's frozen schedulability field. The
        # old line derived "now=$0xN" from `by_status` (a 2026-07-28 constant) and so kept
        # advertising already-discharged duties as free measurements.
        rows = rv1.get("reactivation_rows") or []
        landed = [r["row_id"].split("_")[0] for r in rows if r.get("reactivated")]
        open_rows = [r for r in rows if not r.get("reactivated")]
        open_txt = " ".join(
            f"{r['row_id'].split('_')[0]}={r.get('duty_state', '?')}" for r in open_rows
        )
        lines.append(
            f"DDM-rv1[consumed]: reactivations={counts['reactivations']} "
            f"landed={counts.get('landed', 0)}[{','.join(landed) or '-'}] "
            f"open={counts.get('open', 0)}{' ' + open_txt if open_txt else ''} "
            f"closed={counts['non_reactivations']} "
            f"corrections={len(rv1.get('charter_corrections') or [])} "
            "landed=measurement-landed-NOT-lever-paid (evidence-derived, read disposition)"
        )
    pn1 = report.get("pn1_nodes") or {}
    if pn1.get("available"):
        pivot = pn1["nu_pivot"]
        race = pn1["granularity_race_duty"]
        lines.append(
            "DDM-voi[pn1]: top=S2_NU_AUDIT+R7 ($0, re-routes the sealed burn pre-capacity) "
            f"nu-pivot={pivot['pivot_window']} ({pivot['status']}) "
            f"S1=A:$0-ready B:<$2-operator-GO "
            f"granularity-race={race['status']} no-new-heads"
        )
    alloc = report.get("duty_allocator") or {}
    if alloc.get("available"):
        schedulable = [row for row in alloc["rows"] if row.get("schedulable")]
        lines.append(
            f"DDM-alloc[waterfill]: schedulable={len(schedulable)}/{len(alloc['rows'])} "
            + " > ".join(row["duty"] for row in schedulable)
            + "; pools=token-rate(S2>R7) regret-allocator=GATED(CO5) actuation=NONE"
        )
    lines.extend(
        (
            (
                f"DDM-campaign: verdicts={campaign_digest['verdict_count']} "
                f"plateau={plateau['status']} fork={plateau.get('fork_id') or 'NONE'} "
                f"state={campaign_digest['state_digest'][:12]} actuation=NONE"
            ),
            (
                f"DDM-campaign-SENSE: unmeasured={campaign_nag['unmeasured_sense_rows']}/"
                f"{campaign_nag['standing_sense_rows']} "
                f"blockers={len(campaign_nag['blocker_ids'])} "
                f"next={campaign_next.get('duty') or 'NONE'}"
            ),
            (
                "DDM-campaign-evidence: "
                f"V19={campaign_evidence['v19_receiver_closed_join_status']} "
                f"RD1={campaign_evidence['rd1_dimension_evidence_status']} "
                f"pricing={campaign_evidence['bucket_exchange_rate_status']}"
            ),
            (
                f"DDM-CO5: active={enhancements['active_count']}/"
                f"{enhancements['total_count']} "
                f"re-premised={enhancements.get('re_premised_count', enhancements['held_count'])} "
                f"retired={enhancements.get('retired_count', 0)} "
                f"freshness={enhancements['source_freshness']['tag']} "
                f"gate={enhancements['status']} "
                f"next-gate={(enhancements.get('duty_to_measure') or [{}])[0].get('named_gate', 'NONE')} "
                "actuation=NONE"
                + (
                    " next-producer="
                    + str(
                        (enhancements.get("duty_to_measure") or [{}])[0]["producer_state"]
                    ).split(":", 1)[0]
                    if (enhancements.get("duty_to_measure") or [{}])[0].get("producer_state")
                    else ""
                )
            ),
        )
    )
    return lines


def write_receipt_atomic(path: Path, report: Mapping[str, Any]) -> None:
    """Write one durable advisory receipt atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
