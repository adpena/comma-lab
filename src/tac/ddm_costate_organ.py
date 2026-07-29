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
    """A named in-flight producer whose numbers stay uncounted until committed."""

    name: str
    glob: str  # scanned under .omx/research/
    expectation: str
    named_gate: str


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
            row["status"] = "COMMITTED_ARTIFACT_PRESENT_FOLD_ON_NEXT_ROUND"
        else:
            row["reason"] = "PENDING_COMMITTED_PRODUCER"
        rows.append(row)
    return rows


def _band_position_parents(repo_root: Path, base_band: Mapping[str, Any]) -> dict[str, Any]:
    """Per-parent band placement: W_joint base + the tb1 T2 tr1 full-confirm bases.

    The tr1 arm bases are read machine-readably from the committed sealed ticket
    (``adjudication.arithmetic.{plain,lotto}.full_dseg``), never hardcoded. All bases
    currently above the band upper => corrections stay dead at EVERY live parent.
    """

    from tac.canonical_equations.ddm_pp1_correction_stream_position_band_20260728 import (
        EQUATION_ID,
        position_cost_band,
    )

    rows: list[dict[str, Any]] = []
    if base_band.get("available"):
        rows.append(
            {
                "parent": "W_joint_describe_line",
                "base_d_seg": base_band["base_d_seg"],
                "regime": base_band["regime"],
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


def _sense_laws(arc_index: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The two 07-28-late SENSE laws, anchored to committed artifacts.

    * ``ema_gate_basis_v1`` — instrument-validity precondition (the #85 class):
      gate basis is LIVE params before warmup W = 2/(1-d), EMA shadow after.
    * ``basin_solve_handoff_v1`` — operator-binding preference: on MEASURED
      basin-entry, hand descent to the solve/finisher chain; detectors PRIORITIZE a
      finisher quote (eg1 E2 typed policy), they never auto-stop training.
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
    return {
        "schema": "ddm_costate_sense_laws.v1",
        "rows": [gate_basis, basin],
        "actuation": "NONE",
        "score_claim": False,
    }


def _conditional_validity_review(
    arc_rows: Sequence[Mapping[str, Any]],
    band_parents: Mapping[str, Any],
) -> dict[str, Any]:
    """The substrate-change trigger: evaluate precondition tags against the live state.

    A negative verdict binds only while its preconditions hold. When a precondition
    breaks: a committed successor re-grades it in place (RE_GRADED_BY_COMMITTED_SUCCESSOR),
    otherwise a typed re-grade duty is surfaced (DUE, or ARMED while the band still
    explodes). The FULL historical sweep is owned by the pending rv1 arm; this review
    seeds arc-local rows only.
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
            "rv1_conditional_validity_table (pending producer; this organ seeds arc-local "
            "rows only — no duplicate sweep)"
        ),
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
    sense_laws = _sense_laws(arc_index)
    pending_producers = _pending_producers(repo_root)
    conditional_validity = _conditional_validity_review(arc_evidence, band_parents)
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
        "pending_producers": pending_producers,
        "conditional_validity": conditional_validity,
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
                f"DDM-band: base d_seg={band_pos['base_d_seg']:.6g} "
                f"regime={band_pos['regime'].upper()} (rho_c={band_pos['rho_c']:.3g}, "
                f"upper={band_pos['band_upper']:.3g}) -> corrections "
                f"{'RATIONAL' if band_pos['correction_duty_multiplier'] else 'DEAD'}: "
                f"{band_pos['correction_class_regime']}"
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
                else " -> A PARENT IS IN-BAND: correction-class re-grade DUE"
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
        lines.append(
            f"DDM-validity: preconditions={len(v_rows)} broken={len(broken)} "
            f"re-graded={len(regraded)} re-grade-duties={len(v_duties)} (due={due}) "
            f"live-parent={validity.get('live_parent')} table-owner=rv1[pending]"
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
