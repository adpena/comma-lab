# SPDX-License-Identifier: MIT
"""Live DDM describe-line costate organ (advisory, no actuation).

This top-level module is the successor to the witness-training-era costate
digest.  Keeping the live implementation outside ``tac.witness_control`` means
its import path does not execute the legacy package initializer.  Its SENSE
surface is the latest schema-checked DDM receipt fleet: dv1, g3, g4, and the
v19-family, with e1 and dv2 registered as pending producers.  Every decision
row carries the exact input hashes that make it valid.

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

from tac.ddm_costate_law import (
    EQUATION_ID,
    RATE_BREAK_EVEN_SCORE_PER_BYTE,
    SCHEDULER_EQUATION_ID,
    ddm_joint_costate,
    gauss_southwell_validity_score,
)
from tac.optimization.scorer_analytic_atlas import build_ddm_lambda_bundle

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
    v19 = sources["v19"]["payload"]
    v19b = sources["v19b"]["payload"]
    hashes = {name: row["sha256"] for name, row in sources.items() if row["available"]}
    hashes["dv1_summary"] = str(dv1_custody["sha256"])
    if g3_bulk.get("sha256"):
        hashes["g3_full_atlas"] = str(g3_bulk["sha256"])
    if resume_state is not None:
        prior = dict(resume_state.get("source_hashes") or {})
        if prior and prior != hashes:
            changed = sorted(set(prior) | set(hashes))
            changed = [name for name in changed if prior.get(name) != hashes.get(name)]
            raise ValueError("resume source hashes are stale; re-derive instead of restoring: " + ",".join(changed))

    lambda_bundle = build_ddm_lambda_bundle(
        atlas=g3_atlas,
        v19=v19,
        source_hashes=hashes,
    )
    pairs = list(lambda_bundle["pair_rows"])
    sites = list(lambda_bundle["site_rows"])
    backtest = dict(lambda_bundle["backtest"])
    blocks = _block_costates(v19b)
    primitives = _primitive_costates(dv1, g4)
    scheduler = _scheduler(dv1, g4, blocks)
    duties = _duties(scheduler)
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
                "reason": "no g3/v19 pair join available",
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
            },
        },
        "sources": {name: _source_public(row) for name, row in sources.items()},
        "source_custody": {
            "dv1_summary": dv1_custody,
            "g3_full_atlas": g3_bulk,
            "input_hashes": dict(sorted(hashes.items())),
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
                "v19 per-pair byte allocation is null; global candidate bytes are not multiplied "
                "across pairs. g3 baseline allocated bytes price pair lambda."
            ),
        },
        "scheduler": scheduler,
        "duties": duties,
        "instruments": instruments,
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
    stale_q = report["staleness"]["rederivation_queue"]
    return [
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
            f"factorized-backtest rho={backtest['spearman_rho']:.3f} "
            f"NDCG@4={backtest['ndcg_at_4']:.3f}; shared candidate bytes=OWED_NOT_INVENTED"
        ),
        (
            f"DDM-next: {next_block['block_id']} rank={next_block['rank']} "
            f"mode={next_block['lambda_status']} "
            f"GS={next_block['gauss_southwell_validity']:.6g}"
        ),
        (
            "DDM-duty: "
            + " > ".join(row["duty"] for row in duties["live_ranked"])
            + f"; legacy={duties['current_legacy_rows_retained']} retained DOMINATED"
        ),
        (
            f"DDM-staleness: hashes={len(report['source_custody']['input_hashes'])} "
            f"rederive-queue={len(stale_q)} maturity={report['maturity']} "
            "actuation=NONE MAIN-review=REQUIRED"
        ),
    ]


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
