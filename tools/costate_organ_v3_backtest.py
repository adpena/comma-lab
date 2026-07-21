#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fixed-n=24 retrospective rank backtest for costate ORGAN v3.

The tool consumes the sealed v2 receipt and r1b7 autopsy.  It can seed the
append-only realized-DeltaS corpus when explicitly requested, but it never
mutates a run, launches work, invokes a scorer, or changes the frontier.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.witness_control.costate_organ_v3 import (  # noqa: E402
    COMPOSITION_EQUATION_ID,
    DEFAULT_CORPUS_PATH,
    EMA_EQUATION_ID,
    POOL_EQUATION_ID,
    R1B7_RECEIPT_SHA256,
    append_realized_delta_row,
    denoise_realized_target,
    load_r1b7_survival,
    load_realized_delta_corpus,
    ndcg_at_k,
    paired_bootstrap_delta,
    pool_kkt_marginals,
    rank_metrics,
    row_from_backtest,
    sha256_file,
    sharpen_realizability_row,
    spearman_from_keys,
    top_k_precision,
)

V2_RECEIPT_SHA256 = "f733187fbc8e69e03d4854a8f45baa0951af4a2807a01bfc1841cffca8d59410"
SCHEMA = "costate_organ_v3_rank_sharpen_backtest.v1"
BOOTSTRAP_SEED = 20260721
BOOTSTRAP_REPLICATES = 10_000
AXIS = "[macOS-CPU advisory] NON-PROMOTABLE"
DESIGN_REALIZABILITY = 11_453.0 / 38_077.0


def _latest_equation_event(equation_id: str, registry: Path) -> dict[str, Any] | None:
    latest = None
    for line in registry.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("equation_id") == equation_id:
            latest = row
    return latest


def _metric_functions() -> dict[str, Callable[[Sequence[Any], Sequence[float], Sequence[float]], float]]:
    return {
        "spearman": lambda keys, targets, _weights: spearman_from_keys(keys, targets),
        "weighted_spearman": lambda keys, targets, weights: spearman_from_keys(keys, targets, weights),
        "top8_precision": lambda keys, targets, _weights: top_k_precision(keys, targets, k=8),
        "decision_ndcg_at_8": lambda keys, targets, weights: ndcg_at_k(keys, targets, weights=weights, k=8),
    }


def _prediction_key(primary: float, *tie_breakers: float) -> tuple[float, ...]:
    values = (float(primary), *(float(value) for value in tie_breakers))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("prediction key must be finite")
    return values


def _registered_law(equation_id: str, registry: Path) -> dict[str, Any]:
    event = _latest_equation_event(equation_id, registry)
    if event is None and equation_id == EMA_EQUATION_ID:
        evaluator = REPO / "src/tac/canonical_equations/evaluators.py"
        dsl = REPO / "src/tac/witness_dsl/curriculum_dsl.py"
        if equation_id not in evaluator.read_text() or equation_id not in dsl.read_text():
            raise ValueError(f"required LawRef is not executable/DSL registered: {equation_id}")
        return {
            "equation_id": equation_id,
            "event_type": "LAWREF_EXECUTABLE_DSL_REGISTERED",
            "callable": "tac.canonical_equations.evaluators:eval_ema_decay_run_geometry",
            "dsl_consumer": "tac.witness_dsl.curriculum_dsl:EmaDecayCalibrated",
            "registry_note": "canonical-equations JSONL row absent; no registration fabricated",
        }
    if event is None:
        raise ValueError(f"required LawRef is not registered: {equation_id}")
    payload = event.get("equation_payload", {})
    return {
        "equation_id": equation_id,
        "event_type": event.get("event_type"),
        "callable": payload.get("python_callable_module_path"),
        "registered_at_utc": event.get("written_at_utc"),
    }


def _stage_delta(
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    strata: Sequence[str],
    replicates: int,
    seed_offset: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric_index, (name, metric) in enumerate(_metric_functions().items()):
        ci = paired_bootstrap_delta(
            before_keys=before["keys"],
            after_keys=after["keys"],
            before_targets=before["targets"],
            after_targets=after["targets"],
            before_weights=before["weights"],
            after_weights=after["weights"],
            strata=strata,
            metric=metric,
            replicates=replicates,
            seed=BOOTSTRAP_SEED + seed_offset * 100 + metric_index,
        )
        lower, upper = ci["ci95"]
        ci["noise_band_verdict"] = (
            "POSITIVE_OUTSIDE_CI_NOISE"
            if lower > 0.0
            else "NEGATIVE_OUTSIDE_CI_NOISE"
            if upper < 0.0
            else "INSIDE_CI_NOISE_NO_IMPROVEMENT_CLAIM"
        )
        output[name] = ci
    return output


def _stage(
    name: str,
    keys: Sequence[tuple[float, ...]],
    targets: Sequence[float],
    weights: Sequence[float],
    ids: Sequence[str],
) -> dict[str, Any]:
    metrics = rank_metrics(keys, targets, weights=weights, ids=ids)
    return {
        "name": name,
        "keys": list(keys),
        "targets": list(targets),
        "weights": list(weights),
        "metrics": metrics,
    }


def _seed_corpus(
    rows: Sequence[Mapping[str, Any]],
    *,
    corpus_path: Path,
    source_receipt: str,
    source_receipt_sha256: str,
) -> dict[str, Any]:
    statuses = []
    for row in rows:
        statuses.append(
            append_realized_delta_row(
                row_from_backtest(
                    row,
                    source_receipt=source_receipt,
                    source_receipt_sha256=source_receipt_sha256,
                ),
                corpus_path,
            )
        )
    loaded = load_realized_delta_corpus(corpus_path)
    expected_ids = [str(row["id"]) for row in rows]
    loaded_ids = [str(row["id"]) for row in loaded if str(row["id"]) in set(expected_ids)]
    if sorted(loaded_ids) != sorted(expected_ids):
        raise ValueError("canonical corpus seed/load row IDs drifted")
    try:
        corpus_ref = str(corpus_path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        corpus_ref = str(corpus_path)
    return {
        "path": corpus_ref,
        "rows_requested": len(rows),
        "rows_materialized": len(statuses),
        "all_append_operations_valid": all(
            row["status"] in {"APPENDED", "EXACT_DUPLICATE_NOOP"} for row in statuses
        ),
        "snapshot_rows_reloaded": len(loaded_ids),
        "snapshot_ids_identical": sorted(loaded_ids) == sorted(expected_ids),
        "rank_effect": "IDENTITY_ZERO_DELTA",
    }


def build_backtest(
    *,
    v2_receipt: Path,
    r1b7_receipt: Path,
    registry: Path,
    created_utc: str,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    try:
        parsed_created = datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("created_utc must be an ISO-8601 timestamp") from exc
    if not created_utc.endswith("Z") or parsed_created.tzinfo is None:
        raise ValueError("created_utc must be UTC with a Z suffix")
    v2_sha = sha256_file(v2_receipt)
    if v2_sha != V2_RECEIPT_SHA256:
        raise ValueError(f"v2 receipt SHA-256 drifted: {v2_sha}")
    if sha256_file(r1b7_receipt) != R1B7_RECEIPT_SHA256:
        raise ValueError("r1b7 receipt SHA-256 drifted")
    v2 = json.loads(v2_receipt.read_text())
    rows = list(v2.get("rows", []))
    if len(rows) != 24 or int(v2.get("n_rows", -1)) != 24:
        raise ValueError("v3 backtest requires the exact fixed n=24 v2 corpus")
    if len({str(row.get("id")) for row in rows}) != 24:
        raise ValueError("v2 corpus row IDs are not unique")

    registry_before = sha256_file(registry)
    laws = {
        POOL_EQUATION_ID: _registered_law(POOL_EQUATION_ID, registry),
        EMA_EQUATION_ID: _registered_law(EMA_EQUATION_ID, registry),
    }
    survival = load_r1b7_survival(r1b7_receipt)
    ids = [str(row["id"]) for row in rows]
    strata = [str(row["corpus"]) for row in rows]
    observed_targets = [float(row["realized_benefit_s"]) for row in rows]
    unit_weights = [1.0 if row.get("apparatus_valid") else 0.0 for row in rows]

    v2_values = [float(row["exact_anchor_v2"]) for row in rows]
    v2_keys = [_prediction_key(value) for value in v2_values]
    graded = [
        sharpen_realizability_row(
            row,
            distribution=survival,
            design_realizability=DESIGN_REALIZABILITY,
        )
        for row in rows
    ]
    graded_values = [float(row["lambda"]) for row in graded]
    graded_keys = []
    for value, result, row in zip(graded_values, graded, rows, strict=True):
        # A zero primary is a structural tie, not license to invent epsilon-S.
        # Resolve it only with already-custodied factor evidence: the r1b7 route
        # probability and exact debt.  Nonzero primary ordering is untouched.
        factor_gap = float(row["factors"]["exact_gap"])
        graded_keys.append(
            _prediction_key(
                value,
                result["base_probability"] if value == 0.0 else 0.0,
                factor_gap if value == 0.0 else 0.0,
            )
        )
    pools = pool_kkt_marginals(rows, graded_values)
    pool_keys = []
    for index, result in enumerate(pools):
        value = float(result["value"])
        factor_gap = float(rows[index]["factors"]["exact_gap"])
        pool_keys.append(
            _prediction_key(
                value,
                result["raw_lambda"],
                graded[index]["base_probability"] if value == 0.0 else 0.0,
                factor_gap if value == 0.0 else 0.0,
            )
        )
    target_rows = [denoise_realized_target(row) for row in rows]
    denoised_targets = [float(result["value"]) for result in target_rows]
    apparatus_weights = [float(result["weight"]) for result in target_rows]

    stages = {
        "v2_baseline": _stage("v2_baseline", v2_keys, observed_targets, unit_weights, ids),
        "graded_realizability": _stage("graded_realizability", graded_keys, observed_targets, unit_weights, ids),
        "pool_interaction": _stage("pool_interaction", pool_keys, observed_targets, unit_weights, ids),
        "target_denoising": _stage("target_denoising", pool_keys, denoised_targets, apparatus_weights, ids),
        "receipt_emission": _stage("receipt_emission", pool_keys, denoised_targets, apparatus_weights, ids),
    }
    comparisons: dict[str, Any] = {}
    order = list(stages)
    for index in range(1, len(order)):
        before_name, after_name = order[index - 1], order[index]
        comparisons[f"{before_name}_to_{after_name}"] = _stage_delta(
            before=stages[before_name],
            after=stages[after_name],
            strata=strata,
            replicates=bootstrap_replicates,
            seed_offset=index,
        )
    for index, name in enumerate(order[1:], start=20):
        comparisons[f"v2_baseline_to_{name}"] = _stage_delta(
            before=stages["v2_baseline"],
            after=stages[name],
            strata=strata,
            replicates=bootstrap_replicates,
            seed_offset=index,
        )

    v2_pools = pool_kkt_marginals(rows, v2_values)
    v2_pool_keys = [
        _prediction_key(
            result["value"],
            result["raw_lambda"],
            0.0,
            float(rows[index]["factors"]["exact_gap"]) if float(result["value"]) == 0.0 else 0.0,
        )
        for index, result in enumerate(v2_pools)
    ]
    no_graded = _stage(
        "ablate_graded_realizability",
        v2_pool_keys,
        denoised_targets,
        apparatus_weights,
        ids,
    )
    no_pool = _stage(
        "ablate_pool_interaction",
        graded_keys,
        denoised_targets,
        apparatus_weights,
        ids,
    )
    no_ema = _stage("ablate_ema_delag", pool_keys, observed_targets, apparatus_weights, ids)
    no_weight = _stage("ablate_apparatus_weighting", pool_keys, denoised_targets, unit_weights, ids)
    final = stages["target_denoising"]
    ablations = {
        "graded_realizability": {
            "without": no_graded["metrics"],
            "with": final["metrics"],
            "delta_ci": _stage_delta(
                before=no_graded,
                after=final,
                strata=strata,
                replicates=bootstrap_replicates,
                seed_offset=40,
            ),
        },
        "pool_interaction": {
            "without": no_pool["metrics"],
            "with": final["metrics"],
            "delta_ci": _stage_delta(
                before=no_pool,
                after=final,
                strata=strata,
                replicates=bootstrap_replicates,
                seed_offset=41,
            ),
        },
        "ema_delag": {
            "without": no_ema["metrics"],
            "with": final["metrics"],
            "delta_ci": _stage_delta(
                before=no_ema,
                after=final,
                strata=strata,
                replicates=bootstrap_replicates,
                seed_offset=42,
            ),
        },
        "apparatus_weighting": {
            "without": no_weight["metrics"],
            "with": final["metrics"],
            "delta_ci": _stage_delta(
                before=no_weight,
                after=final,
                strata=strata,
                replicates=bootstrap_replicates,
                seed_offset=43,
            ),
        },
        "receipt_emission": {
            "without": final["metrics"],
            "with": stages["receipt_emission"]["metrics"],
            "delta_ci": comparisons["target_denoising_to_receipt_emission"],
            "expected_rank_effect": "IDENTITY_ZERO_DELTA",
        },
    }

    pool_members: dict[str, list[str]] = {}
    for identifier, result in zip(ids, pools, strict=True):
        for pool in result["claims"]:
            pool_members.setdefault(pool, []).append(identifier)
    shared_pools = {pool: members for pool, members in pool_members.items() if len(members) > 1}
    v2_rank = {
        identifier: rank
        for rank, identifier in enumerate(sorted(ids, key=lambda item: v2_keys[ids.index(item)], reverse=True), start=1)
    }
    pool_rank = {
        identifier: rank
        for rank, identifier in enumerate(
            sorted(ids, key=lambda item: pool_keys[ids.index(item)], reverse=True), start=1
        )
    }
    rank_moves = [
        {
            "id": identifier,
            "v2_rank": v2_rank[identifier],
            "pool_rank": pool_rank[identifier],
            "delta": v2_rank[identifier] - pool_rank[identifier],
        }
        for identifier in ids
        if any(identifier in members for members in shared_pools.values())
    ]

    registry_after = sha256_file(registry)
    target_disagreement = (
        "DISAGREE"
        if stages["target_denoising"]["metrics"]["spearman"] > stages["v2_baseline"]["metrics"]["spearman"]
        and (
            stages["target_denoising"]["metrics"]["top8_precision"]
            <= stages["v2_baseline"]["metrics"]["top8_precision"]
            or stages["target_denoising"]["metrics"]["decision_ndcg_at_8"]
            <= stages["v2_baseline"]["metrics"]["decision_ndcg_at_8"]
        )
        else "AGREE_ON_DIRECTION"
    )
    return {
        "schema": SCHEMA,
        "created_utc": created_utc,
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_changed": False,
        "actuation": "NONE",
        "learned_parameters": 0,
        "n_rows": 24,
        "row_ids": ids,
        "source_custody": {
            "v2_receipt": str(v2_receipt),
            "v2_receipt_sha256": v2_sha,
            "r1b7_receipt": str(r1b7_receipt),
            "r1b7_receipt_sha256": R1B7_RECEIPT_SHA256,
            "registry_sha256_before": registry_before,
            "registry_sha256_after": registry_after,
            "registry_bytes_unchanged": registry_before == registry_after,
        },
        "law_refs": laws,
        "survival_distribution": {
            **survival.__dict__,
            "histogram": dict(survival.histogram),
        },
        "ema_lag_model": {
            "equation_id": EMA_EQUATION_ID,
            "model": "observed_EMA_delta=(1-d**h)*latent_live_delta",
            "inverse": "latent_live_delta=observed_EMA_delta/(1-d**h)",
            "variance_weight": "(1-d**h)**2",
            "decay": 0.997,
            "scope": "only n205 rows with integer source-epoch horizons; target side only",
        },
        "stages": {
            name: {key: value for key, value in stage.items() if key != "keys"} for name, stage in stages.items()
        },
        "prediction_keys": {name: stage["keys"] for name, stage in stages.items()},
        "comparisons": comparisons,
        "ablations": ablations,
        "row_readback": [
            {
                "id": ids[index],
                "graded_realizability": graded[index],
                "pool": pools[index],
                "target": target_rows[index],
            }
            for index in range(24)
        ],
        "pool_interactions": {
            "equation_id": POOL_EQUATION_ID,
            "shared_pools": shared_pools,
            "shared_pool_row_count": len({item for members in shared_pools.values() for item in members}),
            "relative_rank_moves": sorted(rank_moves, key=lambda row: (-abs(row["delta"]), row["id"])),
        },
        "top_metric_direction": target_disagreement,
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed_base": BOOTSTRAP_SEED,
            "confidence": 0.95,
            "method": "paired stratum-preserving percentile bootstrap",
        },
        "verdict_scope": (
            "RETROSPECTIVE_DEVELOPMENT on the sealed 24 rows only: two #205 temporal-stop "
            "rows plus 22 C2 stride-5 carrier smokes. No live-run, n600, family, contest-axis, "
            "or promotion inference. Byte-price remains untested because all 24 rows are byte-free."
        ),
        "composition_equation_id": COMPOSITION_EQUATION_ID,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-receipt", type=Path, required=True)
    parser.add_argument("--r1b7-receipt", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REPO / ".omx/state/canonical_equations_registry.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--created-utc", required=True, help="custodied UTC timestamp ending in Z")
    parser.add_argument("--seed-corpus", action="store_true")
    parser.add_argument("--corpus", type=Path, default=REPO / DEFAULT_CORPUS_PATH)
    args = parser.parse_args(argv)
    payload = build_backtest(
        v2_receipt=args.v2_receipt,
        r1b7_receipt=args.r1b7_receipt,
        registry=args.registry,
        created_utc=args.created_utc,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    if args.seed_corpus:
        v2 = json.loads(args.v2_receipt.read_text())
        payload["receipt_emission"] = _seed_corpus(
            v2["rows"],
            corpus_path=args.corpus,
            source_receipt=str(args.v2_receipt),
            source_receipt_sha256=payload["source_custody"]["v2_receipt_sha256"],
        )
    else:
        payload["receipt_emission"] = {
            "status": "NOT_REQUESTED",
            "rank_effect": "IDENTITY_ZERO_DELTA",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    final = payload["stages"]["target_denoising"]["metrics"]
    print(json.dumps({"output": str(args.output), "n_rows": payload["n_rows"], **final}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
