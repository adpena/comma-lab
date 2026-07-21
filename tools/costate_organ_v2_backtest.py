#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only retrospective rank backtest for exact-anchor costate ORGAN v2.

The corpus is the real #205 as-of trajectory plus the two C2 carrier-smoke
families.  Input bytes are hashed before and after.  Results are advisory and
retrospective; the tool cannot mutate a run or authorize a recommendation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.witness_control.costate_organ_v2 import (  # noqa: E402
    AXIS,
    BREAKEVEN_EQUATION_ID,
    DESIGN_REALIZABILITY,
    FISHER_BANK_SCHEMA,
    FISHER_BANK_SHA256,
    FULL_KERNEL_VISIBLE,
    POOL_KKT_EQUATION_ID,
    byte_price_factor,
    compose_lambda,
    latest_equation_event,
    realizability_factor,
)
from tac.witness_control.shadow_controller import build_shadow_report, load_run_inputs  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in order[i:j]:
            ranks[k] = rank
        i = j
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    am = sum(a) / len(a)
    bm = sum(b) / len(b)
    da = [v - am for v in a]
    db = [v - bm for v in b]
    den = math.sqrt(sum(v * v for v in da) * sum(v * v for v in db))
    return (0.0 if den == 0.0 else
            sum(x * y for x, y in zip(da, db, strict=True)) / den)


def spearman(a: list[float], b: list[float]) -> float:
    return _pearson(_average_ranks(a), _average_ranks(b))


def _variant_key(row: dict[str, Any]) -> str:
    return f"{row['variant']}:{float(row.get('beta', 0.0)):.2f}"


def _read_smokes(path: Path) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    rows: dict[str, dict[str, Any]] = {}
    files = sorted(path.glob("smoke_*.json"))
    for p in files:
        row = json.loads(p.read_text())
        rows[_variant_key(row)] = row
    return rows, files


def _c2_rows(source_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    palette, palette_files = _read_smokes(
        source_root / "experiments/results/c2_perclass_stratum_20260716")
    witness, witness_files = _read_smokes(
        source_root / "experiments/results/c2_witness_own_decomp_20260716")
    out: list[dict[str, Any]] = []

    def baseline(rows: dict[str, dict[str, Any]], fallback: float) -> float:
        return next((r["dseg_subset"] for r in rows.values()
                     if r["variant"] == "baseline"), fallback)

    # The palette receipt set records its baseline in the dated memo (0.013044)
    # but has no redundant smoke_baseline JSON; the witness set does carry one.
    palette_base = baseline(palette, 0.013044)
    witness_base = baseline(witness, 0.0031067742241753468)
    legacy_marginal = {
        # Explicit intervention-vocabulary projection of the persisted current
        # DECIDE duty rows; variants with no typed mapping remain 0/UNIDENTIFIABLE.
        "oneside_lane": (4.203e-4, "lane_edge"),
        "oneside_movable": (3.059e-4, "persistence"),
        "oneside_shallow": (3.298e-4, "horizon_margin"),
        "movable_meancolor": (3.059e-4, "persistence"),
    }
    exact_pool = {
        "oneside_lane": 0.001098,
        "oneside_movable": 0.003222 + 0.001101 + 0.000614,
        "oneside_shallow": 0.001098 + 0.003222 + 0.001101 + 0.000614,
        "movable_meancolor": 0.003222 + 0.001101 + 0.000614,
    }

    for vehicle, rows, base in (("palette", palette, palette_base),
                                ("trained_witness", witness, witness_base)):
        for key, row in sorted(rows.items()):
            variant = row["variant"]
            if variant == "baseline":
                continue
            beta = float(row.get("beta", 0.5))
            charged = int(row.get("counted_bytes", 0))
            realized_benefit = 100.0 * (base - float(row["dseg_subset"]))
            old_base, old_lever = legacy_marginal.get(
                variant, (0.0, "UNIDENTIFIABLE_NOT_MAPPED"))
            old = old_base * beta * 100.0
            pool_gap = exact_pool.get(variant, 0.0001) * 100.0
            correct_side = variant in {"oneside_lane", "oneside_movable", "oneside_shallow"}
            align = 1.0 if correct_side else 0.0
            formulation_valid = vehicle == "palette"
            r = realizability_factor(
                formulation_valid=formulation_valid,
                strength=min(beta / 2.0, 1.0),
            )
            recovery_for_rate = max(pool_gap, 0.0)
            bp = byte_price_factor(realized_recovery_s=recovery_for_rate,
                                   charged_bytes=charged)
            factors = compose_lambda(
                exact_gap=pool_gap,
                visibility=FULL_KERNEL_VISIBLE * align,
                realizability=r["value"],
                byte_price=bp["value"],
            )
            out.append({
                "id": f"c2:{vehicle}:{key}",
                "corpus": "c2_carrier_smoke",
                "vehicle": vehicle,
                "variant": variant,
                "beta": beta,
                "old_decide": old,
                "old_decide_readback": {
                    "equation_id": "factorized_duty_marginal_projected_v1",
                    "mapped_lever": old_lever,
                    "mapping_scope": "explicit C2 intervention-vocabulary projection",
                    "unmapped_is_zero_not_guessed": old_lever == "UNIDENTIFIABLE_NOT_MAPPED",
                },
                "exact_anchor_v2": factors.lambda_value,
                "realized_benefit_s": realized_benefit,
                "factors": factors.to_dict(),
                "factor_context": {
                    "direction_alignment": align,
                    "formulation_valid": formulation_valid,
                    "charged_bytes": charged,
                    "design_realizability": DESIGN_REALIZABILITY,
                },
                "apparatus_valid": True,
                "maturity": "_dev",
                "source_scope": row.get("scope"),
            })
    return out, palette_files + witness_files


def _implied_s(row: dict[str, Any]) -> float:
    if isinstance(row.get("implied_S"), (int, float)):
        return float(row["implied_S"])
    return 100.0 * float(row["d_seg"]) + math.sqrt(10.0 * float(row["d_pose"]))


def _n205_rows(source_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    run = source_root / "experiments/results/levelset_v752_baseline_20260710T185913Z"
    full = load_run_inputs(run)
    verdicts = [r for r in full.verdicts if isinstance(r.get("epoch"), (int, float))]
    out = []
    for epoch in (350, 400, 450):
        report = build_shadow_report(load_run_inputs(run, as_of_epoch=epoch))
        stop = next((r for r in report.recommendations if r.get("action") == "STOP_TRAINING"), None)
        current = max((r for r in verdicts if float(r["epoch"]) <= epoch),
                      key=lambda r: float(r["epoch"]), default=None)
        future = min((r for r in verdicts if float(r["epoch"]) >= epoch + 25),
                     key=lambda r: float(r["epoch"]), default=None)
        if stop is None or current is None or future is None:
            continue
        old_benefit = -float(stop["predicted_dS"])
        realized_benefit = _implied_s(future) - _implied_s(current)
        factors = compose_lambda(exact_gap=old_benefit, visibility=1.0,
                                 realizability=1.0, byte_price=1.0)
        out.append({
            "id": f"n205:stop_ep{epoch}",
            "corpus": "#205_asof_trajectory",
            "old_decide": old_benefit,
            "exact_anchor_v2": factors.lambda_value,
            "realized_benefit_s": realized_benefit,
            "factors": factors.to_dict(),
            "factor_context": {
                "temporal_action": "STOP_TRAINING",
                "ema_lag_correction": "unknown_not_applied",
                "edge_flicker_temporal_duty": True,
            },
            "apparatus_valid": str(full.flags.get("ckpt-every", "")) != "1",
            "maturity": "_dev",
            "source_epochs": [float(current["epoch"]), float(future["epoch"])],
        })
    design = source_root / ".omx/research/costate_controller_design_20260705.md"
    # The canonical #205 directory is a live/truncated custody surface in some
    # checkouts (currently ending ep225).  The dated design memo preserves the
    # measured ep350/ep450 as-of replay and future-25ep realization.  Use those
    # two literal receipt rows only when the underlying future verdicts are absent.
    if not out and design.is_file():
        for epoch, predicted, realized in ((350, 0.0825, 0.0119),
                                           (450, 0.0060, 0.0004)):
            factors = compose_lambda(exact_gap=predicted, visibility=1.0,
                                     realizability=1.0, byte_price=1.0)
            out.append({
                "id": f"n205:memo_stop_ep{epoch}",
                "corpus": "#205_asof_trajectory",
                "old_decide": predicted,
                "exact_anchor_v2": factors.lambda_value,
                "realized_benefit_s": realized,
                "factors": factors.to_dict(),
                "factor_context": {
                    "temporal_action": "STOP_TRAINING",
                    "ema_lag_correction": "unknown_not_applied",
                    "edge_flicker_temporal_duty": True,
                    "receipt_fallback": "costate_controller_design_20260705.md",
                },
                "apparatus_valid": str(full.flags.get("ckpt-every", "")) != "1",
                "maturity": "_dev",
                "source_epochs": [epoch, epoch + 25],
            })
    files = [p for p in (run / "run.log", run / "launch.sh", design) if p.is_file()]
    return out, files


def _correlations(rows: list[dict[str, Any]]) -> dict[str, float]:
    kept = [r for r in rows if r["apparatus_valid"]]
    truth = [r["realized_benefit_s"] for r in kept]
    out = {
        "old_decide_vs_realized": spearman([r["old_decide"] for r in kept], truth),
        "exact_anchor_v2_vs_realized": spearman([r["exact_anchor_v2"] for r in kept], truth),
    }
    names = ("exact_gap", "visibility", "realizability", "byte_price")
    for omitted in names:
        pred = []
        for row in kept:
            f = row["factors"]
            value = 1.0
            for name in names:
                value *= 1.0 if name == omitted else float(f[name])
            pred.append(value)
        out[f"ablate_{omitted}_vs_realized"] = spearman(pred, truth)
    return out


def _fisher_bank_preflight(path: Path) -> dict[str, Any]:
    out = {"path": str(path), "exists": path.is_file(), "sha256": None,
           "sha256_matches": False, "schema": None, "schema_matches": False,
           "ranking_use": "site-level optional; aggregate historical rows have no bank site key"}
    if not path.is_file():
        return out
    out["sha256"] = _sha256(path)
    out["sha256_matches"] = out["sha256"] == FISHER_BANK_SHA256
    try:
        import brotli

        first = json.loads(brotli.decompress(path.read_bytes()).splitlines()[0])
        out["schema"] = first.get("schema")
        out["schema_matches"] = out["schema"] == FISHER_BANK_SCHEMA
        out["header"] = first
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def build_backtest(source_root: Path) -> dict[str, Any]:
    c2, c2_files = _c2_rows(source_root)
    n205, n205_files = _n205_rows(source_root)
    registry = REPO / ".omx/state/canonical_equations_registry.jsonl"
    fisher_path = Path(
        "/Volumes/VertigoDataTier/pact/evidence/r1b5_row_closer_20260720/"
        "fisher_ev/fisher_ev_ordering_38077.jsonl.br")
    fisher = _fisher_bank_preflight(fisher_path)
    break_event = latest_equation_event(BREAKEVEN_EQUATION_ID, registry)
    pool_event = latest_equation_event(POOL_KKT_EQUATION_ID, registry)
    preflight = {
        "break_even_equation_id": BREAKEVEN_EQUATION_ID,
        "break_even_latest_event": (break_event or {}).get("event_type"),
        "break_even_domain_refined": (break_event or {}).get("event_type") == "domain_refined",
        "pool_kkt_equation_id": POOL_KKT_EQUATION_ID,
        "pool_kkt_registered": pool_event is not None,
        "dedicated_opportunity_pool_equation": "FORMALIZATION_PENDING_NOT_REGISTERED",
        "fisher_bank": fisher,
    }
    files = sorted({*c2_files, *n205_files, registry,
                    *([fisher_path] if fisher_path.is_file() else [])})
    before = {str(p): _sha256(p) for p in files}
    rows = n205 + c2
    corr = _correlations(rows)
    after = {str(p): _sha256(p) for p in files}
    passed = (
        corr["exact_anchor_v2_vs_realized"] > corr["old_decide_vs_realized"]
        and preflight["break_even_domain_refined"]
        and preflight["pool_kkt_registered"]
        and fisher["sha256_matches"]
        and fisher["schema_matches"]
    )
    return {
        "schema": "costate_organ_v2_exact_anchor_backtest.v1",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "axis": AXIS,
        "score_claim": False,
        "learned_parameters": 0,
        "pointer_changed": False,
        "actuation": "NONE",
        "validation_scope": "RETROSPECTIVE_DEVELOPMENT; no live generalization/promotion authority",
        "old_decide_scope": (
            "#205 uses emitted STOP projections; C2 uses an explicit projection of persisted "
            "factorized_duty_marginal_projected_v1 rows into the intervention vocabulary; "
            "unmapped variants are UNIDENTIFIABLE and score zero, never a fabricated marginal"),
        "n_rows": len(rows),
        "correlations": corr,
        "acceptance": {
            "requires_v2_gt_old": True,
            "passed": passed,
        },
        "preflight": preflight,
        "rows": rows,
        "source_custody": {
            "sha256_before": before,
            "sha256_after": after,
            "bytes_unchanged": before == after,
        },
        "verdict_scope": (
            "#205 supplies temporal stop rows; C2 supplies stride-5 ranking-only carrier smokes. "
            "Trained-witness flat-band negatives are formulation-scoped; no family negative inferred."
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    payload = build_backtest(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **payload["correlations"],
                      "passed": payload["acceptance"]["passed"]}, sort_keys=True))
    return 0 if payload["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
