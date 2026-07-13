#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reconcile the landed SFESS probe with verified ICLR 2025 equations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.sfess_cached_replay import (  # noqa: E402
    CountedCachedOracle,
    SFESSFixedKSearch,
    load_cached_objective_jsonl,
)
from tac.sfess_oss_reconciliation import run_learned_logit_sfess  # noqa: E402

TABLE = REPO / (
    "experiments/results/ugc_terminal_polish_ab_20260712/"
    "search_exact_enumeration_accepted_proposals.jsonl"
)
CLEAN_RECEIPT = REPO / (
    "experiments/results/sfess_cached_replay_ugc64_20260712T214520Z/"
    "measurement_receipt.json"
)
SAMPLE_COUNTS = (2, 4, 5, 8, 16, 32)
K_VALUES = (1, 2, 3, 4, 5)
EVAL_BUDGET = 64
SEED = 396_400
LEARNING_RATE = 1.0e-4
NOISE_FLOOR_S = 1.0e-12
SCHEMA = "sfess_oss_reconciliation_ugc64.v1"

_CLEAN_ARM_FIELDS = frozenset({
    "arm",
    "k",
    "samples_per_gradient",
    "best_mask",
    "best_s",
    "function_evals",
    "accepted_swaps",
    "padding_calls",
})
_ENRICHED_ARM_FIELDS = frozenset({
    "arm",
    "k",
    "samples_per_gradient",
    "best_mask",
    "best_s",
    "function_evals",
    "gradient_steps",
    "accepted_optimizer_updates",
    "rejected_optimizer_updates",
    "strict_gate_calls",
    "zero_variance_skips",
    "padding_calls",
    "min_sampled_value_spread_s",
    "max_sampled_value_spread_s",
    "final_logits",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _config() -> dict[str, Any]:
    source_paths = (
        Path(__file__).resolve(),
        REPO / "src/tac/sfess_oss_reconciliation.py",
        REPO / "src/tac/sfess_cached_replay.py",
    )
    return {
        "table_sha256": _sha256(TABLE),
        "clean_receipt_sha256": _sha256(CLEAN_RECEIPT),
        "execution_source_sha256": {
            path.relative_to(REPO).as_posix(): _sha256(path) for path in source_paths
        },
        "sample_counts": list(SAMPLE_COUNTS),
        "k_values": list(K_VALUES),
        "eval_budget": EVAL_BUDGET,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "noise_floor_s": NOISE_FLOOR_S,
    }


def _run_arm(k: int, samples: int) -> dict[str, Any]:
    table = load_cached_objective_jsonl(TABLE)
    oracle = CountedCachedOracle(table, EVAL_BUDGET, authorize_lookup=lambda _mask: True)
    result = run_learned_logit_sfess(
        oracle,
        k=k,
        samples_per_gradient=samples,
        seed=SEED,
        learning_rate=LEARNING_RATE,
        comparison_noise_floor_s=NOISE_FLOOR_S,
    )
    return {
        "arm": f"learned_logits_k{k}_m{samples}",
        "k": k,
        "samples_per_gradient": samples,
        "best_mask": list(result.best_mask),
        "best_s": result.best_value,
        "function_evals": result.calls,
        "gradient_steps": result.gradient_steps,
        "accepted_optimizer_updates": result.accepted_optimizer_updates,
        "rejected_optimizer_updates": result.rejected_optimizer_updates,
        "strict_gate_calls": result.strict_gate_calls,
        "zero_variance_skips": result.zero_variance_skips,
        "padding_calls": result.padding_calls,
        "min_sampled_value_spread_s": min(result.sampled_value_spreads, default=0.0),
        "max_sampled_value_spread_s": max(result.sampled_value_spreads, default=0.0),
        "final_logits": list(result.final_logits),
    }


def _run_clean_control(k: int, snapshot_path: Path) -> dict[str, Any]:
    """Rerun the landed clean-room arm inside this comparison envelope."""

    table = load_cached_objective_jsonl(TABLE)
    oracle = CountedCachedOracle(table, EVAL_BUDGET, authorize_lookup=lambda _mask: True)
    result = SFESSFixedKSearch(
        oracle,
        n_bits=table.n_bits,
        k=k,
        samples_per_gradient=5,
        seed=SEED,
        comparison_noise_floor_s=NOISE_FLOOR_S,
    ).run(EVAL_BUDGET, snapshot_path)
    return {
        "arm": f"clean_room_k{k}_m5",
        "k": k,
        "samples_per_gradient": 5,
        "best_mask": list(result.best_mask),
        "best_s": result.best_value,
        "function_evals": result.calls,
        "accepted_swaps": result.accepted,
        "padding_calls": result.padding,
    }


def _validate_partial_checkpoint(checkpoint: dict[str, Any]) -> None:
    """Reject malformed or ambiguous resumed rows before completing the ladder."""

    expected_top_level = {"schema", "config", "clean_arms", "arms"}
    if not isinstance(checkpoint, dict) or set(checkpoint) != expected_top_level:
        raise RuntimeError("partial checkpoint top-level schema mismatch")
    if not isinstance(checkpoint["clean_arms"], list) or not isinstance(checkpoint["arms"], list):
        raise RuntimeError("partial checkpoint arm collections must be lists")

    allowed_clean = {f"clean_room_k{k}_m5": (k, 5) for k in K_VALUES}
    allowed_enriched = {
        f"learned_logits_k{k}_m{samples}": (k, samples)
        for samples in SAMPLE_COUNTS
        for k in K_VALUES
    }

    def validate_rows(
        rows: list[Any],
        *,
        label: str,
        fields: frozenset[str],
        allowed: dict[str, tuple[int, int]],
    ) -> None:
        seen: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != fields:
                raise RuntimeError(f"partial checkpoint {label}[{index}] field schema mismatch")
            arm = row["arm"]
            if not isinstance(arm, str) or arm not in allowed:
                raise RuntimeError(f"partial checkpoint {label}[{index}] has unknown arm identity")
            if arm in seen:
                raise RuntimeError(f"partial checkpoint {label} has duplicate arm identity: {arm}")
            seen.add(arm)
            expected_k, expected_samples = allowed[arm]
            if row["k"] != expected_k or row["samples_per_gradient"] != expected_samples:
                raise RuntimeError(f"partial checkpoint {arm} k/M identity mismatch")

            mask = row["best_mask"]
            if (
                not isinstance(mask, list)
                or not mask
                or any(isinstance(bit, bool) or bit not in (0, 1) for bit in mask)
                or sum(mask) != expected_k
            ):
                raise RuntimeError(f"partial checkpoint {arm} best_mask is invalid")
            if isinstance(row["best_s"], (bool, str)):
                raise RuntimeError(f"partial checkpoint {arm} best_s is not finite numeric")
            try:
                best_s = float(row["best_s"])
            except (TypeError, ValueError, OverflowError) as error:
                raise RuntimeError(
                    f"partial checkpoint {arm} best_s is not finite numeric"
                ) from error
            if not math.isfinite(best_s):
                raise RuntimeError(f"partial checkpoint {arm} best_s is not finite numeric")

            integer_fields = fields - {
                "arm",
                "best_mask",
                "best_s",
                "min_sampled_value_spread_s",
                "max_sampled_value_spread_s",
                "final_logits",
            }
            if any(
                not isinstance(row[name], int)
                or isinstance(row[name], bool)
                or row[name] < 0
                for name in integer_fields
            ):
                raise RuntimeError(f"partial checkpoint {arm} has invalid integer counters")

            if label == "enriched arms":
                spreads = (
                    row["min_sampled_value_spread_s"],
                    row["max_sampled_value_spread_s"],
                )
                if any(
                    isinstance(value, (bool, str))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                    for value in spreads
                ):
                    raise RuntimeError(f"partial checkpoint {arm} has invalid sampled spread")
                logits = row["final_logits"]
                if (
                    not isinstance(logits, list)
                    or len(logits) != len(mask)
                    or any(
                        isinstance(value, (bool, str)) or not math.isfinite(float(value))
                        for value in logits
                    )
                ):
                    raise RuntimeError(f"partial checkpoint {arm} final_logits are invalid")

    validate_rows(
        checkpoint["clean_arms"],
        label="clean arms",
        fields=_CLEAN_ARM_FIELDS,
        allowed=allowed_clean,
    )
    validate_rows(
        checkpoint["arms"],
        label="enriched arms",
        fields=_ENRICHED_ARM_FIELDS,
        allowed=allowed_enriched,
    )


def _recompute_complete_checkpoint(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Re-derive every terminal arm from source and input custody without preserving scratch."""

    with tempfile.TemporaryDirectory(prefix=".resume_auth_", dir=output_dir) as scratch:
        scratch_dir = Path(scratch)
        clean_arms = [
            _run_clean_control(k, scratch_dir / f"clean_k{k}_snapshot.json") for k in K_VALUES
        ]
    arms = [
        _run_arm(k, samples)
        for samples in SAMPLE_COUNTS
        for k in K_VALUES
    ]
    return {"schema": SCHEMA, "config": config, "clean_arms": clean_arms, "arms": arms}


def _build_receipt(
    checkpoint: dict[str, Any],
    *,
    generated_at_utc: str,
    runtime: dict[str, str],
) -> dict[str, Any]:
    """Derive all terminal authority fields from an authenticated complete checkpoint."""

    try:
        datetime.fromisoformat(generated_at_utc)
    except (TypeError, ValueError) as error:
        raise RuntimeError("terminal generated_at_utc is invalid") from error
    if not isinstance(runtime, dict) or set(runtime) != {"python", "platform"} or not all(
        isinstance(runtime[key], str) and runtime[key] for key in ("python", "platform")
    ):
        raise RuntimeError("terminal runtime metadata is invalid")

    clean = json.loads(CLEAN_RECEIPT.read_text(encoding="utf-8"))
    clean_source_best = float(clean["best_non_degenerate_sfess_s"])
    clean_rerun = min(checkpoint["clean_arms"], key=lambda row: row["best_s"])
    clean_best = float(clean_rerun["best_s"])
    if clean_best != clean_source_best:
        raise RuntimeError(
            f"clean control drift: source={clean_source_best!r}, rerun={clean_best!r}"
        )
    exact_best = float(clean["exact_enumeration_s"])
    best = min(checkpoint["arms"], key=lambda row: row["best_s"])
    official_n32 = [row for row in checkpoint["arms"] if row["samples_per_gradient"] == 32]
    best_n32 = min(official_n32, key=lambda row: row["best_s"])
    return {
        **checkpoint,
        "generated_at_utc": generated_at_utc,
        "axis": "[macOS-CPU advisory . frozen cached exact cells . NON-PROMOTABLE]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "paid_dispatch": False,
        "cloud_dispatch": False,
        "scorer_calls": 0,
        "live_trainer_imported_or_mutated": False,
        "live_run_dirs_read_or_mutated": False,
        "research_only": True,
        "clean_room_source_receipt_best_s": clean_source_best,
        "clean_room_rerun_best_arm": clean_rerun,
        "clean_room_rerun_matches_source_receipt": True,
        "exact_enumeration_s": exact_best,
        "best_enriched_arm": best,
        "best_official_n32_arm": best_n32,
        "delta_best_enriched_minus_clean_room_s": float(best["best_s"]) - clean_best,
        "delta_best_n32_minus_clean_room_s": float(best_n32["best_s"]) - clean_best,
        "delta_best_enriched_minus_exact_s": float(best["best_s"]) - exact_best,
        "same_budget_ranking_changed": bool(
            float(best["best_s"]) < clean_best - NOISE_FLOOR_S
        ),
        "verdict": "NO-GO",
        "verdict_reason": (
            "learned logits and the final-paper N=32 calibration tie the in-run clean-room "
            "control and remain above exact enumeration; each live sample would still require "
            "the frozen objective forward"
        ),
        "verdict_scope": (
            "64-state cached terminal-polish objective; fixed-k exact conditional sampling; "
            "independently derived learned-logit SFESS formulation and sample-count calibration; "
            "not a byte-for-byte OSS implementation comparison"
        ),
        "review_status": "recovery-written-UNREVIEWED",
        "across_seed_variance": "UNKNOWN",
        "numpy_torch_mlx_parity": "UNKNOWN_NOT_MEASURED_FOR_ENRICHED_OPTIMIZER",
        "runtime": runtime,
    }


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "stage_checkpoint.json"
    config = _config()
    checkpoint: dict[str, Any]
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("schema") != SCHEMA or checkpoint.get("config") != config:
            raise RuntimeError("checkpoint/config mismatch")
    else:
        checkpoint = {"schema": SCHEMA, "config": config, "clean_arms": [], "arms": []}
    checkpoint.setdefault("clean_arms", [])
    terminal_path = output_dir / "measurement_receipt.json"
    expected_clean_arms = len(K_VALUES)
    expected_enriched_arms = len(SAMPLE_COUNTS) * len(K_VALUES)
    if (
        terminal_path.is_file()
        and len(checkpoint["clean_arms"]) == expected_clean_arms
        and len(checkpoint["arms"]) == expected_enriched_arms
    ):
        receipt = json.loads(terminal_path.read_text(encoding="utf-8"))
        if receipt.get("schema") != SCHEMA or receipt.get("config") != config:
            raise RuntimeError("terminal receipt/config mismatch")
        recomputed = _recompute_complete_checkpoint(output_dir, config)
        if checkpoint != recomputed:
            raise RuntimeError("terminal checkpoint content authentication failed")
        expected_receipt = _build_receipt(
            recomputed,
            generated_at_utc=receipt.get("generated_at_utc"),
            runtime=receipt.get("runtime"),
        )
        if receipt != expected_receipt:
            raise RuntimeError("terminal receipt content authentication failed")
        return receipt
    _validate_partial_checkpoint(checkpoint)
    completed_clean = {row["arm"] for row in checkpoint["clean_arms"]}
    for k in K_VALUES:
        arm_name = f"clean_room_k{k}_m5"
        if arm_name not in completed_clean:
            checkpoint["clean_arms"].append(
                _run_clean_control(k, output_dir / f"clean_k{k}_stage_snapshot.json")
            )
            completed_clean.add(arm_name)
            _atomic_json(checkpoint_path, checkpoint)
    completed = {row["arm"] for row in checkpoint["arms"]}
    for samples in SAMPLE_COUNTS:
        for k in K_VALUES:
            arm_name = f"learned_logits_k{k}_m{samples}"
            if arm_name not in completed:
                checkpoint["arms"].append(_run_arm(k, samples))
                completed.add(arm_name)
                _atomic_json(checkpoint_path, checkpoint)

    recomputed = _recompute_complete_checkpoint(output_dir, config)
    if checkpoint != recomputed:
        raise RuntimeError("pre-terminal checkpoint content authentication failed")
    receipt = _build_receipt(
        recomputed,
        generated_at_utc=datetime.now(UTC).isoformat(),
        runtime={
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    )
    _atomic_json(terminal_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    resolved = args.output_dir.resolve()
    if resolved.parent != (REPO / "experiments/results").resolve():
        raise SystemExit("output directory must be a direct child of experiments/results")
    receipt = run(resolved)
    print(json.dumps({
        "receipt": str(resolved / "measurement_receipt.json"),
        "best_enriched_arm": receipt["best_enriched_arm"],
        "delta_best_enriched_minus_clean_room_s": receipt[
            "delta_best_enriched_minus_clean_room_s"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
