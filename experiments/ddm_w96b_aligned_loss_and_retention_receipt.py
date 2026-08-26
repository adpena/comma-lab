#!/usr/bin/env python3
"""Build-only W96B config sealing and real OFF-tree CAS demand measurement.

This entrypoint never loads a scorer, claims a lane, or launches training.  It
hashes the 35 retained OFF evaluation trees in place, derives the exact
65-epoch/two-seed demand under the reviewed chunking law, and emits two
launch-disabled aligned configs plus their updated MAIN fire order.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for root in (REPO, SRC):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments import ddm_wd3_scorer_aware_width_distillation as wd3
from tac import content_addressed_retention as cas_retention

SCHEMA = "ddm_w96b_aligned_loss_retention_receipt.v1"
FIRE_SCHEMA = "ddm_w96b_sealed_fire_order.v1"
OUTPUT_ROOT = Path("/Volumes/APDataStore/pact/ddm_w96a_aligned_window")
REPLAY = OUTPUT_ROOT / "off_baseline_s1e_rerun.json"
OFF_CONFIG_ROOT = Path("/Volumes/APDataStore/pact/ddm_s1a_stage_a_adapter/launch_requests")
RECORDED_AVAILABLE_BYTES = 22_319_071_232
REFERENCE_ONE_SEED_ALLOCATED_BYTES = 18_465_816_576
RESERVE_BYTES = 8 * 1024**3


class W96BReceiptError(RuntimeError):
    """Receipt construction refused an incomplete or drifting evidence surface."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise W96BReceiptError(f"required file is absent: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, value: object) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.w96b.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return file_record(path)


def _evaluation_roots(replay: Mapping[str, Any]) -> tuple[list[Path], list[Path], dict[int, list[int]]]:
    if replay.get("row_count") != 35 or len(replay.get("rows", [])) != 35:
        raise W96BReceiptError("OFF replay no longer contains the chartered 35 rows")
    roots = []
    epochs: dict[int, list[int]] = {}
    for row in replay["rows"]:
        seed = int(row["seed"])
        epoch = int(row["epoch"])
        receiver_path = Path(row["retained_evaluation_artifacts"]["receiver_pairs"]["path"])
        root = receiver_path.parent.resolve()
        if receiver_path.name != "receiver_pairs.rgb.u8" or not root.is_dir():
            raise W96BReceiptError(f"retained evaluation tree is absent: {root}")
        roots.append(root)
        epochs.setdefault(seed, []).append(epoch)
    if len(set(roots)) != 35 or set(epochs) != set(wd3.STAGE_A_SEEDS):
        raise W96BReceiptError("OFF replay tree/seed census differs")
    scheduled_epochs = set(epochs[20260815])
    if max(scheduled_epochs) != wd3.ALIGNED_FULL_WINDOW_EPOCHS or len(scheduled_epochs) != 14:
        raise W96BReceiptError(f"seed-1 sealed checkpoint schedule differs: {sorted(scheduled_epochs)}")
    canonical = [
        root
        for root, row in zip(roots, replay["rows"], strict=True)
        if int(row["epoch"]) in scheduled_epochs
    ]
    counts = Counter(
        int(row["seed"]) for row in replay["rows"] if int(row["epoch"]) in scheduled_epochs
    )
    if counts != Counter({20260815: 14, 20260816: 14}) or len(canonical) != 28:
        raise W96BReceiptError(f"canonical 65-epoch OFF cohort differs: {dict(counts)}")
    return roots, canonical, {seed: sorted(values) for seed, values in epochs.items()}


def _aligned_config(seed: int, *, builder_sha256: str) -> dict[str, Any]:
    source_path = OFF_CONFIG_ROOT / f"off_seed_{seed}.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(source)
    output = OUTPUT_ROOT / "training" / f"aligned_seed_{seed}"
    config["output"] = str(output)
    config["resume_root"] = str(output / "resume")
    config["epochs"] = wd3.ALIGNED_FULL_WINDOW_EPOCHS
    config["expected_builder_sha256"] = builder_sha256
    config["optimizer"]["lr"] = 2.0e-5
    config["objective"].update(
        {
            "seg_loss_law": wd3.SEG_LOSS_EXPECTED_FLIP_MARGIN,
            "expected_flip_tau_start": wd3.ALIGNED_TAU_START,
            "expected_flip_tau_end": wd3.ALIGNED_TAU_END,
            "full_window_epochs": wd3.ALIGNED_FULL_WINDOW_EPOCHS,
            "pose_start_step": 0,
        }
    )
    config["evaluation_retention"] = {
        "schema": "ddm_w96b_evaluation_retention.v1",
        "mode": "content_addressed_chunks_v1",
        "cas_root": str(wd3.ALIGNED_CAS_ROOT),
        "compact_after_verify": True,
    }
    config["launch_authorized"] = False
    config["r5_exit_verified"] = False
    config["scorer_lane"] = {
        "claimed": False,
        "claim_id": None,
        "agent": "MAIN",
        "platform": "macos-cpu",
    }
    config["metal_lane"] = {
        "claimed": False,
        "claim_id": None,
        "agent": "MAIN",
        "platform": "macos-mps",
    }
    return config


def _sealed_fire_order(
    *,
    configs: list[dict[str, Any]],
    config_records: list[dict[str, Any]],
    storage: Mapping[str, Any],
) -> dict[str, Any]:
    storage_green = storage["two_seed_65epoch_post_dedup_allocated_bytes"] <= RECORDED_AVAILABLE_BYTES
    return {
        "schema": FIRE_SCHEMA,
        "disposition": "BLOCKED_AWAITING_MAIN_LANES_AND_AUTH" if storage_green else "BLOCKED_STORAGE",
        "implementation_gate": "GREEN",
        "storage_gate": "GREEN" if storage_green else "BLOCKED",
        "owner": "MAIN",
        "consumer_store": str(OUTPUT_ROOT) + "/",
        "seed_order": list(wd3.STAGE_A_SEEDS),
        "configs": [
            {
                "seed": int(config["seed"]),
                "record": record,
                "config_sha256": wd3.canonical_sha256(config),
                "launch_authorized": False,
            }
            for config, record in zip(configs, config_records, strict=True)
        ],
        "fire_trigger": (
            "MAIN confirms the measured storage gate is green, claims fresh distinct scorer and Metal lanes, "
            "sets launch_authorized/r5_exit_verified in a newly sealed config, and passes WD3 validation; "
            "run seed 20260815 to its retained stage end before seed 20260816"
            if storage_green
            else "#1165 Vertigo reclaim completes after pk4 cold-move 2026-08-27 and APDataStore free bytes "
            f"reach at least {storage['two_seed_65epoch_post_dedup_allocated_bytes']} B "
            f"({storage['two_seed_65epoch_required_with_reserve_bytes']} B with the binding 8 GiB reserve)"
        ),
        "per_checkpoint_consumer": {
            "owner": "MAIN",
            "action": "rerun the exact S1E evenly-strided n60 screen for each seed separately",
            "population": list(range(0, 600, 10)),
            "n600_fire_gate": "aligned composed_delta <= matched OFF composed_delta / 5",
            "two_seed_close_gate": "both seed-65 screens remain worse than matched OFF / 2",
        },
        "n600": {
            "owner": "MAIN",
            "consumer_store": str(OUTPUT_ROOT) + "/",
            "fire_trigger": (
                "one retained aligned checkpoint passes the >=5x S1E improvement gate; the global n600 "
                "scorer lane is idle and freshly claimed; selected checkpoint/config/CAS payload SHAs are sealed"
            ),
        },
        "scorer_invocations": 0,
        "metal_invocations": 0,
        "training_launched": False,
        "score_claim": False,
    }


def build_receipt(output_root: Path, *, reuse_inventory_from: Path | None = None) -> dict[str, Any]:
    if output_root.resolve() != OUTPUT_ROOT.resolve():
        raise W96BReceiptError(f"W96B output must stay at the additive AP root {OUTPUT_ROOT}")
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    roots, canonical, epochs = _evaluation_roots(replay)
    inventory_source = None
    if reuse_inventory_from is None:
        inventory = cas_retention.inventory_cohorts(
            {"all_35_observed": roots, "two_seed_65epoch": canonical}
        )
    else:
        inventory_source = file_record(reuse_inventory_from)
        prior = json.loads(reuse_inventory_from.read_text(encoding="utf-8"))
        inventory = prior.get("inventory")
        if (
            prior.get("schema") != SCHEMA
            or prior.get("source_replay") != file_record(REPLAY)
            or not isinstance(inventory, Mapping)
            or inventory.get("schema") != "tac_content_addressed_cohort_inventory.v1"
            or inventory.get("source_tree_count") != 35
            or inventory.get("cohorts", {}).get("all_35_observed", {}).get("tree_count") != 35
            or inventory.get("cohorts", {}).get("two_seed_65epoch", {}).get("tree_count") != 28
        ):
            raise W96BReceiptError("reused inventory receipt does not bind the same 35/28-tree corpus")
    seed1_roots = [root for root, row in zip(roots, replay["rows"], strict=True) if int(row["seed"]) == 20260815]
    seed1_eval_allocated = sum(
        int(row["logical_allocated_bytes"])
        for row in inventory["cohorts"]["all_35_observed"]["trees"]
        if Path(row["root"]) in seed1_roots
    )
    fixed_per_seed = REFERENCE_ONE_SEED_ALLOCATED_BYTES - seed1_eval_allocated
    if fixed_per_seed < 0:
        raise W96BReceiptError("measured evaluation allocation exceeds the one-seed reference")
    canonical_dedup = int(inventory["cohorts"]["two_seed_65epoch"]["post_dedup_allocated_bytes"])
    two_seed_demand = canonical_dedup + 2 * fixed_per_seed
    required_with_reserve = two_seed_demand + RESERVE_BYTES
    current_free = shutil.disk_usage(output_root).free
    storage = {
        "measurement_scope": (
            "exact SHA-256 inventory of the 35 existing OFF evaluation trees; the aligned two-seed "
            "65-epoch demand uses the exact 28 matching-window trees plus two copies of w96a's "
            "measured one-seed non-evaluation allocation"
        ),
        "recorded_available_bytes": RECORDED_AVAILABLE_BYTES,
        "live_available_bytes_at_receipt": current_free,
        "reference_one_seed_allocated_bytes": REFERENCE_ONE_SEED_ALLOCATED_BYTES,
        "seed1_evaluation_allocated_bytes": seed1_eval_allocated,
        "derived_non_evaluation_allocated_bytes_per_seed": fixed_per_seed,
        "two_seed_65epoch_evaluation_post_dedup_allocated_bytes": canonical_dedup,
        "two_seed_65epoch_post_dedup_allocated_bytes": two_seed_demand,
        "two_seed_65epoch_required_with_reserve_bytes": required_with_reserve,
        "margin_vs_recorded_available_bytes": RECORDED_AVAILABLE_BYTES - two_seed_demand,
        "margin_with_reserve_vs_recorded_available_bytes": RECORDED_AVAILABLE_BYTES - required_with_reserve,
        "prediction_b_demand_below_recorded_available": two_seed_demand <= RECORDED_AVAILABLE_BYTES,
        "storage_gate_green_with_binding_reserve": required_with_reserve <= current_free,
        "falsifier_route": (
            None
            if two_seed_demand <= RECORDED_AVAILABLE_BYTES
            else "#1165 Vertigo-reclaim boundary after pk4 cold-move 2026-08-27"
        ),
    }
    builder_path = Path(wd3.__file__).resolve()
    builder_sha = sha256_file(builder_path)
    request_root = output_root / "launch_requests"
    configs = [_aligned_config(seed, builder_sha256=builder_sha) for seed in wd3.STAGE_A_SEEDS]
    config_records = [
        atomic_json(request_root / f"aligned_seed_{config['seed']}.json", config) for config in configs
    ]
    fire_order = _sealed_fire_order(configs=configs, config_records=config_records, storage=storage)
    fire_record = atomic_json(output_root / "SEALED_FIRE_ORDER_W96B.json", fire_order)
    receipt = {
        "schema": SCHEMA,
        "axis": "[build + source-backed storage measurement; no scorer, Metal, training, Modal, or score claim]",
        "implementation": {
            "status": "GREEN_PENDING_REVIEW_AND_TEST_RECEIPT",
            "trainer": file_record(builder_path),
            "retention_module": file_record(Path(cas_retention.__file__).resolve()),
            "loss_law": "100*mean(sigmoid(-(z_target-max_other)/tau)) on WD3 selected cells",
            "tau": {"start": 0.15, "end": 0.05, "schedule": "linear over full 65-epoch window"},
            "cosine_eta_min_fraction": 0.01,
            "pose_start_step": 0,
            "default_off": True,
        },
        "source_replay": file_record(REPLAY),
        "inventory_source": inventory_source or "measured_live_in_this_invocation",
        "epochs_by_seed": {str(seed): values for seed, values in epochs.items()},
        "inventory": inventory,
        "storage": storage,
        "configs": config_records,
        "fire_order": fire_record,
        "training_launched": False,
        "scorer_invocations": 0,
        "metal_invocations": 0,
        "score_claim": False,
    }
    atomic_json(output_root / "W96B_BUILD_AND_STORAGE_RECEIPT.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--reuse-inventory-from", type=Path)
    args = parser.parse_args()
    receipt = build_receipt(args.output_root, reuse_inventory_from=args.reuse_inventory_from)
    print(json.dumps({"storage": receipt["storage"], "configs": receipt["configs"], "fire_order": receipt["fire_order"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
