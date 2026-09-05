#!/usr/bin/env python
"""ddm_md3 — seal ONE burn cell that differs from the burn default of record ONLY in its START.

The question (gs4 §5(b), md2 §8): the born vehicle's PERSISTENT --- optimizer-unreachable ---
error set is a subset of a step-0 wrong pool that is **bit-identical across every cell measured so
far**, because every cell declares the same ``initial_state`` file.  Is that pool a property of
the START (a different start paints a different pool, so an init search is a live lever) or of the
DATA (scorer-hard sites no start reaches, so the born accuracy corner is closed for this generator
form)?

**A premise correction this arm owes, MEASURED.**
``ddm_qbr1_born_fairform_burn_prep.build_initial_state`` copies r10's stage-03 ``ema.shadow``
verbatim and **takes no seed**.  The born vehicle's random initialisation is
``ddm_qbflow_packet.initialize_params(20260827)``, behind the entire qbt1 r1..r10 chain.  There is
therefore no init seed in any burn cell's config to flip, and a different RANDOM initialisation
cannot be produced in this arm's budget.  What this cell varies is the **STARTING POINT**: a state
some governed run actually reached and retained, copied verbatim.  Every verdict drawn from it
carries that scope --- different start, same root init seed.

**Why r10_live and not a farther rung (the cell-selection gate, MEASURED by
``ddm_md3_step0_pool_overlap``).**  The pre-registered falsifier is
``J(new persistent, cold persistent) >= 0.70``.  A candidate's persistent set is a subset of its
own step-0 pool, so the pool sizes cap J before the burn starts:

    r10_live 0.8082 | r10@4950 0.7711 | r9 0.7187 | r8 0.4815 | r7 0.1689 | r6 0.1267

Only ``r10_live`` clears 0.70 with real headroom.  Firing r8 or below would install a gate that
**cannot fire by arithmetic** --- ng5's own lesson about a gate wearing the name of one that
works.  ``r10_live`` is also the maximally comparable rung: same source checkpoint, same step,
step-0 ``d_seg_hat`` within 4.3% of the incumbent's.

**The no-op detector is inverted here and it is already satisfied.**  For this cell the step-0
field must DIFFER from the incumbent's --- that is the point.  MEASURED: pool Jaccard 0.6288
(DALI) --- 13,012 sites shared, **7,680 differ** (4,139 wrong only under the new start, 3,541 only
under the incumbent).  The gate below reads that from the retained receipt and refuses if the
pools ever agree.

$0 to seal.  No Metal, no Modal, no contest eval.  Burn-QUALITY object; not a pointer mover.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_ng5_composition_cell as ng5
from experiments import ddm_qbr1_born_fairform_burn_prep as qbr1
from experiments import ddm_qbt1_qbflow_trainer as qbt

ARM = "ddm_md3_different_initialisation"
SEED = ng5.SEED
ARM_NAME = ng5.ARM_NAME
CANDIDATE_ID = "r10_live"
CELL_ID = f"seed_{SEED}_different_init_{CANDIDATE_ID}_{ARM_NAME}"
SEAL_SCHEMA = "ddm_md3_different_init_seal.v1"

AP_STORE = Path("/Volumes/APDataStore/pact/ddm_md3_different_initialisation")
SEALED_CONFIGS = AP_STORE / "sealed_configs"
AUTHORIZED_CONFIGS = AP_STORE / "authorized_configs"
SEALED_SOURCE_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_md3_different_initialisation")
RUN_ROOT = Path(
    "/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/md3_different_initialisation"
)
STEP0_RECEIPT = AP_STORE / "STEP0_POOL_OVERLAP.json"
STATES_INDEX = AP_STORE / "ALTERNATE_INITIAL_STATES.json"

# ng5's sealed config on disk --- the burn default of record this cell is measured against.
NG5_SEALED = Path(
    "/Volumes/APDataStore/pact/ddm_ng5_tau_band_x_continuous_objective/sealed_configs"
    f"/seed_{SEED}_tau_band_x_continuous_objective_control_native100.json"
)

# The ONLY keys this cell may move away from the burn default of record.
ALLOWED_START_MUTATIONS = frozenset({"cell_id", "output", "initial_state"})

# The burn path.  These must be byte-identical to ng5's sealed tree or the cells are not
# comparable; the seal receipt records the reading either way.
BURN_PATH = (
    "experiments/ddm_qbt1_qbflow_trainer.py",
    "experiments/ddm_qbr1_born_fairform_burn_prep.py",
    "experiments/ddm_qbflow_packet.py",
    "src/tac/witness_dsl/curriculum_dsl.py",
)


class MD3CellError(RuntimeError):
    """ddm_md3 refuses rather than sealing a cell that moves more than the start."""


# ---------------------------------------------------------------------------
def candidate_state_fact() -> dict[str, Any]:
    index = json.loads(STATES_INDEX.read_text(encoding="utf-8"))
    for row in index["candidates"]:
        if row["candidate_id"] == CANDIDATE_ID:
            if not row["born_gate_eligible"]:
                raise MD3CellError(f"{CANDIDATE_ID} is born-gate INELIGIBLE and may not start a cell")
            return qbt.file_fact(Path(row["state"]["path"]))
    raise MD3CellError(f"{CANDIDATE_ID} absent from {STATES_INDEX}")


def step0_difference_receipt() -> dict[str, Any]:
    """The inverted no-op detector: this cell's start MUST paint a different step-0 pool."""

    receipt = json.loads(STEP0_RECEIPT.read_text(encoding="utf-8"))
    rows: dict[str, Any] = {}
    for lineage, block in receipt["blocks"].items():
        matches = [r for r in block["candidates"] if r["candidate_id"] == CANDIDATE_ID]
        if not matches:
            raise MD3CellError(f"{lineage}: no step-0 probe row for {CANDIDATE_ID}")
        row = matches[0]
        overlap = row["pool_vs_incumbent"]
        differing = overlap["union"] - overlap["intersection"]
        if differing <= 0:
            raise MD3CellError(
                f"{lineage}: the candidate start paints the SAME step-0 pool as the incumbent -- "
                "this cell would be a no-op on the only variable it moves"
            )
        rows[lineage] = {
            "candidate_pool_sites": overlap["a_sites"],
            "incumbent_pool_sites": overlap["b_sites"],
            "pool_jaccard": overlap["jaccard"],
            "sites_that_differ": differing,
            "candidate_step0_d_seg_hat": row["step0"]["d_seg_hat"],
            "incumbent_step0_d_seg_hat": block["incumbent"]["step0"]["d_seg_hat"],
            "incumbent_persistent_containment": row["incumbent_persistent_containment"],
            "max_attainable_persistent_jaccard_note": (
                "0.8082 (DALI), derived in the memo from the pool sizes; the 0.70 falsifier is "
                "reachable with headroom on this rung and on no farther one"
            ),
        }
    return rows


# ---------------------------------------------------------------------------
def compile_cell() -> tuple[dict[str, Any], dict[str, Any]]:
    """ng5's composition, recompiled in THIS tree, with the START swapped and nothing else."""

    ng5_cell, _control, _audit = ng5.compile_composition_cell()
    cell = copy.deepcopy(ng5_cell)
    cell["cell_id"] = CELL_ID
    cell["output"] = str((RUN_ROOT / "runs" / CELL_ID).resolve())
    cell["initial_state"] = candidate_state_fact()
    qbr1.validate_config(cell, require_launch_authority=False)
    return cell, ng5_cell


def validate_cell(cell: Mapping[str, Any], ng5_cell: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the cell is the burn default of record plus exactly a different start."""

    if cell["initial_state"]["sha256"] == ng5_cell["initial_state"]["sha256"]:
        raise MD3CellError("the cell must NOT start from the incumbent state -- that is the lever")
    differing = {key for key in set(cell) | set(ng5_cell) if cell.get(key) != ng5_cell.get(key)}
    extra = differing - ALLOWED_START_MUTATIONS
    if extra:
        raise MD3CellError(f"the cell moved more than the start: {sorted(extra)}")
    for key in ("objective", "ema", "schedule", "learning_rate", "pair_ids", "selection_weights",
                "total_steps", "milestones", "seed", "resume_from", "area_cap", "chunk_pairs",
                "checkpoint_every_steps", "device", "source_pins", "tau_band", "margin_dual",
                "margin_constraints", "expected_flip_tau_start", "expected_flip_tau_end"):
        if cell.get(key) != ng5_cell.get(key):
            raise MD3CellError(f"the cell moved a held field: {key}")
    if cell.get("resume_from") is not None:
        raise MD3CellError("a different START is not a resume: resume_from stays null")
    if cell.get("launch_authorized") is not False:
        raise MD3CellError("seal must leave the cell unauthorized")
    for lane in ("scorer_lane", "metal_lane"):
        if cell[lane].get("claimed") is not False or cell[lane].get("claim_id") is not None:
            raise MD3CellError(f"seal must leave {lane} unbound for MAIN")

    # The recompiled composition must also equal ng5's SEALED config on disk, on every lever leg.
    sealed = json.loads(NG5_SEALED.read_text(encoding="utf-8"))
    legs = {}
    for key in ("tau_band", "margin_dual", "expected_flip_tau_start", "expected_flip_tau_end"):
        legs[key] = cell.get(key) == sealed.get(key)
        if not legs[key]:
            raise MD3CellError(f"recompiled {key} differs from ng5's sealed config on disk")
    if cell["margin_constraints"].get("initial_lambdas") != sealed["margin_constraints"].get("initial_lambdas"):
        raise MD3CellError("recompiled initial_lambdas differ from ng5's sealed config on disk")
    return {
        "differing_keys": sorted(differing),
        "allowed": sorted(ALLOWED_START_MUTATIONS),
        "levers_match_ng5_sealed_on_disk": legs,
        "incumbent_initial_state_sha256": ng5_cell["initial_state"]["sha256"],
        "cell_initial_state_sha256": cell["initial_state"]["sha256"],
    }


def burn_path_identity(sealed_tree: Path | None = None) -> dict[str, Any]:
    """Read the burn path's hashes here and, when given, inside ng5's tree --- never assert."""

    rows: dict[str, Any] = {}
    ng5_tree = Path(
        "/Volumes/VertigoDataTier/pact/ddm_ng5_tau_band_x_continuous_objective"
        "/sealed_source_d54f65c1ed"
    )
    for name in BURN_PATH:
        here = qbt.file_fact(REPO / name)["sha256"]
        row = {"head": here}
        if (ng5_tree / name).is_file():
            row["ng5_sealed_tree"] = qbt.file_fact(ng5_tree / name)["sha256"]
            row["identical_to_ng5"] = row["ng5_sealed_tree"] == here
        if sealed_tree is not None and (sealed_tree / name).is_file():
            row["md3_sealed_tree"] = qbt.file_fact(sealed_tree / name)["sha256"]
            row["identical_to_md3_tree"] = row["md3_sealed_tree"] == here
        rows[name] = row
    return rows


# ---------------------------------------------------------------------------
def snapshot_source() -> dict[str, Any]:
    revision = qbr1.source_revision()
    destination = SEALED_SOURCE_ROOT / f"sealed_source_{revision[:10]}"
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *BURN_PATH],
        cwd=REPO, text=True, capture_output=True, check=True,
    ).stdout.strip()
    if dirty:
        raise MD3CellError(f"refusing to snapshot a dirty burn path:\n{dirty}")
    if destination.exists():
        return {"schema": "ddm_md3_sealed_source.v1", "revision": revision,
                "root": str(destination), "reused_existing": True}
    destination.mkdir(parents=True)
    archive = destination.parent / f".{destination.name}.tar"
    with archive.open("wb") as stream:
        subprocess.run(["git", "archive", "--format=tar", revision],
                       cwd=REPO, stdout=stream, check=True)
    subprocess.run(["tar", "-xf", str(archive), "-C", str(destination)], check=True)
    archive.unlink()
    shared = ng5.link_shared_inputs(destination)
    return {
        "schema": "ddm_md3_sealed_source.v1",
        "arm": ARM,
        "revision": revision,
        "root": str(destination),
        "shared_inputs": shared,
        "reused_existing": False,
    }


def reroot_and_validate(sealed_tree: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-root the pins into the firing tree and validate with the interpreter that fires."""

    rerooted = SEALED_CONFIGS / f"{CELL_ID}.rerooted.json"
    receipt = SEALED_CONFIGS / f"{CELL_ID}.reroot_receipt.json"
    subprocess.run(
        [str(REPO / ".venv/bin/python"), str(REPO / "experiments/ddm_reseal_pins_inside_sealed_tree.py"),
         "--config-in", str(config_path), "--sealed-tree", str(sealed_tree),
         "--config-out", str(rerooted), "--receipt-out", str(receipt)],
        cwd=REPO, check=True, text=True, capture_output=True,
    )
    in_tree = ng5.validate_inside_sealed_tree(sealed_tree, rerooted)
    return qbt.file_fact(rerooted), {"reroot_receipt": json.loads(receipt.read_text(encoding="utf-8")),
                                     "in_tree_validation": in_tree}


def queue_spec(sealed_tree: Path, authorized: Path) -> dict[str, Any]:
    venv = str(sealed_tree / ".venv/bin/python")
    launch_dir = str((RUN_ROOT / "launch" / CELL_ID).resolve())
    return {
        "schema": "ddm_gv1_cell_queue_spec.v1",
        "written_by": ARM,
        "notes": (
            "ONE different-START cell: ng5's burn default of record with initial_state pointed at "
            "r10's stage-03 LIVE weights (md3_r10_live_state.pt). Burn-QUALITY only; md1/md2's "
            "persistent-partition closure stands and no S_hat delta from this cell is progress "
            "toward sub-0.12. The verdict this cell exists for is "
            "J(persistent_new, persistent_cold): >= 0.70 DATA-ANCHORED, <= 0.45 INIT-ANCHORED."
        ),
        "cells": [
            {
                "cell_id": CELL_ID,
                "sealed_tree": str(sealed_tree),
                "sealed_config": str(SEALED_CONFIGS / f"{CELL_ID}.rerooted.json"),
                "authorized_config": str(authorized),
                "control_label": "cold_control_of_record_seed_20260902",
                "control_run_dir": (
                    "/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/runs/seed_20260902"
                    "/control_native100"
                ),
                "done_receipt": "md3_different_init_DONE.json",
                "peak_family": "ddm_qbr1_born_fairform_burn_prep",
                "measured_peak_rss_gib": "from_ledger",
                "metal_lane_prefix": "md3_different_init_metal",
                "scorer_lane_prefix": "md3_different_init_scorer",
                "milestones": list(qbr1.MILESTONES),
                "falsifiers": [
                    {
                        "name": "the_new_start_reached_the_loop",
                        "at_step": 1,
                        "metric": "objective.tau",
                        "op": "gt",
                        "threshold": 0.04376363754272461,
                    }
                ],
                "launcher_argv": [
                    venv,
                    str(sealed_tree / "tools/launch_detached_process.py"),
                    "--output-dir", launch_dir,
                    "--cwd", str(sealed_tree),
                    "--purpose", f"MD3 different-START burn cell {CELL_ID}",
                    "--authority",
                    ("ddm_md3 sealed cell, fired through the queue driver; "
                     "[macOS-MPS n32 stratified advisory] -- not contest authority"),
                    "--derive-resource-budgets",
                    "--measured-peak-rss-gib",
                    "REWRITTEN_BY_THE_QUEUE_DRIVER_FROM_THE_MEASURED_PEAK_LEDGER",
                    "--measured-thread-need", "4",
                    "--walltime-cap-s", "18000",
                    "--done-receipt", "md3_different_init_DONE.json",
                    "--",
                    venv,
                    str(sealed_tree / "experiments/ddm_qbr1_born_fairform_burn_prep.py"),
                    "run-config",
                    str(authorized),
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
def seal() -> dict[str, Any]:
    started = time.monotonic()
    cell, ng5_cell = compile_cell()
    ng5.refuse_if_the_run_has_already_started(Path(cell["output"]))
    diff = validate_cell(cell, ng5_cell)
    step0 = step0_difference_receipt()

    SEALED_CONFIGS.mkdir(parents=True, exist_ok=True)
    cell_fact = qbt.atomic_json(SEALED_CONFIGS / f"{CELL_ID}.json", cell)
    manifest = snapshot_source()
    sealed_tree = Path(manifest["root"])
    rerooted_fact, reroot = reroot_and_validate(sealed_tree, SEALED_CONFIGS / f"{CELL_ID}.json")

    AUTHORIZED_CONFIGS.mkdir(parents=True, exist_ok=True)
    authorized = AUTHORIZED_CONFIGS / f"{CELL_ID}.json"
    authorized_cell = json.loads((SEALED_CONFIGS / f"{CELL_ID}.rerooted.json").read_text(encoding="utf-8"))
    authorized_cell["launch_authorized"] = True
    authorized_fact = qbt.atomic_json(authorized, authorized_cell)

    spec = queue_spec(sealed_tree, authorized)
    spec_fact = qbt.atomic_json(AP_STORE / "QUEUE_SPEC.json", spec)

    receipt = {
        "schema": SEAL_SCHEMA,
        "arm": ARM,
        "axis": "[seal only; no Metal, no Modal, no contest eval]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "cell_id": CELL_ID,
        "candidate_start": CANDIDATE_ID,
        "premise_correction": (
            "build_initial_state copies r10's EMA shadow verbatim and takes NO seed; the born "
            "vehicle's random init is packet.initialize_params(20260827) behind the whole r1..r10 "
            "chain. This cell varies the STARTING POINT, not the random initialisation, and every "
            "verdict carries that scope."
        ),
        "honest_frame": (
            "burn-QUALITY cell on the born vehicle (S_hat ~0.39-0.43 at ~106 KB), NOT a pointer "
            "mover: md1/md2's persistent-partition closure stands (62-63% of born d_seg is "
            "optimizer-unreachable). No S_hat delta from this cell is progress toward sub-0.12."
        ),
        "one_lever": diff,
        "inverted_no_op_detector": step0,
        "burn_path_identity": burn_path_identity(sealed_tree),
        "sealed_cell_config": cell_fact,
        "rerooted_config": rerooted_fact,
        "authorized_config": authorized_fact,
        "reroot": reroot,
        "sealed_source": manifest,
        "queue_spec": spec_fact,
        "seal_wall_clock_s": time.monotonic() - started,
    }
    qbt.atomic_json(AP_STORE / "SEAL_RECEIPT.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("seal", "show"), default="seal")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "show":
        print(json.dumps(step0_difference_receipt(), indent=2, sort_keys=True))
        return 0
    receipt = seal()
    print(json.dumps({k: v for k, v in receipt.items() if k != "burn_path_identity"},
                     indent=2, sort_keys=True, default=str)[:6000])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
