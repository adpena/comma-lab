#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_ng4 - seal the CONTINUOUS-OBJECTIVE cell: carry r10's terminal objective state.

The QBR1 stage entry carries the WEIGHTS ONLY and restarts the objective around them.  MAIN's
bug-class finding named four restarted states (tau, duals, EMA law, batch geometry); this arm
MEASURED which of them are real, and only TWO are:

* **tau restarts UP**, 0.05 -> 0.15, a 3x wider soft band on a converged margin field.
* **the duals restart from ZERO** while r10 ended holding a converged Lane/Movable pair.
* **batch geometry never restarted at all** -- ``qbt.SELECTION_IDS == r10["pair_ids"]`` and
  ``chunk_pairs == 16`` on both sides.
* **the EMA acts only on the MEASUREMENT channel** (milestones run inside ``qbt.ema_scope``;
  ``ema.update`` never writes the model), and its EXECUTED effective decay is continuous across
  the transition to 2.24e-5.  The "0.99954 -> 0.99908" in the bug-class note compares r10's
  TARGET decay against the cell's EXECUTED constant; r10's own EXECUTED value at its last update
  was 0.9991017964071857.

So this cell carries tau and the duals, holds the EMA identical to the control (matching the
measurement channel is what makes the S_hat comparison valid), and leaves the optimizer COLD --
one lever, objective continuity, never composed with ng1's warm moments ([[m164]]).

$0: seal + a bounded macOS-CPU mechanism smoke.  No Metal, no Modal, no contest eval.  MAIN
fires the cell.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import resource
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_qbr1_born_fairform_burn_prep as qbr1
from experiments import ddm_qbt1_qbflow_trainer as qbt
from tac.witness_dsl.curriculum_dsl import (
    ContinuousObjectiveFromR10,
    compile_qbr1_continuous_objective_config,
)

ARM = "ddm_ng4_continuous_objective"
ARM_ROOT = Path("/Volumes/APDataStore/pact/ddm_ng4_continuous_objective")
SEALED_CONFIGS = ARM_ROOT / "sealed_configs"
SEALED_SOURCE_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_ng4_continuous_objective")
NG4_RUN_ROOT = qbt.QBR1_RETENTION_ROOT / "ng4_continuous_objective"
SMOKE_ROOT = NG4_RUN_ROOT / "bounded_smoke"

SEED = 20260902
ARM_NAME = "control_native100"
CELL_ID = f"seed_{SEED}_continuous_objective_{ARM_NAME}"

RESOLUTION_SCHEMA = "ddm_ng4_continuous_objective_resolution.v1"
SEAL_SCHEMA = "ddm_ng4_continuous_objective_seal_receipt.v1"
SMOKE_SCHEMA = "ddm_ng4_continuous_objective_bounded_smoke.v1"
SNAPSHOT_SCHEMA = "ddm_ng4_sealed_source_manifest.v1"

#: the only TOP-LEVEL config keys the continuous cell may move relative to a freshly compiled
#: control.  ``margin_constraints`` is in the set because the executable multipliers live inside
#: it; :func:`validate_continuous_objective_cell` then sub-diffs that block so the ONLY thing it
#: may have gained is ``initial_lambdas`` -- bounds, step size and mode must be untouched.
ALLOWED_CONTINUATION_MUTATIONS = frozenset(
    {"cell_id", "output", "tau_band", "expected_flip_tau_start", "expected_flip_tau_end",
     "margin_dual", "margin_constraints"}
)

#: ng1's bounded smoke ran ONE cold update from this same start under code that predated ng2's
#: telemetry row, ng3's validators and this arm's.  Reproducing it bit-for-bit is the MEASURED
#: proof that everything landed since is score-neutral on the training path -- which is what lets
#: a cell built on a moved trainer pin be read against a control that ran under an older one.
NG1_COLD_REFERENCE_LIVE_STATE_SHA256 = (
    "27f514180db2b4cda57289bbeb4be5ca8daf64e874921c92ba5c08d613c30973"
)

#: the completed seed-20260902 control run.  Read live (never retyped) so the bar in this arm's
#: falsifiers is the control's own retained milestone, not a decimal copied between memos.
COLD_CONTROL_RUN = Path(
    "/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/runs/seed_20260902/control_native100"
)

SMOKE_ARMS = ("continuous", "control")


class NG4Error(RuntimeError):
    """Fail-closed ng4 contract error."""


# ---------------------------------------------------------------------------
# 1. what r10 actually ended with, and which of it is discontinuous
# ---------------------------------------------------------------------------
def r10_terminal_state() -> dict[str, Any]:
    """The r10 terminal-state table, every value read from a sha-pinned artifact."""

    tau = qbr1.r10_terminal_tau()
    duals = qbr1.r10_terminal_duals()
    checkpoint = torch.load(qbr1.R10_CHECKPOINT, map_location="cpu", weights_only=False)
    ema = checkpoint["ema"]
    num_updates = int(ema["num_updates"])
    warmup = bool(ema["warmup"])
    target_decay = float(ema["decay"])
    executed = min(target_decay, (1.0 + num_updates) / (10.0 + num_updates)) if warmup else target_decay
    return {
        "expected_flip_tau": tau,
        "margin_dual": duals,
        "ema": {
            "r10_target_decay": target_decay,
            "r10_warmup": warmup,
            "r10_num_updates": num_updates,
            "r10_executed_effective_decay": executed,
        },
        "batch_geometry": {
            "r10_chunk_pairs": int(json.loads(qbr1.R10_CONFIG.read_text(encoding="utf-8"))["chunk_pairs"]),
            "r10_pair_ids": list(json.loads(qbr1.R10_CONFIG.read_text(encoding="utf-8"))["pair_ids"]),
        },
        "source_checkpoint": qbt.file_fact(qbr1.R10_CHECKPOINT),
        "source_config": qbt.file_fact(qbr1.R10_CONFIG),
    }


def discontinuity_audit(control: Mapping[str, Any]) -> dict[str, Any]:
    """Per-state: is the r10 -> QBR1 transition discontinuous in the quantity that ACTS?

    This is the arm's own correction to its charter.  Two of the four named states are carried
    by this cell; the other two are MEASURED continuous (or measurement-only) and carrying them
    would move a channel that must stay matched to the control.
    """

    r10 = r10_terminal_state()
    control_ema = control["ema"]
    executed_control = float(control_ema["value"])
    if bool(control_ema["execution"]["warmup"]):
        raise NG4Error("the control's EMA is not the sealed constant-decay law this audit assumes")
    r10_executed = float(r10["ema"]["r10_executed_effective_decay"])
    control_pairs = list(map(int, control["pair_ids"]))
    return {
        "expected_flip_tau": {
            "r10_terminal": float(r10["expected_flip_tau"]["r10_terminal_tau"]),
            "cell_entry_without_this_lever": float(qbt.LEGACY_EXPECTED_FLIP_TAU_BAND[0]),
            "ratio_entry_over_terminal": float(qbt.LEGACY_EXPECTED_FLIP_TAU_BAND[0])
            / float(r10["expected_flip_tau"]["r10_terminal_tau"]),
            "discontinuous": True,
            "channel": "gradient (the seg surrogate is sigmoid(-margin/tau))",
            "carried_by_this_cell": True,
        },
        "margin_dual": {
            "r10_terminal": r10["margin_dual"]["initial_lambdas"],
            "cell_entry_without_this_lever": dict.fromkeys(r10["margin_dual"]["initial_lambdas"], 0.0),
            "discontinuous": True,
            "channel": "gradient (the per-class penalty 100*lambda_c*expected_flip_c)",
            "dual_law_identical": (
                r10["margin_dual"]["r10_bounds"]
                == {k: float(v) for k, v in control["margin_constraints"]["bounds"].items()}
                and r10["margin_dual"]["r10_eta_lambda"]
                == float(control["margin_constraints"]["eta_lambda"])
                and r10["margin_dual"]["r10_constraint_mode"]
                == str(control["margin_constraints"]["mode"])
            ),
            "carried_by_this_cell": True,
        },
        "ema": {
            "r10_executed_effective_decay": r10_executed,
            "cell_executed_effective_decay": executed_control,
            "absolute_gap": abs(r10_executed - executed_control),
            "relative_gap": abs(r10_executed - executed_control) / r10_executed,
            "r10_target_decay_never_reached": r10["ema"]["r10_target_decay"],
            "discontinuous": False,
            "channel": "MEASUREMENT ONLY -- milestones run inside qbt.ema_scope and ema.update "
                       "never writes the model, so the EMA cannot cause the excursion",
            "carried_by_this_cell": False,
            "why_not": "the EXECUTED averaging rate is already continuous to the relative gap "
                       "above; with warmup=False the restarted update counter has no effect on "
                       "it; and holding the EMA identical to the control keeps the measurement "
                       "channel MATCHED, which is what makes the S_hat comparison valid",
        },
        "batch_geometry": {
            "r10_chunk_pairs": r10["batch_geometry"]["r10_chunk_pairs"],
            "cell_chunk_pairs": int(control["chunk_pairs"]),
            "pair_ids_identical": r10["batch_geometry"]["r10_pair_ids"] == control_pairs,
            "discontinuous": False,
            "channel": "n/a -- nothing to carry",
            "carried_by_this_cell": False,
            "why_not": "MEASURED identical on both sides; the only geometry item that differs is "
                       "the seeded chunk ORDER, and it is deliberately held to the CONTROL's "
                       f"seed {SEED} so the pair isolates the objective",
        },
        "optimizer": {
            "discontinuous": True,
            "channel": "gradient (fresh AdamW takes a full lr-sized sign step)",
            "carried_by_this_cell": False,
            "why_not": "ng1 RACED exactly this and LOST (+0.0186 S_hat vs the cold control).  "
                       "Carrying it here would compose two levers in one cell; the warm-AND-"
                       "continuous twin is the SECOND cell if this one wins ([[m164]])",
        },
        "start_point_live_vs_shadow_gap": {
            "discontinuous": True,
            "channel": "the start point itself",
            "carried_by_this_cell": False,
            "why_not": "r10's live weights and its EMA shadow differ by 8.4488e-03 relative (ng1) "
                       "and the cell starts BOTH at the shadow, collapsing the gap.  Closing it "
                       "would move the START POINT, which is a different lever and would break "
                       "comparability with the control -- build_initial_state is the pinned "
                       "same-start for every cell in this generation.  Queued, not closed.",
        },
    }


# ---------------------------------------------------------------------------
# 2. compile + single-lever proof
# ---------------------------------------------------------------------------
def compile_continuous_objective_cell() -> tuple[dict[str, Any], dict[str, Any]]:
    """Compile the continuation cell and a matched control THROUGH the burn prep's compile path.

    Both come from ``qbr1.compile_cell`` in THIS tree, so their pins are this tree's pins and the
    only thing separating them is the lever.  The sealed control on disk is deliberately NOT
    deep-copied: its pins name an older trainer (ng2, ng3 and this arm each moved it), so
    inheriting them would seal a config no tree can satisfy.
    """

    initial_state = qbt.file_fact(qbr1.QBR_INITIAL_STATE)
    control = qbr1.compile_cell(SEED, ARM_NAME, initial_state)
    lever = ContinuousObjectiveFromR10(qbr1.R10_CONFIG, qbr1.R10_CHECKPOINT)
    tau_block, start, end, dual_block, initial_lambdas = compile_qbr1_continuous_objective_config(lever)
    cell = copy.deepcopy(control)
    cell["cell_id"] = CELL_ID
    cell["output"] = str((NG4_RUN_ROOT / "runs" / CELL_ID).resolve())
    cell["tau_band"] = tau_block
    cell["expected_flip_tau_start"] = start
    cell["expected_flip_tau_end"] = end
    cell["margin_dual"] = dual_block
    cell["margin_constraints"] = copy.deepcopy(control["margin_constraints"])
    cell["margin_constraints"]["initial_lambdas"] = dict(sorted(initial_lambdas.items()))
    qbr1.validate_config(cell, require_launch_authority=False)
    return cell, control


def validate_continuous_objective_cell(
    cell: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed unless the cell is the control plus exactly the two carried objective states."""

    differing = {key for key in set(cell) | set(control) if cell.get(key) != control.get(key)}
    extra = differing - ALLOWED_CONTINUATION_MUTATIONS
    if extra:
        raise NG4Error(f"continuation cell moved more than the lever: {sorted(extra)}")
    for key in ("objective", "ema", "schedule", "initial_state", "learning_rate", "pair_ids",
                "selection_weights", "total_steps", "milestones", "seed", "resume_from",
                "area_cap", "chunk_pairs", "checkpoint_every_steps", "device", "source_pins"):
        if cell.get(key) != control.get(key):
            raise NG4Error(f"continuation cell moved a held field: {key}")
    constraints_moved = {
        key for key in set(cell["margin_constraints"]) | set(control["margin_constraints"])
        if cell["margin_constraints"].get(key) != control["margin_constraints"].get(key)
    }
    if constraints_moved != {"initial_lambdas"}:
        raise NG4Error(
            "the ONLY margin-constraint field the dual lever may move is initial_lambdas; moved "
            f"{sorted(constraints_moved)}"
        )
    if cell.get("area_cap") is not None:
        raise NG4Error("ng4 is a ONE-lever race: no area cap may be present")
    if cell.get("resume_from") is not None:
        raise NG4Error("the continuation cell is a COLD-OPTIMIZER transition: resume_from stays null")
    if cell.get("launch_authorized") is not False:
        raise NG4Error("seal must leave the cell unauthorized")
    for lane in ("scorer_lane", "metal_lane"):
        if cell[lane].get("claimed") is not False or cell[lane].get("claim_id") is not None:
            raise NG4Error(f"seal must leave {lane} unbound for MAIN")
    control_lambdas = qbr1.initial_margin_constraint_lambdas(
        control, {name: float(v) for name, v in control["margin_constraints"]["bounds"].items()}
    )
    cell_lambdas = qbr1.initial_margin_constraint_lambdas(
        cell, {name: float(v) for name, v in cell["margin_constraints"]["bounds"].items()}
    )
    if any(value != 0.0 for value in control_lambdas.values()):
        raise NG4Error("the matched control must still seed its duals from zero")
    if cell_lambdas != {k: float(v) for k, v in cell["margin_constraints"]["initial_lambdas"].items()}:
        raise NG4Error("the executable dual seeding does not read the cell's carried multipliers")
    return {
        "differing_keys": sorted(differing),
        "allowed": sorted(ALLOWED_CONTINUATION_MUTATIONS),
        "margin_constraints_fields_moved": sorted(constraints_moved),
        "held_fields_identical": True,
        "area_cap_absent": True,
        "executable_dual_seeding": {"cell": cell_lambdas, "control": control_lambdas},
        "transition": "COLD OPTIMIZER (fresh AdamW, resume_from null) -- the same optimizer "
                      "transition the control took, so the pair isolates objective continuity "
                      "and never composes it with ng1's warm-moment lever",
    }


def cold_control_receipt() -> dict[str, Any]:
    """Read the completed seed-20260902 control's milestones LIVE (never a retyped decimal)."""

    rows: dict[str, Any] = {}
    for step in qbr1.MILESTONES:
        path = COLD_CONTROL_RUN / "milestones" / f"step_{step:06d}" / "MILESTONE.json"
        if not path.is_file():
            raise NG4Error(f"the cold control of record lacks milestone {step}: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        rows[str(step)] = {
            "S_hat": float(row["S_hat"]),
            "d_seg_hat": float(row["d_seg_hat"]),
            "d_pose_hat": float(row["d_pose_hat"]),
            "archive_bytes_exact": int(row["archive_bytes_exact"]),
        }
    start = rows[str(qbr1.MILESTONES[0])]["S_hat"]
    end = rows[str(qbr1.MILESTONES[-1])]["S_hat"]
    peak_step = max(rows, key=lambda key: rows[key]["S_hat"])
    return {
        "run": str(COLD_CONTROL_RUN),
        "axis": "[macOS-MPS n32 stratified advisory; not contest authority]",
        "milestones": rows,
        "start_S_hat": start,
        "endpoint_S_hat": end,
        "peak_step": int(peak_step),
        "peak_S_hat": rows[peak_step]["S_hat"],
        "endpoint_excess_over_start": end - start,
        "endpoint_excess_relative": (end - start) / start,
    }


def falsifiers(control_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Pre-registered before the burn, with the charter's falsifier 2 re-specified and why."""

    start = float(control_receipt["start_S_hat"])
    return {
        "1_primary_the_excursion_must_not_open": {
            "statement": (
                "S_hat(1,000) <= start + 0.005 (no excursion) AND S_hat(5,000) < start"
            ),
            "start_bar": start,
            "no_excursion_bar_at_1000": start + 0.005,
            "endpoint_bar": start,
            "control_1000": float(control_receipt["milestones"]["1000"]["S_hat"]),
            "control_5000": float(control_receipt["milestones"]["5000"]["S_hat"]),
            "if_it_fails": (
                "the objective restart was not the (whole) cause of the cold-transition "
                "excursion; the family framing is refuted at FORMULATION scope for the born "
                "object, and the residual discontinuities this cell did NOT carry (the cold "
                "optimizer, the collapsed live/shadow gap) become the next suspects"
            ),
        },
        "2_the_16_update_damage_must_be_absent": {
            "statement": (
                "the realized within-class error at update 16 is within 1.2x of its update-1 "
                "value, for BOTH Lane and Movable, read from <run>/history.jsonl"
            ),
            "re_specification": (
                "the charter asks for d_seg_hat at step 16 within 1.2x of step 0.  That number "
                "DOES NOT EXIST in this cell's retained set: MILESTONES is validated as exactly "
                "(0, 1000, 2000, 3000, 4000, 5000) and adding a step-16 milestone would change "
                "the sealed schedule and the retained payload set -- a second lever.  "
                "`realized_within_class_error` is the exact-argmax quantity the loop already "
                "computes and journals EVERY update (run_config appends it to history.jsonl), "
                "measured on the realized post-R logits.  It is a per-class error rather than "
                "the HT-weighted all-class d_seg, so it is a WEAKER instrument than the charter "
                "asked for; it is the strongest one available at zero extra cost"
            ),
            "read_from": "<run>/history.jsonl rows 1 and 16, key realized_within_class_error",
            "if_it_fails": (
                "the 16-update damage md1 measured has a source other than the temperature and "
                "dual restart -- report the first-update parameter displacement beside it, since "
                "the cold optimizer's lr-sized sign step is the remaining candidate"
            ),
        },
        "3_the_dual_trajectory_must_be_continuous": {
            "statement": (
                "margin_constraint_lambdas at update 1 is within one eta_lambda step of r10's "
                "terminal pair, for both classes; and no class re-warms from 0"
            ),
            "eta_lambda": float(qbt.MARGIN_CONSTRAINT_ETA_LAMBDA),
            "r10_terminal": qbr1.r10_terminal_duals()["initial_lambdas"],
            "read_from": "<run>/history.jsonl row 1, key margin_constraint_lambdas",
            "if_it_fails": "the carried multipliers did not reach the loop -- an inert lever, "
                           "which is the fake-implementation class this arm must not ship",
        },
        "read_the_decomposition_never_the_composite": (
            "the control's endpoint excess is 91.20% d_seg (ng1, recomputed from components).  A "
            "cell that 'fixed' S_hat by moving bytes or pose would be a different finding, so "
            "every milestone is read as (100*d_seg, sqrt(10*d_pose), rate) and never as S_hat"
        ),
    }


# ---------------------------------------------------------------------------
# 3. sealed source snapshot (+ pin re-root inside it)
# ---------------------------------------------------------------------------
SHARED_INPUT_PATHS = ("upstream", "experiments/results/mlx_fleet_gt_cache")

LEVER_SURFACE = (
    "experiments/ddm_qbt1_qbflow_trainer.py",
    "experiments/ddm_qbr1_born_fairform_burn_prep.py",
    "experiments/ddm_ng4_continuous_objective_cell.py",
    "src/tac/witness_dsl/curriculum_dsl.py",
)


def link_shared_inputs(destination: Path) -> dict[str, str]:
    """Point the sealed tree's large pinned inputs at the repo, the way wc3's tree does."""

    linked: dict[str, str] = {}
    for relative in (*SHARED_INPUT_PATHS, ".venv"):
        target = REPO / relative
        if not target.exists():
            raise NG4Error(f"shared input is absent from the repo: {target}")
        link = destination / relative
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            subprocess.run(["rm", "-rf", str(link)], check=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
        linked[relative] = str(target)
    return linked


def verify_pins_inside(destination: Path) -> dict[str, Any]:
    """Prove the SEALED tree can verify its own pins -- its own process, its own REPO."""

    script = (
        "import json,sys;"
        f"sys.path.insert(0, {str(destination)!r});"
        "from experiments import ddm_qbr1_born_fairform_burn_prep as q;"
        "r=q.verify_inputs();"
        "print(json.dumps({'qbt_trainer': r['qbt_trainer']['sha256'],"
        " 'gt_cache': r['qbt_gt_cache']['sha256'], 'segnet': r['qbt_segnet']['sha256'],"
        " 'posenet': r['qbt_posenet']['sha256']}))"
    )
    completed = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-c", script],
        cwd=destination, text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise NG4Error(
            f"the sealed tree cannot verify its own pins:\n{completed.stderr.strip()[-2000:]}"
        )
    return {"status": "PASS", **json.loads(completed.stdout.strip().splitlines()[-1])}


def snapshot_source(destination: Path | None = None) -> dict[str, Any]:
    """Materialize the sealed source tree the cell fires from, at the current commit."""

    revision = qbr1.source_revision()
    destination = destination or (SEALED_SOURCE_ROOT / f"sealed_source_{revision[:10]}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *LEVER_SURFACE],
        cwd=REPO, text=True, capture_output=True, check=True,
    ).stdout.strip()
    if dirty:
        raise NG4Error(f"refusing to snapshot a dirty lever surface:\n{dirty}")
    if destination.exists():
        raise NG4Error(f"sealed source already exists (never overwrite a seal): {destination}")
    destination.mkdir(parents=True)
    archive = destination.parent / f".{destination.name}.tar"
    with archive.open("wb") as stream:
        subprocess.run(["git", "archive", "--format=tar", revision],
                       cwd=REPO, stdout=stream, check=True)
    subprocess.run(["tar", "-xf", str(archive), "-C", str(destination)], check=True)
    archive.unlink()
    shared = link_shared_inputs(destination)
    manifest = {
        "schema": SNAPSHOT_SCHEMA,
        "arm": ARM,
        "revision": revision,
        "root": str(destination),
        "method": "git archive of the committed revision, extracted; the pinned large inputs and "
                  "the venv are symlinked to the repo (they are sha-pinned, so sharing them is "
                  "what verify_pins checks, not what it could be fooled by)",
        "shared_inputs": shared,
        "pins_verify_inside_the_sealed_tree": verify_pins_inside(destination),
        "files": {name: qbt.file_fact(destination / name) for name in LEVER_SURFACE},
    }
    qbt.atomic_json(destination / "NG4_SEALED_SOURCE_MANIFEST.json", manifest)
    qbt.atomic_json(ARM_ROOT / "SEALED_SOURCE_MANIFEST.json", manifest)
    return manifest


def reroot_sealed_config(sealed_tree: Path) -> dict[str, Any]:
    """Re-root the sealed config's source_pins to the tree that will RUN it, then RE-VALIDATE.

    The re-root tool proves CONTENT identity (every pin's sha256+bytes) and rewrites only the
    ``path`` fields.  Validating the re-rooted config INSIDE the sealed tree afterwards is the
    part that matters: a seal that has not been validated by the interpreter that fires it is a
    claim, not a receipt ([[seal_validates_only_inside_the_tree_that_fires_it_20260904]]).
    """

    config_in = SEALED_CONFIGS / f"{CELL_ID}.json"
    if not config_in.is_file():
        raise NG4Error("run `seal` before `reroot`")
    config_out = SEALED_CONFIGS / f"{CELL_ID}.rerooted.json"
    receipt_out = ARM_ROOT / "PIN_REROOT_RECEIPT.json"
    completed = subprocess.run(
        [str(REPO / ".venv/bin/python"), str(REPO / "experiments/ddm_reseal_pins_inside_sealed_tree.py"),
         "--config-in", str(config_in), "--sealed-tree", str(sealed_tree),
         "--config-out", str(config_out), "--receipt-out", str(receipt_out)],
        cwd=REPO, text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise NG4Error(f"pin re-root failed:\n{completed.stderr.strip()[-2000:]}")
    validation = validate_inside_sealed_tree(sealed_tree, config_out)
    return {
        "reroot_receipt": json.loads(receipt_out.read_text(encoding="utf-8")),
        "rerooted_config": qbt.file_fact(config_out),
        "validated_inside_the_sealed_tree": validation,
    }


def validate_inside_sealed_tree(sealed_tree: Path, config_path: Path) -> dict[str, Any]:
    """Run the sealed tree's OWN ``validate_config`` on the config it will fire."""

    script = (
        "import json,sys;"
        f"sys.path.insert(0, {str(sealed_tree)!r});"
        "from pathlib import Path;"
        "from experiments import ddm_qbr1_born_fairform_burn_prep as q;"
        f"c=json.loads(Path({str(config_path)!r}).read_text());"
        "q.validate_config(c, require_launch_authority=False);"
        "print(json.dumps({'status':'PASS','cell_id':c['cell_id'],"
        " 'tau':[c['expected_flip_tau_start'],c['expected_flip_tau_end']],"
        " 'initial_lambdas':c['margin_constraints']['initial_lambdas']}))"
    )
    completed = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-c", script],
        cwd=sealed_tree, text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise NG4Error(
            f"the sealed tree REFUSES the config it is meant to fire:\n{completed.stderr.strip()[-2500:]}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# 4. seal
# ---------------------------------------------------------------------------
def refuse_if_the_run_has_already_started(run_output: Path) -> None:
    if (run_output / "RESULT.json").exists() or (run_output / "history.jsonl").exists():
        raise NG4Error(f"refusing to re-seal a cell whose run has already started: {run_output}")


def recompile_determinism(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Two compiles of the same lever must agree everywhere this arm OWNS.

    The QBR1 lineage keeps a dated ``ema.lawref.resolved_at`` inside every config, and
    ``qbt.stable_ema_law_identity`` is its sanctioned comparator (ng3).  This arm's own blocks
    carry no timestamp at all, so they must be byte-stable OUTRIGHT -- and the config sha quoted
    for MAIN is therefore a FILE hash, verified with ``shasum``, never by recompiling.
    """

    again, _control = compile_continuous_objective_cell()
    moved = sorted(key for key in set(cell) | set(again) if cell.get(key) != again.get(key))
    for owned in ("tau_band", "margin_dual", "margin_constraints",
                  "expected_flip_tau_start", "expected_flip_tau_end"):
        if owned in moved:
            raise NG4Error(f"a block this arm owns must be byte-stable across compiles, and is not: {owned}")
    stable_left = qbt.stable_ema_law_identity(cell["ema"])
    stable_right = qbt.stable_ema_law_identity(again["ema"])
    return {
        "keys_that_moved": moved,
        "arm_owned_blocks_byte_stable": True,
        "ema_law_identity_stable_under_the_lineage_comparator": stable_left == stable_right,
        "residual": ("ema.lawref.resolved_at is a dated observation the QBR1 lineage keeps inside "
                     "the config; the sealed sha is a FILE property, verified with shasum"),
    }


def seal() -> dict[str, Any]:
    """Compile, prove the single lever, snapshot the source, re-root the pins, and record it."""

    started = time.monotonic()
    cell, control = compile_continuous_objective_cell()
    refuse_if_the_run_has_already_started(Path(cell["output"]))
    diff = validate_continuous_objective_cell(cell, control)
    audit = discontinuity_audit(control)
    control_receipt = cold_control_receipt()
    determinism = recompile_determinism(cell)

    SEALED_CONFIGS.mkdir(parents=True, exist_ok=True)
    cell_fact = qbt.atomic_json(SEALED_CONFIGS / f"{CELL_ID}.json", cell)
    control_fact = qbt.atomic_json(
        SEALED_CONFIGS / f"matched_control_of_record_seed_{SEED}_{ARM_NAME}.reference.json",
        {**control, "DO_NOT_FIRE": "reference recompile of the ALREADY-MEASURED cold control; "
                                   "the measured row is the completed seed-20260902 run"},
    )
    manifest = snapshot_source()
    reroot = reroot_sealed_config(Path(manifest["root"]))

    receipt = {
        "schema": SEAL_SCHEMA,
        "arm": ARM,
        "axis": "[seal + bounded macOS-CPU mechanism smoke only; no Metal, no Modal, no contest eval]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "cell_id": CELL_ID,
        "r10_terminal_state": r10_terminal_state(),
        "discontinuity_audit": audit,
        "single_lever": diff,
        "recompile_determinism": determinism,
        "cold_control_of_record": control_receipt,
        "falsifiers": falsifiers(control_receipt),
        "sealed_cell_config": cell_fact,
        "matched_control_reference": control_fact,
        "sealed_source": manifest,
        "pin_reroot": reroot,
        "elapsed_seconds": time.monotonic() - started,
    }
    ARM_ROOT.mkdir(parents=True, exist_ok=True)
    qbt.atomic_json(ARM_ROOT / "SEAL_RECEIPT.json", receipt)
    return receipt


# ---------------------------------------------------------------------------
# 5. bounded CPU smoke -- one arm per process, so the peak is one arm's peak
# ---------------------------------------------------------------------------
def _sealed_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    cell_path = SEALED_CONFIGS / f"{CELL_ID}.json"
    control_path = SEALED_CONFIGS / f"matched_control_of_record_seed_{SEED}_{ARM_NAME}.reference.json"
    if not cell_path.is_file() or not control_path.is_file():
        raise NG4Error("run `seal` before `smoke`")
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    control = json.loads(control_path.read_text(encoding="utf-8"))
    control.pop("DO_NOT_FIRE", None)
    return cell, control


def smoke_arm(label: str) -> dict[str, Any]:
    """Run ONE real B=16 CPU update for one arm, in its own process.

    ng3's smoke ran both arms plus the differential in a single address space and peaked at
    41.48 GiB.  Two live Metal cells were holding memory when this arm sealed, so ng4 splits the
    arms: the high-water mark of this process is ONE arm's, and the differential below pays no
    forward pass at all (it reads the payloads the smoke already retained).
    """

    if label not in SMOKE_ARMS:
        raise NG4Error(f"unknown smoke arm {label!r}")
    cell, control = _sealed_configs()
    config = cell if label == "continuous" else control
    started = time.monotonic()
    segment = qbr1._run_resume_smoke_segment(config, SMOKE_ROOT / label, stop_after=1)
    row = json.loads((SMOKE_ROOT / label / "history.jsonl").read_text(encoding="utf-8").splitlines()[0])
    result = {
        "schema": f"{SMOKE_SCHEMA}.arm",
        "arm": ARM,
        "smoke_arm": label,
        "axis": "[macOS-CPU bounded mechanism smoke; not a verdict, not contest authority]",
        "score_claim": False,
        "metal_invocations": 0,
        "modal_invocations": 0,
        "contest_eval_invocations": 0,
        "segment": segment,
        "history_row_update_1": row,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "elapsed_seconds": time.monotonic() - started,
        "all_payloads_retained": True,
    }
    qbt.atomic_json(SMOKE_ROOT / label / "ARM_SMOKE_RESULT.json", result)
    return result


def _objective_inputs_from_retained(payload_dir: Path) -> dict[str, Any]:
    """Rebuild real objective inputs from a retained smoke payload -- no new forward pass."""

    files = sorted(payload_dir.glob("pair_*.npz"))
    if not files:
        raise NG4Error(f"no retained payload under {payload_dir}")
    logits, targets, pose6, target_pose6, camera, pair_ids = [], [], [], [], [], []
    for path in files:
        with np.load(path) as data:
            logits.append(torch.from_numpy(data["segnet_logits_f16"].astype(np.float32)))
            targets.append(torch.from_numpy(data["target_argmax_u8"].astype(np.int64)))
            pose6.append(torch.from_numpy(data["posenet_pose6_f32"].astype(np.float32)))
            target_pose6.append(torch.from_numpy(data["target_pose6_f32"].astype(np.float32)))
            camera.append(torch.from_numpy(data["camera_pair_u8"].astype(np.float32)))
        pair_ids.append(int(path.stem.split("_")[1]))
    stacked = torch.stack(logits)
    return {
        "pair_ids": pair_ids,
        "logits": stacked,
        "outputs": {"class_logits": stacked.permute(0, 2, 3, 1).contiguous()},
        "camera": torch.stack(camera),
        "pose6": torch.stack(pose6),
        "target_argmax": torch.stack(targets),
        "target_pose6": torch.stack(target_pose6),
        "sample_weights": qbt.no2_sample_weights(pair_ids, torch.device("cpu")),
    }


def differential(payload_dir: Path) -> dict[str, Any]:
    """At the CONTROL's tau AND zero duals the two configs' objectives must be BIT-FOR-BIT equal.

    This is the charter's differential.  It proves the lever acts ONLY through the two carried
    states and that nothing else leaked out of the ``tau_band`` / ``margin_dual`` blocks into the
    loss.  The sister halves -- that the objectives DIFFER at the cell's own temperature, and
    again at its own multipliers -- are measured beside it, so the test cannot pass by the
    objective being blind to either input.
    """

    inputs = _objective_inputs_from_retained(payload_dir)
    cell, control = _sealed_configs()
    zero = dict.fromkeys(control["margin_constraints"]["bounds"], 0.0)
    carried = {k: float(v) for k, v in cell["margin_constraints"]["initial_lambdas"].items()}

    def evaluate(config: Mapping[str, Any], tau: float, lambdas: Mapping[str, float]) -> dict[str, float]:
        with torch.no_grad():
            _total, components = qbr1.fairform_objective(
                config, inputs["outputs"], inputs["camera"], inputs["pose6"], inputs["logits"],
                inputs["target_argmax"], inputs["target_pose6"], tau, inputs["sample_weights"],
                lambdas,
            )
        return {name: float(value) for name, value in components.items()}

    shared_tau = float(control["expected_flip_tau_start"])
    cell_neutral = evaluate(cell, shared_tau, zero)
    control_neutral = evaluate(control, shared_tau, zero)
    identical = {name: cell_neutral[name] == control_neutral[name] for name in sorted(control_neutral)}
    cell_own_tau = evaluate(cell, float(cell["expected_flip_tau_start"]), zero)
    cell_own_both = evaluate(cell, float(cell["expected_flip_tau_start"]), carried)
    return {
        "pairs": inputs["pair_ids"],
        "shared_tau": shared_tau,
        "shared_lambdas": zero,
        "components_bit_identical_at_shared_tau_and_zero_duals": all(identical.values()),
        "per_component_identical": identical,
        "cell_loss_total_neutralized": cell_neutral["loss_total"],
        "control_loss_total_neutralized": control_neutral["loss_total"],
        "cell_own_tau": float(cell["expected_flip_tau_start"]),
        "cell_loss_total_at_own_tau_zero_duals": cell_own_tau["loss_total"],
        "cell_carried_lambdas": carried,
        "cell_loss_total_at_own_tau_and_carried_duals": cell_own_both["loss_total"],
        "objective_is_not_tau_blind": cell_own_tau["loss_total"] != cell_neutral["loss_total"],
        "objective_is_not_dual_blind": cell_own_both["loss_total"] != cell_own_tau["loss_total"],
        "penalty_neutralized": control_neutral["margin_constraint_penalty_score"],
        "penalty_at_carried_duals": cell_own_both["margin_constraint_penalty_score"],
        "tau_ref_row_is_schedule_invariant": (
            cell_own_tau["seg_expected_flip_realized_tau_ref"]
            == control_neutral["seg_expected_flip_realized_tau_ref"]
        ),
    }


def _displacement(reference: Mapping[str, torch.Tensor], checkpoint_path: str) -> float:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["live_state_dict"]
    total = 0.0
    for name, value in reference.items():
        total += float((state[name].detach().cpu() - value).double().pow(2).sum())
    return math.sqrt(total)


def smoke_finalize() -> dict[str, Any]:
    """Assemble the two per-arm receipts into the no-op detector + differential."""

    arms = {}
    for label in SMOKE_ARMS:
        path = SMOKE_ROOT / label / "ARM_SMOKE_RESULT.json"
        if not path.is_file():
            raise NG4Error(f"run `smoke --arm {label}` first")
        arms[label] = json.loads(path.read_text(encoding="utf-8"))
    reference = {
        name: value.detach().clone()
        for name, value in torch.load(
            qbr1.QBR_INITIAL_STATE, map_location="cpu", weights_only=False)["state_dict"].items()
    }
    cell_state = arms["continuous"]["segment"]["live_state_sha256"]
    control_state = arms["control"]["segment"]["live_state_sha256"]
    result = {
        "schema": SMOKE_SCHEMA,
        "arm": ARM,
        "axis": "[macOS-CPU bounded mechanism smoke; not a verdict, not contest authority]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "metal_invocations": 0,
        "modal_invocations": 0,
        "contest_eval_invocations": 0,
        "updates_per_arm": 1,
        "arms": arms,
        "no_op_detector": {
            "continuous_live_state_sha256": cell_state,
            "control_live_state_sha256": control_state,
            "states_differ": cell_state != control_state,
            "archives_differ": arms["continuous"]["segment"]["archive"]["sha256"]
            != arms["control"]["segment"]["archive"]["sha256"],
        },
        "training_path_is_unmoved_by_this_landing": {
            "control_live_state_sha256": control_state,
            "ng1_pre_telemetry_cold_reference_sha256": NG1_COLD_REFERENCE_LIVE_STATE_SHA256,
            "reproduces_ng1_cold_reference": control_state == NG1_COLD_REFERENCE_LIVE_STATE_SHA256,
            "argument": (
                "ng1 ran this identical one-update cold segment BEFORE ng2's telemetry row, ng3's "
                "validators and ng4's.  An identical trained state now is a MEASURED proof that "
                "every byte landed since is score-neutral on the training path, which is what "
                "lets a cell built on a moved trainer pin be read against a control that ran "
                "under an older one"
            ),
        },
        "dual_seeding_reached_the_loop": {
            label: arms[label]["history_row_update_1"]["margin_constraint_lambdas"]
            for label in SMOKE_ARMS
        },
        "tau_reached_the_loop": {
            label: arms[label]["history_row_update_1"]["objective"]["tau"] for label in SMOKE_ARMS
        },
        "differential_at_the_controls_tau_and_zero_duals": differential(
            SMOKE_ROOT / "control" / "training_payloads" / "update_000001"
        ),
        "first_update_displacement_l2": {
            label: _displacement(reference, arms[label]["segment"]["checkpoint"]["path"])
            for label in SMOKE_ARMS
        },
        "peak_rss_bytes_per_arm": {
            label: arms[label]["peak_rss_bytes"] for label in SMOKE_ARMS
        },
        "all_payloads_retained": True,
    }
    qbt.atomic_json(SMOKE_ROOT / "BOUNDED_SMOKE_RESULT.json", result)
    return result


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("resolve", help="print r10's terminal objective state + the discontinuity audit")
    sub.add_parser("seal", help="compile, prove the single lever, snapshot, re-root, record")
    smoke = sub.add_parser("smoke", help="one bounded CPU update for ONE arm")
    smoke.add_argument("--arm", required=True, choices=SMOKE_ARMS)
    sub.add_parser("smoke-finalize", help="assemble the no-op detector + differential")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "resolve":
        _cell, control = compile_continuous_objective_cell()
        print(json.dumps(
            {"r10_terminal_state": r10_terminal_state(),
             "discontinuity_audit": discontinuity_audit(control)},
            indent=2, sort_keys=True, default=str))
    elif args.command == "seal":
        print(json.dumps(seal(), indent=2, sort_keys=True, default=str))
    elif args.command == "smoke":
        print(json.dumps(smoke_arm(args.arm), indent=2, sort_keys=True, default=str))
    else:
        print(json.dumps(smoke_finalize(), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
