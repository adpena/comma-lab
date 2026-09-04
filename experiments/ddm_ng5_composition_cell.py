#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_ng5 - seal the first TWO-LEVER burn cell: ng3's tau band x ng4's carried duals.

ng3 (tau band [2*delta_R, delta_R]) and ng4 (continuous objective: tau held at r10's terminal
temperature, duals carried) are the two levers of this generation that acted on the excursion.
This arm composes them.

**THE CORRECTION THIS ARM OWES, STATED BEFORE THE BURN.**  The charter's prior-law line says the
two mechanisms "act on different terms ... neither leaks into the other's block".  Read at the
artifacts, that premise is FALSE for the tau leg: BOTH parents write ``tau_band`` and BOTH write
``expected_flip_tau_start`` / ``expected_flip_tau_end``.  A config has ONE ``tau_band`` key, so
"ng4's config PLUS ng3's tau_band block" can only mean ng3's block REPLACES ng4's -- which is
also the only self-consistent reading of the charter's own final clause.  So:

* the tau leg of this composition is **ng3's alone** (msafe_band, law-resolved, unchanged);
* the genuinely additive leg from ng4 is the **carried duals** (``margin_dual`` +
  ``margin_constraints.initial_lambdas``), which :func:`qbr1.validate_margin_dual_block` reads
  with no reference to tau at all;
* ng4's tau half is **SUBSUMED, not dropped**: ng4 cures a stage entry that re-WIDENS the soft
  band 3.0x (0.05 -> 0.15); ng3's band starts at ``m_safe`` = 0.0437..., which is NARROWER than
  r10's terminal 0.05, so the re-widening ng4 removes is removed harder here.  What is genuinely
  given up is tau CONTINUITY at the exact terminal float -- the entry moves 0.05 -> m_safe, a
  1.14x narrowing instead of a 3.0x widening.

The marginal lever of this cell over ng3 is therefore EXACTLY the carried duals; over ng4 it is
the band.  A third geometry [r10's terminal tau -> delta_R] would keep both halves whole, but it
is a band nobody has measured and it needs a new admissible pair plus a new validator branch --
an unmeasured third lever inside a two-lever composition cell, which is the union-is-not-the-sum
trap ([[m164]], 3.705x).  It is named as follow-on #1 rather than smuggled in here.

**No source change.**  ng3's band branch and ng4's dual branch already exist in
``ddm_qbr1_born_fairform_burn_prep.validate_tau_band_block`` / ``validate_margin_dual_block``, and
ng3's band is already in ``qbt.admissible_expected_flip_tau_bands()``.  This composition therefore
moves NO pin: the trainer, the burn prep and the DSL are byte-identical to ng4's sealed tree
(asserted in the seal receipt).  That is the strongest form of comparability available -- the
composition and ng4 run under the same bytes.

$0: seal + a bounded macOS-CPU mechanism smoke.  No Metal, no Modal, no contest eval.  MAIN (or
gov2's queue driver) fires the cell.
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
    ExpectedFlipTauBandMsafe,
    compile_qbr1_continuous_objective_config,
    compile_qbr1_tau_band_config,
)

ARM = "ddm_ng5_tau_band_x_continuous_objective"
ARM_ROOT = Path("/Volumes/APDataStore/pact/ddm_ng5_tau_band_x_continuous_objective")
SEALED_CONFIGS = ARM_ROOT / "sealed_configs"
SEALED_SOURCE_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_ng5_tau_band_x_continuous_objective")
NG5_RUN_ROOT = qbt.QBR1_RETENTION_ROOT / "ng5_tau_band_x_continuous_objective"
SMOKE_ROOT = NG5_RUN_ROOT / "bounded_smoke"

SEED = 20260902
ARM_NAME = "control_native100"
CELL_ID = f"seed_{SEED}_tau_band_x_continuous_objective_{ARM_NAME}"

SEAL_SCHEMA = "ddm_ng5_composition_seal_receipt.v1"
SMOKE_SCHEMA = "ddm_ng5_composition_bounded_smoke.v1"
SNAPSHOT_SCHEMA = "ddm_ng5_sealed_source_manifest.v1"
QUEUE_SPEC_SCHEMA = "ddm_gv1_cell_queue_spec.v1"

#: the only TOP-LEVEL config keys the composition may move relative to a freshly compiled control.
#: Identical to ng4's allowed set -- the union of ng3's (tau) and ng4's (tau + dual) -- because the
#: composition's key surface IS that union; what separates it from either parent is the CONTENT of
#: ``tau_band`` (ng3's) beside ``margin_dual`` (ng4's), never a wider key set.
ALLOWED_COMPOSITION_MUTATIONS = frozenset(
    {"cell_id", "output", "tau_band", "expected_flip_tau_start", "expected_flip_tau_end",
     "margin_dual", "margin_constraints"}
)

#: the two parents' retained bounded-smoke receipts.  The step-1 state shas are READ from these
#: files, never retyped: the no-op detector must prove this cell's first update differs from BOTH
#: parents', and a retyped sha is a claim about a file rather than a reading of it.
NG3_SMOKE_RESULT = (
    qbt.QBR1_RETENTION_ROOT / "ng3_tau_band" / "bounded_smoke" / "BOUNDED_SMOKE_RESULT.json"
)
NG4_SMOKE_RESULT = (
    qbt.QBR1_RETENTION_ROOT / "ng4_continuous_objective" / "bounded_smoke"
    / "BOUNDED_SMOKE_RESULT.json"
)

#: the parents' burned runs.  ng3 is COMPLETE (5,000/5,000); ng4 was still running when this arm
#: sealed, so its milestone set is read for whatever it has and the pre-registered read names the
#: steps that exist rather than asserting six.
NG3_RUN = NG5_RUN_ROOT.parent / "ng3_tau_band" / "runs" / f"seed_{SEED}_tau_band_{ARM_NAME}"
NG4_RUN = (
    NG5_RUN_ROOT.parent / "ng4_continuous_objective" / "runs"
    / f"seed_{SEED}_continuous_objective_{ARM_NAME}"
)

#: the completed seed-20260902 cold control.  Read live so the bar is the control's own retained
#: milestone, not a decimal copied between memos.
COLD_CONTROL_RUN = Path(
    "/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/runs/seed_20260902/control_native100"
)

#: ng1 ran this identical one-update cold segment BEFORE ng2's telemetry row and ng3's/ng4's
#: validators.  ng4 re-measured it bit-identical; this arm re-measures it a third time, because it
#: is the only thing that licenses reading cells sealed on different trainer pins against one
#: another.  Read LIVE from ng4's receipt -- the literal below is the value ng4 recorded and is
#: used only to cross-check that the file still says what the memo says.
NG1_COLD_REFERENCE_KEY = "ng1_pre_telemetry_cold_reference_sha256"

SHARED_INPUT_PATHS = ("upstream", "experiments/results/mlx_fleet_gt_cache")

LEVER_SURFACE = (
    "experiments/ddm_qbt1_qbflow_trainer.py",
    "experiments/ddm_qbr1_born_fairform_burn_prep.py",
    "experiments/ddm_ng5_composition_cell.py",
    "src/tac/witness_dsl/curriculum_dsl.py",
)

#: the source files whose bytes must be IDENTICAL to ng4's sealed tree for the no-pin-movement
#: claim to hold.  ``ddm_ng5_composition_cell.py`` is deliberately absent: this arm's own script
#: never runs inside the sealed tree (the cell fires ``run-config``), so its bytes cannot affect
#: the burn.
NO_PIN_MOVEMENT_SURFACE = (
    "experiments/ddm_qbt1_qbflow_trainer.py",
    "experiments/ddm_qbr1_born_fairform_burn_prep.py",
    "src/tac/witness_dsl/curriculum_dsl.py",
)

NG4_SEALED_TREE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_ng4_continuous_objective/sealed_source_50e2cd2808"
)

SMOKE_ARMS = ("composition", "control")


class NG5Error(RuntimeError):
    """Fail-closed ng5 contract error."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise NG5Error(f"{label} is absent: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. compile the composition, and audit the tau collision it had to resolve
# ---------------------------------------------------------------------------
def compile_composition_cell() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compile the composition, a matched control, and the tau-collision audit.

    Both configs come from ``qbr1.compile_cell`` in THIS tree, so their pins are this tree's pins
    and only the two levers separate them.  Both DSL levers are compiled fresh (never read off
    either parent's sealed config on disk), so every value in both blocks is re-derived here:
    ng3's band through ``margin_band_satisficing_threshold_v1``, ng4's duals from r10's two
    sha-pinned artifacts.
    """

    initial_state = qbt.file_fact(qbr1.QBR_INITIAL_STATE)
    control = qbr1.compile_cell(SEED, ARM_NAME, initial_state)

    band_block, band_start, band_end = compile_qbr1_tau_band_config(ExpectedFlipTauBandMsafe())
    held_block, held_start, held_end, dual_block, initial_lambdas = (
        compile_qbr1_continuous_objective_config(
            ContinuousObjectiveFromR10(qbr1.R10_CONFIG, qbr1.R10_CHECKPOINT)
        )
    )

    cell = copy.deepcopy(control)
    cell["cell_id"] = CELL_ID
    cell["output"] = str((NG5_RUN_ROOT / "runs" / CELL_ID).resolve())
    # the tau leg is ng3's; ng4's held block is compiled only so the collision can be MEASURED
    # rather than asserted, and is then discarded (recorded in the audit below).
    cell["tau_band"] = band_block
    cell["expected_flip_tau_start"] = band_start
    cell["expected_flip_tau_end"] = band_end
    cell["margin_dual"] = dual_block
    cell["margin_constraints"] = copy.deepcopy(control["margin_constraints"])
    cell["margin_constraints"]["initial_lambdas"] = dict(sorted(initial_lambdas.items()))
    qbr1.validate_config(cell, require_launch_authority=False)

    audit = tau_collision_audit(
        control=control,
        band=(band_start, band_end),
        held=(held_start, held_end),
        band_mode=str(band_block["mode"]),
        held_mode=str(held_block["mode"]),
    )
    return cell, control, audit


def tau_collision_audit(
    *,
    control: Mapping[str, Any],
    band: tuple[float, float],
    held: tuple[float, float],
    band_mode: str,
    held_mode: str,
) -> dict[str, Any]:
    """MEASURE that the two parents collide on tau, and record which geometry the cell takes.

    The charter's prior-law line assumed the levers act on different terms.  They do not: both
    write the same two trainer-read scalars and the same ``tau_band`` key.  This function turns
    that into a number rather than a sentence, so the memo's correction is a reading.
    """

    control_band = (float(control["expected_flip_tau_start"]), float(control["expected_flip_tau_end"]))
    return {
        "premise_checked": "the two parent levers act on different config terms",
        "premise_holds": False,
        "why": (
            "both parents write tau_band and both write expected_flip_tau_start/end; a config has "
            "one tau_band key, so the composition must choose ONE tau geometry"
        ),
        "control_band": list(control_band),
        "ng3_msafe_band": {"mode": band_mode, "start": band[0], "end": band[1]},
        "ng4_held_band": {"mode": held_mode, "start": held[0], "end": held[1]},
        "composition_takes": "ng3_msafe_band",
        "entry_step_control_widens_by": control_band[0] / held[0],
        "entry_step_composition_narrows_by": held[0] / band[0],
        "ng4_tau_half_status": (
            "SUBSUMED: ng4 cures a stage entry that re-WIDENS the soft band; the composition's "
            "entry NARROWS it instead, so the re-widening is removed more strongly. What is given "
            "up is tau continuity at r10's exact terminal float"
        ),
        "dual_leg_is_orthogonal_to_tau": (
            "validate_margin_dual_block reads bounds, step size, mode and the multipliers and "
            "never reads a temperature; the dual leg composes with ANY admissible band"
        ),
        "third_geometry_not_taken": {
            "band": [held[0], band[1]],
            "why_not": (
                "[r10 terminal tau -> delta_R] would keep both halves whole but is a band nobody "
                "has measured and needs a new admissible pair plus a new validator branch -- an "
                "unmeasured third lever inside a two-lever cell ([[m164]])"
            ),
        },
    }


def validate_composition_cell(
    cell: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed unless the cell is the control plus exactly ng3's band and ng4's duals."""

    # The two NAMED refusals come first, ahead of the generic held-field sweep: a third lever and
    # a warm optimizer would both be caught by the sweep, but with a message that names a key
    # rather than the mistake.  A gate that fires with the wrong reason costs the next reader the
    # diagnosis, so the specific check owns the case it is about.
    if cell.get("area_cap") is not None:
        raise NG5Error("ng5 composes TWO levers: ng2's area cap is not one of them")
    if cell.get("resume_from") is not None:
        raise NG5Error("the composition is a COLD-OPTIMIZER transition: resume_from stays null")
    differing = {key for key in set(cell) | set(control) if cell.get(key) != control.get(key)}
    extra = differing - ALLOWED_COMPOSITION_MUTATIONS
    if extra:
        raise NG5Error(f"composition cell moved more than the two levers: {sorted(extra)}")
    for key in ("objective", "ema", "schedule", "initial_state", "learning_rate", "pair_ids",
                "selection_weights", "total_steps", "milestones", "seed", "resume_from",
                "area_cap", "chunk_pairs", "checkpoint_every_steps", "device", "source_pins"):
        if cell.get(key) != control.get(key):
            raise NG5Error(f"composition cell moved a held field: {key}")
    constraints_moved = {
        key for key in set(cell["margin_constraints"]) | set(control["margin_constraints"])
        if cell["margin_constraints"].get(key) != control["margin_constraints"].get(key)
    }
    if constraints_moved != {"initial_lambdas"}:
        raise NG5Error(
            "the ONLY margin-constraint field the dual lever may move is initial_lambdas; moved "
            f"{sorted(constraints_moved)}"
        )
    if str(cell["tau_band"].get("mode")) != "msafe_band":
        raise NG5Error("the composition's tau leg must be ng3's law-resolved band")
    if str(cell["margin_dual"].get("mode")) != qbr1.R10_CONTINUATION_MODE:
        raise NG5Error("the composition's dual leg must be ng4's r10 continuation")
    if cell.get("launch_authorized") is not False:
        raise NG5Error("seal must leave the cell unauthorized")
    for lane in ("scorer_lane", "metal_lane"):
        if cell[lane].get("claimed") is not False or cell[lane].get("claim_id") is not None:
            raise NG5Error(f"seal must leave {lane} unbound for MAIN")
    bounds = {name: float(v) for name, v in control["margin_constraints"]["bounds"].items()}
    control_lambdas = qbr1.initial_margin_constraint_lambdas(control, bounds)
    cell_lambdas = qbr1.initial_margin_constraint_lambdas(cell, bounds)
    if any(value != 0.0 for value in control_lambdas.values()):
        raise NG5Error("the matched control must still seed its duals from zero")
    if cell_lambdas != {k: float(v) for k, v in cell["margin_constraints"]["initial_lambdas"].items()}:
        raise NG5Error("the executable dual seeding does not read the cell's carried multipliers")
    return {
        "differing_keys": sorted(differing),
        "allowed": sorted(ALLOWED_COMPOSITION_MUTATIONS),
        "margin_constraints_fields_moved": sorted(constraints_moved),
        "tau_leg": {"source": "ddm_ng3", "mode": cell["tau_band"]["mode"],
                    "band": [cell["expected_flip_tau_start"], cell["expected_flip_tau_end"]]},
        "dual_leg": {"source": "ddm_ng4", "mode": cell["margin_dual"]["mode"],
                     "initial_lambdas": cell["margin_constraints"]["initial_lambdas"]},
        "held_fields_identical": True,
        "area_cap_absent": True,
        "executable_dual_seeding": {"cell": cell_lambdas, "control": control_lambdas},
        "transition": "COLD OPTIMIZER (fresh AdamW, resume_from null) -- the same optimizer "
                      "transition both parents and the control took, so the cell isolates the "
                      "two objective levers and never composes ng1's warm moments ([[m164]])",
    }


def no_pin_movement_receipt() -> dict[str, Any]:
    """MEASURE that this composition moves no source pin relative to ng4's sealed tree.

    ng2, ng3, MAIN and ng4 each moved a pin.  This arm changes no trainer, burn-prep or DSL byte,
    so the composition runs under literally the same interpreter as ng4 -- which is the strongest
    comparability a cell can have against its own parent, and it is a file-hash reading, not a
    claim about intent.
    """

    rows: dict[str, Any] = {}
    identical = True
    for relative in NO_PIN_MOVEMENT_SURFACE:
        repo_fact = qbt.file_fact(REPO / relative)
        tree_path = NG4_SEALED_TREE / relative
        tree_sha = qbt.file_fact(tree_path)["sha256"] if tree_path.is_file() else None
        same = tree_sha is not None and tree_sha == repo_fact["sha256"]
        identical = identical and same
        rows[relative] = {"repo_sha256": repo_fact["sha256"], "ng4_sealed_tree_sha256": tree_sha,
                          "identical": same}
    return {
        "surface": list(NO_PIN_MOVEMENT_SURFACE),
        "ng4_sealed_tree": str(NG4_SEALED_TREE),
        "ng4_sealed_tree_present": NG4_SEALED_TREE.is_dir(),
        "files": rows,
        "no_pin_movement": identical,
        "meaning": (
            "identical bytes on the whole lever surface => the composition and ng4 run under the "
            "same trainer, burn prep and DSL; the only thing separating them is the config"
        ),
    }


# ---------------------------------------------------------------------------
# 2. the pre-registered read: both parents and the cold control, read LIVE
# ---------------------------------------------------------------------------
def milestone_rows(run: Path, label: str, *, required: bool) -> dict[str, Any]:
    """Read a run's retained milestones LIVE.  Absent steps are reported, never invented."""

    rows: dict[str, Any] = {}
    for step in qbr1.MILESTONES:
        path = run / "milestones" / f"step_{step:06d}" / "MILESTONE.json"
        if not path.is_file():
            if required:
                raise NG5Error(f"{label} lacks milestone {step}: {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows[str(step)] = {
            "S_hat": float(payload["S_hat"]),
            "d_seg_hat": float(payload["d_seg_hat"]),
            "d_pose_hat": float(payload["d_pose_hat"]),
            "archive_bytes_exact": int(payload["archive_bytes_exact"]),
        }
    return {"label": label, "run": str(run), "steps_present": sorted(rows, key=int), "rows": rows}


def pre_registered_read() -> dict[str, Any]:
    """The read this cell is judged against, fixed before it burns.

    Three reference curves, all read live: the cold control, ng3 (complete), ng4 (whatever it has
    when the seal runs -- it was mid-burn).  The verdict WORDS are fixed here too, so the harvest
    cannot invent a fourth.
    """

    cold = milestone_rows(COLD_CONTROL_RUN, "cold_control_of_record", required=True)
    ng3 = milestone_rows(NG3_RUN, "ddm_ng3_tau_band", required=False)
    ng4 = milestone_rows(NG4_RUN, "ddm_ng4_continuous_objective", required=False)
    start = float(cold["rows"]["0"]["S_hat"])
    ng3_terminal = ng3["rows"].get("5000")
    return {
        "references": {"cold_control": cold, "ng3_tau_band": ng3, "ng4_continuous_objective": ng4},
        "start_S_hat": start,
        "start_d_seg_hat": float(cold["rows"]["0"]["d_seg_hat"]),
        "ng3_terminal_S_hat": None if ng3_terminal is None else float(ng3_terminal["S_hat"]),
        "ng4_terminal_S_hat_at_seal_time": (
            None if "5000" not in ng4["rows"] else float(ng4["rows"]["5000"]["S_hat"])
        ),
        "read_from": (
            "MILESTONE.json under <run>/milestones/step_*/ -- the history rows the queue driver "
            "reads carry NO S_hat, so the S_hat verdict is a milestone read at harvest and the "
            "queue spec's falsifiers are the mechanism ones the driver can actually see"
        ),
        "verdict_words": ["BELOW-BOTH", "REDUNDANT", "ANTAGONISTIC"],
        "rule": {
            "BELOW-BOTH": (
                "S_hat(5,000) is BELOW both parents' terminals AND below the start "
                f"({start!r}); the two levers are sub-additive but same-signed"
            ),
            "REDUNDANT": (
                "S_hat(5,000) is at or above ng3's terminal but still below the cold control's; "
                "the levers are the same mechanism seen twice and the dual carry adds nothing "
                "on top of the band"
            ),
            "ANTAGONISTIC": (
                "S_hat(5,000) is at or above the cold control's terminal; name which term "
                "flipped (d_seg / d_pose / bytes) from the milestone decomposition"
            ),
        },
        "read_the_decomposition": (
            "the control's endpoint excess is 91.20% d_seg (ng1's recomputation); a cell that "
            "'fixed' S_hat by moving bytes or pose would be a different finding"
        ),
    }


def parent_step1_states() -> dict[str, Any]:
    """Read both parents' retained step-1 state shas from their own smoke receipts."""

    ng3 = _read_json(NG3_SMOKE_RESULT, "ng3 bounded-smoke result")["no_op_detector"]
    ng4 = _read_json(NG4_SMOKE_RESULT, "ng4 bounded-smoke result")
    ng4_noop = ng4["no_op_detector"]
    unmoved = ng4["training_path_is_unmoved_by_this_landing"]
    control = str(ng3["control_live_state_sha256"])
    if control != str(ng4_noop["control_live_state_sha256"]):
        raise NG5Error(
            "the two parents' control arms disagree on the shared step-1 state; the no-op "
            "detector's reference is not shared and the comparison is invalid"
        )
    return {
        "ng3_tau_band_live_state_sha256": str(ng3["tau_band_live_state_sha256"]),
        "ng4_continuous_live_state_sha256": str(ng4_noop["continuous_live_state_sha256"]),
        "shared_control_live_state_sha256": control,
        "ng1_pre_telemetry_cold_reference_sha256": str(unmoved[NG1_COLD_REFERENCE_KEY]),
        "sources": {"ng3": str(NG3_SMOKE_RESULT), "ng4": str(NG4_SMOKE_RESULT)},
    }


# ---------------------------------------------------------------------------
# 3. sealed source tree
# ---------------------------------------------------------------------------
def link_shared_inputs(destination: Path) -> dict[str, str]:
    """Point the sealed tree's large pinned inputs at the repo, the way wc3's tree does."""

    linked: dict[str, str] = {}
    for relative in (*SHARED_INPUT_PATHS, ".venv"):
        target = REPO / relative
        if not target.exists():
            raise NG5Error(f"shared input is absent from the repo: {target}")
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
        raise NG5Error(
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
        raise NG5Error(f"refusing to snapshot a dirty lever surface:\n{dirty}")
    if destination.exists():
        raise NG5Error(f"sealed source already exists (never overwrite a seal): {destination}")
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
    qbt.atomic_json(destination / "NG5_SEALED_SOURCE_MANIFEST.json", manifest)
    qbt.atomic_json(ARM_ROOT / "SEALED_SOURCE_MANIFEST.json", manifest)
    return manifest


def reroot_sealed_config(sealed_tree: Path) -> dict[str, Any]:
    """Re-root the sealed config's source_pins to the tree that will RUN it, then RE-VALIDATE."""

    config_in = SEALED_CONFIGS / f"{CELL_ID}.json"
    if not config_in.is_file():
        raise NG5Error("run `seal` before `reroot`")
    config_out = SEALED_CONFIGS / f"{CELL_ID}.rerooted.json"
    receipt_out = ARM_ROOT / "PIN_REROOT_RECEIPT.json"
    completed = subprocess.run(
        [str(REPO / ".venv/bin/python"),
         str(REPO / "experiments/ddm_reseal_pins_inside_sealed_tree.py"),
         "--config-in", str(config_in), "--sealed-tree", str(sealed_tree),
         "--config-out", str(config_out), "--receipt-out", str(receipt_out)],
        cwd=REPO, text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise NG5Error(f"pin re-root failed:\n{completed.stderr.strip()[-2000:]}")
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
        " 'tau_band_mode':c['tau_band']['mode'],"
        " 'margin_dual_mode':c['margin_dual']['mode'],"
        " 'initial_lambdas':c['margin_constraints']['initial_lambdas']}))"
    )
    completed = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-c", script],
        cwd=sealed_tree, text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise NG5Error(
            f"the sealed tree REFUSES the config it is meant to fire:"
            f"\n{completed.stderr.strip()[-2500:]}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# 4. seal
# ---------------------------------------------------------------------------
def refuse_if_the_run_has_already_started(run_output: Path) -> None:
    if (run_output / "RESULT.json").exists() or (run_output / "history.jsonl").exists():
        raise NG5Error(f"refusing to re-seal a cell whose run has already started: {run_output}")


def recompile_determinism(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Two compiles of the same two levers must agree everywhere this arm OWNS."""

    again, _control, _audit = compile_composition_cell()
    moved = sorted(key for key in set(cell) | set(again) if cell.get(key) != again.get(key))
    for owned in ("tau_band", "margin_dual", "margin_constraints",
                  "expected_flip_tau_start", "expected_flip_tau_end"):
        if owned in moved:
            raise NG5Error(
                f"a block this arm owns must be byte-stable across compiles, and is not: {owned}"
            )
    return {
        "keys_that_moved": moved,
        "arm_owned_blocks_byte_stable": True,
        "ema_law_identity_stable_under_the_lineage_comparator": (
            qbt.stable_ema_law_identity(cell["ema"]) == qbt.stable_ema_law_identity(again["ema"])
        ),
        "residual": ("ema.lawref.resolved_at is a dated observation the QBR1 lineage keeps inside "
                     "the config; the sealed sha is a FILE property, verified with shasum"),
    }


def seal() -> dict[str, Any]:
    """Compile, prove the two levers, snapshot the source, re-root the pins, and record it."""

    started = time.monotonic()
    cell, control, audit = compile_composition_cell()
    refuse_if_the_run_has_already_started(Path(cell["output"]))
    diff = validate_composition_cell(cell, control)
    pins = no_pin_movement_receipt()
    parents = parent_step1_states()
    read = pre_registered_read()
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
        "honest_frame": (
            "burn-QUALITY cell on the born vehicle (S_hat ~0.39-0.43 at ~106 KB), not a pointer "
            "mover: md1's persistent-partition closure stands (62% of born d_seg is "
            "optimizer-unreachable; schedule levers <= 1.61x). No S_hat delta from this cell is "
            "progress toward sub-0.12"
        ),
        "tau_collision_audit": audit,
        "two_levers": diff,
        "no_pin_movement_vs_ng4": pins,
        "parent_step1_states": parents,
        "pre_registered_read": read,
        "recompile_determinism": determinism,
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
    control_path = (
        SEALED_CONFIGS / f"matched_control_of_record_seed_{SEED}_{ARM_NAME}.reference.json"
    )
    if not cell_path.is_file() or not control_path.is_file():
        raise NG5Error("run `seal` before `smoke`")
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    control = json.loads(control_path.read_text(encoding="utf-8"))
    control.pop("DO_NOT_FIRE", None)
    return cell, control


def smoke_arm(label: str) -> dict[str, Any]:
    """Run ONE real B=16 CPU update for one arm, in its own process.

    ng4 MEASURED that the high-water of this segment is the B=16 forward+backward itself, ~40.4
    GiB per arm -- NOT two arms summed.  So the split buys isolation and per-arm attribution, not
    a smaller peak, and each arm must be projected at the full measured figure.
    """

    if label not in SMOKE_ARMS:
        raise NG5Error(f"unknown smoke arm {label!r}")
    cell, control = _sealed_configs()
    config = cell if label == "composition" else control
    started = time.monotonic()
    segment = qbr1._run_resume_smoke_segment(config, SMOKE_ROOT / label, stop_after=1)
    row = json.loads(
        (SMOKE_ROOT / label / "history.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
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
        raise NG5Error(f"no retained payload under {payload_dir}")
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

    Both parents measured this same quantity independently and both got ``loss_total``
    1.0765775442123413.  If the composition's blocks leaked anything into the objective FUNCTION
    rather than acting only through the caller-supplied (tau, lambda) states, this is where it
    shows.  The sister halves -- that the objective is neither tau-blind nor dual-blind -- are
    measured beside it so the test cannot pass by blindness.
    """

    inputs = _objective_inputs_from_retained(payload_dir)
    cell, control = _sealed_configs()
    zero = dict.fromkeys(control["margin_constraints"]["bounds"], 0.0)
    carried = {k: float(v) for k, v in cell["margin_constraints"]["initial_lambdas"].items()}

    def evaluate(
        config: Mapping[str, Any], tau: float, lambdas: Mapping[str, float]
    ) -> dict[str, float]:
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
    identical = {
        name: cell_neutral[name] == control_neutral[name] for name in sorted(control_neutral)
    }
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
        "cross_arm_agreement": _parent_neutralized_loss_totals(cell_neutral["loss_total"]),
    }


def _parent_neutralized_loss_totals(observed: float) -> dict[str, Any]:
    """Cross-check the neutralized loss against BOTH parents' independently measured value.

    ng3 and ng4 each measured this quantity on their own retained field and each recorded it in
    their own receipt.  Reading those two numbers back (rather than retyping the decimal the
    memos quote) turns "the differential agrees with the lineage" into a file reading, and a
    disagreement would mean the objective FUNCTION moved between the three cells.
    """

    parents: dict[str, Any] = {"observed": float(observed)}
    for label, path, key in (
        ("ng3", NG3_SMOKE_RESULT, "differential_at_a_shared_tau"),
        ("ng4", NG4_SMOKE_RESULT, "differential_at_the_controls_tau_and_zero_duals"),
    ):
        if not path.is_file():
            parents[label] = None
            continue
        block = json.loads(path.read_text(encoding="utf-8")).get(key, {})
        value = block.get("control_loss_total_neutralized")
        if value is None:
            value = block.get("control_loss_total_at_shared_tau")
        parents[label] = None if value is None else float(value)
    measured = [v for k, v in parents.items() if k != "observed" and v is not None]
    parents["agrees_with_every_parent_that_measured_it"] = bool(measured) and all(
        v == float(observed) for v in measured
    )
    return parents


def _displacement(reference: Mapping[str, torch.Tensor], checkpoint_path: str) -> float:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["live_state_dict"]
    total = 0.0
    for name, value in reference.items():
        total += float((state[name].detach().cpu() - value).double().pow(2).sum())
    return math.sqrt(total)


def smoke_finalize() -> dict[str, Any]:
    """Assemble the two per-arm receipts into the THREE-WAY no-op detector + differential."""

    arms = {}
    for label in SMOKE_ARMS:
        path = SMOKE_ROOT / label / "ARM_SMOKE_RESULT.json"
        if not path.is_file():
            raise NG5Error(f"run `smoke --arm {label}` first")
        arms[label] = json.loads(path.read_text(encoding="utf-8"))
    reference = {
        name: value.detach().clone()
        for name, value in torch.load(
            qbr1.QBR_INITIAL_STATE, map_location="cpu", weights_only=False)["state_dict"].items()
    }
    parents = parent_step1_states()
    cell_state = arms["composition"]["segment"]["live_state_sha256"]
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
            "composition_live_state_sha256": cell_state,
            "control_live_state_sha256": control_state,
            "ng3_tau_band_live_state_sha256": parents["ng3_tau_band_live_state_sha256"],
            "ng4_continuous_live_state_sha256": parents["ng4_continuous_live_state_sha256"],
            "differs_from_control": cell_state != control_state,
            "differs_from_ng3": cell_state != parents["ng3_tau_band_live_state_sha256"],
            "differs_from_ng4": cell_state != parents["ng4_continuous_live_state_sha256"],
            "differs_from_all_three": (
                cell_state != control_state
                and cell_state != parents["ng3_tau_band_live_state_sha256"]
                and cell_state != parents["ng4_continuous_live_state_sha256"]
            ),
            "archives_differ": arms["composition"]["segment"]["archive"]["sha256"]
            != arms["control"]["segment"]["archive"]["sha256"],
            "why_three_way": (
                "a composition whose first update reproduced EITHER parent's state would be that "
                "parent wearing a new cell_id; the control comparison alone cannot catch that"
            ),
        },
        "training_path_is_unmoved_by_this_landing": {
            "control_live_state_sha256": control_state,
            "ng1_pre_telemetry_cold_reference_sha256": parents[
                "ng1_pre_telemetry_cold_reference_sha256"
            ],
            "reproduces_ng1_cold_reference": control_state
            == parents["ng1_pre_telemetry_cold_reference_sha256"],
            "argument": (
                "ng1 ran this identical one-update cold segment before ng2's telemetry row and "
                "ng3's/ng4's validators; ng4 re-measured it identical. A third identical reading "
                "is what lets cells sealed on different trainer pins be read against each other"
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
        "peak_rss_bytes_per_arm": {label: arms[label]["peak_rss_bytes"] for label in SMOKE_ARMS},
        "all_payloads_retained": True,
    }
    qbt.atomic_json(SMOKE_ROOT / "BOUNDED_SMOKE_RESULT.json", result)
    return result


# ---------------------------------------------------------------------------
# 6. the fire order: a gov2 queue spec, emitted from MEASURED values only
# ---------------------------------------------------------------------------
def queue_spec(sealed_tree: Path | None = None) -> dict[str, Any]:
    """Emit the gov2 queue spec this cell fires through.

    The declared peak is ``"from_ledger"``: gov2's driver reads
    ``.omx/state/measured_peaks.jsonl`` for the named family, so the number that governs the fire
    is a MEASUREMENT of a prior run of this exact segment rather than anything typed here.
    """

    rerooted = SEALED_CONFIGS / f"{CELL_ID}.rerooted.json"
    if not rerooted.is_file():
        raise NG5Error("run `seal` first: the re-rooted config is what the queue fires")
    manifest = _read_json(ARM_ROOT / "SEALED_SOURCE_MANIFEST.json", "sealed source manifest")
    tree = sealed_tree or Path(manifest["root"])
    receipt = _read_json(ARM_ROOT / "SEAL_RECEIPT.json", "seal receipt")
    cell = _read_json(rerooted, "re-rooted config")
    # The driver's falsifiers are DOTTED PATHS INTO A HISTORY ROW, and history rows carry no
    # S_hat -- that lives in MILESTONE.json.  So the driver gets the two MECHANISM falsifiers it
    # can actually see (both are inert-lever detectors: a lever that did not reach the loop), and
    # the S_hat verdict stays the pre-registered milestone read in the seal receipt.  Declaring an
    # S_hat falsifier here would produce a permanent METRIC_ABSENT, which is a gate that never
    # fires wearing the name of one that does.
    falsifiers = [
        {"name": "dual_carry_reached_the_loop", "at_step": 1,
         "metric": "margin_constraint_lambdas.Lane", "op": "lt",
         "threshold": float(cell["margin_dual"]["initial_lambdas"]["Lane"])},
        {"name": "band_reached_the_loop", "at_step": 1,
         "metric": "objective.tau", "op": "gt",
         "threshold": float(cell["expected_flip_tau_start"])},
    ]
    spec = {
        "schema": QUEUE_SPEC_SCHEMA,
        "written_by": ARM,
        "notes": (
            "first TWO-LEVER burn cell: ng3's law-resolved tau band x ng4's carried duals. "
            "Burn-QUALITY only; md1's persistent-partition closure stands and no S_hat delta "
            "from this cell is progress toward sub-0.12"
        ),
        "cells": [
            {
                "cell_id": CELL_ID,
                "sealed_config": str(rerooted),
                "sealed_tree": str(tree),
                "authorized_config": str(ARM_ROOT / "authorized_configs" / f"{CELL_ID}.json"),
                "launcher_argv": [
                    str(tree / ".venv/bin/python"),
                    str(tree / "experiments/ddm_qbr1_born_fairform_burn_prep.py"),
                    "run-config",
                    str(ARM_ROOT / "authorized_configs" / f"{CELL_ID}.json"),
                ],
                "done_receipt": "ng5_composition_DONE.json",
                "scorer_lane_prefix": "ng5_composition_scorer",
                "metal_lane_prefix": "ng5_composition_metal",
                "measured_peak_rss_gib": "from_ledger",
                "peak_family": "ddm_qbr1_born_fairform_burn_prep",
                "control_run_dir": str(COLD_CONTROL_RUN),
                "control_label": "cold_control_of_record_seed_20260902",
                "milestones": list(qbr1.MILESTONES),
                "falsifiers": falsifiers,
                "notes": (
                    "tau leg = ng3 (msafe_band, law-resolved); dual leg = ng4 (r10 continuation). "
                    "The two falsifiers here are INERT-LEVER detectors on history rows; the S_hat "
                    "verdict is the milestone read pre-registered in SEAL_RECEIPT.json. "
                    "Verdict words: BELOW-BOTH / REDUNDANT / ANTAGONISTIC. Reference terminals "
                    f"at seal time: cold {receipt['pre_registered_read']['references']['cold_control']['rows']['5000']['S_hat']!r}, "
                    f"ng3 {receipt['pre_registered_read']['ng3_terminal_S_hat']!r}, "
                    f"ng4 {receipt['pre_registered_read']['ng4_terminal_S_hat_at_seal_time']!r}"
                ),
            }
        ],
    }
    qbt.atomic_json(ARM_ROOT / "QUEUE_SPEC.json", spec)
    return spec


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="print the tau-collision audit and the no-pin-movement receipt")
    sub.add_parser("seal", help="compile, prove the two levers, snapshot, re-root, record")
    smoke = sub.add_parser("smoke", help="one bounded CPU update for ONE arm")
    smoke.add_argument("--arm", required=True, choices=SMOKE_ARMS)
    sub.add_parser("smoke-finalize", help="assemble the 3-way no-op detector + differential")
    sub.add_parser("queue-spec", help="emit the gov2 queue spec for the fire")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        _cell, _control, audit = compile_composition_cell()
        payload: dict[str, Any] = {
            "tau_collision_audit": audit,
            "no_pin_movement_vs_ng4": no_pin_movement_receipt(),
            "parent_step1_states": parent_step1_states(),
        }
    elif args.command == "seal":
        payload = seal()
    elif args.command == "smoke":
        payload = smoke_arm(args.arm)
    elif args.command == "smoke-finalize":
        payload = smoke_finalize()
    else:
        payload = queue_spec()
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
