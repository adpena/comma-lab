#!/usr/bin/env python3
"""ddm_ng3 -- seal the expected-flip tau band cell at the MEASURED R-noise scale.

The third sealed race of the QBR1 generation.  The sealed control anneals the expected-flip
temperature ``tau`` linearly ``0.15 -> 0.05``; in the units the score lives in that is
``6.86 -> 2.29`` times the MEASURED round-trip noise floor ``delta_R``, and ddm_gm1 MEASURED at
n600 that **77.7% of the seg gradient is spent there on pixels that are already correct AND
outside** ``m_safe = 2*delta_R`` -- pixels R cannot flip, so the gradient buys nothing the score
can see.  This arm re-bases the band on the law's own output, ``(m_safe, delta_R)``, changing
exactly two numbers and nothing else.

Actions: ``resolve`` (record the law's live resolution) -> ``seal`` -> ``snapshot`` -> ``smoke``.
NO LAUNCH: ``authorized_configs/`` is never written and no Metal/Modal/contest-eval call is made.
"""
from __future__ import annotations

import argparse
import copy
import json
import resource
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
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
    ExpectedFlipTauBandMsafe,
    compile_qbr1_tau_band_config,
)

ARM = "ddm_ng3_tau_band"
ARM_ROOT = Path("/Volumes/APDataStore/pact/ddm_ng3_tau_band")
SEALED_CONFIGS = ARM_ROOT / "sealed_configs"
SEALED_SOURCE_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_ng3_tau_band")
NG3_RUN_ROOT = qbt.QBR1_RETENTION_ROOT / "ng3_tau_band"
SMOKE_ROOT = NG3_RUN_ROOT / "bounded_smoke"

SEED = 20260902
ARM_NAME = "control_native100"
CELL_ID = f"seed_{SEED}_tau_band_{ARM_NAME}"

RESOLUTION_SCHEMA = "ddm_ng3_tau_band_resolution.v1"
SEAL_SCHEMA = "ddm_ng3_tau_band_seal_receipt.v1"
SMOKE_SCHEMA = "ddm_ng3_tau_band_bounded_smoke.v1"
SNAPSHOT_SCHEMA = "ddm_ng3_sealed_source_manifest.v1"

#: the only config keys the band cell may move relative to a freshly compiled control.
ALLOWED_TAU_BAND_MUTATIONS = frozenset(
    {"cell_id", "output", "tau_band", "expected_flip_tau_start", "expected_flip_tau_end"}
)

#: ng1's bounded smoke ran ONE cold update from this same start under code that predated BOTH
#: ng2's telemetry row and this arm's validator work.  Reproducing it bit-for-bit is the measured
#: proof that everything landed since then is score-neutral on the training path.
NG1_COLD_REFERENCE_LIVE_STATE_SHA256 = (
    "27f514180db2b4cda57289bbeb4be5ca8daf64e874921c92ba5c08d613c30973"
)
NG1_COLD_FIRST_UPDATE_DISPLACEMENT_L2 = 0.055886740188786026

#: the cold control of record, recomputed from components by ng1 (advisory, n32, MPS).
COLD_CONTROL_S_HAT = {
    0: 0.39876797285867277,
    1_000: 0.46687521208987615,
    2_000: 0.48567677825279465,
    3_000: 0.47538291701253005,
    4_000: 0.44219037073377010,
    5_000: 0.42514878445269977,
}

#: ddm_gm1's MEASURED numbers, read off ITS OWN table at the two temperatures THIS band actually
#: visits -- not off its headline.  See CHARTER_CORRECTION below: gm1's "Lane loses 1.60-2.08x"
#: is the tau = 0.5*delta_R column, and this band never goes below delta_R.
GM1_MEASURED = {
    "wasted_share_pct_step0": {"tau_0_15": 77.715, "two_delta_r": 42.278, "one_delta_r": 17.326},
    "waste_removed_pct_step0": {"two_delta_r": 45.60, "one_delta_r": 77.70},
    # Lane's share of the total seg gradient, per gm1 section 2, at steps 0 / 2,000 / 5,000.
    "lane_grad_share_pct": {
        "tau_0_15": (17.59, 14.77, 16.22),
        "two_delta_r": (13.73, 9.28, 10.86),
        "one_delta_r": (11.76, 7.76, 8.74),
        "half_delta_r": (11.00, 7.36, 7.81),
    },
    # ratio of Lane's share at tau=0.15 to its share at the band's own endpoints, over the same
    # three milestones.  These are the numbers falsifier 3 is pre-registered against.
    "lane_share_ratio_at_tau_start_two_delta_r": (1.281, 1.592),
    "lane_share_ratio_at_tau_end_one_delta_r": (1.496, 1.903),
    "lane_share_ratio_at_half_delta_r_OUT_OF_BAND": (1.599, 2.077),
    "schedule_leg_legacy_pct": -41.30,
    "schedule_leg_band_pct": -8.57,
}

#: A MEASURED correction to this arm's own charter, recorded before the burn rather than after.
#:
#: The charter pre-registered falsifier 3 as "Lane's share of the seg gradient at step 0 under the
#: band is 1.6-2.1x lower than under the control's tau", citing gm1's headline "Lane loses
#: 1.60-2.08x of its relative gradient share".  Read at source, that headline is gm1's own
#: sentence "as tau falls 0.15 -> 0.5 delta_R" -- it is the 0.5*delta_R COLUMN.  This band runs
#: 2*delta_R -> 1*delta_R and never reaches 0.5*delta_R, so the charter transferred a constant
#: from outside the band's own range ([[m143]] cross-regime transfer at the column axis).
#:
#: Recomputed from gm1's table at the temperatures this band visits: the Lane cost is
#: 1.281-1.592x at tau_start and 1.496-1.903x at tau_end.  The band is GENTLER on Lane than the
#: charter assumed, which weakens the Lane objection to it -- so the correction runs in the
#: direction that flatters this arm's own cell, which is exactly why it is stated in the source
#: and asserted by a test rather than left in prose.
CHARTER_CORRECTION_FALSIFIER_3 = {
    "charter_text": (
        "Lane's share of the seg gradient at step 0 under the band is 1.6-2.1x lower than under "
        "the control's tau"
    ),
    "why_it_is_wrong": (
        "gm1's 1.60-2.08x is measured at tau = 0.5*delta_R; this band's endpoints are "
        "2*delta_R and 1*delta_R, so 0.5*delta_R is outside its range at every step"
    ),
    "corrected_step0_ratio_at_tau_start": 1.281,
    "corrected_step0_ratio_at_tau_end": 1.496,
    "corrected_range_over_all_three_milestones": {
        "tau_start_two_delta_r": (1.281, 1.592),
        "tau_end_one_delta_r": (1.496, 1.903),
    },
    "direction": "the correction LOWERS the predicted Lane cost, i.e. it favours this arm's cell",
}


class NG3Error(RuntimeError):
    """Fail-closed NG3 contract error."""


# ---------------------------------------------------------------------------
# 1. resolve -- the band is the law's output, never a literal in this file
# ---------------------------------------------------------------------------
def resolve(output: Path | None = None) -> dict[str, Any]:
    """Record the live resolution of the band through the registered law.

    There is no per-vehicle measurement to derive here -- unlike ng2's stiffness, the band IS the
    law's output -- so this action exists to make the resolution a dated, hashed receipt rather
    than an implicit side effect of compiling.
    """

    started = time.monotonic()
    from tac.canonical_equations.margin_band_satisficing_threshold_20260712 import (
        resolve_margin_band_threshold,
    )

    lever = ExpectedFlipTauBandMsafe()
    block, start, end = compile_qbr1_tau_band_config(lever)
    # the sealed config drops the LawRef's volatile observation time so its bytes are stable;
    # this dated receipt is exactly where that timestamp belongs, so it is kept here IN FULL.
    full_manifest = resolve_margin_band_threshold().lawref_manifest
    legacy_lever = ExpectedFlipTauBandMsafe(mode="legacy")
    legacy_block, legacy_start, legacy_end = compile_qbr1_tau_band_config(legacy_lever)
    receipt = {
        "schema": RESOLUTION_SCHEMA,
        "arm": ARM,
        "axis": "[macOS-CPU advisory . law resolution only . NON-PROMOTABLE]",
        "score_claim": False,
        "promotion_eligible": False,
        "band": {"start": start, "end": end},
        "legacy_band": {"start": legacy_start, "end": legacy_end},
        "band_in_delta_r_units": {"start": start / block["delta_r"], "end": end / block["delta_r"]},
        "legacy_band_in_delta_r_units": {
            "start": legacy_start / block["delta_r"],
            "end": legacy_end / block["delta_r"],
        },
        "tau_band_block": block,
        "lawref_manifest_full_including_volatile_fields": full_manifest,
        "lever_notes": lever.notes,
        "gm1_measured": GM1_MEASURED,
        "charter_correction_falsifier_3": CHARTER_CORRECTION_FALSIFIER_3,
        "tau_reference_held_fixed": {
            "value": float(qbt.EXPECTED_FLIP_TAU_REFERENCE),
            "why": (
                "ddm_ng2's fixed-tau telemetry row exists so cells are comparable across "
                "DIFFERENT schedules.  Moving it with the band would destroy exactly the "
                "comparability it was added for, so it stays at 0.05 -- above this band's whole "
                "range, which is the point: it is a fixed ruler, not a training temperature."
            ),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    qbt.atomic_json(output or (ARM_ROOT / "RESOLUTION.json"), receipt)
    return receipt


# ---------------------------------------------------------------------------
# 2. seal
# ---------------------------------------------------------------------------
def compile_tau_band_cell() -> tuple[dict[str, Any], dict[str, Any]]:
    """Compile the band cell and a matched control THROUGH the burn prep's own compile path.

    Both come from ``qbr1.compile_cell`` in THIS tree, so their pins are this tree's pins and the
    only thing separating them is the lever.  The sealed control on disk is deliberately NOT
    deep-copied: its pins name an older trainer, so inheriting them would seal a config no tree
    can satisfy (ng2's method, and for the same reason).
    """

    initial_state = qbt.file_fact(qbr1.QBR_INITIAL_STATE)
    control = qbr1.compile_cell(SEED, ARM_NAME, initial_state)
    lever = ExpectedFlipTauBandMsafe()
    block, start, end = compile_qbr1_tau_band_config(lever)
    cell = copy.deepcopy(control)
    cell["cell_id"] = CELL_ID
    cell["output"] = str((NG3_RUN_ROOT / "runs" / CELL_ID).resolve())
    cell["tau_band"] = block
    cell["expected_flip_tau_start"] = start
    cell["expected_flip_tau_end"] = end
    qbr1.validate_config(cell, require_launch_authority=False)
    return cell, control


def validate_tau_band_cell(cell: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the band cell is the control plus exactly the two temperatures."""

    differing = {key for key in set(cell) | set(control) if cell.get(key) != control.get(key)}
    extra = differing - ALLOWED_TAU_BAND_MUTATIONS
    if extra:
        raise NG3Error(f"tau-band cell moved more than the lever: {sorted(extra)}")
    for key in ("objective", "ema", "schedule", "initial_state", "learning_rate",
                "margin_constraints", "pair_ids", "selection_weights", "total_steps",
                "milestones", "seed", "resume_from", "area_cap", "chunk_pairs",
                "checkpoint_every_steps", "device", "source_pins"):
        if cell.get(key) != control.get(key):
            raise NG3Error(f"tau-band cell moved a held field: {key}")
    if cell.get("area_cap") is not None:
        raise NG3Error("ng3 is a ONE-lever race: no area cap may be present")
    if cell.get("resume_from") is not None:
        raise NG3Error("the band cell is a COLD transition: resume_from must stay null")
    if cell.get("launch_authorized") is not False:
        raise NG3Error("seal must leave the band cell unauthorized")
    for lane in ("scorer_lane", "metal_lane"):
        if cell[lane].get("claimed") is not False or cell[lane].get("claim_id") is not None:
            raise NG3Error(f"seal must leave {lane} unbound for MAIN")
    return {
        "differing_keys": sorted(differing),
        "allowed": sorted(ALLOWED_TAU_BAND_MUTATIONS),
        "held_fields_identical": True,
        "area_cap_absent": True,
        "transition": "COLD (fresh AdamW) -- the same transition the control took, so the pair "
                      "isolates the band and never composes it with ng1's warm lever or ng2's cap",
    }


def control_pin_delta() -> dict[str, Any]:
    """The exact pin difference between the sealed control and this tree, with a git receipt."""

    sealed_path = qbr1.CONFIG_ROOT / f"seed_{SEED}_{ARM_NAME}.json"
    if not sealed_path.is_file():
        raise NG3Error(f"sealed control config is absent: {sealed_path}")
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    live = qbr1.verify_inputs()
    moved = {
        name: {"sealed": sealed["source_pins"].get(name, {}).get("sha256"),
               "ng3": live[name]["sha256"]}
        for name in sorted(set(live) | set(sealed["source_pins"]))
        if sealed["source_pins"].get(name, {}).get("sha256") != live[name]["sha256"]
    }
    diffstat = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--",
         "experiments/ddm_qbt1_qbflow_trainer.py",
         "experiments/ddm_qbr1_born_fairform_burn_prep.py"],
        cwd=REPO, text=True, capture_output=True, check=False,
    ).stdout.strip()
    return {
        "sealed_control_config": str(sealed_path),
        "sealed_control_revision": sealed.get("source_revision"),
        "ng3_revision": qbr1.source_revision(),
        "pins_that_moved": moved,
        "uncommitted_diffstat_at_seal": diffstat,
        "why": (
            "the trainer pin had ALREADY moved at ng2 (a new loss term), so a same-pins twin of "
            "the sealed control was never available to this arm; ng3 moves it once more by "
            "WIDENING the tau geometry check from a literal pair to the law-resolved band.  "
            "Neither change touches the training path, which is why the bounded smoke's control "
            "arm reproduces ng1's pre-telemetry cold reference state bit-for-bit -- a MEASURED "
            "receipt, not an argument"
        ),
    }


def refuse_if_the_run_has_already_started(run_output: Path) -> None:
    for marker in ("RESULT.json", "history.jsonl", "milestones"):
        if (run_output / marker).exists():
            raise NG3Error(f"the band cell has already started: {run_output / marker}")


def falsifiers() -> dict[str, Any]:
    """The three pre-registered falsifiers, fixed before the burn (charter verbatim)."""

    return {
        "1_primary_the_band_must_act_on_the_excursion": {
            "test": "S_hat(5,000) < 0.42514878445269977 AND S_hat(2,000) < 0.48567677825279465",
            "endpoint_bar": COLD_CONTROL_S_HAT[5_000],
            "peak_bar": COLD_CONTROL_S_HAT[2_000],
            "if_it_fails": (
                "the band does not act on the excursion; gm1's static gradient-mass read does "
                "not imply a trajectory effect, and vr1's tau row is refuted at FORMULATION "
                "scope for the born object -- not the family"
            ),
        },
        "2_the_fixed_tau_telemetry_must_be_faithful_in_loop": {
            "test": (
                "seg_expected_flip_realized_tau_ref (tau_ref=0.05) AND the annealed "
                "seg_expected_flip_realized at the band's own tau must BOTH peak at the same "
                "milestone as d_seg_hat"
            ),
            "read_from": "<run>/history.jsonl objective.*, NOT MILESTONE.json",
            "if_it_fails": (
                "sd1's fixed-tau faithfulness does not survive inside the loop; the telemetry, "
                "not the lever, is wrong"
            ),
        },
        "3_lane_share_must_fall_as_gm1_measured": {
            "test": (
                "Lane's share of the seg gradient at step 0 falls by 1.281x at tau_start "
                "(2*delta_R) and 1.496x at tau_end (1*delta_R) relative to tau = 0.15, and stays "
                "inside 1.281-1.592x / 1.496-1.903x across the three milestones gm1 measured"
            ),
            "charter_correction": CHARTER_CORRECTION_FALSIFIER_3,
            "gm1_measured": GM1_MEASURED,
            "if_it_fails": "gm1's static read did not transfer to the live loss",
            "caveat": (
                "gm1 MEASURED this on a FROZEN field by re-weighting; the burn measures it on a "
                "field the band itself has moved.  A miss here is weaker evidence than a miss on "
                "falsifier 1, and it is recorded as such BEFORE the burn"
            ),
        },
        "read_the_decomposition_never_the_composite": (
            "the control's damage is 91.20% d_seg; a band that 'fixed' S_hat by moving bytes or "
            "pose would be a different finding"
        ),
    }


def bind_sealed_source(manifest_path: Path | None = None) -> dict[str, Any]:
    path = manifest_path or (ARM_ROOT / "SEALED_SOURCE_MANIFEST.json")
    if not path.is_file():
        return {"status": "NOT_YET_SNAPSHOTTED", "run": "snapshot after seal"}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {"status": "BOUND", "revision": manifest["revision"], "root": manifest["root"],
            "pins_verify_inside_the_sealed_tree": manifest["pins_verify_inside_the_sealed_tree"]}


#: LawRef observation times a recompile legitimately moves.  ``qbt.stable_ema_law_identity``
#: already pops exactly this field from the EMA leg -- it is the lineage's sanctioned comparator
#: -- so ng3 uses the SAME rule rather than inventing a second one.  The ``tau_band`` block does
#: not appear here: this arm strips its volatile field at the DSL compile boundary instead, so
#: the block ng3 owns is byte-stable outright.
VOLATILE_CONFIG_PATHS = (("ema", "lawref", "resolved_at"),)


def _without_volatile(config: Mapping[str, Any]) -> dict[str, Any]:
    stripped = copy.deepcopy(dict(config))
    for path in VOLATILE_CONFIG_PATHS:
        node: Any = stripped
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, Mapping) else None
            if node is None:
                break
        if isinstance(node, dict):
            node.pop(path[-1], None)
    return stripped


def recompile_determinism(cell: Mapping[str, Any], recompiled: Mapping[str, Any]) -> dict[str, Any]:
    """Assert a second compile reproduces the cell, and say EXACTLY how strong that claim is.

    ng3's first seal produced two different shas from two identical compiles, so this check was
    added.  It found a second volatile field immediately -- ``ema.lawref.resolved_at``, a dated
    LawRef observation the lineage has always kept inside the config.  Rather than change a
    shared compile path late in an arm, ng3 compares through the SAME rule the trainer's own
    ``stable_ema_law_identity`` uses, and REPORTS the residual volatility instead of claiming a
    property that does not hold:

    * the ``tau_band`` block this arm owns is byte-stable outright (the DSL strips its volatile
      field at the compile boundary);
    * everything else is identical up to ``VOLATILE_CONFIG_PATHS``;
    * therefore the sealed config's sha is a FILE property MAIN verifies by hashing the file, and
      it is NOT reproducible by recompiling.  Anyone who needs a recompile to match must compare
      through this function.
    """

    moved = sorted(key for key in set(cell) | set(recompiled)
                   if cell.get(key) != recompiled.get(key))
    # checked FIRST so the specific, actionable message wins over the generic one when the block
    # this arm owns is the thing that moved.
    if "tau_band" in moved:
        raise NG3Error("the tau_band block this arm owns must be byte-stable, and is not")
    if _without_volatile(cell) != _without_volatile(recompiled):
        differing = sorted(key for key in set(cell) | set(recompiled)
                           if _without_volatile(cell).get(key)
                           != _without_volatile(recompiled).get(key))
        raise NG3Error(f"seal is not reproducible across two compiles: {differing}")
    return {
        "stable_identity_reproduces": True,
        "raw_bytes_reproduce": moved == [],
        "keys_that_moved_across_two_compiles": moved,
        "volatile_paths_excluded": [".".join(path) for path in VOLATILE_CONFIG_PATHS],
        "tau_band_block_is_byte_stable": True,
        "what_this_means_for_MAIN": (
            "the quoted config sha is a FILE hash -- verify it with shasum on the sealed file, "
            "never by recompiling; a recompile legitimately moves ema.lawref.resolved_at"
        ),
    }


def seal() -> dict[str, Any]:
    started = time.monotonic()
    resolution_path = ARM_ROOT / "RESOLUTION.json"
    if not resolution_path.is_file():
        raise NG3Error("run `resolve` before `seal`")
    cell, control = compile_tau_band_cell()
    refuse_if_the_run_has_already_started(Path(cell["output"]))
    diff = validate_tau_band_cell(cell, control)
    cell_fact = qbt.atomic_json(SEALED_CONFIGS / f"{CELL_ID}.json", cell)
    # The matched control is a REFERENCE recompile: it exists so the single-lever diff and the
    # smoke's control arm are reproducible.  Its `output` still points at the ALREADY COMPLETED
    # seed-20260902 control run, so it must never be fired.
    control_reference = copy.deepcopy(control)
    control_reference["do_not_fire_reference_recompile_only"] = True
    control_fact = qbt.atomic_json(
        SEALED_CONFIGS / f"matched_control_of_record_seed_{SEED}_{ARM_NAME}.reference.json",
        control_reference,
    )
    if json.loads(Path(cell_fact["path"]).read_text(encoding="utf-8")) != cell:
        raise NG3Error("sealed band-cell JSON round trip differs")
    recompiled, _control_again = compile_tau_band_cell()
    determinism = recompile_determinism(cell, recompiled)
    receipt = {
        "schema": SEAL_SCHEMA,
        "arm": ARM,
        "status": "SEALED_AWAITING_MAIN_METAL_CLAIM",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "training_launched": False,
        "metal_invocations": 0,
        "modal_invocations": 0,
        "contest_eval_invocations": 0,
        "cell_id": CELL_ID,
        "config": cell_fact,
        "matched_control_of_record": control_fact,
        "single_lever": diff,
        "pin_delta": control_pin_delta(),
        "sealed_source": bind_sealed_source(),
        "tau_band": cell["tau_band"],
        "expected_flip_tau_start": cell["expected_flip_tau_start"],
        "expected_flip_tau_end": cell["expected_flip_tau_end"],
        "resolution": qbt.file_fact(resolution_path),
        "cold_control_of_record": COLD_CONTROL_S_HAT,
        "falsifiers": falsifiers(),
        "authorized_configs_written": False,
        "recompile_determinism": determinism,
        "elapsed_seconds": time.monotonic() - started,
    }
    qbt.atomic_json(ARM_ROOT / "SEAL_RECEIPT.json", receipt)
    return receipt


# ---------------------------------------------------------------------------
# 3. sealed source snapshot
# ---------------------------------------------------------------------------
SHARED_INPUT_PATHS = ("upstream", "experiments/results/mlx_fleet_gt_cache")

#: the files whose bytes carry this arm's lever; a dirty one refuses the snapshot.
LEVER_SURFACE = (
    "experiments/ddm_qbt1_qbflow_trainer.py",
    "experiments/ddm_qbr1_born_fairform_burn_prep.py",
    "experiments/ddm_ng3_tau_band_cell.py",
    "src/tac/witness_dsl/curriculum_dsl.py",
)


def link_shared_inputs(destination: Path) -> dict[str, str]:
    """Point the sealed tree's large pinned inputs at the repo, the way wc3's tree does."""

    linked: dict[str, str] = {}
    for relative in (*SHARED_INPUT_PATHS, ".venv"):
        target = REPO / relative
        if not target.exists():
            raise NG3Error(f"shared input is absent from the repo: {target}")
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
    """Prove the SEALED tree can compile a cell -- run its own pin gates in its own process."""

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
        raise NG3Error(
            f"the sealed tree cannot verify its own pins:\n{completed.stderr.strip()[-2000:]}"
        )
    return {"status": "PASS", **json.loads(completed.stdout.strip().splitlines()[-1])}


def snapshot_source(destination: Path | None = None) -> dict[str, Any]:
    """Materialize the sealed source tree the band cell fires from, at the current commit."""

    revision = qbr1.source_revision()
    destination = destination or (SEALED_SOURCE_ROOT / f"sealed_source_{revision[:10]}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *LEVER_SURFACE],
        cwd=REPO, text=True, capture_output=True, check=True,
    ).stdout.strip()
    if dirty:
        raise NG3Error(f"refusing to snapshot a dirty lever surface:\n{dirty}")
    if destination.exists():
        raise NG3Error(f"sealed source already exists (never overwrite a seal): {destination}")
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
        "files": {
            name: qbt.file_fact(destination / name)
            for name in (
                *LEVER_SURFACE,
                "experiments/ddm_qbflow_packet.py",
                "src/tac/canonical_equations/margin_band_satisficing_threshold_20260712.py",
            )
        },
    }
    qbt.atomic_json(destination / "NG3_SEALED_SOURCE_MANIFEST.json", manifest)
    qbt.atomic_json(ARM_ROOT / "SEALED_SOURCE_MANIFEST.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# 4. bounded CPU smoke
# ---------------------------------------------------------------------------
def _objective_inputs_from_retained(payload_dir: Path) -> dict[str, Any]:
    """Rebuild real objective inputs from a retained smoke payload -- no new forward pass.

    ALWAYS KEEP THE PAYLOAD, used the way the rule intends: the differential below reads the
    bytes the smoke already materialized instead of paying for a second scorer forward.
    """

    files = sorted(payload_dir.glob("pair_*.npz"))
    if not files:
        raise NG3Error(f"no retained payload under {payload_dir}")
    logits, targets, pose6, target_pose6, camera = [], [], [], [], []
    pair_ids = []
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
        # the native-interface leg is fed the SAME tensor: the differential asks whether the
        # CONFIG changes anything other than tau, so both legs must be held fixed across the two
        # configs -- it is not trying to reproduce a training value.
        "outputs": {"class_logits": stacked.permute(0, 2, 3, 1).contiguous()},
        "camera": torch.stack(camera),
        "pose6": torch.stack(pose6),
        "target_argmax": torch.stack(targets),
        "target_pose6": torch.stack(target_pose6),
        "sample_weights": qbt.no2_sample_weights(pair_ids, torch.device("cpu")),
    }


def _differential_at_a_shared_tau(cell: Mapping[str, Any], control: Mapping[str, Any],
                                  payload_dir: Path) -> dict[str, Any]:
    """At one shared tau the two configs' objectives must be BIT-FOR-BIT equal.

    This is the charter's differential and it is the sharpest single statement this arm can make:
    it proves the lever is EXACTLY the temperature and that nothing else leaked out of the
    ``tau_band`` block into the loss.  The sister half -- that they DIFFER at their own taus --
    is measured beside it, so the test cannot pass by the objective being tau-blind.
    """

    inputs = _objective_inputs_from_retained(payload_dir)
    lambdas = {"Lane": 0.0, "Movable": 0.0}

    def evaluate(config: Mapping[str, Any], tau: float) -> dict[str, float]:
        with torch.no_grad():
            _total, components = qbr1.fairform_objective(
                config, inputs["outputs"], inputs["camera"], inputs["pose6"], inputs["logits"],
                inputs["target_argmax"], inputs["target_pose6"], tau, inputs["sample_weights"],
                lambdas,
            )
        return {name: float(value) for name, value in components.items()}

    shared_tau = float(control["expected_flip_tau_start"])
    band_at_shared = evaluate(cell, shared_tau)
    control_at_shared = evaluate(control, shared_tau)
    identical = {
        name: band_at_shared[name] == control_at_shared[name] for name in sorted(control_at_shared)
    }
    band_at_own = evaluate(cell, float(cell["expected_flip_tau_start"]))
    return {
        "shared_tau": shared_tau,
        "pairs": inputs["pair_ids"],
        "components_bit_identical_at_shared_tau": all(identical.values()),
        "per_component_identical": identical,
        "band_loss_total_at_shared_tau": band_at_shared["loss_total"],
        "control_loss_total_at_shared_tau": control_at_shared["loss_total"],
        "band_own_tau": float(cell["expected_flip_tau_start"]),
        "band_loss_total_at_own_tau": band_at_own["loss_total"],
        "objective_is_not_tau_blind": band_at_own["loss_total"] != band_at_shared["loss_total"],
        "seg_realized_at_shared_tau": control_at_shared["seg_expected_flip_realized"],
        "seg_realized_at_band_tau": band_at_own["seg_expected_flip_realized"],
        "tau_ref_row_is_schedule_invariant": (
            band_at_own["seg_expected_flip_realized_tau_ref"]
            == control_at_shared["seg_expected_flip_realized_tau_ref"]
        ),
    }


def _displacement(reference: Mapping[str, torch.Tensor], checkpoint_path: str) -> float:
    """L2 displacement of the LIVE weights from the shared start after one update."""

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["live_state_dict"]
    total = 0.0
    for name, value in reference.items():
        total += float((state[name].detach().cpu() - value).double().pow(2).sum())
    return total ** 0.5


def bounded_smoke() -> dict[str, Any]:
    """One CPU update under the band and one under the control, from the identical sealed start."""

    started = time.monotonic()
    sealed = SEALED_CONFIGS / f"{CELL_ID}.json"
    if not sealed.is_file():
        raise NG3Error("run `seal` before `smoke`")
    cell = json.loads(sealed.read_text(encoding="utf-8"))
    control = json.loads(
        (SEALED_CONFIGS
         / f"matched_control_of_record_seed_{SEED}_{ARM_NAME}.reference.json").read_text(
            encoding="utf-8")
    )
    reference = {
        name: value.detach().clone()
        for name, value in torch.load(
            qbr1.QBR_INITIAL_STATE, map_location="cpu", weights_only=False)["state_dict"].items()
    }
    arms: dict[str, Any] = {}
    for label, config in (("tau_band", cell), ("control", control)):
        arms[label] = qbr1._run_resume_smoke_segment(config, SMOKE_ROOT / label, stop_after=1)
    band_state = arms["tau_band"]["live_state_sha256"]
    control_state = arms["control"]["live_state_sha256"]
    telemetry_rows = {}
    for label in ("tau_band", "control"):
        row = json.loads(
            (SMOKE_ROOT / label / "history.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        telemetry_rows[label] = row["objective"]
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
            "tau_band_live_state_sha256": band_state,
            "control_live_state_sha256": control_state,
            "states_differ": band_state != control_state,
            "archives_differ": arms["tau_band"]["archive"]["sha256"]
            != arms["control"]["archive"]["sha256"],
        },
        "training_path_is_unmoved_by_this_landing": {
            "control_live_state_sha256": control_state,
            "ng1_pre_telemetry_cold_reference_sha256": NG1_COLD_REFERENCE_LIVE_STATE_SHA256,
            "reproduces_ng1_cold_reference": control_state
            == NG1_COLD_REFERENCE_LIVE_STATE_SHA256,
            "argument": (
                "ng1 ran this identical one-update cold segment BEFORE ng2's telemetry row and "
                "before ng3's validator work existed.  An identical trained state now is a "
                "MEASURED proof that every byte landed since is score-neutral on the training "
                "path -- which is what lets the band cell be compared against a control that ran "
                "under an older trainer"
            ),
        },
        "differential_at_a_shared_tau": _differential_at_a_shared_tau(
            cell, control, SMOKE_ROOT / "control" / "training_payloads" / "update_000001",
        ),
        "first_update_displacement_l2": {
            label: _displacement(reference, arms[label]["checkpoint"]["path"])
            for label in ("tau_band", "control")
        },
        "ng1_cold_first_update_displacement_l2": NG1_COLD_FIRST_UPDATE_DISPLACEMENT_L2,
        "telemetry_rows_update_1": telemetry_rows,
        "all_payloads_retained": True,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "elapsed_seconds": time.monotonic() - started,
    }
    qbt.atomic_json(SMOKE_ROOT / "BOUNDED_SMOKE_RESULT.json", result)
    return result


# ---------------------------------------------------------------------------
# 5. schedule leg -- gm1's prediction turned into a measurement on THIS arm's own field
# ---------------------------------------------------------------------------
def schedule_leg(output: Path | None = None) -> dict[str, Any]:
    """Measure the tau-schedule leg of the REPORTED surrogate on a FROZEN field, both bands.

    ddm_sd1 MEASURED that the annealed loss factorizes as (schedule leg x field leg) and that the
    schedule leg alone is -40.54% on a frozen field -- 8.4x the field's own signal -- which is how
    a monotone-FALLING reported loss coexisted with a monotone-WORSENING argmax.  ddm_gm1 DERIVED
    that the band cuts that artefact to -8.57%.  Both are numbers from OTHER instruments on the
    n32 milestone logits; this reads the same quantity through the TRAINER'S OWN
    ``expected_flip_margin_loss`` on the payload the bounded smoke already retained, so the
    prediction becomes a measurement on this arm's own object at $0 and on a different instrument.

    Frozen field = one chunk of 16 pairs at control update 1.  It is a cross-check of the
    SCHEDULE leg only; it says nothing about where step 5,000 lands.
    """

    started = time.monotonic()
    payload_dir = SMOKE_ROOT / "control" / "training_payloads" / "update_000001"
    if not payload_dir.is_dir():
        raise NG3Error(f"run `smoke` before `schedule-leg`: {payload_dir} is absent")
    inputs = _objective_inputs_from_retained(payload_dir)
    sealed = json.loads((SEALED_CONFIGS / f"{CELL_ID}.json").read_text(encoding="utf-8"))

    def surrogate(tau: float) -> float:
        with torch.no_grad():
            return float(qbt.expected_flip_margin_loss(
                inputs["logits"], inputs["target_argmax"], tau, inputs["sample_weights"]))

    bands = {
        "legacy": (float(qbt.LEGACY_EXPECTED_FLIP_TAU_BAND[0]),
                   float(qbt.LEGACY_EXPECTED_FLIP_TAU_BAND[1])),
        "msafe_band": (float(sealed["expected_flip_tau_start"]),
                       float(sealed["expected_flip_tau_end"])),
    }
    legs: dict[str, Any] = {}
    for name, (start, end) in bands.items():
        at_start, at_end = surrogate(start), surrogate(end)
        legs[name] = {
            "tau_start": start, "tau_end": end,
            "surrogate_at_start": at_start, "surrogate_at_end": at_end,
            "schedule_leg_pct": (at_end - at_start) / at_start * 100.0,
        }
    reduction = abs(legs["legacy"]["schedule_leg_pct"]) / abs(legs["msafe_band"]["schedule_leg_pct"])
    receipt = {
        "schema": "ddm_ng3_schedule_leg_receipt.v1",
        "arm": ARM,
        "axis": "[macOS-CPU advisory . frozen n16 chunk at control update 1 . NON-PROMOTABLE]",
        "score_claim": False,
        "promotion_eligible": False,
        "instrument": "qbt.expected_flip_margin_loss (the trainer's OWN loss), not a re-implementation",
        "field": {"payload_dir": str(payload_dir), "pairs": inputs["pair_ids"]},
        "legs": legs,
        "artefact_reduction_x": reduction,
        "gm1_predicted": {
            "legacy_schedule_leg_pct": GM1_MEASURED["schedule_leg_legacy_pct"],
            "band_schedule_leg_pct": GM1_MEASURED["schedule_leg_band_pct"],
            "artefact_reduction_x": (abs(GM1_MEASURED["schedule_leg_legacy_pct"])
                                     / abs(GM1_MEASURED["schedule_leg_band_pct"])),
        },
        "tau_ref_fixed_ruler": {
            "tau": float(qbt.EXPECTED_FLIP_TAU_REFERENCE),
            "surrogate": surrogate(float(qbt.EXPECTED_FLIP_TAU_REFERENCE)),
            "why": "this value is schedule-INDEPENDENT by construction; it is the row ng2 added",
        },
        "scope": (
            "SCHEDULE leg only, on ONE frozen 16-pair chunk with a different instrument and a "
            "different sample from gm1's n32 milestone read.  Agreement is a cross-instrument "
            "confirmation of the MECHANISM; it is not evidence about d_seg at step 5,000."
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    qbt.atomic_json(output or (ARM_ROOT / "SCHEDULE_LEG_RECEIPT.json"), receipt)
    return receipt


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("resolve", "seal", "snapshot", "smoke", "schedule-leg"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {"resolve": resolve, "seal": seal, "snapshot": snapshot_source,
               "smoke": bounded_smoke, "schedule-leg": schedule_leg}
    print(json.dumps(handler[args.action](), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
