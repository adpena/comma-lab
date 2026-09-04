# SPDX-License-Identifier: MIT
"""DDM NG2 — derive, seal and CPU-smoke the AREA-CAP cell (vr1 row 3) on the born vehicle.

The sealed QBR1 control cell UNCHANGED except two things:

1. a ONE-SIDED Chan-Vese area cap ``E_c = (lambda_c/2) * relu(A_c - A_c^GT)^2`` on the two rare
   classes the dual ascent already constrains, with ``lambda_c = F_c / (delta_c * A_GT_c)``
   resolved through the registered law's own callable and ``A_GT_c`` from the trainer's own
   selection bincount;
2. the score-neutral fixed-reference-tau surrogate telemetry row (``ddm_sd1`` row 0), which
   changes no weight and is proven byte-neutral by measurement against ng1's pre-telemetry
   cold reference.

NO LAUNCH.  This module derives, seals and smokes on CPU; MAIN fires on the Metal.
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

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_qbr1_born_fairform_burn_prep as qbr1
from experiments import ddm_qbt1_qbflow_trainer as qbt
from tac.witness_dsl.curriculum_dsl import (
    AreaCapBornRareClass,
    compile_qbr1_area_cap_config,
)

ARM = "ddm_ng2_area_cap"
ARM_ROOT = Path("/Volumes/APDataStore/pact/ddm_ng2_area_cap")
SEALED_CONFIGS = ARM_ROOT / "sealed_configs"
SEALED_SOURCE_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_ng2_area_cap")
NG2_RUN_ROOT = qbt.QBR1_RETENTION_ROOT / "ng2_area_cap"
SMOKE_ROOT = NG2_RUN_ROOT / "bounded_smoke"

SEED = 20260902
ARM_NAME = "control_native100"
CELL_ID = f"seed_{SEED}_area_cap_{ARM_NAME}"
CONTROL_RUN = qbr1.RUN_ROOT / f"seed_{SEED}" / ARM_NAME
CLASSES = tuple(name for name, _index in qbt.AREA_CAP_CLASSES)

DERIVATION_SCHEMA = "ddm_ng2_area_cap_derivation.v1"
SEAL_SCHEMA = "ddm_ng2_area_cap_seal_receipt.v1"
SMOKE_SCHEMA = "ddm_ng2_area_cap_bounded_smoke.v1"
SNAPSHOT_SCHEMA = "ddm_ng2_sealed_source_manifest.v1"

#: the only config keys the cap cell may move relative to a freshly compiled control.
ALLOWED_AREA_CAP_MUTATIONS = frozenset({"cell_id", "output", "area_cap"})

#: the excursion window ddm_sd1 measured (step 0 -> the d_seg peak at 2,000).  F_c is read
#: over this window because it is the window the cap has to counter.
BIRTH_FORCE_WINDOW = 2_000

#: ng1's bounded smoke ran ONE cold update from this same start under the PRE-telemetry code.
#: Reproducing it proves the telemetry row changed no trained byte.
NG1_COLD_REFERENCE_LIVE_STATE_SHA256 = (
    "27f514180db2b4cda57289bbeb4be5ca8daf64e874921c92ba5c08d613c30973"
)
NG1_COLD_FIRST_UPDATE_DISPLACEMENT_L2 = 0.055886740188786026

#: the cold control of record, recomputed from components by ng1 and re-read here (advisory).
COLD_CONTROL_S_HAT = {
    0: 0.39876797285867277,
    1_000: 0.46687521208987615,
    2_000: 0.48567677825279465,
    3_000: 0.47538291701253005,
    4_000: 0.44219037073377010,
    5_000: 0.42514878445269977,
}


class NG2Error(RuntimeError):
    """Fail-closed NG2 contract error."""


# ---------------------------------------------------------------------------
# 1. derive
# ---------------------------------------------------------------------------
def measured_birth_force(history_path: Path, window: int = BIRTH_FORCE_WINDOW) -> dict[str, Any]:
    """F_c = median of the control's own effective class weight ``100 * lambda_c^dual``.

    The law's ``F_birth`` is a birth loss WEIGHT.  On this vehicle the per-class growth
    pressure is ``100 * lambda_c^dual * per_class_expected_flip(c)``, so ``100*lambda_c^dual``
    is its weight -- but it is a DUAL variable, not a constant: it starts at 0, spikes to ~5
    within ten updates, then settles.  The median over the excursion window is the robust
    central value and its p10/p90 spread travels with it as the honest dispersion.
    """

    if not history_path.is_file():
        raise NG2Error(f"control history is absent: {history_path}")
    series: dict[str, list[float]] = {name: [] for name in CLASSES}
    with history_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if int(row["completed_steps"]) > window:
                break
            for name in CLASSES:
                series[name].append(100.0 * float(row["margin_constraint_lambdas"][name]))
    out: dict[str, Any] = {
        "definition": "F_c = 100 * lambda_c^dual (the effective per-class recall weight)",
        "source": str(history_path),
        "window_steps": [1, window],
        "per_class": {},
    }
    for name in CLASSES:
        values = sorted(series[name])
        if len(values) != window:
            raise NG2Error(f"control history is short for {name}: {len(values)} of {window}")

        def quantile(fraction: float, ordered: list[float] = values) -> float:
            position = fraction * (len(ordered) - 1)
            low = int(position)
            high = min(low + 1, len(ordered) - 1)
            return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

        out["per_class"][name] = {
            "median": quantile(0.5),
            "p10": quantile(0.10),
            "p90": quantile(0.90),
            "min": values[0],
            "max": values[-1],
            "n": len(values),
        }
    return out


def measured_start_area(device: torch.device | None = None) -> dict[str, Any]:
    """The cell's own step-0 realized class area, from the exact same-start state.

    ``delta_c`` is this measurement minus one: the equilibrium is placed at the area the class
    ALREADY occupies at update zero, so the constraint says exactly one thing for both classes
    -- do not grow past where you began -- and the cap's step-0 retraction is by construction
    its own equilibrium force rather than an arbitrary extra push.
    """

    device = device or torch.device("cpu")
    posenet, segnet = qbt.load_differentiable_scorers(REPO / "upstream", device=device)
    posenet.eval()
    segnet.eval()
    model = qbt.load_initial_model(device)
    initialization = torch.load(qbr1.QBR_INITIAL_STATE, map_location="cpu", weights_only=False)
    model.load_state_dict(
        {name: value.detach().clone().to(device) for name, value in initialization["state_dict"].items()},
        strict=True,
    )
    totals: dict[str, torch.Tensor] = {}
    weight_total = torch.zeros((), dtype=torch.float64)
    for chunk_ids in qbt.training_chunks(qbt.SELECTION_IDS, 16):
        ids = torch.tensor(chunk_ids, dtype=torch.long, device=device)
        target, _pose = qbt._target_arrays(chunk_ids, device)
        weights = qbt.no2_sample_weights(chunk_ids, device).to(torch.float64)
        with torch.no_grad():
            outputs = model(ids, height=qbt.EVAL_H, width=qbt.EVAL_W)
            camera = qbt.roundtrip_to_camera_uint8_ste(outputs["rgb_pair_01"])
            _pose6, logits = qbt.scorer_forward(camera, posenet, segnet)
        soft = torch.softmax(logits.to(torch.float64), dim=1).mean(dim=(2, 3))
        index = logits.argmax(dim=1)
        hard = torch.stack(
            [(index == c).to(torch.float64).mean(dim=(1, 2)) for c in range(logits.shape[1])], dim=1
        )
        gt = torch.stack(
            [(target == c).to(torch.float64).mean(dim=(1, 2)) for c in range(logits.shape[1])], dim=1
        )
        for key, value in (("soft", soft), ("hard", hard), ("gt", gt)):
            totals[key] = totals.get(key, torch.zeros(value.shape[1], dtype=torch.float64)) + (
                value * weights[:, None]
            ).sum(dim=0)
        weight_total = weight_total + weights.sum()
    soft_mass = (totals["soft"] / weight_total).tolist()
    hard_area = (totals["hard"] / weight_total).tolist()
    gt_area = (totals["gt"] / weight_total).tolist()
    per_class = {}
    for name, index in qbt.AREA_CAP_CLASSES:
        per_class[name] = {
            "class_id": int(index),
            "gt_area_ht": gt_area[index],
            "realized_argmax_area_ht": hard_area[index],
            "softmax_mass_ht_T1": soft_mass[index],
            "argmax_over_gt": hard_area[index] / gt_area[index],
            "softmax_over_gt": soft_mass[index] / gt_area[index],
            "delta": hard_area[index] / gt_area[index] - 1.0,
        }
        if per_class[name]["delta"] <= 0.0:
            raise NG2Error(
                f"{name} does not over-paint at the start state; the one-sided cap has no "
                "operating point and delta_c would be non-positive"
            )
    return {
        "state": "the sealed same-start r10 EMA shadow at update zero",
        "axis": "[macOS-CPU advisory; the vehicle's own PyAV gt_n600 target; not contest authority]",
        "gt_lineage": "PYAV_YUV420_TO_RGB (the loss's own target; DALI is the authority axis)",
        "ht_weighted_over_the_sealed_32_pair_selection": True,
        "class_order": list(qbt.PALETTE_CLASSES),
        "softmax_mass_ht_T1": soft_mass,
        "realized_argmax_area_ht": hard_area,
        "gt_area_ht": gt_area,
        "per_class": per_class,
        "why_the_area_is_straight_through": (
            "softmax(T=1) mass over-states the thin classes badly here (see softmax_over_gt); "
            "the argmax value with a softmax-Jacobian gradient is unbiased in value and keeps "
            "the law's delta(phi) boundary localization"
        ),
    }


def derive(output: Path | None = None) -> dict[str, Any]:
    """Measure F_c and delta_c, derive lambda_c through the registered law, write the receipt."""

    started = time.monotonic()
    birth_force = measured_birth_force(CONTROL_RUN / "history.jsonl")
    start_area = measured_start_area()
    gt_area = qbt.selection_gt_area_fractions(list(qbt.SELECTION_IDS))
    force = {name: birth_force["per_class"][name]["median"] for name in CLASSES}
    tolerance = {name: start_area["per_class"][name]["delta"] for name in CLASSES}
    bincount_gt = {name: gt_area[index] for name, index in qbt.AREA_CAP_CLASSES}
    lambdas = qbt.derive_area_cap_lambdas(list(qbt.SELECTION_IDS), force, tolerance)
    from tac.canonical_equations.chan_vese_area_constraint_birth_balance_20260708 import (
        dominance_at_runaway,
    )

    peak_ratio = {"Lane": 1.09291, "Movable": 1.05801}
    dominance = {
        name: dominance_at_runaway(peak_ratio[name], 1.0, tolerance=tolerance[name])
        for name in CLASSES
    }
    receipt = {
        "schema": DERIVATION_SCHEMA,
        "arm": ARM,
        "law": "chan_vese_area_constraint_birth_balance_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "axis": "[macOS-CPU advisory; not contest authority]",
        "birth_force_measured": birth_force,
        "start_area_measured": start_area,
        "gt_area_from_trainer_bincount": {
            "callable": "ddm_qbt1_qbflow_trainer.selection_gt_area_fractions",
            "same_bincount_as": "derive_balanced_class_weights",
            "all_classes": {qbt.PALETTE_CLASSES[i]: gt_area[i] for i in sorted(gt_area)},
            "capped_classes": bincount_gt,
        },
        "derived": {
            "birth_force": force,
            "tolerance": tolerance,
            "gt_area": bincount_gt,
            "lambdas": lambdas,
            "equilibrium_ratio": {name: 1.0 + tolerance[name] for name in CLASSES},
            "dominance_at_the_control_peak": dominance,
            "control_peak_ratio_source": (
                "ddm_sd1 section 4, control cell, DALI-authority read at step 2,000"
            ),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    qbt.atomic_json((output or ARM_ROOT) / "DERIVATION.json", receipt)
    return receipt


# ---------------------------------------------------------------------------
# 2. seal
# ---------------------------------------------------------------------------
def compile_area_cap_cell(derivation: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compile the cap cell and a matched control THROUGH the burn prep's own compile path.

    Both configs come from ``qbr1.compile_cell`` in THIS tree, so their pins are this tree's
    pins and the only thing that separates them is the lever.  The sealed control on disk is
    NOT deep-copied: its pins name the previous trainer, which by construction cannot carry a
    new loss term, so inheriting them would seal a config no tree can satisfy.
    """

    initial_state = qbt.file_fact(qbr1.QBR_INITIAL_STATE)
    control = qbr1.compile_cell(SEED, ARM_NAME, initial_state)
    lever = AreaCapBornRareClass(
        birth_force=derivation["derived"]["birth_force"],
        tolerance=derivation["derived"]["tolerance"],
        gt_area=derivation["derived"]["gt_area"],
    )
    cell = copy.deepcopy(control)
    cell["cell_id"] = CELL_ID
    cell["output"] = str((NG2_RUN_ROOT / "runs" / CELL_ID).resolve())
    cell["area_cap"] = compile_qbr1_area_cap_config(lever)
    qbr1.validate_config(cell, require_launch_authority=False)
    return cell, control


def validate_area_cap_cell(cell: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the cap cell is the control plus exactly the area-cap block."""

    differing = {key for key in set(cell) | set(control) if cell.get(key) != control.get(key)}
    extra = differing - ALLOWED_AREA_CAP_MUTATIONS
    if extra:
        raise NG2Error(f"area-cap cell moved more than the lever: {sorted(extra)}")
    for key in ("objective", "ema", "schedule", "initial_state", "learning_rate",
                "margin_constraints", "expected_flip_tau_start", "expected_flip_tau_end",
                "pair_ids", "selection_weights", "total_steps", "milestones", "seed",
                "resume_from"):
        if cell[key] != control[key]:
            raise NG2Error(f"area-cap cell moved a held field: {key}")
    if cell.get("launch_authorized") is not False:
        raise NG2Error("seal must leave the cap cell unauthorized")
    for lane in ("scorer_lane", "metal_lane"):
        if cell[lane].get("claimed") is not False or cell[lane].get("claim_id") is not None:
            raise NG2Error(f"seal must leave {lane} unbound for MAIN")
    if cell.get("resume_from") is not None:
        raise NG2Error("the cap cell is a COLD transition: resume_from must stay null")
    return {
        "differing_keys": sorted(differing),
        "allowed": sorted(ALLOWED_AREA_CAP_MUTATIONS),
        "held_fields_identical": True,
        "transition": "COLD (fresh AdamW) -- the same transition the control took, so the pair "
                      "isolates the cap and never composes it with ng1's warm lever",
    }


def control_pin_delta() -> dict[str, Any]:
    """The exact pin difference between the sealed control and this tree, with a git receipt."""

    sealed_path = qbr1.CONFIG_ROOT / f"seed_{SEED}_{ARM_NAME}.json"
    if not sealed_path.is_file():
        raise NG2Error(f"sealed control config is absent: {sealed_path}")
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    live = qbr1.verify_inputs()
    moved = {
        name: {"sealed": sealed["source_pins"].get(name, {}).get("sha256"),
               "ng2": live[name]["sha256"]}
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
        "ng2_revision": qbr1.source_revision(),
        "pins_that_moved": moved,
        "uncommitted_diffstat_at_seal": diffstat,
        "why": (
            "a NEW loss term changes the pinned trainer's bytes, so the cap cell cannot be a "
            "same-pins twin; it is a same-START, same-schedule, same-EMA, same-selection twin "
            "whose pin set moves exactly at the file carrying the lever"
        ),
    }


def refuse_if_the_run_has_already_started(run_output: Path) -> None:
    for marker in ("RESULT.json", "history.jsonl", "milestones"):
        if (run_output / marker).exists():
            raise NG2Error(f"the cap cell has already started: {run_output / marker}")


def falsifiers() -> dict[str, Any]:
    """The three pre-registered falsifiers, fixed before the burn."""

    return {
        "1_primary_the_cap_must_beat_the_cold_control_at_both_ends": {
            "test": "S_hat(5,000) < 0.42514878445269977 AND S_hat(2,000) < 0.48567677825279465",
            "control_rows": COLD_CONTROL_S_HAT,
            "read": "each cell's own milestones/step_*/MILESTONE.json, decomposed",
            "if_it_fails": "the cap does not act on the mechanism; the excursion is not "
                           "rare-class over-paint on this vehicle and vr1 row 3 is refuted at "
                           "formulation scope for the born object",
        },
        "2_the_cap_must_actually_bind": {
            "test": "realized argmax area / GT area at step 2,000 within each class's own "
                    "measured start ratio",
            "read": "MILESTONE.json retained argmax vs the pair's GT; the charter's rounded "
                    "form is 1.03 for both classes",
            "if_it_fails": "lambda_c is too soft; the FORM is untouched and the scale is the "
                           "next single lever (the law's own scale is ASSUMED_AWAITING_VERIFICATION)",
        },
        "3_the_fixed_tau_telemetry_must_be_faithful": {
            "test": "seg_expected_flip_realized_tau_ref peaks at the same milestone as d_seg_hat",
            "read": "history.jsonl objective.seg_expected_flip_realized_tau_ref beside the "
                    "milestone d_seg_hat",
            "if_it_fails": "ddm_sd1's fixed-tau faithfulness does not hold in-loop; the "
                           "telemetry row is wrong, not the lever",
        },
    }


def seal() -> dict[str, Any]:
    started = time.monotonic()
    derivation_path = ARM_ROOT / "DERIVATION.json"
    if not derivation_path.is_file():
        raise NG2Error("run `derive` before `seal`")
    derivation = json.loads(derivation_path.read_text(encoding="utf-8"))
    cell, control = compile_area_cap_cell(derivation)
    refuse_if_the_run_has_already_started(Path(cell["output"]))
    diff = validate_area_cap_cell(cell, control)
    cell_fact = qbt.atomic_json(SEALED_CONFIGS / f"{CELL_ID}.json", cell)
    # The matched control is a REFERENCE recompile: it exists so the single-lever diff and the
    # smoke's control arm are reproducible.  Its `output` still points at the ALREADY COMPLETED
    # seed-20260902 control run, so it must never be fired.  `run_config` refuses it because
    # `launch_authorized` is False, and the marker below says so to a human reader too.
    control_reference = copy.deepcopy(control)
    control_reference["do_not_fire_reference_recompile_only"] = True
    control_fact = qbt.atomic_json(
        SEALED_CONFIGS / f"matched_control_of_record_seed_{SEED}_{ARM_NAME}.reference.json",
        control_reference,
    )
    if json.loads(Path(cell_fact["path"]).read_text(encoding="utf-8")) != cell:
        raise NG2Error("sealed cap-cell JSON round trip differs")
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
        "area_cap": cell["area_cap"],
        "derivation": qbt.file_fact(derivation_path),
        "cold_control_of_record": COLD_CONTROL_S_HAT,
        "falsifiers": falsifiers(),
        "authorized_configs_written": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    qbt.atomic_json(ARM_ROOT / "SEAL_RECEIPT.json", receipt)
    return receipt


# ---------------------------------------------------------------------------
# 3. sealed source snapshot
# ---------------------------------------------------------------------------
#: repo paths a sealed tree shares by symlink rather than copying.  Every one of them is
#: sha-pinned by ``qbt.verify_pins`` / ``qbr1.verify_inputs``, so sharing is exactly what those
#: gates check; copying 5 GB of GT cache per seal would buy nothing and cost the SSD tier.
SHARED_INPUT_PATHS = (
    "upstream",
    "experiments/results/mlx_fleet_gt_cache",
)


def link_shared_inputs(destination: Path) -> dict[str, str]:
    """Point the sealed tree's large pinned inputs at the repo, the way wc3's tree does.

    ``git archive`` carries only tracked files, so the frozen scorer weights and the 5 GB GT
    cache are absent from a fresh extract and the tree could not verify its own pins.  wc3's
    ``sealed_source_106d0dd0_v2`` -- the tree the live chain runs -- solves this the same way:
    ``upstream`` and ``experiments/results/mlx_fleet_gt_cache`` are symlinks to the repo.
    """

    linked: dict[str, str] = {}
    for relative in (*SHARED_INPUT_PATHS, ".venv"):
        target = REPO / relative
        if not target.exists():
            raise NG2Error(f"shared input is absent from the repo: {target}")
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
        raise NG2Error(
            f"the sealed tree cannot verify its own pins:\n{completed.stderr.strip()[-2000:]}"
        )
    return {"status": "PASS", **json.loads(completed.stdout.strip().splitlines()[-1])}


def snapshot_source(destination: Path | None = None) -> dict[str, Any]:
    """Materialize the sealed source tree the cap cell fires from, at the current commit.

    ``git archive`` of HEAD, so the tree is exactly the committed revision -- no untracked
    scratch, no uncommitted edit, reproducible from the revision alone.  A dirty working tree
    is refused for the two files the lever lives in, because a snapshot of HEAD that does not
    match what was smoked would seal a different object than the one that was tested.
    """

    revision = qbr1.source_revision()
    destination = destination or (SEALED_SOURCE_ROOT / f"sealed_source_{revision[:10]}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--",
         "experiments/ddm_qbt1_qbflow_trainer.py",
         "experiments/ddm_qbr1_born_fairform_burn_prep.py",
         "experiments/ddm_ng2_area_cap_cell.py",
         "src/tac/witness_dsl/curriculum_dsl.py"],
        cwd=REPO, text=True, capture_output=True, check=True,
    ).stdout.strip()
    if dirty:
        raise NG2Error(f"refusing to snapshot a dirty lever surface:\n{dirty}")
    if destination.exists():
        raise NG2Error(f"sealed source already exists (never overwrite a seal): {destination}")
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
        "method": "git archive of the committed revision, extracted; the pinned large inputs "
                  "and the venv are symlinked to the repo (they are sha-pinned, so sharing them "
                  "is what verify_pins checks, not what it could be fooled by)",
        "shared_inputs": shared,
        "pins_verify_inside_the_sealed_tree": verify_pins_inside(destination),
        "files": {
            name: qbt.file_fact(destination / name)
            for name in (
                "experiments/ddm_qbt1_qbflow_trainer.py",
                "experiments/ddm_qbr1_born_fairform_burn_prep.py",
                "experiments/ddm_ng2_area_cap_cell.py",
                "experiments/ddm_qbflow_packet.py",
                "src/tac/witness_dsl/curriculum_dsl.py",
                "src/tac/canonical_equations/chan_vese_area_constraint_birth_balance_20260708.py",
            )
        },
    }
    qbt.atomic_json(destination / "NG2_SEALED_SOURCE_MANIFEST.json", manifest)
    qbt.atomic_json(ARM_ROOT / "SEALED_SOURCE_MANIFEST.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# 4. bounded CPU smoke
# ---------------------------------------------------------------------------
def _cap_is_zero_below_gt() -> dict[str, Any]:
    """DIFFERENTIAL: the cap term is EXACTLY zero when every capped class is at or under GT.

    Built as a two-arm differential on one synthetic field rather than a single assertion:
    the same helper, the same lambdas, one field under GT area and one over it.  A term that
    returned zero because it was mis-wired would return zero on BOTH arms.
    """

    torch.manual_seed(0)
    height = width = 16
    target = torch.zeros((2, height, width), dtype=torch.long)
    target[:, :4, :] = 1  # Lane occupies 25% of GT
    target[:, 4:8, :] = 3  # Movable occupies 25% of GT
    lambdas = {"Lane": 2_000.0, "Movable": 7_000.0}

    def field(lane_rows: int, movable_rows: int) -> torch.Tensor:
        logits = torch.full((2, 5, height, width), -8.0)
        logits[:, 0] = 8.0
        logits[:, 1, :lane_rows, :] = 16.0
        logits[:, 3, 8:8 + movable_rows, :] = 16.0
        return logits.requires_grad_(True)

    under = field(2, 2)  # both classes at half their GT area
    over = field(8, 8)   # both classes at twice their GT area
    zero_penalty, zero_components = qbt.one_sided_area_cap_penalty(under, target, lambdas)
    live_penalty, live_components = qbt.one_sided_area_cap_penalty(over, target, lambdas)
    zero_penalty.backward()
    under_grad = float(under.grad.abs().max())
    return {
        "under_gt_penalty": float(zero_penalty),
        "under_gt_penalty_is_exactly_zero": float(zero_penalty) == 0.0,
        "under_gt_max_abs_grad": under_grad,
        "under_gt_grad_is_exactly_zero": under_grad == 0.0,
        "under_gt_per_class_over": {
            name: float(zero_components[f"area_cap_over_{name}"]) for name in CLASSES
        },
        "over_gt_penalty": float(live_penalty),
        "over_gt_penalty_is_positive": float(live_penalty) > 0.0,
        "over_gt_per_class_over": {
            name: float(live_components[f"area_cap_over_{name}"]) for name in CLASSES
        },
        "differential_passes": float(zero_penalty) == 0.0 and float(live_penalty) > 0.0,
    }


def _gradient_scales(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Measure the cap's gradient against the terms it must argue with, at the start state.

    The balance law ``A* = A_GT + F/lambda`` is stated in AREA units; the realized effect on
    the weights travels through LOGIT-space gradients, whose normalizations differ (the recall
    term divides by the class's own pixel count, the area term by the frame).  So the balance
    is an order-of-magnitude guide, and this is the measurement that says whether the cap is
    audible at all.  Reported, never silently used to re-tune lambda.
    """

    device = torch.device("cpu")
    posenet, segnet = qbt.load_differentiable_scorers(REPO / "upstream", device=device)
    posenet.eval()
    segnet.eval()
    model = qbt.load_initial_model(device)
    initialization = torch.load(qbr1.QBR_INITIAL_STATE, map_location="cpu", weights_only=False)
    model.load_state_dict(initialization["state_dict"], strict=True)
    chunk_ids = qbt.training_chunks(qbt.SELECTION_IDS, 16)[
        qbr1.schedule_for_seed(int(cell["seed"]))[0]
    ]
    ids = torch.tensor(chunk_ids, dtype=torch.long, device=device)
    target, target_pose6 = qbt._target_arrays(chunk_ids, device)
    weights = qbt.no2_sample_weights(chunk_ids, device)
    outputs = model(ids, height=qbt.EVAL_H, width=qbt.EVAL_W)
    camera = qbt.roundtrip_to_camera_uint8_ste(outputs["rgb_pair_01"])
    _pose6, logits = qbt.scorer_forward(camera, posenet, segnet)
    logits = logits.detach().requires_grad_(True)
    tau = qbt.tau_for_step(0, int(cell["total_steps"]),
                           float(cell["expected_flip_tau_start"]),
                           float(cell["expected_flip_tau_end"]))
    lambdas = {name: float(value) for name, value in cell["area_cap"]["lambdas"].items()}
    dual = {name: stat["median"] / 100.0
            for name, stat in json.loads((ARM_ROOT / "DERIVATION.json").read_text(
                encoding="utf-8"))["birth_force_measured"]["per_class"].items()}

    def grad_norm(term: torch.Tensor) -> float:
        (grad,) = torch.autograd.grad(term, logits, retain_graph=True)
        return float(grad.norm())

    realized = 100.0 * qbt.expected_flip_margin_loss(logits, target, tau, weights)
    recall = sum(
        100.0 * dual[name] * qbt.per_class_expected_flip_margin_loss(
            logits, target, tau, class_id, weights)
        for name, class_id in qbt.AREA_CAP_CLASSES
    )
    cap, _components = qbt.one_sided_area_cap_penalty(logits, target, lambdas, weights)
    scales = {
        "realized_seg_x100": grad_norm(realized),
        "recall_dual_penalty": grad_norm(recall),
        "area_cap": grad_norm(cap),
        "area_cap_loss_value": float(cap),
        "realized_seg_loss_value": float(realized),
        "recall_loss_value": float(recall),
        "tau": tau,
        "chunk_pair_ids": list(chunk_ids),
    }
    scales["cap_over_recall"] = scales["area_cap"] / scales["recall_dual_penalty"]
    scales["cap_over_realized"] = scales["area_cap"] / scales["realized_seg_x100"]
    scales["pose_target_shape"] = list(target_pose6.shape)
    return scales


def bounded_smoke() -> dict[str, Any]:
    """One CPU update with the cap and one without, from the identical sealed start."""

    started = time.monotonic()
    sealed = SEALED_CONFIGS / f"{CELL_ID}.json"
    if not sealed.is_file():
        raise NG2Error("run `seal` before `smoke`")
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
    for label, config in (("area_cap", cell), ("control", control)):
        arms[label] = qbr1._run_resume_smoke_segment(
            config, SMOKE_ROOT / label, stop_after=1
        )
    cap_state = arms["area_cap"]["live_state_sha256"]
    control_state = arms["control"]["live_state_sha256"]
    telemetry_rows = {}
    for label in ("area_cap", "control"):
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
            "area_cap_live_state_sha256": cap_state,
            "control_live_state_sha256": control_state,
            "states_differ": cap_state != control_state,
            "archives_differ": arms["area_cap"]["archive"]["sha256"]
            != arms["control"]["archive"]["sha256"],
        },
        "telemetry_is_score_neutral": {
            "control_live_state_sha256": control_state,
            "ng1_pre_telemetry_cold_reference_sha256": NG1_COLD_REFERENCE_LIVE_STATE_SHA256,
            "reproduces_ng1_cold_reference": control_state
            == NG1_COLD_REFERENCE_LIVE_STATE_SHA256,
            "argument": (
                "ng1 ran this identical one-update cold segment BEFORE the telemetry row "
                "existed; an identical trained state after the row was added is a measured "
                "proof that the row changed no trained byte"
            ),
        },
        "first_update_displacement_l2": {
            label: _displacement(reference, arms[label]["checkpoint"]["path"])
            for label in ("area_cap", "control")
        },
        "ng1_cold_first_update_displacement_l2": NG1_COLD_FIRST_UPDATE_DISPLACEMENT_L2,
        "differential_cap_is_zero_below_gt": _cap_is_zero_below_gt(),
        "gradient_scales_at_the_start_state": _gradient_scales(cell),
        "telemetry_rows_update_1": telemetry_rows,
        "all_payloads_retained": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    qbt.atomic_json(SMOKE_ROOT / "BOUNDED_SMOKE_RESULT.json", result)
    return result


def _displacement(reference: Mapping[str, torch.Tensor], checkpoint_path: str) -> float:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    total = 0.0
    for name, value in reference.items():
        total += float((state[name].detach().cpu() - value).double().pow(2).sum())
    return total ** 0.5


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("derive", "seal", "snapshot", "smoke"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {"derive": derive, "seal": seal, "snapshot": snapshot_source, "smoke": bounded_smoke}
    print(json.dumps(handler[args.action](), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


