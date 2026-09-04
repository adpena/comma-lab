#!/usr/bin/env python3
"""ddm_ng1 - seal the warm-transition cell for the next QBR-generation burn.

The QBR1 seed-20260902 cells opened a +22% S_hat excursion by step 2,000 and ended
+6.6%/+7.2% ABOVE their own warm start.  Verified at source, the transition into those
cells is cold in exactly ONE respect:

* ``ddm_qbr1_born_fairform_burn_prep.run_config`` loads only ``initial_state["state_dict"]``
  (weights) and then builds a FRESH ``torch.optim.AdamW`` whose moments are zero and whose
  bias-correction step counter is zero.
* The learning rate is NOT cold.  r10's authorized config and r10's own retained optimizer
  ``param_groups[0]["lr"]`` are both 2.0e-4, and neither the trainer nor the burn prep
  contains any LR scheduler, so 2e-4 IS this object's terminal LR.  The only annealed
  quantity is the expected-flip ``tau`` (0.15 -> 0.05), and ``validate_config`` refuses any
  other tau geometry, so the anneal shape is structurally frozen.

This module therefore races exactly one lever: carry r10's AdamW state across the
transition.  It does that WITHOUT forking the sealed training loop, by expressing the warm
start as a ``resume_from`` checkpoint at ``completed_steps == 0`` that the sealed
``_load_checkpoint`` consumes.  Everything else - seed, 32-pair selection, 16-pair chunks,
5,000 updates, EMA law, objective weights, margin constraints, retention - is the cold
control's, unchanged.

No launch.  ``seal`` writes immutable configs with ``launch_authorized=false`` and null
lane claims; MAIN authorizes and fires.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

_IMPORT_REPO = Path(__file__).resolve().parents[1]
if str(_IMPORT_REPO) not in sys.path:
    sys.path.insert(0, str(_IMPORT_REPO))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import ddm_qbr1_born_fairform_burn_prep as qbr1
import ddm_qbt1_qbflow_trainer as qbt

REPO = _IMPORT_REPO

# Sealed identity/receipt custody for this arm (small JSON only).
ARM_ROOT = Path("/Volumes/APDataStore/pact/ddm_ng1_warm_transition")
SEALED_CONFIGS = ARM_ROOT / "sealed_configs"
SEED_ROOT = ARM_ROOT / "warm_seeds"

# Run payloads must live under a custody root that qbt.storage_preflight authorizes.
# QBR1_RETENTION_ROOT is the ORIGINAL burn-prep root; it is dormant (the live chain uses
# WC3_QBR1_RETENTION_ROOT), so this never writes near the live burn.
NG1_RUN_ROOT = qbt.QBR1_RETENTION_ROOT / "ng1_warm_transition"
RUN_OUTPUT = NG1_RUN_ROOT / "runs" / "seed_20260902_warm_transition"
SMOKE_ROOT = NG1_RUN_ROOT / "resume_smoke"

SCHEMA = qbr1.SCHEMA
SEAL_SCHEMA = "ddm_ng1_warm_transition_seal.v1"
SMOKE_SCHEMA = "ddm_ng1_warm_transition_bounded_smoke.v1"

SEED = 20260902
ARM_NAME = "control_native100"
WARM_CELL_ID = f"seed_{SEED}_warm_transition_{ARM_NAME}"

COLD_CONTROL_CONFIG = Path(
    "/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/sealed_configs"
    f"/seed_{SEED}_{ARM_NAME}.json"
)
COLD_CONTROL_RUN = Path(
    "/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/runs"
    f"/seed_{SEED}/{ARM_NAME}"
)

# r10 ran 10,010 actual optimizer updates; its moments are bias-correction saturated.
R10_EXPECTED_STEP = 10_010.0
MIN_SATURATED_STEP = 1_000.0


class NG1Error(RuntimeError):
    """Fail-closed refusal for the warm-transition seal."""


def _reference_model() -> torch.nn.Module:
    return qbt.load_initial_model(torch.device("cpu"))


def r10_warm_state() -> dict[str, Any]:
    """Return r10's retained EMA shadow and AdamW state, validated against the QBF1 twin."""

    checkpoint = torch.load(qbr1.R10_CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "stage_03_joint_boundary_interior_birth_end":
        raise NG1Error("r10 source is not the stage-03 end checkpoint")
    shadow = checkpoint.get("ema", {}).get("shadow", {})
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if not isinstance(optimizer_state, Mapping):
        raise NG1Error("r10 checkpoint carries no optimizer state; warm transition impossible")
    reference = _reference_model()
    if set(shadow) != set(reference.state_dict()):
        raise NG1Error("r10 EMA tensor set differs from the exact QBF1 twin")
    groups = optimizer_state["param_groups"]
    if len(groups) != 1:
        raise NG1Error("r10 optimizer is not the single-group AdamW the cells construct")
    n_parameters = len(list(reference.parameters()))
    if len(groups[0]["params"]) != n_parameters:
        raise NG1Error(
            "r10 optimizer parameter count differs from the QBF1 twin: "
            f"{len(groups[0]['params'])} vs {n_parameters}"
        )
    steps = {float(entry["step"]) for entry in optimizer_state["state"].values() if "step" in entry}
    if not steps:
        raise NG1Error("r10 optimizer state carries no step counters")
    if min(steps) < MIN_SATURATED_STEP:
        raise NG1Error(f"r10 optimizer moments are not bias-correction saturated: min step {min(steps)}")
    if steps != {R10_EXPECTED_STEP}:
        raise NG1Error(
            f"r10 optimizer step counters differ from its 10,010 re-derived updates: {sorted(steps)}"
        )
    live = checkpoint["live_state_dict"]
    numerator = sum(float((live[k].float() - shadow[k].float()).pow(2).sum()) for k in sorted(live))
    denominator = sum(float(live[k].float().pow(2).sum()) for k in sorted(live))
    return {
        "shadow": {name: value.detach().cpu().clone() for name, value in shadow.items()},
        "optimizer_state_dict": copy.deepcopy(dict(optimizer_state)),
        "source_checkpoint": qbt.file_fact(qbr1.R10_CHECKPOINT),
        "source_step": int(checkpoint["step"]),
        "optimizer_hyperparameters": {
            key: value for key, value in groups[0].items() if key != "params"
        },
        "optimizer_state_entries": len(optimizer_state["state"]),
        "optimizer_step_counters": sorted(steps),
        "parameters": n_parameters,
        "shadow_vs_live_relative_distance": math.sqrt(numerator) / math.sqrt(denominator),
    }


def assert_adamw_hyperparameters_match(config: Mapping[str, Any], warm: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse unless loading r10's state cannot silently change the cell's AdamW law."""

    fresh = torch.optim.AdamW(_reference_model().parameters(), lr=float(config["learning_rate"]))
    fresh_group = {key: value for key, value in fresh.state_dict()["param_groups"][0].items() if key != "params"}
    r10_group = dict(warm["optimizer_hyperparameters"])
    if float(r10_group["lr"]) != float(config["learning_rate"]):
        raise NG1Error(
            "r10 terminal LR differs from the cell LR; the transition would change two levers: "
            f"{r10_group['lr']} vs {config['learning_rate']}"
        )
    differing = sorted(
        key for key in set(fresh_group) | set(r10_group) if fresh_group.get(key) != r10_group.get(key)
    )
    if differing:
        raise NG1Error(f"r10 AdamW hyperparameters differ from the cell's fresh AdamW: {differing}")
    return {
        "fresh_param_group": fresh_group,
        "r10_param_group": r10_group,
        "identical": True,
        "lr_is_object_tail": float(r10_group["lr"]),
    }


ALLOWED_WARM_MUTATIONS = frozenset({"cell_id", "output", "resume_from", "warm_transition"})


def verify_inherited_pins(config: Mapping[str, Any]) -> dict[str, Any]:
    """Re-verify every inherited source pin against its own recorded path and sha.

    This is stronger than calling ``qbr1.verify_inputs()`` from the working tree: the pins
    name the SEALED tree that will actually run the cell, and one working-tree memo
    (``SPEC_ddm_qbflow_packet_schema_v1_20260827.md``) has drifted since the QBR1 seal, so a
    fresh working-tree compile would either refuse or, worse, re-pin the cell away from its
    own control.  Inheriting the control's pins keeps the race single-lever.
    """

    rows: dict[str, Any] = {}
    for name, pin in sorted(config["source_pins"].items()):
        path = Path(pin["path"])
        if not path.is_file():
            raise NG1Error(f"inherited source pin is absent: {name} -> {path}")
        fact = qbt.file_fact(path)
        if fact != pin:
            raise NG1Error(f"inherited source pin drifted on disk: {name} -> {path}")
        rows[name] = fact
    return rows


def compile_warm_cell() -> dict[str, Any]:
    """Derive the warm cell from the cold control's SEALED config, mutating only the transition.

    The warm cell inherits the control's ``source_pins`` and ``source_revision`` verbatim, so
    the two cells are a same-pins twin pair.  Because every check inside
    ``qbr1.validate_config`` reads only fields outside ``ALLOWED_WARM_MUTATIONS``, and the
    control config already passed that validator at QBR1 seal time, proving the diff is a
    subset of the allowed set proves the warm cell satisfies the sealed validator too.
    """

    if not COLD_CONTROL_CONFIG.is_file():
        raise NG1Error(f"cold control config is absent: {COLD_CONTROL_CONFIG}")
    control = json.loads(COLD_CONTROL_CONFIG.read_text(encoding="utf-8"))
    if control.get("schema") != SCHEMA or control.get("arm_name") != ARM_NAME:
        raise NG1Error("cold control config is not the QBR1 fair-form control")
    if control.get("launch_authorized") is not False or control.get("resume_from") is not None:
        raise NG1Error("cold control config is not the immutable unlaunched cell")
    config = copy.deepcopy(control)
    config["cell_id"] = WARM_CELL_ID
    config["output"] = str(RUN_OUTPUT.resolve())
    config["resume_from"] = str((SEED_ROOT / "warm_seed_mps.pt").resolve())
    config["warm_transition"] = {
        "lever": "adamw_optimizer_state_carried_across_the_stage_transition",
        "carried": ["exp_avg", "exp_avg_sq", "step"],
        "not_carried": [
            "margin_constraint_lambdas (held at 0.0, identical to the cold control)",
            "ema num_updates (held at 0, identical to the cold control)",
            "learning_rate (already the object's tail: r10 terminal lr == 2e-4)",
            "expected_flip tau geometry (structurally frozen at 0.15 -> 0.05)",
        ],
        "mechanism": (
            "the warm state is delivered as a completed_steps=0 resume_from checkpoint so the "
            "sealed run_config/_load_checkpoint path consumes it with no code change"
        ),
        "control_of_record": str(COLD_CONTROL_CONFIG),
    }
    validate_warm_cell(config, control)
    return config


def validate_warm_cell(config: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the warm cell is the control plus exactly the transition."""

    differing = {
        key for key in set(config) | set(control) if config.get(key) != control.get(key)
    }
    extra = differing - ALLOWED_WARM_MUTATIONS
    if extra:
        raise NG1Error(f"warm cell moved more than the transition: {sorted(extra)}")
    if config["objective"] != control["objective"]:
        raise NG1Error("warm cell objective differs from its control")
    if config.get("launch_authorized") is not False:
        raise NG1Error("seal must leave the warm cell unauthorized")
    for lane in ("scorer_lane", "metal_lane"):
        if config[lane].get("claimed") is not False or config[lane].get("claim_id") is not None:
            raise NG1Error(f"seal must leave {lane} unbound for MAIN")
    if not str(config["resume_from"]):
        raise NG1Error("warm cell must name its warm seed")
    return {
        "differing_keys": sorted(differing),
        "allowed": sorted(ALLOWED_WARM_MUTATIONS),
        "sealed_validator_argument": (
            "qbr1.validate_config reads only fields outside ALLOWED_WARM_MUTATIONS, and the "
            "control passed it at QBR1 seal time, so the warm cell satisfies it by inheritance"
        ),
        "source_pins": verify_inherited_pins(config),
        "source_revision": config["source_revision"],
    }


def warm_seed_payload(
    config: Mapping[str, Any],
    warm: Mapping[str, Any],
    *,
    device: str,
    history_fact: Mapping[str, Any],
) -> dict[str, Any]:
    identity_config = copy.deepcopy(dict(config))
    identity_config["device"] = device
    identity = qbr1.config_identity(identity_config)
    # Re-seed before snapshotting so the seed's rng payload is the cell's own post-seed stream.
    # The training loop consumes no global RNG (schedule_for_seed owns a local PCG64 and the
    # model has no dropout), so this is provenance rather than a lever.
    qbt.seed_everything(int(config["seed"]))
    return {
        "schema": qbr1.CHECKPOINT_SCHEMA,
        "stage": "stage_01_fairform_finish",
        "completed_steps": 0,
        "config_identity": identity,
        "config_identity_sha256": qbr1.canonical_sha256(identity),
        "live_state_dict": {name: value.clone() for name, value in warm["shadow"].items()},
        "ema": {
            "decay": float(config["ema"]["value"]),
            "warmup": bool(config["ema"]["execution"]["warmup"]),
            "num_updates": 0,
            "shadow": {name: value.clone() for name, value in warm["shadow"].items()},
        },
        "optimizer_state_dict": copy.deepcopy(dict(warm["optimizer_state_dict"])),
        "rng": qbt._rng_payload(),
        "margin_constraint_lambdas": dict.fromkeys(config["margin_constraints"]["bounds"], 0.0),
        "history_prefix": dict(history_fact),
        "schedule_completion_semantics": "range(completed_steps,total_steps); never add total_steps",
        "ng1_warm_transition": {
            "source_checkpoint": warm["source_checkpoint"],
            "source_step": warm["source_step"],
            "carried": "optimizer_state_dict only; weights are the identical r10 EMA shadow",
        },
    }


def build_warm_seed(
    config: Mapping[str, Any],
    warm: Mapping[str, Any],
    *,
    device: str,
    run_output: Path,
    seed_path: Path,
) -> dict[str, Any]:
    """Write a completed_steps=0 warm checkpoint and PROVE the sealed loader consumes it."""

    history_fact = qbt.atomic_bytes(run_output / "history.jsonl", b"")
    payload = warm_seed_payload(config, warm, device=device, history_fact=history_fact)
    fact = qbt.atomic_torch(seed_path, payload)
    verification = verify_warm_seed(config, seed_path, device=device)
    return {"seed": fact, "history": history_fact, "device": device, "verification": verification}


def verify_warm_seed(config: Mapping[str, Any], seed_path: Path, *, device: str) -> dict[str, Any]:
    """Load the seed exactly as the sealed run_config would, and assert it is genuinely warm."""

    identity_config = copy.deepcopy(dict(config))
    identity_config["device"] = device
    model = _reference_model()
    initialization = torch.load(
        Path(config["initial_state"]["path"]), map_location="cpu", weights_only=False
    )
    model.load_state_dict(
        {name: value.detach().clone() for name, value in initialization["state_dict"].items()},
        strict=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    if optimizer.state_dict()["state"]:
        raise NG1Error("a freshly constructed AdamW already carries state")
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    completed, ema, lambdas = qbr1._load_checkpoint(
        seed_path, config=identity_config, model=model, optimizer=optimizer
    )
    # run_config loads initial_state weights and _load_checkpoint then OVERWRITES them with the
    # seed's live_state_dict.  If those differed, the warm cell would silently start from other
    # weights than the cold control - a second lever hidden inside the first.
    after = model.state_dict()
    moved = sorted(name for name in before if not torch.equal(before[name], after[name]))
    if moved:
        raise NG1Error(
            "warm seed changed the start weights; the transition must carry optimizer state only: "
            f"{moved[:5]}"
        )
    loaded = optimizer.state_dict()
    steps = {float(entry["step"]) for entry in loaded["state"].values() if "step" in entry}
    if completed != 0:
        raise NG1Error(f"warm seed must resume at completed_steps 0, got {completed}")
    if not loaded["state"]:
        raise NG1Error("warm seed did not populate the optimizer state; the transition is still cold")
    if min(steps) < MIN_SATURATED_STEP:
        raise NG1Error("loaded optimizer moments are not bias-correction saturated")
    if any(value != 0.0 for value in lambdas.values()):
        raise NG1Error("warm seed must not carry margin-constraint multipliers (one lever)")
    if int(ema._num_updates) != 0:
        raise NG1Error("warm seed must not advance the EMA update counter (one lever)")
    provenance = qbt.verify_ema_executable_law(
        ema, config["ema"], total_updates=int(config["total_steps"])
    )
    return {
        "loaded_completed_steps": completed,
        "optimizer_state_entries": len(loaded["state"]),
        "optimizer_step_counters_min": min(steps),
        "optimizer_step_counters_max": max(steps),
        "margin_constraint_lambdas": dict(lambdas),
        "ema_num_updates": int(ema._num_updates),
        "ema_law_matched": bool(provenance["matched"]),
        "loader": "ddm_qbr1_born_fairform_burn_prep._load_checkpoint (sealed path, unmodified)",
    }


def cold_control_receipt() -> dict[str, Any]:
    """Bind the already-measured cold control as the control of record."""

    if not COLD_CONTROL_CONFIG.is_file():
        raise NG1Error(f"cold control config is absent: {COLD_CONTROL_CONFIG}")
    milestones = []
    for step in qbr1.MILESTONES:
        path = COLD_CONTROL_RUN / "milestones" / f"step_{step:06d}" / "MILESTONE.json"
        if not path.is_file():
            raise NG1Error(f"cold control lacks measured milestone {step}: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        milestones.append(
            {
                "step": int(row["step"]),
                "S_hat": float(row["S_hat"]),
                "d_seg_hat": float(row["d_seg_hat"]),
                "d_pose_hat": float(row["d_pose_hat"]),
                "archive_bytes_exact": int(row["archive_bytes_exact"]),
            }
        )
    surrogate = surrogate_at_milestones(COLD_CONTROL_RUN / "history.jsonl")
    for row in milestones:
        row["seg_expected_flip_realized"] = surrogate.get(row["step"])
    start, end = milestones[0], milestones[-1]
    peak = max(milestones, key=lambda row: row["S_hat"])
    return {
        "config": qbt.file_fact(COLD_CONTROL_CONFIG),
        "run_root": str(COLD_CONTROL_RUN),
        "axis": "[macOS-MPS n32 stratified advisory; not contest authority]",
        "milestones": milestones,
        "warm_start_S_hat": start["S_hat"],
        "endpoint_S_hat": end["S_hat"],
        "peak_S_hat": peak["S_hat"],
        "peak_step": peak["step"],
        "endpoint_excess_over_warm_start": end["S_hat"] - start["S_hat"],
        "endpoint_excess_fraction": end["S_hat"] / start["S_hat"] - 1.0,
        "endpoint_excess_decomposition": {
            "d_seg": 100.0 * (end["d_seg_hat"] - start["d_seg_hat"]),
            "d_pose": math.sqrt(10.0 * end["d_pose_hat"]) - math.sqrt(10.0 * start["d_pose_hat"]),
            "rate": 25.0 * (end["archive_bytes_exact"] - start["archive_bytes_exact"]) / 37_545_489,
        },
    }


def surrogate_at_milestones(history_path: Path) -> dict[int, float]:
    """Read ``seg_expected_flip_realized`` at each milestone.

    The training surrogate is NOT in ``MILESTONE.json`` (whose keys are S_hat / d_seg_hat /
    d_pose_hat / rate_exact / pair_rows / reencode).  It lives in the append-only
    ``history.jsonl`` under ``objective``.  Falsifier 2 has to name the right file or it is not
    executable.
    """

    if not history_path.is_file():
        raise NG1Error(f"history sidecar is absent: {history_path}")
    wanted = set(qbr1.MILESTONES)
    found: dict[int, float] = {}
    with history_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            step = int(row["completed_steps"])
            if step in wanted:
                found[step] = float(row["objective"]["seg_expected_flip_realized"])
    return found


def falsifiers(control: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "primary": {
            "id": "warm_cell_ends_below_its_own_warm_start",
            "statement": (
                "the warm cell's step-5,000 S_hat must be strictly below the shared warm start "
                f"{control['warm_start_S_hat']!r} AND below the cold control at every milestone "
                "1,000..5,000"
            ),
            "warm_start_S_hat": control["warm_start_S_hat"],
            "cold_control_by_step": {
                str(row["step"]): row["S_hat"] for row in control["milestones"]
            },
            "if_it_fails": (
                "the cold optimizer transition is NOT the cause; the schedule/objective is. The next "
                "race is then the LR magnitude alone, holding the transition warm."
            ),
        },
        "secondary_free_read": {
            "id": "surrogate_versus_exact_decoupling_under_a_warm_transition",
            "statement": (
                "report seg_expected_flip_realized beside d_seg_hat at every milestone; if the "
                "surrogate keeps falling while d_seg_hat rises in the WARM cell too, the loss "
                "itself is miscalibrated and vr1 rows 1/4 become the next race"
            ),
            "source": "<run>/history.jsonl objective.seg_expected_flip_realized (NOT MILESTONE.json)",
            "cold_control_surrogate_by_step": {
                str(row["step"]): row["seg_expected_flip_realized"]
                for row in control["milestones"]
            },
            "cold_control_d_seg_by_step": {
                str(row["step"]): row["d_seg_hat"] for row in control["milestones"]
            },
        },
        "no_op_detector": {
            "id": "warm_first_update_must_differ_from_cold_first_update",
            "statement": (
                "the bounded CPU smoke runs one warm and one cold update from the identical start; "
                "identical post-update weights would prove the seed is inert"
            ),
        },
    }


def refuse_if_the_run_has_already_started(run_output: Path) -> None:
    """A re-seal must never truncate a live run's append-only history.

    ``build_warm_seed`` rewrites ``<output>/history.jsonl`` to zero bytes so the warm seed's
    ``history_prefix`` matches it.  When ``resume_from`` is set, ``run_config`` does NOT
    re-initialize that file, so a careless re-seal against a started run would destroy the
    history the crash-resume path verifies its prefix against.
    """

    history = run_output / "history.jsonl"
    if history.is_file() and history.stat().st_size > 0:
        raise NG1Error(
            f"refusing to re-seal over a started run: {history} is non-empty "
            "(move or archive the run directory first)"
        )
    result = run_output / "RESULT.json"
    if result.is_file():
        raise NG1Error(f"refusing to re-seal over a completed run: {result} exists")


def seal() -> dict[str, Any]:
    refuse_if_the_run_has_already_started(RUN_OUTPUT)
    SEALED_CONFIGS.mkdir(parents=True, exist_ok=True)
    SEED_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_OUTPUT.mkdir(parents=True, exist_ok=True)
    warm = r10_warm_state()
    control_config = json.loads(COLD_CONTROL_CONFIG.read_text(encoding="utf-8"))
    config = compile_warm_cell()
    inheritance = validate_warm_cell(config, control_config)
    hyperparameters = assert_adamw_hyperparameters_match(config, warm)
    burn_seed = build_warm_seed(
        config, warm, device=str(config["device"]), run_output=RUN_OUTPUT, seed_path=SEED_ROOT / "warm_seed_mps.pt"
    )
    control = cold_control_receipt()
    warm_path = SEALED_CONFIGS / f"{WARM_CELL_ID}.json"
    qbt.atomic_json(warm_path, config)
    control_path = SEALED_CONFIGS / f"cold_control_of_record_seed_{SEED}_{ARM_NAME}.json"
    shutil.copyfile(COLD_CONTROL_CONFIG, control_path)
    receipt = {
        "schema": SEAL_SCHEMA,
        "arm": "ddm_ng1_warm_transition",
        "axis": "[seal + bounded CPU mechanism smoke only; no Metal, no Modal, no contest eval]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "launch_authorized": False,
        "source_revision": qbr1.source_revision(),
        "single_lever": "AdamW optimizer state carried across the stage transition",
        "inheritance": inheritance,
        "learning_rate_is_not_a_lever": hyperparameters,
        "r10_warm_state": {
            key: warm[key]
            for key in (
                "source_checkpoint",
                "source_step",
                "optimizer_hyperparameters",
                "optimizer_state_entries",
                "optimizer_step_counters",
                "parameters",
                "shadow_vs_live_relative_distance",
            )
        },
        "warm_seed_burn": burn_seed,
        "sealed_configs": {
            "warm": qbt.file_fact(warm_path),
            "cold_control_of_record": qbt.file_fact(control_path),
        },
        "cold_control_of_record": control,
        "falsifiers": falsifiers(control),
        "authorized_configs_written": False,
        "fire_owner": "MAIN",
    }
    qbt.atomic_json(ARM_ROOT / "SEAL_RECEIPT.json", receipt)
    return receipt


def _displacement(reference: Mapping[str, torch.Tensor], checkpoint_path: str) -> float:
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    live = payload["live_state_dict"]
    numerator = sum(float((live[k].float() - reference[k].float()).pow(2).sum()) for k in sorted(live))
    return math.sqrt(numerator)


def bounded_resume_smoke() -> dict[str, Any]:
    """Warm-origin interruption/resume equivalence + a cold no-op detector, CPU only.

    The Metal lane belongs to the live QBR1 chain, so every segment here runs on CPU via
    the sealed ``_run_resume_smoke_segment``, which forces ``device='cpu'`` itself.
    """

    started = time.monotonic()
    warm = r10_warm_state()
    config = compile_warm_cell()
    cold_config = json.loads(COLD_CONTROL_CONFIG.read_text(encoding="utf-8"))
    reference = warm["shadow"]

    uninterrupted_root = SMOKE_ROOT / "warm_uninterrupted"
    resumed_root = SMOKE_ROOT / "warm_resumed"
    cold_root = SMOKE_ROOT / "cold_reference"
    for root in (uninterrupted_root, resumed_root, cold_root):
        root.mkdir(parents=True, exist_ok=True)

    seeds = {
        "uninterrupted": build_warm_seed(
            config, warm, device="cpu", run_output=uninterrupted_root,
            seed_path=SEED_ROOT / "warm_seed_cpu_uninterrupted.pt",
        ),
        "resumed": build_warm_seed(
            config, warm, device="cpu", run_output=resumed_root,
            seed_path=SEED_ROOT / "warm_seed_cpu_resumed.pt",
        ),
    }

    uninterrupted = qbr1._run_resume_smoke_segment(
        config, uninterrupted_root, stop_after=2,
        resume_from=Path(seeds["uninterrupted"]["seed"]["path"]),
    )
    interrupted = qbr1._run_resume_smoke_segment(
        config, resumed_root, stop_after=1,
        resume_from=Path(seeds["resumed"]["seed"]["path"]),
    )
    resumed = qbr1._run_resume_smoke_segment(
        config, resumed_root, stop_after=2,
        resume_from=Path(interrupted["checkpoint"]["path"]),
    )
    cold = qbr1._run_resume_smoke_segment(cold_config, cold_root, stop_after=1, resume_from=None)

    warm_first_step = _displacement(reference, interrupted["checkpoint"]["path"])
    cold_first_step = _displacement(reference, cold["checkpoint"]["path"])
    equal = {
        "completed_steps": uninterrupted["completed_steps"] == resumed["completed_steps"] == 2,
        "live_state": uninterrupted["live_state_sha256"] == resumed["live_state_sha256"],
        "ema_state": uninterrupted["ema_state_sha256"] == resumed["ema_state_sha256"],
        "archive": uninterrupted["archive"]["sha256"] == resumed["archive"]["sha256"],
    }
    warm_is_not_a_no_op = interrupted["live_state_sha256"] != cold["live_state_sha256"]
    result = {
        "schema": SMOKE_SCHEMA,
        "status": "PASS" if all(equal.values()) and warm_is_not_a_no_op else "FAIL",
        "axis": "[macOS-CPU exact-scorer bounded mechanism smoke; not a verdict, not a score]",
        "score_claim": False,
        "promotion_eligible": False,
        "real_chunk_pairs": 16,
        "scheduled_total_steps": 2,
        "interruption_after_steps": 1,
        "warm_origin": "r10 AdamW state via completed_steps=0 resume_from checkpoint",
        "equal": equal,
        "warm_is_not_a_no_op": warm_is_not_a_no_op,
        "first_update_displacement_l2": {
            "warm": warm_first_step,
            "cold": cold_first_step,
            "warm_over_cold": warm_first_step / cold_first_step if cold_first_step else None,
        },
        "warm_seeds": {name: value["seed"] for name, value in seeds.items()},
        "uninterrupted": uninterrupted,
        "interrupted_prefix": interrupted,
        "resumed": resumed,
        "cold_reference": cold,
        "elapsed_seconds": time.monotonic() - started,
        "metal_invocations": 0,
        "modal_invocations": 0,
        "contest_eval_invocations": 0,
        "all_payloads_retained": True,
    }
    qbt.atomic_json(SMOKE_ROOT / "BOUNDED_SMOKE_RESULT.json", result)
    if result["status"] != "PASS":
        raise NG1Error("warm-transition bounded smoke failed equivalence or the no-op detector")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("seal", help="compile, seal, and verify the warm cell (no launch)")
    sub.add_parser("resume-smoke", help="bounded CPU warm interruption/resume + no-op detector")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = seal() if args.action == "seal" else bounded_resume_smoke()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
