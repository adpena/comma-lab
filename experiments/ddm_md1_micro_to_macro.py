#!/usr/bin/env python3
"""ddm_md1 -- micro -> macro: how the born field's seg error is BORN, MOVES and DIES.

WHAT THIS MEASURES (and why it is not any prior arm's measurement)
------------------------------------------------------------------
Every prior read of the QBR1 burn is either MACRO (the milestone ``S_hat`` every 1,000 steps) or
STATIC MICRO (one frozen field re-weighted: sd1's surrogate-vs-exact map, gm1's gradient mass).
Neither shows the DYNAMICS.  This instrument reconstructs the EXACT per-site argmax at ~69 points
along the trajectory from the retained 16-step checkpoints, and classifies every one of the
6,291,456 sites (32 pairs x 384 x 512) by what its error trajectory DID:

  ALWAYS_CORRECT   never wrong at any swept checkpoint
  PERSISTENT       wrong at step 0 AND wrong at >= 90% of swept checkpoints
  NEW_PERSISTENT   correct at step 0, wrong at the terminal checkpoint
  TRANSIENT_BORN   correct at step 0, wrong somewhere in between, correct again at the terminal
  HEALED           wrong at step 0, not persistent, correct at the terminal
  CHURN            flips correct<->wrong more than ``--churn-flips`` times

The five error classes PARTITION every site that is ever wrong (falling rule, CHURN first), so the
per-class contributions sum EXACTLY to ``d_seg_hat(t)`` at every checkpoint -- that identity is the
calibration gate, and it is arithmetic, not an approximation.  NOTE the charter names four classes;
HEALED is the fifth that a PARTITION requires (a site wrong at 0 that the run FIXED is neither
persistent nor born), and it is reported explicitly rather than folded into another class.

THE READING THIS BUYS
---------------------
Of the TERMINAL seg error, what fraction sits in PERSISTENT sites (the representation cannot place
that boundary; no schedule/optimizer lever moves it) versus TRANSIENT/CHURN/NEW_PERSISTENT (the
optimizer put it there and could take it away)?  That is exactly the reachability question the
sub-0.12 accuracy corner asks.

REFERENCE FORM (verified at source; nothing here is a re-implementation)
-----------------------------------------------------------------------
* ``ddm_qbr1_born_fairform_burn_prep.py:600 _evaluate_milestone`` is the reference forward.  It runs
  ``with qbt.ema_scope(model, ema)`` (:612) -- so every retained milestone is the **EMA SHADOW**.
  The training objective in ``history.jsonl`` is the **LIVE** forward.  sd1 named that object gap
  and left it OWED; this instrument closes it by running BOTH forwards at every checkpoint.
* render + roundtrip + scorer: ``qbt.QBFLOWTorch.forward`` (:402), ``qbt.roundtrip_to_camera_uint8_ste``
  (:516), ``qbt.scorer_forward`` (:533) -- called, not copied.
* checkpoint schema: ``ddm_qbr1_born_fairform_burn_prep.py:534 _save_checkpoint`` holds
  ``live_state_dict``, ``ema.shadow``, ``optimizer_state_dict`` (single AdamW param group over
  ``model.parameters()``, :681), ``completed_steps``.
* parameter roles: ``qbt.state_tensor_role`` (:1946) -- seven roles, not invented here.
* n32 selection + HT weights: ``qbt.SELECTION_IDS`` / ``qbt.SELECTION_WEIGHTS`` (:78, :112);
  ``d_seg_hat = sum_p w_p * d_seg_p / 600`` is ``ddm_qbr1_born_fairform_burn_prep.py:447
  _weighted_mean``.

AUTHORITY AND LINEAGE
---------------------
* GT authority is DALI (``gt_cache_dali.pt``).  The QBF1 vehicle pins PyAV ``gt_n600.npz``
  (``qbt.py:123``), which is the target the burn actually descended against and the lineage the
  recorded milestone ``d_seg_hat`` uses.  BOTH are carried; they are never mixed inside one number.
* This instrument runs on CPU; the burn ran on MPS.  The CPU-vs-retained-MPS argmax residual is
  MEASURED at every milestone step rather than assumed away, and reported as a named number.
* ``[macOS-CPU advisory]``, ``score_claim=false``, non-promotable.  No Metal, no Modal, no eval.

ALWAYS KEEP THE PAYLOAD: every checkpoint's 32-pair argmax and margin-band code is persisted as a
compressed npz with a sha256 before the next checkpoint is touched, and every row is appended to a
JSONL, so a crash loses nothing and ``--mode sweep`` resumes from disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

INSTRUMENT = "ddm_md1_micro_to_macro"
SCHEMA = "ddm_md1_micro_to_macro.v1"
AXIS = "[macOS-CPU advisory; reconstructed from retained 16-step checkpoints; not contest authority]"

# ddm_dr1 n600 R-chain margin noise floor.  Law-resolved from sd1, never retyped by hand:
# experiments/ddm_sd1_surrogate_exact_map.py:DELTA_R_N600
DELTA_R_SOURCE = "experiments/ddm_sd1_surrogate_exact_map.py::DELTA_R_N600"

CELL_COLD = "cold_control_seed_20260902"
CELL_WARM = "warm_transition_seed_20260902"

DEFAULT_COLD = Path(
    "/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/runs/seed_20260902/control_native100"
)
DEFAULT_WARM = Path(
    "/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/ng1_warm_transition/runs"
    "/seed_20260902_warm_transition"
)
DEFAULT_COLD_CONFIG = Path(
    "/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/sealed_configs"
    "/seed_20260902_control_native100.json"
)

# Site-trajectory classes.  CHURN is first in the falling rule because it is defined by a flip
# COUNT and would otherwise be absorbed by whichever endpoint rule matched first.
CLASS_ALWAYS_CORRECT = "ALWAYS_CORRECT"
CLASS_CHURN = "CHURN"
CLASS_PERSISTENT = "PERSISTENT"
CLASS_NEW_PERSISTENT = "NEW_PERSISTENT"
CLASS_TRANSIENT_BORN = "TRANSIENT_BORN"
CLASS_HEALED = "HEALED"
ERROR_CLASSES = (
    CLASS_CHURN,
    CLASS_PERSISTENT,
    CLASS_NEW_PERSISTENT,
    CLASS_TRANSIENT_BORN,
    CLASS_HEALED,
)
SITE_CLASSES = (CLASS_ALWAYS_CORRECT, *ERROR_CLASSES)
CLASS_CODE = {name: index for index, name in enumerate(SITE_CLASSES)}

# The steps at which the sealed burn recorded a MILESTONE (ddm_qbr1_born_fairform_burn_prep
# MILESTONES).  Only those that are multiples of the 16-step checkpoint period have a retained
# weight state, and only those can be swept -- 1,000 and 3,000 are not multiples of 16, so the
# CPU-vs-retained-MPS calibration is available at 0 / 2,000 / 4,000 / 5,000 and nowhere else.
MILESTONE_STEPS = (0, 1000, 2000, 3000, 4000, 5000)

PERSISTENT_FRACTION = 0.90
DEFAULT_CHURN_FLIPS = 4

# Margin bands, in units of delta_R (sd1's convention).  Four bands from three thresholds.
BAND_EDGES_DELTA_R = (1.0, 2.0, 25.0)
BAND_NAMES = ("within_delta_R", "1_to_2_delta_R", "2_to_25_delta_R", "above_25_delta_R")

# The two rare classes the sealed recall-only dual pushes and nothing caps (sd1 section 4).
# Canonical comma10k order, never luma-sorted: 0 Road, 1 Lane, 2 Undrivable, 3 Movable, 4 MyCar.
RARE_CLASS_IDS = (1, 3)

# The sub-0.12 accuracy corner, TRANSFERRED (never re-derived here): the d_seg the born object
# would need at the QXR1 falsifier pose on its bound 106,626 B archive.  Source:
# `.omx/research/ddm_qn1_qbr1_n600_realization_ticket_20260903.md`
# (`d_seg_required_for_0_12_at_the_falsifier_pose`).  It is a DERIVED number on an n600
# population; the d_seg_hat measured here is an n32 Horvitz-Thompson ESTIMATE of that
# population, and qn1's own caveat that n32 -> n600 is untested on this vehicle travels with it.
SUB_012_DSEG_TARGET = 1.3646784205e-4
SUB_012_DSEG_TARGET_SOURCE = (
    ".omx/research/ddm_qn1_qbr1_n600_realization_ticket_20260903.md"
    "::d_seg_required_for_0_12_at_the_falsifier_pose"
)


class MD1Error(RuntimeError):
    """ddm_md1 refuses rather than emitting an unlabelled or unfaithful number."""


# ---------------------------------------------------------------------------
# cadence
# ---------------------------------------------------------------------------
def sweep_steps(total_steps: int = 5000) -> tuple[int, ...]:
    """The charter cadence: dense through the birth, medium through the peak, coarse to the end."""

    steps = [0]
    steps.extend(range(16, min(512, total_steps) + 1, 16))
    steps.extend(range(576, min(2048, total_steps) + 1, 64))
    steps.extend(range(2304, (total_steps // 16) * 16 + 1, 256))
    # Fold in every recorded milestone that actually HAS a 16-step checkpoint, so the CPU
    # reconstruction can be calibrated against the retained MPS argmax at those steps.
    steps.extend(s for s in MILESTONE_STEPS if s % 16 == 0 or s == total_steps)
    if total_steps not in steps:
        steps.append(total_steps)
    return tuple(sorted({int(s) for s in steps if 0 <= s <= total_steps}))


def checkpoint_path(run_root: Path, step: int, total_steps: int = 5000) -> Path:
    stage = run_root / "stage_01_fairform_finish/checkpoints"
    if step == total_steps:
        return stage / "stage_01_end.pt"
    return stage / f"periodic_{step:06d}.pt"


def available_steps(run_root: Path, total_steps: int = 5000) -> tuple[int, ...]:
    """Only steps whose checkpoint file EXISTS and is fully written (live cell safety)."""

    found: list[int] = []
    for step in sweep_steps(total_steps):
        if step == 0:
            found.append(0)
            continue
        path = checkpoint_path(run_root, step, total_steps)
        if path.is_file() and path.stat().st_size > 0:
            found.append(step)
    return tuple(found)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def atomic_npz(path: Path, **arrays: np.ndarray) -> dict[str, Any]:
    """Compressed npz written to a sibling ``.npz`` temp then renamed.

    ``np.savez_compressed`` APPENDS ``.npz`` to any name that does not already end in it, so the
    temp name must itself end in ``.npz`` or the rename target will not exist.
    """

    tmp = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return file_fact(path)


def band_index(abs_margin: np.ndarray, delta_r: float) -> np.ndarray:
    """0..3 band code for |margin| against the delta_R ladder (right-open intervals)."""

    edges = np.asarray(BAND_EDGES_DELTA_R, dtype=np.float64) * float(delta_r)
    return np.searchsorted(edges, np.asarray(abs_margin, dtype=np.float64), side="right").astype(
        np.uint8
    )


def signed_margin(logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    """``logits[g] - max_{c != g} logits[c]``; > 0 iff the site is currently correct.

    Exactly ``qbt.expected_flip_margin_loss``'s margin (:544), computed in numpy on the retained
    field rather than re-deriving a different quantity.
    """

    if logits.ndim != 4 or logits.shape[1] != 5:
        raise MD1Error("logits must be [B,5,H,W]")
    work = np.asarray(logits, dtype=np.float32).copy()
    target_logit = np.take_along_axis(work, target[:, None].astype(np.int64), axis=1)[:, 0]
    np.put_along_axis(work, target[:, None].astype(np.int64), -1.0e9, axis=1)
    return target_logit - work.max(axis=1)


def ht_weights_vector(pair_ids: Sequence[int], selection_ids: Sequence[int], weights: Sequence[float]) -> np.ndarray:
    lookup = dict(zip(selection_ids, weights, strict=True))
    return np.asarray([float(lookup[int(pid)]) for pid in pair_ids], dtype=np.float64)


def weighted_d_seg(per_pair: np.ndarray, weights: np.ndarray, population_n: int) -> float:
    """``sum_p w_p * d_seg_p / N`` -- the sealed HT estimator, in float64."""

    if per_pair.shape != weights.shape:
        raise MD1Error("per-pair d_seg and HT weights differ in shape")
    return float(np.sum(weights * per_pair) / float(population_n))


# ---------------------------------------------------------------------------
# sweep: reconstruct the exact argmax at every swept checkpoint
# ---------------------------------------------------------------------------
def _role_norms(state: Mapping[str, Any], role_of) -> dict[str, float]:
    import torch

    totals: dict[str, float] = {}
    for name, value in state.items():
        role = role_of(name)
        square = float(torch.as_tensor(value).detach().float().pow(2).sum().item())
        totals[role] = totals.get(role, 0.0) + square
    return {role: float(np.sqrt(value)) for role, value in totals.items()}


def _displacement_norms(current: Mapping[str, Any], previous: Mapping[str, Any], role_of) -> dict[str, float]:
    import torch

    totals: dict[str, float] = {}
    for name, value in current.items():
        if name not in previous:
            raise MD1Error(f"checkpoint state tensor missing in predecessor: {name}")
        role = role_of(name)
        delta = torch.as_tensor(value).detach().float() - torch.as_tensor(previous[name]).detach().float()
        totals[role] = totals.get(role, 0.0) + float(delta.pow(2).sum().item())
    return {role: float(np.sqrt(value)) for role, value in totals.items()}


def _optimizer_moment_norms(optimizer_state: Mapping[str, Any], names: Sequence[str], role_of) -> dict[str, Any]:
    import torch

    state = optimizer_state.get("state", {})
    groups = optimizer_state.get("param_groups", [])
    if len(groups) != 1:
        raise MD1Error("QBR1 optimizer is a single AdamW group; a different shape is not this run")
    order = list(groups[0]["params"])
    if len(order) != len(names):
        raise MD1Error("optimizer param order length differs from model parameter order")
    exp_avg: dict[str, float] = {}
    exp_avg_sq: dict[str, float] = {}
    steps: set[int] = set()
    absent: list[str] = []
    for index, name in zip(order, names, strict=True):
        entry = state.get(index)
        if entry is None:
            # AdamW only creates state for a parameter that has received a gradient.  An absent
            # entry is therefore a RECEIPT that the objective never reached this tensor, not a
            # loading defect -- so it is reported by name rather than silently skipped.
            absent.append(name)
            continue
        role = role_of(name)
        exp_avg[role] = exp_avg.get(role, 0.0) + float(
            torch.as_tensor(entry["exp_avg"]).detach().float().pow(2).sum().item()
        )
        exp_avg_sq[role] = exp_avg_sq.get(role, 0.0) + float(
            torch.as_tensor(entry["exp_avg_sq"]).detach().float().sum().item()
        )
        if "step" in entry:
            steps.add(int(torch.as_tensor(entry["step"]).item()))
    return {
        "exp_avg_l2_by_role": {role: float(np.sqrt(value)) for role, value in exp_avg.items()},
        "exp_avg_sq_l1_by_role": {role: float(value) for role, value in exp_avg_sq.items()},
        "adam_step_values": sorted(steps),
        "params_without_optimizer_state": absent,
        "roles_without_optimizer_state": sorted({role_of(name) for name in absent}),
    }


def sweep_checkpoint(
    *,
    qbt,
    model,
    posenet,
    segnet,
    device,
    state: Mapping[str, Any],
    pair_ids: Sequence[int],
    gt_dali: np.ndarray,
    gt_pyav: np.ndarray,
    delta_r: float,
) -> dict[str, Any]:
    """One forward over all 32 pairs from one weight state.  Returns arrays + per-pair rows."""

    import torch

    model.load_state_dict({name: torch.as_tensor(value).to(device) for name, value in state.items()}, strict=True)
    argmax_blocks: list[np.ndarray] = []
    band_blocks: list[np.ndarray] = []
    margin_blocks: list[np.ndarray] = []
    pose_blocks: list[np.ndarray] = []
    for chunk in qbt.pair_chunks(tuple(pair_ids), 16):
        ids = torch.tensor(chunk, dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(ids, height=qbt.EVAL_H, width=qbt.EVAL_W)
            camera = qbt.roundtrip_to_camera_uint8_ste(outputs["rgb_pair_01"])
            pose6, logits = qbt.scorer_forward(camera, posenet, segnet)
            logits_np = logits.detach().cpu().numpy()
            argmax_blocks.append(logits.argmax(dim=1).to(torch.uint8).cpu().numpy())
            pose_blocks.append(pose6.detach().cpu().numpy().astype(np.float32))
        index = np.asarray(chunk, dtype=np.int64)
        margin = signed_margin(logits_np, gt_dali[index])
        margin_blocks.append(margin.astype(np.float32))
        band_blocks.append(band_index(np.abs(margin), delta_r))
        del logits_np, logits, outputs, camera
    argmax = np.concatenate(argmax_blocks, axis=0)
    bands = np.concatenate(band_blocks, axis=0)
    margins = np.concatenate(margin_blocks, axis=0)
    poses = np.concatenate(pose_blocks, axis=0)
    index = np.asarray(pair_ids, dtype=np.int64)
    wrong_dali = argmax != gt_dali[index]
    wrong_pyav = argmax != gt_pyav[index]
    per_pair_dali = wrong_dali.reshape(len(pair_ids), -1).mean(axis=1).astype(np.float64)
    per_pair_pyav = wrong_pyav.reshape(len(pair_ids), -1).mean(axis=1).astype(np.float64)
    n_classes = 5
    pred_area = np.stack([(argmax == c).reshape(len(pair_ids), -1).mean(axis=1) for c in range(n_classes)], axis=1)
    return {
        "argmax": argmax,
        "bands": bands,
        "margins": margins,
        "poses": poses,
        "per_pair_d_seg_dali": per_pair_dali,
        "per_pair_d_seg_pyav": per_pair_pyav,
        "pred_area_by_pair_class": pred_area.astype(np.float64),
    }


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.set_num_threads(int(args.threads))
    from experiments import ddm_ar1_aa_render_price as ar1
    from experiments import ddm_qbt1_qbflow_trainer as qbt
    from experiments import ddm_sd1_surrogate_exact_map as sd1

    delta_r = float(sd1.DELTA_R_N600)
    store = Path(args.store)
    store.mkdir(parents=True, exist_ok=True)
    rows_path = store / f"sweep_rows_{args.cell}.jsonl"
    payload_root = store / "payloads" / args.cell
    payload_root.mkdir(parents=True, exist_ok=True)

    done: set[tuple[int, str]] = set()
    if rows_path.is_file():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done.add((int(row["step"]), str(row["forward"])))

    device = torch.device("cpu")
    posenet, segnet = qbt.load_differentiable_scorers(REPO / "upstream", device=device)
    posenet.eval()
    segnet.eval()
    model = qbt.load_initial_model(device)
    names = [name for name, _ in model.named_parameters()]
    role_of = qbt.state_tensor_role

    gt = ar1.load_ground_truth()
    gt_dali = np.asarray(gt["dali_seg"], dtype=np.uint8)
    gt_pyav = np.asarray(gt["pyav_seg"], dtype=np.uint8)
    pair_ids = list(qbt.SELECTION_IDS)
    weights = ht_weights_vector(pair_ids, qbt.SELECTION_IDS, qbt.SELECTION_WEIGHTS)

    run_root = Path(args.run_root)
    total_steps = int(args.total_steps)
    steps = available_steps(run_root, total_steps)
    if args.max_step is not None:
        steps = tuple(s for s in steps if s <= int(args.max_step))

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    initial_state_path = Path(config["initial_state"]["path"])
    initial = torch.load(initial_state_path, map_location="cpu", weights_only=False)
    initial_state = {name: value.detach().clone().float() for name, value in initial["state_dict"].items()}

    started = time.monotonic()
    written = 0
    for step in steps:
        if args.walltime_budget_s and (time.monotonic() - started) > float(args.walltime_budget_s):
            break
        if step == 0:
            states = {"shadow": initial_state}
            ckpt_fact = file_fact(initial_state_path)
            optimizer_row: dict[str, Any] = {
                "adam_step_values": [],
                "exp_avg_l2_by_role": {},
                "exp_avg_sq_l1_by_role": {},
                "params_without_optimizer_state": [],
                "roles_without_optimizer_state": [],
            }
            displacement: dict[str, float] = {}
            ema_meta = {"num_updates": 0, "decay": None}
        else:
            path = checkpoint_path(run_root, step, total_steps)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            live = {name: value.detach().clone().float() for name, value in payload["live_state_dict"].items()}
            shadow = {name: value.detach().clone().float() for name, value in payload["ema"]["shadow"].items()}
            states = {"live": live, "shadow": shadow}
            ckpt_fact = file_fact(path)
            optimizer_row = _optimizer_moment_norms(payload["optimizer_state_dict"], names, role_of)
            previous_path = checkpoint_path(run_root, step - 16, total_steps)
            if step - 16 == 0:
                displacement = _displacement_norms(live, initial_state, role_of)
            elif previous_path.is_file():
                previous = torch.load(previous_path, map_location="cpu", weights_only=False)
                displacement = _displacement_norms(live, previous["live_state_dict"], role_of)
                del previous
            else:
                displacement = {}
            ema_meta = {
                "num_updates": int(payload["ema"]["num_updates"]),
                "decay": float(payload["ema"]["decay"]),
            }
            del payload

        for kind, state in states.items():
            if (step, kind) in done:
                continue
            forward_started = time.monotonic()
            result = sweep_checkpoint(
                qbt=qbt,
                model=model,
                posenet=posenet,
                segnet=segnet,
                device=device,
                state=state,
                pair_ids=pair_ids,
                gt_dali=gt_dali,
                gt_pyav=gt_pyav,
                delta_r=delta_r,
            )
            npz_path = payload_root / f"{kind}_step_{step:06d}.npz"
            payload_fact = atomic_npz(
                npz_path,
                argmax_u8=result["argmax"],
                band_u8=result["bands"],
                pair_ids=np.asarray(pair_ids, dtype=np.int64),
            )
            margin_hist, margin_edges = np.histogram(
                result["margins"].reshape(-1), bins=np.asarray(args.margin_bins, dtype=np.float64)
            )
            row = {
                "schema": SCHEMA,
                "axis": AXIS,
                "score_claim": False,
                "cell": args.cell,
                "step": int(step),
                "forward": kind,
                "d_seg_hat_dali": weighted_d_seg(result["per_pair_d_seg_dali"], weights, qbt.N),
                "d_seg_hat_pyav": weighted_d_seg(result["per_pair_d_seg_pyav"], weights, qbt.N),
                "per_pair_d_seg_dali": [float(v) for v in result["per_pair_d_seg_dali"]],
                "per_pair_d_seg_pyav": [float(v) for v in result["per_pair_d_seg_pyav"]],
                "pred_area_mean_by_class": [
                    float(v) for v in result["pred_area_by_pair_class"].mean(axis=0)
                ],
                "pred_area_ht_by_class": [
                    float(np.sum(weights * result["pred_area_by_pair_class"][:, c]) / float(qbt.N))
                    for c in range(5)
                ],
                "band_site_counts": [int(v) for v in np.bincount(result["bands"].reshape(-1), minlength=4)],
                "margin_hist_counts": [int(v) for v in margin_hist],
                "margin_hist_edges": [float(v) for v in margin_edges],
                "margin_mean": float(result["margins"].mean()),
                "margin_p01": float(np.quantile(result["margins"], 0.01)),
                "margin_p50": float(np.quantile(result["margins"], 0.50)),
                "pose_mse_hat": float(
                    np.sum(
                        weights
                        * np.square(
                            result["poses"].astype(np.float64)
                            - np.asarray(gt["pyav_pose"], dtype=np.float64)[np.asarray(pair_ids)]
                        ).mean(axis=1)
                    )
                    / float(qbt.N)
                ),
                "ema": ema_meta,
                "optimizer": optimizer_row,
                "displacement_l2_by_role": displacement,
                "weight_l2_by_role": _role_norms(state, role_of),
                "checkpoint": ckpt_fact,
                "payload": payload_fact,
                "elapsed_s": float(time.monotonic() - forward_started),
                "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)),
            }
            with rows_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            written += 1
            del result

    receipt = {
        "schema": SCHEMA + ".sweep_receipt",
        "axis": AXIS,
        "score_claim": False,
        "cell": args.cell,
        "run_root": str(run_root),
        "steps_swept": list(steps),
        "forwards_written_this_process": written,
        "rows_path": str(rows_path),
        "payload_root": str(payload_root),
        "delta_R_n600": delta_r,
        "delta_R_source": DELTA_R_SOURCE,
        "gt_lineage": gt["lineage"],
        "host": {"platform": platform.platform(), "threads": int(args.threads)},
        "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)),
        "elapsed_s": float(time.monotonic() - started),
        "metal_invocations": 0,
        "modal_invocations": 0,
        "contest_eval_invocations": 0,
    }
    out = store / f"SWEEP_RECEIPT_{args.cell}.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


# ---------------------------------------------------------------------------
# analyze: site trajectory classification + the macro bridge
# ---------------------------------------------------------------------------
def classify_sites(wrong: np.ndarray, *, churn_flips: int, persistent_fraction: float) -> np.ndarray:
    """``wrong`` is [T, S] bool over swept checkpoints (T[0] == step 0).  Returns [S] uint8 codes.

    Falling rule, first match wins.  The five error classes plus ALWAYS_CORRECT PARTITION the
    sites, which is what makes the macro bridge an identity rather than an approximation.
    """

    if wrong.ndim != 2 or wrong.shape[0] < 2:
        raise MD1Error("site classification needs [T,S] with T >= 2")
    total = wrong.shape[0]
    ever = wrong.any(axis=0)
    first = wrong[0]
    last = wrong[-1]
    fraction = wrong.mean(axis=0)
    flips = np.count_nonzero(wrong[1:] != wrong[:-1], axis=0)
    codes = np.full(wrong.shape[1], CLASS_CODE[CLASS_ALWAYS_CORRECT], dtype=np.uint8)
    unassigned = ever.copy()
    churn = unassigned & (flips > int(churn_flips))
    codes[churn] = CLASS_CODE[CLASS_CHURN]
    unassigned &= ~churn
    persistent = unassigned & first & (fraction >= float(persistent_fraction))
    codes[persistent] = CLASS_CODE[CLASS_PERSISTENT]
    unassigned &= ~persistent
    new_persistent = unassigned & (~first) & last
    codes[new_persistent] = CLASS_CODE[CLASS_NEW_PERSISTENT]
    unassigned &= ~new_persistent
    transient = unassigned & (~first) & (~last)
    codes[transient] = CLASS_CODE[CLASS_TRANSIENT_BORN]
    unassigned &= ~transient
    codes[unassigned] = CLASS_CODE[CLASS_HEALED]
    del total
    return codes


def load_cube(store: Path, cell: str, forward: str, steps: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (argmax [T,P,H,W] uint8, band [T,P,H,W] uint8) for the swept steps, in order."""

    root = store / "payloads" / cell
    argmax: list[np.ndarray] = []
    bands: list[np.ndarray] = []
    for step in steps:
        path = root / f"{forward}_step_{step:06d}.npz"
        if not path.is_file():
            raise MD1Error(f"missing swept payload: {path}")
        with np.load(path) as payload:
            argmax.append(np.asarray(payload["argmax_u8"], dtype=np.uint8))
            bands.append(np.asarray(payload["band_u8"], dtype=np.uint8))
    return np.stack(argmax, axis=0), np.stack(bands, axis=0)


def macro_bridge(
    wrong: np.ndarray,
    codes: np.ndarray,
    *,
    pair_weights: np.ndarray,
    sites_per_pair: int,
    population_n: int,
) -> dict[str, Any]:
    """Decompose ``d_seg_hat(t)`` into the site classes as an EXACT INTEGER identity.

    The sealed estimator is ``d_seg_hat = sum_p w_p * n_wrong(p) / (N * H * W)`` with integer HT
    weights ``w_p in {15, 30}`` and integer site counts.  The decomposition is therefore carried in
    the integer numerator ``W(t) = sum_p w_p * n_wrong(p, t)``, where the class sums are exact by
    construction -- no float summation order can break the gate.  The float ``d_seg_hat`` is a
    single division of that integer, reported alongside.
    """

    if wrong.ndim != 3:
        raise MD1Error("macro bridge needs wrong[T, P, HW]")
    steps, pairs, per_pair = wrong.shape
    if per_pair != int(sites_per_pair) or pair_weights.shape != (pairs,):
        raise MD1Error("macro bridge geometry differs from the sealed selection")
    weights_int = np.asarray(pair_weights, dtype=np.int64)
    if not np.array_equal(weights_int.astype(np.float64), np.asarray(pair_weights, dtype=np.float64)):
        raise MD1Error("HT weights are not integral; the exact-integer bridge does not apply")
    codes_pp = codes.reshape(pairs, per_pair)
    denominator = float(population_n) * float(per_pair)
    totals = (wrong.sum(axis=2, dtype=np.int64) * weights_int[None, :]).sum(axis=1)
    per_class_int: dict[str, np.ndarray] = {}
    for name in SITE_CLASSES:
        mask = codes_pp == CLASS_CODE[name]
        counts = (wrong & mask[None, :, :]).sum(axis=2, dtype=np.int64)
        per_class_int[name] = (counts * weights_int[None, :]).sum(axis=1)
    stacked = np.zeros_like(totals)
    for name in SITE_CLASSES:
        stacked = stacked + per_class_int[name]
    residual = np.abs(stacked - totals)
    return {
        "weighted_wrong_site_numerator_by_step": [int(v) for v in totals],
        "denominator_population_n_times_sites_per_pair": denominator,
        "d_seg_hat_by_step": [float(v) / denominator for v in totals],
        "numerator_by_class": {name: [int(v) for v in per_class_int[name]] for name in SITE_CLASSES},
        "contribution_by_class": {
            name: [float(v) / denominator for v in per_class_int[name]] for name in SITE_CLASSES
        },
        "calibration_gate_max_abs_integer_residual": int(residual.max()),
        "calibration_gate_exact_zero": bool(np.all(residual == 0)),
    }


def run_analyze(args: argparse.Namespace) -> dict[str, Any]:
    from experiments import ddm_ar1_aa_render_price as ar1
    from experiments import ddm_qbt1_qbflow_trainer as qbt

    store = Path(args.store)
    rows_path = store / f"sweep_rows_{args.cell}.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_forward: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        by_forward.setdefault(str(row["forward"]), {})[int(row["step"])] = row

    gt = ar1.load_ground_truth()
    pair_ids = list(qbt.SELECTION_IDS)
    weights = ht_weights_vector(pair_ids, qbt.SELECTION_IDS, qbt.SELECTION_WEIGHTS)
    gt_key = "dali_seg" if args.gt_lineage == "dali" else "pyav_seg"
    gt_seg = np.asarray(gt[gt_key], dtype=np.uint8)[np.asarray(pair_ids, dtype=np.int64)]
    n_pairs, height, width = gt_seg.shape
    per_pair_sites = height * width
    sites = n_pairs * per_pair_sites
    # Per-site score weight, used only for the terminal SHARE tables; the bridge itself runs on the
    # exact integer numerator so no float summation order can move the calibration gate.
    site_weight = np.repeat(weights / (float(qbt.N) * per_pair_sites), per_pair_sites)

    out: dict[str, Any] = {
        "schema": SCHEMA + ".analysis",
        "axis": AXIS,
        "score_claim": False,
        "cell": args.cell,
        "gt_lineage": args.gt_lineage,
        "churn_flips": int(args.churn_flips),
        "persistent_fraction": PERSISTENT_FRACTION,
        "forwards": {},
    }
    gt_flat = gt_seg.reshape(-1)
    for forward in ("shadow", "live"):
        if forward not in by_forward:
            continue
        steps = sorted(by_forward[forward])
        # At step 0 the EMA shadow is initialised FROM the model, so live == shadow by
        # construction and only one payload was written.  Prepend that shared step-0 payload to
        # the live series so both series start from the SAME t=0 reference the classes need.
        if 0 not in steps and 0 in by_forward.get("shadow", {}):
            steps = [0, *steps]
        argmax_list: list[np.ndarray] = []
        band_list: list[np.ndarray] = []
        for step in steps:
            kind = forward if step in by_forward[forward] else "shadow"
            block_argmax, block_band = load_cube(store, args.cell, kind, [step])
            argmax_list.append(block_argmax[0])
            band_list.append(block_band[0])
        argmax = np.stack(argmax_list, axis=0)
        bands = np.stack(band_list, axis=0)
        del argmax_list, band_list
        flat = argmax.reshape(len(steps), -1)
        wrong = flat != gt_flat[None, :]
        codes = classify_sites(
            wrong, churn_flips=int(args.churn_flips), persistent_fraction=PERSISTENT_FRACTION
        )
        bridge = macro_bridge(
            wrong.reshape(len(steps), n_pairs, per_pair_sites),
            codes,
            pair_weights=weights,
            sites_per_pair=per_pair_sites,
            population_n=qbt.N,
        )
        terminal = wrong[-1]
        terminal_total = float(site_weight[terminal].sum())
        class_rows: dict[str, Any] = {}
        for name in SITE_CLASSES:
            mask = codes == CLASS_CODE[name]
            terminal_mask = mask & terminal
            class_rows[name] = {
                "sites": int(mask.sum()),
                "site_fraction": float(mask.sum() / sites),
                "terminal_wrong_sites": int(terminal_mask.sum()),
                "terminal_d_seg_contribution": float(site_weight[terminal_mask].sum()),
                "terminal_share_of_error": (
                    float(site_weight[terminal_mask].sum() / terminal_total) if terminal_total > 0 else 0.0
                ),
                "gt_class_histogram": [int(v) for v in np.bincount(gt_flat[mask], minlength=5)],
            }
        # per-(GT, runner-up) edge, taken at the TERMINAL checkpoint (the shipped field)
        terminal_pred = flat[-1]
        edge_rows: dict[str, dict[str, int]] = {}
        for name in ERROR_CLASSES:
            mask = (codes == CLASS_CODE[name]) & terminal
            if not mask.any():
                edge_rows[name] = {}
                continue
            pairs_gt = gt_flat[mask].astype(np.int64) * 5 + terminal_pred[mask].astype(np.int64)
            counts = np.bincount(pairs_gt, minlength=25)
            edge_rows[name] = {
                f"{ar1.CLASS_NAMES[g]}->{ar1.CLASS_NAMES[c]}": int(counts[g * 5 + c])
                for g in range(5)
                for c in range(5)
                if counts[g * 5 + c] > 0
            }
        # band membership of each class at the terminal checkpoint
        terminal_band = bands[-1].reshape(-1)
        band_rows: dict[str, list[int]] = {}
        for name in SITE_CLASSES:
            mask = codes == CLASS_CODE[name]
            band_rows[name] = [int(v) for v in np.bincount(terminal_band[mask], minlength=4)]
        # per-pair terminal contribution by class
        per_pair_rows = []
        codes_pp = codes.reshape(n_pairs, -1)
        terminal_pp = terminal.reshape(n_pairs, -1)
        for index, pid in enumerate(pair_ids):
            row = {"pair_id": int(pid), "ht_weight": float(weights[index])}
            for name in SITE_CLASSES:
                row[name] = int(((codes_pp[index] == CLASS_CODE[name]) & terminal_pp[index]).sum())
            per_pair_rows.append(row)
        # --- the excursion: which sites are BORN wrong, and do the born ones recover? ---
        numerator = np.asarray(bridge["weighted_wrong_site_numerator_by_step"], dtype=np.int64)
        peak_index = int(np.argmax(numerator))
        peak_step = int(steps[peak_index])
        at_zero = wrong[0]
        at_peak = wrong[peak_index]
        born = (~at_zero) & at_peak
        healed_by_peak = at_zero & (~at_peak)
        born_recovered = born & (~terminal)
        peak_pred = flat[peak_index]
        rare = np.isin(gt_flat, RARE_CLASS_IDS)
        overpaint = at_peak & np.isin(peak_pred, RARE_CLASS_IDS) & (~rare)
        overpaint_recovered = overpaint & (~terminal)
        excursion_block = {
            "peak_step": peak_step,
            "peak_index": peak_index,
            "d_seg_hat_at_zero": bridge["d_seg_hat_by_step"][0],
            "d_seg_hat_at_peak": bridge["d_seg_hat_by_step"][peak_index],
            "d_seg_hat_terminal": bridge["d_seg_hat_by_step"][-1],
            "wrong_sites_at_zero": int(at_zero.sum()),
            "wrong_sites_at_peak": int(at_peak.sum()),
            "wrong_sites_terminal": int(terminal.sum()),
            "born_sites": int(born.sum()),
            "healed_by_peak_sites": int(healed_by_peak.sum()),
            "born_recovered_sites": int(born_recovered.sum()),
            "born_recovered_fraction": (
                float(born_recovered.sum() / born.sum()) if born.any() else 0.0
            ),
            "born_gt_class_histogram": [int(v) for v in np.bincount(gt_flat[born], minlength=5)],
            "born_pred_class_histogram_at_peak": [
                int(v) for v in np.bincount(peak_pred[born], minlength=5)
            ],
            "born_site_class_histogram": [
                int(v) for v in np.bincount(codes[born], minlength=len(SITE_CLASSES))
            ],
            "rare_overpaint_sites_at_peak": int(overpaint.sum()),
            "rare_overpaint_share_of_peak_error": (
                float(overpaint.sum() / at_peak.sum()) if at_peak.any() else 0.0
            ),
            "rare_overpaint_recovered_fraction": (
                float(overpaint_recovered.sum() / overpaint.sum()) if overpaint.any() else 0.0
            ),
            "rare_overpaint_born_fraction": (
                float((overpaint & born).sum() / overpaint.sum()) if overpaint.any() else 0.0
            ),
        }
        # Take the terminal total from the BRIDGE's exact integer numerator, not from a second
        # float sum, so the reachability row and the bridge row can never disagree in the last ulp.
        denominator = float(bridge["denominator_population_n_times_sites_per_pair"])
        persistent_numerator = int(
            bridge["numerator_by_class"][CLASS_PERSISTENT][len(steps) - 1]
        )
        terminal_numerator = int(bridge["weighted_wrong_site_numerator_by_step"][-1])
        terminal_exact = terminal_numerator / denominator
        persistent_floor = persistent_numerator / denominator
        reachable = (terminal_numerator - persistent_numerator) / denominator
        excursion_block["reachability"] = {
            "target_d_seg": SUB_012_DSEG_TARGET,
            "target_source": SUB_012_DSEG_TARGET_SOURCE,
            "terminal_d_seg_hat": terminal_exact,
            "terminal_weighted_wrong_site_numerator": terminal_numerator,
            "persistent_weighted_wrong_site_numerator": persistent_numerator,
            "terminal_over_target": terminal_exact / SUB_012_DSEG_TARGET,
            "persistent_floor_d_seg_hat": persistent_floor,
            "persistent_floor_over_target": persistent_floor / SUB_012_DSEG_TARGET,
            "optimizer_reachable_d_seg_hat": reachable,
            "optimizer_reachable_share": (
                (terminal_numerator - persistent_numerator) / terminal_numerator
                if terminal_numerator > 0
                else 0.0
            ),
            "note": (
                "persistent_floor is what remains if EVERY non-PERSISTENT terminal error were "
                "removed by a schedule/optimizer lever; it is a FLOOR on this vehicle at this "
                "cadence, not a prediction that any lever reaches it."
            ),
        }
        # WHICH PAIRS lead the birth, and which EDGES the born sites take (the charter asks for
        # both; the per-pair view is what a later arm needs to pick a probe frame).
        born_pp = born.reshape(n_pairs, per_pair_sites)
        overpaint_pp = overpaint.reshape(n_pairs, per_pair_sites)
        gt_pp = gt_flat.reshape(n_pairs, per_pair_sites)
        peak_pred_pp = peak_pred.reshape(n_pairs, per_pair_sites)
        excursion_block["per_pair_birth"] = [
            {
                "pair_id": int(pid),
                "ht_weight": float(weights[i]),
                "born_sites": int(born_pp[i].sum()),
                "born_site_fraction": float(born_pp[i].mean()),
                "rare_overpaint_sites": int(overpaint_pp[i].sum()),
                "gt_lane_area_fraction": float((gt_pp[i] == 1).mean()),
                "gt_movable_area_fraction": float((gt_pp[i] == 3).mean()),
                "peak_lane_over_gt": (
                    float((peak_pred_pp[i] == 1).mean() / (gt_pp[i] == 1).mean())
                    if (gt_pp[i] == 1).any()
                    else None
                ),
                "peak_movable_over_gt": (
                    float((peak_pred_pp[i] == 3).mean() / (gt_pp[i] == 3).mean())
                    if (gt_pp[i] == 3).any()
                    else None
                ),
            }
            for i, pid in enumerate(pair_ids)
        ]
        born_edges = np.bincount(
            gt_flat[born].astype(np.int64) * 5 + peak_pred[born].astype(np.int64), minlength=25
        )
        excursion_block["born_edges_at_peak"] = {
            f"{ar1.CLASS_NAMES[g]}->{ar1.CLASS_NAMES[c]}": int(born_edges[g * 5 + c])
            for g in range(5)
            for c in range(5)
            if born_edges[g * 5 + c] > 0
        }
        trajectory_code = (
            at_zero.astype(np.uint8) + 2 * at_peak.astype(np.uint8) + 4 * terminal.astype(np.uint8)
        )
        excursion_block["trajectory_code_payload"] = atomic_npz(
            store / f"excursion_{args.cell}_{forward}_{args.gt_lineage}.npz",
            trajectory_code_u8=trajectory_code.reshape(n_pairs, height, width),
            pair_ids=np.asarray(pair_ids, dtype=np.int64),
            peak_step=np.asarray([peak_step], dtype=np.int64),
        )
        out["forwards"][forward] = {
            "steps": steps,
            "bridge": bridge,
            "classes": class_rows,
            "terminal_edges": edge_rows,
            "terminal_bands": band_rows,
            "per_pair_terminal": per_pair_rows,
            "excursion": excursion_block,
            "terminal_d_seg_hat": bridge["d_seg_hat_by_step"][-1],
            "terminal_d_seg_hat_from_site_weights": terminal_total,
            "sites": int(sites),
            "class_code_map": {name: int(CLASS_CODE[name]) for name in SITE_CLASSES},
        }
        codes_path = store / f"site_classes_{args.cell}_{forward}_{args.gt_lineage}.npz"
        out["forwards"][forward]["site_class_payload"] = atomic_npz(
            codes_path,
            site_class_u8=codes.reshape(n_pairs, height, width),
            pair_ids=np.asarray(pair_ids, dtype=np.int64),
        )
        del argmax, bands, flat, wrong
    path = store / f"ANALYSIS_{args.cell}_{args.gt_lineage}.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# report: JSONL-only tables (trajectory, over-paint birth, optimizer micro, live-vs-shadow)
# ---------------------------------------------------------------------------
def _rows_for(store: Path, cell: str) -> dict[str, dict[int, dict[str, Any]]]:
    path = store / f"sweep_rows_{cell}.jsonl"
    if not path.is_file():
        return {}
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped.setdefault(str(row["forward"]), {})[int(row["step"])] = row
    return grouped


def gt_area_fractions(gt_seg: np.ndarray, weights: np.ndarray, population_n: int) -> list[float]:
    """HT-weighted GT area fraction per class over the selection, matching the predicted-area row."""

    pairs = gt_seg.shape[0]
    per_pair = np.stack(
        [(gt_seg == c).reshape(pairs, -1).mean(axis=1) for c in range(5)], axis=1
    ).astype(np.float64)
    return [float(np.sum(weights * per_pair[:, c]) / float(population_n)) for c in range(5)]


def first_step_at_or_below(steps: Sequence[int], values: Sequence[float]) -> int | None:
    """First step whose value is <= 0 -- used for the live-minus-shadow sign change.

    The EMA shadow's parameter update is ``theta_bar <- d*theta_bar + (1-d)*theta``, so the shadow
    moves toward the live weights and stops moving away from them once the live field is no longer
    worse.  The step at which ``d_seg(live) - d_seg(shadow)`` first turns non-positive is therefore
    the MEASURED candidate for where the shadow's own d_seg turns over.  It is a candidate, not an
    identity: the EMA is on parameters, not on d_seg.
    """

    if len(steps) != len(values):
        raise MD1Error("sign-change scan needs one value per swept step")
    for step, value in zip(steps, values, strict=True):
        if float(value) <= 0.0:
            return int(step)
    return None


def first_crossing(steps: Sequence[int], values: Sequence[float], threshold: float) -> int | None:
    """First step whose value is at or above ``threshold`` -- the BIRTH of an over-paint.

    Lengths are checked BEFORE the scan: a ragged pair whose first element already crosses would
    otherwise return a step without ever reaching the mismatch, hiding a truncated series.
    """

    if len(steps) != len(values):
        raise MD1Error("over-paint crossing needs one value per swept step")
    for step, value in zip(steps, values, strict=True):
        if float(value) >= float(threshold):
            return int(step)
    return None


def milestone_reproduction(
    store: Path,
    cell: str,
    run_root: Path,
    milestone_steps: Sequence[int],
    pair_ids: Sequence[int],
    weights: np.ndarray | None = None,
    population_n: int = 600,
) -> list[dict[str, Any]]:
    """CPU reconstruction vs the RETAINED MPS argmax at each milestone -- a named residual.

    The burn ran on Metal; this instrument runs on CPU.  Rather than assume the axes agree, the
    disagreeing-site count and the ``d_seg_hat`` delta are measured at every milestone that
    retained its argmax.
    """

    out: list[dict[str, Any]] = []
    for step in milestone_steps:
        realized = run_root / f"milestones/step_{step:06d}/realized"
        payload = store / "payloads" / cell / f"shadow_step_{step:06d}.npz"
        if not realized.is_dir() or not payload.is_file():
            continue
        with np.load(payload) as block:
            mine = np.asarray(block["argmax_u8"], dtype=np.uint8)
        differing = 0
        total = 0
        mine_d_seg: list[float] = []
        retained_d_seg: list[float] = []
        for index, pair_id in enumerate(pair_ids):
            path = realized / f"pair_{pair_id:04d}.npz"
            if not path.is_file():
                continue
            with np.load(path) as block:
                reference = np.asarray(block["segnet_argmax_u8"], dtype=np.uint8)
                target = np.asarray(block["target_argmax_u8"], dtype=np.uint8)
            differing += int((mine[index] != reference).sum())
            total += int(reference.size)
            mine_d_seg.append(float((mine[index] != target).mean()))
            retained_d_seg.append(float((reference != target).mean()))
        if total == 0:
            continue
        row: dict[str, Any] = {
            "step": int(step),
            "sites_compared": total,
            "cpu_vs_retained_mps_differing_sites": differing,
            "cpu_vs_retained_mps_site_fraction": differing / total,
            "d_seg_per_pair_cpu": mine_d_seg,
            "d_seg_per_pair_retained_mps": retained_d_seg,
        }
        if weights is not None and len(mine_d_seg) == len(weights):
            cpu_hat = weighted_d_seg(np.asarray(mine_d_seg), weights, population_n)
            mps_hat = weighted_d_seg(np.asarray(retained_d_seg), weights, population_n)
            recorded = run_root / f"milestones/step_{step:06d}/MILESTONE.json"
            row["d_seg_hat_cpu_pyav"] = cpu_hat
            row["d_seg_hat_retained_mps_pyav"] = mps_hat
            row["d_seg_hat_relative_gap"] = (cpu_hat - mps_hat) / mps_hat if mps_hat else None
            if recorded.is_file():
                sealed = json.loads(recorded.read_text(encoding="utf-8"))
                row["d_seg_hat_recorded_in_milestone_json"] = float(sealed["d_seg_hat"])
                row["recomputed_minus_recorded"] = mps_hat - float(sealed["d_seg_hat"])
        out.append(row)
    return out


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    from experiments import ddm_ar1_aa_render_price as ar1
    from experiments import ddm_qbt1_qbflow_trainer as qbt

    store = Path(args.store)
    gt = ar1.load_ground_truth()
    pair_ids = list(qbt.SELECTION_IDS)
    weights = ht_weights_vector(pair_ids, qbt.SELECTION_IDS, qbt.SELECTION_WEIGHTS)
    index = np.asarray(pair_ids, dtype=np.int64)
    gt_areas = {
        "dali": gt_area_fractions(np.asarray(gt["dali_seg"])[index], weights, qbt.N),
        "pyav": gt_area_fractions(np.asarray(gt["pyav_seg"])[index], weights, qbt.N),
    }
    report: dict[str, Any] = {
        "schema": SCHEMA + ".report",
        "axis": AXIS,
        "score_claim": False,
        "gt_area_ht_by_class": gt_areas,
        "class_names": list(ar1.CLASS_NAMES),
        "overpaint_threshold": float(args.overpaint_threshold),
        "cells": {},
    }
    for cell, run_root in (
        (CELL_COLD, Path(args.run_root)),
        (CELL_WARM, Path(args.warm_run_root)),
    ):
        grouped = _rows_for(store, cell)
        if not grouped:
            continue
        block: dict[str, Any] = {"run_root": str(run_root), "forwards": {}}
        for forward, rows in sorted(grouped.items()):
            steps = sorted(rows)
            series = [rows[s] for s in steps]
            ratios = {
                ar1.CLASS_NAMES[c]: [
                    float(row["pred_area_ht_by_class"][c] / gt_areas["dali"][c]) for row in series
                ]
                for c in range(5)
            }
            block["forwards"][forward] = {
                "steps": steps,
                "d_seg_hat_dali": [float(row["d_seg_hat_dali"]) for row in series],
                "d_seg_hat_pyav": [float(row["d_seg_hat_pyav"]) for row in series],
                "pose_mse_hat": [float(row["pose_mse_hat"]) for row in series],
                "predicted_over_gt_area_ratio": ratios,
                "overpaint_birth_step": {
                    name: first_crossing(steps, values, float(args.overpaint_threshold))
                    for name, values in ratios.items()
                },
                "band_site_counts": [list(row["band_site_counts"]) for row in series],
                "margin_mean": [float(row["margin_mean"]) for row in series],
                "margin_p01": [float(row["margin_p01"]) for row in series],
                "margin_p50": [float(row["margin_p50"]) for row in series],
                "displacement_l2_by_role": [dict(row["displacement_l2_by_role"]) for row in series],
                "optimizer_exp_avg_l2_by_role": [
                    dict(row["optimizer"]["exp_avg_l2_by_role"]) for row in series
                ],
                "optimizer_exp_avg_sq_l1_by_role": [
                    dict(row["optimizer"]["exp_avg_sq_l1_by_role"]) for row in series
                ],
                "weight_l2_by_role": [dict(row["weight_l2_by_role"]) for row in series],
                "elapsed_s_total": float(sum(row["elapsed_s"] for row in series)),
            }
        shadow, live = grouped.get("shadow", {}), grouped.get("live", {})
        shared = sorted(set(shadow) & set(live))
        block["live_minus_shadow"] = {
            "steps": shared,
            "d_seg_hat_dali_live": [float(live[s]["d_seg_hat_dali"]) for s in shared],
            "d_seg_hat_dali_shadow": [float(shadow[s]["d_seg_hat_dali"]) for s in shared],
            "delta": [
                float(live[s]["d_seg_hat_dali"] - shadow[s]["d_seg_hat_dali"]) for s in shared
            ],
            "ratio": [
                float(live[s]["d_seg_hat_dali"] / shadow[s]["d_seg_hat_dali"]) for s in shared
            ],
            "first_step_live_at_or_below_shadow": first_step_at_or_below(
                shared,
                [float(live[s]["d_seg_hat_dali"] - shadow[s]["d_seg_hat_dali"]) for s in shared],
            ),
            "max_ratio": max(
                (float(live[s]["d_seg_hat_dali"] / shadow[s]["d_seg_hat_dali"]) for s in shared),
                default=None,
            ),
            "max_ratio_step": (
                max(shared, key=lambda s: live[s]["d_seg_hat_dali"] / shadow[s]["d_seg_hat_dali"])
                if shared
                else None
            ),
        }
        block["milestone_reproduction"] = milestone_reproduction(
            store, cell, run_root, MILESTONE_STEPS, pair_ids, weights, qbt.N
        )
        report["cells"][cell] = block

    # warm minus cold at IDENTICAL steps (same seed, same data order -> the optimizer state is the
    # only cause available to explain a difference)
    cold = _rows_for(store, CELL_COLD)
    warm = _rows_for(store, CELL_WARM)
    if cold and warm:
        paired: dict[str, Any] = {}
        for forward in ("shadow", "live"):
            shared = sorted(set(cold.get(forward, {})) & set(warm.get(forward, {})))
            if not shared:
                continue
            paired[forward] = {
                "steps": shared,
                "cold_d_seg_hat_dali": [float(cold[forward][s]["d_seg_hat_dali"]) for s in shared],
                "warm_d_seg_hat_dali": [float(warm[forward][s]["d_seg_hat_dali"]) for s in shared],
                "warm_minus_cold": [
                    float(warm[forward][s]["d_seg_hat_dali"] - cold[forward][s]["d_seg_hat_dali"])
                    for s in shared
                ],
                "cold_displacement_total": [
                    float(np.sqrt(sum(v**2 for v in cold[forward][s]["displacement_l2_by_role"].values())))
                    for s in shared
                ],
                "warm_displacement_total": [
                    float(np.sqrt(sum(v**2 for v in warm[forward][s]["displacement_l2_by_role"].values())))
                    for s in shared
                ],
            }
        report["warm_minus_cold"] = paired
    path = store / "REPORT.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(path)
    return report


# ---------------------------------------------------------------------------
# compare: is the warm cell's excursion a SUBSET of the cold cell's?
# ---------------------------------------------------------------------------
def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    store = Path(args.store)
    out: dict[str, Any] = {
        "schema": SCHEMA + ".compare",
        "axis": AXIS,
        "score_claim": False,
        "gt_lineage": args.gt_lineage,
        "forwards": {},
    }
    for forward in ("shadow", "live"):
        cold_path = store / f"excursion_{CELL_COLD}_{forward}_{args.gt_lineage}.npz"
        warm_path = store / f"excursion_{CELL_WARM}_{forward}_{args.gt_lineage}.npz"
        if not (cold_path.is_file() and warm_path.is_file()):
            continue
        with np.load(cold_path) as block:
            cold_code = np.asarray(block["trajectory_code_u8"], dtype=np.uint8).reshape(-1)
            cold_peak = int(block["peak_step"][0])
        with np.load(warm_path) as block:
            warm_code = np.asarray(block["trajectory_code_u8"], dtype=np.uint8).reshape(-1)
            warm_peak = int(block["peak_step"][0])
        # code = at_zero + 2*at_peak + 4*terminal; BORN == correct at 0 and wrong at the peak
        cold_born = ((cold_code & 1) == 0) & ((cold_code & 2) != 0)
        warm_born = ((warm_code & 1) == 0) & ((warm_code & 2) != 0)
        intersection = int((cold_born & warm_born).sum())
        warm_only = int((warm_born & ~cold_born).sum())
        out["forwards"][forward] = {
            "cold_peak_step": cold_peak,
            "warm_peak_step": warm_peak,
            "cold_born_sites": int(cold_born.sum()),
            "warm_born_sites": int(warm_born.sum()),
            "intersection_sites": intersection,
            "warm_only_sites": warm_only,
            "warm_born_absent_from_cold_fraction": (
                float(warm_only / warm_born.sum()) if warm_born.any() else 0.0
            ),
            "warm_born_fraction_of_cold": (
                float(warm_born.sum() / cold_born.sum()) if cold_born.any() else 0.0
            ),
        }
    path = store / f"COMPARE_{args.gt_lineage}.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    out["compare_path"] = str(path)
    return out


# ---------------------------------------------------------------------------
# tables: render the memo's markdown straight from the JSON, so no number is retyped
# ---------------------------------------------------------------------------
def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}g}"


def render_tables(store: Path, gt_lineage: str) -> str:
    from experiments import ddm_ar1_aa_render_price as ar1

    lines: list[str] = []
    report_path = store / "REPORT.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}

    for cell, block in sorted(report.get("cells", {}).items()):
        lines.append(f"### {cell} — trajectory (`d_seg_hat`, DALI authority / PyAV vehicle target)\n")
        for forward, series in sorted(block["forwards"].items()):
            steps = series["steps"]
            lines.append(f"`{forward}` forward — {len(steps)} checkpoints, "
                         f"steps {steps[0]}..{steps[-1]}\n")
            lines.append("| step | d_seg_hat (DALI) | d_seg_hat (PyAV) | " + " | ".join(
                f"{name}/GT" for name in ar1.CLASS_NAMES) + " |")
            lines.append("|---:|---:|---:|" + "---:|" * 5)
            ratios = series["predicted_over_gt_area_ratio"]
            for index, step in enumerate(steps):
                cells = [f"{step}", _fmt(series["d_seg_hat_dali"][index]),
                         _fmt(series["d_seg_hat_pyav"][index])]
                cells.extend(_fmt(ratios[name][index], 6) for name in ar1.CLASS_NAMES)
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
            lines.append("over-paint birth (first step at or above the threshold): "
                         + json.dumps(series["overpaint_birth_step"]) + "\n")
        if block.get("live_minus_shadow", {}).get("steps"):
            lms = block["live_minus_shadow"]
            lines.append(f"### {cell} — live minus EMA shadow (sd1's OWED gap)\n")
            lines.append("| step | live d_seg_hat | shadow d_seg_hat | live-shadow | live/shadow |")
            lines.append("|---:|---:|---:|---:|---:|")
            for index, step in enumerate(lms["steps"]):
                lines.append("| " + " | ".join([
                    f"{step}", _fmt(lms["d_seg_hat_dali_live"][index]),
                    _fmt(lms["d_seg_hat_dali_shadow"][index]),
                    _fmt(lms["delta"][index]), _fmt(lms["ratio"][index], 5)]) + " |")
            lines.append(
                "live first at or below shadow at step "
                f"**{lms.get('first_step_live_at_or_below_shadow')}**; peak live/shadow ratio "
                f"**{_fmt(lms.get('max_ratio') or 0.0, 5)}** at step "
                f"**{lms.get('max_ratio_step')}**\n")
        if block.get("milestone_reproduction"):
            lines.append(f"### {cell} — CPU reconstruction vs the retained MPS argmax\n")
            lines.append(
                "| milestone | sites compared | differing sites | site fraction | "
                "d_seg_hat CPU (PyAV) | d_seg_hat retained MPS | relative gap | recorded − recomputed |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
            for row in block["milestone_reproduction"]:
                lines.append("| " + " | ".join([
                    f"{row['step']}", f"{row['sites_compared']:,}",
                    f"{row['cpu_vs_retained_mps_differing_sites']:,}",
                    _fmt(row["cpu_vs_retained_mps_site_fraction"], 4),
                    _fmt(row.get("d_seg_hat_cpu_pyav") or 0.0, 10),
                    _fmt(row.get("d_seg_hat_retained_mps_pyav") or 0.0, 10),
                    f"{100 * (row.get('d_seg_hat_relative_gap') or 0.0):+.4f}%",
                    _fmt(row.get("recomputed_minus_recorded") or 0.0, 3)]) + " |")
            lines.append("")

    if report.get("warm_minus_cold"):
        lines.append("### warm minus cold at IDENTICAL steps (same seed, same data order)\n")
        for forward, block in sorted(report["warm_minus_cold"].items()):
            lines.append(f"`{forward}` forward\n")
            lines.append("| step | cold d_seg_hat | warm d_seg_hat | warm-cold | cold ‖Δθ‖ | warm ‖Δθ‖ |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            for index, step in enumerate(block["steps"]):
                lines.append("| " + " | ".join([
                    f"{step}", _fmt(block["cold_d_seg_hat_dali"][index]),
                    _fmt(block["warm_d_seg_hat_dali"][index]),
                    _fmt(block["warm_minus_cold"][index]),
                    _fmt(block["cold_displacement_total"][index], 5),
                    _fmt(block["warm_displacement_total"][index], 5)]) + " |")
            lines.append("")

    for path in sorted(store.glob(f"ANALYSIS_*_{gt_lineage}.json")):
        analysis = json.loads(path.read_text(encoding="utf-8"))
        lines.append(f"### {analysis['cell']} — site-trajectory classes ({gt_lineage} authority)\n")
        for forward, block in sorted(analysis["forwards"].items()):
            gate = block["bridge"]
            lines.append(
                f"`{forward}` forward — calibration gate max |Σclasses − total| = "
                f"**{gate['calibration_gate_max_abs_integer_residual']}** (integer), exact zero: "
                f"**{gate['calibration_gate_exact_zero']}**; terminal d_seg_hat "
                f"**{_fmt(block['terminal_d_seg_hat'], 10)}**\n")
            lines.append("| class | sites | site fraction | terminal wrong sites | terminal d_seg contribution | share of terminal error |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for name in SITE_CLASSES:
                row = block["classes"][name]
                lines.append("| " + " | ".join([
                    name, f"{row['sites']:,}", _fmt(row["site_fraction"], 4),
                    f"{row['terminal_wrong_sites']:,}",
                    _fmt(row["terminal_d_seg_contribution"], 6),
                    f"{100 * row['terminal_share_of_error']:.3f}%"]) + " |")
            lines.append("")
            exc = block["excursion"]
            lines.append(
                f"excursion (`{forward}`): peak at step **{exc['peak_step']}**, d_seg_hat "
                f"{_fmt(exc['d_seg_hat_at_zero'], 8)} → {_fmt(exc['d_seg_hat_at_peak'], 8)} → "
                f"{_fmt(exc['d_seg_hat_terminal'], 8)}; **{exc['born_sites']:,}** sites born wrong, "
                f"**{100 * exc['born_recovered_fraction']:.2f}%** of them recovered by the terminal "
                f"checkpoint; **{exc['healed_by_peak_sites']:,}** wrong-at-zero sites healed by the peak; "
                f"rare-class over-paint at the peak **{exc['rare_overpaint_sites_at_peak']:,}** sites = "
                f"**{100 * exc['rare_overpaint_share_of_peak_error']:.2f}%** of the peak error, "
                f"**{100 * exc['rare_overpaint_born_fraction']:.2f}%** of them born during the run, "
                f"**{100 * exc['rare_overpaint_recovered_fraction']:.2f}%** recovered.\n")
            reach = exc["reachability"]
            lines.append(
                f"reachability (`{forward}`): terminal d_seg_hat "
                f"**{_fmt(reach['terminal_d_seg_hat'], 8)}** = "
                f"**{reach['terminal_over_target']:.2f}x** the sub-0.12 target "
                f"{_fmt(reach['target_d_seg'], 8)}; the PERSISTENT floor is "
                f"**{_fmt(reach['persistent_floor_d_seg_hat'], 8)}** = "
                f"**{reach['persistent_floor_over_target']:.2f}x** the target; "
                f"**{100 * reach['optimizer_reachable_share']:.2f}%** of the terminal error is "
                "optimizer-reachable (non-PERSISTENT).\n")
            born_edges = exc.get("born_edges_at_peak", {})
            if born_edges:
                top_born = sorted(born_edges.items(), key=lambda kv: -kv[1])[:8]
                lines.append(f"BORN (GT→predicted) edges at the peak, `{forward}`: "
                             + ", ".join(f"{k} {v:,}" for k, v in top_born) + "\n")
            leaders = sorted(
                exc.get("per_pair_birth", []), key=lambda r: -r["born_sites"]
            )[:6]
            if leaders:
                lines.append(f"pairs leading the birth, `{forward}`: "
                             + ", ".join(
                                 f"pair {r['pair_id']} ({r['born_sites']:,} born, Lane×"
                                 f"{_fmt(r['peak_lane_over_gt'] or 0.0, 4)})"
                                 for r in leaders) + "\n")
            edges = block["terminal_edges"]
            lines.append(f"terminal (GT→predicted) edges by class, `{forward}`:\n")
            for name in ERROR_CLASSES:
                top = sorted(edges.get(name, {}).items(), key=lambda kv: -kv[1])[:6]
                if top:
                    lines.append(f"* **{name}** — " + ", ".join(f"{k} {v:,}" for k, v in top))
            lines.append("")
            bands = block["terminal_bands"]
            lines.append(f"terminal margin band of each class ({', '.join(BAND_NAMES)}), `{forward}`:\n")
            for name in SITE_CLASSES:
                lines.append(f"* **{name}** — " + " / ".join(f"{v:,}" for v in bands[name]))
            lines.append("")

    compare_path = store / f"COMPARE_{gt_lineage}.json"
    if compare_path.is_file():
        compare = json.loads(compare_path.read_text(encoding="utf-8"))
        if compare.get("forwards"):
            lines.append("### warm excursion vs cold excursion (prediction 2)\n")
            lines.append("| forward | cold peak | warm peak | cold born | warm born | intersection | warm-only | warm-only fraction |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for forward, block in sorted(compare["forwards"].items()):
                lines.append("| " + " | ".join([
                    forward, f"{block['cold_peak_step']}", f"{block['warm_peak_step']}",
                    f"{block['cold_born_sites']:,}", f"{block['warm_born_sites']:,}",
                    f"{block['intersection_sites']:,}", f"{block['warm_only_sites']:,}",
                    f"{100 * block['warm_born_absent_from_cold_fraction']:.2f}%"]) + " |")
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode", choices=("sweep", "analyze", "report", "compare", "tables"), required=True
    )
    parser.add_argument("--cell", default=CELL_COLD)
    parser.add_argument("--run-root", default=str(DEFAULT_COLD))
    parser.add_argument("--config", default=str(DEFAULT_COLD_CONFIG))
    parser.add_argument("--store", default="/Volumes/APDataStore/pact/ddm_md1_micro_macro")
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--churn-flips", type=int, default=DEFAULT_CHURN_FLIPS)
    parser.add_argument("--gt-lineage", choices=("dali", "pyav"), default="dali")
    parser.add_argument("--walltime-budget-s", type=float, default=None)
    parser.add_argument("--warm-run-root", default=str(DEFAULT_WARM))
    parser.add_argument("--overpaint-threshold", type=float, default=1.05)
    parser.add_argument(
        "--margin-bins",
        type=float,
        nargs="+",
        default=[-40.0, -20.0, -10.0, -5.0, -2.0, -1.0, -0.5, -0.25, -0.1, -0.05, -0.021881818771362305,
                 0.0, 0.021881818771362305, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "sweep":
        receipt = run_sweep(args)
        print(json.dumps({k: v for k, v in receipt.items() if k != "gt_lineage"}, indent=2, sort_keys=True)[:4000])
    elif args.mode == "report":
        report = run_report(args)
        print(json.dumps({"report_path": report["report_path"], "cells": sorted(report["cells"])}, indent=2))
    elif args.mode == "compare":
        print(json.dumps(run_compare(args), indent=2, sort_keys=True))
    elif args.mode == "tables":
        print(render_tables(Path(args.store), args.gt_lineage))
    else:
        result = run_analyze(args)
        summary = {
            forward: {
                "steps": len(block["steps"]),
                "calibration_gate_exact_zero": block["bridge"]["calibration_gate_exact_zero"],
                "calibration_gate_max_abs_integer_residual": block["bridge"]["calibration_gate_max_abs_integer_residual"],
                "terminal_d_seg_hat": block["terminal_d_seg_hat"],
                "terminal_share": {n: block["classes"][n]["terminal_share_of_error"] for n in SITE_CLASSES},
            }
            for forward, block in result["forwards"].items()
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
