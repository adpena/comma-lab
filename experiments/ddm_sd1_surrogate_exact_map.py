#!/usr/bin/env python3
"""ddm_sd1 -- where does the expected-flip SURROGATE mis-price the EXACT argmax?

WHAT THIS MEASURES
------------------
QBR1 seed 20260902 fell apart on its own objective: ``seg_expected_flip_realized`` (the training
surrogate) fell monotonically 0.005018 -> 0.003254 (-35.1%) while the exact ``d_seg_hat`` rose to a
peak at step 2,000 and ended +9.56% above its start (ng1, ``ddm_ng1_warm_transition_burn_design_20260904.md``).
The loss never registered the excursion.  This instrument asks WHERE that mis-pricing lives --
per (GT class, competitor class) edge, per margin band, per pair -- so the next race is chosen on a
decomposition instead of a guess.

THE TWO QUANTITIES, BOTH FROM THE SAME LOGITS ARRAY (verified at source)
-----------------------------------------------------------------------
``ddm_qbt1_qbflow_trainer.expected_flip_margin_loss`` (:527-548) computes, per pixel::

    target_logit = logits[g]                       # g = GT class at that pixel
    other        = logits with channel g set to -1e9
    margin       = target_logit - other.amax()     # signed; > 0 == currently correct
    p_flip       = sigmoid(-margin / tau)          # THE SURROGATE
    loss         = p_flip.mean() over pixels, HT-weighted over pairs

and the EXACT contribution of the same pixel to ``d_seg`` is ``1[argmax(logits) != g]``, which is
exactly ``1[margin < 0]`` (ties at ``margin == 0`` resolve to the lowest channel index, which the
calibration leg below measures rather than assumes).

So the surrogate and the exact term are two functionals of ONE scalar field -- the margin -- and
the mis-pricing is fully characterised by how ``sigmoid(-m/tau)`` differs from ``1[m < 0]`` over the
realised margin distribution.  No re-render is required: the milestone retains the logits.

WHAT THE MILESTONE RETAINS (verified at source, so nothing is reproduced that was already kept)
----------------------------------------------------------------------------------------------
``ddm_qbr1_born_fairform_burn_prep.py:420 _evaluate_milestone`` -> ``:432 with qbt.ema_scope(model, ema),
torch.no_grad():`` -> ``:439 qbt._retain_eval_outputs`` (``ddm_qbt1_qbflow_trainer.py:1907-1936``)
writes one ``realized/pair_<id>.npz`` per selected pair holding ``camera_pair_u8``,
``segnet_logits_f16`` (5, 384, 512), ``segnet_argmax_u8``, ``target_argmax_u8``,
``posenet_pose6_f32``, ``target_pose6_f32``.  The exact per-pixel argmax is therefore RETAINED, not
merely reproducible, and this instrument reads it rather than re-rendering.  ar1
(``experiments/ddm_ar1_aa_render_price.py``) remains the reference for the render path and supplies
the GT-lineage loader here; it is imported, not rebuilt.

THE OBJECT MIS-MATCH THIS INSTRUMENT SEPARATES (structural, MEASURED at source)
------------------------------------------------------------------------------
The milestone forward runs inside ``ema_scope`` -- it is the **EMA shadow**.  The training objective
in ``history.jsonl`` is the **live-weights** forward.  So the run's own headline decoupling mixes two
causes: (a) a CALIBRATION gap between ``sigmoid(-m/tau)`` and ``1[m<0]`` on one object, and (b) an
OBJECT gap between the live weights the loss saw and the EMA shadow the milestone scored.  This
instrument measures (a) exactly, because both of its quantities come from the SAME retained logits
array, and it reports (b) as a named residual rather than absorbing it.

BANDS
-----
``delta_R = 0.021881818771362305`` is the n600 R-chain margin noise floor measured by ddm_dr1
(``.omx/research/arm_final_messages/ddm_dr1_final_20260903T221804Z.md:3``).  A pixel with
``|margin| < delta_R`` is INSIDE the render->camera->uint8 roundtrip's own noise: its class is not
decided by the field.  Outside it, the class is decided.  Every table is split on that band.

AUTHORITY
---------
* GT authority is DALI (``gt_cache_dali.pt``).  The QBF1 vehicle pins ``gt_n600.npz`` (PyAV
  lineage, ``ddm_qbt1_qbflow_trainer.py:123``), which is the target the surrogate was actually
  computed against, so BOTH are measured and reported and never mixed inside one number.
* No score claim.  ``[macOS-CPU advisory]``, non-promotable.  The milestone numbers this calibrates
  against are ``[macOS-MPS n32 stratified advisory]``.

ALWAYS KEEP THE PAYLOAD: every per-pair row is appended to a JSONL as it is produced (so a crash
loses nothing and the run resumes from disk), and the margin / competitor / argmax fields of the
first ``--retain-fields-pairs`` pairs of every milestone are persisted with sha256s.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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

from experiments import ddm_ar1_aa_render_price as ar1  # reference instrument, reused
from experiments import ddm_qbt1_qbflow_trainer as qbt

INSTRUMENT = "ddm_sd1_surrogate_exact_map"
SCHEMA = "ddm_sd1_surrogate_exact_map.v1"
AXIS = "[macOS-CPU advisory; retained EMA-shadow scorer logits; not contest authority]"

# ddm_dr1 n600 R-chain margin noise floor.  Source:
# .omx/research/arm_final_messages/ddm_dr1_final_20260903T221804Z.md:3
DELTA_R_N600 = 0.021881818771362305
DELTA_R_SOURCE = ".omx/research/arm_final_messages/ddm_dr1_final_20260903T221804Z.md:3"

CLASS_NAMES = ar1.CLASS_NAMES  # canonical comma10k order; never luma-sorted
N_CLASSES = len(CLASS_NAMES)

# tau schedule pins (ddm_qbt1_qbflow_trainer.py:626 tau_for_step, start 0.15 -> end 0.05)
TOTAL_STEPS = 5000

# Comparison taus held fixed across milestones so a per-edge ratio change cannot be an artefact of
# the anneal.  The milestone's own evaluation tau is always measured in addition to these.
REFERENCE_TAUS = (0.15, 0.10, 0.05)

LINEAGES = ("vehicle_pyav", "authority_dali")

# |margin| capture thresholds, as multiples of delta_R.  vr1 row 1's allocator concentrates loss
# mass on small-|margin| pixels; the single delta_R band is only its NARROWEST possible setting, so
# the reach of the whole family is read off this curve instead of off one band.
MARGIN_CAPTURE_MULTIPLES = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0)


class SD1Error(RuntimeError):
    """Fail-closed error for the sd1 instrument."""


# ---------------------------------------------------------------------------
# custody
# ---------------------------------------------------------------------------
def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: Any) -> dict[str, Any]:
    """Write ``payload`` as JSON via tmp+rename and return its file fact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, indent=2, sort_keys=True).encode()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, path)
    return {"path": str(path), "bytes": len(blob), "sha256": sha256_bytes(blob)}


def atomic_npz(path: Path, **arrays: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)
    return ar1.file_fact(path)


# ---------------------------------------------------------------------------
# the surrogate, re-expressed exactly
# ---------------------------------------------------------------------------
def tau_for_milestone(step: int, total_steps: int = TOTAL_STEPS) -> float:
    """The tau the training loss used when it EVALUATED the milestone-``step`` state.

    ``history.jsonl`` line ``completed_steps == S + 1`` carries the objective computed on the state
    that exists after ``S`` updates (the forward precedes the update), and its tau is
    ``tau_for_step(S)``.  At ``S == total_steps`` training is over and the last in-range value
    ``tau_for_step(total_steps - 1)`` is used; the convention is recorded in the output so no reader
    has to infer it, and REFERENCE_TAUS make every cross-milestone claim convention-free.
    """

    return qbt.tau_for_step(min(int(step), int(total_steps) - 1), int(total_steps))


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """``1/(1+exp(-z))`` without overflow, in float64."""

    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def margin_and_competitor(
    logits: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(margin, competitor, argmax)`` for one pair.

    ``margin = logits[target] - max_{c != target} logits[c]`` and ``competitor`` is that arg-max,
    matching ``expected_flip_margin_loss`` (:535-541) which masks the target channel with ``-1e9``
    before ``amax``.  ``argmax`` is the plain unmasked arg-max, i.e. the realised SegNet class.
    """

    if logits.ndim != 3 or logits.shape[0] != N_CLASSES:
        raise SD1Error(f"logits geometry differs: {logits.shape}")
    if target.shape != logits.shape[1:]:
        raise SD1Error(f"target geometry differs: {target.shape} vs {logits.shape[1:]}")
    work = logits.astype(np.float32, copy=True)
    target_index = target.astype(np.int64)[None]
    target_logit = np.take_along_axis(work, target_index, axis=0)[0]
    np.put_along_axis(work, target_index, -1.0e9, axis=0)
    competitor = work.argmax(axis=0).astype(np.uint8)
    margin = target_logit - work.max(axis=0)
    argmax = logits.argmax(axis=0).astype(np.uint8)
    return margin.astype(np.float32), competitor, argmax


# bin layout: ((gt_class * N_CLASSES + competitor) * 2 + annulus) * 2 + flip
N_BINS = N_CLASSES * N_CLASSES * 2 * 2


def bin_index(gt: np.ndarray, competitor: np.ndarray, annulus: np.ndarray, flip: np.ndarray) -> np.ndarray:
    base = gt.astype(np.int64) * N_CLASSES + competitor.astype(np.int64)
    return (base * 2 + annulus.astype(np.int64)) * 2 + flip.astype(np.int64)


def margin_capture(margin: np.ndarray, flip: np.ndarray, delta_r: float) -> dict[str, list[float]]:
    """Pixels and exact flips inside each ``|margin| < k * delta_r`` band.

    This is the REACH CURVE of a margin-magnitude allocator (vr1 row 1): at band width ``k`` such an
    allocator concentrates its mass on ``pixels[k]`` pixels and can therefore see ``flips[k]`` of
    the exact error.  The single ``k = 1`` band is only the narrowest setting of that family, so
    reading row 1's reach off one band alone would understate it.
    """

    absolute = np.abs(margin)
    pixels: list[float] = []
    flips: list[float] = []
    for multiple in MARGIN_CAPTURE_MULTIPLES:
        inside = absolute < float(multiple) * float(delta_r)
        pixels.append(float(inside.sum()))
        flips.append(float((inside & flip).sum()))
    return {
        "multiples": [float(m) for m in MARGIN_CAPTURE_MULTIPLES],
        "pixels": pixels,
        "flips": flips,
    }


def accumulate_pair(
    margin: np.ndarray,
    gt: np.ndarray,
    competitor: np.ndarray,
    taus: Sequence[float],
    delta_r: float = DELTA_R_N600,
    flip: np.ndarray | None = None,
) -> dict[str, Any]:
    """Bin one pair's pixels and return per-bin pixel / surrogate / gradient mass.

    ``surrogate`` is ``sigmoid(-margin/tau)``; ``grad`` is
    ``|d surrogate / d margin| = sigmoid'(...)/tau`` -- the quantity that actually steers the
    descent, which is what separates "the loss counts this pixel" from "the loss can MOVE this
    pixel".

    ``flip`` is the EXACT term.  In exact arithmetic it equals ``1[margin < 0]``, but the retained
    logits are stored as float16 (``ddm_qbt1_qbflow_trainer.py:1918``) and that cast manufactures
    exact ties, so ``margin < 0`` under-counts the true flips by ~1% (MEASURED: 7 of 488 sites on
    pair 4 at step 0).  The caller therefore passes the flip indicator derived from the RETAINED
    ``segnet_argmax_u8`` -- the array whose disagreement rate reproduces the milestone's own
    recorded ``d_seg`` bit-for-bit -- so per-edge exact counts sum to the recorded value exactly.
    ``margin < 0`` is still returned separately as the float16-consistency diagnostic.
    """

    flat_m = margin.reshape(-1).astype(np.float64)
    flat_g = gt.reshape(-1)
    flat_c = competitor.reshape(-1)
    margin_negative = flat_m < 0.0
    flip = margin_negative if flip is None else flip.reshape(-1).astype(bool)
    annulus = np.abs(flat_m) < float(delta_r)
    idx = bin_index(flat_g, flat_c, annulus, flip)
    pixels = np.bincount(idx, minlength=N_BINS).astype(np.int64)
    out: dict[str, Any] = {
        "pixels": pixels,
        "surrogate": {},
        "grad": {},
        "total_pixels": int(flat_m.size),
        "flips": int(flip.sum()),
        "margin_negative": int(margin_negative.sum()),
        "flip_vs_margin_negative_sites": int((flip != margin_negative).sum()),
        "margin_capture": margin_capture(flat_m, flip, delta_r),
    }
    for tau in taus:
        probability = stable_sigmoid(-flat_m / float(tau))
        gradient = probability * (1.0 - probability) / float(tau)
        key = f"{float(tau):.6f}"
        out["surrogate"][key] = np.bincount(idx, weights=probability, minlength=N_BINS)
        out["grad"][key] = np.bincount(idx, weights=gradient, minlength=N_BINS)
    return out


def decode_bins(vector: np.ndarray) -> np.ndarray:
    """Reshape a length-``N_BINS`` vector to ``(gt, competitor, annulus, flip)``."""

    return np.asarray(vector).reshape(N_CLASSES, N_CLASSES, 2, 2)


# ---------------------------------------------------------------------------
# milestone reading
# ---------------------------------------------------------------------------
def milestone_dir(run_root: Path, step: int) -> Path:
    return Path(run_root) / "milestones" / f"step_{int(step):06d}"


def read_milestone_json(run_root: Path, step: int) -> dict[str, Any]:
    path = milestone_dir(run_root, step) / "MILESTONE.json"
    if not path.is_file():
        raise SD1Error(f"milestone json is absent: {path}")
    with open(path, "rb") as handle:
        return json.load(handle)


def read_pair_arrays(run_root: Path, step: int, pair_id: int) -> dict[str, np.ndarray]:
    path = milestone_dir(run_root, step) / "realized" / f"pair_{int(pair_id):04d}.npz"
    if not path.is_file():
        raise SD1Error(f"retained pair payload is absent: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {
            "segnet_logits_f16": np.asarray(payload["segnet_logits_f16"]),
            "segnet_argmax_u8": np.asarray(payload["segnet_argmax_u8"]),
            "target_argmax_u8": np.asarray(payload["target_argmax_u8"]),
        }


def sample_weight_lookup() -> dict[int, float]:
    return {
        int(pair_id): float(weight)
        for pair_id, weight in zip(qbt.SELECTION_IDS, qbt.SELECTION_WEIGHTS, strict=True)
    }


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------
def measure_milestone(
    run_root: Path,
    step: int,
    pair_ids: Sequence[int],
    ground_truth: Mapping[str, Any],
    taus: Sequence[float],
    *,
    rows_handle: Any = None,
    retain_dir: Path | None = None,
    retain_fields_pairs: int = 0,
    delta_r: float = DELTA_R_N600,
) -> dict[str, Any]:
    """Measure one milestone across ``pair_ids`` for both GT lineages."""

    milestone = read_milestone_json(run_root, step)
    weights = sample_weight_lookup()
    totals: dict[str, dict[str, Any]] = {
        lineage: {
            "pixels": np.zeros(N_BINS, dtype=np.int64),
            "surrogate": {f"{t:.6f}": np.zeros(N_BINS) for t in taus},
            "grad": {f"{t:.6f}": np.zeros(N_BINS) for t in taus},
            "capture_pixels": np.zeros(len(MARGIN_CAPTURE_MULTIPLES)),
            "capture_flips": np.zeros(len(MARGIN_CAPTURE_MULTIPLES)),
        }
        for lineage in LINEAGES
    }
    recorded = {row["pair_id"]: row for row in milestone["pair_rows"]}
    pair_rows: list[dict[str, Any]] = []
    retained_fields: list[dict[str, Any]] = []
    f16_argmax_mismatch = 0
    f16_argmax_pixels = 0
    vehicle_target_matches_gt_cache = True

    for order, pair_id in enumerate(pair_ids):
        arrays = read_pair_arrays(run_root, step, pair_id)
        logits = arrays["segnet_logits_f16"].astype(np.float32)
        retained_argmax = arrays["segnet_argmax_u8"]
        targets = {
            "vehicle_pyav": arrays["target_argmax_u8"],
            "authority_dali": np.asarray(ground_truth["dali_seg"][int(pair_id)], dtype=np.uint8),
        }
        if not np.array_equal(
            targets["vehicle_pyav"], np.asarray(ground_truth["pyav_seg"][int(pair_id)], dtype=np.uint8)
        ):
            vehicle_target_matches_gt_cache = False

        row: dict[str, Any] = {
            "schema": SCHEMA,
            "step": int(step),
            "pair_id": int(pair_id),
            "sample_weight": weights[int(pair_id)],
            "lineages": {},
        }
        for lineage, target in targets.items():
            margin, competitor, argmax = margin_and_competitor(logits, target)
            if lineage == "vehicle_pyav":
                # f16 storage leg: does the arg-max of the STORED logits reproduce the arg-max the
                # trainer computed in f32 before the cast?  Measured, never assumed.
                f16_argmax_mismatch += int((argmax != retained_argmax).sum())
                f16_argmax_pixels += int(argmax.size)
            # The EXACT flip indicator and its edge label both come from the RETAINED argmax, so
            # the per-edge counts sum to the milestone's own recorded d_seg bit-for-bit.  For a
            # flipped pixel the class that actually won IS the competitor of record.
            flip_exact = retained_argmax != target
            competitor_exact = np.where(flip_exact, retained_argmax, competitor).astype(np.uint8)
            binned = accumulate_pair(
                margin, target, competitor_exact, taus, delta_r=delta_r, flip=flip_exact
            )
            bucket = totals[lineage]
            bucket["pixels"] += binned["pixels"]
            for key, vector in binned["surrogate"].items():
                bucket["surrogate"][key] += vector
            for key, vector in binned["grad"].items():
                bucket["grad"][key] += vector
            bucket["capture_pixels"] += np.asarray(binned["margin_capture"]["pixels"])
            bucket["capture_flips"] += np.asarray(binned["margin_capture"]["flips"])
            exact_flips = int((retained_argmax != target).sum())
            row["lineages"][lineage] = {
                "d_seg_exact": exact_flips / float(target.size),
                "exact_flips": exact_flips,
                "binned_flips": binned["flips"],
                "margin_lt_zero": binned["margin_negative"],
                "f16_tie_disagreement_sites": binned["flip_vs_margin_negative_sites"],
                "total_pixels": binned["total_pixels"],
                "annulus_pixels": int(decode_bins(binned["pixels"])[:, :, 1, :].sum()),
                "surrogate_mean": {
                    key: float(vector.sum() / binned["total_pixels"])
                    for key, vector in binned["surrogate"].items()
                },
                "predicted_class_pixels": np.bincount(
                    retained_argmax.reshape(-1), minlength=N_CLASSES
                ).tolist(),
                "target_class_pixels": np.bincount(
                    target.reshape(-1).astype(np.int64), minlength=N_CLASSES
                ).tolist(),
            }
        recorded_row = recorded.get(int(pair_id))
        if recorded_row is not None:
            row["recorded_d_seg"] = float(recorded_row["d_seg"])
            row["recorded_minus_recomputed_vehicle"] = (
                float(recorded_row["d_seg"]) - row["lineages"]["vehicle_pyav"]["d_seg_exact"]
            )
        pair_rows.append(row)
        if rows_handle is not None:
            rows_handle.write(json.dumps(row, sort_keys=True) + "\n")
            rows_handle.flush()

        if retain_dir is not None and order < int(retain_fields_pairs):
            margin_v, competitor_v, argmax_v = margin_and_competitor(logits, targets["vehicle_pyav"])
            fact = atomic_npz(
                Path(retain_dir) / f"step_{int(step):06d}_pair_{int(pair_id):04d}_fields.npz",
                margin_vehicle_f32=margin_v,
                competitor_vehicle_u8=competitor_v,
                realized_argmax_u8=argmax_v,
                retained_argmax_u8=retained_argmax,
                target_vehicle_u8=targets["vehicle_pyav"],
                target_dali_u8=targets["authority_dali"],
            )
            retained_fields.append({"pair_id": int(pair_id), **fact})

    result: dict[str, Any] = {
        "step": int(step),
        "tau_eval": tau_for_milestone(step),
        "taus": [float(t) for t in taus],
        "delta_r": float(delta_r),
        "n_pairs": len(pair_rows),
        "milestone_recorded": {
            "d_seg_hat": milestone["d_seg_hat"],
            "d_pose_hat": milestone["d_pose_hat"],
            "S_hat": milestone["S_hat"],
            "archive_bytes_exact": milestone["archive_bytes_exact"],
            "axis": milestone["axis"],
        },
        "f16_storage_argmax_mismatch_sites": f16_argmax_mismatch,
        "f16_storage_argmax_pixels": f16_argmax_pixels,
        "vehicle_target_matches_gt_cache": vehicle_target_matches_gt_cache,
        "retained_fields": retained_fields,
        "pair_rows": pair_rows,
        "bins": {
            lineage: {
                "pixels": totals[lineage]["pixels"].tolist(),
                "surrogate": {k: v.tolist() for k, v in totals[lineage]["surrogate"].items()},
                "grad": {k: v.tolist() for k, v in totals[lineage]["grad"].items()},
            }
            for lineage in LINEAGES
        },
        "margin_capture": {
            lineage: {
                "multiples": [float(m) for m in MARGIN_CAPTURE_MULTIPLES],
                "pixels": totals[lineage]["capture_pixels"].tolist(),
                "flips": totals[lineage]["capture_flips"].tolist(),
            }
            for lineage in LINEAGES
        },
    }
    result["ht_weighted"] = ht_weighted_d_seg(pair_rows)
    return result


def ht_weighted_d_seg(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Horvitz-Thompson weighted d_seg over the selection, per lineage.

    ``expected_flip_margin_loss`` reduces with ``(per_sample * weights).sum() / weights.sum()``
    (:546-548), so the matching exact aggregate is the same weighted mean of the per-pair d_seg.
    """

    out: dict[str, float] = {}
    weights = np.asarray([float(row["sample_weight"]) for row in pair_rows], dtype=np.float64)
    if weights.size == 0:
        return out
    for lineage in LINEAGES:
        values = np.asarray(
            [float(row["lineages"][lineage]["d_seg_exact"]) for row in pair_rows], dtype=np.float64
        )
        out[lineage] = float((values * weights).sum() / weights.sum())
        out[f"{lineage}_unweighted"] = float(values.mean())
    return out


# ---------------------------------------------------------------------------
# the mis-pricing map
# ---------------------------------------------------------------------------
def edge_table(bins: Mapping[str, Any], tau_key: str) -> list[dict[str, Any]]:
    """One row per (GT class, competitor class) edge with its mis-pricing ratio.

    ``price_ratio = surrogate_mass / exact_flips``.  > 1 means the surrogate spends more mass on the
    edge than the edge actually costs; < 1 means the edge costs more than the surrogate charges.
    """

    pixels = decode_bins(np.asarray(bins["pixels"], dtype=np.float64))
    surrogate = decode_bins(np.asarray(bins["surrogate"][tau_key], dtype=np.float64))
    grad = decode_bins(np.asarray(bins["grad"][tau_key], dtype=np.float64))
    total_pixels = float(pixels.sum())
    rows: list[dict[str, Any]] = []
    for gt in range(N_CLASSES):
        for competitor in range(N_CLASSES):
            if gt == competitor:
                continue
            px = pixels[gt, competitor]
            sm = surrogate[gt, competitor]
            gm = grad[gt, competitor]
            flips = float(px[:, 1].sum())
            surrogate_mass = float(sm.sum())
            rows.append(
                {
                    "gt_class": CLASS_NAMES[gt],
                    "competitor_class": CLASS_NAMES[competitor],
                    "edge": f"{CLASS_NAMES[gt]}->{CLASS_NAMES[competitor]}",
                    "pixels": float(px.sum()),
                    "exact_flips": flips,
                    "surrogate_mass": surrogate_mass,
                    "grad_mass": float(gm.sum()),
                    "exact_share_of_d_seg": flips / total_pixels if total_pixels else 0.0,
                    "surrogate_share": surrogate_mass / total_pixels if total_pixels else 0.0,
                    "price_ratio": (surrogate_mass / flips) if flips > 0 else None,
                    "annulus_pixels": float(px[1, :].sum()),
                    "annulus_flips": float(px[1, 1]),
                    "annulus_surrogate_mass": float(sm[1, :].sum()),
                    "annulus_grad_mass": float(gm[1, :].sum()),
                    "interior_flips": float(px[0, 1]),
                    "interior_surrogate_mass": float(sm[0, :].sum()),
                    "phantom_mass": float(sm[:, 0].sum()),
                    "recovered_mass": float(sm[:, 1].sum()),
                }
            )
    return rows


def global_split(bins: Mapping[str, Any], tau_key: str) -> dict[str, float]:
    """The surrogate's mass split against the exact term, over all edges.

    ``surrogate_total = phantom + recovered`` where ``phantom`` sits on pixels that do NOT flip and
    ``recovered`` sits on pixels that DO.  ``unpriced = exact_flips - recovered`` is the flip mass
    the surrogate declines to charge.

    BOUND, and its one caveat.  The sigmoid crosses 0.5 exactly at ``margin == 0``, so when the flip
    indicator is ``margin < 0`` every flip carries surrogate mass > 0.5 and every non-flip carries
    < 0.5: then ``0.5 * exact_flips <= recovered <= exact_flips`` and ``phantom`` is pure over-charge,
    by construction.  ``measure_milestone`` instead passes the RETAINED-argmax indicator, which
    disagrees with ``margin < 0`` at the float16 tie sites, so on real milestone data the bound holds
    only up to that disagreement rate -- MEASURED at 744 sites over 6 control milestones
    (1.97e-05 of pixels, ~0.7% of the flip count).  Well below every effect reported from this split,
    but it is a caveat, not an identity.
    """

    pixels = decode_bins(np.asarray(bins["pixels"], dtype=np.float64))
    surrogate = decode_bins(np.asarray(bins["surrogate"][tau_key], dtype=np.float64))
    grad = decode_bins(np.asarray(bins["grad"][tau_key], dtype=np.float64))
    total = float(pixels.sum())
    flips = float(pixels[:, :, :, 1].sum())
    phantom = float(surrogate[:, :, :, 0].sum())
    recovered = float(surrogate[:, :, :, 1].sum())
    annulus_pixels = float(pixels[:, :, 1, :].sum())
    annulus_flips = float(pixels[:, :, 1, 1].sum())
    annulus_grad = float(grad[:, :, 1, :].sum())
    return {
        "total_pixels": total,
        "exact_flips": flips,
        "d_seg_unweighted": flips / total if total else 0.0,
        "surrogate_total": phantom + recovered,
        "surrogate_mean": (phantom + recovered) / total if total else 0.0,
        "phantom_mass": phantom,
        "recovered_mass": recovered,
        "unpriced_flip_mass": flips - recovered,
        "phantom_fraction_of_surrogate": phantom / (phantom + recovered) if (phantom + recovered) else 0.0,
        "price_ratio": (phantom + recovered) / flips if flips else None,
        "annulus_pixel_fraction": annulus_pixels / total if total else 0.0,
        "annulus_flip_capture": annulus_flips / flips if flips else 0.0,
        "annulus_grad_fraction": annulus_grad / float(grad.sum()) if float(grad.sum()) else 0.0,
        "grad_total": float(grad.sum()),
    }


def price_ratio_spread(rows: Sequence[Mapping[str, Any]], min_flips: float) -> dict[str, Any]:
    """max/min of ``price_ratio`` across edges carrying at least ``min_flips`` exact flips.

    The charter's falsifier is stated on this spread: < 1.3x means the scalar tau is NOT the defect.
    """

    supported = [
        row
        for row in rows
        if row["price_ratio"] is not None and float(row["exact_flips"]) >= float(min_flips)
    ]
    if len(supported) < 2:
        return {"spread": None, "n_edges": len(supported), "min_flips": float(min_flips)}
    ratios = {row["edge"]: float(row["price_ratio"]) for row in supported}
    lo_edge = min(ratios, key=ratios.get)
    hi_edge = max(ratios, key=ratios.get)
    return {
        "spread": ratios[hi_edge] / ratios[lo_edge],
        "n_edges": len(supported),
        "min_flips": float(min_flips),
        "max_edge": hi_edge,
        "max_ratio": ratios[hi_edge],
        "min_edge": lo_edge,
        "min_ratio": ratios[lo_edge],
        "ratios": ratios,
    }


RARE_CLASSES = ("Lane", "Movable")


def excursion_attribution(
    start_rows: Sequence[Mapping[str, Any]],
    end_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Per-edge delta of the exact term against the surrogate, and the vr1 row coverage fractions.

    Each vr1 candidate cure is scored by the share of |delta exact| it could have priced correctly:

    * row 4 (per-edge tau) applies a POSITIVE per-edge scale ``s_e`` to the surrogate.  A positive
      scale preserves sign, so row 4 can repair an edge only where ``sign(delta surrogate_e) ==
      sign(delta exact_e)``.  Its coverage is the |delta exact| mass on sign-agreeing edges.
    * row 1 (per-pixel margin weight) concentrates mass on small-|margin| pixels.  Its coverage here
      is the |delta exact| mass carried by ``|margin| < delta_R`` pixels -- which is the NARROWEST
      setting of that allocator family and therefore a LOWER BOUND on row 1's reach, not its
      answer.  MEASURED, the bound understates the family by roughly 7x: at a band width of
      25 * delta_R (2.0% of pixels) the same excursion capture is 91.9%, against 8.9% here.  Read
      row 1 off the ``margin_capture`` reach curve; this field alone would mis-rank it.
    * row 3 (one-sided area cap) suppresses rare-class OVER-PAINT.  For a flipped pixel the
      competitor IS the predicted class, so row 3's reach is the |delta exact| mass on edges whose
      COMPETITOR is a rare class.
    The three are overlapping coverage fractions, not a partition, and are reported as such.
    """

    start = {row["edge"]: row for row in start_rows}
    end = {row["edge"]: row for row in end_rows}
    edges = sorted(set(start) & set(end))
    per_edge: list[dict[str, Any]] = []
    for edge in edges:
        a, b = start[edge], end[edge]
        d_exact = float(b["exact_flips"]) - float(a["exact_flips"])
        d_surrogate = float(b["surrogate_mass"]) - float(a["surrogate_mass"])
        d_annulus_exact = float(b["annulus_flips"]) - float(a["annulus_flips"])
        per_edge.append(
            {
                "edge": edge,
                "gt_class": a["gt_class"],
                "competitor_class": a["competitor_class"],
                "delta_exact_flips": d_exact,
                "delta_surrogate_mass": d_surrogate,
                "delta_annulus_exact_flips": d_annulus_exact,
                "delta_interior_exact_flips": (
                    float(b["interior_flips"]) - float(a["interior_flips"])
                ),
                "sign_agrees": bool(np.sign(d_exact) == np.sign(d_surrogate) and d_exact != 0.0),
            }
        )
    mass = sum(abs(row["delta_exact_flips"]) for row in per_edge)
    if mass <= 0.0:
        return {"per_edge": per_edge, "total_abs_delta_exact": mass, "coverage": {}}
    agree = sum(abs(r["delta_exact_flips"]) for r in per_edge if r["sign_agrees"])
    annulus = sum(abs(r["delta_annulus_exact_flips"]) for r in per_edge)
    rare = sum(
        abs(r["delta_exact_flips"]) for r in per_edge if r["competitor_class"] in RARE_CLASSES
    )
    return {
        "per_edge": per_edge,
        "total_abs_delta_exact": mass,
        "net_delta_exact": sum(row["delta_exact_flips"] for row in per_edge),
        "net_delta_surrogate": sum(row["delta_surrogate_mass"] for row in per_edge),
        "coverage": {
            "vr1_row4_per_edge_tau_sign_agreeing_fraction": agree / mass,
            "vr1_row1_margin_weight_annulus_fraction": annulus / mass,
            "vr1_row3_area_cap_rare_competitor_fraction": rare / mass,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", required=True, type=Path, help="QBR1 cell run directory (READ-ONLY)")
    parser.add_argument("--cell-id", required=True, help="label recorded in the output")
    parser.add_argument("--steps", type=int, nargs="+", required=True, help="milestone steps to read")
    parser.add_argument("--store", required=True, type=Path, help="output custody root (never under runs/)")
    parser.add_argument("--limit-pairs", type=int, default=0, help="smoke: read only the first N pairs")
    parser.add_argument("--retain-fields-pairs", type=int, default=4, help="pairs whose fields are persisted per milestone")
    parser.add_argument("--min-flips-for-spread", type=float, default=1000.0, help="edge support floor for the spread falsifier")
    parser.add_argument("--threads", type=int, default=4)
    return parser


def _configure_runtime(threads: int) -> dict[str, Any]:
    """Apply the CPU-only thread discipline and record what actually took effect.

    ``torch.set_num_threads`` is a genuine runtime control and IS applied.  The BLAS environment
    variables are set AFTER numpy has already been imported at module scope, so most backends have
    already sized their pool and will ignore them -- they are best-effort only, and the receipt says
    so rather than implying a guarantee.  The binding limits in practice are torch's thread count,
    ``nice``, and the launcher's ``safe_run`` caps.
    """

    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = str(int(threads))
    try:
        import torch

        torch.set_num_threads(int(threads))
        torch_threads = torch.get_num_threads()
    except Exception:  # pragma: no cover - torch is a hard dep of qbt but never required here
        torch_threads = None
    return {
        "threads_requested": int(threads),
        "torch_threads": torch_threads,
        "blas_env_set_after_numpy_import_best_effort_only": True,
        "nice": os.nice(0),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    runtime = _configure_runtime(args.threads)
    store = Path(args.store)
    store.mkdir(parents=True, exist_ok=True)
    cell_store = store / args.cell_id
    fields_dir = cell_store / "fields"
    cell_store.mkdir(parents=True, exist_ok=True)

    ground_truth = ar1.load_ground_truth()
    pair_ids = list(qbt.SELECTION_IDS)
    if args.limit_pairs:
        pair_ids = pair_ids[: int(args.limit_pairs)]

    milestones: list[dict[str, Any]] = []
    rows_path = cell_store / "pair_rows.jsonl"
    with open(rows_path, "w", encoding="utf-8") as rows_handle:
        for step in args.steps:
            taus = sorted({tau_for_milestone(step), *REFERENCE_TAUS})
            milestones.append(
                measure_milestone(
                    Path(args.run_root),
                    int(step),
                    pair_ids,
                    ground_truth,
                    taus,
                    rows_handle=rows_handle,
                    retain_dir=fields_dir,
                    retain_fields_pairs=args.retain_fields_pairs,
                )
            )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "instrument": INSTRUMENT,
        "axis": AXIS,
        "score_claim": False,
        "cell_id": args.cell_id,
        "run_root": str(Path(args.run_root).resolve()),
        "git_head": ar1.git_head(),
        "runtime": runtime,
        "delta_r": {"value": DELTA_R_N600, "source": DELTA_R_SOURCE},
        "class_names": list(CLASS_NAMES),
        "pair_ids": pair_ids,
        "sample_weights": {str(k): v for k, v in sample_weight_lookup().items() if k in set(pair_ids)},
        "gt_lineage": ground_truth["lineage"],
        "milestones": milestones,
        "analysis": analyse(milestones, min_flips=args.min_flips_for_spread),
        "elapsed_seconds": None,
        "peak_rss_gib": None,
        "rows_jsonl": None,
    }
    report["elapsed_seconds"] = time.time() - started
    report["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)
    report["rows_jsonl"] = ar1.file_fact(rows_path)
    fact = atomic_json(cell_store / "SD1_REPORT.json", report)
    print(json.dumps({"report": fact, "analysis": report["analysis"]["headline"]}, indent=2))
    return report


def analyse(milestones: Sequence[Mapping[str, Any]], *, min_flips: float) -> dict[str, Any]:
    """Build the mis-pricing map and the vr1 ranking from the measured milestones."""

    by_step = {int(m["step"]): m for m in milestones}
    steps = sorted(by_step)
    per_step: dict[str, Any] = {}
    for step in steps:
        milestone = by_step[step]
        entry: dict[str, Any] = {"tau_eval": milestone["tau_eval"], "lineages": {}}
        for lineage in LINEAGES:
            bins = milestone["bins"][lineage]
            tau_keys = sorted(bins["surrogate"])
            own = f"{float(milestone['tau_eval']):.6f}"
            per_tau = {}
            for key in tau_keys:
                rows = edge_table(bins, key)
                per_tau[key] = {
                    "global": global_split(bins, key),
                    "spread": price_ratio_spread(rows, min_flips),
                    "edges": rows,
                }
            entry["lineages"][lineage] = {"tau_eval_key": own, "per_tau": per_tau}
        per_step[str(step)] = entry

    transitions: dict[str, Any] = {}
    for lineage in LINEAGES:
        for start, end in itertools.pairwise(steps):
            key_start = f"{float(by_step[start]['tau_eval']):.6f}"
            # The excursion is compared at a FIXED tau so the anneal cannot manufacture a delta.
            fixed = f"{0.05:.6f}"
            transitions[f"{lineage}:{start}->{end}:tau_own"] = excursion_attribution(
                per_step[str(start)]["lineages"][lineage]["per_tau"][key_start]["edges"],
                per_step[str(end)]["lineages"][lineage]["per_tau"][
                    f"{float(by_step[end]['tau_eval']):.6f}"
                ]["edges"],
            )
            transitions[f"{lineage}:{start}->{end}:tau_0.05"] = excursion_attribution(
                per_step[str(start)]["lineages"][lineage]["per_tau"][fixed]["edges"],
                per_step[str(end)]["lineages"][lineage]["per_tau"][fixed]["edges"],
            )

    headline: dict[str, Any] = {}
    if steps:
        first = per_step[str(steps[0])]["lineages"]["authority_dali"]
        key = first["tau_eval_key"]
        headline["first_step_global_dali"] = first["per_tau"][key]["global"]
        headline["first_step_spread_dali"] = first["per_tau"][key]["spread"]
    return {"per_step": per_step, "transitions": transitions, "headline": headline}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
