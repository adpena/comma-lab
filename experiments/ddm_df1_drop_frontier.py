#!/usr/bin/env python3
"""ddm_df1 stage 2 -- the ``dD/dB`` field's shape, and the frontier it implies.

Consumes the fields ``ddm_df1_drop_field.py`` retained (coding-row argmax, pmax,
psecond, selected cost) plus TB2/WJ1/BL1's retained join fields, and answers the
three questions the charter asks, in order:

  * ``field``    -- full-population, scorer-free.  The exact zero/positive partition
                    of ``dD``, its byte mass, the concentration statistics, the joins
                    against bit mass and against WJ1's manufactured-error support,
                    and the addressless threshold frontier in STATIC bytes.
  * ``scorer``   -- the forward-model control (must reproduce the pointer's seg leg)
                    and the REAL ``d_seg`` of the dropped field at each threshold,
                    through the shipped ``SemanticTokenRenderer`` and frozen SegNet.
                    This is where ``dD`` stops being a label count and becomes S.
  * ``reencode`` -- REAL archive bytes for the drop, through the shipped RC64
                    encoder.  ``ddm_fs2`` measured that ``-log2 p`` misprices token
                    moves by up to 11x, so the static byte column is never the claim.

THE OPERATOR, RESTATED
----------------------
``drop(tau)``: the receiver computes its coding row at every position, as it already
does.  Where ``pmax >= tau`` no token is coded and the receiver substitutes its own
argmax.  Deterministic on both sides, so it costs ZERO address bits -- which matters
because ``ddm_tba1`` measured that naming any subset of this object costs more than
the subset holds.

THE IDENTITY THAT MAKES HALF THE FIELD EXACT WITHOUT A PROBE
------------------------------------------------------------
``cost_i < 1 bit`` <=> ``p_sel_i > 0.5`` <=> ``p_sel_i`` is the unique row maximum
<=> the receiver's argmax already equals the transmitted symbol <=> dropping position
``i`` leaves the decoded field, the render, the SegNet argmax, AND PoseNet's input
bit-identical <=> ``dD_i = 0`` exactly.  No scorer can improve on that; it is
arithmetic on a probability row.  The scorer is needed only for the positive mode.

Axis: ``[macOS-CPU advisory]`` throughout.  ``score_claim=false``.  No archive is
promoted, no Modal job is fired, the pointer does not move.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

APDATA = Path("/Volumes/APDataStore/pact")
VERTIGO = Path("/Volumes/VertigoDataTier/pact")
STORE = APDATA / "ddm_df1_dddb_field" / "measurement_v1"
FIELDS = STORE / "retained" / "fields"
TB2 = APDATA / "ddm_tb2_token_bit_attribution" / "measurement_v1" / "retained"
TO2_INPUT = VERTIGO / "ddm_to2_token_ordering_race" / "measurement_v1" / "retained" / "input"

DX2_RUNTIME_ROOT = APDATA / "ddm_dx2" / "r7" / "candidate_runtime_dx2"
DX2_ARCHIVE = DX2_RUNTIME_ROOT / "archive.zip"
DX2_TOKENS = TO2_INPUT / "dx2_tokens_decoded.u8"
GT_DALI = VERTIGO / "ddm_qs3_20260813" / "retained" / "inputs" / "gt_argmax_n600.npy"

N, HEIGHT, WIDTH = 600, 384, 512
PLANE = HEIGHT * WIDTH
POSITIONS = N * PLANE
CLASSES = 5
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")

#: ddm_tx1 sec.0 -- CITED, never re-derived.
S_PER_ARCHIVE_BYTE = 6.658590e-07
#: 100 / 117,964,800 -- one scored argmax cell in S units (ddm_jg1 S_PER_SEG_CELL).
S_PER_SEG_CELL = 100.0 / POSITIONS
#: What one repaired/broken scored cell is worth in bytes.
BYTES_PER_SEG_CELL = S_PER_SEG_CELL / S_PER_ARCHIVE_BYTE

#: The live pointer, recomputed from components -- never the rounded display.
#: ``experiments/results/modal_auth_eval_mirror/contest_auth_eval_ddm_dx2_fx5_cabac_t4_r2.json``
#: carries avg_segnet_dist 0.00020139 and avg_posenet_dist 6.37e-06; those two plus
#: 25*180368/37545489 = 0.1200996476567398 reconstruct S to the last digit.
DX2_ARCHIVE_BYTES = 180_368
DX2_S = 0.14821987563243377
TARGET_S = 0.12
#: THIS body's own contest-CUDA seg leg -- NOT ddm_up3's 0.00030309, which belongs to
#: a different archive (176,420 B).  Quoting up3's number here would gate the
#: instrument against the wrong object.
DX2_D_SEG_T4 = 0.00020139
#: +/-10%.  ddm_jg1 measured its local DALI-lineage instrument at 0.99995x of the T4
#: seg leg, so a 10% band is loose enough to absorb render/host drift and tight enough
#: to refuse an instrument that is looking at a different body.
POINTER_D_SEG_DALI_GATE = (DX2_D_SEG_T4 * 0.90, DX2_D_SEG_T4 * 1.10)

CHUNK = 4_000_000


class Df1Error(RuntimeError):
    """Fail-closed refusal."""


# ----------------------------------------------------------------------------------
# custody
# ----------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def open_fields() -> dict[str, np.memmap]:
    return {
        "cost": np.memmap(FIELDS / "position_rc64_frequency_cost_bits.f64le.bin", dtype="<f8", mode="r", shape=(POSITIONS,)),
        "pmax": np.memmap(FIELDS / "position_coding_pmax.f32le.bin", dtype="<f4", mode="r", shape=(POSITIONS,)),
        "psecond": np.memmap(FIELDS / "position_coding_psecond.f32le.bin", dtype="<f4", mode="r", shape=(POSITIONS,)),
        "argmax": np.memmap(FIELDS / "position_coding_argmax.u8.bin", dtype=np.uint8, mode="r", shape=(POSITIONS,)),
        "decoded": np.memmap(DX2_TOKENS, dtype=np.uint8, mode="r", shape=(POSITIONS,)),
    }


def packbits_mask(path: Path) -> np.ndarray:
    """Unpack an ``(N, H, W)`` little-endian packbits mask to a flat bool array."""
    packed = np.fromfile(path, dtype=np.uint8)
    bits = np.unpackbits(packed.reshape(N, -1), axis=1, bitorder="little", count=PLANE)
    return bits.reshape(-1).astype(bool)


# ----------------------------------------------------------------------------------
# stage: field
# ----------------------------------------------------------------------------------


def weighted_gini(values: np.ndarray) -> float:
    """Gini of a non-negative mass distribution over positions."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    total = float(ordered.sum())
    if total <= 0.0:
        return 0.0
    n = ordered.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (index * ordered).sum()) / (n * total) - (n + 1.0) / n)


def stage_field(out: Path) -> dict[str, Any]:
    fields = open_fields()
    cost, pmax, argmax, decoded = fields["cost"], fields["pmax"], fields["argmax"], fields["decoded"]
    gt = np.load(GT_DALI, mmap_mode="r").reshape(-1)

    manufactured = packbits_mask(TB2 / "join_fields" / "gross_manufactured_native_render_head.n600.packbits")
    top1 = packbits_mask(TB2 / "join_fields" / "top_1pct.n600.packbits")
    top10 = packbits_mask(TB2 / "join_fields" / "top_10pct.n600.packbits")

    total_bits = 0.0
    free_n = 0
    free_bits = 0.0
    flip_n = 0
    flip_bits = 0.0
    sub1_n = 0
    sub1_bits = 0.0
    # joins, on the two exact modes
    joins = {
        name: {"mask_n": 0, "flip_n": 0, "flip_bits": 0.0, "free_bits": 0.0, "mask_bits": 0.0}
        for name in ("manufactured", "top1", "top10")
    }
    per_class = {
        c: {"n": 0, "bits": 0.0, "flip_n": 0, "flip_bits": 0.0, "free_bits": 0.0}
        for c in range(CLASSES)
    }
    # flip pmax distribution -> the addressless threshold frontier
    flip_pmax_hist_edges = np.array(
        [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 1.0 + 1e-6]
    )
    flip_pmax_n = np.zeros(len(flip_pmax_hist_edges) - 1, dtype=np.int64)
    flip_pmax_bits = np.zeros(len(flip_pmax_hist_edges) - 1, dtype=np.float64)
    free_pmax_n = np.zeros(len(flip_pmax_hist_edges) - 1, dtype=np.int64)
    free_pmax_bits = np.zeros(len(flip_pmax_hist_edges) - 1, dtype=np.float64)
    max_flip_pmax = 0.0
    # float32 saturation: pmax == 1.0 exactly.  If ANY flip sits there, no addressless
    # threshold can separate the zero mode from the positive mode -- the decisive test
    # of whether the free bytes are reachable without an address.
    saturated_flip_n = 0
    saturated_flip_bits = 0.0
    saturated_free_n = 0
    saturated_free_bits = 0.0
    # cost distribution split by mode -> AUC, the rank correlation between dD>0 and dB
    auc_edges = np.concatenate(([0.0], np.logspace(-9, 5, 141), [np.inf]))
    cost_hist_flip = np.zeros(len(auc_edges) - 1, dtype=np.int64)
    cost_hist_free = np.zeros(len(auc_edges) - 1, dtype=np.int64)
    # THE ADDRESS BOUND.  The zero mode is not a function of the coding row -- it
    # depends on the symbol the receiver has not decoded yet -- so isolating it needs
    # a side channel carrying the indicator "is my argmax right?".  Under the shipped
    # model that indicator has probability pmax, so its entropy is H_b(pmax) and
    # sum_i H_b(pmax_i) is the information-theoretic FLOOR on any such channel.
    # Because H_b(p) >= -p*log2(p) pointwise, that floor is >= the zero mode's own
    # expected byte mass -- which turns ddm_tba1's empirical "naming any subset costs
    # more than the subset holds" into a statement this object satisfies by
    # construction.  Measured here rather than asserted.
    address_bound_bits = 0.0

    for s in range(0, POSITIONS, CHUNK):
        e = min(s + CHUNK, POSITIONS)
        c = np.asarray(cost[s:e], dtype=np.float64)
        a = np.asarray(argmax[s:e])
        d = np.asarray(decoded[s:e])
        p = np.asarray(pmax[s:e], dtype=np.float64)
        g = np.asarray(gt[s:e])
        flip = a != d
        free = ~flip
        total_bits += float(c.sum())
        free_n += int(free.sum())
        free_bits += float(c[free].sum())
        flip_n += int(flip.sum())
        flip_bits += float(c[flip].sum())
        sub1 = c < 1.0
        sub1_n += int(sub1.sum())
        sub1_bits += float(c[sub1].sum())
        if np.any(sub1 & flip):
            raise Df1Error("cost<1 bit position is a flip: the p_sel>0.5 identity is violated")
        if flip.any():
            max_flip_pmax = max(max_flip_pmax, float(p[flip].max()))
        saturated = p >= 1.0
        saturated_flip_n += int((saturated & flip).sum())
        saturated_flip_bits += float(c[saturated & flip].sum())
        saturated_free_n += int((saturated & free).sum())
        saturated_free_bits += float(c[saturated & free].sum())
        q = np.clip(p, 1e-12, 1.0 - 1e-12)
        address_bound_bits += float(
            np.where(p >= 1.0, 0.0, -q * np.log2(q) - (1.0 - q) * np.log2(1.0 - q)).sum()
        )
        cidx = np.clip(np.searchsorted(auc_edges, c, side="right") - 1, 0, len(cost_hist_flip) - 1)
        cost_hist_flip += np.bincount(cidx[flip], minlength=len(cost_hist_flip))
        cost_hist_free += np.bincount(cidx[free], minlength=len(cost_hist_free))
        idx = np.clip(np.searchsorted(flip_pmax_hist_edges, p, side="right") - 1, 0, len(flip_pmax_n) - 1)
        flip_pmax_n += np.bincount(idx[flip], minlength=len(flip_pmax_n))
        flip_pmax_bits += np.bincount(idx[flip], weights=c[flip], minlength=len(flip_pmax_n))
        free_pmax_n += np.bincount(idx[free], minlength=len(free_pmax_n))
        free_pmax_bits += np.bincount(idx[free], weights=c[free], minlength=len(free_pmax_n))
        for name, mask in (("manufactured", manufactured), ("top1", top1), ("top10", top10)):
            m = mask[s:e]
            joins[name]["mask_n"] += int(m.sum())
            joins[name]["mask_bits"] += float(c[m].sum())
            joins[name]["flip_n"] += int((m & flip).sum())
            joins[name]["flip_bits"] += float(c[m & flip].sum())
            joins[name]["free_bits"] += float(c[m & free].sum())
        for cls in range(CLASSES):
            sel = g == cls
            per_class[cls]["n"] += int(sel.sum())
            per_class[cls]["bits"] += float(c[sel].sum())
            per_class[cls]["flip_n"] += int((sel & flip).sum())
            per_class[cls]["flip_bits"] += float(c[sel & flip].sum())
            per_class[cls]["free_bits"] += float(c[sel & free].sum())

    # Gini on the exact dB field (identity check against TB2's 0.9951593787014772).
    gini_cost = weighted_gini(np.asarray(cost))

    # Gini of the dD/dB field itself.  dD is EXACTLY zero on the free mode, so this
    # statistic is near-degenerate by construction; it is reported because the charter
    # registered a threshold on it, and it is read beside the informative statistic
    # (the dB-mass share of the zero mode), never instead of it.
    flip_flags = np.empty(POSITIONS, dtype=bool)
    for s in range(0, POSITIONS, CHUNK):
        e = min(s + CHUNK, POSITIONS)
        flip_flags[s:e] = np.asarray(argmax[s:e]) != np.asarray(decoded[s:e])

    def enrich(cell_n: int, a_n: int, b_n: int) -> float:
        expected = a_n * b_n / POSITIONS
        return float(cell_n / expected) if expected > 0 else float("inf")

    # AUC = P(cost of a random positive-mode position > cost of a random zero-mode
    # position) + 0.5 P(tie), computed from the shared binned cost distribution.
    # 0.5 means dD>0 carries no information about dB; 1.0 means perfect co-location.
    flip_total = float(cost_hist_flip.sum())
    free_total = float(cost_hist_free.sum())
    free_below = np.concatenate(([0.0], np.cumsum(cost_hist_free.astype(np.float64))[:-1]))
    auc = float(
        (
            (cost_hist_flip.astype(np.float64) * free_below).sum()
            + 0.5 * (cost_hist_flip.astype(np.float64) * cost_hist_free.astype(np.float64)).sum()
        )
        / (flip_total * free_total)
    ) if flip_total and free_total else 0.5

    frontier = []
    cum_flip_n = 0
    cum_flip_bits = 0.0
    cum_free_bits = 0.0
    for i in range(len(flip_pmax_n) - 1, -1, -1):
        cum_flip_n += int(flip_pmax_n[i])
        cum_flip_bits += float(flip_pmax_bits[i])
        cum_free_bits += float(free_pmax_bits[i])
        saved_bytes = (cum_flip_bits + cum_free_bits) / 8.0
        frontier.append(
            {
                "tau_lower_edge": float(flip_pmax_hist_edges[i]),
                "positions_dropped": int(sum(flip_pmax_n[i:]) + sum(free_pmax_n[i:])),
                "labels_broken": cum_flip_n,
                "static_bytes_saved": saved_bytes,
                "static_S_rate_gain": saved_bytes * S_PER_ARCHIVE_BYTE,
                "S_seg_loss_if_amplification_1x": cum_flip_n * S_PER_SEG_CELL,
                "net_S_if_amplification_1x": cum_flip_n * S_PER_SEG_CELL - saved_bytes * S_PER_ARCHIVE_BYTE,
            }
        )
    frontier.reverse()

    # THE DEMAND HAS TWO READINGS AND THEY DIFFER BY 283x.  Shedding bytes at FIXED
    # distortion must pay for the whole distortion leg as well, so the target byte
    # count is (TARGET_S - distortion_leg)/S_per_byte, NOT TARGET_S/S_per_byte.  The
    # latter is the ZERO-distortion ceiling -- only 149.65 B below the shipped size --
    # and quoting it as "the demand" would overstate every share by ~283x.
    rate_leg = DX2_ARCHIVE_BYTES * 25.0 / 37_545_489.0
    distortion_leg = DX2_S - rate_leg
    demand_bytes = DX2_ARCHIVE_BYTES - (TARGET_S - distortion_leg) * 37_545_489.0 / 25.0
    zero_distortion_shed_bytes = DX2_ARCHIVE_BYTES - TARGET_S * 37_545_489.0 / 25.0

    result = {
        "schema": "ddm_df1_field.v1",
        "axis": "[macOS-CPU advisory / scorer-free shipped-receiver instrumentation]",
        "score_claim": False,
        "positions": POSITIONS,
        "exchange": {
            "S_per_archive_byte": S_PER_ARCHIVE_BYTE,
            "S_per_seg_cell": S_PER_SEG_CELL,
            "bytes_per_seg_cell": BYTES_PER_SEG_CELL,
            "rate_leg": rate_leg,
            "distortion_leg": distortion_leg,
            "demand_bytes_at_fixed_distortion": demand_bytes,
            "zero_distortion_shed_bytes": zero_distortion_shed_bytes,
            "cited": "ddm_tx1_toolbox_crosswalk_20260819.md sec.0",
        },
        "total_selected_bits": total_bits,
        "total_selected_bytes": total_bits / 8.0,
        "gini_of_dB": gini_cost,
        "tb2_declared_gini_of_dB": 0.9951593787014772,
        "zero_mode": {
            "definition": "coding-row argmax == transmitted symbol => dD == 0 exactly",
            "positions": free_n,
            "position_share": free_n / POSITIONS,
            "bits": free_bits,
            "bytes": free_bits / 8.0,
            "bit_share": free_bits / total_bits,
            "share_of_demand": (free_bits / 8.0) / demand_bytes,
            "S_if_fully_removed": (free_bits / 8.0) * S_PER_ARCHIVE_BYTE,
        },
        "positive_mode": {
            "positions": flip_n,
            "position_share": flip_n / POSITIONS,
            "bits": flip_bits,
            "bytes": flip_bits / 8.0,
            "bit_share": flip_bits / total_bits,
        },
        "sub_one_bit_identity": {
            "positions": sub1_n,
            "bits": sub1_bits,
            "bytes": sub1_bits / 8.0,
            "note": "cost<1 bit is a strict subset of the zero mode; verified per chunk",
        },
        "max_pmax_among_flips": max_flip_pmax,
        "colocation": {
            "auc_dD_positive_vs_dB": auc,
            "mean_bits_positive_mode": flip_bits / flip_n if flip_n else 0.0,
            "mean_bits_zero_mode": free_bits / free_n if free_n else 0.0,
            "mean_bits_ratio_positive_over_zero": (
                (flip_bits / flip_n) / (free_bits / free_n) if flip_n and free_n and free_bits else float("inf")
            ),
        },
        "address_bound": {
            "definition": "sum_i H_b(pmax_i), the entropy of the indicator 'is the "
            "receiver's argmax correct' under the shipped model; the floor on ANY "
            "side channel that isolates the zero mode",
            "bits": address_bound_bits,
            "bytes": address_bound_bits / 8.0,
            "zero_mode_bytes": free_bits / 8.0,
            "address_cost_over_zero_mode_yield": (
                (address_bound_bits / 8.0) / (free_bits / 8.0) if free_bits else float("inf")
            ),
            "note": "float32-saturated rows (pmax >= 1.0) contribute 0 to this floor "
            "because the model asserts certainty; they are counted honestly, which "
            "makes the bound CONSERVATIVE (a true model would charge more)",
        },
        "float32_saturated_pmax": {
            "note": "pmax == 1.0 exactly in float32; an addressless threshold cannot "
            "separate the two modes inside this cell",
            "flip_positions": saturated_flip_n,
            "flip_bytes": saturated_flip_bits / 8.0,
            "free_positions": saturated_free_n,
            "free_bytes": saturated_free_bits / 8.0,
        },
        "joins": {
            name: {
                **row,
                "mask_bytes": row["mask_bits"] / 8.0,
                "flip_bytes": row["flip_bits"] / 8.0,
                "free_bytes": row["free_bits"] / 8.0,
                "flip_count_enrichment_vs_independence": enrich(row["flip_n"], row["mask_n"], flip_n),
                "share_of_mask_bits_in_positive_mode": row["flip_bits"] / row["mask_bits"] if row["mask_bits"] else 0.0,
            }
            for name, row in joins.items()
        },
        "per_class": {
            CLASS_NAMES[c]: {
                **row,
                "bytes": row["bits"] / 8.0,
                "flip_bytes": row["flip_bits"] / 8.0,
                "free_bytes": row["free_bits"] / 8.0,
                "flip_rate": row["flip_n"] / row["n"] if row["n"] else 0.0,
                "flip_enrichment_vs_independence": enrich(row["flip_n"], row["n"], flip_n),
            }
            for c, row in per_class.items()
        },
        "pmax_bands": [
            {
                "lo": float(flip_pmax_hist_edges[i]),
                "hi": float(flip_pmax_hist_edges[i + 1]),
                "flip_positions": int(flip_pmax_n[i]),
                "flip_bytes": float(flip_pmax_bits[i]) / 8.0,
                "free_positions": int(free_pmax_n[i]),
                "free_bytes": float(free_pmax_bits[i]) / 8.0,
            }
            for i in range(len(flip_pmax_n))
        ],
        "addressless_frontier_static": frontier,
    }
    atomic_json(out / "FIELD.json", result)
    np.save(out / "flip_flags.npy", np.packbits(flip_flags, bitorder="little"))
    return result


# ----------------------------------------------------------------------------------
# stage: scorer
# ----------------------------------------------------------------------------------


def build_dropped_tokens(tau: float) -> np.ndarray:
    """The decoded field the receiver reconstructs under ``drop(tau)``.

    STATIC: the argmax substituted here is the one the SHIPPED trajectory produced.
    For ``tau > max_pmax_among_flips`` the field is bit-identical to the shipped one,
    so the substitution is exact.  Below that the real receiver's contexts diverge
    after the first substituted label; that approximation is measured, not assumed --
    the ``reencode`` stage re-runs the true trajectory.
    """
    fields = open_fields()
    out = np.array(fields["decoded"], dtype=np.uint8)
    for s in range(0, POSITIONS, CHUNK):
        e = min(s + CHUNK, POSITIONS)
        take = np.asarray(fields["pmax"][s:e], dtype=np.float64) >= tau
        block = out[s:e]
        block[take] = np.asarray(fields["argmax"][s:e])[take]
        out[s:e] = block
    return out.reshape(N, HEIGHT, WIDTH)


def stage_scorer(
    out: Path,
    taus: list[float],
    threads: int,
    token_files: list[Path] | None = None,
    label: str = "SCORER",
) -> dict[str, Any]:
    """Real scorer-space ``dD`` for each dropped field, against one shared control.

    ``taus`` scores STATIC fields built from the shipped trajectory's argmax.
    ``token_files`` scores fields somebody else produced -- in practice the TRUE
    closed-loop reconstruction the ``reencode`` stage writes, which is the same
    operator without the static approximation.  Both go through one control in one
    process so every delta is a matched-instrument difference; ddm_rf1 sec.4 caught a
    published ratio that divided a macOS-CPU numerator by a contest-CUDA denominator,
    and this is the structural cure for that class.
    """
    import torch

    torch.set_num_threads(threads)
    import ddm_jg1_seg_solve as jg1

    semantic = jg1.load_semantic_renderer(
        archive_path=DX2_ARCHIVE, runtime_dir=DX2_RUNTIME_ROOT / "runtime"
    )
    net = jg1.load_segnet()
    gt = np.load(GT_DALI, mmap_mode="r")
    indices = np.arange(N)

    def evaluate(tokens: np.ndarray, tag: str) -> dict[str, Any]:
        started = time.perf_counter()
        argmax = np.empty((N, HEIGHT, WIDTH), dtype=np.uint8)
        for start in range(0, N, 20):
            block = indices[start : start + 20]
            frames = jg1.render_frame1(semantic, tokens[block], block)
            argmax[block] = jg1.argmax_from_camera_frames(net, frames)
        per_pair = jg1.d_seg_per_pair(argmax, np.asarray(gt))
        return {
            "tag": tag,
            "d_seg": float(per_pair.mean()),
            "cells_disagreeing": int((argmax != np.asarray(gt)).sum()),
            "elapsed_seconds": time.perf_counter() - started,
            "argmax": argmax,
        }

    shipped_tokens = np.array(
        np.memmap(DX2_TOKENS, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    )
    control = evaluate(shipped_tokens, "forward_model_control")
    base_argmax = control.pop("argmax")
    if not (POINTER_D_SEG_DALI_GATE[0] <= control["d_seg"] <= POINTER_D_SEG_DALI_GATE[1]):
        raise Df1Error(
            f"forward-model control d_seg={control['d_seg']} is outside the pointer "
            f"gate {POINTER_D_SEG_DALI_GATE}; the instrument is not measuring this body"
        )
    print(json.dumps(control, sort_keys=True), flush=True)

    rows = []

    def score_field(tokens: np.ndarray, tag: str, extra: dict[str, Any]) -> None:
        labels_changed = int((tokens != shipped_tokens).sum())
        row = evaluate(tokens, tag)
        argmax = row.pop("argmax")
        cells_changed = int((argmax != base_argmax).sum())
        row.update(
            {
                "token_labels_changed": labels_changed,
                "scored_cells_changed_vs_shipped": cells_changed,
                "render_amplification_cells_per_label": (
                    cells_changed / labels_changed if labels_changed else 0.0
                ),
                "delta_d_seg": row["d_seg"] - control["d_seg"],
                "delta_S_seg": 100.0 * (row["d_seg"] - control["d_seg"]),
                **extra,
            }
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        atomic_json(out / f"{label}.json", {"control": control, "rows": rows})

    for tau in taus:
        score_field(build_dropped_tokens(tau), f"drop_tau_{tau}", {"tau": tau, "field": "static"})
    for path in token_files or []:
        tokens = np.fromfile(path, dtype=np.uint8)
        if tokens.size != POSITIONS:
            raise Df1Error(f"{path} holds {tokens.size} tokens, expected {POSITIONS}")
        score_field(
            tokens.reshape(N, HEIGHT, WIDTH),
            f"field_{path.stem}",
            {"field": "true_closed_loop", "source": file_fact(path)},
        )

    result = {
        "schema": "ddm_df1_scorer.v2",
        "axis": "[macOS-CPU advisory / DALI-lineage GT]",
        "score_claim": False,
        "control": control,
        "rows": rows,
    }
    atomic_json(out / f"{label}.json", result)
    return result


# ----------------------------------------------------------------------------------
# stage: reencode -- the operator run for real, encoder and receiver in lockstep
# ----------------------------------------------------------------------------------


def stage_reencode(out: Path, tau: float, threads: int) -> dict[str, Any]:
    """REAL archive bytes and the TRUE reconstructed field for ``drop(tau)``.

    The static token field the ``scorer`` stage builds substitutes the argmax the
    SHIPPED trajectory produced.  That is exact only while no label actually changes;
    once one does, the real receiver's contexts diverge and every later coding row
    moves.  This stage removes the approximation: it recomputes the coding row from
    the live trajectory, takes THAT row's argmax, encodes only the rows the rule
    still sends, and feeds the reconstructed symbol -- not the transmitted one --
    back into the corrector and the model context.  Encoder and receiver therefore
    stay in lockstep, which is what makes the emitted stream a real archive section
    rather than an accounting estimate.

    The loop is forked from ``ddm_jg2_tail_reencode.encode_tail`` rather than called,
    because the fork is exactly one line deep (encode a SUBSET of each group's rows)
    and jg2's contract is whole-field encoding.  Everything model-side -- the HPAC
    fallback, the group plan, the boundary buckets, the fixed table, the
    ``FreeCorrector``, and the probability quantization -- is jg2's own object.
    ``--tau 2.0`` sends every row and MUST reproduce the shipped stream byte for
    byte; that is this stage's self-test, not a formality.
    """
    import torch

    torch.set_num_threads(threads)
    import ddm_jg2_tail_reencode as jg2

    work = out / "reencode_work"
    work.mkdir(parents=True, exist_ok=True)
    route_b = jg2.load_route_b()
    tag = f"drop_{tau:g}".replace(".", "p")
    library, build = jg2.compile_rc64(work, route_b, tag)
    residual, renderer, renderer_dir = jg2.load_runtime(DX2_RUNTIME_ROOT)

    # These live inside the shipped runtime package, so they are importable only
    # AFTER load_runtime has put that package on sys.path.
    from runtime.free_corrector import FreeCorrector  # type: ignore[import-not-found]
    from runtime.hpac_inference import (  # type: ignore[import-not-found]
        optimize_sparse_evaluator,
    )

    parts = residual.read_residual_archive(DX2_ARCHIVE)
    shipped_stream = parts.token_stream

    target = np.array(
        np.memmap(DX2_TOKENS, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    )
    device = torch.device("cpu")
    base_hpac = residual.materialize_ihs1(parts.hpac_blob, renderer)
    model = renderer.load_hpac(base_hpac, device)
    masks = renderer.group_masks(device)
    sparse = residual._sparse_class(renderer_dir)(model, HEIGHT, WIDTH)
    corrector = FreeCorrector(PLANE)
    plans = []
    for mask in masks:
        flat = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1)).astype(np.int64)
        plans.append((torch.from_numpy(flat).to(device), flat))

    encoder = route_b.NativeRc64Encoder(library)
    reconstructed = np.empty((N, HEIGHT, WIDTH), dtype=np.uint8)
    sent = 0
    skipped = 0
    substituted = 0
    started = time.perf_counter()
    with torch.inference_mode():
        optimize_sparse_evaluator(sparse)
        previous = torch.zeros((1, HEIGHT, WIDTH), dtype=torch.long, device=device)
        for frame in range(N):
            index = torch.tensor([frame], dtype=torch.long, device=device)
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(index, previous)
            if frame:
                previous_cpu = previous[0].to(device="cpu", dtype=torch.uint8).numpy()
                boundary = residual._boundary_buckets(previous_cpu).reshape(-1)
            else:
                boundary = np.full(PLANE, 4, dtype=np.uint8)
            corrector.begin_frame(boundary)
            plane_target = np.asarray(target[frame], dtype=np.uint8).reshape(-1)
            for group, (device_positions, flat_positions) in enumerate(plans):
                base_logits = sparse.selected_logits(current, context, group).cpu().numpy()
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                feature = boundary[flat_positions].astype(np.int64) * CLASSES + predicted
                corrected = base_logits + parts.table.values[feature]
                probability = residual._probability_table(corrected, renderer.HPAC_LOGIT_PRECISION)
                state = corrector.group_state(probability, predicted, flat_positions)
                coding = np.asarray(corrector.coding_row(state), dtype=np.float64)
                row32 = np.ascontiguousarray(coding, dtype=np.float32)
                winner = row32.argmax(axis=1)
                row_pmax = row32[np.arange(row32.shape[0]), winner].astype(np.float64)
                send = row_pmax < tau
                truth = plane_target[flat_positions].astype(np.int64)
                symbols = np.where(send, truth, winner.astype(np.int64))
                if send.any():
                    encoder.encode(truth[send].astype(np.int32), coding[send])
                sent += int(send.sum())
                skipped += int((~send).sum())
                substituted += int((symbols != truth).sum())
                corrector.observe(state, symbols)
                current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(device)
            frame_tokens = current[0].to(device="cpu", dtype=torch.uint8).numpy()
            reconstructed[frame] = frame_tokens
            corrector.end_frame(frame_tokens.reshape(-1))
            previous = current
            if (frame + 1) % 50 == 0:
                print(
                    json.dumps(
                        {
                            "stage": f"reencode_{tag}",
                            "frame": frame + 1,
                            "sent": sent,
                            "skipped": skipped,
                            "substituted": substituted,
                            "elapsed_s": round(time.perf_counter() - started, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    payload = encoder.finish()
    if not payload.startswith(route_b.TOKEN_MAGIC):
        raise Df1Error("RC64 payload lost its magic")
    stream_path = work / f"tail_{tag}.bin"
    stream_path.write_bytes(payload)
    field_path = work / f"reconstructed_tokens_{tag}.u8"
    reconstructed.tofile(field_path)

    # FRAMING, MEASURED not assumed.  ``finish()`` returns TOKEN_MAGIC + body, and the
    # range coder's terminal flush emits whole bytes, so the emitted payload is longer
    # than the section the archive stores even when the coded content is identical.
    # The control run (--tau 2.0, nothing skipped) measured exactly this: emitted
    # 113,784 B = 4 magic + the shipped 113,777 B BYTE FOR BYTE + 3 flush zeros.
    # Therefore a drop's saving is taken against the CONTROL's emitted length, where
    # the framing cancels -- never against the archive's stored length, which would
    # charge the drop for 7 bytes of framing it did not cause.
    body = payload[len(route_b.TOKEN_MAGIC) :]
    prefix = body[: len(shipped_stream)]
    result = {
        "schema": "ddm_df1_reencode.v2",
        "axis": "[macOS-CPU advisory / shipped RC64 encoder]",
        "score_claim": False,
        "tau": tau,
        "rows_sent": sent,
        "rows_skipped": skipped,
        "labels_substituted": substituted,
        "shipped_stream_bytes": len(shipped_stream),
        "emitted_payload_bytes": len(payload),
        "emitted_body_bytes": len(body),
        "magic_bytes": len(route_b.TOKEN_MAGIC),
        "flush_padding_bytes": len(body) - len(shipped_stream) if len(body) >= len(shipped_stream) else None,
        "body_prefix_reproduces_shipped_stream": prefix == shipped_stream,
        "byte_identical_to_shipped_payload": payload == shipped_stream,
        "reconstructed_field_differs_from_shipped": int(
            (reconstructed != target).sum()
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "build": build,
        "stream": file_fact(stream_path),
        "reconstructed_field": file_fact(field_path),
    }
    atomic_json(out / f"REENCODE_{tag}.json", result)
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("field", "scorer", "reencode"), required=True)
    parser.add_argument("--out", type=Path, default=STORE / "analysis")
    parser.add_argument("--taus", type=float, nargs="*", default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--token-files", type=Path, nargs="*", default=None)
    parser.add_argument("--label", default="SCORER")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(APDATA.resolve()):
        raise Df1Error(f"output must remain on the APDataStore SSD tier: {APDATA}")
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "field":
        result = stage_field(out)
        print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}, indent=2, sort_keys=True))
    elif args.stage == "scorer":
        if not args.taus and not args.token_files:
            raise Df1Error("the scorer stage needs --taus and/or --token-files")
        stage_scorer(
            out,
            list(args.taus or []),
            args.threads,
            token_files=list(args.token_files or []),
            label=args.label,
        )
    else:
        if args.tau is None:
            raise Df1Error("--tau is required for the reencode stage")
        stage_reencode(out, args.tau, args.threads)


if __name__ == "__main__":
    main()
