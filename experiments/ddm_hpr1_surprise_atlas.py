"""ddm_hpr1 -- conditional-information atlas of the HPAC prior's receptive-field SHAPE.

WHAT THIS MEASURES.  The shipped HPAC prior reads three geometric neighbourhoods
(reconstructed from the shipped receiver, see ``reconstruct``):

  * ``conv_a``    7x7 dilation 1, causal mask ``col - c + delta*(row - c) < 0``,
                  23 taps, on the CURRENT plane, inside a 64x64 patch;
  * ``conv_b1/2`` 5x5 dilation 2 and 3x3 dilation 4, depthwise, causal-or-equal,
                  14 and 5 taps, on the hidden features;
  * ``conv_past`` 3x3 dilation 1, DENSE (9 taps), on the PREVIOUS plane, with no
                  patch restriction.

This producer does not train and does not code.  For every candidate tap position it
measures, on the REAL 600x384x512 token field, how much the token's uncertainty falls
when that position is added to a fixed causal base set B:

    CMI(tap) = H(X | B) - H(X | B, tap)          [bits/symbol, plug-in]
    NET(tap) = N*CMI(tap) - dParams*(K-1)/2*log2(N)   [bits, two-part MDL]

``NET`` charges the extra context cells honestly, so a position whose information is
already carried by B scores ~0 and a position that adds nothing scores negative.

THIS IS A RANKING, NEVER A CHARGE.  Per the standing law ("first-order token price is a
ranking not a charge"), these numbers pre-register the SIGN of a shape rung; the byte
price comes only from the real coder on the real stream.  Axis:
``[macOS-CPU advisory; measured statistic on the shipped field, scorer-free]``;
``score_claim=false``.

Usage::

  python experiments/ddm_hpr1_surprise_atlas.py reconstruct --out-dir <dir>
  python experiments/ddm_hpr1_surprise_atlas.py atlas --out-dir <dir> --plane past
  python experiments/ddm_hpr1_surprise_atlas.py atlas --out-dir <dir> --plane current
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "2")

import numpy as np

#: Both SSD tiers hold this arm's store; MAIN re-routed NEW payloads to Vertigo on
#: 2026-09-11.  The field input stays where it was first bound and verified.
ARM_STORES = (
    "/Volumes/VertigoDataTier/pact/ddm_hpr1",
    "/Volumes/APDataStore/pact/ddm_hpr1",
)
FIELD = Path("/Volumes/APDataStore/pact/ddm_hpr1/inputs/field.u8")
FIELD_SHA = "a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8"
FRAMES, HEIGHT, WIDTH = 600, 384, 512
CLASSES = 5
PATCH = 64
DELTA = 2
#: Fixed causal base set, in (dy, dx).  Every member satisfies dx + DELTA*dy < 0, so
#: each is strictly before the target in the shipped scan order and is inside the
#: shipped conv_a support.  Four taps keep the base table at 5**4 = 625 cells.
BASE_TAPS = ((0, -1), (-1, 0), (-1, 1), (-1, -1))
CHUNK = 40  # frames per accumulation chunk


class AtlasError(RuntimeError):
    """A probe input or invariant is not what the shipped receiver says it is."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fact(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_json(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return fact(path)


def patch_group_mask(kernel: int, delta: int, type_: str) -> np.ndarray:
    """The shipped ``cpr1.hpac_integer.patch_group_mask``, re-derived here.

    Kept as an independent re-derivation rather than an import so the atlas can be
    checked against the runtime copy instead of inheriting it.
    """
    if type_ not in {"A", "B"}:
        raise AtlasError(f"unsupported mask type: {type_}")
    mask = np.zeros((kernel, kernel), dtype=np.int8)
    center = (kernel - 1) // 2
    for row in range(kernel):
        for column in range(kernel):
            offset = column - center + delta * (row - center)
            if offset < 0 or (type_ == "B" and offset == 0):
                mask[row, column] = 1
    return mask


def active_offsets(kernel: int, delta: int, type_: str, dilation: int) -> list[tuple[int, int]]:
    mask = patch_group_mask(kernel, delta, type_)
    center = (kernel - 1) // 2
    return [
        ((row - center) * dilation, (column - center) * dilation)
        for row, column in zip(*np.nonzero(mask), strict=True)
    ]


def reconstruct(runtime_root: Path) -> dict:
    """Read the shipped receiver and report the exact receptive field it builds."""
    import torch

    from experiments import ddm_rlc1_run as landed

    rx, renderer, _ = landed.io.load_runtime(runtime_root)
    from runtime import ihs2
    from runtime import rc2_hpac_semistatic_mixing as rc2

    parts = rx.read_residual_archive(runtime_root / "archive.zip")
    layout = ihs2.layout_from_runtime(renderer)
    counts = list(layout.row_counts)
    body = rx.materialize_ihs1(parts.hpac_blob, renderer)
    rows, depths = rc2.unpack_rows(body, counts)
    if len(rows) != len(counts):
        raise AtlasError("row/count mismatch in the shipped IHS1 body")
    model = renderer.load_hpac(body, torch.device("cpu"))
    geometry = {}
    for name, module in (
        ("conv_a", model.conv_a),
        ("conv_b1", model.conv_b1),
        ("conv_b2", model.conv_b2),
        ("conv_past", model.conv_past),
        ("spm_dw", model.spm_dw),
        ("spm_pw", model.spm_pw),
        ("head", model.head),
    ):
        kernel = int(module.mask.shape[-1])
        mask = module.mask[0, 0].numpy().astype(np.int8)
        center = (kernel - 1) // 2
        taps = [
            ((int(r) - center) * int(module.dilation), (int(c) - center) * int(module.dilation))
            for r, c in zip(*np.nonzero(mask), strict=True)
        ]
        geometry[name] = {
            "kernel": kernel,
            "dilation": int(module.dilation),
            "groups": int(module.groups),
            "c_in": int(module.weight.shape[1]),
            "c_out": int(module.weight.shape[0]),
            "taps": taps,
            "tap_count": len(taps),
            "stored_values_per_row": int(mask.sum()) * int(module.weight.shape[1]),
        }
    modules = {}
    for name, start, end in layout.module_ranges:
        row_counts = counts[start:end]
        row_depths = [int(d) for d in depths[start:end]]
        bits = sum(d * c for d, c in zip(row_depths, row_counts, strict=True))
        modules[name] = {
            "rows": end - start,
            "values_per_row": int(row_counts[0]),
            "values": int(sum(row_counts)),
            "mean_depth_bits": float(np.mean(row_depths)),
            "packed_bits": int(bits),
            "packed_bytes": bits / 8.0,
        }
    return {
        "runtime_root": str(runtime_root),
        "archive": fact(runtime_root / "archive.zip"),
        "hpac_member_bytes": len(parts.hpac_blob),
        "ihs1_body_bytes": len(body),
        "ihs1_body_sha256": hashlib.sha256(body).hexdigest(),
        "depth_bytes": layout.depth_bytes,
        "raw_tail_bytes": layout.raw_tail_bytes,
        "tail_fields": [
            {"name": f.name, "count": f.count, "dtype": f.dtype, "bytes": f.byte_count}
            for f in layout.tail_fields
        ],
        "geometry": geometry,
        "packed_modules": modules,
        "model_constants": {
            "patch": int(model.P),
            "delta": int(model.delta),
            "channels": int(model.ch),
            "num_classes": int(model.num_classes),
            "frame_dim": int(model.frame_embed.weight.shape[1]),
            "num_pairs": int(model.frame_embed.weight.shape[0]),
            "activation_bound": int(model.activation_bound),
            "use_spm": bool(model.use_spm),
            "use_frame_scale": bool(model.use_frame_scale),
        },
        "rederived_masks_match_runtime": {
            "conv_a": geometry["conv_a"]["taps"] == active_offsets(7, DELTA, "A", 1),
            "conv_b1": geometry["conv_b1"]["taps"] == active_offsets(5, DELTA, "B", 2),
            "conv_b2": geometry["conv_b2"]["taps"] == active_offsets(3, DELTA, "B", 4),
        },
        "axis": "[macOS-CPU advisory; shipped-receiver reconstruction]",
        "score_claim": False,
    }


def _load_field() -> np.memmap:
    if sha256_file(FIELD) != FIELD_SHA:
        raise AtlasError("field sha mismatch: the probe input is not the shipped field")
    return np.memmap(FIELD, dtype=np.uint8, mode="r", shape=(FRAMES, HEIGHT, WIDTH))


def _shift(plane: np.ndarray, dy: int, dx: int, patch_limited: bool):
    """Return (values, valid) for the neighbour at (dy, dx) of every position.

    ``patch_limited`` reproduces conv_a's zero padding at the 64x64 patch border:
    a tap that leaves the patch reads zero in the model, so it carries no
    information there and the position is excluded from the estimate.
    """
    rows = np.arange(plane.shape[-2])[:, None]
    columns = np.arange(plane.shape[-1])[None, :]
    source_rows = rows + dy
    source_columns = columns + dx
    if patch_limited:
        valid = (source_rows // PATCH == rows // PATCH) & (source_columns // PATCH == columns // PATCH)
    else:
        valid = (
            (source_rows >= 0)
            & (source_rows < plane.shape[-2])
            & (source_columns >= 0)
            & (source_columns < plane.shape[-1])
        )
    clipped_rows = np.clip(source_rows, 0, plane.shape[-2] - 1)
    clipped_columns = np.clip(source_columns, 0, plane.shape[-1] - 1)
    return (clipped_rows, clipped_columns), valid


def _conditional_entropy(counts: np.ndarray) -> tuple[float, int, int]:
    """H(X | context) in bits/symbol from a (contexts, CLASSES) count table."""
    totals = counts.sum(axis=-1).astype(np.float64)
    total = float(totals.sum())
    live = int(np.count_nonzero(totals))
    if total == 0:
        return 0.0, 0, 0
    nonzero = counts > 0
    values = counts.astype(np.float64)
    contribution = np.zeros_like(values)
    broadcast = np.broadcast_to(totals[..., None], values.shape)
    contribution[nonzero] = values[nonzero] * np.log2(values[nonzero] / broadcast[nonzero])
    return float(-contribution.sum() / total), int(total), live


def atlas(plane: str, candidates: list[tuple[int, int]], frame_stride: int) -> dict:
    field = _load_field()
    patch_limited = plane == "current"
    base_counts = np.zeros((CLASSES**len(BASE_TAPS), CLASSES), dtype=np.int64)
    joint_counts = {tap: np.zeros((CLASSES**len(BASE_TAPS), CLASSES, CLASSES), dtype=np.int64) for tap in candidates}
    first = 1 if plane == "past" else 0
    frames = list(range(first, FRAMES, frame_stride))
    for start in range(0, len(frames), CHUNK):
        block = frames[start : start + CHUNK]
        current = np.ascontiguousarray(field[block].astype(np.int32))
        base_valid = np.ones(current.shape, dtype=bool)
        base_index = np.zeros(current.shape, dtype=np.int32)
        for tap in BASE_TAPS:
            (rows, columns), valid = _shift(current, tap[0], tap[1], True)
            base_index = base_index * CLASSES + current[:, rows, columns]
            base_valid &= valid[None]
        base_flat = (base_index * CLASSES + current)[base_valid]
        base_counts += np.bincount(base_flat, minlength=base_counts.size).reshape(base_counts.shape)
        del base_flat
        source = current if plane == "current" else np.ascontiguousarray(field[[b - 1 for b in block]].astype(np.int32))
        for tap in candidates:
            (rows, columns), valid = _shift(source, tap[0], tap[1], patch_limited)
            keep = base_valid & valid[None]
            table = joint_counts[tap]
            flat = ((base_index * CLASSES + source[:, rows, columns]) * CLASSES + current)[keep]
            table += np.bincount(flat, minlength=table.size).reshape(table.shape)
            del flat, keep
        del current, source, base_index, base_valid
    base_entropy, base_total, base_live = _conditional_entropy(base_counts)
    rows_out = []
    for tap in candidates:
        counts = joint_counts[tap]
        joint_entropy, total, live = _conditional_entropy(counts.reshape(-1, CLASSES))
        restricted = counts.sum(axis=1)
        restricted_entropy, restricted_total, _ = _conditional_entropy(restricted)
        cmi = restricted_entropy - joint_entropy
        extra_cells = max(live - base_live, 0)
        penalty_bits = extra_cells * (CLASSES - 1) / 2.0 * math.log2(max(total, 2))
        rows_out.append(
            {
                "tap": list(tap),
                "cmi_bits_per_symbol": cmi,
                "base_entropy_bits": restricted_entropy,
                "joint_entropy_bits": joint_entropy,
                "observations": total,
                "live_cells": live,
                "net_bits": cmi * total - penalty_bits,
                "net_bytes_ranking_only": (cmi * total - penalty_bits) / 8.0,
            }
        )
    rows_out.sort(key=lambda row: -row["cmi_bits_per_symbol"])
    return {
        "plane": plane,
        "frame_stride": frame_stride,
        "frames_used": len(frames),
        "base_taps": [list(t) for t in BASE_TAPS],
        "base_entropy_bits": base_entropy,
        "base_observations": base_total,
        "base_live_cells": base_live,
        "field": {"path": str(FIELD), "sha256": FIELD_SHA},
        "rows": rows_out,
        "axis": "[macOS-CPU advisory; measured statistic on the shipped field, scorer-free]",
        "ranking_only": True,
        "score_claim": False,
    }


#: Candidate TEMPORAL tap SETS.  Each is four positions on the previous plane; four
#: keeps the joint table at 5**4 cells so the estimate stays dense.  ``shipped`` is the
#: four highest-information positions inside the shipped 3x3 dilation-1 window; the
#: others move the SAME four positions without adding any.
TEMPORAL_SETS = {
    "shipped_box_best4": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "shipped_box_corner4": ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    "recentred_plus1": ((1, 1), (1, 2), (2, 1), (2, 2)),
    "recentred_plus2": ((2, 2), (2, 3), (3, 2), (3, 3)),
    "dilated2_at_peak": ((0, 0), (0, 2), (2, 0), (2, 2)),
    "dilated3_at_peak": ((-1, -1), (-1, 2), (2, -1), (2, 2)),
    "dilated4_at_peak": ((-1, -1), (-1, 3), (3, -1), (3, 3)),
    "row_spread_plus1": ((1, -1), (1, 1), (1, 3), (1, 5)),
    "column_spread_plus1": ((-1, 1), (1, 1), (3, 1), (5, 1)),
}


#: Candidate SPATIAL tap SETS on the current plane, beyond the four base taps.  Each
#: holds four causal positions (``dx + DELTA*dy < 0``, the shipped type-A rule) and
#: differs only in the DILATION at which the cone is sampled.
SPATIAL_SETS = {
    "a_dil1_ring2": ((-2, 0), (-2, 1), (-1, -2), (0, -2)),
    "a_dil2_ring2": ((-4, 0), (-4, 2), (-2, -4), (0, -4)),
    "a_dil3_ring2": ((-6, 0), (-6, 3), (-3, -6), (0, -6)),
    "a_dil1_far3": ((-3, 0), (-3, 1), (-3, 2), (0, -3)),
    "a_mixed_1_2": ((-2, 0), (-2, 1), (-2, -4), (0, -4)),
}


def set_atlas(sets: dict[str, tuple[tuple[int, int], ...]], frame_stride: int, plane: str = "past") -> dict:
    """Code length of the target given the base set plus each candidate tap SET.

    Unlike the single-tap atlas this sees redundancy: two taps that carry the same
    information score no better than one.  Every candidate holds the same number of
    positions, so model capacity is held and only SHAPE varies.
    """
    field = _load_field()
    width = CLASSES ** len(BASE_TAPS)
    tables = {name: np.zeros((width, CLASSES**len(taps), CLASSES), dtype=np.int64) for name, taps in sets.items()}
    base_counts = np.zeros((width, CLASSES), dtype=np.int64)
    patch_limited = plane == "current"
    frames = list(range(1, FRAMES, frame_stride))
    for start in range(0, len(frames), CHUNK):
        block = frames[start : start + CHUNK]
        current = np.ascontiguousarray(field[block].astype(np.int32))
        previous = current if plane == "current" else np.ascontiguousarray(field[[b - 1 for b in block]].astype(np.int32))
        base_valid = np.ones(current.shape, dtype=bool)
        base_index = np.zeros(current.shape, dtype=np.int32)
        for tap in BASE_TAPS:
            (rows, columns), valid = _shift(current, tap[0], tap[1], True)
            base_index = base_index * CLASSES + current[:, rows, columns]
            base_valid &= valid[None]
        base_counts += np.bincount(
            (base_index * CLASSES + current)[base_valid], minlength=base_counts.size
        ).reshape(base_counts.shape)
        for name, taps in sets.items():
            keep = base_valid.copy()
            index = np.zeros(current.shape, dtype=np.int32)
            for tap in taps:
                (rows, columns), valid = _shift(previous, tap[0], tap[1], patch_limited)
                index = index * CLASSES + previous[:, rows, columns]
                keep &= valid[None]
            table = tables[name]
            flat = ((base_index * (CLASSES ** len(taps)) + index) * CLASSES + current)[keep]
            table += np.bincount(flat, minlength=table.size).reshape(table.shape)
            del flat, keep, index
        del current, previous, base_index, base_valid
    base_entropy, base_total, base_live = _conditional_entropy(base_counts)
    rows_out = []
    for name, taps in sets.items():
        counts = tables[name]
        joint_entropy, total, live = _conditional_entropy(counts.reshape(-1, CLASSES))
        restricted_entropy, _, restricted_live = _conditional_entropy(counts.sum(axis=1))
        cmi = restricted_entropy - joint_entropy
        penalty = max(live - restricted_live, 0) * (CLASSES - 1) / 2.0 * math.log2(max(total, 2))
        rows_out.append(
            {
                "name": name,
                "taps": [list(t) for t in taps],
                "cmi_bits_per_symbol": cmi,
                "restricted_base_entropy_bits": restricted_entropy,
                "joint_entropy_bits": joint_entropy,
                "observations": total,
                "live_cells": live,
                "net_bits": cmi * total - penalty,
                "net_bytes_ranking_only": (cmi * total - penalty) / 8.0,
            }
        )
    rows_out.sort(key=lambda row: -row["net_bits"])
    return {
        "stage": "set_atlas",
        "plane": plane,
        "frame_stride": frame_stride,
        "frames_used": len(frames),
        "base_taps": [list(t) for t in BASE_TAPS],
        "base_entropy_bits": base_entropy,
        "base_observations": base_total,
        "base_live_cells": base_live,
        "field": {"path": str(FIELD), "sha256": FIELD_SHA},
        "rows": rows_out,
        "axis": "[macOS-CPU advisory; measured statistic on the shipped field, scorer-free]",
        "ranking_only": True,
        "score_claim": False,
    }


#: Candidate 3x3 TEMPORAL window geometries, as (dilation, centre_dy, centre_dx).  Each
#: holds exactly nine taps, so model capacity is held and only SHAPE varies.  A dilation
#: change is a geometric parameter of the same kind as the kernel sizes the receiver
#: already ships; a centre shift is an OFFSET fitted to this video and is flagged as such.
TEMPORAL_WINDOWS = {
    "shipped_d1_c00": (1, 0, 0),
    "d2_c00": (2, 0, 0),
    "d3_c00": (3, 0, 0),
    "d1_c11_offset": (1, 1, 1),
    "d2_c11_offset": (2, 1, 1),
    "d3_c22_offset": (3, 2, 2),
}


def window_taps(dilation: int, centre_dy: int, centre_dx: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (centre_dy + dilation * row, centre_dx + dilation * column)
        for row in (-1, 0, 1)
        for column in (-1, 0, 1)
    )


def best_four(taps, single_tap_cmi: dict[tuple[int, int], float]) -> tuple[tuple[int, int], ...]:
    """The four positions of a window with the highest single-tap conditional information.

    A geometry-blind selection rule: every window is reduced the same way, so the joint
    estimate compares what each SHAPE can offer rather than a hand-picked subset.  Four
    keeps the joint table at 5**4 cells, which stays dense at 112 million observations.
    """
    missing = [tap for tap in taps if tap not in single_tap_cmi]
    if missing:
        raise AtlasError(f"single-tap atlas does not cover {missing}; widen --radius")
    return tuple(sorted(sorted(taps, key=lambda tap: -single_tap_cmi[tap])[:4]))


def _candidates(plane: str, radius: int) -> list[tuple[int, int]]:
    if plane == "past":
        return [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    return [
        (dy, dx)
        for dy in range(-radius, 1)
        for dx in range(-radius, radius + 1)
        if dx + DELTA * dy < 0 and (dy, dx) not in BASE_TAPS
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reconstruct", "atlas", "set_atlas", "window_atlas"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/ddm_pc3_pose_carrier_curve/candidate/candidate_runtime"),
    )
    parser.add_argument("--plane", choices=("past", "current"), default="past")
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args(argv)
    if not any(str(args.out_dir).startswith(root) for root in ARM_STORES):
        raise AtlasError("out-dir must be inside this arm's store")
    if args.stage == "reconstruct":
        payload = reconstruct(args.runtime_root)
        write_json(args.out_dir / "RECEPTIVE_FIELD.json", payload)
    elif args.stage == "window_atlas":
        source = args.out_dir / f"ATLAS_past_r{args.radius}.json"
        single = {
            tuple(row["tap"]): row["cmi_bits_per_symbol"]
            for row in json.loads(source.read_text())["rows"]
        }
        sets = {
            name: best_four(window_taps(*geometry), single)
            for name, geometry in TEMPORAL_WINDOWS.items()
        }
        payload = set_atlas(sets, args.frame_stride, "past")
        payload["windows"] = {
            name: {"geometry": list(geometry), "all_taps": [list(t) for t in window_taps(*geometry)]}
            for name, geometry in TEMPORAL_WINDOWS.items()
        }
        payload["single_tap_source"] = str(source)
        write_json(args.out_dir / "WINDOW_ATLAS_past.json", payload)
    elif args.stage == "set_atlas":
        sets = TEMPORAL_SETS if args.plane == "past" else SPATIAL_SETS
        payload = set_atlas(sets, args.frame_stride, args.plane)
        write_json(args.out_dir / f"SET_ATLAS_{args.plane}.json", payload)
    else:
        payload = atlas(args.plane, _candidates(args.plane, args.radius), args.frame_stride)
        write_json(args.out_dir / f"ATLAS_{args.plane}_r{args.radius}.json", payload)
    print(json.dumps({k: v for k, v in payload.items() if k not in ("rows", "packed_modules")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
