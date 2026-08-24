"""ddm_msr1 — an actuator-independent bound on the ZERO-BYTE reach of the dx2
manufactured Seg error.

ddm_mf1 (commit b0c2869ce4) localized the manufactured set and measured two repair
FORMULATIONS, both of which lost jointly, and left the family question open:
"this does not close jointly retraining the counted renderer weights so that the
repair is internalized without a shipped mask."

This instrument answers that family question structurally rather than by trying a
third actuator. Three derivations, all over already-measured retained fields:

1. INTERFACE FLOW BALANCE. Every manufactured pixel is a directed flow `GT a ->
   terminal b` across a class-pair interface. Any actuator that moves the painted
   boundary of interface (a,b) in ONE direction over a region fixes the a->b flow
   in that region and, one for one, deepens the b->a flow. So the a->b and b->a
   flows that COINCIDE inside an actuator's addressing cell cancel: only the NET
   imbalance per cell is reachable. This is true for every boundary-moving
   actuator, learned or hand-written, so it bounds the family, not a formulation.

2. THE ADDRESSING LADDER. Net imbalance is computed at increasing addressing
   granularity, from one sign per interface up to one sign per pixel, with a
   counting lower bound on the address payload each granularity needs. Repair
   value and address cost are then compared in the same byte currency.

3. THE COLLATERAL MARGIN LEDGER. For a scalar logit-shift ``delta`` applied on the
   token-boundary shell, the pixels repaired are the manufactured ones whose
   frozen-head deficit is below ``delta`` and the pixels broken are the currently
   correct shell pixels whose top-1 margin is below ``delta``. Both distributions
   are measured from the retained native logits, so the net curve is measured, not
   modelled; only the "a shift of delta moves both populations" step is assumed,
   and it is stated as an assumption with its falsifier.

Authority: none. No archive changes, no scorer fires, no pointer moves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

EVAL_H, EVAL_W, PAIRS = 384, 512, 600
NUM_PIXELS = PAIRS * EVAL_H * EVAL_W
NUM_CLASSES = 5

# CITED from ddm_tx1_toolbox_crosswalk_20260819.md sec 0. Never re-derived (#1207).
EXCHANGE_RATE_S_PER_BYTE = 6.658590e-07
# CITED from ddm_fb1_sub012_feasibility_bound_20260823.md.
DEMAND_BYTES = 42382
# CITED from ddm_mf1_manufactured_seg_repair_20260823.md: the measured Brotli-q11
# size of the exact per-pixel n600 native address mask.
MF1_PER_PIXEL_ADDRESS_BYTES = 35969

CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 22)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def bytes_per_flip() -> float:
    return (100.0 / NUM_PIXELS) / EXCHANGE_RATE_S_PER_BYTE


def cell_index_at(
    flat_index: np.ndarray, rows_per_cell: int, cols_per_cell: int, per_pair: bool
) -> np.ndarray:
    """Addressing-cell id for the given pixels, without materializing the full body."""
    within = flat_index % (EVAL_H * EVAL_W)
    row = (within // EVAL_W) // rows_per_cell
    col = (within % EVAL_W) // cols_per_cell
    col_cells = -(-EVAL_W // cols_per_cell)
    cell = row * col_cells + col
    if not per_pair:
        return cell
    row_cells = -(-EVAL_H // rows_per_cell)
    return (flat_index // (EVAL_H * EVAL_W)) * (row_cells * col_cells) + cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retained",
        type=Path,
        default=Path(
            "/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/"
            "ddm_mst1_manufactured_stage_split/capture_r2_local/retained"
        ),
    )
    parser.add_argument(
        "--characterize",
        type=Path,
        default=Path(
            "/Volumes/APDataStore/pact/ddm_msr1_manufactured_seg_reduction/characterize_r1"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/Volumes/APDataStore/pact/ddm_msr1_manufactured_seg_reduction/reach_r1"),
    )
    args = parser.parse_args()

    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    per_flip = bytes_per_flip()
    result: dict = {
        "arm": "ddm_msr1",
        "kind": "actuator_independent_family_bound",
        "authority": "none; no archive changed, no scorer fired, no pointer moved",
        "bytes_per_flip": per_flip,
        "exchange_rate_s_per_byte": EXCHANGE_RATE_S_PER_BYTE,
        "exchange_rate_source": "ddm_tx1_toolbox_crosswalk_20260819.md sec 0 (CITED)",
        "demand_bytes": DEMAND_BYTES,
        "consumes": {
            "mf1_memo": "ddm_mf1_manufactured_seg_repair_20260823.md (commit b0c2869ce4)",
            "mf1_per_pixel_address_bytes": MF1_PER_PIXEL_ADDRESS_BYTES,
        },
    }

    gt = np.load(args.retained / "inputs" / "gt_argmax_n600.npy").reshape(-1)
    tokens = np.fromfile(args.retained / "inputs" / "tokens_cpu_stage_complete.u8", dtype=np.uint8)
    terminal = np.load(args.retained / "inputs" / "cuda_terminal_argmax_n600.npy").reshape(-1)
    native = np.load(args.retained / "assembled" / "argmax_native_n600.npy").reshape(-1)
    manufactured = (terminal != gt) & (tokens == gt)
    stage_id = np.load(args.characterize / "earliest_stage_id.n600.npy")
    distance = np.load(args.characterize / "token_boundary_distance.n600.npy")
    deficit = np.load(args.characterize / "native_logit_deficit.float32.n600.npy")
    native_manufactured = manufactured & (stage_id == 0)
    print(f"[{time.time() - started:6.1f}s] loaded fields")

    result["support"] = {
        "manufactured_all_stages": int(manufactured.sum()),
        "manufactured_native_stage": int(native_manufactured.sum()),
        "byte_ceiling_all_stages": float(manufactured.sum() * per_flip),
        "byte_ceiling_native_stage": float(native_manufactured.sum() * per_flip),
    }

    # ---- 1. interface flow balance ---------------------------------------
    index = np.flatnonzero(manufactured)
    flow = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(flow, (gt[index], terminal[index]), 1)
    interfaces = []
    balanced_total = 0
    net_total = 0
    for a in range(NUM_CLASSES):
        for b in range(a + 1, NUM_CLASSES):
            forward, backward = int(flow[a, b]), int(flow[b, a])
            if forward + backward == 0:
                continue
            net = abs(forward - backward)
            balanced = 2 * min(forward, backward)
            balanced_total += balanced
            net_total += net
            interfaces.append(
                {
                    "interface": f"{CLASS_NAMES[a]}|{CLASS_NAMES[b]}",
                    "gt_a_read_b": forward,
                    "gt_b_read_a": backward,
                    "gross": forward + backward,
                    "balanced_cancelling": balanced,
                    "net_imbalance": net,
                    "balanced_share": balanced / (forward + backward),
                }
            )
    interfaces.sort(key=lambda row: -row["gross"])
    result["interface_flow"] = {
        "definition": "manufactured pixels as directed flows GT a -> terminal b across a "
        "class-pair interface; a one-directional boundary move inside an addressing cell "
        "fixes one direction and deepens the other one-for-one, so only the NET imbalance "
        "inside a cell is reachable",
        "rows": interfaces,
        "gross_total": int(manufactured.sum()),
        "balanced_cancelling_total": balanced_total,
        "net_imbalance_total": net_total,
        "balanced_share_of_manufactured": balanced_total / int(manufactured.sum()),
        "interface_global_reach_pixels": net_total,
        "interface_global_reach_bytes": net_total * per_flip,
        "interface_global_reach_share_of_manufactured": net_total / int(manufactured.sum()),
    }
    print(f"[{time.time() - started:6.1f}s] interface flow: net {net_total}, balanced {balanced_total}")

    # ---- 2. addressing ladder --------------------------------------------
    pair_class = gt[index].astype(np.int64) * NUM_CLASSES + terminal[index].astype(np.int64)
    unordered = np.minimum(gt[index], terminal[index]).astype(np.int64) * NUM_CLASSES + np.maximum(
        gt[index], terminal[index]
    ).astype(np.int64)
    direction = (gt[index] < terminal[index]).astype(np.int64)
    del pair_class

    ladder = []
    grids = [
        ("interface_global", None, None, False),
        ("interface_x_pair", EVAL_H, EVAL_W, True),
        ("interface_x_pair_x_96x128", 96, 128, True),
        ("interface_x_pair_x_48x64", 48, 64, True),
        ("interface_x_pair_x_24x32", 24, 32, True),
        ("interface_x_pair_x_12x16", 12, 16, True),
        ("interface_x_pair_x_6x8", 6, 8, True),
        ("interface_x_pair_x_3x4", 3, 4, True),
        ("interface_x_pair_x_1x1_per_pixel", 1, 1, True),
    ]
    for name, rows_per_cell, cols_per_cell, per_pair in grids:
        if rows_per_cell is None:
            cell = np.zeros(index.size, dtype=np.int64)
        else:
            cell = cell_index_at(index, rows_per_cell, cols_per_cell, per_pair)
        key = cell * (NUM_CLASSES * NUM_CLASSES) + unordered
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        sorted_direction = direction[order]
        boundaries = np.flatnonzero(np.diff(sorted_key)) + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [key.size]))
        forward_counts = np.add.reduceat(sorted_direction, starts)
        totals = stops - starts
        backward_counts = totals - forward_counts
        net_per_cell = np.abs(forward_counts - backward_counts)
        reachable = int(net_per_cell.sum())
        active_cells = int(starts.size)
        # Counting lower bound on the address payload: one sign bit per active
        # (cell, interface) group that the actuator must be told about. This is a
        # LOWER bound: it charges nothing for locating the cells themselves.
        address_bytes_lower_bound = active_cells / 8.0
        ladder.append(
            {
                "granularity": name,
                "active_cell_interface_groups": active_cells,
                "reachable_pixels": reachable,
                "reachable_share_of_manufactured": reachable / int(manufactured.sum()),
                "repair_value_bytes": reachable * per_flip,
                "address_cost_bytes_lower_bound": address_bytes_lower_bound,
                "value_minus_cost_bytes": reachable * per_flip - address_bytes_lower_bound,
            }
        )
        print(
            f"[{time.time() - started:6.1f}s] ladder {name:34s} reach {reachable:>7,d} "
            f"value {reachable * per_flip:9.1f} B cost>= {address_bytes_lower_bound:9.1f} B"
        )
    # Replace the per-pixel row's counting bound with mf1's MEASURED Brotli size.
    ladder[-1]["address_cost_bytes_measured_mf1"] = MF1_PER_PIXEL_ADDRESS_BYTES
    ladder[-1]["value_minus_measured_cost_bytes"] = (
        ladder[-1]["repair_value_bytes"] - MF1_PER_PIXEL_ADDRESS_BYTES
    )
    result["addressing_ladder"] = {
        "definition": "reachable = sum over (addressing cell x interface) of |forward - backward| "
        "manufactured flow; address cost is a COUNTING LOWER BOUND of one sign bit per active "
        "group, which charges nothing for locating the groups",
        "rows": ladder,
    }

    # ---- 3. collateral margin ledger -------------------------------------
    shell = distance == 1
    correct_now = terminal == gt
    correct_native = native == gt
    shell_correct = shell & correct_now & correct_native
    shell_manufactured = shell & native_manufactured
    margin_path = args.characterize / "native_top1_margin.float32.n600.npy"
    if margin_path.exists():
        margin = np.load(margin_path)
    else:
        margin = recompute_margin(args.retained, gt)
        np.save(margin_path, margin)
    # Alignment positive control. The per-chunk logits are joined to GT by position
    # only, so a chunk-order or reshape error would silently mislabel every barrier
    # below. The deficit must be exactly zero where the native argmax equals GT and
    # strictly positive where it does not; that holds only under correct alignment.
    misaligned_zero = int((deficit[native == gt] != 0).sum())
    misaligned_positive = int((deficit[native != gt] <= 0).sum())
    if misaligned_zero or misaligned_positive:
        raise ValueError(
            f"logit/GT alignment control failed: {misaligned_zero} correct pixels with "
            f"non-zero deficit, {misaligned_positive} wrong pixels with non-positive deficit"
        )
    result["logit_gt_alignment_control"] = {
        "correct_pixels_with_nonzero_deficit": 0,
        "wrong_pixels_with_nonpositive_deficit": 0,
    }
    print(f"[{time.time() - started:6.1f}s] margins ready; logit/GT alignment control passed")

    # The small-delta end is searched explicitly: selectivity is best as delta -> 0,
    # so a negative-existence claim that skipped it would have an unsearched corner.
    deltas = [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0]
    repaired = deficit[shell_manufactured]
    exposed = margin[shell_correct]
    curve = []
    for delta in deltas:
        fixed = int((repaired < delta).sum())
        broken = int((exposed < delta).sum())
        curve.append(
            {
                "delta_logits": delta,
                "manufactured_repaired": fixed,
                "correct_shell_broken": broken,
                "net_errors_removed": fixed - broken,
                "net_value_bytes": (fixed - broken) * per_flip,
            }
        )
    result["collateral_margin_ledger"] = {
        "definition": "an undirected logit shift of size delta applied on the token-boundary "
        "shell repairs manufactured pixels whose native deficit is below delta and breaks "
        "currently-correct shell pixels whose native top-1 margin is below delta",
        "assumption": "the same shift magnitude reaches both populations; falsified by any "
        "actuator measured to move manufactured deficits without moving correct-pixel margins "
        "by a comparable amount on the same shell",
        "shell_pixels": int(shell.sum()),
        "shell_share_of_body": float(shell.sum() / NUM_PIXELS),
        "shell_manufactured": int(shell_manufactured.sum()),
        "shell_correct": int(shell_correct.sum()),
        "collateral_ratio_correct_per_manufactured": float(
            shell_correct.sum() / max(1, shell_manufactured.sum())
        ),
        "correct_shell_margin_quantiles": {
            str(q): float(v)
            for q, v in zip(
                [0.001, 0.01, 0.05, 0.1, 0.25, 0.5],
                np.quantile(exposed, [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]),
                strict=True,
            )
        },
        "manufactured_shell_deficit_quantiles": {
            str(q): float(v)
            for q, v in zip(
                [0.1, 0.25, 0.5, 0.75, 0.9],
                np.quantile(repaired, [0.1, 0.25, 0.5, 0.75, 0.9]),
                strict=True,
            )
        },
        "curve": curve,
    }
    print(f"[{time.time() - started:6.1f}s] collateral ledger done")

    # ---- 4. directional per-interface bound (actuator-matched) -----------
    # The undirected ledger above charges the whole shell as collateral. The real
    # actuator is narrower: moving interface (a,b) toward a can only fix a->b
    # manufactured pixels and can only break correct pixels of token class b that
    # actually touch class a. This section measures that narrower ledger, so a
    # refusal here is a refusal of the actuator's own best case.
    presence = class_presence_within_radius_one(tokens)
    print(f"[{time.time() - started:6.1f}s] class adjacency presence computed")
    directional = []
    for a in range(NUM_CLASSES):
        for b in range(a + 1, NUM_CLASSES):
            side_a = (tokens == a) & presence[b]
            side_b = (tokens == b) & presence[a]
            if not side_a.any() and not side_b.any():
                continue
            row = {"interface": f"{CLASS_NAMES[a]}|{CLASS_NAMES[b]}", "directions": []}
            for source, target, own_side, other_side in (
                (a, b, side_a, side_b),
                (b, a, side_b, side_a),
            ):
                # Moving the painted boundary toward `source` repairs manufactured
                # `source -> target` pixels and endangers correct `target` pixels.
                repairable = own_side & native_manufactured & (gt == source) & (terminal == target)
                endangered = other_side & correct_now & correct_native & (gt == target)
                rep_deficit = deficit[repairable]
                end_margin = margin[endangered]
                best = {"delta_logits": None, "net_errors_removed": 0, "net_value_bytes": 0.0}
                points = []
                for delta in deltas:
                    fixed = int((rep_deficit < delta).sum())
                    broken = int((end_margin < delta).sum())
                    point = {
                        "delta_logits": delta,
                        "fixed": fixed,
                        "broken": broken,
                        "net_errors_removed": fixed - broken,
                    }
                    points.append(point)
                    if fixed - broken > best["net_errors_removed"]:
                        best = {
                            "delta_logits": delta,
                            "net_errors_removed": fixed - broken,
                            "net_value_bytes": (fixed - broken) * per_flip,
                        }
                row["directions"].append(
                    {
                        "move_toward": CLASS_NAMES[source],
                        "repairable_pixels": int(repairable.sum()),
                        "endangered_correct_pixels": int(endangered.sum()),
                        "collateral_ratio": float(
                            endangered.sum() / max(1, int(repairable.sum()))
                        ),
                        "curve": points,
                        "best": best,
                    }
                )
            directional.append(row)
    best_total = sum(
        max(direction["best"]["net_errors_removed"] for direction in row["directions"])
        for row in directional
    )
    result["directional_interface_bound"] = {
        "definition": "moving interface (a,b) toward a repairs native-stage manufactured "
        "a->b pixels whose native deficit is below delta and breaks currently-correct "
        "token-class-b pixels touching class a whose native top-1 margin is below delta; "
        "the per-interface best direction and delta are taken as an ORACLE upper bound",
        "assumption": "same as the undirected ledger: a boundary move of magnitude delta "
        "reaches both populations on the same interface",
        "rows": directional,
        "oracle_best_net_errors_removed_all_interfaces": best_total,
        "oracle_best_value_bytes": best_total * per_flip,
        "oracle_best_share_of_manufactured": best_total / int(manufactured.sum()),
        "oracle_best_share_of_demand": best_total * per_flip / DEMAND_BYTES,
    }
    print(
        f"[{time.time() - started:6.1f}s] directional oracle best {best_total} errors "
        f"= {best_total * per_flip:.1f} B"
    )

    result["elapsed_seconds"] = time.time() - started
    payload_path = args.out / "reach_masks.npz"
    np.savez_compressed(
        payload_path,
        shell_manufactured=np.packbits(shell_manufactured),
        shell_correct=np.packbits(shell_correct),
    )
    result["payloads"] = {
        "reach_masks": {
            "path": str(payload_path),
            "bytes": payload_path.stat().st_size,
            "sha256": sha256_of_file(payload_path),
        },
        "native_top1_margin": {
            "path": str(margin_path),
            "bytes": margin_path.stat().st_size,
            "sha256": sha256_of_file(margin_path),
        },
    }
    result_path = args.out / "MSR1_REACH.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {result_path} sha256 {sha256_of_file(result_path)}")
    return 0


def class_presence_within_radius_one(tokens: np.ndarray) -> list[np.ndarray]:
    """For each class, a mask of pixels having that class within Chebyshev radius 1."""
    volume = tokens.reshape(PAIRS, EVAL_H, EVAL_W)
    presence = []
    for index in range(NUM_CLASSES):
        occupied = volume == index
        padded = np.pad(occupied, ((0, 0), (1, 1), (1, 1)), mode="edge")
        near = np.zeros_like(occupied)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                near |= padded[:, dy : dy + EVAL_H, dx : dx + EVAL_W]
        presence.append(near.reshape(-1))
    return presence


def recompute_margin(retained: Path, gt: np.ndarray) -> np.ndarray:
    margin = np.zeros(NUM_PIXELS, dtype=np.float32)
    chunk_dirs = sorted(p for p in (retained / "chunks").iterdir() if p.is_dir())
    cursor = 0
    for chunk_dir in chunk_dirs:
        logits = np.load(chunk_dir / "logits_native.float32.npy")
        logits = np.moveaxis(logits, 1, -1)
        flat = logits.reshape(-1, NUM_CLASSES)
        ordered = np.sort(flat, axis=1)
        margin[cursor : cursor + flat.shape[0]] = ordered[:, -1] - ordered[:, -2]
        cursor += flat.shape[0]
    if cursor != NUM_PIXELS:
        raise ValueError(f"chunk logits covered {cursor} pixels, expected {NUM_PIXELS}")
    return margin


if __name__ == "__main__":
    raise SystemExit(main())
