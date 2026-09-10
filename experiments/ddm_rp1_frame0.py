#!/usr/bin/env python3
"""ddm_rp1 -- the frame-0 selector as a SEG-FREE pose actuator, inside the admission.

WHY THIS BELONGS IN THE ADMISSION AND NOT IN A SWEEP
-----------------------------------------------------
SegNet reads the LAST frame of the pair, so frame 0 is structurally seg-free: any pixel
operation applied to it moves d_pose and cannot move d_seg.  The shipped receiver already
carries the mechanism -- ``runtime/frame0_selector.py`` decodes a sparse per-pair choice
over EIGHT integer-only modes, and ``up2.render_frame0`` is "carrier render then selector
override", so the instrument exists and only the ENCODER was missing.

Round 1 measured why it matters here: of 197 pairs the admission dropped, **44 were
pose-bound** -- their token change already SAVED bytes and the carrier re-solve alone could
not give the pose back.  Those pairs are not a population to sweep; they are a named,
measured list.  The move-40 body ships 24 non-identity choices out of 600, i.e. 96 % of
pairs are at IDENTITY and 84 % of the population was already measured optimal, so a
standalone global sweep would spend bytes to re-discover the shipped answer.  **This module
only ever proposes on pairs the admission is about to drop for pose.**

THE PRICE IS THE BLOB, EXACTLY
-------------------------------
The selector is stored as ``F0E1`` + version + u16 count, then the COMBINATORIAL RANK of the
chosen positions among ``C(600, count)``, then 3 bits of label per chosen pair.  So the
marginal cost of adopting one more pair is roughly ``log2((600-k)/k) + log2(e) + 3`` bits --
about 1.1 B at k=24 -- and it is not estimated here: the blob is ENCODED and measured, and
every encode is round-tripped through the receiver's own ``decode_selector`` before its
length is quoted.

``[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT]``; ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_jg1_seg_solve as jg1  # noqa: E402
import ddm_rp1_pose as rp1pose  # noqa: E402
import ddm_rp1_rate_rank as rp1  # noqa: E402
import ddm_up2_shipping_pose_solve as up2  # noqa: E402

N_PAIRS = jg1.N_PAIRS
N_MODES = 8
SPARSE_MAGIC = b"F0E1"
SPARSE_VERSION = 1
_HEADER = struct.Struct("<4sBH")
#: The 6-byte wrapper ``carrier_repack.pack_frame0_selector_carrier`` puts in front of a
#: carrier that carries a selector at all.  Charged once, and only when the body has none.
CARRIER_WRAPPER_BYTES = 6


def encode_selector(choices: np.ndarray) -> bytes:
    """Encode a (600,) choice vector the way the receiver decodes it.

    The inverse of ``frame0_selector._combination_unrank``: the unranker peels
    ``C(position, width)`` off the remainder for width = count..1, so the rank is
    ``sum_w C(positions[w-1], w)`` over the ascending chosen positions.
    """
    choices = np.asarray(choices, dtype=np.uint8)
    if choices.shape != (N_PAIRS,):
        raise rp1.Rp1Error(f"choices must be ({N_PAIRS},), got {choices.shape}")
    if choices.max(initial=0) >= N_MODES:
        raise rp1.Rp1Error("a choice is outside the 8-mode catalog")
    positions = np.flatnonzero(choices)
    count = int(positions.size)
    if not 1 <= count <= N_PAIRS:
        raise rp1.Rp1Error(
            f"the sparse selector must carry 1..{N_PAIRS} non-identity pairs, not {count}"
        )
    rank = sum(math.comb(int(positions[w - 1]), w) for w in range(1, count + 1))
    limit = math.comb(N_PAIRS, count)
    rank_bytes = ((limit - 1).bit_length() + 7) // 8
    bits = 0
    width = 0
    for pos in positions:
        bits = (bits << 3) | (int(choices[pos]) - 1)
        width += 3
    pad = (-width) % 8
    labels = (bits << pad).to_bytes((width + pad) // 8, "big")
    return _HEADER.pack(SPARSE_MAGIC, SPARSE_VERSION, count) + rank.to_bytes(
        rank_bytes, "big"
    ) + labels


def selector_bytes(choices: np.ndarray, runtime_dir: Path) -> int:
    """Encoded length, PROVEN by round-tripping the receiver's own decoder."""
    blob = encode_selector(choices)
    sys.path.insert(0, str(Path(runtime_dir).resolve()))
    try:
        from runtime.frame0_selector import decode_selector  # type: ignore
    finally:
        sys.path.pop(0)
    _, back = decode_selector(blob)
    if not np.array_equal(np.asarray(back, dtype=np.uint8),
                          np.asarray(choices, dtype=np.uint8)):
        raise rp1.Rp1Error("selector round-trip changed the choices")
    return len(blob)


def cmd_sweep(args: argparse.Namespace) -> int:
    """Per-pair d_pose under every one of the 8 modes, on the candidate's own renders."""
    rp1pose.set_active_pointer(args.pointer_tree, args.expect_pointer_sha,
                              args.carrier_bytes)
    rp1.set_threads(args.threads)
    receipts = rp1pose.assert_pointer_and_carrier()
    inst = rp1pose.load_instrument(Path(args.overlay))
    codes = (np.load(args.codes).astype(np.int32) if args.codes
             else np.asarray(inst.state.codes, dtype=np.int32))
    coefficients = up2.codes_to_coefficients(codes, inst.state.coefficient_scales)
    pairs = [int(p) for p in json.loads(Path(args.pairs).read_text())]
    shipped = np.asarray(inst.state.selector_choices, dtype=np.uint8).copy()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "frame0_sweep_rows.jsonl"
    done: set[int] = set()
    if args.resume and rows_path.is_file():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                done.add(int(json.loads(line)["pair"]))
    else:
        rows_path.write_text("")

    started = time.time()
    for n, pair in enumerate(pairs):
        if pair in done:
            continue
        # One pair, eight modes.  The carrier render is identical across modes -- only
        # the selector override differs -- so each mode is one PoseNet forward on the
        # same frame 0, and the shipped mode is measured alongside the alternatives
        # rather than carried in from another run.
        per_mode = np.zeros(N_MODES, dtype=np.float64)
        for mode in range(N_MODES):
            choices_mode = shipped.copy()
            choices_mode[pair] = mode
            state = inst.state.__class__(
                **{**inst.state.__dict__, "selector_choices": choices_mode}
            )
            value, _ = up2.measure_pose(
                inst.posenet, state, coefficients, inst.raw, inst.targets,
                np.array([pair], dtype=np.int64), batch_size=1,
            )
            per_mode[mode] = float(value[0])
        best = int(np.argmin(per_mode))
        row = {
            "pair": pair,
            "shipped_mode": int(shipped[pair]),
            "d_pose_by_mode": per_mode.tolist(),
            "d_pose_shipped": float(per_mode[int(shipped[pair])]),
            "best_mode": best,
            "d_pose_best": float(per_mode[best]),
            "gain": float(per_mode[int(shipped[pair])] - per_mode[best]),
        }
        with rows_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if (n + 1) % 10 == 0 or n + 1 == len(pairs):
            print(json.dumps({"done": n + 1, "of": len(pairs), "pair": pair,
                              "s_per_pair": (time.time() - started) / (n + 1)}), flush=True)
    (out / "SWEEP.json").write_text(json.dumps(
        {"schema": "ddm_rp1_frame0_sweep.v1", "pairs": len(pairs),
         "rows": str(rows_path), "receipts": receipts,
         "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT]",
         "score_claim": False, "elapsed_seconds": time.time() - started},
        indent=2, sort_keys=True))
    return 0


def cmd_adopt(args: argparse.Namespace) -> int:
    """Adopt the modes that pay, and price the blob delta by ENCODING it."""
    rp1pose.set_active_pointer(args.pointer_tree, args.expect_pointer_sha,
                              args.carrier_bytes)
    tree = rp1pose._ACTIVE["tree"]
    state = up2.load_carrier_state(tree, verify_archive=False)
    shipped = np.asarray(state.selector_choices, dtype=np.uint8).copy()
    base_blob = selector_bytes(shipped, tree)

    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines()
            if line.strip()]
    base_pose = np.load(args.base_pose).astype(np.float64)
    d_marginal = 5.0 / math.sqrt(10.0 * float(base_pose.mean())) / N_PAIRS
    s_per_byte = rp1.S_PER_BYTE

    # Rank by gain, then adopt greedily while each adoption's pose credit still beats the
    # blob byte it costs.  The blob is re-ENCODED at every step, so the marginal price is
    # the real one and not the 1.1 B rule of thumb.
    rows.sort(key=lambda r: -r["gain"])
    choices = shipped.copy()
    adopted: list[dict[str, Any]] = []
    running = base_blob
    for row in rows:
        if row["gain"] <= 0 or row["best_mode"] == row["shipped_mode"]:
            continue
        trial = choices.copy()
        trial[int(row["pair"])] = int(row["best_mode"])
        trial_blob = selector_bytes(trial, tree)
        delta_bytes = trial_blob - running
        delta_s = -row["gain"] * d_marginal + delta_bytes * s_per_byte
        if delta_s >= 0:
            continue
        choices = trial
        running = trial_blob
        adopted.append({
            "pair": int(row["pair"]),
            "from_mode": int(row["shipped_mode"]),
            "to_mode": int(row["best_mode"]),
            "d_pose_shipped": row["d_pose_shipped"],
            "d_pose_adopted": row["d_pose_best"],
            "gain": row["gain"],
            "blob_delta_bytes": int(delta_bytes),
            "delta_S": float(delta_s),
        })
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "selector_choices_adopted.npy", choices)
    verdict = {
        "schema": "ddm_rp1_frame0_adopt.v1",
        "axis": "pose [macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT]; "
                "selector bytes EXACT from an encode round-tripped through the "
                "receiver's own decode_selector",
        "score_claim": False,
        "pointer_tree": str(tree),
        "shipped_non_identity_pairs": int((shipped != 0).sum()),
        "shipped_selector_blob_bytes": base_blob,
        "adopted_pairs": len(adopted),
        "adopted": adopted,
        "final_non_identity_pairs": int((choices != 0).sum()),
        "final_selector_blob_bytes": running,
        "selector_blob_delta_bytes": running - base_blob,
        "total_pose_gain": float(sum(a["gain"] for a in adopted)),
        "total_delta_S": float(sum(a["delta_S"] for a in adopted)),
        "pose_marginal_dS_per_unit": d_marginal,
        "choices_path": str(out / "selector_choices_adopted.npy"),
    }
    (out / "FRAME0_ADOPT.json").write_text(json.dumps(verdict, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in verdict.items() if k != "adopted"}, indent=2))
    return 0


def _add_pointer_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pointer-tree", default=None)
    parser.add_argument("--expect-pointer-sha", default=None)
    parser.add_argument("--carrier-bytes", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sweep = sub.add_parser("sweep", help="8-mode frame-0 sweep on a named pair list")
    sweep.add_argument("--overlay", required=True)
    sweep.add_argument("--codes", default=None)
    sweep.add_argument("--pairs", required=True)
    sweep.add_argument("--out-dir", required=True)
    sweep.add_argument("--threads", type=int, default=2)
    sweep.add_argument("--resume", action="store_true")
    _add_pointer_flags(sweep)
    sweep.set_defaults(func=cmd_sweep)

    adopt = sub.add_parser("adopt", help="adopt what pays and price the blob by encoding it")
    adopt.add_argument("--rows", required=True)
    adopt.add_argument("--base-pose", required=True)
    adopt.add_argument("--out-dir", required=True)
    _add_pointer_flags(adopt)
    adopt.set_defaults(func=cmd_adopt)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
