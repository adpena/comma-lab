#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""t1h -- how far are the SHIPPED rr4 carrier codes from their per-pair pose optimum?

PR133's eval-bot-confirmed move was 89.5% a BYTE-FROZEN coefficient re-solve: a greedy
coordinate search over already-transmitted integer codes, accepted against an exact
PoseNet forward, with uint8 rounding inside the loop.  Our pose term is 97.2% of the
remaining sub-0.15 gap.  Nobody has measured how far OUR shipped codes sit from their
per-pair optimum, so this tool measures that before anything is solved.

WHY THE MEASUREMENT IS PER-PAIR SEPARABLE (the fact that makes it affordable)
----------------------------------------------------------------------------
In the shipped F26 receiver the EVEN frame of pair i is rendered from that pair's 12
carrier codes alone (``cpr1/inflate.py:335-350``: ``einsum("bk,kchw->bchw")`` over row i,
then bicubic to camera resolution, then the sparse frame-0 selector).  The ODD frame comes
from the semantic renderer and the token stream and does NOT depend on any carrier code.
``d_pose`` is a per-pair MSE over 6 PoseNet outputs.  Therefore pair i's pose energy is a
function of pair i's 12 codes ONLY, and the 600 pairs are independent sub-problems.

WHAT IS EXACT AND WHAT IS NOT
-----------------------------
Everything scored here is EXACT: the hard ``.round()`` chain, the real selector, the
upstream PoseNet preprocess, and the frozen CPU-torch PoseNet.  No straight-through
estimator, no linearization, no surrogate.  The instrument is proven by a byte-identity
control: the frame_0 this tool renders from the shipped codes must equal the retained
``0.raw`` even frame for every pair it touches.  jc1 learned that lesson the expensive way
-- its first implementation omitted the selector and failed on exactly the 5 covered
frames.  A control failure here aborts.

AXIS HONESTY.  The accept oracle is CPU-torch PoseNet -- the only pose authority this arm
may run.  The frontier's 6.88e-6 ``d_pose`` is a contest-CUDA T4 number.  This tool reports
the CPU-torch base it actually measures and expresses headroom as a RATIO of that base, so
the result transfers as a fraction rather than as a forged absolute.  Nothing here is a
score.

Byte accounting is NOT free.  The carrier codes are delta-coded along time, zigzagged and
Rice-coded per dimension, so moving pair i's code changes the packed bit count at i and at
i+1.  This tool measures the DISTORTION headroom; a solve would additionally have to hold
total bytes at or below the shipped size.  That is stated, not assumed away.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

JC1_SOURCE = REPO / "experiments/ddm_jc1_carrier_pose_jacobian.py"

#: The shipped frontier archive.  Pinned; a mismatch aborts.
ARCHIVE = Path(
    "/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/retained/archive.zip"
)
ARCHIVE_SHA = "35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956"
ARCHIVE_BYTES = 181_161

#: rr4's own inflate output, retained under the rr2 encoder build.  Its sha256 equals
#: ``RESULT_receiver_parseback.json:/inflated_output/sha256`` for the pinned archive, so
#: this render IS the shipped render -- verified, not assumed.
BASE_RAW = Path(
    "/Volumes/APDataStore/pact/ddm_rr2_encoder_build/parseback/inflated/0.raw"
)
BASE_RAW_SHA = "e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9"

UPSTREAM = Path("/Volumes/APDataStore/pact/upstream_eval_mirror_20260815")

#: jc1's retained ground-truth PoseNet rows.  GT depends only on the upstream video and the
#: frozen PoseNet, never on our archive, so it is reusable across archives.  ``--gt-control``
#: re-derives a seeded random sample from the video and reports the deviation, so the reuse
#: is measured rather than argued.
JC1_GT = Path("/Volumes/APDataStore/pact/ddm_jc1/retained/pose6_groundtruth.float64.npy")

OUT = Path("/Volumes/APDataStore/pact/ddm_t1h")

N_PAIRS = 600
CAMERA_H, CAMERA_W = 874, 1164
FRAME_BYTES = CAMERA_H * CAMERA_W * 3
POSE_ROWS = 6
CARRIER_DIM = 12
SEMANTIC_WIDTH_MARKER = 40_252

#: Contest-CUDA T4 decomposition of the pinned archive (me1, verified by S arithmetic).
T4_D_POSE = 6.88e-6
UNCOMPRESSED_BYTES = 37_545_489

AXIS = "[macOS-CPU advisory, exact chain, frozen CPU-torch PoseNet]"
SEED = 1234


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_jc1():
    """Import jc1's exact-chain helpers rather than re-deriving them.

    jc1's forward is byte-identity controlled against the shipped render at n600, so
    reusing it inherits that proof instead of opening a second unverified chain.
    """
    spec = importlib.util.spec_from_file_location("jc1_chain", JC1_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["jc1_chain"] = module
    spec.loader.exec_module(module)
    return module


def decode_codes_and_scales(archive: Path, chain, torch):
    """Return (basis, codes, scales, selector_blob, provenance) for the shipped archive.

    This mirrors ``runtime/f26_inflate.py:450-480`` exactly: split the frame-0 selector off
    the carrier blob, materialize canonical CPR1, unpack the basis and coefficients, then
    rebuild the signed-int12 code lattice and apply the counted QS2 compensation overlay.
    The reconstruction guard (``coeff == codes * scales``) is the receiver's own and is kept
    -- if it fires, the mirror has diverged and no downstream number is trustworthy.
    """
    (renderer, read_archive, split_selector, materialize, apply_overlay,
     _decode_selector, _apply_pixel_mode) = chain
    parts = read_archive(archive)
    if parts.schema != "fixed_boundary_int6" or parts.token_codec != "rc64":
        raise SystemExit(f"unexpected schema {parts.schema!r}/{parts.token_codec!r}")

    carrier_blob, selector_blob = split_selector(parts.carrier_blob)
    canonical = materialize(carrier_blob, renderer)
    semantic_pose = (
        struct.pack("<II", SEMANTIC_WIDTH_MARKER, len(canonical))
        + bytes(SEMANTIC_WIDTH_MARKER)
        + canonical
    )
    _, basis, coeff = renderer.unpack_semantic_pose(semantic_pose)

    basis_count = renderer.CARRIER_DIM * 3 * renderer.CARRIER_H * renderer.CARRIER_W
    _, _, scales, encoded = renderer.decode_compact_carrier(
        canonical, basis_count=basis_count, frames=renderer.N,
        dimensions=renderer.CARRIER_DIM,
    )
    delta = (encoded.astype(np.int64) >> 1) ^ -(encoded.astype(np.int64) & 1)
    base_codes = np.cumsum(delta, axis=0) & 0xFFF
    base_codes = np.where(base_codes >= 0x800, base_codes - 0x1000, base_codes)
    base_codes = base_codes.astype(np.int32)
    expected = torch.from_numpy(base_codes).float() * torch.from_numpy(scales)[None]
    if not torch.equal(coeff, expected):
        raise SystemExit(
            "base-code reconstruction differs from the decoded coefficients -- the mirror "
            "diverges before the overlay; do not proceed"
        )

    provenance = {
        "canonical_cpr1_bytes": len(canonical),
        "selector_blob_bytes": 0 if selector_blob is None else len(selector_blob),
        "compensation_applied": False,
        "compensation_bytes": 0,
        "compensation_changed_coordinates": 0,
    }
    codes = base_codes
    if parts.compensation_blob is not None:
        codes = apply_overlay(base_codes, parts.compensation_blob)
        provenance.update(
            compensation_applied=True,
            compensation_bytes=len(parts.compensation_blob),
            compensation_changed_coordinates=int(
                np.count_nonzero(codes != base_codes)
            ),
        )
    return basis, codes.astype(np.int32), scales.astype(np.float64), selector_blob, provenance


def pose6_for_codes(jc1, torch, F, posenet, yuv6, normalized_basis, code_row, scales,
                    mode, selector_module, frame1):
    """EXACT pose rows for one pair at one integer code row.  No relaxation anywhere."""
    coeff_row = torch.from_numpy(
        (code_row.astype(np.float64) * scales).astype(np.float32)
    )[None]
    frame0 = jc1.render_frame0_differentiable(
        torch, F, normalized_basis, coeff_row, ste=False
    )
    if mode is not None:
        frame0 = jc1.apply_pixel_mode_differentiable(
            torch, frame0, mode, selector_module
        )
    with torch.no_grad():
        pose = jc1.posenet_pose6(torch, F, posenet, yuv6, frame0, frame1)
    return pose[0].detach().numpy().astype(np.float64), frame0


def gt_control(jc1, torch, F, posenet, yuv6, upstream: Path, pair_ids, retained):
    """Re-derive ground-truth PoseNet rows from the video and compare to the retained ones.

    The retained rows come from jc1, which measured a DIFFERENT archive.  That reuse is only
    legitimate because ground truth depends on the upstream video and the frozen PoseNet
    alone -- never on our archive.  That is an argument, and an argument is not a control,
    so this re-derives the sampled pairs through upstream's own ``yuv420_to_rgb`` decode
    (never PyAV rgb24, which manufactures ~100x phantom pose) and reports the deviation.
    """
    wanted = {int(p) for p in pair_ids}
    rows = []
    iterator = jc1.gt_pair_iterator(upstream, upstream / "videos" / "0.mkv")
    for pair_id, (gt0, gt1) in enumerate(iterator):
        if pair_id not in wanted:
            continue
        frames = [
            torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)[None].float()
            for frame in (gt0, gt1)
        ]
        with torch.no_grad():
            pose = jc1.posenet_pose6(torch, F, posenet, yuv6, frames[0], frames[1])
        derived = pose[0].detach().numpy().astype(np.float64)
        reference = np.asarray(retained[pair_id], dtype=np.float64)
        denominator = float(np.abs(reference).max()) or 1.0
        rows.append({
            "pair": pair_id,
            "max_abs_deviation": float(np.abs(derived - reference).max()),
            "max_rel_deviation": float(
                np.abs(derived - reference).max() / denominator
            ),
        })
        wanted.discard(pair_id)
        if not wanted:
            break
    return rows


def select_pairs(spec: str, rng: np.random.Generator) -> np.ndarray:
    """Resolve a pair selection.  ``randomN`` draws a SEEDED RANDOM sample, never a prefix.

    A contiguous prefix of this video is a different population, not a small sample of it
    (measured: pose prefixes run 2.5-4.2x harder than n600).  Sub-n600 selections here are
    therefore random by construction and labelled with their size.
    """
    if spec == "all":
        return np.arange(N_PAIRS, dtype=np.int64)
    if spec.startswith("random"):
        count = int(spec[len("random"):])
        if not 1 <= count <= N_PAIRS:
            raise SystemExit(f"random sample size out of range: {count}")
        return np.sort(rng.choice(N_PAIRS, size=count, replace=False)).astype(np.int64)
    return np.array([int(v) for v in spec.split(",")], dtype=np.int64)


def run(args) -> int:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    rng = np.random.default_rng(SEED)

    from tac.differentiable_eval_roundtrip import differentiable_rgb_to_yuv6 as yuv6

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    archive = Path(args.archive)
    archive_sha = sha256_file(archive)
    if archive_sha != ARCHIVE_SHA or archive.stat().st_size != ARCHIVE_BYTES:
        raise SystemExit(
            f"CUSTODY REFUSED: archive {archive_sha}/{archive.stat().st_size} B differs "
            f"from the pinned frontier {ARCHIVE_SHA}/{ARCHIVE_BYTES} B"
        )
    base_raw = Path(args.base_raw)
    if base_raw.stat().st_size != N_PAIRS * 2 * FRAME_BYTES:
        raise SystemExit(f"base render is {base_raw.stat().st_size} B, unexpected size")

    jc1 = load_jc1()
    chain = jc1.load_ra2b().load_chain()
    renderer = chain[0]
    selector_module = sys.modules["runtime.frame0_selector"]
    basis, codes, scales, selector_blob, provenance = decode_codes_and_scales(
        archive, chain, torch
    )
    normalized_basis = renderer.normalized_basis(basis)

    modes, sel_choices = (None, None)
    if selector_blob is not None:
        modes, sel_choices = chain[5](selector_blob)

    posenet = jc1.build_posenet(torch, Path(args.upstream))
    raw = np.memmap(
        base_raw, dtype=np.uint8, mode="r",
        shape=(2 * N_PAIRS, CAMERA_H, CAMERA_W, 3),
    )
    pose_gt_all = np.load(JC1_GT)
    if pose_gt_all.shape != (N_PAIRS, POSE_ROWS):
        raise SystemExit(f"retained GT has shape {pose_gt_all.shape}, expected (600, 6)")

    # Later passes start from an already-moved lattice.  The byte-identity control is a
    # statement about the SHIPPED codes and cannot fire once we have moved off them, so it
    # is disabled rather than quietly reinterpreted -- a control that cannot fail is worse
    # than no control.  Pass 1 is what proves the chain; later passes inherit that proof.
    on_shipped_lattice = True
    if args.start_codes:
        start = np.load(Path(args.start_codes))
        if start.shape != (N_PAIRS, CARRIER_DIM):
            raise SystemExit(f"start codes have shape {start.shape}, expected (600, 12)")
        codes = start.astype(np.int32)
        on_shipped_lattice = False

    pairs = select_pairs(args.pairs, rng)
    deltas = [int(v) for v in args.deltas.split(",") if int(v) != 0]

    if args.gt_control:
        control_pairs = np.sort(
            rng.choice(N_PAIRS, size=min(args.gt_control, N_PAIRS), replace=False)
        )
        rows = gt_control(
            jc1, torch, F, posenet, yuv6, Path(args.upstream), control_pairs, pose_gt_all
        )
        worst = max((r["max_abs_deviation"] for r in rows), default=float("nan"))
        receipt = {
            "schema": "ddm_t1h_gt_reuse_control.v1",
            "axis": AXIS,
            "score_claim": False,
            "retained_gt_path": str(JC1_GT),
            "pairs_checked": [int(p) for p in control_pairs],
            "worst_max_abs_deviation": worst,
            "rows": rows,
            "note": (
                "Ground truth depends only on the upstream video and the frozen PoseNet, "
                "so it is archive-independent; this measures that rather than asserting it."
            ),
        }
        (out / (args.receipt or "T1H_GT_CONTROL.json")).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0

    started = time.time()
    rows = []
    control_failures = 0
    forwards = 0
    for slot, pair_id in enumerate(pairs):
        pair_id = int(pair_id)
        frame1 = torch.from_numpy(
            np.asarray(raw[2 * pair_id + 1]).copy()
        ).permute(2, 0, 1)[None].float()
        mode = None
        if modes is not None:
            mode = modes[int(sel_choices[pair_id])]
            if mode.kind == selector_module.IDENTITY:
                mode = None

        base_pose, base_frame0 = pose6_for_codes(
            jc1, torch, F, posenet, yuv6, normalized_basis,
            codes[pair_id], scales, mode, selector_module, frame1,
        )
        forwards += 1
        hard = base_frame0[0].round().clamp(0, 255).to(torch.uint8)
        hard_np = hard.permute(1, 2, 0).numpy()
        identical = bool(np.array_equal(hard_np, np.asarray(raw[2 * pair_id])))
        if not identical and on_shipped_lattice:
            control_failures += 1
            if control_failures > args.control_abort_after:
                raise SystemExit(
                    f"CONTROL FAILED on pair {pair_id}: rendered frame_0 differs from the "
                    "shipped 0.raw.  The chain is not the shipped chain; every number "
                    "downstream would be fiction."
                )

        residual = base_pose - pose_gt_all[pair_id]
        base_energy = float((residual ** 2).sum())

        best = {"coord": -1, "delta": 0, "energy": base_energy}
        per_coord = []
        for coord in range(CARRIER_DIM):
            coord_best = base_energy
            coord_arg = 0
            for step in deltas:
                trial = codes[pair_id].copy()
                value = int(trial[coord]) + step
                if not -2048 <= value <= 2047:
                    continue
                trial[coord] = value
                pose, _ = pose6_for_codes(
                    jc1, torch, F, posenet, yuv6, normalized_basis,
                    trial, scales, mode, selector_module, frame1,
                )
                forwards += 1
                energy = float(((pose - pose_gt_all[pair_id]) ** 2).sum())
                if energy < coord_best:
                    coord_best, coord_arg = energy, step
            per_coord.append({"coord": coord, "best_delta": coord_arg,
                              "energy": coord_best})
            if coord_best < best["energy"]:
                best = {"coord": coord, "delta": coord_arg, "energy": coord_best}

        rows.append({
            "pair": pair_id,
            "control_byte_identical": identical,
            "base_energy": base_energy,
            "best_energy": best["energy"],
            "best_coord": best["coord"],
            "best_delta": best["delta"],
            "per_coord": per_coord,
        })
        if args.report_every and (slot + 1) % args.report_every == 0:
            done = slot + 1
            rate = (time.time() - started) / done
            print(
                f"[t1h] {done}/{len(pairs)} pairs  {forwards} forwards  "
                f"{rate:.1f} s/pair  eta {(len(pairs) - done) * rate / 60:.1f} min",
                flush=True,
            )

    base_energies = np.array([r["base_energy"] for r in rows], dtype=np.float64)
    best_energies = np.array([r["best_energy"] for r in rows], dtype=np.float64)
    base_d_pose = float(base_energies.mean() / POSE_ROWS)
    best_d_pose = float(best_energies.mean() / POSE_ROWS)
    improved = int((best_energies < base_energies).sum())

    ratio = best_d_pose / base_d_pose if base_d_pose > 0 else float("nan")
    t4_pose_term = float(np.sqrt(10.0 * T4_D_POSE))
    t4_pose_term_after = float(np.sqrt(10.0 * T4_D_POSE * ratio))
    delta_s_ratio_transfer = t4_pose_term_after - t4_pose_term

    receipt = {
        "schema": "ddm_t1h_pose_coeff_headroom.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "archive": {"path": str(archive), "sha256": archive_sha,
                    "bytes": ARCHIVE_BYTES},
        "base_raw": {"path": str(base_raw), "sha256_expected": BASE_RAW_SHA},
        "carrier_provenance": provenance,
        "pairs_measured": len(pairs),
        "pairs_selection": args.pairs,
        "deltas_swept": deltas,
        "posenet_forwards": forwards,
        "elapsed_seconds": time.time() - started,
        "control": {
            "on_shipped_lattice": on_shipped_lattice,
            "byte_identical_pairs": int(sum(r["control_byte_identical"] for r in rows)),
            "failures": control_failures,
            "note": (
                "byte-identity vs the shipped 0.raw" if on_shipped_lattice else
                "NOT APPLICABLE: this pass starts from a moved lattice, so the shipped "
                "render is not the expected output.  Chain validity is inherited from the "
                "pass that ran on the shipped codes."
            ),
        },
        "base_d_pose_cpu_torch": base_d_pose,
        "single_coord_optimal_d_pose_cpu_torch": best_d_pose,
        "d_pose_ratio_after_over_before": ratio,
        "pairs_improved": improved,
        "pairs_improved_fraction": improved / max(len(pairs), 1),
        "headroom_note": (
            "single-coordinate, integer, distortion-only.  Byte cost of re-encoding the "
            "delta-coded carrier is NOT included and would offset part of any gain."
        ),
        "t4_ratio_transfer": {
            "t4_base_d_pose": T4_D_POSE,
            "t4_pose_term_before": t4_pose_term,
            "t4_pose_term_after_if_ratio_transfers": t4_pose_term_after,
            "delta_S_if_ratio_transfers": delta_s_ratio_transfer,
            "caveat": (
                "The ratio is measured on CPU-torch.  Transferring it to the T4 axis "
                "assumes the improvement is a proportional one, which is an assumption, "
                "not a measurement."
            ),
        },
        "rows": rows,
    }
    name = args.receipt or "T1H_HEADROOM.json"
    (out / name).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    summary = {k: v for k, v in receipt.items() if k != "rows"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--base-raw", type=Path, default=BASE_RAW)
    parser.add_argument("--upstream", type=Path, default=UPSTREAM)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--receipt", type=str, default=None)
    parser.add_argument(
        "--pairs", type=str, default="random48",
        help="'all', 'randomN' (seeded, never a prefix), or an explicit comma list",
    )
    parser.add_argument(
        "--deltas", type=str, default="-1,1",
        help="integer code steps swept per coordinate",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--report-every", type=int, default=4)
    parser.add_argument("--control-abort-after", type=int, default=0)
    parser.add_argument(
        "--gt-control", type=int, default=0,
        help="re-derive this many random GT pairs from the video and compare, then exit",
    )
    parser.add_argument(
        "--start-codes", type=Path, default=None,
        help="(600, 12) int32 .npy lattice to sweep from, for passes after the first",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
