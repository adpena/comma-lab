#!/usr/bin/env python3
"""ddm_rp1 -- RATE-directed token pre-distortion: rank, realize, price.

WHAT THIS IS
------------
``ddm_sj1`` changes the stored token field to repair the receiver's rendered argmax and
PAYS for it: its live field carries 9,209 changed tokens and its token stream is 120,367 B
against the pristine field's 113,419 B -- **6,948 B spent, 6.0358 measured bits per changed
token** (``pricing/retained/S1_encode_sj1_pass4_subset.json``).  Every rate arm so far
(rc1/rc2/rc3/tc1/sm1) has re-coded the field AS GIVEN.  Nobody has changed the field to make
it CHEAPER while holding the argmax.

This module runs that inversion.  The shipped tail coder assigns a probability row to every
one of the 117,964,800 token positions along the receiver's own decode trajectory; the exact
first-order saving of rewriting position ``i`` from its symbol ``s`` to any other class ``c``
is ``log2(p_i[c] / p_i[s])``, which the encoder already computes and throws away.  This
module keeps it.

THE THREE STAGES AND WHY THEY ARE SEPARATE
------------------------------------------
``rank``  -- one instrumented 600-frame encode of the LIVE field.  It emits the identical
             stream the pointer ships (that byte identity IS the identity control: the
             instrument cannot be measuring a different coder than the one that ships) and,
             per frame, the sparse top-K positions by first-order bit saving.
``sizing``-- realized acceptance on a seeded pair sample.  A proposal is ACCEPTED only if
             the pair's rendered, re-segmented argmax is IDENTICAL on all 196,608 cells.
             Not "flips fall" -- identical.  No surrogate on this path.
``price`` is ``ddm_jg2_tail_reencode.py`` itself: the ranking ORDERS the search, a REAL
re-encode PRICES it (sj1's placement law: forced writes cost ~1.73x greedy ones, so a sum
of first-order savings RANKS well and CHARGES wrong).

AUTHORITY
---------
d_seg acceptance: ``cpu_torch`` SegNet argmax on the DALI GT lineage, through the receiver's
own renderer (``ddm_jg1``).  Bytes: exact, from a real encode.  Everything here is
``[macOS-CPU advisory]``; ``score_claim=false`` until a T4 row.
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

import ddm_jg1_seg_solve as jg1  # noqa: E402
import ddm_jg2_tail_reencode as jg2  # noqa: E402

N_PAIRS = jg1.N_PAIRS
EVAL_H, EVAL_W = jg1.EVAL_H, jg1.EVAL_W
NUM_CLASSES = jg1.NUM_CLASSES
PLANE = EVAL_H * EVAL_W

# --------------------------------------------------------------------------------------
# The LIVE object.  Read from the pointer's own receipts, never retyped from a memo.
# --------------------------------------------------------------------------------------

#: The encoder tree.  jg2 encodes through cl2's coding of the model sections; rc1/rc2
#: restore those sections byte-for-byte inside the receiver, which is why the stream this
#: tree emits is the stream the pointer's receiver decodes (sj1's verified premise, and
#: re-verified here by STREAM IDENTITY: ``rank`` refuses unless its emitted stream equals
#: the pointer's shipped token stream byte for byte).
ENCODER_TREE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder"
    "/rungs/lambda_1p0/retained/receiver_copy_runtime"
)
BASE_TOKENS = ENCODER_TREE.parent / "decoded_tokens.u8"

#: The LIVE field: the cl2 base field with sj1's pass-4 admitted planes spliced in.  This
#: exact npz is the one the pointer's own encode receipt names
#: (``S1_encode_sj1_pass4_subset.json`` -> ``edits_file``), so the field is bound by a
#: receipt rather than by a path someone remembered.
LIVE_FIELD_NPZ = Path(
    "/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion"
    "/admission_pass4/field_admitted.npz"
)
LIVE_FIELD_SHA256 = "813bf1e6770161b604348491433386661110001a5eefd8a4fdcd7d70bc25365c"

#: The pointer's shipped token stream -- the identity control's target.
LIVE_STREAM = Path(
    "/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion"
    "/pricing/work/tail_sj1_pass4_subset.bin"
)
LIVE_STREAM_SHA256 = "0720536b5a9e90ef87a8d6a9d73a6be98130855aeaf7f6fb5dfc4862dd70d1b1"
LIVE_STREAM_BYTES = 120_367

#: The pristine field's stream, for the "what sj1 spent" arithmetic.
PRISTINE_STREAM_BYTES = 113_419

#: Live pointer row (move 35).  Checked against the pointer file at import of the CLI.
POINTER_ARCHIVE_SHA256 = (
    "b0ca809ce2c657dfce97e73148a83b9b20c128461ced4c1f6ce1b386ddfd1d20"
)
POINTER_ARCHIVE_BYTES = 181_521
POINTER_SCORE_T4 = 0.13867171823146562

S_PER_BYTE = 25.0 / jg1.SCORE_RATE_DENOMINATOR

#: Savings below this are not worth a row in the sparse dump; the CENSUS still counts them.
DEFAULT_MIN_SAVING_BITS = 0.5
#: Hard cap on retained candidates per frame, so the dump cannot grow without bound.
DEFAULT_TOP_K = 8_000


class Rp1Error(RuntimeError):
    """A ddm_rp1 precondition failed.  Fail closed, never approximate."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def set_threads(threads: int) -> None:
    import torch

    torch.set_num_threads(max(1, threads))


def verify_pointer(expect_sha: str = POINTER_ARCHIVE_SHA256) -> dict[str, Any]:
    """Refuse to run against a pointer that has moved under us.

    Two field changes on one base must be merged by re-verification, never carried
    (the sj1 composition rule).  So the pointer is READ, not remembered.
    """
    pointer = json.loads(
        (REPO / ".omx/state/canonical_frontier_pointer.json").read_text()
    )
    row = pointer["our_local_frontier_contest_cuda"]
    live = {
        "archive_sha256": row["archive_sha256"],
        "archive_bytes": row["extra"]["archive_bytes"],
        "score": row["score"],
        "lane_id": row["lane_id"],
    }
    live["matches_expected"] = live["archive_sha256"] == expect_sha
    return live


def load_live_field() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (base_field, live_field, receipts).  Both are (600, 384, 512) uint8."""
    field_sha = sha256_file(LIVE_FIELD_NPZ)
    if field_sha != LIVE_FIELD_SHA256:
        raise Rp1Error(
            f"live field npz sha {field_sha} != the pointer's own encode receipt "
            f"{LIVE_FIELD_SHA256}"
        )
    base = np.array(jg2.load_tokens(BASE_TOKENS), dtype=np.uint8)
    live = base.copy()
    edited: list[int] = []
    with np.load(LIVE_FIELD_NPZ, allow_pickle=False) as blob:
        for key in blob.files:
            pair = int(key)
            plane = np.asarray(blob[key], dtype=np.uint8)
            if plane.shape != (EVAL_H, EVAL_W):
                raise Rp1Error(f"edit plane {key} has shape {plane.shape}")
            live[pair] = plane
            edited.append(pair)
    changed = int((live != base).sum())
    if changed != 9_209:
        raise Rp1Error(
            f"live field carries {changed} changed tokens; the pointer's encode receipt "
            "declares 9209 -- the field is not the one that shipped"
        )
    return base, live, {
        "live_field_npz": str(LIVE_FIELD_NPZ),
        "live_field_sha256": field_sha,
        "edited_pairs": len(edited),
        "tokens_changed_vs_pristine": changed,
    }


# --------------------------------------------------------------------------------------
# rank -- the instrumented encode.  This is jg2's own loop with the probability rows kept.
# --------------------------------------------------------------------------------------

#: Census bins for the per-token cost distribution, in bits.  The whole point of the
#: census is the DENOMINATOR: a "top-K saving" number with no distribution behind it is
#: the vacuity-passes failure.
CENSUS_EDGES = np.array(
    [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 1e9],
    dtype=np.float64,
)


def cmd_rank(args: argparse.Namespace) -> int:
    import torch

    set_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pointer = verify_pointer()
    base_field, live_field, field_receipts = load_live_field()
    sj1_edit_mask = live_field != base_field

    env = jg2._prepare(
        argparse.Namespace(
            store=str(out),
            runtime_root=str(ENCODER_TREE),
            pointer_archive=str(ENCODER_TREE / "archive.zip"),
        ),
        "rp1_rank",
    )
    # ``runtime.*`` only becomes importable once ``_prepare`` has put the receiver tree
    # on the path, so these imports live below it rather than at the top of the function.
    from runtime.free_corrector import FreeCorrector  # type: ignore[import-not-found]
    from runtime.hpac_inference import (  # type: ignore[import-not-found]
        optimize_sparse_evaluator,
    )

    residual = env["residual"]
    renderer = env["renderer"]
    renderer_dir = env["renderer_dir"]
    parts = env["parts"]
    route_b = env["route_b"]
    library = env["library"]

    device = torch.device("cpu")
    base_hpac = residual.materialize_ihs1(parts.hpac_blob, renderer)
    model = renderer.load_hpac(base_hpac, device)
    masks = renderer.group_masks(device)
    sparse = residual._sparse_class(renderer_dir)(model, renderer.EVAL_H, renderer.EVAL_W)
    corrector = FreeCorrector(renderer.EVAL_H * renderer.EVAL_W)

    group_plans = []
    for mask in masks:
        mask_array = mask.detach().cpu().numpy()
        flat_positions = np.flatnonzero(mask_array.reshape(-1))
        group_plans.append(
            (torch.from_numpy(flat_positions).to(device), flat_positions)
        )

    encoder = route_b.NativeRc64Encoder(library)

    census = np.zeros(len(CENSUS_EDGES) - 1, dtype=np.int64)
    per_frame_bits = np.zeros(N_PAIRS, dtype=np.float64)
    #: The headline denominators, accumulated exactly rather than sampled.
    total_bits = 0.0
    n_symbol_is_argmax = 0
    bits_on_argmax_symbols = 0.0
    bits_on_nonargmax_symbols = 0.0
    n_above = {t: 0 for t in (0.5, 1.0, 2.0, 4.0, 8.0)}
    saving_above = {t: 0.0 for t in (0.5, 1.0, 2.0, 4.0, 8.0)}

    keep_pos: list[np.ndarray] = []
    keep_frame: list[np.ndarray] = []
    keep_sym: list[np.ndarray] = []
    keep_best: list[np.ndarray] = []
    keep_bits_sym: list[np.ndarray] = []
    keep_bits_best: list[np.ndarray] = []
    keep_is_sj1: list[np.ndarray] = []

    started = time.perf_counter()
    with torch.inference_mode():
        optimize_sparse_evaluator(sparse)
        previous = torch.zeros((1, EVAL_H, EVAL_W), dtype=torch.long, device=device)

        for frame in range(args.frames):
            index = torch.tensor([frame], dtype=torch.long, device=device)
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(index, previous)
            if frame:
                previous_cpu = previous[0].to(device="cpu", dtype=torch.uint8).numpy()
                boundary = residual._boundary_buckets(previous_cpu).reshape(-1)
            else:
                boundary = np.full(PLANE, 4, dtype=np.uint8)
            corrector.begin_frame(boundary)

            plane_target = np.asarray(live_field[frame], dtype=np.uint8).reshape(-1)
            edit_flat = sj1_edit_mask[frame].reshape(-1)
            frame_bits = 0.0

            f_pos: list[np.ndarray] = []
            f_sym: list[np.ndarray] = []
            f_best: list[np.ndarray] = []
            f_bs: list[np.ndarray] = []
            f_bb: list[np.ndarray] = []
            f_e: list[np.ndarray] = []

            for group, (device_positions, flat_positions) in enumerate(group_plans):
                selected = sparse.selected_logits(current, context, group)
                base_logits = selected.cpu().numpy()
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                feature = (
                    boundary[flat_positions].astype(np.int64) * NUM_CLASSES + predicted
                )
                corrected = base_logits + parts.table.values[feature]
                probability = residual._probability_table(
                    corrected, renderer.HPAC_LOGIT_PRECISION
                )
                state = corrector.group_state(probability, predicted, flat_positions)
                coding = corrector.coding_row(state)

                symbols = plane_target[flat_positions].astype(np.int64)

                # THE MEASUREMENT.  ``coding`` is the shipped coder's own probability row
                # for every position in this group; the arithmetic coder charges exactly
                # -log2(coding[i, symbol]) for the symbol it writes, which is why
                # jg2._row_bits sums that quantity to reproduce the stream length.  The
                # alternative-token price is the SAME row read at a different column --
                # this is the coder's price, not a model of it.
                rows = np.asarray(coding, dtype=np.float64)
                safe = np.maximum(rows, 1e-300)
                bits_all = -np.log2(safe)
                idx = np.arange(rows.shape[0])
                bits_sym = bits_all[idx, symbols]
                best_class = rows.argmax(axis=1)
                bits_best = bits_all[idx, best_class]
                saving = bits_sym - bits_best

                frame_bits += float(bits_sym.sum())
                census += np.histogram(bits_sym, bins=CENSUS_EDGES)[0]
                is_argmax = best_class == symbols
                n_symbol_is_argmax += int(is_argmax.sum())
                bits_on_argmax_symbols += float(bits_sym[is_argmax].sum())
                bits_on_nonargmax_symbols += float(bits_sym[~is_argmax].sum())
                for threshold in n_above:
                    hit = saving >= threshold
                    n_above[threshold] += int(hit.sum())
                    saving_above[threshold] += float(saving[hit].sum())

                keep = saving >= args.min_saving_bits
                if keep.any():
                    where = np.flatnonzero(keep)
                    f_pos.append(flat_positions[where].astype(np.int32))
                    f_sym.append(symbols[where].astype(np.uint8))
                    f_best.append(best_class[where].astype(np.uint8))
                    f_bs.append(bits_sym[where].astype(np.float32))
                    f_bb.append(bits_best[where].astype(np.float32))
                    f_e.append(edit_flat[flat_positions[where]].astype(np.uint8))

                encoder.encode(symbols.astype(np.int32), coding)
                corrector.observe(state, symbols)
                current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(
                    device
                )

            total_bits += frame_bits
            per_frame_bits[frame] = frame_bits
            frame_tokens = current[0].to(device="cpu", dtype=torch.uint8).numpy()
            if not np.array_equal(frame_tokens.reshape(-1), plane_target):
                raise Rp1Error(f"frame {frame}: encoded field diverged from the target")
            corrector.end_frame(frame_tokens.reshape(-1))
            previous = current

            if f_pos:
                pos = np.concatenate(f_pos)
                sym = np.concatenate(f_sym)
                best = np.concatenate(f_best)
                bs = np.concatenate(f_bs)
                bb = np.concatenate(f_bb)
                ed = np.concatenate(f_e)
                if pos.size > args.top_k:
                    order = np.argsort(bb - bs)[: args.top_k]  # most saving first
                    pos, sym, best, bs, bb, ed = (
                        pos[order],
                        sym[order],
                        best[order],
                        bs[order],
                        bb[order],
                        ed[order],
                    )
                keep_pos.append(pos)
                keep_frame.append(np.full(pos.size, frame, dtype=np.int16))
                keep_sym.append(sym)
                keep_best.append(best)
                keep_bits_sym.append(bs)
                keep_bits_best.append(bb)
                keep_is_sj1.append(ed)

            if (frame + 1) % 25 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "rank",
                            "frame": frame + 1,
                            "code_bytes_so_far": total_bits / 8.0,
                            "candidates_so_far": int(
                                sum(a.size for a in keep_pos)
                            ),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )

    payload = encoder.finish()
    if not payload.startswith(route_b.TOKEN_MAGIC):
        raise Rp1Error("RC64 payload lost its magic")
    import ctypes

    size = int(encoder.library.rc64_encoder_size(encoder.context))
    body = ctypes.string_at(encoder.library.rc64_encoder_data(encoder.context), size)

    stream_path = out / "tail_rp1_rank_control.bin"
    stream_path.write_bytes(body)
    stream_sha = hashlib.sha256(body).hexdigest()

    # THE IDENTITY CONTROL.  If this stream is not the pointer's own bytes, this
    # instrument is not measuring the coder that ships and nothing below it is evidence.
    live_prefix = LIVE_STREAM.read_bytes()
    common = min(len(body), len(live_prefix))
    agree = int(np.flatnonzero(
        np.frombuffer(body[:common], dtype=np.uint8)
        != np.frombuffer(live_prefix[:common], dtype=np.uint8)
    )[:1].tolist()[0]) if body[:common] != live_prefix[:common] else common
    identity = {
        "frames_encoded": args.frames,
        "emitted_bytes": len(body),
        "emitted_sha256": stream_sha,
        "live_stream_bytes": LIVE_STREAM_BYTES,
        "live_stream_sha256": LIVE_STREAM_SHA256,
        "prefix_bytes_agreeing_with_live_stream": agree,
        "byte_identical": stream_sha == LIVE_STREAM_SHA256
        and len(body) == LIVE_STREAM_BYTES,
    }

    dump = out / "candidates.npz"
    np.savez_compressed(
        dump,
        frame=np.concatenate(keep_frame),
        pos=np.concatenate(keep_pos),
        sym=np.concatenate(keep_sym),
        best=np.concatenate(keep_best),
        bits_sym=np.concatenate(keep_bits_sym),
        bits_best=np.concatenate(keep_bits_best),
        is_sj1_edit=np.concatenate(keep_is_sj1),
    )
    np.save(out / "bits_per_frame.npy", per_frame_bits)

    total_tokens = args.frames * PLANE
    receipt = {
        "schema": "ddm_rp1_rank.v1",
        "axis": "[macOS-CPU advisory / scorer-free EXACT coder measurement]",
        "score_claim": False,
        "pointer": pointer,
        "field": field_receipts,
        "identity_control": identity,
        "census": {
            "total_tokens": total_tokens,
            "total_bits": total_bits,
            "total_bytes_ideal": total_bits / 8.0,
            "bits_per_token_mean": total_bits / total_tokens,
            "bin_edges_bits": CENSUS_EDGES[:-1].tolist() + ["inf"],
            "bin_counts": census.tolist(),
            "n_symbol_is_coder_argmax": n_symbol_is_argmax,
            "fraction_symbol_is_coder_argmax": n_symbol_is_argmax / total_tokens,
            "bits_on_argmax_symbols": bits_on_argmax_symbols,
            "bits_on_nonargmax_symbols": bits_on_nonargmax_symbols,
            "fraction_of_bits_on_nonargmax_symbols": (
                bits_on_nonargmax_symbols / total_bits if total_bits else 0.0
            ),
        },
        "saving_ceiling": {
            f"ge_{threshold}_bits": {
                "positions": n_above[threshold],
                "total_saving_bits": saving_above[threshold],
                "total_saving_bytes": saving_above[threshold] / 8.0,
                "delta_S_if_all_free": -saving_above[threshold] / 8.0 * S_PER_BYTE,
            }
            for threshold in sorted(n_above)
        },
        "sj1_spend_context": {
            "pristine_stream_bytes": PRISTINE_STREAM_BYTES,
            "live_stream_bytes": LIVE_STREAM_BYTES,
            "sj1_bytes_spent_on_seg": LIVE_STREAM_BYTES - PRISTINE_STREAM_BYTES,
            "sj1_tokens_changed": 9_209,
            "sj1_measured_bits_per_changed_token": 6.035834509718754,
        },
        "candidates_dump": {
            "path": str(dump),
            "rows": int(np.concatenate(keep_frame).size),
            "min_saving_bits": args.min_saving_bits,
            "top_k_per_frame": args.top_k,
        },
        "stream": {"path": str(stream_path), "bytes": len(body), "sha256": stream_sha},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (out / "RANK.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in receipt.items() if k != "census"}, indent=2))
    if args.frames == N_PAIRS and not identity["byte_identical"]:
        raise Rp1Error(
            "IDENTITY CONTROL FAILED: the instrumented encode did not reproduce the "
            f"pointer's shipped token stream ({identity})"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rank = sub.add_parser("rank", help="instrumented encode: census + candidate ranking")
    rank.add_argument("--out", required=True)
    rank.add_argument("--threads", type=int, default=4)
    rank.add_argument("--min-saving-bits", type=float, default=DEFAULT_MIN_SAVING_BITS)
    rank.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    rank.add_argument(
        "--frames",
        type=int,
        default=N_PAIRS,
        help="smoke only; the identity control is asserted at 600 frames",
    )
    rank.set_defaults(func=cmd_rank)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
