"""ddm_cb1 -- re-FIT the pose carrier's twelve basis atoms to the CURRENT pose-residual
subspace, at FIXED rank, FIXED precision and FIXED lattice, priced on move 49.

What this arm changes and what it does not
------------------------------------------
CHANGED: the 27,648 stored five-bit basis CODES (data, `pc3` section 4).
UNCHANGED: `CARRIER_DIM`, `BASIS_BITS`, `COEFFICIENT_BITS`, the 24x32 band-limit, the
rider set, the renderer, the token field -- all receiver code or another object.

The instrument is not this arm's own
------------------------------------
* move 49's carrier state and its OWN cold parse-back come from ``ddm_pp1`` (its
  ``build_pose_instrument`` / ``open_live_raw``).  ``ddm_sj1_joint_admission``'s
  ``load_pose_instrument(None)`` is NOT used: its ``BODY_RAW`` default still points at
  the cl2 parse-back, and MEASURED here, reading it makes pair 0 score 1.275e-03
  against move 49's true 4.432e-07 (2,878x).  It is correct for that module -- every
  sj1 chain passes an overlay -- and wrong for a base measurement.
* the span geometry (`low_basis`, `span_gram`, `min_image_norm_step`, `realize`,
  `Instrument`, `evaluate_codes`, `sample_pairs`) is ``ddm_br1``'s.
* the per-pair solve is ``ddm_jg5.refine_pair`` verbatim.
* the exact price is ``ddm_up3_carrier_splice``'s ``parse_shipped_body`` /
  ``build_archive``, with the basis fields replaced through ``dataclasses.replace``.
* the basis section's byte price goes through the shipped chain: canonical Huffman
  lengths from the histogram, then the RR5 adaptive-arithmetic rider, exactly as the
  receiver reads them.

Prior law this arm must not launder (read at source, recorded because the charter did
not carry it)
--------------------------------------------------------------------------------
``ddm_pc1_pose_carrier_efficiency_20260905.md`` priced THREE span changes on this
carrier and refused all three:
  * V4 generated separable-DCT rank-12 basis, full re-solve: d_pose >= 0.9986 --
    39,748x past break-even.  ``verdict_scope: FAMILY``, and it explicitly does NOT
    close non-DCT or learned families.
  * V2 basis 5->3 bits at the fidelity-optimal per-atom step (per-atom cosine
    0.970-0.992), full ``jg5.refine_pair`` re-solve warm-started from the
    least-squares projection: 35 of 48 pairs got WORSE, population lower bound
    d_pose >= 2.85e-05 against a 6.13e-06 base.  ``verdict_scope: INSTANCE``.
  * V5 learned rank-8 SVD basis: dominated on rate arithmetic, never solved.
    ``verdict_scope: FORMULATION``.
None of the three is a fixed-rank LEARNED refit, so none closes this object -- but V2
is the sharpest prior on the board: a ~1-3% span perturbation, fully re-solved, cost
4.6x.  The mechanism pc1 named is that the carrier's bytes buy a POSITION (a reachable
box containing each pair's PoseNet solution), not a spectrum.  This arm therefore
prices the population LOWER BOUND on a seeded-random subset FIRST, because d_pose is a
mean of 600 non-negative values and a subset sum / 600 refutes on the whole
population without an n600 run.

Axis
----
`[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT, n600]` for pose;
bytes EXACT through the shipped container.  No score is claimed; MAIN fires.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_br1_pose_basis_reorientation as br1
import ddm_jg5_pose_resolve_on_edited_renders as jg5
import ddm_pp1_pose_actuation as pp1
import ddm_sj1_multipass_token_predistortion as sj1
import ddm_up2_shipping_pose_solve as up2

N_PAIRS = up2.N_PAIRS_TOTAL
CARRIER_DIM = up2.CARRIER_DIM
CARRIER_H, CARRIER_W = 24, 32
BASIS_PLANES = 3
BASIS_SYMBOLS = CARRIER_DIM * BASIS_PLANES * CARRIER_H * CARRIER_W  # 27,648
#: the five-bit signed domain the receiver's zigzag alphabet covers (carrier_codec:
#: ALPHABET_SIZE = 32, symbol = zigzag(code)).
BASIS_CODE_MIN, BASIS_CODE_MAX = -16, 15

POINTER = sj1.LIVE_POINTER
POINTER_TREE = POINTER.tree
POINTER_ARCHIVE = POINTER.archive
POINTER_ARCHIVE_SHA256 = POINTER.archive_sha256
POINTER_ARCHIVE_BYTES = POINTER.archive_bytes
POINTER_SCORE_T4 = POINTER.score_t4
POINTER_D_SEG_T4 = POINTER.d_seg_t4
POINTER_D_POSE_T4 = POINTER.d_pose_t4

#: pp1's own n600 base on move 49 -- the vector this arm must REPRODUCE, not inherit.
PP1_BASE_VECTOR = Path(
    "/Volumes/APDataStore/pact/ddm_pp1/base/pose_base_move49.npy"
)
PP1_BASE_MEAN = 4.543568679770593e-06
#: pp1 section 10, MEASURED: batch-1 vs batch-8 moves the per-pair pose vector by at
#: most 2.586e-09 ABSOLUTE over 16 pairs; the gate is 10x that observed maximum.
POSE_BATCH_BAND_ABS = 2.586e-09
POSE_BASE_GATE_ABS = 10.0 * POSE_BATCH_BAND_ABS

#: the score arithmetic, recomputed from components every time.
RATE_PER_BYTE = 25.0 / 37_545_489.0
ADMIT_BAR = -2e-05

WORK = Path(os.environ.get("CB1_WORK", "/Volumes/VertigoDataTier/pact/ddm_cb1"))


class Cb1Error(RuntimeError):
    """A ddm_cb1 precondition failed. Fail closed; never approximate."""


# ---------------------------------------------------------------------------
# identity + housekeeping
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_threads(threads: int) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    import torch

    torch.set_num_threads(threads)


def assert_pointer() -> dict[str, Any]:
    """Refuse any stage that is not standing on move 49's own shipped bytes."""
    observed = sha256_file(POINTER_ARCHIVE)
    if observed != POINTER_ARCHIVE_SHA256:
        raise Cb1Error(
            f"pointer tree {POINTER_TREE} has archive sha {observed}, not move 49's "
            f"{POINTER_ARCHIVE_SHA256}"
        )
    size = POINTER_ARCHIVE.stat().st_size
    if size != POINTER_ARCHIVE_BYTES:
        raise Cb1Error(f"pointer archive is {size} B, not {POINTER_ARCHIVE_BYTES} B")
    up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=up2.LINEAGE_DALI)
    return {
        "pointer_tree": str(POINTER_TREE),
        "archive_sha256": observed,
        "archive_bytes": size,
        "score_t4": POINTER_SCORE_T4,
        "d_seg_t4": POINTER_D_SEG_T4,
        "d_pose_t4": POINTER_D_POSE_T4,
        "gt_lineage": up2.LINEAGE_DALI,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable))


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


# ---------------------------------------------------------------------------
# the instrument, with a swappable basis
# ---------------------------------------------------------------------------


def load_instrument(basis_codes: np.ndarray | None = None):
    """br1's Instrument on MOVE 49's carrier and MOVE 49's own cold decode.

    ``basis_codes`` replaces the 27,648 stored five-bit codes; the receiver's own
    ``normalized_basis`` is then re-run so every downstream number is produced by the
    shipped render path rather than by a look-alike.
    """
    assert_pointer()
    raw = pp1.open_live_raw()
    inst = pp1.build_pose_instrument(raw)
    if basis_codes is None:
        return inst
    return with_basis(inst, basis_codes)


def with_basis(inst, basis_codes: np.ndarray):
    """A sister Instrument whose basis is ``basis_codes`` and nothing else differs."""
    import torch

    codes = np.asarray(basis_codes, dtype=np.int64)
    if codes.shape != (CARRIER_DIM, BASIS_PLANES, CARRIER_H, CARRIER_W):
        raise Cb1Error(f"basis codes must be (12, 3, 24, 32), got {codes.shape}")
    if codes.min() < BASIS_CODE_MIN or codes.max() > BASIS_CODE_MAX:
        raise Cb1Error(
            f"basis codes escape the shipped signed 5-bit domain "
            f"[{BASIS_CODE_MIN}, {BASIS_CODE_MAX}]: "
            f"[{int(codes.min())}, {int(codes.max())}]"
        )
    renderer = inst.state.renderer
    #: basis_scales are GAUGE -- ``normalized_basis`` divides each atom by its own RMS,
    #: so a positive per-atom scale cancels exactly (pc1 section 5).  Move 49 ships
    #: 1.0 on all twelve; keeping them there leaves the 96-byte scales block's basis
    #: half byte-identical, which is one fewer thing a byte delta can be attributed to.
    basis_raw = torch.from_numpy(codes.astype(np.float32))
    basis_norm = renderer.normalized_basis(basis_raw.clone())
    state = dataclasses.replace(
        inst.state, basis_raw=basis_raw, basis_norm=basis_norm
    )
    blow = br1.low_basis(state)
    gram, bmat = br1.span_gram(blow)
    return br1.Instrument(state, inst.raw, inst.targets, inst.posenet, blow, gram, bmat)


def shipped_basis_codes(inst) -> np.ndarray:
    """The shipped five-bit codes, read back out of ``basis_raw``.

    Move 49 ships ``basis_scales == 1.0``, asserted here rather than assumed, so the
    raw tensor IS the code tensor.
    """
    raw = inst.state.basis_raw.numpy()
    codes = np.rint(raw).astype(np.int64)
    if not np.array_equal(raw, codes.astype(np.float32)):
        raise Cb1Error(
            "move 49's basis_raw is not integral: basis_scales are not 1.0 and this "
            "arm's gauge assumption does not hold"
        )
    return codes


def field_of(inst, codes_row: np.ndarray):
    """The realized low-res frame-0 field for one pair, as the receiver builds it."""
    import torch

    coeff = up2.codes_to_coefficients(
        np.asarray(codes_row, dtype=np.int32)[None], inst.state.coefficient_scales
    )
    return (
        torch.einsum("bk,kchw->bchw", coeff.double(), inst.blow) / math.sqrt(CARRIER_DIM)
    )[0]


def field_jacobian(inst, pair: int):
    """(J, residual, field) for one pair at the 3x24x32 band-limit.

    Shape-identical to ``br1.ceiling_pair``'s first iteration: the field is the free
    24x32x3 parameterisation, rendered through the EXACT receiver path (both rounds,
    the clamp, the selector) with a straight-through gradient on the rounds only.  The
    forward value is therefore bit-exact with the receiver; only the gradient is STE.
    """
    import torch
    from torch.nn import functional

    index = np.array([pair], dtype=np.int64)
    frame1 = up2.frames_to_bchw(inst.raw[2 * index + 1])
    target = torch.from_numpy(inst.targets[pair][None]).double()
    field = field_of(inst, inst.state.codes[pair])[None].float()

    leaf = field.clone().detach().requires_grad_(True)
    carrier = functional.interpolate(
        leaf, size=(br1.EVAL_H, br1.EVAL_W), mode="bicubic", align_corners=False
    )
    low = up2._round_ste((127.5 + br1.AMP * carrier).clamp(0.0, 255.0))
    slave = functional.interpolate(
        low, size=(up2.CAMERA_H, up2.CAMERA_W), mode="bicubic", align_corners=False
    )
    frame0 = up2._round_ste(slave.clamp(0.0, 255.0))
    frame0 = up2.apply_selector_float(
        frame0, inst.state.selector_modes, inst.state.selector_choices[index]
    )
    pose = up2.pose_from_frames(inst.posenet, frame0, frame1)
    rows = []
    for component in range(up2.POSE_DIMS):
        grad = torch.autograd.grad(
            pose[:, component].sum(), leaf, retain_graph=component < up2.POSE_DIMS - 1
        )[0]
        rows.append(grad.reshape(-1))
    jac = torch.stack(rows).double()
    res = (pose.detach().double() - target).reshape(-1)
    return jac, res, field[0].double().reshape(-1)


# ---------------------------------------------------------------------------
# stage: base
# ---------------------------------------------------------------------------


def n600_pose(inst, codes: np.ndarray, batch_size: int = 8) -> np.ndarray:
    coefficients = up2.codes_to_coefficients(
        np.asarray(codes, dtype=np.int32), inst.state.coefficient_scales
    )
    indices = np.arange(N_PAIRS, dtype=np.int64)
    per_pair, _ = up2.measure_pose(
        inst.posenet,
        inst.state,
        coefficients,
        inst.raw,
        inst.targets,
        indices,
        batch_size=batch_size,
    )
    return np.asarray(per_pair, dtype=np.float64)


def cmd_base(args) -> int:
    """Reproduce move 49's pose base, parse its carrier, and repack it byte-identically."""
    set_threads(args.threads)
    receipts = assert_pointer()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    inst = load_instrument()
    codes = np.asarray(inst.state.codes, dtype=np.int32)
    per_pair = n600_pose(inst, codes, batch_size=args.batch_size)
    elapsed = time.time() - started

    reference = np.load(PP1_BASE_VECTOR)
    if reference.shape != per_pair.shape:
        raise Cb1Error("pp1's base vector has a different shape")
    abs_gap = np.abs(per_pair - reference)
    gate_ok = bool(abs_gap.max() <= POSE_BASE_GATE_ABS)
    if not gate_ok:
        raise Cb1Error(
            f"base reproduction max |gap| {abs_gap.max():.3e} exceeds pp1's measured "
            f"band gate {POSE_BASE_GATE_ABS:.3e} (10x the observed batch-order maximum)"
        )

    basis_codes = shipped_basis_codes(inst)
    np.save(out / "pose_base_move49.npy", per_pair)
    np.save(out / "basis_codes_shipped.npy", basis_codes.astype(np.int8))
    np.save(out / "coefficient_codes_shipped.npy", codes)

    order = np.argsort(per_pair)[::-1]
    top12 = [
        {
            "rank": int(i),
            "pair": int(order[i]),
            "d_pose": float(per_pair[order[i]]),
            "share": float(per_pair[order[i]] / per_pair.sum()),
        }
        for i in range(12)
    ]
    report = {
        "schema": "ddm_cb1_base.v1",
        "receipts": receipts,
        "pairs": int(N_PAIRS),
        "d_pose_mean": float(per_pair.mean()),
        "d_pose_median": float(np.median(per_pair)),
        "d_pose_max": float(per_pair.max()),
        "pose_leg": jg5.pose_leg(float(per_pair.mean())),
        "pp1_reference_mean": PP1_BASE_MEAN,
        "reproduction_gate": {
            "max_abs_gap": float(abs_gap.max()),
            "mean_abs_gap": float(abs_gap.mean()),
            "gate_abs": POSE_BASE_GATE_ABS,
            "band_source": "ddm_pp1 section 10 (batch-1 vs batch-8, 16 pairs)",
            "passed": gate_ok,
        },
        "instrument_ratio_vs_t4_print": float(per_pair.mean()) / POINTER_D_POSE_T4,
        "top12": top12,
        "top12_cumulative_share": float(
            per_pair[order[:12]].sum() / per_pair.sum()
        ),
        "basis": {
            "codes_min": int(basis_codes.min()),
            "codes_max": int(basis_codes.max()),
            "distinct_symbols": int(np.unique(basis_codes).size),
            "symbols": int(basis_codes.size),
        },
        "coefficient_codes": {
            "min": int(codes.min()),
            "max": int(codes.max()),
            "absmax": int(np.abs(codes).max()),
        },
        "elapsed_seconds": elapsed,
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT, n600]",
        "score_claim": False,
        "promotable": False,
    }
    write_json(out / "BASE.json", report)
    print(json.dumps({k: report[k] for k in ("d_pose_mean", "pose_leg", "elapsed_seconds")}))
    return 0


def cmd_twins(args) -> int:
    """CONTROL: repack move 49's shipped carrier and get move 49's bytes back."""
    set_threads(args.threads)
    receipts = assert_pointer()
    import ddm_up3_carrier_splice as splice

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    body = splice.parse_shipped_body(POINTER_TREE, verify_sha=False)
    rebuilt = splice.build_archive(
        body,
        np.asarray(body.codes, dtype=np.int32),
        runtime_dir=POINTER_TREE,
        container_search=True,
        verify=True,
    )
    identical = rebuilt["archive_sha256"] == POINTER_ARCHIVE_SHA256
    basis_price = price_basis_section(np.asarray(body.lengths), body.basis_blob, body.basis_bits)
    report = {
        "schema": "ddm_cb1_twins.v1",
        "receipts": receipts,
        "rebuilt_sha256": rebuilt["archive_sha256"],
        "rebuilt_bytes": int(rebuilt["archive_size"]),
        "byte_identical": bool(identical),
        "carrier": {
            "basis_bits_huffman": int(body.basis_bits),
            "basis_bytes_huffman": len(body.basis_blob),
            "rice_payload_bytes": len(body.rice_payload),
            "carrier_stream_bytes": len(body.carrier_stream),
            "carrier_body_bytes": len(body.carrier_body),
            "scales_bytes": len(body.scales),
            "packed_metadata_bytes": len(body.packed_metadata),
            "body_tail_bytes": len(body.body_tail),
            "ks": np.asarray(body.ks).tolist(),
            "lengths": np.asarray(body.lengths).tolist(),
        },
        "basis_section_price": basis_price,
        "score_claim": False,
    }
    write_json(out / "TWINS.json", report)
    if not identical:
        raise Cb1Error(
            f"CONTROL FAILED: repacking move 49's own carrier gives "
            f"{rebuilt['archive_sha256']} ({rebuilt['archive_size']} B), not "
            f"{POINTER_ARCHIVE_SHA256} ({POINTER_ARCHIVE_BYTES} B)"
        )
    print(json.dumps({"byte_identical": identical, "basis_price": basis_price}))
    return 0


# ---------------------------------------------------------------------------
# the basis section's exact byte price, through the shipped chain
# ---------------------------------------------------------------------------


def _rr5():
    runtime_parent = str(POINTER_TREE)
    if runtime_parent not in sys.path:
        sys.path.insert(0, runtime_parent)
    from runtime import rr5_arith_basis as rr5  # type: ignore[import-not-found]

    return rr5


def zigzag_symbols(basis_codes: np.ndarray) -> np.ndarray:
    """The receiver's alphabet: symbol = zigzag(code), 0..31 (carrier_codec:187-190)."""
    codes = np.asarray(basis_codes, dtype=np.int64).reshape(-1)
    symbols = ((codes << 1) ^ (codes >> 63)).astype(np.int64)
    if symbols.min() < 0 or symbols.max() > 31:
        raise Cb1Error("zigzag symbols escape the 32-symbol alphabet")
    return symbols


def price_basis_section(
    lengths: np.ndarray | None, basis_blob: bytes | None, basis_bits: int | None
) -> dict[str, Any]:
    """Report the shipped basis section's two coder legs for a given payload."""
    rr5 = _rr5()
    if lengths is None or basis_blob is None or basis_bits is None:
        raise Cb1Error("price_basis_section needs the shipped fields")
    symbols = rr5.huffman_decode(
        np.asarray(lengths, dtype=np.int64), bytes(basis_blob), int(basis_bits),
        rr5.BASIS_SYMBOLS,
    )
    arith_payload, arith_bits = rr5.encode_basis_arith(symbols)
    return {
        "huffman_bytes": len(bytes(basis_blob)),
        "huffman_bits": int(basis_bits),
        "rider_arith_bytes": len(arith_payload),
        "rider_arith_bits": int(arith_bits),
        "bits_per_symbol_huffman": float(basis_bits) / rr5.BASIS_SYMBOLS,
        "bits_per_symbol_arith": float(arith_bits) / rr5.BASIS_SYMBOLS,
    }


def encode_basis_fields(basis_codes: np.ndarray) -> dict[str, Any]:
    """Huffman lengths + payload for a candidate basis, as ``build_archive`` needs them.

    The rider is applied downstream by ``_apply_entropy_riders`` inside
    ``build_archive``, exactly as the shipped body does it, so this returns the
    HUFFMAN leg (which the container stores) and reports the rider leg for the curve.
    """
    rr5 = _rr5()
    symbols = zigzag_symbols(basis_codes)
    if symbols.size != rr5.BASIS_SYMBOLS:
        raise Cb1Error(f"basis has {symbols.size} symbols, expected {rr5.BASIS_SYMBOLS}")
    histogram = np.bincount(symbols, minlength=rr5.BASIS_ALPHABET)
    lengths = rr5.huffman_lengths_from_histogram(histogram)
    payload, bits = rr5.huffman_encode(symbols, lengths)
    #: fail-closed: the receiver must decode exactly what we mean to ship.
    back = rr5.huffman_decode(lengths, payload, bits, rr5.BASIS_SYMBOLS)
    if not np.array_equal(back, symbols):
        raise Cb1Error("candidate basis Huffman round-trip FAILED")
    arith_payload, arith_bits = rr5.encode_basis_arith(symbols)
    if not np.array_equal(rr5.decode_basis_arith(arith_payload, arith_bits), symbols):
        raise Cb1Error("candidate basis RR5 arithmetic round-trip FAILED")
    return {
        "lengths": np.asarray(lengths, dtype=np.uint8),
        "payload": payload,
        "bits": bits,
        "huffman_bytes": len(payload),
        "rider_arith_bytes": len(arith_payload),
        "rider_arith_bits": int(arith_bits),
        "distinct_symbols": int((histogram > 0).sum()),
    }


# ---------------------------------------------------------------------------
# stage: subspace
# ---------------------------------------------------------------------------


def cmd_subspace(args) -> int:
    """Per-pair field Jacobian, residual and minimum-image-norm free-field step."""
    set_threads(args.threads)
    receipts = assert_pointer()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import torch

    inst = load_instrument()
    pairs = (
        np.arange(N_PAIRS, dtype=np.int64)
        if args.pairs >= N_PAIRS
        else br1.sample_pairs(args.pairs, args.seed)
    )
    shard = pairs[args.shard_index :: args.shard_count]
    dim = BASIS_PLANES * CARRIER_H * CARRIER_W

    steps = np.zeros((len(shard), dim), dtype=np.float64)
    fields = np.zeros((len(shard), dim), dtype=np.float64)
    jac_small = np.zeros((len(shard), up2.POSE_DIMS, CARRIER_DIM), dtype=np.float64)
    #: the FULL field Jacobian is retained, not only its shipped-span projection: any
    #: candidate span's demanded step needs J itself, and re-measuring it per blend
    #: would be the same 72 s over again ("always keep the payload").
    jac_full = np.zeros((len(shard), up2.POSE_DIMS, dim), dtype=np.float32)
    residual = np.zeros((len(shard), up2.POSE_DIMS), dtype=np.float64)
    dpose = np.zeros(len(shard), dtype=np.float64)
    sigma = np.zeros((len(shard), up2.POSE_DIMS), dtype=np.float64)
    started = time.time()
    for position, pair in enumerate(int(p) for p in shard):
        jac, res, field = field_jacobian(inst, pair)
        step = torch.linalg.pinv(jac) @ (-res)
        steps[position] = step.numpy()
        fields[position] = field.numpy()
        #: the coefficient-space Jacobian in the SHIPPED span, for the leverage table.
        jac_small[position] = (jac @ inst.bmat.reshape(CARRIER_DIM, -1).T).numpy()
        jac_full[position] = jac.numpy().astype(np.float32)
        residual[position] = res.numpy()
        dpose[position] = float((res**2).mean())
        sigma[position] = torch.linalg.svdvals(jac).numpy()
        if (position + 1) % 25 == 0:
            print(
                f"shard {args.shard_index}: {position + 1}/{len(shard)} "
                f"in {time.time() - started:.0f}s",
                flush=True,
            )
    tag = f"{args.shard_index:02d}_of_{args.shard_count:02d}"
    np.savez_compressed(
        out / f"subspace_{tag}.npz",
        pairs=shard,
        steps=steps.astype(np.float32),
        fields=fields.astype(np.float32),
        jac_small=jac_small,
        jac_full=jac_full,
        residual=residual,
        dpose=dpose,
        sigma=sigma,
    )
    write_json(
        out / f"SUBSPACE_{tag}.json",
        {
            "schema": "ddm_cb1_subspace_shard.v1",
            "receipts": receipts,
            "pairs": len(shard),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "elapsed_seconds": time.time() - started,
            "sample": "full n600" if args.pairs >= N_PAIRS else f"seeded random n={args.pairs}",
            "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT]",
            "score_claim": False,
        },
    )
    print(f"shard {tag} done in {time.time() - started:.0f}s")
    return 0


def load_subspace(out: Path) -> dict[str, np.ndarray]:
    #: the shard glob must not swallow this arm's own derived outputs written into the
    #: same directory (``subspace_directions.npz`` did exactly that on the first run).
    shards = sorted(out.glob("subspace_[0-9][0-9]_of_[0-9][0-9].npz"))
    if not shards:
        raise Cb1Error(f"no subspace shards under {out}")
    parts = [np.load(path) for path in shards]
    pairs = np.concatenate([p["pairs"] for p in parts])
    order = np.argsort(pairs)
    return {
        "pairs": pairs[order],
        "steps": np.concatenate([p["steps"] for p in parts])[order].astype(np.float64),
        "fields": np.concatenate([p["fields"] for p in parts])[order].astype(np.float64),
        "jac_small": np.concatenate([p["jac_small"] for p in parts])[order],
        "jac_full": np.concatenate([p["jac_full"] for p in parts])[order].astype(
            np.float64
        ),
        "residual": np.concatenate([p["residual"] for p in parts])[order],
        "dpose": np.concatenate([p["dpose"] for p in parts])[order],
        "sigma": np.concatenate([p["sigma"] for p in parts])[order],
    }


def principal_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Principal angles (degrees) between the column spans of two orthonormal bases."""
    qa = np.linalg.qr(a)[0]
    qb = np.linalg.qr(b)[0]
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(singular, -1.0, 1.0)))


def cmd_analyse(args) -> int:
    """SVD of the residual and target subspaces, energy tables, subspace angle."""
    set_threads(args.threads)
    receipts = assert_pointer()
    out = Path(args.out_dir)
    data = load_subspace(Path(args.subspace_dir))
    inst = load_instrument()
    bmat = inst.bmat.numpy()  # (12, 2304) -- the shipped span generators
    shipped_q = np.linalg.qr(bmat.T)[0]  # (2304, 12) orthonormal

    steps, fields, dpose = data["steps"], data["fields"], data["dpose"]
    #: pose-mass weighting: a pair's share of the mean d_pose is exactly its share of
    #: the score's pose leg at first order, so weight the rows by sqrt(d_pose) and the
    #: SVD energy is pose mass (DERIVED from S = sqrt(10 * mean d_pose)).
    weight = np.sqrt(np.maximum(dpose, 0.0))
    weighted_steps = steps * weight[:, None]
    weighted_targets = (fields + steps) * weight[:, None]

    def spectrum(matrix: np.ndarray, keep: int = 24) -> dict[str, Any]:
        _u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        total = float((s**2).sum())
        cumulative = np.cumsum(s**2) / total if total > 0 else np.zeros_like(s)
        return {
            "singular_values": s[:keep].tolist(),
            "cumulative_energy": cumulative[:keep].tolist(),
            "energy_at_12": float(cumulative[min(11, len(cumulative) - 1)]),
            "energy_at_8": float(cumulative[min(7, len(cumulative) - 1)]),
            "vt": vt,
            "total": total,
        }

    residual_spec = spectrum(weighted_steps)
    target_spec = spectrum(weighted_targets)
    field_spec = spectrum(fields * weight[:, None])

    residual_v = residual_spec["vt"][:12].T  # (2304, 12)
    target_v = target_spec["vt"][:12].T

    #: How much of each set does the SHIPPED span already hold?
    def captured(matrix: np.ndarray, basis_q: np.ndarray) -> float:
        #: basis_q is orthonormal, so the projected energy is the energy of the
        #: coordinates -- never form the 600x2304 reconstruction to get a scalar.
        coordinates = matrix @ basis_q
        num = float((coordinates**2).sum())
        den = float((matrix**2).sum())
        return num / den if den > 0 else 0.0

    angles_residual = principal_angles(shipped_q, residual_v)
    angles_target = principal_angles(shipped_q, target_v)

    order = np.argsort(dpose)[::-1]
    top12 = data["pairs"][order[:12]]
    top12_share = float(dpose[order[:12]].sum() / dpose.sum())

    report = {
        "schema": "ddm_cb1_subspace_analysis.v1",
        "receipts": receipts,
        "pairs_measured": len(data["pairs"]),
        "d_pose_mean_over_measured": float(dpose.mean()),
        "residual_subspace": {
            "energy_captured_by_top12_of_itself": residual_spec["energy_at_12"],
            "energy_captured_by_top8_of_itself": residual_spec["energy_at_8"],
            "singular_values": residual_spec["singular_values"][:16],
            "cumulative_energy": residual_spec["cumulative_energy"][:16],
            "energy_captured_by_SHIPPED_span": captured(weighted_steps, shipped_q),
        },
        "target_field_subspace": {
            "energy_captured_by_top12_of_itself": target_spec["energy_at_12"],
            "singular_values": target_spec["singular_values"][:16],
            "cumulative_energy": target_spec["cumulative_energy"][:16],
            "energy_captured_by_SHIPPED_span": captured(weighted_targets, shipped_q),
        },
        "realized_field_subspace": {
            "singular_values": field_spec["singular_values"][:16],
            "cumulative_energy": field_spec["cumulative_energy"][:16],
            "energy_captured_by_SHIPPED_span": captured(fields * weight[:, None], shipped_q),
        },
        "principal_angles_deg_shipped_vs_residual12": angles_residual.tolist(),
        "principal_angles_deg_shipped_vs_target12": angles_target.tolist(),
        "top12_pairs": top12.tolist(),
        "top12_pose_share": top12_share,
        "sigma_summary": {
            "sigma_max_median": float(np.median(data["sigma"][:, 0])),
            "sigma_min_median": float(np.median(data["sigma"][:, -1])),
            "condition_median": float(
                np.median(data["sigma"][:, 0] / np.maximum(data["sigma"][:, -1], 1e-30))
            ),
        },
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT] -- LINEARIZED",
        "score_claim": False,
    }
    np.savez_compressed(
        out / "subspace_directions.npz",
        residual_v=residual_v.astype(np.float32),
        target_v=target_v.astype(np.float32),
        shipped_q=shipped_q.astype(np.float32),
        weight=weight,
        pairs=data["pairs"],
    )
    write_json(out / "SUBSPACE_ANALYSIS.json", report)
    print(json.dumps({k: report[k] for k in ("pairs_measured", "top12_pose_share")}))
    return 0


# ---------------------------------------------------------------------------
# stage: refit
# ---------------------------------------------------------------------------


def quantize_atom(target: np.ndarray) -> tuple[np.ndarray, float]:
    """Best five-bit integer code grid for one target direction, and its cosine.

    The per-atom quantiser step is FREE: the receiver divides every atom by its own
    RMS (``normalized_basis``), so only the code grid's DIRECTION survives.  The step
    is therefore searched, not chosen -- pc1 section 5 MEASURED that the naive
    ``codes/2`` step and the searched step differ by 0.899-0.983 vs 0.991-1.000 in
    per-atom cosine on this very carrier.
    """
    direction = np.asarray(target, dtype=np.float64).reshape(
        BASIS_PLANES, CARRIER_H, CARRIER_W
    )
    #: centre first: the receiver subtracts each atom's upsampled mean, so the DC of
    #: the code grid is unobservable and spending dynamic range on it is pure loss.
    direction = direction - direction.mean()
    peak = np.abs(direction).max()
    if peak <= 0:
        raise Cb1Error("cannot quantise a zero atom")
    best_codes, best_cos = None, -2.0
    for scale in np.linspace(4.0, float(BASIS_CODE_MAX), 240):
        codes = np.clip(
            np.rint(direction * (scale / peak)), BASIS_CODE_MIN, BASIS_CODE_MAX
        )
        if not codes.any():
            continue
        centred = codes - codes.mean()
        denominator = np.linalg.norm(centred) * np.linalg.norm(direction)
        cosine = float((centred * direction).sum() / denominator) if denominator else -2.0
        if cosine > best_cos:
            best_cos, best_codes = cosine, codes
    if best_codes is None:
        raise Cb1Error("no quantiser step produced a usable atom")
    return best_codes.astype(np.int64), best_cos


def build_blend(inst, directions: np.ndarray, keep: int, shipped_rank: np.ndarray):
    """A 12-atom basis: ``keep`` shipped atoms plus ``12 - keep`` residual directions.

    The shipped atoms are kept AS CODES (bit-identical to the shipped ones) so the
    kept half of the span is exactly the incumbent's, and only the replaced atoms are
    quantised.  The new directions are orthogonalised against the kept span first,
    because a direction already inside it adds no reach -- br1 MEASURED that a
    re-orientation inside the span is null to 1.9e-08.
    """
    shipped = shipped_basis_codes(inst)
    kept_index = [int(i) for i in shipped_rank[:keep]]
    dropped_index = [int(i) for i in shipped_rank[keep:]]
    codes = shipped.copy()
    if not dropped_index:
        return codes, {"kept": kept_index, "dropped": [], "cosines": []}

    blow = inst.blow.numpy().reshape(CARRIER_DIM, -1)
    kept_q = np.linalg.qr(blow[kept_index].T)[0] if kept_index else np.zeros((blow.shape[1], 0))
    cosines = []
    accepted: list[np.ndarray] = []
    source = 0
    for slot in dropped_index:
        while True:
            if source >= directions.shape[1]:
                raise Cb1Error("ran out of residual directions to fill the blend")
            vector = np.asarray(directions[:, source], dtype=np.float64)
            source += 1
            if kept_q.shape[1]:
                vector = vector - kept_q @ (kept_q.T @ vector)
            for previous in accepted:
                vector = vector - previous * float(previous @ vector)
            norm = np.linalg.norm(vector)
            if norm > 1e-8:
                vector = vector / norm
                break
        atom, cosine = quantize_atom(vector)
        codes[slot] = atom
        cosines.append({"slot": int(slot), "grid_cosine": cosine})
        accepted.append(vector)
    #: REALIZED fidelity, through the receiver's own normalisation rather than on the
    #: 24x32 grid: ``normalized_basis`` centres and RMS-divides the UPSAMPLED atom, so
    #: the grid cosine above is a proxy and this one is the object that ships.
    realized = with_basis(inst, codes)
    realized_blow = realized.blow.numpy().reshape(CARRIER_DIM, -1)
    for entry, intended in zip(cosines, accepted, strict=True):
        row = realized_blow[entry["slot"]]
        entry["realized_cosine"] = float(
            (row @ intended) / (np.linalg.norm(row) * np.linalg.norm(intended))
        )
    return codes, {
        "gram_condition": float(np.linalg.cond(realized.gram.numpy())),
        "kept": kept_index,
        "dropped": dropped_index,
        "cosines": cosines,
    }


def build_partial(inst, direction: np.ndarray, slot: int, alpha: float):
    """Move ONE atom a FRACTION of the way toward a residual direction.

    ``keep=11`` replaces an atom outright; that is one point, and a verdict drawn from
    one point is a verdict about an endpoint, not a family.  This traces the segment
    between the incumbent (``alpha = 0``, provably null) and the replacement
    (``alpha = 1``), so the arm can say whether the response is monotone in the
    perturbation or has an interior optimum the endpoints hid.

    The mix is taken in the receiver-normalised atom's own coordinates and the result
    is re-quantised to the shipped 5-bit lattice, so every point on the curve is a
    legal shipped basis and not an interpolation of two look-alikes.
    """
    if not 0.0 <= alpha <= 1.0:
        raise Cb1Error(f"alpha must lie in [0, 1], got {alpha}")
    blow = inst.blow.numpy().reshape(CARRIER_DIM, -1)
    incumbent = blow[slot] / np.linalg.norm(blow[slot])
    other = [i for i in range(CARRIER_DIM) if i != slot]
    kept_q = np.linalg.qr(blow[other].T)[0]
    target = np.asarray(direction, dtype=np.float64)
    target = target - kept_q @ (kept_q.T @ target)
    norm = np.linalg.norm(target)
    if norm <= 1e-8:
        raise Cb1Error("the residual direction collapses inside the kept span")
    target = target / norm
    #: align the sign so alpha interpolates rather than cancels.
    if incumbent @ target < 0:
        target = -target
    mixed = (1.0 - alpha) * incumbent + alpha * target
    codes = shipped_basis_codes(inst)
    atom, grid_cosine = quantize_atom(mixed)
    codes[slot] = atom
    realized = with_basis(inst, codes)
    row = realized.blow.numpy().reshape(CARRIER_DIM, -1)[slot]
    return codes, {
        "slot": int(slot),
        "alpha": float(alpha),
        "grid_cosine_to_mix": grid_cosine,
        "realized_cosine_to_incumbent": float(
            (row @ incumbent) / np.linalg.norm(row)
        ),
        "gram_condition": float(np.linalg.cond(realized.gram.numpy())),
    }


def shipped_pose_leverage(inst, data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Rank the twelve shipped ATOMS by the POSE damage dropping each would do.

    The ranking is over atom SLOTS, not over Gram-Schmidt directions.  (My own round-1
    review caught the difference: ``np.linalg.qr(blow.T)`` returns directions whose
    index is an orthogonalisation order, and indexing atom slots with it silently ranks
    the wrong objects.  It is the ATOM that gets replaced, so it is the ATOM that must
    be scored.)

    Write ``W = bmat`` (12 x 2304), ``G = W W^T``, ``X_p = c_p^T W``.  Dropping atom j
    leaves ``span(W_{i != j})``; the part of ``W_j`` outside it is parallel to the dual
    vector ``W*_j = (G^-1 W)_j``, because the dual is orthogonal to every other
    generator by construction.  With ``n_j = W*_j / sqrt((G^-1)_jj)`` a unit vector,
    the field each pair LOSES is ``c_pj / sqrt((G^-1)_jj) * n_j``, and its first-order
    pose cost is

        damage_j = sum_p  w_p^2 * c_pj^2 * ||J_p W*_j||^2 / (G^-1)_jj^2

    with ``J_p W*_j`` read straight off the measured ``jac_small = J_p W^T`` as
    ``(jac_small G^-1)_{:,j}``.  ``w_p^2 = d_pose_p`` is the pair's pose mass, so the
    ranking is in score units, not field energy.
    """
    bmat = inst.bmat.numpy()  # (12, 2304)
    gram_inv = np.linalg.inv(bmat @ bmat.T)
    gram_inv_diagonal = np.diag(gram_inv)
    fields = data["fields"]
    weight_squared = np.maximum(data["dpose"], 0.0)
    scales = inst.state.coefficient_scales.double().numpy()
    codes = np.asarray(inst.state.codes, dtype=np.float64)
    lookup = {int(p): i for i, p in enumerate(data["pairs"])}

    damage = np.zeros(CARRIER_DIM, dtype=np.float64)
    used_energy = np.zeros(CARRIER_DIM, dtype=np.float64)
    for position, pair in enumerate(int(p) for p in data["pairs"]):
        coefficients = codes[pair] * scales  # (12,)
        jac_small = data["jac_small"][position]  # (6, 12) = J_p W^T
        jdual = jac_small @ gram_inv  # (6, 12) = J_p W*^T
        lost = (coefficients**2) / (gram_inv_diagonal**2)
        damage += weight_squared[position] * (jdual**2).sum(axis=0) * lost
        used_energy += weight_squared[position] * coefficients**2
    if len(lookup) != len(fields):
        raise Cb1Error("subspace shards contain duplicate pairs")
    order = np.argsort(damage)  # ascending: least damaging first
    return {
        "atom_pose_damage": damage.tolist(),
        "atom_pose_mass_weighted_coefficient_energy": used_energy.tolist(),
        "least_damaging_first": order.tolist(),
        "gram_inverse_diagonal": gram_inv_diagonal.tolist(),
    }


def demanded_step(inst_candidate, data: dict[str, np.ndarray]) -> dict[str, Any]:
    """up2's basis-penalty instrument, re-derived on an ARBITRARY candidate span.

    For each pair the minimum-IMAGE-norm coefficient step that cancels the residual to
    first order inside the candidate span is

        dc = G^-1 A^T (A G^-1 A^T)^-1 (-r),   A = J W^T,   G = W W^T

    (``br1.min_image_norm_step``, used verbatim on the candidate's own Gram).  Two
    numbers come out of it, both pose-mass weighted: the IMAGE norm ``||dX|| =
    ||dc||_G`` -- the basis-free quantity up2 reported in LSB rms -- and the CODE step
    ``dc / coefficient_scales``, which is what the int12 lattice has to hold.

    This is a LINEARIZED screen.  It ranks candidate spans without a solve; it does
    NOT predict realized d_pose, because the round, the clamp and PoseNet's own
    nonlinearity are exactly what makes a large step fail.  Every verdict in this arm
    still comes from a realized re-solve.
    """
    import torch

    bmat = inst_candidate.bmat.double()
    gram = inst_candidate.gram.double()
    scales = inst_candidate.state.coefficient_scales.double().numpy()
    image_norm = np.zeros(len(data["pairs"]), dtype=np.float64)
    code_step = np.zeros(len(data["pairs"]), dtype=np.float64)
    sigma_min = np.zeros(len(data["pairs"]), dtype=np.float64)
    for position in range(len(data["pairs"])):
        jac = torch.from_numpy(data["jac_full"][position])  # (6, 2304)
        res = torch.from_numpy(data["residual"][position])
        small = jac @ bmat.T  # (6, 12)
        step = br1.min_image_norm_step(small, res, gram)
        image_norm[position] = float(torch.sqrt(step @ gram @ step))
        code_step[position] = float(np.abs(step.numpy() / scales).max())
        whitened = small @ torch.linalg.inv(torch.linalg.cholesky(gram)).T
        sigma_min[position] = float(torch.linalg.svdvals(whitened)[-1])
    weight = np.maximum(data["dpose"], 0.0)
    total = weight.sum()
    #: ``AMP * ||dX||`` is the perturbation in 0..255 LSB, the unit up2 reported.
    lsb = br1.AMP * image_norm / math.sqrt(BASIS_PLANES * CARRIER_H * CARRIER_W)
    return {
        "image_norm_lsb_rms_median": float(np.median(lsb)),
        "image_norm_lsb_rms_pose_weighted": float((lsb * weight).sum() / total),
        "code_step_max_median": float(np.median(code_step)),
        "code_step_max_pose_weighted": float((code_step * weight).sum() / total),
        "code_step_over_rail_fraction": float((code_step > 2047).mean()),
        "sigma_min_whitened_median": float(np.median(sigma_min)),
        "sigma_min_whitened_pose_weighted": float((sigma_min * weight).sum() / total),
        "per_pair_lsb": lsb,
        "per_pair_code_step": code_step,
    }


def cmd_partial(args) -> int:
    """Build the alpha ladder for one atom and price each rung's basis section."""
    set_threads(args.threads)
    receipts = assert_pointer()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    directions = np.load(Path(args.subspace_dir) / "subspace_directions.npz")
    inst = load_instrument()
    rows = []
    for alpha in [float(a) for a in args.alphas.split(",")]:
        codes, detail = build_partial(
            inst, directions["residual_v"][:, 0], args.slot, alpha
        )
        price = encode_basis_fields(codes)
        name = f"basis_codes_slot{args.slot:02d}_alpha{round(alpha * 100):03d}.npy"
        np.save(out / name, codes.astype(np.int8))
        detail["huffman_bytes"] = price["huffman_bytes"]
        detail["rider_arith_bytes"] = price["rider_arith_bytes"]
        detail["codes_path"] = str(out / name)
        rows.append(detail)
    write_json(
        out / "PARTIAL.json",
        {
            "schema": "ddm_cb1_partial.v1",
            "receipts": receipts,
            "slot": args.slot,
            "direction": "residual_v column 0",
            "rungs": rows,
            "score_claim": False,
        },
    )
    print(json.dumps({"rungs": [(r["alpha"], r["rider_arith_bytes"]) for r in rows]}))
    return 0


def cmd_predict(args) -> int:
    """Linearized screen of every built blend against the shipped span."""
    set_threads(args.threads)
    receipts = assert_pointer()
    data = load_subspace(Path(args.subspace_dir))
    inst = load_instrument()
    out = Path(args.out_dir)
    rows = []
    free = np.linalg.norm(data["steps"], axis=1) / math.sqrt(
        BASIS_PLANES * CARRIER_H * CARRIER_W
    ) * br1.AMP
    weight = np.maximum(data["dpose"], 0.0)
    for path in sorted(out.glob("basis_codes_keep*.npy")):
        codes = np.load(path).astype(np.int64)
        candidate = with_basis(inst, codes)
        row = demanded_step(candidate, data)
        row.pop("per_pair_lsb")
        row.pop("per_pair_code_step")
        row["basis_codes"] = path.name
        rows.append(row)
    report = {
        "schema": "ddm_cb1_predict.v1",
        "receipts": receipts,
        "free_field_step_lsb_rms_median": float(np.median(free)),
        "free_field_step_lsb_rms_pose_weighted": float(
            (free * weight).sum() / weight.sum()
        ),
        "blends": rows,
        "axis": "LINEARIZED screen -- ranks spans, predicts no realized d_pose",
        "score_claim": False,
    }
    write_json(out / "PREDICT.json", report)
    print(json.dumps({"blends": len(rows)}))
    return 0


def cmd_refit(args) -> int:
    """Build the blends, quantise them to the shipped lattice, price the basis section."""
    set_threads(args.threads)
    receipts = assert_pointer()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    directions = np.load(Path(args.subspace_dir) / "subspace_directions.npz")
    data = load_subspace(Path(args.subspace_dir))
    inst = load_instrument()

    leverage = shipped_pose_leverage(inst, data)
    #: keep the atoms whose loss would cost the most pose; drop the cheapest.
    rank_desc = np.asarray(leverage["least_damaging_first"], dtype=np.int64)[::-1]
    shipped_rank = rank_desc  # index 0 = most valuable, index 11 = most expendable

    which = args.directions
    source = directions["residual_v"] if which == "residual" else directions["target_v"]
    shipped_price = None
    rows = []
    for keep in [int(k) for k in args.keep.split(",")]:
        codes, detail = build_blend(inst, source, keep, shipped_rank)
        price = encode_basis_fields(codes)
        if keep == CARRIER_DIM and not np.array_equal(codes, shipped_basis_codes(inst)):
            raise Cb1Error("the keep=12 control is not the shipped basis")
        if keep == CARRIER_DIM:
            shipped_price = price
        np.save(out / f"basis_codes_keep{keep:02d}.npy", codes.astype(np.int8))
        rows.append(
            {
                "keep_shipped_atoms": keep,
                "replaced": CARRIER_DIM - keep,
                "kept_slots": detail["kept"],
                "dropped_slots": detail["dropped"],
                "quantised_cosines": detail["cosines"],
                "gram_condition": detail.get("gram_condition"),
                "huffman_bytes": price["huffman_bytes"],
                "rider_arith_bytes": price["rider_arith_bytes"],
                "distinct_symbols": price["distinct_symbols"],
                "codes_path": str(out / f"basis_codes_keep{keep:02d}.npy"),
            }
        )
    if shipped_price is not None:
        for row in rows:
            row["delta_rider_bytes_vs_shipped"] = (
                row["rider_arith_bytes"] - shipped_price["rider_arith_bytes"]
            )
    report = {
        "schema": "ddm_cb1_refit.v1",
        "receipts": receipts,
        "directions_source": which,
        "shipped_atom_pose_leverage": leverage,
        "blends": rows,
        "note": (
            "basis bytes here are the SECTION price; the archive delta is measured by "
            "building the real archive in stage price"
        ),
        "score_claim": False,
    }
    write_json(out / "REFIT.json", report)
    print(json.dumps({"blends": [(r["keep_shipped_atoms"], r["rider_arith_bytes"]) for r in rows]}))
    return 0


# ---------------------------------------------------------------------------
# stage: resolve
# ---------------------------------------------------------------------------


def warm_start_codes(inst_new, shipped_fields: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Least-squares projection of each pair's SHIPPED field onto the NEW span.

    The same warm start pc1 used for V2/V4, so this arm's result is comparable to the
    prior negatives rather than being a different experiment wearing the same name.
    """
    bmat = inst_new.bmat.numpy()  # (12, 2304)
    gram = bmat @ bmat.T
    codes = np.zeros((len(pairs), CARRIER_DIM), dtype=np.int32)
    scales = inst_new.state.coefficient_scales.double().numpy()
    for position in range(len(pairs)):
        coefficients = np.linalg.solve(gram, bmat @ shipped_fields[position])
        codes[position] = br1.realize(coefficients / scales)
    return codes


def warm_start_codes_pose(
    inst_new,
    shipped_fields: np.ndarray,
    residuals: np.ndarray,
    jacobians: np.ndarray,
) -> np.ndarray:
    """A POSE-targeted warm start, so a negative cannot be blamed on the start.

    The field projection above is the shortest path to a look-alike FIELD; it is not
    the shortest path to the right POSE, and the whole point of a refit span is that
    the pose direction is what it holds.  br1's F3 measured the sister failure -- a
    verdict that was really about the SEARCH's reach, not the object -- so this arm
    runs BOTH starts and reports both.

    Linearising at the shipped operating point ``X_p`` with residual ``r_p``, any field
    ``X`` in the new span has pose residual ``r_p + J_p (X - X_p)``.  Writing
    ``X = X_0 + d^T W`` with ``X_0`` the field projection (whose error is orthogonal to
    the span by construction), the minimum-IMAGE-norm ``d`` that cancels the residual is
    ``br1.min_image_norm_step(J_p W^T, r_p + J_p (X_0 - X_p), G)``: the same helper, on
    the candidate's own Gram.  It stays as close to the shipped field as the pose
    constraint allows, which keeps the linearisation honest.
    """
    import torch

    bmat = inst_new.bmat.double()
    bmat_np = bmat.numpy()
    gram = inst_new.gram.double()
    gram_np = gram.numpy()
    scales = inst_new.state.coefficient_scales.double().numpy()
    codes = np.zeros((len(shipped_fields), CARRIER_DIM), dtype=np.int32)
    for position in range(len(shipped_fields)):
        field = shipped_fields[position]
        base_coefficients = np.linalg.solve(gram_np, bmat_np @ field)
        projected = base_coefficients @ bmat_np
        jac = jacobians[position]
        small = torch.from_numpy(jac @ bmat_np.T)
        offset = jac @ (projected - field)
        step = br1.min_image_norm_step(
            small, torch.from_numpy(residuals[position] + offset), gram
        ).numpy()
        codes[position] = br1.realize((base_coefficients + step) / scales)
    return codes


def cmd_resolve(args) -> int:
    """Re-solve the coefficients on a candidate basis and bound the population."""
    set_threads(args.threads)
    receipts = assert_pointer()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    basis_codes = np.load(args.basis_codes).astype(np.int64)
    base_vector = np.load(args.base_pose)
    data = load_subspace(Path(args.subspace_dir))
    inst = load_instrument(basis_codes)

    pairs = (
        np.arange(N_PAIRS, dtype=np.int64)
        if args.pairs >= N_PAIRS
        else br1.sample_pairs(args.pairs, args.seed)
    )
    shard = pairs[args.shard_index :: args.shard_count]
    lookup = {int(p): i for i, p in enumerate(data["pairs"])}
    missing = [int(p) for p in shard if int(p) not in lookup]
    if missing:
        raise Cb1Error(f"subspace shards do not cover pairs {missing[:8]}")
    fields = np.stack([data["fields"][lookup[int(p)]] for p in shard])
    if args.warm_start == "field":
        start = warm_start_codes(inst, fields, shard)
    else:
        start = warm_start_codes_pose(
            inst,
            fields,
            np.stack([data["residual"][lookup[int(p)]] for p in shard]),
            np.stack([data["jac_full"][lookup[int(p)]] for p in shard]),
        )

    dd_threshold = jg5.materiality_dd_threshold(float(base_vector.mean()))
    rows = []
    started = time.time()
    rows_path = out / f"rows_{args.shard_index:02d}_of_{args.shard_count:02d}.jsonl"
    with open(rows_path, "a") as handle:
        for position, pair in enumerate(int(p) for p in shard):
            row = jg5.refine_pair(
                inst,
                pair,
                start[position],
                dd_threshold=dd_threshold,
                outer_rounds=args.outer_rounds,
                max_gn_iterations=args.max_gn_iterations,
            )
            row["pair"] = pair
            row["base_d_pose"] = float(base_vector[pair])
            row["warm_start_codes"] = start[position].tolist()
            rows.append(row)
            handle.write(json.dumps(row, default=_jsonable) + "\n")
            handle.flush()
            print(
                f"pair {pair}: base {base_vector[pair]:.6e} start {row['start_d_pose']:.6e} "
                f"final {row['final_d_pose']:.6e} ({row.get('stop_reason')}) "
                f"{time.time() - started:.0f}s",
                flush=True,
            )
    finals = np.array([r["final_d_pose"] for r in rows], dtype=np.float64)
    bases = np.array([r["base_d_pose"] for r in rows], dtype=np.float64)
    lower_bound = float(finals.sum() / N_PAIRS)
    write_json(
        out / f"RESOLVE_{args.shard_index:02d}_of_{args.shard_count:02d}.json",
        {
            "schema": "ddm_cb1_resolve_shard.v1",
            "receipts": receipts,
            "basis_codes": str(args.basis_codes),
            "warm_start": args.warm_start,
            "basis_codes_sha256": hashlib.sha256(
                np.ascontiguousarray(basis_codes.astype(np.int8)).tobytes()
            ).hexdigest(),
            "pairs_solved": len(rows),
            "sample": "full n600" if args.pairs >= N_PAIRS else f"seeded random n={args.pairs}",
            "subset_final_sum": float(finals.sum()),
            "subset_base_sum": float(bases.sum()),
            "subset_final_mean": float(finals.mean()),
            "subset_base_mean": float(bases.mean()),
            "population_lower_bound_d_pose": lower_bound,
            "base_mean_n600": float(base_vector.mean()),
            "lower_bound_exceeds_base": bool(lower_bound > float(base_vector.mean())),
            "pairs_worse": int((finals > bases).sum()),
            "pairs_better": int((finals < bases).sum()),
            "elapsed_seconds": time.time() - started,
            "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT]",
            "score_claim": False,
        },
    )
    print(
        json.dumps(
            {
                "pairs": len(rows),
                "subset_base_mean": float(bases.mean()),
                "subset_final_mean": float(finals.mean()),
                "population_lower_bound": lower_bound,
            }
        )
    )
    return 0


# ---------------------------------------------------------------------------
# stage: price
# ---------------------------------------------------------------------------


def break_even_d_pose(delta_bytes: int, base_d_pose: float) -> float:
    """The largest d_pose a blend may reach and still clear the admit bar.

    ``net = (leg_new - leg_base) + delta_bytes * RATE <= ADMIT_BAR`` with the seg leg
    carried (the frame-1 sections are byte-identical), so
    ``leg_new <= leg_base + ADMIT_BAR - delta_bytes * RATE`` and the bound follows from
    the square.  Every term is the scoring function's own arithmetic; nothing is fitted.

    ``ADMIT_BAR`` is NEGATIVE, so it TIGHTENS the bound.  My own first pass wrote
    ``- ADMIT_BAR`` and loosened it instead -- a 1.048x break-even where the truth is
    1.036x, i.e. an error in the direction that flatters the candidate.  The test
    ``test_break_even_is_the_scoring_functions_own_arithmetic`` re-composes the net at
    the returned bound and refuses anything that is not exactly the bar.
    """
    leg_base = math.sqrt(10.0 * base_d_pose)
    leg_max = leg_base + ADMIT_BAR - delta_bytes * RATE_PER_BYTE
    if leg_max <= 0.0:
        raise Cb1Error("the byte delta alone already exceeds the whole pose term")
    return leg_max**2 / 10.0


def cmd_verdict(args) -> int:
    """Assemble every blend's realized re-solve against its own break-even."""
    set_threads(args.threads)
    receipts = assert_pointer()
    base_vector = np.load(args.base_pose)
    base_mean = float(base_vector.mean())
    byteonly = json.loads(Path(args.byteonly).read_text())
    deltas = {
        row["blend"]: int(row["delta_archive_bytes"]) for row in byteonly["rows"]
    }
    rows = []
    for entry in args.blend:
        name, directory, delta_key = entry.split("=", 2)
        paths = sorted(Path(directory).glob("rows_*.jsonl"))
        solved = []
        for path in paths:
            for line in path.read_text().splitlines():
                if line.strip():
                    solved.append(json.loads(line))
        if not solved:
            raise Cb1Error(f"blend {name} has no solved rows under {directory}")
        finals = np.array([r["final_d_pose"] for r in solved], dtype=np.float64)
        bases = np.array([r["base_d_pose"] for r in solved], dtype=np.float64)
        starts = np.array([r["start_d_pose"] for r in solved], dtype=np.float64)
        delta = deltas[delta_key]
        allowed = break_even_d_pose(delta, base_mean)
        stops: dict[str, int] = {}
        for r in solved:
            stops[str(r.get("stop_reason"))] = stops.get(str(r.get("stop_reason")), 0) + 1
        rows.append(
            {
                "blend": name,
                "rows_dir": directory,
                "pairs_solved": len(solved),
                "delta_archive_bytes": delta,
                "dS_rate": delta * RATE_PER_BYTE,
                "subset_base_mean": float(bases.mean()),
                "subset_warm_start_mean": float(starts.mean()),
                "subset_final_mean": float(finals.mean()),
                "paired_ratio_final_over_base": float(finals.mean() / bases.mean()),
                "break_even_d_pose": allowed,
                "break_even_ratio": allowed / base_mean,
                "misses_break_even_by": float(finals.mean() / bases.mean())
                / (allowed / base_mean),
                "pairs_worse": int((finals > bases * (1.0 + 1e-7)).sum()),
                "pairs_better": int((finals < bases * (1.0 - 1e-7)).sum()),
                "population_lower_bound": float(finals.sum() / N_PAIRS),
                "lower_bound_fires": bool(finals.sum() / N_PAIRS > base_mean),
                "stop_reasons": stops,
            }
        )
    candidates = [r for r in rows if "control" not in r["blend"].lower()]
    if not candidates:
        raise Cb1Error("every blend is named a control; there is nothing to falsify")
    report = {
        "schema": "ddm_cb1_verdict.v1",
        "receipts": receipts,
        "base_d_pose_n600": base_mean,
        "base_pose_leg": math.sqrt(10.0 * base_mean),
        "zero_distortion_ceiling_S": math.sqrt(10.0 * base_mean),
        "zero_distortion_ceiling_bytes": math.sqrt(10.0 * base_mean) / RATE_PER_BYTE,
        "admit_bar": ADMIT_BAR,
        "sample": "seeded random, never a prefix (br1.sample_pairs)",
        "blends": rows,
        "falsifier": (
            "charter: the refit basis + full re-solve must cut n600 d_pose by >= 3% at "
            "delta bytes <= +200 B"
        ),
        #: named, not positional -- ``rows[1:]`` silently assumed the control was first.
        "falsifier_fired": bool(candidates)
        and all(r["paired_ratio_final_over_base"] > 0.97 for r in candidates),
        "control_blends": [r["blend"] for r in rows if "control" in r["blend"].lower()],
        "candidate_blends": [r["blend"] for r in candidates],
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT]; bytes EXACT",
        "score_claim": False,
        "promotable": False,
    }
    write_json(Path(args.out), report)
    for r in rows:
        print(
            f"{r['blend']:<22} n={r['pairs_solved']:<3} dB {r['delta_archive_bytes']:+6d} "
            f"ratio {r['paired_ratio_final_over_base']:8.3f} "
            f"break-even {r['break_even_ratio']:6.3f} "
            f"miss {r['misses_break_even_by']:8.2f}x "
            f"worse {r['pairs_worse']}/{r['pairs_solved']}"
        )
    return 0


def composed_score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return 100.0 * d_seg + math.sqrt(10.0 * d_pose) + RATE_PER_BYTE * archive_bytes


def cmd_price(args) -> int:
    """Build the candidate archive exactly and price S from components."""
    set_threads(args.threads)
    receipts = assert_pointer()
    import ddm_up3_carrier_splice as splice

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    basis_codes = np.load(args.basis_codes).astype(np.int64)
    coefficient_codes = np.load(args.coefficient_codes).astype(np.int32)
    if coefficient_codes.shape != (N_PAIRS, CARRIER_DIM):
        raise Cb1Error(f"coefficient codes have shape {coefficient_codes.shape}")

    body = splice.parse_shipped_body(POINTER_TREE, verify_sha=False)
    identity = splice.build_archive(
        body, np.asarray(body.codes, dtype=np.int32), runtime_dir=POINTER_TREE,
        container_search=True, verify=True,
    )
    if identity["archive_sha256"] != POINTER_ARCHIVE_SHA256:
        raise Cb1Error("CONTROL FAILED: the identity repack does not reproduce move 49")

    fields = encode_basis_fields(basis_codes)
    candidate_body = dataclasses.replace(
        body,
        basis_bits=fields["bits"],
        basis_blob=fields["payload"],
        lengths=np.asarray(fields["lengths"], dtype=np.uint8),
    )
    built = splice.build_archive(
        candidate_body, coefficient_codes, runtime_dir=POINTER_TREE,
        container_search=True, verify=True,
    )
    archive_path = out / "candidate_archive.zip"
    archive_path.write_bytes(built["archive_bytes"])

    proof = jg5.section_identity(
        built["archive_bytes"], body.archive_bytes, runtime=POINTER_TREE
    )
    if not proof["frame1_sections_all_identical"]:
        raise Cb1Error(
            f"a frame-1 section moved ({proof}); the seg leg cannot be carried"
        )
    archive_bytes = int(built["archive_size"])
    score = composed_score(POINTER_D_SEG_T4, args.d_pose, archive_bytes)
    report = {
        "schema": "ddm_cb1_price.v1",
        "receipts": receipts,
        "identity_control_sha256": identity["archive_sha256"],
        "candidate": {
            "path": str(archive_path),
            "sha256": built["archive_sha256"],
            "bytes": archive_bytes,
        },
        "delta_archive_bytes": archive_bytes - POINTER_ARCHIVE_BYTES,
        "basis_section": {
            "huffman_bytes_shipped": len(body.basis_blob),
            "huffman_bytes_candidate": fields["huffman_bytes"],
            "rider_arith_bytes_candidate": fields["rider_arith_bytes"],
        },
        "frame1_section_identity": proof,
        "d_seg_t4_carried": POINTER_D_SEG_T4,
        "d_pose": args.d_pose,
        "score_projected": score,
        "pointer_score_t4": POINTER_SCORE_T4,
        "net_dS_vs_pointer": score - POINTER_SCORE_T4,
        "admit_bar": ADMIT_BAR,
        "clears_admit_bar": bool(score - POINTER_SCORE_T4 <= ADMIT_BAR),
        "axis": (
            "seg CARRIED from move 49 (frame-1 sections byte-identical); pose "
            "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT]; bytes EXACT"
        ),
        "score_claim": False,
        "promotable": False,
    }
    write_json(out / "PRICE.json", report)
    print(json.dumps({k: report[k] for k in ("delta_archive_bytes", "net_dS_vs_pointer")}))
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    base = sub.add_parser("base", help="reproduce move 49's n600 pose base")
    base.add_argument("--out-dir", default=str(WORK / "base"))
    base.add_argument("--batch-size", type=int, default=8)
    base.add_argument("--threads", type=int, default=6)
    base.set_defaults(func=cmd_base)

    twins = sub.add_parser("twins", help="repack the shipped carrier byte-identically")
    twins.add_argument("--out-dir", default=str(WORK / "base"))
    twins.add_argument("--threads", type=int, default=4)
    twins.set_defaults(func=cmd_twins)

    space = sub.add_parser("subspace", help="per-pair field Jacobian, residual, free step")
    space.add_argument("--out-dir", default=str(WORK / "subspace"))
    space.add_argument("--pairs", type=int, default=600)
    space.add_argument("--seed", type=int, default=20260912)
    space.add_argument("--shard-index", type=int, default=0)
    space.add_argument("--shard-count", type=int, default=1)
    space.add_argument("--threads", type=int, default=4)
    space.set_defaults(func=cmd_subspace)

    analyse = sub.add_parser("analyse", help="SVD, energy tables, subspace angles")
    analyse.add_argument("--subspace-dir", default=str(WORK / "subspace"))
    analyse.add_argument("--out-dir", default=str(WORK / "subspace"))
    analyse.add_argument("--threads", type=int, default=6)
    analyse.set_defaults(func=cmd_analyse)

    refit = sub.add_parser("refit", help="build, quantise and price the blends")
    refit.add_argument("--subspace-dir", default=str(WORK / "subspace"))
    refit.add_argument("--out-dir", default=str(WORK / "refit"))
    refit.add_argument("--keep", default="12,11,10,9,8,6")
    refit.add_argument("--directions", choices=("residual", "target"), default="residual")
    refit.add_argument("--threads", type=int, default=6)
    refit.set_defaults(func=cmd_refit)

    partial = sub.add_parser(
        "partial", help="trace one atom from the incumbent to its replacement"
    )
    partial.add_argument("--subspace-dir", default=str(WORK / "subspace"))
    partial.add_argument("--out-dir", default=str(WORK / "partial"))
    partial.add_argument("--slot", type=int, required=True)
    partial.add_argument("--alphas", default="0.25,0.5,0.75")
    partial.add_argument("--threads", type=int, default=4)
    partial.set_defaults(func=cmd_partial)

    predict = sub.add_parser("predict", help="linearized screen of the built blends")
    predict.add_argument("--subspace-dir", default=str(WORK / "subspace"))
    predict.add_argument("--out-dir", default=str(WORK / "refit"))
    predict.add_argument("--threads", type=int, default=6)
    predict.set_defaults(func=cmd_predict)

    resolve = sub.add_parser("resolve", help="canonical per-pair re-solve on a basis")
    resolve.add_argument("--basis-codes", type=Path, required=True)
    resolve.add_argument("--base-pose", type=Path, required=True)
    resolve.add_argument("--subspace-dir", default=str(WORK / "subspace"))
    resolve.add_argument("--out-dir", required=True)
    resolve.add_argument("--pairs", type=int, default=48)
    resolve.add_argument("--seed", type=int, default=20260912)
    resolve.add_argument("--shard-index", type=int, default=0)
    resolve.add_argument("--shard-count", type=int, default=1)
    resolve.add_argument(
        "--warm-start", choices=("field", "pose"), default="field",
        help="field = least-squares projection (pc1's V2/V4 start); pose = the "
             "minimum-image-norm step that cancels the linearised pose residual",
    )
    resolve.add_argument("--outer-rounds", type=int, default=40)
    resolve.add_argument("--max-gn-iterations", type=int, default=400)
    resolve.add_argument("--threads", type=int, default=2)
    resolve.set_defaults(func=cmd_resolve)

    verdict = sub.add_parser("verdict", help="assemble the blends against break-even")
    verdict.add_argument("--base-pose", type=Path, required=True)
    verdict.add_argument("--byteonly", type=Path, required=True)
    verdict.add_argument("--blend", action="append", required=True,
                         metavar="NAME=ROWS_DIR=BYTEONLY_KEY")
    verdict.add_argument("--out", type=Path, required=True)
    verdict.add_argument("--threads", type=int, default=4)
    verdict.set_defaults(func=cmd_verdict)

    price = sub.add_parser("price", help="build the candidate archive and price it")
    price.add_argument("--basis-codes", type=Path, required=True)
    price.add_argument("--coefficient-codes", type=Path, required=True)
    price.add_argument("--d-pose", type=float, required=True)
    price.add_argument("--out-dir", required=True)
    price.add_argument("--threads", type=int, default=4)
    price.set_defaults(func=cmd_price)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
