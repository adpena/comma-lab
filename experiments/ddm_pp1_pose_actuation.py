#!/usr/bin/env python3
"""ddm_pp1: POSE-DIRECTED per-pair actuation on the pairs that carry the pose term.

The object is the move-49 body (S 0.13632299781031237 @ 179,153 B, archive sha
``73e41a66...``).  Pose is concentrated: a handful of pairs hold half the d_pose mass,
and the shipped per-pair ``frame_embed`` codes are the only renderer-side object that
can be moved for ONE pair without touching the other 599.

Every stage here measures; nothing models.  The chain per candidate move is exactly the
shipped chain, and it is the same chain ``ddm_fe1_pose_price`` walks:

1. render frame ``2p+1`` under the moved codes (the receiver's batch-1 forward);
2. splice that frame over the live decode (odd frames only);
3. measure the pair's d_pose STALE against the live carrier codes;
4. re-solve that pair's twelve carrier coefficients with ``jg5.refine_pair``;
5. measure the pair's d_pose RESOLVED -- the number the admission uses;
6. re-verify the pair's SegNet argmax on the moved render and charge any new cell.

The base is read from move 49's own artifacts and NOTHING is inherited: the live tree,
the live decode (its own cold parse-back ``0.raw``), the live token field and the live
carrier codes all come from the move-49 custody directory, and each is identity-checked
before a number is produced.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # the pointer tree is read-only custody

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_br1_pose_basis_reorientation as br1
import ddm_fe1_frame_embedding_search as fe1
import ddm_jg1_seg_solve as jg1
import ddm_jg5_pose_resolve_on_edited_renders as jg5
import ddm_sj1_multipass_token_predistortion as sj1
import ddm_up2_shipping_pose_solve as up2

N_PAIRS = jg1.N_PAIRS
EVAL_H, EVAL_W = jg1.EVAL_H, jg1.EVAL_W
CAMERA_H, CAMERA_W = jg1.CAMERA_H, jg1.CAMERA_W
FRAME_DIM = fe1.FRAME_DIM
CODE_MIN, CODE_MAX = fe1.CODE_MIN, fe1.CODE_MAX
CELL_COUNT = N_PAIRS * EVAL_H * EVAL_W
S_PER_SEG_CELL = jg1.S_PER_SEG_CELL
RATE_PER_BYTE = jg1.S_PER_ARCHIVE_BYTE

#: MOVE 49 -- the live pointer.  Taken from sj1's own lineage table so the identity
#: cannot drift from the row the campaign banked; asserted, never retyped.
POINTER = sj1.LIVE_POINTER
POINTER_TREE = POINTER.tree
POINTER_ARCHIVE = POINTER_TREE / "archive.zip"
POINTER_ARCHIVE_SHA256 = POINTER.archive_sha256
POINTER_ARCHIVE_BYTES = POINTER.archive_bytes
POINTER_D_POSE_T4 = POINTER.d_pose_t4
POINTER_D_SEG_T4 = POINTER.d_seg_t4
POINTER_SCORE_T4 = POINTER.score_t4

PASS7_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_sj1_pass7")
#: The candidate's OWN cold parse-back of the shipped bytes: the decode the T4 row scored.
LIVE_RAW = PASS7_ROOT / "parseback/0.raw"
#: The token field those bytes decode to (sj1 pass 7 F-gate: 600/600 planes identical to
#: the decoded plane, 0 differing).
LIVE_FIELD = PASS7_ROOT / "admission_pass7/field_admitted.npz"
#: The n600 argmax of that decode -- sj1's own seg leg receipt for the shipped row.
LIVE_ARGMAX = PASS7_ROOT / "seg_final/argmax_n600.npy"

WORK = Path(os.environ.get("PP1_WORK", "/Volumes/VertigoDataTier/pact/ddm_pp1"))


class Pp1Error(RuntimeError):
    """A pp1 identity or contract failed; the caller must stop."""


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


def open_live_raw():
    expected = 2 * N_PAIRS * CAMERA_H * CAMERA_W * 3
    actual = LIVE_RAW.stat().st_size
    if actual != expected:
        raise Pp1Error(f"decode {LIVE_RAW} is {actual} B, expected {expected} B")
    return np.memmap(
        LIVE_RAW, dtype=np.uint8, mode="r", shape=(2 * N_PAIRS, CAMERA_H, CAMERA_W, 3)
    )


def assert_pointer_identity() -> dict[str, Any]:
    """Refuse every stage that is not standing on move 49's own shipped bytes."""
    observed = sha256_file(POINTER_ARCHIVE)
    if observed != POINTER_ARCHIVE_SHA256:
        raise Pp1Error(
            f"pointer tree {POINTER_TREE} has archive sha {observed}, not move 49's "
            f"{POINTER_ARCHIVE_SHA256}"
        )
    size = POINTER_ARCHIVE.stat().st_size
    if size != POINTER_ARCHIVE_BYTES:
        raise Pp1Error(f"pointer archive is {size} B, not {POINTER_ARCHIVE_BYTES}")
    return {
        "pointer_tree": str(POINTER_TREE),
        "pointer_archive_sha256": observed,
        "pointer_archive_bytes": size,
        "pointer_score_t4": POINTER_SCORE_T4,
        "pointer_d_pose_t4": POINTER_D_POSE_T4,
        "pointer_d_seg_t4": POINTER_D_SEG_T4,
        "live_raw": str(LIVE_RAW),
        "live_field": str(LIVE_FIELD),
        "live_argmax": str(LIVE_ARGMAX),
    }


# ----------------------------------------------------------------------------------
# The semantic section on MOVE 49 -- SM1S, not RC1S
# ----------------------------------------------------------------------------------
#
# fe1 read the section as ``brotli -> CK2 -> RC1 rider``.  MEASURED on move 49's own
# bytes, the rider is ``SM1S`` (sm1's counted 24-weight arithmetic mixer, which fe1's
# own handoff named as the successor coder), 31,451 B where fe1's RC1S body was
# 30,246 B.  The SM3R body underneath is the same object -- the frame_embed run is
# still a signed 3-bit codes run located by the receiver's own ``walk_sm3r`` -- but
# the ENCODER above it is different, so fe1's loader refuses (correctly) and this arm
# supplies its own.  Every byte here comes from the pointer tree's own modules; no
# coder is re-implemented.


class Sm1Section:
    """MOVE 49's semantic section, split into the pieces a frame_embed edit needs."""

    def __init__(self, *, container, codes, scales, code_offset, code_length,
                 code_bits, weights, brotli_quality, brotli_lgwin, rc1, sm1) -> None:
        self.container = container
        self.rider = container["rider"]
        self.member = container["member"]
        self.staged = container["staged"]
        self.sm3r_body = container["body"]
        self.template = container["template"]
        self.ck2 = bool(container["ck2_semantic"])
        self.sz1_split = bool(container["sz1_split"])
        self.codes = codes
        self.scales = scales
        self.code_offset = code_offset
        self.code_length = code_length
        self.code_bits = code_bits
        self.weights = weights
        self.brotli_quality = brotli_quality
        self.brotli_lgwin = brotli_lgwin
        self.rc1 = rc1
        self.sm1 = sm1

    def body_with_codes(self, codes: np.ndarray) -> bytes:
        """The SM3R body with ``frame_embed`` replaced -- same length by construction."""
        codes = np.asarray(codes, dtype=np.int64)
        if codes.shape != (N_PAIRS, FRAME_DIM):
            raise Pp1Error(f"codes must be (600, 8), got {codes.shape}")
        if codes.min() < CODE_MIN or codes.max() > CODE_MAX:
            raise Pp1Error(
                f"codes escape the shipped signed {self.code_bits}-bit domain "
                f"[{CODE_MIN}, {CODE_MAX}]"
            )
        packed = self.rc1.pack_signed_codes(
            codes.ravel().astype(np.int32), self.code_bits
        )
        if len(packed) != self.code_length:
            raise Pp1Error(
                f"repacked frame_embed is {len(packed)} B, shipped run is "
                f"{self.code_length} B"
            )
        return (
            self.sm3r_body[: self.code_offset]
            + packed
            + self.sm3r_body[self.code_offset + self.code_length :]
        )

    def rider_with_codes(self, codes: np.ndarray) -> bytes:
        """The SM1S semantic rider carrying ``codes``, through the SHIPPED mixer.

        The 24 mixer weights are the SHIPPED ones, carried verbatim: they are part of
        the rider the receiver reads, so re-fitting them here would be a different
        object (and, being fitted on the video, would also be counted content).
        """
        rider, _payload, _metadata = self.sm1.encode(
            self.body_with_codes(codes), self.template, self.weights
        )
        return rider

    def member_with_codes(
        self, codes: np.ndarray, *, quality: int | None = None, lgwin: int | None = None
    ) -> bytes:
        """The COMPRESSED semantic member, the object the archive actually carries.

        The chain below the body is ``ren2_init.encode_member``'s verbatim: SM1S ->
        (CK2 plane un-interleave) -> Brotli at the shipped shape.  ``ren2_init`` starts
        from the float state and re-packs the SM3R body; this arm starts from the
        SHIPPED body with one code run replaced, so the per-tensor scales and every
        other tensor are the shipped bytes by construction rather than by re-derivation.
        """
        import brotli

        rider = self.rider_with_codes(codes)
        staged = rider
        if self.sz1_split:
            raise Pp1Error("SZ1 semantic split is set; this arm has not been reviewed for it")
        if self.ck2:
            span = len(rider) & ~1
            planes = np.frombuffer(rider[:span], dtype=np.uint8)
            staged = planes[0::2].tobytes() + planes[1::2].tobytes() + rider[span:]
        member = brotli.compress(
            staged,
            quality=self.brotli_quality if quality is None else quality,
            lgwin=self.brotli_lgwin if lgwin is None else lgwin,
        )
        if brotli.decompress(member) != staged:
            raise Pp1Error("member Brotli container failed its own round trip")
        return member


def import_pointer_runtime():
    """The move-49 receiver's OWN modules, loaded from the move-49 tree."""
    import importlib

    runtime_dir = POINTER_TREE / "runtime"
    for entry in (str(runtime_dir.parent), str(runtime_dir.parent / "cpr1")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    ra = importlib.import_module("runtime.residual_archive")
    rc1 = importlib.import_module("runtime.rc1_adaptive_model_sections")
    sm1 = importlib.import_module("runtime.sm1_semantic_mixer")
    renderer = importlib.import_module("inflate")
    return ra, rc1, sm1, renderer


def read_container() -> dict[str, Any]:
    """MOVE 49's semantic container, through ``ddm_ren2_restore_init``'s own reader.

    ``ren2_init`` binds move 48's tree at module scope; the reader resolves it at CALL
    time, so re-pointing the global is the whole re-base and no producer is edited.
    MEASURED and asserted here: move 49's semantic member is move 48's member byte for
    byte (pass 7 moved the token tail and the carrier, never the renderer).
    """
    import ddm_ren2_restore_init as ren2_init

    original = ren2_init.POINTER_TREE
    try:
        ren2_init.POINTER_TREE = POINTER_TREE
        container = ren2_init.read_shipped_semantic_container()
    finally:
        ren2_init.POINTER_TREE = original
    sizes = (
        len(container["body"]), len(container["rider"]), len(container["member"])
    )
    expected = (
        ren2_init.SHIPPED_BODY_BYTES,
        ren2_init.SHIPPED_RIDER_BYTES,
        ren2_init.SHIPPED_MEMBER_BYTES,
    )
    if sizes != expected:
        raise Pp1Error(
            f"move 49's semantic chain is {sizes} B; move 48's pinned chain is {expected}"
        )
    if hashlib.sha256(container["body"]).hexdigest() != ren2_init.SHIPPED_BODY_SHA256:
        raise Pp1Error("move 49's SM3R body is not move 48's pinned body")
    quality, lgwin = ren2_init.identify_brotli_container(
        container["staged"], container["member"]
    )
    container["brotli_quality"] = int(quality)
    container["brotli_lgwin"] = int(lgwin)
    container["member_sha256"] = hashlib.sha256(container["member"]).hexdigest()
    return container


def load_sm1_section() -> Sm1Section:
    """Locate ``frame_embed``'s code run in move 49's SM1S section, exactly.

    The run is found by DRIVING the receiver's own ``walk_sm3r`` over the restored
    body, and the open is refused unless re-encoding the SHIPPED codes through the
    shipped mixer and the shipped container reproduces the shipped MEMBER byte for byte.
    """
    _ra, rc1, sm1, _renderer = import_pointer_runtime()
    container = read_container()
    rider = container["rider"]
    body = container["body"]
    template = container["template"]
    if not body.startswith(rc1.SM3R_MAGIC):
        raise Pp1Error(f"restored semantic body starts {body[:4]!r}, not SM3R")
    _magic, _version, count, _reserved, length = sm1.HEADER.unpack_from(rider)
    end = len(rider) - length - 24
    weights = np.frombuffer(rider[end : end + 24], dtype=np.int8)
    if int(count) != 24 or weights.size != 24:
        raise Pp1Error("SM1 mixer weight vector is not the shipped 24")

    version, mode, keep_percent, reserved = body[4:8]
    cursor = 10
    offsets: list[int] = []

    def read(_kind: str, run_length: int) -> bytes:
        nonlocal cursor
        offsets.append(cursor)
        chunk = body[cursor : cursor + run_length]
        cursor += run_length
        return chunk

    plan = rc1.walk_sm3r(read, template, (version, mode, keep_percent, reserved))
    if cursor != len(body):
        raise Pp1Error(f"SM3R walk ended at {cursor} of {len(body)}")

    index = 1  # plan[0] is the depth table
    located: dict[str, dict[str, Any]] = {}
    for name, value in template.items():
        if value.ndim < 2:
            index += 1
            continue
        if name in rc1.ROW_PRUNE_NAMES:
            index += 1  # prune mask
        scale_item, scale_off = plan[index], offsets[index]
        index += 1
        code_item, code_off = plan[index], offsets[index]
        index += 1
        if code_item["kind"] != "codes":
            raise Pp1Error(f"{name}: expected a codes run, got {code_item['kind']}")
        located[name] = {
            "scales": (scale_off, scale_item["length"]),
            "codes": (code_off, code_item["length"]),
            "bits": code_item["bits"],
            "count": code_item["count"],
        }

    fe = located["frame_embed.weight"]
    bits = int(fe["bits"])
    if bits != fe1.CODE_BITS:
        raise Pp1Error(f"frame_embed depth is {bits} bits, module pinned {fe1.CODE_BITS}")
    scale_off, scale_len = fe["scales"]
    scales = np.frombuffer(body[scale_off : scale_off + scale_len], dtype="<f2")
    code_off, code_len = fe["codes"]
    codes = rc1.unpack_signed_codes(
        body[code_off : code_off + code_len], int(fe["count"]), bits
    ).reshape(N_PAIRS, FRAME_DIM)

    section = Sm1Section(
        container=container,
        codes=codes.astype(np.int8),
        scales=np.asarray(scales, dtype=np.float32),
        code_offset=int(code_off),
        code_length=int(code_len),
        code_bits=bits,
        weights=weights,
        brotli_quality=container["brotli_quality"],
        brotli_lgwin=container["brotli_lgwin"],
        rc1=rc1,
        sm1=sm1,
    )
    if section.rider_with_codes(section.codes) != rider:
        raise Pp1Error(
            "re-encoding the SHIPPED frame_embed codes through the shipped SM1 mixer "
            "did not reproduce the shipped rider: the coder this arm would price with "
            "is not the coder the archive ships"
        )
    if section.member_with_codes(section.codes) != container["member"]:
        raise Pp1Error(
            "the shipped codes do not repack to the shipped semantic MEMBER; the "
            "container this arm would price with is not the shipped container"
        )
    return section


# ----------------------------------------------------------------------------------
# The body: the move-49 renderer, its frame_embed codes, its token field, its decode
# ----------------------------------------------------------------------------------


def load_body(*, with_raw: bool = False, with_segnet: bool = True) -> fe1.Body:
    """fe1's Body on MOVE 49's objects.

    fe1's own ``load_body`` binds move 34's tree through default arguments, which are
    evaluated at def time and so cannot be re-pointed by assigning the module constant.
    Every path is therefore passed explicitly here, and the section comes from this
    arm's own SM1S loader.
    """
    up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=up2.LINEAGE_DALI)
    receipts = assert_pointer_identity()
    receipts["gt_lineage"] = up2.LINEAGE_DALI
    tokens = fe1.load_token_field(LIVE_FIELD)
    gt = jg1.load_gt_seg_labels(up2.LINEAGE_DALI)
    semantic = jg1.load_semantic_renderer(
        archive_path=POINTER_ARCHIVE, runtime_dir=POINTER_TREE / "runtime"
    )
    section = load_sm1_section()
    weight = semantic.frame_embed.weight.detach().numpy()
    recon = section.codes.astype(np.float32) * section.scales[None, :]
    if not np.array_equal(weight, recon):
        raise Pp1Error(
            "the codes located in the SM3R body do not reconstruct the renderer's "
            "loaded frame_embed weight"
        )
    net = jg1.load_segnet() if with_segnet else None
    raw = open_live_raw() if with_raw else None
    return fe1.Body(
        semantic=semantic,
        net=net,
        tokens=tokens,
        gt=gt,
        section=section,
        raw=raw,
        receipts=receipts,
    )


def build_pose_instrument(raw):
    """br1's instrument on MOVE 49's carrier and MOVE 49's decode."""
    state = up2.load_carrier_state(POINTER_TREE, verify_archive=False)
    targets, lineage = up2.load_gt_poses(up2.DEFAULT_DALI_GT)
    if lineage != up2.LINEAGE_DALI:
        raise Pp1Error(f"GT pose lineage is {lineage}, not {up2.LINEAGE_DALI}")
    up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=lineage)
    posenet = up2.load_posenet()
    up2.enable_posenet_gradients()
    blow = br1.low_basis(state)
    gram, bmat = br1.span_gram(blow)
    return br1.Instrument(state, raw, targets, posenet, blow, gram, bmat)


class MemoryOverlayRaw:
    """The live decode with named ODD frames replaced, served the way br1 reads it.

    Copied in shape from ``ddm_fe1_pose_price._MemoryOverlayRaw``: a frame-embedding
    move can only reach frame ``2p+1``, so only odd indices are ever overridden and an
    even index always falls through to the shipped decode.
    """

    def __init__(self, base, frames: dict[int, np.ndarray]) -> None:
        self.base = base
        self.frames = frames
        self.shape = base.shape
        self.dtype = base.dtype

    def __getitem__(self, key):
        arr = np.asarray(key)
        if arr.ndim == 0:
            index = int(arr)
            if index in self.frames:
                return self.frames[index]
            return self.base[index]
        out = np.empty((len(arr), CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
        for position, index in enumerate(int(i) for i in arr.reshape(-1)):
            out[position] = (
                self.frames[index] if index in self.frames else self.base[index]
            )
        return out


# ----------------------------------------------------------------------------------
# stage: base -- the per-pair pose base on move 49, with its controls
# ----------------------------------------------------------------------------------


def cmd_base(args) -> int:
    set_threads(args.threads)
    started = time.time()
    receipts = assert_pointer_identity()
    raw = open_live_raw()
    inst = build_pose_instrument(raw)
    codes = np.asarray(inst.state.codes, dtype=np.int32)
    if codes.shape != (N_PAIRS, up2.CARRIER_DIM):
        raise Pp1Error(f"carrier codes have shape {codes.shape}")
    coefficients = up2.codes_to_coefficients(codes, inst.state.coefficient_scales)
    indices = np.arange(N_PAIRS, dtype=np.int64)
    per_pair, _poses = up2.measure_pose(
        inst.posenet,
        inst.state,
        coefficients,
        inst.raw,
        inst.targets,
        indices,
        batch_size=args.batch_size,
    )
    mean = float(per_pair.mean())
    ratio = mean / POINTER_D_POSE_T4
    order = np.argsort(-per_pair)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "pose_base_move49.npy", per_pair)
    top = [
        {
            "rank": int(r),
            "pair": int(order[r]),
            "d_pose": float(per_pair[order[r]]),
            "share": float(per_pair[order[r]] / per_pair.sum()),
        }
        for r in range(args.top)
    ]
    cumulative = float(per_pair[order[: args.top]].sum() / per_pair.sum())
    report = {
        "schema": "ddm_pp1_pose_base.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT, n600]",
        "score_claim": False,
        "receipts": receipts,
        "pairs": N_PAIRS,
        "d_pose_mean": mean,
        "pose_leg": math.sqrt(10.0 * mean),
        "t4_print": POINTER_D_POSE_T4,
        "instrument_ratio_vs_t4_print": ratio,
        "d_pose_median": float(np.median(per_pair)),
        "d_pose_max": float(per_pair.max()),
        "top_k": top,
        "top_k_cumulative_share": cumulative,
        "per_pair_path": str(out_dir / "pose_base_move49.npy"),
        "per_pair_sha256": sha256_file(out_dir / "pose_base_move49.npy"),
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "POSE_BASE_MOVE49.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: report[k] for k in (
        "d_pose_mean", "instrument_ratio_vs_t4_print", "top_k_cumulative_share",
        "elapsed_seconds")}, indent=1))
    print("top pairs:", [row["pair"] for row in top])
    return 0


# ----------------------------------------------------------------------------------
# stage: controls
# ----------------------------------------------------------------------------------


def cmd_controls(args) -> int:
    """Three identity controls, each refusing a different silent-wrong-object class."""
    set_threads(args.threads)
    started = time.time()
    pairs = [int(p) for p in args.pairs.split(",")]
    body = load_body(with_raw=True, with_segnet=True)
    rows: list[dict[str, Any]] = []

    # C1 -- the semantic render reproduces the shipped decode's ODD frames exactly.
    for pair in pairs:
        frame = fe1.render_pair(body, pair)[0]
        shipped = np.asarray(body.raw[2 * pair + 1])
        deviation = int(np.abs(frame.astype(np.int32) - shipped.astype(np.int32)).max())
        mismatched = int((frame != shipped).sum())
        rows.append({
            "control": "C1_semantic_render_identity",
            "pair": pair,
            "max_abs_deviation": deviation,
            "mismatched_pixels": mismatched,
            "pass": deviation == 0,
        })
        print(f"C1 pair {pair}: max|dev| {deviation}, mismatched {mismatched}", flush=True)

    # C2 -- the carrier render reproduces the shipped decode's EVEN frames exactly.
    import torch

    state = up2.load_carrier_state(POINTER_TREE, verify_archive=False)
    for pair in pairs:
        index = np.array([pair], dtype=np.int64)
        coefficients = up2.codes_to_coefficients(
            np.asarray(state.codes, dtype=np.int32)[index], state.coefficient_scales
        )
        with torch.inference_mode():
            frame0 = up2.render_frame0(coefficients, state, index, differentiable=False)
        got = (
            frame0[0].permute(1, 2, 0).to(torch.float64).numpy().round().astype(np.int32)
        )
        shipped = np.asarray(body.raw[2 * pair]).astype(np.int32)
        deviation = int(np.abs(got - shipped).max())
        rows.append({
            "control": "C2_carrier_render_identity",
            "pair": pair,
            "max_abs_deviation": deviation,
            "pass": deviation == 0,
        })
        print(f"C2 pair {pair}: max|dev| {deviation}", flush=True)

    # C3 -- the per-pair SegNet argmax equals the shipped decode's argmax, cell for cell.
    shipped_argmax = np.load(LIVE_ARGMAX, mmap_mode="r")
    for pair in pairs:
        mine = fe1.argmax_pair(body, pair)
        theirs = np.asarray(shipped_argmax[pair])
        disagree = int((mine != theirs).sum())
        my_flips = int((mine != body.gt[pair]).sum())
        their_flips = int((theirs != body.gt[pair]).sum())
        rows.append({
            "control": "C3_segnet_argmax_identity",
            "pair": pair,
            "cells_disagreeing": disagree,
            "flips_mine": my_flips,
            "flips_shipped": their_flips,
            "pass": disagree == 0 and my_flips == their_flips,
        })
        print(
            f"C3 pair {pair}: disagreeing {disagree}, flips {my_flips}/{their_flips}",
            flush=True,
        )

    # C4 -- re-solving an UNTOUCHED pair's carrier moves d_pose by exactly 0.
    if args.resolve_control_pairs:
        inst = build_pose_instrument(body.raw)
        live_codes = np.asarray(inst.state.codes, dtype=np.int32)
        dd_threshold = jg5.materiality_dd_threshold(args.base_mean_d_pose)
        for pair in (int(p) for p in args.resolve_control_pairs.split(",")):
            before = float(br1.evaluate_codes(inst, pair, live_codes[pair][None])[0])
            refined = jg5.refine_pair(
                inst, pair, live_codes[pair], dd_threshold=dd_threshold,
                outer_rounds=args.outer_rounds, max_gn_iterations=args.max_gn_iterations,
            )
            after = float(refined["final_d_pose"])
            rows.append({
                "control": "C4_unmoved_resolve_is_zero",
                "pair": pair,
                "d_pose_before": before,
                "d_pose_after": after,
                "delta": after - before,
                "codes_moved": int((np.asarray(refined["codes"]) != live_codes[pair]).sum()),
                "pass": after == before,
            })
            print(
                f"C4 pair {pair}: {before:.6e} -> {after:.6e} (delta {after - before:+.3e})",
                flush=True,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "ddm_pp1_controls.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch instruments, DALI GT]",
        "score_claim": False,
        "receipts": body.receipts,
        "rows": rows,
        "all_pass": all(row["pass"] for row in rows),
        "elapsed_seconds": time.time() - started,
    }
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({"out": str(out), "all_pass": report["all_pass"]}))
    return 0 if report["all_pass"] else 3


# ----------------------------------------------------------------------------------
# stage: searchA -- per-pair frame_embed pose descent
# ----------------------------------------------------------------------------------


def single_code_moves(row: np.ndarray) -> list[tuple[int, int]]:
    """Every reachable single-code alternative on the shipped 3-bit signed lattice."""
    out: list[tuple[int, int]] = []
    for dim in range(FRAME_DIM):
        for code in range(CODE_MIN, CODE_MAX + 1):
            if code != int(row[dim]):
                out.append((dim, code))
    return out


def _resolved_row(
    body: fe1.Body,
    base_inst,
    raw,
    pair: int,
    new_row: np.ndarray,
    *,
    live_codes: np.ndarray,
    dd_threshold: float,
    outer_rounds: int,
    max_gn_iterations: int,
    base_flips: int,
    d_pose_base: float,
    base_pose_mean: float,
    with_seg: bool,
) -> dict[str, Any]:
    """Realize one candidate code row end to end: render, seg, stale pose, resolved pose."""
    fe1.set_pair_codes(body, pair, new_row)
    frame = fe1.render_pair(body, pair)
    if with_seg:
        moved_flips = fe1.flips_pair(
            jg1.argmax_from_camera_frames(body.net, frame)[0], body, pair
        )
    else:
        moved_flips = base_flips
    fe1.restore_pair_codes(body, pair)
    overlay = MemoryOverlayRaw(raw, {2 * pair + 1: frame[0]})
    moved_inst = br1.Instrument(
        base_inst.state, overlay, base_inst.targets, base_inst.posenet,
        base_inst.blow, base_inst.gram, base_inst.bmat,
    )
    stale = float(br1.evaluate_codes(moved_inst, pair, live_codes[pair][None])[0])
    refined = jg5.refine_pair(
        moved_inst, pair, live_codes[pair], dd_threshold=dd_threshold,
        outer_rounds=outer_rounds, max_gn_iterations=max_gn_iterations,
    )
    resolved = float(refined["final_d_pose"])
    base_leg = math.sqrt(10.0 * base_pose_mean)
    d_s_pose = (
        math.sqrt(10.0 * (base_pose_mean + (resolved - d_pose_base) / N_PAIRS)) - base_leg
    )
    d_cells = moved_flips - base_flips
    return {
        "codes": [int(c) for c in new_row],
        "flips": moved_flips,
        "d_cells": d_cells,
        "d_pose_stale": stale,
        "d_pose_resolved": resolved,
        "carrier_codes": [int(c) for c in refined["codes"]],
        "refine": {
            k: refined[k] for k in refined
            if k in ("rounds", "evaluations", "gn_iterations", "polish_steps", "stop_reason")
        },
        "dS_seg": 100.0 * d_cells / CELL_COUNT,
        "dS_pose_resolved": d_s_pose,
    }


def cmd_rate_fee(args) -> int:
    """MEASURE the member cost of N changed frame_embed codes ON THE LIVE SHAPE.

    fe1's container law (+70 B flat at the shipped shape, recoverable to +1.3 B by a
    q/lgwin/ck2 search) was measured on the OLD container: a ``brotli -> CK2 -> RC1``
    chain whose rider was a range-coded stream.  Move 49 ships ``SM1S`` -- a COUNTED
    24-weight arithmetic mixer -- so the code run is priced per SYMBOL by a model, not
    by a container break.  Inheriting fe1's fee would be the wrong-coder class this
    campaign already paid for once (sj1 pass 7 sec.3).  It is re-measured here.
    """
    section = load_sm1_section()
    base_member = len(section.member)
    counts = [int(c) for c in args.counts.split(",")]
    rng = np.random.default_rng(args.seed)
    grid = [(q, lg) for q in (9, 10, 11) for lg in (16, 18, 20, 22, 24)]
    started = time.time()
    rows: list[dict[str, Any]] = []
    for count in counts:
        for repeat in range(args.repeats):
            codes = section.codes.astype(np.int64).copy()
            picks = []
            for _ in range(count):
                pair = int(rng.integers(0, N_PAIRS))
                dim = int(rng.integers(0, FRAME_DIM))
                old = int(codes[pair, dim])
                choices = [c for c in range(CODE_MIN, CODE_MAX + 1) if c != old]
                new = int(choices[int(rng.integers(0, len(choices)))])
                codes[pair, dim] = new
                picks.append([pair, dim, old, new])
            shipped = len(section.member_with_codes(codes))
            if args.search_container:
                searched = min(
                    len(section.member_with_codes(codes, quality=q, lgwin=lg))
                    for q, lg in grid
                )
            else:
                searched = shipped
            rows.append({
                "count": count, "repeat": repeat, "picks": picks,
                "member_bytes_shipped_shape": shipped,
                "d_member_shipped_shape": shipped - base_member,
                "member_bytes_searched": searched,
                "d_member_searched": searched - base_member,
            })
            print(
                f"N={count} r{repeat}: shipped {shipped - base_member:+d} B, "
                f"searched {searched - base_member:+d} B",
                flush=True,
            )
    summary = {}
    for count in counts:
        sel = [r for r in rows if r["count"] == count]
        shipped = np.array([r["d_member_shipped_shape"] for r in sel], dtype=float)
        searched = np.array([r["d_member_searched"] for r in sel], dtype=float)
        summary[str(count)] = {
            "n": len(sel),
            "shipped_mean": float(shipped.mean()),
            "shipped_sd": float(shipped.std(ddof=1)) if len(sel) > 1 else 0.0,
            "shipped_min": float(shipped.min()),
            "shipped_max": float(shipped.max()),
            "searched_mean": float(searched.mean()),
            "searched_min": float(searched.min()),
        }
    report = {
        "schema": "ddm_pp1_rate_fee.v1",
        "axis": "[scorer-free EXACT byte measurement on move 49's shipped container]",
        "score_claim": False,
        "base_member_bytes": base_member,
        "container_shape": {
            "brotli_quality": section.brotli_quality,
            "brotli_lgwin": section.brotli_lgwin,
            "ck2": section.ck2,
        },
        "rows": rows,
        "summary": summary,
        "elapsed_seconds": time.time() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(summary, indent=1))
    return 0


def cmd_base_tolerance(args) -> int:
    """MEASURE the batch-1 vs batch-8 reproduction band the search's base gate uses.

    The n600 base vector is produced by ``up2.measure_pose`` at batch 8; the search
    re-evaluates each pair through ``br1.evaluate_codes`` at batch 1.  Both are the
    shipped forward model on the shipped frames, so any gap is float summation order.
    This stage measures that gap on untouched pairs and reports 10x the observed
    maximum as the gate's tolerance -- a measured band, not a chosen epsilon.
    """
    set_threads(args.threads)
    started = time.time()
    base_pose = np.load(args.base_pose)
    pairs = [int(p) for p in args.pairs.split(",")]
    raw = open_live_raw()
    inst = build_pose_instrument(raw)
    live_codes = np.asarray(inst.state.codes, dtype=np.int32)
    rows = []
    for pair in pairs:
        first = float(br1.evaluate_codes(inst, pair, live_codes[pair][None])[0])
        second = float(br1.evaluate_codes(inst, pair, live_codes[pair][None])[0])
        reference = float(base_pose[pair])
        rows.append({
            "pair": pair,
            "batch1_first": first,
            "batch1_second": second,
            "batch1_repeat_is_exact": first == second,
            "n600_batch8": reference,
            "gap_rel": abs(first / reference - 1.0) if reference > 0 else 0.0,
        })
        print(
            f"pair {pair}: batch1 {first:.12e} (repeat exact {first == second}), "
            f"batch8 {reference:.12e}, gap {rows[-1]['gap_rel']:.3e}",
            flush=True,
        )
    observed = max(row["gap_rel"] for row in rows)
    report = {
        "schema": "ddm_pp1_base_tolerance.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT]",
        "score_claim": False,
        "rows": rows,
        "batch1_repeat_exact_all": all(row["batch1_repeat_is_exact"] for row in rows),
        "observed_max_gap_rel": observed,
        "tolerance_10x": 10.0 * observed,
        "elapsed_seconds": time.time() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({
        "observed_max_gap_rel": observed, "tolerance_10x": 10.0 * observed,
        "batch1_repeat_exact_all": report["batch1_repeat_exact_all"],
    }))
    return 0


def cmd_search_a(args) -> int:
    """Per-pair frame_embed pose descent on the shipped 3-bit lattice."""
    set_threads(args.threads)
    started = time.time()
    pairs = [int(p) for p in args.pairs.split(",")]
    base_pose = np.load(args.base_pose)
    if base_pose.shape != (N_PAIRS,):
        raise Pp1Error(f"base pose vector has shape {base_pose.shape}")
    base_pose_mean = float(base_pose.mean())
    body = load_body(with_raw=True, with_segnet=True)
    raw = body.raw
    base_inst = build_pose_instrument(raw)
    live_codes = np.asarray(base_inst.state.codes, dtype=np.int32)
    dd_threshold = jg5.materiality_dd_threshold(base_pose_mean)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"searchA_rows_{args.shard_index}.jsonl"
    done: set[int] = set()
    if args.resume and rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                done.add(int(json.loads(line)["pair"]))

    handle = rows_path.open("a")
    for pair in pairs:
        if pair in done:
            print(f"pair {pair}: resumed, skipping", flush=True)
            continue
        pair_started = time.time()
        fe1.restore_pair_codes(body, pair)
        base_row = body.section.codes[pair].astype(np.int64).copy()
        base_flips = fe1.flips_pair(fe1.argmax_pair(body, pair), body, pair)
        d_pose_base = float(br1.evaluate_codes(base_inst, pair, live_codes[pair][None])[0])
        base_gap_rel = (
            abs(d_pose_base / float(base_pose[pair]) - 1.0)
            if base_pose[pair] > 0 else 0.0
        )
        # The n600 base vector is measured at PoseNet batch 8; the search re-evaluates
        # this pair at batch 1.  up2 sec.6 measured that batch shape moves the pose
        # vector by ~1.8e-07 relative, so an EXACT-equality gate would refuse the float
        # summation order, not a wrong object.  The tolerance is therefore a MEASURED
        # reproduction band (stage ``base-tolerance``), the gate stays strict at it, and
        # the observed gap is recorded for every treated pair.
        if base_gap_rel > args.base_tolerance:
            raise Pp1Error(
                f"pair {pair} base d_pose {d_pose_base} disagrees with the n600 base "
                f"vector {float(base_pose[pair])} by {base_gap_rel:.3e} relative, above "
                f"the measured reproduction band {args.base_tolerance:.3e}"
            )
        candidates: list[dict[str, Any]] = []
        for dim, code in single_code_moves(base_row):
            new_row = base_row.copy()
            new_row[dim] = code
            row = _resolved_row(
                body, base_inst, raw, pair, new_row,
                live_codes=live_codes, dd_threshold=dd_threshold,
                outer_rounds=args.outer_rounds, max_gn_iterations=args.max_gn_iterations,
                base_flips=base_flips, d_pose_base=d_pose_base,
                base_pose_mean=base_pose_mean, with_seg=True,
            )
            row["move"] = [[int(dim), int(code)]]
            candidates.append(row)
            print(
                f"pair {pair} d{dim} {int(base_row[dim]):+d}->{code:+d}: cells "
                f"{row['d_cells']:+d}, pose {d_pose_base:.4e} -> stale "
                f"{row['d_pose_stale']:.4e} -> resolved {row['d_pose_resolved']:.4e}",
                flush=True,
            )

        # Two-code moves, seeded from the best single-code directions (SCOPE reduction:
        # the full two-code grid is 8*7/2 * 7 * 7 = 1,372 renders per pair; the seeded
        # set is the measured-best K singles crossed with every other dimension).
        two_code: list[dict[str, Any]] = []
        if args.two_code_seeds > 0:
            improving = sorted(candidates, key=lambda r: r["d_pose_resolved"])
            seeds = improving[: args.two_code_seeds]
            for seed in seeds:
                seed_dim = int(seed["move"][0][0])
                seed_row = np.asarray(seed["codes"], dtype=np.int64)
                for dim2 in range(FRAME_DIM):
                    if dim2 == seed_dim:
                        continue
                    for code2 in range(CODE_MIN, CODE_MAX + 1):
                        if code2 == int(seed_row[dim2]):
                            continue
                        new_row = seed_row.copy()
                        new_row[dim2] = code2
                        row = _resolved_row(
                            body, base_inst, raw, pair, new_row,
                            live_codes=live_codes, dd_threshold=dd_threshold,
                            outer_rounds=args.outer_rounds,
                            max_gn_iterations=args.max_gn_iterations,
                            base_flips=base_flips, d_pose_base=d_pose_base,
                            base_pose_mean=base_pose_mean, with_seg=True,
                        )
                        row["move"] = seed["move"] + [[int(dim2), int(code2)]]
                        two_code.append(row)
                        print(
                            f"pair {pair} 2-code {row['move']}: cells {row['d_cells']:+d}, "
                            f"resolved {row['d_pose_resolved']:.4e}",
                            flush=True,
                        )
        all_rows = candidates + two_code
        seg_neutral = [r for r in all_rows if r["d_cells"] <= 0]
        best_neutral = min(seg_neutral, key=lambda r: r["d_pose_resolved"], default=None)
        best_any = min(all_rows, key=lambda r: r["d_pose_resolved"], default=None)
        record = {
            "pair": pair,
            "base_codes": [int(c) for c in base_row],
            "base_flips": base_flips,
            "d_pose_base": d_pose_base,
            "d_pose_base_n600_vector": float(base_pose[pair]),
            "base_gap_rel_vs_n600": base_gap_rel,
            "base_tolerance": args.base_tolerance,
            "n_candidates": len(all_rows),
            "n_seg_neutral": len(seg_neutral),
            "best_seg_neutral": best_neutral,
            "best_any": best_any,
            "candidates": all_rows if args.keep_all else [],
            "elapsed_seconds": time.time() - pair_started,
        }
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        if best_neutral is not None:
            print(
                f"PAIR {pair} BEST seg-neutral: {best_neutral['move']} resolved "
                f"{best_neutral['d_pose_resolved']:.6e} vs base {d_pose_base:.6e} "
                f"({best_neutral['d_pose_resolved'] / d_pose_base:.3f}x), dS_pose "
                f"{best_neutral['dS_pose_resolved']:+.3e}",
                flush=True,
            )
    handle.close()
    manifest = {
        "schema": "ddm_pp1_search_a.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT]",
        "score_claim": False,
        "receipts": body.receipts,
        "pairs": pairs,
        "base_pose_path": str(args.base_pose),
        "base_pose_mean": base_pose_mean,
        "dd_threshold": dd_threshold,
        "rows_path": str(rows_path),
        "two_code_seeds": args.two_code_seeds,
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / f"SEARCH_A_{args.shard_index}.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True)
    )
    print(json.dumps({"rows": str(rows_path), "elapsed_s": manifest["elapsed_seconds"]}))
    return 0


def pose_saliency_on_token_grid(inst, pair: int, frame1_uint8: np.ndarray) -> np.ndarray:
    """|d(d_pose)/d(frame_1)| pooled from camera resolution onto the 384x512 token grid.

    A PROPOSAL only.  Nothing downstream trusts it: every candidate it nominates is
    realized through the shipped render, the frozen SegNet argmax and the carrier
    re-solve, and the measurement decides ([[iv1 sec.4]] -- predictions propose,
    measurements decide).
    """
    import torch
    import torch.nn.functional as functional

    index = np.array([pair], dtype=np.int64)
    coefficients = up2.codes_to_coefficients(
        np.asarray(inst.state.codes, dtype=np.int32)[index], inst.state.coefficient_scales
    )
    with torch.inference_mode():
        frame0 = up2.render_frame0(coefficients, inst.state, index, differentiable=False)
    frame0 = frame0.clone()
    frame1 = torch.from_numpy(
        np.ascontiguousarray(frame1_uint8.transpose(2, 0, 1))[None]
    ).float()
    frame1.requires_grad_(True)
    pose = up2.pose_from_frames(inst.posenet, frame0, frame1)
    target = torch.from_numpy(inst.targets[pair][None]).float()
    loss = ((pose - target) ** 2).mean()
    loss.backward()
    grad = frame1.grad.detach().abs().sum(dim=1, keepdim=True)
    pooled = functional.adaptive_avg_pool2d(grad, (EVAL_H, EVAL_W))
    return pooled[0, 0].numpy()


def cmd_search_b(args) -> int:
    """Actuator B: pose-ranked SINGLE-TOKEN edits on the same pairs.

    A token move changes one cell of the pair's 384x512 plane, so the render change is
    strictly SMALLER and more local than a frame_embed move, which re-renders the whole
    frame.  Cells are nominated by the pose saliency of frame_1 and every nomination is
    realized on the same three legs the frame_embed search uses.
    """
    set_threads(args.threads)
    started = time.time()
    pairs = [int(p) for p in args.pairs.split(",")]
    base_pose = np.load(args.base_pose)
    base_pose_mean = float(base_pose.mean())
    body = load_body(with_raw=True, with_segnet=True)
    raw = body.raw
    base_inst = build_pose_instrument(raw)
    live_codes = np.asarray(base_inst.state.codes, dtype=np.int32)
    dd_threshold = jg5.materiality_dd_threshold(base_pose_mean)
    deltas = [int(d) for d in args.deltas.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"search_b_{args.shard_index}.jsonl"
    handle = rows_path.open("a")
    for pair in pairs:
        fe1.restore_pair_codes(body, pair)
        base_plane = body.tokens[pair].copy()
        base_flips = fe1.flips_pair(fe1.argmax_pair(body, pair), body, pair)
        d_pose_base = float(br1.evaluate_codes(base_inst, pair, live_codes[pair][None])[0])
        saliency = pose_saliency_on_token_grid(
            base_inst, pair, np.asarray(raw[2 * pair + 1])
        )
        flat = np.argsort(-saliency.ravel())[: args.cells]
        cells = [(int(i // EVAL_W), int(i % EVAL_W)) for i in flat]
        print(
            f"pair {pair}: base flips {base_flips}, d_pose {d_pose_base:.4e}, "
            f"{len(cells)} saliency cells, top saliency {saliency.ravel()[flat[0]]:.3e}",
            flush=True,
        )
        for row_index, col_index in cells:
            old = int(base_plane[row_index, col_index])
            for delta in deltas:
                new = old + delta
                if not 0 <= new <= 255:
                    continue
                body.tokens[pair][row_index, col_index] = new
                frame = fe1.render_pair(body, pair)
                moved_flips = fe1.flips_pair(
                    jg1.argmax_from_camera_frames(body.net, frame)[0], body, pair
                )
                body.tokens[pair][row_index, col_index] = old
                overlay = MemoryOverlayRaw(raw, {2 * pair + 1: frame[0]})
                moved_inst = br1.Instrument(
                    base_inst.state, overlay, base_inst.targets, base_inst.posenet,
                    base_inst.blow, base_inst.gram, base_inst.bmat,
                )
                stale = float(
                    br1.evaluate_codes(moved_inst, pair, live_codes[pair][None])[0]
                )
                refined = jg5.refine_pair(
                    moved_inst, pair, live_codes[pair], dd_threshold=dd_threshold,
                    outer_rounds=args.outer_rounds,
                    max_gn_iterations=args.max_gn_iterations,
                )
                resolved = float(refined["final_d_pose"])
                base_leg = math.sqrt(10.0 * base_pose_mean)
                row = {
                    "pair": pair, "cell": [row_index, col_index], "old": old, "new": new,
                    "base_flips": base_flips, "flips": moved_flips,
                    "d_cells": moved_flips - base_flips,
                    "saliency": float(saliency[row_index, col_index]),
                    "d_pose_base": d_pose_base, "d_pose_stale": stale,
                    "d_pose_resolved": resolved,
                    "resolved_over_base": resolved / d_pose_base,
                    "carrier_codes": [int(c) for c in refined["codes"]],
                    "dS_seg": 100.0 * (moved_flips - base_flips) / CELL_COUNT,
                    "dS_pose_resolved": math.sqrt(
                        10.0 * (base_pose_mean + (resolved - d_pose_base) / N_PAIRS)
                    ) - base_leg,
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(
                    f"pair {pair} cell ({row_index},{col_index}) {old}->{new}: cells "
                    f"{row['d_cells']:+d}, resolved {resolved:.4e} "
                    f"({row['resolved_over_base']:.4f}x)",
                    flush=True,
                )
    handle.close()
    (out_dir / f"SEARCH_B_{args.shard_index}.json").write_text(json.dumps({
        "schema": "ddm_pp1_search_b.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT]",
        "score_claim": False,
        "receipts": body.receipts,
        "pairs": pairs, "cells": args.cells, "deltas": deltas,
        "proposal": "pose saliency |d(d_pose)/d(frame_1)| area-pooled to the token grid",
        "base_pose_mean": base_pose_mean,
        "rows_path": str(rows_path),
        "elapsed_seconds": time.time() - started,
    }, indent=1, sort_keys=True))
    print(json.dumps({"rows": str(rows_path)}))
    return 0


def cmd_compose(args) -> int:
    """A + B on ONE pair, realized together -- never the sum of two separate rows.

    Composition of a frame_embed move and a token edit is a single render of a single
    frame, so it is measured as one object.  Reporting A+B as A's delta plus B's delta
    would be exactly the additivity this campaign refuses
    ([[composition_of_disjoint_token_edits_is_subadditive...]] and its sisters).
    """
    set_threads(args.threads)
    started = time.time()
    pair = int(args.pair)
    base_pose = np.load(args.base_pose)
    base_pose_mean = float(base_pose.mean())
    body = load_body(with_raw=True, with_segnet=True)
    raw = body.raw
    base_inst = build_pose_instrument(raw)
    live_codes = np.asarray(base_inst.state.codes, dtype=np.int32)
    dd_threshold = jg5.materiality_dd_threshold(base_pose_mean)

    code_moves = [
        tuple(int(x) for x in spec.split(","))
        for spec in args.code_moves.split(";") if spec.strip()
    ]
    token_edits = [
        tuple(int(x) for x in spec.split(","))
        for spec in args.token_edits.split(";") if spec.strip()
    ]

    fe1.restore_pair_codes(body, pair)
    base_plane = body.tokens[pair].copy()
    base_row = body.section.codes[pair].astype(np.int64).copy()
    base_flips = fe1.flips_pair(fe1.argmax_pair(body, pair), body, pair)
    d_pose_base = float(br1.evaluate_codes(base_inst, pair, live_codes[pair][None])[0])

    rows = []
    variants = [
        ("A", code_moves, []),
        ("B", [], token_edits),
        ("A+B", code_moves, token_edits),
    ]
    for label, codes_spec, tokens_spec in variants:
        if not codes_spec and not tokens_spec:
            continue
        new_row = base_row.copy()
        for dim, code in codes_spec:
            new_row[dim] = code
        for row_index, col_index, value in tokens_spec:
            body.tokens[pair][row_index, col_index] = value
        row = _resolved_row(
            body, base_inst, raw, pair, new_row,
            live_codes=live_codes, dd_threshold=dd_threshold,
            outer_rounds=args.outer_rounds, max_gn_iterations=args.max_gn_iterations,
            base_flips=base_flips, d_pose_base=d_pose_base,
            base_pose_mean=base_pose_mean, with_seg=True,
        )
        body.tokens[pair][...] = base_plane
        row.update({
            "label": label, "pair": pair,
            "code_moves": [list(m) for m in codes_spec],
            "token_edits": [list(e) for e in tokens_spec],
            "base_flips": base_flips, "d_pose_base": d_pose_base,
            "resolved_over_base": row["d_pose_resolved"] / d_pose_base,
            "dS_seg_plus_pose": row["dS_seg"] + row["dS_pose_resolved"],
        })
        rows.append(row)
        print(
            f"{label}: cells {row['d_cells']:+d}, pose {d_pose_base:.4e} -> "
            f"{row['d_pose_resolved']:.4e} ({row['resolved_over_base']:.4f}x), dS "
            f"{row['dS_seg_plus_pose']:+.3e}",
            flush=True,
        )
    by_label = {row["label"]: row for row in rows}
    additive = None
    if {"A", "B", "A+B"} <= set(by_label):
        additive = {
            "sum_of_parts_dS": by_label["A"]["dS_seg_plus_pose"]
            + by_label["B"]["dS_seg_plus_pose"],
            "measured_dS": by_label["A+B"]["dS_seg_plus_pose"],
        }
        additive["measured_over_sum"] = (
            additive["measured_dS"] / additive["sum_of_parts_dS"]
            if additive["sum_of_parts_dS"] != 0 else math.nan
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "ddm_pp1_compose.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT]",
        "score_claim": False,
        "receipts": body.receipts,
        "pair": pair,
        "rows": rows,
        "additivity": additive,
        "elapsed_seconds": time.time() - started,
    }, indent=1, sort_keys=True))
    print(json.dumps({"out": str(out), "additivity": additive}))
    return 0


def cmd_refine_moves(args) -> int:
    """Full realized chain on an EXPLICIT move list: render, seg, stale, resolved.

    The seg screen ranks the single-code lattice on the binding constraint; this stage
    pays the carrier re-solve only for the survivors.  A move is ``pair,dim,code``;
    several moves on ONE pair compose into a single multi-code row because the overlay
    holds one frame per pair (fe1's repeated-pair trap).
    """
    set_threads(args.threads)
    started = time.time()
    base_pose = np.load(args.base_pose)
    base_pose_mean = float(base_pose.mean())
    body = load_body(with_raw=True, with_segnet=True)
    raw = body.raw
    base_inst = build_pose_instrument(raw)
    live_codes = np.asarray(base_inst.state.codes, dtype=np.int32)
    dd_threshold = jg5.materiality_dd_threshold(base_pose_mean)

    # ``--compose-per-pair`` decides whether several specs on ONE pair become one
    # multi-code row or several independent single-code rows.  Either way the unit of
    # work is a (pair, moves) tuple, because the overlay holds one frame per pair and a
    # second move on the same frame would silently be priced against the first.
    plan: list[tuple[int, list[tuple[int, int]]]] = []
    for spec in args.moves.split(";"):
        if not spec.strip():
            continue
        pair, dim, code = (int(x) for x in spec.split(","))
        if args.compose_per_pair and plan and plan[-1][0] == pair:
            plan[-1][1].append((dim, code))
        elif args.compose_per_pair and any(entry[0] == pair for entry in plan):
            next(entry for entry in plan if entry[0] == pair)[1].append((dim, code))
        else:
            plan.append((pair, [(dim, code)]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"refine_moves_{args.shard_index}.jsonl"
    done: set[str] = set()
    if args.resume and rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["key"])
    handle = rows_path.open("a")
    for pair, moves in plan:
        row_key = f"{pair}:" + ",".join(f"{d}={c}" for d, c in moves)
        if row_key in done:
            print(f"{row_key}: resumed, skipping", flush=True)
            continue
        fe1.restore_pair_codes(body, pair)
        base_row = body.section.codes[pair].astype(np.int64).copy()
        base_flips = fe1.flips_pair(fe1.argmax_pair(body, pair), body, pair)
        d_pose_base = float(br1.evaluate_codes(base_inst, pair, live_codes[pair][None])[0])
        base_gap_rel = (
            abs(d_pose_base / float(base_pose[pair]) - 1.0) if base_pose[pair] > 0 else 0.0
        )
        if base_gap_rel > args.base_tolerance:
            raise Pp1Error(
                f"pair {pair} base d_pose {d_pose_base} disagrees with the n600 vector "
                f"{float(base_pose[pair])} by {base_gap_rel:.3e}, above the measured "
                f"band {args.base_tolerance:.3e}"
            )
        new_row = base_row.copy()
        for dim, code in moves:
            new_row[dim] = code
        row = _resolved_row(
            body, base_inst, raw, pair, new_row,
            live_codes=live_codes, dd_threshold=dd_threshold,
            outer_rounds=args.outer_rounds, max_gn_iterations=args.max_gn_iterations,
            base_flips=base_flips, d_pose_base=d_pose_base,
            base_pose_mean=base_pose_mean, with_seg=True,
        )
        row.update({
            "key": row_key, "pair": pair, "move": [[int(d), int(c)] for d, c in moves],
            "base_codes": [int(c) for c in base_row], "base_flips": base_flips,
            "d_pose_base": d_pose_base,
            "d_pose_base_n600_vector": float(base_pose[pair]),
            "base_gap_rel_vs_n600": base_gap_rel,
            "resolved_over_base": row["d_pose_resolved"] / d_pose_base,
            "dS_seg_plus_pose": row["dS_seg"] + row["dS_pose_resolved"],
        })
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        print(
            f"{row_key}: cells {row['d_cells']:+d}, pose {d_pose_base:.4e} -> resolved "
            f"{row['d_pose_resolved']:.4e} ({row['resolved_over_base']:.4f}x), dS seg "
            f"{row['dS_seg']:+.3e} + pose {row['dS_pose_resolved']:+.3e} = "
            f"{row['dS_seg_plus_pose']:+.3e}",
            flush=True,
        )
    handle.close()
    (out_dir / f"REFINE_MOVES_{args.shard_index}.json").write_text(json.dumps({
        "schema": "ddm_pp1_refine_moves.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT]",
        "score_claim": False,
        "receipts": body.receipts,
        "moves": args.moves,
        "compose_per_pair": bool(args.compose_per_pair),
        "base_pose_mean": base_pose_mean,
        "dd_threshold": dd_threshold,
        "rows_path": str(rows_path),
        "elapsed_seconds": time.time() - started,
    }, indent=1, sort_keys=True))
    print(json.dumps({"rows": str(rows_path)}))
    return 0


def cmd_seg_screen(args) -> int:
    """Screen every single-code move on the CHEAP binding constraint first.

    The admission requires seg NEUTRALITY, and the seg leg needs only a render and a
    SegNet argmax -- no carrier re-solve.  Refining a candidate the seg leg will reject
    spends ~3 minutes to learn nothing, so the whole single-code lattice is screened
    here at ~4 s per candidate and only the survivors are handed to the pose refine.
    This is an ORDERING change, not a mechanism change: every number the admission uses
    is still realized, and the screen's own d_cells is the same measurement the full
    stage makes.
    """
    set_threads(args.threads)
    started = time.time()
    pairs = [int(p) for p in args.pairs.split(",")]
    body = load_body(with_raw=False, with_segnet=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"seg_screen_{args.shard_index}.jsonl"
    handle = rows_path.open("a")
    totals: list[dict[str, Any]] = []
    for pair in pairs:
        fe1.restore_pair_codes(body, pair)
        base_row = body.section.codes[pair].astype(np.int64).copy()
        base_flips = fe1.flips_pair(fe1.argmax_pair(body, pair), body, pair)
        rows = []
        for dim, code in single_code_moves(base_row):
            new_row = base_row.copy()
            new_row[dim] = code
            fe1.set_pair_codes(body, pair, new_row)
            moved_flips = fe1.flips_pair(fe1.argmax_pair(body, pair), body, pair)
            fe1.restore_pair_codes(body, pair)
            rows.append({
                "pair": pair, "dim": int(dim), "old": int(base_row[dim]),
                "new": int(code), "base_flips": base_flips, "flips": moved_flips,
                "d_cells": moved_flips - base_flips,
            })
            handle.write(json.dumps(rows[-1]) + "\n")
            handle.flush()
        neutral = [r for r in rows if r["d_cells"] <= 0]
        cheap = sorted(rows, key=lambda r: r["d_cells"])[: args.keep_cheapest]
        totals.append({
            "pair": pair, "base_flips": base_flips, "n_moves": len(rows),
            "n_seg_neutral": len(neutral),
            "min_d_cells": min(r["d_cells"] for r in rows),
            "neutral": neutral,
            "cheapest": cheap,
        })
        print(
            f"pair {pair}: base flips {base_flips}, {len(rows)} single-code moves, "
            f"{len(neutral)} seg-neutral, cheapest {min(r['d_cells'] for r in rows):+d} cells",
            flush=True,
        )
    handle.close()
    report = {
        "schema": "ddm_pp1_seg_screen.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch SegNet argmax, DALI GT]",
        "score_claim": False,
        "receipts": body.receipts,
        "rows_path": str(rows_path),
        "per_pair": totals,
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / f"SEG_SCREEN_{args.shard_index}.json").write_text(
        json.dumps(report, indent=1, sort_keys=True)
    )
    print(json.dumps({"rows": str(rows_path), "elapsed_s": report["elapsed_seconds"]}))
    return 0


def cmd_floor_probe(args) -> int:
    """Is the RESOLVED per-pair pose a FLOOR the frame_1 render cannot move?

    Actuator A's single-code sweep is a small neighbourhood.  This probe asks the
    structural question directly: apply perturbations of INCREASING size to the pair's
    eight FiLM codes -- up to every dimension at once, which re-renders the whole frame
    -- and measure where the carrier re-solve lands each time.  If the resolved value
    sits in a narrow band around the base however violently frame_1 moves, then the
    residual is the part of the pose error the carrier's twelve coefficients cannot
    reach, and no frame_1 actuator on this lattice can lower it.
    """
    set_threads(args.threads)
    started = time.time()
    pairs = [int(p) for p in args.pairs.split(",")]
    base_pose = np.load(args.base_pose)
    base_pose_mean = float(base_pose.mean())
    body = load_body(with_raw=True, with_segnet=True)
    raw = body.raw
    base_inst = build_pose_instrument(raw)
    live_codes = np.asarray(base_inst.state.codes, dtype=np.int32)
    dd_threshold = jg5.materiality_dd_threshold(base_pose_mean)
    rng = np.random.default_rng(args.seed)
    sizes = [int(s) for s in args.sizes.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"floor_rows_{args.shard_index}.jsonl"
    handle = rows_path.open("a")
    for pair in pairs:
        fe1.restore_pair_codes(body, pair)
        base_row = body.section.codes[pair].astype(np.int64).copy()
        base_flips = fe1.flips_pair(fe1.argmax_pair(body, pair), body, pair)
        d_pose_base = float(br1.evaluate_codes(base_inst, pair, live_codes[pair][None])[0])
        for size in sizes:
            for repeat in range(args.repeats):
                new_row = base_row.copy()
                dims = rng.permutation(FRAME_DIM)[:size]
                for dim in dims:
                    choices = [
                        c for c in range(CODE_MIN, CODE_MAX + 1) if c != int(base_row[dim])
                    ]
                    new_row[dim] = int(choices[int(rng.integers(0, len(choices)))])
                row = _resolved_row(
                    body, base_inst, raw, pair, new_row,
                    live_codes=live_codes, dd_threshold=dd_threshold,
                    outer_rounds=args.outer_rounds,
                    max_gn_iterations=args.max_gn_iterations,
                    base_flips=base_flips, d_pose_base=d_pose_base,
                    base_pose_mean=base_pose_mean, with_seg=True,
                )
                row.update({
                    "pair": pair, "size": size, "repeat": repeat,
                    "base_codes": [int(c) for c in base_row],
                    "d_pose_base": d_pose_base, "base_flips": base_flips,
                    "resolved_over_base": row["d_pose_resolved"] / d_pose_base,
                    "stale_over_base": row["d_pose_stale"] / d_pose_base,
                })
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(
                    f"pair {pair} size {size} r{repeat}: cells {row['d_cells']:+d}, "
                    f"stale {row['stale_over_base']:.1f}x -> resolved "
                    f"{row['resolved_over_base']:.4f}x base",
                    flush=True,
                )
    handle.close()
    (out_dir / f"FLOOR_PROBE_{args.shard_index}.json").write_text(json.dumps({
        "schema": "ddm_pp1_floor_probe.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT]",
        "score_claim": False,
        "receipts": body.receipts,
        "pairs": pairs, "sizes": sizes, "repeats": args.repeats, "seed": args.seed,
        "rows_path": str(rows_path),
        "elapsed_seconds": time.time() - started,
    }, indent=1, sort_keys=True))
    print(json.dumps({"rows": str(rows_path)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    base = sub.add_parser("base", help="per-pair d_pose on move 49's shipped state")
    base.add_argument("--out-dir", default=str(WORK / "base"))
    base.add_argument("--batch-size", type=int, default=8)
    base.add_argument("--threads", type=int, default=4)
    base.add_argument("--top", type=int, default=24)
    base.set_defaults(func=cmd_base)

    controls = sub.add_parser("controls", help="identity controls before any claim")
    controls.add_argument("--pairs", default="0,88,199,316,599")
    controls.add_argument("--resolve-control-pairs", default="")
    controls.add_argument("--base-mean-d-pose", type=float, required=True)
    controls.add_argument("--outer-rounds", type=int, default=40)
    controls.add_argument("--max-gn-iterations", type=int, default=400)
    controls.add_argument("--threads", type=int, default=4)
    controls.add_argument("--out", default=str(WORK / "base/CONTROLS.json"))
    controls.set_defaults(func=cmd_controls)

    cp = sub.add_parser("compose", help="A, B and A+B on one pair, each realized")
    cp.add_argument("--pair", type=int, required=True)
    cp.add_argument("--code-moves", default="", help="dim,code;dim,code")
    cp.add_argument("--token-edits", default="", help="row,col,value;row,col,value")
    cp.add_argument("--base-pose", type=Path, required=True)
    cp.add_argument("--outer-rounds", type=int, default=40)
    cp.add_argument("--max-gn-iterations", type=int, default=400)
    cp.add_argument("--threads", type=int, default=2)
    cp.add_argument("--out", default=str(WORK / "compose/COMPOSE.json"))
    cp.set_defaults(func=cmd_compose)

    sb = sub.add_parser("search-b", help="pose-ranked single-token edits")
    sb.add_argument("--pairs", required=True)
    sb.add_argument("--base-pose", type=Path, required=True)
    sb.add_argument("--cells", type=int, default=6)
    sb.add_argument("--deltas", default="-1,1")
    sb.add_argument("--out-dir", default=str(WORK / "searchB"))
    sb.add_argument("--shard-index", type=int, default=0)
    sb.add_argument("--outer-rounds", type=int, default=40)
    sb.add_argument("--max-gn-iterations", type=int, default=400)
    sb.add_argument("--threads", type=int, default=2)
    sb.set_defaults(func=cmd_search_b)

    rm = sub.add_parser("refine-moves", help="full realized chain on an explicit move list")
    rm.add_argument("--moves", required=True, help="pair,dim,code;pair,dim,code;...")
    rm.add_argument("--base-pose", type=Path, required=True)
    rm.add_argument("--base-tolerance", type=float, required=True)
    rm.add_argument("--compose-per-pair", action="store_true")
    rm.add_argument("--out-dir", default=str(WORK / "refine"))
    rm.add_argument("--shard-index", type=int, default=0)
    rm.add_argument("--outer-rounds", type=int, default=40)
    rm.add_argument("--max-gn-iterations", type=int, default=400)
    rm.add_argument("--threads", type=int, default=2)
    rm.add_argument("--resume", action="store_true")
    rm.set_defaults(func=cmd_refine_moves)

    ss = sub.add_parser("seg-screen", help="seg-only screen of the single-code lattice")
    ss.add_argument("--pairs", required=True)
    ss.add_argument("--out-dir", default=str(WORK / "segscreen"))
    ss.add_argument("--shard-index", type=int, default=0)
    ss.add_argument("--keep-cheapest", type=int, default=4)
    ss.add_argument("--threads", type=int, default=2)
    ss.set_defaults(func=cmd_seg_screen)

    fp = sub.add_parser("floor-probe", help="is the resolved per-pair pose a floor?")
    fp.add_argument("--pairs", required=True)
    fp.add_argument("--base-pose", type=Path, required=True)
    fp.add_argument("--sizes", default="2,4,8")
    fp.add_argument("--repeats", type=int, default=2)
    fp.add_argument("--seed", type=int, default=20260912)
    fp.add_argument("--out-dir", default=str(WORK / "floor"))
    fp.add_argument("--shard-index", type=int, default=0)
    fp.add_argument("--outer-rounds", type=int, default=40)
    fp.add_argument("--max-gn-iterations", type=int, default=400)
    fp.add_argument("--threads", type=int, default=2)
    fp.set_defaults(func=cmd_floor_probe)

    rf = sub.add_parser("rate-fee", help="member cost of N changed frame_embed codes")
    rf.add_argument("--counts", default="1,2,4,8,16")
    rf.add_argument("--repeats", type=int, default=8)
    rf.add_argument("--seed", type=int, default=20260912)
    rf.add_argument("--search-container", action="store_true")
    rf.add_argument("--out", default=str(WORK / "base/RATE_FEE.json"))
    rf.set_defaults(func=cmd_rate_fee)

    bt = sub.add_parser("base-tolerance", help="measure the batch-1/batch-8 pose band")
    bt.add_argument("--pairs", required=True)
    bt.add_argument("--base-pose", type=Path, required=True)
    bt.add_argument("--threads", type=int, default=4)
    bt.add_argument("--out", default=str(WORK / "base/BASE_TOLERANCE.json"))
    bt.set_defaults(func=cmd_base_tolerance)

    sa = sub.add_parser("search-a", help="per-pair frame_embed pose descent")
    sa.add_argument("--pairs", required=True)
    sa.add_argument("--base-pose", type=Path, required=True)
    sa.add_argument(
        "--base-tolerance", type=float, required=True,
        help="MEASURED batch-1/batch-8 reproduction band from the base-tolerance stage",
    )
    sa.add_argument("--out-dir", default=str(WORK / "searchA"))
    sa.add_argument("--shard-index", type=int, default=0)
    sa.add_argument("--two-code-seeds", type=int, default=0)
    sa.add_argument("--outer-rounds", type=int, default=40)
    sa.add_argument("--max-gn-iterations", type=int, default=400)
    sa.add_argument("--threads", type=int, default=2)
    sa.add_argument("--keep-all", action="store_true")
    sa.add_argument("--resume", action="store_true")
    sa.set_defaults(func=cmd_search_a)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
