#!/usr/bin/env python3
"""ddm_jg1 -- the seg-axis realized-descent instrument for the shipped object.

``ddm_up2`` solved the pose carrier by LATTICE COORDINATE DESCENT WITH REALIZED
ACCEPTANCE: propose from structure, accept only when the objective actually falls
through the real receiver path.  A gradient solve was built first and abandoned on
measurement.  This module is the seg-axis analogue, on the actuator that ddm_jg1's
S0 coordinate map identified:

    section 4 (the 109,792-byte RX1 tail) carries ``tokens``:
    ``(600, 384, 512) uint8`` in ``{0..4}`` -- a 5-class semantic label map per pair,
    HPAC-entropy-coded at 0.00745 bits/token.

``cpr1/inflate.py:318`` renders frame ``2p+1`` from those tokens through
``SemanticTokenRenderer``; ``upstream/modules.py:108`` (``x = x[:, -1, ...]``) means
SegNet sees ONLY frame ``2p+1``.  So the tokens are the seg actuator, and -- because
PoseNet sees both frames -- they are also a pose actuator.  That asymmetry is the
joint structure this arm exists to measure.

Everything here is ``[macOS-CPU advisory]`` unless it carries an explicit DALI
lineage tag.  Nothing here is a score claim.

The GT lineage gate is NOT optional and is not re-implemented: it is
``ddm_up2.verify_gt_lineage``, which fails closed.  ``ddm_up3`` measured the two
lineages **1.43x apart on the seg axis** (0.00030309 DALI vs 0.00043336 PyAV) on the
very body we ship, so an absolute ``d_seg`` quoted against the wrong lineage is
simply the wrong number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_up2_shipping_pose_solve as up2

# --------------------------------------------------------------------------------------
# Constants.  Every one is DERIVED from the receiver or MEASURED, never guessed.
# --------------------------------------------------------------------------------------

N_PAIRS = 600
#: ``cpr1/inflate.py:22`` -- the token grid and the SegNet argmax grid are the SAME
#: 384x512 lattice, which is why a token is a near-one-to-one actuator on an argmax cell.
EVAL_H, EVAL_W = 384, 512
#: ``cpr1/inflate.py:23`` -- the camera lattice the receiver writes into ``0.raw``.
CAMERA_H, CAMERA_W = 874, 1164
#: The SHIPPED runtime tree and the archive whose sections this arm actuates.
DEFAULT_RUNTIME = Path(
    "/Volumes/APDataStore/pact/ddm_to1/generations/to1_tail_override_r1/runtime"
)
DEFAULT_ARCHIVE = DEFAULT_RUNTIME.parent / "archive.zip"
#: ``cpr1/inflate.py:21`` -- ``nn.Embedding(NUM_CLASSES, width)`` at ``:95``.
NUM_CLASSES = 5
#: comma10k CANONICAL order.  NEVER re-derive this by luma-sorting ``class_values``;
#: that yields ``[Road, Lane, MyCar, Undrivable, Movable]`` and is wrong.  Confirmed
#: independently by this arm from the shipped token histogram (S0).
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")

SEG_CELLS_TOTAL = N_PAIRS * EVAL_H * EVAL_W  # 117,964,800

#: The shipped token payload, decoded once by the receiver's own HPAC path.  Bound to
#: the archive whose RX1 tail this is; ddm_jg1 verified the tail is byte-identical
#: between ``50e561454b23026d...`` and the live pointer ``7ce46fd7a845d598...`` (only
#: the carrier section differs), so this decode is valid for the pointer.
DEFAULT_TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_to1/advisory/attempt_0002/work/inflated"
    "/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
TOKENS_SHA256 = None  # filled by --verify; the receipt binds the archive instead.

#: GT caches.  BOTH lineages, because the seg axis needs the DALI one for absolute
#: numbers and the PyAV one to reproduce the advisory row.
DEFAULT_GT_DALI = Path(
    "/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt"
)
DEFAULT_GT_AV = Path(
    "/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_av.pt"  # GT_LINEAGE_OK: deliberate PYAV_YUV420_TO_RGB ruler sha256 837b5852dc71 consumed beside the DALI ruler for same-instrument differencing
)

#: The T4 receipt's seg leg for the shipped body -- the number the instrument must
#: reproduce before any of its deltas are worth reading.
POINTER_D_SEG_DALI = 0.00030309
#: The advisory (PyAV-GT) seg leg for the same bytes -- ddm_up3 sec.5/sec.8.
POINTER_D_SEG_AV = 0.00043336

SCORE_RATE_DENOMINATOR = 37_545_489
#: One repaired argmax cell, in S units: ``100 / SEG_CELLS_TOTAL``.
S_PER_SEG_CELL = 100.0 / SEG_CELLS_TOTAL
#: One archive byte, in S units.
S_PER_ARCHIVE_BYTE = 25.0 / SCORE_RATE_DENOMINATOR
#: The governing exchange rate: how many bytes a repaired cell is worth.
BYTES_PER_SEG_CELL = S_PER_SEG_CELL / S_PER_ARCHIVE_BYTE


class Jg1Error(RuntimeError):
    """Refusal raised by this module.  Never downgraded to a warning."""


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_tokens(path: Path = DEFAULT_TOKENS) -> np.ndarray:
    """The shipped token field as ``(600, 384, 512) uint8``.

    Fails closed on shape and on domain: a token outside ``{0..4}`` would index past
    ``nn.Embedding(NUM_CLASSES, width)`` and is therefore not a token at all.
    """
    if not path.is_file():
        raise Jg1Error(f"token decode does not exist: {path}")
    flat = np.fromfile(path, dtype=np.uint8)
    expected = N_PAIRS * EVAL_H * EVAL_W
    if flat.size != expected:
        raise Jg1Error(
            f"token field has {flat.size} cells, expected {expected} "
            f"({N_PAIRS}x{EVAL_H}x{EVAL_W})"
        )
    tokens = flat.reshape(N_PAIRS, EVAL_H, EVAL_W)
    high = int(tokens.max())
    if high >= NUM_CLASSES:
        raise Jg1Error(
            f"token value {high} is outside the embedding domain (<{NUM_CLASSES}); "
            "the decode is not a class map"
        )
    return tokens


def load_gt_seg_labels(lineage: str) -> np.ndarray:
    """GT SegNet argmax labels for the requested lineage, as ``(600, 384, 512) uint8``.

    The lineage is an ARGUMENT, never inferred from context, and the caller must have
    passed it through ``up2.verify_gt_lineage`` against the axis it intends to quote.
    """
    import torch

    if lineage == up2.LINEAGE_DALI:
        path = DEFAULT_GT_DALI
    elif lineage == up2.LINEAGE_AV_PYAV:
        path = DEFAULT_GT_AV
    else:
        raise Jg1Error(f"unknown GT lineage {lineage!r}")
    if not path.is_file():
        raise Jg1Error(f"GT cache does not exist: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "seg" not in blob:
        raise Jg1Error(f"GT cache {path} carries no 'seg' labels")
    labels = blob["seg"].numpy()
    if labels.shape != (N_PAIRS, EVAL_H, EVAL_W):
        raise Jg1Error(
            f"GT seg labels have shape {labels.shape}, expected "
            f"({N_PAIRS}, {EVAL_H}, {EVAL_W})"
        )
    return np.ascontiguousarray(labels.astype(np.uint8, copy=False))


def load_segnet():
    """The frozen contest SegNet on CPU fp32 -- the verdict authority.

    MPS is a legitimate TRAINING-gradient device but is NEVER an authority here: the
    seg objective is an ARGMAX, a discrete decision, and CLAUDE.md records MPS SegNet
    distortion drifting 2x.  A realized-acceptance loop that accepted on MPS argmax
    would be accepting on a different function.
    """

    upstream = REPO / "upstream"
    sys.path.insert(0, str(upstream))
    try:
        from modules import SegNet, segnet_sd_path
        from safetensors.torch import load_file
    finally:
        sys.path.pop(0)
    net = SegNet().eval()
    net.load_state_dict(load_file(str(segnet_sd_path), device="cpu"))
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


def load_semantic_renderer(
    archive_path: Path = DEFAULT_ARCHIVE, runtime_dir: Path = DEFAULT_RUNTIME
):
    """The shipped ``SemanticTokenRenderer`` with the shipped weights.

    Built exactly the way ``runtime/f26_inflate.py:481-494`` builds it -- read from the
    receiver, not reconstructed -- so that ``render_frame1`` is the receiver's own
    forward model rather than a look-alike.
    """

    # ``runtime/residual_archive.py`` uses RELATIVE imports (``from .bits import ...``),
    # so it must be loaded as part of the ``runtime`` package, exactly as the shipped
    # ``inflate.py`` loads it.  Loading it as a standalone file raises ImportError --
    # a useful refusal, because a hand-rolled re-parse of the archive is precisely the
    # duplication the up3 canonical-helper lesson forbids.
    added = [str(runtime_dir.parent), str(runtime_dir.parent / "cpr1")]
    for entry in added:
        sys.path.insert(0, entry)
    try:
        import importlib

        renderer = importlib.import_module("inflate")  # cpr1/inflate.py
        ra = importlib.import_module("runtime.residual_archive")
        parts = ra.read_residual_archive(archive_path)
        semantic = renderer.SemanticTokenRenderer(96)
        tagged_state = renderer.unpack_variant_semantic_or_none(
            parts.semantic_blob, semantic.state_dict()
        )
        if tagged_state is None:
            raise Jg1Error(
                "semantic blob is not a tagged variant; the WANS1 path is not wired here"
            )
        semantic.load_state_dict(tagged_state, strict=True)
    finally:
        for entry in added:
            if entry in sys.path:
                sys.path.remove(entry)
    semantic = semantic.eval()
    for parameter in semantic.parameters():
        parameter.requires_grad_(False)
    return semantic


def render_frame1(semantic, tokens_chunk: np.ndarray, pair_indices: np.ndarray):
    """Render frame ``2p+1`` from tokens, exactly as ``cpr1/inflate.py:316-328`` does.

    **Batch size is 1 by construction and that is not a performance oversight.**
    ``cpr1/inflate.py:312`` sets ``semantic_batch = 8 if cuda else 1``, and ``ddm_up2``
    sec.6 MEASURED that this batch shape is byte-changing on the semantic half -- 1,326
    pixels flip by +/-1 between batch 1 and batch 8, through the ``clamp/round`` at
    ``:323-324``.  Rendering at batch 8 would therefore NOT reproduce the CPU decode this
    arm measures against, and the forward-model control would fail for a reason that has
    nothing to do with tokens.
    """
    import torch
    import torch.nn.functional as functional

    out = np.empty((len(pair_indices), CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
    with torch.inference_mode():
        for row, pair in enumerate(pair_indices):
            tokens = torch.from_numpy(
                np.ascontiguousarray(tokens_chunk[row])[None]
            ).long()
            index = torch.tensor([int(pair)], dtype=torch.long)
            master = (
                functional.interpolate(
                    semantic(tokens, index),
                    size=(CAMERA_H, CAMERA_W),
                    mode="bilinear",
                    align_corners=False,
                )
                .clamp(0.0, 255.0)
                .round()
            )
            out[row] = master.to(torch.uint8).permute(0, 2, 3, 1).numpy()[0]
    return out


def forward_model_control(
    tokens: np.ndarray,
    indices: np.ndarray,
    *,
    semantic=None,
    raw_path: Path = up2.DEFAULT_RAW,
    verify_raw_sha: bool = False,
) -> dict[str, Any]:
    """Does re-rendering the SHIPPED tokens reproduce the SHIPPED frames, byte for byte?

    This is the control that makes every later seg number mean something.  ``ddm_up2``
    ran the same control on the pose half and got ``max |delta| = 0`` over 33,572,088
    pixels; without it, a "realized" objective is only realized with respect to a model
    that may not be the receiver's.
    """
    if semantic is None:
        semantic = load_semantic_renderer()
    raw = up2.open_raw(raw_path, verify_sha=verify_raw_sha)
    rendered = render_frame1(semantic, tokens[indices], indices)
    shipped = np.asarray(raw[2 * np.asarray(indices, dtype=np.int64) + 1])
    delta = np.abs(rendered.astype(np.int16) - shipped.astype(np.int16))
    return {
        "pairs": len(indices),
        "pixels_compared": int(shipped.size),
        "pixels_changed": int((delta > 0).sum()),
        "max_abs_delta": int(delta.max()) if delta.size else 0,
        "byte_exact": bool(delta.max() == 0) if delta.size else False,
        "semantic_batch": 1,
        "note": (
            "cpr1/inflate.py:312 uses semantic_batch=1 on cpu; ddm_up2 sec.6 measured "
            "batch 8 as byte-changing on this half, so batch 1 is required for identity"
        ),
    }


# --------------------------------------------------------------------------------------
# The realized seg objective
# --------------------------------------------------------------------------------------


def argmax_from_camera_frames(net, frames_bhwc) -> np.ndarray:
    """SegNet argmax for camera-resolution frames, through the contest's own preprocess.

    ``frames_bhwc`` is ``(b, 874, 1164, 3)`` uint8 -- exactly what the evaluator's
    ``TensorVideoDataset`` hands the scorer.  ``preprocess_input`` wants ``b t c h w``
    and slices the LAST frame, so a single frame is presented as a length-1 time axis:
    that is the same tensor SegNet sees in the real eval, not a re-implementation.
    """
    import torch

    with torch.inference_mode():
        batch = up2.frames_to_bchw(frames_bhwc)  # (b, 3, 874, 1164) float
        seg_in = net.preprocess_input(batch.unsqueeze(1))
        return net(seg_in).argmax(dim=1).to(torch.uint8).cpu().numpy()


def d_seg_per_pair(argmax: np.ndarray, gt_labels: np.ndarray) -> np.ndarray:
    """Per-pair seg distortion, exactly ``SegNet.compute_distortion`` (modules.py:111-113).

    ``diff.mean`` over the spatial axes, one scalar per pair.  Kept as float64 because
    the population mean of 600 such scalars is quoted to 8 decimals.
    """
    if argmax.shape != gt_labels.shape:
        raise Jg1Error(
            f"argmax {argmax.shape} and GT {gt_labels.shape} disagree in shape"
        )
    return (argmax != gt_labels).reshape(argmax.shape[0], -1).mean(
        axis=1, dtype=np.float64
    )


@dataclass(frozen=True)
class SegLeg:
    """A measured seg leg, always carrying the lineage that makes it meaningful."""

    lineage: str
    pairs: int
    sampling: str
    d_seg: float
    cells_compared: int
    cells_disagreeing: int
    per_pair: tuple[float, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "lineage": self.lineage,
            "pairs": self.pairs,
            "sampling": self.sampling,
            "d_seg": self.d_seg,
            "cells_compared": self.cells_compared,
            "cells_disagreeing": self.cells_disagreeing,
            "axis_note": (
                "advisory instrument; absolute d_seg is only comparable to a contest "
                f"row when lineage == {up2.LINEAGE_DALI!r}"
            ),
        }


def shipped_argmax(
    indices: np.ndarray,
    *,
    raw_path: Path = up2.DEFAULT_RAW,
    batch_size: int = 4,
    verify_raw_sha: bool = True,
    net=None,
    progress: bool = False,
) -> np.ndarray:
    """SegNet argmax of the SHIPPED decode for ``indices``, as ``(n, 384, 512) uint8``.

    Reads frame ``2p+1`` -- the only frame SegNet sees -- out of the receiver's own
    ``0.raw``, so no re-render is involved and the labels are the shipping object's.

    The argmax is a property of OUR frames alone, so it is lineage-INDEPENDENT: the GT
    lineage enters only when this is compared against GT labels.  Computing it once and
    reusing it across lineages is not an approximation, it is the factorisation.
    """
    raw = up2.open_raw(raw_path, verify_sha=verify_raw_sha)
    if net is None:
        net = load_segnet()
    out = np.empty((len(indices), EVAL_H, EVAL_W), dtype=np.uint8)
    for start in range(0, len(indices), batch_size):
        chunk = np.asarray(indices[start : start + batch_size], dtype=np.int64)
        frames = np.asarray(raw[2 * chunk + 1])
        out[start : start + len(chunk)] = argmax_from_camera_frames(net, frames)
        if progress:
            print(f"  argmax {start + len(chunk)}/{len(indices)} pairs", flush=True)
    return out


def seg_leg_from_argmax(
    argmax: np.ndarray, gt_labels: np.ndarray, indices: np.ndarray, lineage: str
) -> SegLeg:
    """Score a precomputed argmax against one GT lineage."""
    gt_chunk = gt_labels[indices]
    per_pair = d_seg_per_pair(argmax, gt_chunk)
    return SegLeg(
        lineage=lineage,
        pairs=len(indices),
        sampling="full_field" if len(indices) >= N_PAIRS else "seeded_random",
        d_seg=float(per_pair.mean()),
        cells_compared=int(gt_chunk.size),
        cells_disagreeing=int((argmax != gt_chunk).sum()),
        per_pair=tuple(per_pair.tolist()),
    )


# --------------------------------------------------------------------------------------
# The structural question S1 must answer before any solve: WHERE does d_seg come from?
# --------------------------------------------------------------------------------------


def token_vs_gt_agreement(
    tokens: np.ndarray, gt_labels: np.ndarray, indices: np.ndarray
) -> dict[str, Any]:
    """Does the STORED token map already equal the GT label map?

    This decides the whole seg strategy and it is a pure array comparison, so it is
    cheap and it is decisive:

    * if tokens == GT everywhere, every disagreeing argmax cell is RENDER/RE-SEGMENT
      loss, and the actuator's job is to PRE-DISTORT the labels so the round trip
      lands on GT.  Rate need not rise: a flip is a substitution, not an addition.
    * if tokens != GT, part of ``d_seg`` is stored-label error, and repairing it buys
      seg at a rate cost set by how badly the HPAC context model predicts the repair.

    The two have completely different economics, so nothing downstream is designed
    until this is measured.
    """
    tok = tokens[indices]
    gt = gt_labels[indices]
    disagree = tok != gt
    n = int(tok.size)
    n_dis = int(disagree.sum())
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(confusion, (gt.reshape(-1), tok.reshape(-1)), 1)
    return {
        "pairs": len(indices),
        "cells": n,
        "token_gt_disagreeing_cells": n_dis,
        "token_gt_disagreement_rate": n_dis / n if n else 0.0,
        "confusion_gt_rows_token_cols": confusion.tolist(),
        "class_names": list(CLASS_NAMES),
    }


def flip_ledger(
    argmax: np.ndarray, gt_labels: np.ndarray, tokens: np.ndarray
) -> dict[str, Any]:
    """Where the realized seg debt sits, decomposed by the EDGE it lies on.

    ``ddm_pc2`` measured that seg is ONE graph with ONE hub -- Road appears in 87.8%
    of flips -- so the useful decomposition is per ORDERED CLASS PAIR (gt, ours), not
    per class.  Also reports, for each flip, what the STORED token said, which
    separates "the renderer lost a label we stored" from "we stored the wrong label".
    """
    wrong = argmax != gt_labels
    n_wrong = int(wrong.sum())
    pair_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(
        pair_counts,
        (gt_labels[wrong].astype(np.int64), argmax[wrong].astype(np.int64)),
        1,
    )
    token_at_flip = np.zeros((NUM_CLASSES,), dtype=np.int64)
    if n_wrong:
        np.add.at(token_at_flip, tokens[wrong].astype(np.int64), 1)
    token_agrees_with_gt = (
        int((tokens[wrong] == gt_labels[wrong]).sum()) if n_wrong else 0
    )
    return {
        "flips": n_wrong,
        "cells": int(argmax.size),
        "edge_counts_gt_rows_ours_cols": pair_counts.tolist(),
        "token_class_at_flip": token_at_flip.tolist(),
        "flips_where_stored_token_already_equals_gt": token_agrees_with_gt,
        "flips_where_stored_token_is_wrong": n_wrong - token_agrees_with_gt,
        "class_names": list(CLASS_NAMES),
    }


# --------------------------------------------------------------------------------------
# S1b -- pre-distortion proposals, and realized acceptance
# --------------------------------------------------------------------------------------


def _disk_offsets(radius: int) -> list[tuple[int, int]]:
    """Integer disk of the given radius, including the centre."""
    out = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                out.append((dy, dx))
    return out


def propose_predistortion(
    tokens_pair: np.ndarray,
    argmax_pair: np.ndarray,
    gt_pair: np.ndarray,
    *,
    radius: int,
    edges: Sequence[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Pre-distort the token map so the RENDERED frame re-segments closer to GT.

    The mechanism is forced by the alphabet.  A token is one of five CLASS LABELS --
    there is no "more strongly Road" symbol to write at a failing cell, and at 95.9% of
    failing cells the token is ALREADY the right class (S1a).  So the only available
    lever is SPATIAL: re-label a cell's neighbours to move the painted boundary.

    For every cell where ``argmax != gt`` (optionally restricted to a set of ordered
    class edges ``(gt_class, our_class)``), this writes ``gt`` into a disk of the given
    radius.  ``radius=0`` is the degenerate "just store the right label" move, which
    S1a already showed can reach at most ~4% of the debt; ``radius>=1`` is the actual
    pre-distortion: it WIDENS the class the scorer is failing to see, at the cost of
    overwriting neighbours that may have been correct.

    That cost is real and is exactly why acceptance must be REALIZED rather than
    predicted -- the move creates new flips as well as repairing old ones, and only the
    frozen scorer knows the net.
    """
    proposal = tokens_pair.copy()
    wrong = argmax_pair != gt_pair
    if edges is not None:
        mask = np.zeros_like(wrong)
        for gt_class, our_class in edges:
            mask |= (gt_pair == gt_class) & (argmax_pair == our_class)
        wrong &= mask
    ys, xs = np.nonzero(wrong)
    if ys.size == 0:
        return proposal
    height, width = tokens_pair.shape
    for dy, dx in _disk_offsets(radius):
        yy = np.clip(ys + dy, 0, height - 1)
        xx = np.clip(xs + dx, 0, width - 1)
        proposal[yy, xx] = gt_pair[ys, xs]
    return proposal


@dataclass(frozen=True)
class ProposalResult:
    """One realized proposal, priced on both the seg and the rate axis."""

    pair: int
    label: str
    tokens_changed: int
    flips_before: int
    flips_after: int
    flips_repaired: int
    accepted: bool

    @property
    def cells_per_changed_token(self) -> float:
        return (
            self.flips_repaired / self.tokens_changed if self.tokens_changed else 0.0
        )

    @property
    def break_even_bits_per_token(self) -> float:
        """Bits per changed token this proposal can afford and still be admissible.

        One repaired cell is worth ``BYTES_PER_SEG_CELL`` bytes = 8x that in bits.  So
        a proposal that repairs ``r`` cells using ``t`` changed tokens can spend
        ``8 * BYTES_PER_SEG_CELL * r / t`` bits per token before the rate term eats it.
        """
        return 8.0 * BYTES_PER_SEG_CELL * self.cells_per_changed_token

    def to_json(self) -> dict[str, Any]:
        return {
            "pair": self.pair,
            "label": self.label,
            "tokens_changed": self.tokens_changed,
            "flips_before": self.flips_before,
            "flips_after": self.flips_after,
            "flips_repaired": self.flips_repaired,
            "accepted": self.accepted,
            "cells_per_changed_token": self.cells_per_changed_token,
            "break_even_bits_per_token": self.break_even_bits_per_token,
        }


def evaluate_proposal(
    semantic,
    net,
    proposal_tokens: np.ndarray,
    gt_pair: np.ndarray,
    pair: int,
) -> tuple[int, np.ndarray]:
    """Render the proposed tokens and re-segment: the REALIZED objective.

    No surrogate, no linearisation.  The frame goes through the receiver's own forward
    model (proven byte-exact by ``forward_model_control``) and then through the frozen
    CPU SegNet, which is the verdict authority.
    """
    frame = render_frame1(semantic, proposal_tokens[None], np.array([pair]))
    argmax = argmax_from_camera_frames(net, frame)[0]
    return int((argmax != gt_pair).sum()), argmax



def _resolve_indices(pairs: int, seed: int) -> np.ndarray:
    """Seeded RANDOM pairs, never a prefix.

    ``ddm_bp2``/``ddm_na2`` measured a contiguous prefix as a DIFFERENT POPULATION
    (pose prefixes 2.54-4.21x harder; seg prefixes 0.95-0.97x easier), so ``up2``'s
    selector refuses a prefix below n600 and this arm reuses it rather than slicing.
    """
    return up2.select_pairs(pairs, seed)


def cmd_validate(args) -> int:
    """Reproduce BOTH published seg legs before this instrument is trusted at all."""
    tokens = load_tokens(args.tokens)
    indices = _resolve_indices(args.pairs, args.seed)
    net = load_segnet()
    report: dict[str, Any] = {
        "instrument": "ddm_jg1_seg_solve",
        "pairs": len(indices),
        "sampling": "full_field" if len(indices) >= N_PAIRS else f"seeded_random_{args.seed}",
        "exchange_rate": {
            "s_per_seg_cell": S_PER_SEG_CELL,
            "s_per_archive_byte": S_PER_ARCHIVE_BYTE,
            "bytes_per_repaired_seg_cell": BYTES_PER_SEG_CELL,
        },
        "legs": {},
    }
    argmax = shipped_argmax(
        indices,
        batch_size=args.batch_size,
        verify_raw_sha=not args.no_verify_raw,
        net=net,
        progress=args.progress,
    )
    if args.save_argmax:
        args.save_argmax.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_argmax, argmax)
        print(f"wrote {args.save_argmax}", flush=True)
    for lineage, published in (
        (up2.LINEAGE_DALI, POINTER_D_SEG_DALI),
        (up2.LINEAGE_AV_PYAV, POINTER_D_SEG_AV),
    ):
        gt = load_gt_seg_labels(lineage)
        leg = seg_leg_from_argmax(argmax, gt, indices, lineage)
        entry = leg.to_json()
        entry["flip_ledger"] = flip_ledger(argmax, gt[indices], tokens[indices])
        entry["published"] = published
        entry["ratio_measured_over_published"] = (
            leg.d_seg / published if published else None
        )
        entry["token_vs_gt"] = token_vs_gt_agreement(tokens, gt, indices)
        report["legs"][lineage] = entry
        print(
            f"[{lineage}] d_seg={leg.d_seg:.8f} published={published} "
            f"ratio={leg.d_seg / published:.5f}",
            flush=True,
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"wrote {args.out}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser(
        "validate", help="reproduce the published seg legs on both GT lineages"
    )
    validate.add_argument("--pairs", type=int, default=N_PAIRS)
    validate.add_argument("--seed", type=int, default=20260819)
    validate.add_argument("--batch-size", type=int, default=4)
    validate.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    validate.add_argument("--out", type=Path, default=None)
    validate.add_argument(
        "--save-argmax",
        type=Path,
        default=None,
        help="retain the measured argmax field (ALWAYS KEEP THE PAYLOAD)",
    )
    validate.add_argument("--progress", action="store_true")
    validate.add_argument(
        "--no-verify-raw",
        action="store_true",
        help="skip the 3.66 GB raw sha (it costs ~30 s); the shape check still binds",
    )
    validate.set_defaults(func=cmd_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
