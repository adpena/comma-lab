#!/usr/bin/env python3
"""ddm_rt1 -- decompose the hv1 ep0634 render->SegNet round-trip seg loss.

Charter: `.omx/research/ddm_rt1_seg_roundtrip_decomposition_charter_20260816.md`.

The measured premise (ddm_td1, do not re-derive): on the hv1 ep0634 archive the transmitted
label field disagrees with the GT SegNet argmax at only 1,717 of 117,964,800 scored pixels,
while the scored seg term is 34,930.6 flips.  ~95% of the seg axis is manufactured between the
label field we ship and the argmax the scorer reads back.  This tool measures WHERE.

Stage taxonomy inherited from the OPTIMAL-FORM reference `ddm_v14_realization_fidelity` (#624):

    L   transmitted label field           (600, 384, 512) uint8, spatial order
    G   SegNet argmax of the GT video     (600, 384, 512) uint8, cached
    A   SegNet argmax of the shipped decode (this tool measures it)

    S1  neural-render deviation   -- render RGB vs an ideal flat paint of its OWN labels
    S2  paint -> SegNet response  -- ideal flat paint of L, through R, does not read back as L
    S3  resize/uint8 (R)          -- camera-res lift + bilinear down vs direct scorer-res paint
    S4  GT-side flicker floor     -- fl1's spike population, joined from the cached G alone

Every counted row is full n600.  Axis is `[macOS-CPU advisory]` -- NEVER a score.

Instrument pins (et4: batch shape is part of the forward instrument): frozen CPU torch SegNet
from `upstream/models/segnet.safetensors`, batch = 1 pair, torch threads fixed by --threads,
`SegNet.preprocess_input` verbatim (last frame of the pair, bilinear interpolate to 384x512).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "upstream"

# --- provenance pins (charter) -------------------------------------------------------------
BASE_ARCHIVE_SHA256 = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
SCORED_SEG_FLIPS_CUDA = 34930.6  # 100 * d_seg = 0.029611 on [contest-CUDA T4 n600]
D_SEG_CUDA = 2.9611e-04
LABEL_VS_GT_FLIPS_TD1 = 1717  # td1 exact, full field

FRAMES = 600
SEG_H, SEG_W = 384, 512
CAM_H, CAM_W = 874, 1164
SCORED_PX = FRAMES * SEG_H * SEG_W  # 117,964,800
N_CLASSES = 5

DEFAULT_RAW = Path(
    "/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/runs/"
    "base_optimized_n600_r3/output/0.raw"
)
DEFAULT_TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_harvest_compose/ep0634/retained/coders/"
    "s1p25_c1p0/decoded_spatial_tokens.rc64.bin"
)
DEFAULT_GT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy"
)
DEFAULT_WORK = Path("/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816")


class Rt1Error(RuntimeError):
    """Fail-closed error for instrument / custody violations."""


# ============================================================================================
# geometry -- the NN lift used by the v14 receiver family (scorer cell -> camera pixels)
# ============================================================================================
def nn_lift_index() -> tuple[np.ndarray, np.ndarray]:
    """Row/column scorer indices for every camera pixel (nearest-neighbour lift).

    Camera pixel centres are mapped into the scorer lattice, matching `U_nn` in the v14
    canonical-equation note.  Deterministic and independent of any payload.
    """
    rows = np.minimum(((np.arange(CAM_H) + 0.5) * SEG_H / CAM_H).astype(np.int64), SEG_H - 1)
    cols = np.minimum(((np.arange(CAM_W) + 0.5) * SEG_W / CAM_W).astype(np.int64), SEG_W - 1)
    return rows, cols


def boundary(lab: np.ndarray) -> np.ndarray:
    """4-neighbour label boundary -- sq1/gp1 convention (`ddm_sq1_eta_seg_realization.py`)."""
    b = np.zeros(lab.shape, dtype=bool)
    d = lab[:-1, :] != lab[1:, :]
    b[:-1, :] |= d
    b[1:, :] |= d
    d = lab[:, :-1] != lab[:, 1:]
    b[:, :-1] |= d
    b[:, 1:] |= d
    return b


def dilate1(mask: np.ndarray) -> np.ndarray:
    """One 4-neighbour (von Neumann) dilation -- city-block radius exactly 1."""
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    return out


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """gp1's band dilation, reused VERBATIM for parity with the sq1/lr2 rung definitions.

    MEASURED shape, not the label gp1 gives it: each iteration is a full 4-neighbour dilation
    followed by a VERTICAL-ONLY dilation, so one iteration grows a point to 11 pixels --
    a 3x3 square plus one pixel above and below.  It is neither Chebyshev (9 or 25) nor
    city-block (13); it is anisotropic, one pixel taller than wide.  gp1's docstring says
    "Chebyshev"; the code is what the family's measured rungs actually used, so the code --
    not the label -- is what transfers, and this docstring records the real geometry.
    """
    out = mask.copy()
    for _ in range(r):
        acc = out.copy()
        acc[:-1, :] |= out[1:, :]
        acc[1:, :] |= out[:-1, :]
        acc[:, :-1] |= out[:, 1:]
        acc[:, 1:] |= out[:, :-1]
        cur = acc.copy()
        cur[:-1, :] |= acc[1:, :]
        cur[1:, :] |= acc[:-1, :]
        out = cur
    return out


def boundary_ring_index(lab: np.ndarray, max_r: int) -> np.ndarray:
    """Per-pixel city-block distance to the label boundary, capped at max_r + 1.

    Ring 0 = a boundary pixel; ring k = city-block distance exactly k; max_r + 1 = farther
    than max_r (label interior).  Single-step dilation is used here on purpose so the ring
    index reads directly as "pixels from the nearest label edge".
    """
    ring = np.full(lab.shape, max_r + 1, dtype=np.uint8)
    cur = boundary(lab)
    ring[cur] = 0
    for k in range(1, max_r + 1):
        nxt = dilate1(cur)
        ring[nxt & ~cur] = k
        cur = nxt
    return ring


# ============================================================================================
# instrument
# ============================================================================================
class SegInstrument:
    """Frozen CPU-torch SegNet, batch-1, upstream preprocessing verbatim."""

    def __init__(self, threads: int) -> None:
        import torch

        if str(UPSTREAM) not in sys.path:
            sys.path.insert(0, str(UPSTREAM))
        import einops
        from modules import SegNet
        from safetensors.torch import load_file

        torch.set_num_threads(threads)
        torch.set_grad_enabled(False)
        self._torch = torch
        self._einops = einops
        self.threads = threads
        net = SegNet().eval()
        net.load_state_dict(
            load_file(str(UPSTREAM / "models" / "segnet.safetensors"), device="cpu")
        )
        self.net = net

    def logits_from_camera(self, frame1_cam_u8: np.ndarray):
        """Raw (5, 384, 512) SegNet logits for one scored pair's frame_1."""
        torch = self._torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(frame1_cam_u8))[None, None]
            x = self._einops.rearrange(x, "b t h w c -> b t c h w").float()
            return self.net(self.net.preprocess_input(x))[0]

    def argmax_from_camera(self, frame1_cam_u8: np.ndarray) -> np.ndarray:
        """SegNet argmax for one scored pair, given its frame_1 at camera resolution.

        `SegNet.preprocess_input` slices `x[:, -1, ...]`, so only frame_1 reaches the network;
        a length-1 sequence is therefore bit-identical to passing the true pair and halves I/O.
        """
        torch = self._torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(frame1_cam_u8))[None, None]
            x = self._einops.rearrange(x, "b t h w c -> b t c h w").float()
            out = self.net(self.net.preprocess_input(x))
            return out.argmax(dim=1)[0].numpy().astype(np.uint8)


# ============================================================================================
# payload helpers
# ============================================================================================
def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def open_raw(raw: Path) -> np.memmap:
    frame_bytes = CAM_H * CAM_W * 3
    n = raw.stat().st_size // frame_bytes
    if n != 2 * FRAMES:
        raise Rt1Error(f"{raw} holds {n} frames, expected {2 * FRAMES}")
    return np.memmap(raw, dtype=np.uint8, mode="r", shape=(n, CAM_H, CAM_W, 3))


def open_tokens(tokens: Path) -> np.memmap:
    if tokens.stat().st_size != SCORED_PX:
        raise Rt1Error(f"{tokens} is {tokens.stat().st_size} B, expected {SCORED_PX}")
    return np.memmap(tokens, dtype=np.uint8, mode="r", shape=(FRAMES, SEG_H, SEG_W))


def write_field(work: Path, name: str, arr: np.ndarray) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    path = work / f"{name}.npy"
    np.save(path, arr)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_receipt(work: Path, name: str, payload: dict) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    path = work / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def per_class_counts(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Flip counts charged to the REFERENCE class (fl1's charge-by-target convention)."""
    bad = pred != ref
    out = {}
    for c in range(N_CLASSES):
        out[str(c)] = int(np.count_nonzero(bad & (ref == c)))
    return out


# ============================================================================================
# painting legs
# ============================================================================================
def paint_camera(labels_seg: np.ndarray, palette: np.ndarray,
                 rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Flat prototype paint of one label frame at camera resolution (v14's `q_{U_nn(s)}`)."""
    lifted = labels_seg[np.ix_(rows, cols)]
    return palette[lifted]


def paint_scorer(labels_seg: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Flat prototype paint at scorer resolution -- no lift, so SegNet's resize is identity."""
    return palette[labels_seg]


def band_repaint(frame1_cam_u8: np.ndarray, labels_seg: np.ndarray, palette: np.ndarray,
                 radius: int, rows: np.ndarray, cols: np.ndarray,
                 alpha: float = 1.0) -> np.ndarray:
    """Zero-byte legal band repaint -- the FLAT-CONTENT control, not the family's cure.

    Every input is available to the receiver (decoded RGB, transmitted labels, a 15-byte
    palette), so this rung costs no video-derived bytes beyond the palette.  sq1 §2.4 measured
    the two ends of this family on the v4d vehicle: flat/truth content pasted into the band
    gives `eta_net = -3.7640` (32/32 pairs harmed), while paint SOLVED against the frozen head
    gives `eta_net = +0.7895`.  This rung is the flat end, priced on hv1 for the first time.
    """
    band_seg = boundary(labels_seg)
    if radius > 0:
        band_seg = dilate(band_seg, radius)
    band_cam = band_seg[np.ix_(rows, cols)]
    lifted = labels_seg[np.ix_(rows, cols)]
    out = np.array(frame1_cam_u8, dtype=np.uint8, copy=True)
    target = palette[lifted]
    if alpha >= 1.0:
        out[band_cam] = target[band_cam]
    else:
        blended = (
            (1.0 - alpha) * out[band_cam].astype(np.float32)
            + alpha * target[band_cam].astype(np.float32)
        )
        out[band_cam] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return out


# ============================================================================================
# stages
# ============================================================================================
def stage_palette(args: argparse.Namespace) -> int:
    """Derive the per-class anchor colour from the decode itself (receiver-legal, 15 B)."""
    raw = open_raw(args.raw)
    tok = open_tokens(args.tokens)
    rows, cols = nn_lift_index()
    sums = np.zeros((N_CLASSES, 3), dtype=np.float64)
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    t0 = time.time()
    for t in range(FRAMES):
        lifted = np.asarray(tok[t])[np.ix_(rows, cols)]
        frame = np.asarray(raw[2 * t + 1], dtype=np.uint8)
        flat_lab = lifted.reshape(-1)
        flat_rgb = frame.reshape(-1, 3).astype(np.float64)
        counts += np.bincount(flat_lab, minlength=N_CLASSES)
        for ch in range(3):
            sums[:, ch] += np.bincount(flat_lab, weights=flat_rgb[:, ch], minlength=N_CLASSES)
    if np.any(counts == 0):
        raise Rt1Error(f"class absent from the lifted label field: counts={counts.tolist()}")
    palette = np.clip(np.rint(sums / counts[:, None]), 0, 255).astype(np.uint8)
    receipt = {
        "schema": "ddm_rt1_palette.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "derivation": "per-class mean of the shipped decode's camera RGB over NN-lifted labels",
        "palette_rgb": palette.tolist(),
        "class_pixel_counts": counts.tolist(),
        "raw": {"path": str(args.raw), "sha256_pinned": args.raw_sha256},
        "tokens": {"path": str(args.tokens)},
        "wall_s": time.time() - t0,
    }
    # PAYLOAD WRITE ORDER (ddm_pl1, cured by ddm_ql2): the receipt is the cheap,
    # irreplaceable product of this stage; `palette.npy` is a rebuildable array
    # whose values the receipt already carries verbatim (`palette_rgb`). Persist
    # the receipt first -- `write_receipt` also mkdirs `work`, so this ordering
    # additionally guarantees the directory exists before `np.save` runs.
    write_receipt(args.work, "RT1_PALETTE", receipt)
    np.save(args.work / "palette.npy", palette)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _load_palette(work: Path) -> np.ndarray:
    path = work / "palette.npy"
    if not path.exists():
        raise Rt1Error(f"missing {path}; run the `palette` stage first")
    return np.load(path)


def _run_leg(args: argparse.Namespace, leg: str) -> dict:
    """Run one n600 SegNet leg and return its measured record."""
    raw = open_raw(args.raw)
    tok = open_tokens(args.tokens)
    gt = np.load(args.gt, mmap_mode="r")
    if gt.shape != (FRAMES, SEG_H, SEG_W):
        raise Rt1Error(f"GT cache shape {gt.shape} != {(FRAMES, SEG_H, SEG_W)}")
    rows, cols = nn_lift_index()
    palette = None if leg == "base" else _load_palette(args.work)
    inst = SegInstrument(args.threads)

    pred = np.zeros((FRAMES, SEG_H, SEG_W), dtype=np.uint8)
    t0 = time.time()
    for t in range(FRAMES):
        labels = np.asarray(tok[t])
        if leg == "base":
            frame = np.asarray(raw[2 * t + 1])
            pred[t] = inst.argmax_from_camera(frame)
        elif leg == "paint_cam":
            pred[t] = inst.argmax_from_camera(paint_camera(labels, palette, rows, cols))
        elif leg == "paint_scorer":
            pred[t] = inst.argmax_from_camera(paint_scorer(labels, palette))
        elif leg == "band":
            frame = np.asarray(raw[2 * t + 1])
            edited = band_repaint(frame, labels, palette, args.radius, rows, cols, args.alpha)
            pred[t] = inst.argmax_from_camera(edited)
        else:  # pragma: no cover - argparse restricts the choices
            raise Rt1Error(f"unknown leg {leg}")
        if args.progress and (t + 1) % args.progress == 0:
            el = time.time() - t0
            print(f"  {leg} {t + 1}/{FRAMES}  {el:.0f}s  eta {el / (t + 1) * (FRAMES - t - 1):.0f}s",
                  flush=True)
    wall = time.time() - t0

    gt_arr = np.asarray(gt)
    tok_arr = np.asarray(tok)
    flips_vs_gt = int(np.count_nonzero(pred != gt_arr))
    flips_vs_label = int(np.count_nonzero(pred != tok_arr))
    tag = leg if leg != "band" else f"band_r{args.radius}_a{args.alpha:g}"
    payload = write_field(args.work, f"argmax_{tag}", pred)
    rec = {
        "schema": "ddm_rt1_leg.v1",
        "leg": tag,
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "instrument": {
            "scorer": "frozen CPU torch SegNet, upstream/models/segnet.safetensors",
            "batch_pairs": 1,
            "torch_threads": args.threads,
            "preprocess": "upstream SegNet.preprocess_input verbatim",
        },
        "n_pairs": FRAMES,
        "scored_px": SCORED_PX,
        "flips_vs_gt": flips_vs_gt,
        "d_seg_vs_gt": flips_vs_gt / SCORED_PX,
        "seg_S_units": 100.0 * flips_vs_gt / SCORED_PX,
        "flips_vs_label": flips_vs_label,
        "d_label_readback": flips_vs_label / SCORED_PX,
        "per_class_flips_vs_gt_charged_to_gt": per_class_counts(pred, gt_arr),
        "per_class_flips_vs_label_charged_to_label": per_class_counts(pred, tok_arr),
        "payload": payload,
        "wall_s": wall,
    }
    if leg == "band":
        rec["band"] = {"radius_scorer_px": args.radius, "alpha": args.alpha,
                       "carrier_bytes": 15,
                       "note": "band derived from the transmitted labels; only the 15 B palette "
                               "is new video-derived payload"}
    return rec


def stage_instrument(args: argparse.Namespace) -> int:
    """Fail-closed instrument check: reproduce the scored seg term on the retained decode."""
    raw_sha = sha256_file(args.raw)
    if not raw_sha.startswith(args.raw_sha256):
        raise Rt1Error(
            f"raw custody FAILED: {args.raw} sha256 {raw_sha} does not start with "
            f"the charter pin {args.raw_sha256}"
        )
    tok = open_tokens(args.tokens)
    gt = np.load(args.gt, mmap_mode="r")
    label_vs_gt = int(np.count_nonzero(np.asarray(tok) != np.asarray(gt)))
    rec = _run_leg(args, "base")
    ratio = rec["d_seg_vs_gt"] / D_SEG_CUDA
    control_label = label_vs_gt == LABEL_VS_GT_FLIPS_TD1
    passed = control_label and abs(ratio - 1.0) <= args.tolerance
    receipt = {
        "schema": "ddm_rt1_instrument_check.v1",
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "raw": {"path": str(args.raw), "sha256": raw_sha, "sha256_pin": args.raw_sha256},
        "reference_contest_cuda": {"d_seg": D_SEG_CUDA, "flips": SCORED_SEG_FLIPS_CUDA},
        "advisory": rec,
        "advisory_over_cuda_ratio": ratio,
        "td1_control": {
            "label_vs_gt_flips_measured": label_vs_gt,
            "label_vs_gt_flips_td1": LABEL_VS_GT_FLIPS_TD1,
            "reproduced": control_label,
        },
        "round_trip_flips_advisory": rec["flips_vs_label"],
        "round_trip_S_units_advisory": 100.0 * rec["flips_vs_label"] / SCORED_PX,
        "tolerance": args.tolerance,
        "instrument_check": "PASS" if passed else "FAIL",
    }
    write_receipt(args.work, "RT1_INSTRUMENT_CHECK", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed:
        raise Rt1Error(
            "instrument check FAILED -- no decomposition row may be claimed "
            f"(ratio={ratio:.6f}, label control reproduced={control_label})"
        )
    return 0


def stage_leg(args: argparse.Namespace) -> int:
    rec = _run_leg(args, args.leg)
    tag = rec["leg"]
    write_receipt(args.work, f"RT1_LEG_{tag}", rec)
    print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


def stage_geometry(args: argparse.Namespace) -> int:
    """Scorer-free geometry of the measured flip set: boundary rings, GT flicker, per class."""
    base_path = args.work / "argmax_base.npy"
    if not base_path.exists():
        raise Rt1Error(f"missing {base_path}; run the `instrument` stage first")
    pred = np.load(base_path, mmap_mode="r")
    tok = open_tokens(args.tokens)
    gt = np.load(args.gt, mmap_mode="r")

    max_r = args.max_ring
    ring_hist_gt = np.zeros(max_r + 2, dtype=np.int64)
    ring_hist_label = np.zeros(max_r + 2, dtype=np.int64)
    ring_pop = np.zeros(max_r + 2, dtype=np.int64)
    spike_total = 0
    spike_flip = 0
    interior_flip = 0
    # td1 H2: the label->scored amplification r.  td1 could only model it.  At the 1,717 pixels
    # where our transmitted label already disagrees with GT, we can read the transfer directly:
    # how often does the render reproduce the wrong label, and how often does that cost a flip?
    label_err_px = 0
    label_err_render_agrees_with_label = 0
    label_err_scored_flip = 0
    t0 = time.time()
    for t in range(FRAMES):
        labels = np.asarray(tok[t])
        a = np.asarray(pred[t])
        g = np.asarray(gt[t])
        ring = boundary_ring_index(labels, max_r)
        ring_pop += np.bincount(ring.reshape(-1), minlength=max_r + 2)
        ring_hist_gt += np.bincount(ring[a != g], minlength=max_r + 2)
        ring_hist_label += np.bincount(ring[a != labels], minlength=max_r + 2)
        lab_err = labels != g
        if lab_err.any():
            label_err_px += int(np.count_nonzero(lab_err))
            label_err_render_agrees_with_label += int(np.count_nonzero(lab_err & (a == labels)))
            label_err_scored_flip += int(np.count_nonzero(lab_err & (a != g)))
        if 0 < t < FRAMES - 1:
            gp = np.asarray(gt[t - 1])
            gn = np.asarray(gt[t + 1])
            spike = (g != gp) & (g != gn)
            spike_total += int(np.count_nonzero(spike))
            bad = a != g
            interior_flip += int(np.count_nonzero(bad))
            spike_flip += int(np.count_nonzero(bad & spike))
    rec = {
        "schema": "ddm_rt1_geometry.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "max_ring": max_r,
        "ring_definition": "city-block (4-neighbour) distance to the TRANSMITTED label "
                           f"boundary; 0 = on the boundary, {max_r + 1} = farther than "
                           f"{max_r} (label interior)",
        "ring_population": ring_pop.tolist(),
        "flips_vs_gt_by_ring": ring_hist_gt.tolist(),
        "flips_vs_label_by_ring": ring_hist_label.tolist(),
        "gt_flicker_join": {
            "definition": "fl1 spike: GT argmax pixel differing from BOTH stride-2 neighbours",
            "interior_pairs": FRAMES - 2,
            "spike_px_total": spike_total,
            "interior_flips_vs_gt": interior_flip,
            "flips_on_spike_px": spike_flip,
            "share_of_flips_on_spike": (spike_flip / interior_flip) if interior_flip else None,
        },
        "label_error_transfer": {
            "question": "td1 H2 -- the label->scored amplification r, read observationally at "
                        "the pixels where our transmitted label is already wrong",
            "label_error_px": label_err_px,
            "render_reproduced_the_wrong_label": label_err_render_agrees_with_label,
            "scored_flip_at_those_px": label_err_scored_flip,
            "r_observational": (label_err_scored_flip / label_err_px) if label_err_px else None,
            "caveat": "OBSERVATIONAL, not causal: it is the transfer at the label errors that "
                      "exist, not the transfer a deliberate label edit would produce. It bounds "
                      "r for this population; a drop-set edit could differ.",
        },
        "wall_s": time.time() - t0,
    }
    write_receipt(args.work, "RT1_GEOMETRY", rec)
    print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


def stage_margin(args: argparse.Namespace) -> int:
    """How reducible is the residual?  Measure the logit deficit the flips actually need.

    For every scored pixel the top-1 minus top-2 logit gap is recorded.  At a flipped pixel
    the deficit is `logit[argmax] - logit[GT class]`: the amount a finisher must move to
    recover that pixel.  A residual made of near-ties is cheap to attack; a residual with a
    large deficit is a capacity problem, not a tie-breaking problem.
    """
    raw = open_raw(args.raw)
    gt = np.load(args.gt, mmap_mode="r")
    inst = SegInstrument(args.threads)
    torch = inst._torch

    bins = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, np.inf])
    deficit_hist = np.zeros(len(bins) - 1, dtype=np.int64)
    margin_hist_ok = np.zeros(len(bins) - 1, dtype=np.int64)
    runner_up_is_gt = 0
    flips = 0
    deficit_sum = 0.0
    deficits_sample: list[float] = []
    rng = np.random.default_rng(20260816)
    t0 = time.time()
    for t in range(FRAMES):
        lg = inst.logits_from_camera(np.asarray(raw[2 * t + 1]))
        top2 = torch.topk(lg, 2, dim=0)
        pred = top2.indices[0].numpy().astype(np.uint8)
        margin = (top2.values[0] - top2.values[1]).numpy()
        g = np.asarray(gt[t])
        bad = pred != g
        nbad = int(np.count_nonzero(bad))
        flips += nbad
        if nbad:
            gt_logit = np.take_along_axis(lg.numpy(), g[None].astype(np.int64), axis=0)[0]
            top_logit = top2.values[0].numpy()
            deficit = (top_logit - gt_logit)[bad]
            deficit_sum += float(deficit.sum())
            deficit_hist += np.histogram(deficit, bins=bins)[0]
            runner_up_is_gt += int(np.count_nonzero(top2.indices[1].numpy()[bad] == g[bad]))
            if len(deficits_sample) < 200000:
                keep = rng.random(deficit.shape[0]) < 0.05
                deficits_sample.extend(deficit[keep].tolist())
        margin_hist_ok += np.histogram(margin[~bad], bins=bins)[0]
        if args.progress and (t + 1) % args.progress == 0:
            el = time.time() - t0
            print(f"  margin {t + 1}/{FRAMES} {el:.0f}s", flush=True)
    ds = np.array(deficits_sample, dtype=np.float64)
    rec = {
        "schema": "ddm_rt1_margin.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "bins": [float(b) for b in bins],
        "flips_vs_gt": flips,
        "deficit_hist_at_flips": deficit_hist.tolist(),
        "margin_hist_at_correct": margin_hist_ok.tolist(),
        "runner_up_is_gt_class": runner_up_is_gt,
        "runner_up_is_gt_share": runner_up_is_gt / flips if flips else None,
        "mean_deficit_at_flips": deficit_sum / flips if flips else None,
        "deficit_quantiles_5pct_sample": {
            q: float(np.quantile(ds, float(q))) for q in ("0.1", "0.25", "0.5", "0.75", "0.9")
        } if ds.size else None,
        "deficit_sample_n": int(ds.size),
        "wall_s": time.time() - t0,
    }
    write_receipt(args.work, "RT1_MARGIN", rec)
    print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


def stage_edgeshape(args: argparse.Namespace) -> int:
    """Scorer-free: is the edge error a systematic BIAS or symmetric jitter?

    §2.3 measured that 99.22% of the seg axis sits on the label boundary.  A boundary error can
    be (a) a systematic area bias -- the render erodes or dilates a class everywhere, which a
    0-byte deterministic morphological correction could attack -- or (b) symmetric jitter, which
    it cannot.  The signed per-class area error separates the two, and the neighbour test says
    whether the class the scorer wanted is simply one pixel away.
    """
    base_path = args.work / "argmax_base.npy"
    if not base_path.exists():
        raise Rt1Error(f"missing {base_path}; run the `instrument` stage first")
    pred = np.load(base_path, mmap_mode="r")
    tok = open_tokens(args.tokens)
    gt = np.load(args.gt, mmap_mode="r")

    area_pred = np.zeros(N_CLASSES, dtype=np.int64)
    area_gt = np.zeros(N_CLASSES, dtype=np.int64)
    area_label = np.zeros(N_CLASSES, dtype=np.int64)
    gt_in_neighbourhood = 0
    total_flips = 0
    per_class_frame_abs_bias = np.zeros(N_CLASSES, dtype=np.int64)
    edge_pairs: dict[str, int] = {}
    t0 = time.time()
    for t in range(FRAMES):
        a = np.asarray(pred[t])
        g = np.asarray(gt[t])
        lab = np.asarray(tok[t])
        ap = np.bincount(a.reshape(-1), minlength=N_CLASSES)
        gp = np.bincount(g.reshape(-1), minlength=N_CLASSES)
        area_pred += ap
        area_gt += gp
        area_label += np.bincount(lab.reshape(-1), minlength=N_CLASSES)
        per_class_frame_abs_bias += np.abs(ap - gp)
        bad = a != g
        nbad = int(np.count_nonzero(bad))
        total_flips += nbad
        if not nbad:
            continue
        # does the class the scorer wanted appear in the 4-neighbourhood of our own field?
        near = np.zeros(a.shape, dtype=bool)
        near[:-1, :] |= a[1:, :] == g[:-1, :]
        near[1:, :] |= a[:-1, :] == g[1:, :]
        near[:, :-1] |= a[:, 1:] == g[:, :-1]
        near[:, 1:] |= a[:, :-1] == g[:, 1:]
        gt_in_neighbourhood += int(np.count_nonzero(bad & near))
        ys, xs = np.nonzero(bad)
        keys, cnts = np.unique(
            a[ys, xs].astype(np.int64) * N_CLASSES + g[ys, xs].astype(np.int64),
            return_counts=True,
        )
        for key, cnt in zip(keys, cnts, strict=True):
            name = f"{int(key) // N_CLASSES}->{int(key) % N_CLASSES}"
            edge_pairs[name] = edge_pairs.get(name, 0) + int(cnt)
    bias = (area_pred - area_gt)
    rec = {
        "schema": "ddm_rt1_edgeshape.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "area_pred": area_pred.tolist(),
        "area_gt": area_gt.tolist(),
        "area_label": area_label.tolist(),
        "signed_area_bias_pred_minus_gt": bias.tolist(),
        "signed_bias_over_class_flips": {
            str(c): (float(bias[c]) / float(per_class_frame_abs_bias[c])
                     if per_class_frame_abs_bias[c] else None)
            for c in range(N_CLASSES)
        },
        "sum_abs_per_frame_area_error": per_class_frame_abs_bias.tolist(),
        "flips_total": total_flips,
        "flips_with_gt_class_in_4_neighbourhood": gt_in_neighbourhood,
        "share_gt_class_one_pixel_away": (
            gt_in_neighbourhood / total_flips if total_flips else None),
        "confusion_pred_to_gt": dict(sorted(edge_pairs.items(), key=lambda kv: -kv[1])),
        "note": "signed_area_bias is summed over n600; a bias whose magnitude approaches the "
                "per-frame absolute area error is systematic, one near zero is jitter",
        "wall_s": time.time() - t0,
    }
    write_receipt(args.work, "RT1_EDGESHAPE", rec)
    print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


def stage_ledger(args: argparse.Namespace) -> int:
    """Join the landed leg receipts into the per-stage flip ledger, in S units.

    Nothing here re-measures; it only converts counted flips to S with td1's derived law
    `seg_dS_per_flip = 100 / 117,964,800` and records which differences are additive.
    """
    per_flip = 100.0 / SCORED_PX

    def load(name: str) -> dict | None:
        path = args.work / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    check = load("RT1_INSTRUMENT_CHECK")
    if check is None:
        raise Rt1Error("RT1_INSTRUMENT_CHECK.json missing; the instrument gate is unpaid")
    if check.get("instrument_check") != "PASS":
        raise Rt1Error("instrument check did not PASS; no ledger row may be claimed")
    base = check["advisory"]
    legs = {n: load(f"RT1_LEG_{n}") for n in ("paint_cam", "paint_scorer")}
    band = {}
    for path in sorted(args.work.glob("RT1_LEG_band_*.json")):
        rec = json.loads(path.read_text())
        band[rec["leg"]] = rec
    geometry = load("RT1_GEOMETRY")
    margin = load("RT1_MARGIN")

    rows = [
        {"row": "REF contest-CUDA scored seg term", "flips": SCORED_SEG_FLIPS_CUDA,
         "S": 100.0 * D_SEG_CUDA, "source": "hv1 ep0634 FINAL_RESULT / charter pin"},
        {"row": "REF advisory reproduction", "flips": base["flips_vs_gt"],
         "S": base["seg_S_units"], "source": "this unit, n600"},
        {"row": "L0 transmitted labels vs GT", "flips": check["td1_control"][
            "label_vs_gt_flips_measured"],
         "S": check["td1_control"]["label_vs_gt_flips_measured"] * per_flip,
         "source": "td1 control, reproduced exactly"},
        {"row": "RT round trip (render argmax vs shipped labels)",
         "flips": base["flips_vs_label"], "S": base["flips_vs_label"] * per_flip,
         "source": "this unit, n600, EXACT (not modelled)"},
    ]
    for name, rec in legs.items():
        if rec is None:
            continue
        rows.append({
            "row": f"S2 flat-prototype paint ({name}) read back vs its own labels",
            "flips": rec["flips_vs_label"], "S": rec["flips_vs_label"] * per_flip,
            "source": "this unit, n600",
        })
        rows.append({
            "row": f"flat-prototype paint ({name}) scored d_seg vs GT",
            "flips": rec["flips_vs_gt"], "S": rec["seg_S_units"], "source": "this unit, n600",
        })
    for tag, rec in band.items():
        rows.append({
            "row": f"CURE-probe {tag}: scored d_seg vs GT after a zero-byte band repaint",
            "flips": rec["flips_vs_gt"], "S": rec["seg_S_units"], "source": "this unit, n600",
        })

    derived: dict = {}
    if legs["paint_cam"] and legs["paint_scorer"]:
        s3 = legs["paint_cam"]["flips_vs_label"] - legs["paint_scorer"]["flips_vs_label"]
        derived["S3_resize_uint8_flips"] = s3
        derived["S3_resize_uint8_S"] = s3 * per_flip
        derived["S3_note"] = (
            "paint_cam minus paint_scorer on the SAME label field; additive by construction "
            "because both legs read back the same object through the same head"
        )
    if legs["paint_cam"]:
        s1 = base["flips_vs_label"] - legs["paint_cam"]["flips_vs_label"]
        derived["S1_render_vs_flat_paint_flips"] = s1
        derived["S1_render_vs_flat_paint_S"] = s1 * per_flip
        derived["S1_note"] = (
            "NEGATIVE means the trained render reads back BETTER than a flat prototype paint "
            "of the same labels, i.e. the v14 paint stage is not a floor this vehicle sits "
            "above -- the stages are NOT nested and the decomposition is NON-ADDITIVE"
        )
        derived["flat_paint_over_render_ratio"] = (
            legs["paint_cam"]["flips_vs_label"] / base["flips_vs_label"]
        )
    for tag, rec in band.items():
        derived[f"cure_{tag}_delta_flips_vs_base"] = rec["flips_vs_gt"] - base["flips_vs_gt"]
        derived[f"cure_{tag}_delta_S_vs_base"] = (
            rec["flips_vs_gt"] - base["flips_vs_gt"]) * per_flip

    # ALWAYS KEEP THE PAYLOAD: the successor prices a coder for exactly this object, so persist
    # it rather than leaving it as a recipe.
    tok = open_tokens(args.tokens)
    gt = np.load(args.gt, mmap_mode="r")
    pred = np.load(args.work / "argmax_base.npy", mmap_mode="r")
    flip_mask = np.zeros((FRAMES, SEG_H, SEG_W), dtype=np.uint8)
    band_mask = np.zeros((FRAMES, SEG_H, SEG_W), dtype=np.uint8)
    target = np.zeros((FRAMES, SEG_H, SEG_W), dtype=np.uint8)
    for t in range(FRAMES):
        labels = np.asarray(tok[t])
        bad = np.asarray(pred[t]) != np.asarray(gt[t])
        flip_mask[t] = bad.astype(np.uint8)
        band_mask[t] = boundary(labels).astype(np.uint8)
        target[t] = np.where(bad, np.asarray(gt[t]), 255)
    payloads = {
        "flip_mask": write_field(args.work, "flip_mask_vs_gt", flip_mask),
        "free_band_mask": write_field(args.work, "free_band_mask", band_mask),
        "flip_target_class": write_field(args.work, "flip_target_class", target),
    }

    correction_bound = None
    if geometry is not None:
        # The decoder owns the label boundary for free, so a seg-correction channel only has to
        # name WHICH band pixels flip.  This is the i.i.d. entropy floor of that mask -- a bound
        # on the carrier, MODELED, not measured, and it credits no realization loss.
        band_px = int(geometry["ring_population"][0])
        band_flips = int(geometry["flips_vs_gt_by_ring"][0])
        p = band_flips / band_px
        h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        mask_bytes = band_px * h / 8.0
        rate_ds_per_byte = 25.0 / 37_545_489
        gain_all = base["flips_vs_gt"] * per_flip
        correction_bound = {
            "label": "MODELED first-order bound, NOT measured; credits no collateral and no "
                     "class-disambiguation bits",
            "free_band_support_px": band_px,
            "flips_on_band": band_flips,
            "band_flip_density": p,
            "iid_entropy_bits_per_band_px": h,
            "mask_carrier_bytes_iid_floor": mask_bytes,
            "rate_dS_per_byte": rate_ds_per_byte,
            "seg_gain_if_all_flips_fixed_S": -gain_all,
            "eta_sensitivity": {
                str(eta): {
                    "seg_gain_S": -eta * gain_all,
                    "net_S_at_iid_floor": -eta * gain_all + mask_bytes * rate_ds_per_byte,
                }
                # sq1 §2.4 measured these two on the v4d vehicle: pose-constrained and free
                for eta in (0.5406, 0.7895, 1.0)
            },
            "carrier_bytes_that_break_even_at_eta_0p7895": (
                0.7895 * gain_all / rate_ds_per_byte),
        }

    out = {
        "schema": "ddm_rt1_ledger.v1",
        "axis": "[macOS-CPU advisory] -- NEVER a score",
        "score_claim": False,
        "promotable": False,
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "seg_dS_per_flip": per_flip,
        "scored_px": SCORED_PX,
        "instrument_check": check["instrument_check"],
        "advisory_over_cuda_ratio": check["advisory_over_cuda_ratio"],
        "rows": rows,
        "derived": derived,
        "geometry": geometry,
        "margin": margin,
        "edgeshape": load("RT1_EDGESHAPE"),
        "correction_channel_bound": correction_bound,
        "successor_payloads": payloads,
        "successor_payload_note": "flip_target_class uses 255 for 'no correction'; the free band "
                                  "mask is a deterministic function of the shipped tokens and "
                                  "costs the receiver zero bytes",
    }
    write_receipt(args.work, "RT1_LEDGER", out)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",
                    choices=["palette", "instrument", "leg", "geometry", "margin",
                             "edgeshape", "ledger"])
    ap.add_argument("--leg", choices=["base", "paint_cam", "paint_scorer", "band"],
                    default="base")
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--raw-sha256", default="e5539653")
    ap.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--radius", type=int, default=1, help="band radius in scorer pixels")
    ap.add_argument("--alpha", type=float, default=1.0, help="repaint blend toward prototype")
    ap.add_argument("--max-ring", type=int, default=6)
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="allowed |advisory/CUDA - 1| for the instrument check")
    ap.add_argument("--progress", type=int, default=100)
    args = ap.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    if args.stage == "palette":
        return stage_palette(args)
    if args.stage == "instrument":
        return stage_instrument(args)
    if args.stage == "leg":
        return stage_leg(args)
    if args.stage == "margin":
        return stage_margin(args)
    if args.stage == "edgeshape":
        return stage_edgeshape(args)
    if args.stage == "ledger":
        return stage_ledger(args)
    return stage_geometry(args)


if __name__ == "__main__":
    raise SystemExit(main())
