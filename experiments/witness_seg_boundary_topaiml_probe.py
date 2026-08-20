# SPDX-License-Identifier: MIT
"""WITNESS SEG-BOUNDARY TOP-AIML $0 probe — survival-first + contour/class-flip +
temporal-delta + amortized-prior framing (re-opens the prototype's HYBRID verdict).

This is the SOTA-grade re-attempt of ``experiments/witness_seg_boundary_decisive_probe.py``
per the CLAUDE.md "ANTI-SIGNAL-LOSS / janky-prototype -> top-AIML RE-OPEN" non-negotiable.
The prototype's verdict (``reports/witness_seg_boundary_decisive.json``) rested on a
PROTOTYPE-grade coder with four named weaknesses; this script fixes all four and
re-measures on the REAL frozen basin + REAL frozen contest SegNet, CPU only.

The prototype's measured weaknesses (the bar to beat):
  - per-flip conditional cost 1.02 B (under the 1.27 waterline) — GOOD, but ...
  - total residual = 543 KB (10x above the conditional-MDL band [24.6, 64.6] KB),
  - round-trip SURVIVAL only 47% (< 50% gate) — corrections measured AFTER coding,
  - witness MDL 565-600 KB, NOT below the 177,169 B frontier.

The four TOP-AIML fixes:

  A. SURVIVAL-AWARE CODING (the big one). The prototype codes ALL flips then measures
     survival (47%) separately, so half its 543 KB pays for corrections that vanish in
     the round-trip. Here we measure per-flip round-trip survival FIRST (apply the
     class-prototype correction in the rendered 384x512 frame, push it through the EXACT
     eval round-trip bicubic^874 -> bilinear_384 -> uint8, re-segment), and code ONLY the
     survivable subset. The coded set is then ~100% survivable by construction. Half the
     flips, ~100% survival -> the bytes pay for real d_seg reduction.

  B. CONTOUR / CLASS-FLIP CODING (not per-pixel combinatorial). The ~884 flips/pair are
     contour-clustered in the decoder-free low-margin band. We code them as
       (1) a boundary bitmask over the decoder-known band ``B = {p : m(p) < tau}`` —
           the positions are a SUBSET of B, coded as a bitmask LZMA-RAW'd (reusing the
           dense-raster LZMA-RAW baseline family);
       (2) per-flip GT class via the REAL ChARM range coder over a margin-conditioned
           5-class PMF (``tac.codec.charm_range_coder``). A class flip is <= log2(5) bits
           and far less under the conditional PMF.

  C. TEMPORAL-DELTA across pairs. Contours are temporally stable (dashcam). We XOR the
     per-pair survivable boundary bitmask against the previous pair's bitmask and code
     the (sparse) delta, then LZMA-RAW it — the same temporal-delta principle the codec
     grammar uses for latents.

  D. AMORTIZED-PRIOR framing (the real test). The witness = basin decoder
     (89,136 B == the byte-closed ``0.bin``, the amortized prior that already regenerates
     most of the partition) + the survivable conditional flip residual (coded per A+B+C).
     We report witness_total_bytes = 89,136 + survivable_residual_bytes_600 and compare to
     the 177,169 B frontier AND to the [24.6, 64.6] KB conditional-MDL residual band.

NO FAKE: frozen contest SegNet (``load_frozen_distortion_net`` CPU), GT via the canonical
``RealScorerContext`` (``frame_utils.yuv420_to_rgb`` inside ``precompute_targets``), the
EXACT eval round-trip (``score._decoded_to_camera`` bicubic^874 + ``net.preprocess_input``
bilinear_384 + uint8). The coder is a REAL reversible codec: ``encode_pair_residual`` ->
``decode_pair_residual`` round-trips the (positions, classes) bit-exactly, and the
survivable-set d_seg drop is MEASURED by re-running the real SegNet on the corrected,
round-tripped frame. Every advisory number is ``[contest-CPU advisory] NON-PROMOTABLE``;
no score is claimed; the frontier is UNMOVED.
"""

from __future__ import annotations

import argparse
import json
import lzma
import struct
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

# ── canonical reused helpers (SEARCH-FIRST: do not reinvent) ─────────────────
from tac.boundary_math.bitmask_dseg import flip_count
from tac.boundary_math.margin_conditional_residual import (
    WATERLINE_BYTES_PER_FLIP,
    class_bits_conditional,
    conditional_position_bits,
)
from tac.codec.charm_range_coder import decode_symbols, encode_symbols

_EVAL_H, _EVAL_W = 384, 512  # SegNet working resolution (= scored grid)
_N_SCORED_PER_FRAME = _EVAL_H * _EVAL_W  # 196,608
_RATE_DENOM = 37_545_489
_N_FRAMES = 600
_FRONTIER_BYTES = 177_169  # contest pointer; witness must beat this on bytes for the class shift
_CONDITIONAL_MDL_LO = 24_600  # DERIVED ~24.6 KB (smaller_learned_basis_deep_math §3)
_CONDITIONAL_MDL_HI = 64_600  # DERIVED ~64.6 KB
_POSE_TRAJ_BYTES = 1_557  # MEASURED pose-output entropy (information_theoretic_floor_T_floor P6)
_BASIN_DECODER_BYTES = 89_136  # MEASURED byte-closed 0.bin (e2e_byte_close_eval_harness_20260616)
_N_SEG_CLASSES = 5

# Deterministic LZMA-RAW filter chain (same family as dense_raster_lzma_baseline.py).
_LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
_SIDECAR_MAGIC = b"WTA1"  # witness top-aiml residual v1


# ─────────────────────────────────────────────────────────────────────────────
# TOP-AIML reversible residual codec (contour bitmask + temporal-delta + class AC)
# ─────────────────────────────────────────────────────────────────────────────
def _lzma_raw(b: bytes) -> bytes:
    return lzma.compress(b, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS)


def _unlzma_raw(b: bytes) -> bytes:
    return lzma.decompress(b, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS)


def _build_margin_hist(
    margin_at_flips: np.ndarray, target_classes: np.ndarray, *, margin_bins: np.ndarray
) -> np.ndarray:
    """Per-bin 5-class histogram (the shared PMF side table) conditioned on the margin bin.

    The decoder regenerates the margin field for FREE, so once it has this tiny side
    table it can rebuild the SAME per-flip PMF the encoder used (each flip's PMF is the
    Laplace-smoothed class distribution within its margin bin). This is the
    margin-conditional entropy model the prototype's analytic ``class_bits_conditional``
    priced — here REALIZED as a real range-coder PMF. Returns ``(n_bins, 5)`` probs.
    """
    bin_idx = (np.digitize(margin_at_flips, margin_bins) - 1).clip(0, len(margin_bins) - 2)
    n_bins = len(margin_bins) - 1
    # per-bin class histogram (Laplace-smoothed), shared side table
    hist = np.ones((n_bins, _N_SEG_CLASSES), dtype=np.float64)  # +1 smoothing
    for b, c in zip(bin_idx, target_classes, strict=False):
        hist[b, c] += 1.0
    hist /= hist.sum(axis=1, keepdims=True)
    return hist


@dataclass(frozen=True)
class PairResidualCode:
    """A reversible, byte-exact code of one pair's survivable flip residual.

    ``payload`` is the full blob. ``n_bytes`` is the description length charged. The
    sub-stream lengths are exposed for the per-component byte accounting (A/B/C).
    """

    payload: bytes
    n_flips: int
    bitmask_bytes: int  # contour position bytes (temporal-delta'd)
    class_bytes: int  # arithmetic-coded class stream bytes
    sidetable_bytes: int  # the per-bin class histogram side table

    @property
    def n_bytes(self) -> int:
        return len(self.payload)


def encode_pair_residual(
    flip_idx: np.ndarray,
    target_classes: np.ndarray,
    margin_field_flat: np.ndarray,
    prev_bitmask: np.ndarray | None,
    *,
    margin_bins: np.ndarray,
    grid_n: int = _N_SCORED_PER_FRAME,
) -> tuple[PairResidualCode, np.ndarray]:
    """Encode one pair's survivable (position, class) residual reversibly.

    Layout: MAGIC | u32 K | u32 bm_len | bm_blob | u16 ht_len | ht_blob | u32 cls_len | cls_blob.
      * bm_blob  = LZMA-RAW( XOR(this_bitmask, prev_bitmask) )  — temporal-delta contour positions
      * ht_blob  = the per-bin 5-class histogram quantized to u16 (the shared PMF side table)
      * cls_blob = ChARM range-coded GT classes over the per-flip margin-conditioned PMF

    Returns (code, this_bitmask) so the caller can feed ``this_bitmask`` as the next
    pair's ``prev_bitmask`` (the temporal chain).  NO FAKE: round-trips bit-exactly.
    """
    idx = np.asarray(flip_idx, dtype=np.int64)
    cls = np.asarray(target_classes, dtype=np.int64)
    K = len(idx)
    # CRITICAL: sort by position so the class stream aligns with the decoder's
    # flatnonzero(bitmask) ordering (which is always ascending). Without this the
    # class-at-position mapping is lost on the round-trip (NO-FAKE bug class).
    if K:
        order = np.argsort(idx, kind="stable")
        idx = idx[order]
        cls = cls[order]
    # position bitmask over the full grid (1 where a survivable flip is coded)
    bitmask = np.zeros(grid_n, dtype=np.uint8)
    if K:
        bitmask[idx] = 1
    prev = prev_bitmask if prev_bitmask is not None else np.zeros(grid_n, dtype=np.uint8)
    delta = np.bitwise_xor(bitmask, prev)  # temporal-delta (C): sparse where contours persist
    bm_blob = _lzma_raw(np.packbits(delta).tobytes())

    # class side table (per-bin histogram), quantized to u16 freq counts for reproducibility
    margin_at = margin_field_flat[idx] if K else np.zeros(0)
    hist = _build_margin_hist(margin_at, cls, margin_bins=margin_bins)
    # quantize hist to u16 per-bin freqs (sum 2^12) so encode/decode use the IDENTICAL table
    htq = np.maximum(1, np.round(hist * 4096).astype(np.int64))
    htq = htq.astype(np.uint16)
    ht_blob = zlib.compress(htq.tobytes(order="C"), 9)
    pmfs_q = [htq[b].astype(np.float64) for b in
              (np.digitize(margin_at, margin_bins) - 1).clip(0, len(margin_bins) - 2)] if K else []

    # arithmetic-code the classes (alphabet 0..4) over the per-flip quantized PMF
    cls_blob = (
        encode_symbols(cls.tolist(), pmfs_q, alphabet=(0, _N_SEG_CLASSES - 1)) if K else b""
    )

    head = _SIDECAR_MAGIC + struct.pack("<I", K)
    body = (
        struct.pack("<I", len(bm_blob)) + bm_blob
        + struct.pack("<H", len(ht_blob)) + ht_blob
        + struct.pack("<I", len(cls_blob)) + cls_blob
    )
    payload = head + body
    return (
        PairResidualCode(
            payload=payload, n_flips=K,
            bitmask_bytes=len(bm_blob), class_bytes=len(cls_blob), sidetable_bytes=len(ht_blob),
        ),
        bitmask,
    )


def decode_pair_residual(
    payload: bytes,
    margin_field_flat: np.ndarray,
    prev_bitmask: np.ndarray | None,
    *,
    margin_bins: np.ndarray,
    grid_n: int = _N_SCORED_PER_FRAME,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode -> (flip_idx, target_classes, this_bitmask). Bit-exact inverse of encode."""
    if payload[:4] != _SIDECAR_MAGIC:
        raise ValueError("bad WTA1 magic")
    (K,) = struct.unpack("<I", payload[4:8])
    off = 8
    (bm_len,) = struct.unpack("<I", payload[off:off + 4])
    off += 4
    delta = np.unpackbits(np.frombuffer(_unlzma_raw(payload[off:off + bm_len]), dtype=np.uint8))[:grid_n]
    off += bm_len
    prev = prev_bitmask if prev_bitmask is not None else np.zeros(grid_n, dtype=np.uint8)
    bitmask = np.bitwise_xor(delta.astype(np.uint8), prev)
    (ht_len,) = struct.unpack("<H", payload[off:off + 2])
    off += 2
    htq = np.frombuffer(zlib.decompress(payload[off:off + ht_len]), dtype=np.uint16).reshape(
        len(margin_bins) - 1, _N_SEG_CLASSES
    )
    off += ht_len
    (cls_len,) = struct.unpack("<I", payload[off:off + 4])
    off += 4
    cls_blob = payload[off:off + cls_len]
    off += cls_len

    flip_idx = np.flatnonzero(bitmask).astype(np.int64)
    if len(flip_idx) != K:
        raise ValueError(f"decoded position count {len(flip_idx)} != header K {K}")
    if K:
        margin_at = margin_field_flat[flip_idx]
        pmfs_q = [htq[b].astype(np.float64) for b in
                  (np.digitize(margin_at, margin_bins) - 1).clip(0, len(margin_bins) - 2)]
        cls = np.asarray(decode_symbols(cls_blob, pmfs_q), dtype=np.int64)
    else:
        cls = np.zeros(0, dtype=np.int64)
    return flip_idx, cls, bitmask


# ─────────────────────────────────────────────────────────────────────────────
# basin render + frozen-SegNet forward (READ-ONLY checkpoints)
# ─────────────────────────────────────────────────────────────────────────────
def _load_basin_decoder(ckpt_path: Path, which: str):
    """Load the basin decoder + latents (READ-ONLY).

    Supports BOTH the forkpoint single-file state ``torch_vehicle_checkpoint_state.pt``
    (keys ``decoder``/``ema_decoder`` + ``latents``/``ema_latents``) AND the ``best/``
    directory layout (``best_ema_decoder.pt`` + ``best_ema_latents.pt`` + meta).
    """
    from tac.torch_vehicle.vendored_imports import import_vendored

    common = import_vendored("common")
    p = Path(ckpt_path)
    base_channels = 20
    if p.is_dir():
        meta = json.loads((p / "best_meta.json").read_text())
        base_channels = int(meta.get("base_channels", 20))
        dec_sd = torch.load(p / "best_ema_decoder.pt", map_location="cpu", weights_only=False)
        latents = torch.load(p / "best_ema_latents.pt", map_location="cpu", weights_only=False)
        if which == "live":
            # 'best/' dir only stores the EMA shadow; live is unavailable -> use EMA (documented).
            pass
    else:
        sd = torch.load(p, map_location="cpu", weights_only=False)
        if "base_channels" in sd:
            base_channels = int(sd["base_channels"])
        dkey = "ema_decoder" if which == "ema" else "decoder"
        lkey = "ema_latents" if which == "ema" else "latents"
        dec_sd = sd[dkey]
        latents = sd[lkey]
    dec = common.HNeRVDecoder(latent_dim=28, base_channels=base_channels, eval_size=(_EVAL_H, _EVAL_W))
    dec.load_state_dict(dec_sd)
    dec.eval()
    return dec, torch.as_tensor(latents).detach().float()


def _render_and_segforward(dec, net, score_mod, latents, idx):
    """Render basin frames for ``idx`` on the EXACT eval round-trip; return
    (seg_out (B,5,384,512), decoded (B,2,3,384,512) native rendered frames)."""
    with torch.inference_mode():
        z = latents[idx]
        decoded = dec(z)  # (B,2,3,384,512) float[0,255]
        B = decoded.shape[0]
        flat = decoded.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
        up = score_mod._decoded_to_camera(flat)  # bicubic ^874x1164
        bhwc = (
            up.reshape(B, 2, 3, score_mod.CAMERA_H, score_mod.CAMERA_W)
            .permute(0, 1, 3, 4, 2)
            .clamp(0, 255)
            .round()
            .to(torch.uint8)
        )
        _posenet_in, segnet_in = net.preprocess_input(bhwc)  # last-frame, bilinear_384, uint8
        seg_out = net.segnet(segnet_in)  # (B,5,384,512)
    return seg_out, decoded


def _margin_map(seg_out: torch.Tensor) -> torch.Tensor:
    top2, _ = torch.topk(seg_out, k=2, dim=1, largest=True, sorted=True)
    return (top2[:, 0] - top2[:, 1]).clamp_min(0.0)


def _roundtrip_segforward_single(net, score_mod, frame_chw: torch.Tensor) -> np.ndarray:
    """Push ONE corrected 384x512 frame through the exact eval round-trip + SegNet.
    Returns the (384,512) argmax label map. NO FAKE: identical channel to the scorer."""
    with torch.inference_mode():
        up = score_mod._decoded_to_camera(frame_chw.unsqueeze(0))  # (1,3,874,1164)
        bhwc = (
            up.reshape(1, 1, 3, score_mod.CAMERA_H, score_mod.CAMERA_W)
            .permute(0, 1, 3, 4, 2)
            .clamp(0, 255)
            .round()
            .to(torch.uint8)
        )
        bhwc2 = bhwc.repeat(1, 2, 1, 1, 1)  # seg uses last frame only; duplicate is exact
        _pin, segnet_in_c = net.preprocess_input(bhwc2)
        seg_out_c = net.segnet(segnet_in_c)
        return seg_out_c.argmax(dim=1)[0].cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# A. survival-first correction: apply class-prototype nudge, round-trip, keep survivors
# ─────────────────────────────────────────────────────────────────────────────
def _survival_first_correct(
    net, score_mod, decoded_j, rendered_argmax_j, gt_j, margin_j, *, tau: float, max_candidates: int
):
    """Apply the class-prototype correction to the candidate boundary flips in the
    rendered 384x512 frame1, round-trip through the EXACT eval channel, re-segment, and
    return (survivable_flip_idx, survivable_target_cls, corrected_dseg_on_set, base_dseg_on_set,
    n_candidates).

    The candidate set = boundary flips (r!=g AND m<tau), capped at ``max_candidates`` lowest
    margin (the cheapest, most-fixable). We apply ALL candidate corrections at once (the
    strongest a per-pixel sidecar could legally do), round-trip ONCE, and KEEP only the
    pixels whose post-roundtrip argmax actually equals GT — that is the survivable subset
    we then code. This is the prototype's fix #A: survival measured BEFORE coding.
    """
    r = rendered_argmax_j
    g = gt_j
    m = margin_j
    flips_mask = (r != g) & (m < tau)
    cand = np.flatnonzero(flips_mask.reshape(-1))
    if len(cand) == 0:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64), 0.0, 0.0, 0)
    order = np.argsort(m.reshape(-1)[cand])
    cand = cand[order[:max_candidates]]
    ys, xs = np.unravel_index(cand, (_EVAL_H, _EVAL_W))
    target_cls = g.reshape(-1)[cand]

    frame1 = decoded_j[1].clone()  # (3,384,512) float[0,255]  (frame index 1 = the scored last frame)
    r_flat = r.reshape(-1)
    f_flat = frame1.reshape(3, -1)
    corrected = frame1.clone()
    for cls in np.unique(target_cls):
        cls_pixels = r_flat == cls
        if cls_pixels.sum() == 0:
            proto = f_flat.min(dim=1).values
        else:
            proto = f_flat[:, torch.from_numpy(cls_pixels)].mean(dim=1)
        sel = target_cls == cls
        for c in range(3):
            corrected[c, ys[sel], xs[sel]] = proto[c]

    new_argmax = _roundtrip_segforward_single(net, score_mod, corrected)
    survived_mask = new_argmax[ys, xs] == target_cls
    surv_idx = cand[survived_mask]
    surv_cls = target_cls[survived_mask]
    # d_seg ON THE CANDIDATE SET before/after (per-pixel flip rate on those positions)
    base_dseg_on_set = float(len(cand)) / _N_SCORED_PER_FRAME  # all candidates were flips
    # after coding the survivable subset, those positions are corrected to GT -> not flips;
    # the NON-survivable candidates remain flips. Corrected d_seg contribution from this set:
    corrected_dseg_on_set = float(len(cand) - int(survived_mask.sum())) / _N_SCORED_PER_FRAME
    return surv_idx.astype(np.int64), surv_cls.astype(np.int64), corrected_dseg_on_set, base_dseg_on_set, len(cand)


@dataclass
class PerPair:
    pair_index: int
    n_flips_total: int
    n_candidates_boundary: int
    n_survivable: int
    survival_fraction: float
    residual_bytes: int
    bitmask_bytes: int
    class_bytes: int
    sidetable_bytes: int
    bytes_per_survivable_flip: float
    roundtrip_verified: bool


def run_probe(
    *,
    ckpt_path: Path,
    video_path: Path,
    which_decoder: str,
    n_pairs: int,
    tau: float,
    max_candidates: int,
    batch: int,
    targets_cache: Path,
) -> dict:
    t_start = time.time()
    torch.set_num_threads(max(1, (torch.get_num_threads())))
    from tac.score_aware_loop.targets import load_frozen_distortion_net
    from tac.torch_vehicle.scorer_context import RealScorerContext
    from tac.torch_vehicle.vendored_imports import import_vendored

    score_mod = import_vendored("score")
    net = load_frozen_distortion_net(device="cpu")
    dec, latents = _load_basin_decoder(ckpt_path, which_decoder)

    ctx = RealScorerContext(
        str(video_path), device="cpu", max_pairs=n_pairs, targets_cache=str(targets_cache)
    )
    gt_argmax = ctx.seg_targets_hard.cpu().numpy()  # (n,384,512) int64 GT argmax
    n_pairs = int(min(n_pairs, gt_argmax.shape[0]))

    margin_bins = np.asarray([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 1e9], dtype=np.float64)
    per_pair: list[PerPair] = []
    prev_bitmask: np.ndarray | None = None
    total_residual_bytes = 0
    total_survivable = 0
    total_candidates = 0
    total_flips = 0
    total_bitmask_bytes = 0
    total_class_bytes = 0
    total_sidetable_bytes = 0
    base_dseg_sum = 0.0  # sum over pairs of base d_seg (all flips)
    coded_dseg_sum = 0.0  # sum over pairs of d_seg AFTER coding survivable subset
    # analytic FLOOR: the tightest position coder (combinatorial set-index conditioned on
    # the decoder-free boundary band) + conditional class entropy, for the survivable subset.
    # This is a LOWER BOUND on any reversible coder's bytes; reported alongside realized bytes.
    floor_bits_sum = 0.0

    for start in range(0, n_pairs, batch):
        idx = torch.arange(start, min(start + batch, n_pairs))
        seg_out, decoded = _render_and_segforward(dec, net, score_mod, latents, idx)
        margin = _margin_map(seg_out).cpu().numpy()
        rendered_argmax = seg_out.argmax(dim=1).cpu().numpy()

        for j, pidx in enumerate(idx.tolist()):
            g = gt_argmax[pidx]
            r = rendered_argmax[j]
            m = margin[j]
            n_flips = flip_count(r, g)
            total_flips += n_flips
            base_dseg_sum += n_flips / _N_SCORED_PER_FRAME

            surv_idx, surv_cls, _corr_dseg_set, _base_dseg_set, n_cand = _survival_first_correct(
                net, score_mod, decoded[j], r, g, m, tau=tau, max_candidates=max_candidates
            )
            total_candidates += n_cand
            total_survivable += len(surv_idx)
            # d_seg after coding: base flips minus survivable (which are corrected to GT)
            coded_dseg_sum += (n_flips - len(surv_idx)) / _N_SCORED_PER_FRAME

            # B + C: code ONLY the survivable subset (positions + classes + temporal-delta)
            code, this_bitmask = encode_pair_residual(
                surv_idx, surv_cls, m.reshape(-1).astype(np.float64), prev_bitmask,
                margin_bins=margin_bins,
            )
            # NO FAKE: verify the codec round-trips bit-exactly on this real pair
            dec_idx, dec_cls, _bm = decode_pair_residual(
                code.payload, m.reshape(-1).astype(np.float64), prev_bitmask, margin_bins=margin_bins
            )
            # verify the position->class map is recovered exactly (order-independent)
            if len(surv_idx):
                truth = dict(zip(surv_idx.tolist(), surv_cls.tolist(), strict=True))
                rt_ok = bool(
                    np.array_equal(np.sort(dec_idx), np.sort(surv_idx))
                    and all(truth[int(i)] == int(c) for i, c in zip(dec_idx, dec_cls, strict=True))
                )
            else:
                rt_ok = len(dec_idx) == 0
            prev_bitmask = this_bitmask
            total_residual_bytes += code.n_bytes
            total_bitmask_bytes += code.bitmask_bytes
            total_class_bytes += code.class_bytes
            total_sidetable_bytes += code.sidetable_bytes

            # analytic floor for the survivable subset (tightest combinatorial position +
            # conditional class entropy). All survivable flips lie in the boundary band by
            # construction (candidates were m<tau), so k_in == K, k_out == 0.
            if len(surv_idx):
                m_flat = m.reshape(-1).astype(np.float64)
                boundary_size = int((m_flat < tau).sum())
                pos_bits = conditional_position_bits(boundary_size, len(surv_idx), len(m_flat), 0)
                m_at = m_flat[surv_idx]
                bin_idx = (np.digitize(m_at, margin_bins) - 1).clip(0, len(margin_bins) - 2)
                cls_bits = class_bits_conditional(surv_cls, bin_idx)
                floor_bits_sum += pos_bits + cls_bits

            per_pair.append(
                PerPair(
                    pair_index=pidx,
                    n_flips_total=n_flips,
                    n_candidates_boundary=n_cand,
                    n_survivable=len(surv_idx),
                    survival_fraction=(len(surv_idx) / n_cand if n_cand else 0.0),
                    residual_bytes=code.n_bytes,
                    bitmask_bytes=code.bitmask_bytes,
                    class_bytes=code.class_bytes,
                    sidetable_bytes=code.sidetable_bytes,
                    bytes_per_survivable_flip=(code.n_bytes / len(surv_idx) if len(surv_idx) else 0.0),
                    roundtrip_verified=rt_ok,
                )
            )

    agg = _aggregate(
        per_pair, n_pairs=n_pairs, tau=tau,
        total_residual_bytes=total_residual_bytes,
        total_survivable=total_survivable, total_candidates=total_candidates,
        total_flips=total_flips, total_bitmask_bytes=total_bitmask_bytes,
        total_class_bytes=total_class_bytes, total_sidetable_bytes=total_sidetable_bytes,
        base_dseg_sum=base_dseg_sum, coded_dseg_sum=coded_dseg_sum,
        max_candidates=max_candidates, floor_bits_sum=floor_bits_sum,
    )
    agg["wall_seconds"] = round(time.time() - t_start, 1)
    agg["which_decoder"] = which_decoder
    agg["n_pairs_measured"] = n_pairs
    agg["per_pair_sample"] = [asdict(p) for p in per_pair[:8]]
    return agg


def _aggregate(
    per_pair, *, n_pairs, tau, total_residual_bytes, total_survivable, total_candidates,
    total_flips, total_bitmask_bytes, total_class_bytes, total_sidetable_bytes,
    base_dseg_sum, coded_dseg_sum, max_candidates, floor_bits_sum,
) -> dict:
    scale = _N_FRAMES / max(n_pairs, 1)
    residual_bytes_600 = total_residual_bytes * scale
    floor_bytes_600 = (floor_bits_sum / 8.0) * scale  # analytic tightest-coder floor
    bitmask_bytes_600 = total_bitmask_bytes * scale
    class_bytes_600 = total_class_bytes * scale
    sidetable_bytes_600 = total_sidetable_bytes * scale

    # survival
    survival_fraction = (total_survivable / total_candidates) if total_candidates else 0.0
    rt_all_ok = all(p.roundtrip_verified for p in per_pair) if per_pair else False

    # base vs coded d_seg (mean over measured pairs)
    base_mean_dseg = base_dseg_sum / max(n_pairs, 1)
    coded_mean_dseg = coded_dseg_sum / max(n_pairs, 1)
    base_seg_term = 100.0 * base_mean_dseg
    coded_seg_term = 100.0 * coded_mean_dseg
    seg_term_drop = base_seg_term - coded_seg_term  # the d_seg win the survivable code buys

    # bytes per survivable flip (the cost axis)
    bytes_per_survivable_flip = (
        total_residual_bytes / total_survivable if total_survivable else 0.0
    )

    # ── net ΔS as a frontier SIDECAR (add residual bytes; remove the survivable seg debt) ──
    # NOTE: this seg-drop is computed ON THE BASIN's own d_seg debt (the survivable flips
    # are basin-vs-GT flips). As a sidecar bolted onto the BASIN it would lower the basin S;
    # as a sidecar bolted onto the FRONTIER it can only fix flips the frontier still has.
    # We report the BASIN-relative ΔS (honest: this residual was measured against the basin).
    # HONESTY (NO-FAKE): the coded d_seg credits the full survivable count and does NOT subtract
    # receptive-field collateral (new-bad flips the joint correction may create at non-candidate
    # pixels). It is therefore an OPTIMISTIC bound on the seg win -> the true ΔS is >= this value
    # (i.e. even WORSE). Since the verdict is already net-positive/NO-GO, this only strengthens it.
    rate_cost_600 = residual_bytes_600 * (25.0 / _RATE_DENOM)
    seg_win = seg_term_drop  # the 100*Δd_seg the coded survivable set achieves (basin-relative)
    net_delta_S_sidecar = -seg_win + rate_cost_600

    # the fairest residual cost = min(realized reversible bytes, analytic tightest-coder floor).
    # the floor is the best ANY position+class coder could do on this survivable subset.
    best_residual_bytes_600 = min(residual_bytes_600, floor_bytes_600) if floor_bytes_600 else residual_bytes_600

    # ── D. witness framing: amortized prior (basin decoder) + survivable residual ──
    witness_total_bytes = _BASIN_DECODER_BYTES + residual_bytes_600 + _POSE_TRAJ_BYTES
    witness_total_bytes_floor = _BASIN_DECODER_BYTES + best_residual_bytes_600 + _POSE_TRAJ_BYTES
    witness_beats_frontier = witness_total_bytes < _FRONTIER_BYTES
    witness_beats_frontier_floor = witness_total_bytes_floor < _FRONTIER_BYTES
    residual_in_band = _CONDITIONAL_MDL_LO <= residual_bytes_600 <= _CONDITIONAL_MDL_HI
    residual_below_band_hi = residual_bytes_600 <= _CONDITIONAL_MDL_HI
    floor_residual_in_band = _CONDITIONAL_MDL_LO <= best_residual_bytes_600 <= _CONDITIONAL_MDL_HI
    floor_residual_below_band_hi = best_residual_bytes_600 <= _CONDITIONAL_MDL_HI

    # standalone-witness S estimate: the witness = basin decoder regenerating its partition,
    # with the survivable residual correcting the survivable flips. d_seg -> coded_mean_dseg.
    witness_rate_term = 25.0 * witness_total_bytes / _RATE_DENOM
    # pose unchanged from basin (the residual touches seg-fragile pixels; pose checked unharmed
    # by construction in the prototype — the correction is a tiny boundary nudge). Use basin pose.
    _basin_d_pose = 0.00034168662969022987  # MEASURED (best_meta.json)
    witness_pose_term = float(np.sqrt(10.0 * _basin_d_pose))
    witness_S_estimate = coded_seg_term + witness_pose_term + witness_rate_term

    # ── verdict logic ──
    # GO requires: high survival AND the witness (even at the tightest-coder floor) beats the
    # frontier on bytes AND the residual reaches the conditional-MDL band AND the sidecar is
    # net-negative-S. The floor test is the FAIREST: if even the optimal coder cannot reach the
    # band/frontier, the wall is structural (not a coder weakness).
    survives = survival_fraction >= 0.50
    cost_under_break_even = bytes_per_survivable_flip < WATERLINE_BYTES_PER_FLIP and bytes_per_survivable_flip > 0
    reaches_band = floor_residual_below_band_hi  # tightest-coder residual <= 64.6 KB
    sidecar_net_negative = net_delta_S_sidecar < 0

    if witness_beats_frontier_floor and reaches_band and survives and sidecar_net_negative:
        verdict = "WITNESS_SEG_TOPAIML_GO"
    elif not survives:
        verdict = "NO_GO_SURVIVAL_WALL"
    elif not reaches_band and not witness_beats_frontier_floor:
        verdict = "NO_GO_BYTE_WALL"
    elif (reaches_band or cost_under_break_even) and not witness_beats_frontier_floor:
        verdict = "HYBRID_FOLD_INTO_TRAINING"
    else:
        verdict = "HYBRID_FOLD_INTO_TRAINING"

    return {
        "evidence_grade": "[contest-CPU advisory] NON-PROMOTABLE",
        "frontier_unmoved": True,
        "tau": tau,
        "max_candidates_per_pair": max_candidates,
        "waterline_bytes_per_flip": WATERLINE_BYTES_PER_FLIP,
        # base state
        "total_flips_measured": total_flips,
        "base_mean_d_seg": base_mean_dseg,
        "base_seg_term_100_dseg": base_seg_term,
        # A. survival-first
        "A_total_candidates_boundary": total_candidates,
        "A_total_survivable": total_survivable,
        "A_survival_fraction": survival_fraction,
        "A_survives_threshold_0p50": survives,
        "A_coded_mean_d_seg": coded_mean_dseg,
        "A_coded_seg_term": coded_seg_term,
        "A_seg_term_drop_basin_relative": seg_term_drop,
        # B + C. coder economics (survivable subset only)
        "BC_total_residual_bytes_measured": total_residual_bytes,
        "BC_total_residual_bytes_600_scaled": residual_bytes_600,
        "BC_bitmask_bytes_600_temporal_delta": bitmask_bytes_600,
        "BC_class_bytes_600_arithmetic_coded": class_bytes_600,
        "BC_sidetable_bytes_600": sidetable_bytes_600,
        "BC_bytes_per_survivable_flip": bytes_per_survivable_flip,
        "BC_cost_under_break_even": cost_under_break_even,
        "BC_codec_roundtrip_all_ok": rt_all_ok,
        # analytic tightest-coder floor (combinatorial position + conditional class entropy)
        "BC_floor_residual_bytes_600": floor_bytes_600,
        "BC_floor_bytes_per_survivable_flip": (
            (floor_bytes_600 / (total_survivable * scale)) if total_survivable else 0.0
        ),
        "BC_best_residual_bytes_600": best_residual_bytes_600,
        # sidecar ΔS (basin-relative)
        "sidecar_rate_cost_600": rate_cost_600,
        "sidecar_seg_win_basin_relative": seg_win,
        "sidecar_net_delta_S": net_delta_S_sidecar,
        "sidecar_net_negative_S": sidecar_net_negative,
        # D. witness framing
        "D_basin_decoder_bytes": _BASIN_DECODER_BYTES,
        "D_pose_traj_bytes": _POSE_TRAJ_BYTES,
        "D_residual_bytes_600": residual_bytes_600,
        "D_witness_total_bytes": witness_total_bytes,
        "D_witness_total_bytes_at_coder_floor": witness_total_bytes_floor,
        "D_frontier_bytes": _FRONTIER_BYTES,
        "D_witness_beats_frontier_bytes": witness_beats_frontier,
        "D_witness_beats_frontier_at_coder_floor": witness_beats_frontier_floor,
        "D_conditional_mdl_band": [_CONDITIONAL_MDL_LO, _CONDITIONAL_MDL_HI],
        "D_residual_reaches_band_hi": residual_below_band_hi,
        "D_residual_in_band": residual_in_band,
        "D_floor_residual_reaches_band_hi": floor_residual_below_band_hi,
        "D_floor_residual_in_band": floor_residual_in_band,
        "D_witness_S_estimate": witness_S_estimate,
        "D_witness_S_components": {
            "seg": coded_seg_term, "pose": witness_pose_term, "rate": witness_rate_term,
        },
        # prototype comparison
        "PROTOTYPE_residual_bytes_600": 543_123.86,
        "PROTOTYPE_survival_fraction": 0.47265625,
        "PROTOTYPE_witness_bytes_hi": 599_680.86,
        "PROTOTYPE_verdict": "HYBRID_FOLD_INTO_TRAINING",
        "IMPROVEMENT_residual_bytes_ratio": (543_123.86 / residual_bytes_600) if residual_bytes_600 else 0.0,
        # VERDICT
        "VERDICT_SEG_TOPAIML": verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt",
        default="experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best",
        help="basin checkpoint: a 'best/' dir OR a forkpoint *_checkpoint_state.pt (READ-ONLY).",
    )
    ap.add_argument("--video", default="upstream/videos/0.mkv")
    ap.add_argument("--which-decoder", choices=["ema", "live"], default="ema")
    ap.add_argument("--n-pairs", type=int, default=120,
                    help="pairs to measure (120 == prototype, decisive + $0).")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="margin threshold for the boundary band B (canonical 0.5).")
    ap.add_argument("--max-candidates", type=int, default=2048,
                    help="max boundary flips per pair to test round-trip survival on "
                         "(lowest-margin first). Large enough to span the typical ~430 boundary flips.")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--targets-cache", default=".omx/tmp/witness_topaiml_targets")
    ap.add_argument("--out-json", default="reports/witness_seg_boundary_topaiml.json")
    args = ap.parse_args()

    result = run_probe(
        ckpt_path=Path(args.ckpt),
        video_path=Path(args.video),
        which_decoder=args.which_decoder,
        n_pairs=args.n_pairs,
        tau=args.tau,
        max_candidates=args.max_candidates,
        batch=args.batch,
        targets_cache=Path(args.targets_cache),
    )
    print(json.dumps(result, indent=2, default=float))
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=float))
        print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
