# SPDX-License-Identifier: MIT
"""Decompose the d_seg argmax-flip set into R_cap (capacity-fixable) vs R_surv
(round-trip-aliased) — the ROUTING-LEVERAGE CEILING probe.

THE QUESTION (sub-0.15 DAG, FEED 2026-06-25k). The trained torch_vehicle decoder
loses d_seg on a set of GT pixels where its SegNet argmax disagrees with the GT
argmax. That flip set is NOT homogeneous. A capacity-routing lever (give the
hard regions more decoder capacity / finer quant grid) can only repair a flip if
the decoder GENUINELY FAILED to represent the edge — i.e. the argmax is already
wrong on the RAW (pre-round-trip, sharp float) frame. If instead the RAW-frame
argmax is CORRECT and only the eval ROUND-TRIP (bicubic↑ → uint8 → bilinear↓)
aliases the edge into a flip, then NO amount of decoder capacity fixes it — the
information was destroyed by R, not by the decoder. Only a round-trip-in-loop
objective or a sub-pixel representation touches that set.

So we partition the GT-flip set into two disjoint sets (per pixel, per pair):

  * **R_cap**  = GT-flips where RAW-frame argmax is ALSO wrong → the decoder did
                 not represent the edge → capacity-routing CAN fix it.
  * **R_surv** = GT-flips where RAW-frame argmax is CORRECT but the
                 ROUND-TRIPPED-frame argmax flips → R aliased a correctly-placed
                 edge → capacity-routing CANNOT fix it (only round-trip-in-loop /
                 sub-pixel).

and report the **routing-leverage ceiling = R_cap / R_total**: the maximum
fraction of the current d_seg debt that any pure capacity-routing lever could,
in the best case, recover. (It is a CEILING, not an estimate of realized gain.)

DEFINITIONS (the exact set algebra — the NOVEL logic, unit-tested below):
  Per pixel p (over the SegNet 384x512 grid, last-frame parity 1, per pair):
    gt   = GT argmax (cached ``seg_targets_hard``)
    raw  = decoder-output argmax (NO round-trip; sharp float 384x512)
    rt   = round-tripped-frame argmax (the eval-roundtrip R, the d_seg the
           scorer actually measures)
  A pixel is a GT-FLIP iff ``rt != gt`` (this IS the scorer's d_seg per-pixel).
  Within the GT-flip set:
    R_cap  iff ``raw != gt``   (raw also wrong → capacity-fixable)
    R_surv iff ``raw == gt``   (raw correct, rt wrong → round-trip-aliased)
  R_total = R_cap + R_surv = #{rt != gt} = the scorer's flip count.
  Note: pixels where ``raw != gt`` but ``rt == gt`` (the round-trip ACCIDENTALLY
  repaired a raw error) are NOT in any of these sets — they are not d_seg debt.
  leverage = R_cap / R_total  (0.0 when R_total == 0; reported as such).

THE ROUND-TRIP R (bound 1:1 to the production training loop — NOT reimplemented
from a memo): ``tac.torch_vehicle.driver`` lines 2051-2058 apply, per pair frames
flattened to (B*2, 3, 384, 512):
    up   = F.interpolate(flat, size=(874, 1164), mode="bicubic",  align_corners=False)
    down = F.interpolate(up,   size=(384, 512),  mode="bilinear", align_corners=False)
    clamp(0,255) then round() (uint8 snap; STE in training, plain round here).
CAMERA_H,W = 874,1164 (``distortion_finishing_kit.CAMERA_H/CAMERA_W`` line 81).
The SegNet input stays 384x512 — R changes PIXEL VALUES, not the resolution into
SegNet. This module reuses that EXACT op sequence (``_apply_eval_roundtrip``) so
the rt-frame is bit-faithful to what the scorer measures; the raw-frame skips it.

THE SCORER FORWARD (bound to production, NOT a re-derivation): the seg LOGITS
come from ``DistortionNet.preprocess_input`` → ``DistortionNet.segnet`` exactly as
``tac.torch_vehicle.scorer_context.RealScorerContext.seg_pose_forward`` (lines
191-200) does, but run on the AUTHORITY net (``RealScorerContext.distortion_net``,
CPU-TRUSTED — NEVER the train/MPS net, NEVER MPS argmax). GT = the cached
``seg_targets_hard`` (n_pairs, 384, 512) int64.

AUTHORITY / SCOPE: this is a DIAGNOSTIC, not a score actuator. It reports a
CEILING on routing leverage; it does not move the frontier by itself. The
real-data numbers it would produce are ``[contest-CPU advisory] NON-PROMOTABLE``
diagnostics (CPU-authority argmax, real scorer). The d_seg arithmetic routes
through ``tac.contest_score`` (never a hand-rolled evaluate.py). GT decode is
the canonical ``frame_utils.yuv420_to_rgb`` path inside the vendored
``precompute_targets`` (RealScorerContext) — PyAV rgb24 is FORBIDDEN.

  !!!  THE REAL-DATA PATH (``--ckpt``) IS WRITTEN BUT *UNTESTED*.  !!!
It loads a checkpoint, runs the real frozen scorer on real frames, and forwards a
real decoder — all FORBIDDEN while the live MPS training daemons hold the slots
(it would contend / run MPS / forward a real checkpoint). It is fully py_compiled
and bound to the real APIs, but it is RUN LATER, at convergence, when a slot
frees. Only ``--self-test`` (pure numpy/torch, $0, non-contending) is run now.

Hooks per Catalog #125 6-hook wire-in declaration:
  * #1 sensitivity-map = ACTIVE (the per-pixel R_cap mask IS a capacity-routing
    sensitivity prior — exactly the pixels a routing lever should target).
  * #2 Pareto constraint = ACTIVE (the leverage CEILING bounds the achievable
    d_seg move of any pure routing lever → a hard Pareto bound on that family).
  * #3 bit-allocator = N/A (diagnostic; allocates no bits — it bounds what an
    allocator COULD recover).
  * #4 cathedral autopilot = N/A (an offline diagnostic, not a dispatch surface).
  * #5 continual-learning = ACTIVE (R_cap/R_surv split + ceiling reseed the judge
    on whether capacity-routing OR round-trip-in-loop is the right d_seg lever).
  * #6 probe-disambiguator = ACTIVE (this IS the disambiguator between the
    "capacity-routing fixes d_seg" and "the round-trip destroyed the edge"
    interpretations of the d_seg plateau).

Cross-references:
  * ``tac.torch_vehicle.scorer_context.RealScorerContext`` (seg_pose_forward
    191-200; precompute_targets 101-108; distortion_net 94) — the bound scorer.
  * ``tac.torch_vehicle.driver`` (eval-roundtrip 2051-2058) — the bound R.
  * ``tac.torch_vehicle.distortion_finishing_kit`` (CAMERA_H/CAMERA_W 81).
  * ``tac.torch_vehicle.checkpoint`` (read_manifest 300; load_checkpoint 305) —
    the bound checkpoint loader.
  * ``tac.contest_score`` (compute_contest_score 163; seg_term 111) — score math.
  * ``.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md``
    (FEED 2026-06-25k — the spec this probe realizes).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass

import numpy as np

# The SegNet grid the d_seg argmax lives on (last-frame parity-1, per pair).
SEG_H, SEG_W = 384, 512
# The eval round-trip's camera resolution (bound to distortion_finishing_kit:81).
CAMERA_H, CAMERA_W = 874, 1164


# ---------------------------------------------------------------------------
# THE NOVEL LOGIC: the R_cap / R_surv set decomposition (pure, no torch/scorer).
# Unit-tested below on hand-constructed synthetic arrays. This is where bugs hide.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FlipDecomposition:
    """The R_cap / R_surv decomposition of one (or many) (H,W) argmax map(s).

    ``r_total`` is the scorer's per-pixel d_seg flip count (``rt != gt``);
    ``r_cap`` + ``r_surv`` partition it; ``leverage`` = r_cap / r_total (the
    routing-leverage CEILING, 0.0 when r_total == 0). ``n_pixels`` is the total
    pixel count over which the decomposition was computed (for d_seg = r_total /
    n_pixels). ``r_roundtrip_repaired`` counts pixels where the round-trip
    ACCIDENTALLY fixed a raw error (raw != gt but rt == gt) — NOT d_seg debt,
    reported for completeness (a sanity cross-check, not part of the partition).
    """

    r_cap: int
    r_surv: int
    r_total: int
    n_pixels: int
    r_roundtrip_repaired: int
    leverage: float

    @property
    def d_seg(self) -> float:
        """The scorer's d_seg (flip rate) implied by this decomposition."""
        return self.r_total / self.n_pixels if self.n_pixels else 0.0


def classify_flips(
    gt: np.ndarray, raw: np.ndarray, rt: np.ndarray
) -> FlipDecomposition:
    """Partition the GT-flip set of ``(gt, raw, rt)`` argmax maps into R_cap/R_surv.

    Args:
        gt:  GT argmax (cached ``seg_targets_hard``). Any shape; int-like.
        raw: decoder-output argmax with NO round-trip (sharp float frame). Same
             shape as ``gt``.
        rt:  round-tripped-frame argmax (the eval-roundtrip R; the d_seg the
             scorer measures). Same shape as ``gt``.

    Returns:
        :class:`FlipDecomposition`. Per pixel, with ``flip := rt != gt``:
            R_cap  := flip AND (raw != gt)   (raw also wrong → capacity-fixable)
            R_surv := flip AND (raw == gt)   (raw correct, rt aliased → not fixable)
        and ``leverage = R_cap / R_total`` (0.0 when R_total == 0). Pixels with
        ``raw != gt`` but ``rt == gt`` (round-trip accidentally repaired) are
        counted in ``r_roundtrip_repaired`` and are NOT d_seg debt.

    Raises:
        ValueError: on shape mismatch (fail-closed; no silent broadcast).
    """
    g = np.asarray(gt)
    r = np.asarray(raw)
    t = np.asarray(rt)
    if not (g.shape == r.shape == t.shape):
        raise ValueError(
            f"gt/raw/rt must share a shape; got {g.shape} / {r.shape} / {t.shape}"
        )
    n_pixels = int(g.size)
    flip = t != g  # the scorer's per-pixel d_seg set (rt disagrees with GT)
    raw_wrong = r != g  # the decoder failed to represent the edge on the raw frame
    r_cap = int(np.count_nonzero(flip & raw_wrong))
    r_surv = int(np.count_nonzero(flip & ~raw_wrong))
    r_total = int(np.count_nonzero(flip))
    # Sanity cross-check (not part of the partition): round-trip repaired a raw error.
    r_roundtrip_repaired = int(np.count_nonzero(~flip & raw_wrong))
    assert r_cap + r_surv == r_total, (
        "partition invariant violated: r_cap + r_surv != r_total "
        f"({r_cap} + {r_surv} != {r_total})"
    )
    leverage = (r_cap / r_total) if r_total > 0 else 0.0
    return FlipDecomposition(
        r_cap=r_cap,
        r_surv=r_surv,
        r_total=r_total,
        n_pixels=n_pixels,
        r_roundtrip_repaired=r_roundtrip_repaired,
        leverage=leverage,
    )


# ---------------------------------------------------------------------------
# Synthetic self-test of the NOVEL logic (pure numpy — $0, non-contending; RUN now).
# ---------------------------------------------------------------------------
def _run_self_test() -> bool:
    """Run the synthetic ``classify_flips`` test over all per-pixel cases.

    Constructs a tiny array where each pixel is hand-placed into exactly one of
    the five (gt, raw, rt) categories, then asserts the decomposition counts
    match the construction. Returns True on PASS, raises AssertionError on FAIL.

    The five per-pixel categories (the case coverage the prompt requires):
        (A) no flip, raw correct          : gt==raw==rt           -> nothing
        (B) R_cap flip                    : raw!=gt, rt!=gt        -> r_cap
        (C) R_surv flip                   : raw==gt, rt!=gt        -> r_surv
        (D) both-wrong-SAME wrong class   : raw!=gt, rt!=gt, raw==rt (a sub-case
            of R_cap — flip AND raw wrong) -> r_cap
        (E) both-wrong-DIFFERENT class    : raw!=gt, rt!=gt, raw!=rt (also R_cap)
            -> r_cap
        (F) round-trip repaired           : raw!=gt, rt==gt        -> repaired only
    """
    cases: list[tuple[str, int, int, int, str]] = [
        # name,                       gt, raw, rt, expected-bucket
        ("A_no_flip_raw_correct",      1,  1,  1, "none"),
        ("B_r_cap_flip",               1,  2,  3, "r_cap"),
        ("C_r_surv_flip",              1,  1,  4, "r_surv"),
        ("D_both_wrong_same",          2,  3,  3, "r_cap"),
        ("E_both_wrong_different",     2,  3,  4, "r_cap"),
        ("F_roundtrip_repaired",       0,  4,  0, "repaired"),
        # duplicate a couple to make the counts non-trivial (and order-independent):
        ("C2_r_surv_flip",            2,  2,  0, "r_surv"),
        ("B2_r_cap_flip",             4,  0,  1, "r_cap"),
        ("A2_no_flip_raw_correct",    3,  3,  3, "none"),
    ]
    gt = np.array([c[1] for c in cases], dtype=np.int64)
    raw = np.array([c[2] for c in cases], dtype=np.int64)
    rt = np.array([c[3] for c in cases], dtype=np.int64)
    buckets = [c[4] for c in cases]

    exp_r_cap = buckets.count("r_cap")
    exp_r_surv = buckets.count("r_surv")
    exp_repaired = buckets.count("repaired")
    exp_total = exp_r_cap + exp_r_surv  # r_total = flips = r_cap + r_surv
    exp_n = len(cases)
    exp_leverage = exp_r_cap / exp_total if exp_total else 0.0

    dec = classify_flips(gt, raw, rt)

    failures: list[str] = []

    def _check(name: str, got: object, want: object) -> None:
        ok = got == want
        if not ok:
            failures.append(f"  FAIL {name}: got {got!r}, want {want!r}")
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {got} (want {want})")

    print("classify_flips synthetic self-test — per-case coverage:")
    for name, g, r, t, bucket in cases:
        print(f"    case {name:26s} gt={g} raw={r} rt={t} -> {bucket}")
    print("aggregate decomposition:")
    _check("r_cap", dec.r_cap, exp_r_cap)
    _check("r_surv", dec.r_surv, exp_r_surv)
    _check("r_total", dec.r_total, exp_total)
    _check("n_pixels", dec.n_pixels, exp_n)
    _check("r_roundtrip_repaired", dec.r_roundtrip_repaired, exp_repaired)
    _check("leverage", round(dec.leverage, 12), round(exp_leverage, 12))
    _check("partition_invariant(r_cap+r_surv==r_total)", dec.r_cap + dec.r_surv, dec.r_total)

    # Edge case: zero flips → leverage defined as 0.0 (no division by zero).
    z = np.array([1, 2, 3], dtype=np.int64)
    zdec = classify_flips(z, z, z)
    _check("zero_flip.r_total", zdec.r_total, 0)
    _check("zero_flip.leverage", zdec.leverage, 0.0)
    _check("zero_flip.d_seg", zdec.d_seg, 0.0)

    # Edge case: all flips are R_surv → leverage == 0.0 (routing cannot help at all).
    g2 = np.array([0, 0, 0], dtype=np.int64)
    r2 = np.array([0, 0, 0], dtype=np.int64)  # raw all correct
    t2 = np.array([1, 1, 1], dtype=np.int64)  # rt all wrong
    sdec = classify_flips(g2, r2, t2)
    _check("all_surv.r_surv", sdec.r_surv, 3)
    _check("all_surv.r_cap", sdec.r_cap, 0)
    _check("all_surv.leverage", sdec.leverage, 0.0)

    # Edge case: all flips are R_cap → leverage == 1.0 (routing could fix everything).
    g3 = np.array([0, 0, 0], dtype=np.int64)
    r3 = np.array([1, 1, 1], dtype=np.int64)  # raw all wrong
    t3 = np.array([2, 2, 2], dtype=np.int64)  # rt all wrong (different class)
    cdec = classify_flips(g3, r3, t3)
    _check("all_cap.r_cap", cdec.r_cap, 3)
    _check("all_cap.r_surv", cdec.r_surv, 0)
    _check("all_cap.leverage", cdec.leverage, 1.0)

    # Shape-mismatch must fail closed (no silent broadcast).
    raised = False
    try:
        classify_flips(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((3, 3)))
    except ValueError:
        raised = True
    _check("shape_mismatch_raises", raised, True)

    # 2-D coverage: confirm the logic is shape-agnostic (a small (H,W) grid).
    gg = np.array([[1, 1], [1, 1]], dtype=np.int64)
    rr = np.array([[1, 2], [1, 1]], dtype=np.int64)  # one raw-wrong
    tt = np.array([[1, 3], [4, 1]], dtype=np.int64)  # two rt-flips: (0,1)=cap (raw wrong), (1,0)=surv (raw correct)
    twod = classify_flips(gg, rr, tt)
    _check("2d.r_cap", twod.r_cap, 1)
    _check("2d.r_surv", twod.r_surv, 1)
    _check("2d.r_total", twod.r_total, 2)
    _check("2d.n_pixels", twod.n_pixels, 4)

    if failures:
        print("\nSELF-TEST: FAIL")
        for f in failures:
            print(f)
        return False
    print("\nSELF-TEST: PASS")
    return True


# ---------------------------------------------------------------------------
# THE REAL-DATA PATH — WRITTEN + py_compiled, NOT RUN (it would contend with the
# live MPS daemons / run the real scorer / forward a real checkpoint). Run at
# convergence on a freed slot. Bound 1:1 to the production APIs (cited above).
# ---------------------------------------------------------------------------
def _apply_eval_roundtrip(decoded_chw_384x512):
    """Apply the production eval round-trip R to decoder frames, 1:1 with
    ``tac.torch_vehicle.driver`` lines 2052-2058.

    ``decoded_chw_384x512``: a torch tensor (M, 3, 384, 512) float (the
    decoder's direct output, M = B*2 frames). Returns (M, 3, 384, 512) float in
    [0,255] AFTER bicubic↑(874,1164) → uint8-clamp/round → bilinear↓(384,512).
    This is the EXACT op sequence the training loop and the scorer measure d_seg
    against; the RAW path skips this function (sharp float, no round-trip)."""
    import torch.nn.functional as F  # local import: keep module import light + $0

    up = F.interpolate(
        decoded_chw_384x512, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
    )
    down = F.interpolate(up, size=(SEG_H, SEG_W), mode="bilinear", align_corners=False)
    # uint8 snap (the contest packet casts to uint8). Plain round here (no STE — we
    # are not differentiating); clamp first to the valid [0,255] range.
    return down.clamp(0, 255).round()


def _seg_argmax_authority(scorer, decoded_bhwc):
    """Seg-logit argmax on the AUTHORITY net, 1:1 with the scorer-forward of
    ``RealScorerContext.seg_pose_forward`` (scorer_context.py:191-200) but on
    ``scorer.distortion_net`` (CPU authority) NOT the train/MPS net.

    ``decoded_bhwc``: (B, 2, 384, 512, 3) float [0,255]. Returns the per-pair
    SegNet argmax (B, 384, 512) int64 on CPU — comparable to ``seg_targets_hard``.
    """
    import torch

    net = scorer.distortion_net  # AUTHORITY net (CPU-TRUSTED), never the train net
    with torch.inference_mode():
        # preprocess_input internally selects the SegNet (last-frame) input; segnet
        # returns (B, 5, 384, 512) logits matching seg_targets_hard per-pair indexing.
        _posenet_in, segnet_in = net.preprocess_input(decoded_bhwc)
        seg_logits = net.segnet(segnet_in)
        return seg_logits.argmax(dim=1).to("cpu").to(torch.int64)


def _load_decoder_and_latents(ckpt_dir: str):
    """Load the EMA decoder + latents from a checkpoint dir (UNTESTED real path).

    Bound to ``tac.torch_vehicle.checkpoint`` (read_manifest:300, load_checkpoint:305)
    for a rolling/stage checkpoint, and falls back to the canonical ``best/`` layout
    (``best_ema_decoder.pt`` + ``best_ema_latents.pt``, the warm-start layout
    ``driver._load_warm_start_basin`` consumes at lines 1613-1655) when no manifest
    is present.

    The decoder architecture is rebuilt from the manifest's ``base_channels`` /
    ``latent_dim`` / ``taper_channels``. The DEFAULT live basin is the vendored
    ``HNeRVDecoder`` (taper_channels is None) — that case is fully bound here. A
    manifest with a FiLM/activation override is NOT recoverable from the manifest
    fields alone (the manifest carries no ``activation`` / ``pose_film`` flag); we
    fail closed with a precise message rather than silently mis-building (NO-FAKE).

    Returns ``(ema_decoder (eval, CPU), ema_latents (CPU), manifest_dict)``.
    """
    import torch

    from pathlib import Path

    from tac.torch_vehicle.checkpoint import checkpoint_exists, load_checkpoint, read_manifest
    from tac.torch_vehicle.vendored_imports import import_vendored

    ckpt = Path(ckpt_dir)

    def _build_vendored_decoder(base_channels: int, latent_dim: int):
        """Build the DEFAULT live-basin decoder (vendored HNeRVDecoder, no taper)."""
        model = import_vendored("model")  # the vendored module exporting HNeRVDecoder
        return model.HNeRVDecoder(
            latent_dim=int(latent_dim),
            base_channels=int(base_channels),
            eval_size=(SEG_H, SEG_W),
        ).eval()

    def _build_taper_decoder(manifest: dict):
        """Build a ConfigurableTaper decoder when the manifest declares a taper."""
        from tac.torch_vehicle.configurable_taper_decoder import (
            ConfigurableTaperHNeRVDecoder,
        )

        return ConfigurableTaperHNeRVDecoder(
            latent_dim=int(manifest["latent_dim"]),
            base_channels=int(manifest["base_channels"]),
            eval_size=(SEG_H, SEG_W),
            channels=list(manifest["taper_channels"]),
        ).eval()

    if checkpoint_exists(ckpt):
        manifest = read_manifest(ckpt)
        merged = load_checkpoint(ckpt, map_location="cpu")
        ema_sd = merged["ema_decoder"]
        ema_latents = merged["ema_latents"]
        taper = manifest.get("taper_channels")
        if taper is not None:
            decoder = _build_taper_decoder(manifest)
        else:
            decoder = _build_vendored_decoder(
                manifest["base_channels"], manifest["latent_dim"]
            )
        decoder.load_state_dict({k: v.to("cpu") for k, v in ema_sd.items()})
        return decoder, ema_latents.to("cpu"), manifest

    # Fallback: the canonical best/ warm-start layout (no manifest).
    dec_path = ckpt / "best_ema_decoder.pt"
    lat_path = ckpt / "best_ema_latents.pt"
    if not (dec_path.exists() and lat_path.exists()):
        raise FileNotFoundError(
            f"--ckpt {ckpt} is neither a rolling/stage checkpoint (no "
            f"{ckpt}/torch_vehicle_checkpoint_manifest.json) nor a best/ layout "
            f"(missing best_ema_decoder.pt / best_ema_latents.pt). Point --ckpt at "
            f"a driver checkpoint dir or a best/ dir, and pass --base-channels / "
            f"--latent-dim for the best/ fallback."
        )
    sd = torch.load(dec_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    lat = torch.load(lat_path, map_location="cpu", weights_only=False)
    if isinstance(lat, dict):
        lat = lat.get("latents", next(iter(lat.values())))
    # best/ fallback has no manifest → caller MUST supply base_channels/latent_dim.
    raise NotImplementedError(
        "best/ fallback requires explicit --base-channels and --latent-dim (the "
        "best/ layout carries no architecture manifest). The rolling/stage "
        "checkpoint path (with torch_vehicle_checkpoint_manifest.json) is the "
        "fully-bound path; rebuild the decoder via _build_vendored_decoder with "
        "the supplied dims, then load_state_dict(sd) and use lat."
    )


def _measure_real(ckpt_dir: str, video_path: str, max_pairs: int | None, batch_pairs: int) -> dict:
    """[UNTESTED real-data path] Compute R_cap/R_surv on a real trained decoder.

    Steps (all bound to production APIs, NONE run until a slot frees):
      1. Build the real CPU-authority scorer via ``RealScorerContext(video_path,
         device='cpu')`` (scorer_context.py:53-127) → frozen SegNet/PoseNet +
         cached ``seg_targets_hard`` GT argmax (precompute_targets, GT decoded via
         frame_utils.yuv420_to_rgb).
      2. Load the EMA decoder + latents from ``ckpt_dir`` (_load_decoder_and_latents).
      3. For each pair batch: decoder forward → (B,2,3,384,512) float; build the
         RAW BHWC frames (no round-trip) and the RT BHWC frames (eval round-trip R);
         run the AUTHORITY SegNet argmax on each; gt = seg_targets_hard[idx].
      4. Accumulate the R_cap/R_surv decomposition over all pairs and report the
         routing-leverage ceiling + the d_seg contribution (tac.contest_score).

    Returns the JSON-able result dict.
    """
    import torch

    from tac.torch_vehicle.scorer_context import RealScorerContext
    from tac.contest_score import seg_term

    print(
        "[measure_r_cap_r_surv] >>> RUNNING THE UNTESTED REAL-DATA PATH <<<\n"
        "  This loads a checkpoint + runs the REAL frozen scorer on REAL frames.\n"
        "  Only run when NO MPS training daemon holds a slot (it must not contend).",
        flush=True,
    )

    scorer = RealScorerContext(video_path, device="cpu", max_pairs=max_pairs)
    decoder, latents, manifest = _load_decoder_and_latents(ckpt_dir)
    n_pairs = min(int(scorer.n_pairs), int(latents.shape[0]))
    if max_pairs is not None:
        n_pairs = min(n_pairs, int(max_pairs))

    gt_all = scorer.seg_targets_hard.to("cpu").to(torch.int64)  # (n_pairs,384,512)

    tot = FlipDecomposition(0, 0, 0, 0, 0, 0.0)
    acc_cap = acc_surv = acc_total = acc_n = acc_repaired = 0
    decoder.eval()
    with torch.inference_mode():
        for start in range(0, n_pairs, batch_pairs):
            idx = torch.arange(start, min(start + batch_pairs, n_pairs))
            b = int(idx.shape[0])
            z = latents[idx].to("cpu")
            decoded = decoder(z)  # (b, 2, 3, 384, 512) float
            flat = decoded.reshape(b * 2, 3, SEG_H, SEG_W)
            # RAW: decoder output straight to BHWC, NO round-trip (sharp float).
            raw_bhwc = flat.reshape(b, 2, 3, SEG_H, SEG_W).permute(0, 1, 3, 4, 2)
            # RT: apply the production eval round-trip R, then to BHWC.
            rt_flat = _apply_eval_roundtrip(flat)
            rt_bhwc = rt_flat.reshape(b, 2, 3, SEG_H, SEG_W).permute(0, 1, 3, 4, 2)
            raw_argmax = _seg_argmax_authority(scorer, raw_bhwc)  # (b,384,512)
            rt_argmax = _seg_argmax_authority(scorer, rt_bhwc)  # (b,384,512)
            gt_argmax = gt_all[idx]  # (b,384,512)
            dec = classify_flips(
                gt_argmax.numpy(), raw_argmax.numpy(), rt_argmax.numpy()
            )
            acc_cap += dec.r_cap
            acc_surv += dec.r_surv
            acc_total += dec.r_total
            acc_n += dec.n_pixels
            acc_repaired += dec.r_roundtrip_repaired
            print(
                f"  pairs[{start}:{start + b}] r_cap={dec.r_cap} r_surv={dec.r_surv} "
                f"r_total={dec.r_total} leverage={dec.leverage:.4f}",
                flush=True,
            )

    tot = FlipDecomposition(
        r_cap=acc_cap,
        r_surv=acc_surv,
        r_total=acc_total,
        n_pixels=acc_n,
        r_roundtrip_repaired=acc_repaired,
        leverage=(acc_cap / acc_total) if acc_total > 0 else 0.0,
    )
    d_seg = tot.d_seg
    # The d_seg debt a PERFECT routing lever could recover (the ceiling), in
    # score units: 100 * (d_seg fraction that is R_cap). seg_term = 100 * d_seg.
    d_seg_cap_ceiling = (tot.r_cap / tot.n_pixels) if tot.n_pixels else 0.0
    result = {
        **asdict(tot),
        "d_seg": d_seg,
        "d_seg_seg_term": seg_term(d_seg),  # 100 * d_seg (the score contribution)
        "d_seg_cap_recoverable_ceiling": d_seg_cap_ceiling,
        "seg_term_cap_recoverable_ceiling": seg_term(d_seg_cap_ceiling),
        "n_pairs": int(n_pairs),
        "ckpt_dir": str(ckpt_dir),
        "video_path": str(video_path),
        "manifest": {
            k: manifest.get(k)
            for k in ("base_channels", "latent_dim", "n_pairs", "taper_channels", "stage_name")
        },
        "authority": "[contest-CPU advisory] NON-PROMOTABLE — diagnostic ceiling, not a score",
        "untested_path": "real-data path executed (was UNTESTED at build time)",
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose the d_seg argmax-flip set into R_cap (capacity-fixable) vs "
            "R_surv (round-trip-aliased) and report the routing-leverage ceiling "
            "R_cap/R_total."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--self-test",
        action="store_true",
        help="Run the synthetic classify_flips test ($0, pure numpy; RUN now).",
    )
    mode.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=(
            "[UNTESTED real-data path] checkpoint dir (rolling/stage checkpoint or "
            "best/ layout). Loads the decoder + runs the REAL scorer — do NOT run "
            "while an MPS training daemon holds a slot."
        ),
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="upstream/videos/0.mkv",
        help="Contest video for the real scorer GT (default upstream/videos/0.mkv).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Cap pairs for a cheap real run (default: all pairs).",
    )
    parser.add_argument(
        "--batch-pairs",
        type=int,
        default=8,
        help="Pairs per decoder/scorer batch in the real path (default 8).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write the result JSON (real-data path only).",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        ok = _run_self_test()
        return 0 if ok else 1

    # --ckpt: the UNTESTED real-data path.
    print(
        "============================================================\n"
        "[UNTESTED real-data path — run at convergence on a freed MPS slot]\n"
        "This path loads a checkpoint + runs the REAL frozen scorer on REAL\n"
        "frames + forwards a REAL decoder. It was NOT executed at build time\n"
        "(the live MPS training daemons held the slots; running it would\n"
        "contend). It IS bound 1:1 to the production APIs and py_compiles.\n"
        "============================================================",
        flush=True,
    )
    result = _measure_real(
        args.ckpt, args.video_path, args.max_pairs, int(args.batch_pairs)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        from pathlib import Path

        Path(args.json_out).write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"[measure_r_cap_r_surv] wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
