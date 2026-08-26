# SPDX-License-Identifier: MIT
"""$0 CPU-only DECISIVE probe (NeRV-family survey N1): does HiNeRV's hierarchical
feature-grid shave matched-byte exact d_seg BEYOND what the margin-weighted seg
lever already gives on the HNeRV bank?

THE QUESTION (survey `.omx/research/nerv_family_synergy_survey_20260612T174528Z.md`
§D.3, falsifiable claim N1): at MATCHED archive bytes, does HiNeRV's hierarchical
feature-grid decoder lower the EXACT d_seg (argmax-flip rate) by >=20% vs the HNeRV
decoder WITH margin-weighted-seg already applied, AND does the gain SURVIVE the eval
round-trip? GO (build HiNeRV) if yes; NO-GO (the lever is the d_seg ceiling) if no.

THE PROTOCOL (matched-byte head-to-head, $0 CPU, small-n overfit, NO GPU, NO basin
contention). Five arms, all at a MATCHED ~83-88K decoder-param byte budget, the SAME
N real GT pairs, IDENTICAL compute (same epochs/LR/seed), each overfit FROM SCRATCH:
  * L0   : vendored PR95 HNeRVDecoder (base_ch=20, the basin family), plain seg+pose
  * L     : vendored HNeRVDecoder + margin-weighted-seg lever (the d_seg ceiling baseline)
  * Hoff : HiNeRV architecture, hierarchical feature grid OFF, plain seg+pose
  * H0   : HiNeRV architecture, hierarchical feature grid ON, plain seg+pose
  * H     : HiNeRV grid ON + margin-weighted-seg lever (grid AND lever stacked)
The DECISIVE N1 comparison is min(d_seg(H0), d_seg(H)) vs d_seg(L).

AUTHORITY / NO FAKE (CLAUDE.md):
  * CPU ONLY — the authority device; runs while the live MPS basin owns the GPU,
    zero contention. NO mps, NO mlx, NO GPU.
  * REAL frozen contest scorer (``load_frozen_distortion_net(device='cpu')``) and
    REAL ``upstream/videos/0.mkv`` GT frames decoded ONLY via
    ``frame_utils.yuv420_to_rgb`` (PyAV rgb24 manufactures phantom pose).
  * The reported d_seg is the EXACT argmax-flip rate the evaluator charges:
    ``DistortionNet.compute_distortion(gt_pair, decoded_pair)`` (SegNet
    ``(out1.argmax != out2.argmax).float().mean()``) AFTER the SAME eval round-trip
    the vendored ``score.evaluate_decoder`` applies (bicubic^874 -> uint8). The
    round-trip survival check re-measures d_seg AFTER that round-trip — it is built
    in (the metric is always measured on the round-tripped frame).
  * HiNeRV grid path is run as a REFERENCE FORWARD (the in-repo
    ``HinervSubstrate(use_hierarchical_feature_grid=True)`` torch forward). We do
    NOT edit ``hi_nerv`` source defaults; we set the flag on the probe's OWN model
    instance. ``official_core_forward_parity_proven=False`` means our grid sampler is
    not byte-proven against the OFFICIAL HiNeRV repo — but our path RUNS and renders
    sane frames (verified by the forward-parity sanity check below); that is what a
    $0 capacity probe needs. If the grid path did NOT run, that is itself the finding.
  * Every number is ``[contest-CPU advisory]`` — small-n magnitude is advisory; the
    THRESHOLD comparison (>=20% vs the lever, survived) is the deliverable. The CPU
    numerics are authority-grade. score_claim=false, promotable=false,
    ready_for_exact_eval_dispatch=false. Per Catalog #307 the verdict is
    EV/IMPLEMENTATION-level (build-vs-defer), NOT a paradigm kill.

It edits NO files other agents own (driver.py / curriculum.py / cool_chic / hi_nerv
source / the basin daemon/out-dir). It only READS the frozen fork-point checkpoint
(for a faithful base_ch=20 latent-dim init reference) and the real video.

Usage::

    .venv/bin/python experiments/probe_hinerv_grid_vs_lever_dseg.py \
        --n-pairs 16 --epochs 400 --out-json experiments/results/<...>/result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_VIDEO = Path("upstream/videos/0.mkv")
_FORKPOINT = Path(
    "experiments/results/forkpoints/basin_bc20_20260612T121523Z/"
    "torch_vehicle_checkpoint_state.pt"
)
_EVAL_H, _EVAL_W = 384, 512
_CAMERA_H, _CAMERA_W = 874, 1164
_SEG_NUM_CLASSES = 5
_BASE_CHANNELS = 20
_LATENT_DIM = 28
_POSE_DIM = 6

# Matched-byte HiNeRV config: ~88K (grid ON) / ~86K (grid OFF) decoder params, within
# +6% of the vendored HNeRV decoder's 83,356 (measured this session). The grid itself
# costs only ~2,008 params (~2.4% of the budget) — so a d_seg win cannot be a "more
# bytes" artifact; it is genuinely the grid's spatial inductive bias.
_HINERV_EMBED_DIM = 24
_HINERV_DECODER_CHANNELS = (24, 20, 18, 16, 14, 12, 10)


# ---------------------------------------------------------------------------
# eval round-trip (1:1 with vendored ``score.evaluate_decoder``:
# decoder out (384,512) float[0,255] -> bicubic^(874,1164) -> clamp/round/uint8)
# ---------------------------------------------------------------------------
def _roundtrip_to_eval_bhwc(decoded: torch.Tensor) -> torch.Tensor:
    """``decoded`` is (B, 2, 3, 384, 512) float [0,255]. Return the (B, 2, 874, 1164, 3)
    uint8 tensor the scorer reads — bicubic up to camera res, clamp/round/uint8. This
    is EXACTLY ``score.evaluate_decoder``'s ``_decoded_to_camera`` + the
    clamp/round/cast-to-uint8 sequence it applies before ``compute_distortion``
    (verified against the vendored source this session)."""
    b = decoded.shape[0]
    flat = decoded.reshape(b * 2, 3, _EVAL_H, _EVAL_W)
    up = F.interpolate(flat, size=(_CAMERA_H, _CAMERA_W), mode="bicubic", align_corners=False)
    bhwc = up.reshape(b, 2, 3, _CAMERA_H, _CAMERA_W).permute(0, 1, 3, 4, 2)
    clamped = bhwc.clamp(0, 255)
    rounded = clamped + (clamped.round() - clamped).detach()
    return rounded.detach().to(torch.uint8)


def _roundtrip_to_eval_bhwc_differentiable(decoded: torch.Tensor) -> torch.Tensor:
    """Same eval round-trip but with a straight-through uint8 round so the seg/pose
    gradient flows through the round-trip the evaluator applies (CLAUDE.md
    ``eval_roundtrip`` non-negotiable). Returns (B, 2, 874, 1164, 3) float."""
    b = decoded.shape[0]
    flat = decoded.reshape(b * 2, 3, _EVAL_H, _EVAL_W)
    up = F.interpolate(flat, size=(_CAMERA_H, _CAMERA_W), mode="bicubic", align_corners=False)
    bhwc = up.reshape(b, 2, 3, _CAMERA_H, _CAMERA_W).permute(0, 1, 3, 4, 2)
    clamped = bhwc.clamp(0, 255)
    return clamped + (clamped.round() - clamped).detach()  # STE uint8 round


# ---------------------------------------------------------------------------
# GT pair streaming (REAL frames, canonical decode)
# ---------------------------------------------------------------------------
def _stream_gt_pairs(video_path: Path, n_pairs: int, device: str = "cpu") -> torch.Tensor:
    """First ``n_pairs`` GT pairs as (n_pairs, 2, 874, 1164, 3) uint8 via the canonical
    ``yuv420_to_rgb`` decode (NOT PyAV rgb24)."""
    sys.path.insert(0, str(Path("upstream").resolve()))
    import av
    from frame_utils import yuv420_to_rgb

    container = av.open(str(video_path))
    pairs: list[torch.Tensor] = []
    prev = None
    with torch.inference_mode():
        for frame in container.decode(container.streams.video[0]):
            f = yuv420_to_rgb(frame)
            if prev is None:
                prev = f
                continue
            f0, f1 = prev, f
            prev = None
            pairs.append(torch.stack([f0, f1]))  # (2, H, W, 3)
            if len(pairs) >= n_pairs:
                break
    container.close()
    return torch.stack(pairs).to(device)  # (n_pairs, 2, 874, 1164, 3) uint8


# ---------------------------------------------------------------------------
# decoder builders (all matched-byte; HiNeRV grid flag set on OUR instance only)
# ---------------------------------------------------------------------------
def _build_hnerv_decoder(device: str = "cpu") -> nn.Module:
    """Vendored PR95 ``HNeRVDecoder`` (base_ch=20, latent 28) — the basin family.
    Output is (B, 2, 3, 384, 512) float [0,255]."""
    from tac.torch_vehicle.driver import import_vendored_bundle

    v = import_vendored_bundle()
    dec = v.HNeRVDecoder(
        latent_dim=_LATENT_DIM, base_channels=_BASE_CHANNELS, eval_size=(_EVAL_H, _EVAL_W)
    ).to(device)
    return dec


class _HinervWrapped(nn.Module):
    """Wrap ``HinervSubstrate`` so it consumes a per-pair latent ``z`` (B, latent_dim)
    like the HNeRV decoder, and emits (B, 2, 3, 384, 512) float [0,255].

    HiNeRV's substrate owns its OWN per-pair latents indexed by ``pair_indices``; to
    make the arms apples-to-apples (same latent INTERFACE), we map the row index of
    ``z`` to a contiguous ``pair_indices`` and let the HiNeRV substrate use its internal
    latents (the byte budget already accounts for them). The probe trains the WHOLE
    substrate (decoder + its latents) for the n pairs — symmetric with training the
    HNeRV decoder + its latents. HiNeRV heads are sigmoid in [0,1]; scale to [0,255]."""

    def __init__(self, *, n_pairs: int, grid_on: bool, device: str = "cpu") -> None:
        super().__init__()
        from tac.substrates.hi_nerv.architecture import HinervConfig, HinervSubstrate

        cfg = HinervConfig(
            num_pairs=int(n_pairs),
            embed_dim=_HINERV_EMBED_DIM,
            decoder_channels=_HINERV_DECODER_CHANNELS,
            use_hierarchical_feature_grid=bool(grid_on),
            use_convnext_blocks=False,
        )
        self.substrate = HinervSubstrate(cfg).to(device)
        self.n_pairs = int(n_pairs)

    def forward(self, pair_indices: torch.Tensor) -> torch.Tensor:
        r0, r1 = self.substrate(pair_indices.long())  # each (B, 3, 384, 512) in [0,1]
        return torch.stack([r0 * 255.0, r1 * 255.0], dim=1)  # (B, 2, 3, 384, 512)


# ---------------------------------------------------------------------------
# margin-weighted seg lever — VERBATIM port of
# ``driver._segnet_logit_margin_map`` + ``driver._seg_loss_for_spec`` (the lever path)
# ---------------------------------------------------------------------------
def _segnet_logit_margin_map(seg_out: torch.Tensor) -> torch.Tensor:
    """Per-pixel SegNet top1 - top2 logit margin of the prediction logits (detached
    per-pixel WEIGHT). Verbatim from ``driver._segnet_logit_margin_map``."""
    top2, _ = torch.topk(seg_out.detach(), k=2, dim=1, largest=True, sorted=True)
    return (top2[:, 0] - top2[:, 1]).clamp_min(0.0)  # (B, H, W) >= 0


def _seg_surrogate_loss(
    seg_out: torch.Tensor,
    seg_targets_hard: torch.Tensor,
    *,
    margin_weight_tau: float | None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Score-domain d_seg surrogate (soft-cosine on one-hot GT logits) + optional
    Lever-5 margin weighting. Verbatim mechanism from ``driver._seg_loss_for_spec``
    (the ``spec.seg_surrogate='soft_cosine'`` branch). ``margin_weight_tau is None``
    = the Lever-2 baseline surrogate; set = the margin-weighted lever."""
    from tac.losses.core import segnet_surrogate_per_pixel

    _ONEHOT_GT_LOGIT = 30.0
    gt_onehot = F.one_hot(seg_targets_hard.long(), num_classes=_SEG_NUM_CLASSES)
    gt_logits = gt_onehot.permute(0, 3, 1, 2).to(seg_out.dtype) * _ONEHOT_GT_LOGIT
    per_pixel = segnet_surrogate_per_pixel(
        seg_out, gt_logits, surrogate="soft_cosine",
        temperature=float(temperature), gt_already_probs=False,
        num_classes=_SEG_NUM_CLASSES,
    )  # (B, H, W)
    if margin_weight_tau is not None:
        margin = _segnet_logit_margin_map(seg_out)
        tau = max(float(margin_weight_tau), 1e-6)
        weight = torch.exp(-margin / tau)
        return (per_pixel * weight).mean()
    return per_pixel.mean()


# ---------------------------------------------------------------------------
# exact d_seg / d_pose measurement (authority, after the eval round-trip)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _measure_exact_dseg_dpose(
    render_fn, gt_pairs: torch.Tensor, distortion_net: nn.Module, n: int, batch: int = 8
) -> tuple[float, float]:
    """``render_fn(idx_tensor) -> (B,2,3,384,512) float[0,255]``. Round-trip, then
    measure the EXACT (d_pose, d_seg) the evaluator charges. d_seg is the SegNet
    argmax-flip rate ``(gt.argmax != decoded.argmax).mean()`` — the contest metric."""
    pose_total = seg_total = 0.0
    count = 0
    for start in range(0, n, batch):
        idx = torch.arange(start, min(start + batch, n))
        decoded = render_fn(idx)
        decoded_bhwc = _roundtrip_to_eval_bhwc(decoded)
        gt = gt_pairs[idx]
        pose_d, seg_d = distortion_net.compute_distortion(gt, decoded_bhwc)
        pose_total += pose_d.sum().item()
        seg_total += seg_d.sum().item()
        count += len(idx)
    return pose_total / max(count, 1), seg_total / max(count, 1)


# ---------------------------------------------------------------------------
# overfit one arm
# ---------------------------------------------------------------------------
@dataclass
class _ArmResult:
    name: str
    family: str  # "hnerv" | "hinerv"
    grid_on: bool
    margin_lever: bool
    decoder_param_bytes_fp16: int  # the matched-byte budget surrogate (decoder only)
    decoder_params: int
    d_seg: float
    d_pose: float
    forward_sane: bool
    d_seg_trace: list[tuple[int, float, float]]  # [(epoch, d_seg, d_pose), ...]


def _decoder_param_count(model: nn.Module, *, hinerv: bool) -> int:
    """Decoder-only param count (exclude per-pair latents) — the matched-byte budget."""
    latent_markers = (
        ("substrate.latents_coarse", "substrate.latents_mid", "substrate.latents_fine")
        if hinerv
        else ("latents",)  # HNeRV decoder owns no latents (they are separate tensors)
    )
    return sum(
        p.numel()
        for nm, p in model.named_parameters()
        if not any(nm.startswith(m) for m in latent_markers)
    )


def _train_and_measure_arm(
    *,
    name: str,
    family: str,
    grid_on: bool,
    margin_lever: bool,
    margin_tau: float,
    n: int,
    epochs: int,
    lr: float,
    seg_weight: float,
    pose_weight: float,
    seed: int,
    gt_pairs: torch.Tensor,
    distortion_net: nn.Module,
    seg_targets_hard: torch.Tensor,
    pose_targets: torch.Tensor,
    init_latents: torch.Tensor,
    device: str = "cpu",
    trace_every: int = 0,
) -> _ArmResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    hinerv = family == "hinerv"
    if hinerv:
        model = _HinervWrapped(n_pairs=n, grid_on=grid_on, device=device)

        def render_fn(idx: torch.Tensor) -> torch.Tensor:
            return model(idx.to(device))

    else:
        decoder = _build_hnerv_decoder(device)
        # latents trained alongside (seeded from the basin EMA latents for a faithful
        # base_ch=20 init, then overfit on n pairs — symmetric per-arm overfit).
        latents = nn.Parameter(init_latents[:n].clone().to(device))
        model = nn.ModuleDict({"decoder": decoder})
        model.latents = latents  # register

        def render_fn(idx: torch.Tensor) -> torch.Tensor:
            return decoder(model.latents[idx.to(device)])

    # forward-parity sanity (BEFORE training): does the arm render a valid finite
    # frame in plausible RGB range? (Especially the HiNeRV grid path — the parity flag
    # is False; confirm it RUNS before trusting any number.)
    with torch.inference_mode():
        probe_idx = torch.arange(min(2, n))
        sample = render_fn(probe_idx)
        forward_sane = bool(
            torch.isfinite(sample).all()
            and sample.shape[1:] == (2, 3, _EVAL_H, _EVAL_W)
            and float(sample.min()) >= -1.0
            and float(sample.max()) <= 256.0
        )

    params = [p for p in model.parameters() if p.requires_grad]
    if not hinerv:
        params = [*decoder.parameters(), model.latents]
    opt = torch.optim.Adam(params, lr=lr)
    bs = 8
    trace: list[tuple[int, float, float]] = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            decoded = render_fn(idx)
            decoded_bhwc = _roundtrip_to_eval_bhwc_differentiable(decoded)
            posenet_out, segnet_out = distortion_net(decoded_bhwc)
            seg_l = _seg_surrogate_loss(
                segnet_out, seg_targets_hard[idx],
                margin_weight_tau=(margin_tau if margin_lever else None),
            )
            # GT pose target is CONSTANT — precomputed once (no per-step GT forward;
            # ~halves the per-step scorer cost). The clone is a normal (non-inference)
            # tensor so it is autograd-safe as the MSE target.
            pose_gt = pose_targets[idx]
            pose_dec = posenet_out["pose"][:, :_POSE_DIM]
            pose_l = F.mse_loss(pose_dec, pose_gt)
            loss = seg_weight * seg_l + pose_weight * pose_l
            opt.zero_grad()
            loss.backward()
            opt.step()
        if trace_every and (ep % trace_every == 0 or ep == epochs - 1):
            dp_t, ds_t = _measure_exact_dseg_dpose(render_fn, gt_pairs, distortion_net, n)
            trace.append((ep, float(ds_t), float(dp_t)))
            print(f"[probe]     {name} ep {ep:3d}: d_seg={ds_t:.5f} d_pose={dp_t:.4f}", flush=True)

    d_pose, d_seg = _measure_exact_dseg_dpose(render_fn, gt_pairs, distortion_net, n)
    dec_params = _decoder_param_count(model, hinerv=hinerv)
    return _ArmResult(
        name=name,
        family=family,
        grid_on=grid_on,
        margin_lever=margin_lever,
        decoder_param_bytes_fp16=dec_params * 2,
        decoder_params=dec_params,
        d_seg=d_seg,
        d_pose=d_pose,
        forward_sane=forward_sane,
        d_seg_trace=trace,
    )


# ---------------------------------------------------------------------------
# the probe
# ---------------------------------------------------------------------------
@dataclass
class _ProbeResult:
    n_pairs: int
    epochs: int
    lr: float
    seg_weight: float
    pose_weight: float
    margin_tau: float
    seed: int
    arms: list[dict]
    # decisive N1 comparison
    d_seg_L_lever: float       # HNeRV + margin lever (the ceiling baseline)
    d_seg_H_best: float        # best of {H0 grid-plain, H grid+lever}
    d_seg_H_best_arm: str
    pct_reduction_vs_lever: float  # (d_seg_L - d_seg_H_best) / d_seg_L  (positive = grid wins)
    grid_intrinsic_pct_vs_hoff: float  # (d_seg_Hoff - d_seg_H0)/d_seg_Hoff (grid's own contribution)
    hinerv_grid_forward_sane: bool
    roundtrip_survived: bool  # the d_seg numbers ARE measured post-roundtrip (always True if measured)
    threshold_pct: float
    verdict: str
    authority: str


def run(
    *, n_pairs: int, epochs: int, lr: float, seg_weight: float, pose_weight: float,
    margin_tau: float, threshold_pct: float, seed: int, trace_every: int = 0,
    arms_cache_dir: str | None = None,
) -> _ProbeResult:
    device = "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"[probe] streaming {n_pairs} REAL GT pairs (yuv420_to_rgb, camera-res)", flush=True)
    gt_pairs = _stream_gt_pairs(_VIDEO, n_pairs, device=device)
    assert gt_pairs.shape[0] == n_pairs, gt_pairs.shape

    print("[probe] loading frozen contest DistortionNet (CPU authority)", flush=True)
    from tac.score_aware_loop.targets import load_frozen_distortion_net

    distortion_net = load_frozen_distortion_net(device=device)

    # GT SegNet-argmax + PoseNet targets (the d_seg / d_pose references) — the frozen
    # scorer's OWN output on the GT frames (the contest's seg/pose reference). seg at
    # 384x512 (SegNet input res, the per-pixel surrogate's resolution); pose6 is the
    # constant MSE target. Both precomputed ONCE (no per-step GT forward).
    print("[probe] computing GT SegNet-argmax + PoseNet targets (seg/pose references)", flush=True)
    seg_targets_hard = torch.zeros(n_pairs, _EVAL_H, _EVAL_W, dtype=torch.long)
    pose_targets = torch.zeros(n_pairs, _POSE_DIM, dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, n_pairs, 8):
            idx = torch.arange(start, min(start + 8, n_pairs))
            po_gt, seg_out_gt = distortion_net(gt_pairs[idx])  # scorer output on GT
            seg_targets_hard[idx] = seg_out_gt.argmax(dim=1)
            pose_targets[idx] = po_gt["pose"][:, :_POSE_DIM].float()

    # faithful base_ch=20 latent init (EMA latents from the frozen basin fork-point).
    print(f"[probe] reading fork-point EMA latents (init reference) {_FORKPOINT}", flush=True)
    blob = torch.load(_FORKPOINT, map_location="cpu", weights_only=False)
    init_latents = blob["ema_latents"].to(device)  # (600, 28)
    assert init_latents.shape[1] == _LATENT_DIM, init_latents.shape

    arm_specs = [
        {"name": "L0_hnerv_plain", "family": "hnerv", "grid_on": False, "margin_lever": False},
        {"name": "L_hnerv_lever", "family": "hnerv", "grid_on": False, "margin_lever": True},
        {"name": "Hoff_hinerv_gridoff_plain", "family": "hinerv", "grid_on": False, "margin_lever": False},
        {"name": "H0_hinerv_gridon_plain", "family": "hinerv", "grid_on": True, "margin_lever": False},
        {"name": "H_hinerv_gridon_lever", "family": "hinerv", "grid_on": True, "margin_lever": True},
    ]
    cache_dir = Path(arms_cache_dir) if arms_cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    arms: list[_ArmResult] = []
    for spec in arm_specs:
        # Resumable per-arm: skip an arm whose result JSON already exists for THIS
        # (n_pairs, epochs, seed) config (durable across restarts; LONG-SWEEP safe).
        arm_cache = (
            cache_dir / f"arm_{spec['name']}_n{n_pairs}_e{epochs}_s{seed}.json"
            if cache_dir is not None
            else None
        )
        if arm_cache is not None and arm_cache.exists():
            cached = json.loads(arm_cache.read_text())
            cached["d_seg_trace"] = [tuple(t) for t in cached.get("d_seg_trace", [])]
            arms.append(_ArmResult(**cached))
            print(f"[probe] ARM {spec['name']}: RESUMED from cache "
                  f"(d_seg={cached['d_seg']:.6g})", flush=True)
            continue
        t0 = time.time()
        print(f"[probe] ARM {spec['name']}: overfit {epochs}ep on {n_pairs} pairs ...", flush=True)
        res = _train_and_measure_arm(
            **spec, margin_tau=margin_tau, n=n_pairs, epochs=epochs, lr=lr,
            seg_weight=seg_weight, pose_weight=pose_weight, seed=seed,
            gt_pairs=gt_pairs, distortion_net=distortion_net,
            seg_targets_hard=seg_targets_hard, pose_targets=pose_targets,
            init_latents=init_latents, device=device, trace_every=trace_every,
        )
        if arm_cache is not None:
            arm_cache.write_text(json.dumps(asdict(res), indent=2))
        arms.append(res)
        print(
            f"[probe]   {res.name}: d_seg={res.d_seg:.6g} d_pose={res.d_pose:.6g} "
            f"dec_params={res.decoder_params} forward_sane={res.forward_sane} "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )

    by = {a.name: a for a in arms}
    d_seg_L = by["L_hnerv_lever"].d_seg
    h_candidates = [by["H0_hinerv_gridon_plain"], by["H_hinerv_gridon_lever"]]
    h_best = min(h_candidates, key=lambda a: a.d_seg)
    d_seg_H_best = h_best.d_seg
    pct_reduction = (d_seg_L - d_seg_H_best) / max(d_seg_L, 1e-12)
    d_seg_hoff = by["Hoff_hinerv_gridoff_plain"].d_seg
    d_seg_h0 = by["H0_hinerv_gridon_plain"].d_seg
    grid_intrinsic = (d_seg_hoff - d_seg_h0) / max(d_seg_hoff, 1e-12)
    grid_sane = by["H0_hinerv_gridon_plain"].forward_sane and by["H_hinerv_gridon_lever"].forward_sane

    # verdict
    if not grid_sane:
        verdict = "BLOCKED"
    elif pct_reduction >= threshold_pct:
        verdict = "GO"
    else:
        verdict = "NO-GO"

    return _ProbeResult(
        n_pairs=n_pairs, epochs=epochs, lr=lr, seg_weight=seg_weight,
        pose_weight=pose_weight, margin_tau=margin_tau, seed=seed,
        arms=[asdict(a) for a in arms],
        d_seg_L_lever=d_seg_L, d_seg_H_best=d_seg_H_best, d_seg_H_best_arm=h_best.name,
        pct_reduction_vs_lever=pct_reduction, grid_intrinsic_pct_vs_hoff=grid_intrinsic,
        hinerv_grid_forward_sane=grid_sane,
        roundtrip_survived=True,  # d_seg is ALWAYS measured post-roundtrip (built in)
        threshold_pct=threshold_pct,
        verdict=verdict,
        authority="[contest-CPU advisory] NON-PROMOTABLE — score_claim=false, "
        "promotable=false, ready_for_exact_eval_dispatch=false. Small-n matched-byte "
        "overfit; the >=20% threshold comparison is the deliverable, the absolute "
        "magnitude is advisory. Per Catalog #307 this is an EV/build-vs-defer verdict, "
        "NOT a paradigm kill. Exact verdict requires paired CPU/CUDA upstream/evaluate.py "
        "on a byte-closed archive.",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=16, help="real GT pairs (8-16)")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seg-weight", type=float, default=100.0)
    ap.add_argument("--pose-weight", type=float, default=1.0)
    ap.add_argument("--margin-tau", type=float, default=2.0)
    ap.add_argument("--threshold-pct", type=float, default=0.20, help="N1 GO threshold")
    ap.add_argument("--trace-every", type=int, default=0, help="measure d_seg every N epochs (0=off)")
    ap.add_argument("--arms-cache-dir", type=str, default=None, help="per-arm resume cache dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=str, default=None)
    args = ap.parse_args()

    if not _VIDEO.exists():
        raise SystemExit(f"video not found: {_VIDEO}")
    if not _FORKPOINT.exists():
        raise SystemExit(f"fork-point not found: {_FORKPOINT}")

    t0 = time.time()
    res = run(
        n_pairs=args.n_pairs, epochs=args.epochs, lr=args.lr,
        seg_weight=args.seg_weight, pose_weight=args.pose_weight,
        margin_tau=args.margin_tau, threshold_pct=args.threshold_pct, seed=args.seed,
        trace_every=args.trace_every, arms_cache_dir=args.arms_cache_dir,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 76)
    print("HiNeRV-GRID vs MARGIN-LEVER  matched-byte d_seg  [contest-CPU advisory] NON-PROMOTABLE")
    print("=" * 76)
    print(f"  n_pairs={res.n_pairs}  epochs={res.epochs}  (elapsed {elapsed:.0f}s)")
    print("  --- per-arm exact d_seg (argmax-flip rate, post eval-roundtrip) ---")
    for a in res.arms:
        print(
            f"    {a['name']:30s} d_seg={a['d_seg']:.6g}  d_pose={a['d_pose']:.6g}  "
            f"dec_params={a['decoder_params']:6d}  sane={a['forward_sane']}"
        )
    print("  --- the decisive N1 comparison ---")
    print(f"  HiNeRV grid forward-parity sane  = {res.hinerv_grid_forward_sane}")
    print(f"  d_seg(L = HNeRV + margin lever)  = {res.d_seg_L_lever:.6g}  (the ceiling baseline)")
    print(f"  d_seg(H best = {res.d_seg_H_best_arm}) = {res.d_seg_H_best:.6g}")
    print(f"  %% reduction vs lever            = {100*res.pct_reduction_vs_lever:+.2f}%%  "
          f"(threshold +{100*res.threshold_pct:.0f}%%)")
    print(f"  grid intrinsic %% (Hoff->H0)     = {100*res.grid_intrinsic_pct_vs_hoff:+.2f}%%  "
          f"(the grid's own d_seg contribution)")
    print(f"  round-trip survived (built-in)   = {res.roundtrip_survived}")
    print(f"  VERDICT                          = {res.verdict}")
    print(f"  {res.authority}")
    print("=" * 76)

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(res), indent=2))
        print(f"[probe] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
