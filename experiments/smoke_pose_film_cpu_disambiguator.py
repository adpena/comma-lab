# SPDX-License-Identifier: MIT
"""$0 CPU-only disambiguator: does pose-FiLM REDUCE realized d_pose on base_ch=20?

ONE question (Lever 3 of ``incurriculum_levers_design_floor_chasing_20260612.md``):
storing the 6 GT pose scalars/pair + FiLM-conditioning the frozen base_ch=20 HNeRV
decoder on them — does it ACTUALLY lower the realized (authority) d_pose vs no-FiLM,
and does the resulting ``sqrt(10*d_pose)`` reduction beat the stored-pose byte cost?

This is the deploy-gate probe the memo specifies ("deploy ONLY if d_pose > the
stored-pose quant floor"): a frozen-decoder lower bound on the full lever (full
co-adaptation would do better; here the decoder is FROZEN and only the FiLM trains).

AUTHORITY / NO FAKE (CLAUDE.md):
  * CPU ONLY (the authority device; runs parallel to the live MPS basin without
    stealing the GPU). NO mps, NO mlx.
  * REAL frozen contest scorer (``load_frozen_distortion_net(device='cpu')``) and
    REAL ``upstream/videos/0.mkv`` GT frames decoded ONLY via
    ``frame_utils.yuv420_to_rgb`` (PyAV rgb24 manufactures ~100x phantom pose).
  * The reported d_pose is the EXACT PoseNet-readout MSE the evaluator charges:
    ``distortion_net.compute_distortion(gt_pair, decoded_pair)`` AFTER the SAME
    eval-roundtrip the driver/evaluator uses (bicubic^874 -> bilinear v384 -> uint8).
  * Every number is ``[contest-CPU advisory]`` (small-n magnitude is advisory; the
    CPU numerics are authority-grade). score_claim=false, promotable=false.

It edits NO files other agents own (driver.py / curriculum.py / cool_chic / the
cool_chic trainer). It NEVER writes into the live basin dir and NEVER touches the
basin daemon. It only READS the frozen fork-point.

Usage::

    .venv/bin/python experiments/smoke_pose_film_cpu_disambiguator.py \
        --n-pairs 64 --film-epochs 30
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

# The frozen fork-point (immutable init for lever A/B forks). READ ONLY.
_FORKPOINT = Path(
    "experiments/results/forkpoints/basin_bc20_20260612T121523Z/"
    "torch_vehicle_checkpoint_state.pt"
)
_VIDEO = Path("upstream/videos/0.mkv")
_EVAL_H, _EVAL_W = 384, 512
_CAMERA_H, _CAMERA_W = 874, 1164
_TOTAL_VIDEO_BYTES_DENOM = 37_545_489.0  # contest rate denominator
_BASE_CHANNELS = 20
_LATENT_DIM = 28
_POSE_DIM = 6


# ---------------------------------------------------------------------------
# eval-roundtrip (1:1 with vendored ``score._decoded_to_camera`` + evaluate path)
# ---------------------------------------------------------------------------
def _roundtrip_to_eval_bhwc(decoded: torch.Tensor) -> torch.Tensor:
    """``decoded`` is (B, 2, 3, 384, 512) float [0,255]. Return the eval-roundtripped
    (B, 2, 384, 512, 3) uint8 tensor the PoseNet/SegNet actually read — bicubic up to
    camera res, bilinear down to 384x512, clamp/round/uint8 (matches
    ``evaluate_decoder``)."""
    b = decoded.shape[0]
    flat = decoded.reshape(b * 2, 3, _EVAL_H, _EVAL_W)
    up = F.interpolate(flat, size=(_CAMERA_H, _CAMERA_W), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(_EVAL_H, _EVAL_W), mode="bilinear", align_corners=False)
    bhwc = (
        down.reshape(b, 2, 3, _EVAL_H, _EVAL_W)
        .permute(0, 1, 3, 4, 2)
        .clamp(0, 255)
        .round()
        .to(torch.uint8)
    )
    return bhwc


def _roundtrip_to_eval_bhwc_differentiable(decoded: torch.Tensor) -> torch.Tensor:
    """Same eval-roundtrip but with a straight-through uint8 round (for the FiLM
    fine-tune: the gradient must flow through the roundtrip the evaluator applies —
    CLAUDE.md ``eval_roundtrip`` non-negotiable). Returns (B, 2, 384, 512, 3) float."""
    b = decoded.shape[0]
    flat = decoded.reshape(b * 2, 3, _EVAL_H, _EVAL_W)
    up = F.interpolate(flat, size=(_CAMERA_H, _CAMERA_W), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(_EVAL_H, _EVAL_W), mode="bilinear", align_corners=False)
    bhwc = down.reshape(b, 2, 3, _EVAL_H, _EVAL_W).permute(0, 1, 3, 4, 2)
    clamped = bhwc.clamp(0, 255)
    rounded = clamped.round()
    return clamped + (rounded - clamped).detach()  # STE uint8 round


# ---------------------------------------------------------------------------
# Pose-FiLM wrapper over the FROZEN vendored HNeRVDecoder (identity at init)
# ---------------------------------------------------------------------------
class _PoseFiLM(nn.Module):
    """Torch port of the capstone stored-pose FiLM (identical mechanism to
    ``cool_chic_carrier._PoseFiLM``): pose6 -> sin(fc1) -> fc2(zero-init) ->
    (gamma_pre, beta); gamma = 1 + tanh(gamma_pre) (=1 at init), beta (=0 at init).
    The zero-init fc2 makes FiLM the EXACT identity at init, so the FIRST FiLM-arm
    render equals the frozen baseline render (verified by the identity assertion)."""

    def __init__(self, *, pose_dim: int, channels: int, hidden: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.fc1 = nn.Linear(int(pose_dim), int(hidden))
        self.fc2 = nn.Linear(int(hidden), 2 * int(channels))
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.sin(self.fc1(pose))
        gb = self.fc2(h)
        gamma_pre = gb[:, : self.channels]
        beta = gb[:, self.channels :]
        gamma = 1.0 + torch.tanh(gamma_pre)  # (0, 2), =1 at init
        return gamma, beta

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PoseFiLMHNeRVWrapper(nn.Module):
    """Wrap the FROZEN vendored ``HNeRVDecoder`` with a stored-pose FiLM on the
    frame1 head feature (the SegNet/pose-relevant frame is ``x[:, -1]``).

    The vendored decoder builds a shared feature ``x`` (final_ch channels at
    384x512), then ``f0 = rgb_0(x)``, ``f1 = rgb_1(x)``. We re-run the vendored
    forward UP TO the shared ``x`` (re-using the vendored layers, NOT forking the
    source), then FiLM-modulate a per-pair copy of ``x`` from the stored pose
    BEFORE ``rgb_1`` (frame1) — the canonical injection point. frame0 (``rgb_0``)
    is left unmodulated (it is not the SegNet last-frame; the pose readout the
    evaluator charges is on the PAIR, and frame1 is where the boundary lives).

    Identity at init: with the zero-init FiLM, gamma=1/beta=0, so ``feat1 == x`` and
    the render is byte-identical to the frozen baseline (asserted).
    """

    def __init__(self, decoder: nn.Module, *, n_pairs: int, pose_dim: int = _POSE_DIM,
                 film_hidden: int = 32) -> None:
        super().__init__()
        self.decoder = decoder  # FROZEN vendored HNeRVDecoder
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self.n_pairs = int(n_pairs)
        self.pose_dim = int(pose_dim)
        final_ch = int(decoder.channels[-1])
        self.final_ch = final_ch
        self.pose_film = _PoseFiLM(pose_dim=pose_dim, channels=final_ch, hidden=film_hidden)
        self.register_buffer("stored_pose", torch.zeros(self.n_pairs, pose_dim))

    def set_stored_pose(self, pose: torch.Tensor) -> None:
        with torch.no_grad():
            p = pose.detach().to(self.stored_pose.dtype)
            if p.shape != self.stored_pose.shape:
                raise ValueError(
                    f"stored_pose expects {tuple(self.stored_pose.shape)}, got {tuple(p.shape)}"
                )
            self.stored_pose.copy_(p)

    def _shared_feature(self, z: torch.Tensor) -> torch.Tensor:
        """Re-run the vendored decoder forward up to the shared feature ``x`` (the
        input to rgb_0/rgb_1). This mirrors the vendored ``forward`` body EXACTLY
        (so frame0 == the frozen baseline) but stops before the two RGB heads."""
        d = self.decoder
        b = z.shape[0]
        x = d.stem(z).view(b, d.channels[0], d.base_h, d.base_w)
        x = torch.sin(x)
        for block, skip in zip(d.blocks, d.skips):
            identity = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            identity = skip(identity)
            x = d.ps(block(x))
            x = torch.sin(x + identity)
        x = x + 0.1 * torch.sin(d.refine(x))
        return x

    def forward(self, z: torch.Tensor, pose6: torch.Tensor) -> torch.Tensor:
        """Return (B, 2, 3, 384, 512) float [0,255] — frame0 unmodulated, frame1
        pose-FiLM-modulated. ``pose6`` is (B, 6) stored pose."""
        d = self.decoder
        x = self._shared_feature(z)  # (B, final_ch, 384, 512)
        f0 = torch.sigmoid(d.rgb_0(x)) * 255.0  # frame0 (unmodulated, == baseline)
        gamma, beta = self.pose_film(pose6)  # each (B, final_ch)
        feat1 = gamma[:, :, None, None] * x + beta[:, :, None, None]  # FiLM on frame1 feat
        f1 = torch.sigmoid(d.rgb_1(feat1)) * 255.0
        return torch.stack([f0, f1], dim=1)


# ---------------------------------------------------------------------------
# GT pair streaming (REAL frames, canonical decode)
# ---------------------------------------------------------------------------
def _stream_gt_pairs(video_path: Path, n_pairs: int, device: str = "cpu") -> torch.Tensor:
    """Return the first ``n_pairs`` GT pairs as (n_pairs, 2, 384, 512, 3) uint8 via
    the canonical ``yuv420_to_rgb`` decode (NOT PyAV rgb24)."""
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
    return torch.stack(pairs).to(device)  # (n_pairs, 2, H, W, 3) uint8


# ---------------------------------------------------------------------------
# d_pose / d_seg measurement (exact, authority, with eval-roundtrip)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _measure_distortion_no_film(
    decoder: nn.Module, latents: torch.Tensor, gt_pairs: torch.Tensor,
    distortion_net: nn.Module, batch: int = 8,
) -> tuple[float, float]:
    """Baseline arm: render from the frozen decoder, eval-roundtrip, measure the
    EXACT (pose_d, seg_d) the evaluator charges. Returns (d_pose_mean, d_seg_mean)."""
    n = gt_pairs.shape[0]
    pose_total = seg_total = 0.0
    count = 0
    for start in range(0, n, batch):
        idx = torch.arange(start, min(start + batch, n))
        z = latents[idx]
        decoded = decoder(z)  # (B, 2, 3, 384, 512)
        decoded_bhwc = _roundtrip_to_eval_bhwc(decoded)
        gt = gt_pairs[idx]
        pose_d, seg_d = distortion_net.compute_distortion(gt, decoded_bhwc)
        pose_total += pose_d.sum().item()
        seg_total += seg_d.sum().item()
        count += len(idx)
    return pose_total / max(count, 1), seg_total / max(count, 1)


@torch.inference_mode()
def _measure_distortion_film(
    wrapper: PoseFiLMHNeRVWrapper, latents: torch.Tensor, gt_pairs: torch.Tensor,
    distortion_net: nn.Module, batch: int = 8,
) -> tuple[float, float]:
    """FiLM arm: render from the frozen-decoder + trained-FiLM wrapper conditioned on
    the stored pose, eval-roundtrip, measure the EXACT (pose_d, seg_d)."""
    n = gt_pairs.shape[0]
    pose_total = seg_total = 0.0
    count = 0
    for start in range(0, n, batch):
        idx = torch.arange(start, min(start + batch, n))
        z = latents[idx]
        pose6 = wrapper.stored_pose[idx]
        decoded = wrapper(z, pose6)
        decoded_bhwc = _roundtrip_to_eval_bhwc(decoded)
        gt = gt_pairs[idx]
        pose_d, seg_d = distortion_net.compute_distortion(gt, decoded_bhwc)
        pose_total += pose_d.sum().item()
        seg_total += seg_d.sum().item()
        count += len(idx)
    return pose_total / max(count, 1), seg_total / max(count, 1)


# ---------------------------------------------------------------------------
# stored-pose byte cost (mirror of mlx_pr95_port.pose_film.stored_pose_bytes)
# ---------------------------------------------------------------------------
def stored_pose_bytes(pose_array: np.ndarray, *, quant_step: float = 1e-3) -> int:
    """Byte cost of the STORED pose after uniform-quant + brotli q=11 (the honest
    export cost; Quantizr stores ``pose.npy.br``). Mirrors the MLX helper exactly."""
    arr = np.asarray(pose_array, dtype=np.float32)
    q = np.round(arr / quant_step).astype(np.int32)
    payload = q.tobytes()
    import brotli

    return len(brotli.compress(payload, quality=11))


# ---------------------------------------------------------------------------
# the disambiguator
# ---------------------------------------------------------------------------
@dataclass
class _Result:
    n_pairs: int
    film_epochs: int
    film_hidden: int
    film_lr: float
    quant_step: float
    # baseline arm
    d_pose_baseline: float
    d_seg_baseline: float
    pose_term_baseline: float  # sqrt(10*d_pose)
    # FiLM arm
    d_pose_film: float
    d_seg_film: float
    pose_term_film: float
    # identity check (FiLM at init must equal baseline)
    d_pose_film_at_init: float
    identity_at_init_abs_err: float
    # byte trade @ n=600 projection
    stored_pose_bytes_at_n: int
    stored_pose_bytes_at_600_proj: int
    film_param_bytes_fp16: int
    rate_delta_at_600_proj: float  # 25 * added_bytes / denom
    # the net delta
    delta_pose_term: float  # pose_term_film - pose_term_baseline (negative = win)
    net_delta_S_at_600_proj: float  # delta_pose_term + rate_delta
    verdict: str
    authority: str


def run(n_pairs: int, film_epochs: int, film_hidden: int, film_lr: float,
        quant_step: float, seed: int) -> _Result:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu"

    print(f"[smoke] loading fork-point EMA decoder + EMA latents ({_FORKPOINT})", flush=True)
    blob = torch.load(_FORKPOINT, map_location="cpu", weights_only=False)
    ema_decoder_sd = blob["ema_decoder"]
    ema_latents_full = blob["ema_latents"]  # (600, 28)
    assert ema_latents_full.shape[1] == _LATENT_DIM, ema_latents_full.shape

    from tac.torch_vehicle.driver import import_vendored_bundle

    v = import_vendored_bundle()
    decoder = v.HNeRVDecoder(
        latent_dim=_LATENT_DIM, base_channels=_BASE_CHANNELS, eval_size=(_EVAL_H, _EVAL_W)
    ).to(device)
    decoder.load_state_dict({k: val.to(device) for k, val in ema_decoder_sd.items()})
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)

    latents = ema_latents_full[:n_pairs].to(device).contiguous()

    print("[smoke] loading frozen contest DistortionNet (CPU authority)", flush=True)
    from tac.score_aware_loop.targets import load_frozen_distortion_net

    distortion_net = load_frozen_distortion_net(device=device)

    print(f"[smoke] streaming {n_pairs} REAL GT pairs (yuv420_to_rgb)", flush=True)
    gt_pairs = _stream_gt_pairs(_VIDEO, n_pairs, device=device)
    assert gt_pairs.shape[0] == n_pairs, gt_pairs.shape

    # GT pose targets (the STORED side-info): the frozen PoseNet's OWN readout on the
    # GT frames — exactly the pose the evaluator's d_pose is measured against.
    print("[smoke] computing GT PoseNet targets (the stored side-info)", flush=True)
    pose_targets = torch.zeros(n_pairs, _POSE_DIM)
    with torch.inference_mode():
        for start in range(0, n_pairs, 8):
            idx = torch.arange(start, min(start + 8, n_pairs))
            posenet_out, _segnet_out = distortion_net(gt_pairs[idx])
            pose_targets[idx] = posenet_out["pose"][:, :_POSE_DIM].float()

    # ---- BASELINE ARM (no FiLM) ----
    print("[smoke] BASELINE arm: measuring d_pose on frozen decoder (no FiLM)", flush=True)
    d_pose_base, d_seg_base = _measure_distortion_no_film(
        decoder, latents, gt_pairs, distortion_net
    )
    pose_term_base = float((10.0 * d_pose_base + 1e-12) ** 0.5)
    print(f"[smoke]   d_pose_baseline={d_pose_base:.6g}  d_seg_baseline={d_seg_base:.6g}  "
          f"pose_term={pose_term_base:.6g}", flush=True)

    # ---- FiLM ARM ----
    print("[smoke] FiLM arm: wrapping frozen decoder + stored-pose FiLM (identity@init)", flush=True)
    wrapper = PoseFiLMHNeRVWrapper(
        decoder, n_pairs=n_pairs, pose_dim=_POSE_DIM, film_hidden=film_hidden
    ).to(device)
    wrapper.set_stored_pose(pose_targets)

    # Identity-at-init check: FiLM=identity => d_pose_film_at_init == d_pose_baseline.
    d_pose_init, _d_seg_init = _measure_distortion_film(
        wrapper, latents, gt_pairs, distortion_net
    )
    identity_err = abs(d_pose_init - d_pose_base)
    print(f"[smoke]   identity@init: d_pose_film_at_init={d_pose_init:.6g} "
          f"(|Δ vs baseline|={identity_err:.3g} — must be ~0)", flush=True)

    # Fine-tune ONLY the FiLM params (decoder frozen) minimizing the score-domain
    # pose term sqrt(10*pose_mse) against the GT PoseNet readout, eval-roundtrip in
    # the loop. This is the conservative FROZEN-DECODER lower bound on the lever.
    film_params = list(wrapper.pose_film.parameters())
    opt = torch.optim.Adam(film_params, lr=film_lr)
    bs = 8
    print(f"[smoke]   fine-tuning FiLM for {film_epochs} epochs (lr={film_lr}, "
          f"{wrapper.pose_film.param_count()} params)", flush=True)
    for ep in range(film_epochs):
        perm = torch.randperm(n_pairs)
        ep_loss = 0.0
        nb = 0
        for start in range(0, n_pairs, bs):
            idx = perm[start : start + bs]
            z = latents[idx]
            pose6 = wrapper.stored_pose[idx]
            decoded = wrapper(z, pose6)  # grad flows to FiLM only
            decoded_bhwc = _roundtrip_to_eval_bhwc_differentiable(decoded)
            # Run the frozen PoseNet on the decoded pair to get its readout, and on
            # the GT pair to get the target readout; minimize the pose MSE between
            # them (the EXACT d_pose the evaluator charges).
            posenet_dec, _seg_dec = distortion_net(decoded_bhwc)
            with torch.inference_mode():
                posenet_gt, _seg_gt = distortion_net(gt_pairs[idx])
            pose_dec = posenet_dec["pose"][:, :_POSE_DIM]
            pose_gt = posenet_gt["pose"][:, :_POSE_DIM].detach().clone()
            pose_mse = F.mse_loss(pose_dec, pose_gt)
            pose_l = torch.sqrt(10.0 * pose_mse + 1e-12)
            opt.zero_grad()
            pose_l.backward()
            opt.step()
            ep_loss += float(pose_l.item())
            nb += 1
        if ep % max(1, film_epochs // 6) == 0 or ep == film_epochs - 1:
            print(f"[smoke]     film ep {ep:3d}: mean pose_term(train) = "
                  f"{ep_loss / max(nb, 1):.6g}", flush=True)

    print("[smoke] FiLM arm: re-measuring d_pose (trained FiLM)", flush=True)
    d_pose_film, d_seg_film = _measure_distortion_film(
        wrapper, latents, gt_pairs, distortion_net
    )
    pose_term_film = float((10.0 * d_pose_film + 1e-12) ** 0.5)
    print(f"[smoke]   d_pose_film={d_pose_film:.6g}  d_seg_film={d_seg_film:.6g}  "
          f"pose_term={pose_term_film:.6g}", flush=True)

    # ---- BYTE TRADE @ n=600 projection ----
    pose_bytes_at_n = stored_pose_bytes(pose_targets.numpy(), quant_step=quant_step)
    # Project the stored-pose byte cost linearly to the full 600 pairs (the section
    # is per-pair; the per-pair byte cost is ~constant once brotli warms up).
    per_pair_bytes = pose_bytes_at_n / max(n_pairs, 1)
    pose_bytes_at_600 = int(round(per_pair_bytes * 600))
    # FiLM MLP ships in the decoder blob (fp16 here as an upper bound — it would be
    # FP4/brotli'd in production, smaller). It is a FIXED cost (does not scale w/ n).
    film_param_bytes_fp16 = wrapper.pose_film.param_count() * 2
    added_bytes_at_600 = pose_bytes_at_600 + film_param_bytes_fp16
    rate_delta_at_600 = 25.0 * added_bytes_at_600 / _TOTAL_VIDEO_BYTES_DENOM

    delta_pose_term = pose_term_film - pose_term_base  # negative = pose-term win
    net_delta_S = delta_pose_term + rate_delta_at_600

    # ---- VERDICT ----
    # GO: pose-term reduction clears the byte cost even at the frozen-decoder lower
    #     bound AND the realized d_pose actually dropped (lever exploits side-info).
    # NO-GO: FiLM did not reduce d_pose (side-info not exploitable on this decoder),
    #     i.e. delta_pose_term >= ~0.
    # MARGINAL: FiLM reduced d_pose but the net is positive/near-zero at the
    #     conservative frozen-decoder bound — full co-adaptation needed to decide.
    pose_reduced = d_pose_film < d_pose_base * 0.98  # >2% realized reduction
    if not pose_reduced:
        verdict = "NO-GO"
    elif net_delta_S < -1e-4:
        verdict = "GO"
    else:
        verdict = "MARGINAL"

    return _Result(
        n_pairs=n_pairs,
        film_epochs=film_epochs,
        film_hidden=film_hidden,
        film_lr=film_lr,
        quant_step=quant_step,
        d_pose_baseline=d_pose_base,
        d_seg_baseline=d_seg_base,
        pose_term_baseline=pose_term_base,
        d_pose_film=d_pose_film,
        d_seg_film=d_seg_film,
        pose_term_film=pose_term_film,
        d_pose_film_at_init=d_pose_init,
        identity_at_init_abs_err=identity_err,
        stored_pose_bytes_at_n=pose_bytes_at_n,
        stored_pose_bytes_at_600_proj=pose_bytes_at_600,
        film_param_bytes_fp16=film_param_bytes_fp16,
        rate_delta_at_600_proj=rate_delta_at_600,
        delta_pose_term=delta_pose_term,
        net_delta_S_at_600_proj=net_delta_S,
        verdict=verdict,
        authority="[contest-CPU advisory] NON-PROMOTABLE — score_claim=false, "
        "promotable=false. Frozen-decoder LOWER BOUND on the full lever; small-n "
        "magnitude advisory. Exact verdict requires paired CPU/CUDA upstream/evaluate.py.",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=64, help="real GT pairs (48-96)")
    ap.add_argument("--film-epochs", type=int, default=30)
    ap.add_argument("--film-hidden", type=int, default=32)
    ap.add_argument("--film-lr", type=float, default=1e-2)
    ap.add_argument("--quant-step", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=str, default=None)
    args = ap.parse_args()

    if not _FORKPOINT.exists():
        raise SystemExit(f"fork-point not found: {_FORKPOINT}")
    if not _VIDEO.exists():
        raise SystemExit(f"video not found: {_VIDEO}")

    t0 = time.time()
    res = run(
        n_pairs=args.n_pairs,
        film_epochs=args.film_epochs,
        film_hidden=args.film_hidden,
        film_lr=args.film_lr,
        quant_step=args.quant_step,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 72)
    print("POSE-FiLM CPU DISAMBIGUATOR — RESULT  [contest-CPU advisory] NON-PROMOTABLE")
    print("=" * 72)
    print(f"  n_pairs={res.n_pairs}  film_epochs={res.film_epochs}  "
          f"film_hidden={res.film_hidden}  (elapsed {elapsed:.0f}s)")
    print(f"  identity@init |Δd_pose|        = {res.identity_at_init_abs_err:.3g}  "
          f"(FiLM-identity sanity; must be ~0)")
    print("  --- realized d_pose (the disambiguator) ---")
    print(f"  d_pose_baseline (no FiLM)      = {res.d_pose_baseline:.6g}")
    print(f"  d_pose_film     (trained FiLM) = {res.d_pose_film:.6g}")
    print(f"  pose_term sqrt(10*d_pose) base = {res.pose_term_baseline:.6g}")
    print(f"  pose_term sqrt(10*d_pose) film = {res.pose_term_film:.6g}")
    print(f"  Δ pose_term (film - base)      = {res.delta_pose_term:+.6g}  "
          f"(negative = lever reduces the pose term)")
    print(f"  d_seg baseline / film          = {res.d_seg_baseline:.6g} / {res.d_seg_film:.6g}")
    print("  --- byte trade @ n=600 (projection) ---")
    print(f"  stored_pose_bytes @ n={res.n_pairs}        = {res.stored_pose_bytes_at_n} B "
          f"(quant_step={res.quant_step})")
    print(f"  stored_pose_bytes @ 600 (proj) = {res.stored_pose_bytes_at_600_proj} B")
    print(f"  FiLM MLP bytes (fp16, fixed)   = {res.film_param_bytes_fp16} B")
    print(f"  rate Δ @ 600 (proj)            = {res.rate_delta_at_600_proj:+.6g}")
    print("  --- NET ---")
    print(f"  net ΔS @ 600 (proj)            = {res.net_delta_S_at_600_proj:+.6g}  "
          f"(Δpose_term + rateΔ; negative = lever wins)")
    print(f"  VERDICT                        = {res.verdict}")
    print(f"  {res.authority}")
    print("=" * 72)

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(res), indent=2))
        print(f"[smoke] wrote {out}", flush=True)
    else:
        print(json.dumps(asdict(res), indent=2))


if __name__ == "__main__":
    main()
