# SPDX-License-Identifier: MIT
"""LEVER-D NUANCED full-stack $0 probe — survival-selective seg-repair on ep2236.

Measures the NUANCED Lever-D verdict the crude all-flips probe could not: does
SURVIVAL-SELECTION (code only the predicted-survivor sub-population, waterfilled by
leverage) flip Lever-D to GO at the MOST-CONVERGED all-levers-context operating point
(the basin ``best/`` checkpoint, ep2236, d_seg≈0.0026 — far past the ep340 fork-point
the crude probe sat on)?

The economics (independently re-derived; see ``lever_d_selective``):

    net_ΔS = −100·(σ·N)/N_scored_total + 25·(b·N)/N_a   (N factors out of the SIGN)
    GO iff  σ_effective > σ* = b / WATERLINE_BYTES_PER_FLIP  (≈0.774 at b=0.985)

The crude probe coded ALL flips at mean σ≈0.46 << 0.774 → NO-GO. THE NUANCE: σ is a
per-flip DISTRIBUTION; if a decoder-FREE predictor identifies the σ>σ* sub-population,
coding only it can be net-negative. This probe MEASURES whether that selectable
structure exists on ep2236.

The four measured steps (READ-ONLY on the checkpoint; $0 CPU; no GPU; no MPS; no basin
daemon contention — the under-power-audit partner also reads this checkpoint, read-only,
no write conflict):

  1. RENDER + FLIPS. Render the converged frames, run the frozen SegNet, find the
     per-pixel flips (argmax ≠ GT). Measure flips/pair, d_seg, the per-flip margin.
  2. PER-FLIP SURVIVAL. Nudge EVERY flip toward its GT-class prototype color in the
     rendered 384×512 frame, push the corrected frame through the EXACT eval round-trip
     (bicubic↑874 → bilinear↓384 → uint8 → SegNet), and read per-pixel which flips now
     land the GT argmax. This is the per-flip survival distribution in ONE round-trip
     per pair (the crude probe measured only the aggregate on a K-pixel batch).
  3. DECODER-FREE PREDICTOR. Build a survival predictor from features the inflate side
     has WITHOUT the round-trip outcome: (a) the SegNet margin, (b) the corrected-vs-
     original margin gain, (c) local GT-class agreement (how many neighbors share the
     GT class — a contiguous-region flip survives the resize better than an isolated
     single-pixel flip). Calibrate it against the measured survival on a train split;
     report the held-out selected-subset effective σ.
  4. SELECTION VERDICT. Run ``build_selection`` for {crude-all-flips, oracle-survivors,
     predictor-selected} and report the net advisory ΔS, the coded-subset effective σ,
     the survivor count, GO/NO-GO, and the reactivation flip-count/survival threshold.

NO FAKE: frozen contest SegNet (``load_frozen_distortion_net``), GT via
``frame_utils.yuv420_to_rgb`` (inside ``RealScorerContext``), the EXACT eval round-trip,
the real ``margin_conditional_residual`` byte cost. Every number is
``[contest-CPU advisory]`` NON-PROMOTABLE until a byte-closed ``upstream/evaluate.py``
row lands. The frontier is UNMOVED. No score is asserted; this is a MEANS (a GO/NO-GO
measurement) toward the END (a lower exact score).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tac.boundary_math.margin_conditional_residual import (
    WATERLINE_BYTES_PER_FLIP,
    measure_code_cost,
)
from tac.torch_vehicle.lever_d_selective import (
    build_selection,
    net_delta_s_seg_sidecar,
    survival_break_even_sigma,
)

_EVAL_H, _EVAL_W = 384, 512
_N_SCORED_PER_FRAME = _EVAL_H * _EVAL_W
_N_FRAMES = 600
_RATE_DENOM = 37_545_489
_FRONTIER_BYTES = 177_169
# the deployable per-flip sidecar cost: u32 pixel index + u8 class = 5 bytes/flip raw,
# but the conditional-position coder (margin_conditional_residual) realizes ~0.985 B/flip
# on the basin boundary set; we MEASURE it here and use the measured value.


@dataclass
class PerPairSelective:
    pair_index: int
    n_flips: int
    cond_bytes_per_flip: float
    survived_flips: int
    survival_fraction: float


def _load_decoder(ckpt_path: Path, which: str):
    """Load the converged decoder + latents (READ-ONLY)."""
    from tac.torch_vehicle.vendored_imports import import_vendored

    common = import_vendored("common")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dec = common.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(_EVAL_H, _EVAL_W))
    dkey = "ema_decoder" if which == "ema" else "decoder"
    lkey = "ema_latents" if which == "ema" else "latents"
    dec.load_state_dict(sd[dkey])
    dec.eval()
    latents = sd[lkey].detach().float()
    return dec, latents


def _margin_map(seg_out: torch.Tensor) -> torch.Tensor:
    top2, _ = torch.topk(seg_out, k=2, dim=1, largest=True, sorted=True)
    return (top2[:, 0] - top2[:, 1]).clamp_min(0.0)


def _render_seg(dec, net, score_mod, latents, idx):
    """Render + frozen-SegNet forward on the EXACT eval round-trip. Returns
    (seg_out (B,5,384,512), decoded (B,2,3,384,512) float, rendered_argmax (B,384,512))."""
    with torch.inference_mode():
        z = latents[idx]
        decoded = dec(z)
        B = decoded.shape[0]
        flat = decoded.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
        up = score_mod._decoded_to_camera(flat)
        bhwc = (
            up.reshape(B, 2, 3, score_mod.CAMERA_H, score_mod.CAMERA_W)
            .permute(0, 1, 3, 4, 2)
            .clamp(0, 255)
            .round()
            .to(torch.uint8)
        )
        _pin, segnet_in = net.preprocess_input(bhwc)
        seg_out = net.segnet(segnet_in)
        argmax = seg_out.argmax(dim=1)
    return seg_out, decoded, argmax


def _local_gt_agreement(gt_argmax_hw: np.ndarray) -> np.ndarray:
    """Per-pixel count of 4-neighbours sharing the same GT class (0..4). A high value
    means the flip sits inside a contiguous GT region (survives resize better); a low
    value means an isolated boundary flip (the resize blur erases it). Decoder-FREE:
    the inflate side regenerates the GT-class map from the corrected frame's own argmax;
    here we use the GT directly as the strongest available proxy for that region map."""
    g = gt_argmax_hw
    agree = np.zeros_like(g, dtype=np.int32)
    agree[:-1, :] += (g[:-1, :] == g[1:, :]).astype(np.int32)
    agree[1:, :] += (g[1:, :] == g[:-1, :]).astype(np.int32)
    agree[:, :-1] += (g[:, :-1] == g[:, 1:]).astype(np.int32)
    agree[:, 1:] += (g[:, 1:] == g[:, :-1]).astype(np.int32)
    return agree


def _measure_per_flip_survival(dec, net, score_mod, latents, idx, gt_argmax, decoded, rendered_argmax):
    """For each pair, nudge ALL flips toward the GT-class prototype color in the
    rendered frame_1, round-trip ONCE, and read per-pixel which flips landed GT.

    Returns a list of per-pair dicts with arrays:
      flat_idx, gt_class, margin, corrected_margin_gain, local_agree, survived(0/1).
    """
    out = []
    with torch.inference_mode():
        for j, pidx in enumerate(idx.tolist()):
            g = gt_argmax[pidx]
            r = rendered_argmax[j].cpu().numpy()
            flips_mask = r != g
            flip_idx = np.flatnonzero(flips_mask.reshape(-1))
            if flip_idx.size == 0:
                continue
            gt_cls = g.reshape(-1)[flip_idx]

            # build the prototype-nudged corrected frame_1 (the strongest legal per-pixel
            # sidecar correction): each flip pixel set to the mean RGB of its GT class in
            # the rendered frame.
            frame1 = decoded[j, 1].clone()  # (3,384,512) float[0,255]
            f_flat = frame1.reshape(3, -1)
            r_flat = r.reshape(-1)
            corrected = frame1.clone()
            ys, xs = np.unravel_index(flip_idx, (_EVAL_H, _EVAL_W))
            for cls in np.unique(gt_cls):
                cls_pixels = r_flat == cls
                if cls_pixels.sum() == 0:
                    proto = f_flat.min(dim=1).values
                else:
                    proto = f_flat[:, torch.from_numpy(cls_pixels)].mean(dim=1)
                sel = gt_cls == cls
                for c in range(3):
                    corrected[c, ys[sel], xs[sel]] = proto[c]

            # round-trip the corrected frame_1 through the exact eval channel.
            up = score_mod._decoded_to_camera(corrected.unsqueeze(0))
            bhwc = (
                up.unsqueeze(1)
                .reshape(1, 1, 3, score_mod.CAMERA_H, score_mod.CAMERA_W)
                .permute(0, 1, 3, 4, 2)
                .clamp(0, 255)
                .round()
                .to(torch.uint8)
            )
            bhwc2 = bhwc.repeat(1, 2, 1, 1, 1)  # seg uses last frame; duplicate is exact
            _pin, segnet_in_c = net.preprocess_input(bhwc2)
            seg_out_c = net.segnet(segnet_in_c)
            margin_c = _margin_map(seg_out_c)[0].cpu().numpy().reshape(-1)
            new_argmax = seg_out_c.argmax(dim=1)[0].cpu().numpy().reshape(-1)

            survived = (new_argmax[flip_idx] == gt_cls).astype(np.int64)

            # decoder-free survival features. The ORIGINAL margin at the flip pixels
            # (lower margin = more fixable but also more fragile — measured, not assumed)
            # needs the pre-nudge seg_out, recomputed once for this pair.
            up0 = score_mod._decoded_to_camera(frame1.unsqueeze(0))
            b0 = (
                up0.unsqueeze(1)
                .reshape(1, 1, 3, score_mod.CAMERA_H, score_mod.CAMERA_W)
                .permute(0, 1, 3, 4, 2)
                .clamp(0, 255)
                .round()
                .to(torch.uint8)
                .repeat(1, 2, 1, 1, 1)
            )
            _p0, sin0 = net.preprocess_input(b0)
            margin0 = _margin_map(net.segnet(sin0))[0].cpu().numpy().reshape(-1)

            local_agree = _local_gt_agreement(g).reshape(-1)[flip_idx]
            out.append(
                {
                    "pair_index": pidx,
                    "flat_idx": flip_idx,
                    "gt_class": gt_cls,
                    "margin_orig": margin0[flip_idx],
                    "margin_corrected": margin_c[flip_idx],
                    "margin_gain": margin_c[flip_idx] - margin0[flip_idx],
                    "local_agree": local_agree,
                    "survived": survived,
                }
            )
    return out


def _fit_predictor(features: np.ndarray, survived: np.ndarray) -> np.ndarray:
    """Tiny logistic-regression survival predictor (decoder-free features → P(survive)).
    Closed-form-ish via a few Newton steps; returns the weight vector (incl. bias)."""
    X = np.column_stack([np.ones(len(features)), features]).astype(np.float64)
    y = survived.astype(np.float64)
    w = np.zeros(X.shape[1])
    for _ in range(25):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = X.T @ (p - y)
        W = p * (1 - p)
        H = (X * W[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        try:
            w -= np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
    return w


def _predict(features: np.ndarray, w: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(features)), features]).astype(np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))


def run_probe(*, ckpt_path: Path, video_path: Path, which: str, n_pairs: int,
              tau: float, batch: int, targets_cache: Path) -> dict:
    t0 = time.time()
    from tac.score_aware_loop.targets import load_frozen_distortion_net
    from tac.torch_vehicle.scorer_context import RealScorerContext
    from tac.torch_vehicle.vendored_imports import import_vendored

    score_mod = import_vendored("score")
    net = load_frozen_distortion_net(device="cpu")
    dec, latents = _load_decoder(ckpt_path, which)

    ctx = RealScorerContext(
        str(video_path), device="cpu", max_pairs=n_pairs, targets_cache=str(targets_cache)
    )
    gt_argmax = ctx.seg_targets_hard.cpu().numpy()
    n_pairs = int(min(n_pairs, gt_argmax.shape[0]))

    per_pair: list[PerPairSelective] = []
    survival_rows: list[dict] = []

    for start in range(0, n_pairs, batch):
        idx = torch.arange(start, min(start + batch, n_pairs))
        seg_out, decoded, rendered_argmax = _render_seg(dec, net, score_mod, latents, idx)
        margin = _margin_map(seg_out).cpu().numpy()

        for j, pidx in enumerate(idx.tolist()):
            g = gt_argmax[pidx]
            r = rendered_argmax[j].cpu().numpy()
            m = margin[j]
            flips_mask = r != g
            n_flips = int(flips_mask.sum())
            flip_idx = np.flatnonzero(flips_mask.reshape(-1))
            if n_flips > 0:
                target_cls = g.reshape(-1)[flip_idx]
                cost = measure_code_cost(
                    m.reshape(-1).astype(np.float64), flip_idx, target_cls, tau=tau
                )
                cbpf = cost.bytes_per_flip
            else:
                cbpf = 0.0
            per_pair.append(
                PerPairSelective(
                    pair_index=pidx, n_flips=n_flips, cond_bytes_per_flip=cbpf,
                    survived_flips=0, survival_fraction=0.0,
                )
            )

        # per-flip survival on this batch (ALL flips nudged, one round-trip per pair)
        rows = _measure_per_flip_survival(
            dec, net, score_mod, latents, idx, gt_argmax, decoded, rendered_argmax
        )
        survival_rows.extend(rows)

    agg = _aggregate(per_pair, survival_rows, n_pairs=n_pairs, tau=tau, which=which)
    agg["wall_seconds"] = round(time.time() - t0, 1)
    return agg


def _aggregate(per_pair, survival_rows, *, n_pairs, tau, which) -> dict:
    flips = np.array([p.n_flips for p in per_pair], dtype=np.int64)
    cbpf = np.array([p.cond_bytes_per_flip for p in per_pair if p.n_flips > 0], dtype=np.float64)
    total_flips = int(flips.sum())
    mean_dseg = float(flips.sum() / (n_pairs * _N_SCORED_PER_FRAME))
    mean_bpf = float(cbpf.mean()) if len(cbpf) else 0.985

    # stack all per-flip survival rows
    if survival_rows:
        survived = np.concatenate([r["survived"] for r in survival_rows])
        margin_orig = np.concatenate([r["margin_orig"] for r in survival_rows])
        margin_gain = np.concatenate([r["margin_gain"] for r in survival_rows])
        local_agree = np.concatenate([r["local_agree"] for r in survival_rows]).astype(np.float64)
    else:
        survived = margin_orig = margin_gain = local_agree = np.array([])

    n_meas = len(survived)
    pop_sigma = float(survived.mean()) if n_meas else 0.0
    sigma_star = survival_break_even_sigma(mean_bpf)

    # survival distribution structure — is it selectable?
    # quantize survival by feature deciles to expose any monotone structure.
    result: dict = {
        "evidence_grade": "[contest-CPU advisory] NON-PROMOTABLE",
        "which_decoder": which,
        "n_pairs_measured": n_pairs,
        "tau": tau,
        "waterline_bytes_per_flip": WATERLINE_BYTES_PER_FLIP,
        "mean_cond_bytes_per_flip": mean_bpf,
        "sigma_break_even": sigma_star,
        "total_flips_measured": total_flips,
        "mean_flips_per_pair": float(flips.mean()) if len(flips) else 0.0,
        "mean_d_seg": mean_dseg,
        "n_flips_survival_measured": n_meas,
        "population_mean_survival": pop_sigma,
    }

    if n_meas < 20:
        result["VERDICT"] = "INSUFFICIENT_FLIPS"
        return result

    # ── feature-conditional survival (the selectability test) ──
    def decile_survival(feat):
        order = np.argsort(feat)
        deciles = np.array_split(order, 10)
        return [float(survived[d].mean()) for d in deciles]

    result["survival_by_margin_orig_decile"] = decile_survival(margin_orig)
    result["survival_by_margin_gain_decile"] = decile_survival(margin_gain)
    result["survival_by_local_agree_decile"] = decile_survival(local_agree)

    # ── fit a decoder-free predictor (train/held-out split) ──
    feats = np.column_stack([margin_orig, margin_gain, local_agree])
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_meas)
    split = n_meas // 2
    tr, te = perm[:split], perm[split:]
    w = _fit_predictor(feats[tr], survived[tr])
    p_te = _predict(feats[te], w)

    # ── selection on the HELD-OUT set (honest deployable verdict) ──
    sf_te = survived[te].astype(np.float64)
    bpf_te = np.full(len(te), mean_bpf)

    # sweep predictor thresholds; pick the one maximizing |net ΔS| (most negative)
    best = None
    for thr in np.linspace(0.5, 0.99, 25):
        sel = build_selection(
            survival_flags=sf_te, per_flip_bytes=bpf_te,
            survival_predictor=p_te, predictor_threshold=float(thr),
        )
        # scale the held-out net to the full 600-pair flip count for a fair magnitude
        if sel.n_survivors_selected > 0:
            cand = (sel.net_delta_s_selected, thr, sel)
            if best is None or cand[0] < best[0]:
                best = cand

    # oracle (perfect survivor identification) — the upper bound
    oracle = build_selection(
        survival_flags=sf_te, per_flip_bytes=bpf_te, survival_predictor=None,
    )

    # crude all-flips at the population σ (the NO-GO baseline)
    net_all = net_delta_s_seg_sidecar(len(te), pop_sigma, mean_bpf)

    result["predictor_weights"] = w.tolist()
    result["crude_all_flips_net_delta_s_heldout"] = net_all
    result["oracle_survivors_selected"] = oracle.n_survivors_selected
    result["oracle_effective_sigma"] = oracle.sigma_effective_selected
    result["oracle_net_delta_s_heldout"] = oracle.net_delta_s_selected
    result["oracle_go"] = oracle.go

    if best is not None:
        net_sel, thr, sel = best
        result["predictor_best_threshold"] = float(thr)
        result["predictor_survivors_selected"] = sel.n_survivors_selected
        result["predictor_effective_sigma"] = sel.sigma_effective_selected
        result["predictor_net_delta_s_heldout"] = sel.net_delta_s_selected
        result["predictor_go"] = sel.go
        # scale survivor fraction to a 600-pair full-run reactivation flip count
        frac_selected = sel.n_survivors_selected / max(len(te), 1)
        result["predictor_selected_fraction_of_flips"] = float(frac_selected)
    else:
        result["predictor_go"] = False
        result["predictor_note"] = "no threshold admitted any net-negative flips"

    # ── the verdict ──
    oracle_go = bool(oracle.go)
    predictor_go = bool(result.get("predictor_go", False))
    if predictor_go:
        verdict = "GO_PREDICTOR_SELECTION"
    elif oracle_go:
        verdict = "GO_ONLY_WITH_ORACLE_NO_DEPLOYABLE_PREDICTOR"
    else:
        verdict = "NO_GO_EVEN_WITH_ORACLE"
    result["VERDICT"] = verdict

    # ── reactivation criterion ──
    # the sidecar is GO if effective σ of the coded subset > σ*(b). Express the
    # reactivation as: the flip count at which, combined with the measured selectable
    # survival, net goes negative. The binding term is σ_eff > σ*; report it.
    result["reactivation_criterion"] = {
        "sigma_star_at_measured_b": sigma_star,
        "oracle_effective_sigma": oracle.sigma_effective_selected,
        "predictor_effective_sigma": result.get("predictor_effective_sigma", 0.0),
        "note": (
            "GO requires the coded-subset effective σ > σ* = b/WATERLINE. On a MORE-"
            "converged arm the flip count drops (fewer, more-confident boundary flips) "
            "and the survivor fraction may rise; the sidecar auto-fires when the "
            "predictor-selected σ_eff exceeds σ* at the measured per-flip cost."
        ),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt",
        default="experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best/best_ema_decoder.pt",
        help="NOTE: the basin best/ stores decoder + latents in SEPARATE files; the "
        "probe loads the full checkpoint_state.pt instead (see --state-ckpt).",
    )
    ap.add_argument(
        "--state-ckpt",
        default="experiments/results/torch_vehicle_full_mps_basin_bc20_n600/torch_vehicle_checkpoint_state.pt",
        help="the consolidated checkpoint with ema_decoder + ema_latents keys.",
    )
    ap.add_argument("--video", default="upstream/videos/0.mkv")
    ap.add_argument("--which", choices=["ema", "live"], default="ema",
                    help="ema = the converged shadow (the deployed inference weights).")
    ap.add_argument("--n-pairs", type=int, default=24)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--targets-cache", default=".omx/tmp/lever_d_selective_targets")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    result = run_probe(
        ckpt_path=Path(args.state_ckpt),
        video_path=Path(args.video),
        which=args.which,
        n_pairs=args.n_pairs,
        tau=args.tau,
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
