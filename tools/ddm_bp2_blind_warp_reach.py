#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_bp2 — is the scorer-blind 22.70% of the camera frame a POSE ACTUATOR?

PRE-REGISTERED FALSIFIER (charter, 2026-08-02).  The family closes at FORMULATION
scope for the v4d vehicle if EITHER:
  (F1) the blind set's overlap with v4d's frame_0 warp read-set is < 5%, OR
  (F2) the achievable |delta d_pose| from a blind-only perturbation is < 1e-4.

THREE MODES, each self-verifying:

  blind-verify  Re-derives the blind set from the REAL torch operator by autograd
                (D is linear with non-negative weights => blind == zero column sum),
                cross-checks it against ``ddm_ll1_window_solve.blind_mask``, then
                EMPIRICALLY confirms invisibility through the canonical
                ``DistortionNet.preprocess_input`` -- WITH two positive controls
                (a single READ pixel +40 must change both scorer inputs; the same
                +40 on a BLIND pixel must not).  A guard never shown to fire is
                untrusted.

  overlap       Per pair, builds the EXACT tap decomposition of the v4d frame_0
                map (selector two-plane compose + rung-A rolling-shutter blend +
                photometric ``a``) and computes the column mass of ``D . (a*M)`` on
                blind columns: the fraction of frame_0's scorer-visible signal that
                is SOURCED from blind frame_1 pixels.  Closure check: total mass
                must equal ``|a| * 196608``.

  reach         The measurement that decides F2.  Takes one gradient of the REAL
                d_pose w.r.t. the frame_0 camera raster, pulls it back through the
                warp adjoint to frame_1's blind pixels, and line-searches a signed
                step.  Every reported score is RE-MEASURED through the canonical
                unpatched upstream scorer on the REAL re-rendered f0 -- the gradient
                is only a search direction, never an authority.  Runs an ASCENT
                control at each step size (the channel must be steerable in BOTH
                directions or the "reduction" is noise) and asserts d_seg is
                bit-identical.

AXIS: [macOS-CPU advisory] NON-PROMOTABLE.  Frozen CPU-torch scorers on decoded
camera rasters; this is not ``upstream/evaluate.py`` on an archive, so no row here
is a score claim.  score_claim=false.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/pfs1"
)
VENDORED_DEPS = (
    "ddm_r7_token_coder.py",
    "ddm_tr1_runtime.py",
    "pfs1_warp_receiver.py",
    "repair_entropy_coder_runtime_adapters.py",
)
RECEIVER = REPO_ROOT / "experiments" / "inflate_runner_v4d.py"
CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512
N_SCORER_PX = SEG_H * SEG_W


# --------------------------------------------------------------------------- io
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_receiver(stage: Path, archive: Path, template: Path) -> Path:
    """Materialise the vendored decode substrate (no tac) + extract the archive."""
    stage.mkdir(parents=True, exist_ok=True)
    for dep in VENDORED_DEPS:
        shutil.copy(template / dep, stage / dep)
    shutil.copy(RECEIVER, stage / RECEIVER.name)
    arch_dir = stage / "archive"
    if arch_dir.exists():
        shutil.rmtree(arch_dir)
    arch_dir.mkdir()
    from tac.submission_archive import safe_extract_zip

    safe_extract_zip(archive, arch_dir)
    sys.path.insert(0, str(stage))
    return arch_dir


def gt_pair_stream(video: Path, n_pairs: int):
    """Yield (pair_index, gt0, gt1) using the CANONICAL decode only.

    ``frame_utils.yuv420_to_rgb`` on the PyAV FRAME object.  PyAV rgb24 manufactures
    ~100x phantom pose (CLAUDE.md forbidden pattern), so never that.
    """
    import av
    from frame_utils import yuv420_to_rgb

    container = av.open(str(video))
    buf: list[np.ndarray] = []
    emitted = 0
    for frame in container.decode(video=0):
        rgb = yuv420_to_rgb(frame)
        arr = np.asarray(rgb.numpy() if hasattr(rgb, "numpy") else rgb, dtype=np.uint8)
        if arr.shape[:2] != (CAMERA_H, CAMERA_W):
            continue
        buf.append(arr)
        if len(buf) == 2:
            yield emitted, buf[0], buf[1]
            emitted += 1
            buf = []
            if emitted >= n_pairs:
                break
    container.close()


# ------------------------------------------------------------------- scorers
class Scorers:
    """Canonical frozen CPU scorers.  Authority = the UNPATCHED upstream path."""

    def __init__(self, threads: int) -> None:
        from modules import DistortionNet, posenet_sd_path, segnet_sd_path

        torch.set_num_threads(threads)
        self.net = DistortionNet().eval()
        self.net.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))

    @staticmethod
    def _batch(pair_hwc: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(pair_hwc, dtype=np.uint8))[None]

    def distortion(self, gt_pair: np.ndarray, cm_pair: np.ndarray) -> tuple[float, float]:
        with torch.inference_mode():
            pose, seg = self.net.compute_distortion(
                self._batch(gt_pair), self._batch(cm_pair)
            )
        return float(pose[0]), float(seg[0])

    def pose_out(self, pair_hwc: np.ndarray):
        """Canonical (UNPATCHED upstream) PoseNet output for one pair."""
        with torch.inference_mode():
            x = self.net.preprocess_input(self._batch(pair_hwc))[0]
            return self.net.posenet(x)

    def dpose_from(self, gt_out, cm_pair: np.ndarray) -> float:
        """d_pose against a cached GT PoseNet output — skips the GT + SegNet forwards.

        Uses ``PoseNet.compute_distortion`` itself, so it IS the canonical arithmetic;
        the caller asserts equality against :meth:`distortion` on every pair's base arm
        (a per-pair guard that the fast path never drifts from the authority).
        """
        with torch.inference_mode():
            return float(self.net.posenet.compute_distortion(gt_out, self.pose_out(cm_pair))[0])

    def seg_out(self, pair_hwc: np.ndarray):
        """Canonical (UNPATCHED upstream) SegNet output for one pair."""
        with torch.inference_mode():
            x = self.net.preprocess_input(self._batch(pair_hwc))[1]
            return self.net.segnet(x)

    def dseg_from(self, gt_seg_out, cm_pair: np.ndarray) -> float:
        with torch.inference_mode():
            return float(self.net.segnet.compute_distortion(gt_seg_out, self.seg_out(cm_pair))[0])

    def _pose_head(self, cam_bthwc: torch.Tensor) -> torch.Tensor:
        """Differentiable PoseNet head.  Forward is bit-identical to upstream.

        ``upstream/frame_utils.rgb_to_yuv6`` is ``@torch.no_grad()`` (line 50), so the
        canonical ``differentiable_rgb_to_yuv6`` stands in for the GRADIENT ONLY --
        its forward equivalence is asserted at call time, and every reported number
        comes from :meth:`distortion` on the unpatched path.
        """
        import einops

        from tac.differentiable_eval_roundtrip import (
            differentiable_rgb_to_yuv6,
        )

        x = einops.rearrange(cam_bthwc, "b t h w c -> b t c h w")
        b, t = x.shape[0], x.shape[1]
        x = einops.rearrange(x, "b t c h w -> (b t) c h w")
        x = torch.nn.functional.interpolate(x, size=(SEG_H, SEG_W), mode="bilinear")
        pin = einops.rearrange(
            differentiable_rgb_to_yuv6(x), "(b t) c h w -> b (t c) h w", b=b, t=t
        )
        return self.net.posenet(pin)["pose"][..., :6]

    def dpose_grad_wrt_frame0(
        self, gt_pair: np.ndarray, f0_u8: np.ndarray, f1_u8: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """d(d_pose)/d(frame_0 camera raster), plus the differentiable-path d_pose."""
        f0 = torch.from_numpy(f0_u8.astype(np.float32)).requires_grad_(True)
        f1 = torch.from_numpy(f1_u8.astype(np.float32))
        cm = torch.stack([f0, f1])[None]
        gt = torch.from_numpy(gt_pair.astype(np.float32))[None]
        with torch.no_grad():
            ref = self._pose_head(gt)
        loss = (ref - self._pose_head(cm)).pow(2).mean()
        loss.backward()
        if f0.grad is None:  # never an assert: -O would strip the guard
            raise RuntimeError("no gradient reached the frame_0 raster")
        return f0.grad.numpy().astype(np.float64), float(loss.item())


# --------------------------------------------------------------------- modes
def mode_blind_verify(threads: int) -> dict:
    from tac.optimization.ddm_bp2_blind_pose_actuator import d_column_weights
    from tac.optimization.ddm_ll1_window_solve import blind_mask

    weights = d_column_weights()
    blind_autograd = weights == 0.0
    blind_ll1 = blind_mask()
    out: dict[str, object] = {
        "blind_count_autograd": int(blind_autograd.sum()),
        "blind_frac": float(blind_autograd.mean()),
        "read_count": int((~blind_autograd).sum()),
        "read_count_equals_4x_scorer_px": int((~blind_autograd).sum()) == 4 * N_SCORER_PX,
        "agrees_with_ll1_blind_mask": bool(np.array_equal(blind_autograd, blind_ll1)),
        "max_column_sum": float(weights.max()),
        "min_read_column_sum": float(weights[~blind_autograd].min()),
        "no_pixel_read_twice": bool(weights.max() <= 1.0 + 1e-9),
        "column_sum_total": float(weights.sum()),
    }

    scorers = Scorers(threads)
    pair = next(iter(gt_pair_stream(REPO_ROOT / "upstream" / "videos" / "0.mkv", 1)))
    gt = np.stack([pair[1], pair[2]])

    def _pre(arr: np.ndarray):
        with torch.inference_mode():
            return scorers.net.preprocess_input(Scorers._batch(arr))

    p_ref, s_ref = _pre(gt)
    rng = np.random.default_rng(7)
    hard = gt.astype(np.int16)
    delta = rng.integers(-127, 128, size=hard.shape, dtype=np.int16)
    hard[:, blind_autograd, :] = np.clip(hard[:, blind_autograd, :] + delta[:, blind_autograd, :], 0, 255)
    p_h, s_h = _pre(hard.astype(np.uint8))
    out["blind_hard_perturb_px_changed"] = int((hard.astype(np.uint8) != gt).any(-1).sum())
    out["blind_hard_perturb_posenet_in_identical"] = bool(torch.equal(p_h, p_ref))
    out["blind_hard_perturb_segnet_in_identical"] = bool(torch.equal(s_h, s_ref))

    # Positive controls at BOTH ends of the read-weight range.  The min-weight read
    # pixel (column sum ~1e-5) is the sharpest test that the blind/read boundary is
    # where the geometry says it is: if even THAT pixel changes the scorer input,
    # then "identical" on the blind set is a real invariance and not a tolerance.
    masked = np.where(blind_autograd, np.inf, weights)
    blind_yx = np.argwhere(blind_autograd)
    controls = (
        ("control_read_px_max_weight", np.unravel_index(int(weights.argmax()), weights.shape)),
        ("control_read_px_min_weight", np.unravel_index(int(masked.argmin()), masked.shape)),
        ("control_blind_px", tuple(blind_yx[len(blind_yx) // 2])),
    )
    for label, (yy, xx) in controls:
        one = gt.copy()
        one[1, yy, xx, 0] = np.uint8((int(one[1, yy, xx, 0]) + 40) % 256)
        p_o, s_o = _pre(one)
        out[f"{label}_posenet_in_identical"] = bool(torch.equal(p_o, p_ref))
        out[f"{label}_segnet_in_identical"] = bool(torch.equal(s_o, s_ref))
        out[f"{label}_yx"] = [int(yy), int(xx)]
        out[f"{label}_d_weight"] = float(weights[yy, xx])

    out["POSITIVE_CONTROL_FIRED"] = all(
        not out[f"control_read_px_{end}_weight_{net}_in_identical"]
        for end in ("max", "min")
        for net in ("posenet", "segnet")
    )
    out["VERDICT_blind_set_invisible"] = bool(
        out["blind_hard_perturb_posenet_in_identical"]
        and out["blind_hard_perturb_segnet_in_identical"]
        and out["control_blind_px_posenet_in_identical"]
        and out["POSITIVE_CONTROL_FIRED"]
    )
    return out


def mode_overlap(decoder, blind: np.ndarray, n_pairs: int, sink) -> dict:
    from tac.optimization.ddm_bp2_blind_pose_actuator import (
        blind_influence_mass,
        d_column_weights,
        v4d_pair_taps,
    )

    weights = d_column_weights()
    rows = []
    for i in range(n_pairs):
        idx, w, valid = v4d_pair_taps(decoder, i)
        a = float(decoder.ab[i][0])
        row = blind_influence_mass(idx, w, weights, blind, photometric_a=a)
        row.update(
            pair=i,
            selector=int(decoder.sel[i]),
            beta=float(decoder.beta_mags[int(decoder.beta_idx[i])]),
            photometric_a=a,
            s_t=float(decoder.st_vals[decoder.st_idx[i]]),
            warp_valid_frac=float(valid.mean()),
            closure_ok=bool(abs(row["total_mass"] - abs(a) * N_SCORER_PX) < 1e-3),
        )
        rows.append(row)
        sink.write(json.dumps(row) + "\n")
        sink.flush()
    fr = np.array([r["blind_mass_frac"] for r in rows])
    return {
        "n_pairs": len(rows),
        "blind_mass_frac_mean": float(fr.mean()),
        "blind_mass_frac_min": float(fr.min()),
        "blind_mass_frac_max": float(fr.max()),
        "blind_px_active_frac_mean": float(np.mean([r["blind_px_active_frac"] for r in rows])),
        "warp_valid_frac_mean": float(np.mean([r["warp_valid_frac"] for r in rows])),
        "all_closure_ok": bool(all(r["closure_ok"] for r in rows)),
        "F1_falsifier_threshold": 0.05,
        "F1_CLOSES_FAMILY": bool(fr.mean() < 0.05),
    }


#: Descent arms.  Each is "perturb the smallest set of blind coordinates whose
#: |gradient| mass predicts a FIRST-ORDER decrease of t*d_pose".  A fixed +-1 LSB
#: step over ALL blind coordinates is far too coarse: the gradient L1 mass over the
#: blind set is already ~1x d_pose, so the full step lands past the minimum and the
#: second-order term takes over (MEASURED: 0/6 pairs improve, d_pose 300x WORSE).
#:
#: The grid was WIDENED after an n=11 partial run showed the largest arm BINDING on
#: high-d_pose pairs (pair 10: d_pose 0.01554 -> 0.01477 at the top arm).  A grid
#: whose top arm is selected reports a LOWER BOUND on the achievable reduction, and
#: the downstream byte-economics question (does a shipped correction pay for itself?)
#: turns on that number, so the bound must be loose-side-safe.  Widening a
#: deterministic encoder-side search does not bias the falsifier: d_pose is
#: deterministic, so per-pair argmin is a realizable choice, not selection on noise.
GRAD_MASS_TARGETS: tuple[float, ...] = (0.002, 0.01, 0.05, 0.15, 0.35, 0.7, 1.0)


def _apply_blind_step(f1_u8: np.ndarray, blind: np.ndarray, step_flat: np.ndarray) -> np.ndarray:
    out = f1_u8.copy()
    vals = out[blind].astype(np.int16)
    out[blind] = np.clip(vals + step_flat.reshape(vals.shape), 0, 255).astype(np.uint8)
    return out


def mode_reach(
    decoder,
    blind: np.ndarray,
    n_pairs: int,
    sink,
    done: list[dict] | None = None,
    *,
    stride: int = 1,
    offset: int = 0,
) -> dict:
    from tac.optimization.ddm_bp2_blind_pose_actuator import (
        adjoint_taps,
        v4d_pair_taps,
    )

    scorers = Scorers(torch.get_num_threads())
    rows = list(done or [])
    already = {int(r["pair"]) for r in rows}
    t_start = time.time()
    for i, gt0, gt1 in gt_pair_stream(REPO_ROOT / "upstream" / "videos" / "0.mkv", n_pairs):
        if i % stride != offset or i in already:
            continue
        gt = np.stack([gt0, gt1])
        f1 = decoder.f1(i)
        f0 = decoder.f0(i, f1)
        dp0, ds0 = scorers.distortion(gt, np.stack([f0, f1]))
        gt_out = scorers.pose_out(gt)
        gt_seg = scorers.seg_out(gt)
        # guard: the cached-GT fast paths must BE the authority, per pair.
        dp0_fast = scorers.dpose_from(gt_out, np.stack([f0, f1]))
        ds0_fast = scorers.dseg_from(gt_seg, np.stack([f0, f1]))

        grad_f0, dp0_diff = scorers.dpose_grad_wrt_frame0(gt, f0, f1)
        idx, w, _ = v4d_pair_taps(decoder, i)
        a = float(decoder.ab[i][0])
        grad_f1 = a * adjoint_taps(idx, w, grad_f0)
        grad_blind = grad_f1[blind]
        mag = np.abs(grad_blind).ravel()
        sgn = np.sign(grad_blind).ravel()
        order = np.argsort(-mag)
        cumulative = np.cumsum(mag[order])

        row = {
            "pair": i,
            "d_pose_base": dp0,
            "d_seg_base": ds0,
            "fast_path_matches_authority": bool(dp0_fast == dp0 and ds0_fast == ds0),
            "d_pose_base_diff_path": dp0_diff,
            # The gradient surrogate must be the SAME objective, not merely similar;
            # 1e-5 relative is fp32 graph-order noise, anything larger is a different
            # loss and the search direction would be measuring the wrong thing.
            "diff_path_matches_authority": bool(
                abs(dp0_diff - dp0) <= 1e-5 * max(dp0, 1e-12)
            ),
            "grad_blind_l1": float(mag.sum()),
            "grad_all_l1": float(np.abs(grad_f1).sum()),
            "n_blind_coords": int(mag.size),
        }
        row["grad_blind_share"] = (
            row["grad_blind_l1"] / row["grad_all_l1"] if row["grad_all_l1"] else 0.0
        )

        best_dp, best_k, best_t = dp0, 0, 0.0
        k_last = 0
        for target in GRAD_MASS_TARGETS:
            k = min(int(np.searchsorted(cumulative, target * dp0) + 1), mag.size)
            k_last = k
            step = np.zeros(mag.size, dtype=np.int16)
            step[order[:k]] = -sgn[order[:k]].astype(np.int16)
            pert = _apply_blind_step(f1, blind, step)
            dp = scorers.dpose_from(gt_out, np.stack([decoder.f0(i, pert), pert]))
            row[f"k_t{target}"] = k
            row[f"d_pose_t{target}"] = dp
            if dp < best_dp:
                best_dp, best_k, best_t = dp, k, target

        # CONTROL A: same coordinate count, RANDOM signs.  If this moves d_pose as
        # much as the gradient arm, the "steering" is not steering.
        #
        # Seeded PER PAIR, not once per process: a process-level RNG makes this arm
        # depend on which pairs the process happened to visit, so resuming or
        # sharding silently changes it (MEASURED: the only key that differed between
        # a sharded and an unsharded run).  Per-pair seeding makes every row
        # reproducible from the pair index alone.
        rng = np.random.default_rng(20260802 + i)
        step = np.zeros(mag.size, dtype=np.int16)
        step[order[:k_last]] = rng.choice([-1, 1], size=k_last).astype(np.int16)
        pert = _apply_blind_step(f1, blind, step)
        row["d_pose_random_sign_same_k"] = scorers.dpose_from(
            gt_out, np.stack([decoder.f0(i, pert), pert])
        )
        row["k_random_control"] = k_last

        # CAPACITY probe + the d_seg guard: full +-1 LSB gradient-sign over EVERY
        # blind coordinate.  This is the largest structured step the channel admits
        # at 1 LSB, so d_seg surviving it bit-identically is the strongest form of
        # the zero-seg-cost claim.
        for direction, tag in ((-1, "full_desc"), (+1, "full_asc")):
            step = (direction * sgn).astype(np.int16)
            pert = _apply_blind_step(f1, blind, step)
            f0p = decoder.f0(i, pert)
            row[f"d_pose_{tag}"] = scorers.dpose_from(gt_out, np.stack([f0p, pert]))
            if tag == "full_asc":
                ds = scorers.dseg_from(gt_seg, np.stack([f0p, pert]))
                row["d_seg_full_step"] = ds
                row["d_seg_identical_under_full_step"] = bool(ds == ds0)

        row["d_pose_best"] = best_dp
        row["best_k"] = best_k
        row["best_target"] = best_t
        row["delta_d_pose_best"] = best_dp - dp0
        # Explicit arm list, NOT a startswith("d_pose_") scan: that scan silently
        # swept in "d_pose_base_diff_path" (a ~0 term), which is harmless under max()
        # today and would be a wrong answer the moment this became a min or a mean.
        arms = [f"d_pose_t{t}" for t in GRAD_MASS_TARGETS] + [
            "d_pose_random_sign_same_k",
            "d_pose_full_desc",
            "d_pose_full_asc",
        ]
        row["max_abs_delta_any_arm"] = max(abs(row[key] - dp0) for key in arms)
        rows.append(row)
        sink.write(json.dumps(row) + "\n")
        sink.flush()

    rows.sort(key=lambda r: int(r["pair"]))
    dp_base = np.array([r["d_pose_base"] for r in rows])
    dp_best = np.array([r["d_pose_best"] for r in rows])
    rnd = np.array([r["d_pose_random_sign_same_k"] for r in rows])
    mean_base, mean_best = float(dp_base.mean()), float(dp_best.mean())

    def contrib(d: float) -> float:
        return float(np.sqrt(10.0 * d))

    return {
        "n_pairs": len(rows),
        "grad_mass_targets": list(GRAD_MASS_TARGETS),
        "wall_clock_s": round(time.time() - t_start, 1),
        "mean_d_pose_base": mean_base,
        "mean_d_pose_best_blind_only": mean_best,
        "mean_delta_d_pose": mean_best - mean_base,
        "pose_contribution_base": contrib(mean_base),
        "pose_contribution_best": contrib(mean_best),
        "delta_S_pose_term": contrib(mean_best) - contrib(mean_base),
        "frac_pairs_improved": float((dp_best < dp_base).mean()),
        "mean_best_k": float(np.mean([r["best_k"] for r in rows])),
        "mean_abs_delta_random_control": float(np.abs(rnd - dp_base).mean()),
        "mean_abs_delta_best_gradient_arm": float(np.abs(dp_best - dp_base).mean()),
        "mean_d_pose_full_asc": float(np.mean([r["d_pose_full_asc"] for r in rows])),
        "mean_d_pose_full_desc": float(np.mean([r["d_pose_full_desc"] for r in rows])),
        "grad_blind_share_mean": float(np.mean([r["grad_blind_share"] for r in rows])),
        "max_abs_delta_any_arm_mean": float(
            np.mean([r["max_abs_delta_any_arm"] for r in rows])
        ),
        "all_fast_path_matches_authority": bool(
            all(r["fast_path_matches_authority"] for r in rows)
        ),
        "all_diff_path_matches_authority": bool(
            all(r["diff_path_matches_authority"] for r in rows)
        ),
        "d_seg_identical_all_pairs": bool(
            all(r["d_seg_identical_under_full_step"] for r in rows)
        ),
        "F2_falsifier_threshold": 1e-4,
        "F2_CLOSES_FAMILY": bool(
            np.mean([r["max_abs_delta_any_arm"] for r in rows]) < 1e-4
        ),
    }


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("blind-verify", "overlap", "reach"))
    ap.add_argument("--archive", type=Path, help="v4d archive.zip (overlap/reach)")
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    ap.add_argument("--stage", type=Path, help="scratch dir for the vendored substrate")
    ap.add_argument("--pairs", type=int, default=600, help="n600 is the evidence bar")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--pair-stride",
        type=int,
        default=1,
        help=(
            "reach only: shard the pair set N ways.  A single process leaves ~85%% of "
            "the CPU idle (the adjoint is single-threaded numpy), so N shards with "
            "disjoint --out paths cut wall-clock ~linearly; merge the sidecar .jsonl "
            "files afterwards.  Sharding cannot change any per-pair number: every pair "
            "is measured independently against its own GT."
        ),
    )
    ap.add_argument("--pair-offset", type=int, default=0, help="reach only: this shard's residue")
    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reach only: keep per-pair rows already in the sidecar .jsonl and measure "
            "only the missing pairs.  An n600 reach run is ~100 min, which is longer "
            "than a shell will reliably survive (a first attempt was SIGURG-killed at "
            "pair 34), so resumability is mandatory per CLAUDE.md, not an optimisation."
        ),
    )
    ap.add_argument("--out", type=Path, required=True, help="receipt json path")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "upstream"))
    sys.path.insert(0, str(REPO_ROOT / "src"))
    torch.set_num_threads(args.threads)

    receipt: dict[str, object] = {
        "schema": "ddm_bp2_blind_pose_actuator.v1",
        "axis": "[macOS-CPU advisory] frozen CPU-torch scorers; NON-PROMOTABLE",
        "score_claim": False,
        "promotion_eligible": False,
        "mode": args.mode,
        "n_pairs_requested": args.pairs,
    }

    if args.mode == "blind-verify":
        receipt["result"] = mode_blind_verify(args.threads)
    else:
        if args.archive is None or args.stage is None:
            ap.error("--archive and --stage are required for overlap/reach")
        receipt["archive"] = str(args.archive)
        receipt["archive_bytes"] = args.archive.stat().st_size
        receipt["archive_sha256"] = _sha256(args.archive)
        arch_dir = stage_receiver(args.stage, args.archive, args.template)
        import inflate_runner_v4d as receiver

        from tac.optimization.ddm_ll1_window_solve import blind_mask

        decoder = receiver.Decoder(arch_dir)
        blind = blind_mask()
        n = min(args.pairs, int(decoder.n_pairs))
        receipt["n_pairs_run"] = n
        jsonl = args.out.with_suffix(".jsonl")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        done: list[dict] = []
        if args.resume and args.mode == "reach" and jsonl.exists():
            done = [json.loads(ln) for ln in jsonl.read_text().splitlines() if ln.strip()]
            print(f"[resume] carrying {len(done)} already-measured pairs", flush=True)
        receipt["resumed_from_rows"] = len(done)
        with open(jsonl, "a" if done else "w") as sink:
            if args.mode == "overlap":
                receipt["result"] = mode_overlap(decoder, blind, n, sink)
            else:
                receipt["pair_stride"] = args.pair_stride
                receipt["pair_offset"] = args.pair_offset
                receipt["result"] = mode_reach(
                    decoder,
                    blind,
                    n,
                    sink,
                    done,
                    stride=args.pair_stride,
                    offset=args.pair_offset,
                )
        receipt["per_pair_jsonl"] = str(jsonl)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt["result"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
