# SPDX-License-Identifier: MIT
"""Numerically verify the tac differentiable mirror against the REAL frozen upstream scorers.

FOUNDATION reverse-engineering helper for the comma video compression challenge.

This is a NO-FAKE verification harness (CLAUDE.md "NO FAKE IMPLEMENTATIONS"): it
loads the actual frozen SegNet+PoseNet safetensors weights, decodes real frames
from the real contest video ``upstream/videos/0.mkv`` (per Catalog #213), runs
BOTH the upstream forward and the tac differentiable mirror, and reports the
numerical divergence. It never asserts constants; every number is measured.

All results are ``[macOS-CPU advisory]`` — NON-PROMOTABLE (no score claims, no
MPS as authority). The harness forces ``--device cpu`` per CLAUDE.md
"MPS auth eval is NOISE".

Outputs a machine-readable JSON dict to stdout (and optionally ``--json-out``).

Sections:
  1. YUV6 forward equivalence (upstream rgb_to_yuv6 vs differentiable_rgb_to_yuv6)
  2. SegNet logit fidelity (upstream preprocess+forward vs tac mirror)
  3. PoseNet pose fidelity (upstream preprocess+forward vs tac mirror)
  4. s_seg computability (DeepFool top-2 margin backward — flip-risk saliency)
  5. s_pose computability (squared input-Jacobian of first-6 pose dims — Fisher)

Usage::

    .venv/bin/python tools/verify_upstream_scorer_mirror_fidelity.py \
        --upstream-dir upstream --video upstream/videos/0.mkv \
        --num-pairs 2 --json-out .omx/tmp/scorer_mirror_fidelity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _decode_real_frames(video_path: Path, num_frames: int) -> torch.Tensor:
    """Decode the first ``num_frames`` frames from the real contest video.

    Uses the SAME decode path as upstream AVVideoDataset.yuv420_to_rgb so the
    pixel values are byte-identical to what the contest scorer sees on CPU.

    Returns a ``(num_frames, H, W, 3)`` uint8 tensor (camera-native 874x1164).
    """
    upstream_dir = REPO_ROOT / "upstream"
    if str(upstream_dir) not in sys.path:
        sys.path.insert(0, str(upstream_dir))
    import av
    from frame_utils import yuv420_to_rgb

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        frames.append(yuv420_to_rgb(frame))  # (H, W, 3) uint8, matches nvdec
        if len(frames) >= num_frames:
            break
    container.close()
    if len(frames) < num_frames:
        raise RuntimeError(
            f"decoded only {len(frames)} frames from {video_path}, needed {num_frames}"
        )
    return torch.stack(frames)  # (num_frames, H, W, 3) uint8


def _section_yuv6_equivalence() -> dict:
    """Section 1: differentiable_rgb_to_yuv6 vs upstream rgb_to_yuv6 (max abs diff)."""
    from tac.differentiable_eval_roundtrip import (
        assert_yuv6_forward_equivalence_to_upstream,
    )

    # This routine itself loads upstream frame_utils.rgb_to_yuv6 and compares on
    # random RGB tensors at 0 atol. We surface its measured max_abs_error.
    try:
        result = assert_yuv6_forward_equivalence_to_upstream(num_samples=8, atol=1e-6)
        return {
            "ran": True,
            "passed": bool(result["passed"]),
            "max_abs_error": float(result["max_abs_error"]),
            "num_samples": int(result["num_samples"]),
            "per_sample_max_abs_error": [float(x) for x in result["details"]],
        }
    except AssertionError as exc:  # equivalence failed
        return {"ran": True, "passed": False, "error": str(exc)}
    except RuntimeError as exc:  # upstream not importable
        return {"ran": False, "blocker": str(exc)}


def _build_upstream_distortion_net(upstream_dir: Path, device: torch.device):
    """Construct the upstream DistortionNet with REAL frozen weights (no patches)."""
    if str(upstream_dir) not in sys.path:
        sys.path.insert(0, str(upstream_dir))
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    dn = DistortionNet().eval().to(device)
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for p in dn.parameters():
        p.requires_grad = False
    return dn


def _to_btchw(frames_hwc: torch.Tensor, device: torch.device) -> torch.Tensor:
    """(num_frames, H, W, 3) uint8 -> (num_pairs, 2, 3, H, W) float (CHW).

    This is the layout AFTER the upstream DistortionNet.preprocess_input rearrange
    (modules.py:145 ``b t h w c -> b t c h w`` then ``.float()``), i.e. exactly
    what both PoseNet.preprocess_input (modules.py:70) and SegNet.preprocess_input
    (modules.py:107) receive. We feed identical CHW input to both upstream and
    mirror so the ONLY difference under test is the preprocess_input patch.
    """
    nf = frames_hwc.shape[0]
    npairs = nf // 2
    x = frames_hwc[: npairs * 2].reshape(npairs, 2, *frames_hwc.shape[1:])  # (P, 2, H, W, 3)
    x = x.permute(0, 1, 4, 2, 3).float().contiguous().to(device)  # (P, 2, 3, H, W)
    return x


def _section_scorer_fidelity(
    upstream_dir: Path, video_path: Path, num_pairs: int, device: torch.device
) -> dict:
    """Sections 2+3: SegNet logit + PoseNet pose fidelity (upstream vs tac mirror).

    The tac mirror = upstream weights + tac.scorer.make_scorers_differentiable
    (which patches PoseNet.preprocess_input to a differentiable rgb_to_yuv6 and
    AllNorm to reshape). We compare the FORWARD outputs on identical real-frame
    inputs (CHW) to measure whether the patch changes numerics. SegNet's
    preprocess_input is NOT patched, so SegNet fidelity isolates the YUV6/resize
    interaction in PoseNet only; we still run SegNet to capture its real logits
    for the s_seg saliency section.
    """
    out: dict = {}
    try:
        frames = _decode_real_frames(video_path, num_pairs * 2)
    except Exception as exc:
        return {"ran": False, "blocker": f"frame decode failed: {exc}"}

    x_btchw = _to_btchw(frames, device)  # (P, 2, 3, 874, 1164) float, CHW

    dn = _build_upstream_distortion_net(upstream_dir, device)
    posenet = dn.posenet
    segnet = dn.segnet

    # --- Upstream (unpatched, frozen) — call per-scorer preprocess on CHW ---
    with torch.inference_mode():
        up_pose_in = posenet.preprocess_input(x_btchw)  # (P, 12, 192, 256)
        up_pose = posenet(up_pose_in)["pose"].detach().clone()  # (P, 12)
        up_seg_in = segnet.preprocess_input(x_btchw)  # (P, 3, 384, 512)
        up_seg = segnet(up_seg_in).detach().clone()  # (P, 5, 384, 512)

    # --- tac mirror: same weights, differentiable PoseNet preprocess + AllNorm ---
    from tac.scorer import make_scorers_differentiable

    make_scorers_differentiable(posenet, segnet)
    # PoseNet.preprocess_input now uses the tac differentiable yuv6 + align_corners=False.
    with torch.no_grad():
        mir_pose_in = posenet.preprocess_input(x_btchw)
        mir_pose_out = posenet(mir_pose_in)["pose"]  # (P, 12)
        # SegNet.preprocess_input is unchanged by the patch; rerun for completeness.
        mir_seg_in = segnet.preprocess_input(x_btchw)
        mir_seg_out = segnet(mir_seg_in)  # (P, 5, 384, 512)

    pose_abs = (mir_pose_out - up_pose).abs()
    seg_abs = (mir_seg_out - up_seg).abs()
    # argmax-class agreement on segnet (the metric that actually drives d_seg)
    up_argmax = up_seg.argmax(dim=1)
    mir_argmax = mir_seg_out.argmax(dim=1)
    argmax_disagree_frac = (up_argmax != mir_argmax).float().mean().item()

    out["ran"] = True
    out["num_pairs"] = int(num_pairs)
    out["posenet_pose"] = {
        "max_abs_diff": float(pose_abs.max().item()),
        "mean_abs_diff": float(pose_abs.mean().item()),
        "first6_max_abs_diff": float(pose_abs[..., :6].max().item()),
        "upstream_pose_first6_abs_mean": float(up_pose[..., :6].abs().mean().item()),
    }
    out["segnet_logits"] = {
        "max_abs_diff": float(seg_abs.max().item()),
        "mean_abs_diff": float(seg_abs.mean().item()),
        "argmax_disagree_frac": float(argmax_disagree_frac),
        "upstream_logit_abs_mean": float(up_seg.abs().mean().item()),
    }
    # Stash for saliency sections (reuse the patched scorers + real frames).
    out["_x_btchw_shape"] = list(x_btchw.shape)
    return out, posenet, segnet, x_btchw  # type: ignore[return-value]


def _section_s_seg(segnet, x_btchw: torch.Tensor) -> dict:
    """Section 4: s_seg = DeepFool top-2 margin backward (flip-risk saliency).

    For each pixel the nearest decision boundary (Moosavi-Dezfooli et al. 2016,
    multiclass DeepFool) is approximately the distance to flip the argmax from
    the winning class k* to the runner-up class k_hat:

        s_seg_i = ||grad_{x_i}(z_k_hat - z_k*)||^2 / (z_k_hat - z_k*)^2

    where (z_k_hat - z_k*) is the (negative) top-2 logit margin. We compute ONE
    backward of the SUMMED top-2 margin over all pixels (a single Jacobian-vector
    product gives the per-INPUT-pixel gradient-energy contribution); this is the
    canonical cheap surrogate that proves the saliency is gradient-reachable.

    Reports: gradient finite? non-trivial (nonzero variance)? spatially structured
    (boundary energy >> interior energy)?
    """
    # Use the LAST frame of the first pair (SegNet uses x[:, -1, ...]).
    pair0 = x_btchw[:1]  # (1, 2, 3, H, W)
    seg_in = segnet.preprocess_input(pair0)  # (1, 3, 384, 512)
    seg_in = seg_in.detach().clone().requires_grad_(True)
    logits = segnet(seg_in)  # (1, 5, 384, 512)
    # top-2 margin per pixel: m = z_top1 - z_top2 (>= 0). DeepFool minimal
    # perturbation to flip ~ m / ||grad m||. We backprop the SUMMED margin to get
    # the per-input-pixel gradient field.
    top2 = logits.topk(2, dim=1).values  # (1, 2, H, W)
    margin = (top2[:, 0] - top2[:, 1])  # (1, H, W) winning-minus-runnerup, >= 0
    summed = margin.sum()
    grad = torch.autograd.grad(summed, seg_in, retain_graph=False)[0]  # (1, 3, 384, 512)
    # per-pixel gradient energy of the margin wrt input (sum over RGB channels)
    grad_energy = grad.pow(2).sum(dim=1)  # (1, 384, 512)
    # DeepFool flip-risk ~ grad_energy / margin^2 (high where margin small AND grad large)
    flip_risk = grad_energy / (margin.detach().pow(2) + 1e-6)  # (1, 384, 512)

    finite = bool(torch.isfinite(grad).all().item()) and bool(torch.isfinite(flip_risk).all().item())
    # spatial structure: boundary pixels = where margin is in the lowest decile.
    m = margin.detach().reshape(-1)
    q10 = torch.quantile(m, 0.10)
    boundary_mask = (margin.detach() <= q10)  # low-margin = near boundary
    interior_mask = (margin.detach() > torch.quantile(m, 0.90))
    fr = flip_risk.detach()
    boundary_energy = fr[boundary_mask].mean().item() if boundary_mask.any() else 0.0
    interior_energy = fr[interior_mask].mean().item() if interior_mask.any() else 0.0
    return {
        "ran": True,
        "grad_finite": finite,
        "grad_energy_max": float(grad_energy.max().item()),
        "grad_energy_mean": float(grad_energy.mean().item()),
        "grad_nonzero_frac": float((grad.abs() > 0).float().mean().item()),
        "margin_min": float(margin.min().item()),
        "margin_mean": float(margin.mean().item()),
        "flip_risk_boundary_mean": float(boundary_energy),
        "flip_risk_interior_mean": float(interior_energy),
        "boundary_over_interior_ratio": float(
            boundary_energy / (interior_energy + 1e-12)
        ),
        "spatially_structured": bool(boundary_energy > interior_energy),
    }


def _section_s_pose(posenet, x_btchw: torch.Tensor) -> dict:
    """Section 5: s_pose = squared input-Jacobian of first-6 pose dims (Fisher info).

    Under iid Gaussian pixel noise the per-pixel Fisher information of the pose
    output is the diagonal of J^T J where J = d(pose_1..6)/d(x_i):

        s_pose_i = sum_{k=1..6} (d pose_k / d x_i)^2

    We compute it by backpropagating each of the first-6 pose scalars and
    accumulating squared input gradients. This requires the differentiable
    rgb_to_yuv6 + differentiable bilinear resize (both confirmed in the mirror).

    Reports: gradient finite? non-trivial? spatially spread (NOT boundary-peaked,
    unlike s_seg)?
    """
    pair0 = x_btchw[:1].detach().clone().requires_grad_(True)  # (1, 2, 3, H, W)
    pose_in = posenet.preprocess_input(pair0)  # (1, 12, 192, 256)
    pose_out = posenet(pose_in)["pose"]  # (1, 12)
    s_pose = torch.zeros_like(pair0)
    pose_grads_finite = True
    for k in range(6):
        g = torch.autograd.grad(
            pose_out[0, k], pair0, retain_graph=(k < 5), create_graph=False
        )[0]
        if not torch.isfinite(g).all():
            pose_grads_finite = False
        s_pose = s_pose + g.pow(2)
    # collapse to per-pixel (sum over the 2 frames and RGB channels)
    s_pose_per_pixel = s_pose.sum(dim=(1, 2))  # (1, H, W)
    finite = pose_grads_finite and bool(torch.isfinite(s_pose_per_pixel).all().item())
    spp = s_pose_per_pixel.detach().reshape(-1)
    # "spread" diagnostic: fraction of total Fisher energy in the top-10% pixels.
    # s_seg concentrates (boundary); s_pose should be more spread.
    k10 = max(1, int(0.10 * spp.numel()))
    top10_energy = torch.topk(spp, k10).values.sum().item()
    total_energy = spp.sum().item() + 1e-12
    return {
        "ran": True,
        "grad_finite": finite,
        "s_pose_max": float(s_pose_per_pixel.max().item()),
        "s_pose_mean": float(s_pose_per_pixel.mean().item()),
        "s_pose_nonzero_frac": float((s_pose_per_pixel > 0).float().mean().item()),
        "top10pct_energy_frac": float(top10_energy / total_energy),
        "is_nontrivial": bool(s_pose_per_pixel.max().item() > 0.0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument("--video", type=Path, default=REPO_ROOT / "upstream/videos/0.mkv")
    parser.add_argument("--num-pairs", type=int, default=2)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    # NON-NEGOTIABLE: CPU only. No MPS authority (CLAUDE.md "MPS auth eval is NOISE").
    device = torch.device("cpu")
    torch.manual_seed(20260601)

    report: dict = {
        "advisory_tag": "[macOS-CPU advisory]",
        "promotable": False,
        "score_claim": False,
        "device": "cpu",
        "upstream_dir": str(args.upstream_dir),
        "video": str(args.video),
        "models_present": (args.upstream_dir / "models/segnet.safetensors").exists()
        and (args.upstream_dir / "models/posenet.safetensors").exists(),
        "video_present": args.video.exists(),
    }

    report["section1_yuv6_equivalence"] = _section_yuv6_equivalence()

    if not (report["models_present"] and report["video_present"]):
        report["section2_3_scorer_fidelity"] = {
            "ran": False,
            "blocker": "models or video absent locally; static-only verification done",
        }
        report["section4_s_seg"] = {"ran": False, "blocker": "models/video absent"}
        report["section5_s_pose"] = {"ran": False, "blocker": "models/video absent"}
    else:
        fidelity = _section_scorer_fidelity(
            args.upstream_dir, args.video, args.num_pairs, device
        )
        if isinstance(fidelity, tuple):
            fid_dict, posenet, segnet, x_btchw = fidelity
            report["section2_3_scorer_fidelity"] = fid_dict
            report["section4_s_seg"] = _section_s_seg(segnet, x_btchw)
            report["section5_s_pose"] = _section_s_pose(posenet, x_btchw)
        else:
            report["section2_3_scorer_fidelity"] = fidelity
            report["section4_s_seg"] = {"ran": False, "blocker": fidelity.get("blocker", "?")}
            report["section5_s_pose"] = {"ran": False, "blocker": fidelity.get("blocker", "?")}

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
