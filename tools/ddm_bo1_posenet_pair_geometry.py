#!/usr/bin/env python
"""ddm_bo1 — PoseNet stem PAIR geometry: how much of frame_1's pose damage can frame_0 cancel?

WHAT THIS IS. A scorer-free, video-free, deterministic structural probe of the FROZEN
upstream PoseNet. It reads `upstream/modules.py`'s PoseNet, folds BatchNorm into the two
stem branches, and answers one question exactly:

    PoseNet's input is the CHANNEL-CONCATENATION of the two frames' yuv6 maps
    (`modules.py:70-74`, IN_CHANS = 6*2). The stem block (`vision.stem[0]`) is
    MobileOneBlock(conv_kxk[0] + conv_scale), both Conv+BN, i.e. AFFINE, and it has no
    `identity` branch (in_chans 12 != out_chans 64). Therefore the pre-activation

        z = A_0 u_0 + A_1 u_1 + c            u_t := yuv6(R(frame_t))

    is EXACT, and A_0 / A_1 are just the two 6-channel blocks of one folded 3x3 kernel.
    z is the ONLY place the pair mixes; everything downstream is a fixed map Phi(z).
    Hence  z == z*  ==>  d_pose == 0  EXACTLY (no linearisation).

    Given a perturbation delta_1 of frame_1 (which is what a seg objective spends), the
    residual under a frame_0 policy delta_0 is  ||A_0 delta_0 + A_1 delta_1||^2, and we
    report it as a RATIO to ||A_1 delta_1||^2 (= the residual of leaving frame_0 at GT).

        ratio < 1  -> frame_0 bought pose back        ratio > 1 -> frame_0 made it worse

    Policies compared: optimal delta_0 anywhere in range(A_0) (a LOWER BOUND on any
    carrier's residual), delta_0 = +delta_1 (what a warp carrier that transports the
    error does), and delta_0 = -delta_1 (the same carrier, sign-flipped).

INSTRUMENT RUNG — READ BEFORE DESIGNING OFF THESE NUMBERS. Per
`control_surface_exact_dof_quartering_q3_seg_only_pose_null_20260731` the ladder is
    dimension count -> L2 spectrum -> margin/Fisher-weighted (THE object, UNMEASURED).
This tool is RUNG 2: an L2-energy read under an ISOTROPIC delta_1 assumption. It therefore
reports the ENERGY-WEIGHTED QUANTILES, not only the mean, so a reader can see how much of
the conclusion survives reweighting. A direction-extremal (min/max) read is NOT reported as
a headline because it is degenerate here: it is attained in directions carrying ~0 pose
energy. Rung 3 (margin/Fisher-weighted delta_1 from the live base) is OWED.

NOT A SCORE. `score_claim=false`, `promotable=false`. This is a property of the frozen
scorer's weights, independent of any vehicle, archive, or run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = REPO_ROOT / "upstream"

# Output grid of the stem: input is (384, 512) per frame, stem stride is 2 in both branches.
DEFAULT_GRID = (192, 256)
# Relative tolerance for treating a singular value as numerically null.
RCOND = 1e-10
# Max abs error tolerated when checking the BN fold against the live module.
FOLD_TOL = 1e-4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def fold_conv_bn(conv_norm_act: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold an eval-mode Conv+BatchNorm into a single affine (weight, bias), float64.

    BN in eval mode is affine: y = gamma*(x-mu)/sqrt(var+eps) + beta, so it composes with
    the preceding conv exactly. Returns (weight, bias) in float64.
    """
    weight = conv_norm_act.conv.weight.detach().double()
    bn = conv_norm_act.bn
    scale = bn.weight.detach().double() / torch.sqrt(bn.running_var.detach().double() + bn.eps)
    folded_w = weight * scale.view(-1, 1, 1, 1)
    folded_b = bn.bias.detach().double() - bn.running_mean.detach().double() * scale
    return folded_w, folded_b


def stem_effective_kernel(posenet: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return the single folded 3x3 stride-2 kernel equivalent to the whole stem block.

    The block sums a 3x3 stride-2 branch and a 1x1 stride-2 branch. The 1x1 at stride 2
    samples exactly the centre tap of the 3x3 window, so the two fold into ONE 3x3 kernel
    with the 1x1 added at [1, 1]. Fail-closed: verified numerically against the live module.
    """
    block = posenet.vision.stem[0]
    if getattr(block, "identity", None) is not None:
        raise RuntimeError(
            "stem block gained an `identity` branch; the affine fold below assumes it is absent"
        )
    w3, b3 = fold_conv_bn(block.conv_kxk[0])
    w1, b1 = fold_conv_bn(block.conv_scale)
    if w3.shape[-2:] != (3, 3) or w1.shape[-2:] != (1, 1):
        raise RuntimeError(f"unexpected stem kernel shapes: {tuple(w3.shape)} / {tuple(w1.shape)}")
    weight = w3.clone()
    weight[:, :, 1, 1] += w1[:, :, 0, 0]
    bias = b3 + b1

    generator = torch.Generator().manual_seed(0)
    probe = torch.randn(1, weight.shape[1], 64, 64, generator=generator)
    with torch.no_grad():
        reference = (block.conv_kxk[0](probe) + block.conv_scale(probe)).double()
        mine = torch.nn.functional.conv2d(probe.double(), weight, bias, stride=2, padding=1)
    err = (reference - mine).abs().max().item()
    if not err < FOLD_TOL:
        raise RuntimeError(f"BN fold does not reproduce the live stem: max_abs_err={err:.3e}")
    return weight, bias, err


def polyphase_fourier_blocks(kernel: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """Block-diagonalise a 3x3 stride-2 conv over the output grid (circular boundary).

    A stride-2 conv couples each output position to 4 input polyphase components. Writing
    the input index as p = 2m + r (r in {0,1}), the 3x3 taps i in {0,1,2} at padding 1 map
    to (r=1, shift -1), (r=0, shift 0), (r=1, shift 0). So on the half-grid the operator is
    a stride-1 conv from 4*C_in polyphase channels with taps at shifts {0,-1} per axis,
    which the DFT diagonalises exactly. Returns (H2*W2, C_out, 4*C_in) complex128.

    Boundary: circular. Exact on the interior; the 1-pixel border of the half-grid differs
    (~1.8% of positions). The +/-delta policies are cross-checked against boundary-free
    Frobenius ratios by `pair_geometry`.
    """
    h2, w2 = grid
    c_out, c_in = kernel.shape[0], kernel.shape[1]
    # Forward-DFT convention: a tap at half-grid shift +1 contributes exp(-2*pi*i*omega/N).
    # (All reported ratios are invariant to conjugating this, since that only relabels
    # omega -> -omega and every statistic here aggregates over the whole grid; the true
    # forward convention is used so the blocks reproduce a real circular conv exactly.)
    ph = torch.exp(-1j * torch.arange(h2).double() * 2 * math.pi / h2).view(h2, 1, 1, 1)
    pw = torch.exp(-1j * torch.arange(w2).double() * 2 * math.pi / w2).view(1, w2, 1, 1)
    k = kernel.to(torch.complex128)
    m = torch.zeros(h2, w2, c_out, c_in, 2, 2, dtype=torch.complex128)
    m[..., 0, 0] = k[:, :, 1, 1]
    m[..., 0, 1] = k[:, :, 1, 0] * pw + k[:, :, 1, 2]
    m[..., 1, 0] = k[:, :, 0, 1] * ph + k[:, :, 2, 1]
    m[..., 1, 1] = (k[:, :, 0, 0] * ph + k[:, :, 2, 0]) * pw + (k[:, :, 0, 2] * ph + k[:, :, 2, 2])
    return m.reshape(h2 * w2, c_out, c_in * 4)


def _weighted_quantiles(ratio: torch.Tensor, weight: torch.Tensor, qs: list[float]) -> list[float]:
    order = torch.argsort(ratio)
    sorted_ratio, cum = ratio[order], torch.cumsum(weight[order], 0) / weight.sum()
    idx = torch.searchsorted(cum, torch.tensor(qs, dtype=cum.dtype)).clamp(max=len(sorted_ratio) - 1)
    return [sorted_ratio[i].item() for i in idx]


def pair_geometry(kernel: torch.Tensor, grid: tuple[int, int], chunk: int = 2048) -> dict[str, Any]:
    """Compute the frame_0-cancels-frame_1 spectrum for each delta_0 policy."""
    half = kernel.shape[1] // 2
    m0 = polyphase_fourier_blocks(kernel[:, :half], grid)
    m1 = polyphase_fourier_blocks(kernel[:, half:], grid)
    # PRECONDITION. Every ratio below is per-direction in the ROW space of M_1, so it can only
    # account for the whole input space when M_1 has full COLUMN rank. If it does not, there are
    # delta_1 directions with zero pose effect from frame_1 but nonzero effect from frame_0; the
    # ratio is undefined there and the reported mean would silently omit them. Fail closed rather
    # than report a mean that is not the quantity it claims to be. (Real PoseNet: 64 rows vs 24
    # columns per block, numerically full rank at every frequency -- checked below.)
    if m1.shape[1] < m1.shape[2]:
        raise RuntimeError(
            f"M_1 is {m1.shape[1]}x{m1.shape[2]}: fewer rows than columns, so it cannot have "
            "full column rank and the per-direction ratios do not span the input space"
        )

    weights, ratios = [], {"optimal_in_range_A0": [], "delta0_plus": [], "delta0_minus": []}
    worst_rank_ratio = 1.0
    for start in range(0, m0.shape[0], chunk):
        a, b = m0[start:start + chunk], m1[start:start + chunk]
        ub, sb, vhb = torch.linalg.svd(b, full_matrices=False)
        worst_rank_ratio = min(worst_rank_ratio, (sb[:, -1] / sb[:, 0]).min().item())
        ua, sa, _ = torch.linalg.svd(a, full_matrices=False)
        ua_k = ua * (sa > sa[:, :1] * RCOND).unsqueeze(1)
        resid = ub - ua_k @ (ua_k.conj().transpose(-1, -2) @ ub)
        v = vhb.conj().transpose(-1, -2)
        energy = (sb ** 2).flatten()
        weights.append(energy)
        ratios["optimal_in_range_A0"].append((resid.abs() ** 2).sum(1).flatten())
        ratios["delta0_plus"].append((((a + b) @ v).abs() ** 2).sum(1).flatten() / energy)
        ratios["delta0_minus"].append((((b - a) @ v).abs() ** 2).sum(1).flatten() / energy)

    if worst_rank_ratio < RCOND:
        raise RuntimeError(
            f"M_1 is numerically rank-deficient at some frequency (min sigma_min/sigma_max = "
            f"{worst_rank_ratio:.3e} < {RCOND:.0e}); the per-direction ratios are undefined there"
        )
    weight = torch.cat(weights)
    qs = [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    out: dict[str, Any] = {"m1_worst_conditioning_sigma_min_over_sigma_max": worst_rank_ratio}
    for name, parts in ratios.items():
        ratio = torch.cat(parts)
        out[name] = {
            "mean_ratio": (ratio * weight).sum().item() / weight.sum().item(),
            "energy_weighted_quantiles": dict(zip([f"p{int(q * 100)}" for q in qs],
                                                  _weighted_quantiles(ratio, weight, qs),
                                                  strict=True)),
            "share_of_pose_energy_worse_than_leaving_frame0_alone":
                (weight[ratio > 1.0].sum() / weight.sum()).item(),
        }

    # Boundary-free cross-check: for delta_0 = +/-delta_1 the isotropic mean ratio is just a
    # Frobenius-norm ratio of the folded kernel blocks, derivable without any Fourier work.
    k0, k1 = kernel[:, :half], kernel[:, half:]
    for name, closed in (("delta0_plus", ((k0 + k1).norm() / k1.norm()) ** 2),
                         ("delta0_minus", ((k1 - k0).norm() / k1.norm()) ** 2)):
        got, want = out[name]["mean_ratio"], closed.item()
        if abs(got - want) > 1e-6 * max(1.0, want):
            raise RuntimeError(f"{name}: Fourier mean {got:.9f} != closed form {want:.9f}")
        out[name]["closed_form_crosscheck"] = want
    return out


def channel_block_norms(kernel: torch.Tensor) -> dict[str, Any]:
    """Per-yuv6-channel Frobenius norms and alignment of the two frames' stem blocks."""
    half = kernel.shape[1] // 2
    k0, k1 = kernel[:, :half], kernel[:, half:]
    names = ["y00", "y10", "y01", "y11", "U", "V"]
    n0, n1 = k0.norm().item(), k1.norm().item()
    per_channel = {}
    for i, nm in enumerate(names):
        a, b = k0[:, i].norm().item(), k1[:, i].norm().item()
        per_channel[nm] = {
            "frame0_fro": a, "frame1_fro": b,
            "cosine": (k0[:, i].flatten() @ k1[:, i].flatten()).item() / (a * b),
            "share_of_frame1_energy": b ** 2 / n1 ** 2,
        }
    return {
        "frame0_fro": n0, "frame1_fro": n1,
        "cosine_frame0_frame1": (k0.flatten() @ k1.flatten()).item() / (n0 * n1),
        "symmetric_part_fro": (k0 + k1).norm().item() / 2,
        "antisymmetric_part_fro": (k1 - k0).norm().item() / 2,
        "per_channel": per_channel,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-json", type=Path,
                        default=REPO_ROOT / "reports/ddm_bo1/posenet_pair_geometry.json")
    parser.add_argument("--grid", type=str, default="192x256",
                        help="stem output grid HxW (default 192x256 = (384,512) at stride 2)")
    parser.add_argument("--chunk", type=int, default=2048)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(UPSTREAM))
    # Deferred: `modules` resolves only after UPSTREAM is on sys.path (pinned snapshot, never edited).
    from modules import PoseNet, posenet_sd_path
    from safetensors.torch import load_file

    posenet = PoseNet().eval()
    posenet.load_state_dict(load_file(str(posenet_sd_path), device="cpu"))
    kernel, _bias, fold_err = stem_effective_kernel(posenet)
    grid = tuple(int(v) for v in args.grid.lower().split("x"))

    report = {
        "arm": "ddm_bo1",
        "what": "frozen-PoseNet stem pair geometry: frame_0's ability to cancel frame_1's pose damage",
        "score_claim": False,
        "promotable": False,
        "axis_tag": "[frozen-scorer structural]",
        "instrument_rung": "rung-2 (L2 energy, isotropic delta_1); rung-3 margin/Fisher-weighted is OWED",
        "provenance": {
            "git_hash": _git_hash(),
            "upstream_modules_sha256": _sha256(UPSTREAM / "modules.py"),
            "upstream_frame_utils_sha256": _sha256(UPSTREAM / "frame_utils.py"),
            "posenet_safetensors_sha256": _sha256(Path(posenet_sd_path)),
            "seed": 0,
            "dtype": "float64",
        },
        "stem_structure": {
            "in_chans": int(kernel.shape[1]),
            "out_chans": int(kernel.shape[0]),
            "affine_fold_max_abs_err_vs_live_module": fold_err,
            "identity_branch_present": False,
            "note": "z = A_0 u_0 + A_1 u_1 + c is EXACT; z == z* => d_pose == 0 with no linearisation",
        },
        "channel_block_norms": channel_block_norms(kernel),
        "cancellability": pair_geometry(kernel, grid, args.chunk),
        "grid": list(grid),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["cancellability"], indent=2))
    print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
