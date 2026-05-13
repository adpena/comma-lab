"""Lane Ω-W — Water-filling Lagrangian bit-budget allocator.

Operationalises the Shannon reverse-water-filling formula (Cover & Thomas 10.4.2)
to allocate per-output-channel block-FP qint_max values across SegMap conv
weights, replacing the broken Hessian fallback in Lane SO with a
mathematically-rigorous solution.

Design doc (BINDING): docs/paper/water_filling_design_20260429.md
Inner-council sign-offs: Shannon LEAD + Dykstra CO-LEAD + Fridrich + Yousfi +
Contrarian + Ballé + MacKay.

Public API:

    estimate_per_channel_variance(model, ...) -> dict[name -> Tensor[O]]
    estimate_per_channel_hessian(model, calib_x, calib_y, calib_idx, ...) -> dict
    water_fill_bit_budget(hessians, variances, channel_element_counts, total_bits)
        -> dict[name -> list[int]]  (qint_max per channel)
    export_with_water_filling(model, ..., total_bits, output_path) -> dict

Compliance gates:
- eval_roundtrip=True default; False raises WaterFillError loudly.
- CUDA-required default; --device cpu opt-in is for unit-test smoke ONLY
  (deterministic-bytes acceptable for tests; production archive bytes
  MUST be produced from --device cuda). NO MPS fallback.
- No scorer load at inflate (this is COMPRESS-time only).
- pack_payload_tar_xz called WITHOUT exponents= kwarg (Round 1 lesson).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from tac.block_fp_codec import (
    pack_payload_tar_xz,
    verify_roundtrip,
)
from tac.learnable_bit_quant import iter_eligible_conv_names

# ── exceptions ────────────────────────────────────────────────────────────


class WaterFillError(ValueError):
    """Raised when Hessian is non-finite OR water-fill cannot satisfy budget."""


# ── discrete bit ladder (binding per design §1.3) ─────────────────────────

QINT_LEVELS: tuple[int, ...] = (1, 3, 7, 15, 31)
"""The only allowed qint_max values; matches encode_conv_weight clamp range."""

QINT_BITS: tuple[float, ...] = tuple(math.log2(2 * q + 1) for q in QINT_LEVELS)
"""Signed-integer bit-count per element for each qint_max level.
qint_max=1 -> 3 levels {-1,0,1} -> log2(3) ≈ 1.585 bits.
qint_max=3 -> 7 levels -> log2(7) ≈ 2.807 bits.
qint_max=7 -> 15 levels -> log2(15) ≈ 3.907 bits.
qint_max=15 -> 31 levels -> log2(31) ≈ 4.954 bits.
qint_max=31 -> 63 levels -> log2(63) ≈ 5.977 bits.
"""


def bits_for_qint(q: int) -> float:
    """Signed-integer bit-count for one element at qint_max=q."""
    if q not in QINT_LEVELS:
        raise WaterFillError(
            f"bits_for_qint: q={q} not in canonical ladder {QINT_LEVELS}. "
            f"Lane Ω-W only ships these five Q levels."
        )
    return QINT_BITS[QINT_LEVELS.index(q)]


def qint_for_bits(b: float) -> int:
    """Continuous→discrete map. Bin centres at b ∈ {1.0, 2.0, 3.0, 4.0, 5.0+}.

    Round-down rule (design §1.3): any b < 1.5 → qint_max=1 (the floor).
    Any b ≥ 4.5 → qint_max=31 (the ceiling).
    """
    if not math.isfinite(b):
        raise WaterFillError(
            f"qint_for_bits: non-finite input b={b}. Hessian or σ² produced "
            f"NaN/Inf — refuse to silently floor."
        )
    if b < 1.5:
        return 1
    if b < 2.5:
        return 3
    if b < 3.5:
        return 7
    if b < 4.5:
        return 15
    return 31


# ── variance estimation (cheap closed-form) ───────────────────────────────


def estimate_per_channel_variance(
    model: nn.Module,
    extra_protected_patterns: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    """Return {<conv_name>.weight: Tensor[O]} of per-output-channel variance.

    σ_c² is the unbiased variance of the (I·kH·kW) elements within the
    output channel `c`. No forward pass; just walks model.named_modules().
    """
    out: dict[str, torch.Tensor] = {}
    eligible = set(iter_eligible_conv_names(model, extra_protected_patterns))
    for name, mod in model.named_modules():
        if name not in eligible:
            continue
        if not isinstance(mod, nn.Conv2d):
            continue
        w = mod.weight.detach()
        # (O, I, kH, kW) -> per-O variance over flattened (I*kH*kW)
        per_c = w.reshape(w.shape[0], -1).var(dim=1, unbiased=False).cpu()
        if not torch.isfinite(per_c).all():
            n_bad = int((~torch.isfinite(per_c)).sum().item())
            raise WaterFillError(
                f"estimate_per_channel_variance: {name} has {n_bad} non-finite "
                f"per-channel variance(s). Checkpoint is corrupt — refuse to "
                f"silently zero them."
            )
        out[f"{name}.weight"] = per_c
    return out


# ── Hessian estimation (1-step gradient approximation) ────────────────────


def _segmap_eval_roundtrip_render(
    model: nn.Module,
    inputs: torch.Tensor,
    frame_idx: torch.Tensor,
) -> torch.Tensor:
    """Render through the eval_roundtrip chain (uint8 quantize-decode).

    eval_roundtrip means the rendered RGB goes through 384→874→uint8→384
    so the proxy gradient matches the contest scorer's input distribution.
    Reuses tac.segmap_renderer._eval_roundtrip_chain — see CLAUDE.md
    non-negotiable. _eval_roundtrip_chain expects (B, T, C, H, W); SegMap
    forward returns (K, 3, H, W) so we wrap with T=1 then unwrap.
    """
    raw = model(inputs, frame_idx)  # (K, 3, H, W) in [0, 255]
    # Reuse the canonical chain from segmap_renderer to ensure parity with
    # contest eval. We import lazily to avoid pulling tensors at module load.
    from tac.segmap_renderer import _eval_roundtrip_chain  # noqa: PLC0415

    btchw = raw.unsqueeze(1)  # (K, 1, 3, H, W)
    out = _eval_roundtrip_chain(btchw)
    return out.squeeze(1)  # back to (K, 3, H, W)


def estimate_per_channel_hessian(
    model: nn.Module,
    calibration_inputs: torch.Tensor,
    calibration_targets: torch.Tensor,
    calibration_frame_idx: torch.Tensor,
    *,
    eval_roundtrip: bool = True,
    device: str = "cuda",
    extra_protected_patterns: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    """Return {<conv_name>.weight: Tensor[O]} per-channel Hessian.

    H_c ≈ Σ_pairs Σ_{i, kH, kW} (∂L_render/∂w[c, i, kH, kW])²
    aggregated over the (I, kH, kW) elements of each output channel.

    Args:
        model: SegMap (or compatible) renderer, on `device`.
        calibration_inputs: (K, 5, H, W) one-hot mask classes (float).
        calibration_targets: (K, 3, H, W) GT frames in [0, 255].
        calibration_frame_idx: (K,) long, frame indices for affine embedding.
        eval_roundtrip: MUST be True (CLAUDE.md). False raises WaterFillError.
        device: "cuda" (production) or "cpu" (opt-in with banner).
        extra_protected_patterns: forwarded to iter_eligible_conv_names.

    Raises:
        WaterFillError on eval_roundtrip=False, or NaN/Inf in any H_c.
    """
    if not eval_roundtrip:
        raise WaterFillError(
            "estimate_per_channel_hessian: eval_roundtrip=False is FORBIDDEN. "
            "CLAUDE.md non-negotiable: every training/curvature path uses "
            "eval_roundtrip=True. Proxy without roundtrip drifts 2-11x from "
            "auth on PoseNet."
        )
    if device == "mps":
        raise WaterFillError(
            "estimate_per_channel_hessian: device='mps' is FORBIDDEN. "
            "CLAUDE.md: MPS produces 23x PoseNet drift vs CUDA. Use cuda or cpu."
        )
    if device == "cpu":
        # Opt-in banner (Contrarian-required).
        print(
            "[CPU-FALLBACK] estimate_per_channel_hessian on CPU — Hessian "
            "numerics will differ from production CUDA path; rank-correlation "
            "expected but absolute values not comparable."
        )

    target_device = torch.device(device)
    model = model.to(target_device)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
        if p.grad is not None:
            p.grad = None

    inputs = calibration_inputs.to(target_device).float()
    targets = calibration_targets.to(target_device).float()
    frame_idx = calibration_frame_idx.to(target_device).long()

    rendered = _segmap_eval_roundtrip_render(model, inputs, frame_idx)
    if rendered.shape != targets.shape:
        # Common source of confusion — surface the mismatch loudly.
        raise WaterFillError(
            f"estimate_per_channel_hessian: rendered shape {tuple(rendered.shape)} "
            f"!= target shape {tuple(targets.shape)}. Check calibration_inputs "
            f"resolution matches calibration_targets."
        )

    loss = ((rendered - targets) ** 2).mean()
    loss.backward()

    eligible = set(iter_eligible_conv_names(model, extra_protected_patterns))
    out: dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if name not in eligible:
            continue
        if not isinstance(mod, nn.Conv2d):
            continue
        if mod.weight.grad is None:
            raise WaterFillError(
                f"estimate_per_channel_hessian: {name}.weight has no grad. "
                f"Layer disconnected from loss graph (check requires_grad / "
                f"frozen flag) — refuse to assign zero Hessian silently."
            )
        g = mod.weight.grad.detach()  # (O, I, kH, kW)
        per_c = (g.float() ** 2).reshape(g.shape[0], -1).sum(dim=1).cpu()
        if not torch.isfinite(per_c).all():
            n_bad = int((~torch.isfinite(per_c)).sum().item())
            raise WaterFillError(
                f"estimate_per_channel_hessian: {name} produced {n_bad} non-finite "
                f"per-channel Hessian(s). Loss is NaN/Inf — refuse to silently "
                f"zero. Fix upstream (check loss for inf/nan)."
            )
        out[f"{name}.weight"] = per_c

    # Clear grads so caller can reuse model.
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    return out


# ── water-fill core (Dykstra-rigorous λ-bisection) ────────────────────────


def _layout_per_channel_arrays(
    hessians: dict[str, torch.Tensor],
    variances: dict[str, torch.Tensor],
    channel_element_counts: dict[str, list[int]],
) -> tuple[list[tuple[str, int]], torch.Tensor, torch.Tensor]:
    """Flatten the per-layer per-channel data into parallel arrays.

    Returns (channel_index, utility_u, element_count_n).
    channel_index[i] = (layer_name, c) for channel index i in the flat arrays.
    """
    if set(hessians.keys()) != set(variances.keys()):
        raise WaterFillError(
            f"_layout_per_channel_arrays: hessians keys {sorted(hessians)} != "
            f"variances keys {sorted(variances)}. Run both estimators on the "
            f"SAME model with the SAME extra_protected_patterns."
        )
    if set(hessians.keys()) != set(channel_element_counts.keys()):
        raise WaterFillError(
            "_layout_per_channel_arrays: channel_element_counts keys differ "
            "from hessians keys."
        )

    channel_index: list[tuple[str, int]] = []
    util_list: list[float] = []
    count_list: list[int] = []
    for name in sorted(hessians.keys()):
        h = hessians[name]
        v = variances[name]
        n_elements = channel_element_counts[name]
        if h.shape != v.shape:
            raise WaterFillError(
                f"_layout: shape mismatch on {name}: H={tuple(h.shape)} vs "
                f"σ²={tuple(v.shape)}"
            )
        if len(n_elements) != int(h.shape[0]):
            raise WaterFillError(
                f"_layout: {name} channel_element_counts length {len(n_elements)} "
                f"!= num channels {int(h.shape[0])}"
            )
        for c in range(int(h.shape[0])):
            hi = float(h[c].item())
            vi = float(v[c].item())
            if not (math.isfinite(hi) and math.isfinite(vi)):
                raise WaterFillError(
                    f"_layout: {name}[c={c}] H={hi} σ²={vi} non-finite"
                )
            if hi < 0 or vi < 0:
                raise WaterFillError(
                    f"_layout: {name}[c={c}] H={hi} σ²={vi} negative — invalid "
                    f"(both should be sums of squares)"
                )
            channel_index.append((name, c))
            util_list.append(hi * vi)
            count_list.append(int(n_elements[c]))

    util = torch.tensor(util_list, dtype=torch.float64)
    counts = torch.tensor(count_list, dtype=torch.int64)
    return channel_index, util, counts


def _water_fill_continuous(util: torch.Tensor, lam: float) -> torch.Tensor:
    """Eq. 1: b_c* = max(0, 0.5 * log2(u_c / λ)). Returns continuous bits/element."""
    safe_u = util.clamp(min=1e-300)  # avoid log(0); zero-u channels go to floor anyway
    return torch.clamp(0.5 * torch.log2(safe_u / lam), min=0.0)


def _discrete_total_bits(b_continuous: torch.Tensor, counts: torch.Tensor) -> tuple[
    torch.Tensor, int
]:
    """Map continuous bits to discrete Q ladder, return (Q array, total bits)."""
    q = torch.tensor(
        [qint_for_bits(float(b)) for b in b_continuous.tolist()], dtype=torch.int64
    )
    bits_per_elt = torch.tensor(
        [bits_for_qint(int(qq)) for qq in q.tolist()], dtype=torch.float64
    )
    total = int((bits_per_elt * counts.double()).sum().item())
    return q, total


def water_fill_bit_budget(
    hessians: dict[str, torch.Tensor],
    variances: dict[str, torch.Tensor],
    channel_element_counts: dict[str, list[int]],
    total_bits: int,
    *,
    bisect_iters: int = 80,
    budget_tol_frac: float = 0.01,
) -> dict[str, list[int]]:
    """Allocate per-channel qint_max via Shannon reverse water-filling.

    Args:
        hessians: {<conv>.weight: Tensor[O]} per-channel curvature.
        variances: {<conv>.weight: Tensor[O]} per-channel σ².
        channel_element_counts: {<conv>.weight: [n_0, n_1, ...]} where
            n_c = I * kH * kW for channel c. Used to convert bits/element
            to total bits in budget accounting.
        total_bits: scalar B, total signed-integer bit budget (sum over all
            channels of bits_per_element(Q_c) * elements_per_channel).
        bisect_iters: λ-bisection iterations (default 80; doc says ≥50).
        budget_tol_frac: ±fractional slack on B (default 1%).

    Returns:
        {<conv>.weight: [Q_0, Q_1, ...]} suitable as `per_key_qint_max`
        kwarg of pack_payload_tar_xz.

    Raises:
        WaterFillError if budget infeasible (e.g. all-Q1 floor exceeds B).
    """
    if total_bits <= 0:
        raise WaterFillError(f"water_fill_bit_budget: total_bits={total_bits} ≤ 0")

    channel_index, util, counts = _layout_per_channel_arrays(
        hessians, variances, channel_element_counts
    )
    if util.numel() == 0:
        raise WaterFillError(
            "water_fill_bit_budget: zero eligible channels. Either every "
            "conv layer is protected, or hessians/variances dicts are empty."
        )

    # Floor feasibility: even if every channel gets Q=1, do we exceed B?
    floor_bits = float(bits_for_qint(1)) * float(counts.sum().item())
    if floor_bits > total_bits * (1.0 + budget_tol_frac):
        raise WaterFillError(
            f"water_fill_bit_budget: infeasible — Q=1 floor needs "
            f"{floor_bits:.0f} bits but budget is {total_bits}. "
            f"Reduce model size OR raise budget."
        )

    # Ceiling feasibility: is the Q=31 ceiling ≤ B? If so, just allocate ceiling.
    ceil_bits = float(bits_for_qint(31)) * float(counts.sum().item())
    if ceil_bits <= total_bits:
        # Budget exceeds even max precision; ship Q=31 everywhere.
        # (Rare in practice; defensive.)
        result: dict[str, list[int]] = {}
        for (name, _), q in zip(channel_index, [31] * len(channel_index)):
            result.setdefault(name, []).append(q)
        return result

    # Bracket λ. Higher λ → fewer bits assigned. We want Σ bits ≤ B.
    # u_c spans many orders; bracket generously.
    util_pos = util[util > 0]
    if util_pos.numel() == 0:
        # All-zero utility — every channel falls to floor Q=1.
        result_zero: dict[str, list[int]] = {}
        for name, _ in channel_index:
            result_zero.setdefault(name, []).append(1)
        # Verify floor matches budget within tol; if floor > B we already raised.
        return result_zero
    u_min = float(util_pos.min().item())
    u_max = float(util_pos.max().item())
    lam_lo = u_min * 2 ** (-20)  # gives lots of bits
    lam_hi = u_max * 2 ** (+10)  # gives few bits

    # Bisect: at lam_lo, total_bits is HIGH; at lam_hi, total_bits is LOW.
    # We want largest λ such that total ≤ B (equivalently smallest λ s.t. total ≤ B).
    best_q: torch.Tensor | None = None
    best_total: int = -1
    target_lo = int(math.floor(total_bits * (1.0 - budget_tol_frac)))
    target_hi = int(math.ceil(total_bits * (1.0 + budget_tol_frac)))

    for _ in range(bisect_iters):
        lam_mid = math.sqrt(max(lam_lo, 1e-300) * max(lam_hi, 1e-300))
        b_cont = _water_fill_continuous(util, lam_mid)
        q, total = _discrete_total_bits(b_cont, counts)
        if total > total_bits:
            # Too many bits → push λ up
            lam_lo = lam_mid
        else:
            # Within budget → record candidate, try to use more bits (push λ down)
            best_q = q
            best_total = total
            lam_hi = lam_mid
        if target_lo <= total <= target_hi:
            best_q = q
            best_total = total
            break

    if best_q is None:
        # Could not find any λ that fits; fall back to "Q=1 floor everywhere"
        # if floor fits, otherwise raise.
        if floor_bits <= total_bits:
            best_q = torch.ones(util.numel(), dtype=torch.int64)
            best_total = int(floor_bits)
        else:
            raise WaterFillError(
                f"water_fill_bit_budget: bisect failed to find feasible λ "
                f"after {bisect_iters} iterations. budget={total_bits}, "
                f"floor={floor_bits:.0f}, ceiling={ceil_bits:.0f}"
            )

    # Final peel-down pass: if best_q overshoots after re-discretisation
    # (shouldn't given guard above, but defensive), step down lowest-utility
    # channels by one Q level until ≤ B.
    if best_total > total_bits:
        sorted_channels = sorted(
            range(util.numel()), key=lambda i: float(util[i].item())
        )
        peel_idx = 0
        while best_total > total_bits and peel_idx < len(sorted_channels):
            ci = sorted_channels[peel_idx]
            current_q = int(best_q[ci].item())
            if current_q == 1:
                peel_idx += 1
                continue
            level_idx = QINT_LEVELS.index(current_q)
            new_q = QINT_LEVELS[level_idx - 1]
            best_total -= int(
                round((bits_for_qint(current_q) - bits_for_qint(new_q))
                       * float(counts[ci].item()))
            )
            best_q[ci] = new_q
            peel_idx += 1
        if best_total > total_bits:
            raise WaterFillError(
                f"water_fill_bit_budget: peel-down could not reach budget. "
                f"final={best_total}, target={total_bits}"
            )

    # Reassemble per-layer lists
    result_out: dict[str, list[int]] = {}
    for (name, _), q in zip(channel_index, best_q.tolist()):
        result_out.setdefault(name, []).append(int(q))
    return result_out


# ── orchestrator ──────────────────────────────────────────────────────────


def _channel_element_counts_from_model(
    model: nn.Module,
    extra_protected_patterns: tuple[str, ...] = (),
) -> dict[str, list[int]]:
    """Per-eligible-conv: list of I*kH*kW per output channel (constant per layer)."""
    out: dict[str, list[int]] = {}
    eligible = set(iter_eligible_conv_names(model, extra_protected_patterns))
    for name, mod in model.named_modules():
        if name not in eligible:
            continue
        if not isinstance(mod, nn.Conv2d):
            continue
        o, i, kh, kw = mod.weight.shape
        out[f"{name}.weight"] = [int(i * kh * kw)] * int(o)
    return out


def export_with_water_filling(
    model: nn.Module,
    calibration_inputs: torch.Tensor,
    calibration_targets: torch.Tensor,
    calibration_frame_idx: torch.Tensor,
    total_bits: int,
    output_path: str | Path,
    *,
    device: str = "cuda",
    eval_roundtrip: bool = True,
    extra_protected_patterns: tuple[str, ...] = (),
    verify_tol: float = 1e-3,
) -> dict[str, object]:
    """End-to-end water-filling export.

    1. Estimate Hessian (eval_roundtrip enforced).
    2. Estimate variance.
    3. Water-fill to total_bits.
    4. pack_payload_tar_xz with per_key_qint_max.
    5. verify_roundtrip with relaxed tol.

    Returns a summary dict.
    """
    output_path = str(output_path)

    hessians = estimate_per_channel_hessian(
        model,
        calibration_inputs,
        calibration_targets,
        calibration_frame_idx,
        eval_roundtrip=eval_roundtrip,
        device=device,
        extra_protected_patterns=extra_protected_patterns,
    )
    variances = estimate_per_channel_variance(
        model, extra_protected_patterns=extra_protected_patterns
    )
    counts = _channel_element_counts_from_model(
        model, extra_protected_patterns=extra_protected_patterns
    )
    qint_assignment = water_fill_bit_budget(
        hessians, variances, counts, total_bits
    )

    # Compute realised bit total for the summary.
    realised_bits = 0
    for name, q_list in qint_assignment.items():
        per_c_counts = counts[name]
        for q, n in zip(q_list, per_c_counts):
            realised_bits += int(round(bits_for_qint(int(q)) * float(n)))

    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    pack_payload_tar_xz(
        state_dict, output_path, per_key_qint_max=qint_assignment
    )
    mse_map = verify_roundtrip(state_dict, output_path, tol=verify_tol)
    payload_bytes = Path(output_path).stat().st_size

    return {
        "qint_assignment": qint_assignment,
        "realised_bits": realised_bits,
        "payload_bytes": int(payload_bytes),
        "roundtrip_mse_max": float(max(mse_map.values())) if mse_map else 0.0,
        "allocations_per_layer": {
            name: {
                "n_channels": len(q_list),
                "min_q": int(min(q_list)),
                "max_q": int(max(q_list)),
                "mean_q": float(sum(q_list) / len(q_list)),
            }
            for name, q_list in qint_assignment.items()
        },
    }


__all__ = [
    "WaterFillError",
    "QINT_LEVELS",
    "QINT_BITS",
    "bits_for_qint",
    "qint_for_bits",
    "estimate_per_channel_variance",
    "estimate_per_channel_hessian",
    "water_fill_bit_budget",
    "export_with_water_filling",
]
