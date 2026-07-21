# SPDX-License-Identifier: MIT
"""Exact-anchor costate ORGAN v2 (advisory, deterministic, no learned parameters).

The critical-path composition is deliberately inspectable::

    lambda(pair, site) = exact_gap * visibility * realizability * byte_price

Every factor is a separately returned field with a canonical LawRef.  This module
does not launch, signal, mutate a run, or import a trainer.  It is a readback and
ranking primitive beside (not in place of) the v1 factorized adjoint.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tac.canonical_equations.day_consolidation_laws_20260720 import (
    RATE_PRICE_S_PER_BYTE,
    breakeven_bytes,
)

SCHEMA = "costate_organ_exact_anchor.v2"
AXIS = "[macOS-CPU advisory] NON-PROMOTABLE"
COMPOSITION_EQUATION_ID = "costate_organ_exact_anchor_product_v2"
BREAKEVEN_EQUATION_ID = "realization_breakeven_bytes_v1"
POOL_KKT_EQUATION_ID = "witness_measured_reverse_waterfill_v1"
HEAD_EQUATION_ID = "segnet_head_rank4_linear_flipdist_v1"
RESIZE_EQUATION_ID = "separable_resize_full_kernel_direct_sum_v1"
SUPPORT_FILL_EQUATION_ID = "f32_receiver_arithmetic_exactness_admissibility_v1"

# MEASURED n600 capstone anchor (#547) and exact inverse-solve confirmation (#549).
EXACT_ANCHOR_DSEG = 0.00015196
EXACT_ANCHOR_DPOSE = 0.00010184

# MEASURED #580 full real-linear direct-sum nullity per channel.  The old 22.7%
# zero-weight mask is emitted only as a stale comparator and is never consumed.
FULL_KERNEL_NULLITY = 0.806742315223
FULL_KERNEL_VISIBLE = 1.0 - FULL_KERNEL_NULLITY
STALE_AXIS_ALIGNED_NULLITY = 0.22696926089315625

# MEASURED M1 design-time gate: 11,453 realizable of 38,077 banked sites.
DESIGN_REALIZABILITY = 11_453.0 / 38_077.0

FISHER_BANK_SHA256 = "765457d424eaf1de7e05ed8703853175ef415bd3f19fb00137a74a29de52ae00"
FISHER_BANK_ROWS = 38_077
FISHER_BANK_SCHEMA = "r1b5_fisher_ev_ordering_jsonl.v1"

CLASS_FLIP_PRIOR = {"Road": 0.50, "Lane": 0.19, "Undrivable": 0.13}
LANE_SKIP_LIMITED_FRAC = 6205.0 / 8072.0
HEAD_GAUGE_NORM_FRAC = 0.5236


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return value


def score_debt(*, current_dseg: float, current_dpose: float,
               anchor_dseg: float = EXACT_ANCHOR_DSEG,
               anchor_dpose: float = EXACT_ANCHOR_DPOSE) -> dict[str, float]:
    """Exact non-rate score gap to the custodied inverse-solve anchor."""
    ds = _finite_nonnegative("current_dseg", current_dseg)
    dp = _finite_nonnegative("current_dpose", current_dpose)
    asg = _finite_nonnegative("anchor_dseg", anchor_dseg)
    apo = _finite_nonnegative("anchor_dpose", anchor_dpose)
    seg = 100.0 * max(ds - asg, 0.0)
    pose = max(math.sqrt(10.0 * dp) - math.sqrt(10.0 * apo), 0.0)
    return {"seg_s": seg, "pose_s": pose, "total_s": seg + pose}


def exact_resize_adjoint_four_tap(output_cotangent: float,
                                  tap_weights: Sequence[float]) -> tuple[float, ...]:
    """Closed four-tap bilinear adjoint for one scorer-input coordinate."""
    if len(tap_weights) != 4:
        raise ValueError("exact bilinear resize adjoint requires exactly four taps")
    weights = tuple(float(v) for v in tap_weights)
    if any(not math.isfinite(v) or v < 0.0 for v in weights):
        raise ValueError("tap weights must be finite and nonnegative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=2e-7):
        raise ValueError("bilinear tap weights must sum to one")
    g = float(output_cotangent)
    if not math.isfinite(g):
        raise ValueError("output_cotangent must be finite")
    return tuple(g * v for v in weights)


def closed_form_seg_pullback(*, score_costate_s: float, pair_head_norm: float,
                             tap_weights: Sequence[float], frame_index: int,
                             orientation: float = 1.0) -> dict[str, Any]:
    """Compose the frozen rank-4 pair chart with the exact resize adjoint.

    ``score_costate_s`` is the pairwise Seg debt in score units.  The frozen
    head's pair-normal norm maps that scalar into the exact penultimate-feature
    cotangent magnitude; the four-tap resize transpose then maps it to camera
    coordinates.  The signed pair direction stays explicit in ``orientation``.
    Frame 0 is structurally absent from SegNet and therefore returns the exact
    zero pullback.  No autograd/VJP or learned direction participates.
    """
    costate = _finite_nonnegative("score_costate_s", score_costate_s)
    head_norm = float(pair_head_norm)
    orient = float(orientation)
    if not math.isfinite(head_norm) or head_norm <= 0.0:
        raise ValueError("pair_head_norm must be finite and > 0")
    if not math.isfinite(orient) or orient not in {-1.0, 1.0}:
        raise ValueError("orientation must be exactly -1 or +1")
    scorer_cotangent = (0.0 if int(frame_index) == 0 else
                        orient * costate * head_norm)
    return {
        "camera_cotangent_four_tap": exact_resize_adjoint_four_tap(
            scorer_cotangent, tap_weights),
        "scorer_cotangent": scorer_cotangent,
        "frame_index": int(frame_index),
        "head_rank": 4,
        "all_class_gauge_null": True,
        "vjp_used": False,
        "equation_ids": [HEAD_EQUATION_ID, RESIZE_EQUATION_ID],
    }


def visibility_factor(*, task: str, frame_index: int, channel: str,
                      site_space: str = "camera", spatial_scale_px: float | None = None,
                      direction_alignment: float = 1.0) -> dict[str, Any]:
    """Return task/frame/channel visibility without conflating Seg and Pose."""
    task = str(task).lower()
    channel = str(channel).lower()
    if task not in {"seg", "pose"}:
        raise ValueError("task must be seg or pose")
    align = min(max(_finite_nonnegative("direction_alignment", direction_alignment), 0.0), 1.0)
    reasons: list[str] = []
    if task == "seg" and int(frame_index) == 0:
        value = 0.0
        reasons.append("frame_0_structurally_seg_free")
    elif task == "pose" and channel in {"u", "v", "chroma"} and (
            spatial_scale_px is not None and float(spatial_scale_px) < 2.0):
        value = 0.0
        reasons.append("pose_chroma_sub_2px_box_invisible")
    else:
        base = FULL_KERNEL_VISIBLE if site_space == "camera" else 1.0
        value = base * align
        reasons.append("full_kernel_real_linear_visibility" if site_space == "camera"
                       else "scorer_space_visible")
    return {
        "value": value,
        "task": task,
        "frame_index": int(frame_index),
        "channel": channel,
        "direction_alignment": align,
        "full_kernel_nullity": FULL_KERNEL_NULLITY,
        "stale_axis_aligned_nullity_not_consumed": STALE_AXIS_ALIGNED_NULLITY,
        "equation_id": RESIZE_EQUATION_ID,
        "reasons": reasons,
    }


def realizability_factor(*, route: str = "band_design", requested: int | None = None,
                         survived_clean: int | None = None,
                         formulation_valid: bool = True,
                         apparatus_valid: bool = True,
                         strength: float = 1.0) -> dict[str, Any]:
    """Quantization/parse-back gate; returns zero on invalid apparatus/formulation."""
    route = str(route)
    why: list[str] = []
    if not apparatus_valid:
        value = 0.0
        why.append("apparatus_invalid")
    elif not formulation_valid:
        value = 0.0
        why.append("formulation_scoped_negative")
    elif requested is not None or survived_clean is not None:
        if not isinstance(requested, int) or not isinstance(survived_clean, int):
            raise ValueError("requested and survived_clean must be supplied together as ints")
        if requested <= 0 or survived_clean < 0 or survived_clean > requested:
            raise ValueError("invalid requested/survived_clean counts")
        value = survived_clean / requested
        why.append("measured_clean_survival_fraction")
    elif route == "band_design":
        value = DESIGN_REALIZABILITY
        why.append("m1_design_time_anchor_11453_of_38077")
    elif route in {"temporal_stop", "scorer_space"}:
        value = 1.0
        why.append("no_uint8_write_required")
    else:
        value = 0.0
        why.append("unrecognized_route_fail_closed")
    scale = min(max(_finite_nonnegative("strength", strength), 0.0), 1.0)
    return {
        "value": value * scale,
        "raw_value": value,
        "strength": scale,
        "route": route,
        "equation_ids": [
            "witness_realization_lsb_regime_v1",
            "bounded_uint8_resize_preimage_cell_feasibility_v1",
        ],
        "reasons": why,
    }


def byte_price_factor(*, realized_recovery_s: float, charged_bytes: int) -> dict[str, Any]:
    """Net rent fraction from the canonical domain-refined break-even equation."""
    recovery = _finite_nonnegative("realized_recovery_s", realized_recovery_s)
    if not isinstance(charged_bytes, int) or charged_bytes < 0:
        raise ValueError("charged_bytes must be an integer >= 0")
    limit = breakeven_bytes(recovery)
    value = (0.0 if recovery == 0.0 else
             max(0.0, 1.0 - (charged_bytes * RATE_PRICE_S_PER_BYTE) / recovery))
    return {
        "value": value,
        "charged_bytes": charged_bytes,
        "realized_recovery_s": recovery,
        "breakeven_bytes": limit,
        "rate_price_s_per_byte": RATE_PRICE_S_PER_BYTE,
        "equation_id": BREAKEVEN_EQUATION_ID,
        "required_registry_event": "domain_refined",
        "pays_rent": charged_bytes <= limit if recovery > 0.0 else charged_bytes == 0,
    }


def dual_metric_readback(a: Sequence[float], b: Sequence[float],
                         fisher_diag: Sequence[float]) -> dict[str, Any]:
    """Emit Euclidean and diagonal-Fisher cosines separately, including sign flips."""
    if not (len(a) == len(b) == len(fisher_diag)) or len(a) == 0:
        raise ValueError("metric vectors must have the same nonzero length")
    av, bv, fv = tuple(map(float, a)), tuple(map(float, b)), tuple(map(float, fisher_diag))
    if any(not math.isfinite(v) for v in (*av, *bv, *fv)) or any(v < 0.0 for v in fv):
        raise ValueError("metric inputs must be finite and Fisher weights nonnegative")

    def cosine(x: Sequence[float], y: Sequence[float]) -> float | None:
        nx = math.sqrt(sum(v * v for v in x))
        ny = math.sqrt(sum(v * v for v in y))
        return (None if nx == 0.0 or ny == 0.0 else
                sum(u * v for u, v in zip(x, y, strict=True)) / (nx * ny))

    euclid = cosine(av, bv)
    sa = tuple(math.sqrt(w) * v for w, v in zip(fv, av, strict=True))
    sb = tuple(math.sqrt(w) * v for w, v in zip(fv, bv, strict=True))
    fisher = cosine(sa, sb)
    sign_flip = (euclid is not None and fisher is not None and euclid * fisher < 0.0)
    na = math.sqrt(sum(v * v for v in av))
    nb = math.sqrt(sum(v * v for v in bv))
    fna = math.sqrt(sum(w * v * v for w, v in zip(fv, av, strict=True)))
    fnb = math.sqrt(sum(w * v * v for w, v in zip(fv, bv, strict=True)))
    return {
        "euclidean_cosine": euclid,
        "fisher_cosine": fisher,
        "sign_flip_informative": sign_flip,
        "euclidean_relative_norm": None if nb == 0.0 else na / nb,
        "fisher_relative_norm": None if fnb == 0.0 else fna / fnb,
        "equation_id": "optimal_metric_unification_v1",
        "blend_forbidden": True,
    }


def apparatus_validity(*, flags: Mapping[str, Any] | None = None,
                       ema_reset_verified: bool = False,
                       topology_event: bool = False,
                       maturity: str = "_dev") -> dict[str, Any]:
    flags = flags or {}
    contaminated = str(flags.get("ckpt-every", flags.get("ckpt_every", ""))) == "1"
    maturity = maturity if maturity in {"_dev", "_prod"} else "_dev"
    return {
        "valid_for_backtest": not contaminated,
        "bench_contaminated": contaminated,
        "bench_reason": "ckpt-every-1 observer cadence poison" if contaminated else None,
        "ema_lag_correction": "eligible" if ema_reset_verified else "unknown_not_applied",
        "xi_transport_eligible": not topology_event,
        "xi_transport_refusal": "sparse_topology_event" if topology_event else None,
        "maturity": maturity,
        "pointer_eligible": maturity == "_prod",
    }


def ema_delag_delta(*, observed_delta: float, estimated_lag_delta: float | None,
                    reset_verified: bool) -> dict[str, Any]:
    """Remove a custodied EMA lag estimate, or preserve the observation fail-closed."""
    observed = float(observed_delta)
    if not math.isfinite(observed):
        raise ValueError("observed_delta must be finite")
    if not reset_verified or estimated_lag_delta is None:
        return {
            "value": observed,
            "applied": False,
            "status": "unknown_not_applied",
            "observed_delta": observed,
            "estimated_lag_delta": None,
        }
    lag = float(estimated_lag_delta)
    if not math.isfinite(lag):
        raise ValueError("estimated_lag_delta must be finite")
    return {
        "value": observed - lag,
        "applied": True,
        "status": "verified_lag_removed",
        "observed_delta": observed,
        "estimated_lag_delta": lag,
    }


def pontryagin_lqr_conformance_fixture(*, horizon: int = 12,
                                      relaxation: float = 0.2,
                                      tolerance: float = 1e-11) -> dict[str, Any]:
    """Run a bounded scalar-LQR forward/backward conformance oracle.

    This intentionally fixed synthetic fixture validates adjoint conventions; it
    is not a model of the nonlinear training plant.  It compares a relaxed
    Pontryagin sweep against the finite-horizon Riccati solution, central-
    differences both Hamiltonian derivatives, and refuses the second algebraic
    Riccati root because its closed loop is non-stabilizing.
    """
    if not isinstance(horizon, int) or horizon < 2:
        raise ValueError("horizon must be an integer >= 2")
    relax = float(relaxation)
    tol = float(tolerance)
    if not 0.0 < relax <= 0.25:
        raise ValueError("fixture relaxation must be in (0, 0.25]")
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("tolerance must be finite and > 0")

    # Fixed, dimensionless, control-inactive scalar problem.  Keeping the
    # coefficients literal makes the fixture independent of learned/runtime state.
    a, b, q, r, q_terminal = 0.8, 0.4, 1.0, 0.5, 1.0
    x0, u_limit = 0.3, 2.0

    p = [0.0] * (horizon + 1)
    gain = [0.0] * horizon
    p[horizon] = q_terminal
    for k in range(horizon - 1, -1, -1):
        denominator = r + b * b * p[k + 1]
        gain[k] = b * a * p[k + 1] / denominator
        p[k] = q + a * a * p[k + 1] - (
            a * b * p[k + 1]) ** 2 / denominator

    analytic_x = [x0]
    analytic_u: list[float] = []
    for k in range(horizon):
        control = max(-u_limit, min(u_limit, -gain[k] * analytic_x[k]))
        analytic_u.append(control)
        analytic_x.append(a * analytic_x[k] + b * control)
    if any(math.isclose(abs(u), u_limit, rel_tol=0.0, abs_tol=1e-12)
           for u in analytic_u):
        raise ValueError("fixture unexpectedly activated the control bound")
    analytic_lambda = [p[k] * analytic_x[k] for k in range(horizon + 1)]

    controls = [0.0] * horizon
    residuals: list[float] = []
    for _iteration in range(2_000):
        states = [x0]
        for control in controls:
            states.append(a * states[-1] + b * control)
        costates = [0.0] * (horizon + 1)
        costates[horizon] = q_terminal * states[horizon]
        for k in range(horizon - 1, -1, -1):
            costates[k] = q * states[k] + a * costates[k + 1]
        projected = [
            max(-u_limit, min(u_limit, -b * costates[k + 1] / r))
            for k in range(horizon)
        ]
        updated = [
            (1.0 - relax) * controls[k] + relax * projected[k]
            for k in range(horizon)
        ]
        residual = max(abs(updated[k] - controls[k]) for k in range(horizon))
        residuals.append(residual)
        controls = updated
        if residual <= tol:
            break
    else:
        raise ValueError("Pontryagin fixture failed to converge")

    # Recompute state/costate at the converged control before comparing custody.
    states = [x0]
    for control in controls:
        states.append(a * states[-1] + b * control)
    costates = [0.0] * (horizon + 1)
    costates[horizon] = q_terminal * states[horizon]
    for k in range(horizon - 1, -1, -1):
        costates[k] = q * states[k] + a * costates[k + 1]

    eps = 1e-6

    def hamiltonian(x: float, u: float, lambda_next: float) -> float:
        return 0.5 * q * x * x + 0.5 * r * u * u + lambda_next * (a * x + b * u)

    fd_x_error = 0.0
    fd_u_error = 0.0
    projected_error = 0.0
    for k in range(horizon):
        lam_next = costates[k + 1]
        fd_x = (hamiltonian(states[k] + eps, controls[k], lam_next)
                - hamiltonian(states[k] - eps, controls[k], lam_next)) / (2.0 * eps)
        fd_u = (hamiltonian(states[k], controls[k] + eps, lam_next)
                - hamiltonian(states[k], controls[k] - eps, lam_next)) / (2.0 * eps)
        fd_x_error = max(fd_x_error, abs(fd_x - costates[k]))
        fd_u_error = max(fd_u_error, abs(fd_u - (r * controls[k] + b * lam_next)))
        projected_control = max(-u_limit, min(u_limit, -b * lam_next / r))
        projected_error = max(projected_error, abs(controls[k] - projected_control))

    # Infinite-horizon scalar DARE has two algebraic roots here.  Only the
    # positive root yields |a-bK|<1; treating the other as valid is a fail-open bug.
    qa = b * b
    qb = r * (1.0 - a * a) - q * b * b
    qc = -q * r
    discriminant = qb * qb - 4.0 * qa * qc
    roots = ((-qb + math.sqrt(discriminant)) / (2.0 * qa),
             (-qb - math.sqrt(discriminant)) / (2.0 * qa))
    root_rows = []
    for root in roots:
        denominator = r + b * b * root
        root_gain = math.inf if denominator == 0.0 else b * a * root / denominator
        closed_loop_abs = math.inf if not math.isfinite(root_gain) else abs(a - b * root_gain)
        stable = root >= 0.0 and denominator > 0.0 and closed_loop_abs < 1.0
        root_rows.append({
            "p": root,
            "gain": root_gain,
            "closed_loop_abs": closed_loop_abs,
            "accepted": stable,
            "rejection": None if stable else "non_stabilizing_riccati_root",
        })
    if sum(int(row["accepted"]) for row in root_rows) != 1:
        raise ValueError("fixture did not isolate exactly one stabilizing Riccati root")

    monotone = all(residuals[i + 1] <= residuals[i] + 1e-15
                   for i in range(len(residuals) - 1))
    return {
        "schema": "costate_organ_v2.pontryagin_lqr_conformance.v1",
        "fixture_only": True,
        "live_control_authority": False,
        "learned_parameters": 0,
        "horizon": horizon,
        "iterations": len(residuals),
        "sweep_residuals": residuals,
        "sweep_residual_monotone": monotone,
        "control_max_abs_error_vs_analytic": max(
            abs(controls[k] - analytic_u[k]) for k in range(horizon)),
        "costate_max_abs_error_vs_analytic": max(
            abs(costates[k] - analytic_lambda[k]) for k in range(horizon + 1)),
        "hamiltonian_fd_x_max_abs_error": fd_x_error,
        "hamiltonian_fd_u_max_abs_error": fd_u_error,
        "projected_control_max_abs_error": projected_error,
        "riccati_roots": root_rows,
        "actuation": "NONE",
        "axis": "[synthetic local-CPU conformance] NON-PROMOTABLE",
    }


def xi_transport_factor(*, transport_cosine: float | None,
                        sparse_topology_event: bool) -> dict[str, Any]:
    """Optional pair-to-pair xi transport; topology events refuse transport."""
    if sparse_topology_event:
        return {
            "value": 0.0,
            "applied": False,
            "refusal": "sparse_topology_event",
            "equation_id": "worldsheet_transport_residual_event_rate_v1",
        }
    if transport_cosine is None:
        return {
            "value": 1.0,
            "applied": False,
            "refusal": "transport_not_supplied_identity_used",
            "equation_id": "worldsheet_transport_residual_event_rate_v1",
        }
    value = float(transport_cosine)
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError("transport_cosine must be finite in [-1,1]")
    return {
        "value": max(value, 0.0),
        "applied": True,
        "refusal": "negative_transport_alignment_clipped" if value < 0.0 else None,
        "equation_id": "worldsheet_transport_residual_event_rate_v1",
    }


@dataclass(frozen=True)
class OrganV2Factors:
    exact_gap: float
    visibility: float
    realizability: float
    byte_price: float

    @property
    def lambda_value(self) -> float:
        return self.exact_gap * self.visibility * self.realizability * self.byte_price

    def to_dict(self) -> dict[str, float]:
        return {**asdict(self), "lambda": self.lambda_value}


def compose_lambda(*, exact_gap: float, visibility: float, realizability: float,
                   byte_price: float) -> OrganV2Factors:
    values = {
        "exact_gap": _finite_nonnegative("exact_gap", exact_gap),
        "visibility": _finite_nonnegative("visibility", visibility),
        "realizability": _finite_nonnegative("realizability", realizability),
        "byte_price": _finite_nonnegative("byte_price", byte_price),
    }
    if any(v > 1.0 for k, v in values.items() if k != "exact_gap"):
        raise ValueError("visibility, realizability, and byte_price must be <= 1")
    return OrganV2Factors(**values)


def pool_aware_rank(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """KKT-marginal ordering with same-pool remaining-opportunity caps."""
    candidates = [dict(r) for r in rows]
    candidates.sort(key=lambda r: (-float(r.get("lambda", 0.0)), str(r.get("candidate", ""))))
    used: dict[str, float] = {}
    out: list[dict[str, Any]] = []
    for row in candidates:
        pool = str(row.get("opportunity_pool", "UNSCOPED"))
        raw_ceiling = float(row.get("pool_ceiling_s", math.inf))
        ceiling = (raw_ceiling if math.isinf(raw_ceiling) and raw_ceiling > 0.0 else
                   _finite_nonnegative("pool_ceiling_s", raw_ceiling))
        raw = _finite_nonnegative("lambda", float(row.get("lambda", 0.0)))
        remainder = max(ceiling - used.get(pool, 0.0), 0.0)
        marginal = min(raw, remainder)
        used[pool] = used.get(pool, 0.0) + marginal
        row.update({
            "pool_kkt_marginal": marginal,
            "pool_remaining_after": max(remainder - marginal, 0.0),
            "pool_equation_id": POOL_KKT_EQUATION_ID,
            "same_pool_addition_forbidden": True,
        })
        out.append(row)
    out.sort(key=lambda r: (-r["pool_kkt_marginal"], str(r.get("candidate", ""))))
    return out


def aggregate_readback(state: Mapping[str, Any], *, flags: Mapping[str, Any] | None = None,
                       maturity: str = "_dev") -> dict[str, Any]:
    """Cheap shadow/digest row. It exposes debt but refuses to invent pair/site factors."""
    ds, dp = state.get("d_seg"), state.get("d_pose")
    valid_numbers = isinstance(ds, (int, float)) and isinstance(dp, (int, float))
    debt = score_debt(current_dseg=float(ds), current_dpose=float(dp)) if valid_numbers else None
    apparatus = apparatus_validity(flags=flags, maturity=maturity)
    return {
        "schema": SCHEMA,
        "status": "SENSE_READY_SITE_UNBOUND" if debt is not None else "UNAVAILABLE_NO_VERDICT",
        "exact_anchor": {
            "d_seg": EXACT_ANCHOR_DSEG,
            "d_pose": EXACT_ANCHOR_DPOSE,
            "support_fill": "fp32_exact_canonical",
            "support_fill_equation_id": SUPPORT_FILL_EQUATION_ID,
        },
        "score_debt": debt,
        "lambda": None,
        "why_lambda_null": "pair/site/channel/realization/charged-byte custody required",
        "factor_order": ["exact_gap", "visibility", "realizability", "byte_price"],
        "law_refs": [
            HEAD_EQUATION_ID, RESIZE_EQUATION_ID, BREAKEVEN_EQUATION_ID,
            POOL_KKT_EQUATION_ID, COMPOSITION_EQUATION_ID,
        ],
        "visibility": {
            "full_kernel_nullity": FULL_KERNEL_NULLITY,
            "full_kernel_visible": FULL_KERNEL_VISIBLE,
            "frame_0_seg_lambda": 0.0,
            "pose_chroma_sub_2px": 0.0,
        },
        "realizability": {
            "design_anchor": DESIGN_REALIZABILITY,
            "learned_parameters": 0,
            "residual_default": "OFF",
        },
        "rate_gauge": {
            "head_norm_frac": HEAD_GAUGE_NORM_FRAC,
            "lambda_in_gauge": 0.0,
            "scope": "tested dense fixed-shape grammar only",
        },
        "fisher_bank": {
            "sha256": FISHER_BANK_SHA256,
            "schema": FISHER_BANK_SCHEMA,
            "rows": FISHER_BANK_ROWS,
            "xi_transport": "optional; refused on sparse topology events",
        },
        "class_priors": {**CLASS_FLIP_PRIOR, "Lane_skip_limited": LANE_SKIP_LIMITED_FRAC},
        "apparatus": apparatus,
        "opportunity_pool_formalization": "FORMALIZATION_PENDING_DEDICATED_EQUATION_ABSENT",
        "actuation": "NONE",
        "axis": AXIS,
        "score_claim": False,
        "pointer_changed": False,
    }


def latest_equation_event(equation_id: str, registry_path: str | Path) -> dict[str, Any] | None:
    """Read the latest registry event for one equation without mutating the ledger."""
    import json

    latest = None
    for raw in Path(registry_path).read_text(errors="replace").splitlines():
        try:
            row = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if isinstance(row, dict) and row.get("equation_id") == equation_id:
            latest = row
    return latest


__all__ = [
    "AXIS",
    "BREAKEVEN_EQUATION_ID",
    "COMPOSITION_EQUATION_ID",
    "DESIGN_REALIZABILITY",
    "EXACT_ANCHOR_DPOSE",
    "EXACT_ANCHOR_DSEG",
    "FISHER_BANK_SHA256",
    "FULL_KERNEL_NULLITY",
    "FULL_KERNEL_VISIBLE",
    "OrganV2Factors",
    "aggregate_readback",
    "apparatus_validity",
    "byte_price_factor",
    "closed_form_seg_pullback",
    "compose_lambda",
    "dual_metric_readback",
    "ema_delag_delta",
    "exact_resize_adjoint_four_tap",
    "latest_equation_event",
    "pontryagin_lqr_conformance_fixture",
    "pool_aware_rank",
    "realizability_factor",
    "score_debt",
    "visibility_factor",
    "xi_transport_factor",
]
