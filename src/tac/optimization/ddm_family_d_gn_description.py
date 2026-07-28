# SPDX-License-Identifier: MIT
"""family-d: Gauss-Newton/CG second-order solve in DESCRIPTION coordinates (ddm_fd1).

This module EXTENDS the j2 joint-descent engine
(:mod:`tac.optimization.direct_description_joint_descent`) with the gc5-named
"built form" of the crux attack: per pair-block Gauss-Newton normal equations on
the SAME lifted description DOF, solved matrix-free by conjugate gradients,
through the EXACT j2 linearization (grammar paint -> uint8-STE -> fused R ->
frozen SegNet). Nothing here rebuilds the render/loss/acceptance substrate; the
j2 MLX module is subclassed only to expose the seg-logit FEATURE level that the
Gauss-Newton curvature is defined over.

THE SOLVE (NO-FAKE #6 statement — this is a solver, not a search):
  objective (seg leg, #383 terminal-pose law: pose_objective_weight == 0 during
  seg descent; pose collateral is priced by the UNCHANGED v19 realized
  acceptance outside this module):

      f(theta) = 100 * [ CE(logits(theta)) + w_m * hinge(logits(theta)) ]

  Gauss-Newton/generalized-GN curvature at theta0 (PSD by construction):

      H = 100 * J^T H_CE J / N_sites,   J = d logits / d theta (exact STE-smooth
      linearization through paint -> uint8-STE -> R -> SegNet),
      H_CE = blockdiag per site of (diag(p) - p p^T), p = softmax(logits(theta0)).

  The hinge is piecewise linear in logits (curvature 0 a.e.); it enters the
  right-hand side b = -grad f exactly (via the j2 ``loss_and_grad``), not H.
  The normal equations (H + mu * P) delta = b are solved by matrix-free CG.

SCORER-METRIC CUSTODY (ms3/ms4 bundle; honest consumption statement):
  The exact GGN through the frozen SegNet IS the scorer-metric pullback to
  description coordinates — the custodied rank-4 seg head row-Gram is literally
  contained in ``J^T H_CE J`` because the head is part of the differentiated
  forward. The ms4d BUNDLE-COMPLETE receipts are loaded through the fail-closed
  loader and recorded in every proposal's diagnostics (bundle id + component
  status); the per-parameter PRECONDITIONER is NOT read from the bundle (its
  atlas dimensions do not index the j5/v15 lift parameters) — it is MEASURED as
  a Hutchinson Jacobi diagonal of the same exact GGN operator. That degradation
  is explicit, measured, and carried in ``preconditioner_source``.

Rate term (ms1 contract): the description byte delta is a coder staircase that
is locally constant under fp32 theta perturbations below the re-emit
quantization step, so it has no smooth term inside the CG quadratic; it enters
the OBJECTIVE at realization — the v19 acceptance prices realized joint
Delta-S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/37_545_489 on the compiled
archive, with the 1.273108 B/error water level as the exchange-rate law the
acceptance implements. Nothing is admitted on model reduction alone.

Evidence axis: [macOS-MLX research-signal] for proposals; every ACCEPTED point
is realized through archive parse-back + uint8 + R + frozen CPU scorers by the
caller (v19). ``score_claim=false`` everywhere; pointer 0.1910828242 UNMOVED by
this module.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tac.optimization.ddm_metric_custody_bundle import (
    MetricCustodyBundle,
    load_metric_custody_bundle,
)
from tac.optimization.direct_description_joint_descent import (
    DirectDescriptionError,
    DirectDescriptionJointDescentMLXModule,
)

DEFAULT_METRIC_BUNDLE_PATH = (
    ".omx/research/ddm_ms4d_direct_metric_completion_20260724T155932Z/BUNDLE-COMPLETE.json"
)
WATER_LEVEL_BYTES_PER_ERROR = 1.273108  # registered exchange-rate law (endgame §3)


class FamilyDGNError(DirectDescriptionError):
    """family-d Gauss-Newton engine refusal."""


@dataclass(frozen=True, slots=True)
class GNProposalDiagnosticsV1:
    """Measured diagnostics for one Gauss-Newton proposal (schema ddm_fd1_gn_proposal.v1)."""

    schema: str
    block_pair_ids: tuple[int, ...]
    active_parameter_count: int
    hvp_calls: int
    hvp_mode: str
    cg_iterations: int
    cg_relative_residual: float
    damping: float
    preconditioner_source: str
    preconditioner_probes: int
    model_reduction: float
    gradient_norm: float
    step_norm: float
    rayleigh_curvature: float
    metric_custody: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "block_pair_ids": list(self.block_pair_ids),
            "active_parameter_count": self.active_parameter_count,
            "hvp_calls": self.hvp_calls,
            "hvp_mode": self.hvp_mode,
            "cg_iterations": self.cg_iterations,
            "cg_relative_residual": self.cg_relative_residual,
            "damping": self.damping,
            "preconditioner_source": self.preconditioner_source,
            "preconditioner_probes": self.preconditioner_probes,
            "model_reduction": self.model_reduction,
            "gradient_norm": self.gradient_norm,
            "step_norm": self.step_norm,
            "rayleigh_curvature": self.rayleigh_curvature,
            "metric_custody": dict(self.metric_custody),
            "evidence_axis": "[macOS-MLX research-signal]",
            "score_claim": False,
        }


class _SegFeatureModule(DirectDescriptionJointDescentMLXModule):
    """j2 MLX module extension exposing the seg-logit feature level.

    Inherits ``_render_camera`` (the exact paint + secant + uint8-STE map) and
    replicates ONLY the R -> SegNet feature slice of ``_components`` so the
    Gauss-Newton split map theta -> logits is available without touching the
    reviewed j2 hot path.
    """

    def _render_camera_smooth(
        self,
        theta: Any,
        base_camera: Any,
        template_masks: Any,
        realized_secant_basis: Any | None,
        realized_secant_indices: Sequence[int] | None,
    ) -> Any:
        """The smooth surrogate render: exact j2 paint + secants, clip WITHOUT the round-STE.

        This is precisely the map whose Jacobian the j2 STE gradient represents;
        the Gauss-Newton curvature is built on it so H = Js^T H_CE Js is PSD and
        consistent with the engine's first-order geometry.
        """
        mx = self.mx
        template_count = len(self.lift.template_rows)
        start = self.lift.template_parameter_start
        colour_delta = mx.reshape(theta[start : start + template_count * 3], (template_count, 3))
        paint_delta = mx.einsum("kbhw,kc->bhwc", template_masks, colour_delta)
        camera = base_camera + paint_delta[:, None, :, :, :]
        if realized_secant_basis is not None:
            selected = theta[mx.array(np.asarray(realized_secant_indices, dtype=np.int32))]
            camera = camera + mx.tensordot(selected, realized_secant_basis, axes=[[0], [0]])
        return mx.clip(camera, 0.0, 255.0)

    def seg_logits_nchw(
        self,
        theta: Any,
        *,
        base_camera: Any,
        template_masks: Any,
        realized_secant_basis: Any | None,
        realized_secant_indices: Sequence[int] | None,
        smooth: bool = False,
    ) -> Any:
        mx = self.mx
        from tac.local_acceleration.metal_fused_r_operator import fused_r_roundtrip

        if smooth:
            camera = self._render_camera_smooth(
                theta,
                base_camera,
                template_masks,
                realized_secant_basis,
                realized_secant_indices,
            )
        else:
            camera = self._render_camera(
                theta,
                base_camera,
                template_masks,
                realized_secant_basis,
                realized_secant_indices,
            )
        flat = mx.reshape(camera, (-1, 874, 1164, 3))
        scorer_rgb = fused_r_roundtrip(
            flat,
            camera_hw=(874, 1164),
            output_hw=(384, 512),
            ste_round=not smooth,
        )
        pairs = mx.reshape(scorer_rgb, (-1, 2, 384, 512, 3))
        seg_logits = self.scorer.segnet(pairs[:, 1])  # (B, 384, 512, 5)
        return mx.transpose(seg_logits, (0, 3, 1, 2))  # (B, 5, 384, 512)


class FamilyDGaussNewtonEngineV1:
    """Matrix-free Gauss-Newton/CG proposal engine on the j2 lifted description DOF."""

    def __init__(
        self,
        module: DirectDescriptionJointDescentMLXModule,
        *,
        metric_bundle_path: str | Path | None = DEFAULT_METRIC_BUNDLE_PATH,
        repository_root: str | Path = ".",
        hutchinson_probes: int = 8,
        seed: int = 0,
    ) -> None:
        if not isinstance(module, DirectDescriptionJointDescentMLXModule):
            raise FamilyDGNError("family-d engine requires the j2 MLX module substrate")
        # Re-bind the exact module state onto the feature-exposing extension.
        feature = _SegFeatureModule.__new__(_SegFeatureModule)
        feature.__dict__.update(module.__dict__)
        self.module = feature
        self.mx = module.mx
        self.hutchinson_probes = int(hutchinson_probes)
        if self.hutchinson_probes < 1:
            raise FamilyDGNError("family-d preconditioner requires >= 1 Hutchinson probe")
        self.rng = np.random.default_rng(int(seed))
        self.metric_bundle: MetricCustodyBundle | None = None
        self.metric_custody: dict[str, Any] = {
            "bundle_status": "NOT_LOADED",
            "preconditioner_from_bundle": False,
            "reason": (
                "bundle atlas dimensions do not index the lift parameters; "
                "operative preconditioner is the measured Hutchinson Jacobi "
                "diagonal of the exact GGN (documented degradation)"
            ),
        }
        if metric_bundle_path is not None:
            bundle = load_metric_custody_bundle(
                Path(metric_bundle_path),
                repository_root=repository_root,
                require_complete=True,
            )
            self.metric_bundle = bundle
            self.metric_custody.update(
                {
                    "bundle_status": str(bundle.status),
                    "bundle_id": bundle.bundle_id,
                    "components": {
                        str(component_id): str(receipt.status)
                        for component_id, receipt in bundle.components.items()
                    },
                }
            )

    # ------------------------------------------------------------------ maps
    def _closures(
        self,
        theta0: np.ndarray,
        *,
        pair_ids: Sequence[int],
        base_camera: np.ndarray,
        template_masks: np.ndarray,
        realized_secant_basis: np.ndarray | None,
        realized_secant_indices: Sequence[int] | None,
    ) -> tuple[Callable[[Any], Any], Any]:
        mx = self.mx
        base_mx = mx.array(np.asarray(base_camera, dtype=np.float32))
        masks_mx = mx.array(np.asarray(template_masks, dtype=np.float32))
        basis_mx = (
            None
            if realized_secant_basis is None
            else mx.array(np.asarray(realized_secant_basis, dtype=np.float32))
        )

        def logits_fn(theta_mx: Any) -> Any:
            return self.module.seg_logits_nchw(
                theta_mx,
                base_camera=base_mx,
                template_masks=masks_mx,
                realized_secant_basis=basis_mx,
                realized_secant_indices=realized_secant_indices,
                smooth=True,
            )

        theta_mx = mx.array(np.asarray(theta0, dtype=np.float32))
        return logits_fn, theta_mx

    def _hvp_factory(
        self,
        logits_fn: Callable[[Any], Any],
        theta_mx: Any,
        active_mask: np.ndarray,
        *,
        secant_epsilon: float = 0.5,
    ) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
        """Build v -> H v for H = 100 * Js^T H_CE Js / N (Gauss-Newton, PSD).

        Js v is a MEASURED central secant of the smooth surrogate at the
        parameter-quantum scale (``secant_epsilon`` in the same units as j2's
        realized +/-1-quantum secants; ``mx.jvp`` is unavailable through the
        fused-R CustomKernel, and the quantum-scale secant is the linearization
        the engine's own geometry already uses). Js^T (.) is the exact
        reverse-mode ``mx.vjp`` of the same smooth map, so H is symmetric PSD
        up to secant error. Softmax probabilities are frozen at theta0. Every
        array is evaluated eagerly so the lazy graph never accumulates across
        CG iterations (the #205 lesson).
        """
        mx = self.mx
        logits0 = logits_fn(theta_mx)
        mx.eval(logits0)
        probs0 = mx.softmax(logits0, axis=1)
        mx.eval(probs0)
        shape = logits0.shape  # (B, 5, H, W)
        n_sites = float(shape[0] * shape[2] * shape[3])
        mask_mx = mx.array(active_mask.astype(np.float32))
        counters = {"hvp_calls": 0, "mode": f"fd_secant(eps={secant_epsilon})+vjp"}

        def hvp(v: np.ndarray) -> np.ndarray:
            counters["hvp_calls"] += 1
            v_arr = np.asarray(v, dtype=np.float64) * active_mask
            norm = float(np.linalg.norm(v_arr))
            if norm <= 0.0 or not math.isfinite(norm):
                return np.zeros_like(v_arr)
            unit = (v_arr / norm).astype(np.float32)
            unit_mx = mx.array(unit) * mask_mx
            plus = logits_fn(theta_mx + secant_epsilon * unit_mx)
            minus = logits_fn(theta_mx - secant_epsilon * unit_mx)
            u = (plus - minus) * (norm / (2.0 * secant_epsilon))
            # Per-site CE Hessian action: H_CE u = p*u - p * sum_c(p_c u_c).
            s = mx.sum(probs0 * u, axis=1, keepdims=True)
            w = probs0 * (u - s)
            _, vjp_out = mx.vjp(logits_fn, (theta_mx,), (w,))
            out = vjp_out[0] * mask_mx * (100.0 / n_sites)
            mx.eval(out)
            return np.asarray(out, dtype=np.float64)

        return hvp, counters

    # -------------------------------------------------------- preconditioner
    def _measured_jacobi_diagonal(
        self,
        hvp: Callable[[np.ndarray], np.ndarray],
        active_mask: np.ndarray,
    ) -> np.ndarray:
        """Hutchinson estimate of diag(H) over the active set (measured, not asserted)."""
        dim = active_mask.size
        acc = np.zeros(dim, dtype=np.float64)
        for _ in range(self.hutchinson_probes):
            z = self.rng.choice((-1.0, 1.0), size=dim).astype(np.float64) * active_mask
            acc += z * hvp(z)
        diag = acc / float(self.hutchinson_probes)
        floor = max(float(np.max(np.abs(diag))), 1.0e-30) * 1.0e-6
        return np.where(active_mask > 0.0, np.maximum(diag, floor), 1.0)

    # -------------------------------------------------------------------- cg
    @staticmethod
    def _cg(
        hvp: Callable[[np.ndarray], np.ndarray],
        b: np.ndarray,
        *,
        diag_precond: np.ndarray,
        damping: float,
        max_iterations: int,
        tolerance: float,
        active_mask: np.ndarray,
    ) -> tuple[np.ndarray, int, float]:
        """Preconditioned CG on (H + mu * diag(P)) delta = b, restricted to the active set."""

        def operator(v: np.ndarray) -> np.ndarray:
            return hvp(v) + damping * diag_precond * v * active_mask

        x = np.zeros_like(b)
        r = b.copy()
        b_norm = float(np.linalg.norm(b))
        if b_norm <= 0.0 or not math.isfinite(b_norm):
            raise FamilyDGNError("family-d CG right-hand side is zero or non-finite")
        z = r / diag_precond
        p = z.copy()
        rz = float(r @ z)
        iterations = 0
        while iterations < int(max_iterations):
            iterations += 1
            hp = operator(p)
            php = float(p @ hp)
            if not math.isfinite(php) or php <= 0.0:
                # Curvature exhausted along p (numerically flat/indefinite from
                # STE kinks): return the best PSD-model iterate so far.
                break
            alpha = rz / php
            x = x + alpha * p
            r = r - alpha * hp
            if float(np.linalg.norm(r)) / b_norm <= tolerance:
                break
            z = r / diag_precond
            rz_next = float(r @ z)
            p = z + (rz_next / rz) * p
            rz = rz_next
        return x, iterations, float(np.linalg.norm(r)) / b_norm

    # --------------------------------------------------------------- propose
    def propose(
        self,
        theta0: np.ndarray,
        gradient: np.ndarray,
        *,
        pair_ids: Sequence[int],
        base_camera: np.ndarray,
        template_masks: np.ndarray,
        realized_secant_basis: np.ndarray | None,
        realized_secant_indices: Sequence[int] | None,
        active_indices: Sequence[int],
        damping: float,
        cg_iterations: int = 12,
        cg_tolerance: float = 0.1,
    ) -> tuple[np.ndarray, GNProposalDiagnosticsV1]:
        """Solve the damped GN normal equations; return (delta, diagnostics).

        ``gradient`` MUST be the exact j2 ``loss_and_grad`` gradient at
        ``theta0`` on the same pair block (it carries CE + hinge exactly);
        ``delta`` is the descent step (theta0 + delta), zero outside the
        active set. The caller owns realization + v19 acceptance.
        """
        theta0 = np.asarray(theta0, dtype=np.float32)
        gradient = np.asarray(gradient, dtype=np.float64)
        if theta0.shape != gradient.shape:
            raise FamilyDGNError("family-d theta/gradient geometry differs")
        if not math.isfinite(float(damping)) or damping < 0.0:
            raise FamilyDGNError("family-d damping is invalid")
        active_mask = np.zeros(theta0.size, dtype=np.float64)
        for index in active_indices:
            if index < 0 or index >= theta0.size:
                raise FamilyDGNError("family-d active index outside theta")
            active_mask[int(index)] = 1.0
        if not np.any(active_mask):
            raise FamilyDGNError("family-d active set is empty")

        logits_fn, theta_mx = self._closures(
            theta0,
            pair_ids=pair_ids,
            base_camera=base_camera,
            template_masks=template_masks,
            realized_secant_basis=realized_secant_basis,
            realized_secant_indices=realized_secant_indices,
        )
        hvp, counters = self._hvp_factory(logits_fn, theta_mx, active_mask)
        b = -gradient * active_mask
        diag = self._measured_jacobi_diagonal(hvp, active_mask)
        delta, iterations, residual = self._cg(
            hvp,
            b,
            diag_precond=diag,
            damping=float(damping),
            max_iterations=int(cg_iterations),
            tolerance=float(cg_tolerance),
            active_mask=active_mask,
        )
        h_delta = hvp(delta)
        model_reduction = float(b @ delta - 0.5 * (delta @ h_delta))
        step_norm = float(np.linalg.norm(delta))
        rayleigh = float((delta @ h_delta) / (step_norm**2)) if step_norm > 0.0 else 0.0
        diagnostics = GNProposalDiagnosticsV1(
            schema="ddm_fd1_gn_proposal.v1",
            block_pair_ids=tuple(int(v) for v in pair_ids),
            active_parameter_count=int(active_mask.sum()),
            hvp_calls=int(counters["hvp_calls"]),
            hvp_mode=str(counters["mode"]),
            cg_iterations=iterations,
            cg_relative_residual=residual,
            damping=float(damping),
            preconditioner_source="measured_hutchinson_jacobi_diagonal_of_exact_ggn",
            preconditioner_probes=self.hutchinson_probes,
            model_reduction=model_reduction,
            gradient_norm=float(np.linalg.norm(b)),
            step_norm=step_norm,
            rayleigh_curvature=rayleigh,
            metric_custody=dict(self.metric_custody),
        )
        return delta.astype(np.float32), diagnostics


__all__ = [
    "DEFAULT_METRIC_BUNDLE_PATH",
    "WATER_LEVEL_BYTES_PER_ERROR",
    "FamilyDGNError",
    "FamilyDGaussNewtonEngineV1",
    "GNProposalDiagnosticsV1",
]
