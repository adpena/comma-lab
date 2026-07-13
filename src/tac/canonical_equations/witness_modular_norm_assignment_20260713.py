# SPDX-License-Identifier: MIT
"""Module-to-norm law for the current V9 level-set witness.

This is a source-inspected design law, not an optimizer implementation and not
an empirical score anchor.  It records which manifold and local module norm
belongs to each trainable block before a future exact Manifold-Muon build can
be admitted.  Scalar inter-module sensitivities remain uncalibrated.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
)
from tac.canonical_equations.steps_dimension_epochs_to_target_20260713 import (
    EpochsToTargetTicket,
)
from tac.provenance.builders import build_provenance_for_predicted

EQUATION_ID = "witness_modular_norm_assignment_v1"
_UTC = "2026-07-13T00:00:00Z"
_CHECKPOINT = (
    "experiments/results/v9_cgauge_432_coherent_arm_20260711/"
    "levelset_ckpt_stageOctave1_ep251.npz"
)
_MEMO = ".omx/research/muonh_manifold_muon_dig_20260713.md"


@dataclass(frozen=True)
class ModuleNormAssignment:
    """One block of the source-inspected V9 witness product geometry."""

    module_pattern: str
    shapes: tuple[tuple[int, ...], ...]
    trainable_parameters: int
    forward_role: str
    manifold: str
    module_norm: str
    steepest_direction: str
    current_optimizer: str
    candidate_delta: str
    evidence_status: str
    verdict_scope: str

    def __post_init__(self) -> None:
        if not self.module_pattern.strip():
            raise ValueError("module_pattern must be non-empty")
        if not self.shapes or any(not shape or any(int(d) <= 0 for d in shape) for shape in self.shapes):
            raise ValueError("shapes must contain positive dimensions")
        if int(self.trainable_parameters) != sum(math.prod(shape) for shape in self.shapes):
            raise ValueError("trainable_parameters must equal the sum of shape products")
        for name in (
            "forward_role",
            "manifold",
            "module_norm",
            "steepest_direction",
            "current_optimizer",
            "candidate_delta",
            "evidence_status",
            "verdict_scope",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")


# Source inventory: V9 ep251 checkpoint SHA-256
# c59cdec6eec16677c0a2eb5667979dd1c8f883bcd1cf5532302d67acd633c758.
# The six rows currently routed to Muon total 59,136 of 87,575 trainable
# parameters.  `pose_carrier.xi_stored` is frozen and therefore absent.
MODULE_NORM_ASSIGNMENTS: tuple[ModuleNormAssignment, ...] = (
    ModuleNormAssignment(
        module_pattern="in_proj.weight",
        shapes=((96, 80),),
        trainable_parameters=7_680,
        forward_role="coordinate-feature RMS vector -> hidden RMS vector",
        manifold="unconstrained R^(96x80)",
        module_norm="RMS-to-RMS induced norm = sqrt(80/96) * spectral_norm",
        steepest_direction="ordinary Muon matrix-sign dualizer",
        current_optimizer="Muon during the finisher",
        candidate_delta="none",
        evidence_status="DERIVED from linear-module source and modular induced-norm law",
        verdict_scope="current non-decoupled V9 coordinate trunk",
    ),
    ModuleNormAssignment(
        module_pattern="hidden.{0,1,2,3}.weight",
        shapes=((96, 96),) * 4,
        trainable_parameters=36_864,
        forward_role="hidden RMS vector -> hidden RMS vector",
        manifold="product of four unconstrained R^(96x96) matrices",
        module_norm="RMS-to-RMS induced norm = spectral_norm",
        steepest_direction="ordinary Muon matrix-sign dualizer per layer",
        current_optimizer="Muon during the finisher",
        candidate_delta="none",
        evidence_status="DERIVED from linear-module source and modular induced-norm law",
        verdict_scope="current four-layer non-decoupled V9 trunk",
    ),
    ModuleNormAssignment(
        module_pattern="film.weight",
        shapes=((768, 19),),
        trainable_parameters=14_592,
        forward_role="19-D conditioning coordinate -> all four layers' FiLM scales and shifts",
        manifold="polar finishing chart W=Q H0 with Q in St(768,19) and frozen H0=(W0.T W0)^(1/2)",
        module_norm="spectral norm on Q's Stiefel tangent space, with H0 defining the conditioning-coordinate metric",
        steepest_direction="exact tangent-constrained Manifold-Muon LMO on Q plus retraction; deploy Q H0",
        current_optimizer="ordinary ambient Muon; optional ambient-step-plus-polar-projection is OFF",
        candidate_delta="replace only this ambient Muon block with exact polar-chart Manifold Muon",
        evidence_status="DERIVED from the repo's conditioning-isometry invariant and Bernstein's manifold law",
        verdict_scope="current full-column-rank V9 FiLM map; H0 is frozen in the first bounded formulation",
    ),
    ModuleNormAssignment(
        module_pattern="code",
        shapes=((1200, 19),),
        trainable_parameters=22_800,
        forward_role="per-state semantic conditioning coordinates",
        manifold="product (R^19)^1200; no source-derived fixed-radius constraint",
        module_norm="local evaluator-pullback/product Euclidean norm, not a matrix spectral norm",
        steepest_direction="adaptive vector/embedding optimizer pending pullback calibration",
        current_optimizer="AdamW fallback",
        candidate_delta="none; a hypersphere would delete meaningful radial FiLM amplitude",
        evidence_status="DERIVED from embedding lookup and unconstrained code semantics",
        verdict_scope="current raw V9 codes; not a negative on a future gain-decoupled code chart",
    ),
    ModuleNormAssignment(
        module_pattern="pose_carrier.dxi",
        shapes=((600, 6),),
        trainable_parameters=3_600,
        forward_role="per-pair se(3) pose residual coordinate",
        manifold="product se(3)^600 with task-pullback metric",
        module_norm="local PoseNet/R pullback norm, not a matrix spectral norm",
        steepest_direction="Lie-coordinate adaptive optimizer or task-natural step",
        current_optimizer="AdamW fallback",
        candidate_delta="none",
        evidence_status="DERIVED from pose-carrier source semantics",
        verdict_scope="table-mode trainable dxi only; xi_stored is frozen",
    ),
    ModuleNormAssignment(
        module_pattern="in_proj.bias + hidden.{0,1,2,3}.bias",
        shapes=((96,),) * 5,
        trainable_parameters=480,
        forward_role="hidden translations",
        manifold="product Euclidean vectors",
        module_norm="hidden RMS vector norm",
        steepest_direction="normalized vector gradient in the exact modular theory",
        current_optimizer="AdamW fallback",
        candidate_delta="none in this ticket",
        evidence_status="DERIVED from additive hidden-vector role",
        verdict_scope="bias vectors; tuned AdamW remains the incumbent implementation",
    ),
    ModuleNormAssignment(
        module_pattern="film.bias",
        shapes=((768,),),
        trainable_parameters=768,
        forward_role="FiLM scale/shift translation",
        manifold="Euclidean vector",
        module_norm="FiLM-output RMS vector norm",
        steepest_direction="normalized vector gradient in the exact modular theory",
        current_optimizer="AdamW fallback",
        candidate_delta="none in this ticket",
        evidence_status="DERIVED from additive FiLM-output role",
        verdict_scope="bias vector; no Stiefel constraint applies",
    ),
    ModuleNormAssignment(
        module_pattern="out_sdf.weight",
        shapes=((5, 96),),
        trainable_parameters=480,
        forward_role="hidden features -> five class/SDF logits",
        manifold="unconstrained R^(5x96)",
        module_norm="RMS-to-linf induced norm = sqrt(96) * max_row_l2, locally refined by SegNet margin pullback",
        steepest_direction="class-row dualizer or accepted terminal head solve, not generic spectral Muon",
        current_optimizer="AdamW fallback",
        candidate_delta="none; compose with a separately admitted TerminalSolve",
        evidence_status="DERIVED from argmax-margin authority and final-head source",
        verdict_scope="five-class V9 head under local evaluator-cell geometry",
    ),
    ModuleNormAssignment(
        module_pattern="out_sdf.bias",
        shapes=((5,),),
        trainable_parameters=5,
        forward_role="five class/SDF logit translations",
        manifold="Euclidean vector",
        module_norm="linf/logit-margin task norm",
        steepest_direction="class-coordinate dualizer or accepted terminal head solve",
        current_optimizer="AdamW fallback",
        candidate_delta="none",
        evidence_status="DERIVED from argmax-margin authority",
        verdict_scope="five-class V9 head",
    ),
    ModuleNormAssignment(
        module_pattern="out_tex.weight",
        shapes=((3, 96),),
        trainable_parameters=288,
        forward_role="hidden features -> pose-carrying RGB texture logits",
        manifold="unconstrained R^(3x96)",
        module_norm="local sigmoid/R/PoseNet pullback norm; RGB-RMS spectral norm is only a proxy",
        steepest_direction="task-pullback/trust-region direction",
        current_optimizer="AdamW fallback",
        candidate_delta="none",
        evidence_status="DERIVED from evaluator-equivalent witness authority",
        verdict_scope="linear V9 texture head; no claim for optional widened head",
    ),
    ModuleNormAssignment(
        module_pattern="out_tex.bias",
        shapes=((3,),),
        trainable_parameters=3,
        forward_role="pose-carrying RGB texture translation",
        manifold="Euclidean vector",
        module_norm="local sigmoid/R/PoseNet pullback norm",
        steepest_direction="task-pullback/trust-region direction",
        current_optimizer="AdamW fallback",
        candidate_delta="none",
        evidence_status="DERIVED from evaluator-equivalent witness authority",
        verdict_scope="linear V9 texture head",
    ),
    ModuleNormAssignment(
        module_pattern="palette",
        shapes=((5, 3),),
        trainable_parameters=15,
        forward_role="class-probability l1 vector -> RGB-logit linf vector before sigmoid",
        manifold="unconstrained R^(5x3)",
        module_norm="l1-to-linf induced norm = max_abs_entry, locally refined by scorer pullback",
        steepest_direction="entrywise/task-pullback dualizer",
        current_optimizer="AdamW fallback",
        candidate_delta="none",
        evidence_status="DERIVED from softmax-times-palette forward source",
        verdict_scope="five-class, three-channel V9 palette logits",
    ),
)


def weighted_modular_product_norm(
    module_norm_values: Mapping[str, float],
    sensitivity_multipliers: Mapping[str, float],
) -> float:
    """Return ``max_q s_q N_q`` with fail-closed block and scalar custody.

    ``s_q`` is explicitly the multiplier convention used by this equation.  It
    is not asserted to be calibrated for V9; a future probe must supply every
    positive multiplier before this helper is used to set learning rates.
    """

    if not module_norm_values:
        raise ValueError("module_norm_values must be non-empty")
    if set(module_norm_values) != set(sensitivity_multipliers):
        raise ValueError("norm values and sensitivity multipliers must have identical block keys")
    products: list[float] = []
    for key, raw_norm in module_norm_values.items():
        norm = float(raw_norm)
        sensitivity = float(sensitivity_multipliers[key])
        if not math.isfinite(norm) or norm < 0.0:
            raise ValueError(f"module norm for {key!r} must be finite and non-negative")
        if not math.isfinite(sensitivity) or sensitivity <= 0.0:
            raise ValueError(f"sensitivity multiplier for {key!r} must be finite and positive")
        products.append(sensitivity * norm)
    return max(products)


def rms_to_rms_operator_norm(spectral_norm: float, fan_in: int, fan_out: int) -> float:
    """Convert a Euclidean spectral norm to the RMS-to-RMS operator norm."""

    sigma = float(spectral_norm)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("spectral_norm must be finite and non-negative")
    if isinstance(fan_in, bool) or isinstance(fan_out, bool) or int(fan_in) <= 0 or int(fan_out) <= 0:
        raise ValueError("fan_in and fan_out must be positive integers")
    return math.sqrt(int(fan_in) / int(fan_out)) * sigma


def rms_to_linf_operator_norm(max_row_l2: float, fan_in: int) -> float:
    """Return the exact RMS-input to linf-output norm of a row-stacked map."""

    row_norm = float(max_row_l2)
    if not math.isfinite(row_norm) or row_norm < 0.0:
        raise ValueError("max_row_l2 must be finite and non-negative")
    if isinstance(fan_in, bool) or int(fan_in) <= 0:
        raise ValueError("fan_in must be a positive integer")
    return math.sqrt(int(fan_in)) * row_norm


def current_trainable_parameter_count() -> int:
    """Return the complete source-inspected V9 trainable count."""

    return sum(row.trainable_parameters for row in MODULE_NORM_ASSIGNMENTS)


def current_muon_parameter_count() -> int:
    """Return the V9 parameters currently routed to ordinary Muon."""

    return sum(
        row.trainable_parameters
        for row in MODULE_NORM_ASSIGNMENTS
        if row.module_pattern in {
            "in_proj.weight",
            "hidden.{0,1,2,3}.weight",
            "film.weight",
        }
    )


MANIFOLD_MUON_AB_TICKET = EpochsToTargetTicket(
    lever_id="film_polar_chart_exact_manifold_muon_finisher",
    existing_dsl_surface=(
        "WIRING NEEDED: --film-stiefel is ambient optimizer plus polar projection, "
        "not exact tangent-constrained Manifold Muon"
    ),
    status="WIRING_NEEDED",
    start_custody=(
        "both arms originate from the same cold seed-0 n600 vehicle and fork the exact complete pre-Muon boundary "
        "checkpoint; treatment factorizes W0=Q0 H0 without changing W0, freezes H0, and must prove Q0 H0 reconstructs "
        "the common fp32 deploy tensor before any update; arbitrary warm starts and state omissions are inadmissible"
    ),
    target_rule=(
        "from the common exact n600 realized-through-R boundary d_seg_start, first emitted d_seg <= "
        "0.98*d_seg_start; factor 0.98 is ASSUMED policy and the numeric threshold is DERIVED after the common replay"
    ),
    maximum_nominal_epochs=250,
    censoring_rule=(
        "right-censor each arm after 250 nominal finisher epochs; no interpolation; missing crossing remains None; "
        "record exact accepted optimizer updates and internal tangent-dual iterations"
    ),
    control_definition=(
        "TUNED incumbent compiler output: ordinary Muon on in_proj.weight, film.weight, hidden.*.weight; "
        "warm-start momentum; lr=0.002; momentum=0.95; five Newton-Schulz steps; cosine final fraction 0.1; "
        "same AdamW fallback and same event-defined start"
    ),
    treatment_definition=(
        "identical tuned control except film.weight is represented exactly as Q H0, H0 frozen at the common boundary, "
        "and Q uses the exact Stiefel tangent spectral LMO with retraction; in_proj/hidden remain ordinary Muon and "
        "every non-Muon block remains the identical AdamW fallback; deploy folds Q H0 back to film.weight"
    ),
    verdict_scope=(
        "epochs/update/wall-to-fixed-d_seg comparison of the exact film-only Manifold-Muon formulation against the "
        "tuned V9 Muon incumbent; no verdict on all-block constraints, generic Stiefel projection, or score promotion"
    ),
    reformulation_reactivation_queue=(
        "build a typed default-OFF polar-chart arm with tangent-dual, transported momentum, H0, and atomic resume state",
        "prove deterministic polar-factor/reconstruction parity at the common boundary before any optimizer update",
        "prove deterministic NumPy reference and MLX parity for tangent residual, retraction, QH0 EMA deploy, and resume",
        "only then run the governed matched n600 ticket",
    ),
    source_artifacts=(
        _CHECKPOINT,
        "experiments/results/v9_cgauge_432_coherent_arm_20260711/launch.sh",
        "experiments/train_levelset_witness_realized_through_R_mlx.py",
        "src/tac/optimization/muon_finisher_mlx.py",
        "src/tac/optimization/md_decoupling.py",
        "src/tac/canonical_equations/steps_dimension_epochs_to_target_20260713.py",
        ".omx/research/steps_dimension_95kill_20260713_SPEC.md",
        _MEMO,
    ),
    speed_configuration_rule=(
        "both arms require byte-identical compiler-emitted neutral speed settings, all requested speed levers ON, "
        "and direct elapsed-to-crossing including treatment initialization and tangent-dual overhead"
    ),
    measurement_authority_rule=(
        "crossings require deterministic NumPy-fp32 realization through actual R plus the frozen CPU-torch scorer "
        "on all 600 states; MLX optimizer execution is advisory and cannot promote a contest score"
    ),
    authority_axis="[n600 A/B ticket only; UNMEASURED; NON-PROMOTABLE]",
)


_LAW = (
    "M=product_q M_q; ||dtheta||_mod=max_q s_q N_q(dtheta_q); "
    "film: W0=Q0 H0, Q in St(768,19), A.T Q + Q.T A=0, ||A||_2<=1; "
    "A*=argmin_A <grad_Q L,A>; deploy W=Q H0"
)


def build_witness_modular_norm_assignment_v1() -> CanonicalEquation:
    """Build the unanchored source/literature-derived assignment law."""

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="V9 witness modular product norm and film Stiefel tangent law",
        one_line_summary=(
            "Use spectral Muon on trunk matrices, exact tangent-spectral Manifold Muon only on film.weight, "
            "and evaluator-semantic norms on codes, pose coordinates, heads, biases, and palette."
        ),
        latex_form=(
            r"\mathcal M=\prod_q\mathcal M_q,\quad\|\Delta\theta\|_{mod}="
            r"\max_q s_q N_q(\Delta\theta_q);\quad W_{f,0}=Q_0H_0,\ Q\in\mathrm{St}(768,19),\ "
            r"A^\top Q+Q^\top A=0,\ \|A\|_2\le1,\ "
            r"A^*=\arg\min_A\langle\nabla_QL,A\rangle,\ W_f=QH_0"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.witness_modular_norm_assignment_20260713:weighted_modular_product_norm"
        ),
        domain_of_validity={
            "architecture": "source-inspected V9 ep251 non-decoupled level-set witness",
            "checkpoint_sha256": "c59cdec6eec16677c0a2eb5667979dd1c8f883bcd1cf5532302d67acd633c758",
            "theory_status": "INFERRED_FROM_PRIMARY_LITERATURE",
            "architecture_status": "VERIFIED_VIA_SOURCE_INSPECTION",
            "empirical_optimizer_status": "UNMEASURED; exact Manifold Muon is not built",
            "uncalibrated_quantity": "every inter-module sensitivity multiplier s_q",
            "static_checkpoint_probe": {
                "axis": "[macOS-CPU/numpy-fp64 static checkpoint probe; no training; non-promotable]",
                "film_sigma_min": 7.283143447157408,
                "film_sigma_max": 10.076844066238596,
                "unit_polar_projection_relative_frobenius_delta": 0.882328659125916,
                "best_scalar_stiefel_relative_frobenius_delta": 0.0888146994166776,
                "consequence": "direct unit-Stiefel projection is not an admissible common-boundary initialization",
                "source": ".omx/research/muonh_manifold_muon_static_probe_20260713.json",
            },
            "excluded": (
                "optional film-per-layer, concat-code, widened-texture, decoupled-field, v8 per-class, "
                "and future gain-normalized architectures require their own assignment"
            ),
            "verdict_scope": (
                "norm-family assignment only; no steps, wall, d_seg, d_pose, byte, or contest-score claim"
            ),
        },
        units_in={
            "N_q": "native block-norm units",
            "s_q": "positive multiplier converting N_q to the common modular budget",
        },
        units_out={"modular_product_norm": "common dimensionless trust-region budget"},
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.muon_finisher_mlx",
            "tac.optimization.md_decoupling",
            _MEMO,
            ".omx/research/muonh_manifold_muon_DAG_FEED_20260713.md",
        ),
        canonical_producers=(
            "experiments.train_levelset_witness_realized_through_R_mlx.LevelSetRGBWitness",
            _CHECKPOINT,
            "https://thinkingmachines.ai/blog/modular-manifolds/",
            "https://docs.modula.systems/algorithms/manifold/stiefel/",
        ),
        provenance=build_provenance_for_predicted(
            model_id=EQUATION_ID,
            inputs_sha256=hashlib.sha256(_LAW.encode("utf-8")).hexdigest(),
            measurement_axis="[source-inspected plus primary-literature-derived; no empirical anchor]",
            hardware_substrate="numpy-portable design law",
            captured_at_utc=_UTC,
        ),
    )


__all__ = [
    "EQUATION_ID",
    "MANIFOLD_MUON_AB_TICKET",
    "MODULE_NORM_ASSIGNMENTS",
    "ModuleNormAssignment",
    "build_witness_modular_norm_assignment_v1",
    "current_muon_parameter_count",
    "current_trainable_parameter_count",
    "rms_to_linf_operator_norm",
    "rms_to_rms_operator_norm",
    "weighted_modular_product_norm",
]
