# SPDX-License-Identifier: MIT
"""Typed optimal-basis surface for the level-set witness (task #497).

The historical front end called ``curvelet`` is kept byte-compatible, but it is
classified here by what the code actually computes: global polar directional
Fourier plane waves.  Increasing the orientation count with scale gives a
curvelet-inspired *frequency grid*; it does not supply the translations,
spatial windows, or scale-dependent anisotropic supports of a curvelet or
shearlet frame.

This module is deliberately standalone.  It owns no task-#500 metric schedule,
does not edit the hot curriculum module, and emits only flags that the real
trainer already parses.  A genuinely different frame fails closed until both
the train path and the generated ``inflate.py`` path implement the same ops and
an equal-budget real-n600 through-R receipt exists.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from tac.boundary_math.lever_b_levelset_generator import (
    CurveletBankConfig,
    build_coords,
    curvelet_directional_B,
    curvelet_feats,
)
from tac.witness_dsl.basis_control import (
    LEGACY_FOURIER_AB_CONTROL as LEGACY_FOURIER_AB_CONTROL_ID,
)
from tac.witness_dsl.curriculum_dsl import (
    Lever,
    build_real_trainer_parser,
    real_trainer_flags,
)


class BasisFamily(StrEnum):
    """Stable family IDs; names describe implemented mathematics, not ancestry."""

    LEGACY_FOURIER_AB_CONTROL = LEGACY_FOURIER_AB_CONTROL_ID
    POLAR_DIRECTIONAL_FOURIER = "polar_directional_fourier"
    SELF_ORIENTED_FOURIER = "self_oriented_fourier"
    HYBRID_FOURIER_INTERIOR_CURVELET_BOUNDARY = "hybrid_fourier_interior_curvelet_boundary"
    WINDOWED_CURVELET = "windowed_curvelet"
    COMPACT_SHEARLET = "compact_shearlet"
    STEERABLE_GABOR = "steerable_gabor"
    WAVELET = "wavelet"
    SIREN_FINER = "siren_finer"
    HASH_GRID = "hash_grid"
    BSPLINE_RBF = "bspline_rbf"
    ZERNIKE_SPHERICAL = "zernike_spherical"
    LAPLACIAN_EIGEN = "laplacian_eigen"
    NTK_OPTIMAL = "ntk_optimal"


class BasisEvidence(StrEnum):
    MEASURED_THROUGH_R_N600_FORMULATION = "MEASURED_THROUGH_R_N600_FORMULATION"
    SOURCE_DERIVED = "DERIVED_FROM_SOURCE"
    PREDICTED_UNMEASURED = "PREDICTED_UNMEASURED"


@dataclass(frozen=True)
class BasisCandidate:
    family: BasisFamily
    next_measurement_rank: int
    fit_to_curved_codim1_anisotropy: str
    mathematical_atom: str
    byte_cost: str
    mlx_portability: str
    evidence: BasisEvidence
    equal_budget_dseg: float | None
    verdict_scope: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["family"] = self.family.value
        row["evidence"] = self.evidence.value
        return row


_NO_DIFFERENT_FRAME_RECEIPT = (
    "NO real-n600 through-R equal-budget receipt; prediction only"
)


# No unmeasured percentage is guessed.  Numerical d_seg rows are restricted to
# the settled owed-16 warm-start cell at ep675.
_CATALOG = (
    BasisCandidate(
        BasisFamily.LEGACY_FOURIER_AB_CONTROL,
        0,
        "legacy control only: byte-identical historical global plane-wave computation for owed A/B",
        "global polar directional Fourier plane waves, selected only as the control arm",
        "bank regenerated free; learned/video-derived weights counted",
        "implemented in NumPy, MLX, and generated inflate.py",
        BasisEvidence.MEASURED_THROUGH_R_N600_FORMULATION,
        0.004244,
        "control label for bounded warm-start ep675; no curvelet default or ship claim",
    ),
    BasisCandidate(
        BasisFamily.HYBRID_FOURIER_INTERIOR_CURVELET_BOUNDARY,
        1,
        "highest predicted: retain the cheap smooth-interior chart and localize only the boundary annulus",
        "partition-of-unity mixture of global low-frequency atoms and boundary-windowed parabolic atoms",
        "generic windows/scales are rule-118 free; learned/video-derived coefficients counted",
        "portable primitives, but train/inflate implementation absent",
        BasisEvidence.PREDICTED_UNMEASURED,
        None,
        _NO_DIFFERENT_FRAME_RECEIPT,
    ),
    BasisCandidate(
        BasisFamily.WINDOWED_CURVELET,
        2,
        "optimal N-term approximation class for C2 curved edges under parabolic scaling",
        "translated, rotated, spatially windowed atoms with width approximately length squared",
        "generic frame free; selected/learned coefficients counted",
        "NumPy feasible; MLX scatter/window kernel owed",
        BasisEvidence.PREDICTED_UNMEASURED,
        None,
        _NO_DIFFERENT_FRAME_RECEIPT,
    ),
    BasisCandidate(
        BasisFamily.COMPACT_SHEARLET,
        3,
        "same curved-edge approximation class with GPU-friendly shear/dilation indexing",
        "compactly supported anisotropic shears across scale and translation",
        "generic frame free; selected/learned coefficients counted",
        "high after separable shear implementation; currently absent",
        BasisEvidence.PREDICTED_UNMEASURED,
        None,
        _NO_DIFFERENT_FRAME_RECEIPT,
    ),
    BasisCandidate(
        BasisFamily.STEERABLE_GABOR,
        4,
        "strong local tangent selectivity, but fixed aspect ratios are weaker than parabolic multiscale atoms",
        "localized sinusoid times spatial envelope, steered over orientation",
        "generic kernels free; learned coefficients counted",
        "high; dense elementwise MLX ops",
        BasisEvidence.PREDICTED_UNMEASURED,
        None,
        _NO_DIFFERENT_FRAME_RECEIPT,
    ),
    BasisCandidate(
        BasisFamily.WAVELET,
        5,
        "excellent for smooth interiors and point singularities; axis-aligned atoms pay extra for curved contours",
        "translated isotropic/separable multiresolution wavelets",
        "generic transform free; retained coefficients counted",
        "high; separable filterbank",
        BasisEvidence.PREDICTED_UNMEASURED,
        None,
        _NO_DIFFERENT_FRAME_RECEIPT,
    ),
    BasisCandidate(
        BasisFamily.BSPLINE_RBF,
        6,
        "local support prevents global ringing; isotropic knots need adaptive placement along the separatrix",
        "compact B-splines or radial kernels on a deterministic knot hierarchy",
        "generic knots free; video-derived knot amplitudes/locations counted",
        "high for fixed knots; sparse gather parity owed",
        BasisEvidence.PREDICTED_UNMEASURED,
        None,
        _NO_DIFFERENT_FRAME_RECEIPT,
    ),
    BasisCandidate(
        BasisFamily.SIREN_FINER,
        7,
        "learned frequencies can adapt, but are global and need initialization to avoid common-phase saturation",
        "periodic learned first-layer frequencies with FINER/FINER++ bias initialization",
        "initialization code free; learned weights counted",
        "already compiler-portable in trainer/inflate; n600 family verdict owed",
        BasisEvidence.SOURCE_DERIVED,
        None,
        "compiler exists; fixed-beta and bounded-warm-start negatives do not settle fresh-start FINER",
    ),
    BasisCandidate(
        BasisFamily.HASH_GRID,
        8,
        "multiresolution locality is strong, but grid collisions and counted tables are rate liabilities",
        "hashed multiresolution learned feature tables with interpolation",
        "generic hash code free; all learned table entries counted",
        "MLX implementation feasible; deterministic collision/parity contract absent",
        BasisEvidence.PREDICTED_UNMEASURED,
        None,
        _NO_DIFFERENT_FRAME_RECEIPT,
    ),
    BasisCandidate(
        BasisFamily.LAPLACIAN_EIGEN,
        9,
        "metric/manifold adapted globally, but curved-edge sparsity depends on the operator and boundary conditions",
        "eigenfunctions of a registered renderer/Fisher pullback Laplacian",
        "generic eigensolver free; video-derived eigenvectors/coefficients counted unless analytically regenerated",
        "dense MLX matmul portable; eigensystem custody absent",
        BasisEvidence.PREDICTED_UNMEASURED,
        None,
        _NO_DIFFERENT_FRAME_RECEIPT,
    ),
    BasisCandidate(
        BasisFamily.NTK_OPTIMAL,
        10,
        "can align with reachable training dynamics and decision metric, but is checkpoint/metric dependent",
        "leading eigenfeatures of the registered NTK or Fisher-renderer pullback",
        "operator code free; video/checkpoint-derived eigenfeatures counted",
        "HVP/Lanczos feasible; full-n600 custody absent",
        BasisEvidence.PREDICTED_UNMEASURED,
        None,
        _NO_DIFFERENT_FRAME_RECEIPT,
    ),
    BasisCandidate(
        BasisFamily.ZERNIKE_SPHERICAL,
        11,
        "global orthogonal modes fit disks/spheres, not the planar perspective separatrix without a chart",
        "global Zernike or spherical-harmonic polynomials",
        "generic basis free; learned coefficients counted",
        "high portability, low geometry match",
        BasisEvidence.PREDICTED_UNMEASURED,
        None,
        _NO_DIFFERENT_FRAME_RECEIPT,
    ),
    BasisCandidate(
        BasisFamily.POLAR_DIRECTIONAL_FOURIER,
        12,
        "good smooth-interior fallback; global atoms are not sparse/local at curved boundary segments",
        "global plane waves sampled on a multiscale polar frequency grid",
        "bank regenerated free; input weights counted",
        "implemented in NumPy, MLX, and generated inflate.py",
        BasisEvidence.MEASURED_THROUGH_R_N600_FORMULATION,
        0.004244,
        "bounded warm-start ep675, seed0, macOS-CPU advisory; archive bytes unmeasured",
    ),
    BasisCandidate(
        BasisFamily.SELF_ORIENTED_FOURIER,
        13,
        "locally reorients global Fourier phase to a decoder-derived tangent, but still lacks spatial windows",
        "per-pair tangent-coordinate global Fourier channels",
        "features regenerated free; 1536 additional counted decoder parameters in owed-16",
        "implemented but measured about 47-57 GiB extra live memory at n600",
        BasisEvidence.MEASURED_THROUGH_R_N600_FORMULATION,
        0.004259,
        "bounded warm-start ep675, along=8, seed0, macOS-CPU advisory; fresh-start open",
    ),
)


def basis_catalog() -> tuple[BasisCandidate, ...]:
    """Return the immutable candidate ranking used by the task-#497 artifacts."""

    return _CATALOG


@dataclass(frozen=True)
class LegacyBankStructureAudit:
    representation_label: str
    frequency_columns: int
    paired_feature_columns: int
    maximum_envelope_span: float
    spatially_localized: bool
    has_translation_index: bool
    has_spatial_window: bool
    has_parabolic_orientation_count: bool
    missing_true_curvelet_properties: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["missing_true_curvelet_properties"] = list(self.missing_true_curvelet_properties)
        return row


def audit_legacy_polar_bank(*, height: int = 17, width: int = 19) -> LegacyBankStructureAudit:
    """Re-derive that the legacy ``curvelet`` features are global plane waves.

    For every frequency vector, the paired feature envelope is exactly
    ``sin(phi)^2 + cos(phi)^2 = 1`` at every coordinate.  A constant envelope,
    no translation index, and no window mean the atom is not spatially
    localized.  The calculation is a source-structure audit, not a d_seg row.
    """

    if height <= 1 or width <= 1:
        raise ValueError("audit grid dimensions must both exceed one")
    bank = curvelet_directional_B(CurveletBankConfig())
    features = curvelet_feats(build_coords(height, width), bank).astype(np.float64)
    n_columns = int(bank.shape[1])
    paired_energy = features[:, :n_columns] ** 2 + features[:, n_columns:] ** 2
    spans = np.ptp(paired_energy, axis=0)
    maximum_span = float(spans.max(initial=0.0))
    localized = maximum_span > 1e-5
    return LegacyBankStructureAudit(
        representation_label=BasisFamily.POLAR_DIRECTIONAL_FOURIER.value,
        frequency_columns=n_columns,
        paired_feature_columns=int(features.shape[1]),
        maximum_envelope_span=maximum_span,
        spatially_localized=localized,
        has_translation_index=False,
        has_spatial_window=False,
        has_parabolic_orientation_count=True,
        missing_true_curvelet_properties=(
            "spatial_window",
            "translation_index",
            "scale_dependent_anisotropic_support",
            "localized_tight_frame_normalization",
        ),
    )


class UnsupportedBasisFamily(RuntimeError):
    """The requested family has no honest train+inflate compilation surface."""


@dataclass(frozen=True)
class BasisLeverSpec:
    """Typed basis-family stage/config parameter.

    ``legacy_fourier_ab_control`` is the explicit control for the owed A/B.
    ``polar_directional_fourier`` is retained as historical source vocabulary.
    ``self_oriented_fourier`` is retained for scoped reproduction, not selected
    as the winner.  ``siren_finer`` compiles an existing fresh-start family but
    carries no n600 winner claim.  Every genuinely different frame refuses.
    """

    family: BasisFamily = BasisFamily.LEGACY_FOURIER_AB_CONTROL
    bank_n_scales: int = 4
    bank_n_orient0: int = 6
    bank_f0: float = 2.0
    bank_base: float = 2.0
    bank_n_iso: int = 4
    max_bank_freq: float = 64.0
    n_dir_freqs: int = 4
    freq_across: float = 32.0
    freq_along: float = 8.0
    reorient_every: int = 50
    finer_bias_k: float = 10.0

    def _validated(self) -> None:
        for name in ("bank_n_scales", "bank_n_orient0", "bank_n_iso", "n_dir_freqs", "reorient_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        for name in ("bank_f0", "bank_base", "max_bank_freq", "freq_across", "freq_along", "finer_bias_k"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")

    def compile_lever(self) -> Lever:
        self._validated()
        common = {
            "--bank-n-scales": self.bank_n_scales,
            "--bank-n-orient0": self.bank_n_orient0,
            "--bank-f0": self.bank_f0,
            "--bank-base": self.bank_base,
            "--bank-n-iso": self.bank_n_iso,
            "--max-bank-freq": self.max_bank_freq,
        }
        if self.family in (BasisFamily.LEGACY_FOURIER_AB_CONTROL, BasisFamily.POLAR_DIRECTIONAL_FOURIER):
            overrides = (
                {**common, "--self-orient": False, "--basis": LEGACY_FOURIER_AB_CONTROL_ID}
                if self.family is BasisFamily.LEGACY_FOURIER_AB_CONTROL
                else {**common, "--self-orient": False}
            )
            notes = (
                "legacy Fourier A/B control only: historical global plane-wave computation; "
                "curvelet default/strict flip owed to an operator-GO n600 byte-closed "
                "realized-through-R no-regression verdict"
                if self.family is BasisFamily.LEGACY_FOURIER_AB_CONTROL
                else "task497 measured fallback: global polar directional Fourier; owed16 OFF won the "
                "bounded warm-start n600 cell with 1536 fewer decoder params; no archive-byte claim"
            )
        elif self.family is BasisFamily.SELF_ORIENTED_FOURIER:
            overrides = {
                **common,
                "--self-orient": True,
                "--n-dir-freqs": self.n_dir_freqs,
                "--freq-across": self.freq_across,
                "--freq-along": self.freq_along,
                "--reorient-every": self.reorient_every,
            }
            notes = (
                "task497 scoped reproduction only: self-oriented global Fourier; bounded warm-start "
                "through-R n600 did not beat OFF; fresh-start remains open"
            )
        elif self.family is BasisFamily.SIREN_FINER:
            overrides = {
                **common,
                "--self-orient": False,
                "--activation": "hosc",
                "--siren-init": True,
                "--finer-bias-init": True,
                "--finer-bias-k": self.finer_bias_k,
            }
            notes = (
                "task497 open fresh-start learned-frequency family; compiles existing FINER/SIREN "
                "surface but carries no n600 basis-win claim"
            )
        elif self.family is BasisFamily.WINDOWED_CURVELET:
            overrides = {"--basis": "windowed_curvelet"}
            notes = (
                "FEED-cvl-throughR selected windowed-directional frame; trainer MLX/NumPy parity "
                "and generated inflate receiver op parity are wired; equal-config byte-closed n600 "
                "realized d_seg remains OWED (operator-GO, PREPARED_NOT_FIRED); no family-win claim"
            )
        else:
            raise UnsupportedBasisFamily(
                f"{self.family.value} has no train+generated-inflate op-parity implementation and "
                "no equal-budget real-n600 through-R receipt; refusing a design-only family"
            )

        unknown = sorted(set(overrides) - set(real_trainer_flags()))
        if unknown:
            raise RuntimeError(f"basis compiler invented trainer flags: {unknown}")
        return Lever(
            name=f"basis_family::{self.family.value}",
            overrides=overrides,
            notes=notes,
        )


def lever_argv(lever: Lever) -> tuple[str, ...]:
    """Compile a Lever to real argparse tokens and parse it with the real parser."""

    argv: list[str] = []
    for flag, value in lever.overrides.items():
        if isinstance(value, bool):
            argv.append(flag if value else f"--no-{flag[2:]}")
        elif value is not None:
            argv.extend((flag, str(value)))
    parser = build_real_trainer_parser()
    # The real trainer requires --out-dir.  Supply a non-path sentinel only to
    # the parser validation call; it is not part of the basis lever and no
    # directory is created.
    parser.parse_args(["--out-dir", "DSL_VALIDATION_ONLY", *argv])
    return tuple(argv)


@dataclass(frozen=True)
class InflateCompileContract:
    family: BasisFamily
    compiled: bool
    generic_regenerated_state: tuple[str, ...]
    counted_state: tuple[str, ...]
    train_functions: tuple[str, ...]
    inflate_functions: tuple[str, ...]
    rule118_status: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["family"] = self.family.value
        return row


def inflate_compile_contract(spec: BasisLeverSpec) -> InflateCompileContract:
    """Confirm deterministic decoder compilation for implemented families only."""

    lever = spec.compile_lever()
    lever_argv(lever)
    root = Path(__file__).resolve().parents[3]
    inflate_source = (root / "tools/levelset_byte_close_and_eval.py").read_text()
    if spec.family is BasisFamily.WINDOWED_CURVELET:
        required = ("def _windowed_curvelet_feats", "def _basis_feats")
        if not all(token in inflate_source for token in required):
            raise RuntimeError("generated inflate source lost windowed-curvelet regeneration functions")
        inflate_functions = ("_windowed_curvelet_feats", "_basis_feats")
        train_functions = ("windowed_curvelet_feats", "mlx_parity_check")
        regenerated = ("windowed_curvelet_config", "basis_family")
    else:
        required = ("def _curvelet_B", "def _curvelet_feats")
        if not all(token in inflate_source for token in required):
            raise RuntimeError("generated inflate source lost polar-bank regeneration functions")
        inflate_functions = ("_curvelet_B", "_curvelet_feats")
        train_functions = ("curvelet_directional_B", "curvelet_feats")
        regenerated = ("bank_n_scales", "bank_n_orient0", "bank_f0", "bank_base", "bank_n_iso", "max_bank_freq")
    if spec.family is BasisFamily.SELF_ORIENTED_FOURIER:
        if "def _dir_feats" not in inflate_source:
            raise RuntimeError("generated inflate source lost self-orient feature regeneration")
        inflate_functions += ("_dir_feats",)
        regenerated += ("n_dir_freqs", "freq_across", "freq_along", "self_orient_fixed_point")
    elif spec.family is BasisFamily.SIREN_FINER:
        if "def _act" not in inflate_source:
            raise RuntimeError("generated inflate source lost periodic activation support")
        train_functions += ("periodic_activation", "finer_bias_initialization")
        inflate_functions += ("_act",)
    return InflateCompileContract(
        family=spec.family,
        compiled=True,
        generic_regenerated_state=regenerated,
        counted_state=("in_proj.weight", "remaining decoder weights", "per-pair code"),
        train_functions=train_functions,
        inflate_functions=inflate_functions,
        rule118_status=(
            "generic basis-generation code/scalars are deterministic and free; learned/video-derived "
            "weights and codes remain counted"
        ),
    )


@dataclass(frozen=True)
class BasisMetricInterface:
    """Non-owning interface for task #500's metric provider."""

    family_id: str
    primal_atom_map: str
    required_metric_quantity: str
    metric_id: str = "argmax_native_vjp_fidelity_v1"
    state_receipt_schema: str = "reachable_decision_geometry_fidelity.v1"
    selection_receipt_schema: str = "reachable_decision_preconditioner_selection.v1"
    candidate_preconditioner: str = "winner_rival_margin_fisher_natural"
    provider_module: str = "tac.scorer_surrogate.vjp_fidelity"
    selection_status: str = "NO-VERDICT_DATA_CUSTODY"
    selection_law: str = "optimal_basis_equal_budget_through_r_v1"


def basis_metric_interface(family: BasisFamily) -> BasisMetricInterface:
    """Expose basis-perp-metric duality without implementing the metric."""

    return BasisMetricInterface(
        family_id=family.value,
        primal_atom_map="psi_i(x) -> witness parameters theta_B",
        required_metric_quantity="registered renderer/Fisher pullback Gram <psi_i, G_q psi_j>",
    )


@dataclass(frozen=True)
class BasisABConfigPair:
    """Pure V9 ideal-mod32 compile receipt; launch readiness is explicit."""

    control: Any
    treatment: Any
    control_basis: str
    treatment_basis: str
    differing_flags_excluding_out_dir: tuple[str, ...]
    composition_blockers: tuple[str, ...]
    launch_ready: bool = False
    status: str = "BLOCKED_FAIL_CLOSED_BUILD_THEN_OPERATOR_GO"


def _trainer_flag_map(config: Any) -> dict[str, Any]:
    return dict(config.to_trainer_flags())


def v9_ideal_mod32_basis_ab_configs(
    *,
    gt_cache_path: str = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    num_pairs: int = 600,
    epochs: int = 3000,
    control_out_dir: str = "experiments/results/v9_cgauge_ideal_mod32_legacy_fourier_ab_control_20260715",
    treatment_out_dir: str = "experiments/results/v9_cgauge_ideal_mod32_windowed_curvelet_20260715",
) -> BasisABConfigPair:
    """Return the DSL-authored V9 ideal-mod32 basis compile receipt.

    Both arms explicitly select exactly one basis lever.  Excluding ``--out-dir``,
    their compiled trainer flags must differ only by ``--basis``.  That proves
    typed configuration identity; it does *not* prove the treatment composes with
    the consumers selected by those flags.  The current pair is deliberately
    blocked: V9 inherits Fourier-specific IPE and a taper with no generated-
    receiver fold, while the current ``windowed_curvelet`` consumer is the
    invalidated spatial wave packet rather than the clean literal polar-wedge
    construction.  This function is pure construction: no launch, no training,
    and no score/family verdict.
    """

    from tac.witness_dsl.spec_v9_cgauge import compile_v9_cgauge_ideal_mod32_launch_config

    control = compile_v9_cgauge_ideal_mod32_launch_config(
        gt_cache_path=gt_cache_path,
        num_pairs=num_pairs,
        epochs=epochs,
        out_dir=control_out_dir,
    ).with_dsl_lever_factories("LegacyFourierABControl")
    treatment = compile_v9_cgauge_ideal_mod32_launch_config(
        gt_cache_path=gt_cache_path,
        num_pairs=num_pairs,
        epochs=epochs,
        out_dir=treatment_out_dir,
    ).with_dsl_lever_factories("WindowedCurveletBasis")
    c_flags = _trainer_flag_map(control)
    t_flags = _trainer_flag_map(treatment)
    differing = tuple(
        sorted(
            flag
            for flag in set(c_flags) | set(t_flags)
            if flag != "--out-dir" and c_flags.get(flag) != t_flags.get(flag)
        )
    )
    if differing != ("--basis",):
        raise RuntimeError(
            "V9 ideal-mod32 basis A/B must differ only by --basis excluding --out-dir; "
            f"got {differing}"
        )
    blockers: list[str] = []
    if str(t_flags.get("--render-aa", "none")) != "none":
        blockers.append(
            "render_aa=ipe is polar-Fourier-specific: literal curvelet wedges require "
            "per-frequency footprint attenuation before wedge summation"
        )
    # Boolean store_true flags compile as ``(flag, None)``; membership, not
    # truthiness of the mapped value, is the activation predicate.
    if "--dseg-aware-taper" in t_flags:
        blockers.append(
            "dseg-aware taper is GT-derived and is not folded into in_proj weights or "
            "regenerated by the generated receiver"
        )
    blockers.append(
        "the live windowed_curvelet token selects the invalidated spatial wave-packet "
        "implementation; the clean literal polar-frequency-wedge train/MLX/checkpoint/"
        "inflate consumer is absent"
    )
    return BasisABConfigPair(
        control=control,
        treatment=treatment,
        control_basis=str(c_flags["--basis"]),
        treatment_basis=str(t_flags["--basis"]),
        differing_flags_excluding_out_dir=differing,
        composition_blockers=tuple(blockers),
    )


__all__ = [
    "BasisABConfigPair",
    "BasisCandidate",
    "BasisEvidence",
    "BasisFamily",
    "BasisLeverSpec",
    "BasisMetricInterface",
    "InflateCompileContract",
    "LegacyBankStructureAudit",
    "UnsupportedBasisFamily",
    "audit_legacy_polar_bank",
    "basis_catalog",
    "basis_metric_interface",
    "inflate_compile_contract",
    "lever_argv",
    "v9_ideal_mod32_basis_ab_configs",
]
