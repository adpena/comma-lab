# SPDX-License-Identifier: MIT
"""Typed, default-off policy for the round-3 fidelity-wall probe."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

ROUND2_MEASURED_HELDOUT_COSINE = 0.0014157933865487525
MEANINGFUL_NOISE_MULTIPLE = 50.0
PREREGISTERED_INPUT_COSTATE_COSINE_BAR = (
    MEANINGFUL_NOISE_MULTIPLE * ROUND2_MEASURED_HELDOUT_COSINE
)

_BINARY32_FRACTION_BITS = 23
_IDEAL_CONTRACTION_DENOMINATOR = 3
_DERIVED_FIT_EPOCHS = math.ceil(
    _BINARY32_FRACTION_BITS
    * math.log(2.0)
    / math.log(float(_IDEAL_CONTRACTION_DENOMINATOR))
)


@dataclass(frozen=True)
class ReplaceRound3FidelityWallPolicy:
    """Compile one new formulation instance without inventing trainer flags.

    The order, split, prefix, random lift, and decision bars are sealed before
    any real-state feature or target measurement.  Changing any of them is a
    new instance, not a resume of this one.
    """

    mode: str = "replace_round3_fidelity_wall"
    seed: int = 455
    n_pairs: int = 600
    checkpoint_count: int = 3
    holdout_period: int = 5
    train_lattice_stride_on_prefix: int = 4
    fit_epochs: int = _DERIVED_FIT_EPOCHS
    teacher_batch_size: int = 1
    prefix_module: str = "encoder.model.blocks.0.0.bn1"
    prefix_cut: str = "pre_squeeze_excite_pre_global_pool"
    base_feature_count: int = 42
    rff_frequency_count: int = 16
    rff_seed: int = 455
    rung_order: tuple[str, ...] = (
        "pre_se_prefix_linear",
        "pre_se_prefix_rff",
        "margin_or_mass_localizer",
    )
    localizer_order: tuple[str, ...] = (
        "source_margin_risk",
        "rff_costate_mass_ridge",
    )
    input_costate_cosine_bar: float = PREREGISTERED_INPUT_COSTATE_COSINE_BAR
    positive_dot_state_fraction_bar: float = 0.60
    flip_risk_area_fraction: float = 0.047
    costate_mass_uplift_over_uniform_bar: float = 10.0
    fallback: str = "full_exact_teacher"
    live_training_enabled: bool = False
    research_only: bool = True
    score_claim: bool = False
    promotion_eligible: bool = False
    fore_weighting_enabled: bool = False

    def __post_init__(self) -> None:
        if self.mode != "replace_round3_fidelity_wall":
            raise ValueError("mode is sealed for the round-3 formulation")
        if (self.seed, self.n_pairs, self.checkpoint_count) != (455, 600, 3):
            raise ValueError("seed, n600 cohort, and three-stage replay are sealed")
        if self.holdout_period != 5 or self.train_lattice_stride_on_prefix != 4:
            raise ValueError("split and prefix lattice are preregistered")
        if self.fit_epochs != _DERIVED_FIT_EPOCHS or self.teacher_batch_size != 1:
            raise ValueError("fit horizon and exact per-state teacher batching are derived")
        if self.prefix_module != "encoder.model.blocks.0.0.bn1":
            raise ValueError("the local pre-SE prefix cut is sealed")
        if self.prefix_cut != "pre_squeeze_excite_pre_global_pool":
            raise ValueError("prefix locality may not be weakened on resume")
        if self.base_feature_count != 42 or self.rff_frequency_count != 16:
            raise ValueError("feature and RFF widths are preregistered without a sweep")
        if self.rff_seed != self.seed:
            raise ValueError("all RNG derives from the single recorded seed")
        if self.rung_order != (
            "pre_se_prefix_linear",
            "pre_se_prefix_rff",
            "margin_or_mass_localizer",
        ):
            raise ValueError("EV-order rung sequence is sealed")
        if self.localizer_order != (
            "source_margin_risk",
            "rff_costate_mass_ridge",
        ):
            raise ValueError("target-reformulation sub-order is sealed")
        if self.input_costate_cosine_bar != PREREGISTERED_INPUT_COSTATE_COSINE_BAR:
            raise ValueError("direction bar must remain 50x round-2 measured noise")
        if self.positive_dot_state_fraction_bar != 0.60:
            raise ValueError("positive-dot state fraction bar is preregistered")
        if (
            self.flip_risk_area_fraction != 0.047
            or self.costate_mass_uplift_over_uniform_bar != 10.0
        ):
            raise ValueError("target-reformulation area and uplift are preregistered")
        if self.fallback != "full_exact_teacher":
            raise ValueError("failed research arms must fall back to the exact teacher")
        if self.live_training_enabled or not self.research_only:
            raise ValueError("round 3 has no live-training authority")
        if self.score_claim or self.promotion_eligible:
            raise ValueError("local gradient evidence cannot claim score or promotion")
        if self.fore_weighting_enabled:
            raise ValueError("current replay lacks FORE transition/support custody")

    @property
    def train_state_count(self) -> int:
        return self.n_pairs - self.n_pairs // self.holdout_period

    @property
    def heldout_state_count(self) -> int:
        return self.n_pairs // self.holdout_period

    @property
    def effective_cached_label_uses(self) -> int:
        return self.train_state_count * self.fit_epochs

    @property
    def label_only_teacher_amortization_x(self) -> float:
        return self.effective_cached_label_uses / self.train_state_count

    @property
    def inclusive_teacher_amortization_x(self) -> float:
        return self.effective_cached_label_uses / self.n_pairs

    @property
    def localizer_mass_fraction_bar(self) -> float:
        return self.flip_risk_area_fraction * self.costate_mass_uplift_over_uniform_bar

    def compile_measurement_contract(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "train_state_count": self.train_state_count,
                "heldout_state_count": self.heldout_state_count,
                "effective_cached_label_uses": self.effective_cached_label_uses,
                "label_only_teacher_amortization_x": self.label_only_teacher_amortization_x,
                "inclusive_teacher_amortization_x": self.inclusive_teacher_amortization_x,
                "localizer_mass_fraction_bar": self.localizer_mass_fraction_bar,
                "bar_derivation": (
                    "DERIVED 50 * MEASURED round2 heldout cosine noise; "
                    "cosine^2 is the normalized projected directional-energy fraction"
                ),
                "feature_chart": (
                    "bias + 32 local frozen pre-SE activations + 5 source-label one-hot + "
                    "tanh(source margin) + 3 checkpoint one-hot"
                ),
                "target": (
                    "exact adjoint at the sealed prefix; predicted input costate is the exact "
                    "VJP through the frozen local prefix"
                ),
                "ridge_policy": "DERIVED lambda=lambda_max(X'X/n)",
                "step_policy": "DERIVED eta=2/(mu+L)",
                "live_trainer_argv": [],
                "authority": "macOS-CPU local fp32 training-gradient evidence only",
                "constant_provenance": {
                    "seed_n600_split": "SOURCE round-2 sealed replay instance",
                    "prefix": "DERIVED locality cut before first squeeze-excite/global pooling",
                    "rff_width": "ASSUMED single bounded first probe; no sweep",
                    "direction_bar": "DERIVED 50x round-2 measured heldout cosine noise",
                    "positive_dot_bar": "ASSUMED 60 percent cross-state sign-consistency guard",
                    "localizer_area": "MEASURED-INHERITED approximately 4.7 percent annulus area",
                    "localizer_uplift": "ASSUMED 10x uniform-mass concentration gate",
                    "localizer_order": (
                        "DERIVED cheapest parameter-free canonical margin first; then one "
                        "convex RFF ridge on exact input-costate L2 mass"
                    ),
                    "FORE": "REFUSED current instance; no Markov transition/support receipt",
                },
            }
        )
        return payload


__all__ = [
    "MEANINGFUL_NOISE_MULTIPLE",
    "PREREGISTERED_INPUT_COSTATE_COSINE_BAR",
    "ROUND2_MEASURED_HELDOUT_COSINE",
    "ReplaceRound3FidelityWallPolicy",
]
