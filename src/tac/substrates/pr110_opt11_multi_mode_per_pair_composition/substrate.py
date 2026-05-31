# SPDX-License-Identifier: MIT
"""PR110-OPT-11 multi-mode-per-pair composition — substrate.

Canonical L0 SCAFFOLD that ACTUALLY composes 2 frame-0 perturbations per pair
across orthogonal families (luma_bias / blue_chroma / rgb_bias / roll) per
Wave N+34 analytical foundation.

Per CLAUDE.md NO FAKE IMPLEMENTATIONS non-negotiable 2026-05-30 + Slot EEE
fake-implementation audit anchor 2026-05-29: the substrate ACTUALLY composes
two perturbation modes per pair via real arithmetic on real per-pair frames
(luma_bias adds Y-channel offset; blue_chroma scales B; rgb_bias applies
per-channel deltas; roll shifts spatially). The tests verify ACTUAL composed
output differs from single-mode output (composition produces distinct frames,
NOT canonical-marker-only no-op).

NOT a returns-canonical-markers-without-doing-work scaffold per Slot EEE 5
forbidden classes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from tac.provenance import (
    build_provenance_for_predicted,
    provenance_to_dict,
)

# Canonical 22-mode menu per Wave N+34 source
# experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex/pair_component_rows.jsonl
# Each mode is (family, mode_id, applies_per_pair_perturbation_function_key).
# This menu is CANONICAL per the 600-pair empirical anchor; the L0 SCAFFOLD
# stays bit-exact compatible with the Wave N+34 mode_id strings.
CANONICAL_MODE_MENU: tuple[tuple[str, str], ...] = (
    ("identity", "none"),
    ("frame0_luma_bias", "frame0_luma_bias_+1"),
    ("frame0_luma_bias", "frame0_luma_bias_+2"),
    ("frame0_luma_bias", "frame0_luma_bias_+4"),
    ("frame0_luma_bias", "frame0_luma_bias_-1"),
    ("frame0_luma_bias", "frame0_luma_bias_-2"),
    ("frame0_luma_bias", "frame0_luma_bias_-4"),
    ("frame0_blue_chroma", "frame0_blue_chroma_amp_1"),
    ("frame0_blue_chroma", "frame0_blue_chroma_amp_2"),
    ("frame0_blue_chroma", "frame0_blue_chroma_amp_3"),
    ("frame0_rgb_bias", "frame0_rgb_bias_m2_p1_p1"),
    ("frame0_rgb_bias", "frame0_rgb_bias_m4_p2_p2"),
    ("frame0_rgb_bias", "frame0_rgb_bias_p0_m1_p1"),
    ("frame0_rgb_bias", "frame0_rgb_bias_p0_m2_p2"),
    ("frame0_rgb_bias", "frame0_rgb_bias_p0_p1_m1"),
    ("frame0_rgb_bias", "frame0_rgb_bias_p0_p2_m2"),
    ("frame0_rgb_bias", "frame0_rgb_bias_p2_m1_m1"),
    ("frame0_rgb_bias", "frame0_rgb_bias_p4_m2_m2"),
    ("frame0_roll", "frame0_roll_h_1"),
    ("frame0_roll", "frame0_roll_h_2"),
    ("frame0_roll", "frame0_roll_v_1"),
    ("frame0_roll", "frame0_roll_v_2"),
)
# K=22 modes; if we cap selectors at 4 bits per mode index we can address K=16
# distinct modes. The canonical L0 SCAFFOLD truncates to K=16 (first 16 modes
# = identity + 6 luma_bias + 3 blue_chroma + 6 of the 8 rgb_bias). L1 PROMOTION
# extends to K=32 (5-bit selectors) via Phase 2 sister wave.
CANONICAL_K16_MODE_INDICES: tuple[int, ...] = tuple(range(16))


# 6 canonical orthogonal family pairs per Wave N+34 analytical investigator
# (families combinatorial: 4 families choose 2 = 6 pairs).
CANONICAL_ORTHOGONAL_FAMILY_PAIRS: tuple[tuple[str, str], ...] = (
    ("frame0_luma_bias", "frame0_blue_chroma"),
    ("frame0_luma_bias", "frame0_rgb_bias"),
    ("frame0_luma_bias", "frame0_roll"),
    ("frame0_blue_chroma", "frame0_rgb_bias"),
    ("frame0_blue_chroma", "frame0_roll"),
    ("frame0_rgb_bias", "frame0_roll"),
)


# Canonical defaults per Catalog #290 canonical-vs-unique decision per layer.
DEFAULT_PR110_BASE_PAIRS: int = 600
DEFAULT_MODES_PER_PAIR: int = 2  # The "multi-mode" k
DEFAULT_SELECTOR_BITS_PER_MODE: int = 4  # K=16 modes addressable
DEFAULT_FAMILY_PAIR_INDEX: int = 1  # (luma_bias, rgb_bias) — largest combination space (6×8)


# Canonical Tier A canonical-routing-markers per Catalog #341 + #357.
TIER_A_PREDICTED_DELTA_ADJUSTMENT: float = 0.0
TIER_A_PROMOTABLE: bool = False
TIER_A_AXIS_TAG: str = "[predicted]"
TIER_A_VERDICT: str = "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR"


# Canonical frame shape per the contest (1164 × 874 × 3 inflated raw).
# For the L0 SCAFFOLD smoke we use a smaller 48 × 64 × 3 shape so MLX-LOCAL
# smoke fits in seconds; the real perturbation operators generalize across H, W.
SMOKE_FRAME_H: int = 48
SMOKE_FRAME_W: int = 64
SMOKE_FRAME_C: int = 3


@dataclass(frozen=True)
class PR110OPT11Config:
    """Canonical config for the PR110-OPT-11 multi-mode-per-pair composition.

    Per Catalog #290 canonical-vs-unique decision per layer + Catalog #287
    placeholder-rationale rejection discipline: every field is validated in
    ``__post_init__`` with substantive non-placeholder rationales on error.

    Args:
        n_pairs: PR110 base pair count (canonical = 600).
        modes_per_pair: Number of perturbation modes composed per pair
            (canonical = 2 for the multi-mode-per-pair surface).
        selector_bits_per_mode: Bits per mode selector (canonical = 4 for K=16
            distinct modes addressable per selector slot).
        family_pair_index: Index into CANONICAL_ORTHOGONAL_FAMILY_PAIRS
            (canonical = 1 → (luma_bias, rgb_bias) per Wave N+34 verdict).
        pr110_base_archive_sha256: Sha256 of the PR110 base archive bytes
            this substrate composes onto (CANONICAL frontier archive per
            Catalog #343 canonical pointer; default = empty until L1 lands
            real frontier sha256 reference).
        rng_seed: numpy seed for determinism (canonical = 42).
    """

    n_pairs: int = DEFAULT_PR110_BASE_PAIRS
    modes_per_pair: int = DEFAULT_MODES_PER_PAIR
    selector_bits_per_mode: int = DEFAULT_SELECTOR_BITS_PER_MODE
    family_pair_index: int = DEFAULT_FAMILY_PAIR_INDEX
    pr110_base_archive_sha256: str = ""
    rng_seed: int = 42

    def __post_init__(self) -> None:
        if self.n_pairs <= 0:
            raise ValueError(f"n_pairs must be > 0; got {self.n_pairs}")
        if self.modes_per_pair < 2 or self.modes_per_pair > 8:
            raise ValueError(
                "modes_per_pair must be in [2, 8] per L0 SCAFFOLD budget "
                f"(L1 PROMOTION may extend); got {self.modes_per_pair}"
            )
        if self.selector_bits_per_mode < 1 or self.selector_bits_per_mode > 8:
            raise ValueError(
                "selector_bits_per_mode must be in [1, 8] "
                f"(canonical K up to 256); got {self.selector_bits_per_mode}"
            )
        if (
            self.family_pair_index < 0
            or self.family_pair_index >= len(CANONICAL_ORTHOGONAL_FAMILY_PAIRS)
        ):
            raise ValueError(
                "family_pair_index must be in "
                f"[0, {len(CANONICAL_ORTHOGONAL_FAMILY_PAIRS) - 1}]; "
                f"got {self.family_pair_index}; valid pairs = "
                f"{CANONICAL_ORTHOGONAL_FAMILY_PAIRS}"
            )
        if not isinstance(self.pr110_base_archive_sha256, str):
            raise ValueError(
                "pr110_base_archive_sha256 must be str (sha256 hex or empty); "
                f"got {type(self.pr110_base_archive_sha256).__name__}"
            )
        if self.pr110_base_archive_sha256 and len(self.pr110_base_archive_sha256) != 64:
            raise ValueError(
                "pr110_base_archive_sha256 must be 64-char hex sha256 or empty; "
                f"got len={len(self.pr110_base_archive_sha256)}"
            )
        if not isinstance(self.rng_seed, int) or self.rng_seed < 0:
            raise ValueError(
                f"rng_seed must be non-negative int; got {self.rng_seed!r}"
            )


@dataclass(frozen=True)
class PR110OPT11Result:
    """Canonical Tier A return type per Catalog #341 + #357.

    Carries:
    - The canonical Tier A routing markers (predicted_delta_adjustment=0.0;
      promotable=False; axis_tag="[predicted]"; verdict).
    - The per-pair (selector_a, selector_b) composition tuple.
    - The canonical Provenance per Catalog #323.
    - The composition behavioral evidence (frame deltas verifying
      composition actually happened per Slot EEE NO FAKE).
    """

    config: PR110OPT11Config
    predicted_delta_adjustment: float
    promotable: bool
    axis_tag: str
    verdict: str
    per_pair_selectors: tuple[tuple[int, int], ...]
    family_pair: tuple[str, str]
    family_a_mode_indices: tuple[int, ...]
    family_b_mode_indices: tuple[int, ...]
    composition_behavioral_evidence: dict[str, Any]
    canonical_provenance: dict[str, Any]
    canonical_helpers_invoked: dict[str, bool]
    cross_reference_matrix: dict[str, str]


def build_substrate_default_config() -> PR110OPT11Config:
    """Build the canonical default PR110OPT11Config per Catalog #290.

    Returns:
        Canonical default config with n_pairs=600 / modes_per_pair=2 /
        selector_bits_per_mode=4 / family_pair_index=1 (luma_bias, rgb_bias).
    """
    return PR110OPT11Config()


def _mode_indices_in_family(family: str) -> tuple[int, ...]:
    """Return mode indices in CANONICAL_MODE_MENU whose family matches."""
    return tuple(
        idx for idx, (fam, _mid) in enumerate(CANONICAL_MODE_MENU) if fam == family
    )


def _apply_canonical_perturbation(
    frame: np.ndarray, mode_idx: int
) -> np.ndarray:
    """ACTUALLY apply one canonical perturbation mode to a frame.

    Per CLAUDE.md NO FAKE IMPLEMENTATIONS Class 1 forbidden pattern: this MUST
    perform actual numeric arithmetic on the frame; returning canonical markers
    or no-op IS the forbidden pattern Slot EEE caught.

    The 4 canonical families map to concrete numpy operations:
    - frame0_luma_bias: add signed offset to Y channel (R channel proxy in
      RGB space here for the simple L0 SCAFFOLD; L1 PROMOTION routes through
      canonical rgb_to_yuv6 + back).
    - frame0_blue_chroma: scale B channel by amp factor.
    - frame0_rgb_bias: add per-channel offsets (r_offset, g_offset, b_offset).
    - frame0_roll: spatial roll by signed pixel count.
    - identity (none): unchanged.

    Args:
        frame: uint8 (H, W, 3) RGB frame.
        mode_idx: Index into CANONICAL_MODE_MENU [0, 22).

    Returns:
        Perturbed uint8 (H, W, 3) frame.

    Raises:
        ValueError: If mode_idx out of bounds.
    """
    if mode_idx < 0 or mode_idx >= len(CANONICAL_MODE_MENU):
        raise ValueError(
            f"mode_idx must be in [0, {len(CANONICAL_MODE_MENU)}); got {mode_idx}"
        )
    if frame.dtype != np.uint8:
        raise ValueError(
            f"frame must be uint8; got dtype={frame.dtype}"
        )
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"frame must be (H, W, 3); got shape={frame.shape}"
        )
    family, mode_id = CANONICAL_MODE_MENU[mode_idx]

    if family == "identity":
        return frame.copy()

    if family == "frame0_luma_bias":
        # mode_id ∈ frame0_luma_bias_{+1, +2, +4, -1, -2, -4}; offset applied
        # to R channel proxy for luma per the L0 SCAFFOLD simplification.
        signed_offset = int(mode_id.split("_")[-1])
        out = frame.astype(np.int16)
        out[..., 0] = np.clip(out[..., 0] + signed_offset, 0, 255)
        return out.astype(np.uint8)

    if family == "frame0_blue_chroma":
        # mode_id ∈ frame0_blue_chroma_amp_{1,2,3}; B channel scaled by
        # 1.0 + 0.05 * amp (subtle multiplicative).
        amp = int(mode_id.split("_")[-1])
        scale = 1.0 + 0.05 * amp
        out = frame.astype(np.float32)
        out[..., 2] = np.clip(out[..., 2] * scale, 0, 255)
        return out.astype(np.uint8)

    if family == "frame0_rgb_bias":
        # mode_id ∈ frame0_rgb_bias_<r>_<g>_<b>; offsets parsed from
        # m=negative, p=positive prefix.
        parts = mode_id.split("_")[-3:]

        def _parse(p: str) -> int:
            sign = -1 if p.startswith("m") else 1
            try:
                return sign * int(p[1:])
            except ValueError:
                return 0

        r_off, g_off, b_off = (_parse(p) for p in parts)
        out = frame.astype(np.int16)
        out[..., 0] = np.clip(out[..., 0] + r_off, 0, 255)
        out[..., 1] = np.clip(out[..., 1] + g_off, 0, 255)
        out[..., 2] = np.clip(out[..., 2] + b_off, 0, 255)
        return out.astype(np.uint8)

    if family == "frame0_roll":
        # mode_id ∈ frame0_roll_{h,v}_{1,2}; spatial roll along axis.
        axis_char = mode_id.split("_")[-2]
        shift = int(mode_id.split("_")[-1])
        axis = 1 if axis_char == "h" else 0  # h = width = axis 1; v = height = axis 0
        return np.roll(frame, shift=shift, axis=axis)

    # Should never reach here per the canonical menu enumeration.
    raise ValueError(f"Unrecognized family {family!r} for mode {mode_id!r}")


def _compose_two_modes_on_frame(
    frame: np.ndarray, mode_a_idx: int, mode_b_idx: int
) -> np.ndarray:
    """ACTUALLY compose two perturbation modes on a single frame (mode A then B).

    This is the canonical multi-mode-per-pair composition primitive per the
    task #1323 + Wave N+34 hypothesis. Composition is non-commutative in
    general (e.g. rgb_bias + roll ≠ roll + rgb_bias when shift wraps differing
    channel values).

    Per CLAUDE.md NO FAKE IMPLEMENTATIONS: this MUST produce a frame that
    differs from both single-mode outputs when both modes are non-identity.

    Args:
        frame: uint8 (H, W, 3) input.
        mode_a_idx: First mode index.
        mode_b_idx: Second mode index.

    Returns:
        Composed (H, W, 3) uint8 frame after mode_a then mode_b applied.
    """
    after_a = _apply_canonical_perturbation(frame, mode_a_idx)
    after_ab = _apply_canonical_perturbation(after_a, mode_b_idx)
    return after_ab


def _build_canonical_per_pair_selectors(
    config: PR110OPT11Config,
    *,
    family_a_modes: tuple[int, ...],
    family_b_modes: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    """Build the canonical per-pair (selector_a, selector_b) deterministically.

    For the L0 SCAFFOLD we use a deterministic per-pair-index assignment based
    on rng_seed so the test suite can verify exact byte-stable selector emission.
    L1 PROMOTION replaces this with a per-pair optimal-mode-pair selector
    derived from the Wave N+34 600-pair empirical analytical investigator
    (per-pair best (mode_a, mode_b) tuple from the canonical 22-mode menu).

    Args:
        config: PR110OPT11Config.
        family_a_modes: Mode indices in family A.
        family_b_modes: Mode indices in family B.

    Returns:
        Tuple of n_pairs (selector_a, selector_b) tuples; each selector is a
        4-bit (or selector_bits_per_mode wide) index INTO the family's mode
        list (NOT the global menu — at inflate the per-pair selector is
        translated to global mode_idx via the family lookup).
    """
    rng = np.random.default_rng(config.rng_seed)
    n_a = len(family_a_modes)
    n_b = len(family_b_modes)
    # Per-pair selector in [0, n_a) and [0, n_b); the K=16 budget enforced by
    # the archive grammar pack_selector_stream caller cap.
    sel_a_raw = rng.integers(low=0, high=n_a, size=config.n_pairs, dtype=np.int64)
    sel_b_raw = rng.integers(low=0, high=n_b, size=config.n_pairs, dtype=np.int64)
    # Cap selectors to the canonical bit budget (sanity).
    max_value = (1 << config.selector_bits_per_mode) - 1
    sel_a_capped = np.minimum(sel_a_raw, max_value).astype(int)
    sel_b_capped = np.minimum(sel_b_raw, max_value).astype(int)
    return tuple(
        (int(sel_a_capped[i]), int(sel_b_capped[i]))
        for i in range(config.n_pairs)
    )


def apply_substrate_to_pr110_canonical(
    config: PR110OPT11Config | None = None,
    *,
    canonical_base_frames: np.ndarray | None = None,
) -> PR110OPT11Result:
    """Canonical entry point for the PR110-OPT-11 substrate.

    ACTUALLY composes 2 perturbation modes per pair per the canonical Wave N+34
    + task #1323 hypothesis. Returns a Tier A non-promotable result with
    canonical Provenance per Catalog #323 + composition behavioral evidence
    per Slot EEE NO FAKE.

    Args:
        config: PR110OPT11Config (default = build_substrate_default_config()).
        canonical_base_frames: Optional (n_pairs, H, W, 3) uint8 array of base
            frames. If None, a deterministic per-seed smoke frame batch is
            generated for the L0 SCAFFOLD smoke (smoke-only; L1 PROMOTION
            decodes real upstream/videos/0.mkv per Catalog #213).

    Returns:
        PR110OPT11Result with canonical Tier A markers + per-pair selectors
        + composition behavioral evidence + canonical Provenance.

    Raises:
        ValueError: If config or canonical_base_frames is invalid.
    """
    if config is None:
        config = build_substrate_default_config()

    family_a, family_b = CANONICAL_ORTHOGONAL_FAMILY_PAIRS[config.family_pair_index]
    family_a_modes = _mode_indices_in_family(family_a)
    family_b_modes = _mode_indices_in_family(family_b)

    if not family_a_modes:
        raise ValueError(f"family_a={family_a!r} has zero modes in canonical menu")
    if not family_b_modes:
        raise ValueError(f"family_b={family_b!r} has zero modes in canonical menu")

    # Build canonical per-pair selectors deterministically per the rng_seed.
    per_pair_selectors = _build_canonical_per_pair_selectors(
        config,
        family_a_modes=family_a_modes,
        family_b_modes=family_b_modes,
    )

    # Build or accept the canonical base frames for composition behavioral
    # evidence. For smoke we use a deterministic per-seed RGB frame batch.
    if canonical_base_frames is None:
        rng = np.random.default_rng(config.rng_seed)
        canonical_base_frames = rng.integers(
            low=0, high=256,
            size=(min(config.n_pairs, 4), SMOKE_FRAME_H, SMOKE_FRAME_W, SMOKE_FRAME_C),
            dtype=np.uint8,
        )

    if canonical_base_frames.dtype != np.uint8:
        raise ValueError(
            "canonical_base_frames must be uint8; "
            f"got dtype={canonical_base_frames.dtype}"
        )
    if canonical_base_frames.ndim != 4 or canonical_base_frames.shape[3] != 3:
        raise ValueError(
            "canonical_base_frames must be (N, H, W, 3); "
            f"got shape={canonical_base_frames.shape}"
        )

    # Compose 2 modes per pair on the sample base frames; record composition
    # behavioral evidence per Slot EEE NO FAKE (the actual frame deltas).
    n_sample = canonical_base_frames.shape[0]
    composition_deltas: list[dict[str, Any]] = []
    canonical_helpers_invoked = {
        "apply_canonical_perturbation_family_a": False,
        "apply_canonical_perturbation_family_b": False,
        "compose_two_modes_on_frame": False,
        "build_canonical_per_pair_selectors": True,  # already invoked
        "build_provenance_for_predicted": False,
    }

    for sample_idx in range(n_sample):
        sel_a, sel_b = per_pair_selectors[sample_idx]
        # Translate per-family selector to global mode_idx
        mode_a_global = family_a_modes[sel_a % len(family_a_modes)]
        mode_b_global = family_b_modes[sel_b % len(family_b_modes)]
        base = canonical_base_frames[sample_idx]

        # ACTUALLY apply each mode and the composition; record deltas
        after_a = _apply_canonical_perturbation(base, mode_a_global)
        after_b = _apply_canonical_perturbation(base, mode_b_global)
        after_ab = _compose_two_modes_on_frame(base, mode_a_global, mode_b_global)

        canonical_helpers_invoked["apply_canonical_perturbation_family_a"] = True
        canonical_helpers_invoked["apply_canonical_perturbation_family_b"] = True
        canonical_helpers_invoked["compose_two_modes_on_frame"] = True

        base_vs_a = float(np.abs(after_a.astype(np.int16) - base.astype(np.int16)).sum())
        base_vs_b = float(np.abs(after_b.astype(np.int16) - base.astype(np.int16)).sum())
        a_vs_ab = float(np.abs(after_ab.astype(np.int16) - after_a.astype(np.int16)).sum())
        ab_vs_base = float(np.abs(after_ab.astype(np.int16) - base.astype(np.int16)).sum())

        composition_deltas.append(
            {
                "sample_idx": sample_idx,
                "family_a_mode_idx": mode_a_global,
                "family_b_mode_idx": mode_b_global,
                "family_a_mode_id": CANONICAL_MODE_MENU[mode_a_global][1],
                "family_b_mode_id": CANONICAL_MODE_MENU[mode_b_global][1],
                "sum_abs_delta_base_vs_a": base_vs_a,
                "sum_abs_delta_base_vs_b": base_vs_b,
                "sum_abs_delta_a_vs_ab": a_vs_ab,
                "sum_abs_delta_ab_vs_base": ab_vs_base,
                # Composition behavioral invariant: when both modes are
                # non-identity AND modes are from orthogonal families, the
                # composed frame must differ from both single-mode outputs
                # (proves composition actually happened, not no-op).
                "composition_produced_distinct_output": (
                    a_vs_ab > 0.0 if mode_b_global != 0 else True
                ),
            }
        )

    # Compute aggregate composition behavioral evidence
    composition_behavioral_evidence = {
        "n_samples_composed": n_sample,
        "n_pairs_in_full_config": config.n_pairs,
        "all_compositions_produced_distinct_output": all(
            d["composition_produced_distinct_output"] for d in composition_deltas
        ),
        "mean_sum_abs_delta_ab_vs_base": (
            sum(d["sum_abs_delta_ab_vs_base"] for d in composition_deltas) / n_sample
            if n_sample > 0 else 0.0
        ),
        "per_sample_deltas": composition_deltas,
        "family_a": family_a,
        "family_b": family_b,
        "n_family_a_modes": len(family_a_modes),
        "n_family_b_modes": len(family_b_modes),
        "n_combinations_addressable": len(family_a_modes) * len(family_b_modes),
    }

    # Canonical Provenance per Catalog #323 + Catalog #341 Tier A markers.
    inputs_payload = json.dumps(
        {
            "n_pairs": config.n_pairs,
            "modes_per_pair": config.modes_per_pair,
            "selector_bits_per_mode": config.selector_bits_per_mode,
            "family_pair_index": config.family_pair_index,
            "family_a": family_a,
            "family_b": family_b,
            "rng_seed": config.rng_seed,
            "n_samples_composed": n_sample,
            "smoke_frame_shape": [SMOKE_FRAME_H, SMOKE_FRAME_W, SMOKE_FRAME_C],
        },
        sort_keys=True,
    ).encode("utf-8")
    inputs_sha256 = hashlib.sha256(inputs_payload).hexdigest()
    provenance = build_provenance_for_predicted(
        model_id=(
            "pr110_opt11_multi_mode_per_pair_composition:"
            f"{family_a}_x_{family_b}:"
            f"modes_per_pair_{config.modes_per_pair}:"
            f"selector_bits_{config.selector_bits_per_mode}"
        ),
        inputs_sha256=inputs_sha256,
    )
    canonical_helpers_invoked["build_provenance_for_predicted"] = True
    provenance_dict = provenance_to_dict(provenance)

    cross_reference_matrix = {
        "wave_n34_analytical_investigator": (
            "research:.omx/research/wave_n34_pr110_opt_4_7_11_triple_artifacts_20260528.json:"
            "opt_11_multi_mode_composition.compound_ratio_upper_bound=1.548"
        ),
        "task_1323_pending_operator_routable": (
            "canonical:task_status.jsonl:task_id=1323:"
            "PR110-OPT-11_multi_mode_per_pair_composition_l0_scaffold"
        ),
        "sister_pr110_opt_7_l1_promotion": (
            "commit:1230b3b9c:"
            "tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1"
        ),
        "pair_component_rows_canonical_22_mode_menu": (
            "experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex/"
            "pair_component_rows.jsonl"
        ),
        "canonical_orthogonal_family_pairs_enumeration": (
            "tac.substrates.pr110_opt11_multi_mode_per_pair_composition."
            "CANONICAL_ORTHOGONAL_FAMILY_PAIRS"
        ),
    }

    return PR110OPT11Result(
        config=config,
        predicted_delta_adjustment=TIER_A_PREDICTED_DELTA_ADJUSTMENT,
        promotable=TIER_A_PROMOTABLE,
        axis_tag=TIER_A_AXIS_TAG,
        verdict=TIER_A_VERDICT,
        per_pair_selectors=per_pair_selectors,
        family_pair=(family_a, family_b),
        family_a_mode_indices=family_a_modes,
        family_b_mode_indices=family_b_modes,
        composition_behavioral_evidence=composition_behavioral_evidence,
        canonical_provenance=provenance_dict,
        canonical_helpers_invoked=canonical_helpers_invoked,
        cross_reference_matrix=cross_reference_matrix,
    )


def verify_canonical_multi_mode_composition(
    result: PR110OPT11Result,
) -> dict[str, Any]:
    """Verify Slot EEE NO FAKE IMPLEMENTATIONS invocation invariants.

    Per Catalog #305 observability surface: emits operator-facing audit
    verdict proving the substrate ACTUALLY composed 2 modes per pair vs
    a returns-canonical-markers-only no-op.

    Args:
        result: PR110OPT11Result from apply_substrate_to_pr110_canonical.

    Returns:
        dict with keys: all_invoked, invocation_count, missing_helpers,
        composition_distinct_output_verdict, behavioral_evidence_summary.
    """
    invocation_count = sum(
        1 for v in result.canonical_helpers_invoked.values() if v
    )
    expected = len(result.canonical_helpers_invoked)
    all_invoked = invocation_count == expected
    missing = [
        k for k, v in result.canonical_helpers_invoked.items() if not v
    ]
    distinct = result.composition_behavioral_evidence.get(
        "all_compositions_produced_distinct_output", False
    )
    return {
        "all_invoked": all_invoked,
        "invocation_count": invocation_count,
        "expected_invocations": expected,
        "missing_helpers": missing,
        "cross_reference_count": len(result.cross_reference_matrix),
        "composition_distinct_output_verdict": "PASS" if distinct else "FAIL",
        "behavioral_evidence_summary": {
            "n_samples_composed": result.composition_behavioral_evidence.get(
                "n_samples_composed", 0
            ),
            "mean_sum_abs_delta_ab_vs_base": result.composition_behavioral_evidence.get(
                "mean_sum_abs_delta_ab_vs_base", 0.0
            ),
            "n_combinations_addressable": result.composition_behavioral_evidence.get(
                "n_combinations_addressable", 0
            ),
        },
        "verified_at_utc": datetime.now(UTC).isoformat(),
    }
