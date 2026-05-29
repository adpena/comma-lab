# SPDX-License-Identifier: MIT
"""Cathedral consumer: per-pixel inverse-steganalysis on real video frames via MLX.

Per Catalog #335 canonical contract + operator binding 2026-05-29 5-invariant
standing directive + Slot EEE fake-implementation audit remediation:

This consumer auto-discovers the canonical shared helper
:mod:`tac.inverse_steganalysis_real_video_mlx` and surfaces per-candidate
Tier A canonical-routing-markers observability per Catalog #341 + #357.

The consumer is OBSERVABILITY-ONLY per Catalog #341 (Tier A) + Catalog #192
(macOS-CPU advisory NEVER promotable). It does NOT contribute non-zero
adjustments to the cathedral autopilot ranker.

Per CLAUDE.md "MLX portable-local-substrate authority" non-negotiable: this
consumer represents the canonical macOS-MLX research-signal surface for the
canonical 4 inverse-steganalysis paradigms (HILL, MiPOD, UNIWARD, HUGO) on
real ``upstream/videos/0.mkv`` decoded frames.

The canonical contract per Catalog #335 is satisfied:
- ``CONSUMER_NAME``
- ``CONSUMER_VERSION``
- ``CONSUMER_HOOK_NUMBERS`` (hook #4 + #5 + #6)
- ``update_from_anchor(anchor)`` (hook #5; canonical continual-learning
  posterior absorption — currently no-op pending empirical anchor formats)
- ``consume_candidate(candidate) -> dict`` (hook #4; canonical Tier A
  observability-only contribution per Catalog #341)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "per_pixel_inverse_steganalysis_real_video_mlx_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)


# Canonical per-paradigm mapping: candidate substring tokens → canonical cost
# function token. Used to disambiguate which canonical per-pixel cost function
# is most relevant for a given candidate's payload.
_CANONICAL_PARADIGM_TOKEN_MAP: dict[str, str] = {
    "hill_canonical_inverse_steganalysis": "hill_per_pixel_mlx",
    "li_wang_li_huang_2014": "hill_per_pixel_mlx",
    "mipod_canonical_inverse_steganalysis": "mipod_per_pixel_mlx",
    "sedighi_cogranne_fridrich_2016": "mipod_per_pixel_mlx",
    "uniward": "uniward_per_pixel_mlx",
    "holub_fridrich_denemark_2014": "uniward_per_pixel_mlx",
    "hugo_canonical_inverse_steganalysis": "hugo_per_pixel_mlx",
    "pevny_filler_bas_2010": "hugo_per_pixel_mlx",
}


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Currently no-op pending canonical empirical anchor formats for per-pixel
    inverse-steganalysis on real video frames. The canonical anchor format
    is operator-routable per the Slot EEE landing memo: paired-CUDA empirical
    anchor on a real PR110 archive that USES the per-pixel cost matrix would
    be the canonical anchor (predicted vs. empirical ΔS residual).
    """
    _ = anchor  # explicit acknowledgment per CLAUDE.md "Comment-only contracts forbidden"


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Tier A canonical-routing markers per Catalog #341 + #357 + #317:
    OBSERVABILITY-ONLY; ``predicted_delta_adjustment=0.0`` always.

    The consumer inspects the candidate's payload tokens to identify which
    canonical inverse-steganalysis paradigm is relevant and surfaces a
    rationale string. The ranker uses this as a probe-disambiguator hook
    (hook #6) to route the candidate to the correct canonical per-pixel
    MLX implementation if/when the operator queues a paired-CUDA empirical
    anchor.

    Parameters
    ----------
    candidate : Mapping[str, Any]
        Cathedral autopilot candidate row.

    Returns
    -------
    Mapping[str, Any]
        Canonical Tier A contribution per Catalog #341 + #323 + #356:
        - ``predicted_delta_adjustment`` (always 0.0)
        - ``promotable`` (always False)
        - ``score_claim`` (always False)
        - ``axis_tag`` ("[predicted]")
        - ``rationale``: paradigm-routing summary
        - ``matched_paradigm`` (or "none" if no token match)
        - ``canonical_helper_module``
    """
    payload_str = " ".join(
        str(v) for v in candidate.values() if isinstance(v, (str, int, float))
    ).lower()
    # Normalize hyphens / dots to underscores for robust paradigm-name matching
    # (e.g. "Li-Wang-Li-Huang 2014" -> "li_wang_li_huang_2014").
    payload_normalized = payload_str.replace("-", "_").replace(".", "_")
    # Also normalize whitespace runs into single underscores for tokens like
    # "Li Wang Li Huang 2014" -> "li_wang_li_huang_2014".
    payload_normalized = "_".join(payload_normalized.split())

    matched_paradigm: str = "none"
    for token, paradigm in _CANONICAL_PARADIGM_TOKEN_MAP.items():
        token_lc = token.lower()
        if token_lc in payload_normalized or token_lc in payload_str:
            matched_paradigm = paradigm
            break

    if matched_paradigm == "none":
        rationale = (
            "No canonical inverse-steganalysis paradigm token matched; "
            "this consumer is observability-only and no routing applies"
        )
    else:
        rationale = (
            f"Canonical paradigm {matched_paradigm} matched; route via "
            f"tac.inverse_steganalysis_real_video_mlx.compute_{matched_paradigm}_cost_mlx "
            f"for real-video per-pixel cost matrix; macOS-CPU advisory smoke "
            f"available via run_macos_cpu_advisory_smoke; paired-CUDA empirical "
            f"anchor REQUIRED before any score claim per Catalog #192 + #246"
        )

    return {
        "predicted_delta_adjustment": 0.0,
        "promotable": False,
        "score_claim": False,
        "axis_tag": "[predicted]",
        "rationale": rationale,
        "matched_paradigm": matched_paradigm,
        "canonical_helper_module": "tac.inverse_steganalysis_real_video_mlx",
        "confidence": 0.0,
    }
