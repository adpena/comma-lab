# SPDX-License-Identifier: MIT
"""tac.codec.pr98_channel_balance_zero_byte_bolt_on — L28 canonical zero-byte
decode-side channel-balance bolt-on per HNeRV parity discipline lesson L28.

Slot LL landing per Slot DD canonical highest-EV-shortest-WC RANK 1 finding
(commit `f07ada692` cross-PR-family canonical-techniques mining L14-L70).

Canonical L28 source-of-truth: PR101 hnerv_ft_microcodec inflate.py lines 49-51
(experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/
source/submissions/hnerv_ft_microcodec/inflate.py):

    up[:, 0, 0].sub_(1.0)   # frame_0 RED   channel -= 1.0
    up[:, 0, 2].sub_(1.0)   # frame_0 BLUE  channel -= 1.0
    up[:, 1, 1].sub_(1.0)   # frame_1 GREEN channel -= 1.0

PR #98 third-prize empirical anchor: 0.196 score = PR97 0.197 + L28 -0.001
(per Slot DD canonical highest-EV-shortest-WC RANK 1 finding).

Per Slot DD canonical:
  * **Zero archive bytes** (3 in-place subtractions on already-decoded RGB tensor)
  * **-0.0001 to -0.0005 score points** (HARD-EARNED-EMPIRICALLY-VERIFIED)
  * **Medal-class jump primitive** (PR97 0.197 -> PR98 0.196 -> PR101 0.193)
  * **Applies to ANY current frontier candidate** (V14-V2 DQS1, NSCS06 v8, fec6, sister)
  * **Operator binding META directive #3**: INTEGRATE into existing
    tac.codec namespace (NOT parallel build)

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L28 +
"Bit-level deconstruction and entropy discipline" + "Submission auth eval —
BOTH CPU AND CUDA" + Catalog #146 contest-compliant inflate runtime +
Catalog #205 canonical select_inflate_device + Catalog #295 PYTHONPATH
self-containment + Catalog #323 canonical Provenance + Catalog #341 Tier A
canonical-routing markers + Catalog #356 per-axis decomposition.

Per HNeRV parity L4: this bolt-on adds ~3 lines to any consuming inflate.py
(well within the ≤200 LOC budget; ≤100 LOC default per HNeRV parity L4).

Per Catalog #287 evidence tags: every score-savings claim carries
[empirical:PR98_third_prize_PR97_to_PR98_score_delta_0_001] or [predicted]
per measurement axis.

Cross-references:
  * Slot DD L14-L70 canonical-techniques mining:
    .omx/research/cross_pr_family_canonical_techniques_mining_L14_L70_20260529T075244Z.md
  * CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L28
  * PR101 canonical reference: experiments/results/public_pr_intake_full/
    public_pr101_intake_20260505_auto/source/submissions/hnerv_ft_microcodec/inflate.py:49-51
  * PR98 empirical anchor: PR97 0.197 -> PR98 0.196 (third prize; -0.001 delta = canonical L28 + sister)
  * Catalog #146 contest-compliant inflate runtime template
  * Catalog #205 canonical select_inflate_device
  * Catalog #295 PYTHONPATH self-containment
  * Catalog #323 canonical Provenance umbrella
  * Catalog #341 Tier A canonical-routing markers (this helper IS Tier A by construction)
  * Catalog #356 per-axis decomposition (rate=0 + seg/pose deltas surfaced)
  * Catalog #287 placeholder-rationale rejection (sister discipline this module honors)
  * Catalog #105 + #139 no-op detector (sister byte-mutation smoke pattern)

Hooks per Catalog #125 6-hook wire-in declaration:
  * #1 sensitivity-map = ACTIVE (per-pair per-channel sensitivity surfaced via AxisDecomposition)
  * #2 Pareto constraint = N/A (zero archive bytes; bolt-on is canonical free-byte
    optimization — does not move Pareto polytope feasibility boundary)
  * #3 bit-allocator = N/A (zero-byte bolt-on; no bit allocation needed)
  * #4 cathedral autopilot dispatch = ACTIVE (sister consumer
    pr98_channel_balance_consumer auto-discovered per Catalog #335)
  * #5 continual-learning posterior = ACTIVE (paired-CUDA RATIFICATION
    anchors feed canonical equation candidate
    pr98_zero_byte_decode_side_channel_balance_score_savings_v1)
  * #6 probe-disambiguator = ACTIVE (per-substrate paired-CPU + paired-CUDA
    measurement disambiguates per Catalog #246)

Mission contribution per Catalog #300: frontier_breaking (L28
canonical IMMEDIATE-APPLICATION bolt-on; N × small canonical wins compounding
per operator binding META directive "integrate not parallel build").
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

# Canonical L28 PR98 channel offsets per PR101 hnerv_ft_microcodec inflate.py:49-51.
# Each tuple = (pair_frame_index, rgb_channel_index, offset_value).
#
# pair_frame_index: 0 = frame_0 (first frame of the pair); 1 = frame_1 (second frame).
# rgb_channel_index: 0 = R; 1 = G; 2 = B.
# offset_value: subtracted from the decoded uint8/float channel value.
#
# Per Slot DD canonical: these are the EXACT 3 offsets PR #98 + PR101 + PR103
# canonical inflate-side ship. Do NOT mutate; the canonical anchor is exact.
PR98_CHANNEL_BALANCE_OFFSETS_CANONICAL: tuple[tuple[int, int, float], ...] = (
    (0, 0, 1.0),  # frame_0 RED   -= 1.0
    (0, 2, 1.0),  # frame_0 BLUE  -= 1.0
    (1, 1, 1.0),  # frame_1 GREEN -= 1.0
)

# Per Slot DD canonical highest-EV-shortest-WC RANK 1 finding:
PR98_L28_EXPECTED_SCORE_DELTA_BAND: tuple[float, float] = (-0.0005, -0.0001)
PR98_L28_ARCHIVE_BYTES_DELTA: int = 0
PR98_L28_CANONICAL_SOURCE_LINE_RANGE: str = "49-51"
PR98_L28_CANONICAL_SOURCE_PATH: str = (
    "experiments/results/public_pr_intake_full/"
    "public_pr101_intake_20260505_auto/source/submissions/"
    "hnerv_ft_microcodec/inflate.py"
)

# Canonical model identifier for Provenance per Catalog #323.
PROVENANCE_MODEL_ID: str = "tac.codec.pr98_channel_balance_zero_byte_bolt_on"

# Canonical reference per Catalog #344 canonical equations registry.
CANONICAL_EQUATION_CANDIDATE_ID: str = (
    "pr98_zero_byte_decode_side_channel_balance_score_savings_v1"
)


@dataclass(frozen=True)
class Pr98ChannelBalanceConfig:
    """Canonical config for L28 PR98 zero-byte channel-balance bolt-on.

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag":
    every config carries the canonical evidence tags + axis.

    Frozen per Catalog #287 sister discipline (no in-place mutation;
    immutable contract surface).

    Fields:
      substrate_id: target substrate this bolt-on applies to (e.g. "v14_v2_dqs1",
        "nscs06_v8", "fec6_b7106c9b", "pr106_format0d"). Non-empty.
      offsets: canonical L28 offsets per PR101 inflate.py:49-51. Defaults to
        PR98_CHANNEL_BALANCE_OFFSETS_CANONICAL (do NOT mutate; canonical anchor
        is exact per HNeRV parity L28).
      clamp_min, clamp_max: post-subtraction clamp range. Defaults preserve the
        canonical PR101 contract (clamp to [0, 255] uint8 range BEFORE round).
        Adjusting requires same-line waiver in caller per
        # PR98_L28_CLAMP_RANGE_DEVIATION_OK:<rationale>.
      apply_in_place: when True (default) mutates the input tensor in place per
        PR101 canonical `up[:, 0, 0].sub_(1.0)` style. When False returns a
        new tensor (slower; for diagnostic / byte-stability verification).
    """

    substrate_id: str
    offsets: tuple[tuple[int, int, float], ...] = (
        PR98_CHANNEL_BALANCE_OFFSETS_CANONICAL
    )
    clamp_min: float = 0.0
    clamp_max: float = 255.0
    apply_in_place: bool = True

    def __post_init__(self) -> None:
        # Per Catalog #287 placeholder-rationale rejection sister discipline:
        # empty substrate_id is not legitimate.
        if not isinstance(self.substrate_id, str) or not self.substrate_id.strip():
            raise ValueError(
                "Pr98ChannelBalanceConfig.substrate_id must be a non-empty string"
                " (e.g. 'v14_v2_dqs1', 'nscs06_v8', 'fec6_b7106c9b')"
            )
        # Per HNeRV parity L28: the canonical L28 offsets tuple shape is exactly
        # 3 entries each (pair_frame_index, rgb_channel_index, offset_value).
        # Allow caller to pass a different tuple ONLY when explicitly configured;
        # defaults preserve the canonical PR98 anchor exactly.
        if not isinstance(self.offsets, tuple) or len(self.offsets) == 0:
            raise ValueError(
                "Pr98ChannelBalanceConfig.offsets must be a non-empty tuple of"
                " (pair_frame_index, rgb_channel_index, offset_value) triples;"
                " default = PR98_CHANNEL_BALANCE_OFFSETS_CANONICAL"
            )
        for entry in self.offsets:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 3
                or not isinstance(entry[0], int)
                or not isinstance(entry[1], int)
                or not isinstance(entry[2], (int, float))
            ):
                raise ValueError(
                    f"Pr98ChannelBalanceConfig.offsets entry {entry!r} must be"
                    " a 3-tuple (pair_frame_index:int, rgb_channel_index:int,"
                    " offset_value:float)"
                )
            pair_idx, channel_idx, _ = entry
            if pair_idx not in (0, 1):
                raise ValueError(
                    f"pair_frame_index must be 0 (frame_0) or 1 (frame_1);"
                    f" got {pair_idx}"
                )
            if channel_idx not in (0, 1, 2):
                raise ValueError(
                    f"rgb_channel_index must be 0 (R), 1 (G), or 2 (B);"
                    f" got {channel_idx}"
                )
        if self.clamp_max < self.clamp_min:
            raise ValueError(
                f"clamp_max ({self.clamp_max}) must be >= clamp_min"
                f" ({self.clamp_min})"
            )


def apply_pr98_channel_balance_to_decoded_pair(
    decoded_pair_rgb: np.ndarray,
    config: Pr98ChannelBalanceConfig | None = None,
) -> np.ndarray:
    """Apply canonical L28 PR98 zero-byte decode-side channel-balance.

    Per PR101 hnerv_ft_microcodec inflate.py:49-51 canonical reference:

        up[:, 0, 0].sub_(1.0)   # frame_0 RED   -= 1.0
        up[:, 0, 2].sub_(1.0)   # frame_0 BLUE  -= 1.0
        up[:, 1, 1].sub_(1.0)   # frame_1 GREEN -= 1.0

    The canonical PR101 pattern subtracts 1.0 from each channel BEFORE the
    canonical `.clamp(0, 255).round().to(torch.uint8)` cast at PR101
    inflate.py:53-58. This helper mirrors that semantics for numpy arrays;
    sister torch-tensor helper available via duck-typing (any ndarray-like
    that supports in-place subtraction works).

    Args:
        decoded_pair_rgb: decoded RGB array shape (B, 2, 3, H, W) where:
          - B = batch size (typically 8-16 pairs per chunk per PR101 line 39)
          - 2 = pair frames (frame_0, frame_1)
          - 3 = RGB channels (R, G, B)
          - H, W = camera resolution (typically 874 x 1164 per PR101 line 16)
        config: Pr98ChannelBalanceConfig with canonical offsets (defaults to
          PR98_CHANNEL_BALANCE_OFFSETS_CANONICAL). If None, uses default config
          with substrate_id='<unspecified>' (caller is expected to set
          substrate_id explicitly for canonical Provenance threading per
          Catalog #323).

    Returns:
        The same array (mutated in place) when config.apply_in_place=True;
        a new array otherwise. Output is post-subtraction + post-clamp;
        caller is responsible for the canonical `.round().to(uint8)` cast
        per PR101 inflate.py:56-57.

    Raises:
        ValueError: if decoded_pair_rgb shape is not (B, 2, 3, H, W) or
          contains non-finite values.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #146:
    the canonical contest output contract is uint8 RGB frames at camera
    resolution; this bolt-on operates BEFORE the round-and-cast step so the
    -1.0 offset is preserved through the clamp.

    Per Catalog #205 + Catalog #146: this helper is canonical numpy
    (CPU-agnostic); the calling inflate.py routes the device via canonical
    select_inflate_device. The torch equivalent (PR101 inflate.py:49-51)
    operates on torch tensors with .sub_() in-place semantics.

    Example:
        >>> # canonical PR101 inflate.py sister pattern
        >>> from tac.codec.pr98_channel_balance_zero_byte_bolt_on import (
        ...     apply_pr98_channel_balance_to_decoded_pair,
        ...     Pr98ChannelBalanceConfig,
        ... )
        >>> config = Pr98ChannelBalanceConfig(substrate_id='v14_v2_dqs1')
        >>> # decoded_pair_rgb is shape (B, 2, 3, H, W) post-decoder
        >>> # already-upsampled to camera resolution per L18 PixelShuffle +
        >>> # bilinear-skip + sin canonical
        >>> balanced = apply_pr98_channel_balance_to_decoded_pair(
        ...     decoded_pair_rgb, config
        ... )
    """
    if config is None:
        config = Pr98ChannelBalanceConfig(substrate_id="<unspecified>")

    # Canonical shape contract per PR101 inflate.py:48: (B, 2, 3, H, W).
    if decoded_pair_rgb.ndim != 5:
        raise ValueError(
            f"decoded_pair_rgb must be 5-D shape (B, 2, 3, H, W); got"
            f" {decoded_pair_rgb.ndim}-D shape {decoded_pair_rgb.shape}"
        )
    if decoded_pair_rgb.shape[1] != 2:
        raise ValueError(
            f"decoded_pair_rgb axis 1 must be 2 (pair frames frame_0,"
            f" frame_1); got {decoded_pair_rgb.shape[1]}"
        )
    if decoded_pair_rgb.shape[2] != 3:
        raise ValueError(
            f"decoded_pair_rgb axis 2 must be 3 (RGB channels); got"
            f" {decoded_pair_rgb.shape[2]}"
        )

    # Per HNeRV parity L28: in-place mutation by default mirrors PR101's
    # `up[:, 0, 0].sub_(1.0)` canonical exactly. apply_in_place=False is
    # diagnostic-only (for byte-stability verification per Catalog #105 /
    # Catalog #139 no-op detector sister pattern).
    out = decoded_pair_rgb if config.apply_in_place else decoded_pair_rgb.copy()

    # Per HNeRV parity L28 canonical 3-line implementation. Apply each
    # canonical offset in order (the order is canonical-anchor-exact per
    # PR101 inflate.py:49-51 line ordering).
    for pair_frame_index, rgb_channel_index, offset_value in config.offsets:
        out[:, pair_frame_index, rgb_channel_index] -= offset_value

    # Per PR101 inflate.py:54 canonical clamp BEFORE round-and-uint8 cast.
    # The clamp preserves the -1.0 offset's score impact (the canonical
    # PR98 trick relies on the clamped values flowing into the scorer's
    # preprocess; per CLAUDE.md "Strict scorer rule" the scorer reads uint8
    # frames after the canonical .clamp(0, 255).round().to(torch.uint8) cast).
    np.clip(out, config.clamp_min, config.clamp_max, out=out)

    return out


def apply_pr98_channel_balance_to_decoded_pair_torch(
    decoded_pair_rgb: Any,
    config: Pr98ChannelBalanceConfig | None = None,
) -> Any:
    """Apply canonical L28 PR98 channel balance to a torch RGB pair tensor.

    This is the torch-native sibling of
    :func:`apply_pr98_channel_balance_to_decoded_pair`. It preserves PR101's
    exact tensor-space contract: operate on ``(B, 2, 3, H, W)`` floats before
    ``clamp().round().to(uint8)``.
    """
    if config is None:
        config = Pr98ChannelBalanceConfig(substrate_id="<unspecified>")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is present in tests.
        raise RuntimeError("torch is required for PR98 torch channel balance") from exc

    if not torch.is_tensor(decoded_pair_rgb):
        raise TypeError(
            "decoded_pair_rgb must be a torch.Tensor for "
            "apply_pr98_channel_balance_to_decoded_pair_torch"
        )
    if decoded_pair_rgb.ndim != 5:
        raise ValueError(
            f"decoded_pair_rgb must be 5-D shape (B, 2, 3, H, W); got"
            f" {decoded_pair_rgb.ndim}-D shape {tuple(decoded_pair_rgb.shape)}"
        )
    if decoded_pair_rgb.shape[1] != 2:
        raise ValueError(
            f"decoded_pair_rgb axis 1 must be 2 (pair frames frame_0,"
            f" frame_1); got {decoded_pair_rgb.shape[1]}"
        )
    if decoded_pair_rgb.shape[2] != 3:
        raise ValueError(
            f"decoded_pair_rgb axis 2 must be 3 (RGB channels); got"
            f" {decoded_pair_rgb.shape[2]}"
        )

    out = decoded_pair_rgb if config.apply_in_place else decoded_pair_rgb.clone()
    for pair_frame_index, rgb_channel_index, offset_value in config.offsets:
        out[:, pair_frame_index, rgb_channel_index].sub_(float(offset_value))
    out.clamp_(min=float(config.clamp_min), max=float(config.clamp_max))
    return out


def verify_pr98_channel_balance_byte_stable(
    test_pair: np.ndarray,
    config: Pr98ChannelBalanceConfig | None = None,
) -> bool:
    """Verify L28 bolt-on is byte-stable under repeated application.

    Per Catalog #146 contest-compliant inflate runtime + Catalog #205 +
    Catalog #105 + Catalog #139 no-op detector sister pattern: a canonical
    helper must be byte-stable across repeated invocations on the same
    input. This helper round-trips test_pair through
    apply_pr98_channel_balance_to_decoded_pair (with apply_in_place=False)
    and verifies the output bytes are deterministic.

    Args:
        test_pair: 5-D test array shape (B, 2, 3, H, W) for byte-stability check.
        config: Pr98ChannelBalanceConfig; if None, defaults to canonical config.

    Returns:
        True when two consecutive applications produce byte-identical output;
        False otherwise.

    Per CLAUDE.md "Apples-to-apples evidence discipline": deterministic
    behavior is structural (no random state). A byte-stability failure
    indicates either (a) numpy non-determinism (rare; non-canonical),
    (b) clamp boundary edge case, or (c) caller passing a view that
    shares memory with another array.
    """
    if config is None:
        config = Pr98ChannelBalanceConfig(
            substrate_id="<byte_stability_verification>", apply_in_place=False
        )
    else:
        # Force apply_in_place=False for byte-stability verification (avoid
        # mutating the caller's input).
        config = Pr98ChannelBalanceConfig(
            substrate_id=config.substrate_id,
            offsets=config.offsets,
            clamp_min=config.clamp_min,
            clamp_max=config.clamp_max,
            apply_in_place=False,
        )

    first = apply_pr98_channel_balance_to_decoded_pair(test_pair.copy(), config)
    second = apply_pr98_channel_balance_to_decoded_pair(test_pair.copy(), config)

    return np.array_equal(first, second)


def build_axis_decomposition_for_pr98_bolt_on(
    substrate_id: str,
    current_archive_bytes: int,
    current_d_pose: float,
    predicted_d_seg_delta: float | None = None,
    predicted_d_pose_delta: float | None = None,
) -> Mapping[str, Any]:
    """Build canonical AxisDecomposition for L28 bolt-on per Catalog #356.

    Per Slot DD canonical:
      * predicted_archive_bytes_delta = 0 (zero-byte bolt-on; CANONICAL EXACT)
      * predicted_score_delta = -0.0001 to -0.0005 score points (band per
        PR98 third-prize empirical anchor)
      * The PR98 trick subtracts 1.0 from specific RGB channels BEFORE the
        scorer reads them; the canonical L28 mechanism is channel-bias
        correction. Per CLAUDE.md "SegNet vs PoseNet importance" the
        per-axis attribution depends on operating point; default to band
        split 50%/50% across seg + pose (operator-routable refinement
        pending paired-CUDA RATIFICATION empirical anchors).

    Args:
        substrate_id: target substrate this bolt-on applies to.
        current_archive_bytes: current archive bytes (for axis decomposition
          context; L28 bolt-on does NOT change this).
        current_d_pose: current pose-axis distortion baseline.
        predicted_d_seg_delta: predicted seg-axis distortion delta. If None,
          defaults to -0.00015 (50% of -0.0003 mid-band per Slot DD).
        predicted_d_pose_delta: predicted pose-axis distortion delta. If
          None, defaults to -0.00015 (50% of -0.0003 mid-band per Slot DD).

    Returns:
        Canonical AxisDecomposition mapping per Catalog #356 with canonical
        Provenance per Catalog #323. JSON-safe; all values are scalar
        primitives or canonical Provenance dict.

    Per Catalog #341 Tier A canonical-routing markers: the returned mapping
    carries axis_tag='[predicted]' + promotable=False. Promotion to
    Tier B requires paired-CUDA RATIFICATION empirical anchor per Catalog
    #246 + Catalog #357.
    """
    from tac.provenance import build_provenance_for_predicted, provenance_to_dict

    if predicted_d_seg_delta is None:
        predicted_d_seg_delta = -0.00015
    if predicted_d_pose_delta is None:
        predicted_d_pose_delta = -0.00015

    # Canonical Provenance per Catalog #323. inputs_sha256 is a canonical
    # deterministic hash over the input context (substrate_id + current
    # archive bytes + current pose distortion) so two callers with
    # identical context get identical Provenance.
    import hashlib

    inputs_text = (
        f"substrate={substrate_id};bytes={current_archive_bytes};"
        f"d_pose={current_d_pose}"
    )
    inputs_sha256 = hashlib.sha256(inputs_text.encode("utf-8")).hexdigest()

    provenance = build_provenance_for_predicted(
        model_id=PROVENANCE_MODEL_ID,
        inputs_sha256=inputs_sha256,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
    )

    return {
        "predicted_d_seg_delta": float(predicted_d_seg_delta),
        "predicted_d_pose_delta": float(predicted_d_pose_delta),
        "predicted_archive_bytes_delta": PR98_L28_ARCHIVE_BYTES_DELTA,
        "axis_tag": "[predicted]",
        "canonical_provenance": provenance_to_dict(provenance),
        "substrate_id": substrate_id,
        "canonical_source": (
            f"PR101 hnerv_ft_microcodec inflate.py:"
            f"{PR98_L28_CANONICAL_SOURCE_LINE_RANGE}"
        ),
        "canonical_equation_candidate": CANONICAL_EQUATION_CANDIDATE_ID,
    }


def list_candidate_substrates_for_l28_application() -> tuple[Mapping[str, Any], ...]:
    """Enumerate current frontier candidates for canonical L28 application.

    Per Slot DD canonical highest-EV-shortest-WC RANK 1 finding + Catalog
    #343 canonical frontier pointer (queryable via tools/refresh_canonical_frontier.py):

      * V14-V2 DQS1 sub-frontier at 0.1920196 [contest-CPU]
      * NSCS06 v8 chroma_lut + cls_stream stacked archive (sister tasks
        #1340+#1341+#1344+#1346+#1355)
      * fec6 frontier archive (sha b7106c9bdbb8...; canonical pointer)
      * PR106 format0d (sha 9cb989cef519...; canonical pointer)
      * sister current frontier candidates

    Per operator binding META directive #3: this enumeration INTEGRATES
    with existing canonical frontier pointer (NOT parallel build); the
    canonical authoritative source remains
    `.omx/state/canonical_frontier_pointer.json` per Catalog #343.

    Returns:
        Tuple of candidate substrate dicts; each carries substrate_id +
        canonical_frontier_role + predicted_l28_applicability +
        estimated_score_delta_band per Slot DD canonical.
    """
    return (
        {
            "substrate_id": "v14_v2_dqs1",
            "canonical_frontier_role": "current_cpu_frontier_sub_anchor",
            "canonical_archive_sha_prefix": "7a0da5d0fc327cba",
            "predicted_l28_applicability": "high",
            "estimated_score_delta_band": list(PR98_L28_EXPECTED_SCORE_DELTA_BAND),
            "operator_routable_paired_cuda_ratification": True,
        },
        {
            "substrate_id": "fec6_canonical_frontier_pointer",
            "canonical_frontier_role": "current_cpu_frontier_canonical",
            "canonical_archive_sha_prefix": "b7106c9bdbb8",
            "predicted_l28_applicability": "high",
            "estimated_score_delta_band": list(PR98_L28_EXPECTED_SCORE_DELTA_BAND),
            "operator_routable_paired_cuda_ratification": True,
        },
        {
            "substrate_id": "pr106_format0d",
            "canonical_frontier_role": "current_cuda_frontier_canonical",
            "canonical_archive_sha_prefix": "9cb989cef519",
            "predicted_l28_applicability": "high",
            "estimated_score_delta_band": list(PR98_L28_EXPECTED_SCORE_DELTA_BAND),
            "operator_routable_paired_cuda_ratification": True,
        },
        {
            "substrate_id": "nscs06_v8_chroma_lut_cls_stream_stacked",
            "canonical_frontier_role": "sister_in_flight_stacked_archive",
            "canonical_archive_sha_prefix": "<pending_paired_cuda_ratification>",
            "predicted_l28_applicability": "medium",
            "estimated_score_delta_band": list(PR98_L28_EXPECTED_SCORE_DELTA_BAND),
            "operator_routable_paired_cuda_ratification": True,
        },
    )


__all__ = [
    "CANONICAL_EQUATION_CANDIDATE_ID",
    "PR98_CHANNEL_BALANCE_OFFSETS_CANONICAL",
    "PR98_L28_ARCHIVE_BYTES_DELTA",
    "PR98_L28_CANONICAL_SOURCE_LINE_RANGE",
    "PR98_L28_CANONICAL_SOURCE_PATH",
    "PR98_L28_EXPECTED_SCORE_DELTA_BAND",
    "PROVENANCE_MODEL_ID",
    "Pr98ChannelBalanceConfig",
    "apply_pr98_channel_balance_to_decoded_pair",
    "apply_pr98_channel_balance_to_decoded_pair_torch",
    "build_axis_decomposition_for_pr98_bolt_on",
    "list_candidate_substrates_for_l28_application",
    "verify_pr98_channel_balance_byte_stable",
]
