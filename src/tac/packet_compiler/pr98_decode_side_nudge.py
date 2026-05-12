"""PR98 decode-side nudge — per-frame per-channel constant offsets.

Council deliberation 3 (OOO commit 328bf2f9, UNANIMOUS 10/10 vote for
``target_modes=["contest_one_video_replay"]``): PR98's hnerv_muon_finetuned
adds a SMALL decode-side postprocess that subtracts ``1.0`` (in the
``[0, 255]`` RGB float range, post-bicubic-upsample, pre-uint8-round) from
three specific ``(frame_in_pair, channel)`` slots:

- ``up[:, 0, 0].sub_(1.0)``   — frame 0 of pair, RED channel    (-1)
- ``up[:, 0, 2].sub_(1.0)``   — frame 0 of pair, BLUE channel   (-1)
- ``up[:, 1, 1].sub_(1.0)``   — frame 1 of pair, GREEN channel  (-1)

Source port location (READ-ONLY public intake clone, per Catalog #109):
    ``experiments/results/public_pr_intake_full/public_pr98_intake_20260505_auto/
      source/submissions/hnerv_muon_finetuned_from_pr95/inflate.py:55-57``

CRITICAL: target_modes=["contest_one_video_replay"] only
─────────────────────────────────────────────────────────
Per CLAUDE.md "Contest vs production target modes — non-negotiable" and the
unanimous council deliberation: a per-frame per-channel constant nudge is BY
DEFINITION a per-frame stream — there is NO rate-distortion theory that
generalizes a fixed constant offset across unseen videos. It can only be
admissible in:

    target_modes = ["contest_one_video_replay"]
    deployment_target = "t4_contest_runtime"

PR98's README ("decode-side postprocess tuned on the public evaluation path")
makes the overfit-to-scored-video nature EXPLICIT. The nudge is not portable
to ``contest_generalized`` or ``production_generalized`` and the council
DEFERRED any such claim indefinitely.

Byte-cost requirement (~28 KB demand from council)
──────────────────────────────────────────────────
Council deliberation 3 binding stipulation: BEFORE shipping a composition
that includes this primitive, the operator MUST evaluate the ~28 KB
byte-cost of the per-frame-channel-offset stream (3 nudges × N_pairs frames
× per-frame address bits). For PR98's 600 frame-pair archive this maps to:

    3 (slots) × 600 (pairs) × ~12 bits/address ≈ 21,600 bits ≈ 2.7 KB
    PLUS the constant payload (3 × float16 ≈ 6 bytes), PLUS the parser
    section grammar overhead.

PR98 sidesteps this with a HARDCODED inflate.py — the nudge IS the runtime
code, not data. A code-shipped nudge is admissible ONLY when the runtime tree
custody record explicitly lists the three constants AND target_modes excludes
``contest_generalized``.

Score-claim discipline
──────────────────────
- ``score_claim = false``
- ``promotion_eligible = false``
- ``ready_for_exact_eval_dispatch = false``

Per CLAUDE.md FORBIDDEN forbidden_score_claim_with_byte_change_unless_inflate_consumes:
the nudge changes inflated RGB bytes; any composition that includes it MUST
go through ``inflate.sh archive_dir output_dir file_list`` byte-parity proof
on the EXACT archive that will be submitted, and dual-eval CPU + CUDA on
1:1 contest-CI hardware per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

# PR98 nudge IS overfit-to-scored-video; only admissible in
# contest_one_video_replay mode per OOO council deliberation 3 unanimous 10/10.
TARGET_MODES: tuple[str, ...] = ("contest_one_video_replay",)
DEPLOYMENT_TARGET: str = "t4_contest_runtime"

#: The three (frame_in_pair_index, channel_index) slots PR98 nudges.
#: ``frame_in_pair_index`` is 0 for the first frame, 1 for the second frame
#: of a (frame1, frame2) pair. ``channel_index`` is 0=R, 1=G, 2=B in the
#: post-bicubic-upsample float buffer.
PR98_NUDGE_SLOTS: tuple[tuple[int, int, float], ...] = (
    (0, 0, -1.0),  # frame 0 of pair, RED   channel: -1
    (0, 2, -1.0),  # frame 0 of pair, BLUE  channel: -1
    (1, 1, -1.0),  # frame 1 of pair, GREEN channel: -1
)

#: Estimated byte cost of shipping the nudge as a parser-section sidecar.
#: Council deliberation 3 binding stipulation: evaluate this BEFORE shipping
#: any composition. See the module docstring for the derivation.
ESTIMATED_BYTE_COST_AS_SIDECAR_BYTES: int = 28_000

#: When the nudge is shipped as ``inflate.py`` code (PR98's choice), the
#: archive does not charge bytes for the offsets themselves; the cost is
#: entirely in the runtime tree size. PR98's ``inflate.py`` is ~70 LOC.
ESTIMATED_RUNTIME_CODE_OVERHEAD_LOC: int = 3

SCORE_CLAIM: bool = False
PROMOTION_ELIGIBLE: bool = False
READY_FOR_EXACT_EVAL_DISPATCH: bool = False

SOURCE_PORT_REFERENCE: str = (
    "experiments/results/public_pr_intake_full/public_pr98_intake_20260505_auto/"
    "source/submissions/hnerv_muon_finetuned_from_pr95/inflate.py:55-57"
)
COUNCIL_DECISION_REFERENCE: str = (
    "OOO commit 328bf2f9 deliberation 3 (UNANIMOUS 10/10 contest_one_video_replay)"
)


@dataclass(frozen=True)
class PR98DecodeSideNudge:
    """Typed record of the PR98 decode-side per-frame-channel nudge.

    Per council deliberation 3 binding stipulation, the operator must
    demand the ``~28 KB`` byte-cost evaluation BEFORE any composition
    that includes this primitive is shipped. Use
    :meth:`refuse_composition_without_byte_cost_evaluation` to gate.

    Attributes
    ----------
    slots
        Tuple of ``(frame_in_pair_index, channel_index, offset)`` triples.
        Defaults to the PR98 source values.
    target_modes
        Permitted target modes. PR98 nudge is overfit-to-scored-video so
        the only admissible value is ``("contest_one_video_replay",)``.
    deployment_target
        ``t4_contest_runtime``.
    """

    slots: tuple[tuple[int, int, float], ...] = field(
        default_factory=lambda: PR98_NUDGE_SLOTS
    )
    target_modes: tuple[str, ...] = field(default_factory=lambda: TARGET_MODES)
    deployment_target: str = DEPLOYMENT_TARGET

    def __post_init__(self) -> None:
        # Lock down the target_modes per council unanimous vote.
        if "contest_generalized" in self.target_modes:
            raise ValueError(
                "PR98 decode-side nudge is overfit-to-scored-video and MAY "
                "NOT be tagged contest_generalized; see "
                "src/tac/packet_compiler/pr98_decode_side_nudge.py docstring "
                "and OOO commit 328bf2f9 deliberation 3."
            )
        if "production_generalized" in self.target_modes:
            raise ValueError(
                "PR98 decode-side nudge is overfit-to-scored-video and MAY "
                "NOT be tagged production_generalized."
            )
        for slot in self.slots:
            if len(slot) != 3:
                raise ValueError(
                    f"PR98 nudge slot must be (frame_in_pair, channel, offset); got {slot!r}"
                )
            frame_in_pair, channel, _offset = slot
            if frame_in_pair not in (0, 1):
                raise ValueError(
                    f"frame_in_pair must be 0 or 1; got {frame_in_pair!r}"
                )
            if channel not in (0, 1, 2):
                raise ValueError(f"channel must be 0, 1, or 2; got {channel!r}")

    def estimated_byte_cost_as_sidecar(self, n_pairs: int) -> int:
        """Estimated bytes if the nudge were shipped as a parser-section sidecar.

        Derivation:
            ``n_slots × n_pairs × ceil(log2(n_pairs * 2 * 3)) / 8 + 6``

        Returns int rounded UP. Use to evaluate the council ``~28 KB``
        binding stipulation BEFORE shipping a composition.
        """
        if n_pairs < 1:
            raise ValueError("n_pairs must be >= 1")
        import math

        n_slots = len(self.slots)
        address_bits = math.ceil(math.log2(max(n_pairs * 2 * 3, 2)))
        bits = n_slots * n_pairs * address_bits
        # 3 × float16 (6 bytes) constants + ceil(bits/8) addresses.
        return 6 + math.ceil(bits / 8)

    def refuse_composition_without_byte_cost_evaluation(
        self,
        composition_byte_cost_evaluation: int | None,
    ) -> None:
        """Raise ValueError when a composition omits the byte-cost evaluation.

        Council deliberation 3 binding stipulation: NO composition that
        includes the nudge may ship without explicit byte-cost evaluation.
        """
        if composition_byte_cost_evaluation is None:
            raise ValueError(
                "OOO commit 328bf2f9 deliberation 3 binding stipulation: "
                "compositions including the PR98 decode-side nudge MUST "
                "evaluate the ~28 KB byte-cost BEFORE shipment. Pass an "
                "explicit int to composition_byte_cost_evaluation."
            )
        if composition_byte_cost_evaluation < 0:
            raise ValueError(
                "composition_byte_cost_evaluation must be >= 0"
            )


def apply_pr98_decode_side_nudge_inplace(
    upsampled_rgb_float: "Sequence",  # torch.Tensor in practice
    *,
    nudge: PR98DecodeSideNudge | None = None,
) -> "Sequence":
    """Apply the PR98 nudge in-place to a ``(B, 2, 3, H, W)`` float tensor.

    Mirrors PR98's inflate.py:55-57 byte-faithfully:

        ``up[:, 0, 0].sub_(1.0)``
        ``up[:, 0, 2].sub_(1.0)``
        ``up[:, 1, 1].sub_(1.0)``

    The input is expected to be a torch.Tensor of shape ``(B, 2, 3, H, W)``
    in the post-bicubic-upsample float buffer (range [0, 255], not [0, 1]).
    We type the parameter as ``Sequence`` only to keep the module import
    cheap; torch is imported lazily.

    Parameters
    ----------
    upsampled_rgb_float
        ``torch.Tensor`` of shape ``(B, 2, 3, H, W)``. Mutated in place.
    nudge
        Optional :class:`PR98DecodeSideNudge`. Defaults to PR98's exact slots.

    Returns
    -------
    The same tensor (mutated in place).
    """
    n = nudge or PR98DecodeSideNudge()
    if not hasattr(upsampled_rgb_float, "shape"):
        raise TypeError(
            "apply_pr98_decode_side_nudge_inplace expects a torch.Tensor with .shape; "
            f"got {type(upsampled_rgb_float).__name__}"
        )
    if len(upsampled_rgb_float.shape) != 5:
        raise ValueError(
            f"expected (B, 2, 3, H, W); got shape {tuple(upsampled_rgb_float.shape)}"
        )
    if upsampled_rgb_float.shape[1] != 2 or upsampled_rgb_float.shape[2] != 3:
        raise ValueError(
            f"expected (B, 2, 3, H, W); got shape {tuple(upsampled_rgb_float.shape)}"
        )
    for frame_in_pair, channel, offset in n.slots:
        upsampled_rgb_float[:, frame_in_pair, channel].add_(offset)
    return upsampled_rgb_float


__all__ = [
    "PR98DecodeSideNudge",
    "PR98_NUDGE_SLOTS",
    "TARGET_MODES",
    "DEPLOYMENT_TARGET",
    "ESTIMATED_BYTE_COST_AS_SIDECAR_BYTES",
    "ESTIMATED_RUNTIME_CODE_OVERHEAD_LOC",
    "SCORE_CLAIM",
    "PROMOTION_ELIGIBLE",
    "READY_FOR_EXACT_EVAL_DISPATCH",
    "SOURCE_PORT_REFERENCE",
    "COUNCIL_DECISION_REFERENCE",
    "apply_pr98_decode_side_nudge_inplace",
]
