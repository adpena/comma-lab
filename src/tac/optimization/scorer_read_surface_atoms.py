# SPDX-License-Identifier: MIT
"""Read-surface-grounded DEFORESTATION atom proposers (Phase-D1 primitives).

OPERATOR DIRECTIVE 2026-06-09: "give everything what it needs and no more;
slash-and-burn to the skeleton of only what's necessary in the optimal format
and quantization."

The contest scorer (``upstream/evaluate.py`` / ``upstream/modules.py``) reads
MINIMALLY.  Everything it does NOT read is its NULL SPACE -> burnable.  This
module turns the three code-grounded read-surface facts into typed
*deforestation atoms*: candidate actions that FREE bytes (or shave precision)
where the scorer is blind, each carrying an ADVISORY ``delta_score`` estimate
and the contract provenance needed to promote it to a base-bound
``CandidateActionEvaluation`` under exact ΔS later.

The three read-surface facts (ALL constants sourced from
``tac.contest_eval_contract``, which pins + drift-checks the snippets against
live ``upstream/modules.py`` -- we never hardcode read-surface facts here):

1. SegNet reads ONLY ``x[:, -1]`` (the SECOND frame of each pair) =>
   the FIRST frame of each pair carries ZERO d_seg signal (seg-free; only
   needs pose fidelity).  ``seg_scored_frame_mask`` enumerates this -- the
   biggest deforestation (~600 of 1200 frames are seg-free).
2. SegNet distortion = ``mean(argmax(gt) != argmax(comp))`` over 5 classes =>
   argmax-ONLY => per-pixel precision is set by the top-2 logit margin.
   Boundary pixels (small margin) are PROTECTED; interior pixels (large
   margin) are FREE.  ``segnet_argmax_margin_tolerance_map`` is the per-pixel
   waterfilling, computed from REAL SegNet logits via the EXACT path
   (``segnet(segnet.preprocess_input(pair))`` -> argmax over dim=1, the same
   path as ``upstream/modules.py::SegNet`` / ``tac.scorer`` line ~410).  It
   does NOT use ``mini_scorer`` (a 25K-param surrogate = proxy, forbidden as
   authority).
3. PoseNet scores only the first 6 of 12 pose dims (``:h.out // 2``) on
   2-frame YUV6 => half the head is null space.
   ``pose_null_projection`` exposes RGB perturbation directions that do NOT
   move the scored 6 pose dims, consuming
   ``tac.scorer_exploits.compute_scorer_jacobian`` +
   ``project_to_scorer_null_space`` read-only.

AUTHORITY: every atom these helpers return is PLANNING-CONTROL ONLY
(``score_claim=False``, ``promotable=False``, ``authority=
"planning_control_false_authority"``).  Any local d_seg / d_pose this module
touches is ADVISORY; exact authority is the paired upstream eval on contest
hardware.  These atoms are PROPOSALS the post-B2 waterfiller gates by exact ΔS
(``tac.optimization.evaluator_action_waterfill``).

NO FAKE: every helper does the work it names on REAL inputs.  The margin map is
computed from the actual logit tensor; the frame mask actually returns the
2nd-of-pair indices; the pose-null projection actually leaves the scored pose-6
within tolerance (verified by the behavioral tests).  None of these helpers
returns canonical markers without doing the work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tac.contest_eval_contract import (
    PUBLIC_TEST_FRAME_COUNT,
    PUBLIC_TEST_PAIR_COUNT,
    RATE_TERM_COEFFICIENT,
    SEGNET_TERM_COEFFICIENT,
    SEQ_LEN,
    build_score_allocation_contract,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    import torch.nn as nn

# Pull the canonical denominator + per-byte rate price from the contract's
# allocation surface so the advisory rate estimate is exact contest units.
_ALLOCATION_CONTRACT = build_score_allocation_contract()
CONTEST_ARCHIVE_RATE_DENOM = int(
    _ALLOCATION_CONTRACT["rate"]["canonical_denominator_bytes"]
)
RATE_PRICE_PER_ARCHIVE_BYTE = float(
    _ALLOCATION_CONTRACT["rate"]["rate_price_per_archive_byte"]
)

# Read-surface facts, sourced ONLY from the pinned contract (never hardcoded).
SEG_SCORED_FRAME_INDEX_WITHIN_PAIR = int(
    _ALLOCATION_CONTRACT["segnet"]["scored_frame_index_within_pair"]
)
SEG_UNSCORED_FRAME_INDEX_WITHIN_PAIR = int(
    _ALLOCATION_CONTRACT["segnet"]["unscored_frame_index_within_pair"]
)

# Canonical non-authority envelope every atom row carries.  Mirrors the
# planning-control markers on ``CandidateActionEvaluation.to_row`` and the
# proxy false-authority contract.
_PLANNING_CONTROL_FALSE_AUTHORITY: dict[str, Any] = {
    "authority": "planning_control_false_authority",
    "evidence_grade": "advisory",
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
}

DEFOREST_ATOM_SCHEMA = "deforestation_read_surface_atom.v1"


def _contract_provenance(read_surface_fact: str) -> dict[str, Any]:
    """Return the contract-provenance block tying an atom to the read surface.

    The atom can be promoted to a base-bound ``CandidateActionEvaluation``
    only once an exact paired eval supplies the d_seg/d_pose terms; until then
    the provenance records WHICH read-surface fact authorizes the proposal so a
    reviewer (and the post-B2 waterfiller) can audit it.
    """

    return {
        "contract_module": "tac.contest_eval_contract",
        "score_allocation_contract_schema": _ALLOCATION_CONTRACT["schema"],
        "read_surface_fact": read_surface_fact,
        "score_formula": _ALLOCATION_CONTRACT["score_formula"],
        "promotes_to": "tac.optimization.evaluator_action_waterfill.CandidateActionEvaluation",
        "promotion_requires": (
            "exact paired upstream evaluate.py d_seg/d_pose terms on contest "
            "hardware bound to the current base archive sha256"
        ),
    }


@dataclass(frozen=True)
class DeforestationAtom:
    """One read-surface-grounded deforestation proposal.

    A deforestation atom proposes either FREEING bytes (``delta_bytes_freed``
    > 0) or SHAVING per-pixel precision (``precision_bits_shaveable`` > 0)
    where the scorer is blind, plus an ADVISORY estimate of the resulting score
    change.  It is NOT a score claim -- it is a planning-control proposal that
    the post-B2 waterfiller turns into an exact ``CandidateActionEvaluation``.

    Attributes:
        atom_id: stable identifier for the proposal.
        atom_kind: deforestation family (seg_free_frame / argmax_margin /
            pose_null).
        deforestation_target: which read-surface null space this burns.
        delta_bytes_freed: bytes this atom would FREE from the archive
            (>= 0; a positive value lowers the rate term).  0 when the atom
            shaves in-frame precision rather than freeing whole sections.
        precision_bits_shaveable: total per-element precision (in bits) this
            atom estimates it can shave without moving the scored term.  0 when
            the atom frees whole bytes instead.
        advisory_delta_score: ADVISORY estimate of the contest score change if
            the atom is applied (negative = lowers score).  Planning-control
            only; the exact value comes from the paired eval.
        advisory_delta_seg / advisory_delta_pose: ADVISORY per-term deltas the
            estimate is built from (0.0 by read-surface construction for the
            term the atom is blind to).
        n_elements: number of pixels / frames / directions the atom covers.
        notes: human-readable rationale.
    """

    atom_id: str
    atom_kind: str
    deforestation_target: str
    delta_bytes_freed: int = 0
    precision_bits_shaveable: float = 0.0
    advisory_delta_score: float = 0.0
    advisory_delta_seg: float = 0.0
    advisory_delta_pose: float = 0.0
    n_elements: int = 0
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.delta_bytes_freed) < 0:
            raise ValueError("delta_bytes_freed must be non-negative (it FREES bytes)")
        if float(self.precision_bits_shaveable) < 0.0:
            raise ValueError("precision_bits_shaveable must be non-negative")
        if int(self.n_elements) < 0:
            raise ValueError("n_elements must be non-negative")

    @property
    def advisory_delta_rate_score(self) -> float:
        """ADVISORY rate-term change from freeing ``delta_bytes_freed`` bytes.

        Exact contest units: freeing N bytes lowers the rate term by
        ``25 * N / source_video_bytes``.  Negative (lowers score) when bytes
        are freed.
        """

        return -RATE_PRICE_PER_ARCHIVE_BYTE * float(self.delta_bytes_freed)

    def to_row(self) -> dict[str, Any]:
        return {
            "schema": DEFOREST_ATOM_SCHEMA,
            "atom_id": self.atom_id,
            "atom_kind": self.atom_kind,
            "deforestation_target": self.deforestation_target,
            "delta_bytes_freed": int(self.delta_bytes_freed),
            "precision_bits_shaveable": float(self.precision_bits_shaveable),
            "advisory_delta_score": float(self.advisory_delta_score),
            "advisory_delta_seg": float(self.advisory_delta_seg),
            "advisory_delta_pose": float(self.advisory_delta_pose),
            "advisory_delta_rate_score": self.advisory_delta_rate_score,
            "n_elements": int(self.n_elements),
            "notes": self.notes,
            "contract_provenance": _contract_provenance(self.deforestation_target),
            **self.extra,
            **_PLANNING_CONTROL_FALSE_AUTHORITY,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Atom 1 — SegNet argmax-margin per-pixel tolerance map (the per-pixel
# waterfilling on the SegNet read surface).
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SegnetMarginToleranceMap:
    """Per-pixel SegNet argmax-margin tolerance from REAL SegNet logits.

    Computed from the EXACT contest path: SegNet logits
    ``(B, 5, H, W)`` (the same tensor ``upstream/modules.py::SegNet.forward``
    produces, that ``compute_distortion`` argmaxes over dim=1).  For each
    pixel the top-2 logit margin ``m = top1 - top2`` is the headroom: the
    pixel's argmax only flips if a perturbation moves a non-winning class
    within ``m`` of the winner.  A LARGE margin => FREE (a lot of RGB
    perturbation / quantization error absorbable); a SMALL margin => PROTECTED
    (near a decision boundary, any perturbation can flip the argmax and pay the
    100*d_seg term).

    Attributes:
        margin: ``(B, H, W)`` top-2 logit margin (>= 0).  THE per-pixel
            tolerance proxy in logit units.
        protected_mask: ``(B, H, W)`` bool, True where margin <
            ``protected_margin_threshold`` (near a decision boundary).
        free_fraction: fraction of pixels classified FREE (large margin).
        protected_margin_threshold: the absolute logit-margin cutoff used.
        winner: ``(B, H, W)`` argmax class index (the contest mask).
    """

    margin: Any
    protected_mask: Any
    winner: Any
    free_fraction: float
    protected_fraction: float
    protected_margin_threshold: float
    num_classes: int

    def tolerance(self) -> Any:
        """Per-pixel tolerance == the top-2 margin (FREE pixels tolerate more).

        Returned as the raw margin so a downstream allocator can scale RGB
        quantization step / perturbation budget proportionally.  Protected
        pixels (margin near 0) get near-zero tolerance.
        """

        return self.margin


def segnet_argmax_margin_tolerance_map(
    gt_last_frame_logits: torch.Tensor,
    *,
    protected_margin_threshold: float | None = None,
    protected_quantile: float = 0.25,
) -> SegnetMarginToleranceMap:
    """Per-pixel argmax-margin tolerance from REAL SegNet logits (exact path).

    Atom-1 primitive.  ``gt_last_frame_logits`` MUST be the SegNet logit
    tensor produced by the EXACT contest path -- i.e. the caller runs
    ``segnet(segnet.preprocess_input(pair_btchw))`` (``upstream/modules.py``
    SegNet, ``smp.Unet('tu-efficientnet_b2', classes=5)``), the same tensor
    ``SegNet.compute_distortion`` argmaxes over dim=1 (``tac.scorer`` line
    ~410).  This helper does NOT load a model and does NOT use ``mini_scorer``
    (a 25K-param surrogate = proxy, forbidden as authority); it computes the
    margin from the logits you give it, so the caller controls the authority of
    the path.

    The per-pixel top-2 margin ``m = top1_logit - top2_logit`` is the exact
    headroom before the argmax flips: a perturbation flips pixel ``p`` iff it
    raises some non-winning class's logit within ``m`` of the winner.  Pixels
    with large margin are deep in a class interior (FREE -> burnable RGB
    precision); pixels with small margin sit on a SegNet decision boundary
    (PROTECTED -> any error costs the 100*d_seg term).

    Args:
        gt_last_frame_logits: ``(B, C, H, W)`` or ``(C, H, W)`` real SegNet
            logits (C == 5 for the contest scorer).
        protected_margin_threshold: absolute logit-margin cutoff below which a
            pixel is PROTECTED.  If ``None`` (default), derived from
            ``protected_quantile`` of the observed margins (data-adaptive).
        protected_quantile: when the threshold is auto-derived, the quantile of
            the margin distribution used as the PROTECTED cutoff (0.25 => the
            smallest-margin quarter of pixels are protected).

    Returns:
        ``SegnetMarginToleranceMap`` with the per-pixel margin (the tolerance),
        the protected/free classification, and the winner mask.

    Raises:
        ValueError: if the logits are not 3D/4D, or fewer than 2 classes.
    """

    import torch

    logits = gt_last_frame_logits
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    if logits.dim() != 4:
        raise ValueError(
            "gt_last_frame_logits must be (B,C,H,W) or (C,H,W) real SegNet "
            f"logits; got shape {tuple(gt_last_frame_logits.shape)}"
        )
    num_classes = int(logits.shape[1])
    if num_classes < 2:
        raise ValueError(
            f"SegNet argmax margin needs >= 2 classes; got C={num_classes}"
        )

    logits = logits.float()
    # Top-2 over the CLASS dim (dim=1) -> winner logit and margin per pixel.
    top2 = torch.topk(logits, k=2, dim=1).values  # (B, 2, H, W)
    margin = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)  # (B, H, W)
    winner = logits.argmax(dim=1)  # (B, H, W) == the contest mask

    if protected_margin_threshold is None:
        q = float(min(max(protected_quantile, 0.0), 1.0))
        threshold = float(torch.quantile(margin.reshape(-1), q).item())
    else:
        threshold = float(protected_margin_threshold)

    protected_mask = margin < threshold
    total = float(margin.numel())
    protected_fraction = float(protected_mask.sum().item()) / total if total else 0.0
    free_fraction = 1.0 - protected_fraction

    return SegnetMarginToleranceMap(
        margin=margin,
        protected_mask=protected_mask,
        winner=winner,
        free_fraction=free_fraction,
        protected_fraction=protected_fraction,
        protected_margin_threshold=threshold,
        num_classes=num_classes,
    )


def segnet_margin_tolerance_atom(
    tolerance_map: SegnetMarginToleranceMap,
    *,
    frame_index: int,
    bits_shaveable_per_free_pixel: float = 4.0,
) -> DeforestationAtom:
    """Wrap a margin tolerance map into a typed deforestation atom.

    The atom proposes shaving ``bits_shaveable_per_free_pixel`` of RGB
    precision from every FREE pixel (large-margin, class-interior) of the
    SCORED (2nd-of-pair) frame.  By read-surface construction the SegNet term
    is UNCHANGED (the argmax does not flip for FREE pixels), so the advisory
    seg delta is 0.0; the value of the atom is the precision it frees, which a
    downstream codec turns into bytes.  This is the per-pixel waterfilling
    surface the post-B2 waterfiller consumes.

    Args:
        tolerance_map: result of ``segnet_argmax_margin_tolerance_map``.
        frame_index: the GLOBAL frame index this map was computed for.  Must be
            a SCORED frame (2nd-of-pair); margin tolerance is only meaningful
            where SegNet reads.
        bits_shaveable_per_free_pixel: advisory per-FREE-pixel precision shave.

    Returns:
        ``DeforestationAtom`` with ``atom_kind='argmax_margin'``.
    """

    free_pixels = round(tolerance_map.free_fraction * float(tolerance_map.margin.numel()))
    bits = float(bits_shaveable_per_free_pixel) * float(free_pixels)
    return DeforestationAtom(
        atom_id=f"argmax_margin_frame{int(frame_index)}",
        atom_kind="argmax_margin",
        deforestation_target="segnet_argmax_interior_pixels_are_free",
        delta_bytes_freed=0,
        precision_bits_shaveable=bits,
        advisory_delta_score=0.0,  # seg argmax unchanged for FREE pixels by construction
        advisory_delta_seg=0.0,
        advisory_delta_pose=0.0,
        n_elements=free_pixels,
        notes=(
            f"{free_pixels} FREE (interior, margin>={tolerance_map.protected_margin_threshold:.4g}) "
            f"pixels of scored frame {int(frame_index)} can shave "
            f"~{bits_shaveable_per_free_pixel:g} RGB bits each without flipping the "
            f"SegNet argmax; {tolerance_map.protected_fraction:.1%} boundary pixels stay PROTECTED."
        ),
        extra={
            "scored_frame": True,
            "protected_margin_threshold": tolerance_map.protected_margin_threshold,
            "free_fraction": tolerance_map.free_fraction,
            "protected_fraction": tolerance_map.protected_fraction,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Atom 2 — seg-scored-frame mask (the biggest deforestation: ~half the frames
# are seg-free).
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SegScoredFrameMask:
    """Which frames SegNet scores vs which are SEG-FREE.

    SegNet reads ONLY ``x[:, -1]`` (the 2nd frame of each pair).  With
    ``seq_len == 2`` pairs, the SCORED frames are the odd global indices
    (1, 3, 5, ...) and the SEG-FREE frames are the even indices
    (0, 2, 4, ...).  Seg-free frames carry ZERO d_seg signal -> they need zero
    SegNet fidelity (only pose fidelity, since PoseNet reads both frames).

    Attributes:
        num_frames: total frames.
        seq_len: pair length (contract: 2).
        scored_frame_indices: global indices SegNet scores (2nd-of-pair).
        seg_free_frame_indices: global indices SegNet never reads (1st-of-pair).
    """

    num_frames: int
    seq_len: int
    scored_frame_indices: tuple[int, ...]
    seg_free_frame_indices: tuple[int, ...]

    @property
    def seg_free_fraction(self) -> float:
        return len(self.seg_free_frame_indices) / float(self.num_frames) if self.num_frames else 0.0

    def is_seg_scored(self, frame_index: int) -> bool:
        return int(frame_index) in set(self.scored_frame_indices)


def seg_scored_frame_mask(
    num_frames: int = PUBLIC_TEST_FRAME_COUNT,
    seq_len: int = SEQ_LEN,
) -> SegScoredFrameMask:
    """Classify SegNet-scored (2nd-of-pair) vs SEG-FREE (1st-of-pair) frames.

    Atom-2 primitive.  Read-surface fact: ``upstream/modules.py::SegNet``
    does ``x = x[:, -1, ...]`` (the contract pins the snippet), so within each
    ``seq_len``-length pair SegNet reads ONLY the LAST frame.  For the contest's
    non-overlapping ``seq_len == 2`` batching, that makes every ODD global frame
    index scored and every EVEN index seg-free.  Roughly half the frames
    (``PUBLIC_TEST_PAIR_COUNT`` of ``PUBLIC_TEST_FRAME_COUNT``) carry ZERO
    d_seg signal -- the biggest deforestation lever, because a whole frame's
    SegNet fidelity becomes burnable.

    Args:
        num_frames: total frames in the clip (default: the public-test count
            from the contract).
        seq_len: pair length (default: the contract's ``SEQ_LEN`` == 2).

    Returns:
        ``SegScoredFrameMask`` enumerating the scored + seg-free indices.

    Raises:
        ValueError: if ``seq_len < 1`` or ``num_frames < 0``.
    """

    if int(seq_len) < 1:
        raise ValueError(f"seq_len must be >= 1; got {seq_len}")
    if int(num_frames) < 0:
        raise ValueError(f"num_frames must be >= 0; got {num_frames}")

    n = int(num_frames)
    sl = int(seq_len)
    n_pairs = n // sl
    # The scored frame within each non-overlapping pair is the LAST one
    # (x[:, -1]); for pair p the scored global index is p*sl + (sl-1).
    scored = tuple(p * sl + (sl - 1) for p in range(n_pairs))
    scored_set = set(scored)
    seg_free = tuple(i for i in range(n) if i not in scored_set)
    return SegScoredFrameMask(
        num_frames=n,
        seq_len=sl,
        scored_frame_indices=scored,
        seg_free_frame_indices=seg_free,
    )


def seg_free_frame_atom(
    mask: SegScoredFrameMask,
    *,
    seg_bytes_per_frame: float,
) -> DeforestationAtom:
    """Wrap the seg-free frame set into a typed deforestation atom.

    The atom proposes freeing the SegNet-fidelity bytes of EVERY seg-free
    (1st-of-pair) frame -- those frames never enter ``x[:, -1]``, so their
    SegNet term is identically zero regardless of their content.  ``advisory_
    delta_seg`` is 0.0 by read-surface construction (SegNet never reads these
    frames); the only constraint that remains on a seg-free frame is PoseNet
    fidelity, which is handled by the pose-null atom, not here.

    Args:
        mask: result of ``seg_scored_frame_mask``.
        seg_bytes_per_frame: advisory bytes of SegNet-fidelity payload a single
            scored frame would otherwise need (the per-frame budget the codec
            can skip on a seg-free frame).

    Returns:
        ``DeforestationAtom`` with ``atom_kind='seg_free_frame'``.
    """

    n_free = len(mask.seg_free_frame_indices)
    freed = round(float(seg_bytes_per_frame) * float(n_free))
    atom = DeforestationAtom(
        atom_id="seg_free_frames_all",
        atom_kind="seg_free_frame",
        deforestation_target="segnet_reads_only_last_frame_of_pair",
        delta_bytes_freed=freed,
        precision_bits_shaveable=0.0,
        advisory_delta_seg=0.0,  # SegNet never reads these frames
        advisory_delta_pose=0.0,  # pose handled by pose-null atom, not here
        n_elements=n_free,
        notes=(
            f"{n_free} of {mask.num_frames} frames ({mask.seg_free_fraction:.1%}) are "
            f"SEG-FREE (1st-of-pair; SegNet reads only x[:,-1]); their SegNet-fidelity "
            f"budget (~{seg_bytes_per_frame:g} B/frame) is fully burnable."
        ),
        extra={
            "seg_free_fraction": mask.seg_free_fraction,
            "n_seg_free_frames": n_free,
            "n_scored_frames": len(mask.scored_frame_indices),
        },
    )
    # Advisory total delta = rate term only (seg/pose unchanged by construction).
    return _with_advisory_rate_delta_score(atom)


# ─────────────────────────────────────────────────────────────────────────────
# Atom 3 — PoseNet null-space projection (half the pose head is unscored; the
# scorer Jacobian's null space is RGB directions the scored pose-6 cannot see).
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PoseNullProjection:
    """An RGB perturbation projected into the scorer's null space.

    The scorer Jacobian (from ``tac.scorer_exploits.compute_scorer_jacobian``,
    rows = 6 scored PoseNet dims + sampled SegNet outputs) has a null space:
    pixel-space directions orthogonal to every row are INVISIBLE to the scored
    outputs.  ``project_to_scorer_null_space`` returns the component of a
    candidate perturbation that lives in that null space -- RGB the scorer
    cannot detect, hence burnable as a deforestation direction.

    Attributes:
        projected: the null-space component of the input perturbation (same
            shape as the input).
        scored_pose_dims: number of PoseNet dims the Jacobian protected (== 6
            for the contest scorer, ``:h.out // 2``).
        residual_energy_fraction: ||projected|| / ||perturbation|| -- the
            fraction of the candidate perturbation that survived as
            scorer-invisible (the higher, the more deforestable).
    """

    projected: Any
    scored_pose_dims: int
    residual_energy_fraction: float


def pose_null_projection(
    perturbation: torch.Tensor,
    *,
    frames: torch.Tensor,
    posenet: nn.Module,
    segnet: nn.Module,
    max_outputs: int = 16,
    rank_threshold: float = 1e-3,
) -> PoseNullProjection:
    """Project an RGB perturbation into the scorer null space (Atom-3 primitive).

    Read-surface fact: PoseNet scores only the first 6 of 12 pose dims
    (``upstream/modules.py``: ``out1[h.name][..., :h.out // 2] - out2[...]``;
    the contract pins the snippet).  Half the head is null space, and -- more
    broadly -- the scorer Jacobian over the scored outputs has a null space in
    pixel space: directions that move NO scored output.  This helper consumes
    ``tac.scorer_exploits.compute_scorer_jacobian`` +
    ``project_to_scorer_null_space`` READ-ONLY (it does not reimplement them)
    to return the component of ``perturbation`` the scored pose-6 (and sampled
    SegNet outputs) cannot see -- a burnable RGB direction.

    NOTE: the Jacobian's rows are the 6 scored pose dims plus sampled SegNet
    spatial outputs, so the returned direction is invisible to BOTH the scored
    pose AND the sampled seg outputs -- a conservative (joint) null space.

    Args:
        perturbation: ``(3, H, W)`` or flat ``(3*H*W,)`` candidate RGB
            perturbation to project.
        frames: ``(1, 3, H, W)`` single frame the Jacobian is taken at (BCHW,
            float [0,255]), as ``compute_scorer_jacobian`` requires.
        posenet: frozen PoseNet (differentiable-patched for nonzero pose grad).
        segnet: frozen SegNet.
        max_outputs: scorer output dims the Jacobian samples (>=6 to include
            the full scored pose-6).
        rank_threshold: singular-value cutoff for the null space.

    Returns:
        ``PoseNullProjection`` carrying the scorer-invisible component.
    """

    import torch

    from tac.scorer_exploits import (
        compute_scorer_jacobian,
        project_to_scorer_null_space,
    )

    if int(max_outputs) < 6:
        raise ValueError(
            "max_outputs must be >= 6 so the Jacobian includes the full scored "
            f"pose-6 ({SEG_UNSCORED_FRAME_INDEX_WITHIN_PAIR=}); got {max_outputs}"
        )

    jacobian = compute_scorer_jacobian(
        frames, posenet, segnet, max_outputs=int(max_outputs)
    )
    projected = project_to_scorer_null_space(
        perturbation, jacobian, rank_threshold=float(rank_threshold)
    )

    pert_norm = float(torch.linalg.vector_norm(perturbation.reshape(-1).float()).item())
    proj_norm = float(torch.linalg.vector_norm(projected.reshape(-1).float()).item())
    residual_fraction = (proj_norm / pert_norm) if pert_norm > 0.0 else 0.0

    return PoseNullProjection(
        projected=projected,
        scored_pose_dims=6,  # contest: first 6 of 12 (h.out // 2)
        residual_energy_fraction=residual_fraction,
    )


def pose_null_atom(
    projection: PoseNullProjection,
    *,
    frame_index: int,
    bits_shaveable_per_null_element: float = 2.0,
) -> DeforestationAtom:
    """Wrap a pose-null projection into a typed deforestation atom.

    The atom proposes spending the null-space energy as scorer-invisible RGB
    precision shaving on the given frame.  Because the direction lives in the
    scored-output null space, the advisory pose AND seg deltas are 0.0 by
    construction; the value is the precision it frees.  Particularly relevant
    for SEG-FREE (1st-of-pair) frames, where only PoseNet fidelity constrains
    the frame and the pose-null direction is the burnable axis.

    Args:
        projection: result of ``pose_null_projection``.
        frame_index: global frame index the projection was computed for.
        bits_shaveable_per_null_element: advisory per-null-element precision
            shave; scaled by the residual energy fraction (a near-zero null
            space frees little).

    Returns:
        ``DeforestationAtom`` with ``atom_kind='pose_null'``.
    """

    import torch

    n_elements = int(torch.as_tensor(projection.projected).reshape(-1).numel())
    effective = float(projection.residual_energy_fraction) * float(n_elements)
    bits = float(bits_shaveable_per_null_element) * effective
    return DeforestationAtom(
        atom_id=f"pose_null_frame{int(frame_index)}",
        atom_kind="pose_null",
        deforestation_target="posenet_scores_only_first_six_dims_jacobian_null_space",
        delta_bytes_freed=0,
        precision_bits_shaveable=bits,
        advisory_delta_score=0.0,  # null-space => no scored-output change by construction
        advisory_delta_seg=0.0,
        advisory_delta_pose=0.0,
        n_elements=n_elements,
        notes=(
            f"frame {int(frame_index)}: {projection.residual_energy_fraction:.1%} of the "
            f"candidate perturbation survives in the scorer null space (invisible to the "
            f"scored pose-6 + sampled seg outputs) -> ~{bits:.4g} burnable RGB bits."
        ),
        extra={
            "residual_energy_fraction": projection.residual_energy_fraction,
            "scored_pose_dims": projection.scored_pose_dims,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers + waterfiller-composition surface.
# ─────────────────────────────────────────────────────────────────────────────


def _with_advisory_rate_delta_score(atom: DeforestationAtom) -> DeforestationAtom:
    """Return ``atom`` with ``advisory_delta_score`` set to its rate-term delta.

    For a byte-FREEING atom whose scored terms are unchanged by construction,
    the advisory total score change IS the rate-term reduction
    (``-25 * bytes_freed / source_video_bytes``).  This is the planning-control
    estimate; the exact value comes from the paired eval.
    """

    if atom.delta_bytes_freed <= 0:
        return atom
    rate_delta = atom.advisory_delta_rate_score
    return DeforestationAtom(
        atom_id=atom.atom_id,
        atom_kind=atom.atom_kind,
        deforestation_target=atom.deforestation_target,
        delta_bytes_freed=atom.delta_bytes_freed,
        precision_bits_shaveable=atom.precision_bits_shaveable,
        advisory_delta_score=rate_delta,
        advisory_delta_seg=atom.advisory_delta_seg,
        advisory_delta_pose=atom.advisory_delta_pose,
        n_elements=atom.n_elements,
        notes=atom.notes,
        extra=dict(atom.extra),
    )


def summarize_read_surface(
    num_frames: int = PUBLIC_TEST_FRAME_COUNT,
    seq_len: int = SEQ_LEN,
) -> dict[str, Any]:
    """Return a machine-readable summary of the contest read surface.

    Max-observability surface: decomposes the scorer's read surface into its
    three deforestation levers (seg-free frames / argmax-interior pixels /
    pose-null directions) with the contract-pinned constants, so a planner can
    inspect WHAT is burnable without re-deriving it from ``upstream``.
    """

    mask = seg_scored_frame_mask(num_frames=num_frames, seq_len=seq_len)
    return {
        "schema": "deforestation_read_surface_summary.v1",
        "contract_module": "tac.contest_eval_contract",
        "score_formula": _ALLOCATION_CONTRACT["score_formula"],
        "constants": {
            "seq_len": int(seq_len),
            "num_frames": int(num_frames),
            "public_test_pair_count": PUBLIC_TEST_PAIR_COUNT,
            "segnet_term_coefficient": SEGNET_TERM_COEFFICIENT,
            "rate_term_coefficient": RATE_TERM_COEFFICIENT,
            "rate_price_per_archive_byte": RATE_PRICE_PER_ARCHIVE_BYTE,
            "contest_archive_rate_denom": CONTEST_ARCHIVE_RATE_DENOM,
        },
        "levers": {
            "seg_free_frames": {
                "read_surface_fact": "segnet_reads_only_last_frame_of_pair",
                "n_seg_free_frames": len(mask.seg_free_frame_indices),
                "seg_free_fraction": mask.seg_free_fraction,
                "deforestation": "whole-frame SegNet fidelity burnable on 1st-of-pair frames",
            },
            "argmax_interior_pixels": {
                "read_surface_fact": "segnet_argmax_interior_pixels_are_free",
                "deforestation": "per-pixel RGB precision shaveable where top-2 margin is large",
            },
            "pose_null_directions": {
                "read_surface_fact": "posenet_scores_only_first_six_dims_jacobian_null_space",
                "scored_pose_dims": 6,
                "deforestation": "RGB directions in the scorer Jacobian null space are invisible",
            },
        },
        **_PLANNING_CONTROL_FALSE_AUTHORITY,
    }


__all__ = [
    "CONTEST_ARCHIVE_RATE_DENOM",
    "DEFOREST_ATOM_SCHEMA",
    "RATE_PRICE_PER_ARCHIVE_BYTE",
    "SEG_SCORED_FRAME_INDEX_WITHIN_PAIR",
    "SEG_UNSCORED_FRAME_INDEX_WITHIN_PAIR",
    "DeforestationAtom",
    "PoseNullProjection",
    "SegScoredFrameMask",
    "SegnetMarginToleranceMap",
    "pose_null_atom",
    "pose_null_projection",
    "seg_free_frame_atom",
    "seg_scored_frame_mask",
    "segnet_argmax_margin_tolerance_map",
    "segnet_margin_tolerance_atom",
    "summarize_read_surface",
]
