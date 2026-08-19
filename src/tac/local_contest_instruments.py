# SPDX-License-Identifier: MIT
"""The local contest instruments: score either axis, on the lineage that actually ships.

WHY THIS MODULE EXISTS
----------------------
Two arms built the same instrument on the same day and neither could reuse the other.
``ddm_up2`` (2026-08-19) built a local PoseNet gate that reproduces the T4 pose leg to
**0.99993x**; ``ddm_jg1`` (2026-08-19) built a local SegNet gate that reproduces the T4
seg leg to **0.99995x**.  Both are arm-local scripts under ``experiments/``.  Every
future arm that wants a local verdict must either import a sibling arm's script or
write a third copy.

Worse, both re-derived a lineage gate that ALREADY EXISTED.  :mod:`tac.gt_lineage`
landed 2026-08-16 (``ddm_gl1``) with a **content-addressed** registry, after ``ddm_gl1``
measured **six distinct files named** ``gt_argmax_n600.npy``.  ``ddm_up2`` nevertheless
resolves lineage by *filename substring*::

    lineage = LINEAGE_DALI if "dali" in path.name.lower() else "unknown_pt"

That is precisely the laundering ``gl1`` was built to refuse: one verified file's
reputation extended to five unverified ones by name.  This module therefore does not
re-implement the lineage gate -- it **composes** :mod:`tac.gt_lineage` and adds the
layer that was genuinely missing.

WHAT WAS GENUINELY MISSING
--------------------------
:mod:`tac.gt_lineage` answers *"what lineage is this artifact?"*.  :mod:`tac.contest_score`
answers *"what is the score arithmetic?"*.  Neither answers the question an instrument
actually asks:

    **"I am about to quote a number on axis A -- which lineage must I have read, and may
    I quote this number as an absolute at all?"**

That binding is DERIVED from upstream, not chosen::

    upstream/evaluate.py:31-42   device.type == "cuda" -> DefaultDatasetClass = DaliVideoDataset
                                 else                  -> DefaultDatasetClass = AVVideoDataset
    upstream/frame_utils.py:113  DaliVideoDataset asserts device.type == 'cuda'
    upstream/frame_utils.py:188  AVVideoDataset  asserts device.type != 'cuda'

The binding is bijective and assert-enforced on both sides.  A ``[contest-CUDA]`` row is
scored against DALI-lineage GT; a ``[contest-CPU]`` row against PyAV.  **They are
different objectives, not hardware drift.**  ``ddm_pi2`` measured the gap on ``0.mkv``,
n600: seg **x1.4425** multiplicative, pose **+1.4061e-04 ADDITIVE**.  The pose form is
the dangerous one -- an additive floor is 99.996% of a good carrier's total, so a PyAV
pose *absolute* is very nearly a constant that does not respond to the carrier at all.
Two paid rows (``ddm_ps1u`` r2 at +1.686e-02 S, ``ddm_t1h`` at +0.012557 S) were bought
minimising against the wrong lineage.

So this module refuses PyAV pose absolutes **by default**, and makes quoting one require
an explicit opt-in that names the PyAV objective as the thing being optimised.

WHAT THIS MODULE IS NOT
-----------------------
It is **not** a score.  Only ``upstream/evaluate.py`` on contest hardware is a score.
Every receipt this module emits carries ``score_claim=False`` and ``promotable=False``,
and the local legs are ``[macOS-CPU advisory]`` even when they read the DALI lineage --
the lineage makes them *comparable in kind*, not authoritative.

Canonical-vs-unique decisions are recorded in
``.omx/research/ddm_cw1_win_family_canonicalization_20260819.md``.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tac import contest_score, gt_lineage

__all__ = [
    "ADVISORY_FLOOR_CLIP",
    "ADVISORY_POSE_ADDITIVE_FLOOR",
    "ADVISORY_SEG_MULTIPLICATIVE_FACTOR",
    "AXIS_CONTEST_CPU",
    "AXIS_CONTEST_CUDA",
    "AXIS_GT_LINEAGE",
    "AXIS_MACOS_CPU_ADVISORY",
    "InstrumentReceipt",
    "InstrumentRefusal",
    "PoseAbsoluteRefused",
    "assert_axis_lineage",
    "assert_pose_absolute_quotable",
    "contest_score_from_legs",
    "d_pose_per_pair",
    "d_seg_per_pair",
    "known_axes",
    "pose_leg",
    "pose_report_bound",
    "rate_leg",
    "required_lineage_for_axis",
    "resolvable_d_pose_floor",
    "seg_leg",
    "select_pairs",
]


# ---------------------------------------------------------------------------
# The axes, and the lineage each one is actually scored against.
# ---------------------------------------------------------------------------

#: The contest CUDA axis.  Scored against DALI/nvdec GT (evaluate.py:31-42).
AXIS_CONTEST_CUDA = "contest-CUDA"
#: The contest CPU axis (the public leaderboard axis).  Scored against PyAV GT.
AXIS_CONTEST_CPU = "contest-CPU"
#: Our local advisory axis.  Runs on CPU torch, so it is the PyAV *objective* by
#: default -- but it may be pointed at the DALI GT tables to track the CUDA row.
AXIS_MACOS_CPU_ADVISORY = "macOS-CPU advisory"

#: Axis -> required GT decode lineage.  DERIVED from the upstream dispatch above;
#: this table is a transcription of an assert-enforced bijection, not a policy choice.
AXIS_GT_LINEAGE: dict[str, str] = {
    AXIS_CONTEST_CUDA: gt_lineage.DALI_NVDEC,
    AXIS_CONTEST_CPU: gt_lineage.PYAV_YUV420_TO_RGB,
    AXIS_MACOS_CPU_ADVISORY: gt_lineage.PYAV_YUV420_TO_RGB,
}

#: Axes whose *absolute* pose number is quotable without an explicit opt-in.
#: Only the DALI-lineage axis qualifies; see :func:`assert_pose_absolute_quotable`.
_POSE_ABSOLUTE_SAFE_AXES = frozenset({AXIS_CONTEST_CUDA})

# --- the measured lineage gap (ddm_pi2, n600, 0.mkv) -----------------------
#: PyAV-vs-DALI pose gap, **ADDITIVE** in d_pose.  MEASURED ``ddm_pi2`` 2026-08-16 as the
#: MSE between the two GT pose tables; float32 reference 0.00014061325055081397, float64
#: re-reduction 0.00014061324889363773 (``ddm_pr130_reproduce_20260809/FX4_GT_LINEAGE``).
ADVISORY_POSE_ADDITIVE_FLOOR = 1.4061e-04
#: PyAV-vs-DALI seg gap, **MULTIPLICATIVE** in d_seg.  MEASURED ``ddm_pi2``.
ADVISORY_SEG_MULTIPLICATIVE_FACTOR = 1.4425
#: Both constants are properties of ONE CLIP and TWO DECODERS (``ddm_pi2`` §"0.mkv-specific").
#: They are not a general CPU/CUDA law and must be re-measured for any other source video.
ADVISORY_FLOOR_CLIP = "upstream/videos/0.mkv"

#: ``upstream/evaluate.py:95`` prints d_pose at 8 decimals; half-ULP of that report.
_REPORT_HALF_ULP = 0.5e-8

#: Total pairs in the scored field.
N_PAIRS_TOTAL = 600


class InstrumentRefusal(RuntimeError):
    """A local-instrument precondition failed.  Always fail closed, never approximate."""


class PoseAbsoluteRefused(InstrumentRefusal):
    """A pose ABSOLUTE was requested on an axis whose pose absolute is not meaningful."""


def known_axes() -> tuple[str, ...]:
    """The axes this module knows how to bind to a GT lineage."""
    return tuple(sorted(AXIS_GT_LINEAGE))


def required_lineage_for_axis(axis: str) -> str:
    """Return the GT decode lineage ``axis`` is actually scored against, or refuse.

    Raises:
        InstrumentRefusal: the axis is not one this module can bind.
    """
    try:
        return AXIS_GT_LINEAGE[axis]
    except KeyError as error:
        raise InstrumentRefusal(
            f"unknown score axis {axis!r}; known axes: {list(known_axes())}"
        ) from error


def assert_axis_lineage(
    path: str | Path,
    *,
    axis: str,
    instrument: str = "<unnamed instrument>",
    registry: dict[str, gt_lineage.GtArtifactLineage] | None = None,
) -> gt_lineage.GtArtifactLineage:
    """Assert a GT artifact carries the lineage ``axis`` is scored against.

    This is the one call every local instrument should make before reading a GT cache.
    It resolves lineage **by content** (``tac.gt_lineage``), never by filename -- the
    filename route is the laundering bug ``ddm_gl1`` measured across six distinct files
    sharing the name ``gt_argmax_n600.npy``.

    Args:
        path: the GT artifact about to be read.
        axis: the axis whose number the caller intends to quote.
        instrument: caller name, quoted in the refusal.
        registry: optional pre-loaded registry (tests inject a fixture here).

    Returns:
        The registry entry, so callers can record ``sha256`` in their receipt.

    Raises:
        InstrumentRefusal: unknown axis.
        gt_lineage.GtLineageUnknown / GtLineageMismatch: propagated from the guard.
    """
    required = required_lineage_for_axis(axis)
    return gt_lineage.assert_gt_lineage(
        path, required=required, instrument=f"{instrument} [axis={axis}]", registry=registry
    )


def assert_pose_absolute_quotable(
    axis: str, *, allow_pyav_objective: bool = False, instrument: str = "<unnamed instrument>"
) -> None:
    """Refuse a pose ABSOLUTE on an axis where the absolute is dominated by the floor.

    On any PyAV-lineage axis the measured d_pose carries a fixed **additive** floor of
    ``ADVISORY_POSE_ADDITIVE_FLOOR`` (1.4061e-04).  For a good carrier -- the live pointer
    sits at d_pose 7.77e-06 on the CUDA axis -- that floor is 94.8% of the PyAV total, so
    the PyAV absolute barely moves when the carrier improves and a solve that minimises it
    is minimising a near-constant.  ``ddm_ps1u`` r2 and ``ddm_t1h`` both bought paid
    refusals that way.

    DELTAS on the PyAV axis remain meaningful (the floor is additive, so it cancels in a
    difference taken on the same lineage).  Only the ABSOLUTE is refused.

    Args:
        axis: the axis whose pose absolute the caller wants to quote.
        allow_pyav_objective: set True only when the PyAV objective *is* the thing being
            optimised (e.g. an explicit contest-CPU leaderboard row).  Naming it is the
            point: the opt-in makes the different-objective choice visible in the code.
        instrument: caller name, quoted in the refusal.

    Raises:
        PoseAbsoluteRefused: the axis is PyAV-lineage and the opt-in was not given.
        InstrumentRefusal: unknown axis.
    """
    lineage = required_lineage_for_axis(axis)
    if axis in _POSE_ABSOLUTE_SAFE_AXES:
        return
    if allow_pyav_objective:
        return
    raise PoseAbsoluteRefused(
        f"{instrument}: refusing to quote a pose ABSOLUTE on axis {axis!r} "
        f"(lineage {lineage}). That axis carries a fixed additive d_pose floor of "
        f"{ADVISORY_POSE_ADDITIVE_FLOOR:.4e} (ddm_pi2, n600, {ADVISORY_FLOOR_CLIP}), which is "
        "most of the total for any good carrier, so the absolute is nearly floor-constant. "
        "Two paid rows (ddm_ps1u r2 +1.686e-02 S, ddm_t1h +0.012557 S) were bought this way. "
        "Quote a DELTA on this axis (the additive floor cancels), or pass "
        "allow_pyav_objective=True if the PyAV objective is genuinely what you are optimising."
    )


# ---------------------------------------------------------------------------
# Score arithmetic.  ADOPTED from tac.contest_score -- never re-derived here.
# ---------------------------------------------------------------------------


def seg_leg(d_seg: float) -> float:
    """The SegNet leg of the score.  Delegates to :func:`tac.contest_score.seg_term`."""
    return contest_score.seg_term(d_seg)


def pose_leg(d_pose: float) -> float:
    """The PoseNet leg of the score.  Delegates to :func:`tac.contest_score.pose_term`."""
    return contest_score.pose_term(d_pose)


def rate_leg(archive_bytes: int | float) -> float:
    """The rate leg of the score.  Delegates to :func:`tac.contest_score.rate_term`."""
    return contest_score.rate_term(archive_bytes)


def contest_score_from_legs(d_seg: float, d_pose: float, archive_bytes: int | float) -> float:
    """The full score, byte-identical to ``upstream/evaluate.py:92``."""
    return contest_score.compute_contest_score(d_seg, d_pose, archive_bytes)


def pose_report_bound(d_pose: float) -> float:
    """Half-ULP bound of the 8dp d_pose report, in SCORE units.

    The pose leg's derivative in d_pose GROWS as d_pose falls, so the reporting
    bound on a *good* carrier is larger than on a bad one.  Any claim whose magnitude is
    below this bound is not resolvable from the printed report at all.
    """
    contest_score._require_finite_nonneg("d_pose", d_pose)
    if d_pose <= 0.0:
        return pose_leg(_REPORT_HALF_ULP)
    return 5.0 / math.sqrt(10.0 * float(d_pose)) * _REPORT_HALF_ULP


def resolvable_d_pose_floor() -> float:
    """Below this d_pose the 8dp report prints ``0.00000000`` and resolves nothing."""
    return _REPORT_HALF_ULP


# ---------------------------------------------------------------------------
# Population selection.  Seeded RANDOM, never a prefix.
# ---------------------------------------------------------------------------


def select_pairs(pairs: int, seed: int, *, total: int = N_PAIRS_TOTAL) -> np.ndarray:
    """Full field when ``pairs >= total``, else a SEEDED RANDOM sample -- never a prefix.

    A contiguous prefix of this clip is a DIFFERENT POPULATION and the bias has opposite
    sign per axis: pose prefixes measure **2.54-4.21x HARDER** than the population, seg
    prefixes **0.95-0.97x easier** (``ddm_na2``/``ddm_bp2``).  A prefix-drawn pose verdict
    is therefore the canonical false-negative shape, and a prefix-drawn seg win is the
    canonical false-positive.  Sorted output keeps downstream indexing sequential.

    Raises:
        InstrumentRefusal: non-positive ``pairs``.
    """
    if pairs <= 0:
        raise InstrumentRefusal(f"pairs must be positive, got {pairs}")
    if pairs >= total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=pairs, replace=False)).astype(np.int64)


# ---------------------------------------------------------------------------
# The receipt.  Every local number leaves with its lineage attached.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstrumentReceipt:
    """One local measurement, carrying the lineage that makes it meaningful.

    The false-authority fields are **fixed False by construction** -- they are not
    parameters, because there is no local configuration under which a local instrument
    becomes a score.  Only ``upstream/evaluate.py`` on contest hardware is a score.
    """

    instrument: str
    axis: str
    gt_lineage: str
    pairs: int
    sampling: str
    d_seg: float | None = None
    d_pose: float | None = None
    archive_bytes: int | None = None
    gt_sha256: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    #: Never a score.  Never promotable.  Not configurable.
    score_claim: bool = field(default=False, init=False)
    promotable: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        required = required_lineage_for_axis(self.axis)  # refuses unknown axis
        if self.gt_lineage != required:
            raise InstrumentRefusal(
                f"{self.instrument}: receipt declares axis {self.axis!r} (scored against "
                f"{required}) but GT lineage {self.gt_lineage!r}. A receipt may not record a "
                "number measured against one lineage under an axis scored against another."
            )
        if self.pairs <= 0:
            raise InstrumentRefusal(f"{self.instrument}: receipt has pairs={self.pairs}")

    @property
    def is_full_field(self) -> bool:
        """True when the receipt covers the whole scored population."""
        return self.pairs >= N_PAIRS_TOTAL

    def score(self) -> float:
        """The composed score, when all three legs are present.

        Raises:
            InstrumentRefusal: a leg is missing, so no score can be composed.
        """
        if self.d_seg is None or self.d_pose is None or self.archive_bytes is None:
            raise InstrumentRefusal(
                f"{self.instrument}: cannot compose a score from a partial receipt "
                f"(d_seg={self.d_seg}, d_pose={self.d_pose}, bytes={self.archive_bytes})"
            )
        return contest_score_from_legs(self.d_seg, self.d_pose, self.archive_bytes)

    def to_json(self) -> dict[str, Any]:
        """Machine-readable form, with the false-authority flags always present."""
        return {
            "instrument": self.instrument,
            "axis": self.axis,
            "gt_lineage": self.gt_lineage,
            "gt_sha256": self.gt_sha256,
            "pairs": self.pairs,
            "sampling": self.sampling,
            "full_field": self.is_full_field,
            "d_seg": self.d_seg,
            "d_pose": self.d_pose,
            "archive_bytes": self.archive_bytes,
            "score_claim": self.score_claim,
            "promotable": self.promotable,
            "notes": list(self.notes),
            "axis_note": (
                "local advisory instrument; only upstream/evaluate.py on contest hardware "
                "is a score. Absolute comparability to a contest row requires "
                f"gt_lineage == {gt_lineage.DALI_NVDEC}."
            ),
        }


# ---------------------------------------------------------------------------
# The frozen scorers.  Loaded once per process, CPU fp32, never MPS.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def load_segnet():
    """The frozen contest SegNet on CPU fp32 -- the seg verdict authority.

    MPS is a legitimate TRAINING-gradient device but NEVER an authority here: the seg
    objective is an ARGMAX, a discrete decision, and CLAUDE.md records MPS SegNet
    distortion drifting 2x.  A realized-acceptance loop accepting on MPS argmax would be
    accepting on a different function.
    """
    return _load_scorer("SegNet")


@functools.lru_cache(maxsize=1)
def load_posenet():
    """The frozen contest PoseNet on CPU fp32 -- the pose verdict authority."""
    return _load_scorer("PoseNet")


def _load_scorer(kind: str):
    import sys

    from safetensors.torch import load_file

    repo = Path(__file__).resolve().parents[2]
    upstream = repo / "upstream"
    sys.path.insert(0, str(upstream))
    try:
        import modules as upstream_modules
    finally:
        try:
            sys.path.remove(str(upstream))
        except ValueError:  # pragma: no cover - concurrent path mutation
            pass
    if kind == "SegNet":
        net = upstream_modules.SegNet().eval()
        state = upstream_modules.segnet_sd_path
    else:
        net = upstream_modules.PoseNet().eval()
        state = upstream_modules.posenet_sd_path
    net.load_state_dict(load_file(str(state), device="cpu"))
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


def d_seg_per_pair(argmax: np.ndarray, gt_labels: np.ndarray) -> np.ndarray:
    """Per-pair seg distortion, exactly ``SegNet.compute_distortion`` (modules.py:111-113).

    ``(argmax != gt).mean`` over the spatial axes, one float64 scalar per pair -- float64
    because the population mean of 600 such scalars is quoted to 8 decimals.
    """
    if argmax.shape != gt_labels.shape:
        raise InstrumentRefusal(
            f"argmax {argmax.shape} and GT {gt_labels.shape} disagree in shape"
        )
    if argmax.ndim < 2:
        raise InstrumentRefusal(f"argmax must have a leading pair axis, got {argmax.shape}")
    return (argmax != gt_labels).reshape(argmax.shape[0], -1).mean(axis=1, dtype=np.float64)


def d_pose_per_pair(pose: np.ndarray, targets: np.ndarray, *, dims: int = 6) -> np.ndarray:
    """Per-pair pose distortion: MSE over the first ``dims`` PoseNet components.

    ``upstream/modules.py`` scores only the first 6 of the 12 emitted components.
    """
    pose = np.asarray(pose, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if pose.shape[0] != targets.shape[0]:
        raise InstrumentRefusal(
            f"pose has {pose.shape[0]} pairs but targets have {targets.shape[0]}"
        )
    if pose.shape[1] < dims or targets.shape[1] < dims:
        raise InstrumentRefusal(
            f"pose/targets must carry at least {dims} components, got "
            f"{pose.shape[1]}/{targets.shape[1]}"
        )
    return ((pose[:, :dims] - targets[:, :dims]) ** 2).mean(axis=1)


def population_leg(per_pair: Sequence[float] | np.ndarray) -> float:
    """The population mean of a per-pair leg, in float64."""
    values = np.asarray(per_pair, dtype=np.float64)
    if values.size == 0:
        raise InstrumentRefusal("cannot take the population mean of an empty leg")
    return float(values.mean())
