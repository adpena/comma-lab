"""Per-architecture-class CUDA/CPU drift calibration registry.

This is the **adaptive learning layer** that complements the static
``cuda_cpu_axis_calibration`` helper (sister subagent a618cdd6). The static
helper hardcodes HNeRV-cluster R values; this registry maintains
**per-architecture-class** posteriors that update as new
``[contest-CPU]`` × ``[contest-CUDA]`` paired anchors arrive.

Why a single ``R`` is wrong
---------------------------
The empirically-anchored HNeRV-cluster ratios::

    R_pose ≈ 5.04   (pose CUDA / pose CPU)
    R_seg  ≈ 1.17   (seg CUDA / seg CPU)

are derived from PR100/101/102/103/105 — five archives that all share the
``hnerv_ft_microcodec`` decoder family on PR101's near-iid INT8 substrate.
Different architecture classes (``qhnerv_ft``, ``kitchen_sink_ensemble``,
raw-AV1 ``H3``, ``MNeRV``, ``balle_hyperprior``) have different precision
floors, different decoder paths (DALI vs PyAV vs custom), and different
saturation behaviour at the medal-band operating point. A single global ``R``
is correct only inside the HNeRV cluster; outside it, it produces biased
predicted-CPU bands.

Mechanism status: the ratios are empirical. The causal split between loader,
CUDA/CPU kernel numerics, and network-internal amplification remains a
hypothesis until loader and layer probes isolate it. The default decoder split
below is a prior for planning, not a proven decomposition.

Components
----------
1. :class:`ArchitectureProfile`: per-class posterior (``r_pose_mean/std``,
   ``r_seg_mean/std``, ``score_gap_mean/std``, ``decoder_drift_fraction``,
   ``pose_floor_estimate``, ``evidence_anchors``).
2. :class:`DecoderProfile`: per-decoder-pair drift split (default prior: 25%
   pose from decoder, very small for seg).
3. :func:`update_profile_from_anchor`: conjugate Bayesian update with 3-σ
   outlier flagging (per CLAUDE.md "killing as last resort" — flagged
   anchors do NOT auto-promote into the registry).
4. :func:`classify_archive_into_profile`: archive bytes / metadata →
   architecture class (with ``unknown_uncalibrated`` fallback).
5. :func:`decompose_observed_drift`: split observed pose drift into
   decoder-contribution vs network-contribution.
6. :func:`bootstrap_registry_from_hnerv_anchors`: seed PR100/101/102/103/105
   anchors so the registry starts in the known-calibrated state.

Design contract
---------------
- **No GPU dispatch.** Pure CPU / pure data structure.
- **No score claims.** Predicted CPU bands are tagged
  ``[predicted; learning-layer registry posterior]`` and ``score_claim`` /
  ``promotion_eligible`` / ``ready_for_exact_eval_dispatch`` all default
  False.
- **Backwards compatible.** Solvers that don't pass an
  ``architecture_class`` get the HNeRV defaults (current behaviour).
- **Persistence path:** ``.omx/state/cuda_cpu_axis_profile_registry.json``
  (gitignored per session policy). Durable summary lands in
  ``.omx/research/cuda_cpu_axis_profile_summary_<date>.md``.
- **Mutation locality:** every registry mutation goes through this module's
  helpers; raw JSON edits are forbidden so the audit log stays consistent.

The static helper (``cuda_cpu_axis_calibration``) is the BOOTSTRAP, not a
competitor: this module imports its constants when available and falls
back to in-module defaults if it is not yet on disk.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ── Bootstrap import from sister-subagent static helper ────────────────────
# The static helper at ``tac.optimization.cuda_cpu_axis_calibration`` may
# land before or after this module. Import lazily so neither subagent
# blocks the other; provide canonical fallbacks below.
try:  # pragma: no cover — import shim is intentionally permissive
    from tac.optimization.cuda_cpu_axis_calibration import (  # type: ignore
        R_POSE_HNERV as _STATIC_R_POSE_HNERV,
    )
    from tac.optimization.cuda_cpu_axis_calibration import (
        R_SEG_HNERV as _STATIC_R_SEG_HNERV,
    )
    _STATIC_HELPER_AVAILABLE = True
except Exception:  # pragma: no cover — broad guard for staged landing
    _STATIC_R_POSE_HNERV = None
    _STATIC_R_SEG_HNERV = None
    _STATIC_HELPER_AVAILABLE = False


# ── Empirical constants (PR100/101/102/103/105 paired anchors) ─────────────
# These five archives were measured CPU and CUDA on the same archive_bytes
# (contest auth-eval comments on the public PR). The R values come from
# ``reports/public_pr100_108_eval_comment_scorecard_20260508.json``.
_HNERV_CLUSTER_BOOTSTRAP_ANCHORS: list[dict[str, Any]] = [
    {
        "pr_num": 100,
        "title": "hnerv_lc_v2 submission (0.1954)",
        "author": "BradyMeighan",
        "archive_bytes": 178_981,
        "archive_sha": "0a8d343a5dc7f9e93d9d0e6cb8bc15f6c626c050",
        "cuda_pose": 0.00017198,
        "cpu_pose": 0.00003443,
        "cuda_seg": 0.00067623,
        "cpu_seg": 0.00057654,
        "cuda_score": 0.22826957,
        "cpu_score": 0.19538542,
        "observed_r_pose": 0.00017198 / 0.00003443,
        "observed_r_seg": 0.00067623 / 0.00057654,
        "score_gap": 0.22826957 - 0.19538542,
        "source": (
            "reports/public_pr100_108_eval_comment_scorecard_20260508.json "
            "[external_github_pr_comment]"
        ),
    },
    {
        "pr_num": 101,
        "title": "hnerv_ft_microcodec submission",
        "author": "SajayR",
        "archive_bytes": 178_258,
        "archive_sha": "ec7e366844fd8cffff33184e7ad92df22e93a908",
        "cuda_pose": 0.00017103,
        "cpu_pose": 0.00003286,
        "cuda_seg": 0.00066304,
        "cpu_seg": 0.00056023,
        "cuda_score": 0.22635446,
        "cpu_score": 0.19284501,
        "observed_r_pose": 0.00017103 / 0.00003286,
        "observed_r_seg": 0.00066304 / 0.00056023,
        "score_gap": 0.22635446 - 0.19284501,
        "source": (
            "reports/public_pr100_108_eval_comment_scorecard_20260508.json "
            "[external_github_pr_comment]"
        ),
    },
    {
        "pr_num": 102,
        "title": "hnerv_lc_v2_scale095_rplus1 submission",
        "author": "EthanYangTW",
        "archive_bytes": 178_981,
        "archive_sha": "1e330ec5633539c48278ce3cc96d2b15ea7a9eac",
        "cuda_pose": 0.00017347,
        "cpu_pose": 0.00003460,
        "cuda_seg": 0.00067565,
        "cpu_seg": 0.00057599,
        "cuda_score": 0.22839083,
        "cpu_score": 0.19537618,
        "observed_r_pose": 0.00017347 / 0.00003460,
        "observed_r_seg": 0.00067565 / 0.00057599,
        "score_gap": 0.22839083 - 0.19537618,
        "source": (
            "reports/public_pr100_108_eval_comment_scorecard_20260508.json "
            "[external_github_pr_comment]"
        ),
    },
    {
        "pr_num": 103,
        "title": "hnerv_lc_ac submission (0.19)",
        "author": "rem2",
        "archive_bytes": 178_223,
        "archive_sha": "d20270750e603346056f4e867cddd4744272c9d8",
        "cuda_pose": 0.00017198,
        "cpu_pose": 0.00003443,
        "cuda_seg": 0.00067623,
        "cpu_seg": 0.00057654,
        "cuda_score": 0.2277648516247398,
        "cpu_score": 0.19488070288878895,
        "observed_r_pose": 0.00017198 / 0.00003443,
        "observed_r_seg": 0.00067623 / 0.00057654,
        "score_gap": 0.2277648516247398 - 0.19488070288878895,
        "source": (
            "reports/public_pr100_108_eval_comment_scorecard_20260508.json "
            "[external_github_pr_comment]"
        ),
    },
    {
        "pr_num": 105,
        "title": "kitchen_sink (0.19797)",
        "author": "valtterivalo",
        "archive_bytes": 177_857,
        "archive_sha": "9376a6f86c76cb576b5f25afd8c789a8a727077f",
        "cuda_pose": 0.00017267,
        "cpu_pose": 0.00003472,
        "cuda_seg": 0.00070456,
        "cpu_seg": 0.00060913,
        "cuda_score": 0.2304372556953,
        "cpu_score": 0.1979739793436134,
        "observed_r_pose": 0.00017267 / 0.00003472,
        "observed_r_seg": 0.00070456 / 0.00060913,
        "score_gap": 0.2304372556953 - 0.1979739793436134,
        "source": (
            "reports/public_pr100_108_eval_comment_scorecard_20260508.json "
            "[external_github_pr_comment]"
        ),
    },
]

# Default decoder drift fraction prior from the deep-theory discussion. This
# is not proven; it is a planning prior until loader and layer probes isolate
# the split. SegNet is expected to be less sensitive because argmax is more
# robust than pose regression.
DEFAULT_DECODER_POSE_DRIFT_FRACTION = 0.25
DEFAULT_DECODER_SEG_DRIFT_FRACTION = 0.05

# Pose CUDA floor-band prior from the paired public comments. This is a
# diagnostic marker, not a proof that the official CPU axis cannot resolve
# lower pose error.
DEFAULT_POSE_FLOOR_ESTIMATE = 1.4e-4

# Outlier threshold (3 sigma) - anchors deviating beyond this from posterior
# mean are flagged ``outlier_candidate`` and DO NOT auto-promote.
OUTLIER_SIGMA_THRESHOLD = 3.0

# Confidence-band widening factor when an architecture class has fewer than
# this many anchors backing it.
LOW_CALIBRATION_ANCHOR_THRESHOLD = 3
LOW_CALIBRATION_BAND_WIDENING = 1.5

# ── Architecture-class taxonomy ────────────────────────────────────────────
# These labels are used by ``classify_archive_into_profile``. They mirror
# the decoder families in ``tac.cross_paradigm_wiring`` plus families
# observed in the public PR scorecards.
ARCHITECTURE_CLASSES = (
    "hnerv_ft_microcodec",
    "hnerv_lc_v2",
    "qhnerv_ft",
    "kitchen_sink_ensemble",
    "h3_av1_grayscale",
    "mnerv",
    "balle_scale_hyperprior",
    "raw_av1_yuv",
    "rgb_packed_brotli",
    "unknown_uncalibrated",
)

# Classifier heuristics: parser-section names that distinguish families.
# ``decoder_packed_brotli`` is the canonical HNeRV signature; archives
# whose first 4 bytes form ``ff_packed_brotli_hnerv`` and that contain
# ``decoder_packed_brotli`` + ``latents_and_sidecar_brotli`` sections fall
# inside the HNeRV cluster.
HNERV_PACKED_HEADER_PREFIX = b"\xff\xff\xff\xff"

_LINUX_CPU_OS_MARKERS = ("linux", "ubuntu", "debian")
_X86_64_CPU_MARKERS = ("x86_64", "amd64", "x64")
_MACOS_CPU_ADVISORY_MARKERS = (
    "darwin",
    "macos",
    "mac os",
    "apple",
    "apple_silicon",
    "arm64",
    "aarch64",
)
_CUDA_HARDWARE_MARKERS = ("cuda", "nvidia", "gpu", "t4", "h100", "a100", "l4", "sm")
_CONTEST_CPU_AXIS_MARKERS = ("contest_cpu", "contest cpu", "contest-cpu")
_CONTEST_CUDA_AXIS_MARKERS = ("contest_cuda", "contest cuda", "contest-cuda")


# ── Dataclasses ────────────────────────────────────────────────────────────
@dataclass
class DecoderProfile:
    """Drift contribution of a (CPU-decoder, GPU-decoder) pair.

    The 25% default is a hypothesis/prior for planning. Empirical loader and
    layer probes must replace it before it is used in a paper claim or a
    rank/kill decision.

    Decoders are named by their dataset class name to match
    ``upstream/frame_utils.py``'s routing.
    """
    decoder_pair: tuple[str, str]
    pose_drift_fraction: float = DEFAULT_DECODER_POSE_DRIFT_FRACTION
    seg_drift_fraction: float = DEFAULT_DECODER_SEG_DRIFT_FRACTION
    n_anchors: int = 0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoder_pair": list(self.decoder_pair),
            "pose_drift_fraction": float(self.pose_drift_fraction),
            "seg_drift_fraction": float(self.seg_drift_fraction),
            "n_anchors": int(self.n_anchors),
            "notes": str(self.notes),
        }


# Default decoder profile for any HNeRV-class archive: DALI on CUDA + PyAV on
# CPU. The decoder pair is code-grounded; the fraction is only a hypothesis.
DEFAULT_DECODER_PROFILE = DecoderProfile(
    decoder_pair=("DaliVideoDataset", "AVVideoDataset"),
    pose_drift_fraction=DEFAULT_DECODER_POSE_DRIFT_FRACTION,
    seg_drift_fraction=DEFAULT_DECODER_SEG_DRIFT_FRACTION,
    n_anchors=5,  # PR100/101/102/103/105
    notes=(
        "HNeRV cluster default; ground-truth path is DaliVideoDataset (CUDA) "
        "vs AVVideoDataset (CPU). Inflated archives go through "
        "TensorVideoDataset on both sides — only the GT side has the "
        "decoder split."
    ),
)


@dataclass
class ArchitectureProfile:
    """Per-architecture-class CUDA/CPU drift posterior.

    The posterior is updated by :func:`update_profile_from_anchor`. ``r_pose_*``
    and ``r_seg_*`` are the running mean/std across all evidence anchors.
    ``score_gap_*`` is the running mean/std of (cuda_score − cpu_score).

    ``evidence_anchors`` is the audit trail; every accepted anchor lands here
    along with its observed ratios. Outlier-flagged anchors are appended with
    ``outlier_candidate=True`` and DO NOT contribute to the posterior unless
    explicitly promoted.
    """
    architecture_class: str
    decoder_class: str = "DaliVideoDataset_then_TensorVideoDataset"
    n_anchors: int = 0

    r_pose_mean: float = 5.04
    r_pose_std: float = 0.10
    r_seg_mean: float = 1.17
    r_seg_std: float = 0.01
    score_gap_mean: float = 0.0330
    score_gap_std: float = 0.0010

    decoder_drift_fraction: float = DEFAULT_DECODER_POSE_DRIFT_FRACTION
    pose_floor_estimate: float = DEFAULT_POSE_FLOOR_ESTIMATE

    last_updated_utc: str = ""
    evidence_anchors: list[dict[str, Any]] = field(default_factory=list)

    notes: str = ""

    # ── Drift-mechanism discriminator fields (extension; 2026-05-09) ─────
    #
    # Populated by the AVVideoDataset CUDA-CPU drift mechanism discriminator
    # (lane_avvideodataset_cuda_path_mechanism_discriminator). Each field is
    # OPTIONAL — older serialised registries that predate the discriminator
    # leave them at the defaults below. Once the discriminator returns a
    # PRIMARY_MECHANISM_IDENTIFIED or MULTI_MECHANISM_PRIMARY verdict for an
    # architecture class, the corresponding field is filled in.
    #
    # ``loader_drift_correction``: scalar correction subtracted from CUDA
    # pose to estimate the loader-byte-drift component of the gap. ``None``
    # means uncalibrated.
    #
    # ``conv_kernel_determinism_required``: if True, archive builders for
    # this class SHOULD set ``torch.use_deterministic_algorithms(True)``
    # and ``torch.backends.cudnn.deterministic = True`` in their inflate.py.
    #
    # ``head_quantize_post_inference_dtype``: if non-empty, archive builders
    # for this class SHOULD pre-quantize inflate output to a coarser grid.
    # Currently the only canonical value is ``"uint8_round_multiple_of_2"``.
    #
    # ``mechanism_discriminator_verdict``: opaque verdict label from the
    # discriminator; one of "PRIMARY_MECHANISM_IDENTIFIED",
    # "MULTI_MECHANISM_PRIMARY", "MULTI_MECHANISM_PRIMARY_PLUS_CONTRIBUTING",
    # "MULTI_MECHANISM_CONTRIBUTING_ONLY", "FOURTH_MECHANISM_HYPOTHESIS",
    # "INCONCLUSIVE_*", or empty (uncalibrated).
    #
    # ``mechanism_discriminator_evidence_path``: relative path to the
    # ``discriminator_verdict.json`` artifact that produced this update.
    loader_drift_correction: float | None = None
    conv_kernel_determinism_required: bool = False
    head_quantize_post_inference_dtype: str = ""
    mechanism_discriminator_verdict: str = ""
    mechanism_discriminator_evidence_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "architecture_class": self.architecture_class,
            "decoder_class": self.decoder_class,
            "n_anchors": int(self.n_anchors),
            "r_pose_mean": float(self.r_pose_mean),
            "r_pose_std": float(self.r_pose_std),
            "r_seg_mean": float(self.r_seg_mean),
            "r_seg_std": float(self.r_seg_std),
            "score_gap_mean": float(self.score_gap_mean),
            "score_gap_std": float(self.score_gap_std),
            "decoder_drift_fraction": float(self.decoder_drift_fraction),
            "pose_floor_estimate": float(self.pose_floor_estimate),
            "last_updated_utc": str(self.last_updated_utc),
            "evidence_anchors": list(self.evidence_anchors),
            "notes": str(self.notes),
        }
        # Discriminator fields are only emitted when populated, so the
        # serialised registry stays compact for uncalibrated classes.
        if self.loader_drift_correction is not None:
            out["loader_drift_correction"] = float(self.loader_drift_correction)
        if self.conv_kernel_determinism_required:
            out["conv_kernel_determinism_required"] = bool(
                self.conv_kernel_determinism_required
            )
        if self.head_quantize_post_inference_dtype:
            out["head_quantize_post_inference_dtype"] = str(
                self.head_quantize_post_inference_dtype
            )
        if self.mechanism_discriminator_verdict:
            out["mechanism_discriminator_verdict"] = str(
                self.mechanism_discriminator_verdict
            )
        if self.mechanism_discriminator_evidence_path:
            out["mechanism_discriminator_evidence_path"] = str(
                self.mechanism_discriminator_evidence_path
            )
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchitectureProfile:
        loader_correction_raw = data.get("loader_drift_correction", None)
        return cls(
            architecture_class=str(data["architecture_class"]),
            decoder_class=str(data.get("decoder_class", "")),
            n_anchors=int(data.get("n_anchors", 0)),
            r_pose_mean=float(data.get("r_pose_mean", 5.04)),
            r_pose_std=float(data.get("r_pose_std", 0.10)),
            r_seg_mean=float(data.get("r_seg_mean", 1.17)),
            r_seg_std=float(data.get("r_seg_std", 0.01)),
            score_gap_mean=float(data.get("score_gap_mean", 0.033)),
            score_gap_std=float(data.get("score_gap_std", 0.001)),
            decoder_drift_fraction=float(
                data.get("decoder_drift_fraction", DEFAULT_DECODER_POSE_DRIFT_FRACTION)
            ),
            pose_floor_estimate=float(
                data.get("pose_floor_estimate", DEFAULT_POSE_FLOOR_ESTIMATE)
            ),
            last_updated_utc=str(data.get("last_updated_utc", "")),
            evidence_anchors=list(data.get("evidence_anchors", [])),
            notes=str(data.get("notes", "")),
            loader_drift_correction=(
                None if loader_correction_raw is None
                else float(loader_correction_raw)
            ),
            conv_kernel_determinism_required=bool(
                data.get("conv_kernel_determinism_required", False)
            ),
            head_quantize_post_inference_dtype=str(
                data.get("head_quantize_post_inference_dtype", "")
            ),
            mechanism_discriminator_verdict=str(
                data.get("mechanism_discriminator_verdict", "")
            ),
            mechanism_discriminator_evidence_path=str(
                data.get("mechanism_discriminator_evidence_path", "")
            ),
        )

    def confidence_label(self) -> str:
        """Return a human-readable confidence label."""
        if self.n_anchors >= LOW_CALIBRATION_ANCHOR_THRESHOLD:
            return "calibrated"
        if self.n_anchors == 0:
            return "uncalibrated_default"
        return "low-calibration-confidence"

    def predict_cpu_score(
        self,
        *,
        cuda_score: float,
    ) -> dict[str, Any]:
        """Predict the CPU score from a CUDA score using the posterior.

        Returns a dict with ``predicted_cpu_score`` (point estimate),
        ``predicted_cpu_score_low``/``..._high`` (1-σ band, widened by
        ``LOW_CALIBRATION_BAND_WIDENING`` if the class has < 3 anchors),
        and ``confidence_label``.

        The estimate is ``cuda_score − score_gap_mean``. The band uses
        ``score_gap_std``. This is the simplest defensible estimator; the
        registry can grow more sophisticated estimators (e.g. per-anchor
        leave-one-out cross-validation) without changing the call signature.
        """
        widen = (
            LOW_CALIBRATION_BAND_WIDENING
            if self.n_anchors < LOW_CALIBRATION_ANCHOR_THRESHOLD
            else 1.0
        )
        point = float(cuda_score - self.score_gap_mean)
        band_half = widen * self.score_gap_std
        return {
            "predicted_cpu_score": point,
            "predicted_cpu_score_low": point - band_half,
            "predicted_cpu_score_high": point + band_half,
            "score_gap_used": self.score_gap_mean,
            "score_gap_band_half": band_half,
            "confidence_label": self.confidence_label(),
            "evidence_grade": "[predicted; learning-layer registry posterior]",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }


@dataclass
class ProfileUpdate:
    """Diff record emitted by :func:`harvest_new_anchor_and_update`."""
    architecture_class: str
    accepted: bool
    outlier_candidate: bool
    anchor: dict[str, Any]
    before: dict[str, Any]
    after: dict[str, Any]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_class": self.architecture_class,
            "accepted": bool(self.accepted),
            "outlier_candidate": bool(self.outlier_candidate),
            "anchor": dict(self.anchor),
            "before": dict(self.before),
            "after": dict(self.after),
            "notes": str(self.notes),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }


# ── Bootstrap ──────────────────────────────────────────────────────────────
def bootstrap_registry_from_hnerv_anchors() -> dict[str, ArchitectureProfile]:
    """Return the initial registry seeded with PR100/101/102/103/105 anchors.

    The five anchors all sit inside the ``hnerv_ft_microcodec`` class. Other
    classes start empty (``n_anchors=0``) with conservative HNeRV-defaults
    and are flagged ``uncalibrated_default`` until the 25-PR sweep
    (subagent ad0875a8) seeds them.
    """
    registry: dict[str, ArchitectureProfile] = {}

    # HNeRV cluster (calibrated)
    hnerv = ArchitectureProfile(
        architecture_class="hnerv_ft_microcodec",
        decoder_class="DaliVideoDataset_then_TensorVideoDataset",
        notes=(
            "Bootstrapped from 5 paired (CPU, CUDA) anchors PR100/101/102/103/105. "
            "Posterior derived via _recompute_posterior; outlier check uses "
            "OUTLIER_SIGMA_THRESHOLD=3.0 sigma."
        ),
    )
    for anchor in _HNERV_CLUSTER_BOOTSTRAP_ANCHORS:
        # Direct seeding bypasses the outlier check (this IS the calibration
        # set defining the prior). Subsequent anchors WILL be checked.
        hnerv.evidence_anchors.append({
            **anchor,
            "outlier_candidate": False,
            "promoted": False,
            "external_comment_seed": True,
            "seeded_at_bootstrap": True,
            "promotion_eligible": False,
            "score_claim": False,
        })
    _recompute_posterior_in_place(hnerv)
    registry["hnerv_ft_microcodec"] = hnerv

    # Other known families — uncalibrated defaults; flagged as needing
    # anchors from the 25-PR sweep.
    for cls_name in ARCHITECTURE_CLASSES:
        if cls_name == "hnerv_ft_microcodec":
            continue
        registry[cls_name] = ArchitectureProfile(
            architecture_class=cls_name,
            decoder_class=_default_decoder_class_for(cls_name),
            n_anchors=0,
            notes=(
                "Uncalibrated default - no [contest-CPU]x[contest-CUDA] "
                "anchors yet. Awaiting 25-PR sweep (subagent ad0875a8) "
                "or per-class dispatch."
            ),
        )

    return registry


def _default_decoder_class_for(arch_class: str) -> str:
    """Infer the (CPU, CUDA) decoder pair label from architecture class."""
    if arch_class.startswith("hnerv") or arch_class == "kitchen_sink_ensemble":
        return "DaliVideoDataset_then_TensorVideoDataset"
    if arch_class.startswith("h3") or arch_class.startswith("raw_av1"):
        # H3 / raw-AV1 decode through the runtime AV1 pipeline; both CPU and
        # CUDA paths use TensorVideoDataset on inflated frames so decoder
        # contribution is small. The GT pre-decode does still split.
        return "DaliVideoDataset_then_TensorVideoDataset"
    if arch_class == "balle_scale_hyperprior":
        return "AVVideoDataset_then_TensorVideoDataset"
    return "DaliVideoDataset_then_TensorVideoDataset"


# ── Bayesian update primitive ──────────────────────────────────────────────
def _recompute_posterior_in_place(profile: ArchitectureProfile) -> None:
    """Recompute mean/std of R values from the accepted evidence anchors.

    Uses simple unweighted mean + sample std (n-1 denominator). Anchors
    flagged ``outlier_candidate=True`` are excluded.
    """
    anchors = [
        a for a in profile.evidence_anchors
        if not a.get("outlier_candidate", False)
    ]
    profile.n_anchors = len(anchors)
    profile.last_updated_utc = datetime.now(UTC).isoformat()

    if not anchors:
        return

    pose_ratios = [float(a["observed_r_pose"]) for a in anchors]
    seg_ratios = [float(a["observed_r_seg"]) for a in anchors]
    score_gaps = [float(a.get("score_gap", 0.0)) for a in anchors]

    profile.r_pose_mean = sum(pose_ratios) / len(pose_ratios)
    profile.r_seg_mean = sum(seg_ratios) / len(seg_ratios)
    profile.score_gap_mean = sum(score_gaps) / len(score_gaps)

    # Sample std with n-1 (Bessel-corrected) when n >= 2; floor at 1e-6.
    if len(pose_ratios) >= 2:
        profile.r_pose_std = max(
            1e-6,
            math.sqrt(
                sum((x - profile.r_pose_mean) ** 2 for x in pose_ratios)
                / (len(pose_ratios) - 1)
            ),
        )
        profile.r_seg_std = max(
            1e-6,
            math.sqrt(
                sum((x - profile.r_seg_mean) ** 2 for x in seg_ratios)
                / (len(seg_ratios) - 1)
            ),
        )
        profile.score_gap_std = max(
            1e-6,
            math.sqrt(
                sum((x - profile.score_gap_mean) ** 2 for x in score_gaps)
                / (len(score_gaps) - 1)
            ),
        )
    else:
        # Single anchor: keep prior std (the pre-update value) so the
        # confidence band doesn't artificially collapse to zero.
        pass


def update_profile_from_anchor(
    profile: ArchitectureProfile,
    anchor: dict[str, Any],
    *,
    outlier_sigma_threshold: float = OUTLIER_SIGMA_THRESHOLD,
) -> ProfileUpdate:
    """Bayesian-update ``profile`` with a new ``(cuda_score, cpu_score)`` anchor.

    Required anchor fields::

        observed_r_pose : float   (cuda_pose / cpu_pose)
        observed_r_seg  : float   (cuda_seg / cpu_seg)
        score_gap       : float   (cuda_score - cpu_score)

    Optional metadata: ``pr_num``, ``title``, ``archive_sha``, ``archive_bytes``,
    ``cuda_*``, ``cpu_*``, ``source``.

    Outlier policy: if ``|observed_r_pose − r_pose_mean| > sigma_threshold *
    r_pose_std`` AND the profile has at least 3 anchors backing it (so the
    prior is meaningful), flag the anchor and DO NOT promote into the
    posterior. The anchor is appended with ``outlier_candidate=True`` so
    operators can review and explicitly promote later.

    Per CLAUDE.md "killing as last resort" the function never DROPS
    evidence — flagged anchors stay in the audit trail. The recompute
    only excludes them from the running posterior.
    """
    required_keys = ("observed_r_pose", "observed_r_seg", "score_gap")
    for key in required_keys:
        if key not in anchor:
            raise ValueError(
                f"anchor missing required key '{key}'; got {sorted(anchor)}"
            )

    before = {
        "n_anchors": profile.n_anchors,
        "r_pose_mean": profile.r_pose_mean,
        "r_pose_std": profile.r_pose_std,
        "r_seg_mean": profile.r_seg_mean,
        "r_seg_std": profile.r_seg_std,
        "score_gap_mean": profile.score_gap_mean,
        "score_gap_std": profile.score_gap_std,
    }

    obs_pose = float(anchor["observed_r_pose"])
    is_outlier = False
    outlier_reason = ""

    if (
        profile.n_anchors >= LOW_CALIBRATION_ANCHOR_THRESHOLD
        and profile.r_pose_std > 0.0
    ):
        z = abs(obs_pose - profile.r_pose_mean) / profile.r_pose_std
        if z > outlier_sigma_threshold:
            is_outlier = True
            outlier_reason = (
                f"observed_r_pose={obs_pose:.4f} deviates {z:.2f} sigma from "
                f"posterior mean {profile.r_pose_mean:.4f} (threshold "
                f"{outlier_sigma_threshold:.1f} sigma); flagging as "
                f"outlier_candidate per CLAUDE.md "
                f"forbidden_premature_kill_without_research_exhaustion"
            )

    enriched_anchor = dict(anchor)
    enriched_anchor["outlier_candidate"] = is_outlier
    enriched_anchor["promoted"] = not is_outlier
    enriched_anchor["seeded_at_bootstrap"] = False
    enriched_anchor.setdefault("source", "unknown")
    enriched_anchor.setdefault("ingested_utc", datetime.now(UTC).isoformat())
    enriched_anchor["score_claim"] = False
    enriched_anchor["promotion_eligible"] = False
    enriched_anchor["rank_or_kill_eligible"] = False
    enriched_anchor["ready_for_exact_eval_dispatch"] = False
    if outlier_reason:
        enriched_anchor["outlier_reason"] = outlier_reason

    profile.evidence_anchors.append(enriched_anchor)

    if not is_outlier:
        _recompute_posterior_in_place(profile)
    else:
        profile.last_updated_utc = datetime.now(UTC).isoformat()

    after = {
        "n_anchors": profile.n_anchors,
        "r_pose_mean": profile.r_pose_mean,
        "r_pose_std": profile.r_pose_std,
        "r_seg_mean": profile.r_seg_mean,
        "r_seg_std": profile.r_seg_std,
        "score_gap_mean": profile.score_gap_mean,
        "score_gap_std": profile.score_gap_std,
    }

    return ProfileUpdate(
        architecture_class=profile.architecture_class,
        accepted=not is_outlier,
        outlier_candidate=is_outlier,
        anchor=enriched_anchor,
        before=before,
        after=after,
        notes=outlier_reason if is_outlier else "anchor accepted into posterior",
    )


# ── Architecture-class auto-classifier ─────────────────────────────────────
def classify_archive_into_profile(
    archive_path: str | Path | None = None,
    archive_metadata: dict[str, Any] | None = None,
) -> str:
    """Classify an archive into one of :data:`ARCHITECTURE_CLASSES`.

    ``archive_metadata`` is the canonical input — typically the JSON from
    ``tac.archive_byte_profile`` (parser sections + inferred-kind). When the
    metadata is sufficient, the archive bytes are not opened.

    Falls back to ``unknown_uncalibrated`` when nothing matches.
    """
    md = archive_metadata or {}

    # 1) Explicit override via ``architecture_class`` key
    if md.get("architecture_class") in ARCHITECTURE_CLASSES:
        return str(md["architecture_class"])

    # 2) inferred_kind from byte-level profilers
    inferred = str(md.get("inferred_kind", "")).lower()
    title = str(md.get("title", "")).lower()
    description = str(md.get("description", "")).lower()
    sections = md.get("sections") or md.get("parser_sections") or []
    section_names = {str(s.get("name", "")).lower() for s in sections if isinstance(s, dict)}

    text_blob = " ".join([inferred, title, description])

    # HNeRV family — distinguishing micro/lc_v2 sub-classes
    if "ff_packed_brotli_hnerv" in inferred or "decoder_packed_brotli" in section_names:
        if "lc_v2" in text_blob or "lc-v2" in text_blob:
            return "hnerv_lc_v2"
        if "ft_microcodec" in text_blob or "microcodec" in text_blob:
            return "hnerv_ft_microcodec"
        # Default HNeRV bucket
        return "hnerv_ft_microcodec"

    if "qhnerv" in text_blob:
        return "qhnerv_ft"

    if "kitchen_sink" in text_blob:
        return "kitchen_sink_ensemble"

    if re.search(r"\bh3\b", text_blob) or "h3_av1" in text_blob:
        return "h3_av1_grayscale"

    if "mnerv" in text_blob:
        return "mnerv"

    if "balle" in text_blob or "scale_hyperprior" in text_blob:
        return "balle_scale_hyperprior"

    if "raw_av1" in text_blob or ("av1" in text_blob and "yuv" in text_blob):
        return "raw_av1_yuv"

    if "rgb_packed_brotli" in text_blob:
        return "rgb_packed_brotli"

    # 3) Header sniff if a path was provided
    if archive_path is not None:
        try:
            with open(archive_path, "rb") as f:
                head = f.read(8)
            if head.startswith(HNERV_PACKED_HEADER_PREFIX):
                return "hnerv_ft_microcodec"
        except Exception:
            pass

    return "unknown_uncalibrated"


# ── Decoder-aware split ────────────────────────────────────────────────────
def decompose_observed_drift(
    *,
    observed_r_pose: float,
    decoder_profile: DecoderProfile = DEFAULT_DECODER_PROFILE,
    pose_floor_estimate: float = DEFAULT_POSE_FLOOR_ESTIMATE,
) -> dict[str, float]:
    """Split observed pose drift into decoder vs network contributions.

    Uses the decoder-profile fraction as a hypothesis prior. The HNeRV
    cluster has a confirmed GT-side NVDEC-vs-PyAV split, but its share of
    the observed pose drift is not proven until loader/layer probes land.

    The decomposition is multiplicative on the (R - 1) excess::

        excess = R_pose - 1
        decoder_contribution = excess * decoder_profile.pose_drift_fraction
        network_contribution = excess * (1 - decoder_profile.pose_drift_fraction)

    Different exploits target different axes:
        - Decoder contribution: train/render to be robust to the official
          CPU decode path; do not edit upstream or scorer code.
        - Network contribution: matched noise injection at training time is
          a hypothesis, not promotion evidence.
    """
    if observed_r_pose < 0.0:
        raise ValueError("observed_r_pose must be non-negative")

    excess = observed_r_pose - 1.0
    decoder_fraction = decoder_profile.pose_drift_fraction
    network_fraction = 1.0 - decoder_fraction

    return {
        "observed_r_pose": float(observed_r_pose),
        "excess_above_unity": float(excess),
        "decoder_fraction_used": float(decoder_fraction),
        "decoder_contribution": float(excess * decoder_fraction),
        "network_contribution": float(excess * network_fraction),
        "pose_floor_estimate": float(pose_floor_estimate),
        "decoder_pair": list(decoder_profile.decoder_pair),
        "evidence_grade": "[CPU-prep planning-only; mechanism estimate]",
    }


# ── Persistence ─────────────────────────────────────────────────────────────
DEFAULT_REGISTRY_PATH = Path(".omx/state/cuda_cpu_axis_profile_registry.json")
DEFAULT_AUDIT_LOG_PATH = Path(".omx/research/cuda_cpu_axis_profile_updates.jsonl")


def serialize_registry(
    registry: dict[str, ArchitectureProfile],
) -> dict[str, Any]:
    """Return a JSON-serialisable dict for ``registry``."""
    return {
        "schema": "cuda_cpu_axis_profile_registry.v1",
        "evidence_grade": "[CPU-prep learning-layer planning-only]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "static_helper_imported": _STATIC_HELPER_AVAILABLE,
        "static_r_pose_hnerv": _STATIC_R_POSE_HNERV,
        "static_r_seg_hnerv": _STATIC_R_SEG_HNERV,
        "default_decoder_profile": DEFAULT_DECODER_PROFILE.to_dict(),
        "outlier_sigma_threshold": OUTLIER_SIGMA_THRESHOLD,
        "low_calibration_anchor_threshold": LOW_CALIBRATION_ANCHOR_THRESHOLD,
        "low_calibration_band_widening": LOW_CALIBRATION_BAND_WIDENING,
        "default_pose_floor_estimate": DEFAULT_POSE_FLOOR_ESTIMATE,
        "last_serialized_utc": datetime.now(UTC).isoformat(),
        "profiles": {
            name: prof.to_dict() for name, prof in registry.items()
        },
    }


def deserialize_registry(
    payload: dict[str, Any],
) -> dict[str, ArchitectureProfile]:
    """Return an in-memory registry from a serialised payload."""
    profiles_raw = payload.get("profiles", {})
    return {
        name: ArchitectureProfile.from_dict(data)
        for name, data in profiles_raw.items()
    }


def write_registry(
    registry: dict[str, ArchitectureProfile],
    path: Path | str = DEFAULT_REGISTRY_PATH,
) -> Path:
    """Persist ``registry`` to ``path`` (creates parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_registry(registry)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return p


def read_registry(
    path: Path | str = DEFAULT_REGISTRY_PATH,
    *,
    bootstrap_if_missing: bool = True,
) -> dict[str, ArchitectureProfile]:
    """Load registry from ``path``; bootstrap if not present and requested."""
    p = Path(path)
    if not p.exists():
        if bootstrap_if_missing:
            return bootstrap_registry_from_hnerv_anchors()
        raise FileNotFoundError(p)
    payload = json.loads(p.read_text(encoding="utf-8"))
    return deserialize_registry(payload)


# ── Axis-custody helpers ─────────────────────────────────────────────────────
def _hardware_blob(*values: Any) -> str:
    """Return a normalized hardware/provenance text blob for conservative checks."""
    parts = [str(value).strip().lower() for value in values if value is not None]
    return " ".join(parts).replace("-", "_")


def _is_linux_x86_64_cpu_hardware(*values: Any) -> bool:
    """True only for labels that can back a ``[contest-CPU]`` learning anchor."""
    blob = _hardware_blob(*values)
    if not blob:
        return False
    if any(marker in blob for marker in _MACOS_CPU_ADVISORY_MARKERS):
        return False
    has_linux_os = any(marker in blob for marker in _LINUX_CPU_OS_MARKERS)
    has_x86_64 = any(marker in blob for marker in _X86_64_CPU_MARKERS)
    return has_linux_os and has_x86_64


def _is_cuda_eval_hardware(*values: Any) -> bool:
    """True for recognizably CUDA/GPU hardware labels, not MPS advisories."""
    blob = _hardware_blob(*values)
    if not blob:
        return False
    if "mps" in blob or "metal" in blob:
        return False
    return any(marker in blob for marker in _CUDA_HARDWARE_MARKERS)


def _has_contest_cpu_axis_metadata(*values: Any) -> bool:
    """True when nested payload metadata explicitly identifies contest CPU."""
    blob = _hardware_blob(*values)
    return any(marker in blob for marker in _CONTEST_CPU_AXIS_MARKERS)


def _has_contest_cuda_axis_metadata(*values: Any) -> bool:
    """True when nested payload metadata explicitly identifies contest CUDA."""
    blob = _hardware_blob(*values)
    return any(marker in blob for marker in _CONTEST_CUDA_AXIS_MARKERS)


# ── Online-learning hook ───────────────────────────────────────────────────
def _extract_anchor_from_contest_auth_eval(
    contest_auth_eval_payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Convert a ``contest_auth_eval.adjudicated.json`` payload to an anchor.

    The adjudicated JSON typically carries paired CPU/CUDA scores for a
    single archive. Returns None if the pairing is incomplete.

    Expected keys::

        {
          "archive_bytes": int,
          "archive_sha256": str,
          "cpu": {"pose": float, "seg": float, "score": float},
          "cuda": {"pose": float, "seg": float, "score": float},
          ...
        }

    Tolerates several legacy spellings (``contest_cpu_score`` etc.) so the
    harvester works on adjudicated JSONs from multiple subagents.
    """
    cpu = contest_auth_eval_payload.get("cpu") or {}
    cuda = contest_auth_eval_payload.get("cuda") or {}

    # Legacy single-flat-record format
    if not (cpu and cuda):
        flat_cpu = contest_auth_eval_payload.get("contest_cpu") or {}
        flat_cuda = contest_auth_eval_payload.get("contest_cuda") or {}
        cpu = cpu or flat_cpu
        cuda = cuda or flat_cuda

    cpu_pose = cpu.get("pose")
    cpu_seg = cpu.get("seg")
    cpu_score = cpu.get("score")
    cuda_pose = cuda.get("pose")
    cuda_seg = cuda.get("seg")
    cuda_score = cuda.get("score")

    if any(v is None for v in (cpu_pose, cpu_seg, cuda_pose, cuda_seg)):
        return None

    if cpu_pose <= 0.0 or cpu_seg <= 0.0:
        return None

    # Compute scores if not provided
    if cpu_score is None or cuda_score is None:
        archive_bytes = int(
            contest_auth_eval_payload.get("archive_bytes", 0)
        )
        if archive_bytes <= 0:
            return None
        from tac.score_geometry import contest_score
        cpu_score = contest_score(float(cpu_seg), float(cpu_pose), archive_bytes)
        cuda_score = contest_score(float(cuda_seg), float(cuda_pose), archive_bytes)

    archive_bytes = int(contest_auth_eval_payload.get("archive_bytes", 0))
    archive_sha = str(
        contest_auth_eval_payload.get("archive_sha256")
        or contest_auth_eval_payload.get("archive_sha")
        or ""
    )
    runtime_manifest = contest_auth_eval_payload.get("inflate_runtime_manifest") or {}
    runtime_tree_sha = str(
        contest_auth_eval_payload.get("runtime_tree_sha256")
        or contest_auth_eval_payload.get("inflate_runtime_tree_sha256")
        or (
            runtime_manifest.get("runtime_tree_sha256")
            if isinstance(runtime_manifest, dict) else ""
        )
        or ""
    )
    sample_count = contest_auth_eval_payload.get("sample_count")
    if sample_count is None:
        sample_count = contest_auth_eval_payload.get("n_samples")
    cpu_hardware = str(
        cpu.get("hardware")
        or contest_auth_eval_payload.get("cpu_hardware")
        or contest_auth_eval_payload.get("hardware_cpu")
        or ""
    )
    cuda_hardware = str(
        cuda.get("hardware")
        or contest_auth_eval_payload.get("cuda_hardware")
        or contest_auth_eval_payload.get("hardware_cuda")
        or ""
    )
    if (
        archive_bytes <= 0
        or not archive_sha
        or not runtime_tree_sha
        or sample_count is None
        or int(sample_count) <= 0
        or not cpu_hardware
        or not cuda_hardware
    ):
        return None

    if not _has_contest_cpu_axis_metadata(
        cpu.get("evidence_grade"),
        cpu.get("score_axis"),
    ):
        return None
    if not _is_linux_x86_64_cpu_hardware(cpu_hardware):
        return None
    if not _has_contest_cuda_axis_metadata(
        cuda.get("evidence_grade"),
        cuda.get("score_axis"),
    ):
        return None
    if not _is_cuda_eval_hardware(cuda_hardware):
        return None

    return {
        "observed_r_pose": float(cuda_pose) / float(cpu_pose),
        "observed_r_seg": float(cuda_seg) / float(cpu_seg),
        "score_gap": float(cuda_score) - float(cpu_score),
        "cpu_pose": float(cpu_pose),
        "cuda_pose": float(cuda_pose),
        "cpu_seg": float(cpu_seg),
        "cuda_seg": float(cuda_seg),
        "cpu_score": float(cpu_score),
        "cuda_score": float(cuda_score),
        "archive_bytes": archive_bytes,
        "archive_sha": archive_sha,
        "runtime_tree_sha256": runtime_tree_sha,
        "sample_count": int(sample_count),
        "cpu_hardware": cpu_hardware,
        "cuda_hardware": cuda_hardware,
        "cpu_axis_custody": "contest_cpu_linux_x86_64",
        "cuda_axis_custody": "contest_cuda_exact_eval",
        "source": str(
            contest_auth_eval_payload.get("source", "contest_auth_eval.adjudicated")
        ),
        "title": str(contest_auth_eval_payload.get("title", "")),
        "pr_num": contest_auth_eval_payload.get("pr_num"),
    }


def harvest_new_anchor_and_update(
    contest_auth_eval_payload: dict[str, Any],
    *,
    registry: dict[str, ArchitectureProfile] | None = None,
    architecture_class: str | None = None,
    archive_metadata: dict[str, Any] | None = None,
    audit_log_path: Path | str | None = DEFAULT_AUDIT_LOG_PATH,
) -> ProfileUpdate | None:
    """Ingest a single contest_auth_eval anchor and update the registry.

    Steps::

        1. Extract paired (cuda, cpu) numbers from the JSON payload.
        2. Classify into architecture class (or use override).
        3. Run :func:`update_profile_from_anchor` with outlier check.
        4. Append a JSONL line to ``audit_log_path`` for forensics.
        5. Return the ``ProfileUpdate`` with before/after diff.

    Returns None when the payload doesn't carry a paired CPU/CUDA result
    (e.g. only one side has been measured yet).

    The caller is responsible for writing the registry back to disk via
    :func:`write_registry` after a batch of updates — this keeps the
    ingest step pure and testable.
    """
    anchor = _extract_anchor_from_contest_auth_eval(contest_auth_eval_payload)
    if anchor is None:
        return None

    if registry is None:
        registry = bootstrap_registry_from_hnerv_anchors()

    arch_class = (
        architecture_class
        or classify_archive_into_profile(
            archive_path=contest_auth_eval_payload.get("archive_path"),
            archive_metadata=archive_metadata or contest_auth_eval_payload,
        )
    )

    profile = registry.get(arch_class)
    if profile is None:
        # New uncalibrated class — instantiate a default
        profile = ArchitectureProfile(
            architecture_class=arch_class,
            decoder_class=_default_decoder_class_for(arch_class),
            n_anchors=0,
            notes=f"Auto-instantiated by harvest at {datetime.now(UTC).isoformat()}",
        )
        registry[arch_class] = profile

    update = update_profile_from_anchor(profile, anchor)

    if audit_log_path is not None:
        _append_audit_log(audit_log_path, update)

    return update


def _append_audit_log(path: Path | str, update: ProfileUpdate) -> None:
    """Append one JSONL line documenting a profile update."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(update.to_dict(), sort_keys=True) + "\n")


# ── Discriminator → registry wire-in (2026-05-09) ─────────────────────────
def apply_discriminator_verdict_to_registry(
    verdict: dict[str, Any],
    *,
    registry: dict[str, ArchitectureProfile],
    evidence_path: str = "",
) -> dict[str, Any]:
    """Apply a discriminator-verdict JSON to the per-class registry.

    The verdict schema is the one emitted by
    ``tools/analyze_a1_cuda_cpu_drift_discriminator_verdict.py``: it carries
    a top-level ``verdict`` label, an ``isolation_findings`` list, and a
    ``registry_update_spec`` block whose
    ``applies_to_architecture_class`` key names the target class.

    For each isolation finding labelled ``PRIMARY_MECHANISM`` or
    ``CONTRIBUTING_MECHANISM``, the corresponding field on the architecture
    profile is set:

      - ``loader_byte_drift``                  → ``loader_drift_correction``
      - ``conv_kernel_accumulation_drift``    → ``conv_kernel_determinism_required``
      - ``hydra_head_numerical_sensitivity``  → ``head_quantize_post_inference_dtype``

    The class-level ``mechanism_discriminator_verdict`` and
    ``mechanism_discriminator_evidence_path`` are also stamped so a future
    reader can trace back to the artifact that produced the update.

    Returns a diff dict ``{"architecture_class", "before", "after",
    "fields_changed"}`` so callers can audit / persist.

    Per CLAUDE.md ``forbidden_premature_kill_without_research_exhaustion``:
    when ``verdict["verdict"] == "FOURTH_MECHANISM_HYPOTHESIS"`` the
    function records the verdict label and evidence path on the profile but
    does NOT toggle any of the per-mechanism flags. Operators can then
    decide whether a 4th hypothesis (not in our 3-way split) needs a new
    discriminator generation. The discriminator family is NOT killed.
    """
    arch_class = str(
        verdict.get("registry_update_spec", {}).get(
            "applies_to_architecture_class", ""
        )
    )
    if not arch_class:
        raise ValueError(
            "verdict missing registry_update_spec.applies_to_architecture_class; "
            "cannot apply"
        )
    profile = registry.get(arch_class)
    if profile is None:
        raise KeyError(
            f"architecture class {arch_class!r} not in registry; bootstrap "
            f"or add it before applying a discriminator verdict"
        )
    before = profile.to_dict()

    # Always stamp the verdict label + evidence path so the audit trail is
    # present even on FOURTH_MECHANISM_HYPOTHESIS / INCONCLUSIVE_* outcomes.
    profile.mechanism_discriminator_verdict = str(verdict.get("verdict", ""))
    if evidence_path:
        profile.mechanism_discriminator_evidence_path = evidence_path

    # Per-mechanism field updates only when an isolation finding flagged
    # PRIMARY_MECHANISM or CONTRIBUTING_MECHANISM.
    findings = verdict.get("isolation_findings", []) or []
    fields_changed: list[str] = ["mechanism_discriminator_verdict"]
    if evidence_path:
        fields_changed.append("mechanism_discriminator_evidence_path")
    for f in findings:
        verdict_label = str(f.get("verdict", ""))
        hypothesis = str(f.get("mechanism_hypothesis", ""))
        if verdict_label not in ("PRIMARY_MECHANISM", "CONTRIBUTING_MECHANISM"):
            continue
        if hypothesis == "loader_byte_drift":
            # Use the observed delta R_pose as the scalar correction estimate.
            delta = f.get("delta_r_pose_vs_baseline")
            if delta is not None:
                profile.loader_drift_correction = float(delta)
                fields_changed.append("loader_drift_correction")
        elif hypothesis == "conv_kernel_accumulation_drift":
            profile.conv_kernel_determinism_required = True
            fields_changed.append("conv_kernel_determinism_required")
        elif hypothesis == "hydra_head_numerical_sensitivity":
            profile.head_quantize_post_inference_dtype = "uint8_round_multiple_of_2"
            fields_changed.append("head_quantize_post_inference_dtype")
        else:
            # Unknown mechanism hypothesis from a future discriminator
            # generation — preserve as a note rather than silently drop.
            note = (
                f"unknown discriminator hypothesis {hypothesis!r} "
                f"flagged {verdict_label} (evidence: {evidence_path})"
            )
            profile.notes = (profile.notes + " | " + note).strip(" |")
            fields_changed.append("notes")

    profile.last_updated_utc = datetime.now(UTC).isoformat()
    fields_changed.append("last_updated_utc")
    after = profile.to_dict()
    return {
        "architecture_class": arch_class,
        "before": before,
        "after": after,
        "fields_changed": fields_changed,
    }


# ── Adaptive-analyzer query helpers ────────────────────────────────────────
def query_profile_for_archive_class(
    architecture_class: str,
    *,
    registry: dict[str, ArchitectureProfile] | None = None,
) -> ArchitectureProfile:
    """Return the profile for ``architecture_class`` (or HNeRV defaults).

    Backwards-compatible path: callers that don't know the class get the
    HNeRV cluster — which is the historical default behaviour. Unknown
    classes return an ``unknown_uncalibrated`` profile with the prior
    HNeRV defaults.
    """
    if registry is None:
        registry = bootstrap_registry_from_hnerv_anchors()
    if architecture_class in registry:
        return registry[architecture_class]
    if "unknown_uncalibrated" in registry:
        return registry["unknown_uncalibrated"]
    return ArchitectureProfile(architecture_class=architecture_class)


def confidence_aware_score_band(
    *,
    architecture_class: str,
    cuda_score: float,
    registry: dict[str, ArchitectureProfile] | None = None,
) -> dict[str, Any]:
    """Public entry point for cathedral_autopilot / theoretical_floor / meta-Lagrangian.

    Returns a dict with ``predicted_cpu_score``,
    ``predicted_cpu_score_low``/``..._high``, ``confidence_label``,
    plus the architecture class actually used (so callers can detect
    fallback to ``unknown_uncalibrated``).
    """
    profile = query_profile_for_archive_class(
        architecture_class, registry=registry
    )
    band = profile.predict_cpu_score(cuda_score=cuda_score)
    band["architecture_class_used"] = profile.architecture_class
    band["n_anchors_backing"] = profile.n_anchors
    band["r_pose_mean"] = profile.r_pose_mean
    band["r_seg_mean"] = profile.r_seg_mean
    return band


__all__ = [
    "ARCHITECTURE_CLASSES",
    "DEFAULT_AUDIT_LOG_PATH",
    "DEFAULT_DECODER_POSE_DRIFT_FRACTION",
    "DEFAULT_DECODER_PROFILE",
    "DEFAULT_DECODER_SEG_DRIFT_FRACTION",
    "DEFAULT_POSE_FLOOR_ESTIMATE",
    "DEFAULT_REGISTRY_PATH",
    "LOW_CALIBRATION_ANCHOR_THRESHOLD",
    "LOW_CALIBRATION_BAND_WIDENING",
    "OUTLIER_SIGMA_THRESHOLD",
    "ArchitectureProfile",
    "DecoderProfile",
    "ProfileUpdate",
    "apply_discriminator_verdict_to_registry",
    "bootstrap_registry_from_hnerv_anchors",
    "classify_archive_into_profile",
    "confidence_aware_score_band",
    "decompose_observed_drift",
    "deserialize_registry",
    "harvest_new_anchor_and_update",
    "query_profile_for_archive_class",
    "read_registry",
    "serialize_registry",
    "update_profile_from_anchor",
    "write_registry",
]
