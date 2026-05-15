# SPDX-License-Identifier: MIT
"""CUDA-CPU axis diagnostic classifier — paradigm-shift #5 codec-class signature.

Per the Grand Reunion symposium 2026-05-15 Phase E Eureka #1 (Shannon):
the CUDA-CPU delta ``Δ = S_cuda - S_cpu`` is a CODEC-CLASS SIGNATURE, not
random noise. Empirical anchors at landing:

* A1 (within-class refinement): ``+0.034`` (CUDA 0.22635, CPU 0.1928)
* PR102 (within-class refinement / classmate of A1): ``+0.033``
* PR103 (within-class refinement): ``+0.029`` *(observed in paired ledger)*
* PR106 topk8 (residual-sidecar, cross-class): ``-0.021`` (REVERSED sign)

The reversed sign on PR106 topk8 indicates a different codec class — the
residual sidecar attaches to the existing within-class substrate but adds
side-information that the CUDA path's bit-faithful numerics consume more
faithfully than CPU. Per the Daubechies/Mallat geometric framing of
symposium Phase E Eureka #2: substrates that add LOCAL structure
(wavelet, scattering, predictive coding, residual sidecars) tend to flip
the sign of the gap relative to substrates that add GLOBAL structure
(within-class refinement, hyperprior bolt-ons).

Math contract
=============

Given empirical anchors ``{(S_cuda_i, S_cpu_i, class_i)}_{i=1..N}`` we fit a
1-D linear discriminant on the signed gap ``Δ_i = S_cuda_i - S_cpu_i``:

    class_predicted(Δ) = argmin_c |Δ - μ_c|

where ``μ_c`` is the empirical mean gap of class ``c``. Confidence is
``softmax(-|Δ - μ_c| / σ)`` with ``σ`` the pooled within-class standard
deviation. This is the canonical 1-D Gaussian discriminant per Bishop
``Pattern Recognition and Machine Learning`` §4.1.

[verified-against: Bishop 2006 §4.1 (linear discriminant for two classes
with shared variance); Fisher 1936 (the original linear discriminant
analysis). For signature pattern, see symposium Phase E Eureka #1.]

Implication
===========

If the classifier is reliable (e.g. ≥85% top-1 accuracy on held-out
anchors), we can INFER the codec class from a SINGLE-AXIS measurement
(e.g. CUDA only). This halves the paired-eval cost permanently per the
operator's "cost-band-as-Bayes" framing.

Per CLAUDE.md "Apples-to-apples evidence discipline": single-axis
inference is RESEARCH-ONLY until empirically validated against new paired
anchors. The classifier emits an explicit ``confidence`` field and a
``score_claim=False`` tag so downstream consumers do NOT promote
inferences as score-grade evidence.

Lane: ``lane_symposium_impl_cuda_cpu_classifier_20260515``.
Catalog #258.
"""
from __future__ import annotations

import dataclasses
import enum
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

__all__ = (
    "CUDA_CPU_CLASSIFIER_STATE_PATH",
    "CodecClass",
    "CodecClassificationResult",
    "CudaCpuAxisClassifier",
    "DEFAULT_EMPIRICAL_ANCHORS",
    "PairedAnchor",
    "build_default_classifier",
    "load_cached_classifier",
    "save_classifier",
    "update_from_anchor",
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
CUDA_CPU_CLASSIFIER_STATE_PATH: Final[Path] = (
    REPO_ROOT / ".omx" / "state" / "cuda_cpu_axis_diagnostic_classifier.json"
)


class CodecClass(str, enum.Enum):
    """The codec classes the symposium identified at the operating point.

    Per Phase E Eureka #2: substrates split by topology of their added bytes:

    * ``WITHIN_CLASS_REFINEMENT`` — refines an existing substrate without
      adding cross-class structure (A1, PR102, PR103). Positive Δ.
    * ``RESIDUAL_SIDECAR_CROSS_CLASS`` — adds local-structure side
      information to an existing substrate (PR106 topk8). Reversed Δ.
    * ``GLOBAL_BOLTON_HYPERPRIOR`` — adds global-dependency byte structure
      (Z3 Ballé hyperprior). Predicted positive Δ; regression-prone.
    * ``LOCAL_STRUCTURE_WAVELET`` — adds wavelet / scattering /
      predictive-coding local structure. Predicted reversed Δ.
    * ``UNKNOWN`` — classifier cannot determine; falls back to ``--full``
      paired-eval routing.
    """

    WITHIN_CLASS_REFINEMENT = "within_class_refinement"
    RESIDUAL_SIDECAR_CROSS_CLASS = "residual_sidecar_cross_class"
    GLOBAL_BOLTON_HYPERPRIOR = "global_bolton_hyperprior"
    LOCAL_STRUCTURE_WAVELET = "local_structure_wavelet"
    UNKNOWN = "unknown"


@dataclasses.dataclass(frozen=True)
class PairedAnchor:
    """One ``(CUDA, CPU)`` paired anchor for classifier fitting."""

    substrate_id: str
    score_cuda: float
    score_cpu: float
    codec_class: CodecClass

    @property
    def signed_gap(self) -> float:
        return self.score_cuda - self.score_cpu


@dataclasses.dataclass(frozen=True)
class CodecClassificationResult:
    """Output of one classification call."""

    predicted_class: CodecClass
    confidence: float
    signed_gap: float
    class_distances: tuple[tuple[CodecClass, float], ...]
    evidence_grade: str
    score_claim: bool
    notes: str


DEFAULT_EMPIRICAL_ANCHORS: Final[tuple[PairedAnchor, ...]] = (
    PairedAnchor("A1", 0.22635, 0.1928, CodecClass.WITHIN_CLASS_REFINEMENT),
    PairedAnchor("PR102", 0.22839, 0.19538, CodecClass.WITHIN_CLASS_REFINEMENT),
    PairedAnchor("PR103", 0.22500, 0.19610, CodecClass.WITHIN_CLASS_REFINEMENT),
    PairedAnchor(
        "PR106_topk8",
        0.20500,
        0.22600,
        CodecClass.RESIDUAL_SIDECAR_CROSS_CLASS,
    ),
)


@dataclasses.dataclass
class CudaCpuAxisClassifier:
    """1-D linear discriminant over the signed CUDA-CPU score gap.

    Per Bishop 2006 §4.1: with shared variance the class boundary is
    midway between adjacent class means; the predicted class is
    ``argmin_c |Δ - μ_c|``. This is the canonical Fisher LDA for 1-D
    features.
    """

    class_means: dict[CodecClass, float] = dataclasses.field(default_factory=dict)
    pooled_std: float = 0.01
    n_training_anchors: int = 0
    unknown_distance_threshold: float = 0.05  # tolerance band for ``UNKNOWN`` fallback

    @classmethod
    def fit(
        cls,
        anchors: Sequence[PairedAnchor],
        *,
        unknown_distance_threshold: float = 0.05,
    ) -> "CudaCpuAxisClassifier":
        """Fit class means and pooled std-dev from paired anchors."""
        if not anchors:
            return cls(unknown_distance_threshold=unknown_distance_threshold)
        per_class: dict[CodecClass, list[float]] = {}
        for anchor in anchors:
            per_class.setdefault(anchor.codec_class, []).append(anchor.signed_gap)
        class_means: dict[CodecClass, float] = {
            klass: sum(gaps) / len(gaps) for klass, gaps in per_class.items()
        }
        # Pooled standard deviation across classes (shared-variance LDA).
        residual_sq = 0.0
        n_resid = 0
        for klass, gaps in per_class.items():
            mu = class_means[klass]
            for g in gaps:
                residual_sq += (g - mu) ** 2
                n_resid += 1
        if n_resid > len(class_means):
            pooled_var = residual_sq / max(n_resid - len(class_means), 1)
            pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 0.005
        else:
            # Insufficient data to estimate within-class variance; use a
            # conservative prior from the symposium's observed gap range.
            pooled_std = 0.005
        return cls(
            class_means=class_means,
            pooled_std=max(pooled_std, 1e-6),
            n_training_anchors=len(anchors),
            unknown_distance_threshold=unknown_distance_threshold,
        )

    def classify(
        self,
        *,
        score_cuda: float | None = None,
        score_cpu: float | None = None,
        signed_gap: float | None = None,
    ) -> CodecClassificationResult:
        """Classify a (CUDA, CPU) pair OR a directly supplied signed gap.

        At least one of ``signed_gap`` OR both ``score_cuda``/``score_cpu``
        must be supplied. If ``signed_gap`` is provided directly, it takes
        precedence.
        """
        if signed_gap is None:
            if score_cuda is None or score_cpu is None:
                raise ValueError("Must supply signed_gap OR both score_cuda and score_cpu")
            signed_gap = float(score_cuda) - float(score_cpu)
        if not self.class_means:
            return CodecClassificationResult(
                predicted_class=CodecClass.UNKNOWN,
                confidence=0.0,
                signed_gap=signed_gap,
                class_distances=(),
                evidence_grade="research-only-prediction",
                score_claim=False,
                notes="[prediction; first-principles] no training anchors; UNKNOWN fallback. Catalog #258.",
            )
        distances = {klass: abs(signed_gap - mu) for klass, mu in self.class_means.items()}
        sorted_distances = sorted(distances.items(), key=lambda kv: kv[1])
        predicted, top_distance = sorted_distances[0]
        # Softmax confidence over distances under shared-variance Gaussian likelihood.
        log_likelihoods = {
            klass: -0.5 * (d / self.pooled_std) ** 2 for klass, d in distances.items()
        }
        max_ll = max(log_likelihoods.values())
        exps = {klass: math.exp(ll - max_ll) for klass, ll in log_likelihoods.items()}
        total = sum(exps.values()) or 1.0
        probs = {klass: e / total for klass, e in exps.items()}
        confidence = probs[predicted]
        # Fallback: if the top distance is large (predicted ≥ threshold above
        # the pooled std) we ALSO call UNKNOWN to keep the autopilot from
        # acting on a borderline classification.
        if top_distance > self.unknown_distance_threshold:
            predicted = CodecClass.UNKNOWN
            confidence = 0.0
        class_distance_tuple: tuple[tuple[CodecClass, float], ...] = tuple(sorted_distances)
        notes = (
            f"[prediction; first-principles] signed_gap={signed_gap:+.4f}; "
            f"pooled_std={self.pooled_std:.4f}; n_train={self.n_training_anchors}; "
            "Catalog #258."
        )
        return CodecClassificationResult(
            predicted_class=predicted,
            confidence=float(confidence),
            signed_gap=float(signed_gap),
            class_distances=class_distance_tuple,
            evidence_grade="research-only-prediction",
            score_claim=False,
            notes=notes,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "class_means": {klass.value: mu for klass, mu in self.class_means.items()},
            "pooled_std": self.pooled_std,
            "n_training_anchors": self.n_training_anchors,
            "unknown_distance_threshold": self.unknown_distance_threshold,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CudaCpuAxisClassifier":
        raw_means = payload.get("class_means", {})
        if not isinstance(raw_means, Mapping):
            raise ValueError("class_means must be a mapping")
        class_means: dict[CodecClass, float] = {}
        for k, v in raw_means.items():
            class_means[CodecClass(k)] = float(v)  # type: ignore[arg-type]
        return cls(
            class_means=class_means,
            pooled_std=float(payload.get("pooled_std", 0.005)),  # type: ignore[arg-type]
            n_training_anchors=int(payload.get("n_training_anchors", 0)),  # type: ignore[arg-type]
            unknown_distance_threshold=float(payload.get("unknown_distance_threshold", 0.05)),  # type: ignore[arg-type]
        )


def build_default_classifier() -> CudaCpuAxisClassifier:
    """Fit a classifier on the canonical empirical anchors known at landing."""
    return CudaCpuAxisClassifier.fit(DEFAULT_EMPIRICAL_ANCHORS)


def save_classifier(
    classifier: CudaCpuAxisClassifier, *, state_path: Path | None = None
) -> Path:
    target = Path(state_path) if state_path is not None else CUDA_CPU_CLASSIFIER_STATE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(classifier.to_dict(), indent=2, sort_keys=True))
    tmp.replace(target)
    return target


def load_cached_classifier(*, state_path: Path | None = None) -> CudaCpuAxisClassifier | None:
    target = Path(state_path) if state_path is not None else CUDA_CPU_CLASSIFIER_STATE_PATH
    if not target.is_file():
        return None
    return CudaCpuAxisClassifier.from_dict(json.loads(target.read_text()))


def update_from_anchor(
    anchor: Mapping[str, object],
    *,
    state_path: Path | None = None,
) -> CudaCpuAxisClassifier | None:
    """Re-fit the classifier including a new paired anchor.

    The anchor must carry ``substrate_id``, ``score_cuda``, ``score_cpu``,
    and ``codec_class`` (as a :class:`CodecClass` value or its string name).
    """
    needed = {"substrate_id", "score_cuda", "score_cpu", "codec_class"}
    if not needed.issubset(anchor):
        return None
    try:
        klass_raw = anchor["codec_class"]
        klass = klass_raw if isinstance(klass_raw, CodecClass) else CodecClass(str(klass_raw))
        paired = PairedAnchor(
            substrate_id=str(anchor["substrate_id"]),
            score_cuda=float(anchor["score_cuda"]),  # type: ignore[arg-type]
            score_cpu=float(anchor["score_cpu"]),  # type: ignore[arg-type]
            codec_class=klass,
        )
    except (TypeError, ValueError):
        return None
    target = Path(state_path) if state_path is not None else CUDA_CPU_CLASSIFIER_STATE_PATH
    # Append to the canonical anchor set + re-fit.
    anchors = list(DEFAULT_EMPIRICAL_ANCHORS) + [paired]
    classifier = CudaCpuAxisClassifier.fit(anchors)
    save_classifier(classifier, state_path=target)
    return classifier
