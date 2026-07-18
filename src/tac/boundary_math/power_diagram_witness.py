# SPDX-License-Identifier: MIT
"""Frozen-head power-diagram targets and strict ``PDW1`` custody.

The affine/power identity is exact in real arithmetic before the mandatory
float32 target serialization.  ``PDW1`` is byte-close and canonical, but its
float32 coefficients can make an originally exact near-tie numerically
ambiguous; it is not claimed to preserve every boundary tie.  This module works
in the argmax-relevant quotient of the frozen affine segmentation head.  It
does not invert the nonlinear feature-field pullback and does not claim that a
channel-space target is a spatial partition codec.
Labels without paired channel features are therefore rejected by the inverse.

Authority labels used by the receipts are intentionally literal:

* affine/head conversion and ``PDW1`` parse-back are ``DERIVED``;
* cache statistics and file sizes/hashes are ``MEASURED``;
* renderer, through-R, Seg, Pose, score, and spatial-K claims are absent.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

PDW1_MAGIC: Final = b"PDW1"
PDW1_HEADER: Final = struct.Struct("<4sHHI")
TARGET_COMPARISON_VERDICT: Final = "TARGET_ONLY_VS_REALIZATION_NON_EQUIVALENT"


class PowerDiagramWitnessError(ValueError):
    """Raised when target geometry, custody, or bytes violate the contract."""


@dataclass(frozen=True)
class PowerDiagramTarget:
    """Canonical weighted sites plus observed active-facet certificates.

    ``adjacency`` contains class-id pairs in lexicographic order.  The matching
    row in ``tie_normals``/``tie_offsets`` is the exact rank-quotient tie
    hyperplane, not a spatial curve or a full-channel-space normal.
    """

    class_ids: np.ndarray
    sites: np.ndarray
    weights: np.ndarray
    adjacency: tuple[tuple[int, int], ...]
    tie_normals: np.ndarray
    tie_offsets: np.ndarray

    @property
    def n_classes(self) -> int:
        return int(self.class_ids.size)

    @property
    def rank(self) -> int:
        return int(self.sites.shape[1])


@dataclass(frozen=True)
class HeadPowerDiagram:
    """Real-arithmetic affine quotient and its serialized-float32 target."""

    target: PowerDiagramTarget
    quotient_basis: np.ndarray
    centered_weight: np.ndarray
    centered_bias: np.ndarray
    singular_values: np.ndarray
    common_gauge: float


@dataclass(frozen=True)
class InverseFitReceipt:
    """Paired-feature, strictly regularized affine-target fit receipt."""

    head: HeadPowerDiagram
    affine_weight: np.ndarray
    affine_bias: np.ndarray
    regularization: float
    sample_count: int
    sample_agreement: float
    minimum_true_label_margin: float
    residual_rms: float
    objective: float
    exact_on_samples: bool
    authority_label: str = "DERIVED_PAIRED_FEATURE_TARGET_FIT_NOT_RENDERER_INVERSE"


@dataclass(frozen=True)
class F32ParityReceipt:
    """Measured numerical parity of one serialized target on supplied features."""

    sample_count: int
    mismatch_count: int
    sample_agreement: float
    max_pair_score_error: float
    minimum_affine_winner_margin: float
    f32_tie_uncertain_count: int
    exact_on_samples: bool
    boundary_exactness: str = "NO_GENERAL_VERDICT_WITHIN_F32_TIE_UNCERTAINTY"
    authority_label: str = "MEASURED_NUMERICAL_PARITY_NOT_BOUNDARY_THEOREM"


@dataclass(frozen=True)
class VideoFedTargetReceipt:
    """Cached-label statistics paired with frozen-head-derived float32 sites."""

    target: PowerDiagramTarget
    active_classes: tuple[int, ...]
    class_counts: tuple[int, ...]
    adjacency: tuple[tuple[int, int], ...]
    selected_partition_sha256: str
    frozen_head_sha256: str
    selected_shape: tuple[int, ...]
    selected_dtype: str
    selected_partitions: int
    labels_member: str
    authority_label: str = "MEASURED_CACHED_LABELS_DERIVED_FROZEN_HEAD_TARGET_ADVISORY"


@dataclass(frozen=True)
class FileSizeReceipt:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class DescriptionLengthReceipt:
    """Target bytes versus named realization files, without equivalence claims."""

    target_bytes: int
    target_sha256: str
    realization_files: tuple[FileSizeReceipt, ...]
    verdict: str = TARGET_COMPARISON_VERDICT
    spatial_k_lower_bound: str = "NO_VERDICT"
    renderer_authority: str = "NOT_RUN_NO_AUTHORITY"
    score_claim: bool = False
    archive_saving_claim: bool = False


def _as_finite_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim < 2:
        raise PowerDiagramWitnessError(f"{name} must have at least two dimensions")
    out = out.reshape(out.shape[0], -1)
    if not np.isfinite(out).all():
        raise PowerDiagramWitnessError(f"{name} contains non-finite values")
    return out


def _canonical_zero_f32(value: np.ndarray) -> np.ndarray:
    out = np.asarray(value, dtype="<f4").copy()
    out[out == 0] = 0.0  # canonical positive zero
    return out


def sha256_file(path: str | Path, *, chunk_bytes: int = 1 << 20) -> str:
    """Hash exact file bytes without materializing the file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_row_difference_basis(weight: np.ndarray, *, relative_tolerance: float | None = None) -> np.ndarray:
    """Return a deterministic orthonormal basis for ordered head-row differences.

    Ordered, twice-reorthogonalized Gram-Schmidt avoids arbitrary singular-vector
    rotations while still yielding the exact row-difference quotient.  Each basis
    vector has its largest-magnitude coordinate positive.
    """

    rows = _as_finite_matrix(weight, name="weight")
    if rows.shape[0] < 2:
        raise PowerDiagramWitnessError("at least two head rows are required")
    differences = rows[1:] - rows[0]
    scale = float(np.linalg.norm(differences, axis=1).max(initial=0.0))
    if scale == 0:
        raise PowerDiagramWitnessError("head row-difference rank is zero")
    if relative_tolerance is None:
        # Drop numerical zero only.  A looser scientific rank threshold can
        # erase a small but decision-relevant direction under a large dual
        # feature coordinate.  Explicit tolerances remain caller-scoped and
        # their effect is measurable through F32ParityReceipt.
        threshold = np.finfo(np.float64).eps * scale
    else:
        if not math.isfinite(relative_tolerance) or relative_tolerance <= 0:
            raise PowerDiagramWitnessError("relative_tolerance must be finite and > 0")
        threshold = relative_tolerance * scale
    basis: list[np.ndarray] = []
    for difference in differences:
        vector = difference.copy()
        for _ in range(2):
            for prior in basis:
                vector -= prior * float(prior @ vector)
        norm = float(np.linalg.norm(vector))
        if norm <= threshold:
            continue
        vector /= norm
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0:
            vector = -vector
        basis.append(vector)
    if not basis:
        raise PowerDiagramWitnessError("head row-difference rank is zero")
    result = np.stack(basis, axis=1)
    gram = result.T @ result
    if not np.allclose(gram, np.eye(result.shape[1]), atol=5e-12, rtol=5e-12):
        raise PowerDiagramWitnessError("failed to construct an orthonormal quotient basis")
    return result


def canonical_adjacency(adjacency: Iterable[tuple[int, int]], class_ids: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Canonicalize undirected class-id edges as sorted unique ``i < j`` pairs."""

    allowed = {int(value) for value in class_ids}
    edges: set[tuple[int, int]] = set()
    for raw_i, raw_j in adjacency:
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in (raw_i, raw_j)
        ):
            raise PowerDiagramWitnessError("adjacency class ids must be lossless integers")
        i, j = sorted((int(raw_i), int(raw_j)))
        if i == j:
            raise PowerDiagramWitnessError("adjacency self-edges are forbidden")
        if i not in allowed or j not in allowed:
            raise PowerDiagramWitnessError("adjacency references an unknown class id")
        edges.add((i, j))
    return tuple(sorted(edges))


def _tie_loci(
    class_ids: np.ndarray,
    sites: np.ndarray,
    weights: np.ndarray,
    adjacency: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    index = {int(class_id): offset for offset, class_id in enumerate(class_ids)}
    normals = np.empty((len(adjacency), sites.shape[1]), dtype=np.float64)
    offsets = np.empty(len(adjacency), dtype=np.float64)
    for edge_index, (class_i, class_j) in enumerate(adjacency):
        i, j = index[class_i], index[class_j]
        normals[edge_index] = 2.0 * (sites[j] - sites[i])
        offsets[edge_index] = (
            float(sites[i] @ sites[i]) - float(sites[j] @ sites[j]) - float(weights[i]) + float(weights[j])
        )
    return _canonical_zero_f32(normals), _canonical_zero_f32(offsets)


def make_power_diagram_target(
    sites: np.ndarray,
    weights: np.ndarray,
    *,
    class_ids: Sequence[int] | None = None,
    adjacency: Iterable[tuple[int, int]] | None = None,
) -> PowerDiagramTarget:
    """Build a canonical target and derive its explicit adjacency tie loci."""

    sites_f64 = np.asarray(sites, dtype=np.float64)
    weights_f64 = np.asarray(weights, dtype=np.float64)
    if sites_f64.ndim != 2 or sites_f64.shape[0] < 2 or sites_f64.shape[1] < 1:
        raise PowerDiagramWitnessError("sites must have shape (K, rank), K >= 2, rank >= 1")
    if weights_f64.shape != (sites_f64.shape[0],):
        raise PowerDiagramWitnessError("weights must have shape (K,)")
    if not np.isfinite(sites_f64).all() or not np.isfinite(weights_f64).all():
        raise PowerDiagramWitnessError("sites/weights contain non-finite values")
    if class_ids is None:
        ids = np.arange(sites_f64.shape[0], dtype="<u2")
    else:
        raw_values = tuple(class_ids)
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in raw_values):
            raise PowerDiagramWitnessError("class ids must be lossless integers")
        raw_ids = tuple(int(value) for value in raw_values)
        if len(raw_ids) != sites_f64.shape[0]:
            raise PowerDiagramWitnessError("class_ids length must equal site count")
        if any(value < 0 or value > 0xFFFF for value in raw_ids):
            raise PowerDiagramWitnessError("class ids must fit uint16")
        if tuple(sorted(set(raw_ids))) != raw_ids:
            raise PowerDiagramWitnessError("class ids must be strictly increasing")
        ids = np.asarray(raw_ids, dtype="<u2")
    if adjacency is None:
        edges = tuple((int(ids[i]), int(ids[j])) for i in range(ids.size) for j in range(i + 1, ids.size))
    else:
        edges = canonical_adjacency(adjacency, ids)
    sites_f32 = _canonical_zero_f32(sites_f64)
    weights_f32 = _canonical_zero_f32(weights_f64)
    normals, offsets = _tie_loci(ids, sites_f32, weights_f32, edges)
    return PowerDiagramTarget(
        class_ids=ids,
        sites=sites_f32,
        weights=weights_f32,
        adjacency=edges,
        tie_normals=normals,
        tie_offsets=offsets,
    )


def affine_head_to_power_diagram(
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    class_ids: Sequence[int] | None = None,
    adjacency: Iterable[tuple[int, int]] | None = None,
    common_gauge: float | None = None,
    relative_tolerance: float | None = None,
) -> HeadPowerDiagram:
    """Derive the exact real-arithmetic quotient, then its float32 target.

    The common row and common bias are removed because they cancel in argmax.
    ``common_gauge`` shifts every power weight equally and therefore also leaves
    every cell unchanged.  If omitted, it is chosen to center the power weights.
    """

    rows = _as_finite_matrix(weight, name="weight")
    bias64 = np.asarray(bias, dtype=np.float64)
    if bias64.shape != (rows.shape[0],) or not np.isfinite(bias64).all():
        raise PowerDiagramWitnessError("bias must be a finite vector with one value per class")
    basis = canonical_row_difference_basis(rows, relative_tolerance=relative_tolerance)
    centered_rows = rows - rows.mean(axis=0, keepdims=True)
    centered_bias = bias64 - bias64.mean()
    projected_rows = centered_rows @ basis
    sites = projected_rows / 2.0
    ungauged = centered_bias + np.sum(sites * sites, axis=1)
    gauge = float(ungauged.mean()) if common_gauge is None else float(common_gauge)
    if not math.isfinite(gauge):
        raise PowerDiagramWitnessError("common_gauge must be finite")
    weights = ungauged - gauge
    target = make_power_diagram_target(
        sites,
        weights,
        class_ids=class_ids,
        adjacency=adjacency,
    )
    singular_values = np.linalg.svd(centered_rows, compute_uv=False)
    return HeadPowerDiagram(
        target=target,
        quotient_basis=basis,
        centered_weight=centered_rows,
        centered_bias=centered_bias,
        singular_values=singular_values,
        common_gauge=gauge,
    )


def project_channel_features(features: np.ndarray, quotient_basis: np.ndarray) -> np.ndarray:
    """Project paired flattened channel features into the rank quotient."""

    values = np.asarray(features, dtype=np.float64)
    basis = np.asarray(quotient_basis, dtype=np.float64)
    if basis.ndim != 2 or values.ndim < 1 or values.shape[-1] != basis.shape[0]:
        raise PowerDiagramWitnessError("features' final dimension must match quotient basis rows")
    if not np.isfinite(values).all() or not np.isfinite(basis).all():
        raise PowerDiagramWitnessError("features/basis contain non-finite values")
    # ``einsum`` avoids platform-BLAS floating-status leakage while preserving
    # the deterministic NumPy reference contraction.
    return np.einsum("...d,dr->...r", values, basis, optimize=False)


def power_distances(points: np.ndarray, target: PowerDiagramTarget) -> np.ndarray:
    """Classical power distances ``||z-s_c||^2 - omega_c``."""

    z = np.asarray(points, dtype=np.float64)
    if z.ndim < 1 or z.shape[-1] != target.rank or not np.isfinite(z).all():
        raise PowerDiagramWitnessError("points must be finite with final dimension target.rank")
    sites = np.asarray(target.sites, dtype=np.float64)
    weights = np.asarray(target.weights, dtype=np.float64)
    return np.sum((z[..., None, :] - sites) ** 2, axis=-1) - weights


def power_scores(points: np.ndarray, target: PowerDiagramTarget) -> np.ndarray:
    """Stable argmax-equivalent scores with the common ``-||z||²`` removed."""

    z = np.asarray(points, dtype=np.float64)
    if z.ndim < 1 or z.shape[-1] != target.rank or not np.isfinite(z).all():
        raise PowerDiagramWitnessError("points must be finite with final dimension target.rank")
    sites = np.asarray(target.sites, dtype=np.float64)
    weights = np.asarray(target.weights, dtype=np.float64)
    return 2.0 * np.einsum("...r,kr->...k", z, sites, optimize=False) + weights - np.sum(sites * sites, axis=1)


def power_assign(points: np.ndarray, target: PowerDiagramTarget) -> np.ndarray:
    """Nearest-weighted-site class ids with NumPy's deterministic first-min tie rule."""

    # Maximizing the expanded score is exactly nearest-power assignment after
    # removing the common -||z||² term, and avoids catastrophic cancellation
    # for a large quotient coordinate near a small separating normal.
    indices = np.argmax(power_scores(points, target), axis=-1)
    return target.class_ids[indices]


def measure_f32_target_parity(features: np.ndarray, head: HeadPowerDiagram) -> F32ParityReceipt:
    """Measure float32 target parity and expose its near-tie uncertainty band.

    The comparison uses centered affine logits (the real-arithmetic source) and
    the serialized float32 site's power scores.  ``max_pair_score_error`` is
    the maximum absolute error over all class-pair score differences on the
    supplied features; an affine winner margin inside that bound is explicitly
    counted as tie-uncertain, even when the sampled argmax happens to agree.
    """

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != head.centered_weight.shape[1]:
        raise PowerDiagramWitnessError("features must have shape (N, original_head_dimension)")
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise PowerDiagramWitnessError("features must be nonempty and finite")
    affine_scores = np.einsum("nd,kd->nk", values, head.centered_weight, optimize=False) + head.centered_bias
    points = project_channel_features(values, head.quotient_basis)
    target_scores = power_scores(points, head.target)
    affine_indices = np.argmax(affine_scores, axis=1)
    target_indices = np.argmax(target_scores, axis=1)
    mismatch_count = int(np.count_nonzero(affine_indices != target_indices))
    pair_error = 0.0
    for i in range(head.target.n_classes):
        for j in range(i + 1, head.target.n_classes):
            affine_difference = affine_scores[:, j] - affine_scores[:, i]
            target_difference = target_scores[:, j] - target_scores[:, i]
            pair_error = max(
                pair_error,
                float(np.max(np.abs(affine_difference - target_difference))),
            )
    top_two = np.partition(affine_scores, -2, axis=1)[:, -2:]
    winner_margins = top_two[:, 1] - top_two[:, 0]
    uncertain_count = int(np.count_nonzero(winner_margins <= pair_error)) if pair_error > 0 else 0
    return F32ParityReceipt(
        sample_count=int(values.shape[0]),
        mismatch_count=mismatch_count,
        sample_agreement=float(1.0 - mismatch_count / values.shape[0]),
        max_pair_score_error=pair_error,
        minimum_affine_winner_margin=float(np.min(winner_margins)),
        f32_tie_uncertain_count=uncertain_count,
        exact_on_samples=mismatch_count == 0,
    )


def pair_tie_value(points: np.ndarray, target: PowerDiagramTarget, class_i: int, class_j: int) -> np.ndarray:
    """Evaluate the canonical ``i<j`` pair tie hyperplane at quotient points."""

    i, j = sorted((int(class_i), int(class_j)))
    try:
        edge_index = target.adjacency.index((i, j))
    except ValueError as exc:
        raise PowerDiagramWitnessError("requested pair is not in target adjacency") from exc
    z = np.asarray(points, dtype=np.float64)
    return z @ np.asarray(target.tie_normals[edge_index], dtype=np.float64) + float(target.tie_offsets[edge_index])


def is_co_maximum_tie(
    points: np.ndarray,
    target: PowerDiagramTarget,
    class_i: int,
    class_j: int,
    *,
    absolute_tolerance: float = 1e-7,
) -> np.ndarray:
    """True only where the pair ties *and* co-dominates every other class."""

    if int(class_i) == int(class_j):
        raise PowerDiagramWitnessError("co-maximum tie requires two distinct classes")
    if absolute_tolerance < 0 or not math.isfinite(absolute_tolerance):
        raise PowerDiagramWitnessError("absolute_tolerance must be finite and >= 0")
    class_to_index = {int(value): index for index, value in enumerate(target.class_ids)}
    try:
        i, j = class_to_index[int(class_i)], class_to_index[int(class_j)]
    except KeyError as exc:
        raise PowerDiagramWitnessError("unknown class id") from exc
    scores = power_scores(points, target)
    tied = np.abs(scores[..., i] - scores[..., j]) <= absolute_tolerance
    tied_score = np.maximum(scores[..., i], scores[..., j])
    dominant = tied_score >= np.max(scores, axis=-1) - absolute_tolerance
    return tied & dominant


def fit_power_diagram_from_paired_features(
    paired_features: np.ndarray | None,
    target_labels: np.ndarray,
    *,
    n_classes: int | None = None,
    regularization: float = 1e-6,
    adjacency: Iterable[tuple[int, int]] | None = None,
) -> InverseFitReceipt:
    """Fit the unique ridge-regularized affine target from paired features.

    This is a deterministic multiresponse ridge solve against centered one-hot
    targets.  It is not an inverse through the convolutional renderer.
    """

    if paired_features is None:
        raise PowerDiagramWitnessError("labels-only fitting is underdetermined; paired channel features are required")
    x = np.asarray(paired_features, dtype=np.float64)
    labels = np.asarray(target_labels)
    if x.ndim != 2 or x.shape[0] == 0 or not np.isfinite(x).all():
        raise PowerDiagramWitnessError("paired_features must be a nonempty finite (N, D) matrix")
    if labels.shape != (x.shape[0],) or labels.dtype.kind not in "iu":
        raise PowerDiagramWitnessError("target_labels must be an integer vector paired with features")
    inferred = int(labels.max(initial=-1)) + 1
    classes = inferred if n_classes is None else int(n_classes)
    if classes < 2 or np.any(labels < 0) or np.any(labels >= classes):
        raise PowerDiagramWitnessError("target labels fall outside the declared class range")
    ridge = float(regularization)
    if not math.isfinite(ridge) or ridge <= 0:
        raise PowerDiagramWitnessError("regularization must be finite and strictly > 0")
    design = np.concatenate((x, np.ones((x.shape[0], 1), dtype=np.float64)), axis=1)
    desired = np.eye(classes, dtype=np.float64)[labels.astype(np.int64)]
    desired -= 1.0 / classes
    normal = design.T @ design + ridge * np.eye(design.shape[1], dtype=np.float64)
    coefficients = np.linalg.solve(normal, design.T @ desired)
    fitted = design @ coefficients
    residual = fitted - desired
    residual_rms = float(np.sqrt(np.mean(residual * residual)))
    objective = float(np.sum(residual * residual) + ridge * np.sum(coefficients * coefficients))
    weight = coefficients[:-1].T
    bias = coefficients[-1]
    head = affine_head_to_power_diagram(weight, bias, adjacency=adjacency)
    # Receipt authority is the converted float32 target, not the pre-conversion
    # float64 affine logits.  This catches any rank projection or target-byte
    # quantization loss before exact_on_samples can become true.
    points = project_channel_features(x, head.quotient_basis)
    prediction = power_assign(points, head.target).astype(np.int64, copy=False)
    agreement = float(np.mean(prediction == labels))
    target_scores = power_scores(points, head.target)
    label_indices = labels.astype(np.int64)
    true_scores = target_scores[np.arange(labels.size), label_indices]
    rivals = target_scores.copy()
    rivals[np.arange(labels.size), label_indices] = -np.inf
    margins = true_scores - np.max(rivals, axis=1)
    return InverseFitReceipt(
        head=head,
        affine_weight=weight,
        affine_bias=bias,
        regularization=ridge,
        sample_count=int(labels.size),
        sample_agreement=agreement,
        minimum_true_label_margin=float(np.min(margins)),
        residual_rms=residual_rms,
        objective=objective,
        exact_on_samples=bool(np.all(prediction == labels)),
    )


def fit_power_diagram_from_labels(
    target_labels: np.ndarray, *, paired_features: np.ndarray | None = None, **kwargs: object
) -> InverseFitReceipt:
    """Fail closed unless real paired channel features accompany the labels."""

    return fit_power_diagram_from_paired_features(
        paired_features,
        target_labels,
        **kwargs,
    )


def open_stored_npy_memmap(npz_path: str | Path, key: str = "lstars") -> np.memmap:
    """Memory-map one ``ZIP_STORED`` NPY member without inflating its NPZ."""

    path = Path(npz_path)
    member = key if key.endswith(".npy") else f"{key}.npy"
    try:
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo(member)
            if info.compress_type != zipfile.ZIP_STORED:
                raise PowerDiagramWitnessError(f"{path}:{member} is compressed; zero-copy read is unavailable")
            if info.flag_bits & 0x1:
                raise PowerDiagramWitnessError("encrypted NPY members are unsupported")
            if info.compress_size != info.file_size:
                raise PowerDiagramWitnessError("stored NPY central-directory sizes disagree")
            local_header = int(info.header_offset)
            member_size = int(info.file_size)
            central_flags = int(info.flag_bits)
            central_method = int(info.compress_type)
            central_filename = info.filename
            central_crc = int(info.CRC)
            central_compressed_size = int(info.compress_size)
    except (KeyError, zipfile.BadZipFile) as exc:
        raise PowerDiagramWitnessError(f"invalid NPZ or missing member {member!r}") from exc
    with path.open("rb") as handle:
        handle.seek(local_header)
        header = handle.read(30)
        if len(header) != 30:
            raise PowerDiagramWitnessError("truncated local ZIP header")
        fields = struct.unpack("<IHHHHHIIIHH", header)
        if fields[0] != 0x04034B50:
            raise PowerDiagramWitnessError("bad local ZIP header signature")
        local_flags, local_method = int(fields[2]), int(fields[3])
        if local_flags != central_flags or local_method != central_method:
            raise PowerDiagramWitnessError("local/central ZIP flags or compression method disagree")
        filename_size, extra_size = int(fields[-2]), int(fields[-1])
        local_filename = handle.read(filename_size)
        local_extra = handle.read(extra_size)
        if len(local_filename) != filename_size or len(local_extra) != extra_size:
            raise PowerDiagramWitnessError("truncated local ZIP filename or extra fields")
        expected_filename = central_filename.encode("utf-8" if local_flags & 0x800 else "cp437")
        if local_filename != expected_filename:
            raise PowerDiagramWitnessError("local/central ZIP member names disagree")
        if not local_flags & 0x08:
            local_crc = int(fields[6])
            local_compressed_size = int(fields[7])
            local_uncompressed_size = int(fields[8])
            zip64_payload: bytes | None = None
            extra_offset = 0
            while extra_offset < len(local_extra):
                if extra_offset + 4 > len(local_extra):
                    raise PowerDiagramWitnessError("truncated local ZIP extra-field header")
                extra_id, payload_size = struct.unpack_from("<HH", local_extra, extra_offset)
                extra_offset += 4
                if extra_offset + payload_size > len(local_extra):
                    raise PowerDiagramWitnessError("truncated local ZIP extra-field payload")
                if extra_id == 0x0001:
                    if zip64_payload is not None:
                        raise PowerDiagramWitnessError("duplicate local ZIP64 extra field")
                    zip64_payload = local_extra[extra_offset : extra_offset + payload_size]
                extra_offset += payload_size

            zip64_offset = 0

            def resolve_zip64_size(raw_size: int, *, field_name: str) -> int:
                nonlocal zip64_offset
                if raw_size != 0xFFFFFFFF:
                    return raw_size
                if zip64_payload is None or zip64_offset + 8 > len(zip64_payload):
                    raise PowerDiagramWitnessError(f"missing local ZIP64 {field_name} size")
                value = struct.unpack_from("<Q", zip64_payload, zip64_offset)[0]
                zip64_offset += 8
                return int(value)

            local_uncompressed_size = resolve_zip64_size(local_uncompressed_size, field_name="uncompressed")
            local_compressed_size = resolve_zip64_size(local_compressed_size, field_name="compressed")
            if (
                local_crc != central_crc
                or local_compressed_size != central_compressed_size
                or local_uncompressed_size != member_size
            ):
                raise PowerDiagramWitnessError("local/central ZIP CRC or sizes disagree")
        npy_start = local_header + 30 + filename_size + extra_size
        if npy_start + member_size > path.stat().st_size:
            raise PowerDiagramWitnessError("stored NPY member extent exceeds the container")
        handle.seek(npy_start)
        try:
            version = np.lib.format.read_magic(handle)
            shape, fortran_order, dtype = np.lib.format._read_array_header(handle, version)
        except (EOFError, ValueError) as exc:
            raise PowerDiagramWitnessError("invalid embedded NPY header") from exc
        data_offset = handle.tell()
    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise PowerDiagramWitnessError("object arrays cannot be memory-mapped safely")
    expected_data_bytes = math.prod(shape) * dtype.itemsize
    if expected_data_bytes < 0 or expected_data_bytes > member_size:
        raise PowerDiagramWitnessError("NPY array extent exceeds its stored member")
    if data_offset + expected_data_bytes != npy_start + member_size:
        raise PowerDiagramWitnessError("NPY member size/header mismatch")
    return np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=data_offset,
        shape=shape,
        order="F" if fortran_order else "C",
    )


_SAFETENSOR_DTYPES: Final = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
}


def read_safetensors_tensors(path: str | Path, tensor_names: Sequence[str]) -> dict[str, np.ndarray]:
    """Read only named tensors from safetensors without importing a scorer."""

    file_path = Path(path)
    file_size = file_path.stat().st_size
    with file_path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise PowerDiagramWitnessError("truncated safetensors length prefix")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size < 2 or header_size > file_size - 8:
            raise PowerDiagramWitnessError("invalid safetensors header length")
        header_bytes = handle.read(header_size)

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise PowerDiagramWitnessError(f"duplicate safetensors JSON key {key!r}")
            result[key] = value
        return result

    try:
        if not header_bytes.startswith(b"{"):
            raise PowerDiagramWitnessError("safetensors header must begin with an object")
        header = json.loads(header_bytes, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PowerDiagramWitnessError("invalid safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise PowerDiagramWitnessError("safetensors header must be an object")
    data_start = 8 + header_size
    validated: dict[str, tuple[np.dtype, tuple[int, ...], int, int]] = {}
    extents: list[tuple[int, int, str]] = []
    for name, metadata in header.items():
        if name == "__metadata__":
            if not isinstance(metadata, dict):
                raise PowerDiagramWitnessError("safetensors __metadata__ must be an object")
            continue
        if not isinstance(metadata, dict):
            raise PowerDiagramWitnessError(f"malformed metadata for tensor {name!r}")
        dtype_name = metadata.get("dtype")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if dtype_name not in _SAFETENSOR_DTYPES:
            raise PowerDiagramWitnessError(f"unsupported safetensors dtype {dtype_name!r}")
        if (
            not isinstance(shape, list)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
        ):
            raise PowerDiagramWitnessError(f"malformed metadata for tensor {name!r}")
        start, stop = offsets
        dtype = _SAFETENSOR_DTYPES[dtype_name]
        expected = math.prod(shape) * dtype.itemsize
        if start < 0 or stop < start or stop - start != expected or data_start + stop > file_size:
            raise PowerDiagramWitnessError(f"invalid data extent for tensor {name!r}")
        validated[name] = (dtype, tuple(shape), start, stop)
        extents.append((start, stop, name))
    cursor = 0
    for start, stop, name in sorted(extents):
        if start != cursor:
            relation = "overlapping" if start < cursor else "unindexed gap before"
            raise PowerDiagramWitnessError(f"{relation} safetensors tensor {name!r}")
        cursor = stop
    if cursor != file_size - data_start:
        raise PowerDiagramWitnessError("safetensors has trailing unindexed data")
    result: dict[str, np.ndarray] = {}
    for name in tensor_names:
        if name not in validated:
            raise PowerDiagramWitnessError(f"safetensors tensor {name!r} is missing")
        dtype, shape, start, _stop = validated[name]
        mapped = np.memmap(
            file_path,
            dtype=dtype,
            mode="r",
            offset=data_start + start,
            shape=shape,
            order="C",
        )
        result[name] = np.asarray(mapped).copy()
    return result


def read_frozen_segmentation_head(
    path: str | Path,
    *,
    weight_key: str = "segmentation_head.0.weight",
    bias_key: str = "segmentation_head.0.bias",
) -> tuple[np.ndarray, np.ndarray]:
    """Read and validate the frozen affine head tensors only."""

    tensors = read_safetensors_tensors(path, (weight_key, bias_key))
    weight = np.asarray(tensors[weight_key])
    bias = np.asarray(tensors[bias_key])
    if weight.ndim < 2 or bias.shape != (weight.shape[0],):
        raise PowerDiagramWitnessError("frozen segmentation head weight/bias shapes are inconsistent")
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise PowerDiagramWitnessError("frozen segmentation head contains non-finite values")
    return weight, bias


def observed_four_neighbour_adjacency(labels: np.ndarray, *, n_classes: int) -> tuple[tuple[int, int], ...]:
    """Measure sorted unique undirected class adjacency on one 2-D partition."""

    frame = np.asarray(labels)
    if frame.ndim != 2 or frame.dtype.kind not in "iu":
        raise PowerDiagramWitnessError("each cached partition must be a 2-D integer array")
    if np.any(frame < 0) or np.any(frame >= n_classes):
        raise PowerDiagramWitnessError("cached partition contains out-of-range labels")
    edges: set[tuple[int, int]] = set()
    for first, second in ((frame[:, :-1], frame[:, 1:]), (frame[:-1, :], frame[1:, :])):
        changed = first != second
        if not np.any(changed):
            continue
        pairs = np.stack((first[changed], second[changed]), axis=1).astype(np.int64, copy=False)
        pairs.sort(axis=1)
        edges.update((int(i), int(j)) for i, j in np.unique(pairs, axis=0))
    return tuple(sorted(edges))


def _selected_indices(length: int, selection: slice | Sequence[int] | None) -> tuple[int, ...]:
    if selection is None:
        return tuple(range(length))
    if isinstance(selection, slice):
        return tuple(range(*selection.indices(length)))
    indices = tuple(int(value) for value in selection)
    normalized = tuple(value + length if value < 0 else value for value in indices)
    if any(value < 0 or value >= length for value in normalized):
        raise PowerDiagramWitnessError("partition selection index is out of range")
    if len(set(normalized)) != len(normalized):
        raise PowerDiagramWitnessError("partition selection must not repeat indices")
    return normalized


def initialize_video_fed_target(
    cache_path: str | Path,
    frozen_head_path: str | Path,
    *,
    labels_key: str = "lstars",
    selection: slice | Sequence[int] | None = None,
) -> VideoFedTargetReceipt:
    """Build a cached-``L*`` receipt while placing sites from the frozen head.

    Counts, adjacency, and digest come only from cached labels.  Sites and
    weights come only from the frozen affine head.  No feature inverse is
    claimed, and no source-derived table enters the ``PDW1`` target.
    """

    labels = open_stored_npy_memmap(cache_path, labels_key)
    weight, bias = read_frozen_segmentation_head(frozen_head_path)
    n_classes = int(weight.shape[0])
    if labels.ndim == 2:
        if selection is not None:
            raise PowerDiagramWitnessError("selection is invalid for a single 2-D partition")
        frames: Iterable[np.ndarray] = (labels,)
        selected_count = 1
        selected_shape = tuple(int(value) for value in labels.shape)
    elif labels.ndim == 3:
        indices = _selected_indices(labels.shape[0], selection)
        if not indices:
            raise PowerDiagramWitnessError("partition selection is empty")
        frames = (labels[index] for index in indices)
        selected_count = len(indices)
        selected_shape = (selected_count, *tuple(int(value) for value in labels.shape[1:]))
    else:
        raise PowerDiagramWitnessError("cached labels must have shape (H,W) or (N,H,W)")
    counts = np.zeros(n_classes, dtype=np.int64)
    digest = hashlib.sha256()
    edges: set[tuple[int, int]] = set()
    for raw_frame in frames:
        frame = np.asarray(raw_frame)
        if frame.dtype.kind not in "iu" or np.any(frame < 0) or np.any(frame >= n_classes):
            raise PowerDiagramWitnessError("cached partition contains invalid labels")
        contiguous = np.ascontiguousarray(frame)
        digest.update(contiguous.tobytes(order="C"))
        counts += np.bincount(contiguous.ravel(), minlength=n_classes)[:n_classes]
        edges.update(observed_four_neighbour_adjacency(contiguous, n_classes=n_classes))
    head = affine_head_to_power_diagram(weight, bias, adjacency=sorted(edges))
    active = tuple(int(value) for value in np.flatnonzero(counts))
    return VideoFedTargetReceipt(
        target=head.target,
        active_classes=active,
        class_counts=tuple(int(value) for value in counts),
        adjacency=tuple(sorted(edges)),
        selected_partition_sha256=digest.hexdigest(),
        frozen_head_sha256=sha256_file(frozen_head_path),
        selected_shape=selected_shape,
        selected_dtype=np.dtype(labels.dtype).str,
        selected_partitions=selected_count,
        labels_member=labels_key if labels_key.endswith(".npy") else f"{labels_key}.npy",
    )


def _validate_canonical_target(target: PowerDiagramTarget) -> None:
    ids = np.asarray(target.class_ids)
    if ids.ndim != 1 or ids.size < 2 or ids.size > 0xFFFF:
        raise PowerDiagramWitnessError("target class count is invalid")
    if ids.dtype != np.dtype("<u2"):
        raise PowerDiagramWitnessError("target class ids must use canonical little-endian uint16")
    ids_tuple = tuple(int(value) for value in ids)
    if tuple(sorted(set(ids_tuple))) != ids_tuple or any(value < 0 or value > 0xFFFF for value in ids_tuple):
        raise PowerDiagramWitnessError("target class ids are not canonical uint16 order")
    sites = np.asarray(target.sites)
    weights = np.asarray(target.weights)
    if sites.dtype != np.dtype("<f4") or weights.dtype != np.dtype("<f4"):
        raise PowerDiagramWitnessError("target sites/weights must use canonical little-endian float32")
    if sites.ndim != 2 or sites.shape[0] != ids.size or not 1 <= sites.shape[1] <= 0xFFFF:
        raise PowerDiagramWitnessError("target site shape is invalid")
    if weights.shape != (ids.size,) or not np.isfinite(sites).all() or not np.isfinite(weights).all():
        raise PowerDiagramWitnessError("target sites/weights are invalid")
    if np.any((sites == 0) & np.signbit(sites)) or np.any((weights == 0) & np.signbit(weights)):
        raise PowerDiagramWitnessError("negative zero is noncanonical")
    canonical_edges = canonical_adjacency(target.adjacency, ids_tuple)
    if canonical_edges != target.adjacency:
        raise PowerDiagramWitnessError("target adjacency is not sorted and unique")
    expected_normals, expected_offsets = _tie_loci(ids, sites, weights, canonical_edges)
    normals = np.asarray(target.tie_normals)
    offsets = np.asarray(target.tie_offsets)
    if normals.dtype != np.dtype("<f4") or offsets.dtype != np.dtype("<f4"):
        raise PowerDiagramWitnessError("target tie loci must use canonical little-endian float32")
    if normals.shape != expected_normals.shape or offsets.shape != expected_offsets.shape:
        raise PowerDiagramWitnessError("target tie-locus shape is inconsistent")
    if not np.isfinite(normals).all() or not np.isfinite(offsets).all():
        raise PowerDiagramWitnessError("target tie loci contain non-finite values")
    if np.any((normals == 0) & np.signbit(normals)) or np.any((offsets == 0) & np.signbit(offsets)):
        raise PowerDiagramWitnessError("negative zero in tie loci is noncanonical")
    if not np.array_equal(normals, expected_normals) or not np.array_equal(offsets, expected_offsets):
        raise PowerDiagramWitnessError("target tie loci are inconsistent with sites/weights")


def encode_pdw1(target: PowerDiagramTarget) -> bytes:
    """Encode deterministic little-endian ``PDW1`` bytes with no trailer."""

    _validate_canonical_target(target)
    n_classes, rank, n_edges = target.n_classes, target.rank, len(target.adjacency)
    if n_edges > n_classes * (n_classes - 1) // 2:
        raise PowerDiagramWitnessError("too many adjacency edges")
    chunks = [PDW1_HEADER.pack(PDW1_MAGIC, n_classes, rank, n_edges)]
    chunks.append(np.asarray(target.class_ids, dtype="<u2").tobytes(order="C"))
    chunks.append(np.asarray(target.sites, dtype="<f4").tobytes(order="C"))
    chunks.append(np.asarray(target.weights, dtype="<f4").tobytes(order="C"))
    edge_array = np.asarray(target.adjacency, dtype="<u2").reshape(n_edges, 2)
    chunks.append(edge_array.tobytes(order="C"))
    chunks.append(np.asarray(target.tie_normals, dtype="<f4").tobytes(order="C"))
    chunks.append(np.asarray(target.tie_offsets, dtype="<f4").tobytes(order="C"))
    return b"".join(chunks)


def decode_pdw1(payload: bytes | bytearray | memoryview) -> PowerDiagramTarget:
    """Strictly decode canonical ``PDW1``; reject truncation and every trailing byte."""

    view = memoryview(payload)
    if len(view) < PDW1_HEADER.size:
        raise PowerDiagramWitnessError("truncated PDW1 header")
    magic, n_classes, rank, n_edges = PDW1_HEADER.unpack_from(view)
    if magic != PDW1_MAGIC:
        raise PowerDiagramWitnessError("bad PDW1 magic")
    if n_classes < 2 or rank < 1 or n_edges > n_classes * (n_classes - 1) // 2:
        raise PowerDiagramWitnessError("invalid PDW1 counts")
    expected = PDW1_HEADER.size + 2 * n_classes + 4 * n_classes * rank + 4 * n_classes
    expected += 4 * n_edges + 4 * n_edges * rank + 4 * n_edges
    if len(view) != expected:
        relation = "truncated" if len(view) < expected else "trailing"
        raise PowerDiagramWitnessError(f"{relation} PDW1 bytes")
    offset = PDW1_HEADER.size

    def take(dtype: str, count: int) -> np.ndarray:
        nonlocal offset
        dt = np.dtype(dtype)
        size = dt.itemsize * count
        result = np.frombuffer(view[offset : offset + size], dtype=dt, count=count).copy()
        offset += size
        return result

    ids = take("<u2", n_classes)
    sites = take("<f4", n_classes * rank).reshape(n_classes, rank)
    weights = take("<f4", n_classes)
    edge_values = take("<u2", n_edges * 2).reshape(n_edges, 2)
    normals = take("<f4", n_edges * rank).reshape(n_edges, rank)
    tie_offsets = take("<f4", n_edges)
    edges = tuple((int(edge[0]), int(edge[1])) for edge in edge_values)
    target = PowerDiagramTarget(
        class_ids=ids,
        sites=sites,
        weights=weights,
        adjacency=edges,
        tie_normals=normals,
        tie_offsets=tie_offsets,
    )
    _validate_canonical_target(target)
    if encode_pdw1(target) != bytes(view):
        raise PowerDiagramWitnessError("PDW1 bytes are not canonical")
    return target


def compare_target_to_realizations(
    target: PowerDiagramTarget | bytes,
    realization_paths: Sequence[str | Path],
) -> DescriptionLengthReceipt:
    """Compare exact target bytes to named files without implying equivalence."""

    target_bytes = encode_pdw1(target) if isinstance(target, PowerDiagramTarget) else bytes(target)
    # Strictly establish that caller-supplied bytes are canonical target bytes.
    decoded = decode_pdw1(target_bytes)
    if encode_pdw1(decoded) != target_bytes:
        raise PowerDiagramWitnessError("target bytes failed decode/re-encode identity")
    receipts: list[FileSizeReceipt] = []
    for raw_path in realization_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise PowerDiagramWitnessError(f"realization comparator is not a file: {path}")
        receipts.append(FileSizeReceipt(path=str(path), bytes=path.stat().st_size, sha256=sha256_file(path)))
    return DescriptionLengthReceipt(
        target_bytes=len(target_bytes),
        target_sha256=hashlib.sha256(target_bytes).hexdigest(),
        realization_files=tuple(receipts),
    )


# Clear aliases for callers using the spec's generator/cell vocabulary.
generator_to_cells = power_assign
fit_generators_from_partition = fit_power_diagram_from_paired_features


__all__ = [
    "PDW1_MAGIC",
    "TARGET_COMPARISON_VERDICT",
    "DescriptionLengthReceipt",
    "F32ParityReceipt",
    "FileSizeReceipt",
    "HeadPowerDiagram",
    "InverseFitReceipt",
    "PowerDiagramTarget",
    "PowerDiagramWitnessError",
    "VideoFedTargetReceipt",
    "affine_head_to_power_diagram",
    "canonical_adjacency",
    "canonical_row_difference_basis",
    "compare_target_to_realizations",
    "decode_pdw1",
    "encode_pdw1",
    "fit_generators_from_partition",
    "fit_power_diagram_from_labels",
    "fit_power_diagram_from_paired_features",
    "generator_to_cells",
    "initialize_video_fed_target",
    "is_co_maximum_tie",
    "make_power_diagram_target",
    "measure_f32_target_parity",
    "observed_four_neighbour_adjacency",
    "open_stored_npy_memmap",
    "pair_tie_value",
    "power_assign",
    "power_distances",
    "power_scores",
    "project_channel_features",
    "read_frozen_segmentation_head",
    "read_safetensors_tensors",
    "sha256_file",
]
