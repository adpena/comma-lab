# SPDX-License-Identifier: MIT
"""Receiver-closed prerequisite surfaces for the regularized-max probes.

The module is deliberately narrow:

* a continuous RGB proposal may only choose among solutions of the existing
  bounded uint8 Diophantine solver; a fresh oracle on parsed uint8 bytes owns
  ``HARD_ACCEPT``;
* the frozen SegNet head is reduced to its deterministic rank-4 row-difference
  quotient and one strictly interior minimum-norm prototype per valid cell;
* three shared-affine representatives of the same max-affine complex are
  compared through the existing PDW2 serializer and one Brotli-q11 coder.

No function here evaluates a contest score, moves a frontier pointer, or turns
advisory/local evidence into promotion authority.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.boundary_math.power_diagram_witness import (
    GaugeFixedAffineTarget,
    canonical_row_difference_basis,
    encode_pdw2,
    gauge_fixed_assign_f32,
    make_gauge_fixed_affine_target,
    read_frozen_segmentation_head,
    sha256_file,
)
from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import (
    SEGNET_WEIGHTS_SHA256,
)
from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    HardOracleEvaluation,
    LatticeFrameResult,
    Uint8LatticeError,
    parse_uint8_frame,
    serialize_uint8_frame,
)

MATCHED_ADAPTER_SCHEMA: Final = "regmax_logits_to_uint8_preimage_v1"
PROTOTYPE_BANK_SCHEMA: Final = "rank4_valid_cell_prototypes_v1"
COMPARATOR_SCHEMA: Final = "aurenhammer_same_coder_comparator_v1"
M1_BAND_AUDIT_SCHEMA: Final = "m1_positive_anisotropic_band_readiness_audit_v1"
PDW2_GAUGE: Final = "PDW2_REFERENCE_CLASS_AFFINE_GAUGE"
EXPECTED_CLASSES: Final = 5
EXPECTED_RANK: Final = 4
FP32_RECON_FLOOR: Final = 5.960464477539063e-08
M1_LOGICAL_PAIR_COUNT: Final = 600
M1_MEASURED_CANDIDATE_COUNT: Final = 38_077
M1_GT_CACHE_SHA256: Final = (
    "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
)
_VJP_TENSOR_KEYS: Final = frozenset(
    {
        "winner",
        "rival",
        "seg_q",
        "seg_local_lipschitz",
        "head_pair_norms",
        "pose_j_x",
        "pose_j_y",
    }
)


class PrerequisiteSurfaceError(ValueError):
    """Fail-closed malformed input or violated receiver contract."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise PrerequisiteSurfaceError("value is not canonical JSON") from exc


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(np.dtype(array.dtype).str.encode("ascii"))
    digest.update(_canonical_json([int(v) for v in array.shape]))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_path(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise PrerequisiteSurfaceError(f"{label} must be a lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise PrerequisiteSurfaceError(f"{label} must be a lowercase SHA-256") from exc
    if value.lower() != value or value == "0" * 64:
        raise PrerequisiteSurfaceError(f"{label} must be a non-placeholder lowercase SHA-256")
    return value


def _read_json_mapping(
    path: str | Path,
    *,
    label: str,
) -> tuple[Path, Mapping[str, Any], str, int]:
    resolved = Path(path).expanduser().resolve(strict=True)
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrerequisiteSurfaceError(f"{label} must be ASCII JSON") from exc
    if not isinstance(value, dict):
        raise PrerequisiteSurfaceError(f"{label} must be a JSON mapping")
    return resolved, value, hashlib.sha256(raw).hexdigest(), len(raw)


def _immutable(value: np.ndarray, *, dtype: np.dtype[Any] | str | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def _continuous_proposal_to_camera(
    operator: DisjointResizeOperator,
    continuous_pre_step: np.ndarray,
    *,
    channels: int,
) -> tuple[np.ndarray, str, int]:
    raw = np.asarray(continuous_pre_step)
    if raw.dtype.kind not in ("i", "u", "f") or raw.dtype.kind == "b":
        raise PrerequisiteSurfaceError("continuous pre-step must be real numeric")
    values = raw.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise PrerequisiteSurfaceError("continuous pre-step must be finite")
    if values.ndim == 2 and channels == 1:
        values = values[:, :, None]
    clipped = np.clip(values, 0.0, 255.0)
    clipped_values = int(np.count_nonzero(clipped != values))
    camera_shape = (operator.camera_h, operator.camera_w, channels)
    scorer_shape = (operator.scorer_h, operator.scorer_w, channels)
    if clipped.shape == camera_shape:
        return np.ascontiguousarray(clipped), "camera_preimage", clipped_values
    if clipped.shape == scorer_shape:
        camera = operator.bounded_continuous_preimage(clipped)
        if camera.ndim == 2:
            camera = camera[:, :, None]
        return np.ascontiguousarray(camera), "scorer_plane_lift", clipped_values
    raise PrerequisiteSurfaceError(
        "continuous pre-step must match the camera preimage or scorer-plane geometry"
    )


@dataclass(frozen=True, slots=True)
class MatchedPreimageResult:
    frame: np.ndarray
    payload: bytes
    lattice: LatticeFrameResult
    hard_evaluation: HardOracleEvaluation
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _immutable(self.frame, dtype=np.uint8))


def matched_continuous_to_uint8_hard_accept(
    operator: DisjointResizeOperator,
    continuous_pre_step: np.ndarray,
    exact_target: np.ndarray,
    exact_target_numerators: np.ndarray,
    hard_oracle: Callable[[np.ndarray], HardOracleEvaluation],
    *,
    pre_step_family: str,
    max_nodes_per_block: int = 4096,
) -> MatchedPreimageResult:
    """Run the shared bounded solve, parse bytes, and freshly hard-evaluate.

    ``continuous_pre_step`` is only a deterministic preference.  The exact
    equality, uint8 bounds, and proof status are owned by the existing
    :meth:`DisjointResizeOperator.solve_uint8` implementation.  The supplied
    oracle is invoked exactly once on the freshly parsed bytes; no soft score
    or pre-parse evaluation can admit the result.
    """

    if not isinstance(operator, DisjointResizeOperator):
        raise TypeError("operator must be a DisjointResizeOperator")
    if not isinstance(pre_step_family, str) or not pre_step_family.strip():
        raise PrerequisiteSurfaceError("pre_step_family must be a nonempty label")
    target = np.asarray(exact_target)
    channels = 1 if target.ndim == 2 else int(target.shape[-1]) if target.ndim == 3 else 0
    if channels <= 0:
        raise PrerequisiteSurfaceError("exact target must have nonempty scorer channels")
    preferred, proposal_space, clipped_values = _continuous_proposal_to_camera(
        operator,
        continuous_pre_step,
        channels=channels,
    )
    try:
        lattice = operator.solve_uint8(
            exact_target,
            target_numerators=exact_target_numerators,
            preferred_preimage=preferred,
            max_nodes_per_block=max_nodes_per_block,
        )
    except Uint8LatticeError as exc:
        raise PrerequisiteSurfaceError(f"bounded uint8 solve refused input: {exc}") from exc
    payload = serialize_uint8_frame(lattice.frame)
    decoded = parse_uint8_frame(payload)
    evaluation = hard_oracle(decoded)
    if not isinstance(evaluation, HardOracleEvaluation):
        raise PrerequisiteSurfaceError("hard oracle must return HardOracleEvaluation")
    hard_accept = bool(np.all(evaluation.satisfied))
    receipt = {
        "schema": MATCHED_ADAPTER_SCHEMA,
        "consumer": "regmax_sparsemax_entropy_cole_hopf_and_hopfield_probes",
        "pre_step_family": pre_step_family.strip(),
        "proposal_space": proposal_space,
        "continuous_pre_step_sha256": _array_sha256(np.asarray(continuous_pre_step)),
        "preferred_preimage_sha256": _array_sha256(preferred),
        "exact_target_sha256": _array_sha256(np.asarray(exact_target)),
        "exact_target_numerators_sha256": _array_sha256(
            np.asarray(exact_target_numerators)
        ),
        "continuous_values_clipped_to_uint8_box": clipped_values,
        "bounded_solver": (
            "tac.optimization.uint8_lattice_feasibility."
            "DisjointResizeOperator.solve_uint8"
        ),
        "canonical_law": "bounded_uint8_resize_preimage_cell_feasibility_v1",
        "aggregate_status": str(lattice.aggregate_status),
        "certified_exact": bool(lattice.certified_exact),
        "exact_blocks": int(lattice.diagnostics.exact_blocks),
        "proven_affine_infeasible_blocks": int(
            lattice.diagnostics.proven_affine_infeasible_blocks
        ),
        "decoded_uint8_payload_bytes": len(payload),
        "decoded_uint8_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "decoded_uint8_frame_sha256": _array_sha256(decoded),
        "fresh_decoded_uint8_hard_oracle_calls": 1,
        "hard_oracle_satisfied_sha256": _array_sha256(evaluation.satisfied),
        "hard_oracle_margins_sha256": _array_sha256(evaluation.margins),
        "hard_oracle_key": [
            int(evaluation.key[0]),
            float(evaluation.key[1]),
            float(evaluation.key[2]),
        ],
        "hard_accept": hard_accept,
        "authority": "LOCAL_RECEIVER_READINESS_NOT_SCORE_AUTHORITY",
        "pointer": "0.1910828242 [contest-CPU Linux x86_64] UNMOVED",
        "score_claim": False,
        "promotion_eligible": False,
    }
    return MatchedPreimageResult(decoded, payload, lattice, evaluation, receipt)


def _minimum_norm_strict_cell_prototype(
    affine_weight: np.ndarray,
    affine_bias: np.ndarray,
    class_index: int,
    *,
    required_margin: float,
) -> np.ndarray:
    """Enumerate active sets for the minimum-L2 point in one affine cell."""

    rivals = [index for index in range(affine_weight.shape[0]) if index != class_index]
    constraints = np.stack(
        [affine_weight[class_index] - affine_weight[rival] for rival in rivals]
    )
    thresholds = np.asarray(
        [required_margin + affine_bias[rival] - affine_bias[class_index] for rival in rivals],
        dtype=np.float64,
    )
    candidates: list[tuple[float, bytes, np.ndarray]] = []
    for mask in range(1 << len(rivals)):
        active = [index for index in range(len(rivals)) if mask & (1 << index)]
        if not active:
            point = np.zeros((affine_weight.shape[1],), dtype=np.float64)
        else:
            rows = constraints[active]
            rhs = thresholds[active]
            gram = rows @ rows.T
            multipliers = np.linalg.pinv(gram, rcond=1e-13) @ rhs
            point = rows.T @ multipliers
            if np.max(np.abs(rows @ point - rhs), initial=0.0) > 2e-9:
                continue
        if np.all(constraints @ point >= thresholds - 2e-9):
            point_f32 = np.asarray(point, dtype="<f4")
            candidates.append(
                (float(point_f32 @ point_f32), point_f32.tobytes(order="C"), point_f32)
            )
    if not candidates:
        raise PrerequisiteSurfaceError(
            f"frozen rank-4 class {class_index} has no prototype at margin {required_margin}"
        )
    return min(candidates, key=lambda row: (row[0], row[1]))[2]


@dataclass(frozen=True, slots=True)
class FrozenRank4PrototypeBank:
    quotient_basis: np.ndarray
    affine_weight: np.ndarray
    affine_bias: np.ndarray
    prototypes: np.ndarray
    labels: np.ndarray
    margins: np.ndarray
    pdw2_packet: bytes
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "quotient_basis",
            "affine_weight",
            "affine_bias",
            "prototypes",
            "labels",
            "margins",
        ):
            object.__setattr__(self, name, _immutable(getattr(self, name)))


def build_frozen_rank4_prototype_bank(
    frozen_weights_path: str | Path,
    *,
    expected_weights_sha256: str = SEGNET_WEIGHTS_SHA256,
    required_margin: float = 1.0,
) -> FrozenRank4PrototypeBank:
    """Build a deterministic, SHA-pinned five-cell bank from real weights."""

    path = Path(frozen_weights_path).expanduser().resolve(strict=True)
    actual_sha = sha256_file(path)
    if actual_sha != expected_weights_sha256:
        raise PrerequisiteSurfaceError(
            f"frozen SegNet weights SHA mismatch: {actual_sha}"
        )
    if not np.isfinite(required_margin) or required_margin <= 0.0:
        raise PrerequisiteSurfaceError("required_margin must be finite and > 0")
    weight, bias = read_frozen_segmentation_head(path)
    rows = np.asarray(weight, dtype=np.float64).reshape(weight.shape[0], -1)
    if rows.shape[0] != EXPECTED_CLASSES:
        raise PrerequisiteSurfaceError("frozen SegNet head must have exactly five classes")
    centered_weight = rows - rows.mean(axis=0, keepdims=True)
    centered_bias = np.asarray(bias, dtype=np.float64) - float(np.mean(bias))
    basis = canonical_row_difference_basis(rows)
    if basis.shape != (rows.shape[1], EXPECTED_RANK):
        raise PrerequisiteSurfaceError("frozen head quotient rank drifted from four")
    affine_weight = centered_weight @ basis
    reconstructed = (affine_weight @ basis.T).astype(np.float32)
    centered_f32 = centered_weight.astype(np.float32)
    recon_error = float(np.max(np.abs(reconstructed - centered_f32), initial=0.0))
    if recon_error > FP32_RECON_FLOOR:
        raise PrerequisiteSurfaceError(
            f"rank-4 reconstruction exceeded sealed fp32 floor: {recon_error}"
        )
    prototypes = np.stack(
        [
            _minimum_norm_strict_cell_prototype(
                affine_weight,
                centered_bias,
                class_index,
                required_margin=required_margin,
            )
            for class_index in range(EXPECTED_CLASSES)
        ]
    ).astype("<f4", copy=False)
    logits = np.add(
        prototypes @ affine_weight.astype(np.float32).T,
        centered_bias.astype(np.float32),
        dtype=np.float32,
    )
    labels = np.argmax(logits, axis=1).astype(np.int8)
    rival_logits = np.partition(logits, -2, axis=1)[:, -2]
    margins = np.subtract(logits[np.arange(EXPECTED_CLASSES), labels], rival_logits).astype("<f4")
    if not np.array_equal(labels, np.arange(EXPECTED_CLASSES, dtype=np.int8)):
        raise PrerequisiteSurfaceError("prototype bank failed frozen affine cell identity")
    if np.any(margins < np.float32(required_margin - 2e-5)):
        raise PrerequisiteSurfaceError("prototype bank failed required strict-cell margin")
    target = make_gauge_fixed_affine_target(
        affine_weight,
        centered_bias,
        class_ids=tuple(range(EXPECTED_CLASSES)),
    )
    packet = encode_pdw2(target)
    parsed_labels = gauge_fixed_assign_f32(prototypes, target).astype(np.int8)
    if not np.array_equal(parsed_labels, labels):
        raise PrerequisiteSurfaceError("PDW2 reference-gauge parse-back changed prototype cells")
    receipt = {
        "schema": PROTOTYPE_BANK_SCHEMA,
        "consumer": "regmax_probe3_and_r1b_opportunistic_reuse",
        "frozen_weights_path": str(path),
        "frozen_weights_sha256": actual_sha,
        "frozen_head": "segmentation_head.0.weight (5,16,3,3) + bias (5,)",
        "canonical_equation": "segnet_head_rank4_linear_flipdist_v1",
        "rank": EXPECTED_RANK,
        "class_count": EXPECTED_CLASSES,
        "quotient_basis_sha256": _array_sha256(basis.astype("<f8")),
        "affine_weight_sha256": _array_sha256(affine_weight.astype("<f4")),
        "affine_bias_sha256": _array_sha256(centered_bias.astype("<f4")),
        "prototype_sha256": _array_sha256(prototypes),
        "prototype_labels": [int(value) for value in labels],
        "prototype_margins": [float(value) for value in margins],
        "required_margin": float(required_margin),
        "rank4_reconstruction_maxabs_fp32": recon_error,
        "sealed_rank4_reconstruction_fp32_floor": FP32_RECON_FLOOR,
        "gauge": PDW2_GAUGE,
        "reference_class": 0,
        "pdw2_packet_bytes": len(packet),
        "pdw2_packet_sha256": hashlib.sha256(packet).hexdigest(),
        "authority": "REAL_FROZEN_WEIGHTS_LOCAL_STRUCTURAL_READINESS",
        "pointer": "0.1910828242 [contest-CPU Linux x86_64] UNMOVED",
        "score_claim": False,
        "promotion_eligible": False,
    }
    return FrozenRank4PrototypeBank(
        basis,
        affine_weight.astype("<f4"),
        centered_bias.astype("<f4"),
        prototypes,
        labels,
        margins,
        packet,
        receipt,
    )


def _shared_affine_representatives(
    affine_weight: np.ndarray,
    affine_bias: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    rows = np.asarray(affine_weight, dtype=np.float32)
    bias = np.asarray(affine_bias, dtype=np.float32)
    augmented = np.concatenate((rows, bias[:, None]), axis=1).astype(np.float64)
    # The midpoint is the closed-form solution of the shared-affine L-infinity
    # min-generator LP: min_{g,t} t subject to |a_ki-g_i| <= t.
    min_generator = 0.5 * (augmented.min(axis=0) + augmented.max(axis=0))
    # Max-plus principal normalization: every coordinate has tropical maximum 0.
    tropical_principal = augmented.max(axis=0)
    # Orthogonal projection onto the zero-sum gauge; the shared-affine L2 minimizer.
    zero_sum_min_norm = augmented.mean(axis=0)
    gauges = {
        "aurenhammer_min_generator_lp": (
            min_generator,
            "shared-affine Linf min-generator LP midpoint solution",
        ),
        "tropical_residuation_principal": (
            tropical_principal,
            "max-plus principal residuation normalization",
        ),
        "zero_sum_min_norm": (
            zero_sum_min_norm,
            "shared-affine L2 projection onto zero-sum gauge",
        ),
    }
    result: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    for name, (gauge, derivation) in gauges.items():
        representative = np.subtract(augmented, gauge, dtype=np.float64).astype(np.float32)
        result[name] = (representative[:, :-1], representative[:, -1], derivation)
    return result


def serialize_affine_cell_candidate_same_coder(
    candidate_name: str,
    affine_weight: np.ndarray,
    affine_bias: np.ndarray,
    prototypes: np.ndarray,
    *,
    expected_cell_labels: tuple[int, ...] = tuple(range(EXPECTED_CLASSES)),
) -> dict[str, Any]:
    """Serialize any five-cell affine representative through PDW2+Brotli-q11."""

    if not isinstance(candidate_name, str) or not candidate_name.strip():
        raise PrerequisiteSurfaceError("candidate_name must be a nonempty label")
    weight = np.asarray(affine_weight)
    bias = np.asarray(affine_bias)
    points = np.asarray(prototypes)
    expected = np.asarray(expected_cell_labels)
    if (
        weight.ndim != 2
        or weight.shape[0] != EXPECTED_CLASSES
        or bias.shape != (EXPECTED_CLASSES,)
        or points.ndim != 2
        or points.shape[1] != weight.shape[1]
        or expected.shape != (points.shape[0],)
        or not np.issubdtype(expected.dtype, np.integer)
    ):
        raise PrerequisiteSurfaceError("candidate affine/prototype geometry mismatch")
    if not (
        np.isfinite(weight).all()
        and np.isfinite(bias).all()
        and np.isfinite(points).all()
    ):
        raise PrerequisiteSurfaceError("candidate affine/prototype values must be finite")
    if np.any(expected < 0) or np.any(expected >= EXPECTED_CLASSES):
        raise PrerequisiteSurfaceError("expected cell labels are out of range")
    try:
        import brotli
    except ImportError as exc:  # pragma: no cover - deployment dependency gate
        raise PrerequisiteSurfaceError("Brotli is required for the same-coder comparator") from exc
    target: GaugeFixedAffineTarget = make_gauge_fixed_affine_target(
        weight,
        bias,
        class_ids=tuple(range(EXPECTED_CLASSES)),
    )
    packet = encode_pdw2(target)
    coded = brotli.compress(packet, quality=11)
    labels = gauge_fixed_assign_f32(points, target).astype(np.int8)
    return {
        "candidate_name": candidate_name.strip(),
        "packet_serializer": "tac.boundary_math.power_diagram_witness.encode_pdw2",
        "packet_bytes": len(packet),
        "packet_sha256": hashlib.sha256(packet).hexdigest(),
        "coder": "brotli_quality_11",
        "coded_bytes": len(coded),
        "coded_sha256": hashlib.sha256(coded).hexdigest(),
        "prototype_cell_labels": [int(value) for value in labels],
        "expected_prototype_cell_labels": [int(value) for value in expected],
        "prototype_cell_identity_sha256": _array_sha256(labels),
        "exact_cell_identity": bool(np.array_equal(labels, expected.astype(np.int8))),
        "cell_identity_scope": "caller-supplied strict prototype witnesses",
    }


def compare_affine_cell_representatives_same_coder(
    bank: FrozenRank4PrototypeBank,
) -> dict[str, Any]:
    """Compare three gauges through identical PDW2 + Brotli-q11 coding."""

    if not isinstance(bank, FrozenRank4PrototypeBank):
        raise TypeError("bank must be FrozenRank4PrototypeBank")
    rows: dict[str, Any] = {}
    identity_hashes: set[str] = set()
    for name, (weight, bias, derivation) in _shared_affine_representatives(
        bank.affine_weight,
        bank.affine_bias,
    ).items():
        row = serialize_affine_cell_candidate_same_coder(
            name,
            weight,
            bias,
            bank.prototypes,
        )
        row["derivation"] = derivation
        identity_hashes.add(row["prototype_cell_identity_sha256"])
        rows[name] = row
    same_identity = len(identity_hashes) == 1 and all(
        row["prototype_cell_labels"] == list(range(EXPECTED_CLASSES))
        for row in rows.values()
    )
    return {
        "schema": COMPARATOR_SCHEMA,
        "consumer": "regmax_tropical_aurenhammer_comparator_probe2",
        "prototype_bank_sha256": bank.receipt["prototype_sha256"],
        "frozen_weights_sha256": bank.receipt["frozen_weights_sha256"],
        "gauge": PDW2_GAUGE,
        "same_coder": True,
        "same_packet_serializer": True,
        "exact_cell_identity": same_identity,
        "cell_identity_scope": "five frozen-head strict prototype witnesses",
        "representatives": rows,
        "authority": "LOCAL_SAME_CODER_STRUCTURAL_COMPARATOR_NOT_SCORE_AUTHORITY",
        "pointer": "0.1910828242 [contest-CPU Linux x86_64] UNMOVED",
        "score_claim": False,
        "promotion_eligible": False,
    }


def audit_m1_positive_band_prerequisites(
    vjp_manifest_paths: tuple[str | Path, ...],
    prototype_receipt_path: str | Path,
    candidate_receipt_path: str | Path,
    *,
    verify_sidecar_bytes: bool = True,
) -> dict[str, Any]:
    """Audit real M1 inputs and refuse a false full-n600 band assembly.

    The function intentionally produces no radii store.  It may authorize a
    later assembler only when all 600 logical pair sidecars and the exact
    38,077-candidate measured EV field exist.  Manifest-declared sidecar hashes
    are re-read by default; callers may disable that expensive check only for
    small structural unit tests, and the receipt records the weaker mode.
    """

    if not vjp_manifest_paths:
        raise PrerequisiteSurfaceError("at least one VJP custody manifest is required")
    pair_records: dict[int, dict[str, Any]] = {}
    manifest_records: list[dict[str, Any]] = []
    source_hashes_seen: set[tuple[str, str]] = set()
    for raw_path in vjp_manifest_paths:
        manifest_path, manifest, manifest_sha, manifest_bytes = _read_json_mapping(
            raw_path,
            label="VJP manifest",
        )
        if manifest.get("schema") != "vjp_custody_manifest.v1":
            raise PrerequisiteSurfaceError("VJP custody manifest schema mismatch")
        pair_ids = manifest.get("pair_ids")
        sidecars = manifest.get("sidecars")
        source_hashes = manifest.get("source_hashes")
        authority = manifest.get("authority")
        if (
            not isinstance(pair_ids, list)
            or not isinstance(sidecars, list)
            or len(pair_ids) != len(sidecars)
            or not isinstance(source_hashes, dict)
            or not isinstance(authority, dict)
        ):
            raise PrerequisiteSurfaceError("VJP custody manifest fields are malformed")
        if source_hashes.get("cache_sha256") != M1_GT_CACHE_SHA256:
            raise PrerequisiteSurfaceError("VJP manifest is not bound to the real n600 GT cache")
        if source_hashes.get("segnet_weights_sha256") != SEGNET_WEIGHTS_SHA256:
            raise PrerequisiteSurfaceError("VJP manifest SegNet weight custody mismatch")
        if authority.get("score_claim") is not False:
            raise PrerequisiteSurfaceError("VJP custody must remain non-score authority")
        source_hashes_seen.add(
            (str(source_hashes["cache_sha256"]), str(source_hashes["segnet_weights_sha256"]))
        )
        listed_ids: list[int] = []
        for sidecar in sidecars:
            if not isinstance(sidecar, dict):
                raise PrerequisiteSurfaceError("VJP sidecar record must be a mapping")
            pair_id = sidecar.get("pair_id")
            path_value = sidecar.get("path")
            expected_sha = sidecar.get("sha256")
            tensor_hashes = sidecar.get("tensor_hashes")
            if (
                type(pair_id) is not int
                or pair_id < 0
                or pair_id >= M1_LOGICAL_PAIR_COUNT
                or not isinstance(path_value, str)
                or not isinstance(tensor_hashes, dict)
                or not _VJP_TENSOR_KEYS.issubset(tensor_hashes)
            ):
                raise PrerequisiteSurfaceError("VJP sidecar custody fields are malformed")
            expected_sha = _require_sha256(
                expected_sha,
                label=f"VJP sidecar {pair_id} SHA",
            )
            for name in _VJP_TENSOR_KEYS:
                _require_sha256(
                    tensor_hashes[name],
                    label=f"VJP sidecar {pair_id} tensor {name}",
                )
            if pair_id in pair_records:
                raise PrerequisiteSurfaceError(f"duplicate VJP custody for pair {pair_id}")
            sidecar_path = Path(path_value).expanduser().resolve(strict=True)
            actual_sha = _sha256_path(sidecar_path) if verify_sidecar_bytes else None
            if verify_sidecar_bytes and actual_sha != expected_sha:
                raise PrerequisiteSurfaceError(f"VJP sidecar SHA mismatch for pair {pair_id}")
            pair_records[pair_id] = {
                "pair_id": pair_id,
                "path": str(sidecar_path),
                "bytes": sidecar_path.stat().st_size,
                "sha256": expected_sha,
                "tensor_hashes": {
                    name: str(tensor_hashes[name]) for name in sorted(_VJP_TENSOR_KEYS)
                },
            }
            listed_ids.append(pair_id)
        if listed_ids != pair_ids:
            raise PrerequisiteSurfaceError("VJP manifest pair_ids disagree with sidecar order")
        manifest_records.append(
            {
                "path": str(manifest_path),
                "bytes": manifest_bytes,
                "sha256": manifest_sha,
                "pair_ids": listed_ids,
            }
        )
    if len(source_hashes_seen) != 1:
        raise PrerequisiteSurfaceError("VJP manifests disagree on scorer source custody")

    prototype_path, prototype, prototype_sha, prototype_bytes = _read_json_mapping(
        prototype_receipt_path,
        label="rank-4 prototype receipt",
    )
    if (
        prototype.get("schema") != PROTOTYPE_BANK_SCHEMA
        or prototype.get("frozen_weights_sha256") != SEGNET_WEIGHTS_SHA256
        or prototype.get("rank") != EXPECTED_RANK
        or not isinstance(prototype.get("affine_weight_sha256"), str)
    ):
        raise PrerequisiteSurfaceError("rank-4 prototype receipt custody mismatch")
    _require_sha256(
        prototype["affine_weight_sha256"],
        label="rank-4 affine weight",
    )

    candidate_path, candidate, candidate_sha, candidate_bytes = _read_json_mapping(
        candidate_receipt_path,
        label="candidate-selection receipt",
    )
    if candidate.get("schema") != "r2b_sparse_target_selection_receipt.v1":
        raise PrerequisiteSurfaceError("candidate-selection receipt schema mismatch")
    if (
        candidate.get("score_claim") is not False
        or candidate.get("promotion_eligible") is not False
    ):
        raise PrerequisiteSurfaceError(
            "candidate-selection receipt must remain non-score and non-promotable"
        )
    gt_cache = candidate.get("gt_cache")
    baseline = candidate.get("baseline")
    if (
        not isinstance(gt_cache, dict)
        or gt_cache.get("sha256") != M1_GT_CACHE_SHA256
        or not isinstance(baseline, dict)
        or type(baseline.get("flip_count")) is not int
        or type(candidate.get("candidate_evaluation_decisions")) is not int
    ):
        raise PrerequisiteSurfaceError("candidate-selection n600 custody is malformed")

    covered_pairs = sorted(pair_records)
    missing_pairs = sorted(set(range(M1_LOGICAL_PAIR_COUNT)) - set(covered_pairs))
    observed_flip_count = int(baseline["flip_count"])
    observed_decisions = int(candidate["candidate_evaluation_decisions"])
    measured_ordering = candidate.get("measured_fisher_ev_ordering")
    measured_ordering_ready = False
    measured_ordering_record: dict[str, Any] | None = None
    if isinstance(measured_ordering, dict):
        artifact = measured_ordering.get("artifact")
        if (
            measured_ordering.get("candidate_count") == M1_MEASURED_CANDIDATE_COUNT
            and measured_ordering.get("metric") == "fisher_top1_top2_margin"
            and measured_ordering.get("policy")
            == "measured_reverse_waterfill_highest_ev_first"
            and isinstance(artifact, dict)
            and set(artifact) == {"path", "sha256"}
            and isinstance(artifact.get("path"), str)
        ):
            artifact_path = Path(artifact["path"]).expanduser()
            if not artifact_path.is_absolute():
                artifact_path = candidate_path.parent / artifact_path
            artifact_path = artifact_path.resolve(strict=True)
            artifact_sha = _require_sha256(
                artifact.get("sha256"),
                label="measured Fisher EV ordering artifact",
            )
            if _sha256_path(artifact_path) != artifact_sha:
                raise PrerequisiteSurfaceError("measured Fisher EV ordering SHA mismatch")
            measured_ordering_ready = True
            measured_ordering_record = {
                "path": str(artifact_path),
                "sha256": artifact_sha,
                "bytes": artifact_path.stat().st_size,
            }
    blockers: list[dict[str, Any]] = []
    if missing_pairs:
        blockers.append(
            {
                "code": "INCOMPLETE_PAIR_LOCAL_VJP_CUSTODY",
                "required_pair_count": M1_LOGICAL_PAIR_COUNT,
                "observed_pair_count": len(covered_pairs),
                "missing_pair_count": len(missing_pairs),
                "missing_pair_ids_sha256": hashlib.sha256(
                    _canonical_json(missing_pairs)
                ).hexdigest(),
                "verdict_scope": (
                    "full-n600 positive-anisotropic assembly only; the 24 custodied "
                    "sidecars remain valid advisory pair-local evidence"
                ),
            }
        )
    if not measured_ordering_ready:
        blockers.append(
            {
                "code": "EXACT_38077_CANDIDATE_EV_FIELD_ABSENT",
                "required_candidate_count": M1_MEASURED_CANDIDATE_COUNT,
                "observed_baseline_hard_oracle_flip_count": observed_flip_count,
                "observed_candidate_evaluation_decisions": observed_decisions,
                "measured_ordering_record_present": isinstance(measured_ordering, dict),
                "candidate_hard_gate_pass": candidate.get("hard_gate_pass"),
                "candidate_promotion_eligible": candidate.get("promotion_eligible"),
                "verdict_scope": (
                    "the inspected real n600 sparse-selection receipt is not the sealed "
                    "38,077-candidate measured Fisher EV ordering"
                ),
            }
        )
    coverage_digest = hashlib.sha256(
        _canonical_json([pair_records[pair_id] for pair_id in covered_pairs])
    ).hexdigest()
    return {
        "schema": M1_BAND_AUDIT_SCHEMA,
        "surface": "M1_rc6_receiver_closure_blocker_1",
        "consumer": "M1_relaunch_gate_and_r1b_opportunistic_reuse",
        "requested_artifact": "c2_anisotropic_band_artifact.v1",
        "ready_to_assemble": not blockers,
        "artifact_materialized": False,
        "vjp_custody": {
            "manifests": manifest_records,
            "sidecar_bytes_rehashed": verify_sidecar_bytes,
            "covered_pair_count": len(covered_pairs),
            "covered_pair_ids": covered_pairs,
            "missing_pair_count": len(missing_pairs),
            "sidecar_record_digest_sha256": coverage_digest,
            "required_tensor_hashes": sorted(_VJP_TENSOR_KEYS),
        },
        "rank4_head_normals": {
            "receipt_path": (
                str(prototype_receipt_path)
                if not Path(prototype_receipt_path).is_absolute()
                else str(prototype_path)
            ),
            "receipt_sha256": prototype_sha,
            "receipt_bytes": prototype_bytes,
            "frozen_weights_sha256": prototype["frozen_weights_sha256"],
            "affine_weight_sha256": prototype["affine_weight_sha256"],
            "rank": prototype["rank"],
        },
        "candidate_evidence": {
            "receipt_path": str(candidate_path),
            "receipt_sha256": candidate_sha,
            "receipt_bytes": candidate_bytes,
            "gt_cache_sha256": gt_cache["sha256"],
            "ranking": candidate.get("ranking"),
            "observed_baseline_hard_oracle_flip_count": observed_flip_count,
            "observed_candidate_evaluation_decisions": observed_decisions,
            "required_measured_candidate_count": M1_MEASURED_CANDIDATE_COUNT,
            "measured_ordering_ready": measured_ordering_ready,
            "measured_ordering_artifact": measured_ordering_record,
            "hard_gate_pass": candidate.get("hard_gate_pass"),
        },
        "blockers": blockers,
        "refusal": (
            None
            if not blockers
            else "NO-FAKE: no n600 radii or positive-band manifest was emitted"
        ),
        "pointer": "0.1910828242 [contest-CPU Linux x86_64] UNMOVED",
        "authority": "LOCAL_INPUT_CUSTODY_AUDIT_NOT_SCORE_AUTHORITY",
        "score_claim": False,
        "promotion_eligible": False,
    }


def readiness_bundle(
    bank: FrozenRank4PrototypeBank,
    comparator: Mapping[str, Any],
) -> dict[str, Any]:
    """Small canonical receipt bundle for durable tool output."""

    return {
        "schema": "prereq_surfaces_flush_readiness_bundle.v1",
        "surface_2": dict(bank.receipt),
        "surface_3": dict(comparator),
        "consumers": {
            "surface_1": "regmax probes entropy-Hopfield and sparsemax",
            "surface_2": "regmax probe3 and r1b opportunistic reuse",
            "surface_3": "regmax tropical/Aurenhammer comparator probe2",
        },
        "score_claim": False,
        "promotion_eligible": False,
    }


__all__ = [
    "COMPARATOR_SCHEMA",
    "M1_BAND_AUDIT_SCHEMA",
    "MATCHED_ADAPTER_SCHEMA",
    "PROTOTYPE_BANK_SCHEMA",
    "FrozenRank4PrototypeBank",
    "MatchedPreimageResult",
    "PrerequisiteSurfaceError",
    "audit_m1_positive_band_prerequisites",
    "build_frozen_rank4_prototype_bank",
    "compare_affine_cell_representatives_same_coder",
    "matched_continuous_to_uint8_hard_accept",
    "readiness_bundle",
    "serialize_affine_cell_candidate_same_coder",
]
