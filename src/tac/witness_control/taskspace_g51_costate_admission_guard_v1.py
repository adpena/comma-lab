# SPDX-License-Identifier: MIT
"""Fail-closed G51 encoder-diagnostic to costate admission boundary.

G51 measured exact encoder-side chunk/block sizes.  It did not measure
scorer effects or same-object archive ZIP marginals.  Historical v1 field
names made those diagnostics too easy to mistake for sensitivity, costate,
allocator, Pareto, or autopilot inputs.

This module deliberately exposes two different types:

* :class:`G51EncoderDiagnosticV1` is the exact b84 adversarial receipt reduced
  to a diagnostic-only token.  None of the receipt's numeric payload is
  retained.
* :class:`G51CostateEvidenceCandidateV1` records only file identities after
  independently reopening proposed full-n600 scorer-term effect vectors,
  JVP/VJP arrays, and two ZIP archives.  Passing those integrity checks is not
  proof that the arrays came from the claimed receiver/R/scorer transition.

No actionable admission exists yet.  A canonical materializer that computes
and seals the receiver/R/scorer transition is owed.  Until it lands,
``admit_g51_actionable_costate_evidence`` and every consumer request refuse
after diagnostic validation.  Self-attested measurement booleans, arbitrary
runtime files, and merely nonzero arrays can therefore never reach a consumer.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, NoReturn

import numpy as np

from tac.optimization.serialized_archive_economics import (
    SERIALIZED_ARCHIVE_DELTA_SCHEMA,
    build_serialized_archive_delta_contract,
)

G51_ADVERSARIAL_SCHEMA: Final = "tac.taskspace_conditional_quotient_profile_adversarial_interpretation.v1"
G51_ADVERSARIAL_PROFILE_STATUS: Final = "FULL_N600_ENCODER_ONLY_EXACT_BYTE_DIAGNOSTIC"
G51_ADVERSARIAL_RECEIPT_SHA256: Final = "756dd421dd8b6ad21dd6d8b7ed271bdbac1ac84b036ab92bf36b40686a998a13"
G51_ADVERSARIAL_SOURCE_COMMIT: Final = "b84b4c6d948f24f3aa399c1774f557dcecfa3658"

UNTRUSTED_SCORER_EFFECT_RECEIPT_SCHEMA: Final = "tac.taskspace_g51_untrusted_scorer_effect_candidate.v1"
SAME_OBJECT_ARCHIVE_DELTA_SCHEMA: Final = "tac.taskspace_g51_same_object_archive_zip_delta.v1"
EFFECT_BUNDLE_FORMAT: Final = "npz_uncompressed_or_compressed_numpy_v1"
PUBLIC_PAIR_COUNT: Final = 600
EFFECT_AXES: Final = ("seg_score_term_delta", "pose_score_term_delta")
CANONICAL_MATERIALIZER_BLOCKER: Final = "G51_CANONICAL_RECEIVER_R_SCORER_EFFECT_MATERIALIZER_OWED"

ConsumerId = Literal[
    "sensitivity_map",
    "bit_allocator",
    "costate_controller",
    "costate_organ",
    "cathedral_autopilot",
    "pareto_solver",
]

BLOCKED_CONSUMERS: Final[tuple[ConsumerId, ...]] = (
    "sensitivity_map",
    "bit_allocator",
    "costate_controller",
    "costate_organ",
    "cathedral_autopilot",
    "pareto_solver",
)

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z")
_EFFECT_ARRAY_KEYS: Final = (
    "pair_ids",
    "scorer_term_effect_vectors",
    "realized_scorer_jvp",
    "realized_scorer_vjp",
)
_VECTOR_ARRAY_KEYS: Final = _EFFECT_ARRAY_KEYS[1:]
_FORBIDDEN_EXACT_KEYS: Final = frozenset(
    {
        "zlib9_marginal_bytes",
        "per_pair_marginals",
        "ambient_unweighted_gram",
        "ambient_unweighted_group_gram",
        "functional_operator_groups",
        "functional_operator_proposal_surface",
        "best_tested_exact_basis",
        "best_archive",
        "best_archive_claim",
    }
)
_FORBIDDEN_STRING_REFERENCES: Final = frozenset(
    {
        "zlib9_marginal_bytes",
        "per_pair_marginals",
        "ambient_unweighted_gram",
        "ambient_unweighted_group_gram",
        "functional_operator_proposal_surface",
        "conditional_budget_arbitration.best_tested_exact_basis",
    }
)
_FILE_REF_KEYS: Final = frozenset({"path", "bytes", "sha256"})
_FALSE_AUTHORITY: Final = {
    "research_only": True,
    "score_claim": False,
    "candidate_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


class G51CostateAdmissionError(ValueError):
    """Raised before any actionable consumer receives G51-adjacent data."""


@dataclass(frozen=True, slots=True)
class ArtifactIdentityV1:
    """An exact regular-file identity reopened by the guard."""

    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class G51EncoderDiagnosticV1:
    """Quarantined identity of the exact b84 G51 interpretation receipt."""

    receipt: ArtifactIdentityV1
    source_commit: str
    profile_status: str
    pair_count: int
    allowed_use: str = "ENCODER_DIAGNOSTIC_ONLY"
    actionable_costate_input: bool = False
    actionable_consumers: tuple[ConsumerId, ...] = ()
    blockers: tuple[str, ...] = (
        "G51_SCORER_EFFECT_VECTORS_OWED",
        "G51_REALIZED_SCORER_JVP_VJP_OWED",
        "G51_SAME_OBJECT_ARCHIVE_ZIP_DELTA_OWED",
        CANONICAL_MATERIALIZER_BLOCKER,
    )


@dataclass(frozen=True, slots=True)
class G51CostateEvidenceCandidateV1:
    """Integrity-checked but untrusted and non-actionable evidence candidate."""

    diagnostic: G51EncoderDiagnosticV1
    object_id: str
    transition_id: str
    pair_count: int
    effect_receipt: ArtifactIdentityV1
    effect_bundle: ArtifactIdentityV1
    archive_delta_receipt: ArtifactIdentityV1
    baseline_archive: ArtifactIdentityV1
    candidate_archive: ArtifactIdentityV1
    archive_zip_delta_bytes: int
    serialized_archive_delta_contract: Mapping[str, Any]
    effect_array_sha256: Mapping[str, str]
    integrity_checks_passed: bool = True
    claimed_measurement_not_proven: bool = True
    canonical_materializer_bound: bool = False
    actionable_costate_input: bool = False
    actionable_consumers: tuple[ConsumerId, ...] = ()
    blocked_consumers: tuple[ConsumerId, ...] = BLOCKED_CONSUMERS
    blockers: tuple[str, ...] = (CANONICAL_MATERIALIZER_BLOCKER,)
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def refuse_consumer(self, consumer_id: ConsumerId) -> NoReturn:
        """Fail closed for every actionable consumer until a trusted emitter exists."""

        if consumer_id not in self.blocked_consumers:
            raise G51CostateAdmissionError(f"unknown G51 consumer request: {consumer_id!r}")
        raise G51CostateAdmissionError(f"{CANONICAL_MATERIALIZER_BLOCKER}:{consumer_id}")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise G51CostateAdmissionError(f"{field_name} must be a lowercase SHA-256")
    return value


def _require_identifier(value: object, field_name: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise G51CostateAdmissionError(f"{field_name} must be a bounded canonical identifier")
    return value


def _require_exact_int(
    value: object,
    field_name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise G51CostateAdmissionError(f"{field_name} must be an exact integer in [{minimum}, {maximum}]")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise G51CostateAdmissionError("value is not finite canonical JSON") from exc


def _regular_file_bytes(path: Path, field_name: str) -> bytes:
    if path.is_symlink():
        raise G51CostateAdmissionError(f"{field_name} must not be a symlink")
    try:
        mode = path.stat().st_mode
        if not stat.S_ISREG(mode):
            raise G51CostateAdmissionError(f"{field_name} must be a regular file")
        return path.read_bytes()
    except OSError as exc:
        raise G51CostateAdmissionError(f"cannot reopen {field_name}: {path}") from exc


def _load_json_object(
    path: Path,
    *,
    field_name: str,
    canonical: bool,
) -> tuple[dict[str, Any], bytes]:
    payload = _regular_file_bytes(path, field_name)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G51CostateAdmissionError(f"{field_name} is not a JSON object") from exc
    if not isinstance(value, dict):
        raise G51CostateAdmissionError(f"{field_name} must be a JSON object")
    if canonical and _canonical_json(value) != payload:
        raise G51CostateAdmissionError(f"{field_name} must use canonical JSON bytes")
    return value, payload


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    field_name: str,
) -> None:
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        extra = sorted(set(value) - set(expected))
        raise G51CostateAdmissionError(f"{field_name} keys differ; missing={missing}, extra={extra}")


def _scan_forbidden_historical_fields(
    value: object,
    *,
    path: str = "$",
) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            key_lower = key.lower()
            if (
                key in _FORBIDDEN_EXACT_KEYS
                or key_lower.startswith("fits_")
                or ("best" in key_lower and ("archive" in key_lower or "basis" in key_lower))
            ):
                raise G51CostateAdmissionError(f"forbidden historical G51 field at {path}.{key}")
            if key_lower == "status" and isinstance(child, str) and child.upper().startswith("READY"):
                raise G51CostateAdmissionError(f"declarative READY hook is forbidden at {path}.{key}")
            _scan_forbidden_historical_fields(child, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _scan_forbidden_historical_fields(
                child,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(value, str) and value in _FORBIDDEN_STRING_REFERENCES:
        raise G51CostateAdmissionError(f"forbidden historical G51 payload reference at {path}")


def _require_false_authority(
    value: object,
    field_name: str,
) -> None:
    if not isinstance(value, Mapping) or dict(value) != _FALSE_AUTHORITY:
        raise G51CostateAdmissionError(f"{field_name} must equal the canonical false-authority boundary")


def _file_ref(
    value: object,
    *,
    receipt_dir: Path,
    field_name: str,
) -> tuple[ArtifactIdentityV1, bytes]:
    if not isinstance(value, Mapping):
        raise G51CostateAdmissionError(f"{field_name} must be a file reference")
    _require_exact_keys(value, _FILE_REF_KEYS, field_name)
    raw_path = value.get("path")
    if type(raw_path) is not str or not raw_path:
        raise G51CostateAdmissionError(f"{field_name}.path must be non-empty")
    path = Path(raw_path)
    if not path.is_absolute():
        path = receipt_dir / path
    payload = _regular_file_bytes(path, field_name)
    expected_bytes = _require_exact_int(
        value.get("bytes"),
        f"{field_name}.bytes",
        minimum=1,
        maximum=1 << 40,
    )
    expected_sha = _require_sha256(
        value.get("sha256"),
        f"{field_name}.sha256",
    )
    if len(payload) != expected_bytes:
        raise G51CostateAdmissionError(f"{field_name} byte count differs")
    actual_sha = _sha256(payload)
    if actual_sha != expected_sha:
        raise G51CostateAdmissionError(f"{field_name} SHA-256 differs")
    return (
        ArtifactIdentityV1(
            path=str(path.resolve()),
            bytes=len(payload),
            sha256=actual_sha,
        ),
        payload,
    )


def _validate_zip(path: Path, field_name: str) -> None:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            if not infos:
                raise G51CostateAdmissionError(f"{field_name} is an empty ZIP")
            names: set[str] = set()
            total_uncompressed = 0
            for info in infos:
                pure = PurePosixPath(info.filename)
                if info.filename in names or pure.is_absolute() or ".." in pure.parts or not info.filename:
                    raise G51CostateAdmissionError(f"{field_name} has unsafe or duplicate member names")
                names.add(info.filename)
                if info.flag_bits & 0x1:
                    raise G51CostateAdmissionError(f"{field_name} must not contain encrypted members")
                member_mode = (info.external_attr >> 16) & 0xFFFF
                if member_mode and stat.S_ISLNK(member_mode):
                    raise G51CostateAdmissionError(f"{field_name} must not contain symlink members")
                total_uncompressed += info.file_size
                if total_uncompressed > 16 * (1 << 30):
                    raise G51CostateAdmissionError(f"{field_name} exceeds the 16 GiB guard ceiling")
                if not info.is_dir():
                    with archive.open(info, "r") as member:
                        while member.read(1 << 20):
                            pass
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise G51CostateAdmissionError(f"{field_name} is not a valid readable ZIP") from exc


def load_g51_encoder_diagnostic(
    receipt_path: str | Path,
) -> G51EncoderDiagnosticV1:
    """Accept only the exact b84 G51 receipt, and quarantine its payload."""

    path = Path(receipt_path)
    value, payload = _load_json_object(
        path,
        field_name="G51 adversarial interpretation receipt",
        canonical=False,
    )
    digest = _sha256(payload)
    if digest != G51_ADVERSARIAL_RECEIPT_SHA256:
        raise G51CostateAdmissionError("G51 adversarial receipt is not the exact b84 interpretation")
    corrected = value.get("corrected_interpretation")
    governance = value.get("execution_governance")
    if (
        value.get("schema") != G51_ADVERSARIAL_SCHEMA
        or value.get("profile_status") != G51_ADVERSARIAL_PROFILE_STATUS
        or value.get("population_pairs") != PUBLIC_PAIR_COUNT
        or not isinstance(corrected, Mapping)
        or corrected.get("archive_feasibility_claim_allowed") is not False
        or corrected.get("smallest_or_best_archive_claim_allowed") is not False
        or corrected.get("per_pair_values_are_archive_marginals") is not False
        or corrected.get("functional_gram_measured") is not False
        or corrected.get("actuator_effect_gram_measured") is not False
        or corrected.get("bit_allocator_wired") is not False
        or corrected.get("cathedral_autopilot_wired") is not False
        or not isinstance(governance, Mapping)
        or governance.get("candidate_or_promotion_authority") is not False
        or value.get("score_claim") is not False
        or value.get("candidate_payload") is not False
        or value.get("promotion_eligible") is not False
        or value.get("pointer_mutation_performed") is not False
    ):
        raise G51CostateAdmissionError("G51 adversarial receipt weakened its diagnostic-only interpretation")
    blockers = value.get("remaining_blockers")
    required_blockers = {
        "same-object ZIP bytes and ZIP delta were not measured",
        "scorer-costate or JVP/VJP actuator effects were not measured",
    }
    if not isinstance(blockers, list) or not required_blockers.issubset(set(blockers)):
        raise G51CostateAdmissionError("G51 adversarial receipt no longer records the costate blockers")
    return G51EncoderDiagnosticV1(
        receipt=ArtifactIdentityV1(
            path=str(path.resolve()),
            bytes=len(payload),
            sha256=digest,
        ),
        source_commit=G51_ADVERSARIAL_SOURCE_COMMIT,
        profile_status=G51_ADVERSARIAL_PROFILE_STATUS,
        pair_count=PUBLIC_PAIR_COUNT,
    )


def _load_archive_delta(
    receipt_path: Path,
    *,
    expected_effect_bundle_sha256: str,
) -> tuple[
    dict[str, Any],
    ArtifactIdentityV1,
    ArtifactIdentityV1,
    ArtifactIdentityV1,
    dict[str, Any],
]:
    value, payload = _load_json_object(
        receipt_path,
        field_name="same-object archive ZIP delta receipt",
        canonical=True,
    )
    _scan_forbidden_historical_fields(value)
    _require_exact_keys(
        value,
        {
            "schema",
            "object_id",
            "transition_id",
            "population_pairs",
            "same_object_lifecycle",
            "outer_zip_delta_measured",
            "baseline_archive",
            "candidate_archive",
            "measured_zip_delta_bytes",
            "receiver_runtime",
            "public_evaluator_source",
            "population_identity_sha256",
            "baseline_receiver_output_sha256",
            "candidate_receiver_output_sha256",
            "effect_bundle_sha256",
            "truth",
        },
        "same-object archive ZIP delta receipt",
    )
    if (
        value.get("schema") != SAME_OBJECT_ARCHIVE_DELTA_SCHEMA
        or value.get("population_pairs") != PUBLIC_PAIR_COUNT
        or value.get("same_object_lifecycle") is not True
        or value.get("outer_zip_delta_measured") is not True
    ):
        raise G51CostateAdmissionError("archive delta is not a full-n600 same-object outer-ZIP measurement")
    _require_identifier(value.get("object_id"), "archive_delta.object_id")
    _require_identifier(
        value.get("transition_id"),
        "archive_delta.transition_id",
    )
    _require_sha256(
        value.get("population_identity_sha256"),
        "archive_delta.population_identity_sha256",
    )
    _require_sha256(
        value.get("baseline_receiver_output_sha256"),
        "archive_delta.baseline_receiver_output_sha256",
    )
    _require_sha256(
        value.get("candidate_receiver_output_sha256"),
        "archive_delta.candidate_receiver_output_sha256",
    )
    if (
        _require_sha256(
            value.get("effect_bundle_sha256"),
            "archive_delta.effect_bundle_sha256",
        )
        != expected_effect_bundle_sha256
    ):
        raise G51CostateAdmissionError("archive delta does not bind the admitted effect bundle")
    _require_false_authority(value.get("truth"), "archive_delta.truth")
    receipt_dir = receipt_path.parent
    baseline, _ = _file_ref(
        value.get("baseline_archive"),
        receipt_dir=receipt_dir,
        field_name="archive_delta.baseline_archive",
    )
    candidate, _ = _file_ref(
        value.get("candidate_archive"),
        receipt_dir=receipt_dir,
        field_name="archive_delta.candidate_archive",
    )
    _file_ref(
        value.get("receiver_runtime"),
        receipt_dir=receipt_dir,
        field_name="archive_delta.receiver_runtime",
    )
    _file_ref(
        value.get("public_evaluator_source"),
        receipt_dir=receipt_dir,
        field_name="archive_delta.public_evaluator_source",
    )
    if baseline.sha256 == candidate.sha256:
        raise G51CostateAdmissionError("baseline and candidate ZIP identities must differ")
    _validate_zip(Path(baseline.path), "archive_delta.baseline_archive")
    _validate_zip(Path(candidate.path), "archive_delta.candidate_archive")
    actual_delta = candidate.bytes - baseline.bytes
    claimed_delta = value.get("measured_zip_delta_bytes")
    if isinstance(claimed_delta, bool) or not isinstance(claimed_delta, int) or claimed_delta != actual_delta:
        raise G51CostateAdmissionError("same-object archive ZIP delta does not match reopened files")
    delta_contract = build_serialized_archive_delta_contract(
        source_archive_bytes=baseline.bytes,
        candidate_archive_bytes=candidate.bytes,
    )
    if (
        delta_contract.get("schema") != SERIALIZED_ARCHIVE_DELTA_SCHEMA
        or delta_contract.get("archive_delta_bytes") != actual_delta
        or delta_contract.get("score_claim") is not False
        or delta_contract.get("promotion_eligible") is not False
    ):
        raise G51CostateAdmissionError("canonical serialized archive delta contract disagrees")
    receipt_identity = ArtifactIdentityV1(
        path=str(receipt_path.resolve()),
        bytes=len(payload),
        sha256=_sha256(payload),
    )
    return value, receipt_identity, baseline, candidate, delta_contract


def _expected_array_specs() -> dict[str, Any]:
    vector = {
        "dtype": "float32",
        "shape": [PUBLIC_PAIR_COUNT, len(EFFECT_AXES)],
        "axes": list(EFFECT_AXES),
    }
    return {
        "pair_ids": {
            "dtype": "int32",
            "shape": [PUBLIC_PAIR_COUNT],
            "semantic": "ordered_public_pair_ids_0_through_599",
        },
        "scorer_term_effect_vectors": dict(vector),
        "realized_scorer_jvp": dict(vector),
        "realized_scorer_vjp": dict(vector),
    }


def _load_effect_arrays(
    identity: ArtifactIdentityV1,
) -> dict[str, str]:
    try:
        with np.load(identity.path, allow_pickle=False) as bundle:
            if tuple(sorted(bundle.files)) != tuple(sorted(_EFFECT_ARRAY_KEYS)):
                raise G51CostateAdmissionError("effect bundle array registry differs")
            pair_ids = np.asarray(bundle["pair_ids"])
            arrays = {key: np.asarray(bundle[key]) for key in _VECTOR_ARRAY_KEYS}
    except (OSError, ValueError, KeyError) as exc:
        raise G51CostateAdmissionError("effect bundle is not a strict NumPy NPZ") from exc
    if (
        pair_ids.dtype != np.dtype(np.int32)
        or pair_ids.shape != (PUBLIC_PAIR_COUNT,)
        or not np.array_equal(
            pair_ids,
            np.arange(PUBLIC_PAIR_COUNT, dtype=np.int32),
        )
    ):
        raise G51CostateAdmissionError("effect bundle must cover ordered public pair ids 0..599")
    for key, array in arrays.items():
        if (
            array.dtype != np.dtype(np.float32)
            or array.shape != (PUBLIC_PAIR_COUNT, len(EFFECT_AXES))
            or not np.isfinite(array).all()
            or not np.any(array != 0.0)
        ):
            raise G51CostateAdmissionError(f"{key} must be finite, nonzero, float32, and shape (600, 2)")
    identities: dict[str, str] = {}
    for key, array in {"pair_ids": pair_ids, **arrays}.items():
        identities[key] = _sha256(
            _canonical_json(
                {
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                }
            )
            + array.tobytes(order="C")
        )
    return identities


def inspect_g51_costate_evidence_candidate(
    *,
    diagnostic_receipt_path: str | Path,
    scorer_effect_receipt_path: str | Path,
    archive_delta_receipt_path: str | Path,
) -> G51CostateEvidenceCandidateV1:
    """Validate artifact integrity without granting actionable authority."""

    diagnostic = load_g51_encoder_diagnostic(diagnostic_receipt_path)
    effect_receipt_path = Path(scorer_effect_receipt_path)
    effect_value, effect_payload = _load_json_object(
        effect_receipt_path,
        field_name="scorer effect receipt",
        canonical=True,
    )
    _scan_forbidden_historical_fields(effect_value)
    _require_exact_keys(
        effect_value,
        {
            "schema",
            "encoder_diagnostic_sha256",
            "object_id",
            "transition_id",
            "population_pairs",
            "baseline_archive_sha256",
            "candidate_archive_sha256",
            "baseline_receiver_output_sha256",
            "candidate_receiver_output_sha256",
            "archive_delta_receipt_sha256",
            "effect_bundle",
            "effect_bundle_format",
            "effect_arrays",
            "measurement_claims",
            "truth",
        },
        "scorer effect receipt",
    )
    if (
        effect_value.get("schema") != UNTRUSTED_SCORER_EFFECT_RECEIPT_SCHEMA
        or effect_value.get("population_pairs") != PUBLIC_PAIR_COUNT
        or effect_value.get("effect_bundle_format") != EFFECT_BUNDLE_FORMAT
        or effect_value.get("effect_arrays") != _expected_array_specs()
    ):
        raise G51CostateAdmissionError("scorer effect receipt schema/population/array contract differs")
    if (
        _require_sha256(
            effect_value.get("encoder_diagnostic_sha256"),
            "effect.encoder_diagnostic_sha256",
        )
        != diagnostic.receipt.sha256
    ):
        raise G51CostateAdmissionError("scorer effect receipt does not bind the exact G51 diagnostic")
    object_id = _require_identifier(
        effect_value.get("object_id"),
        "effect.object_id",
    )
    transition_id = _require_identifier(
        effect_value.get("transition_id"),
        "effect.transition_id",
    )
    for field_name in (
        "baseline_archive_sha256",
        "candidate_archive_sha256",
        "baseline_receiver_output_sha256",
        "candidate_receiver_output_sha256",
        "archive_delta_receipt_sha256",
    ):
        _require_sha256(
            effect_value.get(field_name),
            f"effect.{field_name}",
        )
    _require_false_authority(effect_value.get("truth"), "effect.truth")
    measurement_claims = effect_value.get("measurement_claims")
    expected_measurement_claims = {
        "public_receiver_invoked": True,
        "realized_through_R": True,
        "frozen_cpu_torch_segnet_invoked": True,
        "frozen_cpu_torch_posenet_invoked": True,
        "scorer_effect_vectors_measured": True,
        "realized_scorer_jvp_measured": True,
        "realized_scorer_vjp_measured": True,
        "numpy_fp32_reference": True,
        "mps_authority": False,
        "proxy": False,
    }
    if measurement_claims != expected_measurement_claims:
        raise G51CostateAdmissionError("scorer effect candidate lacks the claimed receiver/R/scorer/JVP/VJP boundary")
    effect_bundle, _ = _file_ref(
        effect_value.get("effect_bundle"),
        receipt_dir=effect_receipt_path.parent,
        field_name="effect.effect_bundle",
    )
    effect_array_sha256 = _load_effect_arrays(effect_bundle)

    archive_path = Path(archive_delta_receipt_path)
    archive_payload = _regular_file_bytes(
        archive_path,
        "same-object archive ZIP delta receipt",
    )
    if _sha256(archive_payload) != effect_value["archive_delta_receipt_sha256"]:
        raise G51CostateAdmissionError("effect receipt does not bind the archive delta receipt bytes")
    (
        archive_value,
        archive_identity,
        baseline,
        candidate,
        delta_contract,
    ) = _load_archive_delta(
        archive_path,
        expected_effect_bundle_sha256=effect_bundle.sha256,
    )
    if (
        archive_value["object_id"] != object_id
        or archive_value["transition_id"] != transition_id
        or archive_value["baseline_archive"]["sha256"] != effect_value["baseline_archive_sha256"]
        or archive_value["candidate_archive"]["sha256"] != effect_value["candidate_archive_sha256"]
        or archive_value["baseline_receiver_output_sha256"] != effect_value["baseline_receiver_output_sha256"]
        or archive_value["candidate_receiver_output_sha256"] != effect_value["candidate_receiver_output_sha256"]
    ):
        raise G51CostateAdmissionError("scorer effects and archive delta are not the same object transition")
    effect_identity = ArtifactIdentityV1(
        path=str(effect_receipt_path.resolve()),
        bytes=len(effect_payload),
        sha256=_sha256(effect_payload),
    )
    return G51CostateEvidenceCandidateV1(
        diagnostic=diagnostic,
        object_id=object_id,
        transition_id=transition_id,
        pair_count=PUBLIC_PAIR_COUNT,
        effect_receipt=effect_identity,
        effect_bundle=effect_bundle,
        archive_delta_receipt=archive_identity,
        baseline_archive=baseline,
        candidate_archive=candidate,
        archive_zip_delta_bytes=int(delta_contract["archive_delta_bytes"]),
        serialized_archive_delta_contract=delta_contract,
        effect_array_sha256=effect_array_sha256,
        integrity_checks_passed=True,
        claimed_measurement_not_proven=True,
        canonical_materializer_bound=False,
        actionable_costate_input=False,
        score_claim=False,
        promotion_eligible=False,
        ready_for_exact_eval_dispatch=False,
    )


def admit_g51_actionable_costate_evidence(
    *,
    diagnostic_receipt_path: str | Path,
    scorer_effect_receipt_path: str | Path,
    archive_delta_receipt_path: str | Path,
) -> NoReturn:
    """Always refuse until a canonical receiver/R/scorer materializer lands."""

    candidate = inspect_g51_costate_evidence_candidate(
        diagnostic_receipt_path=diagnostic_receipt_path,
        scorer_effect_receipt_path=scorer_effect_receipt_path,
        archive_delta_receipt_path=archive_delta_receipt_path,
    )
    if (
        candidate.integrity_checks_passed is not True
        or candidate.actionable_costate_input is not False
        or candidate.canonical_materializer_bound is not False
    ):
        raise G51CostateAdmissionError("G51 evidence candidate diagnostic boundary drifted")
    raise G51CostateAdmissionError(CANONICAL_MATERIALIZER_BLOCKER)


def request_g51_costate_bit_allocation(
    *,
    diagnostic_receipt_path: str | Path,
    scorer_effect_receipt_path: str | Path,
    archive_delta_receipt_path: str | Path,
) -> NoReturn:
    """Real bit-allocation guard call; no allocator is invoked on current evidence."""

    admit_g51_actionable_costate_evidence(
        diagnostic_receipt_path=diagnostic_receipt_path,
        scorer_effect_receipt_path=scorer_effect_receipt_path,
        archive_delta_receipt_path=archive_delta_receipt_path,
    )
    raise AssertionError("unreachable: G51 admission must fail closed")


__all__ = [
    "BLOCKED_CONSUMERS",
    "CANONICAL_MATERIALIZER_BLOCKER",
    "EFFECT_AXES",
    "EFFECT_BUNDLE_FORMAT",
    "G51_ADVERSARIAL_RECEIPT_SHA256",
    "G51_ADVERSARIAL_SCHEMA",
    "G51_ADVERSARIAL_SOURCE_COMMIT",
    "PUBLIC_PAIR_COUNT",
    "SAME_OBJECT_ARCHIVE_DELTA_SCHEMA",
    "UNTRUSTED_SCORER_EFFECT_RECEIPT_SCHEMA",
    "G51CostateAdmissionError",
    "G51CostateEvidenceCandidateV1",
    "G51EncoderDiagnosticV1",
    "admit_g51_actionable_costate_evidence",
    "inspect_g51_costate_evidence_candidate",
    "load_g51_encoder_diagnostic",
    "request_g51_costate_bit_allocation",
]
