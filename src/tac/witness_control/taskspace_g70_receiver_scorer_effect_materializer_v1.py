# SPDX-License-Identifier: MIT
"""Exact G17 receiver/scorer finite effects with a fail-closed derivative gate.

Two dense-retaining :class:`G17CandidateForwardObservationV1` objects are
enough to derive exact full-population endpoint effects and an outer-ZIP byte
delta.  They are not enough to prove JVP/VJP contractions.  In particular, the
existing ``vjp_custody_n600_extension.v1`` campaign differentiates the frozen
scorers at the ``gt_n600`` source arrangement and does not bind either G17
archive endpoint.

This module therefore does two real, deliberately separate things:

* materializes the finite endpoint effect of one exact n600 G17
  baseline/candidate transition; and
* audits an existing frozen-scorer VJP campaign, then refuses actionable
  costate admission because that campaign cannot prove differentiation at the
  same transition.

No caller-provided effect, JVP, or VJP array is accepted.  There is no
synthetic or self-attested actionable path.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final, NoReturn

import numpy as np

from tac.optimization.serialized_archive_economics import (
    build_serialized_archive_delta_contract,
)
from tac.witness_dsl.taskspace_g17_forward_observation import (
    G17CandidateForwardObservationV1,
    require_candidate_observation_matches_exact_objects,
)

PUBLIC_PAIR_COUNT: Final = 600
PUBLIC_PAIR_IDS: Final = tuple(range(PUBLIC_PAIR_COUNT))
CONTEST_SOURCE_BYTES: Final = 37_545_489
SEG_SCORE_WEIGHT: Final = 100.0
POSE_SCORE_WEIGHT: Final = 10.0
RATE_SCORE_WEIGHT: Final = 25.0

FINITE_EFFECT_SCHEMA: Final = "tac.taskspace_g70_receiver_scorer_finite_effect.v1"
EFFECT_BUNDLE_SCHEMA: Final = "tac.taskspace_g70_receiver_scorer_effect_bundle.v1"
VJP_AUDIT_SCHEMA: Final = "tac.taskspace_g70_vjp_transition_join_audit.v1"
BLOCKER_RECEIPT_SCHEMA: Final = "tac.taskspace_g70_receiver_scorer_effect_blocker.v1"
VJP_CAMPAIGN_SCHEMA: Final = "vjp_custody_n600_extension.v1"
CANONICAL_VJP_CAMPAIGN_PATH: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/extension_n600_20260720/campaign_receipt.json"
)
CANONICAL_VJP_CAMPAIGN_BYTES: Final = 32_297
CANONICAL_VJP_CAMPAIGN_SHA256: Final = "11d441acec2640f607e87514e35c9244c9d8dfec617a810b2c9763b2e116d284"
DERIVATIVE_JOIN_BLOCKER: Final = "G70_TRANSITION_ANCHORED_SCORER_DIFFERENTIATION_OWED"
SOURCE_VJP_RELABEL_BLOCKER: Final = "G70_GT_SOURCE_VJP_CANNOT_BE_RELABELED_AS_G17_TRANSITION_JVP_VJP"
G67_REVIEW_BLOCKER: Final = "G70_G67_ADVERSARIAL_ADMISSION_REVIEW_OWED"

_BUNDLE_ARRAYS: Final = (
    "pair_ids",
    "scorer_term_effect_vectors",
    "baseline_per_pair_d_seg",
    "candidate_per_pair_d_seg",
    "baseline_per_pair_d_pose",
    "candidate_per_pair_d_pose",
)
_FALSE_AUTHORITY: Final = {
    "research_only": True,
    "score_claim": False,
    "candidate_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


class G70EffectMaterializationError(ValueError):
    """An exact object join, archive, campaign, or admission failed closed."""


@dataclass(frozen=True, slots=True)
class ArchiveIdentityV1:
    """Exact in-memory outer-ZIP identity after member-stream reopening."""

    bytes: int
    sha256: str
    member_table_sha256: str
    matching_g17_member_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VJPCampaignTransitionAuditV1:
    """Identity-only audit of the existing source-anchored n600 VJP campaign."""

    schema: str
    campaign_path: str
    campaign_bytes: int
    campaign_sha256: str
    campaign_schema: str
    campaign_status: str
    final_completed_count: int
    final_pair_ids_sha256: str
    refused_pair_ids: tuple[int, ...]
    still_missing_pair_ids: tuple[int, ...]
    actual_frozen_scorer_vjp_campaign: bool
    g17_transition_bound: bool
    realized_transition_jvp_present: bool
    realized_transition_vjp_present: bool
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "campaign_path": self.campaign_path,
            "campaign_bytes": self.campaign_bytes,
            "campaign_sha256": self.campaign_sha256,
            "campaign_schema": self.campaign_schema,
            "campaign_status": self.campaign_status,
            "final_completed_count": self.final_completed_count,
            "final_pair_ids_sha256": self.final_pair_ids_sha256,
            "refused_pair_ids": list(self.refused_pair_ids),
            "still_missing_pair_ids": list(self.still_missing_pair_ids),
            "actual_frozen_scorer_vjp_campaign": self.actual_frozen_scorer_vjp_campaign,
            "g17_transition_bound": self.g17_transition_bound,
            "realized_transition_jvp_present": self.realized_transition_jvp_present,
            "realized_transition_vjp_present": self.realized_transition_vjp_present,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True, slots=True)
class G70FiniteEffectTransitionV1:
    """Real endpoint effect derived from two exact, same-target G17 forwards."""

    baseline: G17CandidateForwardObservationV1 = field(repr=False)
    candidate: G17CandidateForwardObservationV1 = field(repr=False)
    transition_id: str
    baseline_archive: ArchiveIdentityV1
    candidate_archive: ArchiveIdentityV1
    pair_ids: np.ndarray = field(repr=False)
    scorer_term_effect_vectors: np.ndarray = field(repr=False)
    baseline_per_pair_d_seg: np.ndarray = field(repr=False)
    candidate_per_pair_d_seg: np.ndarray = field(repr=False)
    baseline_per_pair_d_pose: np.ndarray = field(repr=False)
    candidate_per_pair_d_pose: np.ndarray = field(repr=False)
    aggregate_effects: dict[str, float]
    serialized_archive_delta_contract: dict[str, Any]
    derivative_audit: VJPCampaignTransitionAuditV1 | None = None
    actionable_costate_input: bool = False
    blockers: tuple[str, ...] = (
        DERIVATIVE_JOIN_BLOCKER,
        G67_REVIEW_BLOCKER,
    )

    def refuse_actionable_admission(self) -> NoReturn:
        """Prevent finite endpoint effects from masquerading as a costate."""

        raise G70EffectMaterializationError(":".join(self.blockers))


@dataclass(frozen=True, slots=True)
class G70EffectBundleFilesV1:
    """Paths and identities of one deterministic, non-actionable emission."""

    bundle_path: Path
    bundle_bytes: int
    bundle_sha256: str
    receipt_path: Path
    receipt_bytes: int
    receipt_sha256: str


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise G70EffectMaterializationError("value is not finite canonical ASCII JSON") from exc


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value)).cast("B"))


def _immutable(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype).copy()
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise G70EffectMaterializationError("derived effect array contains nonfinite values")
    array.setflags(write=False)
    return array


def _validate_zip(archive_bytes: bytes, *, g17_member_bytes: bytes, label: str) -> ArchiveIdentityV1:
    if type(archive_bytes) is not bytes or not archive_bytes:
        raise G70EffectMaterializationError(f"{label} archive must be nonempty exact bytes")
    records: list[dict[str, Any]] = []
    matched_members: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if not infos or len(names) != len(set(names)):
                raise G70EffectMaterializationError(f"{label} ZIP is empty or repeats member names")
            for info in infos:
                path = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if (
                    not info.filename
                    or "\\" in info.filename
                    or path.is_absolute()
                    or ".." in path.parts
                    or info.flag_bits & 0x1
                    or stat.S_ISLNK(mode)
                ):
                    raise G70EffectMaterializationError(f"{label} ZIP has an unsafe member")
                payload = archive.read(info)
                if payload == g17_member_bytes:
                    matched_members.append(info.filename)
                records.append(
                    {
                        "name": info.filename,
                        "is_dir": info.is_dir(),
                        "file_size": info.file_size,
                        "compress_size": info.compress_size,
                        "crc32": info.CRC,
                        "content_sha256": _sha256(payload),
                    }
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise G70EffectMaterializationError(f"{label} ZIP CRC failed at {bad_member!r}")
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise G70EffectMaterializationError(f"{label} archive is not a valid reopenable ZIP") from exc
    if len(matched_members) != 1:
        raise G70EffectMaterializationError(
            f"{label} outer ZIP must contain exactly one member equal to the G17 member bytes"
        )
    return ArchiveIdentityV1(
        bytes=len(archive_bytes),
        sha256=_sha256(archive_bytes),
        member_table_sha256=_sha256(_canonical_json(records)),
        matching_g17_member_names=tuple(matched_members),
    )


def _require_exact_observation_join(
    baseline: G17CandidateForwardObservationV1,
    candidate: G17CandidateForwardObservationV1,
) -> None:
    if (
        type(baseline) is not G17CandidateForwardObservationV1
        or type(candidate) is not G17CandidateForwardObservationV1
    ):
        raise G70EffectMaterializationError("both endpoints must be exact G17 candidate observations")
    if baseline.target is not candidate.target:
        raise G70EffectMaterializationError("baseline and candidate must retain the same exact target object")
    if baseline.target.source_pair_ids != PUBLIC_PAIR_IDS:
        raise G70EffectMaterializationError("G70 only materializes the complete ordered n600 population")
    for observation in (baseline, candidate):
        require_candidate_observation_matches_exact_objects(
            observation,
            target=baseline.target,
            archive_bytes=observation.archive_bytes,
            member_bytes=observation.member_bytes,
            receiver_receipt_bytes=observation.receiver_receipt_bytes,
            decoded_output_bytes=observation.decoded_output_bytes,
        )
    if baseline.receipt.archive_sha256 == candidate.receipt.archive_sha256:
        raise G70EffectMaterializationError("baseline and candidate outer ZIP identities must differ")
    for field_name in (
        "target_forward_receipt_sha256",
        "frozen_scorer_sha256",
        "scorer_runtime_environment_sha256",
    ):
        if getattr(baseline.receipt, field_name) != getattr(candidate.receipt, field_name):
            raise G70EffectMaterializationError(f"baseline/candidate {field_name} differs")


def _pose_additive_effects(
    baseline_per_pair: np.ndarray,
    candidate_per_pair: np.ndarray,
    *,
    baseline_aggregate: float,
    candidate_aggregate: float,
) -> np.ndarray:
    """Aumann-Shapley attribution whose sum is the exact population pose delta."""

    aggregate_delta = candidate_aggregate - baseline_aggregate
    term_delta = math.sqrt(POSE_SCORE_WEIGHT * candidate_aggregate) - math.sqrt(POSE_SCORE_WEIGHT * baseline_aggregate)
    per_pair_delta = candidate_per_pair - baseline_per_pair
    if aggregate_delta == 0.0:
        if baseline_aggregate == 0.0:
            result = np.zeros_like(per_pair_delta, dtype=np.float64)
        else:
            derivative = math.sqrt(POSE_SCORE_WEIGHT) / (2.0 * math.sqrt(baseline_aggregate))
            result = derivative * per_pair_delta / PUBLIC_PAIR_COUNT
    else:
        secant = term_delta / aggregate_delta
        result = secant * per_pair_delta / PUBLIC_PAIR_COUNT
    residual = term_delta - float(np.sum(result, dtype=np.float64))
    result[-1] += residual
    return result


def audit_vjp_campaign_for_transition(
    campaign_receipt_path: Path,
) -> VJPCampaignTransitionAuditV1:
    """Validate n600 campaign completion, then classify its missing G17 join."""

    requested = campaign_receipt_path.expanduser()
    absolute = requested if requested.is_absolute() else Path.cwd() / requested
    try:
        requested_stat = absolute.lstat()
    except OSError as exc:
        raise G70EffectMaterializationError("cannot lstat VJP campaign receipt") from exc
    if stat.S_ISLNK(requested_stat.st_mode) or not stat.S_ISREG(requested_stat.st_mode):
        raise G70EffectMaterializationError("VJP campaign receipt must be a regular non-symlink file")
    path = absolute.resolve(strict=True)
    if path != CANONICAL_VJP_CAMPAIGN_PATH:
        raise G70EffectMaterializationError("VJP campaign is not the descriptor-stable canonical live path")
    payload = path.read_bytes()
    if len(payload) != CANONICAL_VJP_CAMPAIGN_BYTES or _sha256(payload) != CANONICAL_VJP_CAMPAIGN_SHA256:
        raise G70EffectMaterializationError("canonical VJP campaign bytes or SHA-256 changed")
    try:
        campaign = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G70EffectMaterializationError("VJP campaign receipt is not JSON") from exc
    if not isinstance(campaign, dict) or campaign.get("schema") != VJP_CAMPAIGN_SCHEMA:
        raise G70EffectMaterializationError("VJP campaign schema mismatch")
    pair_ids = campaign.get("final_completed_pair_ids")
    refused = campaign.get("refused_pair_ids")
    missing = campaign.get("still_missing_pair_ids")
    if (
        campaign.get("status") != "COMPLETE_N600"
        or campaign.get("final_completed_count") != PUBLIC_PAIR_COUNT
        or pair_ids != list(PUBLIC_PAIR_IDS)
        or refused != []
        or missing != []
    ):
        raise G70EffectMaterializationError("VJP campaign is not terminal complete n600")
    pair_id_array = np.asarray(pair_ids, dtype=np.int32)
    return VJPCampaignTransitionAuditV1(
        schema=VJP_AUDIT_SCHEMA,
        campaign_path=str(path),
        campaign_bytes=len(payload),
        campaign_sha256=_sha256(payload),
        campaign_schema=VJP_CAMPAIGN_SCHEMA,
        campaign_status="COMPLETE_N600",
        final_completed_count=PUBLIC_PAIR_COUNT,
        final_pair_ids_sha256=_array_sha256(pair_id_array),
        refused_pair_ids=(),
        still_missing_pair_ids=(),
        actual_frozen_scorer_vjp_campaign=True,
        g17_transition_bound=False,
        realized_transition_jvp_present=False,
        realized_transition_vjp_present=False,
        blockers=(SOURCE_VJP_RELABEL_BLOCKER, DERIVATIVE_JOIN_BLOCKER),
    )


def materialize_g70_finite_effect_transition(
    *,
    baseline: G17CandidateForwardObservationV1,
    candidate: G17CandidateForwardObservationV1,
    vjp_campaign_receipt_path: Path | None = None,
) -> G70FiniteEffectTransitionV1:
    """Derive exact endpoint effects; never synthesize or accept derivatives."""

    _require_exact_observation_join(baseline, candidate)
    baseline_archive = _validate_zip(
        baseline.archive_bytes,
        g17_member_bytes=baseline.member_bytes,
        label="baseline",
    )
    candidate_archive = _validate_zip(
        candidate.archive_bytes,
        g17_member_bytes=candidate.member_bytes,
        label="candidate",
    )
    pair_ids = _immutable(np.arange(PUBLIC_PAIR_COUNT), dtype=np.dtype(np.int32))
    baseline_seg = _immutable(np.asarray(baseline.receipt.per_pair_d_seg), dtype=np.dtype(np.float64))
    candidate_seg = _immutable(np.asarray(candidate.receipt.per_pair_d_seg), dtype=np.dtype(np.float64))
    baseline_pose = _immutable(np.asarray(baseline.receipt.per_pair_d_pose), dtype=np.dtype(np.float64))
    candidate_pose = _immutable(np.asarray(candidate.receipt.per_pair_d_pose), dtype=np.dtype(np.float64))
    for name, values, aggregate in (
        ("baseline d_seg", baseline_seg, baseline.receipt.aggregate_d_seg),
        ("candidate d_seg", candidate_seg, candidate.receipt.aggregate_d_seg),
        ("baseline d_pose", baseline_pose, baseline.receipt.aggregate_d_pose),
        ("candidate d_pose", candidate_pose, candidate.receipt.aggregate_d_pose),
    ):
        if not math.isclose(float(np.mean(values, dtype=np.float64)), aggregate, rel_tol=0.0, abs_tol=1e-15):
            raise G70EffectMaterializationError(f"{name} per-pair values do not close to the aggregate")

    seg_effect = SEG_SCORE_WEIGHT * (candidate_seg - baseline_seg) / PUBLIC_PAIR_COUNT
    pose_effect = _pose_additive_effects(
        baseline_pose,
        candidate_pose,
        baseline_aggregate=baseline.receipt.aggregate_d_pose,
        candidate_aggregate=candidate.receipt.aggregate_d_pose,
    )
    effects = _immutable(np.stack((seg_effect, pose_effect), axis=1), dtype=np.dtype(np.float64))
    seg_total = SEG_SCORE_WEIGHT * (candidate.receipt.aggregate_d_seg - baseline.receipt.aggregate_d_seg)
    pose_total = math.sqrt(POSE_SCORE_WEIGHT * candidate.receipt.aggregate_d_pose) - math.sqrt(
        POSE_SCORE_WEIGHT * baseline.receipt.aggregate_d_pose
    )
    archive_delta = candidate_archive.bytes - baseline_archive.bytes
    rate_total = RATE_SCORE_WEIGHT * archive_delta / CONTEST_SOURCE_BYTES
    archive_contract = build_serialized_archive_delta_contract(
        source_archive_bytes=baseline_archive.bytes,
        candidate_archive_bytes=candidate_archive.bytes,
    )
    transition_seed = _canonical_json(
        {
            "baseline_forward_receipt_sha256": baseline.receipt.receipt_sha256,
            "candidate_forward_receipt_sha256": candidate.receipt.receipt_sha256,
            "baseline_archive_sha256": baseline_archive.sha256,
            "candidate_archive_sha256": candidate_archive.sha256,
        }
    )
    derivative_audit = (
        None if vjp_campaign_receipt_path is None else audit_vjp_campaign_for_transition(vjp_campaign_receipt_path)
    )
    blockers = [DERIVATIVE_JOIN_BLOCKER, G67_REVIEW_BLOCKER]
    if derivative_audit is not None:
        blockers.insert(1, SOURCE_VJP_RELABEL_BLOCKER)
    return G70FiniteEffectTransitionV1(
        baseline=baseline,
        candidate=candidate,
        transition_id=_sha256(transition_seed),
        baseline_archive=baseline_archive,
        candidate_archive=candidate_archive,
        pair_ids=pair_ids,
        scorer_term_effect_vectors=effects,
        baseline_per_pair_d_seg=baseline_seg,
        candidate_per_pair_d_seg=candidate_seg,
        baseline_per_pair_d_pose=baseline_pose,
        candidate_per_pair_d_pose=candidate_pose,
        aggregate_effects={
            "seg_score_term_delta": seg_total,
            "pose_score_term_delta": pose_total,
            "archive_rate_score_term_delta": rate_total,
            "total_endpoint_score_delta": seg_total + pose_total + rate_total,
        },
        serialized_archive_delta_contract=archive_contract,
        derivative_audit=derivative_audit,
        blockers=tuple(blockers),
    )


def _npy_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, np.ascontiguousarray(value), allow_pickle=False)
    return output.getvalue()


def _deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w") as archive:
        for name in _BUNDLE_ARRAYS:
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _npy_bytes(arrays[name]), compresslevel=9)
    return output.getvalue()


def _atomic_new_file(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise G70EffectMaterializationError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_g70_finite_effect_bundle(
    transition: G70FiniteEffectTransitionV1,
    *,
    output_dir: Path,
) -> G70EffectBundleFilesV1:
    """Write a deterministic finite-effect NPZ and canonical blocker receipt."""

    if type(transition) is not G70FiniteEffectTransitionV1:
        raise G70EffectMaterializationError("writer requires a typed G70 transition")
    arrays = {
        "pair_ids": transition.pair_ids,
        "scorer_term_effect_vectors": transition.scorer_term_effect_vectors,
        "baseline_per_pair_d_seg": transition.baseline_per_pair_d_seg,
        "candidate_per_pair_d_seg": transition.candidate_per_pair_d_seg,
        "baseline_per_pair_d_pose": transition.baseline_per_pair_d_pose,
        "candidate_per_pair_d_pose": transition.candidate_per_pair_d_pose,
    }
    bundle_payload = _deterministic_npz(arrays)
    bundle_name = "g70_receiver_scorer_finite_effects.npz"
    receipt_name = "g70_receiver_scorer_effect_blocker_receipt.json"
    destination = output_dir.expanduser().resolve(strict=False)
    bundle_path = destination / bundle_name
    receipt_path = destination / receipt_name
    _atomic_new_file(bundle_path, bundle_payload)

    def archive_dict(value: ArchiveIdentityV1) -> dict[str, Any]:
        return {
            "bytes": value.bytes,
            "sha256": value.sha256,
            "member_table_sha256": value.member_table_sha256,
            "matching_g17_member_names": list(value.matching_g17_member_names),
        }

    receipt = {
        "schema": BLOCKER_RECEIPT_SCHEMA,
        "effect_schema": FINITE_EFFECT_SCHEMA,
        "bundle_schema": EFFECT_BUNDLE_SCHEMA,
        "transition_id": transition.transition_id,
        "pair_count": PUBLIC_PAIR_COUNT,
        "source_pair_ids_sha256": _array_sha256(transition.pair_ids),
        "target_forward_receipt_sha256": transition.baseline.target.receipt.receipt_sha256,
        "baseline_forward_receipt_sha256": transition.baseline.receipt.receipt_sha256,
        "candidate_forward_receipt_sha256": transition.candidate.receipt.receipt_sha256,
        "frozen_scorer_sha256": transition.baseline.receipt.frozen_scorer_sha256,
        "scorer_runtime_environment_sha256": (transition.baseline.receipt.scorer_runtime_environment_sha256),
        "baseline_archive": archive_dict(transition.baseline_archive),
        "candidate_archive": archive_dict(transition.candidate_archive),
        "baseline_receiver_receipt_sha256": transition.baseline.receipt.receiver_receipt_sha256,
        "candidate_receiver_receipt_sha256": transition.candidate.receipt.receiver_receipt_sha256,
        "baseline_decoded_output_sha256": transition.baseline.receipt.decoded_output_sha256,
        "candidate_decoded_output_sha256": transition.candidate.receipt.decoded_output_sha256,
        "serialized_archive_delta_contract": transition.serialized_archive_delta_contract,
        "aggregate_effects": transition.aggregate_effects,
        "effect_bundle": {
            "name": bundle_name,
            "bytes": len(bundle_payload),
            "sha256": _sha256(bundle_payload),
            "arrays": {
                name: {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": _array_sha256(value),
                }
                for name, value in arrays.items()
            },
            "effect_axes": ["seg_score_term_delta", "pose_score_term_delta"],
            "pose_attribution": "Aumann-Shapley straight-line aggregate endpoint attribution",
        },
        "derivative_audit": (None if transition.derivative_audit is None else transition.derivative_audit.as_dict()),
        "actionable_costate_input": False,
        "actionable_consumers": [],
        "blockers": list(transition.blockers),
        "authority": dict(_FALSE_AUTHORITY),
    }
    receipt_payload = _canonical_json(receipt)
    try:
        _atomic_new_file(receipt_path, receipt_payload)
    except BaseException:
        bundle_path.unlink(missing_ok=True)
        raise
    return G70EffectBundleFilesV1(
        bundle_path=bundle_path,
        bundle_bytes=len(bundle_payload),
        bundle_sha256=_sha256(bundle_payload),
        receipt_path=receipt_path,
        receipt_bytes=len(receipt_payload),
        receipt_sha256=_sha256(receipt_payload),
    )


def require_g70_actionable_costate_input(
    transition: G70FiniteEffectTransitionV1,
) -> NoReturn:
    """Production integration point: always refuse until derivative closure."""

    if type(transition) is not G70FiniteEffectTransitionV1:
        raise G70EffectMaterializationError("actionable admission requires a typed G70 transition")
    transition.refuse_actionable_admission()


__all__ = [
    "BLOCKER_RECEIPT_SCHEMA",
    "DERIVATIVE_JOIN_BLOCKER",
    "EFFECT_BUNDLE_SCHEMA",
    "FINITE_EFFECT_SCHEMA",
    "G67_REVIEW_BLOCKER",
    "SOURCE_VJP_RELABEL_BLOCKER",
    "G70EffectBundleFilesV1",
    "G70EffectMaterializationError",
    "G70FiniteEffectTransitionV1",
    "VJPCampaignTransitionAuditV1",
    "audit_vjp_campaign_for_transition",
    "materialize_g70_finite_effect_transition",
    "require_g70_actionable_costate_input",
    "write_g70_finite_effect_bundle",
]
