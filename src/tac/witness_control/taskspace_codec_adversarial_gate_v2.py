# SPDX-License-Identifier: MIT
"""Fail-closed recurrent campaign gate for task-space codec candidates.

Unlike the G57 v1 retrospective linter, this module advances one immutable
campaign through an ordered receipt chain.  Every live boundary reopens the
current dynamic frontier and all objects used at that boundary.  A receipt
that is refused, retrospective, learning-only, or bound to another object can
never satisfy a launcher prerequisite.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

from tac.contest_score import compute_contest_score
from tac.witness_dsl.dynamic_frontier_target import (
    DynamicFrontierTargetError,
    DynamicFrontierTargetSnapshot,
    load_dynamic_frontier_target,
    verify_dynamic_frontier_target_snapshot,
)
from tac.witness_dsl.taskspace_public_auth_eval_closure import (
    OfficialEvaluationRunReceiptV1,
    PublicAuthClosureError,
    PublicDecodeEqualityReceiptV1,
)
from tac.witness_dsl.taskspace_selected_preimage_operand_adapter_v1 import (
    PRODUCTION_PRE_ENCODE_EVIDENCE_SCHEMA,
    SelectedPreimageOperandAdapterError,
    reopen_program_residual_production_pre_encode_evidence,
)
from tac.witness_dsl.taskspace_selected_preimage_operand_adapter_v1 import (
    SCHEMA as G58_IDENTITY_SCHEMA,
)

RECEIPT_SCHEMA: Final = "tac.taskspace_codec_adversarial_gate_receipt.v2"
EXACT_EVAL_SCHEMA: Final = "tac.taskspace_codec_exact_eval_receipt.v2"
EXACT_REPORT_SCHEMA: Final = "tac.taskspace_codec_exact_score_report.v2"
INTEGRATION_SCHEMA: Final = "tac.taskspace_codec_integration_receipt.v2"
BLOCKER_SCHEMA: Final = "tac.taskspace_codec_integration_blocker.v2"
PUBLIC_AUTH_SCHEMA: Final = "taskspace_layered_public_auth_receipt.v1"
STRICT_PUBLIC_BUNDLE_SCHEMA: Final = "tac.taskspace_codec_strict_public_auth_bundle.v2"

LIVE: Final = "LIVE"
RETROSPECTIVE_ONLY: Final = "RETROSPECTIVE_ONLY"

CAMPAIGN_SEAL: Final = "CAMPAIGN_SEAL"
PRE_ENCODE: Final = "PRE_ENCODE"
ENCODE: Final = "ENCODE"
POST_EVAL: Final = "POST_EVAL"
PRE_PUBLIC_CLOSURE: Final = "PRE_PUBLIC_CLOSURE"
PRE_PROMOTION: Final = "PRE_PROMOTION"
STAGES: Final = (
    CAMPAIGN_SEAL,
    PRE_ENCODE,
    ENCODE,
    POST_EVAL,
    PRE_PUBLIC_CLOSURE,
    PRE_PROMOTION,
)

PROGRAM_RESIDUAL_LAYERED: Final = "PROGRAM_RESIDUAL_LAYERED"
DIRECT_TASK_LAYERED_CONTROL: Final = "DIRECT_TASK_LAYERED_CONTROL"
REPRESENTATIONS: Final = frozenset({PROGRAM_RESIDUAL_LAYERED, DIRECT_TASK_LAYERED_CONTROL})

INTEGRATION_HOOKS: Final = (
    "sensitivity_map",
    "pareto_allocator",
    "bit_allocator",
    "autopilot",
    "continual_posterior",
    "probe_ledger",
)
CONTEST_AXES: Final = frozenset({"contest-CPU", "contest-CUDA"})
_MAX_ARTIFACT_BYTES: Final = 64 * 1024 * 1024
_MAX_RAW_BYTES: Final = 8 * 1024 * 1024 * 1024


class AdversarialGateError(RuntimeError):
    """A live lifecycle, custody, or object-identity invariant failed."""


def canonical_json(value: Any) -> bytes:
    """Return the one accepted receipt serialization."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _lexical_absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _stable_read_with_identity(
    path: Path | str,
    *,
    label: str,
    maximum_bytes: int = _MAX_ARTIFACT_BYTES,
) -> tuple[dict[str, Any], bytes]:
    """Return descriptor-stable identity and the bytes read from that descriptor."""

    target = _lexical_absolute(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise AdversarialGateError(f"{label} cannot be opened as a no-follow regular file: {target}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AdversarialGateError(f"{label} is not a regular file")
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise AdversarialGateError(f"{label} has invalid byte length")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 8 << 20))
            if not chunk:
                raise AdversarialGateError(f"{label} was truncated while reopening")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise AdversarialGateError(f"{label} grew while reopening")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    named = target.stat(follow_symlinks=False)
    identity_named = (
        named.st_dev,
        named.st_ino,
        named.st_size,
        named.st_mtime_ns,
        named.st_ctime_ns,
    )
    if not stat.S_ISREG(named.st_mode) or identity_before != identity_after or identity_after != identity_named:
        raise AdversarialGateError(f"{label} identity changed while reopening")
    payload = b"".join(chunks)
    return (
        {
            "path": str(target),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        },
        payload,
    )


def _stable_read(path: Path | str, *, label: str, maximum_bytes: int = _MAX_ARTIFACT_BYTES) -> bytes:
    return _stable_read_with_identity(
        path,
        label=label,
        maximum_bytes=maximum_bytes,
    )[1]


def _stable_identity_only(
    path: Path | str,
    *,
    label: str,
    maximum_bytes: int,
) -> dict[str, Any]:
    """Stream-hash a potentially large object without retaining its bytes."""

    target = _lexical_absolute(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise AdversarialGateError(f"{label} cannot be opened as a no-follow regular file: {target}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AdversarialGateError(f"{label} is not a regular file")
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise AdversarialGateError(f"{label} has invalid byte length")
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 8 << 20))
            if not chunk:
                raise AdversarialGateError(f"{label} was truncated while hashing")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise AdversarialGateError(f"{label} grew while hashing")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    named = target.stat(follow_symlinks=False)
    identity_named = (
        named.st_dev,
        named.st_ino,
        named.st_size,
        named.st_mtime_ns,
        named.st_ctime_ns,
    )
    if not stat.S_ISREG(named.st_mode) or identity_before != identity_after or identity_after != identity_named:
        raise AdversarialGateError(f"{label} identity changed while hashing")
    return {
        "path": str(target),
        "bytes": before.st_size,
        "sha256": digest.hexdigest(),
    }


def artifact_identity(
    path: Path | str,
    *,
    label: str,
    maximum_bytes: int = _MAX_ARTIFACT_BYTES,
) -> dict[str, Any]:
    return _stable_identity_only(
        path,
        label=label,
        maximum_bytes=maximum_bytes,
    )


def _verify_artifact(
    row: Any,
    *,
    label: str,
    maximum_bytes: int = _MAX_ARTIFACT_BYTES,
    capture_payload: bool = True,
) -> tuple[dict[str, Any], bytes]:
    if not isinstance(row, Mapping) or frozenset(row) != {"path", "bytes", "sha256"}:
        raise AdversarialGateError(f"{label} must be an exact path/bytes/SHA identity")
    if capture_payload:
        identity, payload = _stable_read_with_identity(
            row["path"],
            label=label,
            maximum_bytes=maximum_bytes,
        )
    else:
        identity = _stable_identity_only(
            row["path"],
            label=label,
            maximum_bytes=maximum_bytes,
        )
        payload = b""
    if dict(row) != identity:
        raise AdversarialGateError(f"{label} exact object identity does not match")
    return identity, payload


def _load_json_artifact_with_payload(
    path: Path | str,
    *,
    label: str,
    maximum_bytes: int = _MAX_ARTIFACT_BYTES,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    identity, payload = _stable_read_with_identity(
        path,
        label=label,
        maximum_bytes=maximum_bytes,
    )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdversarialGateError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise AdversarialGateError(f"{label} must contain a JSON object")
    return identity, value, payload


def _load_json_artifact(
    path: Path | str,
    *,
    label: str,
    maximum_bytes: int = _MAX_ARTIFACT_BYTES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity, value, _ = _load_json_artifact_with_payload(
        path,
        label=label,
        maximum_bytes=maximum_bytes,
    )
    return identity, value


def _body_sha(receipt: Mapping[str, Any]) -> str:
    body = dict(receipt)
    body.pop("body_sha256", None)
    return sha256_bytes(canonical_json(body))


def _atomic_write_once(path: Path | str, payload: bytes) -> None:
    target = _lexical_absolute(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        current = _stable_read(target, label="preserved gate receipt")
        if current != payload:
            raise AdversarialGateError(f"write-once gate receipt drifted: {target}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            current = _stable_read(target, label="raced gate receipt")
            if current != payload:
                raise AdversarialGateError(f"gate receipt publication raced with different bytes: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _write_receipt(path: Path | str, body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    value["body_sha256"] = _body_sha(value)
    payload = canonical_json(value)
    _atomic_write_once(path, payload)
    return value


def reopen_receipt(path: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    identity, value, payload = _load_json_artifact_with_payload(
        path,
        label="adversarial gate receipt",
    )
    if value.get("schema") != RECEIPT_SCHEMA or value.get("stage") not in STAGES:
        raise AdversarialGateError("adversarial gate receipt schema or stage drift")
    if value.get("body_sha256") != _body_sha(value):
        raise AdversarialGateError("adversarial gate receipt body SHA mismatch")
    if canonical_json(value) != payload:
        raise AdversarialGateError("adversarial gate receipt is not canonically serialized")
    return identity, value


def _snapshot_from_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_repo_root: Path | str | None = None,
) -> DynamicFrontierTargetSnapshot:
    row = receipt.get("frontier")
    if not isinstance(row, Mapping):
        raise AdversarialGateError("gate receipt lacks frontier snapshot")
    try:
        snapshot = DynamicFrontierTargetSnapshot(**dict(row))
        verify_dynamic_frontier_target_snapshot(snapshot)
    except (TypeError, DynamicFrontierTargetError) as exc:
        raise AdversarialGateError("bound dynamic frontier is stale or changed") from exc
    bound_root = _lexical_absolute(receipt["repo_root"])
    if expected_repo_root is not None and bound_root != _lexical_absolute(expected_repo_root):
        raise AdversarialGateError("campaign was sealed for a different repository root")
    current = load_dynamic_frontier_target(repo_root=bound_root)
    if current != snapshot:
        raise AdversarialGateError("current dynamic frontier differs from campaign seal")
    return snapshot


def _receipt_common(
    *,
    stage: str,
    mode: str,
    campaign_id: str,
    representation: str,
    repo_root: Path | str,
    frontier: DynamicFrontierTargetSnapshot,
    predecessor: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "stage": stage,
        "mode": mode,
        "campaign_id": campaign_id,
        "requested_representation": representation,
        "repo_root": str(_lexical_absolute(repo_root)),
        "frontier": asdict(frontier),
        "predecessor": None if predecessor is None else dict(predecessor),
        "status": "REFUSE",
        "candidate_admission": False,
        "learning_admission": False,
        "next_stage": None,
        "archive": None,
        "decoded_raw": None,
        "exact_eval": None,
        "producer_config": None,
        "evidence": {},
        "refusals": [],
    }


def seal_campaign(
    *,
    campaign_id: str,
    requested_representation: str,
    repo_root: Path | str,
    expected_repo_root: Path | str,
    output_path: Path | str,
    asserted_target_score: float | None = None,
    asserted_pointer_sha256: str | None = None,
    mode: str = LIVE,
) -> dict[str, Any]:
    """Seal a campaign to the current source-bound competitive frontier."""

    if not campaign_id or requested_representation not in REPRESENTATIONS:
        raise AdversarialGateError("campaign id or requested representation is invalid")
    try:
        canonical_root = _lexical_absolute(expected_repo_root)
        frontier = load_dynamic_frontier_target(repo_root=canonical_root)
    except DynamicFrontierTargetError as exc:
        raise AdversarialGateError("cannot seal against the live dynamic frontier") from exc
    body = _receipt_common(
        stage=CAMPAIGN_SEAL,
        mode=mode,
        campaign_id=campaign_id,
        representation=requested_representation,
        repo_root=canonical_root,
        frontier=frontier,
        predecessor=None,
    )
    refusals: list[str] = []
    if _lexical_absolute(repo_root) != canonical_root:
        refusals.append("CALLER_REPOSITORY_ROOT_DIFFERS_FROM_LAUNCHER_ROOT")
    if mode != LIVE:
        refusals.append("RETROSPECTIVE_RECEIPT_CANNOT_ADMIT_LIVE_CAMPAIGN")
    if asserted_target_score is not None and float(asserted_target_score) != frontier.target_score:
        refusals.append("CALLER_TARGET_DIFFERS_FROM_DYNAMIC_FRONTIER")
    if asserted_pointer_sha256 is not None and asserted_pointer_sha256 != frontier.pointer_sha256:
        refusals.append("CALLER_POINTER_SHA_DIFFERS_FROM_DYNAMIC_FRONTIER")
    body["refusals"] = refusals
    if not refusals:
        body.update(status="ADMIT", candidate_admission=True, next_stage=PRE_ENCODE)
    return _write_receipt(output_path, body)


def _advance(
    predecessor_path: Path | str,
    *,
    expected_stage: str,
    next_stage: str,
    output_path: Path | str,
    asserted_representation: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], DynamicFrontierTargetSnapshot]:
    predecessor_identity, predecessor = reopen_receipt(predecessor_path)
    snapshot = _snapshot_from_receipt(predecessor)
    link = {
        **predecessor_identity,
        "body_sha256": predecessor["body_sha256"],
        "stage": predecessor["stage"],
    }
    body = _receipt_common(
        stage=next_stage,
        mode=predecessor["mode"],
        campaign_id=predecessor["campaign_id"],
        representation=predecessor["requested_representation"],
        repo_root=predecessor["repo_root"],
        frontier=snapshot,
        predecessor=link,
    )
    refusals: list[str] = []
    if predecessor["stage"] != expected_stage:
        refusals.append("PREDECESSOR_STAGE_OUT_OF_ORDER")
    if predecessor.get("status") != "ADMIT" or predecessor.get("candidate_admission") is not True:
        refusals.append("PREDECESSOR_NOT_LIVE_CANDIDATE_ADMITTED")
    if predecessor.get("mode") != LIVE:
        refusals.append("RETROSPECTIVE_PREDECESSOR_CANNOT_ADVANCE")
    if asserted_representation is not None and asserted_representation != predecessor["requested_representation"]:
        refusals.append("REQUESTED_REPRESENTATION_SWITCH")
    body["refusals"] = refusals
    return body, predecessor, predecessor_identity, snapshot


def _validate_g58_pre_encode(
    identity_path: Path | str,
    terminal_stage_chain_path: Path | str | None,
    outer_proof_path: Path | str | None,
    *,
    expected_campaign_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Delegate all production interpretation to G58's strict verifier."""

    identity_record, identity = _load_json_artifact(identity_path, label="G58 adapter identity receipt")
    evidence: dict[str, Any] = {
        "adapter_identity_receipt": identity_record,
        "adapter_schema_reopened": identity.get("schema") == G58_IDENTITY_SCHEMA,
    }
    if terminal_stage_chain_path is None or outer_proof_path is None:
        return evidence, ["G58_PRODUCTION_TERMINAL_CHAIN_AND_OUTER_PROOF_OWED"]
    try:
        production = dict(
            reopen_program_residual_production_pre_encode_evidence(
                identity_path=identity_path,
                terminal_stage_chain_path=terminal_stage_chain_path,
                outer_proof_path=outer_proof_path,
            )
        )
    except SelectedPreimageOperandAdapterError as exc:
        return evidence, [f"G58_STRICT_PRODUCTION_VERIFIER_REFUSED:{exc}"]
    if production.get("schema") != PRODUCTION_PRE_ENCODE_EVIDENCE_SCHEMA or production.get("status") != "ADMIT":
        return evidence, ["G58_STRICT_PRODUCTION_EVIDENCE_SCHEMA_DRIFT"]
    if production.get("campaign_receipt") != dict(expected_campaign_receipt):
        return evidence, ["G58_TERMINAL_CHAIN_BOUND_TO_DIFFERENT_CAMPAIGN"]
    if type(production.get("analytic_only")) is not bool:
        return evidence, ["G58_PRODUCTION_FACTOR_ROLE_CUSTODY_AMBIGUOUS"]
    evidence["strict_production_evidence"] = production
    return evidence, []


def _validate_program_producer_config(
    _config_identity: Mapping[str, Any] | None,
    _g58_evidence: Mapping[str, Any],
) -> list[str]:
    """Refuse until a real runner/config binds the exact G58 artifact triple.

    No production call site currently compiles an n600 G49 program/factor set.
    Defining a permissive placeholder schema here would recreate the G57
    substitution bug, so this is an explicit structural blocker.
    """

    return ["PROGRAM_PRODUCER_CONFIG_SCHEMA_OWED"]


def admit_pre_encode(
    *,
    campaign_seal_path: Path | str,
    output_path: Path | str,
    producer_config_path: Path | str,
    g58_identity_receipt_path: Path | str | None = None,
    g58_terminal_stage_chain_path: Path | str | None = None,
    g58_outer_proof_path: Path | str | None = None,
    asserted_representation: str | None = None,
) -> dict[str, Any]:
    """Admit the exact requested producer operand before encoding."""

    body, predecessor, predecessor_identity, _ = _advance(
        campaign_seal_path,
        expected_stage=CAMPAIGN_SEAL,
        next_stage=PRE_ENCODE,
        output_path=output_path,
        asserted_representation=asserted_representation,
    )
    refusals = list(body["refusals"])
    try:
        body["producer_config"] = artifact_identity(
            producer_config_path,
            label="campaign producer config",
        )
    except AdversarialGateError as exc:
        refusals.append(f"PRODUCER_CONFIG_NOT_REOPENABLE:{exc}")
    representation = predecessor["requested_representation"]
    if representation == PROGRAM_RESIDUAL_LAYERED:
        if g58_identity_receipt_path is None:
            refusals.append("G58_ADAPTER_IDENTITY_RECEIPT_REQUIRED")
        else:
            try:
                evidence, g58_refusals = _validate_g58_pre_encode(
                    g58_identity_receipt_path,
                    g58_terminal_stage_chain_path,
                    g58_outer_proof_path,
                    expected_campaign_receipt=predecessor_identity,
                )
                body["evidence"] = evidence
                refusals.extend(g58_refusals)
                refusals.extend(
                    _validate_program_producer_config(
                        body.get("producer_config"),
                        evidence,
                    )
                )
            except AdversarialGateError as exc:
                refusals.append(f"G58_EVIDENCE_NOT_REOPENABLE:{exc}")
    else:
        # Direct-control remains legal only as retrospective G57 evidence until
        # a production direct producer publishes an equally strict adapter.
        refusals.append("DIRECT_CONTROL_PRODUCTION_ADAPTER_NOT_SUPPLIED")
    body["refusals"] = refusals
    if not refusals:
        body.update(status="ADMIT", candidate_admission=True, next_stage=ENCODE)
    return _write_receipt(output_path, body)


def admit_encode(
    *,
    pre_encode_receipt_path: Path | str,
    archive_path: Path | str,
    decoded_raw_path: Path | str,
    output_path: Path | str,
    asserted_representation: str | None = None,
) -> dict[str, Any]:
    """Bind the immutable counted archive and decoded raw object."""

    body, predecessor, _, _ = _advance(
        pre_encode_receipt_path,
        expected_stage=PRE_ENCODE,
        next_stage=ENCODE,
        output_path=output_path,
        asserted_representation=asserted_representation,
    )
    body["producer_config"] = predecessor.get("producer_config")
    refusals = list(body["refusals"])
    try:
        archive = artifact_identity(archive_path, label="counted archive")
        raw = artifact_identity(decoded_raw_path, label="decoded raw", maximum_bytes=_MAX_RAW_BYTES)
    except AdversarialGateError as exc:
        refusals.append(f"ENCODE_OBJECT_NOT_REOPENABLE:{exc}")
        archive = None
        raw = None
    body.update(archive=archive, decoded_raw=raw, refusals=refusals)
    if not refusals:
        body.update(status="ADMIT", candidate_admission=True, next_stage=POST_EVAL)
    return _write_receipt(output_path, body)


def _validate_eval(
    *,
    eval_receipt_path: Path | str,
    eval_report_path: Path | str,
    predecessor: Mapping[str, Any],
    snapshot: DynamicFrontierTargetSnapshot,
) -> tuple[dict[str, Any], bool]:
    receipt_record, payload = _stable_read_with_identity(
        eval_receipt_path,
        label="strict official evaluation receipt",
    )
    try:
        receipt = OfficialEvaluationRunReceiptV1.from_receipt_bytes(payload)
    except PublicAuthClosureError as exc:
        raise AdversarialGateError("exact evaluation is not a strict OfficialEvaluationRunReceiptV1") from exc
    archive, _ = _verify_artifact(predecessor["archive"], label="exact-eval archive")
    raw, _ = _verify_artifact(
        predecessor["decoded_raw"],
        label="exact-eval decoded raw",
        maximum_bytes=_MAX_RAW_BYTES,
        capture_payload=False,
    )
    if (
        receipt.archive_sha256 != archive["sha256"]
        or receipt.archive_nbytes != archive["bytes"]
        or receipt.raw_sha256 != raw["sha256"]
        or receipt.raw_nbytes != raw["bytes"]
    ):
        raise AdversarialGateError("strict official evaluation switched archive or decoded raw")
    report_record = artifact_identity(eval_report_path, label="official evaluation report")
    if report_record["sha256"] != receipt.report_sha256:
        raise AdversarialGateError("official report bytes differ from strict run receipt")
    score = compute_contest_score(
        receipt.avg_segnet_dist,
        receipt.avg_posenet_dist,
        archive["bytes"],
    )
    if not math.isclose(
        score,
        receipt.report_component_recomputed_score,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AdversarialGateError("strict run differs from canonical tac.contest_score DAG")
    verify_dynamic_frontier_target_snapshot(snapshot)
    competitive = score < snapshot.target_score
    return {
        "receipt": receipt_record,
        "report": report_record,
        "axis": f"contest-{receipt.execution_axis.value}",
        "d_seg": receipt.avg_segnet_dist,
        "d_pose": receipt.avg_posenet_dist,
        "score": score,
        "inside_strict_sublevel": competitive,
        "signed_target_slack": snapshot.target_score - score,
        "strict_run_identity_sha256": receipt.identity_sha256,
    }, competitive


def _validate_learning_custody(
    *,
    campaign_id: str,
    exact_eval: Mapping[str, Any],
    integration_receipt_paths: Sequence[Path | str],
    blocker_path: Path | str | None,
) -> tuple[dict[str, Any], bool]:
    records: dict[str, Any] = {}
    if integration_receipt_paths:
        for path in integration_receipt_paths:
            record, value = _load_json_artifact(path, label="integration receipt")
            if value.get("schema") != INTEGRATION_SCHEMA or value.get("status") != "INTEGRATED":
                raise AdversarialGateError("integration receipt schema or status drift")
            hook = value.get("hook")
            if hook not in INTEGRATION_HOOKS or hook in records:
                raise AdversarialGateError("integration hook is unknown or duplicated")
            if value.get("campaign_id") != campaign_id:
                raise AdversarialGateError("integration campaign id mismatch")
            if value.get("exact_eval_receipt_sha256") != exact_eval["receipt"]["sha256"]:
                raise AdversarialGateError("integration is not bound to the exact-eval receipt")
            records[hook] = record
        if set(records) != set(INTEGRATION_HOOKS):
            raise AdversarialGateError("every canonical post-eval integration hook is required")
        return {"integration_receipts": records, "blocker": None}, True
    if blocker_path is None:
        raise AdversarialGateError("post-eval custody needs all integrations or a structured blocker")
    record, blocker = _load_json_artifact(blocker_path, label="structured integration blocker")
    if blocker.get("schema") != BLOCKER_SCHEMA or blocker.get("campaign_id") != campaign_id:
        raise AdversarialGateError("structured blocker schema or campaign mismatch")
    missing = blocker.get("missing_hooks")
    not_killed = blocker.get("not_killed")
    scope = blocker.get("verdict_scope")
    action = blocker.get("next_executable_action")
    if (
        not isinstance(blocker.get("owner"), str)
        or not blocker["owner"]
        or not isinstance(missing, list)
        or not missing
        or any(item not in INTEGRATION_HOOKS for item in missing)
        or not isinstance(not_killed, list)
        or not not_killed
        or any(not isinstance(item, str) or not item for item in not_killed)
        or not isinstance(scope, Mapping)
        or scope.get("level") not in {"FORMULATION", "IMPLEMENTATION", "OPERATING_POINT"}
        or not isinstance(scope.get("name"), str)
        or not scope["name"]
        or not isinstance(action, list)
        or not action
        or any(not isinstance(item, str) or not item for item in action)
    ):
        raise AdversarialGateError("structured blocker lacks narrow scope, owner, hooks, not-killed, or action")
    if blocker.get("exact_eval_receipt_sha256") != exact_eval["receipt"]["sha256"]:
        raise AdversarialGateError("structured blocker is not bound to exact-eval custody")
    return {"integration_receipts": {}, "blocker": record}, False


def admit_post_eval(
    *,
    encode_receipt_path: Path | str,
    eval_receipt_path: Path | str,
    eval_report_path: Path | str,
    output_path: Path | str,
    integration_receipt_paths: Sequence[Path | str] = (),
    blocker_path: Path | str | None = None,
    asserted_representation: str | None = None,
) -> dict[str, Any]:
    """Admit exact score and durable post-eval learning custody."""

    body, predecessor, _, snapshot = _advance(
        encode_receipt_path,
        expected_stage=ENCODE,
        next_stage=POST_EVAL,
        output_path=output_path,
        asserted_representation=asserted_representation,
    )
    body["archive"] = predecessor.get("archive")
    body["decoded_raw"] = predecessor.get("decoded_raw")
    body["producer_config"] = predecessor.get("producer_config")
    refusals = list(body["refusals"])
    exact_eval: dict[str, Any] | None = None
    all_integrated = False
    try:
        exact_eval, competitive = _validate_eval(
            eval_receipt_path=eval_receipt_path,
            eval_report_path=eval_report_path,
            predecessor=predecessor,
            snapshot=snapshot,
        )
        custody, all_integrated = _validate_learning_custody(
            campaign_id=predecessor["campaign_id"],
            exact_eval=exact_eval,
            integration_receipt_paths=integration_receipt_paths,
            blocker_path=blocker_path,
        )
        body["evidence"] = custody
        if not competitive:
            refusals.append("EXACT_ROW_NOT_STRICTLY_COMPETITIVE")
        if not all_integrated:
            refusals.append("STRUCTURED_BLOCKER_ADMITS_LEARNING_ONLY")
    except AdversarialGateError as exc:
        refusals.append(str(exc))
    body.update(exact_eval=exact_eval, refusals=refusals)
    body["learning_admission"] = exact_eval is not None and (
        all_integrated or "STRUCTURED_BLOCKER_ADMITS_LEARNING_ONLY" in refusals
    )
    if not refusals:
        body.update(status="ADMIT", candidate_admission=True, next_stage=PRE_PUBLIC_CLOSURE)
    elif (
        exact_eval is not None
        and body["learning_admission"]
        and set(refusals).issubset({"EXACT_ROW_NOT_STRICTLY_COMPETITIVE", "STRUCTURED_BLOCKER_ADMITS_LEARNING_ONLY"})
    ):
        body["status"] = "LEARNING_ONLY"
    return _write_receipt(output_path, body)


def admit_pre_public_closure(
    *,
    post_eval_receipt_path: Path | str,
    output_path: Path | str,
    asserted_archive_path: Path | str | None = None,
) -> dict[str, Any]:
    """Recheck same-object strict competitiveness immediately before closure."""

    body, predecessor, _, snapshot = _advance(
        post_eval_receipt_path,
        expected_stage=POST_EVAL,
        next_stage=PRE_PUBLIC_CLOSURE,
        output_path=output_path,
    )
    body["archive"] = predecessor.get("archive")
    body["decoded_raw"] = predecessor.get("decoded_raw")
    body["exact_eval"] = predecessor.get("exact_eval")
    body["producer_config"] = predecessor.get("producer_config")
    refusals = list(body["refusals"])
    try:
        archive, _ = _verify_artifact(predecessor["archive"], label="pre-public archive")
        _verify_artifact(
            predecessor["decoded_raw"],
            label="pre-public decoded raw",
            maximum_bytes=_MAX_RAW_BYTES,
            capture_payload=False,
        )
        if asserted_archive_path is not None:
            asserted = artifact_identity(asserted_archive_path, label="asserted pre-public archive")
            if asserted != archive:
                raise AdversarialGateError("public closure attempted an archive switch")
        exact_eval = predecessor["exact_eval"]
        score = compute_contest_score(
            exact_eval["d_seg"],
            exact_eval["d_pose"],
            archive["bytes"],
        )
        verify_dynamic_frontier_target_snapshot(snapshot)
        if not score < snapshot.target_score:
            refusals.append("PRE_PUBLIC_ROW_IS_NOT_STRICTLY_COMPETITIVE")
    except (AdversarialGateError, KeyError, TypeError) as exc:
        refusals.append(str(exc))
    body["refusals"] = refusals
    if not refusals:
        body.update(status="ADMIT", candidate_admission=True, next_stage=PRE_PROMOTION)
    return _write_receipt(output_path, body)


def _verify_public_auth(
    *,
    path: Path | str,
    predecessor: Mapping[str, Any],
    snapshot: DynamicFrontierTargetSnapshot,
) -> dict[str, Any]:
    record, auth = _load_json_artifact(path, label="strict public authority bundle")
    exact = predecessor["exact_eval"]
    archive, _ = _verify_artifact(predecessor["archive"], label="pre-promotion archive")
    raw, _ = _verify_artifact(
        predecessor["decoded_raw"],
        label="pre-promotion decoded raw",
        maximum_bytes=_MAX_RAW_BYTES,
        capture_payload=False,
    )
    if (
        auth.get("schema") != STRICT_PUBLIC_BUNDLE_SCHEMA
        or auth.get("campaign_id") != predecessor["campaign_id"]
        or auth.get("requested_representation") != predecessor["requested_representation"]
    ):
        raise AdversarialGateError("strict public bundle schema, campaign, or representation drifted")
    parsed_runs: list[OfficialEvaluationRunReceiptV1] = []
    run_records: list[dict[str, Any]] = []
    try:
        for field in ("run_a", "run_b"):
            run_record, run_payload = _verify_artifact(auth.get(field), label=f"strict public {field}")
            parsed_runs.append(OfficialEvaluationRunReceiptV1.from_receipt_bytes(run_payload))
            run_records.append(run_record)
        equality_record, equality_payload = _verify_artifact(
            auth.get("equality"),
            label="strict public equality",
        )
        equality = PublicDecodeEqualityReceiptV1.from_receipt_bytes(equality_payload)
        derived_equality = PublicDecodeEqualityReceiptV1.from_runs(*parsed_runs)
    except PublicAuthClosureError as exc:
        raise AdversarialGateError("strict public authority receipts failed typed parse-back") from exc
    if equality.to_receipt_bytes() != derived_equality.to_receipt_bytes():
        raise AdversarialGateError("strict public equality was not derived from the reopened A/B runs")
    if (
        equality.archive_sha256 != archive["sha256"]
        or equality.archive_nbytes != archive["bytes"]
        or equality.raw_sha256 != raw["sha256"]
        or equality.raw_nbytes != raw["bytes"]
    ):
        raise AdversarialGateError("strict public closure switched archive or raw object")
    score = compute_contest_score(
        equality.avg_segnet_dist,
        equality.avg_posenet_dist,
        equality.archive_nbytes,
    )
    for observed, expected, field in (
        (equality.avg_segnet_dist, exact["d_seg"], "d_seg"),
        (equality.avg_posenet_dist, exact["d_pose"], "d_pose"),
        (equality.report_component_recomputed_score, exact["score"], "score"),
    ):
        if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12):
            raise AdversarialGateError(f"strict public receiver exact row switched: {field}")
    if not math.isclose(score, exact["score"], rel_tol=0.0, abs_tol=1e-12):
        raise AdversarialGateError("pre-promotion row differs from canonical contest score")
    verify_dynamic_frontier_target_snapshot(snapshot)
    if not score < snapshot.target_score:
        raise AdversarialGateError("public receiver row is not strictly competitive")
    legacy_record, legacy_payload = _verify_artifact(
        auth.get("legacy_g55_auth"),
        label="G55 promotion compatibility receipt",
    )
    try:
        legacy = json.loads(legacy_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdversarialGateError("G55 compatibility receipt is not JSON") from exc
    if (
        not isinstance(legacy, Mapping)
        or legacy.get("schema") != PUBLIC_AUTH_SCHEMA
        or legacy.get("archive_sha256") != archive["sha256"]
        or legacy.get("archive_bytes") != archive["bytes"]
        or legacy.get("raw_sha256") != raw["sha256"]
        or legacy.get("raw_bytes") != raw["bytes"]
        or legacy.get("pair_count") != 600
        or legacy.get("frame_count") != 1200
        or legacy.get("authority_axis") != f"contest-{equality.execution_axis.value}"
        or not math.isclose(float(legacy.get("d_seg", math.nan)), exact["d_seg"], rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(float(legacy.get("d_pose", math.nan)), exact["d_pose"], rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(float(legacy.get("score", math.nan)), exact["score"], rel_tol=0.0, abs_tol=1e-12)
    ):
        raise AdversarialGateError("G55 compatibility receipt is not derived from strict public authority")
    return {
        "bundle": record,
        "run_a": run_records[0],
        "run_b": run_records[1],
        "equality": equality_record,
        "legacy_g55_auth": legacy_record,
    }


def admit_pre_promotion(
    *,
    pre_public_receipt_path: Path | str,
    public_auth_receipt_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """Admit promotion only for same-object competitive contest closure."""

    body, predecessor, _, snapshot = _advance(
        pre_public_receipt_path,
        expected_stage=PRE_PUBLIC_CLOSURE,
        next_stage=PRE_PROMOTION,
        output_path=output_path,
    )
    body["archive"] = predecessor.get("archive")
    body["decoded_raw"] = predecessor.get("decoded_raw")
    body["exact_eval"] = predecessor.get("exact_eval")
    body["producer_config"] = predecessor.get("producer_config")
    refusals = list(body["refusals"])
    try:
        body["evidence"] = {
            "public_auth_receipt": _verify_public_auth(
                path=public_auth_receipt_path,
                predecessor=predecessor,
                snapshot=snapshot,
            )
        }
    except (AdversarialGateError, TypeError, ValueError) as exc:
        refusals.append(str(exc))
    # The strict public-auth module deliberately reopens research-only
    # observations and explicitly cannot mint contest authority.  Until a
    # sealed C0B/contest-authority owner exports a reopenable production
    # surface, no amount of typed research evidence may promote.
    refusals.append("AUTHORITY_EMITTER_OWED_SEALED_C0B_OR_CONTEST_OWNER")
    body["refusals"] = refusals
    return _write_receipt(output_path, body)


def _reopen_full_chain(path: Path | str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Reopen every predecessor and verify immutable cross-stage links."""

    chain: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    identity, receipt = reopen_receipt(path)
    while True:
        key = (identity["path"], identity["sha256"])
        if key in seen:
            raise AdversarialGateError("gate receipt predecessor chain contains a cycle")
        seen.add(key)
        chain.append((identity, receipt))
        predecessor = receipt.get("predecessor")
        if predecessor is None:
            break
        if not isinstance(predecessor, Mapping):
            raise AdversarialGateError("gate predecessor link is not an exact receipt identity")
        prior_identity, prior = reopen_receipt(predecessor.get("path"))
        expected_link = {
            **prior_identity,
            "body_sha256": prior["body_sha256"],
            "stage": prior["stage"],
        }
        if dict(predecessor) != expected_link:
            raise AdversarialGateError("gate predecessor file/body/stage identity changed")
        for field in (
            "campaign_id",
            "requested_representation",
            "repo_root",
            "frontier",
        ):
            if receipt.get(field) != prior.get(field):
                raise AdversarialGateError(f"gate predecessor chain switched {field}")
        try:
            current_position = STAGES.index(receipt["stage"])
            predecessor_position = STAGES.index(prior["stage"])
        except (KeyError, ValueError) as exc:
            raise AdversarialGateError("gate predecessor chain contains an unknown stage") from exc
        if current_position != predecessor_position + 1:
            raise AdversarialGateError("gate predecessor chain skipped or repeated a lifecycle stage")
        identity, receipt = prior_identity, prior
    if chain[-1][1].get("stage") != CAMPAIGN_SEAL:
        raise AdversarialGateError("gate receipt chain does not terminate at CAMPAIGN_SEAL")
    expected_prefix = STAGES[: STAGES.index(chain[0][1]["stage"]) + 1]
    if tuple(row["stage"] for _, row in reversed(chain)) != expected_prefix:
        raise AdversarialGateError("gate receipt chain is not the exact lifecycle prefix")
    return chain


def require_live_admission_receipt(
    path: Path | str,
    *,
    expected_stage: str,
    expected_campaign_id: str | None = None,
    expected_repo_root: Path | str | None = None,
    expected_representation: str | None = None,
    expected_config_path: Path | str | None = None,
    expected_archive_path: Path | str | None = None,
    expected_public_auth_path: Path | str | None = None,
) -> dict[str, Any]:
    """Launcher guard: return only a current, live, admitted exact stage."""

    chain = _reopen_full_chain(path)
    receipt = chain[0][1]
    _snapshot_from_receipt(receipt, expected_repo_root=expected_repo_root)
    if (
        receipt["mode"] != LIVE
        or receipt["stage"] != expected_stage
        or receipt["status"] != "ADMIT"
        or receipt["candidate_admission"] is not True
    ):
        raise AdversarialGateError("launcher prerequisite is not a live admitted stage")
    if expected_campaign_id is not None and receipt["campaign_id"] != expected_campaign_id:
        raise AdversarialGateError("launcher prerequisite belongs to another campaign cycle")
    if expected_representation is not None and receipt["requested_representation"] != expected_representation:
        raise AdversarialGateError("launcher prerequisite representation mismatch")
    if expected_config_path is not None:
        expected_config = receipt.get("producer_config")
        actual_config = artifact_identity(expected_config_path, label="launcher producer config")
        if not isinstance(expected_config, Mapping) or actual_config != dict(expected_config):
            raise AdversarialGateError("launcher prerequisite producer config mismatch")
    if expected_archive_path is not None:
        expected = receipt.get("archive")
        if not isinstance(expected, Mapping):
            raise AdversarialGateError("launcher prerequisite has no archive identity")
        actual = artifact_identity(expected_archive_path, label="launcher archive")
        if actual != dict(expected):
            raise AdversarialGateError("launcher attempted to switch archive object")
    if expected_public_auth_path is not None:
        expected_auth = receipt.get("evidence", {}).get("public_auth_receipt", {}).get("legacy_g55_auth")
        actual_auth = artifact_identity(
            expected_public_auth_path,
            label="launcher G55 public auth receipt",
        )
        if not isinstance(expected_auth, Mapping) or actual_auth != dict(expected_auth):
            raise AdversarialGateError("launcher public-auth receipt differs from strict G59 closure")
    pre_encode = next(
        (row for _, row in chain if row["stage"] == PRE_ENCODE),
        None,
    )
    if pre_encode is None:
        raise AdversarialGateError("launcher gate chain lacks PRE_ENCODE")
    if pre_encode["requested_representation"] == PROGRAM_RESIDUAL_LAYERED:
        config_blockers = _validate_program_producer_config(
            pre_encode.get("producer_config"),
            pre_encode.get("evidence", {}),
        )
        if config_blockers:
            raise AdversarialGateError(config_blockers[0])
    else:
        raise AdversarialGateError("DIRECT_CONTROL_PRODUCTION_ADAPTER_NOT_SUPPLIED")
    return receipt


def audit_g57_retrospective(
    *,
    campaign_id: str,
    requested_representation: str,
    repo_root: Path | str,
    g57_request_path: Path | str,
    g57_receipt_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """Preserve G57 as regression evidence without conferring live authority."""

    frontier = load_dynamic_frontier_target(repo_root=Path(repo_root))
    request_record, _ = _load_json_artifact(g57_request_path, label="G57 request")
    receipt_record, _ = _load_json_artifact(g57_receipt_path, label="G57 receipt")
    body = _receipt_common(
        stage=PRE_ENCODE,
        mode=RETROSPECTIVE_ONLY,
        campaign_id=campaign_id,
        representation=requested_representation,
        repo_root=repo_root,
        frontier=frontier,
        predecessor=None,
    )
    body.update(
        status="RETROSPECTIVE_ONLY",
        candidate_admission=False,
        learning_admission=True,
        evidence={"g57_request": request_record, "g57_receipt": receipt_record},
        refusals=["RETROSPECTIVE_G57_EVIDENCE_CANNOT_SATISFY_LIVE_LAUNCH"],
    )
    return _write_receipt(output_path, body)


__all__ = [
    "BLOCKER_SCHEMA",
    "CAMPAIGN_SEAL",
    "DIRECT_TASK_LAYERED_CONTROL",
    "ENCODE",
    "EXACT_EVAL_SCHEMA",
    "EXACT_REPORT_SCHEMA",
    "INTEGRATION_HOOKS",
    "INTEGRATION_SCHEMA",
    "LIVE",
    "POST_EVAL",
    "PRE_ENCODE",
    "PRE_PROMOTION",
    "PRE_PUBLIC_CLOSURE",
    "PROGRAM_RESIDUAL_LAYERED",
    "RECEIPT_SCHEMA",
    "RETROSPECTIVE_ONLY",
    "AdversarialGateError",
    "admit_encode",
    "admit_post_eval",
    "admit_pre_encode",
    "admit_pre_promotion",
    "admit_pre_public_closure",
    "artifact_identity",
    "audit_g57_retrospective",
    "canonical_json",
    "reopen_receipt",
    "require_live_admission_receipt",
    "seal_campaign",
    "sha256_bytes",
]
