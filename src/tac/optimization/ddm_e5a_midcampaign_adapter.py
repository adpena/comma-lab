# SPDX-License-Identifier: MIT
"""Typed campaign-checkpoint adapter for the existing E5 WS1 compiler.

The adapter is deliberately not an exporter.  It materializes the live
resume-state shadow recorded by a canonical joint-descent checkpoint into the
same receiver-closed WS1 bytes already accepted by E5, and records the exact
checkpoint -> state custody edge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from tac.optimization.ddm_ws1_warm_start import (
    W_JOINT,
    compile_ws1_warm_start_archive,
    parse_ws1_warm_start_archive,
    receive_ws1_warm_start_archive,
)
from tac.optimization.direct_description_joint_descent import (
    DirectDescriptionJointDescentTypedConfigV1,
    compile_parameterized_archive,
    lift_v15_archive,
    load_stage_checkpoint,
)
from tac.optimization.direct_description_measurement_ladder import rfc8785_canonicalize

SCHEMA = "ddm_e5a_midcampaign_checkpoint_adapter_receipt.v1"
CONFIG_SCHEMA = "DDME5AMidcampaignCheckpointAdapterConfigV1"
SOLVE_MEMBER_SCHEMA = "ddm_e5a_solve_member_adapter_receipt.v1"
SOLVE_MEMBER_CONFIG_SCHEMA = "DDME5ASolveMemberAdapterConfigV1"
BLOCKER = "R6_BLOCKED_E5_MIDCAMP_CHECKPOINT_ADAPTER_ABSENT"
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
KNEE_MEMBERS: Final = (
    "manifest.json",
    "components/00_n600_0145_q8_sparse__128.zip.receipt-bytes",
    "components/01_01_local_statistics_payload.bin",
)


class E5AMidcampaignAdapterError(ValueError):
    """Checkpoint or receiver-state custody failed closed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ssd_path(value: str, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not (
        value.startswith("/Volumes/VertigoDataTier/pact/")
        or value.startswith("/Volumes/APDataStore/pact/")
    ):
        raise E5AMidcampaignAdapterError(f"{label} must use governed SSD custody")
    return path


def _publish_or_verify(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise E5AMidcampaignAdapterError(f"existing immutable artifact differs: {path}")
        return path
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


class DDME5AMidcampaignCheckpointAdapterConfigV1(BaseModel):
    """Hash-bound checkpoint-to-E5-state adapter declaration."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_: Literal["DDME5AMidcampaignCheckpointAdapterConfigV1"] = Field(
        default=CONFIG_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    run_id: Literal["ddm_e5a_midcampaign_e5_adapter_20260725"] = (
        "ddm_e5a_midcampaign_e5_adapter_20260725"
    )
    ticket_path: StrictStr
    ticket_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_path: StrictStr
    checkpoint_bytes: StrictInt = Field(gt=0)
    checkpoint_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    expected_stage_id: StrictStr = Field(min_length=1)
    expected_global_step: StrictInt = Field(ge=0)
    expected_parameter_shadow: Literal["live_resume_state"] = "live_resume_state"
    expected_lane_programs_materialized: StrictBool
    expected_state_bytes: StrictInt = Field(gt=0)
    expected_state_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    output_state_path: StrictStr
    output_receipt_path: StrictStr
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DDME5AMidcampaignCheckpointAdapterConfigV1:
        for label, value in (
            ("checkpoint_path", self.checkpoint_path),
            ("output_state_path", self.output_state_path),
            ("output_receipt_path", self.output_receipt_path),
        ):
            _ssd_path(value, label=label)
        if Path(self.output_state_path) == Path(self.checkpoint_path):
            raise ValueError("materialized state must not overwrite its checkpoint")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))


class DDME5ASolveMemberAdapterConfigV1(BaseModel):
    """Hash-bind one RD1 solve member into E5A's existing WS1 state type."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_: Literal["DDME5ASolveMemberAdapterConfigV1"] = Field(
        default=SOLVE_MEMBER_CONFIG_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    run_id: Literal["ddm_ks1_knee_member_realization_20260725"] = (
        "ddm_ks1_knee_member_realization_20260725"
    )
    source_bundle_path: StrictStr
    source_bundle_bytes: StrictInt = Field(gt=0)
    source_bundle_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    source_receipt_path: StrictStr
    source_receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: Literal["statistics_hard_analytic_composed_frame1"] = (
        "statistics_hard_analytic_composed_frame1"
    )
    expected_description_root_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    expected_base_bytes: StrictInt = Field(gt=0)
    expected_base_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    expected_payload_bytes: Literal[974] = 974
    expected_payload_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    expected_state_bytes: StrictInt = Field(gt=0)
    expected_state_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    output_state_path: StrictStr
    output_receipt_path: StrictStr
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DDME5ASolveMemberAdapterConfigV1:
        for label, value in (
            ("source_bundle_path", self.source_bundle_path),
            ("source_receipt_path", self.source_receipt_path),
        ):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{label} must be repository-relative")
        for label, value in (
            ("output_state_path", self.output_state_path),
            ("output_receipt_path", self.output_receipt_path),
        ):
            _ssd_path(value, label=label)
        if self.expected_base_bytes + self.expected_payload_bytes != self.expected_state_bytes:
            raise ValueError("solve-member base plus payload bytes must close the WS1 state")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))


def _repo_file(value: str, *, label: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise E5AMidcampaignAdapterError(f"{label} escaped repository custody") from exc
    if not path.is_file():
        raise E5AMidcampaignAdapterError(f"{label} is absent")
    return path


def compile_solve_member_bundle(
    config: DDME5ASolveMemberAdapterConfigV1,
) -> tuple[bytes, dict[str, Any]]:
    """Compile the RD1 custody bundle into the canonical E5A WS1 state."""

    source_receipt_path = _repo_file(config.source_receipt_path, label="source_receipt_path")
    source_receipt = source_receipt_path.read_bytes()
    if _sha256(source_receipt) != config.source_receipt_sha256:
        raise E5AMidcampaignAdapterError("RD1 source receipt custody mismatch")
    try:
        receipt_value = json.loads(source_receipt)
    except json.JSONDecodeError as exc:
        raise E5AMidcampaignAdapterError("RD1 source receipt is malformed") from exc

    bundle_path = _repo_file(config.source_bundle_path, label="source_bundle_path")
    bundle = bundle_path.read_bytes()
    if (len(bundle), _sha256(bundle)) != (
        config.source_bundle_bytes,
        config.source_bundle_sha256,
    ):
        raise E5AMidcampaignAdapterError("RD1 knee bundle custody mismatch")
    try:
        with zipfile.ZipFile(bundle_path, "r") as handle:
            if tuple(handle.namelist()) != KNEE_MEMBERS:
                raise E5AMidcampaignAdapterError("RD1 knee bundle member grammar differs")
            manifest_payload = handle.read(KNEE_MEMBERS[0])
            base = handle.read(KNEE_MEMBERS[1])
            payload = handle.read(KNEE_MEMBERS[2])
    except zipfile.BadZipFile as exc:
        raise E5AMidcampaignAdapterError("RD1 knee bundle is malformed") from exc
    try:
        manifest = json.loads(manifest_payload)
    except json.JSONDecodeError as exc:
        raise E5AMidcampaignAdapterError("RD1 knee bundle manifest is malformed") from exc
    components = manifest.get("components")
    if (
        manifest.get("schema") != "ddm_rd1_knee_full_description_bundle.v1"
        or manifest.get("candidate_id") != config.candidate_id
        or manifest.get("description_root_sha256")
        != config.expected_description_root_sha256
        or manifest.get("logical_counted_bytes") != config.expected_state_bytes
        or manifest.get("bundle_is_custody_container_not_counted_archive") is not True
        or not isinstance(components, list)
        or len(components) != 2
    ):
        raise E5AMidcampaignAdapterError("RD1 knee bundle semantic identity differs")
    expected_components = (
        (KNEE_MEMBERS[1], config.expected_base_bytes, config.expected_base_sha256),
        (KNEE_MEMBERS[2], config.expected_payload_bytes, config.expected_payload_sha256),
    )
    for row, (name, bytes_, sha256) in zip(components, expected_components, strict=True):
        if (
            not isinstance(row, dict)
            or (row.get("bundle_name"), row.get("bytes"), row.get("sha256"))
            != (name, bytes_, sha256)
        ):
            raise E5AMidcampaignAdapterError("RD1 knee component manifest differs")
    if (len(base), _sha256(base)) != (
        config.expected_base_bytes,
        config.expected_base_sha256,
    ) or (len(payload), _sha256(payload)) != (
        config.expected_payload_bytes,
        config.expected_payload_sha256,
    ):
        raise E5AMidcampaignAdapterError("RD1 knee component bytes differ")

    candidate_domain = receipt_value.get("candidate_domain")
    if not isinstance(candidate_domain, list):
        raise E5AMidcampaignAdapterError("RD1 source receipt lacks candidate domain")
    matches = [
        row
        for row in candidate_domain
        if isinstance(row, dict) and row.get("candidate_id") == config.candidate_id
    ]
    if len(matches) != 1 or (
        matches[0].get("counted_bytes"),
        matches[0].get("description_root_sha256"),
        matches[0].get("receiver_closure"),
    ) != (
        config.expected_state_bytes,
        config.expected_description_root_sha256,
        "measurement_harness_receiver_closed",
    ):
        raise E5AMidcampaignAdapterError("RD1 source receipt knee row differs")

    state = compile_ws1_warm_start_archive(
        base,
        candidate=W_JOINT,
        payload=payload,
    )
    if (len(state), _sha256(state)) != (
        config.expected_state_bytes,
        config.expected_state_sha256,
    ):
        raise E5AMidcampaignAdapterError("realized knee state identity differs")
    parsed = parse_ws1_warm_start_archive(state)
    receiver = receive_ws1_warm_start_archive(state)
    if (
        parsed.candidate_id != config.candidate_id
        or parsed.exact_reemit() != state
        or receiver.parsed.exact_reemit() != state
    ):
        raise E5AMidcampaignAdapterError("realized knee state receiver closure differs")
    proof = {
        "bundle": {
            "bytes": len(bundle),
            "path": config.source_bundle_path,
            "sha256": _sha256(bundle),
        },
        "candidate_id": config.candidate_id,
        "components": [
            {"bytes": len(base), "name": KNEE_MEMBERS[1], "sha256": _sha256(base)},
            {"bytes": len(payload), "name": KNEE_MEMBERS[2], "sha256": _sha256(payload)},
        ],
        "description_root_sha256": config.expected_description_root_sha256,
        "source_receipt": {
            "path": config.source_receipt_path,
            "sha256": _sha256(source_receipt),
        },
    }
    return state, proof


def materialize_solve_member_bundle(
    config: DDME5ASolveMemberAdapterConfigV1,
) -> dict[str, Any]:
    """Publish one RD1 solve member under E5A's existing state contract."""

    state, source_proof = compile_solve_member_bundle(config)
    state_path = _publish_or_verify(
        _ssd_path(config.output_state_path, label="output_state_path"),
        state,
    )
    receipt: dict[str, Any] = {
        "schema": SOLVE_MEMBER_SCHEMA,
        "status": "PASS",
        "adapter_not_exporter": True,
        "source": source_proof,
        "state": {
            "bytes": len(state),
            "candidate": W_JOINT,
            "candidate_id": config.candidate_id,
            "path": str(state_path),
            "sha256": _sha256(state),
        },
        "proof": {
            "consumer_roundtrip_byte_identical": True,
            "parse_reemit_byte_identical": True,
            "source_bundle_mutated": False,
            "two_typed_streams_consumed": True,
        },
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "typed_config_sha256": config.typed_config_hash(),
    }
    receipt_payload = rfc8785_canonicalize(receipt) + b"\n"
    _publish_or_verify(
        _ssd_path(config.output_receipt_path, label="output_receipt_path"),
        receipt_payload,
    )
    return receipt


def materialize_midcampaign_checkpoint(
    config: DDME5AMidcampaignCheckpointAdapterConfigV1,
) -> dict[str, Any]:
    """Materialize and prove one checkpoint's live resume-state shadow."""

    ticket_path = Path(config.ticket_path)
    ticket_payload = ticket_path.read_bytes()
    if _sha256(ticket_payload) != config.ticket_sha256:
        raise E5AMidcampaignAdapterError("ticket custody mismatch")
    typed = DirectDescriptionJointDescentTypedConfigV1.from_ticket(ticket_path)
    checkpoint_path = _ssd_path(config.checkpoint_path, label="checkpoint_path")
    checkpoint_payload = checkpoint_path.read_bytes()
    if (len(checkpoint_payload), _sha256(checkpoint_payload)) != (
        config.checkpoint_bytes,
        config.checkpoint_sha256,
    ):
        raise E5AMidcampaignAdapterError("checkpoint custody mismatch")
    state, metadata = load_stage_checkpoint(checkpoint_path, config=typed)
    realized = metadata.get("realized_archive")
    cursor = metadata.get("run_cursor")
    if (
        not isinstance(realized, dict)
        or not isinstance(cursor, dict)
        or metadata.get("stage_id") != config.expected_stage_id
        or int(cursor.get("global_step", -1)) != config.expected_global_step
        or realized.get("parameter_shadow") != config.expected_parameter_shadow
        or realized.get("lane_programs_materialized")
        is not config.expected_lane_programs_materialized
    ):
        raise E5AMidcampaignAdapterError("checkpoint semantic cursor/shadow custody mismatch")
    source_archive = Path(typed.source_archive_path).read_bytes()
    if (len(source_archive), _sha256(source_archive)) != (
        typed.source_archive_bytes,
        typed.source_archive_sha256,
    ):
        raise E5AMidcampaignAdapterError("typed ticket source archive custody mismatch")
    lift = lift_v15_archive(source_archive)
    materialized, realized_theta = compile_parameterized_archive(
        lift,
        state.theta,
        include_lane_programs=config.expected_lane_programs_materialized,
    )
    expected_identity = (
        config.expected_state_bytes,
        config.expected_state_sha256,
    )
    if (len(materialized), _sha256(materialized)) != expected_identity or (
        int(realized.get("bytes", -1)),
        realized.get("sha256"),
    ) != expected_identity:
        raise E5AMidcampaignAdapterError("materialized state differs from checkpoint identity")
    parsed = parse_ws1_warm_start_archive(materialized)
    if parsed.exact_reemit() != materialized:
        raise E5AMidcampaignAdapterError("materialized WS1 parse/re-emit differs")
    receiver = receive_ws1_warm_start_archive(materialized)
    if receiver.parsed.exact_reemit() != materialized:
        raise E5AMidcampaignAdapterError("WS1 consumer roundtrip differs")

    state_path = _publish_or_verify(
        _ssd_path(config.output_state_path, label="output_state_path"),
        materialized,
    )
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS",
        "blocker_dissolved": BLOCKER,
        "adapter_not_exporter": True,
        "checkpoint": {
            "bytes": len(checkpoint_payload),
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_payload),
            "schema": metadata["schema"],
            "stage_id": metadata["stage_id"],
            "global_step": cursor["global_step"],
        },
        "resume_manifest": {
            "canonical_resume_registry": metadata["canonical_resume_registry"],
            "ema_shadow_saved": metadata["ema_shadow_saved"],
            "live_weights_saved_for_resume_only": metadata["live_weights_saved_for_resume_only"],
            "optimizer": metadata["optimizer"],
            "parameter_shadow_materialized": "live_resume_state",
            "typed_config_hash": metadata["typed_config_hash"],
        },
        "state": {
            "bytes": len(materialized),
            "candidate": parsed.candidate,
            "lane_programs_materialized": config.expected_lane_programs_materialized,
            "path": str(state_path),
            "realized_parameter_count": int(realized.get("realized_parameter_count", len(realized_theta))),
            "sha256": _sha256(materialized),
        },
        "proof": {
            "checkpoint_identity_matches_manifest": True,
            "consumer_roundtrip_byte_identical": True,
            "parse_reemit_byte_identical": True,
            "source_archive_mutated": False,
        },
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "typed_config_sha256": config.typed_config_hash(),
    }
    receipt_payload = rfc8785_canonicalize(receipt) + b"\n"
    _publish_or_verify(
        _ssd_path(config.output_receipt_path, label="output_receipt_path"),
        receipt_payload,
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solve-member-config")
    args = parser.parse_args(argv)
    if not args.solve_member_config:
        parser.error("--solve-member-config is required")
    config_path = Path(args.solve_member_config).resolve()
    try:
        config_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise E5AMidcampaignAdapterError("config path must stay in the repository") from exc
    payload = config_path.read_bytes()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise E5AMidcampaignAdapterError("solve-member config is malformed") from exc
    config = DDME5ASolveMemberAdapterConfigV1.model_validate(value, strict=True)
    if payload != rfc8785_canonicalize(
        config.model_dump(mode="json", by_alias=True)
    ) + b"\n":
        raise E5AMidcampaignAdapterError("solve-member config must be canonical JSON")
    result = materialize_solve_member_bundle(config)
    print(
        json.dumps(
            {
                "receipt_path": config.output_receipt_path,
                "state": result["state"],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "BLOCKER",
    "CONFIG_SCHEMA",
    "SCHEMA",
    "SOLVE_MEMBER_CONFIG_SCHEMA",
    "SOLVE_MEMBER_SCHEMA",
    "DDME5AMidcampaignCheckpointAdapterConfigV1",
    "DDME5ASolveMemberAdapterConfigV1",
    "E5AMidcampaignAdapterError",
    "compile_solve_member_bundle",
    "materialize_midcampaign_checkpoint",
    "materialize_solve_member_bundle",
]


if __name__ == "__main__":
    raise SystemExit(main())
