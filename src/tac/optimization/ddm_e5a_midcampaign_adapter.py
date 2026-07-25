# SPDX-License-Identifier: MIT
"""Typed campaign-checkpoint adapter for the existing E5 WS1 compiler.

The adapter is deliberately not an exporter.  It materializes the live
resume-state shadow recorded by a canonical joint-descent checkpoint into the
same receiver-closed WS1 bytes already accepted by E5, and records the exact
checkpoint -> state custody edge.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from tac.optimization.ddm_ws1_warm_start import (
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
BLOCKER = "R6_BLOCKED_E5_MIDCAMP_CHECKPOINT_ADAPTER_ABSENT"


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


__all__ = [
    "BLOCKER",
    "CONFIG_SCHEMA",
    "SCHEMA",
    "DDME5AMidcampaignCheckpointAdapterConfigV1",
    "E5AMidcampaignAdapterError",
    "materialize_midcampaign_checkpoint",
]
