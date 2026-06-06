# SPDX-License-Identifier: MIT
"""Fail-closed authority gate for the SNeRV official TUB LF/HF replacement lane."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

SCHEMA = "snerv_official_tub_lf_hf_decoder_replacement_authority_gate.v1"
GATE_SCHEMA = "snerv_official_tub_lf_hf_decoder_replacement_gate_row.v1"
DEFAULT_LANE_ID = "lane_snerv_official_tub_lf_hf_decoder_replacement_20260605"
DEFAULT_MIN_FREE_BYTES = 1_000_000_000
SSD_ROOTS = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)

QUEUE_FALSE_AUTHORITY = {
    **FALSE_AUTHORITY,
    "score_claim": False,
    "score_claim_valid": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "production_hardened_claim": False,
    "source_faithful_stack_claim": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "local_mlx_long_training_allowed": False,
    "dispatch_allowed": False,
    "exact_or_full_video_cuda_allowed": False,
}

EXPORT_BLOCKER = "snerv_official_mfu_hfr_tub_export_not_bound"
RECEIVER_PAYLOAD_BLOCKER = "snerv_official_mfu_hfr_tub_receiver_payload_not_bound"
FRAME_EXPORT_BLOCKER = "snerv_official_mfu_hfr_tub_frame_producing_export_missing"
OUTPUT2_BLOCKER = "snerv_official_tub_output2_receiver_frame_decode_not_bound"
SOURCE_AUTHORITY_BLOCKER = (
    "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority"
)
FULL_REPLAY_BLOCKER = (
    "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing"
)
TUB_SOURCE_FIXTURE_BLOCKER = "snerv_official_tub_source_fixture_replay_missing"
TRAINED_STATE_BLOCKER = "snerv_official_trained_checkpoint_state_dict_not_loaded"
TRAINED_MAPPING_BLOCKER = "snerv_official_trained_checkpoint_state_dict_mapping_missing"
WEIGHT_MAPPING_BLOCKER = "snerv_official_mfu_hfr_tub_weight_mapping_missing"
HFR_WEIGHT_MAPPING_BLOCKER = (
    "snerv_official_trained_checkpoint_hfr_weight_mapping_incomplete"
)
MFU_WEIGHT_MAPPING_BLOCKER = (
    "snerv_official_trained_checkpoint_mfu_weight_mapping_incomplete"
)
MFU_ACTIVATION_NOT_WEIGHT_BLOCKER = (
    "snerv_official_mfu_native_receiver_activation_payload_not_upstream_weight_mapping"
)
TUB_WEIGHT_BLOCKER = (
    "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded"
)
TUB_TEMPORAL_MAPPING_BLOCKER = (
    "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing"
)
TUB_OUTPUT2_MAPPING_BLOCKER = (
    "snerv_official_tub_portable_output2_decoder_weight_mapping_missing"
)
SOURCE_ARTIFACT_MISSING_BLOCKER = (
    "snerv_official_tub_lf_hf_decoder_replacement_source_forward_artifact_missing"
)
CHECKPOINT_REPORT_MISSING_BLOCKER = (
    "snerv_official_tub_lf_hf_decoder_replacement_checkpoint_export_report_missing"
)
AUTHORITY_GATE_MISSING_BLOCKER = (
    "snerv_official_tub_lf_hf_decoder_replacement_authority_gate_missing"
)


class SnervOfficialTubLfHfReplacementAuthorityGateError(ValueError):
    """Raised when the replacement authority gate cannot be built."""


def build_snerv_official_tub_lf_hf_replacement_authority_gate(
    *,
    source_forward_artifacts: Sequence[Mapping[str, Any]] = (),
    checkpoint_export_reports: Sequence[Mapping[str, Any]] = (),
    tub_source_forward_artifacts: Sequence[Mapping[str, Any]] = (),
    output_root: str | Path,
    lane_id: str = DEFAULT_LANE_ID,
    generated_utc: str | None = None,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    allow_local_output: bool = False,
) -> dict[str, Any]:
    """Build a queue-consumable proof/blocker report for the official TUB lane."""

    if not str(lane_id).strip():
        raise SnervOfficialTubLfHfReplacementAuthorityGateError(
            "lane_id must be non-empty"
        )
    generated = generated_utc or datetime.now(UTC).isoformat()
    root = Path(output_root)
    storage_preflight = _storage_preflight(
        root,
        min_free_bytes=int(min_free_bytes),
        allow_local_output=bool(allow_local_output),
    )
    source = _select_source_forward_artifact(source_forward_artifacts)
    checkpoint = _select_checkpoint_export_report(checkpoint_export_reports)
    tub = _select_tub_source_forward_artifact(tub_source_forward_artifacts)

    source_state = _source_state(source)
    checkpoint_state = _checkpoint_state(checkpoint, source_state)
    tub_state = _tub_state(tub, source_state)
    tub_fixture_replay_ready = bool(
        source_state["tub_source_fixture_replay_ready"]
        or tub_state["fixture_source_replay_passed"]
    )
    tub_fixture_replay_blockers = (
        []
        if tub_fixture_replay_ready
        else [TUB_SOURCE_FIXTURE_BLOCKER]
    )
    gates = [
        _gate(
            "official_checkpoint_export_binding",
            [],
            checkpoint_state["official_checkpoint_export_bound"],
            []
            if checkpoint_state["official_checkpoint_export_bound"]
            else [EXPORT_BLOCKER],
        ),
        _gate(
            "receiver_output2_frame_replay",
            ["official_checkpoint_export_binding"],
            source_state["receiver_output2_frame_replay_ready"],
            source_state["receiver_output2_frame_replay_blockers"],
        ),
        _gate(
            "tub_source_fixture_replay",
            ["receiver_output2_frame_replay"],
            tub_fixture_replay_ready,
            tub_fixture_replay_blockers,
        ),
        _gate(
            "trained_checkpoint_state_dict_mapping",
            ["tub_source_fixture_replay"],
            checkpoint_state["trained_checkpoint_state_dict_mapping_ready"],
            checkpoint_state["trained_checkpoint_mapping_blockers"],
        ),
        _gate(
            "tub_temporal_output2_weight_mapping",
            ["trained_checkpoint_state_dict_mapping"],
            checkpoint_state["tub_temporal_output2_weight_mapping_ready"],
            checkpoint_state["tub_temporal_output2_mapping_blockers"],
        ),
        _gate(
            "full_tub_source_forward_replay",
            ["tub_source_fixture_replay", "tub_temporal_output2_weight_mapping"],
            source_state["full_tub_source_forward_replay_ready"],
            [
                blocker
                for blocker in (
                    SOURCE_AUTHORITY_BLOCKER,
                    FULL_REPLAY_BLOCKER,
                )
                if blocker in source_state["source_forward_authority_blockers"]
            ],
        ),
    ]
    replacement_ready = all(not gate["blocked"] for gate in gates)
    replacement_blockers = _dedupe(
        [blocker for gate in gates for blocker in gate["blockers"]]
    )
    gates.append(
        _gate(
            "official_tub_lf_hf_decoder_replacement",
            [gate["gate_id"] for gate in gates],
            replacement_ready,
            replacement_blockers,
        )
    )
    closed = _dedupe(
        [
            *checkpoint_state["closed_campaign_blockers"],
            *source_state["closed_campaign_blockers"],
            *tub_state["closed_campaign_blockers"],
        ]
    )
    raw_evidence_blockers = _dedupe(
        [
            *source_state["blockers"],
            *checkpoint_state["blockers"],
            *tub_state["blockers"],
        ]
    )
    queue_blockers = _dedupe(
        [
            *replacement_blockers,
            *raw_evidence_blockers,
            *(
                [SOURCE_ARTIFACT_MISSING_BLOCKER]
                if source_state.get("artifact_count") == 0
                else []
            ),
            *(
                [CHECKPOINT_REPORT_MISSING_BLOCKER]
                if checkpoint_state.get("artifact_count") == 0
                else []
            ),
        ]
    )
    queue_blockers = [
        blocker for blocker in queue_blockers if blocker not in set(closed)
    ]
    rebuild_command = _rebuild_command(
        output_root=root,
        source_path=source_state.get("source_path"),
        checkpoint_path=checkpoint_state.get("source_path"),
        tub_path=tub_state.get("source_path"),
    )
    next_unblock_command = _next_unblock_command(
        output_root=root,
        source_path=source_state.get("source_path"),
        checkpoint_path=checkpoint_state.get("source_path"),
        tub_path=tub_state.get("source_path"),
    )
    return {
        "schema": SCHEMA,
        "lane_id": str(lane_id),
        "generated_utc": generated,
        "authority": "false_authority_queue_gate_no_score_claim",
        "family": "snerv",
        "solution_family": "official_tub_lf_hf_decoder_replacement",
        "allowed_use": (
            "queue-owned local bounded replacement-lane admission after measured "
            "LF/HF payload and official receiver/source-forward proofs"
        ),
        "forbidden_use": (
            "score claim, promotion, rank/kill decision, exact eval dispatch, "
            "or proof of source-forward parity without full replay rows"
        ),
        "storage_preflight": storage_preflight,
        "source_forward_evidence": source_state,
        "checkpoint_export_evidence": checkpoint_state,
        "tub_source_forward_evidence": tub_state,
        "gate_rows": gates,
        "gate_row_count": len(gates),
        "blocked_gate_row_count": sum(1 for gate in gates if gate["blocked"]),
        "official_checkpoint_export_binding_ready": checkpoint_state[
            "official_checkpoint_export_bound"
        ],
        "receiver_output2_frame_replay_ready": source_state[
            "receiver_output2_frame_replay_ready"
        ],
        "tub_source_fixture_replay_ready": tub_fixture_replay_ready,
        "trained_checkpoint_state_dict_mapping_ready": checkpoint_state[
            "trained_checkpoint_state_dict_mapping_ready"
        ],
        "tub_temporal_output2_weight_mapping_ready": checkpoint_state[
            "tub_temporal_output2_weight_mapping_ready"
        ],
        "full_tub_source_forward_replay_ready": source_state[
            "full_tub_source_forward_replay_ready"
        ],
        "official_tub_lf_hf_decoder_replacement_ready": replacement_ready,
        "closed_campaign_blockers": closed,
        "raw_evidence_blockers": raw_evidence_blockers,
        "queue_blockers": queue_blockers,
        "blockers": _dedupe(
            [
                "snerv_official_tub_lf_hf_decoder_replacement_false_authority",
                *queue_blockers,
            ]
        ),
        "target_consumers": [
            "build_snerv_lf_hf_replacement_queue",
            "nerv_long_training_campaign_plan",
            "nerv_rate_allocator_queue",
            "cathedral_autopilot",
        ],
        "runnable_rebuild_command_argv": rebuild_command,
        "next_unblock_command_argv": next_unblock_command,
        **QUEUE_FALSE_AUTHORITY,
    }


def summarize_snerv_official_tub_lf_hf_replacement_authority_gates(
    authority_gates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize authority-gate artifacts for queue consumers."""

    artifacts = [
        artifact
        for artifact in authority_gates
        if isinstance(artifact, Mapping) and artifact.get("schema") == SCHEMA
    ]
    if not artifacts:
        return {
            "schema": "snerv_official_tub_lf_hf_replacement_authority_state.v1",
            "artifact_count": 0,
            "selected_artifact_schema": None,
            "source_path": None,
            "source_sha256": None,
            "official_tub_lf_hf_decoder_replacement_ready": False,
            "closed_campaign_blockers": [],
            "queue_blockers": [AUTHORITY_GATE_MISSING_BLOCKER],
            "blockers": [AUTHORITY_GATE_MISSING_BLOCKER],
            **QUEUE_FALSE_AUTHORITY,
        }
    selected = max(
        artifacts,
        key=lambda artifact: (
            bool(artifact.get("official_tub_lf_hf_decoder_replacement_ready")),
            str(artifact.get("generated_utc") or ""),
            str(artifact.get("_source_path") or ""),
        ),
    )
    queue_blockers = _dedupe(selected.get("queue_blockers") or ())
    return {
        "schema": "snerv_official_tub_lf_hf_replacement_authority_state.v1",
        "artifact_count": len(artifacts),
        "selected_artifact_schema": selected.get("schema"),
        "selected_artifact_generated_utc": selected.get("generated_utc"),
        "source_path": selected.get("_source_path"),
        "source_sha256": selected.get("_source_sha256"),
        "official_tub_lf_hf_decoder_replacement_ready": (
            selected.get("official_tub_lf_hf_decoder_replacement_ready") is True
        ),
        "official_checkpoint_export_binding_ready": (
            selected.get("official_checkpoint_export_binding_ready") is True
        ),
        "receiver_output2_frame_replay_ready": (
            selected.get("receiver_output2_frame_replay_ready") is True
        ),
        "tub_source_fixture_replay_ready": (
            selected.get("tub_source_fixture_replay_ready") is True
        ),
        "trained_checkpoint_state_dict_mapping_ready": (
            selected.get("trained_checkpoint_state_dict_mapping_ready") is True
        ),
        "tub_temporal_output2_weight_mapping_ready": (
            selected.get("tub_temporal_output2_weight_mapping_ready") is True
        ),
        "full_tub_source_forward_replay_ready": (
            selected.get("full_tub_source_forward_replay_ready") is True
        ),
        "closed_campaign_blockers": _dedupe(
            selected.get("closed_campaign_blockers") or ()
        ),
        "queue_blockers": queue_blockers,
        "blockers": _dedupe([*(selected.get("blockers") or ()), *queue_blockers]),
        "gate_rows": list(selected.get("gate_rows") or ()),
        "runnable_rebuild_command_argv": list(
            selected.get("runnable_rebuild_command_argv") or ()
        ),
        "next_unblock_command_argv": list(
            selected.get("next_unblock_command_argv") or ()
        ),
        **QUEUE_FALSE_AUTHORITY,
    }


def render_snerv_official_tub_lf_hf_replacement_authority_gate_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render a compact operator handoff."""

    lines = [
        "# SNeRV Official TUB LF/HF Replacement Authority Gate",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- lane: `{report.get('lane_id')}`",
        f"- replacement ready: `{report.get('official_tub_lf_hf_decoder_replacement_ready')}`",
        f"- checkpoint export binding ready: `{report.get('official_checkpoint_export_binding_ready')}`",
        f"- receiver output2 frame replay ready: `{report.get('receiver_output2_frame_replay_ready')}`",
        f"- TUB source fixture replay ready: `{report.get('tub_source_fixture_replay_ready')}`",
        f"- trained checkpoint mapping ready: `{report.get('trained_checkpoint_state_dict_mapping_ready')}`",
        f"- full TUB source-forward replay ready: `{report.get('full_tub_source_forward_replay_ready')}`",
        f"- score claim: `{report.get('score_claim')}`",
        "",
        "## Gates",
    ]
    for gate in report.get("gate_rows", []) if isinstance(report, Mapping) else []:
        if not isinstance(gate, Mapping):
            continue
        lines.extend(
            [
                "",
                f"### `{gate.get('gate_id')}`",
                f"- blocked: `{gate.get('blocked')}`",
                f"- depends on: `{', '.join(gate.get('depends_on') or [])}`",
                "- blockers:",
            ]
        )
        blockers = [str(blocker) for blocker in gate.get("blockers") or ()]
        lines.extend(f"  - `{blocker}`" for blocker in blockers)
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "- rebuild gate:",
            f"  - `{_shell_join(report.get('runnable_rebuild_command_argv') or [])}`",
            "- next unblock:",
            f"  - `{_shell_join(report.get('next_unblock_command_argv') or [])}`",
        ]
    )
    return "\n".join(lines) + "\n"


def attach_source_identity(payload: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    """Attach source path and SHA metadata to a loaded JSON payload."""

    source_path = Path(path)
    data = source_path.read_bytes()
    return {
        **dict(payload),
        "_source_path": source_path.as_posix(),
        "_source_sha256": hashlib.sha256(data).hexdigest(),
    }


def load_json_with_source_identity(path: str | Path) -> dict[str, Any]:
    """Load a JSON object and attach path/SHA metadata for custody."""

    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SnervOfficialTubLfHfReplacementAuthorityGateError(
            f"{source_path}: JSON payload must be an object"
        )
    return attach_source_identity(payload, source_path)


def _select_source_forward_artifact(
    artifacts: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    valid = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping)
        and artifact.get("schema") == "snerv_official_mfu_hfr_tub_forward_parity.v1"
    ]
    if not valid:
        return None
    return max(
        valid,
        key=lambda artifact: (
            bool(artifact.get("full_tub_source_forward_parity_proven")),
            str(artifact.get("generated_utc") or ""),
            str(artifact.get("_source_path") or ""),
        ),
    )


def _select_checkpoint_export_report(
    reports: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    valid = [
        report
        for report in reports
        if isinstance(report, Mapping)
        and report.get("schema") == "snerv_checkpoint_archive_export.v1"
    ]
    if not valid:
        return None
    return max(
        valid,
        key=lambda report: (
            int(report.get("checkpoint_epoch") or -1),
            str(report.get("_source_path") or report.get("report_path") or ""),
        ),
    )


def _select_tub_source_forward_artifact(
    artifacts: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    valid = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping)
        and artifact.get("schema") == "snerv_official_tub_source_forward_replay.v1"
    ]
    if not valid:
        return None
    return max(
        valid,
        key=lambda artifact: (
            bool(artifact.get("full_tub_source_forward_parity_proven")),
            str(artifact.get("generated_utc") or ""),
            str(artifact.get("_source_path") or ""),
        ),
    )


def _source_state(source: Mapping[str, Any] | None) -> dict[str, Any]:
    if source is None:
        return {
            "schema": "snerv_official_tub_lf_hf_source_forward_state.v1",
            "artifact_count": 0,
            "selected_artifact_schema": None,
            "source_path": None,
            "source_sha256": None,
            "receiver_output2_frame_replay_ready": False,
            "tub_source_fixture_replay_ready": False,
            "receiver_payload_source_forward_authority": False,
            "full_tub_source_forward_replay_ready": False,
            "closed_campaign_blockers": [],
            "receiver_output2_frame_replay_blockers": [
                RECEIVER_PAYLOAD_BLOCKER,
                FRAME_EXPORT_BLOCKER,
                OUTPUT2_BLOCKER,
            ],
            "source_forward_authority_blockers": [
                SOURCE_AUTHORITY_BLOCKER,
                FULL_REPLAY_BLOCKER,
            ],
            "blockers": [SOURCE_ARTIFACT_MISSING_BLOCKER],
            **QUEUE_FALSE_AUTHORITY,
        }
    replay = source.get("receiver_payload_frame_replay")
    replay = replay if isinstance(replay, Mapping) else {}
    receiver_runtime_decode = replay.get("receiver_runtime_decode_proven") is True
    frame_replay = replay.get("frame_producing_official_payload_replay_proven") is True
    consumes_output2 = replay.get("receiver_frame_decode_consumes_output2") is True
    payload_bytes = _positive_int(replay.get("payload_bytes"))
    payload_sha256 = str(replay.get("payload_sha256") or "").strip()
    nested_tub = source.get("official_tub_source_forward_replay")
    nested_tub = nested_tub if isinstance(nested_tub, Mapping) else {}
    nested_tub_schema = (
        nested_tub.get("schema")
        if nested_tub.get("schema") == "snerv_official_tub_source_forward_replay.v1"
        else None
    )
    nested_tub_fixture_ready = bool(
        nested_tub_schema
        and nested_tub.get(
            "official_tub_temporal_encoder_output2_source_fixture_replay_passed"
        )
        is True
    )
    nested_tub_closed = (
        list(nested_tub.get("closed_blockers") or ())
        if nested_tub_fixture_ready
        else []
    )
    raw_source_blockers = {str(blocker) for blocker in source.get("blockers") or ()}
    source_authority_conflicting_blockers = {
        SOURCE_AUTHORITY_BLOCKER,
        FULL_REPLAY_BLOCKER,
        "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
        "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing",
    }
    source_authority_blocked_by_raw_evidence = bool(
        raw_source_blockers.intersection(source_authority_conflicting_blockers)
    )
    receiver_ready = bool(
        receiver_runtime_decode
        and frame_replay
        and consumes_output2
        and payload_bytes is not None
        and len(payload_sha256) == 64
    )
    full_tub = source.get("full_tub_source_forward_parity_proven") is True
    source_authority = bool(
        full_tub
        and not source_authority_blocked_by_raw_evidence
        and (
            replay.get("source_forward_replay_authority") is True
            or source.get("source_forward_replay_authority") is True
        )
    )
    closed = []
    if receiver_ready:
        closed.extend([RECEIVER_PAYLOAD_BLOCKER, FRAME_EXPORT_BLOCKER, OUTPUT2_BLOCKER])
    closed.extend(nested_tub_closed)
    if source_authority:
        closed.extend([SOURCE_AUTHORITY_BLOCKER, FULL_REPLAY_BLOCKER])
    receiver_blockers = []
    if not receiver_ready:
        if not receiver_runtime_decode or not frame_replay:
            receiver_blockers.extend([RECEIVER_PAYLOAD_BLOCKER, FRAME_EXPORT_BLOCKER])
        if not consumes_output2:
            receiver_blockers.append(OUTPUT2_BLOCKER)
    source_blockers = []
    if not source_authority:
        source_blockers.extend([SOURCE_AUTHORITY_BLOCKER, FULL_REPLAY_BLOCKER])
    return {
        "schema": "snerv_official_tub_lf_hf_source_forward_state.v1",
        "artifact_count": 1,
        "selected_artifact_schema": source.get("schema"),
        "selected_artifact_generated_utc": source.get("generated_utc"),
        "source_path": source.get("_source_path") or source.get("report_path"),
        "source_sha256": source.get("_source_sha256"),
        "official_checkpoint_export_binding_evidence": source.get(
            "official_checkpoint_export_binding_evidence"
        ),
        "official_trained_checkpoint_mapping_manifest": source.get(
            "official_trained_checkpoint_mapping_manifest"
        ),
        "receiver_output2_frame_replay_ready": receiver_ready,
        "tub_source_fixture_replay_ready": nested_tub_fixture_ready,
        "nested_tub_source_forward_artifact_schema": nested_tub_schema,
        "nested_tub_source_forward_artifact_path": source.get("_source_path"),
        "nested_tub_source_forward_artifact_sha256": source.get("_source_sha256"),
        "nested_tub_source_fixture_closed_blockers": _dedupe(nested_tub_closed),
        "receiver_runtime_decode_proven": receiver_runtime_decode,
        "frame_producing_official_payload_replay_proven": frame_replay,
        "receiver_frame_decode_consumes_output2": consumes_output2,
        "receiver_payload_source_forward_authority": source_authority,
        "full_tub_source_forward_replay_ready": source_authority,
        "full_tub_source_forward_parity_proven": full_tub,
        "payload_bytes": payload_bytes,
        "payload_sha256": replay.get("payload_sha256"),
        "decoded_frames_sha256": replay.get("decoded_frames_sha256"),
        "receiver_output2_frame_replay_blockers": _dedupe(receiver_blockers),
        "source_forward_authority_blockers": _dedupe(source_blockers),
        "closed_campaign_blockers": _dedupe(closed),
        "blockers": _dedupe(
            [
                *(source.get("blockers") or ()),
                *receiver_blockers,
                *source_blockers,
            ]
        ),
        **QUEUE_FALSE_AUTHORITY,
    }


def _checkpoint_state(
    checkpoint: Mapping[str, Any] | None,
    source_state: Mapping[str, Any],
) -> dict[str, Any]:
    source_binding = source_state.get("official_checkpoint_export_binding_evidence")
    source_binding = source_binding if isinstance(source_binding, Mapping) else {}
    if checkpoint is None:
        return {
            "schema": "snerv_official_tub_lf_hf_checkpoint_export_state.v1",
            "artifact_count": 0,
            "selected_artifact_schema": None,
            "source_path": None,
            "source_sha256": None,
            "official_checkpoint_export_bound": False,
            "trained_checkpoint_state_dict_mapping_ready": False,
            "tub_temporal_output2_weight_mapping_ready": False,
            "closed_campaign_blockers": [],
            "trained_checkpoint_mapping_blockers": [
                TRAINED_STATE_BLOCKER,
                TRAINED_MAPPING_BLOCKER,
                HFR_WEIGHT_MAPPING_BLOCKER,
                MFU_WEIGHT_MAPPING_BLOCKER,
            ],
            "tub_temporal_output2_mapping_blockers": [
                TUB_WEIGHT_BLOCKER,
                TUB_TEMPORAL_MAPPING_BLOCKER,
                TUB_OUTPUT2_MAPPING_BLOCKER,
            ],
            "blockers": [CHECKPOINT_REPORT_MISSING_BLOCKER, EXPORT_BLOCKER],
            **QUEUE_FALSE_AUTHORITY,
        }
    binding = checkpoint.get("official_checkpoint_export_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    native_bound = binding.get("native_checkpoint_export_bound_to_official_payload") is True
    payload_bound = binding.get("official_receiver_payload_bound") is True
    tensor_map = binding.get("official_receiver_tensor_map_verified") is True
    export_bound = bool(
        binding.get("official_export_bound") is True
        or (native_bound and payload_bound and tensor_map)
        or source_binding.get("official_export_bound") is True
    )
    mapping = binding.get("official_trained_checkpoint_mapping_manifest")
    mapping = mapping if isinstance(mapping, Mapping) else {}
    source_mapping = source_state.get("official_trained_checkpoint_mapping_manifest")
    source_mapping = source_mapping if isinstance(source_mapping, Mapping) else {}
    state_slice = (
        binding.get("official_trained_checkpoint_state_dict_slice_present") is True
        or mapping.get("official_trained_checkpoint_loaded") is True
    )
    combined_mfu_hfr_mapping = (
        binding.get("official_mfu_hfr_trained_checkpoint_weight_mapping_proven") is True
        or mapping.get("official_mfu_hfr_trained_checkpoint_weight_mapping_proven")
        is True
    )
    hfr_mapping = (
        binding.get("official_hfr_trained_checkpoint_weight_mapping_proven") is True
        or mapping.get("official_hfr_trained_checkpoint_weight_mapping_proven") is True
        or combined_mfu_hfr_mapping
    )
    mfu_mapping = (
        binding.get("official_mfu_trained_checkpoint_weight_mapping_proven") is True
        or mapping.get("official_mfu_trained_checkpoint_weight_mapping_proven") is True
        or combined_mfu_hfr_mapping
    )
    mfu_activation_bound = (
        binding.get("official_mfu_receiver_activation_payload_bound") is True
        or mapping.get("official_mfu_receiver_activation_payload_bound") is True
    )
    mfu_hfr_mapping = (
        combined_mfu_hfr_mapping
        or (hfr_mapping and mfu_mapping)
    )
    tub_mapping = (
        binding.get("official_tub_temporal_encoder_weight_mapping_proven") is True
        or mapping.get("official_tub_temporal_encoder_weight_mapping_proven") is True
    )
    tub_output2_mapping = (
        binding.get("official_tub_output2_decoder_weight_mapping_proven") is True
        or mapping.get("official_tub_output2_decoder_weight_mapping_proven") is True
    )
    mapping_verified = bool(
        binding.get("official_trained_checkpoint_state_dict_mapping_verified") is True
        or mapping.get("official_trained_checkpoint_state_dict_mapping_verified")
        is True
    )
    trained_ready = bool(state_slice and mfu_hfr_mapping and mapping_verified)
    tub_ready = bool(trained_ready and tub_mapping and tub_output2_mapping)
    closed = []
    if export_bound:
        closed.append(EXPORT_BLOCKER)
    if state_slice:
        closed.append(TRAINED_STATE_BLOCKER)
    if hfr_mapping:
        closed.append(HFR_WEIGHT_MAPPING_BLOCKER)
    if mfu_mapping:
        closed.append(MFU_WEIGHT_MAPPING_BLOCKER)
        closed.append(MFU_ACTIVATION_NOT_WEIGHT_BLOCKER)
    if mfu_hfr_mapping:
        closed.append(WEIGHT_MAPPING_BLOCKER)
    if mapping_verified:
        closed.append(TRAINED_MAPPING_BLOCKER)
    if tub_ready:
        closed.extend(
            [
                TUB_WEIGHT_BLOCKER,
                TUB_TEMPORAL_MAPPING_BLOCKER,
                TUB_OUTPUT2_MAPPING_BLOCKER,
            ]
        )
    trained_blockers = []
    if not state_slice:
        trained_blockers.append(TRAINED_STATE_BLOCKER)
    if not hfr_mapping:
        trained_blockers.append(HFR_WEIGHT_MAPPING_BLOCKER)
    if not mfu_mapping:
        trained_blockers.append(
            MFU_ACTIVATION_NOT_WEIGHT_BLOCKER
            if mfu_activation_bound
            else MFU_WEIGHT_MAPPING_BLOCKER
        )
    if not mapping_verified:
        trained_blockers.append(TRAINED_MAPPING_BLOCKER)
    tub_blockers = []
    if not tub_mapping:
        tub_blockers.extend(
            [
                TUB_WEIGHT_BLOCKER,
                TUB_TEMPORAL_MAPPING_BLOCKER,
            ]
        )
    if not tub_output2_mapping:
        tub_blockers.extend([TUB_WEIGHT_BLOCKER, TUB_OUTPUT2_MAPPING_BLOCKER])
    blockers = [
        *([] if export_bound else [EXPORT_BLOCKER]),
        *(binding.get("blockers") or ()),
        *(binding.get("preserved_blockers") or ()),
        *trained_blockers,
        *tub_blockers,
    ]
    return {
        "schema": "snerv_official_tub_lf_hf_checkpoint_export_state.v1",
        "artifact_count": 1,
        "selected_artifact_schema": checkpoint.get("schema"),
        "source_path": checkpoint.get("_source_path") or checkpoint.get("report_path"),
        "source_sha256": checkpoint.get("_source_sha256"),
        "checkpoint_epoch": checkpoint.get("checkpoint_epoch"),
        "archive_bytes": checkpoint.get("archive_bytes"),
        "archive_sha256": checkpoint.get("archive_sha256"),
        "packet_bytes": checkpoint.get("packet_bytes"),
        "packet_sha256": checkpoint.get("packet_sha256"),
        "selected_packet_status": binding.get("selected_packet_status"),
        "official_checkpoint_export_bound": export_bound,
        "native_checkpoint_export_bound_to_official_payload": native_bound,
        "official_receiver_payload_bound": payload_bound,
        "official_receiver_tensor_map_verified": tensor_map,
        "official_trained_checkpoint_state_dict_slice_present": state_slice,
        "official_trained_checkpoint_state_dict_mapping_verified": mapping_verified,
        "official_hfr_trained_checkpoint_weight_mapping_proven": hfr_mapping,
        "official_mfu_trained_checkpoint_weight_mapping_proven": mfu_mapping,
        "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": mfu_hfr_mapping,
        "official_mfu_receiver_activation_payload_bound": mfu_activation_bound,
        "official_tub_temporal_encoder_weight_mapping_proven": tub_mapping,
        "official_tub_output2_decoder_weight_mapping_proven": tub_output2_mapping,
        "official_trained_checkpoint_mapping_manifest": dict(mapping) or None,
        "official_source_trained_checkpoint_mapping_manifest": (
            dict(source_mapping) or None
        ),
        "trained_checkpoint_state_dict_mapping_ready": trained_ready,
        "tub_temporal_output2_weight_mapping_ready": tub_ready,
        "closed_campaign_blockers": _dedupe(closed),
        "trained_checkpoint_mapping_blockers": _dedupe(trained_blockers),
        "tub_temporal_output2_mapping_blockers": _dedupe(tub_blockers),
        "blockers": _dedupe(blockers),
        **QUEUE_FALSE_AUTHORITY,
    }


def _tub_state(
    tub: Mapping[str, Any] | None,
    source_state: Mapping[str, Any],
) -> dict[str, Any]:
    if tub is None:
        nested_tub_schema = source_state.get("nested_tub_source_forward_artifact_schema")
        nested_tub_ready = source_state.get("tub_source_fixture_replay_ready") is True
        if nested_tub_schema:
            return {
                "schema": "snerv_official_tub_lf_hf_tub_source_state.v1",
                "artifact_count": 1,
                "selected_artifact_schema": nested_tub_schema,
                "selected_artifact_source": "nested_in_source_forward_artifact",
                "source_path": source_state.get("nested_tub_source_forward_artifact_path"),
                "source_sha256": source_state.get("nested_tub_source_forward_artifact_sha256"),
                "fixture_source_replay_passed": nested_tub_ready,
                "full_tub_source_forward_parity_proven": bool(
                    source_state.get("full_tub_source_forward_parity_proven")
                    is True
                ),
                "closed_campaign_blockers": list(
                    source_state.get("nested_tub_source_fixture_closed_blockers")
                    or ()
                ),
                "blockers": [] if nested_tub_ready else [TUB_SOURCE_FIXTURE_BLOCKER],
                **QUEUE_FALSE_AUTHORITY,
            }
        return {
            "schema": "snerv_official_tub_lf_hf_tub_source_state.v1",
            "artifact_count": 0,
            "selected_artifact_schema": None,
            "source_path": None,
            "source_sha256": None,
            "fixture_source_replay_passed": False,
            "full_tub_source_forward_parity_proven": bool(
                source_state.get("full_tub_source_forward_parity_proven") is True
            ),
            "closed_campaign_blockers": [],
            "blockers": [],
            **QUEUE_FALSE_AUTHORITY,
        }
    fixture = (
        tub.get("official_tub_temporal_encoder_output2_source_fixture_replay_passed")
        is True
    )
    full_tub = tub.get("full_tub_source_forward_parity_proven") is True
    return {
        "schema": "snerv_official_tub_lf_hf_tub_source_state.v1",
        "artifact_count": 1,
        "selected_artifact_schema": tub.get("schema"),
        "selected_artifact_generated_utc": tub.get("generated_utc"),
        "source_path": tub.get("_source_path") or tub.get("report_path"),
        "source_sha256": tub.get("_source_sha256"),
        "fixture_source_replay_passed": fixture,
        "full_tub_source_forward_parity_proven": full_tub,
        "closed_campaign_blockers": list(tub.get("closed_blockers") or ()),
        "blockers": [],
        **QUEUE_FALSE_AUTHORITY,
    }


def _gate(
    gate_id: str,
    depends_on: Sequence[str],
    ready: bool,
    blockers: Sequence[Any],
) -> dict[str, Any]:
    clean_blockers = _dedupe(blockers)
    return {
        "schema": GATE_SCHEMA,
        "gate_id": str(gate_id),
        "depends_on": list(depends_on),
        "blocked": bool(clean_blockers) or not bool(ready),
        "status": (
            "ready_no_authority"
            if bool(ready) and not clean_blockers
            else "blocked_until_prerequisite_evidence"
        ),
        "blockers": clean_blockers if not ready or clean_blockers else [],
        **QUEUE_FALSE_AUTHORITY,
    }


def _rebuild_command(
    *,
    output_root: Path,
    source_path: Any,
    checkpoint_path: Any,
    tub_path: Any,
) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "tools/build_snerv_official_tub_lf_hf_replacement_authority_gate.py",
    ]
    if source_path:
        command.extend(["--source-forward-artifact", str(source_path)])
    if checkpoint_path:
        command.extend(["--checkpoint-export-report", str(checkpoint_path)])
    if tub_path:
        command.extend(["--tub-source-forward-artifact", str(tub_path)])
    command.extend(
        [
            "--output-root",
            output_root.as_posix(),
            "--output-json",
            (output_root / "snerv_official_tub_lf_hf_replacement_authority_gate.json").as_posix(),
            "--output-md",
            (output_root / "snerv_official_tub_lf_hf_replacement_authority_gate.md").as_posix(),
        ]
    )
    return command


def _next_unblock_command(
    *,
    output_root: Path,
    source_path: Any,
    checkpoint_path: Any,
    tub_path: Any,
) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "tools/audit_snerv_official_source_parity.py",
        "--official-repo-dir",
        "/Volumes/VertigoDataTier/pact/experiments/results/"
        "oss_nerv_source_audit_20260602T113720Z/repos/SNeRV",
    ]
    if checkpoint_path:
        command.extend(["--checkpoint-export-report", str(checkpoint_path)])
    if tub_path:
        command.extend(["--tub-source-forward-artifact", str(tub_path)])
    command.extend(
        [
            "--output-forward-parity-artifact",
            (
                output_root
                / "snerv_official_mfu_hfr_tub_forward_parity_after_mapping.json"
            ).as_posix(),
            "--output-json",
            (output_root / "snerv_official_source_parity_audit_after_mapping.json").as_posix(),
        ]
    )
    return command


def _storage_preflight(
    output_root: Path,
    *,
    min_free_bytes: int,
    allow_local_output: bool,
) -> dict[str, Any]:
    root = output_root.expanduser().resolve(strict=False)
    on_ssd = any(_is_relative_to(root, ssd_root) for ssd_root in SSD_ROOTS)
    free = shutil.disk_usage(_nearest_existing_parent(root)).free
    blockers: list[str] = []
    if not on_ssd and not allow_local_output:
        blockers.append("snerv_official_tub_lf_hf_gate_output_root_not_on_ssd_tier")
    if free < int(min_free_bytes):
        blockers.append("snerv_official_tub_lf_hf_gate_output_root_free_space_below_floor")
    if blockers:
        raise SnervOfficialTubLfHfReplacementAuthorityGateError(
            f"{root}: storage preflight blocked: {', '.join(blockers)}"
        )
    return {
        "schema": "snerv_official_tub_lf_hf_gate_storage_preflight.v1",
        "output_root": root.as_posix(),
        "ssd_tier": _ssd_tier(root),
        "free_bytes_before": int(free),
        "min_free_bytes": int(min_free_bytes),
        "allow_local_output": bool(allow_local_output),
        "blockers": [],
    }


def _positive_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _dedupe(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _shell_join(command: Sequence[Any]) -> str:
    return shlex.join(str(part) for part in command)


def _ssd_tier(path: Path) -> str:
    for root in SSD_ROOTS:
        if _is_relative_to(path, root):
            return root.as_posix()
    return "local_or_unknown"


def _nearest_existing_parent(path: Path) -> Path:
    cursor = path
    while not cursor.exists() and cursor.parent != cursor:
        cursor = cursor.parent
    return cursor


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
