# SPDX-License-Identifier: MIT
"""Executable SNeRV official-source forward replay harness.

This module is deliberately narrower than full SNeRV authority.  It loads the
pinned official source graph, assigns deterministic dyadic weights into the
real Torch ``decoder[...]`` modules, maps those official state_dict keys into
the local portable MFU/HFR primitives, and compares the source subgraph outputs.
The TUB row proves graph-input algebra plus deterministic ``output_2`` fusion;
full temporal encoder/output2 replay still needs trained temporal weights.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
import warnings
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from tac.analysis.snerv_official_primitive_replay import (
    build_snerv_official_primitive_replay_binding,
)
from tac.analysis.snerv_official_tub_source_forward_replay import (
    STATE_VALUE_ARTIFACT_BLOCKER,
    TUB_CHECKPOINT_EXPORT_LINEAGE_BLOCKER,
    build_snerv_official_tub_source_forward_replay_artifact,
)
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    decode_official_mfu_hfr_tub_decoder_payload,
    encode_official_mfu_hfr_tub_decoder_payload,
    execute_official_mfu_hfr_tub_decoder_payload,
)
from tac.substrates.snerv_inverse_steg_carrier.official_hfr import (
    OfficialConv2dNchw,
    OfficialHfrConvBlock,
    OfficialHfrHeads,
)
from tac.substrates.snerv_inverse_steg_carrier.official_mfu import (
    OfficialConvTranspose2dNchw,
    OfficialResidualBlockNoBN,
    OfficialResidualBlocksWithInputConv,
    OfficialSnervMfu,
    OfficialSnervMfuSpec,
)
from tac.substrates.snerv_inverse_steg_carrier.official_tub import (
    official_output2_fusion_numpy,
    prepare_official_tub_graph_inputs,
)

SCHEMA = "snerv_official_mfu_hfr_tub_forward_parity.v1"
SOURCE_REPLAY_SCHEMA = "snerv_official_mfu_hfr_tub_source_forward_harness.v1"
TRAINED_CHECKPOINT_MAPPING_SCHEMA = (
    "snerv_official_trained_checkpoint_state_dict_mapping_manifest.v1"
)
RECEIVER_PAYLOAD_FRAME_REPLAY_SCHEMA = (
    "snerv_official_mfu_hfr_tub_receiver_payload_frame_replay.v1"
)
OFFICIAL_SNERV_SHA = "0844a08f9591eea9625f8b961ed91d08030e06d1"
OFFICIAL_REPO_URL = "https://github.com/qwertja/SNeRV"
OFFICIAL_REPO_URL_GIT = "https://github.com/qwertja/SNeRV.git"
DEFAULT_OFFICIAL_SNERV_REPO = Path(
    "/Volumes/VertigoDataTier/pact/experiments/results/"
    "oss_nerv_source_audit_20260602T113720Z/repos/SNeRV"
)

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "production_hardened_claim": False,
    "source_faithful_stack_claim": False,
    "ready_for_exact_eval_dispatch": False,
}

OFFICIAL_MFU_HFR_TUB_SOURCE_FORWARD_BLOCKER = (
    "snerv_official_trained_checkpoint_source_forward_replay_missing"
)
OFFICIAL_MFU_HFR_TUB_WEIGHT_MAPPING_BLOCKER = (
    "snerv_official_mfu_hfr_tub_weight_mapping_missing"
)
OFFICIAL_TRAINED_CHECKPOINT_MAPPING_BLOCKER = (
    "snerv_official_trained_checkpoint_state_dict_mapping_missing"
)
SOURCE_FORWARD_AUTHORITY_RESIDUAL_BLOCKERS: frozenset[str] = frozenset(
    {
        "official_weight_tensor_mapping_not_loaded",
        "full_official_mfu_forward_artifact_not_emitted",
        "official_hfr_weight_tensor_mapping_not_loaded",
        "full_official_hfr_forward_artifact_not_emitted",
        "snerv_official_pytorch_wavelets_runtime_dependency_missing",
        STATE_VALUE_ARTIFACT_BLOCKER,
        TUB_CHECKPOINT_EXPORT_LINEAGE_BLOCKER,
    }
)


def build_snerv_official_trained_checkpoint_mapping_manifest(
    state_dict: Mapping[str, Any] | None = None,
    *,
    decoder_len: int | None = None,
    state_dict_kind: str = "official_trained_checkpoint_state_dict",
    source: str | None = None,
) -> dict[str, Any]:
    """Classify official trained checkpoint coverage without claiming parity.

    This is the executable boundary between PR95-style source-faithful export
    discipline and local receiver payloads.  It only proves that the trained
    official state_dict exposes the MFU/HFR/TUB keys a receiver adapter would
    need; actual upstream source-forward replay remains a separate blocker.
    """

    raw_state = dict(state_dict or {})
    if not raw_state:
        return {
            "schema": TRAINED_CHECKPOINT_MAPPING_SCHEMA,
            "state_dict_kind": state_dict_kind,
            "state_dict_source": source,
            "state_dict_key_count": 0,
            "decoder_len": decoder_len,
            "decoder_len_source": "provided" if decoder_len is not None else None,
            "official_trained_checkpoint_loaded": False,
            "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": False,
            "official_tub_temporal_encoder_weight_mapping_proven": False,
            "official_tub_output2_decoder_weight_mapping_proven": False,
            "state_dict_sha256": None,
            "mapped_weight_key_count": 0,
            "weight_entries": [],
            "component_rows": [],
            "blockers": [
                "snerv_official_trained_checkpoint_state_dict_not_loaded",
                OFFICIAL_MFU_HFR_TUB_SOURCE_FORWARD_BLOCKER,
            ],
            **FALSE_AUTHORITY,
        }

    inferred_decoder_len = decoder_len
    decoder_len_source = "provided"
    if inferred_decoder_len is None:
        inferred_decoder_len = _infer_official_decoder_len(raw_state)
        decoder_len_source = "inferred_from_decoder_prefixes"
    native_mapping = _native_receiver_checkpoint_mapping(raw_state)
    if inferred_decoder_len is None:
        if native_mapping["known_entry_count"]:
            return _native_receiver_checkpoint_mapping_manifest(
                raw_state,
                native_mapping=native_mapping,
                state_dict_kind=state_dict_kind,
                source=source,
            )
        return {
            "schema": TRAINED_CHECKPOINT_MAPPING_SCHEMA,
            "state_dict_kind": state_dict_kind,
            "state_dict_source": source,
            "state_dict_key_count": len(raw_state),
            "decoder_len": None,
            "decoder_len_source": None,
            "official_trained_checkpoint_loaded": True,
            "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": False,
            "official_tub_temporal_encoder_weight_mapping_proven": False,
            "official_tub_output2_decoder_weight_mapping_proven": False,
            "state_dict_sha256": _hash_state_dict_exact(raw_state),
            "mapped_weight_key_count": 0,
            "weight_entries": [],
            "component_rows": [],
            "blockers": [
                "snerv_official_trained_checkpoint_decoder_len_not_resolved",
                OFFICIAL_MFU_HFR_TUB_SOURCE_FORWARD_BLOCKER,
            ],
            **FALSE_AUTHORITY,
        }

    groups = _official_checkpoint_group_prefixes(int(inferred_decoder_len))
    entries: list[dict[str, Any]] = []
    for key in sorted(raw_state):
        group = _official_group_for_key(key, groups)
        if group is None:
            continue
        array = _state_value_array(raw_state[key])
        entries.append(
            {
                "key": key,
                "receiver_key": _receiver_key_for_official_key(
                    key,
                    int(inferred_decoder_len),
                ),
                "component_id": _component_for_group(group),
                "official_group": group,
                "shape": [int(value) for value in array.shape],
                "dtype": str(array.dtype),
                "sha256": _hash_array_exact(array),
                "f64_sha256": _hash_array(array),
                "byte_count": int(np.ascontiguousarray(array).nbytes),
            }
        )
    present_groups = {str(row["official_group"]) for row in entries}
    component_rows = [
        _checkpoint_mapping_component_row(
            component_id="hfr",
            required_groups=("hfr_lh", "hfr_hl", "hfr_hh"),
            present_groups=present_groups,
            entries=entries,
            source_blocker="snerv_hfr_source_forward_replay_requires_upstream_torch_state_dict_mapping",
        ),
        _checkpoint_mapping_component_row(
            component_id="mfu",
            required_groups=(
                "mfu_upsample_mid",
                "mfu_rb_mid",
                "mfu_upsample_high",
                "mfu_rb_high",
            ),
            present_groups=present_groups,
            entries=entries,
            source_blocker="snerv_mfu_source_forward_replay_requires_upstream_torch_state_dict_mapping",
        ),
        _checkpoint_mapping_component_row(
            component_id="tub",
            required_groups=(
                "tub_temporal_encoder_1",
                "tub_temporal_encoder_2",
                "tub_output2_decoder",
            ),
            present_groups=present_groups,
            entries=entries,
            source_blocker="snerv_tub_full_source_forward_replay_requires_temporal_encoder_decoder_fusion_mapping",
        ),
    ]
    mfu_hfr_proven = all(
        row["trained_checkpoint_weight_mapping_proven"] is True
        for row in component_rows
        if row["component_id"] in {"mfu", "hfr"}
    )
    hfr_proven = bool(
        next(row for row in component_rows if row["component_id"] == "hfr")[
            "trained_checkpoint_weight_mapping_proven"
        ]
    )
    mfu_proven = bool(
        next(row for row in component_rows if row["component_id"] == "mfu")[
            "trained_checkpoint_weight_mapping_proven"
        ]
    )
    tub_temporal_proven = all(
        group in present_groups
        for group in ("tub_temporal_encoder_1", "tub_temporal_encoder_2")
    )
    tub_output2_proven = "tub_output2_decoder" in present_groups
    blockers = [
        blocker
        for row in component_rows
        for blocker in row.get("blockers", ())
    ]
    blockers.append(OFFICIAL_MFU_HFR_TUB_SOURCE_FORWARD_BLOCKER)
    return {
        "schema": TRAINED_CHECKPOINT_MAPPING_SCHEMA,
        "state_dict_kind": state_dict_kind,
        "state_dict_source": source,
        "state_dict_key_count": len(raw_state),
        "decoder_len": int(inferred_decoder_len),
        "decoder_len_source": decoder_len_source,
        "state_dict_mapping_dialect": "upstream_official_decoder_state_dict",
        "official_trained_checkpoint_loaded": True,
        "official_hfr_trained_checkpoint_weight_mapping_proven": hfr_proven,
        "official_mfu_trained_checkpoint_weight_mapping_proven": mfu_proven,
        "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": mfu_hfr_proven,
        "official_tub_temporal_encoder_weight_mapping_proven": tub_temporal_proven,
        "official_tub_output2_decoder_weight_mapping_proven": tub_output2_proven,
        "official_mfu_receiver_activation_payload_bound": False,
        "official_tub_receiver_activation_payload_bound": False,
        "official_native_receiver_state_mapping_proven": False,
        "state_dict_sha256": _hash_state_dict_exact(raw_state),
        "mapped_weight_key_count": len(entries),
        "mapped_weight_byte_count": int(sum(int(row["byte_count"]) for row in entries)),
        "mapped_weight_entries_sha256": _hash_weight_entries(entries),
        "weight_entries": entries,
        "activation_entries": [],
        "mapped_activation_key_count": 0,
        "component_rows": component_rows,
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


@dataclass(frozen=True)
class _OfficialFixture:
    model: Any
    decoder_len: int
    spec: OfficialSnervMfuSpec
    state_dict: Mapping[str, Any]
    selected_weight_keys: tuple[str, ...]
    mfu_weight_keys: tuple[str, ...]
    hfr_weight_keys: tuple[str, ...]


def build_snerv_official_source_forward_harness_artifact(
    *,
    official_repo_dir: str | Path = DEFAULT_OFFICIAL_SNERV_REPO,
    repo_root: str | Path,
    checkpoint_export_reports: Sequence[Mapping[str, Any]] = (),
    trained_checkpoint_mapping_manifests: Sequence[Mapping[str, Any]] = (),
    tub_source_forward_artifact: Mapping[str, Any] | None = None,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Return a false-authority source-forward replay artifact.

    The artifact closes the executable MFU/HFR source-fixture mapping question
    and preserves the exact remaining full-stack blockers.  It does not load a
    trained official checkpoint and it does not claim score authority.
    """

    official_root = Path(official_repo_dir)
    local_root = Path(repo_root)
    if generated_utc is None:
        generated_utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    primitive_binding = build_snerv_official_primitive_replay_binding(
        repo_root=local_root,
    )
    receiver_runtime = primitive_binding["official_receiver_runtime_decode_contract"]
    local_adapter_row = _local_receiver_adapter_source_gap(local_root)
    checkpoint_export_binding = _checkpoint_export_binding_evidence(
        checkpoint_export_reports
    )
    export_mapping_manifest = checkpoint_export_binding.get(
        "official_trained_checkpoint_mapping_manifest"
    )
    trained_mapping_inputs: tuple[Mapping[str, Any], ...] = tuple(
        trained_checkpoint_mapping_manifests
    )
    if (
        isinstance(export_mapping_manifest, Mapping)
        and export_mapping_manifest.get("schema") == TRAINED_CHECKPOINT_MAPPING_SCHEMA
    ):
        trained_mapping_inputs = (*trained_mapping_inputs, export_mapping_manifest)
    tub_source_replay = _tub_source_forward_replay_evidence(
        official_root,
        tub_source_forward_artifact,
    )
    tub_mapping_manifest = tub_source_replay.get(
        "official_trained_checkpoint_mapping_manifest"
    )
    if (
        isinstance(tub_mapping_manifest, Mapping)
        and tub_mapping_manifest.get("schema") == TRAINED_CHECKPOINT_MAPPING_SCHEMA
        and tub_mapping_manifest.get("official_trained_checkpoint_loaded") is True
    ):
        trained_mapping_inputs = (*trained_mapping_inputs, tub_mapping_manifest)
    trained_checkpoint_mapping = _trained_checkpoint_mapping_evidence(
        trained_mapping_inputs
    )

    component_rows: list[dict[str, Any]]
    source_replay: dict[str, Any]
    weight_manifest: dict[str, Any]
    receiver_frame_replay: dict[str, Any]
    harness_blockers: list[str] = []
    try:
        fixture = _build_official_fixture(official_root)
        (
            mfu_row,
            hfr_row,
            source_replay,
            weight_manifest,
            receiver_frame_replay,
        ) = _run_mfu_hfr_replay(fixture)
        component_rows = [
            mfu_row,
            hfr_row,
            _tub_component_row_from_source_replay(tub_source_replay),
        ]
    except Exception as exc:  # pragma: no cover - exercised by fail-closed callers.
        harness_blockers.append(f"snerv_official_source_harness_failed:{type(exc).__name__}")
        component_rows = [
            _failed_component_row("mfu", exc),
            _failed_component_row("hfr", exc),
            _tub_component_row_from_source_replay(tub_source_replay),
        ]
        source_replay = {
            "schema": SOURCE_REPLAY_SCHEMA,
            "backend": "official_torch_vs_portable",
            "replay_ran": False,
            "input_bundle_sha256": None,
            "blockers": list(harness_blockers),
        }
        weight_manifest = {
            "schema": "snerv_official_state_dict_mapping_manifest.v1",
            "state_dict_kind": "unavailable_due_to_harness_failure",
            "state_dict_sha256": None,
            "state_dict_key_count": 0,
            "weight_entries": [],
            "official_trained_checkpoint_loaded": False,
            "blockers": list(harness_blockers),
        }
        receiver_frame_replay = _failed_receiver_payload_frame_replay(
            harness_blockers
        )

    mfu_hfr_passed = all(
        row["component_id"] in {"mfu", "hfr"}
        and row.get("source_forward_parity_proven") is True
        for row in component_rows
        if row["component_id"] in {"mfu", "hfr"}
    )
    tub_row = next(
        (
            row
            for row in component_rows
            if row.get("component_id") == "tub"
        ),
        {},
    )
    full_tub_source_forward_parity_proven = (
        tub_row.get("full_tub_source_forward_parity_proven") is True
    )
    tub_source_fixture_forward_parity_proven = (
        tub_row.get("source_fixture_forward_parity_proven") is True
        or tub_row.get("primitive_source_forward_parity_proven") is True
    )
    full_passed = bool(
        mfu_hfr_passed
        and all(row.get("source_forward_parity_proven") is True for row in component_rows)
        and receiver_runtime.get("receiver_runtime_decode_proven") is True
    )
    receiver_ready = bool(
        receiver_frame_replay.get("receiver_runtime_decode_proven") is True
        and receiver_frame_replay.get("frame_producing_official_payload_replay_proven")
        is True
        and receiver_frame_replay.get("receiver_frame_decode_consumes_output2") is True
    )
    state_dict_value_artifact_ready = _tub_state_dict_value_artifact_ready(
        tub_source_replay
    )
    source_forward_mapping_ready = bool(
        trained_checkpoint_mapping.get(
            "official_mfu_hfr_trained_checkpoint_weight_mapping_proven"
        )
        is True
        and trained_checkpoint_mapping.get(
            "official_tub_temporal_encoder_weight_mapping_proven"
        )
        is True
        and trained_checkpoint_mapping.get(
            "official_tub_output2_decoder_weight_mapping_proven"
        )
        is True
    )
    source_forward_replay_verified = bool(
        full_passed
        and receiver_ready
        and source_forward_mapping_ready
    )
    source_forward_authority_candidate = bool(
        source_forward_replay_verified and state_dict_value_artifact_ready
    )
    if (
        source_forward_replay_verified
        and not state_dict_value_artifact_ready
    ):
        harness_blockers.append(STATE_VALUE_ARTIFACT_BLOCKER)
    closed_by_trained_checkpoint = set(
        trained_checkpoint_mapping.get("closed_campaign_blockers") or ()
    )
    closed_by_source_replay = {
        OFFICIAL_MFU_HFR_TUB_SOURCE_FORWARD_BLOCKER,
        "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
        "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing",
        "snerv_official_snerv_t_full_tub_source_forward_replay_missing",
        "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing",
        "snerv_official_tub_normalized_lf_graph_inputs_not_full_source_forward_parity",
        "snerv_hfr_source_forward_replay_requires_upstream_torch_state_dict_mapping",
        "snerv_mfu_source_forward_replay_requires_upstream_torch_state_dict_mapping",
        "snerv_tub_full_source_forward_replay_requires_temporal_encoder_decoder_fusion_mapping",
    } if source_forward_replay_verified else set()
    closed_by_source_authority: set[str] = set()
    closed_by_source_forward = closed_by_source_replay | closed_by_source_authority
    if source_forward_replay_verified:
        nested_closed_blockers = closed_by_trained_checkpoint | closed_by_source_forward
        source_replay = {
            **source_replay,
            "full_stack_source_forward_parity_proven": True,
            "source_forward_replay_verified": True,
            "source_forward_replay_authority": False,
            "blockers": [
                blocker
                for blocker in source_replay.get("blockers") or ()
                if str(blocker) not in nested_closed_blockers
            ],
        }
        receiver_frame_replay = {
            **receiver_frame_replay,
            "source_forward_replay_bound": True,
            "source_forward_replay_verified": True,
            "source_forward_replay_authority": False,
            "blockers": [
                blocker
                for blocker in receiver_frame_replay.get("blockers") or ()
                if str(blocker) not in nested_closed_blockers
            ],
        }
    if closed_by_trained_checkpoint or closed_by_source_forward:
        component_rows = _component_rows_with_closed_blockers_applied(
            component_rows,
            closed_by_trained_checkpoint | closed_by_source_forward,
        )
    blockers = _ordered_unique(
        [
            *harness_blockers,
            *checkpoint_export_binding.get("blockers", ()),
            *trained_checkpoint_mapping.get("blockers", ()),
            *weight_manifest.get("blockers", ()),
            *source_replay.get("blockers", ()),
            *receiver_frame_replay.get("blockers", ()),
            *[
                blocker
                for row in component_rows
                for blocker in row.get("blockers", ())
            ],
        ]
    )
    blockers = [
        blocker
        for blocker in blockers
        if str(blocker) not in closed_by_trained_checkpoint
        and str(blocker) not in closed_by_source_forward
    ]
    source_forward_authority_residual_blockers = (
        _source_forward_authority_residual_blockers(blockers)
    )
    source_forward_authority = bool(
        source_forward_authority_candidate
        and not source_forward_authority_residual_blockers
    )
    if source_forward_authority:
        closed_by_source_authority = {
            "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority",
        }
        blockers = [
            blocker
            for blocker in blockers
            if str(blocker) not in closed_by_source_authority
        ]
    if source_forward_replay_verified:
        source_replay = {
            **source_replay,
            "source_forward_replay_authority": source_forward_authority,
            "source_forward_authority_residual_blockers": (
                source_forward_authority_residual_blockers
            ),
        }
        receiver_frame_replay = {
            **receiver_frame_replay,
            "source_forward_replay_authority": source_forward_authority,
            "source_forward_authority_residual_blockers": (
                source_forward_authority_residual_blockers
            ),
        }

    return {
        "schema": SCHEMA,
        "authority": "false_authority_source_forward_harness_no_score_claim",
        "generated_utc": generated_utc,
        "family": "snerv",
        "official_repo": {
            "repo_url": OFFICIAL_REPO_URL,
            "repo_url_git": OFFICIAL_REPO_URL_GIT,
            "root": official_root.as_posix(),
            "head_sha": _git_head_sha(official_root),
            "expected_head_sha": OFFICIAL_SNERV_SHA,
        },
        "local_repo_root": local_root.as_posix(),
        "official_weight_manifest": weight_manifest,
        "official_checkpoint_export_binding_evidence": checkpoint_export_binding,
        "official_trained_checkpoint_mapping_manifest": trained_checkpoint_mapping,
        "official_tub_source_forward_replay": tub_source_replay,
        "source_forward_training_smoke": tub_source_replay.get(
            "source_forward_training_smoke"
        ),
        "source_forward_replay": source_replay,
        "receiver_payload_frame_replay": receiver_frame_replay,
        "component_rows": component_rows,
        "local_receiver_adapter_source_gap": local_adapter_row,
        "official_mfu_hfr_source_fixture_forward_parity_passed": mfu_hfr_passed,
        "official_mfu_hfr_tub_forward_parity_passed": full_passed,
        "official_mfu_hfr_tub_forward_parity_falsified": False,
        "official_trained_checkpoint_loaded": (
            trained_checkpoint_mapping.get("official_trained_checkpoint_loaded")
            is True
        ),
        "official_hfr_trained_checkpoint_weight_mapping_proven": (
            trained_checkpoint_mapping.get(
                "official_hfr_trained_checkpoint_weight_mapping_proven"
            )
            is True
        ),
        "official_mfu_trained_checkpoint_weight_mapping_proven": (
            trained_checkpoint_mapping.get(
                "official_mfu_trained_checkpoint_weight_mapping_proven"
            )
            is True
        ),
        "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": (
            trained_checkpoint_mapping.get(
                "official_mfu_hfr_trained_checkpoint_weight_mapping_proven"
            )
            is True
        ),
        "official_mfu_receiver_activation_payload_bound": (
            trained_checkpoint_mapping.get(
                "official_mfu_receiver_activation_payload_bound"
            )
            is True
        ),
        "official_tub_receiver_activation_payload_bound": (
            trained_checkpoint_mapping.get(
                "official_tub_receiver_activation_payload_bound"
            )
            is True
        ),
        "official_native_receiver_state_mapping_proven": (
            trained_checkpoint_mapping.get(
                "official_native_receiver_state_mapping_proven"
            )
            is True
        ),
        "official_tub_temporal_encoder_weight_mapping_proven": (
            trained_checkpoint_mapping.get(
                "official_tub_temporal_encoder_weight_mapping_proven"
            )
            is True
        ),
        "official_tub_output2_decoder_weight_mapping_proven": (
            trained_checkpoint_mapping.get(
                "official_tub_output2_decoder_weight_mapping_proven"
            )
            is True
        ),
        "official_trained_checkpoint_state_dict_mapping_verified": bool(
            trained_checkpoint_mapping.get(
                "official_mfu_hfr_trained_checkpoint_weight_mapping_proven"
            )
            is True
            and trained_checkpoint_mapping.get(
                "official_tub_temporal_encoder_weight_mapping_proven"
            )
            is True
            and trained_checkpoint_mapping.get(
                "official_tub_output2_decoder_weight_mapping_proven"
            )
            is True
        ),
        "official_export_bound": (
            checkpoint_export_binding.get("official_export_bound") is True
        ),
        "official_mfu_hfr_weight_mapping_source_fixture_proven": mfu_hfr_passed,
        "official_tub_source_fixture_forward_parity_proven": (
            tub_source_fixture_forward_parity_proven
        ),
        "tub_source_fixture_closed_blockers": list(
            tub_row.get("closed_blockers") or ()
        ),
        "full_tub_source_forward_parity_proven": (
            full_tub_source_forward_parity_proven
        ),
        "source_forward_replay_verified": source_forward_replay_verified,
        "source_forward_replay_authority": source_forward_authority,
        "official_trained_checkpoint_state_dict_value_artifact_ready": (
            state_dict_value_artifact_ready
        ),
        "source_forward_replay_closed_blockers": _ordered_unique(
            sorted(closed_by_source_replay)
        ),
        "source_forward_authority_closed_blockers": _ordered_unique(
            sorted(closed_by_source_authority)
        ),
        "source_forward_authority_residual_blockers": (
            source_forward_authority_residual_blockers
        ),
        "official_mfu_hfr_tub_primitive_replay_binding": primitive_binding,
        "receiver_runtime_decode": receiver_runtime,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _component_rows_with_closed_blockers_applied(
    rows: Sequence[Mapping[str, Any]],
    closed_blockers: set[str],
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        for key in ("blockers", "preserved_blockers"):
            if key not in item:
                continue
            item[key] = [
                blocker
                for blocker in item.get(key) or ()
                if str(blocker) not in closed_blockers
            ]
        out.append(item)
    return out


def _source_forward_authority_residual_blockers(
    blockers: Iterable[Any],
) -> list[str]:
    """Return blockers that keep verified replay from becoming source authority."""

    return _ordered_unique(
        [
            str(blocker)
            for blocker in blockers
            if str(blocker) in SOURCE_FORWARD_AUTHORITY_RESIDUAL_BLOCKERS
        ]
    )


def _checkpoint_export_binding_evidence(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    valid_reports = [
        report
        for report in reports
        if isinstance(report, Mapping)
        and report.get("schema") == "snerv_checkpoint_archive_export.v1"
        and isinstance(report.get("official_checkpoint_export_binding"), Mapping)
    ]
    if not valid_reports:
        return {
            "schema": "snerv_official_checkpoint_export_binding_evidence.v1",
            "artifact_count": 0,
            "selected_artifact_schema": None,
            "selected_artifact_path": None,
            "selected_artifact_sha256": None,
            "official_export_bound": False,
            "official_receiver_payload_bound": False,
            "official_receiver_tensor_map_verified": False,
            "native_checkpoint_export_bound_to_official_payload": False,
            "official_trained_checkpoint_state_dict_slice_present": False,
            "official_trained_checkpoint_state_dict_mapping_verified": False,
            "closed_campaign_blockers": [],
            "blockers": ["snerv_official_checkpoint_export_report_missing"],
            **FALSE_AUTHORITY,
        }
    selected = max(
        valid_reports,
        key=lambda report: (
            int(report.get("checkpoint_epoch") or -1),
            str(report.get("_source_path") or report.get("report_path") or ""),
        ),
    )
    binding = selected.get("official_checkpoint_export_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    native_bound = binding.get("native_checkpoint_export_bound_to_official_payload") is True
    payload_bound = binding.get("official_receiver_payload_bound") is True
    tensor_map = binding.get("official_receiver_tensor_map_verified") is True
    export_bound = bool(native_bound and payload_bound and tensor_map)
    blockers = [
        str(blocker)
        for blocker in (
            [
                *([] if export_bound else ["snerv_official_mfu_hfr_tub_export_not_bound"]),
                *(binding.get("blockers") or ()),
                *(binding.get("preserved_blockers") or ()),
            ]
        )
        if str(blocker)
    ]
    return {
        "schema": "snerv_official_checkpoint_export_binding_evidence.v1",
        "artifact_count": len(valid_reports),
        "selected_artifact_schema": selected.get("schema"),
        "selected_artifact_path": selected.get("_source_path") or selected.get("report_path"),
        "selected_artifact_sha256": selected.get("_source_sha256"),
        "checkpoint_epoch": selected.get("checkpoint_epoch"),
        "archive_bytes": selected.get("archive_bytes"),
        "archive_sha256": selected.get("archive_sha256"),
        "packet_bytes": selected.get("packet_bytes"),
        "packet_sha256": selected.get("packet_sha256"),
        "selected_packet_status": binding.get("selected_packet_status"),
        "official_export_bound": export_bound,
        "official_receiver_payload_bound": payload_bound,
        "official_receiver_tensor_map_verified": tensor_map,
        "native_checkpoint_export_bound_to_official_payload": native_bound,
        "official_trained_checkpoint_state_dict_slice_present": (
            binding.get("official_trained_checkpoint_state_dict_slice_present") is True
        ),
        "official_trained_checkpoint_state_dict_mapping_verified": (
            binding.get("official_trained_checkpoint_state_dict_mapping_verified")
            is True
        ),
        "official_trained_checkpoint_mapping_manifest": binding.get(
            "official_trained_checkpoint_mapping_manifest"
        ),
        "closed_campaign_blockers": (
            ["snerv_official_mfu_hfr_tub_export_not_bound"] if export_bound else []
        ),
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _trained_checkpoint_mapping_evidence(
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    valid = [
        manifest
        for manifest in manifests
        if isinstance(manifest, Mapping)
        and manifest.get("schema") == TRAINED_CHECKPOINT_MAPPING_SCHEMA
    ]
    if not valid:
        return {
            "schema": TRAINED_CHECKPOINT_MAPPING_SCHEMA,
            "artifact_count": 0,
            "state_dict_kind": None,
            "state_dict_source": None,
            "state_dict_key_count": 0,
            "official_trained_checkpoint_loaded": False,
            "official_hfr_trained_checkpoint_weight_mapping_proven": False,
            "official_mfu_trained_checkpoint_weight_mapping_proven": False,
            "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": False,
            "official_mfu_receiver_activation_payload_bound": False,
            "official_tub_receiver_activation_payload_bound": False,
            "official_native_receiver_state_mapping_proven": False,
            "official_tub_temporal_encoder_weight_mapping_proven": False,
            "official_tub_output2_decoder_weight_mapping_proven": False,
            "closed_campaign_blockers": [],
            "blockers": [],
            **FALSE_AUTHORITY,
        }
    selected = max(
        valid,
        key=lambda manifest: (
            bool(manifest.get("official_mfu_hfr_trained_checkpoint_weight_mapping_proven")),
            bool(manifest.get("official_tub_temporal_encoder_weight_mapping_proven")),
            int(manifest.get("mapped_weight_key_count") or 0),
            int(manifest.get("state_dict_key_count") or 0),
            str(manifest.get("state_dict_source") or ""),
        ),
    )
    out = dict(selected)
    out["artifact_count"] = len(valid)
    closed = []
    if out.get("official_trained_checkpoint_loaded") is True:
        closed.append("snerv_official_trained_checkpoint_state_dict_not_loaded")
    if out.get("official_hfr_trained_checkpoint_weight_mapping_proven") is True:
        closed.append("snerv_official_trained_checkpoint_hfr_weight_mapping_incomplete")
    if out.get("official_mfu_trained_checkpoint_weight_mapping_proven") is True:
        closed.append("snerv_official_trained_checkpoint_mfu_weight_mapping_incomplete")
        closed.append(
            "snerv_official_mfu_native_receiver_activation_payload_not_upstream_weight_mapping"
        )
    if out.get("official_mfu_hfr_trained_checkpoint_weight_mapping_proven") is True:
        closed.append(OFFICIAL_MFU_HFR_TUB_WEIGHT_MAPPING_BLOCKER)
    if out.get("official_tub_temporal_encoder_weight_mapping_proven") is True:
        closed.extend(
            [
                "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
                "snerv_official_tub_encoder_decoder_weights_not_loaded",
                "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
            ]
        )
    if out.get("official_tub_output2_decoder_weight_mapping_proven") is True:
        closed.append(
            "snerv_official_tub_portable_output2_decoder_weight_mapping_missing"
        )
    if (
        out.get("official_mfu_hfr_trained_checkpoint_weight_mapping_proven") is True
        and out.get("official_tub_temporal_encoder_weight_mapping_proven") is True
        and out.get("official_tub_output2_decoder_weight_mapping_proven") is True
    ):
        closed.append(OFFICIAL_TRAINED_CHECKPOINT_MAPPING_BLOCKER)
    out["closed_campaign_blockers"] = _ordered_unique(
        [*(out.get("closed_campaign_blockers") or ()), *closed]
    )
    out.setdefault("score_claim", False)
    out.setdefault("promotion_eligible", False)
    out.setdefault("rank_or_kill_eligible", False)
    out.setdefault("production_hardened_claim", False)
    out.setdefault("source_faithful_stack_claim", False)
    out.setdefault("ready_for_exact_eval_dispatch", False)
    return out


def _tub_source_forward_replay_evidence(
    official_root: Path,
    artifact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(artifact, Mapping) and artifact.get(
        "schema"
    ) == "snerv_official_tub_source_forward_replay.v1":
        return dict(artifact)
    return build_snerv_official_tub_source_forward_replay_artifact(
        official_repo_dir=official_root,
        train_one_step=True,
    )


def _tub_state_dict_value_artifact_ready(artifact: Mapping[str, Any]) -> bool:
    path = str(
        artifact.get("official_trained_checkpoint_state_dict_slice_path")
        or artifact.get("official_trained_checkpoint_state_dict_path")
        or ""
    ).strip()
    if not path:
        return False
    try:
        bytes_value = int(
            artifact.get("official_trained_checkpoint_state_dict_slice_bytes") or 0
        )
        member_count = int(
            artifact.get("official_trained_checkpoint_state_dict_slice_member_count")
            or 0
        )
    except (TypeError, ValueError):
        return False
    sha256_value = str(
        artifact.get("official_trained_checkpoint_state_dict_slice_sha256") or ""
    ).strip()
    claimed_names = artifact.get(
        "official_trained_checkpoint_state_dict_slice_member_names"
    )
    if isinstance(claimed_names, (str, bytes)) or not isinstance(
        claimed_names,
        Sequence,
    ):
        return False
    path_obj = Path(path)
    if not (
        artifact.get("official_trained_checkpoint_state_dict_value_artifact_ready")
        is True
        and artifact.get("official_trained_checkpoint_state_dict_slice_file_present")
        is True
        and bytes_value > 0
        and member_count > 0
        and len(sha256_value) == 64
        and path_obj.is_file()
    ):
        return False
    try:
        data = path_obj.read_bytes()
        actual_size = int(path_obj.stat().st_size)
        with zipfile.ZipFile(path_obj, "r") as zf:
            actual_names = sorted(zf.namelist())
    except (OSError, zipfile.BadZipFile):
        return False
    expected_names = sorted(str(name) for name in claimed_names)
    return bool(
        actual_size == bytes_value
        and _hash_bytes(data) == sha256_value
        and len(actual_names) == member_count
        and actual_names == expected_names
        and all(name.endswith(".npy") for name in actual_names)
    )


def _tub_component_row_from_source_replay(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    executed = artifact.get("source_forward_replay_executed") is True
    fixture_passed = (
        artifact.get("official_tub_temporal_encoder_output2_source_fixture_replay_passed")
        is True
    )
    full_tub_parity = artifact.get("full_tub_source_forward_parity_proven") is True
    source_forward_parity = bool(
        full_tub_parity and artifact.get("source_forward_parity_proven") is True
    )
    blockers = _ordered_unique(
        [
            *([] if executed and fixture_passed else ["snerv_official_tub_source_fixture_replay_missing"]),
            *(artifact.get("preserved_blockers") or ()),
            *(artifact.get("blockers") or ()),
        ]
    )
    return {
        "schema": "snerv_official_source_forward_component_replay.v1",
        "component_id": "tub",
        "classification": (
            "official_tub_trained_full_source_forward_parity_proven"
            if source_forward_parity
            else
            "official_tub_temporal_encoder_output2_source_fixture_proven_full_tub_blocked"
            if fixture_passed
            else "official_tub_temporal_encoder_output2_source_fixture_blocked"
        ),
        "backend": "official_snerv_t_source_fixture_vs_portable_numpy_tub",
        "source_forward_parity_proven": source_forward_parity,
        "source_fixture_forward_parity_proven": bool(fixture_passed),
        "primitive_source_forward_parity_proven": bool(fixture_passed),
        "official_tub_temporal_encoder_output2_source_fixture_replay_passed": (
            fixture_passed
        ),
        "portable_output2_fusion_receiver_mapping_proven": bool(
            _nested_bool(
                artifact,
                ("portable_output2_fusion", "portable_output2_fusion_receiver_mapping_proven"),
            )
        ),
        "source_forward_parity_falsified": False,
        "full_stack_source_forward_parity_proven": full_tub_parity,
        "full_tub_source_forward_parity_proven": full_tub_parity,
        "tolerance": 0.0,
        "max_abs_error": _max_nested_abs_error(artifact),
        "graph_input_max_abs_error": _nested_float(
            artifact,
            ("graph_input_parity", "max_abs_error"),
        ),
        "output2_fusion_max_abs_error": _nested_float(
            artifact,
            ("portable_output2_fusion", "max_abs_error"),
        ),
        "official_output_sha256": _nested_value(
            artifact,
            ("full_forward_equivalence", "official_forward_sha256"),
        ),
        "portable_output_sha256": _nested_value(
            artifact,
            ("full_forward_equivalence", "manual_replay_sha256"),
        ),
        "output_hashes_bit_identical": _nested_bool(
            artifact,
            ("full_forward_equivalence", "output_hashes_bit_identical"),
        ),
        "output_shapes": _nested_value(
            artifact,
            ("temporal_path", "output_shapes"),
        ),
        "closed_blockers": list(artifact.get("closed_blockers") or ()),
        "preserved_blockers": list(artifact.get("preserved_blockers") or ()),
        "official_weight_keys": [
            "fixture_source:model/snerv_t.py:temporal_encoder_output2_path",
            "unmapped_temporal_encoder:self.encoder[1]",
            "unmapped_temporal_encoder:self.encoder[2]",
            "unmapped_output2_decoder:self.decoder[self.decoder_len-1]",
        ],
        "official_source_contract": "model/snerv_t.py temporal encoder + output_2",
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _nested_value(source: Mapping[str, Any], keys: Sequence[str]) -> Any:
    current: Any = source
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _nested_bool(source: Mapping[str, Any], keys: Sequence[str]) -> bool:
    return _nested_value(source, keys) is True


def _nested_float(source: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    value = _nested_value(source, keys)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _max_nested_abs_error(artifact: Mapping[str, Any]) -> float | None:
    values = [
        _nested_float(artifact, ("graph_input_parity", "max_abs_error")),
        _nested_float(artifact, ("portable_output2_fusion", "max_abs_error")),
        _nested_float(artifact, ("full_forward_equivalence", "max_abs_error")),
    ]
    finite = [float(value) for value in values if value is not None]
    return max(finite) if finite else None


def _build_official_fixture(official_root: Path) -> _OfficialFixture:
    import torch

    with _official_source_import_context(official_root):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            snerv_mod = importlib.import_module("model.snerv")
        args = SimpleNamespace(
            embed="pe_1_1",
            ks="3_3_3",
            num_blks="1_1",
            enc_strds=[],
            enc_dim="4_4",
            conv_type=["conv"],
            norm="none",
            act="relu",
            dec_strds=[2, 2, 2],
            fc_dim=8,
            fc_hw="1_1",
            reduce=-1,
            lower_width=2,
            num_blocks=1,
            out_bias="none",
        )
        model = snerv_mod.SNeRV(args).double().eval()
    decoder_len = int(model.decoder_len)
    selected = _selected_decoder_weight_keys(model.state_dict(), decoder_len)
    with torch.no_grad():
        _assign_sparse_source_fixture_weights(model, decoder_len)
    state_dict = model.state_dict()
    spec = OfficialSnervMfuSpec(
        low_channels=int(model.decoder[decoder_len + 3].in_channels),
        mid_channels=int(model.decoder[decoder_len + 5].in_channels),
        high_channels=int(model.decoder[decoder_len + 6].main[0].out_channels),
        mid_stride=int(model.decoder[decoder_len + 3].stride[0]),
        high_stride=int(model.decoder[decoder_len + 5].stride[0]),
        num_blocks=len(model.decoder[decoder_len + 4].main[1]),
    )
    mfu_keys = tuple(
        key
        for key in selected
        if key.startswith(
            (
                f"decoder.{decoder_len + 3}.",
                f"decoder.{decoder_len + 4}.",
                f"decoder.{decoder_len + 5}.",
                f"decoder.{decoder_len + 6}.",
            )
        )
    )
    hfr_keys = tuple(
        key
        for key in selected
        if key.startswith(
            (
                f"decoder.{decoder_len}.",
                f"decoder.{decoder_len + 1}.",
                f"decoder.{decoder_len + 2}.",
            )
        )
    )
    return _OfficialFixture(
        model=model,
        decoder_len=decoder_len,
        spec=spec,
        state_dict=state_dict,
        selected_weight_keys=tuple(selected),
        mfu_weight_keys=mfu_keys,
        hfr_weight_keys=hfr_keys,
    )


def _run_mfu_hfr_replay(
    fixture: _OfficialFixture,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch

    mfu = _portable_mfu_from_state_dict(fixture)
    heads = _portable_hfr_from_state_dict(fixture)
    low = _positive_fixture((2, fixture.spec.low_channels, 2, 3), modulo=7)
    skip_mid = _positive_fixture((2, fixture.spec.mid_channels, 4, 6), modulo=11)
    skip_high = _positive_fixture((2, fixture.spec.high_channels, 8, 12), modulo=13)

    dl = fixture.decoder_len
    with torch.no_grad():
        low_t = torch.from_numpy(low)
        skip_mid_t = torch.from_numpy(skip_mid)
        skip_high_t = torch.from_numpy(skip_high)
        up1 = fixture.model.decoder[dl + 3](low_t)
        unet1 = fixture.model.decoder[dl + 4](
            torch.cat([up1, skip_mid_t], dim=1)
        )
        unet1_up = fixture.model.decoder[dl + 5](unet1)
        official_pyr = fixture.model.decoder[dl + 6](
            torch.cat([unet1_up, skip_high_t], dim=1)
        )
        official_yh = torch.stack(
            [
                fixture.model.decoder[dl](official_pyr),
                fixture.model.decoder[dl + 1](official_pyr),
                fixture.model.decoder[dl + 2](official_pyr),
            ],
            dim=2,
        )
    portable_mfu = mfu.forward(low, skip_mid, skip_high)
    portable_hfr = heads.forward(portable_mfu.pyr_out)
    official_mfu_output = np.asarray(official_pyr.detach().cpu().numpy())
    official_hfr_output = np.asarray(official_yh.detach().cpu().numpy())

    mfu_row = _component_row(
        component_id="mfu",
        classification="official_source_fixture_mfu_state_dict_mapping_proven",
        backend="official_snerv_torch_decoder_slice_vs_portable_numpy_mfu",
        inputs={"low": low, "skip_mid": skip_mid, "skip_high": skip_high},
        official_output=official_mfu_output,
        portable_output=portable_mfu.pyr_out,
        official_weight_keys=fixture.mfu_weight_keys,
        source_forward_parity_proven=True,
        full_stack_source_forward_parity_proven=False,
        blockers=[],
    )
    hfr_row = _component_row(
        component_id="hfr",
        classification="official_source_fixture_hfr_state_dict_mapping_proven",
        backend="official_snerv_torch_decoder_slice_vs_portable_numpy_hfr",
        inputs={"pyr_out": official_mfu_output},
        official_output=official_hfr_output,
        portable_output=portable_hfr.yh_out,
        official_weight_keys=fixture.hfr_weight_keys,
        source_forward_parity_proven=True,
        full_stack_source_forward_parity_proven=False,
        blockers=[],
    )
    weight_manifest = _weight_manifest(fixture)
    source_replay = {
        "schema": SOURCE_REPLAY_SCHEMA,
        "backend": "official_torch_vs_portable",
        "replay_ran": True,
        "input_bundle_sha256": _hash_named_arrays(
            {"mfu.low": low, "mfu.skip_mid": skip_mid, "mfu.skip_high": skip_high}
        ),
        "official_output_sha256": _hash_named_arrays(
            {"mfu.pyr_out": official_mfu_output, "hfr.yh_out": official_hfr_output}
        ),
        "portable_output_sha256": _hash_named_arrays(
            {
                "mfu.pyr_out": portable_mfu.pyr_out,
                "hfr.yh_out": portable_hfr.yh_out,
            }
        ),
        "official_output_shape": list(official_hfr_output.shape),
        "portable_output_shape": list(portable_hfr.yh_out.shape),
        "mfu_hfr_source_fixture_forward_parity_proven": True,
        "full_stack_source_forward_parity_proven": False,
        "max_abs_error": max(
            float(mfu_row["max_abs_error"]),
            float(hfr_row["max_abs_error"]),
        ),
        "blockers": [
            "snerv_official_trained_checkpoint_state_dict_not_loaded",
            "snerv_official_snerv_t_full_tub_source_forward_replay_missing",
        ],
        **FALSE_AUTHORITY,
    }
    receiver_frame_replay = _receiver_payload_frame_replay_from_fixture(
        mfu=mfu,
        heads=heads,
        low=low,
        skip_mid=skip_mid,
        skip_high=skip_high,
    )
    return mfu_row, hfr_row, source_replay, weight_manifest, receiver_frame_replay


def _run_tub_graph_input_replay() -> dict[str, Any]:
    import torch

    current = _positive_fixture((3, 4, 4), modulo=17)
    previous = current + 0.125
    next_frame = current + 0.25
    portable = prepare_official_tub_graph_inputs(current, previous, next_frame)
    frames = torch.tensor(
        np.stack([current, previous, next_frame], axis=0),
        dtype=torch.float64,
    )
    lf = (
        frames[:, :, 0::2, 0::2]
        + frames[:, :, 0::2, 1::2]
        + frames[:, :, 1::2, 0::2]
        + frames[:, :, 1::2, 1::2]
    ) * 0.5
    normalized = (lf - lf.min()) / (lf.max() - lf.min())
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    prev_lowpass_over_2 = ((normalized[0:1] + normalized[1:2]) * inv_sqrt2) / 2.0
    next_lowpass_over_2 = ((normalized[0:1] + normalized[2:3]) * inv_sqrt2) / 2.0
    official_outputs = {
        "lf_triplet": np.asarray(lf.detach().cpu().numpy()),
        "normalized_lf": np.asarray(normalized.detach().cpu().numpy()),
        "prev_lowpass_over_2": np.asarray(prev_lowpass_over_2.detach().cpu().numpy()),
        "next_lowpass_over_2": np.asarray(next_lowpass_over_2.detach().cpu().numpy()),
    }
    portable_outputs = {
        "lf_triplet": portable.lf_triplet,
        "normalized_lf": portable.normalized_lf,
        "prev_lowpass_over_2": portable.prev_lowpass_over_2,
        "next_lowpass_over_2": portable.next_lowpass_over_2,
    }
    graph_error = max(
        float(np.max(np.abs(official_outputs[name] - portable_outputs[name])))
        for name in official_outputs
    )
    temporal = torch.arange(1 * 12 * 2 * 3, dtype=torch.float64).reshape(1, 12, 2, 3)
    decoder_output = torch.arange(2 * 18 * 2 * 3, dtype=torch.float64).reshape(
        2, 18, 2, 3
    )
    emb_ch = int(temporal.shape[1]) // 2
    official_output2_decoder_input = torch.cat(
        [temporal[:, :emb_ch], temporal[:, emb_ch:]],
        0,
    )
    official_output2_shuffled = (
        decoder_output.reshape(2, -1, 2, 3, 2, 3)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(2, -1, 4, 9)
    )
    portable_fusion = official_output2_fusion_numpy(
        np.asarray(temporal.detach().cpu().numpy()),
        np.asarray(decoder_output.detach().cpu().numpy()),
        fc_hw=(2, 3),
    )
    official_fusion_outputs = {
        "output2_decoder_input": np.asarray(
            official_output2_decoder_input.detach().cpu().numpy()
        ),
        "output2_shuffled": np.asarray(official_output2_shuffled.detach().cpu().numpy()),
    }
    portable_fusion_outputs = {
        "output2_decoder_input": portable_fusion.decoder_input,
        "output2_shuffled": portable_fusion.output2_fused,
    }
    fusion_error = max(
        float(np.max(np.abs(official_fusion_outputs[name] - portable_fusion_outputs[name])))
        for name in official_fusion_outputs
    )
    official_all_outputs = {**official_outputs, **official_fusion_outputs}
    portable_all_outputs = {**portable_outputs, **portable_fusion_outputs}
    max_abs_error = max(graph_error, fusion_error)
    output_hash = _hash_named_arrays(official_all_outputs)
    blockers = [
        "snerv_official_pytorch_wavelets_runtime_dependency_missing",
        "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
        "snerv_official_tub_portable_output2_decoder_weight_mapping_missing",
        "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing",
    ]
    return {
        "schema": "snerv_official_source_forward_component_replay.v1",
        "component_id": "tub",
        "classification": "official_tub_graph_input_and_output2_fusion_source_fixture_proven_full_tub_blocked",
        "backend": "official_torch_vs_portable",
        "source_forward_parity_proven": False,
        "primitive_source_forward_parity_proven": True,
        "portable_output2_fusion_receiver_mapping_proven": fusion_error == 0.0,
        "source_forward_parity_falsified": False,
        "full_stack_source_forward_parity_proven": False,
        "full_tub_source_forward_parity_proven": False,
        "tolerance": 0.0,
        "max_abs_error": max_abs_error,
        "graph_input_max_abs_error": graph_error,
        "output2_fusion_max_abs_error": fusion_error,
        "input_sha256": _hash_named_arrays(
            {
                "current": current,
                "previous": previous,
                "next_frame": next_frame,
            }
        ),
        "official_output_sha256": output_hash,
        "portable_output_sha256": _hash_named_arrays(portable_all_outputs),
        "output_hashes_bit_identical": output_hash
        == _hash_named_arrays(portable_all_outputs),
        "output_shapes": _shape_map(official_all_outputs),
        "closed_blockers": [
            "snerv_official_snerv_t_output2_fusion_source_forward_replay_missing",
            "snerv_official_tub_portable_output2_fusion_receiver_mapping_missing",
        ],
        "official_weight_keys": [
            "weightless_source_lines:model/snerv_t.py:125-136",
            "weightless_source_lines:model/snerv_t.py:142-150",
            "unmapped_temporal_encoder:self.encoder[1]",
            "unmapped_temporal_encoder:self.encoder[2]",
            "unmapped_output2_decoder:self.decoder[self.decoder_len-1]",
        ],
        "official_source_contract": "model/snerv_t.py lines 125-150",
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _receiver_payload_frame_replay_from_fixture(
    *,
    mfu: OfficialSnervMfu,
    heads: OfficialHfrHeads,
    low: np.ndarray,
    skip_mid: np.ndarray,
    skip_high: np.ndarray,
) -> dict[str, Any]:
    tub_current = _positive_fixture((3, 8, 12), modulo=19)
    tub_previous = tub_current + 0.125
    tub_next_frame = tub_current + 0.25
    mfu_out = mfu.forward(low, skip_mid, skip_high)
    hfr_out = heads.forward(mfu_out.pyr_out)
    frame_channels = int(hfr_out.yh_out.shape[1])
    frame_h = int(hfr_out.yh_out.shape[-2]) * 2
    frame_w = int(hfr_out.yh_out.shape[-1]) * 2
    fc_hw = (2, 3)
    output2_h = max(1, frame_h // fc_hw[0])
    output2_w = max(1, frame_w // fc_hw[1])
    temporal_encoder_concat = _positive_fixture(
        (1, frame_channels * 2, output2_h, output2_w),
        modulo=23,
    )
    tub_output2_raw = _positive_fixture(
        (2, frame_channels * fc_hw[0] * fc_hw[1], output2_h, output2_w),
        modulo=29,
    )
    payload = encode_official_mfu_hfr_tub_decoder_payload(
        mfu=mfu,
        hfr_heads=heads,
        low=low,
        skip_mid=skip_mid,
        skip_high=skip_high,
        tub_current=tub_current,
        tub_previous=tub_previous,
        tub_next_frame=tub_next_frame,
        temporal_encoder_output_shape=tuple(int(v) for v in temporal_encoder_concat.shape),
        fc_hw=fc_hw,
        output2_decoder_output_shape=tuple(int(v) for v in tub_output2_raw.shape),
        tub_temporal_encoder_concat=temporal_encoder_concat,
        tub_output2_raw=tub_output2_raw,
        store_tub_output2_for_receiver_proof=True,
    )
    decoded = decode_official_mfu_hfr_tub_decoder_payload(payload)
    runtime_proof = execute_official_mfu_hfr_tub_decoder_payload(payload)
    decoded_frames = decoded.decode_frames(clip_to_uint8_range=False)
    output2_storage = dict(decoded.header.get("tub_output2_storage") or {})
    blockers = list(runtime_proof.get("source_forward_blockers") or [])
    if output2_storage.get("receiver_frame_decode_consumes_output2") is not True:
        blockers.append("snerv_official_tub_output2_receiver_frame_decode_not_bound")
    return {
        "schema": RECEIVER_PAYLOAD_FRAME_REPLAY_SCHEMA,
        "payload_schema": decoded.schema,
        "payload_bytes": len(payload),
        "payload_sha256": _hash_bytes(payload),
        "receiver_runtime_decode_proven": (
            runtime_proof.get("receiver_runtime_decode_proven") is True
        ),
        "receiver_export_self_consistency_verified": (
            runtime_proof.get("receiver_export_self_consistency_verified") is True
        ),
        "frame_producing_official_payload_replay_proven": True,
        "decoded_frames_shape": [int(v) for v in decoded_frames.shape],
        "decoded_frames_sha256": _hash_array(decoded_frames),
        "output_bundle_sha256": runtime_proof.get("output_bundle_sha256"),
        "official_tub_output2_fusion_executed": bool(
            runtime_proof.get("executed_components", {}).get(
                "official_tub_output2_fusion"
            )
        ),
        "receiver_frame_decode_consumes_output2": bool(
            output2_storage.get("receiver_frame_decode_consumes_output2")
        ),
        "source_forward_replay_bound": False,
        "source_forward_replay_verified": False,
        "source_forward_replay_authority": False,
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _failed_receiver_payload_frame_replay(blockers: Sequence[str]) -> dict[str, Any]:
    return {
        "schema": RECEIVER_PAYLOAD_FRAME_REPLAY_SCHEMA,
        "payload_schema": None,
        "payload_bytes": None,
        "payload_sha256": None,
        "receiver_runtime_decode_proven": False,
        "receiver_export_self_consistency_verified": False,
        "frame_producing_official_payload_replay_proven": False,
        "decoded_frames_shape": None,
        "decoded_frames_sha256": None,
        "output_bundle_sha256": None,
        "official_tub_output2_fusion_executed": False,
        "receiver_frame_decode_consumes_output2": False,
        "source_forward_replay_bound": False,
        "source_forward_replay_verified": False,
        "source_forward_replay_authority": False,
        "blockers": _ordered_unique(list(blockers)),
        **FALSE_AUTHORITY,
    }


def _portable_mfu_from_state_dict(fixture: _OfficialFixture) -> OfficialSnervMfu:
    dl = fixture.decoder_len
    return OfficialSnervMfu(
        spec=fixture.spec,
        upsample_mid=OfficialConvTranspose2dNchw(
            _tensor_array(fixture.state_dict[f"decoder.{dl + 3}.weight"]),
            _tensor_array(fixture.state_dict[f"decoder.{dl + 3}.bias"]),
            stride=fixture.spec.mid_stride,
        ),
        rb_mid=_portable_rb(fixture, f"decoder.{dl + 4}"),
        upsample_high=OfficialConvTranspose2dNchw(
            _tensor_array(fixture.state_dict[f"decoder.{dl + 5}.weight"]),
            _tensor_array(fixture.state_dict[f"decoder.{dl + 5}.bias"]),
            stride=fixture.spec.high_stride,
        ),
        rb_high=_portable_rb(fixture, f"decoder.{dl + 6}"),
    )


def _portable_hfr_from_state_dict(fixture: _OfficialFixture) -> OfficialHfrHeads:
    dl = fixture.decoder_len
    return OfficialHfrHeads(
        lh_head=_portable_hfr_head(fixture, f"decoder.{dl}"),
        hl_head=_portable_hfr_head(fixture, f"decoder.{dl + 1}"),
        hh_head=_portable_hfr_head(fixture, f"decoder.{dl + 2}"),
    )


def _portable_rb(
    fixture: _OfficialFixture,
    prefix: str,
) -> OfficialResidualBlocksWithInputConv:
    blocks = []
    for idx in range(fixture.spec.num_blocks):
        base = f"{prefix}.main.1.{idx}"
        blocks.append(
            OfficialResidualBlockNoBN(
                conv1=OfficialConv2dNchw(
                    _tensor_array(fixture.state_dict[f"{base}.conv1.weight"]),
                    _tensor_array(fixture.state_dict[f"{base}.conv1.bias"]),
                    padding=1,
                ),
                conv2=OfficialConv2dNchw(
                    _tensor_array(fixture.state_dict[f"{base}.conv2.weight"]),
                    _tensor_array(fixture.state_dict[f"{base}.conv2.bias"]),
                    padding=1,
                ),
            )
        )
    return OfficialResidualBlocksWithInputConv(
        input_conv=OfficialConv2dNchw(
            _tensor_array(fixture.state_dict[f"{prefix}.main.0.weight"]),
            _tensor_array(fixture.state_dict[f"{prefix}.main.0.bias"]),
            padding=1,
        ),
        residual_blocks=tuple(blocks),
    )


def _portable_hfr_head(
    fixture: _OfficialFixture,
    prefix: str,
) -> OfficialHfrConvBlock:
    return OfficialHfrConvBlock(
        conv1=OfficialConv2dNchw(
            _tensor_array(fixture.state_dict[f"{prefix}.conv1.weight"]),
            _tensor_array(fixture.state_dict[f"{prefix}.conv1.bias"]),
        ),
        conv2=OfficialConv2dNchw(
            _tensor_array(fixture.state_dict[f"{prefix}.conv2.weight"]),
            _tensor_array(fixture.state_dict[f"{prefix}.conv2.bias"]),
            padding=1,
        ),
    )


def _component_row(
    *,
    component_id: str,
    classification: str,
    backend: str,
    inputs: Mapping[str, np.ndarray],
    official_output: np.ndarray,
    portable_output: np.ndarray,
    official_weight_keys: Sequence[str],
    source_forward_parity_proven: bool,
    full_stack_source_forward_parity_proven: bool,
    blockers: Sequence[str],
) -> dict[str, Any]:
    max_abs_error = float(np.max(np.abs(official_output - portable_output)))
    official_hash = _hash_array(official_output)
    portable_hash = _hash_array(portable_output)
    return {
        "schema": "snerv_official_source_forward_component_replay.v1",
        "component_id": component_id,
        "classification": classification,
        "backend": backend,
        "source_forward_parity_proven": bool(source_forward_parity_proven),
        "source_forward_parity_falsified": False,
        "full_stack_source_forward_parity_proven": bool(
            full_stack_source_forward_parity_proven
        ),
        "tolerance": 0.0,
        "max_abs_error": max_abs_error,
        "input_sha256": _hash_named_arrays(inputs),
        "official_output_sha256": official_hash,
        "portable_output_sha256": portable_hash,
        "output_hashes_bit_identical": official_hash == portable_hash,
        "official_output_shape": list(np.asarray(official_output).shape),
        "portable_output_shape": list(np.asarray(portable_output).shape),
        "official_weight_keys": list(official_weight_keys),
        "official_weight_sha256": _hash_text_lines(official_weight_keys),
        "blockers": list(blockers),
        **FALSE_AUTHORITY,
    }


def _failed_component_row(component_id: str, exc: Exception) -> dict[str, Any]:
    return {
        "schema": "snerv_official_source_forward_component_replay.v1",
        "component_id": component_id,
        "classification": "official_source_fixture_replay_failed",
        "backend": "official_torch_vs_portable",
        "source_forward_parity_proven": False,
        "source_forward_parity_falsified": False,
        "full_stack_source_forward_parity_proven": False,
        "tolerance": 0.0,
        "max_abs_error": None,
        "input_sha256": None,
        "official_output_sha256": None,
        "portable_output_sha256": None,
        "official_weight_keys": [],
        "blockers": [f"snerv_official_source_harness_failed:{type(exc).__name__}"],
        "error": str(exc),
        **FALSE_AUTHORITY,
    }


def _weight_manifest(fixture: _OfficialFixture) -> dict[str, Any]:
    entries = []
    for key in fixture.selected_weight_keys:
        array = _tensor_array(fixture.state_dict[key])
        entries.append(
            {
                "key": key,
                "receiver_key": _receiver_key_for_official_key(key, fixture.decoder_len),
                "shape": list(array.shape),
                "dtype": "float64",
                "sha256": _hash_array(array),
            }
        )
    return {
        "schema": "snerv_official_state_dict_mapping_manifest.v1",
        "state_dict_kind": "synthetic_dyadic_source_fixture_not_official_checkpoint",
        "state_dict_sha256": _hash_weight_entries(entries),
        "state_dict_key_count": len(entries),
        "weight_key_count": len(entries),
        "weight_entries": entries,
        "official_trained_checkpoint_loaded": False,
        "official_mfu_hfr_source_fixture_weight_mapping_proven": True,
        "official_tub_temporal_encoder_weight_mapping_proven": False,
        "blockers": [
            "snerv_official_trained_checkpoint_state_dict_not_loaded",
            "snerv_official_tub_encoder_decoder_weights_not_loaded",
        ],
    }


def _selected_decoder_weight_keys(
    state_dict: Mapping[str, Any],
    decoder_len: int,
) -> list[str]:
    prefixes = tuple(f"decoder.{idx}." for idx in range(decoder_len, decoder_len + 7))
    return sorted(key for key in state_dict if key.startswith(prefixes))


def _infer_official_decoder_len(state_dict: Mapping[str, Any]) -> int | None:
    indices: set[int] = set()
    for key in state_dict:
        parts = str(key).split(".")
        if len(parts) < 3 or parts[0] != "decoder":
            continue
        try:
            indices.add(int(parts[1]))
        except ValueError:
            continue
    if not indices:
        return None
    candidates = []
    for start in sorted(indices):
        groups = _official_checkpoint_group_prefixes(start)
        present = {
            group
            for group, prefixes in groups.items()
            if any(str(key).startswith(prefixes) for key in state_dict)
        }
        hfr_mfu_count = len(
            present
            & {
                "hfr_lh",
                "hfr_hl",
                "hfr_hh",
                "mfu_upsample_mid",
                "mfu_rb_mid",
                "mfu_upsample_high",
                "mfu_rb_high",
            }
        )
        if hfr_mfu_count:
            candidates.append((hfr_mfu_count, start))
    if not candidates:
        return None
    candidates.sort()
    return int(candidates[-1][1])


def _official_checkpoint_group_prefixes(decoder_len: int) -> dict[str, tuple[str, ...]]:
    return {
        "hfr_lh": (f"decoder.{decoder_len}.",),
        "hfr_hl": (f"decoder.{decoder_len + 1}.",),
        "hfr_hh": (f"decoder.{decoder_len + 2}.",),
        "mfu_upsample_mid": (f"decoder.{decoder_len + 3}.",),
        "mfu_rb_mid": (f"decoder.{decoder_len + 4}.",),
        "mfu_upsample_high": (f"decoder.{decoder_len + 5}.",),
        "mfu_rb_high": (f"decoder.{decoder_len + 6}.",),
        "tub_temporal_encoder_1": ("encoder.1.",),
        "tub_temporal_encoder_2": ("encoder.2.",),
        "tub_output2_decoder": (f"decoder.{decoder_len - 1}.",),
    }


def _official_group_for_key(
    key: str,
    groups: Mapping[str, tuple[str, ...]],
) -> str | None:
    for group, prefixes in groups.items():
        if any(str(key).startswith(prefix) for prefix in prefixes):
            return group
    return None


def _component_for_group(group: str) -> str:
    if group.startswith("hfr_"):
        return "hfr"
    if group.startswith("mfu_"):
        return "mfu"
    if group.startswith("tub_"):
        return "tub"
    return "unknown"


def _checkpoint_mapping_component_row(
    *,
    component_id: str,
    required_groups: Sequence[str],
    present_groups: set[str],
    entries: Sequence[Mapping[str, Any]],
    source_blocker: str,
) -> dict[str, Any]:
    missing_groups = [group for group in required_groups if group not in present_groups]
    component_entries = [
        row for row in entries if str(row.get("component_id")) == component_id
    ]
    mapping_proven = not missing_groups and bool(component_entries)
    blockers = (
        [source_blocker]
        if mapping_proven
        else [
            f"snerv_official_trained_checkpoint_{component_id}_weight_mapping_incomplete",
            source_blocker,
        ]
    )
    if component_id == "tub" and not mapping_proven:
        blockers.append("snerv_official_tub_encoder_decoder_weights_not_loaded")
    return {
        "schema": "snerv_official_trained_checkpoint_component_mapping.v1",
        "component_id": component_id,
        "required_groups": list(required_groups),
        "present_groups": [group for group in required_groups if group in present_groups],
        "missing_groups": missing_groups,
        "trained_checkpoint_weight_mapping_proven": mapping_proven,
        "source_forward_parity_proven": False,
        "source_forward_replay_authority": False,
        "mapped_weight_key_count": len(component_entries),
        "mapped_weight_byte_count": int(
            sum(int(row.get("byte_count") or 0) for row in component_entries)
        ),
        "mapped_weight_entries_sha256": _hash_weight_entries(component_entries),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _native_receiver_checkpoint_mapping(
    state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    """Map native receiver/export checkpoint atoms into official payload names."""

    weight_entries: list[dict[str, Any]] = []
    activation_entries: list[dict[str, Any]] = []
    for key in sorted(str(key) for key in state_dict):
        receiver_key = _native_hfr_receiver_key(key)
        if receiver_key is not None:
            array = _state_value_array(state_dict[key])
            weight_entries.append(
                {
                    "key": key,
                    "receiver_key": receiver_key,
                    "component_id": "hfr",
                    "official_group": _native_hfr_group(receiver_key),
                    "mapping_kind": "native_mlx_hfr_head_weight_to_official_receiver_payload",
                    "shape": [int(value) for value in array.shape],
                    "dtype": str(array.dtype),
                    "sha256": _hash_array_exact(array),
                    "f64_sha256": _hash_array(array),
                    "byte_count": int(np.ascontiguousarray(array).nbytes),
                }
            )
            continue
        activation = _native_receiver_activation_entry(key, state_dict[key])
        if activation is not None:
            activation_entries.append(activation)
    return {
        "weight_entries": weight_entries,
        "activation_entries": activation_entries,
        "known_entry_count": len(weight_entries) + len(activation_entries),
        "present_groups": {
            str(row["official_group"])
            for row in [*weight_entries, *activation_entries]
        },
    }


def _native_receiver_checkpoint_mapping_manifest(
    raw_state: Mapping[str, Any],
    *,
    native_mapping: Mapping[str, Any],
    state_dict_kind: str,
    source: str | None,
) -> dict[str, Any]:
    weight_entries = [
        dict(row) for row in native_mapping.get("weight_entries") or ()
    ]
    activation_entries = [
        dict(row) for row in native_mapping.get("activation_entries") or ()
    ]
    present_groups = {
        str(group) for group in native_mapping.get("present_groups") or ()
    }
    hfr_row = _checkpoint_mapping_component_row(
        component_id="hfr",
        required_groups=("hfr_lh", "hfr_hl", "hfr_hh"),
        present_groups=present_groups,
        entries=weight_entries,
        source_blocker=(
            "snerv_hfr_source_forward_replay_requires_upstream_torch_state_dict_mapping"
        ),
    )
    mfu_row = _native_activation_component_row(
        component_id="mfu",
        required_groups=("mfu_low", "mfu_skip_mid", "mfu_skip_high"),
        present_groups=present_groups,
        entries=activation_entries,
        blocker_if_activation_present=(
            "snerv_official_mfu_native_receiver_activation_payload_not_upstream_weight_mapping"
        ),
        blocker_if_missing=(
            "snerv_official_trained_checkpoint_mfu_activation_mapping_incomplete"
        ),
        source_blocker=(
            "snerv_mfu_source_forward_replay_requires_upstream_torch_state_dict_mapping"
        ),
    )
    tub_row = _native_activation_component_row(
        component_id="tub",
        required_groups=("tub_temporal_encoder_concat", "tub_output2_raw"),
        present_groups=present_groups,
        entries=activation_entries,
        blocker_if_activation_present=(
            "snerv_official_tub_receiver_activation_payload_not_temporal_encoder_weights"
        ),
        blocker_if_missing=(
            "snerv_official_tub_encoder_decoder_weights_not_loaded"
        ),
        source_blocker=(
            "snerv_tub_full_source_forward_replay_requires_temporal_encoder_decoder_fusion_mapping"
        ),
    )
    component_rows = [hfr_row, mfu_row, tub_row]
    hfr_proven = hfr_row["trained_checkpoint_weight_mapping_proven"] is True
    mfu_activation = mfu_row["receiver_activation_payload_bound"] is True
    tub_activation = tub_row["receiver_activation_payload_bound"] is True
    blockers = [
        blocker
        for row in component_rows
        for blocker in row.get("blockers", ())
    ]
    blockers.append(OFFICIAL_MFU_HFR_TUB_SOURCE_FORWARD_BLOCKER)
    return {
        "schema": TRAINED_CHECKPOINT_MAPPING_SCHEMA,
        "state_dict_kind": state_dict_kind,
        "state_dict_source": source,
        "state_dict_key_count": len(raw_state),
        "decoder_len": None,
        "decoder_len_source": "not_applicable_native_receiver_state",
        "state_dict_mapping_dialect": "native_mlx_receiver_state",
        "official_trained_checkpoint_loaded": True,
        "official_hfr_trained_checkpoint_weight_mapping_proven": hfr_proven,
        "official_mfu_trained_checkpoint_weight_mapping_proven": False,
        "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": False,
        "official_tub_temporal_encoder_weight_mapping_proven": False,
        "official_tub_output2_decoder_weight_mapping_proven": False,
        "official_mfu_receiver_activation_payload_bound": mfu_activation,
        "official_tub_receiver_activation_payload_bound": tub_activation,
        "official_native_receiver_state_mapping_proven": bool(
            hfr_proven and mfu_activation
        ),
        "state_dict_sha256": _hash_state_dict_exact(raw_state),
        "mapped_weight_key_count": len(weight_entries),
        "mapped_weight_byte_count": int(
            sum(int(row.get("byte_count") or 0) for row in weight_entries)
        ),
        "mapped_weight_entries_sha256": _hash_weight_entries(weight_entries),
        "weight_entries": weight_entries,
        "mapped_activation_key_count": len(activation_entries),
        "mapped_activation_byte_count": int(
            sum(int(row.get("byte_count") or 0) for row in activation_entries)
        ),
        "mapped_activation_entries_sha256": _hash_weight_entries(activation_entries),
        "activation_entries": activation_entries,
        "component_rows": component_rows,
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _native_activation_component_row(
    *,
    component_id: str,
    required_groups: Sequence[str],
    present_groups: set[str],
    entries: Sequence[Mapping[str, Any]],
    blocker_if_activation_present: str,
    blocker_if_missing: str,
    source_blocker: str,
) -> dict[str, Any]:
    missing_groups = [group for group in required_groups if group not in present_groups]
    component_entries = [
        row for row in entries if str(row.get("component_id")) == component_id
    ]
    activation_bound = not missing_groups and bool(component_entries)
    blockers = [
        blocker_if_activation_present if activation_bound else blocker_if_missing,
        source_blocker,
    ]
    if component_id == "tub":
        blockers.append("snerv_official_tub_portable_temporal_encoder_weight_mapping_missing")
        blockers.append("snerv_official_tub_portable_output2_decoder_weight_mapping_missing")
    return {
        "schema": "snerv_official_trained_checkpoint_component_mapping.v1",
        "component_id": component_id,
        "required_groups": list(required_groups),
        "present_groups": [group for group in required_groups if group in present_groups],
        "missing_groups": missing_groups,
        "trained_checkpoint_weight_mapping_proven": False,
        "receiver_activation_payload_bound": activation_bound,
        "source_forward_parity_proven": False,
        "source_forward_replay_authority": False,
        "mapped_weight_key_count": 0,
        "mapped_weight_byte_count": 0,
        "mapped_activation_key_count": len(component_entries),
        "mapped_activation_byte_count": int(
            sum(int(row.get("byte_count") or 0) for row in component_entries)
        ),
        "mapped_activation_entries_sha256": _hash_weight_entries(component_entries),
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _native_hfr_receiver_key(key: str) -> str | None:
    if key.startswith("hfr.") and any(
        key.startswith(f"hfr.{name}.") for name in ("lh", "hl", "hh")
    ):
        return key
    parts = key.split("_")
    if len(parts) != 4 or parts[0] != "hfr":
        return None
    name, conv, param = parts[1:]
    if name not in {"lh", "hl", "hh"} or conv not in {"conv1", "conv2"}:
        return None
    if param not in {"weight", "bias"}:
        return None
    return f"hfr.{name}.{conv}.{param}"


def _native_hfr_group(receiver_key: str) -> str:
    for name in ("lh", "hl", "hh"):
        if receiver_key.startswith(f"hfr.{name}."):
            return f"hfr_{name}"
    return "hfr_unknown"


def _native_receiver_activation_entry(
    key: str,
    value: Any,
) -> dict[str, Any] | None:
    activation_keys = {
        "low": ("inputs.mfu.low", "mfu", "mfu_low"),
        "skip_mid": ("inputs.mfu.skip_mid", "mfu", "mfu_skip_mid"),
        "skip_high": ("inputs.mfu.skip_high", "mfu", "mfu_skip_high"),
        "tub.temporal_encoder_concat": (
            "tub.temporal_encoder_concat",
            "tub",
            "tub_temporal_encoder_concat",
        ),
        "tub_temporal_encoder_concat": (
            "tub.temporal_encoder_concat",
            "tub",
            "tub_temporal_encoder_concat",
        ),
        "tub.output2_raw": ("tub.output2_raw", "tub", "tub_output2_raw"),
        "tub_output2_raw": ("tub.output2_raw", "tub", "tub_output2_raw"),
    }
    if key not in activation_keys:
        return None
    receiver_key, component_id, group = activation_keys[key]
    array = _state_value_array(value)
    return {
        "key": key,
        "receiver_key": receiver_key,
        "component_id": component_id,
        "official_group": group,
        "mapping_kind": "native_mlx_activation_to_official_receiver_payload",
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "sha256": _hash_array_exact(array),
        "f64_sha256": _hash_array(array),
        "byte_count": int(np.ascontiguousarray(array).nbytes),
    }


def _assign_sparse_source_fixture_weights(model: Any, decoder_len: int) -> None:
    state = model.state_dict()
    for key in _selected_decoder_weight_keys(state, decoder_len):
        state[key].zero_()
    for channel in range(6):
        state[f"decoder.{decoder_len + 3}.weight"][channel, channel, 0, 0] = 1.0
    for channel in range(4):
        state[f"decoder.{decoder_len + 5}.weight"][channel, channel, 0, 0] = 1.0
        state[f"decoder.{decoder_len + 4}.main.0.weight"][channel, 6 + channel, 1, 1] = 1.0
        state[f"decoder.{decoder_len + 4}.main.1.0.conv1.weight"][channel, channel, 1, 1] = 1.0
        state[f"decoder.{decoder_len + 4}.main.1.0.conv2.weight"][channel, channel, 1, 1] = 1.0
    for channel in range(3):
        state[f"decoder.{decoder_len + 6}.main.0.weight"][channel, 4 + channel, 1, 1] = 1.0
        state[f"decoder.{decoder_len + 6}.main.1.0.conv1.weight"][channel, channel, 1, 1] = 1.0
        state[f"decoder.{decoder_len + 6}.main.1.0.conv2.weight"][channel, channel, 1, 1] = 1.0
    for head_offset, scale in enumerate((1.0, 2.0, 4.0)):
        head_idx = decoder_len + head_offset
        for channel in range(3):
            state[f"decoder.{head_idx}.conv1.weight"][channel, channel, 0, 0] = 1.0
            state[f"decoder.{head_idx}.conv2.weight"][channel, channel, 1, 1] = scale


def _receiver_key_for_official_key(key: str, decoder_len: int) -> str:
    replacements = {
        f"decoder.{decoder_len}.": "hfr.lh.",
        f"decoder.{decoder_len + 1}.": "hfr.hl.",
        f"decoder.{decoder_len + 2}.": "hfr.hh.",
        f"decoder.{decoder_len + 3}.": "mfu.upsample_mid.",
        f"decoder.{decoder_len + 4}.": "mfu.rb_mid.",
        f"decoder.{decoder_len + 5}.": "mfu.upsample_high.",
        f"decoder.{decoder_len + 6}.": "mfu.rb_high.",
    }
    out = key
    for official, receiver in replacements.items():
        if key.startswith(official):
            out = receiver + key[len(official) :]
            break
    return (
        out.replace(".main.0.", ".input_conv.")
        .replace(".main.1.", ".residual_blocks.")
    )


def _local_receiver_adapter_source_gap(local_root: Path) -> dict[str, Any]:
    path = local_root / "src/tac/substrates/snerv_inverse_steg_carrier/carrier.py"
    source = path.read_text(encoding="utf-8") if path.is_file() else ""
    receiver_markers = (
        "class MultiResolutionFusionUnit",
        "class HighFrequencyRestorer",
        "class SnervTemporalExtension",
    )
    source_markers = (
        "nn.ConvTranspose2d",
        "decoder_len+3",
        "torch.cat([up1, embed_list[-2]]",
        "self.decoder[self.decoder_len-1]",
    )
    return {
        "schema": "snerv_local_receiver_adapter_source_gap.v1",
        "path": path.as_posix(),
        "sha256": _hash_bytes(path.read_bytes()) if path.is_file() else None,
        "receiver_safe_adapter_present": all(marker in source for marker in receiver_markers),
        "official_source_forward_markers_present": all(
            marker in source for marker in source_markers
        ),
        "source_forward_parity_proven": False,
        "classification": "receiver_safe_adapter_not_official_source_forward_semantics",
        "blockers": ["local_receiver_safe_adapter_is_not_official_snerv_source_graph"],
        **FALSE_AUTHORITY,
    }


@contextmanager
def _official_source_import_context(official_root: Path) -> Iterable[None]:
    original_path = list(sys.path)
    sentinel = object()
    module_names = (
        "pytorch_wavelets",
        "matplotlib",
        "matplotlib.path",
        "model",
        "model.snerv",
        "model.layers",
        "model.residual_block",
    )
    saved = {name: sys.modules.get(name, sentinel) for name in module_names}
    for name in module_names:
        sys.modules.pop(name, None)

    wavelets = types.ModuleType("pytorch_wavelets")

    class _UnavailableWavelet:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def cuda(self) -> _UnavailableWavelet:
            return self

        def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("pytorch_wavelets dependency is stubbed")

    wavelets.DWT = _UnavailableWavelet
    wavelets.IDWT = _UnavailableWavelet
    wavelets.DWT1D = _UnavailableWavelet
    matplotlib = types.ModuleType("matplotlib")
    matplotlib_path = types.ModuleType("matplotlib.path")
    matplotlib_path.Path = object
    sys.modules["pytorch_wavelets"] = wavelets
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.path"] = matplotlib_path
    sys.path.insert(0, official_root.as_posix())
    try:
        yield
    finally:
        sys.path[:] = original_path
        for name in module_names:
            sys.modules.pop(name, None)
            value = saved[name]
            if value is not sentinel:
                sys.modules[name] = value


def _positive_fixture(shape: Sequence[int], *, modulo: int) -> np.ndarray:
    values = (np.arange(int(np.prod(shape)), dtype=np.float64).reshape(tuple(shape)) % modulo) + 1
    return (values / 64.0).astype(np.float64)


def _tensor_array(tensor: Any) -> np.ndarray:
    return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float64)


def _state_value_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    return np.asarray(value)


def _hash_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    return _hash_bytes(arr.tobytes())


def _hash_array_exact(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    h = sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(b"\0")
    h.update(json.dumps(list(arr.shape), sort_keys=True).encode("utf-8"))
    h.update(b"\0")
    h.update(arr.tobytes())
    return h.hexdigest()


def _hash_state_dict_exact(state_dict: Mapping[str, Any]) -> str:
    h = sha256()
    for key in sorted(state_dict):
        array = _state_value_array(state_dict[key])
        h.update(str(key).encode("utf-8"))
        h.update(b"\0")
        h.update(_hash_array_exact(array).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _hash_named_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    h = sha256()
    for name in sorted(arrays):
        arr = np.ascontiguousarray(np.asarray(arrays[name], dtype="<f8"))
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(json.dumps(list(arr.shape), sort_keys=True).encode("utf-8"))
        h.update(b"\0")
        h.update(arr.tobytes())
        h.update(b"\0")
    return h.hexdigest()


def _shape_map(arrays: Mapping[str, np.ndarray]) -> dict[str, list[int]]:
    return {
        name: [int(v) for v in np.asarray(array).shape]
        for name, array in arrays.items()
    }


def _hash_text_lines(lines: Iterable[str]) -> str:
    h = sha256()
    for line in lines:
        h.update(str(line).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _hash_weight_entries(entries: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _hash_bytes(payload)


def _hash_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _git_head_sha(path: Path) -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "-C", path.as_posix(), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _ordered_unique(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


__all__ = [
    "DEFAULT_OFFICIAL_SNERV_REPO",
    "FALSE_AUTHORITY",
    "OFFICIAL_SNERV_SHA",
    "SCHEMA",
    "SOURCE_REPLAY_SCHEMA",
    "TRAINED_CHECKPOINT_MAPPING_SCHEMA",
    "build_snerv_official_source_forward_harness_artifact",
    "build_snerv_official_trained_checkpoint_mapping_manifest",
]
