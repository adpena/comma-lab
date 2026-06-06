# SPDX-License-Identifier: MIT
"""Executable SNeRV_T TUB source-forward replay harness.

This harness burns down the executable SNeRV TUB source-forward blocker.  The
upstream ``model/snerv_t.py`` temporal encoder plus ``output_2`` fusion path can
be executed as a source fixture, and an optional one-step source smoke can
persist a receiver-owned state slice.  It still claims no score authority.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import io
import json
import sys
import types
import warnings
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from tac.analysis.source_forward_bit_flip_falsification import (
    build_named_arrays_bit_flip_falsification,
)
from tac.substrates.snerv_inverse_steg_carrier.official_tub import (
    OFFICIAL_SNERV_T_SOURCE_SHA,
    official_output2_fusion_numpy,
    official_tub_frame_reconstruction_numpy,
    prepare_official_tub_graph_inputs,
)

SCHEMA = "snerv_official_tub_source_forward_replay.v1"
COMPONENT_SCHEMA = "snerv_official_tub_source_forward_component.v1"
TRAINED_CHECKPOINT_MAPPING_SCHEMA = (
    "snerv_official_trained_checkpoint_state_dict_mapping_manifest.v1"
)
CHECKPOINT_LOAD_SCHEMA = "snerv_official_tub_checkpoint_load.v1"
OFFICIAL_REPO_URL = "https://github.com/qwertja/SNeRV"
DEFAULT_OFFICIAL_SNERV_REPO = Path(
    "/Volumes/VertigoDataTier/pact/experiments/results/"
    "oss_nerv_source_audit_20260602T113720Z/repos/SNeRV"
)

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}

TUB_FRAME_RECONSTRUCTION_BLOCKER = (
    "snerv_official_tub_frame_reconstruction_source_forward_replay_missing"
)
TUB_CLOSED_BY_FIXTURE_REPLAY: tuple[str, ...] = (
    "snerv_official_tub_graph_inputs_only_not_full_source_forward_parity",
    "snerv_official_snerv_t_output2_fusion_source_forward_replay_missing",
    "snerv_official_tub_portable_output2_fusion_receiver_mapping_missing",
    TUB_FRAME_RECONSTRUCTION_BLOCKER,
)
TUB_PRESERVED_BLOCKERS: tuple[str, ...] = (
    "snerv_official_trained_checkpoint_state_dict_not_loaded",
    "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
    "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
    "snerv_official_tub_portable_output2_decoder_weight_mapping_missing",
    "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing",
)
PYTORCH_WAVELETS_BLOCKER = "snerv_official_pytorch_wavelets_runtime_dependency_missing"
STATE_VALUE_ARTIFACT_BLOCKER = (
    "snerv_official_trained_checkpoint_state_dict_value_artifact_missing"
)
TUB_CHECKPOINT_EXPORT_LINEAGE_BLOCKER = (
    "snerv_official_tub_trained_checkpoint_export_lineage_missing"
)
CHECKPOINT_LOAD_MISSING_KEYS_BLOCKER = (
    "snerv_official_trained_checkpoint_state_dict_missing_keys"
)
CHECKPOINT_LOAD_UNEXPECTED_KEYS_BLOCKER = (
    "snerv_official_trained_checkpoint_state_dict_unexpected_keys"
)
CHECKPOINT_LOAD_SHAPE_MISMATCH_BLOCKER = (
    "snerv_official_trained_checkpoint_state_dict_shape_mismatch"
)
CHECKPOINT_LOAD_PATH_MISSING_BLOCKER = (
    "snerv_official_trained_checkpoint_state_dict_path_missing"
)
CHECKPOINT_LOAD_UNREADABLE_BLOCKER = (
    "snerv_official_trained_checkpoint_state_dict_unreadable"
)


class _OfficialTubCheckpointLoadFailed(RuntimeError):
    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        blockers = ",".join(str(value) for value in self.report.get("blockers", ()))
        super().__init__(blockers or "official TUB checkpoint load failed")


@dataclass(frozen=True)
class TubFixtureConfig:
    """Small source fixture that respects upstream SNeRV_T shape assumptions."""

    embed: str = ""
    ks: str = "0_1_5"
    num_blks: str = "1_1"
    enc_strds: tuple[int, ...] = (2, 2, 2, 2)
    enc_dim: str = "4_4"
    enc2_strds: tuple[int, ...] = (2, 2, 2, 2)
    conv_type: tuple[str, ...] = ("convnext", "pshuffel")
    norm: str = "none"
    act: str = "gelu"
    dec_strds: tuple[int, ...] = (1, 2, 2, 2, 2)
    fc_dim: int = 8
    fc_hw: str = "1_1"
    reduce: float = 1.2
    lower_width: int = 2
    num_blocks: int = 1
    out_bias: str = "tanh"
    crop_list: str = "640_1280"
    emb_size: int = 20

    def to_namespace(self) -> SimpleNamespace:
        payload = asdict(self)
        payload["enc_strds"] = list(self.enc_strds)
        payload["enc2_strds"] = list(self.enc2_strds)
        payload["conv_type"] = list(self.conv_type)
        payload["dec_strds"] = list(self.dec_strds)
        return SimpleNamespace(**payload)


def build_snerv_official_tub_source_forward_replay_artifact(
    *,
    official_repo_dir: str | Path = DEFAULT_OFFICIAL_SNERV_REPO,
    train_one_step: bool = False,
    output_state_dict_path: str | Path | None = None,
    official_trained_checkpoint_state_dict: Mapping[str, Any] | None = None,
    official_trained_checkpoint_state_dict_path: str | Path | None = None,
    official_trained_checkpoint_state_dict_kind: str = (
        "official_trained_checkpoint_state_dict"
    ),
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Return a fail-closed executable TUB source-forward replay artifact."""

    official_root = Path(official_repo_dir)
    if generated_utc is None:
        generated_utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    base = {
        "schema": SCHEMA,
        "family": "snerv",
        "component_id": "tub",
        "official_repo": {
            "repo_url": OFFICIAL_REPO_URL,
            "root": official_root.as_posix(),
            "head_sha": _git_head_sha(official_root),
            "expected_head_sha": OFFICIAL_SNERV_T_SOURCE_SHA,
        },
        "generated_utc": generated_utc,
        "source_forward_replay_executed": False,
        "official_tub_temporal_encoder_output2_source_fixture_replay_passed": False,
        "full_tub_source_forward_parity_proven": False,
        "source_forward_parity_proven": False,
        "official_trained_checkpoint_loaded": False,
        "official_trained_checkpoint_state_dict_mapping_verified": False,
        "official_trained_checkpoint_load": _checkpoint_not_requested_report(),
        "official_pytorch_wavelets_runtime_dependency_installed": _module_installed(
            "pytorch_wavelets"
        ),
        "functional_haar_shim_used_for_fixture": True,
        "closed_blockers": [],
        "preserved_blockers": list(TUB_PRESERVED_BLOCKERS),
        "blockers": [],
        **FALSE_AUTHORITY,
    }

    if not official_root.exists():
        return {
            **base,
            "blockers": [
                "snerv_official_source_checkout_missing",
                *TUB_PRESERVED_BLOCKERS,
            ],
        }

    try:
        payload = _run_source_fixture(
            official_root,
            train_one_step=train_one_step,
            output_state_dict_path=output_state_dict_path,
            official_trained_checkpoint_state_dict=(
                official_trained_checkpoint_state_dict
            ),
            official_trained_checkpoint_state_dict_path=(
                official_trained_checkpoint_state_dict_path
            ),
            official_trained_checkpoint_state_dict_kind=(
                official_trained_checkpoint_state_dict_kind
            ),
        )
    except _OfficialTubCheckpointLoadFailed as exc:
        blockers = _ordered_unique(
            [
                *(
                    str(blocker)
                    for blocker in exc.report.get("blockers", ())
                    if str(blocker)
                ),
                *TUB_PRESERVED_BLOCKERS,
            ]
        )
        return {
            **base,
            "official_trained_checkpoint_load": exc.report,
            "failure": f"{type(exc).__name__}: {exc}",
            "blockers": blockers,
        }
    except Exception as exc:  # pragma: no cover - fail-closed caller path.
        return {
            **base,
            "failure": f"{type(exc).__name__}: {exc}",
            "blockers": [
                "snerv_official_tub_temporal_encoder_output2_fixture_failed",
                *TUB_PRESERVED_BLOCKERS,
                *([] if base["official_pytorch_wavelets_runtime_dependency_installed"] else [PYTORCH_WAVELETS_BLOCKER]),
            ],
        }

    dependency_blockers = (
        []
        if base["official_pytorch_wavelets_runtime_dependency_installed"]
        else [PYTORCH_WAVELETS_BLOCKER]
    )
    preserved = _ordered_unique(
        [*TUB_PRESERVED_BLOCKERS, *dependency_blockers]
    )
    replay_passed = bool(
        payload["graph_input_parity"]["graph_input_parity_passed"]
        and payload["portable_output2_fusion"][
            "portable_output2_fusion_receiver_mapping_proven"
        ]
        and payload["full_forward_equivalence"]["manual_replay_matches_official_forward"]
        and payload["frame_reconstruction_equivalence"][
            "source_forward_frame_reconstruction_matches_official"
        ]
        and payload["temporal_path"]["output_tensors_finite"]
    )
    fixture_blockers = (
        []
        if replay_passed
        else _tub_fixture_replay_blockers(payload)
    )
    mapping_manifest = payload["official_trained_checkpoint_mapping_manifest"]
    checkpoint_load = dict(payload.get("official_trained_checkpoint_load") or {})
    state_dict_artifact = payload.get("official_trained_checkpoint_state_dict_artifact")
    state_dict_artifact = (
        dict(state_dict_artifact) if isinstance(state_dict_artifact, Mapping) else None
    )
    state_dict_value_artifact_ready = _state_dict_value_artifact_ready(
        state_dict_artifact
    )
    mapping_verified = bool(
        mapping_manifest.get("official_trained_checkpoint_state_dict_mapping_verified")
        is True
    )
    checkpoint_export_lineage = _mapping_manifest_has_checkpoint_export_lineage(
        mapping_manifest
    )
    full_source_parity = bool(
        replay_passed
        and mapping_verified
        and checkpoint_export_lineage
    )
    source_forward_authority = bool(
        full_source_parity and state_dict_value_artifact_ready
    )
    preserved = (
        [
            blocker
            for blocker in preserved
            if blocker
            not in {
                "snerv_official_trained_checkpoint_state_dict_not_loaded",
                "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
                "snerv_official_tub_encoder_decoder_weights_not_loaded",
                "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
                "snerv_official_tub_portable_output2_decoder_weight_mapping_missing",
                "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing",
            }
        ]
        if full_source_parity
        else preserved
    )
    if full_source_parity and not state_dict_value_artifact_ready:
        preserved = _ordered_unique([*preserved, STATE_VALUE_ARTIFACT_BLOCKER])
    if replay_passed and mapping_verified and not checkpoint_export_lineage:
        preserved = _ordered_unique([*preserved, TUB_CHECKPOINT_EXPORT_LINEAGE_BLOCKER])
    closed_blockers = list(TUB_CLOSED_BY_FIXTURE_REPLAY) if replay_passed else []
    if full_source_parity:
        closed_blockers.extend(
            [
                "snerv_official_trained_checkpoint_state_dict_not_loaded",
                "snerv_official_trained_checkpoint_state_dict_mapping_missing",
                "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
                "snerv_official_tub_encoder_decoder_weights_not_loaded",
                "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
                "snerv_official_tub_portable_output2_decoder_weight_mapping_missing",
                "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing",
                "snerv_tub_full_source_forward_replay_requires_temporal_encoder_decoder_fusion_mapping",
            ]
        )
    return {
        **base,
        "source_forward_replay_executed": True,
        "official_tub_temporal_encoder_output2_source_fixture_replay_passed": replay_passed,
        "official_trained_checkpoint_loaded": bool(
            mapping_manifest.get("official_trained_checkpoint_loaded") is True
            and (
                checkpoint_load.get("requested") is not True
                or checkpoint_load.get("loaded") is True
            )
        ),
        "official_trained_checkpoint_state_dict_mapping_verified": mapping_verified,
        "official_trained_checkpoint_load": checkpoint_load,
        "official_trained_checkpoint_export_lineage_verified": checkpoint_export_lineage,
        "official_trained_checkpoint_mapping_manifest": mapping_manifest,
        "official_trained_checkpoint_state_dict_artifact": state_dict_artifact,
        "official_trained_checkpoint_state_dict_path": (
            None if state_dict_artifact is None else state_dict_artifact.get("path")
        ),
        "official_trained_checkpoint_state_dict_slice_path": (
            None if state_dict_artifact is None else state_dict_artifact.get("path")
        ),
        "official_trained_checkpoint_state_dict_slice_present": (
            state_dict_artifact is not None
        ),
        "official_trained_checkpoint_state_dict_slice_file_present": (
            state_dict_artifact is not None
            and Path(str(state_dict_artifact.get("path") or "")).is_file()
        ),
        "official_trained_checkpoint_state_dict_slice_bytes": (
            None if state_dict_artifact is None else state_dict_artifact.get("bytes")
        ),
        "official_trained_checkpoint_state_dict_slice_sha256": (
            None if state_dict_artifact is None else state_dict_artifact.get("sha256")
        ),
        "official_trained_checkpoint_state_dict_slice_member_count": (
            None
            if state_dict_artifact is None
            else state_dict_artifact.get("member_count")
        ),
        "official_trained_checkpoint_state_dict_slice_member_names": (
            []
            if state_dict_artifact is None
            else list(state_dict_artifact.get("member_names") or [])
        ),
        "official_trained_checkpoint_state_dict_value_artifact_ready": (
            state_dict_value_artifact_ready
        ),
        "official_trained_checkpoint_state_dict_slice_runner_arg": (
            None
            if state_dict_artifact is None
            else "--snerv-official-trained-checkpoint-state-dict-path"
        ),
        "source_forward_training_smoke": {
            "schema": "snerv_official_tub_source_training_smoke.v1",
            "enabled": bool(train_one_step),
            "optimizer": "torch.optim.SGD",
            "step_count": int(payload["source_training_smoke"]["step_count"]),
            "loss_before": payload["source_training_smoke"]["loss_before"],
            "loss_after": payload["source_training_smoke"]["loss_after"],
            "state_dict_sha256": mapping_manifest.get("state_dict_sha256"),
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "source_fixture_config": payload["source_fixture_config"],
        "source_fixture_scope": (
            "deterministic_official_source_fixture_with_one_step_checkpoint"
            if train_one_step
            else "deterministic_official_source_fixture_not_trained_checkpoint"
        ),
        "source_fixture_not_training_config": True,
        "source_fixture_reason": (
            "small shape chosen to exercise upstream SNeRV_T temporal encoder, "
            "output_2 fusion, five-stage temporal decoder, MFU/HFR heads, "
            "and final source-forward frame reconstruction semantics on CPU"
        ),
        "source_pins": _source_pins(official_root),
        "dependency_contract": {
            "official_requirements_pins_pytorch_wavelets": True,
            "official_pytorch_wavelets_runtime_dependency_installed": base[
                "official_pytorch_wavelets_runtime_dependency_installed"
            ],
            "functional_haar_shim_used_for_fixture": True,
            "shim_scope": "one_level_haar_dwt_dwt1d_idwt_source_fixture_only",
            "shim_score_authority": False,
        },
        "graph_input_parity": payload["graph_input_parity"],
        "portable_output2_fusion": payload["portable_output2_fusion"],
        "temporal_path": payload["temporal_path"],
        "full_forward_equivalence": payload["full_forward_equivalence"],
        "frame_reconstruction_equivalence": payload[
            "frame_reconstruction_equivalence"
        ],
        "component_rows": [
            payload["graph_input_parity"],
            payload["portable_output2_fusion"],
            payload["temporal_path"],
            payload["full_forward_equivalence"],
            payload["frame_reconstruction_equivalence"],
        ],
        "closed_blockers": _ordered_unique(closed_blockers),
        "preserved_blockers": preserved,
        "blockers": _ordered_unique([*fixture_blockers, *preserved]),
        "full_tub_source_forward_parity_proven": full_source_parity,
        "source_forward_parity_proven": full_source_parity,
        "source_forward_replay_authority": source_forward_authority,
        **FALSE_AUTHORITY,
    }


def _run_source_fixture(
    official_root: Path,
    *,
    train_one_step: bool,
    output_state_dict_path: str | Path | None,
    official_trained_checkpoint_state_dict: Mapping[str, Any] | None,
    official_trained_checkpoint_state_dict_path: str | Path | None,
    official_trained_checkpoint_state_dict_kind: str,
) -> dict[str, Any]:
    import torch

    cfg = TubFixtureConfig()
    torch.manual_seed(20260604)
    with _official_tub_import_context(official_root):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            snerv_t = importlib.import_module("model.snerv_t")
        model = snerv_t.SNeRV_T(cfg.to_namespace()).double()
        checkpoint_load = _load_checkpoint_into_official_tub_model(
            model,
            torch_module=torch,
            state_dict=official_trained_checkpoint_state_dict,
            state_dict_path=official_trained_checkpoint_state_dict_path,
            state_dict_kind=official_trained_checkpoint_state_dict_kind,
        )

        current = _positive_fixture((1, 3, 32, 32), modulo=17)
        previous = current + 1.0 / 64.0
        next_frame = current + 1.0 / 32.0
        current_t = torch.from_numpy(current)
        previous_t = torch.from_numpy(previous)
        next_t = torch.from_numpy(next_frame)
        training_smoke = _run_one_step_training_smoke(
            model,
            current_t,
            previous_t,
            next_t,
            enabled=train_one_step,
        )
        model.eval()

        with torch.no_grad():
            manual = _manual_tub_source_replay(model, current_t, previous_t, next_t)
            img_out, embed_list, _dec_time, img_yl, yh_out = model(
                current_t,
                previous_t,
                next_t,
            )
        state_dict = model.state_dict()

    portable = prepare_official_tub_graph_inputs(
        current[0],
        previous[0],
        next_frame[0],
        temporal_encoder_output_shape=tuple(
            int(v) for v in manual["temporal_encoder_concat"].shape
        ),
        fc_hw=(int(model.fc_h), int(model.fc_w)),
        output2_decoder_output_shape=tuple(int(v) for v in manual["output2_raw"].shape),
    )
    portable_fusion = official_output2_fusion_numpy(
        manual["temporal_encoder_concat"],
        manual["output2_raw"],
        fc_hw=(int(model.fc_h), int(model.fc_w)),
    )
    graph_arrays = {
        "lf_triplet": manual["lf_triplet"],
        "normalized_lf": manual["normalized_lf"],
        "prev_lowpass_over_2": manual["prev_lowpass_over_2"],
        "next_lowpass_over_2": manual["next_lowpass_over_2"],
    }
    portable_arrays = {
        "lf_triplet": portable.lf_triplet,
        "normalized_lf": portable.normalized_lf,
        "prev_lowpass_over_2": portable.prev_lowpass_over_2,
        "next_lowpass_over_2": portable.next_lowpass_over_2,
    }
    graph_error = _max_abs_error(graph_arrays, portable_arrays)
    fusion_source_arrays = {
        "output2_decoder_input": manual["output2_decoder_input"],
        "output2_shuffled": manual["output2_shuffled"],
    }
    fusion_portable_arrays = {
        "output2_decoder_input": portable_fusion.decoder_input,
        "output2_shuffled": portable_fusion.output2_fused,
    }
    fusion_error = _max_abs_error(fusion_source_arrays, fusion_portable_arrays)

    official_embed0 = _tensor_array(embed_list[0][0])
    official_temporal_concat = _tensor_array(embed_list[0][1])
    official_decoder0 = _tensor_array(embed_list[1])
    official_final = _tensor_array(embed_list[-1])
    full_forward_arrays = {
        "embed_curr": official_embed0,
        "temporal_encoder_concat": official_temporal_concat,
        "decoder0_shuffled": official_decoder0,
        "final_decoder_output": official_final,
    }
    manual_forward_arrays = {
        "embed_curr": manual["embed_curr"],
        "temporal_encoder_concat": manual["temporal_encoder_concat"],
        "decoder0_shuffled": manual["decoder0_shuffled"],
        "final_decoder_output": manual["final_decoder_output"],
    }
    forward_error = _max_abs_error(full_forward_arrays, manual_forward_arrays)
    forward_passed = forward_error == 0.0
    forward_bit_flip_falsification = build_named_arrays_bit_flip_falsification(
        component_id="tub",
        official_outputs=full_forward_arrays,
        portable_outputs=manual_forward_arrays,
        tolerance=0.0,
        false_authority=FALSE_AUTHORITY,
    )
    official_frame_arrays = {
        "img_yl": _tensor_array(img_yl),
        "yh_out": _tensor_array(yh_out),
        "img_out": _tensor_array(img_out),
    }
    manual_frame_arrays = {
        "img_yl": manual["img_yl"],
        "yh_out": manual["yh_out"],
        "img_out": manual["frame_reconstruction"],
    }
    frame_error = _max_abs_error(official_frame_arrays, manual_frame_arrays)
    frame_passed = frame_error == 0.0

    temporal_arrays = {
        "temporal_encoder_prev": manual["temporal_encoder_prev"],
        "temporal_encoder_next": manual["temporal_encoder_next"],
        "temporal_encoder_concat": manual["temporal_encoder_concat"],
        "output2_raw": manual["output2_raw"],
        "output2_shuffled": manual["output2_shuffled"],
        "final_decoder_output": manual["final_decoder_output"],
        "mfu_up1": manual["mfu_up1"],
        "mfu_unet1": manual["mfu_unet1"],
        "mfu_unet1_up": manual["mfu_unet1_up"],
        "mfu_pyr_out": manual["mfu_pyr_out"],
        "manual_img_yl": manual["img_yl"],
        "manual_yh_out": manual["yh_out"],
        "manual_frame_reconstruction": manual["frame_reconstruction"],
        "full_img_out": _tensor_array(img_out),
        "full_img_yl": _tensor_array(img_yl),
        "full_yh_out": _tensor_array(yh_out),
    }
    checkpoint_loaded = checkpoint_load.get("loaded") is True
    checkpoint_state_dict = (
        checkpoint_load.get("_state_dict")
        if isinstance(checkpoint_load.get("_state_dict"), Mapping)
        else None
    )
    state_dict_artifact = (
        _write_deterministic_state_npz(
            Path(output_state_dict_path),
            state_dict,
        )
        if train_one_step and output_state_dict_path is not None
        else None
    )
    if state_dict_artifact is None and checkpoint_loaded:
        state_dict_artifact = _supplied_checkpoint_state_artifact(checkpoint_load)
    checkpoint_public = {
        key: value for key, value in checkpoint_load.items() if key != "_state_dict"
    }
    manifest_source = (
        str(checkpoint_public.get("state_dict_source") or "")
        if checkpoint_loaded
        else "official_snerv_t_one_step_source_smoke"
    )
    manifest_kind = (
        str(checkpoint_public.get("state_dict_kind") or "")
        if checkpoint_loaded
        else "official_snerv_t_one_step_trained_source_smoke_state_dict"
    )
    return {
        "source_fixture_config": asdict(cfg),
        "source_training_smoke": training_smoke,
        "official_trained_checkpoint_load": checkpoint_public,
        "official_trained_checkpoint_state_dict_artifact": state_dict_artifact,
        "official_trained_checkpoint_mapping_manifest": (
            _build_official_tub_trained_checkpoint_mapping_manifest(
                model_state_dict=(
                    checkpoint_state_dict
                    if checkpoint_loaded and checkpoint_state_dict is not None
                    else state_dict
                ),
                decoder_len=int(model.decoder_len),
                source=manifest_source,
                state_dict_kind=manifest_kind,
            )
            if (train_one_step or checkpoint_loaded)
            else _untrained_tub_source_fixture_mapping_manifest(
                state_dict=state_dict,
                decoder_len=int(model.decoder_len),
            )
        ),
        "graph_input_parity": {
            "schema": COMPONENT_SCHEMA,
            "component_id": "tub_graph_inputs",
            "classification": "official_tub_graph_inputs_match_local_numpy_primitive",
            "graph_input_parity_passed": graph_error == 0.0,
            "max_abs_error": graph_error,
            "official_output_sha256": _hash_named_arrays(graph_arrays),
            "portable_output_sha256": _hash_named_arrays(portable_arrays),
            "output_hashes_bit_identical": _hash_named_arrays(graph_arrays)
            == _hash_named_arrays(portable_arrays),
            "output_shapes": _shape_map(graph_arrays),
            "blockers": [],
            **FALSE_AUTHORITY,
        },
        "portable_output2_fusion": {
            "schema": COMPONENT_SCHEMA,
            "component_id": "tub_output2_portable_fusion",
            "classification": (
                "official_snerv_t_output2_split_concat_shuffle_matches_numpy_receiver_primitive"
            ),
            "portable_output2_fusion_receiver_mapping_proven": fusion_error == 0.0,
            "max_abs_error": fusion_error,
            "source_output_sha256": _hash_named_arrays(fusion_source_arrays),
            "portable_output_sha256": _hash_named_arrays(fusion_portable_arrays),
            "output_hashes_bit_identical": _hash_named_arrays(fusion_source_arrays)
            == _hash_named_arrays(fusion_portable_arrays),
            "output_shapes": _shape_map(fusion_source_arrays),
            "portable_primitive_metadata": portable_fusion.as_jsonable_metadata(),
            "closed_blockers": [
                "snerv_official_tub_portable_output2_fusion_receiver_mapping_missing"
            ],
            "blockers": [
                "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
                "snerv_official_tub_portable_output2_decoder_weight_mapping_missing",
            ],
            **FALSE_AUTHORITY,
        },
        "temporal_path": {
            "schema": COMPONENT_SCHEMA,
            "component_id": "tub_temporal_encoder_output2",
            "classification": (
                "official_snerv_t_temporal_encoder_output2_source_fixture_executed"
            ),
            "source_forward_fixture_replay_passed": True,
            "output_tensors_finite": all(
                bool(np.isfinite(value).all()) for value in temporal_arrays.values()
            ),
            "official_module_classes": {
                "encoder0": type(model.encoder[0]).__name__,
                "encoder1": type(model.encoder[1]).__name__,
                "encoder2": type(model.encoder[2]).__name__,
                "output2_decoder": type(model.decoder[model.decoder_len - 1]).__name__,
            },
            "decoder_len": int(model.decoder_len),
            "fc_hw": [int(model.fc_h), int(model.fc_w)],
            "output_shapes": _shape_map(temporal_arrays),
            "output_sha256": _hash_named_arrays(temporal_arrays),
            "closed_blockers": list(TUB_CLOSED_BY_FIXTURE_REPLAY),
            "blockers": list(TUB_PRESERVED_BLOCKERS),
            **FALSE_AUTHORITY,
        },
        "full_forward_equivalence": {
            "schema": COMPONENT_SCHEMA,
            "component_id": "tub_full_forward_equivalence",
            "classification": (
                "manual_tub_extraction_matches_official_snerv_t_forward_embed_list"
            ),
            "manual_replay_matches_official_forward": forward_passed,
            "max_abs_error": forward_error,
            "official_forward_sha256": _hash_named_arrays(full_forward_arrays),
            "manual_replay_sha256": _hash_named_arrays(manual_forward_arrays),
            "output_hashes_bit_identical": _hash_named_arrays(full_forward_arrays)
            == _hash_named_arrays(manual_forward_arrays),
            "bit_flip_falsification": forward_bit_flip_falsification,
            "bit_flip_falsification_passed": forward_bit_flip_falsification["passed"],
            "output_shapes": _shape_map(full_forward_arrays),
            "blockers": (
                []
                if forward_passed
                else ["snerv_official_tub_manual_forward_replay_mismatch"]
            ),
            **FALSE_AUTHORITY,
        },
        "frame_reconstruction_equivalence": {
            "schema": COMPONENT_SCHEMA,
            "component_id": "tub_mfu_hfr_frame_reconstruction_equivalence",
            "classification": (
                "manual_tub_mfu_hfr_path_plus_numpy_idwt_matches_official_snerv_t_img_out"
            ),
            "source_forward_frame_reconstruction_matches_official": frame_passed,
            "max_abs_error": frame_error,
            "official_forward_sha256": _hash_named_arrays(official_frame_arrays),
            "manual_reconstruction_sha256": _hash_named_arrays(manual_frame_arrays),
            "output_hashes_bit_identical": _hash_named_arrays(official_frame_arrays)
            == _hash_named_arrays(manual_frame_arrays),
            "output_shapes": _shape_map(official_frame_arrays),
            "portable_frame_reconstruction_metadata": manual[
                "frame_reconstruction_metadata"
            ],
            "closed_blockers": [
                TUB_FRAME_RECONSTRUCTION_BLOCKER
            ] if frame_passed else [],
            "blockers": [] if frame_passed else [TUB_FRAME_RECONSTRUCTION_BLOCKER],
            **FALSE_AUTHORITY,
        },
}


def _run_one_step_training_smoke(
    model: Any,
    current_t: Any,
    previous_t: Any,
    next_t: Any,
    *,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "schema": "snerv_official_tub_source_training_smoke.v1",
            "enabled": False,
            "step_count": 0,
            "loss_before": None,
            "loss_after": None,
        }
    import torch

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-7)
    optimizer.zero_grad(set_to_none=True)
    img_out, _embed_list, _dec_time, _img_yl, _yh_out = model(
        current_t,
        previous_t,
        next_t,
    )
    loss = torch.mean((img_out - current_t) ** 2)
    loss_before = float(loss.detach().cpu().item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        img_after, _embed_list, _dec_time, _img_yl, _yh_out = model(
            current_t,
            previous_t,
            next_t,
        )
        loss_after = float(torch.mean((img_after - current_t) ** 2).detach().cpu().item())
    return {
        "schema": "snerv_official_tub_source_training_smoke.v1",
        "enabled": True,
        "step_count": 1,
        "loss_before": loss_before,
        "loss_after": loss_after,
    }


def _checkpoint_not_requested_report() -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_LOAD_SCHEMA,
        "requested": False,
        "loaded": False,
        "state_dict_kind": None,
        "state_dict_source": None,
        "state_dict_key_count": 0,
        "loaded_key_count": 0,
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "blockers": ["snerv_official_trained_checkpoint_state_dict_not_loaded"],
        **FALSE_AUTHORITY,
    }


def _load_checkpoint_into_official_tub_model(
    model: Any,
    *,
    torch_module: Any,
    state_dict: Mapping[str, Any] | None,
    state_dict_path: str | Path | None,
    state_dict_kind: str,
) -> dict[str, Any]:
    if state_dict is not None and state_dict_path is not None:
        report = {
            **_checkpoint_not_requested_report(),
            "requested": True,
            "blockers": [
                "snerv_official_trained_checkpoint_state_dict_ambiguous_inputs"
            ],
        }
        raise _OfficialTubCheckpointLoadFailed(report)
    if state_dict is None and state_dict_path is None:
        return _checkpoint_not_requested_report()

    if state_dict_path is not None:
        resolved = Path(state_dict_path).expanduser().resolve(strict=False)
        try:
            raw_state = _load_checkpoint_state_dict_path(resolved, torch_module=torch_module)
        except _OfficialTubCheckpointLoadFailed:
            raise
        source = resolved.as_posix()
    else:
        raw_state = _coerce_checkpoint_state_dict_mapping(state_dict)
        source = "in_memory_official_trained_checkpoint_state_dict"

    expected = dict(model.state_dict())
    supplied = _coerce_checkpoint_state_dict_mapping(raw_state)
    missing_keys = sorted(key for key in expected if key not in supplied)
    unexpected_keys = sorted(key for key in supplied if key not in expected)
    shape_mismatches: list[dict[str, Any]] = []
    torch_state: dict[str, Any] = {}
    for key in sorted(key for key in supplied if key in expected):
        array = np.asarray(_state_value_array(supplied[key]))
        expected_shape = tuple(int(value) for value in expected[key].shape)
        observed_shape = tuple(int(value) for value in array.shape)
        if observed_shape != expected_shape:
            shape_mismatches.append(
                {
                    "key": key,
                    "expected_shape": list(expected_shape),
                    "observed_shape": list(observed_shape),
                }
            )
            continue
        torch_state[key] = torch_module.as_tensor(
            array,
            dtype=expected[key].dtype,
            device=expected[key].device,
        )

    blockers: list[str] = []
    if missing_keys:
        blockers.append(CHECKPOINT_LOAD_MISSING_KEYS_BLOCKER)
    if unexpected_keys:
        blockers.append(CHECKPOINT_LOAD_UNEXPECTED_KEYS_BLOCKER)
    if shape_mismatches:
        blockers.append(CHECKPOINT_LOAD_SHAPE_MISMATCH_BLOCKER)
    report = {
        "schema": CHECKPOINT_LOAD_SCHEMA,
        "requested": True,
        "loaded": False,
        "state_dict_kind": str(state_dict_kind),
        "state_dict_source": source,
        "state_dict_key_count": len(supplied),
        "expected_key_count": len(expected),
        "loaded_key_count": 0,
        "state_dict_sha256": _hash_state_dict_exact(supplied),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": shape_mismatches,
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }
    if blockers:
        raise _OfficialTubCheckpointLoadFailed(report)

    try:
        model.load_state_dict(torch_state, strict=True)
    except Exception as exc:  # pragma: no cover - validation should catch this.
        report = {
            **report,
            "failure": f"{type(exc).__name__}: {exc}",
            "blockers": [CHECKPOINT_LOAD_UNREADABLE_BLOCKER],
        }
        raise _OfficialTubCheckpointLoadFailed(report) from exc

    return {
        **report,
        "_state_dict": supplied,
        "loaded": True,
        "loaded_key_count": len(torch_state),
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "blockers": [],
    }


def _load_checkpoint_state_dict_path(
    path: Path,
    *,
    torch_module: Any,
) -> dict[str, Any]:
    if not path.is_file():
        report = {
            **_checkpoint_not_requested_report(),
            "requested": True,
            "state_dict_source": path.as_posix(),
            "blockers": [CHECKPOINT_LOAD_PATH_MISSING_BLOCKER],
        }
        raise _OfficialTubCheckpointLoadFailed(report)
    try:
        suffix = path.suffix.lower()
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as npz:
                return {
                    _checkpoint_npz_member_to_state_key(str(key)): np.asarray(npz[key])
                    for key in npz.files
                }
        if suffix == ".json":
            return _coerce_checkpoint_state_dict_mapping(
                json.loads(path.read_text(encoding="utf-8"))
            )
        loaded = torch_module.load(path, map_location="cpu")
        return _coerce_checkpoint_state_dict_mapping(loaded)
    except _OfficialTubCheckpointLoadFailed:
        raise
    except Exception as exc:
        report = {
            **_checkpoint_not_requested_report(),
            "requested": True,
            "state_dict_source": path.as_posix(),
            "failure": f"{type(exc).__name__}: {exc}",
            "blockers": [CHECKPOINT_LOAD_UNREADABLE_BLOCKER],
        }
        raise _OfficialTubCheckpointLoadFailed(report) from exc


def _coerce_checkpoint_state_dict_mapping(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        for nested_key in ("state_dict", "model_state_dict", "decoder_state_dict"):
            nested = raw.get(nested_key)
            if isinstance(nested, Mapping):
                raw = nested
                break
    if not isinstance(raw, Mapping):
        report = {
            **_checkpoint_not_requested_report(),
            "requested": True,
            "blockers": [CHECKPOINT_LOAD_UNREADABLE_BLOCKER],
        }
        raise _OfficialTubCheckpointLoadFailed(report)
    return {str(key): value for key, value in raw.items()}


def _checkpoint_npz_member_to_state_key(member_name: str) -> str:
    # _write_deterministic_state_npz stores each state value as "<key>.npy" so
    # arbitrary keys remain independent zip members.  np.load exposes those
    # member names directly; strip only that storage suffix.
    if member_name.endswith(".npy"):
        return member_name[: -len(".npy")]
    return member_name


def _supplied_checkpoint_state_artifact(
    checkpoint_load: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = Path(str(checkpoint_load.get("state_dict_source") or ""))
    if not path.is_file():
        return None
    state_dict = checkpoint_load.get("_state_dict")
    if not isinstance(state_dict, Mapping):
        return None
    state_dict_keys = sorted(str(key) for key in state_dict)
    return {
        "schema": "snerv_official_tub_supplied_state_dict_file.v1",
        "path": path.as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": _hash_bytes(path.read_bytes()),
        "member_count": len(state_dict_keys),
        "member_names": state_dict_keys,
        "state_dict_keys": state_dict_keys,
        "state_dict_sha256": _hash_state_dict_exact(state_dict),
        **FALSE_AUTHORITY,
    }


def _tub_fixture_replay_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not payload["graph_input_parity"]["graph_input_parity_passed"]:
        blockers.append(
            "snerv_official_tub_graph_inputs_only_not_full_source_forward_parity"
        )
    if not payload["portable_output2_fusion"][
        "portable_output2_fusion_receiver_mapping_proven"
    ]:
        blockers.extend(
            [
                "snerv_official_snerv_t_output2_fusion_source_forward_replay_missing",
                "snerv_official_tub_portable_output2_fusion_receiver_mapping_missing",
            ]
        )
    if not payload["full_forward_equivalence"]["manual_replay_matches_official_forward"]:
        blockers.append("snerv_official_tub_manual_forward_replay_mismatch")
    if not payload["frame_reconstruction_equivalence"][
        "source_forward_frame_reconstruction_matches_official"
    ]:
        blockers.append(TUB_FRAME_RECONSTRUCTION_BLOCKER)
    if not payload["temporal_path"]["output_tensors_finite"]:
        blockers.append("snerv_official_tub_temporal_encoder_output2_nonfinite")
    return _ordered_unique(blockers)


def _build_official_tub_trained_checkpoint_mapping_manifest(
    *,
    model_state_dict: Mapping[str, Any],
    decoder_len: int,
    source: str,
    state_dict_kind: str,
) -> dict[str, Any]:
    groups = _official_tub_checkpoint_group_prefixes(decoder_len)
    entries: list[dict[str, Any]] = []
    for key in sorted(str(key) for key in model_state_dict):
        group = _official_group_for_key(key, groups)
        if group is None:
            continue
        array = _state_value_array(model_state_dict[key])
        entries.append(
            {
                "key": key,
                "receiver_key": _receiver_key_for_official_key(key, decoder_len),
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
        _mapping_component_row(
            component_id="hfr",
            required_groups=("hfr_lh", "hfr_hl", "hfr_hh"),
            present_groups=present_groups,
            entries=entries,
            source_blocker=(
                "snerv_hfr_source_forward_replay_requires_upstream_torch_state_dict_mapping"
            ),
        ),
        _mapping_component_row(
            component_id="mfu",
            required_groups=(
                "mfu_upsample_mid",
                "mfu_rb_mid",
                "mfu_upsample_high",
                "mfu_rb_high",
            ),
            present_groups=present_groups,
            entries=entries,
            source_blocker=(
                "snerv_mfu_source_forward_replay_requires_upstream_torch_state_dict_mapping"
            ),
        ),
        _mapping_component_row(
            component_id="tub",
            required_groups=(
                "tub_temporal_encoder_1",
                "tub_temporal_encoder_2",
                "tub_output2_decoder",
            ),
            present_groups=present_groups,
            entries=entries,
            source_blocker=(
                "snerv_tub_full_source_forward_replay_requires_temporal_encoder_decoder_fusion_mapping"
            ),
        ),
    ]
    rows = {str(row["component_id"]): row for row in component_rows}
    hfr_proven = rows["hfr"]["trained_checkpoint_weight_mapping_proven"] is True
    mfu_proven = rows["mfu"]["trained_checkpoint_weight_mapping_proven"] is True
    tub_proven = rows["tub"]["trained_checkpoint_weight_mapping_proven"] is True
    mfu_hfr_proven = bool(hfr_proven and mfu_proven)
    mapping_verified = bool(mfu_hfr_proven and tub_proven)
    closed: list[str] = ["snerv_official_trained_checkpoint_state_dict_not_loaded"]
    if hfr_proven:
        closed.append("snerv_official_trained_checkpoint_hfr_weight_mapping_incomplete")
    if mfu_proven:
        closed.extend(
            [
                "snerv_official_trained_checkpoint_mfu_weight_mapping_incomplete",
                "snerv_official_mfu_native_receiver_activation_payload_not_upstream_weight_mapping",
            ]
        )
    if mfu_hfr_proven:
        closed.append("snerv_official_mfu_hfr_tub_weight_mapping_missing")
    if tub_proven:
        closed.extend(
            [
                "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
                "snerv_official_tub_encoder_decoder_weights_not_loaded",
                "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
                "snerv_official_tub_portable_output2_decoder_weight_mapping_missing",
            ]
        )
    if mapping_verified:
        closed.append("snerv_official_trained_checkpoint_state_dict_mapping_missing")
    blockers = [
        blocker
        for row in component_rows
        for blocker in row.get("blockers", ())
        if blocker not in closed
    ]
    return {
        "schema": TRAINED_CHECKPOINT_MAPPING_SCHEMA,
        "state_dict_kind": state_dict_kind,
        "state_dict_source": source,
        "state_dict_key_count": len(model_state_dict),
        "decoder_len": int(decoder_len),
        "decoder_len_source": "official_snerv_t_source_model",
        "state_dict_mapping_dialect": "upstream_official_snerv_t_state_dict",
        "official_trained_checkpoint_loaded": True,
        "official_hfr_trained_checkpoint_weight_mapping_proven": hfr_proven,
        "official_mfu_trained_checkpoint_weight_mapping_proven": mfu_proven,
        "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": mfu_hfr_proven,
        "official_tub_temporal_encoder_weight_mapping_proven": tub_proven,
        "official_tub_output2_decoder_weight_mapping_proven": tub_proven,
        "official_trained_checkpoint_state_dict_mapping_verified": mapping_verified,
        "state_dict_sha256": _hash_state_dict_exact(model_state_dict),
        "mapped_weight_key_count": len(entries),
        "mapped_weight_byte_count": int(sum(int(row["byte_count"]) for row in entries)),
        "mapped_weight_entries_sha256": _hash_weight_entries(entries),
        "weight_entries": entries,
        "activation_entries": [],
        "mapped_activation_key_count": 0,
        "component_rows": component_rows,
        "closed_campaign_blockers": _ordered_unique(closed),
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _mapping_manifest_has_checkpoint_export_lineage(manifest: Mapping[str, Any]) -> bool:
    kind = str(manifest.get("state_dict_kind") or "").strip()
    source = str(manifest.get("state_dict_source") or "").strip()
    if kind in {
        "official_trained_checkpoint_state_dict",
        "checkpoint_export_official_trained_checkpoint_state_dict",
        "checkpoint_export_native_mlx_receiver_state_dict",
    }:
        return True
    return bool(
        kind.startswith("checkpoint_export_")
        or source.startswith("export_snerv_checkpoint_archive")
        or source.startswith("snerv_checkpoint_archive_export")
    )


def _untrained_tub_source_fixture_mapping_manifest(
    *,
    state_dict: Mapping[str, Any],
    decoder_len: int,
) -> dict[str, Any]:
    return {
        "schema": TRAINED_CHECKPOINT_MAPPING_SCHEMA,
        "state_dict_kind": "official_snerv_t_source_fixture_initial_state_dict",
        "state_dict_source": "official_snerv_t_source_fixture_initial_state",
        "state_dict_key_count": len(state_dict),
        "decoder_len": int(decoder_len),
        "decoder_len_source": "official_snerv_t_source_model",
        "state_dict_mapping_dialect": "upstream_official_snerv_t_state_dict_untrained_fixture",
        "official_trained_checkpoint_loaded": False,
        "official_hfr_trained_checkpoint_weight_mapping_proven": False,
        "official_mfu_trained_checkpoint_weight_mapping_proven": False,
        "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": False,
        "official_tub_temporal_encoder_weight_mapping_proven": False,
        "official_tub_output2_decoder_weight_mapping_proven": False,
        "official_trained_checkpoint_state_dict_mapping_verified": False,
        "state_dict_sha256": _hash_state_dict_exact(state_dict),
        "mapped_weight_key_count": 0,
        "mapped_weight_byte_count": 0,
        "mapped_weight_entries_sha256": _hash_weight_entries([]),
        "weight_entries": [],
        "activation_entries": [],
        "mapped_activation_key_count": 0,
        "component_rows": [],
        "closed_campaign_blockers": [],
        "blockers": [
            "snerv_official_trained_checkpoint_state_dict_not_loaded",
            "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
            "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
            "snerv_official_tub_portable_output2_decoder_weight_mapping_missing",
            "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing",
        ],
        **FALSE_AUTHORITY,
    }


def _mapping_component_row(
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
    blockers = [source_blocker] if mapping_proven else [
        f"snerv_official_trained_checkpoint_{component_id}_weight_mapping_incomplete",
        source_blocker,
    ]
    if component_id == "tub" and not mapping_proven:
        blockers.extend(
            [
                "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
                "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
                "snerv_official_tub_portable_output2_decoder_weight_mapping_missing",
            ]
        )
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
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _official_tub_checkpoint_group_prefixes(decoder_len: int) -> dict[str, tuple[str, ...]]:
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


def _receiver_key_for_official_key(key: str, decoder_len: int) -> str:
    replacements = {
        f"decoder.{decoder_len}.": "hfr.lh.",
        f"decoder.{decoder_len + 1}.": "hfr.hl.",
        f"decoder.{decoder_len + 2}.": "hfr.hh.",
        f"decoder.{decoder_len + 3}.": "mfu.upsample_mid.",
        f"decoder.{decoder_len + 4}.": "mfu.rb_mid.",
        f"decoder.{decoder_len + 5}.": "mfu.upsample_high.",
        f"decoder.{decoder_len + 6}.": "mfu.rb_high.",
        "encoder.1.": "tub.temporal_encoder_prev.",
        "encoder.2.": "tub.temporal_encoder_next.",
        f"decoder.{decoder_len - 1}.": "tub.output2_decoder.",
    }
    out = str(key)
    for official, receiver in replacements.items():
        if out.startswith(official):
            out = receiver + out[len(official) :]
            break
    return (
        out.replace(".main.0.", ".input_conv.")
        .replace(".main.1.", ".residual_blocks.")
    )


def _manual_tub_source_replay(
    model: Any,
    current: Any,
    previous: Any,
    next_frame: Any,
) -> dict[str, Any]:
    import torch
    from model.layers import OutImg
    from pytorch_wavelets import DWT, DWT1D

    yl, _ = DWT(J=1, wave="haar", mode="periodization").cuda()(
        torch.cat([current, previous, next_frame], 0)
    )
    yl_norm = torch.as_tensor([yl.min(), yl.max()])
    embed = (yl - yl_norm[0]) / (yl_norm[1] - yl_norm[0])
    n, c, h, w = embed[0:2].shape
    embed_lv_p, _embed_hv_p = DWT1D(J=1, wave="haar", mode="periodization").cuda()(
        torch.cat([embed[0:1], embed[1:2]], 0)
        .reshape(n, c, h * w)
        .permute(2, 1, 0)
    )
    embed_lv_n, _embed_hv_n = DWT1D(J=1, wave="haar", mode="periodization").cuda()(
        torch.cat([embed[0:1], embed[2:3]], 0)
        .reshape(n, c, h * w)
        .permute(2, 1, 0)
    )

    embed_curr = model.encoder[0](embed[0:1])
    temporal_prev = model.encoder[1](
        (embed_lv_p.permute(2, 1, 0).reshape(1, c, h, w)) / 2.0
    )
    temporal_next = model.encoder[2](
        (embed_lv_n.permute(2, 1, 0).reshape(1, c, h, w)) / 2.0
    )
    temporal_concat = torch.cat([temporal_prev, temporal_next], 1)
    img_embed = [embed_curr, temporal_concat, yl_norm]
    embed_list = [img_embed]
    output = model.decoder[0](embed_curr)
    out_n, _out_c, out_h, out_w = output.shape
    output = (
        output.view(out_n, -1, model.fc_h, model.fc_w, out_h, out_w)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(out_n, -1, model.fc_h * out_h, model.fc_w * out_w)
    )
    decoder0_shuffled = output
    embed_list.append(output)
    emb_ch = temporal_concat.size(1) // 2
    output2_decoder_input = torch.cat(
        [temporal_concat[:, 0:emb_ch], temporal_concat[:, emb_ch:]],
        0,
    )
    output2 = model.decoder[model.decoder_len - 1](output2_decoder_input)
    output2_raw = output2
    out2_n, _out2_c, out2_h, out2_w = output2.shape
    output2 = (
        output2.view(out2_n, -1, model.fc_h, model.fc_w, out2_h, out2_w)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(out2_n, -1, model.fc_h * out2_h, model.fc_w * out2_w)
    )
    output2_shuffled = output2
    for idx, layer in enumerate(model.decoder[1 : model.decoder_len - 1]):
        if idx < 2:
            output = layer(output)
        elif idx == model.decoder_len - 3:
            output = layer(output, output2)
        else:
            output, output2 = layer(output, output2)
        embed_list.append(output)
    up1 = model.decoder[model.decoder_len + 3](embed_list[-3])
    unet1 = model.decoder[model.decoder_len + 4](
        torch.cat([up1, embed_list[-2]], dim=1)
    )
    unet1_up = model.decoder[model.decoder_len + 5](unet1)
    pyr_out = model.decoder[model.decoder_len + 6](
        torch.cat([unet1_up, embed_list[-1]], dim=1)
    )
    img_yl = OutImg(model.head_layer(pyr_out), model.out_bias)
    lh_out = model.decoder[model.decoder_len](pyr_out)
    hl_out = model.decoder[model.decoder_len + 1](pyr_out)
    hh_out = model.decoder[model.decoder_len + 2](pyr_out)
    yh_out = torch.stack([lh_out, hl_out, hh_out], dim=2)
    frame_reconstruction = official_tub_frame_reconstruction_numpy(
        _tensor_array(img_yl),
        _tensor_array(yh_out),
        yl_norm=tuple(float(v) for v in _tensor_array(yl_norm)),
    )
    prev_lowpass_over_2 = (
        embed_lv_p.permute(2, 1, 0).reshape(1, c, h, w) / 2.0
    )
    next_lowpass_over_2 = (
        embed_lv_n.permute(2, 1, 0).reshape(1, c, h, w) / 2.0
    )
    return {
        "lf_triplet": _tensor_array(yl),
        "normalized_lf": _tensor_array(embed),
        "prev_lowpass_over_2": _tensor_array(prev_lowpass_over_2),
        "next_lowpass_over_2": _tensor_array(next_lowpass_over_2),
        "embed_curr": _tensor_array(embed_curr),
        "temporal_encoder_prev": _tensor_array(temporal_prev),
        "temporal_encoder_next": _tensor_array(temporal_next),
        "temporal_encoder_concat": _tensor_array(temporal_concat),
        "decoder0_shuffled": _tensor_array(decoder0_shuffled),
        "output2_decoder_input": _tensor_array(output2_decoder_input),
        "output2_raw": _tensor_array(output2_raw),
        "output2_shuffled": _tensor_array(output2_shuffled),
        "final_decoder_output": _tensor_array(output),
        "mfu_up1": _tensor_array(up1),
        "mfu_unet1": _tensor_array(unet1),
        "mfu_unet1_up": _tensor_array(unet1_up),
        "mfu_pyr_out": _tensor_array(pyr_out),
        "img_yl": _tensor_array(img_yl),
        "yh_out": _tensor_array(yh_out),
        "yl_out": frame_reconstruction.yl_out,
        "frame_reconstruction": frame_reconstruction.frame,
        "frame_reconstruction_metadata": frame_reconstruction.as_jsonable_metadata(),
    }


@contextmanager
def _official_tub_import_context(official_root: Path) -> Iterable[None]:
    original_path = list(sys.path)
    original_dont_write_bytecode = sys.dont_write_bytecode
    sentinel = object()
    module_names = (
        "pytorch_wavelets",
        "matplotlib",
        "matplotlib.path",
        "model",
        "model.snerv_t",
        "model.layers",
        "model.residual_block",
    )
    saved = {name: sys.modules.get(name, sentinel) for name in module_names}
    for name in module_names:
        sys.modules.pop(name, None)

    wavelets = types.ModuleType("pytorch_wavelets")
    wavelets.DWT = _HaarDWT2
    wavelets.IDWT = _HaarIDWT2
    wavelets.DWT1D = _HaarDWT1D
    matplotlib = types.ModuleType("matplotlib")
    matplotlib_path = types.ModuleType("matplotlib.path")
    matplotlib_path.Path = object
    sys.modules["pytorch_wavelets"] = wavelets
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.path"] = matplotlib_path
    sys.dont_write_bytecode = True
    sys.path.insert(0, official_root.as_posix())
    try:
        yield
    finally:
        sys.path[:] = original_path
        sys.dont_write_bytecode = original_dont_write_bytecode
        for name in module_names:
            sys.modules.pop(name, None)
            value = saved[name]
            if value is not sentinel:
                sys.modules[name] = value


class _HaarDWT2:
    def __init__(self, J: int = 1, wave: str = "haar", mode: str = "periodization") -> None:
        _validate_wavelet_args(J=J, wave=wave, mode=mode)

    def cuda(self) -> _HaarDWT2:
        return self

    def __call__(self, x: Any) -> tuple[Any, list[Any]]:
        import torch

        arr = x
        ll = (
            arr[:, :, 0::2, 0::2]
            + arr[:, :, 0::2, 1::2]
            + arr[:, :, 1::2, 0::2]
            + arr[:, :, 1::2, 1::2]
        ) * 0.5
        lh = (
            arr[:, :, 0::2, 0::2]
            + arr[:, :, 0::2, 1::2]
            - arr[:, :, 1::2, 0::2]
            - arr[:, :, 1::2, 1::2]
        ) * 0.5
        hl = (
            arr[:, :, 0::2, 0::2]
            - arr[:, :, 0::2, 1::2]
            + arr[:, :, 1::2, 0::2]
            - arr[:, :, 1::2, 1::2]
        ) * 0.5
        hh = (
            arr[:, :, 0::2, 0::2]
            - arr[:, :, 0::2, 1::2]
            - arr[:, :, 1::2, 0::2]
            + arr[:, :, 1::2, 1::2]
        ) * 0.5
        return ll, [torch.stack([lh, hl, hh], dim=2)]


class _HaarDWT1D:
    def __init__(self, J: int = 1, wave: str = "haar", mode: str = "periodization") -> None:
        _validate_wavelet_args(J=J, wave=wave, mode=mode)

    def cuda(self) -> _HaarDWT1D:
        return self

    def __call__(self, x: Any) -> tuple[Any, Any]:
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        return (x[..., 0:1] + x[..., 1:2]) * inv_sqrt2, (
            x[..., 0:1] - x[..., 1:2]
        ) * inv_sqrt2


class _HaarIDWT2:
    def __init__(self, wave: str = "haar", mode: str = "periodization") -> None:
        if wave != "haar" or mode != "periodization":
            raise ValueError("SNeRV_T fixture only supports Haar periodization")

    def cuda(self) -> _HaarIDWT2:
        return self

    def __call__(self, coeffs: Sequence[Any]) -> Any:
        import torch

        yl = coeffs[0]
        yh = coeffs[1][0]
        lh, hl, hh = yh[:, :, 0], yh[:, :, 1], yh[:, :, 2]
        out = torch.empty(
            (yl.shape[0], yl.shape[1], yl.shape[2] * 2, yl.shape[3] * 2),
            dtype=yl.dtype,
            device=yl.device,
        )
        out[:, :, 0::2, 0::2] = (yl + lh + hl + hh) * 0.5
        out[:, :, 0::2, 1::2] = (yl + lh - hl - hh) * 0.5
        out[:, :, 1::2, 0::2] = (yl - lh + hl - hh) * 0.5
        out[:, :, 1::2, 1::2] = (yl - lh - hl + hh) * 0.5
        return out


def _validate_wavelet_args(*, J: int, wave: str, mode: str) -> None:
    if J != 1 or wave != "haar" or mode != "periodization":
        raise ValueError("SNeRV_T fixture only supports one-level Haar periodization")


def _source_pins(official_root: Path) -> dict[str, Any]:
    source = official_root / "model/snerv_t.py"
    req = official_root / "requirements.txt"
    return {
        "snerv_t_py": source.as_posix(),
        "snerv_t_py_sha256": _hash_bytes(source.read_bytes()) if source.is_file() else None,
        "requirements_txt": req.as_posix(),
        "requirements_txt_sha256": _hash_bytes(req.read_bytes()) if req.is_file() else None,
        "source_line_ranges": {
            "tub_graph_inputs": "model/snerv_t.py:125-136",
            "output2_fusion": "model/snerv_t.py:142-150",
            "temporal_decoder_loop": "model/snerv_t.py:152-159",
            "mfu_hfr_frame_reconstruction": "model/snerv_t.py:161-173",
        },
        "requirements_lines": {
            "torch": "requirements.txt:1",
            "pytorch_wavelets": "requirements.txt:13",
            "pywavelets": "requirements.txt:14",
        },
    }


def _positive_fixture(shape: Sequence[int], *, modulo: int) -> np.ndarray:
    values = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(tuple(shape))
    return (((values % modulo) + 1.0) / 64.0).astype(np.float64)


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


def _max_abs_error(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
) -> float:
    errors = [
        float(np.max(np.abs(np.asarray(left[name]) - np.asarray(right[name]))))
        for name in left
    ]
    return max(errors) if errors else 0.0


def _shape_map(arrays: Mapping[str, np.ndarray]) -> dict[str, list[int]]:
    return {name: [int(v) for v in np.asarray(array).shape] for name, array in arrays.items()}


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
    for key in sorted(str(key) for key in state_dict):
        array = _state_value_array(state_dict[key])
        h.update(str(key).encode("utf-8"))
        h.update(b"\0")
        h.update(_hash_array_exact(array).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _write_deterministic_state_npz(
    path: Path,
    state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    state_dict_keys = sorted(str(key) for key in state_dict)
    member_names: list[str] = []
    with zipfile.ZipFile(
        resolved,
        mode="w",
        compression=zipfile.ZIP_STORED,
    ) as zf:
        for key in state_dict_keys:
            array = np.ascontiguousarray(_state_value_array(state_dict[key]))
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            member_name = f"{key}.npy"
            member_names.append(member_name)
            info = zipfile.ZipInfo(member_name)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            zf.writestr(info, buffer.getvalue())
    return {
        "schema": "snerv_official_tub_source_state_dict_npz.v1",
        "path": resolved.as_posix(),
        "bytes": int(resolved.stat().st_size),
        "sha256": _hash_bytes(resolved.read_bytes()),
        "member_count": len(member_names),
        "member_names": member_names,
        "state_dict_keys": state_dict_keys,
        "state_dict_sha256": _hash_state_dict_exact(state_dict),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _state_dict_value_artifact_ready(artifact: Mapping[str, Any] | None) -> bool:
    if not isinstance(artifact, Mapping):
        return False
    path = Path(str(artifact.get("path") or ""))
    try:
        bytes_value = int(artifact.get("bytes") or 0)
        member_count = int(artifact.get("member_count") or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        str(artifact.get("path") or "").strip()
        and path.is_file()
        and bytes_value > 0
        and member_count > 0
        and len(str(artifact.get("sha256") or "")) == 64
    )


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


def _module_installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _ordered_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo-dir", default=DEFAULT_OFFICIAL_SNERV_REPO)
    parser.add_argument("--generated-utc", default=None)
    parser.add_argument(
        "--train-one-step",
        action="store_true",
        help=(
            "Run the deterministic one-step official SNeRV_T source fixture so "
            "the emitted mapping is value-bearing, still false-authority."
        ),
    )
    parser.add_argument("--write-json", type=Path, default=None)
    parser.add_argument("--write-state-dict-npz", type=Path, default=None)
    parser.add_argument("--official-trained-checkpoint-state-dict-path", type=Path, default=None)
    parser.add_argument(
        "--official-trained-checkpoint-state-dict-kind",
        default="official_trained_checkpoint_state_dict",
    )
    args = parser.parse_args(argv)
    if args.write_state_dict_npz is not None and not args.train_one_step:
        parser.error("--write-state-dict-npz requires --train-one-step")

    payload = build_snerv_official_tub_source_forward_replay_artifact(
        official_repo_dir=args.official_repo_dir,
        train_one_step=bool(args.train_one_step),
        output_state_dict_path=args.write_state_dict_npz,
        official_trained_checkpoint_state_dict_path=(
            args.official_trained_checkpoint_state_dict_path
        ),
        official_trained_checkpoint_state_dict_kind=(
            args.official_trained_checkpoint_state_dict_kind
        ),
        generated_utc=args.generated_utc,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.write_json is None:
        print(text)
    else:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_LOAD_MISSING_KEYS_BLOCKER",
    "CHECKPOINT_LOAD_PATH_MISSING_BLOCKER",
    "CHECKPOINT_LOAD_SHAPE_MISMATCH_BLOCKER",
    "CHECKPOINT_LOAD_UNEXPECTED_KEYS_BLOCKER",
    "DEFAULT_OFFICIAL_SNERV_REPO",
    "FALSE_AUTHORITY",
    "PYTORCH_WAVELETS_BLOCKER",
    "SCHEMA",
    "STATE_VALUE_ARTIFACT_BLOCKER",
    "TUB_CHECKPOINT_EXPORT_LINEAGE_BLOCKER",
    "TUB_CLOSED_BY_FIXTURE_REPLAY",
    "TUB_PRESERVED_BLOCKERS",
    "build_snerv_official_tub_source_forward_replay_artifact",
    "main",
]
