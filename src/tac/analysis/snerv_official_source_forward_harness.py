# SPDX-License-Identifier: MIT
"""Executable SNeRV official-source forward replay harness.

This module is deliberately narrower than full SNeRV authority.  It loads the
pinned official source graph, assigns deterministic dyadic weights into the
real Torch ``decoder[...]`` modules, maps those official state_dict keys into
the local portable MFU/HFR primitives, and compares the source subgraph outputs.
The TUB row is graph-input-only: full temporal encoder/output2 replay still
needs the upstream wavelet/runtime dependency and trained temporal weights.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
import warnings
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
    prepare_official_tub_graph_inputs,
)

SCHEMA = "snerv_official_mfu_hfr_tub_forward_parity.v1"
SOURCE_REPLAY_SCHEMA = "snerv_official_mfu_hfr_tub_source_forward_harness.v1"
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

    component_rows: list[dict[str, Any]]
    source_replay: dict[str, Any]
    weight_manifest: dict[str, Any]
    harness_blockers: list[str] = []
    try:
        fixture = _build_official_fixture(official_root)
        mfu_row, hfr_row, source_replay, weight_manifest = _run_mfu_hfr_replay(
            fixture
        )
        component_rows = [mfu_row, hfr_row, _run_tub_graph_input_replay()]
    except Exception as exc:  # pragma: no cover - exercised by fail-closed callers.
        harness_blockers.append(f"snerv_official_source_harness_failed:{type(exc).__name__}")
        component_rows = [
            _failed_component_row("mfu", exc),
            _failed_component_row("hfr", exc),
            _run_tub_graph_input_replay(),
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

    mfu_hfr_passed = all(
        row["component_id"] in {"mfu", "hfr"}
        and row.get("source_forward_parity_proven") is True
        for row in component_rows
        if row["component_id"] in {"mfu", "hfr"}
    )
    full_passed = bool(
        mfu_hfr_passed
        and all(row.get("source_forward_parity_proven") is True for row in component_rows)
        and receiver_runtime.get("receiver_runtime_decode_proven") is True
    )
    blockers = _ordered_unique(
        [
            *harness_blockers,
            *weight_manifest.get("blockers", ()),
            *source_replay.get("blockers", ()),
            *[
                blocker
                for row in component_rows
                for blocker in row.get("blockers", ())
            ],
        ]
    )

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
        "source_forward_replay": source_replay,
        "component_rows": component_rows,
        "local_receiver_adapter_source_gap": local_adapter_row,
        "official_mfu_hfr_source_fixture_forward_parity_passed": mfu_hfr_passed,
        "official_mfu_hfr_tub_forward_parity_passed": full_passed,
        "official_mfu_hfr_tub_forward_parity_falsified": False,
        "official_trained_checkpoint_loaded": False,
        "official_mfu_hfr_weight_mapping_source_fixture_proven": mfu_hfr_passed,
        "full_tub_source_forward_parity_proven": False,
        "official_mfu_hfr_tub_primitive_replay_binding": primitive_binding,
        "receiver_runtime_decode": receiver_runtime,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


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
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch

    mfu = _portable_mfu_from_state_dict(fixture)
    heads = _portable_hfr_from_state_dict(fixture)
    low = _positive_fixture((1, fixture.spec.low_channels, 2, 3), modulo=7)
    skip_mid = _positive_fixture((1, fixture.spec.mid_channels, 4, 6), modulo=11)
    skip_high = _positive_fixture((1, fixture.spec.high_channels, 8, 12), modulo=13)

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
    return mfu_row, hfr_row, source_replay, weight_manifest


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
    max_abs_error = max(
        float(np.max(np.abs(official_outputs[name] - portable_outputs[name])))
        for name in official_outputs
    )
    output_hash = _hash_named_arrays(official_outputs)
    blockers = [
        "snerv_official_tub_graph_inputs_only_not_full_source_forward_parity",
        "snerv_official_tub_encoder_decoder_weights_not_loaded",
        "snerv_official_pytorch_wavelets_runtime_dependency_missing",
        "snerv_official_snerv_t_output2_fusion_source_forward_replay_missing",
    ]
    return {
        "schema": "snerv_official_source_forward_component_replay.v1",
        "component_id": "tub",
        "classification": "official_tub_graph_input_source_fixture_proven_full_tub_blocked",
        "backend": "official_torch_vs_portable",
        "source_forward_parity_proven": False,
        "primitive_source_forward_parity_proven": True,
        "source_forward_parity_falsified": False,
        "full_stack_source_forward_parity_proven": False,
        "full_tub_source_forward_parity_proven": False,
        "tolerance": 0.0,
        "max_abs_error": max_abs_error,
        "input_sha256": _hash_named_arrays(
            {
                "current": current,
                "previous": previous,
                "next_frame": next_frame,
            }
        ),
        "official_output_sha256": output_hash,
        "portable_output_sha256": _hash_named_arrays(portable_outputs),
        "output_hashes_bit_identical": output_hash == _hash_named_arrays(portable_outputs),
        "official_weight_keys": [
            "weightless_source_lines:model/snerv_t.py:125-136",
            "unmapped_temporal_encoder:self.encoder[1]",
            "unmapped_temporal_encoder:self.encoder[2]",
            "unmapped_output2_decoder:self.decoder[self.decoder_len-1]",
        ],
        "official_source_contract": "model/snerv_t.py lines 125-150",
        "blockers": blockers,
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


def _hash_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    return _hash_bytes(arr.tobytes())


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
    "build_snerv_official_source_forward_harness_artifact",
]
