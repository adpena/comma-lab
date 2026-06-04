# SPDX-License-Identifier: MIT
"""Executable SNeRV_T TUB source-forward replay harness.

This harness burns down exactly one SNeRV TUB blocker: the upstream
``model/snerv_t.py`` temporal encoder plus ``output_2`` fusion path can be
executed as a source fixture.  It does not load a trained checkpoint, does not
prove a portable receiver mapping for those temporal weights, and does not
claim score authority.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import types
import warnings
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from tac.substrates.snerv_inverse_steg_carrier.official_tub import (
    OFFICIAL_SNERV_T_SOURCE_SHA,
    prepare_official_tub_graph_inputs,
)

SCHEMA = "snerv_official_tub_source_forward_replay.v1"
COMPONENT_SCHEMA = "snerv_official_tub_source_forward_component.v1"
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

TUB_CLOSED_BY_FIXTURE_REPLAY: tuple[str, ...] = (
    "snerv_official_tub_graph_inputs_only_not_full_source_forward_parity",
    "snerv_official_snerv_t_output2_fusion_source_forward_replay_missing",
)
TUB_PRESERVED_BLOCKERS: tuple[str, ...] = (
    "snerv_official_trained_checkpoint_state_dict_not_loaded",
    "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
    "snerv_official_tub_portable_temporal_encoder_output2_receiver_mapping_missing",
    "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing",
)
PYTORCH_WAVELETS_BLOCKER = "snerv_official_pytorch_wavelets_runtime_dependency_missing"


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
        payload = _run_source_fixture(official_root)
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
        and payload["full_forward_equivalence"]["manual_replay_matches_official_forward"]
        and payload["temporal_path"]["output_tensors_finite"]
    )
    return {
        **base,
        "source_forward_replay_executed": True,
        "official_tub_temporal_encoder_output2_source_fixture_replay_passed": replay_passed,
        "source_fixture_config": payload["source_fixture_config"],
        "source_fixture_scope": (
            "deterministic_official_source_fixture_not_trained_checkpoint"
        ),
        "source_fixture_not_training_config": True,
        "source_fixture_reason": (
            "small shape chosen to exercise upstream SNeRV_T temporal encoder, "
            "output_2 fusion, and five-stage temporal decoder semantics on CPU"
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
        "temporal_path": payload["temporal_path"],
        "full_forward_equivalence": payload["full_forward_equivalence"],
        "component_rows": [
            payload["graph_input_parity"],
            payload["temporal_path"],
            payload["full_forward_equivalence"],
        ],
        "closed_blockers": list(TUB_CLOSED_BY_FIXTURE_REPLAY) if replay_passed else [],
        "preserved_blockers": preserved,
        "blockers": preserved,
        "full_tub_source_forward_parity_proven": False,
        "source_forward_parity_proven": False,
        **FALSE_AUTHORITY,
    }


def _run_source_fixture(official_root: Path) -> dict[str, Any]:
    import torch

    cfg = TubFixtureConfig()
    torch.manual_seed(20260604)
    with _official_tub_import_context(official_root):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            snerv_t = importlib.import_module("model.snerv_t")
        model = snerv_t.SNeRV_T(cfg.to_namespace()).double().eval()

        current = _positive_fixture((1, 3, 32, 32), modulo=17)
        previous = current + 1.0 / 64.0
        next_frame = current + 1.0 / 32.0
        current_t = torch.from_numpy(current)
        previous_t = torch.from_numpy(previous)
        next_t = torch.from_numpy(next_frame)

        with torch.no_grad():
            manual = _manual_tub_source_replay(model, current_t, previous_t, next_t)
            img_out, embed_list, _dec_time, img_yl, yh_out = model(
                current_t,
                previous_t,
                next_t,
            )

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

    temporal_arrays = {
        "temporal_encoder_prev": manual["temporal_encoder_prev"],
        "temporal_encoder_next": manual["temporal_encoder_next"],
        "temporal_encoder_concat": manual["temporal_encoder_concat"],
        "output2_raw": manual["output2_raw"],
        "output2_shuffled": manual["output2_shuffled"],
        "final_decoder_output": manual["final_decoder_output"],
        "full_img_out": _tensor_array(img_out),
        "full_img_yl": _tensor_array(img_yl),
        "full_yh_out": _tensor_array(yh_out),
    }
    return {
        "source_fixture_config": asdict(cfg),
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
            "manual_replay_matches_official_forward": forward_error == 0.0,
            "max_abs_error": forward_error,
            "official_forward_sha256": _hash_named_arrays(full_forward_arrays),
            "manual_replay_sha256": _hash_named_arrays(manual_forward_arrays),
            "output_hashes_bit_identical": _hash_named_arrays(full_forward_arrays)
            == _hash_named_arrays(manual_forward_arrays),
            "output_shapes": _shape_map(full_forward_arrays),
            "blockers": [],
            **FALSE_AUTHORITY,
        },
    }


def _manual_tub_source_replay(
    model: Any,
    current: Any,
    previous: Any,
    next_frame: Any,
) -> dict[str, np.ndarray]:
    import torch
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
    output = model.decoder[0](embed_curr)
    out_n, _out_c, out_h, out_w = output.shape
    output = (
        output.view(out_n, -1, model.fc_h, model.fc_w, out_h, out_w)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(out_n, -1, model.fc_h * out_h, model.fc_w * out_w)
    )
    decoder0_shuffled = output
    emb_ch = temporal_concat.size(1) // 2
    output2 = model.decoder[model.decoder_len - 1](
        torch.cat([temporal_concat[:, 0:emb_ch], temporal_concat[:, emb_ch:]], 0)
    )
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
        "output2_raw": _tensor_array(output2_raw),
        "output2_shuffled": _tensor_array(output2_shuffled),
        "final_decoder_output": _tensor_array(output),
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
    parser.add_argument("--write-json", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = build_snerv_official_tub_source_forward_replay_artifact(
        official_repo_dir=args.official_repo_dir,
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
    "DEFAULT_OFFICIAL_SNERV_REPO",
    "FALSE_AUTHORITY",
    "PYTORCH_WAVELETS_BLOCKER",
    "SCHEMA",
    "TUB_CLOSED_BY_FIXTURE_REPLAY",
    "TUB_PRESERVED_BLOCKERS",
    "build_snerv_official_tub_source_forward_replay_artifact",
    "main",
]
