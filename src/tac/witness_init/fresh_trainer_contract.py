"""Trainer-side configuration and resume contract for FreSh initialization.

The spectral sweep itself is framework-neutral and lives in
``fresh_runtime``.  This module keeps the remaining trainer contract pure and
MLX-free: validate the typed argv surface, persist the selected initialization
through the canonical resume registry, and restore its derived along-tangent
frequency before directional features are built.

FreSh is init-only.  Its persisted state is therefore provenance for the
initial condition, not a controller that changes during the epoch loop.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.witness_init.fresh_frequency_shift import (
    inclusive_bias_width_grid,
    tangent_frequency_candidates,
)

FRESH_RESUME_PREFIX = "__fresh_"
FRESH_STATE_VERSION = 1
_SHA256_HEX_LENGTH = 64
MAX_FRESH_CANDIDATES = 256
FRESH_ARM_IDENTITY_ARGS = frozenset({"out_dir", "fresh_init", "fresh_init_control"})


@dataclass
class FreShInitState:
    """Mutable run-scoped state persisted by the canonical resume registry."""

    enabled: bool
    control: bool = False
    applied: bool = False
    candidate_index: int | None = None
    selected_freq_along: float | None = None
    selected_bias_k: float | None = None
    selected_mean_distance: float | None = None
    post_structured_mean_distance: float | None = None
    init_seconds: float | None = None
    selection_receipt_sha256: str | None = None
    reason: str | None = None

    def result_dict(self) -> dict[str, object]:
        """Return the stable JSON-facing initialization disposition."""

        return {
            "enabled": bool(self.enabled),
            "mode": "control" if self.control else ("select" if self.enabled else "off"),
            "applied": bool(self.applied),
            "candidate_index": self.candidate_index,
            "selected_freq_along": self.selected_freq_along,
            "selected_bias_k": self.selected_bias_k,
            "selected_mean_distance": self.selected_mean_distance,
            "post_structured_mean_distance": self.post_structured_mean_distance,
            "init_seconds": self.init_seconds,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "reason": self.reason,
        }


def matched_fresh_arm_config(args: Any) -> tuple[dict[str, object], str]:
    """Return the arm-invariant parsed config and its canonical SHA-256.

    Fixed-quality control/treatment runs may differ only in their output
    directory and in which mutually-exclusive FreSh arm bit is enabled. Every
    other parsed trainer argument is part of the comparison identity. This is
    captured before FreSh mutates ``freq_along`` to the selected value.
    """

    try:
        raw = vars(args)
    except TypeError as exc:
        raise ValueError("FreSh matched config requires an argparse-like object") from exc
    payload = {
        str(key): _json_config_value(value, path=str(key))
        for key, value in sorted(raw.items())
        if str(key) not in FRESH_ARM_IDENTITY_ARGS
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return payload, hashlib.sha256(encoded).hexdigest()


def fresh_training_target_sha256(fields: Mapping[str, object]) -> str:
    """Hash the exact active GT arrays that determine the training trajectory."""

    required = ("gt_f0", "gt_f1", "lstars", "margins", "gt_poses")
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(
            "FreSh target authority is missing field(s): " + ", ".join(missing)
        )
    digest = hashlib.sha256(b"tac.witness_init.fresh_training_targets.v1\0")
    expected_count: int | None = None
    for field_name in required:
        raw_sequence = fields[field_name]
        if isinstance(raw_sequence, (str, bytes, bytearray, Mapping)):
            raise ValueError(f"FreSh target authority {field_name} must be an array sequence")
        try:
            arrays = tuple(raw_sequence)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(
                f"FreSh target authority {field_name} must be an array sequence"
            ) from exc
        if not arrays:
            raise ValueError(f"FreSh target authority {field_name} must not be empty")
        if expected_count is None:
            expected_count = len(arrays)
        elif len(arrays) != expected_count:
            raise ValueError("FreSh target authority fields must have equal pair counts")
        digest.update(field_name.encode("utf-8") + b"\0")
        for index, raw_array in enumerate(arrays):
            array = np.asarray(raw_array)
            if array.dtype.hasobject:
                raise ValueError(
                    f"FreSh target authority {field_name}[{index}] must not have object dtype"
                )
            canonical = np.ascontiguousarray(array)
            digest.update(str(index).encode("ascii") + b"\0")
            digest.update(canonical.dtype.str.encode("ascii") + b"\0")
            digest.update(json.dumps(canonical.shape, separators=(",", ":")).encode("ascii") + b"\0")
            digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def validate_fresh_init_args(args: Any) -> int:
    """Validate the default-off FreSh argv contract and return candidate count.

    Validation is intentionally performed before MLX import or scorer loading.
    When the lever is off this is a strict no-op returning zero.
    """

    select_on = bool(getattr(args, "fresh_init", False))
    control_on = bool(getattr(args, "fresh_init_control", False))
    if select_on and control_on:
        raise ValueError("invalid FreSh init configuration: select and control modes are mutually exclusive")
    if not (select_on or control_on):
        return 0

    problems: list[str] = []
    if not bool(getattr(args, "self_orient", False)):
        problems.append("--fresh-init requires --self-orient")
    if str(getattr(args, "activation", "")) not in {"hosc", "wire"}:
        problems.append("--fresh-init requires periodic --activation hosc/wire")
    if not bool(getattr(args, "siren_init", False)):
        problems.append("--fresh-init requires --siren-init")
    if int(getattr(args, "n_dir_freqs", 0)) <= 0:
        problems.append("--fresh-init requires --n-dir-freqs > 0")
    if bool(getattr(args, "finer_bias_init", False)):
        problems.append("--fresh-init owns first-layer bias selection and conflicts with --finer-bias-init")
    if bool(getattr(args, "freeze_decoder_fit_codes", False)):
        problems.append("--fresh-init conflicts with --freeze-decoder-fit-codes")
    if bool(getattr(args, "seed_islands", False)):
        problems.append(
            "--fresh-init requires --no-seed-islands until candidate selection is routed through "
            "the pair-dependent seed composer"
        )
    if bool(getattr(args, "residual_mode", False)):
        problems.append(
            "--fresh-init requires residual mode OFF until candidate selection is routed through "
            "the pair-dependent bulk composer"
        )
    if bool(getattr(args, "lane_render_band", False)) and int(
        getattr(args, "lane_band_start_epoch", 300)
    ) <= 1:
        problems.append(
            "--fresh-init requires the lane render-band to start after epoch 0 until its "
            "pair-dependent composer is routed into candidate selection"
        )
    if str(getattr(args, "render_aa", "none")) not in {"none", "ipe"}:
        problems.append(
            "--fresh-init supports --render-aa none/ipe but not supersample candidate scoring"
        )

    positive_int_fields = (
        ("fresh_spectrum_size", "--fresh-spectrum-size"),
        ("fresh_sample_pairs", "--fresh-sample-pairs"),
    )
    for attr, flag in positive_int_fields:
        value = getattr(args, attr, None)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
            problems.append(f"{flag} must be a positive integer")

    positive_float_fields = (
        ("freq_along", "--freq-along"),
        ("freq_across", "--freq-across"),
        ("fresh_reference_freq_along", "--fresh-reference-freq-along"),
        ("fresh_tangent_deficit", "--fresh-tangent-deficit"),
    )
    for attr, flag in positive_float_fields:
        if not _finite_positive(getattr(args, attr, None)):
            problems.append(f"{flag} must be finite and > 0")
    if select_on and _finite_positive(getattr(args, "fresh_tangent_deficit", None)) and float(
        args.fresh_tangent_deficit
    ) <= 1.0:
        problems.append("--fresh-tangent-deficit must be > 1 for selection mode")

    try:
        biases = inclusive_bias_width_grid(
            getattr(args, "fresh_bias_k_min", None),
            getattr(args, "fresh_bias_k_max", None),
            getattr(args, "fresh_bias_k_step", None),
        )
    except (TypeError, ValueError) as exc:
        problems.append(f"invalid FreSh bias grid: {exc}")
        biases = ()
    if biases and biases[0] != 0.0:
        problems.append("--fresh-bias-k-min must be 0 so the exact cold-bias baseline is in the sweep")

    try:
        frequencies = tangent_frequency_candidates(
            getattr(args, "freq_along", None),
            reference_frequency=getattr(args, "fresh_reference_freq_along", None),
            tangent_deficit=getattr(args, "fresh_tangent_deficit", None),
        )
    except (TypeError, ValueError) as exc:
        problems.append(f"invalid FreSh frequency grid: {exc}")
        frequencies = ()

    candidate_count = 1 if control_on else len(frequencies) * len(biases)
    if candidate_count > MAX_FRESH_CANDIDATES:
        problems.append(
            f"FreSh grid has {candidate_count} candidates; maximum is {MAX_FRESH_CANDIDATES}"
        )
    if problems:
        raise ValueError("invalid --fresh-init configuration: " + "; ".join(problems))
    return candidate_count


def _json_config_value(value: object, *, path: str) -> object:
    """Normalize argparse values without silently stringifying unknown types."""

    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"FreSh matched config {path} must be finite")
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"FreSh matched config {path} must be finite")
        return result
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [
            _json_config_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if not isinstance(key, str):
                raise ValueError(f"FreSh matched config {path} has a non-string mapping key")
            normalized[key] = _json_config_value(item, path=f"{path}.{key}")
        return normalized
    raise ValueError(
        f"FreSh matched config {path} has unsupported type {type(value).__name__}"
    )


def fresh_state_arrays(
    state: FreShInitState,
    prefix: str = FRESH_RESUME_PREFIX,
) -> dict[str, np.ndarray]:
    """Serialize one init disposition for the canonical resume registry."""

    if not isinstance(state, FreShInitState):
        raise ValueError("state must be FreShInitState")
    if not state.enabled:
        return {}
    _validate_state(state)
    return {
        prefix + "version": np.asarray(FRESH_STATE_VERSION, np.int64),
        prefix + "enabled": np.asarray(1, np.int8),
        prefix + "control": np.asarray(int(state.control), np.int8),
        prefix + "applied": np.asarray(int(state.applied), np.int8),
        prefix + "candidate_index": np.asarray(
            -1 if state.candidate_index is None else state.candidate_index,
            np.int64,
        ),
        prefix + "selected_freq_along": np.asarray(
            _optional_float_sentinel(state.selected_freq_along),
            np.float64,
        ),
        prefix + "selected_bias_k": np.asarray(
            _optional_float_sentinel(state.selected_bias_k),
            np.float64,
        ),
        prefix + "selected_mean_distance": np.asarray(
            _optional_float_sentinel(state.selected_mean_distance),
            np.float64,
        ),
        prefix + "post_structured_mean_distance": np.asarray(
            _optional_float_sentinel(state.post_structured_mean_distance),
            np.float64,
        ),
        prefix + "init_seconds": np.asarray(
            _optional_float_sentinel(state.init_seconds),
            np.float64,
        ),
        prefix + "selection_receipt_sha256": np.asarray(
            state.selection_receipt_sha256 or "none"
        ),
        prefix + "reason": np.asarray(state.reason or "none"),
    }


def restore_fresh_state_from_cfg(
    state: FreShInitState,
    cfg: dict[str, Any],
    prefix: str = FRESH_RESUME_PREFIX,
) -> bool:
    """Restore FreSh provenance, returning False for a legacy/non-FreSh sidecar."""

    if prefix + "enabled" not in cfg:
        return False
    if int(cfg[prefix + "enabled"]) != 1:
        raise ValueError("FreSh resume state has a non-enabled sentinel")
    version = int(cfg.get(prefix + "version", -1))
    if version != FRESH_STATE_VERSION:
        raise ValueError(
            f"unsupported FreSh resume-state version {version}; expected {FRESH_STATE_VERSION}"
        )

    restored = FreShInitState(
        enabled=True,
        control=bool(int(cfg.get(prefix + "control", 0))),
        applied=bool(int(cfg.get(prefix + "applied", 0))),
        candidate_index=_restore_optional_int(cfg.get(prefix + "candidate_index", -1)),
        selected_freq_along=_restore_optional_float(cfg.get(prefix + "selected_freq_along", -1.0)),
        selected_bias_k=_restore_optional_float(cfg.get(prefix + "selected_bias_k", -1.0)),
        selected_mean_distance=_restore_optional_float(
            cfg.get(prefix + "selected_mean_distance", -1.0)
        ),
        post_structured_mean_distance=_restore_optional_float(
            cfg.get(prefix + "post_structured_mean_distance", -1.0)
        ),
        init_seconds=_restore_optional_float(cfg.get(prefix + "init_seconds", -1.0)),
        selection_receipt_sha256=_restore_optional_string(
            cfg.get(prefix + "selection_receipt_sha256", "none")
        ),
        reason=_restore_optional_string(cfg.get(prefix + "reason", "none")),
    )
    _validate_state(restored)
    state.enabled = restored.enabled
    state.control = restored.control
    state.applied = restored.applied
    state.candidate_index = restored.candidate_index
    state.selected_freq_along = restored.selected_freq_along
    state.selected_bias_k = restored.selected_bias_k
    state.selected_mean_distance = restored.selected_mean_distance
    state.post_structured_mean_distance = restored.post_structured_mean_distance
    state.init_seconds = restored.init_seconds
    state.selection_receipt_sha256 = restored.selection_receipt_sha256
    state.reason = restored.reason
    return True


def fresh_checkpoint_cfg_arrays(
    state: FreShInitState,
    args: Any | None = None,
) -> dict[str, np.ndarray]:
    """Return compact ``__cfg_*`` keys for EMA and resume checkpoint consumers."""

    if not state.enabled or not state.applied:
        return {
            "__cfg_fresh_init": np.asarray(0, np.int8),
            "__cfg_fresh_requested": np.asarray(int(state.enabled), np.int8),
            "__cfg_fresh_overwritten": np.asarray(int(state.enabled and not state.applied), np.int8),
        }
    _validate_state(state)
    out = {
        "__cfg_fresh_init": np.asarray(1, np.int8),
        "__cfg_fresh_control": np.asarray(int(state.control), np.int8),
        "__cfg_fresh_applied": np.asarray(int(state.applied), np.int8),
        "__cfg_fresh_candidate_index": np.asarray(
            -1 if state.candidate_index is None else state.candidate_index,
            np.int64,
        ),
        "__cfg_fresh_selected_freq_along": np.asarray(
            _optional_float_sentinel(state.selected_freq_along), np.float64
        ),
        "__cfg_fresh_selected_bias_k": np.asarray(
            _optional_float_sentinel(state.selected_bias_k), np.float64
        ),
        "__cfg_fresh_selected_mean_distance": np.asarray(
            _optional_float_sentinel(state.selected_mean_distance), np.float64
        ),
        "__cfg_fresh_post_structured_mean_distance": np.asarray(
            _optional_float_sentinel(state.post_structured_mean_distance), np.float64
        ),
        "__cfg_fresh_init_seconds": np.asarray(
            _optional_float_sentinel(state.init_seconds), np.float64
        ),
        "__cfg_fresh_selection_receipt_sha256": np.asarray(
            state.selection_receipt_sha256 or "none"
        ),
    }
    if args is not None:
        out.update({
            "__cfg_fresh_spectrum_size": np.asarray(int(args.fresh_spectrum_size), np.int64),
            "__cfg_fresh_sample_pairs": np.asarray(int(args.fresh_sample_pairs), np.int64),
            "__cfg_fresh_reference_freq_along": np.asarray(
                float(args.fresh_reference_freq_along), np.float64),
            "__cfg_fresh_tangent_deficit": np.asarray(float(args.fresh_tangent_deficit), np.float64),
            "__cfg_fresh_bias_k_min": np.asarray(float(args.fresh_bias_k_min), np.float64),
            "__cfg_fresh_bias_k_max": np.asarray(float(args.fresh_bias_k_max), np.float64),
            "__cfg_fresh_bias_k_step": np.asarray(float(args.fresh_bias_k_step), np.float64),
        })
    return out


def restore_fresh_checkpoint_before_features(
    args: Any,
    state: FreShInitState,
    cfg: dict[str, Any],
) -> bool:
    """Restore the selected frequency before directional features are allocated.

    A legacy or non-FreSh checkpoint returns False.  A FreSh checkpoint used
    with the lever disabled is left for the trainer's normal divergence guard
    to reject; this helper never silently turns a DSL lever on.
    """

    if "__cfg_fresh_init" not in cfg or int(cfg["__cfg_fresh_init"]) == 0:
        if state.enabled:
            state.applied = False
            state.reason = "overwritten_by_nonfresh_resume"
        return False
    if not state.enabled:
        return False
    selected_frequency = _restore_optional_float(
        cfg.get("__cfg_fresh_selected_freq_along", -1.0)
    )
    if selected_frequency is None or selected_frequency <= 0.0:
        raise ValueError(
            "FreSh checkpoint is missing a positive __cfg_fresh_selected_freq_along"
        )
    state.applied = bool(int(cfg.get("__cfg_fresh_applied", 1)))
    state.control = bool(int(cfg.get("__cfg_fresh_control", 0)))
    state.candidate_index = _restore_optional_int(cfg.get("__cfg_fresh_candidate_index", -1))
    state.selected_freq_along = selected_frequency
    state.selected_bias_k = _restore_optional_float(
        cfg.get("__cfg_fresh_selected_bias_k", -1.0)
    )
    state.selected_mean_distance = _restore_optional_float(
        cfg.get("__cfg_fresh_selected_mean_distance", -1.0)
    )
    state.post_structured_mean_distance = _restore_optional_float(
        cfg.get("__cfg_fresh_post_structured_mean_distance", -1.0)
    )
    state.init_seconds = _restore_optional_float(cfg.get("__cfg_fresh_init_seconds", -1.0))
    state.selection_receipt_sha256 = _restore_optional_string(
        cfg.get("__cfg_fresh_selection_receipt_sha256", "none")
    )
    state.reason = "restored_from_checkpoint"
    args.freq_along = selected_frequency
    return True


def load_checkpoint_cfg(path: str | Path) -> dict[str, Any]:
    """Read only ``__cfg_*``/``__fresh_*`` scalars from a resolved NPZ path."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"FreSh checkpoint config path is not a file: {source}")
    cfg: dict[str, Any] = {}
    with np.load(source, allow_pickle=False) as archive:
        for key in archive.files:
            if not (key.startswith("__cfg_") or key.startswith(FRESH_RESUME_PREFIX)):
                continue
            value = archive[key]
            cfg[key] = value.item() if value.size == 1 else value.tolist()
    return cfg


def _validate_state(state: FreShInitState) -> None:
    if state.applied:
        if state.candidate_index is None or state.candidate_index < 0:
            raise ValueError("applied FreSh state needs a non-negative candidate_index")
        if not _finite_positive(state.selected_freq_along):
            raise ValueError("applied FreSh state needs a positive selected_freq_along")
        if not _finite_nonnegative(state.selected_bias_k):
            raise ValueError("applied FreSh state needs a non-negative selected_bias_k")
        if not _finite_nonnegative(state.selected_mean_distance):
            raise ValueError("applied FreSh state needs a non-negative selected_mean_distance")
        if not _valid_sha256(state.selection_receipt_sha256):
            raise ValueError("applied FreSh state needs a 64-character receipt SHA-256")
    if state.post_structured_mean_distance is not None and not _finite_nonnegative(
        state.post_structured_mean_distance
    ):
        raise ValueError("post_structured_mean_distance must be finite and non-negative")
    if state.init_seconds is not None and not _finite_nonnegative(state.init_seconds):
        raise ValueError("init_seconds must be finite and non-negative")


def _finite_positive(value: object) -> bool:
    return _finite_nonnegative(value) and float(value) > 0.0


def _finite_nonnegative(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)) or value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0.0


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _optional_float_sentinel(value: float | None) -> float:
    return -1.0 if value is None else float(value)


def _restore_optional_float(value: object) -> float | None:
    number = float(value)
    return None if number < 0.0 else number


def _restore_optional_int(value: object) -> int | None:
    number = int(value)
    return None if number < 0 else number


def _restore_optional_string(value: object) -> str | None:
    text = str(value)
    return None if text == "none" else text


__all__ = [
    "FRESH_RESUME_PREFIX",
    "FRESH_STATE_VERSION",
    "FreShInitState",
    "fresh_checkpoint_cfg_arrays",
    "fresh_state_arrays",
    "load_checkpoint_cfg",
    "restore_fresh_checkpoint_before_features",
    "restore_fresh_state_from_cfg",
    "validate_fresh_init_args",
]
