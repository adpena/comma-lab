#!/usr/bin/env python
"""Build a resumable closed-form ELM/INR seed for #341's affine SDF head.

The mathematical and scope knobs come only from a frozen typed policy.  Run each stage until
it advances to the next one::

  .venv/bin/python tools/elm_inr_head_seed.py accumulate --policy <policy.json>
  .venv/bin/python tools/elm_inr_head_seed.py project    --policy <policy.json>
  .venv/bin/python tools/elm_inr_head_seed.py finalize   --policy <policy.json>
  .venv/bin/python tools/elm_inr_head_seed.py through-r-pair0 \
      --policy <diagnostic-pair0-policy.json> --seed-receipt <receipt.json> \
      --comparison-output <comparison.json>

The tool emits two checkpoint comparators: (1) the unregularized direct-global target-SSE
optimum for the receiver that actually ships and (2) the local POU field folded back to that
same receiver.  It never launches Gauss-Newton and never reports a contest score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import numpy as np


class _HiddenKwargs(TypedDict):
    n_hidden: int
    hidden_dim: int
    activation: str
    wire_w0: float
    wire_s0: float
    hosc_beta: float
    hosc_omega: float


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tac.boundary_math.elm_inr_head_solve import (  # noqa: E402
    RidgeSolveDiagnostics,
    StreamingPartitionedRidge,
    StreamingRidgeNormalEquations,
    atomic_save_npz,
    canonical_json,
    extract_levelset_hidden_numpy,
    partitioned_affine_predict,
    sha256_file,
    smoothed_ce_logit_targets,
    verify_seed_checkpoint_preservation,
    write_seed_checkpoint_atomic,
)
from tac.witness_control.resume_registry import (  # noqa: E402
    RESUME_REGISTRY_MANIFEST_KEY,
    ResumeIntegrityError,
    ResumeRegistry,
)
from tac.witness_dsl.elm_head_seed_policy import (  # noqa: E402
    CompiledElmHeadSeedPolicy,
    ElmHeadSeedScope,
    compile_elm_head_seed_policy,
)

PROBE_DIR = REPO / "experiments/results/basin_finisher_probe_20260707"
DEFAULT_PARAMS = PROBE_DIR / "ema_best_snapshot.npz"
DEFAULT_LABELS = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_OUTPUT_DIR = REPO / "experiments/results/elm_inr_head_seed_20260712"
CANONICAL_PAIR_COUNT = 600
STATE_PREFIX = "__ehs_"
NON_AUTHORITY = (
    "engineering-only affine-head seed and faithful-slice through-R diagnostic; no contest score "
    "claim; upstream/evaluate.py on exact archive bytes remains owed"
)


def _scalar(archive: Mapping[str, np.ndarray], key: str) -> Any:
    value = np.asarray(archive[key])
    return value.item() if value.shape == () else value


class RealHeadSeedContext:
    """Exact #341 checkpoint/features/label custody, streamed one pair at a time."""

    def __init__(self, params_path: Path, feature_state_path: Path, labels_path: Path) -> None:
        from tac.boundary_math.lever_b_generator import self_orientation_directional_feats
        from tac.boundary_math.lever_b_levelset_generator import (
            CurveletBankConfig,
            build_coords,
            curvelet_directional_B,
            curvelet_feats,
        )

        self.params_path = params_path.resolve()
        self.feature_state_path = feature_state_path.resolve()
        self.labels_path = labels_path.resolve()
        for path in (self.params_path, self.feature_state_path, self.labels_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        with np.load(self.params_path, allow_pickle=True) as source:
            self.params = {
                key: np.asarray(source[key], np.float32)
                for key in source.files
                if not key.startswith("__")
            }
            self.cfg = {
                key: np.array(source[key], copy=True)
                for key in source.files
                if key.startswith("__")
            }
        required_params = {
            "code",
            "in_proj.weight",
            "in_proj.bias",
            "film.weight",
            "film.bias",
            "out_sdf.weight",
            "out_sdf.bias",
        }
        missing = sorted(required_params - self.params.keys())
        if missing:
            raise KeyError(f"checkpoint lacks required arrays: {missing}")
        self.n_classes = int(self.params["out_sdf.weight"].shape[0])
        self.hidden_dim = int(self.params["out_sdf.weight"].shape[1])

        with np.load(self.feature_state_path, allow_pickle=False) as feature_state:
            self.feature_pairs = [int(value) for value in np.asarray(feature_state["pairs"]).tolist()]
            self.argmax_state = np.asarray(feature_state["argmax_prev"], np.int8)
        if self.argmax_state.shape[0] != len(self.feature_pairs):
            raise ValueError("feature-state pairs/argmax_prev length mismatch")
        if len(set(self.feature_pairs)) != len(self.feature_pairs):
            raise ValueError("feature-state pair list contains duplicates")
        self._feature_pair_index = {pair: index for index, pair in enumerate(self.feature_pairs)}

        with np.load(self.labels_path, allow_pickle=False) as labels_archive:
            self.n_label_pairs = int(np.asarray(labels_archive["n_pairs"]))
            source_labels = np.asarray(labels_archive["lstars"])
            if source_labels.dtype.kind not in "iu":
                raise ValueError(
                    f"labels lstars must use an integer dtype, got {source_labels.dtype}"
                )
            if source_labels.size:
                label_min = int(source_labels.min())
                label_max = int(source_labels.max())
                if label_min < 0 or label_max >= self.n_classes:
                    raise ValueError(
                        f"labels must be class ids in [0,{self.n_classes}), got "
                        f"[{label_min},{label_max}]"
                    )
            if self.n_classes <= np.iinfo(np.uint8).max + 1:
                compact_dtype = np.dtype(np.uint8)
            elif self.n_classes <= np.iinfo(np.uint16).max + 1:
                compact_dtype = np.dtype(np.uint16)
            else:
                raise ValueError(
                    f"n_classes={self.n_classes} exceeds the compact uint16 label contract"
                )
            self.labels_source_dtype = str(source_labels.dtype)
            # copy=False preserves an already-canonical compact cache; an older int64 cache is
            # validated before narrowing, then retained as uint8/uint16 instead of ~944 MB int64.
            self.labels = source_labels.astype(compact_dtype, copy=False)
        if self.labels.shape[0] != self.n_label_pairs:
            raise ValueError("labels n_pairs/lstars length mismatch")

        self.height, self.width = (
            int(value) for value in np.asarray(self.cfg["__render_hw"]).tolist()
        )
        if self.labels.shape[1:] != (self.height, self.width):
            raise ValueError(
                f"label geometry {self.labels.shape[1:]} != checkpoint render geometry "
                f"{(self.height, self.width)}"
            )
        self.coords = build_coords(self.height, self.width)
        bank = CurveletBankConfig(
            n_scales=int(_scalar(self.cfg, "__bank_n_scales")),
            n_orient0=int(_scalar(self.cfg, "__bank_n_orient0")),
            f0=float(_scalar(self.cfg, "__bank_f0")),
            base=float(_scalar(self.cfg, "__bank_base")),
            n_iso=int(_scalar(self.cfg, "__bank_n_iso")),
        )
        max_bank_frequency = float(_scalar(self.cfg, "__cfg_max_bank_freq"))
        bank_matrix = curvelet_directional_B(
            bank,
            max_freq=None if max_bank_frequency < 0.0 else max_bank_frequency,
        )
        self.curvelet_features = curvelet_feats(self.coords, bank_matrix).astype(np.float32)
        self._self_orientation_directional_feats = self_orientation_directional_feats
        self.hidden_kwargs: _HiddenKwargs = {
            "n_hidden": int(_scalar(self.cfg, "__cfg_n_hidden")),
            "hidden_dim": int(_scalar(self.cfg, "__cfg_hidden_dim")),
            "activation": str(_scalar(self.cfg, "__cfg_activation")),
            "wire_w0": float(_scalar(self.cfg, "__cfg_wire_w0")),
            "wire_s0": float(_scalar(self.cfg, "__cfg_wire_s0")),
            "hosc_beta": float(_scalar(self.cfg, "__cfg_hosc_beta")),
            "hosc_omega": float(_scalar(self.cfg, "__cfg_hosc_omega")),
        }
        if self.hidden_dim != self.hidden_kwargs["hidden_dim"]:
            raise ValueError("out_sdf input width disagrees with checkpoint hidden_dim")

    def features_for_pair(self, pair: int) -> np.ndarray:
        try:
            state_index = self._feature_pair_index[int(pair)]
        except KeyError as exc:
            raise KeyError(f"pair {pair} absent from feature state") from exc
        directional = self._self_orientation_directional_feats(
            self.coords,
            np.asarray(self.argmax_state[state_index], np.int64),
            n_freqs=int(_scalar(self.cfg, "__cfg_n_dir_freqs")),
            freq_across=float(_scalar(self.cfg, "__cfg_freq_across")),
            freq_along=float(_scalar(self.cfg, "__cfg_freq_along")),
        ).astype(np.float32)
        features = np.concatenate([self.curvelet_features, directional], axis=-1).astype(np.float32)
        expected = int(_scalar(self.cfg, "__cfg_in_feat"))
        if features.shape != (self.height * self.width, expected):
            raise ValueError(f"reconstructed features have shape {features.shape}, expected (*,{expected})")
        return features

    def iter_pair_chunks(
        self,
        pair: int,
        *,
        pixel_chunk: int,
        smoothing: float,
        target_temperature: float,
    ):
        features = self.features_for_pair(pair)
        labels = np.asarray(self.labels[int(pair)]).reshape(-1)
        code_row = self.params["code"][2 * int(pair) + 1]
        for start in range(0, features.shape[0], int(pixel_chunk)):
            stop = min(start + int(pixel_chunk), features.shape[0])
            hidden = extract_levelset_hidden_numpy(
                self.params,
                features[start:stop],
                code_row,
                **self.hidden_kwargs,
            )
            targets = smoothed_ce_logit_targets(
                labels[start:stop],
                n_classes=self.n_classes,
                smoothing=smoothing,
                temperature=target_temperature,
            ).reshape(stop - start, self.n_classes)
            yield hidden, targets, self.coords[start:stop]


@dataclass
class ElmHeadSeedResumeController:
    """Run-local event-mode controller registered with the canonical resume apparatus."""

    expected_config_sha256: str
    stage: str = ""
    cursor: int = 0
    payload_sha256: str = ""
    event_mode: bool = True

    def state_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        if not self.stage or not self.payload_sha256:
            raise ResumeIntegrityError("ELM resume controller cannot persist incomplete state")
        return {
            prefix + "schema": np.asarray("elm_head_seed_resume.v1"),
            prefix + "stage": np.asarray(self.stage),
            prefix + "cursor": np.asarray(self.cursor, np.int64),
            prefix + "config_sha256": np.asarray(self.expected_config_sha256),
            prefix + "payload_sha256": np.asarray(self.payload_sha256),
        }

    def restore_from_cfg(self, prefix: str, cfg: dict) -> bool:
        names = ("schema", "stage", "cursor", "config_sha256", "payload_sha256")
        present = [prefix + name in cfg for name in names]
        if not any(present):
            return False
        if not all(present):
            raise ResumeIntegrityError("ELM resume controller state is partial/truncated")
        if str(cfg[prefix + "schema"]) != "elm_head_seed_resume.v1":
            raise ResumeIntegrityError("ELM resume controller schema changed")
        if str(cfg[prefix + "config_sha256"]) != self.expected_config_sha256:
            raise ResumeIntegrityError("ELM resume controller config SHA-256 changed")
        self.stage = str(cfg[prefix + "stage"])
        self.cursor = int(cfg[prefix + "cursor"])
        self.payload_sha256 = str(cfg[prefix + "payload_sha256"])
        return True


def _resume_registry(controller: ElmHeadSeedResumeController) -> ResumeRegistry:
    registry = ResumeRegistry()
    registry.register("elm_head_seed", STATE_PREFIX, controller)
    return registry


def _array_payload_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Content digest for every non-registry array in a resume sidecar."""

    digest = hashlib.sha256()
    for key in sorted(arrays):
        if key == RESUME_REGISTRY_MANIFEST_KEY or key.startswith(STATE_PREFIX):
            continue
        value = np.ascontiguousarray(np.asarray(arrays[key]))
        digest.update(key.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _selected_pairs(
    context: RealHeadSeedContext,
    policy: CompiledElmHeadSeedPolicy,
) -> tuple[list[int], bool]:
    available = list(context.feature_pairs)
    if policy.scope is ElmHeadSeedScope.FULL_P600:
        if available != list(range(CANONICAL_PAIR_COUNT)):
            raise ValueError("full_p600 policy requires feature-state pairs exactly 0..599")
        if context.n_label_pairs < CANONICAL_PAIR_COUNT:
            raise ValueError("full_p600 policy requires at least 600 label pairs")
        return available, False
    if policy.pair_limit is None:
        raise AssertionError("compiled diagnostic policy lacks pair_limit")
    if policy.pair_limit > len(available):
        raise ValueError(
            f"diagnostic pair_limit {policy.pair_limit} exceeds {len(available)} feature-state pairs"
        )
    return available[: policy.pair_limit], True


def _scope_name(
    pair_count: int,
    diagnostic_slice: bool,
    grid_shape: tuple[int, int],
) -> str:
    scope = f"diagnostic_p{pair_count}" if diagnostic_slice else "full_p600"
    return f"{scope}_g{grid_shape[0]}x{grid_shape[1]}"


def _verify_declared_digest(path: Path, declared: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != declared:
        raise RuntimeError(
            f"{label} custody mismatch: policy declares {declared}, exact bytes hash to {actual}"
        )
    return actual


def _implementation_provenance() -> dict[str, Any]:
    """Hash every production source that defines the compiled ELM policy/solve contract."""

    paths = (
        Path(__file__).resolve(),
        REPO / "src/tac/boundary_math/elm_inr_head_solve.py",
        REPO / "src/tac/witness_dsl/elm_head_seed_policy.py",
        REPO / "src/tac/canonical_equations/elm_inr_affine_head_seed_20260712.py",
    )
    return {
        "schema": "elm_head_seed_implementation.v1",
        "files": {
            str(path.relative_to(REPO)): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        },
    }


def _git_provenance() -> dict[str, Any]:
    """Return truthful current git identity without treating dirty state as stable config."""

    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    entries = [line for line in status.splitlines() if line]
    return {
        "available": True,
        "head": head,
        "dirty": bool(entries),
        "dirty_entry_count": len(entries),
        "porcelain_v1_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _runtime_provenance(invocation_history: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "numpy_version": np.__version__,
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.system(),
            "release": platform.release(),
            "cpu_count": os.cpu_count(),
        },
        "hardware_axis": (
            "[local-CPU engineering] NumPy fp64 normal equations + NumPy fp32 witness "
            "checkpoint; no frozen scorer or contest score in this solve"
        ),
        "git": _git_provenance(),
        "invocations": invocation_history,
        "timing_custody": (
            "Each elapsed_seconds value covers that completed CLI invocation from run_stage "
            "entry through preparation of its final state payload, or through final checkpoint "
            "and receipt-payload preparation for finalize; the last fsync/JSON write is excluded. "
            "Interrupted/crashed partial invocations are not invented."
        ),
    }


def _completed_invocation_record(
    *,
    index: int,
    argv: list[str],
    started_utc: str,
    started_monotonic: float,
    stage_before: str,
    stage_after: str,
    cursor_before: int,
    cursor_after: int,
    processed_pairs: int,
    resumed: bool,
    timing_boundary: str,
) -> dict[str, Any]:
    elapsed = time.monotonic() - started_monotonic
    return {
        "index": index,
        "argv": argv,
        "command": shlex.join(argv),
        "started_utc": started_utc,
        "elapsed_seconds": float(elapsed),
        "timing_boundary": timing_boundary,
        "stage_before": stage_before,
        "stage_after": stage_after,
        "cursor_before": cursor_before,
        "cursor_after": cursor_after,
        "processed_pairs": processed_pairs,
        "resumed_from_existing_state": resumed,
    }


def _build_config(
    args: argparse.Namespace,
    policy: CompiledElmHeadSeedPolicy,
    context: RealHeadSeedContext,
    pairs: list[int],
    diagnostic_slice: bool,
    source_sha256: str,
    feature_state_sha256: str,
    labels_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "elm_inr_head_seed_state_v3_20260712",
        "implementation": _implementation_provenance(),
        "typed_policy": {
            "path": str(policy.policy_path),
            "file_sha256": policy.policy_file_sha256,
            "manifest_sha256": policy.policy_manifest_sha256,
            "manifest": policy.manifest,
        },
        "source_checkpoint": str(context.params_path),
        "source_checkpoint_sha256": source_sha256,
        "feature_state": str(context.feature_state_path),
        "feature_state_sha256": feature_state_sha256,
        "labels": str(context.labels_path),
        "labels_sha256": labels_sha256,
        "labels_bytes": context.labels_path.stat().st_size,
        "labels_source_dtype": context.labels_source_dtype,
        "labels_dtype": str(context.labels.dtype),
        "labels_resident_nbytes": int(context.labels.nbytes),
        "selected_pairs": pairs,
        "pair_count": len(pairs),
        "diagnostic_slice": diagnostic_slice,
        "hidden_dim": context.hidden_dim,
        "n_classes": context.n_classes,
        "grid_shape": list(policy.grid_shape),
        "local_and_fold_ridge": policy.ridge,
        "direct_global_target_comparator_ridge": 0.0,
        "pinv_rcond": policy.pinv_rcond,
        "label_smoothing": policy.label_smoothing,
        "target_temperature": policy.target_temperature,
        "pixel_chunk": policy.pixel_chunk,
        "output_dir": str(args.output_dir.resolve()),
        "target": "centered finite label-smoothed categorical log-probabilities",
        "head_scope": "out_sdf.weight/out_sdf.bias only",
        "resume_registry": {
            "helper": "tac.witness_control.resume_registry.ResumeRegistry",
            "controller": "elm_head_seed",
            "prefix": STATE_PREFIX,
            "manifest_key": RESUME_REGISTRY_MANIFEST_KEY,
            "event_mode": True,
        },
        "non_authority": NON_AUTHORITY,
    }


def _config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def _state_arrays(
    *,
    config: Mapping[str, Any],
    stage: str,
    cursor: int,
    local: StreamingPartitionedRidge,
    direct_target: StreamingRidgeNormalEquations,
    local_beta: np.ndarray | None = None,
    global_fold: StreamingRidgeNormalEquations | None = None,
    local_fit_sse: float = 0.0,
    local_fit_elements: int = 0,
    invocation_history: list[dict[str, Any]] | None = None,
    done_receipt: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    config_json = canonical_json(config)
    arrays: dict[str, np.ndarray] = {
        "__config_json": np.asarray(config_json),
        "__config_sha256": np.asarray(hashlib.sha256(config_json.encode("utf-8")).hexdigest()),
        "__local_fit_sse": np.asarray(float(local_fit_sse), np.float64),
        "__local_fit_elements": np.asarray(int(local_fit_elements), np.int64),
        "__invocation_history_json": np.asarray(
            canonical_json({"invocations": list(invocation_history or [])})
        ),
    }
    arrays.update(local.state_dict("local"))
    arrays.update(direct_target.state_dict("direct_target"))
    if local_beta is not None:
        arrays["local_beta"] = np.asarray(local_beta, np.float64)
    if global_fold is not None:
        arrays.update(global_fold.state_dict("global_fold"))
    if done_receipt is not None:
        arrays["__done_receipt_json"] = np.asarray(canonical_json(done_receipt))

    controller = ElmHeadSeedResumeController(
        expected_config_sha256=_config_sha256(config),
        stage=stage,
        cursor=int(cursor),
        payload_sha256=_array_payload_sha256(arrays),
    )
    registry_arrays = _resume_registry(controller).state_arrays()
    if RESUME_REGISTRY_MANIFEST_KEY not in registry_arrays:
        raise ResumeIntegrityError("ELM event-mode state did not stamp the canonical resume manifest")
    arrays.update(registry_arrays)
    return arrays


def _load_state(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def _state_cfg(state: Mapping[str, np.ndarray]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for key, raw in state.items():
        value = np.asarray(raw)
        cfg[key] = value.item() if value.shape == () else value
    return cfg


def _restore_state(
    path: Path,
    expected_config: Mapping[str, Any],
) -> tuple[
    str,
    int,
    StreamingPartitionedRidge,
    StreamingRidgeNormalEquations,
    np.ndarray | None,
    StreamingRidgeNormalEquations | None,
    float,
    int,
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    state = _load_state(path)
    expected_sha = _config_sha256(expected_config)
    stored_config = json.loads(str(np.asarray(state["__config_json"])))
    if canonical_json(stored_config) != canonical_json(expected_config):
        raise RuntimeError("resume refused: custody/config changed from persisted state")
    if str(np.asarray(state["__config_sha256"])) != expected_sha:
        raise ResumeIntegrityError("resume config JSON/hash pair is internally inconsistent")

    controller = ElmHeadSeedResumeController(expected_config_sha256=expected_sha)
    report = _resume_registry(controller).restore(_state_cfg(state))
    if not report.restored.get("elm_head_seed", False):
        raise ResumeIntegrityError("resume refused: ELM controller state did not restore")
    actual_payload_sha = _array_payload_sha256(state)
    if actual_payload_sha != controller.payload_sha256:
        raise ResumeIntegrityError(
            "resume refused: ELM sufficient-statistic payload SHA-256 changed or was truncated"
        )

    local = StreamingPartitionedRidge.from_state_dict(state, "local")
    direct_target = StreamingRidgeNormalEquations.from_state_dict(state, "direct_target")
    local_beta = np.asarray(state["local_beta"], np.float64) if "local_beta" in state else None
    global_fold = (
        StreamingRidgeNormalEquations.from_state_dict(state, "global_fold")
        if "global_fold__gram" in state
        else None
    )
    done_receipt = (
        json.loads(str(np.asarray(state["__done_receipt_json"])))
        if "__done_receipt_json" in state
        else None
    )
    invocation_history_payload = json.loads(str(np.asarray(state["__invocation_history_json"])))
    invocation_history = list(invocation_history_payload["invocations"])
    return (
        controller.stage,
        controller.cursor,
        local,
        direct_target,
        local_beta,
        global_fold,
        float(np.asarray(state["__local_fit_sse"])),
        int(np.asarray(state["__local_fit_elements"])),
        invocation_history,
        done_receipt,
    )


def _save_state(path: Path, **kwargs: Any) -> None:
    atomic_save_npz(path, _state_arrays(**kwargs), compressed=True)


def _diagnostic_dict(value: RidgeSolveDiagnostics) -> dict[str, Any]:
    condition: float | None = value.condition_number if np.isfinite(value.condition_number) else None
    return {
        "rank": value.rank,
        "dimension": value.dimension,
        "condition_number": condition,
        "condition_number_nonfinite": not np.isfinite(value.condition_number),
        "sample_count": value.sample_count,
        "weight_sum": value.weight_sum,
        "ridge": value.ridge,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON using the profiler's file+directory fsync atomic-replace pattern."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _refuse_transient_output(path: Path) -> None:
    resolved = str(path.resolve())
    if "pytest-of-" in resolved or "pytest-" in resolved:
        return
    transient_prefixes = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/folders/")
    if any(resolved.startswith(prefix) for prefix in transient_prefixes):
        raise ValueError(f"persisted ELM state/output cannot use transient path {resolved!r}")


def _prepare(args: argparse.Namespace):
    # Compile the frozen policy before opening any model/cache input.
    policy = compile_elm_head_seed_policy(args.policy)
    feature_state_path = args.feature_state or (PROBE_DIR / f"feats_state_{args.tag}.npz")
    input_paths = {
        "source checkpoint": args.params.resolve(),
        "feature state": feature_state_path.resolve(),
        "labels": args.labels.resolve(),
    }
    for path in input_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    source_sha = _verify_declared_digest(
        input_paths["source checkpoint"],
        policy.policy.source_checkpoint_sha256,
        "source checkpoint",
    )
    feature_sha = _verify_declared_digest(
        input_paths["feature state"],
        policy.policy.feature_state_sha256,
        "feature state",
    )
    # This is intentionally a real full-file digest on every invocation.  Path/size/mtime are
    # not content custody and cannot authorize resume of a 5 GB scorer-label cache.
    labels_sha = _verify_declared_digest(
        input_paths["labels"],
        policy.policy.labels_sha256,
        "labels",
    )
    context = RealHeadSeedContext(
        input_paths["source checkpoint"],
        input_paths["feature state"],
        input_paths["labels"],
    )
    checkpoint_temperature = float(_scalar(context.cfg, "__cfg_softmax_temp"))
    if checkpoint_temperature != policy.target_temperature:
        raise RuntimeError(
            "typed target_temperature does not exactly match source checkpoint "
            f"__cfg_softmax_temp ({policy.target_temperature} != {checkpoint_temperature})"
        )
    pairs, diagnostic_slice = _selected_pairs(context, policy)
    config = _build_config(
        args,
        policy,
        context,
        pairs,
        diagnostic_slice,
        source_sha,
        feature_sha,
        labels_sha,
    )
    scope = _scope_name(len(pairs), diagnostic_slice, policy.grid_shape)
    state_path = args.state or (args.output_dir / f"elm_head_seed_state_{scope}.npz")
    _refuse_transient_output(args.output_dir)
    _refuse_transient_output(state_path)
    return policy, context, pairs, diagnostic_slice, config, scope, state_path


def _checkpoint_custody(source: Path, output: Path) -> dict[str, Any]:
    preservation = verify_seed_checkpoint_preservation(source, output)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "preservation": preservation,
    }


def _gauss_newton_handoff(
    *,
    diagnostic_slice: bool,
    feature_state_path: Path,
    declared_feature_state_sha256: str,
    tag: str,
    direct_checkpoint: Path,
    fold_checkpoint: Path,
) -> tuple[list[str], dict[str, Any] | None, dict[str, Any]]:
    """Emit #341 commands only for full-P plus the tag's exact canonical feature state."""

    actual = feature_state_path.resolve()
    expected = (PROBE_DIR / f"feats_state_{tag}.npz").resolve()
    custody = {
        "actual_feature_state": str(actual),
        "expected_feature_state_for_tag": str(expected),
        "exact_path_match": actual == expected,
        "declared_feature_state_sha256": declared_feature_state_sha256,
        "actual_feature_state_sha256": None,
        "sha256_match": False,
        "tag": tag,
    }
    if diagnostic_slice:
        return [], {
            "code": "DIAGNOSTIC_SCOPE_NONPROMOTABLE",
            "message": (
                "No #341 solve command is emitted for a diagnostic slice. Compile a full_p600 "
                "typed policy; any authorized command must spell --k-pairs 600 explicitly."
            ),
        }, custody
    if actual != expected:
        return [], {
            "code": "FEATURE_STATE_TAG_CUSTODY_MISMATCH",
            "message": (
                "Full-P coefficients are not sufficient for #341 handoff: --feature-state must "
                f"resolve exactly to {expected} for --tag {tag!r}; got {actual}. No command emitted."
            ),
        }, custody
    actual_sha256 = sha256_file(actual)
    custody["actual_feature_state_sha256"] = actual_sha256
    custody["sha256_match"] = actual_sha256 == declared_feature_state_sha256
    if actual_sha256 != declared_feature_state_sha256:
        return [], {
            "code": "FEATURE_STATE_SHA256_MISMATCH",
            "message": (
                "The canonical #341 feature-state path changed bytes after policy compilation: "
                f"declared {declared_feature_state_sha256}, rehashed {actual_sha256}. "
                "No command emitted."
            ),
        }, custody
    return [
        (
            ".venv/bin/python tools/quadratic_basin_finisher_probe.py solve "
            f"--params {direct_checkpoint} --tag {tag} --mask head --k-pairs 600"
        ),
        (
            ".venv/bin/python tools/quadratic_basin_finisher_probe.py solve "
            f"--params {fold_checkpoint} --tag {tag} --mask head --k-pairs 600"
        ),
    ], None, custody


def _validate_done_receipt(context: RealHeadSeedContext, done_receipt: Mapping[str, Any]) -> None:
    receipt_path = Path(str(done_receipt["receipt"]))
    if not receipt_path.is_file() or sha256_file(receipt_path) != done_receipt["receipt_sha256"]:
        raise RuntimeError("done state receipt is missing or its SHA-256 changed")
    for key in ("direct_global_checkpoint", "pou_fold_checkpoint"):
        row = done_receipt[key]
        path = Path(str(row["path"]))
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"done state {key} is missing or its SHA-256 changed")
        verify_seed_checkpoint_preservation(context.params_path, path)


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO / candidate
    return candidate.resolve()


def _file_custody(
    path: str | Path,
    *,
    role: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    resolved = _resolve_repo_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual_sha256 = sha256_file(resolved)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{role} SHA-256 mismatch: declared {expected_sha256}, actual {actual_sha256}"
        )
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "sha256_match": expected_sha256 is None or actual_sha256 == expected_sha256,
        "role": role,
    }


def _load_pair0_seed_receipt(
    receipt_path: Path,
    *,
    context: RealHeadSeedContext,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    """Validate the ELM receipt and exact direct/POU bytes before scorer measurement."""

    custody = _file_custody(receipt_path, role="ELM seed receipt declaring candidate custody")
    payload = json.loads(_resolve_repo_path(receipt_path).read_text(encoding="utf-8"))
    if payload.get("schema") != "elm_inr_head_seed_receipt_v3_20260712":
        raise RuntimeError("pair0 comparison requires the v3 provenance-bearing ELM receipt")
    if payload.get("diagnostic_slice") is not True or payload.get("selected_pairs") != [0]:
        raise RuntimeError("pair0 comparison requires an exact diagnostic selected_pairs=[0] receipt")
    declared_inputs = {
        "source_checkpoint_sha256": config["source_checkpoint_sha256"],
        "feature_state_sha256": config["feature_state_sha256"],
        "labels_sha256": config["labels_sha256"],
    }
    for key, expected in declared_inputs.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"ELM seed receipt {key}={payload.get(key)!r} does not match compiled policy {expected!r}"
            )
    if _resolve_repo_path(payload["source_checkpoint"]) != context.params_path:
        raise RuntimeError("ELM seed receipt source checkpoint path differs from the measured baseline")

    checkpoints: dict[str, Path] = {}
    for name, receipt_key in (
        ("direct_global", "direct_global_checkpoint"),
        ("pou_fold", "pou_fold_checkpoint"),
    ):
        row = payload[receipt_key]
        checkpoint = _resolve_repo_path(row["path"])
        _file_custody(
            checkpoint,
            role=f"{name} ELM candidate checkpoint",
            expected_sha256=row["sha256"],
        )
        verify_seed_checkpoint_preservation(context.params_path, checkpoint)
        checkpoints[name] = checkpoint
    return payload, custody, checkpoints


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _render_pair0_through_r(
    context: RealHeadSeedContext,
    checkpoint: Path,
    features: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Canonical NumPy deploy forward followed by the settled torch camera-R operator."""

    from experiments.train_witness_realized_through_R_mlx import _torch_R_to_camera_uint8
    from tac.boundary_math.lever_b_levelset_generator import (
        int8_dequant_params,
        levelset_rgb_forward_numpy,
    )

    started = time.monotonic()
    with np.load(checkpoint, allow_pickle=True) as archive:
        params = {
            key: np.asarray(archive[key], np.float32)
            for key in archive.files
            if not key.startswith("__")
        }
        cfg = {
            key: np.array(archive[key], copy=True)
            for key in archive.files
            if key.startswith("__")
        }
    deploy = int8_dequant_params(params)
    rgb, _phi = levelset_rgb_forward_numpy(
        deploy,
        features,
        deploy["code"][1],
        **context.hidden_kwargs,
        n_classes=context.n_classes,
        softmax_temp=float(_scalar(cfg, "__cfg_softmax_temp")),
        chroma=bool(int(_scalar(cfg, "__cfg_chroma"))),
    )
    frame = _torch_R_to_camera_uint8(rgb.reshape(context.height, context.width, 3))
    return frame, float(time.monotonic() - started)


def _dependency_versions() -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for distribution in (
        "numpy",
        "torch",
        "safetensors",
        "scipy",
        "psutil",
        "segmentation-models-pytorch",
        "timm",
        "av",
    ):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _through_r_source_custody() -> dict[str, dict[str, dict[str, Any]]]:
    """Hash every source and environment manifest needed to reproduce pair0 through-R."""

    source_files = (
        REPO / "upstream/evaluate.py",
        REPO / "upstream/modules.py",
        REPO / "upstream/frame_utils.py",
        REPO / "experiments/train_witness_realized_through_R_mlx.py",
        REPO / "src/tac/boundary_math/seg_core.py",
        REPO / "src/tac/boundary_math/lever_b_levelset_generator.py",
        REPO / "src/tac/boundary_math/lever_b_generator.py",
    )
    environment_manifests = (REPO / "pyproject.toml", REPO / "uv.lock")
    return {
        "upstream_and_runtime_sources": {
            str(path.relative_to(REPO)): _file_custody(
                path,
                role="source implementing evaluator/scorer/R/witness path",
            )
            for path in source_files
        },
        "environment_manifests": {
            str(path.relative_to(REPO)): _file_custody(
                path,
                role="Python dependency/environment lock manifest",
            )
            for path in environment_manifests
        },
    }


def run_through_r_pair0(args: argparse.Namespace) -> int:
    """Remeasure baseline/direct/POU on pair0 and atomically emit standalone custody."""

    total_started = time.monotonic()
    started_utc = datetime.now(UTC).isoformat()
    if args.seed_receipt is None or args.comparison_output is None:
        raise RuntimeError("through-r-pair0 requires --seed-receipt and --comparison-output")
    policy, context, pairs, diagnostic_slice, config, _scope, _state_path = _prepare(args)
    preparation_seconds = float(time.monotonic() - total_started)
    if not diagnostic_slice or policy.pair_limit != 1 or pairs != [0]:
        raise RuntimeError("through-r-pair0 requires a typed diagnostic policy selecting pair 0 only")
    canonical_feature_state = (PROBE_DIR / f"feats_state_{args.tag}.npz").resolve()
    if context.feature_state_path != canonical_feature_state:
        raise RuntimeError(
            "through-r-pair0 requires the tag's exact canonical feature-state path; "
            f"expected {canonical_feature_state}, got {context.feature_state_path}"
        )
    seed_payload, seed_receipt_custody, candidate_paths = _load_pair0_seed_receipt(
        args.seed_receipt,
        context=context,
        config=config,
    )
    checkpoint_paths = {
        "baseline": context.params_path,
        "direct_global": candidate_paths["direct_global"],
        "pou_fold": candidate_paths["pou_fold"],
    }
    checkpoint_custody = {
        "baseline": _file_custody(
            context.params_path,
            role="unmodified source/baseline checkpoint",
            expected_sha256=config["source_checkpoint_sha256"],
        ),
        "direct_global": _file_custody(
            checkpoint_paths["direct_global"],
            role="direct-global ELM checkpoint",
            expected_sha256=seed_payload["direct_global_checkpoint"]["sha256"],
        ),
        "pou_fold": _file_custody(
            checkpoint_paths["pou_fold"],
            role="POU-fold ELM checkpoint",
            expected_sha256=seed_payload["pou_fold_checkpoint"]["sha256"],
        ),
    }
    if time.monotonic() - total_started > args.max_seconds:
        raise RuntimeError("through-r-pair0 exceeded --max-seconds during custody/preparation")

    features = context.features_for_pair(0)
    target = np.asarray(context.labels[0])
    rendered: dict[str, np.ndarray] = {}
    render_seconds: dict[str, float] = {}
    for name, checkpoint in checkpoint_paths.items():
        if time.monotonic() - total_started > args.max_seconds:
            raise RuntimeError(f"through-r-pair0 exceeded --max-seconds before {name} render")
        rendered[name], render_seconds[name] = _render_pair0_through_r(
            context,
            checkpoint,
            features,
        )

    from experiments.train_witness_realized_through_R_mlx import cpu_verdict_d_seg_argmax_batch
    from tac.boundary_math.seg_core import DEFAULT_VIDEO, SEGNET_CKPT, load_real_segnet

    scorer_load_started = time.monotonic()
    segnet = load_real_segnet("cpu")
    scorer_load_seconds = float(time.monotonic() - scorer_load_started)
    if time.monotonic() - total_started > args.max_seconds:
        raise RuntimeError("through-r-pair0 exceeded --max-seconds before scorer batch")
    names = list(checkpoint_paths)
    scorer_started = time.monotonic()
    d_segs, argmax = cpu_verdict_d_seg_argmax_batch(
        segnet,
        [rendered[name] for name in names],
        [target for _name in names],
    )
    scorer_batch_seconds = float(time.monotonic() - scorer_started)
    if time.monotonic() - total_started > args.max_seconds:
        raise RuntimeError("through-r-pair0 exceeded --max-seconds during scorer batch")

    measurements: dict[str, Any] = {}
    for index, name in enumerate(names):
        realized = np.asarray(argmax[index])
        flip_count = int(np.count_nonzero(realized != target))
        exact_d_seg = float(flip_count / target.size)
        if exact_d_seg != float(d_segs[index]):
            raise AssertionError("scorer helper d_seg disagrees with realized argmax popcount")
        measurements[name] = {
            "d_seg": exact_d_seg,
            "flip_count": flip_count,
            "pixel_count": int(target.size),
            "rendered_camera_uint8_sha256": _array_sha256(rendered[name]),
            "realized_segnet_argmax_sha256": _array_sha256(realized),
            "checkpoint_load_render_and_R_seconds": render_seconds[name],
        }

    input_custody = {
        "typed_policy": _file_custody(
            policy.policy_path,
            role="typed diagnostic policy",
            expected_sha256=policy.policy_file_sha256,
        ),
        "seed_receipt": seed_receipt_custody,
        "feature_state": _file_custody(
            context.feature_state_path,
            role="canonical self-orientation feature state",
            expected_sha256=config["feature_state_sha256"],
        ),
        "gt_cache": _file_custody(
            context.labels_path,
            role="consumed GT SegNet argmax target cache",
            expected_sha256=config["labels_sha256"],
        ),
        "source_video": _file_custody(
            DEFAULT_VIDEO,
            role="GT-cache lineage source video; recorded but not decoded by this measurement",
        ),
        "segnet_weights": _file_custody(
            SEGNET_CKPT,
            role="frozen CPU-torch SegNet scorer weights consumed by this measurement",
        ),
        "checkpoints": checkpoint_custody,
        **_through_r_source_custody(),
    }
    direct_delta = measurements["direct_global"]["d_seg"] - measurements["baseline"]["d_seg"]
    pou_delta = measurements["pou_fold"]["d_seg"] - measurements["baseline"]["d_seg"]
    if direct_delta > 0.0 and pou_delta > 0.0:
        formulation_verdict = (
            "NO-GO for this pair0 smoothed-CE ELM seed formulation because both candidates "
            "worsen d_seg; diagnostic only, not an ELM-family or full-P verdict"
        )
    elif direct_delta < 0.0 or pou_delta < 0.0:
        formulation_verdict = (
            "At least one pair0 candidate improves d_seg; diagnostic evidence only and full-P "
            "receiver-closed verification remains owed"
        )
    else:
        formulation_verdict = (
            "Pair0 is neutral or mixed under exact d_seg; diagnostic only and no family/full-P "
            "verdict is authorized"
        )
    invocation_argv = list(args._invocation_argv)
    invocation_cwd = str(
        Path(getattr(args, "_invocation_cwd", Path.cwd())).expanduser().resolve()
    )
    execution_provenance = _runtime_provenance([])
    execution_provenance.update(
        {
            "hardware_axis": (
                "[macOS-CPU advisory] NumPy-fp32 deploy forward + torch camera-R + one frozen "
                "CPU-torch SegNet batch; not contest-CPU/CUDA authority"
            ),
            "timing_custody": (
                "preparation_and_initial_custody_seconds begins at standalone command entry. "
                "Each per-checkpoint time covers checkpoint load, canonical NumPy render, and "
                "torch camera-R. scorer_load_seconds covers frozen SegNet construction and exact "
                "weight load. scorer_batch_seconds covers one batched frozen-SegNet preprocess, "
                "forward, argmax, and host transfer for all three frames; no per-checkpoint scorer "
                "timing is inferred. Final JSON serialization, file fsync, replace, and parent-dir "
                "fsync are excluded."
            ),
            "dependencies": _dependency_versions(),
            "argv": invocation_argv,
            "command": shlex.join(invocation_argv),
            "cwd": invocation_cwd,
            "repo_root": str(REPO),
        }
    )
    comparison = {
        "schema": "elm_head_seed_through_r_pair0_comparison.v2",
        "created_utc": started_utc,
        "scope": "diagnostic_pair0_only",
        "pair": 0,
        "axis": (
            "[macOS-CPU advisory] NON-PROMOTABLE: NumPy-fp32 deploy forward + torch camera-R "
            "+ frozen CPU-torch SegNet"
        ),
        "score_claim": False,
        "upstream_evaluate_score_run": False,
        "gauss_newton_run": False,
        "measurements": measurements,
        "comparison": {
            "direct_delta_d_seg_vs_baseline": direct_delta,
            "pou_fold_delta_d_seg_vs_baseline": pou_delta,
            "direct_equals_pou_fold": (
                measurements["direct_global"]["d_seg"] == measurements["pou_fold"]["d_seg"]
            ),
            "verdict": formulation_verdict,
        },
        "input_custody": input_custody,
        "implementation": _implementation_provenance(),
        "execution_provenance": execution_provenance,
        "timings": {
            "preparation_and_initial_custody_seconds": preparation_seconds,
            "scorer_load_seconds": scorer_load_seconds,
            "scorer_batch_seconds": scorer_batch_seconds,
            "scorer_batch_size": 3,
            "total_seconds_through_receipt_payload_preparation": None,
            "boundary_note": (
                "Per-checkpoint times cover checkpoint load, canonical NumPy render, and torch "
                "camera-R. scorer_batch_seconds is one measured frozen-SegNet batch over all "
                "three frames; no per-checkpoint scorer time is inferred or retained. Final JSON "
                "serialization/fsync is excluded from total_seconds."
            ),
            "max_seconds": float(args.max_seconds),
        },
        "reproduction": {
            "command": shlex.join(invocation_argv),
            "cwd": invocation_cwd,
            "repo_root": str(REPO),
            "required_exact_input_sha256": {
                "policy": policy.policy_file_sha256,
                "seed_receipt": seed_receipt_custody["sha256"],
                "feature_state": config["feature_state_sha256"],
                "gt_cache": config["labels_sha256"],
                "baseline_checkpoint": checkpoint_custody["baseline"]["sha256"],
                "direct_checkpoint": checkpoint_custody["direct_global"]["sha256"],
                "pou_fold_checkpoint": checkpoint_custody["pou_fold"]["sha256"],
                "segnet_weights": input_custody["segnet_weights"]["sha256"],
                "source_video": input_custody["source_video"]["sha256"],
                "upstream_and_runtime_sources": {
                    path: row["sha256"]
                    for path, row in input_custody["upstream_and_runtime_sources"].items()
                },
                "environment_manifests": {
                    path: row["sha256"]
                    for path, row in input_custody["environment_manifests"].items()
                },
            },
        },
    }
    payload_elapsed = float(time.monotonic() - total_started)
    if payload_elapsed > args.max_seconds:
        raise RuntimeError(
            "through-r-pair0 exceeded --max-seconds before durable receipt emission"
        )
    comparison["timings"]["total_seconds_through_receipt_payload_preparation"] = payload_elapsed
    output = args.comparison_output.resolve()
    _refuse_transient_output(output)
    _atomic_write_json(output, comparison)
    print(
        json.dumps(
            {
                "stage": "through_r_pair0_done",
                "comparison_output": str(output),
                "comparison_output_sha256": sha256_file(output),
                "measurements": measurements,
                "score_claim": False,
            }
        )
    )
    return 0


def run_stage(args: argparse.Namespace) -> int:
    invocation_started = time.monotonic()
    invocation_started_utc = datetime.now(UTC).isoformat()
    invocation_argv = list(
        getattr(
            args,
            "_invocation_argv",
            [sys.executable, str(Path(__file__).resolve()), args.stage],
        )
    )
    policy, context, pairs, diagnostic_slice, config, scope, state_path = _prepare(args)
    grid_shape = policy.grid_shape
    resumed_from_existing_state = state_path.exists()

    if resumed_from_existing_state:
        (
            stage,
            cursor,
            local,
            direct_target,
            local_beta,
            global_fold,
            local_fit_sse,
            local_fit_elements,
            invocation_history,
            done_receipt,
        ) = _restore_state(state_path, config)
    else:
        if args.stage != "accumulate":
            raise RuntimeError(f"state {state_path} does not exist; start with accumulate")
        stage, cursor = "accumulate", 0
        local = StreamingPartitionedRidge(
            context.hidden_dim,
            context.n_classes,
            grid_shape=grid_shape,
            ridge=policy.ridge,
            pinv_rcond=policy.pinv_rcond,
        )
        # This comparator is intentionally unregularized: it is the target-SSE optimum among
        # all single affine heads the current receiver can ship.
        direct_target = StreamingRidgeNormalEquations(
            context.hidden_dim,
            context.n_classes,
            ridge=0.0,
            pinv_rcond=policy.pinv_rcond,
        )
        local_beta = None
        global_fold = None
        local_fit_sse = 0.0
        local_fit_elements = 0
        invocation_history: list[dict[str, Any]] = []
        done_receipt = None
        _save_state(
            state_path,
            config=config,
            stage=stage,
            cursor=cursor,
            local=local,
            direct_target=direct_target,
            invocation_history=invocation_history,
        )

    if stage == "done":
        if done_receipt is None:
            raise RuntimeError("done state lacks output custody")
        _validate_done_receipt(context, done_receipt)
        print(json.dumps({"stage": "done", "state": str(state_path), **done_receipt}))
        return 0
    if stage != args.stage:
        raise RuntimeError(f"state expects stage {stage!r}; invoked {args.stage!r}")

    stage_before = stage
    cursor_before = cursor
    started = time.monotonic()
    processed = 0

    def bounded() -> bool:
        return bool(
            processed
            and (
                (args.max_pairs_per_invocation > 0 and processed >= args.max_pairs_per_invocation)
                or time.monotonic() - started >= args.max_seconds
            )
        )

    if stage == "accumulate":
        while cursor < len(pairs) and not bounded():
            pair = pairs[cursor]
            for hidden, targets, coords in context.iter_pair_chunks(
                pair,
                pixel_chunk=policy.pixel_chunk,
                smoothing=policy.label_smoothing,
                target_temperature=policy.target_temperature,
            ):
                local.update(hidden, targets, coords)
                direct_target.update(hidden, targets)
            cursor += 1
            processed += 1
            _save_state(
                state_path,
                config=config,
                stage="accumulate",
                cursor=cursor,
                local=local,
                direct_target=direct_target,
                invocation_history=invocation_history,
            )
        if cursor == len(pairs):
            local_beta, _ = local.solve()
            global_fold = (
                None
                if grid_shape == (1, 1)
                else StreamingRidgeNormalEquations(
                    context.hidden_dim,
                    context.n_classes,
                    ridge=policy.ridge,
                    pinv_rcond=policy.pinv_rcond,
                )
            )
            stage, cursor = "project", 0
            _save_state(
                state_path,
                config=config,
                stage=stage,
                cursor=cursor,
                local=local,
                direct_target=direct_target,
                local_beta=local_beta,
                global_fold=global_fold,
                invocation_history=invocation_history,
            )
    elif stage == "project":
        if local_beta is None:
            raise RuntimeError("project state lacks local POU coefficients")
        if grid_shape != (1, 1) and global_fold is None:
            raise RuntimeError("multi-domain project state lacks global fold equations")
        while cursor < len(pairs) and not bounded():
            pair = pairs[cursor]
            for hidden, targets, coords in context.iter_pair_chunks(
                pair,
                pixel_chunk=policy.pixel_chunk,
                smoothing=policy.label_smoothing,
                target_temperature=policy.target_temperature,
            ):
                local_prediction = partitioned_affine_predict(
                    hidden,
                    coords,
                    local_beta,
                    grid_shape=grid_shape,
                )
                if global_fold is not None:
                    global_fold.update(hidden, local_prediction)
                local_fit_sse += float(np.sum((local_prediction - targets) ** 2, dtype=np.float64))
                local_fit_elements += int(targets.size)
            cursor += 1
            processed += 1
            _save_state(
                state_path,
                config=config,
                stage="project",
                cursor=cursor,
                local=local,
                direct_target=direct_target,
                local_beta=local_beta,
                global_fold=global_fold,
                local_fit_sse=local_fit_sse,
                local_fit_elements=local_fit_elements,
                invocation_history=invocation_history,
            )
        if cursor == len(pairs):
            stage, cursor = "finalize", 0
            _save_state(
                state_path,
                config=config,
                stage=stage,
                cursor=cursor,
                local=local,
                direct_target=direct_target,
                local_beta=local_beta,
                global_fold=global_fold,
                local_fit_sse=local_fit_sse,
                local_fit_elements=local_fit_elements,
                invocation_history=invocation_history,
            )
    elif stage == "finalize":
        if local_beta is None or local_fit_elements <= 0:
            raise RuntimeError("finalize state lacks local coefficients or measured fit elements")
        direct_beta, direct_diagnostics = direct_target.solve()
        _, local_diagnostics = local.solve()
        if grid_shape == (1, 1):
            folded_beta = np.array(local_beta[0], copy=True)
            fold_diagnostics = None
            fold_vs_local_rmse = 0.0
            fold_second_solve_applied = False
        else:
            if global_fold is None:
                raise RuntimeError("multi-domain finalize state lacks global fold equations")
            folded_beta, fold_diagnostics = global_fold.solve()
            fold_vs_local_rmse = global_fold.residual_rmse(folded_beta)
            fold_second_solve_applied = True

        direct_rmse = direct_target.residual_rmse(direct_beta)
        local_rmse = float(np.sqrt(local_fit_sse / local_fit_elements))
        folded_target_rmse = direct_target.residual_rmse(folded_beta)
        tolerance = max(1e-10, 1e-8 * max(direct_rmse, folded_target_rmse, 1.0))
        if folded_target_rmse + tolerance < direct_rmse:
            raise FloatingPointError(
                "POU fold appears to beat the unregularized direct target-SSE optimum"
            )

        direct_path = args.output_dir / f"elm_head_seed_direct_global_{scope}.npz"
        fold_path = args.output_dir / f"elm_head_seed_pou_fold_{scope}.npz"
        receipt_path = args.output_dir / f"elm_head_seed_receipt_{scope}.json"
        write_seed_checkpoint_atomic(
            context.params_path,
            direct_path,
            weight=direct_beta[:-1].T.astype(np.float32),
            bias=direct_beta[-1].astype(np.float32),
        )
        write_seed_checkpoint_atomic(
            context.params_path,
            fold_path,
            weight=folded_beta[:-1].T.astype(np.float32),
            bias=folded_beta[-1].astype(np.float32),
        )
        direct_custody = _checkpoint_custody(context.params_path, direct_path)
        fold_custody = _checkpoint_custody(context.params_path, fold_path)

        gn_commands, gn_blocker, feature_handoff_custody = _gauss_newton_handoff(
            diagnostic_slice=diagnostic_slice,
            feature_state_path=context.feature_state_path,
            declared_feature_state_sha256=config["feature_state_sha256"],
            tag=args.tag,
            direct_checkpoint=direct_path,
            fold_checkpoint=fold_path,
        )
        invocation_history = [
            *invocation_history,
            _completed_invocation_record(
                index=len(invocation_history),
                argv=invocation_argv,
                started_utc=invocation_started_utc,
                started_monotonic=invocation_started,
                stage_before=stage_before,
                stage_after="done",
                cursor_before=cursor_before,
                cursor_after=0,
                processed_pairs=processed,
                resumed=resumed_from_existing_state,
                timing_boundary="final checkpoints and receipt payload prepared; receipt write excluded",
            ),
        ]

        receipt = {
            "schema": "elm_inr_head_seed_receipt_v3_20260712",
            "stage": "finalize",
            "scope": scope,
            "diagnostic_slice": diagnostic_slice,
            "pair_count": len(pairs),
            "selected_pairs": pairs,
            "source_checkpoint": str(context.params_path),
            "source_checkpoint_sha256": config["source_checkpoint_sha256"],
            "feature_state": str(context.feature_state_path),
            "feature_state_sha256": config["feature_state_sha256"],
            "labels": str(context.labels_path),
            "labels_sha256": config["labels_sha256"],
            "labels_bytes": config["labels_bytes"],
            "labels_source_dtype": config["labels_source_dtype"],
            "labels_dtype": config["labels_dtype"],
            "labels_resident_nbytes": config["labels_resident_nbytes"],
            "direct_global_checkpoint": direct_custody,
            "pou_fold_checkpoint": fold_custody,
            "head_scope": "out_sdf.weight/out_sdf.bias only; all other source arrays exact",
            "grid_shape": list(grid_shape),
            "metrics": {
                "direct_global_target_rmse": direct_rmse,
                "pou_local_target_rmse": local_rmse,
                "folded_global_vs_original_target_rmse": folded_target_rmse,
                "fold_vs_local_rmse": fold_vs_local_rmse,
            },
            "receiver_optimality": {
                "direct_global_is_target_sse_optimum": True,
                "comparator_ridge": 0.0,
                "pou_fold_cannot_beat_direct_on_target_sse_objective": True,
                "measured_fold_minus_direct_rmse": folded_target_rmse - direct_rmse,
                "note": (
                    "The current decoder ships one global affine out_sdf head. Local POU can lower "
                    "its pre-fold fit, but after projection it cannot improve target SSE over the "
                    "unregularized direct-global least-squares optimum in the same receiver family."
                ),
            },
            "fold_second_solve_applied": fold_second_solve_applied,
            "grid1_no_double_ridge": grid_shape == (1, 1),
            "local_diagnostics": [_diagnostic_dict(value) for value in local_diagnostics],
            "direct_global_diagnostics": _diagnostic_dict(direct_diagnostics),
            "global_fold_diagnostics": (
                None if fold_diagnostics is None else _diagnostic_dict(fold_diagnostics)
            ),
            "config": config,
            "execution_provenance": _runtime_provenance(invocation_history),
            "gauss_newton_commands": gn_commands,
            "gauss_newton_blocker": gn_blocker,
            "gauss_newton_feature_state_custody": feature_handoff_custody,
            "automatic_gauss_newton_launch": False,
            "non_authority": NON_AUTHORITY,
        }
        _atomic_write_json(receipt_path, receipt)
        done_receipt = {
            "receipt": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "direct_global_checkpoint": direct_custody,
            "pou_fold_checkpoint": fold_custody,
            "non_authority": NON_AUTHORITY,
        }
        _save_state(
            state_path,
            config=config,
            stage="done",
            cursor=0,
            local=local,
            direct_target=direct_target,
            local_beta=local_beta,
            global_fold=global_fold,
            local_fit_sse=local_fit_sse,
            local_fit_elements=local_fit_elements,
            invocation_history=invocation_history,
            done_receipt=done_receipt,
        )
        print(json.dumps({"stage": "done", **done_receipt}))
        return 0

    invocation_history = [
        *invocation_history,
        _completed_invocation_record(
            index=len(invocation_history),
            argv=invocation_argv,
            started_utc=invocation_started_utc,
            started_monotonic=invocation_started,
            stage_before=stage_before,
            stage_after=stage,
            cursor_before=cursor_before,
            cursor_after=cursor,
            processed_pairs=processed,
            resumed=resumed_from_existing_state,
            timing_boundary="final state payload prepared; final state fsync excluded",
        ),
    ]
    _save_state(
        state_path,
        config=config,
        stage=stage,
        cursor=cursor,
        local=local,
        direct_target=direct_target,
        local_beta=local_beta,
        global_fold=global_fold,
        local_fit_sse=local_fit_sse,
        local_fit_elements=local_fit_elements,
        invocation_history=invocation_history,
    )
    print(
        json.dumps(
            {
                "stage": stage,
                "cursor": cursor,
                "pair_count": len(pairs),
                "processed_this_invocation": processed,
                "state": str(state_path),
                "scope": scope,
                "diagnostic_slice": diagnostic_slice,
                "non_authority": NON_AUTHORITY,
            }
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("accumulate", "project", "finalize", "through-r-pair0"),
    )
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--tag", default="main_gt1", help="feature-state tag used by #341")
    parser.add_argument("--feature-state", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument(
        "--seed-receipt",
        type=Path,
        default=None,
        help="v3 ELM seed receipt required by through-r-pair0",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=None,
        help="durable standalone JSON receipt written by through-r-pair0",
    )
    parser.add_argument(
        "--max-pairs-per-invocation",
        type=int,
        default=0,
        help="operational checkpoint cadence; 0 means bounded only by max-seconds",
    )
    parser.add_argument("--max-seconds", type=float, default=420.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    cli_tokens = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(cli_tokens)
    args._invocation_argv = [sys.executable, str(Path(__file__).resolve()), *cli_tokens]
    args._invocation_cwd = Path.cwd().resolve()
    if args.max_pairs_per_invocation < 0:
        raise SystemExit("--max-pairs-per-invocation must be >=0")
    if not np.isfinite(args.max_seconds) or args.max_seconds <= 0.0:
        raise SystemExit("--max-seconds must be finite and >0")
    if args.stage == "through-r-pair0":
        return run_through_r_pair0(args)
    return run_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
