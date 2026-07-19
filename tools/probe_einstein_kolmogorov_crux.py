#!/usr/bin/env python3
"""Run one safe, receiver-closed PDW1 n24 component-treatment arm.

The only CLI input is a typed JSON config.  The source PDW1P payload is opened
read-only; results are written atomically to a fresh/resumable output directory.
This is [macOS-CPU advisory], research-only, and deliberately emits no pose or
contest score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tac.codec.pdw1_plane_codec import Pdw1PlanePayload, decode_pdw1p, encode_pdw1p, expand_scorer_plane  # noqa: E402
from tac.optimization.einstein_kolmogorov_crux import (  # noqa: E402
    DSPSAState,
    admit_candidate,
    coordinate_candidates,
    dspsa_perturbation,
    project_theta,
    score,
    wang_corners,
    wang_dspsa_step,
)
from tac.optimization.uint8_lattice_feasibility import (  # noqa: E402
    DisjointResizeOperator,
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)
from tac.witness_control.factorized_features import load_frozen_segnet_cpu  # noqa: E402
from tac.witness_dsl.einstein_kolmogorov_crux_20260719 import EinsteinKolmogorovCruxConfig  # noqa: E402

SCHEMA = "einstein_kolmogorov_crux_receipt.v2"
CHECKPOINT_SCHEMA = "einstein_kolmogorov_crux_checkpoint.v2"
CLOSURE_SCHEMA = "einstein_kolmogorov_reproducibility_closure.v1"
_SOURCE_NAMES = frozenset(
    {
        "probe",
        "typed_dsl",
        "optimization",
        "pdw1_codec",
        "uint8_lattice_realization",
        "frozen_segnet_loader",
        "upstream_modules",
    }
)
_THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_source_paths(config: EinsteinKolmogorovCruxConfig) -> dict[str, Path]:
    """Resolve stable implementation dependencies, never generated run artifacts."""
    return {
        "probe": Path(__file__).resolve(),
        "typed_dsl": REPO / "src/tac/witness_dsl/einstein_kolmogorov_crux_20260719.py",
        "optimization": REPO / "src/tac/optimization/einstein_kolmogorov_crux.py",
        "pdw1_codec": REPO / "src/tac/codec/pdw1_plane_codec.py",
        "uint8_lattice_realization": REPO / "src/tac/optimization/uint8_lattice_feasibility.py",
        "frozen_segnet_loader": REPO / "src/tac/witness_control/factorized_features.py",
        "upstream_modules": Path(config.upstream_path) / "modules.py",
    }


def _implementation_fingerprints(config: EinsteinKolmogorovCruxConfig) -> dict[str, str]:
    """Hash the stable source closure, excluding receipts and checkpoints."""
    paths = _implementation_source_paths(config)
    if missing := [name for name, path in paths.items() if not path.is_file()]:
        raise RuntimeError(f"implementation fingerprint source missing: {sorted(missing)}")
    return {name: _sha256_file(path) for name, path in paths.items()}


def _base_git_head() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("base_git_head unavailable; refusing provenance-incomplete receipt") from exc
    head = result.stdout.strip()
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
        raise RuntimeError("base_git_head is not a SHA-1 commit id")
    return head


def _runtime_state(torch_module: Any) -> dict[str, Any]:
    return {
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
            "cache_tag": sys.implementation.cache_tag,
        },
        "numpy": {"version": np.__version__, "module_path": str(Path(np.__file__).resolve())},
        "torch": {
            "version": torch_module.__version__,
            "module_path": str(Path(torch_module.__file__).resolve()),
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "threads": {
            "torch_num_threads": torch_module.get_num_threads(),
            "torch_num_interop_threads": torch_module.get_num_interop_threads(),
            "environment": {name: os.environ.get(name) for name in _THREAD_ENV_NAMES},
        },
    }


def _reproducibility_closure(config: EinsteinKolmogorovCruxConfig, *, torch_module: Any) -> dict[str, Any]:
    paths = _implementation_source_paths(config)
    fingerprints = _implementation_fingerprints(config)
    closure = {
        "schema": CLOSURE_SCHEMA,
        "sources": {name: {"path": str(path.resolve()), "sha256": fingerprints[name]} for name, path in paths.items()},
        "runtime": _runtime_state(torch_module),
        "base_git_head": _base_git_head(),
    }
    _validate_reproducibility_closure(closure)
    return closure


def _validate_reproducibility_closure(closure: Any) -> None:
    if not isinstance(closure, dict) or closure.get("schema") != CLOSURE_SCHEMA:
        raise ValueError("malformed reproducibility closure schema")
    sources = closure.get("sources")
    if not isinstance(sources, dict) or set(sources) != _SOURCE_NAMES:
        raise ValueError("reproducibility closure has incomplete implementation sources")
    for item in sources.values():
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("malformed reproducibility source entry")
        if not Path(item["path"]).is_absolute():
            raise ValueError("reproducibility source paths must be absolute")
        digest = item["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            raise ValueError("reproducibility source digest must be lowercase SHA-256")
    runtime = closure.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {"python", "numpy", "torch", "platform", "threads"}:
        raise ValueError("reproducibility closure has incomplete runtime state")
    expected_runtime_keys = {
        "python": {"version", "implementation", "executable", "cache_tag"},
        "numpy": {"version", "module_path"},
        "torch": {"version", "module_path"},
        "platform": {"platform", "system", "release", "version", "machine", "processor"},
        "threads": {"torch_num_threads", "torch_num_interop_threads", "environment"},
    }
    if any(
        not isinstance(runtime[name], dict) or set(runtime[name]) != keys
        for name, keys in expected_runtime_keys.items()
    ):
        raise ValueError("reproducibility closure runtime state is malformed")
    thread_environment = runtime["threads"]["environment"]
    if not isinstance(thread_environment, dict) or set(thread_environment) != set(_THREAD_ENV_NAMES):
        raise ValueError("reproducibility closure thread environment is incomplete")
    head = closure.get("base_git_head")
    if not isinstance(head, str) or len(head) != 40 or set(head) - set("0123456789abcdef"):
        raise ValueError("reproducibility closure base git head is malformed")


def _require_current_closure(
    config: EinsteinKolmogorovCruxConfig, *, torch_module: Any, expected: dict[str, Any]
) -> None:
    current = _reproducibility_closure(config, torch_module=torch_module)
    if current != expected:
        raise ValueError("source/runtime/package closure changed; refusing artifact mutation")


def _receipt_closure_fields(closure: dict[str, Any]) -> dict[str, Any]:
    """Expose the same complete closure in the final receipt and legacy views."""
    _validate_reproducibility_closure(closure)
    return {
        "reproducibility_closure": closure,
        "implementation_fingerprints": {name: item["sha256"] for name, item in closure["sources"].items()},
        "runtime": closure["runtime"],
        "base_git_head": closure["base_git_head"],
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(path, _json_bytes(payload))


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _atomic_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_immutable_bytes(path: Path, data: bytes) -> None:
    """Publish bytes atomically once; a resume may only reproduce identical bytes."""
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refusing to overwrite immutable artifact: {path}")
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError(f"refusing to overwrite immutable artifact: {path}") from None
    finally:
        tmp.unlink(missing_ok=True)


def _stage_checkpoint_path(output: Path, pair: int) -> Path:
    if pair < 0:
        raise ValueError("stage checkpoint pair must be non-negative")
    return output / "checkpoints" / f"pair_{pair:04d}.json"


def _iteration_checkpoint_path(output: Path, pair: int, iteration: int) -> Path:
    if pair < 0 or iteration <= 0:
        raise ValueError("iteration checkpoint requires non-negative pair and completed iteration > 0")
    return output / "checkpoints" / f"pair_{pair:04d}_iter_{iteration:04d}.json"


def _require_complete_packet_pairs(config: EinsteinKolmogorovCruxConfig, n_pairs: int) -> None:
    expected = tuple(range(n_pairs))
    if config.pair_indices != expected:
        raise ValueError("candidate distortion custody requires pair_indices to cover the complete packet in order")


def _assert_input_fingerprints(config: EinsteinKolmogorovCruxConfig) -> None:
    for name, raw_path, expected in (
        ("packet", config.packet_path, config.packet_sha256),
        ("gt", config.gt_path, config.gt_sha256),
        ("segnet", config.segnet_path, config.segnet_sha256),
    ):
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"{name} input is not a readable file: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(f"{name} fingerprint mismatch: expected {expected}, got {actual}")
    if not Path(config.upstream_path).is_dir():
        raise ValueError("upstream_path is not a readable directory")


def _open_output(config: EinsteinKolmogorovCruxConfig) -> tuple[Path, Path]:
    output = Path(config.output_dir)
    packet = Path(config.packet_path).resolve()
    if output.resolve() == packet.parent or packet.is_relative_to(output.resolve()):
        raise ValueError("output_dir may not contain the archival source packet")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    receipt = output / "receipt.json"
    if receipt.exists():
        raise ValueError(f"refusing to overwrite final receipt: {receipt}")
    checkpoint = output / "checkpoint.json"
    return output, checkpoint


def _checkpoint_payload(
    config: EinsteinKolmogorovCruxConfig,
    *,
    fills: np.ndarray,
    completed_pairs: list[int],
    rows: list[dict[str, Any]],
    reproducibility_closure: dict[str, Any],
    in_progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_reproducibility_closure(reproducibility_closure)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "config_fingerprint": config.fingerprint,
        "config": config.to_dict(),
        "reproducibility_closure": reproducibility_closure,
        "input_fingerprints": {
            "packet": config.packet_sha256,
            "gt": config.gt_sha256,
            "segnet": config.segnet_sha256,
        },
        "completed_pairs": completed_pairs,
        "fills": fills.tolist(),
        "rows": rows,
        "in_progress": in_progress,
    }


def _strict_uint8_fills(raw: Any, expected_shape: tuple[int, ...]) -> np.ndarray:
    def walk(value: Any) -> list[int]:
        if isinstance(value, list):
            return [item for child in value for item in walk(child)]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
            raise ValueError("resume fills must contain integral uint8 values")
        return [value]

    values = walk(raw)
    if len(values) != int(np.prod(expected_shape)):
        raise ValueError("resume fills shape mismatch")
    return np.asarray(values, dtype=np.uint8).reshape(expected_shape)


def _validate_checkpoint_payload(
    config: EinsteinKolmogorovCruxConfig,
    payload: Any,
    source_fills: np.ndarray,
    *,
    expected_closure: dict[str, Any],
    expected_completed: list[int] | None = None,
) -> tuple[np.ndarray, list[int], list[dict[str, Any]], dict[str, Any] | None]:
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("malformed checkpoint schema")
    if payload.get("config_fingerprint") != config.fingerprint or payload.get("config") != config.to_dict():
        raise ValueError("resume config mismatch")
    _validate_reproducibility_closure(expected_closure)
    if payload.get("reproducibility_closure") != expected_closure:
        raise ValueError("resume source/runtime/package closure mismatch")
    expected_hashes = {"packet": config.packet_sha256, "gt": config.gt_sha256, "segnet": config.segnet_sha256}
    if payload.get("input_fingerprints") != expected_hashes:
        raise ValueError("resume input fingerprint mismatch")
    completed_raw = payload.get("completed_pairs")
    if not isinstance(completed_raw, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in completed_raw
    ):
        raise ValueError("malformed completed pair list")
    completed = list(completed_raw)
    expected_prefix = list(config.pair_indices[: len(completed)])
    if completed != expected_prefix or (expected_completed is not None and completed != expected_completed):
        raise ValueError("completed pairs must be the exact configured prefix")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(completed):
        raise ValueError("checkpoint rows must correspond one-to-one with completed pairs")
    if [row.get("pair") if isinstance(row, dict) else None for row in rows] != completed:
        raise ValueError("checkpoint rows have duplicate or mismatched pairs")
    fills = _strict_uint8_fills(payload.get("fills"), source_fills.shape)
    in_progress = payload.get("in_progress")
    if in_progress is not None and not isinstance(in_progress, dict):
        raise ValueError("malformed in-progress DSPSA state")
    return fills, completed, rows, in_progress


def _resume_or_initial(
    config: EinsteinKolmogorovCruxConfig,
    checkpoint: Path,
    source_fills: np.ndarray,
    *,
    expected_closure: dict[str, Any],
) -> tuple[np.ndarray, list[int], list[dict[str, Any]], dict[str, Any] | None]:
    if not checkpoint.exists():
        return source_fills.copy(), [], [], None
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    fills, completed, rows, in_progress = _validate_checkpoint_payload(
        config, payload, source_fills, expected_closure=expected_closure
    )
    for index, pair in enumerate(completed):
        stage = _stage_checkpoint_path(checkpoint.parent, pair)
        if not stage.is_file():
            raise ValueError(f"resume missing preserved stage checkpoint: {stage}")
        stage_payload = json.loads(stage.read_text(encoding="utf-8"))
        _stage_fills, _stage_completed, _stage_rows, stage_progress = _validate_checkpoint_payload(
            config,
            stage_payload,
            source_fills,
            expected_closure=expected_closure,
            expected_completed=list(config.pair_indices[: index + 1]),
        )
        if stage_progress is not None:
            raise ValueError("preserved pair stage may not contain in-progress DSPSA state")
        if in_progress is None and index == len(completed) - 1 and _json_bytes(stage_payload) != _json_bytes(payload):
            raise ValueError("latest checkpoint does not match the final preserved pair stage")
    if in_progress is not None:
        if config.family != "dspsa" or in_progress.get("pair") != len(completed):
            raise ValueError("in-progress state is unreachable from completed pair prefix")
        state = DSPSAState.from_json(
            str(in_progress.get("state_json", "")),
            config_fingerprint=config.fingerprint,
            lower=config.lower_bound,
            upper=config.upper_bound,
        )
        if state.seed != config.seed + int(in_progress["pair"]) or not 0 < state.iteration <= config.iterations:
            raise ValueError("in-progress DSPSA state is unreachable")
        if state.last_objective_plus is None or state.last_objective_minus is None:
            raise ValueError("in-progress DSPSA state lacks two-corner objective evidence")
        if not isinstance(in_progress.get("best_row"), dict):
            raise ValueError("in-progress DSPSA state lacks best-row evidence")
        iteration_path = _iteration_checkpoint_path(checkpoint.parent, int(in_progress["pair"]), state.iteration)
        if not iteration_path.is_file():
            raise ValueError("missing immutable DSPSA iteration checkpoint")
        iteration_payload = json.loads(iteration_path.read_text(encoding="utf-8"))
        _validate_checkpoint_payload(
            config, iteration_payload, source_fills, expected_closure=expected_closure, expected_completed=completed
        )
        if _json_bytes(iteration_payload) != _json_bytes(payload):
            raise ValueError("latest checkpoint does not match preserved DSPSA iteration checkpoint")
    return fills, completed, rows, in_progress


def _palette_for_family(
    config: EinsteinKolmogorovCruxConfig, source: np.ndarray, labels: np.ndarray, gt: Any, segnet: Any
) -> np.ndarray:
    if config.family in {"baseline", "label_run_simplify"}:
        return source.copy()
    if config.family == "coordinate_warm_start":
        return np.clip(source, config.lower_bound, config.upper_bound).astype(np.uint8, copy=False)
    if config.family == "zero":
        return np.zeros_like(source)
    # Compute the allowed encoder-side global table strictly from config-bound gt_f1.
    total = np.zeros((source.shape[1], 3), dtype=np.float64)
    count = np.zeros(source.shape[1], dtype=np.int64)
    import torch

    for pair in config.pair_indices:
        frame = np.asarray(gt["gt_f1"][pair])
        x = torch.from_numpy(frame[None, None]).permute(0, 1, 4, 2, 3).contiguous().float()
        with torch.inference_mode():
            plane = segnet.preprocess_input(x)[0].permute(1, 2, 0).cpu().numpy()
        for class_id in range(source.shape[1]):
            mask = labels[pair] == class_id
            total[class_id] += plane[mask].sum(axis=0)
            count[class_id] += int(mask.sum())
    global_fill = np.clip(np.rint(total / np.maximum(count, 1)[:, None]), 0, 255).astype(np.uint8)
    result = np.broadcast_to(global_fill, source.shape).copy()
    if config.family == "dspsa":
        result = np.clip(result, config.lower_bound, config.upper_bound).astype(np.uint8, copy=False)
    return result


def _simplify_short_horizontal_runs(labels: np.ndarray, *, minimum_run: int) -> np.ndarray:
    """Simultaneously replace short row-runs by their longest adjacent run.

    This is an intentionally simple lossy-topology control, not an optimizer.
    Runs and neighbor lengths are read from the original row; replacements are
    applied to a copy so iteration order cannot affect the result.  A tie uses
    the left neighbor.  Border runs use their only neighbor.
    """

    source = np.asarray(labels)
    if source.dtype != np.uint8 or source.ndim != 3:
        raise ValueError("label simplification requires (pairs,H,W) uint8 labels")
    if not 2 <= minimum_run <= 64:
        raise ValueError("minimum_run must be in [2,64]")
    result = source.copy()
    width = source.shape[2]
    for pair in range(source.shape[0]):
        for y in range(source.shape[1]):
            row = source[pair, y]
            starts = [0]
            for x in range(1, width):
                if row[x] != row[x - 1]:
                    starts.append(x)
            starts.append(width)
            for index in range(len(starts) - 1):
                start, stop = starts[index], starts[index + 1]
                if stop - start >= minimum_run:
                    continue
                left = index - 1
                right = index + 1
                if left < 0 and right >= len(starts) - 1:
                    continue
                if left < 0:
                    replacement = row[starts[right]]
                elif right >= len(starts) - 1:
                    replacement = row[starts[left]]
                else:
                    left_len = starts[left + 1] - starts[left]
                    right_len = starts[right + 1] - starts[right]
                    replacement = row[starts[right]] if right_len > left_len else row[starts[left]]
                result[pair, y, start:stop] = replacement
    return result


def _measure_pair(
    *, payload: Pdw1PlanePayload, pair: int, lstar: np.ndarray, operator: DisjointResizeOperator, segnet: Any
) -> dict[str, Any]:
    import torch

    plane = expand_scorer_plane(payload, pair)
    frame = realize_factor2_uint8_scorer_plane(operator, plane)
    certificate = verify_factor2_uint8_scorer_plane(operator, frame, plane)
    if not certificate.certified_exact:
        raise RuntimeError(f"pair {pair} factor-2 certificate refused")
    x = torch.from_numpy(frame[None, None]).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        prediction = segnet(segnet.preprocess_input(x))[0].argmax(dim=0).cpu().numpy()
    mismatch = int((prediction != lstar).sum())
    return {
        "pair": pair,
        "mismatch_px": mismatch,
        "d_seg": float(mismatch / lstar.size),
        "factor2_certified_exact": True,
        "factor2_numerator_exact": bool(certificate.numerator_exact),
    }


def _evaluate_fill(
    *,
    labels: np.ndarray,
    fills: np.ndarray,
    pair: int,
    lstar: np.ndarray,
    operator: DisjointResizeOperator,
    segnet: Any,
) -> dict[str, Any]:
    payload = Pdw1PlanePayload(labels=labels, fills=fills)
    return _measure_pair(payload=payload, pair=pair, lstar=lstar, operator=operator, segnet=segnet)


def _coordinate_search(
    config: EinsteinKolmogorovCruxConfig,
    *,
    labels: np.ndarray,
    fills: np.ndarray,
    pair: int,
    lstar: np.ndarray,
    operator: DisjointResizeOperator,
    segnet: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    before = _evaluate_fill(labels=labels, fills=fills, pair=pair, lstar=lstar, operator=operator, segnet=segnet)
    current = fills[pair].reshape(-1).tolist()
    accepted = 0
    for _ in range(config.iterations):
        moved = False
        for candidate in coordinate_candidates(current, lower=config.lower_bound, upper=config.upper_bound):
            trial = fills.copy()
            trial[pair] = np.asarray(candidate, dtype=np.uint8).reshape(fills.shape[1:])
            measured = _evaluate_fill(
                labels=labels, fills=trial, pair=pair, lstar=lstar, operator=operator, segnet=segnet
            )
            if admit_candidate(
                before=(before["d_seg"], 0.0, 0),
                after=(measured["d_seg"], 0.0, 0),
                before_mismatches=before["mismatch_px"],
                after_mismatches=measured["mismatch_px"],
            ):
                fills = trial
                current = list(candidate)
                before = measured
                accepted += 1
                moved = True
                break
        if not moved:
            break
    before["coordinate_accepted_moves"] = accepted
    return fills, before


def _dspsa_search(
    config: EinsteinKolmogorovCruxConfig,
    *,
    labels: np.ndarray,
    fills: np.ndarray,
    pair: int,
    lstar: np.ndarray,
    operator: DisjointResizeOperator,
    segnet: Any,
    in_progress: dict[str, Any] | None,
    save_iteration: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    if in_progress is None:
        fills = fills.copy()
        bounded = np.clip(fills[pair], config.lower_bound, config.upper_bound).astype(np.uint8, copy=False)
        fills[pair] = bounded
        baseline = _evaluate_fill(labels=labels, fills=fills, pair=pair, lstar=lstar, operator=operator, segnet=segnet)
        state = DSPSAState(
            theta=project_theta(
                tuple(float(value) for value in fills[pair].reshape(-1)),
                lower=config.lower_bound,
                upper=config.upper_bound,
            ),
            best=tuple(int(value) for value in fills[pair].reshape(-1)),
            best_objective=float(baseline["mismatch_px"]),
            iteration=0,
            seed=config.seed + pair,
            config_fingerprint=config.fingerprint,
        )
        best_row = baseline
    else:
        state = DSPSAState.from_json(
            in_progress["state_json"],
            config_fingerprint=config.fingerprint,
            lower=config.lower_bound,
            upper=config.upper_bound,
        )
        best_row = dict(in_progress["best_row"])
        if best_row.get("pair") != pair or float(best_row.get("mismatch_px", -1)) != state.best_objective:
            raise ValueError("in-progress DSPSA best-row evidence mismatch")
    A = max(1, int(0.1 * max(config.iterations, 1)))
    for _ in range(state.iteration, config.iterations):
        signs = dspsa_perturbation(seed=state.seed, iteration=state.iteration, dimension=len(state.theta))
        plus, minus = wang_corners(state.theta, signs, lower=config.lower_bound, upper=config.upper_bound)
        trial_plus, trial_minus = fills.copy(), fills.copy()
        trial_plus[pair] = np.asarray(plus, dtype=np.uint8).reshape(fills.shape[1:])
        trial_minus[pair] = np.asarray(minus, dtype=np.uint8).reshape(fills.shape[1:])
        plus_row = _evaluate_fill(
            labels=labels, fills=trial_plus, pair=pair, lstar=lstar, operator=operator, segnet=segnet
        )
        minus_row = _evaluate_fill(
            labels=labels, fills=trial_minus, pair=pair, lstar=lstar, operator=operator, segnet=segnet
        )
        previous_best = state.best_objective
        state = wang_dspsa_step(
            state,
            objective_plus=float(plus_row["mismatch_px"]),
            objective_minus=float(minus_row["mismatch_px"]),
            target_first_displacement=config.target_first_displacement,
            gain_alpha=config.gain_alpha,
            A=A,
            lower=config.lower_bound,
            upper=config.upper_bound,
        )
        if state.best_objective < previous_best:
            best_row = plus_row if state.best == plus else minus_row
        fills[pair] = np.asarray(state.best, dtype=np.uint8).reshape(fills.shape[1:])
        save_iteration(state, best_row, fills)
    fills[pair] = np.asarray(state.best, dtype=np.uint8).reshape(fills.shape[1:])
    best_row["dspsa_iterations"] = state.iteration
    best_row["dspsa_best_objective"] = state.best_objective
    best_row["dspsa_calibrated_a"] = state.calibrated_a
    return fills, best_row


def run(config: EinsteinKolmogorovCruxConfig) -> dict[str, Any]:
    _assert_input_fingerprints(config)
    output, checkpoint = _open_output(config)
    source_blob = Path(config.packet_path).read_bytes()
    decoded = decode_pdw1p(source_blob)
    if encode_pdw1p(decoded) != source_blob:
        raise ValueError("source PDW1P fails strict encode-decode-reencode identity")
    _require_complete_packet_pairs(config, decoded.n_pairs)
    source_labels = np.asarray(decoded.labels)
    labels = (
        _simplify_short_horizontal_runs(source_labels, minimum_run=config.label_min_run)
        if config.family == "label_run_simplify"
        else source_labels
    )
    gt = np.load(config.gt_path, allow_pickle=False)
    if "gt_f1" not in gt or "lstars" not in gt:
        raise ValueError("gt archive must contain gt_f1 and lstars")
    if max(config.pair_indices) >= len(gt["lstars"]):
        raise ValueError("configured pair index exceeds gt inputs")
    import torch

    torch.set_num_threads(1)
    segnet = load_frozen_segnet_cpu(config.upstream_path)
    loaded_modules = sys.modules.get("modules")
    expected_modules = (Path(config.upstream_path) / "modules.py").resolve()
    loaded_modules_path = Path(getattr(loaded_modules, "__file__", "")).resolve()
    if loaded_modules is None or loaded_modules_path != expected_modules:
        raise RuntimeError(f"frozen SegNet loaded unexpected modules.py: {loaded_modules_path} != {expected_modules}")
    reproducibility_closure = _reproducibility_closure(config, torch_module=torch)
    operator = DisjointResizeOperator.build(
        camera_h=config.camera_height,
        camera_w=config.camera_width,
        scorer_h=config.scorer_height,
        scorer_w=config.scorer_width,
    )
    fills, completed, rows, in_progress = _resume_or_initial(
        config, checkpoint, np.asarray(decoded.fills), expected_closure=reproducibility_closure
    )
    fills = _palette_for_family(config, np.asarray(decoded.fills), labels, gt, segnet) if not completed else fills
    for pair in config.pair_indices:
        if pair in completed:
            continue
        lstar = np.asarray(gt["lstars"][pair])
        if config.family in {"coordinate", "coordinate_warm_start"}:
            fills, row = _coordinate_search(
                config, labels=labels, fills=fills, pair=pair, lstar=lstar, operator=operator, segnet=segnet
            )
        elif config.family == "dspsa":
            pair_progress = in_progress if in_progress is not None else None

            def save_iteration(
                state: DSPSAState, best_row: dict[str, Any], current_fills: np.ndarray, _pair: int = pair
            ) -> None:
                _require_current_closure(config, torch_module=torch, expected=reproducibility_closure)
                progress = {"pair": _pair, "state_json": state.to_json(), "best_row": best_row}
                iteration_payload = _checkpoint_payload(
                    config,
                    fills=current_fills,
                    completed_pairs=completed,
                    rows=rows,
                    reproducibility_closure=reproducibility_closure,
                    in_progress=progress,
                )
                iteration_path = _iteration_checkpoint_path(output, _pair, state.iteration)
                _atomic_immutable_bytes(iteration_path, _json_bytes(iteration_payload))
                _atomic_json(checkpoint, iteration_payload)

            fills, row = _dspsa_search(
                config,
                labels=labels,
                fills=fills,
                pair=pair,
                lstar=lstar,
                operator=operator,
                segnet=segnet,
                in_progress=pair_progress,
                save_iteration=save_iteration,
            )
            in_progress = None
        else:
            row = _evaluate_fill(labels=labels, fills=fills, pair=pair, lstar=lstar, operator=operator, segnet=segnet)
        if config.family == "label_run_simplify":
            row["label_mutations"] = int((labels[pair] != source_labels[pair]).sum())
        rows.append(row)
        completed.append(pair)
        _require_current_closure(config, torch_module=torch, expected=reproducibility_closure)
        stage_payload = _checkpoint_payload(
            config, fills=fills, completed_pairs=completed, rows=rows, reproducibility_closure=reproducibility_closure
        )
        stage_path = _stage_checkpoint_path(output, pair)
        _atomic_immutable_bytes(stage_path, _json_bytes(stage_payload))
        _atomic_json(checkpoint, stage_payload)
    final = Pdw1PlanePayload(labels=labels, fills=fills)
    packet = encode_pdw1p(final)
    fresh = decode_pdw1p(packet)
    if encode_pdw1p(fresh) != packet:
        raise RuntimeError("candidate PDW1P fails strict parse-back identity")
    _require_current_closure(config, torch_module=torch, expected=reproducibility_closure)
    candidate_path = output / "candidate.pdw1p.bin"
    _atomic_immutable_bytes(candidate_path, packet)
    baseline_bytes = len(source_blob)
    actual_bytes = len(packet)
    mismatch_total = sum(int(row["mismatch_px"]) for row in rows)
    d_seg = mismatch_total / float(len(rows) * config.scorer_height * config.scorer_width)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "research_only": True,
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "verdict_scope": "n24 SegNet-only PDW1 component arm; no compact pose stream or archive",
        "config_fingerprint": config.fingerprint,
        "config": config.to_dict(),
        **_receipt_closure_fields(reproducibility_closure),
        "family": config.family,
        "input_fingerprints": {"packet": config.packet_sha256, "gt": config.gt_sha256, "segnet": config.segnet_sha256},
        "packet": {
            "source_bytes": baseline_bytes,
            "candidate_bytes": actual_bytes,
            "candidate_sha256": hashlib.sha256(packet).hexdigest(),
            "candidate_path": str(candidate_path.resolve()),
            "strict_parseback": True,
            "strict_encode_decode_reencode_identity": True,
            "length_identity": actual_bytes == baseline_bytes,
        },
        "pairs": rows,
        "stage_checkpoints": [
            {
                "pair": pair,
                "path": str(_stage_checkpoint_path(output, pair).resolve()),
                "sha256": _sha256_file(_stage_checkpoint_path(output, pair)),
            }
            for pair in completed
        ],
        "dspsa_iteration_checkpoints": [
            {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for path in sorted((output / "checkpoints").glob("pair_*_iter_*.json"))
        ],
        "d_seg": d_seg,
        "hard_mismatch_px": mismatch_total,
        "label_mutations": int((labels != source_labels).sum()),
        "score_with_pose_zero_for_local_ordering_only": score(d_seg=d_seg, d_pose=0.0, archive_bytes=actual_bytes),
        "checkpoint_path": str(checkpoint),
        "pointer_delta": "UNMOVED",
    }
    _atomic_json(output / "receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Only legal input: typed JSON config.")
    args = parser.parse_args(argv)
    try:
        config = EinsteinKolmogorovCruxConfig.load(args.config)
        print(json.dumps(run(config), sort_keys=True))
    except Exception as exc:  # pragma: no cover - CLI refusal surface
        print(f"[einstein-kolmogorov-crux] REFUSE: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
