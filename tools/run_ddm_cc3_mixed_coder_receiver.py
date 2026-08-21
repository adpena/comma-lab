#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build, inflate, raw-control, and n600-measure the DDM CC3 mixed archive."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.optimization.ddm_cc2_coder_races import (  # noqa: E402
    build_per_stream_price_table,
)
from tac.optimization.ddm_cc3_mixed_coder_receiver import (  # noqa: E402
    EXPECTED_COMPOSITION_MEMBERS,
    build_mixed_archive,
    restore_mixed_archive,
)
from tac.optimization.ddm_pc1_pose_stream import (  # noqa: E402
    CAMERA_H,
    CAMERA_W,
    parse_counted_composition_archive,
    receive_pc1_camera_pairs,
)
from tac.optimization.ddm_runtime_exporter import (  # noqa: E402
    cc3_inflate_sh,
    cc3_runtime_payload,
)
from tac.optimization.ddm_ws1_warm_start import (  # noqa: E402
    receive_ws1_warm_start_archive,
)
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    CLASS_ORDER,
)
from tac.process_group_kill import run_in_process_group  # noqa: E402
from tools.measure_ddm_menu1_realized_flip_menu import (  # noqa: E402
    _config_and_inputs,
    _forward,
    _load_models,
)

CONFIG_SCHEMA = "DDMCC3MixedCoderReceiverConfigV1"
RECEIPT_SCHEMA = "ddm_cc3_mixed_coder_receiver_integration.v1"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
PAIR_COUNT = 600
BATCH_PAIRS = 32
SCORER_THREADS = 4
FRAME_SHAPE = (2, CAMERA_H, CAMERA_W, 3)
RAW_BYTES = PAIR_COUNT * int(np.prod(FRAME_SHAPE, dtype=np.int64))
RATE_DENOMINATOR = 37_545_489
RATE_NUMERATOR = 25
MIXED_BYTES = 136_116
MIXED_SHA256 = "ba18024211ff2e1de189d1a094b157c63aac86b21dca2dbce331e4385e49aebe"
DELTA_BYTES = -3_422
PRICE_TABLE_REPLAY_KEYS = (
    "composition_archive_bytes",
    "composition_archive_sha256",
    "recursive_fixed_zip_overhead_bytes",
    "counted_leaf_stream_count",
    "current_leaf_bytes",
    "selected_leaf_bytes",
    "selected_total_archive_estimate_bytes",
    "selected_total_delta_bytes",
    "selected_total_delta_dseg",
    "selected_total_delta_dpose",
    "selected_total_delta_advisory_action",
    "rows",
    "c1_waterfill_order",
)


class CC3RunError(RuntimeError):
    """A typed input, stage checkpoint, runtime, control, or scorer gate failed."""


class CC3Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_: Literal["DDMCC3MixedCoderReceiverConfigV1"] = Field(
        default=CONFIG_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    run_id: Literal["ddm_cc3_mixed_coder_receiver_integration_20260725T041134Z"]
    lane_id: Literal["lane_ddm_cc3_mixed_coder_receiver_integration_20260725"]
    delegation_checkpoint_key: Literal["codex_delegate:ddm_cc3_mixed_coder_receiver_integration:20260725T041134Z"]
    authority_path: str
    authority_sha256: Literal["9acf366cb6afa5b3f9cb04171e72dcbb05a1ac7b99a52a456b90f657dddee991"]
    authority_bytes: Literal[6966]
    own_python: str
    output_root: str
    source_archive_path: str
    source_archive_sha256: Literal["08d03f75f818b22c9a9a6aad33c6f879001743f80b66d71fe1fc9b3a094567a2"]
    source_archive_bytes: Literal[139538]
    cc2_receipt_path: str
    cc2_receipt_sha256: Literal["f9432959d9c8711276379ef681f5b6985157f49bdd4b2f4a401bfb35ce737ec1"]
    cc2_repository_receipt_path: str
    cc2_repository_receipt_sha256: Literal["def7e5273dc4317d79c3a5260feccaf12d5cb25aa3128fc28f25e60c94e18d00"]
    lp1_receipt_path: str
    lp1_receipt_sha256: Literal["6bd6a5baaa8f5995e93ef594e880beac77e9aa2b2083e661598c84feaba13fd5"]
    menu1_config_path: str
    menu1_config_sha256: Literal["b9fed2b1537b92a2b02d0525cb4d9175d0704e7c4d0c0efd383c6dd818fdb2c7"]
    target_cache_path: str
    target_cache_sha256: Literal["cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"]
    scorer_batch_size: Literal[32] = 32
    scorer_threads: Literal[4] = 4
    minimum_free_bytes: int = Field(ge=16 * 1024**3)
    timeout_seconds: Literal[1800] = 1800
    seed: Literal[0] = 0
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    pointer: Literal["0.1910828242 [contest-CPU]"] = "0.1910828242 [contest-CPU]"


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def _bound_bytes(path_value: str, sha256: str, label: str) -> bytes:
    path = _resolve(path_value)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise CC3RunError(f"{label} SHA-256 differs")
    return payload


def _bound_json(path_value: str, sha256: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_bound_bytes(path_value, sha256, label))
    except json.JSONDecodeError as exc:
        raise CC3RunError(f"{label} is malformed JSON") from exc
    if not isinstance(value, dict):
        raise CC3RunError(f"{label} must be one JSON object")
    return value


def _publish_bytes(path: Path, payload: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise CC3RunError(f"immutable output differs: {path}")
        if executable:
            path.chmod(path.stat().st_mode | 0o111)
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if executable:
        temporary.chmod(0o755)
    os.replace(temporary, path)


def _publish_json(path: Path, value: Any) -> None:
    _publish_bytes(path, _canonical_json(value))


def _load_config(path: Path) -> tuple[CC3Config, str]:
    payload = path.read_bytes()
    try:
        config = CC3Config.model_validate_json(payload, strict=True)
    except Exception as exc:
        raise CC3RunError("typed CC3 config validation failed") from exc
    if payload != _canonical_json(config.model_dump(mode="json", by_alias=True)):
        raise CC3RunError("typed CC3 config is not canonical JSON")
    expected_python = Path(config.own_python).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise CC3RunError(f"CC3 must run in its locked SSD venv: {expected_python}; got {sys.executable}")
    return config, hashlib.sha256(payload).hexdigest()


def _storage_preflight(config: CC3Config, root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    if not str(root.resolve()).startswith("/Volumes/VertigoDataTier/pact/"):
        raise CC3RunError("CC3 output must stay on the primary SSD tier")
    usage = shutil.disk_usage(root)
    if usage.free < config.minimum_free_bytes:
        raise CC3RunError(f"SSD storage preflight failed closed: {usage.free} < {config.minimum_free_bytes}")
    return {
        "auto_cleanup": (
            "success-only subprocess scratch is absent; candidate/control raw and every "
            "32-pair checkpoint are retained as measurement custody"
        ),
        "free_bytes": usage.free,
        "minimum_free_bytes": config.minimum_free_bytes,
        "status": "PASS",
        "tier": "/Volumes/VertigoDataTier/pact",
    }


def _cc2_price_table(receipt: dict[str, Any]) -> dict[str, Any]:
    table = receipt.get("race3")
    if (
        not isinstance(table, dict)
        or table.get("counted_leaf_stream_count") != 27
        or table.get("selected_total_archive_estimate_bytes") != MIXED_BYTES
        or table.get("selected_total_delta_bytes") != DELTA_BYTES
        or len(table.get("rows", [])) != 27
    ):
        raise CC3RunError("CC2 typed 27-leaf price table differs")
    arm_count = sum(len(row.get("arms", [])) for row in table["rows"])
    if arm_count != 135 or not all(row.get("parseback_exact_all_arms") is True for row in table["rows"]):
        raise CC3RunError("CC2 135-frame exact parse-back seal differs")
    return table


def _replay_cc2_frame_matrix(
    source: bytes,
    settled: dict[str, Any],
) -> dict[str, Any]:
    """Freshly re-encode and decode all five coder frames for all 27 leaves."""

    started = time.monotonic()
    replay = build_per_stream_price_table(source)
    if any(replay[key] != settled.get(key) for key in PRICE_TABLE_REPLAY_KEYS):
        raise CC3RunError("fresh CC2 135-frame replay differs from its settled price table")
    core = {key: replay[key] for key in PRICE_TABLE_REPLAY_KEYS}
    return {
        "all_parseback_exact": all(row["parseback_exact_all_arms"] for row in replay["rows"]),
        "delta_bytes": replay["selected_total_delta_bytes"],
        "elapsed_seconds": format(time.monotonic() - started, ".6f"),
        "frame_count": sum(len(row["arms"]) for row in replay["rows"]),
        "leaf_count": len(replay["rows"]),
        "replay_core_sha256": hashlib.sha256(_canonical_json(core)[:-1]).hexdigest(),
        "selected_total_archive_bytes": replay["selected_total_archive_estimate_bytes"],
        "status": "PASS",
    }


def _adopt_completed_receipt(
    receipt_path: Path,
    *,
    config_sha256: str,
    output_root: Path,
) -> dict[str, Any] | None:
    """Validate and return a completed immutable receipt without rerunning."""

    if not receipt_path.is_file():
        return None
    payload = receipt_path.read_bytes()
    try:
        result = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CC3RunError("completed CC3 receipt is malformed") from exc
    if payload != _canonical_json(result):
        raise CC3RunError("completed CC3 receipt is not canonical JSON")
    expected_paths = {
        "archive": output_root / "packet/archive.zip",
        "candidate_raw": output_root / "runtime/inflated/0.raw",
        "control_raw": output_root / "control/0.raw",
    }
    if (
        result.get("schema") != RECEIPT_SCHEMA
        or result.get("typed_config", {}).get("sha256") != config_sha256
        or result.get("archive") != {"bytes": MIXED_BYTES, "sha256": MIXED_SHA256}
        or result.get("source_archive")
        != {
            "bytes": 139_538,
            "sha256": "08d03f75f818b22c9a9a6aad33c6f879001743f80b66d71fe1fc9b3a094567a2",
        }
        or _sha256_file(expected_paths["archive"]) != (MIXED_BYTES, MIXED_SHA256)
    ):
        raise CC3RunError("completed CC3 receipt archive or config custody differs")
    candidate_identity = _sha256_file(expected_paths["candidate_raw"])
    control_identity = _sha256_file(expected_paths["control_raw"])
    if (
        candidate_identity != control_identity
        or candidate_identity != (RAW_BYTES, result.get("endpoint", {}).get("raw", {}).get("sha256"))
        or result.get("control", {}).get("raw", {}).get("sha256") != candidate_identity[1]
    ):
        raise CC3RunError("completed CC3 receipt raw custody differs")
    return result


def _prepare_packet(root: Path, mixed: bytes) -> dict[str, Any]:
    packet = root / "packet"
    paths = {
        "archive.zip": packet / "archive.zip",
        "inflate.py": packet / "inflate.py",
        "inflate.sh": packet / "inflate.sh",
    }
    _publish_bytes(paths["archive.zip"], mixed)
    _publish_bytes(paths["inflate.py"], cc3_runtime_payload(), executable=True)
    _publish_bytes(paths["inflate.sh"], cc3_inflate_sh(), executable=True)
    result: dict[str, Any] = {}
    for name, path in paths.items():
        bytes_, sha256 = _sha256_file(path)
        result[name] = {"bytes": bytes_, "path": str(path), "sha256": sha256}
    return result


def _extract_packet(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        infos = archive.infolist()
        if tuple(info.filename for info in infos) != EXPECTED_COMPOSITION_MEMBERS:
            raise CC3RunError("counted archive outer member identity/order differs")
        for info in infos:
            relative = PurePosixPath(info.filename)
            if info.is_dir() or relative.is_absolute() or ".." in relative.parts:
                raise CC3RunError("unsafe counted archive member")
            _publish_bytes(destination.joinpath(*relative.parts), archive.read(info))


def _run_locked_inflate(
    *,
    config: CC3Config,
    root: Path,
    packet: dict[str, Any],
) -> dict[str, Any]:
    runtime_root = root / "runtime"
    extracted = runtime_root / "archive"
    inflated = runtime_root / "inflated"
    names = runtime_root / "public_test_video_names.txt"
    _publish_bytes(names, b"0.mkv\n")
    _extract_packet(Path(packet["archive.zip"]["path"]), extracted)
    argv = [
        "bash",
        packet["inflate.sh"]["path"],
        str(extracted),
        str(inflated),
        str(names),
    ]
    environment = dict(os.environ)
    python_executable = Path(config.own_python)
    if not python_executable.is_absolute() or not python_executable.is_file():
        raise CC3RunError("configured locked-environment Python is absent")
    # Preserve the venv entrypoint. Resolving its interpreter symlink drops
    # pyvenv.cfg discovery and therefore the declared runtime dependencies.
    environment["PYTHON"] = str(python_executable)
    environment["PATH"] = str(python_executable.parent) + os.pathsep + environment.get("PATH", "")
    started = time.monotonic()
    try:
        completed = run_in_process_group(
            argv,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
        )
        timeout_hit = False
    except subprocess.TimeoutExpired as exc:
        timeout_hit = True
        completed = subprocess.CompletedProcess(
            argv,
            124,
            stdout=(exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""),
            stderr=(exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""),
        )
    wallclock = time.monotonic() - started
    stdout_digest = hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest()
    stderr_digest = hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest()
    stdout_path = runtime_root / "logs" / f"inflate.{stdout_digest}.stdout.txt"
    stderr_path = runtime_root / "logs" / f"inflate.{stderr_digest}.stderr.txt"
    _publish_bytes(stdout_path, completed.stdout.encode("utf-8"))
    _publish_bytes(stderr_path, completed.stderr.encode("utf-8"))
    raw_path = inflated / "0.raw"
    if timeout_hit or completed.returncode != 0 or not raw_path.is_file():
        raise CC3RunError(
            f"locked inflate failed: timeout={timeout_hit}, rc={completed.returncode}, stderr={stderr_path}"
        )
    raw_bytes, raw_sha256 = _sha256_file(raw_path)
    if raw_bytes != RAW_BYTES or wallclock >= config.timeout_seconds:
        raise CC3RunError("locked inflate output geometry or wallclock differs")
    receipts = sorted((runtime_root / ".ddm_runtime_checkpoints").glob("cc3_*/0/inflate_receipt.json"))
    if len(receipts) != 1:
        raise CC3RunError("locked inflate did not leave exactly one CC3 runtime receipt")
    runtime_receipt = json.loads(receipts[0].read_bytes())
    if (
        runtime_receipt.get("schema") != "ddm_cc3_mixed_coder_runtime_inflate_receipt.v1"
        or runtime_receipt.get("resume", {}).get("stage_count") != 19
        or runtime_receipt.get("final", {}).get("sha256") != raw_sha256
    ):
        raise CC3RunError("locked inflate runtime receipt differs")
    return {
        "argv": argv,
        "environment": {
            "PYTHON": environment["PYTHON"],
            "runtime_distributions": {
                name: importlib.metadata.version(name) for name in ("Brotli", "numpy", "scipy", "torch")
            },
        },
        "exit_code": completed.returncode,
        "logs": {
            "stderr": {"path": str(stderr_path), "sha256": stderr_digest},
            "stdout": {"path": str(stdout_path), "sha256": stdout_digest},
        },
        "raw": {"bytes": raw_bytes, "path": str(raw_path), "sha256": raw_sha256},
        "runtime_receipt": {
            "path": str(receipts[0]),
            "sha256": _sha256_file(receipts[0])[1],
        },
        "status": "PASS",
        "timeout_seconds": config.timeout_seconds,
        "under_1800_seconds": wallclock < config.timeout_seconds,
        "wallclock_seconds": format(wallclock, ".6f"),
    }


def _write_raw_stage(path: Path, camera: np.ndarray) -> dict[str, Any]:
    payload = np.ascontiguousarray(camera, dtype=np.uint8)
    expected_bytes = int(payload.nbytes)
    camera_sha256 = hashlib.sha256(payload.tobytes()).hexdigest()
    sidecar = path.with_suffix(".json")
    if path.is_file() and sidecar.is_file():
        row = json.loads(sidecar.read_bytes())
        if (
            row.get("bytes") == expected_bytes
            and row.get("sha256") == _sha256_file(path)[1]
            and row.get("camera_sha256") == camera_sha256
        ):
            return row
        raise CC3RunError(f"preserved raw control checkpoint differs: {path}")
    if path.exists() or sidecar.exists():
        raise CC3RunError(f"partial raw control checkpoint exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("xb") as handle:
        payload.tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    row = {
        "bytes": expected_bytes,
        "camera_sha256": camera_sha256,
        "sha256": _sha256_file(path)[1],
    }
    _publish_json(sidecar, row)
    return row


def _assemble_control(
    *,
    final_path: Path,
    stage_paths: list[Path],
) -> tuple[int, str]:
    if final_path.exists():
        return _sha256_file(final_path)
    digest = hashlib.sha256()
    total = 0
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(f".{final_path.name}.partial.{os.getpid()}")
    with temporary.open("xb") as target:
        for path in stage_paths:
            with path.open("rb") as source:
                while chunk := source.read(8 * 1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
        target.flush()
        os.fsync(target.fileno())
    if total != RAW_BYTES:
        temporary.unlink(missing_ok=True)
        raise CC3RunError("raw control final geometry differs")
    os.replace(temporary, final_path)
    return total, digest.hexdigest()


def _render_raw_control(source: bytes, root: Path) -> dict[str, Any]:
    parent, packet, _manifest = parse_counted_composition_archive(source)
    receiver = receive_ws1_warm_start_archive(parent)
    try:
        movable = next(layer for layer in receiver.layers if layer.role == "Movable")
    except StopIteration as exc:
        raise CC3RunError("raw control W_joint state lacks the Movable layer") from exc
    stage_paths: list[Path] = []
    stage_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for start in range(0, PAIR_COUNT, BATCH_PAIRS):
        stop = min(start + BATCH_PAIRS, PAIR_COUNT)
        pair_ids = tuple(range(start, stop))
        parent_camera = receiver.render_camera_pairs(pair_ids)
        masks = np.stack(
            [
                receiver._mask_for_layer(
                    movable,
                    pair_id,
                    replace_g1_movable=True,
                )
                for pair_id in pair_ids
            ],
            axis=0,
        ).astype(np.bool_)
        camera = receive_pc1_camera_pairs(
            parent_camera=parent_camera,
            packet=packet,
            pair_ids=pair_ids,
            movable_masks=masks,
            torch_module=torch,
        )
        path = root / "control" / "checkpoints" / f"pairs_{start:04d}_{stop:04d}.raw"
        row = {**_write_raw_stage(path, camera), "pair_range": [start, stop]}
        stage_paths.append(path)
        stage_rows.append(row)
        print(f"[CC3] raw control {start:04d}:{stop:04d}", flush=True)
    final_path = root / "control" / "0.raw"
    raw_bytes, raw_sha256 = _assemble_control(final_path=final_path, stage_paths=stage_paths)
    return {
        "all_stage_checkpoints_preserved": True,
        "raw": {"bytes": raw_bytes, "path": str(final_path), "sha256": raw_sha256},
        "source_archive": {"bytes": len(source), "sha256": hashlib.sha256(source).hexdigest()},
        "stage_count": len(stage_rows),
        "stage_rows": stage_rows,
        "wallclock_seconds": format(time.monotonic() - started, ".6f"),
    }


def _load_scorer_row(
    path: Path,
    *,
    raw_sha256: str,
    camera_sha256: str,
    start: int,
    stop: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    row = json.loads(path.read_bytes())
    if (
        row.get("schema") != "ddm_cc3_mixed_coder_batch32_row.v1"
        or row.get("raw_sha256") != raw_sha256
        or row.get("camera_sha256") != camera_sha256
        or row.get("pair_range") != [start, stop]
        or row.get("evidence_axis") != EVIDENCE_AXIS
        or row.get("score_claim") is not False
    ):
        raise CC3RunError(f"preserved scorer checkpoint differs: {path}")
    return row


def _score_n600(config: CC3Config, raw_path: Path, root: Path) -> dict[str, Any]:
    raw_bytes, raw_sha256 = _sha256_file(raw_path)
    if raw_bytes != RAW_BYTES:
        raise CC3RunError("scorer input raw geometry differs")
    menu_path = _resolve(config.menu1_config_path)
    if _sha256_file(menu_path)[1] != config.menu1_config_sha256:
        raise CC3RunError("Menu1 scorer config SHA-256 differs")
    menu_config, _ = _config_and_inputs(menu_path)
    if Path(menu_config.target_cache_path).resolve() != Path(config.target_cache_path).resolve():
        raise CC3RunError("Menu1 target-cache path differs")
    if _sha256_file(Path(config.target_cache_path))[1] != config.target_cache_sha256:
        raise CC3RunError("target-cache SHA-256 differs")
    with np.load(config.target_cache_path, allow_pickle=False) as cache:
        labels = np.asarray(cache["lstars"], dtype=np.uint8)
        poses = np.asarray(cache["gt_poses"], dtype=np.float64)
    if labels.shape != (PAIR_COUNT, 384, 512) or poses.shape != (PAIR_COUNT, 6):
        raise CC3RunError("target-cache geometry differs")
    torch.set_num_threads(SCORER_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    segnet, posenet, custody = _load_models(menu_config)
    custody["batch_size"] = BATCH_PAIRS
    raw = np.memmap(
        raw_path,
        mode="r",
        dtype=np.uint8,
        shape=(PAIR_COUNT, *FRAME_SHAPE),
    )
    rows: list[dict[str, Any]] = []
    checkpoint_root = root / "scorer" / "batch32"
    for start in range(0, PAIR_COUNT, BATCH_PAIRS):
        stop = min(start + BATCH_PAIRS, PAIR_COUNT)
        camera = np.array(raw[start:stop], copy=True, order="C")
        camera_sha256 = hashlib.sha256(camera.tobytes()).hexdigest()
        path = checkpoint_root / f"batch_{start:04d}_{stop:04d}.json"
        row = _load_scorer_row(
            path,
            raw_sha256=raw_sha256,
            camera_sha256=camera_sha256,
            start=start,
            stop=stop,
        )
        if row is None:
            cells, pose6 = _forward(segnet, posenet, camera)
            target = labels[start:stop]
            target_pose = poses[start:stop]
            row = {
                "camera_sha256": camera_sha256,
                "errors": int(np.count_nonzero(cells != target)),
                "evidence_axis": EVIDENCE_AXIS,
                "pair_range": [start, stop],
                "per_class": {
                    class_name: {
                        "errors": int(np.count_nonzero((cells != target) & (target == class_id))),
                        "sites": int(np.count_nonzero(target == class_id)),
                    }
                    for class_id, class_name in enumerate(CLASS_ORDER)
                },
                "pose_coordinates": int(pose6.size),
                "pose_squared_error_sum": float(np.square(pose6 - target_pose).sum(dtype=np.float64)),
                "raw_sha256": raw_sha256,
                "schema": "ddm_cc3_mixed_coder_batch32_row.v1",
                "score_claim": False,
                "sites": int(cells.size),
            }
            _publish_json(path, row)
        rows.append(row)
        print(f"[CC3] frozen scorer {start:04d}:{stop:04d}", flush=True)
    errors = sum(int(row["errors"]) for row in rows)
    sites = sum(int(row["sites"]) for row in rows)
    pose_sse = sum(float(row["pose_squared_error_sum"]) for row in rows)
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in rows)
    per_class = {
        class_name: {
            "errors": sum(int(row["per_class"][class_name]["errors"]) for row in rows),
            "sites": sum(int(row["per_class"][class_name]["sites"]) for row in rows),
        }
        for class_name in CLASS_ORDER
    }
    for value in per_class.values():
        value["d_seg"] = value["errors"] / value["sites"]
    return {
        "checkpoint_root": str(checkpoint_root),
        "d_pose": pose_sse / pose_coordinates,
        "d_seg": errors / sites,
        "errors": errors,
        "evidence_axis": EVIDENCE_AXIS,
        "per_class": per_class,
        "pose_coordinates": pose_coordinates,
        "pose_squared_error_sum": pose_sse,
        "raw": {"bytes": raw_bytes, "path": str(raw_path), "sha256": raw_sha256},
        "schema": "ddm_cc3_mixed_coder_batch32_endpoint.v1",
        "score_claim": False,
        "scorer_batch_size": BATCH_PAIRS,
        "scorer_custody": custody,
        "scorer_threads": SCORER_THREADS,
        "sites": sites,
        "stage_count": len(rows),
    }


def run(config_path: Path, receipt_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    config, config_sha256 = _load_config(config_path)
    output_root = Path(config.output_root).resolve()
    completed = _adopt_completed_receipt(
        receipt_path,
        config_sha256=config_sha256,
        output_root=output_root,
    )
    if completed is not None:
        return completed
    storage = _storage_preflight(config, output_root)
    authority = _bound_bytes(
        config.authority_path,
        config.authority_sha256,
        "delegated authority",
    )
    if len(authority) != config.authority_bytes:
        raise CC3RunError("delegated authority byte count differs")
    source = _bound_bytes(
        config.source_archive_path,
        config.source_archive_sha256,
        "CC2 source archive",
    )
    if len(source) != config.source_archive_bytes:
        raise CC3RunError("CC2 source archive byte count differs")
    cc2 = _bound_json(config.cc2_receipt_path, config.cc2_receipt_sha256, "CC2 SSD receipt")
    _bound_json(
        config.cc2_repository_receipt_path,
        config.cc2_repository_receipt_sha256,
        "CC2 repository receipt",
    )
    lp1 = _bound_json(config.lp1_receipt_path, config.lp1_receipt_sha256, "LP1 receipt")
    if lp1.get("c1_corrected_waterfill", {}).get("corrected_measured_allocated_bytes") != 134_211:
        raise CC3RunError("LP1 corrected measured allocation differs")
    table = _cc2_price_table(cc2)
    frame_replay = _replay_cc2_frame_matrix(source, table)
    if (
        frame_replay["frame_count"] != 135
        or frame_replay["leaf_count"] != 27
        or frame_replay["all_parseback_exact"] is not True
    ):
        raise CC3RunError("fresh CC2 frame-matrix census differs")

    mixed, integration = build_mixed_archive(source, table)
    if (len(mixed), hashlib.sha256(mixed).hexdigest()) != (MIXED_BYTES, MIXED_SHA256) or integration[
        "delta_bytes"
    ] != DELTA_BYTES:
        raise CC3RunError("exact CC3 mixed archive identity differs")
    mixed_replay, integration_replay = build_mixed_archive(source, table)
    if mixed_replay != mixed or integration_replay != integration:
        raise CC3RunError("CC3 exact build replay differs")
    restored, restoration = restore_mixed_archive(mixed)
    if restored != source:
        raise CC3RunError("CC3 exact source restoration differs")
    packet = _prepare_packet(output_root, mixed)
    locked = _run_locked_inflate(config=config, root=output_root, packet=packet)
    control = _render_raw_control(source, output_root)
    if control["raw"]["sha256"] != locked["raw"]["sha256"]:
        raise CC3RunError("mixed receiver and raw-leaf control output bytes differ")
    endpoint = _score_n600(config, Path(locked["raw"]["path"]), output_root)

    cc2_row = next(row for row in cc2["race2"]["rows"] if row.get("arm_id") == "CAMERA_Q8_EXACT")["verdict"]
    cc2_reused = {
        "d_pose": float(cc2_row["d_pose"]),
        "d_seg": float(cc2_row["d_seg"]),
        "decision_basis": cc2_row["decision_basis"],
        "pc1_output_effect": cc2_row["pc1_output_effect"],
    }
    reuse_matches_full_receiver = (
        endpoint["d_seg"] == cc2_reused["d_seg"] and endpoint["d_pose"] == cc2_reused["d_pose"]
    )
    rate_delta = RATE_NUMERATOR * DELTA_BYTES / RATE_DENOMINATOR
    costate_rows = [
        {
            "authority": "MEASURED_EXACT_COUNTED_ARCHIVE",
            "delta": DELTA_BYTES,
            "metric": "archive_bytes",
            "schema": "ddm_cc3_costate_row.v1",
        },
        {
            "authority": "MEASURED_RAW_BYTE_IDENTITY",
            "delta": 0.0,
            "metric": "d_seg",
            "schema": "ddm_cc3_costate_row.v1",
        },
        {
            "authority": "MEASURED_RAW_BYTE_IDENTITY",
            "delta": 0.0,
            "metric": "d_pose",
            "schema": "ddm_cc3_costate_row.v1",
        },
        {
            "authority": "DERIVED_FROM_MEASURED_EXACT_RATE_DELTA",
            "delta": rate_delta,
            "metric": "joint_action",
            "schema": "ddm_cc3_costate_row.v1",
        },
    ]
    result = {
        "admission": {
            "all_27_physical_leaves_exact": restoration["physical_leaf_count"] == 27,
            "all_135_cc2_canonical_frames_parseback_exact": True,
            "archive_build_replay_byte_identical": True,
            "counted_payload_consumer_bijection": True,
            "locked_environment_inflate": locked["status"] == "PASS",
            "mixed_vs_raw_receiver_output_byte_identical": True,
            "receiver_integration_overhead_bytes": 0,
            "selected_8_frames_exact": restoration["decoded_frame_count"] == 8,
        },
        "archive": integration["mixed_archive"],
        "authority": {
            "bytes": config.authority_bytes,
            "path": config.authority_path,
            "sha256": config.authority_sha256,
        },
        "cc2_canonical_frame_matrix": {
            "frame_count": 135,
            "leaf_count": 27,
            "parseback_exact": True,
            "receipt_path": config.cc2_receipt_path,
            "receipt_sha256": config.cc2_receipt_sha256,
        },
        "cc2_reused_preintegration_endpoint": cc2_reused,
        "codec_counts": restoration["codec_counts"],
        "control": control,
        "corrected_lp1": {
            "authority": "DERIVED_COORDINATED_BUDGET",
            "cc3_delta_bytes": DELTA_BYTES,
            "lp1_corrected_measured_allocated_bytes": 134_211,
            "post_integration_corrected_total_bytes": 130_789,
            "receipt_path": config.lp1_receipt_path,
            "receipt_sha256": config.lp1_receipt_sha256,
        },
        "costate_rows": costate_rows,
        "delegation_checkpoint_key": config.delegation_checkpoint_key,
        "endpoint": endpoint,
        "evidence_axis": EVIDENCE_AXIS,
        "falsifier": {
            "gross_cc2_savings_bytes": 3_422,
            "integration_overhead_bytes": 0,
            "net_delta_bytes": DELTA_BYTES,
            "net_lt_negative_1700": True,
            "overhead_fraction": 0.0,
            "overhead_gt_50_percent": False,
            "status": "PASS_PAYS",
        },
        "frame_matrix_replay": frame_replay,
        "integration": integration,
        "lane_id": config.lane_id,
        "locked_inflate": locked,
        "main_review_required": True,
        "packet": packet,
        "pointer": config.pointer,
        "pointer_moved": False,
        "preintegration_reuse_premise": {
            "full_receiver_matches_cc2_reused_endpoint": reuse_matches_full_receiver,
            "status": (
                "CONFIRMED" if reuse_matches_full_receiver else "FALSIFIED_CC2_ACTIVE_ZERO_IDENTITY_SCORE_REUSE"
            ),
            "verdict_scope": (
                "INSTANCE: CC2 CAMERA_Q8_EXACT receipt reused a parent-only endpoint "
                "for this exact active-zero PC1 composition; this does not affect the "
                "lossless mixed-vs-raw identity proof or close the PC1 family."
            ),
        },
        "promotion_eligible": False,
        "research_only": True,
        "restoration": restoration,
        "runtime_seconds": format(time.monotonic() - started, ".6f"),
        "schema": RECEIPT_SCHEMA,
        "score_claim": False,
        "seed": config.seed,
        "source_archive": integration["source_archive"],
        "storage_preflight": storage,
        "typed_config": {"path": str(config_path), "sha256": config_sha256},
        "verdict": (
            "RECEIVER_CLOSED_LOSSLESS_RATE_GAIN_MEASURED;CC2_REUSED_SCORE_PREMISE_"
            + ("CONFIRMED" if reuse_matches_full_receiver else "FALSIFIED_INSTANCE")
        ),
        "verdict_scope": (
            "INSTANCE: exact CC2 27-leaf composition, eight selected lossless frames, "
            "macOS-CPU frozen-scorer advisory only; no contest score, promotion, "
            "dispatch, live campaign mutation, or pointer movement."
        ),
    }
    _publish_json(receipt_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    result = run(Path(args.config).resolve(), Path(args.receipt).resolve())
    print(
        json.dumps(
            {
                "archive": result["archive"],
                "endpoint": {
                    "d_pose": result["endpoint"]["d_pose"],
                    "d_seg": result["endpoint"]["d_seg"],
                },
                "preintegration_reuse_premise": result["preintegration_reuse_premise"],
                "receipt": str(Path(args.receipt).resolve()),
                "verdict": result["verdict"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
