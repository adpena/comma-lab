#!/usr/bin/env python3
"""MP3: lossless leave-one-out and greedy pruning of DX2 corrector families.

The nineteen pruneable objects are causal receiver-code families.  They own no
counted bytes; the 13,515-byte ``hpac_blob`` is a separate, fixed IHS1 neural
model.  This experiment therefore holds that blob and the decoded DX2 field
fixed, cold-refits each surviving family set through the shipped online learner,
and measures the real RC64 stream.  Every stream, reconstructed stored member,
decoded field, and per-stage resume state is retained locally.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

bl1 = importlib.import_module("experiments.ddm_bl1_per_position_bit_allocation")
cp = importlib.import_module("experiments.ddm_cp135_rate_compose")
jg2 = importlib.import_module("experiments.ddm_jg2_tail_reencode")

OUTPUT = REPO / ".omx/tmp/arm_receipts_local/ddm_mp3_hpac_member_prune"
RUNTIME = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2")
TO2 = Path(
    "/Volumes/VertigoDataTier/pact/ddm_to2_token_ordering_race/measurement_v1/retained/input"
)
EXPERIMENT_BOOK = Path("/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book")
ARCHIVE = TO2 / "archive.zip"
TOKENS = TO2 / "dx2_tokens_decoded.u8"
STREAM = TO2 / "dx2_token_stream_rc64.bin"
ENCODER_SOURCE = EXPERIMENT_BOOK / "src/cpr1_sub4/entropy/rc64_backend.c"

EXPECTED = {
    "archive_bytes": 180_368,
    "archive_sha256": "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674",  # gitleaks:allow
    "tokens_bytes": 117_964_800,
    "tokens_sha256": "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb",  # gitleaks:allow
    "stream_bytes": 113_777,
    "stream_sha256": "e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5",  # gitleaks:allow
    "member_bytes": 180_268,
    "member_sha256": "365f1b8d70463b250a2fe95e3599318ac90b31875cce5d66a767819404431c7a",  # gitleaks:allow
    "prefix_bytes": 66_491,
    "prefix_sha256": "0e2dd639e50795a00a3013f1ba66efa06495ed7b0a2ea6bbd920aa50b4ad1877",  # gitleaks:allow
    "hpac_outer_bytes": 13_515,
    "hpac_outer_sha256": "602115b323b0e403d08287af9b273a2d4fb23e026d83c1f6e4609ed77ef98f98",  # gitleaks:allow
    "hpac_ihs1_bytes": 17_952,
    "hpac_ihs1_sha256": "e8c0cfd73d3275adeff2897ea83efa9d045855c43fb3bb66ac037e5c84f2e6dd",  # gitleaks:allow
}

MEMBERS = (
    "shipped_joint",
    "temporal_spatial",
    "surprise_only",
    "spatial_surprise",
    "spatial_boundary",
    "run_surprise",
    "boundary_surprise",
    "temporal_surprise",
    "shipped_fast256",
    "shipped_fast4096",
    "surprise_fast256",
    "spatial4_surprise",
    "homog_surprise",
    "homog_boundary_surprise",
    "spatial4_boundary",
    "homog_spatial4",
    "spatial4_temporal",
    "homog_surprise_fast256",
    "spatial4_surprise_fast256",
)

N = 600
HEIGHT = 384
WIDTH = 512
PLANE = HEIGHT * WIDTH
GROUPS = 190
STAGE_FRAMES = 20
RATE_DENOMINATOR = 37_545_489
S_PER_BYTE = 25.0 / RATE_DENOMINATOR
DEMAND_BYTES = 42_382
LOCAL_RESERVE_BYTES = 20 << 30
PROJECTED_BYTES = 12 << 30
AXIS = "[macOS-CPU advisory / scorer-free exact RC64 byte measurement]"


class Mp3Error(RuntimeError):
    """Fail-closed source, receiver, resume, or retention error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_file(path: Path, digest: str, size: int | None = None) -> dict[str, object]:
    if not path.is_file():
        raise Mp3Error(f"required custody file is absent: {path}")
    fact = file_fact(path)
    if fact["sha256"] != digest or (size is not None and fact["bytes"] != size):
        raise Mp3Error(f"custody drift: {fact}; expected bytes={size}, sha256={digest}")
    return fact


def atomic_bytes(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_fact(path)


def atomic_json(path: Path, payload: object) -> dict[str, object]:
    return atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def atomic_npy(path: Path, value: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_fact(path)


def atomic_npz(path: Path, **values: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_fact(path)


def atomic_copy(source: Path, destination: Path, expected: dict[str, object]) -> dict[str, object]:
    if destination.is_file():
        return verify_file(destination, str(expected["sha256"]), int(expected["bytes"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".partial.{os.getpid()}")
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=1 << 24)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return verify_file(destination, str(expected["sha256"]), int(expected["bytes"]))


def local_preflight(output: Path) -> dict[str, object]:
    resolved = output.resolve()
    allowed = (REPO / ".omx/tmp/arm_receipts_local/ddm_mp3_hpac_member_prune").resolve()
    if resolved != allowed:
        raise Mp3Error(f"output must equal the charter-authorized local root: {allowed}")
    if str(resolved).startswith("/Volumes/"):
        raise Mp3Error("MP3 must not write either full SSD tier")
    usage = shutil.disk_usage(REPO)
    if usage.free < PROJECTED_BYTES + LOCAL_RESERVE_BYTES:
        raise Mp3Error("local tier lacks projected bytes plus the 20-GiB fail-closed reserve")
    return {
        "tier": "local explicit opt-in",
        "root": str(resolved),
        "free_bytes_before": usage.free,
        "projected_bytes": PROJECTED_BYTES,
        "reserve_bytes": LOCAL_RESERVE_BYTES,
        "ssd_writes": False,
    }


def import_rc64() -> Any:
    path = EXPERIMENT_BOOK / "src/cpr1_sub4/entropy/rc64.py"
    spec = importlib.util.spec_from_file_location("_ddm_mp3_rc64", path)
    if spec is None or spec.loader is None:
        raise Mp3Error(f"cannot import RC64 wrapper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_rc64(output: Path) -> tuple[Path, Any]:
    rc64 = import_rc64()
    library = cp._compile_checkpointable_rc64(
        SimpleNamespace(experiment_book=EXPERIMENT_BOOK, output=output)
    )
    atomic_json(
        output / "work/RC64_BUILD.json",
        {
            "schema": "ddm_mp3_rc64_build.v1",
            "source": file_fact(ENCODER_SOURCE),
            "generated_source": file_fact(output / "work/rc64_checkpoint_backend.c"),
            "library": file_fact(library),
        },
    )
    return library, rc64


def load_receiver(library: Path) -> dict[str, Any]:
    import torch

    runtime_text = str(RUNTIME)
    cpr1_text = str(RUNTIME / "cpr1")
    if cpr1_text not in sys.path:
        sys.path.insert(0, cpr1_text)
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    renderer = importlib.import_module("cpr1.inflate")
    residual = importlib.import_module("runtime.residual_archive")
    free_corrector = importlib.import_module("runtime.free_corrector")
    hpac_inference = importlib.import_module("runtime.hpac_inference")
    parts = residual.read_residual_archive(ARCHIVE)
    if (
        len(parts.token_stream) != EXPECTED["stream_bytes"]
        or hashlib.sha256(parts.token_stream).hexdigest() != EXPECTED["stream_sha256"]
    ):
        raise Mp3Error("shipped parser extracted a different token stream")
    with zipfile.ZipFile(ARCHIVE) as archive:
        member = archive.read("p")
    hpac_outer = member[14 : 14 + EXPECTED["hpac_outer_bytes"]]
    if hashlib.sha256(hpac_outer).hexdigest() != EXPECTED["hpac_outer_sha256"]:
        raise Mp3Error("AR1B HPAC outer span no longer reproduces")
    ihs1 = residual.materialize_ihs1(parts.hpac_blob, renderer)
    if len(ihs1) != EXPECTED["hpac_ihs1_bytes"]:
        raise Mp3Error("materialized IHS1 byte count drifted")
    if hashlib.sha256(ihs1).hexdigest() != EXPECTED["hpac_ihs1_sha256"]:
        raise Mp3Error("materialized IHS1 SHA drifted")
    device = torch.device("cpu")
    model = renderer.load_hpac(ihs1, device)
    masks = renderer.group_masks(device)
    sparse = residual._sparse_class(RUNTIME / "cpr1")(model, HEIGHT, WIDTH)
    hpac_inference.optimize_sparse_evaluator(sparse)
    plans = []
    for mask in masks:
        flat = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1)).astype(np.int64)
        plans.append((torch.from_numpy(flat).to(device), flat))
    if len(plans) != GROUPS or sum(len(row[1]) for row in plans) != PLANE:
        raise Mp3Error("shipped 190-group map no longer partitions the plane")
    shipped = dict(free_corrector.SHIPPED_CONFIG)
    families = tuple(shipped["families"])
    if families != MEMBERS:
        raise Mp3Error(f"shipped member order drifted: {families}")
    return {
        "torch": torch,
        "renderer": renderer,
        "residual": residual,
        "free_corrector": free_corrector,
        "shipped_config": shipped,
        "parts": parts,
        "hpac_outer": hpac_outer,
        "ihs1": ihs1,
        "model": model,
        "sparse": sparse,
        "plans": plans,
        "device": device,
        "library": library,
    }


def new_corrector(runtime: dict[str, Any], members: tuple[str, ...]) -> Any:
    config = dict(runtime["shipped_config"])
    config["families"] = members
    return runtime["free_corrector"].Ma1WithinMissCorrector(PLANE, **config)


def source_binding(output: Path, runtime: dict[str, Any]) -> dict[str, object]:
    facts = {
        "archive": verify_file(ARCHIVE, EXPECTED["archive_sha256"], EXPECTED["archive_bytes"]),
        "tokens": verify_file(TOKENS, EXPECTED["tokens_sha256"], EXPECTED["tokens_bytes"]),
        "stream": verify_file(STREAM, EXPECTED["stream_sha256"], EXPECTED["stream_bytes"]),
        "encoder_source": file_fact(ENCODER_SOURCE),
        "shipped_free_corrector": file_fact(RUNTIME / "runtime/free_corrector.py"),
        "shipped_fx2_corrector": file_fact(RUNTIME / "runtime/fx2_model_axis_corrector.py"),
        "shipped_residual_archive": file_fact(RUNTIME / "runtime/residual_archive.py"),
        "implementation": file_fact(Path(__file__)),
    }
    hpac_outer = atomic_bytes(output / "retained/input/hpac_outer.br", runtime["hpac_outer"])
    hpac_ihs1 = atomic_bytes(output / "retained/input/hpac_model.ihs1", runtime["ihs1"])
    stream = atomic_copy(STREAM, output / "retained/input/tokens.rc64", facts["stream"])
    tokens = atomic_copy(TOKENS, output / "retained/input/dx2_tokens_decoded.u8", facts["tokens"])
    result = {
        "schema": "ddm_mp3_source_binding.v1",
        "axis": AXIS,
        "shape": [N, HEIGHT, WIDTH],
        "sources": facts,
        "retained": {
            "hpac_outer": hpac_outer,
            "hpac_ihs1": hpac_ihs1,
            "incumbent_stream": stream,
            "unchanged_field": tokens,
        },
    }
    atomic_json(output / "SOURCE_BINDING.json", result)
    return result


def member_prefix(output: Path) -> tuple[bytes, dict[str, object]]:
    with zipfile.ZipFile(ARCHIVE) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != "p" or infos[0].compress_type != zipfile.ZIP_STORED:
            raise Mp3Error("DX2 archive framing drifted")
        member = archive.read("p")
    stream = STREAM.read_bytes()
    if not member.endswith(stream):
        raise Mp3Error("DX2 token stream is no longer the member suffix")
    if len(member) != EXPECTED["member_bytes"] or hashlib.sha256(member).hexdigest() != EXPECTED["member_sha256"]:
        raise Mp3Error("DX2 stored member drifted")
    prefix = member[: -len(stream)]
    fact = atomic_bytes(output / "retained/input/non_token_prefix.bin", prefix)
    if fact["bytes"] != EXPECTED["prefix_bytes"] or fact["sha256"] != EXPECTED["prefix_sha256"]:
        raise Mp3Error("DX2 non-token prefix drifted")
    return prefix, fact


def write_census(output: Path) -> dict[str, object]:
    rows = [
        {
            "member": member,
            "separable_counted_bytes": 0,
            "reason": "generic causal receiver code; no learned member payload is serialized",
        }
        for member in MEMBERS
    ]
    result = {
        "schema": "ddm_mp3_member_byte_census.v1",
        "hpac_outer_bytes": EXPECTED["hpac_outer_bytes"],
        "member_rows": rows,
        "member_separable_bytes_sum": 0,
        "unattributable_to_members_remainder_bytes": EXPECTED["hpac_outer_bytes"],
        "accounted_total_bytes": EXPECTED["hpac_outer_bytes"],
        "remainder_identity": "fixed 64-channel IHS1 neural HPAC probability model",
        "finding": "the charter premise joins two distinct model surfaces",
    }
    atomic_json(output / "MEMBER_BYTE_CENSUS.json", result)
    return result


def stage_root(output: Path, generation: str, phase: str, start: int, end: int) -> Path:
    return output / "generations" / generation / phase / "stages" / f"frames_{start:04d}_{end - 1:04d}"


def contiguous_receipts(output: Path, generation: str, phase: str) -> list[dict[str, Any]]:
    rows = []
    for start in range(0, N, STAGE_FRAMES):
        end = min(start + STAGE_FRAMES, N)
        root = stage_root(output, generation, phase, start, end)
        path = root / "RECEIPT.json"
        if not path.is_file():
            break
        row = json.loads(path.read_text())
        if row.get("frame_start") != start or row.get("frame_end") != end:
            raise Mp3Error(f"nonmatching {generation}/{phase} stage receipt: {path}")
        for artifact in row["artifacts"].values():
            verify_file(Path(artifact["path"]), str(artifact["sha256"]), int(artifact["bytes"]))
        rows.append(row)
    stage_dir = output / "generations" / generation / phase / "stages"
    all_rows = list(stage_dir.glob("frames_*/RECEIPT.json")) if stage_dir.exists() else []
    if len(all_rows) != len(rows):
        raise Mp3Error(f"{generation}/{phase} receipts are not a contiguous prefix")
    return rows


def state_arrays(
    correctors: dict[str, Any],
    colds: dict[str, Any],
    labels: tuple[str, ...],
    frame_end: int,
    previous: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays = {
        "frame_end": np.asarray([frame_end], dtype=np.int64),
        "previous": np.asarray(previous, dtype=np.uint8),
    }
    for index, label in enumerate(labels):
        captured = jg2.corrector_state(correctors[label])
        lost = jg2.uncaptured_divergent_state(correctors[label], colds[label], set(captured))
        if lost:
            raise Mp3Error(f"checkpoint would lose state for {label}: {lost[:8]}")
        for key, value in captured.items():
            arrays[f"v{index:02d}___{key}"] = value
    return arrays


def restore_states(
    correctors: dict[str, Any], labels: tuple[str, ...], path: Path
) -> tuple[int, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        frame_end = int(payload["frame_end"][0])
        previous = np.asarray(payload["previous"], dtype=np.uint8).copy()
        for index, label in enumerate(labels):
            prefix = f"v{index:02d}___"
            state = {
                key.removeprefix(prefix): payload[key].copy()
                for key in payload.files
                if key.startswith(prefix)
            }
            jg2.load_corrector_state(correctors[label], state)
    return frame_end, previous


def variant_label(removed: tuple[str, ...]) -> str:
    if not removed:
        return "control_all19"
    if len(removed) == 1:
        return "remove__" + removed[0]
    digest = hashlib.sha256("\n".join(removed).encode()).hexdigest()[:12]
    return f"remove_k{len(removed):02d}_{digest}"


def candidate_sets(base: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {
        variant_label(tuple(member for member in MEMBERS if member not in kept)): kept
        for kept in (tuple(member for member in base if member != removed) for removed in base)
    }


def drive_encode(
    output: Path,
    generation: str,
    variants: dict[str, tuple[str, ...]],
    library: Path,
    rc64: Any,
) -> dict[str, Any]:
    result_path = output / "generations" / generation / "ENCODE_RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        for row in result["rows"]:
            verify_file(Path(row["stream"]["path"]), row["stream"]["sha256"], row["stream"]["bytes"])
            verify_file(Path(row["member"]["path"]), row["member"]["sha256"], row["member"]["bytes"])
        return result

    torch_runtime = load_receiver(library)
    torch = torch_runtime["torch"]
    labels = tuple(variants)
    correctors = {label: new_corrector(torch_runtime, variants[label]) for label in labels}
    colds = {label: new_corrector(torch_runtime, variants[label]) for label in labels}
    encoders = {label: rc64.NativeEncoder(library) for label in labels}
    receipts = contiguous_receipts(output, generation, "encode")
    if receipts:
        last = receipts[-1]
        root = stage_root(output, generation, "encode", last["frame_start"], last["frame_end"])
        start_frame, previous_np = restore_states(correctors, labels, root / "corrector_states.npz")
        previous = torch.from_numpy(previous_np.astype(np.int64)).reshape(1, HEIGHT, WIDTH)
        for label in labels:
            encoders[label] = cp._rc64_resume(
                rc64.NativeEncoder, library, (root / f"encoder_{label}.state").read_bytes()
            )
    else:
        start_frame = 0
        previous = torch.zeros((1, HEIGHT, WIDTH), dtype=torch.long)

    truth = np.memmap(TOKENS, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    model = torch_runtime["model"]
    sparse = torch_runtime["sparse"]
    plans = torch_runtime["plans"]
    residual = torch_runtime["residual"]
    parts = torch_runtime["parts"]
    renderer = torch_runtime["renderer"]
    started = time.perf_counter()
    with torch.inference_mode():
        for stage_start in range(start_frame, N, STAGE_FRAMES):
            stage_end = min(stage_start + STAGE_FRAMES, N)
            for frame in range(stage_start, stage_end):
                current = torch.zeros_like(previous)
                context = model.prepare_frame_context(torch.tensor([frame]), previous)
                if frame:
                    previous_cpu = previous[0].to(dtype=torch.uint8).numpy()
                    boundary = residual._boundary_buckets(previous_cpu).reshape(-1)
                else:
                    boundary = np.full(PLANE, 4, dtype=np.uint8)
                for corrector in correctors.values():
                    corrector.begin_frame(boundary)
                plane_target = np.asarray(truth[frame]).reshape(-1)
                for group, (device_positions, flat_positions) in enumerate(plans):
                    base_logits = sparse.selected_logits(current, context, group).cpu().numpy()
                    predicted = base_logits.argmax(axis=1).astype(np.int64)
                    feature = boundary[flat_positions].astype(np.int64) * 5 + predicted
                    corrected = base_logits + parts.table.values[feature]
                    probability = residual._probability_table(corrected, renderer.HPAC_LOGIT_PRECISION)
                    symbols = plane_target[flat_positions].astype(np.int64)
                    for label in labels:
                        state = correctors[label].group_state(probability, predicted, flat_positions)
                        coding = np.asarray(correctors[label].coding_row(state), dtype=np.float32)
                        encoders[label].encode(symbols.astype(np.int32), coding)
                        correctors[label].observe(state, symbols)
                    current.reshape(-1)[device_positions] = torch.from_numpy(symbols)
                frame_tokens = current[0].to(dtype=torch.uint8).numpy()
                if not np.array_equal(frame_tokens, truth[frame]):
                    raise Mp3Error(f"teacher-forced trajectory diverged at frame {frame}")
                for corrector in correctors.values():
                    corrector.end_frame(frame_tokens.reshape(-1))
                previous = current

            root = stage_root(output, generation, "encode", stage_start, stage_end)
            artifacts = {
                "corrector_states": atomic_npz(
                    root / "corrector_states.npz",
                    **state_arrays(
                        correctors,
                        colds,
                        labels,
                        stage_end,
                        previous[0].to(dtype=torch.uint8).numpy(),
                    ),
                )
            }
            for label in labels:
                artifacts[f"encoder_{label}"] = atomic_bytes(
                    root / f"encoder_{label}.state", cp._rc64_snapshot(encoders[label])
                )
            receipt = {
                "schema": "ddm_mp3_encode_stage.v1",
                "generation": generation,
                "frame_start": stage_start,
                "frame_end": stage_end,
                "artifacts": artifacts,
                "elapsed_seconds": time.perf_counter() - started,
            }
            atomic_json(root / "RECEIPT.json", receipt)
            print(
                json.dumps(
                    {
                        "generation": generation,
                        "phase": "encode",
                        "frame_end": stage_end,
                        "elapsed_s": round(time.perf_counter() - started, 3),
                    }
                ),
                flush=True,
            )

    prefix, prefix_fact = member_prefix(output)
    rows = []
    for label in labels:
        payload = encoders[label].finish()
        root = output / "retained/generations" / generation / label
        stream_fact = atomic_bytes(root / "tokens.rc64", payload)
        member_fact = atomic_bytes(root / "member.bin", prefix + payload)
        rows.append(
            {
                "variant": label,
                "members": list(variants[label]),
                "stream": stream_fact,
                "member": member_fact,
                "non_token_prefix": prefix_fact,
            }
        )
    result = {"schema": "ddm_mp3_encode_result.v1", "axis": AXIS, "rows": rows}
    atomic_json(result_path, result)
    return result


def drive_decode(
    output: Path,
    generation: str,
    variants: dict[str, tuple[str, ...]],
    library: Path,
    rc64: Any,
    encode_result: dict[str, Any],
    phase: str = "decode",
) -> dict[str, Any]:
    result_name = "DECODE_RESULT.json" if phase == "decode" else f"{phase.upper()}_RESULT.json"
    result_path = output / "generations" / generation / result_name
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        for row in result["rows"]:
            verify_file(
                Path(row["decoded_tokens"]["path"]),
                row["decoded_tokens"]["sha256"],
                row["decoded_tokens"]["bytes"],
            )
        return result

    rows_by_label = {row["variant"]: row for row in encode_result["rows"]}
    torch_runtime = load_receiver(library)
    torch = torch_runtime["torch"]
    labels = tuple(variants)
    correctors = {label: new_corrector(torch_runtime, variants[label]) for label in labels}
    colds = {label: new_corrector(torch_runtime, variants[label]) for label in labels}
    decoders = {
        label: rc64.NativeDecoder(library, Path(rows_by_label[label]["stream"]["path"]).read_bytes())
        for label in labels
    }
    receipts = contiguous_receipts(output, generation, phase)
    if receipts:
        last = receipts[-1]
        root = stage_root(output, generation, phase, last["frame_start"], last["frame_end"])
        start_frame, previous_np = restore_states(correctors, labels, root / "corrector_states.npz")
        previous = torch.from_numpy(previous_np.astype(np.int64)).reshape(1, HEIGHT, WIDTH)
        with np.load(root / "decoder_states.npz", allow_pickle=False) as saved:
            for label in labels:
                bl1.restore_decoder_state(decoders[label], saved[label])
    else:
        start_frame = 0
        previous = torch.zeros((1, HEIGHT, WIDTH), dtype=torch.long)

    truth = np.memmap(TOKENS, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    model = torch_runtime["model"]
    sparse = torch_runtime["sparse"]
    plans = torch_runtime["plans"]
    residual = torch_runtime["residual"]
    parts = torch_runtime["parts"]
    renderer = torch_runtime["renderer"]
    started = time.perf_counter()
    with torch.inference_mode():
        for stage_start in range(start_frame, N, STAGE_FRAMES):
            stage_end = min(stage_start + STAGE_FRAMES, N)
            decoded_stage = {
                label: np.empty((stage_end - stage_start, HEIGHT, WIDTH), dtype=np.uint8)
                for label in labels
            }
            for frame in range(stage_start, stage_end):
                offset = frame - stage_start
                expected_plane = np.asarray(truth[frame]).reshape(-1)
                current = torch.zeros_like(previous)
                context = model.prepare_frame_context(torch.tensor([frame]), previous)
                if frame:
                    previous_cpu = previous[0].to(dtype=torch.uint8).numpy()
                    boundary = residual._boundary_buckets(previous_cpu).reshape(-1)
                else:
                    boundary = np.full(PLANE, 4, dtype=np.uint8)
                for corrector in correctors.values():
                    corrector.begin_frame(boundary)
                for group, (device_positions, flat_positions) in enumerate(plans):
                    base_logits = sparse.selected_logits(current, context, group).cpu().numpy()
                    predicted = base_logits.argmax(axis=1).astype(np.int64)
                    feature = boundary[flat_positions].astype(np.int64) * 5 + predicted
                    corrected = base_logits + parts.table.values[feature]
                    probability = residual._probability_table(corrected, renderer.HPAC_LOGIT_PRECISION)
                    expected = expected_plane[flat_positions].astype(np.int64)
                    for label in labels:
                        state = correctors[label].group_state(probability, predicted, flat_positions)
                        coding = np.asarray(correctors[label].coding_row(state), dtype=np.float32)
                        decoded = decoders[label].decode(coding).astype(np.int64)
                        if not np.array_equal(decoded, expected):
                            mismatch = int(np.flatnonzero(decoded != expected)[0])
                            raise Mp3Error(
                                f"decode mismatch {label}, frame={frame}, group={group}, index={mismatch}"
                            )
                        correctors[label].observe(state, decoded)
                    current.reshape(-1)[device_positions] = torch.from_numpy(expected)
                frame_tokens = current[0].to(dtype=torch.uint8).numpy()
                for label in labels:
                    decoded_stage[label][offset] = frame_tokens
                    correctors[label].end_frame(frame_tokens.reshape(-1))
                previous = current

            root = stage_root(output, generation, phase, stage_start, stage_end)
            artifacts = {
                "corrector_states": atomic_npz(
                    root / "corrector_states.npz",
                    **state_arrays(
                        correctors,
                        colds,
                        labels,
                        stage_end,
                        previous[0].to(dtype=torch.uint8).numpy(),
                    ),
                ),
                "decoder_states": atomic_npz(
                    root / "decoder_states.npz",
                    **{label: bl1.decoder_state(decoders[label]) for label in labels},
                ),
            }
            for label in labels:
                artifacts[f"decoded_{label}"] = atomic_npy(
                    root / f"decoded_{label}.npy", decoded_stage[label]
                )
            receipt = {
                "schema": "ddm_mp3_decode_stage.v1",
                "generation": generation,
                "frame_start": stage_start,
                "frame_end": stage_end,
                "artifacts": artifacts,
                "elapsed_seconds": time.perf_counter() - started,
            }
            atomic_json(root / "RECEIPT.json", receipt)
            print(
                json.dumps(
                    {
                        "generation": generation,
                        "phase": phase,
                        "frame_end": stage_end,
                        "elapsed_s": round(time.perf_counter() - started, 3),
                    }
                ),
                flush=True,
            )

    rows = []
    for label in labels:
        destination = output / "retained/generations" / generation / label / "decoded_tokens.u8"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f".partial.{os.getpid()}")
        try:
            with temporary.open("wb") as handle:
                for start in range(0, N, STAGE_FRAMES):
                    end = min(start + STAGE_FRAMES, N)
                    value = np.load(
                        stage_root(output, generation, phase, start, end)
                        / f"decoded_{label}.npy",
                        allow_pickle=False,
                    )
                    handle.write(np.ascontiguousarray(value).tobytes())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        fact = verify_file(destination, EXPECTED["tokens_sha256"], EXPECTED["tokens_bytes"])
        rows.append(
            {
                "variant": label,
                "members": list(variants[label]),
                "decoded_tokens": fact,
                "decoded_field_identity": True,
                "decoder_bit_position": decoders[label].bit_position,
            }
        )
    result = {"schema": "ddm_mp3_decode_result.v1", "axis": AXIS, "rows": rows}
    atomic_json(result_path, result)
    return result


def measure_generation(
    output: Path,
    generation: str,
    variants: dict[str, tuple[str, ...]],
    library: Path,
    rc64: Any,
) -> list[dict[str, Any]]:
    encode = drive_encode(output, generation, variants, library, rc64)
    if generation == "loo_k001":
        measured_encode = encode
        decoded_variants = variants
        decode_phase = "decode"
    else:
        best = min(
            encode["rows"],
            key=lambda row: (int(row["stream"]["bytes"]), row["variant"]),
        )
        measured_encode = {**encode, "rows": [best]}
        decoded_variants = {best["variant"]: variants[best["variant"]]}
        decode_phase = "decode_best"
    decode = drive_decode(
        output,
        generation,
        decoded_variants,
        library,
        rc64,
        measured_encode,
        phase=decode_phase,
    )
    decoded = {row["variant"]: row for row in decode["rows"]}
    rows = []
    for row in measured_encode["rows"]:
        delta = int(row["stream"]["bytes"]) - EXPECTED["stream_bytes"]
        item = {
            **row,
            "model_bytes_saved": 0,
            "stream_bytes_added": delta,
            "net_archive_delta_bytes": delta,
            "net_delta_s": delta * S_PER_BYTE,
            "decoded_field": decoded[row["variant"]],
            "distortion_changed": False,
        }
        rows.append(item)
    atomic_json(
        output / "generations" / generation / "MEASUREMENT.json",
        {"schema": "ddm_mp3_generation_measurement.v1", "axis": AXIS, "rows": rows},
    )
    return rows


def control_variant() -> dict[str, tuple[str, ...]]:
    return {"control_all19": MEMBERS, **candidate_sets(MEMBERS)}


def adjudicate_prediction(
    loo_rows: list[dict[str, Any]], optimum_removed: int, freed_bytes: int
) -> dict[str, bool]:
    """Separate the charter's narrow falsifier from its conjunctive prediction."""
    single_negative = any(int(row["net_archive_delta_bytes"]) < 0 for row in loo_rows)
    k_at_least_two = optimum_removed >= 2
    freed_over_1000 = freed_bytes > 1_000
    return {
        "single_member_net_negative_confirmed": single_negative,
        "optimal_pruned_count_at_least_2_confirmed": k_at_least_two,
        "freed_more_than_1000_bytes_confirmed": freed_over_1000,
        "overall_prediction_confirmed": single_negative and k_at_least_two and freed_over_1000,
        "charter_every_loo_positive_falsifier_triggered": all(
            int(row["net_archive_delta_bytes"]) > 0 for row in loo_rows
        ),
    }


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    preflight = local_preflight(output)
    atomic_json(output / "PREFLIGHT.json", preflight)
    launch = {
        "schema": "ddm_mp3_launch_config.v1",
        "command": [sys.executable, str(Path(__file__).resolve()), "--output", str(output)],
        "axis": AXIS,
        "seed": 0,
        "determinism": "no RNG; fixed CPU order; shipped exact RC64",
        "resume": "per-20-frame encoder, decoder, and complete corrector checkpoints",
        "stage_frames": STAGE_FRAMES,
        "field": str(TOKENS),
        "field_mutated": False,
        "scorer_loaded": False,
        "ssd_writes": False,
    }
    atomic_json(output / "LAUNCH_CONFIG.json", launch)
    library, rc64 = compile_rc64(output)
    receiver = load_receiver(library)
    binding = source_binding(output, receiver)
    census = write_census(output)
    initial_rows = measure_generation(output, "loo_k001", control_variant(), library, rc64)
    control = next(row for row in initial_rows if row["variant"] == "control_all19")
    if (
        control["stream"]["bytes"] != EXPECTED["stream_bytes"]
        or control["stream"]["sha256"] != EXPECTED["stream_sha256"]
    ):
        raise Mp3Error(f"mandatory all-19 refit control failed: {control['stream']}")

    loo_rows = [row for row in initial_rows if row["variant"] != "control_all19"]
    if len(loo_rows) != len(MEMBERS):
        raise Mp3Error("leave-one-out table does not contain all 19 members")
    ladder = []
    current = MEMBERS
    k = 1
    current_rows = loo_rows
    previous_delta = 0
    while True:
        best = min(current_rows, key=lambda row: (row["net_archive_delta_bytes"], row["variant"]))
        removed = tuple(member for member in MEMBERS if member not in tuple(best["members"]))
        marginal_delta = int(best["net_archive_delta_bytes"]) - previous_delta
        ladder.append(
            {
                "k_removed": k,
                "removed": list(removed),
                "remaining_member_count": len(best["members"]),
                "stream": best["stream"],
                "model_bytes_saved": 0,
                "stream_bytes_added": best["stream_bytes_added"],
                "net_archive_delta_bytes": best["net_archive_delta_bytes"],
                "net_delta_s": best["net_delta_s"],
                "marginal_archive_delta_bytes": marginal_delta,
                "marginal_delta_s": marginal_delta * S_PER_BYTE,
                "decoded_field": best["decoded_field"],
                "adopted": marginal_delta <= 0,
            }
        )
        if marginal_delta > 0:
            break
        previous_delta = int(best["net_archive_delta_bytes"])
        current = tuple(best["members"])
        if len(current) == 1:
            break
        k += 1
        generation = f"ladder_k{k:03d}"
        current_rows = measure_generation(
            output,
            generation,
            candidate_sets(current),
            library,
            rc64,
        )

    adopted = [row for row in ladder if row["adopted"]]
    optimum_removed = adopted[-1]["k_removed"] if adopted else 0
    optimum_delta = adopted[-1]["net_archive_delta_bytes"] if adopted else 0
    freed_bytes = -optimum_delta
    prediction = adjudicate_prediction(loo_rows, optimum_removed, freed_bytes)
    result = {
        "schema": "ddm_mp3_result.v1",
        "axis": AXIS,
        "source_binding": binding,
        "member_byte_census": census,
        "control": control,
        "leave_one_out": loo_rows,
        "mdl_ladder": ladder,
        "code_length_optimal_pruned_count": optimum_removed,
        "code_length_optimal_member_count": len(MEMBERS) - optimum_removed,
        "freed_bytes": freed_bytes,
        "freed_fraction_of_42382_demand": freed_bytes / DEMAND_BYTES,
        "prior_law_prediction": prediction,
        "prior_law_falsified": not prediction["overall_prediction_confirmed"],
        "shipping_candidate_built": False,
        "scorer_run": False,
        "frontier_moved": False,
    }
    atomic_json(output / "RESULT.json", result)
    write_manifest(output)
    return result


def write_manifest(output: Path) -> dict[str, object]:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            rows.append(file_fact(path))
    payload = {
        "schema": "ddm_mp3_manifest.v1",
        "root": str(output.resolve()),
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
    }
    atomic_json(output / "MANIFEST.json", payload)
    return payload


def self_test() -> None:
    assert len(MEMBERS) == 19
    assert len(set(MEMBERS)) == len(MEMBERS)
    variants = candidate_sets(MEMBERS)
    assert len(variants) == 19
    assert all(len(value) == 18 for value in variants.values())
    assert abs(S_PER_BYTE - 6.658589531221714e-7) < 1e-18
    assert variant_label(()) == "control_all19"
    prediction = adjudicate_prediction(
        [{"net_archive_delta_bytes": -18}, {"net_archive_delta_bytes": 3}], 3, 34
    )
    assert prediction["single_member_net_negative_confirmed"]
    assert prediction["optimal_pruned_count_at_least_2_confirmed"]
    assert not prediction["freed_more_than_1000_bytes_confirmed"]
    assert not prediction["overall_prediction_confirmed"]
    assert not prediction["charter_every_loo_positive_falsifier_triggered"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("ddm_mp3 self-test: PASS")
        return
    result = run(args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
