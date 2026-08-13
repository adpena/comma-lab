#!/usr/bin/env python3
"""Build the SA1 tiny-counted, shipping-axis-targeted Seg actuator.

The local CPU scorer is an ordering instrument only.  Training targets and the
repair mask come from the retained JS1B contest-CUDA T4 argmax fields.  The
selected learned module is inserted as a counted archive member and consumed
by an adapted copy of the real CP135 receiver.  MAIN owns the final T4 sign
gate; this runner emits its pinned fire ticket and never dispatches it.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_js2b_edge_conditioning_relative_gauge as js2b
from experiments import ddm_js3_learned_implicit_conditioning as js3

OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_sa1_20260813")
BASE_ARCHIVE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip"
)
BASE_RUNTIME: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime"
)
T4_FIELDS: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/"
    "contest_cuda/ddm_js1b_20260813b/retained/fields"
)
T4_GT: Final = T4_FIELDS / "gt_argmax_n600.npy"
T4_BASE: Final = T4_FIELDS / "cp135_base_argmax_n600.npy"
STAGE0: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/"
    "contest_cuda/stage0_from_js1b/STAGE0_RESULT.json"
)
BASE_TOKENS: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/"
    "candidates/cp135_base/retained/decoded_tokens_n600.npy"
)
RUNTIME_TEMPLATE: Final = REPO / "experiments/ddm_sa1_runtime/sa1_conditioner.py"

BASE_SHA256: Final = "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
BASE_BYTES: Final = 186_252
T4_GT_SHA256: Final = "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248"
T4_BASE_SHA256: Final = "7648ad42e9f21942f86e81b97cabf46b710af747bba0909f7837ef3891232727"
FIELD_BYTES: Final = 117_964_928
T4_BASE_FLIPS: Final = 34_970
BYTE_BOX: Final = 2_048
POSE_LINEAR_PRICE: Final = 603.0
RATE_PRICE: Final = 25.0 / 37_545_489.0
AXIS: Final = "[macOS-CPU ordering on contest-CUDA T4 target fields; non-promotable]"
N, H, W = 600, 384, 512


class SA1Error(RuntimeError):
    """A custody, byte-box, receiver, resume, or retention invariant failed."""


@contextmanager
def exclusive_run_lock(output: Path) -> Iterator[None]:
    """Prevent two local resumes from writing the same retained run tree."""
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".run.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SA1Error(f"another SA1 runner owns {lock_path}") from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with staging.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)


def retain_bytes(path: Path, payload: bytes) -> None:
    """Create one immutable payload, or require byte-identical resume state."""
    if path.is_file():
        if path.read_bytes() != payload:
            raise SA1Error(f"retained payload differs on resume: {path}")
        return
    atomic_bytes(path, payload)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )


def atomic_npy(path: Path, value: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
    atomic_bytes(path, buffer.getvalue())


def require_file(path: Path, *, size: int | None = None, digest: str | None = None) -> None:
    if not path.is_file():
        raise SA1Error(f"missing required input: {path}")
    if size is not None and path.stat().st_size != size:
        raise SA1Error(f"input byte count differs: {path}")
    if digest is not None and sha256_file(path) != digest:
        raise SA1Error(f"input SHA-256 differs: {path}")


def storage_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    required = 8 * 1024**3
    result = {
        "schema": "ddm_sa1_storage_preflight.v1",
        "tier": str(output),
        "free_bytes": usage.free,
        "required_free_bytes": required,
        "passed": usage.free >= required,
        "cleanup_policy": "block and retain; no generated payload is deleted",
    }
    atomic_json(output / "STORAGE_PREFLIGHT.json", result)
    if not result["passed"]:
        raise SA1Error("SSD storage preflight failed")
    return result


def require_custody() -> dict[str, Any]:
    require_file(BASE_ARCHIVE, size=BASE_BYTES, digest=BASE_SHA256)
    require_file(T4_GT, size=FIELD_BYTES, digest=T4_GT_SHA256)
    require_file(T4_BASE, size=FIELD_BYTES, digest=T4_BASE_SHA256)
    require_file(STAGE0)
    require_file(BASE_TOKENS)
    require_file(RUNTIME_TEMPLATE)
    stage0 = json.loads(STAGE0.read_text())
    if stage0["axis"] != (
        "[contest-CUDA T4 frozen-SegNet argmax fields, n600, batch=16] COMPONENT-ONLY"
    ):
        raise SA1Error("stage-0 target axis differs")
    return {
        "base_archive": file_record(BASE_ARCHIVE),
        "t4_gt": file_record(T4_GT),
        "t4_base": file_record(T4_BASE),
        "stage0": file_record(STAGE0),
        "base_tokens": file_record(BASE_TOKENS),
        "runtime_template": file_record(RUNTIME_TEMPLATE),
    }


def _form_payload(name: str, arrays: dict[str, np.ndarray]) -> bytes:
    header_rows = []
    chunks = []
    for key, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        header_rows.append(
            {"name": key, "dtype": array.dtype.str, "shape": list(array.shape), "bytes": array.nbytes}
        )
        chunks.append(array.tobytes())
    header = json.dumps(
        {"schema": "ddm_sa1_stage_a_form.v1", "name": name, "arrays": header_rows},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return b"SA1A\x01" + len(header).to_bytes(4, "little") + header + b"".join(chunks)


def retain_form(root: Path, name: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    raw = _form_payload(name, arrays)
    coded = brotli.compress(raw, quality=11)
    form_root = root / name
    raw_path = form_root / "payload.raw"
    coded_path = form_root / "payload.br"
    retain_bytes(raw_path, raw)
    retain_bytes(coded_path, coded)
    raw_record = file_record(raw_path)
    coded_record = file_record(coded_path)
    zip_overhead_upper_bound = 160
    return {
        "name": name,
        "raw": raw_record,
        "brotli_q11": coded_record,
        "worst_case_counted_delta_bytes": raw_record["bytes"] + zip_overhead_upper_bound,
        "measured_coded_plus_zip_overhead_upper_bound": (
            coded_record["bytes"] + zip_overhead_upper_bound
        ),
        "fits_byte_box_worst_case": raw_record["bytes"] + zip_overhead_upper_bound <= BYTE_BOX,
    }


def stage_a_design(output: Path) -> dict[str, Any]:
    root = output / "stage_a"
    gt = np.load(T4_GT, mmap_mode="r", allow_pickle=False)
    base = np.load(T4_BASE, mmap_mode="r", allow_pickle=False)
    if gt.shape != (N, H, W) or base.shape != (N, H, W):
        raise SA1Error("T4 field geometry differs")
    mask_path = root / "retained/t4_flip_mask_n600.npy"
    if mask_path.is_file():
        mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        if mask.shape != (N, H, W) or mask.dtype != np.bool_:
            raise SA1Error("retained T4 flip mask differs")
    else:
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.bool_, shape=(N, H, W))
        for start in range(0, N, 16):
            mask[start : start + 16] = gt[start : start + 16] != base[start : start + 16]
        mask.flush()
        del mask
        mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    flips = int(mask.sum())
    if flips != T4_BASE_FLIPS:
        raise SA1Error(f"T4 flip mask count differs: {flips}")

    directed = np.zeros((5, 5), dtype=np.int64)
    g4 = np.zeros((8, 8, 5), dtype=np.int32)
    row_edges = np.linspace(0, H, 9, dtype=np.int64)
    col_edges = np.linspace(0, W, 9, dtype=np.int64)
    for start in range(0, N, 16):
        stop = min(start + 16, N)
        gt_chunk = np.asarray(gt[start:stop])
        base_chunk = np.asarray(base[start:stop])
        wrong = gt_chunk != base_chunk
        if not np.array_equal(np.asarray(mask[start:stop]), wrong):
            raise SA1Error("retained T4 flip mask differs from bound GT/base fields")
        cells = (gt_chunk[wrong].astype(np.int64) * 5 + base_chunk[wrong]).ravel()
        directed += np.bincount(cells, minlength=25).reshape(5, 5)
        for r in range(8):
            for c in range(8):
                tile_gt = gt_chunk[:, row_edges[r] : row_edges[r + 1], col_edges[c] : col_edges[c + 1]]
                tile_base = base_chunk[:, row_edges[r] : row_edges[r + 1], col_edges[c] : col_edges[c + 1]]
                tile_wrong = tile_gt != tile_base
                if tile_wrong.any():
                    g4[r, c] += np.bincount(tile_gt[tile_wrong], minlength=5).astype(np.int32)

    edge_priority = directed.astype(np.float64)
    edge_priority /= max(1.0, float(edge_priority.sum()))
    edge_values = np.rint(edge_priority * 32767.0).astype("<i2")
    edge_rgb = np.zeros((5, 5, 3), dtype=np.int8)
    g4_priority = np.rint(g4 / max(1, int(g4.max())) * 127.0).astype(np.int8)
    g4_rgb = np.zeros((8, 8, 5, 3), dtype=np.int8)

    torch = __import__("torch")
    torch.manual_seed(js3.SEED + 4)
    conv = js3.build_model(torch, __import__("torch.nn.functional", fromlist=["functional"]), 4, 6.0)
    conv_export = js3.serialize_module(conv, "int8", root / "forms/context_conv_h4")
    forms = [
        retain_form(
            root / "forms",
            "per_edge_threshold",
            {"edge_priority_q15": edge_values, "rgb_delta_i8": edge_rgb},
        ),
        retain_form(
            root / "forms",
            "g4_token_bias",
            {"g4_priority_i8": g4_priority, "rgb_delta_i8": g4_rgb},
        ),
        {
            "name": "context_conv_h4",
            "raw": conv_export.report["raw"],
            "brotli_q11": conv_export.report["coded"],
            "worst_case_counted_delta_bytes": int(conv_export.report["raw_bytes"]) + 160,
            "measured_coded_plus_zip_overhead_upper_bound": int(
                conv_export.report["brotli_q11_bytes"]
            )
            + 160,
            "fits_byte_box_worst_case": int(conv_export.report["raw_bytes"]) + 160 <= BYTE_BOX,
        },
    ]
    if not all(row["fits_byte_box_worst_case"] for row in forms):
        raise SA1Error("Stage-A emitted an over-box mechanism")
    result = {
        "schema": "ddm_sa1_stage_a_design.v1",
        "axis": "[contest-CUDA T4 retained argmax fields; scorer-free design]",
        "t4_flips": flips,
        "road_incident_flips": 28_549,
        "directed_confusion_gt_by_base": directed.tolist(),
        "g4_gt_error_counts": g4.tolist(),
        "flip_mask": file_record(mask_path),
        "byte_box": BYTE_BOX,
        "forms": forms,
        "selected_form": "context_conv_h4",
        "selection_reason": (
            "all three fit; context_conv_h4 is the only form with prior measured robust movement "
            "through the current receiver/R chain, while retaining edge and g4 context inside 563 parameters"
        ),
        "f1_fired": False,
    }
    atomic_json(root / "STAGE_A_RESULT.json", result)
    return result


def materialize_float32_pre_r(context: Any, output: Path) -> np.ndarray:
    path = output / "training/inputs/retained/base_pre_r_n32.float32.npy"
    if path.is_file():
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.shape != (js3.SAMPLE_N, 3, H, W) or value.dtype != np.float32:
            raise SA1Error("retained fp32 pre-R surface differs")
        return value
    torch = context.modules.torch
    renderer = context.modules.renderer_runtime.SemanticTokenRenderer(96).eval()
    renderer.load_state_dict(
        {
            record.schema.name: torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32))
            for record in context.records
        },
        strict=True,
    )
    tokens = torch.from_numpy(np.asarray(context.tokens[context.sample]).astype(np.int64, copy=True))
    indices = torch.from_numpy(context.sample.copy()).long()
    chunks = []
    with torch.inference_mode():
        for start in range(0, js3.SAMPLE_N, js3.BATCH):
            chunks.append(renderer(tokens[start : start + js3.BATCH], indices[start : start + js3.BATCH]).cpu().numpy())
    value = np.concatenate(chunks).astype(np.float32, copy=False)
    atomic_npy(path, value)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def deterministic_archive(base_archive: Path, module: bytes) -> bytes:
    with zipfile.ZipFile(base_archive) as archive:
        if archive.namelist() != ["p"]:
            raise SA1Error("CP135 base archive grammar differs")
        payload = archive.read("p")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in (("p", payload), ("sa1_conditioner.br", module)):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value)
    return stream.getvalue()


def build_candidate_archives(output: Path, stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    denominator = N * H * W
    for stage in stages:
        for kind in ("live", "ema"):
            payload_path = Path(stage[kind]["export"]["selected"]["coded"]["path"])
            module = payload_path.read_bytes()
            archive = deterministic_archive(BASE_ARCHIVE, module)
            candidate_root = output / "candidates" / f"{stage['stage']}_{kind}"
            archive_path = candidate_root / "archive.zip"
            repeat_path = candidate_root / "archive.repeat.zip"
            retain_bytes(archive_path, archive)
            retain_bytes(repeat_path, deterministic_archive(BASE_ARCHIVE, module))
            if archive_path.read_bytes() != repeat_path.read_bytes():
                raise SA1Error("candidate archive is not deterministic")
            delta_bytes = len(archive) - BASE_BYTES
            if delta_bytes > BYTE_BOX:
                raise SA1Error("candidate archive exceeds the byte box")
            metrics = stage[kind]["coded_metrics"]
            seg_delta = 100.0 * int(metrics["projected_n600_delta_flips"]) / denominator
            pose_delta = float(metrics["pose_delta_stratified_n32"])
            joint = seg_delta + POSE_LINEAR_PRICE * pose_delta + RATE_PRICE * delta_bytes
            rows.append(
                {
                    "stage": stage["stage"],
                    "step": stage["step"],
                    "kind": kind,
                    "module": file_record(payload_path),
                    "archive": file_record(archive_path),
                    "archive_repeat": file_record(repeat_path),
                    "delta_bytes_vs_cp135": delta_bytes,
                    "local_ordering": {
                        "axis": AXIS,
                        "projected_n600_delta_flips": int(metrics["projected_n600_delta_flips"]),
                        "projected_n600_robust_delta_flips": int(
                            metrics["projected_n600_robust_delta_flips"]
                        ),
                        "pose_delta_stratified_n32": pose_delta,
                        "pose_guard_pass": bool(metrics["pose_guard_pass"]),
                        "seg_delta_s": seg_delta,
                        "pose_linearized_delta_s": POSE_LINEAR_PRICE * pose_delta,
                        "rate_delta_s": RATE_PRICE * delta_bytes,
                        "joint_linearized_delta_s": joint,
                        "verdict_authority": False,
                    },
                    "training_camera": metrics["payloads"]["camera"],
                    "training_correction": metrics["payloads"]["correction"],
                }
            )
    atomic_json(output / "candidates/CANDIDATE_ROWS.json", {"rows": rows})
    return rows


def replace_once(path: Path, old: str, new: str) -> None:
    value = path.read_text()
    if value.count(old) != 1:
        raise SA1Error(f"runtime adapter expected one source match in {path}: {old[:80]!r}")
    atomic_bytes(path, value.replace(old, new).encode())


def adapt_runtime(winner: dict[str, Any], output: Path) -> Path:
    winner_id = f"{winner['stage']}_{winner['kind']}"
    destination = output / "winner" / winner_id / "adapted_runtime"
    if destination.exists():
        retained_archive = destination / "archive.zip"
        if (
            retained_archive.is_file()
            and sha256_file(retained_archive) == winner["archive"]["sha256"]
            and (destination / "runtime/sa1_conditioner.py").is_file()
        ):
            return destination
        raise SA1Error(
            "retained adapted runtime differs; preserve it and choose a fresh --output"
        )
    staging = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    if staging.exists():
        raise SA1Error(f"retained partial runtime requires custody review: {staging}")
    shutil.copytree(
        BASE_RUNTIME,
        staging,
        ignore=shutil.ignore_patterns("archive.zip", "__pycache__", "*.pyc", "._*", ".DS_Store"),
    )
    shutil.copy2(RUNTIME_TEMPLATE, staging / "runtime/sa1_conditioner.py")
    shutil.copy2(Path(winner["archive"]["path"]), staging / "archive.zip")

    residual = staging / "runtime/residual_archive.py"
    replace_once(
        residual,
        'if archive.namelist() != ["p"]:\n            raise ResidualArchiveError("archive must contain exactly member p")',
        'if archive.namelist() != ["p", "sa1_conditioner.br"]:\n            raise ResidualArchiveError("SA1 archive must contain p plus counted conditioner")',
    )

    f26 = staging / "runtime/f26_inflate.py"
    replace_once(f26, "import time\n", "import time\nimport zipfile\n")
    replace_once(
        f26,
        "    parts = read_residual_archive(archive_path)\n",
        '    with zipfile.ZipFile(archive_path) as archive:\n'
        '        if archive.namelist() != ["p", "sa1_conditioner.br"]:\n'
        '            raise InflationError("SA1 archive members differ")\n'
        '        conditioner_blob = archive.read("sa1_conditioner.br")\n'
        "    parts = read_residual_archive(archive_path)\n",
    )
    replace_once(
        f26,
        "    renderer.render_video(semantic, basis, coefficients, tokens, destination, device)\n",
        "    renderer.render_video(semantic, basis, coefficients, tokens, destination, device, conditioner_blob=conditioner_blob)\n",
    )

    renderer = staging / "cpr1/inflate.py"
    replace_once(
        renderer,
        "from torch.nn import functional\n",
        "from torch.nn import functional\nfrom runtime.sa1_conditioner import apply_conditioner\n",
    )
    replace_once(
        renderer,
        "def render_video(semantic, basis, coefficients, tokens, destination: Path, device):\n",
        "def render_video(semantic, basis, coefficients, tokens, destination: Path, device, *, conditioner_blob):\n",
    )
    replace_once(
        renderer,
        "        master = (\n            functional.interpolate(\n                semantic(tokens[start:end].long().to(device), indices),\n",
        "        pre_r = semantic(tokens[start:end].long().to(device), indices)\n"
        "        pre_r = apply_conditioner(conditioner_blob, tokens[start:end].long().to(device), pre_r)\n"
        "        master = (\n            functional.interpolate(\n                pre_r,\n",
    )

    outer = staging / "inflate.py"
    replace_once(outer, f'ARCHIVE_SHA256 = "{BASE_SHA256}"', f'ARCHIVE_SHA256 = "{winner["archive"]["sha256"]}"')
    replace_once(outer, f"ARCHIVE_BYTES = {BASE_BYTES:_}", f"ARCHIVE_BYTES = {winner['archive']['bytes']:_}")
    replace_once(
        outer,
        '        if archive.namelist() != ["p"]:\n            raise ValueError("archive.zip must contain exactly the payload file p")',
        '        if archive.namelist() != ["p", "sa1_conditioner.br"]:\n            raise ValueError("archive.zip must contain p plus the counted SA1 module")',
    )
    os.replace(staging, destination)
    return destination


def receiver_probe(winner: dict[str, Any], runtime: Path, context: Any, output: Path) -> dict[str, Any]:
    winner_id = f"{winner['stage']}_{winner['kind']}"
    probe_root = output / "winner" / winner_id / "receiver_probe_cpu_surface_v2"
    probe_root.mkdir(parents=True, exist_ok=True)
    pair_id = int(context.sample[0])
    camera_path = probe_root / "camera_pair0.uint8.npy"
    correction_path = probe_root / "correction_pair0.float32.npy"
    pre_r_path = probe_root / "pre_r_pair0.float32.npy"
    receipt_path = probe_root / "RECEIVER_PROBE.json"
    if receipt_path.is_file():
        retained = json.loads(receipt_path.read_text())
        for key in ("runtime_camera", "runtime_correction", "runtime_pre_r"):
            record = retained[key]
            require_file(Path(record["path"]), size=record["bytes"], digest=record["sha256"])
        if not retained["receiver_surface_consumes_counted_module"]:
            raise SA1Error("retained receiver probe did not prove counted-module consumption")
        return retained
    if any(path.exists() for path in (camera_path, correction_path, pre_r_path)):
        raise SA1Error(
            "unreceipted receiver-probe payload is retained; preserve it and choose a fresh --output"
        )
    script = r'''
import sys, zipfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional

def atomic_npy(path, value):
    staging = path.with_name('.' + path.name + '.partial')
    with staging.open('wb') as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
    staging.replace(path)

runtime = Path(sys.argv[1])
archive_path = runtime / "archive.zip"
tokens_path = Path(sys.argv[2])
pair_id = int(sys.argv[3])
camera_path = Path(sys.argv[4])
correction_path = Path(sys.argv[5])
pre_r_path = Path(sys.argv[6])
sys.path.insert(0, str(runtime))
from runtime.f26_inflate import _load_renderer
from runtime.residual_archive import read_residual_archive
from runtime.entropy.renderer_weight_codec import decode_wans1
from runtime.sa1_conditioner import apply_conditioner

renderer = _load_renderer(runtime / "cpr1")
parts = read_residual_archive(archive_path)
semantic = renderer.SemanticTokenRenderer(96).eval()
records = decode_wans1(parts.semantic_blob)
semantic.load_state_dict({r.schema.name: torch.from_numpy(np.ascontiguousarray(r.values, dtype=np.float32)) for r in records}, strict=True)
tokens_np = np.load(tokens_path, mmap_mode="r", allow_pickle=False)
tokens = torch.from_numpy(np.asarray(tokens_np[pair_id:pair_id+1]).astype(np.int64, copy=True))
indices = torch.tensor([pair_id], dtype=torch.long)
with zipfile.ZipFile(archive_path) as archive:
    module = archive.read("sa1_conditioner.br")
with torch.inference_mode():
    pre_r = semantic(tokens, indices)
    corrected = apply_conditioner(module, tokens, pre_r)
    camera = functional.interpolate(corrected, size=(renderer.CAMERA_H, renderer.CAMERA_W), mode="bilinear", align_corners=False).clamp(0.0, 255.0).round().to(torch.uint8)
atomic_npy(camera_path, camera.permute(0, 2, 3, 1).cpu().numpy())
atomic_npy(correction_path, (corrected - pre_r).cpu().numpy())
atomic_npy(pre_r_path, pre_r.cpu().numpy())
'''
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(runtime),
            str(BASE_TOKENS),
            str(pair_id),
            str(camera_path),
            str(correction_path),
            str(pre_r_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    atomic_bytes(probe_root / "stdout.log", completed.stdout.encode())
    atomic_bytes(probe_root / "stderr.log", completed.stderr.encode())
    if completed.returncode:
        raise SA1Error("adapted receiver probe failed")
    observed = np.load(camera_path, allow_pickle=False)
    observed_pre_r = np.load(pre_r_path, allow_pickle=False)
    expected_store = np.load(winner["training_camera"]["path"], mmap_mode="r", allow_pickle=False)
    expected = np.asarray(expected_store[0:1])
    exact = np.array_equal(observed, expected)
    training_pre_r = np.load(
        output / "training/inputs/retained/base_pre_r_n32.float32.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    training_pre_r_pair = np.asarray(training_pre_r[0:1])
    pre_r_exact = np.array_equal(observed_pre_r, training_pre_r_pair)
    pre_r_max_abs = float(np.max(np.abs(observed_pre_r - training_pre_r_pair)))
    camera_abs = np.abs(observed.astype(np.int16) - expected.astype(np.int16))
    camera_mismatch_values = int(np.count_nonzero(camera_abs))
    camera_max_abs = int(camera_abs.max())
    training_correction = np.load(
        winner["training_correction"]["path"], mmap_mode="r", allow_pickle=False
    )
    observed_correction = np.load(correction_path, allow_pickle=False)
    correction_max_abs = float(
        np.max(np.abs(observed_correction - np.asarray(training_correction[0:1], dtype=np.float32)))
    )
    changed = int(np.count_nonzero(observed != np.asarray(context.base_pairs[0:1, 1])))
    bounded_cpu_kernel_drift = (
        pre_r_max_abs <= 0.001
        and camera_max_abs <= 1
        and camera_mismatch_values <= 256
        and correction_max_abs <= 0.001
    )
    result = {
        "schema": "ddm_sa1_receiver_probe.v1",
        "pair_id": pair_id,
        "runtime_camera": file_record(camera_path),
        "runtime_correction": file_record(correction_path),
        "runtime_pre_r": file_record(pre_r_path),
        "training_camera": winner["training_camera"],
        "training_correction": winner["training_correction"],
        "pre_r_exact_to_training_object": pre_r_exact,
        "pre_r_max_abs_vs_training_cpu_process": pre_r_max_abs,
        "camera_exact_to_parseback_training_object": exact,
        "camera_mismatch_values_vs_training_cpu_process": camera_mismatch_values,
        "camera_max_abs_vs_training_cpu_process": camera_max_abs,
        "correction_max_abs_vs_retained_float16_training_object": correction_max_abs,
        "camera_values_changed_vs_cp135": changed,
        "bounded_cpu_kernel_drift": bounded_cpu_kernel_drift,
        "receiver_surface_consumes_counted_module": bounded_cpu_kernel_drift and changed > 0,
        "exact_receiver_entrypoint_used": False,
        "exact_t4_receiver_proof_owed": True,
        "returncode": completed.returncode,
        "stdout": file_record(probe_root / "stdout.log"),
        "stderr": file_record(probe_root / "stderr.log"),
    }
    atomic_json(receipt_path, result)
    if not result["receiver_surface_consumes_counted_module"]:
        raise SA1Error("local receiver surface did not consume the counted module within bounds")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    preflight = storage_preflight(output)
    atomic_json(
        output / "state.json",
        {
            "schema": "ddm_sa1_state.v1",
            "status": "RUNNING",
            "complete": False,
            "resumable": True,
            "last_completed_stage": "storage_preflight",
            "t4_dispatched": False,
        },
    )
    custody = require_custody()
    atomic_json(output / "CUSTODY.json", custody)
    stage_a = stage_a_design(output)
    atomic_json(
        output / "state.json",
        {
            "schema": "ddm_sa1_state.v1",
            "status": "RUNNING",
            "complete": False,
            "resumable": True,
            "last_completed_stage": "stage_a",
            "stage_a": file_record(output / "stage_a/STAGE_A_RESULT.json"),
            "t4_dispatched": False,
        },
    )

    training_root = output / "training"
    context = js2b.build_context(training_root)
    t4_gt = np.load(T4_GT, mmap_mode="r", allow_pickle=False)
    t4_base = np.load(T4_BASE, mmap_mode="r", allow_pickle=False)
    context.gt_labels = t4_gt
    context.custody_argmax = t4_base
    pre_r = materialize_float32_pre_r(context, output)
    target_bindings = {
        "gt_argmax_sha256": T4_GT_SHA256,
        "cp135_argmax_sha256": T4_BASE_SHA256,
        "flip_mask_sha256": stage_a["flip_mask"]["sha256"],
        "selection": "seeded stratified-random n32 from full n600 T4 field",
    }
    train_args = argparse.Namespace(
        output=training_root,
        hidden=4,
        max_delta=args.max_delta,
        lr=args.lr,
        stage_steps=args.stage_steps,
        checkpoint_every=1,
        pose_every=1,
        pose_weight=args.pose_weight,
        ema_decay=args.ema_decay,
        grad_clip=args.grad_clip,
        max_wall_seconds=args.max_wall_seconds,
        resume=args.resume,
        target_bindings=target_bindings,
    )
    started = time.perf_counter()
    training = js3.train(train_args, context, pre_r)
    candidates = build_candidate_archives(output, training["stages"])
    eligible = [
        row
        for row in candidates
        if int(row["local_ordering"]["projected_n600_delta_flips"]) < 0
        and bool(row["local_ordering"]["pose_guard_pass"])
    ]
    if not eligible:
        raise SA1Error(
            "F2: no byte-closed candidate jointly improves local Seg and preserves pose"
        )
    winner = min(
        eligible,
        key=lambda row: (
            float(row["local_ordering"]["joint_linearized_delta_s"]),
            int(row["delta_bytes_vs_cp135"]),
            int(row["step"]),
            str(row["kind"]),
        ),
    )
    runtime = adapt_runtime(winner, output)
    proof = receiver_probe(winner, runtime, context, output)
    runtime_manifest = []
    for path in sorted(runtime.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            runtime_manifest.append(
                {"relative_path": path.relative_to(runtime).as_posix(), **file_record(path)}
            )
    winner_root = output / "winner" / f"{winner['stage']}_{winner['kind']}"
    atomic_json(winner_root / "RUNTIME_MANIFEST.json", {"files": runtime_manifest})
    runtime_manifest_record = file_record(winner_root / "RUNTIME_MANIFEST.json")

    fire = {
        "schema": "ddm_sa1_ready_to_fire.v1",
        "status": "READY_TO_FIRE",
        "owner": "MAIN",
        "consumer_store": str(output / "t4_sign_gate_v3"),
        "fire_trigger": (
            "MAIN reconciles active claims and Modal single-flight, then fires this candidate-only "
            "T4 sign gate; admission requires fewer than 34,970 flips after pose-priced byte break-even"
        ),
        "candidate_archive": winner["archive"],
        "candidate_runtime": str(runtime),
        "command": [
            "PYTHONPATH=src:upstream:$PWD",
            ".venv/bin/modal",
            "run",
            "--detach",
            "experiments/ddm_sa1_modal_t4_sign_gate.py::main",
            "--candidate-archive",
            winner["archive"]["path"],
            "--candidate-runtime",
            str(runtime),
            "--run-id",
            "ddm_sa1_t4_sign_20260813",
            "--resume-from",
            "ddm_sa1_t4_sign_20260813",
            "--lane-id",
            "ddm_sa1_t4_sign_gate",
            "--instance-job-id",
            "modal:ddm_sa1_t4_sign_20260813",
            "--claim-agent",
            "main:ddm_sa1",
            "--output-dir",
            str(output / "t4_sign_gate_v3"),
            "--provider-detach-ack",
        ],
        "dispatched": False,
    }
    atomic_json(output / "READY_TO_FIRE.json", fire)
    result = {
        "schema": "ddm_sa1_final_result.v1",
        "status": "READY_TO_FIRE",
        "axis": AXIS,
        "score_claim": False,
        "pointer_moved": False,
        "elapsed_seconds": time.perf_counter() - started,
        "storage_preflight": preflight,
        "custody": custody,
        "stage_a": stage_a,
        "training": {
            "steps_completed": training["steps_completed"],
            "seconds_per_step": training["seconds_per_step"],
            "target_bindings": target_bindings,
            "stage_count": len(training["stages"]),
            "candidate_denominator": len(candidates),
        },
        "winner": winner,
        "receiver_probe": proof,
        "runtime_manifest": runtime_manifest_record,
        "ready_to_fire": file_record(output / "READY_TO_FIRE.json"),
        "t4_gate": {
            "status": "PENDING_MAIN",
            "base_flips": T4_BASE_FLIPS,
            "candidate_flips": None,
            "joint_delta_s": None,
            "dispatched": False,
        },
        "boundaries": {
            "measured": (
                "full-n600 T4 target-map structure and payload bytes; stratified-n32 local ordering, "
                "pose guard, real coder, exact archive delta, and one-pair local CPU receiver-surface consumption"
            ),
            "not_measured": (
                "candidate contest-CUDA exact receiver/Seg/Pose fields, exact score, full-n600 local scorer, or frontier movement"
            ),
            "local_forward_is_ordering_only": True,
            "cp135_pose_carrier_bytes_untouched": True,
        },
    }
    atomic_json(output / "FINAL_RESULT.json", result)
    state = {
        "schema": "ddm_sa1_state.v1",
        "status": "READY_TO_FIRE",
        "complete": True,
        "resumable": True,
        "final_result": file_record(output / "FINAL_RESULT.json"),
        "winner_archive": winner["archive"],
        "t4_dispatched": False,
    }
    atomic_json(output / "state.json", state)
    return result


def parse_stage_steps(value: str) -> tuple[int, ...]:
    return js3.parse_stage_steps(value)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", type=Path, default=OUTPUT)
    value.add_argument("--stage-steps", type=parse_stage_steps, default=(1, 2, 4, 8))
    value.add_argument("--max-delta", type=float, default=6.0)
    value.add_argument("--lr", type=float, default=0.02)
    value.add_argument("--pose-weight", type=float, default=1_000.0)
    value.add_argument("--ema-decay", type=float, default=0.99)
    value.add_argument("--grad-clip", type=float, default=5.0)
    value.add_argument("--max-wall-seconds", type=float, default=1_800.0)
    value.add_argument("--resume", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    with exclusive_run_lock(args.output.resolve()):
        result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
