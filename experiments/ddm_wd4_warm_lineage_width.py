#!/usr/bin/env python3
"""Price receiver-closed warm width removals from the exact fx5 semantic state.

This instrument is deliberately scorer-free.  It decodes the semantic tensors
from the exact promoted fx5 archive, forms deterministic nested channel subsets,
serializes every subset through the real int4 packet and Brotli-q11 path, and
retains a complete candidate archive for every row.  The best threshold-clearing
row is also bound to an additive public receiver copy for later governed scoring.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import struct
import sys
import zipfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _root in (REPO, SRC):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments import ddm_wd2_student_receiver as receiver
from tac.admission_guard import assert_governed_admission

OUTPUT_ROOT = Path("/Volumes/APDataStore/pact/ddm_wd4_warm_lineage_width")
FX5_RUNTIME = Path("/Volumes/APDataStore/pact/ddm_fx5/candidate_runtime_fx5")
FX5_ARCHIVE = FX5_RUNTIME / "archive.zip"
STAGE08 = Path(
    "/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/"
    "artifacts/checkpoints/semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt"
)
STAGE07 = STAGE08.with_name("semantic_renderer_w96_b4_qat4_12k.pt")
HPAC_E960_ROOT = Path(
    "/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/"
    "full_e480b_e960"
)
HPAC_SELECTION = HPAC_E960_ROOT / "endpoint_closure/checkpoint_selection.json"
MC36_CACHE = Path(
    "/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/inputs/"
    "mc36_spatial_tokens_uint8.pt"
)
FX5_RAW = Path("/Volumes/APDataStore/pact/ddm_fx5/decode_r1/inflated/0.raw")
FX5_DECODE_RECEIPT = Path("/Volumes/APDataStore/pact/ddm_fx5/retained/FX5_DECODE_IDENTITY.json")

FX5_ARCHIVE_BYTES = 180_386
FX5_ARCHIVE_SHA256 = "4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841"
FX5_SCORE = 0.14823186109359
FX5_D_SEG = 0.00020139
TARGET_SCORE = 0.12
STAGE08_SHA256 = "3948ccfcd44778dc42affee18a10c3f3baa434d1a2eb2345a013146c1dbfb647"
STAGE07_SHA256 = "1549607db224ea2c4681738dbcc80d2ba9dd453de72db1cf60309985d0602eaf"
MC36_CACHE_SHA256 = "f53db4e8e65789d7d0442e97f8531bfb9765f41a2c37c8509c6ccdaeb8a6c888"
FX5_RAW_SHA256 = "7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7"
REQUIRED_COMPLETE_ARCHIVE_SAVING = 12_155
RATE_DENOMINATOR = 37_545_489
RX1_HEADER = struct.Struct("<4sBBBBHHH")
WIDTHS = (88, 80, 72, 64, 56, 48, 40, 32)
SELECTORS = ("salience", "prefix")
AXIS = "[macOS-CPU scorer-free exact fx5 byte/container + receiver parse-back]"
SEED = 20260821
LOCAL_LANE_ID = "ddm_wd4_warm_lineage_width"
CLAIMS_LEDGER = REPO / ".omx/state/active_lane_dispatch_claims.md"


class WD4Error(RuntimeError):
    """Raised when exact custody, serialization, or receiver closure fails."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise WD4Error(f"required file is absent: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_file(path: Path, *, size: int | None = None, sha256: str | None = None) -> dict[str, Any]:
    record = file_record(path)
    if size is not None and record["bytes"] != size:
        raise WD4Error(f"input byte count changed: {path}")
    if sha256 is not None and record["sha256"] != sha256:
        raise WD4Error(f"input SHA-256 changed: {path}")
    return record


def atomic_bytes(path: Path, value: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_record(path)


def atomic_json(path: Path, value: object) -> dict[str, Any]:
    return atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )


def atomic_torch(path: Path, value: object) -> dict[str, Any]:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return atomic_bytes(path, buffer.getvalue())


def deterministic_zip(member: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, member)
    return output.getvalue()


def storage_preflight(output: Path, required_free_bytes: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    resolved = output.resolve()
    if resolved != OUTPUT_ROOT and OUTPUT_ROOT not in resolved.parents:
        raise WD4Error("output is outside the charter-mandated APDataStore root")
    free = shutil.disk_usage(output).free
    if free < required_free_bytes:
        raise WD4Error(f"need {required_free_bytes} free bytes, observed {free}")
    return {
        "status": "PASS",
        "root": str(resolved),
        "required_free_bytes": required_free_bytes,
        "observed_free_bytes": free,
        "cleanup_policy": "certify-or-block; retained payloads are never auto-deleted",
    }


def _clear_runtime_modules() -> dict[str, Any]:
    prior = {
        name: module
        for name, module in sys.modules.items()
        if name == "runtime" or name.startswith("runtime.") or name == "_f26_renderer"
    }
    for name in prior:
        sys.modules.pop(name, None)
    return prior


def load_fx5_state() -> tuple[OrderedDict[str, torch.Tensor], Any, Any]:
    """Decode the exact SM3R state that the fx5 public receiver consumes."""

    prior = _clear_runtime_modules()
    sys.path.insert(0, str(FX5_RUNTIME.resolve()))
    try:
        f26 = importlib.import_module("runtime.f26_inflate")
        parts = f26.read_residual_archive(FX5_ARCHIVE)
        renderer = f26._load_renderer(FX5_RUNTIME / "cpr1")
        template = renderer.SemanticTokenRenderer(96).state_dict()
        state = renderer.unpack_variant_semantic_or_none(parts.semantic_blob, template)
        if state is None:
            raise WD4Error("fx5 no longer carries a tagged semantic state")
        ordered = OrderedDict(
            (name, value.detach().cpu().float().contiguous()) for name, value in state.items()
        )
        if tuple(ordered) != tuple(template) or len(ordered) != 38:
            raise WD4Error("fx5 semantic tensor schema is not the required 38-tensor state")
        return ordered, parts, renderer
    finally:
        sys.path.pop(0)
        for name in list(sys.modules):
            if name == "runtime" or name.startswith("runtime.") or name == "_f26_renderer":
                sys.modules.pop(name, None)
        sys.modules.update(prior)


def _normalized(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().abs().float().reshape(-1)
    return value / value.mean().clamp_min(1e-12)


def salience_order(state: Mapping[str, torch.Tensor]) -> list[int]:
    """Rank inherited 8-channel GroupNorm groups using every channel consumer/emitter."""

    score = torch.zeros(96, dtype=torch.float64)

    def add(value: torch.Tensor) -> None:
        nonlocal score
        score += _normalized(value).double()

    add(state["token_embed.weight"].abs().mean(dim=0))
    add(state["coord_mix.weight"].abs().mean(dim=(1, 2, 3)))
    add(state["coord_mix.weight"][:, :96].abs().mean(dim=(0, 2, 3)))
    add(state["coord_mix.bias"])
    for block in range(4):
        prefix = f"blocks.{block}"
        add(state[f"{prefix}.dw.weight"].abs().mean(dim=(1, 2, 3)))
        add(state[f"{prefix}.dw.bias"])
        add(state[f"{prefix}.pw.weight"].abs().mean(dim=(1, 2, 3)))
        add(state[f"{prefix}.pw.weight"].abs().mean(dim=(0, 2, 3)))
        add(state[f"{prefix}.pw.bias"])
        add(state[f"{prefix}.norm.weight"])
        add(state[f"{prefix}.norm.bias"])
        film_weight = state[f"{prefix}.film.weight"]
        film_bias = state[f"{prefix}.film.bias"]
        add(film_weight[:96].abs().mean(dim=1) + film_weight[96:].abs().mean(dim=1))
        add(film_bias[:96].abs() + film_bias[96:].abs())
    add(state["head.weight"].abs().mean(dim=(0, 2, 3)))
    # The incumbent uses 12 groups of 8. Removing and reordering individual channels
    # would silently change normalization neighborhoods before any training. Rank whole
    # inherited groups and preserve the original order inside each group instead.
    groups = [tuple(range(start, start + 8)) for start in range(0, 96, 8)]
    groups.sort(key=lambda group: (-float(score[list(group)].sum()), group[0]))
    return [channel for group in groups for channel in group]


def slice_dense_state(
    state: Mapping[str, torch.Tensor], indices: Sequence[int]
) -> OrderedDict[str, torch.Tensor]:
    """Remove a nested shared channel subset without changing the mechanism."""

    selected = torch.tensor(list(indices), dtype=torch.long)
    if selected.numel() < 1 or selected.unique().numel() != selected.numel():
        raise WD4Error("channel selection must be nonempty and unique")
    film_selected = torch.cat((selected, selected + 96))
    coordinate_inputs = torch.cat((selected, torch.arange(96, 100, dtype=torch.long)))
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    result["token_embed.weight"] = state["token_embed.weight"].index_select(1, selected)
    result["frame_embed.weight"] = state["frame_embed.weight"].clone()
    result["coord_mix.weight"] = state["coord_mix.weight"].index_select(0, selected).index_select(
        1, coordinate_inputs
    )
    result["coord_mix.bias"] = state["coord_mix.bias"].index_select(0, selected)
    for block in range(4):
        prefix = f"blocks.{block}"
        result[f"{prefix}.dw.weight"] = state[f"{prefix}.dw.weight"].index_select(0, selected)
        result[f"{prefix}.dw.bias"] = state[f"{prefix}.dw.bias"].index_select(0, selected)
        result[f"{prefix}.pw.weight"] = state[f"{prefix}.pw.weight"].index_select(
            0, selected
        ).index_select(1, selected)
        result[f"{prefix}.pw.bias"] = state[f"{prefix}.pw.bias"].index_select(0, selected)
        result[f"{prefix}.norm.weight"] = state[f"{prefix}.norm.weight"].index_select(0, selected)
        result[f"{prefix}.norm.bias"] = state[f"{prefix}.norm.bias"].index_select(0, selected)
        result[f"{prefix}.film.weight"] = state[f"{prefix}.film.weight"].index_select(
            0, film_selected
        )
        result[f"{prefix}.film.bias"] = state[f"{prefix}.film.bias"].index_select(
            0, film_selected
        )
    result["head.weight"] = state["head.weight"].index_select(1, selected)
    result["head.bias"] = state["head.bias"].clone()
    return result


def tensor_manifest(state: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows = []
    for name, value in state.items():
        payload = value.detach().cpu().numpy().astype("<f4", copy=False).tobytes()
        rows.append(
            {
                "name": name,
                "shape": list(value.shape),
                "elements": value.numel(),
                "float32_sha256": sha256_bytes(payload),
                "nonzero_elements": int(torch.count_nonzero(value)),
            }
        )
    return rows


def lineage_receipt(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Separate semantic lineage from the e960 HPAC-only checkpoint bank."""

    stage08 = torch.load(STAGE08, map_location="cpu", weights_only=False)
    stage08_q4 = OrderedDict(
        (
            name,
            receiver.quantize_tensor(
                value, embedding=name.endswith("embed.weight")
            ),
        )
        for name, value in stage08["state_dict"].items()
    )
    comparisons = []
    differing_tensors = 0
    differing_elements = 0
    for name in state:
        current = state[name]
        parent = stage08_q4[name].detach().cpu().float()
        if current.shape != parent.shape:
            raise WD4Error(f"stage-08 tensor shape differs: {name}")
        changed = int(torch.count_nonzero(current != parent))
        differing_tensors += int(changed > 0)
        differing_elements += changed
        comparisons.append(
            {
                "name": name,
                "differing_elements": changed,
                "elements": current.numel(),
                "max_abs_delta": float((current - parent).abs().max()),
            }
        )
    selection = json.loads(HPAC_SELECTION.read_text(encoding="utf-8"))
    return {
        "semantic_parent": {
            "stage08_terminal_checkpoint": require_file(STAGE08, sha256=STAGE08_SHA256),
            "stage07_terminal_checkpoint": require_file(STAGE07, sha256=STAGE07_SHA256),
            "exact_shipping_parent": "decoded fx5 SM3R mode-6 state derived from stage-08",
            "exact_optimizer_checkpoint_exists": False,
            "reason": (
                "SM3R mixed-depth row pruning is a post-training representation transform; "
                "the archive is the only exact state authority and carries no optimizer state"
            ),
            "stage08_comparison": {
                "differing_tensors": differing_tensors,
                "differing_elements": differing_elements,
                "rows": comparisons,
            },
        },
        "e960_scope": {
            "kind": "HPAC probability-model training only",
            "root": str(HPAC_E960_ROOT),
            "selection_receipt": file_record(HPAC_SELECTION),
            "selected_checkpoint": selection.get("selected_checkpoint"),
            "semantic_state_changed": False,
        },
    }


def source_container() -> dict[str, Any]:
    archive = FX5_ARCHIVE.read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        if bundle.namelist() != ["p"]:
            raise WD4Error("fx5 ZIP members changed")
        member = bundle.read("p")
    magic, version, codec, table_mode, reserved, hpac_bytes, semantic_bytes, carrier_bytes = (
        RX1_HEADER.unpack_from(member)
    )
    if (magic, version, codec, table_mode, reserved) != (b"RX1M", 1, 2, 0, 0x0A):
        raise WD4Error("fx5 RX1 header changed")
    offset = RX1_HEADER.size
    hpac = member[offset : offset + hpac_bytes]
    offset += hpac_bytes
    semantic = member[offset : offset + semantic_bytes]
    offset += semantic_bytes
    carrier = member[offset : offset + carrier_bytes]
    offset += carrier_bytes
    tail = member[offset:]
    if min(map(len, (hpac, semantic, carrier, tail))) <= 0:
        raise WD4Error("fx5 container has an empty required field")
    return {
        "archive": archive,
        "member": member,
        "magic": magic,
        "version": version,
        "codec": codec,
        "table_mode": table_mode,
        "reserved": reserved,
        "hpac": hpac,
        "semantic": semantic,
        "carrier": carrier,
        "tail": tail,
    }


def ck2_interleave(body: bytes) -> bytes:
    span = len(body) & ~1
    planes = np.frombuffer(body[:span], dtype=np.uint8)
    return planes[0::2].tobytes() + planes[1::2].tobytes() + body[span:]


def ck2_uninterleave(body: bytes) -> bytes:
    span = len(body) & ~1
    half = span // 2
    restored = np.empty(span, dtype=np.uint8)
    planes = np.frombuffer(body[:span], dtype=np.uint8)
    restored[0::2] = planes[:half]
    restored[1::2] = planes[half:]
    return restored.tobytes() + body[span:]


def build_archive(container: Mapping[str, Any], packet: bytes) -> tuple[bytes, bytes, bytes]:
    semantic_stream = brotli.compress(
        ck2_interleave(packet), mode=brotli.MODE_GENERIC, quality=11
    )
    if ck2_uninterleave(brotli.decompress(semantic_stream)) != packet:
        raise WD4Error("semantic Brotli+CK2 round-trip differs")
    if max(len(container["hpac"]), len(semantic_stream), len(container["carrier"])) > 0xFFFF:
        raise WD4Error("RX1 uint16 section ceiling exceeded")
    model = RX1_HEADER.pack(
        container["magic"],
        container["version"],
        container["codec"],
        container["table_mode"],
        container["reserved"],
        len(container["hpac"]),
        len(semantic_stream),
        len(container["carrier"]),
    ) + container["hpac"] + semantic_stream + container["carrier"]
    member = model + container["tail"]
    return semantic_stream, member, deterministic_zip(member)


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise WD4Error(f"runtime patch point differs: {label}")
    return source.replace(old, new)


def patch_runtime(source: Path, destination: Path, archive_path: Path) -> dict[str, Any]:
    """Add one WD2S dispatch to the exact fx5 runtime and bind one archive."""

    if destination.exists():
        expected_archive = file_record(archive_path)
        retained_archive = file_record(destination / "archive.zip")
        if retained_archive["sha256"] != expected_archive["sha256"]:
            raise WD4Error("retained runtime is bound to a different candidate")
        f26_path = destination / "runtime/f26_inflate.py"
        residual_path = destination / "runtime/residual_archive.py"
        if "WD2S" not in f26_path.read_text(encoding="utf-8") or "WD2S" not in residual_path.read_text(
            encoding="utf-8"
        ):
            raise WD4Error("retained runtime lacks the additive width branch")
        return {
            "source": str(source.resolve()),
            "destination": str(destination.resolve()),
            "archive": retained_archive,
            "additive_magic": "WD2S",
            "inactive_paths_retained": ["WANS1", "SD1M", "SM3R"],
            "runtime_f26": file_record(f26_path),
            "runtime_parser": file_record(residual_path),
            "receiver": file_record(destination / "cpr1/wd2_receiver.py"),
            "public_entrypoint": file_record(destination / "inflate.py"),
            "resume_reused_byte_identical_runtime": True,
        }
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(Path(receiver.__file__).resolve(), destination / "cpr1/wd2_receiver.py")

    residual_path = destination / "runtime/residual_archive.py"
    residual = residual_path.read_text(encoding="utf-8")
    residual = _replace_once(
        residual,
        'tagged_semantic = semantic_body.startswith((b"SD1M", b"SM3R"))',
        'tagged_semantic = semantic_body.startswith((b"SD1M", b"SM3R", b"WD2S"))',
        "RX1 semantic dispatch",
    )
    residual_path.write_text(residual, encoding="utf-8")

    f26_path = destination / "runtime/f26_inflate.py"
    f26 = f26_path.read_text(encoding="utf-8")
    f26 = _replace_once(
        f26,
        'if not parts.semantic_blob.startswith((WANS1_MAGIC, b"SD1M", b"SM3R")):\n'
        '        raise InflationError("F26 requires WANS1, SD1M, or SM3R semantic weights")',
        'if not parts.semantic_blob.startswith((WANS1_MAGIC, b"SD1M", b"SM3R", b"WD2S")):\n'
        '        raise InflationError("F26 requires WANS1, SD1M, SM3R, or WD2S semantic weights")',
        "F26 semantic guard",
    )
    old_loader = '''    semantic = renderer.SemanticTokenRenderer(96)
    tagged_state = renderer.unpack_variant_semantic_or_none(
        parts.semantic_blob,
        semantic.state_dict(),
    )
    if tagged_state is None:
        records = decode_wans1(parts.semantic_blob)
        tagged_state = {
            record.schema.name: torch.from_numpy(
                np.ascontiguousarray(record.values, dtype=np.float32)
            )
            for record in records
        }
    semantic.load_state_dict(tagged_state, strict=True)
'''
    new_loader = '''    if parts.semantic_blob.startswith(b"WD2S"):
        receiver_path = renderer_dir / "wd2_receiver.py"
        receiver_spec = importlib.util.spec_from_file_location("_f26_wd4_receiver", receiver_path)
        if receiver_spec is None or receiver_spec.loader is None:
            raise InflationError("could not load the counted WD4 width receiver")
        width_receiver = importlib.util.module_from_spec(receiver_spec)
        sys.modules[receiver_spec.name] = width_receiver
        receiver_spec.loader.exec_module(width_receiver)
        semantic = width_receiver.unpack_student(parts.semantic_blob)
    else:
        semantic = renderer.SemanticTokenRenderer(96)
        tagged_state = renderer.unpack_variant_semantic_or_none(
            parts.semantic_blob,
            semantic.state_dict(),
        )
        if tagged_state is None:
            records = decode_wans1(parts.semantic_blob)
            tagged_state = {
                record.schema.name: torch.from_numpy(
                    np.ascontiguousarray(record.values, dtype=np.float32)
                )
                for record in records
            }
        semantic.load_state_dict(tagged_state, strict=True)
'''
    f26_path.write_text(
        _replace_once(f26, old_loader, new_loader, "F26 width model construction"),
        encoding="utf-8",
    )

    shutil.copy2(archive_path, destination / "archive.zip")
    public_path = destination / "inflate.py"
    public = public_path.read_text(encoding="utf-8")
    digest = sha256_file(destination / "archive.zip")
    size = (destination / "archive.zip").stat().st_size
    public, sha_count = re.subn(
        r'^ARCHIVE_SHA256 = "[0-9a-f]{64}"$',
        f'ARCHIVE_SHA256 = "{digest}"',
        public,
        count=1,
        flags=re.MULTILINE,
    )
    public, size_count = re.subn(
        r"^ARCHIVE_BYTES = [0-9_]+$",
        f"ARCHIVE_BYTES = {size:_}",
        public,
        count=1,
        flags=re.MULTILINE,
    )
    if (sha_count, size_count) != (1, 1):
        raise WD4Error("public archive binding points differ")
    public_path.write_text(public, encoding="utf-8")
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "archive": file_record(destination / "archive.zip"),
        "additive_magic": "WD2S",
        "inactive_paths_retained": ["WANS1", "SD1M", "SM3R"],
        "runtime_f26": file_record(f26_path),
        "runtime_parser": file_record(residual_path),
        "receiver": file_record(destination / "cpr1/wd2_receiver.py"),
        "public_entrypoint": file_record(public_path),
    }


def parse_with_runtime(runtime: Path, archive: Path, expected_packet: bytes) -> dict[str, Any]:
    prior = _clear_runtime_modules()
    sys.path.insert(0, str(runtime.resolve()))
    try:
        module = importlib.import_module("runtime.residual_archive")
        parts = module.read_residual_archive(archive)
        if parts.semantic_blob != expected_packet:
            raise WD4Error("patched receiver semantic parse-back differs")
    finally:
        sys.path.pop(0)
        for name in list(sys.modules):
            if name == "runtime" or name.startswith("runtime."):
                sys.modules.pop(name, None)
        sys.modules.update(prior)
    model = receiver.unpack_student(expected_packet)
    return {
        "semantic_packet_sha256": sha256_bytes(expected_packet),
        "semantic_packet_bytes": len(expected_packet),
        "parsed_spec": model.spec.as_dict(),
        "strict_state_load": True,
    }


def metal_probe() -> dict[str, Any]:
    torch_probe = {
        "torch_version": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }
    try:
        from tac.pr130_lift.mlx_semantic_renderer import mlx_device_probe

        mlx_probe = mlx_device_probe(device="gpu")
    except Exception as error:  # pragma: no cover - host-dependent import surface
        mlx_probe = {
            "status": "blocked",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {
        "torch_mps": torch_probe,
        "mlx_gpu": mlx_probe,
        "metal_available": torch_probe["mps_available"] or mlx_probe.get("status") == "available",
        "cpu_fallback_permitted": False,
    }


def stratified_pair_ids(count: int = 32, seed: int = SEED) -> list[int]:
    if not 1 <= count <= 600:
        raise WD4Error("stratified pair count must be in [1,600]")
    generator = np.random.default_rng(seed)
    edges = np.linspace(0, 600, count + 1, dtype=np.int64)
    return [int(generator.integers(edges[index], edges[index + 1])) for index in range(count)]


def require_local_lane_claim(job_id: str) -> dict[str, str]:
    if not job_id:
        raise WD4Error("DDM_WD4_CLAIM_JOB_ID is required for a Metal launch")
    rows = []
    for line in CLAIMS_LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        fields = [field.strip() for field in line.strip("|").split("|")]
        if len(fields) != 8 or fields[0] == "timestamp_utc":
            continue
        rows.append(dict(zip(
            ("timestamp", "agent", "lane", "platform", "job", "eta", "status", "notes"),
            fields,
            strict=True,
        )))
    for row in rows:  # Ledger is newest first; the first matching job is authoritative.
        if row["lane"] == LOCAL_LANE_ID and row["job"] == job_id:
            if row["platform"] not in {"local", "local_mps", "macos_mps_local"}:
                raise WD4Error("WD4 lane claim uses a nonlocal platform")
            if not row["status"].startswith(("active", "training", "launch")):
                raise WD4Error("latest WD4 lane claim is not active")
            return row
    raise WD4Error("matching active WD4 local lane claim was not found")


class DeploymentEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = OrderedDict(
            (name, value.detach().cpu().clone()) for name, value in model.state_dict().items()
        )

    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            source = value.detach().cpu()
            if source.is_floating_point():
                self.shadow[name].mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(source)

    @contextlib.contextmanager
    def apply(self, model: torch.nn.Module):
        current = OrderedDict(
            (name, value.detach().cpu().clone()) for name, value in model.state_dict().items()
        )
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(current, strict=True)


def _retain_training_candidate(
    root: Path,
    model: receiver.StudentSemanticRenderer,
    container: Mapping[str, Any],
) -> dict[str, Any]:
    packet = receiver.pack_student(model)
    semantic_stream, member, archive = build_archive(container, packet)
    repeat = deterministic_zip(member)
    if archive != repeat:
        raise WD4Error("training checkpoint archive repeat differs")
    return {
        "packet": atomic_bytes(root / "semantic.wd2s", packet),
        "semantic_stream": atomic_bytes(root / "semantic.brotli.ck2", semantic_stream),
        "member": atomic_bytes(root / "p", member),
        "archive": atomic_bytes(root / "archive.zip", archive),
        "archive_repeat": atomic_bytes(root / "archive.repeat.zip", repeat),
    }


def _load_tokens() -> torch.Tensor:
    require_file(MC36_CACHE, size=117_967_085, sha256=MC36_CACHE_SHA256)
    payload = torch.load(MC36_CACHE, map_location="cpu", weights_only=False)
    tokens = payload.get("seg")
    if not isinstance(tokens, torch.Tensor) or tuple(tokens.shape) != (600, 384, 512):
        raise WD4Error("MC36 token cache geometry differs")
    return tokens.to(torch.uint8)


def train_gate(
    output: Path,
    *,
    steps: int,
    checkpoint_every: int,
    learning_rate: float,
    resume_from: Path | None,
) -> dict[str, Any]:
    """One bounded, resumable Metal-only warm-survival window; never a scorer verdict."""

    assert_governed_admission("ddm_wd4_warm_lineage_width_train_gate")
    if not (1 <= steps <= 64 and 1 <= checkpoint_every <= steps and learning_rate > 0):
        raise WD4Error("bounded gate requires steps in [1,64] and a positive checkpoint cadence/LR")
    probe = metal_probe()
    atomic_json(output / "METAL_PROBE.json", probe)
    if not probe["torch_mps"]["mps_available"]:
        raise WD4Error("local PyTorch Metal is unavailable; CPU substitution is forbidden")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise WD4Error("set PYTORCH_ENABLE_MPS_FALLBACK=0 before the governed Metal launch")
    lane_claim = require_local_lane_claim(os.environ.get("DDM_WD4_CLAIM_JOB_ID", ""))
    require_file(FX5_RAW, size=3_662_409_600, sha256=FX5_RAW_SHA256)
    decode_receipt = json.loads(FX5_DECODE_RECEIPT.read_text(encoding="utf-8"))
    if not decode_receipt.get("raw_output", {}).get("identical"):
        raise WD4Error("fx5 retained raw output lacks decode identity")
    result = json.loads((output / "RESULT.json").read_text(encoding="utf-8"))
    gate = result.get("gate_candidate")
    if not gate or gate.get("candidate_id") != "salience_dense_d4_w64":
        raise WD4Error("sealed width-64 gate candidate is absent")
    packet_path = Path(gate["payloads"]["packet"]["path"])
    model = receiver.unpack_student(packet_path.read_bytes())
    device = torch.device("mps")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    ema = DeploymentEMA(model)
    pair_ids = stratified_pair_ids()
    start_step = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != "ddm_wd4_warm_gate_checkpoint.v1":
            raise WD4Error("resume checkpoint schema differs")
        if checkpoint.get("pair_ids") != pair_ids or checkpoint.get("seed") != SEED:
            raise WD4Error("resume selection binding differs")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        ema.shadow = checkpoint["ema_state"]
        start_step = int(checkpoint["step"])
    tokens = _load_tokens()
    raw = np.memmap(
        FX5_RAW,
        mode="r",
        dtype=np.uint8,
        shape=(1_200, receiver.CAMERA_H, receiver.CAMERA_W, 3),
    )
    container = source_container()
    root = output / "retained/warm_gate_training"

    def save_checkpoint(step: int, label: str, last_loss: float | None) -> dict[str, Any]:
        checkpoint_root = root / "checkpoints" / label
        with ema.apply(model):
            candidate = _retain_training_candidate(checkpoint_root / "candidate", model, container)
            ema_state = OrderedDict((name, value.clone()) for name, value in ema.shadow.items())
        payload = {
            "schema": "ddm_wd4_warm_gate_checkpoint.v1",
            "stage": "warm_survival",
            "label": label,
            "step": step,
            "total_steps": steps,
            "seed": SEED,
            "pair_ids": pair_ids,
            "learning_rate": learning_rate,
            "model_state": OrderedDict(
                (name, value.detach().cpu()) for name, value in model.state_dict().items()
            ),
            "optimizer_state": optimizer.state_dict(),
            "ema_state": ema_state,
            "last_loss": last_loss,
            "candidate": candidate,
            "source_archive_sha256": FX5_ARCHIVE_SHA256,
            "source_raw_sha256": FX5_RAW_SHA256,
            "token_cache_sha256": MC36_CACHE_SHA256,
            "lane_claim": lane_claim,
        }
        return atomic_torch(checkpoint_root / "checkpoint.pt", payload)

    checkpoints = []
    if start_step == 0:
        checkpoints.append(save_checkpoint(0, "stage_00_initial", None))
    last_loss = None
    for step in range(start_step, steps):
        pair_id = pair_ids[step % len(pair_ids)]
        token = tokens[pair_id : pair_id + 1].long().to(device)
        index = torch.tensor([pair_id], dtype=torch.long, device=device)
        # The retained teacher is exact receiver uint8.  Downsampling it supplies a
        # scorer-free survival target at the semantic renderer's native grid.
        teacher_camera = torch.from_numpy(np.asarray(raw[2 * pair_id + 1]).copy()).permute(2, 0, 1)[
            None
        ].float()
        teacher_native = torch.nn.functional.interpolate(
            teacher_camera,
            size=(receiver.EVAL_H, receiver.EVAL_W),
            mode="bilinear",
            align_corners=False,
        ).to(device)
        parameters = receiver.fake_quantize_state(model)
        student = torch.func.functional_call(model, parameters, (token, index))
        loss = ((student - teacher_native) / 255.0).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema.update(model)
        last_loss = float(loss.detach().cpu())
        completed = step + 1
        if completed % checkpoint_every == 0 and completed < steps:
            checkpoints.append(
                save_checkpoint(completed, f"stage_00_step_{completed:04d}", last_loss)
            )
    checkpoints.append(save_checkpoint(steps, "stage_01_warm_survival", last_loss))
    receipt = {
        "schema": "ddm_wd4_warm_gate_training.v1",
        "axis": "[macOS-Metal scorer-free bounded warm-survival training]",
        "score_claim": False,
        "steps": steps,
        "checkpoint_every": checkpoint_every,
        "learning_rate": learning_rate,
        "pair_ids": pair_ids,
        "pair_ids_sha256": sha256_bytes(np.asarray(pair_ids, dtype="<u2").tobytes()),
        "last_loss": last_loss,
        "checkpoints": checkpoints,
        "all_stage_and_periodic_checkpoints_retained": True,
        "all_candidate_payloads_retained": True,
        "scorer_invocations": 0,
        "lane_claim": lane_claim,
    }
    atomic_json(output / "TRAIN_GATE_RESULT.json", receipt)
    return receipt


def run(output: Path, minimum_free_bytes: int) -> dict[str, Any]:
    storage = storage_preflight(output, minimum_free_bytes)
    inputs = {
        "fx5_archive": require_file(
            FX5_ARCHIVE, size=FX5_ARCHIVE_BYTES, sha256=FX5_ARCHIVE_SHA256
        ),
        "fx5_runtime": str(FX5_RUNTIME.resolve()),
        "builder": file_record(Path(__file__).resolve()),
        "receiver_source": file_record(Path(receiver.__file__).resolve()),
    }
    state, parts, _renderer = load_fx5_state()
    if not parts.semantic_blob.startswith(b"SM3R"):
        raise WD4Error("exact fx5 semantic parent is no longer SM3R")
    lineage = lineage_receipt(state)
    container = source_container()
    source_root = output / "retained/source_fx5"
    source_payloads = {
        "archive": atomic_bytes(source_root / "archive.zip", container["archive"]),
        "member": atomic_bytes(source_root / "p", container["member"]),
        "hpac_stream": atomic_bytes(source_root / "hpac.stream", container["hpac"]),
        "semantic_stream": atomic_bytes(
            source_root / "semantic.brotli.ck2", container["semantic"]
        ),
        "semantic_decoded": atomic_bytes(
            source_root / "semantic.sm3r", bytes(parts.semantic_blob)
        ),
        "carrier_stream": atomic_bytes(source_root / "carrier.stream", container["carrier"]),
        "tail": atomic_bytes(source_root / "tail.bin", container["tail"]),
    }
    tensor_rows = tensor_manifest(state)
    tensor_manifest_record = atomic_json(
        source_root / "fx5_semantic_38_tensor_manifest.json",
        {
            "schema": "ddm_wd4_fx5_semantic_tensor_manifest.v1",
            "tensor_count": len(tensor_rows),
            "total_elements": sum(row["elements"] for row in tensor_rows),
            "tensors": tensor_rows,
        },
    )

    rankings = {
        "salience": salience_order(state),
        "prefix": list(range(96)),
    }
    ranking_record = atomic_json(
        output / "retained/channel_rankings.json",
        {
            "schema": "ddm_wd4_nested_channel_rankings.v1",
            "selection_law": {
                "salience": (
                    "normalized absolute contribution over every shared channel consumer/emitter; "
                    "rank inherited 8-channel GroupNorm groups and preserve within-group order"
                ),
                "prefix": "index-order control",
            },
            "rankings": rankings,
        },
    )
    rows = []
    packet_by_id: dict[str, bytes] = {}
    for selector in SELECTORS:
        for width in WIDTHS:
            candidate_id = f"{selector}_dense_d4_w{width}"
            selected = rankings[selector][:width]
            sliced = slice_dense_state(state, selected)
            spec = receiver.StudentSpec(candidate_id, "dense", width, 4)
            model = receiver.StudentSemanticRenderer(spec)
            if tuple(model.state_dict()) != tuple(sliced):
                raise WD4Error(f"student tensor order differs for {candidate_id}")
            model.load_state_dict(sliced, strict=True)
            packet = receiver.pack_student(model)
            parsed = receiver.unpack_student(packet)
            if tuple(parsed.state_dict()) != tuple(model.state_dict()):
                raise WD4Error(f"student parse schema differs for {candidate_id}")
            if receiver.pack_student(parsed) != packet:
                raise WD4Error(f"student packet parse/repack differs for {candidate_id}")
            semantic_stream, member, archive = build_archive(container, packet)
            repeat = deterministic_zip(member)
            if archive != repeat:
                raise WD4Error(f"ZIP determinism differs for {candidate_id}")
            root = output / "retained/candidates" / candidate_id
            payloads = {
                "packet": atomic_bytes(root / "semantic.wd2s", packet),
                "semantic_stream": atomic_bytes(root / "semantic.brotli.ck2", semantic_stream),
                "member": atomic_bytes(root / "p", member),
                "archive": atomic_bytes(root / "archive.zip", archive),
                "archive_repeat": atomic_bytes(root / "archive.repeat.zip", repeat),
            }
            saving = FX5_ARCHIVE_BYTES - len(archive)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "selector": selector,
                    "width": width,
                    "depth": 4,
                    "selected_channels": selected,
                    "selected_channels_sha256": sha256_bytes(bytes(selected)),
                    "packet_bytes": len(packet),
                    "semantic_stream_bytes": len(semantic_stream),
                    "archive_bytes": len(archive),
                    "complete_archive_saving_bytes": saving,
                    "rate_delta_score": -saving * 25 / RATE_DENOMINATOR,
                    "clears_12155_byte_ceiling": saving >= REQUIRED_COMPLETE_ARCHIVE_SAVING,
                    "payloads": payloads,
                    "all_38_source_tensors_consumed_by_slicing_law": True,
                }
            )
            packet_by_id[candidate_id] = packet
    rows.sort(key=lambda row: (-row["complete_archive_saving_bytes"], -row["width"], row["selector"]))
    clearing = [row for row in rows if row["clears_12155_byte_ceiling"]]
    # Preserve the most capacity among threshold-clearing rows. At equal width choose
    # the predeclared warm salience selector; the prefix row is a control, not a winner
    # selected opportunistically from a 14-byte entropy fluctuation.
    gate_row = max(
        clearing,
        key=lambda row: (
            row["width"],
            int(row["selector"] == "salience"),
            row["complete_archive_saving_bytes"],
        ),
        default=None,
    )
    gate_score_bar = None
    if gate_row is not None:
        rate_credit = -float(gate_row["rate_delta_score"])
        full_seg_win_projection = FX5_SCORE - rate_credit - 100.0 * FX5_D_SEG
        headroom = TARGET_SCORE - full_seg_win_projection
        gate_score_bar = {
            "baseline_score": FX5_SCORE,
            "target_score": TARGET_SCORE,
            "baseline_d_seg": FX5_D_SEG,
            "rate_credit_score": rate_credit,
            "full_seg_win_projected_score": full_seg_win_projection,
            "maximum_total_degradation_score": headroom,
            "early_checkpoint_1p5x_degradation_bar_score": 1.5 * headroom,
            "projection_label": (
                "first-order arithmetic only; a scorer result must measure realized Seg/Pose "
                "degradation and no transfer constant is claimed"
            ),
        }
    runtime_receipt = None
    parse_receipt = None
    if gate_row is not None:
        gate_id = gate_row["candidate_id"]
        gate_archive = Path(gate_row["payloads"]["archive"]["path"])
        runtime = output / "retained/runtime_gate_candidate_group8_salience"
        runtime_receipt = patch_runtime(FX5_RUNTIME, runtime, gate_archive)
        parse_receipt = parse_with_runtime(
            runtime, runtime / "archive.zip", packet_by_id[gate_id]
        )
    hardware = metal_probe()
    metal_record = atomic_json(output / "METAL_PROBE.json", hardware)
    verdict = (
        "CEILING-PASS_GATE-BLOCKED"
        if gate_row is not None and not hardware["metal_available"]
        else "CEILING-PASS_GATE-QUEUED"
        if gate_row is not None
        else "CEILING-DEAD"
    )
    result = {
        "schema": "ddm_wd4_warm_lineage_width_ceiling.v1",
        "axis": AXIS,
        "verdict": verdict,
        "score_claim": False,
        "scorer_invocations": 0,
        "training_launched": False,
        "modal_invocations": 0,
        "frontier_moved": False,
        "storage": storage,
        "inputs": inputs,
        "lineage": lineage,
        "source_payloads": source_payloads,
        "source_semantic": {
            "wire_format": "SM3R mode-6 after CK2+Brotli",
            "decoded_bytes": len(parts.semantic_blob),
            "decoded_sha256": sha256_bytes(parts.semantic_blob),
            "compressed_stream_bytes": len(container["semantic"]),
            "compressed_stream_sha256": sha256_bytes(container["semantic"]),
            "tensor_count": len(state),
            "tensor_manifest": tensor_manifest_record,
            "maximum_complete_archive_credit_bytes": len(container["semantic"]),
            "maximum_credit_scope": "zero-length semantic-stream limiting bound; not a viable candidate",
        },
        "threshold": {
            "required_complete_archive_saving_bytes": REQUIRED_COMPLETE_ARCHIVE_SAVING,
            "baseline_archive_bytes": FX5_ARCHIVE_BYTES,
            "maximum_allowed_archive_bytes": FX5_ARCHIVE_BYTES
            - REQUIRED_COMPLETE_ARCHIVE_SAVING,
            "selection_mode": "complete archive bytes, real Brotli-q11, deterministic ZIP",
        },
        "channel_rankings": ranking_record,
        "candidates": rows,
        "gate_candidate": gate_row,
        "gate_score_bar": gate_score_bar,
        "runtime_patch": runtime_receipt,
        "receiver_parse_back": parse_receipt,
        "metal_probe": metal_record,
        "metal": hardware,
        "gate_status": (
            "BLOCKED: this sandbox exposes neither PyTorch MPS nor MLX Metal, CPU fallback "
            "is forbidden, exact scorer ownership was not granted, and a fleet scorer lane is active"
            if gate_row is not None
            else "NOT-ADMISSIBLE: no serialized width crossed the byte ceiling"
        ),
        "all_materialized_payloads_retained": True,
    }
    atomic_json(output / "RESULT.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", nargs="?", choices=("price", "train-gate"), default="price")
    result.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    result.add_argument("--minimum-free-bytes", type=int, default=2_000_000_000)
    result.add_argument("--steps", type=int, default=32)
    result.add_argument("--checkpoint-every", type=int, default=8)
    result.add_argument("--learning-rate", type=float, default=2.0e-7)
    result.add_argument("--resume-from", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.action == "train-gate":
        try:
            result = train_gate(
                args.output,
                steps=args.steps,
                checkpoint_every=args.checkpoint_every,
                learning_rate=args.learning_rate,
                resume_from=args.resume_from,
            )
        except WD4Error as error:
            blocked = {
                "schema": "ddm_wd4_warm_gate_blocked.v1",
                "disposition": "BLOCKED_PRELAUNCH",
                "error": str(error),
                "cpu_fallback_used": False,
                "training_steps_executed": 0,
                "scorer_invocations": 0,
                "resume_command": (
                    "DDM_WD4_CLAIM_JOB_ID=wd4_warm_gate_r1 "
                    "PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python tools/safe_run.py "
                    "--rss-mb 12288 --projected-gib 12 --timeout 1800 "
                    "--label ddm_wd4_warm_gate --status-receipt "
                    "/Volumes/APDataStore/pact/ddm_wd4_warm_lineage_width/"
                    "GOVERNED_TRAIN_STATUS.json -- .venv/bin/python "
                    "experiments/ddm_wd4_warm_lineage_width.py train-gate"
                ),
            }
            atomic_json(args.output / "TRAIN_GATE_BLOCKED.json", blocked)
            print(json.dumps(blocked, sort_keys=True))
            return 3
        print(json.dumps({"result": str((args.output / "TRAIN_GATE_RESULT.json").resolve())}))
        return 0
    result = run(args.output, args.minimum_free_bytes)
    print(json.dumps({
        "verdict": result["verdict"],
        "gate_candidate": None if result["gate_candidate"] is None else result["gate_candidate"]["candidate_id"],
        "result": str((args.output / "RESULT.json").resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
