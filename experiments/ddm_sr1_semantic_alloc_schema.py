#!/usr/bin/env python3
"""Close the scorer-free PR130 semantic-allocation receiver proof.

The counted ``SD1M`` parser landed with DDM-CX2.  This harness closes the
remaining DDM-SR1 proof obligations on the exact retained PR130 and selected
SD1 archives:

* retain a real all-q4 allocation-record archive and deterministic repeat;
* compare legacy, extended-no-record, and extended-all-q4 decoder tensors;
* compare all 38 selected mixed tensors with the independent SD1 parser; and
* price the counted record with complete deterministic archives.

No scorer, evaluator, trainer, network service, or Modal surface is imported or
invoked.  Every materialized payload is retained under ``--out-dir``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SSD_ROOT = Path("/Volumes/VertigoDataTier/pact")
DEFAULT_BASE_ARCHIVE = SSD_ROOT / "ddm_pr130_reproduce_20260809/reproduction/archive.zip"
DEFAULT_SELECTED_ARCHIVE = SSD_ROOT / "ddm_sd1_semantic_20260809/cpu_screen/archives/selected_mixed_n600.zip"
DEFAULT_EXTENDED_RUNTIME = REPO / "src/tac/pr130_runtime/dv1_cpu_runtime"
DEFAULT_LEGACY_RUNTIME = REPO / "src/tac/pr130_runtime/fx1_runtime_tree"
DEFAULT_TOKEN_CHECKPOINT = SSD_ROOT / "ddm_ai1_20260809/decode/a/checkpoint/tokens.npz"
DEFAULT_RANGE_DECODE_RECEIPT = SSD_ROOT / "ddm_dt1_20260809/range_e2e/range_decode_result.json"
SD1_SOURCE = REPO / "experiments/ddm_sd1_semantic_rd_curve.py"
EXPECTED_BASE_BYTES = 191_052
EXPECTED_BASE_SHA256 = "0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd"
EXPECTED_SELECTED_BYTES = 190_204
EXPECTED_SELECTED_SHA256 = "010a8a5273ae87595191ffc03447fa36e61978ae9f827c2def46dea7075dfa67"
EXPECTED_TENSORS = 38
EXPECTED_QUANTIZED_TENSORS = 16
EXPECTED_TOKEN_SHA256 = "c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece"
EXPECTED_BASE_MODELS_RAW_SHA256 = "62dd72dfa0858a25ca32bdee1e536627a17883b6fc7efd7cd5b2de7b13b84517"
EXPECTED_RAW_BYTES = 3_662_409_600
ORIGINAL_BYTES = 37_545_489
SELECTED_Q3_NAMES = {
    "frame_embed.weight",
    "blocks.1.film.weight",
    "blocks.2.film.weight",
    "blocks.3.film.weight",
}
_RUNTIME_IMPORTS = (
    "carrier_codec",
    "hpac_integer",
    "hpac_integer_sparse",
    "integer_model_io",
    "receiver",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
    )


def persist_exact(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"resume artifact differs: {path}")
    atomic_write_bytes(path, payload)


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_runtime(name: str, runtime_dir: Path) -> ModuleType:
    """Load one self-contained receiver tree without sibling-module leakage."""

    saved = {module: sys.modules.get(module) for module in _RUNTIME_IMPORTS}
    for module in _RUNTIME_IMPORTS:
        sys.modules.pop(module, None)
    sys.path.insert(0, str(runtime_dir))
    try:
        return load_module(name, runtime_dir / "inflate.py")
    finally:
        sys.path.remove(str(runtime_dir))
        for module in _RUNTIME_IMPORTS:
            sys.modules.pop(module, None)
            if saved[module] is not None:
                sys.modules[module] = saved[module]


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def state_digest(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    combined = hashlib.sha256()
    tensors: list[dict[str, Any]] = []
    for name, value in state.items():
        tensor = value.detach().cpu().contiguous()
        payload_view = memoryview(tensor.numpy()).cast("B")
        shape = list(tensor.shape)
        dtype = str(tensor.dtype)
        record = {
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "bytes": len(payload_view),
            "sha256": hashlib.sha256(payload_view).hexdigest(),
        }
        tensors.append(record)
        combined.update(name.encode())
        combined.update(b"\0")
        combined.update(dtype.encode())
        combined.update(b"\0")
        combined.update(json.dumps(shape, separators=(",", ":")).encode())
        combined.update(b"\0")
        combined.update(payload_view)
    return {
        "tensor_count": len(tensors),
        "combined_sha256": combined.hexdigest(),
        "tensors": tensors,
    }


def state_wire(state: Mapping[str, torch.Tensor]) -> bytes:
    """Serialize decoded tensors deterministically for retained byte proof."""

    output = bytearray(b"SR1STATE\x01")
    output.extend(struct.pack("<I", len(state)))
    for name, value in state.items():
        tensor = value.detach().cpu().contiguous()
        name_bytes = name.encode()
        dtype_bytes = str(tensor.dtype).encode()
        shape = tuple(tensor.shape)
        payload = tensor.numpy().tobytes(order="C")
        output.extend(struct.pack("<H", len(name_bytes)))
        output.extend(name_bytes)
        output.extend(struct.pack("<B", len(dtype_bytes)))
        output.extend(dtype_bytes)
        output.extend(struct.pack("<B", len(shape)))
        output.extend(struct.pack(f"<{len(shape)}q", *shape))
        output.extend(struct.pack("<Q", tensor.numel() * tensor.element_size()))
        output.extend(payload)
    return bytes(output)


def tensor_pair_wire(basis: torch.Tensor, coeff: torch.Tensor) -> bytes:
    return state_wire({"basis": basis, "coeff": coeff})


def exact_state_comparison(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if list(reference) != list(candidate):
        raise ValueError("state keys or ordering differ")
    rows = []
    for name in reference:
        equal = torch.equal(reference[name], candidate[name])
        rows.append({"name": name, "exact": equal})
        if not equal:
            raise ValueError(f"tensor parity failed: {name}")
    reference_digest = state_digest(reference)
    candidate_digest = state_digest(candidate)
    if reference_digest != candidate_digest:
        raise ValueError("state digest differs despite elementwise equality")
    return {
        "denominator": len(rows),
        "exact": sum(int(row["exact"]) for row in rows),
        "all_exact": True,
        "combined_sha256": reference_digest["combined_sha256"],
        "tensors": reference_digest["tensors"],
    }


def semantic_pose_raw(base: Any, semantic: bytes) -> bytes:
    carrier = base.model_suffix[: base.carrier_bytes]
    if len(carrier) != base.carrier_bytes:
        raise ValueError("base carrier section is truncated")
    return struct.pack("<II", len(semantic), len(carrier)) + semantic + carrier


def require_ssd_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(SSD_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"SR1 evidence must live under {SSD_ROOT}: {resolved}") from error
    return resolved


def validate_source(path: Path, expected_bytes: int, expected_sha256: str) -> bytes:
    payload = path.read_bytes()
    if len(payload) != expected_bytes or sha256_bytes(payload) != expected_sha256:
        raise ValueError(f"source custody differs: {path}")
    return payload


def receiver_extract(runtime: ModuleType, archive_bytes: bytes) -> dict[str, bytes]:
    """Route an exact archive through the public outer receiver."""

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        if archive.namelist() != ["p"]:
            raise ValueError("receiver proof archive must contain exactly member p")
        payload = archive.read("p")
    parts = runtime.split_payload(payload)
    decoded = runtime.decode_models(parts.models, model_codec=parts.model_codec)
    return {
        "payload": payload,
        "encoded_models": parts.models,
        "tokens": parts.tokens,
        "models_raw": decoded.raw,
        "model_codec": decoded.codec.encode(),
        "token_codec": parts.token_codec.encode(),
    }


def split_model_bundle(models_raw: bytes) -> dict[str, bytes]:
    if len(models_raw) < 8:
        raise ValueError("model bundle is truncated")
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", models_raw)
    semantic_end = 8 + semantic_bytes
    carrier_end = semantic_end + carrier_bytes
    if carrier_end >= len(models_raw):
        raise ValueError("model bundle lacks carrier or HPAC bytes")
    return {
        "semantic_pose": models_raw[:carrier_end],
        "semantic": models_raw[8:semantic_end],
        "carrier": models_raw[semantic_end:carrier_end],
        "hpac": models_raw[carrier_end:],
    }


def files_byte_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(8 << 20)
            right_chunk = right_handle.read(8 << 20)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def mapping_is_subset(subset: Mapping[str, Any], superset: Mapping[str, Any]) -> bool:
    for name, value in subset.items():
        if name not in superset:
            return False
        candidate = superset[name]
        if isinstance(value, Mapping) and isinstance(candidate, Mapping):
            if not mapping_is_subset(value, candidate):
                return False
        elif candidate != value:
            return False
    return True


def load_retained_tokens(
    checkpoint_path: Path,
    range_receipt_path: Path,
    retained: Path,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, dict[str, Any]]]:
    receipt_path = checkpoint_path.with_name("tokens_receipt.json")
    source_receipt = json.loads(receipt_path.read_text())
    checkpoint_record = artifact_record(checkpoint_path)
    if checkpoint_record != source_receipt["cache"]:
        raise ValueError("token checkpoint custody differs from its receipt")
    if (
        not source_receipt["complete"]
        or not source_receipt["finish_token_decode_returned"]
        or not source_receipt["ans_final_state_empty"]
        or source_receipt["decoded_token_sha256"] != EXPECTED_TOKEN_SHA256
        or source_receipt["models_raw_sha256"] != EXPECTED_BASE_MODELS_RAW_SHA256
    ):
        raise ValueError("token checkpoint lacks the exact completed base binding")
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        tokens_array = np.ascontiguousarray(checkpoint["tokens"])
        recorded_sha256 = str(checkpoint["token_sha256"].item())
        finish_returned = bool(checkpoint["finish_token_decode_returned"].item())
        ans_empty = bool(checkpoint["ans_final_state_empty"].item())
    if tokens_array.dtype != np.uint8 or tokens_array.shape != (600, 384, 512):
        raise ValueError("token checkpoint geometry or dtype differs")
    tokens_payload = tokens_array.tobytes(order="C")
    if (
        sha256_bytes(tokens_payload) != EXPECTED_TOKEN_SHA256
        or recorded_sha256 != EXPECTED_TOKEN_SHA256
        or not finish_returned
        or not ans_empty
    ):
        raise ValueError("decoded token bytes or finish proof differ")

    range_receipt = json.loads(range_receipt_path.read_text())
    if (
        not range_receipt["complete"]
        or range_receipt["token_codec"] != "range"
        or not range_receipt["all_tokens_reconstructed"]
        or not range_receipt["exact_target_equality"]
        or range_receipt["frames"] != 600
        or range_receipt["tokens"] != 117_964_800
        or range_receipt["decoded_sha256"] != EXPECTED_TOKEN_SHA256
    ):
        raise ValueError("retained Range decode receipt differs")

    retained_checkpoint = retained / "tokens.source_checkpoint.npz"
    retained_receipt = retained / "tokens.source_receipt.json"
    retained_range_receipt = retained / "tokens.range_decode_receipt.json"
    retained_raw = retained / "tokens.uint8.bin"
    persist_exact(retained_checkpoint, checkpoint_path.read_bytes())
    persist_exact(retained_receipt, receipt_path.read_bytes())
    persist_exact(retained_range_receipt, range_receipt_path.read_bytes())
    persist_exact(retained_raw, tokens_payload)
    records = {
        "token_checkpoint": artifact_record(retained_checkpoint),
        "token_checkpoint_receipt": artifact_record(retained_receipt),
        "range_decode_receipt": artifact_record(retained_range_receipt),
        "decoded_tokens": artifact_record(retained_raw),
    }
    token_proof = {
        "ans_checkpoint_receipt": source_receipt,
        "range_decode_receipt": range_receipt,
        "decoded_bytes_match_range_receipt": True,
    }
    return torch.from_numpy(tokens_array.copy()), token_proof, records


def render_or_resume(
    *,
    label: str,
    renderer: Callable[..., None],
    semantic: torch.nn.Module,
    basis: torch.Tensor,
    coeff: torch.Tensor,
    tokens: torch.Tensor,
    destination: Path,
    resume_from: Path,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    stage_receipt = resume_from / "stages" / f"04_raw_{label}.json"
    if stage_receipt.is_file():
        prior = json.loads(stage_receipt.read_text())
        artifact = prior.get("artifact", {})
        prior_binding = prior.get("binding", {})
        compatible_binding = mapping_is_subset(prior_binding, binding)
        if (
            prior.get("complete")
            and compatible_binding
            and artifact.get("path") == str(destination.resolve())
            and destination.is_file()
            and artifact_record(destination) == artifact
        ):
            if prior_binding != binding:
                prior["binding"] = dict(binding)
                prior["binding_enriched_from_completed_stage"] = True
                prior["binding_enriched_at_utc"] = utc_now()
                atomic_write_json(stage_receipt, prior)
            return prior
        raise ValueError(f"completed raw stage custody differs: {label}")
    if destination.exists():
        raise ValueError(f"raw output exists without a completion receipt: {destination}")

    partial = destination.with_name(f".{destination.name}.partial")
    launch = {
        "schema": "ddm_sr1_raw_render_launch.v1",
        "complete": False,
        "written_at_utc": utc_now(),
        "score_claim": False,
        "label": label,
        "binding": dict(binding),
        "partial_path": str(partial.resolve()),
        "partial_is_certified_rebuildable_scratch": True,
        "resume_policy": "restart this bounded render stage from retained inputs",
    }
    atomic_write_json(resume_from / "stages" / f"04_raw_{label}_launch.json", launch)
    started = time.monotonic()
    renderer(semantic, basis, coeff, tokens, partial, torch.device("cpu"))
    if partial.stat().st_size != EXPECTED_RAW_BYTES:
        raise ValueError(f"rendered RAW length differs: {label}")
    os.replace(partial, destination)
    result = {
        "schema": "ddm_sr1_raw_render_stage.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU scorer-free receiver render]",
        "score_claim": False,
        "label": label,
        "binding": dict(binding),
        "elapsed_seconds": time.monotonic() - started,
        "artifact": artifact_record(destination),
    }
    atomic_write_json(stage_receipt, result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = require_ssd_path(args.out_dir)
    resume_from = require_ssd_path(args.resume_from)
    if resume_from != out_dir / "resume":
        raise ValueError("--resume-from must equal <out-dir>/resume")
    out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out_dir).free < args.minimum_free_bytes:
        raise RuntimeError("storage preflight refused: insufficient SSD free space")

    base_path = args.base_archive.resolve()
    selected_path = args.selected_archive.resolve()
    extended_runtime_dir = args.extended_runtime.resolve()
    legacy_runtime_dir = args.legacy_runtime.resolve()
    base_bytes = validate_source(base_path, EXPECTED_BASE_BYTES, EXPECTED_BASE_SHA256)
    selected_bytes = validate_source(selected_path, EXPECTED_SELECTED_BYTES, EXPECTED_SELECTED_SHA256)
    sd1 = load_module("ddm_sr1_sd1_reference", SD1_SOURCE)
    extended = load_runtime("ddm_sr1_extended_runtime", extended_runtime_dir)
    legacy = load_runtime("ddm_sr1_legacy_runtime", legacy_runtime_dir)
    base = sd1.read_base_archive(base_path)
    selected_semantic = sd1.semantic_blob_from_archive(selected_bytes, base)
    template = extended.SemanticTokenRenderer(96).state_dict()
    quantized_names = [name for name, value in template.items() if value.ndim >= 2]
    if len(template) != EXPECTED_TENSORS:
        raise ValueError("receiver template tensor census differs")
    if len(quantized_names) != EXPECTED_QUANTIZED_TENSORS:
        raise ValueError("receiver quantized-tensor census differs")

    header = (
        extended.SEMANTIC_MIXED_MAGIC
        + bytes([extended.SEMANTIC_MIXED_VERSION, len(quantized_names)])
        + bytes([0x44]) * ((len(quantized_names) + 1) // 2)
    )
    if len(header) != 14:
        raise ValueError("counted all-q4 record is not the measured 14 bytes")
    if not selected_semantic.startswith(extended.SEMANTIC_MIXED_MAGIC):
        raise ValueError("selected archive lacks the counted SD1M record")
    selected_header_bytes = 6 + (int(selected_semantic[5]) + 1) // 2
    if selected_header_bytes != len(header):
        raise ValueError("selected allocation header size differs")
    all_q4_semantic = header + base.semantic_blob
    selected_headerless = selected_semantic[selected_header_bytes:]
    all_q4_archive = sd1.rebuild_archive(base, all_q4_semantic)
    all_q4_repeat = sd1.rebuild_archive(base, all_q4_semantic)
    selected_rebuilt = sd1.rebuild_archive(base, selected_semantic)
    headerless_archive = sd1.rebuild_archive(base, selected_headerless)
    headerless_repeat = sd1.rebuild_archive(base, selected_headerless)
    if all_q4_archive != all_q4_repeat:
        raise ValueError("all-q4 archive repeat differs")
    if headerless_archive != headerless_repeat:
        raise ValueError("headerless comparator repeat differs")
    if selected_rebuilt != selected_bytes:
        raise ValueError("selected archive deterministic rebuild differs")

    outer = {
        "base_no_record": receiver_extract(extended, base_bytes),
        "all_q4_record": receiver_extract(extended, all_q4_archive),
        "selected_mixed": receiver_extract(extended, selected_bytes),
        "selected_headerless_counterfactual": receiver_extract(extended, headerless_archive),
    }
    outer_sections = {name: split_model_bundle(sections["models_raw"]) for name, sections in outer.items()}
    if outer["base_no_record"]["models_raw"] != base.model_raw:
        raise ValueError("public receiver changed the base model bundle")
    if outer["base_no_record"]["tokens"] != base.tokens:
        raise ValueError("public receiver changed the base token section")
    for name in ("all_q4_record", "selected_mixed", "selected_headerless_counterfactual"):
        if outer[name]["tokens"] != base.tokens:
            raise ValueError(f"public receiver changed the {name} token section")
    expected_semantics = {
        "base_no_record": base.semantic_blob,
        "all_q4_record": all_q4_semantic,
        "selected_mixed": selected_semantic,
        "selected_headerless_counterfactual": selected_headerless,
    }
    for name, semantic in expected_semantics.items():
        if outer_sections[name]["semantic"] != semantic:
            raise ValueError(f"public receiver changed the {name} semantic section")
        if outer_sections[name]["carrier"] != outer_sections["base_no_record"]["carrier"]:
            raise ValueError(f"public receiver changed the {name} carrier section")
        if outer_sections[name]["hpac"] != outer_sections["base_no_record"]["hpac"]:
            raise ValueError(f"public receiver changed the {name} HPAC section")

    retained = out_dir / "retained"
    payloads = {
        "base_no_record_archive": (retained / "base_no_record.zip", base_bytes),
        "base_no_record_semantic": (
            retained / "base_no_record.semantic.bin",
            base.semantic_blob,
        ),
        "base_no_record_semantic_pose": (
            retained / "base_no_record.semantic_pose.bin",
            semantic_pose_raw(base, base.semantic_blob),
        ),
        "allocation_record": (retained / "allocation_record.sd1m", header),
        "all_q4_record_archive": (
            retained / "all_q4_record.zip",
            all_q4_archive,
        ),
        "all_q4_record_archive_repeat": (
            retained / "all_q4_record.repeat.zip",
            all_q4_repeat,
        ),
        "all_q4_record_semantic": (
            retained / "all_q4_record.semantic.bin",
            all_q4_semantic,
        ),
        "all_q4_record_semantic_pose": (
            retained / "all_q4_record.semantic_pose.bin",
            semantic_pose_raw(base, all_q4_semantic),
        ),
        "selected_mixed_archive": (
            retained / "selected_mixed.zip",
            selected_bytes,
        ),
        "selected_mixed_archive_rebuilt": (
            retained / "selected_mixed.rebuilt.zip",
            selected_rebuilt,
        ),
        "selected_mixed_semantic": (
            retained / "selected_mixed.semantic.bin",
            selected_semantic,
        ),
        "selected_mixed_semantic_pose": (
            retained / "selected_mixed.semantic_pose.bin",
            semantic_pose_raw(base, selected_semantic),
        ),
        "selected_headerless_counterfactual_archive": (
            retained / "selected_headerless_counterfactual.zip",
            headerless_archive,
        ),
        "selected_headerless_counterfactual_archive_repeat": (
            retained / "selected_headerless_counterfactual.repeat.zip",
            headerless_repeat,
        ),
        "selected_headerless_counterfactual_semantic": (
            retained / "selected_headerless_counterfactual.semantic.bin",
            selected_headerless,
        ),
    }
    for candidate_name, sections in outer.items():
        for section_name in ("payload", "encoded_models", "tokens", "models_raw"):
            payloads[f"{candidate_name}_{section_name}"] = (
                retained / f"{candidate_name}.{section_name}.bin",
                sections[section_name],
            )
        payloads[f"{candidate_name}_hpac"] = (
            retained / f"{candidate_name}.hpac.bin",
            outer_sections[candidate_name]["hpac"],
        )
    for path, payload in payloads.values():
        persist_exact(path, payload)
    artifacts = {name: artifact_record(path) for name, (path, _) in payloads.items()}
    stage_one = {
        "schema": "ddm_sr1_retained_payloads.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "score_claim": False,
        "artifacts": artifacts,
    }
    atomic_write_json(resume_from / "stages/01_retained_payloads.json", stage_one)

    legacy_template = legacy.SemanticTokenRenderer(96).state_dict()
    legacy_reference = legacy.unpack_semantic(base.semantic_blob, legacy_template)
    extended_no_record = extended.unpack_semantic(base.semantic_blob, template)
    extended_all_q4 = extended.unpack_semantic(all_q4_semantic, template)
    selected_reference, selected_allocation, selected_format = sd1.unpack_semantic_state(selected_semantic, template)
    extended_selected = extended.unpack_semantic(selected_semantic, template)
    no_record_allocation, _, no_record_format = extended.semantic_allocation(base.semantic_blob, template)
    all_q4_allocation, _, all_q4_format = extended.semantic_allocation(all_q4_semantic, template)
    parsed_selected_allocation, _, parsed_selected_format = extended.semantic_allocation(selected_semantic, template)
    if no_record_format != "legacy_int4" or set(no_record_allocation.values()) != {4}:
        raise ValueError("no-record fallback is not uniform q4")
    if all_q4_format != "sd1_mixed_v1" or set(all_q4_allocation.values()) != {4}:
        raise ValueError("all-q4 allocation record did not parse as uniform q4")
    if selected_format != "sd1_mixed_v1" or parsed_selected_format != "sd1_mixed_v1":
        raise ValueError("selected allocation format differs")
    if dict(selected_allocation) != parsed_selected_allocation:
        raise ValueError("selected allocation map differs")
    if {name for name, bits in parsed_selected_allocation.items() if bits == 3} != SELECTED_Q3_NAMES:
        raise ValueError("selected q3 tensor set differs")

    legacy_full, legacy_basis, legacy_coeff = legacy.unpack_semantic_pose(
        outer_sections["base_no_record"]["semantic_pose"]
    )
    no_record_full, no_record_basis, no_record_coeff = extended.unpack_semantic_pose(
        outer_sections["base_no_record"]["semantic_pose"]
    )
    all_q4_full, all_q4_basis, all_q4_coeff = extended.unpack_semantic_pose(
        outer_sections["all_q4_record"]["semantic_pose"]
    )
    selected_full, selected_basis, selected_coeff = extended.unpack_semantic_pose(
        outer_sections["selected_mixed"]["semantic_pose"]
    )
    decoded_payloads = {
        "legacy_parser_state": state_wire(legacy_reference),
        "extended_no_record_parser_state": state_wire(extended_no_record),
        "extended_all_q4_parser_state": state_wire(extended_all_q4),
        "independent_selected_parser_state": state_wire(selected_reference),
        "extended_selected_parser_state": state_wire(extended_selected),
        "legacy_full_loader_state": state_wire(legacy_full.state_dict()),
        "extended_no_record_full_loader_state": state_wire(no_record_full.state_dict()),
        "extended_all_q4_full_loader_state": state_wire(all_q4_full.state_dict()),
        "extended_selected_full_loader_state": state_wire(selected_full.state_dict()),
        "legacy_carrier": tensor_pair_wire(legacy_basis, legacy_coeff),
        "extended_no_record_carrier": tensor_pair_wire(no_record_basis, no_record_coeff),
        "extended_all_q4_carrier": tensor_pair_wire(all_q4_basis, all_q4_coeff),
        "extended_selected_carrier": tensor_pair_wire(selected_basis, selected_coeff),
    }
    for name, payload in decoded_payloads.items():
        path = retained / "decoded" / f"{name}.sr1state"
        persist_exact(path, payload)
        artifacts[f"decoded_{name}"] = artifact_record(path)
    parity = {
        "schema": "ddm_sr1_semantic_allocation_parity.v1",
        "complete": True,
        "axis": "[scorer-free receiver structural proof; exact archive bytes]",
        "score_claim": False,
        "legacy_no_record_vs_today_shipped_parser": exact_state_comparison(legacy_reference, extended_no_record),
        "legacy_no_record_full_loader_vs_today_shipped_full_loader": (
            exact_state_comparison(legacy_full.state_dict(), no_record_full.state_dict())
        ),
        "all_q4_record_vs_today_shipped_parser": exact_state_comparison(legacy_reference, extended_all_q4),
        "all_q4_record_full_loader_vs_today_shipped_full_loader": (
            exact_state_comparison(legacy_full.state_dict(), all_q4_full.state_dict())
        ),
        "selected_mixed_vs_independent_sd1_parser": exact_state_comparison(selected_reference, extended_selected),
        "selected_mixed_full_loader_vs_independent_sd1_parser": (
            exact_state_comparison(selected_reference, selected_full.state_dict())
        ),
        "carrier_basis_exact_across_records": bool(
            torch.equal(legacy_basis, no_record_basis)
            and torch.equal(legacy_basis, all_q4_basis)
            and torch.equal(legacy_basis, selected_basis)
        ),
        "carrier_coeff_exact_across_records": bool(
            torch.equal(legacy_coeff, no_record_coeff)
            and torch.equal(legacy_coeff, all_q4_coeff)
            and torch.equal(legacy_coeff, selected_coeff)
        ),
        "allocation": {
            "no_record": no_record_allocation,
            "all_q4_record": all_q4_allocation,
            "selected_mixed": parsed_selected_allocation,
        },
        "public_outer_receiver": {
            name: {
                "model_codec": sections["model_codec"].decode(),
                "token_codec": sections["token_codec"].decode(),
                "models_raw_bytes": len(sections["models_raw"]),
                "models_raw_sha256": sha256_bytes(sections["models_raw"]),
                "tokens_bytes": len(sections["tokens"]),
                "tokens_sha256": sha256_bytes(sections["tokens"]),
            }
            for name, sections in outer.items()
        },
        "decoded_state_wire_byte_identity": {
            "no_record_matches_legacy": (
                decoded_payloads["legacy_parser_state"]
                == decoded_payloads["extended_no_record_parser_state"]
                == decoded_payloads["legacy_full_loader_state"]
                == decoded_payloads["extended_no_record_full_loader_state"]
            ),
            "all_q4_record_matches_legacy": (
                decoded_payloads["legacy_parser_state"]
                == decoded_payloads["extended_all_q4_parser_state"]
                == decoded_payloads["legacy_full_loader_state"]
                == decoded_payloads["extended_all_q4_full_loader_state"]
            ),
            "selected_matches_independent_parser": (
                decoded_payloads["independent_selected_parser_state"]
                == decoded_payloads["extended_selected_parser_state"]
                == decoded_payloads["extended_selected_full_loader_state"]
            ),
            "carrier_no_record_matches_legacy": (
                decoded_payloads["legacy_carrier"] == decoded_payloads["extended_no_record_carrier"]
            ),
            "carrier_all_q4_matches_legacy": (
                decoded_payloads["legacy_carrier"] == decoded_payloads["extended_all_q4_carrier"]
            ),
        },
    }
    if not parity["carrier_basis_exact_across_records"]:
        raise ValueError("carrier basis changed across semantic records")
    if not parity["carrier_coeff_exact_across_records"]:
        raise ValueError("carrier coefficients changed across semantic records")
    if not all(parity["decoded_state_wire_byte_identity"].values()):
        raise ValueError("retained decoded-state wire bytes differ")
    atomic_write_json(resume_from / "stages/02_tensor_parity.json", parity)

    raw_proof: dict[str, Any] | None = None
    if args.render_byte_proof:
        if shutil.disk_usage(out_dir).free < 8_000_000_000:
            raise RuntimeError("RAW identity preflight refused: less than 8 GB free")
        tokens, token_proof, token_artifacts = load_retained_tokens(
            args.token_checkpoint.resolve(),
            args.range_decode_receipt.resolve(),
            retained,
        )
        artifacts.update(token_artifacts)
        raw_dir = retained / "raw_identity"
        legacy_raw = raw_dir / "legacy_no_record.raw"
        all_q4_raw = raw_dir / "extended_all_q4_record.raw"
        common_binding = {
            "axis": "[macOS-CPU scorer-free receiver render]",
            "decoded_token_sha256": EXPECTED_TOKEN_SHA256,
            "token_source_checkpoint_sha256": token_artifacts["token_checkpoint"]["sha256"],
            "token_finish_proof": {
                "finish_token_decode_returned": token_proof["ans_checkpoint_receipt"]["finish_token_decode_returned"],
                "ans_final_state_empty": token_proof["ans_checkpoint_receipt"]["ans_final_state_empty"],
                "source_models_raw_sha256": token_proof["ans_checkpoint_receipt"]["models_raw_sha256"],
                "range_decode_receipt_sha256": token_artifacts["range_decode_receipt"]["sha256"],
                "range_decoded_token_sha256": token_proof["range_decode_receipt"]["decoded_sha256"],
                "decoded_bytes_match_range_receipt": token_proof["decoded_bytes_match_range_receipt"],
            },
            "base_hpac_sha256": sha256_bytes(outer_sections["base_no_record"]["hpac"]),
            "all_q4_hpac_sha256": sha256_bytes(outer_sections["all_q4_record"]["hpac"]),
            "base_range_token_payload_sha256": sha256_bytes(outer["base_no_record"]["tokens"]),
            "all_q4_range_token_payload_sha256": sha256_bytes(outer["all_q4_record"]["tokens"]),
            "token_reuse_basis": (
                "retained Range decode output equals retained token bytes; "
                "all-q4 archive HPAC and Range token payload are byte-identical "
                "to the base inputs consumed by decode_tokens"
            ),
            "device": "cpu",
            "score_claim": False,
        }
        legacy_stage = render_or_resume(
            label="legacy_no_record",
            renderer=legacy.render_video,
            semantic=legacy_full,
            basis=legacy_basis,
            coeff=legacy_coeff,
            tokens=tokens,
            destination=legacy_raw,
            resume_from=resume_from,
            binding={
                **common_binding,
                "archive_sha256": EXPECTED_BASE_SHA256,
                "inflate_sha256": sha256_file(legacy_runtime_dir / "inflate.py"),
                "state_wire_sha256": artifacts["decoded_legacy_full_loader_state"]["sha256"],
                "carrier_wire_sha256": artifacts["decoded_legacy_carrier"]["sha256"],
            },
        )
        all_q4_stage = render_or_resume(
            label="extended_all_q4_record",
            renderer=extended.render_video,
            semantic=all_q4_full,
            basis=all_q4_basis,
            coeff=all_q4_coeff,
            tokens=tokens,
            destination=all_q4_raw,
            resume_from=resume_from,
            binding={
                **common_binding,
                "archive_sha256": sha256_bytes(all_q4_archive),
                "inflate_sha256": sha256_file(extended_runtime_dir / "inflate.py"),
                "state_wire_sha256": artifacts["decoded_extended_all_q4_full_loader_state"]["sha256"],
                "carrier_wire_sha256": artifacts["decoded_extended_all_q4_carrier"]["sha256"],
            },
        )
        byte_equal = files_byte_equal(legacy_raw, all_q4_raw)
        if not byte_equal or legacy_stage["artifact"]["sha256"] != all_q4_stage["artifact"]["sha256"]:
            raise ValueError("legacy and all-q4 decoded RAW outputs differ")
        artifacts["legacy_no_record_raw"] = legacy_stage["artifact"]
        artifacts["extended_all_q4_record_raw"] = all_q4_stage["artifact"]
        raw_proof = {
            "schema": "ddm_sr1_raw_byte_identity.v1",
            "complete": True,
            "written_at_utc": utc_now(),
            "axis": "[macOS-CPU scorer-free receiver render]",
            "score_claim": False,
            "comparison": "streaming byte-for-byte comparison",
            "byte_equal": byte_equal,
            "bytes_compared": EXPECTED_RAW_BYTES,
            "sha256": legacy_stage["artifact"]["sha256"],
            "archive_decode_chain": {
                "outer_models_raw_fed_to_inner_loader": True,
                "all_q4_hpac_matches_base": (
                    outer_sections["all_q4_record"]["hpac"] == outer_sections["base_no_record"]["hpac"]
                ),
                "all_q4_range_tokens_match_base": (
                    outer["all_q4_record"]["tokens"] == outer["base_no_record"]["tokens"]
                ),
                "retained_token_bytes_match_completed_range_decode": token_proof["decoded_bytes_match_range_receipt"],
                "proof_form": ("exact compositional public-receiver closure; the CLI main function was not invoked"),
            },
            "legacy_no_record": legacy_stage,
            "extended_all_q4_record": all_q4_stage,
        }
        atomic_write_json(resume_from / "stages/05_raw_byte_identity.json", raw_proof)

    schema_archive_marginal = len(selected_bytes) - len(headerless_archive)
    pricing = {
        "schema": "ddm_sr1_real_archive_pricing.v1",
        "complete": True,
        "score_claim": False,
        "base_archive_bytes": len(base_bytes),
        "selected_archive_bytes": len(selected_bytes),
        "selected_delta_archive_bytes": len(selected_bytes) - len(base_bytes),
        "selected_rate_delta_s": (25.0 * (len(selected_bytes) - len(base_bytes)) / ORIGINAL_BYTES),
        "allocation_record_raw_bytes": len(header),
        "selected_headerless_counterfactual_archive_bytes": len(headerless_archive),
        "allocation_record_complete_archive_marginal_bytes": schema_archive_marginal,
        "schema_is_already_counted_in_selected_archive": True,
        "subtracting_schema_again_would_double_count": True,
        "honest_net_delta_archive_bytes": len(selected_bytes) - len(base_bytes),
        "all_q4_record_archive_bytes": len(all_q4_archive),
        "all_q4_record_delta_archive_bytes": len(all_q4_archive) - len(base_bytes),
        "pricing_basis": "real deterministic ZIP after real LZMA model coding",
        "headerless_counterfactual_is_not_receiver_valid": True,
    }
    atomic_write_json(resume_from / "stages/03_real_archive_pricing.json", pricing)

    source_files = {
        "proof_harness": Path(__file__).resolve(),
        "sd1_reference": SD1_SOURCE,
        "extended_inflate": extended_runtime_dir / "inflate.py",
        "extended_receiver": extended_runtime_dir / "receiver.py",
        "legacy_inflate": legacy_runtime_dir / "inflate.py",
        "legacy_receiver": legacy_runtime_dir / "receiver.py",
    }
    receipt = {
        "schema": "ddm_sr1_semantic_alloc_schema_receipt.v1",
        "complete": raw_proof is not None and raw_proof["byte_equal"],
        "written_at_utc": utc_now(),
        "axis": "[scorer-free receiver structural proof; exact archive bytes]",
        "score_claim": False,
        "promotion_eligible": False,
        "working_tree_base_commit": git_head(),
        "argv": sys.argv,
        "inputs": {
            "base_archive": artifact_record(base_path),
            "selected_archive": artifact_record(selected_path),
        },
        "sources": {name: artifact_record(path) for name, path in source_files.items()},
        "reused_schema_lineage": {
            "commit": "cf53216e3e856c15f849bcfe96a5dd4717da2d04",
            "path": "src/tac/pr130_runtime/dv1_cpu_runtime/inflate.py",
            "mechanism": "counted SD1M v1 per-tensor q3/q4 allocation parser",
        },
        "retained_payloads": artifacts,
        "parity": parity,
        "raw_byte_identity": raw_proof,
        "pricing": pricing,
        "boundaries": {
            "scorer_run": False,
            "pose_measured": False,
            "modal_used": False,
            "training_run": False,
            "upstream_modified": False,
            "archive_composed_with_other_rate_levers": False,
            "raw_video_byte_identity_measured": raw_proof is not None,
            "exact_pointer_moved": False,
        },
        "pose_fire_order": {
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "MAIN scorer owner",
            "consumer_store": ("/Volumes/VertigoDataTier/pact/ddm_cx2_20260809/evaluation/q4_control/"),
            "fire_trigger": (
                "the n600 scorer lane is free and both exact archives can run on the same pinned real evaluate path"
            ),
            "measurement": (
                "matched uniform-q4 versus selected-mixed n600 d_pose through "
                "upstream/evaluate.py on the identical runtime, GT, device, and pair order"
            ),
        },
    }
    atomic_write_json(out_dir / "SR1_RECEIVER_PROOF.json", receipt)
    atomic_write_json(resume_from / "terminal.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-archive", type=Path, default=DEFAULT_BASE_ARCHIVE)
    parser.add_argument("--selected-archive", type=Path, default=DEFAULT_SELECTED_ARCHIVE)
    parser.add_argument("--extended-runtime", type=Path, default=DEFAULT_EXTENDED_RUNTIME)
    parser.add_argument("--legacy-runtime", type=Path, default=DEFAULT_LEGACY_RUNTIME)
    parser.add_argument("--token-checkpoint", type=Path, default=DEFAULT_TOKEN_CHECKPOINT)
    parser.add_argument(
        "--range-decode-receipt",
        type=Path,
        default=DEFAULT_RANGE_DECODE_RECEIPT,
    )
    parser.add_argument(
        "--render-byte-proof",
        action="store_true",
        help="render and retain the strict legacy/all-q4 RAW byte comparison",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--minimum-free-bytes", type=int, default=100_000_000)
    return parser.parse_args()


def main() -> None:
    receipt = run(parse_args())
    print(
        json.dumps(
            {
                "complete": receipt["complete"],
                "receipt": str(
                    Path(receipt["retained_payloads"]["base_no_record_archive"]["path"]).parents[1]
                    / "SR1_RECEIVER_PROOF.json"
                ),
                "honest_net_delta_archive_bytes": receipt["pricing"]["honest_net_delta_archive_bytes"],
                "legacy_exact": receipt["parity"]["all_q4_record_full_loader_vs_today_shipped_full_loader"][
                    "all_exact"
                ],
                "selected_exact": receipt["parity"]["selected_mixed_full_loader_vs_independent_sd1_parser"][
                    "all_exact"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
