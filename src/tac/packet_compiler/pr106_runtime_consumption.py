"""Runtime-sidecar consumption proof for PR106/R2 sidecar packets.

This module deliberately stops before full-frame inflate or scoring. It imports
the submission runtime's own ``inflate.py`` parser/sidecar decoder, feeds it a
valid semantic sidecar mutation, and records whether the runtime-visible
correction arrays changed. That proves the sidecar bytes are consumed by the
runtime decode path without turning the result into a score claim.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

from tac.packet_compiler.pr106_sidecar_packet import (
    PR106_SIDECAR_FORMAT_BROTLI,
    PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
    emit_pr106_sidecar_packet,
    emit_single_stored_member_archive,
    mutate_pr106_sidecar_semantic_correction,
    parse_pr106_sidecar_packet,
    pr106_sidecar_mutation_manifest,
    read_single_stored_member_archive,
    sha256_hex,
)

_RUNTIME_TRANSIENT_MODULES = ("codec", "model", "pr101_grammar")


@contextmanager
def _runtime_import_context(runtime_dir: Path) -> Iterator[None]:
    saved_path = list(sys.path)
    sentinel = object()
    saved_modules: dict[str, ModuleType | object] = {
        name: sys.modules.get(name, sentinel) for name in _RUNTIME_TRANSIENT_MODULES
    }
    for name in _RUNTIME_TRANSIENT_MODULES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(runtime_dir / "src"))
    sys.path.insert(0, str(runtime_dir))
    try:
        yield
    finally:
        sys.path[:] = saved_path
        for name, module in saved_modules.items():
            if module is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module  # type: ignore[assignment]


def load_pr106_sidecar_runtime(runtime_dir: Path) -> ModuleType:
    """Load a submission runtime ``inflate.py`` without polluting global imports."""

    runtime_dir = Path(runtime_dir)
    inflate_py = runtime_dir / "inflate.py"
    if not inflate_py.is_file():
        raise FileNotFoundError(f"runtime inflate.py not found: {inflate_py}")
    module_name = f"_pact_runtime_{runtime_dir.name}_inflate"
    spec = importlib.util.spec_from_file_location(module_name, inflate_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load import spec for {inflate_py}")
    module = importlib.util.module_from_spec(spec)
    with _runtime_import_context(runtime_dir):
        spec.loader.exec_module(module)
    return module


def _array_bytes(array: Any, dtype_name: str) -> bytes:
    if not hasattr(array, "astype"):
        raise TypeError(f"runtime decoder returned non-array object: {type(array)!r}")
    return array.astype(dtype_name, copy=False).tobytes()


def _tensor_sha256(tensor: Any) -> str:
    if not all(hasattr(tensor, name) for name in ("detach", "cpu", "numpy")):
        raise TypeError(f"expected tensor-like latents; got {type(tensor)!r}")
    return sha256_hex(tensor.detach().cpu().contiguous().numpy().tobytes())


def runtime_sidecar_correction_digest(
    runtime_module: ModuleType,
    member_payload: bytes,
) -> dict[str, object]:
    """Return a stable digest of corrections visible to runtime ``inflate.py``."""

    parsed = runtime_module.parse_sidecar_archive(member_payload)
    if isinstance(parsed, tuple) and len(parsed) == 2:
        pr106_bytes, sidecar_blob = parsed
        dim_arr, delta_q_arr = runtime_module.decode_sidecar_corrections(sidecar_blob)
        format_id = PR106_SIDECAR_FORMAT_BROTLI
    elif isinstance(parsed, tuple) and len(parsed) == 4:
        format_id, pr106_bytes, sidecar_blob, framing_meta = parsed
        if format_id == PR106_SIDECAR_FORMAT_BROTLI:
            dim_arr, delta_q_arr = runtime_module.decode_brotli_sidecar(sidecar_blob)
        elif format_id == PR106_SIDECAR_FORMAT_PR101_GRAMMAR:
            if framing_meta is None:
                raise ValueError("runtime returned no framing_meta for format_id=0x02")
            dim_arr, delta_q_arr = runtime_module.decode_pr101_grammar_sidecar(
                sidecar_blob,
                framing_meta,
            )
        else:
            raise ValueError(f"runtime returned unsupported format_id=0x{format_id:02X}")
    else:
        raise TypeError(
            "runtime parse_sidecar_archive returned unexpected shape: "
            f"{type(parsed)!r} {parsed!r}"
        )

    dim_bytes = _array_bytes(dim_arr, "uint8")
    delta_bytes = _array_bytes(delta_q_arr, "int8")
    _, latents, _ = runtime_module.parse_packed_archive(pr106_bytes)
    source_latents_sha256 = _tensor_sha256(latents)
    corrected_latents = runtime_module.apply_sidecar_corrections(
        latents.clone(),
        dim_arr,
        delta_q_arr,
    )
    corrected_latents_sha256 = _tensor_sha256(corrected_latents)
    return {
        "format_id": f"0x{int(format_id):02X}",
        "n_pairs": len(dim_arr),
        "n_corrections": sum(1 for value in dim_arr.tolist() if int(value) != 255),
        "dim_sha256": sha256_hex(dim_bytes),
        "delta_q_sha256": sha256_hex(delta_bytes),
        "combined_sha256": sha256_hex(dim_bytes + delta_bytes),
        "source_latents_sha256": source_latents_sha256,
        "corrected_latents_sha256": corrected_latents_sha256,
        "latents_changed_by_sidecar": source_latents_sha256 != corrected_latents_sha256,
    }


def prove_pr106_sidecar_runtime_decode_consumption(
    *,
    archive_path: Path,
    runtime_dir: Path,
    expected_member_name: str = "0.bin",
) -> dict[str, object]:
    """Prove a valid sidecar mutation is consumed by the runtime decoder.

    The returned manifest is intentionally non-promotable: it does not run the
    HNeRV decoder, does not write raw frames, and does not score. It closes the
    narrower "runtime sidecar parser/decoder consumes the changed section"
    requirement so exact eval can be reserved for candidates with real byte
    movement.
    """

    archive_path = Path(archive_path)
    runtime_dir = Path(runtime_dir)
    archive_bytes = archive_path.read_bytes()
    member = read_single_stored_member_archive(
        archive_bytes,
        expected_member_name=expected_member_name,
    )
    source_packet = parse_pr106_sidecar_packet(member.payload)
    mutated_packet, mutation = mutate_pr106_sidecar_semantic_correction(source_packet)
    mutated_payload = emit_pr106_sidecar_packet(mutated_packet)
    mutated_member = replace(member, payload=mutated_payload)
    mutated_archive_bytes = emit_single_stored_member_archive(mutated_member)

    runtime = load_pr106_sidecar_runtime(runtime_dir)
    source_digest = runtime_sidecar_correction_digest(runtime, member.payload)
    mutated_digest = runtime_sidecar_correction_digest(runtime, mutated_payload)
    semantic_changed = (
        source_digest["combined_sha256"] != mutated_digest["combined_sha256"]
    )
    corrected_latents_changed = (
        source_digest["corrected_latents_sha256"]
        != mutated_digest["corrected_latents_sha256"]
    )
    manifest = pr106_sidecar_mutation_manifest(
        source_packet,
        mutated_packet,
        mutation,
        source_archive_sha256=sha256_hex(archive_bytes),
        mutated_archive_sha256=sha256_hex(mutated_archive_bytes),
    )
    manifest.update(
        {
            "schema": "pr106_sidecar_runtime_decode_consumption_proof_v1",
            "proof_scope": (
                "actual_submission_inflate_py_sidecar_decode_and_apply_not_full_frame"
            ),
            "runtime_dir": runtime_dir.as_posix(),
            "runtime_inflate_py_sha256": sha256_hex((runtime_dir / "inflate.py").read_bytes()),
            "source_runtime_correction_digest": source_digest,
            "mutated_runtime_correction_digest": mutated_digest,
            "runtime_semantic_digest_changed": semantic_changed,
            "runtime_corrected_latents_digest_changed": corrected_latents_changed,
            "runtime_sidecar_decode_consumption_claim": semantic_changed,
            "runtime_sidecar_apply_consumption_claim": (
                semantic_changed and corrected_latents_changed
            ),
            "full_frame_inflate_output_parity_claim": False,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "required_next_proof": (
                "full-frame source-vs-candidate inflate output parity or exact "
                "same-runtime auth eval with archive/runtime custody"
            ),
        }
    )
    return manifest


def dumps_runtime_consumption_manifest(manifest: dict[str, object]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"
