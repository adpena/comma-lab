# SPDX-License-Identifier: MIT
"""Runtime-sidecar consumption proof for PR106/R2 sidecar packets.

This module deliberately stops before full-frame inflate or scoring. It imports
the submission runtime's own ``inflate.py`` parser/sidecar decoder, feeds it a
valid semantic sidecar mutation, and records whether the runtime-visible
correction arrays changed. That proves the sidecar bytes are consumed by the
runtime decode path without turning the result into a score claim.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

from tac.packet_compiler.pr106_sidecar_packet import (
    PR106_SIDECAR_FORMAT_BROTLI,
    PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA,
    PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
    PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
    canonical_expected_sha256,
    emit_pr106_sidecar_packet,
    emit_single_stored_member_archive,
    mutate_pr106_sidecar_semantic_correction,
    parse_pr106_sidecar_packet,
    pr106_sidecar_consumed_byte_proof,
    pr106_sidecar_mutation_manifest,
    read_single_stored_member_archive,
    sha256_hex,
)

_RUNTIME_TRANSIENT_MODULES = ("codec", "model", "pr101_grammar")
_RUNTIME_TRANSIENT_PACKAGE_PREFIXES = ("tac",)
_RUNTIME_SOURCE_CANDIDATES = (
    "inflate.sh",
    "inflate.py",
    "src/codec.py",
    "src/model.py",
    "src/pr101_grammar.py",
)
_RUNTIME_SOURCE_EXTRA_GLOBS = ("src/tac/**/*.py",)
_RUNTIME_SOURCE_REQUIRED = (
    "inflate.sh",
    "inflate.py",
    "src/codec.py",
    "src/model.py",
)


def _runtime_shadowed_module_names() -> set[str]:
    names = set(_RUNTIME_TRANSIENT_MODULES)
    for name in list(sys.modules):
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in _RUNTIME_TRANSIENT_PACKAGE_PREFIXES
        ):
            names.add(name)
    return names


@contextmanager
def _runtime_import_context(runtime_dir: Path) -> Iterator[None]:
    saved_path = list(sys.path)
    saved_dont_write_bytecode = sys.dont_write_bytecode
    sentinel = object()
    shadowed_names = _runtime_shadowed_module_names()
    saved_modules: dict[str, ModuleType | object] = {
        name: sys.modules.get(name, sentinel) for name in shadowed_names
    }
    for name in shadowed_names:
        sys.modules.pop(name, None)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(runtime_dir / "src"))
    sys.path.insert(0, str(runtime_dir))
    try:
        yield
    finally:
        sys.dont_write_bytecode = saved_dont_write_bytecode
        sys.path[:] = saved_path
        for name in list(sys.modules):
            if name in shadowed_names or any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in _RUNTIME_TRANSIENT_PACKAGE_PREFIXES
            ):
                sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module  # type: ignore[assignment]


def load_pr106_runtime_module(
    runtime_dir: Path,
    relative_path: str | Path,
    *,
    module_tag: str,
) -> ModuleType:
    """Load one submission runtime Python file without polluting global imports."""
    runtime_dir = Path(runtime_dir)
    module_path = runtime_dir / relative_path
    if not module_path.is_file():
        raise FileNotFoundError(f"runtime module not found: {module_path}")
    module_name = f"_pact_runtime_{runtime_dir.name}_{module_tag}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    with _runtime_import_context(runtime_dir):
        spec.loader.exec_module(module)
    return module


def load_pr106_sidecar_runtime(runtime_dir: Path) -> ModuleType:
    """Load a submission runtime ``inflate.py`` without polluting global imports."""

    return load_pr106_runtime_module(runtime_dir, "inflate.py", module_tag="inflate")


def load_pr106_runtime_codec(runtime_dir: Path) -> ModuleType:
    """Load a submission runtime ``src/codec.py`` without polluting global imports."""

    return load_pr106_runtime_module(runtime_dir, "src/codec.py", module_tag="codec")


def pr106_runtime_source_manifest(runtime_dir: Path) -> dict[str, object]:
    """Return a deterministic hash over the runtime files used by this proof."""

    runtime_dir = Path(runtime_dir)
    missing_required = [
        rel for rel in _RUNTIME_SOURCE_REQUIRED if not (runtime_dir / rel).exists()
    ]
    if missing_required:
        raise ValueError(
            "runtime dir missing required source files: " + ", ".join(missing_required)
        )
    rel_paths: list[str] = []
    seen: set[str] = set()
    for rel in _RUNTIME_SOURCE_CANDIDATES:
        path = runtime_dir / rel
        if not path.exists():
            continue
        rel_paths.append(rel)
        seen.add(rel)
    for pattern in _RUNTIME_SOURCE_EXTRA_GLOBS:
        for path in sorted(runtime_dir.glob(pattern)):
            if not path.is_file():
                continue
            rel = path.relative_to(runtime_dir).as_posix()
            if rel in seen:
                continue
            rel_paths.append(rel)
            seen.add(rel)

    files: list[dict[str, object]] = []
    for rel in rel_paths:
        path = runtime_dir / rel
        if not path.is_file():
            raise ValueError(f"runtime source candidate is not a file: {path}")
        payload = path.read_bytes()
        files.append(
            {
                "path": rel,
                "bytes": len(payload),
                "sha256": sha256_hex(payload),
                "mode": f"{path.stat().st_mode & 0o777:04o}",
            }
        )
    if not files:
        raise ValueError(f"runtime dir has no recognized source files: {runtime_dir}")
    tree_payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    content_payload = json.dumps(
        [
            {
                "path": row["path"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
            for row in files
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "schema": "pr106_runtime_source_manifest_v1",
        "runtime_dir": runtime_dir.as_posix(),
        "required_files": list(_RUNTIME_SOURCE_REQUIRED),
        "extra_globs": list(_RUNTIME_SOURCE_EXTRA_GLOBS),
        "files": files,
        "runtime_source_tree_sha256": sha256_hex(tree_payload),
        "runtime_content_tree_sha256": sha256_hex(content_payload),
    }


def _array_bytes(array: Any, dtype_name: str) -> bytes:
    if not hasattr(array, "astype"):
        raise TypeError(f"runtime decoder returned non-array object: {type(array)!r}")
    return array.astype(dtype_name, copy=False).tobytes()


def _tensor_sha256(tensor: Any) -> str:
    if not all(hasattr(tensor, name) for name in ("detach", "cpu", "numpy")):
        raise TypeError(f"expected tensor-like latents; got {type(tensor)!r}")
    return sha256_hex(tensor.detach().cpu().contiguous().numpy().tobytes())


def _apply_runtime_sidecar_corrections(
    runtime_module: ModuleType,
    latents: Any,
    dim_arr: Any,
    delta_q_arr: Any,
) -> Any:
    """Apply runtime corrections while honoring in-place and returned tensors."""

    corrected = runtime_module.apply_sidecar_corrections(
        latents,
        dim_arr,
        delta_q_arr,
    )
    return latents if corrected is None else corrected


def _decode_runtime_sidecar_correction_passes(
    runtime_module: ModuleType,
    member_payload: bytes,
) -> tuple[int, bytes, list[dict[str, Any]]]:
    """Decode a PR106 sidecar packet through the selected runtime parser."""

    parsed = runtime_module.parse_sidecar_archive(member_payload)
    if isinstance(parsed, tuple) and len(parsed) == 2:
        pr106_bytes, sidecar_blob = parsed
        dim_arr, delta_q_arr = runtime_module.decode_sidecar_corrections(sidecar_blob)
        return PR106_SIDECAR_FORMAT_BROTLI, pr106_bytes, [
            {
                "section_name": "sidecar_payload",
                "dim_arr": dim_arr,
                "delta_q_arr": delta_q_arr,
            }
        ]
    if isinstance(parsed, tuple) and len(parsed) == 4:
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
        elif format_id in (
            PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
        ):
            dim_arr, delta_q_arr = (
                runtime_module.decode_pr101_fixed_meta_rank_elided_sidecar(sidecar_blob)
            )
        elif format_id in (
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED,
        ):
            dim_arr, delta_q_arr = (
                runtime_module.decode_pr101_fixed_meta_noop_rank_elided_sidecar(
                    sidecar_blob
                )
            )
        elif (
            format_id
            == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
        ):
            dim_arr, delta_q_arr = (
                runtime_module.decode_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar(
                    sidecar_blob
                )
            )
        elif format_id == PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA:
            (
                base_dim_arr,
                base_delta_q_arr,
                extra_dim_arr,
                extra_delta_q_arr,
            ) = runtime_module.decode_format0d_sidecar(sidecar_blob)
            return int(format_id), pr106_bytes, [
                {
                    "section_name": "base_format0c_sidecar_payload",
                    "dim_arr": base_dim_arr,
                    "delta_q_arr": base_delta_q_arr,
                },
                {
                    "section_name": "extra_pr101_ranked_no_op_payload",
                    "dim_arr": extra_dim_arr,
                    "delta_q_arr": extra_delta_q_arr,
                },
            ]
        else:
            raise ValueError(f"runtime returned unsupported format_id=0x{format_id:02X}")
        return int(format_id), pr106_bytes, [
            {
                "section_name": "sidecar_payload",
                "dim_arr": dim_arr,
                "delta_q_arr": delta_q_arr,
            }
        ]
    raise TypeError(
        "runtime parse_sidecar_archive returned unexpected shape: "
        f"{type(parsed)!r} {parsed!r}"
    )


def _decode_runtime_sidecar_payload(
    runtime_module: ModuleType,
    member_payload: bytes,
) -> tuple[int, bytes, Any, Any]:
    """Decode a single-pass PR106 sidecar packet through the runtime parser."""

    format_id, pr106_bytes, correction_passes = _decode_runtime_sidecar_correction_passes(
        runtime_module,
        member_payload,
    )
    if len(correction_passes) != 1:
        raise ValueError(
            f"format_id=0x{format_id:02X} has {len(correction_passes)} correction passes"
        )
    correction = correction_passes[0]
    return format_id, pr106_bytes, correction["dim_arr"], correction["delta_q_arr"]


def _correction_pass_manifest(correction: dict[str, Any]) -> dict[str, object]:
    dim_bytes = _array_bytes(correction["dim_arr"], "uint8")
    delta_bytes = _array_bytes(correction["delta_q_arr"], "int8")
    return {
        "section_name": str(correction["section_name"]),
        "n_pairs": len(correction["dim_arr"]),
        "n_corrections": sum(
            1 for value in correction["dim_arr"].tolist() if int(value) != 255
        ),
        "dim_sha256": sha256_hex(dim_bytes),
        "delta_q_sha256": sha256_hex(delta_bytes),
        "combined_sha256": sha256_hex(dim_bytes + delta_bytes),
    }


def _runtime_section_identity_rows(
    packet_ir_consumed_byte_proof: dict[str, object],
    consumed_sections: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    proof_sections = packet_ir_consumed_byte_proof.get("sections", [])
    if not isinstance(proof_sections, list):
        return rows
    for section in proof_sections:
        if not isinstance(section, dict):
            continue
        name = str(section.get("name", ""))
        if consumed_sections.get(name) is not True:
            continue
        rows.append(
            {
                "name": name,
                "sha256": section.get("sha256"),
                "hash_domain": section.get("hash_domain"),
                "sha256_domain": section.get("sha256_domain"),
                "bytes": section.get("bytes", section.get("byte_count")),
                "offset": section.get("offset", section.get("offset_start")),
                "consumed": True,
                "score_affecting": section.get("score_affecting") is True,
                "identity_source": (
                    "packet_ir_consumed_byte_proof_filtered_by_runtime_probe"
                ),
                "runtime_consumption_evidence": "runtime_section_mutation_probe",
            }
        )
    return rows


def runtime_sidecar_correction_digest(
    runtime_module: ModuleType,
    member_payload: bytes,
) -> dict[str, object]:
    """Return a stable digest of corrections visible to runtime ``inflate.py``."""

    format_id, pr106_bytes, correction_passes = _decode_runtime_sidecar_correction_passes(
        runtime_module,
        member_payload,
    )
    dim_bytes = b"".join(
        _array_bytes(correction["dim_arr"], "uint8") for correction in correction_passes
    )
    delta_bytes = b"".join(
        _array_bytes(correction["delta_q_arr"], "int8")
        for correction in correction_passes
    )
    _, latents, _ = runtime_module.parse_packed_archive(pr106_bytes)
    source_latents_sha256 = _tensor_sha256(latents)
    corrected_latents = latents.clone()
    for correction in correction_passes:
        corrected_latents = _apply_runtime_sidecar_corrections(
            runtime_module,
            corrected_latents,
            correction["dim_arr"],
            correction["delta_q_arr"],
        )
    corrected_latents_sha256 = _tensor_sha256(corrected_latents)
    return {
        "format_id": f"0x{int(format_id):02X}",
        "n_passes": len(correction_passes),
        "correction_passes": [
            _correction_pass_manifest(correction) for correction in correction_passes
        ],
        "n_pairs": len(correction_passes[0]["dim_arr"]),
        "n_corrections": sum(
            int(pass_manifest["n_corrections"])
            for pass_manifest in (
                _correction_pass_manifest(correction) for correction in correction_passes
            )
        ),
        "dim_sha256": sha256_hex(dim_bytes),
        "delta_q_sha256": sha256_hex(delta_bytes),
        "combined_sha256": sha256_hex(dim_bytes + delta_bytes),
        "source_latents_sha256": source_latents_sha256,
        "corrected_latents_sha256": corrected_latents_sha256,
        "latents_changed_by_sidecar": source_latents_sha256 != corrected_latents_sha256,
    }


def runtime_framing_meta_consumption_probe(
    runtime_module: ModuleType,
    member_payload: bytes,
    baseline_digest: dict[str, object],
) -> dict[str, object]:
    """Probe whether sidecar framing metadata is runtime-visible."""

    packet = parse_pr106_sidecar_packet(member_payload)
    if packet.format_id == PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA:
        if packet.extra_framing_meta is None:
            return {
                "section": "extra_framing_meta",
                "format_id": f"0x{packet.format_id:02X}",
                "applicable": True,
                "runtime_consumption_claim": False,
                "observation": "format_id_0x0D_missing_extra_framing_meta",
            }

        proof = pr106_sidecar_consumed_byte_proof(packet)
        meta_section = next(
            section
            for section in proof["sections"]
            if section["name"] == "extra_framing_meta"
        )
        mutated_payload = bytearray(member_payload)
        mutated_payload[int(meta_section["offset"])] ^= 0x01
        try:
            mutated_digest = runtime_sidecar_correction_digest(
                runtime_module,
                bytes(mutated_payload),
            )
        except Exception as exc:
            return {
                "section": "extra_framing_meta",
                "format_id": f"0x{packet.format_id:02X}",
                "applicable": True,
                "runtime_consumption_claim": True,
                "observation": "runtime_rejected_mutated_extra_framing_meta",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
        changed = (
            baseline_digest.get("combined_sha256")
            != mutated_digest.get("combined_sha256")
            or baseline_digest.get("corrected_latents_sha256")
            != mutated_digest.get("corrected_latents_sha256")
        )
        return {
            "section": "extra_framing_meta",
            "format_id": f"0x{packet.format_id:02X}",
            "applicable": True,
            "runtime_consumption_claim": changed,
            "observation": (
                "runtime_digest_changed"
                if changed
                else "runtime_digest_unchanged"
            ),
            "mutated_runtime_correction_digest": mutated_digest,
        }

    if packet.format_id != PR106_SIDECAR_FORMAT_PR101_GRAMMAR:
        return {
            "section": "framing_meta",
            "format_id": f"0x{packet.format_id:02X}",
            "applicable": False,
            "runtime_consumption_claim": None,
            "observation": "format_has_no_framing_meta",
        }
    if packet.framing_meta is None:
        return {
            "section": "framing_meta",
            "format_id": f"0x{packet.format_id:02X}",
            "applicable": True,
            "runtime_consumption_claim": False,
            "observation": "format_id_0x02_missing_framing_meta",
        }

    mutated_meta = bytearray(packet.framing_meta)
    mutated_meta[0] ^= 0x01
    mutated_payload = emit_pr106_sidecar_packet(
        type(packet)(
            format_id=packet.format_id,
            pr106_bytes=packet.pr106_bytes,
            sidecar_payload=packet.sidecar_payload,
            framing_meta=bytes(mutated_meta),
        )
    )
    try:
        mutated_digest = runtime_sidecar_correction_digest(runtime_module, mutated_payload)
    except Exception as exc:
        return {
            "section": "framing_meta",
            "format_id": f"0x{packet.format_id:02X}",
            "applicable": True,
            "runtime_consumption_claim": True,
            "observation": "runtime_rejected_mutated_framing_meta",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
    changed = (
        baseline_digest.get("combined_sha256") != mutated_digest.get("combined_sha256")
        or baseline_digest.get("corrected_latents_sha256")
        != mutated_digest.get("corrected_latents_sha256")
    )
    return {
        "section": "framing_meta",
        "format_id": f"0x{packet.format_id:02X}",
        "applicable": True,
        "runtime_consumption_claim": changed,
        "observation": "runtime_digest_changed" if changed else "runtime_digest_unchanged",
        "mutated_runtime_correction_digest": mutated_digest,
    }


def _runtime_digest_changed(
    baseline_digest: dict[str, object],
    mutated_digest: dict[str, object],
) -> bool:
    return (
        baseline_digest.get("combined_sha256") != mutated_digest.get("combined_sha256")
        or baseline_digest.get("corrected_latents_sha256")
        != mutated_digest.get("corrected_latents_sha256")
    )


def _runtime_sidecar_section_consumption_probe(
    runtime_module: ModuleType,
    packet: Any,
    *,
    section_name: str,
    baseline_digest: dict[str, object],
) -> dict[str, object]:
    try:
        mutated_packet, mutation = mutate_pr106_sidecar_semantic_correction(
            packet,
            section_name=section_name,
        )
        mutated_payload = emit_pr106_sidecar_packet(mutated_packet)
        mutated_digest = runtime_sidecar_correction_digest(runtime_module, mutated_payload)
    except Exception as exc:
        return {
            "section": section_name,
            "format_id": f"0x{packet.format_id:02X}",
            "runtime_consumption_claim": False,
            "observation": "runtime_rejected_valid_section_mutation",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
    changed = _runtime_digest_changed(baseline_digest, mutated_digest)
    return {
        "section": section_name,
        "format_id": f"0x{packet.format_id:02X}",
        "runtime_consumption_claim": changed,
        "observation": "runtime_digest_changed" if changed else "runtime_digest_unchanged",
        "mutation": mutation.__dict__,
        "mutated_runtime_correction_digest": mutated_digest,
    }


def runtime_sidecar_section_consumption_probes(
    runtime_module: ModuleType,
    member_payload: bytes,
    baseline_digest: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Probe each runtime-visible score-affecting sidecar section independently."""

    packet = parse_pr106_sidecar_packet(member_payload)
    probes: dict[str, dict[str, object]] = {
        "pr106_payload": {
            "section": "pr106_payload",
            "format_id": f"0x{packet.format_id:02X}",
            "runtime_consumption_claim": True,
            "observation": "inner_pr106_payload_parsed_by_runtime_digest",
        }
    }
    if packet.format_id == PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA:
        section_names = (
            "base_format0c_sidecar_payload",
            "extra_pr101_ranked_no_op_payload",
        )
    else:
        section_names = ("sidecar_payload",)
    for section_name in section_names:
        probes[section_name] = _runtime_sidecar_section_consumption_probe(
            runtime_module,
            packet,
            section_name=section_name,
            baseline_digest=baseline_digest,
        )
    framing_probe = runtime_framing_meta_consumption_probe(
        runtime_module,
        member_payload,
        baseline_digest,
    )
    probes[str(framing_probe["section"])] = framing_probe
    return probes


def _section_probe_claim(
    probes: dict[str, dict[str, object]],
    section_name: str,
) -> bool | None:
    probe = probes.get(section_name)
    if probe is None:
        return None
    claim = probe.get("runtime_consumption_claim")
    return claim if isinstance(claim, bool) else None


def runtime_full_frame_streaming_digest(
    runtime_module: ModuleType,
    member_payload: bytes,
    *,
    device: str = "cpu",
    batch_pairs: int | None = None,
    max_pairs: int | None = None,
) -> dict[str, object]:
    """Hash runtime-rendered frames without materializing a ``.raw`` file.

    This follows the paired submission runtime's HNeRV decode loop: parse
    sidecar, parse inner PR106 payload, apply corrections, instantiate
    ``HNeRVDecoder``, bicubic-upsample to camera resolution, round to uint8,
    and stream the exact bytes into SHA-256.

    ``max_pairs`` is for cheap prefix tests only. Full-frame parity may only
    be claimed when it is ``None`` and every pair is rendered.
    """

    if device not in {"cpu", "cuda"}:
        raise ValueError(f"device must be cpu or cuda; got {device!r}")
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError(f"max_pairs must be positive when set; got {max_pairs}")
    if batch_pairs is not None and batch_pairs <= 0:
        raise ValueError(f"batch_pairs must be positive when set; got {batch_pairs}")

    format_id, pr106_bytes, correction_passes = _decode_runtime_sidecar_correction_passes(
        runtime_module,
        member_payload,
    )
    if device == "cuda" and not runtime_module.torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is unavailable")

    decoder_sd, latents, meta = runtime_module.parse_packed_archive(pr106_bytes)
    source_latents_sha256 = _tensor_sha256(latents)
    for correction in correction_passes:
        latents = _apply_runtime_sidecar_corrections(
            runtime_module,
            latents,
            correction["dim_arr"],
            correction["delta_q_arr"],
        )
    corrected_latents_sha256 = _tensor_sha256(latents)
    torch_device = runtime_module.torch.device(device)
    decoder = runtime_module.HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(torch_device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()
    latents = latents.to(torch_device)

    n_pairs_total = int(meta["n_pairs"])
    n_pairs_hashed = min(n_pairs_total, max_pairs) if max_pairs is not None else n_pairs_total
    eval_h, eval_w = meta["eval_size"]
    camera_h = int(getattr(runtime_module, "CAMERA_H", 874))
    camera_w = int(getattr(runtime_module, "CAMERA_W", 1164))
    pair_batch = batch_pairs or int(getattr(runtime_module, "DEFAULT_BATCH_PAIRS", 16))

    sha = hashlib.sha256()

    total_frames = 0
    total_bytes = 0
    start = time.monotonic()
    with runtime_module.torch.inference_mode():
        for i in range(0, n_pairs_hashed, pair_batch):
            j = min(i + pair_batch, n_pairs_hashed)
            decoded = decoder(latents[i:j])
            flat = decoded.reshape((j - i) * 2, 3, eval_h, eval_w)
            up = runtime_module.F.interpolate(
                flat,
                size=(camera_h, camera_w),
                mode="bicubic",
                align_corners=False,
            )
            frames = (
                up.clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(runtime_module.torch.uint8)
                .cpu()
                .numpy()
            )
            payload = frames.tobytes()
            sha.update(payload)
            total_frames += int(frames.shape[0])
            total_bytes += len(payload)

    full_frame = max_pairs is None and n_pairs_hashed == n_pairs_total
    return {
        "schema": "pr106_runtime_full_frame_streaming_digest_v1",
        "format_id": f"0x{int(format_id):02X}",
        "n_passes": len(correction_passes),
        "correction_passes": [
            _correction_pass_manifest(correction) for correction in correction_passes
        ],
        "device": device,
        "batch_pairs": pair_batch,
        "max_pairs": max_pairs,
        "n_pairs_total": n_pairs_total,
        "n_pairs_hashed": n_pairs_hashed,
        "total_frames": total_frames,
        "total_bytes": total_bytes,
        "eval_size": [int(eval_h), int(eval_w)],
        "camera_size": [camera_h, camera_w],
        "source_latents_sha256": source_latents_sha256,
        "corrected_latents_sha256": corrected_latents_sha256,
        "latents_changed_by_sidecar": source_latents_sha256
        != corrected_latents_sha256,
        "full_frame_digest": full_frame,
        "streaming_raw_sha256": sha.hexdigest(),
        "elapsed_seconds": time.monotonic() - start,
        "score_claim": False,
    }


def prove_pr106_same_runtime_full_frame_parity(
    *,
    source_archive_path: Path,
    candidate_archive_path: Path,
    runtime_dir: Path,
    expected_member_name: str | None = None,
    device: str = "cpu",
    batch_pairs: int | None = None,
    max_pairs: int | None = None,
) -> dict[str, object]:
    """Compare two PR106 sidecar archives through one runtime render loop."""

    source_archive_path = Path(source_archive_path)
    candidate_archive_path = Path(candidate_archive_path)
    runtime_dir = Path(runtime_dir)
    source_archive_bytes = source_archive_path.read_bytes()
    candidate_archive_bytes = candidate_archive_path.read_bytes()
    source_member = read_single_stored_member_archive(
        source_archive_bytes,
        expected_member_name=expected_member_name,
    )
    candidate_member = read_single_stored_member_archive(
        candidate_archive_bytes,
        expected_member_name=expected_member_name,
    )
    runtime_manifest = pr106_runtime_source_manifest(runtime_dir)
    runtime = load_pr106_sidecar_runtime(runtime_dir)
    source = runtime_full_frame_streaming_digest(
        runtime,
        source_member.payload,
        device=device,
        batch_pairs=batch_pairs,
        max_pairs=max_pairs,
    )
    candidate = runtime_full_frame_streaming_digest(
        runtime,
        candidate_member.payload,
        device=device,
        batch_pairs=batch_pairs,
        max_pairs=max_pairs,
    )
    same_hash = source["streaming_raw_sha256"] == candidate["streaming_raw_sha256"]
    same_bytes = source["total_bytes"] == candidate["total_bytes"]
    full_scope = bool(source["full_frame_digest"] and candidate["full_frame_digest"])
    return {
        "schema": "pr106_same_runtime_streaming_frame_parity_v1",
        "proof_scope": (
            "same_runtime_streaming_full_frame_hash"
            if full_scope
            else "same_runtime_streaming_prefix_hash"
        ),
        "runtime_dir": runtime_dir.as_posix(),
        "runtime_source_manifest": runtime_manifest,
        "runtime_inflate_py_sha256": sha256_hex((runtime_dir / "inflate.py").read_bytes()),
        "source_archive": {
            "path": source_archive_path.as_posix(),
            "bytes": source_archive_path.stat().st_size,
            "sha256": sha256_hex(source_archive_bytes),
            "member_name": source_member.name,
        },
        "candidate_archive": {
            "path": candidate_archive_path.as_posix(),
            "bytes": candidate_archive_path.stat().st_size,
            "sha256": sha256_hex(candidate_archive_bytes),
            "member_name": candidate_member.name,
        },
        "source": source,
        "candidate": candidate,
        "streaming_output_sha256_equal": same_hash,
        "streaming_output_total_bytes_equal": same_bytes,
        "full_frame_inflate_output_parity_claim": full_scope and same_hash and same_bytes,
        "prefix_parity_claim": (not full_scope) and same_hash and same_bytes,
        "device_axis_label": f"local-{device}-streaming-runtime",
        "contest_axis_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "required_next_proof": (
            "exact contest auth eval with archive/runtime custody and explicit "
            "[contest-CUDA]/[contest-CPU] axis labels"
            if full_scope and same_hash and same_bytes
            else "rerun without --max-pairs for full-frame parity before using parity language"
        ),
    }


def prove_pr106_sidecar_runtime_decode_consumption(
    *,
    archive_path: Path,
    runtime_dir: Path,
    expected_member_name: str | None = None,
    expected_archive_sha256: str | None = None,
    expected_runtime_source_tree_sha256: str | None = None,
    expected_runtime_content_tree_sha256: str | None = None,
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
    archive_sha = sha256_hex(archive_bytes)
    expected_archive_sha, expected_archive_sha_well_formed = canonical_expected_sha256(
        expected_archive_sha256
    )
    blockers: list[str] = []
    if expected_archive_sha_well_formed is False:
        blockers.append("expected_archive_sha256_malformed")
    if (
        expected_archive_sha_well_formed is True
        and expected_archive_sha is not None
        and archive_sha != expected_archive_sha
    ):
        blockers.append("expected_archive_sha256_mismatch")
    runtime_manifest = pr106_runtime_source_manifest(runtime_dir)
    expected_runtime_tree_sha, expected_runtime_tree_sha_well_formed = (
        canonical_expected_sha256(expected_runtime_source_tree_sha256)
    )
    runtime_tree_sha = str(runtime_manifest["runtime_source_tree_sha256"])
    runtime_content_tree_sha = str(runtime_manifest["runtime_content_tree_sha256"])
    expected_runtime_tree_matches = (
        None
        if expected_runtime_tree_sha is None or expected_runtime_tree_sha_well_formed is False
        else runtime_tree_sha == expected_runtime_tree_sha
    )
    expected_runtime_content_tree_sha, expected_runtime_content_tree_sha_well_formed = (
        canonical_expected_sha256(expected_runtime_content_tree_sha256)
    )
    expected_runtime_content_tree_matches = (
        None
        if (
            expected_runtime_content_tree_sha is None
            or expected_runtime_content_tree_sha_well_formed is False
        )
        else runtime_content_tree_sha == expected_runtime_content_tree_sha
    )
    runtime_manifest.update(
        {
            "expected_runtime_source_tree_sha256": expected_runtime_tree_sha,
            "expected_runtime_source_tree_sha256_well_formed": (
                expected_runtime_tree_sha_well_formed
            ),
            "expected_runtime_source_tree_sha256_matches": expected_runtime_tree_matches,
            "expected_runtime_content_tree_sha256": expected_runtime_content_tree_sha,
            "expected_runtime_content_tree_sha256_well_formed": (
                expected_runtime_content_tree_sha_well_formed
            ),
            "expected_runtime_content_tree_sha256_matches": (
                expected_runtime_content_tree_matches
            ),
        }
    )
    if expected_runtime_tree_sha_well_formed is False:
        blockers.append("expected_runtime_source_tree_sha256_malformed")
    elif expected_runtime_tree_matches is False:
        blockers.append("expected_runtime_source_tree_sha256_mismatch")
    if expected_runtime_content_tree_sha_well_formed is False:
        blockers.append("expected_runtime_content_tree_sha256_malformed")
    elif expected_runtime_content_tree_matches is False:
        blockers.append("expected_runtime_content_tree_sha256_mismatch")
    member = read_single_stored_member_archive(
        archive_bytes,
        expected_member_name=expected_member_name,
    )
    archive_manifest = {
        "path": archive_path.as_posix(),
        "bytes": len(archive_bytes),
        "sha256": archive_sha,
        "expected_sha256": expected_archive_sha,
        "expected_sha256_well_formed": expected_archive_sha_well_formed,
        "expected_sha256_matches": (
            None
            if expected_archive_sha is None or expected_archive_sha_well_formed is False
            else archive_sha == expected_archive_sha
        ),
        "member_name": member.name,
        "expected_member_name": expected_member_name,
        "expected_member_name_matches": (
            None if expected_member_name is None else member.name == expected_member_name
        ),
        "member_payload_bytes": len(member.payload),
        "member_payload_sha256": sha256_hex(member.payload),
    }
    if blockers:
        return {
            "schema": "pr106_sidecar_runtime_decode_consumption_proof_v1",
            "proof_scope": (
                "actual_submission_inflate_py_sidecar_decode_and_apply_not_full_frame"
            ),
            "archive": archive_manifest,
            "runtime_dir": runtime_dir.as_posix(),
            "runtime_source_manifest": runtime_manifest,
            "archive_member_name": member.name,
            "blockers": blockers,
            "runtime_sidecar_decode_consumption_claim": False,
            "runtime_sidecar_apply_consumption_claim": False,
            "runtime_semantic_digest_changed": False,
            "runtime_corrected_latents_digest_changed": False,
            "packet_ir_consumed_byte_accounting_passed": False,
            "full_frame_inflate_output_parity_claim": False,
            "contest_axis_claim": False,
            "score_claim": False,
            "proof_not_score": True,
            "evidence_axis": "runtime-sidecar-decode-local-no-score",
            "device_axis_label": "local-runtime-decode-no-full-frame",
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "required_next_proof": (
                "rerun with the expected archive SHA-256 matching the exact "
                "archive under review before any runtime-consumption language"
            ),
        }
    source_packet = parse_pr106_sidecar_packet(member.payload)
    source_proof = pr106_sidecar_consumed_byte_proof(source_packet)
    mutated_packet, mutation = mutate_pr106_sidecar_semantic_correction(source_packet)
    mutated_payload = emit_pr106_sidecar_packet(mutated_packet)
    mutated_member = replace(member, payload=mutated_payload)
    mutated_archive_bytes = emit_single_stored_member_archive(mutated_member)
    mutated_proof = pr106_sidecar_consumed_byte_proof(mutated_packet)
    packet_accounting_passed = (
        source_proof.get("all_payload_bytes_accounted") is True
        and mutated_proof.get("all_payload_bytes_accounted") is True
    )
    mutation_manifest = pr106_sidecar_mutation_manifest(
        source_packet,
        mutated_packet,
        mutation,
        source_archive_sha256=archive_sha,
        mutated_archive_sha256=sha256_hex(mutated_archive_bytes),
    )

    try:
        runtime = load_pr106_sidecar_runtime(runtime_dir)
        source_digest = runtime_sidecar_correction_digest(runtime, member.payload)
        section_probes = runtime_sidecar_section_consumption_probes(
            runtime,
            member.payload,
            source_digest,
        )
        mutated_digest = runtime_sidecar_correction_digest(runtime, mutated_payload)
    except Exception as exc:
        blockers.append(f"runtime_sidecar_decode_exception:{type(exc).__name__}")
        mutation_manifest.update(
            {
                "schema": "pr106_sidecar_runtime_decode_consumption_proof_v1",
                "proof_scope": (
                    "actual_submission_inflate_py_sidecar_decode_and_apply_not_full_frame"
                ),
                "archive": archive_manifest,
                "runtime_dir": runtime_dir.as_posix(),
                "runtime_source_manifest": runtime_manifest,
                "runtime_inflate_py_sha256": sha256_hex(
                    (runtime_dir / "inflate.py").read_bytes()
                )
                if (runtime_dir / "inflate.py").is_file()
                else "",
                "archive_member_name": member.name,
                "blockers": blockers,
                "runtime_exception_type": type(exc).__name__,
                "runtime_exception_message": str(exc),
                "source_packet_ir_consumed_byte_proof": source_proof,
                "mutated_packet_ir_consumed_byte_proof": mutated_proof,
                "packet_ir_consumed_byte_accounting_passed": packet_accounting_passed,
                "runtime_sidecar_decode_consumption_claim": False,
                "runtime_sidecar_apply_consumption_claim": False,
                "runtime_semantic_digest_changed": False,
                "runtime_corrected_latents_digest_changed": False,
                "runtime_all_score_affecting_sections_consumed": False,
                "runtime_consumed_score_affecting_sections": {},
                "runtime_consumed_score_affecting_section_identities": [],
                "full_frame_inflate_output_parity_claim": False,
                "contest_axis_claim": False,
                "score_claim": False,
                "proof_not_score": True,
                "evidence_axis": "runtime-sidecar-decode-local-no-score",
                "device_axis_label": "local-runtime-decode-no-full-frame",
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "required_next_proof": (
                    "update the selected submission runtime decoder or retire this "
                    "PacketIR format before paired exact eval"
                ),
            }
        )
        return mutation_manifest
    semantic_changed = (
        source_digest["combined_sha256"] != mutated_digest["combined_sha256"]
    )
    corrected_latents_changed = (
        source_digest["corrected_latents_sha256"]
        != mutated_digest["corrected_latents_sha256"]
    )
    if not packet_accounting_passed:
        blockers.append("packet_ir_consumed_byte_accounting_failed")
    if not semantic_changed:
        blockers.append("runtime_semantic_digest_not_changed")
    if not corrected_latents_changed:
        blockers.append("runtime_corrected_latents_digest_not_changed")
    framing_meta_section = (
        "extra_framing_meta"
        if source_packet.format_id == PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA
        else "framing_meta"
    )
    framing_meta_probe = section_probes.get(framing_meta_section, {})
    framing_meta_claim = framing_meta_probe.get("runtime_consumption_claim")
    if (
        source_packet.format_id == PR106_SIDECAR_FORMAT_PR101_GRAMMAR
        and framing_meta_claim is not True
    ):
        blockers.append("runtime_framing_meta_consumption_not_proven")
    if (
        source_packet.format_id == PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA
        and framing_meta_claim is not True
    ):
        blockers.append("runtime_extra_framing_meta_consumption_not_proven")
    if source_packet.format_id == PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA:
        base_claim = _section_probe_claim(section_probes, "base_format0c_sidecar_payload")
        extra_claim = _section_probe_claim(section_probes, "extra_pr101_ranked_no_op_payload")
        if base_claim is not True:
            blockers.append("runtime_base_format0c_sidecar_payload_consumption_not_proven")
        if extra_claim is not True:
            blockers.append("runtime_extra_pr101_ranked_no_op_payload_consumption_not_proven")
        decode_claim = bool(base_claim is True and extra_claim is True and packet_accounting_passed)
    else:
        sidecar_claim = _section_probe_claim(section_probes, "sidecar_payload")
        if sidecar_claim is not True:
            blockers.append("runtime_sidecar_payload_consumption_not_proven")
        decode_claim = bool(sidecar_claim is True and packet_accounting_passed)
    apply_claim = bool(decode_claim and corrected_latents_changed)
    runtime_consumed_sections = (
        {
            "pr106_payload": True,
            "sidecar_payload": _section_probe_claim(section_probes, "sidecar_payload"),
            "framing_meta": framing_meta_claim,
        }
        if source_packet.format_id != PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA
        else {
            "pr106_payload": True,
            "base_format0c_sidecar_payload": _section_probe_claim(
                section_probes,
                "base_format0c_sidecar_payload",
            ),
            "extra_pr101_ranked_no_op_payload": _section_probe_claim(
                section_probes,
                "extra_pr101_ranked_no_op_payload",
            ),
            "extra_framing_meta": framing_meta_claim,
        }
    )
    runtime_section_identities = _runtime_section_identity_rows(
        source_proof,
        runtime_consumed_sections,
    )
    runtime_apply_order = (
        [
            "base_format0c_corrections",
            "extra_pr101_ranked_no_op_corrections",
        ]
        if source_packet.format_id == PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA
        else None
    )
    runtime_all_sections_consumed = (
        decode_claim
        and (
            source_packet.format_id
            not in (
                PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
                PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA,
            )
            or framing_meta_claim is True
        )
    )
    manifest = mutation_manifest
    manifest.update(
        {
            "schema": "pr106_sidecar_runtime_decode_consumption_proof_v1",
            "proof_scope": (
                "actual_submission_inflate_py_sidecar_decode_and_apply_not_full_frame"
            ),
            "archive": archive_manifest,
            "runtime_dir": runtime_dir.as_posix(),
            "runtime_source_manifest": runtime_manifest,
            "runtime_inflate_py_sha256": sha256_hex((runtime_dir / "inflate.py").read_bytes()),
            "archive_member_name": member.name,
            "blockers": blockers,
            "source_runtime_correction_digest": source_digest,
            "mutated_runtime_correction_digest": mutated_digest,
            "runtime_semantic_digest_changed": semantic_changed,
            "runtime_corrected_latents_digest_changed": corrected_latents_changed,
            "source_packet_ir_consumed_byte_proof": source_proof,
            "mutated_packet_ir_consumed_byte_proof": mutated_proof,
            "packet_ir_consumed_byte_accounting_passed": packet_accounting_passed,
            "runtime_framing_meta_consumption_probe": framing_meta_probe,
            "runtime_section_consumption_probes": section_probes,
            "runtime_consumed_score_affecting_sections": runtime_consumed_sections,
            "runtime_consumed_score_affecting_section_identities": (
                runtime_section_identities
            ),
            "runtime_apply_order": runtime_apply_order,
            "runtime_all_score_affecting_sections_consumed": runtime_all_sections_consumed,
            "runtime_sidecar_decode_consumption_claim": decode_claim,
            "runtime_sidecar_apply_consumption_claim": apply_claim,
            "full_frame_inflate_output_parity_claim": False,
            "contest_axis_claim": False,
            "score_claim": False,
            "proof_not_score": True,
            "evidence_axis": "runtime-sidecar-decode-local-no-score",
            "device_axis_label": "local-runtime-decode-no-full-frame",
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
