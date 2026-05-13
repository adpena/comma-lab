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
    PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
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
_RUNTIME_SOURCE_CANDIDATES = (
    "inflate.py",
    "src/codec.py",
    "src/model.py",
    "src/pr101_grammar.py",
)


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
    files: list[dict[str, object]] = []
    for rel in _RUNTIME_SOURCE_CANDIDATES:
        path = runtime_dir / rel
        if not path.exists():
            continue
        if not path.is_file():
            raise ValueError(f"runtime source candidate is not a file: {path}")
        payload = path.read_bytes()
        files.append({"path": rel, "bytes": len(payload), "sha256": sha256_hex(payload)})
    if not files:
        raise ValueError(f"runtime dir has no recognized source files: {runtime_dir}")
    tree_payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "pr106_runtime_source_manifest_v1",
        "runtime_dir": runtime_dir.as_posix(),
        "files": files,
        "runtime_source_tree_sha256": sha256_hex(tree_payload),
    }


def _array_bytes(array: Any, dtype_name: str) -> bytes:
    if not hasattr(array, "astype"):
        raise TypeError(f"runtime decoder returned non-array object: {type(array)!r}")
    return array.astype(dtype_name, copy=False).tobytes()


def _tensor_sha256(tensor: Any) -> str:
    if not all(hasattr(tensor, name) for name in ("detach", "cpu", "numpy")):
        raise TypeError(f"expected tensor-like latents; got {type(tensor)!r}")
    return sha256_hex(tensor.detach().cpu().contiguous().numpy().tobytes())


def _decode_runtime_sidecar_payload(
    runtime_module: ModuleType,
    member_payload: bytes,
) -> tuple[int, bytes, Any, Any]:
    """Decode a PR106 sidecar packet through the selected runtime parser."""

    parsed = runtime_module.parse_sidecar_archive(member_payload)
    if isinstance(parsed, tuple) and len(parsed) == 2:
        pr106_bytes, sidecar_blob = parsed
        dim_arr, delta_q_arr = runtime_module.decode_sidecar_corrections(sidecar_blob)
        return PR106_SIDECAR_FORMAT_BROTLI, pr106_bytes, dim_arr, delta_q_arr
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
        else:
            raise ValueError(f"runtime returned unsupported format_id=0x{format_id:02X}")
        return int(format_id), pr106_bytes, dim_arr, delta_q_arr
    raise TypeError(
        "runtime parse_sidecar_archive returned unexpected shape: "
        f"{type(parsed)!r} {parsed!r}"
    )


def runtime_sidecar_correction_digest(
    runtime_module: ModuleType,
    member_payload: bytes,
) -> dict[str, object]:
    """Return a stable digest of corrections visible to runtime ``inflate.py``."""

    format_id, pr106_bytes, dim_arr, delta_q_arr = _decode_runtime_sidecar_payload(
        runtime_module,
        member_payload,
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


def runtime_framing_meta_consumption_probe(
    runtime_module: ModuleType,
    member_payload: bytes,
    baseline_digest: dict[str, object],
) -> dict[str, object]:
    """Probe whether format-0x02 framing metadata is runtime-visible."""

    packet = parse_pr106_sidecar_packet(member_payload)
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

    format_id, pr106_bytes, dim_arr, delta_q_arr = _decode_runtime_sidecar_payload(
        runtime_module,
        member_payload,
    )
    if device == "cuda" and not runtime_module.torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is unavailable")

    decoder_sd, latents, meta = runtime_module.parse_packed_archive(pr106_bytes)
    runtime_module.apply_sidecar_corrections(latents, dim_arr, delta_q_arr)
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
        "device": device,
        "batch_pairs": pair_batch,
        "max_pairs": max_pairs,
        "n_pairs_total": n_pairs_total,
        "n_pairs_hashed": n_pairs_hashed,
        "total_frames": total_frames,
        "total_bytes": total_bytes,
        "eval_size": [int(eval_h), int(eval_w)],
        "camera_size": [camera_h, camera_w],
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
    expected_runtime_tree_matches = (
        None
        if expected_runtime_tree_sha is None or expected_runtime_tree_sha_well_formed is False
        else runtime_tree_sha == expected_runtime_tree_sha
    )
    runtime_manifest.update(
        {
            "expected_runtime_source_tree_sha256": expected_runtime_tree_sha,
            "expected_runtime_source_tree_sha256_well_formed": (
                expected_runtime_tree_sha_well_formed
            ),
            "expected_runtime_source_tree_sha256_matches": expected_runtime_tree_matches,
        }
    )
    if expected_runtime_tree_sha_well_formed is False:
        blockers.append("expected_runtime_source_tree_sha256_malformed")
    elif expected_runtime_tree_matches is False:
        blockers.append("expected_runtime_source_tree_sha256_mismatch")
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

    runtime = load_pr106_sidecar_runtime(runtime_dir)
    source_digest = runtime_sidecar_correction_digest(runtime, member.payload)
    mutated_digest = runtime_sidecar_correction_digest(runtime, mutated_payload)
    framing_meta_probe = runtime_framing_meta_consumption_probe(
        runtime,
        member.payload,
        source_digest,
    )
    semantic_changed = (
        source_digest["combined_sha256"] != mutated_digest["combined_sha256"]
    )
    corrected_latents_changed = (
        source_digest["corrected_latents_sha256"]
        != mutated_digest["corrected_latents_sha256"]
    )
    packet_accounting_passed = (
        source_proof.get("all_payload_bytes_accounted") is True
        and mutated_proof.get("all_payload_bytes_accounted") is True
    )
    if not packet_accounting_passed:
        blockers.append("packet_ir_consumed_byte_accounting_failed")
    if not semantic_changed:
        blockers.append("runtime_semantic_digest_not_changed")
    if not corrected_latents_changed:
        blockers.append("runtime_corrected_latents_digest_not_changed")
    framing_meta_claim = framing_meta_probe.get("runtime_consumption_claim")
    if (
        source_packet.format_id == PR106_SIDECAR_FORMAT_PR101_GRAMMAR
        and framing_meta_claim is not True
    ):
        blockers.append("runtime_framing_meta_consumption_not_proven")
    decode_claim = semantic_changed and packet_accounting_passed
    apply_claim = decode_claim and corrected_latents_changed
    manifest = pr106_sidecar_mutation_manifest(
        source_packet,
        mutated_packet,
        mutation,
        source_archive_sha256=archive_sha,
        mutated_archive_sha256=sha256_hex(mutated_archive_bytes),
    )
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
            "runtime_consumed_score_affecting_sections": {
                "pr106_payload": True,
                "sidecar_payload": decode_claim,
                "framing_meta": framing_meta_claim,
            },
            "runtime_all_score_affecting_sections_consumed": (
                decode_claim
                and (
                    source_packet.format_id != PR106_SIDECAR_FORMAT_PR101_GRAMMAR
                    or framing_meta_claim is True
                )
            ),
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
