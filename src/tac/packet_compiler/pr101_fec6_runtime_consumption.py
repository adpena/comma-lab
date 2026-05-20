# SPDX-License-Identifier: MIT
"""Runtime-consumption proof for the PR101/FEC6 packet.

This proof is intentionally narrower than full-frame inflate parity or scoring.
It imports the candidate submission runtime, routes the archive member through
that runtime's parser, decodes the PR101 source payload with the runtime codec,
and runs mutation probes that must either change runtime-visible decoded state
or fail closed.  A passing proof is only byte-consumption authority for the
PacketIR queue; it is not score, promotion, or dispatch authority.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from tac.packet_compiler.pr101_fec6_packetir import (
    FEC6_FIXED_K16_CODE_BITS,
    PR101_FEC6_DEFAULT_MEMBER_NAME,
    canonical_expected_sha256,
    parse_pr101_fec6_packetir_member,
    read_single_stored_fec6_member_archive,
    sha256_hex,
)
from tac.packet_compiler.pr101_fec6_source_anatomy import (
    PR101_DECODER_BLOB_LEN,
    PR101_LATENT_BLOB_LEN,
)
from tac.repo_io import repo_relative, sha256_file

PR101_FEC6_RUNTIME_CONSUMPTION_PROOF_FAMILY = (
    "pr101_fec6_runtime_consumption_proof_v1"
)
PR101_FEC6_RUNTIME_CONSUMPTION_SCHEMA_VERSION = "deterministic_no_op_proof.v1"

_RUNTIME_MODULE_NAMES = ("codec", "codec_sidecar", "frame_selector", "model")


class PR101FEC6RuntimeConsumptionError(ValueError):
    """Raised when PR101/FEC6 runtime-consumption inputs are malformed."""


def prove_pr101_fec6_runtime_consumption(
    *,
    archive_path: str | Path,
    runtime_dir: str | Path,
    expected_archive_sha256: str | None = None,
    expected_member_name: str = PR101_FEC6_DEFAULT_MEMBER_NAME,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a fail-closed runtime-consumption proof for PR101/FEC6 bytes."""

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    archive = Path(archive_path)
    runtime = Path(runtime_dir)
    archive_bytes = archive.read_bytes()
    archive_sha = sha256_hex(archive_bytes)
    expected_sha, expected_sha_well_formed = canonical_expected_sha256(
        expected_archive_sha256
    )
    member = read_single_stored_fec6_member_archive(
        archive_bytes,
        expected_member_name=expected_member_name,
    )
    packet = parse_pr101_fec6_packetir_member(member.payload)
    blockers: list[str] = []
    if expected_sha_well_formed is False:
        blockers.append("expected_archive_sha256_malformed")
    if expected_sha_well_formed is True and archive_sha != expected_sha:
        blockers.append("expected_archive_sha256_mismatch")

    runtime_module = _load_runtime_inflate(runtime)
    runtime_parse = _runtime_parse_member(runtime_module, member.payload)
    source_payload, selector_kind, selector_codes, selector_specs = runtime_parse[
        "result"
    ]
    source_digest = sha256_hex(source_payload)
    if source_payload != packet.source_pr101_payload:
        blockers.append("runtime_source_payload_mismatch")
    if selector_kind != "compact":
        blockers.append("runtime_selector_kind_not_compact")
    if tuple(selector_codes) != packet.selector_codes:
        blockers.append("runtime_selector_codes_mismatch")
    if len(selector_specs) != len(FEC6_FIXED_K16_CODE_BITS):
        blockers.append("runtime_selector_specs_palette_size_mismatch")

    baseline_decode = _runtime_decode_source(runtime_module, source_payload)
    if baseline_decode["blocker"] is not None:
        blockers.append(baseline_decode["blocker"])

    source_section_probes = _probe_pr101_source_sections(
        runtime_module,
        member.payload,
        source_offset=8,
        source_payload_len=len(source_payload),
        baseline_decode_digest=baseline_decode["decode_digest"],
    )
    probes = [
        _probe_bad_magic(runtime_module, member.payload),
        _probe_trailing_byte(runtime_module, member.payload),
        _probe_source_payload_mutation(
            runtime_module,
            member.payload,
            source_offset=8,
            baseline_decode_digest=baseline_decode["decode_digest"],
        ),
        _probe_selector_code_mutation(
            runtime_module,
            member.payload,
            packet_selector_codes=packet.selector_codes,
        ),
        *source_section_probes,
    ]
    for probe in probes:
        if probe["passed"] is not True:
            blockers.append(f"no_op_detector_failed:{probe['probe_id']}")

    no_op_detector_passed = not blockers
    consumed_ranges = _consumed_byte_ranges_from_probes(probes)
    consumed_section_names = sorted({row["section_name"] for row in consumed_ranges})
    proof_path = None
    return {
        "schema_version": PR101_FEC6_RUNTIME_CONSUMPTION_SCHEMA_VERSION,
        "schema": PR101_FEC6_RUNTIME_CONSUMPTION_SCHEMA_VERSION,
        "proof_family": PR101_FEC6_RUNTIME_CONSUMPTION_PROOF_FAMILY,
        "proof_scope": (
            "runtime_parser_and_codec_byte_consumption_not_full_frame_not_score"
        ),
        "archive_path": repo_relative(archive, repo),
        "archive_sha256": archive_sha,
        "archive_bytes": len(archive_bytes),
        "expected_archive_sha256": expected_sha,
        "expected_archive_sha256_well_formed": expected_sha_well_formed,
        "expected_archive_sha256_matches": (
            None
            if expected_sha is None or expected_sha_well_formed is False
            else archive_sha == expected_sha
        ),
        "runtime_dir": repo_relative(runtime, repo),
        "runtime_inflate_py_sha256": sha256_file(runtime / "inflate.py"),
        "runtime_codec_py_sha256": sha256_file(runtime / "src" / "codec.py"),
        "runtime_frame_selector_py_sha256": sha256_file(
            runtime / "src" / "frame_selector.py"
        ),
        "member_name": member.name,
        "member_payload_bytes": len(member.payload),
        "member_payload_sha256": sha256_hex(member.payload),
        "runtime_bytes_consumed": len(member.payload),
        "consumed_section_names": consumed_section_names,
        "consumed_byte_ranges": consumed_ranges,
        "runtime_consumption_proof_path": proof_path,
        "runtime_consumption_proof_source": (
            "submission_dir/inflate.py::parse_pr101_frame_selector_archive + "
            "submission_dir/src/codec.py::parse_archive + deterministic "
            "source/selector mutation probes"
        ),
        "runtime_parse": {
            "source_len": packet.source_len,
            "selector_len": packet.selector_len,
            "selector_kind": selector_kind,
            "selector_code_count": len(selector_codes),
            "selector_specs_count": len(selector_specs),
            "source_payload_sha256": source_digest,
            "selector_codes_sha256": sha256_hex(
                bytes(int(code) for code in selector_codes)
            ),
            "source_payload_matches_packetir": source_payload
            == packet.source_pr101_payload,
            "selector_codes_match_packetir": tuple(selector_codes)
            == packet.selector_codes,
        },
        "baseline_runtime_decode": {
            key: value
            for key, value in baseline_decode.items()
            if key not in {"blocker", "decode_digest"}
        },
        "baseline_runtime_decode_digest": baseline_decode["decode_digest"],
        "mutation_probes": probes,
        "section_mutation_probe_count": len(source_section_probes),
        "no_op_detector_passed": no_op_detector_passed,
        "runtime_consumption_claim": no_op_detector_passed,
        "full_frame_inflate_output_parity_claim": False,
        "contest_axis_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_operator_probe": False,
        "ready_for_provider_dispatch": False,
        "dispatch_attempted": False,
        "blockers": blockers,
        "required_next_proof": (
            "full-frame source-vs-candidate inflate output parity and paired "
            "contest CPU/CUDA exact eval with archive/runtime custody before "
            "score or promotion language"
        ),
    }


def dumps_pr101_fec6_runtime_consumption_proof(proof: dict[str, Any]) -> str:
    """Return canonical JSON for a PR101/FEC6 runtime-consumption proof."""

    from tac.repo_io import json_text

    return json_text(proof)


def _load_runtime_inflate(runtime_dir: Path) -> ModuleType:
    inflate_py = runtime_dir / "inflate.py"
    src_dir = runtime_dir / "src"
    if not inflate_py.is_file():
        raise PR101FEC6RuntimeConsumptionError(
            f"runtime inflate.py not found: {inflate_py}"
        )
    if not src_dir.is_dir():
        raise PR101FEC6RuntimeConsumptionError(f"runtime src dir not found: {src_dir}")
    module_name = f"_pr101_fec6_runtime_inflate_{sha256_hex(str(inflate_py.resolve()).encode())[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, inflate_py)
    if spec is None or spec.loader is None:
        raise PR101FEC6RuntimeConsumptionError(f"cannot load runtime: {inflate_py}")
    module = importlib.util.module_from_spec(spec)
    with _isolated_runtime_import(src_dir):
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    for attr in ("parse_pr101_frame_selector_archive", "parse_archive"):
        if not callable(getattr(module, attr, None)):
            raise PR101FEC6RuntimeConsumptionError(
                f"runtime missing callable {attr}: {inflate_py}"
            )
    return module


@contextmanager
def _isolated_runtime_import(src_dir: Path) -> Iterator[None]:
    old_path = list(sys.path)
    saved_modules = {
        name: sys.modules[name] for name in _RUNTIME_MODULE_NAMES if name in sys.modules
    }
    for name in _RUNTIME_MODULE_NAMES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(src_dir))
    try:
        yield
    finally:
        sys.path[:] = old_path
        for name in _RUNTIME_MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def _runtime_parse_member(
    runtime_module: ModuleType,
    member_payload: bytes,
) -> dict[str, Any]:
    try:
        result = runtime_module.parse_pr101_frame_selector_archive(member_payload)
    except Exception as exc:  # pragma: no cover - exact exception is runtime-owned
        raise PR101FEC6RuntimeConsumptionError(
            f"runtime parser rejected source member: {exc}"
        ) from exc
    if not isinstance(result, tuple) or len(result) != 4:
        raise PR101FEC6RuntimeConsumptionError(
            "runtime parser returned unexpected shape"
        )
    return {"result": result}


def _runtime_decode_source(runtime_module: ModuleType, source_payload: bytes) -> dict[str, Any]:
    try:
        decoder_sd, latents, meta = runtime_module.parse_archive(source_payload)
    except Exception as exc:  # pragma: no cover - exact exception is runtime-owned
        return {
            "blocker": "runtime_parse_archive_rejected_source_payload",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "decode_digest": None,
        }
    decoder_digest = _state_dict_digest(decoder_sd)
    latents_digest = _tensor_digest(latents)
    decode_digest = sha256_hex(
        (
            decoder_digest
            + "|"
            + latents_digest
            + "|"
            + repr(sorted(meta.items()))
        ).encode()
    )
    return {
        "blocker": None,
        "decoder_state_digest": decoder_digest,
        "decoder_tensor_count": len(decoder_sd),
        "latents_sha256": latents_digest,
        "latents_shape": list(latents.shape),
        "meta": _jsonish(meta),
        "decode_digest": decode_digest,
    }


def _state_dict_digest(state_dict: Any) -> str:
    digest_parts: list[str] = []
    for name in sorted(state_dict):
        tensor = state_dict[name]
        digest_parts.append(
            f"{name}:{list(tensor.shape)}:{tensor.dtype}:{_tensor_digest(tensor)}"
        )
    return sha256_hex("\n".join(digest_parts).encode())


def _tensor_digest(tensor: Any) -> str:
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().contiguous().numpy()
    elif isinstance(tensor, np.ndarray):
        arr = np.ascontiguousarray(tensor)
    else:
        arr = np.ascontiguousarray(np.asarray(tensor))
    return sha256_hex(arr.tobytes())


def _jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonish(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonish(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _probe_bad_magic(runtime_module: ModuleType, member_payload: bytes) -> dict[str, Any]:
    mutated = b"XP11" + member_payload[4:]
    return _expect_runtime_parser_rejects(
        runtime_module,
        mutated,
        probe_id="fp11_magic_mutation_rejected",
        mutation_kind="fp11_magic_first_byte_F_to_X",
    )


def _probe_trailing_byte(runtime_module: ModuleType, member_payload: bytes) -> dict[str, Any]:
    return _expect_runtime_parser_rejects(
        runtime_module,
        member_payload + b"\x00",
        probe_id="trailing_member_byte_rejected",
        mutation_kind="append_trailing_zero_byte_after_selector_payload",
    )


def _expect_runtime_parser_rejects(
    runtime_module: ModuleType,
    mutated_payload: bytes,
    *,
    probe_id: str,
    mutation_kind: str,
) -> dict[str, Any]:
    try:
        runtime_module.parse_pr101_frame_selector_archive(mutated_payload)
    except Exception as exc:
        return {
            "probe_id": probe_id,
            "mutation_kind": mutation_kind,
            "mutated_member_sha256": sha256_hex(mutated_payload),
            "passed": True,
            "runtime_rejected": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "probe_id": probe_id,
        "mutation_kind": mutation_kind,
        "mutated_member_sha256": sha256_hex(mutated_payload),
        "passed": False,
        "runtime_rejected": False,
        "blocker": "runtime_accepted_invalid_member_payload",
    }


def _probe_source_payload_mutation(
    runtime_module: ModuleType,
    member_payload: bytes,
    *,
    source_offset: int,
    baseline_decode_digest: str | None,
) -> dict[str, Any]:
    mutated = bytearray(member_payload)
    mutated[source_offset] ^= 0x01
    mutated_payload = bytes(mutated)
    try:
        source_payload, _kind, _codes, _specs = runtime_module.parse_pr101_frame_selector_archive(
            mutated_payload
        )
    except Exception as exc:
        return {
            "probe_id": "source_payload_byte_mutation_runtime_visible",
            "mutation_kind": "source_payload_first_byte_xor_0x01",
            "section_name": "source_pr101_payload",
            "section_range": [source_offset, _source_end(member_payload)],
            "offset": source_offset,
            "mutated_member_sha256": sha256_hex(mutated_payload),
            "passed": True,
            "runtime_rejected": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    mutated_decode = _runtime_decode_source(runtime_module, source_payload)
    if mutated_decode["blocker"] is not None:
        return {
            "probe_id": "source_payload_byte_mutation_runtime_visible",
            "mutation_kind": "source_payload_first_byte_xor_0x01",
            "section_name": "source_pr101_payload",
            "section_range": [source_offset, _source_end(member_payload)],
            "offset": source_offset,
            "mutated_member_sha256": sha256_hex(mutated_payload),
            "passed": True,
            "runtime_rejected": True,
            "error_type": mutated_decode.get("error_type"),
            "error": mutated_decode.get("error"),
        }
    changed = (
        baseline_decode_digest is not None
        and mutated_decode["decode_digest"] != baseline_decode_digest
    )
    return {
        "probe_id": "source_payload_byte_mutation_runtime_visible",
        "mutation_kind": "source_payload_first_byte_xor_0x01",
        "section_name": "source_pr101_payload",
        "section_range": [source_offset, _source_end(member_payload)],
        "offset": source_offset,
        "mutated_member_sha256": sha256_hex(mutated_payload),
        "passed": changed,
        "runtime_rejected": False,
        "baseline_decode_digest": baseline_decode_digest,
        "mutated_decode_digest": mutated_decode["decode_digest"],
        "runtime_decode_changed": changed,
    }


def _probe_pr101_source_sections(
    runtime_module: ModuleType,
    member_payload: bytes,
    *,
    source_offset: int,
    source_payload_len: int,
    baseline_decode_digest: str | None,
) -> list[dict[str, Any]]:
    """Probe PR101 internal latent and sidecar sections when the source is real-sized."""

    latent_start = PR101_DECODER_BLOB_LEN
    latent_end = latent_start + PR101_LATENT_BLOB_LEN
    section_specs = (
        ("pr101_latent_blob", latent_start, latent_end),
        ("pr101_sidecar_blob", latent_end, source_payload_len),
    )
    probes: list[dict[str, Any]] = []
    for section_name, start, end in section_specs:
        if start >= end or end > source_payload_len:
            continue
        probes.append(
            _probe_source_section_mutation(
                runtime_module,
                member_payload,
                section_name=section_name,
                member_start=source_offset + start,
                member_end=source_offset + end,
                baseline_decode_digest=baseline_decode_digest,
            )
        )
    return probes


def _probe_source_section_mutation(
    runtime_module: ModuleType,
    member_payload: bytes,
    *,
    section_name: str,
    member_start: int,
    member_end: int,
    baseline_decode_digest: str | None,
) -> dict[str, Any]:
    mutated = bytearray(member_payload)
    probe_offset = member_start
    mutated[probe_offset] ^= 0x01
    mutated_payload = bytes(mutated)
    try:
        source_payload, _kind, _codes, _specs = runtime_module.parse_pr101_frame_selector_archive(
            mutated_payload
        )
    except Exception as exc:
        return {
            "probe_id": f"{section_name}_byte_mutation_runtime_visible",
            "mutation_kind": f"{section_name}_first_byte_xor_0x01",
            "section_name": section_name,
            "section_range": [member_start, member_end],
            "offset": probe_offset,
            "mutated_member_sha256": sha256_hex(mutated_payload),
            "passed": True,
            "runtime_rejected": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    mutated_decode = _runtime_decode_source(runtime_module, source_payload)
    if mutated_decode["blocker"] is not None:
        return {
            "probe_id": f"{section_name}_byte_mutation_runtime_visible",
            "mutation_kind": f"{section_name}_first_byte_xor_0x01",
            "section_name": section_name,
            "section_range": [member_start, member_end],
            "offset": probe_offset,
            "mutated_member_sha256": sha256_hex(mutated_payload),
            "passed": True,
            "runtime_rejected": True,
            "error_type": mutated_decode.get("error_type"),
            "error": mutated_decode.get("error"),
        }
    changed = (
        baseline_decode_digest is not None
        and mutated_decode["decode_digest"] != baseline_decode_digest
    )
    return {
        "probe_id": f"{section_name}_byte_mutation_runtime_visible",
        "mutation_kind": f"{section_name}_first_byte_xor_0x01",
        "section_name": section_name,
        "section_range": [member_start, member_end],
        "offset": probe_offset,
        "mutated_member_sha256": sha256_hex(mutated_payload),
        "passed": changed,
        "runtime_rejected": False,
        "baseline_decode_digest": baseline_decode_digest,
        "mutated_decode_digest": mutated_decode["decode_digest"],
        "runtime_decode_changed": changed,
    }


def _source_end(member_payload: bytes) -> int:
    return 8 + int.from_bytes(member_payload[4:8], "little")


def _consumed_byte_ranges_from_probes(probes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    for probe in probes:
        if probe.get("passed") is not True:
            continue
        section_name = probe.get("section_name")
        section_range = probe.get("section_range")
        if not isinstance(section_name, str):
            continue
        if (
            not isinstance(section_range, list)
            or len(section_range) != 2
            or not all(isinstance(value, int) for value in section_range)
        ):
            continue
        ranges.append(
            {
                "section_name": section_name,
                "range": section_range,
                "probe_offset": probe.get("offset"),
                "probe_id": probe.get("probe_id"),
            }
        )
    return ranges


def _probe_selector_code_mutation(
    runtime_module: ModuleType,
    member_payload: bytes,
    *,
    packet_selector_codes: tuple[int, ...],
) -> dict[str, Any]:
    mutation = _same_bit_length_selector_mutation(packet_selector_codes)
    source_len = int.from_bytes(member_payload[4:8], "little")
    selector_len_offset = 8 + source_len
    selector_start = selector_len_offset + 2
    selector_end = selector_start + int.from_bytes(
        member_payload[selector_len_offset:selector_start], "little"
    )
    mutated_selector = _encode_fec6_selector(mutation["mutated_codes"])
    if len(mutated_selector) != selector_end - selector_start:
        return {
            "probe_id": "selector_code_mutation_runtime_visible",
            "mutation_kind": "same_bit_length_selector_code_substitution",
            "section_name": "selector_fec6_payload",
            "section_range": [selector_start, selector_end],
            "passed": False,
            "blocker": "mutated_selector_length_changed",
        }
    mutated_payload = (
        member_payload[:selector_start] + mutated_selector + member_payload[selector_end:]
    )
    try:
        _source_payload, selector_kind, selector_codes, _selector_specs = (
            runtime_module.parse_pr101_frame_selector_archive(mutated_payload)
        )
    except Exception as exc:
        return {
            "probe_id": "selector_code_mutation_runtime_visible",
            "mutation_kind": "same_bit_length_selector_code_substitution",
            "section_name": "selector_fec6_payload",
            "section_range": [selector_start, selector_end],
            "pair_index": mutation["pair_index"],
            "old_code": mutation["old_code"],
            "new_code": mutation["new_code"],
            "mutated_member_sha256": sha256_hex(mutated_payload),
            "passed": True,
            "runtime_rejected": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    changed = list(selector_codes) != list(packet_selector_codes)
    targeted_changed = selector_codes[mutation["pair_index"]] == mutation["new_code"]
    return {
        "probe_id": "selector_code_mutation_runtime_visible",
        "mutation_kind": "same_bit_length_selector_code_substitution",
        "section_name": "selector_fec6_payload",
        "section_range": [selector_start, selector_end],
        "pair_index": mutation["pair_index"],
        "old_code": mutation["old_code"],
        "new_code": mutation["new_code"],
        "mutated_member_sha256": sha256_hex(mutated_payload),
        "passed": bool(selector_kind == "compact" and changed and targeted_changed),
        "runtime_rejected": False,
        "selector_kind": selector_kind,
        "runtime_selector_codes_changed": changed,
        "targeted_code_changed": targeted_changed,
    }


def _same_bit_length_selector_mutation(codes: tuple[int, ...]) -> dict[str, Any]:
    codes_by_len: dict[int, list[int]] = {}
    for code, bits in enumerate(FEC6_FIXED_K16_CODE_BITS):
        codes_by_len.setdefault(len(bits), []).append(code)
    for pair_index, old_code in enumerate(codes):
        peers = [code for code in codes_by_len[len(FEC6_FIXED_K16_CODE_BITS[old_code])] if code != old_code]
        if not peers:
            continue
        mutated = list(codes)
        mutated[pair_index] = peers[0]
        return {
            "pair_index": pair_index,
            "old_code": old_code,
            "new_code": peers[0],
            "mutated_codes": mutated,
        }
    raise PR101FEC6RuntimeConsumptionError(
        "could not find same-bit-length selector mutation"
    )


def _encode_fec6_selector(codes: list[int]) -> bytes:
    bits = "".join(FEC6_FIXED_K16_CODE_BITS[code] for code in codes)
    pad = (-len(bits)) % 8
    padded = bits + ("0" * pad)
    payload = bytes(
        int(padded[index : index + 8], 2) for index in range(0, len(padded), 8)
    )
    return b"FEC6" + len(codes).to_bytes(2, "little") + payload


__all__ = [
    "PR101_FEC6_RUNTIME_CONSUMPTION_PROOF_FAMILY",
    "PR101_FEC6_RUNTIME_CONSUMPTION_SCHEMA_VERSION",
    "PR101FEC6RuntimeConsumptionError",
    "dumps_pr101_fec6_runtime_consumption_proof",
    "prove_pr101_fec6_runtime_consumption",
]
