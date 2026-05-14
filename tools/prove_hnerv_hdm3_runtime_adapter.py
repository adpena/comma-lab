#!/usr/bin/env python3
"""Prove HDM3 runtime-adapter parity against a source HNeRV archive."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import brotli

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_hdm3_runtime_adapter import (  # noqa: E402
    HnervHdm3RuntimeAdapterError,
    restore_hdm3_payload_to_legacy_brotli,
)
from tac.hnerv_lowlevel_packer import (  # noqa: E402
    HnervLowlevelPackError,
    parse_ff_packed_brotli_hnerv,
    read_packed_archive_view,
    sha256_bytes,
)
from tac.hnerv_decoder_recode import parse_decoder_section_for_recode  # noqa: E402
from tac.packet_compiler.pr106_sidecar_packet import (  # noqa: E402
    PR106_SIDECAR_MAGIC,
    parse_pr106_sidecar_packet,
)
from tac.repo_io import repo_relative, sha256_file, write_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402

SCHEMA_VERSION = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--candidate-archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        help=(
            "Optional submission runtime directory containing inflate.py and src/. "
            "When supplied, the proof also imports the actual runtime parser and "
            "decodes the source/candidate packet through that code path."
        ),
    )
    return parser.parse_args(argv)


def build_proof(
    *,
    source_archive: Path,
    candidate_archive: Path,
    output_dir: Path,
    runtime_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_view = read_packed_archive_view(source_archive)
    candidate_view = read_packed_archive_view(candidate_archive)
    source = source_view.archive
    candidate = candidate_view.archive
    source_packed = source_view.packed
    candidate_packed = candidate_view.packed
    _source_parsed, source_raw, source_decoder_codec = parse_decoder_section_for_recode(
        source_packed.decoder_packed_brotli
    )
    _candidate_parsed, candidate_raw, candidate_decoder_codec = parse_decoder_section_for_recode(
        candidate_packed.decoder_packed_brotli
    )
    restored_payload, adapter_proof = restore_hdm3_payload_to_legacy_brotli(
        candidate.payload,
        require_hdm3=True,
    )
    restored_packed = _packed_from_member_payload(restored_payload)
    restored_raw = brotli.decompress(restored_packed.decoder_packed_brotli)
    restored_payload_path = output_dir / "candidate_hdm3_restored_legacy_payload.bin"
    restored_payload_path.write_bytes(restored_payload)

    source_raw_sha = sha256_bytes(source_raw)
    candidate_raw_sha = sha256_bytes(candidate_raw)
    restored_raw_sha = sha256_bytes(restored_raw)
    source_payload_sha = sha256_bytes(source.payload)
    restored_payload_sha = sha256_bytes(restored_payload)
    source_decoder_section_sha = sha256_bytes(source_packed.decoder_packed_brotli)
    restored_decoder_section_sha = sha256_bytes(restored_packed.decoder_packed_brotli)
    source_latents_sha = sha256_bytes(source_packed.latents_and_sidecar_brotli)
    candidate_latents_sha = sha256_bytes(candidate_packed.latents_and_sidecar_brotli)
    restored_latents_sha = sha256_bytes(restored_packed.latents_and_sidecar_brotli)
    candidate_decoder_magic = candidate_packed.decoder_packed_brotli[:4]
    restored_payload_matches_source = restored_payload_sha == source_payload_sha
    restored_decoder_section_matches_source = restored_decoder_section_sha == source_decoder_section_sha
    inflate_output_parity_proven_by_payload_identity = (
        restored_payload_matches_source
        and restored_decoder_section_matches_source
        and restored_raw_sha == source_raw_sha
        and candidate_latents_sha == source_latents_sha
        and restored_latents_sha == source_latents_sha
    )
    inflate_output_parity_proven_by_lossless_decoder_equivalence = (
        candidate_raw_sha == source_raw_sha
        and restored_raw_sha == source_raw_sha
        and candidate_latents_sha == source_latents_sha
        and restored_latents_sha == source_latents_sha
        and candidate_decoder_magic in (b"HDM3", b"HDM4", b"HDM6", b"HDM7")
    )
    submission_runtime_proof = _submission_runtime_parse_proof(
        runtime_dir=runtime_dir,
        source_payload=source.payload,
        candidate_payload=candidate.payload,
    )
    submission_runtime_candidate_parse_claim = (
        submission_runtime_proof.get("candidate_parse_claim") is True
    )
    submission_runtime_equivalence_claim = (
        submission_runtime_proof.get("candidate_decoder_state_matches_source") is True
        and submission_runtime_proof.get("candidate_latents_match_source") is True
        and submission_runtime_proof.get("candidate_meta_matches_source") is True
    )
    public_runtime_ready = bool(
        inflate_output_parity_proven_by_payload_identity
        or (
            inflate_output_parity_proven_by_lossless_decoder_equivalence
            and submission_runtime_candidate_parse_claim
            and submission_runtime_equivalence_claim
        )
    )
    remaining_dispatch_blockers = _dispatch_blockers(
        inflate_output_parity_proven_by_payload_identity
    )
    if (
        inflate_output_parity_proven_by_lossless_decoder_equivalence
        and not inflate_output_parity_proven_by_payload_identity
        and not submission_runtime_candidate_parse_claim
    ):
        remaining_dispatch_blockers.insert(0, "submission_runtime_candidate_parse_missing")
    adapter_proof = {
        **adapter_proof,
        "exact_source_payload_identity_proven_by_archive_proof": (
            inflate_output_parity_proven_by_payload_identity
        ),
        "lossless_decoder_equivalence_proven_by_archive_proof": (
            inflate_output_parity_proven_by_lossless_decoder_equivalence
        ),
        "submission_runtime_candidate_parse_claim": submission_runtime_candidate_parse_claim,
        "submission_runtime_equivalence_claim": submission_runtime_equivalence_claim,
        "remaining_dispatch_blockers": remaining_dispatch_blockers,
    }
    adapter_files = [
        REPO_ROOT / "src/tac/hnerv_hdm3_runtime_adapter.py",
        REPO_ROOT / "experiments/public_runtime_adapters/pr106_belt_and_suspenders_adapter/hdm3_normalize.py",
        REPO_ROOT / "experiments/public_runtime_adapters/pr106_belt_and_suspenders_adapter/inflate.sh",
    ]
    proof: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": "hnerv_hdm3_runtime_adapter_archive_parity_v1",
        "score_claim": False,
        "dispatch_attempted": False,
        "source_archive_path": repo_relative(source_archive, REPO_ROOT),
        "source_archive_sha256": source.archive_sha256,
        "source_archive_bytes": source.archive_bytes,
        "source_member_name": source.member_name,
        "source_payload_kind": source_view.payload_kind,
        "source_payload_sha256": source_payload_sha,
        "source_payload_bytes": len(source.payload),
        "source_decoder_section_sha256": source_decoder_section_sha,
        "source_decoder_section_bytes": len(source_packed.decoder_packed_brotli),
        "source_decoder_section_codec": source_decoder_codec,
        "candidate_archive_path": repo_relative(candidate_archive, REPO_ROOT),
        "candidate_archive_sha256": candidate.archive_sha256,
        "candidate_archive_bytes": candidate.archive_bytes,
        "candidate_member_name": candidate.member_name,
        "candidate_payload_kind": candidate_view.payload_kind,
        "candidate_decoder_section_sha256": sha256_bytes(candidate_packed.decoder_packed_brotli),
        "candidate_decoder_section_bytes": len(candidate_packed.decoder_packed_brotli),
        "candidate_decoder_section_is_hdm3": candidate_packed.decoder_packed_brotli.startswith(b"HDM3"),
        "candidate_decoder_section_is_hdm4": candidate_packed.decoder_packed_brotli.startswith(b"HDM4"),
        "candidate_decoder_section_is_hdm6": candidate_packed.decoder_packed_brotli.startswith(b"HDM6"),
        "candidate_decoder_section_is_hdm7": candidate_packed.decoder_packed_brotli.startswith(b"HDM7"),
        "candidate_decoder_section_is_hdm": candidate_decoder_magic
        in (b"HDM3", b"HDM4", b"HDM6", b"HDM7"),
        "candidate_decoder_section_magic": candidate_decoder_magic.decode("ascii", errors="replace"),
        "candidate_decoder_section_codec": candidate_decoder_codec,
        "source_decoder_raw_sha256": source_raw_sha,
        "candidate_decoder_raw_sha256": candidate_raw_sha,
        "restored_decoder_raw_sha256": restored_raw_sha,
        "candidate_decoder_raw_matches_source": candidate_raw_sha == source_raw_sha,
        "decoder_raw_matches_source": restored_raw_sha == source_raw_sha,
        "restored_decoder_section_sha256": restored_decoder_section_sha,
        "restored_decoder_section_bytes": len(restored_packed.decoder_packed_brotli),
        "restored_decoder_section_matches_source": restored_decoder_section_matches_source,
        "source_latents_and_sidecar_sha256": source_latents_sha,
        "candidate_latents_and_sidecar_sha256": candidate_latents_sha,
        "restored_latents_and_sidecar_sha256": restored_latents_sha,
        "latents_and_sidecar_match_source": candidate_latents_sha == source_latents_sha
        and restored_latents_sha == source_latents_sha,
        "restored_payload_path": repo_relative(restored_payload_path, REPO_ROOT),
        "restored_payload_sha256": restored_payload_sha,
        "restored_payload_bytes": restored_payload_path.stat().st_size,
        "restored_payload_matches_source": restored_payload_matches_source,
        "adapter_proof": adapter_proof,
        "runtime_adapter_files": [
            {
                "path": repo_relative(path, REPO_ROOT),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in adapter_files
        ],
        "submission_runtime_parse_proof": submission_runtime_proof,
        "submission_runtime_parse_exercised": (
            submission_runtime_proof.get("runtime_parse_exercised") is True
        ),
        "submission_runtime_candidate_parse_claim": submission_runtime_candidate_parse_claim,
        "submission_runtime_equivalence_claim": submission_runtime_equivalence_claim,
        "runtime_adapter_integrated": True,
        "public_pr106_dependency_root_unchanged": True,
        "inflate_output_parity_proven_by_payload_identity": inflate_output_parity_proven_by_payload_identity,
        "inflate_output_parity_proven_by_lossless_decoder_equivalence": (
            inflate_output_parity_proven_by_lossless_decoder_equivalence
        ),
        "full_frame_inflate_output_parity_claim": False,
        "ready_for_public_runtime_inflate": public_runtime_ready,
        "ready_for_exact_eval_dispatch": False,
        "remaining_dispatch_blockers": remaining_dispatch_blockers,
    }
    write_json(output_dir / "runtime_adapter_proof.json", proof)
    return proof


def _packed_from_member_payload(payload: bytes):
    if payload and payload[0] == PR106_SIDECAR_MAGIC:
        packet = parse_pr106_sidecar_packet(payload)
        return parse_ff_packed_brotli_hnerv(packet.pr106_bytes)
    return parse_ff_packed_brotli_hnerv(payload)


def _submission_runtime_parse_proof(
    *,
    runtime_dir: Path | None,
    source_payload: bytes,
    candidate_payload: bytes,
) -> dict[str, Any]:
    if runtime_dir is None:
        return {
            "runtime_parse_exercised": False,
            "candidate_parse_claim": False,
            "blockers": ["submission_runtime_dir_not_supplied"],
        }
    try:
        runtime = _load_submission_inflate_module(runtime_dir)
        source_digest = _runtime_payload_digest(runtime, source_payload)
        candidate_digest = _runtime_payload_digest(runtime, candidate_payload)
    except Exception as exc:
        return {
            "runtime_parse_exercised": False,
            "runtime_dir": repo_relative(runtime_dir, REPO_ROOT),
            "candidate_parse_claim": False,
            "blockers": [f"submission_runtime_parse_failed:{type(exc).__name__}:{exc}"],
        }

    decoder_matches = (
        source_digest.get("decoder_state_sha256")
        == candidate_digest.get("decoder_state_sha256")
    )
    latents_match = source_digest.get("latents_sha256") == candidate_digest.get("latents_sha256")
    meta_matches = source_digest.get("meta") == candidate_digest.get("meta")
    blockers: list[str] = []
    if not decoder_matches:
        blockers.append("submission_runtime_decoder_state_mismatch")
    if not latents_match:
        blockers.append("submission_runtime_latents_mismatch")
    if not meta_matches:
        blockers.append("submission_runtime_meta_mismatch")
    return {
        "runtime_parse_exercised": True,
        "runtime_dir": repo_relative(runtime_dir, REPO_ROOT),
        "source": source_digest,
        "candidate": candidate_digest,
        "candidate_decoder_state_matches_source": decoder_matches,
        "candidate_latents_match_source": latents_match,
        "candidate_meta_matches_source": meta_matches,
        "candidate_parse_claim": not blockers,
        "blockers": blockers,
    }


def _load_submission_inflate_module(runtime_dir: Path):
    runtime_dir = runtime_dir.resolve()
    inflate_py = runtime_dir / "inflate.py"
    if not inflate_py.is_file():
        raise FileNotFoundError(f"{inflate_py} does not exist")
    module_name = f"_pact_hdm_runtime_{hashlib.sha256(str(runtime_dir).encode()).hexdigest()[:12]}"
    saved_path = list(sys.path)
    runtime_module_names = ("codec", "model", "pr101_grammar")
    saved_modules = {name: sys.modules.get(name) for name in runtime_module_names}
    for name in runtime_module_names:
        sys.modules.pop(name, None)
    try:
        sys.path.insert(0, str(runtime_dir / "src"))
        sys.path.insert(0, str(runtime_dir))
        spec = importlib.util.spec_from_file_location(module_name, inflate_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load runtime module from {inflate_py}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved_path
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _runtime_payload_digest(runtime: Any, payload: bytes) -> dict[str, Any]:
    if payload and payload[0] == PR106_SIDECAR_MAGIC:
        format_id, pr106_bytes, _sidecar_blob, _framing_meta = runtime.parse_sidecar_archive(payload)
        format_label = f"0x{int(format_id):02X}"
    else:
        pr106_bytes = payload
        format_label = None
    decoder_sd, latents, meta = runtime.parse_packed_archive(pr106_bytes)
    return {
        "format_id": format_label,
        "pr106_payload_sha256": sha256_bytes(pr106_bytes),
        "decoder_state_sha256": _tensor_mapping_digest(decoder_sd),
        "latents_sha256": _tensor_digest(latents),
        "meta": json.loads(json.dumps(meta, sort_keys=True)),
    }


def _tensor_mapping_digest(mapping: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(mapping):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(_tensor_digest(mapping[name]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _tensor_digest(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode())
    digest.update(b"\0")
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _dispatch_blockers(exact_inflate_output_parity_proven: bool) -> list[str]:
    blockers = [
        "strict_pre_submission_compliance_json_missing",
        "lane_dispatch_claim_missing",
        "exact_cuda_auth_eval_missing",
    ]
    if not exact_inflate_output_parity_proven:
        blockers.insert(0, "exact_inflate_output_parity_missing")
    return blockers


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    try:
        proof = build_proof(
            source_archive=args.source_archive,
            candidate_archive=args.candidate_archive,
            output_dir=args.output_dir,
            runtime_dir=args.runtime_dir,
        )
    except (brotli.error, HnervHdm3RuntimeAdapterError, HnervLowlevelPackError) as exc:
        print(f"FATAL: HDM3 runtime adapter proof failed: {exc}", file=sys.stderr)
        return 2
    proof = attach_tool_run_manifest(
        proof,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=[args.source_archive, args.candidate_archive],
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    if args.json_out is not None:
        write_json(args.json_out, proof)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
