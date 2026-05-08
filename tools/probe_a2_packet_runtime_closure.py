#!/usr/bin/env python3
"""Probe A2 PR101 packet-local runtime closure without running scorers.

This is a CPU-prep guard for packets produced by
``tools/build_a2_sensitivity_weighted_pr101_packet.py``. It imports only the
packet-local runtime, parses the stock source archive and the A2 candidate
archive through that runtime, and verifies the decoded state can be loaded into
the packet-local model. It does not inflate raw frames, run PoseNet/SegNet,
dispatch remote work, or claim a score.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import sys
import zipfile
from pathlib import Path
from typing import Any

import torch

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, repo_relative, sha256_file, write_json  # noqa: E402

TOOL_NAME = "tools/probe_a2_packet_runtime_closure.py"
SCHEMA = "a2_packet_runtime_closure_probe.v1"
MAGIC = b"A2K1"
FALSE_AUTHORITY_FIELDS = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
BASE_BLOCKERS = [
    "no_exact_cuda_auth_eval",
    "no_contest_cpu_auth_eval",
    "no_active_level2_lane_dispatch_claim",
    "operator_score_claim_review_not_done",
]


class RuntimeClosureBlocked(ValueError):
    def __init__(self, reason: str, blockers: list[str]):
        super().__init__(reason)
        self.blockers = blockers


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_rel(path: Path) -> str:
    return repo_relative(_repo_path(path), REPO_ROOT)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(json_text(payload).encode("utf-8")).hexdigest()


def _validate_zip_member_name(name: str) -> None:
    path = Path(name)
    if (
        not name
        or name.startswith("/")
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(part.startswith(".") or part == "__MACOSX" for part in path.parts)
    ):
        raise RuntimeClosureBlocked(
            f"unsafe ZIP member name: {name!r}",
            ["unsafe_zip_member_name"],
        )


def _read_single_member_zip(path: Path) -> tuple[dict[str, Any], str, bytes]:
    if not path.is_file():
        raise RuntimeClosureBlocked(f"archive not found: {path}", ["archive_missing"])
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise RuntimeClosureBlocked(
                f"duplicate ZIP members in {path}",
                ["duplicate_zip_members"],
            )
        if len(infos) != 1:
            raise RuntimeClosureBlocked(
                f"expected one archive member in {path}, found {len(infos)}",
                ["archive_not_single_member"],
            )
        info = infos[0]
        _validate_zip_member_name(info.filename)
        payload = zf.read(info)
    return (
        {
            "path": _repo_rel(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "member": {
                "name": info.filename,
                "bytes": info.file_size,
                "compress_size": info.compress_size,
                "crc": f"{info.CRC:08x}",
                "sha256": _sha256_bytes(payload),
                "starts_with_a2_magic": payload.startswith(MAGIC),
            },
        },
        info.filename,
        payload,
    )


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeClosureBlocked(
            f"cannot create import spec for {path}",
            ["packet_runtime_import_spec_failed"],
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_packet_runtime(packet_dir: Path):
    src_dir = packet_dir / "src"
    model_path = src_dir / "model.py"
    codec_path = src_dir / "codec.py"
    if not model_path.is_file() or not codec_path.is_file():
        raise RuntimeClosureBlocked(
            f"packet runtime missing src/model.py or src/codec.py under {packet_dir}",
            ["packet_runtime_missing_model_or_codec"],
        )
    old_model = sys.modules.get("model")
    old_codec = sys.modules.get("codec")
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(src_dir))
    try:
        model_mod = _load_module("model", model_path)
        codec_mod = _load_module("codec", codec_path)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
        try:
            sys.path.remove(str(src_dir))
        except ValueError:
            pass
    return codec_mod, model_mod, old_codec, old_model


def _restore_packet_runtime_modules(old_codec: object | None, old_model: object | None) -> None:
    for name, old in (("codec", old_codec), ("model", old_model)):
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


def _shape(value: object) -> list[int] | None:
    return list(value.shape) if hasattr(value, "shape") else None


def _probe_payload(
    *,
    codec_mod: object,
    model_mod: object,
    label: str,
    member_name: str,
    payload: bytes,
) -> dict[str, Any]:
    decoder_sd, latents, meta = codec_mod.parse_archive(payload)
    if not isinstance(decoder_sd, dict):
        raise RuntimeClosureBlocked(
            f"{label}: parse_archive did not return a decoder state dict",
            ["parse_archive_decoder_not_mapping"],
        )
    if not isinstance(meta, dict):
        raise RuntimeClosureBlocked(
            f"{label}: parse_archive did not return metadata",
            ["parse_archive_meta_not_mapping"],
        )

    decoder = model_mod.HNeRVDecoder(
        latent_dim=int(meta["latent_dim"]),
        base_channels=int(meta["base_channels"]),
        eval_size=tuple(meta["eval_size"]),
    )
    incompatible = decoder.load_state_dict(decoder_sd, strict=True)
    tensor_rows = []
    for name, tensor in decoder_sd.items():
        if not torch.is_tensor(tensor):
            raise RuntimeClosureBlocked(
                f"{label}: decoder state entry {name!r} is not a tensor",
                ["decoder_state_contains_non_tensor"],
            )
        tensor_rows.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": int(tensor.numel()),
            }
        )
    return {
        "label": label,
        "member_name": member_name,
        "archive_member_starts_with_a2_magic": payload.startswith(MAGIC),
        "parse_archive_passed": True,
        "model_state_load_strict_passed": True,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "decoder_tensor_count": len(decoder_sd),
        "decoder_parameter_count": sum(int(tensor.numel()) for tensor in decoder_sd.values()),
        "latents_type": type(latents).__name__,
        "latents_shape": _shape(latents),
        "latents_dtype": str(getattr(latents, "dtype", "")),
        "meta": {
            "n_pairs": int(meta["n_pairs"]),
            "latent_dim": int(meta["latent_dim"]),
            "base_channels": int(meta["base_channels"]),
            "eval_size": list(meta["eval_size"]),
        },
        "tensor_rows": tensor_rows,
    }


def probe_runtime_closure(
    *,
    packet_dir: Path,
    source_archive: Path,
    candidate_archive: Path,
    created_utc: str | None = None,
) -> dict[str, Any]:
    packet_dir = _repo_path(packet_dir)
    source_archive = _repo_path(source_archive)
    candidate_archive = _repo_path(candidate_archive)
    source_record, source_member_name, source_payload = _read_single_member_zip(source_archive)
    candidate_record, candidate_member_name, candidate_payload = _read_single_member_zip(candidate_archive)
    if source_member_name != candidate_member_name:
        raise RuntimeClosureBlocked(
            f"source/candidate member name mismatch: {source_member_name!r} vs {candidate_member_name!r}",
            ["source_candidate_member_name_mismatch"],
        )
    if source_payload.startswith(MAGIC):
        raise RuntimeClosureBlocked(
            "source archive unexpectedly starts with A2 magic",
            ["source_archive_unexpected_a2_magic"],
        )
    if not candidate_payload.startswith(MAGIC):
        raise RuntimeClosureBlocked(
            "candidate archive does not start with A2 magic",
            ["candidate_archive_missing_a2_magic"],
        )

    codec_mod, model_mod, old_codec, old_model = _load_packet_runtime(packet_dir)
    try:
        source_probe = _probe_payload(
            codec_mod=codec_mod,
            model_mod=model_mod,
            label="source_stock_fallback",
            member_name=source_member_name,
            payload=source_payload,
        )
        candidate_probe = _probe_payload(
            codec_mod=codec_mod,
            model_mod=model_mod,
            label="candidate_a2_length_prefix",
            member_name=candidate_member_name,
            payload=candidate_payload,
        )
    finally:
        _restore_packet_runtime_modules(old_codec, old_model)

    manifest = {
        "schema": SCHEMA,
        "tool": TOOL_NAME,
        "created_utc": created_utc or _utc_now(),
        **FALSE_AUTHORITY_FIELDS,
        "status": "runtime_closure_verified_no_score",
        "evidence_grade": "diagnostic_cpu_prep",
        "evidence_semantics": "packet_runtime_parse_and_model_load_only_no_frame_inflate_no_score",
        "dispatch_attempted": False,
        "remote_gpu_run": False,
        "packet_dir": _repo_rel(packet_dir),
        "source_archive": source_record,
        "candidate_archive": candidate_record,
        "runtime_closure": {
            "verified": True,
            "source_stock_fallback_parse_passed": True,
            "candidate_a2_parse_passed": True,
            "strict_model_load_passed": True,
            "source_candidate_member_name_match": True,
            "cleared_blockers": ["packet_local_inflate_parity_not_run"],
            "remaining_blockers": BASE_BLOCKERS,
        },
        "source_probe": source_probe,
        "candidate_probe": candidate_probe,
        "score_affecting_payload_changed": (
            source_record["member"]["sha256"] != candidate_record["member"]["sha256"]
        ),
        "dispatch_blockers": BASE_BLOCKERS,
    }
    manifest["manifest_sha256_excluding_self"] = _canonical_json_sha256(manifest)
    return manifest


def _blocked_manifest(
    *,
    reason: str,
    blockers: list[str],
    args: argparse.Namespace,
    created_utc: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "tool": TOOL_NAME,
        "created_utc": created_utc,
        **FALSE_AUTHORITY_FIELDS,
        "status": "blocked_fail_closed",
        "evidence_grade": "blocked",
        "evidence_semantics": "no_score_no_dispatch",
        "reason": reason,
        "packet_dir": _repo_rel(args.packet_dir),
        "source_archive": _repo_rel(args.source_archive),
        "candidate_archive": _repo_rel(args.candidate_archive),
        "dispatch_attempted": False,
        "remote_gpu_run": False,
        "runtime_closure": {
            "verified": False,
            "missing_wire_contracts": sorted(set(blockers)),
        },
        "dispatch_blockers": sorted({*BASE_BLOCKERS, "runtime_closure_blocked", *blockers}),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--candidate-archive", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--now-utc")
    parser.add_argument("--fail-if-blocked", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    created_utc = args.now_utc or _utc_now()
    if args.candidate_archive is None:
        args.candidate_archive = args.packet_dir / "archive.zip"
    if args.json_out is None:
        args.json_out = args.packet_dir.parent / "a2_runtime_closure_probe.json"

    try:
        manifest = probe_runtime_closure(
            packet_dir=args.packet_dir,
            source_archive=args.source_archive,
            candidate_archive=args.candidate_archive,
            created_utc=created_utc,
        )
        write_json(_repo_path(args.json_out), manifest)
        print(f"runtime closure probe: {repo_relative(_repo_path(args.json_out), REPO_ROOT)}")
        print("status: runtime_closure_verified_no_score")
        return 0
    except RuntimeClosureBlocked as exc:
        manifest = _blocked_manifest(
            reason=str(exc),
            blockers=exc.blockers,
            args=args,
            created_utc=created_utc,
        )
        write_json(_repo_path(args.json_out), manifest)
        print(f"blocked runtime closure probe: {repo_relative(_repo_path(args.json_out), REPO_ROOT)}")
        print(f"reason: {exc}")
        return 1 if args.fail_if_blocked else 2


if __name__ == "__main__":
    raise SystemExit(main())
