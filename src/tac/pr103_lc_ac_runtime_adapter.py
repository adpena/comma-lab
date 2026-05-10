"""Runtime-adapter builder for PR103 ``hnerv_lc_ac`` section-length changes."""

from __future__ import annotations

import importlib.util
import ast
import hashlib
import os
import re
import shutil
import stat
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from tac.hnerv_lowlevel_packer import read_strict_single_member_zip
from tac.pr103_arithmetic_transform_plan import CANDIDATE_SCHEMA
from tac.repo_io import json_text, read_json, repo_relative, sha256_bytes, sha256_file, write_json

ADAPTER_SCHEMA = "pr103_lc_ac_runtime_adapter_v1"
PACKET_SCHEMA = "pr103_lc_ac_histogram_candidate_packet_v1"
ARCHIVE_MANIFEST_SCHEMA = "pr103_lc_ac_histogram_candidate_archive_manifest_v1"
REQUIRED_RUNTIME_CONSTANTS = (
    "BR_LEN",
    "HIST_LEN",
    "MERGED_AC_LEN",
    "LO_LEN",
    "HI_HIST_LEN",
)
EXCLUDED_DIR_NAMES = frozenset({"__pycache__", ".git", ".mypy_cache", ".pytest_cache"})
EXCLUDED_FILE_NAMES = frozenset({".DS_Store"})
EXCLUDED_SUFFIXES = (".pyc", ".pyo")


class Pr103RuntimeAdapterError(ValueError):
    """Raised when a PR103 runtime adapter cannot be built safely."""


def build_pr103_lc_ac_runtime_adapter(
    *,
    candidate_manifest: str | Path,
    source_runtime_dir: str | Path,
    output_runtime_dir: str | Path,
    repo_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Copy and patch a PR103 runtime tree for a materialized candidate.

    The candidate manifest remains the source of truth for section lengths. The
    copied runtime is not a score claim and does not run the scorer.
    """

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    manifest_path = _repo_path(Path(candidate_manifest), repo)
    source_dir = _repo_path(Path(source_runtime_dir), repo)
    output_dir = _repo_path(Path(output_runtime_dir), repo)
    manifest = _read_candidate_manifest(manifest_path)
    constants = _candidate_runtime_constants(manifest)
    candidate_record = _mapping(manifest.get("candidate_archive"))
    source_record = _mapping(manifest.get("source_archive"))
    candidate_archive = _resolve_manifest_path(
        Path(str(candidate_record.get("path") or "")),
        manifest_path=manifest_path,
        repo=repo,
    )
    source_archive = _resolve_manifest_path(
        Path(str(source_record.get("path") or "")),
        manifest_path=manifest_path,
        repo=repo,
    )
    _validate_inputs(source_dir, candidate_archive)
    _validate_candidate_archive_custody(source_record, source_archive, label="source archive")
    _validate_candidate_archive_custody(candidate_record, candidate_archive)
    _validate_output_location(source_dir, output_dir)
    _prepare_output_dir(output_dir, force=force)
    copied_files = _copy_runtime_tree(source_dir, output_dir)
    changes = _patch_inflate_constants(output_dir / "inflate.py", constants)
    shell_patch = _patch_inflate_shell_python(output_dir / "inflate.sh")
    source_probe = _runtime_consumption_probe(
        source_dir / "inflate.py",
        source_archive,
        _source_runtime_constants(manifest),
    )
    probe = _runtime_consumption_probe(output_dir / "inflate.py", candidate_archive, constants)
    parity = _decoder_state_parity_proof(source_probe, probe)
    runtime_files = _runtime_file_records(output_dir)
    blockers = _blockers_for_probe(probe, parity)
    return {
        "schema": ADAPTER_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_manifest": {
            "path": repo_relative(manifest_path, repo),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
        "candidate_archive": {
            "path": repo_relative(candidate_archive, repo),
            "bytes": candidate_archive.stat().st_size,
            "sha256": sha256_file(candidate_archive),
        },
        "source_runtime_dir": repo_relative(source_dir, repo),
        "output_runtime_dir": repo_relative(output_dir, repo),
        "copied_files": copied_files,
        "constant_changes": changes,
        "shell_patch": shell_patch,
        "source_runtime_consumption_probe": _public_probe_record(source_probe),
        "runtime_consumption_probe": probe,
        "decoder_state_parity_proof": parity,
        "semantic_stream_parity": _mapping(manifest.get("semantic_stream_parity")),
        "runtime_files": runtime_files,
        "runtime_tree_sha256": _runtime_tree_sha256(runtime_files),
        "readiness_blockers": blockers,
        "dispatch_blockers": [
            "pr103_runtime_adapter_is_not_dispatch_authorization",
            *blockers,
        ],
    }


def build_pr103_lc_ac_candidate_packet(
    *,
    runtime_adapter_manifest: str | Path,
    packet_dir: str | Path,
    repo_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build a local packet surface for pre-submission compliance.

    This packages the byte-different candidate archive with the adapted runtime
    and writes custody sidecars. It does not run scorers, open a dispatch claim,
    or mark the packet exact-eval ready.
    """

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    manifest_path = _repo_path(Path(runtime_adapter_manifest), repo)
    manifest = _read_adapter_manifest(manifest_path)
    output_dir = _repo_path(Path(packet_dir), repo)
    candidate_record = _mapping(manifest.get("candidate_archive"))
    source_runtime_dir = _repo_path(Path(str(manifest.get("output_runtime_dir") or "")), repo)
    candidate_archive = _repo_path(Path(str(candidate_record.get("path") or "")), repo)
    _validate_inputs(source_runtime_dir, candidate_archive)
    _validate_candidate_archive_custody(candidate_record, candidate_archive)
    _validate_output_location(source_runtime_dir, output_dir)
    _prepare_output_dir(output_dir, force=force)

    packet_archive = output_dir / "archive.zip"
    shutil.copyfile(candidate_archive, packet_archive)
    os.chmod(packet_archive, 0o644)
    runtime_files = _copy_runtime_tree(source_runtime_dir, output_dir)
    archive_manifest = _archive_manifest(packet_archive, repo=repo)
    archive_manifest.update(
        {
            "schema": ARCHIVE_MANIFEST_SCHEMA,
            "score_claim": False,
            "dispatch_attempted": False,
            "ready_for_exact_eval_dispatch": False,
            "source_runtime_adapter_manifest": {
                "path": repo_relative(manifest_path, repo),
                "sha256": sha256_file(manifest_path),
            },
            "decoder_state_parity_proof": _mapping(manifest.get("decoder_state_parity_proof")),
        }
    )
    write_json(output_dir / "archive_manifest.json", archive_manifest)
    report_text = _packet_report_text(
        archive_manifest=archive_manifest,
        runtime_adapter_manifest=manifest,
    )
    report_path = output_dir / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    os.chmod(report_path, 0o644)
    packet_runtime_files = _runtime_file_records(output_dir)
    packet_manifest = {
        "schema": PACKET_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "scorers_invoked": False,
        "packet_dir": repo_relative(output_dir, repo),
        "archive": _file_record(packet_archive, relpath="archive.zip"),
        "archive_manifest": _file_record(output_dir / "archive_manifest.json", relpath="archive_manifest.json"),
        "report": _file_record(report_path, relpath="report.txt"),
        "runtime_adapter_manifest": {
            "path": repo_relative(manifest_path, repo),
            "sha256": sha256_file(manifest_path),
        },
        "runtime_files_copied": runtime_files,
        "runtime_files": packet_runtime_files,
        "runtime_tree_sha256": _runtime_tree_sha256(packet_runtime_files),
        "adapter_runtime_tree_sha256": manifest.get("runtime_tree_sha256"),
        "decoder_state_parity_proof": _mapping(manifest.get("decoder_state_parity_proof")),
        "readiness_blockers": [
            "lane_dispatch_claim_missing",
            "exact_cuda_auth_eval_missing",
        ],
        "dispatch_blockers": [
            "pr103_candidate_packet_is_not_dispatch_authorization",
            "lane_dispatch_claim_missing",
            "exact_cuda_auth_eval_missing",
        ],
    }
    write_json(output_dir / "packet_manifest.json", packet_manifest)
    return packet_manifest


def _read_candidate_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise Pr103RuntimeAdapterError("candidate manifest must be a JSON object")
    if payload.get("schema") != CANDIDATE_SCHEMA:
        raise Pr103RuntimeAdapterError(
            "candidate manifest must come from PR103 histogram candidate materializer"
        )
    if payload.get("score_claim") is True or payload.get("dispatch_attempted") is True:
        raise Pr103RuntimeAdapterError("candidate manifest must be a no-score local artifact")
    return payload


def _read_adapter_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise Pr103RuntimeAdapterError("runtime adapter manifest must be a JSON object")
    if payload.get("schema") != ADAPTER_SCHEMA:
        raise Pr103RuntimeAdapterError(f"runtime adapter manifest must have schema {ADAPTER_SCHEMA}")
    if payload.get("score_claim") is True or payload.get("dispatch_attempted") is True:
        raise Pr103RuntimeAdapterError("runtime adapter manifest must be a no-score local artifact")
    parity = _mapping(payload.get("decoder_state_parity_proof"))
    if parity.get("passed") is not True:
        raise Pr103RuntimeAdapterError("runtime adapter manifest missing decoder-state parity proof")
    return payload


def _candidate_runtime_constants(manifest: dict[str, Any]) -> dict[str, int]:
    contract = _mapping(manifest.get("runtime_adapter_contract"))
    raw = _mapping(contract.get("public_runtime_constants"))
    return _runtime_constants(raw, label="candidate")


def _source_runtime_constants(manifest: dict[str, Any]) -> dict[str, int]:
    contract = _mapping(manifest.get("runtime_adapter_contract"))
    raw = _mapping(contract.get("source_runtime_constants"))
    if not raw:
        raise Pr103RuntimeAdapterError("candidate manifest missing source runtime constants")
    return _runtime_constants(raw, label="source")


def _runtime_constants(raw: dict[str, Any], *, label: str) -> dict[str, int]:
    constants: dict[str, int] = {}
    for name in REQUIRED_RUNTIME_CONSTANTS:
        if name not in raw:
            raise Pr103RuntimeAdapterError(f"{label} manifest missing runtime constant {name}")
        constants[name] = int(raw[name])
        if constants[name] < 0:
            raise Pr103RuntimeAdapterError(f"runtime constant {name} must be nonnegative")
    return constants


def _validate_inputs(source_runtime_dir: Path, candidate_archive: Path) -> None:
    if not source_runtime_dir.is_dir():
        raise FileNotFoundError(f"source runtime directory not found: {source_runtime_dir}")
    for name in ("inflate.py", "inflate.sh"):
        if not (source_runtime_dir / name).is_file():
            raise FileNotFoundError(f"source runtime missing {name}: {source_runtime_dir}")
    if not candidate_archive.is_file():
        raise FileNotFoundError(f"candidate archive not found: {candidate_archive}")


def _validate_candidate_archive_custody(
    record: dict[str, Any],
    candidate_archive: Path,
    *,
    label: str = "candidate archive",
) -> None:
    expected_bytes = record.get("bytes")
    expected_sha = str(record.get("sha256") or "")
    if expected_bytes is None or not expected_sha:
        raise Pr103RuntimeAdapterError(f"candidate manifest missing {label} bytes or sha256")
    actual_bytes = candidate_archive.stat().st_size
    actual_sha = sha256_file(candidate_archive)
    if int(expected_bytes) != actual_bytes:
        raise Pr103RuntimeAdapterError(
            f"{label} byte mismatch: manifest={expected_bytes} actual={actual_bytes}"
        )
    if expected_sha != actual_sha:
        raise Pr103RuntimeAdapterError(
            f"{label} sha256 mismatch: manifest={expected_sha} actual={actual_sha}"
        )


def _validate_output_location(source_runtime_dir: Path, output_runtime_dir: Path) -> None:
    source = source_runtime_dir.resolve()
    output = output_runtime_dir.resolve()
    if output == source:
        raise Pr103RuntimeAdapterError("output runtime directory must differ from source")
    if _path_is_relative_to(output, source):
        raise Pr103RuntimeAdapterError("output runtime directory cannot be inside source runtime")
    if _path_is_relative_to(source, output):
        raise Pr103RuntimeAdapterError("source runtime directory cannot be inside output runtime")


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise Pr103RuntimeAdapterError(
                f"output runtime directory exists; pass force to replace: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _copy_runtime_tree(source_dir: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for source in sorted(source_dir.rglob("*"), key=lambda item: item.relative_to(source_dir).as_posix()):
        rel = source.relative_to(source_dir)
        if _should_exclude(rel):
            continue
        target = output_dir / rel
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if source.is_symlink():
            raise Pr103RuntimeAdapterError(f"runtime adapter refuses symlink: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        os.chmod(target, stat.S_IMODE(source.stat().st_mode))
        copied.append(rel.as_posix())
    return copied


def _patch_inflate_constants(path: Path, constants: dict[str, int]) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    assignments = _integer_constant_assignments(text)
    changes: list[dict[str, Any]] = []
    for name, value in constants.items():
        if name not in assignments:
            raise Pr103RuntimeAdapterError(
                f"inflate.py must contain one top-level integer assignment for {name}"
            )
        pattern = re.compile(rf"^(?P<prefix>{re.escape(name)}\s*=\s*)(?P<value>\d+)\s*$", re.M)
        matches = list(pattern.finditer(text))
        if len(matches) != 1:
            raise Pr103RuntimeAdapterError(
                f"expected exactly one integer assignment for {name}, found {len(matches)}"
            )
        old_value = int(matches[0].group("value"))
        if old_value != assignments[name]:
            raise Pr103RuntimeAdapterError(
                f"AST and text assignment disagree for {name}: ast={assignments[name]} text={old_value}"
            )
        text = pattern.sub(rf"\g<prefix>{value}", text)
        changes.append({"name": name, "old": old_value, "new": int(value), "changed": old_value != value})
    path.write_text(text, encoding="utf-8")
    return changes


def _patch_inflate_shell_python(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    old = 'python "$HERE/inflate.py" "$SRC" "$DST"'
    new = '"${PYTHON:-python}" "$HERE/inflate.py" "$SRC" "$DST"'
    count = text.count(old)
    if count != 1:
        raise Pr103RuntimeAdapterError(
            f"expected exactly one bare python inflate invocation, found {count}"
        )
    path.write_text(text.replace(old, new), encoding="utf-8")
    return {
        "changed": True,
        "old": old,
        "new": new,
        "basis": "allow_managed_interpreter_override_without_changing_default_contest_python",
        "score_claim": False,
        "dispatch_attempted": False,
    }


def _integer_constant_assignments(text: str) -> dict[str, int]:
    module = ast.parse(text)
    assignments: dict[str, int] = {}
    seen: set[str] = set()
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id not in REQUIRED_RUNTIME_CONSTANTS:
            continue
        if target.id in seen:
            raise Pr103RuntimeAdapterError(
                f"inflate.py contains duplicate top-level assignment for {target.id}"
            )
        seen.add(target.id)
        if (
            not isinstance(node.value, ast.Constant)
            or isinstance(node.value.value, bool)
            or not isinstance(node.value.value, int)
        ):
            raise Pr103RuntimeAdapterError(
                f"inflate.py assignment for {target.id} must be an integer literal"
            )
        assignments[target.id] = int(node.value.value)
    return assignments


def _runtime_consumption_probe(
    inflate_py: Path,
    candidate_archive: Path,
    constants: dict[str, int],
) -> dict[str, Any]:
    source = read_strict_single_member_zip(candidate_archive)
    runtime = _load_runtime_module(inflate_py)
    sections = runtime.parse_archive(source.payload)
    if len(sections) != 8:
        raise Pr103RuntimeAdapterError(f"runtime parse_archive returned {len(sections)} sections")
    sca, br_b, hists_b, merged_ac, mins_scales, lo_b, hi_hist_b, wrp_b = sections
    lengths = {
        "BR_LEN": len(br_b),
        "HIST_LEN": len(hists_b),
        "MERGED_AC_LEN": len(merged_ac),
        "LO_LEN": len(lo_b),
        "HI_HIST_LEN": len(hi_hist_b),
    }
    mismatches = {
        key: {"expected": value, "actual": lengths[key]}
        for key, value in constants.items()
        if lengths[key] != value
    }
    if mismatches:
        raise Pr103RuntimeAdapterError(f"runtime parse length mismatch: {mismatches}")
    hi_hist = np.frombuffer(runtime.brotli.decompress(hi_hist_b), dtype=np.uint16)
    state_dict, hi_decoded = runtime.build_state_dict(br_b, hists_b, merged_ac, sca, hi_hist)
    latents = runtime.decode_latents(mins_scales, lo_b, hi_decoded)
    runtime.apply_corrections(latents, wrp_b)
    state_records = _state_dict_records(state_dict)
    latent_record = _tensor_record(latents)
    return {
        "passed": True,
        "member_name": source.member_name,
        "payload_bytes": source.member_bytes,
        "parsed_lengths": lengths,
        "sidecar_tail_bytes": len(wrp_b),
        "state_dict_tensors": len(state_dict),
        "state_dict_params": int(sum(_tensor_numel(tensor) for tensor in state_dict.values())),
        "state_dict_sha256": sha256_bytes(json_text(state_records).encode("utf-8")),
        "state_dict_tensor_records": state_records,
        "latents_shape": [int(value) for value in latents.shape],
        "latents_sha256": latent_record["sha256"],
        "latents_dtype": latent_record["dtype"],
        "full_frame_inflate_ran": False,
        "full_frame_output_parity_inferred": False,
        "score_claim": False,
        "dispatch_attempted": False,
    }


def _load_runtime_module(path: Path) -> ModuleType:
    module_name = "pr103_lc_ac_runtime_adapter_" + sha256_file(path)[:16]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise Pr103RuntimeAdapterError(f"could not import runtime module: {path}")
    module = importlib.util.module_from_spec(spec)
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
    return module


def _runtime_file_records(runtime_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(runtime_dir.rglob("*"), key=lambda item: item.relative_to(runtime_dir).as_posix()):
        rel = path.relative_to(runtime_dir)
        if path.is_file() and not _should_exclude(rel):
            records.append(
                {
                    "relpath": rel.as_posix(),
                    "bytes": path.stat().st_size,
                    "mode": f"{stat.S_IMODE(path.stat().st_mode):04o}",
                    "sha256": sha256_file(path),
                }
            )
    return records


def _file_record(path: Path, *, relpath: str) -> dict[str, Any]:
    return {
        "relpath": relpath,
        "bytes": path.stat().st_size,
        "mode": f"{stat.S_IMODE(path.stat().st_mode):04o}",
        "sha256": sha256_file(path),
    }


def _archive_manifest(path: Path, *, repo: Path) -> dict[str, Any]:
    record = {
        "candidate_archive": {
            "path": repo_relative(path, repo),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "members": [],
        }
    }
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            payload = zf.read(info)
            record["candidate_archive"]["members"].append(
                {
                    "name": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "crc": int(info.CRC),
                    "sha256": sha256_bytes(payload),
                }
            )
    return record


def _packet_report_text(
    *,
    archive_manifest: dict[str, Any],
    runtime_adapter_manifest: dict[str, Any],
) -> str:
    archive = _mapping(archive_manifest.get("candidate_archive"))
    parity = _mapping(runtime_adapter_manifest.get("decoder_state_parity_proof"))
    return "\n".join(
        [
            "PR103 LC-AC histogram candidate packet",
            "",
            "score_claim: false",
            "dispatch_attempted: false",
            "ready_for_exact_eval_dispatch: false",
            f"archive_sha256: {archive.get('sha256')}",
            f"archive_size_bytes: {archive.get('bytes')}",
            f"runtime_tree_sha256: {runtime_adapter_manifest.get('runtime_tree_sha256')}",
            f"decoder_state_parity_passed: {parity.get('passed') is True}",
            f"full_frame_output_parity_inferred: {parity.get('full_frame_output_parity_inferred') is True}",
            "",
            "This packet is a compliance-smoke artifact only. It is not a score claim.",
            "",
        ]
    )


def _state_dict_records(state_dict: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"name": str(name), **_tensor_record(state_dict[name])}
        for name in sorted(state_dict)
    ]


def _tensor_record(value: Any) -> dict[str, Any]:
    if hasattr(value, "detach"):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": [int(item) for item in contiguous.shape],
        "dtype": str(contiguous.dtype),
        "sha256": sha256_bytes(contiguous.tobytes()),
    }


def _tensor_numel(value: Any) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    return int(np.asarray(value).size)


def _decoder_state_parity_proof(
    source_probe: dict[str, Any],
    candidate_probe: dict[str, Any],
) -> dict[str, Any]:
    tensor_records_match = (
        source_probe.get("state_dict_tensor_records")
        == candidate_probe.get("state_dict_tensor_records")
    )
    state_sha_match = source_probe.get("state_dict_sha256") == candidate_probe.get("state_dict_sha256")
    latents_sha_match = source_probe.get("latents_sha256") == candidate_probe.get("latents_sha256")
    latents_shape_match = source_probe.get("latents_shape") == candidate_probe.get("latents_shape")
    passed = bool(tensor_records_match and state_sha_match and latents_sha_match and latents_shape_match)
    return {
        "passed": passed,
        "basis": "source_and_candidate_decode_to_identical_decoder_state_and_latents",
        "source_state_dict_sha256": source_probe.get("state_dict_sha256"),
        "candidate_state_dict_sha256": candidate_probe.get("state_dict_sha256"),
        "state_dict_sha_match": state_sha_match,
        "state_dict_tensor_records_match": tensor_records_match,
        "source_latents_sha256": source_probe.get("latents_sha256"),
        "candidate_latents_sha256": candidate_probe.get("latents_sha256"),
        "latents_sha_match": latents_sha_match,
        "latents_shape_match": latents_shape_match,
        "full_frame_inflate_ran": False,
        "full_frame_output_parity_inferred": passed,
        "score_claim": False,
        "dispatch_attempted": False,
    }


def _public_probe_record(probe: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in probe.items()
        if key != "state_dict_tensor_records"
    }


def _runtime_tree_sha256(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json_text(records).encode("utf-8")).hexdigest()


def _blockers_for_probe(probe: dict[str, Any], parity: dict[str, Any]) -> list[str]:
    blockers = [
        "strict_pre_submission_compliance_json_missing",
        "lane_dispatch_claim_missing",
        "exact_cuda_auth_eval_missing",
    ]
    if probe.get("passed") is not True:
        blockers.insert(0, "runtime_consumption_probe_failed")
    if parity.get("passed") is not True:
        blockers.insert(0, "decoder_state_latent_parity_missing")
    return blockers


def _repo_path(path: Path, repo: Path) -> Path:
    return path if path.is_absolute() else repo / path


def _resolve_manifest_path(path: Path, *, manifest_path: Path, repo: Path) -> Path:
    if path.is_absolute():
        return path
    repo_candidate = repo / path
    if repo_candidate.exists():
        return repo_candidate
    return manifest_path.parent / path


def _should_exclude(path: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
        return True
    return path.name in EXCLUDED_FILE_NAMES or path.suffix in EXCLUDED_SUFFIXES


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


__all__ = [
    "ADAPTER_SCHEMA",
    "ARCHIVE_MANIFEST_SCHEMA",
    "PACKET_SCHEMA",
    "Pr103RuntimeAdapterError",
    "build_pr103_lc_ac_candidate_packet",
    "build_pr103_lc_ac_runtime_adapter",
]
