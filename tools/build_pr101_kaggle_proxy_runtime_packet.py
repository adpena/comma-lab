#!/usr/bin/env python3
"""Build a PR101 Kaggle-proxy runtime packet without claiming score.

This consumes the archive-builder handoff emitted by
``tools/materialize_kaggle_pr101_proxy_candidate.py``.  It copies the public
PR101 runtime and unchanged source archive into a packet directory, then patches
only the three proved inflater bias constants that the runtime consumes.

It does not run scorers, run inflate, dispatch jobs, or promote the proxy
candidate to exact-eval readiness.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Any, Mapping

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, read_json, repo_relative, sha256_file, write_json  # noqa: E402

TOOL_NAME = "tools/build_pr101_kaggle_proxy_runtime_packet.py"
HANDOFF_SCHEMA = "pr101_kaggle_proxy_candidate_archive_builder_handoff_v1"
PACKET_SCHEMA = "pr101_kaggle_proxy_runtime_packet_v1"
PARAM_SCHEMA = "pr101_kaggle_proxy_candidate_params_v1"
DEFAULT_HANDOFF = Path(
    "experiments/results/kaggle_pr101_proxy_sweep_20260510_codex/"
    "pr101_proxy_sweep/local_materialization/archive_builder_handoff.json"
)
DEFAULT_SOURCE_RUNTIME_DIR = Path(
    "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/"
    "source/submissions/hnerv_ft_microcodec"
)
DEFAULT_SOURCE_ARCHIVE = Path(
    "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
)
DEFAULT_PACKET_DIR = Path(
    "experiments/results/kaggle_pr101_proxy_sweep_20260510_codex/"
    "pr101_proxy_sweep/proxy_runtime_packet"
)
MANIFEST_NAME = "runtime_packet_manifest.json"
REQUIRED_PARAMS = (
    "bias_b",
    "bias_g",
    "bias_r",
    "delta_scale",
    "latent_delta_scale",
    "smooth_weight",
)
RUNTIME_CONSUMED_PARAM_SLOTS = {
    "bias_r": "up[:, 0, 0]",
    "bias_b": "up[:, 0, 2]",
    "bias_g": "up[:, 1, 1]",
}
UNSUPPORTED_PARAMS = ("delta_scale", "latent_delta_scale", "smooth_weight")
FALSE_AUTHORITY_FIELDS = {
    "score_claim": False,
    "score_claim_valid": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
    "exact_auth_eval_performed": False,
    "contest_cuda_auth_eval": False,
}
BASE_BLOCKERS = [
    "proxy_substrate_not_contest_exact_eval",
    "no_contest_cuda_auth_eval",
    "unsupported_proxy_params_not_runtime_consumed",
    "local_inflate_or_runtime_consumption_proof_not_run",
    "active_level2_lane_dispatch_claim_required_before_exact_eval",
]
EXCLUDED_DIR_NAMES = frozenset({"__pycache__", ".git", ".mypy_cache", ".pytest_cache"})
EXCLUDED_FILE_NAMES = frozenset({".DS_Store"})
EXCLUDED_SUFFIXES = (".pyc", ".pyo")


class ProxyRuntimePacketError(ValueError):
    """Raised when the proxy runtime packet builder must fail closed."""


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_rel(path: Path) -> str:
    return repo_relative(_repo_path(path), REPO_ROOT)


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProxyRuntimePacketError(f"{field} must be a JSON object")
    return value


def _require_finite_number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProxyRuntimePacketError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ProxyRuntimePacketError(f"{field} must be finite")
    return result


def _mode_string(mode: int) -> str:
    return f"{stat.S_IMODE(mode):04o}"


def _copy_mode(path: Path) -> int:
    return 0o755 if path.stat().st_mode & 0o111 else 0o644


def _should_exclude(relpath: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in relpath.parts):
        return True
    return relpath.name in EXCLUDED_FILE_NAMES or relpath.suffix in EXCLUDED_SUFFIXES


def _file_record(path: Path, *, relpath: str | None = None) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "mode": _mode_string(path.stat().st_mode),
        "relpath": relpath or path.name,
        "sha256": sha256_file(path),
    }


def _zip_record(path: Path, *, relpath: str) -> dict[str, Any]:
    record = _file_record(path, relpath=relpath)
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ProxyRuntimePacketError(f"duplicate ZIP members in archive: {path}")
        record["members"] = [
            {
                "bytes": info.file_size,
                "compress_size": info.compress_size,
                "crc": f"{info.CRC:08x}",
                "name": info.filename,
                "sha256": _sha256_bytes(zf.read(info)),
            }
            for info in sorted(infos, key=lambda item: item.filename)
        ]
    return record


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    return _sha256_bytes(json_text(payload).encode("utf-8"))


def _runtime_tree_sha256(runtime_files: list[dict[str, Any]]) -> str:
    basis = [
        {
            "bytes": row["bytes"],
            "mode": row["mode"],
            "relpath": row["relpath"],
            "sha256": row["sha256"],
        }
        for row in sorted(runtime_files, key=lambda row: row["relpath"])
    ]
    return _canonical_json_sha256(basis)


def _validate_handoff(handoff: Mapping[str, Any]) -> dict[str, Any]:
    if handoff.get("schema") != HANDOFF_SCHEMA:
        raise ProxyRuntimePacketError(f"handoff schema must be {HANDOFF_SCHEMA!r}")
    if handoff.get("param_schema") != PARAM_SCHEMA:
        raise ProxyRuntimePacketError(f"param_schema must be {PARAM_SCHEMA!r}")

    candidate_id = handoff.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ProxyRuntimePacketError("candidate_id must be a non-empty string")

    evidence_boundary = _require_mapping(handoff.get("evidence_boundary"), "evidence_boundary")
    for field, expected in FALSE_AUTHORITY_FIELDS.items():
        if evidence_boundary.get(field) is not expected:
            raise ProxyRuntimePacketError(f"evidence_boundary.{field} must be {expected!r}")

    params_raw = _require_mapping(handoff.get("params"), "params")
    expected_keys = set(REQUIRED_PARAMS)
    actual_keys = set(params_raw)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing:
        raise ProxyRuntimePacketError(f"params missing required keys: {missing}")
    if extra:
        raise ProxyRuntimePacketError(f"params has unsupported keys: {extra}")

    params = {
        key: _require_finite_number(params_raw[key], f"params.{key}")
        for key in REQUIRED_PARAMS
    }
    return {"candidate_id": candidate_id, "params": params}


def _prepare_packet_dir(packet_dir: Path, *, force: bool) -> None:
    if packet_dir.exists() and not packet_dir.is_dir():
        raise ProxyRuntimePacketError(f"packet output path is not a directory: {packet_dir}")
    if not packet_dir.exists():
        packet_dir.mkdir(parents=True)
        return
    if not any(packet_dir.iterdir()):
        return
    if not force:
        raise ProxyRuntimePacketError(
            f"packet output directory is not empty; pass --force to replace: {packet_dir}"
        )
    shutil.rmtree(packet_dir)
    packet_dir.mkdir(parents=True)


def _copy_runtime_tree(source_runtime_dir: Path, packet_dir: Path) -> list[Path]:
    copied_rels: list[Path] = []
    for source in sorted(source_runtime_dir.rglob("*"), key=lambda path: path.relative_to(source_runtime_dir).as_posix()):
        rel = source.relative_to(source_runtime_dir)
        if _should_exclude(rel):
            continue
        if source.is_symlink():
            raise ProxyRuntimePacketError(f"runtime packet refuses symlink: {source}")
        if source.is_dir():
            continue
        target = packet_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        os.chmod(target, _copy_mode(source))
        copied_rels.append(rel)
    return copied_rels


def _bias_literal(value: float) -> str:
    return repr(value)


def _patch_inflate_biases(inflate_path: Path, params: Mapping[str, float]) -> dict[str, Any]:
    text = inflate_path.read_text(encoding="utf-8")
    replacements = {
        "bias_r": (
            "            up[:, 0, 0].sub_(1.0)",
            f"            up[:, 0, 0].add_({_bias_literal(params['bias_r'])})",
        ),
        "bias_b": (
            "            up[:, 0, 2].sub_(1.0)",
            f"            up[:, 0, 2].add_({_bias_literal(params['bias_b'])})",
        ),
        "bias_g": (
            "            up[:, 1, 1].sub_(1.0)",
            f"            up[:, 1, 1].add_({_bias_literal(params['bias_g'])})",
        ),
    }

    patch_rows: list[dict[str, Any]] = []
    for param_name, (old, new) in replacements.items():
        count = text.count(old)
        if count != 1:
            raise ProxyRuntimePacketError(
                f"expected exactly one PR101 bias line for {param_name}; found {count}"
            )
        text = text.replace(old, new, 1)
        patch_rows.append(
            {
                "param": param_name,
                "slot": RUNTIME_CONSUMED_PARAM_SLOTS[param_name],
                "value": params[param_name],
                "replacement": new.strip(),
            }
        )
    inflate_path.write_text(text, encoding="utf-8")
    return {
        "patched_file": "inflate.py",
        "runtime_consumed_params": patch_rows,
    }


def build_proxy_runtime_packet(
    *,
    handoff_path: Path = DEFAULT_HANDOFF,
    source_runtime_dir: Path = DEFAULT_SOURCE_RUNTIME_DIR,
    source_archive: Path = DEFAULT_SOURCE_ARCHIVE,
    packet_dir: Path = DEFAULT_PACKET_DIR,
    force: bool = False,
) -> dict[str, Any]:
    handoff_path = _repo_path(handoff_path)
    source_runtime_dir = _repo_path(source_runtime_dir)
    source_archive = _repo_path(source_archive)
    packet_dir = _repo_path(packet_dir)

    if not handoff_path.is_file():
        raise FileNotFoundError(f"handoff JSON not found: {handoff_path}")
    if not source_runtime_dir.is_dir():
        raise FileNotFoundError(f"source runtime directory not found: {source_runtime_dir}")
    if not source_archive.is_file():
        raise FileNotFoundError(f"source archive not found: {source_archive}")
    if source_runtime_dir.resolve() == packet_dir.resolve():
        raise ProxyRuntimePacketError("packet_dir must differ from source_runtime_dir")

    handoff_raw = _require_mapping(read_json(handoff_path), "handoff")
    validated = _validate_handoff(handoff_raw)
    params = validated["params"]

    _prepare_packet_dir(packet_dir, force=force)
    copied_rels = _copy_runtime_tree(source_runtime_dir, packet_dir)

    inflate_path = packet_dir / "inflate.py"
    if not inflate_path.is_file():
        raise ProxyRuntimePacketError("copied runtime is missing inflate.py")
    runtime_patch = _patch_inflate_biases(inflate_path, params)

    packet_archive = packet_dir / "archive.zip"
    shutil.copyfile(source_archive, packet_archive)
    os.chmod(packet_archive, 0o644)

    runtime_files = [
        _file_record(packet_dir / rel, relpath=rel.as_posix())
        for rel in copied_rels
    ]
    runtime_tree_sha256 = _runtime_tree_sha256(runtime_files)
    source_archive_record = _zip_record(source_archive, relpath="source_archive.zip")
    packet_archive_record = _zip_record(packet_archive, relpath="archive.zip")
    archive_unchanged = source_archive_record["sha256"] == packet_archive_record["sha256"]
    if not archive_unchanged:
        raise ProxyRuntimePacketError("copied archive SHA changed unexpectedly")

    unsupported_params = {
        key: {
            "value": params[key],
            "runtime_consumed": False,
            "blocker": f"{key}_not_runtime_consumed",
        }
        for key in UNSUPPORTED_PARAMS
    }
    blockers = list(BASE_BLOCKERS) + [row["blocker"] for row in unsupported_params.values()]
    manifest: dict[str, Any] = {
        "schema": PACKET_SCHEMA,
        "tool": TOOL_NAME,
        "candidate_id": validated["candidate_id"],
        "handoff_artifact": _repo_rel(handoff_path),
        "packet_dir": _repo_rel(packet_dir),
        "source_runtime_dir": _repo_rel(source_runtime_dir),
        "score_claim": False,
        "score_claim_valid": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "exact_auth_eval_performed": False,
        "contest_cuda_auth_eval": False,
        "scorers_invoked": False,
        "archive_changed": False,
        "archive_unchanged_sha256": packet_archive_record["sha256"],
        "source_archive": source_archive_record,
        "packet_archive": packet_archive_record,
        "runtime_custody": {
            "copied_file_count": len(runtime_files),
            "runtime_files": runtime_files,
            "runtime_tree_sha256": runtime_tree_sha256,
            "deterministic_hash_basis": "sha256(canonical JSON of sorted relpath/mode/bytes/sha256 records)",
            "excluded_dir_names": sorted(EXCLUDED_DIR_NAMES),
            "excluded_file_names": sorted(EXCLUDED_FILE_NAMES),
            "excluded_suffixes": list(EXCLUDED_SUFFIXES),
        },
        "runtime_patch": runtime_patch,
        "runtime_consumed_params": {
            key: params[key]
            for key in RUNTIME_CONSUMED_PARAM_SLOTS
        },
        "unsupported_params": unsupported_params,
        "blockers": blockers,
        "authority_boundary": {
            "proxy_only_input": True,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "next_authoritative_gate": "claimed exact CUDA auth eval of a byte-closed archive/runtime packet",
        },
    }
    manifest["manifest_sha256_excluding_self"] = _canonical_json_sha256(manifest)
    write_json(packet_dir / MANIFEST_NAME, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--source-runtime-dir", type=Path, default=DEFAULT_SOURCE_RUNTIME_DIR)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_proxy_runtime_packet(
        handoff_path=args.handoff,
        source_runtime_dir=args.source_runtime_dir,
        source_archive=args.source_archive,
        packet_dir=args.packet_dir,
        force=args.force,
    )
    print(json_text({
        "schema": "pr101_kaggle_proxy_runtime_packet_stdout_v1",
        "manifest": _repo_rel(_repo_path(args.packet_dir) / MANIFEST_NAME),
        "packet_dir": manifest["packet_dir"],
        "candidate_id": manifest["candidate_id"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "runtime_tree_sha256": manifest["runtime_custody"]["runtime_tree_sha256"],
        "archive_unchanged_sha256": manifest["archive_unchanged_sha256"],
        "blockers": manifest["blockers"],
    }), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
