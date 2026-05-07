#!/usr/bin/env python3
"""Assemble a deterministic PR101 HNeRV runtime packet for a candidate archive.

This is local custody tooling only. It copies a candidate ``archive.zip`` next
to the public PR101 ``hnerv_ft_microcodec`` runtime, records a deterministic
runtime-tree manifest, and can optionally run local source-vs-candidate inflate
parity without invoking scorers or dispatching GPU work.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, repo_relative, sha256_file  # noqa: E402

DEFAULT_RESULT_DIR = Path("experiments/results/pr101_codecop_lgwin18_candidate_20260507_codex")
DEFAULT_SOURCE_RUNTIME_DIR = Path(
    "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec"
)
DEFAULT_SOURCE_ARCHIVE = Path("experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip")
DEFAULT_CANDIDATE_ARCHIVE = (
    DEFAULT_RESULT_DIR
    / "substituted/pr101_lgwin18_pr101_lgwin18_auto_selectFalse_brotli_lgwin18_brotli_quality11/archive.zip"
)
DEFAULT_PACKET_DIR = DEFAULT_RESULT_DIR / "full_runtime_packet"
DEFAULT_FILE_LIST = Path(
    "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/public_test_segments.txt"
)
DEFAULT_CANDIDATE_ID = "pr101_lgwin18_hnerv_ft_microcodec"
EXCLUDED_DIR_NAMES = frozenset({"__pycache__", ".git", ".mypy_cache", ".pytest_cache"})
EXCLUDED_FILE_NAMES = frozenset({".DS_Store"})
EXCLUDED_SUFFIXES = (".pyc", ".pyo")
LOCAL_PARITY_RAW_BYTES_PER_OUTPUT = 600 * 2 * 874 * 1164 * 3


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_rel(path: Path) -> str:
    return repo_relative(_repo_path(path), REPO_ROOT)


def _format_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def _now_utc(value: str | None) -> dt.datetime:
    parsed = _parse_utc(value)
    if parsed is not None:
        return parsed.replace(microsecond=0)
    return dt.datetime.now(tz=dt.UTC).replace(microsecond=0)


def _canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(json_text(payload).encode("utf-8")).hexdigest()


def _mode_string(mode: int) -> str:
    return f"{stat.S_IMODE(mode):04o}"


def _copy_mode(source: Path) -> int:
    return 0o755 if source.stat().st_mode & 0o111 else 0o644


def _should_exclude(path: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
        return True
    return path.name in EXCLUDED_FILE_NAMES or path.suffix in EXCLUDED_SUFFIXES


def _iter_runtime_files(runtime_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in runtime_dir.rglob("*"):
        rel = path.relative_to(runtime_dir)
        if _should_exclude(rel):
            continue
        if path.is_symlink():
            raise ValueError(f"runtime packet refuses symlink: {path}")
        if path.is_file():
            files.append(rel)
    return sorted(files, key=lambda item: item.as_posix())


def _file_record(path: Path, *, relpath: str | None = None) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "mode": _mode_string(path.stat().st_mode),
        "relpath": relpath or path.name,
        "sha256": sha256_file(path),
    }


def _zip_record(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate ZIP members are not valid packet custody input: {path}")
        members = [
            {
                "bytes": info.file_size,
                "compress_size": info.compress_size,
                "crc": f"{info.CRC:08x}",
                "name": info.filename,
                "sha256": hashlib.sha256(zf.read(info)).hexdigest(),
            }
            for info in sorted(infos, key=lambda item: item.filename)
        ]
    record = _file_record(path, relpath="archive.zip")
    record["members"] = members
    return record


def _copy_file_deterministic(source: Path, target: Path, *, mode: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    os.chmod(target, mode)


def _remove_excluded_paths(root: Path) -> list[str]:
    removed: list[str] = []
    if not root.exists():
        return removed
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        rel = path.relative_to(root)
        if not _should_exclude(rel):
            continue
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        removed.append(rel.as_posix())
    return sorted(removed)


def _prepare_packet_dir(packet_dir: Path, *, force: bool) -> None:
    if not packet_dir.exists():
        packet_dir.mkdir(parents=True)
        return
    if not packet_dir.is_dir():
        raise ValueError(f"packet output path exists and is not a directory: {packet_dir}")
    if any(packet_dir.iterdir()):
        if not force:
            raise ValueError(f"packet output directory is not empty; pass --force to replace: {packet_dir}")
        shutil.rmtree(packet_dir)
        packet_dir.mkdir(parents=True)


def _runtime_tree_sha256(runtime_files: list[dict[str, Any]]) -> str:
    hash_basis = [
        {
            "bytes": row["bytes"],
            "mode": row["mode"],
            "relpath": row["relpath"],
            "sha256": row["sha256"],
        }
        for row in runtime_files
    ]
    return _canonical_json_sha256(hash_basis)


def _bash_n(path: Path) -> dict[str, Any]:
    proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
    return {
        "command": f"bash -n {path}",
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stderr": proc.stderr.strip(),
        "stdout": proc.stdout.strip(),
    }


def build_packet(
    *,
    source_runtime_dir: Path,
    candidate_archive: Path,
    packet_dir: Path,
    source_archive: Path | None,
    candidate_id: str,
    recorded_at_utc: dt.datetime,
    force: bool = False,
) -> dict[str, Any]:
    source_runtime_dir = _repo_path(source_runtime_dir)
    candidate_archive = _repo_path(candidate_archive)
    packet_dir = _repo_path(packet_dir)
    source_archive = _repo_path(source_archive) if source_archive is not None else None

    if not source_runtime_dir.is_dir():
        raise FileNotFoundError(f"source runtime directory not found: {source_runtime_dir}")
    if not candidate_archive.is_file():
        raise FileNotFoundError(f"candidate archive not found: {candidate_archive}")
    if source_archive is not None and not source_archive.is_file():
        raise FileNotFoundError(f"source archive not found: {source_archive}")

    _prepare_packet_dir(packet_dir, force=force)

    runtime_rels = _iter_runtime_files(source_runtime_dir)
    copied_runtime_files: list[dict[str, Any]] = []
    for rel in runtime_rels:
        source = source_runtime_dir / rel
        target = packet_dir / rel
        mode = _copy_mode(source)
        _copy_file_deterministic(source, target, mode=mode)
        copied_runtime_files.append(_file_record(target, relpath=rel.as_posix()))

    packet_archive = packet_dir / "archive.zip"
    _copy_file_deterministic(candidate_archive, packet_archive, mode=0o644)

    runtime_tree_sha256 = _runtime_tree_sha256(copied_runtime_files)
    manifest_path = packet_dir / "runtime_custody_manifest.json"
    payload: dict[str, Any] = {
        "schema": "pr101_runtime_packet_custody_v1",
        "tool": "tools.build_pr101_runtime_packet",
        "candidate_id": candidate_id,
        "recorded_at_utc": _format_utc(recorded_at_utc),
        "score_claim": False,
        "dispatch_attempted": False,
        "remote_gpu_run": False,
        "packet_dir": _repo_rel(packet_dir),
        "source_runtime_dir": _repo_rel(source_runtime_dir),
        "source_archive": _zip_record(source_archive) if source_archive is not None else None,
        "candidate_archive_source": _repo_rel(candidate_archive),
        "packet_archive": _zip_record(packet_archive),
        "runtime_custody": {
            "copied_file_count": len(copied_runtime_files),
            "deterministic_hash_basis": "sha256(canonical JSON of sorted relpath/mode/bytes/sha256 records)",
            "excluded_dir_names": sorted(EXCLUDED_DIR_NAMES),
            "excluded_file_names": sorted(EXCLUDED_FILE_NAMES),
            "excluded_suffixes": list(EXCLUDED_SUFFIXES),
            "runtime_files": copied_runtime_files,
            "runtime_tree_sha256": runtime_tree_sha256,
        },
        "runtime_checks": {
            "inflate_sh_bash_n": _bash_n(packet_dir / "inflate.sh"),
        },
        "local_inflate_parity": {
            "attempted": False,
            "reason": "not_requested",
            "raw_bytes_per_output_if_run": LOCAL_PARITY_RAW_BYTES_PER_OUTPUT,
        },
        "blockers_remaining": [
            "active_level2_lane_dispatch_claim",
            "exact_cuda_auth_eval",
            "contest_auth_eval_adjudication",
            "operator_score_claim_review",
        ],
    }
    payload["manifest_sha256_excluding_self"] = _canonical_json_sha256(payload)
    manifest_path.write_text(json_text(payload), encoding="utf-8")
    return payload


def _extract_single_member(zip_path: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise ValueError(f"expected exactly one file member in {zip_path}, found {len(infos)}")
        info = infos[0]
        if info.filename.startswith("/") or ".." in Path(info.filename).parts:
            raise ValueError(f"unsafe ZIP member for parity extraction: {info.filename}")
        target = out_dir / info.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zf.read(info))
    return _file_record(target, relpath=info.filename)


def _output_hashes(output_dir: Path) -> list[dict[str, Any]]:
    return [
        _file_record(path, relpath=path.relative_to(output_dir).as_posix())
        for path in sorted(output_dir.rglob("*"), key=lambda item: item.relative_to(output_dir).as_posix())
        if path.is_file()
    ]


def _expected_output_relpaths(file_list: Path) -> list[Path]:
    relpaths: list[Path] = []
    for raw_line in file_list.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        base = line.rsplit(".", 1)[0] if "." in line else line
        rel = Path(f"{base}.raw")
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe output path implied by file list: {line}")
        relpaths.append(rel)
    return relpaths


def _prepare_output_parent_dirs(file_list: Path, output_dirs: list[Path]) -> list[str]:
    relpaths = _expected_output_relpaths(file_list)
    for output_dir in output_dirs:
        for rel in relpaths:
            (output_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    return [rel.as_posix() for rel in relpaths]


def run_local_inflate_parity(
    *,
    source_runtime_dir: Path,
    source_archive: Path,
    packet_dir: Path,
    file_list: Path,
    parity_dir: Path,
    timeout_seconds: int,
    retain_outputs: bool = False,
) -> dict[str, Any]:
    source_runtime_dir = _repo_path(source_runtime_dir)
    source_archive = _repo_path(source_archive)
    packet_dir = _repo_path(packet_dir)
    file_list = _repo_path(file_list)
    parity_dir = _repo_path(parity_dir)

    if not file_list.is_file():
        raise FileNotFoundError(f"file list not found: {file_list}")
    if not source_archive.is_file():
        raise FileNotFoundError(f"source archive not found: {source_archive}")

    if parity_dir.exists():
        shutil.rmtree(parity_dir)
    data_source = parity_dir / "data/source"
    data_candidate = parity_dir / "data/candidate"
    out_source = parity_dir / "outputs/source"
    out_candidate = parity_dir / "outputs/candidate"
    data_source.mkdir(parents=True)
    data_candidate.mkdir(parents=True)
    out_source.mkdir(parents=True)
    out_candidate.mkdir(parents=True)

    source_member = _extract_single_member(source_archive, data_source)
    candidate_member = _extract_single_member(packet_dir / "archive.zip", data_candidate)
    expected_outputs = _prepare_output_parent_dirs(file_list, [out_source, out_candidate])
    venv_bin = REPO_ROOT / ".venv/bin"
    env = os.environ.copy()
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"

    runs = []
    for label, runtime_dir, data_dir, output_dir in (
        ("source", source_runtime_dir, data_source, out_source),
        ("candidate", packet_dir, data_candidate, out_candidate),
    ):
        cmd = [
            "bash",
            str(runtime_dir / "inflate.sh"),
            str(data_dir),
            str(output_dir),
            str(file_list),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=timeout_seconds,
        )
        runs.append(
            {
                "command": " ".join(cmd),
                "label": label,
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr[-4000:],
                "stdout_tail": proc.stdout[-4000:],
                "succeeded": proc.returncode == 0,
            }
        )

    source_outputs = _output_hashes(out_source)
    candidate_outputs = _output_hashes(out_candidate)
    source_by_rel = {row["relpath"]: row for row in source_outputs}
    candidate_by_rel = {row["relpath"]: row for row in candidate_outputs}
    compared_relpaths = sorted(set(source_by_rel) | set(candidate_by_rel))
    comparisons = [
        {
            "relpath": rel,
            "source_sha256": source_by_rel.get(rel, {}).get("sha256"),
            "candidate_sha256": candidate_by_rel.get(rel, {}).get("sha256"),
            "sha256_equal": source_by_rel.get(rel, {}).get("sha256") == candidate_by_rel.get(rel, {}).get("sha256"),
            "source_bytes": source_by_rel.get(rel, {}).get("bytes"),
            "candidate_bytes": candidate_by_rel.get(rel, {}).get("bytes"),
        }
        for rel in compared_relpaths
    ]
    all_runs_succeeded = all(row["succeeded"] for row in runs)
    all_outputs_equal = bool(compared_relpaths) and all(row["sha256_equal"] for row in comparisons)
    removed_packet_runtime_artifacts = _remove_excluded_paths(packet_dir)
    record = {
        "attempted": True,
        "schema": "pr101_source_candidate_local_inflate_parity_v1",
        "score_claim": False,
        "scorers_invoked": False,
        "parity_dir": _repo_rel(parity_dir),
        "outputs_retained": retain_outputs,
        "timeout_seconds": timeout_seconds,
        "raw_bytes_per_output": LOCAL_PARITY_RAW_BYTES_PER_OUTPUT,
        "expected_output_relpaths": expected_outputs,
        "source_member": source_member,
        "candidate_member": candidate_member,
        "runs": runs,
        "source_outputs": source_outputs,
        "candidate_outputs": candidate_outputs,
        "comparisons": comparisons,
        "all_runs_succeeded": all_runs_succeeded,
        "all_outputs_equal": all_outputs_equal,
        "passed": all_runs_succeeded and all_outputs_equal,
        "packet_runtime_cleanup_after_hashing": {
            "removed_excluded_artifacts": removed_packet_runtime_artifacts,
        },
    }
    if not retain_outputs:
        shutil.rmtree(parity_dir)
        record["parity_dir_removed_after_hashing"] = True
    else:
        record["parity_dir_removed_after_hashing"] = False
    return record


def _update_manifest_with_parity(packet_dir: Path, parity: dict[str, Any]) -> dict[str, Any]:
    manifest_path = _repo_path(packet_dir) / "runtime_custody_manifest.json"
    payload = json_load_object(manifest_path)
    payload["local_inflate_parity"] = parity
    payload["manifest_sha256_excluding_self"] = _canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256_excluding_self"}
    )
    manifest_path.write_text(json_text(payload), encoding="utf-8")
    return payload


def json_load_object(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime-dir", type=Path, default=DEFAULT_SOURCE_RUNTIME_DIR)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--candidate-archive", type=Path, default=DEFAULT_CANDIDATE_ARCHIVE)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--now-utc", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-local-inflate-parity", action="store_true")
    parser.add_argument("--file-list", type=Path, default=DEFAULT_FILE_LIST)
    parser.add_argument("--parity-dir", type=Path, default=DEFAULT_RESULT_DIR / "local_inflate_parity")
    parser.add_argument("--inflate-timeout-seconds", type=int, default=3600)
    parser.add_argument("--keep-local-inflate-outputs", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = build_packet(
        source_runtime_dir=args.source_runtime_dir,
        candidate_archive=args.candidate_archive,
        packet_dir=args.packet_dir,
        source_archive=args.source_archive,
        candidate_id=args.candidate_id,
        recorded_at_utc=_now_utc(args.now_utc),
        force=args.force,
    )
    if args.run_local_inflate_parity:
        try:
            parity = run_local_inflate_parity(
                source_runtime_dir=args.source_runtime_dir,
                source_archive=args.source_archive,
                packet_dir=args.packet_dir,
                file_list=args.file_list,
                parity_dir=args.parity_dir,
                timeout_seconds=args.inflate_timeout_seconds,
                retain_outputs=args.keep_local_inflate_outputs,
            )
        except subprocess.TimeoutExpired as exc:
            parity_dir = _repo_path(args.parity_dir)
            if not args.keep_local_inflate_outputs and parity_dir.exists():
                shutil.rmtree(parity_dir)
            removed_packet_runtime_artifacts = _remove_excluded_paths(_repo_path(args.packet_dir))
            parity = {
                "attempted": True,
                "passed": False,
                "reason": "timeout",
                "command": " ".join(str(part) for part in exc.cmd),
                "timeout_seconds": args.inflate_timeout_seconds,
                "outputs_retained": args.keep_local_inflate_outputs,
                "parity_dir_removed_after_hashing": not args.keep_local_inflate_outputs,
                "packet_runtime_cleanup_after_hashing": {
                    "removed_excluded_artifacts": removed_packet_runtime_artifacts,
                },
            }
        payload = _update_manifest_with_parity(args.packet_dir, parity)

    manifest_path = _repo_path(args.packet_dir) / "runtime_custody_manifest.json"
    print(f"[pr101-runtime-packet] packet_dir={payload['packet_dir']}")
    print(f"[pr101-runtime-packet] manifest={_repo_rel(manifest_path)}")
    print(f"[pr101-runtime-packet] runtime_tree_sha256={payload['runtime_custody']['runtime_tree_sha256']}")
    if payload.get("local_inflate_parity", {}).get("attempted"):
        print(f"[pr101-runtime-packet] local_inflate_parity_passed={payload['local_inflate_parity'].get('passed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
