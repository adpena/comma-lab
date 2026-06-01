# SPDX-License-Identifier: MIT
"""Execute receiver proofs for HPRC representation-spine runner rows."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.hprc.inflate import CAMERA_H, CAMERA_W, CHANNELS
from tac.substrates.hprc.representation_spine import (
    HPRC_REPRESENTATION_SPINE_PROJECTION_SCHEMA,
)
from tac.substrates.hprc.resolution_contract import CONTEST_PAIR_COUNT
from tac.substrates.hprc.spine_bounded_runner import (
    HPRC_SPINE_BOUNDED_RUNNER_PLAN_SCHEMA,
)

HPRC_SPINE_RECEIVER_EXECUTION_REPORT_SCHEMA = "hprc_spine_receiver_execution_report.v1"
HPRC_SPINE_RECEIVER_PROOF_SCHEMA = "hprc_spine_receiver_proof.v1"
DEFAULT_RECEIVER_PROOF_SCRATCH_MARGIN_BYTES = 512 * 1024 * 1024
_PUBLIC_RUNTIME_BY_FAMILY: dict[str, tuple[str, str]] = {
    "pr95_hnerv": ("source/submissions/hnerv_muon/inflate.sh", "0"),
    "hnerv_packed": ("source/submissions/hnerv_ft_microcodec/inflate.sh", "0"),
}
_ARCHIVE_EMBEDDED_RUNTIME_FAMILIES = frozenset({"pact_nerv", "pact_nerv_vq"})


@dataclass(frozen=True)
class SpineReceiverRuntimeOverride:
    """Runtime override used by tests and recovered public-runtime packets."""

    family: str
    inflate_sh: Path


def execute_spine_receiver_rows(
    *,
    runner_plan_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path = ".",
    row_ids: list[str] | tuple[str, ...] = (),
    max_rows: int | None = None,
    runtime_overrides: list[SpineReceiverRuntimeOverride] | tuple[SpineReceiverRuntimeOverride, ...] = (),
    timeout_seconds: float = 900.0,
    max_output_bytes: int = 64 * 1024 * 1024,
    allow_large_output: bool = False,
    expected_raw_bytes_overrides: dict[str, int] | None = None,
    output_contract_overrides: dict[str, str] | None = None,
    keep_work_dir: bool = False,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Run ``inflate.sh`` receiver proofs for selected bounded-runner rows.

    Rows are deduped by ``family + projection_manifest_path`` so repeated hard
    byte ceilings do not run the same multi-GB public inflate twice.
    """

    root = Path(repo_root).expanduser().resolve(strict=False)
    plan_path = _resolve(runner_plan_path, base=root)
    out = _resolve(output_dir, base=root)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "hprc_spine_receiver_execution_report.json"
    if report_path.exists() and not allow_overwrite:
        raise FileExistsError(f"output exists; pass allow_overwrite=True: {report_path}")

    plan = _load_json_object(plan_path)
    if plan.get("schema") != HPRC_SPINE_BOUNDED_RUNNER_PLAN_SCHEMA:
        raise ValueError(
            "runner_plan_path must point to "
            f"{HPRC_SPINE_BOUNDED_RUNNER_PLAN_SCHEMA}"
        )

    selected = _selected_rows(plan, row_ids=row_ids)
    deduped = _dedupe_rows(selected)
    if max_rows is not None:
        deduped = deduped[: int(max_rows)]
    overrides = {item.family: item.inflate_sh for item in runtime_overrides}
    expected_overrides = dict(expected_raw_bytes_overrides or {})
    contract_overrides = dict(output_contract_overrides or {})

    rows = [
        _execute_one_row(
            row=row,
            output_dir=out,
            repo_root=root,
            runtime_overrides=overrides,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            allow_large_output=allow_large_output,
            expected_raw_bytes_overrides=expected_overrides,
            output_contract_overrides=contract_overrides,
            keep_work_dir=keep_work_dir,
        )
        for row in deduped
    ]
    proof_rows = [row for row in rows if row.get("receiver_contract_satisfied") is True]
    blocker_rows = [row for row in rows if row.get("receiver_contract_satisfied") is not True]
    report = {
        "schema": HPRC_SPINE_RECEIVER_EXECUTION_REPORT_SCHEMA,
        "generated_at_utc": _utc_stamp(),
        "repo_root": root.as_posix(),
        "runner_plan_path": plan_path.as_posix(),
        "runner_plan_sha256": _sha256_file(plan_path),
        "selected_input_row_count": len(selected),
        "deduped_execution_row_count": len(deduped),
        "receiver_proof_passed_count": len(proof_rows),
        "receiver_proof_blocked_count": len(blocker_rows),
        "receiver_rows": rows,
        "exact_gate_policy": {
            "schema": "hprc_spine_receiver_exact_gate_policy.v1",
            "receiver_proof_is_preclaim_custody_only": True,
            "ready_for_exact_eval_dispatch": False,
            "blockers": [
                "contest_cpu_cuda_exact_eval_not_executed",
                "receiver_proof_is_not_score_authority",
            ],
        },
        "posterior_update_hooks": [
            _posterior_hook(row) for row in rows
        ],
        **FALSE_AUTHORITY,
    }
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {**report, "report_path": report_path.as_posix()}


def _execute_one_row(
    *,
    row: dict[str, Any],
    output_dir: Path,
    repo_root: Path,
    runtime_overrides: dict[str, Path],
    timeout_seconds: float,
    max_output_bytes: int,
    allow_large_output: bool,
    expected_raw_bytes_overrides: dict[str, int],
    output_contract_overrides: dict[str, str],
    keep_work_dir: bool,
) -> dict[str, Any]:
    family = str(row.get("family") or "")
    projection_path = _resolve_required(row.get("projection_manifest_path"), base=repo_root)
    row_id = str(row.get("runner_row_id") or family)
    projection_slug = hashlib.sha256(projection_path.as_posix().encode("utf-8")).hexdigest()[:12]
    row_slug = _safe_slug(f"{row_id}_{projection_slug}")
    row_out = output_dir / row_slug
    row_out.mkdir(parents=True, exist_ok=True)
    proof_path = row_out / "hprc_spine_receiver_proof.json"
    blockers: list[str] = []
    started = datetime.now(UTC)
    t0 = time.perf_counter()

    projection = _load_json_object(projection_path)
    if projection.get("schema") != HPRC_REPRESENTATION_SPINE_PROJECTION_SCHEMA:
        raise ValueError(f"not a spine projection manifest: {projection_path}")
    spine_manifest = _spine_manifest(projection)
    source = spine_manifest.get("source") if isinstance(spine_manifest.get("source"), dict) else {}
    archive_path = _resolve_required(source.get("path"), base=repo_root)
    archive_checks = _file_checks(archive_path, expected_sha256=str(source.get("sha256") or ""))
    member_name = str(source.get("member_name") or "0.bin")
    num_pairs = _infer_pair_count(family=family, spine_manifest=spine_manifest)
    expected_raw_bytes = int(
        expected_raw_bytes_overrides.get(family)
        or expected_raw_bytes_overrides.get(row_id)
        or num_pairs * 2 * CAMERA_H * CAMERA_W * CHANNELS
    )
    if expected_raw_bytes > int(max_output_bytes) and not allow_large_output:
        blockers.append("predicted_raw_output_exceeds_guardrail")
    storage = _storage_preflight(
        output_dir=row_out,
        expected_raw_bytes=expected_raw_bytes,
        extra_margin_bytes=DEFAULT_RECEIVER_PROOF_SCRATCH_MARGIN_BYTES,
    )
    if not storage["preflight_passed"]:
        blockers.append("receiver_proof_storage_preflight_failed")
    runtime = _runtime_for_family(
        family=family,
        archive_path=archive_path,
        repo_root=repo_root,
        runtime_overrides=runtime_overrides,
    )
    if family in output_contract_overrides:
        runtime["output_contract"] = output_contract_overrides[family]
    if row_id in output_contract_overrides:
        runtime["output_contract"] = output_contract_overrides[row_id]
    if runtime["inflate_sh"] is None and runtime.get("source") != "archive_embedded":
        blockers.append("receiver_runtime_not_resolved")
    elif runtime["inflate_sh"] is not None and not Path(runtime["inflate_sh"]).is_file():
        blockers.append("receiver_inflate_sh_missing")

    proc_summary: dict[str, Any] = {
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "timed_out": False,
        "runtime_files": [],
    }
    raw_path: Path | None = None
    output_summary: dict[str, Any] = {
        "kind": runtime.get("output_contract") or "raw_file",
        "bytes": 0,
        "sha256": None,
        "frame_count": None,
        "path": None,
    }
    work_dir = row_out / "runtime_consumption_work"
    cleanup = {
        "schema": "hprc_spine_receiver_cleanup.v1",
        "work_dir": work_dir.as_posix(),
        "work_dir_preserved": True,
        "work_dir_removed_after_hashing": False,
        "raw_output_retained": True,
        "blockers": [],
    }
    if not blockers:
        proc_summary, raw_path = _run_inflate(
            archive_path=archive_path,
            member_name=member_name,
            family=family,
            runtime=runtime,
            work_dir=work_dir,
            file_base=str(runtime["file_base"]),
            repo_root=repo_root,
            timeout_seconds=timeout_seconds,
        )
    if output_summary["kind"] == "png_tree":
        tree = _png_tree_summary(work_dir / "raw" / str(runtime["file_base"]))
        output_summary.update(tree)
    else:
        raw_exists = raw_path is not None and raw_path.is_file()
        output_summary.update(
            {
                "bytes": raw_path.stat().st_size if raw_exists and raw_path is not None else 0,
                "sha256": _sha256_file(raw_path) if raw_exists and raw_path is not None else None,
                "path": raw_path.as_posix() if raw_path is not None else None,
            }
        )
    if proc_summary.get("timed_out"):
        blockers.append("receiver_inflate_timeout")
    if proc_summary.get("returncode") not in (0, None):
        blockers.append("receiver_inflate_returncode_nonzero")
    if output_summary["kind"] == "png_tree":
        if output_summary.get("frame_count") != num_pairs * 2:
            blockers.append("receiver_png_frame_count_mismatch")
    elif not output_summary["sha256"] and not blockers:
        blockers.append("receiver_raw_output_missing")
    if output_summary["kind"] == "raw_file" and output_summary["bytes"] != expected_raw_bytes:
        blockers.append("receiver_raw_output_bytes_mismatch")

    receiver_ok = not blockers and output_summary["sha256"] is not None
    if receiver_ok and not keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=False)
        cleanup.update(
            {
                "work_dir_preserved": False,
                "work_dir_removed_after_hashing": True,
                "raw_output_retained": False,
            }
        )
    duration = time.perf_counter() - t0
    proof = {
        "schema": HPRC_SPINE_RECEIVER_PROOF_SCHEMA,
        "generated_at_utc": _utc_stamp(),
        "started_at_utc": started.isoformat(),
        "duration_seconds": duration,
        "runner_row_id": row_id,
        "family": family,
        "projection_manifest_path": projection_path.as_posix(),
        "projection_manifest_sha256": _sha256_file(projection_path),
        "archive_path": archive_path.as_posix(),
        "archive_bytes": archive_path.stat().st_size if archive_path.is_file() else None,
        "archive_sha256": archive_checks["actual_sha256"],
        "archive_custody": archive_checks,
        "source_archive": source,
        "archive_member_name": member_name,
        "runtime": runtime,
        "runtime_files": proc_summary.get("runtime_files")
        or (_runtime_files(Path(str(runtime["inflate_sh"]))) if runtime["inflate_sh"] else []),
        "num_pairs": num_pairs,
        "expected_raw_bytes": expected_raw_bytes,
        "receiver_output_kind": output_summary["kind"],
        "receiver_output_bytes": output_summary["bytes"],
        "receiver_output_sha256": output_summary["sha256"],
        "receiver_output_frame_count": output_summary.get("frame_count"),
        "receiver_output_path": output_summary["path"] if keep_work_dir else None,
        "subprocess": proc_summary,
        "storage_preflight": storage,
        "cleanup": cleanup,
        "receiver_contract_satisfied": receiver_ok,
        "runtime_consumption_proof_ready": receiver_ok,
        "exact_readiness_refusal": {
            "schema": "hprc_spine_receiver_exact_readiness_refusal.v1",
            "ready": False,
            "blockers": [
                "receiver_proof_is_not_score_authority",
                "contest_cpu_cuda_exact_eval_not_executed",
            ],
        },
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }
    proof_path.write_text(json.dumps(proof, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "schema": "hprc_spine_receiver_execution_row.v1",
        "runner_row_id": row_id,
        "family": family,
        "projection_manifest_path": projection_path.as_posix(),
        "archive_sha256": proof["archive_sha256"],
        "archive_bytes": proof["archive_bytes"],
        "receiver_contract_satisfied": receiver_ok,
        "runtime_consumption_proof_ready": receiver_ok,
        "receiver_output_sha256": proof["receiver_output_sha256"],
        "receiver_output_bytes": proof["receiver_output_bytes"],
        "receiver_output_kind": proof["receiver_output_kind"],
        "receiver_output_frame_count": proof["receiver_output_frame_count"],
        "proof_path": proof_path.as_posix(),
        "proof_sha256": _sha256_file(proof_path),
        "blockers": proof["blockers"],
        **FALSE_AUTHORITY,
    }


def _run_inflate(
    *,
    archive_path: Path,
    member_name: str,
    family: str,
    runtime: dict[str, Any],
    work_dir: Path,
    file_base: str,
    repo_root: Path,
    timeout_seconds: float,
) -> tuple[dict[str, Any], Path]:
    data_dir = work_dir / "data"
    raw_dir = work_dir / "raw"
    runtime_dir = work_dir / "runtime"
    file_list = work_dir / "file_list.txt"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        member_payload = zf.read(member_name)
        if runtime.get("source") == "archive_embedded":
            zf.extractall(runtime_dir)
    if runtime.get("source") == "archive_embedded":
        inflate_sh = runtime_dir / "inflate.sh"
    else:
        inflate_sh = Path(str(runtime["inflate_sh"]))
    staged_name = "x" if family == "hnerv_packed" and member_name == "x" else f"{file_base}.bin"
    staged_member = data_dir / staged_name
    staged_member.write_bytes(member_payload)
    if family == "hnerv_packed" and staged_name != f"{file_base}.bin":
        (data_dir / f"{file_base}.bin").write_bytes(member_payload)
    file_list.write_text(f"{file_base}.mkv\n", encoding="utf-8")
    env = os.environ.copy()
    venv_bin = repo_root / ".venv/bin"
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    try:
        proc = subprocess.run(
            [
                "bash",
                inflate_sh.as_posix(),
                data_dir.as_posix(),
                raw_dir.as_posix(),
                file_list.as_posix(),
            ],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        summary = {
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "timed_out": False,
            "runtime_files": _runtime_files(inflate_sh),
            "argv": [
                "bash",
                inflate_sh.as_posix(),
                data_dir.as_posix(),
                raw_dir.as_posix(),
                file_list.as_posix(),
            ],
        }
    except subprocess.TimeoutExpired as exc:
        summary = {
            "returncode": None,
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "runtime_files": _runtime_files(inflate_sh) if inflate_sh.is_file() else [],
            "argv": [
                "bash",
                inflate_sh.as_posix(),
                data_dir.as_posix(),
                raw_dir.as_posix(),
                file_list.as_posix(),
            ],
        }
    return summary, raw_dir / f"{file_base}.raw"


def _selected_rows(plan: dict[str, Any], *, row_ids: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    rows = plan.get("selected_runner_rows")
    if not isinstance(rows, list):
        return []
    wanted = {str(item) for item in row_ids}
    selected = [row for row in rows if isinstance(row, dict)]
    if wanted:
        selected = [row for row in selected if str(row.get("runner_row_id") or "") in wanted]
    return selected


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("family") or ""), str(row.get("projection_manifest_path") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _spine_manifest(projection: dict[str, Any]) -> dict[str, Any]:
    body = projection.get("projection") if isinstance(projection.get("projection"), dict) else projection
    manifest = body.get("manifest") if isinstance(body.get("manifest"), dict) else {}
    spine = manifest.get("representation_spine")
    if not isinstance(spine, dict):
        raise ValueError("projection manifest missing representation_spine")
    return spine


def _runtime_for_family(
    *,
    family: str,
    archive_path: Path,
    repo_root: Path,
    runtime_overrides: dict[str, Path],
) -> dict[str, Any]:
    if family in runtime_overrides:
        inflate = _resolve(runtime_overrides[family], base=repo_root)
        return _runtime_row(
            family=family,
            inflate_sh=inflate,
            file_base="0",
            source="override",
            output_contract="raw_file",
        )
    if family in _ARCHIVE_EMBEDDED_RUNTIME_FAMILIES and _archive_has_member(
        archive_path, "inflate.sh"
    ):
        return {
            "schema": "hprc_spine_receiver_runtime.v1",
            "family": family,
            "source": "archive_embedded",
            "inflate_sh": None,
            "archive_runtime_member": "inflate.sh",
            "file_base": "0",
            "output_contract": "png_tree",
        }
    default = _PUBLIC_RUNTIME_BY_FAMILY.get(family)
    if default is None:
        return {"schema": "hprc_spine_receiver_runtime.v1", "inflate_sh": None, "file_base": "0"}
    rel_inflate, file_base = default
    inflate = archive_path.parent / rel_inflate
    return _runtime_row(
        family=family,
        inflate_sh=inflate,
        file_base=file_base,
        source="source_archive_parent",
        output_contract="raw_file",
    )


def _runtime_row(
    *,
    family: str,
    inflate_sh: Path,
    file_base: str,
    source: str,
    output_contract: str,
) -> dict[str, Any]:
    return {
        "schema": "hprc_spine_receiver_runtime.v1",
        "family": family,
        "source": source,
        "inflate_sh": inflate_sh.as_posix(),
        "file_base": file_base,
        "output_contract": output_contract,
    }


def _runtime_files(inflate_sh: Path) -> list[dict[str, Any]]:
    runtime_dir = inflate_sh.parent
    rows: list[dict[str, Any]] = []
    for path in sorted(runtime_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rows.append({"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    return rows


def _archive_has_member(archive_path: Path, member_name: str) -> bool:
    try:
        with zipfile.ZipFile(archive_path) as zf:
            return member_name in {info.filename for info in zf.infolist() if not info.is_dir()}
    except zipfile.BadZipFile:
        return False


def _png_tree_summary(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        return {
            "kind": "png_tree",
            "bytes": 0,
            "sha256": None,
            "frame_count": 0,
            "path": root.as_posix(),
        }
    rows: list[tuple[str, int, str]] = []
    total = 0
    for path in sorted(root.rglob("*.png")):
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total += size
        rows.append((rel, size, _sha256_file(path)))
    h = hashlib.sha256()
    for rel, size, sha in rows:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(str(size).encode("ascii"))
        h.update(b"\0")
        h.update(sha.encode("ascii"))
        h.update(b"\n")
    return {
        "kind": "png_tree",
        "bytes": total,
        "sha256": h.hexdigest(),
        "frame_count": len(rows),
        "path": root.as_posix(),
    }


def _infer_pair_count(*, family: str, spine_manifest: dict[str, Any]) -> int:
    extra = spine_manifest.get("manifest_extra")
    if isinstance(extra, dict):
        value = extra.get("num_pairs")
        if isinstance(value, int) and value > 0:
            return value
    if family in {"pr95_hnerv", "hnerv_packed"}:
        return CONTEST_PAIR_COUNT
    return CONTEST_PAIR_COUNT


def _storage_preflight(*, output_dir: Path, expected_raw_bytes: int, extra_margin_bytes: int) -> dict[str, Any]:
    free = shutil.disk_usage(output_dir).free
    required = int(expected_raw_bytes) + int(extra_margin_bytes)
    return {
        "schema": "hprc_spine_receiver_storage_preflight.v1",
        "path": output_dir.as_posix(),
        "free_bytes": int(free),
        "required_bytes": required,
        "expected_raw_bytes": int(expected_raw_bytes),
        "extra_margin_bytes": int(extra_margin_bytes),
        "preflight_passed": int(free) >= required,
    }


def _file_checks(path: Path, *, expected_sha256: str = "") -> dict[str, Any]:
    exists = path.is_file()
    actual_sha = _sha256_file(path) if exists else None
    blockers: list[str] = []
    if not exists:
        blockers.append("file_missing")
    if exists and expected_sha256 and actual_sha != expected_sha256:
        blockers.append("file_sha256_mismatch")
    return {
        "schema": "hprc_spine_receiver_file_custody.v1",
        "path": path.as_posix(),
        "exists": exists,
        "expected_sha256": expected_sha256 or None,
        "actual_sha256": actual_sha,
        "bytes": path.stat().st_size if exists else None,
        "verified": not blockers,
        "blockers": blockers,
    }


def _posterior_hook(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "hprc_spine_receiver_posterior_hook.v1",
        "family": row.get("family"),
        "archive_sha256": row.get("archive_sha256"),
        "stage": "receiver_proof",
        "scope": "full_video_inflate_runtime",
        "record_positive_after_exact_axis_only": True,
        "record_blocker_now": row.get("receiver_contract_satisfied") is not True,
        "blocker_codes": row.get("blockers", []),
    }


def _resolve_required(value: Any, *, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("required path field missing")
    return _resolve(value, base=base)


def _resolve(path: str | Path, *, base: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p).resolve(strict=False)


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:120]


def _dedupe(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "HPRC_SPINE_RECEIVER_EXECUTION_REPORT_SCHEMA",
    "HPRC_SPINE_RECEIVER_PROOF_SCHEMA",
    "SpineReceiverRuntimeOverride",
    "execute_spine_receiver_rows",
]
