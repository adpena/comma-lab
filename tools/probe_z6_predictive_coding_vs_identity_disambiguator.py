#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Z6 predictive-coding vs identity-predictor disambiguator.

This is a fail-closed L5-v2 probe surface. It either emits the exact paired
smoke commands required for Z6's full-FiLM and identity-predictor regimes, or
it consumes both smoke ``stats.json`` files and records a proxy-only verdict.

The output is deliberately not score authority. Synthetic smoke loss can
detect engineering regressions and decide the next build/eval step; it cannot
promote a paradigm, rank a lane, or replace paired contest CPU/CUDA evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "z6_predictive_coding_vs_identity_disambiguator_v1"
PROBE_ID = "z6_predictive_coding_vs_identity_disambiguator"
SUBSTRATE_TAG = "time_traveler_l5_z6"
LANE_ID = "lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516"
DEFAULT_VIDEO_PATH = "upstream/videos/0.mkv"
DEFAULT_FULL_OUTPUT_DIR = (
    "experiments/results/time_traveler_l5_z6/"
    "disambiguator_full_film_real_video_smoke_20260516_codex"
)
DEFAULT_IDENTITY_OUTPUT_DIR = (
    "experiments/results/time_traveler_l5_z6/"
    "disambiguator_identity_real_video_smoke_20260516_codex"
)
DEFAULT_OUTPUT_JSON = (
    ".omx/research/"
    "l5_v2_z6_identity_predictor_disambiguator_20260516_codex.json"
)
DEFAULT_OUTPUT_MD = (
    ".omx/research/"
    "l5_v2_z6_identity_predictor_disambiguator_20260516_codex.md"
)
TRAINER_PATH = "experiments/train_substrate_time_traveler_l5_z6.py"
CONTEST_ARCHIVE_NORMALIZER_BYTES = 37_545_489.0
FALSE_AUTHORITY_FLAGS = {
    "research_only": True,
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "ready_for_paid_dispatch": False,
    "paradigm_claim_allowed": False,
}
PAIRED_CONTROL_INITIALIZATION = "shared_modules_seed_order_matched_v2"
SMOKE_PROXY_BLOCKERS = [
    "smoke_proxy_synthetic_no_scorer",
    "no_contest_cpu_cuda_pair",
    "no_byte_closed_score_anchor",
    "not_paradigm_claim_authority",
]

ARCHIVE_PAIR_BLOCKERS_NO_SCORE = [
    "no_paired_exact_eval_json",
    "no_contest_cpu_cuda_pair",
    "not_score_authority",
]
INFLATE_OUTPUT_BLOCKERS_NO_SCORE = [
    "inflate_output_comparison_no_score_authority",
    "no_contest_cpu_cuda_pair",
]


def _proxy_evidence_grade(smoke_target_mode: object) -> str:
    if smoke_target_mode == "real-video":
        return "smoke_proxy_real_video_pair_no_scorer"
    return "smoke_proxy_synthetic_pair"


def _proxy_blockers(smoke_target_mode: object) -> list[str]:
    first = (
        "smoke_proxy_real_video_no_scorer"
        if smoke_target_mode == "real-video"
        else "smoke_proxy_synthetic_no_scorer"
    )
    return [
        first,
        "no_contest_cpu_cuda_pair",
        "no_byte_closed_score_anchor",
        "not_paradigm_claim_authority",
    ]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truncate_text(text: str, *, limit: int = 4096) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(resolved)


def _resolve_repo_path(path: Path, *, repo_root: Path) -> Path:
    resolved = path if path.is_absolute() else repo_root / path
    resolved = resolved.expanduser().resolve()
    resolved.relative_to(repo_root)
    text = str(resolved)
    if (
        text.startswith("/tmp/")
        or "/private/tmp/" in text
        or "/var/tmp/" in text
    ):
        raise ValueError(f"refusing to write transient Z6 probe output: {text!r}")
    return resolved


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_archive_blob(path: Path) -> tuple[bytes, list[str]]:
    """Read a Z6 archive blob from either ``0.bin`` or a single-member ZIP."""

    blob = path.read_bytes()
    if path.suffix.lower() != ".zip":
        return blob, []
    with zipfile.ZipFile(path) as zf:
        names = [info.filename for info in zf.infolist() if not info.is_dir()]
        if len(names) != 1:
            raise ValueError(
                f"{path}: expected single-member Z6 archive ZIP, got {names!r}"
            )
        return zf.read(names[0]), names


def _zip_sidecar_members(path: Path) -> tuple[list[dict[str, Any]], bytes | None]:
    """Return ZIP member custody rows and the sole member bytes when available."""

    if path.suffix.lower() != ".zip":
        return [], None
    rows: list[dict[str, Any]] = []
    sole_blob: bytes | None = None
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        for info in infos:
            blob = zf.read(info.filename)
            rows.append(
                {
                    "name": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "crc": f"{info.CRC:08x}",
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "compression_method": info.compress_type,
                }
            )
            sole_blob = blob
    if len(rows) != 1:
        sole_blob = None
    return rows, sole_blob


def _load_z6_archive(path: Path) -> Any:
    src_root = REPO_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from tac.substrates.time_traveler_l5_z6.archive import parse_archive

    blob, _ = _read_archive_blob(path)
    return parse_archive(blob)


def _identity_flag(arc: Any) -> bool | None:
    pcwm = arc.meta.get("predictive_coding_world_model_meta")
    if not isinstance(pcwm, Mapping):
        return None
    value = pcwm.get("identity_predictor")
    if type(value) is not bool:
        return None
    return value


def _derive_archive_zip_path(path: Path, *, identity: bool) -> Path | None:
    if path.suffix.lower() == ".zip":
        return path
    candidate = (
        path.parent / "archive_identity_predictor_disambiguator.zip"
        if identity
        else path.parent / "archive.zip"
    )
    return candidate if candidate.is_file() else None


def _archive_row(
    *,
    path: Path,
    arc: Any,
    repo_root: Path,
    mode: str,
    identity: bool,
) -> dict[str, Any]:
    blob, zip_members = _read_archive_blob(path)
    zip_path = _derive_archive_zip_path(path, identity=identity)
    zip_bytes = zip_path.stat().st_size if zip_path is not None else None
    zip_sha = _sha256_file(zip_path) if zip_path is not None else None
    zip_member_rows: list[dict[str, Any]] = []
    zip_member_matches_path_bytes: bool | None = None
    if zip_path is not None:
        zip_member_rows, sole_member_blob = _zip_sidecar_members(zip_path)
        zip_members = [str(row["name"]) for row in zip_member_rows]
        if sole_member_blob is not None:
            zip_member_matches_path_bytes = sole_member_blob == blob
    pcwm = arc.meta.get("predictive_coding_world_model_meta")
    pcwm_meta = dict(pcwm) if isinstance(pcwm, Mapping) else {}
    return {
        "mode": mode,
        "path": _repo_relative(path, repo_root=repo_root),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "parsed_member_bytes": len(blob),
        "zip_members": zip_members,
        "zip_path": (
            _repo_relative(zip_path, repo_root=repo_root)
            if zip_path is not None
            else None
        ),
        "zip_bytes": zip_bytes,
        "zip_sha256": zip_sha,
        "zip_member_rows": zip_member_rows,
        "zip_member_matches_path_bytes": zip_member_matches_path_bytes,
        "contest_archive_bytes_basis": zip_bytes if zip_bytes is not None else len(blob),
        "contest_archive_sha256_basis": zip_sha if zip_sha is not None else _sha256_file(path),
        "schema_version": arc.schema_version,
        "identity_predictor": _identity_flag(arc),
        "identity_predictor_disambiguator": bool(
            arc.meta.get("identity_predictor_disambiguator", False)
        ),
        "predictor_state_dict_key_count": len(arc.predictor_state_dict),
        "predictor_state_dict_keys": sorted(arc.predictor_state_dict.keys()),
        "latent_dim": int(arc.latent_init.numel()),
        "num_pairs": int(arc.residuals.shape[0]),
        "ego_motion_dim": int(arc.ego_motion.shape[1]),
        "predictor_hidden_dim": arc.meta.get("predictor_hidden_dim"),
        "requested_predictor_hidden_dim": arc.meta.get(
            "requested_predictor_hidden_dim"
        ),
        "effective_predictor_hidden_dim": arc.meta.get(
            "effective_predictor_hidden_dim"
        ),
        "predictor_film_mlp_hidden_dim": arc.meta.get(
            "predictor_film_mlp_hidden_dim"
        ),
        "predictor_architecture": arc.meta.get("predictor_architecture"),
        "predictor_depth": arc.meta.get("predictor_depth"),
        "lambda_residual_entropy": pcwm_meta.get("lambda_residual_entropy"),
        "predictor_kernel_size": pcwm_meta.get("predictor_kernel_size"),
    }


def _state_dict_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    import torch

    return all(torch.equal(left[key], right[key]) for key in left)


def _tensor_equal(left: Any, right: Any) -> bool:
    import torch

    return bool(torch.equal(left, right))


def _archive_pair_checks(full_arc: Any, identity_arc: Any) -> dict[str, bool]:
    return {
        "encoder_state_dict_equal": _state_dict_equal(
            full_arc.encoder_state_dict,
            identity_arc.encoder_state_dict,
        ),
        "decoder_state_dict_equal": _state_dict_equal(
            full_arc.decoder_state_dict,
            identity_arc.decoder_state_dict,
        ),
        "predictor_state_dict_equal": _state_dict_equal(
            full_arc.predictor_state_dict,
            identity_arc.predictor_state_dict,
        ),
        "predictor_keysets_equal": (
            set(full_arc.predictor_state_dict.keys())
            == set(identity_arc.predictor_state_dict.keys())
        ),
        "latent_init_equal": _tensor_equal(full_arc.latent_init, identity_arc.latent_init),
        "residuals_equal": _tensor_equal(full_arc.residuals, identity_arc.residuals),
        "ego_motion_equal": _tensor_equal(full_arc.ego_motion, identity_arc.ego_motion),
    }


def _default_archive_pair_from_run_dir(run_dir: Path) -> tuple[Path, Path]:
    full = run_dir / "0.bin"
    identity = run_dir / "0_identity_predictor_disambiguator.bin"
    if not full.is_file():
        raise ValueError(f"run-dir archive pair missing {full}")
    if not identity.is_file():
        raise ValueError(f"run-dir archive pair missing {identity}")
    return full, identity


def _default_eval_pair_from_run_dir(run_dir: Path) -> tuple[Path | None, Path | None]:
    full = run_dir / "contest_auth_eval.json"
    identity = run_dir / "contest_auth_eval_identity_predictor_disambiguator.json"
    if full.is_file() and identity.is_file():
        return full, identity
    if full.is_file() != identity.is_file():
        missing = identity if full.is_file() else full
        raise ValueError(f"run-dir paired exact-eval JSON missing {missing}")
    return None, None


def _hash_output_tree(root: Path, *, repo_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        file_hash = hashlib.sha256()
        size = 0
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                file_hash.update(chunk)
                size += len(chunk)
        sha = file_hash.hexdigest()
        aggregate.update(rel.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(sha.encode("ascii"))
        aggregate.update(b"\0")
        rows.append(
            {
                "path": rel,
                "repo_path": _repo_relative(path, repo_root=repo_root),
                "bytes": size,
                "sha256": sha,
            }
        )
        total_bytes += size
    return {
        "path": _repo_relative(root, repo_root=repo_root),
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "aggregate_sha256": aggregate.hexdigest() if rows else None,
        "files": rows,
    }


def _hash_runtime_closure(inflate_sh_path: Path, *, repo_root: Path) -> dict[str, Any]:
    """Hash the inflate runtime code closure, excluding archive payloads."""

    runtime_root = inflate_sh_path.parent
    closure_paths: list[Path] = [inflate_sh_path]
    sibling_inflate_py = runtime_root / "inflate.py"
    if sibling_inflate_py.is_file():
        closure_paths.append(sibling_inflate_py)
    python_cache_files_excluded: list[str] = []
    src_root = runtime_root / "src"
    if src_root.is_dir():
        for path in sorted(p for p in src_root.rglob("*") if p.is_file()):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                python_cache_files_excluded.append(
                    path.relative_to(runtime_root).as_posix()
                )
                continue
            closure_paths.append(path)

    seen: set[Path] = set()
    rows: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in sorted(p.resolve() for p in closure_paths):
        if path in seen:
            continue
        seen.add(path)
        rel = path.relative_to(runtime_root).as_posix()
        sha = _sha256_file(path)
        size = path.stat().st_size
        executable = bool(path.stat().st_mode & 0o111)
        aggregate.update(rel.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(sha.encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(int(executable)).encode("ascii"))
        aggregate.update(b"\0")
        rows.append(
            {
                "path": rel,
                "repo_path": _repo_relative(path, repo_root=repo_root),
                "bytes": size,
                "sha256": sha,
                "executable": executable,
            }
        )
        total_bytes += size

    archive_payloads_present = sorted(
        p.name
        for p in runtime_root.iterdir()
        if p.is_file() and p.suffix.lower() in {".bin", ".zip"}
    )
    return {
        "schema": "z6_inflate_runtime_closure_v1",
        "runtime_root": _repo_relative(runtime_root, repo_root=repo_root),
        "entrypoint": _repo_relative(inflate_sh_path, repo_root=repo_root),
        "aggregate_sha256": aggregate.hexdigest() if rows else None,
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "files": rows,
        "archive_payloads_excluded_from_runtime_hash": archive_payloads_present,
        "python_cache_files_excluded_from_runtime_hash": python_cache_files_excluded,
    }


def _count_byte_differences(left: Path, right: Path) -> int:
    differences = 0
    chunk_size = 1024 * 1024
    with left.open("rb") as left_fh, right.open("rb") as right_fh:
        while True:
            left_chunk = left_fh.read(chunk_size)
            right_chunk = right_fh.read(chunk_size)
            if not left_chunk and not right_chunk:
                break
            common = min(len(left_chunk), len(right_chunk))
            differences += sum(
                1
                for idx in range(common)
                if left_chunk[idx] != right_chunk[idx]
            )
            differences += abs(len(left_chunk) - len(right_chunk))
    return differences


def _compare_output_trees(
    *,
    full_output_dir: Path,
    identity_output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    full_tree = _hash_output_tree(full_output_dir, repo_root=repo_root)
    identity_tree = _hash_output_tree(identity_output_dir, repo_root=repo_root)
    full_files = {row["path"]: row for row in full_tree["files"]}
    identity_files = {row["path"]: row for row in identity_tree["files"]}
    common_paths = sorted(set(full_files) & set(identity_files))
    only_full = sorted(set(full_files) - set(identity_files))
    only_identity = sorted(set(identity_files) - set(full_files))
    file_diffs: list[dict[str, Any]] = []
    total_byte_differences = 0
    for rel in common_paths:
        full_path = full_output_dir / rel
        identity_path = identity_output_dir / rel
        byte_differences = _count_byte_differences(full_path, identity_path)
        total_byte_differences += byte_differences
        file_diffs.append(
            {
                "path": rel,
                "full_bytes": full_files[rel]["bytes"],
                "identity_bytes": identity_files[rel]["bytes"],
                "full_sha256": full_files[rel]["sha256"],
                "identity_sha256": identity_files[rel]["sha256"],
                "same_sha256": (
                    full_files[rel]["sha256"] == identity_files[rel]["sha256"]
                ),
                "byte_differences": byte_differences,
            }
        )
    same_file_set = not only_full and not only_identity
    aggregate_match = (
        same_file_set
        and full_tree["aggregate_sha256"] == identity_tree["aggregate_sha256"]
    )
    return {
        "full_output_tree": full_tree,
        "identity_output_tree": identity_tree,
        "same_output_file_set": same_file_set,
        "same_output_aggregate_sha256": aggregate_match,
        "only_full_output_files": only_full,
        "only_identity_output_files": only_identity,
        "common_output_files": common_paths,
        "file_diffs": file_diffs,
        "total_byte_differences": total_byte_differences,
        "runtime_output_changed": (
            not aggregate_match
            or bool(only_full)
            or bool(only_identity)
            or total_byte_differences > 0
        ),
    }


def compare_inflate_output_pair(
    *,
    full_archive_path: Path,
    identity_archive_path: Path,
    inflate_sh_path: Path,
    file_list_path: Path,
    output_root: Path,
    repo_root: Path = REPO_ROOT,
    python_executable: str | None = None,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Run the same inflate runtime for full/identity archives and compare bytes."""

    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(
            f"inflate output root must be absent or empty: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    full_archive_dir = output_root / "full_archive_dir"
    identity_archive_dir = output_root / "identity_archive_dir"
    full_output_dir = output_root / "full_output"
    identity_output_dir = output_root / "identity_output"
    for directory in (
        full_archive_dir,
        identity_archive_dir,
        full_output_dir,
        identity_output_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(full_archive_path, full_archive_dir / "0.bin")
    shutil.copyfile(identity_archive_path, identity_archive_dir / "0.bin")

    env = os.environ.copy()
    env["PYTHON"] = python_executable or sys.executable

    def run_one(mode: str, archive_dir: Path, output_dir: Path) -> dict[str, Any]:
        command = [
            str(inflate_sh_path),
            str(archive_dir),
            str(output_dir),
            str(file_list_path),
        ]
        proc = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return {
            "mode": mode,
            "command": command,
            "returncode": proc.returncode,
            "stdout": _truncate_text(proc.stdout),
            "stderr": _truncate_text(proc.stderr),
        }

    full_run = run_one("full_film_predictor", full_archive_dir, full_output_dir)
    identity_run = run_one(
        "identity_predictor",
        identity_archive_dir,
        identity_output_dir,
    )
    if full_run["returncode"] != 0 or identity_run["returncode"] != 0:
        raise ValueError(
            "inflate output comparison failed: "
            f"full_returncode={full_run['returncode']} "
            f"identity_returncode={identity_run['returncode']}"
        )

    comparison = _compare_output_trees(
        full_output_dir=full_output_dir,
        identity_output_dir=identity_output_dir,
        repo_root=repo_root,
    )
    return {
        "schema": "z6_identity_inflate_output_comparison_v1",
        "evidence_grade": "local_inflate_output_comparison_no_score",
        "evidence_axis": "[local-inflate-output advisory]",
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": list(INFLATE_OUTPUT_BLOCKERS_NO_SCORE),
        "inflate_sh_path": _repo_relative(inflate_sh_path, repo_root=repo_root),
        "inflate_sh_sha256": _sha256_file(inflate_sh_path),
        "runtime_custody": _hash_runtime_closure(
            inflate_sh_path,
            repo_root=repo_root,
        ),
        "file_list_path": _repo_relative(file_list_path, repo_root=repo_root),
        "file_list_sha256": _sha256_file(file_list_path),
        "output_root": _repo_relative(output_root, repo_root=repo_root),
        "python_executable": env["PYTHON"],
        "timeout_seconds": int(timeout_seconds),
        "runs": [full_run, identity_run],
        **comparison,
    }


def _first_number(payload: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


def _first_int(payload: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _archive_sha_from_eval(payload: Mapping[str, Any]) -> str | None:
    for key in ("archive_sha256", "archive_sha", "archive_hash"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        for key in ("archive_sha256", "archive_sha", "archive_hash"):
            value = provenance.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _axis_label(payload: Mapping[str, Any], override: str | None) -> str | None:
    if override:
        return override
    for key in ("axis", "evidence_axis", "lane_tag"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    score_axis = payload.get("score_axis")
    if score_axis == "contest_cuda":
        return "[contest-CUDA]"
    if score_axis == "contest_cpu":
        return "[contest-CPU]"
    if score_axis == "cpu_advisory":
        return "[macOS-CPU advisory]"
    if isinstance(score_axis, str) and score_axis:
        return score_axis
    return None


def _score_fields_from_eval(
    *,
    path: Path,
    expected_archive_bytes: int | None,
    expected_archive_sha256: str | None,
    axis: str | None,
    repo_root: Path,
) -> dict[str, Any]:
    payload = _load_json_object(path)
    score = _first_number(
        payload,
        (
            "score_recomputed_from_components",
            "canonical_score_recomputed",
            "canonical_score",
            "score",
            "final_score",
        ),
    )
    seg = _first_number(payload, ("avg_segnet_dist", "seg_dist", "seg_distortion"))
    pose = _first_number(
        payload,
        ("avg_posenet_dist", "pose_dist", "pose_distortion"),
    )
    archive_bytes = _first_int(
        payload,
        ("archive_size_bytes", "archive_bytes", "bytes"),
    )
    n_samples = _first_int(payload, ("n_samples", "sample_count", "samples"))
    recomputed = None
    recompute_matches_payload = None
    if seg is not None and pose is not None and archive_bytes is not None:
        recomputed = (
            100.0 * seg
            + math.sqrt(10.0 * pose)
            + 25.0 * archive_bytes / CONTEST_ARCHIVE_NORMALIZER_BYTES
        )
        if score is not None:
            recompute_matches_payload = abs(float(score) - recomputed) <= 1e-9
    eval_archive_sha = _archive_sha_from_eval(payload)
    archive_bytes_match = (
        expected_archive_bytes is not None
        and archive_bytes is not None
        and int(expected_archive_bytes) == int(archive_bytes)
    )
    archive_sha_match = (
        expected_archive_sha256 is not None
        and eval_archive_sha is not None
        and expected_archive_sha256 == eval_archive_sha
    )
    return {
        "path": _repo_relative(path, repo_root=repo_root),
        "sha256": _sha256_file(path),
        "axis": _axis_label(payload, axis),
        "score": score,
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "archive_size_bytes": archive_bytes,
        "archive_sha256": eval_archive_sha,
        "n_samples": n_samples,
        "score_recomputed_from_components": recomputed,
        "score_recompute_matches_payload": recompute_matches_payload,
        "archive_bytes_match_expected": archive_bytes_match,
        "archive_sha256_match_expected": archive_sha_match,
        "evidence_grade": payload.get("evidence_grade"),
        "score_axis": payload.get("score_axis"),
        "score_claim_valid": payload.get("score_claim_valid"),
        "promotion_eligible": payload.get("promotion_eligible"),
    }


def _expect_bool(payload: Mapping[str, Any], key: str, expected: bool) -> None:
    value = payload.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a literal JSON boolean")
    if value is not expected:
        raise ValueError(f"{key} must be {str(expected).lower()}")


def _expect_number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{key} must be a JSON number")
    return float(value)


def _expect_optional_number(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{key} must be null or a JSON number")
    return float(value)


def _validate_z6_smoke_stats(
    payload: Mapping[str, Any],
    *,
    expected_identity: bool,
    label: str,
) -> None:
    if payload.get("substrate_tag") != SUBSTRATE_TAG:
        raise ValueError(f"{label}: substrate_tag must be {SUBSTRATE_TAG!r}")
    if payload.get("lane_id") != LANE_ID:
        raise ValueError(f"{label}: lane_id must be {LANE_ID!r}")
    _expect_bool(payload, "smoke", True)
    _expect_bool(payload, "score_claim_valid", False)
    _expect_bool(payload, "promotion_eligible", False)
    _expect_bool(payload, "ready_for_exact_eval_dispatch", False)
    _expect_bool(payload, "identity_predictor", expected_identity)
    if payload.get("paired_control_initialization") != PAIRED_CONTROL_INITIALIZATION:
        raise ValueError(
            f"{label}: paired_control_initialization must be "
            f"{PAIRED_CONTROL_INITIALIZATION!r}; regenerate paired smoke stats "
            "with matched shared initialization"
        )
    _expect_number(payload, "final_loss_proxy")
    _expect_optional_number(payload, "final_recon")
    _expect_optional_number(payload, "final_residual")
    archive_bytes = payload.get("archive_bytes")
    if not isinstance(archive_bytes, int) or isinstance(archive_bytes, bool):
        raise ValueError(f"{label}: archive_bytes must be an integer")
    if archive_bytes <= 0:
        raise ValueError(f"{label}: archive_bytes must be > 0")


def _require_same_config(
    full: Mapping[str, Any],
    identity: Mapping[str, Any],
    key: str,
) -> None:
    if full.get(key) != identity.get(key):
        raise ValueError(
            f"full and identity smoke stats must match {key}: "
            f"{full.get(key)!r} != {identity.get(key)!r}"
        )


def _mode_stats_row(
    payload: Mapping[str, Any],
    *,
    path: Path,
    repo_root: Path,
    mode: str,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "path": _repo_relative(path, repo_root=repo_root),
        "sha256": _sha256_file(path),
        "identity_predictor": payload.get("identity_predictor"),
        "epochs": payload.get("epochs"),
        "requested_epochs": payload.get("requested_epochs"),
        "lambda_residual_entropy": payload.get("lambda_residual_entropy"),
        "predictor_kernel_size": payload.get("predictor_kernel_size"),
        "paired_control_initialization": payload.get(
            "paired_control_initialization"
        ),
        "smoke_target_mode": payload.get("smoke_target_mode"),
        "smoke_ego_motion_mode": payload.get("smoke_ego_motion_mode"),
        "ego_motion_nonzero_fraction": payload.get("ego_motion_nonzero_fraction"),
        "ego_motion_l2": payload.get("ego_motion_l2"),
        "final_loss_proxy": payload.get("final_loss_proxy"),
        "final_recon": payload.get("final_recon"),
        "final_residual": payload.get("final_residual"),
        "archive_bytes": payload.get("archive_bytes"),
        "evidence_grade": payload.get("evidence_grade"),
        "stats_payload": dict(payload),
    }


def _delta_optional(identity_value: Any, full_value: Any) -> float | None:
    if identity_value is None or full_value is None:
        return None
    return float(identity_value) - float(full_value)


def _smoke_command(
    *,
    output_dir: str,
    identity_predictor: bool,
    epochs: int,
    device: str,
    seed: int,
    target_mode: str,
    ego_motion_mode: str,
) -> list[str]:
    command = [
        ".venv/bin/python",
        TRAINER_PATH,
        "--video-path",
        DEFAULT_VIDEO_PATH,
        "--output-dir",
        output_dir,
        "--epochs",
        str(epochs),
        "--device",
        device,
        "--seed",
        str(seed),
        "--smoke-target-mode",
        target_mode,
        "--smoke-ego-motion-mode",
        ego_motion_mode,
        "--smoke",
    ]
    if identity_predictor:
        command.append("--identity-predictor")
    return command


def build_plan_payload(
    *,
    epochs: int = 3,
    device: str = "cpu",
    seed: int = 0,
    target_mode: str = "real-video",
    ego_motion_mode: str = "real-video",
) -> dict[str, Any]:
    """Return the paired smoke command plan without asserting a result."""

    return {
        "schema": SCHEMA,
        "probe_id": PROBE_ID,
        "lane_id": LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "evidence_grade": "plan_only_no_smoke_stats",
        "verdict": "pending_paired_smoke_stats",
        "paired_control_initialization": PAIRED_CONTROL_INITIALIZATION,
        **FALSE_AUTHORITY_FLAGS,
        "blockers": [
            "full_film_smoke_stats_missing",
            "identity_predictor_smoke_stats_missing",
            *SMOKE_PROXY_BLOCKERS,
        ],
        "paired_smoke_commands": [
            {
                "mode": "full_film_predictor",
                "identity_predictor": False,
                "smoke_target_mode": target_mode,
                "smoke_ego_motion_mode": ego_motion_mode,
                "output_dir": DEFAULT_FULL_OUTPUT_DIR,
                "stats_path": f"{DEFAULT_FULL_OUTPUT_DIR}/stats.json",
                "command": _smoke_command(
                    output_dir=DEFAULT_FULL_OUTPUT_DIR,
                    identity_predictor=False,
                    epochs=epochs,
                    device=device,
                    seed=seed,
                    target_mode=target_mode,
                    ego_motion_mode=ego_motion_mode,
                ),
            },
            {
                "mode": "identity_predictor",
                "identity_predictor": True,
                "smoke_target_mode": target_mode,
                "smoke_ego_motion_mode": ego_motion_mode,
                "output_dir": DEFAULT_IDENTITY_OUTPUT_DIR,
                "stats_path": f"{DEFAULT_IDENTITY_OUTPUT_DIR}/stats.json",
                "command": _smoke_command(
                    output_dir=DEFAULT_IDENTITY_OUTPUT_DIR,
                    identity_predictor=True,
                    epochs=epochs,
                    device=device,
                    seed=seed,
                    target_mode=target_mode,
                    ego_motion_mode=ego_motion_mode,
                ),
            },
        ],
        "reactivation_criteria": [
            "run both smoke commands from same git SHA and seed",
            "keep smoke_target_mode matched across both arms",
            "keep smoke_ego_motion_mode matched across both arms",
            "compare stats through this tool",
            "do not assert Z6 paradigm movement until paired contest CPU/CUDA exact eval exists",
            "keep full_main council-gated until real-video training path is implemented",
        ],
    }


def evaluate_stats_pair(
    *,
    full_stats_path: Path,
    identity_stats_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Compare paired Z6 smoke stats and emit a proxy-only verdict."""

    full = _load_json_object(full_stats_path)
    identity = _load_json_object(identity_stats_path)
    _validate_z6_smoke_stats(full, expected_identity=False, label="full")
    _validate_z6_smoke_stats(identity, expected_identity=True, label="identity")
    for key in (
        "epochs",
        "requested_epochs",
        "smoke_epoch_cap",
        "lambda_residual_entropy",
        "predictor_kernel_size",
        "smoke_target_mode",
        "smoke_ego_motion_mode",
    ):
        _require_same_config(full, identity, key)

    delta_loss = (
        float(identity["final_loss_proxy"]) - float(full["final_loss_proxy"])
    )
    delta_recon = _delta_optional(identity.get("final_recon"), full.get("final_recon"))
    delta_residual = _delta_optional(
        identity.get("final_residual"),
        full.get("final_residual"),
    )
    delta_archive = int(full["archive_bytes"]) - int(identity["archive_bytes"])
    if delta_loss > 1e-9:
        verdict = "full_film_predictor_proxy_lower_loss"
        proxy_preferred = "full_film_predictor"
    elif delta_loss < -1e-9:
        verdict = "identity_predictor_proxy_lower_loss"
        proxy_preferred = "identity_predictor"
    else:
        verdict = "indeterminate_tie_smoke_proxy_only"
        proxy_preferred = "tie"

    smoke_target_mode = full.get("smoke_target_mode")
    return {
        "schema": SCHEMA,
        "probe_id": PROBE_ID,
        "lane_id": LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "evidence_grade": _proxy_evidence_grade(smoke_target_mode),
        "verdict": verdict,
        "proxy_preferred_mode": proxy_preferred,
        "paired": True,
        "paired_control_initialization": PAIRED_CONTROL_INITIALIZATION,
        **FALSE_AUTHORITY_FLAGS,
        "blockers": _proxy_blockers(smoke_target_mode),
        "source_stats": [
            _mode_stats_row(
                full,
                path=full_stats_path,
                repo_root=repo_root,
                mode="full_film_predictor",
            ),
            _mode_stats_row(
                identity,
                path=identity_stats_path,
                repo_root=repo_root,
                mode="identity_predictor",
            ),
        ],
        "deltas": {
            "identity_minus_full_loss_proxy": delta_loss,
            "identity_minus_full_recon": delta_recon,
            "identity_minus_full_residual": delta_residual,
            "full_minus_identity_archive_bytes": delta_archive,
        },
        "result_review": {
            "classification": (
                "real_video_smoke_proxy_only"
                if smoke_target_mode == "real-video"
                else "synthetic_smoke_proxy_only"
            ),
            "paired_control_initialization": PAIRED_CONTROL_INITIALIZATION,
            "score_formula_recomputed": False,
            "score_formula_recompute_blocker": "no seg_dist/pose_dist/contest archive score fields in smoke stats",
            "component_score_authority": False,
            "contest_compliance_authority": False,
            "next_authoritative_gate": "paired contest CPU/CUDA exact eval after council-gated full_main implementation",
        },
        "reactivation_criteria": [
            "if full-FiLM wins proxy, implement real-video full_main and run paired smoke on contest video",
            "if identity wins proxy, keep Z6 predictive-coding claim blocked and diagnose predictor/curriculum",
            "only promote/rank/kill after byte-closed paired contest CPU/CUDA exact eval",
        ],
    }


def evaluate_archive_pair(
    *,
    full_archive_path: Path,
    identity_archive_path: Path,
    repo_root: Path = REPO_ROOT,
    full_eval_json_path: Path | None = None,
    identity_eval_json_path: Path | None = None,
    axis: str | None = None,
    decision_delta_s: float = 0.005,
    inflate_output_comparison: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a byte-closed full/identity Z6 archive pair.

    This consumes the real paired archives emitted by the full trainer. Exact
    eval JSON is optional; without it, the probe records archive custody and
    remains explicitly blocked from score, rank, or paradigm authority.
    """

    full_arc = _load_z6_archive(full_archive_path)
    identity_arc = _load_z6_archive(identity_archive_path)
    if _identity_flag(full_arc) is not False:
        raise ValueError("full archive must carry identity_predictor=false")
    if _identity_flag(identity_arc) is not True:
        raise ValueError("identity archive must carry identity_predictor=true")
    if identity_arc.meta.get("identity_predictor_disambiguator") is not True:
        raise ValueError(
            "identity archive must carry identity_predictor_disambiguator=true"
        )

    full_row = _archive_row(
        path=full_archive_path,
        arc=full_arc,
        repo_root=repo_root,
        mode="full_film_predictor",
        identity=False,
    )
    identity_row = _archive_row(
        path=identity_archive_path,
        arc=identity_arc,
        repo_root=repo_root,
        mode="identity_predictor",
        identity=True,
    )
    checks = _archive_pair_checks(full_arc, identity_arc)
    shared_sections_equal = all(checks.values())
    blockers: list[str] = []
    if not shared_sections_equal:
        blockers.append("paired_archive_shared_sections_mismatch")
    if full_row["zip_path"] is not None and full_row["zip_member_matches_path_bytes"] is not True:
        blockers.append("full_archive_zip_member_mismatch")
    if (
        identity_row["zip_path"] is not None
        and identity_row["zip_member_matches_path_bytes"] is not True
    ):
        blockers.append("identity_archive_zip_member_mismatch")

    deltas: dict[str, Any] = {
        "identity_minus_full_archive_bytes": int(identity_row["bytes"]) - int(full_row["bytes"]),
        "identity_minus_full_parsed_member_bytes": (
            int(identity_row["parsed_member_bytes"])
            - int(full_row["parsed_member_bytes"])
        ),
        "identity_minus_full_zip_bytes": (
            None
            if identity_row["zip_bytes"] is None or full_row["zip_bytes"] is None
            else int(identity_row["zip_bytes"]) - int(full_row["zip_bytes"])
        ),
        "identity_minus_full_contest_archive_bytes_basis": (
            int(identity_row["contest_archive_bytes_basis"])
            - int(full_row["contest_archive_bytes_basis"])
        ),
        "identity_minus_full_rate_term_basis": (
            25.0
            * (
                int(identity_row["contest_archive_bytes_basis"])
                - int(full_row["contest_archive_bytes_basis"])
            )
            / CONTEST_ARCHIVE_NORMALIZER_BYTES
        ),
    }

    exact_eval: dict[str, Any] | None = None
    verdict = "pending_paired_exact_eval_json"
    evidence_grade = "byte_closed_archive_pair_no_score"
    result_classification = "byte_closed_archive_pair_no_score"
    paired_eval_paths_provided = (
        full_eval_json_path is not None and identity_eval_json_path is not None
    )
    if (full_eval_json_path is None) != (identity_eval_json_path is None):
        raise ValueError(
            "--full-eval-json and --identity-eval-json must be supplied together"
        )
    if not paired_eval_paths_provided:
        blockers.extend(ARCHIVE_PAIR_BLOCKERS_NO_SCORE)
    else:
        full_eval = _score_fields_from_eval(
            path=full_eval_json_path,
            expected_archive_bytes=full_row["contest_archive_bytes_basis"],
            expected_archive_sha256=full_row["contest_archive_sha256_basis"],
            axis=axis,
            repo_root=repo_root,
        )
        identity_eval = _score_fields_from_eval(
            path=identity_eval_json_path,
            expected_archive_bytes=identity_row["contest_archive_bytes_basis"],
            expected_archive_sha256=identity_row["contest_archive_sha256_basis"],
            axis=axis,
            repo_root=repo_root,
        )
        exact_eval = {
            "full_film_predictor": full_eval,
            "identity_predictor": identity_eval,
        }
        for label, fields in (
            ("full", full_eval),
            ("identity", identity_eval),
        ):
            if fields["axis"] is None:
                blockers.append(f"{label}_exact_eval_axis_missing")
            if fields["score"] is None:
                blockers.append(f"{label}_exact_eval_score_missing")
            if fields["n_samples"] != 600:
                blockers.append(f"{label}_exact_eval_sample_count_not_600")
            if not fields["archive_bytes_match_expected"]:
                blockers.append(f"{label}_exact_eval_archive_bytes_mismatch")
            if not fields["archive_sha256_match_expected"]:
                blockers.append(f"{label}_exact_eval_archive_sha256_mismatch")
            if fields["score_recompute_matches_payload"] is False:
                blockers.append(f"{label}_exact_eval_formula_recompute_mismatch")
        if full_eval["axis"] != identity_eval["axis"]:
            blockers.append("paired_exact_eval_axis_mismatch")

        if not blockers and full_eval["score"] is not None and identity_eval["score"] is not None:
            full_minus_identity_score = float(full_eval["score"]) - float(identity_eval["score"])
            deltas["full_minus_identity_score"] = full_minus_identity_score
            deltas["identity_minus_full_score"] = -full_minus_identity_score
            if full_minus_identity_score <= -float(decision_delta_s):
                verdict = "full_film_predictor_exact_eval_lower_score"
                result_classification = "paired_exact_eval_full_film_lower_score"
            elif full_minus_identity_score >= float(decision_delta_s):
                verdict = "identity_predictor_exact_eval_lower_score"
                result_classification = "paired_exact_eval_identity_lower_score"
            else:
                verdict = "paired_exact_eval_indeterminate_within_delta_s"
                result_classification = "paired_exact_eval_delta_s_tie"
            evidence_grade = f"paired_exact_eval_{full_eval['axis']}"
        else:
            verdict = "blocked_paired_exact_eval_custody"
            evidence_grade = "paired_exact_eval_fail_closed"
            result_classification = "paired_exact_eval_custody_blocked"

    if not shared_sections_equal or any(
        blocker.endswith("_archive_zip_member_mismatch") for blocker in blockers
    ):
        verdict = "blocked_paired_archive_custody"
        evidence_grade = "byte_closed_archive_pair_fail_closed"
        result_classification = "paired_archive_custody_blocked"
    if (
        inflate_output_comparison is not None
        and inflate_output_comparison.get("runtime_output_changed") is False
    ):
        blockers.append("identity_predictor_switch_inflate_output_noop")
        if verdict != "blocked_paired_archive_custody":
            verdict = "blocked_identity_predictor_switch_inflate_output_noop"
            evidence_grade = "inflate_output_comparison_fail_closed"
            result_classification = "identity_predictor_switch_runtime_noop"

    return {
        "schema": SCHEMA,
        "probe_id": PROBE_ID,
        "lane_id": LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "evidence_grade": evidence_grade,
        "verdict": verdict,
        "paired": True,
        "paired_control_initialization": PAIRED_CONTROL_INITIALIZATION,
        "decision_delta_s": float(decision_delta_s),
        **FALSE_AUTHORITY_FLAGS,
        "blockers": blockers,
        "source_archives": [full_row, identity_row],
        "paired_archive_checks": checks,
        "runtime_custody": (
            dict(inflate_output_comparison["runtime_custody"])
            if inflate_output_comparison is not None
            and isinstance(inflate_output_comparison.get("runtime_custody"), Mapping)
            else None
        ),
        "inflate_output_comparison": (
            dict(inflate_output_comparison)
            if inflate_output_comparison is not None
            else None
        ),
        "exact_eval": exact_eval,
        "deltas": deltas,
        "result_review": {
            "classification": result_classification,
            "score_formula_recomputed": (
                exact_eval is not None
                and all(
                    row.get("score_recomputed_from_components") is not None
                    for row in exact_eval.values()
                )
            ),
            "score_claim_authority": False,
            "component_score_authority": exact_eval is not None and not blockers,
            "contest_compliance_authority": exact_eval is not None and not blockers,
            "runtime_custody_available": (
                inflate_output_comparison is not None
                and isinstance(inflate_output_comparison.get("runtime_custody"), Mapping)
            ),
            "inflate_output_comparison_available": (
                inflate_output_comparison is not None
            ),
            "identity_predictor_switch_changes_inflate_output": (
                inflate_output_comparison.get("runtime_output_changed")
                if inflate_output_comparison is not None
                else None
            ),
            "full_minus_identity_score_semantics": (
                "negative means full-FiLM lower score; positive means identity lower score"
            ),
            "next_authoritative_gate": (
                "paired contest-CUDA exact eval with matching ZIP custody"
                if exact_eval is None
                else "operator promotion review; probe itself remains non-authoritative"
            ),
        },
        "reactivation_criteria": [
            "provide both paired contest_auth_eval JSON files for the exact same ZIP sidecars",
            "keep axis labels adjacent to all score language",
            "treat full_minus_identity_score <= -decision_delta_s as a full-FiLM win",
            "do not promote, rank, or kill from this probe without operator review",
        ],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Render a compact operator-facing Markdown report."""

    lines = [
        "# L5 v2 Z6 identity-predictor disambiguator",
        "",
        f"- schema: `{payload.get('schema')}`",
        f"- probe_id: `{payload.get('probe_id')}`",
        f"- lane_id: `{payload.get('lane_id')}`",
        f"- evidence_grade: `{payload.get('evidence_grade')}`",
        f"- verdict: `{payload.get('verdict')}`",
        f"- paired_control_initialization: `{payload.get('paired_control_initialization')}`",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        "- rank_or_kill_eligible: `false`",
        "- ready_for_exact_eval_dispatch: `false`",
        "- ready_for_paid_dispatch: `false`",
        "- paradigm_claim_allowed: `false`",
        "",
        "This report is a Z6-specific probe surface. It can route the next "
        "engineering action, but it is not contest score evidence.",
    ]
    commands = payload.get("paired_smoke_commands")
    if isinstance(commands, list):
        lines.extend(["", "## Paired Smoke Commands"])
        for row in commands:
            if not isinstance(row, Mapping):
                continue
            command = row.get("command")
            rendered = " ".join(command) if isinstance(command, list) else ""
            lines.extend(
                [
                    "",
                    f"### {row.get('mode')}",
                    "",
                    f"- identity_predictor: `{row.get('identity_predictor')}`",
                    f"- smoke_target_mode: `{row.get('smoke_target_mode')}`",
                    f"- smoke_ego_motion_mode: `{row.get('smoke_ego_motion_mode')}`",
                    f"- stats_path: `{row.get('stats_path')}`",
                    "",
                    "```bash",
                    rendered,
                    "```",
                ]
            )
    stats = payload.get("source_stats")
    if isinstance(stats, list):
        lines.extend(["", "## Source Stats"])
        for row in stats:
            if not isinstance(row, Mapping):
                continue
            lines.extend(
                [
                    "",
                    f"### {row.get('mode')}",
                    "",
                    f"- path: `{row.get('path')}`",
                    f"- sha256: `{row.get('sha256')}`",
                    f"- final_loss_proxy: `{row.get('final_loss_proxy')}`",
                    f"- final_recon: `{row.get('final_recon')}`",
                    f"- final_residual: `{row.get('final_residual')}`",
                    f"- archive_bytes: `{row.get('archive_bytes')}`",
                    f"- paired_control_initialization: `{row.get('paired_control_initialization')}`",
                    f"- smoke_target_mode: `{row.get('smoke_target_mode')}`",
                    f"- smoke_ego_motion_mode: `{row.get('smoke_ego_motion_mode')}`",
                    f"- ego_motion_nonzero_fraction: `{row.get('ego_motion_nonzero_fraction')}`",
                ]
            )
    archives = payload.get("source_archives")
    if isinstance(archives, list):
        lines.extend(["", "## Source Archives"])
        for row in archives:
            if not isinstance(row, Mapping):
                continue
            lines.extend(
                [
                    "",
                    f"### {row.get('mode')}",
                    "",
                    f"- path: `{row.get('path')}`",
                    f"- bytes: `{row.get('bytes')}`",
                    f"- sha256: `{row.get('sha256')}`",
                    f"- zip_path: `{row.get('zip_path')}`",
                    f"- zip_bytes: `{row.get('zip_bytes')}`",
                    f"- zip_sha256: `{row.get('zip_sha256')}`",
                    f"- zip_members: `{row.get('zip_members')}`",
                    f"- zip_member_matches_path_bytes: `{row.get('zip_member_matches_path_bytes')}`",
                    f"- contest_archive_bytes_basis: `{row.get('contest_archive_bytes_basis')}`",
                    f"- identity_predictor: `{row.get('identity_predictor')}`",
                    f"- identity_predictor_disambiguator: `{row.get('identity_predictor_disambiguator')}`",
                    f"- predictor_state_dict_key_count: `{row.get('predictor_state_dict_key_count')}`",
                    f"- num_pairs: `{row.get('num_pairs')}`",
                    f"- ego_motion_dim: `{row.get('ego_motion_dim')}`",
                    f"- predictor_architecture: `{row.get('predictor_architecture')}`",
                ]
            )
    checks = payload.get("paired_archive_checks")
    if isinstance(checks, Mapping):
        lines.extend(["", "## Paired Archive Checks"])
        for key, value in checks.items():
            lines.append(f"- {key}: `{value}`")
    runtime_custody = payload.get("runtime_custody")
    if isinstance(runtime_custody, Mapping):
        lines.extend(
            [
                "",
                "## Runtime Custody",
                f"- schema: `{runtime_custody.get('schema')}`",
                f"- runtime_root: `{runtime_custody.get('runtime_root')}`",
                f"- entrypoint: `{runtime_custody.get('entrypoint')}`",
                f"- aggregate_sha256: `{runtime_custody.get('aggregate_sha256')}`",
                f"- file_count: `{runtime_custody.get('file_count')}`",
                f"- total_bytes: `{runtime_custody.get('total_bytes')}`",
                "- archive_payloads_excluded_from_runtime_hash: "
                f"`{runtime_custody.get('archive_payloads_excluded_from_runtime_hash')}`",
                "- python_cache_files_excluded_from_runtime_hash: "
                f"`{len(runtime_custody.get('python_cache_files_excluded_from_runtime_hash') or [])}`",
            ]
        )
    inflate_output = payload.get("inflate_output_comparison")
    if isinstance(inflate_output, Mapping):
        lines.extend(
            [
                "",
                "## Inflate Output Comparison",
                f"- evidence_axis: `{inflate_output.get('evidence_axis')}`",
                f"- evidence_grade: `{inflate_output.get('evidence_grade')}`",
                f"- runtime_output_changed: `{inflate_output.get('runtime_output_changed')}`",
                f"- same_output_file_set: `{inflate_output.get('same_output_file_set')}`",
                f"- same_output_aggregate_sha256: `{inflate_output.get('same_output_aggregate_sha256')}`",
                f"- total_byte_differences: `{inflate_output.get('total_byte_differences')}`",
                f"- inflate_sh_path: `{inflate_output.get('inflate_sh_path')}`",
                f"- file_list_path: `{inflate_output.get('file_list_path')}`",
                f"- output_root: `{inflate_output.get('output_root')}`",
            ]
        )
        full_tree = inflate_output.get("full_output_tree")
        identity_tree = inflate_output.get("identity_output_tree")
        if isinstance(full_tree, Mapping) and isinstance(identity_tree, Mapping):
            lines.extend(
                [
                    f"- full_output_aggregate_sha256: `{full_tree.get('aggregate_sha256')}`",
                    f"- identity_output_aggregate_sha256: `{identity_tree.get('aggregate_sha256')}`",
                    f"- full_output_total_bytes: `{full_tree.get('total_bytes')}`",
                    f"- identity_output_total_bytes: `{identity_tree.get('total_bytes')}`",
                ]
            )
    exact_eval = payload.get("exact_eval")
    if isinstance(exact_eval, Mapping):
        lines.extend(["", "## Exact Eval Inputs"])
        for mode, row in exact_eval.items():
            if not isinstance(row, Mapping):
                continue
            lines.extend(
                [
                    "",
                    f"### {mode}",
                    "",
                    f"- path: `{row.get('path')}`",
                    f"- axis: `{row.get('axis')}`",
                    f"- score: `{row.get('score')}`",
                    f"- score_recomputed_from_components: `{row.get('score_recomputed_from_components')}`",
                    f"- avg_segnet_dist: `{row.get('avg_segnet_dist')}`",
                    f"- avg_posenet_dist: `{row.get('avg_posenet_dist')}`",
                    f"- archive_size_bytes: `{row.get('archive_size_bytes')}`",
                    f"- archive_sha256: `{row.get('archive_sha256')}`",
                    f"- n_samples: `{row.get('n_samples')}`",
                    f"- archive_bytes_match_expected: `{row.get('archive_bytes_match_expected')}`",
                    f"- archive_sha256_match_expected: `{row.get('archive_sha256_match_expected')}`",
                ]
            )
    deltas = payload.get("deltas")
    if isinstance(deltas, Mapping):
        lines.extend(["", "## Deltas"])
        for key, value in deltas.items():
            lines.append(f"- {key}: `{value}`")
    blockers = payload.get("blockers")
    if isinstance(blockers, list):
        lines.extend(["", "## Blockers"])
        for blocker in blockers:
            lines.append(f"- `{blocker}`")
    reactivation = payload.get("reactivation_criteria")
    if isinstance(reactivation, list):
        lines.extend(["", "## Reactivation Criteria"])
        for criterion in reactivation:
            lines.append(f"- {criterion}")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--full-stats", type=Path, default=None)
    parser.add_argument("--identity-stats", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--full-archive", type=Path, default=None)
    parser.add_argument("--identity-archive", type=Path, default=None)
    parser.add_argument("--full-eval-json", type=Path, default=None)
    parser.add_argument("--identity-eval-json", type=Path, default=None)
    parser.add_argument("--axis", default=None)
    parser.add_argument("--decision-delta-s", type=float, default=0.005)
    parser.add_argument(
        "--inflate-sh",
        type=Path,
        default=None,
        help=(
            "Optional inflate.sh runtime for local full-vs-identity output "
            "comparison in archive-pair mode."
        ),
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        default=None,
        help="File list passed to --inflate-sh for output comparison.",
    )
    parser.add_argument(
        "--inflate-output-root",
        type=Path,
        default=None,
        help=(
            "Absent/empty output directory where staged archives and raw "
            "outputs for the local comparison are written."
        ),
    )
    parser.add_argument(
        "--inflate-python",
        default=None,
        help="PYTHON value exported to inflate.sh; defaults to this interpreter.",
    )
    parser.add_argument("--inflate-timeout-seconds", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smoke-target-mode",
        choices=("synthetic", "real-video"),
        default="real-video",
    )
    parser.add_argument(
        "--smoke-ego-motion-mode",
        choices=("ramp", "zero", "random", "real-video"),
        default="real-video",
    )
    parser.add_argument("--output-json", type=Path, default=Path(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", type=Path, default=Path(DEFAULT_OUTPUT_MD))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    try:
        if (args.full_stats is None) != (args.identity_stats is None):
            raise ValueError("--full-stats and --identity-stats must be supplied together")
        if (args.full_archive is None) != (args.identity_archive is None):
            raise ValueError(
                "--full-archive and --identity-archive must be supplied together"
            )
        archive_mode = args.run_dir is not None or args.full_archive is not None
        stats_mode = args.full_stats is not None
        if archive_mode and stats_mode:
            raise ValueError("archive-pair mode and stats-pair mode are mutually exclusive")
        if args.run_dir is not None and args.full_archive is not None:
            raise ValueError("--run-dir cannot be combined with explicit archive paths")
        if not archive_mode and (args.full_eval_json is not None or args.identity_eval_json is not None):
            raise ValueError("exact eval JSON inputs require archive-pair mode")
        inflate_compare_requested = any(
            value is not None
            for value in (args.inflate_sh, args.file_list, args.inflate_output_root)
        )
        if inflate_compare_requested and not archive_mode:
            raise ValueError("inflate output comparison requires archive-pair mode")
        if inflate_compare_requested and not all(
            value is not None
            for value in (args.inflate_sh, args.file_list, args.inflate_output_root)
        ):
            raise ValueError(
                "--inflate-sh, --file-list, and --inflate-output-root must be "
                "supplied together"
            )
        if not archive_mode and not stats_mode:
            payload = build_plan_payload(
                epochs=args.epochs,
                device=args.device,
                seed=args.seed,
                target_mode=args.smoke_target_mode,
                ego_motion_mode=args.smoke_ego_motion_mode,
            )
        elif stats_mode:
            full_stats_path = _resolve_repo_path(args.full_stats, repo_root=repo_root)
            identity_stats_path = _resolve_repo_path(
                args.identity_stats,
                repo_root=repo_root,
            )
            payload = evaluate_stats_pair(
                full_stats_path=full_stats_path,
                identity_stats_path=identity_stats_path,
                repo_root=repo_root,
            )
        else:
            if args.run_dir is not None:
                run_dir = _resolve_repo_path(args.run_dir, repo_root=repo_root)
                full_archive_path, identity_archive_path = (
                    _default_archive_pair_from_run_dir(run_dir)
                )
                default_full_eval_json_path, default_identity_eval_json_path = (
                    _default_eval_pair_from_run_dir(run_dir)
                )
            else:
                default_full_eval_json_path = None
                default_identity_eval_json_path = None
                full_archive_path = _resolve_repo_path(
                    args.full_archive,
                    repo_root=repo_root,
                )
                identity_archive_path = _resolve_repo_path(
                    args.identity_archive,
                    repo_root=repo_root,
                )
            full_eval_json_path = (
                _resolve_repo_path(args.full_eval_json, repo_root=repo_root)
                if args.full_eval_json is not None
                else default_full_eval_json_path
            )
            identity_eval_json_path = (
                _resolve_repo_path(args.identity_eval_json, repo_root=repo_root)
                if args.identity_eval_json is not None
                else default_identity_eval_json_path
            )
            inflate_output_comparison = None
            if inflate_compare_requested:
                inflate_output_comparison = compare_inflate_output_pair(
                    full_archive_path=full_archive_path,
                    identity_archive_path=identity_archive_path,
                    inflate_sh_path=_resolve_repo_path(
                        args.inflate_sh,
                        repo_root=repo_root,
                    ),
                    file_list_path=_resolve_repo_path(
                        args.file_list,
                        repo_root=repo_root,
                    ),
                    output_root=_resolve_repo_path(
                        args.inflate_output_root,
                        repo_root=repo_root,
                    ),
                    repo_root=repo_root,
                    python_executable=args.inflate_python,
                    timeout_seconds=args.inflate_timeout_seconds,
                )
            payload = evaluate_archive_pair(
                full_archive_path=full_archive_path,
                identity_archive_path=identity_archive_path,
                repo_root=repo_root,
                full_eval_json_path=full_eval_json_path,
                identity_eval_json_path=identity_eval_json_path,
                axis=args.axis,
                decision_delta_s=args.decision_delta_s,
                inflate_output_comparison=inflate_output_comparison,
            )
        output_json = _resolve_repo_path(args.output_json, repo_root=repo_root)
        output_md = _resolve_repo_path(args.output_md, repo_root=repo_root)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(payload), encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[z6-disambiguator] FATAL: {exc}", file=sys.stderr)
        return 2

    print(
        "[z6-disambiguator] "
        f"verdict={payload['verdict']} "
        f"evidence_grade={payload['evidence_grade']} "
        "score_claim=false promotion_eligible=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
