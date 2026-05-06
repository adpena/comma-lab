#!/usr/bin/env python
"""Static PR86 HPAC/token anatomy and PR85 transplant planner.

This tool is intentionally local-only. It does not run scorer code, does not
dispatch remote jobs, and does not mutate dispatch state. Its output is a
forensic/planning artifact, not score evidence.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import importlib.metadata
import io
import json
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback.
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PR86_DIR = REPO_ROOT / "experiments/results/public_pr86_intake_20260504_codex"
DEFAULT_PR86_ARCHIVE = DEFAULT_PR86_DIR / "archive.zip"
DEFAULT_PR85_PROFILE = (
    REPO_ROOT
    / "experiments/results/public_pr85_intake_20260503_codex/profile_pr85_bundle.json"
)
DEFAULT_REPLAY_DIR = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_public_pr86_hpac_t4_hedge_20260504T0152Z"
)
DEFAULT_JSON_OUT = DEFAULT_PR86_DIR / "pr86_hpac_token_anatomy_forensics.json"
DEFAULT_MD_OUT = DEFAULT_PR86_DIR / "pr86_hpac_token_anatomy_forensics.md"

EXPECTED_PR86_ARCHIVE_BYTES = 207_579
EXPECTED_PR86_ARCHIVE_SHA256 = (
    "e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef"
)
PR85_T4_BEST_SCORE = 0.25806611029397786
REQUIRED_PR86_MEMBERS = (
    "master.pt.gz",
    "slave.pt.gz",
    "hpac.pt.ppmd",
    "tokens.bin",
    "meta.pt",
)
MEMBER_ROLES = {
    "master.pt.gz": "TokenRendererV62 master weights",
    "slave.pt.gz": "ShrinkSingleNeRV slave weights",
    "hpac.pt.ppmd": "HPACMini entropy model weights",
    "tokens.bin": "constriction queue-coded token stream",
    "meta.pt": "runtime metadata",
}


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _simple_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_simple_scalar(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _simple_scalar(v) for k, v in value.items()}
    return repr(value)


def _is_unsafe_zip_name(name: str) -> bool:
    path = Path(name)
    return (
        not name
        or name.startswith("/")
        or "\\" in name
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or name.startswith("__MACOSX/")
        or "/__MACOSX/" in name
        or Path(name).name.startswith("._")
    )


def inspect_archive_members(
    archive: Path,
    *,
    expected_bytes: int | None = EXPECTED_PR86_ARCHIVE_BYTES,
    expected_sha256: str | None = EXPECTED_PR86_ARCHIVE_SHA256,
) -> dict[str, Any]:
    """Return deterministic PR86 archive member facts and fail-closed status."""
    archive = archive.resolve()
    archive_bytes = archive.stat().st_size
    archive_sha = sha256_path(archive)
    names: list[str] = []
    members: list[dict[str, Any]] = []

    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            data = zf.read(info.filename)
            names.append(info.filename)
            members.append(
                {
                    "name": info.filename,
                    "role": MEMBER_ROLES.get(info.filename, "unexpected"),
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "zip_compress_type": info.compress_type,
                    "zip_compress_type_name": _zip_compress_name(info.compress_type),
                    "crc32_hex": f"{info.CRC:08x}",
                    "sha256": sha256_bytes(data),
                    "header_offset": info.header_offset,
                    "byte_share_of_archive": round(info.file_size / archive_bytes, 6),
                    "magic_hex": data[:12].hex(),
                    "unsafe_name": _is_unsafe_zip_name(info.filename),
                }
            )

    counts = Counter(names)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    required = set(REQUIRED_PR86_MEMBERS)
    present = set(names)
    missing = sorted(required - present)
    unexpected = sorted(present - required)
    unsafe = sorted(name for name in names if _is_unsafe_zip_name(name))
    all_stored = all(row["zip_compress_type"] == zipfile.ZIP_STORED for row in members)
    exact_required = (
        not duplicates and not missing and not unexpected and not unsafe and all_stored
    )

    return {
        "path": repo_rel(archive),
        "exists": archive.is_file(),
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "expected_archive_bytes": expected_bytes,
        "expected_archive_sha256": expected_sha256,
        "expected_identity_match": (
            (expected_bytes is None or archive_bytes == expected_bytes)
            and (expected_sha256 is None or archive_sha == expected_sha256)
        ),
        "member_count": len(members),
        "required_member_names": list(REQUIRED_PR86_MEMBERS),
        "duplicate_member_names": duplicates,
        "missing_required_members": missing,
        "unexpected_members": unexpected,
        "unsafe_member_names": unsafe,
        "all_members_zip_stored": all_stored,
        "sidecar_assumption_status": (
            "passed_exact_required_member_set" if exact_required else "failed_closed"
        ),
        "promotable_member_contract": exact_required,
        "members": members,
    }


def _zip_compress_name(code: int) -> str:
    return {
        zipfile.ZIP_STORED: "stored",
        zipfile.ZIP_DEFLATED: "deflated",
        zipfile.ZIP_BZIP2: "bzip2",
        zipfile.ZIP_LZMA: "lzma",
    }.get(code, f"unknown_{code}")


def inspect_torch_zip_blob(blob: bytes) -> dict[str, Any]:
    if not blob.startswith(b"PK\x03\x04"):
        return {"is_torch_zip_like": False, "magic_hex": blob[:12].hex()}
    try:
        with zipfile.ZipFile(io.BytesIO(blob), "r") as zf:
            infos = zf.infolist()
            names = [info.filename for info in infos]
            return {
                "is_torch_zip_like": True,
                "entry_count": len(infos),
                "entry_names_prefix": names[:20],
                "total_entry_file_size": sum(info.file_size for info in infos),
                "has_data_pkl": any(name.endswith("data.pkl") for name in names),
                "tensor_storage_entry_count": sum("/data/" in name for name in names),
            }
    except Exception as exc:
        return {
            "is_torch_zip_like": False,
            "magic_hex": blob[:12].hex(),
            "inspect_error_type": type(exc).__name__,
            "inspect_error": str(exc),
        }


def inspect_torch_object(blob: bytes) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent.
        return {
            "torch_load_attempted": False,
            "status": "missing_torch",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    try:
        obj = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
    except Exception as exc:
        return {
            "torch_load_attempted": True,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    if isinstance(obj, dict):
        tensor_rows: list[dict[str, Any]] = []
        dtype_counts: Counter[str] = Counter()
        non_tensor_keys: list[str] = []
        simple_meta: dict[str, Any] = {}
        for key, value in obj.items():
            if torch.is_tensor(value):
                dtype_counts[str(value.dtype)] += 1
                if len(tensor_rows) < 24:
                    tensor_rows.append(
                        {
                            "key": str(key),
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "numel": int(value.numel()),
                        }
                    )
            else:
                non_tensor_keys.append(str(key))
                if len(simple_meta) < 64:
                    simple_meta[str(key)] = _simple_scalar(value)
        return {
            "torch_load_attempted": True,
            "status": "loaded",
            "object_type": "dict",
            "key_count": len(obj),
            "tensor_key_count": sum(dtype_counts.values()),
            "non_tensor_key_count": len(non_tensor_keys),
            "dtype_counts": dict(sorted(dtype_counts.items())),
            "tensor_samples": tensor_rows,
            "non_tensor_keys_prefix": sorted(non_tensor_keys)[:32],
            "simple_meta": simple_meta,
        }

    return {
        "torch_load_attempted": True,
        "status": "loaded",
        "object_type": type(obj).__name__,
        "repr_prefix": repr(obj)[:256],
    }


def inspect_member_payload_layers(archive: Path) -> dict[str, Any]:
    """Inspect nested PR86 payload layers without running model inference."""
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive, "r") as zf:
        for name in REQUIRED_PR86_MEMBERS:
            try:
                data = zf.read(name)
            except KeyError:
                rows.append({"name": name, "status": "missing"})
                continue
            row: dict[str, Any] = {
                "name": name,
                "role": MEMBER_ROLES[name],
                "encoded_bytes": len(data),
                "encoded_sha256": sha256_bytes(data),
                "encoded_magic_hex": data[:12].hex(),
            }
            if name.endswith(".pt.gz"):
                try:
                    decoded = gzip.decompress(data)
                    row["inner_codec"] = "gzip"
                    row["decoded_bytes"] = len(decoded)
                    row["decoded_sha256"] = sha256_bytes(decoded)
                    row["decoded_magic_hex"] = decoded[:12].hex()
                    row["torch_zip"] = inspect_torch_zip_blob(decoded)
                    row["torch_object"] = inspect_torch_object(decoded)
                except Exception as exc:
                    row["inner_codec"] = "gzip"
                    row["status"] = "decode_error"
                    row["error_type"] = type(exc).__name__
                    row["error"] = str(exc)
            elif name == "hpac.pt.ppmd":
                row.update(inspect_hpac_ppmd(data))
            elif name == "tokens.bin":
                row.update(inspect_tokens_bin(data))
            elif name == "meta.pt":
                row["torch_zip"] = inspect_torch_zip_blob(data)
                row["torch_object"] = inspect_torch_object(data)
                row["meta_contract"] = meta_contract_from_torch_summary(row["torch_object"])
            rows.append(row)
    return {"members": rows}


def inspect_hpac_ppmd(data: bytes) -> dict[str, Any]:
    try:
        import pyppmd
    except Exception as exc:  # pragma: no cover - environment dependent.
        return {
            "inner_codec": "ppmd",
            "decode_attempted": False,
            "status": "missing_pyppmd",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    try:
        decoded = pyppmd.decompress(data, max_order=4, mem_size=16 << 20)
    except Exception as exc:
        return {
            "inner_codec": "ppmd",
            "decode_attempted": True,
            "status": "decode_error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "inner_codec": "ppmd",
        "decode_attempted": True,
        "status": "decoded",
        "ppmd_max_order": 4,
        "ppmd_mem_size": 16 << 20,
        "decoded_bytes": len(decoded),
        "decoded_sha256": sha256_bytes(decoded),
        "decoded_magic_hex": decoded[:12].hex(),
        "torch_zip": inspect_torch_zip_blob(decoded),
        "torch_object": inspect_torch_object(decoded),
    }


def inspect_tokens_bin(data: bytes) -> dict[str, Any]:
    return {
        "queue_word_dtype": "uint32",
        "uint32_aligned": len(data) % 4 == 0,
        "uint32_word_count": len(data) // 4,
        "constriction_queue_decoder_contract": (
            "np.frombuffer(tokens.bin, dtype=np.uint32) -> "
            "constriction.stream.queue.RangeDecoder"
        ),
        "status": "structurally_parseable" if len(data) % 4 == 0 else "failed_closed",
    }


def meta_contract_from_torch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    meta = summary.get("simple_meta", {}) if summary.get("status") == "loaded" else {}
    if not meta:
        return {"status": "missing_or_unparsed"}
    expected = {
        "N": 600,
        "mode": "hpac",
        "P": 32,
        "delta": 2,
        "ch": 64,
        "hpac_d_film": 8,
        "use_spm": True,
        "ppmd_max_order": 4,
    }
    mismatches = {
        key: {"expected": value, "actual": meta.get(key)}
        for key, value in expected.items()
        if meta.get(key) != value
    }
    return {
        "status": "matches_expected_pr86_hpac_meta" if not mismatches else "mismatch",
        "expected_fields": expected,
        "mismatches": mismatches,
        "tokens_bpp": meta.get("tokens_bpp"),
        "slave_channels": meta.get("slave_channels"),
        "slave_d_lat": meta.get("slave_d_lat"),
        "gt_decoder": meta.get("gt_decoder"),
    }


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


class _CallFinder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        self.calls.append(
            {
                "name": name,
                "lineno": getattr(node, "lineno", None),
                "args": [_unparse(arg) for arg in node.args],
            }
        )
        self.generic_visit(node)


def find_calls(path: Path, call_name: str) -> list[dict[str, Any]]:
    tree = ast.parse(read_text_if_exists(path), filename=str(path))
    finder = _CallFinder()
    finder.visit(tree)
    return [row for row in finder.calls if row["name"] == call_name]


def parse_imports(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    tree = ast.parse(read_text_if_exists(path), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                rows.append(
                    {
                        "module": alias.name.split(".")[0],
                        "import": alias.name,
                        "kind": "import",
                        "lineno": getattr(node, "lineno", None),
                    }
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            rows.append(
                {
                    "module": node.module.split(".")[0],
                    "import": node.module,
                    "kind": "from",
                    "lineno": getattr(node, "lineno", None),
                }
            )
    return sorted(rows, key=lambda row: (row["module"], row["lineno"] or 0))


def parse_classes_and_functions(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        return {"classes": [], "functions": []}
    tree = ast.parse(read_text_if_exists(path), filename=str(path))
    return {
        "classes": [node.name for node in tree.body if isinstance(node, ast.ClassDef)],
        "functions": [node.name for node in tree.body if isinstance(node, ast.FunctionDef)],
    }


def pyproject_dependency_entries(pyproject: Path) -> list[str]:
    if not pyproject.is_file():
        return []
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception:
        return []
    project_deps = data.get("project", {}).get("dependencies", [])
    return [str(dep) for dep in project_deps]


def analyze_source_contract(pr86_dir: Path) -> dict[str, Any]:
    inflate_py = pr86_dir / "inflate.py"
    inflate_sh = pr86_dir / "inflate.sh"
    training_archive = pr86_dir / "training/archive.py"
    training_hpac = pr86_dir / "training/hpac.py"
    training_readme = pr86_dir / "training/README.md"
    pyproject = REPO_ROOT / "pyproject.toml"

    inflate_text = read_text_if_exists(inflate_py)
    archive_text = read_text_if_exists(training_archive)
    hpac_text = read_text_if_exists(training_hpac)
    readme_text = read_text_if_exists(training_readme)
    source_text = inflate_text + "\n" + archive_text + "\n" + hpac_text

    write_tokens_calls = find_calls(training_archive, "write_tokens")
    archive_write_tokens_arg = None
    if write_tokens_calls and len(write_tokens_calls[0]["args"]) >= 2:
        archive_write_tokens_arg = write_tokens_calls[0]["args"][1]
    hpac_decode_calls = find_calls(inflate_py, "decompress_tokens_hpac")
    hpac_decode_device_arg = None
    if hpac_decode_calls and len(hpac_decode_calls[-1]["args"]) > 8:
        hpac_decode_device_arg = hpac_decode_calls[-1]["args"][8]

    training_uses_residual_targets = (
        "compute_residuals(gt_tokens)" in hpac_text
        and "res = gt_residuals" in hpac_text
        and "res[i] = (tok[i] - tok[i-1]) mod" in hpac_text
    )
    archive_computes_raw_gt = "gt = compute_gt_tokens()" in archive_text
    archive_encodes_raw_gt = archive_write_tokens_arg == "gt"
    inflate_reconstructs_residuals = (
        "compute_residuals" in inflate_text
        or ("decoded_prev" in inflate_text and "+ decoded" in inflate_text and "% NUM_CLASSES" in inflate_text)
    )
    if archive_computes_raw_gt and archive_encodes_raw_gt and not inflate_reconstructs_residuals:
        submitted_encoding = "raw_tokens"
    elif training_uses_residual_targets and inflate_reconstructs_residuals:
        submitted_encoding = "residual_tokens"
    else:
        submitted_encoding = "ambiguous"

    dependencies = pyproject_dependency_entries(pyproject)
    local_runtime_deps = {
        "python": sys.version.split()[0],
        "numpy": package_version("numpy"),
        "torch": package_version("torch"),
        "constriction": package_version("constriction"),
        "pyppmd": package_version("pyppmd"),
    }
    code_member_refs = [
        name for name in REQUIRED_PR86_MEMBERS if name in inflate_text or name in archive_text
    ]

    return {
        "paths": {
            "inflate_py": repo_rel(inflate_py),
            "inflate_sh": repo_rel(inflate_sh),
            "training_archive_py": repo_rel(training_archive),
            "training_hpac_py": repo_rel(training_hpac),
            "training_readme": repo_rel(training_readme),
            "pyproject": repo_rel(pyproject),
        },
        "inflate_imports": parse_imports(inflate_py),
        "inflate_symbols": parse_classes_and_functions(inflate_py),
        "inflate_sh_lines": [
            line.strip()
            for line in read_text_if_exists(inflate_sh).splitlines()
            if line.strip()
        ],
        "local_runtime_dependency_versions": local_runtime_deps,
        "pyproject_pr86_dependency_entries": [
            dep for dep in dependencies if "constriction" in dep or "pyppmd" in dep
        ],
        "archive_member_references_in_code": code_member_refs,
        "required_archive_members_referenced": sorted(code_member_refs) == sorted(REQUIRED_PR86_MEMBERS),
        "token_hpac_decode_contract": {
            "training_objective": "residual_tokens" if training_uses_residual_targets else "unknown",
            "training_residual_definition_present": training_uses_residual_targets,
            "archive_compute_gt_tokens_call_present": archive_computes_raw_gt,
            "archive_write_tokens_second_arg": archive_write_tokens_arg,
            "inflate_reconstructs_residuals": inflate_reconstructs_residuals,
            "submitted_archive_token_encoding": submitted_encoding,
            "range_encoder_api_present": "constriction.stream.queue.RangeEncoder" in source_text,
            "range_decoder_api_present": "constriction.stream.queue.RangeDecoder" in source_text,
            "categorical_perfect_false_present": "perfect=False" in source_text,
            "probability_clip_eps": "1e-7" if "1e-7" in source_text else None,
            "explicit_16384_grid_in_archive_or_inflate": "16384" in (archive_text + "\n" + inflate_text),
            "readme_mentions_16384_grid": "16384" in readme_text,
        },
        "device_contract": {
            "comment_claims_hpac_cpu": "Force HPAC decode onto CPU" in inflate_text,
            "main_selects_cuda_when_available": (
                '"cuda" if torch.cuda.is_available() else "cpu"' in inflate_text
            ),
            "decompress_tokens_hpac_device_arg": hpac_decode_device_arg,
            "comment_code_mismatch": (
                "Force HPAC decode onto CPU" in inflate_text
                and hpac_decode_device_arg == "device"
            ),
        },
        "sidecar_dependency_summary": {
            "archive_carries_model_weights": all(
                name in code_member_refs for name in ("master.pt.gz", "slave.pt.gz", "hpac.pt.ppmd")
            ),
            "archive_carries_token_stream": "tokens.bin" in code_member_refs,
            "archive_carries_metadata": "meta.pt" in code_member_refs,
            "external_python_packages_required": [
                name
                for name in ("numpy", "torch", "constriction", "pyppmd")
                if name in {row["module"] for row in parse_imports(inflate_py)}
            ],
            "score_affecting_sidecar_files_detected": [],
            "note": (
                "Static source inspection found the score-affecting model/token "
                "payloads referenced by inflate.py inside archive.zip. The "
                "runtime still depends on submission code plus Python packages."
            ),
        },
    }


def read_pr_view(pr86_dir: Path) -> dict[str, Any]:
    view = read_json_if_exists(pr86_dir / "pr86_view.json") or {}
    body = str(view.get("body", ""))
    claimed = {
        "avg_posenet_dist": _regex_float(body, r"Average PoseNet Distortion:\s*([0-9.]+)"),
        "avg_segnet_dist": _regex_float(body, r"Average SegNet Distortion:\s*([0-9.]+)"),
        "archive_bytes": _regex_int(body, r"Submission file size:\s*([0-9,]+) bytes"),
        "display_score": _regex_float(body, r"Final score:.*?=\s*([0-9.]+)"),
    }
    return {
        "number": view.get("number"),
        "title": view.get("title"),
        "url": view.get("url"),
        "state": view.get("state"),
        "updated_at": view.get("updatedAt"),
        "head_ref": view.get("headRefName"),
        "commit_oids": [row.get("oid") for row in view.get("commits", [])],
        "requires_gpu_claim": "requires gpu" in body.lower() and "\nyes" in body.lower(),
        "claimed_report_values_external": claimed,
        "declared_dependency_note_present": (
            "constriction" in body.lower() and "pyppmd" in body.lower()
        ),
        "score_claim_in_this_artifact": False,
    }


def _regex_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return float(match.group(1)) if match else None


def _regex_int(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return int(match.group(1).replace(",", "")) if match else None


def read_pr85_reference(profile_path: Path) -> dict[str, Any]:
    profile = read_json_if_exists(profile_path) or {}
    archive = profile.get("archive", {})
    segments = profile.get("segments", [])
    by_name = {row.get("name"): row for row in segments}
    return {
        "profile_path": repo_rel(profile_path),
        "archive_bytes": archive.get("archive_size_bytes"),
        "archive_sha256": archive.get("archive_sha256"),
        "member_name": archive.get("member_name"),
        "member_sha256": archive.get("member_sha256"),
        "bundle_format": profile.get("bundle_format"),
        "t4_best_score_reference": PR85_T4_BEST_SCORE,
        "segments": {
            name: {
                "bytes": row.get("bytes"),
                "sha256": row.get("sha256"),
                "decoded_bytes": row.get("decoded_bytes")
                or row.get("decompression", {}).get("decoded_bytes"),
                "magic_ascii": row.get("magic_ascii")
                or row.get("encoded_magic_ascii")
                or row.get("decoded_magic_ascii"),
            }
            for name, row in by_name.items()
        },
    }


def read_replay_status(replay_dir: Path) -> dict[str, Any]:
    replay_dir = replay_dir.resolve()
    score_json = replay_dir / "contest_auth_eval.json"
    infra_failure = replay_dir / "lightning_artifact_infra_failure.json"
    auth_log = replay_dir / "auth_eval.log"
    if score_json.is_file():
        data = read_json_if_exists(score_json) or {}
        score = data.get("score_recomputed_from_components", data.get("canonical_score"))
        return {
            "status": "score_json_present",
            "evidence_grade": "exact_score_json_local_mirror",
            "replay_dir": repo_rel(replay_dir),
            "score_json": repo_rel(score_json),
            "score_recomputed_from_components": score,
            "avg_segnet_dist": data.get("avg_segnet_dist"),
            "avg_posenet_dist": data.get("avg_posenet_dist"),
            "n_samples": data.get("n_samples"),
            "archive_size_bytes": data.get("archive_size_bytes"),
            "score_delta_vs_pr85": score - PR85_T4_BEST_SCORE if isinstance(score, (int, float)) else None,
            "score_claim": False,
        }
    if infra_failure.is_file():
        data = read_json_if_exists(infra_failure) or {}
        log_tail = auth_log.read_text(encoding="utf-8", errors="replace")[-4000:] if auth_log.is_file() else ""
        whitelist_blocked = "UNKNOWN file types in archive" in log_tail
        return {
            "status": (
                "archive_validator_whitelist_blocked"
                if whitelist_blocked
                else data.get("status", "artifact_infra_failure")
            ),
            "evidence_grade": "invalid",
            "replay_dir": repo_rel(replay_dir),
            "source_files": [
                repo_rel(path)
                for path in (infra_failure, auth_log)
                if path.is_file()
            ],
            "failure_class": data.get("failure_class"),
            "terminal_class": data.get("terminal_class"),
            "reason": data.get("reason"),
            "missing_required_files": data.get("missing_required_files", []),
            "archive_identity": data.get("archive_identity"),
            "auth_eval_log_tail": log_tail.strip().splitlines()[-20:],
            "score_claim": False,
        }
    if auth_log.is_file():
        log_tail = auth_log.read_text(encoding="utf-8", errors="replace")[-4000:]
        return {
            "status": "auth_log_present_without_score_json",
            "evidence_grade": "invalid",
            "replay_dir": repo_rel(replay_dir),
            "source_files": [repo_rel(auth_log)],
            "auth_eval_log_tail": log_tail.strip().splitlines()[-20:],
            "score_claim": False,
        }
    return {
        "status": "no_local_replay_artifacts",
        "evidence_grade": "none",
        "replay_dir": repo_rel(replay_dir),
        "score_claim": False,
    }


def build_transplant_opportunities(
    *,
    archive_contract: dict[str, Any],
    pr85_reference: dict[str, Any],
    source_contract: dict[str, Any],
    replay_status: dict[str, Any],
) -> list[dict[str, Any]]:
    member_bytes = {
        row["name"]: row["file_size"] for row in archive_contract.get("members", [])
    }
    pr86_archive_bytes = archive_contract.get("archive_bytes")
    pr85_archive_bytes = pr85_reference.get("archive_bytes")
    pr85_mask_bytes = pr85_reference.get("segments", {}).get("mask", {}).get("bytes")
    pr85_non_mask_bytes = (
        pr85_reference.get("member_sha256")
        and pr85_archive_bytes
        and pr85_mask_bytes
        and pr85_archive_bytes - pr85_mask_bytes
    )
    hpac_token_stack_bytes = sum(
        member_bytes.get(name, 0) for name in ("hpac.pt.ppmd", "tokens.bin", "meta.pt")
    )
    pr86_model_stack_bytes = sum(
        member_bytes.get(name, 0) for name in ("master.pt.gz", "slave.pt.gz")
    )
    token_encoding = source_contract["token_hpac_decode_contract"][
        "submitted_archive_token_encoding"
    ]
    current_replay_status = replay_status.get("status")

    return [
        {
            "id": "hpac_reencode_pr85_mask_tokens",
            "target": "Replace PR85 QMA9 mask segment with an HPAC-coded token stream.",
            "gross_byte_math": {
                "pr85_mask_segment_bytes": pr85_mask_bytes,
                "pr86_hpac_tokens_meta_bytes": hpac_token_stack_bytes,
                "gross_saved_bytes_if_same_contract": (
                    pr85_mask_bytes - hpac_token_stack_bytes
                    if isinstance(pr85_mask_bytes, int)
                    else None
                ),
            },
            "drop_in_status": "not_drop_in",
            "why_not_drop_in": (
                "PR86 tokens.bin is tied to PR86 HPACMini probabilities and raw "
                "SegNet token maps. PR85 uses a QMA9 mask segment inside a "
                "monolithic x bundle; direct byte transplant has no decoded-mask parity proof."
            ),
            "required_local_gates": [
                "full PR86 own-stream decode",
                "byte-exact PR86 decode->encode tokens.bin parity",
                "PR85 baseline token source extraction with shape/range proof",
                "HPAC train or fit on PR85 token maps",
                "PR85 output parity before exact eval",
                "no unexpected archive members or sidecars",
            ],
            "priority_by_replay": (
                "high_if_pr86_exact_score_beats_pr85_and_replay_is_adjudicated"
                if current_replay_status == "score_json_present"
                else "blocked_until_replay_contract_closes"
            ),
            "score_claim": False,
        },
        {
            "id": "full_pr86_runtime_as_external_baseline",
            "target": "Treat PR86 as a full external neural-codec baseline, not a PR85 transplant.",
            "gross_byte_math": {
                "pr85_archive_bytes": pr85_archive_bytes,
                "pr86_archive_bytes": pr86_archive_bytes,
                "gross_saved_bytes_vs_pr85_archive": (
                    pr85_archive_bytes - pr86_archive_bytes
                    if isinstance(pr85_archive_bytes, int)
                    and isinstance(pr86_archive_bytes, int)
                    else None
                ),
                "pr86_model_stack_bytes": pr86_model_stack_bytes,
            },
            "drop_in_status": "full_runtime_replacement_only",
            "required_local_gates": [
                "archive validator must accept PR86 member formats or a contest-legal adapter must be proven",
                "contest_auth_eval.json must exist with 600 CUDA samples",
                "runtime tree hash must be captured for the exact replay",
                "component gates must be adjudicated",
            ],
            "priority_by_replay": "invalid_currently" if current_replay_status != "score_json_present" else "depends_on_score_delta",
            "score_claim": False,
        },
        {
            "id": "hpac_probability_contract_port",
            "target": "Port the HPAC probability model/coder contract into an Apogee-owned PR85 mask coder.",
            "gross_byte_math": {
                "pr86_hpac_model_bytes": member_bytes.get("hpac.pt.ppmd"),
                "pr86_tokens_bytes": member_bytes.get("tokens.bin"),
                "pr86_meta_bytes": member_bytes.get("meta.pt"),
            },
            "drop_in_status": "design_prior_only",
            "contract_facts_to_preserve": {
                "submitted_token_encoding": token_encoding,
                "queue_dtype": "uint32",
                "categorical_perfect_false": source_contract["token_hpac_decode_contract"][
                    "categorical_perfect_false_present"
                ],
                "probability_clip_eps": source_contract["token_hpac_decode_contract"][
                    "probability_clip_eps"
                ],
            },
            "required_local_gates": [
                "deterministic parser for HPAC meta/model",
                "same-order constriction queue roundtrip",
                "CPU/CUDA HPAC decode contract closure",
                "local decoded-token SHA manifest",
            ],
            "score_claim": False,
        },
        {
            "id": "pr86_model_stack_lessons_for_pr85_nonmask_bytes",
            "target": "Use master/slave neural-renderer compression ideas against PR85 model/post/pose bytes.",
            "gross_byte_math": {
                "pr85_non_mask_archive_bytes_estimate": pr85_non_mask_bytes,
                "pr86_master_slave_bytes": pr86_model_stack_bytes,
                "gross_saved_bytes_if_replacing_all_nonmask_bytes": (
                    pr85_non_mask_bytes - pr86_model_stack_bytes
                    if isinstance(pr85_non_mask_bytes, int)
                    else None
                ),
            },
            "drop_in_status": "not_drop_in",
            "why_not_drop_in": (
                "PR86 master/slave generate video frames from tokens; PR85 model/pose/post "
                "segments are a different runtime contract."
            ),
            "required_local_gates": [
                "separate decoded-output parity target",
                "model/runtime closure under PR85 inflate path",
                "byte accounting after any decoder code changes",
            ],
            "score_claim": False,
        },
    ]


def build_recommended_actions(
    replay_status: dict[str, Any],
    opportunities: list[dict[str, Any]],
) -> dict[str, list[str]]:
    hpac_saves = opportunities[0]["gross_byte_math"].get("gross_saved_bytes_if_same_contract")
    common_prefix = (
        "Do not dispatch from this artifact; keep all outputs as local planning evidence."
    )
    return {
        "archive_validator_whitelist_blocked": [
            common_prefix,
            "Classify the current T4 mirror as invalid pre-score evidence, not a PR86 method result.",
            "Before another exact replay, close the archive validator/member-format contract for .pt.gz and .ppmd in a reviewed preflight path.",
            "Continue local HPAC decode/reencode parity work; PR85 transplant remains blocked.",
        ],
        "runtime_or_dependency_failure": [
            common_prefix,
            "Pin and reproduce constriction/pyppmd/torch CPU-CUDA behavior locally before touching PR85 bytes.",
            "Treat any entropy assertion as a decode-contract negative, not a family kill.",
        ],
        "score_json_present_beats_pr85": [
            common_prefix,
            "Adjudicate the exact replay JSON, runtime tree hash, sample count, and component gates.",
            "If clean, prioritize HPAC-on-PR85 mask coding; the gross PR85 mask opportunity is "
            f"{hpac_saves} bytes before new metadata/runtime costs.",
            "Build PR85 token extraction and HPAC byte-exact reencode tests before any archive candidate.",
        ],
        "score_json_present_not_better_than_pr85": [
            common_prefix,
            "Keep PR86 as an architecture prior, but do not replace the PR85 champion wholesale.",
            "Only pursue HPAC transfer if local PR85 token bpp estimates clear the PR85 score break-even after metadata/runtime costs.",
        ],
        "score_components_collapsed": [
            common_prefix,
            "Preserve the exact replay as A-negative only after custody and component recomputation are complete.",
            "Analyze whether collapse is renderer, token, pose, or runtime-custody specific before transplant work.",
        ],
        "no_local_replay_artifacts": [
            common_prefix,
            "Use this report as static intake only; wait for a local mirrored contest_auth_eval.json or failure classification.",
            "Run only build-free local decode/parse tests meanwhile.",
        ],
        "current_observed_replay_branch": _select_current_actions(replay_status),
    }


def _select_current_actions(replay_status: dict[str, Any]) -> list[str]:
    status = replay_status.get("status")
    if status == "archive_validator_whitelist_blocked":
        return [
            "Current local mirror branch: archive validator whitelist blocked before score JSON.",
            "No score claim is available; do not compare PR86 numerically to PR85 from this run.",
            "Fix or preflight the member-format contract before any future exact replay attempt.",
        ]
    if status == "score_json_present":
        score_delta = replay_status.get("score_delta_vs_pr85")
        if isinstance(score_delta, (int, float)) and score_delta < 0:
            return [
                "Current local mirror branch: scored replay beats PR85.",
                "Adjudicate before promotion; then open HPAC-on-PR85 mask transfer as the first local build path.",
            ]
        return [
            "Current local mirror branch: scored replay does not beat PR85.",
            "Use PR86 as a design prior only unless a component-specific transplant gate opens.",
        ]
    if status == "no_local_replay_artifacts":
        return [
            "Current local mirror branch: no replay artifacts found.",
            "Keep work to static parsing and local decode-contract tests.",
        ]
    return [
        f"Current local mirror branch: {status}.",
        "Treat as invalid/non-promotable until contest_auth_eval.json and adjudication artifacts exist.",
    ]


def stable_parse_digest(report: dict[str, Any]) -> str:
    stable = {
        "source_archive": report["source_archive"],
        "archive_member_contract": report["archive_member_contract"],
        "member_payload_layers": report["member_payload_layers"],
        "source_contract": report["model_runtime_dependency_closure"],
        "pr85_reference_static": report["pr85_reference_static"],
        "transplant_opportunities_onto_pr85": report["transplant_opportunities_onto_pr85"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def build_report(
    *,
    pr86_dir: Path = DEFAULT_PR86_DIR,
    archive: Path = DEFAULT_PR86_ARCHIVE,
    pr85_profile: Path = DEFAULT_PR85_PROFILE,
    replay_dir: Path = DEFAULT_REPLAY_DIR,
    inspect_payloads: bool = True,
) -> dict[str, Any]:
    archive_contract = inspect_archive_members(archive)
    member_layers = (
        inspect_member_payload_layers(archive)
        if inspect_payloads
        else {"members": [], "status": "skipped"}
    )
    source_contract = analyze_source_contract(pr86_dir)
    pr85_reference = read_pr85_reference(pr85_profile)
    replay_status = read_replay_status(replay_dir)
    pr_view = read_pr_view(pr86_dir)
    opportunities = build_transplant_opportunities(
        archive_contract=archive_contract,
        pr85_reference=pr85_reference,
        source_contract=source_contract,
        replay_status=replay_status,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "experiments/profile_pr86_hpac_token_anatomy.py",
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "score_claim": False,
        "dispatch_performed": False,
        "planning_only": True,
        "promotable": False,
        "evidence_grade": "local_static_forensic_planning",
        "source_archive": {
            "path": archive_contract["path"],
            "bytes": archive_contract["archive_bytes"],
            "sha256": archive_contract["archive_sha256"],
            "expected_bytes": EXPECTED_PR86_ARCHIVE_BYTES,
            "expected_sha256": EXPECTED_PR86_ARCHIVE_SHA256,
            "expected_identity_match": archive_contract["expected_identity_match"],
        },
        "public_pr86_reference": pr_view,
        "archive_member_contract": archive_contract,
        "member_payload_layers": member_layers,
        "model_runtime_dependency_closure": source_contract,
        "token_hpac_decode_contract": source_contract["token_hpac_decode_contract"],
        "pr85_reference_static": pr85_reference,
        "current_exact_replay_status": replay_status,
        "transplant_opportunities_onto_pr85": opportunities,
        "recommended_next_actions_by_exact_replay_outcome": build_recommended_actions(
            replay_status, opportunities
        ),
    }
    report["stable_parse_digest_sha256"] = stable_parse_digest(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    archive = report["source_archive"]
    contract = report["archive_member_contract"]
    replay = report["current_exact_replay_status"]
    token = report["token_hpac_decode_contract"]
    lines = [
        "# PR86 HPAC/token anatomy forensics",
        "",
        "- `score_claim=false`",
        "- `dispatch_performed=false`",
        f"- source archive: `{archive['path']}`",
        f"- source bytes/SHA-256: `{archive['bytes']}` / `{archive['sha256']}`",
        f"- expected identity match: `{archive['expected_identity_match']}`",
        f"- stable parse digest: `{report['stable_parse_digest_sha256']}`",
        "",
        "## Member byte anatomy",
        "",
        "| member | role | bytes | share | sha256 |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in contract["members"]:
        lines.append(
            "| `{name}` | {role} | {file_size} | {share:.3f} | `{sha}` |".format(
                name=row["name"],
                role=row["role"],
                file_size=row["file_size"],
                share=row["byte_share_of_archive"],
                sha=row["sha256"],
            )
        )
    lines.extend(
        [
            "",
            "## Fail-closed custody",
            "",
            f"- sidecar/member status: `{contract['sidecar_assumption_status']}`",
            f"- duplicate members: `{contract['duplicate_member_names']}`",
            f"- missing required members: `{contract['missing_required_members']}`",
            f"- unexpected members: `{contract['unexpected_members']}`",
            f"- unsafe member names: `{contract['unsafe_member_names']}`",
            "",
            "## Token/HPAC contract",
            "",
            f"- submitted token encoding: `{token['submitted_archive_token_encoding']}`",
            f"- training objective: `{token['training_objective']}`",
            f"- archive writes raw GT tokens: `{token['archive_write_tokens_second_arg'] == 'gt'}`",
            f"- inflate reconstructs residuals: `{token['inflate_reconstructs_residuals']}`",
            f"- constriction queue decoder present: `{token['range_decoder_api_present']}`",
            f"- Categorical perfect=False: `{token['categorical_perfect_false_present']}`",
            f"- probability clip epsilon: `{token['probability_clip_eps']}`",
            "",
            "## Current exact replay branch",
            "",
            f"- status: `{replay['status']}`",
            f"- evidence grade: `{replay['evidence_grade']}`",
            f"- score claim from this report: `false`",
            "",
            "## PR85 transplant opportunities",
            "",
        ]
    )
    for row in report["transplant_opportunities_onto_pr85"]:
        math = row.get("gross_byte_math", {})
        lines.extend(
            [
                f"### `{row['id']}`",
                "",
                f"- target: {row['target']}",
                f"- drop-in status: `{row['drop_in_status']}`",
                f"- gross byte math: `{json.dumps(math, sort_keys=True)}`",
                f"- score claim: `{row['score_claim']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Current branch actions",
            "",
        ]
    )
    for action in report["recommended_next_actions_by_exact_replay_outcome"][
        "current_observed_replay_branch"
    ]:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr86-dir", type=Path, default=DEFAULT_PR86_DIR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_PR86_ARCHIVE)
    parser.add_argument("--pr85-profile", type=Path, default=DEFAULT_PR85_PROFILE)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument(
        "--skip-payload-inspect",
        action="store_true",
        help="Skip nested gzip/PPMd/torch payload inspection.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_report(
        pr86_dir=args.pr86_dir.resolve(),
        archive=args.archive.resolve(),
        pr85_profile=args.pr85_profile.resolve(),
        replay_dir=args.replay_dir.resolve(),
        inspect_payloads=not args.skip_payload_inspect,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
