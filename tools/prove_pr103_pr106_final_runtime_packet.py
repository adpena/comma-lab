#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove the PR103-on-PR106 final runtime packet without running the scorer."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tomllib
import zipfile
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from submissions.pr103_pr106_final_runtime import inflate as runtime  # noqa: E402
from tac.pr103_pr106_runtime_closure import (  # noqa: E402
    Pr103Pr106RuntimeClosure,
    parse_pr103_repacked_pr106_payload,
    sha256_bytes,
)
from tac.repo_io import json_text, sha256_file  # noqa: E402

DEFAULT_CANDIDATE_ARCHIVE = (
    REPO_ROOT / "experiments/results/pr103_repack_pr106_standalone_20260507/archive.zip"
)
DEFAULT_RUNTIME_CLOSURE = (
    REPO_ROOT / "experiments/results/pr103_repack_pr106_standalone_20260507/runtime_closure.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/results/pr103_repack_pr106_standalone_20260507/final_runtime_packet_proof.json"
)
DEFAULT_RUNTIME_DIR = REPO_ROOT / "submissions/pr103_pr106_final_runtime"
RUNTIME_FILES = ("__init__.py", "inflate.py", "inflate.sh")
FORBIDDEN_RUNTIME_MARKERS = (
    "from tac.",
    "import tac.",
    "upstream/evaluate",
    "evaluate.py",
    "tac.scorers",
    "segmentation_models",
    "timm",
)


def _read_single_member(path: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise ValueError(f"{path} has {len(infos)} file members; expected exactly one")
        return infos[0].filename, zf.read(infos[0])


def _runtime_file_record(runtime_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rel in RUNTIME_FILES:
        path = runtime_dir / rel
        out[rel] = {
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    return out


def _runtime_static_scan(runtime_dir: Path) -> dict[str, Any]:
    hits: list[str] = []
    for rel in RUNTIME_FILES:
        path = runtime_dir / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for marker in FORBIDDEN_RUNTIME_MARKERS:
            if marker in text:
                hits.append(f"{rel}:{marker}")
    return {"forbidden_markers": hits, "passed": not hits}


def _dependency_custody(repo_root: Path) -> dict[str, Any]:
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    dependencies = pyproject.get("project", {}).get("dependencies", [])
    dep_text = "\n".join(str(item).lower() for item in dependencies)
    uv_lock_text = (repo_root / "uv.lock").read_text(encoding="utf-8", errors="ignore")
    records: dict[str, Any] = {}
    for package, pattern in {
        "brotli": r"\bbrotli\s*>=\s*1\.0",
        "constriction": r"\bconstriction\s*>=\s*0\.4\s*,\s*<\s*0\.5",
    }.items():
        records[package] = {
            "pyproject_hard_dependency": bool(re.search(pattern, dep_text)),
            "uv_lock_entry": f'name = "{package}"' in uv_lock_text,
        }
    records["runtime_import_versions"] = runtime.runtime_dependency_versions()
    records["passed"] = all(
        item["pyproject_hard_dependency"] and item["uv_lock_entry"]
        for item in records.values()
        if isinstance(item, dict) and "pyproject_hard_dependency" in item
    )
    return records


def _compare_decodes(candidate_payload: bytes, closure_record: dict[str, Any]) -> dict[str, Any]:
    runtime_sd, runtime_latents, runtime_meta = runtime.parse_pr103_pr106_archive(candidate_payload)
    tac_closure = Pr103Pr106RuntimeClosure.from_dict(closure_record["runtime_closure"])
    tac_decoded = parse_pr103_repacked_pr106_payload(candidate_payload, tac_closure)

    tensor_diffs: dict[str, float] = {}
    state_dict_exact = set(runtime_sd) == set(tac_decoded.state_dict)
    if state_dict_exact:
        for name in sorted(runtime_sd):
            left = runtime_sd[name]
            right = tac_decoded.state_dict[name]
            if tuple(left.shape) != tuple(right.shape):
                state_dict_exact = False
                tensor_diffs[name] = float("inf")
                continue
            diff = float((left - right).abs().max().item()) if left.numel() else 0.0
            tensor_diffs[name] = diff
            if diff != 0.0:
                state_dict_exact = False
    latents_diff = float((runtime_latents - tac_decoded.latents).abs().max().item())
    return {
        "runtime_contract": {
            "state_dict_tensors": len(runtime_sd),
            "state_dict_params": int(sum(t.numel() for t in runtime_sd.values())),
            "latents_shape": list(runtime_latents.shape),
            "meta": runtime_meta,
        },
        "matches_tac_closure": {
            "state_dict_exact": state_dict_exact,
            "latents_exact": latents_diff == 0.0,
            "latents_max_abs_diff": latents_diff,
            "max_state_dict_abs_diff": max(tensor_diffs.values()) if tensor_diffs else None,
        },
    }


def build_proof(args: argparse.Namespace) -> dict[str, Any]:
    closure_record = json.loads(args.runtime_closure.read_text())
    member_name, candidate_payload = _read_single_member(args.candidate_archive)
    candidate_record = {
        "path": str(args.candidate_archive.relative_to(REPO_ROOT)),
        "archive_bytes": args.candidate_archive.stat().st_size,
        "archive_sha256": sha256_file(args.candidate_archive),
        "member_name": member_name,
        "payload_bytes": len(candidate_payload),
        "payload_sha256": sha256_bytes(candidate_payload),
    }
    bash_n = subprocess.run(
        ["bash", "-n", str(args.runtime_dir / "inflate.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    runtime_closure_match = closure_record["runtime_closure"] == runtime.RUNTIME_CLOSURE
    decode_record = _compare_decodes(candidate_payload, closure_record)
    static_scan = _runtime_static_scan(args.runtime_dir)
    dependency_custody = _dependency_custody(REPO_ROOT)
    checks = {
        "candidate_payload_sha_matches_runtime": (
            candidate_record["payload_sha256"] == runtime.EXPECTED_PAYLOAD_SHA256
        ),
        "runtime_closure_constants_match_proof_artifact": runtime_closure_match,
        "runtime_decode_matches_tac_closure": all(
            decode_record["matches_tac_closure"][key]
            for key in ("state_dict_exact", "latents_exact")
        ),
        "runtime_static_scan_no_scorer_or_tac_imports": static_scan["passed"],
        "brotli_constriction_dependency_custody": dependency_custody["passed"],
        "inflate_sh_bash_n": bash_n.returncode == 0,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "tool": "tools.prove_pr103_pr106_final_runtime_packet",
        "score_claim": False,
        "score_evidence_grade": "empirical_runtime_packet_closure",
        "passed": passed,
        "checks": checks,
        "candidate_archive": candidate_record,
        "runtime_dir": str(args.runtime_dir.relative_to(REPO_ROOT)),
        "runtime_files": _runtime_file_record(args.runtime_dir),
        "runtime_static_scan": static_scan,
        "dependency_custody": dependency_custody,
        "decode_proof": decode_record,
        "closure_artifact": str(args.runtime_closure.relative_to(REPO_ROOT)),
        "inflate_sh_bash_n_stderr": bash_n.stderr.strip(),
        "exact_cuda_remaining_blockers": [
            "run pre_submission_compliance_check on a release surface after auth-eval artifact exists",
            "claim dispatch lane before exact CUDA auth eval",
            "run archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA",
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-archive", type=Path, default=DEFAULT_CANDIDATE_ARCHIVE)
    parser.add_argument("--runtime-closure", type=Path, default=DEFAULT_RUNTIME_CLOSURE)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    proof = build_proof(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json_text(proof), encoding="utf-8")
    print(f"[pr103-pr106-final-proof] wrote {args.output}")
    print(f"[pr103-pr106-final-proof] passed={proof['passed']}")
    if args.strict and not proof["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
