#!/usr/bin/env python3
"""Assemble the PFP16 A++ deploy/paper evidence bundle."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    _bootstrap_path = Path(__file__).resolve().parent.parent / "tools" / "tool_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("tool_bootstrap", _bootstrap_path)
    if _spec is None or _spec.loader is None:
        raise
    _tool_bootstrap = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_tool_bootstrap)
    ensure_repo_imports = _tool_bootstrap.ensure_repo_imports
    repo_root_from_tool = _tool_bootstrap.repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, read_json, sha256_file  # noqa: E402

DEFAULT_EVIDENCE_DIR = REPO_ROOT / "experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex"
DEFAULT_ARCHIVE = REPO_ROOT / "experiments/results/lane_g_v3_pfp16/archive_lane_g_v3_pfp16.zip"
DEFAULT_BUNDLE_DIR = REPO_ROOT / "experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430"
LEGACY_REMOTE_SCORE_FIELDS = (
    "contest_cuda_score",
    "score_delta_vs_lane_g_v3",
    "hard_kill_triggered",
    "lane_status",
)


def _run_git(args: list[str]) -> str:
    proc = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        return f"<error rc={proc.returncode}: {proc.stderr.strip()[:300]}>"
    return proc.stdout


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _zip_manifest(archive: Path) -> tuple[list[dict[str, Any]], list[str]]:
    manifest: list[dict[str, Any]] = []
    member_hash_lines: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        for info in sorted(zf.infolist(), key=lambda x: x.filename):
            data = zf.read(info.filename)
            digest = hashlib.sha256(data).hexdigest()
            manifest.append(
                {
                    "filename": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "crc": f"{info.CRC:08x}",
                    "sha256": digest,
                    "date_time": list(info.date_time),
                    "compress_type": info.compress_type,
                    "external_attr": info.external_attr,
                }
            )
            member_hash_lines.append(f"{digest}  {info.filename}")
    return manifest, member_hash_lines


def _timing_from_log(log_text: str) -> dict[str, float | None]:
    def grab(pattern: str) -> float | None:
        match = re.search(pattern, log_text)
        return float(match.group(1)) if match else None

    return {
        "frame_generation_seconds": grab(r"Generated 1200 frames .*, ([\d.]+)s\)"),
        "internal_inflate_seconds": grab(r"Total inflate time: ([\d.]+)s"),
        "wrapper_inflate_elapsed_seconds": grab(r"\[inflate\] returncode=0 elapsed=([\d.]+)s"),
        "evaluate_elapsed_seconds": grab(r"\[evaluate\] returncode=0 elapsed=([\d.]+)s"),
        "inflate_timeout_seconds": grab(r"\[inflate\] timeout: ([\d.]+)s"),
    }


def _canonical_eval_record(payload: dict[str, Any]) -> dict[str, Any]:
    provenance = payload.get("provenance") or {}
    return {
        "score_source": "eval/contest_auth_eval.json",
        "final_score": payload.get("final_score"),
        "score_recomputed_from_components": payload.get("score_recomputed_from_components"),
        "avg_segnet_dist": payload.get("avg_segnet_dist"),
        "avg_posenet_dist": payload.get("avg_posenet_dist"),
        "archive_size_bytes": payload.get("archive_size_bytes"),
        "archive_sha256": provenance.get("archive_sha256"),
        "n_samples": payload.get("n_samples"),
        "device": provenance.get("device"),
        "gpu_model": provenance.get("gpu_model"),
        "gpu_t4_match": provenance.get("gpu_t4_match"),
        "upstream_commit": provenance.get("upstream_commit"),
        "eval_tool": provenance.get("tool"),
        "eval_command": provenance.get("sys_argv"),
    }


def _quarantine_remote_parser_fields(
    remote_payload: dict[str, Any], contest_payload: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    sanitized = dict(remote_payload)
    quarantined_fields = {
        field: sanitized.pop(field)
        for field in LEGACY_REMOTE_SCORE_FIELDS
        if field in sanitized
    }
    if not quarantined_fields:
        return sanitized, None

    quarantine = {
        "status": "invalid_superseded",
        "reason": (
            "Legacy remote provenance parsed/adjudicated non-authoritative output and "
            "emitted stale score/regression fields. contest_auth_eval.json is the "
            "only score authority for PFP16 A++."
        ),
        "authoritative_score_source": "../eval/contest_auth_eval.json",
        "authoritative_eval": _canonical_eval_record(contest_payload),
        "quarantined_fields": quarantined_fields,
        "reader_action": (
            "Ignore quarantined fields for score, rank, promotion, regression, "
            "retirement, kill, and paper claims."
        ),
    }
    sanitized["canonical_contest_auth_eval"] = quarantine["authoritative_eval"]
    sanitized["legacy_parser_output_quarantined"] = quarantine
    return sanitized, quarantine


def build_bundle(evidence_dir: Path, archive: Path, bundle_dir: Path) -> dict[str, Any]:
    contest_json = evidence_dir / "contest_auth_eval.json"
    eval_provenance = evidence_dir / "eval_provenance.json"
    auth_eval_log = evidence_dir / "auth_eval.log"
    report = evidence_dir / "report.txt"
    gpu_txt = evidence_dir / "gpu.txt"
    required = [contest_json, eval_provenance, auth_eval_log, report, gpu_txt, archive]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required bundle inputs: {missing}")

    payload = read_json(contest_json)
    provenance = payload.get("provenance") or {}
    archive_sha256 = sha256_file(archive)
    archive_bytes = archive.stat().st_size
    if archive_sha256 != provenance.get("archive_sha256"):
        raise RuntimeError("archive SHA does not match contest_auth_eval provenance")
    if archive_bytes != payload.get("archive_size_bytes"):
        raise RuntimeError("archive byte size does not match contest_auth_eval payload")
    if provenance.get("device") != "cuda" or provenance.get("gpu_t4_match") is not True:
        raise RuntimeError("PFP16 A++ bundle requires cuda and gpu_t4_match=true")
    if payload.get("n_samples") != 600:
        raise RuntimeError("PFP16 A++ bundle requires n_samples=600")

    if bundle_dir.exists():
        raise FileExistsError(f"bundle already exists: {bundle_dir}")

    archive_manifest, member_hash_lines = _zip_manifest(archive)

    _copy(archive, bundle_dir / "archive/archive.zip")
    (bundle_dir / "archive/archive_sha256.txt").parent.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "archive/archive_sha256.txt").write_text(f"{archive_sha256}  archive.zip\n")
    (bundle_dir / "archive/archive_manifest.json").write_text(json_text(archive_manifest))
    (bundle_dir / "archive/archive_member_sha256.txt").write_text("\n".join(member_hash_lines) + "\n")

    with zipfile.ZipFile(archive) as zf:
        zipinfo = "\n".join(f"{i.filename}\t{i.file_size}\t{i.compress_size}\t{i.CRC:08x}" for i in zf.infolist())
    (bundle_dir / "archive/zipinfo.txt").write_text(zipinfo + "\n")

    for name in ["contest_auth_eval.json", "eval_provenance.json", "auth_eval.log", "report.txt", "gpu.txt"]:
        _copy(evidence_dir / name, bundle_dir / f"eval/{name}")

    sys_argv = provenance.get("sys_argv") or []
    run_command = " ".join(str(x) for x in sys_argv)
    (bundle_dir / "eval/run_command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + run_command + "\n")
    (bundle_dir / "eval/inflate_timing.json").write_text(
        json_text(_timing_from_log(auth_eval_log.read_text(errors="replace")))
    )

    legacy_quarantine: dict[str, Any] | None = None
    build_dir = REPO_ROOT / "experiments/results/lane_g_v3_pfp16_landed"
    for name in ["build_provenance.json", "build.log", "provenance.json"]:
        src = build_dir / name
        if src.exists():
            dst = bundle_dir / f"build/{name}"
            if name == "provenance.json":
                remote_payload = read_json(src)
                sanitized, legacy_quarantine = _quarantine_remote_parser_fields(remote_payload, payload)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json_text(sanitized))
            else:
                _copy(src, dst)

    source_dir = bundle_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "git_head.txt").write_text(_run_git(["rev-parse", "HEAD"]).strip() + "\n")
    (source_dir / "git_status.txt").write_text(_run_git(["status", "--short"]))
    diff_text = _run_git(["diff", "--binary"])
    (source_dir / "git_diff.patch").write_text(diff_text)
    (source_dir / "git_diff_sha256.txt").write_text(hashlib.sha256(diff_text.encode()).hexdigest() + "\n")
    (source_dir / "remote_staged_tree_manifest.json").write_text(
        json_text(
            {
                "status": "missing",
                "reason": "Lightning staged tree was non-git; contest_auth_eval provenance recorded pact_commit error.",
                "remote_archive_path": provenance.get("archive_path"),
                "remote_pact_path_inferred": str(provenance.get("archive_path", "")).split("/auth_eval_input/")[0],
            }
        )
    )

    upstream_dir = bundle_dir / "upstream"
    upstream_dir.mkdir(parents=True, exist_ok=True)
    (upstream_dir / "upstream_commit.txt").write_text(str(provenance.get("upstream_commit")) + "\n")

    review_dir = bundle_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "council_lane_pfp16_round1_20260430.md",
        "council_lane_pfp16_round2_20260430.md",
        "council_lane_pfp16_round3_20260430.md",
    ]:
        src = REPO_ROOT / ".omx/research" / name
        if src.exists():
            _copy(src, review_dir / name)
    docs_dir = bundle_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    src = REPO_ROOT / ".omx/research/pfp16_a_plus_plus_exact_t4_eval_runbook_20260430.md"
    if src.exists():
        _copy(src, docs_dir / src.name)

    quarantined_field_names = sorted((legacy_quarantine or {}).get("quarantined_fields", {}).keys())
    summary = {
        "schema_version": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_grade": "A++",
        "score_source": "eval/contest_auth_eval.json",
        "score_authority": "eval/contest_auth_eval.json",
        "score_recomputed_from_components": payload["score_recomputed_from_components"],
        "final_score_reported": payload["final_score"],
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_bytes,
        "n_samples": payload["n_samples"],
        "device": provenance.get("device"),
        "gpu_model": provenance.get("gpu_model"),
        "gpu_t4_match": provenance.get("gpu_t4_match"),
        "upstream_commit": provenance.get("upstream_commit"),
        "pact_commit_in_remote_eval": provenance.get("pact_commit"),
        "known_bundle_gap": "remote_staged_tree_manifest missing because Lightning staged tree was non-git",
        "cuda_auth_eval_source_of_truth": True,
        "custody_manifest": "custody/custody_manifest.json",
        "paper_custody_note": "custody/PFP16_A_PLUS_PLUS_CUSTODY_NOTE.md",
        "legacy_remote_parser_fields_quarantined": bool(legacy_quarantine),
        "quarantined_remote_parser_fields": quarantined_field_names,
    }
    (bundle_dir / "bundle_summary.json").write_text(json_text(summary))
    custody_dir = bundle_dir / "custody"
    custody_dir.mkdir(parents=True, exist_ok=True)
    custody_manifest = {
        "schema_version": 1,
        "created_at_utc": summary["created_at_utc"],
        "bundle": {
            "name": "pfp16_a_plus_plus_final_deploy_bundle_20260430",
            "evidence_grade": "A++",
            "known_gap": summary["known_bundle_gap"],
        },
        "archive": {
            "path": "archive/archive.zip",
            "sha256": archive_sha256,
            "size_bytes": archive_bytes,
            "manifest": "archive/archive_manifest.json",
            "member_hashes": "archive/archive_member_sha256.txt",
            "zipinfo": "archive/zipinfo.txt",
        },
        "authoritative_eval": _canonical_eval_record(payload),
        "score_formula": "100 * seg_dist + sqrt(10 * pose_dist) + 25 * archive_bytes / 37545489",
        "score_claim_policy": {
            "authoritative_json": "eval/contest_auth_eval.json",
            "must_recompute_from_components": True,
            "ignored_score_sources": [
                "eval/auth_eval.log human text",
                "build/provenance.json legacy_parser_output_quarantined",
                "MPS/CPU/proxy/local-renderer outputs",
            ],
        },
        "legacy_parser_quarantine": legacy_quarantine
        or {
            "status": "not_applicable",
            "quarantined_fields": {},
        },
        "paper_ready_claim": (
            "PFP16 A++ exact T4 CUDA auth eval recomputed score "
            f"{payload['score_recomputed_from_components']} on archive SHA "
            f"{archive_sha256} ({archive_bytes} bytes, n={payload['n_samples']})."
        ),
    }
    (custody_dir / "custody_manifest.json").write_text(json_text(custody_manifest))
    (custody_dir / "PFP16_A_PLUS_PLUS_CUSTODY_NOTE.md").write_text(
        "# PFP16 A++ Custody Note\n\n"
        "Paper-ready claim: exact T4 CUDA auth eval recomputes score "
        f"`{payload['score_recomputed_from_components']}` for archive "
        f"`{archive_sha256}` (`{archive_bytes}` bytes, `n={payload['n_samples']}`).\n\n"
        "The only score authority in this bundle is `eval/contest_auth_eval.json`. "
        "It records `experiments/contest_auth_eval.py --device cuda`, Tesla T4 "
        "provenance, the upstream commit, component distances, archive bytes, "
        "and the recomputed contest formula.\n\n"
        "Legacy remote parser/adjudication fields are quarantined in "
        "`build/provenance.json` under `legacy_parser_output_quarantined`. "
        "They must not be used for score, rank, promotion, regression, retirement, "
        "kill, or paper claims.\n\n"
        "This custody note and manifest are metadata sidecars only; they do not "
        "alter `archive/archive.zip`.\n"
    )
    (bundle_dir / "README.md").write_text(
        "# PFP16 A++ Evidence Bundle\n\n"
        "This bundle is a deploy/paper custody packet for the exact T4 CUDA auth eval.\n"
        "Score authority is `eval/contest_auth_eval.json`; MPS/CPU/proxy/log-only outputs are not score evidence.\n\n"
        f"- Recomputed score: `{summary['score_recomputed_from_components']}`\n"
        f"- Archive SHA-256: `{archive_sha256}`\n"
        f"- Archive bytes: `{archive_bytes}`\n"
        f"- GPU: `{summary['gpu_model']}` (`gpu_t4_match=true`)\n"
        f"- Upstream commit: `{summary['upstream_commit']}`\n"
        f"- Custody manifest: `{summary['custody_manifest']}`\n"
        f"- Paper custody note: `{summary['paper_custody_note']}`\n\n"
        "Legacy remote parser fields are superseded: any quarantined "
        "`contest_cuda_score`, `score_delta_vs_lane_g_v3`, `hard_kill_triggered`, "
        "or `lane_status=HARD_KILL_REGRESSION` values in build provenance are "
        "historical parser output only and are invalid for claims.\n\n"
        "Known gap: the remote Lightning staged tree was non-git, so this bundle includes local git/diff state and records the missing remote staged-tree manifest explicitly.\n"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    args = parser.parse_args()
    summary = build_bundle(args.evidence_dir, args.archive, args.bundle_dir)
    print(json_text(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
