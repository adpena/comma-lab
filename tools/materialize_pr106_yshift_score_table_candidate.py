#!/usr/bin/env python3
"""Materialize a PR106 y-shift score-table output into a byte-closed candidate."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, read_json, repo_relative, sha256_file, write_json  # noqa: E402

DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
)
DEFAULT_SCORE_TABLE_ROOT = REPO_ROOT / "reports/raw/kaggle_pr106_yshift_score_table_latest"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/results"
BUILDER = REPO_ROOT / "experiments/build_pr106_yshift_sidechannel.py"
SCORE_TABLE_FILENAME = "score_table.npy"
SCORE_TABLE_MANIFEST_FILENAME = "score_table_manifest.json"


def default_run_id() -> str:
    return "pr106_yshift_score_table_materialized_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _candidate_roots(root: Path) -> Iterable[Path]:
    yield root
    yield root / "score_table"
    yield root / "yshift_run" / "score_table"
    yield root / "pr106_yshift_score_table" / "score_table"
    yield root / "pr106_yshift_score_table" / "yshift_run" / "score_table"


def resolve_score_table_artifacts(
    *,
    score_table_root: Path,
    score_table_npy: Path | None,
    score_table_manifest: Path | None,
) -> tuple[Path, Path]:
    if (score_table_npy is None) ^ (score_table_manifest is None):
        raise ValueError("--score-table-npy and --score-table-manifest must be supplied together")
    if score_table_npy is not None and score_table_manifest is not None:
        npy = score_table_npy
        manifest = score_table_manifest
    else:
        matches: list[tuple[Path, Path]] = []
        for candidate_root in _candidate_roots(score_table_root):
            npy_candidate = candidate_root / SCORE_TABLE_FILENAME
            manifest_candidate = candidate_root / SCORE_TABLE_MANIFEST_FILENAME
            if npy_candidate.is_file() and manifest_candidate.is_file():
                matches.append((npy_candidate, manifest_candidate))
        unique = sorted({(npy.resolve(), manifest.resolve()) for npy, manifest in matches})
        if not unique:
            raise FileNotFoundError(
                "no score_table.npy + score_table_manifest.json pair found under "
                f"{score_table_root}"
            )
        if len(unique) > 1:
            rendered = ", ".join(f"{npy.parent}" for npy, _manifest in unique)
            raise ValueError(
                "multiple score-table artifact pairs found; pass explicit paths: "
                f"{rendered}"
            )
        npy, manifest = unique[0]
    if not npy.is_file():
        raise FileNotFoundError(f"score table .npy not found: {npy}")
    if not manifest.is_file():
        raise FileNotFoundError(f"score table manifest not found: {manifest}")
    return npy, manifest


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": repo_relative(path, REPO_ROOT),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def materialize_candidate(
    *,
    source_archive: Path,
    output_dir: Path,
    score_table_root: Path,
    score_table_npy: Path | None,
    score_table_manifest: Path | None,
    candidate_radius: int,
    score_step: float,
    n_pairs: int,
    python_executable: str,
) -> dict[str, object]:
    if not source_archive.is_file():
        raise FileNotFoundError(f"source archive not found: {source_archive}")
    npy_path, manifest_path = resolve_score_table_artifacts(
        score_table_root=score_table_root,
        score_table_npy=score_table_npy,
        score_table_manifest=score_table_manifest,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        str(BUILDER),
        "--pr106-archive",
        str(source_archive),
        "--out-dir",
        str(output_dir),
        "--search-mode",
        "score_table",
        "--score-table-npy",
        str(npy_path),
        "--score-table-manifest",
        str(manifest_path),
        "--candidate-radius",
        str(candidate_radius),
        "--score-step",
        str(score_step),
        "--n-pairs",
        str(n_pairs),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    archive_path = output_dir / "pr106_yshift_sidechannel_archive.zip"
    metadata_path = output_dir / "build_metadata.json"
    if not archive_path.is_file():
        raise FileNotFoundError(f"builder did not emit expected archive: {archive_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"builder did not emit expected metadata: {metadata_path}")
    build_metadata = read_json(metadata_path)
    if build_metadata.get("score_claim") is not False:
        raise RuntimeError("builder metadata must keep score_claim=false")
    if build_metadata.get("search_mode") != "score_table":
        raise RuntimeError("builder metadata search_mode must be score_table")
    if build_metadata.get("score_table", {}).get("score_table_manifest_validated") is not True:
        raise RuntimeError("builder did not validate the score-table manifest")

    materialization = {
        "schema": "pr106_yshift_score_table_candidate_materialization_v1",
        "lane_id": "lane_pr106_yshift_score_table",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotion_requires": "contest_cuda_adjudication_on_materialized_archive",
        "source_archive": _artifact(source_archive),
        "score_table_npy": _artifact(npy_path),
        "score_table_manifest": _artifact(manifest_path),
        "candidate_radius": int(candidate_radius),
        "score_step": float(score_step),
        "n_pairs": int(n_pairs),
        "builder": {
            "path": repo_relative(BUILDER, REPO_ROOT),
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
        "outputs": {
            "archive": _artifact(archive_path),
            "build_metadata": _artifact(metadata_path),
            "materialization_manifest": {
                "path": repo_relative(output_dir / "materialization_manifest.json", REPO_ROOT)
            },
        },
        "next_step": (
            "Claim lane_pr106_yshift_score_table or stacked PR106 sidechannel lane, run exact "
            "contest-CUDA auth eval on outputs.archive, then adjudicate component fields "
            "before any score claim."
        ),
    }
    manifest_out = output_dir / "materialization_manifest.json"
    write_json(manifest_out, materialization)
    materialization["outputs"]["materialization_manifest"] = _artifact(manifest_out)
    write_json(manifest_out, materialization)
    return materialization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--score-table-root", type=Path, default=DEFAULT_SCORE_TABLE_ROOT)
    parser.add_argument("--score-table-npy", type=Path, default=None)
    parser.add_argument("--score-table-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--candidate-radius", type=int, default=3)
    parser.add_argument("--score-step", type=float, default=1.0)
    parser.add_argument("--n-pairs", type=int, default=600)
    parser.add_argument("--python-executable", default=sys.executable)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / default_run_id())
    manifest = materialize_candidate(
        source_archive=args.source_archive,
        output_dir=output_dir,
        score_table_root=args.score_table_root,
        score_table_npy=args.score_table_npy,
        score_table_manifest=args.score_table_manifest,
        candidate_radius=args.candidate_radius,
        score_step=args.score_step,
        n_pairs=args.n_pairs,
        python_executable=args.python_executable,
    )
    print(json_text(manifest), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
