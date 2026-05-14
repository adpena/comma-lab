#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the private Kaggle bundle for PR106 y-shift score-table CUDA work."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.deploy.claims import dispatch_claim_command  # noqa: E402
from tac.deploy.pr106_yshift import dispatch_claim_spec  # noqa: E402
from tac.deploy.kaggle.pr106_yshift_score_table import (  # noqa: E402
    DEFAULT_JOB_NAME,
    DEFAULT_KERNEL_SLUG,
    DEFAULT_KERNEL_TITLE,
    DEFAULT_SOURCE_BUNDLE_NAME,
    DEFAULT_SOURCE_DATASET_SLUG,
    KagglePr106YshiftBundleSpec,
    write_bundle,
    write_source_bundle,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/kaggle_kernels"
DEFAULT_PR106_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
)
DEFAULT_CLAIMS_PATH = REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md"
DEFAULT_SOURCE_BUNDLE_ROOT = REPO_ROOT / "experiments/kaggle_datasets"


def _kaggle_username() -> str:
    creds = Path.home() / ".kaggle" / "kaggle.json"
    if not creds.is_file():
        raise SystemExit(f"FATAL: missing Kaggle credentials: {creds}")
    payload = json.loads(creds.read_text(encoding="utf-8"))
    username = payload.get("username")
    if not isinstance(username, str) or not username.strip():
        raise SystemExit(f"FATAL: missing username in {creds}")
    return username.strip()


def build_spec(args: argparse.Namespace) -> KagglePr106YshiftBundleSpec:
    username = args.username or _kaggle_username()
    dataset_ref = args.dataset_ref or f"{username}/comma-lab-private-assets"
    source_dataset_ref = args.source_dataset_ref or f"{username}/{DEFAULT_SOURCE_DATASET_SLUG}"
    return KagglePr106YshiftBundleSpec(
        username=username,
        job_name=args.job_name,
        slug=args.slug,
        title=args.title,
        dataset_ref=dataset_ref,
        source_dataset_ref=source_dataset_ref,
        candidate_radius=args.candidate_radius,
        score_step=args.score_step,
        n_pairs=args.n_pairs,
        batch_pairs=args.batch_pairs,
        candidate_batch_size=args.candidate_batch_size,
        source_bundle_name=args.source_bundle_name,
    )


def build_claim_command(args: argparse.Namespace) -> list[str]:
    spec = build_spec(args)
    claim = dispatch_claim_spec(
        spec.score_table_spec(),
        platform="kaggle",
        agent=args.agent,
        predicted_eta_hours=args.predicted_eta_hours,
        force=args.force_claim,
        notes="PR106 yshift score-table private Kaggle CUDA producer",
    )
    return dispatch_claim_command(
        spec=claim,
        status="active_dispatching",
        python_executable=sys.executable,
        claim_tool=REPO_ROOT / "tools/claim_lane_dispatch.py",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    parser.add_argument("--username", default="")
    parser.add_argument("--dataset-ref", default="")
    parser.add_argument("--source-dataset-ref", default="")
    parser.add_argument("--slug", default=DEFAULT_KERNEL_SLUG)
    parser.add_argument("--title", default=DEFAULT_KERNEL_TITLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-bundle-root", type=Path, default=DEFAULT_SOURCE_BUNDLE_ROOT)
    parser.add_argument("--source-bundle-name", default=DEFAULT_SOURCE_BUNDLE_NAME)
    parser.add_argument(
        "--write-source-bundle",
        action="store_true",
        help="Also write the dataset tarball consumed by the Kaggle script kernel.",
    )
    parser.add_argument(
        "--source-dataset-dir",
        type=Path,
        default=None,
        help="Optional dataset staging directory. Defaults under --source-bundle-root.",
    )
    parser.add_argument("--pr106-archive", type=Path, default=DEFAULT_PR106_ARCHIVE)
    parser.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS_PATH)
    parser.add_argument("--candidate-radius", type=int, default=3)
    parser.add_argument("--score-step", type=float, default=1.0)
    parser.add_argument("--n-pairs", type=int, default=600)
    parser.add_argument("--batch-pairs", type=int, default=8)
    parser.add_argument("--candidate-batch-size", type=int, default=32)
    parser.add_argument("--agent", default="codex:gpt-5.5")
    parser.add_argument("--predicted-eta-hours", type=float, default=3.0)
    parser.add_argument("--force-claim", action="store_true")
    parser.add_argument(
        "--print-claim",
        action="store_true",
        help="Print the required claim command and exit without writing a bundle.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_claim:
        print(" \\\n+  ".join(build_claim_command(args)))
        return 0

    spec = build_spec(args)
    bundle_dir = args.output_root / spec.slug
    source_manifest = None
    if args.write_source_bundle:
        dataset_dir = args.source_dataset_dir or args.source_bundle_root / DEFAULT_SOURCE_DATASET_SLUG
        dataset_dir.mkdir(parents=True, exist_ok=True)
        source_manifest = write_source_bundle(
            repo_root=REPO_ROOT,
            output_path=dataset_dir / spec.source_bundle_name,
            spec=spec,
            pr106_archive=args.pr106_archive,
            claims_path=args.claims_path,
        )
        source_ref = spec.source_dataset_ref or f"{spec.username}/{DEFAULT_SOURCE_DATASET_SLUG}"
        dataset_metadata = {
            "title": "comma-lab PR106 yshift source bundle",
            "id": source_ref,
            "licenses": [{"name": "other"}],
        }
        (dataset_dir / "dataset-metadata.json").write_text(
            json.dumps(dataset_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        (dataset_dir / "README.md").write_text(
            "Private source bundle for the PR106 y-shift score-table Kaggle kernel. "
            "Contains no score claim; exact results require contest-auth adjudication.\n",
            encoding="utf-8",
        )

    manifest = write_bundle(
        repo_root=REPO_ROOT,
        bundle_dir=bundle_dir,
        spec=spec,
        pr106_archive=args.pr106_archive,
        claims_path=args.claims_path,
    )
    payload = {"bundle_dir": str(bundle_dir), **manifest}
    if source_manifest is not None:
        payload["source_dataset_dir"] = str(args.source_dataset_dir or args.source_bundle_root / DEFAULT_SOURCE_DATASET_SLUG)
        payload["source_manifest"] = source_manifest
    print(json.dumps(payload, indent=2))
    if source_manifest is not None:
        print(
            "Create/version source dataset before kernel push: "
            f"kaggle datasets create -p {payload['source_dataset_dir']} "
            f"or kaggle datasets version -p {payload['source_dataset_dir']} -m 'PR106 yshift source bundle'"
        )
    print(f"Push after claim/stage review: kaggle kernels push -p {bundle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
