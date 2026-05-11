#!/usr/bin/env python3
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
    KagglePr106YshiftBundleSpec,
    write_bundle,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/kaggle_kernels"
DEFAULT_PR106_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
)
DEFAULT_CLAIMS_PATH = REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md"


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
    return KagglePr106YshiftBundleSpec(
        username=username,
        job_name=args.job_name,
        slug=args.slug,
        title=args.title,
        dataset_ref=dataset_ref,
        candidate_radius=args.candidate_radius,
        score_step=args.score_step,
        n_pairs=args.n_pairs,
        batch_pairs=args.batch_pairs,
        candidate_batch_size=args.candidate_batch_size,
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
    parser.add_argument("--slug", default=DEFAULT_KERNEL_SLUG)
    parser.add_argument("--title", default=DEFAULT_KERNEL_TITLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
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
    manifest = write_bundle(
        repo_root=REPO_ROOT,
        bundle_dir=bundle_dir,
        spec=spec,
        pr106_archive=args.pr106_archive,
        claims_path=args.claims_path,
    )
    print(json.dumps({"bundle_dir": str(bundle_dir), **manifest}, indent=2))
    print(f"Push after claim/stage review: kaggle kernels push -p {bundle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
