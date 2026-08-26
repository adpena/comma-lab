#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Canonical CLI for the submission chain: ledger / custody / verify / close.

Operator binding 2026-08-03: nothing needed for a full byte-closed archived
submission may live in an ad-hoc probe script.  This is the operator-facing
surface for ``tac.submission_chain``; the library holds the logic and the tests.

It SUPERSEDES the manual steps previously carried in
``experiments/ddm_pu2_stage_and_eval.sh`` (hard-coded SSD roots, a hard-coded
``evaluate.py`` invocation, no receipt, no ledger, no custody check).  Those
probe scripts are KEPT as history and are not deleted -- they are the record of
how the row was actually produced.

Every path resolves from an explicit flag, then the environment, then a repo
default.  Nothing is hard-coded to one machine's mounted volume.

    PACT_UPSTREAM_DIR PACT_RUNTIME_SRC PACT_VIDEOS_DIR
    PACT_VIDEO_NAMES_FILE PACT_CHAIN_WORK_DIR

Examples::

    # standing byte ledger for any ix2 container archive
    python tools/submission_chain_cli.py ledger --archive .../archive.zip

    # vendored runtime custody against repo HEAD
    python tools/submission_chain_cli.py custody --runtime-src .../v4d_cx1_pj2ix2

    # byte-identity control against the live own-vehicle frontier
    python tools/submission_chain_cli.py verify --archive .../archive.zip --profile v4d_cx1

    # full chain (inflate + n600 evaluate + typed receipt); eval is OPT-IN
    python tools/submission_chain_cli.py close --archive .../archive.zip \
        --runtime-src .../v4d_cx1_pj2ix2 --run-evaluate --eval-device cpu

Axis: on this host a CPU row is ``[macOS-CPU advisory]`` and NON-PROMOTABLE.
Only ``upstream/evaluate.py`` on contest-compliant hardware over the exact
shipped bytes is a score.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from tac.submission_chain import (  # noqa: E402
    ChainPaths,
    ChainReceipt,
    SubmissionChainError,
    audit_runtime_tree,
    build_byte_ledger,
    git_hash,
    run_inflate,
    run_upstream_evaluate,
    sha256_file,
    utc_now,
    verify_archive_identity,
)

# --------------------------------------------------------------------------- #
# Vehicle profiles.  A profile is (repo_map, known frontier identity) -- the
# ONLY vehicle-specific knowledge in this file.  Adding a vehicle is a dict
# entry, not a new tool.
# --------------------------------------------------------------------------- #
PROFILES: dict[str, dict] = {
    "v4d_cx1": {
        "runtime_files": (
            "inflate.sh",
            "inflate_runner.py",
            "pfs1_warp_receiver.py",
            "ddm_tr1_runtime.py",
            "ddm_r7_token_coder.py",
            "ddm_ix2_archive_container.py",
            "repair_entropy_coder_runtime_adapters.py",
        ),
        # staged basename -> repo-relative source. The shipped name is NOT always
        # the repo name: inflate_runner.py IS experiments/inflate_runner_v4d.py.
        "repo_map": {
            "inflate_runner.py": "experiments/inflate_runner_v4d.py",
            "pfs1_warp_receiver.py": "src/tac/optimization/pfs1_warp_receiver.py",
            "ddm_tr1_runtime.py": "src/tac/optimization/ddm_tr1_runtime.py",
            "ddm_r7_token_coder.py": "experiments/ddm_r7_token_coder.py",
            "ddm_ix2_archive_container.py": "src/tac/optimization/ddm_ix2_archive_container.py",
            "repair_entropy_coder_runtime_adapters.py": (
                "src/tac/optimization/repair_entropy_coder_runtime_adapters.py"
            ),
        },
        # The live own-vehicle frontier (ddm_pu2, 2026-08-03), S=0.7910689
        # [macOS-CPU advisory]. This is the byte-identity acceptance target.
        "frontier_sha256": "c72ef357416b66e716b2863c4c49360306b80cc0fafd094e02394c8a4dd37209",
        "frontier_bytes": 353805,
    }
}


def _emit(payload: object) -> int:
    print(json.dumps(payload, indent=1, default=str))
    return 0


def _cmd_ledger(args: argparse.Namespace) -> int:
    led = build_byte_ledger(args.archive)
    return _emit(led.as_dict())


def _cmd_custody(args: argparse.Namespace) -> int:
    profile = PROFILES.get(args.profile, {})
    custody = audit_runtime_tree(
        args.runtime_src,
        repo_root=args.repo_root or _REPO,
        repo_map=profile.get("repo_map"),
    )
    out = custody.as_dict()
    # Surface the two verdicts a reader must not miss.
    out["_summary"] = {
        "diverged_from_repo_head": [
            f.staged_name for f in custody.files if f.verdict == "DIVERGED"
        ],
        "unmapped_no_repo_source": [
            f.staged_name for f in custody.files if f.verdict == "UNMAPPED"
        ],
        "unreached_dead_weight": list(custody.unreached_files),
        "note": (
            "reachability is TRANSITIVE; a one-level scan of the entry module "
            "misreports second-hop dependencies as dead weight."
        ),
    }
    return _emit(out)


def _cmd_verify(args: argparse.Namespace) -> int:
    profile = PROFILES.get(args.profile, {})
    res = verify_archive_identity(
        args.archive,
        expected_sha256=args.expect_sha or profile.get("frontier_sha256"),
        expected_bytes=args.expect_bytes or profile.get("frontier_bytes"),
    )
    if not res["byte_identical"]:
        res["_diagnosis"] = (
            "NOT byte-identical to the reference. Compare the byte ledger of "
            "both archives (`ledger` subcommand) to localise which SECTION "
            "diverges before re-running any expensive stage."
        )
    return _emit(res)


def _cmd_close(args: argparse.Namespace) -> int:
    """Full chain on an already-built archive: custody + ledger + inflate + eval."""
    paths = ChainPaths.from_env(
        repo_root=args.repo_root or _REPO,
        upstream_dir=args.upstream_dir,
        runtime_src=args.runtime_src,
        videos_dir=args.videos_dir,
        video_names_file=args.video_names_file,
        work_dir=args.work_dir,
    )
    missing = paths.preflight(require_eval_inputs=args.run_evaluate)
    if missing:
        raise SubmissionChainError(
            "path preflight FAILED before any expensive step; missing:\n  "
            + "\n  ".join(missing)
            + "\nSet them via flags or PACT_* environment variables."
        )

    profile = PROFILES.get(args.profile, {})
    archive = Path(args.archive)
    submission_dir = archive.parent

    receipt = ChainReceipt(
        utc=utc_now(),
        git_hash=git_hash(paths.repo_root),
        seed=args.seed,
        host_platform=f"{platform.system()}-{platform.machine()}",
        python_version=platform.python_version(),
        upstream_dir=str(paths.upstream_dir),
        archive_path=str(archive),
        archive_sha256=sha256_file(archive),
        archive_bytes=archive.stat().st_size,
    )
    receipt.byte_ledger = build_byte_ledger(archive).as_dict()
    receipt.runtime_custody = audit_runtime_tree(
        paths.runtime_src, repo_root=paths.repo_root, repo_map=profile.get("repo_map")
    ).as_dict()
    receipt.byte_identity = verify_archive_identity(
        archive,
        expected_sha256=args.expect_sha or profile.get("frontier_sha256"),
        expected_bytes=args.expect_bytes or profile.get("frontier_bytes"),
    )

    if args.run_inflate or args.run_evaluate:
        work = Path(paths.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        arch_dir = work / "archive"
        if arch_dir.exists():
            import shutil

            shutil.rmtree(arch_dir)
        arch_dir.mkdir(parents=True)
        import zipfile

        from tac.submission_archive import safe_extract_zip

        safe_extract_zip(archive, arch_dir)
        inflate_res = run_inflate(
            submission_dir,
            archive_dir=arch_dir,
            out_dir=submission_dir / "inflated",
            video_names_file=paths.video_names_file,
            timeout=args.inflate_timeout,
        )
        receipt.inflate = inflate_res.__dict__

    if args.run_evaluate:
        ev = run_upstream_evaluate(
            submission_dir,
            upstream_dir=paths.upstream_dir,
            videos_dir=paths.videos_dir,
            video_names_file=paths.video_names_file,
            archive_bytes=receipt.archive_bytes,
            device=args.eval_device,
            batch_size=args.batch_size,
            num_threads=args.num_threads,
            timeout=args.eval_timeout,
        )
        receipt.evaluate = ev.__dict__
        receipt.score_axis = ev.score_axis
        receipt.score_claim = ev.score_claim
    else:
        from tac.submission_chain import advisory_axis_label

        receipt.score_axis = advisory_axis_label()
        receipt.notes.append(
            "evaluate.py NOT run (--run-evaluate absent): this receipt carries "
            "byte accounting and custody ONLY, and contains no score."
        )

    if receipt.runtime_custody.get("diverged_count"):
        receipt.notes.append(
            f"{receipt.runtime_custody['diverged_count']} vendored runtime "
            "file(s) DIVERGE from repo HEAD: the shipped receiver is a PINNED "
            "copy. Re-staging from HEAD would ship a different receiver."
        )
    if receipt.runtime_custody.get("unmapped_count"):
        receipt.notes.append(
            f"{receipt.runtime_custody['unmapped_count']} staged file(s) have no "
            "mapped repo source; their provenance is UNVERIFIED."
        )

    out = Path(args.receipt or (submission_dir / "chain_receipt.json"))
    receipt.write(out)
    print(f"[submission-chain] receipt written: {out}", file=sys.stderr)
    return _emit(json.loads(out.read_text()))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default="v4d_cx1", choices=[*sorted(PROFILES), "none"])
    p.add_argument("--repo-root", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("ledger", help="emit the per-section byte ledger for an archive")
    q.add_argument("--archive", required=True)
    q.set_defaults(func=_cmd_ledger)

    q = sub.add_parser("custody", help="audit a vendored runtime tree against repo HEAD")
    q.add_argument("--runtime-src", required=True)
    q.set_defaults(func=_cmd_custody)

    q = sub.add_parser("verify", help="byte-identity control against a reference archive")
    q.add_argument("--archive", required=True)
    q.add_argument("--expect-sha", default=None)
    q.add_argument("--expect-bytes", type=int, default=None)
    q.set_defaults(func=_cmd_verify)

    q = sub.add_parser("close", help="full chain -> typed receipt")
    q.add_argument("--archive", required=True)
    q.add_argument("--runtime-src", default=None)
    q.add_argument("--upstream-dir", default=None)
    q.add_argument("--videos-dir", default=None)
    q.add_argument("--video-names-file", default=None)
    q.add_argument("--work-dir", default=None)
    q.add_argument("--receipt", default=None)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--expect-sha", default=None)
    q.add_argument("--expect-bytes", type=int, default=None)
    q.add_argument("--run-inflate", action="store_true")
    q.add_argument("--run-evaluate", action="store_true",
                   help="run the REAL n600 scorer (hours on CPU); implies --run-inflate")
    q.add_argument("--eval-device", default="cpu", choices=["cpu", "cuda"])
    q.add_argument("--batch-size", type=int, default=16)
    q.add_argument("--num-threads", type=int, default=2)
    q.add_argument("--inflate-timeout", type=int, default=7200)
    q.add_argument("--eval-timeout", type=int, default=24 * 3600)
    q.set_defaults(func=_cmd_close)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SubmissionChainError as exc:
        print(f"[submission-chain] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
