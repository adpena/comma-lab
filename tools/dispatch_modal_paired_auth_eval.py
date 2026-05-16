#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Dispatch paired Modal CPU+CUDA auth evals for one archive/runtime.

Plan-only by default. Pass ``--execute`` to spawn both detached Modal jobs.
The individual Modal wrappers still own lane claims and artifact recovery; this
tool is the canonical operator entry point so CPU/CUDA pairing is the default
instead of an afterthought.

PAIRED-DISPATCH-SKIP-IF-ANCHOR-EXISTS-ENHANCEMENT (2026-05-15)
--------------------------------------------------------------
``--skip-axis-if-promotable-anchor-exists`` opts the planner into auto-
detecting per-axis anchors that ALREADY exist for the archive sha and
skipping the per-axis re-dispatch. The skip-decision routes through
``tac.deploy.modal.anchor_lookup.find_promotable_anchor_for_axis_and_sha``,
which validates 4 custody invariants (axis-grade match / explicit
``score_claim_valid=True`` / exact archive_sha256 match / finite numeric
score) before declaring a hit. Empirical anchor: 2026-05-15 Z3 v2 FULL
paired dispatch re-fired CUDA for an archive sha that already had a
contest-CUDA anchor on disk.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
# Import the canonical anchor-lookup helper. We deliberately add REPO_ROOT/src
# to sys.path so this CLI works when invoked outside an installed environment;
# the package's normal import surface is still preferred when available.
_SRC_DIR = REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tac.deploy.claims import DispatchClaimSpec, terminal_dispatch_claim  # noqa: E402
from tac.deploy.modal.anchor_lookup import (  # noqa: E402
    find_promotable_anchor_for_axis_and_sha,
)
from tac.deploy.modal.auth_eval import (  # noqa: E402
    modal_uploaded_submission_dir_runtime_manifest,
)
from tac.deploy.modal.paired_dispatch import (  # noqa: E402
    PAIRED_AUTH_EVAL_DEFAULT_CLAIM_AGENT,
)

CUDA_REMOTE_SUBMISSION_DIR = "/tmp/modal_auth_eval/submission_dir"
CPU_REMOTE_SUBMISSION_DIR = "/tmp/modal_auth_eval_cpu/submission_dir"
AUTO_RUNTIME_TREE = "auto"
DEFAULT_REPO_INFLATE_SH = "submissions/robust_current/inflate.sh"
DEFAULT_UPLOADED_SUBMISSION_INFLATE_SH = "inflate.sh"


def _utc_now_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "modal_pair"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _optional_arg(cmd: list[str], flag: str, value: str) -> None:
    if value:
        cmd.extend([flag, value])


def _is_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(value or "").strip()))


def _validate_expected_archive_sha256(*, expected: str, actual: str) -> None:
    expected = str(expected or "").strip().lower()
    if not expected:
        return
    if not _is_sha256(expected):
        raise ValueError(
            f"expected archive sha must be a 64-char sha256; got {expected!r}"
        )
    if expected != str(actual or "").strip().lower():
        raise ValueError(
            "expected archive sha does not match selected archive: "
            f"expected={expected} actual={actual}"
        )


def _normalize_inflate_sh_for_submission_dir(
    *, submission_dir: str, inflate_sh: str
) -> str:
    """Return the Modal-wrapper inflate path for an uploaded submission dir.

    ``experiments/modal_auth_eval*.py`` treats ``--inflate-sh`` as relative to
    the uploaded ``--submission-dir`` when that directory is provided. Operators
    naturally pass the local full path (`candidate/submission_dir/inflate.sh`);
    normalize that to `inflate.sh` here so paired dispatch remains the safe
    default instead of failing after Modal app startup.
    """
    if not submission_dir or not inflate_sh:
        return inflate_sh
    submission_path = Path(submission_dir)
    inflate_path = Path(inflate_sh)
    try:
        rel = inflate_path.resolve().relative_to(submission_path.resolve())
    except ValueError:
        try:
            rel = inflate_path.relative_to(submission_path)
        except ValueError:
            return inflate_sh
    return rel.as_posix()


def _resolve_cli_inflate_sh_default(*, submission_dir: str, inflate_sh: str | None) -> str:
    """Resolve the context-aware CLI default for ``--inflate-sh``.

    With ``--submission-dir``, Modal uploads that directory as the runtime root
    and the Modal wrappers interpret ``--inflate-sh`` relative to it. The safe
    default is therefore ``inflate.sh`` inside the uploaded tree, not the repo's
    robust-current runtime.
    """

    if inflate_sh:
        return str(inflate_sh)
    if submission_dir:
        return DEFAULT_UPLOADED_SUBMISSION_INFLATE_SH
    return DEFAULT_REPO_INFLATE_SH


def _modal_uploaded_runtime_hashes_for_axis(
    *,
    submission_dir: str,
    inflate_sh_rel: str,
    remote_submission_dir: str,
) -> dict[str, str]:
    """Return the runtime hashes the Modal wrapper will enforce remotely."""
    if not submission_dir:
        return {}
    from experiments.contest_auth_eval import _runtime_dependency_manifest

    submission_path = Path(submission_dir).resolve()
    local_inflate_sh = (submission_path / inflate_sh_rel).resolve()
    local_manifest = _runtime_dependency_manifest(local_inflate_sh, REPO_ROOT / "upstream")
    projected = modal_uploaded_submission_dir_runtime_manifest(
        local_manifest,
        remote_submission_dir=remote_submission_dir,
    )
    return {
        "runtime_tree_sha256": str(projected.get("runtime_tree_sha256") or ""),
        "runtime_content_tree_sha256": str(
            projected.get("runtime_content_tree_sha256") or ""
        ),
    }


def _resolve_axis_runtime_expectations(
    *,
    submission_dir: str,
    inflate_sh_rel: str,
    expected_runtime_tree_sha256: str,
    expected_cuda_runtime_tree_sha256: str = "",
    expected_cpu_runtime_tree_sha256: str = "",
) -> dict[str, Any]:
    """Resolve per-axis runtime-tree hashes before Modal app startup.

    Uploaded ``--submission-dir`` runtimes hash differently on CUDA and CPU
    Modal wrappers because the remote extraction roots differ. The paired
    launcher therefore must either compute per-axis hashes locally or reject a
    legacy single hash before provider work starts.
    """

    common = str(expected_runtime_tree_sha256 or "").strip()
    cuda_expected = str(expected_cuda_runtime_tree_sha256 or common).strip()
    cpu_expected = str(expected_cpu_runtime_tree_sha256 or common).strip()
    should_auto = (
        bool(submission_dir)
        and (
            not cuda_expected
            or not cpu_expected
            or cuda_expected.lower() == AUTO_RUNTIME_TREE
            or cpu_expected.lower() == AUTO_RUNTIME_TREE
            or common.lower() == AUTO_RUNTIME_TREE
        )
    )
    should_validate = bool(submission_dir) and (
        should_auto or bool(cuda_expected) or bool(cpu_expected)
    )
    observed: dict[str, dict[str, str]] = {}
    if should_validate:
        observed = {
            "contest_cuda": _modal_uploaded_runtime_hashes_for_axis(
                submission_dir=submission_dir,
                inflate_sh_rel=inflate_sh_rel,
                remote_submission_dir=CUDA_REMOTE_SUBMISSION_DIR,
            ),
            "contest_cpu": _modal_uploaded_runtime_hashes_for_axis(
                submission_dir=submission_dir,
                inflate_sh_rel=inflate_sh_rel,
                remote_submission_dir=CPU_REMOTE_SUBMISSION_DIR,
            ),
        }
        if should_auto:
            if not cuda_expected or cuda_expected.lower() == AUTO_RUNTIME_TREE:
                cuda_expected = observed["contest_cuda"]["runtime_tree_sha256"]
            if not cpu_expected or cpu_expected.lower() == AUTO_RUNTIME_TREE:
                cpu_expected = observed["contest_cpu"]["runtime_tree_sha256"]

    for axis, expected in (
        ("contest_cuda", cuda_expected),
        ("contest_cpu", cpu_expected),
    ):
        if expected and not _is_sha256(expected):
            raise ValueError(
                f"{axis} expected runtime tree must be a 64-char sha256 or 'auto'; "
                f"got {expected!r}"
            )
        actual = observed.get(axis, {}).get("runtime_tree_sha256")
        if actual and expected and expected != actual:
            content = observed.get(axis, {}).get("runtime_content_tree_sha256", "")
            raise ValueError(
                "expected runtime tree does not match the Modal uploaded "
                f"--submission-dir runtime for {axis}: expected={expected} "
                f"actual={actual} runtime_content_tree_sha256={content}. "
                "Use --expected-runtime-tree-sha256 auto or the per-axis "
                "--expected-cuda-runtime-tree-sha256/--expected-cpu-runtime-tree-sha256 "
                "flags."
            )

    return {
        "contest_cuda": cuda_expected,
        "contest_cpu": cpu_expected,
        "observed_uploaded_runtime": observed,
        "common_expected_runtime_tree_sha256": (
            cuda_expected if cuda_expected and cuda_expected == cpu_expected else None
        ),
    }


def _terminal_reused_anchor_claim(
    *,
    repo_root: Path,
    claims_root: Path,
    axis: str,
    plan: dict[str, Any],
    anchor: dict[str, Any],
    claim_agent: str,
    pair_group_id: str,
    run_id: str,
) -> str:
    """Record terminal custody for a paired-dispatch axis reused from an anchor."""

    lane_id = str(plan["lanes"][axis])
    job_id = f"{run_id}_{axis}_reused_anchor"
    archive_sha = str(plan["archive"]["sha256"])
    notes = (
        "paired Modal auth eval reused existing promotable anchor; "
        f"axis={axis}; pair_group_id={pair_group_id}; archive_sha={archive_sha}; "
        f"anchor_result_path={anchor.get('result_path')}; "
        f"anchor_score={anchor.get('score')}; "
        f"anchor_source={anchor.get('source')}; "
        f"anchor_runtime_tree_sha256={anchor.get('runtime_tree_sha256')}"
    )
    terminal_dispatch_claim(
        repo_root=repo_root,
        spec=DispatchClaimSpec(
            lane_id=lane_id,
            instance_job_id=job_id,
            agent=claim_agent,
            platform="modal",
        ),
        status="completed_reused_existing_anchor",
        notes=notes,
        python_executable=sys.executable,
        claim_tool=REPO_ROOT / "tools" / "claim_lane_dispatch.py",
        claims_path=claims_root / ".omx" / "state" / "active_lane_dispatch_claims.md",
    )
    return job_id


def _resolve_skipped_axes(
    *,
    archive_sha256: str,
    expected_runtime_tree_sha256_by_axis: dict[str, str],
    skip_if_anchor_exists: bool,
    repo_root: Path,
) -> dict[str, dict[str, Any] | None]:
    """Return ``{"contest_cuda": <anchor|None>, "contest_cpu": <anchor|None>}``.

    Each entry is the anchor dict the lookup helper returned (when an
    existing promotable anchor was found) or ``None`` (dispatch normally).
    When ``skip_if_anchor_exists=False`` this is always ``{cuda: None, cpu: None}``
    so the planner falls back to the historical behavior (always dispatch).
    """
    if not skip_if_anchor_exists:
        return {"contest_cuda": None, "contest_cpu": None}
    cuda_runtime = expected_runtime_tree_sha256_by_axis.get("contest_cuda", "")
    cpu_runtime = expected_runtime_tree_sha256_by_axis.get("contest_cpu", "")
    if not cuda_runtime and not cpu_runtime:
        return {"contest_cuda": None, "contest_cpu": None}
    return {
        "contest_cuda": find_promotable_anchor_for_axis_and_sha(
            "cuda",
            archive_sha256,
            repo_root=repo_root,
            expected_runtime_tree_sha256=cuda_runtime,
        ),
        "contest_cpu": find_promotable_anchor_for_axis_and_sha(
            "cpu",
            archive_sha256,
            repo_root=repo_root,
            expected_runtime_tree_sha256=cpu_runtime,
        ),
    }


def build_plan(
    *,
    archive: Path,
    submission_dir: str,
    inflate_sh: str,
    run_id: str,
    pair_group_id: str,
    lane_id_base: str,
    output_root: Path,
    modal_bin: str,
    gpu: str,
    claim_agent: str,
    claim_notes: str,
    expected_runtime_tree_sha256: str = "",
    expected_cuda_runtime_tree_sha256: str = "",
    expected_cpu_runtime_tree_sha256: str = "",
    expected_archive_sha256: str = "",
    skip_axis_if_promotable_anchor_exists: bool = False,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    archive = archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"archive not found: {archive}")
    archive_sha = _sha256(archive)
    _validate_expected_archive_sha256(expected=expected_archive_sha256, actual=archive_sha)
    archive_bytes = archive.stat().st_size
    notes = claim_notes or (
        "paired Modal auth eval; same archive/runtime required on contest_cuda and contest_cpu axes"
    )
    inflate_sh_for_cmd = _normalize_inflate_sh_for_submission_dir(
        submission_dir=submission_dir,
        inflate_sh=inflate_sh,
    )
    runtime_expectations = _resolve_axis_runtime_expectations(
        submission_dir=submission_dir,
        inflate_sh_rel=inflate_sh_for_cmd,
        expected_runtime_tree_sha256=expected_runtime_tree_sha256,
        expected_cuda_runtime_tree_sha256=expected_cuda_runtime_tree_sha256,
        expected_cpu_runtime_tree_sha256=expected_cpu_runtime_tree_sha256,
    )
    expected_runtime_by_axis = {
        "contest_cuda": str(runtime_expectations.get("contest_cuda") or ""),
        "contest_cpu": str(runtime_expectations.get("contest_cpu") or ""),
    }
    cuda_output = output_root / "modal_auth_eval" / f"{run_id}_cuda"
    cpu_output = output_root / "modal_auth_eval_cpu" / f"{run_id}_cpu"
    cuda_lane = f"{lane_id_base}_contest_cuda"
    cpu_lane = f"{lane_id_base}_contest_cpu"

    cuda_cmd = [
        modal_bin,
        "run",
        "--detach",
        "experiments/modal_auth_eval.py",
        "--archive",
        str(archive),
        "--expected-archive-sha256",
        archive_sha,
        "--inflate-sh",
        inflate_sh_for_cmd,
        "--output-dir",
        str(cuda_output),
        "--gpu",
        gpu,
        "--detach",
        "--provider-detach-ack",
        "--pair-group-id",
        pair_group_id,
        "--lane-id",
        cuda_lane,
        "--instance-job-id",
        f"{run_id}_cuda",
        "--claim-agent",
        claim_agent,
        "--claim-notes",
        f"{notes}; pair_group_id={pair_group_id}; axis=contest_cuda; archive_sha={archive_sha}; bytes={archive_bytes}",
    ]
    cpu_cmd = [
        modal_bin,
        "run",
        "--detach",
        "experiments/modal_auth_eval_cpu.py",
        "--archive",
        str(archive),
        "--expected-archive-sha256",
        archive_sha,
        "--inflate-sh",
        inflate_sh_for_cmd,
        "--output-dir",
        str(cpu_output),
        "--detach",
        "--provider-detach-ack",
        "--pair-group-id",
        pair_group_id,
        "--lane-id",
        cpu_lane,
        "--instance-job-id",
        f"{run_id}_cpu",
        "--claim-agent",
        claim_agent,
        "--claim-notes",
        f"{notes}; pair_group_id={pair_group_id}; axis=contest_cpu; archive_sha={archive_sha}; bytes={archive_bytes}",
    ]
    _optional_arg(cuda_cmd, "--submission-dir", submission_dir)
    _optional_arg(cpu_cmd, "--submission-dir", submission_dir)
    _optional_arg(
        cuda_cmd,
        "--expected-runtime-tree-sha256",
        expected_runtime_by_axis["contest_cuda"],
    )
    _optional_arg(
        cpu_cmd,
        "--expected-runtime-tree-sha256",
        expected_runtime_by_axis["contest_cpu"],
    )

    # Resolve per-axis skip-decisions before assembling the plan dict so the
    # plan carries an authoritative record of which axes were re-used vs
    # freshly dispatched.
    skip_root = repo_root if repo_root is not None else REPO_ROOT
    skipped = _resolve_skipped_axes(
        archive_sha256=archive_sha,
        expected_runtime_tree_sha256_by_axis=expected_runtime_by_axis,
        skip_if_anchor_exists=skip_axis_if_promotable_anchor_exists,
        repo_root=skip_root,
    )
    cuda_anchor = skipped["contest_cuda"]
    cpu_anchor = skipped["contest_cpu"]
    cuda_skipped = cuda_anchor is not None
    cpu_skipped = cpu_anchor is not None

    plan_notes = [
        "This tool is the default Modal auth-eval entry point for score-bearing archives.",
        "Both commands carry the same pair_group_id and exact archive SHA.",
        "Single-axis Modal wrapper use requires an explicit waiver reason.",
    ]
    if skip_axis_if_promotable_anchor_exists:
        plan_notes.append(
            "skip_axis_if_promotable_anchor_exists=True; per-axis dispatch skipped if a "
            "promotable anchor already exists for the archive sha and expected runtime tree "
            "(anchor_lookup.find_promotable_anchor_for_axis_and_sha)."
        )
        if not expected_runtime_tree_sha256:
            plan_notes.append(
                "No expected_runtime_tree_sha256 was supplied, so runtime-bound anchor reuse "
                "is disabled unless --submission-dir allowed the paired launcher to compute "
                "per-axis Modal uploaded runtime-tree hashes."
            )

    return {
        "schema": "modal_paired_auth_eval_dispatch_plan_v2",
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "pair_group_id": pair_group_id,
        "required_axes": ["contest_cuda", "contest_cpu"],
        "archive": {
            "path": str(archive),
            "bytes": archive_bytes,
            "sha256": archive_sha,
            "expected_sha256": expected_archive_sha256 or archive_sha,
            "expected_sha256_match": True,
        },
        "runtime": {
            "submission_dir": submission_dir or None,
            "inflate_sh": inflate_sh_for_cmd,
            "inflate_sh_original": inflate_sh if inflate_sh_for_cmd != inflate_sh else None,
            "expected_runtime_tree_sha256": runtime_expectations.get(
                "common_expected_runtime_tree_sha256"
            ),
            "expected_runtime_tree_sha256_by_axis": {
                key: (value or None) for key, value in expected_runtime_by_axis.items()
            },
            "expected_runtime_content_tree_sha256_by_axis": {
                axis: (
                    runtime_expectations.get("observed_uploaded_runtime", {})
                    .get(axis, {})
                    .get("runtime_content_tree_sha256")
                )
                for axis in ("contest_cuda", "contest_cpu")
            },
            "modal_uploaded_runtime_manifest_by_axis": runtime_expectations.get(
                "observed_uploaded_runtime", {}
            ),
        },
        "outputs": {
            "contest_cuda": str(cuda_output),
            "contest_cpu": str(cpu_output),
        },
        "lanes": {
            "contest_cuda": cuda_lane,
            "contest_cpu": cpu_lane,
        },
        "commands": {
            "contest_cuda": cuda_cmd,
            "contest_cpu": cpu_cmd,
        },
        "skip_axis_if_promotable_anchor_exists": bool(skip_axis_if_promotable_anchor_exists),
        "axes_skipped_due_to_existing_anchor": {
            "contest_cuda": cuda_skipped,
            "contest_cpu": cpu_skipped,
        },
        "existing_anchors_reused": {
            "contest_cuda": cuda_anchor,
            "contest_cpu": cpu_anchor,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "notes": plan_notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", default="")
    parser.add_argument("--submission-dir", default="")
    parser.add_argument(
        "--inflate-sh",
        default=None,
        help=(
            "Inflate entrypoint. Defaults to inflate.sh when --submission-dir "
            "uploads a runtime tree; otherwise defaults to "
            f"{DEFAULT_REPO_INFLATE_SH}."
        ),
    )
    parser.add_argument("--label", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--pair-group-id", default="")
    parser.add_argument("--lane-id-base", default="")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results"))
    parser.add_argument("--modal-bin", default=".venv/bin/modal")
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--claim-agent", default=PAIRED_AUTH_EVAL_DEFAULT_CLAIM_AGENT)
    parser.add_argument("--claim-notes", default="")
    parser.add_argument(
        "--expected-runtime-tree-sha256",
        default="",
        help=(
            "Legacy/common runtime-tree expectation. With --submission-dir, pass "
            "'auto' (or omit the flag) to compute the Modal-uploaded per-axis "
            "runtime-tree hashes locally before provider startup."
        ),
    )
    parser.add_argument(
        "--expected-cuda-runtime-tree-sha256",
        default="",
        help="Contest-CUDA Modal uploaded runtime-tree hash, or 'auto'.",
    )
    parser.add_argument(
        "--expected-cpu-runtime-tree-sha256",
        default="",
        help="Contest-CPU Modal uploaded runtime-tree hash, or 'auto'.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--skip-axis-if-promotable-anchor-exists",
        action="store_true",
        default=False,
        help=(
            "Auto-skip per-axis re-dispatch when a promotable contest anchor "
            "already exists for the archive sha. Routes through "
            "tac.deploy.modal.anchor_lookup.find_promotable_anchor_for_axis_and_sha "
            "(custody-validates evidence_grade + score_claim_valid + "
            "archive_sha256 + finite-numeric score). Default False preserves "
            "the historical always-fire-both-axes behavior. "
            "PAIRED-DISPATCH-SKIP-IF-ANCHOR-EXISTS-ENHANCEMENT 2026-05-15 "
            "addressed empirical Z3 v2 FULL paired-dispatch redundant CUDA "
            "re-fire on a sha that already had a promotable anchor on disk."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Repository root used by --skip-axis-if-promotable-anchor-exists "
            "to scan for existing anchors. Defaults to Path.cwd() so the "
            "scope matches the directory the operator invoked the CLI from."
        ),
    )
    args = parser.parse_args()

    label = _safe_slug(args.label or args.archive.stem)
    run_id = _safe_slug(args.run_id or f"{label}_paired_modal_auth_{_utc_now_compact()}")
    pair_group_id = _safe_slug(args.pair_group_id or run_id)
    lane_id_base = _safe_slug(args.lane_id_base or f"lane_{pair_group_id}")
    resolved_repo_root = (args.repo_root or Path.cwd()).resolve()
    inflate_sh = _resolve_cli_inflate_sh_default(
        submission_dir=args.submission_dir,
        inflate_sh=args.inflate_sh,
    )
    try:
        plan = build_plan(
            archive=args.archive,
            submission_dir=args.submission_dir,
            inflate_sh=inflate_sh,
            run_id=run_id,
            pair_group_id=pair_group_id,
            lane_id_base=lane_id_base,
            output_root=args.output_root,
            modal_bin=args.modal_bin,
            gpu=args.gpu,
            claim_agent=args.claim_agent,
            claim_notes=args.claim_notes,
            expected_runtime_tree_sha256=args.expected_runtime_tree_sha256,
            expected_cuda_runtime_tree_sha256=args.expected_cuda_runtime_tree_sha256,
            expected_cpu_runtime_tree_sha256=args.expected_cpu_runtime_tree_sha256,
            expected_archive_sha256=args.expected_archive_sha256,
            skip_axis_if_promotable_anchor_exists=args.skip_axis_if_promotable_anchor_exists,
            repo_root=resolved_repo_root,
        )
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(plan, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not args.execute:
        return 0

    # Honor per-axis skip-decisions resolved by build_plan. When BOTH axes
    # have promotable anchors on disk we exit cleanly with rc=0 and a loud
    # log line so downstream harvesters know to re-point at the existing
    # anchor instead of waiting for fresh JSONs that will never appear.
    skipped_map = plan.get("axes_skipped_due_to_existing_anchor", {})
    anchors_map = plan.get("existing_anchors_reused", {})
    cuda_skipped = bool(skipped_map.get("contest_cuda"))
    cpu_skipped = bool(skipped_map.get("contest_cpu"))

    if args.skip_axis_if_promotable_anchor_exists and (cuda_skipped or cpu_skipped):
        for axis in ("contest_cuda", "contest_cpu"):
            if not skipped_map.get(axis):
                continue
            anchor = anchors_map.get(axis) or {}
            short_sha = (plan["archive"]["sha256"] or "")[:12]
            print(
                "[ANCHOR-EXISTS-SKIP]"
                f" axis={axis}"
                f" sha={short_sha}"
                f" score={anchor.get('score')!r}"
                f" reusing existing anchor at {anchor.get('result_path')!r}"
                f" (source={anchor.get('source')!r})",
                flush=True,
            )
            # Write re-pointer manifest so harvest tools find the existing
            # anchor under THIS dispatch's label rather than re-scanning.
            output_dir = Path(plan["outputs"][axis])
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                terminal_claim_job_id = _terminal_reused_anchor_claim(
                    repo_root=REPO_ROOT,
                    claims_root=resolved_repo_root,
                    axis=axis,
                    plan=plan,
                    anchor=anchor,
                    claim_agent=args.claim_agent,
                    pair_group_id=pair_group_id,
                    run_id=run_id,
                )
                repointer = output_dir / f"anchor_repointer_{axis}.json"
                repointer.write_text(
                    json.dumps(
                        {
                            "schema": "anchor_repointer_v1",
                            "axis": axis,
                            "archive_sha256": plan["archive"]["sha256"],
                            "pair_group_id": pair_group_id,
                            "reused_anchor": anchor,
                            "terminal_dispatch_claim": {
                                "lane_id": plan["lanes"][axis],
                                "instance_job_id": terminal_claim_job_id,
                                "status": "completed_reused_existing_anchor",
                                "claims_path": str(
                                    resolved_repo_root
                                    / ".omx"
                                    / "state"
                                    / "active_lane_dispatch_claims.md"
                                ),
                            },
                            "written_at_utc": dt.datetime.now(dt.UTC)
                            .isoformat(timespec="seconds")
                            .replace("+00:00", "Z"),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            except OSError as exc:
                print(
                    f"[ANCHOR-EXISTS-SKIP] WARNING: failed to write repointer for axis={axis}: {exc!r}",
                    flush=True,
                )

    if cuda_skipped and cpu_skipped:
        print(
            "[BOTH_AXES_REUSED_FROM_ANCHORS] Both contest_cuda and contest_cpu"
            " anchors already exist for this archive sha;"
            " no Modal dispatch fired (cost saved).",
            flush=True,
        )
        return 0

    for axis in ("contest_cuda", "contest_cpu"):
        if skipped_map.get(axis):
            continue
        proc = subprocess.run(plan["commands"][axis], cwd=REPO_ROOT)
        if proc.returncode:
            return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
