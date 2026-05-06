#!/usr/bin/env python3
"""All-lanes preflight: run every available dispatch dry-run in sequence.

Single command for full pre-dispatch confidence across every launch-ready
PR106-stacking lane. Each sub-tool runs at $0 cost (CPU-only); failures
in any one are reported but DON'T block the others (each lane stands alone).

Currently runs:
  Gate #0: tools/check_dispatch_cli_shell_hazards.py --strict
           (dispatch typo/stale-flag/macOS shell portability guard)
  Gate #1: tools/audit_reverse_engineering_tree.py --strict --summary
           (curated public-submission deconstruction / no raw artifact leaks)
  Gate #2: tools/list_hidden_gems.py --format json
           (hidden-gem registry imports, schema validates, and stays provider-free)
  Gate #3: tools/audit_hidden_gem_readiness.py --status ready_for_patch
           (ready-for-patch hidden gems point at live evidence and targets)
  Gate #4: tools/audit_engineered_corrections.py --self-test
           (engineered-correction payload guard imports and fails closed before dispatch)
  Gate #5: tools/audit_hnerv_frontier_scorecard.py
           (public HNeRV scorecard is fresh enough to route hidden-gem follow-ups)
  Gate #6: tools/build_hnerv_lowlevel_repack_candidate.py
           (real PR106x byte-candidate proof with raw brotli equality; still
            not exact-eval dispatch-ready)
  Gate #7: tools/audit_tooling_consolidation.py
           (advisory inventory of duplicated audit/preflight helper patterns)
  Gate #8: tools/audit_recovered_remote_lanes.py
           (recovered lane scripts have canonical custody/proxy classification)
  Gate #9: tools/audit_untracked_source_artifacts.py
           (no-signal-loss inventory of untracked source/research; strict when
            the reviewed disposition manifest exists)
  Gate #10: tools/audit_orphan_recovery_canonicalization.py
           (orphan pyc recovery deletions must have tracked canonical copies)
  Gate #11: tools/audit_preserved_orphans.py
           (local preserved-orphan shadows must be duplicate, superseded, or absent)
  Gate #12: tools/audit_recovery_custody_snapshots.py
           (unique pyc/signal-loss custody snapshots are intact and explicit)
  Gate #13: tools/audit_reverse_engineering_tree.py --release-strict
           --release-manifest ...
           (public-release reverse-engineering custody is explicitly curated)
  Gate #14: tools/audit_release_index_split.py --strict
           (no staged rollback shadows; no staged private provider/runtime
            state in release commits)
  Gate #15: tools/audit_nested_gitlink_custody.py --strict
           (dirty public-intake/raw-custody gitlinks must be documented and
            their inner dirty status must be visible)
  Gate #16: tools/audit_staged_public_release_hygiene.py --strict
           (staged docs/site/readme public surfaces contain no private paths,
            provider job links, or credential-like strings)
  Lane #1: tools/dispatch_dryrun_apogee_intN.py --all-pareto-frontier
           --allow-forensic-byte-only
           (self-protection check: Apogee intN remains byte-only and blocked
            for exact-eval dispatch until a distortion model or exact evidence exists)
  Lane #2: tools/dispatch_dryrun_omega_w_v3.py
           (default: local smoke only; --require-real-omega-sensitivity
            promotes to remote-dispatch readiness)
  Lane #3: tools/dispatch_dryrun_pr106_sidechannels.py --skip-help-subprocess
           (latent/yshift/LRL1/stacked PR106 sidechannel readiness surfaces)

Lane SJ-KL is intentionally NOT included — its end-to-end local mode
requires real prepared pair tensors (heavy fixture). Its codec library
is covered by 26 tests in test_sjkl_basis.py + the legacy-artifact
forensic drift test landed earlier this session.

Exit code: 0 if every lane PASSES, non-zero count of FAILED lanes otherwise.

Usage:
  .venv/bin/python tools/all_lanes_preflight.py
  .venv/bin/python tools/all_lanes_preflight.py -v   # verbose (PASS lines too)
  .venv/bin/python tools/all_lanes_preflight.py --require-real-omega-sensitivity
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)

from tac.repo_io import json_text  # noqa: E402

TOOLS = REPO / "tools"
SHELL_HAZARDS = TOOLS / "check_dispatch_cli_shell_hazards.py"
REVERSE_ENGINEERING_AUDIT = TOOLS / "audit_reverse_engineering_tree.py"
HIDDEN_GEMS_REGISTRY = TOOLS / "list_hidden_gems.py"
HIDDEN_GEMS_READINESS = TOOLS / "audit_hidden_gem_readiness.py"
ENGINEERED_CORRECTIONS_AUDIT = TOOLS / "audit_engineered_corrections.py"
HNERV_SCORECARD_AUDIT = TOOLS / "audit_hnerv_frontier_scorecard.py"
HNERV_LOWLEVEL_REPACK = TOOLS / "build_hnerv_lowlevel_repack_candidate.py"
TOOLING_CONSOLIDATION_AUDIT = TOOLS / "audit_tooling_consolidation.py"
RECOVERED_REMOTE_LANES_AUDIT = TOOLS / "audit_recovered_remote_lanes.py"
UNTRACKED_SOURCE_AUDIT = TOOLS / "audit_untracked_source_artifacts.py"
UNTRACKED_SOURCE_DISPOSITION_MANIFEST = REPO / ".omx/research/untracked_source_dispositions_20260505_codex.json"
ORPHAN_RECOVERY_AUDIT = TOOLS / "audit_orphan_recovery_canonicalization.py"
PRESERVED_ORPHANS_AUDIT = TOOLS / "audit_preserved_orphans.py"
RECOVERY_CUSTODY_SNAPSHOTS_AUDIT = TOOLS / "audit_recovery_custody_snapshots.py"
RELEASE_INDEX_SPLIT_AUDIT = TOOLS / "audit_release_index_split.py"
NESTED_GITLINK_CUSTODY_AUDIT = TOOLS / "audit_nested_gitlink_custody.py"
STAGED_PUBLIC_RELEASE_HYGIENE_AUDIT = TOOLS / "audit_staged_public_release_hygiene.py"
LOCAL_CUSTODY_RELEASE_MANIFEST = REPO / ".omx/research/local_custody_release_manifest_20260505_codex.json"
REVERSE_ENGINEERING_RELEASE_MANIFEST = (
    REPO / ".omx/research/reverse_engineering_release_manifest_20260505_codex.json"
)
HNERV_SCORECARD = (
    REPO
    / "experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/scorecard.json"
)
PR106X_ARCHIVE = (
    REPO
    / "experiments/results/lightning_batch/"
    / "exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/archive.zip"
)

LANES = [
    {
        "name": "apogee_intN (PR106 HNeRV signed-intN block-FP; forensic-only)",
        "tool": TOOLS / "dispatch_dryrun_apogee_intN.py",
        "args": ["--all-pareto-frontier", "--allow-forensic-byte-only"],
        "forensic_only": True,
    },
    {
        "name": "Lane Ω-W-V3 (water-fill v2 → PR106 HNeRV decoder)",
        "tool": TOOLS / "dispatch_dryrun_omega_w_v3.py",
        "args": [],
        "local_smoke_only": True,
    },
    {
        "name": "PR106 sidechannels (latent/yshift/LRL1/stacked)",
        "tool": TOOLS / "dispatch_dryrun_pr106_sidechannels.py",
        "args": ["--skip-help-subprocess"],
    },
]


@dataclass(frozen=True)
class PreflightStep:
    """One ordered preflight step that can execute independently."""

    section: str
    number: int
    name: str
    runner: Callable[[], tuple[bool, str]]
    pass_summary: str
    fail_summary: str
    forensic_only: bool = False
    local_smoke_only: bool = False


@dataclass(frozen=True)
class PreflightResult:
    """Captured step output rendered later in deterministic order."""

    step: PreflightStep
    passed: bool
    output: str
    elapsed_s: float


def _run_lane(lane: dict, verbose: bool) -> tuple[bool, str]:
    args = [sys.executable, str(lane["tool"])] + lane["args"]
    if verbose:
        args.append("--verbose")
    proc = subprocess.run(args, capture_output=True, text=True)
    return proc.returncode == 0, proc.stdout + proc.stderr


def _run_gate(name: str, tool: Path, extra_args: list[str] | None = None) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(tool), "--repo-root", str(REPO), "--strict", *(extra_args or [])],
        capture_output=True,
        text=True,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode == 0 and not output.strip():
        output = f"{name}: PASS"
    return proc.returncode == 0, output


def _run_hidden_gems_gate() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(HIDDEN_GEMS_REGISTRY), "--format", "json"],
        capture_output=True,
        text=True,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return False, output
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return False, f"hidden-gem registry emitted invalid JSON: {exc}\n{output}"
    entry_count = int(payload.get("entry_count", 0))
    if entry_count <= 0:
        return False, "hidden-gem registry emitted no entries"
    return True, f"hidden-gem registry: PASS ({entry_count} entries)"


def _run_hidden_gem_readiness_gate() -> tuple[bool, str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(HIDDEN_GEMS_READINESS),
            "--format",
            "json",
            "--status",
            "ready_for_patch",
            "--fail-if-missing-targets",
            "--fail-if-missing-evidence",
        ],
        capture_output=True,
        text=True,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return False, output
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return False, f"hidden-gem readiness emitted invalid JSON: {exc}\n{output}"
    summary = payload.get("summary", {})
    entry_count = int(summary.get("entry_count", 0))
    local_patch_count = int(summary.get("eligible_for_local_patch_count", 0))
    dispatch_ready = int(summary.get("ready_for_exact_eval_dispatch_count", 0))
    if dispatch_ready != 0:
        return False, "hidden-gem readiness must not unlock exact-eval dispatch"
    return True, (
        "hidden-gem readiness: PASS "
        f"({local_patch_count}/{entry_count} ready-for-patch rows locally actionable; "
        "0 exact-eval-dispatch-ready)"
    )


def _run_engineered_corrections_gate() -> tuple[bool, str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(ENGINEERED_CORRECTIONS_AUDIT),
            "--self-test",
            "--max-packed-bytes",
            "10000",
            "--fail-if-not-ready",
        ],
        capture_output=True,
        text=True,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return False, output
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return False, f"engineered-correction audit emitted invalid JSON: {exc}\n{output}"
    if payload.get("ready_for_exact_eval_dispatch") is not False:
        return False, "engineered-correction audit must never unlock exact-eval dispatch"
    if payload.get("ready_for_local_patch") is not True:
        return False, output
    return True, "engineered-correction readiness: PASS (self-test local-patch only; 0 exact-eval-dispatch-ready)"


def _run_hnerv_lowlevel_repack_gate() -> tuple[bool, str]:
    if not HNERV_SCORECARD.is_file():
        return False, f"missing HNeRV scorecard: {HNERV_SCORECARD.relative_to(REPO)}"
    if not PR106X_ARCHIVE.is_file():
        return False, f"missing PR106x archive: {PR106X_ARCHIVE.relative_to(REPO)}"
    with tempfile.TemporaryDirectory(prefix="pact_hnerv_lowlevel_preflight_") as tmp:
        tmpdir = Path(tmp)
        json_out = tmpdir / "manifest.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(HNERV_LOWLEVEL_REPACK),
                "--source-archive",
                str(PR106X_ARCHIVE),
                "--scorecard",
                str(HNERV_SCORECARD),
                "--source-label",
                "PR106x",
                "--output-dir",
                str(tmpdir),
                "--json-out",
                str(json_out),
                "--fail-if-blocked",
            ],
            capture_output=True,
            text=True,
        )
        output = proc.stdout + proc.stderr
        if proc.returncode != 0:
            return False, output
        try:
            payload = json.loads(json_out.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"HNeRV low-level repack emitted invalid manifest: {exc}\n{output}"
    audit = payload.get("candidate_diff_audit") or {}
    raw = payload.get("brotli_raw_equivalence") or []
    if payload.get("ready_for_exact_eval_dispatch") is not False:
        return False, "HNeRV low-level repack must not unlock exact-eval dispatch"
    if payload.get("ready_for_archive_preflight") is not True:
        return False, json_text(payload)
    if audit.get("ready_for_archive_preflight") is not True:
        return False, json_text(payload)
    if not raw or any(row.get("raw_equal") is not True for row in raw if isinstance(row, dict)):
        return False, "HNeRV low-level repack missing brotli raw equality proof"
    delta = int(audit.get("total_byte_delta", 0))
    if delta >= 0:
        return False, "HNeRV low-level repack did not produce a rate-positive byte delta"
    return True, (
        "HNeRV low-level repack: PASS "
        f"(archive-preflight candidate only; total_byte_delta={delta}; "
        f"candidate_bytes={payload.get('candidate_archive_bytes')}; "
        "ready_for_exact_eval_dispatch=false)"
    )


def _run_untracked_source_gate() -> tuple[bool, str]:
    cmd = [sys.executable, str(UNTRACKED_SOURCE_AUDIT), "--format", "json"]
    if UNTRACKED_SOURCE_DISPOSITION_MANIFEST.exists():
        cmd.extend(
            [
                "--strict",
                "--disposition-manifest",
                str(UNTRACKED_SOURCE_DISPOSITION_MANIFEST),
            ]
        )
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return False, output
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return False, f"untracked-source audit emitted invalid JSON: {exc}\n{output}"
    summary = payload.get("summary", {})
    count = int(summary.get("untracked_source_like_count", 0))
    undispositioned = int(summary.get("undispositioned_count", 0))
    invalid = int(summary.get("invalid_disposition_count", 0))
    shadowed = int(summary.get("shadowed_index_delete_count", 0))
    resolved_tracked = int(summary.get("resolved_tracked_disposition_count", 0))
    by_class = summary.get("by_class", {})
    by_disposition = summary.get("by_disposition", {})
    manifest = summary.get("disposition_manifest")
    if manifest:
        return True, (
            "untracked-source audit: PASS "
            f"({count} source-like untracked; undispositioned={undispositioned}; "
            f"invalid_dispositions={invalid}; shadowed_index_deletes={shadowed}; "
            f"resolved_tracked_dispositions={resolved_tracked}; "
            f"by_disposition={by_disposition})"
        )
    return True, f"untracked-source audit: ADVISORY ({count} source-like untracked; by_class={by_class})"


def _run_reverse_engineering_release_gate() -> tuple[bool, str]:
    if not REVERSE_ENGINEERING_RELEASE_MANIFEST.is_file():
        return False, f"missing release manifest: {REVERSE_ENGINEERING_RELEASE_MANIFEST.relative_to(REPO)}"
    proc = subprocess.run(
        [
            sys.executable,
            str(REVERSE_ENGINEERING_AUDIT),
            "--repo-root",
            str(REPO),
            "--release-strict",
            "--release-manifest",
            str(REVERSE_ENGINEERING_RELEASE_MANIFEST),
            "--summary",
        ],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def _run_hnerv_scorecard_gate() -> tuple[bool, str]:
    proc = subprocess.run([sys.executable, str(HNERV_SCORECARD_AUDIT)], capture_output=True, text=True)
    return proc.returncode == 0, proc.stdout + proc.stderr


def _run_tooling_consolidation_gate() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(TOOLING_CONSOLIDATION_AUDIT)],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def _run_recovered_remote_lanes_gate() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(RECOVERED_REMOTE_LANES_AUDIT), "--strict"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def _run_preserved_orphans_gate() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(PRESERVED_ORPHANS_AUDIT), "--format", "text"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def _run_recovery_custody_snapshots_gate() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(RECOVERY_CUSTODY_SNAPSHOTS_AUDIT), "--format", "text"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def _run_release_index_split_gate() -> tuple[bool, str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(RELEASE_INDEX_SPLIT_AUDIT),
            "--repo-root",
            str(REPO),
            "--strict",
            "--local-custody-manifest",
            str(LOCAL_CUSTODY_RELEASE_MANIFEST),
        ],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def _run_nested_gitlink_custody_gate() -> tuple[bool, str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(NESTED_GITLINK_CUSTODY_AUDIT),
            "--repo-root",
            str(REPO),
            "--strict",
            "--local-custody-manifest",
            str(LOCAL_CUSTODY_RELEASE_MANIFEST),
        ],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def _run_staged_public_release_hygiene_gate() -> tuple[bool, str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(STAGED_PUBLIC_RELEASE_HYGIENE_AUDIT),
            "--repo-root",
            str(REPO),
            "--strict",
        ],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def _execute_step(step: PreflightStep) -> PreflightResult:
    start = time.perf_counter()
    try:
        passed, output = step.runner()
    except Exception as exc:  # pragma: no cover - defensive fail-closed wrapper.
        passed = False
        output = f"{step.section} #{step.number} raised {type(exc).__name__}: {exc}"
    return PreflightResult(
        step=step,
        passed=passed,
        output=output,
        elapsed_s=time.perf_counter() - start,
    )


def _default_jobs(step_count: int) -> int:
    cpu_count = os.cpu_count() or 2
    return max(1, min(step_count, cpu_count, 8))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Forward --verbose to each dry-run (shows PASS lines).")
    parser.add_argument(
        "--require-real-omega-sensitivity",
        action="store_true",
        help=(
            "Promote Lane Omega-W-V3 from stub-byte smoke to strict readiness: "
            "require a CUDA sensitivity map with source archive SHA metadata."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help=(
            "Maximum independent preflight steps to run concurrently. "
            "Default: min(8, CPU count, step count). Use --jobs 1 for serial output."
        ),
    )
    parser.add_argument(
        "--timings",
        action="store_true",
        help="Append per-step wall-clock timings to the summary.",
    )
    args = parser.parse_args(argv)

    lanes = [dict(lane) for lane in LANES]
    if args.require_real_omega_sensitivity:
        for lane in lanes:
            if Path(lane["tool"]).name == "dispatch_dryrun_omega_w_v3.py":
                lane["args"] = [*lane["args"], "--require-real-sensitivity"]
                lane["local_smoke_only"] = False

    # Verify all sub-tools exist before running any
    for tool in [
        SHELL_HAZARDS,
        REVERSE_ENGINEERING_AUDIT,
        HIDDEN_GEMS_REGISTRY,
        HIDDEN_GEMS_READINESS,
        ENGINEERED_CORRECTIONS_AUDIT,
        HNERV_SCORECARD_AUDIT,
        HNERV_LOWLEVEL_REPACK,
        TOOLING_CONSOLIDATION_AUDIT,
        RECOVERED_REMOTE_LANES_AUDIT,
        UNTRACKED_SOURCE_AUDIT,
        ORPHAN_RECOVERY_AUDIT,
        PRESERVED_ORPHANS_AUDIT,
        RECOVERY_CUSTODY_SNAPSHOTS_AUDIT,
        RELEASE_INDEX_SPLIT_AUDIT,
        NESTED_GITLINK_CUSTODY_AUDIT,
        STAGED_PUBLIC_RELEASE_HYGIENE_AUDIT,
        LOCAL_CUSTODY_RELEASE_MANIFEST,
        *[lane["tool"] for lane in lanes],
    ]:
        if not tool.is_file():
            print(f"FATAL: missing sub-tool {tool.relative_to(REPO)}", file=sys.stderr)
            return 2

    gate_steps = [
        PreflightStep(
            "GATE",
            0,
            "dispatch CLI/shell hazards",
            lambda: _run_gate("dispatch CLI/shell hazards", SHELL_HAZARDS),
            "  ✓ Gate #0: dispatch CLI/shell hazards — PASSED",
            "  ✗ Gate #0: dispatch CLI/shell hazards — FAILED",
        ),
        PreflightStep(
            "GATE",
            1,
            "reverse-engineering tree curation",
            lambda: _run_gate("reverse-engineering tree curation", REVERSE_ENGINEERING_AUDIT, ["--summary"]),
            "  ✓ Gate #1: reverse-engineering tree curation — PASSED",
            "  ✗ Gate #1: reverse-engineering tree curation — FAILED",
        ),
        PreflightStep(
            "GATE",
            2,
            "hidden-gem registry",
            _run_hidden_gems_gate,
            "  ✓ Gate #2: hidden-gem registry — PASSED",
            "  ✗ Gate #2: hidden-gem registry — FAILED",
        ),
        PreflightStep(
            "GATE",
            3,
            "hidden-gem readiness",
            _run_hidden_gem_readiness_gate,
            "  ✓ Gate #3: hidden-gem readiness — PASSED",
            "  ✗ Gate #3: hidden-gem readiness — FAILED",
        ),
        PreflightStep(
            "GATE",
            4,
            "engineered-correction readiness",
            _run_engineered_corrections_gate,
            "  ✓ Gate #4: engineered-correction readiness — PASSED",
            "  ✗ Gate #4: engineered-correction readiness — FAILED",
        ),
        PreflightStep(
            "GATE",
            5,
            "HNeRV frontier scorecard",
            _run_hnerv_scorecard_gate,
            "  ✓ Gate #5: HNeRV frontier scorecard — PASSED",
            "  ✗ Gate #5: HNeRV frontier scorecard — FAILED",
        ),
        PreflightStep(
            "GATE",
            6,
            "HNeRV low-level repack proof",
            _run_hnerv_lowlevel_repack_gate,
            "  ✓ Gate #6: HNeRV low-level repack proof — PASSED",
            "  ✗ Gate #6: HNeRV low-level repack proof — FAILED",
        ),
        PreflightStep(
            "GATE",
            7,
            "tooling consolidation inventory",
            _run_tooling_consolidation_gate,
            "  ✓ Gate #7: tooling consolidation inventory — PASSED",
            "  ✗ Gate #7: tooling consolidation inventory — FAILED",
        ),
        PreflightStep(
            "GATE",
            8,
            "recovered remote lane canonicalization",
            _run_recovered_remote_lanes_gate,
            "  ✓ Gate #8: recovered remote lane canonicalization — PASSED",
            "  ✗ Gate #8: recovered remote lane canonicalization — FAILED",
        ),
        PreflightStep(
            "GATE",
            9,
            "untracked source inventory",
            _run_untracked_source_gate,
            "  ✓ Gate #9: untracked source inventory — PASSED (STRICT DISPOSITION)"
            if UNTRACKED_SOURCE_DISPOSITION_MANIFEST.exists()
            else "  ✓ Gate #9: untracked source inventory — PASSED (ADVISORY)",
            "  ✗ Gate #9: untracked source inventory — FAILED",
        ),
        PreflightStep(
            "GATE",
            10,
            "orphan recovery canonicalization",
            lambda: _run_gate("orphan recovery canonicalization", ORPHAN_RECOVERY_AUDIT),
            "  ✓ Gate #10: orphan recovery canonicalization — PASSED",
            "  ✗ Gate #10: orphan recovery canonicalization — FAILED",
        ),
        PreflightStep(
            "GATE",
            11,
            "preserved-orphan canonicalization",
            _run_preserved_orphans_gate,
            "  ✓ Gate #11: preserved-orphan canonicalization — PASSED",
            "  ✗ Gate #11: preserved-orphan canonicalization — FAILED",
        ),
        PreflightStep(
            "GATE",
            12,
            "recovery custody snapshots",
            _run_recovery_custody_snapshots_gate,
            "  ✓ Gate #12: recovery custody snapshots — PASSED",
            "  ✗ Gate #12: recovery custody snapshots — FAILED",
        ),
        PreflightStep(
            "GATE",
            13,
            "reverse-engineering release manifest",
            _run_reverse_engineering_release_gate,
            "  ✓ Gate #13: reverse-engineering release manifest — PASSED",
            "  ✗ Gate #13: reverse-engineering release manifest — FAILED",
        ),
        PreflightStep(
            "GATE",
            14,
            "release index/worktree split",
            _run_release_index_split_gate,
            "  ✓ Gate #14: release index/worktree split — PASSED",
            "  ✗ Gate #14: release index/worktree split — FAILED",
        ),
        PreflightStep(
            "GATE",
            15,
            "nested gitlink custody",
            _run_nested_gitlink_custody_gate,
            "  ✓ Gate #15: nested gitlink custody — PASSED",
            "  ✗ Gate #15: nested gitlink custody — FAILED",
        ),
        PreflightStep(
            "GATE",
            16,
            "staged public release hygiene",
            _run_staged_public_release_hygiene_gate,
            "  ✓ Gate #16: staged public release hygiene — PASSED",
            "  ✗ Gate #16: staged public release hygiene — FAILED",
        ),
    ]
    lane_steps = [
        PreflightStep(
            "LANE",
            i,
            str(lane["name"]),
            lambda lane=lane: _run_lane(lane, verbose=args.verbose),
            (
                f"  ✓ Lane #{i}: {lane['name']} — SELF-PROTECTED, NOT DISPATCH-READY"
                if lane.get("forensic_only")
                else (
                    f"  ✓ Lane #{i}: {lane['name']} — LOCAL-SMOKE ONLY, NOT DISPATCH-READY"
                    if lane.get("local_smoke_only")
                    else f"  ✓ Lane #{i}: {lane['name']} — PASSED"
                )
            ),
            f"  ✗ Lane #{i}: {lane['name']} — FAILED",
            forensic_only=bool(lane.get("forensic_only")),
            local_smoke_only=bool(lane.get("local_smoke_only")),
        )
        for i, lane in enumerate(lanes, start=1)
    ]
    steps = gate_steps + lane_steps
    if args.jobs is not None and args.jobs < 1:
        print("FATAL: --jobs must be >= 1", file=sys.stderr)
        return 2
    max_workers = args.jobs or _default_jobs(len(steps))
    if max_workers == 1:
        results = [_execute_step(step) for step in steps]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_execute_step, step) for step in steps]
            results = [future.result() for future in futures]

    n_passed = 0
    n_failed = 0
    n_forensic_only = 0
    n_local_smoke_only = 0
    summary_lines: list[str] = []
    bar = "═" * 70
    for result in results:
        step = result.step
        print(f"\n{bar}\n{step.section} #{step.number}: {step.name}\n{bar}")
        print(result.output.rstrip())
        if result.passed:
            n_passed += 1
            n_forensic_only += int(step.forensic_only)
            n_local_smoke_only += int(step.local_smoke_only)
            summary_lines.append(step.pass_summary)
        else:
            n_failed += 1
            summary_lines.append(step.fail_summary)

    bar = "═" * 70
    print(f"\n{bar}\nALL-LANES PREFLIGHT SUMMARY\n{bar}\n")
    for line in summary_lines:
        print(line)
    if args.timings:
        print("\nTimings:")
        for result in sorted(results, key=lambda item: item.elapsed_s, reverse=True):
            step = result.step
            print(f"  {result.elapsed_s:6.2f}s  {step.section} #{step.number}: {step.name}")
    print()
    if n_failed == 0:
        print(f"ALL {n_passed} PREFLIGHT CHECKS PASSED.")
        if n_forensic_only:
            print(f"{n_forensic_only} lane(s) are explicitly forensic-only and not exact-eval dispatch-ready.")
        if n_local_smoke_only:
            print(f"{n_local_smoke_only} lane(s) passed local smoke only and require stricter readiness before dispatch.")
        print(
            "Only lane-local tools that print ready_for_remote_cuda_dispatch=true "
            "or an exact dispatch GO may be sent to GPU."
        )
        return 0
    print(f"{n_failed} of {n_passed + n_failed} LANES FAILED — DO NOT DISPATCH the failed lanes.")
    print("See per-lane output above for specific check failures.")
    return n_failed


if __name__ == "__main__":
    raise SystemExit(main())
