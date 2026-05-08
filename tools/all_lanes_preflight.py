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
  Gate #3: tools/audit_semantic_label_contract.py --fail-on-advisory
           (categorical/CLADE/SPADE/openpilot label order stays canonical)
  Gate #4: tools/audit_hidden_gem_readiness.py --status ready_for_patch
           (ready-for-patch hidden gems point at live evidence and targets)
  Gate #5: tools/audit_engineered_corrections.py --self-test
           (engineered-correction payload guard imports and fails closed before dispatch)
  Gate #6: tools/audit_hnerv_frontier_scorecard.py
           (public HNeRV scorecard is fresh enough to route hidden-gem follow-ups)
  Gate #7: tools/build_hnerv_lowlevel_repack_candidate.py
           (real PR106x byte-candidate proof with raw brotli equality; still
            not exact-eval dispatch-ready)
  Gate #8: tools/audit_tooling_consolidation.py
           (advisory inventory of duplicated audit/preflight helper patterns)
  Gate #9: tools/audit_recovered_remote_lanes.py
           (recovered lane scripts have canonical custody/proxy classification)
  Gate #10: tools/audit_untracked_source_artifacts.py
           (no-signal-loss inventory of untracked source/research; strict when
            the reviewed disposition manifest exists)
  Gate #11: tools/audit_orphan_recovery_canonicalization.py
           (orphan pyc recovery deletions must have tracked canonical copies)
  Gate #12: tools/audit_preserved_orphans.py
           (local preserved-orphan shadows must be duplicate, superseded, or absent)
  Gate #13: tools/audit_recovery_custody_snapshots.py
           (unique pyc/signal-loss custody snapshots are intact and explicit)
  Gate #14: tools/audit_reverse_engineering_tree.py --release-strict
           --release-manifest ...
           (public-release reverse-engineering custody is explicitly curated)
  Gate #15: tools/audit_release_index_split.py --strict
           (no staged rollback shadows; no staged private provider/runtime
            state in release commits)
  Gate #16: tools/audit_nested_gitlink_custody.py --strict
           (dirty public-intake/raw-custody gitlinks must be documented and
            their inner dirty status must be visible)
  Gate #17: tools/audit_staged_public_release_hygiene.py --strict
           (staged docs/site/readme public surfaces contain no private paths,
            provider job links, or credential-like strings)
  Gate #18: tools/build_cross_paradigm_frontier_inventory.py --format json
           (stacking/replacement inventory is current, deterministic, and
            remains inventory-only; geometry feedback stays blocked without
            charged runtime consumers)
  Gate #19: tools/audit_pr91_hpm1_readiness.py +
           tools/audit_pr91_hpm1_runtime_contract.py
           (PR91/HPM1 high-EV categorical path stays byte-custody-clean,
            operator-visible, and fail-closed until HPAC parity/runtime gates)
  Gate #20: tools/pr106_archive_decomposition.py
           (PR101/PR106 frontier archives are treated as single-member
            monolithic packets; logical budgets require internal parser proof)
  Gate #21: tools/check_omega_opt_anchor_discipline.py --strict
           (Omega-OPT nested score hypotheses stay fail-closed until 1:1
            archive/eval anchors exist; scans the generated HStack/VStack
            plan artifact too)
  Gate #22: tools/probe_eval_loader_drift.py
           (CPU/CUDA eval drift mechanism probe. Writes a non-promotable
            DALI-vs-PyAV plan locally, and a real decoded-RGB comparison when
            CUDA+DALI is available. It never promotes/ranks/kills.)
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
import math
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

from tac.geometry_feedback_readiness import (  # noqa: E402
    GEOMETRY_FEEDBACK_ROADMAP_KEYS,
    geometry_feedback_contract_failures,
)
from tac.repo_io import json_text, sha256_bytes  # noqa: E402

TOOLS = REPO / "tools"
SHELL_HAZARDS = TOOLS / "check_dispatch_cli_shell_hazards.py"
REVERSE_ENGINEERING_AUDIT = TOOLS / "audit_reverse_engineering_tree.py"
HIDDEN_GEMS_REGISTRY = TOOLS / "list_hidden_gems.py"
HIDDEN_GEMS_READINESS = TOOLS / "audit_hidden_gem_readiness.py"
SEMANTIC_LABEL_AUDIT = TOOLS / "audit_semantic_label_contract.py"
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
CROSS_PARADIGM_FRONTIER_INVENTORY = TOOLS / "build_cross_paradigm_frontier_inventory.py"
PR91_HPM1_READINESS_AUDIT = TOOLS / "audit_pr91_hpm1_readiness.py"
PR91_HPM1_RUNTIME_CONTRACT_AUDIT = TOOLS / "audit_pr91_hpm1_runtime_contract.py"
FRONTIER_ARCHIVE_LAYOUT_AUDIT = TOOLS / "pr106_archive_decomposition.py"
OMEGA_OPT_ANCHOR_AUDIT = TOOLS / "check_omega_opt_anchor_discipline.py"
EVAL_LOADER_DRIFT_PROBE = TOOLS / "probe_eval_loader_drift.py"
EVAL_LOADER_DRIFT_MISSING_PREREQ_CLASS = "missing_prerequisite"
EVAL_LOADER_DRIFT_PROBE_RUNTIME_ERROR_CLASS = "probe_runtime_error"
EVAL_LOADER_DRIFT_KNOWN_MISSING_PREREQ_CODES = frozenset(
    {
        "upstream_frame_utils_py",
        "pyav_available",
        "cuda_available",
        "dali_available",
        "video_names_file_exists",
        "data_dir_exists",
        "sample_videos_exist",
        "cuda_dali_runtime_available",
    }
)
EVAL_LOADER_DRIFT_REQUIRED_COMPARISON_METRICS = (
    "max_abs_lsb",
    "mean_abs_lsb",
    "rms_abs_lsb",
    "nonzero_fraction",
)
HSTACK_VSTACK_PLAN = REPO / "reports/hstack_vstack_multipass_plan_20260507.json"
PR91_HPM1_READINESS_ARTIFACT = REPO / "experiments/results/pr91_hpm1_readiness_20260506_codex/readiness.json"
PR91_HPM1_RUNTIME_CONTRACT_ARTIFACT = (
    REPO / "experiments/results/pr91_hpm1_runtime_contract_20260506_codex/runtime_contract.json"
)
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
TIMING_PROFILE_SCHEMA = "pact.all_lanes_preflight_timing.v1"
SLOW_STEP_THRESHOLD_S = 0.50

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
        "supports_verbose": False,
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
    if verbose and lane.get("supports_verbose", True):
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


def _run_semantic_label_contract_gate() -> tuple[bool, str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(SEMANTIC_LABEL_AUDIT),
            "--format",
            "json",
            "--fail-on-advisory",
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
        return False, f"semantic-label audit emitted invalid JSON: {exc}\n{output}"
    if payload.get("ok") is not True:
        return False, output
    blocking = len(payload.get("blocking_findings", []))
    advisory = len(payload.get("advisory_findings", []))
    if blocking or advisory:
        return False, (
            "semantic-label audit must have zero blocking/advisory findings; "
            f"blocking={blocking} advisory={advisory}\n{output}"
        )
    return True, "semantic-label contract: PASS (canonical class order; 0 stale findings)"


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


def _run_cross_paradigm_frontier_inventory_gate() -> tuple[bool, str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(CROSS_PARADIGM_FRONTIER_INVENTORY),
            "--repo-root",
            str(REPO),
            "--format",
            "json",
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
        return False, f"cross-paradigm inventory emitted invalid JSON: {exc}\n{output}"
    if payload.get("score_claim") is not False:
        return False, "cross-paradigm inventory must keep score_claim=false"
    if payload.get("dispatch_attempted") is not False:
        return False, "cross-paradigm inventory must keep dispatch_attempted=false"
    if payload.get("ready_for_exact_eval_dispatch") is not False:
        return False, "cross-paradigm inventory must not unlock exact-eval dispatch"
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return False, "cross-paradigm inventory emitted no rows"
    row_keys = {str(row.get("key")) for row in rows if isinstance(row, dict)}
    required_keys = {
        "hnerv_pr103_pr106_ac_repack_runtime_closure",
        "hnerv_wavelet_wr01_apply",
        "categorical_qma9_clade_spade_openpilot",
        "lapose_motion_atom_allocator",
        "meta_lagrangian_cross_paradigm_allocator",
        "telescopic_foveation_field",
    }
    missing_keys = sorted(required_keys - row_keys)
    if missing_keys:
        return False, "cross-paradigm inventory missing required row(s): " + ", ".join(missing_keys)
    anchor_failures = _cross_paradigm_pr103_anchor_failures(rows)
    if anchor_failures:
        return False, "cross-paradigm inventory PR103 anchor contract failed:\n" + "\n".join(
            anchor_failures
        )
    geometry_failures = _geometry_feedback_inventory_failures(payload)
    if geometry_failures:
        return False, "cross-paradigm inventory geometry feedback contract failed:\n" + "\n".join(
            geometry_failures
        )
    missing_code = int(payload.get("missing_code_path_count", -1))
    missing_evidence = int(payload.get("missing_evidence_path_count", -1))
    if missing_code != 0 or missing_evidence != 0:
        return False, (
            "cross-paradigm inventory has missing paths: "
            f"code={missing_code} evidence={missing_evidence}"
        )
    action_queue = payload.get("frontier_action_queue")
    if not isinstance(action_queue, list) or len(action_queue) != len(rows):
        return False, "cross-paradigm inventory action queue must cover every row"
    routing_failures = _cross_paradigm_queue_routing_failures(action_queue)
    if routing_failures:
        return False, "cross-paradigm inventory action queue routing failed:\n" + "\n".join(
            routing_failures
        )
    for item in action_queue:
        if not isinstance(item, dict):
            return False, "cross-paradigm inventory action queue contains a non-object row"
        if item.get("score_claim") is not False:
            return False, "cross-paradigm inventory action queue must keep score_claim=false"
        if item.get("ready_for_exact_eval_dispatch") is not False:
            return False, "cross-paradigm inventory action queue must keep dispatch readiness false"
    return True, (
        "cross-paradigm inventory: PASS "
        f"({len(rows)} rows; 0 missing code/evidence paths; "
        "geometry feedback fail-closed; action queue inventory-only)"
    )


def _frontier_monolithic_layout_failures(payload: dict[str, object]) -> list[str]:
    """Validate PR101/PR106 frontier layout is not budgeted by ZIP members."""
    failures: list[str] = []
    if payload.get("score_claim") is not False:
        failures.append("layout_payload_score_claim_must_be_false")
    if payload.get("evidence_grade") != "empirical_archive_layout_cpu_no_score":
        failures.append("layout_payload_evidence_grade_must_be_cpu_archive_only")
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) < 2:
        return [*failures, "layout_payload_missing_pr101_pr106_runs"]
    expected_member_names = {"x", "0.bin"}
    observed_member_names: set[str] = set()
    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            failures.append(f"layout_run_{idx}_not_object")
            continue
        archive_path = str(run.get("archive_path") or "")
        if run.get("score_claim") is not False:
            failures.append(f"layout_run_{idx}_score_claim_must_be_false")
        physical = run.get("physical_layout")
        if not isinstance(physical, dict):
            failures.append(f"layout_run_{idx}_physical_layout_missing")
            continue
        if physical.get("single_member_monolithic_packet") is not True:
            failures.append(f"layout_run_{idx}_not_single_member_monolithic_packet")
        if physical.get("archive_member_level_component_budgets_valid") is not False:
            failures.append(f"layout_run_{idx}_member_level_component_budgets_not_rejected")
        if physical.get("member_level_mask_budget_valid") is not False:
            failures.append(f"layout_run_{idx}_member_level_mask_budget_not_rejected")
        if physical.get("member_level_pose_budget_valid") is not False:
            failures.append(f"layout_run_{idx}_member_level_pose_budget_not_rejected")
        members = physical.get("members")
        if not isinstance(members, list) or len(members) != 1 or not isinstance(members[0], dict):
            failures.append(f"layout_run_{idx}_single_member_record_missing")
            continue
        member_name = str(members[0].get("name"))
        observed_member_names.add(member_name)
        logical = run.get("logical_layout")
        if not isinstance(logical, dict):
            failures.append(f"layout_run_{idx}_logical_layout_missing")
            continue
        grammar = str(logical.get("grammar") or "")
        if grammar == "pr101_fixed_offset_hnerv_microcodec" and member_name != "x":
            failures.append(f"layout_run_{idx}_pr101_member_name_must_be_x")
        if grammar == "pr106_ff_packed_hnerv" and member_name != "0.bin":
            failures.append(f"layout_run_{idx}_pr106_member_name_must_be_0.bin")
        if "public_pr101" in archive_path and grammar != "pr101_fixed_offset_hnerv_microcodec":
            failures.append(f"layout_run_{idx}_public_pr101_grammar_mismatch")
        if "public_pr106" in archive_path and grammar != "pr106_ff_packed_hnerv":
            failures.append(f"layout_run_{idx}_public_pr106_grammar_mismatch")
        sections = logical.get("sections")
        if not isinstance(sections, list) or not sections:
            failures.append(f"layout_run_{idx}_logical_sections_missing")
    missing_members = sorted(expected_member_names - observed_member_names)
    if missing_members:
        failures.append("layout_expected_member_name_missing: " + ", ".join(missing_members))
    return failures


def _run_frontier_monolithic_layout_gate() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="pact_frontier_layout_preflight_") as tmp:
        output_path = Path(tmp) / "frontier_layout.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(FRONTIER_ARCHIVE_LAYOUT_AUDIT),
                "--output-json",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        output = proc.stdout + proc.stderr
        if proc.returncode != 0:
            return False, output
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"frontier layout audit emitted invalid JSON: {exc}\n{output}"
    failures = _frontier_monolithic_layout_failures(payload)
    if failures:
        return False, "frontier monolithic layout contract failed:\n" + "\n".join(failures)
    return True, (
        "frontier monolithic layout: PASS "
        "(PR101/PR106 are single-member packets; member-level mask/pose budgets rejected; "
        "logical sections parser-proven)"
    )


def _eval_loader_drift_missing_prereq_pass(payload: dict[str, object]) -> tuple[bool, str]:
    unavailable_class = payload.get("comparison_unavailable_class")
    reasons_raw = payload.get("comparison_unavailable_reasons")
    if isinstance(reasons_raw, list):
        reasons = [str(reason) for reason in reasons_raw]
    else:
        reason = payload.get("comparison_unavailable_reason")
        reasons = [str(reason)] if reason else []

    if unavailable_class != EVAL_LOADER_DRIFT_MISSING_PREREQ_CLASS:
        reason_text = "; ".join(reasons) if reasons else "no unavailable reason emitted"
        return False, (
            "eval loader drift probe unavailable for non-prerequisite reason "
            f"({unavailable_class or 'unknown'}): {reason_text}"
        )

    codes_raw = payload.get("comparison_unavailable_codes")
    if not isinstance(codes_raw, list) or not codes_raw:
        return False, "eval loader drift probe missing typed unavailable codes"
    codes = [str(code) for code in codes_raw]
    unknown_codes = sorted(set(codes) - EVAL_LOADER_DRIFT_KNOWN_MISSING_PREREQ_CODES)
    if unknown_codes:
        return False, (
            "eval loader drift probe emitted unknown missing-prereq code(s): "
            + ", ".join(unknown_codes)
        )

    return True, (
        "eval loader drift probe: PASS "
        "(known missing prerequisite(s): "
        + ", ".join(codes)
        + "; non-promotable plan artifact validated)"
    )


def _validate_eval_loader_drift_comparison_rows(rows: object) -> list[str]:
    if not isinstance(rows, list) or not rows:
        return ["comparison was available but emitted no rows"]
    failures: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            failures.append(f"row {index}: expected object, got {type(row).__name__}")
            continue
        if row.get("path_match") is not True:
            failures.append(f"row {index}: video path mismatch")
        if row.get("sequence_index_match") is not True:
            failures.append(f"row {index}: sequence index mismatch")
        comparison = row.get("comparison")
        if not isinstance(comparison, dict):
            failures.append(f"row {index}: comparison object missing")
            continue
        if comparison.get("shape_match") is not True:
            failures.append(f"row {index}: comparison shape mismatch")
        numel = comparison.get("numel")
        if not isinstance(numel, int) or isinstance(numel, bool) or numel <= 0:
            failures.append(f"row {index}: comparison numel must be a positive integer")
        for metric in EVAL_LOADER_DRIFT_REQUIRED_COMPARISON_METRICS:
            value = comparison.get(metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                failures.append(f"row {index}: comparison {metric} must be numeric")
                continue
            if not math.isfinite(float(value)):
                failures.append(f"row {index}: comparison {metric} must be finite")
    return failures


def _run_eval_loader_drift_probe_gate() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="pact_eval_loader_drift_preflight_") as tmp:
        output_path = Path(tmp) / "eval_loader_drift_probe.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(EVAL_LOADER_DRIFT_PROBE),
                "--video-limit",
                "1",
                "--max-batches",
                "1",
                "--json-out",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        output = proc.stdout + proc.stderr
        # Exit 2 means a typed missing prerequisite. Exit 3 means the probe hit
        # a runtime error after prerequisites were available; parse its JSON so
        # all-lanes can fail with the typed reason instead of treating every
        # unavailable comparison as an environment skip.
        if proc.returncode not in {0, 2, 3}:
            return False, output
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"eval loader drift probe emitted invalid JSON: {exc}\n{output}"
    if payload.get("score_claim") is not False:
        return False, "eval loader drift probe must never emit score_claim=true"
    if payload.get("promotion_eligible") is not False:
        return False, "eval loader drift probe must never be promotion eligible"
    if payload.get("rank_or_kill_eligible") is not False:
        return False, "eval loader drift probe must never be rank/kill eligible"
    if payload.get("comparison_available") is False:
        return _eval_loader_drift_missing_prereq_pass(payload)
    failures = _validate_eval_loader_drift_comparison_rows(
        payload.get("comparison_rows")
    )
    if failures:
        return False, "eval loader drift probe comparison schema invalid: " + "; ".join(failures)
    return True, (
        "eval loader drift probe: PASS "
        "(DALI-vs-PyAV decoded-RGB comparison emitted; diagnostic only)"
    )


def _cross_paradigm_queue_routing_failures(action_queue: list[object]) -> list[str]:
    """Validate post-anchor first-tranche routing."""
    failures: list[str] = []
    if not action_queue:
        return ["action_queue_empty"]
    keys: list[str] = []
    for item in action_queue:
        if isinstance(item, dict):
            keys.append(str(item.get("key") or ""))
        else:
            keys.append("")
    required_first_tranche = [
        "categorical_qma9_clade_spade_openpilot",
        "joint_admm_balle_arithmetic_stack",
        "hnerv_per_tensor_context_entropy",
        "telescopic_foveation_field",
        "lapose_motion_atom_allocator",
    ]
    first_tranche = keys[: len(required_first_tranche)]
    missing = sorted(set(required_first_tranche) - set(first_tranche))
    if missing:
        failures.append(
            "first_tranche_missing_required_score_path_row(s): " + ", ".join(missing)
        )
    elif first_tranche != required_first_tranche:
        failures.append(
            "first_tranche_order_mismatch: expected "
            + ", ".join(required_first_tranche)
            + "; observed "
            + ", ".join(first_tranche)
        )
    return failures


def _cross_paradigm_pr103_anchor_failures(rows: list[object]) -> list[str]:
    by_key = {
        str(row.get("key")): row
        for row in rows
        if isinstance(row, dict) and row.get("key") is not None
    }
    anchor = by_key.get("hnerv_pr103_pr106_ac_repack_runtime_closure")
    if anchor is None:
        return ["hnerv_pr103_pr106_ac_repack_runtime_closure: missing anchor row"]
    failures: list[str] = []
    expected = {
        "status": "exact_cuda_a++_anchor_promoted",
        "action_class": "maintain_exact_eval_anchor_and_pivot",
        "role": "current_exact_rate_anchor",
        "priority_tier": 900,
    }
    for field, value in expected.items():
        if anchor.get(field) != value:
            failures.append(f"anchor_{field}_drift: expected {value!r}, got {anchor.get(field)!r}")
    if anchor.get("score_claim") is not False:
        failures.append("anchor_score_claim_must_remain_false")
    if anchor.get("ready_for_exact_eval_dispatch") is not False:
        failures.append("anchor_ready_for_exact_eval_dispatch_must_remain_false")
    snapshot = anchor.get("score_snapshot")
    if not isinstance(snapshot, dict):
        failures.append("anchor_score_snapshot_missing")
        return failures
    if snapshot.get("compliance_passed") is not True:
        failures.append("anchor_contest_final_compliance_not_passed")
    failed_checks = snapshot.get("compliance_failed_checks")
    if failed_checks:
        failures.append(
            "anchor_contest_final_failed_checks: "
            + ", ".join(str(item) for item in failed_checks)
        )
    if int(snapshot.get("compliance_check_count", 0)) <= 0:
        failures.append("anchor_contest_final_checks_missing")
    if snapshot.get("score") != 0.2089810755823297:
        failures.append(f"anchor_score_drift: {snapshot.get('score')!r}")
    if snapshot.get("report_reconstructed_score") != 0.20898105277982337:
        failures.append(
            "anchor_report_reconstructed_score_drift: "
            f"{snapshot.get('report_reconstructed_score')!r}"
        )
    if snapshot.get("score_basis") != "auth_eval_report_components_plus_exact_archive_bytes":
        failures.append(f"anchor_score_basis_drift: {snapshot.get('score_basis')!r}")
    if snapshot.get("anchor_proof_schema") != "pre_submission_compliance_anchor_proof_v1":
        failures.append(
            f"anchor_self_contained_proof_missing: {snapshot.get('anchor_proof_schema')!r}"
        )
    if snapshot.get("archive_bytes") != 185578:
        failures.append(f"anchor_archive_bytes_drift: {snapshot.get('archive_bytes')!r}")
    if (
        snapshot.get("archive_sha256")
        != "ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce"
    ):
        failures.append(f"anchor_archive_sha256_drift: {snapshot.get('archive_sha256')!r}")
    runtime_expected = {
        "runtime_tree_sha256": "54db9e5ddee85ae7f486fae900ff3907932efb1c8d3062bc264b0e5c7456d8f6",
        "device": "cuda",
        "samples": 600,
        "gpu_model": "Tesla T4",
        "upstream_evaluate_py_sha256": "7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b",
    }
    for field, value in runtime_expected.items():
        if snapshot.get(field) != value:
            failures.append(
                f"anchor_{field}_drift: expected {value!r}, got {snapshot.get(field)!r}"
            )
    score_path = str(snapshot.get("path") or "")
    if not score_path.endswith("pre_submission_compliance.contest_final.json"):
        failures.append(f"anchor_score_snapshot_path_not_tracked_compliance_json: {score_path!r}")
    return failures


def _geometry_feedback_inventory_failures(payload: dict[str, object]) -> list[str]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return ["rows: missing or non-list"]
    by_key = {
        str(row.get("key")): row
        for row in rows
        if isinstance(row, dict) and row.get("key") is not None
    }
    failures: list[str] = []
    for key in GEOMETRY_FEEDBACK_ROADMAP_KEYS:
        row = by_key.get(key)
        if row is None:
            failures.append(f"{key}: missing geometry feedback row")
            continue
        if row.get("ready_for_exact_eval_dispatch") is not False:
            failures.append(f"{key}: row ready_for_exact_eval_dispatch must be false")
        contract_failures = geometry_feedback_contract_failures(
            row.get("geometry_feedback_contract")
        )
        for failure in contract_failures:
            failures.append(f"{key}: {failure}")
    return failures


def _json_tool(tool: Path) -> tuple[bool, dict[str, object], str]:
    proc = subprocess.run([sys.executable, str(tool)], capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return False, {}, output
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return False, {}, f"{tool.name} emitted invalid JSON: {exc}\n{output}"
    return True, payload, output


def _load_json_artifact(path: Path) -> tuple[bool, dict[str, object], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, {}, f"{path.relative_to(REPO)} is not readable JSON: {exc}"
    if not isinstance(payload, dict):
        return False, {}, f"{path.relative_to(REPO)} must contain a JSON object"
    return True, payload, ""


def _embedded_canonical_payload_hash(payload: dict[str, object]) -> str:
    manifest = payload.get("tool_run_manifest")
    if not isinstance(manifest, dict):
        return ""
    value = manifest.get("canonical_payload_without_tool_manifest_sha256")
    return str(value) if isinstance(value, str) else ""


def _recomputed_canonical_payload_hash(payload: dict[str, object]) -> str:
    without_manifest = dict(payload)
    without_manifest.pop("tool_run_manifest", None)
    return sha256_bytes(json_text(without_manifest).encode("utf-8"))


def _canonical_payload_hash(payload: dict[str, object]) -> str:
    embedded = _embedded_canonical_payload_hash(payload)
    recomputed = _recomputed_canonical_payload_hash(payload)
    return embedded if embedded and embedded == recomputed else ""


def _manifest_hash_self_consistent(prefix: str, payload: dict[str, object]) -> dict[str, bool]:
    return {
        f"{prefix}_manifest_hash_self_consistent": bool(_embedded_canonical_payload_hash(payload))
        and _embedded_canonical_payload_hash(payload) == _recomputed_canonical_payload_hash(payload)
    }


def _closed_payload_checks(prefix: str, payload: dict[str, object]) -> dict[str, bool]:
    return {
        f"{prefix}_score_claim_false": payload.get("score_claim") is False,
        f"{prefix}_dispatch_attempted_false": payload.get("dispatch_attempted") is False,
        f"{prefix}_ready_false": payload.get("ready_for_exact_eval_dispatch") is False,
        f"{prefix}_promotion_false": payload.get("promotion_eligible") is False,
        f"{prefix}_blockers_nonempty": bool(payload.get("dispatch_blockers")),
    }


def _required_blockers_present(
    prefix: str,
    payload: dict[str, object],
    required_blockers: set[str],
) -> dict[str, bool]:
    blockers = set(payload.get("dispatch_blockers", []))
    return {f"{prefix}_required_blockers_present": required_blockers <= blockers}


def _pr91_runtime_source_inventory_checks(
    prefix: str,
    payload: dict[str, object],
) -> dict[str, bool]:
    inventory = payload.get("runtime_source_inventory")
    if not isinstance(inventory, dict):
        return {f"{prefix}_runtime_source_inventory_present": False}
    files = inventory.get("files")
    source_files = inventory.get("source_files")
    paths = {
        row.get("path")
        for row in (files if isinstance(files, list) else [])
        if isinstance(row, dict)
    }
    return {
        f"{prefix}_runtime_source_inventory_present": True,
        f"{prefix}_runtime_source_inventory_passed": (
            inventory.get("status") == "passed_static_source_inventory"
        ),
        f"{prefix}_runtime_source_required_files_present": (
            inventory.get("required_source_files_present") is True
            and {"inflate.py", "pr86_hpac.py"} <= paths
        ),
        f"{prefix}_runtime_source_not_pycache_only": (
            inventory.get("pycache_only") is False
            and isinstance(source_files, list)
            and len(source_files) >= 2
        ),
    }


def _pr91_runtime_device_contract_checks(
    prefix: str,
    payload: dict[str, object],
) -> dict[str, bool]:
    blockers = set(payload.get("dispatch_blockers", []))
    gates = payload.get("gates")
    device_gate = gates.get("hpac_device_contract_resolved") if isinstance(gates, dict) else {}
    contract = payload.get("hpac_device_contract")
    if not isinstance(device_gate, dict):
        device_gate = {}
    if not isinstance(contract, dict):
        contract = {}
    ambient_blocked = (
        "hpac_device_contract_resolved" in blockers
        and device_gate.get("passed") is False
        and contract.get("passed") is False
        and contract.get("status") == "blocked_ambient_or_contradictory"
        and int(payload.get("ambient_device_call_count", 0)) >= 1
        and int(payload.get("contradiction_count", 0)) >= 1
    )
    resolved_cpu = (
        "hpac_device_contract_resolved" not in blockers
        and device_gate.get("passed") is True
        and contract.get("passed") is True
        and contract.get("status") == "resolved_cpu_only"
        and contract.get("resolved_device") == "cpu"
        and int(payload.get("ambient_device_call_count", 0)) == 0
        and int(payload.get("contradiction_count", 0)) == 0
    )
    return {
        f"{prefix}_runtime_device_contract_fail_closed_or_resolved_cpu": ambient_blocked or resolved_cpu,
        f"{prefix}_runtime_device_contract_not_cuda": contract.get("resolved_device") != "cuda",
    }


def _run_pr91_hpm1_fail_closed_gate() -> tuple[bool, str]:
    ok_readiness, readiness, readiness_output = _json_tool(PR91_HPM1_READINESS_AUDIT)
    if not ok_readiness:
        return False, readiness_output
    ok_runtime, runtime, runtime_output = _json_tool(PR91_HPM1_RUNTIME_CONTRACT_AUDIT)
    if not ok_runtime:
        return False, runtime_output
    ok_readiness_artifact, readiness_artifact, readiness_artifact_error = _load_json_artifact(
        PR91_HPM1_READINESS_ARTIFACT
    )
    if not ok_readiness_artifact:
        return False, readiness_artifact_error
    ok_runtime_artifact, runtime_artifact, runtime_artifact_error = _load_json_artifact(
        PR91_HPM1_RUNTIME_CONTRACT_ARTIFACT
    )
    if not ok_runtime_artifact:
        return False, runtime_artifact_error
    readiness_blockers = set(readiness.get("dispatch_blockers", []))
    runtime_blockers = set(runtime.get("dispatch_blockers", []))
    required_readiness_blockers = {
        "full_hpm1_decode_600_frames",
        "byte_exact_hpm1_reencode",
        "runtime_hpm1_loader_without_sidecars",
        "exact_cuda_auth_eval_after_parity",
    }
    required_runtime_blockers = {"runtime_consumer_sidecar_free_hpm1"}
    readiness_hash = _canonical_payload_hash(readiness)
    readiness_artifact_hash = _canonical_payload_hash(readiness_artifact)
    runtime_hash = _canonical_payload_hash(runtime)
    runtime_artifact_hash = _canonical_payload_hash(runtime_artifact)
    checks = {
        **_closed_payload_checks("live_readiness", readiness),
        **_closed_payload_checks("artifact_readiness", readiness_artifact),
        **_closed_payload_checks("live_runtime", runtime),
        **_closed_payload_checks("artifact_runtime", runtime_artifact),
        **_manifest_hash_self_consistent("live_readiness", readiness),
        **_manifest_hash_self_consistent("artifact_readiness", readiness_artifact),
        **_manifest_hash_self_consistent("live_runtime", runtime),
        **_manifest_hash_self_consistent("artifact_runtime", runtime_artifact),
        **_required_blockers_present("live_readiness", readiness, required_readiness_blockers),
        **_required_blockers_present(
            "artifact_readiness",
            readiness_artifact,
            required_readiness_blockers,
        ),
        **_required_blockers_present("live_runtime", runtime, required_runtime_blockers),
        **_required_blockers_present("artifact_runtime", runtime_artifact, required_runtime_blockers),
        **_pr91_runtime_source_inventory_checks("live_readiness", readiness),
        **_pr91_runtime_source_inventory_checks("artifact_readiness", readiness_artifact),
        **_pr91_runtime_device_contract_checks("live_runtime", runtime),
        **_pr91_runtime_device_contract_checks("artifact_runtime", runtime_artifact),
        "readiness_artifact_hash_matches_live": bool(readiness_hash)
        and readiness_hash == readiness_artifact_hash,
        "runtime_artifact_hash_matches_live": bool(runtime_hash)
        and runtime_hash == runtime_artifact_hash,
        "source_archive_custody_passed": (
            isinstance(readiness.get("source_archive"), dict)
            and readiness["source_archive"].get("matches_expected") is True
        ),
        "member_x_custody_passed": (
            isinstance(readiness.get("member_x"), dict)
            and readiness["member_x"].get("matches_expected") is True
        ),
        "hpm1_mask_custody_passed": (
            isinstance(readiness.get("hpm1_mask_segment"), dict)
            and readiness["hpm1_mask_segment"].get("matches_expected") is True
        ),
        "zip_wire_contract_passed": (
            isinstance(readiness.get("member_x"), dict)
            and isinstance(readiness["member_x"].get("zip_report"), dict)
            and isinstance(readiness["member_x"]["zip_report"].get("wire_contract"), dict)
            and readiness["member_x"]["zip_report"]["wire_contract"].get("passed") is True
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        return False, (
            "PR91/HPM1 fail-closed gate failed: "
            + ", ".join(failed)
            + "\nreadiness="
            + json_text(readiness)
            + "\nruntime="
            + json_text(runtime)
        )
    return True, (
        "PR91/HPM1 readiness/runtime contract: PASS "
        "(static custody visible; runtime contract blocked; "
        "ready_for_exact_eval_dispatch=false; "
        f"readiness_blockers={len(readiness_blockers)}; "
        f"runtime_blockers={len(runtime_blockers)}; "
        "artifact_hashes_match=true)"
    )


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


def _timing_row(result: PreflightResult) -> dict[str, object]:
    step = result.step
    return {
        "section": step.section,
        "number": step.number,
        "name": step.name,
        "passed": result.passed,
        "elapsed_s": round(float(result.elapsed_s), 6),
        "forensic_only": step.forensic_only,
        "local_smoke_only": step.local_smoke_only,
    }


def _build_timing_profile(
    results: list[PreflightResult],
    *,
    max_workers: int,
    wall_elapsed_s: float,
) -> dict[str, object]:
    """Build a deterministic timing payload for DX profiling and CI trends."""
    rows = [
        _timing_row(result)
        for result in sorted(results, key=lambda item: (item.step.section, item.step.number))
    ]
    hot_steps = sorted(
        rows,
        key=lambda row: (
            -float(row["elapsed_s"]),
            str(row["section"]),
            int(row["number"]),
            str(row["name"]),
        ),
    )
    serial_elapsed_s = sum(result.elapsed_s for result in results)
    slow_steps = [
        row
        for row in hot_steps
        if float(row["elapsed_s"]) >= SLOW_STEP_THRESHOLD_S
    ]
    return {
        "schema": TIMING_PROFILE_SCHEMA,
        "max_workers": int(max_workers),
        "wall_elapsed_s": round(float(wall_elapsed_s), 6),
        "serial_elapsed_s": round(float(serial_elapsed_s), 6),
        "parallel_speedup_estimate": round(
            float(serial_elapsed_s) / float(wall_elapsed_s),
            6,
        )
        if wall_elapsed_s > 0
        else 0.0,
        "slowest_step_elapsed_s": round(
            max((result.elapsed_s for result in results), default=0.0),
            6,
        ),
        "slow_step_threshold_s": SLOW_STEP_THRESHOLD_S,
        "slow_step_count": len(slow_steps),
        "step_count": len(results),
        "passed_count": sum(1 for result in results if result.passed),
        "failed_count": sum(1 for result in results if not result.passed),
        "steps": rows,
        "slow_steps": slow_steps,
        "hot_steps": hot_steps,
    }


def _write_timing_profile(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_text(payload), encoding="utf-8")


def _format_timing_row(row: dict[str, object]) -> str:
    return (
        f"  {float(row['elapsed_s']):6.2f}s  "
        f"{row['section']} #{row['number']}: {row['name']}"
    )


def _print_timing_summary(
    results: list[PreflightResult],
    *,
    max_workers: int,
    wall_elapsed_s: float,
) -> None:
    profile = _build_timing_profile(
        results,
        max_workers=max_workers,
        wall_elapsed_s=wall_elapsed_s,
    )
    threshold = float(profile["slow_step_threshold_s"])
    slow_count = int(profile["slow_step_count"])
    hot_steps = [row for row in profile["hot_steps"] if isinstance(row, dict)]
    print("\nTimings:")
    print(
        "  "
        f"wall={float(profile['wall_elapsed_s']):.2f}s; "
        f"serial_sum={float(profile['serial_elapsed_s']):.2f}s; "
        f"workers={int(profile['max_workers'])}; "
        f"estimated_speedup={float(profile['parallel_speedup_estimate']):.2f}x; "
        f"slow_threshold={threshold:.2f}s; "
        f"slow_steps={slow_count}/{int(profile['step_count'])}"
    )
    if slow_count:
        print(f"  Slow steps (>= {threshold:.2f}s):")
        for row in hot_steps[:slow_count]:
            print(_format_timing_row(row))
    else:
        print(f"  Slow steps: none >= {threshold:.2f}s")
    remaining = hot_steps[slow_count:]
    if remaining:
        print("  Remaining steps:")
        for row in remaining:
            print(_format_timing_row(row))


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
    parser.add_argument(
        "--timings-json",
        type=Path,
        default=None,
        help=(
            "Write a deterministic JSON timing profile for slow-gate triage "
            "and Rust-acceleration planning."
        ),
    )
    args = parser.parse_args(argv)

    lanes = [dict(lane) for lane in LANES]
    if args.require_real_omega_sensitivity:
        for lane in lanes:
            if Path(lane["tool"]).name == "dispatch_dryrun_omega_w_v3.py":
                lane["args"] = [*lane["args"], "--require-real-sensitivity"]
                lane["local_smoke_only"] = False
    if args.jobs == 1:
        for lane in lanes:
            if Path(lane["tool"]).name == "dispatch_dryrun_apogee_intN.py":
                lane["args"] = [*lane["args"], "--jobs", "1"]

    # Verify all sub-tools exist before running any
    for tool in [
        SHELL_HAZARDS,
        REVERSE_ENGINEERING_AUDIT,
        HIDDEN_GEMS_REGISTRY,
        HIDDEN_GEMS_READINESS,
        SEMANTIC_LABEL_AUDIT,
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
        CROSS_PARADIGM_FRONTIER_INVENTORY,
        PR91_HPM1_READINESS_AUDIT,
        PR91_HPM1_RUNTIME_CONTRACT_AUDIT,
        FRONTIER_ARCHIVE_LAYOUT_AUDIT,
        OMEGA_OPT_ANCHOR_AUDIT,
        EVAL_LOADER_DRIFT_PROBE,
        HSTACK_VSTACK_PLAN,
        PR91_HPM1_READINESS_ARTIFACT,
        PR91_HPM1_RUNTIME_CONTRACT_ARTIFACT,
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
            "semantic-label contract",
            _run_semantic_label_contract_gate,
            "  ✓ Gate #3: semantic-label contract — PASSED",
            "  ✗ Gate #3: semantic-label contract — FAILED",
        ),
        PreflightStep(
            "GATE",
            4,
            "hidden-gem readiness",
            _run_hidden_gem_readiness_gate,
            "  ✓ Gate #4: hidden-gem readiness — PASSED",
            "  ✗ Gate #4: hidden-gem readiness — FAILED",
        ),
        PreflightStep(
            "GATE",
            5,
            "engineered-correction readiness",
            _run_engineered_corrections_gate,
            "  ✓ Gate #5: engineered-correction readiness — PASSED",
            "  ✗ Gate #5: engineered-correction readiness — FAILED",
        ),
        PreflightStep(
            "GATE",
            6,
            "HNeRV frontier scorecard",
            _run_hnerv_scorecard_gate,
            "  ✓ Gate #6: HNeRV frontier scorecard — PASSED",
            "  ✗ Gate #6: HNeRV frontier scorecard — FAILED",
        ),
        PreflightStep(
            "GATE",
            7,
            "HNeRV low-level repack proof",
            _run_hnerv_lowlevel_repack_gate,
            "  ✓ Gate #7: HNeRV low-level repack proof — PASSED",
            "  ✗ Gate #7: HNeRV low-level repack proof — FAILED",
        ),
        PreflightStep(
            "GATE",
            8,
            "tooling consolidation inventory",
            _run_tooling_consolidation_gate,
            "  ✓ Gate #8: tooling consolidation inventory — PASSED",
            "  ✗ Gate #8: tooling consolidation inventory — FAILED",
        ),
        PreflightStep(
            "GATE",
            9,
            "recovered remote lane canonicalization",
            _run_recovered_remote_lanes_gate,
            "  ✓ Gate #9: recovered remote lane canonicalization — PASSED",
            "  ✗ Gate #9: recovered remote lane canonicalization — FAILED",
        ),
        PreflightStep(
            "GATE",
            10,
            "untracked source inventory",
            _run_untracked_source_gate,
            "  ✓ Gate #10: untracked source inventory — PASSED (STRICT DISPOSITION)"
            if UNTRACKED_SOURCE_DISPOSITION_MANIFEST.exists()
            else "  ✓ Gate #10: untracked source inventory — PASSED (ADVISORY)",
            "  ✗ Gate #10: untracked source inventory — FAILED",
        ),
        PreflightStep(
            "GATE",
            11,
            "orphan recovery canonicalization",
            lambda: _run_gate("orphan recovery canonicalization", ORPHAN_RECOVERY_AUDIT),
            "  ✓ Gate #11: orphan recovery canonicalization — PASSED",
            "  ✗ Gate #11: orphan recovery canonicalization — FAILED",
        ),
        PreflightStep(
            "GATE",
            12,
            "preserved-orphan canonicalization",
            _run_preserved_orphans_gate,
            "  ✓ Gate #12: preserved-orphan canonicalization — PASSED",
            "  ✗ Gate #12: preserved-orphan canonicalization — FAILED",
        ),
        PreflightStep(
            "GATE",
            13,
            "recovery custody snapshots",
            _run_recovery_custody_snapshots_gate,
            "  ✓ Gate #13: recovery custody snapshots — PASSED",
            "  ✗ Gate #13: recovery custody snapshots — FAILED",
        ),
        PreflightStep(
            "GATE",
            14,
            "reverse-engineering release manifest",
            _run_reverse_engineering_release_gate,
            "  ✓ Gate #14: reverse-engineering release manifest — PASSED",
            "  ✗ Gate #14: reverse-engineering release manifest — FAILED",
        ),
        PreflightStep(
            "GATE",
            15,
            "release index/worktree split",
            _run_release_index_split_gate,
            "  ✓ Gate #15: release index/worktree split — PASSED",
            "  ✗ Gate #15: release index/worktree split — FAILED",
        ),
        PreflightStep(
            "GATE",
            16,
            "nested gitlink custody",
            _run_nested_gitlink_custody_gate,
            "  ✓ Gate #16: nested gitlink custody — PASSED",
            "  ✗ Gate #16: nested gitlink custody — FAILED",
        ),
        PreflightStep(
            "GATE",
            17,
            "staged public release hygiene",
            _run_staged_public_release_hygiene_gate,
            "  ✓ Gate #17: staged public release hygiene — PASSED",
            "  ✗ Gate #17: staged public release hygiene — FAILED",
        ),
        PreflightStep(
            "GATE",
            18,
            "cross-paradigm frontier inventory",
            _run_cross_paradigm_frontier_inventory_gate,
            "  ✓ Gate #18: cross-paradigm frontier inventory — PASSED",
            "  ✗ Gate #18: cross-paradigm frontier inventory — FAILED",
        ),
        PreflightStep(
            "GATE",
            19,
            "PR91 HPM1 fail-closed custody",
            _run_pr91_hpm1_fail_closed_gate,
            "  ✓ Gate #19: PR91 HPM1 fail-closed custody — PASSED",
            "  ✗ Gate #19: PR91 HPM1 fail-closed custody — FAILED",
        ),
        PreflightStep(
            "GATE",
            20,
            "frontier monolithic archive layout",
            _run_frontier_monolithic_layout_gate,
            "  ✓ Gate #20: frontier monolithic archive layout — PASSED",
            "  ✗ Gate #20: frontier monolithic archive layout — FAILED",
        ),
        PreflightStep(
            "GATE",
            21,
            "Omega-OPT anchor discipline",
            lambda: _run_gate(
                "Omega-OPT anchor discipline",
                OMEGA_OPT_ANCHOR_AUDIT,
                ["--plan-manifest", str(HSTACK_VSTACK_PLAN)],
            ),
            "  ✓ Gate #21: Omega-OPT anchor discipline — PASSED",
            "  ✗ Gate #21: Omega-OPT anchor discipline — FAILED",
        ),
        PreflightStep(
            "GATE",
            22,
            "eval loader drift diagnostic",
            _run_eval_loader_drift_probe_gate,
            "  ✓ Gate #22: eval loader drift diagnostic — PASSED",
            "  ✗ Gate #22: eval loader drift diagnostic — FAILED",
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
    run_started = time.perf_counter()
    if max_workers == 1:
        results = [_execute_step(step) for step in steps]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_execute_step, step) for step in steps]
            results = [future.result() for future in futures]
    wall_elapsed_s = time.perf_counter() - run_started

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
        _print_timing_summary(
            results,
            max_workers=max_workers,
            wall_elapsed_s=wall_elapsed_s,
        )
    if args.timings_json is not None:
        timing_profile = _build_timing_profile(
            results,
            max_workers=max_workers,
            wall_elapsed_s=wall_elapsed_s,
        )
        _write_timing_profile(args.timings_json, timing_profile)
        print(f"\nTiming profile JSON: {args.timings_json}")
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
