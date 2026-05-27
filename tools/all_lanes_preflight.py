#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
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
  Gate #23: tools/audit_a2_packet_ladder_closure.py --strict
           (A2 packet/probe artifacts keep no-score authority fields and
            inherited dispatch blockers alive until exact eval closure)
  Gate #24: Phase A post-green custody/discoverability source gate
           (A5 readiness, A6 measured-negative, and Modal A1 recovery
            surfaces remain visible and fail-closed without dispatching)
  Gate #25: tools/audit_modal_image_build_order.py --strict
           (Modal image build/env steps must happen before add_local_* mounts,
            otherwise dispatch fails locally before remote execution)
  Gate #26: PR106/R2 sidecar PacketIR + runtime-consumption proof
           (PacketIR parse/emit identity accounts for every payload byte, and
            runtime decodes/applies changed sidecar bytes for both grammars,
            while remaining non-promotable without full-frame parity)
  Gate #27: tools/claim_lane_dispatch.py summary
           (refuse stale active dispatch claims before more dispatch work)
  Gate #28: tools/operator_briefing.py --json
           (operator briefing active lists must use dispatch readiness, not
            score-band plausibility alone)
  Gate #29: terminal dispatch no-signal-loss evidence
           (recent terminal Modal substrate-smoke claims must appear in
            reports/cathedral_autopilot_evidence.jsonl)
  Gate #30: HLM1 non-promotional frontier prose guard
           (control ledgers must not call HLM1 the active floor/frontier
            without the non-promotional qualifier)
  Gate #32: tools/check_tac_terminology.py --strict
           (TAC means Task-Aware Compression; codec/compression and
            tac/comma_lab boundaries stay canonical)
  Gate #33: tools/audit_public_submission_pr.py --self-test --format json
           (public PR audit automation stays importable and no-network
            parser checks remain wired into operator flows)
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
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_TOOL_FILE = Path(__file__).resolve()
_REPO_ROOT_CANDIDATE = _TOOL_FILE.parents[1]
if str(_REPO_ROOT_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_CANDIDATE))
if str(_TOOL_FILE.parent) not in sys.path:
    sys.path.insert(0, str(_TOOL_FILE.parent))

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)


def _prepend_pythonpath(*paths: Path) -> None:
    """Make repo imports stable for child preflight subprocesses."""
    existing = os.environ.get("PYTHONPATH")
    ordered: list[str] = []
    seen: set[str] = set()
    for path in paths:
        value = str(path)
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    if existing:
        for value in existing.split(os.pathsep):
            if value and value not in seen:
                ordered.append(value)
                seen.add(value)
    os.environ["PYTHONPATH"] = os.pathsep.join(ordered)


def _maybe_reexec_repo_venv() -> None:
    """Use the repo virtualenv for direct script/shebang invocations.

    The documented command is ``.venv/bin/python tools/all_lanes_preflight.py``,
    but operators also run ``tools/all_lanes_preflight.py`` directly. The
    latter resolves through ``/usr/bin/env python3`` on macOS, which can miss
    package deps such as ``brotli`` and then fail child preflight tools. Re-exec
    once into the repo venv so both entrypoints exercise the same environment.
    """
    if os.environ.get("PACT_ALL_LANES_PREFLIGHT_REEXECED") == "1":
        return
    venv_python = REPO / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        return
    try:
        current_prefix = Path(sys.prefix).resolve()
        target_prefix = (REPO / ".venv").resolve()
        target = venv_python
    except OSError:
        return
    if current_prefix == target_prefix:
        return
    os.environ["PACT_ALL_LANES_PREFLIGHT_REEXECED"] = "1"
    os.execv(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]])


if __name__ == "__main__":
    _maybe_reexec_repo_venv()

_prepend_pythonpath(REPO / "src", REPO)
ensure_repo_imports(REPO)

from tac.deploy.claims import is_terminal_status as is_dispatch_claim_terminal_status  # noqa: E402
from tac.deploy.modal.paired_dispatch_contract import (  # noqa: E402
    paired_auth_eval_dispatch_command_blockers,
)
from tac.geometry_feedback_readiness import (  # noqa: E402
    GEOMETRY_FEEDBACK_ROADMAP_KEYS,
    geometry_feedback_contract_failures,
)
from tac.hnerv_frontier_defaults import HNERV_ACTIVE_SCORECARD  # noqa: E402
from tac.repo_io import json_text, sha256_bytes  # noqa: E402
from tac.source_index import SourceIndex  # noqa: E402

TOOLS = REPO / "tools"
SHELL_HAZARDS = TOOLS / "check_dispatch_cli_shell_hazards.py"
CANONICAL_TASK_STATUS_AUDIT = TOOLS / "check_canonical_task_status_no_dangling_transitions.py"
TAC_TERMINOLOGY_AUDIT = TOOLS / "check_tac_terminology.py"
PUBLIC_SUBMISSION_PR_AUDIT = TOOLS / "audit_public_submission_pr.py"
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
A2_PACKET_LADDER_CLOSURE_AUDIT = TOOLS / "audit_a2_packet_ladder_closure.py"
A5_PACKET_READINESS_TOOL = TOOLS / "build_pr101_frame_conditional_packet_readiness.py"
A6_BLOCKFP_HYPERPRIOR_ANCHOR = TOOLS / "pr101_a6_blockfp_hyperprior_anchor.py"
MODAL_IMAGE_BUILD_ORDER_AUDIT = TOOLS / "audit_modal_image_build_order.py"
MODAL_A1_SCORE_GRADIENT_DISPATCHER = REPO / "experiments/modal_phase_a1_score_gradient_pr101.py"
LIGHTNING_A1_SCORE_GRADIENT_DISPATCHER = TOOLS / "dispatch_phase_a1_score_gradient_pr101.py"
PR106_R2_ARCHIVE = REPO / "submissions/pr106_latent_sidecar_r2/archive.zip"
PR106_R2_RUNTIME = REPO / "submissions/pr106_latent_sidecar_r2"
PR106_R2_ARCHIVE_SHA256 = "7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f"
PR106_R2_RUNTIME_SOURCE_TREE_SHA256 = "69507e99b9280d917b670052421e28f87b1dd197af039527dcd270774509ffcc"
PR106_R2_PR101_ARCHIVE = REPO / "submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip"
PR106_R2_PR101_RUNTIME = REPO / "submissions/pr106_latent_sidecar_r2_pr101_grammar"
PR106_R2_PR101_ARCHIVE_SHA256 = "c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383"
PR106_R2_PR101_RUNTIME_SOURCE_TREE_SHA256 = "373f19a1a892cf21c432d4949312cc788f4d4d23c02f2c1ca0cb3e666fc5c4bc"
PR106_FORMAT0C_XMEMBER_ARCHIVE = (
    REPO
    / "experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex"
    / "candidates/pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip"
)
PR106_FORMAT0C_XMEMBER_ARCHIVE_SHA256 = "56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7"
PR106_R2_HLM1_XMEMBER_ARCHIVE = (
    REPO
    / "experiments/results/pr106_r2_hdm4_hlm1_xmember_candidate_20260514_codex"
    / "pr106_r2_hdm4_hlm1_xmember_hlm1_latent_candidate.zip"
)
PR106_R2_HLM1_XMEMBER_ARCHIVE_SHA256 = "391400008b69e66f8bd522f4eb2a53c465e58a17e536d171caf039f9e51e874f"
PR106_R2_HLM2_XMEMBER_ARCHIVE = (
    REPO
    / "experiments/results/pr106_r2_hdm4_hlm2_latent_candidate_20260514_codex"
    / "pr106_r2_hdm4_hlm1_xmember_hlm2_latent_candidate.zip"
)
PR106_R2_HLM2_XMEMBER_ARCHIVE_SHA256 = "2c6e5f8d71f687227a28a9a378dc5edfc3215b762015042203b6bf58bfee9378"
CLAIM_LANE_DISPATCH = TOOLS / "claim_lane_dispatch.py"
OPERATOR_BRIEFING = TOOLS / "operator_briefing.py"
CATHEDRAL_AUTOPILOT_EVIDENCE = REPO / "reports" / "cathedral_autopilot_evidence.jsonl"
PR106_R2_SAME_RUNTIME_FULL_FRAME_PARITY = (
    REPO / "experiments/results/pr106_r2_same_runtime_full_frame_parity_local_cpu.json"
)
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
EVAL_LOADER_DRIFT_EXPECTED_CELL_IDS = (
    "cpu_av",
    "cuda_dali",
    "cuda_av_shared_input",
    "cpu_dali",
)
EVAL_LOADER_DRIFT_EXPECTED_PLAN_IDS = (
    "raw_decoder_input_byte_drift_pre_network",
    "forward_kernel_drift_fixed_pyav_input",
    "forward_kernel_drift_fixed_dali_input",
    "decoder_effect_fixed_cpu_forward",
    "decoder_effect_fixed_cuda_forward",
)
EVAL_LOADER_DRIFT_FALSE_CUSTODY_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
    "dispatch_attempted",
)
EVAL_LOADER_DRIFT_AXIS_FALSE_CUSTODY_FIELDS = (
    "contest_cpu_axis_claim",
    "contest_cuda_axis_claim",
    "contest_cuda_claim",
    "contest_cpu_claim",
    "macos_cpu_advisory_claim",
    "mps_claim",
    "promotion_eligible",
    "score_claim_valid",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
    "dispatch_attempted",
)
EVAL_LOADER_DRIFT_LABEL_FALSE_CUSTODY_FIELDS = (
    "contest_cpu_axis_claim",
    "contest_cuda_axis_claim",
    "dispatch_attempted",
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
    HNERV_ACTIVE_SCORECARD
)
HNERV_SCORECARD_REQUIRED_EVALS = (
    (
        "PR106-R2-HDM4-HLM1",
        REPO
        / "experiments/results/modal_auth_eval/"
        / "hnerv_hlm1_fixed_latent_recode_modal_t4_enforced_20260513/"
        / "contest_auth_eval.adjudicated.json",
    ),
    (
        "PR106-R2-HDM4-HLM1-XMEMBER",
        REPO
        / "experiments/results/modal_auth_eval/"
        / "hnerv_hlm1_xmember_modal_t4_20260514/"
        / "contest_auth_eval.json",
    ),
    (
        "PR106-R2-HDM4-HLM2-XMEMBER",
        REPO
        / "experiments/results/modal_auth_eval/"
        / "hnerv_hlm2_xmember_modal_t4_20260514T065903Z/"
        / "contest_auth_eval.json",
    ),
    (
        "PR106-R2-HDM7-HLM2-XMEMBER",
        REPO
        / "experiments/results/modal_auth_eval/"
        / "hnerv_hdm7_hdm6_hlm2_modal_t4_retry1_20260514T090222Z/"
        / "contest_auth_eval.json",
    ),
    (
        "PR106-R2-HDM8-HLM2-XMEMBER",
        REPO
        / "experiments/results/modal_auth_eval/"
        / "pr106_hdm8_fixed_meta_rank_elided_exact_cuda_20260515T002100Z/"
        / "contest_auth_eval.json",
    ),
    (
        "PR106-R2-HDM11-HLM3-MAGICLESS-FMT0B",
        REPO
        / "experiments/results/modal_auth_eval/"
        / "pr106_hdm11_hlm3_fmt0b_t4_20260515T073414Z/"
        / "contest_auth_eval.json",
    ),
    (
        "PR106-R2-HDM12-HLM3-MAGICLESS-FMT0C",
        REPO
        / "experiments/results/modal_auth_eval/"
        / "pr106_format0c_exact_radix_paired_20260515T0918Z_cuda/"
        / "contest_auth_eval.json",
    ),
)
PR106X_ARCHIVE = (
    REPO
    / "experiments/results/lightning_batch/"
    / "exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/archive.zip"
)
TIMING_PROFILE_SCHEMA = "pact.all_lanes_preflight_timing.v1"
SLOW_STEP_THRESHOLD_S = 0.50
DEFAULT_ALL_LANES_PREFLIGHT_TIMEOUT_S = 30.0
DEFAULT_HARD_WATCHDOG_GRACE_S = 2.0

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
    status: str = "passed"


@dataclass(frozen=True)
class PreflightRunContext:
    """Shared wall-clock deadline and cancellation state for one preflight run."""

    started_s: float
    deadline_s: float | None
    cancel_event: threading.Event

    def remaining_s(self) -> float | None:
        if self.deadline_s is None:
            return None
        return self.deadline_s - time.perf_counter()


_THREAD_CONTEXT = threading.local()
_SOURCE_INDEX_GATE_LOCK = threading.Lock()


def _current_run_context() -> PreflightRunContext | None:
    context = getattr(_THREAD_CONTEXT, "context", None)
    return context if isinstance(context, PreflightRunContext) else None


def _run_source_index_gate(runner: Callable[[], tuple[bool, str]]) -> tuple[bool, str]:
    """Serialize broad shared-index gates to avoid cache-lock thrash.

    Gates #0/#3/#8 share one ``SourceIndex`` and each touches hundreds or
    thousands of files. Running them concurrently on the same Python cache
    increases RLock contention and duplicate cache work on macOS; other
    subprocess-heavy gates still run in parallel around this critical section.
    """

    with _SOURCE_INDEX_GATE_LOCK:
        return runner()


def _remaining_wall_budget_s() -> float | None:
    context = _current_run_context()
    if context is None:
        return None
    return context.remaining_s()


def _format_timeout_message(command: object, timeout_s: float) -> str:
    return (
        "TIMEOUT: all-lanes preflight wall-clock budget exhausted while running "
        f"{command!r} (remaining timeout={max(timeout_s, 0.0):.3f}s)"
    )


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _run_subprocess(*popenargs, **kwargs) -> subprocess.CompletedProcess:
    """Run a child command with its timeout capped by the remaining wall budget."""

    command = popenargs[0] if popenargs else kwargs.get("args")
    remaining = _remaining_wall_budget_s()
    existing_timeout = kwargs.get("timeout")
    if remaining is not None:
        if remaining <= 0:
            return subprocess.CompletedProcess(
                command,
                124,
                stdout="",
                stderr=_format_timeout_message(command, remaining) + "\n",
            )
        kwargs["timeout"] = (
            remaining
            if existing_timeout is None
            else min(float(existing_timeout), max(remaining, 0.001))
        )
    try:
        return subprocess.run(*popenargs, **kwargs)  # subprocess-no-check-OK: helper returns CompletedProcess; caller owns returncode handling
    except subprocess.TimeoutExpired as exc:
        stdout = _as_text(exc.stdout)
        stderr = _as_text(exc.stderr)
        stderr += _format_timeout_message(command, float(exc.timeout or 0.0)) + "\n"
        return subprocess.CompletedProcess(command, 124, stdout=stdout, stderr=stderr)


def _run_lane(lane: dict, verbose: bool) -> tuple[bool, str]:
    args = [sys.executable, str(lane["tool"])] + lane["args"]
    if verbose and lane.get("supports_verbose", True):
        args.append("--verbose")
    proc = _run_subprocess(args, capture_output=True, text=True)
    return proc.returncode == 0, proc.stdout + proc.stderr


def _run_gate(name: str, tool: Path, extra_args: list[str] | None = None) -> tuple[bool, str]:
    proc = _run_subprocess(
        [sys.executable, str(tool), "--repo-root", str(REPO), "--strict", *(extra_args or [])],
        capture_output=True,
        text=True,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode == 0 and not output.strip():
        output = f"{name}: PASS"
    return proc.returncode == 0, output


def _run_dispatch_cli_shell_hazards_gate(source_index=None) -> tuple[bool, str]:
    """Run Gate #0 in-process to avoid Python startup cost.

    This preserves the standalone tool's strict behavior: any hazard is a
    failed gate, and scanner exceptions are allowed to bubble to the step
    fail-closed wrapper.
    """

    from tools import check_dispatch_cli_shell_hazards as module

    hazards = module.scan_paths(
        REPO,
        scan_paths=module.DEFAULT_SCAN_PATHS,
        excludes=module.DEFAULT_EXCLUDES,
        source_index=source_index,
    )
    if hazards:
        lines = [
            f"{hazard.path}:{hazard.line}: {hazard.kind}: {hazard.message}"
            for hazard in hazards
        ]
        return False, "\n".join(lines)
    return True, "dispatch CLI/shell hazards: PASS"


def _run_active_dispatch_claims_gate() -> tuple[bool, str]:
    """Refuse pre-dispatch green status while prior dispatch claims are active.

    This is intentionally cheap and local. It catches the repeated failure mode
    where a remote GHA/Modal run has already failed or been abandoned, but the
    claim ledger still contains a nonterminal row that will block the next
    same-lane dispatch.
    """

    proc = _run_subprocess(
        [
            sys.executable,
            str(CLAIM_LANE_DISPATCH),
            "summary",
            "--format",
            "json",
            "--live-only",
        ],
        capture_output=True,
        text=True,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return False, output
    try:
        live_summary = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return False, f"dispatch claim live summary was not JSON: {exc}\n{output}"
    if not isinstance(live_summary, dict):
        return False, "dispatch claim live summary was not a JSON object"
    active = int(live_summary.get("active_count", 0) or 0)
    stale = int(live_summary.get("stale_nonterminal_count", 0) or 0)
    unparsable = int(live_summary.get("unparsable_timestamp_count", 0) or 0)
    invalid_lane_id = int(live_summary.get("invalid_lane_id_count", 0) or 0)
    if active or stale or unparsable or invalid_lane_id:
        return (
            False,
            "dispatch claims are not clean before pre-dispatch work: "
            f"active={active} stale_nonterminal={stale} "
            f"unparsable_timestamp={unparsable} invalid_lane_id={invalid_lane_id}\n{output}",
        )
    historical_proc = _run_subprocess(
        [sys.executable, str(CLAIM_LANE_DISPATCH), "summary", "--format", "json"],
        capture_output=True,
        text=True,
    )
    historical_note = "historical_claim_hygiene=unknown"
    if historical_proc.returncode == 0:
        try:
            historical_summary = json.loads(historical_proc.stdout)
        except json.JSONDecodeError:
            historical_note = "historical_claim_hygiene=warning non_json_summary"
        else:
            if isinstance(historical_summary, dict):
                historical_invalid = int(
                    historical_summary.get("invalid_lane_id_count", 0) or 0
                )
                historical_unparsable = int(
                    historical_summary.get("unparsable_timestamp_count", 0) or 0
                )
                historical_note = (
                    "historical_claim_hygiene="
                    + (
                        "warning "
                        if historical_invalid or historical_unparsable
                        else "pass "
                    )
                    + f"invalid_lane_id={historical_invalid} "
                    + f"unparsable_timestamp={historical_unparsable}"
                )
    return (
        True,
        "active dispatch claims: PASS "
        f"(active={active} stale_nonterminal={stale} "
        f"unparsable_timestamp={unparsable} invalid_lane_id={invalid_lane_id}; "
        f"{historical_note})",
    )


_OPERATOR_PLACEHOLDER_RE = re.compile(r"<[^>\n]+>")
_EXPECTED_XRAY_TOOLS = frozenset(
    {
        "tools/xray_archive_section_entropy_heatmap.py",
        "tools/xray_per_pr_archive_layout_compare.py",
        "tools/xray_per_tensor_saliency_heatmap.py",
        "tools/xray_inflate_op_cost_profiler.py",
        "tools/xray_cpu_cuda_drift_per_arch_class.py",
        "tools/xray_paired_cpu_cuda_axis_delta.py",
        "tools/xray_pair_component_errors.py",
        "tools/xray_hardpair_hitlist.py",
        "tools/xray_substrate_classifier.py",
        "tools/xray_per_frame_difficulty_profile.py",
        "tools/master_gradient_xray.py",
    }
)


def _operator_briefing_dispatch_failures(payload: dict[str, object]) -> list[str]:
    failures: list[str] = []
    claim_summary = payload.get("dispatch_claim_summary")
    if not isinstance(claim_summary, dict):
        failures.append("dispatch_claim_summary_missing_or_not_object")
    else:
        for field in (
            "active_count",
            "stale_nonterminal_count",
            "unparsable_timestamp_count",
            "invalid_lane_id_count",
        ):
            try:
                value = int(claim_summary.get(field, 0) or 0)
            except (TypeError, ValueError):
                failures.append(f"dispatch_claim_summary:{field}:not_integer")
                continue
            if value:
                failures.append(f"dispatch_claim_summary:{field}:{value}")
    historical_summary = payload.get("dispatch_claim_historical_summary")
    if not isinstance(historical_summary, dict):
        failures.append("dispatch_claim_historical_summary_missing_or_not_object")
    else:
        for field in ("unparsable_timestamp_count", "invalid_lane_id_count"):
            try:
                int(historical_summary.get(field, 0) or 0)
            except (TypeError, ValueError):
                failures.append(f"dispatch_claim_historical_summary:{field}:not_integer")
    dispatch_readiness = payload.get("dispatch_readiness")
    if not isinstance(dispatch_readiness, dict):
        failures.append("dispatch_readiness_missing_or_not_object")
    else:
        if dispatch_readiness.get("schema") != "pact.operator_dispatch_readiness.v1":
            failures.append("dispatch_readiness:bad_schema")
        phase1 = dispatch_readiness.get("phase_1_exact_eval_packets")
        if not isinstance(phase1, dict):
            failures.append("dispatch_readiness:phase_1_exact_eval_packets_missing")
        else:
            status = phase1.get("status")
            if status not in {"READY", "BLOCKED", "PENDING"}:
                failures.append("dispatch_readiness:phase_1_exact_eval_packets_bad_status")
        phase6d = dispatch_readiness.get("phase_6d_frontier_feedback_cycle")
        if not isinstance(phase6d, dict):
            failures.append("dispatch_readiness:phase_6d_frontier_feedback_cycle_missing")
        else:
            status = phase6d.get("status")
            if status not in {
                "PENDING",
                "READY_CYCLE",
                "READY_LOCAL_EXECUTION",
                "POST_HARVEST_QUEUE_READY",
                "BLOCKED",
            }:
                failures.append("dispatch_readiness:phase_6d_frontier_feedback_cycle_bad_status")
    exact_packets = payload.get("exact_eval_packets")
    if not isinstance(exact_packets, list):
        failures.append("exact_eval_packets_missing_or_not_list")
    else:
        terminal_or_blocked_packets = 0
        for packet in exact_packets:
            if not isinstance(packet, dict):
                failures.append("exact_eval_packets_contains_non_object_row")
                continue
            lane_id = str(packet.get("lane_id") or "<missing>")
            terminal_blockers = packet.get("terminal_exact_eval_evidence_blockers")
            terminal_action = packet.get("dispatch_action") == "terminal_exact_eval_evidence_stop"
            terminal_packet = bool(terminal_blockers) or terminal_action
            repeat_allowed = packet.get("repeat_dispatch_allowed")
            ready = packet.get("ready_for_submit")
            commands = packet.get("commands")
            if ready is not True or terminal_packet:
                terminal_or_blocked_packets += 1
            if terminal_packet:
                if repeat_allowed is not False:
                    failures.append(
                        f"exact_eval_packets:{lane_id}:"
                        "terminal_evidence_not_suppressing_repeat_dispatch"
                    )
                if ready is not False:
                    failures.append(
                        f"exact_eval_packets:{lane_id}:terminal_evidence_ready_for_submit"
                    )
                if commands:
                    failures.append(
                        f"exact_eval_packets:{lane_id}:terminal_evidence_commands_not_suppressed"
                    )
            if ready is True:
                if not isinstance(commands, dict):
                    failures.append(f"exact_eval_packets:{lane_id}:ready_missing_commands")
                else:
                    for key in ("claim", "submit", "harvest"):
                        if not str(commands.get(key) or "").strip():
                            failures.append(
                                f"exact_eval_packets:{lane_id}:ready_missing_{key}_command"
                            )
        if exact_packets and terminal_or_blocked_packets == len(exact_packets):
            phase1 = (
                dispatch_readiness.get("phase_1_exact_eval_packets")
                if isinstance(dispatch_readiness, dict)
                else {}
            )
            if isinstance(phase1, dict) and phase1.get("status") == "READY":
                failures.append(
                    "dispatch_readiness:phase_1_ready_while_all_exact_packets_blocked"
                )
    readiness_artifacts = payload.get("non_dispatchable_readiness_artifacts")
    if not isinstance(readiness_artifacts, list):
        failures.append("non_dispatchable_readiness_artifacts_missing_or_not_list")
    else:
        for artifact in readiness_artifacts:
            if not isinstance(artifact, dict):
                failures.append("non_dispatchable_readiness_artifacts_contains_non_object_row")
                continue
            kind = str(artifact.get("kind") or "<missing>")
            if artifact.get("ready_for_exact_eval_dispatch") is not False:
                failures.append(
                    f"non_dispatchable_readiness_artifacts:{kind}:"
                    "ready_for_exact_eval_dispatch_not_false"
                )
            if artifact.get("score_claim") is not False:
                failures.append(
                    f"non_dispatchable_readiness_artifacts:{kind}:score_claim_not_false"
                )
            if artifact.get("promotion_eligible") is True:
                failures.append(
                    f"non_dispatchable_readiness_artifacts:{kind}:promotion_eligible_true"
                )
            if artifact.get("rank_or_kill_eligible") is True:
                failures.append(
                    f"non_dispatchable_readiness_artifacts:{kind}:rank_or_kill_eligible_true"
                )
            if kind == "inverse_scorer_cell_candidate_chain":
                artifact_blockers = {
                    str(item)
                    for item in artifact.get("dispatch_blockers", [])
                    if str(item)
                }
                parity_satisfied = artifact.get("inflate_parity_satisfied") is True
                if not parity_satisfied and "candidate_inflate_output_parity_missing" not in artifact_blockers:
                    failures.append(
                        "non_dispatchable_readiness_artifacts:"
                        "inverse_scorer_cell_candidate_chain:missing_parity_blocker"
                    )
            if (
                artifact.get("receiver_contract_satisfied") is True
                and parity_satisfied
                and "exact_auth_eval_required_before_score_claim" not in artifact_blockers
            ):
                failures.append(
                    "non_dispatchable_readiness_artifacts:"
                    "inverse_scorer_cell_candidate_chain:missing_exact_auth_blocker"
                )
    frontier_feedback_cycle = payload.get("frontier_feedback_cycle")
    if not isinstance(frontier_feedback_cycle, dict):
        failures.append("frontier_feedback_cycle_missing_or_not_object")
    else:
        if frontier_feedback_cycle.get("schema") != "pact.frontier_feedback_cycle_summary.v1":
            failures.append("frontier_feedback_cycle:bad_schema")
        if frontier_feedback_cycle.get("cycle_tool_exists") is not True:
            failures.append("frontier_feedback_cycle:cycle_tool_missing")
        for flag in (
            "score_claim",
            "score_claim_valid",
            "promotion_eligible",
            "rank_or_kill_eligible",
            "ready_for_exact_eval_dispatch",
            "dispatch_attempted",
            "gpu_launched",
        ):
            if frontier_feedback_cycle.get(flag) is not False:
                failures.append(f"frontier_feedback_cycle:{flag}_not_false")
        try:
            error_count = int(frontier_feedback_cycle.get("error_count") or 0)
        except (TypeError, ValueError):
            failures.append("frontier_feedback_cycle:error_count_not_integer")
            error_count = 0
        if error_count:
            failures.append(f"frontier_feedback_cycle:error_count:{error_count}")
        status = frontier_feedback_cycle.get("status")
        if status not in {
            "PENDING",
            "READY_CYCLE",
            "READY_LOCAL_EXECUTION",
            "POST_HARVEST_QUEUE_READY",
            "BLOCKED",
        }:
            failures.append("frontier_feedback_cycle:bad_status")
        if status in {
            "READY_CYCLE",
            "READY_LOCAL_EXECUTION",
            "POST_HARVEST_QUEUE_READY",
        } and "run_frontier_rate_attack_feedback_cycle.py" not in str(
            frontier_feedback_cycle.get("next_command") or ""
        ):
            failures.append("frontier_feedback_cycle:next_command_missing_cycle_tool")
        if isinstance(dispatch_readiness, dict):
            phase6d = dispatch_readiness.get("phase_6d_frontier_feedback_cycle")
            if isinstance(phase6d, dict):
                if phase6d.get("status") != status:
                    failures.append("frontier_feedback_cycle:phase_6d_status_mismatch")
                if str(phase6d.get("next_command") or "") != str(
                    frontier_feedback_cycle.get("next_command") or ""
                ):
                    failures.append("frontier_feedback_cycle:phase_6d_next_command_mismatch")
    l5 = payload.get("l5_v2_frontier_readiness")
    if not isinstance(l5, dict):
        failures.append("l5_v2_frontier_readiness_missing_or_not_object")
    else:
        if l5.get("schema") != "pact.l5_v2_frontier_readiness.v1":
            failures.append("l5_v2_frontier_readiness:bad_schema")
        for flag in (
            "score_claim",
            "promotion_eligible",
            "rank_or_kill_eligible",
            "ready_for_exact_eval_dispatch",
            "measurement_schedule_score_claim",
            "measurement_schedule_promotion_eligible",
            "measurement_schedule_ready_for_exact_eval_dispatch",
        ):
            if l5.get(flag) is not False:
                failures.append(f"l5_v2_frontier_readiness:{flag}_not_false")
        for flag in (
            "l5_ready_for_score_or_rank_dispatch",
            "l5_ready_for_dispatch",
        ):
            if l5.get(flag) is True:
                failures.append(
                    f"l5_v2_frontier_readiness:{flag}_true_without_top_level_authority"
                )
        try:
            target_count = int(l5.get("next_exact_eval_target_count") or 0)
        except (TypeError, ValueError):
            failures.append("l5_v2_frontier_readiness:target_count_not_integer")
            target_count = 0
        l5_blockers = [str(blocker) for blocker in l5.get("blockers", []) if str(blocker)]
        if "l5_v2_packetir_matrix_artifact_sha_mismatch" in l5_blockers:
            failures.append(
                "l5_v2_frontier_readiness:l5_v2_packetir_matrix_artifact_sha_mismatch"
            )
        asymptotic_candidates = l5.get("asymptotic_pursuit_candidates")
        try:
            asymptotic_count = int(
                l5.get("asymptotic_pursuit_candidate_count") or 0
            )
        except (TypeError, ValueError):
            failures.append(
                "l5_v2_frontier_readiness:asymptotic_candidate_count_not_integer"
            )
            asymptotic_count = 0
        if not isinstance(asymptotic_candidates, list):
            failures.append(
                "l5_v2_frontier_readiness:"
                "asymptotic_pursuit_candidates_missing_or_not_list"
            )
            asymptotic_candidates = []
        if len(asymptotic_candidates) != asymptotic_count:
            failures.append(
                "l5_v2_frontier_readiness:"
                f"asymptotic_candidate_count_mismatch:{len(asymptotic_candidates)}"
                f"!={asymptotic_count}"
            )
        asymptotic_next_action_status = l5.get(
            "l5_v2_asymptotic_next_action_status"
        )
        if not isinstance(asymptotic_next_action_status, list):
            failures.append(
                "l5_v2_frontier_readiness:"
                "asymptotic_next_action_status_missing_or_not_list"
            )
            asymptotic_next_action_status = []
        if len(asymptotic_next_action_status) != asymptotic_count:
            failures.append(
                "l5_v2_frontier_readiness:"
                f"asymptotic_next_action_status_count_mismatch:"
                f"{len(asymptotic_next_action_status)}!={asymptotic_count}"
            )
        for idx, candidate in enumerate(asymptotic_candidates):
            if not isinstance(candidate, dict):
                failures.append(
                    "l5_v2_frontier_readiness:"
                    f"asymptotic_candidate_{idx}:not_object"
                )
                continue
            candidate_id = str(candidate.get("candidate_id") or idx)
            for flag in (
                "score_claim",
                "promotion_eligible",
                "rank_or_kill_eligible",
                "ready_for_exact_eval_dispatch",
                "ready_for_paid_dispatch",
                "ready_for_l1_scaffold_dispatch",
            ):
                if candidate.get(flag) is not False:
                    failures.append(
                        "l5_v2_frontier_readiness:"
                        f"asymptotic_candidate:{candidate_id}:{flag}_not_false"
                    )
            if candidate.get("local_ledger_present") is not True:
                failures.append(
                    "l5_v2_frontier_readiness:"
                    f"asymptotic_candidate:{candidate_id}:ledger_missing"
                )
            if candidate.get("lane_registry_registered") is not True:
                failures.append(
                    "l5_v2_frontier_readiness:"
                    f"asymptotic_candidate:{candidate_id}:lane_registry_missing"
                )
            next_action_status = candidate.get(
                "l5_v2_asymptotic_next_action_status"
            )
            if not isinstance(next_action_status, dict):
                failures.append(
                    "l5_v2_frontier_readiness:"
                    f"asymptotic_candidate:{candidate_id}:"
                    "next_action_status_missing_or_not_object"
                )
                next_action_status = {}
            else:
                if next_action_status.get("schema") != (
                    "l5_v2_asymptotic_next_action_status_v1"
                ):
                    failures.append(
                        "l5_v2_frontier_readiness:"
                        f"asymptotic_candidate:{candidate_id}:"
                        "next_action_status_bad_schema"
                    )
                if str(next_action_status.get("candidate_id") or "") != candidate_id:
                    failures.append(
                        "l5_v2_frontier_readiness:"
                        f"asymptotic_candidate:{candidate_id}:"
                        "next_action_status_candidate_mismatch"
                    )
                if next_action_status.get("ledger_present") != candidate.get(
                    "local_ledger_present"
                ):
                    failures.append(
                        "l5_v2_frontier_readiness:"
                        f"asymptotic_candidate:{candidate_id}:"
                        "next_action_status_ledger_presence_mismatch"
                    )
                if next_action_status.get("ledger_sha256") != candidate.get(
                    "local_ledger_sha256"
                ):
                    failures.append(
                        "l5_v2_frontier_readiness:"
                        f"asymptotic_candidate:{candidate_id}:"
                        "next_action_status_ledger_sha_mismatch"
                    )
                if next_action_status.get("lane_registry_registered") != (
                    candidate.get("lane_registry_registered")
                ):
                    failures.append(
                        "l5_v2_frontier_readiness:"
                        f"asymptotic_candidate:{candidate_id}:"
                        "next_action_status_registry_mismatch"
                    )
                canonical_replacement_registered = (
                    next_action_status.get("canonical_replacement_lane_registered")
                    is True
                )
                if (
                    next_action_status.get("ledger_present") is True
                    and next_action_status.get("lane_registry_registered") is not True
                    and not canonical_replacement_registered
                ):
                    failures.append(
                        "l5_v2_frontier_readiness:"
                        f"asymptotic_candidate:{candidate_id}:"
                        "next_action_status_lane_registry_missing"
                    )
                next_prerequisite_status = next_action_status.get(
                    "next_prerequisite_status"
                )
                if not isinstance(next_prerequisite_status, dict):
                    failures.append(
                        "l5_v2_frontier_readiness:"
                        f"asymptotic_candidate:{candidate_id}:"
                        "next_prerequisite_status_missing_or_not_object"
                    )
                elif next_prerequisite_status.get("ready_for_l1_build") != (
                    candidate.get("ready_for_l1_build")
                ):
                    failures.append(
                        "l5_v2_frontier_readiness:"
                        f"asymptotic_candidate:{candidate_id}:"
                        "next_prerequisite_l1_build_mismatch"
                    )
            l1_semantics = candidate.get("ready_for_l1_build_semantics")
            valid_l1_semantics = {
                "ready_to_start_l1_scaffold_work_only_not_scaffold_ready",
                "l1_scaffold_present_next_action_completed",
            }
            if l1_semantics not in valid_l1_semantics:
                failures.append(
                    "l5_v2_frontier_readiness:"
                    f"asymptotic_candidate:{candidate_id}:l1_semantics_missing"
                )
            if (
                candidate.get("ready_for_l1_build") is True
                and l1_semantics
                != "ready_to_start_l1_scaffold_work_only_not_scaffold_ready"
            ):
                failures.append(
                    "l5_v2_frontier_readiness:"
                    f"asymptotic_candidate:{candidate_id}:l1_build_semantics_invalid"
                )
            if (
                candidate.get("l1_scaffold_present") is True
                and l1_semantics != "l1_scaffold_present_next_action_completed"
            ):
                failures.append(
                    "l5_v2_frontier_readiness:"
                    f"asymptotic_candidate:{candidate_id}:completed_l1_semantics_invalid"
                )
            if (
                candidate.get("recommended_next_action_completed_or_superseded")
                is True
                and candidate.get("ready_for_recommended_next_action") is True
            ):
                failures.append(
                    "l5_v2_frontier_readiness:"
                    f"asymptotic_candidate:{candidate_id}:completed_action_still_ready"
                )
        tt5l = l5.get("tt5l_campaign_readiness")
        if not isinstance(tt5l, dict):
            failures.append("l5_v2_frontier_readiness:tt5l_campaign_missing_or_not_object")
        else:
            if tt5l.get("schema") != "l5_v2_tt5l_campaign_readiness_v1":
                failures.append("l5_v2_frontier_readiness:tt5l_campaign_bad_schema")
            if tt5l.get("non_pr106_staircase_priority") is not True:
                failures.append(
                    "l5_v2_frontier_readiness:tt5l_non_pr106_priority_not_true"
                )
            for flag in (
                "score_claim",
                "promotion_eligible",
                "rank_or_kill_eligible",
                "ready_for_exact_eval_dispatch",
            ):
                if tt5l.get(flag) is not False:
                    failures.append(
                        f"l5_v2_frontier_readiness:tt5l_campaign:{flag}_not_false"
                    )
            dykstra_valid = tt5l.get("dykstra_feasibility_artifact_valid") is True
            move_level_valid = (
                tt5l.get("move_level_feasibility_artifact_valid") is True
            )
            sideinfo_valid = tt5l.get("sideinfo_gate_evidence_valid") is True
            probe_valid = tt5l.get("probe_gate_evidence_valid") is True
            paired_axis_plan_valid = (
                tt5l.get("paired_axis_plan_evidence_valid") is True
            )
            if l5.get("l5_ready_for_gate_probe_dispatch") is True and not (
                dykstra_valid and move_level_valid
            ):
                failures.append(
                    "l5_v2_frontier_readiness:"
                    "gate_probe_dispatch_without_tt5l_cargo_cult_preconditions"
                )
            if tt5l.get("sideinfo_effect_curve_allowed") is True and not (
                dykstra_valid and move_level_valid and sideinfo_valid
            ):
                failures.append(
                    "l5_v2_frontier_readiness:"
                    "tt5l_sideinfo_effect_curve_without_dykstra_move_level_and_sideinfo"
                )
            if tt5l.get("first_anchor_timing_smoke_allowed") is True and not (
                dykstra_valid
                and move_level_valid
                and sideinfo_valid
                and probe_valid
                and paired_axis_plan_valid
            ):
                failures.append(
                    "l5_v2_frontier_readiness:"
                    "tt5l_timing_smoke_without_dykstra_move_level_sideinfo_probe_paired_axis_plan"
                )
            dykstra_status = tt5l.get("dykstra_feasibility_status")
            if not isinstance(dykstra_status, dict):
                failures.append(
                    "l5_v2_frontier_readiness:tt5l_dykstra_status_missing_or_not_object"
                )
            else:
                if dykstra_status.get("schema") != (
                    "l5_v2_tt5l_dykstra_feasibility_status_v1"
                ):
                    failures.append(
                        "l5_v2_frontier_readiness:tt5l_dykstra_status_bad_schema"
                    )
                if dykstra_status.get("artifact_valid") is not dykstra_valid:
                    failures.append(
                        "l5_v2_frontier_readiness:"
                        "tt5l_dykstra_status_validity_mismatch"
                    )
                for flag in (
                    "score_claim",
                    "promotion_eligible",
                    "ready_for_exact_eval_dispatch",
                ):
                    if dykstra_status.get(flag) is not False:
                        failures.append(
                            "l5_v2_frontier_readiness:"
                            f"tt5l_dykstra_status:{flag}_not_false"
                        )
            next_action = tt5l.get("next_non_pr106_l5_action")
            if not isinstance(next_action, dict):
                failures.append(
                    "l5_v2_frontier_readiness:tt5l_next_action_missing_or_not_object"
                )
            else:
                for flag in (
                    "score_claim",
                    "promotion_eligible",
                    "ready_for_exact_eval_dispatch",
                ):
                    if next_action.get(flag) is not False:
                        failures.append(
                            "l5_v2_frontier_readiness:"
                            f"tt5l_next_action:{flag}_not_false"
                        )
                action_id = str(next_action.get("action_id") or "")
                if "PR106" in action_id or "pr106" in action_id:
                    failures.append(
                        "l5_v2_frontier_readiness:tt5l_next_action_mentions_pr106"
                    )
                if not dykstra_valid and action_id != "run_tt5l_dykstra_score_axis_sanity":
                    failures.append(
                        "l5_v2_frontier_readiness:"
                        "tt5l_missing_dykstra_not_first_action"
                    )
                if dykstra_valid and not sideinfo_valid and action_id != (
                    "materialize_tt5l_contest_full_frame_sideinfo_consumption_proof"
                ):
                    failures.append(
                        "l5_v2_frontier_readiness:"
                        "tt5l_missing_sideinfo_not_next_action"
                    )
        if l5.get("packetir_matrix_dispatch_targets_suppressed") is True and target_count:
            failures.append(
                "l5_v2_frontier_readiness:"
                "packetir_matrix_targets_not_suppressed_after_blocker"
            )
        if target_count and l5.get("target_rows_are_fail_fast_only") is not True:
            failures.append("l5_v2_frontier_readiness:target_rows_not_fail_fast_only")
        targets_all = l5.get("next_exact_eval_targets")
        targets_sample = l5.get("next_exact_eval_targets_sample")
        if target_count and isinstance(targets_all, list):
            target_rows = targets_all
        elif target_count and isinstance(targets_sample, list):
            target_rows = targets_sample
        else:
            target_rows = []
        if target_count and not target_rows:
            failures.append("l5_v2_frontier_readiness:targets_missing")
        if isinstance(targets_sample, list) and len(targets_sample) > target_count:
            failures.append("l5_v2_frontier_readiness:sample_larger_than_target_count")
        if target_count and len(target_rows) != target_count:
            failures.append(
                "l5_v2_frontier_readiness:"
                f"target_row_count_mismatch:{len(target_rows)}!={target_count}"
            )
        if target_rows:
            for idx, target in enumerate(target_rows):
                if not isinstance(target, dict):
                    failures.append(
                        f"l5_v2_frontier_readiness:target_{idx}:not_object"
                    )
                    continue
                for flag in (
                    "score_claim",
                    "promotion_eligible",
                    "ready_for_exact_eval_dispatch",
                ):
                    if target.get(flag) is not False:
                        failures.append(
                            f"l5_v2_frontier_readiness:target_{idx}:{flag}_not_false"
                        )
                if (
                    target_count
                    and target.get("dispatch_status")
                    != "requires_claim_lane_dispatch_before_provider_launch"
                ):
                    failures.append(
                        "l5_v2_frontier_readiness:"
                        f"target_{idx}:dispatch_status_not_claim_gated"
                    )
                command_template = str(target.get("command_template") or "")
                for blocker in paired_auth_eval_dispatch_command_blockers(
                    paired_dispatch_tool=target.get("paired_dispatch_tool"),
                    command_template=command_template,
                ):
                    failures.append(
                        f"l5_v2_frontier_readiness:target_{idx}:{blocker}"
                    )
    groups = (
        ("supplementary_lanes", "active_supplementary_lanes"),
        ("gated_lanes", "active_gated_lanes"),
        ("composition_lanes", "active_composition_lanes"),
    )
    for all_key, active_key in groups:
        rows = payload.get(all_key)
        active_rows = payload.get(active_key)
        if not isinstance(rows, list):
            failures.append(f"{all_key}_missing_or_not_list")
            continue
        if not isinstance(active_rows, list):
            failures.append(f"{active_key}_missing_or_not_list")
            active_rows = []
        expected_active_ids: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                failures.append(f"{all_key}_contains_non_object_row")
                continue
            lane_id = str(row.get("lane_id") or "<missing>")
            dispatch = row.get("dispatch_routing")
            if not isinstance(dispatch, dict):
                failures.append(f"{all_key}:{lane_id}:dispatch_routing_missing")
                continue
            dispatch_active = dispatch.get("active") is True
            if row.get("ready_for_operator_dispatch") is not dispatch_active:
                failures.append(
                    f"{all_key}:{lane_id}:ready_for_operator_dispatch_mismatch"
                )
            if row.get("ready_for_exact_eval_dispatch") is True and not dispatch_active:
                failures.append(
                    f"{all_key}:{lane_id}:exact_eval_ready_without_operator_dispatch"
                )
            one_liner = row.get("one_liner")
            has_placeholder = isinstance(one_liner, str) and bool(
                _OPERATOR_PLACEHOLDER_RE.search(one_liner)
            )
            if has_placeholder and dispatch_active:
                failures.append(f"{all_key}:{lane_id}:active_with_operator_placeholder")
            if (
                row.get("gate_condition")
                and row.get("gate_ready") is not True
                and dispatch_active
            ):
                failures.append(f"{all_key}:{lane_id}:active_with_unsatisfied_gate")
            if dispatch.get("status") == "dispatch_gate_blocked" and not dispatch.get("blockers"):
                failures.append(f"{all_key}:{lane_id}:blocked_dispatch_without_blockers")
            if dispatch_active:
                expected_active_ids.add(lane_id)
        observed_active_ids: set[str] = set()
        for row in active_rows:
            if not isinstance(row, dict):
                failures.append(f"{active_key}_contains_non_object_row")
                continue
            lane_id = str(row.get("lane_id") or "<missing>")
            observed_active_ids.add(lane_id)
            dispatch = row.get("dispatch_routing")
            if not isinstance(dispatch, dict) or dispatch.get("active") is not True:
                failures.append(f"{active_key}:{lane_id}:active_row_not_dispatch_active")
            if row.get("ready_for_operator_dispatch") is not True:
                failures.append(f"{active_key}:{lane_id}:active_row_not_operator_ready")
            one_liner = row.get("one_liner")
            if isinstance(one_liner, str) and _OPERATOR_PLACEHOLDER_RE.search(one_liner):
                failures.append(f"{active_key}:{lane_id}:active_row_has_placeholder")
            if row.get("gate_condition") and row.get("gate_ready") is not True:
                failures.append(f"{active_key}:{lane_id}:active_row_has_unsatisfied_gate")
        if observed_active_ids != expected_active_ids:
            failures.append(
                f"{active_key}_does_not_match_dispatch_routing:"
                f"observed={sorted(observed_active_ids)} expected={sorted(expected_active_ids)}"
            )
    return failures


def _operator_briefing_xray_failures(payload: dict[str, object]) -> list[str]:
    failures: list[str] = []
    rows = payload.get("xray_tools")
    if not isinstance(rows, list):
        return ["xray_tools_missing_or_not_list"]

    observed: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            failures.append("xray_tools_contains_non_object_row")
            continue
        tool = str(row.get("tool") or "")
        observed.add(tool)
        if row.get("tool_exists") is not True:
            failures.append(f"xray_tools:{tool or '<missing>'}:tool_exists_not_true")
        for flag in (
            "score_claim",
            "score_claim_valid",
            "promotion_eligible",
            "rank_or_kill_eligible",
            "ready_for_exact_eval_dispatch",
        ):
            if row.get(flag) is not False:
                failures.append(f"xray_tools:{tool or '<missing>'}:{flag}_not_false")
        blockers = row.get("dispatch_blockers")
        if (
            not isinstance(blockers, list)
            or "diagnostic_tool_no_score_or_dispatch_authority" not in blockers
        ):
            failures.append(f"xray_tools:{tool or '<missing>'}:missing_dispatch_blocker")

    missing = sorted(_EXPECTED_XRAY_TOOLS - observed)
    extra = sorted(observed - _EXPECTED_XRAY_TOOLS)
    if missing:
        failures.append(f"xray_tools_missing_expected:{missing}")
    if extra:
        failures.append(f"xray_tools_unexpected:{extra}")
    return failures


_CLAIM_TABLE_KEYS = (
    "timestamp_utc",
    "agent",
    "lane_id",
    "platform",
    "instance_job_id",
    "predicted_eta_utc",
    "status",
    "notes",
)


def _claim_rows_from_markdown(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "---" in stripped or "timestamp_utc" in stripped:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < len(_CLAIM_TABLE_KEYS):
            continue
        rows.append(dict(zip(_CLAIM_TABLE_KEYS, cells[: len(_CLAIM_TABLE_KEYS)], strict=True)))
    return rows


def _terminal_claim_coverage_from_jsonl(evidence_path: Path) -> set[tuple[str, str, str]]:
    coverage: set[tuple[str, str, str]] = set()
    if not evidence_path.is_file():
        return coverage
    for line in evidence_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        lane_id = str(payload.get("lane_id") or "")
        job_id = str(payload.get("job_name") or payload.get("instance_job_id") or "")
        status = str(payload.get("dispatch_claim_latest_status") or "")
        if (
            lane_id
            and job_id
            and status
            and payload.get("exact_result_review_packet")
            and payload.get("score_claim") is False
            and payload.get("promotion_eligible") is False
            and payload.get("dispatch_claim_terminal_status_recorded") is True
        ):
            coverage.add((lane_id, job_id, status))
        claims = payload.get("covered_terminal_claims")
        if not isinstance(claims, list):
            continue
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            lane_id = str(claim.get("lane_id") or "")
            job_id = str(claim.get("instance_job_id") or "")
            status = str(claim.get("status") or "")
            if lane_id and job_id and status:
                coverage.add((lane_id, job_id, status))
    return coverage


def _is_terminal_dispatch_status(status: str) -> bool:
    return is_dispatch_claim_terminal_status(status)


def _terminal_substrate_claims_missing_evidence(
    claim_rows: list[dict[str, str]],
    evidence_coverage: set[tuple[str, str, str]],
    *,
    earliest_timestamp_utc: str = "2026-05-13T00:00:00Z",
    exact_eval_earliest_timestamp_utc: str = "2026-05-14T06:00:00Z",
) -> list[str]:
    missing: list[str] = []
    for row in claim_rows:
        timestamp = row.get("timestamp_utc", "")
        status = row.get("status", "")
        lane_id = row.get("lane_id", "")
        job_id = row.get("instance_job_id", "")
        if timestamp < earliest_timestamp_utc:
            continue
        if not _is_terminal_dispatch_status(status):
            continue
        exact_cuda_status = status.startswith("completed_contest_cuda_modal_auth_eval")
        if exact_cuda_status and timestamp < exact_eval_earliest_timestamp_utc:
            continue
        if not (
            "substrate_" in job_id
            or exact_cuda_status
            or lane_id.startswith("lane_substrate")
            or lane_id.startswith("lane_pr95_meta_stack")
            or lane_id.startswith("lane_time_traveler")
            or lane_id.startswith("lane_sabor")
            or lane_id.startswith("lane_s2sbs")
            or lane_id.startswith("lane_a1_plus")
        ):
            continue
        if (lane_id, job_id, status) in evidence_coverage:
            continue
        missing.append(f"{timestamp} lane_id={lane_id} job={job_id} status={status}")
    return missing


def _run_terminal_dispatch_evidence_gate() -> tuple[bool, str]:
    """Ensure recent terminal substrate dispatches are not lost outside evidence ledgers."""

    claim_rows = _claim_rows_from_markdown(REPO / ".omx/state/active_lane_dispatch_claims.md")
    evidence_coverage = _terminal_claim_coverage_from_jsonl(CATHEDRAL_AUTOPILOT_EVIDENCE)
    missing = _terminal_substrate_claims_missing_evidence(claim_rows, evidence_coverage)
    if missing:
        return (
            False,
            "terminal substrate dispatch claims missing cathedral evidence rows:\n  "
            + "\n  ".join(missing[:20])
            + "\nBackfill non-exact terminal rows with "
            "`PYTHONPATH=src:upstream:$PWD .venv/bin/python "
            "tools/backfill_terminal_claim_evidence.py`; exact CUDA rows must use "
            "`tools/build_result_review_packet.py`.",
        )
    return True, "terminal substrate dispatch evidence: PASS"


_HLM1_FRONTIER_DRIFT_PATTERNS = (
    re.compile(r"\bHLM1\b.*\b(current|active)\b.*\b(frontier|floor)\b", re.IGNORECASE),
    re.compile(r"\b(current|active)\b.*\b(frontier|floor)\b.*\bHLM1\b", re.IGNORECASE),
    re.compile(r"optimizer routing still uses lower HLM1", re.IGNORECASE),
)
_ACTIVE_FRONTIER_OR_FLOOR_RE = re.compile(
    r"\b(current|active)\b.*\b(frontier|floor)\b|\b(frontier|floor)\b.*\b(current|active)\b",
    re.IGNORECASE,
)
_HISTORICAL_LEDGER_MARKERS = (
    "historical artifact",
    "frozen historical",
    "archival",
    "superseded",
    "do not replay",
)


def _hlm1_frontier_prose_violations(root: Path = REPO) -> list[str]:
    allowed = ("non-promotional", "not the active", "not active", "demoted")
    violations: list[str] = []
    for path in sorted((root / ".omx/research").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        head = "\n".join(text.splitlines()[:10]).lower()
        if any(marker in head for marker in _HISTORICAL_LEDGER_MARKERS):
            continue
        paragraph_has_hlm1 = False
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                paragraph_has_hlm1 = False
                continue
            line_mentions_hlm1 = "HLM1" in line or "hlm1" in line
            paragraph_has_hlm1 = paragraph_has_hlm1 or line_mentions_hlm1
            if not paragraph_has_hlm1:
                continue
            lowered = line.lower()
            if any(token in lowered for token in allowed):
                continue
            # A paragraph can legitimately say "HLM1 is non-promotional; HDM4
            # is the active frontier." Do not flag that HDM4 handoff line.
            if "hdm4" in lowered and not line_mentions_hlm1:
                continue
            if any(pattern.search(line) for pattern in _HLM1_FRONTIER_DRIFT_PATTERNS) or (
                not line_mentions_hlm1 and _ACTIVE_FRONTIER_OR_FLOOR_RE.search(line)
            ):
                violations.append(f"{path.relative_to(root)}:{lineno}: {line.strip()[:180]}")
    return violations


def _run_hlm1_frontier_prose_gate() -> tuple[bool, str]:
    violations = _hlm1_frontier_prose_violations(REPO)
    if violations:
        return (
            False,
            "HLM1 is a non-promotional reference; stale active frontier/floor prose found:\n  "
            + "\n  ".join(violations[:20]),
        )
    return True, "HLM1 non-promotional frontier prose guard: PASS"


def _run_operator_briefing_dispatch_gate() -> tuple[bool, str]:
    """Ensure briefing active lists cannot bypass dispatch gate semantics."""

    proc = _run_subprocess(
        [
            sys.executable,
            str(OPERATOR_BRIEFING),
            "--json",
            "--top",
            "3",
            "--skip-provider-readiness",
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
        return False, f"operator briefing emitted invalid JSON: {exc}: {output[:500]}"
    failures = _operator_briefing_dispatch_failures(payload)
    failures.extend(_operator_briefing_xray_failures(payload))
    if failures:
        return False, "operator briefing dispatch routing failed: " + "; ".join(failures)
    counts = {
        "active_supplementary": len(payload.get("active_supplementary_lanes") or []),
        "active_gated": len(payload.get("active_gated_lanes") or []),
        "active_composition": len(payload.get("active_composition_lanes") or []),
    }
    return (
        True,
        "operator briefing dispatch routing: PASS "
        f"(active_supplementary={counts['active_supplementary']} "
        f"active_gated={counts['active_gated']} "
        f"active_composition={counts['active_composition']}; "
        "active rows require dispatch_routing.active=true and "
        "ready_for_operator_dispatch=true)",
    )


def _run_hidden_gems_gate() -> tuple[bool, str]:
    proc = _run_subprocess(
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
    proc = _run_subprocess(
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


def _run_semantic_label_contract_gate(source_index=None) -> tuple[bool, str]:
    from dataclasses import asdict

    from tools.audit_semantic_label_contract import audit_semantic_label_contract

    result = audit_semantic_label_contract(repo_root=REPO, source_index=source_index)
    payload = {
        "ok": result.ok,
        "contract_ok": result.contract_ok,
        "blocking_findings": [asdict(finding) for finding in result.blocking_findings],
        "advisory_findings": [asdict(finding) for finding in result.advisory_findings],
        "findings": [asdict(finding) for finding in result.findings],
    }
    output = json_text(payload)
    if payload["ok"] is not True:
        return False, output
    blocking = len(payload["blocking_findings"])
    advisory = len(payload["advisory_findings"])
    if blocking or advisory:
        return False, (
            "semantic-label audit must have zero blocking/advisory findings; "
            f"blocking={blocking} advisory={advisory}\n{output}"
        )
    return True, "semantic-label contract: PASS (canonical class order; 0 stale findings)"


def _run_engineered_corrections_gate() -> tuple[bool, str]:
    proc = _run_subprocess(
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
        proc = _run_subprocess(
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
    proc = _run_subprocess(
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
    proc = _run_subprocess(
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
    cmd = [sys.executable, str(HNERV_SCORECARD_AUDIT), "--scorecard", str(HNERV_SCORECARD)]
    for label, path in HNERV_SCORECARD_REQUIRED_EVALS:
        if not path.is_file():
            display_path = path.relative_to(REPO) if path.is_relative_to(REPO) else path
            return (
                False,
                "missing required HNeRV scorecard eval artifact: "
                f"{label}={display_path}",
            )
        cmd.extend(["--required-eval", f"{label}={path.relative_to(REPO)}"])
    proc = _run_subprocess(
        cmd,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def _run_tooling_consolidation_gate(source_index=None) -> tuple[bool, str]:
    from tac.audit_contract import audit_exit_code
    from tools.audit_tooling_consolidation import DEFAULT_SCAN_ROOTS, audit_tooling

    report = audit_tooling(REPO, DEFAULT_SCAN_ROOTS, source_index=source_index)
    payload = report.to_dict()
    counts = payload["summary"]["pattern_counts"]
    lines = [
        "tooling consolidation inventory: PASS "
        f"({payload['summary']['file_count']} files scanned)"
    ]
    for key, count in counts.items():
        lines.append(f"  - {key}: {count}")
    return audit_exit_code(report) == 0, "\n".join(lines)


def _run_recovered_remote_lanes_gate() -> tuple[bool, str]:
    proc = _run_subprocess(
        [sys.executable, str(RECOVERED_REMOTE_LANES_AUDIT), "--strict"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def _run_preserved_orphans_gate() -> tuple[bool, str]:
    proc = _run_subprocess(
        [sys.executable, str(PRESERVED_ORPHANS_AUDIT), "--format", "text"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def _run_recovery_custody_snapshots_gate() -> tuple[bool, str]:
    proc = _run_subprocess(
        [sys.executable, str(RECOVERY_CUSTODY_SNAPSHOTS_AUDIT), "--format", "text"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def _run_release_index_split_gate() -> tuple[bool, str]:
    proc = _run_subprocess(
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
    proc = _run_subprocess(
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
    proc = _run_subprocess(
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
    proc = _run_subprocess(
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
        proc = _run_subprocess(
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


def _eval_loader_drift_false_field_failures(
    row: dict[str, object], *, row_label: str
) -> list[str]:
    return [
        f"{row_label}: {field} must be false"
        for field in EVAL_LOADER_DRIFT_FALSE_CUSTODY_FIELDS
        if row.get(field) is not False
    ]


def _eval_loader_drift_axis_custody_failures(
    payload: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    device_axis_custody = payload.get("device_axis_custody")
    if not isinstance(device_axis_custody, dict):
        failures.append("device_axis_custody must be an object")
    else:
        if device_axis_custody.get("score_axis") != "diagnostic_loader_drift":
            failures.append(
                "device_axis_custody.score_axis must be diagnostic_loader_drift"
            )
        if device_axis_custody.get("claimed_score_axes") != []:
            failures.append("device_axis_custody.claimed_score_axes must stay empty")
        if device_axis_custody.get("score_claim_axis") != "none":
            failures.append("device_axis_custody.score_claim_axis must be none")
        for field in EVAL_LOADER_DRIFT_AXIS_FALSE_CUSTODY_FIELDS:
            if device_axis_custody.get(field) is not False:
                failures.append(f"device_axis_custody.{field}=false required")

    custody_labels = payload.get("custody_labels")
    if not isinstance(custody_labels, dict):
        failures.append("custody_labels must be an object")
    else:
        if custody_labels.get("score_path") != "not_run":
            failures.append("custody_labels.score_path must be not_run")
        if custody_labels.get("score_claim_axis") != "none":
            failures.append("custody_labels.score_claim_axis must be none")
        if custody_labels.get("diagnostic_non_promotable") is not True:
            failures.append("custody_labels.diagnostic_non_promotable must be true")
        for field in EVAL_LOADER_DRIFT_LABEL_FALSE_CUSTODY_FIELDS:
            if custody_labels.get(field) is not False:
                failures.append(f"custody_labels.{field}=false required")
        for field in ("mps_claim", "contest_cuda_claim", "contest_cpu_claim"):
            if field in custody_labels and custody_labels.get(field) is not False:
                failures.append(f"custody_labels.{field}=false required")
    return failures


def _validate_eval_loader_drift_2x2_plan(payload: dict[str, object]) -> list[str]:
    failures: list[str] = []
    cells_raw = payload.get("intended_cells")
    if not isinstance(cells_raw, list):
        failures.append("intended_cells must be a list")
    else:
        cell_ids: list[str] = []
        for index, row in enumerate(cells_raw):
            if not isinstance(row, dict):
                failures.append(f"intended_cells[{index}] must be an object")
                continue
            cell_id = str(row.get("cell_id") or "")
            cell_ids.append(cell_id)
            if not isinstance(row.get("available"), bool):
                failures.append(f"cell {cell_id or index}: available must be boolean")
            if not isinstance(row.get("unsupported_codes"), list):
                failures.append(f"cell {cell_id or index}: unsupported_codes must be a list")
            failures.extend(
                _eval_loader_drift_false_field_failures(row, row_label=f"cell {cell_id or index}")
            )
        if cell_ids != list(EVAL_LOADER_DRIFT_EXPECTED_CELL_IDS):
            failures.append(
                "intended_cells must list the CPU/CUDA loader 2x2 cells in order: "
                + ", ".join(EVAL_LOADER_DRIFT_EXPECTED_CELL_IDS)
            )

    plan_raw = payload.get("cell_discriminator_plan")
    if not isinstance(plan_raw, list):
        failures.append("cell_discriminator_plan must be a list")
    else:
        plan_ids: list[str] = []
        for index, row in enumerate(plan_raw):
            if not isinstance(row, dict):
                failures.append(f"cell_discriminator_plan[{index}] must be an object")
                continue
            comparison_id = str(row.get("comparison_id") or "")
            plan_ids.append(comparison_id)
            if not isinstance(row.get("available"), bool):
                failures.append(f"comparison {comparison_id or index}: available must be boolean")
            if not isinstance(row.get("unavailable_codes"), list):
                failures.append(f"comparison {comparison_id or index}: unavailable_codes must be a list")
            failures.extend(
                _eval_loader_drift_false_field_failures(
                    row, row_label=f"comparison {comparison_id or index}"
                )
            )
        if plan_ids != list(EVAL_LOADER_DRIFT_EXPECTED_PLAN_IDS):
            failures.append(
                "cell_discriminator_plan must list the five decoder/forward discriminator rows in order: "
                + ", ".join(EVAL_LOADER_DRIFT_EXPECTED_PLAN_IDS)
            )

    if not isinstance(payload.get("forward_matrix_complete"), bool):
        failures.append("forward_matrix_complete must be boolean")
    matrix = payload.get("forward_matrix_summary")
    if not isinstance(matrix, dict):
        failures.append("forward_matrix_summary must be an object")
    else:
        for field in ("requested", "complete"):
            if not isinstance(matrix.get(field), bool):
                failures.append(f"forward_matrix_summary.{field} must be boolean")
        if not isinstance(matrix.get("status"), str) or not matrix.get("status"):
            failures.append("forward_matrix_summary.status must be a nonempty string")
        for field in ("required_cell_ids", "unavailable_cell_ids"):
            if not isinstance(matrix.get(field), list):
                failures.append(f"forward_matrix_summary.{field} must be a list")
        if not isinstance(matrix.get("forward_row_count"), int) or isinstance(
            matrix.get("forward_row_count"), bool
        ):
            failures.append("forward_matrix_summary.forward_row_count must be an integer")
        failures.extend(
            _eval_loader_drift_false_field_failures(
                matrix, row_label="forward_matrix_summary"
            )
        )

    contract = payload.get("future_remote_run_contract")
    if not isinstance(contract, dict):
        failures.append("future_remote_run_contract must be an object")
    else:
        failures.extend(
            _eval_loader_drift_false_field_failures(contract, row_label="future_remote_run_contract")
        )
        if contract.get("requires_dispatch_claim_before_remote_gpu_run") is not True:
            failures.append(
                "future_remote_run_contract.requires_dispatch_claim_before_remote_gpu_run must be true"
            )
        command = contract.get("diagnostic_command")
        if not isinstance(command, list) or "--run-forward-cells" not in [str(item) for item in command]:
            failures.append(
                "future_remote_run_contract.diagnostic_command must include --run-forward-cells"
            )
        claim_template = contract.get("claim_command_template")
        if not isinstance(claim_template, list) or "claim" not in [str(item) for item in claim_template]:
            failures.append("future_remote_run_contract.claim_command_template must claim the lane")

    summary = payload.get("local_prerequisite_summary")
    if not isinstance(summary, dict):
        failures.append("local_prerequisite_summary must be an object")
    else:
        for key in (
            "cuda_available",
            "dali_available",
            "missing_cuda_dali_prerequisite_codes",
            "missing_cuda_dali_prerequisite_reasons",
        ):
            if key not in summary:
                failures.append(f"local_prerequisite_summary missing {key}")
        if not isinstance(summary.get("missing_cuda_dali_prerequisite_codes"), list):
            failures.append(
                "local_prerequisite_summary.missing_cuda_dali_prerequisite_codes must be a list"
            )

    return failures


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
        proc = _run_subprocess(
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
    for field in EVAL_LOADER_DRIFT_FALSE_CUSTODY_FIELDS:
        if payload.get(field) is not False:
            return False, f"eval loader drift probe must keep {field}=false"
    score_axis = payload.get("score_axis")
    if score_axis not in {None, "diagnostic_loader_drift"}:
        return False, f"eval loader drift probe must stay on diagnostic_loader_drift axis, got {score_axis!r}"
    plan_failures = _validate_eval_loader_drift_2x2_plan(payload)
    if plan_failures:
        return False, "eval loader drift 2x2 plan schema invalid: " + "; ".join(plan_failures)
    axis_custody_failures = _eval_loader_drift_axis_custody_failures(payload)
    if axis_custody_failures:
        return False, "eval loader drift axis custody invalid: " + "; ".join(
            axis_custody_failures
        )
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
    if tool == PR91_HPM1_READINESS_AUDIT:
        try:
            payload = _direct_pr91_hpm1_readiness_payload()
        except Exception as exc:
            return False, {}, f"{tool.name} direct run failed: {type(exc).__name__}: {exc}"
        return True, payload, json_text(payload)
    if tool == PR91_HPM1_RUNTIME_CONTRACT_AUDIT:
        try:
            payload = _direct_pr91_hpm1_runtime_contract_payload()
        except Exception as exc:
            return False, {}, f"{tool.name} direct run failed: {type(exc).__name__}: {exc}"
        return True, payload, json_text(payload)

    proc = _run_subprocess([sys.executable, str(tool)], capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return False, {}, output
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return False, {}, f"{tool.name} emitted invalid JSON: {exc}\n{output}"
    return True, payload, output


def _direct_pr91_hpm1_readiness_payload() -> dict[str, object]:
    """Build the same JSON payload as tools/audit_pr91_hpm1_readiness.py."""

    from tac.pr91_hpm1_codec import DEFAULT_PR91_ARCHIVE, DEFAULT_PR91_RUNTIME_SOURCE_DIR
    from tac.pr91_hpm1_readiness import audit_pr91_hpm1_readiness
    from tac.tool_manifest import attach_tool_run_manifest

    payload = audit_pr91_hpm1_readiness(
        archive=DEFAULT_PR91_ARCHIVE,
        runtime_source_dir=DEFAULT_PR91_RUNTIME_SOURCE_DIR,
        parity_report=None,
    )
    input_paths = [DEFAULT_PR91_ARCHIVE] if DEFAULT_PR91_ARCHIVE.is_file() else []
    return attach_tool_run_manifest(
        payload,
        tool=PR91_HPM1_READINESS_AUDIT.relative_to(REPO).as_posix(),
        argv=[],
        input_paths=input_paths,
        repo_root=REPO,
        output_path=None,
    )


def _direct_pr91_hpm1_runtime_contract_payload() -> dict[str, object]:
    """Build the same JSON payload as tools/audit_pr91_hpm1_runtime_contract.py."""

    from tac.pr91_hpm1_runtime_contract import (
        DEFAULT_PR91_RELEASE_RUNTIME_SOURCE_DIR,
        audit_pr91_hpm1_runtime_contract,
    )
    from tac.tool_manifest import attach_tool_run_manifest

    payload = audit_pr91_hpm1_runtime_contract(
        source_dir=DEFAULT_PR91_RELEASE_RUNTIME_SOURCE_DIR
    )
    input_paths = [
        path
        for path in (
            DEFAULT_PR91_RELEASE_RUNTIME_SOURCE_DIR / "inflate.py",
            DEFAULT_PR91_RELEASE_RUNTIME_SOURCE_DIR / "pr86_hpac.py",
        )
        if path.is_file()
    ]
    return attach_tool_run_manifest(
        payload,
        tool=PR91_HPM1_RUNTIME_CONTRACT_AUDIT.relative_to(REPO).as_posix(),
        argv=[],
        input_paths=input_paths,
        repo_root=REPO,
        output_path=None,
    )


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


def _run_phase_a_post_green_discoverability_gate() -> tuple[bool, str]:
    failures: list[str] = []
    for label, path in (
        ("A5 packet readiness tool", A5_PACKET_READINESS_TOOL),
        ("A6 measured-negative anchor", A6_BLOCKFP_HYPERPRIOR_ANCHOR),
        ("Modal A1 dispatcher", MODAL_A1_SCORE_GRADIENT_DISPATCHER),
        ("Lightning A1 dispatcher", LIGHTNING_A1_SCORE_GRADIENT_DISPATCHER),
    ):
        if not path.is_file():
            failures.append(f"missing {label}: {path.relative_to(REPO)}")

    sources: dict[str, str] = {}
    for path in (
        A5_PACKET_READINESS_TOOL,
        A6_BLOCKFP_HYPERPRIOR_ANCHOR,
        MODAL_IMAGE_BUILD_ORDER_AUDIT,
        MODAL_A1_SCORE_GRADIENT_DISPATCHER,
    ):
        if path.is_file():
            sources[path.name] = path.read_text(encoding="utf-8")

    a5_source = sources.get(A5_PACKET_READINESS_TOOL.name, "")
    if "build_packet_readiness(" not in a5_source or "--fail-if-not-ready" not in a5_source:
        failures.append("A5 packet-readiness CLI is not discoverable as a fail-if-not-ready gate")

    a6_source = sources.get(A6_BLOCKFP_HYPERPRIOR_ANCHOR.name, "")
    for needle in (
        '"promotion_eligible": False',
        '"rank_or_kill_eligible": False',
        '"family_falsified": False',
        '"score_affecting_payload_changed": False',
        '"charged_bits_changed": False',
    ):
        if needle not in a6_source:
            failures.append(f"A6 anchor missing fail-closed measured-negative field {needle}")

    modal_source = sources.get(MODAL_A1_SCORE_GRADIENT_DISPATCHER.name, "")
    for needle in (
        "plan_cli(",
        '"dispatch_attempted": False',
        '"remote_or_gpu_eval_started": False',
        "_close_modal_recovery_claim(",
        "completed_modal_contest_cuda_recovered",
        "failed_modal_recovered",
        "failed_modal_result_cache_expired",
    ):
        if needle not in modal_source:
            failures.append(f"Modal A1 dispatcher missing no-dispatch/recovery guard {needle}")
    if modal_source:
        claim_idx = modal_source.find("claim_rc = _claim_lane(")
        spawn_idx = modal_source.find("call = run_phase_a1_t4.spawn(")
        if claim_idx < 0 or spawn_idx < 0 or claim_idx > spawn_idx:
            failures.append("Modal A1 dispatcher must claim the lane before spawn")

    if failures:
        return False, "Phase A post-green discoverability failed: " + "; ".join(failures)
    return True, (
        "Phase A post-green discoverability: PASS "
        "(A5 readiness CLI, A6 measured-negative manifest, Modal A1 plan/recover, "
        "and A1 dispatcher surfaces are visible and fail-closed)"
    )


def _pr106_sidecar_runtime_consumption_failures(
    label: str,
    manifest: dict[str, object],
    *,
    expected_format_id: str,
    expected_archive_sha256: str,
    expected_runtime_source_tree_sha256: str,
) -> list[str]:
    failures: list[str] = []
    expected_fields = {
        "schema": "pr106_sidecar_runtime_decode_consumption_proof_v1",
        "format_id": expected_format_id,
        "payload_sha256_changed": True,
        "inner_pr106_payload_sha256_unchanged": True,
        "sidecar_payload_sha256_changed": True,
        "runtime_semantic_digest_changed": True,
        "runtime_corrected_latents_digest_changed": True,
        "runtime_all_score_affecting_sections_consumed": True,
        "runtime_sidecar_decode_consumption_claim": True,
        "runtime_sidecar_apply_consumption_claim": True,
        "full_frame_inflate_output_parity_claim": False,
        "contest_axis_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": [],
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            failures.append(
                f"{label}:{field}_drift expected {expected!r}, got {manifest.get(field)!r}"
            )

    archive = manifest.get("archive")
    if not isinstance(archive, dict):
        failures.append(f"{label}:archive_manifest_missing")
    else:
        if archive.get("sha256") != expected_archive_sha256:
            failures.append(
                f"{label}:archive_sha256_drift expected {expected_archive_sha256!r}, "
                f"got {archive.get('sha256')!r}"
            )
        if archive.get("expected_sha256") != expected_archive_sha256:
            failures.append(f"{label}:expected_archive_sha256_not_threaded")
        if archive.get("expected_sha256_matches") is not True:
            failures.append(f"{label}:expected_archive_sha256_mismatch")

    runtime = manifest.get("runtime_source_manifest")
    if not isinstance(runtime, dict):
        failures.append(f"{label}:runtime_source_manifest_missing")
    else:
        if runtime.get("runtime_source_tree_sha256") != expected_runtime_source_tree_sha256:
            failures.append(
                f"{label}:runtime_source_tree_sha256_drift expected "
                f"{expected_runtime_source_tree_sha256!r}, "
                f"got {runtime.get('runtime_source_tree_sha256')!r}"
            )
        if runtime.get("expected_runtime_source_tree_sha256") != expected_runtime_source_tree_sha256:
            failures.append(f"{label}:expected_runtime_source_tree_sha256_not_threaded")
        if runtime.get("expected_runtime_source_tree_sha256_matches") is not True:
            failures.append(f"{label}:expected_runtime_source_tree_sha256_mismatch")

    for digest_field in (
        "source_runtime_correction_digest",
        "mutated_runtime_correction_digest",
    ):
        digest = manifest.get(digest_field)
        if not isinstance(digest, dict):
            failures.append(f"{label}:{digest_field}_missing")
            continue
        if digest.get("format_id") != expected_format_id:
            failures.append(
                f"{label}:{digest_field}_format_id_drift expected {expected_format_id!r}, "
                f"got {digest.get('format_id')!r}"
            )
        if digest.get("n_pairs") != 600:
            failures.append(f"{label}:{digest_field}_n_pairs_drift")
        if digest.get("latents_changed_by_sidecar") is not True:
            failures.append(f"{label}:{digest_field}_latents_not_changed_by_sidecar")

    source_digest = manifest.get("source_runtime_correction_digest")
    mutated_digest = manifest.get("mutated_runtime_correction_digest")
    if isinstance(source_digest, dict) and isinstance(mutated_digest, dict):
        if source_digest.get("source_latents_sha256") != mutated_digest.get("source_latents_sha256"):
            failures.append(f"{label}:source_latents_changed_under_sidecar_mutation")
        if source_digest.get("corrected_latents_sha256") == mutated_digest.get(
            "corrected_latents_sha256"
        ):
            failures.append(f"{label}:corrected_latents_unchanged_under_sidecar_mutation")
        if source_digest.get("combined_sha256") == mutated_digest.get("combined_sha256"):
            failures.append(f"{label}:combined_digest_unchanged_under_sidecar_mutation")

    sections = manifest.get("runtime_consumed_score_affecting_sections")
    if not isinstance(sections, dict):
        failures.append(f"{label}:runtime_consumed_score_affecting_sections_missing")
    else:
        if sections.get("pr106_payload") is not True:
            failures.append(f"{label}:pr106_payload_runtime_consumption_not_proven")
        if sections.get("sidecar_payload") is not True:
            failures.append(f"{label}:sidecar_payload_runtime_consumption_not_proven")
        expected_framing = True if expected_format_id == "0x02" else None
        if sections.get("framing_meta") is not expected_framing:
            failures.append(
                f"{label}:framing_meta_runtime_consumption_drift "
                f"expected {expected_framing!r}, got {sections.get('framing_meta')!r}"
            )
    return failures


def _pr106_sidecar_packet_ir_identity_failures(
    label: str,
    archive_path: Path,
    *,
    expected_format_id: str,
    expected_archive_sha256: str,
) -> list[str]:
    from tac.packet_compiler import prove_pr106_sidecar_packet_ir_identity

    failures: list[str] = []
    manifest = prove_pr106_sidecar_packet_ir_identity(
        archive_path=archive_path,
        expected_archive_sha256=expected_archive_sha256,
    )
    packet = manifest.get("packet")
    if not isinstance(packet, dict):
        failures.append(f"{label}:packet_ir_manifest_missing_packet")
        return failures

    if packet.get("format_id") != expected_format_id:
        failures.append(
            f"{label}:packet_ir_format_id_drift expected {expected_format_id!r}, "
            f"got {packet.get('format_id')!r}"
        )
    emitted_payload = manifest.get("emitted_payload")
    if not isinstance(emitted_payload, dict):
        failures.append(f"{label}:packet_ir_emitted_payload_missing")
        return failures
    emitted_archive = manifest.get("emitted_archive")
    if not isinstance(emitted_archive, dict):
        failures.append(f"{label}:packet_ir_emitted_archive_missing")
        return failures
    if emitted_payload.get("byte_identical_to_source_member") is not True:
        failures.append(f"{label}:packet_ir_emit_payload_not_identity")
    if emitted_archive.get("byte_identical_to_source_archive") is not True:
        failures.append(f"{label}:stored_zip_reemit_not_identity")
    for field in (
        "runtime_consumption_claim",
        "full_frame_inflate_output_parity_claim",
        "contest_axis_claim",
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
    ):
        if manifest.get(field) is not False:
            failures.append(f"{label}:packet_ir_manifest_{field}_drift")
    if manifest.get("packet_ir_identity_passed") is not True:
        failures.append(f"{label}:packet_ir_identity_not_passed")
    archive = manifest.get("archive")
    if not isinstance(archive, dict):
        failures.append(f"{label}:packet_ir_archive_manifest_missing")
    else:
        if archive.get("sha256") != expected_archive_sha256:
            failures.append(
                f"{label}:packet_ir_archive_sha256_drift expected {expected_archive_sha256!r}, "
                f"got {archive.get('sha256')!r}"
            )
        if archive.get("expected_sha256") != expected_archive_sha256:
            failures.append(f"{label}:packet_ir_expected_archive_sha256_not_threaded")
        if archive.get("expected_sha256_matches") is not True:
            failures.append(f"{label}:packet_ir_expected_archive_sha256_mismatch")

    proof = packet.get("packet_ir_consumed_byte_proof")
    if not isinstance(proof, dict):
        failures.append(f"{label}:packet_ir_consumed_byte_proof_missing")
        return failures

    expected_pr106_section = (
        "pr106_hdm9_hlm3_payload_without_inner_header_or_section_magic"
        if expected_format_id == "0x0C"
        else "pr106_payload"
    )
    expected_score_sections = [expected_pr106_section, "sidecar_payload"]
    if expected_format_id == "0x02":
        expected_score_sections.append("framing_meta")
    expected_payload_bytes = emitted_payload.get("bytes")
    expected_payload_sha256 = emitted_payload.get("sha256")
    expected_proof_fields = {
        "runtime_consumption_claim": False,
        "all_payload_bytes_accounted": True,
        "unconsumed_trailing_bytes": 0,
        "section_gaps": [],
        "score_affecting_section_names": expected_score_sections,
        "emitted_payload_bytes": expected_payload_bytes,
        "emitted_payload_sha256": expected_payload_sha256,
        "accounted_payload_bytes": expected_payload_bytes,
    }
    for field, expected in expected_proof_fields.items():
        if proof.get(field) != expected:
            failures.append(
                f"{label}:packet_ir_proof_{field}_drift "
                f"expected {expected!r}, got {proof.get(field)!r}"
            )
    return failures


def _pr106_same_runtime_full_frame_parity_status() -> tuple[list[str], str]:
    path = PR106_R2_SAME_RUNTIME_FULL_FRAME_PARITY
    if not path.exists():
        return [], "same-runtime full-frame parity manifest absent (optional local artifact)"

    try:
        manifest = json.loads(path.read_text())
    except Exception as exc:
        return [f"same_runtime_full_frame_parity_manifest_unreadable:{type(exc).__name__}"], (
            "same-runtime full-frame parity manifest unreadable"
        )

    failures: list[str] = []
    expected_fields = {
        "schema": "pr106_same_runtime_streaming_frame_parity_v1",
        "proof_scope": "same_runtime_streaming_full_frame_hash",
        "streaming_output_sha256_equal": True,
        "streaming_output_total_bytes_equal": True,
        "full_frame_inflate_output_parity_claim": True,
        "prefix_parity_claim": False,
        "device_axis_label": "local-cpu-streaming-runtime",
        "contest_axis_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            failures.append(
                f"same_runtime_full_frame:{field}_drift "
                f"expected {expected!r}, got {manifest.get(field)!r}"
            )

    source = manifest.get("source")
    candidate = manifest.get("candidate")
    if not isinstance(source, dict) or not isinstance(candidate, dict):
        failures.append("same_runtime_full_frame:source_or_candidate_missing")
    else:
        for field in ("n_pairs_hashed", "total_frames", "total_bytes", "streaming_raw_sha256"):
            if source.get(field) != candidate.get(field):
                failures.append(f"same_runtime_full_frame:{field}_mismatch")
        if source.get("n_pairs_hashed") != 600 or candidate.get("n_pairs_hashed") != 600:
            failures.append("same_runtime_full_frame:n_pairs_hashed_not_600")
        if source.get("total_frames") != 1200 or candidate.get("total_frames") != 1200:
            failures.append("same_runtime_full_frame:total_frames_not_1200")
        if not source.get("streaming_raw_sha256"):
            failures.append("same_runtime_full_frame:streaming_raw_sha256_missing")

    note = (
        "same-runtime full-frame parity manifest present "
        "[local-cpu-streaming-runtime; contest_axis_claim=false; score_claim=false]"
    )
    return failures, note


def _pr106_hlm_runtime_consumption_failures(
    label: str,
    manifest: dict[str, object],
    *,
    expected_codec: str,
    expected_archive_sha256: str,
) -> list[str]:
    failures: list[str] = []
    expected_fields = {
        "schema": "pr106_hlm_runtime_consumption_proof_v1",
        "proof_scope": "runtime_codec_hlm_fixed_latent_decode_not_full_frame",
        "latent_section_codec": expected_codec,
        "archive_sha256": expected_archive_sha256,
        "runtime_hlm_decode_matches_canonical": True,
        "runtime_hlm_valid_mutation_changes_raw": True,
        "runtime_hlm_decode_consumption_claim": True,
        "full_frame_inflate_output_parity_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": [],
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            failures.append(
                f"{label}:hlm_runtime_{field}_drift "
                f"expected {expected!r}, got {manifest.get(field)!r}"
            )
    if not isinstance(manifest.get("latent_section_bytes"), int):
        failures.append(f"{label}:hlm_runtime_latent_section_bytes_missing")
    if not isinstance(manifest.get("latent_section_sha256"), str):
        failures.append(f"{label}:hlm_runtime_latent_section_sha256_missing")
    mutation = manifest.get("runtime_hlm_valid_mutation")
    if not isinstance(mutation, dict) or not mutation.get("mutation_kind"):
        failures.append(f"{label}:hlm_runtime_valid_mutation_manifest_missing")
    return failures


def _run_pr106_sidecar_runtime_consumption_gate() -> tuple[bool, str]:
    from tac.packet_compiler import prove_pr106_sidecar_runtime_decode_consumption
    from tac.packet_compiler.pr106_hlm1_runtime_consumption import (
        prove_pr106_hlm_runtime_consumption,
    )

    failures: list[str] = []
    observed_formats: list[str] = []
    for (
        label,
        archive_path,
        runtime_dir,
        expected_format_id,
        expected_archive_sha256,
        expected_runtime_source_tree_sha256,
    ) in (
        (
            "r2_brotli",
            PR106_R2_ARCHIVE,
            PR106_R2_RUNTIME,
            "0x01",
            PR106_R2_ARCHIVE_SHA256,
            PR106_R2_RUNTIME_SOURCE_TREE_SHA256,
        ),
        (
            "r2_pr101_grammar",
            PR106_R2_PR101_ARCHIVE,
            PR106_R2_PR101_RUNTIME,
            "0x02",
            PR106_R2_PR101_ARCHIVE_SHA256,
            PR106_R2_PR101_RUNTIME_SOURCE_TREE_SHA256,
        ),
        (
            "r2_hlm1_xmember_pr101_grammar",
            PR106_R2_HLM1_XMEMBER_ARCHIVE,
            PR106_R2_PR101_RUNTIME,
            "0x02",
            PR106_R2_HLM1_XMEMBER_ARCHIVE_SHA256,
            PR106_R2_PR101_RUNTIME_SOURCE_TREE_SHA256,
        ),
        (
            "r2_hlm2_xmember_pr101_grammar",
            PR106_R2_HLM2_XMEMBER_ARCHIVE,
            PR106_R2_PR101_RUNTIME,
            "0x02",
            PR106_R2_HLM2_XMEMBER_ARCHIVE_SHA256,
            PR106_R2_PR101_RUNTIME_SOURCE_TREE_SHA256,
        ),
        (
            "r2_format0c_xmember_pr101_grammar",
            PR106_FORMAT0C_XMEMBER_ARCHIVE,
            PR106_R2_PR101_RUNTIME,
            "0x0C",
            PR106_FORMAT0C_XMEMBER_ARCHIVE_SHA256,
            PR106_R2_PR101_RUNTIME_SOURCE_TREE_SHA256,
        ),
    ):
        try:
            failures.extend(
                _pr106_sidecar_packet_ir_identity_failures(
                    label,
                    archive_path,
                    expected_format_id=expected_format_id,
                    expected_archive_sha256=expected_archive_sha256,
                )
            )
        except Exception as exc:
            failures.append(
                f"{label}: packet-ir proof raised {type(exc).__name__}: {exc}"
            )
        try:
            manifest = prove_pr106_sidecar_runtime_decode_consumption(
                archive_path=archive_path,
                runtime_dir=runtime_dir,
                expected_archive_sha256=expected_archive_sha256,
                expected_runtime_source_tree_sha256=expected_runtime_source_tree_sha256,
            )
        except Exception as exc:
            failures.append(f"{label}: proof raised {type(exc).__name__}: {exc}")
            continue
        observed_formats.append(str(manifest.get("format_id")))
        failures.extend(
            _pr106_sidecar_runtime_consumption_failures(
                label,
                manifest,
                expected_format_id=expected_format_id,
                expected_archive_sha256=expected_archive_sha256,
                expected_runtime_source_tree_sha256=expected_runtime_source_tree_sha256,
            )
        )

    try:
        hlm2_manifest = prove_pr106_hlm_runtime_consumption(
            archive_path=PR106_R2_HLM2_XMEMBER_ARCHIVE,
            runtime_dir=PR106_R2_PR101_RUNTIME,
            repo_root=REPO,
            allowed_codecs=("hlm2",),
        )
    except Exception as exc:
        failures.append(f"r2_hlm2_xmember_pr101_grammar: HLM2 proof raised {type(exc).__name__}: {exc}")
    else:
        failures.extend(
            _pr106_hlm_runtime_consumption_failures(
                "r2_hlm2_xmember_pr101_grammar",
                hlm2_manifest,
                expected_codec="hlm2",
                expected_archive_sha256=PR106_R2_HLM2_XMEMBER_ARCHIVE_SHA256,
            )
        )

    parity_failures, parity_note = _pr106_same_runtime_full_frame_parity_status()
    failures.extend(parity_failures)

    if failures:
        return False, "PR106 sidecar runtime-consumption proof failed: " + "; ".join(failures)
    return True, (
        "PR106 sidecar runtime-consumption proof: PASS "
        f"(format_ids={','.join(observed_formats)}; PacketIR identity parse-emit "
        "accounts for every payload byte; runtime decodes/applies sidecar bytes; "
        "expected archive/runtime SHA custody is enforced; "
        "HLM2 runtime codec consumes the fixed-latent section; "
        "runtime-consumption manifests intentionally remain non-promotable; "
        f"{parity_note}; "
        "score_claim=false; ready_for_exact_eval_dispatch=false)"
    )


def _cancelled_result(
    step: PreflightStep,
    *,
    elapsed_s: float = 0.0,
    reason: str = "wall-clock budget exhausted before this step started",
) -> PreflightResult:
    return PreflightResult(
        step=step,
        passed=False,
        output=f"CANCELLED: {reason}",
        elapsed_s=elapsed_s,
        status="cancelled",
    )


def _execute_step(
    step: PreflightStep,
    context: PreflightRunContext | None = None,
) -> PreflightResult:
    if context is not None:
        _THREAD_CONTEXT.context = context
    start = time.perf_counter()
    try:
        if context is not None:
            remaining = context.remaining_s()
            if context.cancel_event.is_set() or (remaining is not None and remaining <= 0):
                context.cancel_event.set()
                return _cancelled_result(step)
        passed, output = step.runner()
        status = (
            "passed"
            if passed
            else (
                "timeout"
                if "TIMEOUT: all-lanes preflight wall-clock budget exhausted" in output
                else "failed"
            )
        )
    except Exception as exc:  # pragma: no cover - defensive fail-closed wrapper.
        passed = False
        output = f"{step.section} #{step.number} raised {type(exc).__name__}: {exc}"
        status = "failed"
    finally:
        if context is not None:
            _THREAD_CONTEXT.context = None
    return PreflightResult(
        step=step,
        passed=passed,
        output=output,
        elapsed_s=time.perf_counter() - start,
        status=status,
    )


def _timing_row(result: PreflightResult) -> dict[str, object]:
    step = result.step
    return {
        "section": step.section,
        "number": step.number,
        "name": step.name,
        "passed": result.passed,
        "status": result.status,
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


def _all_lanes_preflight_timeout_seconds(
    *,
    timeout_s: float | None,
    allow_slow_preflight: bool,
) -> float | None:
    """Return the all-lanes wall-clock budget, or None for explicit slow runs."""
    if allow_slow_preflight:
        return None
    if timeout_s is None:
        timeout_s = DEFAULT_ALL_LANES_PREFLIGHT_TIMEOUT_S
    if timeout_s <= 0:
        raise ValueError("--timeout-s must be positive unless --allow-slow-preflight is set")
    return float(timeout_s)


def _hard_watchdog_enabled() -> bool:
    raw = os.environ.get("PACT_ALL_LANES_PREFLIGHT_HARD_WATCHDOG", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _hard_watchdog_grace_seconds() -> float:
    raw = os.environ.get("PACT_ALL_LANES_PREFLIGHT_HARD_WATCHDOG_GRACE_S", "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = DEFAULT_HARD_WATCHDOG_GRACE_S
    else:
        value = DEFAULT_HARD_WATCHDOG_GRACE_S
    return max(0.0, value)


def _format_hard_watchdog_message(*, timeout_s: float, grace_s: float) -> str:
    return (
        "FATAL: all-lanes preflight hard watchdog fired after "
        f"{timeout_s:.2f}s budget + {grace_s:.2f}s grace. "
        "A preflight gate did not return cooperatively; DO NOT DISPATCH from this run."
    )


def _start_hard_wall_clock_watchdog(
    timeout_s: float | None,
    *,
    exit_func: Callable[[int], object] = os._exit,
) -> threading.Timer | None:
    """Hard-exit if an in-process gate ignores the cooperative budget."""

    if timeout_s is None or not _hard_watchdog_enabled():
        return None
    grace_s = _hard_watchdog_grace_seconds()

    def _crash() -> None:
        sys.stderr.write(_format_hard_watchdog_message(timeout_s=timeout_s, grace_s=grace_s) + "\n")
        sys.stderr.flush()
        exit_func(124)

    timer = threading.Timer(float(timeout_s) + grace_s, _crash)
    timer.daemon = True
    timer.start()
    return timer


def _format_wall_clock_budget_failure(
    results: list[PreflightResult],
    *,
    wall_elapsed_s: float,
    timeout_s: float,
    max_hot_steps: int = 5,
) -> str:
    """Build the fail-closed DX message for slow all-lanes preflight runs."""
    profile = _build_timing_profile(
        results,
        max_workers=0,
        wall_elapsed_s=wall_elapsed_s,
    )
    hot_steps = [row for row in profile["hot_steps"] if isinstance(row, dict)]
    lines = [
        (
            "FATAL: all-lanes preflight exceeded "
            f"{timeout_s:.2f}s wall-clock DX budget "
            f"(wall={wall_elapsed_s:.2f}s)."
        ),
        "DO NOT DISPATCH from this run; optimize or split the slow gates first.",
        "Slowest recorded steps:",
    ]
    for row in hot_steps[:max_hot_steps]:
        lines.append(_format_timing_row(row))
    lines.append("Use --allow-slow-preflight only for intentional profiling/debug runs.")
    return "\n".join(lines)


def _default_jobs(step_count: int) -> int:
    cpu_count = os.cpu_count() or 2
    return max(1, min(step_count, cpu_count, 8))


def _sort_results(results: list[PreflightResult]) -> list[PreflightResult]:
    return sorted(results, key=lambda item: (item.step.section, item.step.number))


def _run_steps_with_budget(
    steps: list[PreflightStep],
    *,
    max_workers: int,
    wall_clock_budget_s: float | None,
    run_started: float,
) -> list[PreflightResult]:
    deadline_s = (
        None
        if wall_clock_budget_s is None
        else run_started + float(wall_clock_budget_s)
    )
    context = PreflightRunContext(
        started_s=run_started,
        deadline_s=deadline_s,
        cancel_event=threading.Event(),
    )
    if max_workers == 1:
        results: list[PreflightResult] = []
        for step in steps:
            if context.cancel_event.is_set():
                results.append(_cancelled_result(step))
                continue
            remaining = context.remaining_s()
            if remaining is not None and remaining <= 0:
                context.cancel_event.set()
                results.append(_cancelled_result(step))
                continue
            result = _execute_step(step, context)
            results.append(result)
            remaining = context.remaining_s()
            if remaining is not None and remaining <= 0:
                context.cancel_event.set()
        return _sort_results(results)

    results_by_step: dict[tuple[str, int], PreflightResult] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_execute_step, step, context): step
            for step in steps
        }
        pending = set(futures)
        while pending:
            remaining = context.remaining_s()
            if remaining is not None and remaining <= 0:
                context.cancel_event.set()
                break
            done, pending = concurrent.futures.wait(
                pending,
                timeout=remaining,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                context.cancel_event.set()
                break
            for future in done:
                step = futures[future]
                try:
                    result = future.result()
                except concurrent.futures.CancelledError:
                    result = _cancelled_result(step)
                results_by_step[(step.section, step.number)] = result
        if pending:
            context.cancel_event.set()
            for future in pending:
                future.cancel()
            concurrent.futures.wait(pending)
            for future in pending:
                step = futures[future]
                if (step.section, step.number) in results_by_step:
                    continue
                if future.cancelled():
                    result = _cancelled_result(step)
                else:
                    try:
                        result = future.result()
                    except concurrent.futures.CancelledError:
                        result = _cancelled_result(step)
                results_by_step[(step.section, step.number)] = result
    for step in steps:
        results_by_step.setdefault((step.section, step.number), _cancelled_result(step))
    return _sort_results(list(results_by_step.values()))


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
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_ALL_LANES_PREFLIGHT_TIMEOUT_S,
        help=(
            "Fail closed when total all-lanes preflight wall-clock exceeds this "
            "DX budget. Default: 30 seconds."
        ),
    )
    parser.add_argument(
        "--allow-slow-preflight",
        action="store_true",
        help=(
            "Disable the all-lanes wall-clock budget. Use only for deliberate "
            "profiling/debug runs, not normal dispatch readiness checks."
        ),
    )
    args = parser.parse_args(argv)
    try:
        wall_clock_budget_s = _all_lanes_preflight_timeout_seconds(
            timeout_s=args.timeout_s,
            allow_slow_preflight=args.allow_slow_preflight,
        )
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

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
        A2_PACKET_LADDER_CLOSURE_AUDIT,
        A5_PACKET_READINESS_TOOL,
        A6_BLOCKFP_HYPERPRIOR_ANCHOR,
        MODAL_A1_SCORE_GRADIENT_DISPATCHER,
        LIGHTNING_A1_SCORE_GRADIENT_DISPATCHER,
        HSTACK_VSTACK_PLAN,
        PR106_R2_ARCHIVE,
        PR106_R2_RUNTIME / "inflate.py",
        PR106_R2_PR101_ARCHIVE,
        PR106_R2_PR101_RUNTIME / "inflate.py",
        PR91_HPM1_READINESS_ARTIFACT,
        PR91_HPM1_RUNTIME_CONTRACT_ARTIFACT,
        LOCAL_CUSTODY_RELEASE_MANIFEST,
        OPERATOR_BRIEFING,
        CANONICAL_TASK_STATUS_AUDIT,
        TAC_TERMINOLOGY_AUDIT,
        *[lane["tool"] for lane in lanes],
    ]:
        if not tool.is_file():
            print(f"FATAL: missing sub-tool {tool.relative_to(REPO)}", file=sys.stderr)
            return 2

    source_index = SourceIndex(REPO)

    gate_steps = [
        PreflightStep(
            "GATE",
            0,
            "dispatch CLI/shell hazards",
            lambda source_index=source_index: _run_source_index_gate(
                lambda: _run_dispatch_cli_shell_hazards_gate(source_index=source_index)
            ),
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
            lambda source_index=source_index: _run_source_index_gate(
                lambda: _run_semantic_label_contract_gate(source_index=source_index)
            ),
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
            lambda source_index=source_index: _run_source_index_gate(
                lambda: _run_tooling_consolidation_gate(source_index=source_index)
            ),
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
        PreflightStep(
            "GATE",
            23,
            "A2 packet ladder closure",
            lambda: _run_gate(
                "A2 packet ladder closure",
                A2_PACKET_LADDER_CLOSURE_AUDIT,
                ["--strict"],
            ),
            "  ✓ Gate #23: A2 packet ladder closure — PASSED",
            "  ✗ Gate #23: A2 packet ladder closure — FAILED",
        ),
        PreflightStep(
            "GATE",
            24,
            "Phase A post-green custody discoverability",
            _run_phase_a_post_green_discoverability_gate,
            "  ✓ Gate #24: Phase A post-green custody discoverability — PASSED",
            "  ✗ Gate #24: Phase A post-green custody discoverability — FAILED",
        ),
        PreflightStep(
            "GATE",
            25,
            "Modal image build order",
            lambda: _run_gate("Modal image build order", MODAL_IMAGE_BUILD_ORDER_AUDIT),
            "  ✓ Gate #25: Modal image build order — PASSED",
            "  ✗ Gate #25: Modal image build order — FAILED",
        ),
        PreflightStep(
            "GATE",
            26,
            "PR106 sidecar runtime consumption",
            _run_pr106_sidecar_runtime_consumption_gate,
            "  ✓ Gate #26: PR106 sidecar runtime consumption — PASSED",
            "  ✗ Gate #26: PR106 sidecar runtime consumption — FAILED",
        ),
        PreflightStep(
            "GATE",
            27,
            "active dispatch claims closed",
            _run_active_dispatch_claims_gate,
            "  ✓ Gate #27: active dispatch claims closed — PASSED",
            "  ✗ Gate #27: active dispatch claims closed — FAILED",
        ),
        PreflightStep(
            "GATE",
            28,
            "operator briefing dispatch routing",
            _run_operator_briefing_dispatch_gate,
            "  ✓ Gate #28: operator briefing dispatch routing — PASSED",
            "  ✗ Gate #28: operator briefing dispatch routing — FAILED",
        ),
        PreflightStep(
            "GATE",
            29,
            "terminal dispatch evidence coverage",
            _run_terminal_dispatch_evidence_gate,
            "  ✓ Gate #29: terminal dispatch evidence coverage — PASSED",
            "  ✗ Gate #29: terminal dispatch evidence coverage — FAILED",
        ),
        PreflightStep(
            "GATE",
            30,
            "HLM1 non-promotional frontier prose",
            _run_hlm1_frontier_prose_gate,
            "  ✓ Gate #30: HLM1 non-promotional frontier prose — PASSED",
            "  ✗ Gate #30: HLM1 non-promotional frontier prose — FAILED",
        ),
        PreflightStep(
            "GATE",
            31,
            "canonical task-status state machine",
            lambda: _run_gate(
                "canonical task-status state machine",
                CANONICAL_TASK_STATUS_AUDIT,
            ),
            "  ✓ Gate #31: canonical task-status state machine — PASSED",
            "  ✗ Gate #31: canonical task-status state machine — FAILED",
        ),
        PreflightStep(
            "GATE",
            32,
            "TAC terminology canonicalization",
            lambda: _run_gate(
                "TAC terminology canonicalization",
                TAC_TERMINOLOGY_AUDIT,
                ["--strict"],
            ),
            "  ✓ Gate #32: TAC terminology canonicalization — PASSED",
            "  ✗ Gate #32: TAC terminology canonicalization — FAILED",
        ),
        PreflightStep(
            "GATE",
            33,
            "public submission PR audit automation",
            lambda: _run_gate(
                "public submission PR audit automation",
                PUBLIC_SUBMISSION_PR_AUDIT,
                ["--self-test", "--format", "json"],
            ),
            "  ✓ Gate #33: public submission PR audit automation — PASSED",
            "  ✗ Gate #33: public submission PR audit automation — FAILED",
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
    hard_watchdog = _start_hard_wall_clock_watchdog(wall_clock_budget_s)
    try:
        results = _run_steps_with_budget(
            steps,
            max_workers=max_workers,
            wall_clock_budget_s=wall_clock_budget_s,
            run_started=run_started,
        )
    finally:
        if hard_watchdog is not None:
            hard_watchdog.cancel()
    source_index.save_persistent_text_facts()
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
    if wall_clock_budget_s is not None and wall_elapsed_s > wall_clock_budget_s:
        print(
            _format_wall_clock_budget_failure(
                results,
                wall_elapsed_s=wall_elapsed_s,
                timeout_s=wall_clock_budget_s,
            )
        )
        return 124
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
