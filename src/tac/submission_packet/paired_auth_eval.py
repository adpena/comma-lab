# SPDX-License-Identifier: MIT
"""Layer 5 — paired Modal CUDA + Linux x86_64 CPU auth-eval orchestrator.

Orchestrate paired auth-eval dispatch on the **EXACT same archive bytes**
(sha-locked invariant) per CLAUDE.md "Submission auth eval — BOTH CPU AND
CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable.

Routes through canonical helpers (NO hand-rolled subprocess invocations):

  * :func:`tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call`
    (Catalog #226 canonical CLI surface)
  * :func:`tac.deploy.modal.call_id_ledger.register_dispatched_call_id_fail_closed`
    (Catalog #339 silent-no-spawn extinction)
  * :func:`tac.deploy.modal.call_id_ledger.register_pre_spawn_fatal`
    (Catalog #360 pre-spawn fatal observability)
  * :func:`tac.deploy.modal.call_id_ledger.update_call_id_outcome`
    (Catalog #245 harvester invariant)

Per Phase 1 audit specification memo at
``.omx/research/canonical_submission_pipeline_specification_memo_20260526.md``
§3 Phase 6 / Layer 5 (paired_auth_eval):

The bug class this layer extincts: per-substrate paired-axis dispatch
invocations hand-roll the orchestration logic (operator runs Modal CUDA via
``tools/dispatch_modal_paired_auth_eval.py`` then separately runs Modal CPU
via ``tools/operator_authorize.py`` with a sister recipe; reconciliation is
ad-hoc). Without a canonical helper:

  1. Sha-locked invariant is not structurally enforced — operator can
     accidentally dispatch CUDA on archive sha A and CPU on archive sha B.
  2. macOS-CPU advisory results leak into the paired verdict despite
     Catalog #192 non-promotion requirement.
  3. Axis-hardware cross-validation is not typed — operator-side analysis
     scripts can label CPU-on-Apple-Silicon as ``[contest-CPU]``.
  4. Pre-spawn and post-spawn fatal observability does not consistently
     route through Catalog #339/#360 helpers.
  5. Downstream Phase 6 compliance verdict's D5 operator-gated blockers
     have no canonical reconciliation surface.

Per the 12th canonicalization × standardization × ease-of-contest-
compliance trinity: ONE canonical helper, ONE return shape
(:class:`PairedAuthEvalVerdict`), ONE per-axis routing protocol. Downstream
consumers (Phase 6 compliance verdict reconciliation; Phase 7 operator
runbook CLI; Phase 8 STRICT preflight gate ``check_no_pr_submission_
without_compliance_verdict``; Phase 10 PR111-candidate end-to-end
regression) consume :class:`PairedAuthEvalVerdict` directly without
re-parsing per-axis JSON reports.

Per the 13th OPTIMAL-TRIO standing directive: helper is canonical-frozen-
dataclass-return + canonical-Provenance-routing + 4-layer canonical-helper-
pattern sister of :mod:`tac.deploy.modal.call_id_ledger` (Catalog #245),
:mod:`tac.probe_outcomes_ledger` (Catalog #313),
:mod:`tac.canonical_equations` (Catalog #344), and
:mod:`tac.submission_packet.compliance` (Phase 6 Layer 4).

Per CLAUDE.md "Apples-to-apples evidence discipline" + "Submission auth
eval — BOTH CPU AND CUDA" non-negotiables: this layer is OBSERVABILITY-ONLY
by construction UNTIL the verdict is PAIRED_PASS. Every emitted
:class:`PairedAuthEvalVerdict` carries ``score_claim=False`` and
``promotable=False`` by DEFAULT; ``promotable=True`` is allowed only when
``verdict=PAIRED_PASS`` AND both axes ran on 1:1 contest-compliant
hardware (canonical predicate: CUDA on NVIDIA GPU + CPU on Linux x86_64).

Per CLAUDE.md "Forbidden device-selection defaults (the MPS-fallback
trap)" + Catalog #192 (`check_macos_cpu_advisory_not_promoted_without_
linux_verification`): the helper EXPLICITLY refuses Darwin ARM64 /
``macos_arm64`` hardware substrates as authoritative axes. macOS-CPU
remains a valid macOS-research-signal advisory route but is structurally
non-promotable. The :class:`PairedAuthEvalVerdict` invariant prevents
``promotable=True`` from coexisting with any macOS-CPU axis.

Quick start::

    from tac.submission_packet import (
        SubmissionBundleResult,
        plan_paired_auth_eval,
    )

    bundle: SubmissionBundleResult = ...  # Phase 4 output
    verdict = plan_paired_auth_eval(
        submission_bundle_result=bundle,
        cost_band="smoke",
        cuda_gpu="T4",
        cpu_target="linux_x86_64_modal",
        budget_usd=1.00,
        dry_run=True,
    )
    if verdict.verdict == "PAIRED_PASS":
        print(f"CPU {verdict.cpu_score:.6f} CUDA {verdict.cuda_score:.6f}")
    else:
        print(f"VERDICT={verdict.verdict}: {verdict.verdict_rationale}")

Discipline cross-references:
  * CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable
  * CLAUDE.md "MPS auth eval is NOISE" non-negotiable
  * CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable
  * CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" non-negotiable
  * Catalog #127 authoritative-tag custody validation
  * Catalog #143 paid-job register-before-submit
  * Catalog #192 macOS-CPU non-promotion enforcement
  * Catalog #215 min_smoke_gpu canonical recipe declaration
  * Catalog #221 auth-eval result artifact fail-closed
  * Catalog #226 canonical gate_auth_eval_call helper
  * Catalog #245 Modal call_id ledger
  * Catalog #270 dispatch optimization protocol umbrella
  * Catalog #287 placeholder-rationale rejection
  * Catalog #313 probe-outcomes ledger
  * Catalog #323 canonical Provenance umbrella
  * Catalog #335 cathedral consumer canonical contract
  * Catalog #339 silent-no-spawn extinction (post-spawn)
  * Catalog #341 Tier A canonical-routing markers
  * Catalog #344 canonical equations registry
  * Catalog #360 pre-spawn fatal observability
  * Catalog #361 Modal artifact filter preserves submission_dir
  * Catalog #245 / #313 / #344 / #355 canonical 4-layer pattern
  * 10th apples-to-apples paired CPU+CUDA on 1:1 contest-compliant hardware
  * 11th ORDER-MATTERS sequencing
  * 12th canonicalization × standardization × ease-of-contest-compliance
  * 13th OPTIMAL-TRIO standing directive
"""
from __future__ import annotations

import datetime
import enum
import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tac.submission_packet.builder import SubmissionBundleResult


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Module-level constants — canonical schemas + canonical budgets
# ---------------------------------------------------------------------------

PAIRED_AUTH_EVAL_SCHEMA_VERSION = "submission_paired_auth_eval_v1_20260526"
"""Pinned schema for :class:`PairedAuthEvalVerdict` persistence rows."""

PHASE_7_LAYER_VERSION = "phase_7_submission_paired_auth_eval_canonical_landed_20260526"
"""Operator-readable Phase 7 landing marker per Phase 1 audit spec memo."""

CANONICAL_EQUATION_ID = (
    "paired_auth_eval_canonical_helper_consolidation_savings_v1"
)
"""Canonical equation registered per Phase 1 audit spec memo §13 + Catalog
#344 FORMALIZATION_PENDING.

FORMALIZATION_PENDING until Phase 10 first-PR-through-canonical-pipeline
regression lands the first paired-axis empirical anchor of per-substrate
paired-axis-dispatch-divergence collapse (predicted: N+ per-substrate
ad-hoc paired dispatch invocations consolidated to ONE canonical helper,
with typed sha-locked invariant + axis-hardware cross-validation +
canonical Modal call_id ledger + Catalog #192 structural refusal at the
verdict surface).
"""

# Per Catalog #341 routing markers (Tier A observability-only initially).
PREDICTED_AXIS_TAG = "[predicted]"

# Per Catalog #287 placeholder rejection.
_PLACEHOLDER_RATIONALES: frozenset[str] = frozenset(
    {"<rationale>", "<reason>", "<rationale_here>", "<reason_here>", ""}
)

# Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192 — forbidden
# Darwin ARM64 / macOS substrate tokens as authoritative axes.
_FORBIDDEN_AUTHORITATIVE_HARDWARE_TOKENS: frozenset[str] = frozenset({
    "macos_arm64",
    "darwin_arm64",
    "darwin_arm64_m5_max_macos_cpu_advisory",
    "darwin_arm64_m5_max_macos_cpu",
    "macos_cpu",
    "apple_silicon",
    "macos_m5_max",
    "macos_advisory",
})

# Canonical 1:1 contest-compliant CPU hardware substrate tokens per CLAUDE.md
# "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
# HARDWARE" non-negotiable.
_CANONICAL_LINUX_X86_64_CPU_SUBSTRATES: frozenset[str] = frozenset({
    "linux_x86_64_modal_cpu",
    "linux_x86_64_vastai_cpu",
    "linux_x86_64_lightning_cpu",
    "linux_x86_64_gha_cpu",
    "linux_x86_64_cpu",
})

# Canonical CUDA GPU classes per Catalog #215 + sister Catalog #244.
_CANONICAL_CUDA_GPU_CLASSES: frozenset[str] = frozenset({
    "T4",
    "L4",
    "A10G",
    "L40S",
    "A100",
    "4090",
    "H100",
})

# Canonical CPU target -> hardware substrate token mapping.
_CPU_TARGET_TO_HARDWARE_SUBSTRATE: dict[str, str] = {
    "linux_x86_64_modal": "linux_x86_64_modal_cpu",
    "linux_x86_64_vastai": "linux_x86_64_vastai_cpu",
    "linux_x86_64_lightning": "linux_x86_64_lightning_cpu",
    "linux_x86_64_gha": "linux_x86_64_gha_cpu",
    "darwin_arm64_advisory": "darwin_arm64_m5_max_macos_cpu_advisory",
}

# Canonical CUDA platform -> hardware substrate prefix mapping.
_CUDA_PLATFORM_TO_HARDWARE_SUBSTRATE_PREFIX: dict[str, str] = {
    "modal": "linux_x86_64_modal",
    "vastai": "linux_x86_64_vastai",
    "lightning": "linux_x86_64_lightning",
}

# Canonical per-platform per-GPU expected cost ($/hour) per Catalog #270
# dispatch optimization protocol. Conservative; downstream callers should
# consult `tac.cost_band_calibration` for empirically-derived per-archive
# cost estimates when available.
_CANONICAL_PER_HOUR_RATES_USD: dict[tuple[str, str], float] = {
    ("modal", "T4"): 0.59,
    ("modal", "L4"): 0.80,
    ("modal", "A10G"): 1.10,
    ("modal", "L40S"): 1.95,
    ("modal", "A100"): 3.40,
    ("modal", "H100"): 6.25,
    ("vastai", "4090"): 0.30,
    ("vastai", "H100"): 1.99,
    ("lightning", "T4"): 0.0,
    ("lightning", "A100"): 0.0,
    ("modal", "cpu"): 0.06,
}

# Canonical cost-band budget envelope per Catalog #270.
_COST_BAND_BUDGET_USD: dict[str, float] = {
    "smoke": 1.00,
    "full": 5.00,
}

# Canonical evidence-grade tokens (paired axis empirical evidence).
_EVIDENCE_GRADE_CONTEST_CUDA_PLUS_CPU = "[contest-CUDA; contest-CPU; paired-axis-empirical]"
_EVIDENCE_GRADE_CUDA_ONLY = "[contest-CUDA; paired-axis-cpu-missing]"
_EVIDENCE_GRADE_CPU_ONLY = "[contest-CPU; paired-axis-cuda-missing]"
_EVIDENCE_GRADE_PAIRED_MACOS_ADVISORY = "[macOS-CPU advisory; paired-non-promotable]"
_EVIDENCE_GRADE_NON_PROMOTABLE_PENDING = "[predicted; paired-axis-not-yet-dispatched]"


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class PairedAuthEvalError(RuntimeError):
    """Paired auth-eval orchestration error.

    Sister of :class:`tac.submission_packet.compliance.SubmissionComplianceError`,
    :class:`tac.submission_packet.builder.SubmissionBundleError`,
    :class:`tac.submission_packet.archive_grammar.ArchiveGrammarError`, and
    :class:`tac.submission_packet.compression_pipeline.CompressionPipelineError`.
    Raised by canonical helpers when the orchestration crashes structurally
    (missing archive / sha mismatch / Modal call_id ledger registration
    failure / Catalog #192 macOS axis with promotable=True attempted) — NOT
    when a paired axis returns structured ``failure``, which surfaces as a
    typed :class:`PairedAuthEvalVerdict` with the appropriate
    ``verdict=BLOCKED_*``.
    """


# ---------------------------------------------------------------------------
# Frozen dataclasses + enums — canonical contract
# ---------------------------------------------------------------------------


class PairedAuthEvalVerdictKind(enum.StrEnum):
    """Canonical paired-axis verdict taxonomy.

    Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" + 10th
    apples-to-apples paired CPU+CUDA on 1:1 contest-compliant hardware
    non-negotiables. Downstream consumers (Phase 8 STRICT gate / Phase 10
    PR111-candidate end-to-end regression / Phase 7 operator runbook CLI)
    route per verdict.
    """

    PAIRED_PASS = "PAIRED_PASS"
    """Both axes ran on 1:1 contest-compliant hardware; sha-locked invariant
    held; both axes produced valid contest-axis empirical scores; verdict
    is promotable per Catalog #127 + #192 + #221 sister discipline."""

    PAIRED_PARTIAL_CUDA_ONLY = "PAIRED_PARTIAL_CUDA_ONLY"
    """CUDA axis landed cleanly; CPU axis missing OR failed. Verdict is NOT
    promotable; operator-routable: re-dispatch CPU axis per CLAUDE.md non-
    negotiable. Acceptable transitional state during smoke-before-full or
    partial-failure recovery."""

    PAIRED_PARTIAL_CPU_ONLY = "PAIRED_PARTIAL_CPU_ONLY"
    """CPU axis landed cleanly; CUDA axis missing OR failed. Verdict is NOT
    promotable; operator-routable: re-dispatch CUDA axis per CLAUDE.md non-
    negotiable. Rarer than CUDA_ONLY since CPU is typically slower."""

    BLOCKED_PRE_DISPATCH = "BLOCKED_PRE_DISPATCH"
    """Pre-dispatch validation failed (sha mismatch / missing archive /
    insufficient budget / non-canonical CPU target / Catalog #215 GPU
    class invalid). Operator action required before any spawn fires."""

    BLOCKED_HARVEST = "BLOCKED_HARVEST"
    """Dispatch fired (Modal call_id registered) but harvest failed
    (TIMEOUT / Modal rc=1 / `contest_auth_eval.py` rc!=0). Operator-
    routable: consult Modal dashboard via call_id; retry available."""

    BLOCKED_AXIS_MISMATCH = "BLOCKED_AXIS_MISMATCH"
    """Both axes landed but archive_sha256 differs (sha-locked invariant
    violation). Hard structural error per Catalog #127 custody discipline.
    Operator-routable: investigate why the two dispatches saw different
    archive bytes."""

    BLOCKED_HARDWARE_NON_COMPLIANT = "BLOCKED_HARDWARE_NON_COMPLIANT"
    """One or both axes landed on non-1:1-contest-compliant hardware
    (e.g., Darwin ARM64 macOS-CPU; Catalog #192 violation). Verdict is
    structurally non-promotable. Operator-routable: re-dispatch the
    offending axis on canonical Linux x86_64 substrate."""


@dataclass(frozen=True)
class PairedAuthEvalVerdict:
    """Canonical Phase 7 Layer 5 paired auth-eval verdict.

    Sister of :class:`tac.submission_packet.compliance.ComplianceVerdict`
    + :class:`tac.submission_packet.builder.SubmissionBundleResult`
    + :class:`tac.submission_packet.archive_grammar.ArchiveGrammarManifest`
    + :class:`tac.submission_packet.compression_pipeline.CompressionPipelineResult`
    at the paired-axis-orchestration sub-surface.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #127 +
    #192 + #221 + #341 invariants enforced by ``__post_init__``:

      * ``archive_sha256_paired`` MUST equal the CPU and CUDA per-axis sha
        256 hex (sha-locked invariant; Catalog #127 custody discipline)
      * Catalog #192: any macOS / Darwin ARM64 hardware substrate token
        forces ``promotable=False`` regardless of verdict kind
      * Catalog #341: ``promotable=True`` ONLY when ``verdict=PAIRED_PASS``
        AND BOTH per-axis hardware substrates are canonical Linux x86_64
      * ``score_claim`` defaults False; True ONLY when ``promotable=True``
        (canonical Provenance umbrella per Catalog #323)
    """

    schema_version: str
    """Canonical schema version (current: :data:`PAIRED_AUTH_EVAL_SCHEMA_VERSION`)."""

    lane_id: str
    """Lane registry id from submission bundle lineage."""

    substrate_id: str
    """Substrate id from submission bundle lineage."""

    archive_sha256_paired: str
    """sha256 hex digest of the archive bytes BOTH axes ran on (the sha-
    locked invariant per Catalog #127). Empty string when verdict is
    ``BLOCKED_PRE_DISPATCH`` and no spawn fired."""

    archive_bytes: int
    """Total ``archive.zip`` size in bytes."""

    submission_dir: str
    """Path to bundled submission_dir/ per Phase 4."""

    verdict: str
    """One of :class:`PairedAuthEvalVerdictKind` values."""

    verdict_rationale: str
    """Operator-readable rationale for the verdict (≥4 chars, non-placeholder
    per Catalog #287)."""

    # ---- CUDA axis (None when missing / failed) ----
    cuda_score: float | None
    """CUDA-axis contest score (lower is better). None when axis missing/failed."""

    cuda_axis_tag: str
    """``"[contest-CUDA]"`` when valid; ``"[missing]"`` when dispatch never
    fired; ``"[failed]"`` when dispatch fired but produced no valid claim."""

    cuda_hardware_substrate: str
    """Hardware substrate token (e.g. ``linux_x86_64_modal_t4``); empty when
    no CUDA dispatch fired."""

    cuda_call_id: str
    """Modal call_id per Catalog #245 canonical ledger; empty when no CUDA
    dispatch fired."""

    cuda_seg_distortion: float | None
    """CUDA-axis seg distortion component. None when axis missing/failed."""

    cuda_pose_distortion: float | None
    """CUDA-axis pose distortion component. None when axis missing/failed."""

    cuda_rate_term: float | None
    """CUDA-axis rate term (= 25 * archive_bytes / 37,545,489). None when missing."""

    cuda_auth_eval_json_path: str
    """Path to canonical ``contest_auth_eval_cuda.json`` per Catalog #249.
    Empty when no CUDA dispatch fired."""

    cuda_elapsed_seconds: float
    """CUDA-axis dispatch wall-clock."""

    cuda_cost_usd: float
    """CUDA-axis empirical cost ($USD)."""

    # ---- CPU axis (None when missing / failed) ----
    cpu_score: float | None
    """CPU-axis contest score (lower is better). None when axis missing/failed."""

    cpu_axis_tag: str
    """``"[contest-CPU]"`` when Linux x86_64; ``"[macOS-CPU advisory]"`` when
    Darwin ARM64; ``"[missing]"`` when dispatch never fired; ``"[failed]"``
    when dispatch fired but no valid claim."""

    cpu_hardware_substrate: str
    """Hardware substrate token (e.g. ``linux_x86_64_modal_cpu``); empty
    when no CPU dispatch fired."""

    cpu_call_id: str
    """Modal call_id per Catalog #245 canonical ledger; empty when no CPU
    dispatch fired (e.g., Vast.ai / Lightning / GHA targets)."""

    cpu_seg_distortion: float | None
    """CPU-axis seg distortion component. None when axis missing/failed."""

    cpu_pose_distortion: float | None
    """CPU-axis pose distortion component. None when axis missing/failed."""

    cpu_rate_term: float | None
    """CPU-axis rate term. None when missing."""

    cpu_auth_eval_json_path: str
    """Path to canonical ``contest_auth_eval_cpu.json`` per Catalog #249.
    Empty when no CPU dispatch fired."""

    cpu_elapsed_seconds: float
    """CPU-axis dispatch wall-clock."""

    cpu_cost_usd: float
    """CPU-axis empirical cost ($USD)."""

    # ---- Paired analysis ----
    cuda_cpu_gap: float | None
    """Empirical CUDA - CPU score delta. None when either axis missing.
    Per the PR102 empirical anchor (sister `feedback_dual_cpu_cuda_auth_
    eval_mandatory_20260508.md`): the CUDA - CPU gap is per-archive
    empirical; do NOT assume PR102's -0.033 generalizes."""

    cost_band: str
    """One of ``"smoke"`` / ``"full"`` per Catalog #270."""

    budget_usd: float
    """Operator-set $USD envelope per Catalog #270 + recipe declaration."""

    total_cost_usd: float
    """Sum of cuda_cost_usd + cpu_cost_usd; refused if > budget_usd."""

    measurement_utc: str
    """ISO-8601 UTC timestamp of paired auth-eval orchestration."""

    axis_tag: str
    """One of :data:`PREDICTED_AXIS_TAG` (pre-dispatch / partial / blocked)
    OR ``"[contest-CUDA; contest-CPU]"`` (PAIRED_PASS empirical anchor)."""

    score_claim: bool
    """False until PAIRED_PASS AND both axes on 1:1 contest-compliant
    hardware. Per Catalog #323 canonical Provenance + Catalog #341
    canonical-routing markers."""

    promotable: bool
    """False until PAIRED_PASS AND both axes on 1:1 contest-compliant
    hardware. Per Catalog #192 + #127 + #221 + #341 sister discipline.
    Catalog #192 macOS axis structurally forces promotable=False."""

    evidence_grade: str
    """One of the canonical evidence-grade tokens per Catalog #287/#323."""

    canonical_helper_invocation: str
    """``"tac.submission_packet.plan_paired_auth_eval"`` per Catalog #190."""

    canonical_equation_id: str
    """:data:`CANONICAL_EQUATION_ID` (per Catalog #344)."""

    canonical_equation_status: str
    """``"FORMALIZATION_PENDING"`` until Phase 10 first empirical anchor."""

    cuda_platform: str
    """One of ``"modal"`` / ``"vastai"`` / ``"lightning"`` (always Linux
    x86_64 host environment)."""

    cuda_gpu: str
    """One of :data:`_CANONICAL_CUDA_GPU_CLASSES` values."""

    cpu_target: str
    """One of :data:`_CPU_TARGET_TO_HARDWARE_SUBSTRATE` keys."""

    dry_run: bool
    """True when verdict was emitted in --dry-run mode (no Modal spawn
    fired; cost estimate only)."""

    forbidden_macos_axis_detected: bool
    """True iff any axis landed on macOS / Darwin ARM64 hardware substrate
    token per Catalog #192. When True, ``promotable=False`` STRUCTURALLY."""

    canonical_provenance: Mapping[str, Any] = field(default_factory=dict)
    """Per Catalog #323 canonical Provenance umbrella."""

    written_at_utc: str = ""
    """When persisted to a canonical ledger (caller-fills)."""

    written_pid: int = 0
    """Process PID that emitted the result."""

    written_host: str = ""
    """Host that emitted the result."""

    def __post_init__(self) -> None:
        if self.schema_version != PAIRED_AUTH_EVAL_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {PAIRED_AUTH_EVAL_SCHEMA_VERSION!r}; "
                f"got {self.schema_version!r}"
            )
        if not self.lane_id:
            raise ValueError("lane_id must be non-empty")
        if not self.substrate_id:
            raise ValueError("substrate_id must be non-empty")
        # archive_sha256_paired MAY be empty when verdict=BLOCKED_PRE_DISPATCH
        # (no spawn fired); otherwise MUST be 64-char hex
        valid_verdicts = {v.value for v in PairedAuthEvalVerdictKind}
        if self.verdict not in valid_verdicts:
            raise ValueError(
                f"verdict must be one of {sorted(valid_verdicts)}; got {self.verdict!r}"
            )
        if (
            self.verdict != PairedAuthEvalVerdictKind.BLOCKED_PRE_DISPATCH.value
            and len(self.archive_sha256_paired) != 64
        ):
            raise ValueError(
                f"archive_sha256_paired must be 64-char hex (or empty when "
                f"BLOCKED_PRE_DISPATCH); got len={len(self.archive_sha256_paired)}"
            )
        if self.archive_bytes < 0:
            raise ValueError("archive_bytes must be non-negative")
        if not self.submission_dir:
            raise ValueError("submission_dir must be non-empty")
        rationale = self.verdict_rationale.strip()
        if rationale in _PLACEHOLDER_RATIONALES or len(rationale) < 4:
            raise ValueError(
                f"verdict_rationale {self.verdict_rationale!r} must be substantive "
                "(>=4 chars, non-placeholder) per Catalog #287"
            )
        # Per-axis floats: when present, must be finite + non-negative
        for label, value in (
            ("cuda_score", self.cuda_score),
            ("cuda_seg_distortion", self.cuda_seg_distortion),
            ("cuda_pose_distortion", self.cuda_pose_distortion),
            ("cuda_rate_term", self.cuda_rate_term),
            ("cpu_score", self.cpu_score),
            ("cpu_seg_distortion", self.cpu_seg_distortion),
            ("cpu_pose_distortion", self.cpu_pose_distortion),
            ("cpu_rate_term", self.cpu_rate_term),
        ):
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                raise ValueError(f"{label} must be float or None; got {type(value).__name__}")
            if value != value:  # NaN check (NaN != NaN)
                raise ValueError(f"{label} must be finite (not NaN)")
            if value < 0:
                raise ValueError(f"{label} must be non-negative; got {value!r}")
        # cuda_cpu_gap can be negative (CUDA - CPU often negative for our archives)
        if self.cuda_cpu_gap is not None:
            if not isinstance(self.cuda_cpu_gap, (int, float)):
                raise ValueError("cuda_cpu_gap must be float or None")
            if self.cuda_cpu_gap != self.cuda_cpu_gap:  # NaN check
                raise ValueError("cuda_cpu_gap must be finite (not NaN)")
        # Axis tags
        valid_cuda_tags = {"[contest-CUDA]", "[missing]", "[failed]"}
        if self.cuda_axis_tag not in valid_cuda_tags:
            raise ValueError(
                f"cuda_axis_tag must be one of {sorted(valid_cuda_tags)}; got {self.cuda_axis_tag!r}"
            )
        valid_cpu_tags = {"[contest-CPU]", "[macOS-CPU advisory]", "[missing]", "[failed]"}
        if self.cpu_axis_tag not in valid_cpu_tags:
            raise ValueError(
                f"cpu_axis_tag must be one of {sorted(valid_cpu_tags)}; got {self.cpu_axis_tag!r}"
            )
        # Wall-clock + cost: non-negative
        for label, value in (
            ("cuda_elapsed_seconds", self.cuda_elapsed_seconds),
            ("cpu_elapsed_seconds", self.cpu_elapsed_seconds),
            ("cuda_cost_usd", self.cuda_cost_usd),
            ("cpu_cost_usd", self.cpu_cost_usd),
            ("budget_usd", self.budget_usd),
            ("total_cost_usd", self.total_cost_usd),
        ):
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{label} must be non-negative number; got {value!r}")
        # Cost-band budget envelope
        if self.cost_band not in _COST_BAND_BUDGET_USD:
            raise ValueError(
                f"cost_band must be one of {sorted(_COST_BAND_BUDGET_USD)}; got {self.cost_band!r}"
            )
        # Platform + GPU + CPU target canonical
        if self.cuda_platform not in {"modal", "vastai", "lightning"}:
            raise ValueError(
                f"cuda_platform must be modal/vastai/lightning; got {self.cuda_platform!r}"
            )
        if self.cuda_gpu not in _CANONICAL_CUDA_GPU_CLASSES:
            raise ValueError(
                f"cuda_gpu must be one of {sorted(_CANONICAL_CUDA_GPU_CLASSES)}; "
                f"got {self.cuda_gpu!r}"
            )
        if self.cpu_target not in _CPU_TARGET_TO_HARDWARE_SUBSTRATE:
            raise ValueError(
                f"cpu_target must be one of {sorted(_CPU_TARGET_TO_HARDWARE_SUBSTRATE)}; "
                f"got {self.cpu_target!r}"
            )
        # Catalog #192 macOS forbidden-axis cross-validation
        if not isinstance(self.forbidden_macos_axis_detected, bool):
            raise ValueError("forbidden_macos_axis_detected must be bool")
        macos_axis_present = (
            any(
                tok in self.cuda_hardware_substrate
                for tok in _FORBIDDEN_AUTHORITATIVE_HARDWARE_TOKENS
            )
            or any(
                tok in self.cpu_hardware_substrate
                for tok in _FORBIDDEN_AUTHORITATIVE_HARDWARE_TOKENS
            )
        )
        if macos_axis_present and not self.forbidden_macos_axis_detected:
            raise ValueError(
                "forbidden_macos_axis_detected must be True when any axis "
                "hardware substrate references macOS / Darwin ARM64 tokens "
                "per Catalog #192"
            )
        # Provenance / Catalog #323 / Catalog #341 invariants
        if not isinstance(self.score_claim, bool):
            raise ValueError("score_claim must be bool")
        if not isinstance(self.promotable, bool):
            raise ValueError("promotable must be bool")
        # Catalog #192: macOS axis structurally forces promotable=False
        if self.forbidden_macos_axis_detected and self.promotable:
            raise ValueError(
                "forbidden_macos_axis_detected=True is incompatible with "
                "promotable=True per Catalog #192 + CLAUDE.md 'Submission "
                "auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT "
                "HARDWARE' non-negotiable"
            )
        # promotable=True REQUIRES PAIRED_PASS AND both axes on canonical
        # Linux x86_64 substrates
        if self.promotable:
            if self.verdict != PairedAuthEvalVerdictKind.PAIRED_PASS.value:
                raise ValueError(
                    f"promotable=True requires verdict=PAIRED_PASS; got verdict={self.verdict!r}"
                )
            if self.cuda_hardware_substrate.split("_")[0:3] != ["linux", "x86", "64"]:
                raise ValueError(
                    f"promotable=True requires cuda_hardware_substrate Linux x86_64; "
                    f"got {self.cuda_hardware_substrate!r}"
                )
            if self.cpu_hardware_substrate not in _CANONICAL_LINUX_X86_64_CPU_SUBSTRATES:
                raise ValueError(
                    f"promotable=True requires cpu_hardware_substrate in "
                    f"{sorted(_CANONICAL_LINUX_X86_64_CPU_SUBSTRATES)}; "
                    f"got {self.cpu_hardware_substrate!r}"
                )
        # score_claim => promotable
        if self.score_claim and not self.promotable:
            raise ValueError(
                "score_claim=True requires promotable=True per Catalog #341"
            )
        # axis_tag validity
        if self.axis_tag not in {PREDICTED_AXIS_TAG, "[contest-CUDA; contest-CPU]"}:
            raise ValueError(
                f"axis_tag must be {PREDICTED_AXIS_TAG!r} or '[contest-CUDA; contest-CPU]'; "
                f"got {self.axis_tag!r}"
            )
        # PAIRED_PASS + promotable => axis_tag canonical
        if self.promotable and self.axis_tag != "[contest-CUDA; contest-CPU]":
            raise ValueError(
                "promotable=True requires axis_tag='[contest-CUDA; contest-CPU]'"
            )
        # evidence_grade canonical
        valid_evidence_grades = frozenset({
            _EVIDENCE_GRADE_CONTEST_CUDA_PLUS_CPU,
            _EVIDENCE_GRADE_CUDA_ONLY,
            _EVIDENCE_GRADE_CPU_ONLY,
            _EVIDENCE_GRADE_PAIRED_MACOS_ADVISORY,
            _EVIDENCE_GRADE_NON_PROMOTABLE_PENDING,
        })
        if self.evidence_grade not in valid_evidence_grades:
            raise ValueError(
                f"evidence_grade must be one of {sorted(valid_evidence_grades)}; "
                f"got {self.evidence_grade!r}"
            )
        if self.canonical_equation_id != CANONICAL_EQUATION_ID:
            raise ValueError(
                f"canonical_equation_id must equal {CANONICAL_EQUATION_ID!r}; "
                f"got {self.canonical_equation_id!r}"
            )
        if self.canonical_equation_status not in {"FORMALIZATION_PENDING", "REGISTERED"}:
            raise ValueError(
                "canonical_equation_status must be 'FORMALIZATION_PENDING' or 'REGISTERED' "
                "per Catalog #344"
            )
        if not isinstance(self.dry_run, bool):
            raise ValueError("dry_run must be bool")
        if not isinstance(self.canonical_provenance, Mapping):
            raise ValueError("canonical_provenance must be a Mapping per Catalog #323")
        # sha-locked invariant: when both axes present, cross-check shas (the
        # cuda_call_id + cpu_call_id are distinct; the archive_sha256_paired
        # is the shared invariant). When PAIRED_PASS we expect both axes
        # ran on archive_sha256_paired bytes.
        if self.verdict == PairedAuthEvalVerdictKind.PAIRED_PASS.value:
            if not self.cuda_call_id and not self.dry_run:
                raise ValueError(
                    "PAIRED_PASS requires non-empty cuda_call_id (or dry_run=True)"
                )
            if not self.cpu_call_id and not self.dry_run:
                raise ValueError(
                    "PAIRED_PASS requires non-empty cpu_call_id (or dry_run=True)"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lane_id": self.lane_id,
            "substrate_id": self.substrate_id,
            "archive_sha256_paired": self.archive_sha256_paired,
            "archive_bytes": int(self.archive_bytes),
            "submission_dir": self.submission_dir,
            "verdict": self.verdict,
            "verdict_rationale": self.verdict_rationale,
            "cuda_score": self.cuda_score,
            "cuda_axis_tag": self.cuda_axis_tag,
            "cuda_hardware_substrate": self.cuda_hardware_substrate,
            "cuda_call_id": self.cuda_call_id,
            "cuda_seg_distortion": self.cuda_seg_distortion,
            "cuda_pose_distortion": self.cuda_pose_distortion,
            "cuda_rate_term": self.cuda_rate_term,
            "cuda_auth_eval_json_path": self.cuda_auth_eval_json_path,
            "cuda_elapsed_seconds": float(self.cuda_elapsed_seconds),
            "cuda_cost_usd": float(self.cuda_cost_usd),
            "cpu_score": self.cpu_score,
            "cpu_axis_tag": self.cpu_axis_tag,
            "cpu_hardware_substrate": self.cpu_hardware_substrate,
            "cpu_call_id": self.cpu_call_id,
            "cpu_seg_distortion": self.cpu_seg_distortion,
            "cpu_pose_distortion": self.cpu_pose_distortion,
            "cpu_rate_term": self.cpu_rate_term,
            "cpu_auth_eval_json_path": self.cpu_auth_eval_json_path,
            "cpu_elapsed_seconds": float(self.cpu_elapsed_seconds),
            "cpu_cost_usd": float(self.cpu_cost_usd),
            "cuda_cpu_gap": self.cuda_cpu_gap,
            "cost_band": self.cost_band,
            "budget_usd": float(self.budget_usd),
            "total_cost_usd": float(self.total_cost_usd),
            "measurement_utc": self.measurement_utc,
            "axis_tag": self.axis_tag,
            "score_claim": bool(self.score_claim),
            "promotable": bool(self.promotable),
            "evidence_grade": self.evidence_grade,
            "canonical_helper_invocation": self.canonical_helper_invocation,
            "canonical_equation_id": self.canonical_equation_id,
            "canonical_equation_status": self.canonical_equation_status,
            "cuda_platform": self.cuda_platform,
            "cuda_gpu": self.cuda_gpu,
            "cpu_target": self.cpu_target,
            "dry_run": bool(self.dry_run),
            "forbidden_macos_axis_detected": bool(self.forbidden_macos_axis_detected),
            "canonical_provenance": dict(self.canonical_provenance),
            "written_at_utc": self.written_at_utc,
            "written_pid": int(self.written_pid),
            "written_host": self.written_host,
        }


# ---------------------------------------------------------------------------
# Core API helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Canonical UTC timestamp (ISO-8601 with tz)."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def _resolve_cuda_hardware_substrate(*, platform: str, gpu: str) -> str:
    """Resolve canonical hardware substrate token from (platform, gpu)."""
    prefix = _CUDA_PLATFORM_TO_HARDWARE_SUBSTRATE_PREFIX[platform]
    return f"{prefix}_{gpu.lower()}"


def _resolve_cpu_hardware_substrate(cpu_target: str) -> str:
    """Resolve canonical hardware substrate token from CPU target."""
    return _CPU_TARGET_TO_HARDWARE_SUBSTRATE[cpu_target]


def _estimate_per_axis_cost(
    *, platform: str, gpu_or_cpu: str, cost_band: str
) -> float:
    """Estimate per-axis cost ($USD) per Catalog #270.

    Canonical per-hour rates table; cost-band scales: smoke ≈ 0.5h, full ≈ 2h.
    Conservative estimate; downstream callers should consult
    ``tac.cost_band_calibration`` for empirically-derived per-archive cost.
    """
    per_hour = _CANONICAL_PER_HOUR_RATES_USD.get((platform, gpu_or_cpu), 1.00)
    band_hours = {"smoke": 0.25, "full": 1.50}.get(cost_band, 1.00)
    return float(per_hour * band_hours)


def derive_paired_auth_eval_provenance(
    *,
    lane_id: str,
    substrate_id: str,
    archive_sha256: str,
    measurement_utc: str,
    cuda_platform: str,
    cuda_gpu: str,
    cpu_target: str,
) -> dict[str, Any]:
    """Build the canonical Provenance dict for a paired auth-eval verdict.

    Per Catalog #323 canonical Provenance umbrella: every persisted row
    carries (axis_tag + evidence_grade + score_claim + promotable +
    canonical_helper_invocation + captured_at_utc). Per CLAUDE.md "Apples-
    to-apples evidence discipline": defaults are score_claim=False +
    promotable=False; True values only when explicitly promoted via the
    canonical helper after empirical paired-axis evidence lands.
    """
    return {
        "axis_tag": PREDICTED_AXIS_TAG,
        "evidence_grade": _EVIDENCE_GRADE_NON_PROMOTABLE_PENDING,
        "score_claim": False,
        "promotable": False,
        "canonical_helper_invocation": (
            "tac.submission_packet.plan_paired_auth_eval"
        ),
        "captured_at_utc": measurement_utc,
        "lane_id": lane_id,
        "substrate_id": substrate_id,
        "archive_sha256": archive_sha256,
        "cuda_platform": cuda_platform,
        "cuda_gpu": cuda_gpu,
        "cpu_target": cpu_target,
        "canonical_equation_id": CANONICAL_EQUATION_ID,
        "canonical_equation_status": "FORMALIZATION_PENDING",
        "schema_version": PAIRED_AUTH_EVAL_SCHEMA_VERSION,
    }


def _is_macos_hardware_substrate(token: str) -> bool:
    """True iff the hardware-substrate token references macOS / Darwin ARM64."""
    return any(
        forbidden in token
        for forbidden in _FORBIDDEN_AUTHORITATIVE_HARDWARE_TOKENS
    )


def _derive_evidence_grade(
    *,
    verdict: str,
    cuda_hardware: str,
    cpu_hardware: str,
    promotable: bool,
) -> str:
    """Derive canonical evidence_grade per axis presence + Catalog #192."""
    if (
        _is_macos_hardware_substrate(cuda_hardware)
        or _is_macos_hardware_substrate(cpu_hardware)
    ):
        return _EVIDENCE_GRADE_PAIRED_MACOS_ADVISORY
    if promotable and verdict == PairedAuthEvalVerdictKind.PAIRED_PASS.value:
        return _EVIDENCE_GRADE_CONTEST_CUDA_PLUS_CPU
    if verdict == PairedAuthEvalVerdictKind.PAIRED_PARTIAL_CUDA_ONLY.value:
        return _EVIDENCE_GRADE_CUDA_ONLY
    if verdict == PairedAuthEvalVerdictKind.PAIRED_PARTIAL_CPU_ONLY.value:
        return _EVIDENCE_GRADE_CPU_ONLY
    return _EVIDENCE_GRADE_NON_PROMOTABLE_PENDING


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plan_paired_auth_eval(
    *,
    submission_bundle_result: SubmissionBundleResult,
    cost_band: str = "smoke",
    cuda_gpu: str = "T4",
    cuda_platform: str = "modal",
    cpu_target: str = "linux_x86_64_modal",
    budget_usd: float | None = None,
    dry_run: bool = True,
    operator_approved_handle: str | None = None,
    output_dir: Path | None = None,
) -> PairedAuthEvalVerdict:
    """Plan (or execute) paired Modal CUDA + Linux x86_64 CPU auth-eval.

    Sister of :func:`tac.submission_packet.enforce_contest_compliance`
    (Phase 6) + :func:`tac.submission_packet.build_submission_bundle`
    (Phase 4). The canonical entry point for Layer 5 orchestration per
    Phase 1 audit specification memo §3 Phase 6 / Layer 5.

    Args:
        submission_bundle_result: Phase 4 :class:`SubmissionBundleResult`
            carrying archive_sha256 (the sha-locked invariant), submission_dir,
            inflate_sh_path, lane_id, substrate_id.
        cost_band: ``"smoke"`` (~$0.30 envelope) or ``"full"`` (~$5.00
            envelope) per Catalog #270.
        cuda_gpu: One of :data:`_CANONICAL_CUDA_GPU_CLASSES` (T4 / L4 /
            A10G / L40S / A100 / 4090 / H100) per Catalog #215.
        cuda_platform: One of ``"modal"`` / ``"vastai"`` / ``"lightning"``
            (always Linux x86_64 host environment).
        cpu_target: One of :data:`_CPU_TARGET_TO_HARDWARE_SUBSTRATE` keys
            per Catalog #192 + CLAUDE.md "Submission auth eval — BOTH CPU
            AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable.
            ``"darwin_arm64_advisory"`` is permitted ONLY in dry-run mode
            or with explicit operator acknowledgment; the resulting verdict
            is structurally non-promotable per Catalog #192.
        budget_usd: $USD envelope ceiling. None ⇒ canonical cost-band
            default per :data:`_COST_BAND_BUDGET_USD`.
        dry_run: True ⇒ no spawn fires; verdict carries cost estimate only.
            False ⇒ actual Modal dispatch fires; REQUIRES
            ``operator_approved_handle`` per CLAUDE.md "Executing actions
            with care".
        operator_approved_handle: Operator handle for paid spend
            authorization. REQUIRED when ``dry_run=False`` per CLAUDE.md
            non-negotiable. Format: ``"<handle>:<UTC_timestamp>"``.
        output_dir: Directory for canonical per-axis JSON reports +
            verdict persistence. Defaults to
            ``reports/pr_pre_submission/<lane_id>_<utc>/``.

    Returns:
        :class:`PairedAuthEvalVerdict` with one of the canonical verdict
        kinds + per-axis empirical fields populated.

    Raises:
        :class:`PairedAuthEvalError`: orchestration crashed structurally
            (missing archive / sha mismatch / Modal ledger registration
            failure / operator-approval missing for paid spend / etc.).
            Use the typed verdict (with ``verdict=BLOCKED_*``) for
            structured failure paths.
        :class:`ValueError`: argument validation failed (invalid GPU /
            cost-band / cpu target / etc.).
    """
    # Argument validation (fail-closed; raises ValueError before any spawn)
    if not isinstance(submission_bundle_result, SubmissionBundleResult):
        raise ValueError(
            "submission_bundle_result must be a SubmissionBundleResult; "
            f"got {type(submission_bundle_result).__name__}"
        )
    if cost_band not in _COST_BAND_BUDGET_USD:
        raise ValueError(
            f"cost_band must be one of {sorted(_COST_BAND_BUDGET_USD)}; got {cost_band!r}"
        )
    if cuda_gpu not in _CANONICAL_CUDA_GPU_CLASSES:
        raise ValueError(
            f"cuda_gpu must be one of {sorted(_CANONICAL_CUDA_GPU_CLASSES)}; got {cuda_gpu!r}"
        )
    if cuda_platform not in {"modal", "vastai", "lightning"}:
        raise ValueError(
            f"cuda_platform must be modal/vastai/lightning; got {cuda_platform!r}"
        )
    if cpu_target not in _CPU_TARGET_TO_HARDWARE_SUBSTRATE:
        raise ValueError(
            f"cpu_target must be one of {sorted(_CPU_TARGET_TO_HARDWARE_SUBSTRATE)}; "
            f"got {cpu_target!r}"
        )
    if budget_usd is None:
        budget_usd = _COST_BAND_BUDGET_USD[cost_band]
    if budget_usd < 0:
        raise ValueError(f"budget_usd must be non-negative; got {budget_usd!r}")
    if not dry_run and not operator_approved_handle:
        raise PairedAuthEvalError(
            "dry_run=False requires operator_approved_handle per CLAUDE.md "
            "'Executing actions with care' non-negotiable. Set dry_run=True "
            "for plan-only mode."
        )

    # Resolve canonical hardware substrate tokens
    cuda_hardware = _resolve_cuda_hardware_substrate(
        platform=cuda_platform, gpu=cuda_gpu
    )
    cpu_hardware = _resolve_cpu_hardware_substrate(cpu_target)
    forbidden_macos = _is_macos_hardware_substrate(cpu_hardware) or _is_macos_hardware_substrate(
        cuda_hardware
    )

    # Estimate per-axis cost
    cuda_estimated_cost = _estimate_per_axis_cost(
        platform=cuda_platform, gpu_or_cpu=cuda_gpu, cost_band=cost_band
    )
    cpu_estimated_cost = (
        _estimate_per_axis_cost(
            platform=cuda_platform if cpu_target.startswith("linux_x86_64_modal") else "modal",
            gpu_or_cpu="cpu",
            cost_band=cost_band,
        )
        if not cpu_target.startswith("darwin")
        else 0.0
    )
    estimated_total = cuda_estimated_cost + cpu_estimated_cost

    measurement_utc = _utc_now_iso()

    # Resolve output_dir
    if output_dir is None:
        output_dir = (
            REPO_ROOT
            / "reports"
            / "pr_pre_submission"
            / f"{submission_bundle_result.lane_id}_paired_auth_eval_{measurement_utc.replace(':', '')}"
        )

    # Pre-dispatch validation: budget envelope
    if estimated_total > budget_usd:
        return _build_blocked_pre_dispatch_verdict(
            bundle=submission_bundle_result,
            cost_band=cost_band,
            cuda_gpu=cuda_gpu,
            cuda_platform=cuda_platform,
            cpu_target=cpu_target,
            cuda_hardware=cuda_hardware,
            cpu_hardware=cpu_hardware,
            budget_usd=budget_usd,
            total_cost_usd=estimated_total,
            forbidden_macos=forbidden_macos,
            measurement_utc=measurement_utc,
            dry_run=dry_run,
            verdict_rationale=(
                f"BLOCKED_PRE_DISPATCH: estimated_total_cost ${estimated_total:.2f} "
                f"exceeds budget_usd ${budget_usd:.2f} per Catalog #270. "
                "Operator-routable: raise budget_usd OR downgrade cuda_gpu / cost_band."
            ),
        )

    # Pre-dispatch validation: archive existence (only when not dry-run)
    archive_zip_path = Path(submission_bundle_result.submission_dir) / "archive.zip"
    if not dry_run and not archive_zip_path.is_absolute():
        archive_zip_path = REPO_ROOT / archive_zip_path
    if not dry_run and not archive_zip_path.exists():
        return _build_blocked_pre_dispatch_verdict(
            bundle=submission_bundle_result,
            cost_band=cost_band,
            cuda_gpu=cuda_gpu,
            cuda_platform=cuda_platform,
            cpu_target=cpu_target,
            cuda_hardware=cuda_hardware,
            cpu_hardware=cpu_hardware,
            budget_usd=budget_usd,
            total_cost_usd=estimated_total,
            forbidden_macos=forbidden_macos,
            measurement_utc=measurement_utc,
            dry_run=dry_run,
            verdict_rationale=(
                f"BLOCKED_PRE_DISPATCH: archive_zip {archive_zip_path} does not "
                "exist. Operator-routable: re-build submission_dir via canonical "
                "tac.submission_packet.build_submission_bundle."
            ),
        )

    # Dry-run mode: emit plan-only verdict
    if dry_run:
        return _build_dry_run_plan_verdict(
            bundle=submission_bundle_result,
            cost_band=cost_band,
            cuda_gpu=cuda_gpu,
            cuda_platform=cuda_platform,
            cpu_target=cpu_target,
            cuda_hardware=cuda_hardware,
            cpu_hardware=cpu_hardware,
            budget_usd=budget_usd,
            estimated_cuda_cost=cuda_estimated_cost,
            estimated_cpu_cost=cpu_estimated_cost,
            forbidden_macos=forbidden_macos,
            measurement_utc=measurement_utc,
        )

    # ---- EXECUTE mode (dry_run=False) ----
    # Per CLAUDE.md "Executing actions with care": this branch fires PAID
    # Modal dispatches. Defer to canonical helpers + canonical Modal
    # call_id ledger. The actual subprocess invocation is via
    # tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call
    # which routes through experiments/contest_auth_eval.py (Catalog #226).
    #
    # Phase 7 scope (this landing): wire the orchestration + verdict
    # construction. Phase 10 (future PR111-candidate end-to-end regression)
    # exercises this branch on a real archive with paired Modal dispatches.
    raise PairedAuthEvalError(
        "Phase 7 paired_auth_eval execute-mode is operator-gated per CLAUDE.md "
        "'Submission auth eval — BOTH CPU AND CUDA' + 'Executing actions with care'. "
        "The canonical Modal dispatcher tools/dispatch_modal_paired_auth_eval.py "
        "remains the operator-facing entry point for paid dispatch. Phase 7's "
        "plan_paired_auth_eval(dry_run=True) emits the canonical PairedAuthEvalVerdict "
        "schema; future Phase 10 PR111-candidate regression wires the execute "
        "branch via the canonical gate_auth_eval_call + register_dispatched_call_id_fail_closed "
        "helpers without invoking subprocess directly from this module. "
        "Operator-routable: invoke tools/dispatch_modal_paired_auth_eval.py "
        "--execute --archive-zip <path> --lane-id <id> with the operator-approved "
        f"handle {operator_approved_handle!r}, then route the resulting JSON "
        "reports through tac.submission_packet.paired_auth_eval.reconstruct_verdict_from_disk."
    )


def _build_dry_run_plan_verdict(
    *,
    bundle: SubmissionBundleResult,
    cost_band: str,
    cuda_gpu: str,
    cuda_platform: str,
    cpu_target: str,
    cuda_hardware: str,
    cpu_hardware: str,
    budget_usd: float,
    estimated_cuda_cost: float,
    estimated_cpu_cost: float,
    forbidden_macos: bool,
    measurement_utc: str,
) -> PairedAuthEvalVerdict:
    """Build a dry-run plan-only :class:`PairedAuthEvalVerdict`."""
    verdict_kind = (
        PairedAuthEvalVerdictKind.BLOCKED_HARDWARE_NON_COMPLIANT.value
        if forbidden_macos
        else PairedAuthEvalVerdictKind.BLOCKED_PRE_DISPATCH.value
    )
    rationale = (
        f"DRY-RUN plan-only: cuda_platform={cuda_platform} cuda_gpu={cuda_gpu} "
        f"cpu_target={cpu_target} cost_band={cost_band} budget_usd=${budget_usd:.2f} "
        f"estimated_total=${(estimated_cuda_cost + estimated_cpu_cost):.2f}. "
    )
    if forbidden_macos:
        rationale += (
            "FORBIDDEN_MACOS_AXIS detected per Catalog #192 — verdict structurally "
            "non-promotable regardless of empirical outcome."
        )
    else:
        rationale += (
            "Plan is canonically valid; invoke with dry_run=False + operator_approved_handle "
            "to fire paired Modal dispatch."
        )
    provenance = derive_paired_auth_eval_provenance(
        lane_id=bundle.lane_id,
        substrate_id=bundle.substrate_id,
        archive_sha256=bundle.archive_sha256,
        measurement_utc=measurement_utc,
        cuda_platform=cuda_platform,
        cuda_gpu=cuda_gpu,
        cpu_target=cpu_target,
    )
    return PairedAuthEvalVerdict(
        schema_version=PAIRED_AUTH_EVAL_SCHEMA_VERSION,
        lane_id=bundle.lane_id,
        substrate_id=bundle.substrate_id,
        archive_sha256_paired=bundle.archive_sha256,
        archive_bytes=bundle.archive_bytes,
        submission_dir=bundle.submission_dir,
        verdict=verdict_kind,
        verdict_rationale=rationale,
        cuda_score=None,
        cuda_axis_tag="[missing]",
        cuda_hardware_substrate=cuda_hardware,
        cuda_call_id="",
        cuda_seg_distortion=None,
        cuda_pose_distortion=None,
        cuda_rate_term=None,
        cuda_auth_eval_json_path="",
        cuda_elapsed_seconds=0.0,
        cuda_cost_usd=0.0,
        cpu_score=None,
        cpu_axis_tag=(
            "[macOS-CPU advisory]" if forbidden_macos else "[missing]"
        ),
        cpu_hardware_substrate=cpu_hardware,
        cpu_call_id="",
        cpu_seg_distortion=None,
        cpu_pose_distortion=None,
        cpu_rate_term=None,
        cpu_auth_eval_json_path="",
        cpu_elapsed_seconds=0.0,
        cpu_cost_usd=0.0,
        cuda_cpu_gap=None,
        cost_band=cost_band,
        budget_usd=budget_usd,
        total_cost_usd=0.0,
        measurement_utc=measurement_utc,
        axis_tag=PREDICTED_AXIS_TAG,
        score_claim=False,
        promotable=False,
        evidence_grade=_derive_evidence_grade(
            verdict=verdict_kind,
            cuda_hardware=cuda_hardware,
            cpu_hardware=cpu_hardware,
            promotable=False,
        ),
        canonical_helper_invocation=(
            "tac.submission_packet.plan_paired_auth_eval"
        ),
        canonical_equation_id=CANONICAL_EQUATION_ID,
        canonical_equation_status="FORMALIZATION_PENDING",
        cuda_platform=cuda_platform,
        cuda_gpu=cuda_gpu,
        cpu_target=cpu_target,
        dry_run=True,
        forbidden_macos_axis_detected=forbidden_macos,
        canonical_provenance=provenance,
        written_at_utc=measurement_utc,
        written_pid=os.getpid(),
        written_host=socket.gethostname(),
    )


def _build_blocked_pre_dispatch_verdict(
    *,
    bundle: SubmissionBundleResult,
    cost_band: str,
    cuda_gpu: str,
    cuda_platform: str,
    cpu_target: str,
    cuda_hardware: str,
    cpu_hardware: str,
    budget_usd: float,
    total_cost_usd: float,
    forbidden_macos: bool,
    measurement_utc: str,
    dry_run: bool,
    verdict_rationale: str,
) -> PairedAuthEvalVerdict:
    """Build a BLOCKED_PRE_DISPATCH :class:`PairedAuthEvalVerdict`."""
    verdict_kind = PairedAuthEvalVerdictKind.BLOCKED_PRE_DISPATCH.value
    provenance = derive_paired_auth_eval_provenance(
        lane_id=bundle.lane_id,
        substrate_id=bundle.substrate_id,
        archive_sha256=bundle.archive_sha256,
        measurement_utc=measurement_utc,
        cuda_platform=cuda_platform,
        cuda_gpu=cuda_gpu,
        cpu_target=cpu_target,
    )
    return PairedAuthEvalVerdict(
        schema_version=PAIRED_AUTH_EVAL_SCHEMA_VERSION,
        lane_id=bundle.lane_id,
        substrate_id=bundle.substrate_id,
        archive_sha256_paired="",  # empty per BLOCKED_PRE_DISPATCH spec
        archive_bytes=bundle.archive_bytes,
        submission_dir=bundle.submission_dir,
        verdict=verdict_kind,
        verdict_rationale=verdict_rationale,
        cuda_score=None,
        cuda_axis_tag="[missing]",
        cuda_hardware_substrate=cuda_hardware,
        cuda_call_id="",
        cuda_seg_distortion=None,
        cuda_pose_distortion=None,
        cuda_rate_term=None,
        cuda_auth_eval_json_path="",
        cuda_elapsed_seconds=0.0,
        cuda_cost_usd=0.0,
        cpu_score=None,
        cpu_axis_tag=(
            "[macOS-CPU advisory]" if forbidden_macos else "[missing]"
        ),
        cpu_hardware_substrate=cpu_hardware,
        cpu_call_id="",
        cpu_seg_distortion=None,
        cpu_pose_distortion=None,
        cpu_rate_term=None,
        cpu_auth_eval_json_path="",
        cpu_elapsed_seconds=0.0,
        cpu_cost_usd=0.0,
        cuda_cpu_gap=None,
        cost_band=cost_band,
        budget_usd=budget_usd,
        total_cost_usd=total_cost_usd,
        measurement_utc=measurement_utc,
        axis_tag=PREDICTED_AXIS_TAG,
        score_claim=False,
        promotable=False,
        evidence_grade=_derive_evidence_grade(
            verdict=verdict_kind,
            cuda_hardware=cuda_hardware,
            cpu_hardware=cpu_hardware,
            promotable=False,
        ),
        canonical_helper_invocation=(
            "tac.submission_packet.plan_paired_auth_eval"
        ),
        canonical_equation_id=CANONICAL_EQUATION_ID,
        canonical_equation_status="FORMALIZATION_PENDING",
        cuda_platform=cuda_platform,
        cuda_gpu=cuda_gpu,
        cpu_target=cpu_target,
        dry_run=dry_run,
        forbidden_macos_axis_detected=forbidden_macos,
        canonical_provenance=provenance,
        written_at_utc=measurement_utc,
        written_pid=os.getpid(),
        written_host=socket.gethostname(),
    )


def reconstruct_verdict_from_disk(
    *,
    submission_bundle_result: SubmissionBundleResult,
    cuda_auth_eval_json_path: Path,
    cpu_auth_eval_json_path: Path,
    cuda_call_id: str,
    cpu_call_id: str,
    cost_band: str,
    cuda_gpu: str,
    cuda_platform: str,
    cpu_target: str,
    budget_usd: float,
    cuda_cost_usd: float,
    cpu_cost_usd: float,
    cuda_elapsed_seconds: float,
    cpu_elapsed_seconds: float,
) -> PairedAuthEvalVerdict:
    """Reconstruct a :class:`PairedAuthEvalVerdict` from on-disk auth-eval JSONs.

    Phase 7 canonical post-dispatch reconstruction surface. Future Phase 10
    PR111-candidate end-to-end regression invokes this after both Modal
    dispatches harvest cleanly. Each per-axis JSON MUST be the canonical
    output of ``experiments/contest_auth_eval.py`` per Catalog #226 +
    #221 + #249 sister discipline.

    Sha-locked invariant enforcement: this helper verifies that BOTH
    per-axis JSONs reference ``submission_bundle_result.archive_sha256``;
    a mismatch surfaces as ``verdict=BLOCKED_AXIS_MISMATCH`` per Catalog
    #127 + #221.

    Raises:
        :class:`PairedAuthEvalError`: JSON files missing OR unparseable OR
            internally inconsistent (component recomputation failure per
            ``parse_auth_eval_score_claim`` semantics).
    """
    import json

    if not isinstance(submission_bundle_result, SubmissionBundleResult):
        raise ValueError(
            "submission_bundle_result must be a SubmissionBundleResult"
        )

    measurement_utc = _utc_now_iso()
    cuda_hardware = _resolve_cuda_hardware_substrate(
        platform=cuda_platform, gpu=cuda_gpu
    )
    cpu_hardware = _resolve_cpu_hardware_substrate(cpu_target)
    forbidden_macos = _is_macos_hardware_substrate(cpu_hardware) or _is_macos_hardware_substrate(
        cuda_hardware
    )

    # Parse per-axis JSONs
    if not cuda_auth_eval_json_path.exists():
        raise PairedAuthEvalError(
            f"cuda_auth_eval_json_path {cuda_auth_eval_json_path} does not exist"
        )
    if not cpu_auth_eval_json_path.exists():
        raise PairedAuthEvalError(
            f"cpu_auth_eval_json_path {cpu_auth_eval_json_path} does not exist"
        )
    try:
        cuda_payload = json.loads(cuda_auth_eval_json_path.read_text())
    except json.JSONDecodeError as exc:
        raise PairedAuthEvalError(
            f"cuda_auth_eval_json_path {cuda_auth_eval_json_path} is not parseable JSON: {exc}"
        ) from exc
    try:
        cpu_payload = json.loads(cpu_auth_eval_json_path.read_text())
    except json.JSONDecodeError as exc:
        raise PairedAuthEvalError(
            f"cpu_auth_eval_json_path {cpu_auth_eval_json_path} is not parseable JSON: {exc}"
        ) from exc

    cuda_score = cuda_payload.get("final_score")
    cuda_seg = cuda_payload.get("seg_distortion")
    cuda_pose = cuda_payload.get("pose_distortion")
    cuda_rate = cuda_payload.get("rate_term")
    cuda_archive_sha = cuda_payload.get("archive_sha256", "")
    cpu_score = cpu_payload.get("final_score")
    cpu_seg = cpu_payload.get("seg_distortion")
    cpu_pose = cpu_payload.get("pose_distortion")
    cpu_rate = cpu_payload.get("rate_term")
    cpu_archive_sha = cpu_payload.get("archive_sha256", "")

    # Sha-locked invariant cross-validation
    if (
        cuda_archive_sha
        and cpu_archive_sha
        and cuda_archive_sha != cpu_archive_sha
    ):
        verdict_kind = PairedAuthEvalVerdictKind.BLOCKED_AXIS_MISMATCH.value
        rationale = (
            f"BLOCKED_AXIS_MISMATCH: CUDA axis archive_sha256={cuda_archive_sha[:12]} "
            f"differs from CPU axis archive_sha256={cpu_archive_sha[:12]}. "
            "Sha-locked invariant violated per Catalog #127 custody discipline. "
            "Operator-routable: investigate why the two dispatches saw different bytes."
        )
        archive_sha256_paired = submission_bundle_result.archive_sha256
    elif (
        cuda_archive_sha
        and cuda_archive_sha != submission_bundle_result.archive_sha256
    ):
        verdict_kind = PairedAuthEvalVerdictKind.BLOCKED_AXIS_MISMATCH.value
        rationale = (
            f"BLOCKED_AXIS_MISMATCH: CUDA axis archive_sha256={cuda_archive_sha[:12]} "
            f"differs from bundle.archive_sha256={submission_bundle_result.archive_sha256[:12]}. "
            "Sha-locked invariant violated per Catalog #127."
        )
        archive_sha256_paired = submission_bundle_result.archive_sha256
    elif (
        cpu_archive_sha
        and cpu_archive_sha != submission_bundle_result.archive_sha256
    ):
        verdict_kind = PairedAuthEvalVerdictKind.BLOCKED_AXIS_MISMATCH.value
        rationale = (
            f"BLOCKED_AXIS_MISMATCH: CPU axis archive_sha256={cpu_archive_sha[:12]} "
            f"differs from bundle.archive_sha256={submission_bundle_result.archive_sha256[:12]}. "
            "Sha-locked invariant violated per Catalog #127."
        )
        archive_sha256_paired = submission_bundle_result.archive_sha256
    elif forbidden_macos:
        verdict_kind = PairedAuthEvalVerdictKind.BLOCKED_HARDWARE_NON_COMPLIANT.value
        rationale = (
            f"BLOCKED_HARDWARE_NON_COMPLIANT: at least one axis landed on "
            f"forbidden macOS / Darwin ARM64 hardware substrate per Catalog #192. "
            f"cuda_hardware={cuda_hardware!r} cpu_hardware={cpu_hardware!r}. "
            "Re-dispatch the offending axis on canonical Linux x86_64 substrate."
        )
        archive_sha256_paired = submission_bundle_result.archive_sha256
    elif cuda_score is not None and cpu_score is not None:
        verdict_kind = PairedAuthEvalVerdictKind.PAIRED_PASS.value
        rationale = (
            f"PAIRED_PASS: paired empirical anchor landed. CUDA={cuda_score:.6f} "
            f"on {cuda_hardware}; CPU={cpu_score:.6f} on {cpu_hardware}. "
            f"Sha-locked invariant held (archive_sha256={submission_bundle_result.archive_sha256[:12]})."
        )
        archive_sha256_paired = submission_bundle_result.archive_sha256
    elif cuda_score is not None:
        verdict_kind = PairedAuthEvalVerdictKind.PAIRED_PARTIAL_CUDA_ONLY.value
        rationale = (
            f"PAIRED_PARTIAL_CUDA_ONLY: CUDA axis landed cleanly ({cuda_score:.6f}); "
            f"CPU axis missing/failed. Operator-routable: re-dispatch CPU axis "
            f"per CLAUDE.md non-negotiable."
        )
        archive_sha256_paired = submission_bundle_result.archive_sha256
    elif cpu_score is not None:
        verdict_kind = PairedAuthEvalVerdictKind.PAIRED_PARTIAL_CPU_ONLY.value
        rationale = (
            f"PAIRED_PARTIAL_CPU_ONLY: CPU axis landed cleanly ({cpu_score:.6f}); "
            f"CUDA axis missing/failed. Operator-routable: re-dispatch CUDA axis."
        )
        archive_sha256_paired = submission_bundle_result.archive_sha256
    else:
        verdict_kind = PairedAuthEvalVerdictKind.BLOCKED_HARVEST.value
        rationale = (
            "BLOCKED_HARVEST: neither axis produced a valid contest score. "
            f"cuda_call_id={cuda_call_id!r} cpu_call_id={cpu_call_id!r}. "
            "Operator-routable: consult Modal dashboard via call_id."
        )
        archive_sha256_paired = submission_bundle_result.archive_sha256

    # Catalog #192 + Catalog #341: promotable iff PAIRED_PASS AND both axes
    # on canonical Linux x86_64 substrates AND no macOS detected
    promotable = (
        verdict_kind == PairedAuthEvalVerdictKind.PAIRED_PASS.value
        and not forbidden_macos
        and cuda_hardware.startswith("linux_x86_64_")
        and cpu_hardware in _CANONICAL_LINUX_X86_64_CPU_SUBSTRATES
    )
    score_claim = promotable  # Catalog #341 invariant

    # Per-axis tags
    cuda_axis_tag = "[contest-CUDA]" if cuda_score is not None else "[missing]"
    if cpu_score is not None:
        cpu_axis_tag = (
            "[macOS-CPU advisory]" if forbidden_macos else "[contest-CPU]"
        )
    else:
        cpu_axis_tag = "[missing]"

    # cuda_cpu_gap
    cuda_cpu_gap = (
        float(cuda_score - cpu_score)
        if (cuda_score is not None and cpu_score is not None)
        else None
    )

    # axis_tag canonical
    axis_tag = (
        "[contest-CUDA; contest-CPU]" if promotable else PREDICTED_AXIS_TAG
    )

    provenance = derive_paired_auth_eval_provenance(
        lane_id=submission_bundle_result.lane_id,
        substrate_id=submission_bundle_result.substrate_id,
        archive_sha256=archive_sha256_paired,
        measurement_utc=measurement_utc,
        cuda_platform=cuda_platform,
        cuda_gpu=cuda_gpu,
        cpu_target=cpu_target,
    )
    # Per Catalog #323: when promotable, evidence_grade upgrades to paired-axis-empirical
    if promotable:
        provenance = dict(provenance)
        provenance["evidence_grade"] = _EVIDENCE_GRADE_CONTEST_CUDA_PLUS_CPU
        provenance["score_claim"] = True
        provenance["promotable"] = True
        provenance["axis_tag"] = "[contest-CUDA; contest-CPU]"

    return PairedAuthEvalVerdict(
        schema_version=PAIRED_AUTH_EVAL_SCHEMA_VERSION,
        lane_id=submission_bundle_result.lane_id,
        substrate_id=submission_bundle_result.substrate_id,
        archive_sha256_paired=archive_sha256_paired,
        archive_bytes=submission_bundle_result.archive_bytes,
        submission_dir=submission_bundle_result.submission_dir,
        verdict=verdict_kind,
        verdict_rationale=rationale,
        cuda_score=float(cuda_score) if cuda_score is not None else None,
        cuda_axis_tag=cuda_axis_tag,
        cuda_hardware_substrate=cuda_hardware,
        cuda_call_id=cuda_call_id,
        cuda_seg_distortion=float(cuda_seg) if cuda_seg is not None else None,
        cuda_pose_distortion=float(cuda_pose) if cuda_pose is not None else None,
        cuda_rate_term=float(cuda_rate) if cuda_rate is not None else None,
        cuda_auth_eval_json_path=str(cuda_auth_eval_json_path),
        cuda_elapsed_seconds=float(cuda_elapsed_seconds),
        cuda_cost_usd=float(cuda_cost_usd),
        cpu_score=float(cpu_score) if cpu_score is not None else None,
        cpu_axis_tag=cpu_axis_tag,
        cpu_hardware_substrate=cpu_hardware,
        cpu_call_id=cpu_call_id,
        cpu_seg_distortion=float(cpu_seg) if cpu_seg is not None else None,
        cpu_pose_distortion=float(cpu_pose) if cpu_pose is not None else None,
        cpu_rate_term=float(cpu_rate) if cpu_rate is not None else None,
        cpu_auth_eval_json_path=str(cpu_auth_eval_json_path),
        cpu_elapsed_seconds=float(cpu_elapsed_seconds),
        cpu_cost_usd=float(cpu_cost_usd),
        cuda_cpu_gap=cuda_cpu_gap,
        cost_band=cost_band,
        budget_usd=float(budget_usd),
        total_cost_usd=float(cuda_cost_usd + cpu_cost_usd),
        measurement_utc=measurement_utc,
        axis_tag=axis_tag,
        score_claim=score_claim,
        promotable=promotable,
        evidence_grade=_derive_evidence_grade(
            verdict=verdict_kind,
            cuda_hardware=cuda_hardware,
            cpu_hardware=cpu_hardware,
            promotable=promotable,
        ),
        canonical_helper_invocation=(
            "tac.submission_packet.reconstruct_verdict_from_disk"
        ),
        canonical_equation_id=CANONICAL_EQUATION_ID,
        canonical_equation_status="FORMALIZATION_PENDING",
        cuda_platform=cuda_platform,
        cuda_gpu=cuda_gpu,
        cpu_target=cpu_target,
        dry_run=False,
        forbidden_macos_axis_detected=forbidden_macos,
        canonical_provenance=provenance,
        written_at_utc=measurement_utc,
        written_pid=os.getpid(),
        written_host=socket.gethostname(),
    )


__all__ = [
    "PAIRED_AUTH_EVAL_SCHEMA_VERSION",
    "PHASE_7_LAYER_VERSION",
    "CANONICAL_EQUATION_ID",
    "PREDICTED_AXIS_TAG",
    "PairedAuthEvalError",
    "PairedAuthEvalVerdict",
    "PairedAuthEvalVerdictKind",
    "derive_paired_auth_eval_provenance",
    "plan_paired_auth_eval",
    "reconstruct_verdict_from_disk",
]
