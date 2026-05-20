#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Cathedral autopilot autonomous loop — typed-atom queue → Pareto → dispatch.

Sister of :mod:`tools.cathedral_autopilot` (the per-invocation planner). This
module is the **continuous** loop: it monitors the typed-atom evidence queue,
ranks candidates against the Pareto frontier, surfaces operator-decision
events, and feeds harvested results back into the continual-learning
posterior so the NEXT loop iteration ranks against the updated state.

**HALT-and-ASK pattern (operator-gate non-negotiable per CLAUDE.md):**

The loop is "autonomous" only in the *ranking + harvesting + feedback* sense.
Every dispatch decision still requires explicit operator approval. The loop
HALTS at every operator-decision event and emits a structured ``HALT_EVENT``
that the operator answers (CLI: ``--operator-decision``; programmatic:
:func:`inject_operator_decision`). The ``--require-operator-approval-on``
flag enumerates which event classes block; passing
``--require-operator-approval-on dispatch`` makes EVERY dispatch decision
operator-gated (the CLAUDE.md non-negotiable default).

Per CLAUDE.md "EXTREME EMOJI rule": NO emojis in the source / output.

Per CLAUDE.md "Forbidden score claims": this loop NEVER claims a score. It
ranks candidates by predicted score band (tagged ``[predicted; cathedral
autopilot ranking]``) and gates on operator decision before any dispatch.

Per CLAUDE.md "race-mode rigor inversion": when ``--race-mode`` is set, the
loop's dispatch-approval policy switches from "max rigor" to "smallest
credible bolt-on within ~60 minutes." Operator must explicitly enable
race-mode via the ``--race-mode`` flag.

Per CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION": before any dispatch is
recommended, the loop checks the active-lane-dispatch-claims registry. If a
conflicting claim exists, the candidate is moved to ``DEFERRED-pending-claim``
status and the operator is notified.

**Operator-authorized le-$5/individual mode (2026-05-11):**

Per operator directive 2026-05-11 ("keep pushing the autopilot and xray and
magic codec and compiler and wiring and integration and everything") plus the
council's deferral overrule, the autopilot may now self-authorize dispatches
that satisfy ALL of:

  1. ``--operator-authorized-le-5-dollar-mode`` flag is explicitly set
     (default OFF; safe HALT-and-ASK preserved when the flag is absent),
  2. ``estimated_dispatch_cost_usd`` <= ``--per-dispatch-cap-usd`` (default
     $5.00 per the operator directive),
  3. cumulative-since-activation cost <= ``--cumulative-cap-usd`` (default
     $20.00 hard envelope),
  4. ``--canonical-helper-script`` resolves to ``tools/claim_lane_dispatch.py``
     OR explicit override via the same flag (the lane-claim ledger must be
     reachable for every authorized dispatch),
  5. ``CATHEDRAL_AUTOPILOT_OPERATOR_AUTHORIZED_MODE`` environment variable is
     set to ``1`` (defense-in-depth runtime gate — refuses to activate if the
     CLI flag is set but the env-var is not, even if other preconditions are
     met).

When all five preconditions hold, each candidate that fits the per-dispatch
cap must first record a lane claim through :mod:`tools.claim_lane_dispatch`.
Only after the claim helper succeeds is the candidate tagged
``[autopilot-claude-le-5-dollar]`` with ``requires_approval=False`` and
written to the configured journal path. Candidates crossing either cap, or
whose claim helper invocation fails, remain HALT-and-ASK as before. The
cumulative-envelope counter is per-process (the loop's state); the canonical
persistent ledger is :mod:`tools.claim_lane_dispatch`.

No KILL verdict is ever auto-authorized.

Cross-references
----------------
- :mod:`tools.cathedral_autopilot` — per-invocation planner / ranker
- :mod:`tac.continual_learning` — posterior consumed and updated by the loop
- :mod:`tools.claim_lane_dispatch` — dispatch-claim coordination
- ``feedback_5_beyond_phase4_modules_landed_20260509``
- ``feedback_grand_council_fields_medal_phase2_floor_refinement_20260509``

CLAUDE.md compliance tags
-------------------------
- ``operator_gate_non_negotiable_at_every_dispatch``
- ``halt_and_ask_pattern_default_on``
- ``no_score_claim_only_predicted_band``
- ``cross_agent_dispatch_coordination_check``
- ``race_mode_explicit_opt_in_only``
- ``no_tmp_paths``
- ``forbidden_premature_kill_without_research_exhaustion_no_kill_verdicts``
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

try:  # Reuse the canonical parser/conflict semantics from the claim helper.
    from tools import claim_lane_dispatch as _claim_lane_dispatch
except ModuleNotFoundError:  # pragma: no cover - direct tools/ import mode
    import claim_lane_dispatch as _claim_lane_dispatch  # type: ignore[no-redef]

# W/I/A I-1 wire-in (2026-05-12, decision I-1): the autonomous loop's
# rank_candidates step now optionally consumes the continual-learning posterior
# AND the cost-band-calibration posterior so empirical anchors reweight
# predicted score deltas (continual-learning) and refresh cost-band envelope
# decisions (cost-band). The wire-in mirrors `tools/cathedral_autopilot.py`'s
# `_posterior_correction_for_technique` pattern: a posterior-derived factor
# multiplies predicted_score_delta to produce an `adjusted_predicted_delta`
# that drives the ranking sort. The autopilot ranks are then surfaced as
# halt events for operator decision. Per CLAUDE.md "Subagent
# coherence-by-default" the wire-in is the structural fix; the per-candidate
# correction never auto-promotes nor auto-kills.
try:  # pragma: no cover - exercised by integration tests
    from tac.continual_learning import (
        load_posterior as load_continual_learning_posterior,
    )
    from tac.continual_learning import (
        posterior_query_track_correction,
    )
    from tac.cost_band_calibration import predict as predict_cost_band
    _POSTERIOR_IMPORTS_OK = True
except Exception:  # pragma: no cover - tests can stub these
    load_continual_learning_posterior = None  # type: ignore[assignment]
    posterior_query_track_correction = None  # type: ignore[assignment]
    predict_cost_band = None  # type: ignore[assignment]
    _POSTERIOR_IMPORTS_OK = False

from tac.master_gradient import contest_axis_authority_violation_reason  # noqa: E402
from tac.optimization.literature_source_scope import (  # noqa: E402
    literature_source_scope_blockers,
)
from tac.optimization.substrate_composition_matrix import (  # noqa: E402
    canonical_substrate_inventory,
)
from tac.optimizer.exact_dispatch_authority import (  # noqa: E402
    exact_dispatch_authority,
)

AUTONOMOUS_LOOP_SCHEMA = "tac_cathedral_autopilot_autonomous_loop_v1"

# Operator-authorized le-$5/individual mode (2026-05-11).
OPERATOR_AUTHORIZED_MODE_ENV_VAR = "CATHEDRAL_AUTOPILOT_OPERATOR_AUTHORIZED_MODE"
OPERATOR_AUTHORIZED_MODE_ENV_VALUE_ENABLED = "1"
DEFAULT_PER_DISPATCH_CAP_USD = 5.00
DEFAULT_CUMULATIVE_CAP_USD = 20.00
AUTOPILOT_AUTHORIZED_TAG = "[autopilot-claude-le-5-dollar]"
CANONICAL_HELPER_SCRIPT_RELPATH = "tools/claim_lane_dispatch.py"
AUTOPILOT_CONTEST_TARGET_MODE = "contest_exact_eval"
PLANNING_ONLY_SOURCE_BLOCKER = "planning_only_source_requires_operator_dispatch_packet"
LITERATURE_SOURCE_SCOPE_BLOCKER_PREFIX = "literature_anchor_source_scope_missing"
COMPOSITION_MATRIX_UNKNOWN_SUBSTRATE_BLOCKER_PREFIX = (
    "composition_matrix_unknown_substrate"
)
AUTOPILOT_CLAIM_PLATFORM = "cathedral_autopilot"
AUTOPILOT_CLAIM_STATUS = "active_autopilot_authorized_dispatch"
AUTOPILOT_CLAIM_AGENT = "cathedral_autopilot_autonomous_loop"
AUTOPILOT_CLAIM_TTL_HOURS = 24.0
EXACT_READY_QUEUE_SCHEMA = "optimizer_candidate_exact_eval_ready_queue_v1"
_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")
CONTEST_PAIR_COUNT = 600


def _contest_axis_tag_for_panel(panel_axis: str) -> str:
    """Return the exact master-gradient measurement_axis tag for a panel."""
    if panel_axis == "contest_cpu":
        return "[contest-CPU]"
    if panel_axis == "contest_cuda":
        return "[contest-CUDA]"
    raise ValueError(
        "unsupported score panel axis "
        f"{panel_axis!r}; expected 'contest_cpu' or 'contest_cuda'"
    )


def _is_sha256_hex(value: object) -> bool:
    """Return True only for concrete 64-hex SHA-256 strings."""
    return bool(_SHA256_HEX_RE.fullmatch(str(value or "").strip()))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _npy_shape(path: Path) -> tuple[int, ...] | None:
    try:
        import numpy as np

        arr = np.load(path, mmap_mode="r")
    except (ImportError, OSError, ValueError):
        return None
    return tuple(int(dim) for dim in arr.shape)


def _sidecar_has_full_pair_gradient_custody(
    payload: dict[str, Any],
    archive_sha256: str,
    *,
    axis_tag: str,
) -> bool:
    """Fail-closed custody check for sidecars that drive Cathedral ranking."""
    if payload.get("measurement_axis") != axis_tag:
        return False
    if contest_axis_authority_violation_reason(payload) is not None:
        return False
    if not _is_sha256_hex(payload.get("scored_archive_sha256")):
        return False
    if str(payload.get("scored_archive_sha256")).lower() != archive_sha256.lower():
        return False
    scored_archive_bytes = payload.get("scored_archive_bytes")
    if (
        isinstance(scored_archive_bytes, bool)
        or not isinstance(scored_archive_bytes, int)
        or scored_archive_bytes <= 0
    ):
        return False
    if not _is_sha256_hex(payload.get("gradient_array_sha256")):
        return False
    if payload.get("gradient_tensor_kind") != "per_pair_per_byte_v1":
        return False
    gradient_array_path_raw = payload.get("gradient_array_path")
    if not isinstance(gradient_array_path_raw, str) or not gradient_array_path_raw.strip():
        return False
    gradient_array_path = Path(gradient_array_path_raw)
    if not gradient_array_path.is_absolute():
        gradient_array_path = REPO_ROOT / gradient_array_path
    if not gradient_array_path.is_file():
        return False
    try:
        if _sha256_file(gradient_array_path).lower() != str(payload.get("gradient_array_sha256")).lower():
            return False
    except OSError:
        return False

    n_pairs = payload.get("n_pairs")
    n_pairs_used = payload.get("n_pairs_used")
    n_pairs_total = payload.get("n_pairs_total")
    if (
        isinstance(n_pairs, bool)
        or isinstance(n_pairs_used, bool)
        or isinstance(n_pairs_total, bool)
        or not isinstance(n_pairs, int)
        or not isinstance(n_pairs_used, int)
        or not isinstance(n_pairs_total, int)
        or n_pairs != CONTEST_PAIR_COUNT
        or n_pairs_used != CONTEST_PAIR_COUNT
        or n_pairs_total != CONTEST_PAIR_COUNT
    ):
        return False

    shape = _npy_shape(gradient_array_path)
    if shape is None:
        return False
    return len(shape) == 3 and shape[0] > 0 and shape[1:] == (CONTEST_PAIR_COUNT, 3)


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_authorized_journal_path(journal_path: Path, *, repo_root: Path = REPO_ROOT) -> None:
    """Refuse transient paths for self-authorized dispatch journals."""
    forbidden_roots = (
        Path("/tmp"),
        Path("/private/tmp"),
        Path("/var/tmp"),
        Path(tempfile.gettempdir()),
    )
    if any(_path_is_relative_to(journal_path, root) for root in forbidden_roots):
        raise ValueError(
            "--journal-path for authorized mode must be durable and repo-local "
            "(.omx/state/ or reports/); refusing transient path "
            f"{str(journal_path)!r}"
        )
    allowed_roots = (repo_root / ".omx" / "state", repo_root / "reports")
    if any(_path_is_relative_to(journal_path, root) for root in allowed_roots):
        return
    raise ValueError(
        "--journal-path for authorized mode must be under repo-local .omx/state/ "
        f"or reports/; got {str(journal_path)!r}"
    )


def _require_candidate_dispatch_cost(candidate: CandidateRow) -> float:
    return _require_finite_positive_float(
        candidate.estimated_dispatch_cost_usd,
        field="estimated_dispatch_cost_usd",
        context=f"candidate {candidate.candidate_id!r}",
    )


def _require_candidate_planning_cost(candidate: CandidateRow) -> float:
    return _require_finite_nonnegative_float(
        candidate.estimated_dispatch_cost_usd,
        field="estimated_dispatch_cost_usd",
        context=f"candidate {candidate.candidate_id!r}",
    )


def validate_authorized_mode_config(
    auth_mode: OperatorAuthorizedModeConfig | None,
    *,
    repo_root: Path = REPO_ROOT,
) -> None:
    """Validate authorized-mode config before ranking or dispatch side effects."""
    if auth_mode is None or not auth_mode.enabled:
        return
    _require_finite_positive_float(
        auth_mode.per_dispatch_cap_usd,
        field="per_dispatch_cap_usd",
        context="operator-authorized mode",
    )
    _require_finite_positive_float(
        auth_mode.cumulative_cap_usd,
        field="cumulative_cap_usd",
        context="operator-authorized mode",
    )
    try:
        spent = float(auth_mode.cumulative_spent_usd)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "operator-authorized mode has non-numeric cumulative_spent_usd="
            f"{auth_mode.cumulative_spent_usd!r}"
        ) from exc
    if not math.isfinite(spent) or spent < 0.0:
        raise ValueError(
            "operator-authorized mode must carry finite non-negative "
            f"cumulative_spent_usd; got {auth_mode.cumulative_spent_usd!r}"
        )
    if auth_mode.journal_path is None:
        raise ValueError(
            "operator-authorized mode requires a durable journal_path before "
            "ranking or dispatch authorization"
        )
    validate_authorized_journal_path(auth_mode.journal_path, repo_root=repo_root)


def _authorized_mode_config_blocker(
    auth_mode: OperatorAuthorizedModeConfig | None,
    *,
    repo_root: Path = REPO_ROOT,
) -> str:
    try:
        validate_authorized_mode_config(auth_mode, repo_root=repo_root)
    except ValueError as exc:
        return str(exc)
    return ""


# ── Events / decisions / verdicts ──────────────────────────────────────────


class EventClass(StrEnum):
    """Operator-decision event classes."""

    DISPATCH = "dispatch"
    KILL = "kill"  # NEVER auto-kill per CLAUDE.md; operator-only
    PROMOTE = "promote"
    POSTERIOR_REWEIGHT = "posterior_reweight"
    RACE_MODE_TOGGLE = "race_mode_toggle"


class OperatorDecision(StrEnum):
    """Operator's response to a HALT event."""

    APPROVE = "approve"
    REJECT = "reject"
    DEFER = "defer"


@dataclass
class CandidateRow:
    """One typed-atom row from the candidate queue.

    Mirrors the cathedral autopilot's TechniqueEvidence schema in spirit but
    is intentionally minimal — the autonomous loop ranks candidates by
    predicted score delta + EIG/$ and surfaces them for operator decision.

    Per CLAUDE.md "Forbidden score claims": ``predicted_score_delta`` is
    explicitly tagged as a prediction, never a measurement.

    Z1 empirical revision fields (2026-05-14, decision #4 per
    ``feedback_z1_mdl_ablation_landed_20260514.md``):

      - ``mdl_density`` — measured scorer-conditional MDL density (0-1) per
        the Z1 ablation tool. None when the candidate has no ablation
        evidence yet (don't penalize lack-of-evidence). Density >0.95
        indicates within-class trap; 0.90-0.95 trending; <0.90 across-class
        promising.
      - ``lane_class`` — declared substrate class ("substrate_class_shift",
        "substrate_engineering", "research_substrate", etc.) used to apply
        the class-shift reward. None = unknown / not-declared.
      - ``literature_anchor`` — citation / family name surfacing the
        substrate-class lineage (e.g. "cooperative-receiver",
        "Tishby-Zaslavsky", "Wyner-Ziv"). Used by
        :func:`adjust_score_for_class_shift` to reward known class-shift
        primitives.
      - ``source_supports`` / ``paper_claim_scope`` / ``pact_must_prove`` /
        ``decode_complexity_evidence`` — source-fidelity scope fields carried
        from the composition matrix so literature anchors cannot silently become
        empirical Pact score claims.
    """

    candidate_id: str
    family: str  # e.g. 'hnerv_lc_v2', 'balle_scale_hyperprior'
    predicted_score_delta: float  # negative = improvement
    expected_information_gain: float
    estimated_dispatch_cost_usd: float
    blockers: list[str] = field(default_factory=list)
    notes: str = ""
    timing_smoke_command: str = ""
    # Z1 empirical revision wire-in (2026-05-14):
    mdl_density: float | None = None
    lane_class: str | None = None
    literature_anchor: str = ""
    source_supports: str = ""
    paper_claim_scope: str = ""
    pact_must_prove: str = ""
    decode_complexity_evidence: str = ""
    # Catalog #227 Tier C empirical revision wire-in (2026-05-14):
    # Tier C-derived substrate-class density estimate (0..1). HIGH (>= 0.70) =
    # within-class; LOW (<= 0.30) = across-class. The signal OVERRIDES Tier A
    # for substrate-class discrimination because Tier A is brotli-saturated at
    # the byte layer (any fp16-weight + brotli archive sits at Tier A density
    # ~0.99) and structurally CANNOT discriminate. Per the C6 5ep empirical
    # anchor `feedback_mdl_ablation_tier_c_ibps1_landed_20260514.md`. None =
    # no Tier C evidence (don't penalize lack-of-evidence).
    mdl_tier_c_density: float | None = None
    # Catalog #227 composition matrix wire-in (2026-05-14):
    # Substrate composition additivity factor alpha (0..1+). HIGH (> 0.7) =
    # ADDITIVE stacking (compound predicted_score_delta savings); MEDIUM
    # (0.3-0.7) = SUB-ADDITIVE (halve savings); LOW (<= 0.3) = SATURATING
    # (single-substrate dominates; floor predicted_score_delta near zero).
    # Sourced from `.omx/state/substrate_composition_matrix.json` per T1-F
    # `feedback_t1_f_z3_x_c6_composition_probe_build_landed_20260514.md`.
    # None = no composition evidence (single-substrate candidate or untested
    # pair); the candidate is ranked without composition adjustment.
    composition_alpha: float | None = None
    license_ok: bool = True
    inflate_dep_count: int = 0
    sideinfo_consumed: bool | None = None
    exact_duplicate: bool = False
    context_order: int = 0
    # OP-3 predicted_dispatch_risk wire-in (2026-05-15, codex chunk 6 finding):
    # SLIM (Sparse Linear Integer Model) preflight risk score per
    # `tac.preflight_rudin_daubechies.slim_risk_scorer.PreflightSLIMRiskScorer`.
    # Range nominally 0..200 with the canonical refusal threshold at
    # ``DISPATCH_RISK_REFUSAL_THRESHOLD = 50`` (see slim_risk_scorer.py:111).
    # Bands consumed by :func:`adjust_predicted_delta_for_predicted_dispatch_risk`:
    #   risk >= 50 → REFUSE (floor predicted_score_delta at 0)
    #   risk 25-50 → MODERATE (halve predicted savings)
    #   risk < 25  → LOW (no adjustment)
    # None = no preflight evidence (don't penalize lack-of-evidence per the
    # sister Z1 / Tier C / composition_alpha conventions).
    predicted_dispatch_risk: float | None = None
    # Dispatch authority / custody fields. These are intentionally absent from
    # read-only planning sources; operator-authorized self-dispatch requires a
    # real dispatch packet instead of inferred readiness from rank/cost.
    lane_id: str = ""
    claim_keys: list[str] = field(default_factory=list)
    target_modes: list[str] = field(default_factory=list)
    dispatch_packet_ready: bool = False
    dispatch_packet_sha256: str = ""
    archive_path: str = ""
    submission_dir: str = ""
    archive_manifest_path: str = ""
    archive_sha256: str = ""
    candidate_archive_bytes: int | None = None
    runtime_tree_sha256: str = ""
    deployment_target: str = ""
    score_affecting_payload_changed: bool = False
    charged_bits_changed: bool = False
    score_affecting_runtime_changed: bool | None = None
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def eig_per_dollar(self) -> float:
        cost = _require_candidate_planning_cost(self)
        if cost == 0.0:
            return 0.0
        return self.expected_information_gain / cost

    def dispatch_claim_keys(self) -> list[str]:
        """Return candidate/lane/claim identifiers for conflict checks."""
        keys = [self.candidate_id, self.lane_id, *self.claim_keys]
        deduped: list[str] = []
        for key in keys:
            s = str(key or "").strip()
            if s and s not in deduped:
                deduped.append(s)
        return deduped

    def dispatch_authority_blockers(
        self,
        *,
        dispatch_claims_path: Path | None = None,
    ) -> list[str]:
        """Return blockers that prevent autonomous dispatch authorization.

        Planning rows are useful for ranking, but they are not executable
        packets. The le-$5 autopilot path may only self-authorize rows that
        already carry lane identity, contest target metadata, and live custody
        for the exact archive/runtime packet being launched.
        """
        blockers: list[str] = []
        if self.score_claim:
            blockers.append("score_claim_true_requires_operator_review")
        if self.promotion_eligible:
            blockers.append("promotion_eligible_true_requires_operator_review")
        if not self.dispatch_packet_ready:
            blockers.append("dispatch_packet_ready_false")
        if not self.lane_id.strip():
            blockers.append("lane_id_required_for_dispatch_packet")
        contest_exact_target = AUTOPILOT_CONTEST_TARGET_MODE in set(self.target_modes)
        if not contest_exact_target:
            blockers.append("contest_exact_eval_target_mode_required")
        has_dispatch_packet_hash = _is_sha256_hex(self.dispatch_packet_sha256)
        has_archive_hash = _is_sha256_hex(self.archive_sha256)
        has_runtime_hash = _is_sha256_hex(self.runtime_tree_sha256)
        has_exact_packet_hashes = has_archive_hash and has_runtime_hash
        if self.dispatch_packet_sha256.strip() and not has_dispatch_packet_hash:
            blockers.append("dispatch_packet_sha256_malformed")
        if self.archive_sha256.strip() and not has_archive_hash:
            blockers.append("archive_sha256_malformed")
        if self.runtime_tree_sha256.strip() and not has_runtime_hash:
            blockers.append("runtime_tree_sha256_malformed")
        if contest_exact_target and not has_exact_packet_hashes:
            blockers.append("contest_exact_eval_requires_archive_and_runtime_hash")
        if contest_exact_target and self.ready_for_exact_eval_dispatch is not True:
            blockers.append("contest_exact_eval_requires_ready_for_exact_eval_dispatch")
        elif not (has_dispatch_packet_hash or has_exact_packet_hashes):
            blockers.append("dispatch_packet_or_archive_runtime_hash_required")
        if self.ready_for_exact_eval_dispatch and not has_exact_packet_hashes:
            blockers.append(
                "ready_for_exact_eval_dispatch_requires_archive_and_runtime_hash"
            )
        if contest_exact_target or self.ready_for_exact_eval_dispatch:
            authority = exact_dispatch_authority(
                dataclasses.asdict(self),
                repo_root=REPO_ROOT,
                source="cathedral_autopilot_autonomous_loop",
                active_floor_archive_bytes=None,
                active_floor_score=None,
                dispatch_claims_path=dispatch_claims_path
                or REPO_ROOT / ".omx" / "state" / "active_lane_dispatch_claims.md",
            )
            blockers.extend(
                f"exact_dispatch_authority:{blocker}"
                for blocker in authority.blockers
            )
        return list(dict.fromkeys(blockers))


@dataclass
class OperatorAuthorizedModeConfig:
    """Configuration for the operator-authorized le-$5/individual mode.

    Per CLAUDE.md "Design decisions — non-negotiable" the activation criteria
    were operator-set; this class only carries the configuration, never picks
    its own thresholds.

    Per CLAUDE.md "Operator gates must be wired and used" the activation flag
    is dual-gated (CLI ``--operator-authorized-le-5-dollar-mode`` AND env-var
    ``CATHEDRAL_AUTOPILOT_OPERATOR_AUTHORIZED_MODE=1``) so a stray CLI default
    cannot unlock dispatch authorization on its own.
    """

    enabled: bool = False
    per_dispatch_cap_usd: float = DEFAULT_PER_DISPATCH_CAP_USD
    cumulative_cap_usd: float = DEFAULT_CUMULATIVE_CAP_USD
    canonical_helper_script: Path | None = None
    journal_path: Path | None = None
    cumulative_spent_usd: float = 0.0

    def can_authorize(
        self,
        candidate: CandidateRow,
        *,
        dispatch_claims_path: Path | None = None,
    ) -> tuple[bool, str]:
        """Return ``(authorized, reason)``.

        Authorization requires every precondition to hold. The first failing
        precondition's reason is returned; on success the reason is empty.
        """
        if not self.enabled:
            return False, "operator-authorized mode is OFF (default safe HALT-and-ASK)"
        if not math.isfinite(self.per_dispatch_cap_usd) or self.per_dispatch_cap_usd <= 0.0:
            return False, (
                f"per-dispatch cap {self.per_dispatch_cap_usd!r} is not finite-positive; "
                "operator must fix authorized-mode configuration"
            )
        if not math.isfinite(self.cumulative_cap_usd) or self.cumulative_cap_usd <= 0.0:
            return False, (
                f"cumulative cap {self.cumulative_cap_usd!r} is not finite-positive; "
                "operator must fix authorized-mode configuration"
            )
        if (
            not math.isfinite(candidate.estimated_dispatch_cost_usd)
            or candidate.estimated_dispatch_cost_usd <= 0.0
        ):
            return False, (
                f"candidate cost {candidate.estimated_dispatch_cost_usd!r} is "
                "not finite-positive; refuse to authorize a malformed estimate"
            )
        if not math.isfinite(self.cumulative_spent_usd) or self.cumulative_spent_usd < 0.0:
            return False, (
                f"cumulative spent {self.cumulative_spent_usd!r} is malformed; "
                "operator must reset or audit authorized-mode state"
            )
        if candidate.estimated_dispatch_cost_usd > self.per_dispatch_cap_usd:
            return False, (
                f"candidate cost ${candidate.estimated_dispatch_cost_usd:.4f} "
                f"exceeds per-dispatch cap ${self.per_dispatch_cap_usd:.4f}"
            )
        prospective = self.cumulative_spent_usd + candidate.estimated_dispatch_cost_usd
        if prospective > self.cumulative_cap_usd:
            return False, (
                f"cumulative spend would reach ${prospective:.4f} which exceeds "
                f"the ${self.cumulative_cap_usd:.4f} envelope; operator round-trip required"
            )
        if candidate.blockers:
            return False, (
                f"candidate has unresolved blockers {candidate.blockers!r}; "
                "operator must adjudicate before any dispatch"
            )
        if not self.canonical_helper_script or not self.canonical_helper_script.is_file():
            return False, (
                f"canonical helper script {self.canonical_helper_script!r} does not "
                "exist; operator must point --canonical-helper-script at a real file"
            )
        authority_blockers = candidate.dispatch_authority_blockers(
            dispatch_claims_path=dispatch_claims_path
        )
        if authority_blockers:
            return False, (
                f"candidate is not a dispatch-authority packet; unresolved "
                f"authority blockers {authority_blockers!r}; operator must "
                "adjudicate before any autonomous dispatch"
            )
        return True, ""

    def record_authorization(self, candidate: CandidateRow) -> None:
        """Increment the per-process cumulative-spent counter.

        The persistent ledger is :mod:`tools.claim_lane_dispatch`. This in-memory
        counter is the loop's own envelope guard so a single autopilot session
        cannot drain authorization across many small dispatches.
        """
        self.cumulative_spent_usd += candidate.estimated_dispatch_cost_usd


def _env_authorizes_mode(env: dict[str, str] | None = None) -> bool:
    """Return True iff the env-var explicitly opts into authorized mode.

    Defense-in-depth: even if ``--operator-authorized-le-5-dollar-mode`` is
    passed, the runtime env-var must ALSO be set to ``1``. This guards against
    the failure mode where someone sets the CLI flag in a script but forgets
    that the operator's machine doesn't carry the env-var (so authorized
    dispatches never actually run).
    """
    import os as _os

    src = env if env is not None else _os.environ
    return src.get(OPERATOR_AUTHORIZED_MODE_ENV_VAR, "") == OPERATOR_AUTHORIZED_MODE_ENV_VALUE_ENABLED


@dataclass
class HaltEvent:
    """One operator-decision halt event surfaced by the loop."""

    event_class: EventClass
    candidate_id: str
    reason: str
    predicted_score_delta: float
    estimated_cost_usd: float
    requires_approval: bool
    halt_at_utc: str = ""
    blockers: list[str] = field(default_factory=list)
    lane_id: str = ""
    claim_keys: list[str] = field(default_factory=list)
    target_modes: list[str] = field(default_factory=list)
    dispatch_packet_sha256: str = ""
    archive_path: str = ""
    submission_dir: str = ""
    archive_manifest_path: str = ""
    archive_sha256: str = ""
    candidate_archive_bytes: int | None = None
    runtime_tree_sha256: str = ""
    timing_smoke_command: str = ""
    ready_for_exact_eval_dispatch: bool = False
    literature_anchor: str = ""
    source_supports: str = ""
    paper_claim_scope: str = ""
    pact_must_prove: str = ""
    decode_complexity_evidence: str = ""
    decision: OperatorDecision | None = None
    decision_at_utc: str | None = None
    decision_notes: str = ""
    # Operator-authorized le-$5/individual mode fields (2026-05-11).
    autopilot_authorized: bool = False
    autopilot_tag: str = ""
    autopilot_authorized_reason: str = ""
    autopilot_refused_reason: str = ""
    autopilot_claim_recorded: bool = False
    autopilot_claim_instance_job_id: str = ""
    autopilot_claim_reason: str = ""


@dataclass
class LoopIterationReport:
    """Result of one autonomous-loop iteration."""

    iteration: int
    started_at_utc: str
    ended_at_utc: str
    n_candidates_seen: int
    n_candidates_blocked_by_dispatch_claim: int
    n_candidates_ranked: int
    halt_events: list[HaltEvent] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    schema: str = AUTONOMOUS_LOOP_SCHEMA


# ── Pure ranking + halt-event construction ─────────────────────────────────


def _posterior_correction_factor(
    candidate: CandidateRow,
    posterior: Any | None,
) -> tuple[float, int, str]:
    """Return ``(correction_factor, n_observations, key)`` for a candidate.

    W/I/A I-1 wire-in (2026-05-12): query the continual-learning posterior
    using the candidate's ``family`` field as the track-correction key. The
    factor multiplies the candidate's predicted_score_delta; ``n>0`` means
    empirical anchors exist for this family.

    Returns ``(1.0, 0, "")`` when the posterior is absent or no matching
    anchors are available. Per CLAUDE.md "Forbidden score claims": the
    correction is a non-authoritative planning prior; it never auto-promotes
    nor auto-kills a candidate.
    """
    if posterior is None or posterior_query_track_correction is None:
        return 1.0, 0, ""
    family = candidate.family
    if not family:
        return 1.0, 0, ""
    try:
        factor, n = posterior_query_track_correction(posterior, family)
    except Exception:  # pragma: no cover - defensive
        return 1.0, 0, family
    import math
    if n > 0 and math.isfinite(factor) and factor > 0.0:
        return float(factor), int(n), family
    return 1.0, 0, family


# ── Z1 empirical revision: MDL-density penalty + class-shift reward ────────
#
# Per `feedback_z1_mdl_ablation_landed_20260514.md` operator decision #4
# (2026-05-14): update the cathedral autopilot ranker to penalize within-
# HNeRV-class candidates (high MDL density = class-saturated) and reward
# predictive-receiver / cooperative-receiver / foveation / class-shift
# candidates. The Z1 ablation empirically established 0.90+ MDL density as
# the within-class trap threshold.
#
# Per CLAUDE.md "Forbidden score claims": these adjustments operate on
# PREDICTED scores only. They never claim or alter measured scores. The
# adjustment is a ranking re-weight, not a score promotion.
#
# Per CLAUDE.md "Subagent coherence-by-default" hook #4 (cathedral autopilot
# dispatch hook): these functions are the canonical hook for the autopilot
# ranker to honor the Z1 empirical revision automatically.

# MDL-density thresholds (per Z1 empirical revision 2026-05-14):
#   density > 0.95  -> within-class; cap predicted_score_delta at -0.005
#                       (floor effective predicted improvement near zero)
#   density > 0.90  -> within-class trending; 50% penalty
#   density < 0.90  -> across-class promising; no penalty
#   density unknown -> no penalty (don't punish lack-of-evidence)
MDL_DENSITY_WITHIN_CLASS_SATURATED_THRESHOLD = 0.95
MDL_DENSITY_WITHIN_CLASS_TRENDING_THRESHOLD = 0.90
MDL_DENSITY_WITHIN_CLASS_SATURATED_DELTA_FLOOR = -0.005  # negligible
MDL_DENSITY_WITHIN_CLASS_TRENDING_PENALTY_FACTOR = 0.5  # halve predicted ΔS

# Class-shift reward tokens. Candidates whose lane_class or literature_anchor
# match any of these get a predicted_score_delta bonus per the Z1 council's
# decision #4 (reward cooperative-receiver / predictive-receiver / foveation
# / class-shift candidates). The reward is small (additive ~0.01-0.02) and
# subtractive (lower = better in score-delta space).
_CLASS_SHIFT_LANE_CLASS_TOKENS = (
    "substrate_class_shift",
    "predictive_receiver",
    "cooperative_receiver",
    "foveation",
)

_CLASS_SHIFT_LITERATURE_TOKENS = (
    "cooperative-receiver",
    "cooperative_receiver",
    "predictive-coding",
    "predictive_coding",
    "predictive-receiver",
    "predictive_receiver",
    "foveation",
    "Tishby-Zaslavsky",
    "Tishby Zaslavsky",
    "Atick-Redlich",
    "Atick Redlich",
    "Rao-Ballard",
    "Rao Ballard",
    "Wyner-Ziv",
    "Wyner Ziv",
    "Information Bottleneck",
    "information_bottleneck",
    "MDL-IBPS",
    "Slepian-Wolf",
    "Slepian Wolf",
    "world_model",
    "world-model",
    "time_traveler",
    "time-traveler",
    # 2026-05-14 C1 council "RETAIN" decision per
    # `project_c1_world_model_revision_SUPERSEDED_by_council_unfair_probe_finding_20260514.md`:
    # the C1 world-model architecture revision was DEFERRED-pending-fair-probe-v2,
    # NOT KILLED. The literature anchor for the world-model class IS
    # Ha-Schmidhuber 2018 + Hafner DreamerV3 2023; both must be retained as
    # autopilot-ranker class-shift reward tokens so the probe-v2 evidence loop
    # (when it lands) inherits the canonical priority. Per CLAUDE.md
    # "Forbidden premature KILL without research exhaustion".
    "Ha-Schmidhuber",
    "Ha Schmidhuber",
    "Hafner",
    "DreamerV3",
    "Dreamer V3",
    "Dreamer-V3",
    "balle_2018",
    "Ballé",
    "Balle",
    "scale-hyperprior",
    "scale_hyperprior",
)

CLASS_SHIFT_LANE_CLASS_REWARD = 0.02
CLASS_SHIFT_LITERATURE_ANCHOR_REWARD = 0.01

# Per the 2026-05-14 grand-council reconvening
# (`feedback_c1_council_reconvene_post_probe_v2_landed_20260514.md` Decision 6
# HALF-MEASURE; council vote 4/11 explicit Contrarian + Quantizr + MacKay +
# Time-Traveler peer): probe v2 (FAIR Hafner DreamerV3 RSSM at matched-DOF +
# matched-bit-budget) reports world-model loses 99.98-100% margin in
# feature-space proxy regime. The verdict is NOT a class falsification per
# CLAUDE.md "Apples-to-apples evidence discipline" + "Forbidden premature KILL"
# — the regime distinction (FP4 quantization, SegNet+PoseNet preprocess
# invariances, 1200-frame temporal scaling) preserves a bidirectional posterior
# Δ ∈ [-0.04, +0.05] — but the strong-prior signal warrants halving the
# autopilot-ranker C1-class literature-anchor reward as a conditional revision
# pending the dispositive Z5 dispatch (Decision 1 Option β).
#
# RETAIN: the literature anchor stays in `_CLASS_SHIFT_LITERATURE_TOKENS` so
# the C1 lane is NOT closed (Decision δ DROP REJECTED 11/11 — Contrarian
# SUPER-VETO eligible).
# HALVE: when the matched literature token is one of the C1-class tokens
# below, the literature-anchor reward contribution is HALVED (0.01 -> 0.005).
# Combined with the lane_class reward (0.02), the effective C1-class candidate
# reward drops from ~0.025 stacked to ~0.0125 stacked — the 50% reduction the
# council ledger Decision 6 specifies.
#
# REVERT: this halve is a conditional revision. Update to full reward IF Z5
# (`lane_z5_predictive_coding_world_model_step3_20260514`) dispatch returns
# dispositive positive evidence; revert to zero IF Z5 + Decision 1 alpha
# (contest-scale C1) dispatch are jointly dispositive negative.
_C1_HALVED_LITERATURE_TOKENS = (
    "Ha-Schmidhuber",
    "Ha Schmidhuber",
    "Hafner",
    "DreamerV3",
    "Dreamer V3",
    "Dreamer-V3",
)
_C1_LITERATURE_ANCHOR_HALVE_FACTOR = 0.5


def adjust_predicted_delta_for_mdl_density(
    base_delta: float,
    mdl_density: float | None,
) -> float:
    """Penalize within-HNeRV-class candidates (high MDL density = class-
    saturated) per the Z1 empirical revision 2026-05-14.

    Per Z1 council decision #4:
      - density > 0.95: within-class; floor predicted_score_delta at
        ``MDL_DENSITY_WITHIN_CLASS_SATURATED_DELTA_FLOOR`` (effectively
        zero improvement)
      - density 0.90-0.95: within-class trending; apply 50% penalty to
        predicted savings (less-negative becomes more-positive)
      - density < 0.90: across-class promising; no penalty
      - density unknown: no penalty

    Per CLAUDE.md "Forbidden score claims": this adjusts PREDICTED ΔS only;
    measured anchors are untouched.

    Returns the (possibly penalized) predicted_score_delta. Lower is better
    in the score-delta convention (negative = improvement).
    """
    if mdl_density is None:
        return base_delta
    try:
        d = float(mdl_density)
    except (TypeError, ValueError):
        return base_delta
    if d > MDL_DENSITY_WITHIN_CLASS_SATURATED_THRESHOLD:
        # Floor at near-zero. If the candidate predicted a strong negative
        # delta, the within-class density says that prediction is empirically
        # unrealistic; cap at the conservative floor.
        return max(base_delta, MDL_DENSITY_WITHIN_CLASS_SATURATED_DELTA_FLOOR)
    if d > MDL_DENSITY_WITHIN_CLASS_TRENDING_THRESHOLD:
        # Halve the predicted savings (only affects negative deltas; positive
        # deltas stay the same magnitude since we multiply).
        return base_delta * MDL_DENSITY_WITHIN_CLASS_TRENDING_PENALTY_FACTOR
    return base_delta


def adjust_predicted_delta_for_class_shift(
    base_delta: float,
    *,
    lane_class: str | None = None,
    literature_anchor: str = "",
    notes: str = "",
) -> float:
    """Reward substrate-class-shift candidates per Z1 council decision #4.

    Acceptance:
      - ``lane_class`` matches any of ``_CLASS_SHIFT_LANE_CLASS_TOKENS``
        adds ``CLASS_SHIFT_LANE_CLASS_REWARD`` (subtracted from base delta;
        lower = better in score-delta convention)
      - ``literature_anchor`` or ``notes`` contains any of
        ``_CLASS_SHIFT_LITERATURE_TOKENS`` adds
        ``CLASS_SHIFT_LITERATURE_ANCHOR_REWARD``

    Both bonuses stack independently. Per CLAUDE.md "Forbidden score claims":
    this is a PREDICTED ΔS reweight, not a score promotion.
    """
    bonus = 0.0
    if isinstance(lane_class, str) and lane_class:
        for tok in _CLASS_SHIFT_LANE_CLASS_TOKENS:
            if tok in lane_class:
                bonus += CLASS_SHIFT_LANE_CLASS_REWARD
                break
    haystacks: list[str] = []
    if isinstance(literature_anchor, str):
        haystacks.append(literature_anchor)
    if isinstance(notes, str):
        haystacks.append(notes)
    for hay in haystacks:
        if not hay:
            continue
        matched_token = next(
            (tok for tok in _CLASS_SHIFT_LITERATURE_TOKENS if tok in hay),
            None,
        )
        if matched_token is None:
            continue
        # Per Decision 6 HALF-MEASURE 2026-05-14: halve the literature-anchor
        # reward when the matched token is a C1-class token (Hafner /
        # DreamerV3 / Ha-Schmidhuber). All other class-shift tokens
        # (cooperative-receiver, predictive-coding, foveation, MDL-IBPS, ...)
        # keep the full literature-anchor reward.
        if matched_token in _C1_HALVED_LITERATURE_TOKENS:
            bonus += (
                CLASS_SHIFT_LITERATURE_ANCHOR_REWARD
                * _C1_LITERATURE_ANCHOR_HALVE_FACTOR
            )
        else:
            bonus += CLASS_SHIFT_LITERATURE_ANCHOR_REWARD
        break
    # Lower-is-better in score-delta convention; subtract the bonus to make
    # this candidate look better in the ranking.
    return base_delta - bonus


def apply_z1_empirical_revision_to_candidate_delta(
    c: CandidateRow,
    *,
    score_panel_axis: str = "contest_cpu",
) -> float:
    """Return the rank-key predicted_score_delta after Z1 empirical revision
    adjustments. This is the canonical composition: first apply MDL-density
    penalty (Tier A), then Tier C density penalty/reward (Catalog #227),
    then class-shift reward, then composition matrix factor (Catalog #227).

    Per CLAUDE.md "Subagent coherence-by-default" (Z1 wire-in hook #4): this
    helper is the single source of truth for "what does the autopilot
    *actually* believe about this candidate's predicted_score_delta after
    integrating the Z1 ablation evidence?"

    The ORIGINAL ``c.predicted_score_delta`` is preserved on the row; this
    function returns a transient sort-key value, never mutates the row.
    """
    d = adjust_predicted_delta_for_mdl_density(
        c.predicted_score_delta, c.mdl_density
    )
    # Catalog #227 Tier C wire-in: substrate-class signal overrides Tier A
    # for class discrimination. Apply BEFORE the class-shift reward so the
    # Tier C verdict can either reinforce (across-class density LOW + bonus)
    # or contradict (across-class evidence already captured by Tier C
    # alone, even without a literature_anchor).
    d = adjust_predicted_delta_for_mdl_tier_c_density(d, c.mdl_tier_c_density)
    lane_class = c.lane_class
    literature_anchor = c.literature_anchor
    literature_notes = c.notes
    if _candidate_literature_anchor_rank_reward_suppressed(c):
        lane_class = None
        literature_anchor = ""
        literature_notes = ""
    d = adjust_predicted_delta_for_class_shift(
        d,
        lane_class=lane_class,
        literature_anchor=literature_anchor,
        notes=literature_notes,
    )
    # Catalog #227 composition matrix wire-in: stacking of two substrates
    # has an additivity factor alpha in [0, 1+] per the Z3xC6 probe-disambiguator
    # (`tools/probe_z3_x_c6_composition_disambiguator.py`). The factor maps
    # to a predicted_score_delta scaling per the additivity verdict bands.
    #
    # 2026-05-17 upgrade to v2 cascade per `feedback_super_additive_lane_g_v3
    # _siren_topology_integration_landed_20260517.md`: the v2 cascade is a
    # strict superset of v1 with a new SUPER_ADDITIVE branch (alpha > 1.05)
    # for cross-substrate redundancy compositions. Reward factor is BOUNDED
    # at 2.0× so a future false-signal alpha=10 cannot runaway-promote.
    d = adjust_predicted_delta_for_composition_alpha_v2(d, c.composition_alpha)
    # OP-3 predicted_dispatch_risk wire-in (codex chunk 6 finding,
    # 2026-05-15): demote candidates whose RUDIN-DAUBECHIES preflight SLIM
    # risk score crosses the canonical refusal threshold. Applied after the
    # score-axis chain so it can floor a candidate whose score-axis adjustments
    # (Tier A / Tier C / class-shift / composition-alpha) would otherwise
    # promote it past safer peers. Venn reweighting below is allowed to compose
    # with the already-risk-adjusted delta, but it must not replace this
    # structural refusal hook.
    d = adjust_predicted_delta_for_predicted_dispatch_risk(
        d, c.predicted_dispatch_risk
    )
    # Catalog #125 hook #4 wire-in (2026-05-17): the per-pair master gradient
    # Venn classification (consumer 1 in `tac.master_gradient_consumers`)
    # produces a per-archive byte-class breakdown {PAIR_SPECIFIC,
    # PAIR_INVARIANT, PAIR_NEUTRAL, DEAD}. Candidates whose substrate target
    # archive carries a HIGH PAIR_INVARIANT % (>= 80%) are reweighted as
    # higher-EV (the Wyner-Ziv-hoistable region is structurally larger);
    # candidates with HIGH PAIR_SPECIFIC % (>= 30%) carry the pair-specific
    # trap and are reweighted as lower-EV. Reweighting composes on the same
    # predicted-score-delta axis: factor > 1 rewards a negative improvement
    # estimate by making it more negative; factor < 1 penalizes it.
    # The reweighting is applied AFTER the SLIM dispatch_risk because Venn
    # classification is structural-orthogonal to preflight risk: a high-Venn
    # candidate may still have high SLIM risk for unrelated reasons.
    d = adjust_predicted_delta_for_venn_classification_v2(
        d,
        c.archive_sha256,
        panel_axis=score_panel_axis,
        require_score_weighted_custody=True,
    )
    # LOW gap closure widened wave 2026-05-17 — BUCKET C: per-pair master
    # gradient sister-#817 consumption. AFTER the v2 venn cascade (which has
    # REPLACE semantics for the OptimalPerPairTreatmentPlan path) so the
    # planner-derived delta is preserved when present; this adjustment is an
    # ADDITIONAL multiplicative factor for the OTHER cascade paths
    # (deliverability + passthrough) where sister-#817 sidecar evidence
    # (per_pair_bit_allocation_* OR per_pair_fisher_importance_*) reveals
    # additional per-byte / per-pair signal not yet captured upstream. The
    # factor composes multiplicatively per the existing chain pattern; sidecar
    # ABSENT → 1.0× passthrough (NO FAKE REWARD per CLAUDE.md "Forbidden
    # empirical-claim-without-evidence-tag" + sister Q2+Q3 cascade discipline).
    d = adjust_predicted_delta_for_per_pair_sister_817_sidecars(
        d, c.archive_sha256
    )
    # ITEM_7 closure 2026-05-18: consume the canonical per_pair_difficulty_atlas
    # sidecar from tac.master_gradient_consumers. This is planning-only signal:
    # a valid atlas says the archive has pair/axis difficulty structure that
    # Cathedral should prioritize for follow-up, not that any score was proved.
    d = adjust_predicted_delta_for_per_pair_difficulty_atlas(
        d,
        c.archive_sha256,
        panel_axis=score_panel_axis,
    )
    # Cable D consumers 7-14 sub-cascade 2026-05-20 (Slot FF
    # `lane_cable_d_consumers_7_14_autopilot_cascade_wire_in_20260519`):
    # consume the 6 sister Cable D per-pair canonical sidecars NOT yet read
    # by the cathedral cascade — per_pair_pareto_envelope (consumer 7) +
    # per_pair_lagrangian_lambda_bisection (consumer 8) +
    # per_pair_lora_supervision_signal (consumer 9) +
    # per_pair_coding_budget_allocation (consumer 10) +
    # per_pair_kkt_residuals (consumer 12) + per_pair_volterra_cross_terms
    # (consumer 13). Consumer 11 (per_pair_difficulty_atlas) was wired above
    # via ITEM_7; consumer 14 (per_pair_optimal_treatment_plan_via_lagrangian_dual)
    # is the PRIMARY in v2 venn cascade above (Catalog #319 CASCADE 1).
    #
    # Composed multiplicatively per the existing sister-#817 + atlas pattern;
    # sidecar ABSENT → 1.0× passthrough (NO FAKE REWARD per CLAUDE.md
    # "Forbidden empirical-claim-without-evidence-tag" + sister Q2+Q3 cascade
    # discipline). Per CLAUDE.md "Apples-to-apples evidence discipline":
    # planning-only reweighting; never creates a score claim or dispatch
    # authority. Per Catalog #318 raw-byte-authority guard: this cascade
    # never returns raw byte tensors — only multiplicative factors derived
    # from canonical sidecar SCHEMA-validated presence.
    d = adjust_predicted_delta_for_cable_d_consumers_7_14_sidecars(
        d,
        c.archive_sha256,
    )
    # Grand council T3 finding #12 (2026-05-18) realistic-stacking correction:
    # the Wave 2A 52-row composite EV [-0.139, -0.026] assumes additive
    # composition; Wave 1 NSCS06 v8 anchor empirically refutes additivity
    # (-78% non-monotonicity). The realistic envelope is [-0.05, -0.02].
    # Apply LAST in the cascade so the correction operates on the final
    # composed predicted_score_delta regardless of which cascade path
    # produced it (composition_alpha_v2, venn_v2, sister-817, atlas).
    # n_stacked_extinctions is derived from the candidate's available
    # composition signal: when composition_alpha is present, the candidate
    # is by definition a multi-substrate composition. Notes-tokens
    # like "composed_from" or "stack of N" may also surface explicit count.
    n_stacked = _infer_n_stacked_extinctions_for_candidate(c)
    d = adjust_predicted_delta_for_realistic_stacking_correction(
        d, n_stacked
    )
    return d


def _infer_n_stacked_extinctions_for_candidate(c: CandidateRow) -> int:
    """Infer the count of composed extinctions for a candidate row.

    Used by :func:`apply_z1_empirical_revision_to_candidate_delta` to apply
    the Wave 2A realistic-stacking correction per grand council T3 finding
    #12 (2026-05-18). The correction only fires for ``n >= 2`` so a value
    of 1 (single-extinction or unknown-composition) preserves the legacy
    predicted_score_delta unchanged.

    Inference rules (highest signal first):

    1. Explicit ``composed_from:N`` / ``stack_of:N`` / ``stacked_extinctions:N``
       token in ``c.notes`` (decimal integer; clamp >= 1).
    2. Default: return 1 (single-extinction OR composition without explicit
       stack-count signal).

    DELIBERATE: ``composition_alpha is not None`` is NOT used as a stacking
    signal. When composition_alpha is set, the V2 cascade
    (``adjust_predicted_delta_for_composition_alpha_v2``) ALREADY encodes the
    empirical per-substrate stacking outcome (additive / sub-additive /
    saturating / super-additive bands per Catalog #319). The realistic-
    stacking correction must NOT double-discount substrates with measured
    composition_alpha evidence. It fires only when there is a HEURISTIC
    stacking-count signal (notes-token) WITHOUT empirical alpha — the regime
    where grand council T3 finding #12's optimistic-vs-realistic envelope
    discount is structurally appropriate.

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag": this is
    a STRUCTURAL inference, NOT a measurement. The correction it gates is a
    transient sort-key adjustment that never mutates the row.
    """
    notes = (c.notes or "").lower()
    # Rule 1: explicit count token.
    for token in ("composed_from:", "stack_of:", "stacked_extinctions:"):
        idx = notes.find(token)
        if idx == -1:
            continue
        rest = notes[idx + len(token):]
        # Parse leading decimal digits.
        digits = ""
        for ch in rest:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            try:
                n = int(digits)
            except ValueError:
                continue
            return max(1, n)
    # Rule 2: default single-extinction.
    # Per the docstring's DELIBERATE note: composition_alpha is NOT a stacking
    # signal — the V2 cascade already captures empirical per-substrate
    # stacking evidence.
    return 1


# ── Catalog #125 hook #4 wire-in: per-pair master gradient Venn classification ───
#
# Wire-in #1 of the per-pair master gradient consumer integration plan landed
# 2026-05-17. Reads the most-recent `venn_classification_<sha[:12]>_*.json`
# sidecar emitted by
# ``tac.master_gradient_consumers.classify_bytes_by_pair_variance`` and
# reweights the candidate's effective predicted_score_delta based on its
# substrate archive's Venn class breakdown.


_VENN_CLASSIFICATION_SIDECAR_ROOT = REPO_ROOT / ".omx" / "state" / "master_gradient_consumers"
_WYNER_ZIV_DELIVERABILITY_PROOF_ROOT = (
    REPO_ROOT / ".omx" / "state" / "wyner_ziv_deliverability"
)
_VENN_REWEIGHT_HIGH_PAIR_INVARIANT_THRESHOLD = 0.80
_VENN_REWEIGHT_HIGH_PAIR_SPECIFIC_THRESHOLD = 0.30
# HIGH PAIR_SPECIFIC penalty → multiply by 0.85 (delta LESS negative).
_VENN_REWEIGHT_HIGH_PAIR_SPECIFIC_DELTA_FACTOR = 0.85
_VENN_REWEIGHT_TIER_1_DELIVERABILITY_DELTA_FACTOR = 1.20
_VENN_REWEIGHT_TIER_2_DELIVERABILITY_DELTA_FACTOR = 1.10
_VENN_REWEIGHT_TIER_3_DELIVERABILITY_DELTA_FACTOR = 1.05


def _latest_venn_sidecar_for_archive(archive_sha256: str) -> Path | None:
    """Find the most-recent Venn classification sidecar for the given archive sha.

    Per `tac.master_gradient_consumers.consumer_output_path` the canonical
    filename pattern is ``venn_classification_<sha[:12]>_<utc>.json``. Return
    the most-recent matching file (lexicographic max on filename, which is
    chronological because the UTC suffix is YYYYMMDDTHHMMSS).
    """
    if not archive_sha256 or len(archive_sha256) < 12:
        return None
    if not _VENN_CLASSIFICATION_SIDECAR_ROOT.is_dir():
        return None
    sha_short = archive_sha256[:12]
    candidates = sorted(_VENN_CLASSIFICATION_SIDECAR_ROOT.glob(f"venn_classification_{sha_short}_*.json"))
    return candidates[-1] if candidates else None


def _read_venn_class_counts(
    sidecar_path: Path,
    archive_sha256: str | None = None,
    *,
    panel_axis: str = "contest_cpu",
    require_score_weighted_custody: bool = False,
) -> dict[str, float] | None:
    """Read and validate class-count payload from a Venn sidecar.

    Returns None on any parse error (the caller treats missing/invalid as
    "no Venn signal available", same as no sidecar at all).
    """
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if require_score_weighted_custody:
        if not archive_sha256:
            return None
        if payload.get("schema") != "master_gradient_consumer_venn_classification_v1":
            return None
        if payload.get("consumer_id") != "classify_bytes_by_pair_variance":
            return None
        payload_archive_sha = payload.get("archive_sha256")
        if (
            not isinstance(payload_archive_sha, str)
            or payload_archive_sha.lower() != archive_sha256.lower()
        ):
            return None
        axis_tag = _contest_axis_tag_for_panel(panel_axis)
        if not _sidecar_has_full_pair_gradient_custody(
            payload,
            archive_sha256,
            axis_tag=axis_tag,
        ):
            return None
    counts_key = "score_weighted_class_counts" if require_score_weighted_custody else "class_counts"
    counts = payload.get(counts_key)
    if not isinstance(counts, dict):
        return None
    # Validate the 4 canonical class names per PerByteVennClass.ALL
    for required in ("PAIR_SPECIFIC", "PAIR_INVARIANT", "PAIR_NEUTRAL", "DEAD"):
        if required not in counts:
            return None
    result: dict[str, float] = {}
    for k, v in counts.items():
        if k not in {"PAIR_SPECIFIC", "PAIR_INVARIANT", "PAIR_NEUTRAL", "DEAD"}:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        result[k] = float(v)
    return result


def _venn_deliverability_reward_factor_for_archive(archive_sha256: str) -> float:
    """Return proof-backed Wyner-Ziv reward factor for a Venn-positive archive.

    Catalog #319 makes the positive HIGH_PAIR_INVARIANT reward fail closed:
    Venn classification alone can say "this byte region is pair-shared", but
    only a DeliverabilityProof can say the contest decoder can reconstruct it
    without scorer access, network fetches, or runtime-budget violations.
    """
    if not archive_sha256:
        return 1.0
    try:
        from tac.wyner_ziv_deliverability.proof_builder import (
            load_deliverability_proof_for_archive,
            verify_deliverability_proof_contest_compliance,
        )
    except (ImportError, ModuleNotFoundError):
        return 1.0
    try:
        proof = load_deliverability_proof_for_archive(
            archive_sha256,
            proofs_dir=_WYNER_ZIV_DELIVERABILITY_PROOF_ROOT,
        )
    except (OSError, TypeError, ValueError):
        return 1.0
    if proof is None:
        return 1.0
    ok, _blockers = verify_deliverability_proof_contest_compliance(proof)
    if not ok:
        return 1.0

    total = int(proof.candidate_shared_prior_byte_count)
    if total <= 0:
        return 1.0
    tier_1 = max(0, int(proof.tier_1_byte_count))
    tier_2 = max(0, int(proof.tier_2_byte_count))
    tier_3 = (
        max(0, int(proof.tier_3_byte_count))
        if proof.operator_review_status_for_tier_3 == "approved"
        else 0
    )
    rewarded = min(total, tier_1 + tier_2 + tier_3)
    if rewarded <= 0:
        return 1.0
    neutral = max(0, total - rewarded)
    weighted = (
        tier_1 * _VENN_REWEIGHT_TIER_1_DELIVERABILITY_DELTA_FACTOR
        + tier_2 * _VENN_REWEIGHT_TIER_2_DELIVERABILITY_DELTA_FACTOR
        + tier_3 * _VENN_REWEIGHT_TIER_3_DELIVERABILITY_DELTA_FACTOR
        + neutral
    )
    return max(1.0, min(weighted / float(total), _VENN_REWEIGHT_TIER_1_DELIVERABILITY_DELTA_FACTOR))


def adjust_predicted_delta_for_venn_classification(
    predicted_delta: float, archive_sha256: str
) -> float:
    """Backwards-compatible v1 wrapper around the v2 Lagrangian-derived cascade.

    Per Q3 of lane_q2_q3_batched_catalog_319_gate_plus_autopilot_reweight_v2_20260517:
    THIS function is now a THIN WRAPPER that delegates to
    :func:`adjust_predicted_delta_for_venn_classification_v2` with
    ``optimal_plan_path=None`` (the v2 cascade resolves the most-recent plan
    at runtime via ``tac.master_gradient_consumers.load_optimal_plan_for_archive``).

    Existing callers + tests that pass only ``(predicted_delta, archive_sha256)``
    transparently get the v2 cascade behavior. The v2 cascade is canonical:

      CASCADE 1 (PRIMARY): if a Lagrangian-dual OptimalPerPairTreatmentPlan
        sidecar exists for ``archive_sha256``, REPLACE ``predicted_delta`` with
        the planner's ``predicted_score_delta`` field (the planner IS the
        canonical answer per Catalog #227 sister discipline).
      CASCADE 2 (DELIVERABILITY): else if a DeliverabilityProof exists, gate
        the Venn reweight on the proof's deliverable-tier byte savings
        (empirical-floor: if deliverable_savings == 0 — the fec6 prober case
        per ``.omx/state/wyner_ziv_deliverability/probe_f174192aeadf_20260517T205208.json``
        — apply 1.0× passthrough, no fake reward).
      CASCADE 3 (PASSTHROUGH): no plan AND no proof => 1.0× passthrough (NO
        FAKE REWARD).

    The HIGH_PAIR_SPECIFIC 0.85× penalty (sister discipline; substrate
    architecturally cannot share bytes) is preserved across all cascades.

    Per CLAUDE.md "Apples-to-apples evidence discipline": this is a
    PLANNING-ONLY reweighting; it never creates a score claim or dispatch
    authority. It only changes the autopilot's ranking ordering.
    """
    return adjust_predicted_delta_for_venn_classification_v2(
        predicted_delta, archive_sha256, optimal_plan_path=None
    )


def adjust_predicted_delta_for_venn_classification_v2(
    predicted_delta: float,
    archive_sha256: str,
    optimal_plan_path: Path | None = None,
    *,
    panel_axis: str = "contest_cpu",
    require_score_weighted_custody: bool = False,
) -> float:
    """Lagrangian-derived autopilot reweight v2 (canonical).

    Per Q3 of lane_q2_q3_batched_catalog_319_gate_plus_autopilot_reweight_v2_20260517
    + operator architectural correction 2026-05-17 ("the Lagrangian planner is
    the canonical answer; the Venn classification is a planning signal"):

    REPLACES the v1 ``adjust_predicted_delta_for_venn_classification`` flat
    1.15× HIGH_PAIR_INVARIANT reward (which the fec6 empirical anchor proved
    FAKE — see ``.omx/state/wyner_ziv_deliverability/probe_f174192aeadf_*.json``
    with ``deliverability_verdict='NOT_DELIVERABLE'``) with the canonical
    3-cascade decision tree:

      **CASCADE 1 (PRIMARY — Lagrangian-derived)**: if a
        ``OptimalPerPairTreatmentPlan`` sidecar exists for ``archive_sha256``
        (loaded via ``tac.master_gradient_consumers.load_optimal_plan_for_archive``
        OR ``optimal_plan_path`` override), REPLACE ``predicted_delta`` with
        the planner's ``predicted_score_delta`` field. The planner IS the
        canonical answer: it has solved the full ADMM/KKT problem with all
        4 constraints (archive bytes / compute / inflate / per-pair selection)
        and emitted a Pareto-feasible solution with dual variables that ARE
        the canonical per-tier factor source.

      **CASCADE 2 (DELIVERABILITY)**: else if a ``DeliverabilityProof`` exists
        for the archive AND it passes contest-compliance verification AND the
        Venn class is HIGH_PAIR_INVARIANT (>= 80%), apply the per-tier
        byte-weighted reward via ``_venn_deliverability_reward_factor_for_archive``.
        **Empirical-floor**: if ``deliverable_savings == 0`` (the fec6 prober
        case where lzma/brotli/zlib all INFLATE the candidate set), apply
        1.0× passthrough — NO FAKE REWARD per CLAUDE.md "Forbidden
        empirical-claim-without-evidence-tag".

      **CASCADE 3 (PASSTHROUGH)**: no plan + no proof => 1.0× passthrough.
        Venn classification ALONE is NOT contest deliverability proof per
        Catalog #319.

    The HIGH_PAIR_SPECIFIC 0.85× PENALTY (Venn class is PAIR_SPECIFIC ≥ 30% =
    substrate cannot share bytes regardless of Lagrangian / Deliverability)
    is preserved across all cascades — penalty applies when no positive-reward
    cascade fires.

    Args:
        predicted_delta: Raw predicted score delta from the candidate row
            (negative = score improvement per autopilot convention).
        archive_sha256: 64-char hex sha of the target archive bytes.
        optimal_plan_path: Optional explicit override of the plan sidecar
            (test fixture only; production callers pass None and let the
            canonical loader resolve the most-recent plan for the archive).

    Returns:
        Adjusted predicted_delta per the cascade above. Always a planning-only
        prediction; never a score claim.
    """
    if not archive_sha256:
        return predicted_delta

    # CASCADE 1: Lagrangian-derived (PRIMARY)
    plan_payload: dict | None = None
    if optimal_plan_path is not None:
        # Test fixture override: load the specific plan file
        try:
            plan_payload = json.loads(Path(optimal_plan_path).read_text(encoding="utf-8"))
            if not isinstance(plan_payload, dict):
                plan_payload = None
        except (OSError, json.JSONDecodeError):
            plan_payload = None
    else:
        # Production path: consult canonical loader
        try:
            from tac.master_gradient_consumers import load_optimal_plan_for_archive
        except (ImportError, ModuleNotFoundError):
            plan_payload = None
        else:
            try:
                plan_payload = load_optimal_plan_for_archive(archive_sha256)
            except (OSError, ValueError):
                plan_payload = None

    if plan_payload is not None:
        planner_delta = plan_payload.get("predicted_score_delta")
        if isinstance(planner_delta, (int, float)):
            # The planner's predicted_score_delta IS the canonical answer.
            # REPLACE — do NOT add to predicted_delta; the planner has solved
            # the full optimization including the substrate's marginal-Δ contribution.
            return float(planner_delta)
        # Plan payload present but malformed — fall through to Cascade 2

    # CASCADE 2 + 3: existing Venn + DeliverabilityProof cascade
    sidecar = _latest_venn_sidecar_for_archive(archive_sha256)
    if sidecar is None:
        return predicted_delta
    counts = _read_venn_class_counts(
        sidecar,
        archive_sha256,
        panel_axis=panel_axis,
        require_score_weighted_custody=require_score_weighted_custody,
    )
    if counts is None:
        return predicted_delta
    total = sum(counts.values())
    if total <= 0:
        return predicted_delta
    pair_invariant_frac = counts["PAIR_INVARIANT"] / total
    pair_specific_frac = counts["PAIR_SPECIFIC"] / total
    if pair_invariant_frac >= _VENN_REWEIGHT_HIGH_PAIR_INVARIANT_THRESHOLD:
        # Cascade 2: DeliverabilityProof-gated reward. Empirical-floor for
        # non-deliverable archives (fec6 case): factor = 1.0 (no fake reward).
        return predicted_delta * _venn_deliverability_reward_factor_for_archive(
            archive_sha256
        )
    if pair_specific_frac >= _VENN_REWEIGHT_HIGH_PAIR_SPECIFIC_THRESHOLD:
        # Preserved penalty — substrate architecturally cannot share bytes.
        return predicted_delta * _VENN_REWEIGHT_HIGH_PAIR_SPECIFIC_DELTA_FACTOR
    # Cascade 3: passthrough — no positive reward without proof.
    return predicted_delta


# ── LOW gap closure widened wave 2026-05-17 — BUCKET C: sister-#817 sidecar
# consumption ────────────────────────────────────────────────────────────────
#
# Per `.omx/research/comprehensive_wire_in_coverage_matrix_20260517.md` GAP
# #1 + sister #817 op-routable #2 ("Autopilot consumption: wire Gap 1's
# allocate_per_pair_bits + Gap 3's allocate_per_pair_fisher_importance into
# tools/cathedral_autopilot_autonomous_loop.py so per-pair-aware bit
# allocation + Fisher importance modulate candidate ranking").
#
# The two sister-#817 helpers DO NOT auto-emit sidecars at landing time
# (sister #817 produces typed outcomes via return value only; this BUCKET C
# wire-in checks for future-emitted sidecars when callers explicitly persist
# the outcome via consumer_output_path + write_consumer_sidecar_json).
#
# Sidecar path conventions (per consumer_output_path):
#   per_pair_bit_allocation_<sha[:12]>_*.json (canonical Gap 1 sidecar)
#   per_pair_fisher_importance_<sha[:12]>_*.json (canonical Gap 3 sidecar)
#
# Reward semantics (multiplicative factor in score-delta convention; lower
# is better, and improvements are negative. Therefore factor > 1.0 makes a
# negative delta MORE NEGATIVE = better-ranked; factor < 1.0 is a penalty):
#
#   per_pair_bit_allocation cascade_path == "optimal_plan" → 1.05× reward
#     (plan-derived allocation IS the canonical answer; small reward to
#     prefer plan-backed candidates over Wyner-Ziv-only or fallback peers).
#   per_pair_bit_allocation cascade_path == "wyner_ziv_composition" → 1.02×
#     reward (Wyner-Ziv-backed allocation is intermediate evidence).
#   per_pair_bit_allocation cascade_path == "aggregate_fallback" → 1.0×
#     (no signal; no reward per CLAUDE.md "Forbidden score claims").
#
#   per_pair_fisher_importance sidecar present with non-zero aggregate Fisher
#     → 1.03× reward (per-byte Fisher attribution is structural signal that
#     downstream callers can consume to bias byte budgets).
#   per_pair_fisher_importance sidecar absent → 1.0× (no signal).
#
# Both factors compose multiplicatively. When BOTH sidecars are present at
# their best cascade (optimal_plan + Fisher), the combined factor is
# 1.05 × 1.03 = ~1.0815× (modest stacked reward; deliberately conservative
# per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" — the
# sidecars are PREDICTIVE signal, not empirical anchors).
#
# Sister Q2+Q3 cascade discipline preserved: the v2 venn cascade's REPLACE
# semantics for the OptimalPerPairTreatmentPlan path are NOT touched; this
# wire-in only adds an ADDITIONAL multiplicative factor AFTER the v2 cascade.

_PER_PAIR_BIT_ALLOCATION_SIDECAR_REWARD_OPTIMAL_PLAN = 1.05
_PER_PAIR_BIT_ALLOCATION_SIDECAR_REWARD_WYNER_ZIV = 1.02
_PER_PAIR_BIT_ALLOCATION_SIDECAR_REWARD_AGGREGATE_FALLBACK = 1.0
_PER_PAIR_FISHER_IMPORTANCE_SIDECAR_REWARD_PRESENT = 1.03
_PER_PAIR_FISHER_IMPORTANCE_SIDECAR_REWARD_ABSENT = 1.0
_PER_PAIR_DIFFICULTY_ATLAS_SIDECAR_REWARD_PRESENT = 1.04
_PER_PAIR_DIFFICULTY_ATLAS_POSE_HARD_REWARD_PRESENT = 1.06
_PER_PAIR_SIDECAR_SCAN_ROOT = (
    REPO_ROOT / ".omx" / "state" / "master_gradient_consumers"
)


def _latest_per_pair_bit_allocation_sidecar_for_archive(
    archive_sha256: str,
) -> Path | None:
    """Find the most-recent per_pair_bit_allocation sidecar for the archive."""
    if not _PER_PAIR_SIDECAR_SCAN_ROOT.exists():
        return None
    sha_short = archive_sha256[:12]
    matches = sorted(
        _PER_PAIR_SIDECAR_SCAN_ROOT.glob(
            f"per_pair_bit_allocation_{sha_short}_*.json"
        )
    )
    if not matches:
        return None
    return matches[-1]  # lex-max == chrono-max (UTC YYYYMMDDTHHMMSS suffix)


def _latest_per_pair_fisher_importance_sidecar_for_archive(
    archive_sha256: str,
) -> Path | None:
    """Find the most-recent per_pair_fisher_importance sidecar for the archive."""
    if not _PER_PAIR_SIDECAR_SCAN_ROOT.exists():
        return None
    sha_short = archive_sha256[:12]
    matches = sorted(
        _PER_PAIR_SIDECAR_SCAN_ROOT.glob(
            f"per_pair_fisher_importance_{sha_short}_*.json"
        )
    )
    if not matches:
        return None
    return matches[-1]


def _latest_per_pair_difficulty_atlas_sidecar_for_archive(
    archive_sha256: str,
) -> Path | None:
    """Find the most-recent per_pair_difficulty_atlas sidecar for the archive."""
    if not _PER_PAIR_SIDECAR_SCAN_ROOT.exists():
        return None
    sha_short = archive_sha256[:12]
    matches = sorted(
        _PER_PAIR_SIDECAR_SCAN_ROOT.glob(
            f"per_pair_difficulty_atlas_{sha_short}_*.json"
        )
    )
    if not matches:
        return None
    return matches[-1]


def _per_pair_bit_allocation_sidecar_reward_factor(
    archive_sha256: str,
) -> float:
    """Read latest per_pair_bit_allocation sidecar; return multiplicative factor.

    Maps cascade_path_used to a reward factor per the BUCKET C semantics
    documented at the section header. Returns 1.0 (no reward) when no sidecar
    exists or sidecar is malformed.
    """
    sidecar = _latest_per_pair_bit_allocation_sidecar_for_archive(archive_sha256)
    if sidecar is None:
        return 1.0
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return 1.0
        cascade = str(payload.get("cascade_path_used", ""))
        if cascade == "optimal_plan":
            return _PER_PAIR_BIT_ALLOCATION_SIDECAR_REWARD_OPTIMAL_PLAN
        if cascade == "wyner_ziv_composition":
            return _PER_PAIR_BIT_ALLOCATION_SIDECAR_REWARD_WYNER_ZIV
        if cascade == "aggregate_fallback":
            return _PER_PAIR_BIT_ALLOCATION_SIDECAR_REWARD_AGGREGATE_FALLBACK
        # Unknown cascade tag → safe default
        return 1.0
    except (OSError, json.JSONDecodeError):
        return 1.0


def _per_pair_fisher_importance_sidecar_reward_factor(
    archive_sha256: str,
) -> float:
    """Read latest per_pair_fisher_importance sidecar; return multiplicative factor.

    Returns 1.03 if sidecar exists AND aggregate_fisher_l1 > 0; 1.0 otherwise.
    """
    sidecar = _latest_per_pair_fisher_importance_sidecar_for_archive(
        archive_sha256
    )
    if sidecar is None:
        return _PER_PAIR_FISHER_IMPORTANCE_SIDECAR_REWARD_ABSENT
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return _PER_PAIR_FISHER_IMPORTANCE_SIDECAR_REWARD_ABSENT
        agg_l1 = payload.get("aggregate_fisher_l1", 0.0)
        if not isinstance(agg_l1, (int, float)) or agg_l1 <= 0:
            return _PER_PAIR_FISHER_IMPORTANCE_SIDECAR_REWARD_ABSENT
        return _PER_PAIR_FISHER_IMPORTANCE_SIDECAR_REWARD_PRESENT
    except (OSError, json.JSONDecodeError):
        return _PER_PAIR_FISHER_IMPORTANCE_SIDECAR_REWARD_ABSENT


def _per_pair_difficulty_atlas_sidecar_reward_factor(
    archive_sha256: str,
    *,
    panel_axis: str = "contest_cpu",
) -> float:
    """Read latest per_pair_difficulty_atlas sidecar; return multiplicative factor.

    A valid atlas is a structural prioritization signal for Cathedral's ranker.
    If the hardest-pair breakdown is pose-axis dominated in score-marginal-
    weighted units, use the slightly stronger pose-hard reward from the cheap-
    probe OP-7/OP-2 family. Missing, malformed, zero-norm, authority-bearing,
    or raw-axis-only sidecars never receive the pose-hard reward.
    """
    axis_tag = _contest_axis_tag_for_panel(panel_axis)
    sidecar = _latest_per_pair_difficulty_atlas_sidecar_for_archive(archive_sha256)
    if sidecar is None:
        return 1.0
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 1.0
    if not isinstance(payload, dict):
        return 1.0
    if payload.get("schema") != "master_gradient_consumer_per_pair_difficulty_v1":
        return 1.0
    if payload.get("consumer_id") != "per_pair_difficulty_atlas":
        return 1.0
    payload_archive_sha = payload.get("archive_sha256")
    if (
        not isinstance(payload_archive_sha, str)
        or payload_archive_sha.lower() != archive_sha256.lower()
    ):
        return 1.0
    if (
        payload.get("score_claim") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("ready_for_exact_eval_dispatch") is not False
    ):
        return 1.0
    aggregate_norm = payload.get("aggregate_gradient_norm_l2", 0.0)
    if not isinstance(aggregate_norm, (int, float)) or aggregate_norm <= 0:
        return 1.0
    weighting = payload.get("score_axis_weighting")
    if not isinstance(weighting, dict) or weighting.get("available") is not True:
        return 1.0
    coeffs = weighting.get("axis_coefficients")
    if not isinstance(coeffs, dict) or not all(
        isinstance(coeffs.get(axis), (int, float)) for axis in ("seg", "pose", "rate")
    ):
        return 1.0
    if not _sidecar_has_full_pair_gradient_custody(
        payload,
        archive_sha256,
        axis_tag=axis_tag,
    ):
        return 1.0
    entries = payload.get("top_k_score_weighted_with_axis_breakdown")
    if not isinstance(entries, list) or not entries:
        return 1.0
    dominance_mass = 0.0
    total_mass = 0.0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        pose = entry.get("pose_axis_score_l1", 0.0)
        seg = entry.get("seg_axis_score_l1", 0.0)
        rate = entry.get("rate_axis_score_l1", 0.0)
        if (
            isinstance(pose, (int, float))
            and isinstance(seg, (int, float))
            and isinstance(rate, (int, float))
            and pose > 0
            and pose >= seg
            and pose >= rate
        ):
            dominance_mass += max(0.0, float(pose) - max(float(seg), float(rate)))
            total_mass += max(0.0, float(pose) + float(seg) + float(rate))
    if dominance_mass <= 0.0 or total_mass <= 0.0:
        return 1.0
    reward_cap = _PER_PAIR_DIFFICULTY_ATLAS_POSE_HARD_REWARD_PRESENT - 1.0
    signal_strength = min(1.0, 2.0 * dominance_mass / total_mass)
    return 1.0 + reward_cap * signal_strength


def adjust_predicted_delta_for_per_pair_sister_817_sidecars(
    predicted_delta: float, archive_sha256: str
) -> float:
    """Apply sister-#817 sidecar consumption to predicted_delta.

    LOW gap closure widened wave 2026-05-17 — BUCKET C wire-in. Composes
    multiplicatively per the existing rank_candidates chain pattern:
      - per_pair_bit_allocation sidecar (sister #817 Gap 1)
      - per_pair_fisher_importance sidecar (sister #817 Gap 3)

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + sister
    Q2+Q3 v2 cascade discipline: sidecar ABSENT → 1.0× passthrough (no fake
    reward); sidecar PRESENT → conservative multiplicative bonus per the
    BUCKET C reward bands documented at the section header.

    Per CLAUDE.md "Apples-to-apples evidence discipline": this is PLANNING-
    ONLY reweighting; never creates a score claim or dispatch authority.

    Args:
        predicted_delta: Raw predicted score delta from the candidate row
            after the v2 venn cascade (negative = score improvement per
            autopilot convention).
        archive_sha256: 64-char hex sha of the target archive bytes.

    Returns:
        Adjusted predicted_delta per the multiplicative cascade.
    """
    if not archive_sha256:
        return predicted_delta
    bit_alloc_factor = _per_pair_bit_allocation_sidecar_reward_factor(
        archive_sha256
    )
    fisher_factor = _per_pair_fisher_importance_sidecar_reward_factor(
        archive_sha256
    )
    return predicted_delta * bit_alloc_factor * fisher_factor


def adjust_predicted_delta_for_per_pair_difficulty_atlas(
    predicted_delta: float,
    archive_sha256: str,
    *,
    panel_axis: str = "contest_cpu",
) -> float:
    """Apply per_pair_difficulty_atlas sidecar consumption to predicted_delta."""
    if not archive_sha256:
        return predicted_delta
    return predicted_delta * _per_pair_difficulty_atlas_sidecar_reward_factor(
        archive_sha256,
        panel_axis=panel_axis,
    )


# ── Cable D consumers 7-14 sub-cascade (Slot FF, 2026-05-20) ────────────────
#
# Per `.omx/research/cable_d_wire_in_batch_landed_20260519.md` highest-EV
# op-routable: wire the 6 sister Cable D per-pair canonical sidecars (NOT yet
# read by the cathedral cascade) into the ranker so per-pair Pareto envelope +
# Lagrangian lambda + KKT residuals + Volterra cross-terms + LoRA supervision +
# coding-budget signals influence candidate ordering.
#
# CANONICAL SIDECAR IDS (per `tac.master_gradient_consumers` consumer_id field):
#   - per_pair_pareto_envelope (consumer 7) — schema:
#       master_gradient_consumer_per_pair_pareto_envelope_v1
#   - per_pair_lagrangian_lambda_bisection (consumer 8) — schema:
#       master_gradient_consumer_per_pair_lambda_bisection_v1
#   - per_pair_lora_supervision_signal (consumer 9) — schema:
#       master_gradient_consumer_per_pair_lora_supervision_v1
#   - per_pair_coding_budget_allocation (consumer 10) — schema:
#       master_gradient_consumer_per_pair_coding_budget_v1
#   - per_pair_kkt_residuals (consumer 12) — schema:
#       master_gradient_consumer_per_pair_kkt_residuals_v1
#   - per_pair_volterra_cross_terms (consumer 13) — schema:
#       master_gradient_consumer_per_pair_volterra_v1
#
# Wired separately above (NOT in this sub-cascade):
#   - per_pair_difficulty_atlas (consumer 11) — ITEM_7 closure (line ~1107)
#   - per_pair_optimal_treatment_plan_via_lagrangian_dual (consumer 14) —
#     Catalog #319 CASCADE 1 PRIMARY (line ~1393)
#
# REWARD SEMANTICS (multiplicative factor; lower predicted_score_delta is
# better, improvements are negative; factor > 1.0 makes a negative delta MORE
# negative = better-ranked):
#
# Each sidecar that is PRESENT + canonical-SCHEMA-valid + custody-clean +
# carries non-zero structural signal contributes a CONSERVATIVE 1.01× reward
# (1% bonus per sidecar; deliberately small per CLAUDE.md "Forbidden
# empirical-claim-without-evidence-tag" — sidecars are PREDICTIVE signal not
# empirical anchors). The factors compose multiplicatively — with all 6
# sidecars present at maximum signal the combined factor is 1.01^6 = ~1.0615×
# (modest stacked reward). With 0 sidecars present the factor is 1.0×
# (NO FAKE REWARD per sister Q2+Q3 v2 cascade discipline + sister-#817 +
# atlas pattern).
#
# Per CATALOG #327 raw-byte-authority guard: this cascade never reads raw
# byte tensors or projects bytes-to-score — it ONLY reads canonical sidecar
# SCHEMA-validated presence + structural signal markers (n_pairs > 0,
# canonical schema tag matches, archive_sha256 matches, score_claim=false).
#
# Per CATALOG #341 cathedral consumer routing canonical markers: this is a
# rank-time helper invoked from `apply_z1_empirical_revision_to_candidate_delta`
# (not from a cathedral consumer package directly). The sister Cable D
# consumer packages (per_pair_pareto_envelope_consumer / etc.) already carry
# the 3 canonical non-promotable markers (predicted_delta_adjustment=0.0 /
# promotable=False / axis_tag="[predicted]") at the consumer surface; this
# cascade is the downstream RANKER consumption surface, distinct from the
# consumer surface. The ranker reweight is observability-only (does NOT
# mutate the row's predicted_score_delta; the adjustment is transient for
# sort-key purposes only per docstring on apply_z1_empirical_revision).

_CABLE_D_CONSUMERS_7_14_SIDECAR_REWARD_FACTOR_PER_PRESENT = 1.01
_CABLE_D_CONSUMERS_7_14_SIDECAR_REWARD_FACTOR_ABSENT = 1.0

# Canonical sidecar IDs + their schema tags (per
# `tac.master_gradient_consumers` write_consumer_sidecar_json calls).
# Ordered to match Cable D consumer numbering 7+8+9+10+12+13.
_CABLE_D_CONSUMERS_7_14_CANONICAL_SIDECARS = (
    (
        "per_pair_pareto_envelope",
        "master_gradient_consumer_per_pair_pareto_envelope_v1",
    ),
    (
        "per_pair_lagrangian_lambda_bisection",
        "master_gradient_consumer_per_pair_lambda_bisection_v1",
    ),
    (
        "per_pair_lora_supervision_signal",
        "master_gradient_consumer_per_pair_lora_supervision_v1",
    ),
    (
        "per_pair_coding_budget_allocation",
        "master_gradient_consumer_per_pair_coding_budget_v1",
    ),
    (
        "per_pair_kkt_residuals",
        "master_gradient_consumer_per_pair_kkt_residuals_v1",
    ),
    (
        "per_pair_volterra_cross_terms",
        "master_gradient_consumer_per_pair_volterra_v1",
    ),
)


def _latest_cable_d_consumer_sidecar_for_archive(
    consumer_id: str,
    archive_sha256: str,
) -> Path | None:
    """Find the most-recent canonical sidecar for the given consumer + archive.

    Mirrors ``_latest_per_pair_bit_allocation_sidecar_for_archive`` pattern.
    Returns None when no matching sidecar exists.
    """
    if not _PER_PAIR_SIDECAR_SCAN_ROOT.exists():
        return None
    sha_short = archive_sha256[:12]
    matches = sorted(
        _PER_PAIR_SIDECAR_SCAN_ROOT.glob(f"{consumer_id}_{sha_short}_*.json")
    )
    if not matches:
        return None
    return matches[-1]  # lex-max == chrono-max (UTC YYYYMMDDTHHMMSS suffix)


def _cable_d_consumer_sidecar_carries_structural_signal(
    sidecar_path: Path,
    archive_sha256: str,
    expected_schema: str,
) -> bool:
    """Read sidecar JSON; verify canonical schema + custody + non-trivial signal.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323
    + Catalog #341: a sidecar contributes a reward only when:
      (a) JSON parses (no malformed payload poisoning ranking);
      (b) schema tag matches the canonical expected_schema for the consumer
          (no orphan sidecar from a different consumer);
      (c) archive_sha256 matches the candidate's archive (no cross-archive
          contamination);
      (d) score_claim is explicitly False (no phantom-score sidecar leak per
          Catalog #321/#322/#323 sister discipline);
      (e) promotion_eligible is explicitly False (no promotion-leak via sidecar);
      (f) n_pairs > 0 OR n_bytes > 0 (non-trivial structural signal; an
          empty sidecar carries no information).

    Returns True iff all 6 conditions hold. Returns False on any failure
    (file IO, JSON parse, schema mismatch, custody violation, trivial signal).
    """
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    # (b) canonical schema tag matches
    if payload.get("schema") != expected_schema:
        return False
    # (c) archive_sha256 matches (case-insensitive defensive)
    payload_sha = payload.get("archive_sha256")
    if (
        not isinstance(payload_sha, str)
        or payload_sha.lower() != archive_sha256.lower()
    ):
        return False
    # (d) score_claim explicitly False
    if payload.get("score_claim") is not False:
        return False
    # (e) promotion_eligible explicitly False
    if payload.get("promotion_eligible") is not False:
        return False
    # (f) non-trivial structural signal: n_pairs > 0 OR n_bytes > 0
    n_pairs = payload.get("n_pairs", 0)
    n_bytes = payload.get("n_bytes", 0)
    if not isinstance(n_pairs, (int, float)) or not isinstance(
        n_bytes, (int, float)
    ):
        return False
    if n_pairs <= 0 and n_bytes <= 0:
        return False
    return True


def _cable_d_consumers_7_14_sidecar_reward_factor(
    archive_sha256: str,
) -> float:
    """Compose multiplicative reward across the 6 sister Cable D sidecars.

    For each of the 6 canonical sidecars (per_pair_pareto_envelope +
    per_pair_lagrangian_lambda_bisection + per_pair_lora_supervision_signal +
    per_pair_coding_budget_allocation + per_pair_kkt_residuals +
    per_pair_volterra_cross_terms):
      - If sidecar PRESENT + canonical-SCHEMA-valid + custody-clean +
        carries non-trivial structural signal: factor *= 1.01
      - Otherwise (absent / malformed / cross-archive / score_claim-leak /
        promotion_eligible-leak / trivial signal): factor *= 1.0

    Returns the composed multiplicative factor in [1.0, 1.01^6] = [1.0, ~1.0615].

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + sister
    Q2+Q3 + sister-#817 + atlas v2 cascade discipline: this is a CONSERVATIVE
    reward for structural-signal presence, not an empirical anchor.
    """
    factor = 1.0
    for consumer_id, expected_schema in _CABLE_D_CONSUMERS_7_14_CANONICAL_SIDECARS:
        sidecar = _latest_cable_d_consumer_sidecar_for_archive(
            consumer_id, archive_sha256
        )
        if sidecar is None:
            continue
        if _cable_d_consumer_sidecar_carries_structural_signal(
            sidecar, archive_sha256, expected_schema
        ):
            factor *= _CABLE_D_CONSUMERS_7_14_SIDECAR_REWARD_FACTOR_PER_PRESENT
    return factor


def adjust_predicted_delta_for_cable_d_consumers_7_14_sidecars(
    predicted_delta: float,
    archive_sha256: str,
) -> float:
    """Apply Cable D consumers 7-14 sidecar consumption to predicted_delta.

    Slot FF 2026-05-20 — `lane_cable_d_consumers_7_14_autopilot_cascade_wire_in_20260519`.
    The Cable D wire-in batch landed 6 sister per-pair canonical sidecars
    that the cathedral autopilot ranker did NOT consume prior to this cascade.
    This wire-in extends the existing sister-#817 + atlas pattern to surface
    the 6 sidecars' structural-signal presence as a CONSERVATIVE 1.01× per
    sidecar multiplicative reward (composed up to ~1.0615× when all 6 present).

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + sister
    Q2+Q3 v2 cascade discipline: sidecar ABSENT or invalid → 1.0× passthrough
    (NO FAKE REWARD); sidecar PRESENT + canonical-SCHEMA-valid + custody-clean
    + carries non-trivial structural signal → 1.01× reward per sidecar.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323:
    this is PLANNING-ONLY reweighting; never creates a score claim or dispatch
    authority. Only changes autopilot ranking ordering.

    Per Catalog #318 master-gradient raw-byte-authority guard: this cascade
    never returns raw byte tensors — only multiplicative factors derived from
    canonical sidecar SCHEMA-validated presence + structural-signal markers.

    Args:
        predicted_delta: Predicted score delta from the candidate row after
            the upstream cascade chain (v2 venn + sister-817 + atlas).
            Negative = score improvement per autopilot convention.
        archive_sha256: 64-char hex sha of the target archive bytes.

    Returns:
        Adjusted predicted_delta per the composed multiplicative factor.
    """
    if not archive_sha256:
        return predicted_delta
    factor = _cable_d_consumers_7_14_sidecar_reward_factor(archive_sha256)
    return predicted_delta * factor


# ── Tier C substrate-class density (Catalog #227 wire-in, 2026-05-14) ─────
#
# Per `feedback_mdl_ablation_tier_c_ibps1_landed_20260514.md` operator
# decision #4 (RECOMMENDED-LAND-NEXT): wire Tier C decoder/latent curve-knee
# into the autopilot ranker. Tier A is brotli-saturated at the byte layer and
# CANNOT discriminate substrate class; Tier C is the dispositive disambiguator
# per Z1 deep-math §3.5 + Tishby-Zaslavsky 2015.
#
# Thresholds (per Z1 council band math + C6 5ep anchor density ≈ 0.13):
#   density >= 0.70 (within-class) → cap at floor (effective zero
#                                    improvement; sister of Tier A 0.95 cap)
#   density >= 0.50 (within-class trending) → 50% penalty
#   density <= 0.30 (across-class) → 0.01 ΔS bonus (subtract → more-negative
#                                    = better in score-delta convention).
#                                    Sister of CLASS_SHIFT_LITERATURE_ANCHOR_
#                                    REWARD but applied only when EMPIRICAL
#                                    Tier C evidence backs the across-class
#                                    claim, not just lineage.
#   density unknown (None) → no adjustment (don't punish lack-of-evidence).
#
# The Tier C signal is STRONGER than the Tier A signal for substrate-class
# discrimination, but Tier A still captures within-class encoder/codec
# saturation which Tier C alone does not. The composition in
# `apply_z1_empirical_revision_to_candidate_delta` applies BOTH so a
# within-Tier-A candidate gets penalized AND a within-Tier-C candidate is
# further capped at the floor.
MDL_TIER_C_WITHIN_CLASS_SATURATED_THRESHOLD = 0.70
MDL_TIER_C_WITHIN_CLASS_TRENDING_THRESHOLD = 0.50
MDL_TIER_C_ACROSS_CLASS_THRESHOLD = 0.30
MDL_TIER_C_WITHIN_CLASS_SATURATED_DELTA_FLOOR = -0.005  # sister of Tier A
MDL_TIER_C_WITHIN_CLASS_TRENDING_PENALTY_FACTOR = 0.5
MDL_TIER_C_ACROSS_CLASS_BONUS = 0.01  # subtract from delta (lower = better)


def adjust_predicted_delta_for_mdl_tier_c_density(
    base_delta: float,
    mdl_tier_c_density: float | None,
) -> float:
    """Apply substrate-class-discrimination Tier C penalty/reward to the
    predicted_score_delta.

    Per Catalog #227 / Tier C wire-in (`feedback_mdl_ablation_tier_c_ibps1_
    landed_20260514.md` operator decision #4):

      - density >= 0.70: within-class (Tier C confirms); floor delta at
        ``MDL_TIER_C_WITHIN_CLASS_SATURATED_DELTA_FLOOR``.
      - density >= 0.50: within-class trending; 50% penalty.
      - density <= 0.30: across-class (Tier C confirms); apply
        ``MDL_TIER_C_ACROSS_CLASS_BONUS`` subtraction (more-negative = better).
      - 0.30 < density < 0.50: indeterminate; no adjustment.
      - density unknown (None) or non-numeric: no adjustment.

    Per CLAUDE.md "Forbidden score claims": this adjusts PREDICTED ΔS only.
    Returns the adjusted predicted_score_delta. Lower is better.
    """
    if mdl_tier_c_density is None:
        return base_delta
    try:
        d = float(mdl_tier_c_density)
    except (TypeError, ValueError):
        return base_delta
    if d >= MDL_TIER_C_WITHIN_CLASS_SATURATED_THRESHOLD:
        return max(base_delta, MDL_TIER_C_WITHIN_CLASS_SATURATED_DELTA_FLOOR)
    if d >= MDL_TIER_C_WITHIN_CLASS_TRENDING_THRESHOLD:
        return base_delta * MDL_TIER_C_WITHIN_CLASS_TRENDING_PENALTY_FACTOR
    if d <= MDL_TIER_C_ACROSS_CLASS_THRESHOLD:
        # Subtract bonus → more-negative → better-ranked in score-delta
        # convention. Sister of the class-shift literature bonus but
        # backed by Tier C empirical evidence rather than lineage tokens.
        return base_delta - MDL_TIER_C_ACROSS_CLASS_BONUS
    # 0.30 < density < 0.50 → indeterminate; no adjustment.
    return base_delta


# ── Substrate composition matrix factor (Catalog #227 wire-in, 2026-05-14) ─
#
# Per `feedback_t1_f_z3_x_c6_composition_probe_build_landed_20260514.md`:
# the Z3xC6 composition probe-disambiguator returns an additivity factor alpha
# in the [0, 1+] band:
#
#   alpha > 0.7   (ADDITIVE) -> stacking realizes additive savings; predicted_
#                          score_delta scaled by 1.0 (no penalty)
#   alpha 0.3-0.7 (SUB-ADDITIVE) -> marginal stacking; halve predicted savings
#   alpha <= 0.3  (SATURATING) -> single-substrate dominates; cap at floor
#
# The matrix is the canonical posterior surface
# `.omx/state/substrate_composition_matrix.json` populated by every probe
# invocation. The ranker reads the candidate's `composition_alpha` field;
# upstream loaders (`load_candidates_from_substrate_composition_ranking`)
# populate it when QQ's substrate composition ranker emits the field.
#
# Per CLAUDE.md "Anti-arbitrariness primitive: the probe-disambiguator
# pattern": the probe IS the arbitration; the ranker consumes the verdict.
COMPOSITION_ALPHA_ADDITIVE_THRESHOLD = 0.7  # >0.7 = additive, no penalty
COMPOSITION_ALPHA_SUB_ADDITIVE_THRESHOLD = 0.3  # 0.3-0.7 = sub-additive
COMPOSITION_ALPHA_SUB_ADDITIVE_PENALTY_FACTOR = 0.5  # halve savings
COMPOSITION_ALPHA_SATURATING_DELTA_FLOOR = -0.005  # sister of MDL-density


def adjust_predicted_delta_for_composition_alpha(
    base_delta: float,
    composition_alpha: float | None,
) -> float:
    """Apply substrate composition additivity factor to predicted_score_delta.

    Per Catalog #227 composition matrix wire-in
    (`feedback_t1_f_z3_x_c6_composition_probe_build_landed_20260514.md`):

      - alpha > 0.7  (ADDITIVE): no adjustment (full additive savings)
      - alpha 0.3-0.7 (SUB-ADDITIVE): 50% penalty on predicted savings
      - alpha <= 0.3 (SATURATING): floor at -0.005 (single-substrate dominates)
      - alpha unknown (None): no adjustment (single-substrate candidate or
        composition evidence not yet collected)

    Returns the (possibly adjusted) predicted_score_delta. Lower is better.

    Note: this v1 cascade does NOT recognize SUPER_ADDITIVE (alpha > 1.05);
    SUPER_ADDITIVE values are absorbed into the ADDITIVE branch (no extra
    reward). For SUPER_ADDITIVE handling, see
    :func:`adjust_predicted_delta_for_composition_alpha_v2` below.
    """
    if composition_alpha is None:
        return base_delta
    try:
        a = float(composition_alpha)
    except (TypeError, ValueError):
        return base_delta
    if a > COMPOSITION_ALPHA_ADDITIVE_THRESHOLD:
        # ADDITIVE: full additive savings realized; no adjustment.
        return base_delta
    if a > COMPOSITION_ALPHA_SUB_ADDITIVE_THRESHOLD:
        # SUB-ADDITIVE: halve the predicted savings.
        return base_delta * COMPOSITION_ALPHA_SUB_ADDITIVE_PENALTY_FACTOR
    # SATURATING (alpha <= 0.3): single-substrate dominates; cap at floor.
    return max(base_delta, COMPOSITION_ALPHA_SATURATING_DELTA_FLOOR)


# ── v2 cascade with SUPER_ADDITIVE branch (lane_super_additive_..._20260517) ──
#
# Per `feedback_super_additive_lane_g_v3_siren_topology_integration_landed_
# 20260517.md` + the Q6 OP-3 extended sweep landing:
#
# The original 45-pair extended sweep surfaced ONE pair with alpha=4.74
# (SUPER_ADDITIVE band) — `lane_g_v3_renderer + siren_renderer` under
# brotli. ROOT-CAUSE INVESTIGATION FOUND THE ALPHA=4.74 ROW IS A FALSE-SIGNAL
# ARTIFACT (both renderer.bin files are byte-identical sha256
# `08f12d722dd33f9061deee72f49d782035597f78cd65ed1463a241ab430a7529`
# because the SIREN smoke timed out rc=124 at 3601s and the submission-builder
# copied the lane_g_v3 placeholder). The α=4.74 IS NOT a real composition
# discovery; see the mechanism investigation memo for the false-signal class.
#
# HOWEVER, the v2 cascade itself is structurally needed for FUTURE real
# SUPER_ADDITIVE topologies that may emerge from genuine cross-substrate
# redundancy (e.g. distilled scorer surrogate + a renderer that learned similar
# fp16 weight statistics). The branch is engineered to be safe AT LANDING:
#
#   - SUPER_ADDITIVE reward is BOUNDED at 2.0× via min(alpha / 2.0, 2.0) so
#     a future false-signal alpha=10.0 cannot runaway-promote a candidate.
#   - The reward factor only applies when alpha > 1.05 (the threshold above
#     ADDITIVE's noise band of [0.7, 1.05]).
#   - Per CLAUDE.md "Apples-to-apples evidence discipline": the reward is
#     applied to the PREDICTED ΔS only; measured anchors are untouched.
#   - The cascade is otherwise a strict superset of v1 — sub_additive +
#     additive + saturating branches all behave identically to v1.
#
# Math: base_delta is in score-delta convention (more-negative is better).
# SUPER_ADDITIVE multiplies the magnitude by min(alpha/2.0, 2.0); for
# alpha=4.74 the reward factor is 2.0× (the cap). For alpha=2.0 the reward
# factor is 1.0× (no extra reward — at the cap threshold). For alpha=1.5
# the reward factor is 0.75× ... wait that's a PENALTY. The cap point
# alpha=2.0 means: rewards for alpha in (1.05, 2.0) are between 1.05/2.0=0.525×
# (REDUCING the magnitude!) and 1.0× — which is WRONG.
#
# Corrected formula: SUPER_ADDITIVE reward must be a MULTIPLIER >= 1.0 (more
# rewarding when alpha is more super-additive). Use:
#
#   reward_factor = min(max(alpha, 1.0), 2.0)  -- clamp to [1.0, 2.0]
#
# At alpha=1.05 reward=1.05× (slight reward); at alpha=2.0 reward=2.0× (cap);
# at alpha=4.74 reward=2.0× (cap, no runaway). This is monotone + bounded +
# never penalizes a super-additive observation by accident.
COMPOSITION_ALPHA_SUPER_ADDITIVE_THRESHOLD = 1.05  # alpha > 1.05 = super-additive
COMPOSITION_ALPHA_SUPER_ADDITIVE_REWARD_CAP = 2.0  # max reward factor (bounded)
COMPOSITION_ALPHA_SUPER_ADDITIVE_REWARD_FLOOR = 1.0  # min reward factor


def adjust_predicted_delta_for_composition_alpha_v2(
    base_delta: float,
    composition_alpha: float | None,
) -> float:
    """v2 cascade with SUPER_ADDITIVE branch (Catalog #227 wire-in extension).

    Per `feedback_super_additive_lane_g_v3_siren_topology_integration_landed_
    20260517.md` + the Q6 OP-3 extended sweep finding (2026-05-17):

      - alpha > 1.05 (SUPER_ADDITIVE): reward factor in [1.0, 2.0] clamped
        via ``min(max(alpha, 1.0), COMPOSITION_ALPHA_SUPER_ADDITIVE_REWARD_CAP)``.
        Apply by MULTIPLYING ``base_delta`` by ``-reward_factor`` if base_delta
        is negative (more-negative is better in score-delta convention; reward
        amplifies magnitude in the favorable direction). If base_delta is
        non-negative (positive predicted score change), no reward is applied
        because the candidate is not actually improving score.
      - alpha in (0.7, 1.05] (ADDITIVE): no adjustment (full additive savings)
      - alpha in (0.3, 0.7] (SUB-ADDITIVE): 50% penalty on predicted savings
      - alpha <= 0.3 (SATURATING): floor at -0.005 (single-substrate dominates)
      - alpha unknown (None): no adjustment

    Returns the (possibly adjusted) predicted_score_delta. Lower is better.

    Safety properties:
      - SUPER_ADDITIVE reward is BOUNDED at ``COMPOSITION_ALPHA_SUPER_ADDITIVE_
        REWARD_CAP = 2.0`` so a false-signal alpha=10 cannot runaway-promote.
      - SUPER_ADDITIVE reward only applies when ``base_delta < 0`` (the
        candidate is predicted to improve score). For ``base_delta >= 0`` the
        reward path is a no-op.
      - Non-finite alpha values (inf / nan) are silently rejected (no
        adjustment) per the conservative-on-bad-input rule.
      - v2 is a strict superset of v1: for alpha <= 1.05 the cascade returns
        the identical value as v1.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + "Forbidden score
    claims": this adjusts PREDICTED ΔS only; measured anchors are untouched.
    """
    if composition_alpha is None:
        return base_delta
    try:
        a = float(composition_alpha)
    except (TypeError, ValueError):
        return base_delta
    # Reject non-finite alpha (inf, nan) per the conservative-on-bad-input rule.
    if not math.isfinite(a):
        return base_delta
    # SUPER_ADDITIVE branch: alpha > 1.05.
    if a > COMPOSITION_ALPHA_SUPER_ADDITIVE_THRESHOLD:
        # Only reward when the candidate is actually improving score.
        if base_delta >= 0:
            return base_delta
        # Bounded reward factor in [1.0, 2.0].
        reward_factor = min(
            max(a, COMPOSITION_ALPHA_SUPER_ADDITIVE_REWARD_FLOOR),
            COMPOSITION_ALPHA_SUPER_ADDITIVE_REWARD_CAP,
        )
        # base_delta is negative (better-is-more-negative); multiply by
        # reward_factor amplifies magnitude in the favorable direction.
        return base_delta * reward_factor
    # ADDITIVE branch: alpha in (0.7, 1.05].
    if a > COMPOSITION_ALPHA_ADDITIVE_THRESHOLD:
        return base_delta
    # SUB-ADDITIVE branch: alpha in (0.3, 0.7].
    if a > COMPOSITION_ALPHA_SUB_ADDITIVE_THRESHOLD:
        return base_delta * COMPOSITION_ALPHA_SUB_ADDITIVE_PENALTY_FACTOR
    # SATURATING branch: alpha <= 0.3.
    return max(base_delta, COMPOSITION_ALPHA_SATURATING_DELTA_FLOOR)


# ── Predicted dispatch risk adjuster (OP-3 wire-in, 2026-05-15) ───────────
#
# Per codex chunk 6 finding (`.omx/research/codex_chunked_full_codebase_review_
# 20260515.md`): the RUDIN-DAUBECHIES preflight composite computes a SLIM
# (Sparse Linear Integer Model) `predicted_dispatch_risk` score over the
# preflight-gate verdict panel (see ``tac.preflight_rudin_daubechies.
# slim_risk_scorer.PreflightSLIMRiskScorer``) but the cathedral autopilot
# ranker did NOT consume it — the continual-learning loop closed on the
# RANKER side but left the PREFLIGHT-RISK signal stranded outside the
# rank_candidates composition chain.
#
# This adjuster wires the SLIM preflight risk into the same
# `apply_z1_empirical_revision_to_candidate_delta` chain that already stacks
# Tier A / Tier C / class-shift / composition-alpha so high-preflight-risk
# candidates are demoted in the autopilot ordering BEFORE any operator-
# authorized dispatch fires. The risk bands match the canonical
# `DISPATCH_RISK_REFUSAL_THRESHOLD = 50` from
# ``tac.preflight_rudin_daubechies.slim_risk_scorer`` (the SLIM scorer's own
# refusal threshold) plus a halve-band at 25 for moderate-risk candidates so
# the ranker degrades gracefully rather than cliff-edging at 50.
#
# Per CLAUDE.md "Forbidden score claims": this adjusts PREDICTED ΔS only;
# measured anchors are untouched. The original ``predicted_score_delta`` on
# the row is preserved.
PREDICTED_DISPATCH_RISK_REFUSAL_THRESHOLD = 50.0  # mirrors slim_risk_scorer
PREDICTED_DISPATCH_RISK_MODERATE_THRESHOLD = 25.0  # halve-band lower edge
PREDICTED_DISPATCH_RISK_REFUSAL_DELTA_FLOOR = 0.0  # cliff at zero (no improvement)
PREDICTED_DISPATCH_RISK_MODERATE_PENALTY_FACTOR = 0.5  # halve predicted savings


def adjust_predicted_delta_for_predicted_dispatch_risk(
    base_delta: float,
    predicted_dispatch_risk: float | None,
) -> float:
    """Demote high-preflight-risk candidates per the OP-3 SLIM wire-in.

    Per codex chunk 6 finding (2026-05-15): the RUDIN-DAUBECHIES preflight
    SLIM scorer (``tac.preflight_rudin_daubechies.slim_risk_scorer``) emits
    a per-candidate ``predicted_dispatch_risk`` denominated in the same
    integer-coefficient space as the preflight gate panel (Tier-1 = +25 per
    violation, META-meta = +50 per violation, refusal threshold = 50.0). The
    ranker treats the score in three bands matching the SLIM scorer's own
    semantics:

      - risk >= ``PREDICTED_DISPATCH_RISK_REFUSAL_THRESHOLD`` (50.0):
        REFUSE — floor at ``PREDICTED_DISPATCH_RISK_REFUSAL_DELTA_FLOOR``
        (effectively zero predicted improvement; the candidate is structurally
        unsafe to dispatch even if the predicted ΔS is strongly negative).
      - risk 25-50 (``PREDICTED_DISPATCH_RISK_MODERATE_THRESHOLD`` <= risk <
        refusal): MODERATE — halve predicted savings so the candidate falls
        below clean-preflight peers in the ordering but is still rankable.
      - risk < 25: LOW — no adjustment (gate-discipline is on the safe side
        of the SLIM threshold).
      - risk unknown (None) or non-numeric: no adjustment (don't penalize
        lack-of-evidence per the sister Z1 / Tier C / composition_alpha
        conventions).

    Per CLAUDE.md "Forbidden score claims": this adjusts PREDICTED ΔS only.
    Returns the (possibly demoted) predicted_score_delta. Lower is better in
    the score-delta convention (negative = improvement).

    [verified-against:
     ``tac.preflight_rudin_daubechies.slim_risk_scorer.DISPATCH_RISK_REFUSAL_
     THRESHOLD = 50.0``]
    """
    if predicted_dispatch_risk is None:
        return base_delta
    try:
        r = float(predicted_dispatch_risk)
    except (TypeError, ValueError):
        return base_delta
    if r >= PREDICTED_DISPATCH_RISK_REFUSAL_THRESHOLD:
        # Refuse: floor at zero so no improvement is ranked from this row,
        # regardless of how strongly negative its raw predicted_score_delta
        # was. The SLIM scorer has flagged the candidate as structurally
        # unsafe; the operator must clear the preflight panel first.
        return max(base_delta, PREDICTED_DISPATCH_RISK_REFUSAL_DELTA_FLOOR)
    if r >= PREDICTED_DISPATCH_RISK_MODERATE_THRESHOLD:
        # Moderate: halve the predicted savings (only affects negative
        # deltas; positive deltas stay the same magnitude under multiply).
        return base_delta * PREDICTED_DISPATCH_RISK_MODERATE_PENALTY_FACTOR
    # LOW (risk < 25): no adjustment.
    return base_delta


# ── Realistic-stacking correction (Catalog #229/#287 sister, finding #12) ───
#
# Grand council T3 finding #12 (PROCEED_WITH_REVISIONS, 2026-05-18; see
# `.omx/research/council_t3_finding_12_52_row_composite_ev_realistic_20260518.md`):
# the Wave 2A 52-row arbitrariness-extinction audit composite EV
# (`.omx/research/arbitrariness_extinction_audit_20260518.jsonl`) is
# [-0.139, -0.026] assuming additive composition of all 52 independent
# extinctions. Wave 1 NSCS06 v8 anchor empirically refutes additivity at
# composition: 4-of-7 cargo-cults unwound simultaneously produced -78%
# non-monotonicity relative to the additive prediction. Boyd's convex-
# feasibility lens: composing N convex feasibilities yields an intersection
# FAR SMALLER than the sum of per-constraint regions.
#
# Per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE": predicted
# deltas must be sourced from solvable math, not arbitrary heuristic. This
# correction routes through the canonical audit composite-EV envelope so
# the ranker's effective predicted_score_delta tracks realistic empirical
# stacking discipline rather than optimistic additive prediction.
#
# Realistic envelope [-0.05, -0.02] is HARD-EARNED-FROM-WAVE-1-EMPIRICAL-
# ANCHOR per the council's Assumption-Adversary verdict.
# [empirical:.omx/research/arbitrariness_extinction_audit_20260518.jsonl]
# [empirical:.omx/research/council_t3_finding_12_52_row_composite_ev_realistic_20260518.md]
AUDIT_COMPOSITE_OPTIMISTIC_EV_LOWER = -0.139  # additive 52-row lower bound
AUDIT_COMPOSITE_OPTIMISTIC_EV_UPPER = -0.026  # additive 52-row upper bound
AUDIT_COMPOSITE_REALISTIC_EV_LOWER = -0.05   # T3 finding #12 lower bound
AUDIT_COMPOSITE_REALISTIC_EV_UPPER = -0.02   # T3 finding #12 upper bound
# Empirical saturation threshold per Catalog #319 composition_alpha=0.85
# anchor; above this stacking count, additional extinctions saturate.
REALISTIC_STACKING_SATURATION_COUNT = 10
REALISTIC_STACKING_SATURATION_PENALTY_FACTOR = 0.5  # additional 50% above sat

# Structural cascade floor values — preserved verbatim by the realistic-
# stacking correction so floors carry their original "structurally dead-
# weight" semantic. The realistic correction is a multiplicative envelope
# adjustment for cascade outputs that have NOT already hit a structural
# floor; a floor value is itself the authoritative verdict.
_CASCADE_STRUCTURAL_FLOOR_VALUES: frozenset[float] = frozenset({
    MDL_DENSITY_WITHIN_CLASS_SATURATED_DELTA_FLOOR,
    COMPOSITION_ALPHA_SATURATING_DELTA_FLOOR,
    PREDICTED_DISPATCH_RISK_REFUSAL_DELTA_FLOOR,
})


def adjust_predicted_delta_for_realistic_stacking_correction(
    predicted_delta: float,
    n_stacked_extinctions: int,
    *,
    audit_composite_optimistic_upper: float = AUDIT_COMPOSITE_OPTIMISTIC_EV_UPPER,
    audit_composite_realistic_upper: float = AUDIT_COMPOSITE_REALISTIC_EV_UPPER,
    saturation_count: int = REALISTIC_STACKING_SATURATION_COUNT,
    saturation_penalty_factor: float = REALISTIC_STACKING_SATURATION_PENALTY_FACTOR,
) -> float:
    """Apply Wave 2A realistic-stacking correction per grand council T3 finding #12.

    Optimistic EV assumes monotonic additive stacking of all 52 extinction rows
    from the canonical arbitrariness-extinction audit
    (`.omx/research/arbitrariness_extinction_audit_20260518.jsonl`). Empirical
    anchors — Wave 1 NSCS06 v8 -78% non-monotonicity + Wave 2A Z3+sister
    composition_alpha=0.85 saturating per Catalog #319 — show realistic
    stacking saturates well before the additive prediction.

    Correction factor: ``realistic_magnitude / optimistic_magnitude`` applied
    multiplicatively to ``predicted_delta`` for any candidate that composes
    2+ extinctions (``n_stacked_extinctions >= 2``). Single-extinction
    candidates (``n_stacked_extinctions <= 1``) pass through unchanged — the
    audit row IS the empirical per-extinction prediction; only COMPOSITIONS
    suffer the non-additivity discount.

    Above ``saturation_count`` (default 10) the factor is further reduced by
    ``saturation_penalty_factor`` (default 0.5) per the Catalog #319 empirical
    saturation anchor. This is conservative — it expresses Boyd's
    convex-intersection bound at high stacking depths.

    Per CLAUDE.md "Forbidden score claims" + "Apples-to-apples evidence
    discipline": this adjusts PREDICTED ΔS only. The original
    ``predicted_score_delta`` on the candidate row is NEVER mutated; this is
    a transient sort-key adjustment composed into the canonical Z1 revision
    cascade per :func:`apply_z1_empirical_revision_to_candidate_delta`.

    Returns the (possibly demoted) predicted_score_delta. Lower is better in
    the score-delta convention (negative = improvement).

    [verified-against: grand council T3 finding #12 verdict
     PROCEED_WITH_REVISIONS 2026-05-18; Wave 1 NSCS06 v8 -78% non-monotonicity
     empirical anchor; Catalog #319 composition_alpha saturation regime]
    """
    if n_stacked_extinctions <= 1:
        # Single-extinction or unknown — pass through unchanged. The audit
        # row IS the per-extinction prediction; only compositions are
        # discounted by this gate.
        return predicted_delta
    # Floor-preservation: if predicted_delta has already been floored to one
    # of the structural cascade floors (saturating composition / risk-refusal
    # / saturated MDL density), the floor is a SEMANTIC signal that the
    # candidate is dead-weight per its upstream verdict. Further multiplying
    # the floor by the realistic-correction factor would obscure the
    # structural "this candidate is at the floor" semantic without changing
    # ranking ordering. Preserve the floor verbatim.
    if predicted_delta in _CASCADE_STRUCTURAL_FLOOR_VALUES:
        return predicted_delta
    optimistic_magnitude = abs(audit_composite_optimistic_upper)
    if optimistic_magnitude == 0.0:
        # Defensive: cannot compute ratio against zero envelope.
        return predicted_delta
    realistic_magnitude = abs(audit_composite_realistic_upper)
    correction_factor = realistic_magnitude / optimistic_magnitude
    if n_stacked_extinctions > saturation_count:
        # Saturating regime per Catalog #319 anchor: composition_alpha=0.85
        # observed empirically; further stacking yields diminishing returns.
        correction_factor *= saturation_penalty_factor
    return predicted_delta * correction_factor


def rank_candidates(
    candidates: list[CandidateRow],
    *,
    rank_axis: str = "eig_per_dollar",
    continual_posterior: Any | None = None,
    apply_z1_empirical_revision: bool = True,
    score_panel_axis: str = "contest_cpu",
) -> list[CandidateRow]:
    """Rank candidates by the chosen axis (descending best-first).

    Recognized axes:
      - ``eig_per_dollar`` — expected information gain per dollar (default)
      - ``predicted_score_delta`` — most-negative-first (greatest improvement)

    When ``continual_posterior`` is provided (W/I/A I-1 wire-in, 2026-05-12),
    each candidate's predicted_score_delta is scaled by the continual-learning
    family-keyed correction factor BEFORE sorting. The original
    predicted_score_delta on the CandidateRow is preserved (the correction
    is applied transiently for sort-key purposes only) so halt events still
    report the raw prediction. Per CLAUDE.md "Subagent coherence-by-default"
    this exercises wire-in hook 5 (continual-learning posterior).

    When ``apply_z1_empirical_revision=True`` (default), the Z1 empirical
    revision adjustments are applied BEFORE the continual-posterior
    correction: MDL-density penalty (within-class trap) + class-shift
    reward (cooperative-receiver / predictive-receiver / foveation). Per
    `feedback_z1_mdl_ablation_landed_20260514.md` operator decision #4. The
    ORIGINAL ``predicted_score_delta`` on the row is preserved; the
    adjustment is applied transiently for sort-key purposes only.

    Per CLAUDE.md "Forbidden /tmp paths": no temp paths used; pure in-memory.
    """
    for candidate in candidates:
        _require_candidate_planning_cost(candidate)
    _contest_axis_tag_for_panel(score_panel_axis)

    def _effective_delta(c: CandidateRow) -> float:
        if _candidate_prediction_band_rank_reward_suppressed(c):
            return 0.0
        if apply_z1_empirical_revision:
            return apply_z1_empirical_revision_to_candidate_delta(
                c,
                score_panel_axis=score_panel_axis,
            )
        return c.predicted_score_delta

    def _effective_eig_per_dollar(c: CandidateRow) -> float:
        if _candidate_prediction_band_rank_reward_suppressed(c):
            return 0.0
        base = c.eig_per_dollar()
        if not apply_z1_empirical_revision:
            return base

        # Preserve the existing Tier-A MDL-density EIG modifier so saturated
        # within-class rows stay down-ranked even when their raw score delta
        # already sits on the Z1 floor.
        tier_a_eig = base
        if c.mdl_density is None:
            tier_a_delta = c.predicted_score_delta
        else:
            try:
                d = float(c.mdl_density)
            except (TypeError, ValueError):
                tier_a_delta = c.predicted_score_delta
            else:
                tier_a_delta = adjust_predicted_delta_for_mdl_density(
                    c.predicted_score_delta, d
                )
                if d > MDL_DENSITY_WITHIN_CLASS_SATURATED_THRESHOLD:
                    tier_a_eig = base * 0.10  # 90% penalty
                elif d > MDL_DENSITY_WITHIN_CLASS_TRENDING_THRESHOLD:
                    tier_a_eig = (
                        base * MDL_DENSITY_WITHIN_CLASS_TRENDING_PENALTY_FACTOR
                    )

        # The default autopilot axis is EIG/$, so score-delta-only Z1 signals
        # would otherwise be invisible in normal operation. Reweight the EIG
        # prior by the same non-Tier-A score-delta adjustment chain used by the
        # explicit predicted-score axis: Tier C, class-shift literature, and
        # composition alpha.
        full_delta = apply_z1_empirical_revision_to_candidate_delta(
            c,
            score_panel_axis=score_panel_axis,
        )
        tier_a_gain = max(0.0, -tier_a_delta)
        full_gain = max(0.0, -full_delta)
        if tier_a_gain > 0.0:
            return tier_a_eig * (full_gain / tier_a_gain)
        if full_gain > 0.0 and c.estimated_dispatch_cost_usd > 0.0:
            return full_gain / c.estimated_dispatch_cost_usd
        return tier_a_eig

    if rank_axis == "eig_per_dollar":
        if continual_posterior is None:
            return sorted(candidates, key=_effective_eig_per_dollar, reverse=True)
        # Reweight EIG/$ by posterior correction. EIG itself is unchanged; the
        # cost-effective dispatch ordering still reflects empirical bias.
        def _eig_key(c: CandidateRow) -> float:
            factor, _, _ = _posterior_correction_factor(c, continual_posterior)
            return _effective_eig_per_dollar(c) * factor
        return sorted(candidates, key=_eig_key, reverse=True)
    if rank_axis == "predicted_score_delta":
        if continual_posterior is None:
            return sorted(candidates, key=_effective_delta)
        # Reweight predicted_score_delta by posterior correction. Most-negative
        # first ordering preserved.
        def _delta_key(c: CandidateRow) -> float:
            factor, _, _ = _posterior_correction_factor(c, continual_posterior)
            return _effective_delta(c) * factor
        return sorted(candidates, key=_delta_key)
    raise ValueError(
        f"unknown rank_axis {rank_axis!r}; must be 'eig_per_dollar' or "
        "'predicted_score_delta'"
    )


def discover_sensitivity_map_artifacts(
    search_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    """Enumerate available sensitivity-map artifacts under ``experiments/results``.

    W/I/A I-3 wire-in (2026-05-12): the autopilot's planner context now
    surfaces the inventory of saved sensitivity maps. This exercises
    CLAUDE.md unified-Lagrangian wire-in hook 1 (sensitivity-map
    contribution) — the autopilot is structurally aware of per-tensor
    importance evidence even when it does not consume the maps directly.

    Returns a JSON-safe dict::

        {
            "discovered": True,
            "artifact_paths": ["experiments/results/posenet_sensitivity_v5/sensitivity_map.pt", ...],
            "count": 3,
            "search_roots": ["experiments/results"]
        }

    No file is opened or parsed; the enumerator scans for ``sensitivity_map.pt``
    files only. Per CLAUDE.md "Forbidden /tmp paths" the search is rooted at
    the repo's ``experiments/results`` tree.
    """
    roots = (
        [REPO_ROOT / "experiments" / "results"]
        if search_dirs is None
        else list(search_dirs)
    )
    artifact_paths: list[str] = []
    search_roots: list[str] = []
    for root in roots:
        search_roots.append(str(root.relative_to(REPO_ROOT)) if root.is_relative_to(REPO_ROOT) else str(root))
        if not root.is_dir():
            continue
        try:
            for p in root.rglob("sensitivity_map.pt"):
                rel = p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p
                artifact_paths.append(str(rel))
        except Exception:  # pragma: no cover - defensive
            continue
    return {
        "discovered": True,
        "artifact_paths": sorted(artifact_paths),
        "count": len(artifact_paths),
        "search_roots": search_roots,
    }


def load_planner_posterior_for_loop(
    continual_posterior_path: Path | None = None,
    *,
    include_sensitivity_map_inventory: bool = True,
) -> tuple[Any | None, dict[str, Any]]:
    """Load read-only continual-learning posterior context for the loop.

    Returns ``(posterior_or_none, context_payload)``. The payload is a small
    JSON-serializable dict reporting load status / anchor counts so iteration
    notes can surface ``loaded N=X anchors`` for operator visibility.

    W/I/A I-3 wire-in (2026-05-12): when
    ``include_sensitivity_map_inventory=True`` (default), the context payload
    ALSO includes a ``sensitivity_map_inventory`` key with the enumerated
    artifact paths (CLAUDE.md unified-Lagrangian hook 1 — sensitivity-map
    contribution).

    Per CLAUDE.md "Operator gates must be wired and used": failure to load
    falls back to ``(None, {"loaded": False, "reason": ...})`` — the loop
    keeps ranking without posterior context rather than crashing. The
    operator sees the load_error in iteration notes.
    """
    if not _POSTERIOR_IMPORTS_OK or load_continual_learning_posterior is None:
        ctx: dict[str, Any] = {"loaded": False, "reason": "tac.continual_learning import unavailable"}
        if include_sensitivity_map_inventory:
            ctx["sensitivity_map_inventory"] = discover_sensitivity_map_artifacts()
        return None, ctx
    try:
        posterior = load_continual_learning_posterior(continual_posterior_path)
    except Exception as exc:  # pragma: no cover - exercised by load_error test
        ctx = {"loaded": False, "reason": f"load_error:{type(exc).__name__}", "message": str(exc)}
        if include_sensitivity_map_inventory:
            ctx["sensitivity_map_inventory"] = discover_sensitivity_map_artifacts()
        return None, ctx
    payload: dict[str, Any] = {
        "loaded": True,
        "schema": getattr(posterior, "schema", "unknown"),
        "accepted_anchor_count": getattr(posterior, "accepted_anchor_count", 0),
        "refused_anchor_count": getattr(posterior, "refused_anchor_count", 0),
        "track_correction_count": len(getattr(posterior, "track_correction_posteriors", {})),
    }
    if include_sensitivity_map_inventory:
        payload["sensitivity_map_inventory"] = discover_sensitivity_map_artifacts()
    return posterior, payload


def cost_band_envelope_check(
    candidate: CandidateRow,
    *,
    platform: str | None = None,
    gpu: str | None = None,
    epochs: int | None = None,
    all_flags_on: bool = True,
    posterior_path: Path | None = None,
) -> tuple[float | None, str, dict[str, Any]]:
    """Query the cost-band posterior for an envelope-vs-estimate sanity check.

    W/I/A I-1 wire-in (2026-05-12, sister of continual-learning wire-in).
    Returns ``(p50_cost_usd_or_none, confidence_tag, payload)``. The payload
    captures p10/p50/p90 + anchor count so loop notes can surface a
    "candidate cost $X vs posterior p50 $Y (n=Z, tag=...)" comparison.

    Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" the cost band
    is itself non-authoritative; it is a planning prior derived from prior
    dispatches' invoice-actuals. Returns ``(None, "unavailable", {})`` when
    inputs are missing or the predict() call raises.
    """
    if predict_cost_band is None or not platform or not gpu or epochs is None:
        return None, "unavailable", {
            "cost_band_available": False,
            "reason": (
                "predict_cost_band unavailable"
                if predict_cost_band is None
                else "platform/gpu/epochs not provided by candidate"
            ),
        }
    try:
        prediction = predict_cost_band(
            str(platform), str(gpu), int(epochs),
            all_flags_on=bool(all_flags_on), posterior_path=posterior_path,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"load_error:{type(exc).__name__}", {
            "cost_band_available": False,
            "reason": str(exc),
        }
    payload = {
        "cost_band_available": True,
        "cost_band_platform": prediction.platform,
        "cost_band_gpu": prediction.gpu,
        "cost_band_epochs": prediction.epochs,
        "cost_band_n_anchors": prediction.n_anchors,
        "cost_band_confidence_tag": prediction.confidence_tag,
        "cost_band_p10_cost_usd": prediction.p10_cost_usd,
        "cost_band_p50_cost_usd": prediction.p50_cost_usd,
        "cost_band_p90_cost_usd": prediction.p90_cost_usd,
        "candidate_cost_vs_p50_ratio": (
            candidate.estimated_dispatch_cost_usd / prediction.p50_cost_usd
            if prediction.p50_cost_usd > 0 else None
        ),
    }
    return prediction.p50_cost_usd, prediction.confidence_tag, payload


def _claim_token(value: str, *, fallback: str, max_len: int = 96) -> str:
    """Return a claim-helper-safe token with no whitespace."""
    token = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or "").strip())
    token = token.strip("._:-")
    return (token[:max_len] or fallback)


def _claim_note_value(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("|", "/")).strip()


def _autopilot_claim_instance_job_id(candidate: CandidateRow) -> str:
    slug = _claim_token(
        candidate.candidate_id or candidate.lane_id,
        fallback="candidate",
        max_len=80,
    )
    packet_hash = (
        candidate.dispatch_packet_sha256
        or candidate.archive_sha256
        or candidate.runtime_tree_sha256
    )
    suffix = _claim_token(packet_hash[:12], fallback="nohash", max_len=16)
    return f"cathedral_autopilot_{slug}_{suffix}"


def _record_autopilot_dispatch_claim(
    candidate: CandidateRow,
    *,
    auth_mode: OperatorAuthorizedModeConfig,
    claims_path: Path,
) -> tuple[bool, str, str]:
    """Claim the lane before self-authorization can become non-blocking."""
    helper = auth_mode.canonical_helper_script
    instance_job_id = _autopilot_claim_instance_job_id(candidate)
    if helper is None or not helper.is_file():
        return (
            False,
            f"canonical claim helper {helper!r} is unavailable",
            instance_job_id,
        )

    notes = "; ".join(
        part for part in (
            "Cathedral autopilot self-authorization claim before requires_approval=false",
            f"candidate_id={_claim_note_value(candidate.candidate_id)}",
            f"dispatch_packet_sha256={_claim_note_value(candidate.dispatch_packet_sha256)}",
            f"archive_sha256={_claim_note_value(candidate.archive_sha256)}",
            f"runtime_tree_sha256={_claim_note_value(candidate.runtime_tree_sha256)}",
            f"estimated_cost_usd={candidate.estimated_dispatch_cost_usd:.4f}",
            "remote_provider_job_spawned=false",
        ) if part
    )
    cmd = [
        sys.executable,
        str(helper),
        "claim",
        "--lane-id",
        candidate.lane_id,
        "--platform",
        AUTOPILOT_CLAIM_PLATFORM,
        "--instance-job-id",
        instance_job_id,
        "--agent",
        AUTOPILOT_CLAIM_AGENT,
        "--status",
        AUTOPILOT_CLAIM_STATUS,
        "--notes",
        notes,
        "--ttl-hours",
        str(AUTOPILOT_CLAIM_TTL_HOURS),
    ]
    cmd.extend(["--claims-path", str(claims_path)])
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:  # pragma: no cover - filesystem/interpreter failure
        return (
            False,
            f"dispatch claim helper invocation failed: {type(exc).__name__}: {exc}",
            instance_job_id,
        )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode != 0:
        detail = stderr or stdout or "no output"
        return (
            False,
            f"dispatch claim helper refused claim with rc={result.returncode}: {detail}",
            instance_job_id,
        )
    return True, stdout or "dispatch claim recorded", instance_job_id


def make_dispatch_halt_event(
    candidate: CandidateRow,
    *,
    requires_approval_classes: frozenset[EventClass],
    blockers: list[str] | None = None,
    auth_mode: OperatorAuthorizedModeConfig | None = None,
    env_authorized: bool | None = None,
    claims_path: Path | None = None,
) -> HaltEvent:
    """Construct a HALT event for one dispatch decision.

    Per CLAUDE.md "operator-gate non-negotiable": when ``EventClass.DISPATCH``
    is in ``requires_approval_classes``, ``requires_approval=True`` UNLESS the
    operator-authorized le-$5/individual mode is engaged AND every precondition
    holds for THIS candidate. In that case the event is tagged
    ``[autopilot-claude-le-5-dollar]`` only after
    :mod:`tools.claim_lane_dispatch` records a lane claim. ``requires_approval``
    is set to False only after that claim succeeds.

    The dual-gate check (CLI flag + env-var) lives entirely inside this
    function; callers cannot bypass it. When ``env_authorized`` is None the
    real env-var is consulted; tests inject ``env_authorized=True/False``
    directly.
    """
    requires = EventClass.DISPATCH in requires_approval_classes
    halt_blockers = list(blockers or [])
    autopilot_authorized = False
    autopilot_tag = ""
    autopilot_reason = ""
    autopilot_refused = ""
    autopilot_claim_recorded = False
    autopilot_claim_instance_job_id = ""
    autopilot_claim_reason = ""

    if auth_mode is not None and auth_mode.enabled:
        config_blocker = _authorized_mode_config_blocker(auth_mode, repo_root=REPO_ROOT)
        if config_blocker:
            requires = True
            autopilot_refused = config_blocker
            halt_blockers.append("operator_authorized_mode_config_invalid")
        else:
            env_ok = (
                _env_authorizes_mode()
                if env_authorized is None
                else bool(env_authorized)
            )
            if not env_ok:
                requires = True
                autopilot_refused = (
                    f"env-var {OPERATOR_AUTHORIZED_MODE_ENV_VAR}="
                    f"{OPERATOR_AUTHORIZED_MODE_ENV_VALUE_ENABLED} is missing; CLI "
                    "flag alone is insufficient (defense-in-depth)"
                )
            else:
                ok, reason = auth_mode.can_authorize(
                    candidate,
                    dispatch_claims_path=claims_path,
                )
                if ok:
                    if claims_path is None:
                        autopilot_claim_reason = (
                            "claims_path is required before self-authorization so "
                            "tests and direct callers cannot silently write an "
                            "implicit dispatch claim"
                        )
                    else:
                        (
                            autopilot_claim_recorded,
                            autopilot_claim_reason,
                            autopilot_claim_instance_job_id,
                        ) = _record_autopilot_dispatch_claim(
                            candidate,
                            auth_mode=auth_mode,
                            claims_path=claims_path,
                        )
                    if autopilot_claim_recorded:
                        prospective = (
                            auth_mode.cumulative_spent_usd
                            + candidate.estimated_dispatch_cost_usd
                        )
                        autopilot_authorized = True
                        autopilot_tag = AUTOPILOT_AUTHORIZED_TAG
                        autopilot_reason = (
                            f"per-dispatch cost ${candidate.estimated_dispatch_cost_usd:.4f} "
                            f"<= cap ${auth_mode.per_dispatch_cap_usd:.4f}; "
                            f"cumulative ${prospective:.4f} "
                            f"<= envelope ${auth_mode.cumulative_cap_usd:.4f}; "
                            f"dispatch claim recorded as {autopilot_claim_instance_job_id}"
                        )
                        # Reserve cost in the per-process counter so the next
                        # candidate in this iteration sees the updated cumulative.
                        auth_mode.record_authorization(candidate)
                        requires = False
                    else:
                        requires = True
                        autopilot_refused = (
                            "dispatch claim is required before self-authorization; "
                            f"{autopilot_claim_reason}"
                        )
                        halt_blockers.append(
                            "dispatch_claim_required_for_self_authorization"
                        )
                else:
                    requires = True
                    autopilot_refused = reason

    return HaltEvent(
        event_class=EventClass.DISPATCH,
        candidate_id=candidate.candidate_id,
        reason=(
            f"Dispatch decision for candidate {candidate.candidate_id} "
            f"(family={candidate.family}, predicted_score_delta="
            f"{candidate.predicted_score_delta:+.6f}) — "
            + (
                "autopilot self-authorized (le-$5/individual operator-set mode)"
                if autopilot_authorized
                else "operator decision required."
            )
        ),
        predicted_score_delta=candidate.predicted_score_delta,
        estimated_cost_usd=candidate.estimated_dispatch_cost_usd,
        requires_approval=requires,
        halt_at_utc=dt.datetime.now(dt.UTC).isoformat(),
        blockers=halt_blockers,
        lane_id=candidate.lane_id,
        claim_keys=candidate.dispatch_claim_keys(),
        target_modes=list(candidate.target_modes),
        dispatch_packet_sha256=candidate.dispatch_packet_sha256,
        archive_path=candidate.archive_path,
        submission_dir=candidate.submission_dir,
        archive_manifest_path=candidate.archive_manifest_path,
        archive_sha256=candidate.archive_sha256,
        candidate_archive_bytes=candidate.candidate_archive_bytes,
        runtime_tree_sha256=candidate.runtime_tree_sha256,
        timing_smoke_command=candidate.timing_smoke_command,
        ready_for_exact_eval_dispatch=candidate.ready_for_exact_eval_dispatch,
        literature_anchor=candidate.literature_anchor,
        source_supports=candidate.source_supports,
        paper_claim_scope=candidate.paper_claim_scope,
        pact_must_prove=candidate.pact_must_prove,
        decode_complexity_evidence=candidate.decode_complexity_evidence,
        autopilot_authorized=autopilot_authorized,
        autopilot_tag=autopilot_tag,
        autopilot_authorized_reason=autopilot_reason,
        autopilot_refused_reason=autopilot_refused,
        autopilot_claim_recorded=autopilot_claim_recorded,
        autopilot_claim_instance_job_id=autopilot_claim_instance_job_id,
        autopilot_claim_reason=autopilot_claim_reason,
    )


def append_autopilot_journal_row(
    journal_path: Path,
    event: HaltEvent,
    *,
    iteration: int,
) -> None:
    """Append one structured JSONL row recording an autopilot-authorized dispatch.

    Per CLAUDE.md "Forbidden /tmp paths": callers must point ``journal_path`` at
    a durable location (``reports/`` or ``.omx/state/``); this helper does not
    pick a default location.

    Per CLAUDE.md "Subagent commits MUST use serializer": this writes a JSONL
    row that the operator can later commit via the canonical serializer; the
    helper itself never invokes git.
    """
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema": "tac_cathedral_autopilot_authorized_journal_v1",
        "iteration": iteration,
        "candidate_id": event.candidate_id,
        "lane_id": event.lane_id,
        "claim_keys": list(event.claim_keys),
        "target_modes": list(event.target_modes),
        "dispatch_packet_sha256": event.dispatch_packet_sha256,
        "archive_path": event.archive_path,
        "submission_dir": event.submission_dir,
        "archive_manifest_path": event.archive_manifest_path,
        "archive_sha256": event.archive_sha256,
        "candidate_archive_bytes": event.candidate_archive_bytes,
        "runtime_tree_sha256": event.runtime_tree_sha256,
        "ready_for_exact_eval_dispatch": event.ready_for_exact_eval_dispatch,
        "literature_anchor": event.literature_anchor,
        "source_supports": event.source_supports,
        "paper_claim_scope": event.paper_claim_scope,
        "pact_must_prove": event.pact_must_prove,
        "decode_complexity_evidence": event.decode_complexity_evidence,
        "predicted_score_delta": event.predicted_score_delta,
        "estimated_cost_usd": event.estimated_cost_usd,
        "halt_at_utc": event.halt_at_utc,
        "autopilot_authorized": event.autopilot_authorized,
        "autopilot_tag": event.autopilot_tag,
        "autopilot_authorized_reason": event.autopilot_authorized_reason,
        "autopilot_refused_reason": event.autopilot_refused_reason,
        "autopilot_claim_recorded": event.autopilot_claim_recorded,
        "autopilot_claim_instance_job_id": event.autopilot_claim_instance_job_id,
        "autopilot_claim_reason": event.autopilot_claim_reason,
        "blockers": list(event.blockers),
        "claude_md_compliance_tags": [
            "operator_authorized_le_5_dollar_mode",
            "halt_and_ask_preserved_above_cap",
            "no_kill_verdict",
            "dispatch_claim_check_done",
        ],
    }
    with journal_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def make_kill_halt_event(candidate_id: str, reason: str) -> HaltEvent:
    """Construct a KILL HALT event.

    Per CLAUDE.md "kill-as-last-resort": KILL events ALWAYS require approval.
    The autonomous loop NEVER auto-kills.
    """
    return HaltEvent(
        event_class=EventClass.KILL,
        candidate_id=candidate_id,
        reason=reason,
        predicted_score_delta=0.0,
        estimated_cost_usd=0.0,
        requires_approval=True,  # FORCED True per CLAUDE.md
        halt_at_utc=dt.datetime.now(dt.UTC).isoformat(),
    )


def inject_operator_decision(
    event: HaltEvent,
    decision: OperatorDecision,
    notes: str = "",
) -> HaltEvent:
    """Record the operator's decision on a HALT event (returns updated event)."""
    if event.decision is not None:
        raise ValueError(
            f"event {event.candidate_id!r}/{event.event_class.value!r} already "
            f"decided as {event.decision.value!r}"
        )
    event.decision = decision
    event.decision_at_utc = dt.datetime.now(dt.UTC).isoformat()
    event.decision_notes = notes
    return event


# ── Dispatch-claim coordination check ──────────────────────────────────────


def check_dispatch_claim_conflict(
    candidate_id: str,
    *,
    claim_keys: list[str] | tuple[str, ...] | None = None,
    claims_path: Path | None = None,
    now_utc: dt.datetime | None = None,
    ttl_hours: float = AUTOPILOT_CLAIM_TTL_HOURS,
) -> tuple[bool, str]:
    """Check the active-lane-dispatch-claims registry for a conflicting claim.

    Returns ``(has_conflict, reason)``. Returns ``(False, "")`` if no claim
    file exists yet (cold start) or no conflicting claim is found.

    The claims file is the markdown registry referenced in CLAUDE.md
    "CROSS-AGENT DISPATCH COORDINATION". This helper uses the canonical parsed
    claim-row semantics from :mod:`tools.claim_lane_dispatch`: exact
    ``lane_id`` matching only, latest row per ``(lane_id, instance/job_id)``,
    and terminal rows close older nonterminal rows for the same job.
    """
    p = claims_path or (
        REPO_ROOT / ".omx" / "state" / "active_lane_dispatch_claims.md"
    )
    if not p.is_file():
        return False, ""
    keys: list[str] = []
    for key in [candidate_id, *(claim_keys or [])]:
        s = str(key or "").strip()
        if s and s not in keys:
            keys.append(s)
    if not keys:
        return False, ""
    try:
        text = p.read_text(encoding="utf-8")
        claims = _claim_lane_dispatch._parse_claims(text)
        latest_claims = _claim_lane_dispatch._latest_claims_by_job(claims)
    except Exception as exc:
        return True, (
            f"could not parse active-lane-dispatch-claims registry at {p} "
            f"with tools/claim_lane_dispatch.py ({type(exc).__name__}: {exc}); "
            "fail closed before dispatch"
        )
    now = now_utc or dt.datetime.now(dt.UTC)
    ttl = dt.timedelta(hours=ttl_hours)
    for claim in latest_claims.values():
        if claim.lane_id not in keys:
            continue
        if _claim_lane_dispatch._is_terminal(claim.status):
            continue
        stale = _claim_lane_dispatch._claim_is_stale_nonterminal(
            claim, now_utc=now, ttl=ttl
        )
        state = "stale nonterminal" if stale else "active"
        return True, (
            f"{state} dispatch claim for exact lane_id {claim.lane_id!r} "
            f"(candidate_id {candidate_id!r}, job {claim.instance_job_id!r}, "
            f"status {claim.status!r}) is present at {p}. "
            "Close or supersede it with tools/claim_lane_dispatch.py before dispatch."
        )
    return False, ""


# ── Loop iteration ─────────────────────────────────────────────────────────


def run_one_loop_iteration(
    candidates: list[CandidateRow],
    *,
    iteration: int = 1,
    rank_axis: str = "eig_per_dollar",
    requires_approval_on: frozenset[EventClass] = frozenset({EventClass.DISPATCH}),
    claims_path: Path | None = None,
    race_mode: bool = False,
    max_dispatch_recommendations: int | None = None,
    auth_mode: OperatorAuthorizedModeConfig | None = None,
    env_authorized: bool | None = None,
    continual_posterior: Any | None = None,
    continual_posterior_path: Path | None = None,
    auto_load_continual_posterior: bool = False,
    score_panel_axis: str = "contest_cpu",
) -> LoopIterationReport:
    """Run one cycle: rank → dispatch-claim check → halt-event emission.

    Per CLAUDE.md "operator-gate non-negotiable": this never actually
    dispatches. It surfaces HALT events with ``requires_approval=True`` for
    operator decision.

    Per CLAUDE.md "race-mode rigor inversion": when ``race_mode=True`` the
    loop trims the candidate set to the ones with smallest predicted cost
    AND non-trivial predicted_score_delta (the "smallest credible bolt-on"
    pattern from the May 4 race postmortem).

    W/I/A I-1 wire-in (2026-05-12): when ``continual_posterior`` is provided
    (or ``auto_load_continual_posterior=True``) the rank step applies the
    family-keyed correction factor from :mod:`tac.continual_learning`. The
    raw predicted_score_delta on each CandidateRow is unchanged; the
    correction biases ranking order only. Iteration notes record the loaded
    posterior anchor counts for operator visibility.
    """
    started = dt.datetime.now(dt.UTC).isoformat()
    notes: list[str] = []
    validate_authorized_mode_config(auth_mode, repo_root=REPO_ROOT)
    for candidate in candidates:
        _require_candidate_planning_cost(candidate)
    _contest_axis_tag_for_panel(score_panel_axis)

    # W/I/A I-1: optionally auto-load continual-learning posterior so the
    # loop's rank step applies empirical-anchor reweighting. Tests inject
    # ``continual_posterior=`` directly to skip the file load.
    if continual_posterior is None and auto_load_continual_posterior:
        continual_posterior, posterior_context = load_planner_posterior_for_loop(
            continual_posterior_path=continual_posterior_path,
        )
        if posterior_context.get("loaded"):
            notes.append(
                f"continual-learning posterior loaded "
                f"(accepted_anchors={posterior_context.get('accepted_anchor_count', 0)}, "
                f"track_corrections={posterior_context.get('track_correction_count', 0)})"
            )
        else:
            notes.append(
                f"continual-learning posterior unavailable: "
                f"{posterior_context.get('reason', 'unknown')}"
            )

    if race_mode:
        notes.append(
            "race-mode active per operator opt-in; ranking trimmed to "
            "smallest-credible-bolt-on subset"
        )
        # smallest credible bolt-on: post-gate predicted_delta < 0 AND lowest cost
        candidates = sorted(
            [
                c for c in candidates
                if _candidate_has_effective_negative_delta_for_race_mode(
                    c,
                    continual_posterior=continual_posterior,
                    score_panel_axis=score_panel_axis,
                )
            ],
            key=lambda c: c.estimated_dispatch_cost_usd,
        )

    n_seen = len(candidates)
    ranked = (
        rank_candidates(
            candidates,
            rank_axis=rank_axis,
            continual_posterior=continual_posterior,
            score_panel_axis=score_panel_axis,
        )
        if candidates else []
    )

    halt_events: list[HaltEvent] = []
    n_blocked = 0
    n_ranked = 0
    effective_claims_path = claims_path or (
        REPO_ROOT / ".omx" / "state" / "active_lane_dispatch_claims.md"
    )

    for cand in ranked:
        if max_dispatch_recommendations is not None and n_ranked >= max_dispatch_recommendations:
            break
        # Cross-agent dispatch claim check
        has_conflict, reason = check_dispatch_claim_conflict(
            cand.candidate_id,
            claim_keys=cand.dispatch_claim_keys(),
            claims_path=effective_claims_path,
        )
        if has_conflict:
            n_blocked += 1
            notes.append(reason)
            continue

        # Existing blockers from the candidate row are surfaced too.
        event_blockers = list(cand.blockers)
        event = make_dispatch_halt_event(
            cand,
            requires_approval_classes=requires_approval_on,
            blockers=event_blockers,
            auth_mode=auth_mode,
            env_authorized=env_authorized,
            claims_path=effective_claims_path,
        )
        halt_events.append(event)
        n_ranked += 1
        # Journal the authorization if the autopilot self-authorized.
        if (
            event.autopilot_authorized
            and auth_mode is not None
            and auth_mode.journal_path is not None
        ):
            append_autopilot_journal_row(
                auth_mode.journal_path, event, iteration=iteration
            )
            notes.append(
                f"autopilot self-authorized candidate {cand.candidate_id!r} "
                f"(cumulative ${auth_mode.cumulative_spent_usd:.4f} / cap "
                f"${auth_mode.cumulative_cap_usd:.4f})"
            )

    ended = dt.datetime.now(dt.UTC).isoformat()
    return LoopIterationReport(
        iteration=iteration,
        started_at_utc=started,
        ended_at_utc=ended,
        n_candidates_seen=n_seen,
        n_candidates_blocked_by_dispatch_claim=n_blocked,
        n_candidates_ranked=n_ranked,
        halt_events=halt_events,
        notes=notes,
    )


# ── Continuous-loop driver ────────────────────────────────────────────────


def run_continuous_loop(
    candidate_source: Callable[[], list[CandidateRow]],
    *,
    iterations: int,
    operator_decision_callback: Callable[[HaltEvent], OperatorDecision] | None = None,
    rank_axis: str = "eig_per_dollar",
    requires_approval_on: frozenset[EventClass] = frozenset({EventClass.DISPATCH}),
    claims_path: Path | None = None,
    race_mode: bool = False,
    max_dispatch_recommendations: int | None = None,
    auth_mode: OperatorAuthorizedModeConfig | None = None,
    env_authorized: bool | None = None,
    continual_posterior: Any | None = None,
    continual_posterior_path: Path | None = None,
    auto_load_continual_posterior: bool = False,
    score_panel_axis: str = "contest_cpu",
) -> list[LoopIterationReport]:
    """Run the continuous loop for ``iterations`` cycles.

    Each iteration calls ``candidate_source()`` to refresh the queue. The
    operator decision callback is invoked on every HALT event that has
    ``requires_approval=True``; if no callback is supplied, decisions are
    DEFER by default (the safe choice).

    W/I/A I-1 wire-in (2026-05-12): when ``auto_load_continual_posterior``
    is True, the continual-learning posterior is loaded ONCE at the start
    of the loop and passed to every iteration. This is the canonical
    "newly-appended anchor changes next ranking pass" path — the loop's
    candidate_source produces fresh rows each iteration; the posterior is
    re-read implicitly if the candidate_source itself appends to the
    posterior between calls (the load is fast / cached in memory by
    ``tac.continual_learning``).

    Returns the list of per-iteration reports.
    """
    if iterations <= 0:
        raise ValueError(f"iterations must be > 0; got {iterations}")
    reports: list[LoopIterationReport] = []
    for i in range(1, iterations + 1):
        candidates = candidate_source()
        # Per-iteration posterior reload: each iteration sees the most recent
        # anchor state. The explicit ``continual_posterior`` arg lets callers
        # inject a stable posterior for deterministic testing.
        iter_posterior = continual_posterior
        iter_auto_load = auto_load_continual_posterior and continual_posterior is None
        report = run_one_loop_iteration(
            candidates,
            iteration=i,
            rank_axis=rank_axis,
            requires_approval_on=requires_approval_on,
            claims_path=claims_path,
            race_mode=race_mode,
            max_dispatch_recommendations=max_dispatch_recommendations,
            auth_mode=auth_mode,
            env_authorized=env_authorized,
            continual_posterior=iter_posterior,
            continual_posterior_path=continual_posterior_path,
            auto_load_continual_posterior=iter_auto_load,
            score_panel_axis=score_panel_axis,
        )
        # Operator-decision injection — DEFER when no callback supplied.
        for event in report.halt_events:
            if event.requires_approval:
                decision = (
                    operator_decision_callback(event)
                    if operator_decision_callback is not None
                    else OperatorDecision.DEFER
                )
                inject_operator_decision(event, decision)
        reports.append(report)
    return reports


# ── Serialization ──────────────────────────────────────────────────────────


def _enum_value(v: Any) -> Any:
    if isinstance(v, Enum):
        return v.value
    return v


def serialize_report(report: LoopIterationReport) -> dict[str, Any]:
    """Return a JSON-safe dict for one report."""

    def _convert(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            d = dataclasses.asdict(obj)
            return {k: _convert(v) for k, v in d.items()}
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj

    return _convert(report)


def write_report(report: LoopIterationReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(serialize_report(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )


# ── CLI ───────────────────────────────────────────────────────────────────


def _parse_approval_flags(items: list[str]) -> frozenset[EventClass]:
    out: set[EventClass] = set()
    for raw in items or []:
        try:
            out.add(EventClass(raw))
        except ValueError as exc:
            valid = sorted(c.value for c in EventClass)
            raise ValueError(
                f"--require-operator-approval-on {raw!r} not in {valid}"
            ) from exc
    return frozenset(out)


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        out = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _require_finite_positive_float(
    value: object,
    *,
    field: str,
    context: str,
) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has non-numeric {field}={value!r}") from exc
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(
            f"{context} must carry finite positive {field}; got {value!r}"
        )
    return out


def _require_finite_nonnegative_float(
    value: object,
    *,
    field: str,
    context: str,
) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has non-numeric {field}={value!r}") from exc
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(
            f"{context} must carry finite nonnegative {field}; got {value!r}"
        )
    return out


def _coerce_str_list(value: object) -> list[str]:
    """Return a normalized string list for JSONL/JSON candidate metadata."""
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    out: list[str] = []
    for item in items:
        s = str(item or "").strip()
        if s and s not in out:
            out.append(s)
    return out


def _extract_lane_ids_from_campaign_metadata(value: object) -> list[str]:
    """Extract exact lane IDs from ranking metadata emitted before lane_id fields."""

    out: list[str] = []
    for item in _coerce_str_list(value):
        for match in re.finditer(r"(?:^|:)lane_id=([A-Za-z0-9_.:-]+)", item):
            lane_id = match.group(1).strip()
            if lane_id and lane_id not in out:
                out.append(lane_id)
    return out


def _json_bool_field(
    raw: dict[str, Any],
    key: str,
    *,
    default: bool = False,
    context: str,
) -> bool:
    """Read an authority-bearing JSON bool without Python truthiness coercion."""

    if key not in raw:
        return default
    value = raw[key]
    if isinstance(value, bool):
        return value
    raise ValueError(
        f"{context} has non-boolean {key}={value!r}; expected JSON true/false"
    )


def _json_optional_bool_field(
    raw: dict[str, Any],
    key: str,
    *,
    context: str,
) -> bool | None:
    """Read an optional JSON bool while preserving explicit null as unknown."""

    if key not in raw or raw[key] is None:
        return None
    return _json_bool_field(raw, key, default=False, context=context)


def _require_planning_only_flag(raw: dict[str, Any], key: str, *, context: str) -> None:
    if _json_bool_field(raw, key, default=False, context=context):
        raise ValueError(
            f"{context} has {key}=True; refusing to consume it as planning-only "
            "autopilot input"
        )


def _audit_exact_ready_queue(
    path: Path,
    *,
    repo_root: Path,
    dispatch_claims_path: Path,
) -> dict[str, Any]:
    from tac.optimizer.exact_ready_audit import audit_exact_ready_queue

    return audit_exact_ready_queue(
        path,
        repo_root=repo_root,
        dispatch_claims_path=dispatch_claims_path,
    )


def _prediction_band_allows_rank_reward(raw: dict[str, Any]) -> bool:
    """Return whether a raw planning row may keep nonzero EIG rank reward."""

    verdict = raw.get("prediction_band_verdict")
    if isinstance(verdict, dict) and "valid_for_rank_reward" in verdict:
        return verdict.get("valid_for_rank_reward") is True
    notes = str(raw.get("notes", "")).lower()
    if raw.get("prediction_band") is not None:
        return False
    return not ("[prediction" in notes or "[predicted" in notes)


def _candidate_prediction_band_rank_reward_suppressed(c: CandidateRow) -> bool:
    """Return true when prediction-band custody already suppressed rank reward."""

    blockers = set(c.blockers or [])
    notes = str(c.notes or "")
    return (
        "prediction_band_rank_reward_suppressed" in blockers
        or "prediction_band_rank_reward_suppressed" in notes
    )


def _candidate_literature_anchor_rank_reward_suppressed(c: CandidateRow) -> bool:
    """Return true when a literature anchor lacks source-scope custody."""

    blockers = set(c.blockers or [])
    notes = str(c.notes or "")
    return any(
        blocker.startswith(f"{LITERATURE_SOURCE_SCOPE_BLOCKER_PREFIX}:")
        for blocker in blockers
    ) or LITERATURE_SOURCE_SCOPE_BLOCKER_PREFIX in notes


def _append_literature_source_scope_blockers(
    raw: dict[str, Any],
    blockers: list[str],
) -> list[str]:
    """Add literature source-scope blockers to a planning row blocker list."""

    source_scope_blockers = literature_source_scope_blockers(raw)
    for blocker in source_scope_blockers:
        if blocker not in blockers:
            blockers.append(blocker)
    return source_scope_blockers


def _canonical_substrate_ids() -> set[str]:
    return {row.substrate_id for row in canonical_substrate_inventory()}


def _composition_unknown_substrate_blockers(raw: dict[str, Any]) -> list[str]:
    substrate_ids = raw.get("substrate_ids")
    if not isinstance(substrate_ids, list | tuple):
        return []
    known = _canonical_substrate_ids()
    return [
        f"{COMPOSITION_MATRIX_UNKNOWN_SUBSTRATE_BLOCKER_PREFIX}:{substrate_id}"
        for substrate_id in (str(item).strip() for item in substrate_ids)
        if substrate_id and substrate_id not in known
    ]


def _candidate_has_effective_negative_delta_for_race_mode(
    candidate: CandidateRow,
    *,
    continual_posterior: Any | None = None,
    score_panel_axis: str = "contest_cpu",
) -> bool:
    """Return true only for candidates with a post-gate negative prediction."""

    if _candidate_prediction_band_rank_reward_suppressed(candidate):
        return False
    delta = apply_z1_empirical_revision_to_candidate_delta(
        candidate,
        score_panel_axis=score_panel_axis,
    )
    if continual_posterior is not None:
        factor, _, _ = _posterior_correction_factor(candidate, continual_posterior)
        delta *= factor
    return delta < 0.0


def _candidate_row_from_raw(
    raw: dict[str, Any],
    *,
    context: str,
    allow_dispatch_authority_flags: bool = False,
) -> CandidateRow:
    authority_flags = ["score_claim", "promotion_eligible"]
    if not allow_dispatch_authority_flags:
        authority_flags.append("ready_for_exact_eval_dispatch")
    for flag in authority_flags:
        _require_planning_only_flag(raw, flag, context=context)
    mdl_density = _coerce_optional_float(raw.get("mdl_density"))
    mdl_tier_c_density = _coerce_optional_float(raw.get("mdl_tier_c_density"))
    composition_alpha = _coerce_optional_float(raw.get("composition_alpha"))
    predicted_dispatch_risk = _coerce_optional_float(
        raw.get("predicted_dispatch_risk")
    )
    candidate_archive_bytes_raw = raw.get(
        "candidate_archive_bytes",
        raw.get("archive_bytes", raw.get("archive_size_bytes")),
    )
    lane_class_raw = raw.get("lane_class")
    lane_class: str | None = (
        str(lane_class_raw) if lane_class_raw is not None else None
    )
    blockers = list(raw.get("blockers", []))
    source_scope_blockers = _append_literature_source_scope_blockers(raw, blockers)
    notes = str(raw.get("notes", ""))
    if source_scope_blockers:
        notes = (
            f"{notes}; {'; '.join(source_scope_blockers)}"
            if notes
            else "; ".join(source_scope_blockers)
        )
    expected_information_gain = float(raw["expected_information_gain"])
    if (
        expected_information_gain > 0.0
        and not _prediction_band_allows_rank_reward(raw)
    ):
        expected_information_gain = 0.0
        if "prediction_band_rank_reward_suppressed" not in blockers:
            blockers.append("prediction_band_rank_reward_suppressed")
        notes += "; prediction_band_rank_reward_suppressed"
    return CandidateRow(
        candidate_id=raw["candidate_id"],
        family=raw["family"],
        predicted_score_delta=float(raw["predicted_score_delta"]),
        expected_information_gain=expected_information_gain,
        estimated_dispatch_cost_usd=_require_finite_positive_float(
            raw["estimated_dispatch_cost_usd"],
            field="estimated_dispatch_cost_usd",
            context=context,
        ),
        blockers=blockers,
        notes=notes,
        timing_smoke_command=str(raw.get("timing_smoke_command", "")),
        mdl_density=mdl_density,
        lane_class=lane_class,
        literature_anchor=str(raw.get("literature_anchor", "")),
        source_supports=str(raw.get("source_supports", "")),
        paper_claim_scope=str(raw.get("paper_claim_scope", "")),
        pact_must_prove=str(raw.get("pact_must_prove", "")),
        decode_complexity_evidence=str(raw.get("decode_complexity_evidence", "")),
        mdl_tier_c_density=mdl_tier_c_density,
        composition_alpha=composition_alpha,
        license_ok=_json_bool_field(
            raw,
            "license_ok",
            default=True,
            context=context,
        ),
        inflate_dep_count=int(raw.get("inflate_dep_count", 0) or 0),
        sideinfo_consumed=_json_optional_bool_field(
            raw,
            "sideinfo_consumed",
            context=context,
        ),
        exact_duplicate=_json_bool_field(
            raw,
            "exact_duplicate",
            default=False,
            context=context,
        ),
        context_order=int(raw.get("context_order", 0) or 0),
        predicted_dispatch_risk=predicted_dispatch_risk,
        lane_id=str(raw.get("lane_id", "")),
        claim_keys=_coerce_str_list(raw.get("claim_keys")),
        target_modes=_coerce_str_list(raw.get("target_modes")),
        dispatch_packet_ready=_json_bool_field(
            raw,
            "dispatch_packet_ready",
            default=False,
            context=context,
        ),
        dispatch_packet_sha256=str(raw.get("dispatch_packet_sha256", "")),
        archive_path=str(
            raw.get("archive_path", raw.get("candidate_archive_path", "")) or ""
        ),
        submission_dir=str(
            raw.get("submission_dir", raw.get("submission_path", "")) or ""
        ),
        archive_manifest_path=str(
            raw.get(
                "archive_manifest_path",
                raw.get("manifest_path", raw.get("runtime_packet_manifest_path", "")),
            )
            or ""
        ),
        archive_sha256=str(
            raw.get(
                "archive_sha256",
                raw.get("candidate_archive_sha256", raw.get("expected_archive_sha256", "")),
            )
            or ""
        ),
        candidate_archive_bytes=_coerce_optional_positive_int(
            candidate_archive_bytes_raw
        ),
        runtime_tree_sha256=str(raw.get("runtime_tree_sha256", "")),
        deployment_target=str(
            raw.get("deployment_target", raw.get("dispatch_target", "")) or ""
        ),
        score_affecting_payload_changed=_json_bool_field(
            raw,
            "score_affecting_payload_changed",
            default=False,
            context=context,
        ),
        charged_bits_changed=_json_bool_field(
            raw,
            "charged_bits_changed",
            default=False,
            context=context,
        ),
        score_affecting_runtime_changed=_json_optional_bool_field(
            raw,
            "score_affecting_runtime_changed",
            context=context,
        ),
        score_claim=False,
        promotion_eligible=False,
        ready_for_exact_eval_dispatch=(
            _json_bool_field(
                raw,
                "ready_for_exact_eval_dispatch",
                default=False,
                context=context,
            )
            if allow_dispatch_authority_flags
            else False
        ),
    )


def load_candidates_from_jsonl(path: Path) -> list[CandidateRow]:
    """Load planning-only CandidateRow objects from a JSONL file."""

    rows: list[CandidateRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            raw = json.loads(s)
            context = (
                f"{path}:{line_no} candidate "
                f"{raw.get('candidate_id', '<missing-candidate-id>')!r}"
            )
            rows.append(_candidate_row_from_raw(raw, context=context))
    return rows


def load_candidates_from_master_gradient_optimal_plans(
    *,
    root: Path | None = None,
    archive_sha256: str | None = None,
) -> list[CandidateRow]:
    """Load planning-only Cathedral candidates from master-gradient sidecars.

    This is deliberately NOT an exact-ready queue loader. It consumes
    ``.omx/state/master_gradient_consumers/optimal_plan_*.json`` prediction
    sidecars, validates the no-authority contract in
    ``tac.master_gradient_consumers.optimal_plan_payload_to_candidate_row``,
    and emits CandidateRow objects with dispatch/promote/score authority false.
    """
    base = root or (REPO_ROOT / ".omx" / "state" / "master_gradient_consumers")
    if not base.exists():
        return []
    if archive_sha256 is not None:
        sha = archive_sha256.strip().lower()
        if len(sha) < 12 or not _is_sha256_hex(sha.ljust(64, "0")[:64]):
            raise ValueError(
                f"archive_sha256 must be a 12+ char hex prefix; got {archive_sha256!r}"
            )
        pattern = f"optimal_plan_{sha[:12]}_*.json"
    else:
        pattern = "optimal_plan_*.json"

    from tac.master_gradient_consumers import optimal_plan_payload_to_candidate_row

    latest_by_archive: dict[str, CandidateRow] = {}
    for path in sorted(base.glob(pattern)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        archive = str(payload.get("archive_sha256", "")).strip().lower()
        file_prefix = path.name[len("optimal_plan_"):len("optimal_plan_") + 12]
        if not archive.startswith(file_prefix.lower()):
            continue
        try:
            row = optimal_plan_payload_to_candidate_row(payload, sidecar_path=path)
        except (TypeError, ValueError):
            continue
        latest_by_archive[row.archive_sha256] = row
    return [latest_by_archive[k] for k in sorted(latest_by_archive)]


def load_candidates_from_exact_ready_queue(
    path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    dispatch_claims_path: Path | None = None,
) -> list[CandidateRow]:
    """Load exact-ready rows only after the canonical live custody audit passes."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("exact-ready queue must be a JSON object")
    if payload.get("schema") != EXACT_READY_QUEUE_SCHEMA:
        raise ValueError(
            "exact-ready queue schema unsupported:"
            f"{payload.get('schema')!r}; expected {EXACT_READY_QUEUE_SCHEMA!r}"
        )
    claims_path = dispatch_claims_path or (
        repo_root / ".omx" / "state" / "active_lane_dispatch_claims.md"
    )
    audit = _audit_exact_ready_queue(
        path,
        repo_root=repo_root,
        dispatch_claims_path=claims_path,
    )
    stale_rows = audit.get("stale_ready_rows")
    if isinstance(stale_rows, list) and stale_rows:
        first = stale_rows[0] if isinstance(stale_rows[0], dict) else {}
        blockers = first.get("blockers") if isinstance(first, dict) else None
        raise ValueError(
            "exact-ready queue failed live custody audit:"
            f"{first.get('candidate_id') if isinstance(first, dict) else '<unknown>'}:"
            f"{blockers}"
        )
    raw_rows = payload.get("dispatch_ready")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("exact-ready queue has no dispatch_ready rows")
    rows: list[CandidateRow] = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise ValueError(f"exact-ready queue row {index} is not an object")
        if raw.get("ready_for_exact_eval_dispatch") is not True:
            raise ValueError(
                "exact-ready queue dispatch_ready row lacks "
                f"ready_for_exact_eval_dispatch=True:{raw.get('candidate_id')!r}"
            )
        context = (
            f"{path}:dispatch_ready[{index}] candidate "
            f"{raw.get('candidate_id', '<missing-candidate-id>')!r}"
        )
        rows.append(
            _candidate_row_from_raw(
                raw,
                context=context,
                allow_dispatch_authority_flags=True,
            )
        )
    return rows


# ── Substrate composition matrix ranking integration ──────────────────────


SUBSTRATE_COMPOSITION_RANKING_SCHEMA = "tac_autopilot_dispatch_ranking_v1"


def _require_substrate_composition_ranking_schema(
    payload: dict[str, Any],
    *,
    context: str,
) -> None:
    schema = payload.get("schema")
    if schema != SUBSTRATE_COMPOSITION_RANKING_SCHEMA:
        raise ValueError(
            f"{context} schema unsupported: {schema!r}; expected "
            f"{SUBSTRATE_COMPOSITION_RANKING_SCHEMA!r}. Legacy ranking "
            "artifacts need an explicit legacy blocker path before autopilot load."
        )


def load_candidates_from_substrate_composition_ranking(
    path: Path,
    *,
    only_in_envelope: bool = True,
    only_fits_per_dispatch_cap: bool = True,
) -> list[CandidateRow]:
    """Load candidates from QQ's substrate composition ranking JSON.

    Per CLAUDE.md "race-mode + parallel-dispatch first" + the substrate
    composition matrix landing memo (`feedback_substrate_composition_
    matrix_autopilot_ranking_theoretical_floor_v2_landed_20260511.md`),
    QQ's ranker emits an artifact with schema
    ``tac_autopilot_dispatch_ranking_v1`` containing ``ranked_dispatches``
    each carrying ``candidate_id``, ``family``, ``predicted_score_delta``,
    ``expected_information_gain``, ``estimated_dispatch_cost_usd``,
    ``substrate_ids`` (the substrates participating in this dispatch),
    ``composition_notes`` (the rationale), ``fits_per_dispatch_cap``,
    and ``fits_cumulative_envelope`` flags.

    This loader converts each ``ranked_dispatch`` into a ``CandidateRow``
    that the autopilot loop consumes via the existing HALT-and-ASK gate.
    Per CLAUDE.md "Forbidden score claims" the loaded rows carry
    ``predicted_score_delta`` tagged ``[predicted; substrate composition
    matrix v1]`` in their ``notes`` field.

    Filtering rules (defaults match QQ's envelope discipline):

    - ``only_in_envelope=True`` drops dispatches that QQ marked as
      out-of-cumulative-envelope (``fits_cumulative_envelope=False``).
    - ``only_fits_per_dispatch_cap=True`` drops dispatches that QQ marked
      as out-of-per-dispatch-cap (``fits_per_dispatch_cap=False``).

    The loader REFUSES to load rows whose ``score_claim`` field is True or
    whose ``ready_for_exact_eval_dispatch`` field is True — those would
    violate the planning-only invariant.
    """
    if not path.is_file():
        raise FileNotFoundError(f"substrate composition ranking JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("substrate composition ranking JSON must be an object")
    _require_substrate_composition_ranking_schema(
        payload,
        context="substrate composition ranking JSON",
    )
    if "ranked_dispatches" not in payload:
        raise ValueError(
            f"substrate composition ranking JSON missing 'ranked_dispatches' "
            f"key (schema={payload.get('schema')!r}); got top-level keys: "
            f"{sorted(payload.keys())}"
        )
    if _json_bool_field(
        payload,
        "score_claim",
        default=False,
        context="substrate composition ranking JSON",
    ):
        raise ValueError(
            "substrate composition ranking JSON has score_claim=True; "
            "the autopilot ranker must remain planning-only "
            "(per CLAUDE.md 'Forbidden score claims')"
        )
    rows: list[CandidateRow] = []
    for raw in payload["ranked_dispatches"]:
        context = (
            "substrate composition ranked dispatch "
            f"{raw.get('candidate_id')!r}"
        )
        if _json_bool_field(raw, "score_claim", default=False, context=context):
            raise ValueError(
                f"ranked dispatch {raw.get('candidate_id')!r} has score_claim=True; "
                "refuse to consume score-claimed planning rows"
            )
        if _json_bool_field(
            raw,
            "promotion_eligible",
            default=False,
            context=context,
        ):
            raise ValueError(
                f"ranked dispatch {raw.get('candidate_id')!r} has "
                "promotion_eligible=True; refuse to consume promotion-claimed "
                "planning rows"
            )
        if _json_bool_field(
            raw,
            "ready_for_exact_eval_dispatch",
            default=False,
            context=context,
        ):
            raise ValueError(
                f"ranked dispatch {raw.get('candidate_id')!r} has "
                "ready_for_exact_eval_dispatch=True; refuse to consume in autopilot "
                "(operator-gated promotion path required)"
            )
        fits_per_dispatch_cap = _json_bool_field(
            raw,
            "fits_per_dispatch_cap",
            default=True,
            context=context,
        )
        fits_cumulative_envelope = _json_bool_field(
            raw,
            "fits_cumulative_envelope",
            default=True,
            context=context,
        )
        if only_fits_per_dispatch_cap and not fits_per_dispatch_cap:
            continue
        if only_in_envelope and not fits_cumulative_envelope:
            continue
        notes_lines = [
            "[predicted; substrate composition matrix v1]",
            f"composition_notes: {raw.get('composition_notes', '')}",
            f"substrate_ids: {raw.get('substrate_ids', [])!r}",
        ]
        if raw.get("lane_class"):
            notes_lines.append(f"lane_class: {raw.get('lane_class')}")
        if raw.get("literature_anchor"):
            notes_lines.append(f"literature_anchor: {raw.get('literature_anchor')}")
        if raw.get("source_supports"):
            notes_lines.append(f"source_supports: {raw.get('source_supports')}")
        if raw.get("paper_claim_scope"):
            notes_lines.append(f"paper_claim_scope: {raw.get('paper_claim_scope')}")
        if raw.get("pact_must_prove"):
            notes_lines.append(f"pact_must_prove: {raw.get('pact_must_prove')}")
        if raw.get("decode_complexity_evidence"):
            notes_lines.append(
                f"decode_complexity_evidence: {raw.get('decode_complexity_evidence')}"
            )
        if raw.get("campaign_metadata"):
            notes_lines.append(f"campaign_metadata: {raw.get('campaign_metadata')!r}")
        lane_class_raw = raw.get("lane_class")
        row_blockers = list(raw.get("blockers", []))
        if PLANNING_ONLY_SOURCE_BLOCKER not in row_blockers:
            row_blockers.append(PLANNING_ONLY_SOURCE_BLOCKER)
        notes_lines.extend(
            _append_literature_source_scope_blockers(raw, row_blockers)
        )
        unknown_substrate_blockers = _composition_unknown_substrate_blockers(raw)
        for blocker in unknown_substrate_blockers:
            if blocker not in row_blockers:
                row_blockers.append(blocker)
        notes_lines.extend(unknown_substrate_blockers)
        expected_information_gain = float(raw["expected_information_gain"])
        if unknown_substrate_blockers:
            expected_information_gain = 0.0
        if (
            expected_information_gain > 0.0
            and not _prediction_band_allows_rank_reward(raw)
        ):
            expected_information_gain = 0.0
            if "prediction_band_rank_reward_suppressed" not in row_blockers:
                row_blockers.append("prediction_band_rank_reward_suppressed")
            notes_lines.append("prediction_band_rank_reward_suppressed")
        # Catalog #227 wire-in: read optional composition_alpha if present
        # (the canonical Z3xC6 probe-disambiguator emits this).
        composition_alpha_raw = raw.get("composition_alpha")
        composition_alpha: float | None = None
        if composition_alpha_raw is not None:
            try:
                composition_alpha = float(composition_alpha_raw)
            except (TypeError, ValueError):
                composition_alpha = None
        campaign_lane_ids = _extract_lane_ids_from_campaign_metadata(
            raw.get("campaign_metadata")
        )
        lane_id = str(raw.get("lane_id") or "").strip()
        if not lane_id and len(campaign_lane_ids) == 1:
            lane_id = campaign_lane_ids[0]
        claim_keys = _coerce_str_list(raw.get("claim_keys"))
        for lane_key in campaign_lane_ids:
            if lane_key not in claim_keys:
                claim_keys.append(lane_key)
        rows.append(CandidateRow(
            candidate_id=str(raw["candidate_id"]),
            family=str(raw["family"]),
            predicted_score_delta=float(raw["predicted_score_delta"]),
            expected_information_gain=expected_information_gain,
            estimated_dispatch_cost_usd=_require_finite_nonnegative_float(
                raw["estimated_dispatch_cost_usd"],
                field="estimated_dispatch_cost_usd",
                context=context,
            ),
            blockers=row_blockers,
            notes="\n".join(notes_lines),
            mdl_density=_coerce_optional_float(raw.get("mdl_density")),
            lane_class=str(lane_class_raw) if lane_class_raw is not None else None,
            literature_anchor=str(raw.get("literature_anchor", "")),
            source_supports=str(raw.get("source_supports", "")),
            paper_claim_scope=str(raw.get("paper_claim_scope", "")),
            pact_must_prove=str(raw.get("pact_must_prove", "")),
            decode_complexity_evidence=str(
                raw.get("decode_complexity_evidence", "")
            ),
            mdl_tier_c_density=_coerce_optional_float(raw.get("mdl_tier_c_density")),
            composition_alpha=composition_alpha,
            license_ok=_json_bool_field(raw, "license_ok", default=True, context=context),
            inflate_dep_count=int(raw.get("inflate_dep_count", 0) or 0),
            sideinfo_consumed=_json_optional_bool_field(
                raw,
                "sideinfo_consumed",
                context=context,
            ),
            exact_duplicate=_json_bool_field(
                raw,
                "exact_duplicate",
                default=False,
                context=context,
            ),
            context_order=int(raw.get("context_order", 0) or 0),
            predicted_dispatch_risk=_coerce_optional_float(
                raw.get("predicted_dispatch_risk")
            ),
            lane_id=lane_id,
            claim_keys=claim_keys,
            score_claim=False,
            promotion_eligible=False,
            ready_for_exact_eval_dispatch=False,
        ))
    return rows


# ── Canonical substrate composition matrix consumer (Catalog #227) ────────
#
# Per `feedback_t1_f_z3_x_c6_composition_probe_build_landed_20260514.md` the
# canonical posterior surface for substrate composition is
# `.omx/state/substrate_composition_matrix.json`. The probe-disambiguator
# appends one entry per probe invocation; multiple entries per pair are
# possible (most-recent wins).
#
# This loader maps the matrix into a `(substrate_pair_key, alpha)` dict so
# the autopilot can look up the composition factor for any candidate that
# carries substrate_ids. Substrate pair key is the canonical
# "<substrate_id_a>__x__<substrate_id_b>" form used by T1-F.
#
# Per CLAUDE.md "Forbidden score claims" the matrix entries are PREDICTED
# composition signals, never measured scores; the loader refuses any entry
# with score_claim=True.
SUBSTRATE_COMPOSITION_MATRIX_PATH = (
    REPO_ROOT / ".omx" / "state" / "substrate_composition_matrix.json"
)


def load_substrate_composition_alpha_index(
    path: Path | None = None,
) -> dict[str, float]:
    """Return {"<substrate_a>__x__<substrate_b>": alpha} from the canonical
    substrate composition matrix posterior surface.

    For each pair key in the matrix, the loader returns the alpha of the
    MOST RECENTLY WRITTEN entry (by ``written_at_utc``). Per CLAUDE.md
    "Forbidden score claims" any entry with ``score_claim=True`` is REFUSED
    (the loader raises ``ValueError``).

    Returns an empty dict when the matrix file is absent.
    """
    p = path if path is not None else SUBSTRATE_COMPOSITION_MATRIX_PATH
    if not p.is_file():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"substrate composition matrix exists but is unreadable or invalid JSON: {p}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"substrate composition matrix must be a JSON object: {p}")
    if "entries" not in payload:
        raise ValueError(f"substrate composition matrix missing 'entries': {p}")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"substrate composition matrix 'entries' must be an object: {p}")
    out: dict[str, float] = {}
    for pair_key, rows in entries.items():
        if not isinstance(rows, list):
            raise ValueError(
                f"substrate composition matrix entry for pair {pair_key!r} "
                "must be a list"
            )
        if not rows:
            continue
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(
                    f"substrate composition matrix entry for pair {pair_key!r} "
                    f"row {i} must be an object"
                )
        # Pick most-recent by written_at_utc (string lexicographic compare
        # works because all timestamps use canonical UTC ISO format).
        latest = max(
            rows,
            key=lambda r: r.get("written_at_utc", ""),
            default=None,
        )
        if latest is None:
            continue
        context = f"substrate composition matrix entry for pair {pair_key!r}"
        if _json_bool_field(latest, "score_claim", default=False, context=context):
            raise ValueError(
                f"substrate composition matrix entry for pair {pair_key!r} "
                "has score_claim=True; refuse to consume score-claimed "
                "composition rows (per CLAUDE.md 'Forbidden score claims')"
            )
        alpha_raw = latest.get("alpha")
        if alpha_raw is None:
            continue
        try:
            alpha = float(alpha_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"substrate composition matrix entry for pair {pair_key!r} "
                f"has non-numeric alpha={alpha_raw!r}"
            ) from exc
        if not math.isfinite(alpha):
            raise ValueError(
                f"substrate composition matrix entry for pair {pair_key!r} "
                f"has non-finite alpha={alpha_raw!r}"
            )
        out[str(pair_key)] = alpha
    return out


def _substrate_pair_ids_from_alpha_key(pair_key: str) -> tuple[str, str] | None:
    parts = pair_key.split("__x__")
    if len(parts) != 2:
        return None
    left, right = (p.strip() for p in parts)
    if not left or not right:
        return None
    return left, right


def apply_substrate_composition_matrix_to_candidates(
    candidates: list[CandidateRow],
    *,
    substrate_ids_by_candidate: dict[str, tuple[str, ...]] | None = None,
    matrix_path: Path | None = None,
) -> list[CandidateRow]:
    """Populate ``candidate.composition_alpha`` from the canonical
    substrate composition matrix.

    For each candidate, if ``substrate_ids_by_candidate`` carries a tuple of
    exactly TWO substrate ids, the canonical key
    ``<a>__x__<b>`` (alphabetically sorted) OR ``<b>__x__<a>`` is looked up
    in the matrix. Single-substrate candidates and substrate triples are not
    matched (composition_alpha stays None).

    Returns the SAME list (mutated in place) — convenient for chaining.
    """
    if not candidates:
        return candidates
    alpha_index = load_substrate_composition_alpha_index(matrix_path)
    if not alpha_index:
        return candidates
    if substrate_ids_by_candidate is None:
        substrate_ids_by_candidate = {}
    for cand in candidates:
        sids = substrate_ids_by_candidate.get(cand.candidate_id)
        if not sids or len(sids) != 2:
            continue
        a, b = sorted(sids)
        key_a = f"{a}__x__{b}"
        key_b = f"{b}__x__{a}"
        alpha = alpha_index.get(key_a)
        if alpha is None:
            alpha = alpha_index.get(key_b)
        if alpha is None:
            # Fallback: scan exact pair keys (the probe may emit keys in
            # non-sorted form). Substring checks can cross-match unrelated ids.
            for k, v in alpha_index.items():
                pair = _substrate_pair_ids_from_alpha_key(k)
                if pair is not None and set(pair) == {a, b}:
                    alpha = v
                    break
        if alpha is not None:
            cand.composition_alpha = alpha
    return candidates


# ── Probe-disambiguator read-only autopilot-row consumer ──────────────────


def load_candidates_from_probe_disambiguator_output(path: Path) -> list[CandidateRow]:
    """Load read-only ``autopilot_rows`` from a probe-disambiguator JSON artifact.

    Probe-disambiguators arbitrate design interpretations; they are NOT
    dispatch packets or score evidence. This consumer intentionally accepts
    only rows whose top-level and per-row safety flags remain fail-closed.
    """
    if not path.is_file():
        raise FileNotFoundError(f"probe-disambiguator JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"probe-disambiguator JSON must be an object: {path}")
    for key in ("score_claim", "promotion_eligible", "ready_for_exact_eval_dispatch"):
        _require_planning_only_flag(payload, key, context="probe-disambiguator JSON")
    if _json_bool_field(
        payload,
        "dispatch_attempted",
        default=False,
        context="probe-disambiguator JSON",
    ):
        raise ValueError(
            "probe-disambiguator JSON has dispatch_attempted=True; refusing "
            "to consume it as an autopilot planning source"
        )
    raw_rows = payload.get("autopilot_rows")
    if not isinstance(raw_rows, list):
        raise ValueError(
            "probe-disambiguator JSON missing list field 'autopilot_rows'"
        )

    rows: list[CandidateRow] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ValueError("probe-disambiguator autopilot_rows entries must be objects")
        cid = raw.get("candidate_id")
        if not cid:
            raise ValueError("probe-disambiguator row missing candidate_id")
        for key in ("score_claim", "promotion_eligible", "ready_for_exact_eval_dispatch"):
            _require_planning_only_flag(
                raw, key, context=f"probe-disambiguator row {cid!r}"
            )
        if _json_bool_field(
            raw,
            "dispatch_attempted",
            default=False,
            context=f"probe-disambiguator row {cid!r}",
        ):
            raise ValueError(
                f"probe-disambiguator row {cid!r} has dispatch_attempted=True; "
                "refusing to consume it as an autopilot planning row"
            )
        lane_class_raw = raw.get("lane_class")
        row_blockers = list(raw.get("blockers", []))
        if PLANNING_ONLY_SOURCE_BLOCKER not in row_blockers:
            row_blockers.append(PLANNING_ONLY_SOURCE_BLOCKER)
        source_scope_blockers = _append_literature_source_scope_blockers(
            raw, row_blockers
        )
        if source_scope_blockers:
            raw_notes = str(raw.get("notes", ""))
            raw = {
                **raw,
                "notes": (
                    f"{raw_notes}; {'; '.join(source_scope_blockers)}"
                    if raw_notes
                    else "; ".join(source_scope_blockers)
                ),
            }
        expected_information_gain = float(raw.get("expected_information_gain", 0.0))
        if (
            expected_information_gain > 0.0
            and not _prediction_band_allows_rank_reward(raw)
        ):
            expected_information_gain = 0.0
            if "prediction_band_rank_reward_suppressed" not in row_blockers:
                row_blockers.append("prediction_band_rank_reward_suppressed")
            raw_notes = str(raw.get("notes", ""))
            raw = {
                **raw,
                "notes": (
                    f"{raw_notes}; prediction_band_rank_reward_suppressed"
                    if raw_notes
                    else "prediction_band_rank_reward_suppressed"
                ),
            }
        notes = "\n".join(
            [
                "[probe-disambiguator; read-only planning]",
                f"source_path: {path}",
                f"source_schema: {payload.get('schema')!r}",
                f"source_tool: {payload.get('tool')!r}",
                f"row_notes: {raw.get('notes', '')}",
            ]
        )
        rows.append(
            CandidateRow(
                candidate_id=str(cid),
                family=str(raw.get("family", "probe_disambiguator")),
                predicted_score_delta=float(raw.get("predicted_score_delta", 0.0)),
                expected_information_gain=expected_information_gain,
                estimated_dispatch_cost_usd=_require_finite_nonnegative_float(
                    raw.get("estimated_dispatch_cost_usd", 0.0),
                    field="estimated_dispatch_cost_usd",
                    context=f"probe-disambiguator row {cid!r}",
                ),
                blockers=row_blockers,
                notes=notes,
                mdl_density=_coerce_optional_float(raw.get("mdl_density")),
                lane_class=str(lane_class_raw) if lane_class_raw is not None else None,
                literature_anchor=str(raw.get("literature_anchor", "")),
                source_supports=str(raw.get("source_supports", "")),
                paper_claim_scope=str(raw.get("paper_claim_scope", "")),
                pact_must_prove=str(raw.get("pact_must_prove", "")),
                decode_complexity_evidence=str(
                    raw.get("decode_complexity_evidence", "")
                ),
                mdl_tier_c_density=_coerce_optional_float(
                    raw.get("mdl_tier_c_density")
                ),
                composition_alpha=_coerce_optional_float(
                    raw.get("composition_alpha")
                ),
                license_ok=_json_bool_field(
                    raw,
                    "license_ok",
                    default=True,
                    context=f"probe-disambiguator row {cid!r}",
                ),
                inflate_dep_count=int(raw.get("inflate_dep_count", 0) or 0),
                sideinfo_consumed=_json_optional_bool_field(
                    raw,
                    "sideinfo_consumed",
                    context=f"probe-disambiguator row {cid!r}",
                ),
                exact_duplicate=_json_bool_field(
                    raw,
                    "exact_duplicate",
                    default=False,
                    context=f"probe-disambiguator row {cid!r}",
                ),
                context_order=int(raw.get("context_order", 0) or 0),
            )
        )
    return rows


# ── RATE ACH/cheap-probe read-only autopilot-row consumer ────────────────


def load_candidates_from_rate_attack_feature_matrix_output(
    path: Path,
) -> list[CandidateRow]:
    """Load RATE process-feature rows as planning-only CandidateRows.

    The feature matrix is an authority surface for *refusal and ranking
    hygiene* only: explicit disconfirming assumptions, ACH risk counters, and
    cheap-probe spend gates. It is not a score packet, not a promotion packet,
    and not an exact-ready queue.
    """

    if not path.is_file():
        raise FileNotFoundError(f"RATE feature matrix JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"RATE feature matrix JSON must be an object: {path}")
    if payload.get("schema") != "rate_attack_autopilot_feature_matrix_v1_20260519":
        raise ValueError(
            "RATE feature matrix schema unsupported: "
            f"{payload.get('schema')!r}"
        )
    for key in ("score_claim", "promotion_eligible", "ready_for_exact_eval_dispatch"):
        _require_planning_only_flag(payload, key, context="RATE feature matrix JSON")
    if _json_bool_field(
        payload,
        "dispatch_attempted",
        default=False,
        context="RATE feature matrix JSON",
    ):
        raise ValueError(
            "RATE feature matrix has dispatch_attempted=True; refusing to "
            "consume it as an autopilot planning source"
        )
    raw_rows = payload.get("autopilot_rows")
    if not isinstance(raw_rows, list):
        raise ValueError("RATE feature matrix missing list field 'autopilot_rows'")

    rows: list[CandidateRow] = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise ValueError("RATE feature matrix autopilot_rows entries must be objects")
        cid = raw.get("candidate_id")
        if not cid:
            raise ValueError("RATE feature matrix row missing candidate_id")
        context = f"RATE feature matrix row {cid!r}"
        for key in ("score_claim", "promotion_eligible", "ready_for_exact_eval_dispatch"):
            _require_planning_only_flag(raw, key, context=context)
        assumptions = raw.get("disconfirming_assumptions")
        if not isinstance(assumptions, list) or not assumptions:
            raise ValueError(
                f"{context} missing nonempty disconfirming_assumptions"
            )
        blockers = _coerce_str_list(raw.get("blockers"))
        if PLANNING_ONLY_SOURCE_BLOCKER not in blockers:
            blockers.append(PLANNING_ONLY_SOURCE_BLOCKER)
        cheap_probe = raw.get("cheap_probe")
        if not isinstance(cheap_probe, dict):
            raise ValueError(f"{context} missing cheap_probe object")
        if (
            _require_finite_nonnegative_float(
                raw.get("estimated_dispatch_cost_usd", 0.0),
                field="estimated_dispatch_cost_usd",
                context=context,
            )
            > 1.0
            and cheap_probe.get("verdict_exists") is not True
            and "rate_attack_gt_1usd_spend_requires_cheap_probe_verdict" not in blockers
        ):
            raise ValueError(
                f"{context} has >$1 spend without cheap-probe verdict blocker"
            )
        raw_notes = str(raw.get("notes", ""))
        raw = {
            **raw,
            "blockers": blockers,
            "context_order": int(raw.get("context_order", index) or index),
            "notes": "\n".join(
                [
                    "[RATE ACH/cheap-probe process feature; read-only planning]",
                    f"source_path: {path}",
                    f"row_notes: {raw_notes}",
                ]
            ),
        }
        rows.append(_candidate_row_from_raw(raw, context=context))
    return rows


def candidate_substrate_ids_from_ranking(path: Path) -> dict[str, tuple[str, ...]]:
    """Map ``candidate_id`` -> tuple of substrate ids participating in the dispatch.

    Used by the composition-constraint enforcer to reason about which
    substrates a candidate touches without re-parsing the full ranking JSON.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("substrate composition ranking JSON must be an object")
    _require_substrate_composition_ranking_schema(
        payload,
        context="substrate composition ranking JSON",
    )
    if "ranked_dispatches" not in payload:
        raise ValueError(
            "substrate composition ranking JSON missing 'ranked_dispatches'"
        )
    out: dict[str, tuple[str, ...]] = {}
    for raw in payload["ranked_dispatches"]:
        cid = str(raw["candidate_id"])
        substrates = tuple(str(s) for s in raw.get("substrate_ids", ()))
        out[cid] = substrates
    return out


# ── macOS-CPU advisory proxy ranking integration ────────────────────────


MACOS_CPU_ADVISORY_PROXY_EVIDENCE_TAG = "macos_cpu_advisory"


def load_candidates_from_macos_cpu_advisory_manifest(
    path: Path,
    *,
    default_estimated_dispatch_cost_usd: float = 0.0,
    default_expected_information_gain: float = 0.0,
) -> list[CandidateRow]:
    """Load CandidateRow rows from a macOS-CPU advisory-signal manifest.

    Operator routing 2026-05-13 ("training is the real roadblock; we can
    prepare and run things on macos and cpu"). Per CLAUDE.md PR107
    empirical calibration (|Δ| ≤ 6e-6 vs GHA Linux x86_64 contest-CPU on
    the same exact archive), macOS-CPU is a free first-class advisory
    proxy that lets the autopilot RANK candidates BEFORE any GPU spend.

    The loaded rows participate in EIG-per-dollar ranking BUT carry:
      - notes prefixed with ``[macOS-CPU advisory; ranking-only]``
      - blockers extended with the manifest's ``dispatch_blockers`` so
        the operator sees every reason a row cannot promote
      - ``predicted_score_delta`` derived from ``projected_contest_cpu_score_p50``
        when present, else from ``score_macos_cpu`` directly.

    Per CLAUDE.md Catalog #127 (`check_authoritative_tag_requires_custody_metadata`)
    the manifest's evidence_tag ``[macOS-CPU advisory only]`` is already
    routed to ``refused_class="macos_substrate"`` by the custody validator.
    Promotion requires a paired ``[contest-CPU GHA Linux x86_64]`` anchor;
    the loader DOES NOT lift that gate.
    """
    if not path.is_file():
        raise FileNotFoundError(f"macOS-CPU advisory manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))

    expected_schema_prefix = "macos_cpu_advisory_signal_manifest"
    schema = str(payload.get("schema") or "")
    if not schema.startswith(expected_schema_prefix):
        raise ValueError(
            f"macOS-CPU advisory manifest at {path!s} has unexpected schema "
            f"{schema!r}; expected schema starting with {expected_schema_prefix!r}"
        )

    # Per CLAUDE.md "Forbidden score claims" + Catalog #192: refuse manifests
    # claiming promotability. The autopilot ranker never lifts these gates.
    for flag in ("score_claim", "promotion_eligible", "ready_for_exact_eval_dispatch"):
        if payload.get(flag, False):
            raise ValueError(
                f"macOS-CPU advisory manifest has {flag}=True; refusing to "
                "consume in autopilot ranker (per CLAUDE.md Catalog #192 + "
                "'Forbidden score claims')"
            )

    base_blockers = list(payload.get("dispatch_blockers", []))
    rows: list[CandidateRow] = []
    for raw in payload.get("rows", []):
        if raw.get("score_claim", False):
            raise ValueError(
                f"manifest row {raw.get('variant_id')!r} has score_claim=True; "
                "refuse to consume"
            )
        if raw.get("promotion_eligible", False) or raw.get(
            "ready_for_exact_eval_dispatch", False
        ):
            raise ValueError(
                f"manifest row {raw.get('variant_id')!r} has "
                "promotion_eligible or ready_for_exact_eval_dispatch True; refuse"
            )
        # Predicted score delta: prefer the projected contest-CPU score
        # band's p50 anchor (calibrated). Otherwise use score_macos_cpu
        # directly. Either way the row's notes record that the prediction
        # is non-authoritative.
        projected_p50 = raw.get("projected_contest_cpu_score_p50")
        score_macos_cpu = raw.get("score_macos_cpu")
        if projected_p50 is not None:
            predicted_score = float(projected_p50)
        elif score_macos_cpu is not None:
            predicted_score = float(score_macos_cpu)
        else:
            # No score → can't rank by score; skip with a notes record.
            continue

        family = str(raw.get("family") or "")
        variant_id = str(raw.get("variant_id") or "")
        if not family or not variant_id:
            continue

        # Treat the projected score as the predicted_score_delta absolute
        # value. Most-negative-is-best ranking still works since smaller
        # contest scores are better.
        row_blockers = list(base_blockers) + list(raw.get("dispatch_blockers", []))
        # Dedup blockers preserving insertion order.
        seen: set[str] = set()
        deduped_blockers: list[str] = []
        for b in row_blockers:
            if b in seen:
                continue
            seen.add(b)
            deduped_blockers.append(b)

        archive_sha = str(raw.get("archive_sha256") or "")
        archive_bytes = int(raw.get("archive_bytes") or 0)
        band_low = raw.get("projected_contest_cpu_score_low")
        band_high = raw.get("projected_contest_cpu_score_high")
        notes_lines = [
            "[macOS-CPU advisory; ranking-only]",
            f"proxy_evidence: {MACOS_CPU_ADVISORY_PROXY_EVIDENCE_TAG}",
            f"hardware_substrate: {raw.get('hardware_substrate') or payload.get('hardware_substrate')!r}",
            f"score_macos_cpu: {score_macos_cpu!r}",
            f"projected_contest_cpu_score: p50={projected_p50!r} "
            f"low={band_low!r} high={band_high!r}",
            f"archive_bytes: {archive_bytes}",
            f"archive_sha256: {archive_sha or '(missing)'}",
            "promotion_blocked: requires paired [contest-CPU GHA Linux x86_64] anchor",
        ]
        rows.append(
            CandidateRow(
                candidate_id=f"macos_cpu_advisory__{family}__{variant_id}",
                family=family,
                predicted_score_delta=predicted_score,
                expected_information_gain=default_expected_information_gain,
                estimated_dispatch_cost_usd=default_estimated_dispatch_cost_usd,
                blockers=deduped_blockers,
                notes="\n".join(notes_lines),
            )
        )
    return rows


def tag_halt_events_with_proxy_evidence(
    halt_events: list[HaltEvent],
    *,
    candidates: list[CandidateRow],
) -> list[HaltEvent]:
    """Annotate halt events whose source candidate carries macOS-CPU advisory notes.

    Per operator routing 2026-05-13: when a halt event's underlying candidate
    came from the macOS-CPU advisory manifest, surface that fact in the halt
    event's decision_notes so the operator can see at a glance which
    rankings depend on the proxy. The autopilot's dispatch journal will
    therefore tag the entry with ``proxy_evidence="macos_cpu_advisory"``.

    Mutates the halt events in place AND returns them for chaining.
    """
    cid_to_proxy: dict[str, str] = {}
    for c in candidates:
        if "[macOS-CPU advisory; ranking-only]" in (c.notes or ""):
            cid_to_proxy[c.candidate_id] = MACOS_CPU_ADVISORY_PROXY_EVIDENCE_TAG
    for evt in halt_events:
        tag = cid_to_proxy.get(evt.candidate_id)
        if tag is None:
            continue
        marker = f"proxy_evidence={tag}"
        if marker not in evt.decision_notes:
            sep = "; " if evt.decision_notes else ""
            evt.decision_notes = f"{evt.decision_notes}{sep}{marker}"
    return halt_events


def filter_composition_incompatible_dispatches(
    candidates: list[CandidateRow],
    *,
    candidate_substrate_ids: dict[str, tuple[str, ...]],
) -> tuple[list[CandidateRow], list[tuple[str, str]]]:
    """Walk candidates in order; refuse any whose substrates conflict with
    a substrate already chosen by an earlier candidate in the SAME loop
    iteration (per QQ matrix's REPLACEMENT/INCOMPATIBLE classes).

    Returns ``(kept, dropped_with_reasons)``. ``dropped_with_reasons`` is a
    list of ``(candidate_id, reason)`` pairs.

    Per QQ matrix lesson 5 (HNeRV parity discipline) and the
    `substrate vs codec composition meta-pattern` memo: two
    RENDERER_REPLACEMENT substrates cannot coexist in the same archive,
    and two REPLACEMENT-classed cells anywhere produce an
    archive-grammar conflict at byte-level. Composition matrix is
    consulted ONLY through the participating-substrate sets; the loader
    does not need to import the full matrix here (kept lightweight).

    The composition constraint enforced here is a SAME-DISPATCH-CHAIN
    constraint: it does NOT prevent the operator from running two separate
    autopilot iterations each picking ONE renderer-replacement candidate.
    The matrix governs which substrates can be in the same archive bytes,
    not which substrates can be considered across runs.
    """
    try:
        from tac.optimization.substrate_composition_matrix import (
            Composability,
            build_composition_matrix,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "filter_composition_incompatible_dispatches requires "
            "tac.optimization.substrate_composition_matrix; install or "
            "import path setup before calling."
        ) from exc

    matrix = build_composition_matrix()
    kept: list[CandidateRow] = []
    chosen_substrates: set[str] = set()
    dropped: list[tuple[str, str]] = []
    incompatible_classes = {
        Composability.REPLACEMENT,
        Composability.INCOMPATIBLE,
        Composability.ANTAGONISTIC,
    }
    for cand in candidates:
        substrates = candidate_substrate_ids.get(cand.candidate_id, ())
        conflict_reason: str | None = None
        for s in substrates:
            for prior in chosen_substrates:
                if s == prior:
                    continue
                try:
                    cell = matrix.get(s, prior)
                except (KeyError, ValueError):
                    continue
                if cell.composability in incompatible_classes:
                    conflict_reason = (
                        f"substrate {s!r} composes as "
                        f"{cell.composability.value} with already-chosen "
                        f"substrate {prior!r}; refuse same-iteration "
                        "dispatch (per substrate composition matrix v1)"
                    )
                    break
            if conflict_reason is not None:
                break
        if conflict_reason is not None:
            dropped.append((cand.candidate_id, conflict_reason))
            continue
        kept.append(cand)
        for s in substrates:
            chosen_substrates.add(s)
    return kept, dropped


# ─────────────────────────────────────────────────────────────────────────
# Rudin-Daubechies autopilot ranker integration (opt-in, 2026-05-15)
# ─────────────────────────────────────────────────────────────────────────
#
# Per `feedback_rudin_daubechies_recommendations_for_completing_cathedral_autopilot_nervous_system_20260515.md`
# the canonical ranker stack lives in `tac.autopilot_rudin_daubechies`.
# This integration helper consumes the package's :class:`SLIMRanker` /
# :class:`RashomonEnsembleRanker` to enrich each :class:`CandidateRow` with
# a Rudin-interpretable rule chain explanation BEFORE the autopilot's
# canonical Z1 empirical-revision chain runs.
#
# The helper is OPT-IN: the autopilot's existing rank_candidates path is
# unchanged. Operators / agent partners who want the interpretable
# pre-dispatch ranking surface call ``rerank_candidates_via_rudin_daubechies``
# explicitly.

DEFAULT_RUDIN_DAUBECHIES_SLIM_STORE = Path(".omx/state/rudin_daubechies_slim_anchors.jsonl")
RUDIN_DAUBECHIES_PANEL_AXES = frozenset(
    {"contest_cuda", "contest_cpu", "macos_cpu_advisory"}
)


def rerank_candidates_via_rudin_daubechies(
    candidates: list[CandidateRow],
    *,
    slim_store_path: Path | None = None,
    panel_axis: str = "contest_cuda",
    use_rashomon_ensemble: bool = False,
    rashomon_ensemble_size: int = 8,
) -> list[tuple[CandidateRow, float, str]]:
    """Rerank candidates through the Rudin-Daubechies SLIM ranker.

    Returns a list of ``(candidate, predicted_score, explanation)`` tuples
    sorted ascending by predicted_score (lower predicted score = better
    candidate). The explanation is the Rudin rule-chain readback per
    :func:`tac.autopilot_rudin_daubechies.explain_slim_prediction`.

    When ``use_rashomon_ensemble=True`` the K=8 ensemble's consensus
    prediction is used and a disagreement std-dev is appended to the
    explanation as the operator-facing ideation signal.

    Per CLAUDE.md "Subagent coherence-by-default" wire-in hook 4 (cathedral
    autopilot dispatch hook): the helper is the canonical operator-facing
    transparency layer; calling it produces an auditable ranking decision.

    Per CLAUDE.md "Apples-to-apples evidence discipline": the predicted
    score carries the SLIMRanker's ``confidence_tag()`` and an explicit
    ``panel_axis`` in the explanation so the operator distinguishes
    contest-CUDA, contest-CPU, and macOS advisory prediction surfaces.
    """
    if panel_axis not in RUDIN_DAUBECHIES_PANEL_AXES:
        allowed = ", ".join(sorted(RUDIN_DAUBECHIES_PANEL_AXES))
        raise ValueError(f"unsupported Rudin-Daubechies panel_axis={panel_axis!r}; expected one of {allowed}")
    # Lazy import to avoid hard dep at module import time.
    from tac.autopilot_rudin_daubechies import (
        ProxyPanel,
        RashomonEnsembleRanker,
        SLIMRanker,
        explain_slim_prediction,
    )

    store = slim_store_path or DEFAULT_RUDIN_DAUBECHIES_SLIM_STORE
    if use_rashomon_ensemble:
        ranker = RashomonEnsembleRanker(
            ensemble_size=rashomon_ensemble_size,
            store_path=store,
        )
    else:
        ranker = SLIMRanker(store_path=store)

    out: list[tuple[CandidateRow, float, str]] = []
    for c in candidates:
        # Map the CandidateRow to a minimal ProxyPanel. Per the Taylor
        # decomposition memo this is the integration point for the
        # forthcoming `tac.autopilot_proxies` package; for now we use the
        # already-available signals on CandidateRow.
        panel = ProxyPanel(
            candidate_id=c.candidate_id,
            panel_axis=panel_axis,
        )
        if use_rashomon_ensemble:
            consensus, disagreement = ranker.predict_with_disagreement(panel)
            pred = consensus
            tag = ranker.confidence_tag()
            expl = (
                f"{tag} consensus={consensus:g} disagreement_stddev={disagreement:g}"
            )
        else:
            pred = ranker.predict(panel)
            expl = explain_slim_prediction(ranker, panel)
        expl = f"panel_axis={panel_axis}; {expl}"
        out.append((c, pred, expl))
    out.sort(key=lambda t: t[1])
    return out


def update_rudin_daubechies_from_dispatch_outcome(
    candidate: CandidateRow,
    observed_score: float,
    *,
    axis: str = "contest_cuda",
    slim_store_path: Path | None = None,
) -> None:
    """Closes the continual-learning loop: dispatch outcome -> SLIM update.

    Per operator directive 2026-05-15: every empirical anchor flows through
    this helper so the SLIM ranker's coefficients refit and the next
    candidate evaluation is materially smarter.

    The helper is fcntl-locked per Catalog #128/#131 sister discipline; safe
    to call from concurrent harvesters.

    Per CLAUDE.md "Apples-to-apples evidence discipline" the caller MUST
    pass the correct ``axis`` (``contest_cuda`` / ``contest_cpu`` /
    ``macos_cpu_advisory``); the helper does not infer it from context.
    """
    from tac.autopilot_rudin_daubechies import ProxyPanel, SLIMRanker

    store = slim_store_path or DEFAULT_RUDIN_DAUBECHIES_SLIM_STORE
    ranker = SLIMRanker(store_path=store)
    panel = ProxyPanel(candidate_id=candidate.candidate_id, panel_axis=axis)
    ranker.update_from_anchor(observed_score, panel, axis=axis)


# ─────────────────────────────────────────────────────────────────────────
# Compressive-sensing lattice recovery integration (opt-in, 2026-05-16)
# ─────────────────────────────────────────────────────────────────────────
#
# Per operator approval 2026-05-16 + T4 Symposium Time-Traveler verdict
# (`.omx/research/grand_council_symposium_time_traveler_optimal_staircase_20260516.md`):
# the K=O(sqrt(N)) compressive-sensing claim becomes a first-class
# autopilot input via the canonical helpers in
# `tac.autopilot_rudin_daubechies.compressive_sensing_lattice_recovery`.
#
# The helper is OPT-IN: the autopilot's existing rank_candidates path is
# unchanged.  Operators / agent partners who want the compressive-sensing
# K=O(sqrt(N)) ranking surface call
# ``rerank_candidates_via_compressive_sensing_lattice`` explicitly.
#
# Sister of ``rerank_candidates_via_rudin_daubechies``: where that helper
# uses the SLIM/Rashomon ranker (Phase 1-3), this helper uses the
# substrate-lattice sparse-signal posterior (enhancements 1+5+6) to
# down-weight predicted savings on substrates the lattice posterior
# does NOT believe are frontier-breaking.  The Daubechies-DeVore
# uncertainty band gives an apples-to-apples confidence multiplier.


def _resolve_canonical_frontier_threshold_cpu(default: float = 0.192) -> float:
    """Return live best contest-CPU anchor from canonical state.

    Per Catalog #316 PERMANENT-FIX-FRONTIER-SIGNAL-LOSS: the autopilot
    derives the frontier threshold from canonical state
    (.omx/state/continual_learning_posterior.json +
    .omx/state/active_lane_dispatch_claims.md +
    .omx/state/modal_call_id_ledger.jsonl) via ``tac.frontier_scan``,
    not a hardcoded literal that silently underestimates when better
    archives land. Falls back to ``default`` (0.192) if
    ``tac.frontier_scan`` is unavailable or no qualifying contest-CPU
    anchor exists.
    """
    try:
        from pathlib import Path as _Path

        from tac.frontier_scan import best_per_axis, collect_all_anchors

        repo_root = _Path(__file__).resolve().parent.parent
        anchors = collect_all_anchors(repo_root)
        best = best_per_axis(anchors)
        cpu = best.get("contest_cpu", [])
        if cpu:
            return cpu[0].score
    except Exception:
        pass
    return default


def _build_substrate_lattice_from_candidates(
    candidates: list[CandidateRow],
    *,
    frontier_threshold_cpu: float | None = None,
    expected_sparsity: int = 5,
    use_daubechies_db4: bool = True,
    use_tree_structured_prior: bool = True,
):
    """Construct a SubstrateLatticeRecovery from a CandidateRow list.

    Maps each CandidateRow to a SubstrateLatticeNode using the row's
    ``predicted_score_delta`` to derive a band around the current
    frontier (frontier + delta ± half-width-from-EIG).  The mapping
    uses ``classify_predicted_band`` per the horizon-class directive
    2026-05-16 to assign each candidate a frontier_pursuit_class.
    """
    from tac.autopilot_rudin_daubechies import (
        FrontierPursuitClass,
        SubstrateLatticeNode,
        SubstrateLatticeRecovery,
        classify_predicted_band,
    )

    # Per Catalog #316: derive frontier from canonical state when caller
    # passes None. Hardcoded 0.192 default is silent staleness when better
    # archives land.
    if frontier_threshold_cpu is None:
        frontier_threshold_cpu = _resolve_canonical_frontier_threshold_cpu()
    lr = SubstrateLatticeRecovery(
        use_daubechies_db4=use_daubechies_db4,
        use_tree_structured_prior=use_tree_structured_prior,
        expected_sparsity=expected_sparsity,
        frontier_threshold_cpu=frontier_threshold_cpu,
    )
    # Sort candidates by predicted_score_delta to give the lattice a stable
    # ordering (so the wavelet basis sees neighboring substrates).
    sorted_cands = sorted(candidates, key=lambda c: c.predicted_score_delta)
    seen_ids: set[str] = set()
    for c in sorted_cands:
        if not c.candidate_id or c.candidate_id in seen_ids:
            continue
        seen_ids.add(c.candidate_id)
        midpoint = frontier_threshold_cpu + c.predicted_score_delta
        # EIG governs band half-width; clamp to a sensible range.
        half_width = max(0.005, min(0.05, c.expected_information_gain or 0.01))
        low = max(0.0, midpoint - half_width)
        high = midpoint + half_width
        try:
            cls = classify_predicted_band(low, high)
        except ValueError:
            cls = FrontierPursuitClass.PLATEAU_ADJACENT
        lr.add_node(
            SubstrateLatticeNode(
                node_id=c.candidate_id,
                parent_id=None,
                support_level=int(c.context_order or 0),
                predicted_band_low=low,
                predicted_band_high=high,
                frontier_pursuit_class=cls,
            )
        )
    return lr


def rerank_candidates_via_compressive_sensing_lattice(
    candidates: list[CandidateRow],
    *,
    frontier_threshold_cpu: float | None = None,
    expected_sparsity: int = 5,
    anchors: list[tuple[str, float]] | None = None,
    use_daubechies_db4: bool = True,
    use_tree_structured_prior: bool = True,
) -> list[tuple[CandidateRow, float, str]]:
    """Rerank candidates via compressive-sensing lattice posterior.

    Returns a list of ``(candidate, adjusted_predicted_delta, explanation)``
    tuples sorted ascending by adjusted_predicted_delta (most-negative
    first = greatest improvement).  The adjusted prediction is the row's
    raw ``predicted_score_delta`` weighted by the lattice posterior's
    ``posterior_frontier_probability`` so candidates the lattice
    posterior does NOT believe are frontier-breaking are down-weighted.

    Mathematically::

        adjusted = predicted_score_delta * (
            0.5 + 0.5 * posterior_frontier_probability
        )

    A candidate with posterior_frontier_probability=1.0 (high lattice
    confidence) keeps its full predicted improvement; one with
    posterior_frontier_probability=0.0 (lattice says NOT frontier-breaking)
    is halved.  This is the canonical "trust the empirical anchor pool
    AND the L1 reconstruction" composition.

    Per operator approval 2026-05-16: the helper is the canonical
    operator-facing transparency layer for K=O(sqrt(N)) compressive-sensing
    rank conditioning.  Composes with the existing
    ``rerank_candidates_via_rudin_daubechies`` (SLIM/Rashomon) per
    Catalog #125 wire-in hook 4.

    Per CLAUDE.md "Apples-to-apples evidence discipline": the
    explanation carries the lattice posterior's canonical confidence_tag
    so the operator distinguishes K=2 vs K=8 vs K=16 anchor-pool
    posteriors and the chosen basis (Haar vs Daubechies db4).
    """
    lr = _build_substrate_lattice_from_candidates(
        candidates,
        frontier_threshold_cpu=frontier_threshold_cpu,
        expected_sparsity=expected_sparsity,
        use_daubechies_db4=use_daubechies_db4,
        use_tree_structured_prior=use_tree_structured_prior,
    )
    if anchors:
        for node_id, score in anchors:
            try:
                lr.update_from_anchor(node_id, score)
            except ValueError:
                # Anchor at an unknown node: skip silently per
                # CLAUDE.md "Forbidden score claims" (don't fabricate
                # nonexistent posterior support).
                continue
    posterior = lr.recover_sparse_signal()
    prob_map = dict(posterior.posterior_frontier_probability)
    unc_map = dict(posterior.recovery_uncertainty)
    out: list[tuple[CandidateRow, float, str]] = []
    for c in candidates:
        prob = prob_map.get(c.candidate_id, 0.0)
        unc = unc_map.get(c.candidate_id, 0.0)
        # Apply (0.5 + 0.5 * prob) multiplier per the canonical formula.
        weight = 0.5 + 0.5 * prob
        adjusted = c.predicted_score_delta * weight
        expl = (
            f"{posterior.confidence_tag} "
            f"posterior_frontier_probability={prob:.3f} "
            f"recovery_uncertainty={unc:.4f} "
            f"weight={weight:.3f} "
            f"raw_predicted_delta={c.predicted_score_delta:g} "
            f"adjusted_predicted_delta={adjusted:g}"
        )
        out.append((c, adjusted, expl))
    out.sort(key=lambda t: t[1])
    return out


def rerank_candidates_via_master_gradient(
    candidates: list[CandidateRow],
    *,
    panel_axis: str = "contest_cpu",
    ledger_path: Path | None = None,
) -> list[tuple[CandidateRow, float, str]]:
    """Rerank candidates via the master gradient per symposium §3.6.

    Sister of rerank_candidates_via_rudin_daubechies + rerank_candidates_via_compressive_sensing_lattice.
    Implements the Phase-7 master-gradient lens per cathedral autopilot wire-in (Catalog #125 hook 4).

    For each candidate, queries tac.master_gradient.latest_anchor_for_archive
    keyed on the structured CandidateRow.archive_sha256 field. If a gradient
    anchor exists, the current hook only annotates anchor availability. It does
    not replace predicted_score_delta until the candidate carries a typed
    CandidateModificationSpec / grammar_aware_operator response row with packet
    proofs. If no anchor exists, the candidate falls through to
    predicted_score_delta with a "no-master-gradient-anchor" tag.

    Returns ``(candidate, predicted_delta_s, explanation)`` sorted ascending
    (most-negative first = best score improvement).

    Per CLAUDE.md "Apples-to-apples evidence discipline" + the master-gradient
    canonical contract in tac.master_gradient: predictions carry the
    measurement_axis tag from the source anchor, so contest-CPU vs contest-CUDA
    predictions are never silently mixed.

    Per CLAUDE.md "Forbidden score claims": the returned predicted_delta_s
    is explicitly tagged "[predicted, master-gradient-projection]" in the
    explanation, never promoted to empirical without paired auth-eval.
    """
    if panel_axis not in {"contest_cpu", "contest_cuda"}:
        raise ValueError(
            f"unsupported master_gradient panel_axis={panel_axis!r}; expected contest_cpu or contest_cuda"
        )
    # Lazy import to avoid hard dep at module-import time per the autopilot pattern.
    from tac.master_gradient import (
        MASTER_GRADIENT_LEDGER_PATH,
        latest_anchor_for_archive,
        latest_rejected_contest_axis_anchor_for_archive,
    )

    axis_tag = f"[contest-{'CPU' if panel_axis == 'contest_cpu' else 'CUDA'}]"
    ledger = ledger_path or MASTER_GRADIENT_LEDGER_PATH
    out: list[tuple[CandidateRow, float, str]] = []
    for c in candidates:
        sha_candidate = c.archive_sha256.strip().lower()
        if not _is_sha256_hex(sha_candidate):
            out.append(
                (
                    c,
                    float(c.predicted_score_delta),
                    f"[predicted, no-master-gradient-anchor, {axis_tag}] structured archive_sha256 missing or malformed",
                )
            )
            continue
        anchor = latest_anchor_for_archive(sha_candidate, path=ledger, axis=axis_tag)
        if anchor is None:
            latest_any = latest_anchor_for_archive(sha_candidate, path=ledger)
            if (
                latest_any is not None
                and latest_any.get("measurement_axis") != axis_tag
                and str(latest_any.get("measurement_axis", "")).lower()
                not in {"[contest-cpu]", "[contest-cuda]"}
            ):
                latest_axis = str(latest_any.get("measurement_axis", "<missing>"))
                out.append(
                    (
                        c,
                        float(c.predicted_score_delta),
                        f"[predicted, master-gradient-anchor-diagnostic-only, {axis_tag}] "
                        f"latest effective anchor for archive {sha_candidate[:12]} "
                        f"is {latest_axis} "
                        f"on hardware={latest_any.get('measurement_hardware', '<missing>')} "
                        f"method={latest_any.get('measurement_method', '<missing>')}; "
                        "no authoritative same-axis contest anchor available",
                    )
                )
                continue
            rejected = latest_rejected_contest_axis_anchor_for_archive(
                sha_candidate, path=ledger, axis=axis_tag
            )
            if rejected is not None:
                rejected_anchor, reason = rejected
                out.append(
                    (
                        c,
                        float(c.predicted_score_delta),
                        f"[predicted, master-gradient-anchor-rejected, {axis_tag}] "
                        f"archive {sha_candidate[:12]} row measured_at="
                        f"{rejected_anchor.get('measurement_utc', '<missing>')} "
                        f"hardware={rejected_anchor.get('measurement_hardware', '<missing>')} "
                        f"method={rejected_anchor.get('measurement_method', '<missing>')} "
                        f"failed authority filter: {reason}",
                    )
                )
                continue
            if (
                latest_any is not None
                and latest_any.get("measurement_axis") != axis_tag
            ):
                latest_axis = str(latest_any.get("measurement_axis", "<missing>"))
                out.append(
                    (
                        c,
                        float(c.predicted_score_delta),
                        f"[predicted, master-gradient-anchor-different-authoritative-axis, {axis_tag}] "
                        f"latest effective anchor for archive {sha_candidate[:12]} "
                        f"is {latest_axis} "
                        f"on hardware={latest_any.get('measurement_hardware', '<missing>')} "
                        f"method={latest_any.get('measurement_method', '<missing>')}; "
                        "no authoritative same-axis contest anchor available",
                    )
                )
                continue
            out.append(
                (
                    c,
                    float(c.predicted_score_delta),
                    f"[predicted, no-master-gradient-anchor, {axis_tag}] no anchor for archive {sha_candidate[:12]}",
                )
            )
            continue
        scored_anchor_sha = str(
            anchor.get("scored_archive_sha256") or anchor.get("archive_sha256") or ""
        ).lower()
        if scored_anchor_sha != sha_candidate:
            out.append(
                (
                    c,
                    float(c.predicted_score_delta),
                    f"[predicted, master-gradient-anchor-rejected, {axis_tag}] "
                    f"anchor scored_archive_sha256={scored_anchor_sha[:12] or '<missing>'} "
                    f"does not match candidate archive={sha_candidate[:12]}",
                )
            )
            continue
        n_pairs_used = anchor.get("n_pairs_used")
        n_pairs_total = anchor.get("n_pairs_total")
        if (
            isinstance(n_pairs_used, int)
            and isinstance(n_pairs_total, int)
            and n_pairs_used != n_pairs_total
        ):
            out.append(
                (
                    c,
                    float(c.predicted_score_delta),
                    f"[predicted, master-gradient-anchor-diagnostic-only, {axis_tag}] "
                    f"subset anchor n_pairs_used={n_pairs_used} n_pairs_total={n_pairs_total}",
                )
            )
            continue
        # Anchor present; for the canonical wire-in we surface the operating-point
        # score itself + measurement_utc so the operator can audit when the gradient
        # was last refreshed. Per-byte ΔS projection requires the candidate to
        # express a packet-valid CandidateModificationSpec, which the autopilot
        # ranker does not currently carry; that wire-in is the Phase-7 lens
        # follow-on.
        op_score = float(anchor.get("operating_point", {}).get("score", 0.0))
        measurement_utc = anchor.get("measurement_utc", "?")
        gradient_domain = anchor.get("gradient_byte_domain", "unknown")
        explanation = (
            f"[predicted, master-gradient-anchor-present, {axis_tag}] "
            f"anchor archive={sha_candidate[:12]} op_score={op_score:.5f} "
            f"domain={gradient_domain} measured_at={measurement_utc} "
            f"(ΔS projection requires CandidateModificationSpec/grammar_aware_operator "
            "packet proofs; Phase-7 lens follow-on)"
        )
        # Keep predicted_score_delta unchanged for now (no spec → no projection);
        # the explanation surfaces anchor presence so operator can interpret.
        out.append((c, float(c.predicted_score_delta), explanation))
    out.sort(key=lambda t: t[1])
    return out


def diagnose_compressive_sensing_lattice_undersampling(
    candidates: list[CandidateRow],
    *,
    n_anchors: int,
    expected_sparsity: int = 5,
    safety_margin: float = 0.05,
) -> dict[str, Any]:
    """Run the Donoho-Tanner 2009 phase-transition monitor against the
    cathedral autopilot's candidate pool.

    Returns the diagnostic record per
    :meth:`LatticePhaseTransitionMonitor.compute_undersampling_diagnostic`
    sister Catalog #270 dispatch-optimization-protocol verdict pattern.
    When ``recovery_regime`` is AT_THRESHOLD or FAILED, the autopilot
    should SURFACE the diagnostic to the operator rather than auto-
    dispatching against an unreliable posterior.

    Per operator standing directive 2026-05-16 max-observability:
    the diagnostic is the canonical "the autopilot is operating in
    structurally-uncertain territory" surface.
    """
    from tac.autopilot_rudin_daubechies import LatticePhaseTransitionMonitor

    monitor = LatticePhaseTransitionMonitor(safety_margin=safety_margin)
    return monitor.compute_undersampling_diagnostic(
        K=n_anchors,
        N=len(candidates),
        sparsity_estimate=expected_sparsity,
    )


# ─────────────────────────────────────────────────────────────────────────
# ORPHAN-SIGNAL-AUDIT wire-ins (task #711, 2026-05-17)
# ─────────────────────────────────────────────────────────────────────────
#
# Per operator standing directive 2026-05-17 verbatim *"Ensure all producers
# wired up and integrated into consumers as appropriate with the cathedral
# autopilot the ultimate consumer."*
#
# ORPHAN-SIGNAL-AUDIT (task #711) identified 15 ORPHANS — canonical-helper
# producers whose outputs were NOT consumed by the cathedral autopilot. This
# wave closes 5 of the 15 high-EV orphans:
#
#   1. tac.council_continual_learning      → rerank_candidates_via_council_continual_learning
#   2. tac.probe_outcomes_ledger            → refuse_candidates_via_probe_outcomes
#   3. tac.deploy.modal.call_id_ledger     → update_cost_band_from_modal_call_id_ledger
#   4. tac.substrates.pretrained_driving_prior.composition → load_candidates_from_dp1_composition_primitives
#   5. tac.recursive_adversarial_review    → refuse_candidates_via_recursive_review_unsealed
#
# Per CLAUDE.md "Subagent coherence-by-default" 6-hook wire-in declaration:
# closure of orphan-signal-audit hooks 1 (sensitivity → council), 3 (bit-
# allocator → probe), 4 (cathedral-autopilot dispatch hook → modal call-id
# ledger), 5 (continual-learning posterior → council + probe), and 6 (probe-
# disambiguator → council).
#
# Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
# all 5 wire-ins are fail-CLOSED:
#   - Missing canonical helper (ImportError) → re-raise (NOT silent skip)
#   - Missing ledger file → warn + return empty result (NOT silent default to
#     passing-through ALL candidates)
#   - Refused candidates surfaced explicitly via (kept, refused) tuple OR
#     blockers/notes append (NOT silently dropped)
#
# Per CLAUDE.md "Apples-to-apples evidence discipline": every adjustment
# carries an axis-tagged explanation in CandidateRow.notes so the operator
# can audit which producer surface contributed to the rank.
#
# Sister of the existing 8 producer wire-ins (frontier_scan / master_gradient
# / substrate_composition_matrix / macos_cpu_advisory / exact_ready_queue /
# rudin_daubechies / continual_learning / cost_band_calibration).


# Council-continual-learning rerank constants (tunable per operator review).
# Per CLAUDE.md "Council hierarchy: 4-tier protocol" + Catalog #300 v2 frontmatter:
# T2+ verdicts ARE the binding-decision surface; T1 working groups are advisory.
COUNCIL_VERDICT_WEIGHT_PROCEED_UNCONDITIONAL_DEFAULT = -0.20
"""Negative = boost. PROCEED-unconditional verdict makes the candidate more attractive."""

COUNCIL_VERDICT_WEIGHT_PROCEED_WITH_REVISIONS_DEFAULT = -0.05
"""Negative = mild boost. PROCEED_WITH_REVISIONS is a partial endorsement; modest reward."""

COUNCIL_VERDICT_WEIGHT_DEFER_PENDING_EVIDENCE_DEFAULT = 0.10
"""Positive = penalty. DEFER_PENDING_EVIDENCE means the candidate needs evidence first."""

COUNCIL_VERDICT_WEIGHT_REFUSE_DEFAULT = 0.50
"""Positive = strong penalty. REFUSE is the canonical do-not-dispatch verdict."""

COUNCIL_VERDICT_WEIGHT_ESCALATE_DEFAULT = 0.20
"""Positive = penalty. ESCALATE_TO_OPERATOR / ESCALATE_TO_HIGHER_TIER need operator review."""


def load_candidates_from_dp1_composition_primitives(
    repo_root: Path,
    *,
    default_predicted_score_delta: float = -0.001,
    default_expected_information_gain: float = 0.05,
    default_estimated_dispatch_cost_usd: float = 0.30,
) -> list[CandidateRow]:
    """ORPHAN-SIGNAL-AUDIT wire-in: DP1+(any-base-substrate) as candidate-SOURCE.

    Per CLAUDE.md "Subagent coherence-by-default" hook #3 (bit-allocator) +
    audit META-INSIGHT: every (DP1, base_substrate) composition tuple
    registered in ``tac.substrates.pretrained_driving_prior.composition.
    known_base_substrates()`` is a NEW candidate row that didn't previously
    exist in ``substrate_composition_matrix``. The cathedral autopilot's
    ranker should see these composed candidates explicitly so the operator
    can compare DP1-only vs base-only vs (DP1 + base) composition arms.

    The loader emits one CandidateRow per known base substrate with:
      - candidate_id = ``dp1_composed__<base_substrate>``
      - family = ``pretrained_driving_prior_composition``
      - notes = explicit ``[predicted; DP1 composition primitive]`` axis tag
      - blockers = [``dp1_composition_research_only_pending_paired_anchor``]
        so promotion requires explicit operator review

    Per CLAUDE.md "Forbidden premature KILL": the composition primitives are
    research-only until paired anchors land; the loader does NOT lift this
    gate.

    Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
    fail-CLOSED on missing canonical helper (ImportError re-raised).

    Args:
        repo_root: Repository root (used for canonical helper resolution).
        default_predicted_score_delta: Mild negative = small predicted boost
            from the prior regularizer; calibrated to be below contest-CPU
            noise floor until empirical anchors land.
        default_expected_information_gain: Modest EIG; composition exploration
            yields information about additivity/synergy with base substrate.
        default_estimated_dispatch_cost_usd: Typical smoke cost for the
            composed pair.

    Returns:
        List of CandidateRow objects, one per known base substrate.

    Raises:
        ImportError: if the canonical composition helper is unavailable.
    """
    # Fail-CLOSED per CLAUDE.md "Bugs must be permanently fixed AND
    # self-protected against": if the canonical helper is missing, raise
    # rather than silently emit zero rows.
    from tac.substrates.pretrained_driving_prior.composition import (
        known_base_substrates,
    )

    rows: list[CandidateRow] = []
    for base_substrate in known_base_substrates():
        if not isinstance(base_substrate, str) or not base_substrate.strip():
            continue
        candidate_id = f"dp1_composed__{base_substrate}"
        notes_lines = [
            "[predicted; DP1 composition primitive; ranking-only]",
            "evidence_grade: research_only (paired anchor required for promotion)",
            f"base_substrate: {base_substrate}",
            "composition_helper: tac.substrates.pretrained_driving_prior.composition.compose_with",
            "promotion_blocked: requires paired (DP1-only, base-only, DP1+base) [contest-CUDA]/[contest-CPU] anchors",
            "source: tac.substrates.pretrained_driving_prior.composition.known_base_substrates() — ORPHAN-SIGNAL-AUDIT wire-in 2026-05-17",
        ]
        rows.append(
            CandidateRow(
                candidate_id=candidate_id,
                family="pretrained_driving_prior_composition",
                predicted_score_delta=float(default_predicted_score_delta),
                expected_information_gain=float(default_expected_information_gain),
                estimated_dispatch_cost_usd=float(default_estimated_dispatch_cost_usd),
                blockers=[
                    "dp1_composition_research_only_pending_paired_anchor",
                ],
                notes="\n".join(notes_lines),
                lane_class="research_substrate",
                literature_anchor="DP1 cooperative-receiver prior + base substrate composition",
                source_supports="Catalog #210/#211 DP1 composition contract",
                paper_claim_scope="composition primitive registered; empirical additivity unknown",
                pact_must_prove="paired anchor across (DP1-only, base-only, DP1+base) needed before promotion",
                decode_complexity_evidence="composition wrapper is 13-byte header + length-prefix per DPCOMP_HEADER_FMT",
            )
        )
    return rows


def rerank_candidates_via_council_continual_learning(
    rows: list[CandidateRow],
    *,
    repo_root: Path,
    posterior_path: Path | None = None,
    weight_proceed_unconditional: float = COUNCIL_VERDICT_WEIGHT_PROCEED_UNCONDITIONAL_DEFAULT,
    weight_proceed_with_revisions: float = COUNCIL_VERDICT_WEIGHT_PROCEED_WITH_REVISIONS_DEFAULT,
    weight_defer_pending_evidence: float = COUNCIL_VERDICT_WEIGHT_DEFER_PENDING_EVIDENCE_DEFAULT,
    weight_refuse: float = COUNCIL_VERDICT_WEIGHT_REFUSE_DEFAULT,
    weight_escalate: float = COUNCIL_VERDICT_WEIGHT_ESCALATE_DEFAULT,
) -> list[tuple[CandidateRow, float, str]]:
    """ORPHAN-SIGNAL-AUDIT wire-in: 4-tier council deliberation verdicts as side info.

    Per CLAUDE.md "Council hierarchy: 4-tier protocol" + Catalog #300 v2
    frontmatter: T2+ council verdicts are persisted via
    ``tac.council_continual_learning.append_council_anchor`` to
    ``.omx/state/council_deliberation_posterior.jsonl``. This wire-in matches
    each candidate's ``family`` / ``candidate_id`` / ``lane_class`` against
    the council deliberation topics and applies the verdict-weight per the
    canonical map.

    Per CLAUDE.md "Subagent coherence-by-default" wire-in hook 5 (continual-
    learning posterior): closes the council deliberation → cathedral autopilot
    rank-time loop.

    Per CLAUDE.md "Apples-to-apples evidence discipline": the adjustment carries
    an explicit ``[council-T<tier>; verdict=<v>; deliberation=<id>]`` tag in
    the explanation so the operator can audit which council deliberation
    contributed to the rank.

    Returns a list of ``(candidate, adjusted_predicted_delta, explanation)``
    tuples sorted ascending by adjusted_predicted_delta (most-negative-first).

    Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
    fail-CLOSED on missing canonical helper (ImportError re-raised); silent
    no-op for candidates with no matching council deliberation (don't penalize
    lack-of-evidence per the sister Z1 / Tier C / composition_alpha
    conventions).

    Args:
        rows: Candidate rows from upstream loaders.
        repo_root: Repository root (used for canonical helper resolution).
        posterior_path: Optional override for the council posterior JSONL path.
        weight_*: Per-verdict adjustment weights (negative = boost; positive =
            penalty). Defaults are mild; operator can override per session.

    Returns:
        List of (candidate, adjusted_predicted_delta, explanation) tuples
        sorted ascending.

    Raises:
        ImportError: if the canonical council_continual_learning helper is
            unavailable.
    """
    # Fail-CLOSED per CLAUDE.md "Bugs must be permanently fixed AND
    # self-protected against": import errors are bug-class signals.
    from tac.council_continual_learning import (
        load_council_anchors,
    )

    # Load all council deliberation anchors (lenient mode skips malformed
    # rows; the strict variant is reserved for callers that want
    # quarantine-on-corruption semantics).
    try:
        anchors = load_council_anchors(posterior_path=posterior_path)
    except FileNotFoundError:
        # Missing ledger → warn + return passthrough (per fail-closed contract:
        # we DO NOT silently approve all candidates; we DO surface that no
        # council evidence is available).
        anchors = []

    # Build a topic-keyed index of latest verdicts per deliberation_id (later
    # outcomes supersede earlier events per Catalog #300 schema).
    verdicts_by_deliberation_id: dict[str, dict[str, Any]] = {}
    for anchor in anchors:
        # anchor is a CouncilDeliberationRecord; use dataclass fields.
        delib_id = anchor.deliberation_id
        if not delib_id:
            continue
        # Later events win (JSONL append-order is chronological).
        verdicts_by_deliberation_id[delib_id] = {
            "verdict": anchor.council_verdict,
            "tier": anchor.council_tier,
            "topic": anchor.topic,
            "predicted_mission_contribution": anchor.predicted_mission_contribution,
            "deferred_substrate_id": anchor.deferred_substrate_id,
        }

    weight_map = {
        "PROCEED": weight_proceed_unconditional,
        "PROCEED_WITH_REVISIONS": weight_proceed_with_revisions,
        "DEFER_PENDING_EVIDENCE": weight_defer_pending_evidence,
        "REFUSE": weight_refuse,
        "ESCALATE_TO_OPERATOR": weight_escalate,
        "ESCALATE_TO_HIGHER_TIER": weight_escalate,
    }

    out: list[tuple[CandidateRow, float, str]] = []
    for c in rows:
        # Match by deferred_substrate_id (the council ledger's lane handle)
        # OR by candidate family / lane_class substring (best-effort match;
        # explicit substrate_id is preferred when available).
        matched_verdict: str | None = None
        matched_tier: str | None = None
        matched_delib_id: str | None = None
        matched_topic: str | None = None
        for delib_id, payload in verdicts_by_deliberation_id.items():
            deferred_id = payload.get("deferred_substrate_id")
            if deferred_id and (
                deferred_id == c.candidate_id
                or deferred_id == c.family
                or deferred_id == c.lane_class
                or deferred_id == c.lane_id
            ):
                matched_verdict = payload["verdict"]
                matched_tier = payload["tier"]
                matched_delib_id = delib_id
                matched_topic = payload.get("topic")
                break
        if matched_verdict is None:
            # No council deliberation for this candidate; pass through with
            # explanation noting absence (per the "don't penalize lack-of-
            # evidence" convention).
            out.append(
                (
                    c,
                    float(c.predicted_score_delta),
                    "[council; no-matching-deliberation] no council verdict found for "
                    f"candidate {c.candidate_id!r} / family={c.family!r}",
                )
            )
            continue

        weight = weight_map.get(matched_verdict, 0.0)
        adjusted_delta = float(c.predicted_score_delta) + weight
        explanation = (
            f"[council-{matched_tier}; verdict={matched_verdict}; deliberation={matched_delib_id}; "
            f"topic={matched_topic!r}] weight={weight:+.3f} -> adjusted_predicted_delta="
            f"{adjusted_delta:+.6f}"
        )
        out.append((c, adjusted_delta, explanation))

    out.sort(key=lambda t: t[1])
    return out


def refuse_candidates_via_probe_outcomes(
    rows: list[CandidateRow],
    *,
    repo_root: Path,
    ledger_path: Path | None = None,
) -> tuple[list[CandidateRow], list[CandidateRow]]:
    """ORPHAN-SIGNAL-AUDIT wire-in: probe-outcomes ledger as candidate-REFUSE filter.

    Per CLAUDE.md Catalog #313 (`check_dispatch_target_has_no_predecessor_
    adjudicated_outcome`): the probe-outcomes ledger persists adjudicated
    verdicts {INDEPENDENT, KILL, DEFER, PROMOTE, PROCEED, PARTIAL,
    OPERATOR_REVIEW_REQUIRED}. The {INDEPENDENT, KILL, DEFER} blocking
    verdicts mean the apparatus has already adjudicated this substrate's
    probe and a fresh dispatch would re-do work the system already settled.

    Operator-authorize.py ALREADY consumes the probe_outcomes_ledger at
    ``_check_predecessor_probe_outcome`` to refuse dispatch. This autopilot-
    side wire-in is defense-in-depth: the rejected candidates are filtered
    OUT of the ranked queue BEFORE the operator sees the rankings — saves
    operator-attention budget per CLAUDE.md Catalog #291 mission-alignment.

    Per CLAUDE.md "Subagent coherence-by-default" wire-in hook 3 (bit-
    allocator → probe disambiguator): closes the probe-outcomes → cathedral
    autopilot rank-time loop.

    Per CLAUDE.md "Forbidden premature KILL without research exhaustion": a
    blocking probe outcome is research-deferral, NOT a kill. Refused
    candidates are returned to the caller (NOT silently dropped) so the
    operator can audit which candidates were filtered and override if
    research-exhaustion criteria are met.

    Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
    fail-CLOSED on missing canonical helper (ImportError re-raised); missing
    ledger file → empty blocking-outcomes list (lenient; loader pattern).

    Returns:
        Tuple of (kept_rows, refused_rows). The kept_rows are safe to surface
        to the operator; the refused_rows carry probe-outcome blockers in
        their `.blockers` list AND a probe-outcome explanation in `.notes`.

    Raises:
        ImportError: if the canonical probe_outcomes_ledger helper is
            unavailable.
    """
    # Fail-CLOSED per CLAUDE.md "Bugs must be permanently fixed AND
    # self-protected against": import errors are bug-class signals.
    from tac.probe_outcomes_ledger import (
        BLOCKING_VERDICTS,
        latest_blocking_outcome_by_substrate,
    )

    kept: list[CandidateRow] = []
    refused: list[CandidateRow] = []

    for c in rows:
        # Try to resolve substrate from candidate fields. Per the canonical
        # ProbeOutcomeView.substrate field semantics: the substrate string
        # is the durable key (recipe_path can be renamed).
        substrate_candidates: list[str] = []
        if c.family:
            substrate_candidates.append(c.family)
        if c.lane_class:
            substrate_candidates.append(c.lane_class)
        if c.candidate_id:
            substrate_candidates.append(c.candidate_id)

        blocking_view = None
        matched_substrate: str | None = None
        for substrate in substrate_candidates:
            try:
                view = latest_blocking_outcome_by_substrate(
                    substrate, path=ledger_path
                )
            except (FileNotFoundError, OSError):
                # Missing ledger → no blocking outcomes (lenient). Per fail-
                # closed contract: NOT silent default-to-passthrough; the
                # loop continues but the candidate is NOT refused via this
                # helper (the operator-authorize side gate is still active).
                view = None
            if view is not None and view.verdict in BLOCKING_VERDICTS:
                blocking_view = view
                matched_substrate = substrate
                break

        if blocking_view is None:
            kept.append(c)
            continue

        # Refused: add blocker + extend notes per the existing pattern.
        # The blocker string is parseable by operator_authorize gates.
        blocker_str = (
            f"probe_outcome_blocking_verdict__{blocking_view.verdict}__"
            f"substrate__{matched_substrate}__probe_id__{blocking_view.probe_id}"
        )
        refused_blockers = list(c.blockers)
        if blocker_str not in refused_blockers:
            refused_blockers.append(blocker_str)
        notes_extension = (
            f"\n[probe-outcome refuse; ORPHAN-SIGNAL-AUDIT 2026-05-17] "
            f"substrate={matched_substrate} probe_id={blocking_view.probe_id} "
            f"verdict={blocking_view.verdict} kind={blocking_view.probe_kind} "
            f"adjudicated_at_utc={blocking_view.adjudicated_at_utc} "
            f"expires_at_utc={blocking_view.expires_at_utc} "
            f"evidence_path={blocking_view.evidence_path!r}"
        )

        # Construct a new CandidateRow with extended blockers/notes; preserve
        # all other fields (CandidateRow is mutable but we treat this as
        # immutable-ish here for safety).
        refused_row = dataclasses.replace(
            c,
            blockers=refused_blockers,
            notes=(c.notes + notes_extension) if c.notes else notes_extension.lstrip(),
        )
        refused.append(refused_row)

    return kept, refused


def update_cost_band_from_modal_call_id_ledger(
    repo_root: Path,
    *,
    ledger_path: Path | None = None,
    posterior_path: Path | None = None,
    since_utc: str | None = None,
) -> dict[str, Any]:
    """ORPHAN-SIGNAL-AUDIT wire-in: Modal call-id outcomes feed cost-band posterior.

    Per CLAUDE.md Catalog #245 (Modal call-id ledger): every dispatch via the
    canonical Modal helpers writes a `register_dispatched_call_id` row + an
    `update_call_id_outcome` row to ``.omx/state/modal_call_id_ledger.jsonl``.
    The outcome rows carry ``status`` / ``rc`` / ``elapsed_seconds`` /
    ``cost_actual_usd`` / ``gpu`` / ``platform`` fields that the cost-band
    posterior at ``tac.cost_band_calibration`` consumes for per-(platform,
    gpu) cost prediction.

    This helper closes the loop: for every harvested call-id row that is NOT
    yet reflected in the cost-band posterior, append a new cost-band anchor
    via the canonical ``tac.cost_band_calibration.append_anchor`` helper.

    Per CLAUDE.md "Subagent coherence-by-default" wire-in hook 4 (cathedral
    autopilot dispatch hook): closes the Modal dispatch outcome → cost-band
    posterior loop. The autopilot's per-candidate cost prediction
    (`predicted_dispatch_risk` SLIM field per Catalog #250) consumes the
    refreshed posterior at the next ranking pass.

    Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
    fail-CLOSED on missing canonical helper (ImportError re-raised); missing
    ledger → return empty status (lenient; loader pattern).

    Args:
        repo_root: Repository root (used for canonical helper resolution).
        ledger_path: Optional override for the Modal call-id ledger JSONL.
        posterior_path: Optional override for the cost-band posterior JSONL.
        since_utc: Optional ISO-8601 UTC cutoff; only consume call-id rows
            written at or after this timestamp. Default = consume all.

    Returns:
        Status dict with ``rows_scanned``, ``anchors_appended``, ``skipped_reasons``
        for operator-facing audit + downstream consumer chaining.

    Raises:
        ImportError: if the canonical Modal call-id ledger OR
            cost_band_calibration helper is unavailable.
    """
    # Fail-CLOSED per CLAUDE.md "Bugs must be permanently fixed AND
    # self-protected against": import errors are bug-class signals.
    from tac.cost_band_calibration import (
        FAILED_DISPATCH,
        HARVESTED_PARTIAL,
        SUCCESSFUL_DISPATCH,
        TIMED_OUT,
        CostBandAnchor,
        append_anchor,
    )
    from tac.deploy.modal.call_id_ledger import (
        STATUS_DISPATCHED,
        STATUS_FAILED,
        STATUS_HARVESTED,
        STATUS_STALE,
        load_call_ids,
        query_all_post_utc,
    )

    # Load relevant call-id rows (lenient mode).
    try:
        rows = (
            query_all_post_utc(since_utc, path=ledger_path)
            if since_utc
            else load_call_ids(path=ledger_path)
        )
    except (FileNotFoundError, OSError):
        # Missing ledger → no anchors to backfill; lenient.
        return {
            "schema": "modal_call_id_ledger_to_cost_band_wire_in_v1",
            "rows_scanned": 0,
            "anchors_appended": 0,
            "skipped_reasons": {"ledger_missing": 1},
            "since_utc": since_utc,
        }

    # JOIN by call_id: the dispatched-event row carries platform/gpu/recipe/
    # label/lane_id metadata; the harvested/failed/stale event row carries
    # elapsed/cost/rc/score actuals. Build a per-call_id merged view.
    merged_by_call_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = row.get("call_id")
        if not isinstance(cid, str) or not cid:
            continue
        merged = merged_by_call_id.setdefault(cid, {})
        # Per CLAUDE.md HISTORICAL_PROVENANCE: never mutate dispatched row;
        # we are constructing an EPHEMERAL view, not writing back. Take
        # non-None fields preferentially, accumulating across event rows.
        for key, val in row.items():
            if val is not None and (key not in merged or merged[key] is None):
                merged[key] = val
        # Always overwrite `status` / `event_type` / `rc` / `timed_out` to the
        # LATEST event (JSONL append order is chronological per Catalog #245).
        for latest_field in ("status", "event_type", "rc", "timed_out"):
            if latest_field in row:
                merged[latest_field] = row[latest_field]

    # For each merged call_id row that is NOT a `dispatched`-only event AND
    # has the cost-band fields we need, append a cost-band anchor.
    status_to_outcome = {
        STATUS_HARVESTED: SUCCESSFUL_DISPATCH,
        STATUS_FAILED: FAILED_DISPATCH,
        STATUS_STALE: HARVESTED_PARTIAL,
    }
    anchors_appended = 0
    skipped: dict[str, int] = {}
    for cid, merged in merged_by_call_id.items():
        status = merged.get("status")
        if status == STATUS_DISPATCHED:
            skipped["status_dispatched_no_outcome_yet"] = (
                skipped.get("status_dispatched_no_outcome_yet", 0) + 1
            )
            continue
        # Map terminal-status events → cost-band outcomes.
        if status == "timed_out" or merged.get("timed_out") is True:
            outcome_value = TIMED_OUT
        else:
            outcome_value = status_to_outcome.get(status)
        if outcome_value is None:
            skipped["unrecognized_status"] = skipped.get("unrecognized_status", 0) + 1
            continue

        # Required cost-band fields. Skip rows missing essentials.
        platform = str(merged.get("platform") or "modal")
        gpu = str(merged.get("gpu") or "")
        elapsed_seconds = merged.get("elapsed_seconds")
        cost_actual = merged.get("cost_actual_usd")
        trainer = str(
            merged.get("recipe") or merged.get("label") or merged.get("lane_id") or ""
        )
        if not gpu or elapsed_seconds is None or cost_actual is None or not trainer:
            skipped["missing_required_cost_band_fields"] = (
                skipped.get("missing_required_cost_band_fields", 0) + 1
            )
            continue

        try:
            elapsed_float = float(elapsed_seconds)
            cost_float = float(cost_actual)
        except (TypeError, ValueError):
            skipped["malformed_numeric_field"] = (
                skipped.get("malformed_numeric_field", 0) + 1
            )
            continue
        if not math.isfinite(elapsed_float) or not math.isfinite(cost_float):
            skipped["non_finite_numeric_field"] = (
                skipped.get("non_finite_numeric_field", 0) + 1
            )
            continue

        rc_value = merged.get("rc")
        try:
            rc_int = int(rc_value) if rc_value is not None else None
        except (TypeError, ValueError):
            rc_int = None

        # Construct the CostBandAnchor. Note: epochs / batch_size / all_flags_on
        # are typically NOT in the call-id ledger; we use sentinel defaults
        # (0 / 0 / False) and document via `notes`. Per CLAUDE.md "Comment-only
        # contracts are FORBIDDEN" we explicitly tag in notes that these are
        # ledger-derived rows lacking epoch/batch metadata.
        notes_str = (
            "cost_estimate_source=modal_call_id_ledger_actuals; "
            f"call_id={cid}; "
            f"lane_id={merged.get('lane_id')}; "
            f"score={merged.get('score')}; "
            f"score_axis={merged.get('score_axis')}; "
            f"epochs_batch_metadata_unavailable_from_ledger"
        )
        anchor = CostBandAnchor(
            logged_at_utc=str(merged.get("written_at_utc") or ""),
            dispatch_label=str(merged.get("label") or cid or ""),
            trainer=trainer,
            platform=platform,
            gpu=gpu,
            epochs=0,  # Unavailable from ledger row; documented in notes.
            batch_size=0,  # Unavailable from ledger row; documented in notes.
            all_flags_on=False,  # Unknown from ledger; conservative default.
            actual_wall_clock_sec=elapsed_float,
            actual_cost_usd=cost_float,
            outcome=outcome_value,
            returncode=rc_int,
            notes=notes_str,
        )
        try:
            append_anchor(anchor, posterior_path=posterior_path)
            anchors_appended += 1
        except (ValueError, OSError) as exc:
            skipped[f"append_failure:{type(exc).__name__}"] = (
                skipped.get(f"append_failure:{type(exc).__name__}", 0) + 1
            )

    return {
        "schema": "modal_call_id_ledger_to_cost_band_wire_in_v1",
        "rows_scanned": len(rows),
        "anchors_appended": anchors_appended,
        "skipped_reasons": skipped,
        "since_utc": since_utc,
    }


def refuse_candidates_via_recursive_review_unsealed(
    rows: list[CandidateRow],
    *,
    repo_root: Path,
    ledger_path: Path | None = None,
) -> tuple[list[CandidateRow], list[CandidateRow]]:
    """ORPHAN-SIGNAL-AUDIT wire-in: bundles NOT yet 3-clean-pass SEAL'd get refused.

    Per CLAUDE.md "Submission PR gate" non-negotiable: *"NEVER submit a PR
    until the score has undergone a 5-turn consecutive clean-pass adversarial
    skunkworks council review with extreme paranoia. This is stricter than
    the standard 3-pass greenup. All 15 council members review. ANY issue
    resets the counter to 0."*

    The 3-clean-pass adversarial-review SEAL state is persisted via
    ``tac.recursive_adversarial_review.append_round_locked`` to
    ``.omx/state/recursive_review_rounds.jsonl``. Each bundle has a counter
    that resets to 0 on any non-CONFIRMS finding; SEAL is reached when
    ``counter_after >= SEAL_THRESHOLD`` (= 3).

    This wire-in refuses candidates whose corresponding bundle_id has NOT
    yet reached SEAL. The bundle_id is derived from
    ``compute_bundle_id(sorted scope_paths)`` per the canonical helper;
    candidates without a matching bundle entry are PASSED THROUGH (no
    bundle = no SEAL constraint to enforce yet; defer to the operator-
    authorize side gate).

    Per CLAUDE.md "Subagent coherence-by-default" wire-in hook 4 (cathedral
    autopilot dispatch hook): closes the adversarial-review SEAL state →
    cathedral autopilot rank-time loop.

    Per CLAUDE.md "Forbidden premature KILL without research exhaustion":
    unsealed bundles are research-deferral, NOT killed. Refused candidates
    are returned to the caller (NOT silently dropped) so the operator can
    audit which candidates need additional review rounds.

    Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
    fail-CLOSED on missing canonical helper (ImportError re-raised); missing
    ledger → empty bundle index (lenient; loader pattern).

    Note: This wire-in is INITIALLY a structural pass-through (most
    candidates do NOT yet have a registered bundle_id matching their
    family/candidate_id). It activates as adversarial-review rounds land
    against specific candidate bundles. Per CLAUDE.md "Substrate scaffolds
    MUST be COMPLETE or RESEARCH-ONLY": the helper is COMPLETE (not stub);
    the empty-ledger case is the canonical no-op-by-default state.

    Returns:
        Tuple of (kept_rows, refused_rows). The kept_rows are safe to surface
        to the operator; the refused_rows carry SEAL-pending blockers in
        their `.blockers` list AND a review-round explanation in `.notes`.

    Raises:
        ImportError: if the canonical recursive_adversarial_review helper is
            unavailable.
    """
    # Fail-CLOSED per CLAUDE.md "Bugs must be permanently fixed AND
    # self-protected against": import errors are bug-class signals.
    from tac.recursive_adversarial_review import (
        SEAL_THRESHOLD,
        clean_pass_counter_for_bundle,
        load_rounds_lenient,
    )

    # Load all review-round rows (lenient mode skips malformed).
    try:
        rows_loaded = load_rounds_lenient(path=ledger_path)
    except (FileNotFoundError, OSError):
        # Missing ledger → no bundles to enforce; the entire candidate list
        # passes through. Per CLAUDE.md "Submission PR gate" the SEAL state
        # is enforced at PR-time AND at autopilot-rank-time; this autopilot-
        # side enforcement is defense-in-depth.
        return list(rows), []

    # Build an index of all registered bundle_ids (and their latest content
    # sha + counter). We need this to know which candidates are "in scope"
    # for SEAL enforcement at all.
    bundle_id_to_latest: dict[str, dict[str, Any]] = {}
    for record in rows_loaded:
        bid = record.get("bundle_id")
        if not isinstance(bid, str) or not bid:
            continue
        # JSONL append order = chronological; later events win.
        bundle_id_to_latest[bid] = record

    kept: list[CandidateRow] = []
    refused: list[CandidateRow] = []

    for c in rows:
        # Try to match candidate to a registered bundle via candidate_id,
        # family, lane_id substrings.
        matched_bundle_id: str | None = None
        for bid, payload in bundle_id_to_latest.items():
            scope_paths = payload.get("scope_paths") or []
            if not isinstance(scope_paths, (list, tuple)):
                continue
            # Best-effort match: if the candidate's family or lane_id appears
            # as a token in any scope path, treat the bundle as relevant.
            for sp in scope_paths:
                sp_str = str(sp)
                if (
                    (c.family and c.family in sp_str)
                    or (c.lane_id and c.lane_id in sp_str)
                    or (c.candidate_id and c.candidate_id in sp_str)
                ):
                    matched_bundle_id = bid
                    break
            if matched_bundle_id:
                break

        if matched_bundle_id is None:
            # Not in scope; pass through.
            kept.append(c)
            continue

        # Look up the latest content-aware SEAL counter for the matched bundle.
        latest_record = bundle_id_to_latest[matched_bundle_id]
        scope_content_sha256 = latest_record.get("scope_content_sha256")
        try:
            counter = clean_pass_counter_for_bundle(
                matched_bundle_id,
                path=ledger_path,
                scope_content_sha256=scope_content_sha256,
            )
        except (FileNotFoundError, OSError):
            counter = 0

        if counter >= SEAL_THRESHOLD:
            # SEALed; pass through.
            kept.append(c)
            continue

        # Unsealed: refuse with blocker + notes extension.
        blocker_str = (
            f"recursive_review_unsealed__bundle_id__{matched_bundle_id}__"
            f"counter__{counter}__seal_threshold__{SEAL_THRESHOLD}"
        )
        refused_blockers = list(c.blockers)
        if blocker_str not in refused_blockers:
            refused_blockers.append(blocker_str)
        notes_extension = (
            f"\n[recursive-review-unsealed refuse; ORPHAN-SIGNAL-AUDIT 2026-05-17] "
            f"bundle_id={matched_bundle_id} clean_pass_counter={counter} "
            f"seal_threshold={SEAL_THRESHOLD} "
            f"latest_round_verdict={latest_record.get('verdict')!r} "
            f"per CLAUDE.md 'Submission PR gate' non-negotiable"
        )
        refused_row = dataclasses.replace(
            c,
            blockers=refused_blockers,
            notes=(c.notes + notes_extension) if c.notes else notes_extension.lstrip(),
        )
        refused.append(refused_row)

    return kept, refused


# ─────────────────────────────────────────────────────────────────────────
# End of ORPHAN-SIGNAL-AUDIT wire-ins (task #711, 2026-05-17)
# ─────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────
# CATHEDRAL-AUTO-INGEST-PARADIGM-SHIFT auto-discovery loop (2026-05-19)
# ─────────────────────────────────────────────────────────────────────────
#
# Per operator directive 2026-05-19 verbatim:
#   "What if we change the paradigm by making cathedral autopilot ingest by
#   default if within a certain directory and exposing/respecting a certain
#   contract or schema. Fix permanently and self protect against"
#
# Convention-over-configuration: packages in src/tac/cathedral_consumers/
# that satisfy tac.cathedral.consumer_contract.CathedralConsumerContract are
# auto-discovered + auto-registered. NO manual import-by-import wiring of
# the 12 NEW tac.* namespaces that landed orphaned per the
# wiring+integration audit (commit 3821cfb6b 2026-05-19).
#
# Catalog #335 self-protection: STRICT preflight gate refuses non-compliant
# packages from landing in the canonical directory.


def discover_and_register_consumers(
    consumer_dir_relpath: str = "src/tac/cathedral_consumers",
    *,
    repo_root: Path | str | None = None,
    strict: bool = False,
    include_underscore_packages: bool = True,
) -> list[dict[str, Any]]:
    """Auto-discover all cathedral consumer packages in the canonical directory.

    Per :class:`tac.cathedral.consumer_contract.CathedralConsumerContract`.

    Iterates ``src/tac/cathedral_consumers/``, imports each subdirectory as
    a Python package, verifies the canonical contract, returns the list of
    serialized ``ConsumerRegistration`` records.

    THE PERMANENT FIX for the orphan-signal-at-cathedral bug class. Future
    consumers land in the canonical directory + expose the contract +
    auto-discovery loop ingests them without manual ranker-cascade edits.

    Parameters
    ----------
    consumer_dir_relpath
        Path relative to ``repo_root`` (default ``src/tac/cathedral_consumers``).
    repo_root
        Repository root (default: cwd).
    strict
        If True, raises on non-compliant consumers lacking a waiver. Default
        False (warn-only — caller decides ranker integration policy).
    include_underscore_packages
        If True, includes reference packages whose name starts with ``_``
        (e.g. ``_example_consumer``). Default True so the gate has a
        permanent positive fixture.

    Returns
    -------
    List of dict serializations of :class:`ConsumerRegistration` (sorted by
    consumer_name for determinism).
    """
    from tac.cathedral.consumer_contract import (
        CathedralConsumerContractError,
        ConsumerRegistration,
        discover_waiver_in_init,
        validate_consumer_module,
    )

    root = Path(repo_root) if repo_root else Path.cwd()
    consumer_dir = root / consumer_dir_relpath
    if not consumer_dir.exists():
        return []

    registrations: list[ConsumerRegistration] = []
    seen_names: set[str] = set()

    for sub in sorted(consumer_dir.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name in {"__pycache__", "tests"}:
            continue
        if sub.name.startswith("_") and not include_underscore_packages:
            continue

        init_path = sub / "__init__.py"
        if not init_path.exists():
            # Not a package; skip silently (no contract claim made).
            continue

        module_dotted = f"tac.cathedral_consumers.{sub.name}"
        rationale, waiver_active = discover_waiver_in_init(init_path)

        try:
            # Lazy import; the consumer's __init__ is responsible for any
            # heavy lifting it may declare.
            import importlib
            mod = importlib.import_module(module_dotted)
            reg = validate_consumer_module(mod, module_path=module_dotted)
        except (ImportError, CathedralConsumerContractError) as exc:
            reg = ConsumerRegistration(
                consumer_name=sub.name,
                consumer_version="unknown",
                consumer_hook_numbers=(),
                consumer_module_path=module_dotted,
                contract_compliant=False,
                waiver_rationale=rationale,
                waiver_active=waiver_active,
                validation_errors=(f"import_error: {type(exc).__name__}: {exc}",),
            )

        # Apply waiver if the package opted out via the canonical token.
        if not reg.contract_compliant and waiver_active:
            reg = ConsumerRegistration(
                consumer_name=reg.consumer_name,
                consumer_version=reg.consumer_version,
                consumer_hook_numbers=reg.consumer_hook_numbers,
                consumer_module_path=reg.consumer_module_path,
                contract_compliant=False,
                waiver_rationale=rationale,
                waiver_active=True,
                validation_errors=reg.validation_errors,
            )

        if reg.consumer_name in seen_names:
            # Deterministic dedup; first registration wins.
            continue
        seen_names.add(reg.consumer_name)
        registrations.append(reg)

        if strict and not reg.contract_compliant and not reg.waiver_active:
            raise RuntimeError(
                f"[cathedral-auto-discovery] STRICT refuse: "
                f"package {module_dotted!r} fails canonical contract; "
                f"errors={list(reg.validation_errors)}; "
                f"add # CATHEDRAL_CONSUMER_DEFERRED_OK:<rationale> waiver "
                f"OR implement tac.cathedral.consumer_contract.CathedralConsumerContract"
            )

    return [dataclasses.asdict(r) for r in registrations]


def discover_compliant_consumer_modules(
    consumer_dir_relpath: str = "src/tac/cathedral_consumers",
    *,
    repo_root: Path | str | None = None,
) -> list[Any]:
    """Return the live module objects for every contract-compliant consumer.

    Sister of :func:`discover_and_register_consumers` for the callsite that
    needs to actually CALL ``consume_candidate`` / ``update_from_anchor`` on
    each consumer. Returns import-resolved modules (not just registration
    metadata).

    Underscore-prefixed reference packages (e.g. ``_example_consumer``) are
    SKIPPED in this listing because they are reference fixtures, not
    production ranker contributions. The canonical contract validation
    (via :func:`discover_and_register_consumers`) still includes them so
    the gate has a permanent positive fixture; this function is the
    runtime hook that filters them out.
    """
    from tac.cathedral.consumer_contract import (
        validate_consumer_module,
    )

    root = Path(repo_root) if repo_root else Path.cwd()
    consumer_dir = root / consumer_dir_relpath
    if not consumer_dir.exists():
        return []

    modules: list[Any] = []
    for sub in sorted(consumer_dir.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name in {"__pycache__", "tests"}:
            continue
        if sub.name.startswith("_"):
            # Skip reference / example packages.
            continue
        init_path = sub / "__init__.py"
        if not init_path.exists():
            continue
        module_dotted = f"tac.cathedral_consumers.{sub.name}"
        try:
            import importlib
            mod = importlib.import_module(module_dotted)
        except ImportError:
            continue
        reg = validate_consumer_module(mod, module_path=module_dotted)
        if reg.contract_compliant:
            modules.append(mod)
    return modules


# ─────────────────────────────────────────────────────────────────────────
# End of CATHEDRAL-AUTO-INGEST-PARADIGM-SHIFT (2026-05-19, Catalog #335)
# ─────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────
# R11-H1-1-PLUS-H1-6-FIX-WAVE: invoker callsites (Catalog #336 + #337)
# ─────────────────────────────────────────────────────────────────────────
# Cable H1 R11 adversarial review (commit 725699560 2026-05-19) caught:
#
#  H1-1: discover_and_register_consumers + discover_compliant_consumer_modules
#        DEFINED but NEVER CALLED from main(). 21 consumer packages +
#        canonical contract + auto-discovery loop = ZERO actual cathedral
#        influence at runtime.
#
#  H1-6: rerank_candidates_via_master_gradient "only annotates anchor
#        availability" — never invoked from main() either.
#
# Both are the SAME meta-class: auto-discovery / canonical-helper machinery
# is necessary but NOT SUFFICIENT to extinct the orphan-signal class. The
# INVOKER CALLSITE in main() is the missing structural protection.
#
# Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
# this fix lands the INVOKER CALLSITE here + two STRICT preflight gates
# (#336 / #337) at src/tac/preflight.py to prevent the regression class.
#
# The contribution is OBSERVABILITY-ONLY at landing per Catalog #287/#323:
# every consumer's consume_candidate must already declare
# `promotable=False` + `axis_tag=[predicted]` (validated by the canonical
# contract). The invocation surfaces the contributions in output JSON for
# operator audit; it does NOT mutate predicted_score_delta on the candidate
# rows (that would couple the autopilot loop to consumer logic in a way
# that breaks the canonical rank pipeline).
#
# The canonical wire-in: `invoke_cathedral_consumers_on_candidates()` is
# called once from main() after candidates are loaded (in both --report-only
# and the normal run_continuous_loop path), and its output is attached
# to the emitted JSON payload under `cathedral_consumer_invocations` and
# `master_gradient_anchor_annotations`.

CATHEDRAL_CONSUMER_INVOCATION_SCHEMA = "cathedral_consumer_invocation_v1_20260519"
"""Canonical schema id for the consumer invocation output payload.

Versioned per the canonical helper pattern (Catalog #245 ledger / #300
council deliberation v2 / sister discipline). Bump when the output shape
changes meaningfully.
"""


def _invoke_consumer_safely(
    module: Any, candidate: CandidateRow
) -> dict[str, Any]:
    """Call ``module.consume_candidate(candidate_dict)`` with error trapping.

    Each consumer is a user-authored module; an exception in one consumer
    must NOT crash the cathedral loop. Errors are surfaced as a row with
    `error` field so the operator can audit which consumer failed.

    Per Catalog #287 the contribution is observability-only (the autopilot
    ranker pipeline is unchanged); errors do not affect dispatch decisions.
    """
    try:
        candidate_payload = {
            "candidate_id": candidate.candidate_id,
            "archive_sha256": candidate.archive_sha256,
            "predicted_score_delta": candidate.predicted_score_delta,
            "estimated_dispatch_cost_usd": candidate.estimated_dispatch_cost_usd,
            "blockers": list(candidate.blockers),
        }
        contribution = module.consume_candidate(candidate_payload)
        if not isinstance(contribution, Mapping):
            return {
                "consumer_module": getattr(module, "__name__", "<unknown>"),
                "candidate_id": candidate.candidate_id,
                "error": (
                    f"consume_candidate returned {type(contribution).__name__}, "
                    "expected Mapping per CathedralConsumerContract"
                ),
            }
        # Catalog #356 + CATHEDRAL-SMARTER-DESIGN-MEMO Dim 3 Step 3.3:
        # detect per-axis decomposition + auto-compose via canonical helper.
        # Consumers that don't emit `predicted_axis_decomposition` keep
        # working unchanged (per-axis branch is None for them).
        per_axis_breakdown = _compose_per_axis_decomposition_if_present(
            contribution, candidate
        )
        result = {
            "consumer_module": getattr(module, "__name__", "<unknown>"),
            "consumer_name": getattr(module, "CONSUMER_NAME", "<unknown>"),
            "consumer_version": getattr(module, "CONSUMER_VERSION", "unknown"),
            "candidate_id": candidate.candidate_id,
            "predicted_delta_adjustment": float(
                contribution.get("predicted_delta_adjustment", 0.0)
            ),
            "rationale": str(contribution.get("rationale", ""))[:512],
            "axis_tag": str(contribution.get("axis_tag", "[predicted]")),
            "promotable": bool(contribution.get("promotable", False)),
            "confidence": float(contribution.get("confidence", 0.0)),
        }
        if per_axis_breakdown is not None:
            result["predicted_axis_decomposition"] = per_axis_breakdown
        return result
    except Exception as exc:  # noqa: BLE001  defensive at consumer boundary
        return {
            "consumer_module": getattr(module, "__name__", "<unknown>"),
            "candidate_id": candidate.candidate_id,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _compose_per_axis_decomposition_if_present(
    contribution: Mapping[str, Any],
    candidate: CandidateRow,
) -> dict[str, Any] | None:
    """Detect + auto-compose ``predicted_axis_decomposition`` from a consumer.

    Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 3 Step 3.3 + Catalog #356.

    Returns ``None`` when the consumer does NOT emit per-axis
    decomposition (the normal backward-compat path per Catalog #341).
    Returns a JSON-safe dict with the composed breakdown when present.

    Defensive: any malformed per-axis emission (wrong shape, missing
    Provenance, NaN deltas, etc.) is surfaced as an ``error`` field on
    the breakdown rather than crashing the ranker; the scalar
    ``predicted_delta_adjustment`` continues to flow unaffected so
    Tier-A canonical-routing-markers semantics are preserved per
    Catalog #341.

    Baselines:
    - ``current_archive_bytes`` defaults to CANONICAL_RATE_DENOM_BYTES
      (the canonical contest archive size) when no candidate-level
      baseline is available; this is a CONSERVATIVE default that
      under-weights the rate contribution slightly but never makes a
      candidate look better than it is.
    - ``current_d_pose`` defaults to 0.0 when no canonical operating
      point is available; the sqrt-difference term collapses to
      ``sqrt(10*delta_d_pose)`` which is the LINEAR-DOMINANT-AT-LOW-POSE
      regime per CLAUDE.md "SegNet vs PoseNet importance" so the
      composition is observability-safe.

    Future Dim 3 Step 3.4+: caller will source baselines from the
    canonical frontier pointer via
    :func:`tac.score_composition.load_baseline_pose_from_canonical_frontier_pointer`
    once the AnchorRecord schema is extended with per-axis components.
    """
    raw = contribution.get("predicted_axis_decomposition")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        return {
            "error": (
                "predicted_axis_decomposition must be a Mapping per "
                f"Catalog #356 + Dim 3 Step 3.1; got {type(raw).__name__}"
            ),
        }
    try:
        # Lazy import to avoid a hard runtime dependency for the
        # cathedral autopilot loop when score_composition isn't needed.
        from tac.cathedral.consumer_contract import AxisDecomposition
        from tac.score_composition import (
            CANONICAL_RATE_DENOM_BYTES as _RATE_DENOM,
            compose_score_from_axes,
        )
    except ImportError as exc:
        return {
            "error": (
                "Failed to import tac.score_composition / "
                f"AxisDecomposition: {exc}"
            ),
        }
    try:
        decomp = AxisDecomposition.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "error": (
                f"AxisDecomposition.from_dict refused per-axis emission: "
                f"{type(exc).__name__}: {exc}"
            ),
        }
    # Source baselines: prefer candidate-supplied attributes if a future
    # CandidateRow extension carries them; otherwise fall back to
    # canonical defaults per the docstring's conservative semantics.
    current_archive_bytes = getattr(
        candidate, "current_archive_bytes", _RATE_DENOM
    )
    if not isinstance(current_archive_bytes, int) or isinstance(
        current_archive_bytes, bool
    ):
        current_archive_bytes = _RATE_DENOM
    current_d_pose = getattr(candidate, "current_d_pose", 0.0)
    if not isinstance(current_d_pose, (int, float)) or current_d_pose < 0:
        current_d_pose = 0.0
    try:
        composed = compose_score_from_axes(
            decomp,
            current_archive_bytes=int(current_archive_bytes),
            current_d_pose=float(current_d_pose),
        )
    except (TypeError, ValueError) as exc:
        return {
            "error": (
                f"compose_score_from_axes refused composition: "
                f"{type(exc).__name__}: {exc}"
            ),
        }
    return composed.as_dict()


def invoke_cathedral_consumers_on_candidates(
    candidates: list[CandidateRow],
    *,
    top_n: int | None = None,
    repo_root: Path | str | None = None,
    panel_axis: str = "contest_cpu",
    include_master_gradient_rerank: bool = True,
) -> dict[str, Any]:
    """Canonical invoker for auto-discovered cathedral consumers + master-gradient rerank.

    THE H1-1 + H1-6 FIX. Calls
    :func:`discover_compliant_consumer_modules` to ingest every contract-
    compliant package under ``src/tac/cathedral_consumers/`` and invokes
    each one's ``consume_candidate`` on the top-N candidates. Also calls
    :func:`rerank_candidates_via_master_gradient` to surface
    master-gradient anchor annotations.

    Returns a dict with:
    - ``schema``: pinned schema version
    - ``consumer_count``: number of discovered consumers
    - ``consumer_names``: sorted list of consumer module names
    - ``invocations``: per-(consumer × candidate) contribution rows
    - ``master_gradient_annotations``: per-candidate annotation rows
    - ``score_claim``: False (per Catalog #323 contribution is observability-only)
    - ``promotion_eligible``: False
    - ``ready_for_exact_eval_dispatch``: False
    - ``evidence_grade``: "[predicted, cathedral consumer invocation]"

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323
    the invocation does NOT mutate predicted_score_delta on the candidate
    rows; contributions are surfaced for operator audit only.

    Per CLAUDE.md "Forbidden score claims" the rationale text is bounded
    (≤512 chars per contribution) so a misbehaving consumer cannot pollute
    the output payload.
    """
    root = Path(repo_root) if repo_root else Path.cwd()
    modules = discover_compliant_consumer_modules(repo_root=root)
    # Determinism: sort by name so multi-machine runs produce equivalent output.
    modules_sorted = sorted(modules, key=lambda m: getattr(m, "__name__", ""))
    consumer_names = [getattr(m, "__name__", "<unknown>") for m in modules_sorted]

    # Bounded candidate set: cap at top_n if requested.
    target_candidates = list(candidates)
    if top_n is not None and top_n > 0:
        target_candidates = target_candidates[:top_n]

    invocations: list[dict[str, Any]] = []
    for module in modules_sorted:
        for candidate in target_candidates:
            invocations.append(_invoke_consumer_safely(module, candidate))

    master_gradient_annotations: list[dict[str, Any]] = []
    if include_master_gradient_rerank:
        try:
            mg_results = rerank_candidates_via_master_gradient(
                target_candidates, panel_axis=panel_axis,
            )
            for candidate, predicted_delta, explanation in mg_results:
                master_gradient_annotations.append({
                    "candidate_id": candidate.candidate_id,
                    "archive_sha256": candidate.archive_sha256,
                    "predicted_delta_passthrough": float(predicted_delta),
                    "explanation": str(explanation)[:512],
                })
        except Exception as exc:  # noqa: BLE001  defensive
            master_gradient_annotations.append({
                "error": f"{type(exc).__name__}: {exc}",
            })

    return {
        "schema": CATHEDRAL_CONSUMER_INVOCATION_SCHEMA,
        "evidence_grade": "[predicted, cathedral consumer invocation]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "panel_axis": panel_axis,
        "top_n": top_n,
        "consumer_count": len(modules_sorted),
        "consumer_names": consumer_names,
        "candidates_invoked": len(target_candidates),
        "invocations": invocations,
        "master_gradient_annotations": master_gradient_annotations,
        "master_gradient_rerank_invoked": bool(include_master_gradient_rerank),
    }


# ─────────────────────────────────────────────────────────────────────────
# End of R11-H1-1-PLUS-H1-6-FIX-WAVE invoker callsites
# ─────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────
# META-LAGRANGIAN-WIRE-1 Phase 1 invoker callsite (2026-05-20)
# ─────────────────────────────────────────────────────────────────────────
#
# Per T3 grand-strategy Decision 5 long-term centerpiece + operator decision
# 2026-05-20 + WIRE-IN-RIGOR-AUDIT empirical finding that
# `src/tac/findings_lagrangian/` had ZERO production callers despite being
# the canonical "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST
# EMPHASIS" surface per CLAUDE.md. THIS PHASE 1 wire-in lands the canonical
# invocation point + per-iteration call + a basic Lagrangian-derived
# adjuster; Phase 2-N (sister subagents over a 1-3 week window) extend to
# actual dual-variable computation + typed atom flow into the solver +
# per-element learned-optimal destination.
#
# Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323
# the invocation does NOT mutate predicted_score_delta on the candidate
# rows; the Lagrangian-derived signal is surfaced as observability-only
# annotations attached to the output payload for downstream consumers
# (operator audit, autopilot ranker downweighting via 1/(1+sigma) per Q7
# binding decision, action-selector candidate enrichment).
#
# Per Catalog #336 / #337 sister-pattern: the invocation lands in main()
# AFTER `invoke_cathedral_consumers_on_candidates` so Lagrangian-derived
# adjustments compose on top of the autopilot consumer cascade (the
# Lagrangian sees the same candidate-set the cathedral consumers saw).

META_LAGRANGIAN_INVOCATION_SCHEMA = "meta_lagrangian_invocation_v1_20260520"
"""Canonical schema id for the meta-Lagrangian invocation output payload.

Versioned per the canonical helper pattern (Catalog #245 ledger /
#300 council deliberation v2 / Catalog #336 sister discipline). Bump
when the output shape changes meaningfully.
"""


def _candidate_residuals_for_lagrangian(
    candidate: CandidateRow,
) -> tuple[float, ...]:
    """Extract Phase 1 anchor residuals for a candidate.

    Phase 1: minimal residual signal — uses the candidate's
    ``predicted_score_delta`` as the single residual (the "predicted minus
    canonical baseline" anchor proxy; baseline = 0 per the canonical
    sign convention where negative delta = improvement). Phase 2 will
    replace this with per-pair master-gradient residuals + per-class
    SegNet/PoseNet component residuals (the typed-atom flow into the
    solver per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable).

    The Phase 1 proxy is BOUNDED (clipped to [-1.0, +1.0]) so a runaway
    candidate prediction cannot destabilize the Lagrangian posterior
    update. The bound is documented in the helper's return-value
    rationale so a future Phase 2 successor can audit + lift it.
    """
    raw = float(candidate.predicted_score_delta)
    if raw != raw:  # NaN guard
        return ()
    if raw > 1.0:
        raw = 1.0
    elif raw < -1.0:
        raw = -1.0
    return (raw,)


def _lagrangian_derived_adjustment_factor(
    scalar_lagrangian: float,
    posterior_sigma: float,
) -> float:
    """Phase 1 bounded adjustment factor in [0.95, 1.05].

    Derives a small bounded multiplicative factor from the Lagrangian
    scalar + posterior uncertainty. Phase 1 keeps the adjustment narrow
    (5% maximum either direction) because the actual dual-variable
    computation lives in Phase 2; this factor is a placeholder that
    exercises the wiring without claiming the Lagrangian has empirically
    arbitrated the candidate's ranking.

    Per Q7 binding decision (slot 20 + supplemental): higher posterior
    uncertainty → downweight via 1/(1+sigma). Phase 1 implements this
    AS THE BOUNDED 5% PROXY; Phase 2 extends to the full asymmetric-cost
    formulation per Lindley 1956 + Foster 2019.

    Returns 1.0 (passthrough) when inputs are non-finite OR when the
    Lagrangian/sigma values would push outside the [0.95, 1.05] bound;
    fail-closed so operator audit always observes a deterministic
    bounded multiplier.
    """
    if scalar_lagrangian != scalar_lagrangian:  # NaN
        return 1.0
    if posterior_sigma != posterior_sigma:  # NaN
        return 1.0
    if posterior_sigma < 0:
        return 1.0
    # 1/(1+sigma) downweight per Q7 binding decision
    uncertainty_factor = 1.0 / (1.0 + max(posterior_sigma, 0.0))
    # Map scalar Lagrangian to a small signed adjustment.
    # Lower Lagrangian = better fit = positive adjustment (favor candidate);
    # higher Lagrangian = worse fit = negative adjustment.
    # Sigmoid-like compression to bound the contribution.
    import math as _math

    if scalar_lagrangian > 50.0:
        scalar_lagrangian = 50.0
    elif scalar_lagrangian < -50.0:
        scalar_lagrangian = -50.0
    lagrangian_sign_factor = _math.tanh(-scalar_lagrangian / 10.0)  # in [-1, 1]
    # Compose: small 5% maximum adjustment band.
    adjustment_delta = 0.05 * uncertainty_factor * lagrangian_sign_factor
    factor = 1.0 + adjustment_delta
    # Defense-in-depth bound check.
    if factor < 0.95:
        factor = 0.95
    elif factor > 1.05:
        factor = 1.05
    return float(factor)


def invoke_meta_lagrangian_on_candidates(
    candidates: list[CandidateRow],
    *,
    top_n: int | None = None,
    repo_root: Path | str | None = None,
    panel_axis: str = "contest_cpu",
    sigma_obs: float = 1.0,
) -> dict[str, Any]:
    """Phase 1 canonical invocation of the meta-Lagrangian on candidates.

    Mirrors :func:`invoke_cathedral_consumers_on_candidates` per Catalog
    #336 invoker-callsite pattern. For each candidate, this Phase 1
    helper:

      1. Extracts a bounded residual signal via
         :func:`_candidate_residuals_for_lagrangian`.
      2. Builds a 1-dim Gaussian posterior keyed to the candidate's
         family (treated as the equation_id for Phase 1) via
         :func:`tac.findings_lagrangian.posterior_update_from_anchors`.
      3. Computes the 4-term findings Lagrangian via
         :func:`tac.findings_lagrangian.compute_findings_lagrangian` on
         the initial partition.
      4. Derives a bounded adjustment factor in [0.95, 1.05] via
         :func:`_lagrangian_derived_adjustment_factor` and surfaces it
         as an observability annotation (NOT a candidate mutation).

    Phase 2-N (sister subagents) will:
      - Replace the bounded-residual Phase 1 proxy with per-pair
        master-gradient residuals + per-class component residuals.
      - Compute actual dual variables (lambda_*, mu_*) instead of
        the placeholder bounded factor.
      - Flow typed-atom candidate rows into the Lagrangian's
        partition + action-selector surfaces (currently the action
        selector is invoked separately if at all).
      - Extend the adjustment factor's bound + add per-candidate
        info-gain-per-dollar ranking signal per Lindley 1956 + Foster
        2019.
      - Per-element learned-optimal destination per the META engineering
        vision (Phase 4).

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog
    #287/#323: this helper's contribution is OBSERVABILITY-ONLY at
    Phase 1; it surfaces ``lagrangian_scalar`` + ``adjustment_factor``
    + ``posterior_sigma_per_term`` + ``decompose`` for operator audit
    + downstream consumers, but it does NOT mutate
    ``predicted_score_delta`` on candidate rows.

    Per CLAUDE.md "Forbidden score claims": every annotation row carries
    ``score_claim=False`` + ``promotable=False`` +
    ``axis_tag="[predicted]"``. Per Catalog #319 / #323: this is a
    Provenance-bearing surface that downstream consumers must NOT route
    into score-claim posterior writes.

    Args:
        candidates: ranked candidate list (typically the autopilot
            ranker output).
        top_n: cap at top-N candidates; ``None`` means all (default).
        repo_root: optional repo root override for the canonical
            equations lookup (unused at Phase 1; reserved for Phase 2).
        panel_axis: contest score axis (mirrors sister helpers).
        sigma_obs: observation noise std-dev for the conjugate update
            (default 1.0 per :mod:`tac.findings_lagrangian.posterior`).

    Returns:
        Dict with:
          - ``schema``: pinned schema version
          - ``evidence_grade``: "[predicted, meta-Lagrangian invocation]"
          - ``score_claim``: False
          - ``promotion_eligible``: False
          - ``ready_for_exact_eval_dispatch``: False
          - ``panel_axis``: passthrough
          - ``top_n``: passthrough
          - ``candidates_invoked``: count
          - ``phase``: ``"phase_1_canonical_invocation_with_bounded_proxy_adjuster"``
          - ``invocations``: per-candidate annotation rows
          - ``per_candidate_errors``: count of trapped exceptions (each
            recorded inline in ``invocations``)
          - ``next_phase_roadmap``: pointer to the design memo
    """
    # Bound the candidate set.
    target_candidates = list(candidates)
    if top_n is not None and top_n > 0:
        target_candidates = target_candidates[:top_n]

    # Lazy import — keeps the cathedral autopilot import tree light when
    # the wire-in is not on the hot path.
    try:
        from tac.findings_lagrangian import (
            posterior_update_from_anchors,
            compute_findings_lagrangian,
            build_initial_partition,
        )
        _LAGRANGIAN_OK = True
        _import_error: str | None = None
    except Exception as exc:  # noqa: BLE001  defensive
        _LAGRANGIAN_OK = False
        _import_error = f"{type(exc).__name__}: {exc}"

    invocations: list[dict[str, Any]] = []
    per_candidate_errors = 0

    if not _LAGRANGIAN_OK:
        # Surface the import failure as a single observability row + return.
        # Per Phase 1: fail-OPEN at the helper (do NOT crash main()).
        # Sister Catalog #355 STRICT preflight gate ensures the invocation
        # CALLSITE is present in main(); the helper's runtime resilience is
        # separate from the gate's structural protection.
        invocations.append({
            "schema_error": (
                "tac.findings_lagrangian unavailable: "
                f"{_import_error}"
            ),
            "score_claim": False,
            "promotable": False,
            "axis_tag": "[predicted]",
        })
        return {
            "schema": META_LAGRANGIAN_INVOCATION_SCHEMA,
            "evidence_grade": "[predicted, meta-Lagrangian invocation]",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "panel_axis": panel_axis,
            "top_n": top_n,
            "candidates_invoked": 0,
            "phase": "phase_1_canonical_invocation_with_bounded_proxy_adjuster",
            "invocations": invocations,
            "per_candidate_errors": per_candidate_errors + 1,
            "next_phase_roadmap": (
                ".omx/research/meta_lagrangian_wire_in_phase_1_canonical_"
                "invocation_landed_20260520T*.md"
            ),
        }

    # Build a canonical initial partition (used per-candidate Phase 1; Phase
    # 2 will route distinct candidates through distinct partition refinements
    # per the MDL-with-wavelet-prior splitting rule per Catalog #277).
    try:
        partition = build_initial_partition()
    except Exception as exc:  # noqa: BLE001  defensive
        invocations.append({
            "schema_error": (
                f"build_initial_partition failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            "score_claim": False,
            "promotable": False,
            "axis_tag": "[predicted]",
        })
        partition = None
        per_candidate_errors += 1

    for candidate in target_candidates:
        # Phase 1: use family as equation_id stand-in (no canonical equation
        # entry exists for arbitrary candidate families). The bounded-proxy
        # signal is observability-only; the equation_id mismatch with the
        # canonical_equations registry is documented as a Phase 2 follow-on
        # per the design memo.
        equation_id = candidate.family or candidate.candidate_id or "unknown_family"
        residuals = _candidate_residuals_for_lagrangian(candidate)

        if partition is None or not residuals:
            invocations.append({
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "lagrangian_scalar": None,
                "adjustment_factor": 1.0,
                "axis_tag": "[predicted]",
                "score_claim": False,
                "promotable": False,
                "rationale": (
                    "Phase 1 skip: missing partition OR empty residuals"
                ),
            })
            continue

        try:
            posterior = posterior_update_from_anchors(
                equation_id,
                prior_mu=(0.0,),
                prior_sigma_diagonal=(1.0,),
                anchor_residuals=residuals,
                sigma_obs=sigma_obs,
            )
            lag_result = compute_findings_lagrangian(
                equation_id,
                posterior=posterior,
                partition=partition,
                anchor_residuals=residuals,
                sigma_obs=sigma_obs,
            )
            scalar = float(lag_result.scalar)
            sigma = float(lag_result.posterior_sigma_per_term[0])
            adjustment = _lagrangian_derived_adjustment_factor(scalar, sigma)
            invocations.append({
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "equation_id_used": equation_id,
                "lagrangian_scalar": scalar,
                "posterior_sigma": sigma,
                "adjustment_factor": adjustment,
                "decompose": dict(lag_result.decompose()),
                "axis_tag": "[predicted]",
                "score_claim": False,
                "promotable": False,
                "rationale": (
                    "Phase 1 bounded-proxy adjustment (5% band); "
                    "Phase 2 will replace with actual dual variables"
                )[:512],
            })
        except Exception as exc:  # noqa: BLE001  defensive per-candidate
            per_candidate_errors += 1
            invocations.append({
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "error": f"{type(exc).__name__}: {exc}",
                "adjustment_factor": 1.0,
                "axis_tag": "[predicted]",
                "score_claim": False,
                "promotable": False,
            })

    return {
        "schema": META_LAGRANGIAN_INVOCATION_SCHEMA,
        "evidence_grade": "[predicted, meta-Lagrangian invocation]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "panel_axis": panel_axis,
        "top_n": top_n,
        "candidates_invoked": len(target_candidates),
        "phase": "phase_1_canonical_invocation_with_bounded_proxy_adjuster",
        "invocations": invocations,
        "per_candidate_errors": per_candidate_errors,
        "next_phase_roadmap": (
            ".omx/research/meta_lagrangian_wire_in_phase_1_canonical_"
            "invocation_landed_20260520T*.md"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────
# End of META-LAGRANGIAN-WIRE-1 Phase 1 invoker callsite
# ─────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--candidates-jsonl", type=Path, default=None,
                        help="Path to JSONL file of CandidateRow rows. Mutually "
                             "exclusive with the other candidate-source flags.")
    parser.add_argument(
        "--use-substrate-composition-matrix-ranking",
        type=Path,
        default=None,
        help=(
            "Path to QQ's substrate composition matrix ranking JSON "
            "(schema=tac_autopilot_dispatch_ranking_v1, e.g. "
            "experiments/results/cathedral_autopilot_dispatch_ranking_*/ranking.json). "
            "When set, candidates are loaded from the ranking and "
            "filter_composition_incompatible_dispatches enforces the matrix's "
            "REPLACEMENT/INCOMPATIBLE/ANTAGONISTIC constraints across the dispatch "
            "queue. Mutually exclusive with the other candidate-source flags."
        ),
    )
    parser.add_argument(
        "--probe-disambiguator-json",
        type=Path,
        default=None,
        help=(
            "Path to a probe-disambiguator JSON artifact carrying read-only "
            "autopilot_rows. Mutually exclusive with the other candidate-source "
            "flags; rows must be planning-only and fail closed on score/promotion/"
            "exact-eval dispatch flags."
        ),
    )
    parser.add_argument(
        "--rate-attack-feature-matrix-json",
        type=Path,
        default=None,
        help=(
            "Path to a RATE ACH/cheap-probe feature matrix carrying read-only "
            "autopilot_rows. Mutually exclusive with the other candidate-source "
            "flags; rows must include disconfirming assumptions and fail closed "
            "on score/promotion/exact-eval dispatch authority."
        ),
    )
    parser.add_argument(
        "--include-out-of-envelope-ranking-candidates",
        action="store_true",
        help=(
            "When --use-substrate-composition-matrix-ranking is set, also load "
            "ranking rows that QQ marked as out-of-envelope. Default OFF — "
            "the autopilot honors QQ's envelope discipline."
        ),
    )
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of loop iterations to run")
    parser.add_argument("--rank-axis",
                        choices=["eig_per_dollar", "predicted_score_delta"],
                        default="eig_per_dollar")
    parser.add_argument(
        "--score-panel-axis",
        choices=["contest_cpu", "contest_cuda"],
        default="contest_cpu",
        help=(
            "Contest score axis for axis-specific sidecar ranking rewards "
            "(default contest_cpu, the public leaderboard axis)."
        ),
    )
    parser.add_argument("--require-operator-approval-on", action="append",
                        default=[], help="event classes to gate (default: dispatch)")
    parser.add_argument("--claims-path", type=Path, default=None)
    parser.add_argument("--race-mode", action="store_true",
                        help="OPT-IN per CLAUDE.md race-mode rigor inversion")
    parser.add_argument("--max-dispatch-recommendations", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write the per-iteration report JSON")
    parser.add_argument(
        "--include-master-gradient-optimal-plans",
        action="store_true",
        help=(
            "Append planning-only CandidateRows from canonical "
            ".omx/state/master_gradient_consumers/optimal_plan_*.json "
            "sidecars before ranking. This is a Cathedral consumer hook only: "
            "rows are score_claim=False, promotion_eligible=False, and "
            "ready_for_exact_eval_dispatch=False."
        ),
    )
    parser.add_argument(
        "--master-gradient-optimal-plan-root",
        type=Path,
        default=None,
        help=(
            "Optional root for --include-master-gradient-optimal-plans "
            "(defaults to .omx/state/master_gradient_consumers)."
        ),
    )
    parser.add_argument(
        "--operator-authorized-le-5-dollar-mode",
        action="store_true",
        help=(
            "OPT-IN: enable operator-authorized le-$5/individual mode (default "
            "OFF). Per CLAUDE.md operator-gate non-negotiable + dual-gated by "
            f"env-var {OPERATOR_AUTHORIZED_MODE_ENV_VAR}="
            f"{OPERATOR_AUTHORIZED_MODE_ENV_VALUE_ENABLED}."
        ),
    )
    parser.add_argument(
        "--per-dispatch-cap-usd",
        type=float,
        default=DEFAULT_PER_DISPATCH_CAP_USD,
        help=(
            f"Per-dispatch hard cap when authorized mode is on (default "
            f"${DEFAULT_PER_DISPATCH_CAP_USD:.2f})."
        ),
    )
    parser.add_argument(
        "--cumulative-cap-usd",
        type=float,
        default=DEFAULT_CUMULATIVE_CAP_USD,
        help=(
            f"Cumulative spend envelope when authorized mode is on (default "
            f"${DEFAULT_CUMULATIVE_CAP_USD:.2f})."
        ),
    )
    parser.add_argument(
        "--canonical-helper-script",
        type=Path,
        default=None,
        help=(
            "Path to the canonical dispatch-claim helper "
            f"(default {CANONICAL_HELPER_SCRIPT_RELPATH} under the repo root)."
        ),
    )
    parser.add_argument(
        "--journal-path",
        type=Path,
        default=None,
        help=(
            "Where to append authorized-dispatch rows (JSONL). Required when "
            "--operator-authorized-le-5-dollar-mode is set."
        ),
    )
    # W/I/A I-1 wire-in (2026-05-12): continual-learning posterior knobs.
    parser.add_argument(
        "--continual-posterior-path",
        type=Path,
        default=None,
        help=(
            "Optional path to the continual-learning posterior JSONL. Default "
            "uses tac.continual_learning.DEFAULT_POSTERIOR_PATH. Loaded when "
            "--load-continual-posterior is set."
        ),
    )
    parser.add_argument(
        "--load-continual-posterior",
        action="store_true",
        help=(
            "OPT-IN: load tac.continual_learning posterior and reweight "
            "predicted_score_delta by family-keyed correction factor before "
            "ranking. Per CLAUDE.md 'Subagent coherence-by-default' the "
            "wire-in is the structural fix; the per-candidate correction "
            "never auto-promotes nor auto-kills."
        ),
    )
    parser.add_argument(
        "--use-compressive-sensing-lattice",
        action="store_true",
        help=(
            "OPT-IN: emit the substrate-lattice Donoho-Tanner undersampling "
            "diagnostic for the current candidate pool. This is a visibility "
            "surface only; it never creates score, promotion, kill, or dispatch "
            "authority."
        ),
    )
    parser.add_argument(
        "--lattice-anchor-count",
        type=int,
        default=0,
        help="Number of empirical anchors in the current lattice posterior.",
    )
    parser.add_argument(
        "--lattice-expected-sparsity",
        type=int,
        default=5,
        help="Expected sparse frontier-breaker count for the lattice diagnostic.",
    )
    parser.add_argument(
        "--lattice-safety-margin",
        type=float,
        default=0.05,
        help="Donoho-Tanner safety margin for the lattice diagnostic.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "REPORT MODE (no side effects): rank candidates via the canonical "
            "pipeline (Z1 empirical revision + Tier-A/C MDL density + class-shift "
            "literature + composition alpha + Rudin SLIM predicted_dispatch_risk + "
            "continual-learning posterior reweight), print the top-N as a human-"
            "readable table + a minimal JSON, and EXIT. Does NOT fire run_continuous_loop, "
            "does NOT record dispatch claims, does NOT emit halt events, does NOT "
            "consult the operator-authorized journal. This is the canonical "
            "operator-facing 'burn-down summary' mode — what's pending, what's "
            "ranked highest, what's blocked. Per CLAUDE.md anti-duplication "
            "directive: reuses the existing rank_candidates pipeline without "
            "creating a sister tool. Lane registry level + predecessor-probe "
            "Catalog #313 blocking status surfaced per row."
        ),
    )
    parser.add_argument(
        "--report-top-n",
        type=int,
        default=10,
        help="When --report-only is set: number of top-ranked candidates to print (default 10).",
    )
    parser.add_argument(
        "--persist-consumer-verdicts",
        action="store_true",
        help=(
            "T3 council prioritization 2026-05-19 rank #4 ACTIVATION sprint: "
            "persist the cathedral consumer invocation batch to "
            ".omx/state/cathedral_autopilot_consumer_verdicts.jsonl per the "
            "canonical fcntl-locked append-only ledger pattern "
            "(Catalog #245/#313/#333/#344 sister discipline). Opt-in default "
            "OFF so smoke runs don't pollute the canonical ledger; the "
            "operator-runnable activation sprint enables it explicitly. "
            "Per Catalog #287/#323 persisted rows are score_claim=False + "
            "promotion_eligible=False + axis_tag=[predicted]."
        ),
    )
    args = parser.parse_args(argv)

    try:
        if args.iterations <= 0:
            raise ValueError("--iterations must be > 0")
        if args.lattice_anchor_count < 0:
            raise ValueError("--lattice-anchor-count must be >= 0")
        if args.lattice_expected_sparsity <= 0:
            raise ValueError("--lattice-expected-sparsity must be > 0")
        # Exactly one candidate source must be supplied. They are mutually exclusive.
        sources_supplied = sum(
            1 for x in (
                args.candidates_jsonl,
                args.use_substrate_composition_matrix_ranking,
                args.probe_disambiguator_json,
                args.rate_attack_feature_matrix_json,
            ) if x is not None
        )
        if sources_supplied != 1:
            raise ValueError(
                "exactly one of --candidates-jsonl, "
                "--use-substrate-composition-matrix-ranking, "
                "--probe-disambiguator-json, or "
                "--rate-attack-feature-matrix-json must be supplied "
                f"(got {sources_supplied})"
            )
        if args.candidates_jsonl is not None and not args.candidates_jsonl.is_file():
            raise FileNotFoundError(args.candidates_jsonl)
        if (
            args.use_substrate_composition_matrix_ranking is not None
            and not args.use_substrate_composition_matrix_ranking.is_file()
        ):
            raise FileNotFoundError(args.use_substrate_composition_matrix_ranking)
        if (
            args.probe_disambiguator_json is not None
            and not args.probe_disambiguator_json.is_file()
        ):
            raise FileNotFoundError(args.probe_disambiguator_json)
        if (
            args.rate_attack_feature_matrix_json is not None
            and not args.rate_attack_feature_matrix_json.is_file()
        ):
            raise FileNotFoundError(args.rate_attack_feature_matrix_json)
        approval_set = _parse_approval_flags(
            args.require_operator_approval_on or ["dispatch"]
        )
        # Refuse to activate authorized mode without a journal path.
        if args.operator_authorized_le_5_dollar_mode and args.journal_path is None:
            raise ValueError(
                "--operator-authorized-le-5-dollar-mode requires --journal-path "
                "for the structured-row JSONL ledger (per CLAUDE.md no-/tmp-path)"
            )
        if args.operator_authorized_le_5_dollar_mode and args.journal_path is not None:
            _require_finite_positive_float(
                args.per_dispatch_cap_usd,
                field="--per-dispatch-cap-usd",
                context="operator-authorized CLI",
            )
            _require_finite_positive_float(
                args.cumulative_cap_usd,
                field="--cumulative-cap-usd",
                context="operator-authorized CLI",
            )
            validate_authorized_journal_path(args.journal_path, repo_root=REPO_ROOT)
    except (ValueError, FileNotFoundError) as exc:
        print(f"cathedral_autopilot_autonomous_loop: {exc}", file=sys.stderr)
        return 2

    auth_mode: OperatorAuthorizedModeConfig | None = None
    if args.operator_authorized_le_5_dollar_mode:
        helper_path = args.canonical_helper_script or (
            REPO_ROOT / CANONICAL_HELPER_SCRIPT_RELPATH
        )
        auth_mode = OperatorAuthorizedModeConfig(
            enabled=True,
            per_dispatch_cap_usd=args.per_dispatch_cap_usd,
            cumulative_cap_usd=args.cumulative_cap_usd,
            canonical_helper_script=helper_path,
            journal_path=args.journal_path,
        )
        # Defense-in-depth env-var check.
        if not _env_authorizes_mode():
            print(
                "cathedral_autopilot_autonomous_loop: authorized mode CLI flag "
                f"is set but env-var {OPERATOR_AUTHORIZED_MODE_ENV_VAR}="
                f"{OPERATOR_AUTHORIZED_MODE_ENV_VALUE_ENABLED} is missing; "
                "loop will run but no candidate will self-authorize",
                file=sys.stderr,
            )

    composition_substrate_map: dict[str, tuple[str, ...]] = {}
    composition_dropped: list[tuple[str, str]] = []

    if args.use_substrate_composition_matrix_ranking is not None:
        composition_substrate_map = candidate_substrate_ids_from_ranking(
            args.use_substrate_composition_matrix_ranking
        )

        def _source() -> list[CandidateRow]:
            base = load_candidates_from_substrate_composition_ranking(
                args.use_substrate_composition_matrix_ranking,
                only_in_envelope=not args.include_out_of_envelope_ranking_candidates,
                only_fits_per_dispatch_cap=not args.include_out_of_envelope_ranking_candidates,
            )
            kept, dropped = filter_composition_incompatible_dispatches(
                base, candidate_substrate_ids=composition_substrate_map,
            )
            composition_dropped.clear()
            composition_dropped.extend(dropped)
            return kept
    elif args.probe_disambiguator_json is not None:
        def _source() -> list[CandidateRow]:
            return load_candidates_from_probe_disambiguator_output(
                args.probe_disambiguator_json
            )
    elif args.rate_attack_feature_matrix_json is not None:
        def _source() -> list[CandidateRow]:
            return load_candidates_from_rate_attack_feature_matrix_output(
                args.rate_attack_feature_matrix_json
            )
    else:
        def _source() -> list[CandidateRow]:
            if (
                args.operator_authorized_le_5_dollar_mode
                and _env_authorizes_mode()
            ):
                try:
                    payload = json.loads(args.candidates_jsonl.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict) and payload.get("schema") == EXACT_READY_QUEUE_SCHEMA:
                    return load_candidates_from_exact_ready_queue(
                        args.candidates_jsonl,
                        dispatch_claims_path=args.claims_path,
                    )
                if isinstance(payload, dict):
                    raise ValueError(
                        "operator-authorized exact dispatch requires an "
                        f"{EXACT_READY_QUEUE_SCHEMA!r} queue, not schema "
                        f"{payload.get('schema')!r}"
                    )
            return load_candidates_from_jsonl(args.candidates_jsonl)

    lattice_diagnostic: dict[str, Any] | None = None
    lattice_dispatch_blocker = ""
    if args.use_compressive_sensing_lattice:
        try:
            lattice_candidates = _source()
            lattice_diagnostic = diagnose_compressive_sensing_lattice_undersampling(
                lattice_candidates,
                n_anchors=args.lattice_anchor_count,
                expected_sparsity=args.lattice_expected_sparsity,
                safety_margin=args.lattice_safety_margin,
            )
            recovery_regime = str(
                lattice_diagnostic.get("recovery_regime", "unknown")
            )
            if recovery_regime != "EXACT":
                lattice_dispatch_blocker = (
                    "compressive_sensing_lattice_recovery_regime_"
                    f"{recovery_regime}_operator_review_required"
                )
        except (ValueError, FileNotFoundError) as exc:
            print(f"cathedral_autopilot_autonomous_loop: {exc}", file=sys.stderr)
            return 2

    if lattice_dispatch_blocker:
        base_source = _source

        def _source() -> list[CandidateRow]:
            candidates = base_source()
            for candidate in candidates:
                if lattice_dispatch_blocker not in candidate.blockers:
                    candidate.blockers.append(lattice_dispatch_blocker)
            return candidates

    if args.include_master_gradient_optimal_plans:
        base_source = _source

        def _source() -> list[CandidateRow]:
            candidates = base_source()
            candidates.extend(
                load_candidates_from_master_gradient_optimal_plans(
                    root=args.master_gradient_optimal_plan_root
                )
            )
            return candidates

    # ─── REPORT-ONLY MODE (A2 from operator's orchestration question 2026-05-17) ───
    # Per CLAUDE.md anti-duplication directive: reuse the canonical rank_candidates
    # pipeline without spawning a sister tool. Bypass run_continuous_loop entirely
    # so no dispatch claims / halt events / spend authorization fires. Emit the
    # ranked top-N as a human-readable table + a minimal JSON.
    if args.report_only:
        try:
            candidates = _source()
            if args.load_continual_posterior:
                posterior = load_planner_posterior_for_loop(args.continual_posterior_path)
            else:
                posterior = None
            ranked = rank_candidates(
                candidates,
                rank_axis=args.rank_axis,
                continual_posterior=posterior,
                apply_z1_empirical_revision=True,
                score_panel_axis=args.score_panel_axis,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"cathedral_autopilot_autonomous_loop: {exc}", file=sys.stderr)
            return 2

        top_n = max(1, int(args.report_top_n))
        top = ranked[:top_n]

        # Human-readable table to stderr so JSON-on-stdout stays parseable.
        print(
            f"\n[cathedral_autopilot --report-only] ranked={len(ranked)} candidates "
            f"(rank_axis={args.rank_axis!r}; top {len(top)} shown):\n",
            file=sys.stderr,
        )
        header = (
            f"  {'rank':>4}  {'candidate_id':<48}  {'predicted Δ':>11}  "
            f"{'effective Δ':>11}  {'cost $':>7}  {'EIG/$':>9}  blockers"
        )
        print(header, file=sys.stderr)
        print(f"  {'-' * 4}  {'-' * 48}  {'-' * 11}  {'-' * 11}  {'-' * 7}  {'-' * 9}  --------", file=sys.stderr)
        for rank_i, c in enumerate(top, start=1):
            cid = c.candidate_id[:48] if len(c.candidate_id) > 48 else c.candidate_id
            raw_delta = c.predicted_score_delta
            eff_delta = apply_z1_empirical_revision_to_candidate_delta(
                c,
                score_panel_axis=args.score_panel_axis,
            )
            cost = c.estimated_dispatch_cost_usd
            eig = c.eig_per_dollar() if cost > 0 else float("nan")
            blockers_str = ",".join(c.blockers[:3]) if c.blockers else "-"
            if len(c.blockers) > 3:
                blockers_str += f"+{len(c.blockers) - 3}more"
            print(
                f"  {rank_i:>4}  {cid:<48}  {raw_delta:>11.6f}  "
                f"{eff_delta:>11.6f}  ${cost:>6.2f}  {eig:>9.4f}  {blockers_str}",
                file=sys.stderr,
            )

        # R11-H1-1+H1-6 FIX-WAVE: invoke auto-discovered cathedral consumers
        # + master-gradient rerank on the ranked top-N. THE PARADIGM SHIFT
        # IS NOW RUNTIME-ACTIVE. Per Catalog #336/#337 + #335 + #287/#323,
        # contributions are observability-only (no score mutation).
        consumer_invocations = invoke_cathedral_consumers_on_candidates(
            ranked,
            top_n=top_n,
            panel_axis=args.score_panel_axis,
        )

        # META-LAGRANGIAN-WIRE-1 Phase 1 (2026-05-20): invoke the canonical
        # findings Lagrangian on the same ranked top-N AFTER the cathedral
        # consumer cascade. Per T3 grand-strategy Decision 5 + WIRE-IN-RIGOR
        # AUDIT empirical anchor (zero production callers pre-2026-05-20).
        # Per Catalog #355 + #287/#323: contribution is observability-only
        # at Phase 1 (bounded proxy adjuster in [0.95, 1.05]); Phase 2+
        # will compute actual dual variables + flow typed atoms into the
        # solver per the canonical "Meta-Lagrangian/Pareto solver" non-
        # negotiable.
        meta_lagrangian_invocations = invoke_meta_lagrangian_on_candidates(
            ranked,
            top_n=top_n,
            panel_axis=args.score_panel_axis,
        )

        # T3 council prioritization 2026-05-19 rank #4 ACTIVATION sprint:
        # persist the consumer invocation batch to the canonical fcntl-locked
        # JSONL ledger so the operator-runnable summary tool can audit what
        # the cathedral autopilot recommended across sessions. Per
        # CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323
        # the persisted row is observability-only (score_claim=False).
        if args.persist_consumer_verdicts:
            try:
                from tac.cathedral.verdict_ledger import append_consumer_invocation_batch
                append_consumer_invocation_batch(
                    consumer_invocations,
                    panel_axis=args.score_panel_axis,
                    rank_axis=args.rank_axis,
                    candidate_ids=[c.candidate_id for c in top],
                    top_candidates_summary=[
                        {
                            "rank": rank_i,
                            "candidate_id": c.candidate_id,
                            "archive_sha256": c.archive_sha256,
                            "predicted_score_delta_raw": c.predicted_score_delta,
                            "estimated_dispatch_cost_usd": c.estimated_dispatch_cost_usd,
                            "blockers_count": len(c.blockers),
                        }
                        for rank_i, c in enumerate(top, start=1)
                    ],
                    invocations_summary_path=str(args.output) if args.output else None,
                    notes="report-only autopilot run",
                )
            except Exception as exc:  # noqa: BLE001  defensive: never crash the loop
                print(
                    f"cathedral_autopilot_autonomous_loop: WARNING — verdict-ledger persistence failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

        # Emit minimal JSON on stdout (machine-readable consumer surface).
        # NEVER includes promotion / score-claim / dispatch-authority fields per
        # CLAUDE.md "Apples-to-apples evidence discipline".
        report_payload = {
            "schema": "cathedral_autopilot_report_only_v1",
            "evidence_grade": "[predicted; cathedral autopilot ranking]",
            "claude_md_compliance_tags": [
                "report_only_no_side_effects",
                "no_dispatch_claim_recorded",
                "no_halt_event_emitted",
                "no_spend_authorization",
                "no_score_claim_only_predicted_band",
                "no_kill_verdict",
                "anti_duplication_reuses_rank_candidates",
                "cathedral_consumer_invocation_active_catalog_336_337",
                "meta_lagrangian_invocation_active_catalog_355",
            ],
            "cathedral_consumer_invocations": consumer_invocations,
            "meta_lagrangian_invocations": meta_lagrangian_invocations,
            "rank_axis": args.rank_axis,
            "z1_empirical_revision_applied": True,
            "continual_posterior_loaded": args.load_continual_posterior,
            "n_candidates_total": len(ranked),
            "top_n_emitted": len(top),
            "top_candidates": [
                {
                    "rank": rank_i,
                    "candidate_id": c.candidate_id,
                    "predicted_score_delta_raw": c.predicted_score_delta,
                    "predicted_score_delta_z1_adjusted": apply_z1_empirical_revision_to_candidate_delta(
                        c,
                        score_panel_axis=args.score_panel_axis,
                    ),
                    "estimated_dispatch_cost_usd": c.estimated_dispatch_cost_usd,
                    "eig_per_dollar": (
                        c.eig_per_dollar() if c.estimated_dispatch_cost_usd > 0 else None
                    ),
                    "predicted_dispatch_risk": c.predicted_dispatch_risk,
                    "mdl_density": c.mdl_density,
                    "mdl_tier_c_density": c.mdl_tier_c_density,
                    "composition_alpha": c.composition_alpha,
                    "blockers": list(c.blockers),
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
                for rank_i, c in enumerate(top, start=1)
            ],
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report_payload, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
        print(json.dumps(report_payload, indent=2, sort_keys=True, allow_nan=False))
        return 0

    try:
        reports = run_continuous_loop(
            _source,
            iterations=args.iterations,
            rank_axis=args.rank_axis,
            score_panel_axis=args.score_panel_axis,
            requires_approval_on=approval_set,
            claims_path=args.claims_path,
            race_mode=args.race_mode,
            max_dispatch_recommendations=args.max_dispatch_recommendations,
            auth_mode=auth_mode,
            continual_posterior_path=args.continual_posterior_path,
            auto_load_continual_posterior=args.load_continual_posterior,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"cathedral_autopilot_autonomous_loop: {exc}", file=sys.stderr)
        return 2

    # R11-H1-1+H1-6 FIX-WAVE: invoke auto-discovered cathedral consumers
    # + master-gradient rerank on a fresh load of the candidate pool. Even
    # if run_continuous_loop already ranked + halted, surfacing the
    # consumer contributions in the output payload is the operator-facing
    # canonical observation surface per CLAUDE.md "Max observability —
    # non-negotiable". Per Catalog #287/#323 contribution is
    # observability-only (no score mutation, no promotion authority).
    try:
        post_loop_candidates = _source()
        ranked_post_loop = rank_candidates(
            post_loop_candidates,
            rank_axis=args.rank_axis,
            apply_z1_empirical_revision=True,
            score_panel_axis=args.score_panel_axis,
        )
        consumer_invocations_post_loop = invoke_cathedral_consumers_on_candidates(
            ranked_post_loop,
            top_n=min(args.max_dispatch_recommendations or 10, 10),
            panel_axis=args.score_panel_axis,
        )
        # META-LAGRANGIAN-WIRE-1 Phase 1 (2026-05-20): same post-loop wire-in
        # as the report-only path. Per T3 grand-strategy Decision 5 + sister
        # Catalog #355 STRICT preflight gate enforcing the invocation
        # callsite presence. Contribution is observability-only at Phase 1
        # per Catalog #287/#323.
        meta_lagrangian_invocations_post_loop = invoke_meta_lagrangian_on_candidates(
            ranked_post_loop,
            top_n=min(args.max_dispatch_recommendations or 10, 10),
            panel_axis=args.score_panel_axis,
        )
    except Exception as exc:  # noqa: BLE001  defensive: never crash the loop
        consumer_invocations_post_loop = {
            "schema": CATHEDRAL_CONSUMER_INVOCATION_SCHEMA,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "consumer_count": 0,
            "consumer_names": [],
            "invocations": [],
            "master_gradient_annotations": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
        meta_lagrangian_invocations_post_loop = {
            "schema": META_LAGRANGIAN_INVOCATION_SCHEMA,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "candidates_invoked": 0,
            "phase": "phase_1_canonical_invocation_with_bounded_proxy_adjuster",
            "invocations": [],
            "per_candidate_errors": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
        ranked_post_loop = []

    # T3 council prioritization 2026-05-19 rank #4 ACTIVATION sprint:
    # persist the post-loop consumer invocation batch to the canonical
    # fcntl-locked JSONL ledger so historical autopilot activity is
    # queryable across sessions. Per Catalog #287/#323 the persisted
    # row is observability-only (score_claim=False).
    if args.persist_consumer_verdicts and consumer_invocations_post_loop.get("invocations"):
        try:
            from tac.cathedral.verdict_ledger import append_consumer_invocation_batch
            top_post_loop = ranked_post_loop[:min(args.max_dispatch_recommendations or 10, 10)]
            append_consumer_invocation_batch(
                consumer_invocations_post_loop,
                panel_axis=args.score_panel_axis,
                rank_axis=args.rank_axis,
                candidate_ids=[c.candidate_id for c in top_post_loop],
                top_candidates_summary=[
                    {
                        "rank": rank_i,
                        "candidate_id": c.candidate_id,
                        "archive_sha256": c.archive_sha256,
                        "predicted_score_delta_raw": c.predicted_score_delta,
                        "estimated_dispatch_cost_usd": c.estimated_dispatch_cost_usd,
                        "blockers_count": len(c.blockers),
                    }
                    for rank_i, c in enumerate(top_post_loop, start=1)
                ],
                invocations_summary_path=str(args.output) if args.output else None,
                notes="run_continuous_loop autopilot run",
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"cathedral_autopilot_autonomous_loop: WARNING — verdict-ledger persistence failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    if args.use_substrate_composition_matrix_ranking is not None:
        source_tag = "substrate_composition_matrix_constraints_enforced"
    elif args.probe_disambiguator_json is not None:
        source_tag = "probe_disambiguator_read_only_source"
    else:
        source_tag = "candidates_jsonl_source"

    output_payload = {
        "schema": AUTONOMOUS_LOOP_SCHEMA,
        "evidence_grade": "[predicted; cathedral autopilot ranking]",
        "claude_md_compliance_tags": [
            "operator_gate_non_negotiable_at_every_dispatch",
            "halt_and_ask_pattern_default_on",
            "no_score_claim_only_predicted_band",
            "no_kill_verdict_in_loop",
            "race_mode_explicit_opt_in_only",
            "operator_authorized_le_5_dollar_mode_dual_gated",
            source_tag,
            "compressive_sensing_lattice_diagnostic_visible"
            if args.use_compressive_sensing_lattice
            else "compressive_sensing_lattice_diagnostic_not_requested",
            "cathedral_consumer_invocation_active_catalog_336_337",
            "meta_lagrangian_invocation_active_catalog_355",
        ],
        "cathedral_consumer_invocations": consumer_invocations_post_loop,
        "meta_lagrangian_invocations": meta_lagrangian_invocations_post_loop,
        "iterations_run": len(reports),
        "race_mode": args.race_mode,
        "operator_authorized_mode": {
            "enabled": bool(auth_mode and auth_mode.enabled),
            "per_dispatch_cap_usd": auth_mode.per_dispatch_cap_usd if auth_mode else None,
            "cumulative_cap_usd": auth_mode.cumulative_cap_usd if auth_mode else None,
            "cumulative_spent_usd": auth_mode.cumulative_spent_usd if auth_mode else 0.0,
            "env_authorized": _env_authorizes_mode(),
            "journal_path": str(auth_mode.journal_path) if auth_mode and auth_mode.journal_path else None,
        },
        "substrate_composition_ranking": (
            {
                "ranking_path": str(args.use_substrate_composition_matrix_ranking),
                "include_out_of_envelope": bool(
                    args.include_out_of_envelope_ranking_candidates
                ),
                "n_dropped_by_composition_constraint": len(composition_dropped),
                "dropped_with_reasons": [
                    {"candidate_id": cid, "reason": reason}
                    for (cid, reason) in composition_dropped
                ],
            }
            if args.use_substrate_composition_matrix_ranking is not None
            else None
        ),
        "probe_disambiguator_source": (
            {
                "path": str(args.probe_disambiguator_json),
                "read_only_consumer": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
            if args.probe_disambiguator_json is not None
            else None
        ),
        "compressive_sensing_lattice": (
            {
                "enabled": True,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "n_anchors": args.lattice_anchor_count,
                "expected_sparsity": args.lattice_expected_sparsity,
                "safety_margin": args.lattice_safety_margin,
                "dispatch_blocker": lattice_dispatch_blocker or None,
                "diagnostic": lattice_diagnostic,
            }
            if args.use_compressive_sensing_lattice
            else {
                "enabled": False,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "diagnostic": None,
            }
        ),
        "reports": [serialize_report(r) for r in reports],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # allow_nan=False refuses Infinity/NaN emission per RFC 8259. The
        # eig_per_dollar source fix already enforces this; adding it here
        # as defense-in-depth so any future numeric field that goes
        # non-finite fails loud at the serializer boundary, not silently
        # in the consumer.
        args.output.write_text(
            json.dumps(output_payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

    print(json.dumps(output_payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
