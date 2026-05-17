# SPDX-License-Identifier: MIT
"""Canonical lattice-state ledger — fcntl-locked JSONL append-only audit trail.

Operator binding constraint 2026-05-16 verbatim *"Remember we need outside
nerv-family too"* + the canonical Path 2 LATTICE OF CLASS-SHIFTS framework
(per `feedback_path_2_lattice_of_class_shifts_operator_approved_supersedes_l5_v2_staircase_20260516.md`)
require a queryable index of every substrate's lattice coordinate (rule /
horizon_class / architectural_class / status / evidence_score) so the
cathedral autopilot ranker + Wave-N dispatch planners + future subagents
can answer:

    - Which lattice rule does substrate X target?
    - Which substrates target rule N (Rule #1 / #2 / #3 / #4 / #5)?
    - Which architectural classes are over-represented in the next dispatch?
    - Which lattice rules currently have NO coverage?
    - How many in-flight measurements are outside the NeRV family?

This module is the canonical primary index. It mirrors the Catalog #245 Modal
call_id ledger + Catalog #313 probe-outcomes ledger 4-layer pattern, but the
schema is FORKED for lattice coordinates (substrate ↔ rule ↔ horizon_class ↔
architectural_class ↔ status ↔ paradigm-vs-implementation classification).

Schema (one event per JSONL row)
────────────────────────────────
Every row is one event in a substrate's lattice-coordinate lifecycle. The
ledger is APPEND-ONLY per CLAUDE.md "HISTORICAL_PROVENANCE" classification +
Catalog #110 / #113 / #132 — rows are NEVER mutated; new state becomes NEW
rows referencing the same ``lattice_node_id``. The current state is the
chronologically latest row for that node.

Required event_types::

    - "registered"             — initial lattice-coordinate assignment
    - "reclassified"           — coordinate updated (e.g. horizon_class changed)
    - "promoted"               — substrate advanced (e.g. status lifted_pending_council → dispatched)
    - "deferred"               — substrate moved to deferred status
    - "reactivated"            — deferred substrate reactivated
    - "operator_override"      — operator explicitly changed a coordinate

Required lattice_rule tokens (per the canonical 5 falling rules)::

    - "rule_1_chroma_preserving_neural_optional"   — NSCS06 v7 paradigm
    - "rule_2_nullspace_split_pr95_paradigm"       — NSCS01 paradigm
    - "rule_3_dykstra_stack_composition"           — A-STACK paradigm
    - "rule_4_daubechies_wavelet_multi_scale"      — Z6+Rudin+Tishby asymptotic
    - "rule_5_request_operator_review"             — whiteboard mode

Required horizon_class tokens (per Catalog #309)::

    - "plateau_adjacent"      — predicted CPU band [0.190, 0.200]
    - "frontier_pursuit"      — predicted CPU band [0.150, 0.190]
    - "asymptotic_pursuit"    — predicted CPU band [0.050, 0.150]
    - "won"                   — already achieved (e.g. NSCS06 Rule #1)
    - "n_a"                   — not applicable (apparatus / disambiguator)

Required architectural_class tokens (the OPERATOR-BINDING NERV-FAMILY axis)::

    - "nerv_family"                          — HiNeRV, sane_hnerv, FFNeRV, DSNeRV, TCNeRV, BlockNeRV, etc.
    - "pr95_paradigm_nullspace_split"        — NSCS01
    - "pr95_paradigm_downsample_renderer"    — NSCS02
    - "balle_2018_end_to_end_joint_codec"    — NSCS03
    - "stack_composition"                    — A-STACK, stack_of_stacks
    - "chroma_preserving_no_neural"          — NSCS06 family
    - "wire_grammar_class_shift"             — Wunderkind G1 v1/v2/v3
    - "cooperative_receiver_codec"           — ATW v1/v2
    - "predictive_coding_hierarchical"       — Z6/Z7/Z8 (Rao-Ballard + Daubechies + DreamerV3 + Wyner-Ziv)
    - "interpretable_ml_compositional"       — Rudin floor substrate
    - "info_bottleneck_pure"                 — Tishby IB-pure substrate
    - "imp_iterative_magnitude_pruning"      — Lane 17 IMP
    - "stc_steganography"                    — STC clean-source, STC v2
    - "apogee_qat"                           — apogee_int4/5/6/7/8 (QAT)
    - "wyner_ziv_frame_zero"                 — D4 substrate
    - "polytope_margin_overlay"              — D1 substrate
    - "world_model_foveation"                — C1 substrate
    - "mdl_information_bottleneck"           — C6 e4 MDL-IBPS
    - "self_compress"                        — Selfcomp self_compress_nn, PR56
    - "balle_renderer_hyperprior"            — balle_renderer (NSCS03 sister)
    - "siren_implicit_neural"                — SIREN
    - "wavelet_codec"                        — wavelet substrate
    - "vq_vae_codec"                         — VQ-VAE
    - "cool_chic_neural"                     — Cool-Chic
    - "grayscale_lut_renderer"               — grayscale_lut, NSCS06 luma-only variant
    - "z3_balle_hyperprior_bolton"           — Z3 family
    - "g1_entropy_coded_class_shift"         — Z3-G1 wire-grammar variant
    - "lapose_residual_bolton"               — A1+LAPose
    - "wavelet_residual_bolton"              — A1+wavelet_residual
    - "frame_exploit_selector"               — PR101 frame-exploit family
    - "latent_sidecar_compositional"         — PR106 r2 family
    - "pretrained_driving_prior"             — DP1 (Comma2k19 codebook)
    - "boundary_only_renderer"               — SABOR
    - "lossless_byte_recovery"               — S2SBS byte stuffing
    - "byte_stuffing_steganography"          — sister of STC
    - "ego_motion_focused_renderer"          — ego_nerv (NERV-FAMILY + ego-motion specialization)
    - "diffusion_renderer"                   — diffusion_renderer substrate
    - "differentially_private_sims"          — DP sims renderer
    - "hybrid_renderer_residual"             — hybrid_renderer_residual
    - "sar_coherent_pose_pairs"              — SAR
    - "sane_hnerv_family"                    — sane_hnerv canonical (NeRV-family sub-class)
    - "balle_renderer_e2e"                   — balle_renderer (NSCS03 trainer surface)

Required status tokens::

    - "lifted_dispatch_ready"            — _full_main lifted; recipe dispatch_enabled=true
    - "lifted_pending_council"           — _full_main lifted; recipe research_only=true (Phase 2 council pending)
    - "deferred_per_probe_disambiguator" — DEFER verdict from probe per Catalog #313
    - "deferred_per_operator_decision"   — DEFERRED per operator (e.g. NSCS06 family)
    - "deferred_per_audit"               — DEFER per audit verdict (e.g. Wunderkind G1 v2)
    - "not_yet_lifted"                   — substrate scaffold exists; _full_main raises NotImplementedError
    - "dispatched_evidence_landed"       — paid dispatch fired; contest-CUDA or contest-CPU anchor exists
    - "scaffold_l0"                      — registry sketch only; no implementation
    - "killed"                           — KILL verdict (rare; LAST RESORT per CLAUDE.md)
    - "archived"                         — terminal verdict; reactivation criteria documented

Required paradigm_vs_implementation_classification tokens (per Catalog #307)::

    - "paradigm_intact"               — substrate's paradigm is HARD-EARNED
    - "implementation_cargo_cult"     — implementation-level cargo-cult unwound
    - "implementation_falsified"      — implementation-level FALSIFIED; paradigm intact
    - "paradigm_falsified"            — paradigm-level FALSIFIED (rare)
    - "tbd"                           — not yet classified

Schema fields per row::

    {
        "schema_version": 1,
        "event_type": "registered" | ...,
        "lattice_node_id": "nscs01",
        "substrate": "nscs01_nullspace_split_renderer",  # canonical trainer/recipe ID
        "recipe_path": ".omx/operator_authorize_recipes/substrate_nscs01_*.yaml",
        "trainer_path": "experiments/train_substrate_nscs01_nullspace_split_renderer.py",
        "lattice_rule": "rule_2_nullspace_split_pr95_paradigm",
        "horizon_class": "plateau_adjacent",  # may be reclassified at dispatch
        "architectural_class": "pr95_paradigm_nullspace_split",
        "status": "lifted_pending_council",
        "paradigm_vs_implementation_classification": "paradigm_intact",
        "evidence_score": null,           # float; canonical [score, axis] anchor
        "evidence_score_axis": null,      # str; per "Apples-to-apples evidence discipline"
        "evidence_artifact_path": null,   # str; path to durable evidence artifact
        "lane_id": "lane_nscs01_*",
        "registered_at_utc": "2026-05-16T...",
        "expires_at_utc": "2026-06-15T...",  # 30-day staleness per Catalog #298
        "written_at_utc": "...",
        "written_pid": ...,
        "written_host": "...",
        "agent": "claude" | "codex" | "operator",
        "subagent_id": "coherence_audit_*",
        "session_id": null,
        "notes": "free-form context",
    }

Path discipline
───────────────
- ``LATTICE_STATE_LEDGER_PATH`` = ``.omx/state/lattice_state.jsonl``
  COMMITTED per HISTORICAL_PROVENANCE classification (Catalog #113).
- The lock file ``.lock`` is gitignored (LIVE_STATE).
- ``.tmp.<uuid12>`` files are gitignored (LIVE_STATE).
- Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact" — the ledger
  lives at ``.omx/state/`` under the repo root.

Bare writes are FORBIDDEN
─────────────────────────
Per CLAUDE.md Catalog #131 (``check_no_bare_writes_to_shared_state``) every
write to ``LATTICE_STATE_LEDGER_PATH`` MUST acquire ``fcntl.flock(LOCK_EX)``
on the lock file + use a unique ``.tmp.<uuid12>`` + ``os.replace``. The
public API (``register_lattice_node`` / ``update_lattice_node``) does this;
direct ``open(...).write(...)`` outside the canonical helper is refused by
Catalog #131 sister gate (this module's path should be registered there in
a follow-on landing).

Memory: feedback_coherence_audit_lattice_coordinate_assignment_20260516.md.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import json
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LATTICE_STATE_LEDGER_PATH = REPO_ROOT / ".omx" / "state" / "lattice_state.jsonl"
LATTICE_STATE_LEDGER_LOCK = LATTICE_STATE_LEDGER_PATH.with_suffix(
    LATTICE_STATE_LEDGER_PATH.suffix + ".lock"
)

# Schema version pinned for forward compatibility.
SCHEMA_VERSION = 1

# Lock acquisition timeout (seconds). Single-row appends are <10ms; 30s is
# generous even under heavy fan-out contention from sibling subagents.
LOCK_TIMEOUT_SECONDS = 30

# Staleness window — coordinates older than this default age are advisory.
DEFAULT_STALENESS_WINDOW_DAYS = 30

# Canonical event taxonomy.
EVENT_REGISTERED = "registered"
EVENT_RECLASSIFIED = "reclassified"
EVENT_PROMOTED = "promoted"
EVENT_DEFERRED = "deferred"
EVENT_REACTIVATED = "reactivated"
EVENT_OPERATOR_OVERRIDE = "operator_override"

VALID_EVENT_TYPES = frozenset(
    {
        EVENT_REGISTERED,
        EVENT_RECLASSIFIED,
        EVENT_PROMOTED,
        EVENT_DEFERRED,
        EVENT_REACTIVATED,
        EVENT_OPERATOR_OVERRIDE,
    }
)

# Canonical lattice-rule taxonomy per the 5 falling rules.
RULE_1 = "rule_1_chroma_preserving_neural_optional"
RULE_2 = "rule_2_nullspace_split_pr95_paradigm"
RULE_3 = "rule_3_dykstra_stack_composition"
RULE_4 = "rule_4_daubechies_wavelet_multi_scale"
RULE_5 = "rule_5_request_operator_review"

VALID_LATTICE_RULES = frozenset({RULE_1, RULE_2, RULE_3, RULE_4, RULE_5})

# Canonical horizon-class taxonomy (per Catalog #309).
HORIZON_PLATEAU_ADJACENT = "plateau_adjacent"
HORIZON_FRONTIER_PURSUIT = "frontier_pursuit"
HORIZON_ASYMPTOTIC_PURSUIT = "asymptotic_pursuit"
HORIZON_WON = "won"
HORIZON_NA = "n_a"

VALID_HORIZON_CLASSES = frozenset(
    {
        HORIZON_PLATEAU_ADJACENT,
        HORIZON_FRONTIER_PURSUIT,
        HORIZON_ASYMPTOTIC_PURSUIT,
        HORIZON_WON,
        HORIZON_NA,
    }
)

# Canonical status taxonomy.
STATUS_LIFTED_DISPATCH_READY = "lifted_dispatch_ready"
STATUS_LIFTED_PENDING_COUNCIL = "lifted_pending_council"
STATUS_DEFERRED_PROBE = "deferred_per_probe_disambiguator"
STATUS_DEFERRED_OPERATOR = "deferred_per_operator_decision"
STATUS_DEFERRED_AUDIT = "deferred_per_audit"
STATUS_NOT_YET_LIFTED = "not_yet_lifted"
STATUS_DISPATCHED_EVIDENCE = "dispatched_evidence_landed"
STATUS_SCAFFOLD_L0 = "scaffold_l0"
STATUS_KILLED = "killed"
STATUS_ARCHIVED = "archived"

VALID_STATUSES = frozenset(
    {
        STATUS_LIFTED_DISPATCH_READY,
        STATUS_LIFTED_PENDING_COUNCIL,
        STATUS_DEFERRED_PROBE,
        STATUS_DEFERRED_OPERATOR,
        STATUS_DEFERRED_AUDIT,
        STATUS_NOT_YET_LIFTED,
        STATUS_DISPATCHED_EVIDENCE,
        STATUS_SCAFFOLD_L0,
        STATUS_KILLED,
        STATUS_ARCHIVED,
    }
)

# Paradigm-vs-implementation classification (per Catalog #307).
CLASSIFICATION_PARADIGM_INTACT = "paradigm_intact"
CLASSIFICATION_IMPLEMENTATION_CARGO_CULT = "implementation_cargo_cult"
CLASSIFICATION_IMPLEMENTATION_FALSIFIED = "implementation_falsified"
CLASSIFICATION_PARADIGM_FALSIFIED = "paradigm_falsified"
CLASSIFICATION_TBD = "tbd"

VALID_CLASSIFICATIONS = frozenset(
    {
        CLASSIFICATION_PARADIGM_INTACT,
        CLASSIFICATION_IMPLEMENTATION_CARGO_CULT,
        CLASSIFICATION_IMPLEMENTATION_FALSIFIED,
        CLASSIFICATION_PARADIGM_FALSIFIED,
        CLASSIFICATION_TBD,
    }
)

# Canonical architectural-class taxonomy — the operator-binding NeRV-family axis.
# A substrate is OUTSIDE-NERV-FAMILY iff its architectural_class is NOT in
# NERV_FAMILY_ARCHITECTURAL_CLASSES. The operator's 2026-05-16 binding
# constraint requires at least 3-4 outside-NeRV substrates in any K-measurement
# Wave-N plan.
NERV_FAMILY_ARCHITECTURAL_CLASSES = frozenset(
    {
        "nerv_family",
        "ego_motion_focused_renderer",  # ego_nerv is NeRV-family + ego-motion specialization
        "sane_hnerv_family",
    }
)


class LatticeStateLedgerCorruptError(RuntimeError):
    """Raised when the lattice-state ledger file is corrupt and cannot be
    safely appended to.

    Sister of ``ProbeOutcomesLedgerCorruptError`` + ``CallIdLedgerCorruptError``
    + ``ActiveJobsCorruptError`` (Catalog #138 strict-load discipline). The
    append helpers raise this rather than silently overwriting the bad file,
    which would erase the historical audit trail. The corrupt file is
    QUARANTINED to ``.corrupt.<utc>`` so the operator can inspect.
    """


_ledger_lock_depth_tls = threading.local()


def _get_ledger_lock_depth() -> int:
    return int(getattr(_ledger_lock_depth_tls, "depth", 0))


def _set_ledger_lock_depth(value: int) -> None:
    _ledger_lock_depth_tls.depth = int(value)


def _ledger_lock_held() -> bool:
    """Return True if THIS thread is currently inside ``_ledger_lock``."""
    return _get_ledger_lock_depth() > 0


@contextlib.contextmanager
def _ledger_lock(lock_path: Path | None = None):
    """Acquire fcntl exclusive lock on the ledger lock file.

    Process-advisory (``fcntl.flock`` ``LOCK_EX``); multiple processes
    contending serialize on the lock file. Same-process re-entry is counted.
    """
    p = lock_path or LATTICE_STATE_LEDGER_LOCK
    p.parent.mkdir(parents=True, exist_ok=True)
    depth = _get_ledger_lock_depth()
    if depth > 0:
        _set_ledger_lock_depth(depth + 1)
        try:
            yield None
        finally:
            _set_ledger_lock_depth(_get_ledger_lock_depth() - 1)
        return
    fd = os.open(str(p), os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"could not acquire {p} within {LOCK_TIMEOUT_SECONDS}s"
                    ) from None
                time.sleep(0.05)
        _set_ledger_lock_depth(_get_ledger_lock_depth() + 1)
        try:
            yield fd
        finally:
            _set_ledger_lock_depth(_get_ledger_lock_depth() - 1)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _now_iso() -> str:
    """Return UTC timestamp in ISO-8601 format with microsecond precision."""
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _quarantine_corrupt_file(path: Path) -> Path:
    if not path.exists():
        return path
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    quarantine = path.with_suffix(path.suffix + f".corrupt.{ts}")
    counter = 0
    while quarantine.exists():
        counter += 1
        quarantine = path.with_suffix(path.suffix + f".corrupt.{ts}.{counter}")
    os.rename(path, quarantine)
    return quarantine


def load_nodes(path: Path | None = None) -> list[dict[str, Any]]:
    """LENIENT loader — returns rows; malformed lines silently skipped."""
    p = path or LATTICE_STATE_LEDGER_PATH
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_nodes_strict(path: Path | None = None) -> list[dict[str, Any]]:
    """STRICT loader for mutating callers — raises on corrupt state.

    MUST be called from inside ``_ledger_lock`` by mutating callers per
    Catalog #138 strict-load discipline.
    """
    p = path or LATTICE_STATE_LEDGER_PATH
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise LatticeStateLedgerCorruptError(
            f"lattice-state ledger at {p} could not be read: {exc}"
        ) from exc
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise LatticeStateLedgerCorruptError(
                f"lattice-state ledger at {p} line {lineno} contains invalid "
                f"JSON: {exc}. Mutating writes are refused to preserve the "
                "audit trail."
            ) from exc
        if not isinstance(row, dict):
            raise LatticeStateLedgerCorruptError(
                f"lattice-state ledger at {p} line {lineno} has non-dict root "
                f"(type={type(row).__name__})."
            )
        rows.append(row)
    return rows


def _validate_event_record(record: dict[str, Any]) -> None:
    lattice_node_id = record.get("lattice_node_id")
    if not isinstance(lattice_node_id, str) or not lattice_node_id.strip():
        raise ValueError("lattice_node_id must be a non-empty string")
    if any(c in lattice_node_id for c in ("\n", "\t", "\x1f")):
        raise ValueError("lattice_node_id must not contain newlines/tabs/0x1f")

    substrate = record.get("substrate")
    if not isinstance(substrate, str) or not substrate.strip():
        raise ValueError("substrate must be a non-empty string")

    event_type = record.get("event_type")
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(
            f"event_type must be one of {sorted(VALID_EVENT_TYPES)!r}, got {event_type!r}"
        )

    lattice_rule = record.get("lattice_rule")
    if lattice_rule not in VALID_LATTICE_RULES:
        raise ValueError(
            f"lattice_rule must be one of {sorted(VALID_LATTICE_RULES)!r}, got {lattice_rule!r}"
        )

    horizon_class = record.get("horizon_class")
    if horizon_class not in VALID_HORIZON_CLASSES:
        raise ValueError(
            f"horizon_class must be one of {sorted(VALID_HORIZON_CLASSES)!r}, got {horizon_class!r}"
        )

    status = record.get("status")
    if status not in VALID_STATUSES:
        raise ValueError(
            f"status must be one of {sorted(VALID_STATUSES)!r}, got {status!r}"
        )

    classification = record.get("paradigm_vs_implementation_classification")
    if classification not in VALID_CLASSIFICATIONS:
        raise ValueError(
            f"paradigm_vs_implementation_classification must be one of "
            f"{sorted(VALID_CLASSIFICATIONS)!r}, got {classification!r}"
        )

    architectural_class = record.get("architectural_class")
    if not isinstance(architectural_class, str) or not architectural_class.strip():
        raise ValueError("architectural_class must be a non-empty string")

    schema_version = record.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION}, got {schema_version!r}"
        )


def _save_ledger(rows: list[dict[str, Any]], path: Path | None = None) -> None:
    """Atomic write — unique tmp + fsync + os.replace.

    Runtime-asserts the caller holds ``_ledger_lock`` (Catalog #140).
    """
    if not _ledger_lock_held():
        raise RuntimeError(
            "_save_ledger called WITHOUT holding _ledger_lock. This is a "
            "CONCURRENCY BUG: concurrent appends can silently drop rows."
        )
    p = path or LATTICE_STATE_LEDGER_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    tmp = p.with_suffix(p.suffix + f".tmp.{uuid.uuid4().hex[:12]}")
    try:
        tmp.write_text(payload, encoding="utf-8")
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _append_event_locked(
    record: dict[str, Any],
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    quarantine_on_corrupt: bool = True,
) -> dict[str, Any]:
    """Append a single event record under fcntl lock (load-validate-append-save)."""
    _validate_event_record(record)
    p_path = path or LATTICE_STATE_LEDGER_PATH
    l_path = lock_path or LATTICE_STATE_LEDGER_LOCK

    with _ledger_lock(l_path):
        try:
            rows = load_nodes_strict(p_path)
        except LatticeStateLedgerCorruptError as exc:
            if quarantine_on_corrupt:
                quarantine_path = _quarantine_corrupt_file(p_path)
                raise LatticeStateLedgerCorruptError(
                    f"lattice-state ledger at {p_path} was corrupt; "
                    f"quarantined to {quarantine_path}. Append refused."
                ) from exc
            raise

        new_rows = [*rows, record]
        _save_ledger(new_rows, p_path)
        return record


def _compute_expires_at_utc(
    registered_at_utc: str,
    *,
    staleness_window_days: int = DEFAULT_STALENESS_WINDOW_DAYS,
) -> str:
    try:
        registered = _dt.datetime.fromisoformat(registered_at_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"registered_at_utc must be ISO-8601: {registered_at_utc!r} ({exc})"
        ) from exc
    expires = registered + _dt.timedelta(days=staleness_window_days)
    return expires.isoformat(timespec="microseconds").replace("+00:00", "Z")


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────


def register_lattice_node(
    *,
    lattice_node_id: str,
    substrate: str,
    lattice_rule: str,
    horizon_class: str,
    architectural_class: str,
    status: str,
    paradigm_vs_implementation_classification: str = CLASSIFICATION_TBD,
    recipe_path: str | None = None,
    trainer_path: str | None = None,
    evidence_score: float | None = None,
    evidence_score_axis: str | None = None,
    evidence_artifact_path: str | None = None,
    lane_id: str | None = None,
    registered_at_utc: str | None = None,
    staleness_window_days: int = DEFAULT_STALENESS_WINDOW_DAYS,
    agent: str = "claude",
    subagent_id: str | None = None,
    session_id: str | None = None,
    notes: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Append a ``registered`` event row for a lattice coordinate assignment.

    Returns the appended record including server-side fields
    (``written_at_utc`` / ``expires_at_utc`` / ``written_pid`` /
    ``written_host``).
    """
    if not isinstance(lattice_node_id, str) or not lattice_node_id.strip():
        raise ValueError("lattice_node_id must be a non-empty string")
    if not isinstance(substrate, str) or not substrate.strip():
        raise ValueError("substrate must be a non-empty string")
    if lattice_rule not in VALID_LATTICE_RULES:
        raise ValueError(
            f"lattice_rule must be one of {sorted(VALID_LATTICE_RULES)!r}, got {lattice_rule!r}"
        )
    if horizon_class not in VALID_HORIZON_CLASSES:
        raise ValueError(
            f"horizon_class must be one of {sorted(VALID_HORIZON_CLASSES)!r}, got {horizon_class!r}"
        )
    if status not in VALID_STATUSES:
        raise ValueError(
            f"status must be one of {sorted(VALID_STATUSES)!r}, got {status!r}"
        )
    if paradigm_vs_implementation_classification not in VALID_CLASSIFICATIONS:
        raise ValueError(
            f"paradigm_vs_implementation_classification must be one of "
            f"{sorted(VALID_CLASSIFICATIONS)!r}, got "
            f"{paradigm_vs_implementation_classification!r}"
        )

    resolved_registered_at = registered_at_utc or _now_iso()
    expires_at_utc = _compute_expires_at_utc(
        resolved_registered_at,
        staleness_window_days=staleness_window_days,
    )

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_type": EVENT_REGISTERED,
        "lattice_node_id": lattice_node_id,
        "substrate": substrate,
        "recipe_path": recipe_path,
        "trainer_path": trainer_path,
        "lattice_rule": lattice_rule,
        "horizon_class": horizon_class,
        "architectural_class": architectural_class,
        "status": status,
        "paradigm_vs_implementation_classification": paradigm_vs_implementation_classification,
        "evidence_score": float(evidence_score) if evidence_score is not None else None,
        "evidence_score_axis": evidence_score_axis,
        "evidence_artifact_path": evidence_artifact_path,
        "lane_id": lane_id,
        "registered_at_utc": resolved_registered_at,
        "expires_at_utc": expires_at_utc,
        "staleness_window_days": staleness_window_days,
        "agent": agent,
        "subagent_id": subagent_id,
        "session_id": session_id,
        "notes": notes,
        "written_at_utc": _now_iso(),
        "written_pid": os.getpid(),
        "written_host": socket.gethostname(),
    }
    reserved = set(record.keys())
    for k, v in extra.items():
        if k in reserved:
            raise ValueError(f"extra kwarg {k!r} collides with a reserved schema field")
        record[k] = v
    return _append_event_locked(record, path=path, lock_path=lock_path)


def update_lattice_node(
    *,
    lattice_node_id: str,
    event_type: str,
    horizon_class: str | None = None,
    status: str | None = None,
    paradigm_vs_implementation_classification: str | None = None,
    evidence_score: float | None = None,
    evidence_score_axis: str | None = None,
    evidence_artifact_path: str | None = None,
    notes: str | None = None,
    agent: str = "claude",
    subagent_id: str | None = None,
    session_id: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Append an update event row inheriting fields not explicitly overridden.

    Per CLAUDE.md HISTORICAL_PROVENANCE + Catalog #110/#132 — updates are
    NEW rows referencing the same lattice_node_id, NEVER mutations.
    """
    if not isinstance(lattice_node_id, str) or not lattice_node_id.strip():
        raise ValueError("lattice_node_id must be a non-empty string")
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(
            f"event_type must be one of {sorted(VALID_EVENT_TYPES)!r}, got {event_type!r}"
        )

    existing = latest_node_state(lattice_node_id, path=path)
    if existing is None:
        raise ValueError(
            f"lattice_node_id {lattice_node_id!r} has no prior registered event; "
            "call register_lattice_node() first"
        )

    resolved_horizon_class = horizon_class if horizon_class is not None else existing.get("horizon_class")
    if resolved_horizon_class not in VALID_HORIZON_CLASSES:
        raise ValueError(
            f"horizon_class must be one of {sorted(VALID_HORIZON_CLASSES)!r}, got {resolved_horizon_class!r}"
        )
    resolved_status = status if status is not None else existing.get("status")
    if resolved_status not in VALID_STATUSES:
        raise ValueError(
            f"status must be one of {sorted(VALID_STATUSES)!r}, got {resolved_status!r}"
        )
    resolved_classification = (
        paradigm_vs_implementation_classification
        if paradigm_vs_implementation_classification is not None
        else existing.get("paradigm_vs_implementation_classification")
    )
    if resolved_classification not in VALID_CLASSIFICATIONS:
        raise ValueError(
            f"paradigm_vs_implementation_classification must be one of "
            f"{sorted(VALID_CLASSIFICATIONS)!r}, got {resolved_classification!r}"
        )

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_type": event_type,
        "lattice_node_id": lattice_node_id,
        "substrate": existing.get("substrate"),
        "recipe_path": existing.get("recipe_path"),
        "trainer_path": existing.get("trainer_path"),
        "lattice_rule": existing.get("lattice_rule"),
        "horizon_class": resolved_horizon_class,
        "architectural_class": existing.get("architectural_class"),
        "status": resolved_status,
        "paradigm_vs_implementation_classification": resolved_classification,
        "evidence_score": (
            float(evidence_score) if evidence_score is not None else existing.get("evidence_score")
        ),
        "evidence_score_axis": (
            evidence_score_axis if evidence_score_axis is not None else existing.get("evidence_score_axis")
        ),
        "evidence_artifact_path": (
            evidence_artifact_path if evidence_artifact_path is not None else existing.get("evidence_artifact_path")
        ),
        "lane_id": existing.get("lane_id"),
        "registered_at_utc": existing.get("registered_at_utc"),
        "expires_at_utc": existing.get("expires_at_utc"),
        "staleness_window_days": existing.get("staleness_window_days"),
        "agent": agent,
        "subagent_id": subagent_id,
        "session_id": session_id,
        "notes": notes,
        "written_at_utc": _now_iso(),
        "written_pid": os.getpid(),
        "written_host": socket.gethostname(),
    }
    reserved = set(record.keys())
    for k, v in extra.items():
        if k in reserved:
            raise ValueError(f"extra kwarg {k!r} collides with a reserved schema field")
        record[k] = v
    return _append_event_locked(record, path=path, lock_path=lock_path)


# ─────────────────────────────────────────────────────────────────────────
# Query helpers
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LatticeCoverageReport:
    """Summary of lattice-rule coverage + outside-NeRV count."""

    total_nodes: int
    rule_counts: dict[str, int]
    uncovered_rules: list[str]
    nerv_family_count: int
    outside_nerv_count: int
    horizon_counts: dict[str, int]
    architectural_class_counts: dict[str, int]
    in_flight_count: int  # status in {lifted_dispatch_ready, lifted_pending_council}
    dispatched_count: int  # status == dispatched_evidence_landed
    deferred_count: int  # status in {deferred_*}


def latest_node_state(
    lattice_node_id: str,
    *,
    path: Path | None = None,
) -> dict[str, Any] | None:
    """Return the latest (chronologically) row for ``lattice_node_id``."""
    rows = load_nodes(path)
    matching = [r for r in rows if r.get("lattice_node_id") == lattice_node_id]
    if not matching:
        return None
    return max(matching, key=lambda r: r.get("written_at_utc", ""))


def query_by_substrate(
    substrate: str,
    *,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return all events (chronological) for a given substrate."""
    rows = load_nodes(path)
    matching = [r for r in rows if r.get("substrate") == substrate]
    return sorted(matching, key=lambda r: r.get("written_at_utc", ""))


def query_by_rule(
    lattice_rule: str,
    *,
    path: Path | None = None,
    latest_only: bool = True,
) -> list[dict[str, Any]]:
    """Return all events (or latest-per-node) matching a lattice_rule."""
    rows = load_nodes(path)
    matching = [r for r in rows if r.get("lattice_rule") == lattice_rule]
    if not latest_only:
        return sorted(matching, key=lambda r: r.get("written_at_utc", ""))
    # Reduce to latest per lattice_node_id
    by_node: dict[str, dict[str, Any]] = {}
    for r in matching:
        nid = r.get("lattice_node_id")
        if nid is None:
            continue
        existing = by_node.get(nid)
        if existing is None or r.get("written_at_utc", "") > existing.get("written_at_utc", ""):
            by_node[nid] = r
    return sorted(by_node.values(), key=lambda r: r.get("lattice_node_id", ""))


def query_by_architectural_class(
    architectural_class: str,
    *,
    path: Path | None = None,
    latest_only: bool = True,
) -> list[dict[str, Any]]:
    """Return all events (or latest-per-node) matching an architectural_class."""
    rows = load_nodes(path)
    matching = [r for r in rows if r.get("architectural_class") == architectural_class]
    if not latest_only:
        return sorted(matching, key=lambda r: r.get("written_at_utc", ""))
    by_node: dict[str, dict[str, Any]] = {}
    for r in matching:
        nid = r.get("lattice_node_id")
        if nid is None:
            continue
        existing = by_node.get(nid)
        if existing is None or r.get("written_at_utc", "") > existing.get("written_at_utc", ""):
            by_node[nid] = r
    return sorted(by_node.values(), key=lambda r: r.get("lattice_node_id", ""))


def query_outside_nerv_family(
    *,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return latest-state-per-node for all OUTSIDE-NeRV-family substrates.

    Per operator binding constraint 2026-05-16 *"Remember we need outside
    nerv-family too"*: any K-measurement Wave-N plan must include at least
    3-4 outside-NeRV frontier-pursuit substrates.
    """
    rows = load_nodes(path)
    by_node: dict[str, dict[str, Any]] = {}
    for r in rows:
        nid = r.get("lattice_node_id")
        if nid is None:
            continue
        existing = by_node.get(nid)
        if existing is None or r.get("written_at_utc", "") > existing.get("written_at_utc", ""):
            by_node[nid] = r
    return sorted(
        [r for r in by_node.values() if r.get("architectural_class") not in NERV_FAMILY_ARCHITECTURAL_CLASSES],
        key=lambda r: r.get("lattice_node_id", ""),
    )


def query_uncovered_rules(*, path: Path | None = None) -> list[str]:
    """Return canonical rules with no in-flight or dispatched coverage."""
    rows = load_nodes(path)
    # Reduce to latest per node
    by_node: dict[str, dict[str, Any]] = {}
    for r in rows:
        nid = r.get("lattice_node_id")
        if nid is None:
            continue
        existing = by_node.get(nid)
        if existing is None or r.get("written_at_utc", "") > existing.get("written_at_utc", ""):
            by_node[nid] = r
    covered: set[str] = set()
    active_statuses = {
        STATUS_LIFTED_DISPATCH_READY,
        STATUS_LIFTED_PENDING_COUNCIL,
        STATUS_DISPATCHED_EVIDENCE,
    }
    for r in by_node.values():
        if r.get("status") in active_statuses:
            rule = r.get("lattice_rule")
            if rule:
                covered.add(rule)
    return sorted(VALID_LATTICE_RULES - covered)


def compute_coverage_report(*, path: Path | None = None) -> LatticeCoverageReport:
    """Compute the canonical coverage + outside-NeRV summary."""
    rows = load_nodes(path)
    by_node: dict[str, dict[str, Any]] = {}
    for r in rows:
        nid = r.get("lattice_node_id")
        if nid is None:
            continue
        existing = by_node.get(nid)
        if existing is None or r.get("written_at_utc", "") > existing.get("written_at_utc", ""):
            by_node[nid] = r

    rule_counts: dict[str, int] = {}
    horizon_counts: dict[str, int] = {}
    arch_class_counts: dict[str, int] = {}
    nerv_count = 0
    outside_count = 0
    in_flight = 0
    dispatched = 0
    deferred = 0

    for r in by_node.values():
        rule = r.get("lattice_rule")
        if rule:
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
        horizon = r.get("horizon_class")
        if horizon:
            horizon_counts[horizon] = horizon_counts.get(horizon, 0) + 1
        arch = r.get("architectural_class")
        if arch:
            arch_class_counts[arch] = arch_class_counts.get(arch, 0) + 1
            if arch in NERV_FAMILY_ARCHITECTURAL_CLASSES:
                nerv_count += 1
            else:
                outside_count += 1
        status = r.get("status")
        if status in {STATUS_LIFTED_DISPATCH_READY, STATUS_LIFTED_PENDING_COUNCIL}:
            in_flight += 1
        elif status == STATUS_DISPATCHED_EVIDENCE:
            dispatched += 1
        elif status in {STATUS_DEFERRED_PROBE, STATUS_DEFERRED_OPERATOR, STATUS_DEFERRED_AUDIT}:
            deferred += 1

    return LatticeCoverageReport(
        total_nodes=len(by_node),
        rule_counts=rule_counts,
        uncovered_rules=query_uncovered_rules(path=path),
        nerv_family_count=nerv_count,
        outside_nerv_count=outside_count,
        horizon_counts=horizon_counts,
        architectural_class_counts=arch_class_counts,
        in_flight_count=in_flight,
        dispatched_count=dispatched,
        deferred_count=deferred,
    )


__all__ = [
    "LATTICE_STATE_LEDGER_PATH",
    "LATTICE_STATE_LEDGER_LOCK",
    "SCHEMA_VERSION",
    "LOCK_TIMEOUT_SECONDS",
    "DEFAULT_STALENESS_WINDOW_DAYS",
    "VALID_EVENT_TYPES",
    "VALID_LATTICE_RULES",
    "VALID_HORIZON_CLASSES",
    "VALID_STATUSES",
    "VALID_CLASSIFICATIONS",
    "NERV_FAMILY_ARCHITECTURAL_CLASSES",
    "RULE_1",
    "RULE_2",
    "RULE_3",
    "RULE_4",
    "RULE_5",
    "HORIZON_PLATEAU_ADJACENT",
    "HORIZON_FRONTIER_PURSUIT",
    "HORIZON_ASYMPTOTIC_PURSUIT",
    "HORIZON_WON",
    "HORIZON_NA",
    "STATUS_LIFTED_DISPATCH_READY",
    "STATUS_LIFTED_PENDING_COUNCIL",
    "STATUS_DEFERRED_PROBE",
    "STATUS_DEFERRED_OPERATOR",
    "STATUS_DEFERRED_AUDIT",
    "STATUS_NOT_YET_LIFTED",
    "STATUS_DISPATCHED_EVIDENCE",
    "STATUS_SCAFFOLD_L0",
    "STATUS_KILLED",
    "STATUS_ARCHIVED",
    "CLASSIFICATION_PARADIGM_INTACT",
    "CLASSIFICATION_IMPLEMENTATION_CARGO_CULT",
    "CLASSIFICATION_IMPLEMENTATION_FALSIFIED",
    "CLASSIFICATION_PARADIGM_FALSIFIED",
    "CLASSIFICATION_TBD",
    "EVENT_REGISTERED",
    "EVENT_RECLASSIFIED",
    "EVENT_PROMOTED",
    "EVENT_DEFERRED",
    "EVENT_REACTIVATED",
    "EVENT_OPERATOR_OVERRIDE",
    "LatticeStateLedgerCorruptError",
    "LatticeCoverageReport",
    "load_nodes",
    "load_nodes_strict",
    "register_lattice_node",
    "update_lattice_node",
    "latest_node_state",
    "query_by_substrate",
    "query_by_rule",
    "query_by_architectural_class",
    "query_outside_nerv_family",
    "query_uncovered_rules",
    "compute_coverage_report",
]
