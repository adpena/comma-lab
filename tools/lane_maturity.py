#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Lane Maturity Harness — canonical CLI for the lane registry.

Tracks every lane's level (0/1/2/3) status against the 8-gate Level-3
checklist defined in
`feedback_production_hardened_standard_definition_20260430.md`:

  1. impl_complete            — production code lands
  2. real_archive_empirical   — real-archive empirical measurement
  3. contest_cuda             — [contest-CUDA] score, NEVER MPS, NEVER advisory
  4. contest_cpu              — [contest-CPU] Linux x86_64 public-axis replay
  5. strict_preflight         — STRICT preflight check covers the bug class
  6. three_clean_review       — 3-clean-pass adversarial review counter @ 3/3
  7. memory_entry             — memory file with empirical result + cross-refs
  8. deploy_runbook           — remote_lane script + heartbeat + watchdog +
                                 harvest path

Computed level rules (anything that disagrees with these is a CLI bug):

  level 0 — 0 gates satisfied (SKETCH)
  level 1 — 1+ gate satisfied  (SCAFFOLD)
  level 2 — impl_complete AND real_archive_empirical satisfied (INTEGRATION)
  level 3 — ALL 8 gates satisfied  (FULL PRODUCTION HARDENED + RECURSIVE
            ADVERSARIAL REVIEWED)

NOTE on the level-2-vs-1 tie-break: a lane that has, say, 4 gates satisfied
but not impl_complete OR not real_archive_empirical is STILL only level 1 —
because Level 2 specifically requires those two gates. The audit gate-count
is informational; the rule above is binding.

Subcommands
───────────
  audit                  — colored table to stdout (default if no args).
  mark <id> --gate G --evidence E
                         — set gate G of lane <id> to true with evidence E.
                           Errors out if E looks like a path (contains '/')
                           but does not exist on disk.
  unmark <id> --gate G --reason R
                         — revert gate G to false; reason logged in audit.
  validate               — exit nonzero if registry is inconsistent.
  report                 — write reports/lane_maturity.md.
  add-lane <id> --name N --phase P
                         — register a new lane at level 0.

Every mutation appends a JSONL record to .omx/state/lane_maturity_audit.log.

Cooperators
───────────
This CLI MUST be the only mutator of .omx/state/lane_registry.json. Bare
hand-edits land a lane at potentially-wrong level + skip the audit log.
Preflight Check 90 (`check_lane_registry_consistent`) verifies the registry
is internally consistent at every commit.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback is best-effort.
    fcntl = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent

REGISTRY_REL = ".omx/state/lane_registry.json"
AUDIT_LOG_REL = ".omx/state/lane_maturity_audit.log"
LOCK_REL = ".omx/state/lane_maturity.lock"
REPORT_REL = "reports/lane_maturity.md"

EXPECTED_SCHEMA_VERSION = 1
CONTEST_CPU_GATE_DEFINITION = (
    "[contest-CPU] Linux x86_64 public-axis replay for the same archive/runtime; "
    "not promotion authority and never macOS/MPS advisory."
)

# The 8 gates, in fixed display order.
GATES = [
    "impl_complete",
    "real_archive_empirical",
    "contest_cuda",
    "contest_cpu",
    "strict_preflight",
    "three_clean_review",
    "memory_entry",
    "deploy_runbook",
]

# Gates that — if EITHER is unsatisfied — disqualify a lane from Level 2 even
# if 4+ gates are otherwise true. See module docstring "level-2-vs-1 tie-break".
LEVEL_2_REQUIRED_GATES = {"impl_complete", "real_archive_empirical"}

# Heuristic: an "evidence" string LOOKS LIKE a file path if it contains a
# slash AND starts with one of these prefixes.
_FILE_PATH_PREFIXES = (
    "/", "src/", "tests/", "scripts/", ".omx/", "reports/", "memory/",
    "experiments/", "tools/", "submissions/", "docs/", "configs/",
    "src/tac/", "upstream/",
)


# ── Time / IO helpers ────────────────────────────────────────────────────


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _registry_path(repo_root: Path | None = None) -> Path:
    return (repo_root or REPO_ROOT) / REGISTRY_REL


def _audit_log_path(repo_root: Path | None = None) -> Path:
    return (repo_root or REPO_ROOT) / AUDIT_LOG_REL


def _lock_path(repo_root: Path | None = None) -> Path:
    return (repo_root or REPO_ROOT) / LOCK_REL


def _report_path(repo_root: Path | None = None) -> Path:
    return (repo_root or REPO_ROOT) / REPORT_REL


@contextmanager
def _mutation_lock(repo_root: Path | None = None):
    """Serialize lane-registry load/modify/save/audit transactions."""

    path = _lock_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_registry(repo_root: Path | None = None) -> dict[str, Any]:
    """Load + parse the lane registry. Raises ValueError on schema mismatch."""
    path = _registry_path(repo_root)
    if not path.exists():
        raise FileNotFoundError(f"lane registry missing: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"registry must be a JSON object, got {type(data).__name__}")
    sv = data.get("schema_version")
    if sv != EXPECTED_SCHEMA_VERSION:
        raise ValueError(
            f"registry schema_version mismatch: expected {EXPECTED_SCHEMA_VERSION}, "
            f"got {sv!r}"
        )
    if "lanes" not in data or not isinstance(data["lanes"], list):
        raise ValueError("registry missing 'lanes' list")
    _normalize_registry_gates(data)
    return data


def _normalize_registry_gates(data: dict[str, Any]) -> None:
    """Migrate older registries to the current gate set in memory.

    `.omx/state/lane_registry.json` is local custody state and may predate a
    stricter gate list. Normalizing here keeps operator tools fail-closed under
    the newest standard without turning every historical registry into a manual
    rescue task. Newly added gates are always false until explicit evidence is
    recorded.
    """

    gate_definitions = data.setdefault("gate_definitions", {})
    if isinstance(gate_definitions, dict):
        ordered_defs: dict[str, Any] = {}
        for gate in GATES:
            if gate == "contest_cpu":
                ordered_defs[gate] = gate_definitions.get(gate) or CONTEST_CPU_GATE_DEFINITION
            elif gate in gate_definitions:
                ordered_defs[gate] = gate_definitions[gate]
        for gate, definition in gate_definitions.items():
            if gate not in ordered_defs:
                ordered_defs[gate] = definition
        data["gate_definitions"] = ordered_defs

    level_rules = data.setdefault("level_rules", {})
    if isinstance(level_rules, dict):
        level_rules["3"] = (
            "ALL 8 gates satisfied — FULL PRODUCTION HARDENED + "
            "RECURSIVE ADVERSARIAL REVIEWED."
        )

    for lane in data.get("lanes", []):
        if not isinstance(lane, dict):
            continue
        gates = lane.setdefault("gates", {})
        if not isinstance(gates, dict):
            continue
        had_contest_cpu = "contest_cpu" in gates
        ordered_gates: dict[str, Any] = {}
        for gate, value in gates.items():
            ordered_gates[gate] = value
            if gate == "contest_cuda" and not had_contest_cpu:
                ordered_gates["contest_cpu"] = {"status": False, "evidence": ""}
        if not had_contest_cpu and "contest_cpu" not in ordered_gates:
            ordered_gates["contest_cpu"] = {"status": False, "evidence": ""}
        if not had_contest_cpu:
            lane["gates"] = ordered_gates
            lane["level"] = compute_level(lane["gates"])


def save_registry(data: dict[str, Any], repo_root: Path | None = None) -> None:
    path = _registry_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = _now_iso()
    payload = json.dumps(data, indent=2, sort_keys=False) + "\n"
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{id(data)}.tmp")
    with tmp_path.open("w") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def append_audit_log(record: dict[str, Any], repo_root: Path | None = None) -> None:
    path = _audit_log_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ── Level computation ────────────────────────────────────────────────────


def looks_like_filepath(evidence: str) -> bool:
    """Heuristic: evidence string LOOKS LIKE a file path."""
    if not evidence or not isinstance(evidence, str):
        return False
    # Strip leading bracketed tags like "[empirical:..." or "[contest-CUDA] ..."
    # The actual path inside brackets is colon-separated; we match the prefix
    # heuristic on the FULL string.
    s = evidence.strip()
    # If it doesn't contain a '/' it can't be a real path.
    if "/" not in s:
        return False
    # The whole string need not be a path; we look at the first whitespace-
    # delimited token.
    first_token = s.split()[0].rstrip(",.;)")
    # Strip a leading "[tag:" prefix if present
    if "[" in first_token and ":" in first_token:
        # e.g. "[empirical:reports/..." → reports/...
        idx = first_token.index(":")
        first_token = first_token[idx + 1:].rstrip("]")
    return any(first_token.startswith(p) for p in _FILE_PATH_PREFIXES)


def extract_filepath(evidence: str) -> str | None:
    """Best-effort: extract the file path from an evidence string.

    Returns the path token, or None if no path-like substring found.
    """
    if not looks_like_filepath(evidence):
        return None
    s = evidence.strip()
    first_token = s.split()[0].rstrip(",.;)")
    if "[" in first_token and ":" in first_token:
        idx = first_token.index(":")
        first_token = first_token[idx + 1:].rstrip("]")
    return first_token


def compute_level(gates: dict[str, dict[str, Any]]) -> int:
    """Compute the level from gate status (0/1/2/3).

    Rules (binding):
      - Level 3: ALL 8 gates true.
      - Level 2: impl_complete AND real_archive_empirical true.
      - Level 1: at least 1 gate true.
      - Level 0: 0 gates true.

    A lane that has 4 gates true but is missing impl_complete is LEVEL 1,
    not Level 2, because Level 2 requires those specific gates.
    """
    n_true = sum(1 for g in GATES if gates.get(g, {}).get("status") is True)
    if n_true == len(GATES):
        return 3
    impl = bool(gates.get("impl_complete", {}).get("status"))
    emp = bool(gates.get("real_archive_empirical", {}).get("status"))
    if impl and emp:
        return 2
    if n_true >= 1:
        return 1
    return 0


def _evidence_exact_readiness_refusal(
    evidence: str,
    *,
    repo_root: Path,
) -> dict[str, Any] | None:
    path_text = extract_filepath(evidence)
    if path_text is None and evidence:
        path_text = evidence.split()[0]
    if not path_text:
        return None
    path = Path(path_text)
    candidate = path if path.is_absolute() else repo_root / path
    if candidate.suffix != ".json" or not candidate.is_file():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    refusal = payload.get("exact_readiness_refusal")
    return refusal if isinstance(refusal, dict) else None


# ── Validation ───────────────────────────────────────────────────────────


def validate_registry(
    data: dict[str, Any], repo_root: Path | None = None,
) -> list[str]:
    """Return list of validation errors. Empty = valid."""
    errors: list[str] = []
    root = repo_root or REPO_ROOT

    seen_ids: set[str] = set()
    for lane in data.get("lanes", []):
        lid = lane.get("id")
        if not lid:
            errors.append("lane has no 'id'")
            continue
        if lid in seen_ids:
            errors.append(f"duplicate lane id: {lid}")
        seen_ids.add(lid)

        gates = lane.get("gates", {})
        if not isinstance(gates, dict):
            errors.append(f"{lid}: 'gates' must be a dict")
            continue

        # Every gate must be present (even if false)
        for gname in GATES:
            if gname not in gates:
                errors.append(f"{lid}: gate '{gname}' missing")
                continue
            g = gates[gname]
            if not isinstance(g, dict) or "status" not in g or "evidence" not in g:
                errors.append(
                    f"{lid}: gate '{gname}' must be {{status, evidence}}, "
                    f"got {g!r}"
                )
                continue
            if not isinstance(g["status"], bool):
                errors.append(
                    f"{lid}: gate '{gname}' status must be bool, "
                    f"got {g['status']!r}"
                )

        # Computed level must match stored level
        stored = lane.get("level")
        computed = compute_level(gates)
        if stored != computed:
            errors.append(
                f"{lid}: stored level {stored!r} disagrees with computed "
                f"level {computed} (gate count: "
                f"{sum(1 for g in GATES if gates.get(g, {}).get('status') is True)}/{len(GATES)})"
            )
        real_archive_gate = gates.get("real_archive_empirical", {})
        if (
            isinstance(stored, int)
            and stored >= 2
            and isinstance(real_archive_gate, dict)
            and real_archive_gate.get("status") is True
        ):
            refusal = _evidence_exact_readiness_refusal(
                str(real_archive_gate.get("evidence") or ""),
                repo_root=root,
            )
            if refusal is not None and refusal.get("ready") is False:
                errors.append(
                    f"{lid}: L2+ real_archive_empirical evidence has "
                    "exact_readiness_refusal.ready=false; unmark "
                    "real_archive_empirical or use a non-authority packaging "
                    "ledger instead"
                )

        # Evidence path heuristic — if it LOOKS like a path, it MUST exist
        for gname, g in gates.items():
            if not isinstance(g, dict):
                continue
            if g.get("status") is not True:
                # Don't validate paths on unsatisfied gates (evidence often "")
                continue
            ev = g.get("evidence", "")
            if not ev:
                errors.append(
                    f"{lid}: gate '{gname}' status=true but evidence is empty"
                )
                continue
            path = extract_filepath(ev)
            if path is None:
                # Evidence is descriptive text only — that's OK.
                continue
            if not (root / path).exists():
                errors.append(
                    f"{lid}: gate '{gname}' evidence path does not exist: {path}"
                )

    return errors


# ── Mutations (mark / unmark / add-lane) ─────────────────────────────────


def mark_gate(
    data: dict[str, Any],
    lane_id: str,
    gate: str,
    evidence: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Set lane.gates[gate] = {status: true, evidence}; bump computed level.

    Returns the updated lane dict (mutates `data` in place).

    Errors:
      - lane_id not in registry
      - gate not in GATES
      - evidence empty
      - evidence LOOKS LIKE a path but does not exist on disk
    """
    if gate not in GATES:
        raise ValueError(
            f"unknown gate '{gate}'. Valid gates: {', '.join(GATES)}"
        )
    if not evidence or not evidence.strip():
        raise ValueError("evidence is required and must be non-empty")

    lanes = data["lanes"]
    lane = next((lane_entry for lane_entry in lanes if lane_entry["id"] == lane_id), None)
    if lane is None:
        raise ValueError(f"unknown lane id: {lane_id!r}")

    path = extract_filepath(evidence)
    if path is not None:
        root = repo_root or REPO_ROOT
        if not (root / path).exists():
            raise ValueError(
                f"evidence path does not exist on disk: {path} "
                f"(if this is descriptive text not a path, rephrase to "
                f"avoid leading '{path.split('/')[0]}/')"
            )

    lane.setdefault("gates", {})[gate] = {"status": True, "evidence": evidence}
    lane["level"] = compute_level(lane["gates"])
    return lane


def unmark_gate(
    data: dict[str, Any],
    lane_id: str,
    gate: str,
    reason: str,
) -> dict[str, Any]:
    """Set lane.gates[gate] = {status: false, evidence: ""}; bump level."""
    if gate not in GATES:
        raise ValueError(
            f"unknown gate '{gate}'. Valid gates: {', '.join(GATES)}"
        )
    lanes = data["lanes"]
    lane = next((lane_entry for lane_entry in lanes if lane_entry["id"] == lane_id), None)
    if lane is None:
        raise ValueError(f"unknown lane id: {lane_id!r}")
    if not reason or not reason.strip():
        raise ValueError("--reason is required for unmark")
    lane.setdefault("gates", {})[gate] = {"status": False, "evidence": ""}
    lane["level"] = compute_level(lane["gates"])
    return lane


def add_lane(
    data: dict[str, Any],
    lane_id: str,
    name: str,
    phase: float | int,
    notes: str = "",
) -> dict[str, Any]:
    """Register a new lane at level 0."""
    lanes = data["lanes"]
    if any(lane_entry["id"] == lane_id for lane_entry in lanes):
        raise ValueError(f"lane id already exists: {lane_id}")
    new_lane = {
        "id": lane_id,
        "name": name,
        "phase": phase,
        "level": 0,
        "gates": {
            g: {"status": False, "evidence": ""} for g in GATES
        },
        "notes": notes,
    }
    lanes.append(new_lane)
    return new_lane


# Top-level field allowlist for `set_field` mutation surface. These keys are
# read by preflight checks (Catalog #124, etc.) and council-policy gates; the
# CLI exposes them so subagents do not need to bare-edit the registry JSON.
#
# Per CLAUDE.md "Lane maturity registry — non-negotiable":
#   "Mutations only via tools/lane_maturity.py mark/unmark/add-lane."
# `set-field` extends that mutation surface for non-gate top-level metadata
# (lane_class / research_only / reactivation_criteria) and for the
# `design_evidence` dict that Check 124 looks for as one of 4 acceptance
# locations for the 8 representation-lane fields.
_SET_FIELD_TOP_LEVEL_ALLOWED = (
    "name",                  # display metadata; avoids bare registry edits
    "notes",                 # display metadata; avoids bare registry edits
    "lane_class",            # Check 124 opt-out: substrate_engineering
    "research_only",         # Check 124 opt-out: research-only by construction
    "reactivation_criteria", # forbidden_premature_kill_without_research_exhaustion
    # Catalog #272 — Distinguishing-Feature Integration Contract.
    # Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
    # non-negotiable. Anchor 2026-05-15: Z3-G1 trained a 1KB SegNet-class
    # CDF (the "smart" distinguishing thing) but the archive's
    # `hyperprior_weights_int8` slot is `b""` — the smart thing was
    # engineered, never wired. Smoke score == Z3 v2 baseline to 5 decimals.
    "distinguishing_feature_name",       # Catalog #272: human-readable novel-contribution name
    "distinguishing_bytes_path",         # Catalog #272: archive section(s) embodying the feature
    "inflate_consumer_function",         # Catalog #272: inflate.py function(s) that read those bytes
    "byte_mutation_smoke_passes",        # Catalog #272: proof artifact path; self-attested bool/string rejected
)
_SET_FIELD_DESIGN_EVIDENCE_ALLOWED = (
    "archive_grammar",
    "parser_section_manifest",
    "inflate_runtime_loc_budget",
    "runtime_dep_closure",
    "export_format",
    "score_aware_loss",
    "bolt_on_loc_budget",
    "no_op_detector_planned",
)


def set_field(
    data: dict[str, Any],
    lane_id: str,
    field: str,
    value: Any,
) -> dict[str, Any]:
    """Set a top-level scalar field OR a `design_evidence` sub-field on a lane.

    `field` may be one of:
      - top-level allowlist: name, notes, lane_class, research_only, reactivation_criteria
      - `design_evidence.<sub>` where <sub> is one of the 8 Check-124 fields

    Errors:
      - lane_id not in registry
      - field not in either allowlist
      - value is None or empty-string (use unset semantics via direct edit if
        ever needed — set-field is for explicit declaration)
    """
    lanes = data["lanes"]
    lane = next((lane_entry for lane_entry in lanes if lane_entry["id"] == lane_id), None)
    if lane is None:
        raise ValueError(f"unknown lane id: {lane_id!r}")

    if value in (None, ""):
        raise ValueError(
            f"value must be non-empty (got {value!r}); "
            "set-field is for explicit declaration"
        )

    if field.startswith("design_evidence."):
        sub = field.split(".", 1)[1]
        if sub not in _SET_FIELD_DESIGN_EVIDENCE_ALLOWED:
            raise ValueError(
                f"unknown design_evidence sub-field {sub!r}. "
                f"Allowed: {', '.join(_SET_FIELD_DESIGN_EVIDENCE_ALLOWED)}"
            )
        de = lane.setdefault("design_evidence", {})
        if not isinstance(de, dict):
            raise ValueError(
                f"lane {lane_id!r}: existing 'design_evidence' is not a dict "
                f"(got {type(de).__name__}); manual rescue needed"
            )
        de[sub] = value
        return lane

    if field not in _SET_FIELD_TOP_LEVEL_ALLOWED:
        raise ValueError(
            f"unknown field {field!r}. Allowed top-level: "
            f"{', '.join(_SET_FIELD_TOP_LEVEL_ALLOWED)}; or "
            f"design_evidence.<sub> where <sub> in "
            f"{', '.join(_SET_FIELD_DESIGN_EVIDENCE_ALLOWED)}"
        )
    lane[field] = value
    return lane


# ── Output (audit / report) ──────────────────────────────────────────────


# ANSI colors (no rich dep needed). Falls back to no-color if not a TTY.
_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _COLOR else s
def _green(s: str) -> str: return _c("32", s)
def _yellow(s: str) -> str: return _c("33", s)
def _red(s: str) -> str: return _c("31", s)
def _bold(s: str) -> str: return _c("1", s)
def _dim(s: str) -> str: return _c("2", s)


_LEVEL_COLOR = {
    0: _red,
    1: _yellow,
    2: _yellow,
    3: _green,
}


def render_audit_table(data: dict[str, Any]) -> str:
    """Render colored audit table to a string."""
    lines: list[str] = []
    lanes = data.get("lanes", [])

    # Group by phase
    phases: dict[float | int, list[dict[str, Any]]] = {}
    for lane in lanes:
        phases.setdefault(lane.get("phase", "?"), []).append(lane)

    # Header
    lines.append(_bold("LANE MATURITY AUDIT") + f" — {data.get('updated_at', '?')}")
    lines.append("")

    # Summary
    by_level = {0: 0, 1: 0, 2: 0, 3: 0}
    for lane in lanes:
        by_level[lane.get("level", 0)] += 1
    summary = (
        f"{_bold('Total lanes:')} {len(lanes)}  "
        f"{_green('L3=')}" + str(by_level[3]) + "  "
        f"{_yellow('L2=')}" + str(by_level[2]) + "  "
        f"{_yellow('L1=')}" + str(by_level[1]) + "  "
        f"{_red('L0=')}" + str(by_level[0])
    )
    lines.append(summary)
    lines.append("")

    # Per-phase tables
    for phase in sorted(phases.keys(), key=lambda x: float(x) if isinstance(x, (int, float)) else 99):
        plist = phases[phase]
        lines.append(_bold(f"── PHASE {phase} ──"))
        # Header row
        gate_short = {
            "impl_complete": "impl",
            "real_archive_empirical": "emp",
            "contest_cuda": "cuda",
            "contest_cpu": "cpu",
            "strict_preflight": "preflt",
            "three_clean_review": "3clean",
            "memory_entry": "mem",
            "deploy_runbook": "deploy",
        }
        hdr = f"  {'lvl':<3} {'lane id':<35} "
        for g in GATES:
            hdr += f"{gate_short[g]:>6} "
        lines.append(_dim(hdr))
        for lane in plist:
            lvl = lane.get("level", 0)
            lvl_str = _LEVEL_COLOR[lvl](f"L{lvl}")
            gates = lane.get("gates", {})
            row = f"  {lvl_str:<3} {lane['id']:<35} "
            for g in GATES:
                ok = gates.get(g, {}).get("status") is True
                mark = _green("✓") if ok else _red("✗")
                row += f"{mark:>6} "
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


def render_report_md(data: dict[str, Any]) -> str:
    """Render Markdown report (auto-generated; safe to commit)."""
    lanes = data.get("lanes", [])
    by_level = {0: 0, 1: 0, 2: 0, 3: 0}
    for lane in lanes:
        by_level[lane.get("level", 0)] += 1

    # DerivedOutputGuard regen header (Catalog #113 META gate). Must appear in
    # the first 4KB. The HTML-comment form is recognized by the regex
    # `(generated_at|generated_utc).*?(from_state_hash|source_state_hash|input_sha)`.
    import hashlib as _hashlib
    state_payload = json.dumps(data, sort_keys=True).encode("utf-8")
    from_state_hash = _hashlib.sha256(state_payload).hexdigest()
    generated_at = data.get("updated_at", "")

    lines: list[str] = []
    lines.append(
        f"<!-- generated_at: {generated_at} from_state_hash: {from_state_hash} "
        f"regenerated_by: tools/lane_maturity.py report -->"
    )
    lines.append("")
    lines.append("# Lane Maturity Report")
    lines.append("")
    lines.append(
        "*Auto-generated by `tools/lane_maturity.py report`. "
        "Do not hand-edit — re-run the command to refresh.*"
    )
    lines.append("")
    lines.append(f"- Updated: `{generated_at or '?'}`")
    lines.append(f"- Total lanes: **{len(lanes)}**")
    lines.append(f"- Level 3 (FULL PRODUCTION HARDENED): **{by_level[3]}**")
    lines.append(f"- Level 2 (INTEGRATION):              **{by_level[2]}**")
    lines.append(f"- Level 1 (SCAFFOLD):                 **{by_level[1]}**")
    lines.append(f"- Level 0 (SKETCH):                   **{by_level[0]}**")
    lines.append("")
    lines.append("## Gate definitions")
    lines.append("")
    for g, desc in (data.get("gate_definitions") or {}).items():
        lines.append(f"- **`{g}`** — {desc}")
    lines.append("")

    # Per-phase
    phases: dict[float | int, list[dict[str, Any]]] = {}
    for lane in lanes:
        phases.setdefault(lane.get("phase", "?"), []).append(lane)

    for phase in sorted(phases.keys(), key=lambda x: float(x) if isinstance(x, (int, float)) else 99):
        lines.append(f"## Phase {phase}")
        lines.append("")
        lines.append(
            "| Lane | Level | impl | emp | cuda | cpu | preflt | 3clean | mem | deploy | Notes |"
        )
        lines.append(
            "|------|-------|------|-----|------|-----|--------|--------|-----|--------|-------|"
        )
        for lane in phases[phase]:
            lvl = lane.get("level", 0)
            gates = lane.get("gates", {})

            def _cell(g: str, gates: dict[str, Any] = gates) -> str:
                return "✓" if gates.get(g, {}).get("status") is True else "✗"

            notes = (lane.get("notes") or "").replace("\n", " ").replace("|", "\\|")
            lines.append(
                f"| `{lane['id']}` ({lane.get('name','')}) | "
                f"**L{lvl}** | "
                f"{_cell('impl_complete')} | "
                f"{_cell('real_archive_empirical')} | "
                f"{_cell('contest_cuda')} | "
                f"{_cell('contest_cpu')} | "
                f"{_cell('strict_preflight')} | "
                f"{_cell('three_clean_review')} | "
                f"{_cell('memory_entry')} | "
                f"{_cell('deploy_runbook')} | "
                f"{notes} |"
            )
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


# ── CLI ──────────────────────────────────────────────────────────────────


def cmd_audit(args: argparse.Namespace) -> int:
    data = load_registry()
    print(render_audit_table(data))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    data = load_registry()
    errors = validate_registry(data)
    if errors:
        print(_red(f"VALIDATION FAILED ({len(errors)} error(s)):"), file=sys.stderr)
        for e in errors:
            print(f"  • {e}", file=sys.stderr)
        return 2
    print(_green(f"OK — {len(data['lanes'])} lane(s) validated cleanly."))
    return 0


def cmd_mark(args: argparse.Namespace) -> int:
    with _mutation_lock():
        data = load_registry()
        before = json.dumps(
            next((lane_entry for lane_entry in data["lanes"] if lane_entry["id"] == args.lane_id), None),
            sort_keys=True,
        )
        try:
            lane = mark_gate(data, args.lane_id, args.gate, args.evidence)
        except ValueError as e:
            print(_red(f"ERROR: {e}"), file=sys.stderr)
            return 2
        save_registry(data)
        after = json.dumps(lane, sort_keys=True)
        append_audit_log({
            "timestamp": _now_iso(),
            "command": "mark",
            "args": {"lane_id": args.lane_id, "gate": args.gate, "evidence": args.evidence},
            "before_state": before,
            "after_state": after,
        })
    print(_green(
        f"OK — {args.lane_id}.{args.gate} = true (level now L{lane['level']})"
    ))
    return 0


def cmd_unmark(args: argparse.Namespace) -> int:
    with _mutation_lock():
        data = load_registry()
        before = json.dumps(
            next((lane_entry for lane_entry in data["lanes"] if lane_entry["id"] == args.lane_id), None),
            sort_keys=True,
        )
        try:
            lane = unmark_gate(data, args.lane_id, args.gate, args.reason)
        except ValueError as e:
            print(_red(f"ERROR: {e}"), file=sys.stderr)
            return 2
        save_registry(data)
        after = json.dumps(lane, sort_keys=True)
        append_audit_log({
            "timestamp": _now_iso(),
            "command": "unmark",
            "args": {
                "lane_id": args.lane_id,
                "gate": args.gate,
                "reason": args.reason,
            },
            "before_state": before,
            "after_state": after,
        })
    print(_yellow(
        f"OK — {args.lane_id}.{args.gate} = false "
        f"(level now L{lane['level']}); reason: {args.reason}"
    ))
    return 0


def cmd_add_lane(args: argparse.Namespace) -> int:
    with _mutation_lock():
        data = load_registry()
        try:
            lane = add_lane(data, args.lane_id, args.name, args.phase, args.notes)
        except ValueError as e:
            print(_red(f"ERROR: {e}"), file=sys.stderr)
            return 2
        save_registry(data)
        append_audit_log({
            "timestamp": _now_iso(),
            "command": "add-lane",
            "args": {
                "lane_id": args.lane_id, "name": args.name,
                "phase": args.phase, "notes": args.notes,
            },
            "before_state": "null",
            "after_state": json.dumps(lane, sort_keys=True),
        })
    print(_green(f"OK — added lane {args.lane_id} at L0 (phase {args.phase})"))
    return 0


def _coerce_set_field_value(field: str, raw: str) -> Any:
    """Coerce CLI string input to the appropriate Python type for `field`.

    Bool fields: 'true'/'false'/'1'/'0' (case-insensitive)
    Int fields: int(raw)
    List fields (runtime_dep_closure / reactivation_criteria): comma-separated
    Other: pass-through string.
    """
    f = field.split(".", 1)[-1] if "." in field else field
    bool_fields = {"research_only", "no_op_detector_planned"}
    int_fields = {"inflate_runtime_loc_budget", "bolt_on_loc_budget"}
    list_fields = {"runtime_dep_closure", "reactivation_criteria"}
    if f in bool_fields:
        low = raw.strip().lower()
        if low in ("true", "1", "yes", "y"):
            return True
        if low in ("false", "0", "no", "n"):
            return False
        raise ValueError(f"bool field {field!r} expects true/false, got {raw!r}")
    if f in int_fields:
        try:
            return int(raw)
        except ValueError as e:
            raise ValueError(
                f"int field {field!r} expects integer, got {raw!r}: {e}"
            ) from e
    if f in list_fields:
        items = [s.strip() for s in raw.split(",") if s.strip()]
        return items
    return raw


def cmd_set_field(args: argparse.Namespace) -> int:
    with _mutation_lock():
        data = load_registry()
        before = json.dumps(
            next((lane_entry for lane_entry in data["lanes"] if lane_entry["id"] == args.lane_id), None),
            sort_keys=True,
        )
        try:
            coerced = _coerce_set_field_value(args.field, args.value)
            lane = set_field(data, args.lane_id, args.field, coerced)
        except ValueError as e:
            print(_red(f"ERROR: {e}"), file=sys.stderr)
            return 2
        save_registry(data)
        after = json.dumps(lane, sort_keys=True)
        append_audit_log({
            "timestamp": _now_iso(),
            "command": "set-field",
            "args": {
                "lane_id": args.lane_id,
                "field": args.field,
                "value": coerced,
            },
            "before_state": before,
            "after_state": after,
        })
    print(_green(
        f"OK — {args.lane_id}.{args.field} = {coerced!r}"
    ))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    data = load_registry()
    md = render_report_md(data)
    out = _report_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(_green(f"OK — wrote {out.relative_to(REPO_ROOT)}"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog="lane_maturity",
    )
    sub = p.add_subparsers(dest="cmd")

    pa = sub.add_parser("audit", help="print colored audit table")
    pa.set_defaults(func=cmd_audit)

    pv = sub.add_parser("validate", help="exit nonzero if registry inconsistent")
    pv.set_defaults(func=cmd_validate)

    pm = sub.add_parser("mark", help="set a gate to true with evidence")
    pm.add_argument("lane_id", help="lane id (e.g. lane_g_v3)")
    pm.add_argument("--gate", required=True, choices=GATES,
                    help=f"one of: {', '.join(GATES)}")
    pm.add_argument("--evidence", required=True,
                    help="evidence string (file path will be checked for existence)")
    pm.set_defaults(func=cmd_mark)

    pu = sub.add_parser("unmark", help="revert a gate to false")
    pu.add_argument("lane_id", help="lane id")
    pu.add_argument("--gate", required=True, choices=GATES)
    pu.add_argument("--reason", required=True,
                    help="why this gate is being reverted")
    pu.set_defaults(func=cmd_unmark)

    pal = sub.add_parser("add-lane", help="register a new lane at level 0")
    pal.add_argument("lane_id", help="unique lane id")
    pal.add_argument("--name", required=True, help="display name")
    pal.add_argument("--phase", required=True, type=float,
                     help="phase number (1, 1.5, 2, 3)")
    pal.add_argument("--notes", default="", help="optional notes")
    pal.set_defaults(func=cmd_add_lane)

    psf = sub.add_parser(
        "set-field",
        help=(
            "set a top-level scalar field "
            "(name / notes / lane_class / research_only / reactivation_criteria) OR a "
            "design_evidence.<sub> field for Catalog #124"
        ),
    )
    psf.add_argument("lane_id", help="lane id")
    psf.add_argument(
        "--field",
        required=True,
        help=(
            "field name. Top-level: lane_class, research_only, "
            "reactivation_criteria, name, notes. Or design_evidence.<sub> where <sub> "
            "in {archive_grammar, parser_section_manifest, "
            "inflate_runtime_loc_budget, runtime_dep_closure, export_format, "
            "score_aware_loss, bolt_on_loc_budget, no_op_detector_planned}."
        ),
    )
    psf.add_argument(
        "--value",
        required=True,
        help=(
            "value (string). Bools accept true/false/1/0/yes/no. "
            "Ints accept integer literals. Lists "
            "(runtime_dep_closure / reactivation_criteria) accept "
            "comma-separated values."
        ),
    )
    psf.set_defaults(func=cmd_set_field)

    pr = sub.add_parser("report",
                        help=f"write {REPORT_REL}")
    pr.set_defaults(func=cmd_report)

    pfr = sub.add_parser(
        "strict-flip-readiness-report",
        help=(
            "audit lanes for missing 8 archive-grammar fields (post-strict-flip "
            "should be 0); print per-lane missing-field counts."
        ),
    )
    pfr.set_defaults(func=cmd_strict_flip_readiness)

    return p


def cmd_strict_flip_readiness(args: argparse.Namespace) -> int:
    """Audit lanes for missing 8 archive-grammar fields (Catalog #124).

    Per CLAUDE.md "Lane maturity registry" + Catalog #124, every L1+
    representation lane MUST have all 8 design_evidence sub-fields populated
    OR explicitly opt out via lane_class=substrate_engineering /
    research_only=true. After Catalog #124's strict-flip, this count should
    be 0 for representation lanes.

    Output: per-lane missing-field tally + summary. Exit 0 if all clean,
    1 if any lane is missing fields without an opt-out. Tagged
    [empirical: tools/lane_maturity.py strict-flip-readiness-report].
    """
    data = load_registry()
    representation_lane_classes = {
        "representation_codec", "representation_only", "codec_only",
    }
    n_lanes = 0
    n_missing_anywhere = 0
    rows: list[tuple[str, list[str], str]] = []
    for lane in data["lanes"]:
        lane_id = lane["id"]
        lane_class = lane.get("lane_class", "")
        research_only = lane.get("research_only", False)
        if research_only:
            continue
        # Skip lanes with substrate_engineering / non-representation lane_class
        if lane_class and lane_class not in representation_lane_classes:
            continue
        n_lanes += 1
        de = lane.get("design_evidence", {}) or {}
        missing = [
            sub for sub in _SET_FIELD_DESIGN_EVIDENCE_ALLOWED
            if sub not in de or de[sub] in (None, "", [], {})
        ]
        if missing:
            n_missing_anywhere += 1
            rows.append((lane_id, missing, lane_class or "<unset>"))
    rows.sort(key=lambda x: (-len(x[1]), x[0]))
    print("STRICT-FLIP READINESS REPORT (Catalog #124)")
    print("─" * 70)
    print(f"Audited lanes: {n_lanes} (excluding research_only=true and "
          f"non-representation lane_class)")
    print(f"Lanes with 1+ missing design_evidence sub-field: {n_missing_anywhere}")
    if rows:
        print()
        print(f"{'lane_id':<60} {'lane_class':<25} {'missing'}")
        print(f"{'-'*60} {'-'*25} {'-'*8}")
        for lane_id, missing, lane_class in rows:
            print(f"{lane_id[:60]:<60} {lane_class[:25]:<25} {len(missing)}/8")
            for sub in missing:
                print(f"  - {sub}")
    print()
    if n_missing_anywhere == 0:
        print(_green("OK — all representation lanes have all 8 design_evidence "
                     "sub-fields populated; Catalog #124 strict-flip is held."))
        return 0
    print(_yellow(f"WARN — {n_missing_anywhere} lane(s) missing fields. "
                  "Backfill via `lane_maturity.py set-field <id> --field "
                  "design_evidence.<sub> --value <v>` before relying on "
                  "Catalog #124 strict-flip."))
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        # Default: audit
        return cmd_audit(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
