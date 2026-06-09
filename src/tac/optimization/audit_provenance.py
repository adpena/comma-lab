# SPDX-License-Identifier: MIT
"""Canonical audit-provenance manifest — the AUDIT-HALLUCINATION firewall.

Source: operator NON-NEGOTIABLE V6 directive 2026-06-09 (P2 of the
ObjectiveReachability + AuditProvenance hardening packet). Empirical anchors —
three audit-provenance lapses on 2026-06-09 alone:

1. **wrong-candidate** — ep22399's ``avg_segnet_dist = 0.7115`` was attributed
   to the WRONG candidate (a Haar score renderer, not the SNeRV-B candidate the
   prose claimed). Cite
   ``.omx/research/snerv_b_first_scorer_probe_verdict_20260609.md`` finding 3.
2. **phantom-gates** — pact phantom-gates were claimed TRUE in prose while the
   registry showed False.
3. **surface-conflation** — the ``0.71``-vs-``0.0023`` conflation: ep22399's
   0.7115 was the EXPORT/receiver-side surface; today's 0.0023 is the LIVE
   in-memory render. The same class as PSNR != d_seg. The "0.71 -> <0.2" SNeRV-B
   prediction was therefore against the wrong baseline.

All three share one structural cause: an audit CLAIM made WITHOUT (a) naming the
exact candidate, (b) naming the SURFACE the value lives on, and (c) a
reproduce_command another agent can re-run. The cure is a typed claim record
whose ``surface`` field is MANDATORY (today's 0.71-vs-0.0023 was a surface
conflation) and whose ``reproduce_command`` is MANDATORY (an unreproducible
claim is a hallucination). ``verify()`` fails closed on a missing surface OR a
missing reproduce_command.

The five canonical surfaces (the operator-explicit metric-laundering firewall —
Vehicle-OS rule 5): a measured value lives on EXACTLY ONE of::

    live          — the in-memory render / live frozen-scorer batch surface
    receiver      — the receiver/parse-back surface (post-export decode)
    export        — the exported archive's decoded frames
    exact_archive — upstream/evaluate.py on the exact archive.zip bytes (authority)
    telemetry     — a training telemetry/proxy field (never an authority)

Conflating two surfaces (the 0.71 export value vs the 0.0023 live value) is the
bug; the mandatory ``surface`` field makes it structurally impossible to compare
two values without first declaring they share a surface.

Per CLAUDE.md "Results must become system intelligence" + "Beauty, simplicity,
and developer experience": records are emitted as durable JSON under
``.omx/state/audit_provenance/`` (NEVER ``/tmp``) so the Catalog #387 gate +
operator dashboards + the next subagent inherit the verified-claim ledger.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" + Catalog #287 placeholder discipline:
an unverifiable record (a claim whose reproduce_command cannot be safely
executed in this environment) is marked ``verified_by_main_agent=False`` with a
truthful ``unverifiable`` status — NOT a fabricated "verified" stamp.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "CANONICAL_AUDIT_SURFACES",
    "AuditProvenanceFinding",
    "AuditProvenanceRecord",
    "AuditProvenanceVerifyError",
    "RecheckResult",
    "audit_provenance_claim_records",
    "audit_provenance_path_for_claim",
    "default_state_dir",
    "emit_audit_provenance_record",
    "load_audit_provenance_records",
    "recheck",
]

# The five canonical surfaces a measured value can live on (Vehicle-OS rule 5
# metric-laundering firewall). The ``surface`` field MUST be one of these.
CANONICAL_AUDIT_SURFACES: tuple[str, ...] = (
    "live",
    "receiver",
    "export",
    "exact_archive",
    "telemetry",
)

# Forbidden placeholder literals for the reproduce_command (Catalog #287).
_FORBIDDEN_REPRODUCE_PLACEHOLDERS: frozenset[str] = frozenset(
    {"", "tbd", "<value>", "<command>", "placeholder", "pending", "n/a", "na", "none"}
)


class AuditProvenanceVerifyError(ValueError):
    """Raised by :meth:`AuditProvenanceRecord.verify` on a hallucination finding."""


@dataclass(frozen=True)
class AuditProvenanceRecord:
    """Typed claim record for one audit-provenance assertion.

    Schema (per operator V6 spec 2026-06-09):

    * ``claim`` — the human-readable assertion (e.g. "SNeRV-B live d_seg is 0.0023").
    * ``file`` — the source file the claim is made in / about.
    * ``line_or_field`` — the line number or JSON/telemetry field the value comes
      from (e.g. ``telemetry.jsonl:ep0:live_argmax_d_seg`` or ``inflate.py:49``).
    * ``candidate_id`` — the EXACT candidate the value belongs to (the wrong-
      candidate lapse was attributing ep22399 to the wrong candidate).
    * ``observed_value`` — the value as claimed.
    * ``expected_value`` — the value the claim is compared against (or "" when the
      record is a standalone observation, not a comparison).
    * ``reproduce_command`` — MANDATORY. The exact command another agent runs to
      re-derive ``observed_value``. An unreproducible claim is a hallucination.
    * ``verified_by_main_agent`` — True ONLY when the main agent ran the
      reproduce_command and confirmed ``observed_value`` (or :func:`recheck` did).
    * ``surface`` — MANDATORY. One of :data:`CANONICAL_AUDIT_SURFACES`. The
      0.71-vs-0.0023 conflation was a surface conflation; this field makes it
      structurally impossible.
    """

    claim: str
    file: str
    line_or_field: str
    candidate_id: str
    observed_value: str
    surface: str
    reproduce_command: str
    expected_value: str = ""
    verified_by_main_agent: bool = False
    notes: str = ""
    source_artifacts: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.claim, str) or not self.claim.strip():
            raise ValueError("claim must be a non-empty string")
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError(
                "candidate_id must be a non-empty string (the wrong-candidate "
                "lapse was an unnamed/wrong candidate)"
            )

    # -- the fail-closed hallucination check -------------------------------

    def _surface_finding(self) -> str | None:
        """Surface-field validity (MANDATORY; the 0.71-vs-0.0023 conflation cure)."""
        s = (self.surface or "").strip().lower()
        if not s:
            return (
                "MISSING-SURFACE: surface is empty — every measured value MUST "
                f"declare its surface (one of {CANONICAL_AUDIT_SURFACES}). The "
                "0.71-vs-0.0023 lapse was a surface conflation (export vs live)."
            )
        if s not in CANONICAL_AUDIT_SURFACES:
            return (
                f"INVALID-SURFACE: surface {self.surface!r} not in canonical "
                f"surfaces {CANONICAL_AUDIT_SURFACES}."
            )
        return None

    def _reproduce_finding(self) -> str | None:
        """reproduce_command validity (MANDATORY; an unreproducible claim is a hallucination)."""
        cmd = (self.reproduce_command or "").strip()
        if cmd.lower() in _FORBIDDEN_REPRODUCE_PLACEHOLDERS:
            return (
                f"MISSING-REPRODUCE-COMMAND: reproduce_command {self.reproduce_command!r} "
                f"is empty or a forbidden placeholder — an unreproducible audit "
                f"claim is a hallucination (Catalog #287)."
            )
        return None

    def provenance_findings(self) -> tuple[str, ...]:
        """Return the audit-provenance findings (empty tuple = clean).

        Two fail-closed conditions (the operator-mandatory fields):

        * **missing/invalid surface** — the 0.71-vs-0.0023 conflation cure.
        * **missing reproduce_command** — the unreproducible-claim cure.

        A record carrying both a valid surface AND a real reproduce_command
        produces ZERO findings (whether or not it has been verified yet — the
        unverified-but-reproducible record is honest, not a hallucination).
        """
        findings: list[str] = []
        sf = self._surface_finding()
        if sf:
            findings.append(sf)
        rf = self._reproduce_finding()
        if rf:
            findings.append(rf)
        return tuple(findings)

    def verify(self) -> None:
        """Fail closed on a missing surface OR missing reproduce_command.

        Raises :class:`AuditProvenanceVerifyError` when
        :meth:`provenance_findings` is non-empty.
        """
        findings = self.provenance_findings()
        if findings:
            raise AuditProvenanceVerifyError(
                f"audit_provenance_manifest.verify() failed for candidate "
                f"{self.candidate_id!r}:\n  " + "\n  ".join(findings)
            )

    # -- serialization -----------------------------------------------------

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "audit_provenance_manifest.v1",
            "claim": self.claim,
            "file": self.file,
            "line_or_field": self.line_or_field,
            "candidate_id": self.candidate_id,
            "observed_value": self.observed_value,
            "expected_value": self.expected_value,
            "reproduce_command": self.reproduce_command,
            "verified_by_main_agent": self.verified_by_main_agent,
            "surface": self.surface,
            "notes": self.notes,
            "source_artifacts": list(self.source_artifacts),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AuditProvenanceRecord:
        schema = str(payload.get("schema", ""))
        if schema and schema != "audit_provenance_manifest.v1":
            raise ValueError(
                f"unexpected schema {schema!r}; expected "
                f"'audit_provenance_manifest.v1'"
            )
        return cls(
            claim=str(payload["claim"]),
            file=str(payload.get("file", "")),
            line_or_field=str(payload.get("line_or_field", "")),
            candidate_id=str(payload["candidate_id"]),
            observed_value=str(payload.get("observed_value", "")),
            surface=str(payload.get("surface", "")),
            reproduce_command=str(payload.get("reproduce_command", "")),
            expected_value=str(payload.get("expected_value", "")),
            verified_by_main_agent=bool(payload.get("verified_by_main_agent", False)),
            notes=str(payload.get("notes", "")),
            source_artifacts=_as_str_tuple(payload.get("source_artifacts", ())),
        )


# ---------------------------------------------------------------------------
# recheck — run the reproduce_command and compare (where safely executable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecheckResult:
    """Outcome of :func:`recheck` on a single record."""

    candidate_id: str
    status: str  # "verified" | "mismatch" | "unverifiable" | "error"
    detail: str
    observed_in_output: bool = False

    def ok(self) -> bool:
        return self.status == "verified"


# Command prefixes considered safe to execute for a recheck. A reproduce_command
# whose first token is not in this allowlist is marked ``unverifiable`` rather
# than executed (no arbitrary shell). This is conservative by design — a record
# whose command we cannot safely run stays honestly unverified, never
# falsely-verified.
_SAFE_RECHECK_PREFIXES: tuple[tuple[str, ...], ...] = (
    (".venv/bin/python",),
    ("python",),
    ("python3",),
    ("cat",),
    ("jq",),
    ("grep",),
    ("rg",),
    ("git", "show"),
    ("git", "log"),
)


def _command_is_safe(argv: Sequence[str]) -> bool:
    return any(
        tuple(argv[: len(prefix)]) == prefix for prefix in _SAFE_RECHECK_PREFIXES
    )


def recheck(
    record: AuditProvenanceRecord,
    repo_root: str | Path | None = None,
    *,
    timeout_s: float = 60.0,
) -> RecheckResult:
    """Run ``record.reproduce_command`` and compare ``observed_value``.

    Returns a :class:`RecheckResult`:

    * ``unverifiable`` — the record has a provenance finding (missing surface /
      reproduce_command) OR the command is not in the safe allowlist OR there is
      no ``observed_value`` to look for. Honest non-verification, NOT a failure
      stamp.
    * ``verified`` — the command ran (rc 0) AND ``observed_value`` appears in its
      output.
    * ``mismatch`` — the command ran but ``observed_value`` is absent from its
      output (the wrong-candidate / phantom-gate lapse signature).
    * ``error`` — the command raised / timed out / returned non-zero.

    Per CLAUDE.md: this NEVER mutates the record's ``verified_by_main_agent`` in
    place (frozen dataclass). The caller emits a NEW record with the verified
    stamp after a ``verified`` result.
    """
    if record.provenance_findings():
        return RecheckResult(
            candidate_id=record.candidate_id,
            status="unverifiable",
            detail="record has provenance findings (missing surface / reproduce_command)",
        )
    if not record.observed_value.strip():
        return RecheckResult(
            candidate_id=record.candidate_id,
            status="unverifiable",
            detail="no observed_value to compare; recheck cannot confirm a claim with no value",
        )
    try:
        argv = shlex.split(record.reproduce_command)
    except ValueError as exc:
        return RecheckResult(
            candidate_id=record.candidate_id,
            status="unverifiable",
            detail=f"reproduce_command is not shell-parseable: {exc}",
        )
    if not argv or not _command_is_safe(argv):
        return RecheckResult(
            candidate_id=record.candidate_id,
            status="unverifiable",
            detail=(
                f"reproduce_command {record.reproduce_command!r} is not in the safe "
                f"recheck allowlist; mark unverifiable rather than execute arbitrary shell"
            ),
        )
    repo = Path(repo_root) if repo_root is not None else None
    try:
        proc = subprocess.run(
            argv,
            cwd=str(repo) if repo else None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return RecheckResult(
            candidate_id=record.candidate_id,
            status="error",
            detail=f"reproduce_command failed to run: {exc}",
        )
    output = (proc.stdout or "") + (proc.stderr or "")
    observed_in_output = record.observed_value.strip() in output
    if proc.returncode != 0:
        return RecheckResult(
            candidate_id=record.candidate_id,
            status="error",
            detail=f"reproduce_command returned rc={proc.returncode}",
            observed_in_output=observed_in_output,
        )
    if observed_in_output:
        return RecheckResult(
            candidate_id=record.candidate_id,
            status="verified",
            detail=f"observed_value {record.observed_value!r} found in command output",
            observed_in_output=True,
        )
    return RecheckResult(
        candidate_id=record.candidate_id,
        status="mismatch",
        detail=(
            f"observed_value {record.observed_value!r} NOT found in command output "
            f"(the wrong-candidate / phantom-gate / surface-conflation signature)"
        ),
        observed_in_output=False,
    )


# ---------------------------------------------------------------------------
# durable surface + audit helper (consumed by the Catalog #387 gate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditProvenanceFinding:
    """One audit-provenance finding for the Catalog #387 gate."""

    candidate_id: str
    finding: str
    record_path: str

    def message(self) -> str:
        return (
            f"AUDIT-PROVENANCE [{self.candidate_id}] ({self.record_path}): "
            f"{self.finding}"
        )


def _as_str_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(v) for v in value)
    raise ValueError(f"expected a sequence of strings; got {type(value)!r}")


def default_state_dir(repo_root: str | Path | None = None) -> Path:
    """Return the durable record directory ``.omx/state/audit_provenance``.

    NEVER ``/tmp`` per CLAUDE.md "Forbidden /tmp paths" — records are durable
    operator-facing evidence consumed by the Catalog #387 gate.
    """
    if repo_root is None:
        # This file lives at src/tac/optimization/; repo root is 4 up.
        repo_root = Path(__file__).resolve().parents[3]
    return Path(repo_root) / ".omx" / "state" / "audit_provenance"


def audit_provenance_path_for_claim(
    candidate_id: str, repo_root: str | Path | None = None
) -> Path:
    # Sanitize candidate_id into a filesystem-safe stem.
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in candidate_id)
    return default_state_dir(repo_root) / f"{safe}.json"


def emit_audit_provenance_record(
    record: AuditProvenanceRecord,
    repo_root: str | Path | None = None,
    *,
    verify: bool = False,
) -> Path:
    """Write ``record`` as durable JSON; return the path.

    When ``verify=True`` the record is checked for the missing-surface /
    missing-reproduce-command conditions BEFORE writing (so a hallucination can
    never be emitted as if it were clean).
    """
    if verify:
        record.verify()
    out_dir = default_state_dir(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = audit_provenance_path_for_claim(record.candidate_id, repo_root)
    out_path.write_text(
        json.dumps(record.as_dict(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return out_path


def load_audit_provenance_records(
    repo_root: str | Path | None = None,
) -> list[tuple[AuditProvenanceRecord, Path]]:
    """Load every emitted record under ``.omx/state/audit_provenance``."""
    state_dir = default_state_dir(repo_root)
    out: list[tuple[AuditProvenanceRecord, Path]] = []
    if not state_dir.is_dir():
        return out
    for path in sorted(state_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            record = AuditProvenanceRecord.from_dict(payload)
        except (OSError, ValueError, KeyError):
            # A corrupt/malformed record is itself a provenance finding: emit a
            # sentinel record that fails closed (empty surface + reproduce).
            out.append(
                (
                    AuditProvenanceRecord(
                        claim=f"corrupt/malformed record at {path}",
                        file=str(path),
                        line_or_field="",
                        candidate_id=path.stem,
                        observed_value="",
                        surface="",
                        reproduce_command="",
                    ),
                    path,
                )
            )
            continue
        out.append((record, path))
    return out


def audit_provenance_claim_records(
    repo_root: str | Path | None = None,
) -> list[AuditProvenanceFinding]:
    """Return every audit-provenance finding across emitted records.

    A record whose :meth:`AuditProvenanceRecord.provenance_findings` is non-empty
    (missing/invalid surface OR missing reproduce_command) contributes one
    :class:`AuditProvenanceFinding` per finding. This is the canonical helper the
    Catalog #387 gate delegates to.
    """
    findings: list[AuditProvenanceFinding] = []
    for record, path in load_audit_provenance_records(repo_root):
        for finding in record.provenance_findings():
            findings.append(
                AuditProvenanceFinding(
                    candidate_id=record.candidate_id,
                    finding=finding,
                    record_path=str(path),
                )
            )
    return findings
