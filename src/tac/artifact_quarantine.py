"""Artifact quarantine — old-archive anchors are HARVEST-SIGNAL-ONLY (operator 2026-07-21).

Operator binding (verbatim): "Never fucking do that again. We need to maybe quarantine that
stuff because it's wasting a ton of time. The witness is one half, but the from scratch
inverse solve is the other half. But none of that is related at all to those old shitty
archives except for their signal."

The manifest at ``.omx/state/artifact_quarantine.json`` lists identifiers (archive SHA
prefixes, checkpoint run dirs, anchor tokens) of retired-vehicle ARTIFACTS. Their
measurements are citable signal; their bytes are never an anchor/carrier/warm-start/
transplant source in the v10 line. ``scan_text`` finds quarantined identifiers in a dispatch
prompt; a prompt that names one is refused unless it carries the waiver line
``QUARANTINE-WAIVER: HARVEST-SIGNAL-ONLY — <rationale>`` (>= 10 chars of real rationale).

Fail-open on a MISSING manifest (loud warning, never brick dispatch); fail-closed on a
present-but-unparseable manifest (corruption is not a license).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

MANIFEST_RELPATH = Path(".omx/state/artifact_quarantine.json")

# Tracked, versioned DEFAULT quarantine set (operator 2026-07-21). The gitignored
# manifest EXTENDS this list locally; a fresh checkout is protected by these rows alone.
DEFAULT_QUARANTINED: tuple[dict, ...] = (
    {"id": "149fefd097c1fa85", "kind": "archive_sha_prefix",
     "what": "ep725 witness byte-closed archive (83,838 B, pose-blind)",
     "reason": "old witness-vehicle artifact; anchored the 07-21 archive-gravity recurrence"},
    {"id": "d2ad27cf1c6a5e34", "kind": "archive_sha_prefix",
     "what": "r5 composed v5 archive (witness-derived)",
     "reason": "derivative of the quarantined ep725 archive"},
    {"id": "ep725", "kind": "token",
     "what": "the witness checkpoint epoch anchor",
     "reason": "recurring anchor phrase; prompts citing it consume the old vehicle"},
    {"id": "levelset_n600_witness_20260717T113932Z", "kind": "path_token",
     "what": "the 2026-07-17 witness run dir (checkpoints/EMA)",
     "reason": "old trained checkpoints — warm-start/transplant source"},
    {"id": "b0a431e9259cd3c5", "kind": "checkpoint_sha_prefix",
     "what": "ep725 EMA checkpoint", "reason": "same artifact class"},
    {"id": "196acd18", "kind": "archive_sha_prefix",
     "what": "bank 0.18804 PR128-splice-on-PR110 archive",
     "reason": "borrowed lineage NON-SUBMISSION bank"},
    {"id": "6bae0201", "kind": "archive_sha_prefix",
     "what": "pr101_frame_exploit_selector frontier archive", "reason": "old frontier lineage"},
    {"id": "b83bf348", "kind": "archive_sha_prefix",
     "what": "PR101 GOLD upstream baseline archive", "reason": "inherited lineage"},
    {"id": "r1 dxi", "kind": "token_ci",
     "what": "the banked R1 store-nothing pose dxi artifact",
     "reason": "trained pose artifact; its NUMBERS remain citable signal"},
    # --- 2026-07-21 RECURRENCE #2: the M1/C2 byte-close family. The first manifest
    # registered the ep725/r5 witness family but MISSED these; per_stratum decoded
    # a386a854 as a "control" and a receiver prompt cited it as a scaffold. These are
    # TRACKED defaults (not the gitignored manifest) so they reach EVERY worktree and
    # EVERY Claude/Codex subagent via `import tac`, not just this checkout. ---
    {"id": "a386a854", "kind": "archive_sha_prefix",
     "what": "M1/C2 banded-generator byte-close (90,566 B) — STOPPED MID-RUN, pose OFF "
             "(d_pose 127.4), segnet not conditioned for joint descent",
     "reason": "07-21 archive-gravity RECURRENCE #2 — un-converged; harvest-signal-only"},
    {"id": "3748485f", "kind": "archive_sha_prefix",
     "what": "r5 waterfill d2_jrd_fixture_transfer byte-close (79,008 B)",
     "reason": "witness/r5-derived, pre-realization-crux"},
    {"id": "d633e6bf", "kind": "archive_sha_prefix",
     "what": "m1_c2_glue_rebuild pre_archive (94,344 B)",
     "reason": "old-vehicle byte-close, pre-realization-crux"},
    {"id": "e9e42971", "kind": "archive_sha_prefix",
     "what": "m1_c2_glue_rebuild post_archive (94,352 B)",
     "reason": "old-vehicle byte-close, pre-realization-crux"},
    {"id": "m1_byteclose_20260721", "kind": "path_token",
     "what": "M1/C2 byte-close evidence dir", "reason": "old M1/C2 bytes — harvest-signal-only"},
    {"id": "m1_c2_glue_rebuild_20260719", "kind": "path_token",
     "what": "M1/C2 glue-rebuild evidence dir", "reason": "old M1/C2 bytes — harvest-signal-only"},
)
_WAIVER_RE = re.compile(
    r"QUARANTINE-WAIVER:\s*HARVEST-SIGNAL-ONLY\s*[—-]\s*(\S.{9,})", re.IGNORECASE
)


@dataclass(frozen=True)
class QuarantineHit:
    identifier: str
    kind: str
    what: str
    reason: str


class QuarantineManifestError(RuntimeError):
    """Manifest present but unusable — fail closed."""


def _repo_root(start: Path | None = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for cand in [p, *p.parents]:
        if (cand / ".omx").is_dir():
            return cand
    return Path.cwd()


def load_manifest(repo_root: Path | None = None) -> dict | None:
    root = repo_root or _repo_root()
    path = root / MANIFEST_RELPATH
    if not path.is_file():
        return None  # absent manifest: scan_text falls back to DEFAULT_QUARANTINED
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:  # corruption != license
        raise QuarantineManifestError(f"unreadable quarantine manifest {path}: {exc}") from exc
    if not isinstance(data.get("quarantined"), list):
        raise QuarantineManifestError(f"manifest {path} lacks a 'quarantined' list")
    return data


def has_waiver(text: str) -> bool:
    return bool(_WAIVER_RE.search(text))


def scan_text(text: str, repo_root: Path | None = None) -> list[QuarantineHit]:
    """Return quarantine hits in ``text`` (empty if waived, manifest-less, or clean)."""
    manifest = load_manifest(repo_root)
    rows = list(DEFAULT_QUARANTINED)
    if manifest is not None:
        known = {str(r.get("id")) for r in rows}
        rows += [r for r in manifest["quarantined"] if str(r.get("id")) not in known]
    if has_waiver(text):
        return []
    hits: list[QuarantineHit] = []
    lower = text.lower()
    for row in rows:
        ident = str(row.get("id", ""))
        if not ident:
            continue
        kind = str(row.get("kind", "token"))
        haystack = lower if kind.endswith("_ci") or kind == "token" else text
        needle = ident.lower() if haystack is lower else ident
        if needle in haystack:
            hits.append(
                QuarantineHit(
                    identifier=ident,
                    kind=kind,
                    what=str(row.get("what", "")),
                    reason=str(row.get("reason", "")),
                )
            )
    return hits


def refuse_message(hits: list[QuarantineHit]) -> str:
    lines = [
        "ARTIFACT QUARANTINE REFUSAL (operator 2026-07-21 — old archives are",
        "HARVEST-SIGNAL-ONLY; bytes are never an anchor in the v10 line):",
    ]
    for h in hits:
        lines.append(f"  • '{h.identifier}' ({h.kind}) — {h.what} [{h.reason}]")
    lines.append(
        "If ONLY measurements/receipts (never bytes/weights) are consumed, add the line: "
        "'QUARANTINE-WAIVER: HARVEST-SIGNAL-ONLY — <real rationale>' to the prompt."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RUNTIME DECODE GATE (operator 2026-07-21: "apply to main and all worktrees and
# all Claude and Codex subagents"). The prompt-scan above guards only the SPAWN
# text; this guards the BYTES at the moment of decode. Because every decode path
# imports `tac` and passes an archive through here, this fires in main, in every
# worktree, and in every Claude/Codex subagent — regardless of which spawn path
# (or none) created the work, and regardless of whether the prompt named the SHA.
# ---------------------------------------------------------------------------

_SIGNAL_ONLY_ENV = "TAC_ARTIFACT_QUARANTINE_SIGNAL_ONLY"


class QuarantineViolation(RuntimeError):
    """A quarantined artifact's BYTES were opened for consumption. Fail closed."""


def sha256_of_file(path: Path | str) -> str:
    """Streaming sha256 of a file's bytes (empty string if unreadable)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def is_quarantined_archive(
    path: Path | str, repo_root: Path | None = None
) -> list[QuarantineHit]:
    """QuarantineHits for an archive FILE — matched by content sha256 AND by path.

    sha256 catches a rename/copy of the exact bytes (the a386a854 recurrence class);
    the path string catches evidence-dir path tokens. Waiver applies as in scan_text.
    """
    p = Path(path)
    hits: list[QuarantineHit] = []
    sha = sha256_of_file(p)
    if sha:
        hits += scan_text(sha, repo_root)
    hits += scan_text(str(p), repo_root)
    seen: set[str] = set()
    uniq: list[QuarantineHit] = []
    for h in hits:
        if h.identifier not in seen:
            seen.add(h.identifier)
            uniq.append(h)
    return uniq


def assert_not_quarantined_archive(
    path: Path | str, *, context: str = "", repo_root: Path | None = None
) -> None:
    """Refuse to OPEN a quarantined archive's bytes for consumption.

    Call this at every decode/byte-close/measurement archive-open site. A quarantined
    archive may be opened SIGNAL-ONLY (metrics/receipts, never bytes-as-carrier) by
    setting ``TAC_ARTIFACT_QUARANTINE_SIGNAL_ONLY=<>=10-char rationale>``; that logs
    loudly and returns. Otherwise raises :class:`QuarantineViolation`.
    """
    hits = is_quarantined_archive(path, repo_root)
    if not hits:
        return
    waiver = os.environ.get(_SIGNAL_ONLY_ENV, "").strip()
    if len(waiver) >= 10:
        print(
            f"[artifact_quarantine] SIGNAL-ONLY open of quarantined archive {path} "
            f"(context={context!r}; rationale={waiver!r})",
            file=sys.stderr,
        )
        return
    raise QuarantineViolation(
        f"REFUSED to decode quarantined archive {path} (context={context!r}).\n"
        f"{refuse_message(hits)}\n"
        f"If ONLY signal (metrics/receipts) is consumed, set {_SIGNAL_ONLY_ENV}=<rationale>."
    )
