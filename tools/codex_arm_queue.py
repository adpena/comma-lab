#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Codex arm queue + SATURATION actuator.

The fleet is supposed to stay saturated: whenever a codex arm lands, the next
charter fires. That only happens reliably if it is MECHANICAL. Today (2026-08-04)
proved the alternative twice — a routing law that lived in a memo was applied on
zero spawns, and a sandbox flag omitted by hand killed an arm 25 minutes from a
frontier row. So saturation lives here, not in anyone's memory.

NOT the same object as ``.omx/state/dispatch_queue.md`` (last touched 2026-05-17),
which queues PAID GPU LANE dispatches (``scripts/remote_lane_*.sh``, dollar cost
bands, Vast.ai/Modal). Same word, different thing — see the retrieval-hazard class
(task #867). This queue holds CODEX ARM CHARTERS: prompt files spawned via the
canonical detached ``codex exec`` Pattern A.

Queue file: ``.omx/state/codex_arm_queue.jsonl`` — append-only rows:
    {"name","prompt_path","rank","owns_scorer","status","note", ...}
``status`` ∈ queued | live | landed | dropped. Latest row per ``name`` wins, so
status changes are appends, never rewrites (append-only custody).

Safety, all fail-closed:
  * hard cap (default 4) on concurrent codex arms — never exceeded;
  * at most ONE scorer-owning arm live at a time (the one-full-n600 rule);
  * refuses to spawn a name that is already live;
  * refuses a charter whose prompt file is missing;
  * every spawn carries ``--add-dir`` for the SSD tier (the flag whose absence
    killed fz3) and the canonical Pattern A detachment;
  * ``TAC_CODEX_SATURATE_OFF=1`` is the kill switch;
  * ``--dry-run`` is the default for ``saturate``; spawning requires ``--spawn``.

Usage:
    codex_arm_queue.py status                     # live arms, cap, next charter
    codex_arm_queue.py add --name X --prompt P --rank 30 [--owns-scorer]
    codex_arm_queue.py mark --name X --status landed
    codex_arm_queue.py saturate                   # report the gap (dry run)
    codex_arm_queue.py saturate --spawn           # actually fill the gap
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tac.subagent_contract import RETAINED_REASONING  # noqa: E402

QUEUE = _REPO / ".omx" / "state" / "codex_arm_queue.jsonl"
RUNS = _REPO / ".omx" / "tmp" / "codex_runs"
SPAWN_LOG = _REPO / ".omx" / "state" / "codex_arm_spawn_log.jsonl"
FINAL_MESSAGES = _REPO / ".omx" / "research" / "arm_final_messages"
FINAL_MESSAGE_INDEX = _REPO / ".omx" / "state" / "codex_arm_queue.final_messages.jsonl"
NEXT_IF_RESUMED = _REPO / ".omx" / "state" / "codex_arm_queue.next_if_resumed.jsonl"
ARM_CAPABILITIES = _REPO / ".omx" / "state" / "codex_arm_capabilities.json"

DEFAULT_CAP = 4
# EVERY connected SSD tier, not just tier-1. Granting only VertigoDataTier was a
# silent wall: when tier-1 filled to 98% and charters were re-routed to
# APDataStore, arms inherited a sandbox that forbade their own output directory
# and died with `PermissionError: [Errno 1] Operation not permitted` on mkdir --
# a capacity-shaped error message masking a permission cause (ddm_sd2 and
# ddm_vh2, both 2026-08-10, 997 GiB free at the time). Existence-gated so an
# unmounted drive cannot break the spawn.
SSD_ADD_DIRS = (
    "/Volumes/VertigoDataTier/pact",
    "/Volumes/APDataStore/pact",
)


def _ssd_add_dir_args() -> list[str]:
    """Emit one --add-dir per SSD tier that is actually mounted."""
    args: list[str] = []
    for tier in SSD_ADD_DIRS:
        if Path(tier).is_dir():
            args.extend(["--add-dir", tier])
    return args
# THE arm model. Single source of truth: the pin used to live inside
# keeper_source(), where a stale generation went unnoticed until the operator
# caught it (2026-08-08, "You should be using GPT five point six" -> then "a
# five point six SOL"). The value is VERIFIED against the operator's own codex
# config (~/.codex/config.toml: model = "gpt-5.6-sol"), not guessed. Env
# override exists so a model bump never requires a code edit mid-campaign.
_DEFAULT_ARM_MODEL = "gpt-5.6-sol"
ARM_MODEL = os.environ.get("TAC_CODEX_ARM_MODEL", _DEFAULT_ARM_MODEL)
# NEVER AGAIN (operator 2026-08-08, verbatim: "We are never spawning on five
# point five again"). This is a REFUSAL, not a default: spawn() fails closed if
# the resolved model matches, so an env override cannot resurrect the old
# generation either. Substring match catches every 5.5 variant/suffix.
BANNED_ARM_MODEL_SUBSTRINGS: tuple[str, ...] = ("gpt-5.5",)
# The reasoning-effort enum, READ FROM the installed codex binary's own variant
# table (strings | grep -> "...noneminimallowmediumhighxhighmaxultra..."),
# ascending. Recorded so a future effort choice is picked from the real
# vocabulary rather than invented. RE-VERIFIED against 0.147.0 on 2026-08-08
# after the cask upgrade (0.145.0 -> 0.147.0): vocabulary UNCHANGED, `ultra`
# still the ceiling. Re-read it after any future codex upgrade -- this list is
# a transcription of an external binary, not a value we own.
CODEX_EFFORT_ENUM: tuple[str, ...] = (
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
)
# The operator's admissible RANGE (2026-08-08: "high to ultra effort levels
# depending on the task at hand"). Effort is PER-TASK, chosen at `add` time --
# it is not one global constant. Below-`high` is refused for arm work.
ARM_EFFORT_LEVELS: tuple[str, ...] = ("high", "xhigh", "max", "ultra")
# Fallback for rows queued before `effort` existed. NOT a recommendation: pick
# the level deliberately per arm.
DEFAULT_ARM_EFFORT = os.environ.get("TAC_CODEX_ARM_EFFORT", "xhigh")


def resolve_arm_effort(effort: str | None) -> str:
    """Effort for one arm; refuses anything outside the operator's range."""
    value = (effort or DEFAULT_ARM_EFFORT).strip()
    if value not in ARM_EFFORT_LEVELS:
        raise SystemExit(
            f"REFUSED: arm effort {value!r} is outside the operator's range "
            f"{ARM_EFFORT_LEVELS} (codex enum: {CODEX_EFFORT_ENUM})."
        )
    return value


def assert_arm_model_admissible(model: str) -> None:
    """Fail closed on a banned model generation. Called before every spawn."""
    for banned in BANNED_ARM_MODEL_SUBSTRINGS:
        if banned in model:
            raise SystemExit(
                f"REFUSED: arm model {model!r} contains banned {banned!r}. "
                "Operator directive 2026-08-08: never spawn on 5.5 again. "
                # Cite the CANONICAL default, never the live (possibly
                # overridden) value -- under an override the live value IS the
                # banned one, and echoing it would tell the reader 5.5 is our
                # default. Same class as the guard-hook message this window.
                f"Set TAC_CODEX_ARM_MODEL to a live model "
                f"(canonical default {_DEFAULT_ARM_MODEL!r})."
            )


def charter_file_path(prompt_path: str) -> tuple[Path | None, str | None]:
    """Resolve a charter file or return a typed file-path-contract refusal."""

    contract = "charters must be files; --prompt expects a file path, not inline text"
    if not isinstance(prompt_path, str) or not prompt_path.strip():
        return None, contract
    if any(marker in prompt_path for marker in ("\n", "\r", "\x00")):
        return None, contract
    try:
        raw_path = Path(prompt_path)
        resolved = raw_path if raw_path.is_absolute() else _REPO / raw_path
        if not resolved.is_file():
            return None, f"prompt file missing ({prompt_path})"
    except (OSError, ValueError):
        # In particular, suppress ENAMETOOLONG from Path.stat()/is_file(): it
        # means an inline charter was supplied at the file-path boundary.
        return None, contract
    return resolved, None


KILL_SWITCH = "TAC_CODEX_SATURATE_OFF"
_LIVE_STATUSES = frozenset({"queued", "live"})
_NEXT_SCHEMA = "codex_arm_queue.next_if_resumed.v1"
_FINAL_SCHEMA = "codex_arm_queue.final_message.v1"
_HEADING_RX = re.compile(r"^(#{1,6})\s+(.*)$")

# --- the retraction channel (ddm_sc3, 2026-08-16) --------------------------------
# THE DEFECT this closes: a plan row's ``row_id`` is minted from
# ``(schema, arm, source_path, line_start, block_sha256)``. When a source memo is
# CORRECTED, re-extraction mints a NEW row_id and the STALE row persists forever.
# Both are then served to readers, so correcting the source could not reach the
# consumer -- there was no field on the row that could say "do not act on this".
# MEASURED cost of the gap (ddm_fb1, 2026-08-16): three live rows carried an
# admission bar of ``archive < 186,269 B`` against a 182,759 B live frontier. A
# candidate landing exactly at that bar PASSES while scoring +0.002337165 WORSE
# than what we ship -- 233.7x the 1e-5 naming bar, and SILENT.
#
# The channel is APPEND-ONLY: a retraction is a NEW row in the same store, keyed to
# the target's row_id. No existing row is ever mutated or deleted. Retraction rows
# carry their OWN schema string, so a pre-existing reader that filters on
# ``schema == _NEXT_SCHEMA`` simply does not see them -- additive by construction.
_NEXT_RETRACTION_SCHEMA = "codex_arm_queue.next_if_resumed.retraction.v1"

#: The row is dead. Do not act on it. EXCLUDED from the default plan view.
RETRACTION_SUPERSEDED = "SUPERSEDED"
#: The row is still actionable, but a NAMED clause inside it is stale. INCLUDED in
#: the default plan view, stamped with the notice -- excluding it would hide the
#: live clauses beside the dead one, which is the silent-drop disease wearing a
#: retraction's coat.
RETRACTION_AMEND_REQUIRED = "AMEND_REQUIRED"
RETRACTION_DISPOSITIONS: tuple[str, ...] = (
    RETRACTION_SUPERSEDED,
    RETRACTION_AMEND_REQUIRED,
)

#: Catalog #287 sister discipline: a placeholder rationale is not a rationale. A
#: retraction whose reason is "<reason>" records nothing and clears no debt.
_PLACEHOLDER_REASONS = frozenset(
    {
        "",
        "<reason>",
        "<rationale>",
        "reason",
        "rationale",
        "tbd",
        "todo",
        "n/a",
        "na",
        "none",
        "placeholder",
        "pending",
        "stale",
    }
)
_MIN_REASON_CHARS = 24


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO))
    except ValueError:
        return str(path)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_key(row: dict, key_fields: tuple[str, ...]) -> tuple:
    return tuple(row.get(k) for k in key_fields)


def _append_jsonl_once(path: Path, row: dict, key_fields: tuple[str, ...]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    key = _json_key(row, key_fields)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        for line in handle:
            if not line.strip():
                continue
            try:
                existing = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _json_key(existing, key_fields) == key:
                return False
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        return True


def _utcstamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _normalize_phrase(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _is_next_if_resumed_phrase(text: str) -> bool:
    alpha = re.sub(r"[^a-z]+", "", text.lower())
    return (
        alpha.startswith("nextifresumed")
        or alpha.endswith("nextifresumed")
        or ("livehypotheses" in alpha and "nextifresumed" in alpha)
    )


def _inline_next_starts_line(line: str) -> bool:
    stripped = line.strip()
    stripped = re.sub(r"^(?:[-*+>]|\d+[.)])\s+", "", stripped)
    stripped = stripped.strip("*_` ")
    return _normalize_phrase(stripped).startswith("nextifresumed")


def next_if_resumed_blocks(text: str) -> list[dict]:
    """Extract Markdown-ish NEXT_IF_RESUMED blocks without inventing rows.

    Accepts the contract spellings (``NEXT_IF_RESUMED`` / ``NEXT-IF-RESUMED``)
    and the title-case receipt form (``Next If Resumed``). Heading blocks run
    until the next heading of the same or higher level. Inline contract lines
    run until the next blank line or heading.
    """
    lines = text.splitlines()
    blocks: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        heading = _HEADING_RX.match(line)
        if heading and _is_next_if_resumed_phrase(heading.group(2)):
            level = len(heading.group(1))
            j = i + 1
            while j < len(lines):
                next_heading = _HEADING_RX.match(lines[j])
                if next_heading and len(next_heading.group(1)) <= level:
                    break
                j += 1
            block_text = "\n".join(lines[i:j]).strip()
            if block_text:
                blocks.append(
                    {
                        "line_start": i + 1,
                        "line_end": j,
                        "heading": heading.group(2).strip(),
                        "text": block_text,
                    }
                )
            i = j
            continue
        if not heading and _inline_next_starts_line(line):
            j = i + 1
            while j < len(lines):
                if not lines[j].strip() or _HEADING_RX.match(lines[j]):
                    break
                j += 1
            block_text = "\n".join(lines[i:j]).strip()
            if block_text:
                blocks.append(
                    {
                        "line_start": i + 1,
                        "line_end": j,
                        "heading": "NEXT-IF-RESUMED",
                        "text": block_text,
                    }
                )
            i = j
            continue
        i += 1
    return blocks


def _infer_arm_name(path: Path) -> str:
    rel = _rel(path)
    if "arm_final_messages/" in rel:
        stem = path.stem
        match = re.match(r"(.+)_20\d{6}T\d{6}Z$", stem)
        return match.group(1) if match else stem
    for part in reversed(path.parts):
        match = re.match(r"ddm_([a-z0-9]+)_20\d{6}", part)
        if match:
            return match.group(1)
    stem = path.stem.lower()
    match = re.match(r"ddm_([a-z0-9]+)_", stem)
    if match:
        return match.group(1)
    match = re.match(r"([a-z0-9]+)[_-]", stem)
    return match.group(1) if match else stem


def _read_jsonl(path: Path) -> list[dict]:
    """Every well-formed JSON object in an append-only store, order preserved."""
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _retraction_row_id(target_row_id: str, reason: str, disposition: str) -> str:
    return _sha256_bytes(
        "|".join(
            [_NEXT_RETRACTION_SCHEMA, target_row_id, disposition, reason]
        ).encode("utf-8")
    )


def _validate_reason(reason: str) -> str:
    text = (reason or "").strip()
    if text.lower().strip(" .") in _PLACEHOLDER_REASONS or len(text) < _MIN_REASON_CHARS:
        raise ValueError(
            "retraction reason must be a real rationale of at least "
            f"{_MIN_REASON_CHARS} chars (Catalog #287 sister discipline); got {reason!r}"
        )
    return text


def load_next_if_resumed(
    path: Path = NEXT_IF_RESUMED,
    *,
    include_superseded: bool = False,
) -> list[dict]:
    """Plan rows with their retraction state resolved. DEFAULT EXCLUDES the dead ones.

    Every returned row is a COPY of the stored plan row plus three derived keys:

    * ``retraction_disposition`` -- ``None`` | ``SUPERSEDED`` | ``AMEND_REQUIRED``
      (the strongest disposition filed against it; SUPERSEDED wins a tie);
    * ``retracted`` -- bool, true when any retraction targets the row;
    * ``retractions`` -- the full retraction rows, so a reader can print WHY.

    ``SUPERSEDED`` rows are dropped unless ``include_superseded=True``.
    ``AMEND_REQUIRED`` rows are KEPT and stamped: hiding a row that still carries
    live follow-ons in order to suppress one stale clause would trade a loud defect
    for a silent one.

    Legacy-compatible: rows written before the retraction channel existed have no
    retraction filed against them, so they load exactly as before with
    ``retracted=False``.
    """
    stored = _read_jsonl(path)
    by_target: dict[str, list[dict]] = {}
    for row in stored:
        if row.get("schema") != _NEXT_RETRACTION_SCHEMA:
            continue
        target = row.get("target_row_id")
        if isinstance(target, str) and target:
            by_target.setdefault(target, []).append(row)

    out: list[dict] = []
    for row in stored:
        if row.get("schema") != _NEXT_SCHEMA:
            continue
        hits = by_target.get(str(row.get("row_id")), [])
        dispositions = {str(h.get("disposition")) for h in hits}
        if RETRACTION_SUPERSEDED in dispositions:
            disposition = RETRACTION_SUPERSEDED
        elif RETRACTION_AMEND_REQUIRED in dispositions:
            disposition = RETRACTION_AMEND_REQUIRED
        else:
            disposition = None
        if disposition == RETRACTION_SUPERSEDED and not include_superseded:
            continue
        annotated = dict(row)
        annotated["retracted"] = bool(hits)
        annotated["retraction_disposition"] = disposition
        annotated["retractions"] = hits
        out.append(annotated)
    return out


def next_if_resumed_debt(path: Path = NEXT_IF_RESUMED) -> dict:
    """The retraction LEDGER: what was retracted, by whom, and why.

    A retracted row is DEBT somebody must clear (re-file the plan against the live
    base, or close the chain). This is the surface-on-request half of the channel --
    without it, retraction would silently shrink the queue and nobody would ever
    learn that a fire order went stale. A skip that reports nothing reads as green.
    """
    rows = load_next_if_resumed(path, include_superseded=True)
    superseded = [r for r in rows if r["retraction_disposition"] == RETRACTION_SUPERSEDED]
    amend = [r for r in rows if r["retraction_disposition"] == RETRACTION_AMEND_REQUIRED]
    return {
        "surface": _rel(path),
        "plan_rows_total": len(rows),
        "plan_rows_live": len(rows) - len(superseded),
        "superseded": superseded,
        "amend_required": amend,
        "counts": {
            RETRACTION_SUPERSEDED: len(superseded),
            RETRACTION_AMEND_REQUIRED: len(amend),
        },
    }


def retract_next_if_resumed_row(
    target_row_id: str,
    *,
    reason: str,
    citation: str,
    retracted_by: str,
    disposition: str = RETRACTION_SUPERSEDED,
    path: Path = NEXT_IF_RESUMED,
) -> dict:
    """Append a retraction against an EXISTING plan row. Never mutates that row.

    Fails closed on an unknown ``target_row_id``: a retraction that targets nothing
    is a no-op that LOOKS like a cleared hazard, which is worse than no retraction
    at all. It also fails closed on a placeholder reason and an unknown disposition.
    """
    if disposition not in RETRACTION_DISPOSITIONS:
        raise ValueError(
            f"disposition must be one of {RETRACTION_DISPOSITIONS}; got {disposition!r}"
        )
    text = _validate_reason(reason)
    cite = (citation or "").strip()
    if not cite:
        raise ValueError("retraction requires a citation naming the artifact that justifies it")
    who = (retracted_by or "").strip()
    if not who:
        raise ValueError("retraction requires retracted_by")

    targets = {
        str(row.get("row_id"))
        for row in _read_jsonl(path)
        if row.get("schema") == _NEXT_SCHEMA
    }
    if target_row_id not in targets:
        raise ValueError(
            f"no plan row with row_id {target_row_id!r} in {_rel(path)} — refusing to "
            "file a retraction that targets nothing"
        )

    row = {
        "schema": _NEXT_RETRACTION_SCHEMA,
        "row_id": _retraction_row_id(target_row_id, text, disposition),
        "target_row_id": target_row_id,
        "disposition": disposition,
        "reason": text,
        "citation": cite,
        "retracted_by": who,
        "written_at_utc": datetime.now(UTC).isoformat(),
        "reader_costate_digest": "tools/costate_digest.py section_arm_next_if_resumed",
        "score_claim": False,
    }
    _append_jsonl_once(path, row, ("row_id",))
    return row


def _auto_retract_reextracted_block(
    out_path: Path,
    *,
    arm_name: str,
    source_rel: str,
    line_start: int,
    new_row_id: str,
    new_block_sha256: str,
) -> list[dict]:
    """Retract prior rows for the SAME (arm, source, line) whose block text changed.

    This is the mechanical half. Correcting a memo and re-extracting used to leave
    the pre-correction row live beside the corrected one, with no way for a reader
    to tell them apart. Now the correction itself files the retraction, so the fix
    reaches the consumer without anybody remembering to do it by hand.

    Scoped deliberately tight: same arm, same source path, same starting line, and a
    DIFFERENT block hash. A different source file is a different plan, not a
    correction, and must not be retracted by inference.
    """
    filed: list[dict] = []
    for row in _read_jsonl(out_path):
        if row.get("schema") != _NEXT_SCHEMA:
            continue
        if (
            row.get("name") != arm_name
            or row.get("source_path") != source_rel
            or row.get("line_start") != line_start
        ):
            continue
        prior_id = str(row.get("row_id"))
        if prior_id == new_row_id or row.get("block_sha256") == new_block_sha256:
            continue
        filed.append(
            retract_next_if_resumed_row(
                prior_id,
                reason=(
                    "superseded by re-extraction of the same NEXT_IF_RESUMED block from "
                    f"{source_rel}:{line_start} after the source text changed "
                    f"(block {str(row.get('block_sha256'))[:12]} -> {new_block_sha256[:12]}); "
                    "the corrected block is the live plan"
                ),
                citation=f"{source_rel}:{line_start}",
                retracted_by="tools/codex_arm_queue.py::extract_next_if_resumed",
                disposition=RETRACTION_SUPERSEDED,
                path=out_path,
            )
        )
    return filed


def extract_next_if_resumed(
    sources: list[Path],
    *,
    provenance: str,
    name: str | None = None,
    out_path: Path = NEXT_IF_RESUMED,
) -> dict:
    """Append NEXT_IF_RESUMED blocks from source files to the arm queue surface."""
    written = 0
    seen = 0
    files_with_rows = 0
    auto_retracted = 0
    for source in sources:
        try:
            text = source.read_text(encoding="utf-8", errors="replace")
            source_sha256 = _sha256_file(source)
        except OSError:
            continue
        blocks = next_if_resumed_blocks(text)
        if blocks:
            files_with_rows += 1
        arm_name = name or _infer_arm_name(source)
        source_rel = _rel(source)
        source_kind = (
            "persisted_final_message"
            if "arm_final_messages/" in source_rel
            else "arm_receipt"
        )
        for block in blocks:
            seen += 1
            block_sha256 = _sha256_bytes(block["text"].encode("utf-8"))
            row_id = _sha256_bytes(
                "|".join(
                    [
                        _NEXT_SCHEMA,
                        arm_name,
                        source_rel,
                        str(block["line_start"]),
                        block_sha256,
                    ]
                ).encode("utf-8")
            )
            row = {
                "schema": _NEXT_SCHEMA,
                "row_id": row_id,
                "name": arm_name,
                "provenance": provenance,
                "source_kind": source_kind,
                "source_path": source_rel,
                "source_sha256": source_sha256,
                "line_start": block["line_start"],
                "line_end": block["line_end"],
                "heading": block["heading"],
                "text": block["text"],
                "block_sha256": block_sha256,
                "written_at_utc": datetime.now(UTC).isoformat(),
                "reader_main_harvest": _rel(NEXT_IF_RESUMED),
                "reader_costate_digest": "tools/costate_digest.py section_arm_next_if_resumed",
                "score_claim": False,
            }
            if _append_jsonl_once(out_path, row, ("row_id",)):
                written += 1
                auto_retracted += len(
                    _auto_retract_reextracted_block(
                        out_path,
                        arm_name=arm_name,
                        source_rel=source_rel,
                        line_start=block["line_start"],
                        new_row_id=row_id,
                        new_block_sha256=block_sha256,
                    )
                )
    return {
        "sources": len(sources),
        "blocks_seen": seen,
        "written": written,
        "files_with_rows": files_with_rows,
        "auto_retracted": auto_retracted,
    }


def persist_final_message(name: str, rc: int, elapsed: int, last_path: Path | None = None) -> dict | None:
    """Copy the full codex ``-o`` final message into research custody and index it."""
    source = last_path or (RUNS / f"{name}.last.txt")
    try:
        data = source.read_bytes()
    except OSError:
        return None
    if not data.strip():
        return None
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "arm"
    FINAL_MESSAGES.mkdir(parents=True, exist_ok=True)
    dest = FINAL_MESSAGES / f"{safe_name}_{_utcstamp()}.md"
    counter = 1
    while dest.exists():
        dest = FINAL_MESSAGES / f"{safe_name}_{_utcstamp()}_{counter}.md"
        counter += 1
    dest.write_bytes(data)
    sha256 = _sha256_bytes(data)
    row = {
        "schema": _FINAL_SCHEMA,
        "name": name,
        "rc": int(rc),
        "elapsed": int(elapsed),
        "path": _rel(dest),
        "sha256": sha256,
        "source_path": _rel(source),
        "written_at_utc": datetime.now(UTC).isoformat(),
        "score_claim": False,
    }
    row["row_id"] = _sha256_bytes(
        f"{_FINAL_SCHEMA}|{row['name']}|{row['path']}|{row['sha256']}".encode()
    )
    _append_jsonl_once(FINAL_MESSAGE_INDEX, row, ("row_id",))
    extract_next_if_resumed([dest], provenance="harvested-final", name=name, out_path=NEXT_IF_RESUMED)
    return row


# --- queue state (pure-ish) ------------------------------------------------------


def load_rows(path: Path = QUEUE) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a torn tail; never crash the actuator
    return rows


def latest_by_name(rows: list[dict]) -> dict[str, dict]:
    """Latest row per name wins FIELD-BY-FIELD — status changes are appends.

    Merging, not replacing: a ``mark`` row carries only {name,status}, so a naive
    last-row-wins silently drops ``prompt_path`` and the charter becomes unspawnable
    (observed 2026-08-04: ``would spawn: fz4 ()``). Later rows override the fields
    they actually set; everything else survives from the row that set it.
    """
    out: dict[str, dict] = {}
    for row in rows:
        name = row.get("name")
        if isinstance(name, str) and name:
            merged = dict(out.get(name, {}))
            merged.update({k: v for k, v in row.items() if v is not None})
            out[name] = merged
    return out


def append_row(row: dict, path: Path = QUEUE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault("ts", time.time())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_arm_names(ps_output: str) -> set[str]:
    """Extract arm names from `ps` command lines via the -o receipt path.

    Split from the process listing so it is testable against a real fixture line —
    the earlier version asserted only `isinstance(..., set)`, which passes on a
    permanently broken function (and it WAS broken: see live_arm_names).
    """
    names: set[str] = set()
    for line in ps_output.splitlines():
        if "codex exec" not in line:
            continue
        for token in line.replace("'", " ").replace('"', " ").split():
            if token.endswith(".last.txt"):
                names.add(Path(token).name[: -len(".last.txt")])
    return names


def live_arm_names() -> set[str]:
    """Names of codex arms with a running process, read from the OS not the ledger.

    The ledger can lie (an arm dies without marking itself — fz3 did exactly that);
    the process table cannot.

    `ps -eo pid,command`, NOT `pgrep -af`: macOS pgrep has no `-a` flag, silently
    prints bare PIDs, and the name parser could never match — so this returned an
    empty set on every call while four arms were running (MEASURED 2026-08-04).
    A liveness detector that always reports zero makes the cap vacuous.
    """
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid,command"], capture_output=True, text=True, timeout=15
        )
    except Exception:
        return set()
    return parse_arm_names(proc.stdout)


def next_charters(rows: list[dict], live: set[str], slots: int, scorer_taken: bool) -> list[dict]:
    """Rank-ordered charters that may fire right now, honouring the scorer rule.

    A live-marked row with NO process but a CLEAN ``rc=0`` exit receipt is a
    FINISHED arm awaiting harvest — NOT died-resumable. Auto-respawning it
    re-runs a completed charter (measured 2026-08-04: rf1 respawned after its
    harvest because the row was never marked landed; the duplicate had to be
    killed). Clean finishes need an explicit ``mark --status queued`` (or a
    fresh ``add``) to run again; only receipt-less or nonzero/signal exits
    remain implicitly resumable.
    """
    latest = latest_by_name(rows)

    def _finished_clean(r: dict) -> bool:
        if r.get("status") != "live":
            return False
        receipt = _done_receipt(r.get("name", ""))
        return receipt is not None and receipt.startswith("rc=0")

    ready = [
        r
        for r in latest.values()
        if r.get("status") in _LIVE_STATUSES
        and r.get("name") not in live
        and not _finished_clean(r)
    ]
    ready.sort(key=lambda r: (r.get("rank", 999), r.get("name", "")))
    picked: list[dict] = []
    for row in ready:
        if len(picked) >= slots:
            break
        if row.get("owns_scorer"):
            if scorer_taken:
                continue  # one full-n600 job at a time, fleet-wide
            scorer_taken = True
        picked.append(row)
    return picked


# --- spawning --------------------------------------------------------------------


# BSD-correct detach: fork + setsid(2) + exec. `nohup ... & disown` is NOT
# sufficient — disown clears the shell's JOB TABLE but the child stays in the
# shell's process group AND gets reparented to PID 1 when the tool-shell exits.
# macOS has NO setsid(1), so it must be done in Python.
#
# ROOT CAUSE, MEASURED 2026-08-04 via the exit receipts: the killer of every
# arm generation (nohup 11:13, setsid 11:34, receipted 11:44 — signal=TERM at
# elapsed 335/337/337 s) is ~/Library/LaunchAgents/com.vertigo.claude-code-reaper
# → ~/Projects/fleet/scripts/claude-code-reaper.sh, a launchd agent firing every
# 60 s that SIGTERMs any process matching \b(claude|codex)\b with no TTY and
# (PPID==1 or stdin in {null,pipe}) older than 300 s. Differential proof: a
# plain-bash control detached by the IDENTICAL fork+setsid shim survived to
# natural completion — the harness reaps nothing; the reaper kills by NAME.
_DETACH_PY = (
    "import os,sys\n"
    "log=sys.argv[1]; cmd=sys.argv[2:]\n"
    "if os.fork()>0: os._exit(0)\n"
    "os.setsid()\n"
    "fd=os.open(log,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o644)\n"
    "os.dup2(fd,1); os.dup2(fd,2)\n"
    "os.dup2(os.open(os.devnull,os.O_RDONLY),0)\n"
    "os.execvp(cmd[0],cmd)\n"
)


def keeper_path(name: str) -> str:
    return f".omx/tmp/codex_runs/{name}_keeper.py"


def keeper_source(name: str, prompt_path: str, effort: str | None = None) -> str:
    """Source of the per-arm KEEPER — the reaper-proof supervisor.

    The reaper kills on: name-match \\b(claude|codex)\\b AND no-TTY AND
    (PPID==1 OR stdin in {/dev/null, pipe}) AND age>300s. The keeper breaks two
    conjuncts at once, using only plain POSIX facts:

      * The keeper IS the detached setsid leader, and its ps line is
        ``python3 .omx/tmp/codex_runs/<name>_keeper.py`` — ``codex_runs`` has
        no word boundary (underscore is a word char), so the reaper's grep
        never examines it at all.
      * codex runs as the keeper's normal CHILD (PPID != 1) with stdin bound
        to a REGULAR FILE — the reaper's stdin_is_dead() flags only
        /dev/null|PIPE|FIFO, so a REG-file stdin reads as a live session.

    The keeper also owns the EXIT RECEIPT at ``<name>.done``:

        rc=0                 -> codex finished cleanly
        rc=N (N>0)           -> codex exited with an error of its own
        signal=TERM|INT|HUP|QUIT -> something reaped the keeper (handler fired)
        no .done file        -> SIGKILL, or the keeper itself vanished

    Python signal handlers interrupt ``proc.wait()`` immediately, so the
    bash foreground-trap-deferral class (round-2 finding: fg traps are
    serviced only between commands — exactly the reap case wrote no receipt)
    cannot recur here.
    """
    instruction = (
        f"Read and execute the charter at {prompt_path} in full, plus the common "
        f"contract it points to at .omx/tmp/codex_runs/_common_contract.md. "
        f"Follow every constraint in both.\n\n{RETAINED_REASONING}"
    )
    relay_instruction = (
        f"CONTINUATION generation %d of arm {name}: your predecessor ran out of "
        f"context mid-charter. Its durable state is on disk — FIRST read the newest "
        f"receipts under .omx/research matching this arm's receipt dir (named in the "
        f"charter) and run `git log --oneline -15` to see work already committed. Do "
        f"NOT redo completed work; resume from the newest NEXT-IF-RESUMED block (or "
        f"infer the frontier from receipts if absent). CONTEXT LAW: write receipts "
        f"incrementally, commit early via the serializer, never accumulate large "
        f"outputs or web-search results in context. Then continue executing the "
        f"charter at {prompt_path} plus the common contract at "
        f".omx/tmp/codex_runs/_common_contract.md.\n\n{RETAINED_REASONING}"
    )
    # Model + effort are BOTH resolved here, per-arm (operator 2026-08-08: a
    # 5.6-sol "of high to ultra effort levels depending on the task at hand").
    # Effort is a per-task CHOICE, no longer a hardcoded xhigh. OPERATOR LAW
    # (2026-08-05): every CONVOCATION arm (gc*/pantheon passes) runs at the
    # MAXIMUM tier -- now literally expressible as effort="ultra". For the MOST
    # IMPORTANT convocations (operator-flagged or route-changing adjudications),
    # MAIN ALSO runs a parallel FABLE leg (Agent tool, model:"fable" carve-out)
    # on the same charter and reconciles both receipts.
    assert_arm_model_admissible(ARM_MODEL)
    resolved_effort = resolve_arm_effort(effort)
    argv_prefix = [
        "codex", "exec", "--skip-git-repo-check", "-s", "workspace-write",
        *_ssd_add_dir_args(),
        "-m", ARM_MODEL, "-c", f"model_reasoning_effort={resolved_effort}",
        "-o", f".omx/tmp/codex_runs/{name}.last.txt",
    ]
    return (
        "# Auto-generated per-arm keeper — see tools/codex_arm_queue.py:keeper_source.\n"
        "# RELAY: on context-exhaustion the keeper relaunches codex with a fresh\n"
        "# context + disk-state continuation header (operator directive 2026-08-04:\n"
        "# arms must exceed one context and run autonomously for however long it takes).\n"
        "import os, shutil, signal, subprocess, sys, time\n"
        f"NAME = {name!r}\n"
        f"ARGV_PREFIX = {argv_prefix!r}\n"
        f"INSTRUCTION = {instruction!r}\n"
        f"RELAY_INSTRUCTION = {relay_instruction!r}\n"
        "MAX_GEN = 12\n"
        "CONTEXT_SIGS = ('ran out of room in the model\\'s context window',\n"
        "                'exceeds the context window')\n"
        "CAPACITY_SIGS = ('Selected model is at capacity',)\n"
        "DONE = '.omx/tmp/codex_runs/' + NAME + '.done'\n"
        "STDIN_PATH = '.omx/tmp/codex_runs/' + NAME + '.stdin'\n"
        "LOG = '.omx/tmp/codex_runs/' + NAME + '.log'\n"
        "LAST = '.omx/tmp/codex_runs/' + NAME + '.last.txt'\n"
        "RELAY_LOG = '.omx/tmp/codex_runs/' + NAME + '.relay'\n"
        "start = time.time()\n"
        "open(STDIN_PATH, 'ab').close()  # REGULAR file: reaper reads it as a live stdin\n"
        "stdin_f = open(STDIN_PATH, 'rb')\n"
        "log_f = open(LOG, 'ab')\n"
        "proc = None\n"
        "def _mk(name_):\n"
        "    def h(signum, frame):\n"
        "        try:\n"
        "            if proc is not None:\n"
        "                proc.terminate()\n"
        "        except Exception:\n"
        "            pass\n"
        "        with open(DONE, 'w') as f:\n"
        "            f.write('signal=%s elapsed=%d\\n' % (name_, int(time.time() - start)))\n"
        "        os._exit(143)\n"
        "    return h\n"
        "for s in ('TERM', 'INT', 'HUP', 'QUIT'):\n"
        "    signal.signal(getattr(signal, 'SIG' + s), _mk(s))\n"
        "def _head():\n"
        "    try:\n"
        "        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()\n"
        "    except Exception:\n"
        "        return ''\n"
        "def _log_tail():\n"
        "    try:\n"
        "        with open(LOG, 'rb') as f:\n"
        "            f.seek(max(0, os.path.getsize(LOG) - 8000))\n"
        "            return f.read().decode('utf-8', 'replace')\n"
        "    except Exception:\n"
        "        return ''\n"
        "def _research_mtime():\n"
        "    newest = 0.0\n"
        "    for root, _dirs, files in os.walk('.omx/research'):\n"
        "        for fn in files:\n"
        "            try:\n"
        "                newest = max(newest, os.path.getmtime(os.path.join(root, fn)))\n"
        "            except OSError:\n"
        "                pass\n"
        "    return newest\n"
        "gen = 1\n"
        "rc = 1\n"
        "cap_tries = 0\n"
        "while True:\n"
        "    gen_head, gen_mtime, gen_start = _head(), _research_mtime(), time.time()\n"
        "    prompt = INSTRUCTION if gen == 1 else (RELAY_INSTRUCTION % gen)\n"
        "    proc = subprocess.Popen(ARGV_PREFIX + [prompt], stdin=stdin_f, stdout=log_f, stderr=log_f)\n"
        "    rc = proc.wait()\n"
        "    if rc == 0:\n"
        "        break\n"
        "    tail = _log_tail()\n"
        "    if cap_tries < 3 and any(sig in tail for sig in CAPACITY_SIGS):\n"
        "        cap_tries += 1\n"
        "        with open(RELAY_LOG, 'a') as f:\n"
        "            f.write('capacity backoff try=%d rc=%d sleep=%d\\n' % (cap_tries, rc, 300 * cap_tries))\n"
        "        time.sleep(300 * cap_tries)\n"
        "        continue\n"
        "    exhausted = any(sig in tail for sig in CONTEXT_SIGS)\n"
        "    progressed = (_head() != gen_head) or (_research_mtime() > gen_mtime)\n"
        "    if not exhausted or gen >= MAX_GEN or (gen >= 2 and not progressed):\n"
        "        break\n"
        "    try:\n"
        "        shutil.copyfile(LAST, LAST + '.gen%d' % gen)\n"
        "    except OSError:\n"
        "        pass\n"
        "    with open(RELAY_LOG, 'a') as f:\n"
        "        f.write('relay gen=%d->%d rc=%d gen_elapsed=%d progressed=%s\\n'\n"
        "                % (gen, gen + 1, rc, int(time.time() - gen_start), progressed))\n"
        "    gen += 1\n"
        "elapsed = int(time.time() - start)\n"
        "try:\n"
        "    subprocess.run([sys.executable, 'tools/codex_arm_queue.py', 'persist-final',\n"
        "                    '--name', NAME, '--rc', str(rc), '--elapsed', str(elapsed),\n"
        "                    '--last', LAST], check=False)\n"
        "except Exception:\n"
        "    pass\n"
        "with open(DONE, 'w') as f:\n"
        "    f.write('rc=%d elapsed=%d gen=%d\\n' % (rc, elapsed, gen))\n"
        "sys.exit(rc)\n"
    )


def spawn_command(name: str, prompt_path: str) -> str:
    """The canonical detached spawn: fork+setsid shim exec'ing the KEEPER.

    Reaper-shape invariant (pinned by tests): this command string contains NO
    standalone ``codex``/``claude`` word — the codex argv lives inside the
    keeper FILE, which ps never shows. ``prompt_path`` is intentionally not in
    the command either; the keeper embeds the full instruction.
    """
    q = shlex.quote
    log = f".omx/tmp/codex_runs/{name}_keeper.log"
    return " ".join(["python3 -c", q(_DETACH_PY), q(log), "python3", q(keeper_path(name))])


def spawn(name: str, prompt_path: str, effort: str | None = None) -> bool:
    RUNS.mkdir(parents=True, exist_ok=True)
    # Fail closed on a banned generation BEFORE any file is written or process
    # forked -- a refusal must not leave a half-written keeper behind.
    assert_arm_model_admissible(ARM_MODEL)
    resolved_effort = resolve_arm_effort(effort)
    _prompt_file, refusal = charter_file_path(prompt_path)
    if refusal is not None:
        print(f"  REFUSED {name}: {refusal}", file=sys.stderr)
        return False
    for advisory in lint_charter_capability_advisories(str(_prompt_file)):
        # Advisory by construction: never turn a measured sandbox limitation
        # into a mechanism verdict or a spawn refusal.  The warning names the
        # MAIN handoff route instead.
        print(f"charter-lint WARN [{name}]: {advisory}")
    # Clear STALE evidence from a previous generation: a leftover `.done`
    # (death receipt) or `.last.txt` (clean-finish marker) would corrupt the
    # next death-vs-completion read — the exact ambiguity the receipt exists
    # to remove. Confirmed by executed control 2026-08-04 review round.
    for stale in (RUNS / f"{name}.done", RUNS / f"{name}.last.txt"):
        stale.unlink(missing_ok=True)
    (_REPO / keeper_path(name)).write_text(
        keeper_source(name, prompt_path, resolved_effort), encoding="utf-8"
    )
    subprocess.run(["bash", "-c", spawn_command(name, prompt_path)], cwd=_REPO, check=False)
    append_row({
        "name": name, "prompt_path": prompt_path, "status": "live", "event": "spawned",
        # Model+effort on the SPAWN row: what actually ran is a receipt, not an
        # inference from whatever the constants happen to say when you read back.
        "model": ARM_MODEL, "effort": resolved_effort,
    })
    try:
        SPAWN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SPAWN_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ts": time.time(), "name": name, "prompt": prompt_path}) + "\n")
    except Exception:
        pass
    return True


# --- CLI -------------------------------------------------------------------------


def _done_receipt(name: str) -> str | None:
    """Terminal receipt line for an arm, or None if no ``.done`` exists.

    The old status labeled every live-marked-but-processless arm ``[DIED]``
    without reading receipts — rc=0 finishes showed as deaths (the known
    instrument mislabel). The ``.done`` file is the keeper's exit receipt and
    is the authority on how the arm actually ended.
    """
    try:
        content = (RUNS / f"{name}.done").read_text(errors="replace").strip()
    except OSError:
        return None
    return content.splitlines()[-1] if content else "(empty receipt)"


def _watcher_line() -> str:
    """Fleet-watcher liveness from the heartbeat file (codex_arm_watch.py).

    Surfaced at every queue interaction so an unarmed watcher is visible —
    without it, arm completions silently revert to poll-on-request.
    """
    hb = RUNS / "_watcher.alive"
    try:
        age = time.time() - hb.stat().st_mtime
    except OSError:
        return (
            "fleet watcher: NOT RUNNING — arm completions will NOT notify MAIN "
            "(arm via Monitor: .venv/bin/python tools/codex_arm_watch.py, persistent)"
        )
    if age > 90:  # codex_arm_watch.HEARTBEAT_STALE_S
        return f"fleet watcher: STALE (heartbeat {int(age)}s ago) — re-arm the Monitor"
    # Liveness is NOT delivery. Measured 2026-08-08: a detached watcher kept the
    # heartbeat at 0s all session (this line read ALIVE/green) while MAIN received
    # ZERO notifications — its stdout was a file, not the Monitor's pipe. A green
    # indicator for a condition nobody checked is the vacuity genus; report the
    # DELIVERY CHANNEL the watcher now stamps into the heartbeat.
    channel = ""
    try:
        for row in hb.read_text().splitlines():
            if row.startswith("channel="):
                channel = row.split("=", 1)[1].strip()
    except OSError:
        channel = ""
    # socket = the harness Monitor's channel (MEASURED); fifo = a shell pipe.
    # Both are IPC to a consuming parent. Accepting only "fifo" false-alarmed on the
    # CORRECT config — a false alarm on the good state trains the reader to ignore the light.
    if channel in ("socket", "fifo"):
        return f"fleet watcher: ALIVE + DELIVERING (stdout={channel}; heartbeat {int(age)}s ago)"
    if channel:
        return (
            f"fleet watcher: RUNNING but NOT DELIVERING (stdout={channel}, not a Monitor pipe; "
            f"heartbeat {int(age)}s ago) — arm it as a persistent Monitor or MAIN gets no "
            "arm-completion notifications"
        )
    return (
        f"fleet watcher: ALIVE (heartbeat {int(age)}s ago) — delivery channel UNKNOWN "
        "(pre-channel watcher; restart it to report whether MAIN actually receives events)"
    )


def _surface_line() -> str:
    return (
        f"final messages: {_rel(FINAL_MESSAGES)} ; "
        f"NEXT_IF_RESUMED surface: {_rel(NEXT_IF_RESUMED)}"
    )


def cmd_status(args) -> int:
    rows = load_rows()
    live = live_arm_names()
    latest = latest_by_name(rows)
    # Report the DENOMINATOR (m50): a charter marked `live` whose process is gone is
    # SPAWNABLE, and listing only status=="queued" hid exactly those — status said
    # "live 0, queued 4" while saturate was about to fire four arms in neither count.
    spawnable = sorted(
        (
            r
            for r in latest.values()
            if r.get("status") in _LIVE_STATUSES and r.get("name") not in live
        ),
        key=lambda r: (r.get("rank", 999), r.get("name", "")),
    )
    # Split processless-live rows by their .done receipt: FINISHED-unharvested
    # (keeper wrote an exit receipt) vs truly DIED (no receipt at all).
    n_finished = sum(
        1
        for r in spawnable
        if r.get("status") == "live" and _done_receipt(r.get("name", "")) is not None
    )
    n_stale = (
        sum(1 for r in spawnable if r.get("status") == "live") - n_finished
    )
    scorer_live = any(latest.get(n, {}).get("owns_scorer") for n in live)
    print(f"codex arms live: {len(live)}/{args.cap}  {sorted(live) if live else ''}")
    print(f"scorer slot: {'TAKEN' if scorer_live else 'free'}")
    print(_watcher_line())
    print(_surface_line())
    print(f"spawnable charters: {len(spawnable)} of {len(latest)} tracked", end="")
    tail = []
    if n_finished:
        tail.append(f"{n_finished} FINISHED (unharvested — read .done)")
    if n_stale:
        tail.append(f"{n_stale} marked live but DEAD — resumable")
    print(f"  ({'; '.join(tail)})" if tail else "")
    for row in spawnable[: args.limit]:
        flag = " [SCORER]" if row.get("owns_scorer") else ""
        stale = ""
        if row.get("status") == "live":
            receipt = _done_receipt(row.get("name", ""))
            stale = f" [FINISHED {receipt}]" if receipt is not None else " [DIED]"
        print(
            f"  rank {row.get('rank', '?'):>3}  {row.get('name')}{flag}{stale}"
            f" — {row.get('note', '')[:80]}"
        )
    gap = max(0, args.cap - len(live))
    print(f"SATURATION GAP: {gap} slot(s) open" if gap else "SATURATED")
    return 0


_BUILD_CHARTER_TOKENS = ("build", "race", "train", "implement", "measure", "solve", "compose")


def _charter_is_build_by_tokens(text: str) -> bool:
    return any(t in text.lower() for t in _BUILD_CHARTER_TOKENS)


def _fm_advisory_module():
    """Return the Pact fmtools advisory module only when it is importable and available.

    Any import/model/venv failure degrades to None so queue output is byte-identical on hosts
    without fmtools. The returned module is used only for WARN lines, never refusals.
    """
    try:
        from tac import fm_advisory as _fm
    except Exception:
        return None
    try:
        return _fm if _fm.available() else None
    except Exception:
        return None


def lint_charter_fm_advisories(prompt_path: str) -> list[str]:
    """Warn-only fmtools enrichment for charter lint.

    Deterministic checks remain the gate. These lines are intentionally separated from
    ``lint_charter_optimal_form`` so TAC_CHARTER_LINT_STRICT never upgrades an FM result
    into a refusal.
    """
    fm = _fm_advisory_module()
    if fm is None:
        return []
    try:
        text = Path(prompt_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    warnings: list[str] = []
    is_build = _charter_is_build_by_tokens(text)
    try:
        cls = fm.charter_class(text, timeout=15)
    except Exception:
        cls = None
    if cls:
        label = cls.get("charter_class") or "unknown"
        note = ""
        if label in {"build_race_train_measure", "mixed"} and not is_build:
            note = "; deterministic build-token gate did not fire"
        elif label in {"audit_analysis", "convocation"} and is_build:
            note = "; deterministic build-token gate fired"
        rationale = str(cls.get("rationale") or "").strip()
        tail = f" ({rationale[:120]})" if rationale else ""
        warnings.append(f"fmtools advisory charter_class={label}{note}{tail}")

    try:
        reduction = fm.mechanism_reduction_language(text, timeout=15)
    except Exception:
        reduction = None
    if reduction and reduction.get("flags"):
        flags = ", ".join(str(f) for f in reduction.get("flags", []))
        warnings.append(f"fmtools advisory mechanism_reduction_language={flags}")
    return warnings


def lint_charter_capability_advisories(
    prompt_path: str, registry_path: Path = ARM_CAPABILITIES
) -> list[str]:
    """Warn when a charter demands a capability denied on the arm surface.

    This is deliberately read-only and advisory.  A missing or malformed
    registry produces one visible warning but never blocks queueing or spawn.
    """

    try:
        text = Path(prompt_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"capability registry unavailable ({exc})"]
    if not isinstance(payload, dict) or payload.get("schema") != "pact.codex_arm_capabilities.v1":
        return ["capability registry schema differs; denied-capability advice unavailable"]
    warnings: list[str] = []
    for row in payload.get("capabilities", []):
        if not isinstance(row, dict) or row.get("status") != "denied_in_codex_arm_sandbox":
            continue
        patterns = row.get("demand_patterns", [])
        if not isinstance(patterns, list):
            continue
        try:
            demanded = any(re.search(str(pattern), text, re.IGNORECASE) for pattern in patterns)
        except re.error as exc:
            warnings.append(
                f"capability registry pattern invalid for {row.get('capability_family', 'unknown')} ({exc})"
            )
            continue
        if not demanded:
            continue
        evidence = row.get("evidence", [])
        evidence_paths = [
            str(item.get("path"))
            for item in evidence
            if isinstance(item, dict) and item.get("path")
        ]
        warnings.append(
            "charter demands denied arm capability "
            f"{row.get('capability_family', 'unknown')!r}; "
            f"MAIN-handoff: {row.get('main_handoff', 'route to MAIN')}; "
            f"measured receipt: {', '.join(evidence_paths) or 'registry row'}"
        )
    return warnings


# ---------------------------------------------------------------------------
# Charter-time RECALL + VALIDATION advisories (operator 2026-08-16:
# "Need to be more proactive about recall and validation prior to issuing
# charters").  Three arms found stale premises in charters MAIN wrote from
# working memory on one day: gx1 (five stale premises, incl. a rate rung quoted
# against a superseded archive and a target already executed 12h earlier by
# td1), pv1 (an entire charter duplicating live sibling ps1u), sx1 (four,
# incl. a cure chartered as unbuilt that rg1b had landed that morning).
#
# CLAUDE.md OPERATOR PRIORITY item 1 already binds PROACTIVE RECALL before
# designing/proposing/killing.  It was volitional, so it depended on MAIN
# remembering to remember — the goldfish failure the law itself names.  This
# moves the recall to the APPARATUS: the checks run at the spawn site, so the
# charter is measured against the corpus whether or not MAIN thought to look.
#
# Advisory (WARN) by construction.  The measured failure was NOT-CHECKING, and
# a printed warning at spawn already breaks that chain; a refusal would trade a
# cheap miss for an expensive false block.
# ---------------------------------------------------------------------------

RESEARCH_DIR = _REPO / ".omx" / "research"
CORRECTIONS_INDEX = RESEARCH_DIR / "ddm_au1_20260805" / "au1_corrections_index.jsonl"
FRONTIER_POINTER = _REPO / ".omx" / "state" / "canonical_frontier_pointer.json"

# Claims that assert something does NOT exist / has NOT happened.  Per memory
# `negative-existence claims = #1 false-claim class`, these need an exhaustive
# search or an explicit scope — at charter time nobody has done either.
_NEGATIVE_EXISTENCE = re.compile(
    r"\b("
    r"never (?:been |yet )?(?:run|fired|measured|built|tried|attempted|executed|owned)"
    r"|has never|have never|was never|were never"
    r"|un-?owned|unbuilt|un-?built|unmeasured|un-?fired"
    r"|no(?:body| arm| one| agent)? (?:has|have|ever)"
    r"|not (?:yet )?(?:owned|built|measured|run|fired)"
    r"|nobody has|no arm has|first(?:-| )of(?:-| )family"
    r")\b",
    re.IGNORECASE,
)

# Stopwords that carry no discriminating power when matching a charter claim
# against the recent memo corpus.
_RECALL_STOPWORDS = frozenset(
    """
    charter measure measured measurement should would could there their these those
    through against because before after within number result results verdict
    should_be family families arm arms which where whose while about above below
    return returns running current currently already always never nothing
    """.split()
)


def _distinctive_tokens(text: str) -> set[str]:
    """Tokens with enough specificity to join a charter claim to a memo.

    Hyphens normalize to underscores because charters write ``token-drop``
    while memo filenames write ``token_drop`` — the 2026-08-16 miss.  Compound
    parts are emitted separately so a 2-word compound still joins when the
    memo spells it differently.
    """

    normalized = text.replace("-", "_")
    tokens: set[str] = set()
    # Snake_case / arm-style identifiers are the highest-precision join key.
    for match in re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", normalized, re.IGNORECASE):
        low = match.lower()
        tokens.add(low)
        for part in low.split("_"):
            if len(part) >= 4 and part not in _RECALL_STOPWORDS:
                tokens.add(part)
    # Bare words: 5+ chars catches Schur/token/drop-class subjects that the
    # earlier 6-char floor dropped, while the stopword set holds precision.
    for match in re.findall(r"\b[a-zA-Z]{5,}\b", normalized):
        low = match.lower()
        if low not in _RECALL_STOPWORDS:
            tokens.add(low)
    return tokens


def _recent_memos(days: int, limit: int = 400) -> list[Path]:
    """Memos touched inside the window, newest first, bounded."""

    if not RESEARCH_DIR.is_dir():
        return []
    cutoff = time.time() - days * 86400
    rows: list[tuple[float, Path]] = []
    for path in RESEARCH_DIR.rglob("*.md"):
        # Worktree copies duplicate main's memos and would double-count.
        if ".omx/tmp/codex_worktrees" in str(path):
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= cutoff:
            rows.append((mtime, path))
    rows.sort(key=lambda row: row[0], reverse=True)
    return [path for _, path in rows[:limit]]


def _lint_ownership(text: str, days: int) -> list[str]:
    """Is the charter's 'nobody has done this' target already owned?

    The gx1/pv1/sx1 class: MAIN charters a target as un-owned/unbuilt while a
    memo from the last few days already executed it.  Joins on distinctive
    tokens drawn from the negative-existence sentence itself.
    """

    claims = [
        line.strip()
        for line in text.splitlines()
        if _NEGATIVE_EXISTENCE.search(line)
    ]
    if not claims:
        return []
    claim_tokens = set()
    for claim in claims:
        claim_tokens |= _distinctive_tokens(claim)
    if len(claim_tokens) < 2:
        return []
    # PROPORTIONAL threshold, not a tuned constant: a claim with more
    # distinctive terms must match more of them.  A flat cut calibrated on one
    # example is exactly the constants-are-poison trap this lint exists to
    # catch.  Measured on the 2026-08-16 incident (10 claim tokens -> cut 6):
    # 5 hits, and all five ARE the arms that reported the stale premise; the
    # overlap histogram falls away steeply below that (16 at 5, 26 at 4).
    threshold = max(4, math.ceil(0.55 * len(claim_tokens)))
    hits: list[tuple[int, str]] = []
    for memo in _recent_memos(days):
        try:
            # Head-only: titles/headlines/abstracts carry the subject, and a
            # bounded read keeps the spawn-site cost flat.
            head = memo.read_text(encoding="utf-8", errors="replace")[:8192].lower()
        except OSError:
            continue
        overlap = sum(1 for token in claim_tokens if token in head)
        if overlap >= threshold:
            hits.append((overlap, memo.name))
    if not hits:
        return []
    hits.sort(reverse=True)
    top = "; ".join(f"{name} ({n} shared terms)" for n, name in hits[:3])
    return [
        f"RECALL: charter asserts a negative-existence claim ({claims[0][:110]!r}) "
        f"but {len(hits)} memo(s) from the last {days}d overlap its subject — {top}. "
        "Read them before spawning; if the target is already owned, re-aim or "
        "cite the memo as the baseline."
    ]


def _lint_stale_numbers(text: str) -> list[str]:
    """Does the charter quote a number the corpus already corrected?

    Consumes au1's corrections index (task #953) rather than building a second
    one.  A charter literal that appears as some memo's `refuted_value` is the
    gx1 class (-15,157 B quoted against a superseded archive).
    """

    if not CORRECTIONS_INDEX.is_file():
        return []
    literals = {
        token.replace(",", "")
        for token in re.findall(r"\b\d[\d,]{3,}(?:\.\d+)?\b", text)
    }
    if not literals:
        return []
    warnings: list[str] = []
    seen: set[str] = set()
    try:
        with CORRECTIONS_INDEX.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                refuted = str(row.get("refuted_value", "")).replace(",", "")
                if not refuted or refuted not in literals or refuted in seen:
                    continue
                seen.add(refuted)
                warnings.append(
                    f"RECALL: charter quotes {refuted} — recorded as a REFUTED value in "
                    f"{Path(str(row.get('source', '?'))).name}"
                    f" (corrected to {row.get('corrected_value', '?')}). Re-derive before citing."
                )
                if len(warnings) >= 5:
                    break
    except OSError:
        return []
    return warnings


def _lint_frontier_literals(text: str) -> list[str]:
    """Frontier-shaped numbers must match the LIVE pointer, not a memory of it.

    pv1 measured this twice on one charter: a d_pose literal belonging to a
    different archive, and a frontier S the pointer file no longer carries.
    """

    if not FRONTIER_POINTER.is_file():
        return []
    try:
        pointer = json.loads(FRONTIER_POINTER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    live: set[str] = set()
    for value in pointer.values():
        if isinstance(value, dict) and "score" in value:
            live.add(f"{float(value['score']):.10f}")
    if not live:
        return []
    # Contest S literals: 0.1x-0.9x carried to >=6 decimals (a quoted score,
    # never an incidental ratio).
    quoted = {
        f"{float(token):.10f}"
        for token in re.findall(r"\b0\.\d{6,}\b", text)
        if 0.05 < float(token) < 1.0
    }
    stale = quoted - live
    if not stale:
        return []
    return [
        "RECALL: charter quotes score-shaped literal(s) "
        f"{sorted(stale)[:3]} that match NO anchor in the live "
        "canonical_frontier_pointer.json — re-derive from the pointer "
        "(a superseded frontier is the commonest stale premise)."
    ]


def _lint_bare_task_ids(text: str) -> list[str]:
    """Bare #NNNN ids do not resolve for an arm; memo filenames do.

    Reported independently by gx1, pv1 and sx1 on 2026-08-16: the harness task
    list and the repo's canonical_task_status.jsonl are different ledgers, and
    arms only see the repo.
    """

    ids = set(re.findall(r"#\d{3,4}\b", text))
    if not ids:
        return []
    if re.search(r"\b[\w./-]+\.md\b", text):
        return []
    return [
        f"RECALL: charter cites bare task ids {sorted(ids)[:5]} and NO memo filename — "
        "arms cannot resolve harness ids against the repo ledger. Cite memo "
        "filenames (and shas) instead."
    ]


def lint_charter_recall_advisories(prompt_path: str, days: int = 14) -> list[str]:
    """Charter-time recall/validation advisories (operator 2026-08-16).

    Four legs, all advisory, all read-only: ownership (is the target already
    executed?), staleness (does a quoted number appear as a refuted value?),
    live-pointer agreement (are frontier-derived numbers current?), and
    id-resolvability (bare task ids vs memo filenames).  Any leg may fail to
    read its store; that produces silence, never a block.
    """

    try:
        text = Path(prompt_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if "recall_lint_na:" in text.lower():
        return []
    out: list[str] = []
    for leg in (
        lambda: _lint_ownership(text, days),
        lambda: _lint_stale_numbers(text),
        lambda: _lint_frontier_literals(text),
        lambda: _lint_bare_task_ids(text),
    ):
        try:
            out.extend(leg())
        except Exception as exc:  # advisory: a broken leg must never block a spawn
            out.append(f"recall-lint leg unavailable ({type(exc).__name__}: {exc})")
    return out


def lint_charter_optimal_form(prompt_path: str) -> list[str]:
    """Charter-time toy guard (operator 2026-08-06 'naive first pass' correction).

    Naive/toy implementations are born at CHARTER time — every downstream audit
    (ty1 citation guards, vo2 form-grade gate, Catalog #307) fires only after a
    toy has produced a verdict. This lint fires at the birth point: a charter
    that builds/races a mechanism must either carry an '## OPTIMAL FORM' block
    (family reference cited + scope-vs-mechanism delta + provenance pins) or an
    explicit 'OPTIMAL_FORM_NA:<rationale>' waiver (audits/convocations/pure
    analysis). MVP-first reduces SCOPE (n, epochs), never MECHANISM fidelity.
    Warn-only by default; TAC_CHARTER_LINT_STRICT=1 refuses (legacy respawn
    charters predate the law, so strict is opt-in until conformance sweep).
    """
    problems: list[str] = []
    try:
        text = Path(prompt_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"charter unreadable ({exc})"]
    low = text.lower()
    has_block = "## optimal form" in low
    waiver = "optimal_form_na:" in low
    if waiver:
        rationale = low.split("optimal_form_na:", 1)[1].splitlines()[0].strip()
        if len(rationale) < 8 or rationale.startswith("<"):
            problems.append("OPTIMAL_FORM_NA waiver rationale is placeholder/too short")
        return problems
    is_build = _charter_is_build_by_tokens(text)
    if not is_build:
        return problems
    if not has_block:
        problems.append(
            "build/race charter lacks '## OPTIMAL FORM' block "
            "(family reference + scope-vs-mechanism delta + provenance pins) "
            "and no OPTIMAL_FORM_NA:<rationale> waiver"
        )
        return problems
    section = low.split("## optimal form", 1)[1]
    section = section.split("\n## ", 1)[0]
    if "reference" not in section and "receipt" not in section:
        problems.append("OPTIMAL FORM block cites no family reference/receipt")
    # {7,64}: 7+ truncated git hash, 40 full git SHA-1, 64 full SHA-256. The
    # earlier {7,40} bound REJECTED a full SHA-256 — the leading \b forces the
    # match to start at the token's first char, 40 chars cannot reach char 64,
    # and every shorter ending lands mid-token where there is no \b — so the
    # lint punished the STRONGER pin while accepting a 7-char truncation
    # (task #1002, measured 2026-08-10).
    if not re.search(r"\b[0-9a-f]{7,64}\b", section):
        problems.append("OPTIMAL FORM block carries no sha/commit provenance pin")
    # Negative-signal accounting (operator 2026-08-15: "Anything that doesn't
    # account for all of our negative signal is naive or toy to a certain
    # extent"). A build charter citing NO prior negative (dead-end / refusal /
    # refutation / no-go / parked family) designs blind to the measured
    # failure corpus — naive by construction regardless of mechanism grade.
    # Coarse token instrument, warn-only; NEGATIVES_NA:<rationale> waives
    # genuinely first-of-family work with no bearing negatives.
    if "negatives_na:" in low:
        rationale = low.split("negatives_na:", 1)[1].splitlines()[0].strip()
        if len(rationale) < 8 or rationale.startswith("<"):
            problems.append("NEGATIVES_NA waiver rationale is placeholder/too short")
    elif not re.search(r"dead[- ]end|refus|refut|no[- ]go|negative|parked", low):
        problems.append(
            "charter accounts for no prior negative signal "
            "(operator 2026-08-15: unaccounted negatives = naive/toy) — cite "
            "the bearing dead-ends/refusals or NEGATIVES_NA:<rationale>"
        )
    return problems


def cmd_add(args) -> int:
    prompt_file, refusal = charter_file_path(args.prompt)
    if refusal is not None:
        print(f"REFUSED {args.name}: {refusal}", file=sys.stderr)
        return 2
    assert prompt_file is not None
    problems = lint_charter_optimal_form(str(prompt_file))
    advisories = [
        *lint_charter_capability_advisories(str(prompt_file)),
        *lint_charter_fm_advisories(str(prompt_file)),
        *lint_charter_recall_advisories(str(prompt_file)),
    ]
    if problems:
        # STRICT BY DEFAULT (operator 2026-08-13 "No naive or toy ever"): a charter
        # that fails the optimal-form lint is refused at the spawn site. The explicit
        # escape is TAC_CHARTER_LINT_STRICT=0 — a tracked waiver, never a silent default.
        strict = os.environ.get("TAC_CHARTER_LINT_STRICT", "1") != "0"
        tag = "REFUSED" if strict else "WARN"
        for p in problems:
            print(f"charter-lint {tag} [{args.name}]: {p}")
        for p in advisories:
            print(f"charter-lint WARN [{args.name}]: {p}")
        if strict:
            return 3
    else:
        for p in advisories:
            print(f"charter-lint WARN [{args.name}]: {p}")
    append_row(
        {
            "name": args.name,
            "prompt_path": args.prompt,
            "rank": args.rank,
            "owns_scorer": bool(args.owns_scorer),
            "status": "queued",
            "note": args.note or "",
            # Per-task effort, chosen at queue time (operator 2026-08-08).
            "effort": resolve_arm_effort(getattr(args, "effort", None)),
        }
    )
    print(
        f"queued {args.name} (rank {args.rank}, "
        f"effort {resolve_arm_effort(getattr(args, 'effort', None))})"
    )
    return 0


def cmd_lint(args) -> int:
    prompt_file, refusal = charter_file_path(args.prompt)
    if refusal is not None:
        print(f"REFUSED lint: {refusal}", file=sys.stderr)
        return 2
    assert prompt_file is not None
    problems = lint_charter_optimal_form(str(prompt_file))
    for problem in problems:
        print(f"charter-lint REFUSED [{args.name}]: {problem}")
    for advisory in (
        *lint_charter_capability_advisories(str(prompt_file)),
        *lint_charter_fm_advisories(str(prompt_file)),
        *lint_charter_recall_advisories(str(prompt_file)),
    ):
        print(f"charter-lint WARN [{args.name}]: {advisory}")
    return 3 if problems else 0


def cmd_mark(args) -> int:
    append_row({"name": args.name, "status": args.status, "event": "mark"})
    print(f"{args.name} -> {args.status}")
    return 0


def cmd_saturate(args) -> int:
    if os.environ.get(KILL_SWITCH) == "1":
        print(f"saturation OFF ({KILL_SWITCH}=1)")
        return 0
    rows = load_rows()
    live = live_arm_names()
    latest = latest_by_name(rows)
    gap = max(0, args.cap - len(live))
    if not gap:
        print(f"SATURATED ({len(live)}/{args.cap}) — nothing to fire")
        return 0
    scorer_taken = any(latest.get(n, {}).get("owns_scorer") for n in live)
    picks = next_charters(rows, live, gap, scorer_taken)
    if not picks:
        print(f"GAP {gap} but QUEUE EMPTY — feed the queue (codex_arm_queue.py add ...)")
        return 0
    for row in picks:
        name, prompt = row.get("name"), row.get("prompt_path", "")
        if not args.spawn:
            print(f"  would spawn: {name} ({prompt})")
            continue
        if spawn(name, prompt, row.get("effort")):
            print(f"  spawned {name}")
            time.sleep(2)
    if not args.spawn:
        print("dry run — pass --spawn to actually fire")
    else:
        print(_watcher_line())
    return 0


def cmd_persist_final(args) -> int:
    row = persist_final_message(args.name, args.rc, args.elapsed, Path(args.last))
    if row is None:
        print(f"no final message persisted for {args.name}: {args.last}")
        return 1
    print(json.dumps(row, sort_keys=True))
    return 0


def cmd_extract_next(args) -> int:
    sources = [Path(p) for p in args.source]
    for pattern in args.source_glob:
        sources.extend(sorted(_REPO.glob(pattern)))
    summary = extract_next_if_resumed(
        sources,
        provenance=args.provenance,
        name=args.name,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


def cmd_retract(args) -> int:
    try:
        row = retract_next_if_resumed_row(
            args.row_id,
            reason=args.reason,
            citation=args.citation,
            retracted_by=args.by,
            disposition=args.disposition,
        )
    except ValueError as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(json.dumps(row, sort_keys=True))
    return 0


def cmd_next(args) -> int:
    rows = load_next_if_resumed(include_superseded=args.include_superseded)
    debt = next_if_resumed_debt()
    print(
        f"plan rows: {len(rows)} shown | live {debt['plan_rows_live']} of "
        f"{debt['plan_rows_total']} | superseded {debt['counts'][RETRACTION_SUPERSEDED]} | "
        f"amend-required {debt['counts'][RETRACTION_AMEND_REQUIRED]}"
    )
    for row in rows[-args.limit :]:
        flag = row["retraction_disposition"] or "live"
        print(f"  [{flag}] {row.get('name')} {row.get('source_path')}:{row.get('line_start')}")
        for hit in row["retractions"]:
            print(f"      ! {hit.get('disposition')}: {hit.get('reason')}")
    if not args.include_superseded and debt["counts"][RETRACTION_SUPERSEDED]:
        print("  (pass --include-superseded to see the retracted rows and their reasons)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status")
    p.add_argument("--limit", type=int, default=12)
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("add")
    p.add_argument("--name", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--rank", type=int, default=100)
    p.add_argument("--owns-scorer", action="store_true")
    p.add_argument("--note", default="")
    p.add_argument(
        "--effort",
        default=None,
        choices=list(ARM_EFFORT_LEVELS),
        help=(
            "reasoning effort for THIS arm, chosen per task (operator 2026-08-08: "
            f"high->ultra). Default {DEFAULT_ARM_EFFORT}."
        ),
    )
    p.set_defaults(fn=cmd_add)
    p = sub.add_parser("mark")
    p.add_argument("--name", required=True)
    p.add_argument("--status", required=True, choices=["queued", "live", "landed", "dropped"])
    p.set_defaults(fn=cmd_mark)
    p = sub.add_parser("lint")
    p.add_argument("--prompt", required=True)
    p.add_argument("--name", default="charter")
    p.set_defaults(fn=cmd_lint)
    p = sub.add_parser("saturate")
    p.add_argument("--spawn", action="store_true")
    p.set_defaults(fn=cmd_saturate)
    p = sub.add_parser("persist-final")
    p.add_argument("--name", required=True)
    p.add_argument("--rc", type=int, required=True)
    p.add_argument("--elapsed", type=int, required=True)
    p.add_argument("--last", required=True)
    p.set_defaults(fn=cmd_persist_final)
    p = sub.add_parser("extract-next")
    p.add_argument("--source", action="append", default=[])
    p.add_argument("--source-glob", action="append", default=[])
    p.add_argument("--provenance", default="manual")
    p.add_argument("--name")
    p.set_defaults(fn=cmd_extract_next)
    p = sub.add_parser(
        "retract",
        help=(
            "file an APPEND-ONLY retraction against a NEXT_IF_RESUMED plan row so a "
            "correction at the source reaches the readers that serve it"
        ),
    )
    p.add_argument("--row-id", required=True)
    p.add_argument("--reason", required=True, help="a real rationale; placeholders are refused")
    p.add_argument("--citation", required=True, help="the artifact that justifies the retraction")
    p.add_argument("--by", required=True, help="who filed it (arm name or tool path)")
    p.add_argument(
        "--disposition",
        default=RETRACTION_SUPERSEDED,
        choices=list(RETRACTION_DISPOSITIONS),
        help=(
            f"{RETRACTION_SUPERSEDED}: row is dead, hidden by default. "
            f"{RETRACTION_AMEND_REQUIRED}: row still actionable, one named clause stale."
        ),
    )
    p.set_defaults(fn=cmd_retract)
    p = sub.add_parser("next", help="show live NEXT_IF_RESUMED plan rows + retraction debt")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--include-superseded", action="store_true")
    p.set_defaults(fn=cmd_next)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
