#!/usr/bin/env python3
# no-argparse-OK: library module; the executable studies that consume it own their own argv
"""The ADVISORY fmtools lane — one bridge, reused by every census in this repo.

Why this exists
---------------
Three regex censuses in the 2026-09-04 wave each missed something a regex
structurally cannot see:

1. Catalog #344's finding token ``ratified`` matched inside ``stratified`` --
   16 of 29 flagged memos (55.2%) were false positives (ddm_eq1, MEASURED).
2. The GT-lineage gate's artifact regexes accept only ``.npy`` / ``.pt``, so the
   ``.npz`` PyAV table the born trainer pins as authority is invisible to it
   (ddm_bh1 finding 2).
3. ddm_ql3's provenance-comment scan missed retired n96 constants that carried
   no provenance comment; only a value fingerprint found them.

Each miss is a CLASSIFICATION judgement about prose or code intent. This module
is the second lane for those judgements: the Apple on-device Foundation Model,
through our own ``fmtools`` (``~/Projects/fmtools``), reached by SUBPROCESS under
the fmtools virtualenv so this repo's venv gains zero dependencies.

The firewall (binding, from CLAUDE.md and the fmtools capability memory)
------------------------------------------------------------------------
* **ADVISORY ONLY.** An FM label is never a score, a verdict, a promotion, a
  kill, or a blocking authority. It is a second opinion printed beside a
  deterministic verdict. Disagreements are LOGGED, never enforced. Sister of the
  MPS/MLX never-a-score firewall.
* **FAIL-OPEN.** Missing venv, missing model, timeout, non-zero exit, unparseable
  output -- all yield "no advice" and the deterministic verdict stands alone,
  labelled honestly as such. Never a stub, never a fabricated label.
* **PRE-FILTERED CANDIDATES ONLY.** The model is ~0.4-1.1 s per call. It runs on
  the handful of items a deterministic pre-filter already surfaced, never per
  line and never on a latency-critical or per-turn path.
* **$0, on-device, no network.**

Since fmtools 0.0.219 the subprocess contract is a supported CLI
(``fmtools classify``: JSON Lines in, JSON Lines out, documented exit codes,
explicit fail-open), so callers no longer hand-roll the schema/asyncio/
error-swallowing dance. This module is the thin pact-side adapter to it.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass

__all__ = [
    "DEFAULT_TIMEOUT_S",
    "DISAGREEMENT_LOG",
    "AdvisoryVerdict",
    "classify_texts",
    "fmtools_python",
    "log_disagreements",
    "unavailable_label",
]

#: Where lane disagreements accumulate. This is the whole point of running two
#: lanes: the deterministic verdict always stands, and the disagreement becomes
#: a durable row someone can read later instead of a print that scrolls away.
DISAGREEMENT_LOG = ".omx/state/fmtools_advisory_disagreements.jsonl"

#: Wall-clock ceiling for the WHOLE batch subprocess, seconds. Sized for a few
#: dozen pre-filtered candidates at roughly one second each, plus interpreter
#: start-up. A batch that exceeds it fails open rather than wedging the caller.
DEFAULT_TIMEOUT_S: float = 300.0

#: Interpreter search order. Env overrides first so a caller can point at a
#: different checkout without editing code.
_PYTHON_ENV_VARS = ("PACT_FMTOOLS_PYTHON", "DASH_FM_PYTHON")
_PYTHON_DEFAULT = "~/Projects/fmtools/.venv/bin/python"


@dataclass(frozen=True, slots=True)
class AdvisoryVerdict:
    """The outcome of one advisory batch.

    Attributes:
        labels: ``{id: label}`` for every item the model actually labelled.
            Items it could not label are simply absent -- a caller must treat a
            missing id as "no advice", never as a negative.
        ran: True iff the fmtools subprocess executed and returned parseable
            output. When False the caller MUST label its report
            "fmtools confirmation owed" rather than implying the model agreed.
        reason: Why it did not run, when ``ran`` is False; ``None`` otherwise.
        errors: ``{id: error}`` for items the batch returned but could not label
            (timeout, guardrail refusal, out-of-set answer). Observability only.
    """

    labels: dict[str, str]
    ran: bool
    reason: str | None = None
    errors: dict[str, str] | None = None

    def label_for(self, item_id: str, *, default: str = "no_advice") -> str:
        """Return the label for ``item_id``, or ``default`` when there is none."""
        return self.labels.get(item_id, default)


def fmtools_python() -> str | None:
    """Return the fmtools virtualenv interpreter, or ``None`` when absent.

    ``None`` means the advisory lane is genuinely ABSENT -- callers fail open and
    say so. It never means "the model said no".
    """
    candidates = [os.environ.get(var) for var in _PYTHON_ENV_VARS]
    candidates.append(_PYTHON_DEFAULT)
    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.expanduser(candidate)
        if os.path.exists(path):
            return path
    return None


def unavailable_label(prefix: str = "") -> str:
    """The honest sentence to print when the advisory lane did not run."""
    return (
        f"{prefix}fmtools advisory lane did NOT run (on-device model or venv "
        "unavailable) -- the deterministic verdict stands alone and its "
        "fmtools confirmation is OWED. This is not agreement."
    )


def log_disagreements(
    lane: str,
    rows: list[dict],
    *,
    repo_root: str | None = None,
    log_path: str = DISAGREEMENT_LOG,
) -> int:
    """Append lane disagreements to the JSONL ledger. Never raises.

    Args:
        lane: Which census produced these (e.g. ``"catalog_344"``).
        rows: One dict per disagreement. Free-form, but each SHOULD carry the
            deterministic verdict and the advisory label so a reader can see
            which lane said what without re-running either.
        repo_root: Repository root; defaults to this file's parent's parent.
        log_path: Ledger path, relative to ``repo_root``.

    Returns:
        Number of rows written; 0 on any failure (this is observability, and
        observability must never break the thing it observes).
    """
    if not rows:
        return 0
    try:
        import datetime

        root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target = os.path.join(root, log_path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with open(target, "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps({"utc": stamp, "lane": lane, **row}, sort_keys=True) + "\n"
                )
        return len(rows)
    except Exception:
        return 0


def classify_texts(
    items: dict[str, str],
    *,
    labels: list[str],
    instruction: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_chars: int = 4000,
    python_executable: str | None = None,
) -> AdvisoryVerdict:
    """Classify pre-filtered texts through the on-device model. Never raises.

    Args:
        items: ``{id: text}``. Ids are echoed back; text is truncated by fmtools
            to ``max_chars``.
        labels: The closed label set. fmtools constrains generation to it, and
            an answer outside it comes back as an error rather than a label.
        instruction: The system instruction. State what each label means AND
            what must NOT trigger the positive one -- that sentence is where the
            lane's precision comes from.
        timeout_s: Ceiling for the whole batch.
        max_chars: Per-item truncation passed through to fmtools.
        python_executable: Override the interpreter (tests, alternate checkout).

    Returns:
        An :class:`AdvisoryVerdict`. Every failure mode yields ``ran=False`` with
        a ``reason``; nothing in this function propagates an exception.
    """
    if not items:
        return AdvisoryVerdict({}, False, "no candidates")

    interpreter = python_executable or fmtools_python()
    if not interpreter:
        return AdvisoryVerdict({}, False, "fmtools venv not found")

    argv = [interpreter, "-m", "fmtools.cli", "classify", "--max-chars", str(max_chars)]
    for label in labels:
        argv += ["--label", label]
    argv += ["--instruction", instruction]

    payload = "".join(
        json.dumps({"id": str(item_id), "text": str(text)}) + "\n"
        for item_id, text in items.items()
    )
    try:
        proc = subprocess.run(
            argv, input=payload, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        return AdvisoryVerdict({}, False, f"timeout after {timeout_s:.0f}s")
    except OSError as exc:
        return AdvisoryVerdict({}, False, f"subprocess failed: {exc}")

    # fmtools exits 0 under its default fail-open policy even when individual
    # rows failed, so a non-zero code here means the batch itself was refused.
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        tail = detail[-1] if detail else "no stderr"
        return AdvisoryVerdict({}, False, f"exit {proc.returncode}: {tail}")

    parsed: dict[str, str] = {}
    errors: dict[str, str] = {}
    saw_a_row = False
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        saw_a_row = True
        row_id = str(row.get("id", ""))
        if row.get("ok") and isinstance(row.get("label"), str):
            parsed[row_id] = row["label"]
        else:
            errors[row_id] = str(row.get("error", "unknown"))

    if not saw_a_row:
        return AdvisoryVerdict({}, False, "fmtools produced no rows")
    return AdvisoryVerdict(parsed, True, None, errors or None)
