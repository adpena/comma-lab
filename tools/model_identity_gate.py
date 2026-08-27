#!/usr/bin/env python
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""Model-identity HARD-FAIL gate (operator 2026-07-21: "if you're not fable, it hard
fails, and I know instantly").

Wired as a Claude Code **UserPromptSubmit** hook. On every prompt it reads the session
transcript (the only authoritative record of which model actually generated each turn —
hook stdin does NOT carry the live model, and the agent's own system-prompt self-ID can
be STALE across mid-session /model switches, as measured 2026-07-21), finds the most
recent MAIN-THREAD assistant turn's model ID, and compares it to the operator's saved
default model in ~/.claude/settings.json.

Verdicts:
  MISMATCH  -> exit 2 (hard block; stderr is shown to the operator INSTANTLY and the
               prompt is NOT processed by the wrong model).
  MATCH     -> exit 0, silent.
  UNKNOWN   -> exit 0 with a loud stdout warning (first turn of a fresh session has no
               assistant entries yet; an unreadable/format-changed transcript must not
               brick every session — the warning IS the signal in that case).

Requirement source = the operator's OWN saved default (`model` in ~/.claude/settings.json,
written by /model). This auto-tracks intent: an intentional /model change updates the
default so the gate follows; a silent harness reroute does NOT touch settings.json so the
gate fires. Optional pin override: .omx/state/required_model (single token) beats settings.

Intentional bypass (operator-only): env TAC_MODEL_GATE_ALLOW=1 or touch
.omx/state/model_gate_allow — loudly acknowledged, never silent.

Matching is case-insensitive substring of the required token in the actual model ID
("fable" matches "claude-fable-5"). Sidechain (subagent) and "<synthetic>" entries are
ignored — only main-thread turns count.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PIN_FILE = REPO / ".omx" / "state" / "required_model"
ALLOW_FILE = REPO / ".omx" / "state" / "model_gate_allow"
ALLOW_ENV = "TAC_MODEL_GATE_ALLOW"


def _required_token() -> str:
    """Pin file beats the client saved default; empty string disables (warn)."""
    try:
        pin = PIN_FILE.read_text().strip()
        if pin:
            return pin
    except OSError:
        pass
    try:
        cfg = json.loads((Path.home() / ".claude" / "settings.json").read_text())
        return str(cfg.get("model", "") or "").strip()
    except Exception:
        return ""


def _last_main_thread_model(transcript_path: str) -> str:
    """Most recent main-thread assistant model ID, '' if none/unreadable.

    Reads the tail of the file (last 512 KB) — transcripts grow large and the newest
    entries are what matter; a partial first line from mid-file seek is skipped by the
    per-line JSON parse guard.
    """
    try:
        p = Path(transcript_path)
        size = p.stat().st_size
        with open(p, "rb") as f:
            if size > 512 * 1024:
                f.seek(size - 512 * 1024)
                f.readline()  # discard partial line
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    model = ""
    for line in tail.splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "assistant" or d.get("isSidechain"):
            continue
        m = str(((d.get("message") or {}).get("model")) or "")
        if m and m != "<synthetic>":
            model = m  # keep last seen = most recent
    return model


def main() -> int:
    if os.environ.get(ALLOW_ENV) == "1" or ALLOW_FILE.exists():
        print(
            "[model-gate] BYPASS ACTIVE (operator kill-switch) — model identity NOT enforced this prompt."
        )
        return 0

    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}
    transcript = str(payload.get("transcript_path", "") or "")

    required = _required_token()
    if not required:
        print("[model-gate] WARN: no required model configured (settings.json 'model' empty, no pin) — not enforced.")
        return 0

    actual = _last_main_thread_model(transcript) if transcript else ""
    if not actual:
        print(
            f"[model-gate] WARN: no main-thread assistant turn observable yet (fresh session or unreadable "
            f"transcript) — cannot verify model this prompt; required={required!r}. The gate enforces from "
            f"the 2nd prompt onward."
        )
        return 0

    if required.lower() in actual.lower():
        return 0  # match — silent

    sys.stderr.write(
        f"\n{'=' * 72}\n"
        f"MODEL IDENTITY HARD FAIL (operator directive 2026-07-21)\n"
        f"  required : {required!r} (from /model saved default; pin: {PIN_FILE})\n"
        f"  actual   : {actual!r} (last main-thread turn in transcript)\n"
        f"  The main thread is NOT running the required model — likely a silent\n"
        f"  harness fallback/reroute. Prompt BLOCKED before the wrong model acts.\n"
        f"  Fix: restart the session (default is {required!r}), or intentionally\n"
        f"  bypass with {ALLOW_ENV}=1 / touch {ALLOW_FILE}.\n"
        f"{'=' * 72}\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
