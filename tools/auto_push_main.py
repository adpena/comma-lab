#!/usr/bin/env python3
"""Stop-hook: auto-push `main` to origin at turn boundaries — so pushing is structural, not remembered.

Operator directive 2026-07-09: *"We should regularly commit and push to main moving forward because we
will have another agent working on an orthogonal data viz Marimo project and we want it to have full
context into the amazing work you are doing."* + *"We could add an automatic hook so you don't have to
think about it."*

This fires on the Claude Code **Stop** hook (end of each assistant turn — a natural batch boundary). The
common case (main ahead of origin, diff clean) pushes silently. The rare case (the diff trips the
secret/fleet-IP scan) HOLDS the push and logs a loud warning instead — never leak. It is **fail-open**:
any error exits 0 so it can never block a turn; the worst failure mode is "did not push" (safe direction),
never "pushed something it shouldn't" or "wedged the session".

Guardrails (from CLAUDE.md "Public Disclosure Hygiene" + the standing memory
`regularly-commit-push-main-for-parallel-marimo-agent`): contest IP is open-source (contest closed) but
OPERATIONAL hygiene stays — Tailscale-fleet IPs (100.64.0.0/10 CGNAT), credentials, private keys, API
tokens must never reach the public `origin/main`. The scan below is the gate; a hit holds the push.

Design decisions:
  * Only ever touches branch `main` (the sole source of truth). Any other checked-out branch => no-op.
  * Cheap no-op when not ahead: one local `git rev-list --count` — no network unless there's work to push.
  * Never rebases/merges autonomously. A non-fast-forward rejection (someone else pushed) is logged for
    manual reconcile, not silently force-resolved.
  * Skips while a rebase/merge is mid-flight (an inconsistent HEAD must not be pushed).
  * Durable log at .omx/state/auto_push.log (append; never /tmp).

Usage: wired as a Stop hook in .claude/settings.json. `--dry-run` reports the decision without pushing.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Loop-safety marker: the HEAD we last surfaced a HOLD/push-failure for, so we
# warn at most once per drift state (belt-and-suspenders with stop_hook_active).
_MARKER = ".omx/state/auto_push_marker.json"


def _repo_root() -> Path:
    import os
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env and Path(env, ".git").exists():
        return Path(env)
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,  # subprocess-no-check-OK: best-effort repo-root discovery; empty output falls through to the path fallback
                             text=True, timeout=10).stdout.strip()
        if out:
            return Path(out)
    except Exception:
        pass
    return Path(__file__).resolve().parents[1]


def _git(root: Path, *args: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True, timeout=timeout)


# Secret / fleet-hygiene patterns. A HIT holds the push. Tuned to be specific (require a key-name prefix
# or a known token shape) so ordinary code/sha256 hashes do not false-positive; when unsure the safe
# direction is to HOLD (fail-safe), and the operator pushes manually after a glance.
_SCAN = [
    ("tailscale_fleet_ip", re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b")),
    ("private_key_block", re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("generic_secret_assignment", re.compile(
        r"(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|vast_api_key|password|passwd)"
        r"\s*[=:]\s*[\"']?[A-Za-z0-9/+_\-]{16,}")),
]


# Files excluded from the outgoing-diff scan: they legitimately contain secret PATTERNS + secret-SHAPED
# test fixtures by design (this scanner and its tests). Excluding self is the gitleaks-allowlist pattern;
# any NEW file that must carry example secrets should be added here (and hand-reviewed).
_SCAN_EXCLUDE_PATHSPECS = ("tools/auto_push_main.py", "src/tac/tests/test_auto_push_main.py",
                           ".gitleaks.toml")  # gitleaks config documents the CGNAT range (100.64.0.0/10)


def scan_diff(diff_text: str) -> list[tuple[str, str]]:
    """Return [(pattern_name, redacted_sample), ...] for every scan hit in the diff (added+context)."""
    hits: list[tuple[str, str]] = []
    for name, rx in _SCAN:
        m = rx.search(diff_text)
        if m:
            s = m.group(0)
            redacted = (s[:6] + "…" + s[-2:]) if len(s) > 10 else "…"
            hits.append((name, redacted))
    return hits


def _log(root: Path, msg: str) -> None:
    try:
        import datetime
        line = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')} {msg}\n"
        p = root / ".omx" / "state" / "auto_push.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(line)
    except Exception:
        pass


def _marker_head(root: Path) -> str:
    try:
        return json.loads((root / _MARKER).read_text()).get("warned_head", "")
    except Exception:
        return ""


def _write_marker(root: Path, head: str) -> None:
    try:
        import datetime
        p = root / _MARKER
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "warned_head": head,
            "at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        }))
    except Exception:
        pass


# --- fmtools ADVISORY second layer (on-device Apple FM; tighten-only; NEVER an authority) ------------
# The deterministic _SCAN regex above is the AUTHORITATIVE floor (it alone decides push-vs-hold on the
# known high-risk shapes). This layer is a semantic SECOND OPINION for NOVEL/contextual secrets the regex
# cannot pattern-match. Firewall (memory `reference-apple-ondevice-fm-fmtools-classifier-capability`,
# #259): the on-device FM is ADVISORY, NEVER a score/verdict authority. Here it is allowed to ADD a hold
# (tighten) but can NEVER permit a push the regex blocked, and its absence/timeout/error ⇒ regex-only
# (fail-open to push). A held push is a REVERSIBLE, cheap safety pause (one manual `git push`), so for a
# leak gate — where a false-negative = a PUBLIC leak but a false-positive = one manual push — letting the
# FM tighten is the correct risk posture, and it does not make the FM a verdict authority.
# Isolation per the established dashboard_fm_events / triality_drift_detector pattern: the FM runs under
# the SEPARATE fmtools venv in a SUBPROCESS — the pact venv gains ZERO deps.
_FM_SECRET_SCRIPT = r'''
import asyncio, json, sys
try:
    import apple_fm_sdk as fm
    from fmtools import local_extract
except Exception:
    print("{}"); raise SystemExit(0)

@fm.generable()
class SecretCheck:
    verdict: str = fm.guide(anyOf=["secret", "clean"],
        description="'secret' if the text contains real sensitive content that must not be public; else 'clean'.")
    category: str = fm.guide(anyOf=["none", "credential", "private_ip_or_host", "api_key_or_token", "private_key", "other"],
        description="The kind of sensitive content found, or 'none'.")
    reason: str = fm.guide(description="A short phrase naming what was found, or empty.")

@local_extract(SecretCheck, retries=1, instructions=(
    "You inspect ADDED source-diff lines about to be pushed to a PUBLIC GitHub repo and decide if they "
    "contain a REAL secret that must not be public. Return 'secret' ONLY when you are CONFIDENT the text "
    "contains an actual credential in usable form: a real password, API key, access token, a private-key "
    "block, or a private/internal infrastructure IP (Tailscale CGNAT 100.64.0.0/10) or internal hostname "
    "with a REAL value. Return 'clean' for everything else, INCLUDING: ordinary source code; config keys "
    "with no real value (e.g. api_key = os.environ['API_KEY']); documentation; math, metrics and "
    "hyperparameters (e.g. 'Pearson 0.978', loss terms, d_seg); sha256/hex hashes; public URLs; and "
    "placeholders (<tailscale-ip>, REDACTED, example.com). This is an OPEN-SOURCE research project — its "
    "research CONTENT (math, methods, results, ideas) is PUBLISHED ON PURPOSE and is ALWAYS 'clean'. When "
    "uncertain, return 'clean': a deterministic regex is the safety floor for known secret shapes, and you "
    "are a PRECISION layer for real credentials it cannot pattern-match. Examples -> "
    "\"DATABASE_URL = postgres://user:S3cr3tPw9x@host/db\": secret (embedded password). "
    "\"Pearson correlation 0.978 measured on n600\": clean (research metric). "
    "\"api_key = os.environ['API_KEY']\": clean (no real value). "
    "\"ssh into 100.101.102.103\": secret (private fleet IP). "
    "\"border-radius: 0; color: #5ab0ff\": clean (CSS)."))
async def _check(diff_added_text: str) -> SecretCheck:
    """(instructions above)"""

async def _main():
    try:
        text = sys.stdin.read()[:6000]
    except Exception:
        print("{}"); return
    if not text.strip():
        print("{}"); return
    try:
        r = await _check(text)
        print(json.dumps({
            "contains_secret": (str(getattr(r, "verdict", "") or "").lower() == "secret"),
            "category": str(getattr(r, "category", "") or ""),
            "reason": str(getattr(r, "reason", "") or ""),
        }))
    except Exception:
        print("{}")

asyncio.run(_main())
'''


def _fm_python() -> str | None:
    """The fmtools-venv interpreter (env override → DASH_FM_PYTHON → the known default).
    None when absent — the advisory layer is then silently OFF (regex-only)."""
    import os
    for cand in (os.environ.get("AUTO_PUSH_FM_PYTHON"),
                 os.environ.get("DASH_FM_PYTHON"),
                 os.path.expanduser("~/Projects/fmtools/.venv/bin/python")):
        if cand and os.path.exists(cand):
            return cand
    return None


# Literals that appear ONLY in the FM prompt's few-shot exemplars above. A small on-device model can
# ECHO its own exemplar back as a "finding" (observed 2026-07-09: flagged the fake DATABASE_URL exemplar
# on a diff that contained no such string — 3rd instance of the self-reference class: regex exclusions →
# gitleaks allowlist → the FM prompt itself). A finding whose reason cites an exemplar literal that is
# NOT present in the scanned text is a prompt echo, not a detection, and is discarded.
_FM_PROMPT_EXEMPLARS = ("S3cr3tPw9x", "100.101.102.103")


def _fm_prompt_echo(reason: str, scanned_text: str) -> bool:
    """True when the FM's reason quotes a prompt-exemplar literal absent from the scanned text
    (hallucinated echo). If the diff GENUINELY contains the exemplar string, it still flags."""
    return any(ex in (reason or "") and ex not in scanned_text for ex in _FM_PROMPT_EXEMPLARS)


def fm_secret_advisory(added_text: str, timeout: float = 12.0) -> dict | None:
    """Run the on-device FM secret classifier on the diff's ADDED lines. Returns a dict
    {contains_secret, category, reason} or None (fail-silent on any absence/error/timeout).
    Prompt-echo findings (see _fm_prompt_echo) are downgraded to clean with a marker reason."""
    fm_py = _fm_python()
    if not fm_py or not added_text.strip():
        return None
    try:
        proc = subprocess.run([fm_py, "-c", _FM_SECRET_SCRIPT], input=added_text,
                             capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return None
        out = json.loads(proc.stdout.strip() or "{}")
        if not (isinstance(out, dict) and out):
            return None
        if out.get("contains_secret") and _fm_prompt_echo(str(out.get("reason", "")), added_text):
            out = {**out, "contains_secret": False, "category": "none",
                   "reason": f"fm-prompt-echo-discarded: {out.get('reason', '')}"}
        return out
    except Exception:
        return None


def _added_lines(diff_text: str, cap: int = 6000) -> str:
    """The '+' added lines of a unified diff (drop the +++ header lines), capped for the FM window."""
    added = [ln[1:] for ln in diff_text.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    return "\n".join(added)[:cap]


# --- gitleaks: third DETERMINISTIC layer (curated ruleset + entropy + our Tailscale rule) ------------
# Authoritative like the regex — a finding HOLDS the push. gitleaks' default ruleset subsumes + far
# exceeds our inline regex (hundreds of curated credential patterns + entropy); we add ONE rule it lacks
# (Tailscale CGNAT) via .gitleaks.toml, which also carries the self-exclusion allowlist (scanner + tests +
# config). MEASURED 2026-07-09: 0 false-positives on our sha256 provenance over 40 commits — precise
# enough to hold authoritatively. Absent binary / absent config / error / timeout ⇒ None ⇒ skipped
# (fail-open; the regex + fmtools still gate). Runs only on push-eligible turns (ahead>0), ~200ms.
def gitleaks_scan(root: Path, timeout: float = 60.0) -> list[str] | None:
    """Scan the outgoing commits (origin/main..main) with gitleaks + the repo .gitleaks.toml. Returns a
    list of 'RuleID @ File' finding strings (HOLD), [] if clean, or None (skip/fail-open) when gitleaks or
    its config is absent, or gitleaks errors/times out. Never raises. Secrets are --redact'd — we surface
    only rule id + file, never the value."""
    import os
    import shutil
    import tempfile
    if not shutil.which("gitleaks") or not (root / ".gitleaks.toml").exists():
        return None
    report = None
    try:
        tmpdir = root / ".omx" / "tmp"
        tmpdir.mkdir(parents=True, exist_ok=True)
        fd, report = tempfile.mkstemp(prefix="gitleaks_", suffix=".json", dir=str(tmpdir))
        os.close(fd)
        proc = subprocess.run(
            ["gitleaks", "git", "--log-opts=origin/main..main", "-c", ".gitleaks.toml",
             "--no-banner", "--redact", "-f", "json", "-r", report],
            cwd=str(root), capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            return []
        if proc.returncode != 1:  # >1 = gitleaks internal error, NOT a clean verdict → fail-open
            return None
        try:
            data = json.loads(Path(report).read_text() or "[]")
        except Exception:
            return ["gitleaks-finding (report unparseable)"]
        out = [f"{f.get('RuleID', '?')} @ {f.get('File', '?')}" for f in (data if isinstance(data, list) else [])]
        return out or ["gitleaks-finding"]
    except Exception:
        return None
    finally:
        if report:
            try:
                Path(report).unlink()
            except Exception:
                pass


def _block_reason(verdict: dict) -> str:
    """The transcript-surfaced message for a HOLD / push-failure (needs my action)."""
    if verdict.get("action") == "hold":
        return (
            f"Auto-push HELD: the outgoing diff (origin/main..{verdict.get('ahead')} commit(s)) tripped "
            f"the operational-hygiene scan — {verdict.get('hits')}. NOT pushed (fail-safe: never leak a "
            f"fleet IP / credential to the public remote). Review the flagged content; if benign, push by "
            f"hand (git push origin main). Kill-switch: touch .omx/state/auto_push.disabled. "
            f"Details in .omx/state/auto_push.log."
        )
    return (
        f"Auto-push FAILED (git push origin main → rc={verdict.get('rc')}): {verdict.get('detail')}. "
        f"{_derived_push_failure_cause()} "
        f"Not auto-rebasing (won't touch the working tree mid-session)."
    )


def _derived_push_failure_cause() -> str:
    """DERIVE the cause from the repo, never assert one we did not check.

    2026-08-02 defect this replaces: this message asserted "origin/main likely moved
    (the parallel Marimo agent pushed) — reconcile manually (git pull --rebase ...)"
    on EVERY push failure, regardless of cause.  MEASURED counter-example: the push
    was refused by the LOCAL pre-push hook (preflight exceeding its budget under I/O
    contention) while origin/main had NOT moved -- HEAD was 2 ahead and 0 behind.
    Acting on the asserted cause would have run a rebase, for nothing, against a
    153-file dirty tree.  The stderr naming the real cause was already in hand and
    printed one line above; the hook simply guessed instead of reading it.

    Same genus as the vacuity/censoring classes this repo already gates: an
    instrument reporting a CONCLUSION it never measured.  So: fetch, count, and say
    only what the count supports -- including "I could not determine it".
    """
    fetched = False
    try:
        root = _repo_root()
        # Refresh the remote-tracking ref first: a stale origin/main answers the
        # wrong question (it reports what we knew, not what is true now).  Track
        # whether it SUCCEEDED -- claiming "VERIFIED" off a stale ref would be the
        # very defect this function exists to remove, one level deeper.
        fetched = _git(root, "fetch", "--quiet", "origin", "main", timeout=20.0).returncode == 0
        out = _git(root, "rev-list", "--count", "HEAD..origin/main", timeout=10.0)
        behind = int((out.stdout or "").strip()) if out.returncode == 0 else None
    except Exception:
        behind = None

    if behind is None:
        return ("Cause NOT determined (behind-count unavailable) — read the git detail "
                "above; do not assume the remote moved.")
    if not fetched:
        return (f"Cause UNCERTAIN: the fetch failed, so the {behind}-behind count is read "
                f"off a possibly-STALE origin/main — treat it as unverified. Read the git "
                f"detail above before rebasing.")
    if behind > 0:
        return (f"VERIFIED: origin/main moved ({behind} commit(s) we do not have) — "
                f"reconcile with `git pull --rebase origin main`, resolve, push.")
    return ("VERIFIED: origin/main did NOT move (we are 0 behind) — so this is a LOCAL "
            "refusal (pre-push hook) or a remote-side rejection; the git detail above "
            "names it. Do NOT rebase — it would be a no-op against your working tree.")


def run(dry_run: bool = False, use_fmtools: bool = True, fm_hold: bool = False,
        use_gitleaks: bool = True) -> dict:
    """Core logic. Returns a small verdict dict (for --dry-run / tests). Never raises."""
    root = _repo_root()

    # Kill-switch: operator drops this file to pause all auto-push (advertised in _block_reason).
    if (root / ".omx" / "state" / "auto_push.disabled").exists():
        return {"action": "skip", "reason": "kill-switch .omx/state/auto_push.disabled present"}

    # Only ever push `main`; a feature/detached branch is out of scope.
    branch = _git(root, "symbolic-ref", "--short", "HEAD").stdout.strip()
    if branch != "main":
        return {"action": "skip", "reason": f"branch={branch!r} not main"}

    # Do not push a mid-flight rebase/merge (inconsistent HEAD).
    for marker in ("rebase-merge", "rebase-apply", "MERGE_HEAD", "CHERRY_PICK_HEAD"):
        if (root / ".git" / marker).exists():
            return {"action": "skip", "reason": f"{marker} in progress"}

    ahead = _git(root, "rev-list", "--count", "origin/main..main").stdout.strip()
    if ahead in ("", "0"):
        return {"action": "noop", "reason": "not ahead of origin/main"}

    # Self-reference exclusion: the scanner's own source + its tests legitimately contain secret PATTERNS
    # (the regex definitions) and secret-SHAPED fixtures (example tokens / a made-up CGNAT IP). Scanning
    # them flags the scanner against itself — it held its OWN bootstrap push 2026-07-09 and needed a manual
    # push. Exclude them via git pathspec (the gitleaks allowlist pattern); they are hand-reviewed. Every
    # OTHER file is still fully scanned.
    _excludes = [f":(exclude){p}" for p in _SCAN_EXCLUDE_PATHSPECS]
    diff = _git(root, "diff", "origin/main..main", "--", ".", *_excludes, timeout=45).stdout
    hits = scan_diff(diff)
    if hits:
        names = ", ".join(f"{n}({s})" for n, s in hits)
        _log(root, f"HOLD ahead={ahead} — scan flagged: {names} — NOT pushed (manual review)")
        print(f"[auto-push] HELD: unpushed diff flagged by hygiene scan: {names}. "
              f"Reviewed+push manually if benign. (See .omx/state/auto_push.log)", file=sys.stderr)
        return {"action": "hold", "ahead": ahead, "hits": names}

    # Third deterministic layer: gitleaks (curated ruleset + entropy + our Tailscale rule). AUTHORITATIVE
    # like the regex — a finding HOLDS the push. Absent/error/timeout ⇒ None ⇒ skipped (regex already gated).
    if use_gitleaks:
        gl = gitleaks_scan(root)
        if gl:
            names = "; ".join(gl[:6])
            _log(root, f"HOLD ahead={ahead} — gitleaks flagged: {names} — NOT pushed (manual review)")
            print(f"[auto-push] HELD: gitleaks flagged {len(gl)} finding(s): {names}. Review+push manually "
                  f"if benign. (See .omx/state/auto_push.log)", file=sys.stderr)
            return {"action": "hold", "ahead": ahead, "hits": f"gitleaks[{names}]"}

    # fmtools advisory second layer: regex was clean, so ask the on-device FM about the ADDED lines for
    # novel/contextual secrets the regex can't pattern-match. Absent/timeout/error ⇒ None ⇒ push proceeds.
    # DEFAULT = LOG-ONLY (never holds — the established drift-detector firewall): the FM's verdict is
    # logged for audit but the deterministic regex remains the sole authority that holds a push. This is
    # because the FM reliably false-positives on sha256/hex hashes, which our commits stamp everywhere —
    # so making it *hold* would block most legit pushes. Operator opt-in AUTO_PUSH_FM_HOLD=1 makes it hold
    # on a flag (max leak-prevention, accepts hash false-positives). Either way it can only ADD a hold,
    # never permit a push the regex blocked.
    fm_note = None
    if use_fmtools:
        adv = fm_secret_advisory(_added_lines(diff))
        if adv and adv.get("contains_secret"):
            fm_note = f"{adv.get('category', '?')}: {adv.get('reason', '')}".strip()[:180]
            if fm_hold:
                _log(root, f"HOLD ahead={ahead} — fmtools flag [{fm_note}] — NOT pushed (AUTO_PUSH_FM_HOLD)")
                print(f"[auto-push] HELD (on-device FM, AUTO_PUSH_FM_HOLD): {fm_note}. Review+push manually "
                      f"if benign. (See .omx/state/auto_push.log)", file=sys.stderr)
                return {"action": "hold", "ahead": ahead, "hits": f"fmtools[{fm_note}]"}
            _log(root, f"FM-ADVISORY ahead={ahead} — flagged [{fm_note}] — pushing anyway (regex clean; "
                       f"log-only mode; set AUTO_PUSH_FM_HOLD=1 to hold instead)")

    if dry_run:
        return {"action": "would_push", "ahead": ahead, "hits": None,
                "gitleaks": "on" if use_gitleaks else "off",
                "fmtools": ("hold" if fm_hold else "advisory") if use_fmtools else "off", "fm_flag": fm_note}

    head = _git(root, "rev-parse", "--short", "main").stdout.strip()
    res = _git(root, "push", "origin", "main", timeout=90)
    if res.returncode == 0:
        _log(root, f"pushed origin/main -> {head} (was {ahead} ahead) [scan clean]")
        return {"action": "pushed", "head": head, "ahead": ahead}
    tail = (res.stderr or res.stdout or "").strip().replace("\n", " ")[-200:]
    _log(root, f"push FAILED rc={res.returncode}: {tail} — manual reconcile (no auto-rebase)")
    print(f"[auto-push] push failed (rc={res.returncode}): {tail}", file=sys.stderr)
    return {"action": "push_failed", "rc": res.returncode, "detail": tail}


def main(argv: list[str] | None = None) -> int:
    import os
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report the decision without pushing")
    ap.add_argument("--no-fmtools", action="store_true",
                    help="skip the on-device FM advisory layer (regex-only). Also via AUTO_PUSH_FM=0.")
    ap.add_argument("--fm-hold", action="store_true",
                    help="make an FM flag HOLD the push (default: log-only). Also via AUTO_PUSH_FM_HOLD=1.")
    ap.add_argument("--no-gitleaks", action="store_true",
                    help="skip the gitleaks deterministic layer. Also via AUTO_PUSH_GITLEAKS=0.")
    args = ap.parse_args(argv)
    use_fmtools = (not args.no_fmtools) and os.environ.get("AUTO_PUSH_FM", "1") != "0"
    fm_hold = args.fm_hold or os.environ.get("AUTO_PUSH_FM_HOLD", "0") == "1"
    use_gitleaks = (not args.no_gitleaks) and os.environ.get("AUTO_PUSH_GITLEAKS", "1") != "0"
    try:
        # Claude Code passes hook JSON on stdin; read it for the stop_hook_active
        # re-entry flag (loop guard #1) and to never block on the pipe.
        stop_hook_active = False
        if not sys.stdin.isatty():
            raw = ""
            try:
                raw = sys.stdin.read()
            except Exception:
                raw = ""
            try:
                stop_hook_active = bool(json.loads(raw).get("stop_hook_active")) if raw.strip() else False
            except Exception:
                stop_hook_active = False

        verdict = run(dry_run=args.dry_run, use_fmtools=use_fmtools, fm_hold=fm_hold,
                      use_gitleaks=use_gitleaks)
        if args.dry_run:
            print(f"[auto-push] {verdict}")
            return 0

        # The HAPPY path (pushed / noop / skip) is SILENT — no stdout — so the
        # Stop-hook stream carries only a decision when one is warranted. run()
        # has already logged + emitted a stderr note for hold/push_failed.
        if verdict.get("action") in ("hold", "push_failed") and not stop_hook_active:
            root = _repo_root()
            head = _git(root, "rev-parse", "HEAD").stdout.strip()
            if head and _marker_head(root) != head:  # loop guard #2: once per HEAD
                _write_marker(root, head)
                print(json.dumps({"decision": "block", "reason": _block_reason(verdict)}))
    except Exception as exc:  # fail-open: never break a turn
        print(f"[auto-push] non-fatal error (skipped): {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
