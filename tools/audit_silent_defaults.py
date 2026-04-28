#!/usr/bin/env python3
"""Audit argparse defaults that may silently override profile values.

THE BUG PATTERN (fix #1 of the DX hardening pass): an argparse `default=`
that is NOT None will ALWAYS take precedence over a profile-supplied value
unless the call site has special-cased it. The KL distill bug (Quantizr
council CRITICAL 2026-04-26) was exactly this: `--kl-distill-weight` had
default=0.0 in argparse, so the profile's value was DEAD; KL distill never
fired in any production training run.

This tool is an AUDIT, not a fix. It scans every `add_argument(...)` in
`experiments/` and `src/tac/experiments/` and produces a report at
`reports/silent_defaults.md` listing each default that has a matching key
in `tac.profiles.PROFILES`. Operators decide per-flag whether the default
is intentional (CLI-only flag, never set by profile) or dangerous (silent
profile override). The fix in each case is one of:

  1. Change the default to None and resolve from the profile in the body.
  2. Document explicitly that this CLI flag has no profile counterpart.
  3. Remove the flag if no caller uses it (dead code).

Run:
    .venv/bin/python tools/audit_silent_defaults.py

The report is overwritten in-place. No GPU is consumed.
"""
from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCAN_DIRS = [REPO / "experiments", REPO / "src" / "tac" / "experiments"]
REPORT = REPO / "reports" / "silent_defaults.md"


def _import_profile_keys() -> set[str]:
    """Collect every key that could come from any profile.

    A profile's keys are all the keys of every dict in PROFILES. We union
    across all profiles so a key set on ANY profile counts as "reachable
    via profile" — even if the user's chosen profile doesn't set it,
    another profile could.
    """
    sys.path.insert(0, str(REPO / "src"))
    try:
        from tac.profiles import PROFILES  # type: ignore
    except Exception as exc:  # pragma: no cover — surface import errors
        raise SystemExit(f"failed to import tac.profiles.PROFILES: {exc}")
    keys: set[str] = set()
    for prof in PROFILES.values():
        if isinstance(prof, dict):
            keys.update(prof.keys())
    return keys


def _argname_to_key(arg_name: str) -> str:
    """Map `--kl-distill-weight` → `kl_distill_weight` (argparse convention).

    Only positional/long flags are considered; we strip leading dashes and
    convert internal dashes to underscores. Mixed-case flags are kept lower.
    """
    stripped = arg_name.lstrip("-")
    return stripped.replace("-", "_").lower()


def _parse_default(node: ast.AST) -> str:
    """Best-effort literal repr of an argparse default value.

    Constants and simple containers are stringified; anything more complex
    (function calls, attribute lookups) is reported as `<expr>` so the
    auditor sees "this default is computed at parse time".
    """
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
        try:
            return repr(ast.literal_eval(node))
        except (ValueError, SyntaxError):
            return "<container-expr>"
    if isinstance(node, ast.Call):
        # Expressions like default=str(_repo / "...") resolve at runtime.
        return f"<call:{ast.unparse(node)[:60]}>"
    if isinstance(node, ast.Attribute):
        return f"<attr:{ast.unparse(node)}>"
    if isinstance(node, ast.Name):
        return f"<name:{node.id}>"
    try:
        return f"<expr:{ast.unparse(node)[:60]}>"
    except Exception:
        return "<unparseable>"


def _scan_file(path: Path) -> list[dict]:
    """Return one record per `add_argument(...)` call in the file.

    Each record carries: file, line, arg_name, default_repr, is_action_flag.
    Action flags (store_true / store_false / count / append) are tagged so
    the report can sort them separately — they're rarely a profile-override
    risk because their default is implied (False / None).
    """
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []
    records: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # match `*.add_argument(...)` regardless of receiver name (p / parser / sub).
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        arg_name = first.value
        if not arg_name.startswith("-"):
            continue  # positional args don't follow profile-resolution pattern
        default_repr = "<not-set>"
        action_kind = None
        for kw in node.keywords:
            if kw.arg == "default":
                default_repr = _parse_default(kw.value)
            elif kw.arg == "action" and isinstance(kw.value, ast.Constant):
                action_kind = kw.value.value
        records.append({
            "file": str(path.relative_to(REPO)),
            "line": node.lineno,
            "arg_name": arg_name,
            "key": _argname_to_key(arg_name),
            "default_repr": default_repr,
            "action": action_kind,
        })
    return records


def collect_records() -> list[dict]:
    """Walk SCAN_DIRS for *.py files and accumulate add_argument records."""
    out: list[dict] = []
    for root in SCAN_DIRS:
        if not root.exists():
            continue
        for py in sorted(root.rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            out.extend(_scan_file(py))
    return out


def _is_risky_default(rec: dict) -> bool:
    """True iff the default is non-None and not a placeholder.

    A None default means "user didn't pass; resolver wins" — the safe
    pattern. Action flags whose default is False/None are also safe (the
    profile resolver can detect "user didn't pass --use-x" via None).
    """
    repr_str = rec["default_repr"]
    if repr_str in ("None", "<not-set>"):
        return False
    if rec.get("action") in ("store_true", "store_false") and repr_str in ("False", "True"):
        # store_true with default=False is the dangerous case ONLY if a profile
        # would set it True — caller is expected to mark it. Action flags with
        # default=None are safe; default=False is the "silent override" risk.
        return repr_str == "False" and rec.get("action") == "store_true"
    return True


def write_report(records: list[dict], profile_keys: set[str]) -> None:
    """Render `reports/silent_defaults.md` grouped by file.

    Sections:
      1. CRITICAL (default != None AND key is in PROFILES) — the KL bug class
      2. SUSPICIOUS (default != None, key NOT in PROFILES) — review for dead code
      3. SAFE (default == None) — for completeness, summary count only
    """
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    critical: list[dict] = []
    suspicious: list[dict] = []
    safe_count = 0
    for rec in records:
        risky = _is_risky_default(rec)
        if not risky:
            safe_count += 1
            continue
        if rec["key"] in profile_keys:
            critical.append(rec)
        else:
            suspicious.append(rec)

    by_file_crit: dict[str, list[dict]] = defaultdict(list)
    for rec in critical:
        by_file_crit[rec["file"]].append(rec)

    lines: list[str] = []
    lines.append("# Silent Argparse Defaults Audit")
    lines.append("")
    lines.append("Generated by `tools/audit_silent_defaults.py`. This is the KL distill")
    lines.append("bug pattern — argparse defaults that silently override profile values.")
    lines.append("")
    lines.append(f"- **CRITICAL** (matches a key in `tac.profiles.PROFILES`): **{len(critical)}**")
    lines.append(f"- **SUSPICIOUS** (non-None default, no profile match): **{len(suspicious)}**")
    lines.append(f"- **SAFE** (default=None or action implies None): **{safe_count}**")
    lines.append(f"- **TOTAL ARGUMENTS SCANNED**: **{len(records)}**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## CRITICAL — silent profile overrides")
    lines.append("")
    lines.append("Each entry below has a non-None default AND a matching profile key.")
    lines.append("Fix: change the default to `None`, resolve in the body via the")
    lines.append("profile's value when the user did not pass it explicitly.")
    lines.append("")
    if not by_file_crit:
        lines.append("_(none — clean!)_")
        lines.append("")
    for fname in sorted(by_file_crit):
        lines.append(f"### `{fname}`")
        lines.append("")
        lines.append("| line | arg | default | profile key |")
        lines.append("|------|-----|---------|-------------|")
        for rec in sorted(by_file_crit[fname], key=lambda r: r["line"]):
            default_md = rec["default_repr"].replace("|", "\\|")
            lines.append(
                f"| {rec['line']} | `{rec['arg_name']}` | "
                f"`{default_md}` | `{rec['key']}` |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## SUSPICIOUS — non-None default with no profile match")
    lines.append("")
    lines.append("These are CLI-only flags. Verify each is intentional (e.g. `--device`,")
    lines.append("`--output-dir`); if any is a feature toggle that SHOULD be in a profile,")
    lines.append("add the profile key + change the default to None.")
    lines.append("")
    by_file_sus: dict[str, list[dict]] = defaultdict(list)
    for rec in suspicious:
        by_file_sus[rec["file"]].append(rec)
    for fname in sorted(by_file_sus):
        lines.append(f"### `{fname}` ({len(by_file_sus[fname])} flag(s))")
        lines.append("")
        lines.append("| line | arg | default |")
        lines.append("|------|-----|---------|")
        for rec in sorted(by_file_sus[fname], key=lambda r: r["line"]):
            default_md = rec["default_repr"].replace("|", "\\|")
            lines.append(
                f"| {rec['line']} | `{rec['arg_name']}` | `{default_md}` |"
            )
        lines.append("")

    REPORT.write_text("\n".join(lines))
    print(f"[audit_silent_defaults] wrote {REPORT} "
          f"({len(critical)} critical, {len(suspicious)} suspicious, {safe_count} safe)")


def main() -> int:
    profile_keys = _import_profile_keys()
    records = collect_records()
    if not records:
        print("[audit_silent_defaults] no add_argument calls found — check SCAN_DIRS",
              file=sys.stderr)
        return 1
    write_report(records, profile_keys)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
