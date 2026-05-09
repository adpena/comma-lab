#!/usr/bin/env python3
# no-argparse-OK: zero-argument audit tool — only writes reports/silent_defaults.md
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
from collections.abc import Iterable
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)
SCAN_DIRS = [REPO / "experiments", REPO / "src" / "tac" / "experiments"]
REPORT = REPO / "reports" / "silent_defaults.md"


def _import_profile_keys() -> set[str]:
    """Collect every key that could come from any profile.

    A profile's keys are all the keys of every dict in PROFILES. We union
    across all profiles so a key set on ANY profile counts as "reachable
    via profile" — even if the user's chosen profile doesn't set it,
    another profile could.
    """
    try:
        from tac.profiles import PROFILES  # type: ignore
    except Exception as exc:  # pragma: no cover — surface import errors
        raise SystemExit(f"failed to import tac.profiles.PROFILES: {exc}") from exc
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


def _file_has_profile_mechanism(text: str) -> tuple[bool, bool]:
    """Detect whether the script (a) has a --profile argparse flag AND
    (b) uses a user-provided-flag mechanism that already handles the
    silent-default-override problem correctly.

    Returns:
        (has_profile_flag, has_user_provided_flags_mechanism)

    Scripts without a --profile flag CAN'T have silent profile overrides
    by definition — the audit's CRITICAL classification is a false
    positive. Scripts WITH a --profile flag AND a _user_provided_flags()
    style mechanism (or equivalent argparse-namespace inspection) handle
    the override correctly, so the audit's CRITICAL there is also FALSE.
    Only scripts with --profile but NO override-detection mechanism are
    real risks.
    """
    # A script must (a) declare --profile AND (b) actually import PROFILES
    # to be at risk of silent overrides. Some scripts have --profile but
    # explicitly note it's "informational" and don't load PROFILES (e.g.
    # experiments/train_imp_cycle.py:114-115). Those are NOT at risk.
    has_profile_flag = "--profile" in text and (
        "from tac.profiles import" in text or "tac.profiles import PROFILES" in text
    )
    # Heuristic: any of these helper names indicate the script knows about
    # the silent-override problem and handles it explicitly.
    has_mechanism = any(
        marker in text
        for marker in (
            "_user_provided_flags",
            "_apply_profile",
            "_resolve(args.",
            "_resolve(getattr(args",
            "user_set",
            "argv_flags",
            "_resolve_pose_dim_for_resume",
        )
    )
    return has_profile_flag, has_mechanism


def _scan_file(path: Path, *, source_index=None) -> list[dict]:
    """Return one record per `add_argument(...)` call in the file.

    Each record carries: file, line, arg_name, default_repr, action_kind,
    has_profile_flag, has_override_mechanism.
    """
    try:
        text = source_index.read_text(path) if source_index is not None else path.read_text()
        if "add_argument" not in text:
            return []
        tree = source_index.python_ast(path) if source_index is not None else ast.parse(text, filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    has_profile_flag, has_mechanism = _file_has_profile_mechanism(text)
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
            "has_profile_flag": has_profile_flag,
            "has_override_mechanism": has_mechanism,
        })
    return records


def collect_records(
    paths: Iterable[Path] | None = None,
    *,
    source_index=None,
) -> list[dict]:
    """Walk SCAN_DIRS for *.py files and accumulate add_argument records."""
    out: list[dict] = []
    if paths is None:
        iter_paths: list[Path] = []
        for root in SCAN_DIRS:
            if not root.exists():
                continue
            iter_paths.extend(sorted(root.rglob("*.py")))
    else:
        iter_paths = sorted(Path(path) for path in paths)
    for py in iter_paths:
        if "__pycache__" in py.parts:
            continue
        try:
            rel = py.resolve().relative_to(REPO.resolve())
        except ValueError:
            rel = py
        if rel.parts[:2] == ("experiments", "results"):
            continue
        out.extend(_scan_file(py, source_index=source_index))
    return out


def _is_risky_default(rec: dict) -> bool:
    """True iff the default is non-None and not a placeholder.

    A None default means "user didn't pass; resolver wins" — the safe
    pattern. False positives now filtered:

    1. ``store_true`` with default=False is STRUCTURALLY CORRECT (that's
       what store_true does — flag absent → False). A profile setting it
       True would force the user to type ``--use-x`` to disable, which is
       broken UX, so profiles in this codebase don't override store_true
       defaults. Same for store_false + default=True.
    2. Files without a ``--profile`` argparse flag CANNOT have silent
       profile overrides — the audit's CRITICAL is a false positive there.
    3. Files with a ``--profile`` flag AND a ``_user_provided_flags`` /
       ``_apply_profile`` / ``_resolve(args.X, ...)`` style mechanism
       handle the override correctly — false positive.
    """
    repr_str = rec["default_repr"]
    if repr_str in ("None", "<not-set>"):
        return False
    # store_true / store_false with their structurally-correct defaults
    # are NOT risky — that's the only valid pairing for these actions.
    action = rec.get("action")
    if action == "store_true" and repr_str == "False":
        return False
    if action == "store_false" and repr_str == "True":
        return False
    # File-level filter: scripts without --profile cannot have silent profile
    # overrides. Older tests and ad hoc callers pass minimal records without
    # this key; treat missing as "unknown" rather than "safe".
    if rec.get("has_profile_flag") is False:
        return False
    # File-level filter: scripts with an override-detection mechanism
    # (_user_provided_flags / _apply_profile / _resolve(args.X)) handle
    # the silent-default problem correctly.
    # CAVEAT: this is broad. A file with SOME _resolve calls and SOME
    # un-resolved non-None defaults will pass the file-level check yet
    # contain real bugs in the un-resolved flags. The 3 real bugs caught
    # this loop in train_renderer.py (--grad-clip, --fp4-codebook,
    # --wall-clock-timeout) were all of this shape — fixed in commit
    # 256c5e42. Per-flag verification is a future enhancement.
    return rec.get("has_override_mechanism") is not True


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
    canonical_entrypoints = [
        "src/tac/experiments/train_renderer.py",
        "experiments/pipeline.py",
    ]
    present_entrypoints = [path for path in canonical_entrypoints if (REPO / path).is_file()]
    if present_entrypoints:
        lines.append(
            "- **CANONICAL TRAINING ENTRYPOINTS SCANNED**: "
            + ", ".join(f"`{path}`" for path in present_entrypoints)
        )
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


def summarize_records(records: list[dict], profile_keys: set[str]) -> dict[str, int]:
    """Return the audit counts without writing the markdown report.

    Preflight imports this pure helper so the normal green gate does not fork
    a subprocess or dirty ``reports/silent_defaults.md`` on every run. The CLI
    still owns report rendering for operator-facing audits.
    """
    critical = 0
    suspicious = 0
    safe_count = 0
    for rec in records:
        risky = _is_risky_default(rec)
        if not risky:
            safe_count += 1
            continue
        if rec["key"] in profile_keys:
            critical += 1
        else:
            suspicious += 1
    return {
        "critical": critical,
        "suspicious": suspicious,
        "safe": safe_count,
        "total": len(records),
    }


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
