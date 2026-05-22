#!/usr/bin/env python3
"""Cross-substrate driver mode-hardcode audit per Catalog #326 + DRIVER-FIX wave 2026-05-18.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
non-negotiable + Z6-v2 Wave 2 DEFER 2026-05-18 anchor: scan every
``scripts/remote_lane_substrate_*.sh`` driver for the bug class where the
trainer is invoked with ``--smoke`` flag (or equivalent ``smoke=1`` /
``SMOKE=1`` shell pattern) WITHOUT consulting an env var to determine
smoke-vs-full mode.

Empirical anchor: Z6-v2 Wave 2 full canary dispatch
``fc-01KRW7ZCYK5XF6MSHD24R71A46`` ran ``_smoke_main`` despite the recipe
requesting ``Z6_EPOCHS=100`` full-mode because the Z6-v2 Wave 2 recipe
``env_overrides`` block did NOT set ``SMOKE_ONLY=0``. The driver's
``SMOKE_ONLY="${SMOKE_ONLY:-1}"`` default produced smoke-mode regardless
of intent. The bug class is: "driver mode-routing default biases toward
smoke without explicit recipe-side opt-in for full".

Per-driver verdict taxonomy:

- ``HARDCODES_SMOKE_NO_RECIPE_OPT_OUT``: driver hardcodes ``--smoke`` AND
  the matching recipe does NOT declare ``smoke_only: true`` /
  ``research_only: true`` / ``dispatch_enabled: false``. This is the
  active bug class.
- ``HARDCODES_SMOKE_RECIPE_OPTED_OUT``: driver hardcodes ``--smoke`` but
  the recipe explicitly opts out of full-mode via the standard
  research-only / smoke-only / dispatch-disabled mechanism. Acceptable.
- ``CONSUMES_ENV_DEFAULTS_SMOKE``: driver supports env-var smoke/full
  toggle BUT default is smoke. If the recipe does NOT set the env var,
  the driver runs smoke. This is the Z6 bug class.
- ``CONSUMES_ENV_DEFAULTS_FULL``: driver supports env-var smoke/full
  toggle AND default is full (or recipe sets it explicitly). Healthy.
- ``NO_SMOKE_FLAG``: driver passes no ``--smoke`` flag at all (trainer
  internally branches on epochs). Out of scope for this audit.

Usage::

    .venv/bin/python tools/audit_substrate_driver_mode_hardcode.py
    .venv/bin/python tools/audit_substrate_driver_mode_hardcode.py --apply  # writes report JSON

The audit JSON path is
``.omx/state/substrate_driver_mode_hardcode_audit_<utc>.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVERS_GLOB = "scripts/remote_lane_substrate_*.sh"


def _recipes_dir(repo_root: Path) -> Path:
    return repo_root / ".omx" / "operator_authorize_recipes"

# Patterns: any line invoking trainer w/ --smoke flag.
# After stripping trailing comments, ANY occurrence of --smoke on a
# code line is hardcoded — the conditional accumulator pattern is
# detected separately by _SMOKE_CONDITIONAL_PATTERNS.
_SMOKE_FLAG_PATTERNS = (
    # Match --smoke as a whole-word shell token. Use lookarounds because
    # `\b` does not match around `-` (non-word char). The flag must be
    # preceded by whitespace or start-of-line and followed by whitespace,
    # end-of-line, `\`, or `)`.
    re.compile(r"(?:^|\s)--smoke(?=$|\s|\\|\))"),
)
# Patterns that indicate --smoke is INSIDE a conditional accumulator
# (not unconditional hardcoding):
_SMOKE_CONDITIONAL_PATTERNS = (
    re.compile(r"SMOKE_FLAG_ARGS\s*\+=\s*\(\s*--smoke\s*\)"),  # Z6-style accumulator
    re.compile(r"SMOKE_ARGS\s*\+=\s*\(\s*--smoke\s*\)"),
    re.compile(r"if\s+\[.*?--smoke"),  # if-test with --smoke literal
)
# Env-var smoke/full mode tokens
_MODE_ENV_TOKENS = (
    "SMOKE_ONLY",
    "TRAINER_MODE",
    "DISPATCH_MODE",
    "RUN_MODE",
    "SMOKE_MODE",
    "FULL_MODE",
    "MODE_FULL",
)
_MODE_ENV_TOKEN_RE = re.compile(
    r"\b([A-Z0-9_]*(?:"
    + "|".join(re.escape(token) for token in _MODE_ENV_TOKENS)
    + r")[A-Z0-9_]*)\b"
)
_DEVICE_ARG_ENV_RE = re.compile(
    r"--device\s+(?:[\"']?\$\{?([A-Z0-9_]*DEVICE[A-Z0-9_]*)\}?[\"']?)"
)
# Recipe-side opt-out markers (driver hardcoded --smoke is acceptable IF
# the recipe explicitly opts out of full-mode dispatch)
_RECIPE_OPT_OUT_TOKENS = (
    "smoke_only: true",
    "smoke_only:true",
    "research_only: true",
    "research_only:true",
    "dispatch_enabled: false",
    "dispatch_enabled:false",
)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _substrate_id_from_driver(driver_path: Path) -> str:
    name = driver_path.stem
    prefix = "remote_lane_substrate_"
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


def _find_matching_recipes(
    substrate_id: str, driver_path: Path, repo_root: Path
) -> list[Path]:
    """Find recipes that reference this driver.

    Two lookup strategies (union):
    1. Filename prefix match (e.g. ``substrate_<id>_*.yaml`` → recipe).
    2. Reverse content lookup: recipes whose ``remote_driver:`` or
       ``lane_script:`` field points at this driver's relative path. This
       catches recipes like ``substrate_z6_v2_candidate_1_*.yaml`` that
       share the Z6 driver but don't carry the substrate_id in the
       filename.
    """
    recipes_dir = _recipes_dir(repo_root)
    if not recipes_dir.is_dir():
        return []
    pattern_a = f"substrate_{substrate_id}_*.yaml"
    pattern_b = f"substrate_{substrate_id}.yaml"
    hits: set[Path] = set()
    for pattern in (pattern_a, pattern_b):
        hits.update(recipes_dir.glob(pattern))
    # Reverse content lookup
    try:
        driver_rel = str(driver_path.relative_to(repo_root))
    except ValueError:
        driver_rel = str(driver_path)
    driver_name = driver_path.name
    for recipe_path in recipes_dir.glob("substrate_*.yaml"):
        text = _read(recipe_path)
        # Match recipes whose remote_driver: or lane_script: line points at
        # this driver (either by relative path or by basename).
        for line in text.splitlines():
            stripped = line.strip()
            if not (stripped.startswith("remote_driver:") or stripped.startswith("lane_script:")):
                continue
            if driver_rel in stripped or driver_name in stripped:
                hits.add(recipe_path)
                break
    return sorted(hits)


def _recipe_opted_out_of_full_mode(recipe_path: Path) -> bool:
    """True iff recipe has a TOP-LEVEL YAML key (not a comment) setting any
    of the opt-out fields to true / false-as-appropriate.

    Checks:
    - ``research_only: true`` (top-level)
    - ``smoke_only: true`` (top-level)
    - ``dispatch_enabled: false`` (top-level)

    Comment lines (starting with ``#`` after leading whitespace) are excluded
    so explanatory prose mentioning ``research_only: true`` does NOT
    falsely satisfy the opt-out test.
    """
    text = _read(recipe_path)
    if not text:
        return False
    for line in text.splitlines():
        stripped = line.lstrip()
        # Skip comment lines
        if stripped.startswith("#") or not stripped:
            continue
        # Strip trailing comment
        code = stripped.split("#", 1)[0].rstrip()
        if not code:
            continue
        # Match exact YAML key: value (lenient on whitespace)
        for key, expected in (
            ("research_only", "true"),
            ("smoke_only", "true"),
            ("dispatch_enabled", "false"),
        ):
            m = re.match(rf"^{key}\s*:\s*([A-Za-z0-9_\"']+)\s*$", code)
            if m and m.group(1).strip().strip('"').strip("'").lower() == expected:
                return True
    return False


def _driver_has_hardcoded_smoke(driver_text: str) -> bool:
    """True iff driver passes --smoke flag unconditionally on a command line
    (NOT inside a conditional accumulator like SMOKE_FLAG_ARGS+=(--smoke)).

    Strip trailing comments before matching so a same-line ``# DRIVER_MODE_*``
    waiver does NOT mask the bug class — the waiver is detected separately at
    the gate-acceptance layer.
    """
    lines = driver_text.splitlines()
    for line in lines:
        # Strip trailing comments to expose --smoke even when followed by a
        # waiver comment.
        stripped = line.split("#", 1)[0]
        if "--smoke" not in stripped:
            continue
        # Reject lines that are conditional accumulator patterns (Z6-style)
        # — those use env-var gating and are not literally hardcoded.
        cond_match = any(p.search(stripped) for p in _SMOKE_CONDITIONAL_PATTERNS)
        if cond_match:
            continue
        # Match unconditional --smoke patterns
        for pattern in _SMOKE_FLAG_PATTERNS:
            if pattern.search(stripped):
                return True
    return False


def _driver_consumes_env_mode(driver_text: str) -> tuple[bool, str | None]:
    """Detect if driver uses any env-var to select mode.

    Returns ``(consumes, env_var_name_or_none)``. Detects either:
    - ``VAR="${VAR:-...}"`` parameter expansion of mode token, OR
    - ``if [ "$VAR" = "1" ]; then SMOKE_FLAG_ARGS+=(--smoke); fi`` conditional pattern.
    """
    env_vars = _driver_mode_env_vars(driver_text)
    return bool(env_vars), env_vars[0] if env_vars else None


def _driver_mode_env_vars(driver_text: str) -> tuple[str, ...]:
    """Return concrete shell env var names used for smoke/full mode routing."""

    seen: set[str] = set()
    ordered: list[str] = []
    for match in _MODE_ENV_TOKEN_RE.finditer(driver_text):
        var = match.group(1)
        if var in seen:
            continue
        seen.add(var)
        ordered.append(var)
    return tuple(ordered)


def _driver_env_default_value(driver_text: str, env_var: str) -> str | None:
    """Extract the default value from ``VAR="${VAR:-default}"`` pattern."""
    # Match patterns like SMOKE_ONLY="${SMOKE_ONLY:-1}"
    pattern = re.compile(
        rf'^\s*{re.escape(env_var)}\s*=\s*["\']?\$\{{{re.escape(env_var)}:-([^}}]*)\}}',
        re.MULTILINE,
    )
    match = pattern.search(driver_text)
    if match:
        value = match.group(1).strip().strip('"').strip("'")
        if value:
            return value
    branch_default = _driver_env_empty_branch_default_value(driver_text, env_var)
    if branch_default is not None:
        return branch_default
    return None


def _driver_device_env_vars(driver_text: str) -> tuple[str, ...]:
    """Return shell env vars passed to trainer ``--device`` flags."""

    seen: set[str] = set()
    ordered: list[str] = []
    for match in _DEVICE_ARG_ENV_RE.finditer(driver_text):
        var = match.group(1)
        if var in seen:
            continue
        seen.add(var)
        ordered.append(var)
    return tuple(ordered)


def _mode_prefix(env_var: str) -> str:
    for suffix in (
        "TRAINER_MODE",
        "DISPATCH_MODE",
        "RUN_MODE",
        "SMOKE_MODE",
        "FULL_MODE",
        "MODE_FULL",
        "SMOKE_ONLY",
    ):
        if env_var.endswith(suffix):
            return env_var[: -len(suffix)]
    return ""


def _device_value_is_cpu(value: str | None) -> bool:
    return (value or "").strip().strip('"').strip("'").lower() == "cpu"


def _full_mode_cpu_device_unsafe_recipes(
    *,
    recipes: list[Path],
    driver_text: str,
    env_vars: tuple[str, ...],
    repo_root: Path,
) -> list[dict[str, Any]]:
    """Find recipes whose full trainer mode resolves to ``--device cpu``.

    This is the NSCS06 v8 rc=1 sister class after the rc=22 mode-refusal fix:
    the driver honored ``TRAINER_MODE=full`` but still sent ``--device cpu`` to
    the trainer. Full-mode CPU is a late trainer refusal and should be caught
    before provider spend.
    """

    device_vars = _driver_device_env_vars(driver_text)
    if not device_vars:
        return []

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(repo_root))
        except ValueError:
            return str(p)

    unsafe: list[dict[str, Any]] = []
    for rp in recipes:
        if _recipe_opted_out_of_full_mode(rp):
            continue
        rp_env_values = _recipe_mode_env_values(rp, env_vars)
        full_mode_vars = {
            var: value
            for var, value in rp_env_values.items()
            if _mode_env_value_forces_full(var, value)
        }
        if not full_mode_vars:
            continue
        full_prefixes = {_mode_prefix(var) for var in full_mode_vars}
        for device_var in device_vars:
            device_prefix = device_var[: -len("DEVICE")] if device_var.endswith("DEVICE") else ""
            if full_prefixes and "" not in full_prefixes and device_prefix not in full_prefixes:
                continue
            effective_device = (
                _recipe_sets_env_var(rp, device_var)
                or _driver_env_default_value(driver_text, device_var)
            )
            if not _device_value_is_cpu(effective_device):
                continue
            unsafe.append(
                {
                    "recipe": _rel(rp),
                    "mode_env_values_in_recipe": full_mode_vars,
                    "device_env_var": device_var,
                    "effective_device": effective_device,
                }
            )
    return unsafe


def _driver_env_empty_branch_default_value(driver_text: str, env_var: str) -> str | None:
    """Detect shell branches like ``elif [ -z "$VAR" ]; then VAR="1"``."""

    lines = driver_text.splitlines()
    empty_test = re.compile(
        rf"\[\s+-z\s+[\"']?\${re.escape(env_var)}[\"']?\s+\]"
    )
    assignment = re.compile(
        rf"^\s*{re.escape(env_var)}\s*=\s*[\"']?([^\"'\n#]+)[\"']?\s*(?:#.*)?$"
    )
    for idx, line in enumerate(lines):
        if not empty_test.search(line):
            continue
        for follow in lines[idx + 1 : min(len(lines), idx + 8)]:
            match = assignment.match(follow)
            if match:
                return match.group(1).strip()
    return None


def _recipe_sets_env_var(recipe_path: Path, env_var: str) -> str | None:
    """Read the recipe's ``env_overrides`` block and extract ``env_var``.

    This intentionally does not scan the whole recipe: notes, comments, or
    unrelated top-level keys mentioning ``SMOKE_ONLY: "0"`` are not dispatch
    environment overrides and must not make a smoke-default driver look safe.
    """
    text = _read(recipe_path)
    if not text:
        return None
    in_env_overrides = False
    base_indent: int | None = None
    key_pattern = re.compile(
        rf"^\s*{re.escape(env_var)}\s*:\s*['\"]?([^'\"\n#]+?)['\"]?\s*(?:#.*)?$"
    )
    top_level_key = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*\s*:")
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if re.match(r"^env_overrides\s*:\s*(?:#.*)?$", line):
            in_env_overrides = True
            base_indent = len(line) - len(line.lstrip())
            continue
        if not in_env_overrides:
            continue
        indent = len(line) - len(line.lstrip())
        if base_indent is not None and indent <= base_indent and top_level_key.match(line):
            break
        match = key_pattern.match(line)
        if match:
            return match.group(1).strip()
    return None


def _mode_env_value_forces_full(env_var: str, value: str | None) -> bool:
    if value is None:
        return False
    low = value.strip().strip('"').strip("'").lower()
    if not low:
        return False
    if "smoke" in env_var.lower():
        return low in ("0", "false", "no", "off", "full")
    return low in ("full", "0", "false", "no", "off")


def _mode_env_value_forces_smoke(env_var: str, value: str | None) -> bool:
    if value is None:
        return False
    low = value.strip().strip('"').strip("'").lower()
    if not low:
        return False
    if "smoke" in env_var.lower():
        return low in ("1", "true", "yes", "on", "smoke")
    return low in ("smoke", "1", "true", "yes", "on")


def _recipe_mode_env_values(recipe_path: Path, env_vars: tuple[str, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    for env_var in env_vars:
        value = _recipe_sets_env_var(recipe_path, env_var)
        if value is not None:
            values[env_var] = value
    return values


def _driver_refuses_non_smoke_mode(driver_text: str, env_var: str) -> bool:
    """Detect the "refuses-non-smoke at startup" anti-pattern.

    OVERNIGHT-RR 2026-05-21 root-cause anchor at NSCS06 v8 chroma-LUT driver
    Stage 0: ``if [ "$NSCS06_V8_TRAINER_MODE" != "smoke" ]; then ... exit 22; fi``

    The driver READS the env var (so it satisfies ``_driver_consumes_env_mode``)
    but uses it to REFUSE non-smoke values rather than to BRANCH on the value
    and route to smoke vs full code paths. When the matching recipe declares
    ``dispatch_enabled: true`` + ``NSCS06_V8_TRAINER_MODE: full`` the dispatch
    crashes at rc=22 ~2s with stdout ``FATAL: ... only 'smoke' is supported``.

    The match pattern looks for an ``if [ "$VAR" != "smoke" ]`` OR ``if [
    "$VAR" != smoke ]`` test followed by an ``exit`` statement within ~10
    lines (to allow log statements between the test and exit). Quoted /
    unquoted / single-quoted "smoke" token all accepted.
    """
    lines = driver_text.splitlines()
    # Pattern: if [ "$VAR" != "smoke" ] (or unquoted/single-quoted "smoke")
    test_pattern_braced = re.compile(
        rf'if\s+\[\s+["\']?\${{{re.escape(env_var)}}}["\']?\s*!=\s*["\']?smoke["\']?\s*\]'
    )
    test_pattern_simple = re.compile(
        rf'if\s+\[\s+["\']?\${re.escape(env_var)}["\']?\s*!=\s*["\']?smoke["\']?\s*\]'
    )
    for i, line in enumerate(lines):
        stripped = line.split("#", 1)[0]
        if not (
            test_pattern_braced.search(stripped)
            or test_pattern_simple.search(stripped)
        ):
            continue
        # OVERNIGHT-RR refinement: the multi-value validation pattern
        # ``if [ "$VAR" != "smoke" ] && [ "$VAR" != "full" ]; then ... exit; fi``
        # is the CANONICAL valid-mode-validator (drivers that accept full mode);
        # exempt it from the refuse-pattern bug class.
        if "full" in stripped.lower() and "!=" in stripped:
            # The test mentions both `smoke` AND `full` with != — this is the
            # multi-value validator pattern, not the single-smoke-only refuse
            # pattern. Skip.
            continue
        # Found the refuse-test; look ahead ~10 lines for an exit statement
        for j in range(i + 1, min(i + 11, len(lines))):
            look = lines[j].split("#", 1)[0]
            if re.search(r"^\s*exit\s+\d+", look):
                return True
            if re.search(r"^\s*fi\s*$", look):
                # if-block closed without exit; not the refuse pattern
                break
    return False


def _classify_driver(driver_path: Path, repo_root: Path) -> dict[str, Any]:
    text = _read(driver_path)
    substrate_id = _substrate_id_from_driver(driver_path)
    recipes = _find_matching_recipes(substrate_id, driver_path, repo_root)
    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(repo_root))
        except ValueError:
            return str(p)
    recipe_paths_str = [_rel(p) for p in recipes]
    any_recipe_opted_out = any(_recipe_opted_out_of_full_mode(p) for p in recipes)

    has_hardcoded = _driver_has_hardcoded_smoke(text)
    # The SMOKE_FLAG_ARGS+=(--smoke) conditional accumulator is also a
    # "could-route-smoke" pattern when the env-var default biases to smoke.
    has_conditional_smoke = any(
        p.search(text) for p in _SMOKE_CONDITIONAL_PATTERNS
    )
    consumes_env, env_var = _driver_consumes_env_mode(text)
    env_vars = _driver_mode_env_vars(text)

    # OVERNIGHT-RR 2026-05-21 anchor: driver-refuses-non-smoke-mode bug class.
    # The driver READS the env var but uses it to REFUSE rather than BRANCH.
    # Detected by ``if [ "$VAR" != "smoke" ]; then ... exit N; fi`` pattern.
    # Bug class fires when ANY recipe forces full mode AND does NOT opt out.
    refuse_per_var: dict[str, bool] = {
        var: _driver_refuses_non_smoke_mode(text, var) for var in env_vars
    }
    full_mode_cpu_unsafe = _full_mode_cpu_device_unsafe_recipes(
        recipes=recipes,
        driver_text=text,
        env_vars=env_vars,
        repo_root=repo_root,
    )
    if full_mode_cpu_unsafe:
        return {
            "substrate_id": substrate_id,
            "driver_path": _rel(driver_path),
            "recipes": recipe_paths_str,
            "verdict": "FULL_MODE_DEVICE_CPU_BUG_CLASS",
            "explanation": (
                "Driver/recipe resolves trainer full mode to --device cpu. "
                "NSCS06 v8 rc=1 anchor: after the rc=22 mode-refusal fix, "
                "TRAINER_MODE=full reached the trainer but failed late at "
                "device_or_die because full-mode CPU is non-promotional and "
                "not a valid paid-dispatch substrate. Device must resolve to "
                "cuda for full mode, or the recipe must opt out as smoke-only/"
                "research-only/dispatch-disabled."
            ),
            "any_recipe_opted_out_of_full_mode": any_recipe_opted_out,
            "consumes_env_var": env_var,
            "consumes_env_vars": list(env_vars),
            "device_env_vars": list(_driver_device_env_vars(text)),
            "unsafe_recipes": full_mode_cpu_unsafe,
        }
    any_var_refused = any(refuse_per_var.values())
    if any_var_refused:
        unsafe_recipes_refuse = []
        for rp in recipes:
            if _recipe_opted_out_of_full_mode(rp):
                continue
            rp_env_values = _recipe_mode_env_values(rp, env_vars)
            rp_env_full = any(
                _mode_env_value_forces_full(var, value)
                for var, value in rp_env_values.items()
            )
            if rp_env_full:
                unsafe_recipes_refuse.append(_rel(rp))
        if unsafe_recipes_refuse:
            return {
                "substrate_id": substrate_id,
                "driver_path": _rel(driver_path),
                "recipes": recipe_paths_str,
                "verdict": "REFUSES_NON_SMOKE_RECIPE_FORCES_FULL_BUG_CLASS",
                "explanation": (
                    f"Driver refuses non-smoke values for env vars "
                    f"{[v for v, r in refuse_per_var.items() if r]} via FATAL+exit "
                    "pattern at startup (Stage 0 mode-refusal); matching recipes "
                    f"force full mode WITHOUT opt-out: {unsafe_recipes_refuse}. "
                    "OVERNIGHT-RR 2026-05-21 anchor: NSCS06 v8 chroma-LUT QQ "
                    "dispatch fc-01KS5QRXWNVYC54E2Y9Z8KZ4W2 rc=22 elapsed=1.7s "
                    "with stdout `FATAL: ... only 'smoke' is supported in L0 "
                    "SCAFFOLD`. The trainer's _full_main was Phase 2 BUILD-flipped "
                    "(OVERNIGHT-V) AND the recipe was atomically flipped, but the "
                    "driver Stage 0 refuse-guard was not. ACTIVE BUG CLASS — "
                    "driver must branch on env var to route smoke vs full code "
                    "paths, not refuse non-smoke. CLAUDE.md `Forbidden substrate "
                    "driver hardcoding smoke=1 / --smoke regardless of dispatch "
                    "env vars`."
                ),
                "any_recipe_opted_out_of_full_mode": any_recipe_opted_out,
                "consumes_env_var": env_var,
                "consumes_env_vars": list(env_vars),
                "env_var_default_in_driver": (
                    _driver_env_default_value(text, env_var)
                    if (consumes_env and env_var)
                    else None
                ),
                "refuse_per_var": refuse_per_var,
                "unsafe_recipes": unsafe_recipes_refuse,
            }

    if not has_hardcoded and not consumes_env and not has_conditional_smoke:
        verdict = "NO_SMOKE_FLAG"
        explanation = (
            "Driver does not pass --smoke flag; trainer determines mode internally."
        )
    elif has_hardcoded and not consumes_env:
        all_recipes_opted_out = bool(recipes) and all(
            _recipe_opted_out_of_full_mode(p) for p in recipes
        )
        if all_recipes_opted_out:
            verdict = "HARDCODES_SMOKE_RECIPE_OPTED_OUT"
            explanation = (
                "Driver unconditionally passes --smoke flag; matching recipe "
                "set all declares research_only/smoke_only/dispatch_enabled=false. "
                "Acceptable per CLAUDE.md HNeRV parity L2 (research-only by construction)."
            )
        else:
            unsafe_recipes = [
                _rel(p) for p in recipes if not _recipe_opted_out_of_full_mode(p)
            ]
            verdict = "HARDCODES_SMOKE_NO_RECIPE_OPT_OUT"
            explanation = (
                "Driver unconditionally passes --smoke flag AND no matching "
                "recipe set fully opts out via research_only/smoke_only/"
                "dispatch_enabled=false. ACTIVE BUG CLASS — driver cannot "
                f"honor a full-mode recipe. Unsafe recipes: {unsafe_recipes}"
            )
    elif consumes_env and not has_hardcoded and not has_conditional_smoke:
        # Driver references env var but does not pass --smoke anywhere
        # (literal nor conditional accumulator).
        verdict = "CONSUMES_ENV_NO_HARDCODE"
        explanation = (
            f"Driver references env var {env_var} but does not literally "
            f"hardcode --smoke (may use conditional accumulator pattern)."
        )
    else:
        # consumes_env AND (has_hardcoded OR has_conditional_smoke) — most
        # likely Z6 pattern with SMOKE_FLAG_ARGS+=(--smoke) inside
        # `if [ "$SMOKE_ONLY" = "1" ]`
        default_values = {
            var: value
            for var in env_vars
            if (value := _driver_env_default_value(text, var)) is not None
        }
        default_value = default_values.get(env_var) if env_var else None
        # Check per-recipe: does each individual recipe opt out OR override env?
        recipe_status: list[dict[str, Any]] = []
        driver_default_is_full = any(
            _mode_env_value_forces_full(var, value)
            for var, value in default_values.items()
        )
        driver_default_is_smoke = any(
            _mode_env_value_forces_smoke(var, value)
            for var, value in default_values.items()
        )
        for rp in recipes:
            rp_rel = _rel(rp)
            rp_opted_out = _recipe_opted_out_of_full_mode(rp)
            rp_env_values = _recipe_mode_env_values(rp, env_vars)
            rp_env_full = any(
                _mode_env_value_forces_full(var, value)
                for var, value in rp_env_values.items()
            )
            # A recipe is "safe" if (a) opted out, (b) sets env to full, or
            # (c) driver default is already full.
            safe = rp_opted_out or rp_env_full or driver_default_is_full
            recipe_status.append({
                "recipe": rp_rel,
                "opted_out_of_full_mode": rp_opted_out,
                "env_var_values_in_recipe": rp_env_values,
                "env_var_forces_full": rp_env_full,
                "safe_for_driver_smoke_default": safe,
            })
        if driver_default_is_full and not driver_default_is_smoke:
            verdict = "CONSUMES_ENV_DEFAULTS_FULL"
            explanation = (
                f"Driver consumes env vars {list(env_vars)} with full-mode defaults "
                f"(full-mode); recipes can opt into smoke explicitly."
            )
        elif driver_default_is_smoke:
            # Bug class fires if ANY recipe is neither opted-out nor overrides env to full
            unsafe_recipes = [r["recipe"] for r in recipe_status if not r["safe_for_driver_smoke_default"]]
            if unsafe_recipes:
                verdict = "CONSUMES_ENV_DEFAULTS_SMOKE_BUG_CLASS"
                explanation = (
                    f"Driver consumes env vars {list(env_vars)} with defaults {default_values} "
                    f"(smoke-mode) AND at least one matching recipe does NOT override "
                    f"the env var to full AND does NOT opt out of full-mode. "
                    f"Z6-v2 Wave 2 BUG CLASS — unsafe recipes: {unsafe_recipes}"
                )
            elif recipes:
                verdict = "CONSUMES_ENV_DEFAULTS_SMOKE_RECIPE_OK"
                explanation = (
                    f"Driver consumes env vars {list(env_vars)} with defaults {default_values} "
                    f"(smoke-mode); ALL matching recipes either explicitly set full OR "
                    f"opt out of full-mode entirely. Acceptable."
                )
            else:
                verdict = "CONSUMES_ENV_DEFAULTS_SMOKE_NO_RECIPE"
                explanation = (
                    f"Driver consumes env vars {list(env_vars)} with defaults {default_values} "
                    f"(smoke-mode) but no matching recipe was found. Out of scope."
                )
        else:
            # No statically-determinable default. Check if ALL matching recipes
            # are safe (either opted out or explicitly override env). If so,
            # this is acceptable — the driver uses a multi-key mode-resolution
            # pattern (e.g. Z6's Z6_TRAINER_MODE > SMOKE_ONLY > default).
            all_recipes_safe = all(r["safe_for_driver_smoke_default"] for r in recipe_status) if recipe_status else False
            if all_recipes_safe and recipes:
                verdict = "CONSUMES_ENV_MULTI_KEY_DEFAULT_RECIPE_OK"
                explanation = (
                    f"Driver consumes env vars {list(env_vars)} via multi-key mode resolution "
                    f"(no single statically-determinable default); all matching recipes "
                    f"are safe (either opt out OR explicitly override env to full)."
                )
            else:
                verdict = "CONSUMES_ENV_UNKNOWN_DEFAULT_BUG_CLASS"
                explanation = (
                    f"Driver consumes env vars {list(env_vars)} but default value could not be "
                    f"determined statically AND at least one matching recipe does NOT "
                    f"override the env var to full AND does NOT opt out of full-mode."
                )
        # Attach per-recipe detail to row
        return {
            "substrate_id": substrate_id,
            "driver_path": _rel(driver_path),
            "recipes": recipe_paths_str,
            "verdict": verdict,
            "explanation": explanation,
            "any_recipe_opted_out_of_full_mode": any_recipe_opted_out,
            "consumes_env_var": env_var,
            "consumes_env_vars": list(env_vars),
            "env_var_default_in_driver": default_value,
            "env_var_defaults_in_driver": default_values,
            "per_recipe_safety": recipe_status,
        }
    return {
        "substrate_id": substrate_id,
        "driver_path": _rel(driver_path),
        "recipes": recipe_paths_str,
        "verdict": verdict,
        "explanation": explanation,
        "any_recipe_opted_out_of_full_mode": any_recipe_opted_out,
        "consumes_env_var": env_var,
        "consumes_env_vars": list(env_vars),
        "env_var_default_in_driver": (
            _driver_env_default_value(text, env_var) if (consumes_env and env_var) else None
        ),
    }


def audit_all_drivers(repo_root: Path | None = None) -> dict[str, Any]:
    if repo_root is None:
        repo_root = _DEFAULT_REPO_ROOT
    repo_root = Path(repo_root).resolve()
    drivers = sorted((repo_root / "scripts").glob("remote_lane_substrate_*.sh"))
    rows = [_classify_driver(d, repo_root) for d in drivers]
    verdict_counts: dict[str, int] = {}
    for row in rows:
        verdict_counts[row["verdict"]] = verdict_counts.get(row["verdict"], 0) + 1
    bug_class_rows = [
        row for row in rows
        if row["verdict"] in (
            "HARDCODES_SMOKE_NO_RECIPE_OPT_OUT",
            "CONSUMES_ENV_DEFAULTS_SMOKE_BUG_CLASS",
            "CONSUMES_ENV_UNKNOWN_DEFAULT_BUG_CLASS",
            "REFUSES_NON_SMOKE_RECIPE_FORCES_FULL_BUG_CLASS",
            "FULL_MODE_DEVICE_CPU_BUG_CLASS",
        )
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "audit_tool": "tools/audit_substrate_driver_mode_hardcode.py",
        "catalog_reference": "Catalog #326 (NEW) + DRIVER-FIX wave 2026-05-18 anchor",
        "empirical_anchor_z6_v2_wave_2_defer": {
            "smoke_call_id": "fc-01KRW7RHFHP640BHTQ0FZM3M38",
            "full_canary_call_id": "fc-01KRW7ZCYK5XF6MSHD24R71A46",
            "date_utc": "2026-05-18",
            "root_cause": (
                "Z6-v2 Wave 2 recipe env_overrides did NOT set SMOKE_ONLY=0; "
                "driver default SMOKE_ONLY=1 produced --smoke flag; trainer "
                "entered _smoke_main with synthetic-cfg overriding council-binding spec."
            ),
        },
        "total_drivers_scanned": len(rows),
        "verdict_counts": verdict_counts,
        "bug_class_count": len(bug_class_rows),
        "bug_class_drivers": [r["substrate_id"] for r in bug_class_rows],
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit substrate drivers for smoke/full mode hardcoding bug class.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write report JSON to .omx/state/substrate_driver_mode_hardcode_audit_<utc>.json",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_DEFAULT_REPO_ROOT,
        help="Repo root path (default: auto-detect).",
    )
    parser.add_argument(
        "--format",
        choices=("json", "summary"),
        default="summary",
        help="Output format (default: summary).",
    )
    args = parser.parse_args()

    report = audit_all_drivers(args.repo_root)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("=" * 70)
        print("Substrate driver mode-hardcode audit (Catalog #326)")
        print(f"Generated: {report['generated_at_utc']}")
        print(f"Drivers scanned: {report['total_drivers_scanned']}")
        print()
        print("Verdict counts:")
        for verdict, count in sorted(report["verdict_counts"].items()):
            marker = " ← BUG CLASS" if "BUG_CLASS" in verdict or "NO_RECIPE_OPT_OUT" in verdict else ""
            print(f"  {verdict:50s} {count:3d}{marker}")
        print()
        print(f"Bug class count: {report['bug_class_count']}")
        if report["bug_class_drivers"]:
            print("Bug class drivers:")
            for sid in report["bug_class_drivers"]:
                print(f"  - {sid}")

    if args.apply:
        utc_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = args.repo_root / ".omx/state" / f"substrate_driver_mode_hardcode_audit_{utc_stamp}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote: {out_path.relative_to(args.repo_root)}")
        return 0 if report["bug_class_count"] == 0 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
