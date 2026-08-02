#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# no-argparse-OK: git hook entrypoint — controlled via env vars (PREFLIGHT_HOOK_ENABLED/REVIEW_GATE_ENABLED/etc)
"""Pre-commit / pre-push hook: runs preflight + ruff undefined-name check.

This is the cheap-and-fast safety net that catches SHIRAZ-class bugs at
commit time so they never reach a GPU.

Layers (in order):
  1. ruff F821 (undefined name) — catches NameError-class bugs like the
     auth_eval `expected_raw` scope leak that crashed every authoritative
     evaluation for weeks.
  2. tac.preflight --scope dev — the bounded developer validator stack.
     PREFLIGHT_FULL=1 switches to the exhaustive release/custody stack.
  3. CI-blind tests — what GitHub Actions structurally cannot execute (no Linux mlx
     wheel => pytest.importorskip skips it and reports GREEN), restricted to what is
     reachable from the staged files. This hook is their ONLY automated surface.
     Granularity matters: a MODULE-scope mlx gate hides the whole file from CI, an
     IN-TEST gate hides only those tests. The step runs whole files for the first and
     node ids for the second — running whole files for both duplicated CI instead of
     covering it (see `_mlx_gate_scope`).
  4. Hands off to review_gate_hook for the standard review-tracker check.

Install:
    ln -sf ../../tools/preflight_hook.py .git/hooks/pre-commit
    ln -sf ../../tools/preflight_hook.py .git/hooks/pre-push
    chmod +x tools/preflight_hook.py

Environment overrides:
    PREFLIGHT_HOOK_ENABLED=0   Skip preflight (review gate still runs)
    PREFLIGHT_FULL=1           Run full whole-repo preflight instead of fast mode
    PREFLIGHT_ALLOW_SLOW=1     Explicitly allow slow release/custody preflight
    PREFLIGHT_TIMEOUT_SECONDS  Override preflight subprocess timeout
    REVIEW_GATE_ENABLED=0      Skip review gate
    REVIEW_GATE_OVERRIDE=1     Override review gate (still runs preflight)
    PREFLIGHT_SKIP_RUFF=1      Skip ruff F821 step (e.g., when ruff missing)
    PREFLIGHT_SKIP_CI_BLIND_TESTS=1  Skip the CI-blind test step (NOT recommended)
    PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS  Override that step's subprocess timeout
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ANSI colors
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
BOLD = "\033[1m"
RST = "\033[0m"


def _staged_py_files() -> list[str]:
    """Return staged .py files relative to repo root."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=REPO_ROOT,
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [f for f in out.splitlines() if f.endswith(".py") and (REPO_ROOT / f).exists()]


def run_ruff_undefined_name(staged: list[str]) -> int:
    """Run ruff on staged .py files, fail on F821 (undefined name) only.

    F821 is the rule that catches scope-leak bugs like the auth_eval
    `expected_raw` NameError. Keep this isolated from project-level broad-lint
    ignores so per-file style carve-outs cannot suppress undefined-name checks.
    """
    if not staged or os.environ.get("PREFLIGHT_SKIP_RUFF") == "1":
        return 0
    try:
        result = subprocess.run(
            [
                ".venv/bin/ruff",
                "check",
                "--isolated",
                "--force-exclude",
                "--select",
                "F821",
                "--ignore-noqa",
                "--exclude",
                "experiments/archive",
                "--exclude",
                "experiments/results",
                "--no-cache",
                *staged,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        # ruff not installed — soft-fail (preflight still catches drift)
        print(f"{YELLOW}[preflight-hook] ruff missing, skipping F821 check{RST}",
              file=sys.stderr)
        return 0
    if result.returncode != 0:
        print(f"\n{RED}{BOLD}[preflight-hook] BLOCKED: undefined-name (F821) found{RST}")
        print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print(f"\n{RED}This is the bug class that hid the auth_eval `expected_raw` "
              f"NameError for weeks.{RST}", file=sys.stderr)
        print("  Fix the undefined name and re-stage.", file=sys.stderr)
        print("  Skip (NOT recommended): PREFLIGHT_SKIP_RUFF=1 git commit ...",
              file=sys.stderr)
        return 1
    return 0


def _ci_blind_test_modules() -> list[Path]:
    """Test modules that GitHub Actions CI structurally CANNOT execute.

    `.github/workflows/ci.yml` runs pytest on ubuntu-24.04 and installs `.[dev,runtime]`.
    `mlx` is a SEPARATE optional extra (pyproject `mlx = ["mlx>=0.5"]`) with no Linux
    wheels, so every module whose import guard is `pytest.importorskip("mlx...")` is
    SKIPPED there — and pytest reports a skip as green. Those modules therefore have NO
    automated surface at all: CI cannot run them, and this hook (until this step) ran no
    tests. `test_ddm_tb1_tr1_renderer.py` sat red on main for ~2 days that way.

    Detection is static (never imports the modules it guards): a cheap substring
    prefilter, then an AST confirmation that an actual `importorskip("mlx...")` CALL
    exists. The AST step matters — a test that merely mentions the guard inside a string
    literal (this file's own tests do) is not itself MLX-gated, and a substring-only
    detector would mis-classify it.

    CAUTION (MEASURED 2026-08-01): membership here does NOT mean CI skips the whole
    module. It means the module contains an mlx gate SOMEWHERE. Of the 57 members, 25
    gate per-test, so GitHub Actions imports them, collects every test, and skips only
    the gated ones. `_mlx_gate_scope` draws that line and `_ci_blind_targets_for` acts
    on it; this function stays deliberately broad so the scope decision has one owner.
    """
    tests_dir = REPO_ROOT / "src" / "tac" / "tests"
    if not tests_dir.is_dir():
        return []
    blind: list[Path] = []
    for path in sorted(tests_dir.glob("test_*.py")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "importorskip" not in text or "mlx" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            blind.append(path)  # unparseable => assume gated; ties go to running it
            continue
        if _has_mlx_importorskip_call(tree):
            blind.append(path)
    return blind


def _is_mlx_importorskip_call(node: ast.AST) -> bool:
    """True for a CALL of importorskip("mlx...") — any of the observed forms
    (`pytest.importorskip(...)`, a bare `importorskip(...)`, with or without `reason=`)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if name != "importorskip" or not node.args:
        return False
    first = node.args[0]
    return (isinstance(first, ast.Constant) and isinstance(first.value, str)
            and first.value.split(".")[0] == "mlx")


def _has_mlx_importorskip_call(tree: ast.AST) -> bool:
    """True when the module CALLS importorskip("mlx...") ANYWHERE (module or test scope)."""
    return any(_is_mlx_importorskip_call(n) for n in ast.walk(tree))


def _mlx_gate_scope(tree: ast.Module) -> str:
    """Where the module's MLX skip-gate sits: "module" or "test".

    This distinction decides whether GitHub Actions runs the module AT ALL, and it is
    the difference between a step that covers CI's blind spot and one that duplicates
    CI's work:

    * "module" — an `importorskip("mlx...")` (or an mlx-conditioned `pytestmark` /
      module-level `pytest.skip(allow_module_level=True)`) executes at IMPORT time, so
      CI collects ZERO tests from the file. The whole module is CI-blind.
    * "test" — the gate sits INSIDE test bodies, so CI imports the module, collects
      every test, and skips only the gated ones. The rest are already covered there.

    MEASURED 2026-08-01 (denominator: the 57 modules `_ci_blind_test_modules()` returns):
    32 are module-scope, 25 are test-scope. A `--collect-only` run with every `mlx*`
    import blocked (simulating the Linux CI wheel gap) collected 0 tests from all 32
    module-scope modules and 769 tests from the 25 test-scope ones — i.e. those 769 run
    in GitHub Actions on every push. Treating all 57 as wholesale-blind made this hook
    re-run them locally: `test_compact_renderer_mlx_spine_runner.py` contributes 309 of
    the 769 and costs 664s standalone (ddm_tr6 measured 718s for it under different
    load), on 9.4% of git-tracked single-file commits — the largest selection footprint
    of the 57. Its 8 genuinely-gated tests account for 24.2s of that 664s.
    """
    for node in _module_scope_nodes(tree):
        if _is_mlx_importorskip_call(node):
            return "module"
        # `pytest.skip(..., allow_module_level=True)` at module scope
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name == "skip" and any(kw.arg == "allow_module_level" for kw in node.keywords):
                return "module"
        # an mlx-conditioned `pytestmark` skips the whole module on CI too
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, "id", None) == "pytestmark" for t in targets) \
                    and _mentions_mlx(node):
                return "module"
    return "test"


def _module_scope_nodes(tree: ast.Module):
    """Every node reachable from module scope, PRUNED at any def/class body.

    `ast.walk` would descend into test bodies and erase exactly the distinction
    `_mlx_gate_scope` exists to draw.
    """
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _mentions_mlx(node: ast.AST) -> bool:
    """True when any identifier / attribute / string literal under `node` says "mlx".

    Used for the module-scope `pytestmark` probe, where the question is only "is this
    mark conditioned on MLX at all". Deliberately loose there; NOT used to decide which
    individual tests run — see `_carries_mlx_skip_gate` for that.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and "mlx" in sub.id.lower():
            return True
        if isinstance(sub, ast.Attribute) and "mlx" in sub.attr.lower():
            return True
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                and "mlx" in sub.value.lower():
            return True
        if isinstance(sub, ast.arg) and "mlx" in sub.arg.lower():
            return True
        if isinstance(sub, ast.alias) and "mlx" in (sub.name or "").lower():
            return True
    return False


def _carries_mlx_skip_gate(node: ast.AST) -> bool:
    """True when executing `node` can itself trigger the MLX skip that CI hits.

    The question is not "does this mention MLX" — a monkeypatched test may name a dozen
    `..._mlx_...` helpers and still RUN on Linux CI. It is "does this SKIP on CI", whose
    only sources are:
      * `pytest.importorskip("mlx...")` executed in the body,
      * a real `import mlx` / `from mlx import ...` (ImportError on Linux),
      * a decorator conditioned on MLX (`@pytest.mark.skipif(not _MLX_AVAILABLE, ...)`).
    Selecting on the mention instead of the gate re-runs CI's work: MEASURED on
    `test_compact_renderer_mlx_spine_runner.py`, mention-based tainting selects 177 of
    302 tests where gate-based tainting selects 4.
    """
    for sub in ast.walk(node):
        if _is_mlx_importorskip_call(sub):
            return True
        if isinstance(sub, ast.Import):
            if any((a.name or "").split(".")[0] == "mlx" for a in sub.names):
                return True
        if isinstance(sub, ast.ImportFrom):
            if (sub.module or "").split(".")[0] == "mlx":
                return True
    decorators = getattr(node, "decorator_list", None) or []
    return any(_mentions_mlx(dec) for dec in decorators)


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name:
                names.add(name)
    return names


def _mlx_gated_test_names(tree: ast.Module) -> list[str]:
    """pytest name selectors for the tests a test-scope module hides from CI.

    Transitive by fixpoint over the module's own defs: a helper that carries the gate
    taints every test that calls it, one or many hops away, and a FIXTURE that carries
    the gate taints every test that requests it by argument name (there is no conftest.py
    under src/tac/tests, so a module's own defs close the fixture graph).
    """
    # name -> every def with that name: two classes can each define `test_x`, and if
    # EITHER carries the gate the name is tainted. Keeping only the first would silently
    # drop the second from the selection.
    defs: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.setdefault(node.name, []).append(node)
    tainted = {name for name, nodes in defs.items()
               if any(_carries_mlx_skip_gate(n) for n in nodes)}
    changed = True
    while changed:
        changed = False
        for name, nodes in defs.items():
            if name in tainted:
                continue
            for node in nodes:
                params = {a.arg for a in node.args.args} | \
                    {a.arg for a in node.args.kwonlyargs}
                if (_called_names(node) & tainted) or (params & tainted):
                    tainted.add(name)
                    changed = True
                    break

    selected: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name.startswith("test") and node.name in tainted:
            selected.append(node.name)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and sub.name.startswith("test") and sub.name in tainted:
                    selected.append(f"{node.name}::{sub.name}")
    return selected


def _module_reference_tokens(rel_path: str) -> set[str]:
    """Every importable dotted name a test could use to reach `rel_path`.

    `src/tac/witness_dsl/qa84_rowband_grammar.py` ->
        {"tac.witness_dsl.qa84_rowband_grammar", "witness_dsl.qa84_rowband_grammar",
         "qa84_rowband_grammar"}
    All trailing sub-paths are included because `src/` is on sys.path and tests import by
    package path, by sub-package, or by bare module name depending on their own sys.path
    surgery. No heuristics/thresholds: an extra match only costs runtime, never silence.
    """
    if not rel_path:
        return set()
    parts = Path(rel_path).with_suffix("").parts
    if parts and parts[0] == "src":
        parts = parts[1:]
    return {".".join(parts[i:]) for i in range(len(parts))}


def _select_ci_blind_tests(staged: list[str]) -> list[str]:
    """CI-blind test modules that reference any staged module (or are staged themselves)."""
    blind = _ci_blind_test_modules()
    if not blind or not staged:
        return []
    staged_set = {str(REPO_ROOT / s) for s in staged}
    tokens: set[str] = set()
    for rel in staged:
        tokens |= _module_reference_tokens(rel)
    selected: list[str] = []
    # Whole-word match, not substring: a bare stem like "io" or "foo" must appear as a
    # name, not inside another identifier. Over-matching costs only runtime; UNDER-matching
    # costs silence, so ties go to running the test.
    pattern = re.compile("|".join(rf"\b{re.escape(t)}\b" for t in sorted(tokens))) \
        if tokens else None
    for path in blind:
        staged_here = str(path) in staged_set
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            # Unreadable: a STAGED blind module must still be run whole (dropping it
            # would be the silence this step exists to remove); an unstaged one cannot
            # be matched against the tokens at all.
            if staged_here:
                selected.append(str(path.relative_to(REPO_ROOT)))
            continue
        if not staged_here and (pattern is None or not pattern.search(text)):
            continue
        selected.extend(_ci_blind_targets_for(path, text))
    return selected


def _ci_blind_targets_for(path: Path, text: str) -> list[str]:
    """pytest targets for one selected module: the whole file, or just its gated tests.

    A module-scope gate makes GitHub Actions collect NOTHING from the file, so the whole
    file is this hook's job. A test-scope gate leaves CI running every non-gated test, so
    running the whole file here duplicates CI instead of covering it — only the gated
    node ids are this hook's job.

    Fails SAFE in both directions: an unparseable file, or a test-scope module where the
    static scan finds no gated test (the gate hides behind something this scan cannot
    see), falls back to the whole file rather than silently running less.
    """
    rel = str(path.relative_to(REPO_ROOT))
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [rel]
    if _mlx_gate_scope(tree) == "module":
        return [rel]
    names = _mlx_gated_test_names(tree)
    if not names:
        return [rel]
    return [f"{rel}::{name}" for name in names]


# --------------------------------------------------------------------------------------
# CI-blind step wall-clock ceiling.
#
# This is a CONTROL knob, not a tuning constant: it decides which commits SURVIVE the
# hook, so it acts on the SELECTION channel (see task #847 and
# `governance_knobs_are_unladdered_control_provenance_20260801`). It therefore carries a
# derivation, an owner, and a re-derivation trigger rather than a chosen number.
#
# DERIVATION (all MEASURED 2026-08-01 on Primary.local, `-m "not slow"`):
#   * The step's cost is the cost of the SELECTED targets in ONE pytest invocation.
#     After the gate-scope split above, the worst case is the largest module-scope
#     module plus every test-scope node id that can be co-selected.
#   * Largest module-scope module measured alone: test_micro_batch_bit_identity_probe.py
#     = 64 passed in 214.4s wall (ddm_tr6 measured 257s for the same module under
#     different load on 2026-08-01 -> run-to-run spread 257/214 = 1.20x).
#   * Second input: the pre-split worst case, both slow modules in one invocation,
#     was 975s (ddm_tr6). That is what the pre-split ceiling had to cover.
#   * Ceiling = pre-split worst case 975s x the measured 1.20x load spread = 1170s,
#     rounded up to the next minute = 1200s. Deliberately derived against the PRE-split
#     worst case, not the post-split one: the split reduces what the step actually runs,
#     and a bound that only just covers the reduced set would re-fire the moment a new
#     module-scope module lands. Headroom here costs nothing on a green run (the bound
#     is a timeout, not a budget) and costs a false BLOCK when it is too small.
# OWNER: the pre-commit hook surface (tools/preflight_hook.py).
# RE-DERIVATION TRIGGER: any of (a) a CI-blind module measured above 600s alone,
#   (b) the measured p99 of the step's wall clock exceeding half this bound,
#   (c) the gate-scope split being reverted or narrowed. `test_preflight_hook.py`
#   pins this constant to the measurement above, so lowering it fails a test.
_CI_BLIND_PRE_SPLIT_WORST_MEASURED_SECONDS = 975
_CI_BLIND_LOAD_SPREAD = 1.20
_CI_BLIND_TIMEOUT_DEFAULT_SECONDS = 1200


def _ci_blind_timeout_seconds() -> int:
    raw = os.environ.get("PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        return value if value > 0 else _CI_BLIND_TIMEOUT_DEFAULT_SECONDS
    return _CI_BLIND_TIMEOUT_DEFAULT_SECONDS


def run_ci_blind_tests(staged: list[str]) -> int:
    """Run the CI-blind (MLX-gated) tests reachable from the staged files.

    This is the ONLY automated moment on the only machine that can execute them, so it
    BLOCKS on red and on timeout rather than warning: a soft-pass here would re-create
    the exact silence it exists to remove (skip-as-green).
    """
    if os.environ.get("PREFLIGHT_SKIP_CI_BLIND_TESTS") == "1":
        return 0
    selected = _select_ci_blind_tests(staged)
    if not selected:
        return 0
    timeout = _ci_blind_timeout_seconds()
    try:
        result = subprocess.run(
            [".venv/bin/python", "-m", "pytest", *selected,
             "-q", "--no-header", "-m", "not slow"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        print(f"{YELLOW}[preflight-hook] .venv missing, skipping CI-blind tests{RST}",
              file=sys.stderr)
        return 0
    except subprocess.TimeoutExpired:
        print(f"\n{RED}{BOLD}[preflight-hook] BLOCKED: CI-blind tests timed out{RST}",
              file=sys.stderr)
        print(f"  targets ({len(selected)}): {' '.join(selected[:12])}"
              f"{' ...' if len(selected) > 12 else ''}", file=sys.stderr)
        print(f"  timeout: {timeout}s (PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS overrides)",
              file=sys.stderr)
        return 1
    if result.returncode == 5:
        # pytest EXIT_NOTESTSCOLLECTED: everything in the selection was deselected by
        # `-m "not slow"`. Nothing ran, so nothing is proven — say so instead of
        # reporting the green tick that a bare `!= 0` check would have called a failure
        # and a bare `== 0` check would have called a pass.
        print(f"{YELLOW}[preflight-hook] CI-blind step: no tests collected in "
              f"{len(selected)} selected module(s) (all `slow`?) — nothing verified{RST}",
              file=sys.stderr)
        return 0
    if result.returncode != 0:
        print(f"\n{RED}{BOLD}[preflight-hook] BLOCKED: CI-blind (MLX-gated) test failed{RST}")
        print(result.stdout[-6000:], file=sys.stderr)
        if result.stderr:
            print(result.stderr[-2000:], file=sys.stderr)
        print(f"\n{RED}GitHub Actions CANNOT catch this: mlx has no Linux wheel, so CI "
              f"SKIPS these modules and reports green.{RST}", file=sys.stderr)
        print("  This hook is the only automated surface that runs them. Fix, then commit.",
              file=sys.stderr)
        print("  Skip (NOT recommended): PREFLIGHT_SKIP_CI_BLIND_TESTS=1 git commit ...",
              file=sys.stderr)
        return 1
    # Report the DENOMINATOR, not just a green tick: "OK" over an empty or accidentally
    # narrowed selection is the vacuity failure this step exists to refuse.
    modules = len({t.split("::", 1)[0] for t in selected})
    node_ids = sum(1 for t in selected if "::" in t)
    detail = f"{modules} module(s)"
    if node_ids:
        detail += (f", {node_ids} gated node id(s) — the rest of those modules run in "
                   f"GitHub Actions")
    print(f"{GREEN}[preflight-hook] CI-blind tests OK ({detail}){RST}", file=sys.stderr)
    return 0


def run_preflight() -> int:
    """Run the bounded preflight validator stack.

    Default hook mode is intentionally fast and source-index friendly:
    `tac.preflight --no-codebase` catches artifact/profile wiring without
    scanning every recovered public-PR source tree and reverse-engineering
    custody mirror on each commit. Operators can still request the full
    whole-repo scan with `PREFLIGHT_FULL=1`, but it keeps the normal 30s DX
    budget unless `PREFLIGHT_ALLOW_SLOW=1` is set for a deliberate release or
    custody sweep.
    """
    if os.environ.get("PREFLIGHT_HOOK_ENABLED", "1") == "0":
        return 0
    cmd = _preflight_command()
    timeout = _preflight_timeout_seconds()
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        print(f"{YELLOW}[preflight-hook] .venv missing, skipping preflight{RST}",
              file=sys.stderr)
        return 0
    except subprocess.TimeoutExpired as exc:
        print(f"\n{RED}{BOLD}[preflight-hook] BLOCKED: preflight timed out{RST}",
              file=sys.stderr)
        print(f"  command: {' '.join(cmd)}", file=sys.stderr)
        print(f"  timeout: {timeout}s", file=sys.stderr)
        if exc.stdout:
            print(str(exc.stdout)[-4000:], file=sys.stderr)
        if exc.stderr:
            print(str(exc.stderr)[-4000:], file=sys.stderr)
        print(f"\n{RED}The hook must stay bounded during normal development.{RST}", file=sys.stderr)
        print(
            "  Use PREFLIGHT_ALLOW_SLOW=1 only for deliberate release/custody sweeps.",
            file=sys.stderr,
        )
        return 1
    if result.returncode != 0:
        print(f"\n{RED}{BOLD}[preflight-hook] BLOCKED: preflight failed{RST}")
        print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print(f"\n{RED}A drift / arity / profile / arch / filename rule fired.{RST}",
              file=sys.stderr)
        print("  Fix the issue, then commit.", file=sys.stderr)
        print("  Skip (NOT recommended): PREFLIGHT_HOOK_ENABLED=0 git commit ...",
              file=sys.stderr)
        return 1
    _echo_preflight_scope_coverage(result.stderr)
    return 0


def _echo_preflight_scope_coverage(stderr: str) -> None:
    """Surface the preflight DENOMINATOR where a committer actually reads it.

    A green hook that examined nothing is the exact bug this reports on
    ("vacuity is indistinguishable from PASS"). The CLI emits a VACUOUS ledger
    line; without echoing it here the committer sees only the hook's silence.
    One concise line, not an alarm: the number is the signal.
    """
    for line in (stderr or "").splitlines():
        if "PREFLIGHT VACUOUS" in line:
            print(
                f"{YELLOW}[preflight-hook] preflight examined 0 gates this "
                f"commit (fast --no-codebase mode). This hook is NOT gate "
                f"coverage.{RST}",
                file=sys.stderr,
            )
            print(
                "  Full developer gate set: PREFLIGHT_FULL=0 "
                ".venv/bin/python -m tac.preflight --scope dev",
                file=sys.stderr,
            )
            return


# Everything the hook does OUTSIDE its two timed subprocesses: ruff on the staged files,
# the review gate, and git's own commit work. MEASURED 2026-08-01 (Primary.local):
# ruff 0.014s on one staged file, review gate 0.041s, and .omx/state/commit-serializer.log
# p50 `commit_seconds` = 3.2s (n=9112) in the mode where preflight itself costs 0.52s.
# Rounded up to one minute so the bound stays whole-minute legible.
_HOOK_FIXED_OVERHEAD_SECONDS = 60


def effective_hook_wall_clock_bound_seconds() -> int:
    """Upper bound on how long this hook can hold its caller, UNDER THE CURRENT ENV.

    Public because the commit serializer's lock patience must be derived from it rather
    than re-guessed: the serializer holds `.commit-lock` across `git commit`, so its
    lock timeout has to exceed whatever bound THIS hook is running under — including the
    per-invocation `PREFLIGHT_*_TIMEOUT_SECONDS` overrides, which the child `git commit`
    inherits from the same environment. Reading the bounds here makes that exact instead
    of approximate, and makes any future change to either timeout propagate on its own.
    """
    return (_preflight_timeout_seconds()
            + _ci_blind_timeout_seconds()
            + _HOOK_FIXED_OVERHEAD_SECONDS)


def _preflight_command() -> list[str]:
    """Return the preflight command for the current hook mode."""
    cmd = [".venv/bin/python", "-m", "tac.preflight"]
    if os.environ.get("PREFLIGHT_FULL", "0") == "1":
        cmd.extend(["--scope", "all"])
        if os.environ.get("PREFLIGHT_ALLOW_SLOW", "0") == "1":
            cmd.append("--allow-slow-preflight")
    else:
        cmd.append("--no-codebase")
        # VACUITY ACKNOWLEDGEMENT (2026-08-01, task #842). `--no-codebase`
        # skips EVERY codebase gate call site, so this hook mode examines
        # 0 gates — MEASURED: 0 of 27 declared in 0.52s. The CLI now refuses
        # rc=3 on an empty scope rather than printing a bare "PREFLIGHT
        # PASSED"; this hook is the one caller with a designed reason to
        # accept that, so it must NAME the emptiness rather than inherit it
        # silently. The verdict still prints VACUOUS and `run_preflight`
        # echoes the coverage number on every commit — acknowledgement
        # suppresses the refusal ONLY, never the report.
        cmd.append("--acknowledge-empty-scope")
    return cmd


def _preflight_timeout_seconds() -> int:
    """Return a positive preflight timeout with conservative defaults."""
    raw = os.environ.get("PREFLIGHT_TIMEOUT_SECONDS")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        return value if value > 0 else 30
    if (
        os.environ.get("PREFLIGHT_FULL", "0") == "1"
        and os.environ.get("PREFLIGHT_ALLOW_SLOW", "0") == "1"
    ):
        return 600
    return 30


def run_review_gate() -> int:
    """Hand off to the existing review-tracker gate hook."""
    hook = REPO_ROOT / "tools" / "review_gate_hook.py"
    if not hook.exists():
        return 0
    env = os.environ.copy()
    if _is_pre_push_invocation():
        env.setdefault("REVIEW_GATE_MODE", "pre-push")
    else:
        env.setdefault("REVIEW_GATE_MODE", "pre-commit")
    try:
        result = subprocess.run(
            [".venv/bin/python", str(hook)],
            cwd=REPO_ROOT,
            env=env,
        )
    except FileNotFoundError:
        return 0
    return result.returncode


def _is_pre_push_invocation() -> bool:
    """Return true when this shared hook script was invoked by pre-push."""

    try:
        hook_path = Path(sys.argv[0])
        return hook_path.name == "pre-push"
    except Exception:
        return False


def run_hook_path_heavy_import_scan() -> int:
    """Catalog #184 heavy-import scan, run REGARDLESS of preflight scope.

    Round-1 recursive-adversarial review 2026-08-02 MEASURED that the gate landed
    in 583b3f75a6 sits inside ``check_preflight_hook_codebase_default``, which
    lives in the ``codebase`` scope -- the scope THIS HOOK SKIPS. Verbatim:

        $ python -m tac.preflight --no-codebase --acknowledge-empty-scope
        PREFLIGHT VACUOUS -- examined 0 of 27 declared -- scopes SKIPPED: codebase
        rc: 0    mentions 184/heavy-import: False

    So "unconditional" was unconditional inside a function that never executes --
    the vacuity-equals-pass genus, built into the fix for the bimodal-cost bug it
    was meant to guard. Its 18 unit tests passed the whole time: a passing test
    proves the UNIT, never the WIRING.

    There is no always-run preflight scope to move it into (``--no-codebase``
    examines 0 of 27), so the hook must call it directly. Affordable because the
    same morning's torch deferral cut ``import tac.preflight`` from 43.86 s cold /
    0.48 s warm to 30 ms MEASURED -- this scan is a single-file AST parse on top.

    Fail-OPEN on import error, deliberately: if ``tac`` will not import, step 2
    has already failed and returned, so an import error here means the guard
    itself is broken, and blocking every commit on a broken guard is worse than
    the bug it guards. The message is loud so the fail-open is never silent.
    """

    try:
        from tac.preflight import _check_184_module_scope_heavy_imports as scan
    except Exception as exc:  # pragma: no cover - guard-broken path
        print(
            f"[preflight-hook] heavy-import scan UNAVAILABLE (failing OPEN): {exc}",
            file=sys.stderr,
        )
        return 0

    # ROUND-2 review, 2026-08-02: the scan returns [] for a MISSING target, which is
    # byte-identical to a clean scan -- and the round-1 test suite blessed that as
    # correct ("missing file is a no-op, not a crash"). So a wrong REPO_ROOT would
    # pass SILENTLY: the vacuity genus one layer below the round-1 fix for the
    # vacuity genus. An empty scope is VACUOUS, never a PASS -- report the
    # DENOMINATOR instead of inferring cleanliness from an empty result.
    # ROUND-3 review, 2026-08-02 — the STRUCTURAL fix, after hitting this genus at three
    # layers in three rounds. The cause is not three separate bugs: it is that `[]` is the
    # SUCCESS CHANNEL, so every failure mode that yields empty is indistinguishable from a
    # clean pass. Measured, all returning "clean":
    #   round 1  gate never invoked (wrong scope)     -> examined 0 of 27
    #   round 2  scan target MISSING                  -> scan(...) == []
    #   round 3  scan target UNPARSEABLE (SyntaxError swallowed inside) -> scan(...) == []
    # A fourth existence check would repeat the mistake. Instead, establish the DENOMINATOR
    # independently: prove the target parses HERE, so an empty result means "examined 1 file,
    # found 0 violations" rather than "produced nothing, for one of three reasons."
    target = REPO_ROOT / "src" / "tac" / "preflight.py"
    examined = 0
    try:
        ast.parse(target.read_text(encoding="utf-8"))
        examined = 1
    except FileNotFoundError:
        reason = f"target {target} does not exist"
    except SyntaxError as exc:
        reason = f"target {target} does not parse ({exc.__class__.__name__}: {exc.msg})"
    except Exception as exc:  # pragma: no cover - unreadable target
        reason = f"target {target} unreadable ({exc.__class__.__name__}: {exc})"

    if examined == 0:
        print(
            f"[preflight-hook] heavy-import scan VACUOUS (failing OPEN): {reason} — "
            f"0 files examined. This is NOT a pass; an empty result here means the "
            f"instrument produced nothing, not that the code is clean.",
            file=sys.stderr,
        )
        return 0

    # ROUND-5: scan() raising propagated UNCAUGHT and crashed the whole hook with a
    # traceback — a bug INSIDE the guard would have blocked every commit in the repo,
    # which is the same over-reach the two fail-opens above exist to avoid. Symmetry:
    # a broken guard fails OPEN and LOUD wherever it breaks, never only where I predicted.
    try:
        violations = scan(REPO_ROOT)
    except Exception as exc:  # pragma: no cover - guard-internal bug path
        print(
            f"[preflight-hook] heavy-import scan CRASHED (failing OPEN): "
            f"{exc.__class__.__name__}: {exc} — 0 files examined. This is NOT a pass.",
            file=sys.stderr,
        )
        return 0

    if violations:
        print(
            "[preflight-hook] Catalog #184: module-scope heavy import on the hook "
            "path — this is the 43.86s-cold / rc=124 bug class.",
            file=sys.stderr,
        )
        for v in violations:
            print(f"[preflight-hook]   {v}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    staged = _staged_py_files()

    # Step 1: ruff F821 on staged files only (fast, ~50ms per file)
    rc = run_ruff_undefined_name(staged)
    if rc != 0:
        return rc

    # Step 1b: the heavy-import guard runs BEFORE preflight, not after.
    #
    # ROUND-4 review, 2026-08-02: it was placed after run_preflight(), and run_preflight()
    # EARLY-RETURNS on failure — so the guard was skipped whenever preflight failed. That is
    # circular, because the bug this guard exists to prevent (a module-scope heavy import,
    # 43.86 s cold) is itself a cause of preflight failing/timing out at rc=124. The guard
    # for the timeout was skipped exactly when the timeout fired.
    #
    # My own round-1 test asserted the WRONG order
    # (test_scan_step_runs_after_preflight_so_a_vacuous_preflight_cannot_hide_it): a
    # VACUOUS-but-rc0 preflight indeed cannot hide it, but I reasoned only about the rc0
    # case and never the failure case. Running FIRST satisfies both, and additionally guards
    # preflight's own cost BEFORE preflight pays it. It costs ~30 ms.
    rc = run_hook_path_heavy_import_scan()
    if rc != 0:
        return rc

    # Step 2: bounded developer preflight (PREFLIGHT_FULL=1 for full release scan)
    rc = run_preflight()
    if rc != 0:
        return rc

    # Step 3: CI-blind (MLX-gated) tests reachable from the staged files — the set
    # GitHub Actions structurally cannot execute (skip-as-green).
    rc = run_ci_blind_tests(staged)
    if rc != 0:
        return rc

    # Step 4: review gate
    return run_review_gate()


if __name__ == "__main__":
    sys.exit(main())
