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
    PREFLIGHT_SCOPE=dev|all    Gate scope for this run. `dev` = the bounded
                               developer stack (layer 2 above; ~23s, 60s bound).
                               Unset = the fast 0-gate mode (default, unchanged).
    PREFLIGHT_FULL=1           Run full whole-repo preflight instead of fast mode
                               (equivalent to PREFLIGHT_SCOPE=all; takes precedence)
    PREFLIGHT_ALLOW_SLOW=1     Explicitly allow slow release/custody preflight
    PREFLIGHT_TIMEOUT_SECONDS  Override preflight subprocess timeout
    REVIEW_GATE_ENABLED=0      Skip review gate
    REVIEW_GATE_OVERRIDE=1     Override review gate (still runs preflight)
    PREFLIGHT_SKIP_RUFF=1      Skip ruff F821 step (e.g., when ruff missing)
    PREFLIGHT_SKIP_CI_BLIND_TESTS=1  Skip the CI-blind test step (NOT recommended)
    PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS  Override that step's subprocess timeout
    PREFLIGHT_CI_BLIND_FORCE=1 Run CI-blind tests even while a governed Metal burn is
                               live (default: DEFER with a receipt — see #1107)
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
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


def _staged_added_py_files() -> list[str]:
    """Return newly-added staged Python files for additive-coverage warnings."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=A"],
            cwd=REPO_ROOT,
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [f for f in out.splitlines() if f.endswith(".py") and (REPO_ROOT / f).exists()]


def _staged_doc_files() -> list[str]:
    """Staged .md files.

    Separate from :func:`_staged_py_files` because that one is `.py`-only by
    contract (ruff and the subset-selection AST scan both need real Python). The
    negative-verdict scan needs markdown: measured 2026-08-03, labelled negative
    verdict assertions occur on 80 lines across 53 tracked `.md`, and until this
    landing no commit-time check read a `.md` file at all.

    `.py` is deliberately NOT included -- see `tac.negative_verdict_gate.
    staged_files` for the measured precision reason.
    """
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=REPO_ROOT,
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [
        f
        for f in out.splitlines()
        if f.endswith(".md") and (REPO_ROOT / f).exists()
    ]


def _staged_shell_files() -> list[str]:
    """Staged shell scripts relative to repo root."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=REPO_ROOT,
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [
        f
        for f in out.splitlines()
        if f.endswith(".sh") and (REPO_ROOT / f).exists()
    ]


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

    A package `__init__.py` resolves to its PACKAGE names, not to `...__init__`
    (`src/tac/verdicts/__init__.py` -> {"tac.verdicts", "verdicts"}). Measured 2026-08-03
    (`ddm_vw1`, task #936): keeping the trailing component was wrong in BOTH directions.
    It over-matched, because the bare stem `__init__` is a whole-word hit in essentially
    every Python file — staging one 2-line `__init__.py` selected 14 heavy MLX GPU test
    modules, and `14 of 14` were matched SOLELY by that token while `0 of 14` referenced
    the staged package at all. Under GPU contention from a concurrent MLX run that
    surfaced as a `Fatal Python error: Bus error`, i.e. a flaky hard block on every
    `__init__.py` edit. It also UNDER-matched, which is the failure this step exists to
    prevent: a test doing `from tac.verdicts import X` was never matched by the token
    `tac.verdicts.__init__`, so real importers were silently skipped.

    Nested package `__init__.py` files add a second over-match class: the bare leaf token can
    be a common domain word. Measured 2026-08-06 (`ddm_cb2`, task #983), staging
    `src/tac/pr130_lift/pose/__init__.py` emitted the token `pose`, selecting 29 unrelated
    MLX-gated targets whose only match was ordinary pose terminology. Dropping only that
    nested bare leaf keeps the importable package suffix (`pr130_lift.pose`) while removing
    the generic domain-word match.
    """
    if not rel_path:
        return set()
    parts = Path(rel_path).with_suffix("").parts
    if parts and parts[0] == "src":
        parts = parts[1:]
    package_init = len(parts) > 1 and parts[-1] == "__init__"
    if package_init:
        parts = parts[:-1]
    tokens: set[str] = set()
    for i in range(len(parts)):
        if package_init and len(parts) > 2 and i == len(parts) - 1:
            continue
        tokens.add(".".join(parts[i:]))
    return tokens


def _select_ci_blind_tests(staged: list[str]) -> list[str]:
    """CI-blind test modules that reference any staged module (or are staged themselves)."""
    blind = _ci_blind_test_modules()
    if not blind or not staged:
        return []
    # Seal-pinned custody trees (candidate_seal.v1) are byte-frozen copies of
    # SHIPPED runtime code. Their basenames (inflate, hpac_inference, ...) are
    # not modules any local test covers — there is no change to cover — and
    # token-matching them pulls unrelated heavy MLX modules into this step
    # (the #936 selector over-pull class; jg5 custody commit, 2026-08-20).
    # Custody coverage lives in the T4 identity receipt + the seal itself.
    # Function-local import: the hook path stays free of module-scope weight
    # (Catalog #184). Failure falls toward scanning MORE, never less.
    try:
        from tac.subset_selection_gate import (
            _under_custody_dir,
            seal_pinned_custody_dirs,
        )
        _custody = seal_pinned_custody_dirs(REPO_ROOT)
        if _custody:
            staged = [s for s in staged if not _under_custody_dir(s, _custody)]
    except Exception:
        pass
    if not staged:
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


_GOVERNED_METAL_TRAINER_SIGNATURES = (
    "train_tr1_partition_renderer_mlx.py",
    "train_levelset_witness_realized_through_R_mlx.py",
    "train_witness_realized_through_R_mlx.py",
)


def _governed_metal_burn_pids() -> list[str]:
    """PIDs of live governed Metal trainers (empty list = device free).

    The CI-blind pytest step allocates MLX/Metal buffers; running it while a governed
    burn holds the device produced hard SIGSEGV crashes twice on 2026-08-17 (#1107) —
    false-red blocks on commits that touched nothing MLX-related. The hook therefore
    DEFERS loudly with a receipt instead of racing the device (an MLX-allocating test
    suite counts as a Metal fire under the one-Metal-fire law).
    """
    pids: list[str] = []
    for sig in _GOVERNED_METAL_TRAINER_SIGNATURES:
        try:
            out = subprocess.run(["pgrep", "-f", sig], capture_output=True, text=True,
                                 timeout=10)
        except Exception:
            continue
        pids.extend(p.strip() for p in out.stdout.split() if p.strip())
    return sorted(set(pids))


def _record_ci_blind_deferral(selected: list[str], burn_pids: list[str]) -> None:
    """Append a machine-readable deferral receipt — the burn-end re-verify queue.

    A deferral WITHOUT a durable receipt would be skip-as-green wearing a log line;
    this row is what the burn-end re-verify consumes, so the deferral is a debt with
    an address, not a silence.
    """
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=REPO_ROOT, timeout=10).stdout.strip()
    except Exception:
        head = "unknown"
    row = {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "head_at_deferral": head,
        "burn_pids": burn_pids,
        "deferred_targets": selected,
        "reverify_cmd": (".venv/bin/python -m pytest " + " ".join(selected)
                         + " -q -m 'not slow'"),
    }
    path = REPO_ROOT / ".omx" / "state" / "ci_blind_deferred.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception as exc:  # the receipt failing must not hide the deferral itself
        print(f"{YELLOW}[preflight-hook] could not write deferral receipt: {exc}{RST}",
              file=sys.stderr)


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
    if os.environ.get("PREFLIGHT_CI_BLIND_FORCE") != "1":
        burn_pids = _governed_metal_burn_pids()
        if burn_pids:
            _record_ci_blind_deferral(selected, burn_pids)
            print(f"{YELLOW}{BOLD}[preflight-hook] CI-blind step DEFERRED — NOT "
                  f"verified: {len(selected)} target(s) not run; governed Metal burn "
                  f"live (pids {', '.join(burn_pids)}){RST}", file=sys.stderr)
            print("  MLX-allocating tests segfault under a live Metal burn (#1107, "
                  "measured 2x on 2026-08-17). Receipt appended to "
                  ".omx/state/ci_blind_deferred.jsonl — re-verify at burn end.",
                  file=sys.stderr)
            print("  Force anyway: PREFLIGHT_CI_BLIND_FORCE=1 git commit ...",
                  file=sys.stderr)
            return 0
    budget = _ci_blind_timeout_seconds()
    # ONE pytest process for every selected target CO-LOADS their MLX modules, and
    # that co-load is its own failure class (#1154): ddm_cu1 measured a
    # `Fatal Python error: Bus error` across 36 co-selected targets where the
    # 11th "passes standalone in 0.47s" — no individual test was at fault. A
    # co-load crash reads as a red commit and forced a documented skip during the
    # oc2 consolidation. Running each MODULE in its own subprocess extincts the
    # class: a module that crashes ALONE is a real red, and a crash that only
    # appears under co-load can no longer happen at all.
    by_module: dict[str, list[str]] = {}
    for target in selected:
        by_module.setdefault(target.split("::", 1)[0], []).append(target)
    modules = len(by_module)
    node_ids = sum(1 for target in selected if "::" in target)

    started = time.monotonic()
    collected_any = False
    per_module_rc: list[tuple[str, int]] = []
    for module, targets in by_module.items():
        remaining = budget - (time.monotonic() - started)
        if remaining <= 0:
            print(f"\n{RED}{BOLD}[preflight-hook] BLOCKED: CI-blind tests exhausted "
                  f"the {budget}s aggregate budget{RST}", file=sys.stderr)
            ran = len(per_module_rc)
            print(f"  {ran}/{modules} module(s) completed; not run: "
                  f"{' '.join(list(by_module)[ran:][:12])}", file=sys.stderr)
            print("  timeout: PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS overrides",
                  file=sys.stderr)
            return 1
        try:
            result = subprocess.run(
                [".venv/bin/python", "-m", "pytest", *targets,
                 "-q", "--no-header", "-m", "not slow"],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=remaining,
            )
        except FileNotFoundError:
            print(f"{YELLOW}[preflight-hook] .venv missing, skipping CI-blind tests{RST}",
                  file=sys.stderr)
            return 0
        except subprocess.TimeoutExpired:
            print(f"\n{RED}{BOLD}[preflight-hook] BLOCKED: CI-blind module timed out{RST}",
                  file=sys.stderr)
            print(f"  module: {module} ({len(targets)} target(s))", file=sys.stderr)
            print(f"  aggregate budget: {budget}s "
                  f"(PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS overrides)", file=sys.stderr)
            return 1
        per_module_rc.append((module, result.returncode))
        if result.returncode == 5:
            # pytest EXIT_NOTESTSCOLLECTED for THIS module: everything in it was
            # deselected by `-m "not slow"`. Nothing ran, so nothing is proven.
            continue
        if result.returncode != 0:
            print(f"\n{RED}{BOLD}[preflight-hook] BLOCKED: CI-blind (MLX-gated) test "
                  f"failed in {module}{RST}")
            print(f"  rc={result.returncode} — this module ran ALONE, so the failure is "
                  f"real and not the #1154 co-load class.", file=sys.stderr)
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
        collected_any = True
    if not collected_any:
        print(f"{YELLOW}[preflight-hook] CI-blind step: no tests collected in "
              f"{modules} selected module(s) (all `slow`?) — nothing verified{RST}",
              file=sys.stderr)
        return 0
    # Report the DENOMINATOR, not just a green tick: "OK" over an empty or accidentally
    # narrowed selection is the vacuity failure this step exists to refuse.
    empty = sum(1 for _module, rc in per_module_rc if rc == 5)
    detail = f"{modules} module(s), isolated"
    if node_ids:
        detail += (f", {node_ids} gated node id(s) — the rest of those modules run in "
                   f"GitHub Actions")
    if empty:
        detail += f"; {empty} module(s) collected nothing"
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
                "  Full developer gate set, through this hook: "
                "PREFLIGHT_SCOPE=dev git commit ...   (or standalone: "
                ".venv/bin/python -m tac.preflight --scope dev)",
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


def _preflight_scope() -> str:
    """Resolve this hook run's gate scope: ``none`` | ``dev`` | ``all`` (task #905).

    THREE modes, not two. Until this branch landed the hook could only choose
    between ``--no-codebase`` (0 gates — MEASURED 0 of 27 declared, in 0.52s)
    and ``--scope all`` (the exhaustive release/custody sweep). The bounded
    developer stack that layer 2 of this module's docstring has advertised since
    it was written was UNREACHABLE FROM THE HOOK: ``tac.preflight``'s own
    ``--scope`` already defaults to ``dev``, but the hook never selected it.

    That absent branch is what made #905 read as "the RED gates block us". The
    only reachable non-empty mode was the release sweep, which scans surfaces a
    commit does not control (paths outside the repo, untracked files), so
    "turn the hook on" meant "block every commit on the wrong scope". The fix is
    the missing path, not a relaxed gate.

    ``PREFLIGHT_FULL=1`` keeps precedence so every existing caller, runbook, and
    test is unchanged, and the DEFAULT is unchanged: this ADDS a mode. No commit
    that passed before blocks now.
    """
    if os.environ.get("PREFLIGHT_FULL", "0") == "1":
        return "all"
    requested = os.environ.get("PREFLIGHT_SCOPE", "").strip().lower()
    if requested in {"dev", "all"}:
        return requested
    return "none"


def _preflight_command() -> list[str]:
    """Return the preflight command for the current hook mode."""
    cmd = [".venv/bin/python", "-m", "tac.preflight"]
    scope = _preflight_scope()
    if scope == "all":
        cmd.extend(["--scope", "all"])
        if os.environ.get("PREFLIGHT_ALLOW_SLOW", "0") == "1":
            cmd.append("--allow-slow-preflight")
    elif scope == "dev":
        cmd.extend(["--scope", "dev"])
        # DELIBERATELY no `--acknowledge-empty-scope` here. That waiver is
        # earned by the 0-gate mode alone, which has a designed reason to accept
        # an empty scope. A dev scope that somehow examined 0 gates is a broken
        # instrument, and it must refuse rc=3 rather than inherit another mode's
        # vacuity waiver — vacuity-indistinguishable-from-PASS is the bug this
        # hook reports on, so it must not be re-imported through the new branch.
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
    scope = _preflight_scope()
    if scope == "all" and os.environ.get("PREFLIGHT_ALLOW_SLOW", "0") == "1":
        return 600
    if scope == "dev":
        # The mirrored half of the #905 gap: the dev scope had no timeout branch
        # either. MEASURED (ddm_rg2, 2026-08-16): 22.7s warm / 24.3s cold. Against
        # the 30s DX bound that is only 19-24% headroom, so one slow-gate
        # regression would fail commits on the CLOCK instead of on a finding —
        # and to a committer who sees "hook failed", a timeout is
        # indistinguishable from a real refusal. Bound derived as 2x the COLD
        # measurement (48.6s), rounded up to a whole minute for legibility.
        return 60
    return 30


def run_review_gate() -> int:
    """Hand off to the existing review-tracker gate hook.

    An ABSENT gate is not a PASSING gate (ddm_si1, task #929). Both early
    returns below used to hand back ``0`` -- the same value a clean review
    returns -- so a deleted/renamed ``review_gate_hook.py`` or a missing
    interpreter silently retired a CLAUDE.md non-negotiable with no output at
    all. The second case is the ddm_ob1 shape exactly: the interpreter is not
    there and the wrapper reports success. Both now refuse and say why.
    Live count at landing: 0 (both paths present).
    """
    hook = REPO_ROOT / "tools" / "review_gate_hook.py"
    if not hook.exists():
        print(
            f"[preflight-hook] REFUSE: review gate missing at {hook}. An absent "
            "gate is not a passing gate; restore it or run the review "
            "explicitly.",
            file=sys.stderr,
        )
        return 1
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
        print(
            "[preflight-hook] REFUSE: could not launch the review gate "
            "(.venv/bin/python not found). The gate did not run, which is not "
            "the same as the gate passing.",
            file=sys.stderr,
        )
        return 1
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
        from tac.preflight import _CHECK_184_HOOK_IMPORT_PATH_MODULES as targets
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
    # ROUND-8: the denominator must be DERIVED from the same source the scan iterates,
    # never hardcoded beside it. It previously pinned "src/tac/preflight.py" literally
    # while the scan walks the _CHECK_184_HOOK_IMPORT_PATH_MODULES TUPLE. They agree at
    # len 1 — but adding a second module would make this claim "examined 1" while the scan
    # covered 2, with the second never parse-verified: a SILENTLY WRONG denominator, which
    # is exactly what a denominator exists to prevent. Coupled now, so it cannot drift.
    examined = 0
    reasons: list[str] = []
    for rel in targets:
        target = REPO_ROOT / rel
        try:
            ast.parse(target.read_text(encoding="utf-8"))
            examined += 1
        except FileNotFoundError:
            reasons.append(f"{rel} does not exist")
        except SyntaxError as exc:
            reasons.append(f"{rel} does not parse ({exc.__class__.__name__}: {exc.msg})")
        except Exception as exc:  # pragma: no cover - unreadable target
            reasons.append(f"{rel} unreadable ({exc.__class__.__name__}: {exc})")

    if examined != len(targets):
        print(
            f"[preflight-hook] heavy-import scan VACUOUS (failing OPEN): "
            f"{'; '.join(reasons)} — {examined} of {len(targets)} files examined. This is "
            f"NOT a pass; an empty result here means the instrument produced nothing, not "
            f"that the code is clean.",
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


def run_subset_selection_scan(staged: list[str]) -> int:
    """`ddm_ss1` landing 2 — refuse a NEWLY-ADDED silent pair/frame subset slice.

    Same reasoning as the Catalog #184 scan directly above, and for the same
    measured reason: `--no-codebase` (this hook's default) examines 0 of 27
    preflight gates, so a gate registered only in `preflight_all()` never runs at
    commit time. Verified 2026-08-03 — even Catalog #401, which is STRICT, does
    not appear in this hook's output.

    Scope is the staged diff's ADDED LINES. That is what lets it be STRICT
    honestly: the repo carries 207 pre-existing sites (136 files, of 10,699
    tracked in-scope .py, measured 2026-08-03), so a whole-repo strict flip would
    refuse every commit. Added-lines scope makes the live count 0 on a clean
    commit while still refusing every new one. The 207 stay visible as debt via
    `tac.subset_selection_gate.scan_repo`.

    Fail-OPEN and LOUD on a broken guard, matching the sibling above: blocking
    every commit because the guard itself broke is worse than the bug it guards.
    """
    if os.environ.get("SUBSET_SELECTION_SCAN_ENABLED", "1") == "0":
        print(
            "[preflight-hook] subset-selection scan DISABLED by env — NOT a pass.",
            file=sys.stderr,
        )
        return 0
    try:
        from tac.subset_selection_gate import scan_staged
    except Exception as exc:  # pragma: no cover - guard-broken path
        print(
            f"[preflight-hook] subset-selection scan UNAVAILABLE (failing OPEN): {exc}",
            file=sys.stderr,
        )
        return 0

    if not staged:
        # No .py staged: nothing to examine. Silent by design — a doc-only commit
        # is not a subset-selection event, and a line here would be pure noise.
        return 0
    try:
        violations = scan_staged(
            repo_root=REPO_ROOT, strict=False, verbose=False, files=staged
        )
    except Exception as exc:  # pragma: no cover - guard-internal bug path
        print(
            f"[preflight-hook] subset-selection scan CRASHED (failing OPEN): "
            f"{exc.__class__.__name__}: {exc} — 0 files examined. This is NOT a pass.",
            file=sys.stderr,
        )
        return 0

    if violations:
        print(
            f"\n{RED}{BOLD}[preflight-hook] BLOCKED: new subset slice with no declared "
            f"selection mode{RST}",
            file=sys.stderr,
        )
        print(
            "  A silent subset IS a video-order prefix, and a prefix measures "
            "2.54-4.21x HARDER\n  than the population on the pose axis "
            "(ddm_mi1 receipt d2853c92090c) — the false-negative shape.",
            file=sys.stderr,
        )
        for v in violations:
            print(f"[preflight-hook]   {v}", file=sys.stderr)
        return 1
    # One terse line stating the DENOMINATOR. The sibling #184 scan is silent on
    # success, but this landing's whole thesis is that a silent instrument is
    # indistinguishable from one that examined nothing — so it says what it read.
    # "in scope", not "examined": scan_staged may skip seal-pinned custody files
    # by design (reported in its own detail), and this line must not claim more
    # than the gate did (m50 denominator honesty; jg5 custody commit 2026-08-20).
    print(
        f"{GREEN}[preflight-hook] subset-selection: {len(staged)} staged .py "
        f"in scope, no new silent subsets{RST}",
        file=sys.stderr,
    )
    return 0


def run_negative_verdict_scan(staged_docs: list[str]) -> int:
    """`ddm_ks1` — refuse a NEWLY-ASSERTED negative verdict with no declared scope.

    Why this is a hook step and not a preflight gate: two STRICT kill-verdict
    gates already exist — Catalog #307
    (`check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification`)
    and #308 (`check_kill_verdict_enumerates_alternative_probe_methodologies`) —
    and both are registered inside `preflight_all()`'s `if check_codebase:` block.
    `--no-codebase` is THIS hook's default mode, so neither has ever executed at
    commit time. Measured 2026-08-03: within their own filename+date scope they
    can refuse 9 of 221 negative-bearing docs, because their trigger list is 15
    exact phrases — a memo saying `REFUTED at the formulation level` or a bare
    `FALSIFIED` does not trip either one.

    This step is deliberately narrower in what it demands (a declared scope, and
    for FAMILY/PARADIGM the evidence that breadth costs) and much broader in what
    it reaches — including `.md`, which the hook did not previously collect at
    all, and which is where negatives are actually written.

    Fail-OPEN and LOUD on a broken guard, matching the siblings above: blocking
    every commit because the guard itself broke is worse than the bug it guards.
    """
    if os.environ.get("NEGATIVE_VERDICT_SCAN_ENABLED", "1") == "0":
        print(
            "[preflight-hook] negative-verdict scan DISABLED by env — NOT a pass.",
            file=sys.stderr,
        )
        return 0
    try:
        from tac.negative_verdict_gate import scan_staged
    except Exception as exc:  # pragma: no cover - guard-broken path
        print(
            f"[preflight-hook] negative-verdict scan UNAVAILABLE (failing OPEN): {exc}",
            file=sys.stderr,
        )
        return 0

    if not staged_docs:
        return 0
    try:
        violations = scan_staged(
            repo_root=REPO_ROOT, strict=False, verbose=False, files=staged_docs
        )
    except Exception as exc:  # pragma: no cover - guard-internal bug path
        print(
            f"[preflight-hook] negative-verdict scan CRASHED (failing OPEN): "
            f"{exc.__class__.__name__}: {exc} — 0 files examined. This is NOT a pass.",
            file=sys.stderr,
        )
        return 0

    if violations:
        print(
            f"\n{RED}{BOLD}[preflight-hook] BLOCKED: new negative verdict with no "
            f"declared scope{RST}",
            file=sys.stderr,
        )
        print(
            "  The narrowest scope the evidence supports is the honest one "
            "(instance<formulation<family<paradigm).\n"
            "  An INSTANCE kill costs only the declaration; a FAMILY/PARADIGM kill "
            "costs more, on purpose.\n"
            "  Canonical record: tac.verdicts.emit_verdict(is_negative=True) — it "
            "already enforces the same ladder.",
            file=sys.stderr,
        )
        for v in violations:
            print(f"[preflight-hook]   {v}", file=sys.stderr)
        return 1
    print(
        f"{GREEN}[preflight-hook] negative-verdict scope: {len(staged_docs)} staged "
        f".md examined, no new unscoped negatives{RST}",
        file=sys.stderr,
    )
    return 0


def _staged_added_lines(files: list[str]) -> list[tuple[str, str]]:
    """Return staged added lines as ``(path, line)`` pairs for exact diff scope."""
    if not files:
        return []
    result = subprocess.run(
        ["git", "diff", "--cached", "--unified=0", "--", *files],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff --cached failed")

    added: list[tuple[str, str]] = []
    current_path: str | None = None
    for raw in result.stdout.splitlines():
        if raw.startswith("+++ b/"):
            current_path = raw.removeprefix("+++ b/")
            continue
        if raw.startswith("+++ /dev/null"):
            current_path = None
            continue
        if current_path and raw.startswith("+") and not raw.startswith("+++"):
            added.append((current_path, raw[1:]))
    return added


def run_followon_regrow_scan(staged_docs: list[str]) -> int:
    """`ddm_fo1` — warn when new cheap follow-ons enter docs without a typed closeout.

    This is deliberately warn-only: prose follow-ons are real work queues, but a
    hook cannot honestly prove the owner/fire-order/result join by parsing one
    sentence. The guard's job is to make the regrowth visible at the commit
    surface and print its denominator; the durable closeout lives in the fo1
    typed ledger / join receipts.
    """
    if os.environ.get("FOLLOWON_REGROW_SCAN_ENABLED", "1") == "0":
        print(
            "[preflight-hook] follow-on regrow scan DISABLED by env — NOT a pass.",
            file=sys.stderr,
        )
        return 0
    if not staged_docs:
        return 0
    try:
        from tac.followon_ledger import ACTION_RX, CHEAP_RX
    except Exception as exc:  # pragma: no cover - guard-broken path
        print(
            f"[preflight-hook] follow-on regrow scan UNAVAILABLE (failing OPEN): {exc}",
            file=sys.stderr,
        )
        return 0
    try:
        added_lines = _staged_added_lines(staged_docs)
    except Exception as exc:  # pragma: no cover - guard-internal bug path
        print(
            f"[preflight-hook] follow-on regrow scan CRASHED (failing OPEN): "
            f"{exc.__class__.__name__}: {exc} — 0 added lines examined. This is NOT a pass.",
            file=sys.stderr,
        )
        return 0

    matches = [
        (path, line)
        for path, line in added_lines
        if ACTION_RX.search(line) and CHEAP_RX.search(line)
    ]
    if matches:
        print(
            f"{YELLOW}[preflight-hook] follow-on regrow: {len(staged_docs)} staged .md, "
            f"{len(added_lines)} added lines, {len(matches)} new cheap follow-on lines. "
            f"Ensure FIRED/FOLDED/QUEUED-WITH-FIRE-ORDER plus owner before handoff.{RST}",
            file=sys.stderr,
        )
        for path, line in matches[:10]:
            print(
                f"[preflight-hook]   {path}: {line.strip()[:180]}",
                file=sys.stderr,
            )
        return 0
    print(
        f"{GREEN}[preflight-hook] follow-on regrow: {len(staged_docs)} staged .md, "
        f"{len(added_lines)} added lines, no new cheap follow-ons{RST}",
        file=sys.stderr,
    )
    return 0


_DRIVER_RC_EXIT0_WAIVER = "DRIVER_RC_EXIT0_OK:"
_DRIVER_RC_ECHO_RE = re.compile(r'(^|;)\s*echo\b.*\brc\s*[:=]')
_DRIVER_EXIT_ZERO_RE = re.compile(r'(^|;)\s*exit\s+0(\s|;|$)')
_DRIVER_PROPAGATING_EXIT_RE = re.compile(
    r'(^|;)\s*(exit|return)\s+'
    r'(?:"?\$\{?[A-Za-z_][A-Za-z0-9_]*\}?"?|\$\?)(\s|;|$)'
)


def _shell_driver_rc_receipt_warnings(files: list[str]) -> list[str]:
    """Warn on shell drivers that log rc values but fall through to success.

    This is intentionally conservative and warn-only. The high-cost failure is a
    detached watcher seeing ``rc=0`` after a driver script merely echoed a failed
    stage's rc. Same-line ``DRIVER_RC_EXIT0_OK:<reason>`` waives explicit
    sentinels and intentional fire-order stop exits.
    """
    warnings: list[str] = []
    for rel in files:
        path = REPO_ROOT / rel
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError as exc:
            warnings.append(f"{rel}: unreadable ({exc.__class__.__name__})")
            continue
        rc_echo_lines = [
            (idx, line)
            for idx, line in enumerate(lines, start=1)
            if _DRIVER_RC_ECHO_RE.search(line)
            and _DRIVER_RC_EXIT0_WAIVER not in line
        ]
        if not rc_echo_lines:
            continue
        propagating_exit = any(
            _DRIVER_PROPAGATING_EXIT_RE.search(line)
            and _DRIVER_RC_EXIT0_WAIVER not in line
            for line in lines
        )
        for idx, line in enumerate(lines, start=1):
            if _DRIVER_EXIT_ZERO_RE.search(line) and _DRIVER_RC_EXIT0_WAIVER not in line:
                warnings.append(
                    f"{rel}:{idx}: logs rc but has unconditional exit 0; add "
                    f"exit \"$rc\"/overall rc propagation or same-line "
                    f"{_DRIVER_RC_EXIT0_WAIVER}<reason>"
                )
        if not propagating_exit:
            first_idx, first_line = rc_echo_lines[0]
            warnings.append(
                f"{rel}:{first_idx}: logs rc but no variable exit/return propagates "
                f"that status; first rc line: {first_line.strip()[:140]}"
            )
    return warnings


def run_shell_driver_rc_receipt_scan(staged_shell: list[str]) -> int:
    """Warn-only guard for detached shell drivers that can emit false success receipts."""
    if os.environ.get("SHELL_DRIVER_RC_RECEIPT_SCAN_ENABLED", "1") == "0":
        print(
            "[preflight-hook] shell-driver rc receipt scan DISABLED by env — NOT a pass.",
            file=sys.stderr,
        )
        return 0
    if not staged_shell:
        return 0
    warnings = _shell_driver_rc_receipt_warnings(staged_shell)
    if warnings:
        print(
            f"{YELLOW}[preflight-hook] shell-driver rc receipts: "
            f"{len(staged_shell)} staged .sh examined, {len(warnings)} warning(s). "
            f"A detached .done receipt inherits the script exit code, not echoed stage rc.{RST}",
            file=sys.stderr,
        )
        for warning in warnings[:10]:
            print(f"[preflight-hook]   {warning}", file=sys.stderr)
        return 0
    print(
        f"{GREEN}[preflight-hook] shell-driver rc receipts: {len(staged_shell)} "
        f"staged .sh examined, no rc fallthrough patterns{RST}",
        file=sys.stderr,
    )
    return 0


def run_guarded_constant_frozen_literal_scan(staged: list[str]) -> int:
    """`ddm_gk1` landing 2 — refuse a NEWLY-ADDED frozen output of a live derivation.

    The class: a derivation exists in-repo, is documented, and is NOT CALLED —
    its output was frozen into a literal. The value is right TODAY, so nothing
    fails and no test goes red; if the data moved, nobody would find out. Such a
    constant passes every provenance check while running none. Canonical
    instance: `derive_margin_floor` (`src/tac/optimization/lane_guard.py:547`,
    documented "data-derived per run, never a bare constant") vs the frozen
    `margin_floor: float = 0.1` default it produced.

    Placement rationale, same as steps 1c/1d and for the same MEASURED reason:
    `--no-codebase` is this hook's default and examines 0 of 27 preflight gates
    (ddm_ss1), which is why two STRICT kill-verdict gates (Catalog #307/#308)
    have never fired at commit. A gate registered only in `preflight_all()` is
    decoration. Placed before `run_preflight()`, which early-returns on failure.

    Scope is the staged diff's ADDED LINES, which is what lets it be STRICT
    honestly. The whole-repo live count is 0 at landing (the one pre-existing
    site was migrated byte-identically in the same commit), so added-lines scope
    costs no coverage here and prevents an unrelated edit to a file from dragging
    an old site in.

    DECLARATION-DRIVEN: it fires only on `<name> = <value>` where BOTH the
    identifier and the exact value are declared by a GuardedConstant in
    `tac.witness_dsl.guarded_constant_registry`. ddm_gd5 (#864) BUILT and
    REFUTED the auto-derived version (import-reachability fires on 1229 of 3251
    modules), so this one does not infer its own targets — and the honest cost is
    that a constant nobody has declared is invisible to it. The denominator is
    printed for exactly that reason.

    Fail-OPEN and LOUD on a broken guard, matching the siblings above.
    """
    if os.environ.get("GUARDED_CONSTANT_SCAN_ENABLED", "1") == "0":
        print(
            "[preflight-hook] guarded-constant scan DISABLED by env — NOT a pass.",
            file=sys.stderr,
        )
        return 0
    try:
        from tac.run_constant_gates import scan_staged_for_guarded_constant_frozen_literals
    except Exception as exc:  # pragma: no cover - guard-broken path
        print(
            f"[preflight-hook] guarded-constant scan UNAVAILABLE (failing OPEN): {exc}",
            file=sys.stderr,
        )
        return 0
    if not staged:
        return 0
    try:
        violations, unexamined = scan_staged_for_guarded_constant_frozen_literals(
            repo_root=REPO_ROOT, files=staged
        )
    except Exception as exc:  # pragma: no cover - guard-internal bug path
        print(
            f"[preflight-hook] guarded-constant scan CRASHED (failing OPEN): "
            f"{exc.__class__.__name__}: {exc} — 0 files examined. This is NOT a pass.",
            file=sys.stderr,
        )
        return 0

    for note in unexamined:
        print(f"{YELLOW}[preflight-hook]   UNEXAMINED: {note}{RST}", file=sys.stderr)
    if violations:
        print(
            f"\n{RED}{BOLD}[preflight-hook] BLOCKED: new frozen output of a live "
            f"derivation{RST}",
            file=sys.stderr,
        )
        print(
            "  A derivation that exists and never runs cannot adapt, and nothing "
            "goes red when\n  the data moves — it presents as a provenanced constant "
            "while running no provenance.",
            file=sys.stderr,
        )
        for v in violations:
            print(f"[preflight-hook]   {v.describe()}", file=sys.stderr)
        return 1
    print(
        f"{GREEN}[preflight-hook] guarded-constant: {len(staged)} staged .py examined "
        f"({len(unexamined)} unexamined), no new frozen derivation outputs{RST}",
        file=sys.stderr,
    )
    return 0


_MODAL_CLOSURE_SCHEMA = "modal_endpoint_closure_manifest.v1"
_MODAL_CLOSURE_WAIVER = re.compile(
    r"\.spawn\([^\n]*#\s*MODAL_CLOSURE_MANIFEST_FREE:\s*(?P<reason>.+)$"
)
_PLACEHOLDER_WAIVER_WORDS = {"later", "none", "placeholder", "tbd", "test", "todo"}


def scan_modal_closure_manifest_source(path: str, source: str) -> str | None:
    """Return a coverage warning for one new Modal dispatcher, else ``None``."""

    if not re.fullmatch(r"experiments/[^/]*_modal_[^/]*\.py", path) or ".spawn(" not in source:
        return None
    if _MODAL_CLOSURE_SCHEMA in source and "closure_manifest" in source:
        return None

    spawn_lines = [line for line in source.splitlines() if ".spawn(" in line]
    waiver_reasons: list[str] = []
    for line in spawn_lines:
        match = _MODAL_CLOSURE_WAIVER.search(line)
        if not match:
            return f"{path}: .spawn() has no closure manifest and no same-line waiver"
        reason = match.group("reason").strip()
        words = re.findall(r"[A-Za-z0-9]+", reason.lower())
        if (
            len(reason) < 20
            or len(words) < 3
            or any(word in _PLACEHOLDER_WAIVER_WORDS for word in words)
        ):
            return f"{path}: placeholder MODAL_CLOSURE_MANIFEST_FREE rationale rejected"
        waiver_reasons.append(reason)
    if not waiver_reasons:
        return f"{path}: Modal spawn coverage could not be classified"
    return None


def run_modal_closure_manifest_scan(staged_added: list[str]) -> int:
    """AC1 warn-only coverage for newly-added Modal dispatchers.

    Existing launchers are intentionally outside this first landing. A warning
    cannot prove that an endpoint returns payloads, but it prevents silent
    expansion of the unclosable-dispatcher denominator.
    """

    candidates = [
        path
        for path in staged_added
        if re.fullmatch(r"experiments/[^/]*_modal_[^/]*\.py", path)
    ]
    warnings: list[str] = []
    for path in candidates:
        try:
            source = (REPO_ROOT / path).read_text(encoding="utf-8")
            warning = scan_modal_closure_manifest_source(path, source)
        except OSError as exc:
            warning = f"{path}: unreadable ({exc})"
        if warning:
            warnings.append(warning)
    if warnings:
        print(
            f"{YELLOW}[preflight-hook] Modal endpoint closure coverage WARNING: "
            f"{len(candidates)} new dispatcher(s), {len(warnings)} uncovered{RST}",
            file=sys.stderr,
        )
        for warning in warnings:
            print(f"[preflight-hook]   {warning}", file=sys.stderr)
    elif candidates:
        print(
            f"{GREEN}[preflight-hook] Modal endpoint closure coverage: "
            f"{len(candidates)} new dispatcher(s), 0 uncovered{RST}",
            file=sys.stderr,
        )
    return 0


def run_staged_secrets_scan() -> int:
    """Step 1i: BLOCKING secrets scan over the staged diff (operator 2026-08-20).

    The .playwright-mcp incident: a bare `git add` in a 236-file commit swept in
    browser-console logs carrying credential-shaped strings, publicly, for four
    months. History cannot be selectively cleaned (each sha commits to its full
    tree + parent), so the only durable cure is refusing the commit that carries
    the secret. `gitleaks protect --staged` scans exactly the bytes about to be
    committed — the typing-moment surface, not a post-hoc audit.

    Ruleset: configs/gitleaks_pact.toml extends the default rules with the
    campaign's own token shapes (Modal/HF/Anthropic/Tailscale/wandb) — the
    default ruleset alone missed every one of those (the #1086 vacuous-control
    lesson: a control that cannot fail proves nothing).

    FAIL-CLOSED on every path (rv17 SEC-F1, 2026-08-20): findings REFUSE with
    redacted output; absent binary, timeout, and tool error ALSO REFUSE — a
    guard advertised BLOCKING that passes when its tool is missing commits the
    very skip-as-green vacuity class (#875/#1050) it exists to prevent. The
    ONE escape is TAC_SECRETS_WAIVE=1: a deliberate, loud, operator-reviewed
    exception — never a silent default.
    """

    if os.environ.get("TAC_SECRETS_WAIVE") == "1":
        print(
            f"{YELLOW}[preflight-hook] staged secrets scan WAIVED via "
            f"TAC_SECRETS_WAIVE=1 — deliberate exception, not a pass{RST}",
            file=sys.stderr,
        )
        return 0
    try:
        staged_count = len(
            subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True, text=True, check=True, cwd=REPO_ROOT,
            ).stdout.splitlines()
        )
    except (subprocess.CalledProcessError, OSError):
        staged_count = -1
    if staged_count == 0:
        return 0
    config = REPO_ROOT / "configs" / "gitleaks_pact.toml"
    cmd = ["gitleaks", "protect", "--staged", "--no-banner", "--redact"]
    if config.is_file():
        cmd += ["--config", str(config)]
    else:
        print(
            f"{YELLOW}[preflight-hook] staged secrets scan: campaign ruleset "
            f"missing at {config} — falling back to gitleaks defaults (the "
            f"Modal/HF/Anthropic/Tailscale shapes are NOT covered){RST}",
            file=sys.stderr,
        )
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=120
        )
    except FileNotFoundError:
        print(
            f"{RED}[preflight-hook] staged secrets scan REFUSED: gitleaks "
            f"binary absent ({staged_count} staged file(s) UNSCANNED). Install "
            f"via `brew install gitleaks`, or deliberate exception: "
            f"TAC_SECRETS_WAIVE=1{RST}",
            file=sys.stderr,
        )
        return 1
    except subprocess.TimeoutExpired:
        print(
            f"{RED}[preflight-hook] staged secrets scan REFUSED: gitleaks "
            f"TIMED OUT at 120s ({staged_count} staged file(s) UNSCANNED). "
            f"Deliberate exception: TAC_SECRETS_WAIVE=1{RST}",
            file=sys.stderr,
        )
        return 1
    if result.returncode == 0:
        print(
            f"{GREEN}[preflight-hook] staged secrets scan: "
            f"{staged_count} staged file(s), 0 findings{RST}",
            file=sys.stderr,
        )
        return 0
    if result.returncode == 1:
        print(
            f"{RED}[preflight-hook] SECRETS IN STAGED DIFF — commit REFUSED "
            f"({staged_count} staged file(s) scanned). Redacted findings:{RST}",
            file=sys.stderr,
        )
        tail = (result.stdout + result.stderr).strip().splitlines()[-20:]
        for line in tail:
            print(f"[preflight-hook]   {line}", file=sys.stderr)
        print(
            f"{RED}[preflight-hook] Remove the secret (and rotate it if real). "
            f"Deliberate exception: TAC_SECRETS_WAIVE=1{RST}",
            file=sys.stderr,
        )
        return 1
    print(
        f"{RED}[preflight-hook] staged secrets scan REFUSED: gitleaks error "
        f"rc={result.returncode} ({staged_count} staged file(s) UNSCANNED — "
        f"if `protect` was dropped by a gitleaks upgrade, migrate this step "
        f"to the supported staged-scan command). Deliberate exception: "
        f"TAC_SECRETS_WAIVE=1: {(result.stderr or '').strip()[:200]}{RST}",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    staged = _staged_py_files()
    staged_added = _staged_added_py_files()
    staged_docs = _staged_doc_files()
    staged_shell = _staged_shell_files()

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

    # Step 1c: `ddm_ss1` subset-selection guard. Placed here for the same reason as
    # 1b — before run_preflight(), which early-returns on failure — so a preflight
    # failure cannot skip it. ~AST parse of the staged files only; ~ms.
    rc = run_subset_selection_scan(staged)
    if rc != 0:
        return rc

    # Step 1d: `ddm_ks1` negative-verdict scope guard. Same placement rationale as
    # 1b/1c — before run_preflight(), which early-returns on failure. This one
    # additionally covers .md, the surface no commit-time check read before.
    rc = run_negative_verdict_scan(staged_docs)
    if rc != 0:
        return rc

    # Step 1e: `ddm_fo1` follow-on regrow guard. Warn-only by design; it makes
    # newly-added cheap follow-ons visible before they become ownerless backlog.
    rc = run_followon_regrow_scan(staged_docs)
    if rc != 0:
        return rc

    # Step 1f: rr8-F2 shell-driver receipt guard. Warn-only: a hook cannot infer every
    # intentional stop marker, but it can expose the false-success receipt class.
    rc = run_shell_driver_rc_receipt_scan(staged_shell)
    if rc != 0:
        return rc

    # Step 1g: `ddm_gk1` guarded-constant frozen-derivation guard. Same placement
    # rationale as 1b/1c/1d/1e/1f — before run_preflight(), which early-returns on failure.
    rc = run_guarded_constant_frozen_literal_scan(staged)
    if rc != 0:
        return rc

    # Step 1h: AC1 additive Modal endpoint-closure coverage. Warn-only in this
    # first landing; only newly-added dispatcher files are in its denominator.
    rc = run_modal_closure_manifest_scan(staged_added)
    if rc != 0:
        return rc

    # Step 1i: BLOCKING staged-diff secrets scan. Same placement rationale as
    # 1b-1g — before run_preflight(), which early-returns on failure. This is
    # the .playwright-mcp never-again guard: the credential-shaped strings that
    # sat in public history for four months entered via one bare `git add`;
    # this step reads exactly the staged bytes at the typing moment.
    rc = run_staged_secrets_scan()
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
