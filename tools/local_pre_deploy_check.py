#!/usr/bin/env python3
"""Local pre-deploy harness — catches bug classes BEFORE paid Modal/Vast.ai dispatch.

Per operator 2026-05-15: *"shouldn't you run python compile locally before deploy
to find those bugs"*. Empirical anchor: Z3 v2 smoke `fc-01KRNHEGC9ZE48Y68GGJHP7FXN`
($2 wasted) + Z4 smoke `fc-01KRNHE942JSV7VRGXGR1FJGHQ` ($2 wasted) both crashed
on bugs that local 30s pre-flight would have caught:
- Z3 v2: learned-packet archive member/runtime mismatch before auth eval
- Z4: trainer didn't reach auth_eval stage (pre-auth-eval crash class)

This harness gates EVERY operator-authorize dispatch. Bug-class extinction.

Usage:
    .venv/bin/python tools/local_pre_deploy_check.py \\
        --trainer experiments/train_substrate_z3_balle_hyperprior_bolton.py \\
        --recipe substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch \\
        [--strict]   # exit 1 on any FAIL (default: warn-only)

Checks:
1. **PY-COMPILE**: `python -m py_compile <trainer>` — catches syntax errors
2. **ARCHIVE-GRAMMAR**: greps `_build_archive_zip` for `filename="0.bin"` per
   the repo's monolithic learned-packet convention. The contest itself does
   not require a specific member name; `evaluate.sh` just unzips `archive.zip`.
   For this repo's single-packet trainers, `0.bin` pairs with `inflate.sh`'s
   `${base}.bin` fallback for the public file-list base `0`.
3. **AUTH-EVAL-REACHABILITY**: greps trainer for `auth_eval_renderer` invocation
   OR `gate_auth_eval_call` (canonical helper per Catalog #226). Refuses if
   trainer has neither (per CLAUDE.md "Auth eval EVERYWHERE" non-negotiable).
4. **CANONICAL-INFLATE-DEVICE**: per Catalog #205, refuses inline
   `torch.device("cuda" if ...)` in submission inflate.py.
5. **DETERMINISTIC-ZIP**: per Catalog #19, refuses bare `ZipFile.write` (must
   use `ZipInfo + writestr` with fixed timestamp).

Exit codes:
- 0 = all checks pass; safe to dispatch
- 1 = at least one check FAILED; do NOT dispatch (refund the $2 smoke)
- 2 = bad invocation (missing trainer / recipe)

Cost of running: ~5-15s wall-clock; $0. Cost of NOT running: $2-15 per failed
smoke + 30-60min wall-clock waiting for Modal to crash.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import py_compile
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_ARCHIVE_MEMBER = "0.bin"
ARCHIVE_BUILD_PATTERN = re.compile(
    r'zipfile\.ZipInfo\s*\(\s*filename\s*=\s*["\']([^"\']+)["\']'
)
ARCHIVE_MEMBER_WAIVER = re.compile(r"#\s*ARCHIVE_MEMBER_OK:\s*\S")
X_MEMBER_RUNTIME_EVIDENCE_PATTERNS = (
    re.compile(r"\$\{?DATA_DIR\}?/x"),
    re.compile(r"\barchive_dir\s*/\s*['\"]x['\"]"),
    re.compile(r"\bPath\s*\([^)]*archive_dir[^)]*\)\s*/\s*['\"]x['\"]"),
    re.compile(r"\bcandidates\s*=\s*\[[^\]]*['\"]x['\"]"),
    re.compile(r"\barchive_dir/x\b"),
    re.compile(r"\barchive_grammar\b[^\n]*['\"][^'\"]*\bx\b[^'\"]*['\"]", re.I),
    re.compile(r"\bsingle\s+(?:zip\s+)?member\s+``?x``?", re.I),
)

AUTH_EVAL_CALL_NAMES = {
    "_canon_gate_auth_eval_call",
    "auth_eval_renderer",
    "gate_auth_eval_call",
}
AUTH_EVAL_ENTRYPOINTS = ("_full_main", "main")
SUBPROCESS_AUTH_EVAL_CALLS = {
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
    "os.system",
}


# Catalog #270 scope clarification (2026-05-17 per
# ``lane_catalog_270_scope_fix_tool_vs_substrate_dispatch_20260517``).
# Tool dispatches (``tools/*.py`` or ``dispatch_kind: tool``) are categorically
# NOT subject to substrate-trainer-only harness checks (auth_eval_reachability
# / archive_grammar / full_main_implemented / canonical_inflate_device /
# deterministic_zip). Sister of ``tac.deploy.dispatch_protocol.is_tool_dispatch``
# and ``tools/canonical_dispatch_optimization_protocol._is_tool_dispatch``.
def _is_tool_dispatch_for_harness(
    trainer: Path, recipe: str | None
) -> bool:
    """Detect tool dispatch for harness checks.

    Recipe lookup mirrors ``check_recipe_status_consistent_with_trainer_state``:
    accepts recipe NAME (without ``.yaml``) and resolves under
    ``.omx/operator_authorize_recipes/``. If the recipe declares
    ``dispatch_kind: tool``, return True. Otherwise fall through to the
    implicit trainer-path check: ``tools/*.py`` (not
    ``experiments/train_substrate_*.py``) → True.
    """
    if recipe is not None:
        candidate_recipe_paths = [
            REPO_ROOT / ".omx" / "operator_authorize_recipes" / f"{recipe}.yaml",
            REPO_ROOT / ".omx" / "operator_authorize_recipes" / f"{recipe}",
        ]
        recipe_path = next(
            (p for p in candidate_recipe_paths if p.is_file()), None
        )
        if recipe_path is not None:
            try:
                recipe_text = recipe_path.read_text(encoding="utf-8")
            except OSError:
                recipe_text = ""
            m = re.search(
                r"^\s*dispatch_kind\s*:\s*['\"]?(\w+)", recipe_text, re.M
            )
            if m:
                kind = m.group(1).strip().lower()
                if kind == "tool":
                    return True
                if kind == "hf_jobs_research_surrogate":
                    return bool(
                        re.search(
                            r"^\s*platform\s*:\s*['\"]?hf_jobs\b",
                            recipe_text,
                            re.M,
                        )
                    )
                if kind in {"substrate", "local_research_signal"}:
                    return False
    try:
        rel = trainer.resolve().relative_to(REPO_ROOT.resolve())
    except (OSError, ValueError):
        return False
    posix = rel.as_posix()
    if posix.startswith("experiments/train_substrate_") and posix.endswith(".py"):
        return False
    return posix.startswith("tools/") and posix.endswith(".py")


# Sister 2026-05-17 (lane_one_arg_local_mps_vs_modal_dispatch_switch_20260517):
# Local research-signal dispatches (``platform: local_mps`` / ``local_cpu`` or
# ``dispatch_kind: local_research_signal``) get the SAME skip-set as tool
# dispatches per the Catalog #270 precedent. They run on the operator's
# machine, route through canonical mps_research_signal /
# macos_cpu_advisory_signal manifests, and forbid contest auth_eval emission
# entirely.
_LOCAL_RESEARCH_SIGNAL_PLATFORMS = frozenset({"local_mps", "local_cpu"})


def _recipe_text_for_harness(recipe: str | None) -> str:
    """Load an operator-authorize recipe body for harness-only classification."""

    if recipe is None:
        return ""
    candidate_recipe_paths = [
        REPO_ROOT / ".omx" / "operator_authorize_recipes" / f"{recipe}.yaml",
        REPO_ROOT / ".omx" / "operator_authorize_recipes" / f"{recipe}",
    ]
    recipe_path = next((p for p in candidate_recipe_paths if p.is_file()), None)
    if recipe_path is None:
        return ""
    try:
        return recipe_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _is_hf_jobs_research_surrogate_dispatch_for_harness(recipe: str | None) -> bool:
    """Return True for the explicit HF Jobs advisory-surrogate dispatch kind."""

    recipe_text = _recipe_text_for_harness(recipe)
    if not recipe_text:
        return False
    return bool(
        re.search(
            r"^\s*dispatch_kind\s*:\s*['\"]?hf_jobs_research_surrogate\b",
            recipe_text,
            re.M,
        )
        and re.search(r"^\s*platform\s*:\s*['\"]?hf_jobs\b", recipe_text, re.M)
    )


def _is_local_research_signal_dispatch_for_harness(
    trainer: Path, recipe: str | None
) -> bool:
    """Detect local research-signal dispatch for harness checks.

    Recipe lookup mirrors :func:`_is_tool_dispatch_for_harness`. Returns True
    when the recipe declares ``dispatch_kind: local_research_signal`` OR
    ``platform: local_mps`` / ``platform: local_cpu``.
    """
    if recipe is None:
        return False
    recipe_text = _recipe_text_for_harness(recipe)
    if not recipe_text:
        return False
    m_kind = re.search(
        r"^\s*dispatch_kind\s*:\s*['\"]?(\w+)", recipe_text, re.M
    )
    if m_kind:
        value = m_kind.group(1).strip().lower()
        if value == "local_research_signal":
            return True
        if value in {"substrate", "tool"}:
            return False
    m_platform = re.search(
        r"^\s*platform\s*:\s*['\"]?([\w_]+)", recipe_text, re.M
    )
    if m_platform:
        platform = m_platform.group(1).strip().lower()
        if platform in _LOCAL_RESEARCH_SIGNAL_PLATFORMS:
            return True
    return False


def _import_trainer_module(trainer: Path):
    digest = hashlib.sha256(str(trainer.resolve()).encode("utf-8")).hexdigest()[:12]
    module_name = f"_local_pre_deploy_{trainer.stem}_{digest}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, trainer)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {trainer}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
        raise
    return module


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _qualified_call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        prefix = _qualified_call_name(func.value)
        return f"{prefix}.{func.attr}" if prefix else func.attr
    return ""


def _node_contains_contest_auth_eval_literal(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and "contest_auth_eval.py" in child.value
        ):
            return True
    return False


def _iter_calls_in_function(fn: ast.FunctionDef | ast.AsyncFunctionDef):
    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.calls: list[ast.Call] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is fn:
                self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node is fn:
                self.generic_visit(node)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_Call(self, node: ast.Call) -> None:
            self.calls.append(node)
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(fn)
    return visitor.calls


def _auth_eval_invocation_reason(call: ast.Call) -> str | None:
    call_name = _call_name(call.func)
    if call_name in AUTH_EVAL_CALL_NAMES:
        return "canonical helper"
    qualified = _qualified_call_name(call.func)
    if (
        qualified in SUBPROCESS_AUTH_EVAL_CALLS
        and _node_contains_contest_auth_eval_literal(call)
    ):
        return "contest_auth_eval subprocess"
    return None


def _literal_zipinfo_members(text: str) -> list[tuple[str, int]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    members: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) != "ZipInfo":
            continue
        member: str | None = None
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            member = node.args[0].value
        for keyword in node.keywords:
            if (
                keyword.arg == "filename"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                member = keyword.value.value
        if member is not None:
            members.append((member, getattr(node, "lineno", 0)))
    return members


def _archive_member_line_has_waiver(text: str, line_no: int | None) -> bool:
    if line_no is not None and line_no > 0:
        lines = text.splitlines()
        if line_no <= len(lines) and ARCHIVE_MEMBER_WAIVER.search(lines[line_no - 1]):
            return True
    return False


def _archive_member_has_any_waiver(text: str) -> bool:
    return any(ARCHIVE_MEMBER_WAIVER.search(line) for line in text.splitlines())


def _has_x_member_runtime_evidence(text: str) -> bool:
    return any(pattern.search(text) for pattern in X_MEMBER_RUNTIME_EVIDENCE_PATTERNS)


def _archive_member_supported(
    member: str,
    text: str,
    *,
    line_no: int | None = None,
    allow_file_waiver: bool = False,
) -> bool:
    if member == CANONICAL_ARCHIVE_MEMBER:
        return True
    if _archive_member_line_has_waiver(text, line_no):
        return True
    if allow_file_waiver and _archive_member_has_any_waiver(text):
        return True
    return member == "x" and _has_x_member_runtime_evidence(text)


def _unsupported_archive_member_message(member: str) -> str:
    if member == "x":
        return (
            "member 'x' requires explicit runtime evidence that inflate.sh/"
            "inflate.py consumes ${DATA_DIR}/x (or archive_dir / 'x'), "
            "or same-line # ARCHIVE_MEMBER_OK:<rationale>"
        )
    return (
        f"member {member!r} is not the repo learned-packet default "
        f"{CANONICAL_ARCHIVE_MEMBER!r}; add runtime evidence or "
        "same-line # ARCHIVE_MEMBER_OK:<rationale> if intentional"
    )


def check_py_compile(trainer: Path) -> tuple[bool, str]:
    """Return (passed, message)."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pyc", delete=True) as tmp:
            py_compile.compile(str(trainer), cfile=tmp.name, doraise=True)
        return True, f"PASS: {trainer.name} compiles cleanly"
    except py_compile.PyCompileError as e:
        return False, f"FAIL: {trainer.name} has syntax errors:\n{e.msg}"


def check_trainer_importable(trainer: Path) -> tuple[bool, str]:
    """Import the trainer module so missing exports fail before Modal spend."""
    try:
        _import_trainer_module(trainer)
        return True, f"PASS: {trainer.name} imports cleanly"
    except Exception as exc:
        return False, f"FAIL: {trainer.name} import failed: {exc!r}"


def check_archive_grammar(trainer: Path) -> tuple[bool, str]:
    """Refuse non-standard learned-packet member names without a waiver."""
    text = trainer.read_text()
    violations: list[str] = []
    literal_members = _literal_zipinfo_members(text)
    if not literal_members:
        for match in ARCHIVE_BUILD_PATTERN.finditer(text):
            literal_members.append((match.group(1), text.count("\n", 0, match.start()) + 1))
    for member, line_no in literal_members:
        if _archive_member_supported(member, text, line_no=line_no):
            continue
        violations.append(
            f"  {trainer.name}:{line_no}: filename={member!r}; "
            f"{_unsupported_archive_member_message(member)}"
        )
    if violations:
        msg = (
            "FAIL: learned-packet archive ZIP member lacks matching runtime evidence; "
            "violations:\n" + "\n".join(violations)
        )
        msg += (
            "\n  Fix: change _build_archive_zip to filename=\"0.bin\""
            "\n  OR prove the runtime consumes the intentional member (for example ${DATA_DIR}/x)"
            "\n  OR add same-line waiver only when inflate.sh consumes that member: "
            "# ARCHIVE_MEMBER_OK:<rationale>"
        )
        return False, msg
    try:
        module = _import_trainer_module(trainer)
        build_archive_zip = getattr(module, "_build_archive_zip", None)
        if build_archive_zip is not None:
            import zipfile

            signature = inspect.signature(build_archive_zip)
            if "bin_bytes" not in signature.parameters:
                return (
                    True,
                    f"PASS: {trainer.name} archive source-text grammar passed "
                    "dynamic check skipped (_build_archive_zip has nonstandard "
                    "signature)",
                )
            parameters = list(signature.parameters.values())
            provided_required = {parameters[0].name, "bin_bytes"} if parameters else {"bin_bytes"}
            missing_required = [
                param.name
                for param in parameters
                if param.default is inspect._empty
                and param.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
                and param.name not in provided_required
            ]
            if missing_required:
                return (
                    True,
                    f"PASS: {trainer.name} archive source-text grammar passed "
                    "dynamic check skipped (_build_archive_zip requires "
                    f"additional arguments {missing_required!r})",
                )

            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_path = Path(tmp_dir) / "archive.zip"
                build_archive_zip(zip_path, bin_bytes=b"local-predeploy")
                with zipfile.ZipFile(zip_path) as zf:
                    names = zf.namelist()
                unsupported = [
                    name
                    for name in names
                    if not _archive_member_supported(
                        name, text, allow_file_waiver=True
                    )
                ]
                if unsupported:
                    return (
                        False,
                        f"FAIL: _build_archive_zip emitted unsupported members "
                        f"{unsupported!r} from {names!r}; "
                        + "; ".join(
                            _unsupported_archive_member_message(name)
                            for name in unsupported
                        ),
                    )
    except Exception as exc:
        return False, f"FAIL: archive grammar dynamic check failed: {exc!r}"
    return True, f"PASS: {trainer.name} archive ZIP member follows repo standard/evidence"


def check_auth_eval_reachability(trainer: Path) -> tuple[bool, str]:
    """Refuse trainer with no auth_eval invocation (Z4 bug class)."""
    text = trainer.read_text()
    try:
        tree = ast.parse(text, filename=str(trainer))
    except SyntaxError as exc:
        return False, f"FAIL: {trainer.name} cannot be parsed for auth_eval reachability: {exc}"
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    entrypoints = [name for name in AUTH_EVAL_ENTRYPOINTS if name in functions]
    if not entrypoints:
        return (
            False,
            f"FAIL: {trainer.name} has no auth_eval invocation; "
            "no reachable auth_eval invocation; "
            "no main/_full_main entrypoint was found for the local predeploy call graph.",
        )

    visited: set[str] = set()
    stack = list(entrypoints)
    while stack:
        function_name = stack.pop()
        if function_name in visited:
            continue
        visited.add(function_name)
        function = functions[function_name]
        for call in _iter_calls_in_function(function):
            reason = _auth_eval_invocation_reason(call)
            if reason is not None:
                return (
                    True,
                    f"PASS: {trainer.name} invokes auth_eval via reachable auth_eval invocation "
                    f"from {function_name} ({reason})",
                )
            callee = _call_name(call.func)
            if callee in functions and callee not in visited:
                stack.append(callee)
    return (
        False,
        f"FAIL: {trainer.name} has no reachable auth_eval invocation from "
        f"entrypoints {entrypoints!r}; "
        f"trainer will reach Modal but never produce auth_eval_*.json. "
        f"Per CLAUDE.md 'Auth eval EVERYWHERE' non-negotiable, every training script must end with CUDA auth eval. "
        f"Fix: import + call gate_auth_eval_call from tac.substrates._shared.smoke_auth_eval_gate per Catalog #226.",
    )


def check_canonical_inflate_device(trainer: Path) -> tuple[bool, str]:
    """Per Catalog #205: no inline torch.device("cuda" if ...) in submission inflate.py.

    The trainer typically writes inflate.py via _write_runtime; check that
    template uses canonical select_inflate_device.
    """
    text = trainer.read_text()
    bad_pattern = re.compile(
        r'torch\.device\s*\(\s*["\']cuda["\']\s+if\s+torch\.cuda\.is_available'
    )
    matches = list(bad_pattern.finditer(text))
    if not matches:
        return True, f"PASS: {trainer.name} no inline torch.device cuda-fallback"
    line_nos = [text.count("\n", 0, m.start()) + 1 for m in matches]
    return (
        False,
        f"FAIL: {trainer.name} contains inline torch.device cuda-if-available at lines {line_nos}; "
        f"per Catalog #205 use canonical select_inflate_device from tac.substrates._shared.inflate_runtime",
    )


def check_deterministic_zip(trainer: Path) -> tuple[bool, str]:
    """Per Catalog #19: refuse bare ZipFile.write (must use ZipInfo + writestr)."""
    text = trainer.read_text()
    bad_pattern = re.compile(r'\bZipFile\s*\([^)]*\)\.write\s*\(')
    matches = list(bad_pattern.finditer(text))
    if not matches:
        return True, f"PASS: {trainer.name} uses deterministic-ZIP pattern"
    line_nos = [text.count("\n", 0, m.start()) + 1 for m in matches]
    return (
        False,
        f"FAIL: {trainer.name} uses bare ZipFile.write at lines {line_nos}; "
        f"per Catalog #19 use ZipInfo(filename, date_time=fixed) + writestr",
    )


def check_full_main_implemented(trainer: Path) -> tuple[bool, str]:
    """Refuse trainer with NotImplementedError stub `_full_main`.

    Empirical anchor 2026-05-15: Z4 + Z5 + s2sbs all have ``_full_main``
    that raises ``NotImplementedError`` — Modal dispatch reaches the trainer
    but immediately crashes pre-auth-eval. Per CLAUDE.md "Substrate scaffolds
    MUST be COMPLETE or RESEARCH-ONLY" + Catalog #220: such trainers must be
    tagged ``research_only=true`` in their recipe AND NOT dispatchable
    (for example ``dispatch_enabled=false`` or explicit blockers).
    """
    text = trainer.read_text()
    # Match `def _full_main(...)` followed by docstring then `raise NotImplementedError`
    # within the function body (first ~30 lines after def).
    full_main_pat = re.compile(
        r"def\s+_full_main\s*\([^)]*\)[^:]*:\s*\n((?:[ \t]+.*\n){0,40})",
        re.MULTILINE,
    )
    match = full_main_pat.search(text)
    if not match:
        return True, f"PASS: {trainer.name} has no _full_main (smoke-only trainer; OK)"
    body = match.group(1)
    if re.search(r"\braise\s+NotImplementedError\b", body):
        line_no = text.count("\n", 0, match.start()) + 1
        return (
            False,
            f"FAIL: {trainer.name}:{line_no}: _full_main raises NotImplementedError "
            f"(stub trainer; not dispatchable). Per CLAUDE.md 'Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY' + Catalog #220, "
            f"tag the recipe with research_only=true plus a real non-dispatchable flag "
            f"(for example dispatch_enabled=false), OR implement _full_main before dispatch.",
        )
    return True, f"PASS: {trainer.name} _full_main appears implemented"


def _recipe_list_has_entries(recipe_text: str, key: str) -> bool:
    """Return true when a simple YAML list key has at least one entry."""
    lines = recipe_text.splitlines()
    for idx, line in enumerate(lines):
        match = re.match(rf"^{re.escape(key)}:\s*(.*)$", line)
        if not match:
            continue
        tail = match.group(1).strip()
        if tail and tail not in {"[]", "null", "None"}:
            return True
        for child in lines[idx + 1 :]:
            if child and not child.startswith((" ", "\t", "-")):
                break
            if re.match(r"^\s*-\s+\S", child):
                return True
        return False
    return False


def _recipe_scalar_value(recipe_text: str, key: str) -> str:
    """Return a simple YAML scalar value for ``key`` from a recipe body."""

    match = re.search(rf"^\s*{re.escape(key)}:\s*(.*?)\s*(?:#.*)?$", recipe_text, re.M)
    if not match:
        return ""
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


def _recipe_scalar_truthy(recipe_text: str, key: str) -> bool:
    return _recipe_scalar_value(recipe_text, key).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def check_recipe_status_consistent_with_trainer_state(
    trainer: Path, recipe: str | None
) -> tuple[bool, str]:
    """7th check (operator-routed 2026-05-15 Phase 7).

    Surfaces the Z3 v2 / Z4 / Z5 bug class at the 30s harness BEFORE Modal
    dispatch fires. Refuses recipe-vs-trainer-state divergence:

    - Trainer's _full_main raises NotImplementedError BUT recipe lacks a
      non-dispatchable flag → Modal will reach trainer and crash pre-auth-eval
      ($2-15 per smoke).
    - Trainer's _full_main is implemented BUT recipe still declares a real
      dispatch refusal (smoke_only direct-dispatch path, dispatch_enabled=false,
      blockers) → this harness must not print "Safe to dispatch";
      ``operator_authorize.py`` will refuse the same recipe before
      claim/provider setup.

    Mirrors the canonical Catalog #240 STRICT preflight gate per CLAUDE.md
    "Bugs must be permanently fixed AND self-protected against": one
    structural fix at the harness surface + one STRICT preflight gate at the
    code surface.
    """
    if recipe is None:
        return True, "PASS: no recipe specified; recipe-vs-trainer-state check skipped (use --recipe to enable)"
    # Locate recipe YAML
    candidate_recipe_paths = [
        REPO_ROOT / ".omx" / "operator_authorize_recipes" / f"{recipe}.yaml",
        REPO_ROOT / ".omx" / "operator_authorize_recipes" / f"{recipe}",
    ]
    recipe_path = next((p for p in candidate_recipe_paths if p.is_file()), None)
    if recipe_path is None:
        return (
            False,
            f"FAIL: recipe '{recipe}' not found at .omx/operator_authorize_recipes/{recipe}.yaml; "
            "did you mistype the name? Use the substrate_<id>_modal_<gpu>_dispatch form (no .yaml).",
        )
    try:
        recipe_text = recipe_path.read_text()
    except OSError as exc:
        return False, f"FAIL: cannot read recipe {recipe_path}: {exc}"

    # Check trainer _full_main state.
    try:
        trainer_text = trainer.read_text()
    except OSError as exc:
        return False, f"FAIL: cannot read trainer {trainer}: {exc}"

    full_main_def = re.search(r"^def\s+_full_main\s*\([^)]*\)[^:]*:", trainer_text, re.M)
    if full_main_def is None:
        return True, f"PASS: {trainer.name} has no _full_main (smoke-only trainer)"
    body_start = full_main_def.end()
    body = trainer_text[body_start : body_start + 3000]
    raises_not_impl = bool(re.search(r"\braise\s+NotImplementedError\b", body))

    # Check recipe dispatchability flags. Keep this intentionally aligned with
    # ``tools.operator_authorize._recipe_dispatch_refusal`` plus
    # ``_smoke_only_direct_dispatch_refusal`` so the pre-deploy harness cannot
    # claim "Safe to dispatch" for a recipe the actuator refuses. Important:
    # ``research_only=true`` is false-authority metadata, not a dispatch
    # refusal in ``operator_authorize.py``; byte-closed research-smoke recipes
    # may be dispatchable while still forbidden from making score/promotion
    # claims.
    smoke_only = bool(re.search(r"^\s*smoke_only:\s*true\b", recipe_text, re.M))
    research_only = bool(
        re.search(r"^\s*research_only:\s*true\b", recipe_text, re.M)
    )
    dispatch_disabled = bool(
        re.search(r"^\s*dispatch_enabled:\s*false\b", recipe_text, re.M)
    )
    dispatch_blocked = _recipe_list_has_entries(recipe_text, "dispatch_blockers")
    pre_promotion_blocked = _recipe_list_has_entries(
        recipe_text, "pre_promotion_blockers"
    )
    non_dispatchable_reasons = [
        reason
        for reason, active in (
            ("smoke_only=true", smoke_only),
            ("dispatch_enabled=false", dispatch_disabled),
            ("dispatch_blockers", dispatch_blocked),
            ("pre_promotion_blockers", pre_promotion_blocked),
        )
        if active
    ]
    recipe_non_dispatchable = bool(non_dispatchable_reasons)

    if (
        trainer.name == "train_substrate_pretrained_driving_prior.py"
        and _recipe_scalar_truthy(recipe_text, "DPP_RUN_FULL")
        and (_recipe_scalar_value(recipe_text, "DPP_DATASET_NAME") or "comma2k19")
        == "comma2k19"
        and _recipe_scalar_truthy(recipe_text, "DPP_USE_STREAMER")
    ):
        return (
            False,
            "FAIL: DP1 full-run recipe selects DPP_USE_STREAMER=1 for "
            "comma2k19, but the current trainer has no recipe/CLI path for "
            "supplying a Comma2k19LocalStreamer dataset_sha256_manifest or "
            "explicit chunk_ids. Full training will fail with no chunk ids. "
            "Set DPP_USE_STREAMER=0 and provide DPP_CACHE_DIR for the "
            "canonical Comma2k19LocalCache path, or wire a real streamer "
            "manifest before dispatch.",
        )

    if raises_not_impl and not recipe_non_dispatchable:
        return (
            False,
            f"FAIL: recipe-vs-trainer-state divergence — "
            f"trainer {trainer.name} `_full_main` raises NotImplementedError "
            f"(Phase 2 council-gated) but recipe {recipe_path.name} lacks "
            f"a real non-dispatchable flag. Modal dispatch would crash pre-auth-eval, "
            f"burning $2-15 per smoke (Z3 v2 / Z4 / Z5 bug class). "
            f"Fix: add `dispatch_enabled: false` or `dispatch_blockers: ["
            f"phase_2_council_approval_required_to_lift_full_main_NotImplementedError"
            f"]` to {recipe_path.name} top level; `research_only: true` alone "
            f"is not a dispatch refusal. Per Catalog #240 + CLAUDE.md "
            f"'Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY'.",
        )

    if raises_not_impl and recipe_non_dispatchable:
        return True, (
            "PASS: trainer `_full_main` is NotImplementedError-stubbed AND "
            "recipe is research-only tagged (transparent non-dispatchable)"
        )

    if recipe_non_dispatchable:
        return (
            False,
            "FAIL: trainer `_full_main` is implemented but recipe is still "
            "non-dispatchable "
            f"({', '.join(non_dispatchable_reasons)}). "
            "operator_authorize.py would refuse before claim/provider setup; "
            "clear these recipe blockers or do not use local_pre_deploy_check "
            "as dispatch proof.",
        )

    return True, (
        "PASS: recipe-vs-trainer-state consistent "
        f"(trainer `_full_main` implemented; recipe is dispatchable"
        f"{'; research_only=true false-authority metadata preserved' if research_only else ''})"
    )


def check_dispatch_optimization_protocol(trainer: Path, recipe: str | None) -> tuple[bool, str]:
    """8th check (operator-routed 2026-05-15 DISPATCH-OPTIMIZATION-PROTOCOL).

    Routes through ``tools/canonical_dispatch_optimization_protocol.verify_dispatch_protocol_complete``
    to compute the umbrella Tier 1/2/3 verdict. Per operator directive
    2026-05-15 NON-NEGOTIABLE: *"remember the multiple deployments that
    have failed over and over because of missing optimizations? we should
    investigate and develop a protocol for those too and enforce best
    practices and production hardened optimization, extreme optimization
    and correctness and performance and scalability"*. Mirrors Catalog
    #270 STRICT preflight gate at the 30s harness layer.
    """
    # F1 fix (codex review bklem3v5j HIGH 2026-05-15) per Catalog #279
    # `check_local_pre_deploy_helper_import_failure_fails_closed`: this branch
    # USED to return PASS-VACUOUS on ImportError, which silently bypassed the
    # 8th gate when `tools/canonical_dispatch_optimization_protocol.py` was
    # missing or an untracked working-tree-only file. Per CLAUDE.md "Bugs must
    # be permanently fixed AND self-protected against" non-negotiable + sister
    # Catalog #138 / #245 / #248 / #270 fail-closed-on-corrupt-state pattern,
    # the harness MUST fail CLOSED here so a missing helper raises a strict
    # exit-1 in operator-authorize, NOT a free pass.
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        import canonical_dispatch_optimization_protocol as proto_mod  # type: ignore[import-not-found]
    except ImportError as exc:
        return False, (
            f"FAIL: Catalog #270 protocol helper unavailable "
            f"(ImportError: {exc}); refused per fail-closed-on-import "
            "discipline (Catalog #279). The 8th gate cannot be VACUOUS "
            "because operator-authorize uses the harness exit code as its "
            "gate; a missing helper would silently bypass Tier 1/2/3 "
            "umbrella protection. Fix: ensure "
            "`tools/canonical_dispatch_optimization_protocol.py` is "
            "tracked in git AND importable from the active python "
            "environment. Per CLAUDE.md 'Bugs must be permanently fixed "
            "AND self-protected against' non-negotiable."
        )
    finally:
        try:
            sys.path.remove(str(REPO_ROOT / "tools"))
        except ValueError:
            pass
    verdict = proto_mod.verify_dispatch_protocol_complete(trainer, recipe)
    if verdict.overall_pass:
        return True, (
            "PASS: dispatch optimization protocol Tier 1/2/3 all complete "
            f"(tier1.signals={sum(verdict.tier1.pass_signals.values())}/"
            f"{len(verdict.tier1.pass_signals)}; "
            f"tier2.signals={sum(verdict.tier2.pass_signals.values())}/"
            f"{len(verdict.tier2.pass_signals)}; "
            f"tier3.signals={sum(verdict.tier3.pass_signals.values())}/"
            f"{len(verdict.tier3.pass_signals)})"
        )
    head = "; ".join(verdict.blockers[:2])
    return False, (
        f"FAIL: dispatch optimization protocol failed — "
        f"tier1={verdict.tier1.overall_pass} "
        f"tier2={verdict.tier2.overall_pass} "
        f"tier3={verdict.tier3.overall_pass}; "
        f"first 2 of {len(verdict.blockers)} blockers: {head}. "
        "Per Catalog #270 + CLAUDE.md 'Production-hardened dispatch "
        "optimization protocol' non-negotiable. Run "
        "`tools/canonical_dispatch_optimization_protocol.py --trainer "
        f"{trainer.name} --recipe {recipe}` for full report."
    )


CHECKS = [
    ("py_compile", check_py_compile),
    ("trainer_importable", check_trainer_importable),
    ("full_main_implemented", check_full_main_implemented),
    ("archive_grammar", check_archive_grammar),
    ("auth_eval_reachability", check_auth_eval_reachability),
    ("canonical_inflate_device", check_canonical_inflate_device),
    ("deterministic_zip", check_deterministic_zip),
    # 7th check (operator-routed 2026-05-15 Phase 7) — recipe-vs-trainer-state
    # consistency surface; mirrors Catalog #240 STRICT preflight gate.
    ("recipe_status_consistent_with_trainer_state", "USES_RECIPE"),
    # 8th check (operator-routed 2026-05-15 DISPATCH-OPTIMIZATION-PROTOCOL) —
    # umbrella Tier 1/2/3 verdict; mirrors Catalog #270 STRICT preflight gate.
    ("dispatch_optimization_protocol", "USES_PROTOCOL"),
]

# Catalog #270 scope fix (2026-05-17): these substrate-trainer-only checks
# are skipped for tool dispatches (``tools/*.py`` or ``dispatch_kind: tool``).
# Sister of ``tac.deploy.dispatch_protocol.is_tool_dispatch`` and
# ``tools/canonical_dispatch_optimization_protocol._is_tool_dispatch``.
_TOOL_DISPATCH_SKIPPED_CHECKS = frozenset({
    "full_main_implemented",
    "archive_grammar",
    "auth_eval_reachability",
    "deterministic_zip",
})

# Sister 2026-05-17 (lane_one_arg_local_mps_vs_modal_dispatch_switch_20260517):
# Local research-signal dispatches skip the same substrate-only checks AS
# tool dispatches PLUS the dispatch_optimization_protocol umbrella (which
# the local-research-signal scope-fix in canonical_dispatch_optimization_protocol
# also handles, but skipping at the harness level avoids the round-trip).
_LOCAL_RESEARCH_SIGNAL_SKIPPED_CHECKS = frozenset({
    "full_main_implemented",
    "archive_grammar",
    "auth_eval_reachability",
    "canonical_inflate_device",
    "deterministic_zip",
})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local pre-deploy check; catches bug classes before paid dispatch."
    )
    parser.add_argument(
        "--trainer",
        required=True,
        type=Path,
        help="Path to experiments/train_substrate_*.py to validate",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="Optional recipe name (for log clarity; not validated here)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 on any FAIL (default: warn-only)",
    )
    args = parser.parse_args()

    trainer = args.trainer
    if not trainer.is_absolute():
        trainer = (REPO_ROOT / trainer).resolve()
    if not trainer.is_file():
        print(f"FATAL: trainer not found: {trainer}", file=sys.stderr)
        return 2

    label = args.recipe or trainer.name
    is_tool = _is_tool_dispatch_for_harness(trainer, args.recipe)
    is_hf_jobs_research_surrogate = _is_hf_jobs_research_surrogate_dispatch_for_harness(
        args.recipe
    )
    is_local_research_signal = _is_local_research_signal_dispatch_for_harness(
        trainer, args.recipe
    )
    if is_local_research_signal:
        kind_banner = "local_research_signal"
    elif is_hf_jobs_research_surrogate:
        kind_banner = "hf_jobs_research_surrogate"
    elif is_tool:
        kind_banner = "tool"
    else:
        kind_banner = "substrate"
    print(f"[local-pre-deploy] validating: {trainer.name}  recipe={label}  kind={kind_banner}")
    print(f"[local-pre-deploy] mode: {'STRICT (exit 1 on fail)' if args.strict else 'WARN-ONLY'}")
    if is_local_research_signal:
        print(
            "[local-pre-deploy] local research-signal dispatch detected "
            "(lane_one_arg_local_mps_vs_modal_dispatch_switch_20260517 + "
            "Catalog #270 sister scope clarification); substrate-only "
            f"checks {sorted(_LOCAL_RESEARCH_SIGNAL_SKIPPED_CHECKS)} will "
            "be skipped. The MPS / macOS-CPU contract is fail-closed: "
            "results route through canonical mps_research_signal / "
            "macos_cpu_advisory_signal manifests and are PERMANENTLY "
            "non-authoritative per CLAUDE.md 'MPS auth eval is NOISE' + "
            "Catalog #1 + #192 non-negotiables."
        )
    elif is_hf_jobs_research_surrogate:
        print(
            "[local-pre-deploy] HF Jobs research-surrogate dispatch detected; "
            "substrate-only contest-archive checks "
            f"{sorted(_TOOL_DISPATCH_SKIPPED_CHECKS)} will be skipped, while "
            "the explicit non-promotional surrogate contract is enforced by "
            "the dispatch protocol and recipe provenance gates."
        )
    elif is_tool:
        print(
            "[local-pre-deploy] tool dispatch detected (Catalog #270 scope "
            "clarification); substrate-only checks "
            f"{sorted(_TOOL_DISPATCH_SKIPPED_CHECKS)} will be skipped."
        )

    failed: list[str] = []
    for name, fn in CHECKS:
        if is_local_research_signal and name in _LOCAL_RESEARCH_SIGNAL_SKIPPED_CHECKS:
            print(
                f"  ⊘ [{name}] SKIPPED-FOR-LOCAL-RESEARCH-SIGNAL: substrate-"
                "only check; not applicable to local research-signal "
                "dispatches per lane_one_arg_local_mps_vs_modal_dispatch_"
                "switch_20260517 (2026-05-17)."
            )
            continue
        if is_hf_jobs_research_surrogate and name in _TOOL_DISPATCH_SKIPPED_CHECKS:
            print(
                f"  ⊘ [{name}] SKIPPED-FOR-HF-JOBS-RESEARCH-SURROGATE: "
                "contest-archive-only check; not applicable to advisory "
                "HF Jobs surrogate training. Non-promotional authority is "
                "enforced by recipe provenance gates."
            )
            continue
        if is_tool and name in _TOOL_DISPATCH_SKIPPED_CHECKS:
            print(
                f"  ⊘ [{name}] SKIPPED-FOR-TOOL-DISPATCH: substrate-only "
                "check; not applicable to tool dispatches per Catalog #270 "
                "scope clarification (2026-05-17)."
            )
            continue
        if fn == "USES_RECIPE":
            # 7th check needs both trainer + recipe.
            ok, msg = check_recipe_status_consistent_with_trainer_state(trainer, args.recipe)
        elif fn == "USES_PROTOCOL":
            # 8th check needs both trainer + recipe; routes through canonical
            # dispatch optimization protocol helper.
            ok, msg = check_dispatch_optimization_protocol(trainer, args.recipe)
        else:
            ok, msg = fn(trainer)
        marker = "✓" if ok else "✗"
        print(f"  {marker} [{name}] {msg}")
        if not ok:
            failed.append(name)

    if failed:
        banner = "\n[local-pre-deploy] " + "=" * 70
        print(banner)
        print(f"[local-pre-deploy] {len(failed)} CHECK(S) FAILED: {', '.join(failed)}")
        print(
            "[local-pre-deploy] Per operator 2026-05-15 directive: every Modal/Vast.ai "
            "dispatch must pass this 30s harness BEFORE spending."
        )
        print(banner.lstrip("\n"))
        return 1 if args.strict else 0

    print(f"\n[local-pre-deploy] ALL {len(CHECKS)} CHECKS PASSED. Safe to dispatch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
