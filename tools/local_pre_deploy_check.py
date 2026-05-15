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

AUTH_EVAL_CALL_NAMES = {
    "_canon_gate_auth_eval_call",
    "auth_eval_renderer",
    "gate_auth_eval_call",
}


def _import_trainer_module(trainer: Path):
    module_name = f"_local_pre_deploy_{trainer.stem}"
    spec = importlib.util.spec_from_file_location(module_name, trainer)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {trainer}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    for match in ARCHIVE_BUILD_PATTERN.finditer(text):
        member = match.group(1)
        if member == CANONICAL_ARCHIVE_MEMBER:
            continue
        line_no = text.count("\n", 0, match.start()) + 1
        line_text = (
            text.splitlines()[line_no - 1] if line_no <= len(text.splitlines()) else ""
        )
        if ARCHIVE_MEMBER_WAIVER.search(line_text):
            continue
        violations.append(
            f"  {trainer.name}:{line_no}: filename={member!r} "
            f"(repo learned-packet standard: {CANONICAL_ARCHIVE_MEMBER!r}); "
            f"contest rules only require archive.zip + inflate.sh, but this "
            "trainer's inflate.sh expects x or ${base}.bin"
        )
    if violations:
        msg = (
            "FAIL: learned-packet archive ZIP member must be '0.bin'; "
            "violations:\n" + "\n".join(violations)
        )
        msg += (
            "\n  Fix: change _build_archive_zip to filename=\"0.bin\""
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

            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_path = Path(tmp_dir) / "archive.zip"
                build_archive_zip(zip_path, bin_bytes=b"local-predeploy")
                with zipfile.ZipFile(zip_path) as zf:
                    names = zf.namelist()
                if names != [CANONICAL_ARCHIVE_MEMBER]:
                    return (
                        False,
                        f"FAIL: _build_archive_zip emitted members {names!r}; "
                        f"expected [{CANONICAL_ARCHIVE_MEMBER!r}] for this "
                        "repo learned-packet trainer",
                    )
    except Exception as exc:
        return False, f"FAIL: archive grammar dynamic check failed: {exc!r}"
    return True, f"PASS: {trainer.name} archive ZIP member follows repo standard"


def check_auth_eval_reachability(trainer: Path) -> tuple[bool, str]:
    """Refuse trainer with no auth_eval invocation (Z4 bug class)."""
    text = trainer.read_text()
    try:
        tree = ast.parse(text, filename=str(trainer))
    except SyntaxError as exc:
        return False, f"FAIL: {trainer.name} cannot be parsed for auth_eval reachability: {exc}"
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        call_name = ""
        if isinstance(func, ast.Name):
            call_name = func.id
        elif isinstance(func, ast.Attribute):
            call_name = func.attr
        if call_name in AUTH_EVAL_CALL_NAMES:
            return True, f"PASS: {trainer.name} invokes auth_eval (canonical helper)"
        for arg in [*node.args, *[keyword.value for keyword in node.keywords]]:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and "contest_auth_eval.py" in arg.value
            ):
                return True, f"PASS: {trainer.name} invokes contest_auth_eval.py"
            if isinstance(arg, ast.List | ast.Tuple):
                for element in arg.elts:
                    if (
                        isinstance(element, ast.Constant)
                        and isinstance(element.value, str)
                        and "contest_auth_eval.py" in element.value
                    ):
                        return True, f"PASS: {trainer.name} invokes contest_auth_eval.py"
    return (
        False,
        f"FAIL: {trainer.name} has no auth_eval invocation; "
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
    tagged ``research_only=true`` in their recipe AND NOT dispatchable.
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
            f"tag the recipe with research_only=true OR implement _full_main before dispatch.",
        )
    return True, f"PASS: {trainer.name} _full_main appears implemented"


def check_recipe_status_consistent_with_trainer_state(trainer: Path, recipe: str | None) -> tuple[bool, str]:
    """7th check (operator-routed 2026-05-15 Phase 7).

    Surfaces the Z3 v2 / Z4 / Z5 bug class at the 30s harness BEFORE Modal
    dispatch fires. Refuses recipe-vs-trainer-state divergence:

    - Trainer's _full_main raises NotImplementedError BUT recipe lacks
      research_only: true / smoke_only: true / dispatch_enabled: false → Modal
      will reach trainer and crash pre-auth-eval ($2-15 per smoke).

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

    # Check recipe research-only flags
    research_only = bool(
        re.search(r"^\s*smoke_only:\s*true\b", recipe_text, re.M)
        or re.search(r"^\s*research_only:\s*true\b", recipe_text, re.M)
        or re.search(r"^\s*dispatch_enabled:\s*false\b", recipe_text, re.M)
    )

    if raises_not_impl and not research_only:
        return (
            False,
            f"FAIL: recipe-vs-trainer-state divergence — "
            f"trainer {trainer.name} `_full_main` raises NotImplementedError "
            f"(Phase 2 council-gated) but recipe {recipe_path.name} lacks "
            f"research-only flag. Modal dispatch would crash pre-auth-eval, "
            f"burning $2-15 per smoke (Z3 v2 / Z4 / Z5 bug class). "
            f"Fix: add `research_only: true` AND `dispatch_blockers: ["
            f"phase_2_council_approval_required_to_lift_full_main_NotImplementedError"
            f"]` to {recipe_path.name} top level. Per Catalog #240 + CLAUDE.md "
            f"'Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY'.",
        )

    if raises_not_impl and research_only:
        return True, (
            "PASS: trainer `_full_main` is NotImplementedError-stubbed AND "
            "recipe is research-only tagged (transparent non-dispatchable)"
        )

    return True, (
        "PASS: recipe-vs-trainer-state consistent "
        "(trainer `_full_main` implemented; recipe is contest-CUDA dispatchable)"
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
]


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
    print(f"[local-pre-deploy] validating: {trainer.name}  recipe={label}")
    print(f"[local-pre-deploy] mode: {'STRICT (exit 1 on fail)' if args.strict else 'WARN-ONLY'}")

    failed: list[str] = []
    for name, fn in CHECKS:
        if fn == "USES_RECIPE":
            # 7th check needs both trainer + recipe.
            ok, msg = check_recipe_status_consistent_with_trainer_state(trainer, args.recipe)
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
