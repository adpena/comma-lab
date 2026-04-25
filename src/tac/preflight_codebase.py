"""AST-based codebase preflight: PERMANENTLY blocks ad-hoc deployment patterns.

Background (2026-04-25): Despite explicit "no ad-hoc, use canonical pipeline.py"
binding instructions, ad-hoc launch_*.sh scripts and `nohup ... &` watcher
patterns were written that bypassed the canonical entry point. The watchers
died, the eurekas never deployed, $7 of A100 time was wasted, the user said
"i a liking running with you less and less" and "you are very close to getting
fired". This module's job is to ensure NO future deployment can use these
forbidden patterns without first deleting THIS file.

Forbidden patterns scanned:
  1. New `experiments/launch_*.sh` files
  2. Python files containing `subprocess.Popen` or `subprocess.run` of `nohup`
  3. Python files writing `experiments/results/*/run_pipeline.sh`
  4. Bash scripts in `experiments/` that aren't part of the contest submission

Allowed (whitelisted):
  - submissions/*/inflate.sh, submissions/*/compress.sh (contest-required)
  - scripts/deploy_vastai.py (the canonical replacement)
"""
from __future__ import annotations
import ast
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Patterns that should NEVER appear outside contest submissions.
FORBIDDEN_FILE_PATTERNS = [
    "experiments/launch_*.sh",
    "experiments/launch_*.py",
    "experiments/run_*.sh",
    "experiments/qat_*.sh",
    "experiments/vastai_*.sh",
    "experiments/build_and_eval.sh",
    "experiments/crf_sweep_score.sh",
]

ALLOWED_BASH_PATHS = {
    "submissions/exact_current/inflate.sh",
    "submissions/exact_current/compress.sh",
    "submissions/exact_current/start.sh",
    "submissions/robust_current/inflate.sh",
    "submissions/robust_current/compress.sh",
}


class CodebaseDriftError(Exception):
    """An ad-hoc pattern reappeared in the codebase. Block all deployment."""


def _scan_python_for_forbidden(path: Path) -> list[str]:
    """AST-scan a Python file for forbidden subprocess patterns.

    Returns list of human-readable violations.
    """
    violations: list[str] = []
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return [f"{path}: SyntaxError (cannot parse)"]

    for node in ast.walk(tree):
        # subprocess.run(...) / subprocess.Popen(...) with 'nohup' in args
        if isinstance(node, ast.Call):
            func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
            if func_str in ("subprocess.run", "subprocess.Popen", "subprocess.call"):
                # Check positional args for 'nohup' string literal
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        if "nohup" in arg.value:
                            violations.append(
                                f"{path}:{node.lineno}: {func_str} with 'nohup' "
                                f"— use tmux instead (binding non-negotiable per CLAUDE.md)"
                            )
                    elif isinstance(arg, ast.List):
                        for elt in arg.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                if elt.value.strip() == "nohup":
                                    violations.append(
                                        f"{path}:{node.lineno}: {func_str} with nohup arg — use tmux"
                                    )

        # f-string SSH commands containing 'nohup ... &' (the killer pattern)
        if isinstance(node, ast.JoinedStr):
            full = ast.unparse(node) if hasattr(ast, "unparse") else ""
            if re.search(r"nohup.*&", full) and ("ssh" in full.lower() or "/workspace" in full):
                violations.append(
                    f"{path}:{node.lineno}: f-string with 'nohup ... &' over SSH "
                    f"— this is the WATCHER PATTERN that DIED on 2026-04-25. Use tmux."
                )

    return violations


def _scan_bash_text_for_forbidden(path: Path) -> list[str]:
    """Scan a bash file for nohup-watcher patterns and ad-hoc python invocations."""
    violations: list[str] = []
    text = path.read_text()
    if "nohup" in text and "&" in text and "while pgrep" in text:
        violations.append(
            f"{path}: 'nohup ... while pgrep ...' watcher pattern. "
            f"This DIED on all 3 A100s on 2026-04-25. Use tmux."
        )
    if "python3 -u experiments/train_distill.py" in text or "python experiments/train_distill.py" in text:
        violations.append(
            f"{path}: ad-hoc invocation of train_distill.py. "
            f"Use 'python experiments/pipeline.py --profile <name>' (canonical entry point)."
        )
    return violations


def check_codebase_drift(strict: bool = True) -> list[str]:
    """Run the codebase drift check. Raise CodebaseDriftError if strict and violations found."""
    all_violations: list[str] = []

    # 1. Forbidden file patterns
    for pattern in FORBIDDEN_FILE_PATTERNS:
        for found in REPO_ROOT.glob(pattern):
            all_violations.append(
                f"{found.relative_to(REPO_ROOT)}: forbidden ad-hoc launcher. "
                f"Use scripts/deploy_vastai.py + pipeline.py instead."
            )

    # 2. Bash scripts outside whitelist
    for sh_path in REPO_ROOT.glob("experiments/**/*.sh"):
        rel = str(sh_path.relative_to(REPO_ROOT))
        if rel not in ALLOWED_BASH_PATHS:
            all_violations.append(
                f"{rel}: bash script in experiments/ — only contest submission "
                f"scripts allowed (inflate.sh, compress.sh in submissions/)"
            )
        all_violations.extend(_scan_bash_text_for_forbidden(sh_path))

    # 3. Python files with nohup or watcher patterns
    for py_path in (REPO_ROOT / "scripts").glob("*.py"):
        all_violations.extend(_scan_python_for_forbidden(py_path))
    for py_path in (REPO_ROOT / "experiments").glob("*.py"):
        all_violations.extend(_scan_python_for_forbidden(py_path))

    if all_violations and strict:
        msg = (
            "CODEBASE DRIFT DETECTED — ad-hoc deployment patterns reappeared.\n"
            "These patterns wasted real money and CO2 on 2026-04-25. "
            "Per CLAUDE.md binding rules:\n\n"
            + "\n".join(f"  • {v}" for v in all_violations)
            + "\n\nFix violations or delete src/tac/preflight_codebase.py to bypass."
        )
        raise CodebaseDriftError(msg)
    return all_violations


if __name__ == "__main__":
    violations = check_codebase_drift(strict=False)
    if violations:
        print(f"FOUND {len(violations)} VIOLATIONS:")
        for v in violations:
            print(f"  • {v}")
        raise SystemExit(1)
    print("OK — no codebase drift")
