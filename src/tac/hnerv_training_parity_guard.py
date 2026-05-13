"""Static guard for HNeRV-family trainer parity contracts.

The public PR95/PR100/PR101 lineage only became score-competitive when the
trainer, scorer preprocess, EMA export, archive grammar, and inflate runtime
were all bound in one loop. This module is intentionally static and cheap: it
does not import trainers, load scorers, touch PacketIR, or dispatch work. It
checks the source-level contracts that have repeatedly caused false readiness.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

FORBIDDEN_RUNTIME_TOKENS: tuple[str, ...] = (
    "PoseNet",
    "SegNet",
    "FastViT",
    "EfficientNet",
    "from upstream.modules",
    "import upstream.modules",
    "rgb_to_yuv6",
)

FORBIDDEN_NETWORK_TOKENS: tuple[str, ...] = (
    "--extra-index-url",
    "--find-links",
    "--index-url",
    "curl ",
    "git clone",
    "http://",
    "https://",
    "pip install",
    "uv pip install",
    "uv run --with",
    "wget ",
)


@dataclass(frozen=True)
class HnervTrainingParityReport:
    """Result of a static HNeRV trainer parity scan."""

    path: str
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations


def _parse_module(text: str) -> ast.Module | None:
    try:
        return ast.parse(text)
    except SyntaxError:
        return None


def _top_level_function(module: ast.Module | None, name: str) -> ast.FunctionDef | None:
    if module is None:
        return None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _call_name(func.value)
        return f"{base}.{func.attr}" if base else func.attr
    return ""


def _function_calls(func: ast.FunctionDef | None) -> list[tuple[int, str, ast.Call]]:
    if func is None:
        return []
    calls: list[tuple[int, str, ast.Call]] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            calls.append((getattr(node, "lineno", 0), _call_name(node.func), node))
    calls.sort(key=lambda item: item[0])
    return calls


def _has_call_named(calls: list[tuple[int, str, ast.Call]], name: str) -> bool:
    return any(call_name == name or call_name.endswith("." + name) for _, call_name, _ in calls)


def _first_call_lineno(calls: list[tuple[int, str, ast.Call]], name: str) -> int | None:
    for lineno, call_name, _node in calls:
        if call_name == name or call_name.endswith("." + name):
            return lineno
    return None


def _has_bool_keyword(calls: list[tuple[int, str, ast.Call]], keyword: str, value: bool) -> bool:
    for _lineno, _name, call in calls:
        for kw in call.keywords:
            if kw.arg != keyword:
                continue
            if isinstance(kw.value, ast.Constant) and kw.value.value is value:
                return True
    return False


def _contains_ready_for_exact_eval_true(func: ast.FunctionDef | None) -> bool:
    if func is None:
        return False
    for node in ast.walk(func):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=False):
            if (
                isinstance(key, ast.Constant)
                and key.value == "ready_for_exact_eval_dispatch"
                and isinstance(value, ast.Constant)
                and value.value is True
            ):
                return True
    return False


def _assigned_string_literals(text: str, function_name: str) -> dict[str, str]:
    """Return simple string assignments inside a top-level function.

    The HNeRV trainers emit runtime files through literal templates. AST
    extraction checks the emitted text rather than comments documenting the
    forbidden patterns.
    """

    module = _parse_module(text)
    if module is None:
        return {}
    values: dict[str, str] = {}
    for top in module.body:
        if not isinstance(top, ast.FunctionDef) or top.name != function_name:
            continue
        for node in ast.walk(top):
            target_names: list[str] = []
            value_node: ast.AST | None = None
            if isinstance(node, ast.Assign):
                target_names = [
                    target.id
                    for target in node.targets
                    if isinstance(target, ast.Name)
                ]
                value_node = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target_names = [node.target.id]
                value_node = node.value
            if value_node is None:
                continue
            try:
                value = ast.literal_eval(value_node)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(value, str):
                continue
            for target in target_names:
                values[target] = value
        break
    return values


def inspect_hnerv_training_parity_source(
    text: str,
    *,
    path_label: str = "<source>",
) -> HnervTrainingParityReport:
    """Inspect one HNeRV-family trainer for parity-readiness contracts.

    Covered contracts:
      - differentiable rgb_to_yuv6 patch is applied before scorer construction;
      - score loss calls keep eval-roundtrip enabled;
      - EMA is updated, applied for validation/export, and saved as shadow state;
      - full training builds a real archive/runtime packet in-loop;
      - emitted inflate runtime uses the exact contest 3-argument signature.
    """

    violations: list[str] = []
    module = _parse_module(text)
    full_main_fn = _top_level_function(module, "_full_main")
    runtime_fn = _top_level_function(module, "_write_runtime")
    full_main_calls = _function_calls(full_main_fn)

    if full_main_fn is None:
        violations.append("_full_main function missing")
    if runtime_fn is None:
        violations.append("_write_runtime function missing")

    if full_main_fn is not None:
        patch_lineno = _first_call_lineno(full_main_calls, "patch_upstream_yuv6_globally")
        scorer_lineno = _first_call_lineno(full_main_calls, "load_differentiable_scorers")
        if patch_lineno is None:
            violations.append("_full_main missing patch_upstream_yuv6_globally")
        if scorer_lineno is None:
            violations.append("_full_main missing load_differentiable_scorers")
        if patch_lineno is not None and scorer_lineno is not None and patch_lineno > scorer_lineno:
            violations.append(
                "_full_main constructs scorers before patching rgb_to_yuv6"
            )
        if _has_bool_keyword(full_main_calls, "apply_eval_roundtrip", False):
            violations.append("_full_main contains apply_eval_roundtrip=False")
        if not _has_bool_keyword(full_main_calls, "apply_eval_roundtrip", True):
            violations.append("_full_main missing apply_eval_roundtrip=True")
        for token, call_name in (
            ("EMA(", "EMA"),
            ("ema.update", "ema.update"),
            ("ema.apply", "ema.apply"),
            ("ema.state_dict()", "ema.state_dict"),
        ):
            if not _has_call_named(full_main_calls, call_name):
                violations.append(f"_full_main missing EMA export call {token!r}")
        for call_name in ("pack_archive", "_write_runtime", "_build_archive_zip"):
            if not _has_call_named(full_main_calls, call_name):
                violations.append(
                    f"_full_main missing archive build-in-loop call {call_name!r}"
                )
        if _contains_ready_for_exact_eval_true(full_main_fn):
            violations.append(
                "_full_main marks ready_for_exact_eval_dispatch=True before exact eval"
            )

    templates = _assigned_string_literals(text, "_write_runtime")
    inflate_sh = templates.get("inflate_sh", "")
    inflate_py = templates.get("inflate_py", "")
    emitted_runtime = "\n".join(value for value in (inflate_sh, inflate_py) if value)
    if runtime_fn is not None and not emitted_runtime:
        violations.append(
            "_write_runtime emits no literal inflate_sh/inflate_py templates"
        )

    if inflate_sh:
        has_arg1 = (
            "$1" in inflate_sh or "${1}" in inflate_sh or "$DATA_DIR" in inflate_sh
        )
        has_arg2 = (
            "$2" in inflate_sh or "${2}" in inflate_sh or "$OUTPUT_DIR" in inflate_sh
        )
        has_arg3 = (
            "$3" in inflate_sh or "${3}" in inflate_sh or "$FILE_LIST" in inflate_sh
        )
        if not (has_arg1 and has_arg2 and has_arg3):
            violations.append(
                "inflate.sh template missing exact archive_dir/output_dir/file_list args"
            )
        if "set -euo pipefail" not in inflate_sh and "set -e" not in inflate_sh:
            violations.append("inflate.sh template missing set -e/pipefail")
        if '"$@"' in inflate_sh:
            violations.append("inflate.sh template uses passthrough \"$@\"")
        for token in FORBIDDEN_NETWORK_TOKENS:
            if token in inflate_sh:
                violations.append(
                    f"inflate.sh template contains runtime network token {token!r}"
                )
    elif runtime_fn is not None:
        violations.append("_write_runtime missing inflate_sh template")

    if inflate_py:
        has_three_arg_check = (
            "len(sys.argv) != 4" in inflate_py or "len(sys.argv)!=4" in inflate_py
        )
        if not has_three_arg_check:
            violations.append("inflate.py template does not enforce 3 positional args")
        if "sys.argv[3]" not in inflate_py:
            violations.append("inflate.py template does not consume file_list argv")
        if "splitlines()" not in inflate_py:
            violations.append("inflate.py template does not iterate file_list lines")
        consumes_zero_bin = (
            "archive_dir / '0.bin'" in inflate_py
            or 'archive_dir / "0.bin"' in inflate_py
        )
        if not consumes_zero_bin:
            violations.append("inflate.py template does not consume archive_dir/0.bin")
    elif runtime_fn is not None:
        violations.append("_write_runtime missing inflate_py template")

    for token in FORBIDDEN_RUNTIME_TOKENS:
        if token in emitted_runtime:
            violations.append(f"emitted runtime contains forbidden scorer token {token!r}")

    return HnervTrainingParityReport(path=path_label, violations=tuple(violations))


def inspect_hnerv_training_parity_file(path: Path) -> HnervTrainingParityReport:
    text = path.read_text(encoding="utf-8", errors="replace")
    return inspect_hnerv_training_parity_source(text, path_label=str(path))


def assert_hnerv_training_parity_file(path: Path) -> None:
    """Raise AssertionError with a compact message if ``path`` fails."""

    report = inspect_hnerv_training_parity_file(path)
    if report.violations:
        raise AssertionError(
            f"HNeRV trainer parity guard failed for {report.path}:\n  "
            + "\n  ".join(report.violations)
        )


__all__ = [
    "HnervTrainingParityReport",
    "assert_hnerv_training_parity_file",
    "inspect_hnerv_training_parity_file",
    "inspect_hnerv_training_parity_source",
]
