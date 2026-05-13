"""Static guard for HNeRV-family trainer parity contracts.

The public PR95/PR100/PR101 lineage only became score-competitive when the
trainer, scorer preprocess, EMA export, archive grammar, and inflate runtime
were all bound in one loop. This module is intentionally static and cheap: it
does not import trainers, load scorers, touch PacketIR, or dispatch work. It
checks the source-level contracts that have repeatedly caused false readiness.
"""

from __future__ import annotations

import ast
import shlex
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


def _walk_executable_function_body(func: ast.FunctionDef | None) -> list[ast.AST]:
    if func is None:
        return []

    nodes: list[ast.AST] = []

    class Visitor(ast.NodeVisitor):
        def generic_visit(self, node: ast.AST) -> None:
            nodes.append(node)
            super().generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            nodes.append(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            nodes.append(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            nodes.append(node)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            nodes.append(node)

    visitor = Visitor()
    for statement in func.body:
        visitor.visit(statement)
    return nodes


def _function_calls(func: ast.FunctionDef | None) -> list[tuple[int, str, ast.Call]]:
    if func is None:
        return []
    calls: list[tuple[int, str, ast.Call]] = []
    for node in _walk_executable_function_body(func):
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
    for node in _walk_executable_function_body(func):
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


def _shell_tokens_by_line(script: str) -> list[tuple[str, list[str]]]:
    tokenized: list[tuple[str, list[str]]] = []
    for raw_line in script.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            tokens = shlex.split(raw_line, comments=True, posix=True)
        except ValueError:
            tokens = []
        if tokens:
            tokenized.append((raw_line, tokens))
    return tokenized


def _shell_assignment_value(tokens: list[str], name: str) -> str | None:
    prefix = name + "="
    for token in tokens:
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def _shell_refers_to(token: str, *, positional: str, variable: str) -> bool:
    return token in {
        positional,
        "${" + positional[1:] + "}",
        variable,
        "${" + variable[1:] + "}",
    }


def _inspect_inflate_sh_template(inflate_sh: str) -> list[str]:
    violations: list[str] = []
    tokenized = _shell_tokens_by_line(inflate_sh)
    executable_text = "\n".join(line for line, _tokens in tokenized)

    data_arg = output_arg = file_arg = False
    for _line, tokens in tokenized:
        data_arg = data_arg or _shell_assignment_value(tokens, "DATA_DIR") in {
            "$1",
            "${1}",
        }
        output_arg = output_arg or _shell_assignment_value(tokens, "OUTPUT_DIR") in {
            "$2",
            "${2}",
        }
        file_arg = file_arg or _shell_assignment_value(tokens, "FILE_LIST") in {
            "$3",
            "${3}",
        }

    invokes_inflate_py_with_three_args = False
    for _line, tokens in tokenized:
        if "$@" in tokens:
            violations.append("inflate.sh template uses passthrough \"$@\"")
        if not any(token.endswith("inflate.py") for token in tokens):
            continue
        inflate_index = next(
            index for index, token in enumerate(tokens) if token.endswith("inflate.py")
        )
        args = tokens[inflate_index + 1:]
        if len(args) < 3:
            continue
        if (
            _shell_refers_to(args[0], positional="$1", variable="$DATA_DIR")
            and _shell_refers_to(args[1], positional="$2", variable="$OUTPUT_DIR")
            and _shell_refers_to(args[2], positional="$3", variable="$FILE_LIST")
        ):
            invokes_inflate_py_with_three_args = True

    if not (data_arg and output_arg and file_arg and invokes_inflate_py_with_three_args):
        violations.append(
            "inflate.sh template missing exact archive_dir/output_dir/file_list args"
        )
    if not any(
        tokens
        and tokens[0] == "set"
        and any(token == "-e" or token.startswith("-e") for token in tokens[1:])
        for _line, tokens in tokenized
    ):
        violations.append("inflate.sh template missing set -e/pipefail")
    for token in FORBIDDEN_NETWORK_TOKENS:
        if token in executable_text:
            violations.append(
                f"inflate.sh template contains runtime network token {token!r}"
            )
    return violations


def _is_sys_argv_index(node: ast.AST, index: int) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    if _call_name(node.value) != "sys.argv":
        return False
    slice_node = node.slice
    return isinstance(slice_node, ast.Constant) and slice_node.value == index


def _is_len_sys_argv(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and _call_name(node.func) == "len"
        and len(node.args) == 1
        and _call_name(node.args[0]) == "sys.argv"
    )


def _is_constant_value(node: ast.AST, value: object) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


def _is_len_sys_argv_not_equal_four(node: ast.Compare) -> bool:
    """Return True only for an executable exact-arity rejection condition.

    A mere comparison against ``4`` is not enough: ``len(sys.argv) == 4`` in a
    failure branch is the inverse contract and must not certify a contest
    runtime signature.
    """
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    if not isinstance(node.ops[0], ast.NotEq):
        return False
    right = node.comparators[0]
    return (
        _is_len_sys_argv(node.left)
        and _is_constant_value(right, 4)
    ) or (
        _is_constant_value(node.left, 4)
        and _is_len_sys_argv(right)
    )


def _main_enforces_three_args(main_fn: ast.FunctionDef | None) -> bool:
    if main_fn is None:
        return False
    for node in _walk_executable_function_body(main_fn):
        if isinstance(node, ast.Compare) and _is_len_sys_argv_not_equal_four(node):
            return True
    return False


def _name_assigned_from_sys_argv(main_fn: ast.FunctionDef | None, index: int) -> set[str]:
    if main_fn is None:
        return set()
    names: set[str] = set()
    for node in _walk_executable_function_body(main_fn):
        value_node: ast.AST | None = None
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            value_node = node.value
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            value_node = node.value
            targets = [node.target]
        if value_node is None:
            continue
        source = value_node
        if isinstance(source, ast.Call) and source.args:
            source = source.args[0]
        if not _is_sys_argv_index(source, index):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _expr_reads_archive_zero_bin(node: ast.AST, archive_names: set[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    call_name = _call_name(node.func)
    if call_name != "read_bytes" and not call_name.endswith(".read_bytes"):
        return False
    read_target = node.func.value if isinstance(node.func, ast.Attribute) else None
    if not isinstance(read_target, ast.BinOp) or not isinstance(read_target.op, ast.Div):
        return False
    return (
        isinstance(read_target.left, ast.Name)
        and read_target.left.id in archive_names
        and isinstance(read_target.right, ast.Constant)
        and read_target.right.value == "0.bin"
    )


def _iterates_file_list_lines(
    main_fn: ast.FunctionDef | None,
    file_list_names: set[str],
) -> bool:
    if main_fn is None:
        return False
    for node in _walk_executable_function_body(main_fn):
        if not isinstance(node, ast.For):
            continue
        split_call = node.iter
        if (
            not isinstance(split_call, ast.Call)
            or _call_name(split_call.func) != "splitlines"
        ):
            continue
        read_call = split_call.func.value if isinstance(split_call.func, ast.Attribute) else None
        if (
            not isinstance(read_call, ast.Call)
            or not _call_name(read_call.func).endswith(".read_text")
        ):
            continue
        read_target = read_call.func.value if isinstance(read_call.func, ast.Attribute) else None
        if isinstance(read_target, ast.Name) and read_target.id in file_list_names:
            return True
    return False


def _runtime_references_forbidden_scorer_symbol(module: ast.Module) -> str | None:
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name == "upstream.modules" or module_name.startswith("upstream.modules."):
                return "from upstream.modules"
            for alias in node.names:
                if alias.name in FORBIDDEN_RUNTIME_TOKENS:
                    return alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "upstream.modules" or alias.name.startswith("upstream.modules."):
                    return "import upstream.modules"
                if alias.name in FORBIDDEN_RUNTIME_TOKENS:
                    return alias.name
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_RUNTIME_TOKENS:
            return node.id
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_RUNTIME_TOKENS:
            return node.attr
    return None


def _inspect_inflate_py_template(inflate_py: str) -> list[str]:
    violations: list[str] = []
    try:
        module = ast.parse(inflate_py)
    except SyntaxError:
        return ["inflate.py template is not parseable Python"]

    main_fn = _top_level_function(module, "main")
    archive_names = _name_assigned_from_sys_argv(main_fn, 1)
    output_names = _name_assigned_from_sys_argv(main_fn, 2)
    file_list_names = _name_assigned_from_sys_argv(main_fn, 3)

    if not _main_enforces_three_args(main_fn):
        violations.append("inflate.py template does not enforce 3 positional args")
    if not file_list_names:
        violations.append("inflate.py template does not consume file_list argv")
    if not _iterates_file_list_lines(main_fn, file_list_names):
        violations.append("inflate.py template does not iterate file_list lines")
    reads_archive = False
    if main_fn is not None:
        reads_archive = any(
            _expr_reads_archive_zero_bin(node, archive_names)
            for node in _walk_executable_function_body(main_fn)
        )
    if not reads_archive:
        violations.append("inflate.py template does not consume archive_dir/0.bin")
    if not output_names:
        violations.append("inflate.py template does not consume output_dir argv")

    forbidden = _runtime_references_forbidden_scorer_symbol(module)
    if forbidden is not None:
        violations.append(f"emitted runtime contains forbidden scorer token {forbidden!r}")
    return violations


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
        violations.extend(_inspect_inflate_sh_template(inflate_sh))
    elif runtime_fn is not None:
        violations.append("_write_runtime missing inflate_sh template")

    if inflate_py:
        violations.extend(_inspect_inflate_py_template(inflate_py))
    elif runtime_fn is not None:
        violations.append("_write_runtime missing inflate_py template")

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
