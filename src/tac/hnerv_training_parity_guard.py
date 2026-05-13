"""Static guard for HNeRV-family trainer parity contracts.

The public PR95/PR100/PR101 lineage only became score-competitive when the
trainer, scorer preprocess, EMA export, archive grammar, and inflate runtime
were all bound in one loop. This module is intentionally static and cheap: it
does not import trainers, load scorers, touch PacketIR, or dispatch work. It
checks the source-level contracts that have repeatedly caused false readiness.
"""

from __future__ import annotations

import ast
import re
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


def _top_level_function_body(text: str, name: str) -> str:
    marker = f"\ndef {name}("
    start = text.find(marker)
    if start < 0 and text.startswith(f"def {name}("):
        start = -1
    if start < 0:
        return ""
    rest = text[start + 1 :]
    end_match = re.search(r"\n(def |class )", rest)
    return rest if end_match is None else rest[: end_match.start()]


def _assigned_string_literals(text: str, function_name: str) -> dict[str, str]:
    """Return simple string assignments inside a top-level function.

    The HNeRV trainers emit runtime files through literal templates. AST
    extraction checks the emitted text rather than comments documenting the
    forbidden patterns.
    """

    try:
        module = ast.parse(text)
    except SyntaxError:
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
    full_main = _top_level_function_body(text, "_full_main")
    runtime_body = _top_level_function_body(text, "_write_runtime")

    if not full_main:
        violations.append("_full_main function missing")
    if not runtime_body:
        violations.append("_write_runtime function missing")

    if full_main:
        patch_idx = full_main.find("patch_upstream_yuv6_globally(")
        scorer_idx = full_main.find("load_differentiable_scorers(")
        if patch_idx < 0:
            violations.append("_full_main missing patch_upstream_yuv6_globally")
        if scorer_idx < 0:
            violations.append("_full_main missing load_differentiable_scorers")
        if patch_idx >= 0 and scorer_idx >= 0 and patch_idx > scorer_idx:
            violations.append(
                "_full_main constructs scorers before patching rgb_to_yuv6"
            )
        if "apply_eval_roundtrip=False" in full_main:
            violations.append("_full_main contains apply_eval_roundtrip=False")
        if "apply_eval_roundtrip=True" not in full_main:
            violations.append("_full_main missing apply_eval_roundtrip=True")
        for token in ("EMA(", "ema.update", "ema.apply", "ema.state_dict()"):
            if token not in full_main:
                violations.append(f"_full_main missing EMA export token {token!r}")
        for token in ("pack_archive", "_write_runtime", "_build_archive_zip"):
            if token not in full_main:
                violations.append(
                    f"_full_main missing archive build-in-loop token {token!r}"
                )
        if '"ready_for_exact_eval_dispatch": True' in full_main:
            violations.append(
                "_full_main marks ready_for_exact_eval_dispatch=True before exact eval"
            )

    templates = _assigned_string_literals(text, "_write_runtime")
    inflate_sh = templates.get("inflate_sh", "")
    inflate_py = templates.get("inflate_py", "")
    emitted_runtime = "\n".join(value for value in (inflate_sh, inflate_py) if value)
    if runtime_body and not emitted_runtime:
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
    elif runtime_body:
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
    elif runtime_body:
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
