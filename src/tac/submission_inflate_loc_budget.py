# SPDX-License-Identifier: MIT
"""Submission ``inflate.py`` source-size audit helpers.

The contest score charges archive bytes, not ``inflate.py`` bytes. This audit is
therefore a maintainability and review-surface guard, not score evidence. It
flags large submission runtimes so compression work is pointed at reusable
helpers, byte-closed archive payloads, and contest-relevant packers instead of
minifying Python source that the scorer never charges.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REVIEW_TARGET_INFLATE_PY_LINES = 100
DEFAULT_MAX_INFLATE_PY_LINES = 200
INFLATE_PY_LOC_DEFAULT_BUDGET_WAIVER = "INFLATE_LOC_DEFAULT_BUDGET_WAIVED:"
INFLATE_PY_LOC_HARD_WAIVER = "INFLATE_LOC_WAIVER:"
INFLATE_PY_LOC_BUDGET_WAIVER = "INFLATE_PY_LOC_BUDGET_OK:"  # legacy Catalog #328 token
_WAIVER_PLACEHOLDERS = ("<rationale>", "<reason>")


@dataclass(frozen=True)
class InflatePyLocBudgetFinding:
    """One ``inflate.py`` whose physical line count exceeds the budget."""

    rel_path: str
    line_count: int
    max_lines: int
    review_target_lines: int = DEFAULT_REVIEW_TARGET_INFLATE_PY_LINES
    budget_tier: str = "hard_budget"
    severity: str = "violation"
    size_driver_categories: tuple[str, ...] = ()
    technique_applicability: tuple[str, ...] = ()
    shared_runtime_helper_adopted: bool = False

    def format(self) -> str:
        if self.budget_tier == "default_budget":
            waiver_text = f"`# {INFLATE_PY_LOC_DEFAULT_BUDGET_WAIVER}<rationale>`"
            limit_text = f"default review target={self.review_target_lines}"
        else:
            waiver_text = (
                f"`# {INFLATE_PY_LOC_HARD_WAIVER}<rationale>` or legacy "
                f"`# {INFLATE_PY_LOC_BUDGET_WAIVER}<rationale>`"
            )
            limit_text = f"max_lines={self.max_lines}"
        return (
            f"{self.rel_path}: {self.line_count} physical lines exceeds "
            f"{limit_text}. This is not score evidence because "
            "the contest rate term charges archive.zip bytes, not inflate.py "
            "source bytes. Extract reusable helpers or add "
            f"{waiver_text} in the first 40 lines for an intentional "
            "source-faithful/runtime-closure "
            "exception."
        )


def _physical_line_count(text: str) -> int:
    if not text:
        return 0
    return len(text.splitlines())


def _has_valid_loc_budget_waiver(
    text: str,
    *,
    tokens: tuple[str, ...],
    first_n_lines: int = 40,
) -> bool:
    for line in text.splitlines()[:first_n_lines]:
        token = next((candidate for candidate in tokens if candidate in line), None)
        if token is None:
            continue
        tail = line.split(token, 1)[1].strip()
        if not tail:
            continue
        lowered = tail.lower()
        if any(placeholder in lowered for placeholder in _WAIVER_PLACEHOLDERS):
            continue
        return True
    return False


def _classify_size_driver(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    categories: list[str] = []
    if "torch.load" in lowered or "state_dict" in lowered:
        categories.append("state_dict_loader")
    if "for " in lowered and "file_list" in lowered:
        categories.append("per_video_inflate_loop")
    if "sys.path" in lowered or "from tac." in lowered or "import tac." in lowered:
        categories.append("runtime_dependency_closure")
    if any(token in lowered for token in ("brotli", "lzma", "zstd", "gzip", "zipfile")):
        categories.append("compressed_payload_decode")
    if "torch" in lowered:
        categories.append("torch_renderer")
    if "numpy" in lowered or "np." in lowered:
        categories.append("numpy_decoder")
    if len(text.splitlines()) >= 500:
        categories.append("monolithic_runtime")
    return tuple(categories) or ("unclassified_runtime_source",)


def _classify_technique_applicability(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    suggestions = ["shared_per_video_loop"]
    if "torch.load" in lowered or "state_dict" in lowered:
        suggestions.append("shared_state_dict_loader_with_sha256")
    if "select_inflate_device" not in lowered and "torch" in lowered:
        suggestions.append("canonical_select_inflate_device")
    if "raw_output_path" not in lowered and "file_list" in lowered:
        suggestions.append("shared_safe_raw_output_path")
    if "sys.path" in lowered or "from tac." in lowered:
        suggestions.append("empty_pythonpath_runtime_closure_review")
    if len(text.splitlines()) >= 500:
        suggestions.append("split_parser_renderer_writer_sections")
    return tuple(dict.fromkeys(suggestions))


def _uses_shared_runtime_helper(text: str) -> bool:
    """Return true only for real shared-helper imports, not prose mentions."""

    modules = {
        "tac.substrates._shared.inflate_runtime",
        "tac.substrates._shared.inflate_runtime_extensions",
    }
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _uses_shared_runtime_helper_line_scan(text, modules)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in modules:
            return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in modules:
                    return True
    return False


def _uses_shared_runtime_helper_line_scan(text: str, modules: set[str]) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if any(
            stripped.startswith(f"from {module} import") or stripped.startswith(f"import {module}")
            for module in modules
        ):
            return True
    return False


def iter_submission_inflate_py_files(repo_root: Path | str) -> list[Path]:
    """Return tracked-submission ``inflate.py`` surfaces under ``submissions/``.

    Public PR intakes and generated experiment results are intentionally outside
    this helper. They are forensics or rebuildable custody, while the direct
    ``submissions/<lane>/inflate.py`` tree is the operator-facing runtime surface
    that preflight should keep small enough to review.
    """

    root = Path(repo_root)
    submissions_root = root / "submissions"
    if not submissions_root.is_dir():
        return []
    targets: list[Path] = []
    for path in submissions_root.rglob("inflate.py"):
        if "__pycache__" in path.parts:
            continue
        if "_intake_" in path.parts:
            continue
        targets.append(path)
    return sorted(targets)


def scan_submission_inflate_py_loc_budget(
    repo_root: Path | str,
    *,
    max_lines: int = DEFAULT_MAX_INFLATE_PY_LINES,
    review_target_lines: int = DEFAULT_REVIEW_TARGET_INFLATE_PY_LINES,
) -> list[InflatePyLocBudgetFinding]:
    """Find submission ``inflate.py`` files above ``max_lines``.

    The count is physical source lines, matching the operator-facing review
    burden and the existing symposium audit numbers. Syntax-invalid files are
    still countable; unreadable files are skipped so this audit does not become
    a filesystem-permission gate.
    """

    root = Path(repo_root).resolve()
    review_target = min(review_target_lines, max_lines)
    findings: list[InflatePyLocBudgetFinding] = []
    for path in iter_submission_inflate_py_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        line_count = _physical_line_count(text)
        if line_count <= review_target:
            continue
        over_hard_budget = line_count > max_lines
        if over_hard_budget:
            if _has_valid_loc_budget_waiver(
                text,
                tokens=(INFLATE_PY_LOC_HARD_WAIVER, INFLATE_PY_LOC_BUDGET_WAIVER),
            ):
                continue
            budget_tier = "hard_budget"
            severity = "violation"
        else:
            if _has_valid_loc_budget_waiver(
                text,
                tokens=(
                    INFLATE_PY_LOC_DEFAULT_BUDGET_WAIVER,
                    INFLATE_PY_LOC_HARD_WAIVER,
                    INFLATE_PY_LOC_BUDGET_WAIVER,
                ),
            ):
                continue
            budget_tier = "default_budget"
            severity = "warn"
        rel_path = str(path.relative_to(root))
        findings.append(
            InflatePyLocBudgetFinding(
                rel_path=rel_path,
                line_count=line_count,
                max_lines=max_lines,
                review_target_lines=review_target,
                budget_tier=budget_tier,
                severity=severity,
                size_driver_categories=_classify_size_driver(text),
                technique_applicability=_classify_technique_applicability(text),
                shared_runtime_helper_adopted=_uses_shared_runtime_helper(text),
            )
        )
    return findings
