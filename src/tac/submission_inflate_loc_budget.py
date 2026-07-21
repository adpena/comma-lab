# SPDX-License-Identifier: MIT
"""Historical compatibility surface for the retired ``inflate.py`` LOC audit.

The operator permanently removed the former 100-line review target and
200-line hard limit on 2026-07-21. ``inflate.py`` is a free, unsized interpreter;
only the contest decode-time budget binds. The historical constants, finding
type, and helper symbols remain importable so old callers do not break, but the
scanner is a permanent no-op. Rule-118 anti-fake protection remains in Catalog
#417 receiver-consumption bijection and the payload-cleanliness audit bundle.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REVIEW_TARGET_INFLATE_PY_LINES = 100  # historical; never enforced
DEFAULT_MAX_INFLATE_PY_LINES = 200  # historical; never enforced
INFLATE_PY_LOC_DEFAULT_BUDGET_WAIVER = "INFLATE_LOC_DEFAULT_BUDGET_WAIVED:"
INFLATE_PY_LOC_HARD_WAIVER = "INFLATE_LOC_WAIVER:"
INFLATE_PY_LOC_BUDGET_WAIVER = "INFLATE_PY_LOC_BUDGET_OK:"  # legacy Catalog #328 token
_WAIVER_PLACEHOLDERS = ("<rationale>", "<reason>")


@dataclass(frozen=True)
class InflatePyLocBudgetFinding:
    """Historical finding shape retained for import/schema compatibility."""

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
        return (
            f"{self.rel_path}: historical LOC telemetry={self.line_count}; "
            "restriction retired 2026-07-21 and this record has no admission "
            "or waiver authority"
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
    """Return tracked ``inflate.py`` surfaces for informational inventory.

    Public PR intakes and generated experiment results are intentionally outside
    this helper. They are forensics or rebuildable custody, while the direct
    ``submissions/<lane>/inflate.py`` tree remains useful for receiver and
    payload-cleanliness audits; source length is unrestricted.
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
    """Return no findings for every source size.

    Permanent operator decision, 2026-07-21: the line-count restriction is
    void. Parameters remain accepted only for source compatibility. Anti-fake
    enforcement lives in #417 and payload-cleanliness, not source length.
    """

    _ = (repo_root, max_lines, review_target_lines)
    return []
