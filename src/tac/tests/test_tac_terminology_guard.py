from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "check_tac_terminology.py"
ALL_LANES = REPO / "tools" / "all_lanes_preflight.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("check_tac_terminology_under_test", TOOL)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_live_docs_pass_tac_terminology_guard() -> None:
    module = _load_tool()

    findings = module.check_repo(REPO)

    assert findings == []


def test_forbidden_tac_codec_definition_is_rejected(tmp_path: Path) -> None:
    module = _load_tool()
    for relpath in module.CANONICAL_FILES:
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "`tac` means **Task-Aware Codec**\n"
        "src/tac/README.md\n"
        "src/comma_lab/README.md\n"
        "docs/terminology_and_boundaries.md\n",
        encoding="utf-8",
    )

    findings = module.check_repo(tmp_path)

    rendered = "\n".join(finding.render() for finding in findings)
    assert "Task-Aware Compression, not Task-Aware Codec" in rendered


def test_init_files_have_positive_terminology_requirements(tmp_path: Path) -> None:
    module = _load_tool()
    for relpath, text in {
        "README.md": (
            "`tac` means **Task-Aware Compression**\n"
            "src/tac/README.md\n"
            "src/comma_lab/README.md\n"
            "docs/terminology_and_boundaries.md\n"
        ),
        "docs/terminology_and_boundaries.md": (
            "This document is the canonical naming and package-boundary reference\n"
            'Never expand TAC as "Task-Aware Codec."\n'
            "Contest Compliance Boundary\n"
            "Procedural generation from an archive-contained seed\n"
            "Constants in `inflate.py` may describe how to decode a charged payload\n"
            "Package Ownership\n"
        ),
        "src/tac/README.md": (
            "# tac - Task-Aware Compression\n"
            "Video coding for machines\n"
            "Feature coding for machines\n"
            "Use **compression** for the project and research program\n"
            "procedural generation from archive-contained seeds or weights\n"
        ),
        "src/comma_lab/README.md": (
            "It is intentionally not the compression engine.\n"
            "`tac`: Task-Aware Compression library\n"
            "Never create score authority from `comma_lab` alone\n"
        ),
        "src/tac/__init__.py": "Task-Aware Compression (tac)\n",
        "src/comma_lab/__init__.py": "repository operations only\n",
        "pyproject.toml": (
            'description = "Task-Aware Compression:\n'
            '"task-aware-compression"\n'
            '"video-coding-for-machines"\n'
        ),
    }.items():
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    findings = module.check_repo(tmp_path)

    rendered = "\n".join(finding.render() for finding in findings)
    assert "src/tac/__init__.py" in rendered
    assert "src/comma_lab/__init__.py" in rendered


def test_cli_strict_passes_live_docs() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL), "--strict", "--json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert '"ok": true' in result.stdout


def test_all_lanes_preflight_exposes_terminology_guard() -> None:
    text = ALL_LANES.read_text(encoding="utf-8")

    assert "TAC_TERMINOLOGY_AUDIT" in text
    assert "Gate #32: TAC terminology canonicalization" in text
    assert '["--strict"]' in text
