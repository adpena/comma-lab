from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "check_tac_terminology.py"
ALL_LANES = REPO / "tools" / "all_lanes_preflight.py"

MINIMAL_CANONICAL_TEXTS = {
    "README.md": (
        "`tac` means **Task-Aware Compression**\n"
        "src/tac/README.md\n"
        "src/comma_lab/README.md\n"
        "docs/terminology_and_boundaries.md\n"
        "docs/contest_compliance_authority.md\n"
    ),
    "HANDOFF.md": (
        "`tac` means Task-Aware Compression\n"
        "`comma_lab` is the lab\noperations\n"
        "docs/terminology_and_boundaries.md\n"
        "100 * seg_distortion + sqrt(10 * pose_distortion) + 25 * archive_bytes /\n"
        "37,545,489\n"
    ),
    "PROGRAM.md": (
        "`tac` means Task-Aware Compression\n"
        "A codec is a concrete encoder/decoder or wire format\n"
        "`comma_lab` owns lab operations\n"
        "docs/terminology_and_boundaries.md\n"
    ),
    "SYSTEM_MAP.md": "# The canonical Task-Aware Compression\n# library and compression engine\n",
    "docs/contest_compliance_authority.md": (
        "# Contest Compliance Authority\n"
        "Authority Ladder\n"
        "Public PR Precedents\n"
        "archive_seeded\n"
        "runtime_constant\n"
        "score-bearing information must be charged through `archive.zip`\n"
        "#35 tensor_inversion\n"
        "#68 loophole_v2\n"
        "#78 qzs3_script_payload_r147\n"
    ),
    "docs/terminology_and_boundaries.md": (
        "This document is the canonical naming and package-boundary reference\n"
        'Never expand TAC as "Task-Aware Codec."\n'
        "`comma_lab.task_codec` is a legacy compatibility namespace\n"
        "Contest Compliance Boundary\n"
        "Procedural generation from an archive-contained seed\n"
        "Constants in `inflate.py` may describe how to decode a charged payload\n"
        "Package Ownership\n"
        "docs/contest_compliance_authority.md\n"
    ),
    "src/tac/README.md": (
        "# tac - Task-Aware Compression\n"
        "Video coding for machines\n"
        "Feature coding for machines\n"
        "Use **compression** for the project and research program\n"
        "procedural generation from archive-contained seeds or weights\n"
        "docs/contest_compliance_authority.md\n"
    ),
    "src/comma_lab/README.md": (
        "It is intentionally not the compression engine.\n"
        "`tac`: Task-Aware Compression library\n"
        "`comma_lab.task_codec`\n"
        "Never create score authority from `comma_lab` alone\n"
        "docs/contest_compliance_authority.md\n"
    ),
    "src/tac/__init__.py": (
        "Task-Aware Compression (tac)\n"
        "compression optimized against\ndownstream machine-perception tasks\n"
        '"codec" names a concrete encoder/decoder or wire\nformat\n'
    ),
    "src/comma_lab/__init__.py": "repository operations for Task-Aware Compression research\nlossless_review_tracker\nstate_sync\n",
    "pyproject.toml": (
        'description = "Task-Aware Compression:\n'
        '"task-aware-compression"\n'
        '"video-coding-for-machines"\n'
    ),
}


def _load_tool():
    spec = importlib.util.spec_from_file_location("check_tac_terminology_under_test", TOOL)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_canonical_texts(tmp_path: Path, overrides: dict[str, str] | None = None) -> None:
    texts = dict(MINIMAL_CANONICAL_TEXTS)
    if overrides:
        texts.update(overrides)
    for relpath, text in texts.items():
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


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
        "HANDOFF placeholder\n"
        "PROGRAM placeholder\n"
        "SYSTEM_MAP placeholder\n"
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
    _write_canonical_texts(
        tmp_path,
        {
            "src/tac/__init__.py": "Task-Aware Compression (tac)\n",
            "src/comma_lab/__init__.py": "repository operations only\n",
        },
    )

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
