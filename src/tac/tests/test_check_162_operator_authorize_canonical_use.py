from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_operator_authorize_canonical_use,
)


def _write_wrapper(repo: Path, name: str, body: str) -> Path:
    scripts = repo / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    wrapper = scripts / name
    wrapper.write_text(body)
    return wrapper


def test_check_162_allows_wrapper_that_delegates_to_canonical_tool(
    tmp_path: Path,
) -> None:
    _write_wrapper(
        tmp_path,
        "operator_authorize_example.sh",
        """#!/bin/bash
set -euo pipefail
.venv/bin/python tools/operator_authorize.py --recipe example "$@"
""",
    )

    assert check_operator_authorize_canonical_use(repo_root=tmp_path, strict=True) == []


def test_check_162_allows_wrapper_that_delegates_to_smoke_before_full(
    tmp_path: Path,
) -> None:
    _write_wrapper(
        tmp_path,
        "operator_authorize_substrate_example_modal_a100_dispatch.sh",
        """#!/bin/bash
set -euo pipefail
.venv/bin/python tools/run_modal_smoke_before_full.py \\
    --recipe substrate_example_modal_a100_dispatch "$@"
""",
    )

    assert check_operator_authorize_canonical_use(repo_root=tmp_path, strict=True) == []


def test_check_162_flags_wrapper_without_canonical_tool(tmp_path: Path) -> None:
    _write_wrapper(
        tmp_path,
        "operator_authorize_legacy.sh",
        """#!/bin/bash
set -euo pipefail
read -r -p "Proceed? [y/N] " confirm
""",
    )

    violations = check_operator_authorize_canonical_use(repo_root=tmp_path)

    assert len(violations) == 1
    assert "tools/operator_authorize.py" in violations[0]


def test_check_162_ignores_comment_only_canonical_mentions(tmp_path: Path) -> None:
    _write_wrapper(
        tmp_path,
        "operator_authorize_comment_only.sh",
        """#!/bin/bash
# TODO: migrate to tools/operator_authorize.py --recipe example
set -euo pipefail
echo "still bespoke"
""",
    )

    violations = check_operator_authorize_canonical_use(repo_root=tmp_path)

    assert len(violations) == 1
    assert "operator_authorize_comment_only.sh" in violations[0]


def test_check_162_honors_header_legacy_waiver(tmp_path: Path) -> None:
    _write_wrapper(
        tmp_path,
        "operator_authorize_legacy_allowed.sh",
        """#!/bin/bash
# OPERATOR_AUTHORIZE_LEGACY_OK: bespoke one-release compatibility bridge
set -euo pipefail
read -r -p "Proceed? [y/N] " confirm
""",
    )

    assert check_operator_authorize_canonical_use(repo_root=tmp_path, strict=True) == []


def test_check_162_strict_mode_raises_on_uncanonical_wrapper(
    tmp_path: Path,
) -> None:
    _write_wrapper(
        tmp_path,
        "operator_authorize_bad.sh",
        """#!/bin/bash
set -euo pipefail
echo "bespoke duplicate implementation"
""",
    )

    with pytest.raises(PreflightError, match="check_operator_authorize_canonical_use"):
        check_operator_authorize_canonical_use(repo_root=tmp_path, strict=True)
