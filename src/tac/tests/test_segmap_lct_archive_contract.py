# SPDX-License-Identifier: MIT
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import PreflightError, check_segmap_lct_archive_contract


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip("\n"))


def test_segmap_lct_contract_rejects_missing_archive_member(tmp_path: Path) -> None:
    _write(
        tmp_path / "scripts" / "remote_lane_sa_segmap_clone.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        SEGMAP_ENABLE_LCT="${SEGMAP_ENABLE_LCT:-0}"
        SEGMAP_CLASS_TARGETS_FILENAME="${SEGMAP_CLASS_TARGETS_FILENAME:-class_targets.fp16}"
        if [ "$SEGMAP_ENABLE_LCT" = "1" ]; then
            TRAIN_LCT_ARGS=(--learnable-class-targets --class-targets-filename "$SEGMAP_CLASS_TARGETS_FILENAME")
        fi
        if [ "$SEGMAP_ENABLE_LCT" = "1" ]; then
            echo "SEGMAP_CLASS_TARGETS_FILENAME=$SEGMAP_CLASS_TARGETS_FILENAME" >> "$INFLATE_CONFIG"
        fi
        """,
    )

    violations = check_segmap_lct_archive_contract(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert any("LCT_PAYLOAD" in violation for violation in violations)
    assert any("members.append" in violation for violation in violations)
    with pytest.raises(PreflightError):
        check_segmap_lct_archive_contract(repo_root=tmp_path, strict=True, verbose=False)


def test_segmap_lct_contract_accepts_producer_lane(tmp_path: Path) -> None:
    _write(
        tmp_path / "scripts" / "remote_lane_sa_segmap_clone.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        SEGMAP_ENABLE_LCT="${SEGMAP_ENABLE_LCT:-0}"
        SEGMAP_CLASS_TARGETS_FILENAME="${SEGMAP_CLASS_TARGETS_FILENAME:-class_targets.fp16}"
        if [ "$SEGMAP_ENABLE_LCT" = "1" ]; then
            TRAIN_LCT_ARGS=(--learnable-class-targets --class-targets-filename "$SEGMAP_CLASS_TARGETS_FILENAME")
        fi
        if [ "$SEGMAP_ENABLE_LCT" = "1" ]; then
            LCT_PAYLOAD="$LOG_DIR/train/$SEGMAP_CLASS_TARGETS_FILENAME"
            cp "$LCT_PAYLOAD" "$LOG_DIR/archive_src/$SEGMAP_CLASS_TARGETS_FILENAME"
        fi
        python - <<'PY'
        members = []
        if '$SEGMAP_ENABLE_LCT' == '1':
            members.append('$SEGMAP_CLASS_TARGETS_FILENAME')
        PY
        if [ "$SEGMAP_ENABLE_LCT" = "1" ]; then
            echo "SEGMAP_CLASS_TARGETS_FILENAME=$SEGMAP_CLASS_TARGETS_FILENAME" >> "$INFLATE_CONFIG"
        fi
        """,
    )

    assert check_segmap_lct_archive_contract(repo_root=tmp_path, strict=True, verbose=False) == []


def test_segmap_lct_contract_rejects_pass_through_without_config(tmp_path: Path) -> None:
    _write(
        tmp_path / "scripts" / "remote_lane_sh_shannon_arithmetic.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        SEGMAP_CLASS_TARGETS_FILENAME="${SEGMAP_CLASS_TARGETS_FILENAME:-class_targets.fp16}"
        UPSTREAM_CLASS_TARGETS="$EXTRACT_DIR/$SEGMAP_CLASS_TARGETS_FILENAME"
        cp "$UPSTREAM_CLASS_TARGETS" "$LOG_DIR/archive_src/$SEGMAP_CLASS_TARGETS_FILENAME"
        """,
    )

    violations = check_segmap_lct_archive_contract(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert any("SEGMAP_CLASS_TARGETS_FILENAME=$SEGMAP_CLASS_TARGETS_FILENAME" in v for v in violations)


def test_segmap_lct_contract_accepts_pass_through_repack(tmp_path: Path) -> None:
    _write(
        tmp_path / "scripts" / "remote_lane_sh_shannon_arithmetic.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        SEGMAP_CLASS_TARGETS_FILENAME="${SEGMAP_CLASS_TARGETS_FILENAME:-class_targets.fp16}"
        UPSTREAM_CLASS_TARGETS="$EXTRACT_DIR/$SEGMAP_CLASS_TARGETS_FILENAME"
        cp "$UPSTREAM_CLASS_TARGETS" "$LOG_DIR/archive_src/$SEGMAP_CLASS_TARGETS_FILENAME"
        echo "SEGMAP_CLASS_TARGETS_FILENAME=$SEGMAP_CLASS_TARGETS_FILENAME" >> "$INFLATE_CONFIG"
        """,
    )

    assert check_segmap_lct_archive_contract(repo_root=tmp_path, strict=True, verbose=False) == []
