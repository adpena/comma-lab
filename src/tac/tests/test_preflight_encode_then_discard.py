from __future__ import annotations

import inspect
import textwrap
from pathlib import Path

import pytest

from tac import preflight
from tac.preflight import MetaBugViolation


def _write_remote_lane(root: Path, name: str, body: str) -> Path:
    path = root / "scripts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


def test_encode_then_discard_strict_blocks_anchor_copy_after_payload_build(tmp_path: Path) -> None:
    root = tmp_path
    _write_remote_lane(
        root,
        "remote_lane_bad_noop.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        ANCHOR_DIR="$PWD/anchor"
        ITER_DIR="$PWD/iter"
        # Stage 2: encode payload
        python experiments/build_payload.py --output "$ITER_DIR/renderer.bin.br"
        cp "$ANCHOR_DIR/masks.mkv" "$ITER_DIR/masks.mkv"
        """,
    )

    violations = preflight.check_remote_lane_scripts_use_computed_payloads(
        repo_root=root,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "encode-then-discard antipattern" in violations[0]
    with pytest.raises(MetaBugViolation, match="ENCODE-THEN-DISCARD"):
        preflight.check_remote_lane_scripts_use_computed_payloads(
            repo_root=root,
            strict=True,
            verbose=False,
        )


def test_encode_then_discard_allows_anchor_staging_before_payload_build(tmp_path: Path) -> None:
    root = tmp_path
    _write_remote_lane(
        root,
        "remote_lane_anchor_stage_then_build.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        ANCHOR_DIR="$PWD/anchor"
        ITER_DIR="$PWD/iter"
        cp "$ANCHOR_DIR/masks.mkv" "$ITER_DIR/masks.mkv"
        # Stage 2: encode payload
        python experiments/build_payload.py --output "$ITER_DIR/renderer.bin.br"
        """,
    )

    assert preflight.check_remote_lane_scripts_use_computed_payloads(
        repo_root=root,
        strict=True,
        verbose=False,
    ) == []


def test_encode_then_discard_allows_explicit_waiver(tmp_path: Path) -> None:
    root = tmp_path
    _write_remote_lane(
        root,
        "remote_lane_waived_anchor_copy.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        # UNIWARD-NO-OP-WAIVED: this copy is a forensic replay of anchor bytes
        ANCHOR_DIR="$PWD/anchor"
        ITER_DIR="$PWD/iter"
        # Stage 2: encode payload
        python experiments/build_payload.py --output "$ITER_DIR/renderer.bin.br"
        cp "$ANCHOR_DIR/archive.zip" "$ITER_DIR/archive.zip"
        """,
    )

    assert preflight.check_remote_lane_scripts_use_computed_payloads(
        repo_root=root,
        strict=True,
        verbose=False,
    ) == []


def test_encode_then_discard_strict_passes_on_live_codebase() -> None:
    assert preflight.check_remote_lane_scripts_use_computed_payloads(
        strict=True,
        verbose=False,
    ) == []


def test_encode_then_discard_is_wired_strict_in_preflight_all() -> None:
    src = inspect.getsource(preflight.preflight_all)

    assert "check_remote_lane_scripts_use_computed_payloads(strict=True" in src
