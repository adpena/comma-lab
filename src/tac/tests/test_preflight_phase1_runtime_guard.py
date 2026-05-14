# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    check_phase1_trainer_runtime_emits_contest_compliant_inflate,
)


TRAINER_REL = "experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py"


def _write_trainer(repo: Path, body: str) -> None:
    path = repo / TRAINER_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_check146_ignores_docstring_forbidden_token_mentions(tmp_path: Path) -> None:
    _write_trainer(
        tmp_path,
        '''
def _write_runtime():
    """Document the prohibition: no PoseNet, SegNet, or rgb_to_yuv6 at inflate."""
    # Comments may also mention PoseNet while explaining the guard.
    inflate_sh = (
        "#!/usr/bin/env bash\\n"
        "set -euo pipefail\\n"
        "DATA_DIR=\\"$1\\"\\n"
        "OUTPUT_DIR=\\"$2\\"\\n"
        "FILE_LIST=\\"$3\\"\\n"
        "exec \\"$HERE/inflate.py\\" \\"$DATA_DIR\\" \\"$OUTPUT_DIR\\" \\"$FILE_LIST\\"\\n"
    )
    inflate_py = (
        "from pathlib import Path\\n"
        "def main(file_list):\\n"
        "    archive_dir = Path('.')\\n"
        "    member = archive_dir / 'x'\\n"
        "    data = member.read_bytes()\\n"
        "    for line in file_list.read_text().splitlines():\\n"
        "        pass\\n"
    )
''',
    )

    assert (
        check_phase1_trainer_runtime_emits_contest_compliant_inflate(
            repo_root=tmp_path, strict=True, verbose=False
        )
        == []
    )


def test_check146_rejects_scorer_import_in_emitted_template(tmp_path: Path) -> None:
    _write_trainer(
        tmp_path,
        '''
def _write_runtime():
    inflate_sh = (
        "#!/usr/bin/env bash\\n"
        "set -euo pipefail\\n"
        "DATA_DIR=\\"$1\\"\\n"
        "OUTPUT_DIR=\\"$2\\"\\n"
        "FILE_LIST=\\"$3\\"\\n"
        "exec \\"$HERE/inflate.py\\" \\"$DATA_DIR\\" \\"$OUTPUT_DIR\\" \\"$FILE_LIST\\"\\n"
    )
    inflate_py = (
        "from upstream.modules import PoseNet\\n"
        "for line in file_list.read_text().splitlines():\\n"
        "    pass\\n"
    )
''',
    )

    with pytest.raises(MetaBugViolation, match="FORBIDDEN_INFLATE_TOKEN"):
        check_phase1_trainer_runtime_emits_contest_compliant_inflate(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_check146_rejects_legacy_passthrough_runtime(tmp_path: Path) -> None:
    _write_trainer(
        tmp_path,
        '''
def _write_runtime():
    inflate_sh = (
        "#!/bin/bash\\n"
        "set -euo pipefail\\n"
        "exec uv run --with compressai==1.2.8 \\"$HERE/inflate.py\\" \\"$@\\"\\n"
    )
    inflate_py = (
        "for line in file_list.read_text().splitlines():\\n"
        "    pass\\n"
    )
''',
    )

    violations = check_phase1_trainer_runtime_emits_contest_compliant_inflate(
        repo_root=tmp_path, strict=False, verbose=False
    )

    assert any("missing one of $1/$2/$3" in item for item in violations)
    assert any('"$@"' in item and "passthrough" in item for item in violations)


def test_check146_rejects_runtime_local_archive_zip_fallback(tmp_path: Path) -> None:
    _write_trainer(
        tmp_path,
        '''
def _write_runtime():
    inflate_sh = (
        "#!/usr/bin/env bash\\n"
        "set -euo pipefail\\n"
        "DATA_DIR=\\"$1\\"\\n"
        "OUTPUT_DIR=\\"$2\\"\\n"
        "FILE_LIST=\\"$3\\"\\n"
        "exec \\"$HERE/inflate.py\\" \\"$DATA_DIR\\" \\"$OUTPUT_DIR\\" \\"$FILE_LIST\\"\\n"
    )
    inflate_py = (
        "from pathlib import Path\\n"
        "HERE = Path(__file__).resolve().parent\\n"
        "def main(archive_dir, output_dir, file_list):\\n"
        "    archive_zip = HERE / 'archive.zip'\\n"
        "    for line in file_list.read_text().splitlines():\\n"
        "        pass\\n"
    )
''',
    )

    with pytest.raises(MetaBugViolation, match="RUNTIME_LOCAL_ARCHIVE_FALLBACK"):
        check_phase1_trainer_runtime_emits_contest_compliant_inflate(
            repo_root=tmp_path, strict=True, verbose=False
        )
