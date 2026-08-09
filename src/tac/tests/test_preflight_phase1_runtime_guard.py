# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    check_phase1_trainer_runtime_emits_contest_compliant_inflate,
)

TRAINER_REL = "experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py"
FX1_RUNTIME_REL = "src/tac/pr130_runtime/fx1_runtime_tree/inflate.sh"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_trainer(repo: Path, body: str) -> None:
    path = repo / TRAINER_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    _write_fx1_runtime(
        repo,
        (REPO_ROOT / FX1_RUNTIME_REL).read_text(encoding="utf-8"),
    )


def _write_fx1_runtime(repo: Path, text: str) -> None:
    path = repo / FX1_RUNTIME_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def test_check146_accepts_live_fx1_three_argument_adapter(tmp_path: Path) -> None:
    live = (REPO_ROOT / FX1_RUNTIME_REL).read_text(encoding="utf-8")
    _write_fx1_runtime(tmp_path, live)

    assert check_phase1_trainer_runtime_emits_contest_compliant_inflate(
        repo_root=tmp_path, strict=True, verbose=False
    ) == []


def test_check146_rejects_missing_protected_fx1_runtime(tmp_path: Path) -> None:
    with pytest.raises(MetaBugViolation, match="protected shared FX1 runtime is missing"):
        check_phase1_trainer_runtime_emits_contest_compliant_inflate(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_check146_does_not_accept_commented_fx1_adapter(tmp_path: Path) -> None:
    live = (REPO_ROOT / FX1_RUNTIME_REL).read_text(encoding="utf-8")
    commented = "\n".join(f"# {line}" for line in live.splitlines()) + "\n"
    _write_fx1_runtime(tmp_path, commented)

    with pytest.raises(
        MetaBugViolation,
        match="shared FX1 runtime does not adapt the evaluator three-argument call",
    ):
        check_phase1_trainer_runtime_emits_contest_compliant_inflate(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_check146_positive_control_rejects_fx1_legacy_passthrough(
    tmp_path: Path,
) -> None:
    live = (REPO_ROOT / FX1_RUNTIME_REL).read_text(encoding="utf-8")
    repaired_tail = '''if [ "$#" -ne 3 ]; then
    echo "usage: inflate.sh <archive-dir> <output-dir> <video-names-file>" >&2
    exit 64
fi

ARCHIVE_DIR=$1
OUTPUT_DIR=$2
VIDEO_NAMES_FILE=$3
mkdir -p -- "$OUTPUT_DIR"

cd -- "$SCRIPT_DIR"
while IFS= read -r video_name || [ -n "$video_name" ]; do
    [ -n "$video_name" ] || continue
    base=${video_name%.*}
    "$PYBIN" inflate.py "$ARCHIVE_DIR" "$base" "$OUTPUT_DIR/$base.raw"
done < "$VIDEO_NAMES_FILE"
'''
    legacy_tail = '''cd -- "$SCRIPT_DIR"
exec "$PYBIN" inflate.py "$@"
'''
    assert repaired_tail in live, "positive control no longer targets the live repair"
    _write_fx1_runtime(tmp_path, live.replace(repaired_tail, legacy_tail, 1))

    with pytest.raises(
        MetaBugViolation,
        match="shared FX1 runtime does not adapt the evaluator three-argument call",
    ):
        check_phase1_trainer_runtime_emits_contest_compliant_inflate(
            repo_root=tmp_path, strict=True, verbose=False
        )


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
