"""Tests for preflight_bootstrap_safety — the static gate that prevents
re-living the LANE-B silent-failure cascade (2026-04-26, 6.5h + ~$2 wasted).
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.preflight import PreflightError, preflight_bootstrap_safety  # noqa: E402


def _write(tmp_path: Path, name: str, body: str) -> None:
    (tmp_path / name).write_text(textwrap.dedent(body))


def test_real_scripts_pass():
    """The repo's actual bootstrap scripts must all pass after today's fixes."""
    violations = preflight_bootstrap_safety(strict=False, verbose=False)
    assert violations == [], f"real scripts failed preflight: {violations}"


def test_canonical_bootstrap_passes(tmp_path: Path):
    """A correct bootstrap with `set -euo pipefail` and no shell `zip`
    must be silent."""
    _write(tmp_path, "good_bootstrap.sh", """\
        #!/bin/bash
        set -euo pipefail
        python -c "import zipfile; zipfile.ZipFile('a.zip', 'w').close()"
    """)
    violations = preflight_bootstrap_safety(scripts_dir=tmp_path, strict=False, verbose=False)
    assert violations == []


def test_missing_set_e_caught(tmp_path: Path):
    """The LANE-B kill-chain root: `set -uo pipefail` (no -e)."""
    _write(tmp_path, "bad_bootstrap.sh", """\
        #!/bin/bash
        set -uo pipefail
        zip archive.zip foo.bin
    """)
    violations = preflight_bootstrap_safety(scripts_dir=tmp_path, strict=False, verbose=False)
    assert any("missing `set -e`" in v for v in violations), (
        f"expected `set -e` violation, got: {violations}"
    )


def test_shell_zip_caught(tmp_path: Path):
    """The LANE-B kill-chain trigger: invokes `zip` shell binary."""
    _write(tmp_path, "bad_bootstrap.sh", """\
        #!/bin/bash
        set -euo pipefail
        zip archive.zip renderer.bin masks.mkv
    """)
    violations = preflight_bootstrap_safety(scripts_dir=tmp_path, strict=False, verbose=False)
    assert any("invokes `zip` shell binary" in v for v in violations), (
        f"expected zip violation, got: {violations}"
    )


def test_zipfile_not_flagged(tmp_path: Path):
    """`python ... zipfile` must NOT match the `zip` regex."""
    _write(tmp_path, "good_bootstrap.sh", """\
        #!/bin/bash
        set -euo pipefail
        python -c "import zipfile; zipfile.ZipFile('a.zip', 'w').close()"
        ls *.zip
    """)
    violations = preflight_bootstrap_safety(scripts_dir=tmp_path, strict=False, verbose=False)
    assert violations == [], f"false positive: {violations}"


def test_unzip_gzip_not_flagged(tmp_path: Path):
    """`unzip` and `gzip` must NOT match — only bare `zip`."""
    _write(tmp_path, "good_bootstrap.sh", """\
        #!/bin/bash
        set -euo pipefail
        unzip archive.zip
        gzip output.txt
    """)
    violations = preflight_bootstrap_safety(scripts_dir=tmp_path, strict=False, verbose=False)
    assert violations == []


def test_zip_in_comment_not_flagged(tmp_path: Path):
    """A literal mention of `zip` inside a comment must NOT trigger."""
    _write(tmp_path, "good_bootstrap.sh", """\
        #!/bin/bash
        # Note: do not use shell zip — use python zipfile instead
        set -euo pipefail
        echo done
    """)
    violations = preflight_bootstrap_safety(scripts_dir=tmp_path, strict=False, verbose=False)
    assert violations == []


def test_strict_raises(tmp_path: Path):
    _write(tmp_path, "bad_bootstrap.sh", "set -uo pipefail\nzip a.zip foo\n")
    with pytest.raises(PreflightError, match="LANE-B kill chain"):
        preflight_bootstrap_safety(scripts_dir=tmp_path, strict=True, verbose=False)


def test_set_eu_combined_form(tmp_path: Path):
    """`set -eu` (combined) should still match — -e is present."""
    _write(tmp_path, "good_bootstrap.sh", """\
        #!/bin/bash
        set -eu
    """)
    violations = preflight_bootstrap_safety(scripts_dir=tmp_path, strict=False, verbose=False)
    assert violations == []


def test_set_e_alone(tmp_path: Path):
    """Plain `set -e` should pass."""
    _write(tmp_path, "good_bootstrap.sh", "#!/bin/bash\nset -e\n")
    violations = preflight_bootstrap_safety(scripts_dir=tmp_path, strict=False, verbose=False)
    assert violations == []


def test_no_scripts_dir(tmp_path: Path):
    """Empty scripts dir is OK (no bootstraps to check)."""
    violations = preflight_bootstrap_safety(scripts_dir=tmp_path, strict=False, verbose=False)
    assert violations == []


def test_missing_scripts_dir(tmp_path: Path):
    """Pointing at a non-existent dir must NOT crash."""
    violations = preflight_bootstrap_safety(
        scripts_dir=tmp_path / "nope", strict=False, verbose=False
    )
    # Should return a single "not found" violation, not raise.
    assert len(violations) == 1 and "not found" in violations[0]
