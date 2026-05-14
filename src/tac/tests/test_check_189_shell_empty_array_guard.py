# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    check_shell_empty_arrays_guarded_under_set_u,
)


def _write_script(repo: Path, name: str, body: str) -> Path:
    scripts = repo / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    path = scripts / name
    path.write_text(body)
    return path


def test_flags_unguarded_empty_array_under_nounset(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_script(
        repo,
        "operator_authorize_substrate_bad_modal_a100_dispatch.sh",
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "SMOKE_ARGS=()\n"
        "python tools/run.py \"${SMOKE_ARGS[@]}\" \"$@\"\n",
    )

    out = check_shell_empty_arrays_guarded_under_set_u(
        repo_root=repo, strict=False, verbose=False
    )

    assert len(out) == 1
    assert "operator_authorize_substrate_bad_modal_a100_dispatch.sh:4" in out[0]
    assert "unguarded empty-array expansion" in out[0]


def test_accepts_guarded_empty_array_under_nounset(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_script(
        repo,
        "operator_authorize_substrate_good_modal_a100_dispatch.sh",
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "SMOKE_ARGS=()\n"
        "python tools/run.py ${SMOKE_ARGS[@]+\"${SMOKE_ARGS[@]}\"} \"$@\"\n",
    )

    assert (
        check_shell_empty_arrays_guarded_under_set_u(
            repo_root=repo, strict=False, verbose=False
        )
        == []
    )


def test_waiver_allows_proven_non_empty_array(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_script(
        repo,
        "operator_authorize_substrate_waived_modal_a100_dispatch.sh",
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "REQUIRED_ARGS=(--recipe foo)\n"
        "python tools/run.py \"${REQUIRED_ARGS[@]}\" \"$@\" "
        "# SHELL_EMPTY_ARRAY_OK:required literal args\n",
    )

    assert (
        check_shell_empty_arrays_guarded_under_set_u(
            repo_root=repo, strict=False, verbose=False
        )
        == []
    )


def test_skips_shell_without_nounset(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_script(
        repo,
        "operator_authorize_substrate_legacy_modal_a100_dispatch.sh",
        "#!/bin/bash\n"
        "set -eo pipefail\n"
        "ARGS=()\n"
        "python tools/run.py \"${ARGS[@]}\" \"$@\"\n",
    )

    assert (
        check_shell_empty_arrays_guarded_under_set_u(
            repo_root=repo, strict=False, verbose=False
        )
        == []
    )


def test_strict_raises(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_script(
        repo,
        "operator_authorize_substrate_bad_modal_a100_dispatch.sh",
        "#!/bin/bash\n"
        "set -o nounset\n"
        "ARGS=()\n"
        "python tools/run.py \"${ARGS[@]}\"\n",
    )

    with pytest.raises(MetaBugViolation):
        check_shell_empty_arrays_guarded_under_set_u(
            repo_root=repo, strict=True, verbose=False
        )


def test_live_repo_clean() -> None:
    assert (
        check_shell_empty_arrays_guarded_under_set_u(strict=False, verbose=False)
        == []
    )
