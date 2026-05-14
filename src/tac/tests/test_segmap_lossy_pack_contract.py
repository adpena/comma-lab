# SPDX-License-Identifier: MIT
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import PreflightError, check_segmap_hm_sa_lossy_pack_contract


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip("\n"))


def test_hm_sa_lossy_contract_preflight_rejects_lossless_tol(tmp_path: Path) -> None:
    _write(
        tmp_path / "scripts" / "remote_lane_sa_segmap_clone.sh",
        """
        #!/bin/bash
        set -euo pipefail
        PAYLOAD="$LOG_DIR/segmap_weights.tar.xz"
        python -c "from tac.block_fp_codec import verify_roundtrip; verify_roundtrip(state, '$PAYLOAD', tol=1e-6)"
        python experiments/contest_auth_eval.py --device cuda
        """,
    )

    violations = check_segmap_hm_sa_lossy_pack_contract(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert any("tol=1e-6" in v for v in violations)
    with pytest.raises(PreflightError):
        check_segmap_hm_sa_lossy_pack_contract(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )


def test_hm_sa_lossy_contract_preflight_requires_exact_eval_gate(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "scripts" / "remote_lane_hm_s_segmap_homography.sh",
        """
        #!/bin/bash
        set -euo pipefail
        PAYLOAD="$LOG_DIR/segmap_weights.tar.xz"
        python -c "from tac.block_fp_codec import SEGMAP_LOSSY_ROUNDTRIP_MSE_TOL, segmap_lossy_contract_metadata, verify_roundtrip; contract = segmap_lossy_contract_metadata(SEGMAP_LOSSY_ROUNDTRIP_MSE_TOL); verify_roundtrip(state, '$PAYLOAD', tol=SEGMAP_LOSSY_ROUNDTRIP_MSE_TOL, lossy_contract=contract); print('segmap_pack_roundtrip.json'); print('archive_level_exact_eval_required')"
        """,
    )

    violations = check_segmap_hm_sa_lossy_pack_contract(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert any("contest_auth_eval.py" in v for v in violations)
    assert any("--device cuda" in v for v in violations)


def test_hm_sa_lossy_contract_preflight_accepts_current_contract(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "scripts" / "remote_lane_sa_segmap_clone.sh",
        """
        #!/bin/bash
        set -euo pipefail
        PAYLOAD="$LOG_DIR/segmap_weights.tar.xz"
        python -c "from tac.block_fp_codec import SEGMAP_LOSSY_ROUNDTRIP_MSE_TOL, segmap_lossy_contract_metadata, verify_roundtrip; contract = segmap_lossy_contract_metadata(SEGMAP_LOSSY_ROUNDTRIP_MSE_TOL); verify_roundtrip(state, '$PAYLOAD', tol=SEGMAP_LOSSY_ROUNDTRIP_MSE_TOL, lossy_contract=contract); print('segmap_pack_roundtrip.json'); print('archive_level_exact_eval_required')"
        python experiments/contest_auth_eval.py --device cuda
        """,
    )

    assert (
        check_segmap_hm_sa_lossy_pack_contract(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )
        == []
    )
