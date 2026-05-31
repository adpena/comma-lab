# SPDX-License-Identifier: MIT
"""Public replay intake must remain contract-first and fail-closed."""

from __future__ import annotations

import zipfile
from pathlib import Path

from experiments.preflight_public_replay_intake import build_preflight
from tac.optimization.archive_bound_candidate_contract import (
    ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA,
)


def test_public_replay_preflight_emits_archive_bound_contract_without_exact_authority(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("0.bin", b"public-frontier-intake-smoke")

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    inflate_sh = runtime_dir / "inflate.sh"
    inflate_sh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "exit 0\n",
        encoding="utf-8",
    )
    inflate_sh.chmod(0o755)

    payload = build_preflight(archive, inflate_sh, upstream_dir=tmp_path / "upstream")

    assert payload["public_replay_preclaim_ready"] is True
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    contract = payload["archive_bound_candidate_contract"]
    assert contract["schema"] == ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
    assert contract["archive_bound_candidate_ready_for_exact_handoff"] is False
    assert contract["archive_file_custody"]["custody_complete"] is True
    assert contract["runtime_adapter_ready"] is True
    assert contract["receiver_contract_satisfied"] is False
    assert "public_frontier" in contract["archive_substrate_tags"]
    assert "archive_bound_receiver_runtime_proof_missing" in contract["blockers"]
    assert (
        "proxy_or_advisory_signal_masquerades_as_score_authority"
        in contract["canonical_anti_pattern_ids"]
    )
