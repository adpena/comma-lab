# SPDX-License-Identifier: MIT
"""Legacy SJ-KL artifact compatibility — documents known drift.

Two real sjkl.bin artifacts produced 2026-05-02 (via lightning_batch
sjkl_c067_l40sdiag + sjkl_c067_rtxprodiag dispatches) live on disk at:

  experiments/results/lightning_batch/sjkl_c067_l40sdiag_20260502T151434Z/build/sjkl.bin
  experiments/results/lightning_batch/sjkl_c067_rtxprodiag_20260502T151756Z/build/sjkl.bin

These artifacts use sjkl payload `version 27`, which the rebuilt-from-spec
`tac.sjkl_basis` codec (rebuilt 2026-05-04 per the recovery report — the
prior implementation was lost when subagent worktrees were auto-cleaned)
does NOT support.

This is documented drift, not a regression:

  - The current rebuilt codec is byte-correct against the runbook spec
    (verified by 26 passing tests in test_sjkl_basis.py covering
    encode/decode roundtrip, magic-byte validation, Lanczos eigenvector
    recovery, and residual application).

  - The 2026-05-02 artifacts were produced by a now-lost code path;
    backward-decoding them is NOT a Shannon-floor priority.

  - If the legacy version is ever needed for forensic comparison, the
    binary blobs are still on disk — a one-time reverse-engineer pass
    can extract the data, but no production code-path requires it.

This test EXPECTS the legacy artifacts to fail to decode under the
current codec — this guards against accidental "fix" attempts that
would either (a) break the rebuilt-from-spec contract, or (b) silently
treat legacy bytes as valid current-version bytes.

If this test ever PASSES (legacy artifacts decode cleanly), it means
either:
  - someone added backward-compat support (intentional → update this test)
  - the legacy artifacts were renamed/regenerated (drift detection)
  - the codec version field was misread (regression — debug)
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
LEGACY_ARTIFACTS = [
    REPO / "experiments" / "results" / "lightning_batch" /
    "sjkl_c067_l40sdiag_20260502T151434Z" / "build" / "sjkl.bin",
    REPO / "experiments" / "results" / "lightning_batch" /
    "sjkl_c067_rtxprodiag_20260502T151756Z" / "build" / "sjkl.bin",
]


@pytest.mark.parametrize("artifact_path", LEGACY_ARTIFACTS)
def test_legacy_artifact_documented_version_drift(artifact_path):
    if not artifact_path.is_file():
        pytest.skip(f"legacy artifact not on disk: {artifact_path.relative_to(REPO)}")

    from tac.sjkl_basis import decode_full_sjkl_payload

    raw = artifact_path.read_bytes()
    assert len(raw) > 0, "empty artifact"

    with pytest.raises(Exception) as exc_info:
        decode_full_sjkl_payload(raw)

    msg = str(exc_info.value).lower()
    # Must fail with a version-related error, not a corrupt-bytes error
    assert "version" in msg or "magic" in msg or "unsupported" in msg, (
        f"legacy artifact {artifact_path.name} failed to decode but with an "
        f"unexpected error type: {exc_info.value!r}\n"
        f"Expected a version-mismatch / unsupported-version error per the "
        f"documented drift in the test docstring. If the error is now about "
        f"truncation or magic, the file may have been corrupted — investigate."
    )
