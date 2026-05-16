# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REMOTE_DRIVER = REPO / "scripts/remote_lane_substrate_time_traveler_l5_z6.sh"


def test_z6_remote_driver_defaults_smoke_to_three_epochs_and_full_to_300() -> None:
    """The first-anchor smoke path must not inherit the full-run epoch default."""

    text = REMOTE_DRIVER.read_text(encoding="utf-8")

    smoke_idx = text.index('SMOKE_ONLY="${SMOKE_ONLY:-1}"')
    epochs_idx = text.index('Z6_EPOCHS="${Z6_EPOCHS:-}"')
    assert smoke_idx < epochs_idx
    assert 'if [ "$SMOKE_ONLY" = "1" ]; then\n        Z6_EPOCHS="3"' in text
    assert 'else\n        Z6_EPOCHS="300"\n    fi' in text
    assert 'Z6_EPOCHS="${Z6_EPOCHS:-300}"' not in text
