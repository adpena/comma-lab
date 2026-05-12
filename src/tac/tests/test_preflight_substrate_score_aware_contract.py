from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_score_aware_losses_use_canonical_scorer_contract,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_score_aware_loss_contract_blocks_direct_scorer_calls(tmp_path):
    _write(
        tmp_path / "src/tac/substrates/bad/score_aware_loss.py",
        """
class BadScoreAwareLoss:
    def forward(self):
        self.seg_scorer(x)
        self.pose_scorer(y)
""",
    )

    violations = check_substrate_score_aware_losses_use_canonical_scorer_contract(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 3
    assert any("missing canonical score_pair_components" in item for item in violations)
    assert any(":4:" in item and "direct self.seg_scorer" in item for item in violations)
    assert any(":5:" in item and "direct self.seg_scorer" in item for item in violations)
    with pytest.raises(PreflightError):
        check_substrate_score_aware_losses_use_canonical_scorer_contract(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )


def test_score_aware_loss_contract_accepts_canonical_helper(tmp_path):
    _write(
        tmp_path / "src/tac/substrates/good/score_aware_loss.py",
        """
from tac.substrates.score_aware_common import score_pair_components

class GoodScoreAwareLoss:
    def forward(self):
        return score_pair_components(
            seg_scorer=self.seg_scorer,
            pose_scorer=self.pose_scorer,
            rgb_0_rt=a,
            rgb_1_rt=b,
            gt_rgb_0=c,
            gt_rgb_1=d,
        )
""",
    )

    assert (
        check_substrate_score_aware_losses_use_canonical_scorer_contract(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )
        == []
    )
