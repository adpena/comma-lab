# SPDX-License-Identifier: MIT
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
    assert any("missing AST call to a canonical scorer entry point" in item for item in violations)
    assert any(":4:" in item and "direct seg_scorer scorer forward" in item for item in violations)
    assert any(":5:" in item and "direct pose_scorer scorer forward" in item for item in violations)
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


def test_score_aware_loss_contract_ignores_comment_and_string_bypass(tmp_path):
    _write(
        tmp_path / "src/tac/substrates/bypass/score_aware_loss.py",
        """
class BypassScoreAwareLoss:
    def forward(self):
        # score_pair_components(
        note = "score_pair_components("
        return 0
""",
    )

    violations = check_substrate_score_aware_losses_use_canonical_scorer_contract(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "missing AST call" in violations[0]


def test_score_aware_loss_contract_blocks_alias_direct_scorer_call(tmp_path):
    _write(
        tmp_path / "src/tac/substrates/alias/score_aware_loss.py",
        """
from tac.substrates.score_aware_common import score_pair_components

class AliasScoreAwareLoss:
    def forward(self):
        scorer = self.seg_scorer
        canonical = score_pair_components(
            seg_scorer=self.seg_scorer,
            pose_scorer=self.pose_scorer,
            rgb_0_rt=a,
            rgb_1_rt=b,
            gt_rgb_0=c,
            gt_rgb_1=d,
        )
        return scorer(x), canonical
""",
    )

    violations = check_substrate_score_aware_losses_use_canonical_scorer_contract(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "seg_scorer alias `scorer`" in violations[0]


# ── canonical entry-point set (ddm_rd1g, 2026-08-16) ──────────────────────
# The gate previously accepted only the bare `score_pair_components`, so it
# reported 55 "missing canonical call" violations against src/tac/substrates/
# of which 51 were substrates calling the canonical F3 dispatcher. One stale
# token, 51 files. These tests pin BOTH halves of the correction: the two
# canonical sisters are accepted, AND the widening did not open a hole.


@pytest.mark.parametrize(
    "entry_point",
    ["score_pair_components", "score_pair_components_with_cache", "score_pair_components_dispatch"],
)
def test_score_aware_loss_contract_accepts_every_canonical_entry_point(
    tmp_path, entry_point
):
    """All three canonical entry points satisfy the contract.

    All three are defined in tac.substrates.score_aware_common and all three
    call _require_preprocess on both scorers before any forward, so accepting
    the sisters admits no path the bare name did not already admit.
    """
    _write(
        tmp_path / f"src/tac/substrates/{entry_point}/score_aware_loss.py",
        f"""
from tac.substrates.score_aware_common import {entry_point}

class CanonicalScoreAwareLoss:
    def forward(self):
        return {entry_point}(
            seg_scorer=self.seg_scorer,
            pose_scorer=self.pose_scorer,
            rgb_0_rt=a,
            rgb_1_rt=b,
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


def test_score_aware_loss_contract_accepts_module_style_canonical_import(tmp_path):
    """`import ...score_aware_common` + attribute call is also canonical."""
    _write(
        tmp_path / "src/tac/substrates/modstyle/score_aware_loss.py",
        """
import tac.substrates.score_aware_common as sac

class ModStyleScoreAwareLoss:
    def forward(self):
        return sac.score_pair_components_dispatch(
            seg_scorer=self.seg_scorer,
            pose_scorer=self.pose_scorer,
            rgb_0_rt=a,
            rgb_1_rt=b,
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


def test_score_aware_loss_contract_rejects_locally_defined_lookalike(tmp_path):
    """A same-named LOCAL function must not satisfy the contract.

    This is the hole the name-set widening would have opened, and that the
    paired import requirement closes. The pre-widening gate had this hole
    already for the bare name; the corrected gate has it for none of the three.
    """
    _write(
        tmp_path / "src/tac/substrates/lookalike/score_aware_loss.py",
        """
class LookalikeScoreAwareLoss:
    def score_pair_components_dispatch(self, **kwargs):
        return 0.0, 0.0

    def forward(self):
        return score_pair_components_dispatch(
            seg_scorer=self.seg_scorer,
            pose_scorer=self.pose_scorer,
        )
""",
    )

    violations = check_substrate_score_aware_losses_use_canonical_scorer_contract(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "missing AST call to a canonical scorer entry point" in violations[0]


def test_score_aware_loss_contract_rejects_constant_only_import(tmp_path):
    """Importing a CONSTANT from the canonical module is not delegation.

    This is the real shape of the remaining live violators (e.g. atw_codec_v2
    imports only CONTEST_POSE_SQRT_WEIGHT): the module is imported, but no
    scorer entry point is.
    """
    _write(
        tmp_path / "src/tac/substrates/constonly/score_aware_loss.py",
        """
from tac.substrates.score_aware_common import CONTEST_POSE_SQRT_WEIGHT

class ConstOnlyScoreAwareLoss:
    def forward(self):
        return CONTEST_POSE_SQRT_WEIGHT
""",
    )

    violations = check_substrate_score_aware_losses_use_canonical_scorer_contract(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "missing AST call to a canonical scorer entry point" in violations[0]


def test_score_aware_loss_contract_accepts_relative_canonical_import(tmp_path):
    """`from .score_aware_common import X` is the same canonical import.

    Own round-1 review catch (ddm_rd1g): a relative ImportFrom sets
    ``node.module`` to the bare ``score_aware_common`` with ``level=1``, so a
    dotted-prefix match would have manufactured a fresh false positive of
    exactly the class this correction removes.
    """
    _write(
        tmp_path / "src/tac/substrates/relimport/score_aware_loss.py",
        """
from .score_aware_common import score_pair_components_dispatch

class RelImportScoreAwareLoss:
    def forward(self):
        return score_pair_components_dispatch(
            seg_scorer=self.seg_scorer,
            pose_scorer=self.pose_scorer,
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
