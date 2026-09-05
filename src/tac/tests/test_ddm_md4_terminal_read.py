"""Tests for ``experiments/ddm_md4_terminal_read.py``.

Two surfaces here are load-bearing and nothing else is: (a) the PRE-REGISTERED verdict
thresholds, because the word they emit is the arm's deliverable, and (b) the relative-L2
displacement used by the resume-boundary control, because a silent key drop or a silent
shape mismatch there would understate a discontinuity — the exact confound the control
exists to detect.  Every test below fails under the mutation its name describes; none of
them merely asserts a constant.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_module():
    path = REPO / "experiments" / "ddm_md4_terminal_read.py"
    spec = importlib.util.spec_from_file_location("ddm_md4_terminal_read", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MD4 = _load_module()


# ------------------------------------------------------------------ pre-registered thresholds
def test_verdict_word_is_data_anchored_at_and_above_the_pre_registered_bar() -> None:
    assert MD4._verdict_word(0.70) == "DATA-ANCHORED"
    assert MD4._verdict_word(0.8536) == "DATA-ANCHORED"
    assert MD4._verdict_word(1.0) == "DATA-ANCHORED"


def test_verdict_word_is_init_anchored_at_and_below_the_pre_registered_bar() -> None:
    assert MD4._verdict_word(0.45) == "INIT-ANCHORED"
    assert MD4._verdict_word(0.12) == "INIT-ANCHORED"
    assert MD4._verdict_word(0.0) == "INIT-ANCHORED"


def test_verdict_word_is_indeterminate_strictly_between_the_two_bars() -> None:
    assert MD4._verdict_word(0.4501) == "INDETERMINATE"
    assert MD4._verdict_word(0.55) == "INDETERMINATE"
    assert MD4._verdict_word(0.6999) == "INDETERMINATE"


def test_thresholds_match_the_charter_and_are_not_silently_widened() -> None:
    # ddm_md3 charter: "Jaccard >= 0.70 -> data-anchored" / "<= 0.45 -> init-anchored".
    assert MD4.DATA_ANCHORED_AT_OR_ABOVE == 0.70
    assert MD4.INIT_ANCHORED_AT_OR_BELOW == 0.45


# ---------------------------------------------------------------------- displacement measure
def _state(values: dict[str, list[float]]) -> dict[str, object]:
    torch = pytest.importorskip("torch")
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in values.items()}


def test_relative_l2_is_zero_for_identical_states() -> None:
    a = _state({"w": [1.0, 2.0, 3.0]})
    assert MD4._relative_l2(a, a) == pytest.approx(0.0)


def test_relative_l2_is_normalised_by_the_second_argument() -> None:
    a = _state({"w": [2.0, 0.0]})
    b = _state({"w": [1.0, 0.0]})
    # |a-b| = 1, |b| = 1 -> 1.0 ; the reference is b, so swapping halves it.
    assert MD4._relative_l2(a, b) == pytest.approx(1.0)
    assert MD4._relative_l2(b, a) == pytest.approx(0.5)


def test_relative_l2_pools_across_every_shared_tensor_key() -> None:
    a = _state({"u": [3.0], "v": [4.0]})
    b = _state({"u": [0.0], "v": [0.0]})
    with pytest.raises(MD4.MD4Error):
        MD4._relative_l2(a, b)  # reference norm is zero
    c = _state({"u": [0.0], "v": [5.0]})
    # numerator sqrt(9 + 1) over reference norm 5
    assert MD4._relative_l2(a, c) == pytest.approx((10.0**0.5) / 5.0)


def test_relative_l2_refuses_a_key_set_mismatch_instead_of_silently_intersecting() -> None:
    a = _state({"u": [1.0], "v": [1.0]})
    b = _state({"u": [1.0]})
    with pytest.raises(MD4.MD4Error, match="different tensor key sets"):
        MD4._relative_l2(a, b)


def test_relative_l2_refuses_a_shape_mismatch() -> None:
    a = _state({"u": [1.0, 2.0]})
    b = _state({"u": [1.0]})
    with pytest.raises(MD4.MD4Error, match="shape mismatch"):
        MD4._relative_l2(a, b)


# ------------------------------------------------------------------------- ceiling arithmetic
def test_jaccard_ceiling_rederives_the_value_md3_reported_for_r10_live() -> None:
    # MEASURED step-0 quantities from ddm_md3's STEP0_POOL_OVERLAP.json (dali block).
    pool_cold, pool_new = 16553, 17151
    persistent_cold, intersection_cap = 11842, 10777
    assumed_new = int(pool_new * (persistent_cold / pool_cold))
    ceiling = intersection_cap / (persistent_cold + assumed_new - intersection_cap)
    assert round(ceiling, 4) == MD4.MD3_MEMO_CEILING


def test_missing_artifact_refuses_rather_than_returning_an_empty_receipt(tmp_path: Path) -> None:
    with pytest.raises(MD4.MD4Error, match="required artifact absent"):
        MD4._read_json(tmp_path / "not_here.json")


# ------------------------------------------------------------------------ within-pool null
def test_within_pool_null_reduces_to_md2_shared_pool_constant() -> None:
    # MEASURED md2 inputs: ng5 persistent 11,019 sites, cold persistent 11,842, one shared
    # step-0 pool of 16,553.  md2 published J_null = 0.5263033946906265 for that pair.
    null = MD4._within_pool_null_jaccard(11_019, 11_842, 16_553, 16_553, 16_553)
    assert null["jaccard"] == pytest.approx(0.5263033946906265, rel=0, abs=1e-15)


def test_within_pool_null_falls_when_the_pools_share_fewer_sites() -> None:
    shared = MD4._within_pool_null_jaccard(11_000, 11_842, 17_151, 16_553, 16_553)
    partial = MD4._within_pool_null_jaccard(11_000, 11_842, 17_151, 16_553, 13_012)
    assert partial["jaccard"] < shared["jaccard"]


def test_within_pool_null_is_far_above_an_all_sites_null() -> None:
    within = MD4._within_pool_null_jaccard(11_000, 11_842, 17_151, 16_553, 13_012)
    all_sites = MD4._within_pool_null_jaccard(11_000, 11_842, 6_291_456, 6_291_456, 6_291_456)
    assert within["jaccard"] > 100 * all_sites["jaccard"]


def test_within_pool_null_refuses_an_empty_pool() -> None:
    with pytest.raises(MD4.MD4Error, match="pool is empty"):
        MD4._within_pool_null_jaccard(10, 10, 0, 16_553, 0)


def test_within_pool_null_refuses_an_impossible_pool_intersection() -> None:
    with pytest.raises(MD4.MD4Error, match="not within"):
        MD4._within_pool_null_jaccard(10, 10, 100, 200, 150)
    with pytest.raises(MD4.MD4Error, match="not within"):
        MD4._within_pool_null_jaccard(10, 10, 100, 200, -1)
