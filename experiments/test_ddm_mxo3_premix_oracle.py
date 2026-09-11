from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_mxo3_premix_oracle as oracle

Q15 = 32768


def _row(probability: np.ndarray, hit_class: np.ndarray) -> np.ndarray:
    """A coded row whose argmax entry carries exactly the given probability."""
    n = len(probability)
    row = np.empty((n, oracle.K), dtype=np.uint32)
    hit_mass = np.rint(probability * oracle.TOTAL).astype(np.int64)
    hit_mass = np.clip(hit_mass, 4, oracle.TOTAL - 4 * oracle.K)
    share = (oracle.TOTAL - hit_mass) // (oracle.K - 1)
    row[:] = share[:, None]
    row[np.arange(n), hit_class] = hit_mass
    row[:, 0] += (oracle.TOTAL - row.sum(axis=1, dtype=np.int64)).astype(np.uint32) * (
        hit_class != 0
    )
    row[:, 1] += (oracle.TOTAL - row.sum(axis=1, dtype=np.int64)).astype(np.uint32) * (
        hit_class == 0
    )
    assert np.all(row.sum(axis=1, dtype=np.uint64) == oracle.TOTAL)
    return row


def _frame(rng: np.random.Generator, n: int, planted: bool) -> dict[str, np.ndarray]:
    """One synthetic surface frame.

    The shipped probability is deliberately blind: it is the same 0.9 everywhere.
    When `planted` is set, the realised hit really does depend on whether the 23
    pre-mix opinions agree, which is exactly the structure the oracle must find.
    """
    hit_class = rng.integers(0, oracle.K, size=n).astype(np.uint8)
    probability = np.full(n, 0.9)
    agreeing = rng.random(n) < 0.5
    sign = np.where(agreeing, 1.0, -1.0)
    mixer = np.full(n, 0.9)
    opinion = sign[:, None] * rng.uniform(0.4, 0.6, size=(n, 23))
    odds = mixer / (1.0 - mixer)
    family = (odds[:, None] * np.exp(opinion))
    family = family / (1.0 + family)
    truth = np.where(agreeing, 0.99, 0.5) if planted else probability
    hit = rng.random(n) < truth
    other = (hit_class.astype(np.int64) + 1 + rng.integers(0, oracle.K - 1, size=n)) % oracle.K
    symbols = np.where(hit, hit_class, other).astype(np.uint8)
    return {
        "family_q15": np.clip(np.rint(family * Q15), 1, Q15 - 1).astype(np.uint16),
        "mixer_q15": np.clip(np.rint(mixer * Q15), 1, Q15 - 1).astype(np.uint16),
        "frequencies": _row(probability, hit_class.astype(np.int64)),
        "hit_class": hit_class,
        "symbols": symbols,
    }


class _Frames:
    def __init__(self, frames: list[dict[str, np.ndarray]]) -> None:
        self.frames = frames

    def __call__(self, path):
        index = int(path.stem.split("_")[-1])
        return _Holder(self.frames[index])


class _Holder:
    def __init__(self, data: dict[str, np.ndarray]) -> None:
        self.data = data

    def __enter__(self):
        return self.data

    def __exit__(self, *_args):
        return False


def _run(monkeypatch, planted: bool, context_set: str) -> dict[str, object]:
    rng = np.random.default_rng(11)
    frames = [_frame(rng, 40_000, planted) for _ in range(4)]
    monkeypatch.setattr(oracle, "checked_npz", _Frames(frames))
    state = oracle.accumulate(Path("/unused"), len(frames), np.random.default_rng(3), context_set)
    return oracle.report(state, len(frames))


def test_cell_bits_is_zero_on_pure_cells_and_one_bit_on_even_cells() -> None:
    counts = np.array([100, 100, 40], dtype=np.int64)
    assert oracle.cell_bits(counts, np.array([100, 0, 20], dtype=np.int64)) == pytest.approx(40.0)
    assert oracle.cell_bits(np.array([8], np.int64), np.array([0], np.int64)) == 0.0


def test_binary_bits_prices_the_realised_side() -> None:
    probability = np.array([0.25, 0.25])
    bits = oracle.binary_bits(probability, np.array([True, False]))
    assert bits[0] == pytest.approx(2.0)
    assert bits[1] == pytest.approx(-np.log2(0.75))


def test_premix_context_is_flat_when_no_family_disagrees() -> None:
    mixer = np.full(64, 20000, dtype=np.uint16)
    family = np.repeat(mixer[:, None], 23, axis=1)
    axes = ("std", "mean", "peak", "agree")
    context = oracle.premix_context(family, mixer, axes)
    assert np.all(context == context[0])
    family[:, 3] = 32000
    assert np.any(oracle.premix_context(family, mixer, axes) != context)


@pytest.mark.parametrize("context_set", sorted(oracle.CONTEXT_SETS))
def test_oracle_finds_a_planted_premix_signal(monkeypatch, context_set) -> None:
    """Positive control: the instrument must register a known effect."""
    result = _run(monkeypatch, planted=True, context_set=context_set)
    assert result["premix_net_bytes"] > 0.05 * result["shipped_binary_bytes"]
    assert result["premix_net_bytes"] > 10 * result["premix_overfit_bytes"]


@pytest.mark.parametrize("context_set", sorted(oracle.CONTEXT_SETS))
def test_oracle_control_leaves_nothing_when_the_premix_is_noise(monkeypatch, context_set) -> None:
    """Negative control: with no signal, the net bound must collapse to ~0."""
    result = _run(monkeypatch, planted=False, context_set=context_set)
    assert abs(result["premix_net_bytes"]) < 0.01 * result["shipped_binary_bytes"]
    assert result["premix_gross_bytes"] > 0.0


def test_blocks_refit_per_stretch_and_widen_the_bound(monkeypatch) -> None:
    """Per-block cells must still register the signal and cost more overfit."""
    rng = np.random.default_rng(11)
    frames = [_frame(rng, 40_000, True) for _ in range(4)]
    monkeypatch.setattr(oracle, "checked_npz", _Frames(frames))
    one = oracle.report(
        oracle.accumulate(Path("/unused"), 4, np.random.default_rng(3), "summary", 1), 4
    )
    four = oracle.report(
        oracle.accumulate(Path("/unused"), 4, np.random.default_rng(3), "summary", 4), 4
    )
    assert four["cells"] == 4 * one["cells"]
    assert four["premix_overfit_bytes"] > one["premix_overfit_bytes"]
    assert four["premix_gross_bytes"] > one["premix_gross_bytes"]
    assert four["premix_net_bytes"] > 0.05 * four["shipped_binary_bytes"]


def test_kt_alpha_matches_the_shipped_corrector() -> None:
    """The held-out prior is the corrector's own, not a number of our choosing."""
    source = Path(
        "/Volumes/VertigoDataTier/pact/ddm_mxo2_low_rank_stacker_v3"
        "/runtime_copy/runtime/f26_corrector_native.c"
    )
    if not source.exists():
        pytest.skip("instrumented receiver copy is not mounted")
    assert f"#define KT_ALPHA {oracle.KT_ALPHA}" in source.read_text()


def test_holdout_registers_the_planted_signal_and_rejects_noise(monkeypatch) -> None:
    """The held-out number scores nothing by a table fitted on itself."""
    planted = _run(monkeypatch, planted=True, context_set="summary")
    assert planted["holdout_premix_bytes"] > 0.05 * planted["shipped_binary_bytes"]
    noise = _run(monkeypatch, planted=False, context_set="summary")
    assert abs(noise["holdout_premix_bytes"]) < 0.01 * noise["shipped_binary_bytes"]
    # Held out, extra cells cannot manufacture a gain the way in-sample cells can:
    # on noise they COST bytes, while the in-sample partition still shows a gain.
    rich = _run(monkeypatch, planted=False, context_set="rich")
    assert abs(rich["holdout_premix_bytes"]) < 0.02 * rich["shipped_binary_bytes"]
    assert rich["holdout_premix_bytes"] <= 0.0 < rich["premix_gross_bytes"]


def test_backoff_returns_the_prior_on_an_unseen_cell() -> None:
    """An unseen context must not be charged a coin flip."""
    counts = np.zeros(3, dtype=np.float64)
    hits = np.zeros(3, dtype=np.float64)
    prior = np.array([0.9, 0.5, 0.01])
    np.testing.assert_allclose(oracle.kt_estimate(counts, hits, prior), prior)
    seen = oracle.kt_estimate(np.array([1000.0]), np.array([900.0]), np.array([0.01]))
    assert 0.85 < seen[0] < 0.9


def test_holdout_rejects_a_finer_partition_that_carries_no_signal(monkeypatch) -> None:
    """With back-off, a richer but empty partition must be roughly neutral."""
    rich = _run(monkeypatch, planted=False, context_set="rich")
    summary = _run(monkeypatch, planted=False, context_set="summary")
    assert abs(rich["holdout_premix_bytes"]) < 0.01 * rich["shipped_binary_bytes"]
    assert abs(summary["holdout_premix_bytes"]) < 0.01 * summary["shipped_binary_bytes"]
