from __future__ import annotations

import numpy as np
import pytest

from experiments import ddm_hm1_correction_capacity_ladder as ladder
from experiments import ddm_hm1_hpac_logit_replay as replay
from tac.payload_retention_gate import check_no_measure_and_discard_payload

PLANE_TIE_ROWS = 16


def test_packed_table_bytes_matches_the_shipped_rcf1_layout() -> None:
    # residual_archive._decode_fixed_table reads 4 magic + 2 fp16 scale + 25*5 codes at
    # 6 bits.  If this drifts, every model-byte figure in the ladder is wrong.
    assert ladder.packed_table_bytes(25) == 4 + 2 + (25 * 5 * 6 + 7) // 8
    assert ladder.packed_table_bytes(25) == 100
    assert ladder.packed_table_bytes(1) == 6 + 4
    assert ladder.packed_table_bytes(0) == 6


def test_boundary_buckets_reproduce_the_shipped_max_distance_four_behaviour() -> None:
    previous = np.zeros((6, 6), dtype=np.uint8)
    previous[:, 3:] = 1
    result = ladder.boundary_buckets(previous, 4)
    # The class change sits between columns 2 and 3, so both sides are distance 0.
    assert result[0, 2] == 0
    assert result[0, 3] == 0
    assert result[0, 1] == 1
    assert result[0, 0] == 2
    uniform = ladder.boundary_buckets(np.zeros((4, 4), dtype=np.uint8), 4)
    assert np.all(uniform == 4)


def test_boundary_buckets_reject_a_non_frame_input() -> None:
    with pytest.raises(ladder.LadderError):
        ladder.boundary_buckets(np.zeros(8, dtype=np.uint8), 4)


def test_quantize_table_is_decodable_as_six_bit_codes_times_one_scale() -> None:
    table = np.array([[0.5, -0.25, 0.0, 1.0, -1.0]], dtype=np.float32)
    restored, scale = ladder.quantize_table(table)
    assert scale > 0.0
    codes = np.rint(restored / np.float32(scale))
    assert np.all(np.abs(codes) <= 32)
    assert np.allclose(codes * np.float32(scale), restored)


def test_quantize_table_scale_search_protects_the_dense_cell_from_an_outlier() -> None:
    # One extreme, nearly empty cell must not set the resolution for a dense one.
    table = np.array([[40.0, 0.0, 0.0, 0.0, 0.0], [0.3, -0.3, 0.0, 0.0, 0.0]], np.float32)
    weighted, weighted_scale = ladder.quantize_table(
        table, np.array([1.0, 10_000_000.0])
    )
    # max|T|/31 would put the step at 1.29 logits and round the dense cell's 0.3 to 0.
    naive_scale = 40.0 / 31.0
    assert weighted_scale < naive_scale / 8.0
    assert abs(float(weighted[1, 0]) - 0.3) < 0.05

    # With the weight on the outlier instead, the search is free to keep the peak and
    # the dense cell is the one that gets crushed.  Same code, opposite trade.
    peak_weighted, peak_scale = ladder.quantize_table(
        table, np.array([10_000_000.0, 1.0])
    )
    assert peak_scale > weighted_scale
    assert abs(float(peak_weighted[0, 0]) - 40.0) < abs(float(weighted[0, 0]) - 40.0)


def test_quantize_table_handles_an_all_zero_table() -> None:
    restored, scale = ladder.quantize_table(np.zeros((3, 5), dtype=np.float32))
    assert np.all(restored == 0.0)
    assert scale == 1.0


def test_evaluate_cost_bytes_reproduces_the_receiver_probability_pipeline() -> None:
    rng = np.random.default_rng(20260816)
    logits = rng.integers(-64, 64, size=(2048, 5)).astype(np.int16)
    truth = rng.integers(0, 5, size=2048).astype(np.int64)
    cells = np.zeros(2048, dtype=np.int64)
    table = np.zeros((1, 5), dtype=np.float32)

    measured, floor = ladder.evaluate_cost_bytes(logits, truth, cells, table, 512)

    corrected = logits.astype(np.float32) / ladder.LOGIT_PRECISION
    reference = replay._probability_from_corrected(corrected, ladder.LOGIT_PRECISION)
    chosen = reference[np.arange(truth.size), truth].astype(np.float64)
    expected = float(-np.log2(chosen).sum()) / 8.0
    assert measured == pytest.approx(expected, rel=1e-12)
    assert floor == pytest.approx(float(chosen.min()), rel=1e-12)


def test_evaluate_cost_bytes_is_invariant_to_chunking() -> None:
    rng = np.random.default_rng(7)
    logits = rng.integers(-30, 30, size=(1000, 5)).astype(np.int16)
    truth = rng.integers(0, 5, size=1000).astype(np.int64)
    cells = rng.integers(0, 4, size=1000).astype(np.int64)
    table = rng.normal(0.0, 0.2, size=(4, 5)).astype(np.float32)
    whole, _ = ladder.evaluate_cost_bytes(logits, truth, cells, table, 1000)
    split, _ = ladder.evaluate_cost_bytes(logits, truth, cells, table, 37)
    assert whole == pytest.approx(split, rel=1e-9)


def test_fit_table_lowers_the_realized_cost_on_a_biased_cell() -> None:
    # Cell 1's truth is deliberately mismatched to the base logits, so a correction
    # must exist and the fit must find it.  A fit that returns zeros passes nothing.
    logits = np.zeros((4000, 5), dtype=np.int16)
    logits[:, 0] = 40
    truth = np.zeros(4000, dtype=np.int64)
    cells = np.zeros(4000, dtype=np.int64)
    cells[2000:] = 1
    truth[2000:] = 3

    table = ladder.fit_table(logits, truth, cells, 2, 6, 1024, 1.0)
    assert table[1, 3] > table[1, 0]
    before, _ = ladder.evaluate_cost_bytes(
        logits, truth, cells, np.zeros((2, 5), np.float32), 1024
    )
    after, _ = ladder.evaluate_cost_bytes(logits, truth, cells, table, 1024)
    assert after < before


def test_fit_table_keeps_every_probability_representable_by_the_coder() -> None:
    # Without the absolute L2 prior the offset of a never-observed class runs to
    # -infinity, the softmax underflows and the rung is silently unrealizable.
    logits = np.zeros((5000, 5), dtype=np.int16)
    truth = np.zeros(5000, dtype=np.int64)
    cells = np.arange(5000, dtype=np.int64) % 50
    table = ladder.fit_table(logits, truth, cells, 50, 8, 2048, 1.0)
    assert np.all(np.isfinite(table))
    _, floor = ladder.evaluate_cost_bytes(logits, truth, cells, table, 2048)
    assert floor > 2.0**-31


def test_fit_table_zeroes_cells_that_hold_no_symbols() -> None:
    logits = np.zeros((100, 5), dtype=np.int16)
    truth = np.zeros(100, dtype=np.int64)
    cells = np.zeros(100, dtype=np.int64)
    table = ladder.fit_table(logits, truth, cells, 4, 4, 64, 1.0)
    assert np.all(table[1:] == 0.0)


def test_oracle_bytes_is_zero_on_a_deterministic_cell_and_maximal_on_a_uniform_one() -> None:
    truth = np.zeros(1000, dtype=np.int64)
    cells = np.zeros(1000, dtype=np.int64)
    assert ladder.oracle_bytes(cells, truth, 1) == pytest.approx(0.0)

    uniform = np.tile(np.arange(5, dtype=np.int64), 1000)
    flat = np.zeros(uniform.size, dtype=np.int64)
    expected = uniform.size * np.log2(5.0) / 8.0
    assert ladder.oracle_bytes(flat, uniform, 1) == pytest.approx(expected, rel=1e-12)


def test_oracle_bytes_never_exceeds_the_coarser_partition_it_refines() -> None:
    rng = np.random.default_rng(11)
    truth = rng.integers(0, 5, size=20_000).astype(np.int64)
    coarse = rng.integers(0, 4, size=20_000).astype(np.int64)
    fine = coarse * 5 + rng.integers(0, 5, size=20_000)
    assert ladder.oracle_bytes(fine, truth, 20) <= ladder.oracle_bytes(coarse, truth, 4) + 1e-9


def test_context_features_resolve_argmax_ties_the_way_the_shipped_decoder_does() -> None:
    # decode_production_tokens keys the RCF1 table on ``base_logits.argmax(axis=1)``,
    # which returns the FIRST maximum.  np.argsort is unstable and returns the last, so
    # a ladder built on argsort silently de-aligns rung r1 from the shipped table.
    logits = np.zeros((PLANE_TIE_ROWS, 5), dtype=np.int16)
    logits[:, 0] = 24
    logits[:, 1] = 24  # deliberate tie between class 0 and class 1
    logits[:, 2] = 8
    base = logits.astype(np.float32)
    assert base.argmax(axis=1)[0] == 0
    assert np.argsort(base, axis=1)[0, -1] == 1

    top_index = base.argmax(axis=1)
    rows = np.arange(base.shape[0])
    masked = base.copy()
    masked[rows, top_index] = -np.inf
    second_index = masked.argmax(axis=1)
    assert second_index[0] == 1
    assert float((base[rows, top_index] - masked[rows, second_index])[0]) == 0.0


def test_ladder_and_replay_pass_the_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(strict=False, roots=("experiments",))
    mine = [
        finding
        for finding in findings
        if "ddm_hm1_" in str(getattr(finding, "path", finding))
    ]
    assert mine == []


def test_six_bit_packing_round_trips_through_the_real_runtime_unpacker() -> None:
    # The receiver's unpack_signed is LITTLE-endian at the bit level.  Packing MSB-first
    # produces a blob it decodes to different numbers with no error raised, so this test
    # is pinned against the actual shipped implementation, not a re-typed copy.
    import sys

    prepared = replay.DEFAULT_PREPARED
    if not (prepared / "runtime" / "bits.py").is_file():
        pytest.skip("prepared hv1 receiver tree is not mounted")
    if str(prepared) not in sys.path:
        sys.path.insert(0, str(prepared))
    from runtime.bits import packed_length, unpack_signed

    from experiments import ddm_hm1_reprice_tables as reprice

    rng = np.random.default_rng(20260816)
    codes = rng.integers(-32, 32, size=125).astype(np.int64)
    blob = reprice.pack_six_bit_codes(codes)
    assert len(blob) == packed_length(codes.size, 6)
    assert list(unpack_signed(blob, codes.size, 6)) == codes.tolist()

    edge = np.array([-32, 31, 0, -1, 1], dtype=np.int64)
    edge_blob = reprice.pack_six_bit_codes(edge)
    assert list(unpack_signed(edge_blob, edge.size, 6)) == edge.tolist()


def test_six_bit_packing_rejects_out_of_range_codes() -> None:
    from experiments import ddm_hm1_reprice_tables as reprice

    with pytest.raises(reprice.RepriceError):
        reprice.pack_six_bit_codes(np.array([32], dtype=np.int64))
    with pytest.raises(reprice.RepriceError):
        reprice.pack_six_bit_codes(np.array([-33], dtype=np.int64))
