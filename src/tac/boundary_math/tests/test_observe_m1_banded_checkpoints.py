from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tac.witness_dsl.integer_plane_emitter_policy import (
    BasisMode,
    IntegerPlaneEmitterPolicy,
    IntegerPlaneEmitterStageCheckpoint,
    PolicyMode,
    STEMode,
)
from tools import observe_m1_banded_checkpoints as observer


def _tensor(shape: list[int], value: float) -> dict[str, object]:
    count = int(np.prod(shape))
    return {"dtype": "float32", "shape": shape, "data": [value] * count}


def _residual(value: float, *, frame0_value: float = 0.0) -> dict[str, object]:
    codes = np.full((600, 2, 4), value, dtype=np.float32)
    codes[:, 0] = np.float32(frame0_value)
    return {
        "pair_plane_codes": {
            "dtype": "float32",
            "shape": [600, 2, 4],
            "data": codes.reshape(-1).tolist(),
        },
        "shared_rgb_head": _tensor([4, 3], value),
    }


def _checkpoint(
    *,
    band_sha256: str = observer.EXPECTED_BAND_MANIFEST_SHA256,
    source_sha256: str = "b" * 64,
    topology_sha256: str = "c" * 64,
) -> IntegerPlaneEmitterStageCheckpoint:
    policy = IntegerPlaneEmitterPolicy(
        basis=BasisMode.R1B4_WINDOWED_CURVELET,
        mode=PolicyMode.BANDED_TRAINING,
    )
    contract = policy.compile_contract()
    return IntegerPlaneEmitterStageCheckpoint(
        policy_contract=contract,
        config_sha256="a" * 64,
        stage_name="band_fit",
        stage_index=1,
        epoch=7,
        global_step=125,
        next_pair=16,
        basis_id=BasisMode.R1B4_WINDOWED_CURVELET.value,
        ste_id=STEMode.SATURATION_AWARE_UINT8.value,
        fixed_capacity_signature=contract["capacity_signature"],
        live_residual_parameters=_residual(0.25, frame0_value=0.125),
        ema_shadow=_residual(0.2),
        optimizer_state={"step": 125, "algorithm": "fixture"},
        rng_state={
            "seed": 123,
            "stage_complete": False,
            "run_custody": {
                "band_sha256": band_sha256,
                "source_sha256": source_sha256,
            },
        },
        topology_state_sha256=topology_sha256,
        discrete_state_sha256="d" * 64,
        event_state_sha256="e" * 64,
        dual_state_sha256="f" * 64,
    )


def test_fallback_pair_sample_is_sealed_pcg64_choice_without_replacement() -> None:
    expected_prefix = (133, 581, 87, 512, 350, 81, 317, 52, 162, 21, 125, 47)
    sample = observer.pair_sample()
    assert sample[: len(expected_prefix)] == expected_prefix
    assert len(sample) == observer.FALLBACK_BOOTSTRAP_SIZE
    assert len(set(sample)) == observer.FALLBACK_BOOTSTRAP_SIZE
    assert observer.pair_sample_sha256(sample) == ("95dc07d7f9501c2e3ecef063286d60420c085e950cc60f469b2bb269fa597f70")


def test_recurring_cohort_freezes_top32_then_seeded_complement() -> None:
    population = tuple(range(128))
    per_pair = {pair_id: float((pair_id * 17) % 23) for pair_id in population}
    cohort = observer.freeze_recurring_cohort(per_pair, population_ids=population)
    assert cohort == (
        4,
        27,
        50,
        73,
        96,
        119,
        8,
        31,
        54,
        77,
        100,
        123,
        12,
        35,
        58,
        81,
        104,
        127,
        16,
        39,
        62,
        85,
        108,
        20,
        43,
        66,
        89,
        112,
        1,
        24,
        47,
        70,
        106,
        107,
        17,
        38,
        65,
        41,
        14,
        32,
        114,
        109,
        19,
        13,
        42,
        30,
        125,
        99,
    )
    assert len(set(cohort)) == observer.PAIR_SAMPLE_SIZE


def test_strict_checkpoint_envelope_and_filename_counters(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    path = tmp_path / checkpoint.filename("m1_fixture")
    payload = checkpoint.to_bytes()
    parsed = observer.parse_checkpoint_bytes(path, payload)
    assert parsed.sha256 == observer.sha256_bytes(payload)
    assert parsed.checkpoint.stage_index == 1
    assert parsed.checkpoint.epoch == 7
    assert parsed.checkpoint.global_step == 125

    wrong_name = tmp_path / path.name.replace("step000000000125", "step000000000126")
    with pytest.raises(observer.ObserverError, match="filename counters"):
        observer.parse_checkpoint_bytes(wrong_name, payload)

    noncanonical = json.dumps(json.loads(payload), indent=2).encode("ascii")
    with pytest.raises(observer.ObserverError, match="strict checkpoint envelope"):
        observer.parse_checkpoint_bytes(path, noncanonical)


def test_restart_deduplication_reads_only_canonical_unique_rows(tmp_path: Path) -> None:
    rows = tmp_path / observer.ROWS_NAME
    digest = "1" * 64
    row = {
        "schema": observer.ROW_SCHEMA,
        "checkpoint": {"sha256": digest},
        "score_claim": False,
    }
    observer.append_canonical_jsonl(rows, row)
    assert observer.load_processed_checkpoint_sha256s(rows) == {digest}

    observer.append_canonical_jsonl(rows, row)
    with pytest.raises(observer.ObserverError, match="repeats checkpoint SHA"):
        observer.load_processed_checkpoint_sha256s(rows)

    malformed = tmp_path / "noncanonical.jsonl"
    malformed.write_text(json.dumps(row) + "\n", encoding="ascii")
    with pytest.raises(observer.ObserverError, match="noncanonical"):
        observer.load_processed_checkpoint_sha256s(malformed)


def test_rank_restart_dedupes_pair_rows_and_binds_one_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / observer.RANK_ROWS_NAME
    digest = "2" * 64
    row = {
        "schema": observer.RANK_ROW_SCHEMA,
        "checkpoint": {"sha256": digest},
        "pair_id": 17,
        "dseg_rank": 1,
        "d_seg": 0.25,
    }
    observer.append_canonical_jsonl(path, row)
    checkpoint_sha, rows = observer.load_per_pair_rank_rows(path)
    assert checkpoint_sha == digest
    assert rows == {17: row}

    observer.append_canonical_jsonl(path, row)
    with pytest.raises(observer.ObserverError, match="repeats pair ID"):
        observer.load_per_pair_rank_rows(path)


def test_binding_output_and_panel_names_are_exact() -> None:
    assert observer.output_names() == {
        "facets": "facets.jsonl",
        "per_pair_rank": "facets_perpair_rank.jsonl",
        "cohort": "recurring_cohort.json",
        "preflight": "bootstrap_preflight.json",
        "cleanup": "scratch_cleanup_receipt.jsonl",
        "panels": "panels",
        "errors": "observer_errors.jsonl",
    }
    digest = "a" * 64
    assert observer.panel_name(digest, 7) == f"m1_{digest}_pair0007.png"
    with pytest.raises(observer.ObserverError, match="lowercase hexadecimal"):
        observer.panel_name("A" * 64, 7)


def test_per_class_and_candidate_stratum_accounting_is_exhaustive() -> None:
    shape = (
        observer.PAIR_SAMPLE_SIZE,
        observer.SCORER_HEIGHT,
        observer.SCORER_WIDTH,
    )
    labels = np.zeros(shape, dtype=np.int64)
    predictions = labels.copy()
    for class_id in range(1, 5):
        labels[0, 0, class_id] = class_id
        predictions[0, 0, class_id] = (class_id + 1) % 5
    predictions[0, 1, 0] = 1

    realizable = np.zeros(shape, dtype=np.bool_)
    dead = np.zeros(shape, dtype=np.bool_)
    realizable[0, 0, 1] = True
    dead[0, 0, 2] = True
    outside = ~(realizable | dead)
    changed = np.zeros(shape, dtype=np.bool_)
    changed[0, 0, 1] = True
    changed[0, 1, 0] = True
    masks = observer.StratumMasks(realizable, dead, outside)

    result = observer.facet_accounting(labels, predictions, changed, masks)
    assert [row["class_name"] for row in result["per_class"]] == list(observer.CLASS_NAMES)
    assert result["mismatch_count"] == 5
    assert result["per_class"][0]["mismatch_count"] == 1
    assert all(row["mismatch_count"] == 1 for row in result["per_class"][1:])
    strata = result["d_seg_by_candidate_stratum"]
    assert strata["realizable_band"]["mismatch_count"] == 1
    assert strata["structurally_dead_candidate"]["mismatch_count"] == 1
    assert strata["outside_candidate"]["mismatch_count"] == 3
    assert sum(row["total_d_seg_contribution"] for row in strata.values()) == pytest.approx(result["overall_d_seg"])
    residency = result["changed_pixel_residency"]
    assert residency["changed_pixel_count"] == 2
    assert residency["inside_realizable_band"] == 1
    assert residency["outside_realizable_band"] == 1


def test_out_of_band_excursion_is_keyed_by_gt_to_emitted_class_pair() -> None:
    shape = (1, observer.SCORER_HEIGHT, observer.SCORER_WIDTH)
    labels = np.zeros(shape, dtype=np.int8)
    predictions = labels.copy()
    predictions[0, 0, 0] = 1
    predictions[0, 0, 1] = 1
    source = np.full((*shape, 3), 10, dtype=np.uint8)
    receiver = source.copy()
    receiver[0, 0, 0, 0] = 13
    receiver[0, 0, 1, 0] = 11
    radii = np.ones((*shape, 3), dtype=np.float32)
    realizable = np.zeros(shape, dtype=np.bool_)
    realizable[0, 0, :2] = True
    row, mask = observer.out_of_band_excursion(labels, predictions, receiver, source, radii, realizable)
    assert row["realizable_pixel_count"] == 2
    assert row["excursion_pixel_count"] == 1
    assert row["excursion_fraction"] == 0.5
    road_to_lane = row["gt_to_emitted_class_rows"][1]
    assert road_to_lane["gt_class_name"] == "Road"
    assert road_to_lane["emitted_class_name"] == "Lane"
    assert road_to_lane["pixel_count"] == 2
    assert road_to_lane["excursion_count"] == 1
    assert bool(mask[0, 0, 0])


def test_pair_internal_temporal_instability_and_tails_are_deterministic() -> None:
    frame0 = np.zeros((2, observer.SCORER_HEIGHT, observer.SCORER_WIDTH), dtype=np.int8)
    frame1 = frame0.copy()
    frame1[0, 0, :2] = 1
    frame1[1, 0, 0] = 1
    row = observer.temporal_argmax_instability((9, 3), frame0, frame1)
    assert row["definition"].startswith("pair_internal_consecutive_frame")
    assert row["worst_pairs"][0]["pair_id"] == 9
    assert observer.top_pair_rows((9, 3), (0.5, 0.5)) == [
        {"pair_id": 3, "value": 0.5},
        {"pair_id": 9, "value": 0.5},
    ]


def test_live_code_byte_estimate_uses_exact_float32_zlib_level9() -> None:
    codes = np.arange(600 * 2 * 4, dtype=np.float32).reshape(600, 2, 4)
    row = observer.estimate_pair_plane_code_bytes(codes)
    payload = np.ascontiguousarray(codes, dtype="<f4").tobytes(order="C")
    assert row["source_state"] == "live_not_ema"
    assert row["raw_bytes"] == len(payload)
    assert row["zlib_level"] == 9
    assert row["zlib_level9_bytes"] < row["raw_bytes"]
    assert row["raw_sha256"] == observer.sha256_bytes(payload)
    with pytest.raises(observer.ObserverError, match="float32"):
        observer.estimate_pair_plane_code_bytes(codes.astype(np.float64))


def test_checkpoint_band_and_binding_custody_fail_closed() -> None:
    bad_band_checkpoint = _checkpoint(band_sha256="9" * 64)
    bad_band_parsed = SimpleNamespace(checkpoint=bad_band_checkpoint)
    band = SimpleNamespace(source_sha256="b" * 64)
    binding = SimpleNamespace(coordinate_basis=lambda: np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(observer.ObserverError, match="band SHA"):
        observer._checkpoint_custody(bad_band_parsed, band, binding)

    topology = np.zeros((2, 2), dtype=np.float32)
    wrong_topology_checkpoint = _checkpoint(topology_sha256="8" * 64)
    wrong_topology_parsed = SimpleNamespace(checkpoint=wrong_topology_checkpoint)
    binding = SimpleNamespace(coordinate_basis=lambda: topology)
    with pytest.raises(observer.ObserverError, match="topology"):
        observer._checkpoint_custody(wrong_topology_parsed, band, binding)


def test_static_custody_refuses_unsealed_band_before_loading(tmp_path: Path) -> None:
    band = tmp_path / "band.json"
    band.write_text("{}", encoding="ascii")
    config = observer.ObserverConfig(
        checkpoint_dir=tmp_path,
        output_dir=tmp_path,
        band_manifest=band,
        carrier_binding=tmp_path / "binding.json",
        gt_cache=tmp_path / "gt.npz",
        upstream_dir=tmp_path,
        live_base_raw=None,
    )
    with pytest.raises(observer.ObserverError, match="sealed M1 SHA"):
        observer._validate_static_custody(config)


def test_partial_inputs_are_never_hashed(tmp_path: Path) -> None:
    partial = tmp_path / "base_camera_frames.raw.partial"
    partial.write_bytes(b"do not inspect")
    with pytest.raises(observer.ObserverError, match="refusing to hash partial"):
        observer.sha256_file(partial)


def test_success_only_full_scratch_cleanup_is_restart_deduped(tmp_path: Path) -> None:
    scratch = tmp_path / observer.FULL_BASE_NAME
    scratch.write_bytes(b"observer-owned-certified-scratch")
    rank = tmp_path / observer.RANK_ROWS_NAME
    rank.write_bytes(b"complete rank table\n")
    (tmp_path / observer.COHORT_RECEIPT_NAME).write_bytes(b"durable cohort")
    recurring = tmp_path / observer.COHORT_BASE_NAME
    recurring.write_bytes(b"recurring snapshot")
    bootstrap_manifest = {
        "snapshot_npy_sha256": observer.sha256_file(scratch),
        "source_path": "/read-only/live/base_camera_frames.raw",
        "source_bytes": observer.FULL_RAW_BYTES,
        "source_sha256": "b" * 64,
        "source_stat": {"device": 1, "inode": 2, "mtime_ns": 3},
    }
    cohort_manifest = {
        "snapshot_path": str(recurring),
        "snapshot_npy_bytes": recurring.stat().st_size,
        "snapshot_npy_sha256": observer.sha256_file(recurring),
    }
    observer._cleanup_full_bootstrap_scratch(
        output_dir=tmp_path,
        bootstrap_kind="full_n600_scratch",
        bootstrap_manifest=bootstrap_manifest,
        cohort_manifest=cohort_manifest,
        rank_path=rank,
    )
    assert not scratch.exists()
    receipt = tmp_path / observer.CLEANUP_RECEIPT_NAME
    first = receipt.read_bytes()
    assert first.count(b"\n") == 2
    observer._cleanup_full_bootstrap_scratch(
        output_dir=tmp_path,
        bootstrap_kind="full_n600_scratch",
        bootstrap_manifest=bootstrap_manifest,
        cohort_manifest=cohort_manifest,
        rank_path=rank,
    )
    assert receipt.read_bytes() == first


def test_mechanism_signature_carries_flip_boundary_and_flicker_axes() -> None:
    labels = np.zeros((observer.SCORER_HEIGHT, observer.SCORER_WIDTH), dtype=np.int8)
    labels[:, observer.SCORER_WIDTH // 2 :] = 1
    emitted = labels.copy()
    emitted[0, observer.SCORER_WIDTH // 2 - 1] = 1
    emitted[10, 10] = 1
    frame0 = emitted.copy()
    frame0[20, 20] = 2
    row = observer.mechanism_signature(labels, emitted, frame0)
    assert row["flip_count"] == 2
    assert row["boundary_flip_count"] == 1
    assert row["boundary_flip_fraction"] == 0.5
    assert row["temporal_flicker_flag"] is True
    assert len(row["class_flip_composition"]) == 25
    assert len(row["vector"]) == 27


def test_panel_plan_is_data_derived_and_consumes_registered_break_even() -> None:
    rows = {}
    for pair_id in range(24):
        vector = [0.0] * 27
        vector[pair_id % 5] = 1.0
        vector[-2] = (pair_id % 3) / 2.0
        vector[-1] = float(pair_id % 2)
        rows[pair_id] = {
            "d_seg": (24 - pair_id) / 1000.0,
            "mechanism_signature": {"vector": vector},
        }
    first = observer.derive_panel_plan(rows)
    second = observer.derive_panel_plan(rows)
    assert first == second
    assert 2 <= first["k_selected"] <= 20
    assert len(first["selection_score_curve"]) == 19
    assert first["good_turing"]["break_even"]["equation_id"] == ("realization_breakeven_bytes_v1")
    assert first["good_turing"]["break_even"]["callable_roundtrip_bytes"] == 150.0
    assert first["total_panels_admitted"] == len(first["pair_ids"])
