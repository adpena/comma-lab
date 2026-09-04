"""Tests for ``experiments/ddm_md1_micro_to_macro.py``.

The load-bearing surfaces are (a) the cadence, (b) the margin/band arithmetic that must agree with
the trainer's own ``expected_flip_margin_loss`` margin, (c) the HT estimator that must reproduce the
sealed ``_weighted_mean``, (d) the site-trajectory falling rule -- which is only useful if it
PARTITIONS, and (e) the macro bridge, whose whole value is that its class sums are an identity
rather than an approximation.  Every test here would fail if the code were mutated in the way the
test names describe; none of them merely asserts a constant.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

md1 = pytest.importorskip("experiments.ddm_md1_micro_to_macro")


# ---------------------------------------------------------------------------
# cadence
# ---------------------------------------------------------------------------
def test_sweep_steps_is_sorted_unique_and_spans_the_run() -> None:
    steps = md1.sweep_steps(5000)
    assert steps[0] == 0
    assert steps[-1] == 5000
    assert list(steps) == sorted(set(steps))


def test_sweep_steps_is_dense_through_the_birth_and_coarse_at_the_end() -> None:
    steps = md1.sweep_steps(5000)
    birth = [s for s in steps if 0 < s <= 512]
    assert birth == list(range(16, 513, 16))
    peak = [s for s in steps if 512 < s <= 2048]
    # the medium ladder PLUS the milestone step 2,000, which is a 16-step checkpoint
    assert peak == sorted(set(range(576, 2049, 64)) | {2000})
    tail = [s for s in steps if 2048 < s < 5000]
    # the coarse ladder PLUS the milestone steps that have a 16-step checkpoint
    assert tail == sorted(set(range(2304, 4865, 256)) | {4000})


def test_sweep_steps_every_nonzero_step_is_a_multiple_of_the_checkpoint_period() -> None:
    # every retained checkpoint is periodic_{16k}; a cadence that asked for a non-multiple would
    # silently skip that step in available_steps and shrink the trajectory without saying so.
    for step in md1.sweep_steps(5000):
        assert step % 16 == 0 or step == 5000


def test_sweep_steps_truncates_for_a_shorter_run() -> None:
    steps = md1.sweep_steps(1000)
    assert steps[-1] == 1000
    assert max(steps) <= 1000
    assert 2304 not in steps


def test_sweep_steps_folds_in_every_checkpointed_milestone_step() -> None:
    # the CPU-vs-retained-MPS calibration is only possible at a milestone that has a checkpoint.
    steps = set(md1.sweep_steps(5000))
    for milestone in md1.MILESTONE_STEPS:
        if milestone % 16 == 0 or milestone == 5000:
            assert milestone in steps, milestone
        else:
            assert milestone not in steps, milestone
    assert {0, 2000, 4000, 5000} <= steps
    assert 1000 not in steps and 3000 not in steps


def test_checkpoint_path_uses_stage_end_for_the_terminal_step() -> None:
    root = Path("/nowhere/run")
    assert md1.checkpoint_path(root, 5000, 5000).name == "stage_01_end.pt"
    assert md1.checkpoint_path(root, 16, 5000).name == "periodic_000016.pt"
    assert md1.checkpoint_path(root, 4992, 5000).name == "periodic_004992.pt"


def test_available_steps_only_returns_steps_whose_file_exists(tmp_path: Path) -> None:
    stage = tmp_path / "stage_01_fairform_finish/checkpoints"
    stage.mkdir(parents=True)
    (stage / "periodic_000016.pt").write_bytes(b"x")
    (stage / "periodic_000048.pt").write_bytes(b"x")
    steps = md1.available_steps(tmp_path, 5000)
    assert steps == (0, 16, 48)


def test_available_steps_rejects_a_zero_byte_checkpoint(tmp_path: Path) -> None:
    # a LIVE cell can be mid-write; a zero-byte file must never be swept as if it were a state.
    stage = tmp_path / "stage_01_fairform_finish/checkpoints"
    stage.mkdir(parents=True)
    (stage / "periodic_000016.pt").write_bytes(b"")
    assert md1.available_steps(tmp_path, 5000) == (0,)


# ---------------------------------------------------------------------------
# margin + bands
# ---------------------------------------------------------------------------
def test_signed_margin_matches_the_trainer_expected_flip_margin() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(20260904)
    logits = rng.normal(size=(3, 5, 7, 11)).astype(np.float32)
    target = rng.integers(0, 5, size=(3, 7, 11)).astype(np.uint8)
    mine = md1.signed_margin(logits, target)
    reference_logits = torch.from_numpy(logits)
    reference_target = torch.from_numpy(target.astype(np.int64))
    index = reference_target[:, None].long()
    target_logit = reference_logits.gather(1, index).squeeze(1)
    other = reference_logits.clone()
    other.scatter_(1, index, -1.0e9)
    reference = (target_logit - other.amax(dim=1)).numpy()
    assert np.allclose(mine, reference, rtol=0, atol=1e-5)


def test_signed_margin_is_positive_exactly_when_argmax_is_correct() -> None:
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(2, 5, 6, 6)).astype(np.float32)
    target = rng.integers(0, 5, size=(2, 6, 6)).astype(np.uint8)
    margin = md1.signed_margin(logits, target)
    correct = logits.argmax(axis=1) == target
    assert np.array_equal(margin > 0, correct)


def test_signed_margin_refuses_wrong_geometry() -> None:
    with pytest.raises(md1.MD1Error):
        md1.signed_margin(np.zeros((2, 4, 6, 6), dtype=np.float32), np.zeros((2, 6, 6), dtype=np.uint8))


def test_band_index_boundaries_are_right_open_in_delta_R_units() -> None:
    delta = 0.5
    values = np.asarray([0.0, 0.4999, 0.5, 0.75, 1.0, 5.0, 12.4999, 12.5, 100.0])
    bands = md1.band_index(values, delta)
    # edges are 1x, 2x, 25x delta -> 0.5, 1.0, 12.5
    assert list(bands) == [0, 0, 1, 1, 2, 2, 2, 3, 3]


def test_band_index_scales_with_delta_R() -> None:
    values = np.asarray([0.03])
    assert md1.band_index(values, 0.021881818771362305)[0] == 1
    assert md1.band_index(values, 0.05)[0] == 0


def test_band_names_and_edges_stay_in_step() -> None:
    assert len(md1.BAND_NAMES) == len(md1.BAND_EDGES_DELTA_R) + 1


# ---------------------------------------------------------------------------
# the HT estimator
# ---------------------------------------------------------------------------
def test_weighted_d_seg_reproduces_the_sealed_weighted_mean() -> None:
    pytest.importorskip("torch")
    from experiments import ddm_qbr1_born_fairform_burn_prep as qbr1
    from experiments import ddm_qbt1_qbflow_trainer as qbt

    rng = np.random.default_rng(11)
    per_pair = rng.random(len(qbt.SELECTION_IDS))
    rows = [
        {"pair_id": int(pid), "d_seg": float(value)}
        for pid, value in zip(qbt.SELECTION_IDS, per_pair, strict=True)
    ]
    reference = qbr1._weighted_mean(rows, "d_seg")
    weights = md1.ht_weights_vector(qbt.SELECTION_IDS, qbt.SELECTION_IDS, qbt.SELECTION_WEIGHTS)
    assert md1.weighted_d_seg(per_pair, weights, qbt.N) == pytest.approx(reference, rel=0, abs=1e-15)


def test_ht_weights_vector_follows_the_pair_order_it_is_given() -> None:
    pytest.importorskip("torch")
    from experiments import ddm_qbt1_qbflow_trainer as qbt

    reversed_ids = tuple(reversed(qbt.SELECTION_IDS))
    weights = md1.ht_weights_vector(reversed_ids, qbt.SELECTION_IDS, qbt.SELECTION_WEIGHTS)
    assert list(weights) == list(reversed(qbt.SELECTION_WEIGHTS))


def test_weighted_d_seg_refuses_mismatched_shapes() -> None:
    with pytest.raises(md1.MD1Error):
        md1.weighted_d_seg(np.zeros(4), np.zeros(3), 600)


# ---------------------------------------------------------------------------
# site trajectory classification
# ---------------------------------------------------------------------------
def _cube(*columns: list[int]) -> np.ndarray:
    return np.asarray(columns, dtype=bool).T


def test_classify_sites_assigns_each_named_class() -> None:
    # columns are sites; rows are checkpoints (row 0 == step 0)
    always_correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    persistent = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    new_persistent = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    transient = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    healed = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    churn = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    cube = _cube(always_correct, persistent, new_persistent, transient, healed, churn)
    codes = md1.classify_sites(cube, churn_flips=4, persistent_fraction=0.90)
    expected = [
        md1.CLASS_ALWAYS_CORRECT,
        md1.CLASS_PERSISTENT,
        md1.CLASS_NEW_PERSISTENT,
        md1.CLASS_TRANSIENT_BORN,
        md1.CLASS_HEALED,
        md1.CLASS_CHURN,
    ]
    assert [md1.SITE_CLASSES[c] for c in codes] == expected


def test_classify_sites_churn_takes_priority_over_the_endpoint_rules() -> None:
    # wrong at 0, wrong at the end, but flipping 8 times: CHURN, not PERSISTENT/NEW_PERSISTENT.
    trajectory = [1, 0, 1, 0, 1, 0, 1, 0, 1]
    codes = md1.classify_sites(_cube(trajectory), churn_flips=4, persistent_fraction=0.90)
    assert md1.SITE_CLASSES[codes[0]] == md1.CLASS_CHURN


def test_classify_sites_churn_threshold_is_strictly_greater_than() -> None:
    four_flips = [0, 1, 1, 0, 0, 1, 1, 0, 0, 0]  # exactly 4 flips -> not churn
    codes = md1.classify_sites(_cube(four_flips), churn_flips=4, persistent_fraction=0.90)
    assert md1.SITE_CLASSES[codes[0]] == md1.CLASS_TRANSIENT_BORN
    codes = md1.classify_sites(_cube(four_flips), churn_flips=3, persistent_fraction=0.90)
    assert md1.SITE_CLASSES[codes[0]] == md1.CLASS_CHURN


def test_classify_sites_persistent_needs_both_step_zero_and_the_fraction() -> None:
    # wrong at 0 and wrong at 8 of 10 checkpoints -> below 0.90, so HEALED/NEW_PERSISTENT not
    # PERSISTENT.  This is the boundary the "representation-bound" reading rests on.
    eight_of_ten = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    codes = md1.classify_sites(_cube(eight_of_ten), churn_flips=4, persistent_fraction=0.90)
    assert md1.SITE_CLASSES[codes[0]] == md1.CLASS_HEALED
    codes = md1.classify_sites(_cube(eight_of_ten), churn_flips=4, persistent_fraction=0.80)
    assert md1.SITE_CLASSES[codes[0]] == md1.CLASS_PERSISTENT


def test_classify_sites_correct_at_zero_never_becomes_persistent() -> None:
    born_and_stays = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    codes = md1.classify_sites(_cube(born_and_stays), churn_flips=4, persistent_fraction=0.90)
    assert md1.SITE_CLASSES[codes[0]] == md1.CLASS_NEW_PERSISTENT


def test_classify_sites_partitions_every_site_exactly_once() -> None:
    rng = np.random.default_rng(20260904)
    cube = rng.random((23, 4096)) < 0.3
    codes = md1.classify_sites(cube, churn_flips=4, persistent_fraction=0.90)
    counts = np.bincount(codes, minlength=len(md1.SITE_CLASSES))
    assert int(counts.sum()) == cube.shape[1]
    assert set(np.unique(codes)) <= set(range(len(md1.SITE_CLASSES)))
    # every site that is ever wrong lands in an ERROR class, never in ALWAYS_CORRECT
    ever = cube.any(axis=0)
    assert not np.any(codes[ever] == md1.CLASS_CODE[md1.CLASS_ALWAYS_CORRECT])
    assert np.all(codes[~ever] == md1.CLASS_CODE[md1.CLASS_ALWAYS_CORRECT])


def test_classify_sites_refuses_a_single_checkpoint() -> None:
    with pytest.raises(md1.MD1Error):
        md1.classify_sites(np.zeros((1, 4), dtype=bool), churn_flips=4, persistent_fraction=0.9)


# ---------------------------------------------------------------------------
# the macro bridge
# ---------------------------------------------------------------------------
def _bridge(cube: np.ndarray, codes: np.ndarray, pairs: int, weights: np.ndarray) -> dict:
    return md1.macro_bridge(
        cube.reshape(cube.shape[0], pairs, cube.shape[1] // pairs),
        codes,
        pair_weights=weights,
        sites_per_pair=cube.shape[1] // pairs,
        population_n=600,
    )


def test_macro_bridge_class_sums_reproduce_the_total_exactly() -> None:
    rng = np.random.default_rng(3)
    cube = rng.random((17, 2048)) < 0.25
    codes = md1.classify_sites(cube, churn_flips=4, persistent_fraction=0.90)
    weights = np.asarray([15.0] * 6 + [30.0] * 2)
    bridge = _bridge(cube, codes, 8, weights)
    assert bridge["calibration_gate_max_abs_integer_residual"] == 0
    assert bridge["calibration_gate_exact_zero"] is True
    # the integer numerator must equal the sealed HT numerator recomputed independently
    counts = cube.reshape(17, 8, 256).sum(axis=2)
    reference = (counts * weights.astype(np.int64)[None, :]).sum(axis=1)
    assert bridge["weighted_wrong_site_numerator_by_step"] == [int(v) for v in reference]


def test_macro_bridge_d_seg_matches_the_sealed_weighted_mean() -> None:
    rng = np.random.default_rng(19)
    cube = rng.random((4, 1024)) < 0.4
    codes = md1.classify_sites(cube, churn_flips=4, persistent_fraction=0.90)
    weights = np.asarray([15.0, 15.0, 30.0, 30.0])
    bridge = _bridge(cube, codes, 4, weights)
    per_pair = cube.reshape(4, 4, 256).mean(axis=2)
    for step in range(4):
        expected = md1.weighted_d_seg(per_pair[step], weights, 600)
        assert bridge["d_seg_hat_by_step"][step] == pytest.approx(expected, rel=0, abs=1e-18)


def test_macro_bridge_reports_every_class_even_when_empty() -> None:
    cube = np.zeros((5, 16), dtype=bool)
    codes = md1.classify_sites(cube, churn_flips=4, persistent_fraction=0.9)
    bridge = _bridge(cube, codes, 4, np.asarray([15.0] * 4))
    assert set(bridge["contribution_by_class"]) == set(md1.SITE_CLASSES)
    assert all(v == 0.0 for v in bridge["contribution_by_class"][md1.CLASS_PERSISTENT])
    assert bridge["calibration_gate_exact_zero"] is True


def test_macro_bridge_persistent_contribution_is_flat_when_the_site_never_recovers() -> None:
    cube = np.ones((6, 4), dtype=bool)
    codes = md1.classify_sites(cube, churn_flips=4, persistent_fraction=0.9)
    bridge = _bridge(cube, codes, 2, np.asarray([15.0, 30.0]))
    numerators = bridge["numerator_by_class"][md1.CLASS_PERSISTENT]
    assert numerators == [15 * 2 + 30 * 2] * 6
    assert bridge["numerator_by_class"][md1.CLASS_CHURN] == [0] * 6


def test_macro_bridge_refuses_non_integral_ht_weights() -> None:
    cube = np.ones((3, 4), dtype=bool)
    codes = md1.classify_sites(cube, churn_flips=4, persistent_fraction=0.9)
    with pytest.raises(md1.MD1Error):
        _bridge(cube, codes, 2, np.asarray([15.5, 30.0]))


def test_macro_bridge_refuses_a_flat_cube() -> None:
    cube = np.ones((3, 4), dtype=bool)
    codes = md1.classify_sites(cube, churn_flips=4, persistent_fraction=0.9)
    with pytest.raises(md1.MD1Error):
        md1.macro_bridge(cube, codes, pair_weights=np.asarray([15.0]), sites_per_pair=4, population_n=600)


# ---------------------------------------------------------------------------
# payload custody
# ---------------------------------------------------------------------------
def test_atomic_npz_writes_the_final_name_and_leaves_no_partial(tmp_path: Path) -> None:
    target = tmp_path / "block.npz"
    fact = md1.atomic_npz(target, values=np.arange(9, dtype=np.uint8))
    assert target.is_file()
    assert not list(tmp_path.glob("*.partial.npz"))
    assert fact["sha256"] == md1.sha256_file(target)
    assert fact["bytes"] == target.stat().st_size
    with np.load(target) as payload:
        assert list(payload["values"]) == list(range(9))


def test_atomic_npz_overwrites_in_place(tmp_path: Path) -> None:
    target = tmp_path / "block.npz"
    md1.atomic_npz(target, values=np.zeros(4, dtype=np.uint8))
    second = md1.atomic_npz(target, values=np.ones(4, dtype=np.uint8))
    assert second["sha256"] == md1.sha256_file(target)
    with np.load(target) as payload:
        assert list(payload["values"]) == [1, 1, 1, 1]


def test_load_cube_refuses_a_missing_payload(tmp_path: Path) -> None:
    with pytest.raises(md1.MD1Error):
        md1.load_cube(tmp_path, "cell", "shadow", [0])


def test_load_cube_returns_the_steps_in_the_order_requested(tmp_path: Path) -> None:
    root = tmp_path / "payloads" / "cell"
    root.mkdir(parents=True)
    for step, value in ((0, 3), (16, 4)):
        md1.atomic_npz(
            root / f"shadow_step_{step:06d}.npz",
            argmax_u8=np.full((1, 2, 2), value, dtype=np.uint8),
            band_u8=np.zeros((1, 2, 2), dtype=np.uint8),
        )
    argmax, bands = md1.load_cube(tmp_path, "cell", "shadow", [16, 0])
    assert argmax[0, 0, 0, 0] == 4
    assert argmax[1, 0, 0, 0] == 3
    assert bands.shape == argmax.shape


# ---------------------------------------------------------------------------
# contract / honesty
# ---------------------------------------------------------------------------
def test_axis_label_is_advisory_and_names_the_reconstruction() -> None:
    assert "advisory" in md1.AXIS
    assert "not contest authority" in md1.AXIS


def test_delta_R_is_law_resolved_and_never_retyped_in_this_module() -> None:
    source = Path(md1.__file__).read_text(encoding="utf-8")
    # the only permitted appearance of the n600 delta_R decimal is inside the default margin
    # histogram edge list, which is a REPORTING grid, not a threshold; the band threshold itself
    # must come from sd1's constant.
    assert "DELTA_R_N600 = " not in source
    assert md1.DELTA_R_SOURCE.endswith("DELTA_R_N600")


def test_class_codes_are_stable_and_always_correct_is_zero() -> None:
    # the persisted site_class_u8 payload is only readable if these codes do not drift.
    assert md1.CLASS_CODE[md1.CLASS_ALWAYS_CORRECT] == 0
    assert list(md1.SITE_CLASSES) == [
        md1.CLASS_ALWAYS_CORRECT,
        md1.CLASS_CHURN,
        md1.CLASS_PERSISTENT,
        md1.CLASS_NEW_PERSISTENT,
        md1.CLASS_TRANSIENT_BORN,
        md1.CLASS_HEALED,
    ]


def test_parser_defaults_point_at_the_cold_control_and_the_declared_store() -> None:
    args = md1.build_parser().parse_args(["--mode", "sweep"])
    assert args.cell == md1.CELL_COLD
    assert args.store.endswith("ddm_md1_micro_macro")
    assert args.gt_lineage == "dali"
    assert args.threads == 4


def test_parser_refuses_an_unknown_gt_lineage() -> None:
    with pytest.raises(SystemExit):
        md1.build_parser().parse_args(["--mode", "analyze", "--gt-lineage", "mps"])


def test_module_does_not_write_under_any_cell_run_root() -> None:
    source = Path(md1.__file__).read_text(encoding="utf-8")
    # every write goes through `store`; the run roots are only ever read.
    for marker in ("DEFAULT_COLD", "DEFAULT_WARM"):
        assert marker in source
    assert 'store / f"sweep_rows_' in source
    assert 'payload_root = store / "payloads"' in source


def test_receipt_schema_declares_zero_paid_invocations() -> None:
    source = Path(md1.__file__).read_text(encoding="utf-8")
    for key in ("metal_invocations", "modal_invocations", "contest_eval_invocations"):
        assert f'"{key}": 0' in source


def test_json_round_trip_of_a_bridge_block() -> None:
    cube = np.asarray([[0, 1], [1, 1], [1, 0]], dtype=bool)
    codes = md1.classify_sites(cube, churn_flips=4, persistent_fraction=0.9)
    bridge = _bridge(cube, codes, 2, np.asarray([15.0, 30.0]))
    assert json.loads(json.dumps(bridge)) == bridge


# ---------------------------------------------------------------------------
# report + compare helpers
# ---------------------------------------------------------------------------
def test_first_crossing_returns_the_first_step_at_or_above_the_threshold() -> None:
    steps = [0, 16, 32, 48]
    values = [1.0, 1.02, 1.05, 1.09]
    assert md1.first_crossing(steps, values, 1.05) == 32
    assert md1.first_crossing(steps, values, 1.10) is None
    assert md1.first_crossing(steps, values, 1.0) == 0


def test_first_crossing_refuses_ragged_inputs_before_scanning() -> None:
    # the first element already crosses, so a lazy zip would return 0 and hide the truncation.
    with pytest.raises(md1.MD1Error):
        md1.first_crossing([0, 16], [1.0], 1.0)


def test_gt_area_fractions_is_ht_weighted_not_a_plain_mean() -> None:
    # two pairs, all-Road and all-Lane, with weights 15 and 30: the HT area must follow the
    # weights, so a plain mean would give 0.5/0.5 and is wrong.
    gt = np.zeros((2, 2, 2), dtype=np.uint8)
    gt[1] = 1
    areas = md1.gt_area_fractions(gt, np.asarray([15.0, 30.0]), 45)
    assert areas[0] == pytest.approx(15.0 / 45.0)
    assert areas[1] == pytest.approx(30.0 / 45.0)
    assert sum(areas) == pytest.approx(1.0)


def test_rare_class_ids_are_lane_and_movable_in_canonical_order() -> None:
    ar1 = pytest.importorskip("experiments.ddm_ar1_aa_render_price")
    assert [ar1.CLASS_NAMES[c] for c in md1.RARE_CLASS_IDS] == ["Lane", "Movable"]


def test_rows_for_returns_empty_when_the_cell_has_no_jsonl(tmp_path: Path) -> None:
    assert md1._rows_for(tmp_path, "absent_cell") == {}


def test_rows_for_groups_by_forward_and_keys_by_step(tmp_path: Path) -> None:
    path = tmp_path / "sweep_rows_cell.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"forward": f, "step": s, "d_seg_hat_dali": 0.1})
            for f, s in (("shadow", 0), ("shadow", 16), ("live", 16))
        )
        + "\n",
        encoding="utf-8",
    )
    grouped = md1._rows_for(tmp_path, "cell")
    assert sorted(grouped) == ["live", "shadow"]
    assert sorted(grouped["shadow"]) == [0, 16]
    assert sorted(grouped["live"]) == [16]


def test_milestone_reproduction_skips_steps_without_a_retained_milestone(tmp_path: Path) -> None:
    payload_root = tmp_path / "payloads" / "cell"
    payload_root.mkdir(parents=True)
    md1.atomic_npz(
        payload_root / "shadow_step_000000.npz",
        argmax_u8=np.zeros((1, 2, 2), dtype=np.uint8),
        band_u8=np.zeros((1, 2, 2), dtype=np.uint8),
    )
    rows = md1.milestone_reproduction(tmp_path, "cell", tmp_path / "run", (0, 1000), [4])
    assert rows == []


def test_milestone_reproduction_counts_disagreeing_sites(tmp_path: Path) -> None:
    payload_root = tmp_path / "payloads" / "cell"
    payload_root.mkdir(parents=True)
    mine = np.zeros((1, 2, 2), dtype=np.uint8)
    mine[0, 0, 0] = 1
    md1.atomic_npz(
        payload_root / "shadow_step_000000.npz",
        argmax_u8=mine,
        band_u8=np.zeros((1, 2, 2), dtype=np.uint8),
    )
    realized = tmp_path / "run" / "milestones" / "step_000000" / "realized"
    realized.mkdir(parents=True)
    md1.atomic_npz(
        realized / "pair_0004.npz",
        segnet_argmax_u8=np.zeros((2, 2), dtype=np.uint8),
        target_argmax_u8=np.zeros((2, 2), dtype=np.uint8),
    )
    rows = md1.milestone_reproduction(tmp_path, "cell", tmp_path / "run", (0,), [4])
    assert len(rows) == 1
    assert rows[0]["cpu_vs_retained_mps_differing_sites"] == 1
    assert rows[0]["sites_compared"] == 4
    assert rows[0]["d_seg_per_pair_cpu"] == [0.25]
    assert rows[0]["d_seg_per_pair_retained_mps"] == [0.0]


def test_compare_reads_born_sites_out_of_the_trajectory_code(tmp_path: Path) -> None:
    # code = at_zero + 2*at_peak + 4*terminal.  BORN == correct at 0 AND wrong at the peak.
    cold = np.asarray([0, 2, 6, 3, 1], dtype=np.uint8).reshape(1, 1, 5)
    warm = np.asarray([0, 2, 0, 0, 0], dtype=np.uint8).reshape(1, 1, 5)
    md1.atomic_npz(
        tmp_path / f"excursion_{md1.CELL_COLD}_shadow_dali.npz",
        trajectory_code_u8=cold,
        peak_step=np.asarray([2048], dtype=np.int64),
    )
    md1.atomic_npz(
        tmp_path / f"excursion_{md1.CELL_WARM}_shadow_dali.npz",
        trajectory_code_u8=warm,
        peak_step=np.asarray([1024], dtype=np.int64),
    )
    args = md1.build_parser().parse_args(
        ["--mode", "compare", "--store", str(tmp_path), "--gt-lineage", "dali"]
    )
    out = md1.run_compare(args)
    block = out["forwards"]["shadow"]
    assert block["cold_born_sites"] == 2  # codes 2 and 6
    assert block["warm_born_sites"] == 1  # code 2
    assert block["intersection_sites"] == 1
    assert block["warm_only_sites"] == 0
    assert block["warm_born_absent_from_cold_fraction"] == 0.0
    assert block["cold_peak_step"] == 2048
    assert block["warm_peak_step"] == 1024


def test_compare_emits_nothing_when_a_cell_is_missing(tmp_path: Path) -> None:
    args = md1.build_parser().parse_args(["--mode", "compare", "--store", str(tmp_path)])
    assert md1.run_compare(args)["forwards"] == {}


def test_optimizer_moment_norms_reports_params_that_never_received_a_gradient() -> None:
    torch = pytest.importorskip("torch")
    names = ["params.render_in_w", "params.pose_in_w", "params.pose_out_w"]
    state = {
        "state": {0: {"exp_avg": torch.ones(2), "exp_avg_sq": torch.full((2,), 4.0), "step": torch.tensor(16)}},
        "param_groups": [{"params": [0, 1, 2]}],
    }
    row = md1._optimizer_moment_norms(state, names, lambda n: "pose_head" if "pose" in n else "rgb_renderer")
    assert row["params_without_optimizer_state"] == ["params.pose_in_w", "params.pose_out_w"]
    assert row["roles_without_optimizer_state"] == ["pose_head"]
    assert row["exp_avg_l2_by_role"]["rgb_renderer"] == pytest.approx(2.0**0.5)
    assert row["exp_avg_sq_l1_by_role"]["rgb_renderer"] == pytest.approx(8.0)
    assert row["adam_step_values"] == [16]


def test_optimizer_moment_norms_refuses_a_multi_group_optimizer() -> None:
    pytest.importorskip("torch")
    with pytest.raises(md1.MD1Error):
        md1._optimizer_moment_norms({"state": {}, "param_groups": [{"params": []}, {"params": []}]}, [], lambda n: n)


def test_optimizer_moment_norms_refuses_a_param_order_length_mismatch() -> None:
    pytest.importorskip("torch")
    with pytest.raises(md1.MD1Error):
        md1._optimizer_moment_norms({"state": {}, "param_groups": [{"params": [0, 1]}]}, ["a"], lambda n: n)


def test_displacement_norms_refuses_a_predecessor_missing_a_tensor() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(md1.MD1Error):
        md1._displacement_norms({"a": torch.zeros(2)}, {}, lambda n: n)


def test_displacement_norms_groups_by_role_and_is_an_l2_over_the_group() -> None:
    torch = pytest.importorskip("torch")
    current = {"x": torch.tensor([3.0]), "y": torch.tensor([4.0])}
    previous = {"x": torch.tensor([0.0]), "y": torch.tensor([0.0])}
    out = md1._displacement_norms(current, previous, lambda n: "same")
    assert out["same"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# tables renderer (the memo's numbers are rendered, never retyped)
# ---------------------------------------------------------------------------
def test_fmt_is_significant_digits_not_fixed_decimals() -> None:
    assert md1._fmt(0.0025193532307942, 6) == "0.00251935"
    assert md1._fmt(1.0, 6) == "1"
    assert md1._fmt(1234567.0, 4) == "1.235e+06"


def test_render_tables_returns_empty_when_the_store_has_nothing(tmp_path: Path) -> None:
    assert md1.render_tables(tmp_path, "dali") == ""


def test_render_tables_renders_the_report_trajectory_and_the_birth_row(tmp_path: Path) -> None:
    pytest.importorskip("experiments.ddm_ar1_aa_render_price")
    report = {
        "cells": {
            "cell_x": {
                "run_root": "/nowhere",
                "forwards": {
                    "shadow": {
                        "steps": [0, 16],
                        "d_seg_hat_dali": [0.001, 0.002],
                        "d_seg_hat_pyav": [0.0011, 0.0021],
                        "pose_mse_hat": [1e-4, 2e-4],
                        "predicted_over_gt_area_ratio": {
                            "Road": [1.0, 1.0],
                            "Lane": [1.0, 1.2],
                            "Undrivable": [1.0, 1.0],
                            "Movable": [1.0, 1.0],
                            "MyCar": [1.0, 1.0],
                        },
                        "overpaint_birth_step": {"Lane": 16, "Movable": None},
                    }
                },
                "live_minus_shadow": {
                    "steps": [16],
                    "d_seg_hat_dali_live": [0.006],
                    "d_seg_hat_dali_shadow": [0.002],
                    "delta": [0.004],
                    "ratio": [3.0],
                },
                "milestone_reproduction": [
                    {
                        "step": 0,
                        "sites_compared": 6291456,
                        "cpu_vs_retained_mps_differing_sites": 51,
                        "cpu_vs_retained_mps_site_fraction": 8.106e-6,
                    }
                ],
            }
        },
        "warm_minus_cold": {
            "shadow": {
                "steps": [16],
                "cold_d_seg_hat_dali": [0.002],
                "warm_d_seg_hat_dali": [0.0018],
                "warm_minus_cold": [-0.0002],
                "cold_displacement_total": [0.055],
                "warm_displacement_total": [0.008],
            }
        },
    }
    (tmp_path / "REPORT.json").write_text(json.dumps(report), encoding="utf-8")
    out = md1.render_tables(tmp_path, "dali")
    assert "cell_x" in out
    assert '"Lane": 16' in out
    assert "live minus EMA shadow" in out
    assert "CPU reconstruction vs the retained MPS argmax" in out
    assert "warm minus cold at IDENTICAL steps" in out
    assert "51" in out


def test_render_tables_renders_the_class_table_and_the_calibration_gate(tmp_path: Path) -> None:
    pytest.importorskip("experiments.ddm_ar1_aa_render_price")
    classes = {
        name: {
            "sites": 10,
            "site_fraction": 0.1,
            "terminal_wrong_sites": 2,
            "terminal_d_seg_contribution": 0.0005,
            "terminal_share_of_error": 0.25,
            "gt_class_histogram": [1, 2, 3, 4, 0],
        }
        for name in md1.SITE_CLASSES
    }
    analysis = {
        "cell": "cell_x",
        "forwards": {
            "shadow": {
                "bridge": {
                    "calibration_gate_max_abs_integer_residual": 0,
                    "calibration_gate_exact_zero": True,
                },
                "terminal_d_seg_hat": 0.00275,
                "classes": classes,
                "terminal_edges": {name: {"Road->Lane": 7} for name in md1.ERROR_CLASSES},
                "terminal_bands": {name: [1, 2, 3, 4] for name in md1.SITE_CLASSES},
                "excursion": {
                    "peak_step": 2048,
                    "d_seg_hat_at_zero": 0.0025,
                    "d_seg_hat_at_peak": 0.0032,
                    "d_seg_hat_terminal": 0.00275,
                    "born_sites": 100,
                    "born_recovered_fraction": 0.5,
                    "healed_by_peak_sites": 20,
                    "rare_overpaint_sites_at_peak": 40,
                    "rare_overpaint_share_of_peak_error": 0.4,
                    "rare_overpaint_born_fraction": 0.9,
                    "rare_overpaint_recovered_fraction": 0.3,
                    "reachability": {
                        "target_d_seg": md1.SUB_012_DSEG_TARGET,
                        "target_source": md1.SUB_012_DSEG_TARGET_SOURCE,
                        "terminal_d_seg_hat": 0.00275,
                        "terminal_over_target": 20.15,
                        "persistent_floor_d_seg_hat": 0.0021,
                        "persistent_floor_over_target": 15.39,
                        "optimizer_reachable_d_seg_hat": 0.00065,
                        "optimizer_reachable_share": 0.2364,
                        "note": "floor",
                    },
                },
            }
        },
    }
    (tmp_path / "ANALYSIS_cell_x_dali.json").write_text(json.dumps(analysis), encoding="utf-8")
    out = md1.render_tables(tmp_path, "dali")
    assert "calibration gate max |Σclasses − total| = **0**" in out
    assert "exact zero: **True**" in out
    assert "peak at step **2048**" in out
    assert "20.15x** the sub-0.12 target" in out
    assert "23.64%** of the terminal error is optimizer-reachable" in out
    assert "Road->Lane 7" in out
    for name in md1.SITE_CLASSES:
        assert name in out


def test_render_tables_skips_an_analysis_of_a_different_gt_lineage(tmp_path: Path) -> None:
    pytest.importorskip("experiments.ddm_ar1_aa_render_price")
    (tmp_path / "ANALYSIS_cell_x_pyav.json").write_text(
        json.dumps({"cell": "cell_x", "forwards": {}}), encoding="utf-8"
    )
    assert "cell_x" not in md1.render_tables(tmp_path, "dali")


def test_first_step_at_or_below_finds_the_sign_change() -> None:
    assert md1.first_step_at_or_below([0, 16, 32, 48], [0.004, 0.002, -0.001, 0.003]) == 32
    assert md1.first_step_at_or_below([0, 16], [0.004, 0.002]) is None
    assert md1.first_step_at_or_below([0, 16], [0.0, 0.002]) == 0


def test_first_step_at_or_below_refuses_ragged_inputs() -> None:
    with pytest.raises(md1.MD1Error):
        md1.first_step_at_or_below([0, 16], [0.0])


def test_sub_012_target_is_transferred_with_its_source_and_never_re_derived() -> None:
    assert md1.SUB_012_DSEG_TARGET == 1.3646784205e-4
    assert md1.SUB_012_DSEG_TARGET_SOURCE.startswith(".omx/research/ddm_qn1_")
    source = Path(md1.__file__).read_text(encoding="utf-8")
    # the target must appear exactly once as a constant, never re-typed inside a computation
    assert source.count("1.3646784205e-4") == 1


def test_milestone_reproduction_adds_the_ht_calibration_when_weights_are_given(tmp_path: Path) -> None:
    payload_root = tmp_path / "payloads" / "cell"
    payload_root.mkdir(parents=True)
    mine = np.zeros((2, 2, 2), dtype=np.uint8)
    mine[0, 0, 0] = 1
    md1.atomic_npz(
        payload_root / "shadow_step_000000.npz",
        argmax_u8=mine,
        band_u8=np.zeros((2, 2, 2), dtype=np.uint8),
    )
    realized = tmp_path / "run" / "milestones" / "step_000000" / "realized"
    realized.mkdir(parents=True)
    for pair in (4, 31):
        md1.atomic_npz(
            realized / f"pair_{pair:04d}.npz",
            segnet_argmax_u8=np.zeros((2, 2), dtype=np.uint8),
            target_argmax_u8=np.zeros((2, 2), dtype=np.uint8),
        )
    (tmp_path / "run" / "milestones" / "step_000000" / "MILESTONE.json").write_text(
        json.dumps({"d_seg_hat": 0.0}), encoding="utf-8"
    )
    rows = md1.milestone_reproduction(
        tmp_path, "cell", tmp_path / "run", (0,), [4, 31], np.asarray([15.0, 30.0]), 600
    )
    assert rows[0]["d_seg_hat_cpu_pyav"] == pytest.approx(15.0 * 0.25 / 600.0)
    assert rows[0]["d_seg_hat_retained_mps_pyav"] == 0.0
    assert rows[0]["d_seg_hat_relative_gap"] is None
    assert rows[0]["recomputed_minus_recorded"] == 0.0


def test_milestone_reproduction_omits_the_ht_block_without_weights(tmp_path: Path) -> None:
    payload_root = tmp_path / "payloads" / "cell"
    payload_root.mkdir(parents=True)
    md1.atomic_npz(
        payload_root / "shadow_step_000000.npz",
        argmax_u8=np.zeros((1, 2, 2), dtype=np.uint8),
        band_u8=np.zeros((1, 2, 2), dtype=np.uint8),
    )
    realized = tmp_path / "run" / "milestones" / "step_000000" / "realized"
    realized.mkdir(parents=True)
    md1.atomic_npz(
        realized / "pair_0004.npz",
        segnet_argmax_u8=np.zeros((2, 2), dtype=np.uint8),
        target_argmax_u8=np.zeros((2, 2), dtype=np.uint8),
    )
    rows = md1.milestone_reproduction(tmp_path, "cell", tmp_path / "run", (0,), [4])
    assert "d_seg_hat_cpu_pyav" not in rows[0]


def test_reachability_share_is_derived_from_the_integer_numerator(tmp_path: Path) -> None:
    # the share must be an exact integer ratio, so it can never disagree with the bridge row.
    cube = np.zeros((3, 4), dtype=bool)
    cube[:, 0] = True  # PERSISTENT
    cube[1:, 1] = True  # NEW_PERSISTENT
    codes = md1.classify_sites(cube, churn_flips=4, persistent_fraction=0.90)
    bridge = _bridge(cube, codes, 2, np.asarray([15.0, 15.0]))
    persistent = bridge["numerator_by_class"][md1.CLASS_PERSISTENT][-1]
    total = bridge["weighted_wrong_site_numerator_by_step"][-1]
    assert persistent > 0 and total > persistent
    # PERSISTENT holds site 0 (pair 0), NEW_PERSISTENT holds site 1 (pair 0): both weight 15
    assert persistent == 15
    assert total == 30


def test_analyze_refuses_a_cell_with_no_swept_rows(tmp_path: Path) -> None:
    args = md1.build_parser().parse_args(
        ["--mode", "analyze", "--cell", "absent_cell", "--store", str(tmp_path)]
    )
    with pytest.raises(md1.MD1Error):
        md1.run_analyze(args)


def test_analyze_refuses_an_empty_rows_file(tmp_path: Path) -> None:
    (tmp_path / "sweep_rows_empty_cell.jsonl").write_text("\n", encoding="utf-8")
    args = md1.build_parser().parse_args(
        ["--mode", "analyze", "--cell", "empty_cell", "--store", str(tmp_path)]
    )
    with pytest.raises(md1.MD1Error):
        md1.run_analyze(args)
