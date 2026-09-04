"""Falsifiers for the ddm_fs1 frame-0 selector splice.

Two classes of test live here.

STORE-FREE tests exercise the adoption arithmetic and the refusal paths.  They run
everywhere and they are the ones that protect the selection logic from silently
buying the wrong set.

STORE-BOUND tests need the retained afr1 body on the SSD tier.  The headline one --
``test_identity_control_reproduces_the_shipped_archive`` -- is the control the whole
byte-close rests on: rebuilding the shipped body through this module's own writer,
with the selector tail UNCHANGED, must reproduce ``archive.zip`` bit-for-bit.  If it
cannot, the +20 B this arm reports would be a mixture of the selector and the
container, and no delta would be attributable.  They skip when the store is absent
rather than passing vacuously ([[m50]]: report the denominator).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "experiments" / "ddm_fs1_frame0_selector_reselection.py"
AFR1_RUNTIME = Path(
    "/Volumes/APDataStore/pact/ddm_pr1_pose_resolve/runtime_afr1"
)
AFR1_ARCHIVE = AFR1_RUNTIME / "archive.zip"
PR1_SWEEP = Path(
    "/Volumes/APDataStore/pact/ddm_pr1_pose_resolve/retained/"
    "selector_sweep_base_shipped_n600.json"
)
#: ddm_pr1 retained/measure_base_shipped_codes_n600.json -- batch 8, n600, DALI GT.
PR1_BASE_D_POSE = 6.3656845167356244e-06


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "ddm_fs1_frame0_selector_reselection", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fs1 = _load_module()

store_bound = pytest.mark.skipif(
    not AFR1_ARCHIVE.is_file(),
    reason=f"retained afr1 body absent: {AFR1_ARCHIVE}",
)
sweep_bound = pytest.mark.skipif(
    not PR1_SWEEP.is_file(), reason=f"pr1 selector sweep absent: {PR1_SWEEP}"
)


# --------------------------------------------------------------------------
# STORE-FREE: the score arithmetic and the adoption rules.
# --------------------------------------------------------------------------


def test_composed_score_reproduces_the_afr1_receipt_from_components():
    """#877: recompute S from the legs; never read the report's 2-dp display."""
    recomputed = fs1.composed_score(
        fs1.AFR1_D_SEG_T4, fs1.AFR1_D_POSE_T4, fs1.FRONTIER_ARCHIVE_BYTES
    )
    assert recomputed == pytest.approx(fs1.AFR1_SCORE_T4, rel=0, abs=5e-9)


def test_rate_leg_denominator_is_the_upstream_source_size():
    """``25 * bytes / uncompressed`` must return the receipt's own rate leg."""
    rate = 25.0 * fs1.FRONTIER_ARCHIVE_BYTES / fs1.UNCOMPRESSED_SOURCE_BYTES
    assert rate == pytest.approx(0.11985594327989708, rel=0, abs=1e-12)


def _synthetic_sweep(rows):
    return {"rows": rows, "batch_size": 1, "pairs_swept": list(range(fs1.FRAME_COUNT))}


def _row(pair, shipped, best, d_shipped, d_best):
    return {
        "pair": pair,
        "shipped_mode": shipped,
        "best_mode": best,
        "d_pose_at_shipped_mode": d_shipped,
        "d_pose_at_best_mode": d_best,
        "gain": d_shipped - d_best,
        "ratio": d_shipped / d_best if d_best > 0 else float("inf"),
    }


def _flat_rows():
    return [_row(p, 0, 0, 1e-6, 1e-6) for p in range(fs1.FRAME_COUNT)]


def test_adoption_refuses_a_sweep_taken_on_a_different_body():
    rows = _flat_rows()
    rows[7] = _row(7, 3, 5, 2e-6, 1e-6)
    shipped = np.zeros(fs1.FRAME_COUNT, dtype=np.uint8)  # pair 7 is NOT active here
    with pytest.raises(fs1.Fs1Error, match="different body"):
        fs1.adoption_from_sweep(
            _synthetic_sweep(rows), margin=1.01, shipped_choices=shipped
        )


def test_ratio_gate_admits_only_rows_above_the_margin():
    rows = _flat_rows()
    rows[1] = _row(1, 0, 2, 1.001e-6, 1.0e-6)  # ratio 1.001 -- below the gate
    rows[2] = _row(2, 0, 3, 2.0e-6, 1.0e-6)  # ratio 2.0 -- above
    shipped = np.zeros(fs1.FRAME_COUNT, dtype=np.uint8)
    shipped[0] = 4  # one shipped-active pair so the blob stays encodable
    rows[0] = _row(0, 4, 4, 1e-6, 1e-6)
    adopted = fs1.adoption_from_sweep(
        _synthetic_sweep(rows), margin=1.01, shipped_choices=shipped
    )
    assert [c["pair"] for c in adopted["changed_pairs"]] == [2]
    assert adopted["active_pairs"] == 2


def test_byte_optimal_never_loses_to_the_ratio_gate_on_the_same_admissible_set():
    """The frontier scan is exact, so it must dominate the take-everything rule."""
    rng = np.random.default_rng(11)
    rows = _flat_rows()
    shipped = np.zeros(fs1.FRAME_COUNT, dtype=np.uint8)
    shipped[0] = 4
    rows[0] = _row(0, 4, 4, 1e-6, 1e-6)
    for pair in range(1, 60):
        base = float(rng.uniform(1e-7, 5e-6))
        rows[pair] = _row(pair, 0, int(rng.integers(1, 8)), base, base / rng.uniform(1.02, 4.0))
    sweep = _synthetic_sweep(rows)
    gate = fs1.adoption_from_sweep(sweep, margin=1.01, shipped_choices=shipped)
    optimal = fs1.adoption_from_sweep(
        sweep,
        margin=1.01,
        shipped_choices=shipped,
        strategy="byte_optimal",
        base_d_pose=1e-5,
    )
    gate_net = fs1.price_adoption(gate, base_d_pose=1e-5)["net_delta_S"]
    optimal_net = fs1.price_adoption(optimal, base_d_pose=1e-5)["net_delta_S"]
    assert optimal_net <= gate_net


def test_byte_optimal_always_takes_a_deactivation_because_it_is_free():
    """Turning pair 85 OFF carries gain AND lowers the active count; never skip it."""
    rows = _flat_rows()
    rows[85] = _row(85, 3, 0, 4e-6, 2e-6)
    shipped = np.zeros(fs1.FRAME_COUNT, dtype=np.uint8)
    shipped[85] = 3
    shipped[9] = 4
    rows[9] = _row(9, 4, 4, 1e-6, 1e-6)
    adopted = fs1.adoption_from_sweep(
        _synthetic_sweep(rows),
        margin=1.01,
        shipped_choices=shipped,
        strategy="byte_optimal",
        base_d_pose=1e-5,
    )
    assert [c["pair"] for c in adopted["changed_pairs"]] == [85]
    assert adopted["deactivated"] == 1
    assert adopted["delta_bytes"] <= 0


def test_byte_optimal_refuses_without_a_base_d_pose():
    with pytest.raises(fs1.Fs1Error, match="base_d_pose"):
        fs1.adoption_from_sweep(
            _synthetic_sweep(_flat_rows()),
            margin=1.01,
            shipped_choices=np.zeros(fs1.FRAME_COUNT, dtype=np.uint8),
            strategy="byte_optimal",
        )


def test_unknown_strategy_refuses():
    with pytest.raises(fs1.Fs1Error, match="unknown adoption strategy"):
        fs1.adoption_from_sweep(
            _synthetic_sweep(_flat_rows()),
            margin=1.0,
            shipped_choices=np.zeros(fs1.FRAME_COUNT, dtype=np.uint8),
            strategy="whatever",
        )


def test_price_adoption_signs_are_the_score_function_and_not_a_convention():
    adoption = {"delta_bytes": 20, "n600_mean_d_pose_gain": 2.0e-7}
    priced = fs1.price_adoption(adoption, base_d_pose=6.3656845167356244e-06)
    assert priced["delta_S_pose"] < 0.0  # less pose distortion lowers S
    assert priced["delta_S_rate"] > 0.0  # more bytes raise S
    assert priced["delta_S_rate"] == pytest.approx(
        25.0 * 20 / fs1.UNCOMPRESSED_SOURCE_BYTES, rel=0, abs=1e-18
    )
    assert priced["net_delta_S"] == pytest.approx(
        priced["delta_S_pose"] + priced["delta_S_rate"], rel=0, abs=1e-18
    )


def _receiver_source(sha: str, size: int) -> str:
    return (
        "# a stand-in for the public entrypoint's pin block\n"
        f'ARCHIVE_SHA256 = "{sha}"\n'
        f"ARCHIVE_BYTES = {size:_d}\n"
        "def main():\n"
        "    return 0\n"
    )


def test_receiver_pin_diff_accepts_exactly_the_two_rewritten_constants(tmp_path):
    source = tmp_path / "before.py"
    target = tmp_path / "after.py"
    source.write_text(_receiver_source("a" * 64, 180_002), encoding="utf-8")
    target.write_text(_receiver_source("b" * 64, 180_022), encoding="utf-8")
    verdict = fs1._receiver_pin_only_diff(source, target)
    assert verdict["pin_constants_only"] is True
    assert [row["line"] for row in verdict["changed_lines"]] == [2, 3]


def test_receiver_pin_diff_refuses_a_change_outside_the_pin_block(tmp_path):
    """The one thing this guard exists to catch: a receiver edit smuggled in beside
    a legitimate re-pin.  ``repin_receiver`` is trusted to touch two lines; that
    trust is CHECKED here rather than asserted in a comment."""
    source = tmp_path / "before.py"
    target = tmp_path / "after.py"
    source.write_text(_receiver_source("a" * 64, 180_002), encoding="utf-8")
    target.write_text(
        _receiver_source("b" * 64, 180_022).replace("return 0", "return 1"),
        encoding="utf-8",
    )
    with pytest.raises(fs1.Fs1Error, match="outside its pin constants"):
        fs1._receiver_pin_only_diff(source, target)


def test_receiver_pin_diff_refuses_a_changed_line_count(tmp_path):
    source = tmp_path / "before.py"
    target = tmp_path / "after.py"
    source.write_text(_receiver_source("a" * 64, 180_002), encoding="utf-8")
    target.write_text(
        _receiver_source("b" * 64, 180_022) + "EXTRA = 1\n", encoding="utf-8"
    )
    with pytest.raises(fs1.Fs1Error, match="changed line count"):
        fs1._receiver_pin_only_diff(source, target)


def test_receiver_pin_diff_refuses_an_unchanged_receiver(tmp_path):
    """A staged receiver that still pins the OLD archive would be refused by the
    shipped entrypoint at inflate time; catch it at stage time instead."""
    source = tmp_path / "before.py"
    target = tmp_path / "after.py"
    body = _receiver_source("a" * 64, 180_002)
    source.write_text(body, encoding="utf-8")
    target.write_text(body, encoding="utf-8")
    with pytest.raises(fs1.Fs1Error, match="exactly the two pin lines"):
        fs1._receiver_pin_only_diff(source, target)


def test_stage_noise_filter_covers_appledouble_and_pycache():
    assert fs1._is_stage_noise(Path("runtime/._f26_inflate.py"))
    assert fs1._is_stage_noise(Path("runtime/__pycache__/x.pyc"))
    assert fs1._is_stage_noise(Path(".DS_Store"))
    assert not fs1._is_stage_noise(Path("runtime/f26_inflate.py"))
    assert not fs1._is_stage_noise(Path("inflate.py"))


def test_build_refuses_when_the_container_identity_control_fails(monkeypatch, tmp_path):
    """A failed identity control must STOP the build, not annotate the report."""

    class _Body:
        archive_sha256 = fs1.FRONTIER_ARCHIVE_SHA256
        archive_size = fs1.FRONTIER_ARCHIVE_BYTES

    monkeypatch.setattr(fs1, "ShippedBody", lambda *a, **k: _Body())
    monkeypatch.setattr(
        fs1,
        "control_identity",
        lambda body: {"identical_to_shipped_archive": False},
    )
    args = fs1.build_parser().parse_args(
        [
            "build",
            "--archive",
            str(tmp_path / "a.zip"),
            "--runtime",
            str(tmp_path),
            "--choices",
            str(tmp_path / "c.npy"),
            "--out-dir",
            str(tmp_path / "out"),
            "--out",
            str(tmp_path / "r.json"),
        ]
    )
    with pytest.raises(fs1.Fs1Error, match="identity control FAILED"):
        fs1.run_build(args)


# --------------------------------------------------------------------------
# STORE-BOUND: the real afr1 body.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def body():
    if not AFR1_ARCHIVE.is_file():
        pytest.skip(f"retained afr1 body absent: {AFR1_ARCHIVE}")
    return fs1.ShippedBody(AFR1_ARCHIVE, AFR1_RUNTIME)


@store_bound
def test_the_retained_body_is_the_frontier_body(body):
    assert body.archive_sha256 == fs1.FRONTIER_ARCHIVE_SHA256
    assert body.archive_size == fs1.FRONTIER_ARCHIVE_BYTES


@store_bound
def test_selector_tail_is_invariant_through_the_rr5_dx2_and_cap1_layers(body):
    """``ShippedBody`` asserts this at construction; state it as its own falsifier."""
    assert body.receiver_tail_bytes == body.stored_tail_bytes
    assert len(body.stored_tail_bytes) == 9
    assert body.selector_blob == fs1.STORED_PREFIX + body.stored_tail_bytes


@store_bound
def test_identity_control_reproduces_the_shipped_archive(body):
    """THE control: unchanged tail in, byte-identical archive out."""
    control = fs1.control_identity(body)
    assert control["identical_to_shipped_archive"] is True
    assert control["rebuilt_sha256"] == fs1.FRONTIER_ARCHIVE_SHA256
    assert control["rebuilt_bytes"] == fs1.FRONTIER_ARCHIVE_BYTES


@store_bound
def test_a_spliced_archive_parses_back_to_the_requested_selector(body):
    from tac.semantic_pipeline.frame0_selector_codec import (
        encode_selector,
        stored_tail,
    )

    choices = body.selector_choices.copy()
    choices[85] = 0  # the actively harmful shipped op
    choices[5] = 6
    choices[500] = 2
    candidate = fs1.write_archive(
        body,
        stored_tail(encode_selector(choices)),
        quality=fs1.BROTLI_QUALITY,
        lgwin=fs1.BROTLI_LGWIN,
    )
    parsed = fs1.parse_back_parts(candidate, body.runtime_dir)
    assert np.array_equal(parsed.pop("selector_choices"), choices)
    base = fs1.parse_back_parts(body.archive_bytes_raw, body.runtime_dir)
    base.pop("selector_choices")
    for key in parsed:
        if key.startswith("selector_blob"):
            continue
        assert parsed[key] == base[key], f"section {key} changed under a selector splice"


@store_bound
def test_the_archive_byte_delta_equals_the_selector_blob_delta(body):
    """The rank is near-uniform, so brotli passes the tail through 1:1.

    MEASURED here rather than assumed: it is what makes the receiver's closed blob
    formula an exact archive price instead of a lower bound.
    """
    from tac.semantic_pipeline.frame0_selector_codec import (
        encode_selector,
        selector_blob_length,
        stored_tail,
    )

    rng = np.random.default_rng(3)
    for count in (12, 24, 42, 80):
        choices = np.zeros(fs1.FRAME_COUNT, dtype=np.uint8)
        positions = rng.choice(fs1.FRAME_COUNT, size=count, replace=False)
        choices[positions] = rng.integers(1, 8, size=count).astype(np.uint8)
        blob = encode_selector(choices)
        candidate = fs1.write_archive(
            body,
            stored_tail(blob),
            quality=fs1.BROTLI_QUALITY,
            lgwin=fs1.BROTLI_LGWIN,
        )
        expected = selector_blob_length(count) - selector_blob_length(5)
        assert len(candidate) - body.archive_size == expected, (
            f"k={count}: archive moved {len(candidate) - body.archive_size} B but the "
            f"blob formula says {expected} B"
        )


@store_bound
@sweep_bound
def test_pr1_margin_gate_reproduces_the_published_adoption(body):
    """pr1 Sec 12.1: 39 changed, 42 active, 50 B blob, +36 B, -1.032e-04 net dS."""
    sweep = json.loads(PR1_SWEEP.read_text(encoding="utf-8"))
    adoption = fs1.adoption_from_sweep(
        sweep, margin=1.01, shipped_choices=body.selector_choices
    )
    assert adoption["changed_count"] == 39
    assert adoption["active_pairs"] == 42
    assert adoption["blob_bytes"] == 50
    assert adoption["delta_bytes"] == 36
    assert adoption["n600_mean_d_pose_gain"] == pytest.approx(2.013298e-07, rel=1e-6)
    priced = fs1.price_adoption(adoption, base_d_pose=PR1_BASE_D_POSE)
    assert priced["net_delta_S"] == pytest.approx(-1.032126e-04, rel=1e-5)


@store_bound
@sweep_bound
def test_the_byte_optimum_lies_inside_pr1s_own_robustness_gate(body):
    """Every pair the exact frontier buys already clears pr1's >1% margin."""
    sweep = json.loads(PR1_SWEEP.read_text(encoding="utf-8"))
    ungated = fs1.adoption_from_sweep(
        sweep,
        margin=1.0,
        shipped_choices=body.selector_choices,
        strategy="byte_optimal",
        base_d_pose=PR1_BASE_D_POSE,
    )
    gated = fs1.adoption_from_sweep(
        sweep,
        margin=1.01,
        shipped_choices=body.selector_choices,
        strategy="byte_optimal",
        base_d_pose=PR1_BASE_D_POSE,
    )
    assert np.array_equal(ungated["choices"], gated["choices"])
    assert gated["min_adopted_ratio"] > 1.01
    assert gated["active_pairs"] == 24
    assert gated["delta_bytes"] == 20


@store_bound
def test_the_measured_selector_op_is_the_one_the_receiver_ships(body):
    """The pose instrument applies the selector in FLOAT; the archive applies it in
    INTEGER.  ``ddm_up2.apply_selector_float`` asserts the two agree "because every
    input is already an exact integer".  That is a docstring, so measure it: for all
    eight catalog modes, on uint8 frames that include both clamp boundaries, the
    float path must equal the shipped ``apply_pixel_mode`` exactly.  If it did not,
    every d_pose gain this arm reports would be measuring a different operator than
    the one the T4 will run.
    """
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(REPO / "experiments"))
    try:
        import ddm_up2_shipping_pose_solve as up2
    finally:
        sys.path.pop(0)

    modes = body.f0s.SPARSE_PIXEL_MODES
    assert len(modes) == 8
    rng = np.random.default_rng(41)
    # 3 frames of a small field, deliberately saturated at 0 and 255 so the clamp
    # branch is exercised rather than skipped.
    bhwc = rng.integers(0, 256, size=(3, 24, 32, 3), dtype=np.uint8)
    bhwc[0, :4, :4, :] = 0
    bhwc[1, :4, :4, :] = 255
    bchw = torch.from_numpy(bhwc.astype(np.float32)).permute(0, 3, 1, 2).contiguous()
    for index, mode in enumerate(modes):
        expected = body.f0s.apply_pixel_mode(bhwc.copy(), mode)
        got = up2.apply_selector_float(
            bchw.clone(), modes, np.full(3, index, dtype=np.uint8)
        )
        got_bhwc = got.permute(0, 2, 3, 1).numpy()
        assert np.array_equal(got_bhwc, expected.astype(np.float32)), (
            f"selector mode {index} (kind={mode.kind}, a={mode.a}, b={mode.b}, "
            f"c={mode.c}) differs between the float instrument and the shipped "
            "integer receiver"
        )


@store_bound
def test_blob_length_matches_the_shipped_body(body):
    from tac.semantic_pipeline.frame0_selector_codec import selector_blob_length

    active = int(np.count_nonzero(body.selector_choices))
    assert active == 5
    assert selector_blob_length(active) == len(body.selector_blob) == 14
    assert math.comb(fs1.FRAME_COUNT, active) > 0
