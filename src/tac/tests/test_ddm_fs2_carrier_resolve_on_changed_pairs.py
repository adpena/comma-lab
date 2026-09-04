"""Falsifiers for the ddm_fs2 carrier re-solve on the selector-changed pairs.

STORE-FREE tests exercise the scope derivation, the merge/adoption rule, the
score arithmetic and every refusal path.  They run everywhere.

STORE-BOUND tests need fs1's retained candidate-B body on the SSD tier.  The
headline one -- ``test_identity_control_reproduces_this_body`` -- is the control
the whole byte-close rests on: re-encoding the SHIPPED carrier codes through
``ddm_up3``'s writer at THIS body's container shape must reproduce
``archive.zip`` bit for bit.  If it cannot, the +1 B this arm reports would be a
mixture of the carrier and the container and no delta would be attributable.
They skip when the store is absent rather than passing vacuously ([[m50]]:
report the denominator).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "experiments" / "ddm_fs2_carrier_resolve_on_changed_pairs.py"

FS1_STORE = Path("/Volumes/APDataStore/pact/ddm_fs1_frame0_selector")
CANDB_RUNTIME = FS1_STORE / "measure_runtime_B_byte_optimal_101"
BASE_RUNTIME = FS1_STORE / "measure_runtime_BASE"

#: The 21 pairs whose frame-0 op moved between afr1 and fs1's candidate B.
#: Listed here as a FALSIFIER of the derivation, never as its input -- the module
#: computes them by diffing the two bodies' own selector vectors.
EXPECTED_CHANGED_PAIRS = [
    5, 70, 71, 77, 85, 95, 161, 173, 221, 259, 372,
    436, 479, 488, 504, 514, 518, 547, 555, 585, 586,
]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "ddm_fs2_carrier_resolve_on_changed_pairs", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fs2():
    return _load_module()


def _bodies_present() -> bool:
    return (CANDB_RUNTIME / "archive.zip").is_file() and (
        BASE_RUNTIME / "archive.zip"
    ).is_file()


store = pytest.mark.skipif(
    not _bodies_present(), reason="fs1 candidate-B store is not mounted"
)


# ---------------------------------------------------------------- constants


def test_module_loads_and_exposes_the_documented_surface(fs2):
    for name in (
        "changed_selector_pairs", "parse_pairs_argument", "run_solve",
        "run_codes", "run_build", "run_compose", "build_parser", "main",
    ):
        assert hasattr(fs2, name), name


def test_body_constants_match_the_fs1_pointer_receipt(fs2):
    # The twenty-fourth pointer move, recomputed from components (#877).
    assert fs2.FS1_ARCHIVE_BYTES == 180_022
    assert len(fs2.FS1_ARCHIVE_SHA256) == 64
    composed = fs2.composed_score(
        fs2.FS1_D_SEG_T4, fs2.FS1_D_POSE_T4, fs2.FS1_ARCHIVE_BYTES
    )
    assert composed == pytest.approx(fs2.FS1_SCORE_T4, rel=0, abs=1e-15)


def test_the_base_body_is_the_afr1_frontier(fs2):
    assert fs2.BASE_ARCHIVE_BYTES == 180_002
    assert fs2.BASE_ARCHIVE_SHA256 != fs2.FS1_ARCHIVE_SHA256


def test_byte_to_score_is_the_contest_exchange_rate(fs2):
    assert fs2.BYTE_TO_SCORE == 25.0 / 37_545_489.0
    assert fs2.rate_leg(180_022) == pytest.approx(0.11986926045895953, abs=1e-15)


def test_pose_leg_is_the_scoring_functions_own_form(fs2):
    assert fs2.pose_leg(6.169860284911831e-06) == pytest.approx(
        0.007854845819563762, abs=1e-15
    )
    assert fs2.pose_leg(0.0) == 0.0


def test_container_shape_is_this_bodys_not_up3s_default(fs2):
    # ddm_fs1 Sec 3.3 MEASURED q=9/lgwin=16 as this body's minimum; up3's module
    # default is q=11/lgwin=24, which costs 2 bytes MORE here.
    assert (fs2.BROTLI_QUALITY, fs2.BROTLI_LGWIN) == (9, 16)
    assert fs2.CONTAINER_OPTIONS[0] == (9, 16)
    assert (11, 24) in fs2.CONTAINER_OPTIONS  # measured, never assumed worse


def test_shipped_container_carries_the_bodys_ck2_bit(fs2):
    class Body:
        ck2_carrier = False

    assert fs2._shipped_container(Body()) == ((False, 9, 16),)
    Body.ck2_carrier = True
    assert fs2._shipped_container(Body()) == ((True, 9, 16),)


# ---------------------------------------------------------------- scope


def test_parse_pairs_defaults_to_the_whole_diff(fs2):
    derived = np.array([5, 70, 85], dtype=np.int64)
    assert fs2.parse_pairs_argument(None, derived).tolist() == [5, 70, 85]


def test_parse_pairs_accepts_a_subset_in_either_separator(fs2):
    derived = np.array([5, 70, 85], dtype=np.int64)
    assert fs2.parse_pairs_argument("70 5", derived).tolist() == [5, 70]
    assert fs2.parse_pairs_argument("70,5", derived).tolist() == [5, 70]


def test_parse_pairs_refuses_a_pair_outside_the_diff(fs2):
    derived = np.array([5, 70], dtype=np.int64)
    with pytest.raises(fs2.Fs2Error, match="not in the selector diff"):
        fs2.parse_pairs_argument("5 99", derived)


def test_changed_pairs_refuses_a_wrong_length_selector(fs2, monkeypatch):
    monkeypatch.setattr(
        fs2, "selector_choices_of", lambda runtime, sha: np.zeros(7, dtype=np.int64)
    )
    with pytest.raises(fs2.Fs2Error, match="expected"):
        fs2.changed_selector_pairs(Path("a"), Path("b"))


def test_changed_pairs_is_a_diff_not_an_activity_test(fs2, monkeypatch):
    base = np.zeros(fs2.N_PAIRS, dtype=np.int64)
    cand = np.zeros(fs2.N_PAIRS, dtype=np.int64)
    base[60] = 4          # active in both, SAME mode -> not stale
    cand[60] = 4
    base[85] = 3          # active -> identity: stale
    cand[5] = 6           # identity -> active: stale
    calls = {"n": 0}

    def fake(runtime, sha):
        calls["n"] += 1
        return base if calls["n"] == 1 else cand

    monkeypatch.setattr(fs2, "selector_choices_of", fake)
    pairs, receipt = fs2.changed_selector_pairs(Path("a"), Path("b"))
    assert pairs.tolist() == [5, 85]
    assert receipt["changed_count"] == 2
    assert receipt["transitions"][0] == {"pair": 5, "from_mode": 0, "to_mode": 6}
    assert receipt["transitions"][1] == {"pair": 85, "from_mode": 3, "to_mode": 0}
    assert 60 in receipt["base_active"] and 60 in receipt["candidate_active"]


def test_selector_read_refuses_an_unidentified_body(fs2, tmp_path):
    runtime = tmp_path / "rt"
    runtime.mkdir()
    (runtime / "archive.zip").write_bytes(b"not an archive")
    with pytest.raises(fs2.Fs2Error, match="refusing to read a selector"):
        fs2.selector_choices_of(runtime, "0" * 64)


# ---------------------------------------------------------------- resume


def test_load_done_is_resumable_and_survives_a_torn_line(fs2, tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(
        json.dumps({"pair": 5, "final_d_pose": 1.0}) + "\n"
        + "{not json\n"
        + "\n"
        + json.dumps({"pair": 70, "final_d_pose": 2.0}) + "\n",
        encoding="utf-8",
    )
    done = fs2.load_done(rows)
    assert sorted(done) == [5, 70]


def test_load_done_on_a_missing_file_is_empty_not_an_error(fs2, tmp_path):
    assert fs2.load_done(tmp_path / "absent.jsonl") == {}


def test_load_done_keeps_the_last_row_for_a_repeated_pair(fs2, tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(
        json.dumps({"pair": 5, "final_d_pose": 9.0}) + "\n"
        + json.dumps({"pair": 5, "final_d_pose": 1.0}) + "\n",
        encoding="utf-8",
    )
    assert fs2.load_done(rows)[5]["final_d_pose"] == 1.0


def test_sha256_array_is_content_addressed(fs2):
    a = np.arange(12, dtype=np.int32)
    assert fs2.sha256_array(a) == hashlib.sha256(a.tobytes()).hexdigest()
    assert fs2.sha256_array(a) != fs2.sha256_array(a.astype(np.int64))


# ---------------------------------------------------------------- CLI


def test_cli_exposes_exactly_the_documented_subcommands(fs2):
    parser = fs2.build_parser()
    actions = [
        a for a in parser._actions if getattr(a, "choices", None) and a.dest == "command"
    ]
    assert actions and set(actions[0].choices) == {
        "solve", "codes", "build", "compose",
    }


def test_cli_solve_defaults_gate_the_two_named_bodies(fs2):
    args = fs2.build_parser().parse_args([
        "solve", "--runtime", "a", "--base-runtime", "b", "--gt-cache", "g",
        "--renderer", "r", "--tokens", "t", "--out", "o",
    ])
    assert args.expect_archive_sha256 == fs2.FS1_ARCHIVE_SHA256
    assert args.base_archive_sha256 == fs2.BASE_ARCHIVE_SHA256
    assert args.materiality_operating_point == fs2.FS1_D_POSE_ADVISORY_N600_BATCH8


def test_cli_build_writes_at_the_bodys_own_shape_by_default(fs2):
    args = fs2.build_parser().parse_args([
        "build", "--runtime", "a", "--codes", "c", "--out-dir", "d", "--out", "o",
    ])
    assert args.container_search is False
    assert args.expect_archive_sha256 == fs2.FS1_ARCHIVE_SHA256


def test_cli_codes_defaults_name_their_source(fs2):
    args = fs2.build_parser().parse_args([
        "codes", "--runtime", "a", "--rows", "r", "--out", "o",
    ])
    assert args.base_d_pose == fs2.FS1_D_POSE_ADVISORY_N600_BATCH8
    assert "measure_candB" in args.base_d_pose_source
    assert args.adopt_all is False


# ---------------------------------------------------------------- arithmetic


def test_the_solve_gain_enters_S_as_an_n600_mean_not_a_subset_mean(fs2):
    # 21 pairs gaining 2.0295e-05 in TOTAL is a 3.382e-08 n600 mean; quoting the
    # subset's own mean (9.66e-07) would overstate the score effect by 28.6x.
    total_gain = 2.0294604512804962e-05
    assert total_gain / fs2.N_PAIRS == pytest.approx(3.382434085467494e-08, rel=1e-12)
    assert total_gain / 21 / (total_gain / fs2.N_PAIRS) == pytest.approx(
        fs2.N_PAIRS / 21, rel=1e-12
    )


def test_the_measured_projection_clears_the_admit_bar(fs2):
    base = fs2.FS1_D_POSE_ADVISORY_N600_BATCH8
    candidate = base - 2.0294604512804962e-05 / fs2.N_PAIRS
    net = fs2.pose_leg(candidate) - fs2.pose_leg(base) + 1 * fs2.BYTE_TO_SCORE
    assert net < -2e-5
    assert net == pytest.approx(-2.0894606394851853e-05, rel=1e-9)


def test_one_byte_of_rice_payload_is_the_exchange_rate(fs2):
    one_byte = 1 * fs2.BYTE_TO_SCORE
    assert one_byte == pytest.approx(6.658589531221714e-07, rel=1e-12)


# ---------------------------------------------------------------- store-bound


@store
def test_the_scope_is_derived_from_the_two_bodies_and_it_is_the_21(fs2):
    pairs, receipt = fs2.changed_selector_pairs(BASE_RUNTIME, CANDB_RUNTIME)
    assert pairs.tolist() == EXPECTED_CHANGED_PAIRS
    assert receipt["changed_count"] == 21
    # fs1's own receipt: 5 active before, 24 after.
    assert len(receipt["base_active"]) == 5
    assert len(receipt["candidate_active"]) == 24


@store
def test_identity_control_reproduces_this_body(fs2):
    """The control the byte-close rests on, at THIS body's container shape."""
    sys.path.insert(0, str(REPO / "experiments"))
    try:
        import ddm_up3_carrier_splice as up3
    finally:
        sys.path.pop(0)
    shipped = (CANDB_RUNTIME / "archive.zip").read_bytes()
    body = up3.parse_shipped_body(CANDB_RUNTIME, verify_sha=False)
    built = up3.build_archive(
        body, body.codes, runtime_dir=CANDB_RUNTIME,
        container_options=fs2._shipped_container(body),
    )
    assert built["archive_sha256"] == hashlib.sha256(shipped).hexdigest()
    assert built["archive_size"] == len(shipped) == fs2.FS1_ARCHIVE_BYTES
    assert built["rice_payload_identical"] is True
    assert built["packed_metadata_identical"] is True
    # The pricer's own anchor: the shipped Rice payload, reproduced.
    assert built["rice_bits"] == 78_628
    assert built["container"]["identical_to_shipped_shape"] is True


@store
def test_up3s_module_default_container_would_cost_this_body_two_bytes(fs2):
    """The reason the shape is named here rather than inherited."""
    sys.path.insert(0, str(REPO / "experiments"))
    try:
        import ddm_up3_carrier_splice as up3
    finally:
        sys.path.pop(0)
    body = up3.parse_shipped_body(CANDB_RUNTIME, verify_sha=False)
    at_default = up3.build_archive(
        body, body.codes, runtime_dir=CANDB_RUNTIME,
        container_options=(
            (bool(body.ck2_carrier), up3.BROTLI_QUALITY, up3.BROTLI_LGWIN),
        ),
    )
    assert at_default["archive_size"] == fs2.FS1_ARCHIVE_BYTES + 2


@store
def test_the_solved_rows_are_all_at_a_receiver_refusal_not_a_tolerance(fs2):
    """Every re-solve row must stop because the receiver refused a real step."""
    summary = Path(
        "/Volumes/VertigoDataTier/pact/ddm_fs2_carrier_resolve/retained/"
        "solve_candB/SUMMARY.json"
    )
    if not summary.is_file():
        pytest.skip("the fs2 solve has not run in this store")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["schema"] == "tac.ddm_fs2.solve.v1"
    assert payload["score_claim"] is False
    assert payload["pairs"] == 21
    assert payload["stop_reasons"] == {"no_improving_step": 21}
    assert payload["n600_mean_gain_d_pose"] == pytest.approx(
        payload["total_gain_d_pose"] / fs2.N_PAIRS, rel=1e-12
    )
    assert math.isclose(
        payload["total_start_d_pose"] - payload["total_final_d_pose"],
        payload["total_gain_d_pose"],
        rel_tol=1e-12,
    )
