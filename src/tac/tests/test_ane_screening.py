"""Contract tests for the ANE SCREENING backend (``tac.ane_screening``).

The invariant these tests exist to protect: a screening backend may RANK, and
it may never emit a number.  Every ADOPTED pair is re-measured on ``cpu_torch``
fp32 first, and the receipt carries both values, the backend name, and the
custody of the exact ``.mlpackage`` that produced the screened value.  If any
of that can be skipped silently, the backend is a fake measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.ane_screening import (
    AUTHORITY_BACKEND,
    SCORER_BACKENDS,
    SCREENING_BACKENDS,
    SEG_AUTHORITY_FLIP_BAR,
    AneScreeningError,
    assert_backend_name,
    assert_cpu_confirm_contract,
    backend_is_authority,
    load_pose_backend,
    mlpackage_provenance,
    rank_agreement,
    screening_receipt,
    sha256_tree,
    write_json,
)

# --------------------------------------------------------------- backend names


def test_backend_roster_is_exactly_three_and_names_the_authority() -> None:
    assert SCORER_BACKENDS == ("cpu_torch", "coreml_cpu_fp32", "ane_fp16_screen")
    assert AUTHORITY_BACKEND == "cpu_torch"
    assert set(SCREENING_BACKENDS) == set(SCORER_BACKENDS) - {AUTHORITY_BACKEND}


def test_only_cpu_torch_is_authority() -> None:
    assert backend_is_authority("cpu_torch") is True
    for name in SCREENING_BACKENDS:
        assert backend_is_authority(name) is False


def test_unknown_backend_fails_closed() -> None:
    with pytest.raises(AneScreeningError, match="unknown scorer backend"):
        assert_backend_name("ane_fp16")  # near-miss of a real name
    with pytest.raises(AneScreeningError):
        assert_backend_name("mps")


def test_authority_flip_bar_is_the_operator_value() -> None:
    # 2026-07-13 preregistration: SegNet argmax flip rate <= 0.0033%.
    assert pytest.approx(3.3e-5) == SEG_AUTHORITY_FLIP_BAR


# ------------------------------------------------------ the CPU-confirm contract


def test_cpu_torch_needs_no_confirm() -> None:
    verdict = assert_cpu_confirm_contract(
        backend="cpu_torch", adopted=[1, 2, 3], confirmed=None
    )
    assert verdict["cpu_confirm_required"] is False
    assert verdict["confirmed_pairs"] == 3


def test_screening_backend_without_any_confirm_raises() -> None:
    with pytest.raises(AneScreeningError, match="never re-measured"):
        assert_cpu_confirm_contract(
            backend="ane_fp16_screen", adopted=[7, 8], confirmed=[]
        )


def test_screening_backend_with_partial_confirm_raises_and_names_the_gap() -> None:
    with pytest.raises(AneScreeningError) as excinfo:
        assert_cpu_confirm_contract(
            backend="ane_fp16_screen", adopted=[1, 2, 3], confirmed=[1, 3]
        )
    assert "[2]" in str(excinfo.value)


def test_screening_backend_with_full_confirm_passes() -> None:
    verdict = assert_cpu_confirm_contract(
        backend="coreml_cpu_fp32", adopted=[4, 5], confirmed=[4, 5, 9]
    )
    assert verdict["cpu_confirm_required"] is True
    assert verdict["cpu_confirm_satisfied"] is True
    assert verdict["adopted_pairs"] == 2


def test_confirming_a_superset_is_allowed_but_adopting_a_superset_is_not() -> None:
    assert_cpu_confirm_contract(backend="ane_fp16_screen", adopted=[1], confirmed=[1, 2])
    with pytest.raises(AneScreeningError):
        assert_cpu_confirm_contract(
            backend="ane_fp16_screen", adopted=[1, 2], confirmed=[1]
        )


# ------------------------------------------------------------------- receipts


def test_receipt_carries_both_values_and_refuses_a_score_claim() -> None:
    receipt = screening_receipt(
        backend="ane_fp16_screen",
        screen_values={3: 1.0e-6, 5: 2.0e-6},
        confirm_values={3: 1.1e-6, 5: 2.0e-6},
        provenance={"mlpackage_sha256": "deadbeef"},
    )
    assert receipt["score_claim"] is False
    assert receipt["promotable"] is False
    assert receipt["scorer_backend"] == "ane_fp16_screen"
    assert receipt["authority_backend"] == "cpu_torch"
    assert receipt["provenance"]["mlpackage_sha256"] == "deadbeef"
    row = {r["pair"]: r for r in receipt["pairs"]}
    assert row[3]["screened_value"] == pytest.approx(1.0e-6)
    assert row[3]["confirmed_value"] == pytest.approx(1.1e-6)
    assert row[3]["abs_drift"] == pytest.approx(1.0e-7)
    assert row[5]["abs_drift"] == 0.0


def test_receipt_refuses_an_unconfirmed_adoption() -> None:
    with pytest.raises(AneScreeningError):
        screening_receipt(
            backend="ane_fp16_screen",
            screen_values={1: 0.5, 2: 0.25},
            confirm_values={1: 0.5},
        )


def test_receipt_is_json_serialisable_and_round_trips(tmp_path: Path) -> None:
    receipt = screening_receipt(
        backend="coreml_cpu_fp32",
        screen_values={0: 1.0},
        confirm_values={0: 1.0},
    )
    out = tmp_path / "receipt.json"
    digest = write_json(out, receipt)
    assert len(digest) == 64
    assert json.loads(out.read_text())["scorer_backend"] == "coreml_cpu_fp32"


# ----------------------------------------------------------------- provenance


def test_sha256_tree_hashes_a_file(tmp_path: Path) -> None:
    target = tmp_path / "a.bin"
    target.write_bytes(b"hello")
    # sha256("hello")
    assert sha256_tree(target) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_sha256_tree_is_content_addressed_not_mtime_addressed(tmp_path: Path) -> None:
    for name in ("one", "two"):
        package = tmp_path / name
        (package / "sub").mkdir(parents=True)
        (package / "sub" / "w.bin").write_bytes(b"\x01\x02")
        (package / "meta.json").write_text('{"k": 1}')
    assert sha256_tree(tmp_path / "one") == sha256_tree(tmp_path / "two")


def test_sha256_tree_changes_when_any_member_byte_changes(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "w.bin").write_bytes(b"\x01\x02")
    before = sha256_tree(package)
    (package / "w.bin").write_bytes(b"\x01\x03")
    assert sha256_tree(package) != before


def test_sha256_tree_refuses_a_missing_path(tmp_path: Path) -> None:
    with pytest.raises(AneScreeningError, match="cannot hash missing path"):
        sha256_tree(tmp_path / "nope")


def test_mlpackage_provenance_names_the_package_and_the_tool(tmp_path: Path) -> None:
    package = tmp_path / "m.mlpackage"
    package.mkdir()
    (package / "coremldata.bin").write_bytes(b"x")
    record = mlpackage_provenance(package)
    assert record["mlpackage"].endswith("m.mlpackage")
    assert len(record["mlpackage_sha256"]) == 64
    assert "coremltools_version" in record


# -------------------------------------------------------------------- loading


def test_load_pose_backend_returns_the_torch_model_unchanged_for_authority() -> None:
    sentinel = object()
    assert load_pose_backend("cpu_torch", torch_posenet=sentinel) is sentinel


def test_load_pose_backend_refuses_a_screening_backend_without_a_package() -> None:
    with pytest.raises(AneScreeningError, match="pose-mlpackage"):
        load_pose_backend("ane_fp16_screen", torch_posenet=object())


# ------------------------------------------------------------ rank agreement


def test_rank_agreement_reports_an_exact_argmin_match() -> None:
    verdict = rank_agreement([3.0, 1.0, 2.0], [3.5, 1.1, 2.2])
    assert verdict["argmin_agrees"] is True
    assert verdict["kendall_tau_b"] == pytest.approx(1.0)


def test_rank_agreement_catches_an_argmin_disagreement() -> None:
    verdict = rank_agreement([1.0, 1.01], [1.02, 1.0])
    assert verdict["argmin_screened"] == 0
    assert verdict["argmin_confirmed"] == 1
    assert verdict["argmin_agrees"] is False


def test_rank_agreement_refuses_mismatched_or_empty_input() -> None:
    with pytest.raises(AneScreeningError):
        rank_agreement([1.0], [1.0, 2.0])
    with pytest.raises(AneScreeningError):
        rank_agreement([], [])


# ------------------------------------------------- wiring into the instruments


def _experiments_on_path() -> None:
    root = Path(__file__).resolve().parents[3] / "experiments"
    if str(root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(root))


def test_pr1_selector_exposes_the_backend_and_package_flags() -> None:
    _experiments_on_path()
    pr1 = pytest.importorskip("ddm_pr1_pose_resolve_on_renderer_change")
    args = pr1.build_parser().parse_args(
        [
            "selector", "--runtime", "/x", "--gt-cache", "/x", "--renderer", "/x",
            "--tokens", "/x", "--pairs-list", "1", "--out", "/x",
            "--scorer-backend", "ane_fp16_screen", "--pose-mlpackage", "/y",
        ]
    )
    assert args.scorer_backend == "ane_fp16_screen"
    assert str(args.pose_mlpackage) == "/y"


def test_pr1_selector_defaults_to_the_authority_backend() -> None:
    _experiments_on_path()
    pr1 = pytest.importorskip("ddm_pr1_pose_resolve_on_renderer_change")
    args = pr1.build_parser().parse_args(
        ["selector", "--runtime", "/x", "--gt-cache", "/x", "--renderer", "/x",
         "--tokens", "/x", "--pairs-list", "1", "--out", "/x"]
    )
    assert args.scorer_backend == AUTHORITY_BACKEND


def test_pr1_selector_rejects_an_invented_backend_name() -> None:
    _experiments_on_path()
    pr1 = pytest.importorskip("ddm_pr1_pose_resolve_on_renderer_change")
    with pytest.raises(SystemExit):
        pr1.build_parser().parse_args(
            ["selector", "--runtime", "/x", "--gt-cache", "/x", "--renderer", "/x",
             "--tokens", "/x", "--pairs-list", "1", "--out", "/x",
             "--scorer-backend", "ane"]
        )


def test_fs1_measure_defaults_to_the_authority_backend() -> None:
    _experiments_on_path()
    fs1 = pytest.importorskip("ddm_fs1_frame0_selector_reselection")
    args = fs1.build_parser().parse_args(
        ["measure", "--runtime", "/x", "--gt-cache", "/x", "--renderer", "/x",
         "--tokens", "/x", "--out", "/x"]
    )
    assert args.scorer_backend == AUTHORITY_BACKEND


def test_fs1_measure_refuses_a_screening_backend_before_it_touches_disk() -> None:
    """The refusal fires on the FLAG -- not after the instrument build.

    Passing a nonexistent runtime proves the ordering: a screening backend
    raises AneScreeningError, while the authority backend gets far enough to
    fail on the missing archive.
    """
    _experiments_on_path()
    fs1 = pytest.importorskip("ddm_fs1_frame0_selector_reselection")
    common = ["--runtime", "/nonexistent", "--gt-cache", "/x", "--renderer", "/x",
              "--tokens", "/x", "--out", "/x"]
    screened = fs1.build_parser().parse_args(
        ["measure", *common, "--scorer-backend", "ane_fp16_screen"]
    )
    with pytest.raises(AneScreeningError, match="SCREENING"):
        fs1.run_measure(screened)
    authority = fs1.build_parser().parse_args(["measure", *common])
    with pytest.raises(FileNotFoundError):
        fs1.run_measure(authority)


def test_pr1_selector_exposes_the_confirm_all_modes_experiment() -> None:
    """True rank agreement is only measurable when all 8 modes are confirmed."""
    _experiments_on_path()
    pr1 = pytest.importorskip("ddm_pr1_pose_resolve_on_renderer_change")
    args = pr1.build_parser().parse_args(
        ["selector", "--runtime", "/x", "--gt-cache", "/x", "--renderer", "/x",
         "--tokens", "/x", "--pairs-list", "1", "--out", "/x",
         "--scorer-backend", "ane_fp16_screen", "--pose-mlpackage", "/y",
         "--confirm-all-modes"]
    )
    assert args.confirm_all_modes is True
    source = Path(pr1.__file__).read_text()
    # a partial confirm must NOT be reported as rank agreement
    assert "TRUE rank agreement" in source


# --------------------------------------------- ddm_ane2 measured axis verdicts


def test_measured_verdict_admits_the_backend_ane2_measured_fit_for_ranking():
    from tac.ane_screening import assert_backend_admissible_for_axis

    assert assert_backend_admissible_for_axis("coreml_cpu_fp32", "pose_rank") == "coreml_cpu_fp32"


def test_measured_verdict_refuses_the_fp16_screen_on_the_axis_it_failed():
    from tac.ane_screening import AneScreeningError, assert_backend_admissible_for_axis

    with pytest.raises(AneScreeningError, match=r"10\.26%"):
        assert_backend_admissible_for_axis("ane_fp16_screen", "pose_rank")


def test_measured_verdict_refusal_names_the_arm_and_the_artifact():
    from tac.ane_screening import AneScreeningError, assert_backend_admissible_for_axis

    with pytest.raises(AneScreeningError) as excinfo:
        assert_backend_admissible_for_axis("ane_fp16_screen", "pose_value")
    message = str(excinfo.value)
    assert "ddm_ane2" in message
    assert "units_posenet_fp16.json" in message
    assert "6.32x" in message


def test_an_unmeasured_pairing_is_allowed_through_not_assumed_bad():
    from tac.ane_screening import assert_backend_admissible_for_axis, backend_axis_verdict

    assert backend_axis_verdict("cpu_torch", "pose_rank") is None
    assert assert_backend_admissible_for_axis("cpu_torch", "pose_rank") == "cpu_torch"


def test_measured_verdict_still_fails_closed_on_an_unknown_backend():
    from tac.ane_screening import AneScreeningError, assert_backend_admissible_for_axis

    with pytest.raises(AneScreeningError, match="unknown scorer backend"):
        assert_backend_admissible_for_axis("ane_split_k64", "seg_argmax")


def test_every_verdict_row_carries_a_measurement_an_arm_and_an_artifact():
    from tac.ane_screening import BACKEND_AXIS_VERDICTS

    for (backend, axis), row in BACKEND_AXIS_VERDICTS.items():
        assert isinstance(row["ok"], bool), (backend, axis)
        assert row["measured"].strip(), (backend, axis)
        assert row["arm"].startswith("ddm_"), (backend, axis)
        assert row["artifact"].endswith(".json"), (backend, axis)
