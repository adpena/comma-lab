"""Behavioral coverage for the additive C1 two-plane timing receiver."""

from __future__ import annotations

import json
import math
import time
import zipfile
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator
from tac.witness_dsl import v10_two_plane_timing_receiver as receiver
from tac.witness_dsl.v10_production_receiver import (
    MEMBER_NAME,
    PREDICTOR_RESIDUAL_Y_CODEC_ID,
    build_production_archive,
)

CAMERA_HW = (8, 10)
SCORER_HW = (3, 4)
PAIR_COUNT = 7
TIMING_KEYS = (
    "parse_seconds",
    "expansion_seconds",
    "solve0_seconds",
    "solve1_seconds",
    "assembly_io_seconds",
    "verification_seconds",
)


def _planes(pair_count: int = PAIR_COUNT) -> tuple[np.ndarray, np.ndarray]:
    base = np.arange(pair_count * SCORER_HW[0] * SCORER_HW[1] * 3, dtype=np.uint16)
    y0 = ((base * 17 + 3) % 256).astype(np.uint8).reshape(pair_count, *SCORER_HW, 3)
    delta = np.arange(1, pair_count + 1, dtype=np.uint16)[:, None, None, None]
    y1 = ((y0.astype(np.uint16) + delta) % 256).astype(np.uint8)
    return y0, y1


def _names(root: Path, video_name: str = "0.raw") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "video_names.txt"
    path.write_text(f"{video_name}\n", encoding="utf-8")
    return path


def _build(
    root: Path,
    *,
    y0: np.ndarray | None = None,
    y1: np.ndarray | None = None,
    pair_ids: tuple[int, ...] | None = None,
    residual: np.ndarray | None = None,
) -> Path:
    default0, default1 = _planes()
    frame0 = default0 if y0 is None else y0
    frame1 = default1 if y1 is None else y1
    archive_dir = root / "archive"
    build_production_archive(
        frame1,
        frame0_y_planes=frame0,
        archive_path=archive_dir / "archive.zip",
        camera_height=CAMERA_HW[0],
        camera_width=CAMERA_HW[1],
        y_codec_id=PREDICTOR_RESIDUAL_Y_CODEC_ID,
        predictor_modes="spatial-smooth-121.v1",
        predictor_pair_ids=pair_ids if pair_ids is not None else tuple(range(len(frame0))),
        quotient_residual=residual,
    )
    return archive_dir


def _inflate(
    archive_dir: Path,
    root: Path,
    *,
    output_name: str = "output",
    receipt_name: str = "timing.json",
    video_name: str = "0.raw",
    **kwargs: Any,
) -> receiver.TwoPlaneTimedInflateResult:
    return receiver.timed_inflate_two_plane_archive(
        archive_dir,
        root / output_name,
        _names(root, video_name),
        timing_receipt_path=root / receipt_name,
        expected_pair_count=PAIR_COUNT,
        expected_camera_hw=CAMERA_HW,
        expected_scorer_hw=SCORER_HW,
        **kwargs,
    )


class _ImmediateFuture:
    def __init__(self, function: Any, *args: Any) -> None:
        self._function = function
        self._args = args

    def result(self) -> Any:
        return self._function(*self._args)


class _InlineProcessPool:
    calls: ClassVar[list[str]] = []

    def __init__(self, *, max_workers: int, initializer: Any, initargs: tuple[Any, ...]) -> None:
        assert max_workers >= 4
        initializer(*initargs)

    def submit(self, function: Any, *args: Any) -> _ImmediateFuture:
        self.calls.append(function.__name__)
        return _ImmediateFuture(function, *args)

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        assert wait and cancel_futures


class _FakeMlx:
    array_calls: ClassVar[int] = 0

    @staticmethod
    def array(value: Any) -> np.ndarray:
        _FakeMlx.array_calls += 1
        return np.asarray(value)

    @staticmethod
    def take(value: np.ndarray, indices: np.ndarray, *, axis: int) -> np.ndarray:
        return np.take(value, indices, axis=axis)

    @staticmethod
    def zeros_like(value: np.ndarray) -> np.ndarray:
        return np.zeros_like(value)

    @staticmethod
    def where(condition: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.where(condition, left, right)

    @staticmethod
    def eval(_value: np.ndarray) -> None:
        return None


class _DivergentMlx(_FakeMlx):
    @staticmethod
    def where(condition: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        result = np.where(condition, left, right).copy()
        result[0, 0] ^= np.uint8(1)
        return result


class _FloatOutputMlx(_FakeMlx):
    @staticmethod
    def where(condition: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.where(condition, left, right).astype(np.float32)


class _NoMetalMlx(_FakeMlx):
    @staticmethod
    def eval(_value: np.ndarray) -> None:
        raise RuntimeError("No Metal device available")


def test_distinct_two_plane_happy_path_exact_counts_and_timing(tmp_path: Path) -> None:
    archive = _build(tmp_path)
    result = _inflate(archive, tmp_path)
    assert result.completed
    assert result.raw_path is not None
    assert result.raw_bytes == PAIR_COUNT * 2 * CAMERA_HW[0] * CAMERA_HW[1] * 3
    assert result.numerator_values_verified == PAIR_COUNT * 2 * SCORER_HW[0] * SCORER_HW[1] * 3
    assert result.receipt["both_planes_exact"] is True
    assert result.receipt["plane0_stages_preserved"] == PAIR_COUNT
    assert result.receipt["contest_budget_verdict"] is None
    assert result.receipt["score_claim"] is False
    assert result.output_tree_sha256 == receiver._tree_sha256(Path(result.receipt["output_root"]))
    for key in TIMING_KEYS:
        assert math.isfinite(result.receipt["timing"][key])
        assert result.receipt["timing"][key] > 0


def test_official_extracted_0_bin_abi_matches_canonical_archive_input(tmp_path: Path) -> None:
    archive = _build(tmp_path)
    canonical = _inflate(
        archive,
        tmp_path,
        output_name="canonical",
        receipt_name="canonical.json",
        video_name="nested/0.mkv",
    )
    with zipfile.ZipFile(archive / "archive.zip", "r") as handle:
        packet = handle.read(MEMBER_NAME)
    extracted = tmp_path / "archive-input"
    extracted.mkdir()
    (extracted / MEMBER_NAME).write_bytes(packet)
    official = _inflate(
        extracted,
        tmp_path,
        output_name="official",
        receipt_name="official.json",
        video_name="nested/0.mkv",
    )
    assert official.raw_sha256 == canonical.raw_sha256
    assert official.receipt["archive_sha256"] == canonical.receipt["archive_sha256"]
    assert official.receipt["packet_sha256"] == canonical.receipt["packet_sha256"]
    assert official.receipt["archive_input_kind"] == "extracted_0_bin"
    assert official.receipt["canonical_archive_reconstructed"] is True
    assert official.receipt["raw_relative_path"] == "nested/0.raw"
    assert official.raw_path == tmp_path / "official" / "nested" / "0.raw"


def test_mixed_archive_and_extracted_packet_must_agree(tmp_path: Path) -> None:
    archive = _build(tmp_path)
    with zipfile.ZipFile(archive / "archive.zip", "r") as handle:
        packet = bytearray(handle.read(MEMBER_NAME))
    packet[-1] ^= 1
    (archive / MEMBER_NAME).write_bytes(packet)
    with pytest.raises(receiver.TwoPlaneTimingReceiverError, match="disagree"):
        _inflate(archive, tmp_path)


@pytest.mark.parametrize("video_name", ("../0.mkv", "/0.mkv", "nested/../../0.mkv"))
def test_official_video_name_traversal_refuses_before_output(tmp_path: Path, video_name: str) -> None:
    archive = _build(tmp_path)
    output = tmp_path / "unsafe-name-output"
    with pytest.raises(receiver.TwoPlaneTimingReceiverError, match="escape"):
        _inflate(
            archive,
            tmp_path,
            output_name=output.name,
            receipt_name="unsafe-name.json",
            video_name=video_name,
        )
    assert not output.exists()


def test_byte_equal_and_memory_aliased_planes_refuse(tmp_path: Path) -> None:
    y0, _y1 = _planes()
    equal_archive = _build(tmp_path / "equal", y0=y0, y1=y0)
    with pytest.raises(receiver.TwoPlaneTimingReceiverError, match="byte-equal"):
        _inflate(equal_archive, tmp_path / "equal-run")

    alias = np.zeros((PAIR_COUNT, *SCORER_HW, 3), dtype=np.uint8)
    with pytest.raises(receiver.TwoPlaneTimingReceiverError, match="alias"):
        receiver._validate_targets(
            alias,
            alias,
            tuple(range(PAIR_COUNT)),
            expected_pair_count=PAIR_COUNT,
            expected_scorer_hw=SCORER_HW,
            expected_y0_sha256=None,
            expected_y1_sha256=None,
        )


def test_wrong_codec_policy_pair_ids_and_residual_refuse(tmp_path: Path) -> None:
    y0, y1 = _planes()
    legacy = tmp_path / "legacy"
    build_production_archive(
        y1,
        archive_path=legacy / "archive.zip",
        camera_height=CAMERA_HW[0],
        camera_width=CAMERA_HW[1],
    )
    with pytest.raises(receiver.TwoPlaneTimingReceiverError, match="predictor-residual"):
        _inflate(legacy, tmp_path / "legacy-run")

    wrong_ids = _build(tmp_path / "ids", pair_ids=tuple(range(10, 10 + PAIR_COUNT)))
    with pytest.raises(receiver.TwoPlaneTimingReceiverError, match=r"0\.\.N-1"):
        _inflate(wrong_ids, tmp_path / "ids-run")

    residual = np.zeros((PAIR_COUNT, *CAMERA_HW, 3), dtype="<i2")
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_HW[0],
        camera_w=CAMERA_HW[1],
        scorer_h=SCORER_HW[0],
        scorer_w=SCORER_HW[1],
    )
    owned = np.zeros(CAMERA_HW, dtype=bool)
    for rows in operator.row_supports:
        for cols in operator.col_supports:
            owned[np.ix_(rows.indices, cols.indices)] = True
    row, col = np.argwhere(~owned)[0]
    residual[0, row, col, 0] = 1
    residual_archive = _build(tmp_path / "residual", residual=residual)
    with pytest.raises(receiver.TwoPlaneTimingReceiverError, match="quotient residual"):
        _inflate(residual_archive, tmp_path / "residual-run")


def test_wrong_geometry_and_noncanonical_or_trailing_packet_refuse(tmp_path: Path) -> None:
    archive = _build(tmp_path / "geometry")
    with pytest.raises(receiver.TwoPlaneTimingReceiverError, match="geometry"):
        receiver.timed_inflate_two_plane_archive(
            archive,
            tmp_path / "geometry-output",
            _names(tmp_path / "geometry-names"),
            timing_receipt_path=tmp_path / "geometry.json",
            expected_pair_count=PAIR_COUNT,
            expected_camera_hw=(10, 12),
            expected_scorer_hw=SCORER_HW,
        )

    path = archive / "archive.zip"
    with zipfile.ZipFile(path) as handle:
        packet = handle.read(MEMBER_NAME)
    path.write_bytes(receiver._canonical_zip(packet + b"trailer"))
    with pytest.raises(receiver.TwoPlaneTimingReceiverError, match="parsing refused"):
        _inflate(archive, tmp_path / "trailing-run")


def test_packet_reencode_drift_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _build(tmp_path)
    original = receiver.reserialize_parsed_packet
    monkeypatch.setattr(receiver, "reserialize_parsed_packet", lambda packet: original(packet) + b"x")
    with pytest.raises(receiver.TwoPlaneTimingReceiverError, match="re-encode differs"):
        _inflate(archive, tmp_path / "drift-run")


def test_plane0_checkpoint_survives_and_resume_skips_solve0(tmp_path: Path) -> None:
    archive = _build(tmp_path)
    interrupted = _inflate(
        archive,
        tmp_path,
        output_name="resume-output",
        receipt_name="interrupted.json",
        stop_after_plane0_pairs=2,
    )
    assert not interrupted.completed
    assert interrupted.raw_bytes == 0
    state = Path(interrupted.receipt["state_root"])
    x0_path = state / "plane0" / "pair-000001.bin"
    before = x0_path.read_bytes()
    resumed = _inflate(
        archive,
        tmp_path,
        output_name="resume-output",
        receipt_name="resumed.json",
        resume=True,
    )
    assert resumed.completed
    assert x0_path.read_bytes() == before
    pair1 = next(row for row in resumed.receipt["timing"]["per_pair"] if row["pair_index"] == 1)
    assert pair1["resumed_plane0"] is True
    assert pair1["solve0_seconds"] == 0


@pytest.mark.parametrize("leg", ["plane0", "plane0_manifest", "pair", "pair_manifest"])
def test_resume_refuses_edited_stage_or_manifest(tmp_path: Path, leg: str) -> None:
    archive = _build(tmp_path)
    interrupted = _inflate(
        archive,
        tmp_path,
        output_name=f"edited-{leg}",
        receipt_name=f"interrupted-{leg}.json",
        stop_after_pairs=2,
    )
    state = Path(interrupted.receipt["state_root"])
    choices = {
        "plane0": state / "plane0" / "pair-000000.bin",
        "plane0_manifest": state / "plane0_manifests" / "pair-000000.json",
        "pair": state / "pairs" / "pair-000000.bin",
        "pair_manifest": state / "pair_manifests" / "pair-000000.json",
    }
    target = choices[leg]
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 1
    target.write_bytes(payload)
    with pytest.raises(receiver.TwoPlaneTimingReceiverError, match=r"drifted|failed|canonical|not valid JSON"):
        _inflate(
            archive,
            tmp_path,
            output_name=f"edited-{leg}",
            receipt_name=f"resumed-{leg}.json",
            resume=True,
        )


def test_all_completed_resume_is_non_authoritative_but_succeeds(tmp_path: Path) -> None:
    archive = _build(tmp_path)
    first = _inflate(archive, tmp_path, output_name="resume-all", receipt_name="fresh.json")
    resumed = _inflate(
        archive,
        tmp_path,
        output_name="resume-all",
        receipt_name="resume-all.json",
        resume=True,
    )
    assert resumed.completed and resumed.resumed_pairs == PAIR_COUNT
    assert resumed.raw_sha256 == first.raw_sha256
    assert resumed.receipt["fresh"] is False
    assert resumed.receipt["timing"]["solve0_seconds"] == 0
    assert resumed.receipt["timing"]["solve1_seconds"] == 0


@pytest.mark.parametrize(
    ("leg", "recoverable"),
    [
        ("plane0_stage_only", True),
        ("pair_stage_only", True),
        ("plane0_manifest_only", False),
        ("pair_manifest_only", False),
    ],
)
def test_resume_repairs_only_the_safe_stage_only_crash_window(
    tmp_path: Path,
    leg: str,
    recoverable: bool,
) -> None:
    archive = _build(tmp_path)
    first = _inflate(archive, tmp_path, output_name="crash-window", receipt_name="first.json")
    state = Path(first.receipt["state_root"])
    paths = {
        "plane0_stage_only": state / "plane0_manifests" / "pair-000000.json",
        "pair_stage_only": state / "pair_manifests" / "pair-000000.json",
        "plane0_manifest_only": state / "plane0" / "pair-000000.bin",
        "pair_manifest_only": state / "pairs" / "pair-000000.bin",
    }
    removed = paths[leg]
    removed.unlink()

    def invocation() -> receiver.TwoPlaneTimedInflateResult:
        return _inflate(
            archive,
            tmp_path,
            output_name="crash-window",
            receipt_name=f"resume-{leg}.json",
            resume=True,
        )

    if recoverable:
        recovered = invocation()
        assert recovered.completed
        assert removed.is_file()
    else:
        with pytest.raises(receiver.TwoPlaneTimingReceiverError, match="manifest exists without"):
            invocation()


@pytest.mark.parametrize(
    ("manifest_dir", "field"),
    [
        ("plane0_manifests", "numerator0_equal_values"),
        ("pair_manifests", "numerator1_equal_values"),
    ],
)
def test_resume_refuses_float_coercion_in_integer_manifest_fields(
    tmp_path: Path,
    manifest_dir: str,
    field: str,
) -> None:
    archive = _build(tmp_path)
    first = _inflate(archive, tmp_path, output_name="strict-types", receipt_name="first.json")
    path = Path(first.receipt["state_root"]) / manifest_dir / "pair-000000.json"
    row = json.loads(path.read_bytes())
    row[field] = float(row[field])
    path.write_bytes(receiver._canonical_json(row))
    with pytest.raises(receiver.TwoPlaneTimingReceiverError, match="must be an exact integer"):
        _inflate(
            archive,
            tmp_path,
            output_name="strict-types",
            receipt_name=f"resume-{manifest_dir}.json",
            resume=True,
        )


def test_parallel_phases_verify_in_pool_and_match_serial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _build(tmp_path)
    serial = _inflate(archive, tmp_path, output_name="serial", receipt_name="serial.json")
    _InlineProcessPool.calls.clear()
    monkeypatch.setattr(receiver, "ProcessPoolExecutor", _InlineProcessPool)
    parallel = _inflate(
        archive,
        tmp_path,
        output_name="parallel",
        receipt_name="parallel.json",
        workers=4,
    )
    assert parallel.raw_sha256 == serial.raw_sha256
    assert parallel.stage_tree_sha256 == serial.stage_tree_sha256
    assert parallel.chunk_tree_sha256 == serial.chunk_tree_sha256
    assert "_process_solve" in _InlineProcessPool.calls
    assert "_process_verify" in _InlineProcessPool.calls
    assert parallel.receipt["execution"] == {
        "mode": "process_pool",
        "workers": 4,
        "fixed_pair_assembly_order": True,
        "timed_solver_operator_builds": 1,
    }


def test_process_pool_host_refusal_is_typed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _build(tmp_path)

    def refuse_pool(**_kwargs: Any) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(receiver, "ProcessPoolExecutor", refuse_pool)
    with pytest.raises(receiver.TwoPlaneTimingReceiverError, match="process pool host-custody refusal"):
        _inflate(
            archive,
            tmp_path,
            output_name="pool-refusal",
            receipt_name="pool-refusal.json",
            workers=4,
        )


def test_preflight_refuses_before_any_output_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _build(tmp_path)
    output = tmp_path / "no-space"
    monkeypatch.setattr(
        receiver,
        "_storage_preflight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            receiver.TwoPlaneTimingReceiverError("storage preflight refused")
        ),
    )
    with pytest.raises(receiver.TwoPlaneTimingReceiverError, match="storage preflight refused"):
        _inflate(archive, tmp_path, output_name=output.name, receipt_name="no-space.json")
    assert not output.exists()


def test_mlx_twin_parity_includes_exact_numerators_and_false_authority() -> None:
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_HW[0],
        camera_w=CAMERA_HW[1],
        scorer_h=SCORER_HW[0],
        scorer_w=SCORER_HW[1],
    )
    y0, y1 = _planes(6)
    _FakeMlx.array_calls = 0
    row = receiver.parity_check_mlx_two_plane(
        operator,
        y0,
        y1,
        pair_ids=tuple(range(6)),
        mlx_module=_FakeMlx,
        test_only_allow_unverified_backend=True,
    )
    assert row["algorithmic_parity_passed"] is True
    assert row["parity_passed"] is False
    assert row["backend_kind"] == "explicit_test_double"
    assert row["numerator_values_verified"] == 6 * 2 * SCORER_HW[0] * SCORER_HW[1] * 3
    assert row["axis"] == receiver.MLX_TIMING_AXIS
    assert row["score_claim"] is row["contest_timing_verdict_eligible"] is False
    assert all(
        plane["numerator_exact"] and plane["certified_exact"] for pair in row["rows"] for plane in pair["planes"]
    )
    # Injected backends cannot run the real-device probe: two static-plan
    # transfers plus one target transfer per plane only.
    assert _FakeMlx.array_calls == 2 + 6 * 2


def test_mlx_divergence_is_a_finding_and_no_metal_is_host_refusal() -> None:
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_HW[0],
        camera_w=CAMERA_HW[1],
        scorer_h=SCORER_HW[0],
        scorer_w=SCORER_HW[1],
    )
    y0, y1 = _planes(6)
    divergent = receiver.parity_check_mlx_two_plane(
        operator,
        y0,
        y1,
        pair_ids=tuple(range(6)),
        mlx_module=_DivergentMlx,
        test_only_allow_unverified_backend=True,
    )
    assert divergent["parity_passed"] is False
    assert divergent["divergences"]
    assert all(item["failure_kinds"] for item in divergent["divergences"])
    custody = receiver.mlx_runtime_status(mlx_module=_NoMetalMlx)
    assert custody["backend_identity_verified"] is False
    assert custody["metal_usable"] is False
    assert "injected MLX backend" in custody["host_custody_refusal"]


def test_mlx_refuses_coercive_ids_and_native_float_output() -> None:
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_HW[0],
        camera_w=CAMERA_HW[1],
        scorer_h=SCORER_HW[0],
        scorer_w=SCORER_HW[1],
    )
    y0, y1 = _planes(6)
    with pytest.raises(receiver.TwoPlaneTimingReceiverError, match="exact integer"):
        receiver.parity_check_mlx_two_plane(
            operator,
            y0,
            y1,
            pair_ids=tuple(float(index) for index in range(6)),
            mlx_module=_FakeMlx,
            test_only_allow_unverified_backend=True,
        )
    with pytest.raises(receiver.TwoPlaneTimingReceiverError, match="native dtype float32"):
        receiver.realize_factor2_uint8_scorer_plane_mlx(
            operator,
            y0[0],
            mlx_module=_FloatOutputMlx,
        )


def test_receiver_source_has_no_forbidden_runtime_imports_or_frame_copy_shortcut() -> None:
    source_path = Path(receiver.__file__)
    lowered = source_path.read_text(encoding="utf-8").lower()
    for forbidden in (
        "import torch",
        "import segnet",
        "import posenet",
        "distortionnet",
        "from tools.measure_v10_free_predictor_floor",
        "frame0 = frame1",
        "frame1 = frame0",
    ):
        assert forbidden not in lowered


def test_timing_receipt_is_canonical_and_write_once(tmp_path: Path) -> None:
    archive = _build(tmp_path)
    result = _inflate(archive, tmp_path)
    payload = result.timing_receipt_path.read_bytes()
    assert receiver._canonical_json(json.loads(payload)) == payload
    with pytest.raises(receiver.TwoPlaneTimingReceiverError, match="write-once"):
        receiver.timed_inflate_two_plane_archive(
            archive,
            tmp_path / "another-output",
            _names(tmp_path),
            timing_receipt_path=result.timing_receipt_path,
            expected_pair_count=PAIR_COUNT,
            expected_camera_hw=CAMERA_HW,
            expected_scorer_hw=SCORER_HW,
        )


def test_timing_receipt_inside_output_root_refuses_before_writes(tmp_path: Path) -> None:
    archive = _build(tmp_path)
    output = tmp_path / "nested-receipt-output"
    with pytest.raises(receiver.TwoPlaneTimingReceiverError, match="outside the hashed output root"):
        receiver.timed_inflate_two_plane_archive(
            archive,
            output,
            _names(tmp_path),
            timing_receipt_path=output / "timing.json",
            expected_pair_count=PAIR_COUNT,
            expected_camera_hw=CAMERA_HW,
            expected_scorer_hw=SCORER_HW,
        )
    assert not output.exists()


def test_receiver_total_closes_after_pre_receipt_custody_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _build(tmp_path)
    original = receiver._source_hashes

    def delayed_source_hashes() -> dict[str, str]:
        time.sleep(0.02)
        return dict(original())

    monkeypatch.setattr(receiver, "_source_hashes", delayed_source_hashes)
    result = _inflate(archive, tmp_path)
    timing = result.receipt["timing"]
    assert timing["total_boundary"] == "entry_through_pre_receipt_evidence_collection"
    assert timing["receipt_serialization_and_persistence_included"] is False
    assert "outer_process_wall_seconds" not in timing
    assert timing["total_seconds"] - timing["component_sum_seconds"] >= 0.015
