from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import tac.boundary_math.integer_plane_banded_trainer as trainer
from tac.boundary_math.integer_plane_banded_trainer import (
    BAND_SCHEMA,
    PLANE_SHAPE,
    BandArtifact,
    C2BandedTrainerError,
    StagePlan,
    TrainerConfig,
    build_parser,
    canonical_json,
    load_training_state,
    policy_from_args,
    sha256_file,
    train_streamed,
)
from tac.boundary_math.integer_plane_emitter_byte_close import (
    C2ByteCloseError,
    build_counted_archive,
    decode_counted_archive,
    parse_counted_archive,
)
from tac.boundary_math.power_diagram_witness import (
    encode_pdw2,
    make_gauge_fixed_affine_target,
)
from tac.witness_dsl.curriculum_dsl import IntegerPlaneEmitter
from tac.witness_dsl.integer_plane_emitter_policy import (
    IntegerPlaneEmitterPolicy,
    PolicyMode,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


class StreamingFixture:
    def __init__(self, pair_count: int, config: TrainerConfig) -> None:
        self.pair_count = pair_count
        self.base_sha256 = config.base_archive_sha256
        self.source_sha256 = config.source_sha256
        self.band_sha256 = config.band_sha256
        self.band_mode = config.band_mode
        self.fetch_shapes: list[tuple[int, ...]] = []

    def fetch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = (len(indices), *PLANE_SHAPE)
        self.fetch_shapes.append(shape)
        base = np.empty(shape, dtype=np.float32)
        base[:, 0] = np.float32(96.0)
        base[:, 1] = np.float32(160.0)
        source = base + np.float32(8.0)
        radii = np.empty(shape, dtype=np.float32)
        radii[..., 0] = np.float32(1.0)
        radii[..., 1] = np.float32(2.0)
        radii[..., 2] = np.float32(3.0)
        return base, source, radii


def _config(tmp_path: Path, *, pair_count: int = 600, batch: int = 2) -> TrainerConfig:
    return TrainerConfig(
        policy=IntegerPlaneEmitterPolicy(mode=PolicyMode.BANDED_TRAINING),
        base_archive_sha256=SHA_A,
        base_decoder_sha256=SHA_B,
        source_sha256=SHA_C,
        band_sha256=SHA_D,
        band_mode="positive_anisotropic",
        output_dir=tmp_path,
        run_id="test_c2",
        seed=20260719,
        pair_batch_size=batch,
        checkpoint_every_steps=100,
        stages=(
            StagePlan("warmup", 1, 2e-3, 1e-6),
            StagePlan("band_fit", 1, 1e-3, 1e-5),
            StagePlan("rate_polish", 1, 2e-4, 1e-3),
        ),
        pair_count=pair_count,
    )


def test_active_dsl_argv_is_consumed_by_exact_trainer_parser(tmp_path: Path) -> None:
    policy = IntegerPlaneEmitterPolicy(mode=PolicyMode.BANDED_TRAINING)
    lever = IntegerPlaneEmitter(policy=policy)
    argv: list[str] = []
    for flag, value in lever.overrides.items():
        argv.extend((flag, str(value)))
    argv.extend(
        (
            "--base-archive",
            str(tmp_path / "base.zip"),
            "--base-decoder",
            str(tmp_path / "inflate.py"),
            "--base-archive-sha256",
            SHA_A,
            "--base-decoder-sha256",
            SHA_B,
            "--band-manifest",
            str(tmp_path / "band.json"),
            "--output-dir",
            str(tmp_path / "out"),
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--run-id",
            "parse_test",
            "--stage-plan-json",
            json.dumps([row.to_dict() for row in trainer.DEFAULT_STAGE_PLAN]),
            "--receipt",
            str(tmp_path / "receipt.json"),
        )
    )
    parsed = build_parser().parse_args(argv)
    assert policy_from_args(parsed).compile_contract() == policy.compile_contract()
    hash_index = argv.index("--integer-plane-emitter-policy-sha256") + 1
    bad = list(argv)
    bad[hash_index] = SHA_A
    with pytest.raises(C2BandedTrainerError, match="policy hash mismatch"):
        policy_from_args(build_parser().parse_args(bad))
    missing = argv[: hash_index - 1] + argv[hash_index + 1 :]
    with pytest.raises(SystemExit):
        build_parser().parse_args(missing)


def _write_band(tmp_path: Path, *, isotropic: bool = False) -> Path:
    source = np.empty((2, *PLANE_SHAPE), dtype=np.uint8)
    source[:, 0] = 96
    source[:, 1] = 160
    radii = np.full((2, *PLANE_SHAPE), 255.0, dtype=np.float32)
    radii[:, :, 0, 0, 0] = 2.0
    radii[:, :, 0, 0, 1] = 2.0 if isotropic else 3.0
    radii[:, :, 0, 0, 2] = 2.0 if isotropic else 4.0
    source_path = tmp_path / "source.npy"
    radii_path = tmp_path / "radii.npy"
    np.save(source_path, source)
    np.save(radii_path, radii)
    artifact_records = {}
    for name in (
        "ranked_ev_field",
        "necessity",
        "resize",
        "channel_sensitivity",
        "kkt",
        "inner_jacobian_secant_qp",
        "curvelet_carrier",
        "xi_factorization",
        "gauge_binding",
    ):
        artifact_path = tmp_path / f"{name}.json"
        artifact_path.write_bytes(canonical_json({"fixture": name}))
        artifact_records[name] = {
            "path": artifact_path.name,
            "sha256": sha256_file(artifact_path),
        }
    manifest = {
        "schema": BAND_SCHEMA,
        "mode": "positive_anisotropic",
        "pair_count": 2,
        "geometry": list(PLANE_SHAPE),
        "source_planes": {"path": source_path.name, "sha256": sha256_file(source_path)},
        "radii": {"path": radii_path.name, "sha256": sha256_file(radii_path)},
        "custody": {
            "derivation": "derive_hyperplane_channel_band",
            "margins_sha256": SHA_A,
            "winner_sha256": SHA_B,
            "rival_sha256": SHA_C,
            "unit_head_normal_pullback_rgb_sha256": SHA_D,
            "pair_norms_sha256": "e" * 64,
            "config": {"scale": 1.0, "local_lipschitz": 1.0, "max_rgb_radius": 4.0},
            "ev_selection": {
                "policy": "measured_reverse_waterfill_highest_ev_first",
                "candidate_flip_count": 4,
                "selected_pixel_count": 4,
                "inactive_radius": 255.0,
                "rate_break_even_score_per_byte": 25.0 / 37_545_489.0,
                "stopped_below_break_even": True,
                "blanket_fix": False,
                "artifact_records": artifact_records,
                "law_refs": [
                    "frozen_scorer_fisher_curvature_margin_colocation_v1",
                    "fisher_curvature_equals_categorical_fisher_trace_caustic_v1",
                    "realization_necessity_preimage_per_stratum_v1",
                    "resize_exploit_flip_fix_frontier_v1",
                    "segnet_head_rank4_linear_flipdist_v1",
                    "posenet_luma_chroma_sensitivity_asymmetry_v1",
                    "flip_margin_step_law_v1",
                    "instant_projected_input_adjoint_v1",
                    "shearlet_nterm_upper_bounds_task_rate_v1",
                    "curvelet_directional_basis_dseg_reduction_v1",
                    "cgauge_curvelet_parabolic_bank_v1",
                    "scorer_obligation_matrix_factorization_v1",
                    "lane_band_ego_factorization_source_reparam_v1",
                    "witness_measured_reverse_waterfill_v1",
                    "meta_lagrangian_dual_solver_per_axis_kkt_residual_v1",
                    "cgauge_master_action_v1",
                ],
                "metric": "fisher_top1_top2_margin",
                "carrier_basis": "cgauge_curvelet_parabolic_bank_v1",
                "realization_predictor": "first_order_plus_secant_plus_qp_inner_jacobian",
                "pose_factorization": "single_se3_xi_twist",
                "gauge_status": "GAUGE_IDENTITY_VERIFIED_NOT_MODEL_FACTORIZED",
            },
        },
    }
    path = tmp_path / "band.json"
    path.write_bytes(canonical_json(manifest))
    return path


def test_positive_band_requires_anisotropic_derived_custody(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(trainer, "LOGICAL_PAIR_COUNT", 2)
    assert BandArtifact.load(_write_band(tmp_path)).mode == "positive_anisotropic"
    other = tmp_path / "isotropic"
    other.mkdir()
    with pytest.raises(C2BandedTrainerError, match="isotropic"):
        BandArtifact.load(_write_band(other, isotropic=True))


def test_positive_band_refuses_blanket_or_wrong_metric_ev_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(trainer, "LOGICAL_PAIR_COUNT", 2)
    path = _write_band(tmp_path)
    doc = json.loads(path.read_bytes())
    doc["custody"]["ev_selection"]["blanket_fix"] = True
    path.write_bytes(canonical_json(doc))
    with pytest.raises(C2BandedTrainerError, match="EV stop policy"):
        BandArtifact.load(path)
    doc["custody"]["ev_selection"]["blanket_fix"] = False
    doc["custody"]["ev_selection"]["metric"] = "euclidean"
    path.write_bytes(canonical_json(doc))
    with pytest.raises(C2BandedTrainerError, match="EV stop policy"):
        BandArtifact.load(path)


def test_positive_band_refuses_ev_artifact_hash_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(trainer, "LOGICAL_PAIR_COUNT", 2)
    path = _write_band(tmp_path)
    (tmp_path / "ranked_ev_field.json").write_bytes(b"tampered")
    with pytest.raises(C2BandedTrainerError, match="artifact custody mismatch"):
        BandArtifact.load(path)


def test_n600_logical_state_streams_real_geometry_and_records_peak_rss(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = StreamingFixture(600, config)
    receipt = train_streamed(config, source, stop_after_steps=1)
    assert receipt["logical_geometry"] == [600, *PLANE_SHAPE]
    assert receipt["trainable_arrays"] == ["pair_plane_codes", "shared_rgb_head"]
    assert receipt["peak_rss_bytes"] > 0
    assert source.fetch_shapes == [(2, *PLANE_SHAPE)]
    assert len(receipt["checkpoint_paths"]) == 2  # initial + resumable step


def test_deterministic_resume_matches_fresh_and_preserves_stage_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(trainer, "LOGICAL_PAIR_COUNT", 4)
    full = _config(tmp_path / "full", pair_count=4, batch=4)
    full_receipt = train_streamed(full, StreamingFixture(4, full))
    split_a = replace(full, output_dir=tmp_path / "split_a")
    first = train_streamed(split_a, StreamingFixture(4, split_a), stop_after_steps=2)
    split_b = replace(full, output_dir=tmp_path / "split_b")
    resumed = train_streamed(
        split_b,
        StreamingFixture(4, split_b),
        resume_from=first["checkpoint_paths"][-1],
    )
    full_state = load_training_state(full_receipt["checkpoint_paths"][-1], full)
    resumed_state = load_training_state(resumed["checkpoint_paths"][-1], split_b)
    assert np.array_equal(full_state.codes, resumed_state.codes)
    assert np.array_equal(full_state.head, resumed_state.head)
    assert np.array_equal(full_state.ema_codes, resumed_state.ema_codes)
    assert np.array_equal(full_state.ema_head, resumed_state.ema_head)
    names = [Path(path).name for path in full_receipt["checkpoint_paths"]]
    assert any("stage000_warmup" in name for name in names)
    assert any("stage001_band_fit" in name for name in names)
    assert any("stage002_rate_polish" in name for name in names)
    assert len(names) == len(set(names))
    with pytest.raises(C2BandedTrainerError, match="config drift"):
        load_training_state(
            full_receipt["checkpoint_paths"][-1],
            replace(full, base_archive_sha256="f" * 64),
        )


def _write_base_archive(path: Path) -> None:
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, b"counted-base-packet")


def _write_decoder(path: Path) -> None:
    path.write_text(
        "import os,sys\n"
        "n=int(os.environ['INFLATE_MAX_PAIRS'])\n"
        "fb=874*1164*3\n"
        "with open(sys.argv[2], 'wb') as f:\n"
        "  [f.write(bytes([96])*fb + bytes([160])*fb) for _ in range(n)]\n",
        encoding="ascii",
    )


def _write_checkpoint(tmp_path: Path, base_archive: Path, decoder: Path, *, changed: bool) -> Path:
    config = TrainerConfig(
        policy=IntegerPlaneEmitterPolicy(mode=PolicyMode.BANDED_TRAINING),
        base_archive_sha256=sha256_file(base_archive),
        base_decoder_sha256=sha256_file(decoder),
        source_sha256=SHA_C,
        band_sha256=SHA_D,
        band_mode="positive_anisotropic",
        output_dir=tmp_path,
        run_id="codec",
    )
    state = trainer._state_from_fresh(config)
    if changed:
        state.codes[:] = np.float32(4.0)
        state.head[:] = np.float32(1.0)
        state.ema_codes[:] = state.codes
        state.ema_head[:] = state.head
    checkpoint = trainer._checkpoint(config, state, stage=config.stages[0], stage_complete=False)
    return checkpoint.write_new(tmp_path, "codec_changed" if changed else "codec_pre")


def _pdw2_packet() -> bytes:
    target = make_gauge_fixed_affine_target(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([0.0, 0.25], dtype=np.float32),
    )
    return encode_pdw2(target)


def test_counted_archive_is_deterministic_strict_and_decoder_equal(tmp_path: Path) -> None:
    base = tmp_path / "base.zip"
    decoder = tmp_path / "inflate.py"
    _write_base_archive(base)
    _write_decoder(decoder)
    checkpoint = _write_checkpoint(tmp_path, base, decoder, changed=False)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    receipt = build_counted_archive(
        base_archive=base, checkpoint_path=checkpoint, output=first, pdw2_packet=_pdw2_packet()
    )
    build_counted_archive(base_archive=base, checkpoint_path=checkpoint, output=second, pdw2_packet=_pdw2_packet())
    assert first.read_bytes() == second.read_bytes()
    parsed = parse_counted_archive(first)
    assert receipt["accounted_archive_bytes"] == receipt["archive_bytes"]
    assert receipt["archive_bytes"] == first.stat().st_size
    assert parsed.manifest["authority"] == "ema"
    assert parsed.manifest["pdw2_spatial_receiver_consumed"] is False
    assert receipt["pdw2_spatial_receiver_consumed"] is False
    with pytest.raises(C2ByteCloseError, match="spatial/RGB pullback"):
        build_counted_archive(
            base_archive=base,
            checkpoint_path=checkpoint,
            output=tmp_path / "forbidden.zip",
            pdw2_packet=_pdw2_packet(),
            pdw2_role="receiver_consumed",
        )
    raw = tmp_path / "decoded.raw"
    decoded = decode_counted_archive(
        archive=first,
        base_decoder=decoder,
        scratch_root=tmp_path / "scratch",
        pair_cap=2,
        output_raw=raw,
    )
    assert decoded["numpy_decode_equal"] is True
    assert decoded["archive_bytes_full"] == first.stat().st_size
    assert decoded["decoded_raw_bytes_capped"] == 2 * 2 * 874 * 1164 * 3

    trailing = tmp_path / "trailing.zip"
    shutil.copyfile(first, trailing)
    with trailing.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(C2ByteCloseError, match="trailing"):
        parse_counted_archive(trailing)
    unknown = tmp_path / "unknown.zip"
    shutil.copyfile(first, unknown)
    with zipfile.ZipFile(unknown, "a") as archive:
        archive.writestr("unknown.bin", b"x")
    with pytest.raises(C2ByteCloseError, match="unknown section"):
        parse_counted_archive(unknown)

    changed_checkpoint = _write_checkpoint(tmp_path / "changed", base, decoder, changed=True)
    changed_archive = tmp_path / "changed.zip"
    build_counted_archive(
        base_archive=base,
        checkpoint_path=changed_checkpoint,
        output=changed_archive,
        pdw2_packet=_pdw2_packet(),
    )
    changed_raw = tmp_path / "changed.raw"
    changed_decode = decode_counted_archive(
        archive=changed_archive,
        base_decoder=decoder,
        scratch_root=tmp_path / "scratch_changed",
        pair_cap=2,
        output_raw=changed_raw,
    )
    assert changed_decode["decoded_raw_sha256"] != decoded["decoded_raw_sha256"]
