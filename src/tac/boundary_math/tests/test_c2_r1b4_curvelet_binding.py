from __future__ import annotations

import hashlib
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

import tac.boundary_math.integer_plane_banded_trainer as trainer
from tac.boundary_math.c2_r1b4_curvelet_binding import (
    C2R1B4CurveletBinding,
    C2R1B4CurveletBindingError,
)
from tac.boundary_math.integer_plane_banded_trainer import (
    PLANE_SHAPE,
    StagePlan,
    TrainerConfig,
    canonical_json,
    sha256_file,
    train_streamed,
)
from tac.boundary_math.integer_plane_emitter_byte_close import build_counted_archive
from tac.boundary_math.power_diagram_witness import encode_pdw2, make_gauge_fixed_affine_target
from tac.boundary_math.r1b4_section_receiver import (
    BOUNDARY_NAME,
    build_r1b4_archive,
    decode_r1b4_archive,
    encode_replay_payload,
    parse_r1b4_archive,
)
from tac.boundary_math.windowed_curvelet_frame import WindowedCurveletConfig
from tac.optimization.boundary_coordinate_joint_solve import decode_boundary_packet
from tac.optimization.r1b3_producer_preflight import encode_xi0_payload
from tac.witness_dsl.integer_plane_emitter_policy import BasisMode, IntegerPlaneEmitterPolicy, PolicyMode


def _write_binding(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    predecessor = tmp_path / "curvelet_carrier.json"
    predecessor.write_bytes(canonical_json({"c2_banded_trainer_binding": "ABSENT"}))
    predecessor_sha = sha256_file(predecessor)
    band = tmp_path / "band_manifest.json"
    band.write_bytes(
        canonical_json(
            {
                "custody": {
                    "ev_selection": {
                        "artifact_records": {
                            "curvelet_carrier": {
                                "path": predecessor.name,
                                "sha256": predecessor_sha,
                            }
                        }
                    }
                }
            }
        )
    )
    binding = tmp_path / "carrier_binding.json"
    binding.write_bytes(
        canonical_json(
            {
                "atom_indices": [0, 1, 2, 3],
                "authority": "build_and_local_verify_only_no_launch_no_score",
                "band_manifest": {
                    "path": str(band.resolve()),
                    "predecessor_carrier_record_sha256": predecessor_sha,
                    "sha256": sha256_file(band),
                },
                "basis_id": "r1b4_windowed_curvelet",
                "family": "windowed_curvelet",
                "frame_config": asdict(WindowedCurveletConfig(n_scales=1, n_orient0=2, n_trans=1)),
                "logical_pair_count": 600,
                "quantization_strata": {
                    "candidate_pixels": 38_077,
                    "carrier_spend_policy": (
                        "optimization_realizable_only_dead_zero_weight_shared_bytes_unattributed"
                    ),
                    "realizable_pixels": 11_453,
                    "substep_dead_pixels": 26_624,
                },
                "receiver": {
                    "archive_section": "boundary_coordinate.bgj",
                    "byte_accounting": "counted_zip_member_actual_bytes",
                    "factor2_consumption": "exact_uint8_scorer_target_to_camera_preimage",
                    "packet_schema": "boundary_coordinate_packet.v1",
                    "receiver_schema": "r1b4_section_receiver.v1",
                    "semantic_frame": 1,
                    "unknown_or_trailing_sections": "refuse",
                },
                "schema": "c2_r1b4_curvelet_carrier_binding.v1",
                "scorer_geometry": [384, 512, 3],
                "verdict_scope": "fixture; no score",
            }
        )
    )
    return binding


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


def _pdw2_packet() -> bytes:
    target = make_gauge_fixed_affine_target(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([0.0, 0.25], dtype=np.float32),
    )
    return encode_pdw2(target)


def _c2_control_archive(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "one_packet.zip"
    decoder = tmp_path / "inflate.py"
    _write_base_archive(base)
    _write_decoder(decoder)
    config = TrainerConfig(
        policy=IntegerPlaneEmitterPolicy(mode=PolicyMode.BANDED_TRAINING),
        base_archive_sha256=sha256_file(base),
        base_decoder_sha256=sha256_file(decoder),
        source_sha256="a" * 64,
        band_sha256="b" * 64,
        band_mode="positive_anisotropic",
        output_dir=tmp_path,
        run_id="curvelet-control",
    )
    state = trainer._state_from_fresh(config)
    checkpoint = trainer._checkpoint(config, state, stage=config.stages[0], stage_complete=False)
    checkpoint_path = checkpoint.write_new(tmp_path, "curvelet_control")
    archive = tmp_path / "c2_control.zip"
    build_counted_archive(
        base_archive=base,
        checkpoint_path=checkpoint_path,
        output=archive,
        pdw2_packet=_pdw2_packet(),
    )
    return archive, decoder


def test_binding_regenerates_receiver_basis_and_refuses_unconsumed_frame0(tmp_path: Path) -> None:
    binding = C2R1B4CurveletBinding.load(_write_binding(tmp_path))
    assert binding.coordinate_basis().shape == (384, 512, 4)
    assert binding.selected_pixel_count == 11_453
    assert binding.dead_pixel_count == 26_624
    codes = np.zeros((600, 2, 4), dtype=np.float32)
    head = np.ones((4, 3), dtype=np.float32)
    payload, receipt = binding.export_packet(codes, head)
    packet = decode_boundary_packet(payload)
    assert packet.family.value == "windowed_curvelet"
    assert packet.coefficients.shape == (600, 4, 3)
    assert receipt["packet_bytes"] == len(payload)
    assert receipt["receiver_consumed"] is True
    assert receipt["quantization_strata"]["optimization_weight_on_dead_stratum"] == 0
    target = tmp_path / "packet.bgj"
    created = binding.write_packet_new(target, codes, head)
    resumed = binding.write_packet_new(target, codes, head)
    assert created["write_disposition"] == "created_new"
    assert resumed["write_disposition"] == "reused_byte_identical_existing"
    codes[0, 0, 0] = 1.0
    with pytest.raises(C2R1B4CurveletBindingError, match="frame-0"):
        binding.export_packet(codes, head)


class _Stream:
    def __init__(self, config: TrainerConfig) -> None:
        self.pair_count = config.pair_count
        self.base_sha256 = config.base_archive_sha256
        self.source_sha256 = config.source_sha256
        self.band_sha256 = config.band_sha256
        self.band_mode = config.band_mode

    def fetch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = (len(indices), *PLANE_SHAPE)
        base = np.full(shape, 128.0, dtype=np.float32)
        source = base.copy()
        radii = np.full(shape, 255.0, dtype=np.float32)
        return base, source, radii


def test_streamed_trainer_uses_bound_topology_and_emits_ema_packet(tmp_path: Path) -> None:
    binding = C2R1B4CurveletBinding.load(_write_binding(tmp_path / "binding"))
    config = TrainerConfig(
        policy=IntegerPlaneEmitterPolicy(
            basis=BasisMode.R1B4_WINDOWED_CURVELET,
            mode=PolicyMode.BANDED_TRAINING,
        ),
        base_archive_sha256="a" * 64,
        base_decoder_sha256="b" * 64,
        source_sha256="c" * 64,
        band_sha256=binding.band_manifest_sha256,
        band_mode="positive_anisotropic",
        output_dir=tmp_path / "train",
        run_id="curvelet_train",
        carrier_binding=binding,
        smoke_pair_cap=2,
        pair_batch_size=2,
        stages=(
            StagePlan("warmup", 1, 2e-3, 1e-6),
            StagePlan("band_fit", 1, 1e-3, 1e-5),
            StagePlan("rate_polish", 1, 2e-4, 1e-3),
        ),
    )
    receipt = train_streamed(config, _Stream(config))
    assert receipt["config"]["basis"] == "r1b4_windowed_curvelet"
    assert receipt["config"]["carrier_binding"]["topology_sha256"] == binding.topology_sha256
    packet = receipt["carrier_packet"]
    assert packet["receiver_consumed"] is True
    assert Path(packet["path"]).read_bytes()
    final = trainer.load_training_state(receipt["checkpoint_paths"][-1], config)
    assert np.count_nonzero(final.ema_codes[:, 0]) == 0


def test_exported_packet_is_counted_and_mutation_consumed_by_real_r1b4_receiver(tmp_path: Path) -> None:
    binding = C2R1B4CurveletBinding.load(_write_binding(tmp_path / "binding"))
    codes = np.zeros((600, 2, 4), dtype=np.float32)
    head = np.ones((4, 3), dtype=np.float32)
    zero_packet, _ = binding.export_packet(codes, head)
    codes[:, 1, 0] = np.float32(4.0)
    active_packet, active_receipt = binding.export_packet(codes, head)
    assert active_packet != zero_packet
    assert active_receipt["nonzero_coefficients"] > 0

    base, decoder = _c2_control_archive(tmp_path / "base")
    xi0 = encode_xi0_payload(np.linspace(29.0, 33.0, 600, dtype=np.float32))
    archives = {}
    raws = {}
    for label, packet in (("zero", zero_packet), ("active", active_packet)):
        archive = tmp_path / f"{label}.zip"
        build_r1b4_archive(
            base_archive=base,
            boundary_payload=packet,
            replay_payload=encode_replay_payload(()),
            xi0_payload=xi0,
            source_manifest_hashes={"fixture": hashlib.sha256(b"fixture").hexdigest()},
            output=archive,
            pair_cap=2,
        )
        parsed = parse_r1b4_archive(archive)
        assert parsed.manifest["sections"][BOUNDARY_NAME]["bytes"] == len(packet)
        assert parsed.boundary_payload == packet
        raw = tmp_path / f"{label}.raw"
        decode_r1b4_archive(
            archive=archive,
            base_decoder=decoder,
            scratch_root=tmp_path / f"scratch-{label}",
            output_raw=raw,
            receipt_path=tmp_path / f"{label}.json",
            allow_unsealed_discovery=True,
        )
        archives[label] = parsed
        raws[label] = raw
    assert archives["active"].manifest["sections"][BOUNDARY_NAME]["sha256"] == hashlib.sha256(
        active_packet
    ).hexdigest()
    shape = (2, 2, 874, 1164, 3)
    zero = np.memmap(raws["zero"], mode="r", dtype=np.uint8, shape=shape)
    active = np.memmap(raws["active"], mode="r", dtype=np.uint8, shape=shape)
    assert np.array_equal(zero[:, 0], active[:, 0])
    assert not np.array_equal(zero[:, 1], active[:, 1])
    del zero, active
