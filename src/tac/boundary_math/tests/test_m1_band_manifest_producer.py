from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tac.boundary_math.integer_plane_banded_trainer import canonical_json
from tac.boundary_math.power_diagram_witness import encode_pdw2, make_gauge_fixed_affine_target
from tac.optimization.joint_seg_pose_rate import MarginBandConfig, derive_hyperplane_channel_band
from tac.witness_dsl.integer_plane_emitter_policy import BasisMode
from tools import materialize_c2_integer_plane_emitter_fire as fire
from tools import produce_m1_band_manifest as producer


def test_quantization_gate_marks_every_substep_channel_dead() -> None:
    raw = np.asarray(
        [[0.0, 0.999999, 0.5], [1.0, 1.999, 8.0], [9.0, 2.25, 0.25]],
        dtype=np.float64,
    )
    steps, realizable = producer.quantize_band_widths(raw)
    assert steps.tolist() == [[0, 0, 0], [1, 1, 8], [8, 2, 0]]
    assert realizable.tolist() == [
        [False, False, False],
        [True, True, True],
        [True, True, False],
    ]


def test_vectorized_widths_match_canonical_hyperplane_derivation() -> None:
    margin = np.asarray([0.4, 1.2], dtype=np.float32)
    norms = np.asarray([2.0, 3.0], dtype=np.float32)
    pullback = np.asarray([[0.5, -0.25, 0.0], [0.1, 0.3, -0.6]], dtype=np.float32)
    lipschitz = np.asarray([0.5, 1.25], dtype=np.float32)
    actual = producer.derive_candidate_widths(margin, norms, pullback, lipschitz)
    expected = derive_hyperplane_channel_band(
        margin.reshape(1, 2),
        np.asarray([[1, 2]], dtype=np.int8),
        np.asarray([[0, 0]], dtype=np.int8),
        pullback.reshape(1, 2, 3),
        norms.reshape(1, 2),
        MarginBandConfig(scale=producer.SCALE, local_lipschitz=1.0, max_rgb_radius=producer.MAX_RGB_RADIUS),
        local_lipschitz_field=lipschitz.reshape(1, 2),
    ).channel_radii.reshape(2, 3)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_tensor_merkle_selects_native_margin_only_for_refresh_rows() -> None:
    rows = [
        {
            "pair_id": 0,
            "tensor_hashes": {"cached_margin": "a" * 64, "native_margin": "b" * 64},
        },
        {
            "pair_id": 1,
            "tensor_hashes": {
                "cached_margin": "c" * 64,
                "native_margin": "d" * 64,
                "cached_winner": "e" * 64,
            },
        },
    ]
    expected = hashlib.sha256(
        canonical_json([[0, "cached_margin", "a" * 64], [1, "native_margin", "d" * 64]])
    ).hexdigest()
    assert producer._merkle_tensor(rows, "margin", active_margin=True) == expected


def _pdw2_packet() -> bytes:
    target = make_gauge_fixed_affine_target(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([0.0, 0.25], dtype=np.float32),
    )
    return encode_pdw2(target)


def test_materializer_emits_valid_blocked_config_without_trainer_argv(
    monkeypatch, tmp_path: Path
) -> None:
    base = tmp_path / "archive.zip"
    with zipfile.ZipFile(base, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("0.bin", b"base")
    decoder = tmp_path / "inflate.py"
    decoder.write_text("pass\n")
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"cache")
    manifest = tmp_path / "band_manifest.json"
    manifest.write_bytes(b"{}")
    pdw2 = tmp_path / "seg_head_target.pdw2"
    pdw2.write_bytes(_pdw2_packet())
    fake_band = SimpleNamespace(
        mode="positive_anisotropic",
        manifest_path=manifest.resolve(),
        source_sha256="f" * 64,
    )
    monkeypatch.setattr(fire.BandArtifact, "load", lambda _path: fake_band)
    monkeypatch.setattr(fire, "_beneath_ssd", lambda path: Path(path).resolve())
    monkeypatch.setattr(
        fire,
        "storage_preflight",
        lambda path, required_free_bytes: {
            "path": str(Path(path).resolve()),
            "required_free_bytes": required_free_bytes,
            "free_bytes": required_free_bytes,
            "ok": True,
            "refusal_rc": 0,
        },
    )
    args = argparse.Namespace(
        output_dir=tmp_path / "out",
        cold_store=tmp_path / "cold",
        base_archive=base,
        base_decoder=decoder,
        cache=cache,
        band_manifest=manifest,
        pdw2_packet=pdw2,
        required_free_bytes=1,
        basis=BasisMode.RAW_CENTERED.value,
        resume_from=None,
        seed=producer.SEED,
        run_id="dry",
        pair_batch_size=2,
        checkpoint_every_steps=50,
        ema_decay=0.997,
    )
    config = fire.materialize(args)
    assert config["readiness"] == "BLOCKED"
    assert config["trainer_argv"] is None
    assert config["launch"] is False
    assert len(config["blocking_gates"]) == 3
    path = tmp_path / "fire.json"
    path.write_bytes(canonical_json(config))
    receipt = fire.validate_materialized(path)
    assert receipt["valid"] is True
    assert receipt["readiness"] == "BLOCKED"
    assert receipt["fire_executed"] is False
