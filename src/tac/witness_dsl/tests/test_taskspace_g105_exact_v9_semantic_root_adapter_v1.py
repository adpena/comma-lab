from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tac.boundary_math.lever_b_levelset_generator import (
    PolarDirectionalFourierBankConfig,
    build_coords,
    levelset_rgb_forward_numpy,
    polar_directional_fourier_B,
    polar_directional_fourier_feats,
)
from tac.witness_dsl import taskspace_g105_exact_v9_semantic_root_adapter_v1 as subject


def _fixture(*, film_per_layer: bool = True, film_concat_code: bool = True):
    basis = subject.V9PolarFourierConfigV1(
        n_scales=1,
        n_orient0=1,
        f0=1.25,
        base=2.0,
        n_iso=0,
        max_freq=None,
    )
    config = subject.V9RuntimeConfigV1(
        input_dim=basis.input_dim,
        hidden_dim=2,
        hidden_layer_count=2,
        modulation_dim=3,
        softmax_temp=0.75,
        hosc_beta=3.5,
        hosc_omega=1.25,
        chroma=True,
        film_per_layer=film_per_layer,
        film_concat_code=film_concat_code,
        basis=basis,
    )
    rng = np.random.default_rng(105)

    def array(shape, scale=0.2):
        return rng.normal(0.0, scale, size=shape).astype(np.float32)

    params = {
        "in_proj.weight": array((2, config.input_dim)),
        "in_proj.bias": array((2,)),
        "film.weight": array((8, 3), 0.05),
        "film.bias": array((8,), 0.05),
        "hidden.0.weight": array((2, 2)),
        "hidden.0.bias": array((2,)),
        "hidden.1.weight": array((2, 2)),
        "hidden.1.bias": array((2,)),
        "out_sdf.weight": array((5, 2)),
        "out_sdf.bias": array((5,)),
        "out_tex.weight": array((3, 2)),
        "out_tex.bias": array((3,)),
        "palette": array((5, 3), 0.4),
    }
    if film_per_layer:
        for layer in range(2):
            params[f"film_pl.{layer}.weight"] = array((4, 3), 0.05)
            params[f"film_pl.{layer}.bias"] = array((4,), 0.05)
    if film_concat_code:
        for layer in range(2):
            params[f"concat_pl.{layer}.weight"] = array((2, 3), 0.05)
            params[f"concat_pl.{layer}.bias"] = array((2,), 0.05)
    code = array((1200, 3), 0.1)
    program = subject.compile_from_state(
        config=config,
        params=params,
        interleaved_code=code,
    )
    return config, params, code, program


@pytest.mark.parametrize(
    ("film_per_layer", "film_concat_code"),
    [(False, False), (True, True)],
)
def test_independent_numpy_forward_is_exact_repo_v9_parity(
    film_per_layer: bool,
    film_concat_code: bool,
) -> None:
    config, _params, _code, program = _fixture(
        film_per_layer=film_per_layer,
        film_concat_code=film_concat_code,
    )
    repo_basis = polar_directional_fourier_B(
        PolarDirectionalFourierBankConfig(
            n_scales=config.basis.n_scales,
            n_orient0=config.basis.n_orient0,
            f0=config.basis.f0,
            base=config.basis.base,
            n_iso=config.basis.n_iso,
        ),
        max_freq=config.basis.max_freq,
    )
    repo_features = polar_directional_fourier_feats(
        build_coords(subject.SCORER_H, subject.SCORER_W),
        repo_basis,
    )
    own_features = subject.build_runtime_features(config)
    assert np.array_equal(own_features, repo_features)

    own_rgb, own_phi = subject.forward_float32(program, 17)
    repo_rgb, repo_phi = levelset_rgb_forward_numpy(
        program.params,
        repo_features,
        program.y1_code[17],
        n_hidden=config.hidden_layer_count,
        hidden_dim=config.hidden_dim,
        n_classes=5,
        activation="hosc",
        softmax_temp=config.softmax_temp,
        wire_w0=20.0,
        wire_s0=10.0,
        hosc_beta=config.hosc_beta,
        hosc_omega=config.hosc_omega,
        chroma=config.chroma,
    )
    assert np.array_equal(own_rgb, repo_rgb)
    assert np.array_equal(own_phi, repo_phi)
    expected_u8 = np.clip(np.rint(repo_rgb), 0.0, 255.0).astype(np.uint8).reshape(384, 512, 3)
    assert np.array_equal(subject.render_scorer_y1(program, 17), expected_u8)


def test_packet_is_strict_and_counts_only_odd_y1_rows() -> None:
    config, params, code, program = _fixture()
    packet = subject.encode_packet(program)
    parsed = subject.parse_packet(packet)
    assert subject.encode_packet(parsed) == packet
    assert parsed.config == config

    changed_even = code.copy()
    changed_even[0::2] += 123.0
    changed_even_packet = subject.encode_packet(
        subject.compile_from_state(
            config=config,
            params=params,
            interleaved_code=changed_even,
        )
    )
    assert changed_even_packet == packet

    changed_odd = code.copy()
    changed_odd[1, 0] += 1.0
    changed_odd_packet = subject.encode_packet(
        subject.compile_from_state(
            config=config,
            params=params,
            interleaved_code=changed_odd,
        )
    )
    assert changed_odd_packet != packet

    accounting = subject.candidate_wire_accounting(packet)
    assert accounting.counted_y1_rows == 600
    assert accounting.excluded_y0_rows == 600
    assert accounting.y1_code_data_bytes == 600 * config.modulation_dim * 2
    assert accounting.packet_bytes == (
        accounting.header_bytes
        + accounting.section_directory_bytes
        + accounting.config_bytes
        + accounting.model_section_bytes
        + accounting.y1_section_bytes
    )
    assert accounting.model_section_bytes == (
        accounting.model_tensor_data_bytes + accounting.model_tensor_metadata_bytes
    )
    assert accounting.y1_section_bytes == (
        accounting.y1_code_data_bytes + accounting.y1_code_metadata_bytes
    )
    assert accounting.to_dict()["outer_zip_bytes_measured"] is False
    assert accounting.to_dict()["candidate_or_score_claim"] is False


def test_packet_corruption_and_original_coord_inr_cross_cast_fail_closed() -> None:
    _config, _params, _code, program = _fixture()
    packet = subject.encode_packet(program)
    corrupt = bytearray(packet)
    corrupt[-1] ^= 1
    with pytest.raises(subject.ExactV9SemanticRootError, match="CRC"):
        subject.parse_packet(bytes(corrupt))
    with pytest.raises(subject.ExactV9SemanticRootError, match="header"):
        subject.parse_packet(b"TSR1PKT1" + packet[8:])


def _sha(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _target_evidence() -> subject.V9G46Batch16TrainingTargetEvidenceV1:
    return subject.V9G46Batch16TrainingTargetEvidenceV1(
        active_target_authority_sha256=_sha("active-target"),
        target_margins_sha256=_sha("batch16-margins"),
        margin_aggregate_receipt_sha256=_sha("margin-aggregate"),
        consumer_binding_sha256=_sha("consumer-binding"),
        evidence_sha256=_sha("target-evidence"),
    )


def test_g46_receipt_intake_uses_canonical_compile_ready_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = tmp_path / "12_encoder_only_receipt.json"
    receipt_path.write_text("{}\n", encoding="utf-8")
    observed: list[Path] = []

    def _canonical_loader(path: Path) -> dict[str, object]:
        observed.append(path)
        raise subject.FreshTeacherMaterializationError("canonical sentinel")

    monkeypatch.setattr(subject, "load_compile_ready_materialization_receipt", _canonical_loader)
    with pytest.raises(subject.ExactV9SemanticRootError, match="canonically compile-ready"):
        subject._portable_g46_receipt_identity(receipt_path)
    assert observed == [receipt_path]


@pytest.mark.parametrize(
    "persisted_name",
    ["legacy_fourier_ab_control", "polar_fourier", "polar_directional_fourier"],
)
def test_checkpoint_basis_names_use_canonical_exact_polar_alias(
    persisted_name: str,
) -> None:
    assert subject._require_exact_polar_basis(persisted_name) == "legacy_fourier_ab_control"


def test_checkpoint_basis_refuses_nonpolar_family() -> None:
    with pytest.raises(subject.ExactV9SemanticRootError, match="exact polar Fourier ABI"):
        subject._require_exact_polar_basis("windowed_curvelet")


def test_fresh_producer_gate_requires_g46_margins_and_live_verdict_batch16() -> None:
    evidence = _target_evidence()
    scalars = {
        "__cfg_target_authority_sha256": evidence.active_target_authority_sha256,
        "__cfg_g46_target_labels_sha256": evidence.target_labels_sha256,
        "__cfg_g46_target_margins_sha256": evidence.target_margins_sha256,
        "__cfg_g46_source_pair_chain_sha256": evidence.source_pair_chain_sha256,
        "__cfg_g46_margin_aggregate_schema": evidence.margin_aggregate_schema,
        "__cfg_g46_margin_aggregate_sha256": evidence.margin_aggregate_receipt_sha256,
        "__cfg_g46_target_consumer_binding_sha256": evidence.consumer_binding_sha256,
        "__cfg_g46_target_evidence_sha256": evidence.evidence_sha256,
        "__cfg_g46_target_scorer_batch_size": 16,
        "__cfg_g46_margin_same_forward": 1,
        "__cfg_verdict_batch": 16,
    }
    subject._validate_checkpoint_target_binding(scalars, evidence)

    for key, wrong in (
        ("__cfg_g46_target_labels_sha256", _sha("legacy-labels")),
        ("__cfg_g46_target_margins_sha256", _sha("batch32-margins")),
        ("__cfg_g46_margin_same_forward", 0),
        ("__cfg_verdict_batch", 32),
    ):
        mutated = {**scalars, key: wrong}
        with pytest.raises(subject.ExactV9SemanticRootError, match=key):
            subject._validate_checkpoint_target_binding(mutated, evidence)

    with pytest.raises(subject.ExactV9SemanticRootError, match="batch geometry"):
        replace(evidence, live_verdict_batch_size=32)
