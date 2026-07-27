from __future__ import annotations

import io
import zipfile
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
    partitioned_packet = subject.encode_packet(
        subject.compile_from_y1_state(
            config=config,
            params=params,
            y1_code=code[1::2],
        )
    )
    assert partitioned_packet == packet
    with pytest.raises(
        subject.ExactV9SemanticRootError,
        match=r"float\[600,modulation_dim\]",
    ):
        subject.compile_from_y1_state(
            config=config,
            params=params,
            y1_code=code,
        )

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
    assert accounting.y1_code_data_bytes < 600 * config.modulation_dim * 2
    assert accounting.y1_code_metadata_bytes == subject._Y1_HEADER.size
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


def test_y1_entropy_codec_arbitrates_rice_against_raw_int16() -> None:
    _config, _params, _code, program = _fixture()
    rng = np.random.default_rng(17)
    high_entropy = rng.integers(
        -32768,
        32768,
        size=program.y1_code_q.shape,
        dtype=np.int16,
    )
    packet = subject.encode_packet(
        replace(program, y1_code_q=high_entropy)
    )
    parsed = subject.parse_packet(packet)
    assert np.array_equal(parsed.y1_code_q, high_entropy)
    _flags, sections = subject._split_sections(packet)
    (
        _magic,
        _pair_count,
        _modulation_dim,
        _exponent,
        codec,
        rice_k,
        byte_length,
    ) = subject._Y1_HEADER.unpack_from(sections[b"Y1CD"])
    assert codec == subject._Y1_CODEC_RAW_I16_LE
    assert rice_k == 0
    assert byte_length == high_entropy.nbytes


def _deterministic_single_member_zip(payload: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        info = zipfile.ZipInfo(
            "candidate.packet",
            date_time=(1980, 1, 1, 0, 0, 0),
        )
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(
            info,
            payload,
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )
    return output.getvalue()


def test_y1_wire_variants_preserve_tag_and_expose_outer_zip_inversion() -> None:
    _config, _params, _code, program = _fixture()
    alternating = np.broadcast_to(
        ((np.arange(600) % 2) * 2000 - 1000).astype("<i2")[:, None],
        program.y1_code_q.shape,
    ).copy()
    selected = replace(
        program,
        y1_code_q=np.ascontiguousarray(alternating, dtype="<i2"),
    )
    variants = dict(subject.encode_packet_y1_variants(selected))
    raw = variants[subject.Y1WireCodecV1.RAW_I16_LE]
    rice = variants[subject.Y1WireCodecV1.DELTA_RICE_BEST_K]

    assert len(rice) < len(raw)
    assert len(_deterministic_single_member_zip(raw)) < len(
        _deterministic_single_member_zip(rice)
    )
    for codec, packet in variants.items():
        parsed = subject.parse_packet(packet)
        assert parsed.y1_wire_codec is codec
        assert np.array_equal(parsed.y1_code_q, alternating)
        assert subject.encode_packet(parsed) == packet
    assert subject.render_scorer_y1(subject.parse_packet(raw), 37).tobytes() == (
        subject.render_scorer_y1(subject.parse_packet(rice), 37).tobytes()
    )


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


def _fresh_root_custody_scalars() -> dict[str, object]:
    projection_sha = _sha("physical-g109-projection")
    scalars: dict[str, object] = {
        "__cfg_fresh_producer": 1,
        "__cfg_fresh_lineage_schema": "tac.fresh_producer_lineage.v1",
        "__cfg_fresh_seed": 105,
        "__cfg_fresh_initial_state_sha256": _sha("initial-state"),
        "__cfg_fresh_dsl_compile_hash": _sha("root-dsl"),
        "__cfg_fresh_target_projection_sha256": projection_sha,
        "__cfg_fresh_current_launch_dsl_compile_hash": _sha("launch-dsl"),
        subject.CHECKPOINT_PROJECTION_SHA_KEY: projection_sha,
    }
    scalars["__cfg_fresh_lineage_root_sha256"] = subject._sha256(
        subject._canonical_json(
            {
                "schema": scalars["__cfg_fresh_lineage_schema"],
                "seed": scalars["__cfg_fresh_seed"],
                "dsl_compile_hash": scalars[
                    "__cfg_fresh_dsl_compile_hash"
                ],
                "target_projection_sha256": scalars[
                    "__cfg_fresh_target_projection_sha256"
                ],
                "initial_state_sha256": scalars[
                    "__cfg_fresh_initial_state_sha256"
                ],
            }
        )
    )
    return scalars


def test_fresh_root_custody_binds_typed_hashes_to_physical_g109_projection() -> None:
    scalars = _fresh_root_custody_scalars()
    custody = subject._validate_fresh_lineage_root_custody(scalars)
    assert custody["schema"] == "tac.fresh_producer_lineage.v1"
    assert custody["fresh_lineage_root_sha256"] == scalars[
        "__cfg_fresh_lineage_root_sha256"
    ]
    assert custody["physical_target_projection_sha256"] == scalars[
        subject.CHECKPOINT_PROJECTION_SHA_KEY
    ]
    assert custody["node_metadata_current_launch_dsl_compile_hash"] == scalars[
        "__cfg_fresh_current_launch_dsl_compile_hash"
    ]
    assert custody["root_self_consistent"] is True
    assert custody["origin_proven"] is False
    assert custody["complete_trajectory_proven"] is False

    for key in (
        "__cfg_fresh_producer",
        "__cfg_fresh_lineage_schema",
        "__cfg_fresh_seed",
        "__cfg_fresh_lineage_root_sha256",
        "__cfg_fresh_initial_state_sha256",
        "__cfg_fresh_dsl_compile_hash",
        "__cfg_fresh_target_projection_sha256",
        "__cfg_fresh_current_launch_dsl_compile_hash",
        subject.CHECKPOINT_PROJECTION_SHA_KEY,
    ):
        mutated = dict(scalars)
        mutated.pop(key)
        with pytest.raises(subject.ExactV9SemanticRootError):
            subject._validate_fresh_lineage_root_custody(mutated)

    copied_marker = dict(scalars)
    copied_marker["__cfg_fresh_target_projection_sha256"] = _sha(
        "foreign-g109-projection"
    )
    with pytest.raises(subject.ExactV9SemanticRootError, match="physical G109"):
        subject._validate_fresh_lineage_root_custody(copied_marker)

    forged_root = dict(scalars)
    forged_root["__cfg_fresh_lineage_root_sha256"] = _sha("forged-root")
    with pytest.raises(subject.ExactV9SemanticRootError, match="does not recompute"):
        subject._validate_fresh_lineage_root_custody(forged_root)


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


def test_fresh_producer_gate_requires_physical_g109_and_live_verdict_batch16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _target_evidence()
    scalars = {
        subject.CHECKPOINT_PROJECTION_KEY: "{}",
        subject.CHECKPOINT_PROJECTION_SHA_KEY: "f" * 64,
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

    monkeypatch.setattr(
        subject,
        "reopen_v9_training_target_projection",
        lambda **_kwargs: {
            "aggregate_receipt": {
                "path": "/physical/g109.json",
                "bytes": 1,
                "sha256": "e" * 64,
            }
        },
    )
    monkeypatch.setattr(
        subject,
        "checkpoint_target_arrays_from_projection",
        lambda *_args, **_kwargs: {
            key: np.asarray(value) for key, value in scalars.items()
        },
    )
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


def test_legacy_top_level_compiler_refuses_unpartitioned_g111_pose_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase = subject.V9PhaseAdvectionTrainingEvidenceV1(
        weight=1.0,
        start_epoch=0,
        classes=(0, 1, 2),
        band=1.0,
        gap_xi="interp",
        reference="gt_advected",
        evidence_sha256=_sha("phase"),
    )
    monkeypatch.setattr(
        subject,
        "_portable_g46_receipt_identity",
        lambda _path: {},
    )
    monkeypatch.setattr(
        subject,
        "_checkpoint_runtime_state",
        lambda _path: (
            {
                "pose_carrier.xi_stored": np.zeros((600, 6), dtype=np.float32),
                "pose_carrier.dxi": np.zeros((600, 6), dtype=np.float32),
            },
            {},
            "a" * 64,
            1,
        ),
    )
    with pytest.raises(
        subject.ExactV9SemanticRootError,
        match="partition it through G112",
    ):
        subject.compile_fresh_checkpoint(
            checkpoint=Path("/not-opened-g111.npz"),
            g46_encoder_receipt=Path("/not-opened-g46.json"),
            phase_advection=phase,
            training_target=_target_evidence(),
        )
