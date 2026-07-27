# SPDX-License-Identifier: MIT
"""G110 generated-Y1 pose packet, exact public warp, and fail-closed custody."""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from tac.boundary_math.xi_pose_coder import (
    quantize_xi,
    serialize_xi_payload,
)
from tac.witness_dsl import (
    taskspace_g110_generated_y1_pose_product_v1 as pose_product,
)
from tac.witness_dsl import (
    taskspace_g110_generic_two_layer_public_product_v1 as product,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = REPO_ROOT / product.PUBLIC_RUNTIME_RELATIVE_ROOT


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    prior = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = prior
    return module


def _g105_packet() -> bytes:
    fixture = _module(
        REPO_ROOT
        / "src"
        / "tac"
        / "witness_dsl"
        / "tests"
        / "test_taskspace_g105_exact_v9_semantic_root_adapter_v1.py",
        "_g110_pose_g105_fixture",
    )
    return fixture.subject.encode_packet(fixture._fixture()[3])


def _xi() -> np.ndarray:
    pair = np.arange(600, dtype=np.float64)
    xi = np.zeros((600, 6), dtype=np.float64)
    xi[:, 0] = 0.00001 * pair
    xi[:, 1] = 0.00002 * np.sin(pair / 17.0)
    xi[:, 2] = 0.00003 * np.cos(pair / 31.0)
    xi[:, 3] = 0.000001 * (pair % 13)
    xi[:, 4] = -0.000001 * (pair % 7)
    xi[:, 5] = 0.000001 * np.sin(pair / 11.0)
    return xi


def _packet() -> bytes:
    semantic = _g105_packet()
    q, scales = quantize_xi(_xi(), q_levels=4096)
    xip2 = serialize_xi_payload(q, scales, coder="delta_ar_zlib")
    return pose_product._encode_packet(
        semantic_packet=semantic,
        final_y1_binding=hashlib.sha256(b"binding").hexdigest(),
        xip2_payload=xip2,
        pitch=0.0,
    )


def test_public_pose_plugin_has_no_undeclared_brotli_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = (
        RUNTIME_ROOT
        / "frame0_variants"
        / "generated_y1_pose_xip2_v1.py"
    )
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "brotli" or name.startswith("brotli."):
            raise ModuleNotFoundError("clean upstream has no brotli")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    public = _module(plugin, "_g110_public_without_brotli")
    assert public.accepts_packet(_packet()) is True
    assert public.parse_packet(_packet())["xi"].shape == (600, 6)


def test_public_inflate_prepares_clean_and_repeat_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public = _module(
        RUNTIME_ROOT / "inflate.py",
        "_g110_public_output_contract",
    )
    monkeypatch.setattr(public, "EXPECTED_RAW_BYTES", 17)
    monkeypatch.setattr(public, "MIN_OUTPUT_HEADROOM_BYTES", 0)
    output_root = tmp_path / "inflated"
    final_path, temporary_path = public._prepare_output(
        output_root,
        "video.raw",
    )
    assert output_root.is_dir()
    assert final_path == output_root / "video.raw"
    assert temporary_path == output_root / ".video.raw.g110.tmp"

    final_path.write_bytes(b"x" * 17)
    repeated_final, repeated_temporary = public._prepare_output(
        output_root,
        "video.raw",
    )
    assert repeated_final == final_path
    assert repeated_temporary == temporary_path

    (output_root / "foreign.bin").write_bytes(b"x")
    with pytest.raises(
        public.PublicInflateError,
        match="outside the exact G110 product",
    ):
        public._prepare_output(output_root, "video.raw")


def test_pose_packet_xip2_and_archive_are_canonical() -> None:
    packet = _packet()
    parsed = pose_product.parse_g110_generated_y1_pose_v1(packet)
    assert parsed.semantic_packet == _g105_packet()
    assert parsed.q.shape == (600, 6)
    assert parsed.scales.shape == (6,)
    assert parsed.xi_eff.shape == (600, 6)
    assert parsed.xip2_payload.startswith(b"XIP2")
    archive = pose_product.build_g110_generated_y1_pose_archive(packet)
    assert pose_product.parse_g110_generated_y1_pose_archive(archive) == packet
    assert archive == pose_product.build_g110_generated_y1_pose_archive(packet)

    corrupted = bytearray(packet)
    corrupted[-1] ^= 1
    with pytest.raises(pose_product.G110GeneratedY1PoseError, match="CRC32"):
        pose_product.parse_g110_generated_y1_pose_v1(bytes(corrupted))

    parsed_xip2 = parsed.xip2_payload + b"\x00"
    trailing = pose_product._encode_packet(
        semantic_packet=parsed.semantic_packet,
        final_y1_binding=parsed.final_y1_binding_sha256,
        xip2_payload=parsed_xip2,
        pitch=parsed.pitch,
    )
    with pytest.raises(
        pose_product.G110GeneratedY1PoseError,
        match="trailing/noncanonical",
    ):
        pose_product.parse_g110_generated_y1_pose_v1(trailing)


def test_public_pose_plugin_matches_numpy_authority_on_actual_camera_y1() -> None:
    packet = _packet()
    parsed = pose_product.parse_g110_generated_y1_pose_v1(packet)
    public = _module(
        RUNTIME_ROOT / "frame0_variants" / "generated_y1_pose_xip2_v1.py",
        "_g110_public_generated_y1_pose",
    )
    assert public.accepts_packet(packet) is True
    state = public.parse_packet(packet)
    assert public.semantic_packet(state) == parsed.semantic_packet

    rows = np.arange(874, dtype=np.uint16)[:, None]
    columns = np.arange(1164, dtype=np.uint16)[None, :]
    camera_y1 = np.empty((874, 1164, 3), dtype=np.uint8)
    camera_y1[..., 0] = (rows + columns) % 256
    camera_y1[..., 1] = (2 * rows + columns) % 256
    camera_y1[..., 2] = (rows + 3 * columns) % 256
    scorer_y1 = np.zeros((384, 512, 3), dtype=np.uint8)

    pair_id = 173
    expected = parsed.render_camera_y0(pair_id, camera_y1)
    observed = public.render_camera_y0(
        state,
        pair_id,
        scorer_y1,
        camera_y1,
    )
    assert np.array_equal(observed, expected)
    assert observed.dtype == np.uint8
    assert observed.flags.c_contiguous


def test_pose_public_binding_and_frame0_dispatch_are_unambiguous() -> None:
    packet = _packet()
    parsed = pose_product.parse_g110_generated_y1_pose_v1(packet)
    public = _module(
        RUNTIME_ROOT / "frame0_variants" / "generated_y1_pose_xip2_v1.py",
        "_g110_public_generated_y1_pose_binding",
    )
    public.parse_packet(packet)
    semantic_digest = hashlib.sha256()
    provider = product.open_final_y1_provider(parsed.semantic_packet)
    for pair_id in range(600):
        y1 = provider.render_scorer_y1(pair_id)
        semantic_digest.update(pair_id.to_bytes(2, "big"))
        semantic_digest.update(memoryview(y1).cast("B"))
    expected_binding = hashlib.sha256(
        product.FINAL_Y1_DOMAIN
        + hashlib.sha256(parsed.semantic_packet).digest()
        + semantic_digest.digest()
    ).hexdigest()
    packet_with_binding = pose_product._encode_packet(
        semantic_packet=parsed.semantic_packet,
        final_y1_binding=expected_binding,
        xip2_payload=parsed.xip2_payload,
        pitch=parsed.pitch,
    )
    bound_state = public.parse_packet(packet_with_binding)
    public.verify_final_y1_population(bound_state, semantic_digest.digest())
    wrong = bytearray(semantic_digest.digest())
    wrong[0] ^= 1
    with pytest.raises(ValueError, match="binding differs"):
        public.verify_final_y1_population(bound_state, bytes(wrong))

    lowrank = _module(
        RUNTIME_ROOT / "frame0_variants" / "conditional_lowrank_rice_v1.py",
        "_g110_public_lowrank_dispatch",
    )
    assert public.accepts_packet(packet) is True
    assert lowrank.accepts_packet(packet) is False
    assert lowrank.accepts_packet(b"G110TL01" + b"\x00" * 100) is True
    assert public.accepts_packet(b"G110TL01" + b"\x00" * 100) is False


def test_public_g105_preserves_both_outer_selected_y1_wire_families() -> None:
    semantic = _g105_packet()
    program = pose_product.parse_v9_packet(semantic)
    public = _module(
        RUNTIME_ROOT / "semantic_variants" / "v9_hosc_dual_head_odd_y1_v1.py",
        "_g110_public_g105_outer_wire_variants",
    )
    variants = pose_product.encode_packet_y1_variants(program)
    assert tuple(codec for codec, _packet in variants) == tuple(
        pose_product.Y1WireCodecV1
    )
    reference_frames = {
        pair_id: product.open_final_y1_provider(semantic).render_scorer_y1(
            pair_id
        )
        for pair_id in (0, 17, 599)
    }
    for codec, packet in variants:
        repo_parsed = pose_product.parse_v9_packet(packet)
        public_parsed = public.parse_packet(packet)
        assert repo_parsed.y1_wire_codec is codec
        assert int(public_parsed.y1_wire_codec) == int(codec)
        assert public.encode_packet(public_parsed) == packet
        for pair_id, expected in reference_frames.items():
            assert np.array_equal(
                public.render_scorer_y1(public_parsed, pair_id),
                expected,
            )


def test_public_g105_reuses_parse_time_invariants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _g105_packet()
    public = _module(
        RUNTIME_ROOT / "semantic_variants" / "v9_hosc_dual_head_odd_y1_v1.py",
        "_g110_public_g105_cached_invariants",
    )
    parsed = public.parse_packet(packet)
    features_identity = id(parsed._runtime_features_f64)
    params_identities = {
        name: id(value)
        for name, value in parsed._runtime_params_f64.items()
    }
    expected = public.render_scorer_y1(parsed, 17)

    def refuse_rebuild(_config: object) -> np.ndarray:
        raise AssertionError("pair render rebuilt invariant Fourier features")

    monkeypatch.setattr(public, "build_runtime_features", refuse_rebuild)
    assert np.array_equal(public.render_scorer_y1(parsed, 17), expected)
    assert id(parsed._runtime_features_f64) == features_identity
    assert {
        name: id(value)
        for name, value in parsed._runtime_params_f64.items()
    } == params_identities


def test_public_g105_refuses_unbounded_invariant_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public = _module(
        RUNTIME_ROOT / "semantic_variants" / "v9_hosc_dual_head_odd_y1_v1.py",
        "_g110_public_g105_cache_bound",
    )
    monkeypatch.setattr(public, "MAX_RUNTIME_FEATURE_CACHE_BYTES", 1)
    with pytest.raises(
        public.ExactV9SemanticRootError,
        match="bounded receiver envelope",
    ):
        public.parse_packet(_g105_packet())


def test_complete_archive_arbitration_uses_exact_outer_zip_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic = _g105_packet()
    program = pose_product.parse_v9_packet(semantic)
    expected_population = hashlib.sha256(b"same-rendered-y1").digest()
    monkeypatch.setattr(
        pose_product,
        "_population_digest",
        lambda _provider: expected_population,
    )
    q, scales = quantize_xi(_xi(), q_levels=4096)
    xip2 = serialize_xi_payload(q, scales, coder="delta_ar_zlib")
    packet, archive, selected, records = (
        pose_product._select_complete_archive_y1_wire(
            semantic_program=program,
            expected_population_digest=expected_population,
            xip2_payload=xip2,
            pitch=0.0,
        )
    )
    assert len(records) == (
        len(pose_product.Y1WireCodecV1)
        * len(pose_product.G110OuterZipMethodV1)
    ) == 4
    assert {record.y1_wire_codec for record in records} == set(
        pose_product.Y1WireCodecV1
    )
    assert {record.outer_zip_method for record in records} == set(
        pose_product.G110OuterZipMethodV1
    )
    assert {
        (record.y1_wire_codec, record.outer_zip_method)
        for record in records
    } == {
        (codec, method)
        for codec in pose_product.Y1WireCodecV1
        for method in pose_product.G110OuterZipMethodV1
    }
    assert selected.archive_bytes == min(
        record.archive_bytes for record in records
    )
    expected_selected = min(
        records,
        key=lambda record: (
            record.archive_bytes,
            int(record.y1_wire_codec),
            int(record.outer_zip_method),
            record.archive_sha256,
        ),
    )
    assert selected == expected_selected
    assert len(archive) == selected.archive_bytes
    assert hashlib.sha256(archive).hexdigest() == selected.archive_sha256
    assert pose_product.parse_g110_generated_y1_pose_archive(archive) == packet
    parsed_product = pose_product.parse_g110_generated_y1_pose_v1(packet)
    parsed_semantic = pose_product.parse_v9_packet(
        parsed_product.semantic_packet
    )
    assert parsed_semantic.y1_wire_codec is selected.y1_wire_codec


def test_outer_zip_canonical_arbitration_selects_deflate_when_smaller() -> None:
    fixture = _module(
        REPO_ROOT
        / "src"
        / "tac"
        / "witness_dsl"
        / "tests"
        / "test_taskspace_g105_exact_v9_semantic_root_adapter_v1.py",
        "_g110_pose_store_wins_fixture",
    )
    base_config = fixture._fixture()[0]
    config = replace(
        base_config,
        hidden_dim=512,
        hidden_layer_count=4,
        modulation_dim=64,
        film_per_layer=False,
        film_concat_code=False,
    )
    hidden = config.hidden_dim
    layers = config.hidden_layer_count
    modulation = config.modulation_dim
    params = {
        "in_proj.weight": np.zeros(
            (hidden, config.input_dim),
            dtype=np.float32,
        ),
        "in_proj.bias": np.zeros((hidden,), dtype=np.float32),
        "film.weight": np.zeros(
            (2 * hidden * layers, modulation),
            dtype=np.float32,
        ),
        "film.bias": np.zeros(
            (2 * hidden * layers,),
            dtype=np.float32,
        ),
        "out_sdf.weight": np.zeros((5, hidden), dtype=np.float32),
        "out_sdf.bias": np.zeros((5,), dtype=np.float32),
        "out_tex.weight": np.zeros((3, hidden), dtype=np.float32),
        "out_tex.bias": np.zeros((3,), dtype=np.float32),
        "palette": np.zeros((5, 3), dtype=np.float32),
    }
    for layer in range(layers):
        params[f"hidden.{layer}.weight"] = np.zeros(
            (hidden, hidden),
            dtype=np.float32,
        )
        params[f"hidden.{layer}.bias"] = np.zeros(
            (hidden,),
            dtype=np.float32,
        )
    program = pose_product.compile_v9_from_state(
        config=config,
        params=params,
        interleaved_code=np.zeros(
            (1200, modulation),
            dtype=np.float32,
        ),
    )
    rng = np.random.default_rng(110)
    program = replace(
        program,
        tensors=tuple(
            replace(tensor, data=rng.bytes(len(tensor.data)))
            for tensor in program.tensors
        ),
        y1_code_q=np.ascontiguousarray(
            rng.integers(
                -32768,
                32768,
                size=program.y1_code_q.shape,
                dtype=np.int16,
            ),
            dtype="<i2",
        ),
    )
    semantic_packet = pose_product.encode_packet_y1_variants(program)[0][1]
    xip2 = serialize_xi_payload(
        rng.integers(
            -32768,
            32768,
            size=(600, 6),
            dtype=np.int16,
        ),
        np.ones(6, dtype=np.float64),
        coder="delta_ar_zlib",
    )
    packet = pose_product._encode_packet(
        semantic_packet=semantic_packet,
        final_y1_binding="1" * 64,
        xip2_payload=xip2,
        pitch=0.0,
    )
    stored = pose_product._build_g110_archive_for_method(
        packet,
        pose_product.G110OuterZipMethodV1.STORE,
    )
    deflated = pose_product._build_g110_archive_for_method(
        packet,
        pose_product.G110OuterZipMethodV1.DEFLATE,
    )
    assert len(deflated) < len(stored)
    assert pose_product.build_g110_generated_y1_pose_archive(packet) == deflated
    assert pose_product.parse_g110_generated_y1_pose_archive(deflated) == packet
    with pytest.raises(
        pose_product.G110GeneratedY1PoseError,
        match="canonical method/layout",
    ):
        pose_product.parse_g110_generated_y1_pose_archive(stored)


def test_g111_pose_partition_is_total_disjoint_and_fail_closed() -> None:
    semantic = {
        "code": np.zeros((1200, 4), dtype=np.float32),
        "in_proj.weight": np.zeros((4, 4), dtype=np.float32),
    }
    pose = {
        "pose_carrier.xi_stored": np.zeros((600, 6), dtype=np.float32),
        "pose_carrier.dxi": np.zeros((600, 6), dtype=np.float32),
    }
    scalars: dict[str, object] = {
        "__cfg_pose_carrier_contract_schema": (
            product.G111_POSE_CHECKPOINT_CONTRACT_SCHEMA
        ),
        "__cfg_pose_carrier": 1,
        "__cfg_pose_carrier_source": "generated_y1",
        "__cfg_pose_carrier_residual_mode": "table",
        "__cfg_pose_carrier_xi_formula": "xi_stored+residual_scale*dxi",
        "__cfg_pose_carrier_y1_selected_preimage_schema": (
            product.G111_POSE_Y1_SELECTED_PREIMAGE_SCHEMA
        ),
        "__cfg_pose_carrier_native_hw": np.asarray([874, 1164], dtype=np.int64),
        "__cfg_pose_carrier_residual_scale": 1.0,
        "__cfg_pose_carrier_s_t": 0.044,
        "__cfg_pose_carrier_s_r": 0.0,
        "__cfg_pose_carrier_pitch": 0.0,
    }
    semantic_out, pose_out = product._partition_g111_checkpoint_params(
        {**semantic, **pose},
        scalars,
    )
    assert set(semantic_out) == set(semantic)
    assert set(pose_out) == product.G111_POSE_PARAM_KEYS
    assert not set(semantic_out).intersection(pose_out)
    assert set(semantic_out).union(pose_out) == set(semantic).union(pose)

    with pytest.raises(product.G110TwoLayerError, match="partial"):
        product._partition_g111_checkpoint_params(
            {**semantic, "pose_carrier.dxi": pose["pose_carrier.dxi"]},
            scalars,
        )
    wrong_source = dict(scalars)
    wrong_source["__cfg_pose_carrier_source"] = "generated"
    with pytest.raises(product.G110TwoLayerError, match="custody"):
        product._partition_g111_checkpoint_params(
            {**semantic, **pose},
            wrong_source,
        )


def test_hash_only_initializer_custody_is_forbidden() -> None:
    with pytest.raises(TypeError, match="physical_checkpoint"):
        pose_product.G111GeneratedY1PoseInitializerCustodyV1(
            checkpoint_sha256="0" * 64,
            semantic_packet_sha256="1" * 64,
            xi_initializer_sha256="2" * 64,
            even_code_exclusion_sha256="3" * 64,
            pitch=0.0,
            residual_scale=1.0,
            tensor_partition_sha256="4" * 64,
            xi_initializer=np.zeros((600, 6), dtype=np.float64),
        )
    with pytest.raises(TypeError, match="physical_partition_receipt"):
        pose_product.G110G112CompileCustodyV1(
            partition_receipt_sha256="0" * 64,
            semantic_child_sha256="0" * 64,
            pose_initializer_sha256="1" * 64,
            semantic_packet_sha256="2" * 64,
            target_projection_sha256="3" * 64,
            target_capsule_receipt_sha256="4" * 64,
            pose_targets_sha256="5" * 64,
            semantic_child=object(),
            pose_initializer=object(),
        )


def test_official_compile_accepts_only_recursive_g112_partition_receipt() -> None:
    parameters = inspect.signature(
        pose_product.compile_g110_generated_y1_pose_v1
    ).parameters
    assert "g112_partition_receipt" in parameters
    assert "expected_g112_partition_receipt_sha256" in parameters
    assert "g112_semantic_child" not in parameters
    assert "expected_g112_semantic_child_sha256" not in parameters
    assert "g112_pose_initializer" not in parameters
    assert "expected_g112_pose_initializer_sha256" not in parameters
    assert "fresh_g111_checkpoint" not in parameters
    assert "semantic_packet" not in parameters
    assert "expected_post_g105_refit_checkpoint_sha256" in parameters
    assert "expected_post_g105_refit_run_receipt_sha256" in parameters


def test_g110_requires_complete_recursive_g112_source_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    partition_receipt = (tmp_path / "30_g112_partition_receipt.json").resolve()
    partition_receipt.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(
        pose_product,
        "open_g112_partition_receipt",
        lambda *_args, **_kwargs: SimpleNamespace(
            source_chain=SimpleNamespace(
                complete_trajectory_proven=False,
                nodes=(),
                current=None,
            )
        ),
    )
    with pytest.raises(
        pose_product.G110GeneratedY1PoseError,
        match="complete physical trajectory",
    ):
        pose_product.G110G112CompileCustodyV1.from_physical_partition_receipt(
            partition_receipt_path=partition_receipt,
            expected_partition_receipt_sha256=hashlib.sha256(
                partition_receipt.read_bytes()
            ).hexdigest(),
            target_capsule_receipt=tmp_path / "unused-g109.json",
            expected_target_capsule_receipt_sha256="1" * 64,
        )


@pytest.mark.parametrize(
    "selected_xip2_coder",
    ["none", "delta_ar_zlib"],
)
def test_post_g105_refit_reopens_resumable_exact_public_stage(
    tmp_path: Path,
    selected_xip2_coder: str,
) -> None:
    semantic = b"semantic"
    semantic_sha = hashlib.sha256(semantic).hexdigest()
    final_y1_binding = hashlib.sha256(b"final-y1").hexdigest()
    target_sha = hashlib.sha256(b"target").hexdigest()
    pose_sha = hashlib.sha256(b"poses").hexdigest()
    initializer_sha = pose_product._xi_digest(_xi())
    semantic_child_sha = hashlib.sha256(b"semantic-child").hexdigest()
    pose_initializer_artifact_sha = hashlib.sha256(
        b"pose-initializer-artifact"
    ).hexdigest()
    partition_receipt_sha = hashlib.sha256(
        b"g112-partition-receipt"
    ).hexdigest()
    deploy_sha = hashlib.sha256(b"g111-deploy").hexdigest()
    resume_sha = hashlib.sha256(b"g111-resume").hexdigest()
    lineage_receipt_sha = hashlib.sha256(b"g111-lineage").hexdigest()
    checkpoint_id_sha = hashlib.sha256(b"g111-checkpoint-id").hexdigest()
    root_sha = hashlib.sha256(b"g111-root").hexdigest()
    target_projection_sha = hashlib.sha256(
        b"target-projection"
    ).hexdigest()
    run_id = "g110-post-g105-refit-test"
    seed = 110
    xi_eff = _xi()

    checkpoint = (tmp_path / "pose-refit-final.npz").resolve()
    np.savez(
        checkpoint,
        schema=np.asarray(pose_product.POST_G105_REFIT_CHECKPOINT_SCHEMA),
        run_id=np.asarray(run_id),
        seed=np.asarray(seed),
        source_contract=np.asarray(pose_product.SOURCE_DOMAIN),
        render_order=np.asarray(pose_product.RENDER_ORDER),
        y1_selected_preimage_schema=np.asarray(
            pose_product.V10_FACTOR2_SELECTED_PREIMAGE_SCHEMA
        ),
        source_g112_partition_receipt_sha256=np.asarray(
            partition_receipt_sha
        ),
        source_g112_semantic_child_sha256=np.asarray(semantic_child_sha),
        source_g112_pose_initializer_sha256=np.asarray(
            pose_initializer_artifact_sha
        ),
        source_g111_deploy_checkpoint_sha256=np.asarray(deploy_sha),
        source_g111_resume_checkpoint_sha256=np.asarray(resume_sha),
        source_g111_lineage_receipt_sha256=np.asarray(
            lineage_receipt_sha
        ),
        source_g111_checkpoint_id_sha256=np.asarray(checkpoint_id_sha),
        source_g111_root_sha256=np.asarray(root_sha),
        semantic_packet_sha256=np.asarray(semantic_sha),
        final_y1_binding_sha256=np.asarray(final_y1_binding),
        xi_initializer_sha256=np.asarray(initializer_sha),
        target_projection_sha256=np.asarray(target_projection_sha),
        target_capsule_receipt_sha256=np.asarray(target_sha),
        pose_targets_sha256=np.asarray(pose_sha),
        exact_public_receiver_in_loop=np.asarray(1, dtype=np.int8),
        pitch=np.asarray(0.0, dtype=np.float64),
        q_levels=np.asarray(4096, dtype=np.int64),
        selected_xip2_coder=np.asarray(selected_xip2_coder),
        xi_eff=np.ascontiguousarray(xi_eff, dtype=np.float64),
    )
    checkpoint_binding = {
        "path": str(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    run_body = {
        "schema": pose_product.POST_G105_REFIT_RUN_SCHEMA,
        "run_id": run_id,
        "seed": seed,
        "source_git_sha": "1" * 40,
        "command": [
            "g110-post-g105-refit",
            "--resume-from",
            "pose-stage",
        ],
        "fresh_own_lineage": True,
        "source_contract": pose_product.SOURCE_DOMAIN,
        "render_order": pose_product.RENDER_ORDER,
        "y1_selected_preimage_schema": (
            pose_product.V10_FACTOR2_SELECTED_PREIMAGE_SCHEMA
        ),
        "source_g112_partition_receipt_sha256": partition_receipt_sha,
        "source_g112_semantic_child_sha256": semantic_child_sha,
        "source_g112_pose_initializer_sha256": (
            pose_initializer_artifact_sha
        ),
        "source_g111_deploy_checkpoint_sha256": deploy_sha,
        "source_g111_resume_checkpoint_sha256": resume_sha,
        "source_g111_lineage_receipt_sha256": lineage_receipt_sha,
        "source_g111_checkpoint_id_sha256": checkpoint_id_sha,
        "source_g111_root_sha256": root_sha,
        "semantic_packet_sha256": semantic_sha,
        "final_y1_binding_sha256": final_y1_binding,
        "xi_initializer_sha256": initializer_sha,
        "target_projection_sha256": target_projection_sha,
        "target_capsule_receipt_sha256": target_sha,
        "pose_targets_sha256": pose_sha,
        "selected_xip2_coder": selected_xip2_coder,
        "g110_selected_xip2_coder_abi_closed": True,
        "exact_public_receiver_in_loop": True,
        "resumable_from_disk": True,
        "stage_checkpoints_preserved": True,
        "stage_checkpoints": [checkpoint_binding],
        "final_checkpoint": checkpoint_binding,
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
    }
    run = {
        **run_body,
        "receipt_sha256": hashlib.sha256(
            json.dumps(
                run_body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
    }
    run_path = (tmp_path / "pose-refit-run.json").resolve()
    run_path.write_text(
        json.dumps(run, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    base = object.__new__(pose_product.G110G112CompileCustodyV1)
    object.__setattr__(
        base,
        "partition_receipt_sha256",
        partition_receipt_sha,
    )
    object.__setattr__(base, "semantic_child_sha256", semantic_child_sha)
    object.__setattr__(
        base,
        "pose_initializer_sha256",
        pose_initializer_artifact_sha,
    )
    object.__setattr__(base, "source_deploy_checkpoint_sha256", deploy_sha)
    object.__setattr__(base, "source_resume_checkpoint_sha256", resume_sha)
    object.__setattr__(
        base,
        "source_lineage_receipt_sha256",
        lineage_receipt_sha,
    )
    object.__setattr__(
        base,
        "source_checkpoint_id_sha256",
        checkpoint_id_sha,
    )
    object.__setattr__(base, "source_root_sha256", root_sha)
    object.__setattr__(base, "target_projection_sha256", target_projection_sha)
    object.__setattr__(base, "target_capsule_receipt_sha256", target_sha)
    object.__setattr__(base, "pose_targets_sha256", pose_sha)
    initializer = object.__new__(pose_product.G112PoseInitializerV1)
    object.__setattr__(initializer, "xi_init", _xi())
    object.__setattr__(initializer, "pitch", 0.0)

    reopened = pose_product._verify_post_g105_refit(
        checkpoint=checkpoint,
        expected_checkpoint_sha256=checkpoint_binding["sha256"],
        run_receipt=run_path,
        expected_run_receipt_sha256=hashlib.sha256(
            run_path.read_bytes()
        ).hexdigest(),
        base_custody=base,
        initializer=initializer,
        semantic_packet=semantic,
        final_y1_binding=final_y1_binding,
    )
    assert np.array_equal(reopened.xi_eff, xi_eff)
    assert reopened.selected_xip2_coder == selected_xip2_coder
    assert reopened.checkpoint_sha256 == checkpoint_binding["sha256"]
    assert reopened.run_receipt_sha256 == hashlib.sha256(
        run_path.read_bytes()
    ).hexdigest()
    with pytest.raises(
        pose_product.G110GeneratedY1PoseError,
        match="externally expected SHA-256",
    ):
        pose_product._verify_post_g105_refit(
            checkpoint=checkpoint,
            expected_checkpoint_sha256="0" * 64,
            run_receipt=run_path,
            expected_run_receipt_sha256=hashlib.sha256(
                run_path.read_bytes()
            ).hexdigest(),
            base_custody=base,
            initializer=initializer,
            semantic_packet=semantic,
            final_y1_binding=final_y1_binding,
        )


@pytest.mark.parametrize(
    ("selected_xip2_coder", "expected_coder_id"),
    [("none", 0), ("delta_ar_zlib", 4)],
)
def test_compile_propagates_selected_xip2_coder_into_public_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    selected_xip2_coder: str,
    expected_coder_id: int,
) -> None:
    semantic_packet = _g105_packet()
    semantic_sha = hashlib.sha256(semantic_packet).hexdigest()
    xi_eff = _xi()
    initializer = SimpleNamespace(
        semantic_packet_sha256=semantic_sha,
        xi_init=xi_eff,
        pitch=0.0,
    )
    custody = SimpleNamespace(
        semantic_packet_sha256=semantic_sha,
        semantic_child=SimpleNamespace(semantic_packet=semantic_packet),
        pose_initializer=initializer,
        partition_receipt_sha256="1" * 64,
        semantic_child_sha256="2" * 64,
        pose_initializer_sha256="3" * 64,
        source_checkpoint_id_sha256="4" * 64,
        source_root_sha256="5" * 64,
    )
    population_digest = hashlib.sha256(b"g110 compiler seam").digest()
    initial_binding = pose_product._final_y1_binding_from_population(
        semantic_packet,
        population_digest,
    )

    monkeypatch.setattr(
        pose_product,
        "G110G112CompileCustodyV1",
        SimpleNamespace(
            from_physical_partition_receipt=lambda **_kwargs: custody
        ),
    )
    monkeypatch.setattr(
        pose_product,
        "_population_digest",
        lambda _provider: population_digest,
    )
    monkeypatch.setattr(
        pose_product,
        "final_y1_binding_sha256",
        lambda _provider: initial_binding,
    )

    observed_custody: dict[str, object] = {}

    def _open_refit(**kwargs: object) -> pose_product.PostG105PoseRefitV1:
        observed_custody.update(kwargs)
        return pose_product.PostG105PoseRefitV1(
            xi_eff=xi_eff,
            pitch=0.0,
            q_levels=4096,
            selected_xip2_coder=selected_xip2_coder,
            checkpoint_sha256="6" * 64,
            run_receipt_sha256="7" * 64,
            xi_eff_sha256=pose_product._xi_digest(xi_eff),
        )

    monkeypatch.setattr(pose_product, "_verify_post_g105_refit", _open_refit)
    result = pose_product.compile_g110_generated_y1_pose_v1(
        target_capsule_receipt=tmp_path / "g109.json",
        expected_target_capsule_receipt_sha256="8" * 64,
        g112_partition_receipt=tmp_path / "g112.json",
        expected_g112_partition_receipt_sha256="9" * 64,
        post_g105_refit_checkpoint=tmp_path / "g119.npz",
        expected_post_g105_refit_checkpoint_sha256="a" * 64,
        post_g105_refit_run_receipt=tmp_path / "g119.json",
        expected_post_g105_refit_run_receipt_sha256="b" * 64,
    )

    assert observed_custody["expected_checkpoint_sha256"] == "a" * 64
    assert observed_custody["expected_run_receipt_sha256"] == "b" * 64
    assert result.selected_xip2_coder == selected_xip2_coder
    reopened_packet = pose_product.parse_g110_generated_y1_pose_archive(
        result.archive
    )
    assert reopened_packet == result.packet
    parsed = pose_product.parse_g110_generated_y1_pose_v1(reopened_packet)
    assert parsed.xip2_payload[4] == expected_coder_id
    expected_q, expected_scales = quantize_xi(xi_eff, q_levels=4096)
    assert np.array_equal(parsed.q, expected_q)
    assert np.array_equal(parsed.scales, expected_scales)
    public = _module(
        RUNTIME_ROOT / "frame0_variants" / "generated_y1_pose_xip2_v1.py",
        f"_g110_public_selected_xip2_{selected_xip2_coder}",
    )
    assert public.accepts_packet(reopened_packet) is True
    public.parse_packet(reopened_packet)
