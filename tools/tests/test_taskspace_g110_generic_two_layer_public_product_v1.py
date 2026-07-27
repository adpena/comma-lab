# SPDX-License-Identifier: MIT
"""G110 generic-provider, conditional stream, and public-runtime checks."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import tac.witness_dsl.taskspace_g110_generic_two_layer_public_product_v1 as product
from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    verify_factor2_uint8_scorer_plane,
)
from tac.witness_dsl.taskspace_g95_population_pose_preimage_chart_v1 import (
    bilinear_resize_align_corners_false_numpy,
)
from tac.witness_dsl.taskspace_pfree_semantic_root_v1 import (
    encode_semantic_root_y1_v1,
    render_semantic_root_y1_scorer,
)
from tac.witness_dsl.v10_factor2_selected_preimage_v1 import (
    realize_factor2_uint8_numpy,
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


def _g108_fixture_module() -> ModuleType:
    return _module(
        REPO_ROOT / "tools" / "tests" / "test_taskspace_g108_semantic_root_public_product_v1.py",
        "_g110_g108_fixture",
    )


def _g105_fixture_module() -> ModuleType:
    return _module(
        REPO_ROOT
        / "src"
        / "tac"
        / "witness_dsl"
        / "tests"
        / "test_taskspace_g105_exact_v9_semantic_root_adapter_v1.py",
        "_g110_g105_fixture",
    )


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _conditional() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = np.zeros((2, 2, 3, 3), dtype=np.int8)
    basis[0, 0, 0, 0] = 2
    basis[1, 1, 2, 1] = 3
    scales = np.array([0.75, 0.25], dtype=np.float32)
    coefficients = np.zeros((600, 2), dtype=np.int16)
    coefficients[1:300, 0] = 2
    coefficients[300:, 0] = 3
    coefficients[450:, 1] = -1
    return basis, scales, coefficients


@pytest.fixture(scope="module")
def g103_packet() -> bytes:
    """Build a receiver packet directly; G103 producer custody stays refused."""

    root = _g108_fixture_module()._invariant_full_n600_root()
    semantic_packet = encode_semantic_root_y1_v1(root)
    provider = product.open_final_y1_provider(semantic_packet)
    basis, scales, coefficients = _conditional()
    return product._encode_packet(
        semantic_packet=semantic_packet,
        final_y1_binding=product.final_y1_binding_sha256(provider),
        basis=basis,
        scales=scales,
        coefficients=coefficients,
    )


def test_g103_provider_regression_and_zero_residual_row_is_legal(
    g103_packet: bytes,
) -> None:
    parsed = product.parse_g110_two_layer_v1(g103_packet)
    provider = product.open_final_y1_provider(parsed.semantic_packet)
    assert provider.variant_id == product.G103_VARIANT_ID
    assert int(np.count_nonzero(np.all(parsed.coefficients_q == 0, axis=1))) == 1
    assert parsed.coefficients_q[0].tolist() == [0, 0]

    root = _g108_fixture_module()._invariant_full_n600_root()
    for pair_id in (0, 1, 137, 599):
        expected_y1 = render_semantic_root_y1_scorer(root, pair_id)
        observed = provider.render_scorer_y1(pair_id)
        assert np.array_equal(observed, expected_y1)
    pair0 = parsed.render_scorer_pair(provider, 0)
    assert np.array_equal(pair0[0], pair0[1])


def test_one_combined_scale_and_temporal_rice_are_canonical(
    g103_packet: bytes,
) -> None:
    parsed = product.parse_g110_two_layer_v1(g103_packet)
    assert parsed.combined_scales.shape == (2,)
    assert np.array_equal(parsed.combined_scales, np.array([0.75, 0.25], dtype=np.float32))
    assert np.array_equal(parsed.coefficients_q, _conditional()[2])
    assert 0 <= parsed.rice_k <= 15
    assert b"basis_scales" not in g103_packet
    assert b"coefficient_scales" not in g103_packet

    noncanonical = bytearray(g103_packet)
    noncanonical[-1] ^= 1
    with pytest.raises(product.G110TwoLayerError, match="CRC32"):
        product.parse_g110_two_layer_v1(bytes(noncanonical))


def test_archive_is_one_receiver_counted_member(
    g103_packet: bytes,
) -> None:
    archive_bytes = product.build_g110_public_archive(g103_packet)
    assert product.parse_g110_public_archive(archive_bytes) == g103_packet
    assert archive_bytes == product.build_g110_public_archive(g103_packet)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        infos = archive.infolist()
        assert [info.filename for info in infos] == [product.PACKET_MEMBER]
        assert infos[0].compress_type == zipfile.ZIP_DEFLATED
        assert archive.read(infos[0]) == g103_packet


def test_rank_zero_semantic_floor_and_typed_archive_matrix(
    g103_packet: bytes,
) -> None:
    semantic_packet = product.parse_g110_two_layer_v1(
        g103_packet
    ).semantic_packet
    floor_packet = product.build_g110_rank_zero_semantic_floor_packet(
        semantic_packet
    )
    parsed = product.parse_g110_two_layer_v1(floor_packet)
    assert parsed.basis_q.shape == (0, 0, 0, 3)
    assert parsed.combined_scales.shape == (0,)
    assert parsed.coefficients_q.shape == (600, 0)
    pair = product.render_g110_rank_zero_scorer_pair(floor_packet, 137)
    assert pair.shape == (2, 384, 512, 3)
    assert np.array_equal(pair[0], pair[1])

    alternatives = []
    for method in product.G110OuterZipMethodV1:
        archive_bytes = product.build_g110_counted_archive_variant(
            floor_packet,
            method,
        )
        assert (
            product.parse_g110_counted_archive_variant(
                archive_bytes,
                method,
            )
            == floor_packet
        )
        alternatives.append((method, archive_bytes))
    expected_method, expected_archive = min(
        alternatives,
        key=lambda item: (
            len(item[1]),
            int(item[0]),
            hashlib.sha256(item[1]).hexdigest(),
        ),
    )
    assert product.build_g110_public_archive(floor_packet) == expected_archive
    assert product.parse_g110_public_archive(expected_archive) == floor_packet
    for method, archive_bytes in alternatives:
        if method is expected_method:
            continue
        with pytest.raises(
            product.G110TwoLayerError,
            match="canonical method/layout",
        ):
            product.parse_g110_public_archive(archive_bytes)


def test_compile_custody_cannot_be_hash_asserted() -> None:
    with pytest.raises(TypeError, match="content-reading"):
        product.G110Batch16SourcePoseCustodyV1(
            target_margins_sha256=_sha("m"),
            pose_targets_sha256=_sha("p"),
            target_capsule_receipt_sha256=_sha("r"),
            fresh_checkpoint_sha256=_sha("c"),
        )
    forged = object.__new__(product.G110Batch16SourcePoseCustodyV1)
    basis, scales, coefficients = _conditional()
    with pytest.raises(TypeError, match="unexpected keyword argument 'custody'"):
        product.compile_g110_two_layer_v1(
            b"not-a-packet",
            basis_q=basis,
            combined_scales=scales,
            coefficients_q=coefficients,
            custody=forged,  # type: ignore[call-arg]
            target_capsule_receipt=Path("absent-g109.json"),
            expected_target_capsule_receipt_sha256=_sha("g109"),
            fresh_g105_checkpoint=Path("absent-g105.npz"),
            conditional_operand_receipt=Path("absent-conditional.json"),
            expected_conditional_operand_receipt_sha256=_sha("conditional"),
        )


def test_conditional_quotient_gauges_and_dead_or_overflow_ranks_refuse() -> None:
    basis, scales, coefficients = _conditional()

    negative_basis = basis.copy()
    negative_basis[0] *= -1
    with pytest.raises(product.G110TwoLayerError, match="sign gauge"):
        product._validate_conditional(negative_basis, scales, coefficients)

    nonprimitive_coefficients = coefficients.copy()
    nonprimitive_coefficients[:, 0] *= 2
    with pytest.raises(product.G110TwoLayerError, match="coefficient/scale gauge"):
        product._validate_conditional(basis, scales, nonprimitive_coefficients)

    decoder_dead_scales = scales.copy()
    decoder_dead_scales[0] = np.nextafter(
        np.float32(0.0),
        np.float32(1.0),
    )
    with pytest.raises(product.G110TwoLayerError, match="decoder-dead"):
        product._validate_conditional(basis, decoder_dead_scales, coefficients)

    overflowing_scales = scales.copy()
    overflowing_scales[0] = np.finfo(np.float32).max
    with pytest.raises(product.G110TwoLayerError, match="overflow"):
        product._validate_conditional(basis, overflowing_scales, coefficients)


def test_conditional_operands_reopen_physical_checkpoint_and_run(
    tmp_path: Path,
) -> None:
    basis, scales, coefficients = _conditional()
    semantic_sha = _sha("semantic")
    target_sha = _sha("target-capsule")
    pose_sha = _sha("poses")
    run_id = "g110-test-physical-producer"
    seed = 110
    checkpoint_path = (tmp_path / "conditional-final.npz").resolve()
    np.savez(
        checkpoint_path,
        schema=np.asarray(product.CONDITIONAL_PRODUCER_CHECKPOINT_SCHEMA),
        run_id=np.asarray(run_id),
        seed=np.asarray(seed),
        fresh_own_lineage=np.asarray(1),
        joint_pose_conditioned=np.asarray(1),
        semantic_packet_sha256=np.asarray(semantic_sha),
        target_capsule_receipt_sha256=np.asarray(target_sha),
        pose_targets_sha256=np.asarray(pose_sha),
        basis_q=basis,
        combined_scales=scales,
        coefficients_q=coefficients,
    )
    checkpoint_binding = {
        "path": str(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    run_path = (tmp_path / "producer-run.json").resolve()
    run_body = {
        "schema": product.CONDITIONAL_PRODUCER_RUN_SCHEMA,
        "run_id": run_id,
        "seed": seed,
        "source_git_sha": "1" * 40,
        "command": ["g110-real-producer", "--resume-from", "stage"],
        "fresh_own_lineage": True,
        "joint_pose_conditioned": True,
        "resumable_from_disk": True,
        "stage_checkpoints_preserved": True,
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
        "semantic_packet_sha256": semantic_sha,
        "target_capsule_receipt_sha256": target_sha,
        "pose_targets_sha256": pose_sha,
        "producer_checkpoint_sha256": checkpoint_binding["sha256"],
        "producer_checkpoint_bytes": checkpoint_binding["bytes"],
        "stage_checkpoints": [checkpoint_binding],
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
    run_path.write_text(
        json.dumps(run, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    run_binding = {
        "path": str(run_path),
        "bytes": run_path.stat().st_size,
        "sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
    }
    custody = object.__new__(product.G110Batch16SourcePoseCustodyV1)
    object.__setattr__(custody, "target_capsule_receipt_sha256", target_sha)
    object.__setattr__(custody, "pose_targets_sha256", pose_sha)
    product._verify_conditional_producer(
        checkpoint_binding=checkpoint_binding,
        run_binding=run_binding,
        custody=custody,
        semantic_packet=b"semantic",
        basis=basis,
        scales=scales,
        coefficients=coefficients,
    )


def test_custody_file_resolver_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "physical.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "alias.json"
    link.symlink_to(target)
    assert product._resolve_regular_nonsymlink(target, name="fixture") == target
    with pytest.raises(product.G110TwoLayerError, match="must not be a symlink"):
        product._resolve_regular_nonsymlink(link, name="fixture")


def test_archive_parse_rejects_oversized_uncompressed_member() -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.writestr(
            product.PACKET_MEMBER,
            b"\x00" * (product.MAX_PACKET_BYTES + 1),
        )
    oversized_archive = stream.getvalue()
    assert len(oversized_archive) < product.MAX_ARCHIVE_BYTES
    with pytest.raises(product.G110TwoLayerError, match="unsafe/noncanonical"):
        product.parse_g110_public_archive(oversized_archive)


def test_public_g103_plugin_preserves_committed_provider(
    g103_packet: bytes,
) -> None:
    parsed = product.parse_g110_two_layer_v1(g103_packet)
    repo_provider = product.open_final_y1_provider(parsed.semantic_packet)
    public = _module(
        RUNTIME_ROOT / "semantic_variants" / "original_coordinr_film_mlp_v1.py",
        "_g110_public_g103",
    )
    assert public.accepts_packet(parsed.semantic_packet) is True
    public_root = public.parse_packet(parsed.semantic_packet)
    for pair_id in (0, 137, 599):
        assert np.array_equal(
            public.render_scorer_y1(public_root, pair_id),
            repo_provider.render_scorer_y1(pair_id),
        )


def test_public_v9_plugin_is_exact_g105_receiver() -> None:
    fixture = _g105_fixture_module()
    subject = fixture.subject
    _config, _params, _code, program = fixture._fixture()
    packet = subject.encode_packet(program)
    repo = product.open_final_y1_provider(packet)
    public = _module(
        RUNTIME_ROOT / "semantic_variants" / "v9_hosc_dual_head_odd_y1_v1.py",
        "_g110_public_v9",
    )
    assert repo.variant_id == product.V9_VARIANT_ID
    assert public.accepts_packet(packet) is True
    parsed = public.parse_packet(packet)
    for pair_id in (0, 17, 599):
        assert np.array_equal(
            public.render_scorer_y1(parsed, pair_id),
            repo.render_scorer_y1(pair_id),
        )


def test_public_conditional_plugin_matches_repo_and_binding(
    g103_packet: bytes,
) -> None:
    parsed = product.parse_g110_two_layer_v1(g103_packet)
    provider = product.open_final_y1_provider(parsed.semantic_packet)
    conditional = _module(
        RUNTIME_ROOT / "frame0_variants" / "conditional_lowrank_rice_v1.py",
        "_g110_public_conditional",
    )
    state = conditional.parse_packet(g103_packet)
    assert conditional.semantic_packet(state) == parsed.semantic_packet
    for pair_id in (0, 1, 451, 599):
        y1 = provider.render_scorer_y1(pair_id)
        assert np.array_equal(
            conditional.render_scorer_y0(state, pair_id, y1),
            parsed.render_scorer_pair(provider, pair_id)[0],
        )

    population = hashlib.sha256()
    for pair_id in range(600):
        y1 = provider.render_scorer_y1(pair_id)
        population.update(pair_id.to_bytes(2, "big"))
        population.update(memoryview(y1).cast("B"))
    conditional.verify_final_y1_population(state, population.digest())
    wrong = bytearray(population.digest())
    wrong[0] ^= 1
    with pytest.raises(ValueError, match="binding differs"):
        conditional.verify_final_y1_population(state, bytes(wrong))


def test_bilinear_boundary_vector_matches_independent_canonical_kernel() -> None:
    grid = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    expected = bilinear_resize_align_corners_false_numpy(
        grid,
        output_height=384,
        output_width=512,
    )
    observed_repo = product._bilinear_resize(
        grid,
        output_height=384,
        output_width=512,
    )
    conditional = _module(
        RUNTIME_ROOT / "frame0_variants" / "conditional_lowrank_rice_v1.py",
        "_g110_public_conditional_boundary",
    )
    observed_public = conditional._bilinear_resize(grid)
    assert np.array_equal(observed_repo, expected)
    assert np.array_equal(observed_public, expected)
    assert np.array_equal(expected[0, 0], grid[0, 0])
    assert np.array_equal(expected[-1, -1], grid[-1, -1])


def test_public_v10_factor2_is_exact_for_both_planes(g103_packet: bytes) -> None:
    parsed = product.parse_g110_two_layer_v1(g103_packet)
    provider = product.open_final_y1_provider(parsed.semantic_packet)
    scorer_pair = parsed.render_scorer_pair(provider, 451)
    inflate = _module(RUNTIME_ROOT / "inflate.py", "_g110_factor2")
    operator = DisjointResizeOperator.build(
        camera_h=inflate.CAMERA_H,
        camera_w=inflate.CAMERA_W,
        scorer_h=inflate.SCORER_H,
        scorer_w=inflate.SCORER_W,
    )
    for scorer_plane in scorer_pair:
        camera = inflate._realize_factor2(scorer_plane)
        canonical = realize_factor2_uint8_numpy(scorer_plane)
        assert np.array_equal(camera, canonical)
        proof = verify_factor2_uint8_scorer_plane(operator, camera, scorer_plane)
        assert proof.certified_exact


def test_public_dispatch_is_unambiguous_for_both_semantic_variants(
    g103_packet: bytes,
) -> None:
    inflate = _module(RUNTIME_ROOT / "inflate.py", "_g110_inflate")
    semantic_plugins = inflate._load_plugins(
        RUNTIME_ROOT / "semantic_variants",
        calls=inflate.SEMANTIC_PLUGIN_CALLS,
        expected=inflate.EXPECTED_SEMANTIC_PLUGINS,
    )
    semantic_g103 = product.parse_g110_two_layer_v1(g103_packet).semantic_packet
    assert sum(module.accepts_packet(semantic_g103) is True for module in semantic_plugins.values()) == 1

    fixture = _g105_fixture_module()
    v9_packet = fixture.subject.encode_packet(fixture._fixture()[3])
    assert sum(module.accepts_packet(v9_packet) is True for module in semantic_plugins.values()) == 1
    forbidden = ("from tac", "import tac", "SegNet", "PoseNet", "upstream.evaluate")
    runtime_sources = [
        *sorted(RUNTIME_ROOT.rglob("*.py")),
        RUNTIME_ROOT / "inflate.sh",
    ]
    assert not any(
        token in path.read_text("utf-8")
        for path in runtime_sources
        for token in forbidden
    )


def test_public_plugin_loading_is_repeatable_without_runtime_mutation() -> None:
    inflate = _module(RUNTIME_ROOT / "inflate.py", "_g110_inflate_repeatable")
    before = sorted(
        path.relative_to(RUNTIME_ROOT).as_posix()
        for path in RUNTIME_ROOT.rglob("*")
    )
    for _pass in range(2):
        semantic_plugins = inflate._load_plugins(
            RUNTIME_ROOT / "semantic_variants",
            calls=inflate.SEMANTIC_PLUGIN_CALLS,
            expected=inflate.EXPECTED_SEMANTIC_PLUGINS,
        )
        frame0_plugins = inflate._load_plugins(
            RUNTIME_ROOT / "frame0_variants",
            calls=inflate.FRAME0_PLUGIN_CALLS,
            expected=inflate.EXPECTED_FRAME0_PLUGINS,
        )
        assert set(semantic_plugins) == set(
            inflate.EXPECTED_SEMANTIC_PLUGINS.values()
        )
        assert set(frame0_plugins) == set(
            inflate.EXPECTED_FRAME0_PLUGINS.values()
        )
    after = sorted(
        path.relative_to(RUNTIME_ROOT).as_posix()
        for path in RUNTIME_ROOT.rglob("*")
    )
    assert after == before
