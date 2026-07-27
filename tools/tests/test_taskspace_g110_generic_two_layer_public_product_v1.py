# SPDX-License-Identifier: MIT
"""G110 generic-provider, conditional stream, and public-runtime checks."""

from __future__ import annotations

import hashlib
import importlib.util
import io
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
from tac.witness_dsl.taskspace_pfree_semantic_root_v1 import (
    encode_semantic_root_y1_v1,
    render_semantic_root_y1_scorer,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = REPO_ROOT / product.PUBLIC_RUNTIME_RELATIVE_ROOT


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
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
    basis[1, 1, 2, 1] = -3
    scales = np.array([0.75, 0.25], dtype=np.float32)
    coefficients = np.zeros((600, 2), dtype=np.int16)
    coefficients[1:300, 0] = 2
    coefficients[300:, 0] = 3
    coefficients[450:, 1] = -4
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


def test_compile_custody_cannot_be_hash_asserted() -> None:
    with pytest.raises(TypeError, match="content-reading"):
        product.G110Batch16SourcePoseCustodyV1(
            target_margins_sha256=_sha("m"),
            pose_targets_sha256=_sha("p"),
            target_capsule_receipt_sha256=_sha("r"),
            fresh_checkpoint_sha256=_sha("c"),
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
        proof = verify_factor2_uint8_scorer_plane(operator, camera, scorer_plane)
        assert proof.certified_exact


def test_public_dispatch_is_unambiguous_for_both_semantic_variants(
    g103_packet: bytes,
) -> None:
    inflate = _module(RUNTIME_ROOT / "inflate.py", "_g110_inflate")
    semantic_plugins = inflate._load_plugins(
        RUNTIME_ROOT / "semantic_variants",
        calls=inflate.SEMANTIC_PLUGIN_CALLS,
    )
    semantic_g103 = product.parse_g110_two_layer_v1(g103_packet).semantic_packet
    assert sum(module.accepts_packet(semantic_g103) is True for module in semantic_plugins.values()) == 1

    fixture = _g105_fixture_module()
    v9_packet = fixture.subject.encode_packet(fixture._fixture()[3])
    assert sum(module.accepts_packet(v9_packet) is True for module in semantic_plugins.values()) == 1
    assert not any(
        token in (RUNTIME_ROOT / "inflate.py").read_text("utf-8")
        for token in ("from tac", "import tac", "SegNet", "PoseNet", "upstream.evaluate")
    )
