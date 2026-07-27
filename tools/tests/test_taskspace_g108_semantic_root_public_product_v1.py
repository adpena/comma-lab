# SPDX-License-Identifier: MIT
"""G108 public product, standalone receiver, and clean-root closure proofs."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import experiments.run_g102_semantic_root_s00_s01_n600_v1 as g102
import tac.witness_dsl.taskspace_g108_semantic_root_public_product_v1 as product
from tac.witness_dsl.taskspace_pfree_semantic_root_v1 import (
    GeneratorActivationV1,
    GeneratorArchitectureV1,
    GeneratorNumericContractV1,
    QuantizedGeneratorTensorV1,
    QuantizedSharedGeneratorV1,
    QuantizedTensorDTypeV1,
    QuantizedTensorRoleV1,
    RGBGaugeOwnershipV1,
    SemanticRealizationProfileV1,
    SemanticRoleV1,
    SemanticRootY1V1,
    TemporalLatentStreamV1,
    encode_semantic_root_y1_v1,
    realize_semantic_root_y1_v10_factor2,
    render_semantic_root_y1_scorer,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = REPO_ROOT / product.PUBLIC_RUNTIME_RELATIVE_ROOT
SEMANTIC_PLUGIN = RUNTIME_ROOT / "semantic_variants" / "original_coordinr_film_mlp_v1.py"
INFLATE_MODULE = RUNTIME_ROOT / "inflate.py"
EXPECTED_RAW_BYTES = 1200 * 874 * 1164 * 3


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_root() -> SemanticRootY1V1:
    fixture = _module(
        REPO_ROOT / "src" / "tac" / "witness_dsl" / "tests" / "test_taskspace_pfree_semantic_root_v1.py",
        "_g108_g103_fixture",
    )
    return fixture._root()  # type: ignore[no-any-return,attr-defined]


def _tensor(
    tensor_id: int,
    role: QuantizedTensorRoleV1,
    values: np.ndarray,
    dtype: QuantizedTensorDTypeV1,
) -> QuantizedGeneratorTensorV1:
    array = np.asarray(values)
    data = array.astype(np.int8).tobytes() if dtype is QuantizedTensorDTypeV1.INT8 else array.astype(">i2").tobytes()
    return QuantizedGeneratorTensorV1(
        tensor_id=tensor_id,
        role=role,
        dtype=dtype,
        shape=tuple(int(value) for value in array.shape),
        scale_exponent=-7 if dtype is QuantizedTensorDTypeV1.INT8 else -12,
        zero_point=0,
        data=data,
    )


def _invariant_full_n600_root() -> SemanticRootY1V1:
    """A real wire-valid n600 program whose rendered population is invariant."""

    hidden = 8
    input_weight = np.zeros((hidden, 4), dtype=np.int8)
    input_weight[0, 0] = 1
    tensors = (
        _tensor(
            0,
            QuantizedTensorRoleV1.INPUT_WEIGHT,
            input_weight,
            QuantizedTensorDTypeV1.INT8,
        ),
        _tensor(
            1,
            QuantizedTensorRoleV1.INPUT_BIAS,
            np.zeros(hidden, dtype=np.int16),
            QuantizedTensorDTypeV1.INT16_BE,
        ),
        _tensor(
            2,
            QuantizedTensorRoleV1.HIDDEN_WEIGHT,
            np.zeros((hidden, hidden), dtype=np.int8),
            QuantizedTensorDTypeV1.INT8,
        ),
        _tensor(
            3,
            QuantizedTensorRoleV1.HIDDEN_BIAS,
            np.zeros(hidden, dtype=np.int16),
            QuantizedTensorDTypeV1.INT16_BE,
        ),
        _tensor(
            4,
            QuantizedTensorRoleV1.FILM_WEIGHT,
            np.zeros((2 * hidden, 1), dtype=np.int8),
            QuantizedTensorDTypeV1.INT8,
        ),
        _tensor(
            5,
            QuantizedTensorRoleV1.FILM_BIAS,
            np.zeros(2 * hidden, dtype=np.int16),
            QuantizedTensorDTypeV1.INT16_BE,
        ),
        _tensor(
            6,
            QuantizedTensorRoleV1.OUTPUT_WEIGHT,
            np.zeros((3, hidden), dtype=np.int8),
            QuantizedTensorDTypeV1.INT8,
        ),
        _tensor(
            7,
            QuantizedTensorRoleV1.OUTPUT_BIAS,
            np.zeros(3, dtype=np.int16),
            QuantizedTensorDTypeV1.INT16_BE,
        ),
    )
    model = QuantizedSharedGeneratorV1(
        architecture=GeneratorArchitectureV1.ORIGINAL_COORDINR_FILM_MLP_V1,
        numeric_contract=GeneratorNumericContractV1.INT8_WEIGHT_INT16_STATE_INT32_ACCUM_Q12,
        activation=GeneratorActivationV1.HARD_TANH_Q12,
        input_dim=4,
        hidden_dim=hidden,
        hidden_layer_count=1,
        modulation_dim=1,
        tensors=tensors,
    )
    return SemanticRootY1V1(
        background_role=SemanticRoleV1.ROAD,
        profile=SemanticRealizationProfileV1(
            role_rgb=(
                (96, 92, 88),
                (180, 168, 72),
                (48, 56, 68),
                (140, 96, 72),
                (72, 92, 128),
            ),
            texture_gain_q4=16,
            edge_gain_q4=16,
            chroma_gain_q4=16,
            parallax_gain_q4=16,
            renderer_seed=0x108,
        ),
        shared_generator=model,
        temporal_latents=TemporalLatentStreamV1.from_array(
            np.zeros((600, 1), dtype=np.int16),
            rice_k=0,
        ),
        rgb_gauge_ownership=RGBGaugeOwnershipV1.DERIVED_BY_SHARED_GENERATOR,
        topology_templates=(),
        topology_events=(),
        rgb_basis=(),
        pair_rgb_gauges=(),
        irreducible_rgb_quotient=(),
    )


def _runtime_sha256() -> str:
    records = g102._public_runtime_records(
        REPO_ROOT,
        product.PUBLIC_RUNTIME_RELATIVE_ROOT,
    )
    return g102._public_runtime_sha256(records)


def test_g102_interface_is_complete_but_source_authority_remains_closed() -> None:
    g102._require_module_interface(product)
    capability = product.semantic_root_y1_v1_capability()
    assert frozenset(capability) == g102.REQUIRED_CAPABILITY_KEYS
    assert capability["public_codec_section_sha256"] == _runtime_sha256()
    assert capability["scorer_free_receiver"] is True
    assert capability["own_lineage"] is False
    assert capability["exact_post_r_seg_closure"] is False
    assert capability["exact_post_r_pose_closure"] is False
    with pytest.raises(g102.G102RunnerError, match=g102.RGB_CLOSURE_BLOCKER):
        g102._validate_capability(
            capability,
            expected_codec_sha=_runtime_sha256(),
        )
    for call in (
        product.compile_semantic_root_y1_v1_stage,
        product.semantic_root_y1_v1_source_lineage_manifest,
        product.semantic_root_y1_v1_g17_whole_object_state,
    ):
        with pytest.raises(
            product.G108SourceClosureOwed,
            match=product.SOURCE_COMPILER_BLOCKER,
        ):
            call()


def test_public_archive_is_one_counted_self_describing_packet() -> None:
    packet = encode_semantic_root_y1_v1(_fixture_root())
    archive = product.build_semantic_root_y1_v1_public_archive(packet)
    assert archive == product.build_semantic_root_y1_v1_public_archive(packet)
    assert product.parse_semantic_root_y1_v1_public_archive(archive) == packet
    with zipfile.ZipFile(io.BytesIO(archive)) as opened:
        infos = opened.infolist()
        assert [row.filename for row in infos] == [product.PACKET_MEMBER]
        assert infos[0].compress_type == zipfile.ZIP_DEFLATED
        assert infos[0].file_size == len(packet)
        assert opened.read(infos[0]) == packet
    assert len(archive) < len(packet)


def test_public_archive_refuses_extra_member_and_packet_mutation() -> None:
    packet = encode_semantic_root_y1_v1(_fixture_root())
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.writestr(product.PACKET_MEMBER, packet)
        archive.writestr("proof.json", b"{}")
    with pytest.raises(product.G108PublicProductError, match="member set"):
        product.parse_semantic_root_y1_v1_public_archive(stream.getvalue())

    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.writestr(product.PACKET_MEMBER, packet[:-1] + bytes([packet[-1] ^ 1]))
    with pytest.raises(ValueError, match="CRC32"):
        product.parse_semantic_root_y1_v1_public_archive(stream.getvalue())


def test_standalone_variant_and_factor2_match_committed_g103() -> None:
    root = _fixture_root()
    packet = encode_semantic_root_y1_v1(root)
    variant = _module(SEMANTIC_PLUGIN, "_g108_semantic_variant")
    inflate = _module(INFLATE_MODULE, "_g108_public_inflate")
    assert variant.accepts_packet(packet) is True
    parsed = variant.parse_packet(packet)
    for pair_id in (0, 1, 137, 599):
        expected_scorer = render_semantic_root_y1_scorer(root, pair_id)
        observed_scorer = variant.render_scorer_y1(parsed, pair_id)
        assert np.array_equal(observed_scorer, expected_scorer)
        expected_camera, proof = realize_semantic_root_y1_v10_factor2(
            root,
            pair_id,
        )
        assert proof.certified_exact
        assert np.array_equal(
            inflate._realize_factor2(observed_scorer),
            expected_camera,
        )


def test_runtime_dispatch_is_unambiguous_and_reserves_v9_plugin_seam(
    tmp_path: Path,
) -> None:
    packet = encode_semantic_root_y1_v1(_fixture_root())
    inflate = _module(INFLATE_MODULE, "_g108_dispatch")
    plugins = inflate._load_plugins(
        RUNTIME_ROOT / "semantic_variants",
        calls=inflate.SEMANTIC_PLUGIN_CALLS,
    )
    assert [key for key, module in plugins.items() if module.accepts_packet(packet)] == [product.SEMANTIC_VARIANT_ID]
    source = INFLATE_MODULE.read_text()
    assert "matches = [" in source
    assert "exactly one required" in source
    assert not any(
        token in source
        for token in (
            "from tac",
            "import tac",
            "upstream.evaluate",
            "SegNet",
            "PoseNet",
        )
    )
    extracted = tmp_path / "archive"
    extracted.mkdir()
    (extracted / product.PACKET_MEMBER).write_bytes(packet)
    assert inflate._load_packet(extracted) == packet


@pytest.mark.skipif(
    os.environ.get("PACT_G108_FULL_N600") != "1",
    reason="requires two exact 3.66GB clean-root public decodes",
)
@pytest.mark.timeout(1800)
def test_full_n600_clean_extract_double_decode() -> None:
    """Exercise actual ``inflate.sh`` twice and auto-clean its rebuildable raw."""

    requested_root = os.environ.get("PACT_G108_FULL_ROOT")
    if not requested_root:
        pytest.fail("PACT_G108_FULL_ROOT must select an operator-approved storage tier")
    base = Path(requested_root).resolve()
    if base.is_symlink() or not base.is_dir():
        pytest.fail("PACT_G108_FULL_ROOT is not a regular directory")
    work = Path(tempfile.mkdtemp(prefix="g108_public_closure.", dir=base))
    started = time.monotonic()
    try:
        packet = encode_semantic_root_y1_v1(_invariant_full_n600_root())
        archive_bytes = product.build_semantic_root_y1_v1_public_archive(packet)
        results = []
        for label in ("clean_a", "clean_b"):
            case = work / label
            archive_root = case / "archive"
            runtime_root = case / "runtime"
            output_root = case / "output"
            archive_root.mkdir(parents=True)
            output_root.mkdir()
            shutil.copytree(RUNTIME_ROOT, runtime_root)
            archive_path = case / "archive.zip"
            archive_path.write_bytes(archive_bytes)
            g102._safe_extract_archive(archive_path, archive_root)
            names = case / "public_test_video_names.txt"
            names.write_text("0.mkv\n")
            guard, _guard_sha = g102._write_python_import_guard(
                case_root=case,
                forbidden_repo_root=REPO_ROOT,
                allowed_runtime_root=runtime_root,
            )
            completed = subprocess.run(
                [
                    "bash",
                    str(runtime_root / "inflate.sh"),
                    str(archive_root),
                    str(output_root),
                    str(names),
                ],
                cwd=case,
                env={
                    "PATH": f"{guard}:{os.environ.get('PATH', '/usr/bin:/bin')}",
                    "PYTHON": str(guard / "python"),
                    "PYTHON_BIN": str(guard / "python"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONPATH": "",
                },
                capture_output=True,
                text=True,
                check=False,
                timeout=1800,
            )
            assert completed.returncode == 0, completed.stderr
            raw = output_root / "0.raw"
            assert [path.name for path in output_root.iterdir()] == ["0.raw"]
            assert raw.stat().st_size == EXPECTED_RAW_BYTES
            digest = hashlib.sha256()
            with raw.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 << 20), b""):
                    digest.update(chunk)
            results.append(
                {
                    "label": label,
                    "output_bytes": raw.stat().st_size,
                    "output_sha256": digest.hexdigest(),
                    "stdout": completed.stdout.strip(),
                }
            )
        assert results[0]["output_sha256"] == results[1]["output_sha256"]
        print(
            "G108_FULL_RECEIVER_PROOF="
            + json.dumps(
                {
                    "packet_bytes": len(packet),
                    "packet_sha256": hashlib.sha256(packet).hexdigest(),
                    "archive_bytes": len(archive_bytes),
                    "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
                    "runtime_tree_sha256": _runtime_sha256(),
                    "runs": results,
                    "double_decode_equal": True,
                    "elapsed_seconds": time.monotonic() - started,
                    "raw_cleanup": "success-path finally removal of exact temp root",
                    "source_closure_claim": False,
                    "score_claim": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    finally:
        shutil.rmtree(work)
