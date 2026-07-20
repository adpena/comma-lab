from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from tac.boundary_math.r1b4_section_receiver import ReplayWrite, encode_replay_payload
from tac.boundary_math.windowed_curvelet_frame import WindowedCurveletConfig
from tac.optimization.boundary_coordinate_joint_solve import (
    BoundaryCoordinatePacket,
    FrameFamily,
    encode_boundary_packet,
)
from tac.optimization.r1b2_mdl_xi0_compile import (
    RANK4_SCHEMA,
    R1B2CompileError,
    audit_full_kernel,
    audit_rank4_secants,
    audit_vjp_campaign,
    build_receipt,
    compile_candidate_archive,
    sha256_file,
)
from tac.optimization.r1b3_producer_preflight import encode_xi0_payload


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _incomplete_campaign(tmp_path: Path) -> Path:
    sidecar = tmp_path / "pair_0000.vjp.npz"
    sidecar.write_bytes(b"declared-sidecar")
    manifest = tmp_path / "source" / "manifest.json"
    _write_json(
        manifest,
        {
            "sidecars": [
                {
                    "pair_id": 0,
                    "path": str(sidecar),
                    "bytes": sidecar.stat().st_size,
                    "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                    "tensor_hashes": {"seg_g_y": "a" * 64},
                }
            ],
            "refusals": [],
        },
    )
    campaign = tmp_path / "campaign.json"
    _write_json(
        campaign,
        {
            "schema": "vjp_custody_n600_extension.v1",
            "updated_at_utc": "2026-07-20T00:00:00Z",
            "status": None,
            "final_completed_count": None,
            "source_manifests": [
                {
                    "path": str(manifest),
                    "sha256": sha256_file(manifest),
                    "completed_pair_ids": [0],
                    "refused_pair_ids": [],
                }
            ],
            "chunks": [],
        },
    )
    return campaign


def test_live_campaign_audit_decomposes_partial_without_rehashing_bulk(tmp_path: Path) -> None:
    result = audit_vjp_campaign(_incomplete_campaign(tmp_path))
    assert result["completed_pair_count"] == 1
    assert result["missing_pair_count"] == 599
    assert result["completed_pair_ids"] == [0]
    assert result["missing_pair_ids"][0] == 1
    assert result["sidecar_bytes_rehashed_by_r1b2"] is False
    assert result["per_pair_declared_custody"][0]["declared_sha256"] == hashlib.sha256(b"declared-sidecar").hexdigest()
    assert "VJP_CAMPAIGN_NOT_TERMINAL_COMPLETE_N600" in result["blockers"]
    assert "VJP_COMPLETED_PAIR_COUNT_1_NOT_600" in result["blockers"]


def test_campaign_audit_refuses_manifest_hash_drift(tmp_path: Path) -> None:
    campaign = _incomplete_campaign(tmp_path)
    value = json.loads(campaign.read_text())
    value["source_manifests"][0]["sha256"] = "0" * 64
    _write_json(campaign, value)
    with pytest.raises(R1B2CompileError, match="manifest SHA drift"):
        audit_vjp_campaign(campaign)


def test_campaign_audit_treats_later_completion_as_superseding_scoped_refusal(
    tmp_path: Path,
) -> None:
    campaign = _incomplete_campaign(tmp_path)
    campaign_value = json.loads(campaign.read_text())
    manifest = Path(campaign_value["source_manifests"][0]["path"])
    manifest_value = json.loads(manifest.read_text())
    manifest_value["refusals"] = [{"pair_id": 0}]
    _write_json(manifest, manifest_value)
    campaign_value["source_manifests"][0]["sha256"] = sha256_file(manifest)
    campaign_value["source_manifests"][0]["refused_pair_ids"] = [0]
    _write_json(campaign, campaign_value)

    result = audit_vjp_campaign(campaign)
    assert result["refused_pair_ids"] == []
    assert "VJP_REFUSED_PAIR_IDS_PRESENT" not in result["blockers"]


def _packet(path: Path) -> None:
    packet = BoundaryCoordinatePacket(
        family=FrameFamily.WINDOWED_CURVELET,
        frame_config=asdict(WindowedCurveletConfig(n_scales=1, n_orient0=2, n_trans=1)),
        scorer_height=384,
        scorer_width=512,
        atom_indices=np.asarray([0], dtype=np.uint32),
        coefficients=np.zeros((600, 1, 3), dtype=np.int8),
        scales=np.ones(600, dtype=np.float16),
    )
    path.write_bytes(encode_boundary_packet(packet))


def _base_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in (("0.bin", b"base" * 256), ("ipe_manifest.json", b"{}")):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED)


def test_typed_receiver_fixture_refuses_production_carrier_overhead(tmp_path: Path) -> None:
    control = tmp_path / "control.zip"
    boundary = tmp_path / "boundary.bgj"
    replay = tmp_path / "replay.r1k"
    xi0 = tmp_path / "xi0.xi0"
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    _base_zip(control)
    _packet(boundary)
    replay.write_bytes(encode_replay_payload(()))
    xi0.write_bytes(encode_xi0_payload(np.full(600, 31.0, dtype=np.float32)))
    kwargs = {
        "control_archive": control,
        "boundary_packet": boundary,
        "replay_payload": replay,
        "xi0_payload": xi0,
        "source_manifest_hashes": {"vjp": "a" * 64},
    }
    expected_refusal = "compiled carrier delta 2114 exceeds conditional limit 1852"
    with pytest.raises(R1B2CompileError, match=expected_refusal):
        compile_candidate_archive(output=first, **kwargs)
    with pytest.raises(R1B2CompileError, match=expected_refusal):
        compile_candidate_archive(output=second, **kwargs)
    assert not first.exists()
    assert not second.exists()
    assert list(tmp_path.glob(".*first*")) == []
    assert list(tmp_path.glob(".*second*")) == []


def test_rejected_oversize_candidate_leaves_no_output_or_staging(tmp_path: Path) -> None:
    control = tmp_path / "control.zip"
    boundary = tmp_path / "boundary.bgj"
    replay = tmp_path / "replay.r1k"
    xi0 = tmp_path / "xi0.xi0"
    output = tmp_path / "candidate.zip"
    _base_zip(control)
    _packet(boundary)
    replay.write_bytes(
        encode_replay_payload(
            tuple(ReplayWrite(pair // 1_164, 0, 0, pair % 1_164, 0, pair % 256) for pair in range(800))
        )
    )
    xi0.write_bytes(encode_xi0_payload(np.linspace(29.0, 33.0, 600, dtype=np.float32)))
    with pytest.raises(R1B2CompileError, match="conditional limit"):
        compile_candidate_archive(
            control_archive=control,
            boundary_packet=boundary,
            replay_payload=replay,
            xi0_payload=xi0,
            source_manifest_hashes={"vjp": "a" * 64},
            output=output,
        )
    assert not output.exists()
    assert list(tmp_path.glob(".*candidate*")) == []


def test_compile_refuses_non_n600_boundary_packet(tmp_path: Path) -> None:
    control = tmp_path / "control.zip"
    boundary = tmp_path / "boundary.bgj"
    replay = tmp_path / "replay.r1k"
    xi0 = tmp_path / "xi0.xi0"
    _base_zip(control)
    packet = BoundaryCoordinatePacket(
        family=FrameFamily.WINDOWED_CURVELET,
        frame_config=asdict(WindowedCurveletConfig(n_scales=1, n_orient0=2, n_trans=1)),
        scorer_height=2,
        scorer_width=2,
        atom_indices=np.asarray([0], dtype=np.uint32),
        coefficients=np.zeros((2, 1, 3), dtype=np.int8),
        scales=np.ones(2, dtype=np.float16),
    )
    boundary.write_bytes(encode_boundary_packet(packet))
    replay.write_bytes(b"replay")
    xi0.write_bytes(b"xi0")
    with pytest.raises(R1B2CompileError, match="exactly n600"):
        compile_candidate_archive(
            control_archive=control,
            boundary_packet=boundary,
            replay_payload=replay,
            xi0_payload=xi0,
            source_manifest_hashes={},
            output=tmp_path / "candidate.zip",
        )


def test_compile_refuses_nonproduction_boundary_geometry(tmp_path: Path) -> None:
    control = tmp_path / "control.zip"
    boundary = tmp_path / "boundary.bgj"
    replay = tmp_path / "replay.r1k"
    xi0 = tmp_path / "xi0.xi0"
    _base_zip(control)
    packet = BoundaryCoordinatePacket(
        family=FrameFamily.WINDOWED_CURVELET,
        frame_config=asdict(WindowedCurveletConfig(n_scales=1, n_orient0=2, n_trans=1)),
        scorer_height=2,
        scorer_width=2,
        atom_indices=np.asarray([0], dtype=np.uint32),
        coefficients=np.zeros((600, 1, 3), dtype=np.int8),
        scales=np.ones(600, dtype=np.float16),
    )
    boundary.write_bytes(encode_boundary_packet(packet))
    replay.write_bytes(encode_replay_payload(()))
    xi0.write_bytes(encode_xi0_payload(np.full(600, 31.0, dtype=np.float32)))
    with pytest.raises(R1B2CompileError, match="production 384x512"):
        compile_candidate_archive(
            control_archive=control,
            boundary_packet=boundary,
            replay_payload=replay,
            xi0_payload=xi0,
            source_manifest_hashes={"vjp": "a" * 64},
            output=tmp_path / "candidate.zip",
        )


def test_absent_rank4_secants_is_explicit_blocker() -> None:
    audited, blockers = audit_rank4_secants(None, vjp_campaign_sha256="a" * 64)
    assert audited is None
    assert blockers == ["R1B2_RANK4_FIRST_ORDER_REALIZED_SECANT_CUSTODY_ABSENT"]


def _rank4_manifest(tmp_path: Path) -> Path:
    boundary = tmp_path / "boundary.bgj"
    _packet(boundary)
    tensors: dict[tuple[str, int], dict] = {}
    for object_name, object_type in (
        ("frechet_first_order", "frechet_first_order_tangent.v1"),
        ("realized_uint8_secant", "realized_uint8_endpoint_secant.v1"),
    ):
        for count in (27, 28):
            path = tmp_path / f"{object_name}_{count}.bin"
            path.write_bytes(f"{object_name}:{count}".encode())
            tensors[(object_name, count)] = {
                "object_type": object_type,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "shape": [count, 4],
                "dtype": "float32",
            }
    rows = []
    for pair_index in range(600):
        count = 28 if pair_index < 119 else 27
        rows.append(
            {
                "pair_index": pair_index,
                "batch_size": 16,
                "head_rank": 4,
                "stratum_counts": {
                    "moderate_margin_1e_3_to_1": count,
                    "tie_tight_lt_1e_3": 0,
                    "other": 0,
                },
                "custody_blocks": [
                    {
                        "block_id": f"pair-{pair_index:04d}-moderate",
                        "base_id": f"control-pair-{pair_index:04d}",
                        "base_sha256": "b" * 64,
                        "delta_sha256": "c" * 64,
                        "delta_norm": 0.25,
                        "norm_id": "pixel_l2_fp32",
                        "evaluation_scale": 1.0,
                        "remainder_bound": 0.01,
                        "quantization_cells_crossed": count,
                        "hard_oracle_endpoint": {
                            "schema": "r1b2_hard_oracle_endpoint.v1",
                            "endpoint_id": f"endpoint-{pair_index:04d}",
                            "endpoint_sha256": "d" * 64,
                            "pair_index": pair_index,
                            "batch_size": 16,
                            "seed": 1234,
                            "through_uint8_rounding": True,
                            "realized_flip_count": count,
                        },
                        "frechet_first_order": tensors[("frechet_first_order", count)],
                        "realized_uint8_secant": tensors[("realized_uint8_secant", count)],
                    }
                ],
            }
        )
    manifest = tmp_path / "rank4.json"
    _write_json(
        manifest,
        {
            "schema": RANK4_SCHEMA,
            "pair_count": 600,
            "score_claim": False,
            "batch_size": 16,
            "seed": 1234,
            "head_rank": 4,
            "moderate_margin_lower_inclusive": 1e-3,
            "moderate_margin_upper_exclusive": 1.0,
            "moderate_margin_flip_count": 16_319,
            "moderate_margin_score_debt": 0.01383,
            "vjp_campaign_sha256": "a" * 64,
            "boundary_packet_path": str(boundary),
            "boundary_packet_sha256": sha256_file(boundary),
            "per_pair": rows,
        },
    )
    return manifest


def test_rank4_custody_keeps_frechet_tangent_and_uint8_secant_typed_apart(tmp_path: Path) -> None:
    manifest = _rank4_manifest(tmp_path)
    audited, blockers = audit_rank4_secants(manifest, vjp_campaign_sha256="a" * 64)
    assert blockers == []
    assert audited is not None
    value = json.loads(manifest.read_text())
    value["per_pair"][0]["custody_blocks"][0]["frechet_first_order"]["object_type"] = (
        "realized_uint8_endpoint_secant.v1"
    )
    _write_json(manifest, value)
    with pytest.raises(R1B2CompileError, match="frechet_first_order tensor custody mismatch"):
        audit_rank4_secants(manifest, vjp_campaign_sha256="a" * 64)


def _full_kernel_manifest(tmp_path: Path) -> Path:
    replay = tmp_path / "replay.r1k"
    preimages = tmp_path / "preimages.bin"
    replay.write_bytes(b"compact-replay")
    preimages.write_bytes(bytes([0, 127, 255]))
    manifest = tmp_path / "full_kernel.json"
    _write_json(
        manifest,
        {
            "schema": "r1b2_full_kernel_mdl_selection.v1",
            "pair_count": 600,
            "score_claim": False,
            "offline_search": True,
            "receiver_search": False,
            "replay_schema": "r1b2_compact_full_kernel_replay.v1",
            "hard_oracle_after_preimage_selection": True,
            "preimage_custody": {
                "schema": "r1b2_full_resize_kernel_uint8_preimages.v1",
                "dtype": "uint8",
                "lower_bound": 0,
                "upper_bound": 255,
                "selection_objective": "minimum_description_length",
                "exact_search": True,
                "preimage_count": 3,
                "path": str(preimages),
                "sha256": sha256_file(preimages),
            },
            "receiver_proof": {
                "search_invocations": 0,
                "deterministic_decode_runs": 2,
                "decoded_sha256_run1": "a" * 64,
                "decoded_sha256_run2": "a" * 64,
                "receiver_entrypoint": "inflate.py:main",
                "receiver_source_sha256": "b" * 64,
            },
            "replay_path": str(replay),
            "replay_sha256": sha256_file(replay),
        },
    )
    return manifest


def test_full_kernel_requires_exact_bounded_uint8_preimages_before_oracle(tmp_path: Path) -> None:
    manifest = _full_kernel_manifest(tmp_path)
    audited, blockers = audit_full_kernel(manifest)
    assert blockers == []
    assert audited is not None
    value = json.loads(manifest.read_text())
    value["preimage_custody"]["dtype"] = "float32"
    _write_json(manifest, value)
    with pytest.raises(R1B2CompileError, match="exact bounded uint8 MDL"):
        audit_full_kernel(manifest)


def test_partial_receipt_preserves_scope_and_every_headline_stratum() -> None:
    control = {
        "row": {
            "archive_bytes": 94_344,
            "archive_sha256": "d" * 64,
            "pair_count": 600,
            "batch_size": 16,
            "seed": 1234,
            "d_seg": 0.003515794640406966,
            "d_pose": 127.36588287353516,
            "score": 36.10275630841103,
        }
    }
    vjp = {
        "completed_pair_count": 64,
        "missing_pair_count": 536,
        "refused_pair_ids": [11],
    }
    receipt = build_receipt(
        control=control,
        vjp=vjp,
        rank4=None,
        full_kernel=None,
        xi0=None,
        blockers=["VJP_CAMPAIGN_NOT_TERMINAL_COMPLETE_N600"],
    )
    assert receipt["verdict"] == "DECOMPOSED_PARTIAL_R1B2_PRODUCTION_CUSTODY_BLOCKED"
    assert "no R1b2 candidate measurement" in receipt["verdict_scope"]
    assert receipt["headline_decomposition"]["moderate_margin_1e_3_to_1"]["flips"] == 16_319
    assert receipt["headline_decomposition"]["tie_tight"]["flips"] == 1_607
    assert receipt["headline_decomposition"]["vjp_custody"] == {
        "completed_pairs": 64,
        "missing_pairs": 536,
        "refused_pairs": [11],
    }
    assert receipt["gates"]["candidate_d_seg_gate_pass"] is None
    assert receipt["pointer"].endswith("UNMOVED")
