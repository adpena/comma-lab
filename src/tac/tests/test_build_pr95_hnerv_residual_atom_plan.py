from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

import brotli
import pytest

from experiments.build_pr95_hnerv_residual_atom_plan import PR95AtomPlanError, emit_plan, sha256_file


def _stored_zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, payload)


def _latent_raw(rows: list[list[int]]) -> bytes:
    n_pairs = len(rows)
    latent_dim = len(rows[0])
    out = io.BytesIO()
    out.write(struct.pack("<II", n_pairs, latent_dim))
    out.write(b"\x00\x00" * latent_dim)
    out.write(b"\x00\x3c" * latent_dim)
    lo = bytearray(n_pairs * latent_dim)
    hi = bytearray(n_pairs * latent_dim)
    prev = [0] * latent_dim
    offset = 0
    for pair_index, row in enumerate(rows):
        for dim_index, value in enumerate(row):
            delta = value if pair_index == 0 else value - prev[dim_index]
            zz = 2 * delta if delta >= 0 else -2 * delta - 1
            lo[offset] = zz & 0xFF
            hi[offset] = (zz >> 8) & 0xFF
            offset += 1
        prev = row
    out.write(lo)
    out.write(hi)
    return out.getvalue()


def _top_blob(rows: list[list[int]]) -> bytes:
    meta = brotli.compress(b'{"latent_dim":2,"n_pairs":3,"eval_size":[2,2],"base_channels":1}', quality=5)
    decoder = brotli.compress(struct.pack("<I", 0), quality=5)
    latents = brotli.compress(_latent_raw(rows), quality=5)
    out = io.BytesIO()
    for payload in (meta, decoder, latents):
        out.write(struct.pack("<I", len(payload)))
        out.write(payload)
    return out.getvalue()


def _exact_json(path: Path, archive: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "archive_size_bytes": archive.stat().st_size,
                "avg_posenet_dist": 0.00017185,
                "avg_segnet_dist": 0.00070728,
                "score_pose_contribution": 0.04145479465634826,
                "score_rate_contribution": 0.11873675,
                "score_recomputed_from_components": 0.23091954465634829,
                "score_seg_contribution": 0.070728,
                "n_samples": 3,
                "provenance": {"archive_sha256": sha256_file(archive)},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _component_trace_json(
    path: Path,
    archive: Path,
    *,
    samples: list[dict[str, float | int]],
    archive_sha256: str | None = None,
    cross_check_all_match: bool = True,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "score_claim": False,
                "evidence_grade": "diagnostic_component_trace",
                "n_samples": len(samples),
                "expected_contest_samples": len(samples),
                "avg_posenet_dist": 0.00017185,
                "avg_segnet_dist": 0.00070728,
                "archive_size_bytes": archive.stat().st_size,
                "score_recomputed_from_components": 0.23091954465634829,
                "trace_inputs": {
                    "archive_sha256": archive_sha256 or sha256_file(archive),
                    "device": "cuda:0",
                },
                "contest_auth_eval_cross_check": {
                    "all_match": cross_check_all_match,
                    "checks": {
                        "n_samples": {"match": cross_check_all_match},
                        "archive_size_bytes": {"match": cross_check_all_match},
                    },
                },
                "samples": samples,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_emits_proxy_ledger_without_candidate(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _stored_zip(archive, _top_blob([[1, 2], [5, 2], [5, 9]]))
    exact = tmp_path / "exact.json"
    _exact_json(exact, archive)

    manifest = emit_plan(
        source_archive=archive,
        exact_json=exact,
        output_dir=tmp_path / "out",
        component_trace_json=None,
        top_k=2,
        build_plan_json=None,
    )

    assert manifest["score_claim"] is False
    assert manifest["component_trace"]["available"] is False
    assert manifest["dispatch_readiness"]["ready_for_exact_eval"] is False
    assert manifest["candidate_build"] is None
    ledger = json.loads((tmp_path / "out" / "pr95_hnerv_pair_opportunity_ledger.json").read_text())
    assert len(ledger) == 3
    assert ledger[0]["estimated_min_patch_bytes"] > 0
    assert ledger[0]["break_even_seg_dist_reduction"] > 0


def test_component_trace_emits_deterministic_signed_policy_and_candidate(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _stored_zip(archive, _top_blob([[1, 2], [5, 4], [2, 9]]))
    exact = tmp_path / "exact.json"
    _exact_json(exact, archive)
    trace = tmp_path / "component_trace.json"
    _component_trace_json(
        trace,
        archive,
        samples=[
            {
                "pair_index": 0,
                "posenet_dist": 0.00017185,
                "segnet_dist": 0.00070728,
                "score_combined_contribution_first_order": 0.9,
            },
            {
                "pair_index": 1,
                "posenet_dist": 0.00025000,
                "segnet_dist": 0.00050000,
                "score_combined_contribution_first_order": 0.8,
            },
            {
                "pair_index": 2,
                "posenet_dist": 0.00009370,
                "segnet_dist": 0.00091456,
                "score_combined_contribution_first_order": 1.2,
            },
        ],
    )

    manifest = emit_plan(
        source_archive=archive,
        exact_json=exact,
        output_dir=tmp_path / "out",
        component_trace_json=trace,
        top_k=3,
        build_plan_json=None,
        signed_policy_pairs=2,
        signed_policy_dims_per_pair=2,
        build_generated_signed_policy=True,
    )

    assert manifest["component_trace"]["available"] is True
    assert manifest["component_trace"]["cross_check_all_match"] is True
    assert manifest["component_trace"]["archive_sha256"] == sha256_file(archive)
    signed = manifest["signed_atom_policy"]
    assert signed["selected_pairs"] == [2, 1]
    policy = json.loads(Path(signed["path"]).read_text())
    assert policy["exact_eval_ready"] is True
    assert policy["selection"]["excluded_initial_anchor_pair"] is True
    assert policy["selection"]["selected_pairs"] == [2, 1]
    assert len(policy["atoms"]) == 4
    targets = {(atom["pair_index"], atom["dim_index"]) for atom in policy["atoms"]}
    assert len(targets) == len(policy["atoms"])
    for atom in policy["atoms"]:
        assert atom["pair_index"] != 0
        assert atom["delta"] in {-1, 1}
        assert atom["expected_old_value"] + atom["delta"] != atom["expected_old_value"]
        assert atom["source_delta_from_previous_pair"] * atom["delta"] < 0
        assert atom["dispatchable_after_lane_claim"] is True

    build = manifest["candidate_build"]
    assert build["score_claim"] is False
    assert Path(build["candidate_archive"]).exists()
    assert len(build["changed_atoms"]) == len(policy["atoms"])

    manifest_2 = emit_plan(
        source_archive=archive,
        exact_json=exact,
        output_dir=tmp_path / "out2",
        component_trace_json=trace,
        top_k=3,
        build_plan_json=None,
        signed_policy_pairs=2,
        signed_policy_dims_per_pair=2,
        build_generated_signed_policy=False,
    )
    policy_2 = json.loads(Path(manifest_2["signed_atom_policy"]["path"]).read_text())
    assert policy_2 == policy


def test_build_candidate_from_explicit_latent_atom_plan(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _stored_zip(archive, _top_blob([[1, 2], [5, 2], [5, 9]]))
    exact = tmp_path / "exact.json"
    _exact_json(exact, archive)
    member_blob = zipfile.ZipFile(archive).read("0.bin")
    atom_plan = tmp_path / "atom_plan.json"
    atom_plan.write_text(
        json.dumps(
            {
                "source_archive_sha256": sha256_file(archive),
                "source_member_sha256": __import__("hashlib").sha256(member_blob).hexdigest(),
                "forbid_sidecars": True,
                "atoms": [
                    {
                        "kind": "latent_uint8_delta",
                        "pair_index": 1,
                        "dim_index": 0,
                        "expected_old_value": 5,
                        "delta": -1,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = emit_plan(
        source_archive=archive,
        exact_json=exact,
        output_dir=tmp_path / "out",
        component_trace_json=None,
        top_k=2,
        build_plan_json=atom_plan,
    )

    build = manifest["candidate_build"]
    assert build["score_claim"] is False
    assert build["archive_byte_delta"] != 0 or build["candidate_archive_sha256"] != build["source_archive_sha256"]
    with zipfile.ZipFile(build["candidate_archive"]) as zf:
        assert zf.namelist() == ["0.bin"]
    assert manifest["dispatch_readiness"]["ready_for_exact_eval"] is True
    assert "--archive" in manifest["dispatch_readiness"]["example_exact_eval_command"]


def test_rejects_noop_and_sidecar_atom_plans(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _stored_zip(archive, _top_blob([[1, 2], [5, 2], [5, 9]]))
    exact = tmp_path / "exact.json"
    _exact_json(exact, archive)
    member_blob = zipfile.ZipFile(archive).read("0.bin")
    atom_plan = tmp_path / "noop_plan.json"
    atom_plan.write_text(
        json.dumps(
            {
                "source_archive_sha256": sha256_file(archive),
                "source_member_sha256": __import__("hashlib").sha256(member_blob).hexdigest(),
                "forbid_sidecars": False,
                "atoms": [
                    {
                        "kind": "latent_uint8_set",
                        "pair_index": 1,
                        "dim_index": 0,
                        "expected_old_value": 5,
                        "value": 5,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PR95AtomPlanError, match="forbid sidecars"):
        emit_plan(
            source_archive=archive,
            exact_json=exact,
            output_dir=tmp_path / "out",
            component_trace_json=None,
            top_k=2,
            build_plan_json=atom_plan,
        )

    atom_plan.write_text(
        json.dumps(
            {
                "source_archive_sha256": sha256_file(archive),
                "source_member_sha256": __import__("hashlib").sha256(member_blob).hexdigest(),
                "forbid_sidecars": True,
                "atoms": [
                    {
                        "kind": "latent_uint8_set",
                        "pair_index": 1,
                        "dim_index": 0,
                        "expected_old_value": 5,
                        "value": 5,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PR95AtomPlanError, match="no-op"):
        emit_plan(
            source_archive=archive,
            exact_json=exact,
            output_dir=tmp_path / "out_noop",
            component_trace_json=None,
            top_k=2,
            build_plan_json=atom_plan,
        )

    atom_plan.write_text(
        json.dumps(
            {
                "source_archive_sha256": sha256_file(archive),
                "source_member_sha256": __import__("hashlib").sha256(member_blob).hexdigest(),
                "forbid_sidecars": True,
                "atoms": [
                    {
                        "kind": "latent_uint8_delta",
                        "pair_index": 1,
                        "dim_index": 0,
                        "expected_old_value": 5,
                        "delta": 1,
                    },
                    {
                        "kind": "latent_uint8_delta",
                        "pair_index": 1,
                        "dim_index": 0,
                        "expected_old_value": 6,
                        "delta": -1,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PR95AtomPlanError, match="duplicate target"):
        emit_plan(
            source_archive=archive,
            exact_json=exact,
            output_dir=tmp_path / "out_duplicate",
            component_trace_json=None,
            top_k=2,
            build_plan_json=atom_plan,
        )


def test_rejects_component_trace_schema_mismatches(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _stored_zip(archive, _top_blob([[1, 2], [5, 4], [2, 9]]))
    exact = tmp_path / "exact.json"
    _exact_json(exact, archive)
    trace = tmp_path / "component_trace.json"
    samples = [
        {
            "pair_index": 0,
            "posenet_dist": 0.00017185,
            "segnet_dist": 0.00070728,
            "score_combined_contribution_first_order": 0.9,
        },
        {
            "pair_index": 1,
            "posenet_dist": 0.00025000,
            "segnet_dist": 0.00050000,
            "score_combined_contribution_first_order": 0.8,
        },
        {
            "pair_index": 2,
            "posenet_dist": 0.00009370,
            "segnet_dist": 0.00091456,
            "score_combined_contribution_first_order": 1.2,
        },
    ]
    _component_trace_json(trace, archive, samples=samples, archive_sha256="bad-sha")

    with pytest.raises(PR95AtomPlanError, match="archive SHA mismatch"):
        emit_plan(
            source_archive=archive,
            exact_json=exact,
            output_dir=tmp_path / "out_bad_sha",
            component_trace_json=trace,
            top_k=3,
            build_plan_json=None,
        )

    _component_trace_json(trace, archive, samples=samples, cross_check_all_match=False)
    with pytest.raises(PR95AtomPlanError, match="all_match must be true"):
        emit_plan(
            source_archive=archive,
            exact_json=exact,
            output_dir=tmp_path / "out_bad_crosscheck",
            component_trace_json=trace,
            top_k=3,
            build_plan_json=None,
        )
