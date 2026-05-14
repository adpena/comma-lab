# SPDX-License-Identifier: MIT
from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

import brotli
import pytest

from tac.pr95_hnerv import sha256_bytes, sha256_file
from tac.pr95_residual_atoms import PR95AtomPlanError, emit_plan


def _stored_zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)


def _latent_raw(rows: list[list[int]]) -> bytes:
    n_pairs = len(rows)
    latent_dim = len(rows[0])
    out = io.BytesIO()
    out.write(struct.pack("<II", n_pairs, latent_dim))
    out.write(b"\x00\x00" * latent_dim)
    out.write(b"\x00<" * latent_dim)
    lo = bytearray(n_pairs * latent_dim)
    hi = bytearray(n_pairs * latent_dim)
    previous = [0] * latent_dim
    offset = 0
    for pair_index, row in enumerate(rows):
        for dim_index, value in enumerate(row):
            delta = value if pair_index == 0 else value - previous[dim_index]
            zz = 2 * delta if delta >= 0 else -2 * delta - 1
            lo[offset] = zz & 0xFF
            hi[offset] = (zz >> 8) & 0xFF
            offset += 1
        previous = list(row)
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
                "score_recomputed_from_components": 0.23092,
                "n_samples": 3,
                "provenance": {"archive_sha256": sha256_file(archive)},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _component_trace_json(path: Path, archive: Path, *, samples: list[dict[str, float | int]]) -> None:
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
                "trace_inputs": {"archive_sha256": sha256_file(archive), "device": "cuda:0"},
                "contest_auth_eval_cross_check": {"all_match": True},
                "samples": samples,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _fixture_archive(tmp_path: Path, rows: list[list[int]]) -> tuple[Path, Path]:
    archive = tmp_path / "archive.zip"
    _stored_zip(archive, _top_blob(rows))
    exact = tmp_path / "exact.json"
    _exact_json(exact, archive)
    return archive, exact


def test_emits_proxy_ledger_without_candidate(tmp_path: Path) -> None:
    archive, exact = _fixture_archive(tmp_path, [[1, 2], [5, 2], [5, 9]])

    manifest = emit_plan(
        source_archive=archive,
        exact_json=exact,
        output_dir=tmp_path / "out",
        component_trace_json=None,
        top_k=2,
    )

    assert manifest["score_claim"] is False
    assert manifest["exact_eval_ready"] is False
    assert manifest["candidate_build"] is None
    ledger = json.loads(Path(manifest["ledger_json"]).read_text())
    assert ledger["ranking_basis"] == "latent_proxy_only"
    assert [row["pair_index"] for row in ledger["pairs"]] == [2, 1]


def test_component_trace_emits_signed_policy_and_candidate(tmp_path: Path) -> None:
    archive, exact = _fixture_archive(tmp_path, [[1, 2], [5, 4], [2, 9]])
    trace = tmp_path / "component_trace.json"
    _component_trace_json(
        trace,
        archive,
        samples=[
            {"pair_index": 0, "posenet_dist": 0.00017185, "segnet_dist": 0.00070728, "score_combined_contribution_first_order": 0.1},
            {"pair_index": 1, "posenet_dist": 0.00025, "segnet_dist": 0.0005, "score_combined_contribution_first_order": 0.8},
            {"pair_index": 2, "posenet_dist": 0.0000937, "segnet_dist": 0.000914, "score_combined_contribution_first_order": 1.2},
        ],
    )

    manifest = emit_plan(
        source_archive=archive,
        exact_json=exact,
        output_dir=tmp_path / "out",
        component_trace_json=trace,
        top_k=3,
        signed_policy_pairs=2,
        signed_policy_dims_per_pair=2,
        build_generated_signed_policy=True,
    )

    signed = json.loads(Path(manifest["signed_policy_json"]).read_text())
    assert signed["score_claim"] is False
    assert signed["exact_eval_ready"] is False
    assert signed["atoms"]
    build = manifest["candidate_build"]
    assert build["score_claim"] is False
    assert build["exact_eval_ready"] is False
    assert Path(build["archive"]).is_file()
    assert build["archive_sha256"] != sha256_file(archive)


def test_build_candidate_from_explicit_latent_atom_plan(tmp_path: Path) -> None:
    archive, exact = _fixture_archive(tmp_path, [[1, 2], [5, 2], [5, 9]])
    member_blob = zipfile.ZipFile(archive).read("0.bin")
    atom_plan = tmp_path / "atom_plan.json"
    atom_plan.write_text(
        json.dumps(
            {
                "source_archive_sha256": sha256_file(archive),
                "source_member_sha256": sha256_bytes(member_blob),
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
        build_plan_json=atom_plan,
        top_k=2,
    )

    build = manifest["candidate_build"]
    assert build["archive_bytes"] > 0
    assert build["archive_sha256"] != sha256_file(archive)


def test_rejects_noop_sidecar_and_duplicate_atom_plans(tmp_path: Path) -> None:
    archive, exact = _fixture_archive(tmp_path, [[1, 2], [5, 2], [5, 9]])
    member_blob = zipfile.ZipFile(archive).read("0.bin")
    base = {
        "source_archive_sha256": sha256_file(archive),
        "source_member_sha256": sha256_bytes(member_blob),
        "forbid_sidecars": True,
    }
    atom_plan = tmp_path / "bad.json"

    atom_plan.write_text(json.dumps({**base, "forbid_sidecars": False, "atoms": [
        {"kind": "latent_uint8_set", "pair_index": 1, "dim_index": 0, "expected_old_value": 5, "value": 4}
    ]}))
    with pytest.raises(PR95AtomPlanError, match="forbid sidecars"):
        emit_plan(source_archive=archive, exact_json=exact, output_dir=tmp_path / "out_sidecar", build_plan_json=atom_plan)

    atom_plan.write_text(json.dumps({**base, "atoms": [
        {"kind": "latent_uint8_set", "pair_index": 1, "dim_index": 0, "expected_old_value": 5, "value": 5}
    ]}))
    with pytest.raises(PR95AtomPlanError, match="no-op"):
        emit_plan(source_archive=archive, exact_json=exact, output_dir=tmp_path / "out_noop", build_plan_json=atom_plan)

    atom_plan.write_text(json.dumps({**base, "atoms": [
        {"kind": "latent_uint8_delta", "pair_index": 1, "dim_index": 0, "expected_old_value": 5, "delta": 1},
        {"kind": "latent_uint8_delta", "pair_index": 1, "dim_index": 0, "expected_old_value": 6, "delta": -1},
    ]}))
    with pytest.raises(PR95AtomPlanError, match="duplicate target"):
        emit_plan(source_archive=archive, exact_json=exact, output_dir=tmp_path / "out_dup", build_plan_json=atom_plan)
