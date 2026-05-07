from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.pr106_sidechannel_stack_readiness import (
    build_pr106_sidechannel_stack_readiness_from_paths,
)
from tac.repo_io import sha256_file

REPO = Path(__file__).resolve().parents[3]


def test_pr106_stack_readiness_emits_fail_closed_meta_lagrangian_ledger(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)

    payload = build_pr106_sidechannel_stack_readiness_from_paths(
        repo_root=tmp_path,
        baseline_json_path=fixture["baseline"],
        pr106_anchor_archive=fixture["anchor"],
        latent_metadata_path=fixture["latent_metadata"],
        yshift_metadata_path=fixture["yshift_metadata"],
        lrl1_metadata_path=fixture["lrl1_metadata"],
        three_sister_stacked_metadata_path=fixture["three_sister_metadata"],
        wavelet_sidechannel_manifest_path=fixture["wavelet_manifest"],
        wavelet_stacked_metadata_path=fixture["wavelet_stack_metadata"],
        wavelet_apply_gate_path=fixture["wavelet_gate"],
        wr01_exact_eval_packet_path=fixture["wr01_packet"],
    )

    assert payload["schema"] == "pr106_sidechannel_stack_readiness.v1"
    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["remote_gpu_run"] is False
    assert payload["ready_for_local_stack_planning"] is True
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "latent_exact_cuda_artifact_missing" in payload["blockers"]
    assert "yshift_exact_cuda_artifact_missing" in payload["blockers"]
    assert "lrl1_exact_cuda_artifact_missing" in payload["blockers"]
    assert "wr01_exact_eval_packet_exact_cuda_artifact_missing" in payload["blockers"]
    assert "wavelet_gate:requires_component_benefit_evidence_over_break_even" in payload["blockers"]

    ledger = payload["meta_lagrangian_atom_ledger"]
    assert ledger["score_claim"] is False
    assert ledger["ready_for_exact_eval_dispatch"] is False
    assert ledger["atom_count"] == 6

    rows = {row["atom_id"]: row for row in ledger["rows"]}
    wr01 = rows["pr106_sidechannel_stack:wr01_exact_eval_packet"]
    assert wr01["byte_delta"] == -9
    assert wr01["expected_total_score_delta"] < 0
    assert wr01["byte_closed_archive_manifest_attached"] is True
    assert wr01["archive_ready_for_stack_review"] is True
    assert wr01["ready_for_exact_eval_dispatch"] is False
    assert "requires_exact_cuda_auth_eval" in wr01["exact_dispatch_blockers"]["blockers"]

    latent = rows["pr106_sidechannel_stack:latent"]
    assert latent["proxy_row"] is True
    assert latent["byte_closed_archive_manifest_attached"] is False
    assert "proxy_row_not_dispatchable" in latent["dispatch_blockers"]


def test_pr106_stack_readiness_blocks_truthy_score_claim(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    latent_payload = json.loads(fixture["latent_metadata"].read_text(encoding="utf-8"))
    latent_payload["score_claim"] = True
    fixture["latent_metadata"].write_text(json.dumps(latent_payload), encoding="utf-8")

    payload = build_pr106_sidechannel_stack_readiness_from_paths(
        repo_root=tmp_path,
        baseline_json_path=fixture["baseline"],
        pr106_anchor_archive=fixture["anchor"],
        latent_metadata_path=fixture["latent_metadata"],
        yshift_metadata_path=fixture["yshift_metadata"],
        lrl1_metadata_path=fixture["lrl1_metadata"],
        three_sister_stacked_metadata_path=fixture["three_sister_metadata"],
        wavelet_sidechannel_manifest_path=fixture["wavelet_manifest"],
        wavelet_stacked_metadata_path=fixture["wavelet_stack_metadata"],
        wavelet_apply_gate_path=fixture["wavelet_gate"],
        wr01_exact_eval_packet_path=fixture["wr01_packet"],
    )

    assert payload["ready_for_local_stack_planning"] is False
    assert "latent_score_claim_true" in payload["blockers"]


def test_build_pr106_sidechannel_stack_readiness_cli(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    out = tmp_path / "stack_readiness.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_pr106_sidechannel_stack_readiness.py"),
            "--baseline-json",
            str(fixture["baseline"]),
            "--pr106-anchor-archive",
            str(fixture["anchor"]),
            "--latent-metadata",
            str(fixture["latent_metadata"]),
            "--yshift-metadata",
            str(fixture["yshift_metadata"]),
            "--lrl1-metadata",
            str(fixture["lrl1_metadata"]),
            "--three-sister-stacked-metadata",
            str(fixture["three_sister_metadata"]),
            "--wavelet-sidechannel-manifest",
            str(fixture["wavelet_manifest"]),
            "--wavelet-stacked-metadata",
            str(fixture["wavelet_stack_metadata"]),
            "--wavelet-apply-gate",
            str(fixture["wavelet_gate"]),
            "--wr01-exact-eval-packet",
            str(fixture["wr01_packet"]),
            "--json-out",
            str(out),
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "pr106_sidechannel_stack_readiness.v1"
    assert payload["meta_lagrangian_atom_ledger"]["atom_count"] == 6


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    anchor = _write_bytes(tmp_path / "anchor.zip", 1000)
    latent_archive = _write_bytes(tmp_path / "latent.zip", 1023)
    yshift_archive = _write_bytes(tmp_path / "yshift.zip", 1044)
    lrl1_archive = _write_bytes(tmp_path / "lrl1.zip", 1050)
    wr01_archive = _write_bytes(tmp_path / "wr01.zip", 991)
    archive_manifest = tmp_path / "release_surface" / "archive_manifest.json"
    archive_manifest.parent.mkdir()
    _write_json(
        archive_manifest,
        {
            "score_claim": False,
            "archive": {
                "sha256": sha256_file(wr01_archive),
                "bytes": wr01_archive.stat().st_size,
            },
        },
    )
    fixture = {
        "baseline": tmp_path / "baseline.json",
        "anchor": anchor,
        "latent_metadata": tmp_path / "latent_metadata.json",
        "yshift_metadata": tmp_path / "yshift_metadata.json",
        "lrl1_metadata": tmp_path / "lrl1_metadata.json",
        "three_sister_metadata": tmp_path / "three_sister_metadata.json",
        "wavelet_manifest": tmp_path / "wavelet_manifest.json",
        "wavelet_stack_metadata": tmp_path / "wavelet_stack_metadata.json",
        "wavelet_gate": tmp_path / "wavelet_gate.json",
        "wr01_packet": tmp_path / "wr01_packet.json",
    }
    _write_json(
        fixture["baseline"],
        {
            "archive_size_bytes": 1000,
            "avg_posenet_dist": 0.01,
            "avg_segnet_dist": 0.001,
            "score_recomputed_from_components": 0.2,
        },
    )
    _write_json(
        fixture["latent_metadata"],
        {
            "score_claim": False,
            "archive_path": latent_archive.as_posix(),
            "dispatch_blockers": ["requires_scorer_backed_cuda_latent_search"],
        },
    )
    _write_json(
        fixture["yshift_metadata"],
        {
            "score_claim": False,
            "archive_path": yshift_archive.as_posix(),
        },
    )
    _write_json(
        fixture["lrl1_metadata"],
        {
            "score_claim": False,
            "archive_path": lrl1_archive.as_posix(),
        },
    )
    _write_json(
        fixture["three_sister_metadata"],
        {
            "score_claim": False,
            "archive_path": (tmp_path / "three_sister.zip").as_posix(),
            "delta_bytes_vs_pr106_zip": 109,
            "dispatch_blockers": ["compose_time_scaffold_only"],
        },
    )
    _write_json(
        fixture["wavelet_manifest"],
        {
            "score_claim": False,
            "candidate_archive_byte_delta": 388,
            "dispatch_blockers": ["candidate_sidechannel_not_applied_by_inflate_runtime"],
        },
    )
    _write_json(
        fixture["wavelet_stack_metadata"],
        {
            "score_claim": False,
            "archive_path": (tmp_path / "wavelet_stack.zip").as_posix(),
            "delta_bytes_vs_pr106_zip": 387,
            "wavelet_runtime_mode": "explicit_noop_consume_only",
        },
    )
    _write_json(
        fixture["wavelet_gate"],
        {
            "score_claim": False,
            "archive_byte_delta": 387,
            "dispatch_blockers": [
                "requires_component_benefit_evidence_over_break_even",
                "requires_exact_cuda_auth_eval",
            ],
        },
    )
    _write_json(
        fixture["wr01_packet"],
        {
            "score_claim": False,
            "dispatch_attempted": False,
            "static_packet_ready": True,
            "candidate_static_preflight_ready": True,
            "byte_delta": -9,
            "evidence_grade": "empirical_archive_candidate_until_exact_cuda",
            "archive_identity": {"path": wr01_archive.as_posix()},
            "source_archive_sha256": "a" * 64,
            "source_archive_bytes": anchor.stat().st_size,
            "release_surface": {
                "files": {
                    "archive_manifest.json": {
                        "path": archive_manifest.as_posix(),
                    }
                }
            },
        },
    )
    return fixture


def _write_bytes(path: Path, size: int) -> Path:
    path.write_bytes(b"x" * size)
    return path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
