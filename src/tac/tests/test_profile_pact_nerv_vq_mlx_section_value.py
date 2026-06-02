# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import tools.materialize_pact_nerv_vq_section_cut_candidate as cut_tool  # noqa: E402
import tools.profile_pact_nerv_selector_v3_mlx_section_value as shared_profiler  # noqa: E402
import tools.profile_pact_nerv_vq_mlx_section_value as profiler  # noqa: E402
from tac.archive_byte_profile import contest_rate_term  # noqa: E402
from tac.auth_eval_schema import contest_formula_score  # noqa: E402
from tac.submission_archive import write_deterministic_zip_member  # noqa: E402
from tac.substrates.hprc.representation_spine import (  # noqa: E402
    build_pact_nerv_vq_spine_from_archive,
    write_representation_spine_projection,
)
from tac.substrates.hprc.spine_acquisition import build_spine_acquisition_report  # noqa: E402
from tac.substrates.hprc.spine_bounded_runner import (  # noqa: E402
    build_spine_bounded_runner_plan,
)
from tac.substrates.pact_nerv_vq.architecture import (  # noqa: E402
    PactNervVqConfig,
    PactNervVqSubstrate,
)
from tac.substrates.pact_nerv_vq.archive import pack_archive, parse_archive  # noqa: E402


def test_reference_cache_resolver_accepts_direct_and_profile_root(
    tmp_path: Path,
) -> None:
    direct = tmp_path / "direct_reference_cache"
    direct.mkdir()
    (direct / "manifest.json").write_text('{"schema":"cache_manifest.v1"}\n')

    profile_root = tmp_path / "profile_cache_root"
    (profile_root / "baseline").mkdir(parents=True)
    (profile_root / "baseline" / "manifest.json").write_text(
        '{"schema":"cache_manifest.v1"}\n',
        encoding="utf-8",
    )

    assert profiler._resolve_reference_cache_dir(direct) == direct
    assert profiler._resolve_reference_cache_dir(profile_root) == (
        profile_root / "baseline"
    )
    with pytest.raises(FileNotFoundError):
        profiler._resolve_reference_cache_dir(tmp_path / "missing_reference_cache")


def test_materialize_caches_reuses_archive_matched_existing_cache_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "profile"
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive-bytes")
    variant = shared_profiler.VariantSpec(
        variant_id="baseline",
        neutralized_section=None,
        archive_zip_path=archive,
        submission_dir=tmp_path / "submission",
        bin_path=tmp_path / "0.bin",
        archive_bytes=archive.stat().st_size,
        archive_sha256=shared_profiler._sha256_file(archive),
        bin_sha256="unused-bin-sha",
        variant_dir=tmp_path / "variant",
    )
    cache_dir = output_dir / "mlx_caches" / "baseline"
    cache_dir.mkdir(parents=True)
    manifest = cache_dir / "manifest.json"
    manifest.write_text('{"schema":"mlx_cache_manifest.v1"}\n', encoding="utf-8")
    report = output_dir / "mlx_cache_reports" / "baseline.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "archive": {
                    "bytes": variant.archive_bytes,
                    "sha256": variant.archive_sha256,
                },
                "cache_manifest": manifest.as_posix(),
            }
        ),
        encoding="utf-8",
    )

    def fail_run(*_args, **_kwargs):
        raise AssertionError("cache materializer should not rerun")

    monkeypatch.setattr(shared_profiler.subprocess, "run", fail_run)

    rows = shared_profiler._materialize_caches(
        variants=[variant],
        output_dir=output_dir,
        repo_root=REPO,
        upstream_dir=tmp_path / "upstream",
        video_names_file=None,
        max_pairs=600,
        inflate_timeout=1800,
        allow_large_tensor_cache=True,
    )

    row = rows["baseline"]
    assert row["reused_existing_cache_report"] is True
    assert row["argv"] == []
    assert row["reuse_integrity"]["archive_sha256"] == variant.archive_sha256


def test_materialize_variants_builds_parseable_pvq_neutralizations(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path / "archive.zip")
    baseline_blob = profiler._read_archive_member(archive, "0.bin")

    variants, absent = profiler._materialize_variants(
        archive=archive,
        baseline_blob=baseline_blob,
        output_dir=tmp_path / "profile",
        requested_sections=[
            "decoder_qw",
            "codebooks_q",
            "selectors_rc",
            "receiver_state",
            "residual_rc",
        ],
    )

    rows = {variant.variant_id: variant for variant in variants}
    assert set(rows) == {
        "baseline",
        "neutralize_decoder_qw",
        "neutralize_codebooks_q",
        "neutralize_selectors_rc",
    }
    assert {row["section"] for row in absent} == {"receiver_state", "residual_rc"}
    decoder = parse_archive(rows["neutralize_decoder_qw"].bin_path.read_bytes())
    assert all(float(t.abs().max()) == 0.0 for t in decoder.decoder_state_dict.values())
    codebook = parse_archive(rows["neutralize_codebooks_q"].bin_path.read_bytes())
    assert float(codebook.codebook.abs().max()) == 0.0
    selectors = parse_archive(rows["neutralize_selectors_rc"].bin_path.read_bytes())
    assert int(selectors.indices.abs().max()) == 0


def test_vq_profiler_emits_hprc_component_profile_with_pvq_layout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = _archive(tmp_path / "archive.zip")
    projection = _projection(tmp_path, archive)
    upstream_dir = tmp_path / "upstream"
    upstream_dir.mkdir()
    video_names_file = tmp_path / "video_names.txt"
    video_names_file.write_text("0.raw\n", encoding="utf-8")
    reference_cache_root = tmp_path / "reference_cache_root"
    (reference_cache_root / "baseline").mkdir(parents=True)
    (reference_cache_root / "baseline" / "manifest.json").write_text(
        '{"schema":"cache_manifest.v1"}\n',
        encoding="utf-8",
    )

    def fake_materialize_caches(**kwargs):
        assert Path(kwargs["upstream_dir"]) == upstream_dir
        assert Path(kwargs["video_names_file"]) == video_names_file
        output_dir = Path(kwargs["output_dir"])
        rows = {}
        for variant in kwargs["variants"]:
            report = output_dir / "fake_cache_reports" / f"{variant.variant_id}.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text('{"schema":"fake_cache.v1"}\n', encoding="utf-8")
            rows[variant.variant_id] = {
                "cache_dir": (output_dir / "fake_cache" / variant.variant_id).as_posix(),
                "work_dir": (output_dir / "fake_work" / variant.variant_id).as_posix(),
                "report_output": report.as_posix(),
                "argv": ["fake-cache"],
                "stdout": "",
                "stderr_tail": "",
            }
        return rows

    def fake_mlx_responses(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        assert Path(kwargs["reference_cache_dir"]) == reference_cache_root / "baseline"
        assert kwargs["response_family_prefix"] == "pact_nerv_vq_section_value"
        rows = {}
        for index, variant in enumerate(kwargs["variants"]):
            avg_seg = 0.02 + index * 0.001
            avg_pose = 0.10 + index * 0.002
            payload = {
                "schema_version": "mlx_scorer_response_payload.v1",
                "response_family": (
                    f"{kwargs['response_family_prefix']}_{variant.variant_id}"
                ),
                "avg_segnet_dist": avg_seg,
                "avg_posenet_dist": avg_pose,
                "score_rate_contribution": contest_rate_term(variant.archive_bytes),
                "canonical_score": contest_formula_score(
                    seg_dist=avg_seg,
                    pose_dist=avg_pose,
                    archive_bytes=variant.archive_bytes,
                ),
                "components": {"artifacts": {}},
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
            response = output_dir / "mlx_responses" / f"{variant.variant_id}.json"
            response.parent.mkdir(parents=True, exist_ok=True)
            response.write_text(json.dumps(payload), encoding="utf-8")
            rows[variant.variant_id] = payload
        return rows

    monkeypatch.setattr(profiler, "_materialize_caches", fake_materialize_caches)
    monkeypatch.setattr(profiler, "_run_mlx_responses", fake_mlx_responses)

    rc = profiler.main(
        [
            "--archive",
            archive.as_posix(),
            "--projection-manifest",
            projection.as_posix(),
            "--output-dir",
            (tmp_path / "profile").as_posix(),
            "--upstream-dir",
            upstream_dir.as_posix(),
            "--video-names-file",
            video_names_file.as_posix(),
            "--reference-cache-dir",
            reference_cache_root.as_posix(),
            "--sections",
            "decoder_qw",
            "codebooks_q",
            "selectors_rc",
            "residual_rc",
            "--max-pairs",
            "600",
            "--device",
            "cpu",
        ]
    )

    assert rc == 0
    profile_path = tmp_path / "profile" / "pact_nerv_vq_mlx_section_value_profile.json"
    compat_path = tmp_path / "profile" / "hprc_mlx_component_neutralization_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    compat = json.loads(compat_path.read_text(encoding="utf-8"))
    assert profile == compat
    assert profile["schema"] == "hprc_mlx_component_neutralization_profile.v1"
    assert profile["source_schema"] == "pact_nerv_vq_section_value_profile.v1"
    assert profile["family"] == "pact_nerv_vq"
    assert profile["upstream_dir"] == upstream_dir.as_posix()
    assert profile["video_names_file"] == video_names_file.as_posix()
    assert profile["reference_cache_dir"] == reference_cache_root.as_posix()
    assert profile["resolved_reference_cache_dir"] == (
        reference_cache_root / "baseline"
    ).as_posix()
    assert "pvq_section_layout" in profile
    assert profile["projection_gap_analysis"]["schema"] == (
        "hprc_archive_projection_gap_analysis.v1"
    )
    assert profile["projection_gap_analysis"]["status"] == (
        "no_archive_projection_gap_detected_by_section_neutralization"
    )
    assert profile["projection_gap_analysis"][
        "requires_direct_model_vs_archive_replay"
    ] is False
    rows = {row["variant_id"]: row for row in profile["section_value_rows"]}
    assert set(rows) == {
        "baseline",
        "neutralize_decoder_qw",
        "neutralize_codebooks_q",
        "neutralize_selectors_rc",
        "residual_absent_no_admission",
    }
    assert rows["neutralize_codebooks_q"]["neutralized_section"] == "codebooks_q"
    assert rows["neutralize_codebooks_q"]["family"] == "pact_nerv_vq"
    assert rows["residual_absent_no_admission"]["admission_status"] == (
        "demote_residual_token_variant"
    )
    assert "contest_cpu_cuda_exact_eval_not_executed" in profile["blockers"]
    assert profile["score_claim"] is False
    assert profile["ready_for_exact_eval_dispatch"] is False


def test_bounded_runner_opens_pvq_full_video_section_value_work_order(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path / "archive.zip", num_pairs=600)
    projection = _projection(tmp_path, archive)
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[projection],
        hard_byte_ceilings=[178_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")

    plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        repo_root=REPO,
    )

    orders = plan.get("section_value_profile_work_orders", [])
    assert len(orders) == 1
    order = orders[0]
    assert order["status"] == "queued_for_full_video_mlx_section_value_profile"
    assert order["tool"] == "tools/profile_pact_nerv_vq_mlx_section_value.py"
    assert "codebooks_q" in order["sections"]
    assert "--upstream-dir" in order["argv"]
    assert (REPO / "upstream").as_posix() in order["argv"]
    assert "--max-pairs" in order["argv"]
    assert "600" in order["argv"]


def test_bounded_runner_opens_pvq_section_cut_materializer(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path / "archive.zip", num_pairs=600)
    projection = _projection(tmp_path, archive)
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[projection],
        hard_byte_ceilings=[178_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "source_schema": "pact_nerv_vq_section_value_profile.v1",
                "family": "pact_nerv_vq",
                "max_pairs": 600,
                "projection_manifest_path": projection.as_posix(),
                "scope_status": {"full_video": "executed", "section": "executed"},
                "section_value_rows": [
                    {
                        "variant_id": "neutralize_decoder_qw",
                        "family": "pact_nerv_vq",
                        "neutralized_section": "decoder_qw",
                        "marginal_status": "cut_candidate_distortion_nonworse",
                        "archive_bytes_removed_vs_baseline": 128,
                        "delta_nonrate_score": -0.5,
                        "delta_rate_score": -0.1,
                        "delta_total_mlx_score_advisory": -0.6,
                        "projection_manifest_path": projection.as_posix(),
                        "ready_for_exact_eval_dispatch": False,
                        "score_claim": False,
                    },
                    {
                        "variant_id": "neutralize_codebooks_q",
                        "family": "pact_nerv_vq",
                        "neutralized_section": "codebooks_q",
                        "marginal_status": "cut_candidate_distortion_nonworse",
                        "archive_bytes_removed_vs_baseline": 64,
                        "delta_nonrate_score": -0.25,
                        "delta_rate_score": -0.05,
                        "delta_total_mlx_score_advisory": -0.3,
                        "projection_manifest_path": projection.as_posix(),
                        "ready_for_exact_eval_dispatch": False,
                        "score_claim": False,
                    },
                ],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        repo_root=REPO,
        mlx_profile_paths=[profile_path],
    )

    work_orders = plan["section_cut_materializer_work_orders"]
    assert len(work_orders) == 1
    order = work_orders[0]
    assert order["status"] == "queued_for_byte_closed_section_cut_materializer"
    assert order["materializer_tool"] == (
        "tools/materialize_pact_nerv_vq_section_cut_candidate.py"
    )
    assert order["cut_sections"] == ["decoder_qw", "codebooks_q"]
    assert order["archive_zip_path"] == archive.as_posix()
    assert order["full_video_profile_path"] == profile_path.as_posix()
    assert "--run-receiver-proof" in order["argv"]
    assert order["score_claim"] is False


def test_materialize_vq_section_cut_candidate_combines_measured_cuts(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path / "archive.zip")
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "section_value_rows": [],
                "score_claim": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rc = cut_tool.main(
        [
            "--archive",
            archive.as_posix(),
            "--profile",
            profile.as_posix(),
            "--output-dir",
            (tmp_path / "cut").as_posix(),
            "--sections",
            "decoder_qw",
            "codebooks_q",
            "selectors_rc",
        ]
    )

    assert rc == 0
    report = json.loads(
        (tmp_path / "cut" / "pact_nerv_vq_section_cut_candidate.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["sections_cut"] == ["decoder_qw", "codebooks_q", "selectors_rc"]
    with zipfile.ZipFile(report["candidate_archive"]["path"]) as zf:
        arc = parse_archive(zf.read("0.bin"))
    assert all(float(t.abs().max()) == 0.0 for t in arc.decoder_state_dict.values())
    assert float(arc.codebook.abs().max()) == 0.0
    assert int(arc.indices.abs().max()) == 0


def _cfg() -> PactNervVqConfig:
    return PactNervVqConfig(
        latent_dim=4,
        embed_dim=12,
        initial_grid_h=2,
        initial_grid_w=2,
        decoder_channels=(10, 8),
        num_upsample_blocks=2,
        codebook_size=8,
        num_pairs=2,
        output_height=8,
        output_width=8,
    )


def _archive(path: Path, *, num_pairs: int = 2) -> Path:
    torch.manual_seed(123)
    cfg = _cfg()
    model = PactNervVqSubstrate(cfg)
    state = model.state_dict()
    blob = pack_archive(
        {
            k: v
            for k, v in state.items()
            if k
            not in {
                "latents",
                "quantizer.codebook",
                "quantizer.ema_cluster_size",
                "quantizer.ema_w",
            }
        },
        state["quantizer.codebook"].clone(),
        torch.arange(num_pairs, dtype=torch.long) % cfg.codebook_size,
        {
            "embed_dim": cfg.embed_dim,
            "initial_grid_h": cfg.initial_grid_h,
            "initial_grid_w": cfg.initial_grid_w,
            "decoder_channels": list(cfg.decoder_channels),
            "sin_frequency": cfg.sin_frequency,
            "num_upsample_blocks": cfg.num_upsample_blocks,
            "output_height": cfg.output_height,
            "output_width": cfg.output_width,
            "codebook_decay": cfg.codebook_decay,
            "commitment_weight": cfg.commitment_weight,
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        write_deterministic_zip_member(zf, "0.bin", blob)
        write_deterministic_zip_member(zf, "inflate.sh", b"#!/bin/sh\n")
    return path


def _projection(tmp_path: Path, archive: Path) -> Path:
    spine = build_pact_nerv_vq_spine_from_archive(archive)
    projection = write_representation_spine_projection(
        output_dir=tmp_path / "projection",
        spine=spine,
        basename="pact_nerv_vq_representation_spine",
    )
    return Path(projection["manifest_path"])
