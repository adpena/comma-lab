# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import tools.materialize_pact_nerv_selector_v4_section_cut_candidate as cut_tool  # noqa: E402
import tools.profile_pact_nerv_selector_v4_mlx_section_value as profiler  # noqa: E402
from tac.archive_byte_profile import contest_rate_term  # noqa: E402
from tac.auth_eval_schema import contest_formula_score  # noqa: E402
from tac.submission_archive import write_deterministic_zip_member  # noqa: E402
from tac.substrates.pact_nerv_selector_v4.architecture import (  # noqa: E402
    PactNervSelectorV4Config,
    PactNervSelectorV4Substrate,
)
from tac.substrates.pact_nerv_selector_v4.archive import (  # noqa: E402
    pack_archive,
    parse_archive,
)


def test_materialize_variants_builds_parseable_psv4_neutralizations(
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
            "latents_rc",
            "selectors_rc",
            "receiver_state",
            "residual_rc",
        ],
    )

    rows = {variant.variant_id: variant for variant in variants}
    assert set(rows) == {
        "baseline",
        "neutralize_decoder_qw",
        "neutralize_latents_rc",
        "neutralize_selectors_rc",
    }
    assert {row["section"] for row in absent} == {"receiver_state", "residual_rc"}
    decoder = parse_archive(rows["neutralize_decoder_qw"].bin_path.read_bytes())
    assert all(float(t.abs().max()) == 0.0 for t in decoder.decoder_state_dict.values())
    latents = parse_archive(rows["neutralize_latents_rc"].bin_path.read_bytes())
    assert float(latents.latents.abs().max()) == 0.0
    selectors = parse_archive(rows["neutralize_selectors_rc"].bin_path.read_bytes())
    assert selectors.selector_bytes == b""


def test_v4_profiler_emits_hprc_component_profile_with_psv4_layout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = _archive(tmp_path / "archive.zip")
    projection = tmp_path / "projection.json"
    projection.write_text('{"schema":"projection_fixture.v1"}\n', encoding="utf-8")
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
        assert kwargs["response_family_prefix"] == "pact_nerv_selector_v4_section_value"
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
                "n_samples": 600,
                "candidate_cache_pairs": 600,
                "reference_cache_pairs": 600,
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
            "latents_rc",
            "selectors_rc",
            "residual_rc",
            "--max-pairs",
            "600",
            "--device",
            "cpu",
        ]
    )

    assert rc == 0
    profile_path = (
        tmp_path / "profile" / "pact_nerv_selector_v4_mlx_section_value_profile.json"
    )
    compat_path = tmp_path / "profile" / "hprc_mlx_component_neutralization_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    compat = json.loads(compat_path.read_text(encoding="utf-8"))
    assert profile == compat
    assert profile["schema"] == "hprc_mlx_component_neutralization_profile.v1"
    assert profile["source_schema"] == "pact_nerv_selector_v4_section_value_profile.v1"
    assert profile["axis_tag"] == "[macOS-MLX research-signal]"
    assert profile["upstream_dir"] == upstream_dir.as_posix()
    assert profile["video_names_file"] == video_names_file.as_posix()
    assert profile["reference_cache_dir"] == reference_cache_root.as_posix()
    assert profile["resolved_reference_cache_dir"] == (
        reference_cache_root / "baseline"
    ).as_posix()
    assert "psv4_section_layout" in profile
    assert "psv3_section_layout" not in profile
    assert profile["residual_admission_policy"]["schema"] == (
        "pact_nerv_selector_v4_residual_admission_policy.v1"
    )
    admission = profile["byte_price_admission_plan"]
    assert admission["schema"] == "compact_nerv_byte_price_controller.v1"
    assert admission["source_schema"] == profile["schema"]
    assert admission["input_row_count"] == 5
    assert "[macOS-MLX research-signal]" in admission["decision_rows"][0][
        "axis_labels"
    ]
    assert "axis_label_missing" not in admission["decision_rows"][0]["blockers"]
    assert admission["score_claim"] is False
    assert admission["ready_for_exact_eval_dispatch"] is False
    rows = {row["variant_id"]: row for row in profile["section_value_rows"]}
    assert set(rows) == {
        "baseline",
        "neutralize_decoder_qw",
        "neutralize_latents_rc",
        "neutralize_selectors_rc",
        "residual_absent_no_admission",
    }
    assert rows["neutralize_decoder_qw"]["neutralized_section"] == "decoder_qw"
    assert rows["neutralize_decoder_qw"]["axis_tag"] == "[macOS-MLX research-signal]"
    assert rows["neutralize_decoder_qw"]["projection_manifest_path"] == (
        projection.as_posix()
    )
    assert rows["residual_absent_no_admission"]["admission_status"] == (
        "demote_residual_token_variant"
    )
    assert "full_video_mlx_response_not_executed" not in profile["blockers"]
    assert profile["scope_status"]["full_video"] == "executed"
    assert profile["mlx_response_coverage"]["status"] == "executed"
    assert "contest_cpu_cuda_exact_eval_not_executed" in profile["blockers"]
    assert profile["score_claim"] is False
    assert profile["ready_for_exact_eval_dispatch"] is False


def test_materialize_v4_section_cut_candidate_combines_measured_cuts(
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
            "latents_rc",
            "selectors_rc",
        ]
    )

    assert rc == 0
    report = json.loads(
        (tmp_path / "cut" / "pact_nerv_selector_v4_section_cut_candidate.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["sections_cut"] == ["latents_rc", "selectors_rc"]
    with zipfile.ZipFile(report["candidate_archive"]["path"]) as zf:
        arc = parse_archive(zf.read("0.bin"))
    assert float(arc.latents.abs().max()) == 0.0
    assert arc.selector_bytes == b""


def _cfg() -> PactNervSelectorV4Config:
    return PactNervSelectorV4Config(
        latent_dim=4,
        embed_dim=12,
        initial_grid_h=2,
        initial_grid_w=2,
        decoder_channels=(10, 8),
        num_upsample_blocks=2,
        num_pairs=2,
        output_height=8,
        output_width=8,
        selector_palette_size=16,
    )


def _archive(path: Path) -> Path:
    torch.manual_seed(123)
    cfg = _cfg()
    model = PactNervSelectorV4Substrate(cfg)
    state = model.state_dict()
    blob = pack_archive(
        {k: v for k, v in state.items() if k not in {"latents", "selectors"}},
        state["latents"].clone(),
        b"\x00\x01\x02\x03",
        {
            "embed_dim": cfg.embed_dim,
            "initial_grid_h": cfg.initial_grid_h,
            "initial_grid_w": cfg.initial_grid_w,
            "decoder_channels": list(cfg.decoder_channels),
            "sin_frequency": cfg.sin_frequency,
            "num_upsample_blocks": cfg.num_upsample_blocks,
            "output_height": cfg.output_height,
            "output_width": cfg.output_width,
            "selector_palette_size": cfg.selector_palette_size,
        },
        palette_size=cfg.selector_palette_size,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        write_deterministic_zip_member(zf, "0.bin", blob)
        write_deterministic_zip_member(zf, "inflate.sh", b"#!/bin/sh\n")
    return path
