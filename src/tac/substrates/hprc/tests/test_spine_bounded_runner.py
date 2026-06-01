# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from tac.archive_byte_profile import contest_rate_term
from tac.substrates.hprc.representation_spine import (
    HprcRepresentationFamily,
    build_generic_neural_spine_packet,
    write_representation_spine_projection,
)
from tac.substrates.hprc.spine_acquisition import build_spine_acquisition_report
from tac.substrates.hprc.spine_bounded_runner import build_spine_bounded_runner_plan

REPO = Path(__file__).resolve().parents[5]


def test_spine_bounded_runner_forces_receiver_proof_and_coverage(tmp_path: Path) -> None:
    rnerv = _projection(
        tmp_path / "rnerv",
        family=HprcRepresentationFamily.RNERV,
        decoder=b"d" * 20,
        latents=b"l" * 8,
    )
    short_vq = _projection(
        tmp_path / "short_vq",
        family=HprcRepresentationFamily.PACT_NERV_VQ,
        decoder=b"d" * 8,
        selectors=b"s" * 4,
        manifest_extra={"num_pairs": 32},
    )
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[rnerv, short_vq],
        hard_byte_ceilings=[178_000, 216_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")

    plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        repo_root=REPO,
    )

    rows = {row["runner_row_id"]: row for row in plan["compact_base_sweep_rows"]}
    assert rows["rnerv:178000"]["action"] == (
        "receiver_proof_then_full_video_mlx_replay_then_exact_gate"
    )
    assert rows["rnerv:178000"]["requires_receiver_proof"] is True
    assert rows["rnerv:178000"]["requires_full_video_mlx_replay"] is True
    assert rows["pact_nerv_vq:178000"]["action"] == (
        "train_or_scale_to_full_coverage_emit_spine_then_receiver_proof"
    )
    assert "declared_pair_coverage_below_full_video" in rows["pact_nerv_vq:178000"]["blockers"]
    assert plan["score_claim"] is False


def test_section_value_admission_demotes_bad_residual_tokens(tmp_path: Path) -> None:
    residual = _projection(
        tmp_path / "residual",
        family=HprcRepresentationFamily.PACT_NERV,
        decoder=b"d" * 30,
        residual=b"r" * 17,
    )
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[residual],
        hard_byte_ceilings=[178_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "max_pairs": 600,
                "scope_status": {"full_video": True},
                "section_value_rows": [
                    {"variant_id": "baseline", "neutralized_section": "none"},
                    {
                        "variant_id": "neutralize_residual_rc",
                        "neutralized_section": "residual_rc",
                        "archive_bytes_removed_vs_baseline": 17,
                        "delta_nonrate_score": -0.25,
                        "delta_rate_score": -contest_rate_term(17),
                        "delta_total_mlx_score_advisory": -0.25 - contest_rate_term(17),
                        "marginal_status": "cut_candidate_distortion_nonworse",
                    },
                    {
                        "variant_id": "neutralize_decoder_qw",
                        "neutralized_section": "decoder_qw",
                        "archive_bytes_removed_vs_baseline": 30,
                        "delta_nonrate_score": 1.0,
                        "delta_rate_score": -contest_rate_term(30),
                        "delta_total_mlx_score_advisory": 1.0 - contest_rate_term(30),
                        "marginal_status": "protect_candidate_value_exceeds_rate_price",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        repo_root=REPO,
        mlx_profile_paths=[profile_path],
    )

    sections = {row["section_name"]: row for row in plan["section_value_rows"]}
    residual_row = sections["residual_rc"]
    assert residual_row["evidence_status"] == "measured_mlx_advisory"
    assert residual_row["admission_status"] == "demote_or_block_residual_tokens"
    assert residual_row["admission_objective_delta"] > 0
    residual_candidates = plan["residual_token_admission_rows"]
    assert any(
        row["schema"] == "hprc_residual_token_candidate_admission_row.v1"
        and row["admission_status"] == "demote_existing_residual_section"
        for row in residual_candidates
    )
    assert sections["decoder_qw"]["admission_status"] == "admit_section_bytes_for_receiver_proof"
    assert any(hook["status"] == "demote_from_measured_value_per_byte" for hook in plan["posterior_update_hooks"])


def test_spine_bounded_runner_cli_writes_plan(tmp_path: Path) -> None:
    tool = _load_tool()
    manifest = _projection(
        tmp_path / "rnerv",
        family=HprcRepresentationFamily.RNERV,
        decoder=b"d" * 20,
        latents=b"l" * 8,
    )
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[manifest],
        hard_byte_ceilings=[178_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    output = tmp_path / "runner.json"

    rc = tool.main(
        [
            "--acquisition-report",
            acquisition_path.as_posix(),
            "--output",
            output.as_posix(),
            "--repo-root",
            REPO.as_posix(),
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "hprc_spine_bounded_runner_plan.v1"
    assert payload["compact_base_sweep_rows"][0]["requires_receiver_proof"] is True


def _projection(
    out: Path,
    *,
    family: HprcRepresentationFamily,
    decoder: bytes,
    latents: bytes = b"",
    codebooks: bytes = b"",
    selectors: bytes = b"",
    residual: bytes = b"",
    manifest_extra: dict[str, object] | None = None,
) -> Path:
    spine = build_generic_neural_spine_packet(
        family=family,
        decoder_blob=decoder,
        latents_blob=latents,
        codebooks_blob=codebooks,
        selectors_blob=selectors,
        residual_blob=residual,
        manifest_extra=manifest_extra,
    )
    written = write_representation_spine_projection(output_dir=out, spine=spine)
    return Path(written["manifest_path"])


def _load_tool():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(
        "build_hprc_spine_bounded_runner_test",
        REPO / "tools/build_hprc_spine_bounded_runner.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
