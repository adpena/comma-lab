# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "probe_z6_predictive_coding_vs_identity_disambiguator.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("z6_disambiguator_tool", TOOL_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _stats(*, identity: bool, loss: float, archive_bytes: int = 1024) -> dict[str, object]:
    return {
        "lane_id": "lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516",
        "substrate_tag": "time_traveler_l5_z6",
        "smoke": True,
        "requested_epochs": 3,
        "epochs": 3,
        "smoke_epoch_cap": 3,
        "final_loss_proxy": loss,
        "final_recon": loss / 2.0,
        "final_residual": loss / 4.0,
        "archive_bytes": archive_bytes,
        "lambda_residual_entropy": 1.0,
        "predictor_kernel_size": 3,
        "paired_control_initialization": (
            "shared_modules_seed_order_matched_v2"
        ),
        "smoke_target_mode": "real-video",
        "smoke_ego_motion_mode": "ramp",
        "ego_motion_nonzero_fraction": 1.0,
        "ego_motion_l2": 2.714,
        "identity_predictor": identity,
        "score_claim_valid": False,
        "evidence_grade": "smoke-no-scorer",
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
    }


def _write_single_member_zip(path: Path, member: str, blob: bytes) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, blob)


def _score(seg: float, pose: float, archive_bytes: int) -> float:
    return 100.0 * seg + (10.0 * pose) ** 0.5 + 25.0 * archive_bytes / 37_545_489.0


def _archive_eval_json(
    *,
    archive_zip: Path,
    seg: float,
    pose: float,
    archive_sha_at_top_level: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "archive_size_bytes": archive_zip.stat().st_size,
        "n_samples": 600,
        "score_axis": "contest_cuda",
        "evidence_grade": "A++",
        "score_recomputed_from_components": _score(
            seg,
            pose,
            archive_zip.stat().st_size,
        ),
        "score_claim_valid": True,
        "promotion_eligible": True,
    }
    if archive_sha_at_top_level:
        payload["archive_sha256"] = hashlib_sha256(archive_zip)
    else:
        payload["provenance"] = {"archive_sha256": hashlib_sha256(archive_zip)}
    return payload


def hashlib_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archive_pair(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    from tac.substrates.time_traveler_l5_z6 import (
        Z6PredictiveCodingConfig,
        Z6PredictiveCodingSubstrate,
        pack_archive,
    )

    torch.manual_seed(123)
    cfg = Z6PredictiveCodingConfig(
        latent_dim=4,
        decoder_embed_dim=4,
        decoder_channels=(4,),
        decoder_num_upsample_blocks=1,
        num_pairs=2,
        output_height=6,
        output_width=8,
        predictor_hidden_dim=8,
        predictor_film_mlp_hidden_dim=4,
        predictor_ego_motion_dim=3,
    )
    sub = Z6PredictiveCodingSubstrate(cfg)
    base_meta = {
        "decoder_embed_dim": cfg.decoder_embed_dim,
        "decoder_channels": list(cfg.decoder_channels),
        "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "predictor_hidden_dim": cfg.predictor_hidden_dim,
        "predictor_film_mlp_hidden_dim": cfg.predictor_film_mlp_hidden_dim,
        "predictor_architecture": "unit_test_single_layer_film",
        "predictor_depth": 1,
        "smoke": True,
    }
    full_blob = pack_archive(
        sub.encoder.state_dict(),
        sub.decoder.state_dict(),
        sub.predictor.state_dict(),
        sub.latent_init.detach().cpu(),
        sub.residuals.detach().cpu(),
        sub.ego_motion_buffer.detach().cpu(),
        dict(base_meta),
        identity_predictor=False,
    )
    identity_blob = pack_archive(
        sub.encoder.state_dict(),
        sub.decoder.state_dict(),
        sub.predictor.state_dict(),
        sub.latent_init.detach().cpu(),
        sub.residuals.detach().cpu(),
        sub.ego_motion_buffer.detach().cpu(),
        {**base_meta, "identity_predictor_disambiguator": True},
        identity_predictor=True,
    )
    full_path = tmp_path / "0.bin"
    identity_path = tmp_path / "0_identity_predictor_disambiguator.bin"
    full_zip = tmp_path / "archive.zip"
    identity_zip = tmp_path / "archive_identity_predictor_disambiguator.zip"
    full_path.write_bytes(full_blob)
    identity_path.write_bytes(identity_blob)
    _write_single_member_zip(full_zip, "0.bin", full_blob)
    _write_single_member_zip(identity_zip, "0.bin", identity_blob)
    return full_path, identity_path, full_zip, identity_zip


def test_z6_disambiguator_plan_is_fail_closed_and_paired() -> None:
    tool = _load_tool()

    payload = tool.build_plan_payload(epochs=3, device="cpu", seed=7)

    assert payload["schema"] == tool.SCHEMA
    assert payload["verdict"] == "pending_paired_smoke_stats"
    assert payload["paired_control_initialization"] == (
        "shared_modules_seed_order_matched_v2"
    )
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["ready_for_paid_dispatch"] is False
    assert payload["paradigm_claim_allowed"] is False
    commands = payload["paired_smoke_commands"]
    assert [row["identity_predictor"] for row in commands] == [False, True]
    assert "--smoke" in commands[0]["command"]
    assert "--smoke-ego-motion-mode" in commands[0]["command"]
    assert "--smoke-target-mode" in commands[0]["command"]
    assert "real-video" in commands[0]["command"]
    assert "real-video" in commands[1]["command"]
    assert "--identity-predictor" not in commands[0]["command"]
    assert "--identity-predictor" in commands[1]["command"]
    assert "no_contest_cpu_cuda_pair" in payload["blockers"]


def test_z6_disambiguator_compares_paired_smoke_stats(tmp_path: Path) -> None:
    tool = _load_tool()
    full_path = tmp_path / "experiments/results/z6/full/stats.json"
    identity_path = tmp_path / "experiments/results/z6/identity/stats.json"
    full_path.parent.mkdir(parents=True)
    identity_path.parent.mkdir(parents=True)
    full_path.write_text(
        json.dumps(_stats(identity=False, loss=0.10, archive_bytes=1200)),
        encoding="utf-8",
    )
    identity_path.write_text(
        json.dumps(_stats(identity=True, loss=0.15, archive_bytes=1000)),
        encoding="utf-8",
    )

    payload = tool.evaluate_stats_pair(
        full_stats_path=full_path,
        identity_stats_path=identity_path,
        repo_root=tmp_path,
    )

    assert payload["verdict"] == "full_film_predictor_proxy_lower_loss"
    assert payload["evidence_grade"] == "smoke_proxy_real_video_pair_no_scorer"
    assert payload["proxy_preferred_mode"] == "full_film_predictor"
    assert payload["score_claim"] is False
    assert payload["paradigm_claim_allowed"] is False
    assert payload["deltas"]["identity_minus_full_loss_proxy"] == pytest.approx(0.05)
    assert payload["deltas"]["full_minus_identity_archive_bytes"] == 200
    assert payload["result_review"]["classification"] == "real_video_smoke_proxy_only"
    assert payload["result_review"]["paired_control_initialization"] == (
        "shared_modules_seed_order_matched_v2"
    )
    assert "smoke_proxy_real_video_no_scorer" in payload["blockers"]
    assert [row["mode"] for row in payload["source_stats"]] == [
        "full_film_predictor",
        "identity_predictor",
    ]
    assert payload["source_stats"][0]["stats_payload"]["identity_predictor"] is False
    assert payload["source_stats"][1]["stats_payload"]["identity_predictor"] is True
    assert payload["source_stats"][0]["paired_control_initialization"] == (
        "shared_modules_seed_order_matched_v2"
    )


def test_z6_disambiguator_rejects_authoritative_or_mismatched_stats(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    full_path = tmp_path / "full.json"
    identity_path = tmp_path / "identity.json"
    full = _stats(identity=False, loss=0.10)
    identity = _stats(identity=True, loss=0.15)
    identity["score_claim_valid"] = True
    full_path.write_text(json.dumps(full), encoding="utf-8")
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(ValueError, match="score_claim_valid must be false"):
        tool.evaluate_stats_pair(
            full_stats_path=full_path,
            identity_stats_path=identity_path,
            repo_root=tmp_path,
        )

    identity["score_claim_valid"] = False
    identity["paired_control_initialization"] = "stale_unmatched_seed_order"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="paired_control_initialization"):
        tool.evaluate_stats_pair(
            full_stats_path=full_path,
            identity_stats_path=identity_path,
            repo_root=tmp_path,
        )

    identity["paired_control_initialization"] = (
        "shared_modules_seed_order_matched_v2"
    )
    identity["predictor_kernel_size"] = 5
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="must match predictor_kernel_size"):
        tool.evaluate_stats_pair(
            full_stats_path=full_path,
            identity_stats_path=identity_path,
            repo_root=tmp_path,
        )


def test_z6_disambiguator_rejects_wrong_substrate(tmp_path: Path) -> None:
    tool = _load_tool()
    full_path = tmp_path / "full.json"
    identity_path = tmp_path / "identity.json"
    full = _stats(identity=False, loss=0.10)
    identity = _stats(identity=True, loss=0.15)
    full["substrate_tag"] = "time_traveler_l5_z7"
    full_path.write_text(json.dumps(full), encoding="utf-8")
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(ValueError, match="substrate_tag"):
        tool.evaluate_stats_pair(
            full_stats_path=full_path,
            identity_stats_path=identity_path,
            repo_root=tmp_path,
        )


def test_z6_disambiguator_validates_byte_closed_archive_pair(tmp_path: Path) -> None:
    tool = _load_tool()
    full_path, identity_path, full_zip, identity_zip = _archive_pair(tmp_path)

    payload = tool.evaluate_archive_pair(
        full_archive_path=full_path,
        identity_archive_path=identity_path,
        repo_root=tmp_path,
    )

    assert payload["verdict"] == "pending_paired_exact_eval_json"
    assert payload["evidence_grade"] == "byte_closed_archive_pair_no_score"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "no_paired_exact_eval_json" in payload["blockers"]
    assert all(payload["paired_archive_checks"].values())
    assert payload["source_archives"][0]["identity_predictor"] is False
    assert payload["source_archives"][1]["identity_predictor"] is True
    assert payload["source_archives"][0]["zip_path"] == str(full_zip.relative_to(tmp_path))
    assert payload["source_archives"][1]["zip_path"] == str(identity_zip.relative_to(tmp_path))
    assert payload["source_archives"][0]["zip_members"] == ["0.bin"]
    assert payload["source_archives"][1]["zip_members"] == ["0.bin"]
    assert payload["source_archives"][0]["zip_member_matches_path_bytes"] is True
    assert payload["source_archives"][1]["zip_member_matches_path_bytes"] is True
    assert payload["source_archives"][0]["zip_member_rows"][0]["sha256"] == (
        hashlib_sha256(full_path)
    )
    assert payload["deltas"]["identity_minus_full_zip_bytes"] == (
        identity_zip.stat().st_size - full_zip.stat().st_size
    )
    assert payload["result_review"]["classification"] == "byte_closed_archive_pair_no_score"


def test_z6_disambiguator_rejects_swapped_archive_flags(tmp_path: Path) -> None:
    tool = _load_tool()
    full_path, _identity_path, _full_zip, _identity_zip = _archive_pair(tmp_path)

    with pytest.raises(ValueError, match="identity_predictor=true"):
        tool.evaluate_archive_pair(
            full_archive_path=full_path,
            identity_archive_path=full_path,
            repo_root=tmp_path,
        )


def test_z6_disambiguator_compares_exact_eval_pair_fail_closed(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    full_path, identity_path, full_zip, identity_zip = _archive_pair(tmp_path)
    full_eval = tmp_path / "full_contest_auth_eval.json"
    identity_eval = tmp_path / "identity_contest_auth_eval.json"
    full_eval.write_text(
        json.dumps(
            _archive_eval_json(
                archive_zip=full_zip,
                seg=0.0005,
                pose=0.0001,
                archive_sha_at_top_level=False,
            )
        ),
        encoding="utf-8",
    )
    identity_eval.write_text(
        json.dumps(
            _archive_eval_json(archive_zip=identity_zip, seg=0.0007, pose=0.0001)
        ),
        encoding="utf-8",
    )

    payload = tool.evaluate_archive_pair(
        full_archive_path=full_path,
        identity_archive_path=identity_path,
        repo_root=tmp_path,
        full_eval_json_path=full_eval,
        identity_eval_json_path=identity_eval,
    )

    assert payload["verdict"] == "full_film_predictor_exact_eval_lower_score"
    assert payload["evidence_grade"] == "paired_exact_eval_[contest-CUDA]"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["blockers"] == []
    assert payload["deltas"]["full_minus_identity_score"] < -0.005
    assert payload["exact_eval"]["full_film_predictor"]["archive_sha256_match_expected"] is True
    assert payload["exact_eval"]["identity_predictor"]["archive_bytes_match_expected"] is True
    assert payload["result_review"]["component_score_authority"] is True


def test_z6_disambiguator_blocks_stale_zip_sidecar_member(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    full_path, identity_path, _full_zip, _identity_zip = _archive_pair(tmp_path)
    full_blob = full_path.read_bytes()
    mutated = full_blob.replace(
        b"unit_test_single_layer_film",
        b"unit_test_single_layer_filn",
        1,
    )
    assert mutated != full_blob
    full_path.write_bytes(mutated)

    payload = tool.evaluate_archive_pair(
        full_archive_path=full_path,
        identity_archive_path=identity_path,
        repo_root=tmp_path,
    )

    assert payload["verdict"] == "blocked_paired_archive_custody"
    assert payload["evidence_grade"] == "byte_closed_archive_pair_fail_closed"
    assert "full_archive_zip_member_mismatch" in payload["blockers"]
    assert payload["source_archives"][0]["zip_member_matches_path_bytes"] is False
    assert payload["score_claim"] is False


def test_z6_disambiguator_cli_writes_archive_pair_json_and_markdown(
    tmp_path: Path,
) -> None:
    _archive_pair(tmp_path)
    output_json = tmp_path / ".omx/research/z6_archive_probe.json"
    output_md = tmp_path / ".omx/research/z6_archive_probe.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--repo-root",
            str(tmp_path),
            "--run-dir",
            ".",
            "--output-json",
            str(output_json.relative_to(tmp_path)),
            "--output-md",
            str(output_md.relative_to(tmp_path)),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["verdict"] == "pending_paired_exact_eval_json"
    assert payload["paired_archive_checks"]["predictor_state_dict_equal"] is True
    assert "## Source Archives" in output_md.read_text(encoding="utf-8")
    assert "score_claim=false" in proc.stdout


def test_z6_disambiguator_run_dir_auto_discovers_paired_exact_eval_jsons(
    tmp_path: Path,
) -> None:
    _full_path, _identity_path, full_zip, identity_zip = _archive_pair(tmp_path)
    (tmp_path / "contest_auth_eval.json").write_text(
        json.dumps(
            _archive_eval_json(
                archive_zip=full_zip,
                seg=0.0005,
                pose=0.0001,
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "contest_auth_eval_identity_predictor_disambiguator.json").write_text(
        json.dumps(
            _archive_eval_json(
                archive_zip=identity_zip,
                seg=0.0007,
                pose=0.0001,
            )
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / ".omx/research/z6_archive_probe_exact.json"
    output_md = tmp_path / ".omx/research/z6_archive_probe_exact.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--repo-root",
            str(tmp_path),
            "--run-dir",
            ".",
            "--output-json",
            str(output_json.relative_to(tmp_path)),
            "--output-md",
            str(output_md.relative_to(tmp_path)),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["verdict"] == "full_film_predictor_exact_eval_lower_score"
    assert payload["evidence_grade"] == "paired_exact_eval_[contest-CUDA]"
    assert payload["blockers"] == []
    assert payload["exact_eval"]["full_film_predictor"]["path"] == (
        "contest_auth_eval.json"
    )
    assert payload["exact_eval"]["identity_predictor"]["path"] == (
        "contest_auth_eval_identity_predictor_disambiguator.json"
    )
    assert "## Exact Eval Inputs" in output_md.read_text(encoding="utf-8")


def test_z6_disambiguator_run_dir_blocks_partial_default_exact_eval_json_pair(
    tmp_path: Path,
) -> None:
    _full_path, _identity_path, full_zip, _identity_zip = _archive_pair(tmp_path)
    (tmp_path / "contest_auth_eval.json").write_text(
        json.dumps(
            _archive_eval_json(
                archive_zip=full_zip,
                seg=0.0005,
                pose=0.0001,
            )
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / ".omx/research/z6_archive_probe_partial.json"
    output_md = tmp_path / ".omx/research/z6_archive_probe_partial.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--repo-root",
            str(tmp_path),
            "--run-dir",
            ".",
            "--output-json",
            str(output_json.relative_to(tmp_path)),
            "--output-md",
            str(output_md.relative_to(tmp_path)),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 2
    assert "run-dir paired exact-eval JSON missing" in proc.stderr
    assert not output_json.exists()


def test_z6_disambiguator_cli_writes_plan_json_and_markdown(tmp_path: Path) -> None:
    output_json = tmp_path / ".omx/research/z6_probe.json"
    output_md = tmp_path / ".omx/research/z6_probe.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--repo-root",
            str(tmp_path),
            "--output-json",
            str(output_json.relative_to(tmp_path)),
            "--output-md",
            str(output_md.relative_to(tmp_path)),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["verdict"] == "pending_paired_smoke_stats"
    assert output_md.read_text(encoding="utf-8").startswith(
        "# L5 v2 Z6 identity-predictor disambiguator"
    )
    assert "score_claim=false" in proc.stdout
