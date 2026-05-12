from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _load_trainer_module():
    path = Path("experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py")
    spec = importlib.util.spec_from_file_location("phase1_t1_trainer_for_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase1_trainer_auth_eval_refuses_scaffold_before_training() -> None:
    module = _load_trainer_module()

    assert module.PHASE1_SCAFFOLD_ONLY is True
    blockers = module.phase1_scaffold_blockers()
    assert "auth_eval_custody_not_wired" in blockers
    assert "exact_cuda_score_not_run" in blockers


def test_phase1_direct_score_rate_penalty_scales_minibatch_to_full_archive() -> None:
    module = _load_trainer_module()

    per_pair_bits = 7.0
    penalty_batch_1 = module.contest_rate_penalty_from_batch_bits(
        torch.tensor(per_pair_bits * 1),
        batch_pairs=1,
        total_pairs=600,
    )
    penalty_batch_16 = module.contest_rate_penalty_from_batch_bits(
        torch.tensor(per_pair_bits * 16),
        batch_pairs=16,
        total_pairs=600,
    )

    assert torch.allclose(penalty_batch_1, penalty_batch_16)


def test_phase1_direct_score_rate_penalty_rejects_invalid_batch_shape() -> None:
    module = _load_trainer_module()

    with pytest.raises(ValueError, match="batch_pairs must be positive"):
        module.contest_rate_penalty_from_batch_bits(
            torch.tensor(1.0),
            batch_pairs=0,
            total_pairs=600,
        )
    with pytest.raises(ValueError, match="cannot exceed total_pairs"):
        module.contest_rate_penalty_from_batch_bits(
            torch.tensor(1.0),
            batch_pairs=601,
            total_pairs=600,
        )


def test_phase1_trainer_loads_pr95_parity_profile_contract(tmp_path: Path) -> None:
    module = _load_trainer_module()
    profile = tmp_path / "profile_pr95_hnerv_muon_intake.json"
    profile.write_text(
        json.dumps(
            {
                "schema": "pr95_hnerv_muon_static_intake_profile_v1",
                "trainer_parity_contract": {
                    "schema": module.PR95_TRAINER_PARITY_SCHEMA,
                    "source_tree_sha256": "abc123",
                    "stage_schedule_digest": "def456",
                    "stage_schedule": [{"order": 1, "name": "stage1_v328_ce"}],
                    "t1_trainer_config": {
                        "required_flags": {
                            "--enable-scorer-domain-loss": True,
                            "--yuv6-mode": "monkey_patch_global",
                        }
                    },
                    "preflight_contract": {
                        "local_trainer_parity_preflight_passed": True,
                        "source_stage_count": 8,
                        "stage_order_matches_release_view": True,
                        "ready_for_score_bearing_t1_hnerv_parity_dispatch": False,
                        "score_bearing_dispatch_blockers": [
                            "active_dispatch_claim_and_provider_job_not_started_no_gpu_or_remote_per_scope"
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = module.load_pr95_trainer_parity_contract(profile)

    assert loaded["status"] == "loaded"
    assert loaded["score_claim"] is False
    assert loaded["contract_schema_matches_expected"] is True
    assert loaded["local_trainer_parity_preflight_passed"] is True
    assert loaded["source_stage_count"] == 8
    assert loaded["t1_trainer_config"]["required_flags"]["--yuv6-mode"] == "monkey_patch_global"


def test_phase1_trainer_default_pr95_parity_profile_is_tracked_and_loadable() -> None:
    module = _load_trainer_module()

    assert ".omx/research" in module.DEFAULT_PR95_PARITY_PROFILE.as_posix()
    assert module.DEFAULT_PR95_PARITY_PROFILE.is_file()

    loaded = module.load_pr95_trainer_parity_contract(module.DEFAULT_PR95_PARITY_PROFILE)

    assert loaded["status"] == "loaded"
    assert loaded["contract_schema_matches_expected"] is True
    assert loaded["local_trainer_parity_preflight_passed"] is True
    assert loaded["source_stage_count"] == 8


def test_phase1_ema_proxy_eval_is_chunked_to_avoid_full_table_decoder_oom() -> None:
    module = _load_trainer_module()

    class _NoopEMA:
        shadow: dict[str, torch.Tensor] = {}

        def apply(self, target: torch.nn.Module) -> None:
            del target

    class _ChunkAssertingBalle(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[int] = []

        def forward(self, y: torch.Tensor) -> dict[str, torch.Tensor]:
            self.calls.append(int(y.shape[0]))
            return {
                "y_hat": y,
                "rate_total_bits": y.new_tensor(float(y.shape[0] * 7)),
            }

    class _ChunkAssertingDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[int] = []

        def forward(self, y_hat: torch.Tensor) -> torch.Tensor:
            batch = int(y_hat.shape[0])
            self.calls.append(batch)
            if batch > 2:
                raise AssertionError(f"proxy eval decoded unchunked batch {batch}")
            return y_hat[:, :1].view(batch, 1, 1, 1, 1).expand(batch, 2, 3, 1, 1)

    decoder = _ChunkAssertingDecoder()
    balle = _ChunkAssertingBalle()

    metrics = module._eval_ema_proxy(
        decoder=decoder,
        balle=balle,
        ema_decoder=_NoopEMA(),
        ema_balle=_NoopEMA(),
        latents=torch.zeros(5, 2),
        target_pixels=torch.zeros(5, 2, 3, 1, 1),
        noise_std=0.0,
        enable_eval_roundtrip_in_training=True,
        eval_batch_size=2,
    )

    assert decoder.calls == [2, 2, 1]
    assert balle.calls == [2, 2, 1]
    assert metrics["ema_proxy_pixel_l1"] == 0.0
    assert metrics["ema_proxy_rate_bits"] == 35.0


def test_phase1_trainer_missing_pr95_parity_profile_is_non_promotable(
    tmp_path: Path,
) -> None:
    module = _load_trainer_module()

    loaded = module.load_pr95_trainer_parity_contract(tmp_path / "missing.json")

    assert loaded["status"] == "missing"
    assert loaded["score_claim"] is False
    assert loaded["local_trainer_parity_preflight_passed"] is False
    assert loaded["ready_for_score_bearing_t1_hnerv_parity_dispatch"] is False
    assert loaded["score_bearing_dispatch_blockers"] == [
        "pr95_parity_profile_missing_run_experiments_profile_pr95_hnerv_muon_intake"
    ]


def test_phase1_trainer_auth_eval_cli_exits_nonzero_for_smoke_scaffold(
    tmp_path: Path,
) -> None:
    trainer_path = Path("experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py")
    result = subprocess.run(
        [
            sys.executable,
            str(trainer_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--smoke",
            "--allow-missing-canonical-a1",
            "--auth-eval",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "--auth-eval refused" in combined
    assert "scaffold-only" in combined
    assert not (tmp_path / "out" / "contest_auth_eval.json").exists()


def test_phase1_trainer_auth_eval_refuses_even_with_scorer_domain_loss(
    tmp_path: Path,
) -> None:
    trainer_path = Path("experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py")
    result = subprocess.run(
        [
            sys.executable,
            str(trainer_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--smoke",
            "--allow-missing-canonical-a1",
            "--enable-scorer-domain-loss",
            "--auth-eval",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "--auth-eval refused" in combined
    assert "scaffold-only" in combined
    assert not (tmp_path / "out" / "contest_auth_eval.json").exists()


def test_phase1_trainer_scorer_domain_refuses_disabled_yuv6(
    tmp_path: Path,
) -> None:
    trainer_path = Path("experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py")
    result = subprocess.run(
        [
            sys.executable,
            str(trainer_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--smoke",
            "--allow-missing-canonical-a1",
            "--enable-scorer-domain-loss",
            "--disable-differentiable-yuv6",
            "--no-auth-eval",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "--enable-scorer-domain-loss requires --enable-differentiable-yuv6" in combined
    assert not (tmp_path / "out").exists()


def test_phase1_trainer_scorer_domain_refuses_zero_non_smoke_epochs(
    tmp_path: Path,
) -> None:
    trainer_path = Path("experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py")
    result = subprocess.run(
        [
            sys.executable,
            str(trainer_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--enable-scorer-domain-loss",
            "--epochs",
            "0",
            "--no-auth-eval",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "--enable-scorer-domain-loss requires --epochs > 0" in combined
    assert not (tmp_path / "out").exists()


def test_phase1_trainer_scorer_domain_refuses_zero_smoke_epochs(
    tmp_path: Path,
) -> None:
    trainer_path = Path("experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py")
    result = subprocess.run(
        [
            sys.executable,
            str(trainer_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--smoke",
            "--allow-missing-canonical-a1",
            "--enable-scorer-domain-loss",
            "--epochs",
            "0",
            "--no-auth-eval",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "--enable-scorer-domain-loss requires --epochs > 0" in combined
    assert not (tmp_path / "out").exists()


def test_phase1_trainer_non_smoke_cli_exits_nonzero_before_training(
    tmp_path: Path,
) -> None:
    trainer_path = Path("experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py")
    result = subprocess.run(
        [
            sys.executable,
            str(trainer_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--no-auth-eval",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "non-smoke training refused" in combined
    assert "auth_eval_custody_not_wired" in combined
    assert not (tmp_path / "out" / "archive.zip").exists()


def test_phase1_trainer_maybe_run_auth_eval_direct_refuses_scaffold(
    tmp_path: Path,
) -> None:
    module = _load_trainer_module()
    with pytest.raises(SystemExit, match="--auth-eval refused"):
        module.maybe_run_auth_eval(
            archive_path=tmp_path / "archive.zip",
            submission_dir=tmp_path / "submission",
            output_dir=tmp_path / "out",
            enabled=True,
            dispatch_lane_id="lane_t1_phase1",  # FAKE_LANE_OK: auth-eval fixture
            dispatch_claims_path=tmp_path / "claims.md",
        )


def test_phase1_trainer_maybe_run_auth_eval_disabled_still_noops(tmp_path: Path) -> None:
    module = _load_trainer_module()
    assert (
        module.maybe_run_auth_eval(
            archive_path=tmp_path / "archive.zip",
            submission_dir=tmp_path / "submission",
            output_dir=tmp_path / "out",
            enabled=False,
            dispatch_lane_id=None,
            dispatch_claims_path=tmp_path / "claims.md",
        )
        is None
    )
