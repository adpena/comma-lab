# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import torch

from tac.repo_io import write_json
from tac.substrates.pact_nerv_ia3.architecture import (
    PactNervIa3Config,
    PactNervIa3Substrate,
)
from tac.substrates.pact_nerv_ia3.archive_candidate import (
    PACT_NERV_IA3_BYTE_CLOSED_CANDIDATE_SCHEMA,
    PACT_NERV_IA3_RECEIVER_INFLATE_PROOF_SCHEMA,
    materialize_pact_nerv_ia3_byte_closed_candidate,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _cfg() -> PactNervIa3Config:
    return PactNervIa3Config(
        latent_dim=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        pose_dim=6,
        ia3_init_delta_std=0.01,
        num_pairs=3,
        output_height=24,
        output_width=32,
    )


def test_pact_nerv_ia3_byte_closed_candidate_emits_receiver_proof(tmp_path: Path) -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    pt_path = tmp_path / "ia3.pt"
    torch.save(PactNervIa3Substrate(cfg).state_dict(), pt_path)
    report_path = tmp_path / "parity.json"
    write_json(
        report_path,
        {
            "schema": "pact_nerv_ia3_mlx_pytorch_forward_parity.v1",
            "config": {
                "latent_dim": cfg.latent_dim,
                "embed_dim": cfg.embed_dim,
                "initial_grid_h": cfg.initial_grid_h,
                "initial_grid_w": cfg.initial_grid_w,
                "decoder_channels": list(cfg.decoder_channels),
                "sin_frequency": cfg.sin_frequency,
                "num_upsample_blocks": cfg.num_upsample_blocks,
                "pose_dim": cfg.pose_dim,
                "ia3_init_delta_std": cfg.ia3_init_delta_std,
                "num_pairs": cfg.num_pairs,
                "output_height": cfg.output_height,
                "output_width": cfg.output_width,
            },
            "score_claim": False,
        },
    )

    manifest = materialize_pact_nerv_ia3_byte_closed_candidate(
        pytorch_state_dict_path=pt_path,
        parity_report_path=report_path,
        output_dir=tmp_path / "candidate",
        repo_root=REPO_ROOT,
        label="unit",
        overwrite=False,
    )

    proof_ref = Path(manifest["receiver_verification"]["proof_path"])
    proof_path = proof_ref if proof_ref.is_absolute() else REPO_ROOT / proof_ref
    proof = __import__("json").loads(proof_path.read_text(encoding="utf-8"))
    archive_ref = Path(manifest["candidate_archive_path"])
    archive_path = archive_ref if archive_ref.is_absolute() else REPO_ROOT / archive_ref

    assert manifest["schema"] == PACT_NERV_IA3_BYTE_CLOSED_CANDIDATE_SCHEMA
    assert manifest["byte_closed_candidate_emitted"] is True
    assert manifest["receiver_contract_satisfied"] is True
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert archive_path.is_file()
    assert proof["schema"] == PACT_NERV_IA3_RECEIVER_INFLATE_PROOF_SCHEMA
    assert proof["passed"] is True
    assert proof["runtime_consumption_proof_passed"] is True
    assert proof["receiver_frame_count"] == cfg.num_pairs * 2
