# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.analysis.compact_vq_pivot_audit import (
    COMPACT_VQ_MISMATCH_STATUS,
    build_compact_vq_pivot_audit,
)

REPO = Path(__file__).resolve().parents[3]


def test_compact_vq_pivot_audit_demotes_per_pair_latent_vq(
    tmp_path: Path,
) -> None:
    upstream = _fake_upstream(tmp_path / "upstream")
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "family": "pact_nerv_vq",
                "max_pairs": 600,
                "scorer_batch_pairs": 1,
                "scope_status": {"full_video": "executed"},
                "score_components": {"canonical_score": 90.0},
                "section_value_rows": [],
            }
        ),
        encoding="utf-8",
    )

    audit = build_compact_vq_pivot_audit(
        repo_root=REPO,
        upstream_dir=upstream,
        mlx_profile_paths=[profile],
    )

    assert audit["schema"] == "compact_vq_pivot_audit.v1"
    assert audit["verdict"] == COMPACT_VQ_MISMATCH_STATUS
    assert audit["scorer_contract"]["score_formula_observed"] is True
    assert audit["scorer_contract"]["segnet_last_frame_only_observed"] is True
    assert audit["implementation_contract"]["per_pair_single_vector_vq_present"] is True
    assert audit["implementation_contract"]["residual_tokenization_present"] is False
    assert audit["profile_signal"]["best_full_video_mlx_score"] == 90.0
    assert "compact_vq_is_per_pair_latent_not_residual_tokenization" in audit["blockers"]
    assert audit["score_claim"] is False
    assert audit["ready_for_exact_eval_dispatch"] is False


def _fake_upstream(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "frame_utils.py").write_text(
        "seq_len = 2\n"
        "camera_size = (1164, 874)\n"
        "segnet_model_input_size = (512, 384)\n",
        encoding="utf-8",
    )
    (path / "evaluate.py").write_text(
        "import math\n"
        "score = 100 * segnet_dist +  math.sqrt(posenet_dist * 10)  + 25 * rate\n",
        encoding="utf-8",
    )
    (path / "modules.py").write_text(
        "x = x[:, -1, ...]\n"
        "rgb_to_yuv6(x)\n"
        "einops.rearrange(x, 'b (t c) h w -> b t c h w')\n"
        "out1[h.name][..., : h.out // 2]\n",
        encoding="utf-8",
    )
    return path
