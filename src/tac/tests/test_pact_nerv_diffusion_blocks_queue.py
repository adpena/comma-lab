from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from comma_lab.scheduler.pact_nerv_diffusion_blocks_queue import (
    PACT_NERV_DIFFUSION_BLOCKS_QUEUE_SCHEMA,
    PACT_NERV_DIFFUSION_BLOCKS_SCHEDULE_SCHEMA,
    PACT_NERV_DIFFUSION_DISTILLED_SMOKE_SCHEMA,
    PACT_NERV_IA3_MLX_SMOKE_SCHEMA,
    build_pact_nerv_diffusion_blocks_mlx_queue,
    build_pact_nerv_diffusion_blocks_schedule,
)
from tac.substrates.pact_nerv_ia3.archive_candidate import (
    PACT_NERV_IA3_BYTE_CLOSED_CANDIDATE_SCHEMA,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_diffusion_blocks_schedule_is_equal_mass_and_fail_closed() -> None:
    schedule = build_pact_nerv_diffusion_blocks_schedule(
        block_count=3,
        difficulty_mass_source="scorer_region_waterfill",
        overlap_fraction=0.1,
    )

    assert schedule["schema"] == PACT_NERV_DIFFUSION_BLOCKS_SCHEDULE_SCHEMA
    assert schedule["paper_basis"] == "arxiv:2506.14202v3"
    assert schedule["score_claim"] is False
    assert schedule["ready_for_exact_eval_dispatch"] is False
    assert [row["difficulty_mass_interval"] for row in schedule["blocks"]] == [
        [0.0, 1 / 3],
        [1 / 3, 2 / 3],
        [2 / 3, 1.0],
    ]
    assert schedule["blocks"][0]["local_training_axis"] == "[macOS-MLX research-signal]"
    assert "region" in schedule["optimization_functional"]["interaction_axes"]
    assert "before_entropy_coder" in schedule["optimization_functional"]["entropy_positions"]
    assert schedule["composition_constraints"]["receiver_runtime"] == "deterministic_student_only"
    assert "numpy_or_pytorch_export_forward_parity" in schedule["proof_obligations"]


def test_diffusion_blocks_queue_runs_local_probes_without_authority(tmp_path: Path) -> None:
    queue = build_pact_nerv_diffusion_blocks_mlx_queue(
        repo_root=tmp_path,
        queue_id="dbq",
        output_root=tmp_path / "out",
        source_video_path=tmp_path / "0.mkv",
        block_count=2,
        max_pairs=4,
        difficulty_mass_source="master_gradient",
    )

    assert queue["schema"] == "experiment_queue.v1"
    assert queue["metadata"]["schema"] == PACT_NERV_DIFFUSION_BLOCKS_QUEUE_SCHEMA
    assert queue["metadata"]["axis"] == "[macOS-MLX research-signal]"
    assert queue["metadata"]["score_claim"] is False
    assert queue["metadata"]["ready_for_exact_eval_dispatch"] is False
    steps = queue["experiments"][0]["steps"]
    assert [step["id"] for step in steps] == [
        "emit_diffusion_blocks_schedule",
        "run_pact_nerv_diffusion_distilled_smoke",
        "run_pact_nerv_ia3_mlx_renderer_smoke",
        "plan_pr95_mlx_blockwise_control",
        "prove_pact_nerv_ia3_mlx_pytorch_forward_parity",
        "materialize_pact_nerv_ia3_byte_closed_candidate",
        "emit_mlx_local_deterministic_replay_bundle",
    ]
    assert steps[1]["requires"] == ["emit_diffusion_blocks_schedule"]
    assert steps[1]["resources"]["kind"] == "local_cpu"
    assert "experiments/train_substrate_pact_nerv_diffusion_distilled.py" in steps[1]["command"]
    assert steps[1]["postconditions"][0]["equals"] == PACT_NERV_DIFFUSION_DISTILLED_SMOKE_SCHEMA
    assert steps[2]["requires"] == ["emit_diffusion_blocks_schedule"]
    assert steps[2]["resources"]["kind"] == "local_mlx"
    assert "experiments/train_substrate_pact_nerv_ia3_mlx_local.py" in steps[2]["command"]
    assert steps[2]["postconditions"][0]["key"] == "schema_version"
    assert steps[2]["postconditions"][0]["equals"] == PACT_NERV_IA3_MLX_SMOKE_SCHEMA
    assert "--smoke-mode" in steps[3]["command"]
    assert "--execute-smoke" in steps[3]["command"]
    assert steps[3]["resources"]["kind"] == "local_mlx"
    assert steps[3]["postconditions"][0]["equals"] == "pr95_mlx_long_training_plan.v1"
    assert steps[4]["requires"] == ["run_pact_nerv_ia3_mlx_renderer_smoke"]
    assert "tools/prove_pact_nerv_ia3_mlx_pytorch_forward_parity.py" in steps[4]["command"]
    assert steps[4]["postconditions"][0]["type"] == "json_equals"
    assert steps[4]["postconditions"][1]["key"] == "parity_passed"
    assert steps[4]["postconditions"][1]["equals"] is True
    assert steps[5]["requires"] == ["prove_pact_nerv_ia3_mlx_pytorch_forward_parity"]
    assert "tools/materialize_pact_nerv_ia3_byte_closed_candidate.py" in steps[5]["command"]
    assert steps[5]["postconditions"][0]["required_equals"] == {"schema": PACT_NERV_IA3_BYTE_CLOSED_CANDIDATE_SCHEMA}
    assert steps[5]["postconditions"][0]["required_true"] == [
        "byte_closed_candidate_emitted",
        "receiver_contract_satisfied",
        "full_frame_inflate_parity_satisfied",
        "runtime_adapter_ready",
        "candidate_runtime_adapter_blocker_cleared",
    ]
    assert "runtime_consumption_proof_sha256" in steps[5]["postconditions"][0]["required_nonempty"]
    assert steps[5]["postconditions"][3]["path"].endswith("receiver_inflate_proof.json")
    assert steps[5]["resources"]["kind"] == "local_cpu"
    assert steps[6]["requires"] == [
        "run_pact_nerv_diffusion_distilled_smoke",
        "run_pact_nerv_ia3_mlx_renderer_smoke",
        "plan_pr95_mlx_blockwise_control",
        "prove_pact_nerv_ia3_mlx_pytorch_forward_parity",
        "materialize_pact_nerv_ia3_byte_closed_candidate",
    ]
    assert "tools/build_mlx_local_replay_bundle.py" in steps[6]["command"]
    assert "--command-json" in steps[6]["command"]
    assert steps[6]["postconditions"][0]["type"] == "json_equals"
    assert steps[6]["postconditions"][2]["key"] == "replay_readiness.byte_closed_receiver_proof_present"
    assert steps[6]["postconditions"][2]["equals"] is True
    assert steps[6]["postconditions"][3]["key"] == "replay_readiness.runtime_custody_present"
    assert steps[6]["postconditions"][3]["equals"] is True
    assert steps[6]["resources"]["kind"] == "local_cpu"


def test_diffusion_blocks_queue_cli(tmp_path: Path) -> None:
    queue_out = tmp_path / "queue.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_pact_nerv_diffusion_blocks_queue.py"),
            "--queue-out",
            str(queue_out),
            "--queue-id",
            "dbq",
            "--output-root",
            str(tmp_path / "out"),
            "--block-count",
            "2",
            "--max-pairs",
            "4",
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    stdout = json.loads(result.stdout)
    queue = json.loads(queue_out.read_text(encoding="utf-8"))

    assert stdout["schema"] == "pact_nerv_diffusion_blocks_queue_cli_result.v1"
    assert stdout["score_claim"] is False
    assert queue["metadata"]["schema"] == PACT_NERV_DIFFUSION_BLOCKS_QUEUE_SCHEMA


def test_diffusion_blocks_schedule_cli(tmp_path: Path) -> None:
    output = tmp_path / "schedule.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_pact_nerv_diffusion_blocks_schedule.py"),
            "--output",
            str(output),
            "--block-count",
            "2",
            "--difficulty-mass-source",
            "residual_mass",
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    stdout = json.loads(result.stdout)
    schedule = json.loads(output.read_text(encoding="utf-8"))

    assert stdout["schema"] == "pact_nerv_diffusion_blocks_schedule_cli_result.v1"
    assert schedule["schema"] == PACT_NERV_DIFFUSION_BLOCKS_SCHEDULE_SCHEMA
    assert schedule["difficulty_mass_source"] == "residual_mass"
