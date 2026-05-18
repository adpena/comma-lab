# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REMOTE_DRIVER = REPO / "scripts/remote_lane_substrate_time_traveler_l5_z6.sh"


def test_z6_remote_driver_defaults_smoke_to_three_epochs_and_full_to_300() -> None:
    """The first-anchor smoke path must not inherit the full-run epoch default."""

    text = REMOTE_DRIVER.read_text(encoding="utf-8")

    mode_idx = text.index('Z6_TRAINER_MODE="${Z6_TRAINER_MODE:-}"')
    smoke_idx = text.index('SMOKE_ONLY="${SMOKE_ONLY:-}"')
    epochs_idx = text.index('Z6_EPOCHS="${Z6_EPOCHS:-}"')
    assert mode_idx < smoke_idx < epochs_idx
    assert 'case "$Z6_TRAINER_MODE" in' in text
    assert 'full|FULL|Full)' in text
    assert 'SMOKE_ONLY="0"' in text
    assert 'elif [ -z "$SMOKE_ONLY" ]; then' in text
    assert 'defaulting to smoke' in text
    assert smoke_idx < epochs_idx
    assert 'if [ "$SMOKE_ONLY" = "1" ]; then\n        Z6_EPOCHS="3"' in text
    assert 'else\n        Z6_EPOCHS="300"\n    fi' in text
    assert 'Z6_EPOCHS="${Z6_EPOCHS:-300}"' not in text
    assert 'SMOKE_FLAG_ARGS=()' in text
    assert 'if [ "$SMOKE_ONLY" = "1" ]; then\n    SMOKE_FLAG_ARGS+=(--smoke)\nfi' in text


def test_z6_remote_driver_threads_wave2_lane_id_and_required_flags(tmp_path: Path) -> None:
    """Wave 2 recipe env must reach both claim verification and trainer argv."""

    workspace = tmp_path / "workspace"
    (workspace / "tools").mkdir(parents=True)
    (workspace / "scripts").mkdir(parents=True)
    (workspace / "experiments").mkdir(parents=True)

    (workspace / "tools" / "claim_lane_dispatch.py").write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

cmd = sys.argv[1] if len(sys.argv) > 1 else ""
if cmd == "summary":
    print(json.dumps({
        "active": [{
            "lane_id": os.environ["Z6_LANE_ID"],
            "instance_job_id": os.environ["Z6_DISPATCH_INSTANCE_JOB_ID"],
        }]
    }))
elif cmd == "claim":
    claim_args_path = os.environ.get("CLAIM_ARGS_PATH")
    if claim_args_path:
        with open(claim_args_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(sys.argv[1:]) + "\\n")
    print("terminal-claim-ok")
else:
    raise SystemExit(f"unexpected claim helper command: {cmd}")
""",
        encoding="utf-8",
    )

    (workspace / "scripts" / "remote_archive_only_eval.sh").write_text(
        "bootstrap_runtime_deps() { return 0; }\n",
        encoding="utf-8",
    )

    trainer = workspace / "experiments" / "train_substrate_time_traveler_l5_z6.py"
    trainer.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

argv = sys.argv[1:]
out = Path(argv[argv.index("--output-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "argv.json").write_text(json.dumps(argv), encoding="utf-8")
(out / "stats.json").write_text(
    json.dumps({"evidence_grade": "training-artifact-no-score-claim"}),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    claim_args_path = tmp_path / "claim_args.jsonl"
    lane_id = "lane_z6_v2_candidate_1_wave_2_build_trainer_extension_and_recipe_20260517"
    env = os.environ.copy()
    env.update(
        {
            "WORKSPACE": str(workspace),
            "PYBIN": sys.executable,
            "CLAIM_ARGS_PATH": str(claim_args_path),
            "LOG_DIR": str(tmp_path / "logs"),
            "OUTPUT_DIR": str(output_dir),
            "Z6_OUTPUT_DIR": str(output_dir),
            "Z6_DISPATCH_INSTANCE_JOB_ID": "z6-wave2-test-job",
            "Z6_LANE_ID": lane_id,
            "Z6_RECIPE_PATH": ".omx/operator_authorize_recipes/substrate_z6_v2_candidate_1_multi_layer_film_modal_t4_smoke_dispatch.yaml",
            "TAG": "substrate_z6_v2_candidate_1_multi_layer_film",
            "Z6_TRAINER_MODE": "full",
            "SMOKE_ONLY": "1",
            "Z6_EPOCHS": "100",
            "Z6_PREDICTOR_ARCHITECTURE": "multi_layer_film_depth_3_300k",
            "Z6_PREDICTOR_PARAM_COUNT_TARGET": "300000",
            "Z6_PREDICTOR_HIDDEN_DIM": "72",
            "Z6_PREDICTOR_FILM_MLP_HIDDEN_DIM": "40",
            "Z6_EGO_SOURCE": "scorer_logit",
            "Z6_ENABLE_PAIRED_CONTROL_INITIALIZATION": "shared_modules_seed_order_matched_v2",
            "Z6_EMIT_IDENTITY_PREDICTOR_DISAMBIGUATOR_ARCHIVE": "true",
            "Z6_PAIRED_CONTROL_DISAMBIGUATOR_DECISION_CRITERION_DELTA_S": "0.005",
        }
    )

    result = subprocess.run(
        ["bash", str(REMOTE_DRIVER)],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
        timeout=30,
    )

    assert result.returncode == 0, (
        "Z6 remote lane script failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    argv = json.loads((output_dir / "argv.json").read_text(encoding="utf-8"))
    assert "--smoke" not in argv
    expected_pairs = {
        "--predictor-hidden-dim": "72",
        "--predictor-film-mlp-hidden-dim": "40",
        "--predictor-architecture": "multi_layer_film_depth_3_300k",
        "--predictor-param-count-target": "300000",
        "--ego-source": "scorer_logit",
        "--enable-paired-control-initialization": "shared_modules_seed_order_matched_v2",
        "--paired-control-disambiguator-decision-criterion-delta-s": "0.005",
    }
    for flag, value in expected_pairs.items():
        assert flag in argv
        assert argv[argv.index(flag) + 1] == value
    assert "--emit-identity-predictor-disambiguator-archive" in argv

    provenance = json.loads((tmp_path / "logs" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["lane_id"] == lane_id
    assert provenance["recipe"].endswith(
        "substrate_z6_v2_candidate_1_multi_layer_film_modal_t4_smoke_dispatch.yaml"
    )
    assert provenance["smoke_only"] == "0"
    assert provenance["predictor_architecture"] == "multi_layer_film_depth_3_300k"
    assert provenance["predictor_hidden_dim"] == "72"
    assert provenance["predictor_film_mlp_hidden_dim"] == "40"
    assert provenance["ego_source"] == "scorer_logit"
    assert provenance["emit_identity_predictor_disambiguator_archive"] == "true"
    claim_rows = [
        json.loads(line)
        for line in claim_args_path.read_text(encoding="utf-8").splitlines()
    ]
    assert claim_rows
    terminal_args = claim_rows[-1]
    assert terminal_args[terminal_args.index("--status") + 1] == (
        "completed_z6_pcwm_remote_driver_no_score_claim"
    )
    notes = terminal_args[terminal_args.index("--notes") + 1]
    assert "evidence_marker=[training-artifact-no-score-claim]" in notes
    assert "score_claim=false" in notes


def test_z6_remote_driver_refuses_missing_stats_completion_marker(
    tmp_path: Path,
) -> None:
    """A zero-rc trainer without stats.json is a failed harvest, not DONE."""

    workspace = tmp_path / "workspace"
    (workspace / "tools").mkdir(parents=True)
    (workspace / "scripts").mkdir(parents=True)
    (workspace / "experiments").mkdir(parents=True)

    (workspace / "tools" / "claim_lane_dispatch.py").write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

cmd = sys.argv[1] if len(sys.argv) > 1 else ""
if cmd == "summary":
    print(json.dumps({
        "active": [{
            "lane_id": os.environ["Z6_LANE_ID"],
            "instance_job_id": os.environ["Z6_DISPATCH_INSTANCE_JOB_ID"],
        }]
    }))
elif cmd == "claim":
    claim_args_path = os.environ.get("CLAIM_ARGS_PATH")
    if claim_args_path:
        with open(claim_args_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(sys.argv[1:]) + "\\n")
    print("terminal-claim-ok")
else:
    raise SystemExit(f"unexpected claim helper command: {cmd}")
""",
        encoding="utf-8",
    )
    (workspace / "scripts" / "remote_archive_only_eval.sh").write_text(
        "bootstrap_runtime_deps() { return 0; }\n",
        encoding="utf-8",
    )
    trainer = workspace / "experiments" / "train_substrate_time_traveler_l5_z6.py"
    trainer.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

argv = sys.argv[1:]
out = Path(argv[argv.index("--output-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "argv.txt").write_text(" ".join(argv), encoding="utf-8")
raise SystemExit(0)
""",
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    log_dir = tmp_path / "logs"
    claim_args_path = tmp_path / "claim_args_missing_stats.jsonl"
    env = os.environ.copy()
    env.update(
        {
            "WORKSPACE": str(workspace),
            "PYBIN": sys.executable,
            "CLAIM_ARGS_PATH": str(claim_args_path),
            "LOG_DIR": str(log_dir),
            "OUTPUT_DIR": str(output_dir),
            "Z6_OUTPUT_DIR": str(output_dir),
            "Z6_DISPATCH_INSTANCE_JOB_ID": "z6-missing-stats-test-job",
            "Z6_LANE_ID": "lane_z6_missing_stats_completion_marker_test",
            "Z6_TRAINER_MODE": "full",
            "Z6_EPOCHS": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(REMOTE_DRIVER)],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
        timeout=30,
    )

    assert result.returncode == 31, result.stdout + result.stderr
    assert "missing required stats.json" in result.stderr
    completion_log = log_dir / "completion.log"
    if completion_log.exists():
        assert "LANE_Z6_PCWM_DONE" not in completion_log.read_text(
            encoding="utf-8"
        )
    claim_rows = [
        json.loads(line)
        for line in claim_args_path.read_text(encoding="utf-8").splitlines()
    ]
    assert claim_rows
    terminal_args = claim_rows[-1]
    assert terminal_args[terminal_args.index("--status") + 1] == (
        "failed_z6_pcwm_remote_driver_rc_31"
    )
    notes = terminal_args[terminal_args.index("--notes") + 1]
    assert "evidence_marker=[not-yet-classified]" in notes
    assert "score_claim=unknown" in notes


def test_z6_remote_driver_refuses_stale_stats_completion_marker(
    tmp_path: Path,
) -> None:
    """A trainer-written stale stats.json cannot satisfy this invocation."""

    workspace = tmp_path / "workspace"
    (workspace / "tools").mkdir(parents=True)
    (workspace / "scripts").mkdir(parents=True)
    (workspace / "experiments").mkdir(parents=True)

    (workspace / "tools" / "claim_lane_dispatch.py").write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

cmd = sys.argv[1] if len(sys.argv) > 1 else ""
if cmd == "summary":
    print(json.dumps({
        "active": [{
            "lane_id": os.environ["Z6_LANE_ID"],
            "instance_job_id": os.environ["Z6_DISPATCH_INSTANCE_JOB_ID"],
        }]
    }))
elif cmd == "claim":
    claim_args_path = os.environ.get("CLAIM_ARGS_PATH")
    if claim_args_path:
        with open(claim_args_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(sys.argv[1:]) + "\\n")
    print("terminal-claim-ok")
else:
    raise SystemExit(f"unexpected claim helper command: {cmd}")
""",
        encoding="utf-8",
    )
    (workspace / "scripts" / "remote_archive_only_eval.sh").write_text(
        "bootstrap_runtime_deps() { return 0; }\n",
        encoding="utf-8",
    )
    trainer = workspace / "experiments" / "train_substrate_time_traveler_l5_z6.py"
    trainer.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

argv = sys.argv[1:]
out = Path(argv[argv.index("--output-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
stats = out / "stats.json"
stats.write_text(
    json.dumps({"evidence_grade": "stale-training-artifact-no-score-claim"}),
    encoding="utf-8",
)
old_epoch = 946684800
os.utime(stats, (old_epoch, old_epoch))
raise SystemExit(0)
""",
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    log_dir = tmp_path / "logs"
    claim_args_path = tmp_path / "claim_args_stale_stats.jsonl"
    env = os.environ.copy()
    env.update(
        {
            "WORKSPACE": str(workspace),
            "PYBIN": sys.executable,
            "CLAIM_ARGS_PATH": str(claim_args_path),
            "LOG_DIR": str(log_dir),
            "OUTPUT_DIR": str(output_dir),
            "Z6_OUTPUT_DIR": str(output_dir),
            "Z6_DISPATCH_INSTANCE_JOB_ID": "z6-stale-stats-test-job",
            "Z6_LANE_ID": "lane_z6_stale_stats_completion_marker_test",
            "Z6_TRAINER_MODE": "full",
            "Z6_EPOCHS": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(REMOTE_DRIVER)],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
        timeout=30,
    )

    assert result.returncode == 33, result.stdout + result.stderr
    assert "stale stats.json" in result.stderr
    completion_log = log_dir / "completion.log"
    if completion_log.exists():
        assert "LANE_Z6_PCWM_DONE" not in completion_log.read_text(
            encoding="utf-8"
        )
    claim_rows = [
        json.loads(line)
        for line in claim_args_path.read_text(encoding="utf-8").splitlines()
    ]
    assert claim_rows
    terminal_args = claim_rows[-1]
    assert terminal_args[terminal_args.index("--status") + 1] == (
        "failed_z6_pcwm_remote_driver_rc_33"
    )
    notes = terminal_args[terminal_args.index("--notes") + 1]
    assert "evidence_marker=[not-yet-classified]" in notes
    assert "score_claim=unknown" in notes


def test_z6_remote_driver_quarantines_preexisting_stats_before_trainer(
    tmp_path: Path,
) -> None:
    """A previous stats.json is preserved aside and cannot satisfy Stage 5."""

    workspace = tmp_path / "workspace"
    (workspace / "tools").mkdir(parents=True)
    (workspace / "scripts").mkdir(parents=True)
    (workspace / "experiments").mkdir(parents=True)

    (workspace / "tools" / "claim_lane_dispatch.py").write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

cmd = sys.argv[1] if len(sys.argv) > 1 else ""
if cmd == "summary":
    print(json.dumps({
        "active": [{
            "lane_id": os.environ["Z6_LANE_ID"],
            "instance_job_id": os.environ["Z6_DISPATCH_INSTANCE_JOB_ID"],
        }]
    }))
elif cmd == "claim":
    claim_args_path = os.environ.get("CLAIM_ARGS_PATH")
    if claim_args_path:
        with open(claim_args_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(sys.argv[1:]) + "\\n")
    print("terminal-claim-ok")
else:
    raise SystemExit(f"unexpected claim helper command: {cmd}")
""",
        encoding="utf-8",
    )
    (workspace / "scripts" / "remote_archive_only_eval.sh").write_text(
        "bootstrap_runtime_deps() { return 0; }\n",
        encoding="utf-8",
    )
    trainer = workspace / "experiments" / "train_substrate_time_traveler_l5_z6.py"
    trainer.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

argv = sys.argv[1:]
out = Path(argv[argv.index("--output-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "trainer_ran.txt").write_text("trainer exited without stats rewrite", encoding="utf-8")
raise SystemExit(0)
""",
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    preexisting_stats = output_dir / "stats.json"
    preexisting_stats.write_text(
        json.dumps({"evidence_grade": "previous-run-no-score"}),
        encoding="utf-8",
    )
    log_dir = tmp_path / "logs"
    claim_args_path = tmp_path / "claim_args_preexisting_stats.jsonl"
    env = os.environ.copy()
    env.update(
        {
            "WORKSPACE": str(workspace),
            "PYBIN": sys.executable,
            "CLAIM_ARGS_PATH": str(claim_args_path),
            "LOG_DIR": str(log_dir),
            "OUTPUT_DIR": str(output_dir),
            "Z6_OUTPUT_DIR": str(output_dir),
            "Z6_DISPATCH_INSTANCE_JOB_ID": "z6-preexisting-stats-test-job",
            "Z6_LANE_ID": "lane_z6_preexisting_stats_quarantine_test",
            "Z6_TRAINER_MODE": "full",
            "Z6_EPOCHS": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(REMOTE_DRIVER)],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
        timeout=30,
    )

    assert result.returncode == 31, result.stdout + result.stderr
    assert "missing required stats.json" in result.stderr
    assert not preexisting_stats.exists()
    quarantine_dir = log_dir / "stale_stats_quarantine"
    quarantined = list(quarantine_dir.glob("stats.before_z6-preexisting-stats-test-job.*.json"))
    assert len(quarantined) == 1
    assert json.loads(quarantined[0].read_text(encoding="utf-8")) == {
        "evidence_grade": "previous-run-no-score"
    }
    assert "quarantined_preexisting_stats_json" in (
        log_dir / "run.log"
    ).read_text(encoding="utf-8")
    completion_log = log_dir / "completion.log"
    if completion_log.exists():
        assert "LANE_Z6_PCWM_DONE" not in completion_log.read_text(
            encoding="utf-8"
        )
    claim_rows = [
        json.loads(line)
        for line in claim_args_path.read_text(encoding="utf-8").splitlines()
    ]
    assert claim_rows
    terminal_args = claim_rows[-1]
    assert terminal_args[terminal_args.index("--status") + 1] == (
        "failed_z6_pcwm_remote_driver_rc_31"
    )
