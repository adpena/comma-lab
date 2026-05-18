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
    lane_id = "lane_z6_v2_candidate_1_wave_2_build_trainer_extension_and_recipe_20260517"
    env = os.environ.copy()
    env.update(
        {
            "WORKSPACE": str(workspace),
            "PYBIN": sys.executable,
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
            "Z6_EGO_SOURCE": "posenet_projection",
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
        "--predictor-architecture": "multi_layer_film_depth_3_300k",
        "--predictor-param-count-target": "300000",
        "--ego-source": "posenet_projection",
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
    assert provenance["emit_identity_predictor_disambiguator_archive"] == "true"
