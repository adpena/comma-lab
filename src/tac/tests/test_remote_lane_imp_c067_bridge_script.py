from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "remote_lane_imp_c067_bridge.sh"


def _text() -> str:
    return SCRIPT.read_text()


def test_imp_c067_remote_script_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_imp_c067_remote_script_uses_remote_conventions() -> None:
    text = _text()
    assert "set -euo pipefail" in text
    assert 'source "$WORKSPACE/env.sh"' in text
    assert 'PYBIN="${PYBIN:-/opt/conda/bin/python}"' in text
    assert 'OUTPUT_DIR="${OUTPUT_DIR:-experiments/results/${RUN_ID}}"' in text
    assert 'HEARTBEAT="$OUT_DIR/heartbeat.log"' in text
    assert "launcher_provenance.json" in text
    assert "byte_screen_classification.json" in text
    assert "zipfile.ZipFile" in text


def test_imp_c067_remote_script_preflights_nvdec_before_builder() -> None:
    text = _text()
    assert "=== Stage 0: NVDEC and source preflight ===" in text
    assert 'bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali' in text
    assert "FATAL: NVDEC probe failed" in text
    assert "IMP_C067_REQUIRE_NVDEC" in text
    probe_call = text.index('bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali')
    builder_call = text.index('"$PYBIN" -u experiments/build_imp_c067_bridge_candidates.py')
    assert probe_call < builder_call


def test_imp_c067_remote_script_pins_current_c067_source_and_rejects_old_anchors() -> None:
    text = _text()
    assert (
        "experiments/results/c067_breakthrough_candidate_matrix_20260502T1030Z/"
        "line_search_source_c067_fixedslice/archive.zip"
    ) in text
    assert "226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a" in text
    assert "276214" in text
    assert "refusing old Lane G/ASYM source anchor" in text
    assert "IMP_C067_ALLOW_SOURCE_OVERRIDE" in text
    assert "IMP_C067_ALLOW_OLD_ANCHOR" in text


def test_imp_c067_remote_script_uses_real_builder_cli_flags_only() -> None:
    text = _text()
    for token in (
        "experiments/build_imp_c067_bridge_candidates.py",
        "--source-archive",
        "--output-dir",
        "--cycle-counts",
        "--sparsity-increment",
        "--qzs3-block-sizes",
        "--payload-member-name",
        "--payload-format",
        "--brotli-quality",
        "--force",
    ):
        assert token in text
    assert "--archive" not in text
    builder_block = text.split("BUILD_CMD=(", 1)[1].split(")", 1)[0]
    assert "--device" not in builder_block
    assert "-u experiments/contest_auth_eval.py" not in text


def test_imp_c067_remote_script_classifies_outputs_as_non_score_byte_screen() -> None:
    text = _text()
    for token in (
        "EMPIRICAL BYTE-SCREEN ONLY - NON-SCORE",
        "score_claim",
        "promotion_eligible",
        "empirical_byte_screen_non_score",
        "exact_cuda_eval_later_required_for_any_score_claim",
        "safe_to_promote_rank_kill_from_this_run",
        "archive.zip -> inflate.sh -> upstream/evaluate.py via experiments/contest_auth_eval.py --device cuda",
    ):
        assert token in text
