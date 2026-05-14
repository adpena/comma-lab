# SPDX-License-Identifier: MIT
from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER = REPO_ROOT / "scripts" / "pfp16_a_plus_plus_exact_t4_eval.sh"
EXPECTED_SHA = "0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f"
EXPECTED_BYTES = "686635"


def test_pfp16_a_plus_plus_helper_is_bash_valid() -> None:
    subprocess.run(["bash", "-n", str(HELPER)], check=True)


def test_pfp16_a_plus_plus_helper_pins_exact_archive_and_cuda_t4() -> None:
    text = HELPER.read_text()

    assert EXPECTED_SHA in text
    assert EXPECTED_BYTES in text
    assert "experiments/results/lane_g_v3_pfp16/archive_lane_g_v3_pfp16.zip" in text
    assert "experiments/contest_auth_eval.py" in text
    assert "--device cuda" in text
    assert 'prov.get("gpu_t4_match") is True' in text
    assert "payload[\"n_samples\"] == 600" in text


def test_pfp16_a_plus_plus_helper_does_not_use_known_unsafe_paths() -> None:
    text = HELPER.read_text()

    assert "modal_auth_eval.py" not in text
    assert "modal_train_lane.py" not in text
    assert "build_lane_g_v3_pfp16_stack.py" not in text
    assert "AUTH_EVAL_DEVICE=cpu" not in text
    assert "--device cpu" not in text
