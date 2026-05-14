# SPDX-License-Identifier: MIT
from pathlib import Path

from tac.preflight import check_no_unconditional_ensurepip


REPO_ROOT = Path(__file__).resolve().parents[3]
REMOTE_POSE_TTO = REPO_ROOT / "scripts" / "remote_pose_tto_bootstrap.sh"
REMOTE_POSE_TTO_ONLY = REPO_ROOT / "scripts" / "remote_pose_tto_only_bootstrap.sh"
DALI_BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_dali_hash_pinned.py"


def test_remote_bootstraps_guard_ensurepip_upgrade() -> None:
    assert check_no_unconditional_ensurepip(strict=False) == []


def test_remote_pose_tto_bootstrap_guarded_ensurepip() -> None:
    text = REMOTE_POSE_TTO.read_text()
    ensurepip_index = text.index("ensurepip --upgrade")
    guard_window = text[max(0, ensurepip_index - 260):ensurepip_index]
    assert "import pip" in guard_window
    assert "if !" in guard_window


def test_remote_pose_tto_only_bootstrap_guarded_ensurepip() -> None:
    text = REMOTE_POSE_TTO_ONLY.read_text()
    ensurepip_index = text.index("ensurepip --upgrade")
    guard_window = text[max(0, ensurepip_index - 260):ensurepip_index]
    assert "import pip" in guard_window
    assert "if !" in guard_window


def test_dali_hash_bootstrap_records_installer_availability_action() -> None:
    text = DALI_BOOTSTRAP.read_text()
    assert "ensure_remote_uv.sh" in text
    assert "installer_bootstrap_action" in text
    assert "uv_bootstrapped_by_ensure_remote_uv" in text
    assert "pip_bootstrapped_by_guarded_ensurepip" in text
    assert "[sys.executable, \"-c\", \"import pip\"]" in text
    assert "[sys.executable, \"-m\", \"ensurepip\", \"--upgrade\"]" in text
