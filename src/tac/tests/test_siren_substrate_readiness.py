from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.substrates.siren_readiness import audit_siren_substrate_readiness

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "audit_siren_substrate_readiness.py"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_minimal_ready_repo(repo: Path) -> None:
    _write(
        repo / "experiments/train_substrate_siren.py",
        """
from tac.substrates._shared.trainer_skeleton import decode_real_pairs
CONTEST_AUTH_EVAL_SCRIPT = "experiments/contest_auth_eval.py"
TIER_1_OPERATOR_REQUIRED_FLAGS = {
    "--video-path": {}, "--output-dir": {}, "--epochs": {},
    "--batch-size": {}, "--upstream-dir": {}, "--device": {},
}
def _smoke_main(args): pass
SMOKE_FLAG = "--smoke"
def _full_main(args):
    from tac.substrates.siren.archive import pack_archive
    pack_archive()
    return {
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "auth": "contest_auth_eval_cuda.json",
    }
def _write_runtime(path): print("inflate.sh")
def _build_archive_zip(path): print("archive.zip")
""",
    )
    _write(
        repo / ".omx/operator_authorize_recipes/substrate_siren_modal_a100_dispatch.yaml",
        """
schema_version: 1
lane_id: lane_substrate_siren_20260512
target_modes:
  - contest_one_video_replay
  - research_substrate
remote_driver: scripts/remote_lane_substrate_siren.sh
readiness_gate: tools/audit_siren_substrate_readiness.py --fail-if-not-ready
required_input_files:
  - flag: --video-path
    default_path: upstream/videos/0.mkv
required_input_files_trainer: experiments/train_substrate_siren.py
""",
    )
    _write(repo / "src/tac/substrates/siren/architecture.py", "class SirenSubstrate: pass\n")
    _write(
        repo / "src/tac/substrates/siren/archive.py",
        """
SRV1_MAGIC = b"SRV1"
SRV1_HEADER_FMT = "<4sBHHBHHII"
SRV1_HEADER_SIZE = 22
def pack_archive(): pass
def parse_archive(): pass
# monolithic single-file ``0.bin``
""",
    )
    _write(
        repo / "src/tac/substrates/siren/inflate.py",
        """
import zlib
def _write_png_rgb(path, rgb_uint8):
    return zlib.compress(b"x")
def inflate_one_video(): pass
def main_cli(): pass
parse_archive = object()
SirenSubstrate = object
""",
    )
    _write(
        repo / "src/tac/substrates/siren/score_aware_loss.py",
        """
def score_pair_components(): pass
def loss(apply_eval_roundtrip=True):
    if not apply_eval_roundtrip:
        raise ValueError("apply_eval_roundtrip=False is forbidden")
    alpha_rate = beta_seg = gamma_pose = 1
""",
    )
    _write(repo / "src/tac/substrates/siren/tests/test_siren_roundtrip.py", "def test_x(): pass\n")
    _write(
        repo / "src/tac/substrates/siren/tests/test_score_aware_loss_real_scorer_forward.py",
        "def test_x(): pass\n",
    )


def test_readiness_fails_closed_when_trainer_missing(tmp_path: Path) -> None:
    _write_minimal_ready_repo(tmp_path)
    (tmp_path / "experiments/train_substrate_siren.py").unlink()

    payload = audit_siren_substrate_readiness(tmp_path)

    assert payload["local_contract_ready"] is False
    assert payload["ready_for_remote_dispatch"] is False
    assert payload["score_claim"] is False
    assert any(blocker.startswith("missing_trainer:") for blocker in payload["local_blockers"])


def test_readiness_fails_closed_when_recipe_omits_target_modes(tmp_path: Path) -> None:
    _write_minimal_ready_repo(tmp_path)
    recipe = tmp_path / ".omx/operator_authorize_recipes/substrate_siren_modal_a100_dispatch.yaml"
    recipe.write_text(recipe.read_text(encoding="utf-8").replace("target_modes:", "old_modes:"), encoding="utf-8")

    payload = audit_siren_substrate_readiness(tmp_path)

    assert payload["local_contract_ready"] is False
    assert "recipe_target_modes_missing_or_invalid" in payload["local_blockers"]
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_readiness_accepts_minimal_complete_local_surfaces(tmp_path: Path) -> None:
    _write_minimal_ready_repo(tmp_path)

    payload = audit_siren_substrate_readiness(tmp_path)

    assert payload["local_contract_ready"] is True
    assert payload["ready_for_first_anchor_training"] is True
    assert payload["ready_for_remote_dispatch"] is False
    assert payload["promotion_eligible"] is False
    assert payload["manifest_hash"].startswith("sha256:")


def test_live_cli_json_is_fail_closed_but_locally_ready() -> None:
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--json", "--fail-if-not-ready"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["lane_id"] == "lane_substrate_siren_20260512"
    assert payload["local_contract_ready"] is True
    assert payload["ready_for_remote_dispatch"] is False
    assert "active_lane_dispatch_claim_required_before_gpu_spend" in payload["dispatch_blockers"]
    assert payload["score_claim"] is False
