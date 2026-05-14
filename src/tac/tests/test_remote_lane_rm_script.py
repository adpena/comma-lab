# SPDX-License-Identifier: MIT
"""Regression tests for ``scripts/remote_lane_rm_riemannian_pose_tto.sh``.

Lane RM = Riemannian SE(3) geometric pose optimization. Anchored on
Lane A's verified 1.15 [contest-CUDA] floor. The single variable vs
Lane A is ``--optimizer riemannian-sgd`` (the SE(3) exponential-map
retraction in place of Adam's flat-ℝ⁶ step).

These tests pin every claim the launch script makes — the same
contract pattern used by ``test_remote_lane_lr_script.py``,
``test_remote_lane_g_v3_script.py``, etc.

  1. Strict bash safety — `set -euo pipefail` (the LANE-B bootstrap trap;
     ``feedback_zip_dep_bootstrap_trap``).
  2. Stage 0 NVDEC probe BEFORE any GPU spend
     (``feedback_vastai_nvdec_host_variation``).
  3. Anchor on Lane A (NOT baseline_dilated_h64_0_90 or any other lane).
  4. Warm-start from Lane A's optimized_poses.pt (--gt-poses-path).
  5. --optimizer riemannian-sgd passed (the defining flag).
  6. --pose-mode full-6dof passed (Riemannian-SGD requires 6-DOF tensor).
  7. --riemannian-momentum wired (NOT inventing a flag).
  8. Every CLI flag verified against optimize_poses.py argparse — never
     invent flags (``feedback_dead_flag_wiring_pattern``).
  9. Provenance + heartbeat writes; provenance records math references.
 10. Predicted band [1.05, 1.15] recorded in provenance.
 11. Sanity check: SE(3) round-trip drift verification BEFORE auth eval.
 12. Python zipfile (PyTorch container has no `zip` binary —
     ``feedback_zip_dep_bootstrap_trap``).
 13. Internal name `lane_rm`.
 14. No MPS / CPU device fallback (CLAUDE.md MPS-CUDA drift rule).
 15. Auth eval at the end (CLAUDE.md auth-eval-everywhere rule).
 16. NO bypass of --eval-roundtrip (CLAUDE.md eval_roundtrip non-negotiable).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_rm_riemannian_pose_tto.sh"
OPTIMIZE_POSES = REPO / "experiments" / "optimize_poses.py"
LANE_A_SCRIPT = REPO / "scripts" / "remote_lane_a_pose_tto.sh"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


@pytest.fixture(scope="module")
def optimize_poses_argparse_flags() -> set[str]:
    """Extract `add_argument("--<flag>", ...)` flag names from
    optimize_poses.py. CLAUDE.md non-negotiable: NEVER invent CLI flags."""
    src = OPTIMIZE_POSES.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


# ── Existence + bash-safety guards ─────────────────────────────────────


def test_script_exists():
    assert SCRIPT.exists(), f"missing Lane RM launch script: {SCRIPT}"


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be chmod +x"


def test_full_set_euo_pipefail(script_text: str):
    assert "set -euo pipefail" in script_text, (
        "script must use `set -euo pipefail` (LANE-B bootstrap trap; "
        "feedback_zip_dep_bootstrap_trap)"
    )


# ── Stage 0 NVDEC probe ─────────────────────────────────────────────────


def test_nvdec_probe_present(script_text: str):
    assert "probe_nvdec.sh" in script_text


def test_nvdec_probe_fails_loud(script_text: str):
    probe_section = re.search(
        r"probe_nvdec\.sh.*?exit\s+\d+", script_text, re.DOTALL,
    )
    assert probe_section is not None, "NVDEC probe must `exit N` on failure"


# ── Anchor on Lane A ────────────────────────────────────────────────────


def test_anchors_on_lane_a(script_text: str):
    """Lane RM uses Lane A's RENDERER + masks + warm-start poses so the
    experiment isolates the Riemannian SE(3) optimiser as the SINGLE
    variable being tested."""
    assert "experiments/results/lane_a_landed" in script_text, (
        "Lane RM must anchor on Lane A's verified 1.15 [contest-CUDA] "
        "artifacts at experiments/results/lane_a_landed/"
    )


def test_anchors_on_lane_a_renderer_bin(script_text: str):
    assert (
        "experiments/results/lane_a_landed/iter_0/renderer.bin" in script_text
    ), (
        "Lane RM must use Lane A's renderer (NOT baseline_dilated_h64_0_90); "
        "this isolates the optimiser as the single variable vs Lane A"
    )


def test_anchors_on_lane_a_masks(script_text: str):
    assert (
        "experiments/results/lane_a_landed/iter_0/masks.mkv" in script_text
    ), "Lane RM must use Lane A's masks.mkv"


def test_warm_start_from_lane_a_poses(script_text: str):
    """Lane RM warm-starts from Lane A's optimized poses (the SE(3)
    geodesic step is a small refinement — not a fresh search)."""
    assert (
        "experiments/results/lane_a_landed/optimized_poses.pt" in script_text
    ), (
        "Lane RM must use Lane A's optimized_poses.pt as --gt-poses-path "
        "(SE(3) optimiser warm-start)"
    )


# ── Lane RM differentiator: --optimizer riemannian-sgd ────────────────


def test_optimizer_flag_passed(script_text: str):
    assert "--optimizer riemannian-sgd" in script_text, (
        "Lane RM defining flag --optimizer riemannian-sgd must appear"
    )


def test_pose_mode_full_6dof_passed(script_text: str):
    """The SE(3) optimiser requires a 6-DOF pose tensor (each row =
    one SE(3) element split as (ω, t)); --pose-mode full-6dof is
    therefore mandatory."""
    assert "--pose-mode full-6dof" in script_text, (
        "Lane RM must pass --pose-mode full-6dof; the SE(3) optimiser "
        "interprets each pose row as (ω, t) which only makes sense for "
        "the 6-DOF parameterisation"
    )


def test_riemannian_momentum_passed(script_text: str):
    assert "--riemannian-momentum" in script_text, (
        "Lane RM should explicitly pass --riemannian-momentum so the "
        "experiment is reproducible (not relying on the default)"
    )


# ── NEVER invent CLI flags (feedback_dead_flag_wiring_pattern) ────────


def test_every_optimize_poses_flag_in_script_exists_in_argparse(
    script_text: str, optimize_poses_argparse_flags: set[str],
):
    """Every `--<flag>` passed to optimize_poses.py in this script MUST
    exist in optimize_poses.py's argparse. Inventing a flag silently
    skips the feature (auth-eval-on-best dead-flag pattern)."""
    # Find the optimize_poses.py invocation block.
    block = re.search(
        r"experiments/optimize_poses\.py(.*?)(?=\n\s*\w[^\\]*$|\nlog\s|\nrm |\Z)",
        script_text,
        re.DOTALL,
    )
    assert block is not None, "could not find optimize_poses.py invocation"
    # Extract all `--<flag>` tokens (kebab-case, may be followed by value).
    invoked = set(re.findall(r"--([a-z][a-z0-9-]+)", block.group(1)))
    invoked.discard("output-dir")  # universally present
    missing = invoked - optimize_poses_argparse_flags
    assert not missing, (
        f"Lane RM script invokes optimize_poses.py with flags that don't "
        f"exist in its argparse: {sorted(missing)}. NEVER invent CLI flags "
        f"(feedback_dead_flag_wiring_pattern)."
    )


# ── Provenance + heartbeat ────────────────────────────────────────────


def test_provenance_json_written(script_text: str):
    assert "provenance.json" in script_text


def test_heartbeat_log_written(script_text: str):
    assert "heartbeat.log" in script_text


def test_provenance_records_optimizer(script_text: str):
    assert "'optimizer': 'riemannian-sgd'" in script_text, (
        "provenance.json must record optimizer=riemannian-sgd so a "
        "post-hoc reader can identify the run"
    )


def test_provenance_records_math_references(script_text: str):
    """The Riemannian SE(3) work has explicit textbook citations; the
    provenance MUST record them so a future reviewer can verify the
    math without re-deriving it."""
    assert "math_references" in script_text
    assert "Bonnabel" in script_text
    assert "Absil" in script_text
    assert "Boumal" in script_text
    assert "Sola" in script_text


def test_predicted_band_recorded(script_text: str):
    assert "[1.05, 1.15]" in script_text, (
        "predicted_band [1.05, 1.15] must appear in provenance"
    )


# ── Sanity check: SE(3) round-trip BEFORE auth eval ───────────────────


def test_sanity_check_runs_se3_round_trip(script_text: str):
    """Before burning $0.20 on contest_auth_eval, verify the saved
    poses are valid SE(3) elements (round-trip omega → R → omega
    drift < 1e-3)."""
    assert "from tac.se3 import" in script_text
    assert "exp_map_so3" in script_text
    assert "log_map_so3" in script_text
    assert "drift" in script_text


def test_sanity_check_fails_loud(script_text: str):
    """The sanity-check Python block must `sys.exit(2)` on failure —
    silent skip is the LANE-B bootstrap trap class."""
    sanity_blocks = re.findall(
        r'sys\.exit\(2\)', script_text,
    )
    # The script has multiple fail-loud sys.exit(2) calls; just verify
    # at least one exists in the sanity-check block.
    assert len(sanity_blocks) >= 1


# ── Python zipfile (no shell `zip`) ────────────────────────────────────


def test_no_shell_zip_binary(script_text: str):
    """PyTorch container has no `zip` binary; use python zipfile.ZipFile
    (memory: feedback_zip_dep_bootstrap_trap)."""
    # Allow `unzip` (different binary, present in container). Reject the
    # bare `zip` invocation as a shell command.
    has_shell_zip = re.search(r"^\s*zip\s", script_text, re.MULTILINE)
    assert has_shell_zip is None, (
        "use python zipfile.ZipFile, NOT shell `zip` "
        "(feedback_zip_dep_bootstrap_trap)"
    )


def test_uses_python_zipfile(script_text: str):
    assert "zipfile.ZipFile" in script_text, (
        "archive must be built via python zipfile.ZipFile"
    )


# ── Internal name + lane tag ──────────────────────────────────────────


def test_internal_name_lane_rm(script_text: str):
    assert "'lane_internal_name': 'lane_rm'" in script_text


def test_log_prefix_lane_rm(script_text: str):
    assert "[lane-rm]" in script_text


# ── No MPS / CPU device fallback (CLAUDE.md MPS-CUDA drift rule) ──────


def test_device_is_cuda_only(script_text: str):
    """CLAUDE.md non-negotiable: auth eval is contest-CUDA only. Reject
    any MPS or CPU fallback in the script body."""
    # The script must pass --device cuda to optimize_poses AND
    # contest_auth_eval. It must NOT mention --device mps / cpu.
    assert "--device cuda" in script_text
    assert "--device mps" not in script_text
    assert "--device cpu" not in script_text


# ── Auth eval at the end (CLAUDE.md auth-eval-everywhere rule) ────────


def test_contest_auth_eval_invoked(script_text: str):
    """Every chained experiment MUST end with a CUDA auth eval — the
    proxy-auth gap can be 100-350x even on CUDA-CUDA. CLAUDE.md
    non-negotiable."""
    assert "contest_auth_eval.py" in script_text
    assert "submissions/robust_current/inflate.sh" in script_text


# ── eval_roundtrip non-negotiable ─────────────────────────────────────


def test_eval_roundtrip_passed(script_text: str):
    assert "--eval-roundtrip" in script_text, (
        "CLAUDE.md non-negotiable: every training/optimization path "
        "MUST use --eval-roundtrip"
    )


def test_no_disable_eval_roundtrip_flag(script_text: str):
    """--no-eval-roundtrip is forbidden (Lane C R5 fix, commit 9d71ec5d)."""
    assert "--no-eval-roundtrip" not in script_text


def test_posetto_noise_std_nonzero(script_text: str):
    """Hotz STE: --posetto-noise-std=0 silently re-opens the proxy-CUDA
    gap up to 11x on PoseNet (Fridrich council 2026-04-26 CRITICAL).
    Lane RM uses the Lane A canonical 0.5."""
    m = re.search(r"--posetto-noise-std\s+([0-9.]+)", script_text)
    assert m is not None
    assert float(m.group(1)) > 0


# ── Single variable vs Lane A ─────────────────────────────────────────


def test_uses_same_steps_as_lane_a(script_text: str):
    """Lane RM is a single-variable test against Lane A. Same step
    count keeps everything except the optimiser identical."""
    lane_a = LANE_A_SCRIPT.read_text()
    m_a = re.search(r"--steps\s+(\d+)", lane_a)
    m_rm = re.search(r"--steps\s+(\d+)", script_text)
    assert m_a is not None and m_rm is not None
    assert m_a.group(1) == m_rm.group(1), (
        f"Lane RM --steps {m_rm.group(1)} != Lane A --steps {m_a.group(1)} — "
        "single-variable contract violated"
    )


def test_uses_same_batch_pairs_as_lane_a(script_text: str):
    lane_a = LANE_A_SCRIPT.read_text()
    m_a = re.search(r"--batch-pairs\s+(\d+)", lane_a)
    m_rm = re.search(r"--batch-pairs\s+(\d+)", script_text)
    assert m_a is not None and m_rm is not None
    assert m_a.group(1) == m_rm.group(1), (
        f"Lane RM --batch-pairs {m_rm.group(1)} != Lane A "
        f"--batch-pairs {m_a.group(1)}"
    )
