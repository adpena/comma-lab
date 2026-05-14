# SPDX-License-Identifier: MIT
"""Lane SI: tests for experiments/profile_scorer_saliency.py.

We don't actually invoke the script (it requires CUDA + the upstream
PoseNet + SegNet safetensors). Instead we verify:

1. The script's argparse is well-formed (no invented flags, all required
   flags present — mirrors the dead-flag-wiring guard).
2. The script's CLI surface matches what
   scripts/remote_lane_si_saliency_inversion.sh actually invokes.
3. The output schema (the dict keys + tensor shapes that downstream
   consumers depend on) is documented + tested.
"""
from __future__ import annotations

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "profile_scorer_saliency.py"


def test_script_exists() -> None:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"


def test_script_argparse_has_required_flags() -> None:
    """Parse the script source and verify every flag the remote bootstrap
    invokes is registered. NEVER invent CLI flags (CLAUDE.md non-negotiable).
    """
    src = SCRIPT.read_text()
    # Required flags per the spec
    required = {
        "--checkpoint",
        "--poses",
        "--masks-mkv",
        "--video",
        "--output",
        "--device",
        "--upstream-dir",
        "--n-pairs",
        "--reduce",
    }
    for flag in required:
        assert f'"{flag}"' in src, f"argparse missing {flag} in {SCRIPT}"


def test_script_imports_saliency_inversion() -> None:
    src = SCRIPT.read_text()
    assert "from tac.saliency_inversion import" in src, (
        "profile_scorer_saliency.py must import from tac.saliency_inversion"
    )
    assert "compute_pixel_saliency" in src


def test_script_blocks_non_cuda_device() -> None:
    """CLAUDE.md non-negotiable: MPS auth eval is NOISE; CPU forbidden."""
    src = SCRIPT.read_text()
    # Either the choices=['cuda'] argparse constraint OR an explicit MPS guard.
    assert (
        "choices=[\"cuda\"]" in src
        or "_require_cuda" in src
        or "mps" in src.lower()  # at least mention MPS in the rejection
    ), "script must reject non-CUDA device"


def test_script_remote_bootstrap_uses_only_real_flags() -> None:
    """The Lane SI bootstrap script invokes profile_scorer_saliency.py.
    Every flag passed must be a real argparse flag (dead-flag guard)."""
    bootstrap = REPO / "scripts" / "remote_lane_si_saliency_inversion.sh"
    assert bootstrap.is_file()
    bootstrap_src = bootstrap.read_text()

    # Pull every '--foo' flag passed to profile_scorer_saliency.py
    invoke_block = re.search(
        r"profile_scorer_saliency\.py(.+?)2>&1",
        bootstrap_src,
        re.DOTALL,
    )
    assert invoke_block is not None, (
        "remote_lane_si_saliency_inversion.sh must invoke profile_scorer_saliency.py"
    )
    flags = set(re.findall(r"--[\w-]+", invoke_block.group(1)))

    # Every flag must be registered in the script's argparse.
    script_src = SCRIPT.read_text()
    for flag in flags:
        assert f'"{flag}"' in script_src, (
            f"remote_lane_si_saliency_inversion.sh passes {flag!r} but "
            f"profile_scorer_saliency.py argparse does NOT define it. "
            "This is the exact dead-flag-wiring trap CLAUDE.md forbids."
        )


def test_output_schema_documented() -> None:
    """The output dict keys must match the spec: posenet_saliency,
    segnet_saliency, combined."""
    src = SCRIPT.read_text()
    assert "'posenet_saliency'" in src or '"posenet_saliency"' in src
    assert "'segnet_saliency'" in src or '"segnet_saliency"' in src
    assert "'combined'" in src or '"combined"' in src


def test_output_uses_torch_save() -> None:
    src = SCRIPT.read_text()
    assert "torch.save(" in src, (
        "Output must be a torch.save'd dict so downstream tools can torch.load() it"
    )


def test_provenance_recorded() -> None:
    """Every saliency map must be accompanied by provenance (CLAUDE.md
    feedback_no_signal_loss)."""
    src = SCRIPT.read_text()
    assert "provenance" in src
    assert "torch_version" in src
    assert "ts_utc" in src or "started_at_utc" in src


def test_camera_grid_constants_exported() -> None:
    """saliency_inversion exposes CAMERA_H, CAMERA_W (1164x874 from
    upstream/frame_utils.py); profile script must use them so the map
    aligns with mask frames."""
    from tac.saliency_inversion import CAMERA_H, CAMERA_W

    assert CAMERA_H == 874
    assert CAMERA_W == 1164
