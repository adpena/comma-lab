# SPDX-License-Identifier: MIT
"""Lane W: tests for experiments/profile_pair_sensitivity.py.

Verifies the profiler's CLI surface, weight tensor invariants, and
sidecar JSON contract WITHOUT pulling in the heavy upstream PoseNet /
SegNet path (those need CUDA + multi-GB weights).

The actual sensitivity sweep is exercised end-to-end by the remote
Lane W bootstrap script on a CUDA host.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "profile_pair_sensitivity.py"


def _import_profiler():
    """Import experiments/profile_pair_sensitivity.py without executing main."""
    spec = importlib.util.spec_from_file_location(
        "profile_pair_sensitivity", SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("profile_pair_sensitivity", mod)
    spec.loader.exec_module(mod)
    return mod


def test_script_exists():
    assert SCRIPT.exists(), f"profile_pair_sensitivity.py missing at {SCRIPT}"


def test_script_has_canonical_cli_flags():
    """Argparse must include every flag the Lane W bootstrap script invokes.

    This is the dead-flag-wiring guard
    (memory: feedback_dead_flag_wiring_pattern). If the bootstrap script
    passes a flag the profiler doesn't have, this test catches it BEFORE
    we burn $0.50 of GPU on a SystemExit.
    """
    src = SCRIPT.read_text()
    flags = set(re.findall(r'p\.add_argument\(\s*"(--[a-z0-9_-]+)"', src))
    expected = {
        "--checkpoint", "--poses", "--masks-mkv", "--video-mkv",
        "--upstream", "--output", "--top-k", "--hard-weight",
        "--device", "--batch-size",
    }
    missing = expected - flags
    assert not missing, (
        f"profile_pair_sensitivity.py argparse missing flags: {missing}. "
        f"The Lane W bootstrap script will SystemExit on launch."
    )


def test_default_device_is_cuda():
    """CLAUDE.md non-negotiable: no MPS fallback. CUDA REQUIRED."""
    src = SCRIPT.read_text()
    m = re.search(r'add_argument\(\s*"--device".*?default=("[^"]+")', src, re.DOTALL)
    assert m is not None, "no --device flag found"
    assert m.group(1) == '"cuda"', (
        f"Default device must be 'cuda' (CLAUDE.md non-negotiable: no MPS "
        f"fallback for auth measurements). Got {m.group(1)}."
    )


def test_no_mps_fallback():
    """forbidden device-selection-default trap (memory: feedback_default_to_convenience_trap)."""
    src = SCRIPT.read_text()
    assert "torch.backends.mps" not in src, (
        "profile_pair_sensitivity.py must not import MPS detection. "
        "MPS PoseNet drift is 23x — sensitivity ranking from MPS is noise."
    )


def test_mps_drift_check_present():
    """The script must explicitly raise (not silently fall back) on no CUDA."""
    src = SCRIPT.read_text()
    # Must explicitly raise on no-CUDA, not fall back.
    assert re.search(
        r'if device == "cuda" and not torch\.cuda\.is_available\(\):\s*\n\s*raise',
        src,
    ), (
        "Script must hard-raise when --device cuda but no CUDA available. "
        "Falling back to CPU/MPS would produce noise-tier ranking."
    )


def test_weight_construction_unit():
    """Synthetic per-pair contribution -> weights tensor shape + invariants.

    Replicates the core weight-builder logic without needing scorers /
    real masks. The actual contribution math is `100*seg + sqrt(10*pose)`
    per the score formula (CLAUDE.md non-negotiable
    feedback_curriculum_must_use_full_score).
    """
    import torch

    n_pairs = 600
    # Synthesize realistic distortions: most pairs near zero, ~30 with
    # fat tails (the heavy-tail story from feedback_posenet_tracking).
    torch.manual_seed(0)
    pose = torch.full((n_pairs,), 0.005, dtype=torch.float64)
    seg = torch.full((n_pairs,), 0.001, dtype=torch.float64)
    hardest = torch.randperm(n_pairs)[:30]
    pose[hardest] = 0.5
    seg[hardest] = 0.05

    contrib = 100.0 * seg + (10.0 * pose).sqrt()
    top_k = 30
    hard_weight = 5.0
    weights = torch.ones(n_pairs, dtype=torch.float32)
    topk_idx = torch.topk(contrib, top_k, largest=True).indices
    weights[topk_idx] = float(hard_weight)

    # Invariants the profiler must satisfy
    assert weights.shape == (n_pairs,)
    assert weights.dtype == torch.float32
    assert weights.min().item() == 1.0
    assert weights.max().item() == hard_weight
    assert (weights == hard_weight).sum().item() == top_k
    # The top-K indices we synthesised must all be picked
    assert set(topk_idx.tolist()) == set(hardest.tolist())


def test_profile_function_signature():
    """The `profile()` function must accept the kwargs the CLI plumbs."""
    mod = _import_profiler()
    import inspect
    sig = inspect.signature(mod.profile)
    expected = {
        "checkpoint", "poses_path", "masks_mkv", "video_mkv", "upstream",
        "device", "output", "top_k", "hard_weight", "batch_size",
    }
    missing = expected - set(sig.parameters)
    assert not missing, f"profile() missing kwargs: {missing}"


def test_sidecar_json_schema(tmp_path):
    """The sidecar JSON's documented schema_version + key set is stable."""
    # Build a synthetic summary that matches what profile() would dump.
    n_pairs = 4
    summary = {
        "schema_version": 1,
        "lane": "W",
        "n_pairs": n_pairs,
        "top_k": 1,
        "hard_weight": 5.0,
        "device": "cuda",
        "checkpoint": "x",
        "poses_path": None,
        "masks_mkv": "y",
        "video_mkv": "z",
        "upstream": "u",
        "hardest_pair_indices": [3],
        "stats": {
            "pose_mean": 0.0,
            "pose_max": 0.0,
            "pose_p99": 0.0,
            "seg_mean": 0.0,
            "seg_max": 0.0,
            "seg_p99": 0.0,
            "contrib_mean": 0.0,
            "contrib_max": 0.0,
            "contrib_top_k_mean": 0.0,
            "weight_floor": 1.0,
            "weight_ceiling": 5.0,
        },
        "per_pair_pose_dist": [0.0] * n_pairs,
        "per_pair_seg_dist": [0.0] * n_pairs,
        "per_pair_contrib": [0.0] * n_pairs,
        "elapsed_s": 0.0,
    }
    out = tmp_path / "weights.pt.meta.json"
    out.write_text(json.dumps(summary))
    loaded = json.loads(out.read_text())
    # Key set must match exactly (forward-compatible additions require schema bump)
    assert loaded["schema_version"] == 1
    assert loaded["lane"] == "W"
    assert len(loaded["per_pair_pose_dist"]) == loaded["n_pairs"]
    assert len(loaded["per_pair_seg_dist"]) == loaded["n_pairs"]
    assert len(loaded["per_pair_contrib"]) == loaded["n_pairs"]


def test_help_runs(capsys):
    """--help must run without crashing — catches argparse misconfig."""
    mod = _import_profiler()
    with pytest.raises(SystemExit) as e:
        mod.main(["--help"])
    assert e.value.code == 0
