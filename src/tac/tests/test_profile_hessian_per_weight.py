# SPDX-License-Identifier: MIT
"""Tests for experiments/profile_hessian_per_weight.py — Lane Ω Phase 1.

These tests do NOT call the full profiler (which requires upstream scorers,
GT video, masks, and CUDA). Instead they pin the structural properties:

  1. The script exists, has the expected argparse flags, and the device
     argument is restricted to {cuda, cpu} (no MPS).
  2. The eligible-parameter selection respects SC_PROTECTED_NAME_PATTERNS:
     protected layers (renderer.head, motion.head, FiLM linears, fuse_conv)
     are NEVER included in the eligible set.
  3. The pair-weights loader handles uniform fallback, dict {"weights": ...},
     dict {"pair_weights": ...}, and naked tensor — and rejects mismatched
     length / negative / zero-sum.
  4. The CLI rejects --device mps loud and clear (CLAUDE.md non-negotiable).
  5. The output dict structure matches the documented schema.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "profile_hessian_per_weight.py"


def _load_module():
    """Import experiments/profile_hessian_per_weight.py without running main()."""
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    if str(REPO / "upstream") not in sys.path:
        sys.path.insert(0, str(REPO / "upstream"))
    spec = importlib.util.spec_from_file_location(
        "profile_hessian_per_weight", str(SCRIPT),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Existence and CLI structure ──────────────────────────────────────────


def test_script_exists():
    assert SCRIPT.exists(), f"missing profiler script: {SCRIPT}"


def test_argparse_has_required_flags():
    """Every flag the Lane Ω script will pass must exist."""
    src = SCRIPT.read_text()
    flags = set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))
    expected = {
        "checkpoint", "video", "masks-mkv", "poses", "upstream", "output",
        "top-k", "pair-weights", "all-pairs", "device", "pair-batch",
        "include-protected-conv2d",
    }
    missing = expected - flags
    assert not missing, f"profile_hessian_per_weight.py missing flags: {sorted(missing)}"


def test_device_choice_restricted():
    """--device must accept ONLY cuda or cpu (no MPS)."""
    src = SCRIPT.read_text()
    # Find the --device argument's choices
    m = re.search(r"--device\".*?choices=(\[[^\]]+\])", src, re.DOTALL)
    assert m, "--device must use choices= to restrict the value set"
    choices_text = m.group(1)
    assert "cuda" in choices_text
    assert "cpu" in choices_text
    assert "mps" not in choices_text, (
        "MPS forbidden — Fisher importance on MPS has 23x drift "
        "(CLAUDE.md non-negotiable)"
    )


def test_pair_weights_mutually_exclusive():
    """--pair-weights and --all-pairs MUST be mutually exclusive (so the
    operator can't accidentally pass both and have one silently win)."""
    src = SCRIPT.read_text()
    assert "add_mutually_exclusive_group" in src, (
        "--pair-weights / --all-pairs must be in a mutually exclusive group"
    )
    # The mutually exclusive group must be required=True (operator MUST
    # pick one).
    assert "add_mutually_exclusive_group(required=True)" in src or \
        "add_mutually_exclusive_group(\n        required=True\n    )" in src, (
        "the mutually exclusive group must be required=True so operator "
        "explicitly chooses uniform vs Lane W weighting"
    )


# ── Pair-weights loader ───────────────────────────────────────────────────


def test_pair_weights_uniform_fallback(tmp_path):
    mod = _load_module()
    w = mod._load_pair_weights(None, n_pairs=10)
    assert w.shape == (10,)
    assert torch.allclose(w.sum(), torch.tensor(1.0))
    assert torch.allclose(w, torch.full((10,), 0.1))


def test_pair_weights_naked_tensor(tmp_path):
    mod = _load_module()
    src = torch.tensor([1.0, 2.0, 3.0, 4.0])
    p = tmp_path / "w.pt"
    torch.save(src, p)
    w = mod._load_pair_weights(str(p), n_pairs=4)
    assert torch.allclose(w.sum(), torch.tensor(1.0))
    # Order preserved, normalized
    assert torch.allclose(w, src / src.sum())


def test_pair_weights_dict_weights_key(tmp_path):
    mod = _load_module()
    src = {"weights": torch.tensor([10.0, 20.0]), "metadata": {"foo": "bar"}}
    p = tmp_path / "w.pt"
    torch.save(src, p)
    w = mod._load_pair_weights(str(p), n_pairs=2)
    assert torch.allclose(w, torch.tensor([1.0 / 3, 2.0 / 3]))


def test_pair_weights_dict_pair_weights_key(tmp_path):
    mod = _load_module()
    src = {"pair_weights": torch.tensor([1.0, 1.0, 2.0])}
    p = tmp_path / "w.pt"
    torch.save(src, p)
    w = mod._load_pair_weights(str(p), n_pairs=3)
    assert torch.allclose(w.sum(), torch.tensor(1.0))


def test_pair_weights_length_mismatch_raises(tmp_path):
    mod = _load_module()
    p = tmp_path / "w.pt"
    torch.save(torch.tensor([1.0, 2.0]), p)
    with pytest.raises(ValueError, match="entries"):
        mod._load_pair_weights(str(p), n_pairs=4)


def test_pair_weights_negative_rejected(tmp_path):
    mod = _load_module()
    p = tmp_path / "w.pt"
    torch.save(torch.tensor([1.0, -2.0, 3.0]), p)
    with pytest.raises(ValueError, match="negative"):
        mod._load_pair_weights(str(p), n_pairs=3)


def test_pair_weights_zero_sum_rejected(tmp_path):
    mod = _load_module()
    p = tmp_path / "w.pt"
    torch.save(torch.zeros(3), p)
    with pytest.raises(ValueError, match="degenerate"):
        mod._load_pair_weights(str(p), n_pairs=3)


def test_pair_weights_missing_dict_key_rejected(tmp_path):
    mod = _load_module()
    p = tmp_path / "w.pt"
    torch.save({"foo": torch.tensor([1.0, 2.0])}, p)
    with pytest.raises(ValueError, match="weights"):
        mod._load_pair_weights(str(p), n_pairs=2)


# ── Eligible-parameter selection ──────────────────────────────────────────


def test_eligible_params_excludes_protected():
    """The eligible parameter set must NOT include any layer matching
    SC_PROTECTED_NAME_PATTERNS."""
    mod = _load_module()
    sys.path.insert(0, str(REPO / "src"))
    from tac.renderer import AsymmetricPairGenerator
    from tac.self_compress import SC_PROTECTED_NAME_PATTERNS

    # FiLM-conditioned model so renderer.head/motion.head/film_*/fuse_conv all exist.
    model = AsymmetricPairGenerator(
        num_classes=5, embed_dim=6, base_ch=12, mid_ch=16,
        motion_hidden=8, depth=1, pose_dim=6, use_dsconv=False,
        use_zoom_flow=False, padding_mode="zeros", use_dilation=False,
    )
    eligible = mod._select_eligible_params(model)
    assert len(eligible) > 0, "Lane Ω must have some eligible weights"
    for name in eligible:
        # Strip the .weight suffix to compare against module names
        base = name[: -len(".weight")]
        for pat in SC_PROTECTED_NAME_PATTERNS:
            assert not (base == pat or base.endswith("." + pat)), (
                f"eligible weight {name} should be protected (matches {pat})"
            )


def test_eligible_params_owv3_mode_includes_protected_conv2d_not_linear():
    """OWV3 sensitivity maps need measured Conv2d sensitivity for protected
    convs while protected FiLM Linear layers remain out of scope."""
    mod = _load_module()
    sys.path.insert(0, str(REPO / "src"))
    from tac.renderer import AsymmetricPairGenerator

    model = AsymmetricPairGenerator(
        num_classes=5, embed_dim=6, base_ch=12, mid_ch=16,
        motion_hidden=8, depth=1, pose_dim=6, use_dsconv=False,
        use_zoom_flow=False, padding_mode="zeros", use_dilation=False,
    )
    eligible = mod._select_eligible_params(
        model,
        include_protected_conv2d=True,
    )
    assert "renderer.fuse_conv.weight" in eligible
    assert "renderer.head.weight" in eligible
    assert "motion.head.weight" in eligible
    assert "renderer.film_bottleneck.scale.weight" not in eligible
    assert "renderer.film_bottleneck.shift.weight" not in eligible


def test_eligible_params_records_exclusion_reasons():
    mod = _load_module()
    sys.path.insert(0, str(REPO / "src"))
    from tac.renderer import AsymmetricPairGenerator

    model = AsymmetricPairGenerator(
        num_classes=5, embed_dim=6, base_ch=12, mid_ch=16,
        motion_hidden=8, depth=1, pose_dim=6, use_dsconv=False,
        use_zoom_flow=False, padding_mode="zeros", use_dilation=False,
    )
    eligible, excluded = mod._select_eligible_params_with_exclusions(
        model,
        include_protected_conv2d=False,
    )
    assert "renderer.fuse_conv.weight" not in eligible
    assert excluded["renderer.fuse_conv.weight"] == "protected_conv2d"
    assert excluded["renderer.film_bottleneck.scale.weight"] == "protected_linear"


def test_eligible_params_includes_bulk_convs():
    """The bulk feature-extraction Conv2d weights MUST be eligible."""
    mod = _load_module()
    sys.path.insert(0, str(REPO / "src"))
    from tac.renderer import AsymmetricPairGenerator

    model = AsymmetricPairGenerator(
        num_classes=5, embed_dim=6, base_ch=8, mid_ch=12,
        motion_hidden=8, depth=1, pose_dim=0, use_dsconv=False,
    )
    eligible = mod._select_eligible_params(model)
    # stem_conv, down_conv, bottleneck.conv1, bottleneck.conv2 should all be in
    eligible_bases = {n[: -len(".weight")] for n in eligible}
    expected_substrings = ["stem_conv", "down_conv", "bottleneck"]
    for sub in expected_substrings:
        assert any(sub in name for name in eligible_bases), (
            f"expected eligible layer name containing {sub!r}"
        )


def test_eligible_params_skips_conv_transpose():
    """ConvTranspose2d must be skipped (different STE behavior)."""
    mod = _load_module()
    parent = torch.nn.Module()
    parent.regular = torch.nn.Conv2d(3, 4, 3)
    parent.transposed = torch.nn.ConvTranspose2d(4, 3, 3)
    eligible = mod._select_eligible_params(parent)
    assert "regular.weight" in eligible
    assert "transposed.weight" not in eligible


# ── Output schema ─────────────────────────────────────────────────────────


def test_output_schema_documented_keys():
    """The module docstring documents an output schema; verify the listed
    metadata keys at least appear as string literals in the source."""
    src = SCRIPT.read_text()
    for key in ("checkpoint", "poses", "masks_mkv", "video_mkv",
                "pair_weights", "top_k", "n_pairs_seen",
                "n_eligible_layers", "n_eligible_weights",
                "imp_min", "imp_max", "imp_p95", "imp_p99",
                "git_hash", "torch_version", "device", "elapsed_s"):
        assert f'"{key}"' in src, f"output schema key {key!r} missing from source"
