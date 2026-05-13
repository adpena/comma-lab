"""Tests for experiments/profile_fp4_layer_sensitivity.py — Lane F-V4 Phase 1.

Pin the structural properties of the per-layer FP4 sensitivity profiler
without running the heavy CUDA-only end-to-end measurement (which requires
upstream scorers + GT video + masks):

  1. The script exists, has the expected argparse flags, and --device is
     restricted to {cuda, cpu} (no MPS).
  2. Eligible-layer selection picks Conv2d / Linear with weight.ndim>=2
     and skips ConvTranspose2d.
  3. quantize_one_layer_fp4 round-trips a layer's weight in place and
     restore_layer puts the original back exactly.
  4. The output schema (delta / delta_pose / delta_seg / param_count /
     baseline / metadata) is documented in the source.
  5. The CLI rejects --device mps loud and clear.
  6. Encoded grayscale masks are remapped to class ids before renderer input.

Also pin the qat_finetune.py companion piece:

  7. bit_allocation_from_sensitivity sorts layers by sensitivity ascending
     and assigns FP4 to the bottom target_rate fraction of params.
  8. target_rate=1.0 → uniform FP4 (all layers FP4); target_rate=0.0 →
     no FP4 (all layers FP16). Sweep boundary correctness.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "profile_fp4_layer_sensitivity.py"
QAT_FINETUNE = REPO / "experiments" / "qat_finetune.py"


def _load_module(path: Path, name: str):
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    if str(REPO / "upstream") not in sys.path:
        sys.path.insert(0, str(REPO / "upstream"))
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so @dataclass (which inspects
    # cls.__module__ via sys.modules) can resolve the loading module.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Existence + CLI structure ───────────────────────────────────────────


def test_script_exists():
    assert SCRIPT.exists(), f"missing profile script: {SCRIPT}"


def test_argparse_has_required_flags():
    """Every flag the Lane F-V4 script will pass must exist."""
    src = SCRIPT.read_text()
    flags = set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))
    expected = {
        "checkpoint", "video", "masks-mkv", "poses", "upstream",
        "output", "device", "n-pairs", "block-size", "predicted-band",
    }
    missing = expected - flags
    assert not missing, f"profile_fp4_layer_sensitivity.py missing flags: {sorted(missing)}"


def test_device_choice_restricted():
    """--device must accept ONLY cuda or cpu (no MPS)."""
    src = SCRIPT.read_text()
    m = re.search(r"--device\".*?choices=(\[[^\]]+\])", src, re.DOTALL)
    assert m, "--device must use choices= to restrict the value set"
    choices = m.group(1)
    assert "cuda" in choices
    assert "cpu" in choices
    assert "mps" not in choices, (
        "MPS forbidden — PoseNet on MPS drifts 23x (CLAUDE.md non-negotiable)"
    )


def test_cpu_requires_explicit_diagnostic_flag_in_source():
    src = SCRIPT.read_text()
    assert "--allow-diagnostic-cpu" in src
    assert "diagnostic-only and non-promotable" in src
    assert "promotion_eligible" in src
    assert "diagnostic_cpu" in src


def test_device_mps_rejection_message_in_source():
    """The source must contain an explicit MPS-rejection message so the
    operator gets a clear FATAL on accidental --device mps."""
    src = SCRIPT.read_text()
    assert "FATAL" in src and "mps" in src.lower(), (
        "source must reject --device mps loud (CLAUDE.md non-negotiable)"
    )


# ── Eligible-layer selection ────────────────────────────────────────────


def test_select_eligible_layers_includes_conv2d_and_linear():
    mod = _load_module(SCRIPT, "profile_fp4_layer_sensitivity")
    parent = nn.Module()
    parent.conv = nn.Conv2d(3, 4, 3)
    parent.lin = nn.Linear(4, 5)
    eligible = mod.select_eligible_layers(parent)
    assert "conv" in eligible
    assert "lin" in eligible


def test_select_eligible_layers_skips_conv_transpose():
    mod = _load_module(SCRIPT, "profile_fp4_layer_sensitivity")
    parent = nn.Module()
    parent.regular = nn.Conv2d(3, 4, 3)
    parent.transposed = nn.ConvTranspose2d(4, 3, 3)
    eligible = mod.select_eligible_layers(parent)
    assert "regular" in eligible
    assert "transposed" not in eligible


def test_select_eligible_layers_skips_1d_weights():
    """Layers with weight.ndim<2 (e.g., 1D BatchNorm scale) must be skipped."""
    mod = _load_module(SCRIPT, "profile_fp4_layer_sensitivity")
    parent = nn.Module()
    parent.conv = nn.Conv2d(3, 4, 3)
    parent.bn = nn.BatchNorm2d(4)  # weight is 1D
    eligible = mod.select_eligible_layers(parent)
    assert "conv" in eligible
    assert "bn" not in eligible, "1D BatchNorm weight is not FP4-eligible"


def test_gray_mask_to_class_ids_remaps_encoded_luma_values():
    mod = _load_module(SCRIPT, "profile_fp4_layer_sensitivity")
    gray = torch.tensor([[0, 63, 126, 189, 252, 255]], dtype=torch.uint8)
    classes = mod._gray_mask_to_class_ids(gray)
    assert classes.tolist() == [[0, 1, 2, 3, 4, 4]]


# ── Per-layer quant + restore ───────────────────────────────────────────


def test_quantize_one_layer_round_trip_preserves_shape():
    mod = _load_module(SCRIPT, "profile_fp4_layer_sensitivity")
    layer = nn.Conv2d(8, 16, 3)
    original = mod.quantize_one_layer_fp4(layer, block_size=32)
    assert layer.weight.shape == original.shape
    assert original.shape == (16, 8, 3, 3)


def test_restore_layer_recovers_original_exactly():
    mod = _load_module(SCRIPT, "profile_fp4_layer_sensitivity")
    layer = nn.Conv2d(8, 16, 3)
    expected = layer.weight.detach().clone()
    original = mod.quantize_one_layer_fp4(layer, block_size=32)
    # After quantization, weight should differ.
    assert not torch.allclose(layer.weight, expected), (
        "quantize_one_layer_fp4 should perturb the weight"
    )
    mod.restore_layer(layer, original)
    assert torch.allclose(layer.weight, expected), (
        "restore_layer must recover the original weight exactly"
    )


def test_quantize_one_layer_does_not_mutate_other_layers():
    """The whole point: quantizing one layer must NOT affect others."""
    mod = _load_module(SCRIPT, "profile_fp4_layer_sensitivity")
    parent = nn.Module()
    parent.a = nn.Conv2d(4, 8, 3)
    parent.b = nn.Conv2d(8, 8, 3)
    a_before = parent.a.weight.detach().clone()
    b_before = parent.b.weight.detach().clone()
    mod.quantize_one_layer_fp4(parent.b, block_size=32)
    # a is untouched
    assert torch.allclose(parent.a.weight, a_before)
    # b is modified
    assert not torch.allclose(parent.b.weight, b_before)


# ── Output schema documented ────────────────────────────────────────────


def test_output_schema_documented_keys():
    """The module docstring + torch.save dict must reference the expected keys."""
    src = SCRIPT.read_text()
    for key in ("delta", "delta_pose", "delta_seg", "param_count",
                "baseline", "metadata"):
        assert f'"{key}"' in src, f"output schema key {key!r} missing from source"


def test_metadata_records_predicted_band():
    src = SCRIPT.read_text()
    assert "predicted_band" in src, (
        "metadata must record the council predicted_band per CLAUDE.md"
    )


def test_metadata_records_git_hash():
    src = SCRIPT.read_text()
    assert "git_hash" in src


# ── bit_allocation_from_sensitivity (Phase 2 selector) ──────────────────


@pytest.fixture
def qat_module():
    return _load_module(QAT_FINETUNE, "qat_finetune")


@pytest.fixture
def small_model():
    """Small renderer-shaped model with several Conv2d layers."""
    parent = nn.Module()
    parent.conv1 = nn.Conv2d(3, 8, 3)   # 216 params
    parent.conv2 = nn.Conv2d(8, 16, 3)  # 1152 params
    parent.conv3 = nn.Conv2d(16, 32, 3)  # 4608 params (BIGGEST)
    parent.conv4 = nn.Conv2d(32, 5, 1)  # 160 params
    return parent


def test_bit_allocation_from_sensitivity_basic(tmp_path, qat_module, small_model):
    """Layers with smallest sensitivity (FP4-tolerable) get bulk_bits."""
    delta = {
        "conv1": 0.001,  # least sensitive → FP4
        "conv3": 0.002,  # next → FP4 (this is the BIG layer, dominates rate)
        "conv2": 0.05,   # → likely FP16 boundary
        "conv4": 0.10,   # most sensitive → FP16
    }
    param_count = {
        "conv1": 216, "conv2": 1152, "conv3": 4608, "conv4": 160,
    }
    sens_path = tmp_path / "layer_sensitivity.pt"
    torch.save({
        "delta": delta,
        "delta_pose": {k: v for k, v in delta.items()},
        "delta_seg": {k: v for k, v in delta.items()},
        "param_count": param_count,
        "baseline": {"pose_d": 0.01, "seg_d": 0.005, "distortion": 0.8},
        "metadata": {"checkpoint": "stub"},
    }, sens_path)

    allocation, sens_used = qat_module.bit_allocation_from_sensitivity(
        sens_path, small_model, target_rate=0.70,
        bulk_bits=4, critical_bits=16,
    )
    # Total params = 6136; 70% target = 4295
    # Sorted ASC by sensitivity: conv1(216), conv3(4608), conv2(1152), conv4(160)
    # Greedy fill:
    #   conv1 (216):  216 ≤ 4295 → FP4, cumulative=216
    #   conv3 (4608): 216+4608=4824 > 4295 → FP16 (skipped, doesn't fit)
    #   conv2 (1152): 216+1152=1368 ≤ 4295 → FP4, cumulative=1368
    #   conv4 (160):  1368+160=1528 ≤ 4295 → FP4, cumulative=1528
    # Result: only conv3 stays FP16 (the BIG layer that doesn't fit budget).
    assert allocation["conv1.weight"] == 4, (
        f"conv1 (least sensitive) must be FP4. allocation={allocation}"
    )
    assert allocation["conv3.weight"] == 16, (
        f"conv3 alone is 4608 > 70% budget (4295), must be FP16. "
        f"allocation={allocation}"
    )
    assert allocation["conv2.weight"] == 4, (
        "conv2 fits in remaining budget after conv1 → FP4"
    )
    assert allocation["conv4.weight"] == 4, (
        "conv4 (most sensitive but smallest) fits → FP4 by greedy. The "
        "intent of mixed-precision is per-PARAM not per-LAYER protection."
    )


def test_bit_allocation_target_rate_1_0_uniform_fp4(tmp_path, qat_module, small_model):
    """target_rate=1.0 → all layers FP4 (matches Lane F-V3 uniform FP4)."""
    delta = {f"conv{i}": float(i) * 0.01 for i in range(1, 5)}
    param_count = {f"conv{i}": 100 for i in range(1, 5)}
    sens_path = tmp_path / "layer_sensitivity.pt"
    torch.save({
        "delta": delta,
        "param_count": param_count,
        "baseline": {},
        "metadata": {},
    }, sens_path)
    allocation, _ = qat_module.bit_allocation_from_sensitivity(
        sens_path, small_model, target_rate=1.0,
        bulk_bits=4, critical_bits=16,
    )
    for k, v in allocation.items():
        assert v == 4, f"target_rate=1.0 must give all layers FP4 (got {k}={v})"


def test_bit_allocation_target_rate_0_0_no_fp4(tmp_path, qat_module, small_model):
    """target_rate=0.0 → no FP4 (matches FP32 baseline)."""
    delta = {f"conv{i}": float(i) * 0.01 for i in range(1, 5)}
    param_count = {f"conv{i}": 100 for i in range(1, 5)}
    sens_path = tmp_path / "layer_sensitivity.pt"
    torch.save({
        "delta": delta,
        "param_count": param_count,
        "baseline": {},
        "metadata": {},
    }, sens_path)
    allocation, _ = qat_module.bit_allocation_from_sensitivity(
        sens_path, small_model, target_rate=0.0,
        bulk_bits=4, critical_bits=16,
    )
    for k, v in allocation.items():
        assert v == 16, f"target_rate=0.0 must give all layers FP16 (got {k}={v})"


def test_bit_allocation_skips_layers_not_in_model(tmp_path, qat_module, small_model):
    """Sensitivity entries for layers that don't exist in the model are skipped."""
    delta = {
        "conv1": 0.001,
        "ghost_layer": 0.01,  # not in small_model
    }
    param_count = {"conv1": 216, "ghost_layer": 1000}
    sens_path = tmp_path / "layer_sensitivity.pt"
    torch.save({
        "delta": delta,
        "param_count": param_count,
        "baseline": {},
        "metadata": {},
    }, sens_path)
    allocation, sens_used = qat_module.bit_allocation_from_sensitivity(
        sens_path, small_model, target_rate=1.0,
        bulk_bits=4, critical_bits=16,
    )
    # ghost_layer must be skipped.
    assert "ghost_layer.weight" not in allocation
    assert "conv1.weight" in allocation


def test_bit_allocation_raises_on_no_matching_layers(tmp_path, qat_module, small_model):
    """If NO sensitivity entries match the model, raise."""
    delta = {"ghost1": 0.01, "ghost2": 0.02}
    param_count = {"ghost1": 100, "ghost2": 200}
    sens_path = tmp_path / "layer_sensitivity.pt"
    torch.save({
        "delta": delta,
        "param_count": param_count,
        "baseline": {},
        "metadata": {},
    }, sens_path)
    with pytest.raises(ValueError, match="No sensitivity entries match"):
        qat_module.bit_allocation_from_sensitivity(
            sens_path, small_model, target_rate=0.5,
        )


def test_bit_allocation_raises_on_missing_delta_key(tmp_path, qat_module, small_model):
    sens_path = tmp_path / "layer_sensitivity.pt"
    torch.save({"not_delta": {}, "metadata": {}}, sens_path)
    with pytest.raises(ValueError, match="missing 'delta'"):
        qat_module.bit_allocation_from_sensitivity(
            sens_path, small_model, target_rate=0.5,
        )


# ── qat_finetune CLI exposes the new flags ──────────────────────────────


def test_qat_finetune_argparse_has_v4_flags():
    src = QAT_FINETUNE.read_text()
    flags = set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))
    for needed in (
        "mixed-precision-from-sensitivity",
        "mixed-precision-target-rate",
        "mixed-precision-bulk-bits",
        "mixed-precision-critical-bits",
    ):
        assert needed in flags, f"qat_finetune.py argparse missing {needed!r}"


def test_qat_finetune_validates_mutual_exclusion():
    """The two mixed-precision sources must be mutually exclusive."""
    src = QAT_FINETUNE.read_text()
    assert (
        "mutually exclusive" in src
        and "mixed_precision_from_sensitivity" in src
        and "mixed_precision_json" in src
    ), "qat_finetune.py must reject both --mixed-precision-* sources at once"


def test_qat_finetune_validates_target_rate_range():
    """target_rate must be in [0, 1] — guards against typos like 70 vs 0.7."""
    src = QAT_FINETUNE.read_text()
    assert "0.0 <= cfg.mixed_precision_target_rate <= 1.0" in src or (
        "mixed_precision_target_rate must be in [0, 1]" in src
    ), "qat_finetune.py must validate target_rate range"
