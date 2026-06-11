# SPDX-License-Identifier: MIT
"""Torch-parity + NO-FAKE tests for the PR95 8-stage curriculum MLX port.

The CONTRACT (CLAUDE.md NO-FAKE supreme rule): every ported mechanism must
ACTUALLY perform its work on the real inputs. Each test below FAILS if the
mechanism is a no-op:

- C1a entropy (L16): matches the torch ``cat_entropy_v2`` to fp32 epsilon on the
  SAME weights, AND its gradient is non-zero on the weights (NOT a constant).
- fake_quant / QAT (L14): bit-identical to torch ``fake_quantize``, AND actually
  changes the render (STE forward is not identity for non-grid weights).
- sigma noise (L17): no-op at sigma=0, real Gaussian at sigma>0 (correct std).
- the 8-stage spec: matches the EXACT torch ``make_config`` per-stage tuple
  (epochs, seg_loss, sigma, c1a_lambda, qat, optimizer/lr) for ALL 8 stages.
- the scheduler: carries weights stage->stage, switches the bridge seg-loss per
  stage, and the optimizer-schedule selector resolves Muon-vs-AdamW correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

try:
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    _HAS_MLX = True
except Exception:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    tree_flatten = tree_unflatten = None  # type: ignore[assignment]
    _HAS_MLX = False

skip_no_mlx = pytest.mark.skipif(not _HAS_MLX, reason="mlx.core not available")

# The proven PR95 torch reference (source of truth for the parity gate).
_PR95_SRC = (
    Path(__file__).resolve().parents[4]
    / "experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto"
    / "source/submissions/hnerv_muon/src"
)
_PR95_STAGES = _PR95_SRC / "stages"


def _import_torch_reference():
    """Import the torch ``losses`` reference (cat_entropy_v2 + fake_quantize)."""
    if str(_PR95_SRC) not in sys.path:
        sys.path.insert(0, str(_PR95_SRC))
    import losses as torch_losses  # type: ignore[import-not-found]

    return torch_losses


# ===========================================================================
# (1) C1a entropy (L16) — torch parity + NO-FAKE gradient.
# ===========================================================================


@skip_no_mlx
def test_c1a_entropy_matches_torch_to_fp32_epsilon():
    """``cat_entropy_v2_mlx`` matches torch ``cat_entropy_v2`` on the SAME weights.

    The torch version iterates ``named_modules()`` selecting Conv2d/Linear and
    reads ``mod.weight``; the MLX version takes the equivalent weight arrays. The
    per-tensor abs-max-normalize + soft-histogram + entropy math is layout-invariant,
    so the NHWC MLX weights give the SAME value as the torch OIHW weights for
    identical weight values. Tensors are kept < sample_size so the random subsample
    is NOT hit (deterministic parity).
    """
    from tac.mlx_pr95_port.mlx_losses import cat_entropy_v2_mlx

    torch_losses = _import_torch_reference()

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(28, 64)
            self.conv = nn.Conv2d(8, 16, 3)

    np.random.seed(0)
    m = _Tiny()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.tensor(np.random.randn(*p.shape).astype(np.float32) * 0.05))

    with torch.no_grad():
        t_c1a = float(torch_losses.cat_entropy_v2(m, sigma=0.2, sample_size=2000).item())

    lin_w = mx.array(m.lin.weight.detach().numpy())
    conv_w_nhwc = mx.array(
        np.transpose(m.conv.weight.detach().numpy(), (0, 2, 3, 1)).copy()
    )
    mlx_c1a = float(cat_entropy_v2_mlx([lin_w, conv_w_nhwc], sigma=0.2).item())

    assert abs(t_c1a - mlx_c1a) < 1e-4, (
        f"C1a entropy mismatch: torch {t_c1a} vs mlx {mlx_c1a}"
    )


@skip_no_mlx
def test_c1a_GRADIENT_matches_torch_to_fp32_epsilon():
    """The C1a ENTROPY GRADIENT (not just the value) matches torch to fp32 epsilon.

    Both torch and MLX DETACH the abs-max scale (torch ``.detach()``,
    MLX ``stop_gradient``), so the gradient flows only through the soft-histogram.
    This is the load-bearing parity: the curriculum ADDS this gradient to the
    score-aware grad, so a gradient mismatch would silently corrupt the C1a effect.
    A naive finite-difference (which lets the detached scale vary) does NOT match
    either backend — only the analytic torch gradient is the correct reference.
    """
    from tac.mlx_pr95_port.mlx_losses import cat_entropy_v2_mlx

    torch_losses = _import_torch_reference()
    np.random.seed(0)
    weight = np.random.randn(32, 16).astype(np.float32) * 0.05

    lin = nn.Linear(16, 32, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor(weight))

    class _M(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.lin = layer

    m = _M(lin)
    m.lin.weight.requires_grad_(True)
    torch_losses.cat_entropy_v2(m, sigma=0.2, sample_size=2000).backward()
    g_torch = m.lin.weight.grad.detach().numpy()

    g_mlx = np.asarray(mx.grad(lambda x: cat_entropy_v2_mlx([x], sigma=0.2))(mx.array(weight)))
    rel = np.abs(g_torch - g_mlx).max() / (np.abs(g_torch).max() + 1e-12)
    assert rel < 1e-4, f"C1a gradient mismatch: rel {rel}"


@skip_no_mlx
def test_c1a_entropy_sigma_sweep_sharpens_with_smaller_sigma():
    """Smaller sigma (0.2 -> 0.1, PR95 stage 7) is a REAL change (not a no-op)."""
    from tac.mlx_pr95_port.mlx_losses import cat_entropy_v2_mlx

    np.random.seed(3)
    w = mx.array(np.random.randn(64, 28).astype(np.float32) * 0.05)
    e_02 = float(cat_entropy_v2_mlx([w], sigma=0.2).item())
    e_01 = float(cat_entropy_v2_mlx([w], sigma=0.1).item())
    assert abs(e_02 - e_01) > 1e-3, "sigma sweep 0.2->0.1 must change the entropy"


@skip_no_mlx
def test_c1a_gradient_is_nonzero_on_weights_only_NOFAKE():
    """C1a gradient is NON-ZERO on the weights (NOT a stub) and 0 on latents/biases."""
    from tac.local_acceleration.pr95_hnerv_mlx import HNeRVSyntheticTrainingBundleMLX
    from tac.mlx_pr95_port.curriculum_mechanisms import (
        StageMechanisms,
        add_c1a_entropy_gradient,
        weight_tensor_keys,
    )

    b = HNeRVSyntheticTrainingBundleMLX(latent_count=2, base_channels=8, seed=1)
    wk = weight_tensor_keys(b.trainable_parameters())
    zero = {k: mx.zeros_like(v) for k, v in tree_flatten(b.trainable_parameters())}
    g0 = tree_unflatten(list(zero.items()))
    mech = StageMechanisms(cat_lambda=0.01, cat_sigma=0.2)
    g1 = add_c1a_entropy_gradient(
        g0, b.trainable_parameters(), wk, mech, rng_key=mx.random.key(0)
    )
    g1f = dict(tree_flatten(g1))
    gmax = max(float(mx.max(mx.abs(g1f[k])).item()) for k in wk)
    assert gmax > 0.0, "C1a gradient must be nonzero on weights (NO-FAKE)"
    assert float(mx.max(mx.abs(g1f["latents"])).item()) == 0.0, (
        "C1a must not touch latents"
    )
    # biases (1D *.bias) must remain zero.
    for k, v in g1f.items():
        if k.endswith(".bias"):
            assert float(mx.max(mx.abs(v)).item()) == 0.0, f"C1a touched bias {k}"


@skip_no_mlx
def test_c1a_lambda_zero_is_a_noop():
    """cat_lambda=0 (stages 1-4) leaves the gradient unchanged (the C1a term is off)."""
    from tac.local_acceleration.pr95_hnerv_mlx import HNeRVSyntheticTrainingBundleMLX
    from tac.mlx_pr95_port.curriculum_mechanisms import (
        StageMechanisms,
        add_c1a_entropy_gradient,
        weight_tensor_keys,
    )

    b = HNeRVSyntheticTrainingBundleMLX(latent_count=2, base_channels=8, seed=2)
    wk = weight_tensor_keys(b.trainable_parameters())
    ones = {k: mx.ones_like(v) for k, v in tree_flatten(b.trainable_parameters())}
    g0 = tree_unflatten(list(ones.items()))
    mech = StageMechanisms(cat_lambda=0.0)
    assert not mech.c1a_active
    g1 = add_c1a_entropy_gradient(g0, b.trainable_parameters(), wk, mech)
    g1f = dict(tree_flatten(g1))
    for k in wk:
        assert float(mx.max(mx.abs(g1f[k] - 1.0)).item()) == 0.0


# ===========================================================================
# (2) fake_quant / QAT (L14) — bit-parity + NO-FAKE render change.
# ===========================================================================


@skip_no_mlx
def test_fake_quantize_bit_identical_to_torch():
    """``fake_quantize_mlx`` is bit-identical to torch ``fake_quantize`` (layout-invariant)."""
    from tac.mlx_pr95_port.mlx_losses import fake_quantize_mlx

    torch_losses = _import_torch_reference()
    np.random.seed(0)
    w_oihw = np.random.randn(16, 8, 3, 3).astype(np.float32) * 0.05
    t = torch_losses.fake_quantize(torch.tensor(w_oihw)).detach().numpy()

    w_nhwc = np.transpose(w_oihw, (0, 2, 3, 1)).copy()
    m = np.asarray(fake_quantize_mlx(mx.array(w_nhwc)))
    m_oihw = np.transpose(m, (0, 3, 1, 2))
    assert np.abs(t - m_oihw).max() == 0.0, "fake_quant must be bit-identical to torch"


@skip_no_mlx
def test_fake_quantize_zero_tensor_safe():
    """A zero weight tensor (scale->1.0 fallback) does not NaN."""
    from tac.mlx_pr95_port.mlx_losses import fake_quantize_mlx

    z = mx.zeros((4, 4))
    out = np.asarray(fake_quantize_mlx(z))
    assert np.all(out == 0.0)


@skip_no_mlx
def test_qat_changes_render_NOFAKE():
    """QAT actually changes the forward render (STE forward != identity)."""
    from tac.local_acceleration.pr95_hnerv_mlx import HNeRVSyntheticTrainingBundleMLX
    from tac.mlx_pr95_port.curriculum_mechanisms import (
        StageMechanisms,
        apply_stage_weight_transforms,
        weight_tensor_keys,
    )

    b = HNeRVSyntheticTrainingBundleMLX(latent_count=2, base_channels=8, seed=0)
    wk = weight_tensor_keys(b.trainable_parameters())
    idx = mx.array([0, 1], dtype=mx.int32)
    r_plain = np.asarray(b(idx))

    flat = dict(tree_flatten(b.parameters()))
    names = list(flat.keys())
    tmap = apply_stage_weight_transforms(
        {n: flat[n] for n in names}, wk, StageMechanisms(use_qat=True)
    )
    b.update(tree_unflatten(list(tmap.items())))
    r_qat = np.asarray(b(idx))
    assert np.abs(r_plain - r_qat).max() > 1e-4, "QAT must change the render (NO-FAKE)"


@skip_no_mlx
def test_qat_gradient_flows_through_ste():
    """QAT STE passes the gradient through (the live weight gets a real grad)."""
    from tac.mlx_pr95_port.mlx_losses import fake_quantize_mlx

    w = mx.array(np.random.RandomState(0).randn(4, 4).astype(np.float32))

    def f(x):
        return mx.sum(fake_quantize_mlx(x) ** 2)

    g = mx.grad(f)(w)
    # STE: d(fake_quant(x))/dx == 1, so grad == d(sum(q^2))/dx evaluated as 2*q...
    # the key NO-FAKE property is the gradient is NON-ZERO (STE not stop-gradient'd).
    assert float(mx.max(mx.abs(g)).item()) > 0.0


# ===========================================================================
# (3) sigma weight-noise (L17).
# ===========================================================================


@skip_no_mlx
def test_sigma_noise_zero_is_noop():
    from tac.mlx_pr95_port.mlx_losses import apply_sigma_noise_mlx

    w = mx.array(np.ones((8, 8), dtype=np.float32))
    out = np.asarray(apply_sigma_noise_mlx(w, 0.0))
    assert np.abs(out - 1.0).max() == 0.0


@skip_no_mlx
def test_sigma_noise_injects_correct_std_NOFAKE():
    from tac.mlx_pr95_port.mlx_losses import apply_sigma_noise_mlx

    w = mx.zeros((256, 256))
    out = np.asarray(apply_sigma_noise_mlx(w, 0.2, rng_key=mx.random.key(0)))
    std = float(np.std(out))
    assert abs(std - 0.2) < 0.02, f"sigma=0.2 noise std {std} must be ~0.2 (NO-FAKE)"


@skip_no_mlx
def test_sigma_noise_schedule_02_to_01_changes_magnitude():
    """The PR95 L17 schedule (0.2 stages 5-6 -> 0.1 stages 7-8) is a REAL change."""
    from tac.mlx_pr95_port.mlx_losses import apply_sigma_noise_mlx

    w = mx.zeros((256, 256))
    s02 = float(np.std(np.asarray(apply_sigma_noise_mlx(w, 0.2, rng_key=mx.random.key(1)))))
    s01 = float(np.std(np.asarray(apply_sigma_noise_mlx(w, 0.1, rng_key=mx.random.key(2)))))
    assert s02 > s01 * 1.5, "sigma 0.2 noise must be visibly larger than 0.1"


# ===========================================================================
# (4) The 8-stage spec matches the torch make_config EXACTLY.
# ===========================================================================


def _torch_stage_configs():
    """Parse the per-stage ``StageConfig(...)`` literals from the torch SOURCE.

    The torch stage modules ``from .common import StageConfig`` and ``from data
    import ...``; ``common``/``data`` transitively require the challenge repo, so
    importing them is environment-heavy. Instead we statically parse the
    ``StageConfig(...)`` keyword literals from each stage file's ``make_config``
    body via ``ast`` — this verifies our MLX spec against the EXACT source-of-truth
    values (the ``StageConfig`` constructor defaults from ``common.py`` are applied
    for any field a stage does not override).
    """
    import ast

    # StageConfig defaults from torch common.py (applied where a stage omits a field).
    common_src = (_PR95_STAGES / "common.py").read_text()
    common_tree = ast.parse(common_src)
    defaults: dict[str, object] = {}
    for node in ast.walk(common_tree):
        if isinstance(node, ast.ClassDef) and node.name == "StageConfig":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    name = stmt.target.id  # type: ignore[attr-defined]
                    try:
                        defaults[name] = ast.literal_eval(stmt.value)
                    except (ValueError, SyntaxError):
                        defaults[name] = None  # Callable / Optional defaults.
            break

    modfiles = [
        "stage1_v328_ce.py", "stage2_v331_softplus.py", "stage3_v332_smooth.py",
        "stage4_v332_qat.py", "stage5_c1a_l7.py", "stage6_lambda_sweep.py",
        "stage7_sigma_sweep.py", "stage8_muon_finetune.py",
    ]
    rows = []
    for fname in modfiles:
        tree = ast.parse((_PR95_STAGES / fname).read_text())
        # Resolve make_config's function-parameter defaults (e.g. stage8 passes
        # ``epochs=epochs`` / ``muon_weight_decay=muon_weight_decay`` where those
        # are the function's keyword-arg defaults).
        fn_defaults: dict[str, object] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "make_config":
                a = node.args
                for name, dflt in zip(
                    [arg.arg for arg in a.args][len(a.args) - len(a.defaults):],
                    a.defaults, strict=False,
                ):
                    try:
                        fn_defaults[name] = ast.literal_eval(dflt)
                    except (ValueError, SyntaxError):
                        fn_defaults[name] = None
                break
        # Find the StageConfig(...) call inside make_config's return.
        call = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "StageConfig"
            ):
                call = node
                break
        assert call is not None, f"no StageConfig(...) in {fname}"
        kw: dict[str, object] = {}
        for k in call.keywords:
            if k.arg in ("seg_loss_fn", "resume_from", "output_dir"):
                continue  # callable / path — not part of the spec literals.
            if isinstance(k.value, ast.Name) and k.value.id in fn_defaults:
                kw[k.arg] = fn_defaults[k.value.id]  # e.g. epochs=epochs.
                continue
            try:
                kw[k.arg] = ast.literal_eval(k.value)
            except (ValueError, SyntaxError):
                kw[k.arg] = None
        merged = dict(defaults)
        merged.update(kw)
        rows.append({
            "name": merged["name"],
            "epochs": merged["epochs"],
            "cat_lambda": merged["cat_lambda"],
            "cat_sigma": merged["cat_sigma"],
            "use_qat": merged["use_qat"],
            "use_muon": merged["use_muon"],
            "adamw_lr": merged["adamw_lr"],
            "muon_lr": merged["muon_lr"],
            "seg_weight": merged["seg_weight"],
            "pose_weight": merged["pose_weight"],
            "ema_decay": merged["ema_decay"],
            "latent_lr_mult": merged["latent_lr_mult"],
            "muon_weight_decay": merged["muon_weight_decay"],
        })
    return rows


def test_8stage_spec_matches_torch_make_config_exactly():
    """Every MLX StageSpec field matches the torch ``make_config`` value EXACTLY.

    This is the FIDELITY gate: epochs, c1a_lambda, c1a_sigma, qat, the
    PR95-canonical optimizer flag, AdamW/Muon LRs, and the shared constants all
    match the proven torch reference. (No MLX needed — pure spec parity.)
    """
    from tac.mlx_pr95_port.curriculum import CURRICULA

    mlx_stages = CURRICULA["pr95_8stage"]
    torch_rows = _torch_stage_configs()
    assert len(mlx_stages) == len(torch_rows) == 8

    # The torch ``StageConfig`` does not carry a seg-loss NAME (it carries a
    # callable), so we map our seg_loss_form to the expected torch family per stage.
    expected_seg_form = {
        "stage1_v328_ce": "ce_seg_loss",
        "stage2_v331_softplus": "tau_softplus_seg_loss",
        "stage3_v332_smooth": "smooth_disagreement_seg_loss",
        "stage4_v332_qat": "smooth_disagreement_seg_loss",
        "stage5_c1a_l7": "l7_softplus_seg_loss",
        "stage6_lambda_sweep": "l7_softplus_seg_loss",
        "stage7_sigma_sweep": "l7_softplus_seg_loss",
        "stage8_muon_finetune": "l7_softplus_seg_loss",
    }

    for spec, row in zip(mlx_stages, torch_rows, strict=False):
        assert spec.name == row["name"], (spec.name, row["name"])
        assert spec.epochs == row["epochs"], (spec.name, "epochs")
        assert spec.cat_lambda == row["cat_lambda"], (spec.name, "cat_lambda")
        assert spec.cat_sigma == row["cat_sigma"], (spec.name, "cat_sigma")
        assert spec.use_qat == row["use_qat"], (spec.name, "use_qat")
        assert spec.use_muon_canonical == row["use_muon"], (spec.name, "use_muon")
        assert spec.adamw_lr == row["adamw_lr"], (spec.name, "adamw_lr")
        assert spec.muon_lr == row["muon_lr"], (spec.name, "muon_lr")
        assert spec.seg_weight == row["seg_weight"]
        assert spec.pose_weight == row["pose_weight"]
        assert spec.ema_decay == row["ema_decay"]
        assert spec.latent_lr_mult == row["latent_lr_mult"]
        assert spec.muon_weight_decay == row["muon_weight_decay"], (
            spec.name, "muon_weight_decay"
        )
        assert spec.seg_loss_form == expected_seg_form[spec.name]


def test_8stage_canonical_total_is_29650():
    """The canonical PR95 curriculum is the 29,650-epoch schedule (L14)."""
    from tac.mlx_pr95_port.curriculum import CURRICULA

    total = sum(s.epochs for s in CURRICULA["pr95_8stage"])
    assert total == 29650


def test_only_stage8_uses_muon_canonical():
    """PR95-canonical: ONLY stage 8 has use_muon=True (the rest are AdamW)."""
    from tac.mlx_pr95_port.curriculum import CURRICULA

    flags = [s.use_muon_canonical for s in CURRICULA["pr95_8stage"]]
    assert flags == [False] * 7 + [True], flags


def test_qat_active_stages_4_through_8():
    """QAT joins at stage 4 and stays on through stage 8 (L14)."""
    from tac.mlx_pr95_port.curriculum import CURRICULA

    qat = [s.use_qat for s in CURRICULA["pr95_8stage"]]
    assert qat == [False, False, False, True, True, True, True, True], qat


def test_c1a_active_stages_5_through_8():
    """C1a entropy joins at stage 5 (lambda 0.01) and stays on (lambda 0.02)."""
    from tac.mlx_pr95_port.curriculum import CURRICULA

    lams = [s.cat_lambda for s in CURRICULA["pr95_8stage"]]
    assert lams == [0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.02, 0.02], lams


# ===========================================================================
# (5) The scheduler / optimizer-schedule selector.
# ===========================================================================


def test_optimizer_schedule_resolution():
    from tac.mlx_pr95_port.curriculum import (
        CURRICULA,
        OPTIMIZER_SCHEDULE_MUON_THROUGHOUT,
        OPTIMIZER_SCHEDULE_PR95,
        resolve_use_muon,
    )

    stages = CURRICULA["pr95_8stage"]
    pr95 = [resolve_use_muon(s, OPTIMIZER_SCHEDULE_PR95) for s in stages]
    assert pr95 == [False] * 7 + [True], "PR95-faithful: only stage 8 uses Muon"
    muon = [resolve_use_muon(s, OPTIMIZER_SCHEDULE_MUON_THROUGHOUT) for s in stages]
    assert muon == [True] * 8, "muon_throughout: all stages use Muon"


def test_optimizer_schedule_unknown_raises():
    from tac.mlx_pr95_port.curriculum import CURRICULA, resolve_use_muon

    with pytest.raises(ValueError):
        resolve_use_muon(CURRICULA["pr95_8stage"][0], "not_a_schedule")


def test_build_curriculum_compression_preserves_structure():
    from tac.mlx_pr95_port.curriculum import build_pr95_8stage_curriculum

    base = build_pr95_8stage_curriculum()
    compressed = build_pr95_8stage_curriculum(total_epochs=80)
    assert len(compressed) == 8
    assert sum(s.epochs for s in compressed) <= 88  # ~80 with rounding
    # Structure (loss/qat/lambda/sigma/opt transitions) is unchanged.
    for b, c in zip(base, compressed, strict=False):
        assert b.name == c.name
        assert b.seg_loss_form == c.seg_loss_form
        assert b.use_qat == c.use_qat
        assert b.cat_lambda == c.cat_lambda
        assert b.use_muon_canonical == c.use_muon_canonical
        assert c.epochs >= 1


def test_build_curriculum_rejects_nonpositive():
    from tac.mlx_pr95_port.curriculum import build_pr95_8stage_curriculum

    with pytest.raises(ValueError):
        build_pr95_8stage_curriculum(epoch_scale=0)
    with pytest.raises(ValueError):
        build_pr95_8stage_curriculum(total_epochs=0)


# ===========================================================================
# (6) End-to-end: the scheduler drives a trainer through all 8 stages, carrying
#     weights + switching the seg-loss + arming QAT/C1a/sigma — NO-FAKE.
# ===========================================================================


class _ColorProtoSeg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        protos = torch.tensor(
            [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
            dtype=torch.float32,
        )
        self.c = nn.Conv2d(3, 5, 1)
        self.c.weight.data = protos.reshape(5, 3, 1, 1) / 128.0
        self.c.bias.data = -(protos**2).sum(1) / (2 * 128.0 * 128.0)

    def forward(self, x):
        return self.c(x * 255.0)


class _FrozenDNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segnet = _ColorProtoSeg()
        self.posenet = None

    def preprocess_input(self, bhwc):
        last = bhwc[:, -1].permute(0, 3, 1, 2)
        return last.repeat(1, 4, 1, 1), last / 255.0


def _build_frozen_dnet():
    dnet = _FrozenDNet().eval()
    for p in dnet.parameters():
        p.requires_grad = False
    return dnet


@skip_no_mlx
def test_curriculum_drives_pr95_reference_trainer_end_to_end():
    """The scheduler runs all 8 stages on the PR95-reference trainer without crash.

    NO-FAKE checks: (a) the bridge's seg-loss form is switched to the LAST stage's
    family after the run; (b) the trainer's mechanisms reflect the last stage (QAT
    on, C1a lambda 0.02, sigma 0.1); (c) each stage actually ran its epoch count;
    (d) weights moved across the curriculum (the loop is not inert).
    """
    from tac.local_acceleration.pr95_hnerv_mlx import HNeRVSyntheticTrainingBundleMLX
    from tac.mlx_pr95_port.curriculum import (
        OPTIMIZER_SCHEDULE_MUON_THROUGHOUT,
        build_pr95_8stage_curriculum,
    )
    from tac.mlx_pr95_port.mlx_trainer import (
        MlxScoreAwareConfig,
        MlxScoreAwareTrainer,
    )
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    dnet = _build_frozen_dnet()
    n, h, w = 4, 24, 32
    seg_tgt = torch.zeros(n, h, w, dtype=torch.long)
    seg_tgt[:, : h // 2] = 1
    bridge = TorchScorerBridge(
        dnet, seg_tgt, None, seg_loss_form="ce_seg_loss",
        scorer_hw=(h, w), eval_roundtrip=False,
    )
    bundle = HNeRVSyntheticTrainingBundleMLX(latent_count=n, base_channels=8, seed=0)
    cfg = MlxScoreAwareConfig(
        epochs=0, batch_size=2, eval_every=1, seed=0,
        scorer_hw=(h, w), eval_roundtrip=False, pose_enabled=False,
    )
    tr = MlxScoreAwareTrainer(bundle, bridge, cfg)

    w0 = np.asarray(
        dict(tree_flatten(bundle.trainable_parameters()))["decoder.stem.weight"]
    ).copy()

    stages = build_pr95_8stage_curriculum(total_epochs=24)
    result = tr.run_curriculum(
        stages, optimizer_schedule=OPTIMIZER_SCHEDULE_MUON_THROUGHOUT
    )

    # (a) bridge seg-loss switched to the final stage's l7_softplus family.
    from tac.score_aware_loop.live_segnet_loss import STAGE_SEG_LOSS_FNS

    assert bridge.seg_loss_fn is STAGE_SEG_LOSS_FNS["l7_softplus_seg_loss"]
    # (b) the trainer mechanisms reflect the final stage.
    assert tr.mechanisms.use_qat is True
    assert tr.mechanisms.cat_lambda == 0.02
    assert tr.mechanisms.cat_sigma == 0.1
    assert tr.mechanisms.sigma_weight_noise == 0.1
    # (c) all 8 stages ran their epoch counts.
    assert len(result.stages) == 8
    assert [s["epochs"] for s in result.stages] == [s.epochs for s in stages]
    # (d) the weights moved (the loop is not inert).
    w1 = np.asarray(
        dict(tree_flatten(bundle.trainable_parameters()))["decoder.stem.weight"]
    )
    assert np.abs(w1 - w0).max() > 0.0, "curriculum must actually move the weights"


@skip_no_mlx
def test_qat_vjp_restores_original_weights_in_bundle_REGRESSION():
    """REGRESSION: after the QAT-active vjp, the bundle holds the ORIGINAL weights.

    The traced ``mx.vjp`` forward installs the QAT-transformed weights into the
    bundle as a SIDE EFFECT. If the original primals are not restored after the
    vjp, the bundle is left holding the QUANTIZED weights and the subsequent
    optimizer step updates the quantized master (corruption). This checks the
    bundle's live weight is EXACTLY the pre-vjp weight (and NOT its fake-quant) —
    it FAILS if the restore-primals fix regresses.
    """
    from tac.local_acceleration.pr95_hnerv_mlx import HNeRVSyntheticTrainingBundleMLX
    from tac.mlx_pr95_port.curriculum_mechanisms import StageMechanisms
    from tac.mlx_pr95_port.mlx_losses import fake_quantize_mlx
    from tac.mlx_pr95_port.mlx_trainer import (
        MlxScoreAwareConfig,
        MlxScoreAwareTrainer,
    )
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    dnet = _build_frozen_dnet()
    n, h, w = 2, 24, 32
    seg_tgt = torch.zeros(n, h, w, dtype=torch.long)
    seg_tgt[:, : h // 2] = 1
    bridge = TorchScorerBridge(
        dnet, seg_tgt, None, seg_loss_form="ce_seg_loss",
        scorer_hw=(h, w), eval_roundtrip=False,
    )
    bundle = HNeRVSyntheticTrainingBundleMLX(latent_count=n, base_channels=8, seed=0)
    tr = MlxScoreAwareTrainer(
        bundle, bridge,
        MlxScoreAwareConfig(epochs=0, batch_size=2, scorer_hw=(h, w),
                            eval_roundtrip=False, pose_enabled=False, seed=0),
    )
    tr.mechanisms = StageMechanisms(use_qat=True)
    key = "decoder.blocks.0.conv.weight"
    w_before = np.asarray(dict(tree_flatten(bundle.trainable_parameters()))[key]).copy()
    quantized = np.asarray(fake_quantize_mlx(mx.array(w_before)))
    # Sanity: the weight is non-grid (fake-quant actually changes it), so the
    # buggy-vs-fixed states are distinguishable.
    assert np.abs(quantized - w_before).max() > 1e-6

    # Drive ONLY the vjp (no optimizer step), then inspect the bundle's live weight.
    indices = mx.array(np.array([0, 1]).astype(np.int32))
    render = tr._render(indices)
    mx.eval(render)
    res = bridge.loss_and_pixel_grad(render, torch.tensor([0, 1]))
    tr._vjp_grads(indices, res.pixel_cotangent)

    w_live = np.asarray(dict(tree_flatten(bundle.trainable_parameters()))[key])
    # FIXED: the bundle holds the ORIGINAL float weights (bit-identical).
    assert np.array_equal(w_live, w_before), (
        "QAT vjp left the bundle holding NON-original weights (restore-primals "
        "regressed)"
    )
    # And it is NOT the quantized version (the discriminating check).
    assert not np.allclose(w_live, quantized, atol=1e-7)


@skip_no_mlx
def test_curriculum_switches_seg_loss_per_stage():
    """``configure_stage`` switches the bridge seg-loss to each stage's family."""
    from tac.local_acceleration.pr95_hnerv_mlx import HNeRVSyntheticTrainingBundleMLX
    from tac.mlx_pr95_port.curriculum import (
        OPTIMIZER_SCHEDULE_PR95,
        build_pr95_8stage_curriculum,
    )
    from tac.mlx_pr95_port.mlx_trainer import (
        MlxScoreAwareConfig,
        MlxScoreAwareTrainer,
    )
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge
    from tac.score_aware_loop.live_segnet_loss import STAGE_SEG_LOSS_FNS

    dnet = _build_frozen_dnet()
    n, h, w = 2, 24, 32
    seg_tgt = torch.zeros(n, h, w, dtype=torch.long)
    bridge = TorchScorerBridge(
        dnet, seg_tgt, None, seg_loss_form="ce_seg_loss",
        scorer_hw=(h, w), eval_roundtrip=False,
    )
    bundle = HNeRVSyntheticTrainingBundleMLX(latent_count=n, base_channels=8, seed=0)
    tr = MlxScoreAwareTrainer(
        bundle, bridge,
        MlxScoreAwareConfig(epochs=0, batch_size=2, scorer_hw=(h, w),
                            eval_roundtrip=False, pose_enabled=False),
    )
    stages = build_pr95_8stage_curriculum(total_epochs=8)
    expected = [
        "ce_seg_loss", "tau_softplus_seg_loss", "smooth_disagreement_seg_loss",
        "smooth_disagreement_seg_loss", "l7_softplus_seg_loss",
        "l7_softplus_seg_loss", "l7_softplus_seg_loss", "l7_softplus_seg_loss",
    ]
    for spec, form in zip(stages, expected, strict=False):
        tr.configure_stage(spec, optimizer_schedule=OPTIMIZER_SCHEDULE_PR95)
        assert bridge.seg_loss_fn is STAGE_SEG_LOSS_FNS[form]
        # PR95-faithful schedule: only stage 8 uses Muon.
        assert tr.opt_config.use_muon == (spec.name == "stage8_muon_finetune")
