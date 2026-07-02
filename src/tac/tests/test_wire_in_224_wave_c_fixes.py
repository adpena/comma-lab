# SPDX-License-Identifier: MIT
"""#224 FIX-ALL Wave C behavior + self-protect tests.

FIX-1 (LAUNCH-BLOCKER): MLX ``optim.AdamW`` ``bias_correction`` DEFAULTS FALSE => the all-levers
small-n beta2 (0.9999999) leaves ``sqrt(v)`` ~sqrt(1-beta2) too small at step 1 => ~100x effective-LR
blowup => divergence. The fix threads ``bias_correction=_adam_bias_correction_for(adam_beta2)`` into
BOTH AdamW constructions (main + stage-transition moment-reset), gated ON only off the 0.999 default
(so the default path is BYTE-IDENTICAL). These tests assert the OPTIMIZER NUMERICS (step-1 ratio back
to O(1), not ~100x) with real MLX AdamW + the source-level self-protect (both sites thread the gate).

FIX-2 (OPTIMAL-AA switch): the all-levers config now emits ``--render-aa supersample --aa-supersample 2
--aa-self-orient-fine-mode full`` (the observation-correct AA; probe-confirmed ~63GB n600 peak, ~65GB
headroom on the 128GB M5 Max) and it COEXISTS with ``--structured-init`` (the guard that used to reject
that combination is relaxed: base-grid structured-init + shared coord-INR weights evaluated at fine
coords). These tests assert the emitted argv + the guard removal + byte-identical-off.

Advisory [macOS-MLX research-signal]; pointer 0.19110 UNMOVED. Every claim here is optimizer-numerics /
config-argv / source-structure — NOT a scorer d_seg claim (per NO-FAKE #3, no synthetic-input score).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_HAS_MLX = importlib.util.find_spec("mlx") is not None

_TRAINER_SRC = (
    REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"
).read_text()


def _load_levelset_trainer():
    name = "train_levelset_witness_realized_through_R_mlx"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# FIX-1 — the bias-correction gate predicate
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_MLX, reason="mlx not available (trainer import)")
def test_adam_bias_correction_gate_default_off_highbeta2_on():
    mod = _load_levelset_trainer()
    f = mod._adam_bias_correction_for
    # DEFAULT path stays MLX-default (False) => byte-identical-off
    assert f(0.999) is False
    assert f(0.999 + 1e-12) is False          # within tolerance of the default
    # all-levers high-beta2 path => bias correction REQUIRED
    assert f(0.9999999) is True
    assert f(0.99999988) is True              # the derived small-n optimum
    assert f(0.99) is True                    # any non-default beta2 => on


# ---------------------------------------------------------------------------
# FIX-1 — real MLX AdamW step-1 update magnitude: bias correction removes the
# ~100x blowup at high beta2. This is the CORE of the launch-blocker.
# ---------------------------------------------------------------------------
def _step1_update_norm(beta2: float, bias_correction: bool) -> float:
    import mlx.core as mx
    import mlx.optimizers as optim

    p = {"w": mx.zeros((16,))}
    g = {"w": mx.ones((16,))}                 # fixed unit gradient
    opt = optim.AdamW(learning_rate=1e-3, betas=[0.9, beta2], weight_decay=0.0,
                      bias_correction=bias_correction)
    new = opt.apply_gradients(g, p)
    mx.eval(new)
    return float(mx.sqrt(mx.sum((new["w"] - p["w"]) ** 2)).item())


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not available")
def test_beta2_step1_bias_correction_removes_100x_blowup():
    default = _step1_update_norm(0.999, False)          # the UNCHANGED default path (MLX default)
    fixed = _step1_update_norm(0.9999999, True)         # the FIXED all-levers path
    bug = _step1_update_norm(0.9999999, False)          # the BUG the fix removes

    # (a) THE FIX: the all-levers step-1 update is O(1) relative to the default path — NOT ~100x.
    fixed_ratio = fixed / default
    assert 0.05 < fixed_ratio < 20.0, f"fixed step-1 ratio not O(1): {fixed_ratio}"

    # (b) THE BUG (documents what the fix removes): without bias correction, high beta2 blows the
    #     step-1 update up ~100x vs the default beta2 (same no-bc path).
    bug_ratio = bug / default
    assert bug_ratio > 30.0, f"expected ~100x blowup without bias correction, got {bug_ratio}"

    # (c) with bias correction, the step-1 update is ~beta2-invariant (both ~ lr*sign(g)).
    bc_lo = _step1_update_norm(0.999, True)
    bc_hi = _step1_update_norm(0.9999999, True)
    assert 0.5 < (bc_hi / bc_lo) < 2.0


# ---------------------------------------------------------------------------
# FIX-1 — self-protect: BOTH AdamW constructions that carry the adam_beta2 betas
# MUST thread the bias_correction gate (main + stage-transition moment-reset).
# ---------------------------------------------------------------------------
def test_both_adam_beta2_sites_thread_bias_correction():
    betas_sites = _TRAINER_SRC.count('betas=[0.9, float(getattr(args, "adam_beta2"')
    # both AdamW constructions must thread `bias_correction=` (main via _adam_bc local, reset inline);
    # the helper def uses `-> bool` (no `bias_correction=`), so the count == the # of gated AdamW sites.
    gated_sites = _TRAINER_SRC.count("bias_correction=")
    calls = _TRAINER_SRC.count("_adam_bias_correction_for(")  # helper def + 2 call sites
    assert betas_sites == 2, f"expected 2 adam_beta2 AdamW sites, found {betas_sites}"
    assert gated_sites == 2, (
        f"expected both adam_beta2 AdamW sites to thread bias_correction=, found {gated_sites}")
    assert calls >= 3, f"expected helper def + 2 call sites, found {calls}"


# ---------------------------------------------------------------------------
# FIX-2 — the supersample + structured-init incompatibility guard is RELAXED.
# ---------------------------------------------------------------------------
def test_supersample_structured_init_guard_relaxed():
    assert "--render-aa supersample is incompatible with --structured-init" not in _TRAINER_SRC, (
        "the supersample+structured-init fail-closed guard should be relaxed (Wave C FIX-2)")
    # the wiring note that documents the relaxation is present
    assert "render_aa_supersample_structured_init" in _TRAINER_SRC


# ---------------------------------------------------------------------------
# FIX-2 — the all-levers config emits supersample-full + it coexists with
# structured-init; the DEFAULT config emits NEITHER (byte-identical-off).
# ---------------------------------------------------------------------------
def _all_levers_flags():
    from tac import witness_autoconfig as wac
    cfg = wac.derive_config("experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
                            num_pairs=600, epochs=1000, all_levers=True)
    return dict(cfg.to_trainer_flags("out")), wac


def _default_flags():
    from tac import witness_autoconfig as wac
    cfg = wac.derive_config("experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
                            num_pairs=600, epochs=1000, all_levers=False)
    return dict(cfg.to_trainer_flags("out"))


def test_all_levers_base_dict_is_supersample_full():
    from tac import witness_autoconfig as wac
    base = wac._all_levers_base(300)
    assert base["render_aa"] == "supersample"
    assert base["aa_supersample"] == 2
    assert base["aa_self_orient_fine_mode"] == "full"
    assert "aa_ipe_footprint" not in base  # ipe-only field dropped


def test_all_levers_emits_supersample_full_coexisting_with_structured_init():
    flags, _wac = _all_levers_flags()
    assert flags["--render-aa"] == "supersample"
    assert flags["--aa-supersample"] == 2
    assert flags["--aa-self-orient-fine-mode"] == "full"
    # coexistence: structured-init + supersample both present (the relaxed-guard payoff)
    assert "--structured-init" in flags
    assert "--self-orient" in flags
    # FIX-1: the high-beta2 is emitted on the all-levers path
    assert abs(float(flags["--adam-beta2"]) - 0.9999999) < 1e-12


def test_default_config_byte_identical_off_no_aa_no_beta2():
    flags = _default_flags()
    # the AA + adam-beta2 overrides are all-levers-ONLY => default path is byte-identical-off
    for k in ("--render-aa", "--aa-supersample", "--aa-self-orient-fine-mode", "--adam-beta2",
              "--aa-ipe-footprint"):
        assert k not in flags, f"default (all_levers=False) must NOT emit {k}"
