# Council Audit — Lane DARTS-S V1 SegMap Training Freeze (2026-04-29 PM)

**Status**: ROOT CAUSE IDENTIFIED. Confidence: **HIGH**.

**Bug location**: `src/tac/segmap_renderer.py:281` — `up_u8 = up.clamp(0, 255).round()` inside `_eval_roundtrip_chain`. The `.round()` operator has zero gradient, which severs the gradient chain from loss back to all SegMap parameters. The model never learned because every `loss.backward()` returned an all-zero gradient on every trainable parameter.

**One-sentence root cause**: `_eval_roundtrip_chain` uses bare `tensor.round()` (which has zero gradient in PyTorch) instead of a Straight-Through Estimator (STE), so the entire training-time roundtrip → scorer → loss path is gradient-disconnected from the SegMap renderer.

---

## 1. Executive Summary

The Lane DARTS-S V1 sweep ran 400 epochs of `optimizer.step()` on a SegMap renderer (94,419 trainable params) and the loss/seg_dist/pose_dist/kl_aux were IDENTICAL to four decimal places across every epoch. The display NaN was a separate, already-fixed printer bug (Check 85 STRICT). The deeper bug — and the one that wasted 5 hours of GPU time — is that `_eval_roundtrip_chain` (the function that simulates the contest-eval `384→874→uint8→384` resize chain in differentiable form) calls bare `up.clamp(0, 255).round()`. PyTorch's `torch.round()` has zero gradient almost everywhere (verified empirically below), so the rendered output is gradient-disconnected from `rt_btchw`, which is the ONLY tensor handed to the scorer. Every loss term — pose MSE, SegNet CE, KL distill — flows through `rt_btchw`, so every loss term has zero gradient w.r.t. the SegMap renderer's parameters. AdamW with `lr=1e-3` and `weight_decay=1e-4` then performs 400 step()s on all-zero grads, which only nudges weights by the weight-decay shrinkage of `(1 − lr·wd)^400 ≈ 0.99996`. The model is effectively frozen at initialization.

This bug ALSO silently broke any earlier SegMap training run — Lane SC++, Lane SA-v2, Lane SO, Lane MM (insofar as it used the same trainer). The only reason the bug wasn't caught earlier is that the 4 unit tests in `test_segmap_renderer.py` (`test_segmap_trainer_train_epoch_loss_finite`, `test_train_epoch_chunked_path_runs_and_steps_optimizer`) use **MockPoseNet/MockSegNet whose preprocess paths skip the eval_roundtrip chain** — the unit tests assert `pre_param != post_param` but the assertion succeeds via the AdamW `weight_decay` shrinkage of the initial weights, NOT because gradients flowed. The test passes vacuously.

---

## 2. Code-Level Evidence

### 2.1 The smoking gun — `src/tac/segmap_renderer.py:259-285`

```python
def _eval_roundtrip_chain(
    rgb_pair_btchw: torch.Tensor,
    noise_std: float = 0.5,
) -> torch.Tensor:
    ...
    flat = rgb_pair_btchw.reshape(b * t, c, h, w)
    up = F.interpolate(flat, size=CAMERA_SIZE[::-1], mode="bicubic", align_corners=False)
    up_u8 = up.clamp(0, 255).round()  # ← LINE 281 — STE-friendly proxy for uint8 cast
    if noise_std > 0:
        up_u8 = up_u8 + noise_std * torch.randn_like(up_u8)
    back = F.interpolate(up_u8, size=(h, w), mode="bicubic", align_corners=False)
    return back.clamp(0, 255).reshape(b, t, c, h, w)
```

The comment claims `.round()` is "STE-friendly". It is NOT. In PyTorch, `torch.round()`'s autograd derivative is the derivative of the floor/round step function, which is zero almost everywhere. Verified:

```
$ python -c "
import torch
x = torch.tensor([1.3], requires_grad=True)
y = x.round()
y.backward()
print('grad:', x.grad)
"
grad: tensor([0.])
```

End-to-end through the same chain `_eval_roundtrip_chain` builds:

```
$ python -c "
import torch
import torch.nn.functional as F
x = torch.randn(1, 3, 4, 4, requires_grad=True)
up = F.interpolate(x, size=(8, 8), mode='bicubic', align_corners=False)
up_u8 = up.clamp(0, 255).round()
noise = 0.5 * torch.randn_like(up_u8)
mixed = up_u8 + noise
back = F.interpolate(mixed, size=(4, 4), mode='bicubic', align_corners=False)
loss = back.sum()
loss.backward()
print('x.grad max abs:', x.grad.abs().max().item())
print('x.grad nonzero count:', (x.grad != 0).sum().item())
"
x.grad max abs: 0.0
x.grad nonzero count: 0
```

`x.grad` is exactly zero everywhere. Adding the Gaussian noise after rounding does NOT restore the gradient because `randn_like` produces a fresh leaf tensor with no dependency on `x`.

### 2.2 The chain through which the dead gradient propagates

In `SegMapTrainer.train_epoch` (`src/tac/segmap_renderer.py:467-541`):

```python
rendered = self.model(masks_flat, frame_indices)               # has grad ✓
rendered_btchw = rendered.reshape(mb, t, 3, h, w)              # has grad ✓
rt_btchw = _eval_roundtrip_chain(rendered_btchw, ...)          # gradient KILLED here
posenet_out, segnet_out = scorer_forward_pair(rt_btchw, ...)   # zero grad to renderer
...
pose_dist = pose_diff_sq.mean()                                # zero grad to renderer
seg_dist  = seg_ce_per.mean()                                  # zero grad to renderer
loss = self.config.segnet_loss_weight * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
# kl_distill branch (line 530-541) ALSO uses rt_btchw → zero grad to renderer
(loss / n_minibatches_estimate).backward()                     # writes zeros to .grad
```

EVERY downstream tensor — `posenet_out`, `segnet_out`, `pose_dist`, `seg_dist`, `kl_loss`, `loss` — has `grad_fn != None` (so `.backward()` does not raise) but the gradient flowing back through `rt_btchw` is exactly zero, so the gradient delivered to `model.parameters()` is exactly zero.

### 2.3 Why AdamW still appears to "step"

`SegMapTrainer.train_epoch` calls `optimizer.step()` once per epoch. AdamW with all-zero `grad`:
- First-moment update: `m = β₁·m + (1−β₁)·0 = β₁·m` — so `m` decays toward 0.
- Second-moment update: `v = β₂·v + (1−β₂)·0² = β₂·v` — so `v` also decays toward 0.
- Parameter update: `param -= lr · m / (√v + eps)` — with both m and v starting at 0, this is exactly zero.
- Weight decay (decoupled, AdamW): `param -= lr · weight_decay · param` — this DOES move the param, but by an absolutely tiny amount: `param × (1 − 1e-3 × 1e-4) ≈ param × 0.9999999`.

Over 400 epochs: `param × 0.9999999^400 ≈ param × 0.99996` — the weights shrink by 0.004%. This is below the 4-decimal-place precision of the displayed metrics, so `pose_dist`, `seg_dist`, `kl_aux`, and `loss` print as IDENTICAL across epochs — exactly what the user observed.

### 2.4 Why the unit tests didn't catch this

`test_segmap_trainer_train_epoch_loss_finite` (line 202-229) and `test_train_epoch_chunked_path_runs_and_steps_optimizer` (line 312-347) BOTH only assert `not torch.equal(pre_param, post_param)`. With `weight_decay=1e-4` and `lr=1e-3`, that assertion passes via the weight-decay shrinkage even when the data-loss gradient is exactly zero. The tests are vacuous.

---

## 3. Hypothesis Ranking

The audit prompt enumerated 6 candidate root causes. Code-level evidence:

| # | Hypothesis | Verdict | Evidence |
|---|-----------|---------|----------|
| 1 | Render output detached from model params (`.detach()` or `torch.no_grad()` in SegMap.forward / mask_grayscale_lut) | **PRIMARY ROOT CAUSE — confirmed in `_eval_roundtrip_chain`, NOT `SegMap.forward`** | The detach-equivalent is `up.clamp(0, 255).round()` at `src/tac/segmap_renderer.py:281`. SegMap.forward itself is gradient-clean. The mask-grayscale-lut is NOT in the path for the kl_distill variant (one-hot pairs go in directly per `train_segmap.py:222`). |
| 2 | Gradients are zero from saturated scorer (pose_dist=158 → saturated nonlinearities) | RULED OUT | Even saturated PoseNet would deliver SOME gradient. The test at §2.1 shows the gradient is mathematically zero before reaching the scorer. |
| 3 | EMA shadow shadowing the model (EMA written back into model after step) | RULED OUT | `EMA.update` (`src/tac/training.py:347-364`) only writes to `self.shadow`. `EMA.apply` is never called inside `train_epoch`; only `export_inference_state_dict` calls it (and restores afterwards). The live model weights are never overwritten by EMA. |
| 4 | Frame-index FiLM conditioning produces identical output per (mask, idx) | RULED OUT | `frame_affine_embedding.weight` is initialized `N(0, 0.01)` so embeddings differ; `tanh(small)` ≈ small but nonzero. SegMap output is NOT constant w.r.t. inputs. |
| 5 | `kl_distill` variant config bug (kl_distill_weight or temperature_start zeroes the loss) | RULED OUT | `temperature_start=2.0`, `0.002 * kl_loss` is finite (4.48 reported). The KL branch ADDS to loss correctly. The KL aux ALSO routes through `rt_btchw` (line 534-535), so even WITHOUT the round() bug, this would be a weak signal — but with the round() bug, it's also dead. |
| 6 | SegMap.forward returns constant output | RULED OUT | `_make_tiny_segmap` test at line 105-119 shows distinct output per input. SegMap.forward is fine. |

The fact that hypothesis #1 manifests in `_eval_roundtrip_chain` rather than the SegMap forward path is what makes this bug subtle: the canonical "look at the model forward for `.detach()`" review pattern misses it. The dead-gradient point is in a SHARED utility function that every callsite uses.

### 3.1 Cross-impact — does this bug affect Lane G v3?

**No.** Lane G v3 trained the legacy renderer via `experiments/train_distill.py` → `Trainer.train()` in `src/tac/training.py`. That training path uses the canonical `Uint8STE.apply` (`src/tac/quantization.py:189`) for its uint8 roundtrip, which is a proper `torch.autograd.Function` with explicit identity-backward. Lane G v3's gradient flow is intact — that's why it converged to score 1.05.

The DARTS-S sweep is the FIRST production lane to use `SegMapTrainer.train_epoch` end-to-end with the contest scorers. It exposed the bug that has been latent since the SegMap trainer was added.

### 3.2 Other lanes affected

Any lane that uses `SegMapTrainer.train_epoch` is affected:
- Lane SC++ (Selfcomp clone) — would have produced garbage if it ever shipped.
- Lane SA-v2 (SegMap with augmentation) — same.
- Lane SO (SegMap optimization) — same.
- Lane MM v2 (FALSIFIED at 2.63 per `project_lane_mm_v2_landed_2_63_falsified_20260429.md`) — was almost certainly contaminated by this bug. The "encoder-only grayscale-LUT FALSIFIED" verdict needs to be REVISITED after the fix lands. Lane MM v2 may not actually be falsified; it may have just never trained.

---

## 4. Specific Fix Patch

### 4.1 The fix (3 lines changed in `src/tac/segmap_renderer.py`)

Replace `_eval_roundtrip_chain` (lines 259-285) so the uint8 roundtrip uses a proper STE. The `Uint8STE` class already exists in `src/tac/quantization.py:189` and is the canonical pattern.

```python
# src/tac/segmap_renderer.py

from tac.quantization import uint8_ste  # ← NEW IMPORT (top of file)

def _eval_roundtrip_chain(
    rgb_pair_btchw: torch.Tensor,
    noise_std: float = 0.5,
) -> torch.Tensor:
    """Apply the canonical 384 -> 874 -> uint8 -> 384 contest eval chain.

    Mirrors the established roundtrip pattern from src/tac/losses.py / training.
    The bicubic resize to camera-H + uint8 cast + bicubic-back to scorer-H
    simulates the lossy decode of the ``.raw`` rgb24 the contest evaluator
    sees. Without this step the proxy-auth gap is 2-11x on PoseNet
    (feedback_proxy_auth_math_useless).

    DARTS-S incident fix (2026-04-29): the prior implementation called
    `up.clamp(0, 255).round()`, which has ZERO gradient (verified empirically).
    Every SegMap training run silently received zero gradient on the rendered
    output, freezing the model at initialization. Replaced with the canonical
    `uint8_ste` (Uint8STE autograd.Function from tac.quantization) which uses
    saturation-aware identity-backward, the proven pattern Lane G v3 uses.
    """
    b, t, c, h, w = rgb_pair_btchw.shape
    flat = rgb_pair_btchw.reshape(b * t, c, h, w)
    up = F.interpolate(flat, size=CAMERA_SIZE[::-1], mode="bicubic", align_corners=False)
    up_u8 = uint8_ste(up)  # ← STE: forward = clamp(round(x), 0, 255), backward = identity in [0,255], 0 outside
    if noise_std > 0:
        up_u8 = up_u8 + noise_std * torch.randn_like(up_u8)
    back = F.interpolate(up_u8, size=(h, w), mode="bicubic", align_corners=False)
    return back.clamp(0, 255).reshape(b, t, c, h, w)
```

Note: the trailing `back.clamp(0, 255)` retains its gradient (clamp gradient is well-defined: 1 inside, 0 outside the bounds), so it does NOT need to be replaced.

### 4.2 Tightening the unit tests so this bug class can never resurrect

The existing `test_segmap_trainer_train_epoch_loss_finite` and `test_train_epoch_chunked_path_runs_and_steps_optimizer` are vacuous because they only check that params changed — and AdamW weight_decay always changes them. Add a NEW test that asserts gradient flows back to the renderer:

```python
# src/tac/tests/test_segmap_renderer.py

def test_segmap_trainer_train_epoch_grad_reaches_renderer() -> None:
    """REGRESSION: every backward call from train_epoch MUST deliver
    nonzero gradient to at least one SegMap parameter. The DARTS-S
    incident (2026-04-29) had `_eval_roundtrip_chain` call bare
    tensor.round() which silently zeroed all gradients to the renderer;
    `assert not torch.equal(pre_param, post_param)` passes vacuously
    via AdamW weight_decay shrinkage even when the data-loss gradient
    is exactly zero. This test asserts the gradient itself, not the
    post-step weights.
    """
    torch.manual_seed(0)
    cfg = _make_eval_roundtrip_true_config()
    h = SEGMAP_INPUT_SIZE[1] // 16
    w = SEGMAP_INPUT_SIZE[0] // 16
    model = _make_tiny_segmap()
    # Disable weight_decay so the only path to non-zero param.grad is the
    # data loss — nothing else.
    cfg = cfg.model_copy(update={"weight_decay": 0.0})
    trainer = SegMapTrainer(model, cfg, _MockPoseNet(), _MockSegNet(), device="cpu")

    b, t = 1, 2
    masks = F.softmax(torch.randn(b, t, 5, h, w), dim=2)
    gt = torch.rand(b, t, h, w, 3) * 255.0

    # Run one chunk; before optimizer.step(), inspect param.grad directly.
    trainer.optimizer.zero_grad(set_to_none=True)
    masks_flat = masks.reshape(b * t, 5, h, w)
    frame_indices = torch.arange(0, b * t, dtype=torch.long)
    rendered = trainer.model(masks_flat, frame_indices).reshape(b, t, 3, h, w)
    from tac.segmap_renderer import _eval_roundtrip_chain
    rt = _eval_roundtrip_chain(rendered, noise_std=0.5)
    posenet_out, segnet_out = trainer.posenet(trainer.posenet.preprocess_input(rt)), trainer.segnet(trainer.segnet.preprocess_input(rt))
    loss = posenet_out["pose"][..., :6].pow(2).mean() + F.cross_entropy(
        segnet_out, segnet_out.argmax(dim=1)
    )
    loss.backward()

    # Every TRAINABLE renderer parameter must have a non-zero gradient.
    nz = sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().max().item() > 0.0
    )
    total = sum(1 for p in model.parameters() if p.requires_grad)
    assert nz > 0, (
        f"DARTS-S regression: 0/{total} renderer params received non-zero "
        f"gradient. _eval_roundtrip_chain is gradient-disconnected."
    )
```

Also UPDATE `test_segmap_trainer_train_epoch_loss_finite` to add gradient inspection:

```python
def test_segmap_trainer_train_epoch_loss_finite() -> None:
    ...
    # Existing pre/post param check stays.
    # ADD: verify the gradient itself was non-zero (not just the post-step weights).
    grad_norms = [p.grad.abs().max().item() for p in model.parameters() if p.grad is not None]
    assert max(grad_norms) > 0, "all parameter gradients zero — roundtrip disconnected?"
```

### 4.3 Operational fix — also add a frozen-loss watchdog

Per the Check 85 follow-up note in `feedback_check_85_metric_key_display_bug_landed_20260429.md`:

> 3. **Frozen-loss watchdog** — if seg_dist/pose_dist deltas < 1e-6 for >50 epochs AND loss > 0, kill — model isn't learning. This would have caught Lane DARTS-S V1's frozen training in ~30 min instead of 5h.

Add to `experiments/train_segmap.py` after the epoch metric collection:

```python
# Frozen-loss watchdog: if seg_dist + pose_dist haven't moved by ≥1e-5
# in the last 50 epochs (and loss > 0), abort — gradient is dead.
if epoch >= 50:
    recent = history[-50:]
    seg_delta = max(h["seg_dist"] for h in recent) - min(h["seg_dist"] for h in recent)
    pose_delta = max(h["pose_dist"] for h in recent) - min(h["pose_dist"] for h in recent)
    if seg_delta < 1e-5 and pose_delta < 1e-5 and epoch_metrics["loss"] > 0:
        raise RuntimeError(
            f"FROZEN-LOSS WATCHDOG: epoch={epoch} pose_dist/seg_dist "
            f"unchanged for 50 epochs (Δseg={seg_delta:.2e}, Δpose={pose_delta:.2e}). "
            f"Likely cause: gradient disconnect (DARTS-S incident 2026-04-29). "
            f"Inspect _eval_roundtrip_chain / SegMap.forward / scorer wiring."
        )
```

---

## 5. Test Plan

**File**: `src/tac/tests/test_segmap_renderer.py`
**New test**: `test_segmap_trainer_train_epoch_grad_reaches_renderer` (see §4.2 above)
**Updated test**: `test_segmap_trainer_train_epoch_loss_finite` (add gradient-norm assertion)

**File**: `src/tac/tests/test_eval_roundtrip_gradient.py` (NEW)
**Test**: `test_eval_roundtrip_chain_passes_gradient` — explicit unit test on `_eval_roundtrip_chain` that asserts `x.grad.abs().max() > 0` after `loss = chain(x).sum(); loss.backward()`. This is the ATOMIC test for the bug. Pseudo-code:

```python
import torch
from tac.segmap_renderer import _eval_roundtrip_chain

def test_eval_roundtrip_chain_passes_gradient():
    x = torch.rand(1, 2, 3, 24, 32, requires_grad=True) * 255.0
    out = _eval_roundtrip_chain(x, noise_std=0.0)  # noise_std=0 to make it deterministic
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().max() > 0, (
        "DARTS-S regression: _eval_roundtrip_chain gradient is zero. "
        "Likely .round() instead of uint8_ste()."
    )
```

---

## 6. Preflight Check Proposal — STRICT Check 86

**Name**: `check_no_bare_round_in_eval_roundtrip_chain`
**Bug class extinguished**: gradient-disconnected uint8 roundtrip in any training-time loss path.

**Pseudocode** (to live in `src/tac/preflight.py` next to Check 85):

```python
def check_no_bare_round_in_eval_roundtrip_chain(
    repo_root: Path,
    strict: bool = False,
) -> list[str]:
    """STRICT 86: any function whose name contains 'roundtrip', 'eval_round',
    or 'simulate_uint8' must NOT call bare `.round()` on a tensor that is
    later returned. Use `tac.quantization.uint8_ste` (proper STE) instead.

    Bug class: PyTorch's torch.round has zero gradient. Bare .round() in a
    training-time path silently zeros gradients to the upstream tensor.
    DARTS-S incident 2026-04-29 wasted 5h GPU + $1.41 to discover.
    """
    violations = []
    targets = ["roundtrip", "eval_round", "simulate_uint8", "_to_uint8"]
    for py in (repo_root / "src" / "tac").rglob("*.py"):
        src = py.read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not any(t in node.name.lower() for t in targets):
                continue
            for inner in ast.walk(node):
                # Match `<expr>.round()` calls
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                    if inner.func.attr == "round" and not inner.args:
                        # Allow if the line has a `# STE_OVERRIDE_OK` waiver
                        line = src.split("\n")[inner.lineno - 1]
                        if "STE_OVERRIDE_OK" not in line:
                            violations.append(
                                f"{py.relative_to(repo_root)}:{inner.lineno} "
                                f"function `{node.name}` calls bare `.round()` — use "
                                f"`tac.quantization.uint8_ste` for proper STE. "
                                f"(DARTS-S incident; CLAUDE.md non-negotiable)"
                            )
    if strict and violations:
        raise PreflightError("\n".join(violations))
    return violations
```

Promotion path: ship `strict=False` initially, fix every legacy violation, then flip to `strict=True` in `preflight_all()`. Live STRICT count goes 85 → 86.

---

## 7. Standalone Repro Script

Written to `/tmp/repro_darts_s_freeze.py`:

```python
#!/usr/bin/env python
"""Reproduce the DARTS-S V1 training freeze in 30 lines, no GPU needed.

Demonstrates that with the BUGGY `_eval_roundtrip_chain` (bare `.round()`),
SegMap params receive zero gradient and AdamW only moves them via weight_decay
shrinkage. After the FIX (use `uint8_ste`), gradients flow correctly.

Run: python /tmp/repro_darts_s_freeze.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Projects/pact/src"))

import torch
import torch.nn.functional as F
from tac.segmap_renderer import SegMap, _eval_roundtrip_chain
from tac.quantization import uint8_ste

torch.manual_seed(0)
device = "cpu"

# Tiny SegMap matching the DARTS-S "default" config shape ratio.
model = SegMap(hidden=24, block_hidden=24, num_blocks=8, max_frame_index=16).to(device)
trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"trainable params: {sum(p.numel() for p in trainable_params):,}")

opt = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)

# Synthetic mask + frame index batch matching one mini-batch of the real run.
b, t = 8, 2
h, w = 24, 32  # downscale of 384x512 by 16
masks = F.softmax(torch.randn(b * t, 5, h, w, device=device), dim=1) * 0.2
frame_indices = torch.arange(b * t, device=device, dtype=torch.long)

print("\n=== buggy path (bare .round() in roundtrip) ===")
for step in range(5):
    opt.zero_grad(set_to_none=True)
    rendered = model(masks, frame_indices).reshape(b, t, 3, h, w)
    # Buggy roundtrip — exactly what segmap_renderer.py:281 does today.
    flat = rendered.reshape(b * t, 3, h, w)
    up = F.interpolate(flat, size=(h * 4, w * 4), mode="bicubic", align_corners=False)
    up_u8 = up.clamp(0, 255).round()  # ← BUG: zero gradient here
    back = F.interpolate(up_u8, size=(h, w), mode="bicubic", align_corners=False)
    rt = back.clamp(0, 255).reshape(b, t, 3, h, w)
    loss = rt.pow(2).mean()  # any downstream loss using rt has zero grad
    loss.backward()
    grad_norms = [p.grad.abs().max().item() if p.grad is not None else 0.0
                  for p in trainable_params]
    pre_param0 = trainable_params[0].detach().clone()
    opt.step()
    delta = (trainable_params[0] - pre_param0).abs().max().item()
    print(f"  step={step} loss={loss.item():.4f} max(|grad|)={max(grad_norms):.2e} "
          f"max(|Δw|)={delta:.2e}")

# Reset.
torch.manual_seed(0)
model = SegMap(hidden=24, block_hidden=24, num_blocks=8, max_frame_index=16).to(device)
trainable_params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)

print("\n=== fixed path (uint8_ste in roundtrip) ===")
for step in range(5):
    opt.zero_grad(set_to_none=True)
    rendered = model(masks, frame_indices).reshape(b, t, 3, h, w)
    flat = rendered.reshape(b * t, 3, h, w)
    up = F.interpolate(flat, size=(h * 4, w * 4), mode="bicubic", align_corners=False)
    up_u8 = uint8_ste(up)  # ← FIX: STE forward = round/clamp, backward = identity
    back = F.interpolate(up_u8, size=(h, w), mode="bicubic", align_corners=False)
    rt = back.clamp(0, 255).reshape(b, t, 3, h, w)
    loss = rt.pow(2).mean()
    loss.backward()
    grad_norms = [p.grad.abs().max().item() if p.grad is not None else 0.0
                  for p in trainable_params]
    pre_param0 = trainable_params[0].detach().clone()
    opt.step()
    delta = (trainable_params[0] - pre_param0).abs().max().item()
    print(f"  step={step} loss={loss.item():.4f} max(|grad|)={max(grad_norms):.2e} "
          f"max(|Δw|)={delta:.2e}")

print("\n=== diagnosis ===")
print("If buggy max(|grad|) is exactly 0 and fixed max(|grad|) > 0,")
print("the DARTS-S training freeze is confirmed reproduced + fix verified.")
```

---

## 8. Council Roll Call

### Karpathy (engineering practitioner; arch-search rigor)
**Verdict**: GUILTY. This is the canonical "I forgot the STE" bug. The `.round()` call is a textbook example of a gradient-killing operation hiding in an innocuous-looking utility. The fix is one line (`uint8_ste(up)` instead of `up.clamp(0, 255).round()`). The unit test that "verified" the trainer was passing parameters changed via weight_decay shrinkage, not via meaningful gradients — that's a vacuous assertion. The real assertion is `param.grad.abs().max() > 0`. **Sign-off: ship the patch + the gradient-norm test + STRICT Check 86.**

### Shannon (information-theoretic grounding; LEAD)
**Verdict**: GUILTY. The gradient-disconnect is information-theoretically equivalent to the model receiving NO bits of supervisory signal each step. The "loss = 277" is a CONSTANT in the random-init manifold of `SegMap` parameters; AdamW with weight_decay-only updates produces a low-temperature random walk near initialization, which is consistent with the observed 4-decimal-place stability. From an R(D) standpoint, this run was 5h of zero-information gradient descent — the rate-distortion frontier was never approached. **Sign-off: confirm fix; rerun the sweep; expect score in the predicted [0.27, 0.55] band.**

### Selfcomp (paradigm anchor; built the working 0.38 SegMap)
**Verdict**: GUILTY — and notable. My own production training pipeline used a DIFFERENT trainer (single-frame autoencoder + per-frame TTO chain) that did NOT have this bug. The PR #56 inflate.py path I shipped does not call `.round()` in any training-time gradient chain — it uses uint8 quantization only at archive-build time. The SegMapTrainer in this repo was a NEW reimplementation built to clone the SegMap arch; the dead-gradient slipped in during the eval_roundtrip add. **Sign-off: my SegMap shipped at 0.38 because the bug was absent in my pipeline. Rebuilding the trainer correctly should also reach the [0.27, 0.55] band — the architecture itself is sound.**

### Hotz (raw engineering; "let compute speak")
**Verdict**: GUILTY. Five hours of 4090 burning compute on `param * 0.99996^400` is exactly the kind of dead-loop the `eval_roundtrip` path was supposed to prevent (proxy/auth gap). Ironic. The fix takes 30 seconds; the lesson is "every loss term that touches a `.round()` is a tracer round for a frozen training run." The fact that this slipped past 4 unit tests means the unit tests don't actually test gradient flow, they test param-mutation. Replace them. **Sign-off: ship the fix today; the `frozen-loss watchdog` is mandatory — it would have caught this in 30 minutes instead of 5 hours.**

### Carmack (engineering shortcuts; codebase complexity)
**Verdict**: GUILTY — and the complexity contributed. The `_eval_roundtrip_chain` function is a 26-line standalone with a misleading inline comment ("STE-friendly proxy for uint8 cast") that PROMISES STE behavior but delivers zero gradient. If the author had written `up_u8 = uint8_ste(up)` in the first place — calling the canonical helper that already existed in `tac.quantization` — the bug would never have happened. **The codebase has TWO uint8-roundtrip implementations: one correct (`uint8_ste`), one buggy (`_eval_roundtrip_chain`). DELETE the buggy one. Inline `uint8_ste` into the chain.** Reduce surface area; one canonical implementation per gradient-sensitive op. **Sign-off: ship the fix AND delete the duplicate; STRICT Check 86 is non-negotiable.**

### Fridrich (steganalysis / inverse-detection; co-LEAD)
**Verdict**: GUILTY. The eval_roundtrip simulates the contest evaluator's lossy uint8 decode — but if the simulation passes ZERO gradient, the renderer never learns to be ROBUST to that decode. The "proxy/auth gap" the roundtrip was supposed to close is in fact MAXIMIZED by the broken roundtrip: the model is trained on identity-map gradients (none) and then evaluated with a real lossy decode. The auth/proxy drift on Lane DARTS-S V1 would have been infinite (proxy: garbage; auth: garbage). **Sign-off: this bug invalidates EVERY SegMap-trainer-derived score. All claims attached to Lane MM v2 (FALSIFIED at 2.63), Lane SC++, Lane SA-v2, Lane SO need re-validation after the fix. The "FALSIFIED" verdict on Lane MM v2 is itself FALSIFIED — the lane never got a fair shake.**

### Yousfi (challenge designer; co-LEAD)
**Verdict**: GUILTY — process failure. The contest scorer's uint8 decode is the central invariant the entire challenge depends on. Every training pipeline that simulates it MUST produce a gradient through it; otherwise the lane is racing without an engine. The fact that this bug shipped past 4 unit tests AND a 3-clean-pass adversarial review AND 85 STRICT preflight checks points to a coverage gap: the existing checks scan for `.detach()` and `torch.no_grad()` but NOT for `.round()` in gradient-sensitive paths. **Sign-off: STRICT Check 86 closes the gap; the old `.round()`-in-roundtrip pattern joins the 86 other extinct bug classes. Re-dispatch the DARTS-S sweep on Modal (Vast.ai cost is sunk) after the fix lands.**

### The Contrarian (challenge weak arguments)
**Verdict**: GUILTY — but the user's framing was right. The user wrote: "this should not have happened, our engineering is bugged." The bug is real; the engineering IS bugged. The conservative reading would be "maybe the loss landscape is just hard for SegMap" — REJECTED. The mathematical evidence is decisive: `torch.round()` returns a tensor with zero gradient, the chain through `_eval_roundtrip_chain` zeros the upstream gradient, and AdamW with all-zero grads only moves weights via `weight_decay`. There is no "maybe" here. **Sign-off: ship the fix; ALSO re-examine the FALSIFIED verdicts on every Lane that used SegMapTrainer. The "Selfcomp clone is bolted-on doesn't work" hypothesis (`project_lane_mm_v2_landed_2_63_falsified_20260429.md`) was built on a contaminated experiment.**

### Quantizr (adversarial; competitor reverse-engineering)
**Verdict**: GUILTY. This bug is exactly the kind of "looks plausible, doesn't actually train" trap that distinguishes real-world deployable codecs from theoretical ones. My own pipeline (Quantizr-PR #55) used `kl_distill_segnet_only` with `temperature=2.0` and `weight=0.002` — but I never tried to use it via this trainer because my joint-trainer was different. The score 0.33 I ship is from a 5-stage pipeline (anchor→finetune→joint→QAT→final) that uses the `train_distill.py` path, not `SegMapTrainer`. **Sign-off: the fix unlocks lanes that were previously blocked. Rerun DARTS-S sweep AFTER fix with budget for all 5 configs (~$5 on Vast.ai 4090).**

### Dykstra (convex feasibility; CO-LEAD)
**Verdict**: GUILTY. The achievable region of the SegMap arch sweep was projected to be `[0.27, 0.55]` per `feedback_council_10_member_inner_grand_council_advisory_20260429.md`. The DARTS-S V1 run was supposed to test that projection. Instead it tested `lim_{n→∞} param × 0.99996^n` — a degenerate run that delivered zero feasible-region information. **Sign-off: re-dispatch the sweep AFTER the fix; the predicted band stands; the bug was operational, not theoretical.**

### MacKay (information theory + Bayesian inference; memorial seat)
**Verdict**: GUILTY. From an MDL perspective, a training run that delivers zero data-information per step is a 5-hour-long random walk in a high-dimensional weight space, with the entropy of the trajectory bounded only by the AdamW second-moment decay. The model has learned NOTHING; it has merely shrunk. The `Δseg < 1e-5 for 50+ epochs` watchdog proposed in §4.3 is the right MDL-aware halting criterion: when the description length of the model's output is independent of training step, you are not learning. **Sign-off: ship the watchdog; it generalizes to every training script in the repo.**

### Ballé (modern neural compression SOTA)
**Verdict**: GUILTY. Every neural-compression codec I have built uses `torch.autograd.Function` subclasses for all rate-distortion-relevant quantization (the entropy bottleneck, scale hyperprior, GDN nonlinearity all rely on STE for the rounding step). Bare `.round()` in a forward pass is a known anti-pattern; the canonical references (Ballé 2018, Minnen 2018) all use `torch.autograd.Function` with explicit identity-backward. The `Uint8STE` class in `src/tac/quantization.py` is the correct implementation — `_eval_roundtrip_chain` should call it, not roll its own broken version. **Sign-off: ship the fix; this is the textbook bug from my 2018 paper's appendix.**

---

## Verdict — UNANIMOUS

**12/12 council members vote GUILTY**: Lane DARTS-S V1's training freeze is caused by `up.clamp(0, 255).round()` at `src/tac/segmap_renderer.py:281` zeroing the gradient through `_eval_roundtrip_chain`, so every `loss.backward()` writes zeros to every SegMap parameter, and AdamW only moves the weights by the weight_decay shrinkage of `~0.99996×` over 400 epochs.

**Confidence**: HIGH (verified empirically with two-line PyTorch repros).

**Bug location**: `src/tac/segmap_renderer.py:281`.

**Fix**: replace `up.clamp(0, 255).round()` with `uint8_ste(up)` (the canonical STE already shipped in `src/tac/quantization.py:189`); add gradient-norm assertions to existing tests; add a new direct gradient-flow test on `_eval_roundtrip_chain`; add the frozen-loss watchdog to `experiments/train_segmap.py`; promote a STRICT preflight Check 86 (`check_no_bare_round_in_eval_roundtrip_chain`) to extinct the bug class.

**Cross-impact**: every prior result from `SegMapTrainer.train_epoch` is invalidated and must be re-evaluated after the fix lands. Specifically the FALSIFIED verdict on Lane MM v2 should be revisited — it was likely also gradient-disconnected and never had a fair training run.

**Cost of the bug**: 5 hours of Vast.ai 4090 + $1.41 on the failed sweep + potentially weeks of latent invalid measurements on every prior SegMapTrainer-derived run.

**Cost of the fix**: one-line change + 3 tests + 1 preflight check. Total dev time: ~1 hour. Total council time: this report.
