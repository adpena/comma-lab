# 3. The Gradient Bug

This section describes the discovery and resolution of a gradient obstruction in the upstream scorer pipeline. We present it in detail because the bug class is general, the failure mode is subtle, and the fix was worth 0.27 points --- more than any architectural change in the project.

## 3.1 Symptoms

Test-time optimization (TTO) v1 through v4 showed a consistent pattern: SegNet distortion improved steadily over 500 optimization steps, while PoseNet distortion either stagnated or got worse. We attributed this to PoseNet being "harder to optimize" --- its loss surface is noisier, it operates on frame pairs rather than individual frames, and the ego-motion prediction task is geometrically more complex than segmentation. These are plausible explanations. They are also wrong.

TTO v1 achieved a 6.3% overall improvement. We interpreted this as evidence that the optimization pipeline was working, with PoseNet improvements being smaller and noisier than SegNet improvements. In reality, PoseNet was not being optimized at all. The 6.3% came entirely from SegNet gradients moving pixels in ways that incidentally reduced PoseNet loss --- what we later called "SegNet spillover."

## 3.2 Root cause

The upstream scorer code (`frame_utils.py`, line 50) contains:

```python
@torch.no_grad()
def rgb_to_yuv6(x):
    ...
```

This decorator creates an autograd barrier. PoseNet's `preprocess_input()` calls `rgb_to_yuv6` to convert RGB frames to YUV 4:2:0 (6 channels). Any gradient flowing backward through PoseNet's loss, through the network, through preprocessing, hits this barrier and becomes zero. The autograd graph is silently detached at the color-space conversion.

One decorator. Zero PoseNet gradients. Every TTO experiment in the project was blind.

## 3.3 Why it was hard to find

Five factors conspired to hide this bug:

**1. The training pipeline had its own fix.** Our training code applied `_patch_scorers_for_training`, which replaced `preprocess_input` with a differentiable version. Training worked. The TTO pipeline loaded scorers through a different code path (`load_scorers()`) that did not apply the patch. Two paths into the same scorers, one patched, one not.

**2. PoseNet loss still changed.** Because SegNet gradients moved pixels, and those pixel changes affected PoseNet's output, PoseNet loss was not constant during optimization. It fluctuated. It sometimes improved. It looked like noisy optimization. The distinction between "optimizing PoseNet" and "PoseNet changing as a side effect of optimizing SegNet" is invisible without inspecting the gradient tensor directly.

**3. The decorator is in upstream code.** We did not write `rgb_to_yuv6`. It lives in a dependency we treat as frozen. Grepping our own codebase for gradient issues would never find it.

**4. Gradient norms were not monitored.** We tracked loss values, not gradient magnitudes. A zero gradient produces a non-zero loss (the forward pass works fine) and a non-zero loss change (from other gradients moving pixels). Standard training dashboards do not flag this.

**5. The prior was wrong.** PoseNet is known to be hard to optimize --- high-variance loss, geometric sensitivity, temporal dependencies. "PoseNet is hard" is a perfectly reasonable explanation for slow convergence. It just happened to be wrong in this case, because the real explanation was "PoseNet gradients are zero."

## 3.4 Discovery

The bug was found during an adversarial review session by the skunkworks council. The Contrarian demanded an explanation for a specific observation: 50 steps of gradient descent made PoseNet *worse*, not better. Not "didn't improve much" --- actively worse. If the optimizer has access to PoseNet gradients, it cannot make PoseNet worse on average across many steps (assuming a reasonable learning rate). That is not how gradient descent works.

George Hotz traced the call chain: TTO loss function $\rightarrow$ PoseNet forward pass $\rightarrow$ `preprocess_input` $\rightarrow$ `rgb_to_yuv6` $\rightarrow$ `@torch.no_grad`. The gradient highway had a toll booth that charged infinity.

The council voted unanimously to fix immediately. The fix was committed within 15 minutes.

## 3.5 Connection to obfuscated gradients

This failure mode is an instance of *obfuscated gradients* [Athalye et al. 2018], though arising from a software bug rather than deliberate defense. Athalye et al. identified three types of gradient masking in adversarial robustness: shattered gradients (non-differentiable operations), stochastic gradients (randomized defenses), and vanishing/exploding gradients. Our case is closest to shattered gradients --- a non-differentiable barrier (`no_grad`) inserted into an otherwise differentiable pipeline.

The parallel is instructive. In adversarial ML, obfuscated gradients give a false sense of robustness: the model appears resistant to gradient-based attacks, but only because the attacker's gradients are broken. In our case, obfuscated gradients gave a false sense of optimization: PoseNet appeared to improve under gradient descent, but only because SegNet gradients incidentally moved pixels in helpful directions.

Athalye et al.'s remedy --- check whether the attack (optimization) actually works on individual examples --- is exactly the diagnostic that caught our bug. When we asked "does gradient descent actually reduce PoseNet loss?", the answer was no.

## 3.6 The 1ms validation check

The fix includes a runtime gradient validation that runs before every TTO optimization loop:

```python
def _validate_gradient_flow(scorers, sample_input):
    """~1ms check that gradients flow through both scorers."""
    x = sample_input.clone().requires_grad_(True)
    for name, scorer in scorers.items():
        out = scorer(x)
        loss = out.mean()
        grad = torch.autograd.grad(loss, x, retain_graph=True)[0]
        if grad.abs().max() == 0:
            raise RuntimeError(f"Zero gradients through {name}")
```

This check costs approximately 1ms per scorer. It prevents the bug class entirely: if any future code change, dependency update, or scorer modification breaks the gradient flow, the error is caught immediately rather than after hours of blind optimization.

The pattern generalizes. Any pipeline that optimizes through frozen networks should validate gradient flow before trusting results. The cost is negligible. The cost of *not* checking was, in our case, weeks of GPU time and every TTO result being invalid.

## 3.7 Impact

| Metric | Before fix (TTO v4) | After fix (TTO v5a) | Change |
|--------|---------------------|----------------------|--------|
| Auth score | 0.70 | 0.43 | -38.6% |
| PoseNet distortion | ~0.012 | 0.00209 | -82.6% (8.2x) |
| SegNet distortion | ~0.0015 | 0.00149 | -0.7% |
| Steps before early stop | 151 | 500 (full) | 3.3x |
| PoseNet score contribution | 0.415 | 0.172 | -58.6% |

The gradient fix is the single largest improvement in the project's history, larger than switching from codec postfiltering to a neural renderer (0.87 vs. 1.33) and larger than adding TTO itself (0.70 vs. 0.87). It cost 15 minutes of engineering and 0 additional parameters.
