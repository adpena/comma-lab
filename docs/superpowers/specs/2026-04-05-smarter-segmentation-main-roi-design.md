# Smarter Segmentation Main-ROI Design

## Context

Track B currently holds an authoritative local floor of **3.33** with a uniform x265 path at `432x324 / medium / crf23 / keyint48 / bframes4 / ref4 / lanczos+lanczos`.
A naive fixed-rectangle ROI two-pass prototype failed badly (`5.73`), which rules out hard, static masking but does **not** rule out ROI-aware compression as an architectural direction.

The official challenge rules matter here:

- submissions are evaluated from `archive.zip` + `inflate.sh`
- heavy artifacts used during **inflate** should be included in the archive and count toward compressed size
- offline compression-time analysis can be heavier, because the compression script is optional for evaluation
- official inflation has a 30 minute limit

That means any learned/staticness analysis should stay on the **compression side** unless we later prove that shipping decoder-side weights is worth the byte and runtime cost.

## Problem Statement

We need a smarter ROI experiment that:

1. preserves the **main ROI** instead of flattening it into an afterthought
2. adapts over time instead of using one fixed rectangle
3. keeps byte accounting honest
4. does not add heavy inflate-time dependencies
5. gives us one small, measurable, authoritative experiment rather than another vague research lane

## Candidate approaches

### Approach A — Hand-tuned static rectangles

Keep the existing central ROI, tune its geometry, and add one auxiliary ROI.

**Pros**
- Cheap to implement
- No new dependencies
- Easy to measure

**Cons**
- Already partially falsified by the failed fixed-ROI prototype
- Cannot react to ego-motion or scene changes
- High risk of preserving the wrong pixels

### Approach B — Dynamic tile saliency from motion/staticness analysis (**recommended**)

At compression time, compute a low-cost per-tile importance map from temporal change, edge energy, and center/road prior; smooth it across time; then derive a soft main ROI plus optional auxiliary ROI envelopes.

**Pros**
- Dynamic without needing a shipped neural decoder
- Respects the user constraint to keep the main ROI central and important
- Lets BAT00 do real work on offline analysis while local CPU remains authoritative for scorer claims
- Small, reversible extension from the current ROI path

**Cons**
- More moving parts than a fixed rectangle
- Needs careful temporal smoothing to avoid flicker
- Still heuristic unless later upgraded with learned features

### Approach C — Learned semantic mask / embedding-guided ROI

Use a learned segmentation or embedding model during compression to estimate task-critical regions and derive masks from that.

**Pros**
- Highest theoretical upside
- Best match for the task-driven scorer if it works

**Cons**
- More engineering cost now
- Easy to overfit the research lane before we have a stable dynamic heuristic baseline
- If moved into inflate, weights would likely need to ship and count

## Recommendation

Use **Approach B** first.

Specifically:

- keep the **main ROI** as the primary protected region
- make it **dynamic and temporally smoothed**, not fixed
- allow **one auxiliary ROI** when the importance map consistently highlights a second region
- keep the reconstruction path simple: base stream + main ROI stream + optional aux ROI stream
- do all analysis during compression, using `uv run` for Python tooling

This is the smallest experiment that tests the actual hypothesis:

> the scorer cares about semantically important, temporally stable structure more than uniform fidelity, but the protected region must be chosen dynamically and smoothly.

## Design

### 1. Compression-side analysis only

Add a Python analysis step, run via `uv run`, that inspects each source video and emits per-frame or per-window ROI metadata.

Input:
- original video
- current operating resolution (`432x324` initially)

Output:
- main ROI track: `x, y, w, h` per time window
- optional aux ROI track when justified
- metadata summary for audit/debugging

No new inflate-time model dependency is introduced in this phase.

### 2. Importance-map construction

For each frame/window, compute a coarse tile map using:

- temporal change magnitude between adjacent frames
- edge/texture energy
- center-weight prior to avoid neglecting the main ROI
- optional lane-band / road-horizon prior

Then smooth across time with hysteresis / EMA so the ROI does not jump around.

This is explicitly designed to avoid neglecting the main ROI:
- the central driving corridor remains a first-class prior
- auxiliary regions are additive, not replacements for the main ROI

### 3. ROI extraction policy

Derive:

- **Main ROI:** always present, centered on the dominant saliency mass and biased toward the driving corridor
- **Aux ROI:** optional, only emitted when persistent saliency remains outside the main ROI over a threshold window

Guardrails:
- min/max ROI size bounds
- even dimensions only
- movement clamped per window to prevent jitter
- disable aux ROI when it is too unstable or too small

### 4. Encoding strategy

Keep the existing honest two/three-stream architecture:

- degraded base stream
- higher-quality main ROI stream
- optional higher-quality aux ROI stream

For the first experiment, do **not** add alpha masks, learned blending, or decoder-side inpainting. Keep reconstruction straightforward and byte accounting obvious.

### 5. Evaluation strategy

Run one small measured cycle, sequentially:

1. smoke-check packaging/inflation
2. proxy check on a tiny subset if needed
3. one authoritative local CPU evaluation for the dynamic-main-ROI candidate
4. compare against the current `3.33` floor

Promotion bar:
- package succeeds
- inflate succeeds
- shape/frame-count checks pass
- full local evaluation beats `3.33`, or the result is recorded as a rejection

## Constraints

- Use `uv` for Python package management and Python execution.
- Do not move heavy learned assets into `inflate.sh` in this phase.
- Keep `current_workflow` and `rule_faithful` accounting separate.
- BAT00 can assist with offline analysis/profiling, but not as the source of authoritative score claims.
- Keep the main ROI explicit in naming, outputs, and experiment writeups.

## Success criteria

A successful first version is **not** “perfect segmentation.”
It is:

- one reproducible dynamic-main-ROI experiment on disk
- honest bytes and runtime notes
- clear evidence whether dynamic ROI beats the uniform `3.33` floor
- enough metadata to understand why it won or lost

## Failure criteria

Reject or pause this lane if:

- ROI metadata is unstable enough to create obvious jitter
- aux ROI appears often but adds bytes without lowering score
- analysis overhead becomes large without improving candidate quality
- local authoritative score is not competitive with `3.33`

## Execution boundary

This spec authorizes the next implementation phase to build:

1. a compression-time ROI analysis helper
2. metadata plumbing into the existing ROI-capable `compress.sh`
3. one small authoritative local evaluation cycle

It does **not** authorize:

- decoder-side learned models
- shipping large segmentation weights in the evaluated path
- broad refactors outside the current submission path
