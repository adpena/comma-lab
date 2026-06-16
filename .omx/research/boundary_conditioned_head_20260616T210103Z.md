# Boundary-conditioned / depthwise-separable HEAD — the hungriest-tensor handler

**Y2 parallel build, 2026-06-16.** `[contest-CPU advisory]` — NO score claims; this proves a
MECHANISM (head capacity at O(depth), not O(width²)) + PARITY, and slots in as an OPTIONAL
production A/B arm wired LATER by P2/integration. NON-PROMOTABLE; pointer 0.19110 UNMOVED.
Module: `src/tac/torch_vehicle/boundary_head.py`; tests: `tests/test_boundary_head.py` (28).

## The problem (from the gate-2 d_seg sensitivity map)
`reports/dseg_sensitivity_map_n600.json`: `rgb_1` — the frame-1 output head SegNet reads to
decide d_seg — has the highest sensitivity density (`blocks.5` / 384×512 stage,
delta_per_kparam 4.64e-4; the HIGH band feeding it ~2.84× under-provisioned). The naive fix
is to widen the final taper channel `final_ch`, but the vendored `refine` block is two FULL
3×3 convs, so its parameter cost is **O(final_ch²)** — widening the d_seg-critical head blows
up the rate quadratically. Verified (`full_refine_param_count`):

| final_ch | full refine (O(C²)) | doubling ratio |
|---|---|---|
| 16 | 2,328 | — |
| 32 | 9,264 | 3.98× |
| 64 | 36,960 | 3.99× |

Doubling width ≈ quadruples params — the C² trap.

## The mechanism (this module): depthwise-separable refine
A separable refine factorizes each full 3×3 conv into a per-channel **depthwise** 3×3
(`groups=cf`, O(cf)) + a 1×1 **pointwise** cross-channel mix (Howard 2017 MobileNets /
Chollet 2017 Xception). Each stage = `cf² + 11·cf` params. Two consequences, both
MEASURED + tested:

1. **Strictly cheaper than the full refine at every width**, advantage GROWING with width
   (exactly where you'd be tempted to widen the head):

   | final_ch | full refine | separable (2-stage) | savings |
   |---|---|---|---|
   | 8 | 588 | 304 | 48% |
   | 16 | 2,328 | 864 | 63% |
   | 20 | 3,630 | 1,240 | 66% |
   | 32 | 9,264 | 2,752 | 70% |
   | 64 | 36,960 | 9,600 | 74% |

2. **Capacity along the DEPTH axis is EXACTLY LINEAR** in `n_stages` (held at fixed width):
   at cf=16 → `[432, 864, 1296, 1728]` (constant +432/stage, `test_separable_refine_depth_axis_is_exactly_linear`).

   ⇒ **gain head capacity by adding depth at O(stages)**, NOT by widening at O(width²). This
   is the headline mechanism: the separable factorization lets the d_seg-critical head carry
   more effective capacity without paying the quadratic that caps the final taper channel.

   *Honest framing of the O(C) claim:* the per-stage pointwise 1×1 is itself O(cf²) IN WIDTH;
   the linear-cost lever is the **depth** axis (add stages), and the separable refine is
   uniformly cheaper than the full refine at any given width. The module does NOT claim a
   single op is O(C) regardless of width — it claims the capacity-adding KNOB (depth) is O(1)
   per unit, vs the full-refine width knob which is O(C²). Both claims are tested.

## The optional secondary: boundary-conditioned low-rank head residual
A rank-r additive correction to the `rgb_1` PRE-sigmoid logits: `U(V(x))`, V=1×1 cf→r,
U=1×1 r→3 (`lowrank_head_residual_param_count(cf,r) = cf·r + 4·r + 3`, O(cf)). DEFAULT OFF.
**Zero-init U** ⇒ enabling it is an EXACT no-op at construction (graceful A/B start; the
warm-started head is unperturbed until training moves it — `test_lowrank_residual_is_zero_init_no_op_then_shifts_output`).
Intent: spend a few params PURELY on the contested-argmax boundary where d_seg is decided
(ties to Lever-D + boundary-math seg core #52). Secondary to the separable refine.

## Parity proof (the faithful-default gate)
`BoundaryHeadTaperDecoder(boundary_head_enabled=False)` (the DEFAULT) defers ENTIRELY to the
parent `ConfigurableTaperHNeRVDecoder.forward` → **bit-identical** to the parent, which is in
turn a faithful generalization of the vendored PR95 decoder. Proven:
- `test_default_off_is_bit_identical_to_parent` — same state_dict keys + `torch.equal` forward.
- `test_default_off_round_trips_vendored_weights_bit_identical` — cross-load REAL vendored
  `HNeRVDecoder(base_channels=20)` weights → default-off child → bit-identical forward (max diff 0.0).
- `test_lowrank_only_enabled_path_equals_parent_at_zero_init` — guards the enabled-path forward
  (a verbatim copy of the parent forward) against drift: vendored refine + zero-init residual
  ⇒ still bit-identical to parent (max diff 0.0).

The default decoder bytes / codec round-trip / Muon partition are therefore UNCHANGED unless
this arm is explicitly selected for a byte-closed candidate.

## Byte-level accounting
The archive decoder blob is ~int8(params)+brotli, so param count is the rate proxy at this
layer. Separable savings translate ~directly to decoder-blob bytes when the arm is selected.
For the production base_ch20 taper, `final_ch=10` (vendored [20,20,20,15,11,10,10]; decoder
83,356 params): full refine 915 params vs separable-2stg 420 / 3stg 630. So the lever lets
the head go DEEPER (1→2→3 stages) for the cost of ~half the vendored refine — capacity to
the d_seg-critical head with a NEGATIVE byte delta vs the vendored refine, or extra depth at
a fraction of the cost of widening. Exact archive int8+brotli byte delta is measured at
byte-close time on the selected candidate (review R4 = verify at the POST-int8-brotli archive,
not param-count) — deferred to P2/integration; here we bank the param-level mechanism + the
~48–74%-savings table above.

## Forward correctness (NO-FAKE, REAL compute)
Enabled forward: finite, correct shape `(2,2,3,384,512)`, output in `[0,255]`
(`test_enabled_separable_forward_shape_and_finite`); it TRAINS — real gradient flows to BOTH
the separable refine AND the low-rank residual (`test_enabled_forward_grad_flows_to_separable_refine_and_residual`).
Param-count formulas equal ACTUAL `nn.Module` param counts at every tested width/stage.

## Numpy-portable forward (the inflate path is torch-free)
Every op is a standard conv2d / depthwise-conv2d (groups=C) / 1×1 conv / sigmoid / sin —
expressible in numpy as: depthwise = per-channel 2D correlation with each channel's k×k kernel
+ bias; pointwise/1×1 = `einsum('bchw,oc->bohw')` + bias; sin/sigmoid elementwise. The DEFAULT
path is the vendored refine (already numpy-portable), so the inflate runtime is unchanged
unless this arm is selected. A numpy reference for the enabled arm is a small follow-on (P2).

## Module API
```python
from tac.torch_vehicle.boundary_head import (
    BoundaryHeadTaperDecoder,            # ConfigurableTaperHNeRVDecoder subclass, default-OFF
    DepthwiseSeparableRefine,            # drop-in for self.refine (O(stages) depth)
    BoundaryLowRankHeadResidual,         # zero-init rank-r rgb_1 logit correction
    full_refine_param_count,             # exact O(C^2) vendored-refine cost
    separable_refine_param_count,        # exact n_stages*(cf^2 + 11*cf)
    lowrank_head_residual_param_count,   # exact cf*r + 4*r + 3
)
# DEFAULT path == vendored decoder, bit-identical:
dec = BoundaryHeadTaperDecoder(latent_dim=28, channels=[20,20,20,15,11,10,10])
# A/B arm (separable refine, deeper head, + optional boundary residual):
dec = BoundaryHeadTaperDecoder(
    latent_dim=28, channels=[20,20,20,15,11,10,10],
    boundary_head_enabled=True, separable_n_stages=2,
    lowrank_residual_enabled=True, lowrank_rank=4,
)
```

## A/B-arm wiring note (NOT done here — P2/integration owns it)
This is an OPTIONAL production A/B arm. Wiring (LATER, by the integration/P2 owner — this
build does NOT edit `driver.py` / `launch_*.py` / `configurable_taper_decoder.py`):
1. Production launcher gains a `--boundary-head` / `--separable-stages N` / `--boundary-rank r`
   flag set defaulting to the OFF/vendored path (byte-identical default).
2. The driver's decoder construction selects `BoundaryHeadTaperDecoder` when the flag is set;
   the codec/export path is unchanged for the default; for the enabled arm, the separable-refine
   + residual state_dict keys must be added to the export blob (a new key group) — a P2 export hook.
3. The export byte-delta is verified at the POST-int8-brotli archive (R4), and the A/B is
   decided by a paired smoke (separable-head vs vendored-head) on d_seg at matched bytes.

## Honest verdict (Catalog #307 IMPLEMENTATION-LEVEL)
The MECHANISM works and is PROVEN (capacity at O(depth) not O(width²); strictly cheaper than
full refine; parity bit-identical). It is NOT a measured d_seg win — `rgb_1`'s high OUTPUT
sensitivity is EXPECTED (it writes the scored pixels) and ≠ under-capacity (the review's
perturbation-asymmetry). The production A/B (separable/boundary head vs vendored head, matched
bytes, paired d_seg smoke) decides whether the head is actually the binding lever. `--final-cap`
widening remains a cheap orthogonal empirical check; this module is the principled alternative
to it (depth, not width).

## Wire-in hooks (per Catalog #125)
- #1 sensitivity-map: CONSUMES `reports/dseg_sensitivity_map_n600.json` (rgb_1/HIGH-band finding) — ACTIVE.
- #2 Pareto: ADDS a head-capacity-vs-rate point (separable depth) to the decoder Pareto — ACTIVE-design.
- #3 bit-allocator: N/A (capacity allocator, not a bit allocator; ties to taper not codec levels).
- #4 cathedral autopilot: N/A (research module; not archive-deployable until P2 export hook).
- #5 continual-learning: N/A ($0 mechanism proof; no empirical anchor to seed).
- #6 probe-disambiguator: the production A/B (separable head vs vendored, matched bytes) IS the
  disambiguator for "is rgb_1 capacity-starved or just output-sensitive?" — ACTIVE-design (P2).
