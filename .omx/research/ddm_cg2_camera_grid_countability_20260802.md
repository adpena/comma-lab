---
title: "cg2 — the camera grid is the worst grid in the chain: its address costs more than its authority is worth"
lane_id: lane_ddm_cg2_camera_grid_countability_20260802
arm: ddm_cg2 (make the camera-grid resolution countable)
date_utc: 2026-08-02
authority: "[macOS-CPU advisory / DERIVED-from-MEASURED-geometry] SCORER-FREE. NON-PROMOTABLE."
pointer_delta: "UNMOVED. This arm closes a family; it does not move the exact score."
verdict: STRUCTURAL_WASH_DOMINATED
verdict_scope: "FAMILY for independent per-pixel camera-res corrections; FORMULATION for structured/clustered camera-res schemes (not searched)"
council_predicted_mission_contribution: frontier_protecting
---

# cg2 — camera-grid countability

**THE ASK:** can a camera-resolution section be added to the grammar so the 230,904 blind
px/frame become COUNTABLE AND USEFUL — real ΔS, or structurally a wash?

**THE VERDICT: STRUCTURAL WASH, and stronger — structurally DOMINATED.** Not "we could not
find a win"; the camera grid is the *worst* grid in the chain by construction, because it
simultaneously maximises address cost and minimises score authority per element. The blind
set cannot become useful because "blind" *means* zero score authority; and the *non*-blind
camera pixels are dominated by the scorer grid, unconditionally.

`#401` stays correct, stays built, and stays worth 0 B on this vehicle. **eb1 R7's closure
holds.** This arm supplies the reason it will keep holding for any future grammar, which R7
explicitly left open (*"if a future grammar counts camera pixels, #401 becomes a real rate
lever immediately"* — that conditional is now answered: such a grammar would itself be the
error).

---

## §1 — the blind set, re-derived on THIS vehicle (item 1: is 230,904 stale?)

**Not stale. Exact. And vehicle-INDEPENDENT.** MEASURED by impulse-probing the real kernel:

| quantity | measured | #401's value |
|---|---|---|
| blind camera ROWS | **106** / 874 | 106 |
| blind camera COLS | **140** / 1164 | 140 |
| **blind px/frame** | **230,904** | 230,904 |
| blind fraction | **22.6969 %** | 22.6969 % |
| retained sub-grid | **768 × 1024** = 786,432 | same |
| smallest nonzero weight | 2.604e-3 (clean separation from 0) | 2.60e-3 |

*Why it is vehicle-independent:* blindness is a property of the **upstream scorer's** resize
`(874,1164) → (384,512)`, bilinear / `align_corners=False` / `antialias=False`
(`upstream/modules.py:109`), which **every** candidate passes through. v4d reaches it via
`ddm_tr1_runtime.bicubic_up_to_camera_float` → the scorer's own downsample. The number is a
kernel property, not a content property, so it is **exact for all 600 pairs by construction**
— n600-valid without a per-frame sweep.

**Instrument validated against the primary artifact:** my separable matmul reproduces
`torch.nn.functional.interpolate` to **2.84e-14** max abs on a real frame. (I first saw
`overflow encountered in matmul` RuntimeWarnings; I did not suppress them until I had proven
against torch that the results were exact — they are Accelerate BLAS noise, not numeric error.)

**Positive control, with denominators:** all 874 rows and 1164 cols checked; every one of the
106 blind rows has exactly-zero weight, and every one of the 768 retained rows has a nonzero
weight (the mask is not vacuous). Asserted in-script — the run aborts if it fails.

**Receiver re-verified independently:** v4d = 360,238 B, **6 members**; `state/tokens.dr7t` =
346,478 B (96.18 %) decoding to `(600, 24, 32, 4)` uint4 on a 24×32 grid → render 384×512 →
generic bicubic up. **No counted section holds camera-resolution pixel data.** `#401` is
default-OFF (`tools/levelset_byte_close_and_eval.py:692`, both prompt line refs confirmed) and
its only non-test caller passes `False` (`tools/probe_einstein_kolmogorov_xi_bridge.py:881`).
Note it is wired into the **levelset** byte-close chain, *not* into the live TR1/v4d receiver
at all. A prior arm already executed it at n600 on a pure-generator base:
`EXECUTED_SCORE_NEUTRAL_ON_PURE_GENERATOR_BASE`, `n600_batch32_exact: true`, Δbytes = Δd_seg =
Δd_pose = **exactly 0** (`ddm_pa2_zero_byte_decode_family_20260724T194836Z/receipt.json`).

---

## §2 — the DOF ledger: blind is the SMALLER part of the invisibility

eb1's correction (b) — *`blind ⊂ ker(A)`, do not union* — is right, and here is what it costs.
Writing `D` for the 874×1164 → 384×512 operator, per channel per frame:

| quantity | value |
|---|---|
| camera DOF | 1,017,336 |
| **rank(D)** | **196,608 = EXACTLY the scorer grid ⇒ D is SURJECTIVE** |
| **dim ker(D)** | **820,728 = 80.674 % of camera DOF invisible** |
| blind set | 230,904 = **only 28.13 % of ker(D)** |
| retained / scorer overparameterization | **exactly 4.0×** |
| camera / scorer overparameterization | 5.174× |

So **#401 removes 28 % of the invisibility; 72 % of it survives inside the retained sub-grid
by construction.** Filling the blind set takes a 5.17× overparameterization down to 4.0×. It
never approaches 1.0×, because `D` is surjective — every score effect a camera-res payload can
produce is *already* expressible at the scorer grid, and `y = D·x` is a deterministic function,
so `H(y) ≤ H(x)`: the scorer-grid description can never cost more bits than the camera-res one.

**Block structure (MEASURED, and it is what makes the next section airtight):** exactly **2
taps per output row and 2 per output col ⇒ 4 camera px per scorer px**; and each retained
row/col feeds **exactly ONE** output row/col. The blocks are **disjoint**. Therefore
**one camera pixel influences exactly one scorer pixel — blast radius 1.**

---

## §3 — the decisive arithmetic: the address costs more than the fix is worth

Two measured constants, then one derivation:

- one fixed argmax flip is worth `ΔS = 100/(600·384·512)` = **8.477105e-7**
- one archive byte is worth `ΔS = 25/37,545,489` = **6.658590e-7**
- ⇒ **break-even `W` = 1.273108215332031 B/flip.** This *reproduces the registered constant
  `W = 1.27310821533` exactly* — an independent positive control on the S-arithmetic.
  (Sanity: v4d's rate term recomputes to 0.2398677 vs the lineage doc's 0.23987. ✓)

### (a) UNCONDITIONAL — camera-res is dominated by scorer-res, no assumptions

For the *same* set of flips, addressing at camera resolution costs exactly
`log2(786,432/196,608) = 2 bits` **more** per correction than addressing at the scorer grid —
0.25 B, i.e. **0.196 × W**, ~20 % of a flip's entire worth, burned on nothing. Those 2 bits
select which of the 4 taps inside the block to touch, i.e. they buy **amplitude granularity
only** — and §5 shows amplitude granularity is available **for free** from a generic rule. No
uniformity or clustering assumption enters this. **Camera-res is strictly dominated, always.**

### (b) ABSOLUTE — per-pixel correction is net-negative at *any* grid

Under uniform addressing, the address alone costs:

| grid | address bits | address bytes | vs `W` |
|---|---|---|---|
| camera full | 21.54 | 2.693 | **2.12×** |
| camera retained (#401 applied) | 21.17 | 2.646 | **2.08×** |
| scorer grid | 19.17 | 2.396 | 1.88× |
| v4d token cell | 11.58 | 1.448 | 1.14× |

**Saying *where* costs 2.08× what the fix can ever be worth — before a single value bit.** And
because a camera pixel's blast radius is exactly 1 (§2), it can never amortise that address
across multiple flips. With an 8-bit value the correction costs 3.646 B = **2.86× break-even**,
requiring ≥2.86 flips per correction that the geometry structurally forbids.

**Honest caveat on (b):** 21.17 bits assumes uniform iid positions. Real flips cluster on the
codim-1 separatrix, which lowers address entropy — so (b) is not airtight on its own. It is
(a) that is unconditional, and (a) already settles the fork. I flag this rather than let the
bigger number carry weight it has not earned.

**The general law this exposes:** finer grid ⇒ *more* address bits AND *smaller* blast radius.
Both terms move the wrong way together. The camera grid is the finest grid in the chain, so it
is the worst place to put counted payload. This is also *why* v4d looks the way it does: a
**dense** 24×32 token grid pays **no addresses at all** (~1.50 bits/token measured) and each
cell drives a 16×16 region.

---

## §4 — pricing the grammar change (item 2)

**Byte budget.** v4d = 360,238 B = **600.4 B/pair** for the whole archive; tokens are
577.5 B/pair. A dense per-pixel section for frame1 alone (3 channels) would need:

| section grid | values/pair | bits/value to fit the ENTIRE archive budget |
|---|---|---|
| camera full | 3,052,008 | 0.001574 |
| camera retained (#401) | 2,359,296 | 0.002036 |
| scorer grid | 589,824 | 0.008143 |

Real residual coders live at 0.5–2 bits/value. **A dense camera-res section is 2–3 orders of
magnitude outside the entire archive budget**, and #401's 22.7 % discount moves it from 300×
over to 245× over. Sparse is the only survivable form, and §3 kills sparse.

**Receiver complexity.** +1 ZIP member; +1 `SECTION_ENTRY` (`>16sQQ32s` = 64 B) in the
`DDMTR1P1` packet plus its SHA-256; and the Catalog #417 counted-consumption bijection
obligation — `parse_packet` enforces order, contiguity, per-section SHA and `cursor ==
len(packet)` (`ddm_tr1_runtime.py:801-802`), and `pose_warp.stp` likewise asserts `off ==
len(payload)`. All of that structural cost to carry a payload that is dominated before it
ships a single byte.

---

## §5 — rule-118 adjudication (item 3)

The test that decides every case: **does the artifact change when the video changes?**

**FREE in inflate.py (generic algorithm):**
- the blind-mask derivation — impulse-probing the resize kernel. It names 230,904 positions,
  but it is computed from constants in *upstream code*, contains no video content, and is
  identical for every clip. Legitimately free despite its size.
- `generic_inpaint_fill` (separable linear interp over retained coords).
- `bicubic_up_to_camera_float` (A = −0.75 computed in code, not stored).
- **any deterministic dither/placement rule** mapping a generated 384 field to camera uint8.

**COUNTED in archive.zip (video-derived payload):**
- any per-frame/per-pair camera-grid **values** — residual pixels, HF sidecar, palette indices;
- any table indexed by frame or pair, *however encoded* — including one emitted as a Python
  literal inside `inflate.py`. That is the hide-data-in-code fake (NO-FAKE #6/#7), and at
  camera resolution the temptation is at its largest, which is precisely why the boundary
  deserves stating before anyone builds here.

**One trap named explicitly.** The blind pixels of the **output frames** are free to *write* —
no scorer reads them. It is tempting to call them "free capacity." They are not: the decoder
must already know what to put there, so they can never reduce counted bytes. **Free pixels ≠
free bits.** Any scheme claiming to "store information in the blind region" is storing it in
the archive and paying for it there.

---

## §6 — the fork (item 4), and what IS real

**FORK: STRUCTURAL WASH — dominated.** A camera-resolution counted section should not be built.
The blind pixels cannot be made "useful" (blind *means* zero authority); the non-blind ones are
dominated by the scorer grid unconditionally (§3a) and are 2–3 orders of magnitude outside the
byte budget (§4). Every steelman I could construct fails: PoseNet shares the same `D`; a
"free side-channel" in blind output pixels is the §5 trap; structured addressing describes a
*region* and should be priced at the region's own scale; sub-pixel placement is §6a below and
needs no section.

**What IS real, and it is not a rate lever:**

**(a) The camera grid is a FREE amplitude lattice.** MEASURED at a probe scorer pixel: 4 taps,
weights `{0.4062, 0.2318, 0.2305, 0.1315}` summing to 1; the smallest single-uint8-tick step is
**0.1315**, i.e. **2.93 extra bits of amplitude authority per scorer pixel** — obtainable at
**zero counted bytes**, because the rule that picks the 4 camera values is generic (§5). The
live receiver currently spends none of it: `np.clip(np.rint(up), 0, 255)` is plain rounding.
This is a **RE-ANCHOR of #149** (camera-res sub-pixel PLACEMENT, closed-form, COMPLETED, $0),
not a discovery — and per the standing re-anchor discipline I am labelling it as such rather
than dressing it up. **Pricing its residual d_seg value requires the scorer ⇒ GATED**, named
below.

**(b) A decode wall-clock note.** 22.6969 % of the camera-res upsample/warp output is never
read by any scorer, so it need not be computed exactly — a compute saving inside the 30-min
budget, **not** a rate saving. **DERIVED, not measured on the real receiver** — I did not time
`inflate_runner_v4d`.

---

## §7 — honest non-findings and scope

- **I ran no scorer.** Every d_seg/d_pose statement here is *structural* (zero weight ⇒ zero
  effect), never scorer-measured. The single full-n600 scorer slot was held by `ddm_lr1`.
- **Verdict scope on the ladder:** FAMILY-level for independent per-pixel camera-res
  corrections (the geometry forbids amortisation). FORMULATION-level for structured/clustered
  camera-res schemes — I did not search those, and §3b's absolute bound weakens under
  clustering; only §3a's relative dominance survives there, which is enough to dominate but not
  to call the sub-family dead.
- **gc17 §2b's pre-registered failure mode is consistent with this and I did not need to build
  either form.** Its cure — *"band-designed at the scorer's 384 grid … a naive camera-res chroma
  dither pays ~luma pose cost and gets no free lunch"* — is the same conclusion from the pose
  side that §3a reaches from the rate side: **design at 384, realise at camera res for free.**
  I did not build the naive form (known-refuted) or the cured form (it is a *placement* rule,
  not a section, so it is out of this arm's scope and into #149's).
- **Not measured:** actual decode wall-clock; any clustered-address camera-res encoding; the
  realized d_seg value of the free amplitude lattice. (I attempted an n600 round-trip
  realization-error sweep to size that last one; it died at exit 144 — the documented SIGURG
  kill — and I am reporting it as not-run rather than quoting a partial. Nothing in this
  artifact depends on it.)
- **The steelman that survives, stated precisely so the closure is not over-claimed.** What is
  dominated is **storing camera-grid VALUES**. A compact generative *program* whose output
  happens to be evaluated at camera resolution is **not** touched by §3 — `H(y) ≤ H(x)` bounds
  ideal codes, and a real coder could describe, say, a thin diagonal line more cheaply at
  camera res than the anti-aliased 2-px band it becomes at 384. But that description is a
  **latent program**, not a camera-res section: the decoder can always render it and apply `D`
  for free. **That is exactly what v4d already is** — a 24×32 token grid rendered up. So this
  steelman does not reopen the section question; it re-describes the vehicle we already have,
  and it belongs in the token/renderer budget, not in a new counted camera-res member.

## §8 — the ONE next measurement (gated, names its own gate)

**Does spending the free amplitude lattice move d_seg?** Replace the receiver's
`clip(rint(up))` with the generic dither that minimises `‖D(clip(rint(cam))) − y*‖` per disjoint
2×2 block, hold the archive **byte-identical** (the rule is rule-118 free), and re-measure d_seg
through the frozen CPU-torch SegNet at n600. Zero rate risk by construction; the only question
is whether the sub-LSB placement authority converts into argmax flips.
**GATE: needs the full-n600 scorer slot** — queued, not taken.

## §9 — triality

- **DAG:** FEED-cg2 (this file). Closes the conditional eb1 R7 left open.
- **DSL:** N/A — a receiver/grammar adjudication, no trainer lever, no launch, no curriculum.
- **equations:** consumes `blind_coordinate_rate_lever_v1` (scope sharpened: the byte delta is
  real but applies only to a camera-res-storing representation, which §3 shows is dominated);
  independently reproduces the registered break-even `W = 1.27310821533 B/flip`.
