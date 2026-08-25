# CLICK-POLISH → WITNESS design synthesis — applying PR128's exact-gated discrete polish to the task-space level-set SDF witness (v7.5.2 · v7.5.3/two-trunk · v8) — 2026-07-10

**Operator directive (verbatim):** *"we also want to consider how those techniques might be leveraged and
applied to our ultimate task space level set SDF witness whether one trunk or two trunk or v8."*
**"Those techniques" = PR128's four mechanisms** (exact-score-gated ±1/±2 discrete "click" polish on stored
per-pair codes · pair-locality → DIAGONAL BATCHING · sidecar-folding · native-CPU-axis selection),
fully specified in `pr128_intake_reverse_engineering_20260710.md`.

**Subagent:** `clickpolish-witness-design` (checkpointed). **DESIGN ONLY** — no code, no launches, no GPU.
**Pointer contest-CPU 0.19110 UNMOVED (MEASURED, `canonical_frontier_pointer.json`)** — everything here is
`[macOS-CPU advisory · research-signal · NON-PROMOTABLE]` MEANS; only a byte-closed `upstream/evaluate.py`
n600 exact row < 0.19110 moves it. verdict_scope discipline on every dismissal (default narrowest).

**STORES CONSULTED:** `pr128_intake_reverse_engineering_20260710.md` (the four mechanisms + measured
negatives) · `src/tac/through_r/mc_finisher.py` (#396 — OUR existing exact-metric gradient-free finisher,
READ IN FULL) · `src/tac/through_r/harness.py` (`measure_through_r`; `SUPPORTED_BACKENDS=("cpu-torch",)`) ·
`tools/levelset_byte_close_and_eval.py` (`build_levelset_blob` §377, `build_pose_carrier_section` §722, the
byte-close render/R/eval path) · `src/tac/boundary_math/lever_b_levelset_generator.py`
(`levelset_rgb_forward_numpy` §717 — the locality proof) · `src/tac/boundary_math/xi_pose_coder.py`
(`quantize_xi` q_levels=4096) · `r1_dxi_shippability_byteclose_20260708.md` (the 7.2 KB ξ section = our
per-pair stored codes; banked d_pose 0.001610 → 0.127) · `t5_crucible2/SPEC_v752_20260709.md` (single-trunk
sealed body + terminal-band 4a/4b/4c) · `fullstack_fractal_optimal_synthesis_20260710.md` (W=(G,ξ,T) home
map + the terminal-band 4a′ MC-finisher placement + v7.5.3 skeleton) · `SPEC_v8_perclass_decomposition_20260708.md`
+ `t5_crucible3/SPEC_v8.1_20260709.md` (per-class carriers · Movable 6289 B site coder · b_c no_offset) ·
CLAUDE.md §WITNESS CAPSTONE + §v7.5/v8 VEHICLE LINE + the resize/flip ledger (#391) + margin-saliency (#141)
per the sensitivity-derivation discipline.

---

## 0. HEADLINE (answer-first)

**The witness is pair-local by construction, so the diagonal-batching exploit applies directly — but the
apparatus for it ALREADY EXISTS as #396 (`mc_finisher.py`), operating today over SHARED head tensors where
there is NO locality. The recommendation is to UNIFY: add a pair-local DIAGONAL mode to #396, not build a
second tool.** The witness's clickable per-pair codes are the FiLM `code` table (per-frame, d_seg axis) and
the ξ pose-carrier `dxi`/`xi_stored` tables (per-pair, d_pose axis); both are pair-local (proven from the
forward, §2), both are already the substrate of the CPU-only byte-close path (§5 — no MLX-GPU selection
surface exists to violate the CPU-axis lesson). **The top-EV witness application is the ξ (dxi) terminal
click-polish on the pose axis** (ESTIMATED −0.005..−0.017 ΔS on the already-banked 0.127 floor,
rollback-guarded), a gradient-free sibling of the #383 conditioning-gated pose finish. Every application is
a TERMINAL post-launch stage (slots at 4a′/4c′ in the sealed terminal-band); **none blocks the pilot or the
launch.** Sidecar-folding has LIMITED witness applicability (the witness carries no PR101-style redundant
sidecar — the ξ is already a single folded table; the discipline it yields is "any polish output re-encodes
into the EXISTING section at the existing grid, never a new section").

---

## 1. PER-VEHICLE CLICKABLE-CODE INVENTORY (Q1) — every per-(pair|frame|class-pair) quantized stored quantity

`S = 100·d_seg + √(10·d_pose) + 25·|archive.zip|/37_545_489`. "Clickable" = a stored quantized quantity a
±k discrete step can perturb and re-score exactly through the real byte-close. Byte costs are MEASURED where
tagged (r1_dxi memo / byte-close accounting), else DERIVED from shape×dtype.

### 1.1 v7.5.2 SINGLE TRUNK (mod32cap; the sealed launch vehicle)

| stored quantity | shape | quantization grid | archive home | byte cost | score term | clickable? |
|---|---|---|---|---|---|---|
| **FiLM `code`** | **(2·P, mod_dim)=(1200, 32)** per-FRAME | int8 symmetric (`_int8_symmetric`), one brotli stream (`code_brotli`) | code blob | part of ~88 KB blob; raw 1200·32 = 38,400 int8 → brotli'd (MEASURED archive 89,772 B total, r1_dxi) | **d_seg** (via FiLM→φ→argmax); frame0 rows → d_pose | **YES — pair-local** (§2). d_seg-clickable = the **600 frame1 rows** (SegNet reads last frame) |
| **pose-carrier `dxi`** | (P, 6)=(600, 6) | int16 per-channel fixed-point, `q_levels=4096` (12-bit), delta_ar temporal-Δ + arithmetic coder | pose-carrier section | +3,433 B over no-dxi (MEASURED; total ξ section 7,195 B) | **d_pose** (ξ_eff→H→warp) | **YES — pair-local** (§2) |
| **pose-carrier `xi_stored`** | (P, 6)=(600, 6) | same ξ grid (shipped SUMMED with dxi as ξ_eff) | pose-carrier section | 3,762 B (no-dxi baseline) inside the 7,195 B | **d_pose** | **YES — pair-local**; but ships folded with dxi as one ξ_eff table (§4) |
| INR trunk weights (in_proj, film.weight/bias, hidden.*, out_sdf, out_tex, palette) | shared | int8 symmetric, one brotli stream (`base_brotli`) | base blob | dominates the ~88 KB blob | d_seg (+ tex→d_seg via chroma) | **NO locality** — a click touches ALL 600 pairs (weight-click class; §2). Matches PR128's weight-click rejection |

### 1.2 v7.5.3 / two-trunk W=(G,ξ,T) (adds texture trunk + analytic lane band)

Inherits v7.5.2's `code` + ξ tables verbatim, PLUS:

| stored quantity | shape | grid | home | byte cost | score term | clickable? |
|---|---|---|---|---|---|---|
| **texture trunk `out_tex` / Gabor head** (T) | 375 counted params (F=24 Gabor × K=5 × 3 + bias); the `_B` Gabor bank is rule-118 FREE | int8 | base blob (shared) | ~375 int8 (MEASURED design, #395) | d_seg (interior texture legibility) | **SHARED per-class head → NO pair-locality** (weight-click class). Its per-CLASS structure is a class axis, not a pair axis |
| **analytic lane-band coords** (#224 Wave E) | per-pair lane trajectory coeffs (openpilot poly), ~1–2 KB | manifold coords, counted | 5th lane block | ~1–2 KB (DERIVED, synthesis §1.4) | d_seg (lane band placement) | **CANDIDATE pair-local** — IF the coeffs are stored per-pair (needs per-pair confirmation; §6 build-list) |

### 1.3 v8 PER-CLASS carriers (explicit realization of W)

| class | stored quantity | shape/bytes | grid | score term | clickable? |
|---|---|---|---|---|---|
| **Movable(3)** | sparse-site coder (K=9 tracked-box slots, per-pair) | **6,289 B** MEASURED (−31% vs raw) | int-quantized site params | d_seg (Movable coverage) | **YES — per-pair-per-class** (site slots are per-pair); a second (class) axis |
| **Road(0)** | flat per-pair scene colour + horizon arc `y(x)` deg-3 | fill ~16 B; arc 4 coeffs + ξ intercept, 4167 B | int | d_seg + d_pose (horizon = ξ chart) | per-pair colour = clickable; arc coeffs are per-pair-per-class |
| **Undrivable(2)** | flat basin default + 3 lateral extent curves `x_L(y)/x_R(y)` | ~4 coeffs each + intercepts | int | d_seg | curve coeffs per-pair-per-class = clickable |
| **Lane(1)** | analytic band (training lever) + annulus-jitter residual (#333) | band ~1–2 KB | int | d_seg | per-pair band coeffs = candidate clickable |
| **MyCar(4)** | single static hood mask | ~0.1–0.5 KB, IoU 0.994 (#139) | — | d_seg | **STATIC — one store for all pairs → NO pair-locality** (weight-click class) |
| all-pairs ties | b_c = no_offset (5-scalar global, #386 RULED) | 0 B | — | d_seg | global 5-scalar → NO locality (already SATURATED/refuted) |

**Counts summary:** the pair-local clickable surface per vehicle — v7.5.2: **600 frame1-`code` rows (32-d
each) + 600 ξ pairs (6-d)**; v7.5.3: + per-pair lane-band coeffs (if per-pair); v8: + per-class-per-pair
site/curve/colour codes (Movable 6289 B, Road/Undriv curves) with a SECOND (class) axis. The SHARED
weight-click class (INR trunk, T head, MyCar static, b_c global) is NON-local in EVERY vehicle and is the
witness analog of PR128's rejected 229 K weight-code search.

---

## 2. PAIR-LOCALITY VERDICTS (Q2) — proven/refuted per code type from the actual forward

The diagonal-batching exploit requires candidate *i* to affect ONLY pair *i*'s score contribution. The
witness forward is `levelset_rgb_forward_numpy(params, feats, code_row, ...)`
(`lever_b_levelset_generator.py:717`), called PER PAIR/FRAME. The decisive line is **748**:

```python
film = (code_row @ p["film.weight"].T + p["film.bias"]).reshape(n_hidden, 2, hidden_dim)
...
phi = h @ p["out_sdf.weight"].T + p["out_sdf.bias"]   # (P,K) → argmax → d_seg
```

- **`code` (FiLM latent) — PAIR-LOCAL ✅ (PROVEN).** `code_row` is `params["code"][frame_row]` — the mod
  vector of ONE frame. It enters ONLY through `film = code_row @ film.weight.T`, which modulates ONLY that
  frame's hidden activations → ONLY that frame's `phi` → ONLY that frame's argmax partition → ONLY that
  pair's d_seg contribution. A ±k click on `code[2p+1, d]` changes exactly pair p's frame1 argmax and
  nothing else. Accepted clicks on different pairs **add exactly** (the metric is a mean over pairs, and the
  renders are independent). This is EXACTLY PR128's latent locality. verdict_scope: n/a (structural).

- **`dxi` / `xi_stored` (ξ pose twist) — PAIR-LOCAL ✅ (PROVEN).** ξ_eff = xi_stored[p] + dxi[p]; the
  homography H is derived per pair from the shipped ξ (`build_pose_carrier_section`, r1_dxi §1) and warps
  only pair p's frame0 → only pair p's d_pose. A ±k click on the 12-bit ξ grid for pair p is exactly local.

- **INR trunk weights / T head / palette — NON-LOCAL ❌ (PROVEN, weight-click class).** `params["film.weight"]`,
  `hidden.*`, `out_sdf/out_tex`, `palette` are SHARED across all pairs (`p[...]` outside the pair loop). A
  click touches all 600 pairs' renders — no per-candidate isolation, no diagonal batching. This is the
  witness analog of PR128's finding that "every weight-code click was rejected — the decoder is at a strict
  discrete local optimum." **DO-NOT-diagonal-batch these** (they are the #396 default-target class, confirmed
  one-candidate-per-render — see §3). verdict_scope: FORMULATION (a witness that is under-trained, unlike
  PR128's QAT-converged decoder, may still have exploitable shared-weight residual — but not via diagonal
  batching; via #396's existing shared-tensor mode).

- **v8 per-class fields — PAIR-LOCAL with a SECOND (class) axis ⚠️ (analyzed).** A per-class-per-pair code
  (Movable site slot, Road/Undriv curve coeff) touches only class c's φ_c contribution to pair p's argmax.
  **Across pairs: independent** (diagonal-batchable, 600 candidates/render). **Within a pair, across classes:
  NOT independent** — φ_c and φ_c′ compete at the same argmax, so clicking class c and class c′ on the SAME
  pair p interacts. The diagonal batch therefore stays **per-pair (one click per pair per render)**, sweeping
  the (class, dim, δ) candidate space ACROSS renders — richer geometry (K× more candidate types) but the
  same 600-exact-per-render locality. Shared per-class params (a curve's global coeffs, MyCar static mask)
  are weight-click class (non-local).

**Net:** the witness satisfies the diagonal-batching precondition on `code` (d_seg), ξ (d_pose), and v8's
per-class-per-pair codes; it FAILS it (correctly, matching PR128) on all shared weights.

---

## 3. EXACT-GATED TERMINAL POLISH AS A PIPELINE STAGE + the #396 unify-vs-separate decision (Q3)

### 3.1 Where it slots (both vehicles)

The sealed terminal-band (SPEC_v752 §2.1; fullstack synthesis §1.6) is:
`4a head-solve (#341 GN/CG) → 4a′ MC exact-metric ratchet (#396) → 4b pose-conditioning gate (#383) →
4c pose finish / banked-R1`. The click-polish slots as a **specialization of 4a′** (d_seg, over the `code`
table) and a **new 4c′** (d_pose, over ξ, gradient-free sibling of 4c). Both run AFTER training + byte-close,
BEFORE exact eval, through OUR real decode + frozen CPU scorers + real re-encoded bytes. **NEVER blocks the
pilot/launch** — it is a terminal post-training stage on a frozen checkpoint, resumable, rollback-guarded.

### 3.2 #396 vs the click-polish — understand both before claiming subsumption

I read `mc_finisher.py` in full. **#396 IS the exact-gated discrete/continuous accept/reject finisher
already** — the same monotone-ratchet-through-the-real-metric mechanism PR128 describes:
- `mode="int8"` already exists (±k on the code, clamped [-128,127] — PR128's discrete support, line 193).
- The accept ladder (SCREEN subset → CONFIRM n600 through-R → strict monotone ratchet, `_try_batch`) is
  exactly PR128's "score each candidate exactly, keep only if the exact metric strictly improves."
- `ΔS = 100·Δd_seg + 25·Δbytes/37_545_489` with an injected `byte_cost_fn` is exactly PR128's rate-aware
  net-improvement gate (line 504, `s_component`).
- Resumable (atomic npz + JSONL), deterministic, provenance-stamped, P9-honest (CONFIRM = the ONLY authority).

**What #396 does NOT do today, and PR128's exploit needs:**
1. **Targets shared HEAD tensors** (`DEFAULT_PARAM_TARGETS = out_sdf/out_tex/palette` — line 61). These are
   the NON-LOCAL weight-click class (§2). So #396 today confirms **one candidate per n600 render** — it does
   NOT exploit locality because its targets have none.
2. **No diagonal-batch measure.** `ProposalEngine.propose()` picks one tensor + a subset + deltas → ONE
   candidate → ONE confirm. To realize PR128's 600-candidates-per-render, the measure must apply the SAME
   (dim, δ) click to the code column across ALL 600 pairs and return the **per-pair d_seg VECTOR** (not the
   mean), then accept the per-pair subset that improves.
3. **Whole-proposal accept** (with bisect salvage), not per-pair accept.

**RECOMMENDATION — UNIFY (one surface, no duplicate apparatus).** The click-polish is a specialized
high-throughput MODE of #396's family: `int8` mode (exists) + a per-pair/per-frame code-table target class +
a diagonal-batch `measure_fn` that returns the per-pair d_seg vector + a per-pair independent accept mask.
It SHARES #396's ratchet, resume, provenance, byte-cost accounting, and P9 authority discipline verbatim.
**Build it as a new mode in `src/tac/through_r/mc_finisher.py`** (e.g. `PairLocalDiagonalFinisher` /
`targets="code"` + `diagonal=True`), NOT as a parallel `tools/latent_click_polish.py`. Rationale
(canonicalization discipline, CLAUDE.md "Results must become system intelligence" + gate-consolidation):
one exact-metric finisher, one authority path (`measure_through_r`), one duty-to-measure ledger row, one set
of confound guards. **#396's `FinisherProblem` decoupling (injected `render_fn`/`measure_fn`/`byte_cost_fn`)
is PRECISELY the seam that makes this a mode, not a fork** — inject the diagonal per-pair d_seg measure for
the `code` target, and (for the pose axis) inject the byte-close d_pose measure for the ξ target.

Neither subsumes the other cleanly: #396 handles fp32 continuous micro-steps, bisect salvage, guided
proposal over shared tensors (the click-polish has none of these); the click-polish exploits locality for a
~600× throughput multiplier (#396 has none of this). They are complementary → one module, two modes.

**SCOPE NOTE:** the SEPARATE `tools/latent_click_polish.py` proposed in the PR128 intake draw-from #1 (against
OUR borrowed PR110 HNeRV frontier payload) is a DIFFERENT substrate (HNeRV latents, not the witness) and a
borrowed-substrate defensive bank — it stays a distinct tool. The unify recommendation is scoped to the
WITNESS finisher only.

---

## 4. SIDECAR-FOLDING FOR THE WITNESS (Q4)

**What "fold" means (PR101/PR128):** a stored correction sidecar folds into base codes ONLY when the
corrections are representable on the base quantization grid — then the sidecar section is deleted at zero
distortion cost (PR128: −605 B). It worked because the sidecar and base shared a representation space.

**Witness applicability — LIMITED (honest):**
- **ξ (pose): ALREADY a single folded table.** The byte-close ships ξ_eff = xi_stored + dxi coded together
  as ONE section (7,195 B) — there is no separate sidecar to delete. The dxi's +3,433 B over the no-dxi
  baseline is NOT sidecar overhead; it is the genuine coding cost of the trained residual's high-frequency
  content (the r1_dxi memo: "the per-pair dxi adds high-freq jitter that kills the temporal-delta
  smoothness"). So there is **no free fold** here — the bytes are the information. verdict_scope: INSTANCE
  (this R1 ξ). A finer/coarser ξ grid is a rate/pose tradeoff for the click-polish to explore, not a fold.
- **FiLM `code`: one table, no sidecar.** No PR101-style per-pair single-dim correction sidecar exists in
  the witness archive → nothing to fold.
- **The discipline this yields (the real deliverable):** any click-polish OUTPUT (a correction table) MUST
  re-encode into the EXISTING section at the EXISTING grid (into `code_brotli` / the ξ section), **never as a
  new archive section** — else the diagonal-batch's rate gain is eaten by ~150 B/section ZIP+header overhead
  (the L20 monolithic-section discipline). This is "born-folded": the click IS a re-quantization of an
  existing code, so it folds by construction. Registered as a byte-close gate: post-polish archive section
  COUNT must equal pre-polish section count (no new section).
- **v8:** the per-class carriers are already edge-centric-deduplicated (one home per fact, `owns_explicitly`
  lists; synthesis §1.4). A per-class click-polish must likewise re-encode into the owning class section, and
  the standing pairwise NON-DERIVABILITY byte-close audit (#385 §5) already enforces no-redundant-bytes — the
  fold discipline is subsumed by that gate.

---

## 5. THE CPU-AXIS SELECTION LESSON (Q5) — grep verdict: CLEAN, no violating surface

PR128's lesson: GPU-selected clicks lost ~30% of their gain on CPU (bicubic LSB flips borderline
judge-pixels) → select on the authority axis. I grepped the byte-close + finisher paths:

- **Byte-close render/R/eval (`levelset_byte_close_and_eval.py`): CPU/numpy end-to-end.** The render is
  `levelset_rgb_forward_numpy` (numpy fp32, float64 accumulation, argmax-stable); the R operator is the
  frozen CPU-torch `_torch_R_to_camera_uint8` (from `twr`); the eval device resolves via `--device`
  (`resolve_eval_device`, CPU authority; CUDA only on Linux x86_64). **No MLX / `mx.` in the path.**
- **Through-R harness (`harness.py`): `SUPPORTED_BACKENDS = ("cpu-torch",)` — the ONLY authority** (line 52);
  `load_frozen_segnet` → `load_real_segnet("cpu")`; `.eval()` deterministic, chunk-size-independent.
- **#396 CONFIRM path:** wraps `measure_through_r` → cpu-torch. Its SCREEN is a CPU subset (not GPU), so even
  the non-authority screen is CPU — no cross-axis selection risk.
- **MLX is confined to the TRAINER's forward/gradient device** (`train_levelset_witness_realized_through_R_mlx.py`),
  which is NEVER an authority (memory L53, L70; MPS/MLX = gradient/speed only).

**Flag: NONE.** The witness byte-close/verdict/finisher surface is CPU-locked by construction. The forward
recommendation: keep the diagonal-batch SCREEN on a CPU subset, and NEVER add an MLX-GPU screen for speed
(it would re-introduce the exact bicubic-LSB axis gap PR128 measured). This is a design constraint on the
build, not a current defect.

---

## 6. EV + SEQUENCING (Q6)

### EV table — honest bands, derivation-labelled

PR128 anchor: ~1,565–2,162 net code changes → Δd_seg −2.72e-5 (score −0.00272) on a near-optimal HNeRV
payload = ~3,200 net flips (0.0000272·600·384·512). The witness d_seg floor unit = one confirmed flip =
`SEG_WEIGHT/(600·384·512)` = **8.48e-7 ΔS** (`delta_s_floor_per_confirmed_flip`, DERIVED). The witness is
NOT at a discrete optimum (mod32cap d_seg 0.0034-class, ~6× PR128's 0.00053, and lane-erasure-limited, not
smooth-latent-limited), so its terminal residual is LARGER but STRUCTURALLY harder to click (lane dashes are
erased below the argmax margin — margin-saliency #141 says the flip-prone mass is thin/high-curvature, which
is exactly where discrete clicks CAN help but where the frozen-SegNet label-noise floor also lives).

| vehicle | axis | mechanism | ΔS band | grade + derivation |
|---|---|---|---|---|
| **v7.5.2/3** | **d_pose** | **ξ (dxi) diagonal click-polish** (TOP EV) | **−0.005 .. −0.017** | **ESTIMATED** — marginal of √(10·d_pose) at d_pose 0.00161 is high; a plausible 0.00161→0.0012 gives 0.127→0.110 = −0.017; conservative floor if only half the pairs click. Rollback-guarded vs banked 0.001610 (never worse). Derived from the r1_dxi banked anchor + the √ marginal law |
| v7.5.2/3 | d_seg | `code` frame1 diagonal click-polish (4a′) | −0.0003 .. −0.0015 | ESTIMATED — PR128's ~3,200-flip yield scaled to the witness's larger-but-structurally-harder residual (lane-erasure). Upper bound assumes flip-yield ≈ PR128; lower bound reflects the frozen-SegNet label-noise floor eating thin-lane clicks (#141) |
| v7.5.3 | d_seg (lane) | lane-band per-pair coeff polish | −0.0002 .. −0.001 | ESTIMATED — gated on per-pair clickability confirmation; small coeff table |
| v7.5.3 | rate | (fold discipline, not a gain) | ~0 | DERIVED — no PR101-style redundant sidecar in the witness; ξ already one folded table (§4) |
| **v8** | **d_seg** | per-class-per-pair diagonal polish | **−0.0005 .. −0.002** | ESTIMATED — richer class×pair geometry (Movable 6289 B site slots are the coverage enemy, Unit A); gated on v8 byte-close maturity |

**Honesty:** all bands are ESTIMATED (no witness click-polish has been measured). The ONLY measured anchor is
PR128's cross-substrate yield; the witness's structural difference (lane-erasure, not near-optimal smooth
latents) makes the d_seg band wide and the lower bound genuinely possible. The ξ/pose band is the most
defensible because it rides a MEASURED banked floor (0.127) with a high √-marginal and a hard rollback guard.

### Sequencing (bias: land a lower exact score soonest, never block the launch)

- **ALL post-launch, terminal-band stages** — a frozen-checkpoint polish, resumable, rollback-guarded. It
  rides the byte-closed n600 row; it does NOT gate the pilot or the launch (which are the pointer-movers).
- Order within the terminal band: `4a head-solve → 4a′ code diagonal polish (d_seg) → 4b pose gate →
  4c pose finish OR 4c′ ξ diagonal polish (d_pose, gradient-free alternative to 4c) → byte-close → exact eval`.
- The ξ/pose polish (4c′) is the highest-EV and composes with (or substitutes for) the #383 gradient pose
  finish — both target the pose terminal, both CPU-gated, both rollback-guarded vs 0.001610.

### The ONE next build step per vehicle

- **v7.5.2:** extend `src/tac/through_r/mc_finisher.py` with a pair-local DIAGONAL mode over the `code` table
  (frame1 rows) — per-pair d_seg vector measure + per-pair accept. `[needs-build]` (substrate READY: byte-close
  + `code` table + through-R harness all exist).
- **v7.5.3:** the SAME diagonal mode + inject the byte-close d_pose measure for the ξ target (4c′). `[needs-build]`.
- **v8:** per-class-per-pair diagonal target once the v8 per-class carrier byte-close path exists. `[gated-on v8 chain]`.

---

## 7. BUILD LIST (with {READY | needs-build | gated-on} tags)

1. **[needs-build]** `src/tac/through_r/mc_finisher.py` — add a **pair-local DIAGONAL mode**: a `code`-table
   (per-frame) target class + diagonal candidate expansion (one (dim, δ) click applied to a code column across
   all 600 pairs per render) + per-pair independent accept mask. SHARES the existing ratchet/resume/provenance/
   int8-mode/byte-cost. This is the unify decision (§3). Substrate READY (byte-close + `code` + harness exist).
2. **[needs-build]** `src/tac/through_r/harness.py::measure_through_r` — expose the **per-pair d_seg VECTOR**
   (it already computes per-pair argmax; currently returns the mean). Small; the diagonal accept needs it.
3. **[needs-build]** Pose-axis 4c′: inject a **byte-close d_pose measure_fn** (evaluate CPU on the shipped
   `.raw`) as the ξ diagonal finisher's authority; target the ξ q_levels=4096 grid; rate-aware accept via the
   ξ coder byte-delta (r1_dxi accounting); **rollback-guard vs banked 0.001610** (never ship worse ξ).
4. **[READY]** Register the finisher mode as a **TOOL (not a Lever)** per #396's classification (fullstack §1.6)
   with a duty-to-measure ledger row + its P7 falsifiers pre-registered: (a) diagonal-batch net ΔS ≤ 0 → no
   gain at FORMULATION scope; (b) guided ≥ blind accept-rate (inherit #396's own falsifier); (c) post-polish
   archive section-count == pre-polish (the born-folded / no-new-section gate, §4).
5. **[gated-on v8 chain]** Per-class-per-pair diagonal target (Movable site slots, Road/Undriv curve coeffs)
   once the v8 per-class carrier byte-close exists; the second (class) axis sweeps across renders, per-pair
   accept unchanged; skip shared per-class coeffs + MyCar static (weight-click class).
6. **[READY, SEPARATE — do NOT fold into #396]** The PR128 draw-from #1 `tools/latent_click_polish.py` against
   OUR borrowed PR110 HNeRV payload is a DIFFERENT substrate + a borrowed-substrate defensive bank; it stays a
   distinct tool with its own byte-close. (Listed for completeness / de-confliction, not part of the witness line.)

**Not-owed / explicitly N/A:** sidecar-folding as a rate LEVER (§4 — the witness carries no redundant sidecar;
the fold is a born-folded discipline subsumed by build-item 4c + the #385 non-derivability gate). CPU-axis
guard (§5 — clean by construction; the only owed item is the design constraint "screen on CPU, never MLX",
carried into build-item 1).

---

## Triality legs + honesty

- **DAG:** `### FEED-clickpolish-witness-design` appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **DSL:** N/A-with-rationale at this landing — this is a DESIGN memo; the finisher MODE + its duty-to-measure
  registration land at BUILD time in `mc_finisher.py` (a TOOL, not a DSL Lever, per #396's classification), so
  there is no DSL leg to drift. Every named mechanism is either #396 (built) or explicitly `[needs-build]`.
- **equations:** DEFERRED-to-first-measured-row (stated) — no canonical equation is registered here because
  every ΔS band is ESTIMATED (no witness click-polish measured). The first measured diagonal-polish n600 row
  registers the anchor (candidate law: `pair_local_diagonal_click_polish_dseg_v1` / `..._dpose_v1`), staying
  COUNCIL-FLAGGED until its anchor lands (SPEC_v8 §5 discipline).

**Pointer contest-CPU 0.19110 UNMOVED — this synthesis is MEANS (design-only, no launch, dual-chain wall
stands). Only a byte-closed `upstream/evaluate.py` n600 exact row < 0.19110 moves it.**

---

## Observability surface

*(OBSERVABILITY-ADDENDUM 2026-08-25 — APPEND-ONLY per Catalog #110/#113. This
section is an INDEX into this memo's own content per Catalog #305's 6 facets;
it adds no new claim. Facets with no counterpart in this memo say so plainly.)*

1. **Per-layer inspection** — §1 "PER-VEHICLE CLICKABLE-CODE INVENTORY (Q1)" enumerates every per-(pair|frame|class-pair) quantized stored quantity for each of the three vehicles (§1.1 v7.5.2, §1.2 v7.5.3/two-trunk, §1.3 v8 per-class), which is the per-layer inventory this design is polished against.
2. **Per-signal decomposition** — §2 "PAIR-LOCALITY VERDICTS (Q2)" gives proven/refuted per code type from the actual forward; §6 "EV table — honest bands, derivation-labelled" decomposes expected value per candidate.
3. **Run-to-run diff** — §3 "EXACT-GATED TERMINAL POLISH AS A PIPELINE STAGE" places the polish as a named stage (§3.1 where it slots in both vehicles), so a polished build differs from its base at exactly that stage.
4. **Post-hoc query** — `tools/levelset_byte_close_and_eval.py` produces the byte-closed `archive.zip`; `canonical_frontier_pointer.json` is the frontier query surface; `lever_b_levelset_generator.py` / `harness.py` are the named code surfaces.
5. **Cite-chain** — §5 "THE CPU-AXIS SELECTION LESSON (Q5)" records the grep verdict (CLEAN, no violating surface); the closing "Triality legs + honesty" section carries the DAG/DSL/equations chain.
6. **Counterfactual hooks** — §3.2 "#396 vs the click-polish — understand both before claiming subsumption" is an explicit two-mechanism counterfactual; §7 "BUILD LIST" tags every item {READY | needs-build | gated-on}.
