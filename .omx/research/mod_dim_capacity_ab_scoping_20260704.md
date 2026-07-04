# mod-dim / boundary-capacity A/B — SCOPING (gated, NOT launched)

**Date:** 2026-07-04 · **Task:** #299 · **Status:** SCOPED, fires ONLY on {ep75+ capacity-confirm}
∧ {operator GO}. Pointer 0.19110 UNMOVED (this is a DISAMBIGUATOR, means not end).

## The question (operator, 2026-07-04)
Why mod-32 for #205? Is there a principled optimum between 19 and 32? Could mod-19 still be
optimal with enough time / convergence support? How much must CE prime for the rest of the
schedule?

## The live signal that triggered this
Fresh seeded run (`levelset_n600_witness_20260704T174257Z`, mod-19): CE d_seg **FLAT** 0.02868
(ep25) → 0.02874 (ep50) while ep_loss fell 410→359 (−12%). #205 (mod-32) at the same epochs was
0.0103→0.0078 and DESCENDING. Surrogate improves, hard verdict pins = the decoupling, in CE.

## Proactive-recall CORRECTION (this is why the naive "bump n-dir-freqs" is WRONG)
The fresh run is NOT "isotropic mod-19." Its basis config: `--self-orient --n-dir-freqs 2
--mod-dim 19`. Two prior MEASURED findings reshape the hypothesis:
- **#277 / along-tangent memory (2026-07-03):** the lane-dash residual root cause is an
  along-tangent frequency deficit; the lever is `--n-dir-freqs 2→4`. So directional bandwidth
  is a genuine, measured d_seg axis — and the fresh run sits at the LOW end (2).
- **Aliasing coupling (DAG, measured $0 numpy):** at `--freq-across 32`, self-orient octaves are
  {32,64,128,256,512,1024}; only {32,64} are clean — the rest FOLD at the 512 render grid. So
  `--n-dir-freqs 2` @ freq-across 32 was chosen ANTI-ALIAS-correctly. Going 2→4 REQUIRES coupling
  `--freq-across` down to ~8 (octaves {8,16,32,64}, all clean). A naive n-dir-freqs bump at
  freq-across 32 aliases — a self-inflicted wound our own prior work already caught.

## The three candidate causes of the CE plateau (to be separated)
1. **Isotropic embedding capacity** — mod-19 (~Whitney 2·8+1=17) sits BELOW the trainability-critical
   over-parametrization ratio; the leaner basis fits the smooth bulk (loss falls) but can't sharpen
   the boundary partition (d_seg pins). #205's 0.00475 CE floor was at mod-32 (4× over intrinsic,
   lazy-training regime). This costs RATE (~68% more per-pair payload).
2. **Along-tangent directional bandwidth** — n-dir-freqs 2 starves the boundary-TANGENT axis the
   dashes live on (#277). Fix = n-dir-freqs 4 @ freq-across 8 (anti-alias-coupled). NEARLY RATE-FREE
   (a few directional Fourier features). The cheap win if it's the cause.
3. **Convergence-limited (not capacity)** — mod-19 CAN reach the floor but CE at ep50 hasn't; needs
   freq-warmup / margin-weighted-CE (#141) / more CE epochs (tau onset is ep300 — 250 CE epochs
   remain, so time is not yet the binding constraint). If this, the A/B is deprioritized and the
   fix is a convergence-support lever, not a capacity bump.

## The decisive design — 3 arms (everything else = fresh_seeded, CE-only, same seed)
| arm | mod-dim | directional | tests | rate cost |
|---|---|---|---|---|
| **A** (have it) | 19 | self-orient n-dir-freqs 2 @ fa32 | = the fresh run plateau (MEASURED) | baseline |
| **B** | 19 | n-dir-freqs 4 @ **freq-across 8** (anti-alias-coupled) | along-tangent bandwidth | ~free |
| **C** | 32 | n-dir-freqs 2 @ fa32 (as fresh) | isotropic capacity, un-confounded vs #205 | +~68% payload |

Vary ONLY {mod-dim, n-dir-freqs+freq-across}. Everything else pinned to fresh_seeded (paint+seed+
eikonal ON, same seed, CE-only). #205 is a CONFOUNDED mod-32 proxy (differs in seed/paint) — arm C
is the clean one.

**Reading the 2×2:** B breaks plateau → directional was the wall (cheap, ship it). C breaks →
capacity was the wall (pay rate, or find the anisotropic-cheaper equivalent). Neither → convergence-
limited (pivot to freq-warmup/margin-CE). Both → both bind (take the cheaper sufficient one).

## MVP-first phasing (allergic-to-toys honored: disambiguator → n600 confirm)
- **Phase 1 (regime disambiguator, LABELED provisional):** n192 CE-only to ep150, 3 arms sequential
  (~1–1.5 h/arm). Measures per arm: d_seg(ep) trajectory SHAPE, per-class d_seg (is the LANE
  forming? — the tau-priming question), loss↔d_seg decoupling gap, time-to-break, byte cost. This
  is a REGIME + RANK disambiguator, NOT a score claim (reduced-n is legal for regime-finding per
  MVP-first; the winner is confirmed at n600).
- **Phase 2 (evidence):** the single winning arm at FULL n600 to the CE floor + into tau,
  byte-closed → the real row.

## CE-priming criterion (the deepest measurement)
CE's job is not the final d_seg — it is to hand tau a boundary tau can SHARPEN not ERODE (tau =
MCF, erodes sub-critical structure — the nucleation lesson). So the Phase-1 pass-metric is
**per-class boundary MASS above tau's critical nucleus**, esp. the thin lane — NOT just total
d_seg. An arm that lowers total d_seg but leaves the lane sub-critical still fails tau.

## Gating + containment (binding)
- Fires ONLY on: ep75+ confirms capacity-limited (d_seg still flat, monitor → DIVERGING/plateau not
  transient) AND operator GO. If ep75 breaks downward → convergence-limited → DEPRIORITIZE this A/B,
  pivot to convergence-support levers.
- **Memory:** NO concurrent heavy. If capacity-confirmed, preserve+stop the fresh run first (frees
  ~77 GiB), THEN run arms governed via `tools/launch_witness_run.py` (the governed launcher, the
  policy-derived safe-frac). Reduced-n arms are small enough that the governor MAY admit one
  alongside the fresh run — but default to preserve-then-measure per containment.
- Every arm: resumable + per-stage ckpt, TAC_MEM_PROBE=1, deterministic seed, governed launcher.

## Deliverable
A `d_seg-floor × mod-dim × n-dir-freqs × rate` frontier — the empirical answer to "is there an
optimum between 19 and 32, and is it a scalar or an anisotropic reallocation." Pointer moves only
through the Phase-2 n600 winner byte-closed through `upstream/evaluate.py`.
