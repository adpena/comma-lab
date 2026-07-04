# $0 DISAMBIGUATOR BATTERY — RESULTS (independent adversarial verifier)

**Date:** 2026-07-04 · **Task:** #299 disambiguator (council memo
`council_grand_symposium_ce_plateau_20260704.md`). **Role:** INDEPENDENT adversarial verifier
(the parent designed the mod-dim capacity A/B; this pass MEASURES which of five causes is real
and challenges the capacity framing). **Axis:** `[macOS-CPU advisory] NON-PROMOTABLE` — frozen
CPU-torch SegNet argmax (the SAME advisory authority the trainer verdict uses); MLX/MPS never
touched. **Pointer 0.19110 UNMOVED — this is a DISAMBIGUATOR (means, not end).**

## Headline verdict (one line)
**The CE plateau is NOT a capacity problem (cause 1), NOT along-tangent bandwidth (cause 2), and
NOT will-fix-with-more-CE convergence (cause 3). It is a SEED-COMPOSE ISLAND-GRADIENT STARVATION +
NON-COMPARABLE-BASELINE artifact (cause 5, with a specific mechanism): the seeded witness has NO
gradient pathway to form the two seeded classes (Lane, Movable) itself, and the witness-alone
CE-floor watch is measuring the deploy-absence of a seed that carries 71% of the island d_seg
during training. DO NOT fire the mod-dim capacity A/B. RE-POINT the watch and FIX the absorption
pathway.**

## Harness (VALIDATED — the self-consistency / bug gate, Rudin D1-adjacent)
Reproduced the trainer's witness-alone deploy verdict from each run's EMA npz: int8-dequant
(`int8_dequant_params`) → curvelet + self-orient fixed-point feats → `levelset_rgb_forward_numpy`
(the ONE CODEPATH) → `_torch_R_to_camera_uint8` → frozen CPU-torch SegNet argmax → per-GT-class
decomposition. Subset K=24 pairs (LABELED PROVISIONAL regime disambiguator; legal per MVP-first —
NOT a score claim). Validation: my subset d_seg reproduces BOTH runs' logged n600 verdicts within
subset sampling noise, so the render→SegNet→per-class apparatus is trustworthy:

| run | my K=24 subset d_seg | logged n600 verdict |
|---|---|---|
| fresh mod-19 ep75 | 0.03067 | 0.02886 (ep75) |
| #205 mod-32 ep300 | 0.00399 | 0.00476 (ep300 CE floor) |

(The small subset gaps are first-24-pair sampling, NOT a render bug — both drift the same
direction; the per-class STRUCTURE below is the robust signal.)

## Per-probe results

| # | probe | status | verdict |
|---|---|---|---|
| **D3** | per-class d_seg decomposition | **MEASURED** | plateau is 71% Lane+Movable, both at **100% within-class flip**; bulk (Road/Undrivable/MyCar) SOLVED. Aims everything. |
| **D4** | seed-shield audit + #205 witness-alone-equiv | **MEASURED + code audit** | verdict IS correctly witness-alone (deploy-faithful, seed excluded — HONEST). BUT #205 has **no seed** → **non-comparable baseline CONFIRMED**: #205's witness FORMED islands during CE; fresh's cannot. |
| **D1** | gradient-flow: is θ_island grad dead? | **mechanism-established** (code) + realized signature; direct autograd magnitude = `pending_harness` | Effectively starved, but NOT a bare-`.round()` bug — it is the seed-compose satisfying every seg loss on the island. Causal pathway PROVEN from code (see Mechanism). |
| **D2** | overfit-one-pair mod-19 vs mod-32, no rate | `pending_harness` (needs MLX training; CONTAINMENT: live GPU run active) | Capacity refuted CIRCUMSTANTIALLY (mod-19 solves bulk; produces misplaced-but-nonzero island phi). D2 now LOW priority — the fix is the absorption pathway, not mod-dim. |
| **D5** | margin-weighted-CE micro-probe | `pending_harness` — **but reframed INERT by the mechanism** | CE (and margin-weighting) is computed on the seed-COMPOSED frame → satisfied by the seed → cannot drive the witness. The real lever is a witness-ALONE island loss / seed anneal, not CE re-weighting. |

## D3 — per-class decomposition (the aiming probe; K=24, VALIDATED harness)

**FRESH (mod-19, seed) ep75 — subset d_seg 0.0307:**

| class | GT area | within-class flip | % of d_seg | flips go to |
|---|---|---|---|---|
| Road | 0.228 | 0.011 | 8.1 | Undrivable |
| **Lane** | 0.0063 | **1.000** | **20.7** | → Road |
| Undrivable | 0.494 | 0.012 | 19.6 | → Road |
| **Movable** | 0.0153 | **1.000** | **50.0** | → Road / Undrivable |
| MyCar | 0.256 | 0.002 | 1.6 | — |

**Lane + Movable = 71% of the plateau, BOTH at EXACTLY 100% within-class flip** = the deployed
witness produces ZERO realized Lane/Movable that SegNet reads correctly. These are EXACTLY the two
seeded island classes (`island_classes=[1,3]`, `lane_cls=1 movable_cls=3` from the run log). The
three NON-seeded classes are all solved (<1.2% flip). This 1-to-1 correspondence (100% flip ⇔
seeded class; solved ⇔ non-seeded) is the smoking gun — an isotropic capacity wall would elevate
ALL boundary classes gradually, not zero-out exactly the two seeded ones.

**#205 (mod-32, NO seed) ep300 — subset d_seg 0.0040:**

| class | GT area | within-class flip | % of d_seg |
|---|---|---|---|
| Road | 0.228 | 0.0076 | 43.8 |
| **Lane** | 0.0063 | **0.203** | 32.3 |
| Undrivable | 0.494 | 0.0011 | 14.1 |
| **Movable** | 0.0153 | **0.0125** | 4.8 |
| MyCar | 0.256 | 0.0008 | 5.0 |

#205's witness FORMED Lane (80% correct) and Movable (98.75% correct) itself, with NO seed. Its CE
d_seg DESCENDED monotonically through CE (0.0103 ep25 → 0.0078 ep50 → 0.00476 ep300), i.e. the
witness built the islands. The fresh run's witness-alone d_seg is FLAT/creeping (0.02868 ep25 →
0.02886 ep75) because the seed does that job and is deploy-excluded.

## D4 — the non-comparability (CONFIRMED — this dissolves the "fresh should beat #205" premise)

The verdict path (`realized_verdict` → `_render_numpy_deploy`) renders the EMA shadow only; the
seed is a SEPARATE module (not in EMA/blob/deploy). So the witness-alone verdict is CORRECT and
deploy-honest. **But that is exactly why the comparison is invalid:** fresh's witness trained WITH
a seed carrying the island classes (excluded at verdict); #205 had no such crutch and had to form
everything. `within-flip` Lane 1.000 (fresh) vs 0.203 (#205); Movable 1.000 vs 0.0125. "Fresh CE
0.0287 should beat #205 CE 0.0078" compares a seed-crutched witness scored without its crutch to a
witness that formed everything itself. The parent's premise is a **non-comparable baseline**
(Assumption-Adversary cause 5, CONFIRMED). Tishby/Contrarian are RIGHT: the CE-floor watch is
pointed at the wrong quantity.

## Mechanism (why the witness cannot form the islands — code, D1)

`_compose_chain` (train_levelset_witness…:2143) adds `seed_mod.residual[pi] * mask[pi]` to the
witness RGB on frame1 (the SegNet-scored frame), masked to the Lane+Movable island support. EVERY
realized-through-R seg lever reads this SAME composed frame `_f1`/`_slog` (…:2570):
base CE (`base_loss`), lane-edge (LEVER-3), margin-saliency (LEVER-4), **island-amplify**
(`amplify_w=1.0`, on the composed `_signed`), AND **persistence** (`persist_gate["w"]`, on the
composed `_slog`; weight ramps 0→1 over ep0–300, = 0.25 at ep75 — NOT delayed, but it too reads
the composed frame). On the island pixels the composed frame is already correct (the seed carries
it) → every seg loss ≈ satisfied → `∂L/∂(composed) ≈ 0` → since `composed = witness + seed`,
`∂L/∂witness ≈ 0` on the island → the witness gets ~no gradient to form Lane/Movable. There is NO
seg lever measured on the witness-ALONE render. The "witness absorbs the seed" design intent has
no gradient pathway to drive it; the seed PREVENTS its own absorption. d_seg CREEPING UP (not down)
during CE is consistent — the witness is not forming the islands and will not under more CE.

**Nuance (witness-own phi probe, reported for honesty):** the fresh witness's OWN phi-argmax does
produce Lane/Movable pixels (3.3%/2.2% area) but MISPLACED (Road under-produced 0.07 vs GT 0.23,
Undrivable over 0.64 vs 0.49); #205's phi has near-zero lane (0.014%) yet 80% realized-lane-correct.
So phi-argmax is a POOR proxy — the realized RGB appearance (palette+tex through R→SegNet) is what
SegNet reads, and that is the authority (d_seg). The realized per-class table above is the verdict;
the phi probe only shows the fresh partition is misorganized on exactly the starved classes.

## The five causes — adjudicated
1. **Isotropic capacity (mod-19):** REFUTED. Class-confined 100%-flip signature ≠ capacity;
   mod-19 solves all bulk boundaries; #205's advantage is the ABSENCE of a seed, not +13 mod-dim.
2. **Along-tangent bandwidth (n-dir-freqs):** REFUTED as the plateau cause. That lever tunes
   lane-DASH fine structure; it cannot explain zero realized island across whole classes.
3. **Convergence-limited (more CE fixes it):** REFUTED. The island-formation gradient is
   structurally absent (all levers free-ride the seed); d_seg is creeping UP.
4. **Dead-gradient BUG:** PARTIALLY — θ_island IS effectively starved, but it is a DESIGN gap
   (seed-compose satisfaction), not a bare-`.round()` typo.
5. **Non-comparable baseline:** CONFIRMED and PRIMARY.

## What the parent should do (challenging the capacity framing)
1. **DO NOT fire the mod-dim capacity A/B (arms B/C).** It tests capacity/bandwidth; the plateau
   is neither. Spending GPU on it would repeat the "wall dissolved into an artifact" pattern.
2. **RE-POINT the watch** (Tishby/Contrarian CONFIRMED): total witness-alone d_seg during CE is
   uninformative while the seed carries islands. Watch the **witness-alone REALIZED per-class
   island flip** (currently 100% for Lane/Movable) — does it fall as CE proceeds? If it stays
   ~100%, the witness is not absorbing and deploy will fail on those classes.
3. **FIX the absorption gradient pathway (THE lever, not mod-dim):** every seg lever reads the
   seed-composed frame. Either (a) also apply the island-formation levers (amplify / persistence /
   a CE term) on the WITNESS-ALONE render so they actually push the witness, or (b) ANNEAL the seed
   residual → 0 during CE (e.g. decay to 0 by ~ep250) so the witness is forced to take over before
   tau erodes sub-critical structure. Without one of these, deployed Lane+Movable stay broken.
4. **Movable (50% of d_seg) is the highest-risk deploy gap:** Lane has the analytic
   `--lane-render-band` (ep350, free rule-118 fallback); **Movable has NO such fallback** and
   relies entirely on witness absorption. Verify Movable absorption is even achievable under the
   current design before trusting the run's final score.
5. **D2 (mod-19 no-seed overfit capacity oracle)** is now LOW priority (capacity refuted
   circumstantially). Run it only if the absorption fix does not resolve the island flip — and it
   is an MLX-training probe (`pending_harness` under CONTAINMENT while the live GPU run is active).

## Recursive self-reflection (Catalog #363, Round-2/3 — no verdict from an unrun probe)
- D3/D4: `VERIFIED_VIA_EMPIRICAL_ANCHOR` (K=24 measured; harness reproduces both logged n600
  verdicts). Subset caveat labeled; structure is robust.
- D1 mechanism: `VERIFIED_VIA_SOURCE_INSPECTION` (compose + all-levers-read-composed-frame quoted
  from the trainer); the direct autograd magnitude is `ASSUMED_AWAITING_VERIFICATION` →
  `pending_harness`, NOT claimed as a measured number.
- D2/D5: `pending_harness` — explicitly NOT asserted as results. D5 additionally reframed inert by
  the D1 mechanism (CE reads the composed frame).
- Adversarial self-challenge answered: the witness-own-phi probe was run BECAUSE it could have
  falsified the mechanism (if the witness produced correct island phi that R washed out, the fix
  would be palette/R, not gradient). It did not falsify — it showed a misorganized partition on the
  starved classes, consistent with gradient starvation.

**Pointer 0.19110 UNMOVED.** Artifacts (provisional, advisory): scratchpad
`d3_d4_perclass.py` / `witness_own_argmax.py`; per-class JSON `fresh_clean.json` / `pr205_clean.json`.
