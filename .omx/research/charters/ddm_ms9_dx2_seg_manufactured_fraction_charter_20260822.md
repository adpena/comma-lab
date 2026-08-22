# ddm_ms9_dx2_seg_manufactured_fraction — the seg term is worth 30,248 B of archive budget, and whether it is MANUFACTURED or REPRESENTATIONAL has only ever been measured on a vehicle six pointer moves ago

## MANDATE

**The reframe this arm serves (verified, derive it once yourself).** The zero-distortion archive
ceiling is 0.12·37,545,489/25 = **180,218.3 B**; DX2 ships **180,368 B**. The gap at zero distortion is
**149.7 B**. So the campaign's "shed 42,382 B" and "shed 150 B while eliminating all distortion" are
the SAME demand read two ways, and every unit of distortion is interchangeable with archive bytes at
**6.658e-7 S/B**:

| term | S | worth, in archive bytes | share of the 0.0282198 gap |
|---|---:|---:|---:|
| seg 100·0.00020139 | 0.020139 | **30,248 B** | **71.4%** |
| pose √(10·6.37e-6) | 0.0079812 | 11,987 B | 28.3% |
| distortion total | 0.028120 | 42,235 B | 99.65% |

Every rate arm on 08-22 held distortion FIXED and attacked bytes. Four axes closed at 0 B (coder RB1 ·
addressing AD2, already free · ordering TO2 · self-context CX3). **The seg term alone is worth 71.4% of
the gap and nobody has asked, on this body, whether it is even reducible.**

**The open question, and why it is open.** TD1 (`ddm_td1_token_drop_schur_arithmetic_20260816.md:15`)
recorded *"~95% of our seg term is render→SegNet round-trip loss, not label error"* — i.e. MANUFACTURED
by the realization path (render → R resize → uint8 → SegNet argmax), not caused by the transmitted
representation lacking fidelity. RT1 built the decomposition
(`ddm_rt1_seg_roundtrip_decomposition_20260816.md`). **But RT1's own frontier line reads
`hv1 ep0634 S 0.15959729295498598 @ 182,759 B` — the hv1 lineage, SIX pointer moves before DX2.**
Cross-vehicle constant transfer is a named poison class in this campaign. The 95% is NOT a property of
DX2; it is a hypothesis about DX2 that has never been tested.

**Why the answer changes the campaign either way.** If DX2's seg error is largely MANUFACTURED, then up
to ~30,248 B of archive budget is recoverable by fixing the realization path **without changing the
transmitted representation at all** — a lever orthogonal to every rate axis closed today. If it is
largely REPRESENTATIONAL, the seg term is a property of what we transmit, the realization-fix family is
empty on this body, and the campaign must stop treating TD1's number as inherited.

## SCOPE

1. **Verify pins, refuse on drift.** DX2 archive sha
   `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` @ 180,368 B, d_seg
   **0.00020139**, d_pose 6.37e-6, S 0.14821987563243377. RT1's memo + its decomposition machinery.
   TD1's memo. Reproduce DX2's d_seg from the shipped bytes before decomposing it.
2. **REUSE RT1's decomposition, re-run it on DX2 — do not rebuild it.** RT1 already built this
   instrument; rebuilding is waste and risks a different object. Port it to the DX2 body, state exactly
   what you changed to make it run here, and verify it reproduces RT1's hv1 number on hv1's artifacts
   if those are still retained (a positive control that the instrument still measures what it did).
   If the control cannot run, say so and label the measurement accordingly.
3. **Decompose d_seg 0.00020139 by WHERE THE ERROR ENTERS**, n600, with denominators (m50). At minimum
   separate: (a) representation — the transmitted token field cannot express the correct partition even
   ideally realized; (b) render — the learned renderer's token→pixel map loses it; (c) R — the exact
   composite resize to (512,384); (d) uint8 — the quantization step; (e) argmax — ties/margin collapse
   at the frozen head. **Per-class rows, Lane on its own** (0.59% of area, GT IoU 0.263, ~19% of all
   flips — a headline that averages Lane away is the wrong object). Report the MANUFACTURED FRACTION
   with its exact definition stated.
4. **State the recoverable byte-equivalent, and be honest about reachability.** For each manufactured
   stage, how much d_seg would a PERFECT fix at that stage remove, and what is that worth in archive
   bytes at 6.658e-7 S/B? These are CEILINGS, not achievable wins — label them so on their face. A
   ceiling with no known mechanism is still the right number to report, but it is not a plan.
5. **Name the cheapest real mechanism per manufactured stage, or say there is none.** Upstream is
   READ-ONLY and R/uint8/argmax are frozen — so any fix acts on what we RENDER, not on the evaluator.
   For each stage with material manufactured loss, name a concrete lever that acts pre-R (per #149's
   camera-resolution placement lens is one prior; there may be others) with its byte cost, or state
   plainly that the stage is manufactured-but-unreachable. **"Manufactured" does not imply "curable" —
   conflating them is the fake this charter refuses.**

## HARD CONSTRAINTS

- `upstream/` READ-ONLY — the evaluator path is frozen and is the thing we measure against, never a
  thing we edit. Any "fix" that touches R, uint8, or the scorer is inadmissible by construction.
- NO Modal fire. NO Metal fires (MAIN-fire-only). Local advisory launches ONLY via the canonical firer
  (`tools/fire_local_advisory.py`) — hand-assembled dispatch is the error factory.
- GT decode lineage: use the DALI-GT table where the tool family expects it; a PyAV-lineage GT on the
  pose axis is a measured wrong-objective defect. Say which GT lineage each number used.
- Shipped receiver bytes are CUSTODY — never edit in place. This arm MEASURES.
- The jo1 r9 run directory is SACRED (terminal by SELF-REFUSAL, `EXACT_DELTA_NONNEGATIVE`).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD (P0): every per-stage field, every per-class mask, every intermediate render
  persists with sha256 + bytes. Scalar-only artifacts while the fields exist in memory are forbidden
  AT THE TYPING MOMENT — a sister arm was rebuilt for exactly this (`#898`).
- **Receipts to `/Volumes/VertigoDataTier/pact/ddm_ms9_dx2_seg_manufactured_fraction/` — NOT
  APDataStore (~11 GiB free).** Say which tier you used.
- File ownership: RT1 owns the decomposition instrument · TD1 owns the 95% claim on hv1 · TO2/CX3/AD2/RB1
  own the rate-axis rows. CITE them; do not touch their trees.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_rt1_seg_roundtrip_decomposition_20260816.md` — the instrument EXISTS and its own frontier line is
  `hv1 ep0634 S 0.15959729295498598 @ 182,759 B`. Its numbers are hv1's, NOT DX2's. Reuse the machinery;
  do NOT reuse the fractions. Cross-vehicle constant transfer is a named poison class here.
- `ddm_ri1_rc1_full_rgb_receiver_20260822.md` + `ddm_ni1_nr1_k32_receiver_distortion_20260822.md` — both
  byte-feasible lossy re-representations measured DEAD on distortion (**43.66×** and **247.71×** over
  their d_seg ceilings), with a measured amplification exponent of **16.69** between them: *which*
  tokens differ dominates *how many* by more than an order of magnitude. Consequence for you: seg
  responds violently and non-linearly to small representation changes. Do not linearize across regimes.
- `ddm_lq1_lane_quotient_representability_20260822.md` — a full-Hamming ORACLE assignment removes only
  **16.6%** of RC1's mismatches ⇒ 83.4% representational IN THAT FAMILY; and the Lane-recall oracle
  recovered 417,267 Lane pixels at the price of **+2,755,323 total mismatches (+194%)**. **Seg
  mechanisms die on COLLATERAL, not on targeting** — measured three times on 08-22. Any per-class
  ceiling you report must state its collateral, or it is not a ceiling.
- `ddm_vf1_evaluator_visible_floor_20260822.md`=f65e641edfc987a127dd2813d4136bbb01ad1c46ef4b211c80176416afcb87b4 —
  **0 of 117,964,800 token positions carry qualifying DX2 evidence.** No retained token-level
  sensitivity corpus exists. Do not assume any region is inert; derive or measure.
- The 08-22 rate stack — RB1 0 B (7 streams) · AD2 addressing already free · TO2 orderings 196–687%
  worse · CX3 named-context 0 B. **Do not attack rate here.** This arm is the distortion side of the
  same exchange and its currency conversion is 6.658e-7 S/B.
- `#1202` (self-audit, same day): a sister charter of mine raced a WEAKER mechanism class against a
  tuned incumbent and I filed it as mechanism-reduced. Do not repeat it — see OPTIMAL FORM.

## OPTIMAL FORM

- **REFERENCE FORM (cited): the exact upstream evaluation path, unmodified** —
  `interpolate(x, size=(segnet_model_input_size[1], segnet_model_input_size[0]), mode='bilinear')` →
  uint8 → frozen SegNet argmax, over all 600 pairs, on DX2's ACTUAL rendered frames. Not a proxy, not a
  subset, not a re-implementation whose parity is unverified. Any deviation is declared per row as
  SCOPE (legal: e.g. a strided pilot BEFORE the full n600 run, stated as such) vs MECHANISM (requires
  an explicit TOY-BRACKET declaration that the row cannot produce a body-level verdict).
- **n600 or it is not evidence.** Prefix subsets are measured ANTI-CONSERVATIVE on some axes in this
  campaign (pose prefixes measure 2.54–4.21× harder than the population; seg prefixes 0.95–0.97×
  easier). A strided pilot is fine to shape the run; the verdict is n600.
- VERIFIED ARITHMETIC (check once, then use): DX2 S 0.14821987563243377 @ 180,368 B.
  rate 25·180368/37545489 = 0.1200996 · seg 0.020139 · pose 0.0079812 · distortion 0.028120.
  S<0.12 needs ≤ **137,986 B** → shed **42,382 B**; 6.658e-7 S/B. Zero-distortion ceiling 180,218.3 B ⇒
  zero-distortion gap **149.7 B**; distortion is worth **42,235 B**; **seg alone is worth 30,248 B**.
- **PRIOR-LAW PREDICTION (falsifiable):** DX2's manufactured fraction is materially LOWER than TD1's
  hv1 95% — because nineteen pointer moves of optimization preferentially harvest the cheap manufactured
  loss first — but still the MAJORITY of the seg term, landing in **60–90%**. Concretely: a perfect fix
  at the render/R/uint8 stages would remove ≥60% of d_seg 0.00020139, a ceiling worth ≥18,149 B of
  archive budget. Lane's manufactured fraction is HIGHER than the body average (thin structure is what a
  bilinear downsample destroys).
  **FALSIFIER:** measured manufactured fraction **< 25%** ⇒ DX2's seg term is REPRESENTATION-limited,
  TD1's 95% does not transfer across the six intervening pointer moves, the realization-fix family is
  empty on this body, and the seg axis requires a representation change rather than a realization fix.
  Report either outcome plainly with the number in the FIRST line — both are complete, campaign-directing
  results, and the falsifier would retire an inherited number the campaign has been carrying unexamined.

## DELIVERABLE

`.omx/research/ddm_ms9_dx2_seg_manufactured_fraction_20260822.md` — DX2's d_seg reproduced from the
shipped bytes + the RT1-instrument port with its positive control (or the honest statement that the
control could not run) + the per-stage decomposition (representation / render / R / uint8 / argmax)
with per-class rows and **Lane on its own** + the manufactured fraction with its exact definition + the
per-stage recoverable byte-equivalent labelled CEILING-not-plan + the cheapest real pre-R mechanism per
material stage or the honest "manufactured but unreachable" + the verdict on the prior-law prediction
with verdict_scope at the NARROWEST level the evidence supports. Every figure carries its GT lineage
and its denominator. Commit via the serializer. End with the own-vehicle frontier line.
