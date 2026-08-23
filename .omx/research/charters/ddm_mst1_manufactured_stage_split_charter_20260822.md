# ddm_mst1_manufactured_stage_split — 90.47% of DX2's seg error is manufactured downstream and worth 27,363 B; WHICH stage manufactures it decides whether any mechanism exists

## MANDATE

MS9 (`ddm_ms9_dx2_seg_manufactured_fraction_20260822.md`) measured the representation-vs-downstream
split on the LIVE DX2 body, exactly, at n600:

| charged object | errors / 117,964,800 px | fraction | seg S | byte-equivalent ceiling |
|---|---:|---:|---:|---:|
| transmitted representation error | 9,182 | — | — | — |
| …still wrong at final argmax | **2,264** | **9.5298%** | 0.00191922 | 2,882.3 B |
| **manufactured after a CORRECT transmitted label** | **21,493** | **90.4702%** | 0.01821984 | **27,362.9 B** |
| final DX2 seg error | **23,757** | 100% | 0.02013906 | 30,245.2 B |

The numerator reproduces the official eight-decimal `d_seg=0.00020139` exactly
(23,757/117,964,800 = 0.00020139058430989585). **27,362.9 B is 64.6% of the campaign's entire
42,382 B demand.**

**MS9's verdict is honestly PARTIAL: it did not resolve WHICH stage manufactures the 21,493**, because
its charter withheld the n600 scorer lane. `verdict_scope=INSTANCE:DX2_T4_n600`,
`MEASURED-EXACT-FIELD-REPLAY / PARTIAL-STAGE-SPLIT`. **This charter grants that lane.** The stage split
is the measurement that decides whether a mechanism can exist at all: `upstream/` is FROZEN, so R,
uint8, the SegNet forward and the argmax cannot be altered — only what we RENDER can change. A stage
that manufactures error inside the frozen path is manufactured-but-UNREACHABLE; a stage that
manufactures it in the render is manufactured-and-ADDRESSABLE. Those are opposite campaign outcomes and
the current evidence cannot tell them apart.

**The constraint a naive framing would destroy — carry it in every row.** The path FIXES **6,918**
representation errors (9,182 transmitted-wrong − 2,264 still-wrong) while BREAKING 21,493 correct ones.
It is doing real work in both directions. MS9 states the ceiling as *"removing every manufactured flip
while **preserving pose, rate, and all beneficial downstream corrections**."* Any mechanism that makes
realization "more faithful" risks surrendering those 6,918. **Report beneficial fixes per stage
alongside manufactured errors per stage; a stage's net is not its gross.**

## SCOPE

1. **Verify pins, reuse MS9's exact fields, refuse on drift.** DX2 archive sha
   `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` @ 180,368 B · MS9's memo and its
   retained per-pixel fields (`G` contest-CUDA DALI GT argmax, `L` transmitted decoded semantic label,
   `A` terminal contest-CUDA argmax) · TO2's decoded token field
   `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`. **REUSE MS9's fields — do not
   re-derive them.** Reproduce 23,757 / 21,493 / 2,264 / 6,918 before splitting anything; if any
   disagrees, that IS the finding, report it first and stop.
2. **Instrument the intermediate states between `L` and `A`.** The frozen path is
   render(L) → R (`interpolate` to (512,384) bilinear) → uint8 → SegNet forward → argmax. Capture the
   argmax-equivalent decision at each reachable intermediate so each of the 21,493 manufactured pixels
   is CHARGED to the earliest stage at which it becomes wrong. Say exactly how you define "wrong" at a
   pre-argmax stage (e.g. via the frozen head applied to the intermediate) and why that definition is
   faithful rather than convenient. **Stages you cannot separate, name as unseparated** — MS9 modelled
   honesty here and a forced split is worse than an admitted merge.
3. **Charge BOTH directions per stage.** Per stage: manufactured-here (was right, becomes wrong) AND
   repaired-here (was wrong, becomes right). The 6,918 beneficial fixes must be attributed too. A stage
   with high gross manufacture but high repair is a different object from one that only destroys.
4. **Per-class, Lane on its own row.** Canonical comma10k order 0=Road (23.2% area) · 1=Lane (0.59%
   area, GT IoU 0.263, ~19% of all d_seg flips) · 2=Undrivable (49.5%) · 3=Movable (1.24%) · 4=MyCar
   (25.4%). Report manufactured-per-stage per class and as a rate per class area. Lane is thin
   structure and a bilinear downsample is exactly what destroys thin structure — if that shows, it
   shows per-class or not at all.
5. **Adjudicate REACHABILITY per stage, and refuse to conflate it with magnitude.** For each stage
   carrying material manufacture: is it reachable by changing only what we RENDER (pre-R), given
   `upstream/` is frozen? Name a concrete candidate lever with its byte cost where one exists (the
   camera-resolution sub-pixel placement lens #149 is one prior; there may be others), or state plainly
   **manufactured-but-unreachable**. **"Manufactured" does not imply "curable" — conflating them is the
   fake this charter refuses**, and it is the same clause MS9 honored.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY — the frozen path is the thing we MEASURE AGAINST, never a thing we edit. Any
  proposed fix that touches R, uint8, the SegNet forward, or the argmax is inadmissible by construction.
- **SCORER LANE GRANTED for this arm** (MS9 was denied it; that denial is what left the split partial).
  Local advisory launches ONLY via the canonical firer `tools/fire_local_advisory.py` — hand-assembled
  dispatch is the error factory. NO Modal fire. NO Metal fires (MAIN-fire-only).
- GT lineage: contest-CUDA DALI GT, matching MS9's `G`. A PyAV-lineage GT is a measured wrong-objective
  defect on the pose axis and a lineage mismatch here would silently redefine every count. State the
  lineage on every number.
- Shipped receiver bytes are CUSTODY — instrument by reading; never edit in place.
- The jo1 r9 run directory is SACRED (terminal by SELF-REFUSAL, `EXACT_DELTA_NONNEGATIVE`).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD (P0): **every per-stage per-pixel field is a primary artifact** — persist with
  sha256 + bytes, not just the counts. A sister arm was rebuilt from scratch for keeping only scalars
  (`#898`); MS9's fields exist precisely because it complied.
- **Receipts to `/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_mst1_manufactured_stage_split/` — **BOTH SSD TIERS ARE AT 100% (measured 08-22; this killed the prior generation of this arm at rc=1 with zero artifacts). Local disk has ~500 GiB free and is the EXPLICIT-OPT-IN destination per the disk rule while the tiers are full.** Do NOT write to /Volumes/* — a write there will kill you.
  (~11 GiB free).** Say which tier you used.
- File ownership: MS9 owns the representation-vs-downstream split and its fields (CITE, reuse, do not
  re-derive) · BL1 is concurrently instrumenting per-position CODE LENGTH and XS1 cross-section
  conditioning — do not duplicate either; if BL1's field lands, the cost×manufactured join is a stretch
  goal, not a dependency.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_ms9_dx2_seg_manufactured_fraction_20260822.md` — the parent. Its byte figures are
  **oracle-equivalent CEILINGS at 1.2731082153 B per eliminated flip**, explicitly *"not mechanisms and
  not predicted archive savings,"* and it states **"No measured mechanism currently achieves that."**
  Inherit that discipline exactly. Also inherit its correction: use the exact integer numerator
  23,757/117,964,800, never the rounded eight-decimal display value (that rounding cost 2.8 B in MAIN's
  own shorthand).
- `ddm_rt1_seg_roundtrip_decomposition_20260816.md` — the instrument's origin, but its numbers are
  **hv1's** (`hv1 ep0634 S 0.15959729295498598 @ 182,759 B`, SIX pointer moves before DX2). MS9 already
  re-measured the top-level split on DX2; do NOT reuse RT1's per-stage fractions either — the same
  cross-vehicle transfer objection applies one level down.
- `ddm_lq1_lane_quotient_representability_20260822.md` — the Lane-recall oracle recovered 417,267 Lane
  pixels at the price of **+2,755,323 total mismatches (+194%)**. **Seg mechanisms die on COLLATERAL,
  not on targeting** — measured three times on 08-22. If Lane concentrates the manufacture, that is a
  LOCATION, not a licence; every reachability claim owes a collateral consideration.
- `ddm_ri1_rc1_full_rgb_receiver_20260822.md` + `ddm_ni1_nr1_k32_receiver_distortion_20260822.md` —
  representation changes measured **43.66×** and **247.71×** over their d_seg ceilings, with a measured
**[MAIN ERRATUM 2026-08-22: the `247.71×` NI1/NR1-K32 figure in this section is WITHDRAWN — fabricated, no receipt; NI1's d_seg is NOT MEASURED and its token-agreement proxy is 1.079× DX2, and at 122,250 B it is byte-feasible for sub-0.12. The RI1 `43.66×` is real and MEASURED. See `.omx/research/ddm_ni1_247x_erratum_20260822.md`.]**
  amplification exponent of **16.69**: seg responds violently and non-linearly to small representation
  changes. Do not linearize across regimes; a per-stage count is not a per-stage response curve.
- The 08-22 rate stack (RB1 0 B · AD2 already-free · TO2 196–687% worse · CX3 0 B · EF1 generic
  estimators 3.21× worse). **Do not attack rate here.** This is the distortion side; convert with the
  two-readings law at 6.658e-7 S/B (memory
  `the-demand-has-two-readings-distortion-is-worth-42235-bytes`).

## OPTIMAL FORM

- **REFERENCE FORM (cited): the exact upstream evaluation path, unmodified** —
  `interpolate(x, size=(segnet_model_input_size[1], segnet_model_input_size[0]), mode='bilinear')` →
  uint8 → frozen SegNet → argmax, over all 600 pairs, on DX2's actual rendered frames, contest-CUDA DALI
  GT. Not a proxy, not a re-implementation with unverified parity. Deviations declared per row as SCOPE
  (legal — e.g. a strided pilot BEFORE the n600 verdict, stated as such) vs MECHANISM (requires an
  explicit TOY-BRACKET declaration that the row cannot produce a body-level verdict).
- Family exemplar for CONDUCT: `ddm_ms9_dx2_seg_manufactured_fraction_20260822.md` — it reproduced
  d_seg from its own retained numerator, refused to reuse RT1's cross-vehicle ratio, labelled its byte
  figures as ceilings-not-mechanisms, declared its own split PARTIAL rather than forcing sub-stages it
  could not separate, and corrected MAIN's rounded arithmetic. Match that bar, including the
  willingness to declare a merge you cannot honestly split.
- **n600 or it is not evidence.** Seg prefixes measure 0.95–0.97× easier than the population in this
  campaign; a strided pilot may shape the run, the verdict is full.
- VERIFIED ARITHMETIC (re-derive once, then use): DX2 S 0.14821987563243377 @ 180,368 B.
  rate 0.1200996 · seg 0.02013906 · pose 0.0079812 · distortion 0.028120. S<0.12 needs ≤137,986 B →
  shed **42,382 B**; 6.658e-7 S/B; **1.2731082153 B per eliminated flip**. Manufactured 21,493 px =
  0.01821984 S = **27,362.9 B = 64.6% of the demand**. Zero-distortion ceiling 180,218.3 B ⇒ the
  zero-distortion gap is 149.7 B and distortion is worth 42,235 B.
- **PRIOR-LAW PREDICTION (falsifiable):** the **R stage (bilinear downsample to 512×384) dominates**,
  carrying **≥50%** of the 21,493 manufactured pixels — it is the one irreversibly lossy step in the
  frozen path, and Lane's manufactured rate per class area exceeds the body average because thin
  structure is exactly what a downsample averages away. R is FROZEN, so its share is addressable ONLY
  pre-R (sub-pixel placement at camera resolution before the average, #149's lens), which means the
  reachable fraction is smaller than the manufactured fraction.
  **FALSIFIER:** no single stage carries >30% of the manufacture ⇒ the loss is DISTRIBUTED across the
  frozen path with no single point of attack, no pre-R lever can address a majority of it, and the
  27,362.9 B ceiling is largely unreachable — which would mean the seg axis needs a representation
  change after all, not a realization fix. Report either outcome with the per-stage table in the FIRST
  line; both are complete and campaign-directing.

## DELIVERABLE

`.omx/research/ddm_mst1_manufactured_stage_split_20260822.md` — MS9's four counts reproduced
(23,757 / 21,493 / 2,264 / 6,918) + the per-stage charge table with each of the 21,493 assigned to its
EARLIEST wrong stage, unseparated stages named as unseparated + **repaired-per-stage alongside
manufactured-per-stage** (the 6,918 attributed) + per-class rows with **Lane on its own** in both count
and per-area rate + the per-stage REACHABILITY adjudication (pre-R lever with byte cost, or
manufactured-but-unreachable) + byte-equivalent ceilings labelled CEILING-not-plan at
1.2731082153 B/flip + GT lineage on every number + the verdict on the prior-law prediction with
verdict_scope at the NARROWEST level the evidence supports. Commit via the serializer. End with the
own-vehicle frontier line.
