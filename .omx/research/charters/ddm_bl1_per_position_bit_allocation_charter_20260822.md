# ddm_bl1_per_position_bit_allocation — five arms measured the token stream in aggregate; nobody has asked WHERE its 910,216 bits actually go

## MANDATE

Five arms measured the DX2 token stream on 2026-08-22. **Every one produced an aggregate.**

| arm | axis | result | granularity |
|---|---|---|---|
| RB1 | coder search | 0 B headroom, 7 streams | whole-stream |
| AD2 | addressing vs payload | 0 B — implicit raster, already free | whole-stream |
| TO2 | serialization order | 9 forms, 196.07%–686.94% worse | whole-stream |
| CX3 | named context summaries | 0 B; best IDEAL 117,224 B already worse than shipped | whole-stream |
| EF1 | generic estimators | best ZPAQ-m5 **365,322 B = 3.21× WORSE**; PPMd o32 402,241 B | whole-stream |

The stream codes **117,964,800 positions in 113,777 B = 910,216 bits = 0.007716 bits/position**. At
under one hundredth of a bit per position, *most positions must cost nearly nothing* and the bits must
be concentrated somewhere. **Nobody has looked.** Every verdict above is compatible with the mass being
concentrated in 1% of positions or spread evenly across all of them, and those two worlds demand
completely different next moves: a concentrated tail is a NAMED TARGET for a mechanism; a diffuse
spread means only a globally better model can help.

**Why this is a measurement and not an estimate — the point that makes it worth an arm.** The
incumbent is an RC64 arithmetic stream under a learned 19-member HPAC context law. An arithmetic coder
**already computes** the per-symbol cost: the code length at each position is exactly
`−log2 p(symbol | context)` under the model the receiver actually uses. This is not a proxy, not a
surrogate, not a re-implementation — it is the incumbent's own arithmetic, read out. That exactness is
what separates this from CX3 (which raced alternative conditioning sets) and EF1 (which raced
alternative estimators). Neither instrumented the shipped model.

**The join that could matter most.** MS9 is concurrently decomposing d_seg 0.00020139 into manufactured
vs representational per stage and per class. If the EXPENSIVE-TO-CODE positions coincide spatially with
the SEG-ERROR positions, then one mechanism moves both axes at once — and per the two-readings law
(memory `the-demand-has-two-readings-distortion-is-worth-42235-bytes`), rate and distortion are
interchangeable at **6.658e-7 S/B**, so a joint target is worth its bytes AND its distortion. If they
are disjoint, that is equally decisive: the axes are independent and must be attacked separately.

## SCOPE

1. **Verify pins, reuse the decoded field, refuse on drift.** DX2 archive sha
   `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` @ 180,368 B · TO2's decoded token
   field sha `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` (117,964,800 B) · TO2's
   checkpoint receipt `c0c05971396ff066c16cc0a82a46c5fe3e99a9c0000b4a93933e4bb2a57359f9` · RC64 stream
   sha `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5`. **REUSE TO2's decoded
   array** — a fifth decode is waste. Reproduce 113,777 B and 0.007716 bits/position first.
2. **Instrument the SHIPPED decode to emit exact per-position code length.** Read
   `−log2 p(symbol | context)` from the incumbent's own model at each of the 117,964,800 positions.
   **VALIDATION THAT MAKES IT REAL: the per-position costs must SUM to the shipped stream size** (to
   within the arithmetic coder's documented terminal overhead, which you state explicitly). If the sum
   does not reconcile to 113,777 B, the instrument is wrong and the measurement is void — say so and
   stop rather than reporting an unreconciled field.
3. **Report the DISTRIBUTION, not a summary statistic.** Full cost histogram + the concentration curve
   (what fraction of total bits do the top 0.1% / 1% / 5% / 10% / 50% of positions carry?) + Gini or an
   equivalent stated concentration measure. Denominators on every row (m50).
4. **Join the cost field to the structures that could make it actionable.** At minimum: **per class**
   (canonical comma10k order 0=Road 23.2% area · 1=Lane 0.59% area, GT IoU 0.263, ~19% of all d_seg
   flips · 2=Undrivable 49.5% · 3=Movable 1.24% · 4=MyCar 25.4% — **Lane on its own row**, and report
   bits-per-position per class, not just total bits per class) · **per frame/time** (EF1 measured
   ZPAQ's prefix-average rate bottoming at 400 frames and RISING at 600, with last-marginal-rate also
   rising — does the SHIPPED model show the same late-clip degradation, or is that generic-only?) ·
   **spatially** (the 190 groups `g=(x mod 64)+2*(y mod 64)` and raster position) · **vs the seg-error
   locations** if MS9's per-stage field has landed and is joinable — if it has not, name the join as
   owed rather than substituting a stale field.
5. **Adjudicate: concentrated target or diffuse mass.** State the concentration plainly and what it
   implies. If a small position set carries most of the bits, name it precisely (how many positions,
   which classes/frames/groups, how many bytes) — that set becomes the campaign's next target and its
   size is the ceiling on any mechanism aimed at it. If the mass is diffuse, say so: no targeted
   mechanism can work and only a globally better model helps. **Do NOT propose a mechanism in this
   arm** — locating the bits is the deliverable; spending them is a successor's job with its own gate.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. NO Metal fires (MAIN-fire-only). NO scorer runs — this is pure
  instrumentation of an existing decode; it changes no bytes and cannot move d_seg or d_pose.
- **Shipped receiver bytes are CUSTODY — instrument by reading, never by editing in place.** If you
  must add a readout path, add it additively and prove the unmodified decode still reproduces the exact
  token array byte-for-byte.
- The jo1 r9 run directory is SACRED (terminal by SELF-REFUSAL, `EXACT_DELTA_NONNEGATIVE`).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD (P0): **the per-position cost field itself is the primary artifact** — persist
  it with sha256 + bytes, not just the histogram. A sister arm was rebuilt from scratch for discarding
  exactly this kind of field while keeping only its scalars (`#898`). Scalar-only output here is a
  DEF-CON-1000 violation at the typing moment.
- **Receipts to `/Volumes/VertigoDataTier/pact/ddm_bl1_per_position_bit_allocation/` — NOT APDataStore
  (~11 GiB free).** The cost field is ~118M values; state its dtype, size, and which tier you used.
- File ownership: TO2 owns the decoded-field checkpoint · CX3 the named-summary ladder · EF1 the
  estimator race · AD2 the anatomy · RB1 the coder race · MS9 the seg decomposition (CONCURRENT — cite,
  do not touch). XS1 is concurrently testing cross-section conditioning; do not duplicate it.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_ef1_token_entropy_floor_20260822.md` — best generic estimator **365,322 B, 3.21× WORSE** than
  shipped; PPMd bottoms 402,241 B at order 32 then worsens. `verdict_scope=FAMILY` for both estimator
  families, **UNKNOWN** for differently-trained HPAC networks. EF1 refused to call any achieved size a
  floor: *"Achieved code sizes are upper bounds... EF1 establishes no nontrivial lower bound."* Inherit
  that discipline exactly — this arm reports an ALLOCATION, never a floor.
- `ddm_cx3_context_axis_ceiling_20260822.md` — named conditional-entropy ladder **0 B**; best
  model-inclusive challenger 125,210 B, hindsight ideal 117,224 B already 3,447 B worse than shipped
  before model cost. Do not re-race conditioning sets; this arm instruments the EXISTING one.
- `ddm_to2_token_ordering_race_20260822.md` — nine generic orderings × three coders, 196–687% worse.
  Reordering is a SUBSTITUTE for a context model, not a complement. Do not reorder.
- `ddm_rb1_rate_bound_decomposition_20260822.md`=fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09 —
  **0 B**, seven streams, per-stream isolation (its line 160 states that isolation as the operating
  assumption; XS1 is concurrently testing it — do not duplicate).
- `ddm_lq1_lane_quotient_representability_20260822.md` — the Lane-recall oracle recovered 417,267 Lane
  pixels at the price of **+2,755,323 total mismatches (+194%)**. **Seg mechanisms die on COLLATERAL,
  not on targeting** — measured three times on 08-22. If you find Lane costs disproportionate bits, that
  is a LOCATION, not a licence; any successor spending there owes a collateral column.
- `ddm_vf1_evaluator_visible_floor_20260822.md`=f65e641edfc987a127dd2813d4136bbb01ad1c46ef4b211c80176416afcb87b4 —
  **0 of 117,964,800 token positions carry qualifying DX2 evidence.** There is no retained token-level
  sensitivity corpus. **This arm builds the first token-level field of any kind** — do not assume any
  position class is inert, and do not conflate "cheap to code" with "unimportant to the score."
- `#1202` (self-audit, same day): my EF1 charter raced a WEAKER mechanism class against a tuned
  incumbent. This arm avoids that by instrumenting the incumbent ITSELF rather than racing anything.

## OPTIMAL FORM

- **REFERENCE FORM (cited): the shipped decoder, instrumented — not re-implemented.** The RC64
  arithmetic stream under its learned 19-member HPAC context law over 190 groups, traversing
  frame-outermost → group → raster (TO2's anatomy; AD2 receipt `80124acd…b73511`). Per-position cost is
  `−log2 p(symbol | context)` from that model. **Substituting a re-implemented or approximate model is
  a MECHANISM reduction and is FORBIDDEN**; the sum-reconciliation check in scope item 2 is what proves
  you used the real one.
- Family exemplar for CONDUCT: `ddm_ef1_token_entropy_floor_20260822.md` — it reproduced its pins,
  reused the decoded array, labelled upper bounds as upper bounds, scoped FAMILY/FAMILY/UNKNOWN per
  object, and named the fake it was refusing. Match that.
- **n600, all 117,964,800 positions.** A strided pilot is fine to shape the run; the verdict is full.
  Prefix subsets measure differently on this campaign's axes (pose prefixes 2.54–4.21× harder, seg
  prefixes 0.95–0.97× easier) and the time-join in scope item 4 is destroyed by truncation.
- VERIFIED ARITHMETIC: DX2 S 0.14821987563243377 @ 180,368 B. rate 0.1200996 · seg 0.020139 ·
  pose 0.0079812 · distortion 0.028120. S<0.12 needs ≤137,986 B → shed 42,382 B; 6.658e-7 S/B.
  Token member 113,777 B = 910,216 bits over 117,964,800 positions = **0.007716 bits/position**;
  the target 71,395 B = 0.004842 bits/position. Zero-distortion ceiling 180,218.3 B ⇒ distortion is
  worth 42,235 B, **seg alone 30,248 B**.
- **PRIOR-LAW PREDICTION (falsifiable):** the cost field is STRONGLY heavy-tailed — the top **1%** of
  positions carry **>50%** of the 910,216 bits — because a mean of 0.0077 bits/position requires the
  overwhelming majority to be near-perfectly predicted. Lane costs materially more bits-per-position
  than its area share (thin, high-diversity, GT IoU 0.263), and the shipped model shows the same
  late-clip rate rise EF1 measured generically.
  **FALSIFIER:** top 1% of positions carry **<25%** of total bits ⇒ the mass is DIFFUSE, no concentrated
  target exists, every targeted rate mechanism on this stream is structurally dominated, and only a
  globally better predictor can move it. Report either outcome with the concentration curve in the
  FIRST line — both are complete and campaign-directing, and the falsifier would close the
  targeted-mechanism family on measured evidence rather than on argument.

## DELIVERABLE

`.omx/research/ddm_bl1_per_position_bit_allocation_20260822.md` — the reproduced 113,777 B and
0.007716 bits/position + the instrumented readout with its **sum-reconciliation to the shipped stream
size** + the full cost histogram and concentration curve (0.1/1/5/10/50%) + the per-class join with
**Lane on its own row in bits-per-position** + the per-frame/time curve read against EF1's late-clip
rise + the spatial/group join + the seg-error join or the honest statement that it is owed + the
concentrated-vs-diffuse adjudication naming the target set precisely (positions, classes, frames,
bytes) if one exists + the verdict on the prior-law prediction with verdict_scope at the NARROWEST
level the evidence supports. The per-position cost FIELD persists as a retained artifact with sha256.
No mechanism proposals. Commit via the serializer. End with the own-vehicle frontier line.
