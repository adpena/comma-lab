# DDM RFO1 — fresh eyes, hybrid composition, and recursive fractal optimization

Date: 2026-08-14  
Arm: `ddm_rfo1_fresh_hybrid_compose`  
Disposition: research/compose receipt; no scorer, trainer, Modal, Metal, or live EC2 state touched  
Authority labels used below: **MEASURED**, **DERIVED**, and **CONJECTURE**

## Executive conclusion

The effective floor did not move. It remains the retained cp135 archive at
`S = 0.16195513827824176`, 186,252 bytes, `[contest-CUDA T4, n600]`, archive
SHA-256 `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`.

The shortest credible routes, in order, are:

1. **Harvest EC2 without interfering with it.** Its exact admission law is now
   corrected: +1,707 bytes needs at least 1,341 net Seg flips with zero pose
   damage, and 1,353 net flips for a nameable `1e-5` improvement. The repeated
   1,340 figure is the rounded-law result, not the integer exact gate.
2. **Build `RFO1-MICRO35`, the smallest bank composition that can be
   nameable.** It combines qs2, re1, HP4's receiver-identical five-byte repack,
   one additional sign-verified net flip, and in-compile compensation. Its
   hard gate is `F_union >= 35`, `Delta B <= +29`, and
   `Delta d_pose <= 5.9739759814e-10`. Nothing is banked until the actual union
   is built, decoded, recounted, and scored.
3. **If EC2 misses, train the missing interaction, not another copy of the
   existing hybrid.** The cp135/F26 receiver already combines semantic tokens
   with a learned 600x8 frame latent. The absent object is a small bilinear
   interaction between EC1's oriented context hidden state and that existing
   frame/HPAC state, followed by joint probability-state retraining. Calling a
   mere token-plus-frame-latent concatenation “semantic x latent” would be a
   fake novelty claim.
4. **Keep #978's multi-token row live, but define it as a jointly learned
   representative/probability-state residual.** HY1 proved the grammar can
   describe the changed sites cheaply; HC1 later proved its direct C1
   representative does not survive as the desired semantic preimage. The
   representation, not carriage, is the remaining question.
5. **Do not reopen generic lossless coding or adaptive absolute precision.**
   Same-state ANS loses RC64 by 6–9 bytes, CAP1 is already banked, and pz4a's
   best 500-byte coefficient saving is dominated by its 2,732-byte allocation
   map. Only representation-changing, rate-aware in-loop QAT may reopen the
   quantization family, and only after a scorer-free whole-container pre-proof
   of at least 2,000 bytes.

The primary fresh-eye finding is that the current vehicle is already a
semantic/latent hybrid but lacks an explicit low-rank *interaction* where the
edge conditioner is formed. This changes the successor from “add a latent” to
“make the oriented correction conditional on the latent already paid for.”

## Corrected cp135 waterfall and atomic prices

### Primary derivation

**MEASURED, primary source.** `upstream/evaluate.py` defines

```text
S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37,545,489.
```

**MEASURED, primary source.** `upstream/modules.py` makes the asymmetry clear:

- SegNet consumes only the second frame of each pair, resized to 512x384, and
  charges exact argmax disagreement.
- PoseNet consumes both frames after RGB-to-YUV6 conversion and compares the
  first six pose outputs.

Therefore frame 0 is Seg-free but Pose-active, while frame 1 is jointly
Seg/Pose active. The optimal local attack is not an RGB-fidelity edit. It is an
integer, receiver-closed search over frame-1 semantic decisions with frame-0
and same-pair compensation, priced through the final archive.

**MEASURED.** The retained cp135 archive was re-hashed and inspected at:

```text
/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/retained/candidates/
  hp3_step2/split_brotli_per_section_opt_cap1_metadata__rc64/archive.zip
```

It is 186,252 bytes and has the pinned SHA-256. It contains one stored ZIP
member `p`: 186,152 payload bytes plus exactly 100 bytes of ZIP framing. The
payload anatomy is 70,825 model bytes, 96 residual bytes, and 115,231 RC64
token/probability bytes. The decoded token plane has 117,964,800 symbols.

**DERIVED, exact Decimal arithmetic.** The score atoms are:

| Atom | Exact price |
|---|---:|
| one archive byte | `+6.658589531221713479e-7 S` |
| one net Seg flip | `-8.477105034722222222e-7 S` |
| rate/Seg exchange | `0.785479182332663186 flips/byte` |
| byte allowance per Seg flip | `1.273108... bytes/flip` |

Using cp135's report-precision components consistently gives:

| Component | cp135 contribution |
|---|---:|
| `100*d_seg`, `d_seg=0.00029643` | `0.029643000000000000` |
| `sqrt(10*d_pose)`, `d_pose=0.00000688` | `0.008294576541331088` |
| rate, 186,252 bytes | `0.124017561736910658` |
| reconstructed total | `0.161955138278241746` |
| gap to 0.15 | `0.011955138278241746` |

This reconstruction agrees with the pinned score to rounding. The charter's
`0.1243317` rate contribution is not cp135's 186,252-byte rate; it corresponds
to the older 186,724-byte PR135-sized object. Mixing it into the cp135
waterfall yields about 0.1622693. All gates below use the retained cp135 size.

The precise matched-worker base pose value
`6.885642960696714e-6` is used only for candidate deltas. It is not mixed into
the report-8-decimal absolute waterfall, because one displayed pose ULP is
material at this scale.

### Exact EC2 gate

For `F` net Seg flips, byte delta `B`, and pose delta `p`, the exact local
score delta is

```text
Delta S(F,B,p) =
  -F*(100/117,964,800)
  + sqrt(10*(p0+p)) - sqrt(10*p0)
  + B*(25/37,545,489),

p0 = 6.885642960696714e-6.
```

At `B=+1,707`:

| Pose assumption | Beats base | Improves by at least `1e-5` |
|---|---:|---:|
| zero pose damage | 1,341 flips | 1,353 flips |
| `+1.3e-7` **direct S-term** tax | 1,341 flips | 1,353 flips |
| `Delta d_pose=+1.3e-7` | 1,433 flips | 1,445 flips |

The phrase “1.3e-7 pose budget” is dimensionally ambiguous across the recalled
documents. The later HV1 review also shows that JS7's historical universal
budget was arithmetically mis-stated; the correct limit is candidate-specific
inversion of the square-root term. No row may use a bare 1.3e-7 without naming
whether it is `Delta d_pose` or direct score damage.

## One merged ranking across all four movements

Projected deltas are not rows. Each `Delta S` below is either exact conditional
arithmetic or explicitly unavailable until a retained object exists.

| Rank | Candidate / scale | Label and projected `Delta S` | Bytes | Named consumer | Disposition and falsifier |
|---:|---|---|---:|---|---|
| 1 | EC2 oriented implicit conditioner | **DERIVED conditional.** At +1,707 B: `Delta S(F,1707,p)` above; nameable at 1,353 flips only when pose tax is zero | +1,707 design price; actual archive controls | #984 composed campaign | **IN-FLIGHT, MAIN-owned.** Harvest only. Falsified as a base move if exact retained archive has nonnegative `Delta S`; no interaction with its run/store/claims from RFO1. |
| 2 | `RFO1-MICRO35` bank union | **DERIVED conditional.** Hard gate `Delta S <= -1e-5`; projected `-1.0247340278e-5` only if full re1 pose leakage is compensated | <= +29 | #984 composition store | **QUEUED.** Falsified if union has `<35` distinct net flips, `>+29 B`, `Delta d_pose>5.9739759814e-10`, or any receiver mismatch. |
| 3 | Oriented-context x existing frame-latent bilinear gate | **CONJECTURE.** `Delta S(F,B,p)`; no numeric benefit invented. A 4-channel gate has a roughly 36-weight logical core, but whole-container bytes are unknown | target <=128 incremental B beyond chosen EC checkpoint; measured archive controls | #982 trained receiver, then #984 | **QUEUED-CONTINGENT.** Scorer-free build first after EC2 harvest. Kill this implementation if real incremental bytes exceed 128 before training, or if stratified-random held-out edge ordering does not beat EC1 at matched capacity. |
| 4 | #978 multi-token semantic-support x learned representative x joint probability state | **CONJECTURE.** Exact gate `Delta S(F,B,p)<0`; no transfer from HY1's carriage-only number | unknown; must retain every candidate | #978 then #984 | **QUEUED-CONTINGENT.** Kill the tested implementation if a random n>=32 direct receiver probe cannot improve sign-verified Seg survival at matched pose before full training. |
| 5 | Level-set/witness representative through cp135 receiver | **CONJECTURE.** Exact gate only; HY1's grammar result does not price the missing preimage | unknown | #978 representation store | **HELD behind a $0 receiver probe.** Kill the tested representative if its uint8/resize/parse-back image cannot preserve the target semantic cells on stratified-random n>=32. |
| 6 | PR135 int12 quantize-then-compensate with qs5 Schur step | **CONJECTURE, formulation narrowed.** Only grouped/implicit or learned in-loop precision remains | must save >=2,000 whole-container B before scorer | #984 rate branch | **HELD.** pz4a closed absolute per-cell allocation. Fire only if byte-only parse-back saves >=2,000 B without a larger map. |
| 7 | Same-state RC64/ANS or more CAP1 work | **MEASURED negative, current-state instance.** ANS is +6 to +9 B; CAP1 already banked | loss | none | **FOLDED/DEAD for the same probability state.** Reopen only with a new symbol state, not coder tuning. |

## Movement 1 — fresh-eye derivation versus live routes

### What the objective demands

The exact objective demands four things in this order:

1. Choose frame-1 semantic changes by exact Seg value, not RGB error or a
   proxy margin alone.
2. Realize those changes through the actual cp135 decoder and resize/uint8
   lattice.
3. Use frame 0 and same-pair degrees of freedom to minimize the nonlinear
   Pose term without paying an explicit overlay stream.
4. Jointly re-price model, probability state, coder output, and ZIP container;
   section-local byte wins do not imply an archive win.

EC1/EC2 address steps 1–2 implicitly. qs5 proves a small version of step 3.
cp135/HP3/CAP1 address step 4. The live program is directionally correct, but
the interaction between EC1's oriented local context and the already-paid
frame latent is not explicit at the conditioner input, and the bank has not
been union-built.

### Missing rows the derivation demands

| Priority | Missing row | Why it is demanded | Cheapest valid falsifier |
|---:|---|---|---|
| 1 | Exact EC2 harvest with integer gate | Design metrics cannot establish final bytes, distinct net flips, or pose | Retained endpoint archive, decode equality, exact component recomputation |
| 2 | qs2/re1/HP4 union with fresh compensation | Separate exact wins can overlap or interact; only the whole object is additive authority | Byte-only union build and distinct-support recount, then one queued exact worker row |
| 3 | Frame-conditioned oriented head | Base frame latent is paid but EC1's local correction is formed without an explicit bilinear frame interaction | Matched scorer-free archive pricing plus random n>=32 receiver-survival screen |
| 4 | Multi-token jointly trained representative | HY1 proved grammar coverage; HC1 killed one direct representative, leaving joint representation as the crux | Random n>=32 receiver-closed representation screen, not prefix n8 |
| 5 | Whole-container QAT pre-proof | pz4a showed the allocation map can erase gross precision savings | Parser-equal archive build demonstrating >=2,000 B before scorer use |

Online primary literature supports the *mechanism class*, not any number here:
rate-distortion autoencoders jointly learn discrete latents and entropy models,
and learned video codecs jointly optimize learned state rather than treating
representation and carriage independently. Recent task-oriented joint
token/coding work makes the same structural point in another domain. These are
priors for the interaction rows, not authority for Pact admission:

- Habibian et al., *Video Compression with Rate-Distortion Autoencoders*,
  ICCV 2019: https://openaccess.thecvf.com/content_ICCV_2019/html/Habibian_Video_Compression_With_Rate-Distortion_Autoencoders_ICCV_2019_paper.html
- Rippel et al., *Learned Video Compression*, ICCV 2019:
  https://openaccess.thecvf.com/content_ICCV_2019/html/Rippel_Learned_Video_Compression_ICCV_2019_paper.html
- *Joint Token Compression and Modulation*, arXiv:2608.00368:
  https://arxiv.org/abs/2608.00368

## Movement 2 — hybrid inventory

### H1: semantic x latent token hybrid (#978)

**MEASURED implementation anatomy.** The F26/cp135 renderer already embeds the
five-class semantic token and applies a learned 600x8 frame embedding as FiLM
inside four TokenBlocks. Therefore the base is already semantic x latent.

**CONJECTURE, genuinely new mechanism.** The unraced row is a coupled
multi-token semantic support whose *rendered representative and probability
state* are learned together, optionally with a low-rank residual conditioned
on the existing frame latent. It must not store an explicit changed-site list.

**Exact price.** Unknown until materialized. Admission is
`Delta S(F,B,p)<0`, and nameability is `Delta S<=-1e-5`. No HY1 projected
score transfers because HC1 later refuted its direct representative.

**Consumer and order.** Owner `#978`; consumer `#984`. First run the $0
stratified-random n>=32 receiver-survival screen, then a retained short train,
then whole-container price, then queue exact n600 after EC2 releases the slot.

### H2: EC1 oriented conditioning x HPAC/frame context (#982)

**MEASURED implementation anatomy.** EC1 forms a 25-channel decoded-token
neighborhood, reduces it through a four-channel context head, and injects a
96-channel delta before TokenBlocks. The base frame embedding is applied later
inside the TokenBlocks. Downstream nonlinearity can create interactions, but
the EC1 correction itself is frame-invariant at formation time.

**CONJECTURE, new mechanism.** Project the existing 8-D frame latent to four
scale/gate values and modulate EC1's four-channel hidden state before its
96-channel head. A minimal logical core is 8x4 weights plus four biases. It
adds no new per-pair latent table; only small shared adapter weights are new.
Train the gate and probability state jointly. This is the literal
oriented-context x existing-HPAC-state interaction that is absent from EC1.

**Exact price.** Parameterized only: `Delta S(F,B,p)`. A provisional build cap
of +128 whole-container bytes implies zero-pose break-even at 101 net flips and
nameability at 113 net flips; these are gates, not predicted performance.

**Consumer and order.** Owner `#982`; consumer `#984`. Do not fork the live
EC2 store. Rebuild from the sealed base only after EC2 harvest. First price the
archive; then use random n>=32; then a retained resumable train; then exact
n600.

### H3: int12 quantize-then-compensate x qs5 Schur

**MEASURED boundary.** The live carrier is already logically signed int12;
“int16 to sub-int16” is not a new row. pz4a's absolute variable-precision
lattice saved at most 500 gross bytes, while its allocation-depth wire cost
2,732 bytes. Even zero metadata leaves only a 500-byte ceiling. qs5 proved
in-compile Schur compensation, but its tested three-pair instance gained only
17 Seg flips and lost `+2.519822e-6 S` after +26 bytes.

**CONJECTURE, remaining mechanism.** Group precision by a deterministic
existing structural partition, or learn rate-aware quantization in-loop, then
apply the Schur pose compensation to the quantized object before export. No
per-cell precision map is allowed.

**Exact price.** Fire only after a parser-equal, scorer-free archive is at
least 2,000 bytes smaller. At `B=-2,000`, the rate credit is
`-0.001331717906244343 S`; the exact pose/Seg outcome still controls admission.

**Consumer and order.** Owner `PZ4-QAT`; consumer `#984`. Byte pre-proof,
short retained QAT, exact worker, composition. This is held, not fired.

### H4: level-set/witness solver x cp135 receiver

**MEASURED boundary.** HY1's C1 token grammar represents all 27,351 changed
sites and its F26 coder cost was only +11 bytes, but later HC1 exact evidence
showed the direct C1 representative produced `S≈0.404` and did not preserve the
desired semantic preimage. The cheap carriage result survives; the direct
representative does not.

**CONJECTURE, remaining mechanism.** Treat the level-set solver as a teacher
that selects semantic support, then distill a receiver-native latent/token
representative jointly through the cp135 decoder and HP3 probability model.
Never ship the level set or scorer-derived dense field as free code.

**Exact price.** Unknown until the receiver-native representative exists.
The $0 falsifier is random n>=32 uint8/resize/parse-back survival. Prefix n8 is
not a population verdict.

**Consumer and order.** Owner `#978 representation`; consumer `#984`. It is
behind H1/H2 because HC1 already showed the direct preimage risk is severe.

## Movement 3 — composition arithmetic and candidate specification

### Current bank and container fold

| Piece | Exact/observed receipt | Status in a union |
|---|---|---|
| qs2 r2 | **MEASURED** `Delta S=-4.374914e-6`, +34 B, 32 fewer Seg flips, `Delta d_pose=+1.86901161214e-10` | banked alone; support must be recounted in union |
| re1 round 1 | **MEASURED** `Delta S=-1.2068738491654126e-6`, 0 B, 2 fewer Seg flips, `Delta d_pose=+8.108145266e-10` | banked alone; support must be recounted in union |
| qs2 + re1 | **DERIVED additive projection** `-5.5817878491654126e-6` | held; not a measured union |
| HP4 order-0 split/repack | **MEASURED byte-only** receiver-identical archive at -5 B | fold into next build; not worth a scorer row alone |
| bank + HP4 | **DERIVED additive projection** `-8.911082614776269e-6` | still below nameable threshold |
| qs5 compensation | **MEASURED mechanism**, tested instance refused | technique portable; 17-flip ceiling is instance-scoped |

### Composed candidate: `RFO1-MICRO35`

This is a candidate specification, not a result.

**Mechanism.** Build one archive from the cp135 base with:

1. qs2's exact coefficient choices;
2. re1's exact realization choices;
3. HP4's receiver-identical order-0 split/repack;
4. at least one additional non-overlapping, sign-verified net Seg flip found
   on the same exact integer lattice; and
5. an in-compile Schur solve that reduces the union's pose leakage without a
   new payload section.

**Hard whole-object gate.** The parsed archive must satisfy all of:

```text
archive_bytes <= 186,281       # Delta B <= +29
distinct net Seg flips >= 35   # recount, never 32+2+1 by assertion
Delta d_pose <= 5.973975981397870e-10
receiver decode equality = PASS
Delta S <= -1e-5
```

The existing qs2+re1 pose deltas sum to `9.97715687814e-10`. Therefore the
compensation must recover at least `4.003180896742130e-10` in `d_pose`, equal
to about `2.4120028669e-7 S` at this base. Full removal of re1's pose score tax
would project `Delta S=-1.024734027824849e-5`; without compensation, even the
extra flip projects only `-9.758793118248492e-6` and is not nameable.

**Payload and provenance contract.** Retain every attempted archive, decoded
payload, section bytes, and deterministic manifest on the SSD. Record each
object's bytes and SHA-256. No scalar-only materializer is permitted. Build is
byte-only first; the exact scorer step remains queued behind EC2.

**Falsifiers.** Refuse the candidate if supports overlap below 35 net flips,
the Schur solve cannot meet the pose cap, any section/container interaction
pushes the archive above 186,281 bytes, parse-back differs, or exact component
recomputation gives `Delta S>-1e-5`.

### EC2 composition rule

EC2 is not pre-composed by arithmetic. When MAIN harvests it:

1. verify retained checkpoint, archive, repeat archive, hashes, and stage
   manifests;
2. compute actual archive bytes, distinct net Seg flips, and exact pose;
3. apply `Delta S(F,B,p)` to the actual object;
4. only if it passes, rebuild a union with qs2/re1/HP4 and recount everything;
5. compose pz4 QAT only if its independent >=2,000-byte pre-proof exists.

The EC2 design-price shorthand `+1,707 B / >=1,340 flips` is superseded for
admission by the exact 1,341 base gate and 1,353 nameable gate before pose tax.

## Movement 4 — recursive fractal optimization

`AT-OPTIMUM` below is always scoped to the measured state/implementation, not
a universal family theorem.

| Scale | Element | Disposition | Receipt / cheapest movement | Projected score effect |
|---|---|---|---|---:|
| constant | evaluator weights and denominator | **AT-OPTIMUM / immutable** | primary `evaluate.py`; do not optimize the ruler | none |
| constant | per-flip/per-byte admission constants | **AT-OPTIMUM** | exact Decimal derivation above | decision law only |
| section | 70,825-B F26 model | **MOVABLE** | EC2/H2 alter representation; cheapest measurement is retained whole-model serialize plus parse-back | `Delta S(F,B,p)` |
| section | 96-B residual | **AT-OPTIMUM, current state** | fd135/cp135 anatomy; too small for a 1e-5 rate-only move | <=`6.39e-5` total removable ceiling, but no demonstrated removal |
| section | 115,231-B token/probability stream | **MOVABLE by probability state** | EC1/EC2/H1 joint context; byte-only re-encode before scorer | measured `Delta B`, then exact law |
| section | CAP1 metadata geometry | **AT-OPTIMUM, current layout** | CAP1 fixed pack is already included; LP135 banked the saving | no remaining same-layout row found |
| section | 600x8 frame embedding | **AT-OPTIMUM for post-hoc prediction; MOVABLE by interaction** | HP4 predictors all lose; H2 reuses the existing state rather than recoding it | HP4 fold `-5 B`; H2 unknown |
| coder | RC64 versus F26 ANS | **AT-OPTIMUM, same symbol state** | LP135: ANS +6 B control / +9 B HP3 | ANS loses `3.995e-6` to `5.993e-6 S` |
| coder | context-changing entropy model | **MOVABLE** | EC1/EC2/H1; not a generic coder swap | actual `Delta B` |
| receiver code | cp135 decode/parse path | **AT-OPTIMUM for current bits** | exact receiver equality and retained archive; free code cannot hide learned state | none without new representation |
| receiver code | oriented x frame-latent interaction | **UNKNOWN** | H2 36-weight logical-core build, whole-container price, random n>=32 screen | target break-even at <=128 B is 101 flips |
| archive container | one stored member / 100-B ZIP frame | **MOVABLE by five bytes only** | HP4 receiver-identical order-0 split/repack | `-3.3292947656e-6 S` |
| archive container | further CAP1/ZIP tuning | **AT-OPTIMUM, current object** | same-state searches consumed; reopen only with section changes | no live row |
| chain | qs2+re1+HP4 | **MOVABLE but below naming gate** | build union, recount interactions | additive projection `-8.9110826148e-6` |
| chain | `RFO1-MICRO35` | **UNKNOWN** | byte-only union, then exact queued row | conditional `<=-1e-5` |
| chain | EC2 then bank | **UNKNOWN / IN-FLIGHT dependency** | MAIN harvest, then rebuild whole union | never sum separate receipts as authority |

### Quantization application map

| Tool | Status | Application boundary |
|---|---|---|
| adaptive absolute per-cell precision | **AT-OPTIMUM negative / FORMULATION DEAD** | pz4a: 500 B gross ceiling, 2,732 B map; do not reopen without a map-free representation |
| aware in-loop QAT | **UNKNOWN / MOVABLE** | only a jointly trained representation with retained stage checkpoints and >=2,000-B byte pre-proof |
| “sub-int16” | **FOLDED as misdescription** | current logical carrier is already signed int12 |
| grouped/implicit sub-int12 | **UNKNOWN** | deterministic grouping from existing structure; no explicit per-cell map; parser-equal pre-proof first |
| quantize then qs5 Schur compensate | **UNKNOWN, mechanism plausible** | compensation must be solved on the quantized exported object and whole archive; qs5's 3-pair result does not transfer |

## Fire chain and ownership

No T4 or Modal action was fired by this arm. The single exact n600 slot remains
with the live EC2 campaign. Every follow-on named above has one disposition:

1. **IN-FLIGHT — EC2 harvest.** Owner: MAIN / EC2 live lane. Consumer:
   EC2 retained store then #984 composed campaign. Fire trigger: the existing
   Modal call becomes terminal and every stage checkpoint/archive/hash is
   present. RFO1 must not pollute or mutate the live run, store, or claim.
2. **QUEUED — `RFO1-MICRO35`.** Owner: MAIN-assigned #984 byte-closed builder.
   Consumer: a new SSD retained candidate store plus the #984 composition
   receipt. Fire trigger: EC2 releases the scorer slot; byte-only build may
   begin earlier only under a distinct claimed lane and must retain all
   candidates.
3. **QUEUED-CONTINGENT — H2 low-rank interaction.** Owner: #982 successor.
   Consumer: #982 retained trained-receiver store, then #984. Fire trigger:
   EC2 harvest misses the exact nameable gate or exposes frame-conditioned
   residual structure; build from the sealed base, never the live store.
4. **QUEUED-CONTINGENT — H1 joint multi-token representative.** Owner: #978.
   Consumer: #978 representation store, then #984. Fire trigger: a
   stratified-random n>=32 receiver-survival probe beats the HC1 direct
   representative at matched pose and the prior scorer lane is free.
5. **HELD — grouped/rate-aware PZ4-QAT.** Owner: PZ4-QAT successor. Consumer:
   #984 rate branch. Fire trigger: a retained parser-equal byte-only object
   saves at least 2,000 whole-container bytes without an allocation map.
6. **FOLDED — HP4 order-0 repack.** Owner: next composition builder. Consumer:
   every next whole-archive build. Fire trigger: automatic whenever a new
   candidate archive is packaged; do not spend a scorer row on it alone.

## RECALL EVIDENCE

### Stores and queries searched

The recall was content-first, not charter-name-only. Searches covered:

- primary scorer/runtime: `upstream/evaluate.py`, `upstream/modules.py`, the
  retained cp135 archive, F26 receiver/runtime anatomy;
- `.omx/research/` and arm-final receipts for `cp135`, `F26`, `HP3`, `CAP1`,
  `RC64`, `ANS`, `semantic latent`, `multi-token`, `conditioner`, `frame_embed`,
  `Schur`, `compensat`, `quantiz`, `precision`, `qs2`, `qs5`, `re1`, `HP4`,
  `HY1`, `HC1`, `PZ4A`, and `JS7`;
- canonical equation registry via `tools/list_canonical_equations.py --json`,
  filtered for gap, pose budget, receiver survival, flip/byte exchange,
  realization, and composition;
- `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, design/SPEC
  surfaces, task-status rows, active lane claims, `probe_outcomes.jsonl`, and
  `main_hot_state.md` for `#978`, `#982`, `#984`, EC2, and composition;
- all `ExperimentBook.md` instances through the fd135 full-corpus pass;
- primary online/OSS literature for joint latent/entropy learning,
  task-oriented token coding, and derivative-free local trust regions.

The bounded task-store join did not yield a unique clean row for every
overloaded numeric label `#978/#982/#984`; this memo therefore preserves the
charter's campaign labels and does not call any route ownerless.

### Findings beyond the charter seeds and what changed

1. **HC1 superseded HY1's direct representative.** HY1's grammar/carriage
   result remains real; the direct C1 preimage is dead. This changed H4 from a
   direct compose proposal to a teacher-to-receiver-native distillation probe.
2. **The base already is semantic x latent.** F26 uses semantic tokens plus a
   600x8 frame FiLM state. This changed H1/H2 from “add latent context” to an
   explicit low-rank interaction at the oriented conditioner and a jointly
   trained representative/probability state.
3. **QS5 has a final refused row.** The mechanism survives, but the tested
   object is `+26 B`, 17 flips better, and `+2.519822e-6 S` worse. This removed
   any claim that Schur compensation is itself a banked score win.
4. **RE1 has a final exact admitted row.** Its 0-byte, 2-flip result and exact
   pose leakage enabled the `RFO1-MICRO35` pose cap.
5. **HP4 closes post-hoc frame-embedding predictors.** Every predictive form
   loses, while its order-0 repack saves five bytes exactly. This changed HP4
   from a model route into an automatic container fold.
6. **HV1 corrected the JS7 budget lineage.** A universal 1.3e-7 shorthand is
   not safe; exact candidate-specific square-root inversion is required. This
   produced the corrected EC2 and MICRO35 gates.
7. **LP135 closes same-state coder work.** RC64 already beats ANS by 6–9 bytes
   and CAP1 is banked. This moved entropy work from coder choice to probability
   representation.
8. **PZ4A's failure is metadata-dominated.** It closes adaptive absolute
   precision, not rate-aware in-loop quantization. This narrowed rather than
   globally killed the quantization family.

## Measurement and authority boundary

- **MEASURED this arm:** cp135 archive presence, bytes, SHA-256, ZIP member
  anatomy, and primary scorer/runtime equations; all are read-only. Scalar
  arithmetic was recomputed exactly.
- **NOT measured this arm:** no new Seg or Pose value, no new candidate archive,
  no union support, no training, no n600 replay, no contest score, and no
  leaderboard movement.
- No payload-bearing run was launched, so this arm materialized no payload to
  retain. Future builders are explicitly bound to per-candidate retention.
- `[contest-CUDA T4, n600]` labels above are inherited only from pinned retained
  receipts. Projections are not promoted to that axis.
- The common contract's older `S=0.7539807 @357,836 B [macOS-CPU advisory]`
  line is superseded by the live hot-state cp135 contest pointer used here.

OWN-VEHICLE FRONTIER: `S = 0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`; unchanged.

## NEXT_IF_RESUMED

- **IN-FLIGHT** — owner: MAIN / EC2 lane; consumer store: EC2 retained store then #984; fire trigger: existing Modal call terminal with complete stage checkpoints, archives, and hashes; harvest and apply the exact 1,341/1,353-plus-pose gates without touching the live run beforehand.
- **QUEUED** — owner: MAIN-assigned #984 builder; consumer store: new SSD `RFO1-MICRO35` retained store and #984 composition receipt; fire trigger: a distinct lane is claimed for byte-only build and the EC2 scorer slot is released for exact evaluation; build, retain, decode, recount, and enforce the 35-flip/+29-B/pose cap.
- **QUEUED-CONTINGENT** — owner: #982 successor; consumer store: #982 retained trained-receiver store then #984; fire trigger: EC2 misses the nameable gate or exposes frame-conditioned residual structure; price the four-channel bilinear gate before a resumable retained train.
- **QUEUED-CONTINGENT** — owner: #978; consumer store: #978 receiver-native representation store then #984; fire trigger: random n>=32 parse-back survival beats HC1 at matched pose and the scorer lane is free; train semantic support, representative, and probability state jointly.
- **HELD** — owner: PZ4-QAT successor; consumer store: #984 rate branch; fire trigger: a retained parser-equal byte-only archive saves at least 2,000 bytes without a per-cell allocation map; only then run rate-aware in-loop QAT plus exported-object compensation.

## LIVE-HYPOTHESES

- A four-channel bilinear gate from the already-counted 8-D frame latent into EC1's oriented hidden state can buy frame-specific edge corrections at tens rather than thousands of logical weights; it is plausible because EC1 currently forms its correction before the explicit frame FiLM, while its edge ranking is already highly informative.
- The qs2/re1 supports plus one neighboring exact-lattice edit can reach 35 distinct net flips at no more than +29 bytes; it is plausible because the separate rows already realize 34 flips and HP4 returns five container bytes, but overlap must be measured.
- QS5-style in-compile compensation can recover at least `4.0031808967e-10 d_pose` from the MICRO35 union without a new payload section; it is plausible because QS5 proved the exact exported-object Schur mechanism and the required correction is smaller than re1's observed pose leak.
- A jointly trained multi-token representative can preserve HY1's cheap grammar while avoiding HC1's failed direct preimage; it is plausible because carriage represented every changed site, so the unresolved variable is receiver-native realization rather than addressability.
- Grouped or implicit sub-int12 QAT can pass the 2,000-byte gate without pz4a's allocation wire; it is plausible only if an existing structural partition supplies precision classes for free and training absorbs the quantization error.

## DEAD-ENDS

- The charter's `0.1243317` cp135 rate term and 1,340-flip EC2 integer gate are closed as arithmetic shortcuts: the retained 186,252-byte object gives `0.124017561736910658`, and exact ceiling gives 1,341 flips before pose tax.
- A bare “semantic x latent” addition is closed as fake novelty: the F26/cp135 receiver already combines semantic tokens and a 600x8 frame latent.
- HY1's direct C1 representative is closed for that implementation: HC1 exact evaluation produced roughly `S=0.404`; only its cheap grammar/addressability receipt survives.
- QS5's tested three-pair candidate is closed as a score row: 17 fewer flips and better pose do not repay +26 bytes, leaving it `+2.519822e-6 S` worse.
- Post-hoc HP4 frame-embedding predictors are closed for the tested formulations: all predicted forms enlarge the archive; only the receiver-identical five-byte repack survives.
- Same-symbol-state ANS and further CAP1 packing are closed for the current object: ANS loses RC64 by 6–9 bytes and CAP1 is already banked.
- Adaptive absolute per-cell precision is closed for the pz4a formulation: a 500-byte gross ceiling cannot repay a 2,732-byte allocation map, even before score risk.
- Explicit/linear terminal routes pk4, ps135b, js8, js1c, and the pz4a instance remain closed absent new mechanism evidence; this arm found none and did not retry them.
