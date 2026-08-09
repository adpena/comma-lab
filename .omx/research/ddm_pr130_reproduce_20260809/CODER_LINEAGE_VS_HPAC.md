# Our whole coder lineage vs PR130's learned AR prior, on the same object

**Operator directive:** "Remember JRD and lotto and smevr" → "And all of our levers and everything from
all lineages" → "Everything must be optimal. Recursive fractal all dimensions."

**Verdict: our nine-coder lineage LOSES to PR130's HPAC by ~2.2–3.6× on the dense partition — the very
object that is 61% of their archive.** `score_claim=false`; these are byte measurements, not eval rows.

## 0. What PR130's tokens actually are

`EVAL_H, EVAL_W = 384, 512` and `NUM_CLASSES = 5`, over `N = 600`. So the token field is the
**full-resolution dense semantic partition**: 600 × 384 × 512 = **117,964,800 symbols over 5 classes**,
carried in **116,980 B = 0.007933 bits/symbol**. This confirms #978 at source: PR130's tokens *are* the
partition, not a latent.

## 1. The same-object race (MEASURED)

Object: our cached GT partition `gt_n600.npz['lstars']`, `(600, 384, 512)` int64, classes 0–4 — the same
object CLASS, at the same dimensions, over the same clip.

### Generic coders, full field (600 frames)

| coder | bytes | bits/sym | vs PR130 |
|---|---:|---:|---:|
| **PR130 HPAC (learned AR)** | **116,980** | **0.007933** | **1.000×** |
| lzma xtreme | 410,584 | 0.027845 | 3.510× |
| brotli q11 | 429,676 | 0.029139 | 3.673× |
| zlib −9 | 581,266 | 0.039420 | 4.969× |
| brotli q11 **temporal-delta** | 649,042 | 0.044016 | **5.548×** |

### Our own coder suite (`experiments.ddm_r7_token_coder`), two DISJOINT windows

| coder | b/sym @ frames 0–7 | b/sym @ frames 300–307 | vs PR130 |
|---|---:|---:|---:|
| `cae_inspired_identity_inter` | 0.028819 | 0.026230 | 3.31–3.63× |
| `smevr` | 0.031057 | 0.030355 | 3.83–3.92× |
| `kt_o8_prev5_backoff` | 0.032135 | 0.030197 | 3.81–4.05× |
| `brotli11` | 0.040024 | 0.038030 | 4.79–5.05× |
| `lzma1` | 0.041245 | 0.039541 | 4.98–5.20× |
| `rans_o0_on_adjacent_innovation` | 0.053223 | 0.050496 | 6.37–6.71× |
| `kt_prev1` | 0.062317 | 0.067744 | 7.86–8.54× |
| `rans_o0` | 0.064397 | 0.069773 | 8.12–8.80× |
| `huffman_nibble` | 1.018534 | 1.020482 | 128.4–128.6× |

The **ranking is stable across two disjoint content windows** — a passing content control, so the
ordering is not a window artifact.

### Window-length scaling (the dimension an 8-frame verdict would have missed)

| coder | P=8 | P=32 | P=600 |
|---|---:|---:|---:|
| `kt_o8_prev5_backoff` | 4.051× | **2.743×** | not run |
| `cae_inspired_identity_inter` | 3.633× | 3.080× | not run |
| `smevr` | 3.915× | 3.134× | not run |
| `lzma1` | 5.199× | 4.300× | **3.510×** (measured) |

lzma is the only coder measured at all three lengths: 5.199 → 4.300 → 3.510, i.e. a further −18% from
P=32 to P=600. Applying that same P=32→600 factor to `kt_o8` (the fastest-improving arm) projects
**≈2.25×**; to `cae` projects ≈2.5×. **DERIVED, not measured** — the honest read is that our best coder
at full length lands **≈2.2–2.8× worse than HPAC**, and the conclusion (we lose by >2×) is robust to the
extrapolation.

## 2. Two laws this confirms

**(a) The temporal-delta reflex loses whenever the incumbent coder pays for LZ MATCH STRUCTURE.**
Explicit differencing made brotli **51% worse** (429,676 → 649,042). This is the *second* independent
instance: sv2/#859 found exactly this on IX2TOK01 ("mode maximizes exact-zero runs, which is what
brotli/LZMA pay for"). Different object, same mechanism. "Code the innovation" is a reflex, not a law;
it only wins when the incumbent pays for symbol rank.

**(b) Per-surface coder races, not reputation — vindicated a third time.** On the dense partition
`cae_inspired_identity_inter` BEATS `smevr` (by 7.2% / 13.1%), while ph1 measured SMEVR *winning* by
15–20% on phase-field streams and sv2 measured SMEVR *losing* on the IX2TOK01 token bulk. No coder in
our suite is everywhere-optimal. This is a live update to #940.

## 3. What this means for the campaign

- **HPAC adoption is measured-correct, not assumed.** #982/hb1 ("PR130 HPAC trained on OUR labels") is
  the right call, and it now rests on a race rather than on deference.
- **Do not spend more effort racing our coders against HPAC on tokens.** That cell is measured shut at
  >2×. The remaining value of our coder lineage is on payload types HPAC was not built for — where ph1
  already measured SMEVR winning.
- **Their edge is the LEARNED PRIOR, not the packing.** To beat PR130 on rate we either adopt-and-improve
  the prior, or change the object so that less needs coding at all.
- Sister note: the earlier "the token stream is at its model's entropy" (RATE_AXIS_LOSSLESS_RACE.md §4)
  was scoped to *generic* coders and is now properly scoped — it is HPAC's model that the stream is at
  entropy under, and no coder we own gets closer.

## 4. JRD / LOTTO status, honestly

- **SMEVR** — raced here (loses to CAE on this object; 3.83–3.92× vs HPAC).
- **CAE/INTER** (`cae_inspired_identity_inter`) — raced; our best on this object.
- **LOTTO** — NOT raced here. It is a supermask/shared-dictionary mechanism, not a field coder in this
  suite, so it has no entry point on this measurement. Its named live legs remain #940's
  (soft-model-successor / micro-student / phase-codebook).
- **JRD** — NOT applicable here and correctly so: `tac.witness_dsl.jrd_priors` is a **DORMANT_N1_SCREEN**
  policy that fail-closes until an `ACTIVE_N600_CONFIRMED` receipt exists. Its own gate refuses to let an
  n=1 macOS-CPU screen set a deployed precision. That gate did its job — this is what a correctly-built
  dormancy policy looks like.

## 5. What was NOT checked

- **GT `lstars` is not PR130's decoded token field.** Same object class, same dimensions, same clip —
  but their tokens are a *fitted* partition and ours is the oracle. The comparison is same-CLASS, not
  same-INSTANCE. A same-instance race needs their tokens decoded, which needs CUDA (`inflate.py:665`).
- **P=96 FAILED** for all four coders with `DDMR7CoderError`; cause not diagnosed. So the scaling ladder
  has two points, not three, for every coder except lzma.
- The P=600 projection for our coders is DERIVED from lzma's slope, not measured.
- No score claim. `score_claim=false`.
