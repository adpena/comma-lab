# ddm_gf1 — the generator FORM on lb1's OWN field: REFUSED, and the mechanism is a CAPACITY CEILING

`axis: [macOS-CPU scorer-free exact byte measurement]` · `score_claim: false` · `promotable: false`
`verdict_scope: FAMILY` (the HG1 analytic generator family, against any 5-class field)
Task #1334. Instrument `experiments/ddm_gf1_generator_form_on_lb1_field.py`.

## 1. The verdict

**REFUSED at 5.09×.** The generator form cannot carry lb1's own field losslessly at any price
near the bar. All figures below are MEASURED, from `RESULT.json` (44.5 s, scorer-free).

| | measured |
|---|---:|
| bar (re-derived at source, §2) | **85,020 B** |
| generator packet fitted to lb1's field | 47,603 B |
| **capacity gap on lb1's field** | **1,325,033 mismatches** (1.12324%) |
| exact residual to make the round-trip lossless | **385,448 B** coded |
| **total replacement** | **433,051 B** |
| **over the bar by** | **348,031 B = 5.09×** |

⚠ **My own pre-measurement bracket was wrong and is retracted.** The charter projected
8.4× / 16.1× / 31.7× at 0.5 / 1.0 / 2.0 coded B per correction. The MEASURED cost is
**0.2909 coded B per correction** — below my assumed floor, so the honest ratio is **5.09×,
not 8.4×**. The verdict is unchanged (5.09 ≫ 1) but the number was mine to get right.
The residual codes cheaply because the mismatches are spatially and temporally clustered,
which is the same structure §3a exploits.

### 3a. A real product fell out: the ORDER race, on a third object

The residual was raced across three linearization orders before coding — and the ordering
moved bytes more than the coder choice did:

| order | raw B | best coded B |
|---|---:|---:|
| `frame_raster` | 2,784,644 | 456,000 |
| `class_frame_raster` | 2,833,357 | 418,188 |
| **`tile16_time`** | 2,991,616 | **385,448** |

`tile16_time` is the LARGEST raw stream and the SMALLEST coded one — **−15.5% vs frame raster**
while carrying 7.4% more raw bytes. Within the winner, coders spread only 18.5%
(lzma2_extreme 385,448 · brotli_q11 392,957 · zlib_9 456,693).

This independently reproduces [[#1201]] (**reordering substitutes for a context model**) on an
object that law was not derived from. Two of the three orders here have never been raced on the
live body's streams.

## 2. The bar, re-derived (never quoted)

Parsed from the LIVE pointer archive's own RX1M header
(`runtime_candidate_native/archive.zip`, 180,083 B, sha `5b856e66…`):

```
RX1M hdr 14 + hpac 13,515 + semantic 30,856 + carrier 22,010  = models 66,395
payload 179,983 − models 66,395                                = tokens 113,588
token subsystem = 14 + 13,515 + 113,588                        = 127,117 B
demand          = 180,083 − 137,986                            =  42,097 B
BAR             = 127,117 − 42,097                             =  85,020 B
```

## 3. THE MECHANISM — the generator's capacity is target-INDEPENDENT

This is the finding, and it is stronger than the verdict.

| the HG1 generator fit to… | mismatches vs its own target | fraction |
|---|---:|---:|
| GT (bz2, `ddm_bz2` FULL_PACKAGE_RESULT) | 1,325,581 | 1.12371% |
| **lb1's own field (gf1)** | **1,325,033** | **1.12324%** |
| difference | **548** | **0.04%** |

Two completely different targets, and the generator lands **548 mismatches apart out of
117,964,800**. The ~1.12% is not a property of GT, nor of lb1's field — it is the **expressive
ceiling of the four analytic streams** (horizon interp · lane fit · movable boxes · mycar).

**Consequence, and it closes the route:** bz2's packet is small *because it is allowed to be
wrong on 1.12% of positions*. Its 47,779 B and its 1.32M errors are the same fact. You cannot
inherit the FORM without inheriting the FIT — making the form exact costs 1,325,033 corrections
at a MEASURED 0.2909 coded B each = 385,448 B, which is 4.53× the entire bar by itself, before
the 47,603 B packet is added.

## 4. What this RETRACTS

[[generator-form-is-2x-cheaper-than-model-plus-coded-tokens]] closed with:
*"A successor body that wants born-small's rate should inherit the generator FORM, not the GT-fit
field. The form is the asset; the field is the liability."*

**That advice is REFUTED by this measurement.** The 2.178× is real and remains real — but it is
2.178× *for a lossy approximator at 1.12% error*, and the two are inseparable. The memory is
corrected at source.

The measurement the race SHOULD have carried, and did not: the generator was never asked to
reproduce a field it had not generated. bz2's `parseback_equals_fitted_field: True /
corrections: 0` is a tautology — bz2 DEFINED its field as the generator's output.

## 5. Sisters + what survives

- [[token-error-amplifies-to-argmax-error-no-attenuation]] — the affine transfer law
  (`argmax ≈ 17,241 + 1.1435 × tokens`, n=2 matched-PYAV). Corrected the same day: the 1.157×
  ratio does NOT transfer (1.9738× at lb1's error scale), and the intercept is a
  render-manufactured floor no token work removes.
- [[the-cross-two-objects-each-hold-one-half-of-sub012]] — {byte-feasible} ∩
  {distortion-feasible} measured EMPTY at n=4; gf1 does not add a fifth body, it closes a
  proposed BRIDGE between two of them.
- **Surviving products, three:**
  1. the affine law's corrected bar (token error ≤ 0.0212% for sub-0.12) and the measurement
     that **lb1's own field already clears it at 0.0176%** — lb1 has a pure RATE problem, not a
     token-accuracy problem;
  2. **the order race (§3a)** — `tile16_time` beats `frame_raster` by 15.5% coded on a third
     independent object, and two of its three orders have never been raced on the live streams;
  3. the measured **0.2909 coded B per correction** on clustered residuals — a real price for
     any future "code the exceptions" proposal, replacing the 0.5–2.0 B/correction guesses that
     several charters (including my own) have been assuming.

## 6. Scope + honesty

- The field measured is `ddm_dc1_20260816/retained/redecoded_tokens_n600.u8`
  (sha `9ba2e52b…`), an lb1-**LINEAGE** decode; byte-identity to the live pointer's decoded
  field is NOT independently verified here. The capacity gap is ~1.3M; lineage members differ
  by ≲5k positions, so the verdict is insensitive to that gap by ~265×. Stated, not assumed.
- No score is claimed. This is a byte measurement with no scorer in the loop.
- Payloads retained under `/Volumes/APDataStore/pact/ddm_gf1_generator_form_on_lb1_field/retained/`
  per the ALWAYS-KEEP-THE-PAYLOAD rule.
