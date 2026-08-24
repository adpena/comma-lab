# ddm_tv2 — the evaluator's tolerance to token-field movement, measured

**Type: MEASURED (local CPU advisory n600 rows) against a MEASURED matched base.**
`score_claim: false` · `promotion_eligible: false` · `pointer_moved: false` ·
`evidence_grade: macOS-CPU advisory` · GT lineage `PYAV_YUV420_TO_RGB` (**NOT** the authority
`DALI_NVDEC`).

`verdict_scope`: **the dx2 object** — archive sha
`976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`, 180,368 B — its shipped F26
receiver, and its 600×384×512 five-class token field. A different object has a different field, a
different coder, and a different curve.

---

## 0. What this measures, and what it does NOT

One question: **how far can dx2's 117,964,800-position token field move before the frozen scorer
notices?**

Four alternative rate representations died this campaign (`hg1`, `hr3`, `et1`, `ws1`). Each was
priced *plus the cost of reproducing dx2's token field exactly*, and **no arm ever checked whether
that exactness was required.** `ddm_vf1` left the denominator at literally **0 of 117,964,800
positions**. The scorer never reads the token field; it reads rendered frames, reduced to an argmax
partition and six pose scalars. So an equivalence class of token fields exists in principle, and its
size has never been measured.

**This measures TOLERANCE ALONE. It is not a lever and it cannot become one.** Nothing here names
which positions moved and nothing codes anything. A representation wanting to exploit tolerance must
transmit the changed set, and this campaign has measured three separate times that naming a subset
costs more than the subset holds — `mf1`'s best perfectly-addressed repair removed 26 net seg errors
and still cost **+35,969 B**; `tba1` measured the same law directly; every lossy `ld1` rung enlarged
the archive. Tolerance and the address tax have never been measured apart, so a small tolerance and
a large address tax have been **indistinguishable**. This arm isolates the numerator so they never
are again.

Read every "credit" figure below as an **ADDRESSING-FREE UPPER BOUND**: it assumes the changed set
is free to name, which no real representation gets.

---

## 1. The k = 0 positive control — PASSED, at two levels

Publishing dx2's **unmodified** field through the injection path and rendering it produced:

| quantity | matched base (`mst1`) | `k = 0` control | ratio |
|---|---|---|---:|
| `avg_segnet_dist` | 0.00034740 | **0.00034740** | 1.000000 |
| `avg_posenet_dist` | 0.00014701 | **0.00014701** | 1.000000 |
| `archive_size_bytes` | 180,368 | 180,368 | — |
| `n_samples` | 600 | 600 | — |

And the stronger form — **the scorer's INPUT bytes are identical, not merely its output scalars**:
the control's `0.raw` has SHA-256
`7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7`, byte-identical to the matched
base across all **3,662,409,600 B**. Scalar agreement can coincide; 3.66 GB of byte identity cannot.

That single control settles four things by measurement rather than assumption:

1. the token-cache injection reproduces the shipped decode exactly at `k = 0`;
2. the render path is deterministic;
3. the matched base's `d_seg` / `d_pose` **transfer exactly** to these rows — identical scorer input
   bytes cannot produce different scorer output;
4. **`mst1`'s instrumentation is decode- and render-neutral.** The base came from `mst1`'s
   INSTRUMENTED tree; this control ran on the **SHIPPED** dx2 receiver. Byte-identity across two
   different trees excludes the `#1034` cross-instrument genus **by evidence, not by construction**.

### 1.1 The control k = 0 CANNOT provide — consumption, proven separately

**A `k = 0` control cannot distinguish "the injection works" from "the injection is silently
ignored", because at `k = 0` the injected field IS the base field — both hypotheses predict
byte-identical output.** Had the receiver quietly fallen back to decoding the shipped payload, every
rung would have returned the base numbers and the curve would have read as *perfect tolerance*: a
vacuous instrument scoring a maximal result. That is the `skip == green` failure shape, and it would
have been invisible without a second control.

So consumption is proven directly. The receiver reports the SHA-256 of the field it actually
rendered, and on every perturbed row it equals **that row's** counterfactual field and differs from
the base:

| row | receiver's `decoded_token_sha256` | that row's field sha | base field |
|---|---|---|---|
| `cond_cond` k=10,000 | `384df43d355f1434…` | `384df43d355f1434…` ✓ | `cc10a7b0…` ✗ |
| `cond_cond` k=100,000 | `f772e4e0e501d1ff…` | `f772e4e0e501d1ff…` ✓ | `cc10a7b0…` ✗ |
| `cond_cond` k=1,334,939 | `7140f154eaa3cd41…` | `7140f154eaa3cd41…` ✓ | `cc10a7b0…` ✗ |
| `unif_unif` k=100,000 | `e989b6ddab65f6f8…` | `e989b6ddab65f6f8…` ✓ | `cc10a7b0…` ✗ |

The receiver also stamps `token_codec: "ddm_tv1_counterfactual_field"` and
`provenance: "seeded counterfactual; NOT an arithmetic decode"`, and its
`token_decode_or_checkpoint_load` stage runs 12.5 s instead of a full arithmetic decode. The
counterfactual field is what rendered. **Vacuity is excluded by measurement, not by design intent.**

**Which tree, and why (correcting a charter instruction).** The charter directed keeping the
instrumented tree. That instruction was over-constrained, and the measurement beats the reasoning:
the instrumented tree is **structurally incapable** of this arm.
`cpr1/ddm_mst1_manufactured_stage_split.py:243` raises unless the decoded field hashes to
`cc10a7b0…` — dx2's exact field — so it refuses every counterfactual by construction, with a second
gate at `:456`; and its `_ensure_vertigo_store` demands 28 GiB free on Vertigo, which currently has
8.4 GiB. All rows therefore ran on the **shipped receiver**, and point 4 above is what licenses
that. No `INFLATE_DDM_MST1_STORE` is required on this path.

---

## 2. The instrument

### 2.1 Injection point

The shipped receiver decodes the payload to a token field, writes it to its own resumable checkpoint,
then renders. A counterfactual field is published into the receiver's **own** token cache
(`F26_ADVISORY_DECODE_CACHE_ROOT` — the designed knob), so the receiver loads it and runs the
**entire unmodified downstream**: semantic renderer → RGB → the real R/uint8 path →
3,662,409,600 B of raw frames → `upstream/evaluate.py` at n600. The receiver validates each cache on
load against a canonical manifest **plus a full-payload SHA-256**, so reuse is proven at consumption.

What is bypassed is only the arithmetic decode — a bijection from a payload to a field. It *cannot*
run on a counterfactual field because no payload for one exists. That is the point: the field is
priced independently of any coder.

### 2.2 The conditional is the shipped coder's own, proven

The reference arm resamples from the distribution the field's bits are actually priced against. That
distribution lives inside `residual_archive.decode_production_tokens` and is discarded; only two
digests survive. The replay feeds the already-decoded symbols through the identical model/corrector
loop and retains the corrector's coding row per position.

**The replay is proven, not asserted** — and structurally so: the capture script *raises* unless both
shipped digests reproduce (`ddm_tv1_capture_coding_conditionals.py:184-187`).

| digest | shipped receipt | replay |
|---|---|---|
| `corrected_quantized_logit_sha256` | `8269fe1aad0316…` | **identical** |
| `corrected_cdf_input_sha256` | `370a5e2a85ccbb…` | **identical** |

### 2.3 The arms — the reassignment rule IS the mechanism

With Gini **0.9951** on the bit mass and a near-certain median position (median cost 1.008e-8 bits),
"reassign a random position" is a *family*, and the wrong member measures receiver robustness to
garbage rather than the size of the scorer's equivalence cell. WHERE a position is drawn and WHAT
value it takes are independent choices, so they are run as a **2×2 with the uniform corner as
control** — never confounded on a diagonal:

| arm | positions drawn by | value drawn from |
|---|---|---|
| `cond_cond` **(reference form)** | model uncertainty `1 − p` | coder conditional, value ≠ current |
| `cond_unif` | model uncertainty `1 − p` | uniform over the 4 alternatives |
| `unif_cond` | uniform over the field | coder conditional, value ≠ current |
| `unif_unif` **(maximal control)** | uniform over the field | uniform over the 4 alternatives |
| `unif_marg` | uniform over the field | global class marginal, value ≠ current |

`cond_cond` is the **reference form the charter requires**: the shipped coder's own conditional in
both axes — movement within the manifold the field was priced against. The marginal arm
(`unif_marg`) is present as a declared comparison, not as a substitute. Uniform-only is **not** the
tolerance curve, and is reported only as the maximal-perturbation control.

Sampling is exact, not approximate: weighted draws without replacement use **Gumbel top-k**
(Plackett–Luce), and the uncertainty weight is computed as `−expm1(−cost·ln2)` rather than `1 − 2^−cost`,
which would lose every significant digit in float32 at the 1.008e-8-bit median. Replacement values
are drawn with the incumbent class zeroed, and a conditional row that underflows to all-zero falls
back to the global marginal **and is counted**, so a silent fallback cannot relabel part of the
conditional arm as a marginal arm.

**Never a contiguous prefix.** Prefix bias on this object is measured and axis-dependent — pose
**2.5–4.2× harder** on a contiguous prefix, seg ≈0.96× — and this is a pose-bearing measurement.
Every arm draws across the whole field.

Seeds are derived as `sha256("ddm_tv1|<arm>|<k>|20260824")[:8]` and recorded per row.

---

## 3. Independent verification — recounted, not read

A manifest is the producer's own claim about its own output; confirming it against itself proves
nothing. Every `k` was **recounted directly from the retained field bytes**, and the credit column
was **re-priced against `ddm_tb2`'s independently produced per-position cost field** — a different
instrument reaching the same object by a different route.

| arm | k target | **k recounted** | frames touched | bytes held (tb2) | tb2 ÷ tv1 manifest | credit (S) | **τ break-even** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cond_cond` | 1,000 | **1,000** ✓ | 499 / 600 | 221.8 | **1.0000** | 1.4769e-04 | 0.17422 |
| `cond_cond` | 10,000 | **10,000** ✓ | 600 / 600 | 2,206.7 | **1.0000** | 1.4694e-03 | 0.17333 |
| `cond_cond` | 100,000 | **100,000** ✓ | 600 / 600 | 21,158.1 | **1.0000** | 1.4088e-02 | 0.16619 |
| `cond_cond` | 1,000,000 | **1,000,000** ✓ | 600 / 600 | 104,166.7 | **1.0000** | 6.9360e-02 | 0.08182 |
| `cond_cond` | 1,334,939 | **1,334,939** ✓ | 600 / 600 | 107,906.1 | **1.0000** | 7.1850e-02 | 0.06349 |
| `cond_unif` | 100,000 | **100,000** ✓ | 600 / 600 | 21,158.0 | **1.0000** | 1.4088e-02 | 0.16619 |
| `unif_cond` | 100,000 | **100,000** ✓ | 600 / 600 | **87.8** | **1.0000** | 5.8462e-05 | 0.00069 |
| `unif_marg` | 100,000 | **100,000** ✓ | 600 / 600 | **92.5** | **1.0000** | 6.1592e-05 | 0.00073 |
| `unif_unif` | 100,000 | **100,000** ✓ | 600 / 600 | **96.3** | **1.0000** | 6.4122e-05 | 0.00076 |

Every `k` recounts exactly. Every independent re-pricing agrees with the manifest to **ratio 1.0000**.
`self_transitions = 0` on all nine fields — no drawn position kept its incumbent class, so the
realized `k` is the intended `k` and not an inflated draw count.

**One inherited claim is corrected here.** `tv1`'s memo states *"Every built field touches all 600
frames."* That is **false at k = 1,000, which touches 499 of 600** — 1,000 draws over 600 frames
leaves ~101 frames empty by construction.

**It is NOT prefix bias, and must not be re-filed as such.** The untouched frames are a *randomly
scattered* subset, not a contiguous block. The measured prefix hazard on this object
(`[[m88]]`; pose 2.5–4.2× harder, seg ≈0.96×) is a property of contiguous prefixes over a
scene-skewed population, and it does not apply to a random subset. Coverage below 600 here is a
sampling fact, not a bias.

### 3.0 Which quantity the Δ column is — units, level, aggregation

Incomplete coverage splits the measurement into **two different quantities that are NOT
interchangeable**, and at k = 1,000 they differ by **600/499 = 1.2024×**:

| quantity | definition | what it is for |
|---|---|---|
| **Archive-level `Δd_seg`** (**published**) | the **600-frame mean** — untouched frames contribute exactly zero and **stay in the denominator** | this is what the SCORE is; every τ, bar, and verdict below uses it |
| Per-touched-frame sensitivity | `Δd_seg × 600 / frames_touched` | only for EXTRAPOLATING to a rung with different coverage |

Every Δ and τ in this memo is **archive-level**, because the score is an archive-level mean.
Substituting the per-touched-frame quantity would inject a **1.20× error into the k = 1,000 rung** —
precisely the small end where the instrument-floor argument lives, so the confusion would be
maximally damaging exactly where it is least visible. Coverage is therefore printed beside every
rung (`[[the-instruments-own-units-level-and-aggregation-are-part-of-the-claim-20260816]]`).

### 3.1 The position rule alone changes the credit 241×, before any scorer runs

| position rule | k | bytes held in the shipped stream | credit (S) |
|---|---:|---:|---:|
| `cond_*` (model uncertainty) | 100,000 | **21,158.1** | 1.4088e-02 |
| `unif_*` (uniform over field) | 100,000 | **87.8** | 5.8462e-05 |

**A uniform-position perturbation cannot pay under ANY tolerance**, because the positions it moves
hold no bits: moving 100,000 uniformly-chosen tokens releases **87.8 bytes**. Its break-even τ is
0.00069 — 241× stricter than the conditional arm's. This is a **derived constraint on the whole
question**, established before a single scorer row, and it is why `cond_cond` is the headline rather
than a refinement, and why a uniform-only study would have answered a different question than the
one asked.

### 3.2 Why — the boundary geometry, and why the credit saturates

Measured distance-to-class-boundary for each arm's drawn positions, against the field-wide reference
that **2.17% of positions carry 94.53% of the bits**:

| arm | k | ON boundary (d=0) | d=1 | d=2 | d=3 | interior (d≥4) |
|---|---:|---:|---:|---:|---:|---:|
| `cond_cond` | 1,000 | **91.60%** | 2.00% | 1.30% | 1.40% | 3.70% |
| `cond_cond` | 10,000 | **90.66%** | 3.68% | 1.56% | 0.90% | 3.20% |
| `cond_cond` | 100,000 | **90.10%** | 3.66% | 1.62% | 1.01% | 3.61% |
| `cond_cond` | 1,000,000 | **79.83%** | 6.44% | 3.08% | 2.06% | 8.60% |
| `cond_cond` | 1,334,939 | **74.31%** | 7.60% | 3.85% | 2.58% | 11.66% |
| `cond_unif` | 100,000 | 90.23% | 3.67% | 1.52% | 1.06% | 3.52% |
| `unif_cond` | 100,000 | **2.06%** | 1.85% | 1.62% | 1.57% | 92.89% |
| `unif_marg` | 100,000 | **2.15%** | 1.83% | 1.69% | 1.64% | 92.69% |
| `unif_unif` | 100,000 | **2.21%** | 1.81% | 1.71% | 1.60% | 92.66% |

Two things fall out, and they are the same fact seen twice:

1. **The uniform arms reproduce the field's base rate exactly** (2.06 / 2.15 / 2.21% against 2.17%).
   They are, by construction, statistically indistinguishable from a random sample of the object, so
   they sit in the near-certain interior 92.7% of the time and hold ~90 B. That is the 241×.
2. **The conditional arm's boundary concentration DECAYS with k** — 91.6% → 74.3% as k goes from
   10³ to 1.33e6 — because the high-uncertainty boundary positions get **exhausted** and the draw is
   forced into the interior. This is precisely the mechanism behind the credit saturation in §4.1:
   the marginal position at the last rung collapses to 0.0893 bits because by then it is an
   *interior* position, and interior positions are what the coder is already certain about.

The two observations are one law: **on this object, bits live on the codim-1 boundary, and any
sampling rule is priced by how much of that boundary it can still reach.**

---

## 4. The bars, derived BEFORE any row

Exchange rate **6.658590e-07 S/B**, CITED from `ddm_tx1_toolbox_crosswalk_20260819.md` §0, never
re-derived. Seg marginal is exactly `d(S)/d(d_seg) = 100`, so break-even is
`Δd_seg* = 6.658590e-09 × B`.

The quantity that decides everything is the **transfer factor**

> `τ = Δd_seg_measured / (k / 117,964,800)`

i.e. how many argmax pixels the scorer actually loses per token moved, relative to the naive 1:1
prediction.

**τ > 1 is possible and is not a bug.** SegNet reads *regions* through a stride-2 stem and the
renderer paints regions, so one moved token can move more than one argmax pixel. `#1199` measured an
amplification exponent of ~16.7 for agreement → `d_seg` on this object, so amplification is the
prior, not tolerance.

### 4.0 What τ MEANS mechanically (DERIVED — this is a reading, not a measurement)

τ is not an arbitrary ratio. **It is the rate at which the render → SegNet round trip RECOVERS a
token change.**

The base `d_seg = 0.00034740` means **99.965% of all positions currently AGREE with the GT argmax**.
So a position drawn at random is almost certainly an agreeing one, and moving its token can only
convert agreement into disagreement — **but only if the change survives the round trip**. Hence:

| τ | what it says about the object |
|---:|---|
| ≈ 1 | the round trip is faithful — the token field *is* the argmax partition, which is exactly what this representation was built to make true |
| ≈ 0 | the round trip discards the change — the representation is spending bits on a field the scorer largely ignores |
| > 1 | one token change spills across neighbouring pixels (region effects, the stride-2 stem) |

**This is why the prior is τ ≈ 1 or above, not tolerance.** The entire purpose of the semantic token
field is that SegNet should see it; a large tolerance would mean the representation is paying for
precision the scorer cannot read — an interesting finding, but the opposite of the design intent.

The bars demand **τ ≤ 0.17** (small k) or **τ ≤ 0.06** (k = 1.33e6) for tolerance to pay. That
requires the round trip to discard **more than 83%** of all token changes. Both outcomes are
informative, and the measurement — not this reading — decides.

| bar | what it prices | required τ for tolerance to pay |
|---|---|---:|
| **B1 mean-price** | k random positions at 0.00771594 b/pos | **τ ≤ 0.00076** |
| **B2 top-k oracle** | the k costliest positions, address given away | **τ ≤ 0.0851** (k=1e6) |
| **B3 hg1 marginal** | `hg1`'s own measured drop elasticity, 0.348 b/correction | **τ ≤ 0.0342** |
| **B4 hg1 full residual** | deleting `hg1`'s entire 359,280 B residual | **τ ≤ 0.2114** |

B2 is an oracle bound that ignores addressing entirely and is therefore unreachable; it is included
because it is the most generous number the evidence permits. **B4 is the bar that decides the four
dead representations.**

### 4.1 The credit ceiling, and how fast it saturates (MEASURED, before any row)

The whole token stream is **113,776.2 B**, so **no tolerance-based scheme on this object can ever
release more than that** — 0.07576 S, or **2.68× the 0.028220 S remaining gap**. Tolerance is
therefore *large enough in principle* to matter; the only question is what the scorer charges.

But the credit saturates hard, and the marginal position collapses:

| k | bytes held | % of the 113,776 B stream | marginal bits/position |
|---:|---:|---:|---:|
| 1,000 | 221.8 | 0.195% | 1.7744 |
| 10,000 | 2,206.7 | 1.940% | 1.7644 |
| 100,000 | 21,158.1 | 18.596% | 1.6846 |
| 1,000,000 | 104,166.7 | **91.554%** | 0.7379 |
| 1,334,939 | 107,906.1 | 94.841% | **0.0893** |

**`k = 10⁶` already captures 91.55% of the entire stream.** The next 334,939 positions — 33% more
movement — add only **3,739 B, or 3.3% more credit**, at a marginal cost **9.3× cheaper per position
than the k=10⁶ average**. That is the `average ≠ marginal` law (`ddm_fs3`) reappearing on a new
axis: past k≈10⁶ the curve is buying distortion at full price and credit at a tenth of it.

**Consequence, independent of the scorer:** the useful range of any tolerance exploitation on this
object ends near k≈10⁶. Rungs beyond it can only worsen the exchange, so a large-k result cannot
rescue a small-k failure.

---

## 5. RESULTS — the curve

### 5.0 What a rung costs, and what the instrument can resolve

**Measured on the `k = 0` control, running alone:** inflate **499.4 s** + evaluate **523.3 s** =
**1031.2 s (17.2 min)** per rung, peak RSS **10,741 MiB (10.5 GiB)**, writing 3,662,409,600 B of raw
frames. Two corrections to prior estimates: the charter's "~7 min per row" was **the inflate leg
only**, and `tv1`'s memo estimated ~5 GiB peak RSS against a measured 10.5 GiB. The inflate leg is
that short *only* because the published token cache replaces the arithmetic decode
(`token_decode_or_checkpoint_load` = 12.5 s).

**Per-rung wall-clock, MEASURED** (18-core host; the k=0 control ran alone, the rest 4-wide):

| rung | concurrency | inflate | evaluate | total |
|---|---:|---:|---:|---:|
| `k0_control` | **1 (solo)** | 499.4 s | 523.3 s | **1024.3 s** |
| `cond_cond` k=10,000 | 4 | 1364.2 s | 602.7 s | 1968.4 s |
| `cond_cond` k=100,000 | 4 | 1221.9 s | 630.8 s | 1854.5 s |
| `cond_cond` k=1,334,939 | 4 | 1263.4 s | 619.9 s | 1884.8 s |
| `unif_unif` k=100,000 | 4 | 1214.3 s | 633.2 s | 1849.4 s |

**The concurrency exchange, measured rather than assumed:** 4-wide makes each row **1.81–1.92×
slower**, so throughput is **≈2.16× — not 4×**. A plan that priced 4-wide as a 4× speedup would have
been wrong by nearly half.

**And the tax is not spread evenly — it has a located cause.** The *inflate* leg degrades 2.4–2.7×
while *evaluate* degrades only 1.15–1.21×, because inflate contains **two sub-phases with opposite
scaling**:

| sub-phase | bound by | scales with concurrency? |
|---|---|---|
| neural render + resize (~434 s solo) | **CPU** | yes, up to core count |
| raw write + SHA-256 of 3,662,409,600 B (~50 s solo) | **disk** | **no — it contends** |

The store sustains a **measured ~39.7 MB/s** (`/Volumes/APDataStore`, ExFAT via fskit, sampled over
4 s while three rows finalized). Three rows finalizing together must move ~11 GB through that one
pipe, so the finalize phase **serialises regardless of how many rows are running** — and during it
the inflate processes sit at 0.0–2.4% CPU, which looks exactly like a hang and is not one.

**Operational consequence:** concurrency helps the render and does nothing for the finalize. Past
~3–4 rows the added rows buy CPU overlap and pay disk contention, which is why throughput saturates
near 2.2× rather than tracking the 18 cores.

The governor (`safe_run.py` SUM-over-RAM crash guard, adaptive ceiling 116.0 GiB) refused further
launches while those four were in flight — projecting 177.0 GiB against 116.0 GiB
(`53.8 used + 115.3 active-growth + 8.0 new`) — so **concurrency on this host is bounded by the
in-flight set's growth budget, not by a policy number.** Every refusal was treated as *wait*, never
as *retry harder*.

**Resolution floor.** The report carries 8 decimals, so `d_seg` resolves to **1e-8 = 1.18 argmax
pixels**. Against the 1:1 prediction `8.477e-9·k`:

| k | 1:1 `Δd_seg` | in ulp of 1e-8 | smallest resolvable τ |
|---:|---:|---:|---:|
| 1,000 | 8.477e-06 | 848 | 1.2e-03 |
| 10,000 | 8.477e-05 | 8,477 | 1.2e-04 |
| 100,000 | 8.477e-04 | 84,771 | 1.2e-05 |
| 1,000,000 | 8.477e-03 | 847,714 | 1.2e-06 |

Every decision threshold in §4 (0.2114 / 0.0851 / 0.0342 / 0.00076) is resolvable at every rung from
k = 1,000 up. **`k = 1` and `k = 10` sit below the instrument floor** (0.85 and 8.5 ulp) and were not
run — a declared SCOPE reduction, not an omission. The ladder therefore begins at k = 10³.

### 5.1 The curve

Matched base: `d_seg 0.00034740` · `d_pose 0.00014701` · `S 0.19318153` (the `k = 0` control, which
is byte-identical to `mst1`'s base — so base and candidate share one instrument by construction).
All Δ are **archive-level 600-frame means** per §3.0. Every figure is recomputed from the 8dp report
components; no rounded display was read.

| arm | k | coverage | `d_seg` | `Δd_seg` | **τ** | τ bar | over | `Δd_pose` | ΔS_seg | **ΔS_pose** | ΔS_total | credit (S) | **cost/credit** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cond_cond` | 1,000 | **499/600** | 0.00035619 | +0.00000879 | **1.0369** | 0.17422 | 6.0× | +0.00029610 | 0.0009 | **0.0282** | **+0.0291** | 1.477e-04 | **197.1×** |
| `cond_cond` | 10,000 | 600/600 | 0.00042406 | +0.00007666 | **0.9043** | 0.17333 | 5.2× | +0.00278920 | 0.0077 | **0.1330** | **+0.1407** | 1.469e-03 | **95.7×** |
| `cond_cond` | 100,000 | 600/600 | 0.00097857 | +0.00063117 | **0.7446** | 0.16619 | 4.5× | +0.03004845 | 0.0631 | **0.5112** | **+0.5743** | 1.409e-02 | **40.8×** |
| `cond_cond` | 1,334,939 | 600/600 | 0.00633976 | +0.00599236 | **0.5295** | 0.06349 | 8.3× | +0.67927745 | 0.5992 | **2.5682** | **+3.1675** | 7.185e-02 | **44.1×** |
| `unif_cond` | 100,000 | 600/600 | 0.00049314 | +0.00014574 | **0.1719** | 0.00069 | 249× | +0.20613559 | 0.0146 | **1.3979** | **+1.4125** | 5.846e-05 | **24,161×** |
| `unif_marg` | 100,000 | 600/600 | 0.00048485 | +0.00013745 | **0.1621** | 0.00073 | 222× | +0.14163509 | 0.0137 | **1.1524** | **+1.1661** | 6.159e-05 | **18,933×** |
| `unif_unif` | 100,000 | 600/600 | 0.00049059 | +0.00014319 | **0.1689** | 0.00076 | 222× | +0.15258047 | 0.0143 | **1.1975** | **+1.2118** | 6.411e-05 | **18,898×** |

**Tolerance loses at every rung** — 95.7× at k = 10⁴, 40.8× at k = 10⁵, 44.1× at `hg1`'s exact
cardinality. The k = 10⁵ rung moves under **0.085%** of the field and costs **20.4× the entire
remaining gap**; the k = 1.33e6 rung costs **112.2× the gap**.

**τ EXCEEDS 1 at the small end and decreases monotonically with k** —
**1.0369** → 0.9043 → 0.7446 → 0.5295 across k = 10³ → 10⁴ → 10⁵ → 1.33e6.

**τ > 1 at k = 10³ means AMPLIFICATION: one moved token costs MORE than one argmax pixel.** §4.0
predicted this was possible (the renderer paints regions; SegNet reads them through a stride-2 stem)
and the smallest rung measures it. There is not merely *no* slack at the boundary — the object is
**worse than 1:1 there**.

### 5.1.2 LAW — damage per token is monotone DECREASING in k, and exceeds 1.0 at the small end

> **The first token you move is the worst trade you will ever make, and every subsequent one is
> cheaper — because selection is value-ranked over a Gini-0.9951 field, so the most bit-valuable
> positions ARE the most damage-prone positions.**

This **inverts the usual intuition** that small perturbations are cheap because damage is locally
linear. Here damage per token is *maximal* at k → 0. It is the enrichment signature, and it is
corroborated by two independent arms that measured the enrichment directly: `tb2` (Gini **0.9951**;
top 1% of positions = **96.32%** of the bits) and `wj1` (cost and render-manufactured error
**90.96× count / 257.48× bit** enriched over independence).

**This closes the curve at BOTH ends, which is stronger than "every rung loses":**

| end | what pins it | measured |
|---|---|---:|
| small k | **amplification** — τ > 1, and credit is tiny | 197.1× at k = 10³ (221.8 B released for **103.1%** of the entire gap, moving 0.00085% of the field) |
| large k | **the credit ceiling** (§4.1) — saturation | 44.1× at k = 1.33e6; k = 10⁶ already holds 91.55% of the 113,776 B stream, marginal position 0.0893 bits |
| interior | the ratio's measured **minimum** | **40.8×** at k = 10⁵ |

`cost/credit` over k is **U-shaped — 197.1 → 95.7 → 40.8 → 44.1 — with a measured interior minimum
of 40.8×**, and both ends rise away from it. So the strongest honest form is not "every rung we ran
lost" but **"the best point of the whole family is measured, and it is 40.8× above break-even."**

### 5.1.3 Why the interior cannot hide a winner — STRUCTURAL, not interpolative

"No unmeasured rung can win" would otherwise rest on an unstated smoothness premise — *that the ratio
has no ~40× dip between decade-spaced samples* — which four log-spaced points cannot establish. It
does not need to, because **the selection rule makes both curves concave by construction**:

> **A value-ranked draw over a sorted field yields concave cumulatives on BOTH axes, so the ratio's
> measured minimum is the family's minimum.**

Positions are drawn weighted by model uncertainty, so higher-weight positions enter earlier *in
expectation* and the marginal contribution of the j-th position decreases in j. `credit(k)` is
therefore a concave increasing function; and §5.1.2's law — damage per token monotone decreasing in
rank — makes `damage(k)` concave increasing too. **Two concave increasing functions cannot produce a
40× interior dip in their ratio between decade samples**; that would require one of them to be
violently non-monotone at sub-decade scale, contradicting the sorted-selection mechanism that
generates both.

**Measured, not just argued** — both marginals are strictly decreasing across every sampled interval:

| interval | marginal credit (B/position) | marginal damage (S/position) |
|---|---:|---:|
| 10³ → 10⁴ | 0.220544 | 1.240e-05 |
| 10⁴ → 10⁵ | 0.210571 | 4.818e-06 |
| 10⁵ → 10⁶ | 0.092232 | — |
| 10⁶ → 1.33e6 | **0.011164** | 2.10e-06 (10⁵ → 1.33e6) |

**One honest caveat about the form of the concavity.** Each rung uses an independent seed
(`sha256("ddm_tv1|<arm>|<k>|20260824")`), so the k = 10³ set is **not a subset** of the k = 10⁴ set —
these are independent weighted draws, not a nested greedy path. The concavity is therefore a property
**in expectation** over the draw, not a pathwise cumulative. The measured marginals above confirm it
holds on the realized draws; the structural argument is why it must.

So the interior is bounded by the **construction of the selection**, not by the density of the
sampling. (Scoped to this family and this object — §9.)

**The same wall, reached from the coder side.** `ddm_df1` derived that the address floor for the
free-byte set is **3.1468× the prize it holds**, because the error is where the bits are, so naming
the cheap positions costs more than they hold. This arm measured the *physical* form of the same
fact: **the cheap positions are not cheap, because cheapness and damage are the same enrichment.**
Two arms, two methods, one wall — which is what makes the result transferable past this arm.

τ falls at larger k only because the conditional draw exhausts the boundary and reaches into the
interior (§3.2) — the same mechanism that saturates the credit, so it buys nothing.

**Where this sits on the campaign's exchange ladder.** 40.8× is *not* a catastrophic outlier; it is
the **third-best exchange ratio the campaign has measured**:

| mechanism | damage ÷ credit |
|---|---:|
| `tba1`-D3 | **21.62×** |
| W72 renderer | **35.54×** (matched-lineage; the circulating 46.32× divided a macOS-CPU numerator by the contest-CUDA pointer — `#1034`, corrected by `ddm_df1`) |
| **tolerance-cond (this arm)** | **40.8×** |
| `ni1` | 247.69× |
| `dg2` diagonal | 686–792× |

Tolerance therefore fails for the campaign's **general** reason, not a special one — which makes the
verdict sturdier, not weaker: there is no cheap corner here that a better perturbation design would
find. (`nr1`'s "349×" is deliberately absent: it is a proxy-**understatement** factor, a different
category, and does not belong in an exchange-ratio ladder.)

### 5.1.1 FLAGGED — the credit column is a STATIC accounting, and it OVERSTATES

`ddm_ds1` measured that static `−log₂p` accounting on this exact token field **inverts verdicts**:
static said 6,187 B / 0.762× ADOPT where the real closed-loop receiver said 12,224 B / 11.567×
REFUSE — a **14.59× misprice, in the direction that manufactures false wins**, because skipping
feeds the model its own guesses so confidence rises while accuracy collapses. This is `fs2`'s
direction-dependence law (`[[price-token-field-levers-by-real-reencode]]`) measured an order of
magnitude larger.

**That bias applies to this memo's credit column, and it must be flagged rather than waved past.**
The distortion side is safe — every Δ came from a real scorer row through the real receiver. But the
**credit** side (`bytes held in the shipped stream`) is a *static* sum of per-position costs in the
**unmodified** stream. The ratio-1.0000 agreement in §3 is between two *static* instruments
(`tv1`'s conditional and `tb2`'s cost field) computing the same static quantity; **independent
agreement on a static number does not make it a real re-encode.** No modified stream was ever
re-encoded here, because this arm deliberately codes nothing (§0).

**Direction of the error, and why the verdict survives it.** Static accounting **overstates**
releasable bytes, so the true credit is *smaller* than published and every `cost/credit` ratio is
therefore a **LOWER bound**. The real ratios are ≥ 40.8× / 44.1× / 95.7×, not ≤. The verdict is
conservative by construction, and the flag strengthens it — but the credit column should not be
lifted out of this memo and reused as if it were a measured re-encode.

### 5.2 The finding nobody predicted: POSE is the killer, not seg

The whole question was framed on the seg axis — `vf1`'s 0/117,964,800 denominator, `hg1`/`hr3`'s
residuals, the pre-registered prediction, and every bar in §4. **The seg axis is not what kills it.**

| arm | k | ΔS_seg | ΔS_pose | **pose ÷ seg** | `d_pose` as multiple of base |
|---|---:|---:|---:|---:|---:|
| `cond_cond` | 100,000 | 0.0631 | 0.5112 | **8.10×** | **205×** |
| `cond_cond` | 1,334,939 | 0.5992 | 2.5682 | **4.29×** | **4,622×** |
| `unif_unif` | 100,000 | 0.0143 | 1.1975 | **83.63×** | **1,039×** |

Seg contributed **11%** of the damage at k = 10⁵ and **1.2%** in the uniform arm. The mechanism is
the concavity of the pose term: `√(10·d_pose)` sits at 0.038342 at base, where its derivative is
enormous, so even a modest absolute rise in `d_pose` converts into whole units of S.

---

## 6. THE τ-INVERSION — seg-support and pose-support are near-DISJOINT

**This is a law about the object, not a supporting detail of the tolerance verdict.** It re-reads
already-closed results and constrains every future perturbation design, so it is stated on its own.

The 2×2 (never a diagonal) separates WHERE a position is drawn from WHAT value it takes. Holding k
fixed at 100,000 and varying only the **position rule**:

| position rule | ON boundary (d=0) | interior (d≥4) | **seg damage (τ)** | **pose damage (Δd_pose)** | bytes held |
|---|---:|---:|---:|---:|---:|
| `cond` (model uncertainty) | **90.10%** | 3.61% | **0.7446** | +0.0300 | 21,158.1 |
| `unif` (uniform over field) | **2.21%** | **92.66%** | **0.1689** | **+0.1526** | 96.3 |

**The two distortion axes rank the two position rules in OPPOSITE order.** Uniform positions do
**4.4× LESS seg damage** and **5.1× MORE pose damage** than conditional ones. The mechanism:

- **Seg damage tracks boundary proximity.** The argmax partition changes where its boundary moves, so
  perturbing the codim-1 boundary hits `d_seg` at or above 1:1 (τ 0.745–**1.037**) while leaving
  region interiors intact.
- **Pose damage tracks photometric disruption ANYWHERE.** PoseNet scores the *frames* — photometry,
  which is mostly interior. Uniform draws land 92.7% in region interiors and paint isolated
  wrong-class blobs through the middle of large regions: nearly invisible to the argmax, ruinous to
  the pose head (`d_pose` → 1,039× base).

**So "distortion" is not one quantity. It is two quantities living on two nearly disjoint spatial
supports** — `d_seg` on the 2.17% of positions at the class boundary, `d_pose` on the 92.8%
interior. The campaign has been pricing distortion as a single scalar; that is only safe for a
perturbation whose support is known.

### 6.0 The sharper statement: seg SLACK and pose DAMAGE are CO-LOCATED

Near-disjoint *damage* is the observation. The **complement** is the stronger claim, and it comes
from reading τ's decrease with k as a fact about *place* rather than about *scale*:

| where | **τ** | reading | pose cost |
|---|---:|---|---:|
| boundary-concentrated draws (90.1% on boundary) | **0.9043** | 90.4% of changes reach the argmax → **essentially ZERO seg slack** | cheap (+0.0300) |
| interior-concentrated draws (92.7% interior) | **0.5295** | ~47% of changes never reach the argmax → **real seg slack EXISTS** | **catastrophic (1,039× base)** |

**The only place the argmax tolerates movement is the place PoseNet is watching.** Seg slack is not
absent from the object — it is *guarded*. Every position is expensive on at least one axis, and
which axis is decided by the boundary/interior split.

### 6.0.1 This gives `vf1`'s 0 / 117,964,800 census a MECHANISM — and an honest residual

`vf1`'s census (zero free positions) has stood as a brute empirical fact with no explanation. The
two scorers partition the field between them:

| region | share of field | which scorer guards it |
|---|---:|---|
| boundary, d = 0 | **2.17%** | `d_seg` (94.53% of the bits also live here) |
| interior, d ≥ 4 | **92.80%** | `d_pose` (photometry) |
| **near-boundary annulus, d = 1–3** | **5.04%** | **NOT classified by this instrument** |

**The two guarded regions account for 94.97% of the field, not 100%.** The remaining **5.04%** is the
d = 1–3 annulus, which this arm's 2×2 does not assign to either axis: no arm was targeted at it. It
is not measured to be free — it carries **3.61% of the stream's bits** and 5.92% of the model's
uncertainty mass — but it is **not measured to be guarded either**. Stating it as "every position is
expensive" would publish a clean 100% that this instrument did not earn. **If that 5.04% is
genuinely free, then `vf1`'s census is the result needing explanation, not this one.**

**The caveat is honest AND bounded.** That annulus holds 3.61% of 113,776.2 B = **4,107.3 B =
2.735e-03 S**, which is **9.69% of the 42,382 B demand**. So **even if all 5.04% turns out to be
entirely free, it cannot close the gap** — it is capped at under a tenth of it. The `vf1` question
remains open as a matter of understanding; it is now proven unable to produce the win.

### 6.0.2 The vise, in one sentence

**Essentially all of the rate lives on exactly the positions with essentially no seg slack.**

| region | share of positions | share of bits | bit enrichment | τ (seg slack) | guarded by |
|---|---:|---:|---:|---:|---|
| boundary d=0 | 2.17% | **94.53%** | **43.6× over** | **0.9043** (none) | seg |
| interior d≥4 | 92.80% | **1.86%** | **49.9× under** | 0.5295 (real) | pose |

The slack is in the interior, which is **50× under-enriched in bits** and pose-guarded. **This is why
the 21–46× exchange band is so stable across unrelated mechanisms:** every one of them is either
trying to buy bits from the one place that will not yield, or yield from the one place that has no
bits.

### 6.0.3 FACTOR SEPARATION — the inversion is driven by WHERE, not by WHAT

Arm names are `<position rule>_<value rule>`. The 2×2 exists precisely so the two factors can be
varied one at a time. At k = 100,000:

| arm | position | value | **ΔS_seg** | **ΔS_pose** |
|---|---|---|---:|---:|
| `cond_cond` | **boundary** | cond | **+0.0631** | **+0.5112** |
| `unif_cond` | interior | cond | +0.0146 | +1.3979 |
| `unif_marg` | interior | marg | +0.0137 | +1.1524 |
| `unif_unif` | interior | unif | +0.0143 | +1.1975 |

**Vary POSITION with the value held at `cond`** (`cond_cond` → `unif_cond`):
seg **4.32× down**, pose **2.73× up**. The inversion.

**Vary VALUE with the position held at `unif`** (three rows):
seg spans 0.0137–0.0146 — a **6.6%** range. Pose spans 1.1524–1.3979 — a **21.3%** range.

**So the τ-inversion is an isolated effect of WHERE, not an interpretation.** The orthogonal factor
moves seg by 6.6% while position moves it by 432%; §6.0's attribution is now a controlled two-factor
result rather than a reading of the diagonal.

**One honest qualification: the value factor is inert on SEG, but NOT fully inert on POSE.**
`unif_cond` — conditional values, which sit on the model's own prediction — is **21.3% worse on pose**
than `unif_marg`. So "inert across all three" is true for seg and an overstatement for pose. The
value factor is **second-order** (6.6% / 21.3%) against a **first-order** position factor (432% /
273%); it is not zero.

**Still missing: the fourth corner `cond_unif`** (boundary positions, uniform values), which is
queued. Its absence does not affect the separation above — that is established by holding value at
`cond` across the position change — but the 2×2 is not literally complete until it lands.

### 6.1 What this re-reads (no new measurement required)

- **`dg2`'s 93.3%-pose diagonal** — a joint field+model move rewrites region *interiors*, which is
  precisely the pose support. It paid on pose because of where it lived.
- **`#1222`** — PoseNet scores the frames, i.e. photometry, i.e. interiors. The τ-inversion is that
  finding's **spatial** statement.
- **`#1234` / `msr1`'s vise, now one shape** — boundaries are byte-EXPENSIVE (94.53% of bits on
  2.17% of positions), carry `d_seg`, and already sit at an oracle ceiling of one pixel; interiors
  are byte-CHEAP and catastrophic to touch on pose. **There is no set that is simultaneously
  expensive in bytes and cheap in damage.** That is a sharper statement of why every mechanism this
  campaign has priced lands in the same 21–46× band.

### 6.2 The confound this 2×2 prevented

A study using **only the uniform arm** — the natural naive choice, and the one a "reassign random
tokens" framing produces — would have reported **τ = 0.169**, within 1.3× of the k = 10⁵ break-even,
and read as *nearly viable*. It is in fact the **18,901×** worst arm measured, because its pose cost
is 84× its seg cost and its positions hold 241× fewer bytes. **The uniform arm is cheap on the axis
everyone was watching and ruinous on the axis nobody was.** Confounding position rule with value
rule on a diagonal would have hidden this completely.

**Generalised — this is not a precision miss, it is a SIGN-LEVEL inversion of the verdict:**

> **A single perturbation family cannot measure tolerance, because the families rank-INVERT between
> the two axes.** Whichever family you pick, it is cheap on one axis and ruinous on the other, so a
> one-family study reports the axis it happens to be cheap on and reads as nearly viable.

The requirement is therefore structural, not a matter of care: **any tolerance or perturbation study
on this object must run at least two position families and score BOTH axes**, or its verdict can
land on the wrong side of break-even. This transfers well past this arm and is the same shape as the
campaign's closed-from-one-rung genus.

### 6.3 OPEN — the question §6.0 raises and this arm does NOT answer

The interior carries **real seg slack** (τ 0.5295 ⇒ ~47% of interior token changes never reach the
argmax). That slack is forbidden **only by pose**. The obvious next thought is *"move interiors in a
pose-null way"*.

**This arm does not answer that, and no answer should be derived from memory.** The question lives in
a neighbourhood with at least three prior results that must be RE-READ first, none of which were
consulted here:

- **`#837`** — is the exactly-pose-null `frame_1` subspace seg-reachable?
- **`j11`** — its pose-null / seg-null proposal split.
- **`#1029`** — the projected-vs-unprojected semantic-cell census.

Recorded as **OPEN with named recall targets**, deliberately without a verdict. Deriving one from
what anyone remembers is the exact move that has cost this campaign arms this week; and note that
`ddm_pz1` already measured a closely related trap — a field built to be null in one frame's lattice
is **resampled onto a different lattice** by the pose warp, where it is no longer null.

---

## 7. The linkage to the four dead representations — checked, not asserted

### 7.1 What the "unique-home residual" actually is (read at source)

The name suggested an addressing cost. It is **both**, and the split is measured.
`experiments/ddm_hg1_heterogeneous_analytic_generator_gate.py:466-470` emits a ULEB zigzag address
delta followed by one target-class byte per correction.

| arm | mechanism | verdict |
|---|---|---|
| `hg1` | 1,334,939 `(address, value)` corrections; raw 2,871,598 B = **46.49% value / 53.51% address**; coded 359,280 B = 2.153 bits/correction joint | **MIXED** |
| `hr3` | calls `hg1.encode_residual` directly; 362,473 B = 3,223 B INR + 359,176 B residual. The learned INR removed **104 B — 0.03%** | **MIXED**, identical mechanism |
| `et1` | 482,678 B is *implicit generator topology* (a BSP tree), not a correction residual at all | **TOPOLOGY-BOUND** |
| `ws1` | per-stratum lifetime-event + innovation streams; no correction residual exists | **NEITHER** |

So the linkage is **not uniform across the four**. Tolerance can only speak to `hg1` and `hr3`,
whose bytes are corrections. `et1`'s bytes are topology and `ws1`'s are event streams; a tolerance
result does not reach them, and claiming otherwise would be the assertion this section exists to
avoid.

### 7.2 The elasticity, from hg1's own table

Post-coder the value and address halves are not separable, so tolerance does not shave a value field
while the address stays — **it deletes whole records**. `hg1` measured that elasticity itself:

| bracket | corrections | coded B | bits/correction |
|---|---:|---:|---:|
| exact | 1,334,939 | 359,280 | 2.153 |
| BL1+MS9 protected | 422,024 | 319,518 | 6.057 |
| **dropped** | **912,915 (68.4%)** | **saved 39,762 (11.1%)** | **0.3484** |

**Dropping 68.4% of all corrections bought 11.1% of the bytes** — the
`perfect-localization-is-the-tax` law again, from inside hg1's own receipts.

### 7.3 The honest limit of this linkage

The perturbation is drawn from the *coder's* uncertainty over the *incumbent* field. `hg1`'s
1,334,939 differing positions are where an *analytic generator* disagrees with the incumbent — a
structured, generator-specific set. Both sets are boundary-concentrated and of the same cardinality,
so this curve **bounds the order of magnitude of hg1's answer; it does not supply hg1's number.**
Only hg1's own field through the real receiver does that, and everything needed is retained.

**None of the four arms ever fired a scorer.** `hg1`: *"No scorer was fired."* `ws1`: *"This arm did
not run a scorer."* `et1` named evaluator-cell equivalence a LIVE HYPOTHESIS. `hr3` gated a fire on a
residual that never arrived. All four assumed exact reproduction and priced it. **This arm fired the
scorer, and the assumption was correct.**

### 7.4 The verdict, at hg1's exact cardinality

The k = 1,334,939 rung was built to `hg1`'s **exact** correction count so the comparison needs no
extrapolation in k:

| quantity | required for `hg1` to reopen | **measured** | margin |
|---|---:|---:|---:|
| τ (B4 — delete hg1's whole 359,280 B residual) | ≤ **0.2114** | **0.5295** | **2.51× over, on SEG ALONE** |
| τ (B3 — drop only the cheap 68.4%) | ≤ **0.0342** | **0.5295** | **15.5× over** |
| cost ÷ credit | < 1.0 | **44.1×** | — |

And that is before the pose term, which at this rung is **4.29× the seg damage** (ΔS_pose 2.5682 vs
ΔS_seg 0.5992). Including it, the total distortion cost is **112.2× the entire remaining gap**.

**`hg1` and `hr3` stay closed, now for a MEASURED reason.** Their unique-home residuals bought
something the scorer genuinely charges for. `et1` (BSP topology) and `ws1` (event streams) are
**not adjudicated here at all** — their bytes are not corrections, so this curve does not reach them,
and §7.1 is the reason rather than an excuse.

---

## 8. Prediction, adjudicated

**Pre-registered prediction:** *"the tolerance is TINY: the curve rises steeply from k=1, and by
k=1,000 `d_seg` has already exceeded the whole 42,382 B demand's worth of distortion."*
**Pre-registered falsifier:** *"`d_seg` stays inside the break-even band out to k ≥ 10⁴"* — which
would have reopened four representation families at once.

**FALSIFIER DID NOT FIRE.** At k = 10⁴ the measured τ is **0.9043** against a break-even bar of
**0.17333** — outside the band by **5.2×**. Every rung is outside its bar (5.2× / 4.5× / 8.3× /
222×). The four families stay closed.

**The prediction was RIGHT on magnitude and RIGHT on k, and wrong only on the axis.** The k = 10³
rung settles this exactly, and it is more favourable to the prediction than the k ≥ 10⁴ rungs alone
suggested:

| claim | verdict | measured |
|---|---|---|
| tolerance is tiny | **CONFIRMED, and understated** | τ = **1.0369** at k = 10³ — *amplification*, worse than 1:1 |
| the curve rises steeply from k = 1 | **CONFIRMED** | the steepest transfer IS at the small end; τ decreases monotonically thereafter |
| **the gap's worth of distortion is exceeded by k = 1,000** | **CONFIRMED — it crosses at exactly k = 1,000** | ΔS_total = **0.029104 = 1.031× the 0.028220 gap** |
| …carried by `d_seg` | **WRONG — carried by POSE** | at k = 10³, ΔS_pose **0.028225** vs ΔS_seg **0.000879** — pose is **32.1×** seg, and pose alone equals the gap to three digits |

**The k = 10³ rung is the cleanest single number in this memo**: moving **0.00085%** of the field —
1,000 of 117,964,800 positions — costs **1.031× the entire remaining gap to 0.12**, while releasing
221.8 B. `d_seg` alone would not have crossed until k ≈ 4.2e4, ~40× later; the prediction landed on
the right k because total distortion is what crosses, and pose supplies 97% of it.

**Adjudicated answer to the question this arm was chartered to settle: the exact-reproduction
constraint was an HONEST price, not a self-imposed one.** The scorer's equivalence class around
dx2's token field is not merely small — at the small end it is barely larger than a single point,
with ~90% of every token change surviving to the argmax. Four representations were priced against
exactness; that exactness was real.

---

## 9. NOT CLAIMED

- **No score claim, no pointer move.** Every row is `macOS-CPU advisory` on `PYAV_YUV420_TO_RGB` GT
  lineage. Comparing these `d_seg` values to dx2's contest-CUDA 0.00020139 is forbidden (`#1034`);
  the only legitimate comparison is against the matched base measured on the same axis.
- **Not a lever.** No addressing scheme is proposed, priced, or implied. The credit column is an
  addressing-free upper bound that no real representation gets.
- **Not a verdict on `et1` or `ws1`.** Their bytes are topology and event streams, not corrections.
- **Not hg1's number.** §6.3 states the bound this supplies and the number it does not.
- **CPU seg deltas are upper bounds** on the CUDA axis; there is no measured CPU→CUDA seg transfer
  law.
- **The τ-inversion (§6) is measured at one k on one object.** Its spatial statement — seg-support on
  the boundary, pose-support in the interior — is supported by a 2×2 at k = 10⁵ plus the measured
  boundary occupancies. It is **not** a fitted law over k, and the interior/boundary split is
  `[0,1,2,3,≥4]` 4-neighbour buckets, not a continuous distance.
- **No claim that a better perturbation design would pay.** §4.1 bounds the whole family: the credit
  ceiling is 113,776 B and k = 10⁶ already reaches 91.55% of it, so no larger-k or better-targeted
  variant of *this* mechanism has room to invert a 40× deficit.
- **`et1` and `ws1` are not adjudicated** (§7.4).
- **The d = 1–3 annulus (5.04% of the field) is NOT classified** by this instrument (§6.0.1). "Every
  position is expensive" is measured for 94.97% of the field, not for all of it.
- **§6.3 is OPEN, not answered.** No claim is made about pose-null interior movement.

### Instrument defects found and fixed during this arm

Recorded because a silent instrument defect is the failure this campaign keeps paying for:

- **The drainer's in-flight counter under-counted its own rows.** It matched `rows/<label>/work`,
  but rows it fires live at `rows/<label>/attempt_NNNN/work` (the canonical firer refuses a
  non-empty attempt dir, so every retry mints a fresh one). It therefore reported **zero live while
  three were rendering** — which reads as free capacity. Fixed, tested, and the reason written into
  the code so it cannot return.
- **What saved it is worth recording too: two INDEPENDENT ceilings held while the counter was
  wrong** — the drainer's own `max_own_inflight` cap and the host governor's SUM-over-RAM admission
  gate. Neither depended on the broken count, so no row was over-committed. Defence in depth doing
  exactly its job.

---

## Reproducing and extending this table

The curve is regenerable from retained receipts with one command; later rungs fold in by adding
`--row` pairs. It **refuses** any row whose archive is not the shipped 180,368 B, so a mismatched
row cannot silently enter the table:

```bash
TV=/Volumes/APDataStore/pact/ddm_tv1_tolerance_curve
.venv/bin/python experiments/ddm_tv1_curve_table.py \
  --base-eval $TV/rows/k0_control/contest_auth_eval.json \
  --row $TV/retained/manifests/<label>.json:<attempt_dir>/contest_auth_eval.json \
  --out <out.json>
```

Independent re-verification of any field (recount `k` from the bytes, re-price against `tb2`):

```bash
.venv/bin/python experiments/ddm_tv2_verify_fields.py \
  --base-field $TV/retained/base_inputs/dx2_tokens_n600.u8 \
  --perturbed-dir $TV/fields/perturbed --manifest-dir $TV/retained/manifests \
  --cost-field <tb2 position_rc64_frequency_cost_bits.f64le.bin> \
  --labels <label> --json-out <out.json>
```

Retained payloads (3,539,162,029 B, referenced not duplicated, every sha independently re-verified):
`/Volumes/APDataStore/pact/ddm_tv2_tolerance_curve/retained/payload_retention_manifest.json`.

---

STORES CONSULTED: `.omx/research/ddm_tv1_evaluator_tolerance_curve_20260824.md` (predecessor instrument, now committed)
· `ddm_wq1_what_was_never_asked_20260824.md` (D3) · `ddm_tri1_triple_composition_and_pair_closure_20260824.md`
(SPEC B) · `ddm_tb2_token_bit_attribution_20260823.md` (Gini 0.9951; cost field) ·
`ddm_tx1_toolbox_crosswalk_20260819.md` §0 (exchange rate) ·
`ddm_mst1_manufactured_stage_split_20260822.md` (matched base) · `hg1` / `hr3` / `et1` / `ws1` memos
and `experiments/ddm_hg1_heterogeneous_analytic_generator_gate.py` (residual mechanism at source) ·
retained `mst1` base receipt and `tb2` retained fields on `/Volumes/APDataStore`.
