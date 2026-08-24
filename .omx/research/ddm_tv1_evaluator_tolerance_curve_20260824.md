# ddm_tv1 — the evaluator's tolerance to token-field movement, measured

**Type: MEASURED (local CPU advisory rows) over MEASURED retained fields.**
`score_claim: false` · `promotion_eligible: false` · `pointer_moved: false` ·
`evidence_grade: macOS-CPU advisory` · GT lineage `PYAV_YUV420_TO_RGB` (NOT the authority
`DALI_NVDEC`).

`verdict_scope`: **the dx2 object** — archive sha
`976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`, 180,368 B — its shipped
F26 receiver, and its 600×384×512 five-class token field. A different object has a different
field, a different coder, and a different curve.

---

## 0. What this measures, and what it does NOT

This arm asks one question: **how much can dx2's 117,964,800-position token field move before
the frozen scorer notices?** `ddm_vf1` left that denominator at literally **0 / 117,964,800**;
four alternative rate representations (`hg1`, `hr3`, `et1`, `ws1`) were each priced *plus the
cost of reproducing that field exactly*, and no arm ever checked whether the exactness was
required.

**It measures TOLERANCE ALONE. It is not a lever and it cannot become one.** Nothing here names
which positions moved and nothing here codes anything. A representation that wanted to exploit
tolerance would have to transmit the changed set, and this campaign has measured three times over
that naming a subset costs more than the subset holds — `mf1`'s best perfectly-addressed repair
still cost **+35,969 B** of address payload; every lossy `ld1` rung enlarged the archive. The two
have never been measured apart, so a small tolerance and a large address tax have been
indistinguishable. This arm isolates the numerator so they never are again.

Read every "credit" number below as an **ADDRESSING-FREE UPPER BOUND**: it assumes the changed
set is free to name, which no real representation gets.

---

## 1. The object, measured

The token field is `600 × 384 × 512` uint8 over **5 classes** — `NUM_CLASSES = 5`,
`EVAL_H, EVAL_W = 384, 512`, `N = 600` in the shipped renderer. That is **one token position per
SegNet argmax pixel**: the scorer's own disagreement is counted over the same 117,964,800 cells.
The base's `d_seg = 0.00034740` is therefore exactly **40,981 disagreeing argmax pixels**.

The class histogram reproduces the canonical comma10k areas to four decimals, which is what tells
us the tokens *are* the semantic partition and not some internal code:

| token | class | positions | share |
|---:|---|---:|---:|
| 0 | Road | 27,406,888 | 23.2331% |
| 1 | Lane markings | 691,095 | 0.5858% |
| 2 | Undrivable | 58,413,222 | 49.5175% |
| 3 | Movable | 1,460,458 | 1.2380% |
| 4 | MyCar | 29,993,137 | 25.4255% |

Bit mass, from `tb2`'s retained per-position cost field
(`position_rc64_frequency_cost_bits.f64le.bin`, 943,718,400 B):

- total **910,209.28 bits = 113,776.16 B**; mean **0.00771594 bits/position**; **median 1.008e-8
  bits** — the coder is essentially certain at the median position.
- The model's own uncertainty mass, `Σ(1 − p_realized)`, is **339,034.1 positions = 0.2874% of the
  field**. That is the entire in-manifold wander available to this representation.

**Boundary geometry (MEASURED, this arm).** Using the receiver's own 4-neighbour
`_boundary_buckets` definition:

| distance to a class boundary | positions | % of field | bits | % of bits | uncertainty mass |
|---:|---:|---:|---:|---:|---:|
| 0 | 2,555,705 | 2.1665% | 860,441.9 | **94.5323%** | **90.7356%** |
| 1 | 2,133,074 | 1.8082% | 19,915.4 | 2.1880% | 3.4926% |
| 2 | 1,965,747 | 1.6664% | 7,789.1 | 0.8557% | 1.4608% |
| 3 | 1,842,840 | 1.5622% | 5,178.2 | 0.5689% | 0.9721% |
| 4 (interior) | 109,467,434 | 92.7967% | 16,884.7 | 1.8550% | 3.3389% |

**2.17% of the field carries 94.53% of the bits.** Every rate question about this object is a
question about that 2.17%.

---

## 2. The instrument, and its controls

### 2.1 Injection point

The shipped receiver decodes the payload to a token field, writes it to its own resumable
checkpoint (`tokens_cpu_stage_complete.u8` + receipt), and then renders. `ddm_tv1` publishes a
counterfactual field into the receiver's own token cache
(`F26_ADVISORY_DECODE_CACHE_ROOT`, the designed knob) so the receiver loads it and runs the
**entire unmodified downstream**: semantic renderer → RGB → the real R/uint8 path → 3,662,409,600 B
of raw frames → `upstream/evaluate.py` at n600.

What is bypassed is only the arithmetic decode — a bijection from a payload to a field. It *cannot*
be run on a counterfactual field, because no payload for it exists. That is the point: the field is
being priced independently of any coder.

### 2.2 The conditional is the shipped coder's own, proven

The reference arm resamples from the distribution the field's bits are actually priced against.
That distribution is computed inside `residual_archive.decode_production_tokens` and thrown away —
only two digests of it survive. `experiments/ddm_tv1_capture_coding_conditionals.py` replays the
identical model/corrector loop, feeding the already-decoded symbols, and retains the corrector's
coding row per position.

**The replay is proven, not asserted.** It reproduced BOTH shipped digests bit-exactly:

| digest | shipped receipt | tv1 replay |
|---|---|---|
| `corrected_quantized_logit_sha256` | `8269fe1aad0316…` | **identical** |
| `corrected_cdf_input_sha256` | `370a5e2a85ccbb…` | **identical** |

Cross-check against a second, independently produced field: `−log2(coding_row[realized])` vs
`tb2`'s retained cost field over 200,000 random positions — **max abs diff 4.179e-06 bits, corr
1.0000000000**, and **0 of 200,000** positions had all four alternatives underflow float32. Also,
`tb2`'s `decoded_tokens_instrumented.u8` and `mst1`'s checkpoint field share SHA-256
`cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` — two arms, one field.

### 2.2b The k = 0 positive control — PASSED byte-for-byte

Publishing dx2's **unmodified** field through the same injection path and rendering it produced
`0.raw` with SHA-256
`7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7` — **identical to the matched
base's `0.raw` across all 3,662,409,600 bytes.**

That single control settles four things at once, by measurement rather than assumption:

1. the token-cache injection reproduces the shipped decode exactly at `k = 0`;
2. the render path is deterministic;
3. the matched base's `d_seg = 0.00034740` / `d_pose = 0.00014701` **transfer exactly** to my rows —
   identical scorer input bytes cannot produce different scorer output;
4. **mst1's instrumentation is decode- and render-neutral.** The base was produced by mst1's
   INSTRUMENTED tree; these rows use the SHIPPED dx2 receiver. Byte-identity across two different
   trees proves the instrument did not move between base and candidate — the `#1034`
   cross-instrument genus is excluded by evidence, not by construction.

Point 4 also had to be settled the other way round: **the instrumented tree is structurally
incapable of this measurement.** `cpr1/ddm_mst1_manufactured_stage_split.py:243` raises
`Mst1Error("decoded semantic-token field differs from MS9/TO2")` unless the token field hashes to
`cc10a7b0…` — dx2's exact field — so it refuses every counterfactual by construction; a second gate
at `:456` repeats the check, and `_ensure_vertigo_store` requires the capture store to live on
Vertigo with `MIN_FREE_BYTES = 28 GiB` free (Vertigo currently has 8.4 GiB). The shipped receiver
was not a convenience; it was the only tree that can run this arm. A source diff confirms the two
render paths are the same arithmetic in the same order — the instrumented copy only names the
intermediates and adds `capture.record(...)`; all other `cpr1/` files and the whole `runtime/` tree
are byte-identical.

### 2.3 The arms — the reassignment rule IS the mechanism

With Gini 0.9951 on the bit mass and a near-certain median position, "reassign a random position"
is a *family*, and the wrong member measures receiver robustness to garbage rather than the size of
the scorer's equivalence cell. WHERE a position is drawn and WHAT value it takes are two
independent choices, so they are run as a 2×2 with the uniform corner as control — never
confounded on a diagonal:

| arm | positions drawn by | value drawn from |
|---|---|---|
| `cond_cond` **(reference form)** | model uncertainty `1 − p` | coder conditional, value ≠ current |
| `cond_unif` | model uncertainty `1 − p` | uniform over the 4 alternatives |
| `unif_cond` | uniform over the field | coder conditional, value ≠ current |
| `unif_unif` **(maximal control)** | uniform over the field | uniform over the 4 alternatives |
| `unif_marg` | uniform over the field | global class marginal, value ≠ current |

Every arm produces **exactly `k` changed positions** (verified per row), so the ladder is
comparable across arms. Positions are drawn without replacement — weighted arms by Gumbel top-k
(exact Plackett–Luce). Seeds are derived as `sha256("ddm_tv1|<arm>|<k>|20260824")[:8]`, recorded
per row, so the same inputs reproduce the same positions.

**Never a contiguous prefix.** Prefix bias on this object is measured and axis-dependent — pose
2.5–4.2× harder on a prefix, seg ≈0.96× — and this is a pose-bearing measurement. Every built
field touches **all 600 frames**.

### 2.4 The position rule alone changes the credit 220×

Measured at `k = 100,000`, before any scorer ran:

| arm | on a class boundary | bits held in the shipped stream | addressing-free credit |
|---|---:|---:|---:|
| `cond_*` | **90.2%** | 169,264 bits = 21,158 B | 1.4088e-02 S |
| `unif_*` | 2.1% | ~740 bits = ~92.5 B | ~6.2e-05 S |

**A uniform-position perturbation cannot pay under any tolerance**, because the positions it moves
hold no bits: moving 100,000 uniformly-chosen tokens releases 92.5 bytes. That is a derived
constraint on the whole question, and it is why the conditional arm is the headline rather than a
refinement.

---

## 3. The bars, derived BEFORE any row

Exchange rate **6.658590e-07 S/B**, CITED from `ddm_tx1_toolbox_crosswalk_20260819.md` §0, never
re-derived. Seg marginal is exactly `d(S)/d(d_seg) = 100`.

**1:1-transfer prediction.** If one changed token buys one changed argmax pixel,
`Δd_seg = k / 117,964,800 = 8.477e-9 · k`. The measured quantity that decides everything is the
**transfer factor** `τ = Δd_seg_measured / (k/N)`.

Three break-even bars, each `Δd_seg* = 6.658590e-07 × B(k) / 100`:

| bar | what it prices | `Δd_seg*` | required τ for tolerance to pay |
|---|---|---|---:|
| **B1 mean-price** | k random positions at 0.00771594 b/pos | `6.42e-12 · k` | **τ ≤ 0.00076** |
| **B2 top-k oracle** | the k most expensive positions, address given away | `2.58e-8` at k=1 … `7.21e-4` at k=1e6 | **τ ≤ 0.0851** |
| **B3 hg1 marginal** | `hg1`'s own measured drop elasticity, 0.348 b/correction | `2.896e-10 · k` | **τ ≤ 0.03417** |

B2 is an oracle bound that ignores addressing entirely and is therefore unreachable; it is included
because it is the most generous number the evidence permits. **B3 is the bar that matters for the
four dead representations**, and it is derived in §5 from `hg1`'s own receipts.

Per-row addressing-free credit, measured from each realized position set:

| arm | k | bits held | bytes | credit (S) | % of the 113,776 B stream |
|---|---:|---:|---:|---:|---:|
| `cond_cond` | 1,000 | 1,774.6 | 221.8 | 1.4770e-04 | 0.195% |
| `cond_cond` | 10,000 | 17,654.0 | 2,206.8 | 1.4694e-03 | 1.940% |
| `cond_cond` | 100,000 | 169,265.0 | 21,158.1 | 1.4088e-02 | 18.596% |
| `cond_cond` | 1,000,000 | 833,333.9 | 104,166.7 | 6.9360e-02 | **91.556%** |

The `k = 10⁶` rung is decisive on its own: if the scorer could not see it, the released credit
(0.06936 S) would be **2.46× the entire remaining gap to 0.12** (0.028220 S).

---

## 4. RESULTS

### 4.0 What a row costs, and what the instrument can resolve

**Cost per row (MEASURED).** Setup + compliance + archive read + model load + token-cache load is
a **fixed ~430 s** before any rendering (the three instrumented rows that died in `render_video`
did so at 431.6 / 440.1 / 447.3 s). Render is ~25–30 min; `evaluate.py` at n600 is **462.5 s** on
the matched base. So a row is **≈35–40 min wall clock** and writes 3,662,409,600 B of raw frames.
Measured peak resident set of the inflate process is **≈5 GiB**, not the GB-scale-per-row figure a
projection of 10 GiB assumed.

**Resolution floor.** The report carries 8 decimals, so `d_seg` resolves to **1e-8 = 1.18 argmax
pixels**. Against the 1:1 prediction `8.477e-9·k`, the smallest transfer factor each rung can
resolve is:

| k | 1:1 `Δd_seg` | in ulp of 1e-8 | smallest resolvable τ |
|---:|---:|---:|---:|
| 1,000 | 8.477e-06 | 848 | 1.2e-03 |
| 10,000 | 8.477e-05 | 8,477 | 1.2e-04 |
| 100,000 | 8.477e-04 | 84,771 | 1.2e-05 |
| 1,000,000 | 8.477e-03 | 847,714 | 1.2e-06 |

All three decision thresholds from §5.2 (0.2114 / 0.0342 / 0.00076) are resolvable at every rung
from `k = 10⁴` up. `k = 1` and `k = 10` are **below the instrument floor** (0.85 and 8.5 ulp) and
were therefore not run — a declared scope reduction, not an omission.

### 4.1 The curve

---

## 5. The linkage to the four dead representations — checked, not asserted

### 5.1 What the "unique-home residual" actually is (source-read, not inferred from the name)

The name suggested an addressing cost. It is **both**, and the split is measured. The byte
mechanism is a sparse list of `(address, value)` records —
`experiments/ddm_hg1_heterogeneous_analytic_generator_gate.py:466-470` emits a ULEB zigzag address
delta followed by one target-class byte per correction. `hg1`'s memo agrees at `:100`: *"every
correction has one address and one target class, with canonical order enforced by the receiver."*

| arm | mechanism | verdict |
|---|---|---|
| `hg1` | 1,334,939 `(address, value)` corrections; raw 2,871,598 B = **46.49% value / 53.51% address** (mean 1.151 B/address); coded 359,280 B = 2.153 bits/correction joint | **MIXED** |
| `hr3` | calls `hg1.encode_residual` directly (`:37, :1493`); 362,473 B = 3,223 B INR + 359,176 B residual. The learned INR removed **104 B — 0.03%** of hg1's residual | **MIXED**, identical mechanism |
| `et1` | 482,678 B is *implicit generator topology* (the BSP tree), not a correction residual at all | **TOPOLOGY-BOUND** |
| `ws1` | per-stratum lifetime-event + innovation streams; no correction residual exists | **NEITHER** |

Post-coder the value and address halves are not separable, so the raw split is the only measured
decomposition. **Tolerance does not shave a value field while the address stays — it deletes whole
records.** So the relevant elasticity is measured, and `hg1` measured it:

| bracket | corrections | coded B | bits/correction |
|---|---:|---:|---:|
| exact | 1,334,939 | 359,280 | 2.153 |
| BL1+MS9 protected | 422,024 | 319,518 | 6.057 |
| **dropped** | **912,915 (68.4%)** | **saved 39,762 (11.1%)** | **0.3484** |

**Dropping 68.4% of all corrections bought 11.1% of the bytes.** That is the
`perfect-localization-is-the-tax` law again, from inside hg1's own table.

### 5.2 The three transfer thresholds this arm has to resolve

`hg1`'s container was 460,408 B against a 137,986 B cap — it must shed ≥322,422 B. Its residual is
359,280 B, so **deleting the residual entirely would put it at 101,128 B, 36,858 B UNDER the cap.**
The whole question is therefore one number: what does the scorer charge for 1,334,939 wrong
positions? Converting each byte bracket at 6.658590e-07 S/B against the 1:1 prediction
`Δd_seg = k/117,964,800`:

| what tolerance would buy | bytes released | break-even `Δd_seg` | 1:1 `Δd_seg` | **required transfer τ** |
|---|---:|---:|---:|---:|
| delete hg1's **entire** residual (k = 1,334,939) | 359,280 | 2.3922e-03 | 1.1317e-02 | **τ ≤ 0.2114** |
| drop only hg1's **cheap** 68.4% (marginal 0.348 b/corr) | 39,762 | 2.6474e-04 | 7.7389e-03 | **τ ≤ 0.0342** |
| leave k **uniformly random** positions uncoded | 0.000964·k | 6.42e-12·k | 8.477e-9·k | **τ ≤ 0.00076** |

These are three different questions and they have thresholds 278× apart. The first is the one that
decides four dead arms, and **τ ≤ 0.2114 is not an implausible number a priori** — SegNet reads
regions through a stride-2 stem, so isolated flips could well be absorbed.

### 5.3 The honest limit of this linkage

My perturbation is drawn from the *coder's* uncertainty over the *incumbent* field. `hg1`'s
1,334,939 differing positions are where an *analytic generator* disagrees with the incumbent — a
structured, generator-specific set (45.67% at distance 0 from the **generated** boundary, 86.09%
within 8 px, 98.806% of its incumbent-model cost inside BL1's top 1%). Both sets are
boundary-concentrated and of the same cardinality, so this curve **bounds the order of magnitude of
hg1's answer; it does not supply hg1's number.** Only hg1's own field through the real receiver
does that, and everything needed is retained (residual mask 14,745,600 B, SHA `81132b00…`; the
encoder already accepts a `protected` argument).

**None of the four arms ever fired a scorer.** `hg1`: *"No scorer was fired"* (`:16`). `ws1`:
*"This arm did not run a scorer"* (`:7`). `et1` named evaluator-cell equivalence as a LIVE
HYPOTHESIS (`:221-222`). `hr3` gated a fire on a residual ≤36,858 B that never arrived (`:205`).
All four assumed exact reproduction and priced it.

---

## 6. Prediction, adjudicated

*(see §6)*

---
