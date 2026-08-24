# ddm_gb1 — decode-scan conditioning, realized: the free win holds at **63 B**, and the two things that would have killed it were both in the fire order

**Date:** 2026-08-24 · **Arm:** `ddm_gb1` · **Pointer:** UNMOVED · **No Modal job, no Metal job, no
scorer, no training, no frame rendered. $0.**
**Axis:** `[macOS-CPU advisory / scorer-free EXACT byte measurement]`.
`score_claim=false` · `promotable=false` · no archive promoted.
**verdict_scope:** see §9.

---

## 0. Result first

**1. `ddm_mi1`'s one positive re-derives, independently, slightly smaller.** Fitting my own
8-cell log-odds offset over the within-tile decode index on my own re-derived base gives
**63.09 B held-out** (3 seeds, spread 0.255 B = 0.40%) against `mi1`'s **64.20 B** — **1.7%
lower**. Every one of `mi1`'s five base constants reproduces to the last digit (§3). The
mechanism reproduces too, and the 8-bin table sharpens it: **the model's error is
sign-monotone along the scan — it over-predicts flips in the first half and under-predicts
them in the second — while the flip rate itself is V-shaped, which `mi1`'s 4-cell quadrant
proxy could not see** (§4).

**2. I ran a null control `mi1` did not, and it costs the headline 1.6 B.** Re-labelling every
position with a cell drawn independently of position still returns **+1.59 B held-out** from
8 cells over 50,009,121 live positions. So an 8-cell offset table manufactures ~1.6 B from
nothing, and the noise-floor-corrected signal is **≈61.5 B**, not 63.1 B. Signal/null = **39.6×**,
so this is mechanism and not split noise — but the floor travels with the number from here.

**3. The charter's "price with a REAL re-encode, never static `−log₂p`" is satisfied, and the
gap it was guarding against is 0.09 B.** `ddm_ds1`/`ddm_fs2`'s 14.59× and 0.93×/0.09× mispricing
were measured for **token-field edits**, which change which SYMBOL is transmitted. This family
changes no symbol. Priced in the coder's own 31-bit integer frequencies instead of float
`−log₂p`, `groupbin8` returns **63.00 B** against the ledger's **63.09 B** — a **0.14%**
difference (§5). The whole-stream chain is measured end to end: ledger **113,776.179 B** →
integer-frequency cost **113,776.160 B** → physical stream **113,777 B**.

**4. The real transfer loss is elsewhere, it is 200× larger, and it is measured.** `ddm_fx5`'s
race predicted **−86.58 B** of code length for the 13→19 member move and the physical re-encode
on this body returned **−70 B**: a realization ratio of **0.808**. That, not `fs2`, is the
discount a code-length number owes.

**5. PHYSICAL ROWS: PENDING** — three full-n600 re-encodes through the shipped RC64 encoder
along the receiver's own decode trajectory are in flight (control + two members). §5 carries
them. Until they land, **nothing here is a rate claim**, and the 63 B is a model ledger.

**6. Five corrections, and two of them would have wasted the build.** `mi1`'s fire order names
the wrong incumbent (13-member D1; the shipped body is the **19-member fx5 E1**) and the wrong
harness (`fx2`'s race replays a **different token field** — digest `9ba2e52b…` against the DX2
body's `cc10a7b0…`, MEASURED). Its decode-wall premise is stale by **8×** in the safe direction,
which changes the *reason* for its build order but not the order. My own charter's "extend the
rr4 context by a factor of 8" would have diluted the incumbent's counts 8× and is refused with
its arithmetic. And nobody in this lineage has said that shipping this needs a **C port**. §1.

**Scale, stated honestly:** 63 B is **0.15%** of the 42,381.16 B demand and **0.055%** of the
token stream. It is worth banking because it is free — zero stored bytes, zero seg, zero pose,
by construction and by digest — not because it is large.

---

## 1. Corrections — five, and two of them would have cost the build

**(a) The decode wall is not 160.2 s. It is 1,294 s, and it was MEASURED on contest-CUDA.**
`mi1` §7 justifies building `groupbin8` first because "the decode wall is the binding constraint
(`fx1`: 160.2 s of margin, and members are the expensive dial)". That 160.2 s is `ddm_fx1`'s
own `[macOS-CPU advisory]` replay of an **11-member, rr4-body** candidate, and the fx1 memo
disclaims it in place. Re-read from the Modal result blobs myself rather than from a relay
(`/Volumes/APDataStore/pact/{ddm_rc2/t4_row_r2,ddm_fx5/t4_row_r1,ddm_dx2/r7/t4_row_r1}/MODAL_REMOTE_RESULT.json`),
all `[contest-CUDA T4 n600]`:

| body | members | inflate s | decode+render s | charged s | archive B |
|---|---:|---:|---:|---:|---:|
| `rc2` r2 | 13 | 458.753 | 454.596 | 498.476 | 180,456 |
| `fx5` r1 | 19 | 558.626 | 553.280 | 610.529 | 180,386 |
| **`dx2` r7 — the shipped body** | 19 + CABAC | **505.546** | **500.837** | 559.799 | **180,368** |

**Margin against the 1,800 s budget: 1,294.45 s on inflate, 1,240.20 s charged. The body uses
28% of the wall.** One member costs **16.4 s** (13→19 is +98.68 s of decode+render over six
members), and T4 run-to-run variance on decode-identical member counts is **53.08 s** (fx5
558.626 vs dx2 505.546) — **3.2× a single member's cost, so a one-member delta is not resolvable
in a single row.** *(A relayed figure of "+60.15 s over six members" is an inflate-vs-charged
comparison across two different columns; I recomputed rather than inherit it, and the correct
figure is +98.68 s / 16.4 s per member.)*

**`mi1`'s build order is right and its reason is wrong.** Build `groupbin8` first because it is
the best-conditioned rung — max|β| 0.11 with no cell at the Newton clip, the most seed-stable
row on the ladder — not because members are expensive. A future arm inheriting the wall reason
would refuse rungs it can afford.

**And there IS a live wall, on an axis no memo in this lineage mentions.** `ddm_rc2`'s
contest-**CPU** row measured **2,850.78 s** for the 13-member body
(`/Volumes/APDataStore/pact/ddm_rc2/cpu_row_r1/MODAL_REMOTE_RESULT.json`) — **1,050 s over the
1,800 s budget, today, before any new member.** If evaluation ever lands on CPU the body fails
on wall-clock alone. That is where the margin question is actually live.

**(b) `mi1`'s fire order names the wrong incumbent.** It says "race it against the shipped
13-member D1 build". The shipped build is the **19-member fx5 E1**: `FX5_BUILD_MANIFEST.json`
carries `member_count_candidate: 19`, `mixer_context: cls_boundary_agree_homog_ubin8`, and the
three patched digests — including `fx2_model_axis_corrector.py` `77e81ac8…`, **which `mi1` itself
pinned in its own §2 source table.** Racing a new member against D1 measures its marginal value
against a weaker incumbent; six of the members that would have absorbed it are missing. That is
anti-conservative in exactly the direction that manufactures a win.

**(c) `mi1`'s fire order names a harness that replays a different body — MEASURED by digest.**
"under `fx2`'s own harness" points at `/Volumes/APDataStore/pact/ddm_fx2/race.py`, whose
`ReplayAssets.tokens_u8` is
`ddm_hv1_base_advisory_n600_cpu/.../tokens_cpu_stage_complete.u8`, sha256 **`9ba2e52b3096…`**.
The DX2 body's decoded token field is sha256 **`cc10a7b09353c0af…`** — the digest `mi1`'s own
custody table pins, and the one `ddm_bl1` refuses without. They are different fields, and the
harness's `LIVE_BYTES` constant (110,511.28 B) is the *other* body's code length. **A member
raced there is priced on a body we do not ship.** The correct instrument on the DX2 body is
`experiments/ddm_jg2_tail_reencode.py`, which drives the shipped Python HPAC fallback, the
shipped `FreeCorrector` and the shipped RC64 encoder along the receiver's own trajectory. That
is what this arm used.

**(b) and (c) are one class, and it has a cheap cure.** Both are the same failure: *a fire order
named an incumbent and a harness without checking that either is the shipped object.* Neither
is careless — `mi1` correctly pinned the fx2 digest `77e81ac8…` in its own source table and then
still wrote "13-member D1" three sections later, because the member count lives in a different
file from the digest. **The cure is two greps, not more care: verify the incumbent against the
build manifest (`FX5_BUILD_MANIFEST.json::member_count_candidate`) and the harness against its
asset digests, before racing.** A race against the wrong incumbent and a race on the wrong body
both inflate a candidate, and both are invisible in the result.

**(d) My charter's "extend the shipped rr4 context by a factor of 8" is refused, with
arithmetic.** Re-indexing the 51,200-cell rr4 head by `groupbin8` gives 409,600 cells and
**divides every incumbent count by 8**. The incumbent runs at only **0.0516% of observations
below `MIN_COUNT`** (`fx1_logistic_mixer_corrector.family_specs` docstring, MEASURED); an 8×
split spends precisely the density that makes it work, and it degrades the incumbent whether or
not the new axis carries anything. A mixer **member** is additive instead: it enters at weight
0 (`initial = [1.0 if f.name == "shipped_joint" else 0.0 ...]`,
`fx1_logistic_mixer_corrector.py:521`) and its cold cells return a multiplier of **exactly 1.0**
(`MixerFamily.multiplier`, gated on `counts >= MIN_COUNT`), so it **nests the incumbent
bit-for-bit** and can only be admitted by evidence. The charter's "symmetric by construction"
requirement is satisfied *more* strongly this way, not less: encoder and receiver instantiate
the identical object from the same frozen `SHIPPED_CONFIG`, so symmetry is structural rather
than maintained. `mi1` said "member"; `mi1` was right.

**(e) `mi1`'s three "× the naming bar" figures use three different denominators, and one of
them is a units slip.** At `ddm_tx1` §0's exchange rate `25/37,545,489 = 6.658590e-07 S/B`
(cited, re-derived here to 7 significant figures):

| item | bytes | ΔS | vs the 1e-5 bar | vs fx5's 3.5e-6 admit threshold | `mi1` said |
|---|---:|---:|---:|---:|---:|
| `mi1` `patch192` | 211.13 | 1.405828e-04 | **14.06×** | 40.17× | 14.1× ✔ (1e-5) |
| `ma1` adopted | 104.58 | 6.963553e-05 | 6.96× | **19.90×** | 20.0× ✔ (3.5e-6) |
| `mi1` `groupbin8` | 64.20 | 4.274814e-05 | 4.27× | 12.21× | **6.4× ✘** |
| **this arm** `groupbin8` | **63.09** | **4.200638e-05** | **4.20×** | **12.00×** | — |

The first two are right against *different* bars. The third matches neither — and `64.20 / 10 =
6.42`, so it reads as bytes-divided-by-ten rather than a ratio to anything. **`groupbin8` is
4.20× the 1e-5 bar, not 6.4×** — the rung is real and it is **1.5× smaller than advertised**
relative to its own bar. Pick one denominator and state it; this table states both.

**(f) Shipping this needs a C port, and no memo in this lineage says so.** The shipped decode
path is `native-hpac` with the C corrector, and `runtime/f26_corrector_native.c` compiles the
member list in as constants — `N_FAMILIES`, `FAMILY_RULE`, `FAMILY_SIZE`, `FAMILY_COUNT_LIMIT`,
and a `case` in `family_rule_index`. `runtime/native_free_corrector.py::EXPECTED_SHIPPED_CONFIG`
**REFUSES** on any drift, including an added member; I verified the refusal fires on my patched
config (20 live members against 19 expected). A refusal is correct — a silent mismatch
desynchronises the decoder, which is `ddm_rr2`'s S = 27.83 — but it means a Python-only member
falls back to the Python corrector, which `ddm_rr8` measured at **1,419.9 s** inflate against the
native **464.6 s**. That spends **955 s of margin to save ~50 B**. **The port is mechanical and
precedented:** `ddm_fx5` did exactly this for six members in **3 files**
(`f26_corrector_native.c`, `fx2_model_axis_corrector.py`, `native_free_corrector.py`), and
`groupbin8` adds one `case` of pure int64 index arithmetic plus one row in each of three tables.
Named as a sized blocker, not hidden.

---

## 2. The mechanism, and why it is causal by construction

`ddm_mi1`'s finding, in one sentence: **the shipped model's confidence does not fully account
for how much of its own causal context has been decoded yet.**

The shipped group plan is `grid = columns + HPAC_DELTA * rows` over a `HPAC_PATCH = 64` tile
with `HPAC_DELTA = 2`, enumerated `for group in range((1 + HPAC_DELTA) * HPAC_PATCH -
HPAC_DELTA)` = `range(190)` (`cpr1/inflate.py:33-34, 275-287`, read at source). The decoder walks
those 190 masks in increasing `group`, so

```
g(x, y) = (x mod 64) + 2 * (y mod 64)        in 0 .. 189
groupbin8 = (g * 8) // 190                   in 0 .. 7
```

is **the index of the decode step currently being taken**. It is not a property of the symbol;
it is a property of the position, and the decoder selects the position before it decodes the
symbol there. **There is no ordering hazard to check — only an identity to verify**, and
`stage_verify` verifies it against the shipped retained `group_index.u8` rather than trusting
the transcription: **0 mismatches over all 196,608 plane positions, max group 189.** MEASURED.

The feature is absent from every conditioning site in the shipped stack because the network
tiles the frame at 64 px and gives **every tile the same coordinate grid** — so which tile, and
where in the scan, cannot reach it except through content.

**And it is the cheapest feature on the board, which the C port should exploit.** MEASURED: in
**0 of 190** groups does `groupbin8` vary across the group's positions — it is constant by
construction, because a group *is* the level set of `g`. So the decoder computes it **once per
group from its own loop variable** (`groupbin8 = group_index * 8 / 190`), not once per position.
Every other member's feature (`spatial4`, `homog`, `ubin`, `run`) is per-position. My Python
patch derives it from `flat` per position, which is correct and wasteful; the C port should
hoist it to a scalar and will then cost strictly less decode time than any existing member.

---

## 3. Re-derivation — the base, in my own code, before any analysis

Nothing below is inherited. Every prior-arm constant is recomputed from the retained fields and
compared; a disagreement fails the stage closed.

| quantity | this arm (MEASURED) | prior arm | agreement |
|---|---:|---:|---|
| positions | 117,964,800 | — | — |
| flips (`coding argmax ≠ transmitted`) | **227,671** | hc1 / df1 / mi1 227,671 | **exact** |
| float32-saturated positions | **67,955,679** | df1 / mi1 67,955,679 | **exact** |
| indicator code length | **111,275.62229665746 B** | hc1 / mi1 111,275.62229665744 | 1.5e-11 B |
|  · "argmax is wrong" branch | **76,601.5389368755 B** | hc1 / mi1 76,601.5389368755 | **exact** |
|  · "argmax is right" branch | **34,674.08335978196 B** | hc1 / mi1 34,674.083359781944 | 1.5e-11 B |
| live (non-saturated) positions | **50,009,121** | mi1 50,009,121 | **exact** |

Custody: TO2 tokens `cc10a7b0…`, `df1` coding argmax `db498280…`, `df1` coding pmax
`f37e3d8a…`, `df1` RC64 frequency cost `99d7833d…` — all four verified by sha256 at every stage.

---

## 4. The mechanism table — MEASURED, and it corrects `mi1`'s shape

Live positions only, seed 20260824, fold 0 offsets:

| bin | scan position | live positions | flips | observed flip rate | model predicts | fitted β |
|---:|---|---:|---:|---:|---:|---:|
| 0 | **first** | 1,955,187 | 11,904 | 0.6088% | 0.6159% | **−0.0226** |
| 1 | | 5,236,118 | 32,024 | 0.6116% | 0.6367% | **−0.0590** |
| 2 | | 8,644,173 | 47,923 | 0.5544% | 0.5657% | **−0.0182** |
| 3 | | 9,389,913 | 31,073 | **0.3309%** | 0.3454% | **−0.0593** |
| 4 | | 9,617,011 | 34,252 | 0.3562% | 0.3450% | **+0.0427** |
| 5 | | 8,327,058 | 35,624 | 0.4278% | 0.3965% | **+0.1068** |
| 6 | | 4,983,769 | 24,664 | 0.4949% | 0.4633% | **+0.0903** |
| 7 | **last** | 1,855,892 | 10,207 | 0.5500% | 0.5067% | **+0.0825** |

**β is negative for the whole first half of the scan and positive for the whole second half.**
β < 0 means the model over-estimated the flip probability — over-cautious; β > 0 means it
under-estimated it — over-confident. `mi1`'s mechanism claim is confirmed by independently
fitted offsets, and the sign flip is clean at the midpoint.

**The refinement:** the *observed* flip rate is **not** monotone along the scan. It falls
0.6088% → 0.3309% and then climbs back to 0.5500% — a V with its minimum at bin 3. `mi1` read
it as a monotone fall (0.492% → 0.408%) because a 4-cell quadrant proxy cannot resolve a V. The
model **tracks the fall and misses the recovery**, which is why the error is sign-monotone even
though the rate is not. That is a sharper statement of the same mechanism and it is what makes
8 bins the right granularity: 4 cells average the two arms of the V together, and 190 cells
overfit (`mi1`: 51.25 held-out against 104.37 in-sample).

Every cell holds ≥ 1.85M live positions and max|β| = 0.14 — **no cell reaches the Newton clip**,
so unlike `mi1`'s `tile48` / `patch192` rungs this number is not an upper bound on a shippable
form. It is the shippable form.

---

## 5. The price — three ways, and the two that matter

### 5.1 The instrument chain, measured end to end

| step | bytes | source |
|---|---:|---|
| float `−log₂p` ledger, whole token field | **113,776.17901781687** | `ddm_fx5` encode receipt / `ddm_bl1` `EXPECTED` |
| RC64 integer-frequency cost `Σ(31 − log₂ freq)/8` | **113,776.16007613277** | **this arm**, re-derived from `df1`'s retained field |
| physical emitted stream (content) | **113,777** | `ddm_bl1` / `ddm_rr9` / `ddm_fx5` |

Ledger → real coder bits: **0.019 B over 117,964,800 positions.** Coder bits → physical stream:
**0.84 B** of framing and flush. **For a probability-law change that moves no symbol, the ledger
IS the byte count.** The RC64 model carries its own nesting control: at `ratio = 1` it
reproduces the retained field to **exactly 0.0 bits**.

### 5.2 The static offset, both ways

Two-fold cross-fit over a **seeded random** split of all 50,009,121 live positions — never a
prefix, because a prefix of this field is a different population.

| seed | held-out, float ledger | held-out, **real coder bits** | in-sample | max\|β\| |
|---|---:|---:|---:|---:|
| 20260824 | 62.93 | 62.84 | 66.72 | 0.141 |
| 777 | 63.14 | 63.05 | 66.65 | 0.112 |
| 31337 | 63.19 | 63.10 | 66.63 | 0.114 |
| **mean** | **63.09** | **63.00** | 66.67 | — |
| spread | 0.255 (0.40%) | 0.256 | — | — |
| **NULL** (cells shuffled) | **1.59** | **1.57** | 5.07 | 0.037 |

`mi1` measured **64.20** on seed 20260824; I measure **62.93** on the same seed with an
independently written fit. **1.7% apart across two implementations that share no code** — and
in-sample/held-out is 66.67/63.09, a ratio of 1.057, against `mi1`'s 65.77/64.20. The
instrument agrees; the small residual is Newton schedule and fold assignment.

**Noise-floor-corrected signal: ≈61.5 B.** Carry the floor with the number.

### 5.3 The realized stream — PENDING

Three full-n600 re-encodes through `experiments/ddm_jg2_tail_reencode.py` against the DX2 body,
unedited token field, checkpointed every 25 frames:

| row | runtime | members | emitted stream B | Δ vs control | archive B | tokens changed |
|---|---|---:|---:|---:|---:|---:|
| **control** | shipped DX2 r7 | 19 | PENDING | — | — | — |
| `cls_groupbin8` | +1 member, 40 cells | 20 | PENDING | PENDING | PENDING | PENDING |
| `groupbin8_surprise` | +1 member, 2,560 cells | 20 | PENDING | PENDING | PENDING | PENDING |

The control is the proof the encoder inverts the shipping decoder: `stage_control` **refuses**
unless the re-emitted stream is byte-identical to the shipped 113,777 B token section. No delta
from this encoder is trustworthy until it passes.

**The no-op detector.** Each candidate runtime differs from the shipped DX2 r7 tree in **exactly
one file** (`runtime/fx2_model_axis_corrector.py`) and by exactly the intended patch — the
`groupbin8` feature, the member rules, and one name added to the frozen member tuple. Verified
by recursive diff. So the row measures the member and not two changes at once.

**The discount a code-length number owes — and the flaw in quoting it as one.** `ddm_fx5` is the
only measured precedent on this stack: its race predicted **−86.58 B** for D1→E1 and the
physical re-encode returned **−70 B**, a ratio of **0.808**. **I caught myself committing the
error I flag in §1(c) by quoting it.** The −86.58 B was measured on the fx2 race harness, i.e.
on the **fx1-era body** (tokens `9ba2e52b…`); the −70 B was measured on the **DX2 body**
(`cc10a7b0…`). The ratio therefore folds two effects together — code-length→bytes realization
AND a cross-body transfer — and it is not a clean realization ratio for either. Quote it as
*"the one measured race-to-stream ratio on this stack, cross-body, 0.808"*, never as
*"code length realizes at 0.81×"*. **This arm avoids the confound by construction: §5.2 and §5.3
are both measured on the DX2 body**, so their ratio is a clean realization number and it is the
one to carry forward.

---

## 6. The model's own cost, counted

The charter is right that an uncounted table is a fake saving. Here is the count.

| cost | value | how it is charged |
|---|---:|---|
| **stored archive bytes** | **0** | the member is an online KT counter over a context both sides derive from already-decoded state; nothing is transmitted. **PROVEN**, not asserted: §5.3's `archive_delta_bytes` equals its `token_stream_delta_bytes`, so no section grew. |
| **warm-up** | charged **in-stream** | encoder and receiver run the identical cold-start trajectory, so every early mis-estimate is paid for inside the emitted stream. It is inside the §5.3 number by construction, not waived. |
| **decoder tables** | 61.4 KB (`groupbin8_surprise`) / 0.96 KB (`cls_groupbin8`) | `counts` + `hits` + `phat_q`, int64 × cells. Runtime memory, never archive. |
| **decode wall** | **16.4 s** on T4 | MEASURED from rc2→fx5 (+98.68 s of decode+render over six members), against 1,294 s of margin and 53 s of run-to-run noise. |
| **rule 118** | clean | the member reads already-decoded symbols and the position index. No learned table ships; no video-derived constant enters the runtime. |

**Distortion is zero by construction and proven by digest.** A probability model feeds the
coder; the coder emits the transmitted symbol whatever the model said. `stage_fit` never touches
a token. §5.3 re-encodes the **same retained token field** and its receipt carries
`tokens_changed` and the reconstructed-field digest. `d_seg` and `d_pose` are unchanged because
every SegNet cell and every PoseNet input is bit-identical — this is an identity, not a
measurement, and the digest is what proves the identity was honoured.

---

## 7. Honest limits

- **This arm measured no `d_seg`, no `d_pose`, no `S`, and promoted no archive.** `dD = 0` is an
  identity for this family, argued in §6 and proven by token digest, not measured on a scorer.
- **63.09 B is a static two-fold cross-fit; the shipped mechanism is an online KT counter.**
  `ddm_fx2` measured online beating static by ~5,500 B on a *temporal* split; my *random* split
  removes non-stationarity but charges no warm-up. §5.3 is the arbiter and the sign of the
  difference is not known in advance.
- **The offset in §5.2 sits on the final coding row; the member in §5.3 sits inside the mixer.**
  They are not the same object. Part of the 63 B may already be reachable by the incumbent 19
  members once they refit around a 20th, and part may be unreachable by a KT odds multiplier
  that the offset table could take. §5.3 is what settles it; §5.2 is a target, not a promise.
- **The null control bounds the instrument, not the mechanism.** +1.59 B says an 8-cell table
  finds ~1.6 B in nothing; it does not bound what a *smoother* parameterisation of scan position
  could find, in either direction.
- **One assumption traced to zero rather than argued.** The fit clips probabilities at
  `PROB_EPS = 1e-12`, which would silently flatten the confident tail if any live position sat
  below it. MEASURED: **0 of 50,009,121 live positions have `q < 1e-12`** — float32 makes
  `1 − pmax` either exactly 0 (excluded) or ≥ ~6e-8. The clip is inert on this field. The RC64
  path does not clip at all, and it agrees with the clipped ledger to 0.09 B, which
  independently bounds the same thing.
- **`groupbin8` is one context on one binning.** `mi1` left `tile48 × groupbin8` (384 cells)
  unpriced and it remains unpriced here.
- **The C port is not done.** Until it is, this member cannot ship at the native decode speed,
  and the §5.3 archive is a measurement artifact rather than a submission candidate.
- **A candidate-side parse-back is OWED.** The control row proves the encoder inverts the
  shipping *decoder* for the base runtime, and encoder and receiver instantiate the identical
  corrector object from the same frozen config — but I did not run a real receiver over a
  *candidate* stream. `tokens_changed = 0` is an encoder-side statement. The decode-identity
  claim is strong (same object, same trajectory, control passes) and it is not yet closed by an
  independent parse-back. That is the first thing to run before any promotion.

---

## 8. Verdict

**PENDING §5.3.** The verdict shape, declared before the rows land so it cannot be fitted to
them:

- `verdict_scope:` **INSTANCE** — the DX2 body, archive `976f706d…`, n600, 117,964,800
  positions, shipped 19-member fx5 E1 corrector.
- **ADOPT** if a member clears **+20 B of real emitted stream** at full n600 with
  `tokens_changed = 0` — `mi1`'s own falsifier, set at under a third of the measured code-length
  gain to allow for realization loss and for the mixer having already absorbed part of it.
- **DEFER-with-C-port-blocker** if it clears the bar but the port is not built: the saving is
  real and the vehicle cannot carry it yet.
- **HONEST NEGATIVE** if it does not clear +20 B: the code-length gain did not survive the
  mixer, and that is a finding about the realization gap, not about the mechanism.

---

## 9. Fire order for MAIN — queued, not fired

1. **Read §5.3's control row first.** If the control is not byte-identical to the shipped
   113,777 B stream, every other number in §5.3 is void and the encoder is the bug.
2. **If a member clears +20 B:** port it to `runtime/f26_corrector_native.c` following
   `ddm_fx5`'s 3-file precedent — one `case` in `family_rule_index`, one row each in
   `FAMILY_RULE` / `FAMILY_SIZE` / `FAMILY_COUNT_LIMIT`, `N_FAMILIES` bumped, and
   `EXPECTED_SHIPPED_CONFIG` extended. Prove Python/C parity on the emitted stream before
   anything else.
3. **Then one T4 row** for the wall, not for the bytes: the bytes are already exact from the
   local encode. Budget **16.4 s** of additional decode against **1,294 s** of margin. A single
   row cannot resolve one member against **53 s** of run-to-run variance, so treat any inflate
   result inside ±53 s of 505.5 s as "unchanged" — and do not read a favourable row as evidence
   the member is free.
4. **Do not fire `#938`/`eu2` or `ddm_cl1`.** `mi1` closed the paid-probability-model family on
   this body by arithmetic and nothing here reopens it.
5. **Unpriced and cheap:** `tile48 × groupbin8` (384 cells). The two axes are independent, the
   join is the natural composite, and nobody has measured it.

**The honest scale, repeated because it is the thing most likely to be dropped in transit:**
this is **0.15% of the demand**. It is worth taking because it is free and because free wins
compose; it is not a route to sub-0.12.

---

## 10. Custody

**Inputs**, all retained fields from prior arms, consumed by sha256 and never re-derived:

| input | bytes | sha256 |
|---|---:|---|
| TO2 decoded token field | 117,964,800 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` |
| `df1` coding argmax | 117,964,800 | `db498280c22c3aa1b787310e25435116911933216cae558f309f8b10baf7994e` |
| `df1` coding pmax (f32le) | 471,859,200 | `f37e3d8a21d02647437bf950d7a8a75b751c2a9644c7b8ad48aca2833be4794b` |
| `df1` RC64 frequency cost (f64le) | 943,718,400 | `99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86` |
| `hm1` group index | 196,608 | consumed for the decode-order identity check |
| shipped runtime | — | `/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2` (archive `976f706d…`) |

**Outputs** at `/Volumes/APDataStore/pact/ddm_gb1_groupbin8_conditioning/` (APDataStore had
124 GiB free; **Vertigo has 8.4 GiB free and is effectively full**):

| artifact | what it is |
|---|---|
| `measurement_v1/VERIFY.json` | the five reproduced base constants, the decode-order identity, the ledger→coder→stream chain |
| `measurement_v1/FIT_seeds_*.json` | **the payload** — per-fold per-cell offsets, the per-cell mechanism table, the null control, all three seeds |
| `measurement_v1/PATCH_*.json` | the build receipt for each candidate runtime, with the patched-file digest |
| `runtime_cls_groupbin8/`, `runtime_groupbin8_surprise/` | the candidate runtimes (one file changed from the shipped tree, verified by recursive diff) |
| `retained/S1_control_600.json`, `retained/*gb1_*.json` | the physical re-encode receipts (§5.3) |
| `work/encode_*.checkpoint.npz`, `work/tail_*.bin` | **the emitted streams themselves**, kept as bytes, not reduced to a length |
| `launch_*/run.log`, `launch_*/launch_manifest.json` | governed-launcher provenance for every row |

The per-cell table is retained rather than summarised because **it IS the mechanism** — it is
what says the model is over-cautious early in the scan and over-confident late, and a downstream
arm needs the cells, not the total.

**Reproduce:** `.venv/bin/python experiments/ddm_gb1_groupbin8_conditioning.py --stage
{verify,fit,patch}`. Verify 16 s; fit ~2 min for three seeds plus the null; patch < 1 s. Every
stage fails closed on a custody or internal-consistency mismatch and writes its own receipt.
