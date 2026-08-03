# ddm_rd2 — RESOLVER + DECOMPOSITION: the fix was half-landed, the REJECT is stage-scoped, and Road|Lane carries 64.8% of the instability

**Arm:** ddm_rd2 · 2026-08-02 · axis `[macOS-CPU advisory]` · `score_claim=false` `promotable=false`
· exact contest pointer **UNMOVED** · **no n600 scorer job fired** (arm `ddm_pg1` holds the slot).

**STORES CONSULTED:** `.omx/research/ddm_cv1_seven_surface_convocation_20260802.md` (§2 RESOLVER, §11 ratios)
· `ddm_uv1_ep854_pose_illegibility_reject_20260802.md` · `ddm_fl1_perclass_flicker_floors_20260731.md` +
its driver at source · `ddm_xp1_20260731/{xp1_verdict.json, chunk_*.npz}` · `experiments/ddm_v4c_resolve.py`
· `experiments/ddm_v4d_resolve.py` · `experiments/ddm_v4d_build_composed_archive.py` · the live archives
`gr1_cell_drop50_archive.zip` / `ep854_v3warp_base_archive.zip` · `tac.canonical_equations.gap_decomposition_against_floor_20260802`.
**Deliberately not loaded:** burn telemetry, gc-series memo bodies.

**Denominator for every ΔS below** (registered equation, PR130 at **191,052 B** per na1's correction):
total gap **0.7262340**, **1% of gap = 10,906.7 B**. My independent recomputation reproduces the
registered `bytes_per_percent_of_gap` to 6 dp, which is the instrument check for everything that follows.

---

## 1. HEADLINE A — the resolver fix was HALF-LANDED; I finished it

`ddm_uv1` parametrized `ddm_v4c_resolve.py` (`resolve_base()` + `--base-archive`) and explicitly refused
to add a third literal because *"it reproduces the same wall one row later."* **The wall reproduced anyway,
in the sibling file.** MEASURED at source, before the fix:

| binding | site | state |
|---|---|---|
| `v4c.build_oracle("celldrop50", s_r=1.0)` | `ddm_v4d_resolve.py:336, 557, 723` | hardcoded ×3 |
| `stamp_fit_context(..., base="celldrop50")` | `:401` | hardcoded (the sf1 staleness stamp itself) |
| `PHOTO_JL = V4C / "photo_celldrop50_resolve.partial.jsonl"` | `:65` | hardcoded input path |
| base flags in argparse | `:842-852` | **none** — `--mode/--k/--max-seconds/--tiebreak-lambda/--accept-eps/--resume` only |

**Why this is the load-bearing finding and not a tidiness note.** `ddm_v4d_resolve.py` is the **refine
stage** — dim0-offset re-solve + `(a,b)` re-fit + beta select. It is *precisely* the stage uv1 named as
its own honesty limit: *"the live chain adds a dim0-offset refine and beta select on top of what I ran;
that composite stage historically moved this chain 0.010384 → 0.007646 (−27%)."* uv1 could not run it —
not because it chose not to, but because **the stage was structurally unreachable for any base but
`celldrop50`**.

⇒ **The ep854 REJECT is scoped to the stages that could be run.** That is a strictly narrower scope than
uv1 stated, and neither uv1 nor cv1 could see it, because the limit lived in a file uv1 did not touch.

**Landed** (commit `a27f720b20`): `--base` / `--base-archive` / `--photo-jl` / `--out-dir`, bound once in
`main()` before dispatch via `_bind_base_context()`. **MEASURED**: default invocation leaves every global
byte-identical (`(_BASE_LABEL, _BASE_ARCHIVE, PHOTO_JL, OUT)` unchanged); a non-default base **without**
`--out-dir` **refuses**. That refusal is the point — the `*.partial.jsonl` resume caches carry **no base
field**, so a second base writing into `ddm_v4d_20260731/` would resume as if clean while mixing two
bases. Cross-base contamination is now unrepresentable rather than merely discouraged.

### The backcast count (the question cv1 §2 posed and left open)

Delegated, exhaustive, denominators reported. **MEASURED** identifiers/numbers; **DERIVED** classification.

| tier | definition | count |
|---|---|---:|
| A | blocker names the hardcoded `BASES` **verbatim** | **1** (ep854) |
| A+B | recorded reason *is* "pose could not be re-solved against this base" | **2** (+ wr1 Knee-B) |
| **A+B+C** | **measured seg-or-rate win over the live base, pose structurally unobtainable, base never in `BASES`** | **11** |
| A+B+C+D | all infrastructural closures in window (incl. other mechanisms: #417 receiver location, format change, dead lineage) | **15** |

**9 of the 11 have never had a pose number of any kind.** Named: `wr1_kneeB` (174,578 B — the first
byte-closed archive inside the sub-0.15 byte budget, rate −0.263 S) · `ep806` / `ep809` / `ep934` (all
with `composed/pose_term_s = None`, and ep934 *beats* its control) · `ep879` (never scored on any axis) ·
gr1 `cell_drop35` / `cell_drop63` / `cell_rung_a` (a re-race whose own memo says its ordering is
*"SEG-only… pose-BLIND"*) · `r1c ep641` · `dw1 control-B`. **4 rows are genuine SCIENTIFIC refutations**
and stay closed.

**Capability windows, MEASURED from git:** `BASES` born hardcoded `8f46cc147a` 2026-07-30T02:38 → v4c
fixed `2330ffbdcf` 2026-08-02T08:16 (**3.24 days**). The *builder* was parametrized `07a1945629`
2026-08-01T21:12 — **11h04m before** the resolver, which is the exact asymmetry that let a transplant be
BUILT but never RE-SOLVED. **My correction: the window did not close on 08-02.** The refine stage stayed
literal until `a27f720b20` today, so for the 9 never-posed candidates the limit was continuous from
2026-07-30 to now.

**Scope honesty inherited from the backcast:** its repo-wide `grep -rn` **timed out at 200 s and returned
`count: 0` with zero matches** — a FALSE CLEAN; the scoped re-run found 4 matches in 2 files. The
negative-existence bound above is therefore **not exhaustive**; it is "did not find in the scopes
searched" (7 corpus queries over a 12,030-doc index, 1,942 in-window memos, 409 ledger rows, 718 disk
dirs). Also noted: `ls .omx/research` = 9,706 entries vs corpus `research(7398)` — **~2,300 entries are
not indexed and nobody knows why.** That is a live instrument defect, not a footnote.

---

## 2. HEADLINE B — the −0.0866 re-priced under the D1 acceptance rule

Operator D1: *"don't judge moves that produce significant wins on segment and rate if they hurt pose."*
Decision column = seg+rate. Pose = a separate, named line.

**Correction carried first (m46):** uv1 priced against `v4d_pw1` (S 0.9476092). The **live best is
`dc1_fold` (S 0.8983775)** — superseded twice since. Re-priced against the true baseline:

| | live `dc1_fold` | composed `cr2_ep854` | Δ |
|---|---:|---:|---:|
| seg | 0.4311790 | 0.3944070 | **−0.0367720** |
| rate | 0.2399150 (360,309 B) | 0.1901220 (285,529 B) | **−0.0497929** |
| **DECISION COLUMN (seg+rate)** | **0.6710940** | **0.5845290** | **−0.0865649** |
| pose (separate line) | 0.2272835 | 19.4620483 | +19.235 |

**−0.0865649 S = 11.920% of the 0.7262340 gap = 9,443 B-equivalent.** The win is **robust to the baseline
correction** (shift 9.3e-6 — both baselines share d_seg exactly and differ by 14 B of rate), so uv1's
number survives; only its *denominator* and *break-even* change.

**57.5% of the win is RATE, 42.5% is seg.** Independently reproduces na1's 57%. This row is primarily a
**rate** move that was vetoed on pose.

**What the baseline correction *does* change — the break-even TIGHTENS 1.338×:**

| | pose break-even | source |
|---|---:|---|
| vs superseded `v4d_pw1` (uv1 used this) | 0.0131903 | uv1 §1 pre-registered |
| **vs LIVE BEST `dc1_fold`** | **0.0098501** | this memo |

Best MEASURED ep854 d_pose after full θ re-solve = 2.138939 ⇒ **217.1× over** (not uv1's 162×).

### The stale-vs-real pose question (na1's ck1 route), answered

**It is NOT stale in the `ck1` sense.** uv1 ran a full θ re-solve *against ep854* — single-plane GN +
two-plane GN from 3 starts + `(a,b)` re-fit from 2 starts — with a **passing positive control** (same
solver, same 4 pairs: gr1 **0.000709** vs ep854 **2.138939**, 3,019× separation). The `ck1` precedent
(+0.185 S regression that was *entirely* stale parameters) does not apply: these parameters were
re-solved, and the control proves the machinery was working when it returned 2.139.

**But it is STAGE-INCOMPLETE, and that is the live question.** uv1's re-solve is the v4c-level solve. The
v4d refine stage was structurally unreachable (§1) and has **never** been run against ep854. Honest
arithmetic on whether it could rescue the row:

- required: 2.138939 → ≤ 0.0098501 = **−99.54%**
- the v4d stage's historical effect on the *gr1* chain: **−27%** (0.010384 → 0.007646)
- ⇒ it would need to deliver ~**370× its historical rate**, on a different base.

**INFERRED, not measured** — and the −27% is a *borrowed number* from another base, so it is a
hypothesis here, not a bound. My read: rescue is very unlikely, and I am **not** claiming it is
impossible, because the measurement is now cheap and was previously impossible. The correct status is
**REJECT scoped to v4c-level stages; full-chain test now available and BLOCKED-ON-SLOT.**

### The pose-repair budget curve (computed here; nobody had)

The composition frees **74,794 B**. The natural question — "can we buy the pose back with the freed
bytes?" — has a **measured NO**, and the direction is counter-intuitive:

| extra bytes spent on a pose carrier | max d_pose that still beats live |
|---:|---:|
| 0 | 0.0098501 |
| 7,200 (a stored-target sidecar) | 0.0095260 |
| 74,794 (ALL freed bytes) | 0.0069877 |

Spending the freed bytes **tightens** the pose requirement, because the freed bytes *are* the win — they
are already counted in the −0.0498 rate line. **There is no separate pose budget.** The repair must come
from d_pose itself, not from rate headroom. This retires "just spend the freed bytes on pose" as a move.

### One decomposition-flavoured observation, honestly scoped

uv1's 4-pair re-solve is `[0.53951, 7.01989, 0.99609, 0.00027]` — **one pair carries 82.0% of the mass**,
and one pair is already at 0.00027 (**inside** break-even). And **MEASURED**: `decode_token_codes` on the
live archive returns shape **(600, 24, 32, 4)** — *axis 0 is the pair index*, so the token field is
**per-pair addressable**. A per-pair hybrid base is therefore mechanically constructible; measured ship
cost **3,341 B renderer + 535 B selector + 75 B mask = 3,951 B = 0.00263 S = 3.0% of the win** (UPPER
BOUND: assumes both sections duplicate, and **excludes** the entropy penalty of mixing two token
distributions in one `dr7t` stream, which is measurable and unmeasured).

**Scope: INSTANCE(4 pairs), per m88** — a prefix verdict on a skewed per-pair quantity is instance-scoped,
and `bp2` measured that exact failure this session (a −0.122 S prefix "win" became +0.152 at n600). This
is a **motivation for an n600 per-pair census, not a result.** I did *not* build the hybrid ladder into a
verdict: doing so requires per-pair gr1 fallback values I do not have (only their mean), and mixing a
control mean with shipped per-pair values would be an apples-to-oranges error in my own work.

---

## 3. DECOMPOSITION — the harness, validated, with per-EDGE emission

**Landed:** `src/tac/optimization/ddm_rd2_perclass_peredge_decomposition.py` + 29 tests (all pass, ruff
clean), commit `a27f720b20`.

### Why a new pass is required (MEASURED, this is not a preference)

`ddm_xp1`'s cached chunks carry `cls_gt (120,5)` and `cls_base (120,5)` — **marginals only, no 5×5
joint**. Marginals do not determine the joint (pinned as a test). **The per-EDGE decomposition has never
been computed at ANY endpoint**, and cannot be back-derived from either cache.

### Instrument validation — $0, scorer-free, against fl1's registered vector

| class | mine | fl1 registered | abs err |
|---|---:|---:|---:|
| Road | 0.18894 | 0.18890 | 3.6e-5 |
| Lane | 0.23162 | 0.23160 | 2.2e-5 |
| Undrivable | 0.03939 | 0.03940 | 1.0e-5 |
| Movable | 0.02847 | 0.02850 | 3.3e-5 |
| MyCar | 0.04343 | 0.04340 | 2.8e-5 |
| **TOTAL** | **0.53184** | **0.53180** | **4.4e-5** |

All within fl1's own 4-dp publication rounding. Row-sum identity (per-class == per-edge row sums) holds;
charge-class is invariant to neighbour choice. **The instrument is validated; only the `pstars` input is
blocked.**

### THE NEW OBJECT — per-EDGE GT-flicker (top rows, n600)

| edge | S | share |
|---|---:|---:|
| **Lane → Road** | 0.22912 | **43.1%** |
| **Road → Lane** | 0.11541 | **21.7%** |
| MyCar → Road | 0.04272 | 8.0% |
| Road → MyCar | 0.03883 | 7.3% |
| Undrivable → Road | 0.03214 | 6.0% |
| Road → Undrivable | 0.02898 | 5.4% |
| Movable → Undrivable | 0.01851 | 3.5% |

**The single Road|Lane edge carries 64.8% of all instability.** This reframes cv1 §11's "Road is 44% of
the residual": the per-class view attributes mass to *Road*, but the per-edge view says that mass is
overwhelmingly **Road↔Lane confusion**. **Road's residual is not a Road problem — it is a Road|Lane
BOUNDARY problem.** That is the signal `ddm_pc2` needs and it is the reason directive 4 asked for edges.

**Geometric split:** **38.23%** of flicker mass sits in class **interiors** (no differing 4-neighbour) —
mass that no boundary/edge term can reach. ~62% is on boundaries. Both views ship from one pass.

### Three corrections to cv1 §11 (mine to make; §11 is MAIN's own)

1. **Denominator mismatch.** fl1's registered 0.005318 and its per-class vector are the **/598 interior**
   convention, *not* the /600 its prose calls "primary" — MEASURED: my /600 numbers are off by exactly
   600/598. §11 joined that /598 reference against xp1's /600 residual. Effect is small (0.33%) but it is
   the unanchored-quantity class. The module now takes `denom` explicitly at every call site.
2. **Road is at ratio 0.997, not 1.00.** Under the matched denominator, Road's ep641 residual 0.18845 vs
   reference 0.18894 is **already marginally below**. "Sitting EXACTLY on its floor" was an artifact of
   the mismatch plus rounding — which independently supports the operator's *"there are no floors."*
3. **It is CROSS-VEHICLE, not merely cross-endpoint.** xp1's endpoint is a witness-trunk checkpoint
   (`ddm_r1c_20260731/window_01/checkpoints/stage_seg_trunk_tau_final.npz`); the live base is the TR1
   LOTTO token renderer (`gr1_cell_drop50_archive.zip`). Per L18, cross-vehicle transfer is weaker than
   cross-endpoint. §11's ratios are **labelled structure, never numbers**, and I have not treated them
   otherwise.

**No floor language anywhere.** `ExhaustionIndicator` has no `floor` field, hard-wires
`reference_is_a_bound = False` (a caller passing it raises `TypeError`, tested), and reads
`REPRESENTATION_EXHAUSTED_NEEDS_NEW_CARRIER` — *this representation has run out*, never *this is
unreachable*.

---

## 4. BLOCKED-ON-SLOT — exact invocations (`ddm_pg1` holds the scorer slot)

**(a) The live-base re-join** — needs `pstars` = our rendered `frame_1` frozen-CPU-torch SegNet argmax at
the live base, shape (600,384,512), values 0-4, class order `[Road, Lane, Undrivable, Movable, MyCar]`.
That is the ONLY blocked input; everything downstream is $0 and validated:

```python
from tac.optimization.ddm_rd2_perclass_peredge_decomposition import (
    residual_confusion, flicker_confusion, edge_rows, boundary_band_masses, exhaustion_table)
dec  = residual_confusion(lstars, pstars)              # 5x5 joint, per-class == row sums
ref  = flicker_confusion(lstars, denom="all")          # commensurable with a /600 residual
rows = edge_rows(dec)                                  # ranked ordered edges  -> ddm_pc2
geo  = boundary_band_masses(lstars, (lstars != pstars))# on/off-boundary split
tbl  = exhaustion_table(dec.per_class_S, ref.per_class_S)
```

**(b) The ep854 full-chain pose test** — previously impossible, now a command. Requires the v4c photo
stage against ep854 first, then:

```bash
.venv/bin/python experiments/ddm_v4c_resolve.py --mode photo --base ep854 \
    --base-archive /Volumes/VertigoDataTier/pact/ddm_cr2_20260801/ep854_v3warp_base_archive.zip
.venv/bin/python experiments/ddm_v4d_resolve.py --mode refine --base ep854 \
    --base-archive /Volumes/VertigoDataTier/pact/ddm_cr2_20260801/ep854_v3warp_base_archive.zip \
    --out-dir /Volumes/VertigoDataTier/pact/ddm_v4d_ep854_20260802
```

**Pre-registered falsifier:** ACCEPT iff mean d_pose ≤ **0.0098501** (vs LIVE BEST). Anything above and
the composition stays REJECT — now scoped to the *full* chain rather than to v4c-level stages.

**(c) The $0 pre-screen uv1 built, run on the 9 never-posed candidates.** `corr(f1_new, f1_incumbent)` +
per-channel means, from two renders, seconds each, **no scorer slot**. corr ≈ 0.119 predicted ep854's
2,871× miss. This triages 9 stranded rows before any of them is priced. **Highest value-per-minute item
in this memo and it needs nothing from anyone.**

---

## 5. My own round-1 adversarial review — defects in my own work

- **I shipped a silent failure while hunting silent failures.** My review-gate loop was
  `mark-file … >/dev/null 2>&1 && echo pass1`; pass 1 failed, the redirect ate the error, and the `&&`
  suppressed the echo — so I "saw" two passes and had one. Caught only because the serializer refused.
  Exactly the genus in `m50`, committed by me, in the same session I wrote it up.
- **My first denominator was wrong and the instrument caught it.** I ran the harness at /600 and missed
  fl1 by a uniform 0.33%. I could have called it rounding. Re-deriving the ratio gave exactly 600/598,
  which located a real convention mismatch in §11. The lesson is that a uniform small error is a
  *signature*, not noise.
- **My first hybrid ladder was an apples-to-oranges verdict.** I built a "keep k pairs on ep854" table
  whose fallback used gr1's *control mean* against ep854's *per-pair* values, and it produced an absurd
  −0.17 S. I deleted the verdict column and kept only the skew statistic. Had I trusted it I would have
  reported a fabricated win.
- **I nearly reported the wrong gap denominator.** uv1 quotes 11.16% (0.7754681); my charter says
  0.7263025. Both are "the gap" — against different baselines. Resolving it surfaced that uv1 priced
  against a **twice-superseded** baseline, which is where the 1.338× break-even tightening came from.
- **Would my tests pass if the code were broken?** The load-bearing one is
  `test_confusion_is_not_determined_by_marginals` — without it, "the joint is new information" is a claim,
  not a property. `test_exhaustion_reference_is_a_bound_cannot_be_set_by_a_caller` is what makes "no
  floors" structural rather than a naming convention.
- **Scope I did NOT close:** the v4d parametrization is verified for **default-identity** and **refusal**,
  not end-to-end against ep854 — that needs the slot. I did not re-run the celldrop50 chain to prove
  byte-identity of its *outputs*; I proved the globals are unchanged, which is a strictly weaker claim,
  and I am stating it as the weaker claim.
- **What would change my read:** if the v4d refine stage on ep854 returns anything under ~0.02 d_pose,
  my "very unlikely" becomes wrong and the row is live. That is a real possibility, not a courtesy.

---

## 6. Verdict-scope ladder

| claim | scope | basis |
|---|---|---|
| v4d had 5 hardcoded base bindings + no base flag | **MEASURED, EXACT** | source inspection, line numbers |
| v4d default invocation is globals-identical; non-default refuses | **MEASURED** | executed both paths |
| ep854 REJECT is scoped to v4c-level stages | **DERIVED** | uv1's own honesty limit ∧ my measured v4d limit |
| v4d rescue is very unlikely (needs −99.54% vs historical −27%) | **INFERRED** | −27% is a borrowed number from the gr1 base |
| seg+rate −0.0865649 = 11.920% of gap, 57.5% rate | **MEASURED** | two byte-closed n600 rows, recomputed from components |
| pose break-even vs LIVE BEST = 0.0098501 | **DERIVED** | closed form from the live-best row |
| freed bytes are not a pose budget | **DERIVED, EXACT** | the curve; the bytes are already in the win |
| harness reproduces fl1 to 4.4e-5 | **MEASURED** | n600 cached lstars, $0 |
| Road\|Lane carries 64.8% of GT-flicker | **MEASURED** | n600, new object |
| 38.23% of flicker mass is interior | **MEASURED** | n600, 4-neighbour |
| fl1 registered vector is /598 not /600 | **MEASURED** | exact 600/598 ratio, pinned as a test |
| Road ratio is 0.997 not 1.00 | **MEASURED** (cross-VEHICLE labelled structure) | matched denominator |
| 11 infrastructural stranded candidates | **MEASURED** ids/numbers, **DERIVED** classification | backcast; bound NOT exhaustive |
| per-pair token hybrid is constructible at 3,951 B | **MEASURED** shape + section sizes; **UNMEASURED** entropy term | archive inspection |
| 4-pair skew (82% in one pair) | **INSTANCE(4 pairs)** per m88 | uv1 probe set |

---

## 7. NEXT-IF-RESUMED

1. **Run the $0 corr pre-screen on all 9 never-posed candidates** (§4c). No slot, no scorer, minutes.
   It is the only item that converts 9 stranded rows into a ranked list without spending anything.
2. **Fire §4b** (ep854 full chain) when the slot frees — it closes the last stage-scope gap on the
   campaign's largest single actionable row.
3. **Fire §4a** the moment any arm produces live-base `pstars`; hand `edge_rows` to `ddm_pc2`.
4. **Audit the other 8 pose/theta re-solvers for the same hardcoded-base class.** `ddm_ck1` carries its
   base *in the filename*; `ddm_p3v2_optimal_form_pose_resolve.py` is unexamined by me. The bug class is
   "a solver hardwired to the base it was born against," and I fixed exactly one instance of it.
5. **Someone should explain the 9,706-vs-7,398 corpus index gap.** ~2,300 research docs are not indexed;
   every corpus-scoped negative-existence claim in this campaign inherits that hole.
