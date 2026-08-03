# ddm_xa1 — cross-axis technique transfer: one correction to a live sister's table, and five closures

`verdict_scope` tokens attached per row. Exact contest pointer **0.1910828242 UNMOVED**.
Own-vehicle rows are `[macOS-CPU advisory]`, `score_claim=false`, `promotable=false`.
**Zero scorer forwards were run in this arm** — the n600 slot is held by `ob1`.
Live best **S = 0.7910689 @ 353,805 B**; gap **0.6189279**; 1% of gap = 0.0061893 S = 9,295 B.

STORES CONSULTED: `ddm_ss1` (**the governing artifact — see §0.0**), `ddm_pu2`, `ddm_ms8`,
`ddm_p4x`, `ddm_sq1`, `ddm_na2`, `ddm_mh1`, `ddm_sv1`, `ddm_os1`, `ddm_pg1`, `ddm_dc1`;
the registered laws `ddm_os1_termination_census_from_cost_proxy_v1` and
`ddm_pw1_menu_saturation_discriminator_v1`; `tools/sb1_seg_batch.py`;
`/Volumes/VertigoDataTier/pact/ddm_sb1_20260729/qa03/` (receipt + 120 instance rows);
`/Volumes/VertigoDataTier/pact/ddm_pb1_20260729/p2c_aimed_archive.zip` (sha `b9a7983b`);
`/Volumes/VertigoDataTier/pact/ddm_dc1_20260802/dc1_degen_meta.json`.

---

## §0.0 — This arm was largely overtaken mid-flight by `ddm_ss1`, and defers to it

`ddm_ss1_selection_vs_search_20260803.md` landed **during this arm's run** and is the live sister
my own charter names. It builds the cross-axis technique ladder this arm set out to build, with a
better denominator, and it supersedes most of what I had drafted. Saying so plainly is cheaper
than letting two memos compete.

**ss1 §4.1 is the transfer table**, three rungs, cost- and reach-ordered, each containing the one
above: **SELECTION** (a different element of the current menu) → **MORE SEARCH** (did we stop too
early inside this basin) → **PLACEMENT** (wrong neighbourhood: refit the menu **or re-initialize
the solve** — ss1's key structural point is that *these are the same rung*, which is the pose↔rate
bridge). **ss1 §4.3 is the free diagnostic** mapping an observed stop-reason to its binding rung.

**What ss1 already settled, that I therefore do not claim:**

* **The 5.7× placement-vs-selection ratio does not generalize.** ss1 §4.2: *"`#873`'s 5.7× is a
  measurement of ONE menu's diagnostic state, not a law about placement vs selection. Do not let
  the ratio steer anything. **The DIAGNOSTIC generalizes; the ratio does not.**"*
* **Occupancy across all five measured menus.** `st_grid` is the only one with any dead codeword
  (63.6%); `selector` **0/2**, `rs_beta_mags` **0/13**, `token_quant_levels` **0/16**, SMEVR
  mode-base **0/16**. *"no placement refit is available anywhere else on this vehicle."*
* **The proxy-vs-realized classification** I had queued as an open row: ss1 measured it —
  **REAL 44 / PROXY 4 / MIXED 4** over a 52-surface call-graph census (against its own hand-listed
  11, which it reports as a 21% sample). **I retract that row.**
* **Both multi-start cures are already built and switched off** — `ab_multistart_gn`
  (`ddm_v4c_resolve.py:195`), `--ab-starts` default `"neutral"`, derived bounds at `:633,639` with
  no live caller: *"Two CLI defaults, no new code"* but *"not flippable blind"* (its supporting
  evidence is n=8 on a base `uv1` later rejected).
* **Solver census shape:** SINGLE_START 10 / MULTI_START 2 (one off by default); every damping
  ladder on the chain BOUND_ONLY, 5 of 5; 6 files reported VACUOUS rather than clean (`m50`).

**What survives as this arm's contribution is one row of ss1's own table, measured the other way
round** (§0.1), plus four closures and one apparatus finding. That is the whole memo.

## §0.1 — THE FINDING: ss1's surface #7 is marked CURED from source; the cure has never been run

ss1's surface table, row 7, verbatim:

| 7 | `sb1_seg_batch` `--max-quanta` (QA03) | seg | **CURED** — `sv1`: default 32, `for/else` `stop_reason`, emits `n_cap_saturated` | N/A | REAL |

Every clause of that is true **about the source file**. `dc1` raised the default 4 → 32 on 08-01,
added `for/else` `stop_reason`, and added `n_cap_saturated` + `cap_saturated_frac` to the receipt.
`sv1` §0 verified it at source (*"STALE — already cured"*). ss1 inherited the verdict.

**But the cure has never been executed.** Exactly one QA03 run exists anywhere on disk —
`/Volumes/VertigoDataTier/pact/ddm_sb1_20260729/qa03/`, whose receipt reads
`max_quanta_per_cell = 4` — and no receipt in the repo or on the SSD carries `cap_saturated_frac`
outside the three `.omx/research` memos that *describe* the field. The chain is: `sv1` read the
source → `ss1` inherited "CURED" → **nobody read the receipt.**

This matters for ss1's conclusion, not just its bookkeeping. ss1 §4.3 concludes: *"on the live
chain the free evidence says PLACEMENT on three of the four defect rows and **MORE-SEARCH on none
of them**."* Its own diagnostic row for MORE-SEARCH is *"solve stops because the **iteration cap**
binds while still descending."* The single existing QA03 receipt shows exactly that: **51/120
instances (42.5%) stopped on the cap, carrying 64.7% of all realized flips** — and §3 Probe A
measures, for the first time, that they were **still descending at essentially their initial rate**
when cut off.

**So the seg token solve is a MORE-SEARCH row on the live chain, and it is the only seg-axis solver
row in the table.** ss1's "MORE-SEARCH on none of them" holds for the four pose/rate rows it read;
it does not hold once surface #7 is read from its receipt instead of its source. `verdict_scope:
INSTANCE` (this receipt, this atlas aim). This is a correction to one cell, offered to a sister
arm whose framework I am otherwise adopting wholesale.

**The generalizable form** — and it is a genuine cross-axis law, because it is the *same* failure
`os1` documented one level down: **a cure landed in source is not a measurement.** `os1`'s census
exists because *emission* fixes the future while the past stays unreadable; here the emission
landed and the future never arrived. A surface may be marked CURED only from a receipt produced
*after* the cure. Registered as a memory law, not a fifth tracking surface.

---

## §0.2 — What I refute in my charter (four items)

**R1. Two quotations the charter attributes to `ddm_mh1` do not exist in `ddm_mh1`.**
The charter cites *"the more attractive the transfer, the harder the gate applies"* and *"never a
17th tracking memo — mh1's law"*. Neither string is in the file; a repo-wide grep for
`more attractive the transfer` across `.omx/` returns zero hits. The real texts:

* mh1:165, applied to its own int8 orphan — *"It would be easy to headline '0.043 S sitting
  unclaimed'; that would be the ancestor-number error with a more attractive number attached."*
  The general rule mh1 does state is structural, not aesthetic (mh1:166): **"The mechanism
  transfers and must be re-measured on TR1; the number does not."**
* mh1:271-288 — *"**This memo would be the 17th.** … Each sweep has *created a surface* rather
  than *draining a canonical one*"*, binding consequence *"routing lands as **rows in
  `canonical_task_status.jsonl`**"*.

The paraphrases were directionally right and verbally wrong. I comply with the real forms: this
arm banks **no number** as a transfer, and lands routing as ledger rows.

**R2. The charter's premise is half wrong, and the correction changes the cure.**
Charter: *"a technique that cures a pathology on one axis has no structural reason to reach the
others — it isn't suppressed, it's just never carried."* **False in the strong form.** The
transfers happen and are then not *banked*. `ddm_sq1`'s headline seg cure (realizer v1,
`eta_net −3.7640 → +0.7895`, 32/32 pairs, sd 0.0545) lists among the two riders that made it work
**"multi-start (`dec` and `truth` inits, `pu2` mechanism)"** — the pose→seg multi-start transfer
already happened, the same day, recorded as a parenthetical rather than as a technique. ss1 §5.-1
found the code-level twin: both multi-start cures **already built and switched off**. So the
disease is not non-transfer; it is **non-banking of transfers that already occurred** — mh1's
SPLIT-BANK genus (*"whichever number the write-up put in its headline got banked, and the rest of
the same measurement did not"*) operating on techniques instead of numbers. Hunting for new
transfers is the lower-value half.

**R3. `TRUNCATED SOLVE (#850)` as a seed is STALE — and is now stale by three independent arms.**
`sv1` (08-01), `pg1` (08-02) and `ss1` (08-03) each re-derived it: `terminal_pose_gn.py:490-497`
carries the stop-on-rejection proof, `:1085-1092` honours it across resume, consumers are only
`tools/pb1_*` and `tools/rehearse_terminal_pose_gn.py`, and *"the live `ddm_v4c_resolve →
ddm_v4d_resolve → build → inflate_runner_v4d` chain never calls it."* The charter's companion
figure is also wrong: `pg1:28` measured the descent at the shipped bound as **1.2% per
relinearization, not 13–23%** — *"directionally right, wrong magnitude"* — the 30.7%/19.9% drops
happen in the first two relins, which the shipped bound already buys. **I had cited the 13–23%
figure in my own first draft and am striking it.**

**R4. The charter fuses DEGENERACY and PLACEMENT under "dead codewords". Both are closed, for
different reasons, by different arms.**
`dc1` closed **degeneracy**: `s_t` is the unique exactly-degenerate menu — and the finding is
sharper than ms8's own framing. dc1 §1 re-derived it on ms8's own per-pair factors through the
receiver (`pfs1_warp_receiver.py:45-46`): **max relative homography difference 4.539e-16 over 600
pairs**, with `mq1` §2 independently at 5.98e-16 on arbitrary factors — the "two derivations".
Consequence, verbatim: *"**`gap_lattice` for `s_t` is IDENTICALLY ZERO** … **ms8's entire −0.049177
is `gap_search`** … The contradiction was a labelling error in ms8, not a measurement error."*
dc1 §7 then corrects the discriminator a second time: *"A menu whose value is exactly degenerate
with a continuous coordinate that already ships is a **search-reach parameter, not a quantizer**
… **Test degeneracy BEFORE fitting a codebook** — it is a 10-line algebraic check on the receiver,
and it is free."* And its self-scoping: *"the method was not a menu method. It was a **degeneracy
method with a sample size of one**."*
`ss1` closed **placement**: 0% dead on all four other menus, so *"no placement refit is available
anywhere else on this vehicle."* My §3 Probe D independently reproduced the
`token_quant_levels` 0/16 leg before I had read ss1; I report it as a **reproduction**, not a
finding.

---

## §1 — Denominators (`m50`: report the denominator and where the instrument stops)

| population | authority | count |
|---|---|---:|
| solver surfaces on the live chain | **`ss1` call-graph census (defer to this)** | **52** |
| — of those, SINGLE_START / MULTI_START | `ss1` | 10 / 2 (one off by default) |
| — of those, REAL / PROXY / MIXED objective | `ss1` | 44 / 4 / 4 |
| solver sites this arm classified | this arm | 8 (a **15%** subset of ss1's 52) |
| charter-seeded candidate techniques | this arm | 10, all classified |
| techniques found beyond the charter seed | this arm | 4 |
| scorer-free probes executed this arm | this arm | 4 |
| — of those, NEW / reproductions of a sister | this arm | 3 / 1 |

**My instrument's limit, stated because it is real and because ss1's is better.** My automated
descent-loop scan (iteration-bound regex ∧ accept-test regex over `experiments/` + `tools/` +
`src/tac/`, excluding `results/`, `manim_levelset/`, `tests/`, `.venv/`) returned **18 candidate
modules** and **missed at least two live sites `sv1`/`os1` had already named by hand**
(`ddm_pfs1_ep_warp_pose_solve.py:183`, `ddm_v4c_resolve.py:521`) because they bound on
`relin_bound` rather than a `max_*` name. My 8 sites are the union of that scan and the
hand-named live-chain sites. **It is not a census and must not be cited as one — cite ss1's 52.**

**Not reached:** the rate-side coder search (`ba29`'s 19-coordinate sweep, `br1`'s 264
evaluations) was read for verdicts but not classified on the ladder — those are sweeps over a
discrete design set, not descent loops. Named rather than silently dropped.

---

## §2 — Transfer table (consuming `ss1`'s ladder; ranked by pathology-present × cheapness)

| # | technique | MEASURED-ON | target | pathology on target? | status |
|---|---|---|---|---|---|
| 1 | **MORE SEARCH** (uncap + continue) | pose (`pu2` 67/21; `pg1`) | **seg `sb1` qa03** | **YES — measured §3 A/C** | **OPEN — the one row, §4** |
| 2 | **PLACEMENT-as-re-initialization** (multi-start) | pose (`pu2`); seg *realizer* (`sq1` rider) | **seg `sb1` qa03** | **YES — structural §3 B** | **OPEN — same run, §4** |
| 3 | technique banking (meta) | rate (mh1, numbers) | techniques, all axes | **YES — §0.2 R2** | OPEN, $0 |
| 4 | PLACEMENT-as-menu-refit | rate (`ms8` `st_grid`) | seg `token_quant_levels` | **NO — 0/16 dead** | §5 (ss1; reproduced §3 D) |
| 5 | the 5.7× placement:selection ratio | rate, one menu | any | **NO — ratio does not generalize** | §5 (ss1 §4.2) |
| 6 | degeneracy / dead-codeword refit | rate (`s_t`) | seg, pose menus | **NO — unique, algebraically** | §5 (dc1 §1/§7) |
| 7 | truncated solve | pose | `terminal_pose_gn` #850 | **NO — cured + off chain** | §5 (three arms) |
| 8 | proxy-vs-realized accept test | seg (`sq1` rider) | all solvers | measured by ss1 | §5 — **row retracted** |
| 9 | null-space exploitation | pose (`sq1` P7, 56.5×→1.039×) | rate (`m86` 22.70% D-blind) | YES | already routed — mh1 #3 |
| 10 | null space, seg→pose direction | — | pose | **NO — subspace already owned** | §5 |
| 11 | waterfill | rate | seg | YES, blocked on a missing curve | already routed — mh1 A3 |
| 12 | tail-targeting | pose (QA43/`pu2`) | seg | already the default (top-k=120) | §5 |
| 13 | cosine / generic metric | — | any | **NO — raced and lost 4/4** | §5 (`na2` §7.4) |
| 14 | 8 generic-unraced defaults | `gd1` §2 | seg+pose+rate | YES | already routed — `gd1` |
| 15 | existence hinge (objective level) | seg (`p4x`, built not raced) | pose | pose leg OWED by p4x itself | already routed — p4x |
| 16 | blanket granularity lift | — | any | **NO — refuted per-role** | §5 (`na2` LAW B) |
| 17 | content-vs-solve discriminator | seg (`sq1`) | rate realization | plausible, unmeasured | OPEN, named §4 |

---

## §3 — Probes executed (all $0, scorer-free, from receipts already on disk)

### Probe A — the seg termination census, and the descent rate at truncation `[NEW]`

`os1`'s census law states its own reach honestly — *"0 of 21,700 `.omx/research` JSON receipts and
19 of 8,204 SSD receipts (0.23%) carry any iteration/evaluation count"*. The seg solver is inside
that 0.23%, and better: its rows carry `accepted_steps` explicitly, so the census is **direct, not
reconstructed** — stronger evidence than the pose site, where `os1` had to infer and refused on
14.7% of rows.

Source: `qa03_instances.jsonl`, **120 rows**; receipt `max_quanta_per_cell = 4`,
`net_flips_total = 1866`, `seg_S_delta = -0.0015818277994791665`, `verdict_scope: INSTANCE (this
endpoint b9a7983b, this scorer, this atlas aim)`. `stop_reason` postdates these rows, so class is
inferred exactly as the shipped resume path infers it (`sb1_seg_batch.py:264-266`).

```
class        n   flips   flip share   mean gain/step
cap         51    1207        64.7%            5.917
converged   57     659        35.3%            5.402
no_move     12       0         0.0%            0.000
```

Reproduces `dc1`'s two published numbers exactly (42.5%; 64.7%).

**The new quantity.** `dc1` measured a *count* (how many hit the cap). The pose-side metric is a
*rate* (how fast it was still descending). Transferred to seg, normalized by each instance's own
base so instance size cannot drive it:

```
depth stratum   n    mean per-step yield (% of that instance's own base flips)
     1         11        0.7786%
     2         27        0.6385%
     3         19        0.6099%
     4 CAPPED  51        0.6719%   <-- truncation happens HERE
```

**No saturation signature.** The capped stratum's per-step yield is **86.3%** of the shallowest
stratum's and *above* both intermediate strata. The solve was cut off while descending at
essentially its initial rate — which is precisely ss1 §4.3's MORE-SEARCH signature.

**Confound controlled:** mean `pair_base_flips` 888.7 (cap) vs 843.2 (converged) — only **5.4%**
larger, which cannot produce a 64.7% flip share; capped instances deliver **2.0×** the yield *as a
fraction of their own base* (2.688% vs 1.365%), which the normalized table already divides out.

**Honest limit:** the strata are different instances, so this is a *cross-stratum* comparison, not
a within-instance decay curve — the receipt stores per-instance totals only. The bias runs
conservative for the conclusion: instances that converged early are the ones that ran out of
moves, so they should if anything show *faster* decay, and they show none.

### Probe B — is the seg solver single-start, and is `qa04` a second start? `[NEW]`

Read at source, `tools/sb1_seg_batch.py`:

* **qa03** (`:199-270`): initial state is `rtp.codes` — the shipped base — and nothing else.
  `_best_single_quantum` (`:181-196`) evaluates all 8 of (4 channels × ±1), unit step only;
  acceptance is strict improvement; exit on `best[0] >= cur`. Textbook **greedy coordinate
  descent** ⇒ terminates at a **coordinate-wise local minimum**. For a CNN-argmax objective —
  non-separable by construction, and `pc2` measured the response as regional with Road in 87.8% of
  all flips — that is not a joint local minimum. Basin-trapping is **structural, not speculative**.
* **qa04** (`:321-381`): I expected `--n-prop 8 --seed` to be a multi-start. **It is not.**
  `rng.shuffle(moves)` then `moves[:n_prop]` *subsamples the same 8 unit moves*; at the default
  `n_prop=8` it evaluates all of them, identical to qa03's set. And there is **no step loop** —
  qa04 takes exactly one quantum. It is strictly weaker than qa03, not a second start. Recording
  the correction because the flag names read like multi-start and would mislead the next reader.

`dc1` had already named the sibling defect and left it open: *"the *greedy-vs-line-search* defect
is named and left unbuilt — it is a method change and must be raced."*

### Probe C — was the cure ever exercised? `[NEW — the §0.1 finding]`

A filesystem sweep finds **exactly one** QA03 run in existence, at `max_quanta_per_cell = 4`, and
no receipt anywhere carrying `cap_saturated_frac`. `dc1` named and costed the follow-up (*"uncap
QA03 and re-run the 51 censored instances on one scorer [job]"*); it was never fired. See §0.1.

This is also where `sv1`'s own later lesson lands unapplied: `sv1` measured, on pose and one day
after `dc1`'s raise, that a bigger constant is not the cure — *"Raising the ladder 4 → 12 and
relins 4 → 32 … the relin bound stops binding … but **`damp_cap` is still 60/60 = 100%**. The same
unfalsified question, one notch out."* `sv1` §1 then did the right follow-up **for its own axis**
(checking whether `pw1`'s replacement bracket saturates: no, 0/600, `CLOSED_INTERIOR_OPTIMUM`).
Nobody has done that check for the seg raise, because the run that would produce the evidence has
not happened.

### Probe D — `ms8`'s dead-codeword discriminator on the seg token codebook `[REPRODUCTION of ss1 §4.2]`

Executed before I had read ss1; reported as an independent reproduction, not a finding. Parsing
the qa03 base directly (`ddm_tr1_runtime.parse_archive`, **no SegNet loaded**),
`p2c_aimed_archive.zip`, sha `b9a7983b`, 767,812 B — verified to be the exact base the qa03
receipt names — `token_codes` is `(600, 24, 32, 4)` uint8, **1,843,200 tokens**:

```
idx     0      1      2      3      4      5      6      7
%   26.61   4.86   6.18   7.02   7.63   7.88   7.70   7.05
idx     8      9     10     11     12     13     14     15
%    6.00   4.92   3.90   2.99   2.12   1.51   1.01   2.62
```

**DEAD codewords 0/16**, mode share 26.61%; per channel 0/16 dead on all four, full support
`[0,15]`. **Agrees exactly with ss1's `token_quant_levels` 0/16.** Structurally the opposite of
`st_grid`'s six leading zeros.

*Vehicle-scope note recorded in passing:* the live-best archive
(`ddm_pu2_20260803/submission_pu2/archive.zip`, 353,805 B) **refuses** this parser —
`TR1RuntimeError: TR1 archive members/order differ`. The seg token solver operates on the
767,812 B TR1 base, not on the live v4d container. Any yield measured on it is
`verdict_scope: INSTANCE` on **that** base and does not compose with the live row without a
re-derivation. This is a real gate on §4 and I would rather state it than let the row look larger.

**The result cuts twice.** ms8's decomposition has two halves; on seg the PLACEMENT half is
measured clean, and **the SELECTION half is exactly what qa03 already does** — greedy re-selection
of each cell's codes against realized flips. So seg's only lever in that decomposition is the one
ms8 measured as the *weaker*, and we are already pulling it. That argues for making the selection
solver *better* (§4) and is the honest reason not to expect a large number from it.

---

## §4 — The one row, with cost-to-falsify

Priced against gap **0.6189279**; per-flip price **8.477105e-07 S/flip** (= QA03's own
`seg_S_delta / net_flips_total`, and independently equal to `pu2` §10's n600 figure). **No ΔS is
claimed** — this arm ran no scorer.

### Uncap **and** multi-start the seg token solve, as one run `[HYPOTHESIS-ON: seg]`

Transfers #1 and #2 land on the same run, so this is one experiment with three arms and a control
stratum, not three experiments: **(A)** cap 32, single start — isolates MORE-SEARCH, and is
`dc1`'s own owed re-run; **(B)** cap 32, multi-start — isolates PLACEMENT-as-re-initialization;
**(C)** cap 4 — reproduces the shipped 1,866 flips and proves the harness is unchanged.

*Measured, not claimed:*

* **LOWER bound**, from the measured capped-stratum rate, one further step each:
  `51 × 5.917 = 301.8` flips = **2.558e-04 S = 0.0413% of gap** (+16.2% on QA03's own yield).
* **Assumption-laden ceiling**, flat rate all the way to cap 32: `51 × 28 × 5.917 = 8,449` flips
  = 7.163e-03 S = **1.157% of gap**. *Not* a point estimate.
* **I refuse an upper bound.** The decay curve is flat, so the observable contains no saturation
  point; per the `os1` discipline the instrument should refuse rather than guess.

**This row is SMALL and is priced as small.** QA03's entire realized yield is 0.2556% of the
current gap, on a base that is not the live container (Probe D note). It earns its rank on
*pathology-present × cheapness* — measured pathology, cure already written and tested, three arms
— not on size. Ranked by gain it would be far down, and `pu2` supplies the cautionary precedent:
it cut its own multi-start forward estimate from 3.6–7.1% of gap to **~0.6%** once it ran a
non-tail control, verbatim *"should be priced as a low-value item, not a headline."*

**Two design requirements inherited from sister arms, both load-bearing:**

1. **A non-tail control stratum is mandatory.** `pu2` measured the multi-start defect as
   **TAIL-SPECIFIC**, not population-wide (tail median ratio **0.2013**, 5/6 improved; non-tail
   control **0.9388**, 2/6) — and `ms8` has no control-group analogue, which is exactly the gap
   that let its forward estimate stand too high. QA03 already selects a top-k=120 tail by atlas
   rank, so **without a non-tail stratum this run cannot distinguish "multi-start works" from
   "the tail is where everything works."**
2. **Seg-only is INADMISSIBLE.** Per `sq1`/`p4x`/`m85`/#889, any reading here needs a matched-base
   pose control ≥32 pairs. `sq1` measured seg edits to frame_1 at **56.5×** pose collateral
   unconstrained, bought back to **1.039×** only under the rank-6 yuv6-null projection with a
   31.5% eta tax — and that projection **does not commute with a pixel-granular mask** (two of the
   six constraints are block-mean), so the band must snap to whole 2×2 blocks.

**Cost to falsify:** one scorer job; the shipped cap-4 run over all 120 instances took
**1,593.9 s**, and `dc1` costed the uncap at 8 SegNet evals/step over only the 51 censored
instances. **Pre-registered falsifiers:** arm B ≤ arm A refutes the multi-start transfer to greedy
coordinate descent on an integer lattice; arm A ≤ arm C + noise refutes the uncap and vindicates
cap = 4 retroactively; **tail-vs-control ratio ≥ 0.9 refutes tail-specificity** and would make the
row population-wide instead. **Fire condition:** n600 scorer slot released by `ob1`.

**Also open, named not claimed:** carry `sq1`'s CONTENT-vs-SOLVE discriminator to the rate axis
(`na2` LAW B: *"the discriminating axis is not granularity — it is CONTENT vs SOLVE"*, derived
from η −3.7640 vs +0.7895 on the same band, same addresses, same bytes). The rate-side form —
*where do we still ship stored CONTENT that a SOLVE against the frozen head could regenerate?* —
has never been asked generally; mh1's *"code the generator not the flip set"* orphan is one
already-routed instance. $0 to enumerate, scorer-gated to price.

---

## §5 — Pathology ABSENT (a deliverable, not a shortfall)

Nine transfers are **closed**, which is the point: each removes a plausible direction from the
queue. Six were closed by sister arms and are cited, not re-derived; three I closed here.

1. **Menu placement refit → seg `token_quant_levels`.** ABSENT: **0/16 dead** (ss1; reproduced
   independently here, Probe D).
2. **Menu placement refit → `selector`, `rs_beta_mags`, SMEVR mode-base.** ABSENT: 0/2, 0/13
   (zero *by construction*), 0/16. *"no placement refit is available anywhere else on this
   vehicle"* (ss1 §4.2).
3. **The 5.7× ratio as a steering law.** ABSENT. *"Carry the diagnostic; discard the ratio"*
   (ss1 §0). The residual selector re-selection is **−0.000748 at zero bytes, 6/600 pairs =
   0.096% of gap** — tidy-up scale.
4. **Degeneracy refit → any other menu.** ABSENT and *explained*, not merely observed: dc1 §7
   derives from the receiver algebra why exactly one menu can be degenerate (`beta` multiplies the
   rotation triple but the receiver applies **two** scales `1 ± β/2` blended by row, so no single
   rescale of `p[3:6]` reproduces it).
5. **Truncated solve → `terminal_pose_gn` #850.** ABSENT: cured and off the live chain, verified
   by three arms.
6. **Proxy-vs-realized accept test as an open question.** ABSENT: measured by ss1
   (REAL 44 / PROXY 4 / MIXED 4). **My queued row is retracted.**
7. **Cosine / generic metric → any axis.** ABSENT as a race: lost 4/4, and it is the standing
   law's own first instance. The live item is a **$0 registry landing** — `canonical_anti_patterns`
   has *"0 registered classes"* for the generic-metric class (`na2` §2.2 G3), already named there.
8. **Null space, seg→pose direction** (pose edits placed where SegNet is blind). ABSENT because the
   subspace is already owned and already used: SegNet reads only the last frame (`x[:, -1, ...]`),
   so frame_0 *is* the seg-null subspace, and CLAUDE.md already routes it — *"frame_0 is
   structurally seg-free (d_seg obligation 8.5e-9) — a cheaper PLACE for joint-trained pose
   output."* Both scorers share the same `D` (`pz1`), so the 22.70% `D`-blind set is invisible to
   *both* and is a **rate** resource, not a seg or pose one — which is why transfer #9 routes to
   rate. `pu2` §1.1 confirms the asymmetry from the other side: the pose-knob route is
   `d_seg`-neutral **by construction** (all five knobs read only inside `Decoder.f0()`), verified
   end-to-end at `d_seg 0.00431179 → 0.00431179`, bit-identical.
9. **Blanket granularity lift.** ABSENT / refuted per-role: `na2` LAW B assigns objective →
   regional, address → supra-pixel, **actuator → per-pixel is CORRECT**, warning *"Banning
   per-pixel actuation would kill the best realizer we own."* `p4x`'s hinge is legitimate precisely
   because it lifts the **objective**, the one role where the lens survives.

---

## §6 — Routing

Landed as rows in `canonical_task_status.jsonl` via the canonical writer, per mh1:288 and `m89`
(cite CONTENT, never bare ids). **No ΔS is claimed on any row.** One row registered earlier in
this arm's run was **retracted in-ledger** once ss1's measurement was read, rather than left to
look open.
