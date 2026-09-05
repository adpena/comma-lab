# ddm_hc2 — the ~76,600 B "WHERE are the flips" half of the token stream, decomposed: **the flip set does not cluster**, and the boundary-offset premise is true but already priced. CEILING-REFUSED.

**Arm:** ddm_hc2 (Fable, 2026-09-05), charter `.omx/research/charters/ddm_hc2_wrong_half_flip_location_decomposition_20260905.md`.
**Pointer at spawn and at close:** fs2 — S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600], archive sha
`a8f3a3791499b2b62ee4d16bc67f15f819f454dc9b88e3cce04fe50a30427bb6`. **The exact pointer did not move.** This arm bought
none of the demand; it was chartered to price a door, and it priced it shut.

**Typed verdict: `CEILING-REFUSED`.** No representation of the flip LOCATION SET beats per-site indicator coding.
Against the charter's comparator (the rows' per-site sum on the flip sites, **76,266.24 B**) every representation loses
by **at least +30,289.62 B**. Against the full indicator it would actually have to replace (**110,909.03 B**) the best —
an **acausal** boundary-offset band — saves **4,353.17 B**, still under the 5,000 B bar. The **causal** clustering family
bound is **+23.82 B**, 210× under it. Scope: **FAMILY** (location-set representations of the flip set on this object).

Axis: `[macOS-CPU advisory / scorer-free EXACT byte measurement]` for the control; `[model-ledger code length on the
coder's own rows; REFUSAL-ONLY]` for every ceiling number. `score_claim=false`, `promotable=false`. No coder was built,
no archive, no scorer, no Modal, no Metal. $0, one CPU process, measured peak 6.38 GiB. Labels MEASURED / DERIVED /
INFERRED as marked. Craft per `docs/operating_manual_craft_handoff.md` (§4 re-derive, §5 label, §6 attack your own
conclusion, §7 answer first).

---

## 1. The instrument, and the control that lets me trust it

The object is the shipped fs2 token stream. `ddm_mc1` retained the coder's OWN per-position coding rows
(600 × 196,608 × 5 float32, 2.36 GB). The charter told me to re-verify mc1's byte-identical re-encode before trusting
them. I verified it **more strongly than a re-run would**, and cheaper (MEASURED, `CONTROL.json`):

| control | result |
|---|---|
| sha256 of `coding_rows.f32.npy`, `base_argmax.u8.npy`, `boundary_bucket.u8.npy`, tokens, `control_stream.bin` | **all five match mc1's `ROWS_RESULT.json`** |
| mc1's emitted stream inside the **shipped archive** (`archive.zip` sha `a8f3a379…`, member `p`) | **found byte-identical at offset 66,512, length 113,411** |
| ideal code length recomputed from the rows over all 600 pairs | **113,410.8702 B** vs mc1's recorded 113,410.8557 B — **Δ 0.0146 B** (1.3e-7 relative) |
| live (float32-unsaturated) positions | **50,009,121** — exactly mi1's and mc1's independently derived count |
| flips at saturated positions | **0** |

Re-running the encoder would have proved "these rows re-emit the stream". Finding the retained stream *inside the
shipped archive* proves "these rows re-emit **the shipped block**", which is the claim that matters. The 20-minute
re-encode was therefore not repeated — stated plainly because it is a deviation from the charter's literal wording.

**The flip definition, and a correction to a tempting shortcut (MEASURED).** hc1's split is
`-log2(p_sel) = -log2(pmax)` when the argmax is right, `-log2(1-pmax) + -log2(p_sel/(1-pmax))` when it is wrong. For
that identity to be exact, "argmax" must be the argmax **of the coding row**. mc1's retained `base_argmax` is the argmax
of the **pre-corrector** logits (`ddm_mc1_...py:221`, before `parts.table.values[feature]` is added) and disagrees at
**32,449 of 117,964,800 positions (0.0275 %)**. Using it gives 2.7062 bits/flip; the coding-row argmax gives
**2.6812**, against hc1's 2.6917 on the dx2 body. I use the coding-row argmax, tie-safely
(`row[token] < row.max()`), and record the other as a control.

---

## 2. The decomposition — what the wrong half is made of (MEASURED, all 600 pairs)

| term | bytes | share of the 113,410.87 B stream |
|---|---:|---:|
| **INDICATOR — "is my argmax right?"** | **110,909.07** | **97.79 %** |
| · "yes" branch — confirmation | 34,642.82 | 30.55 % |
| · **"no" branch — the WHERE** | **76,266.24** | **67.25 %** |
| CONDITIONAL — "then which of the other four?" | 2,501.80 | 2.21 % |
| flips | 227,555 of 117,964,800 | 2.6812 bits/flip |

hc1 measured 111,275.62 / 34,674.08 / 76,601.54 / 2,500.54 on the **dx2** body; the fs2 body agrees to **0.3–0.4 %** on
every term. mc1 independently recorded a base indicator of **110,909.01 B** on these same rows; my float64
re-accumulation gives **110,909.07 B** — 0.06 B apart, two implementations. The instrument is where I thought it was.

### 2.1 The geometry — the flip set does not cluster

Connected components of the per-pair 384×512 flip mask (MEASURED, `components_8conn.npz`):

| connectivity | components | sites | mean | median | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| **8** | **172,193** | 227,555 | 1.3215 | 1 | 2 | 5 | **21** |
| 4 | 192,719 | 227,555 | 1.1808 | 1 | — | — | 17 |

8-connected size distribution, with the incumbent's own no-branch cost on each bucket:

| size | components | sites | incumbent bytes |
|---|---:|---:|---:|
| 1 | **139,640 (81.10 %)** | 139,640 (61.4 %) | 46,717.62 |
| 2–3 | 27,248 | 61,655 | 20,961.13 |
| 4–7 | 4,981 | 23,152 | 7,614.16 |
| 8–15 | 315 | 2,948 | 921.28 |
| **16–31** | **9 (0.0052 %)** | **160 (0.070 %)** | **52.05** |
| ≥ 32 | 0 | 0 | 0.00 |

**The charter's prior law predicted the saving would concentrate in "the ≤ 20 % of components with ≥ 16 sites". There
are nine of them in the entire 600-pair body** — 0.0052 % of components, 0.070 % of the sites, 52.05 B of 76,266.24 B.
The premise is wrong by ~3,800×, and it is wrong *before any coder is priced*.

The reusable quantity is the **address-count ratio**

    rho = components / sites = 172,193 / 227,555 = 0.7567,

so a component code removes only **24.33 %** of the address events the per-site code pays, and must buy the 55,362
absorbed sites back with a shape code.

**Honest boundary on that arithmetic.** It does NOT by itself refuse the arm: 55,362 absorbed sites at the incumbent's
mean 0.33515 B/site is **18,554 B** of optimistic headroom, *above* the 5,000 B bar. The geometry made the arm
necessary; the measurement decided it. Anyone quoting rho as the closure is overstating what it settles.

### 2.2 The boundary-offset premise is TRUE

Chebyshev distance from each flip to the nearest pixel of its *coded* class in the mixer's argmax field (MEASURED):

| distance | 1 | 2 | 3 | 4 | 5 | 6 | 7 | ≥8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flips | **225,339** | 705 | 283 | 211 | 137 | 87 | 68 | 725 |

**99.03 % of flips are one-pixel boundary moves.** And 99.45 % of them lie in the D=1 band — 3,077,386 positions,
**2.61 % of the field**. The premise the charter's (b) rests on is real.

**And the incumbent has already applied it.** Of its 110,909.07 B, only **5,338.20 B (4.81 %)** is spent outside that
band (3,193.22 B outside D=2; 2,311.51 B outside D=3). A band restriction the model already performs is not new side
information — and the offset field's *support* IS the band, so deciding which band position carries a nonzero offset
IS the indicator. **(b) renames the question; it does not reduce it.**

---

## 3. The ceilings (MEASURED; held-out, pair-level two-fold, seeds 20260905/777/31337, reported at the value MOST FAVOURABLE to the alternative)

Two comparators are reported, and the difference matters. **(c-narrow) = 76,266.24 B** is the charter's: the rows'
per-site sum on the flip sites, which is what hc1 and MAIN call "the WHERE". **(c-full) = 110,909.03 B** is the whole
indicator, which is what a location-set code actually replaces — the "yes" branch is the same question asked at
non-flip positions. c-full is the more generous of the two, so a refusal against it is the stronger statement.

| representation | bytes | vs c-narrow | vs c-full |
|---|---:|---:|---:|
| **(a1)** raster-gap seed code + shapes | 231,091.54 | +154,825.29 | +120,182.50 |
| **(a2)** mixer-conditioned seed field (KT per q-bin, 64) + shapes | 138,873.45 | +62,607.20 | +27,964.41 |
| **(a2b)** β-per-cell seed field (q32 × d, full-resolution logit) + shapes | **128,683.87** | **+52,417.63** | **+17,774.84** |
| **(b)** D=1, geometry only | 148,942.78 | +72,676.54 | +38,033.75 |
| **(b)** D=1, geometry × q32 | 108,396.72 | +32,130.48 | −2,512.31 |
| **(b)** D=2, geometry × q32 | 107,113.13 | +30,846.89 | −3,795.90 |
| **(b)** D=3, geometry only | 147,418.28 | +71,152.03 | +36,509.24 |
| **(b)** D=3, geometry × q32 — **best of everything** | **106,555.86** | **+30,289.62** | **−4,353.17** |
| **(c)** incumbent | 76,266.24 / 110,909.03 | — | — |

Seed-to-seed spread is ≤ 87 B on (a) and ≤ 35 B on (b); the ordering never changes.

**(a) splits cleanly, and the split is the mechanism.** The a2b component code is **88,853.80 B of seed field +
39,830.07 B of shape code** (1,553 distinct shapes). Recognising 172,193 *seeds* costs 80.1 % of what recognising
227,555 *flips* costs, because 81 % of the seeds are the whole component. Then the shape code — 1.85 bits/component —
is spent buying back sites the incumbent priced at 0.335 B (2.68 bits) each.

**Per-component attribution of the clustering gain** (incumbent cost on the component, minus its seed's own cost,
minus its shape bits — the exact quantity the clustering is supposed to earn):

* total **−23,754.00 B**. The clustering is a net **loss**.
* components with ≥ 16 sites contribute **−54.88 B**, i.e. **0.231 %** of it, and with the wrong sign.
* only **2,326 of 172,193 components (1.35 %)** show any positive gain at all.

**Robustness — the verdict does not depend on my shape model (DERIVED + MEASURED).** The zeroth-order entropy of the
observed shape distribution, in-sample, with no cross-fit and no escape symbol, is **1.6191 bits/component =
34,849.39 B** — only 4,981 B below what I actually charged. So the shape code is already near its own floor, and even
at that floor (a2b) costs 123,703.19 B: **+47,436.95 B vs c-narrow, +12,794.16 B vs c-full.** Pushing further, if the
shape code were **entirely free**, the seed field alone (88,853.80 B) *still* loses **+12,587.56 B** against c-narrow.
It would beat c-full by 22,055 B — so the one reading under which a component code could ever win requires *both*
free shapes *and* the generous comparator, and the measured shape entropy forecloses the first. (A hard floor
independent of the dictionary: distinguishing "singleton" from "not" is 0.6995 bits/component = 15,055.67 B on its own.)

---

## 4. The family bound — what the causal neighbourhood is worth (MEASURED)

The explicit codes above price three *specific* representations. This prices the whole family. Instrument: mi1's
`q' = σ(logit(1-pmax) + β_cell)` on the indicator, which nests the shipped model exactly at β = 0, Newton-fit,
cross-fitted by pair, saturated positions excluded. Any representation that exploits only the local clustering of flips
induces a conditional law over the indicator given the causal context, so the best member of this family at that context
resolution **upper-bounds** it. Held-out bytes saved against the 110,909.03 B indicator, 3-seed minimum:

| cell | what it is | cells | held-out B |
|---|---|---:|---:|
| `none` | pure recalibration — the noise floor | 1 | **+2.37** |
| `q32` | recalibration per confidence bin | 32 | −32.64 |
| **`pat4`** | **the 4 raster-causal neighbours' flip bits** | 16 | **+23.82** |
| `pat8` | 8 causal neighbours | 256 | −3.39 |
| `q32 × pat4` | both | 512 | −213.91 |
| `q32 × pat8` | both, richer | 8,192 | −403.70 |
| `d_other` *(ACAUSAL)* | current-frame boundary distance | 5 | +4,147.72 |
| `q32 × d_other` *(ACAUSAL)* | both | 160 | **+5,823.45** |

**The entire causal spatial-clustering signal beyond the mixer's own probability is +23.82 B on 110,909 B — 210× under
the bar, and one order of magnitude above the instrument's 2.37 B noise floor.** The mixer's 7×7 masked receptive field
has already consumed it. mi1's oracle-mirage signature reproduces exactly: held-out collapses as cells grow
(+23.82 → −213.91 → −403.70) — anyone quoting an in-sample number on this axis is quoting noise.

Two routes, one answer: the explicit component code loses by 17,775 B, and the conditional bound over the same
information is worth 24 B. That agreement is why the scope is FAMILY and not merely formulation.

---

## 5. The one cell that clears the bar, and why it is not a candidate

`q32 × d_other` saves **+5,823.45 B** — 1.16× the bar. It is not a candidate, and the reason is not taste:

* **`d_other` is acausal.** It is the Chebyshev distance from a position to the nearest *different-class* pixel of the
  **current frame's** argmax field. The decoder decodes in groups; when it codes position *i* it does not know the
  argmax of the positions it has not yet reached. This is exactly mc1's `oracle_block` shape (+3,420 B on information
  the receiver cannot have).
* **The causal version is already measured and drained.** mi1 priced the shipped causal boundary feature
  (`_boundary_buckets(frame t−1)`, in the stack) at **+5.27 B** held-out — **1,105× smaller**. The value lives entirely
  in the acausal half.
* (b)'s best row is the same feature under a band restriction, which is why it lands at 4,353.17 B rather than 5,823 B:
  restricting to the band throws away part of the acausal signal.

The charter's rule was "if ANY representation clears 5,000 B, write the next-step charter's price." **None does.** The
best *representation* is 4,353.17 B (87.1 % of the bar) against the more generous comparator. I price the near-miss
below anyway, because leaving a 5,823 B diagnostic unexplained would be the orphan, and then I recommend against it.

**Priced, NOT queued — a within-frame partial boundary distance.**

* **Representation.** Extend the corrector's feature from `d(frame t−1)·5 + predicted` to also carry
  `d_partial` = the boundary distance computed on the *already-decoded groups of the current frame*. Legal: it reads
  only decoded tokens.
* **Ceiling.** Bounded above by **5,823.45 B** (the full-field version). Nearest existing evidence for the "how much of
  this frame is decoded" axis: mi1's `groupbin8` **+64.20 B** and `group190` **+51.25 B** — both ~90× under the bar.
* **Decode cost (DERIVED).** My full-field version costs ~14 ms/pair for five chessboard transforms. Per decode group
  (190 groups/pair) that is ~2.7 s/pair × 600 pairs ≈ **27 min**, which alone nearly exhausts the 30-minute budget. An
  incremental distance update would have to be built first.
* **What changes in the shipped stream.** The corrector table's feature cardinality grows from 25 to 25·(D+1); model
  bytes grow by the table delta, charged against a ceiling that is already only 1.17× the bar.
* **Recommendation: do not charter it.** Ceiling 1.17× the bar, causal fraction measured near zero by two independent
  proxies, decode cost at the budget wall. Recorded as a LIVE-HYPOTHESIS with its price.

---

## 6. Verdict against the pre-registered falsifier

The charter: *"FALSIFIER: the component-level ideal codelength (with its own side-information counted: component count,
seeds, shapes) is within 5,000 B of the per-site sum → the wrong half is already priced at its structural floor by the
mixer → hc1's half is CLOSED at FORMULATION scope and the rate corner has no representation door left on this object."*

**The falsifier FIRED, and in the stronger direction than it was written for.** It anticipated a near-tie; what was
measured is that every representation is **substantially worse** than the per-site sum — the best by +30,289.62 B, the
best component code by +52,417.63 B. The prediction of a 15–30 % saving (≥ 11,000 B) is falsified by sign as well as
magnitude.

**Scope: FAMILY, not formulation** — stronger than the charter asked for, and here is the exact boundary of that claim.
Two independent instruments cover the family: (i) the explicit codes carry arbitrary-length shape information and lose;
(ii) the conditional bound over the causal neighbourhood is +23.82 B. What is **not** covered: representations whose
extra information is neither local clustering nor the mixer's own probability — in particular the address law's named
escape, a **joint field+model move where the model refit IS the addressing**
([[perfect-localization-is-worthless-the-address-is-the-tax]]). This arm did not touch that, and nothing here closes it.

**This is the fourth independent instance of the address law**, on a fourth surface: mf1 (manufactured seg error,
+35,969 B address payload), tb2 (bit mass and task mass are the same 0.2 % of positions), tba1 (naming any subset costs
more than the subset holds), and now hc2 — *clustering* the addresses is not an escape from naming them, because the
clusters are 81 % singletons and the shape token is itself an address.

---

## 7. RECALL EVIDENCE (consumed, not re-derived)

* `token-stream-is-one-binary-question` / `ddm_hc1_hpac_calibration_reliability_20260824.md` — the split this
  decomposes; its 111,275.62 / 76,601.54 / 2,500.54 reproduce here to 0.3–0.4 % on a second body. Its warning that the
  circulating `d` concentration is **acausal** (d=0 carries 49.85 %, not 94.53 %) is exactly the trap §5 avoids.
* `ddm_mc1_motion_compensated_previous_plane_20260904.md` — the rows, the control, and the oracle-vs-derivable
  discipline. Its `oracle_block` (+3,420 B, not decoder-derivable) is the same shape as my `q32 × d_other`.
* `ddm_mi1_indicator_model_axis_20260824.md` — the β-per-cell instrument, the 2.37–2.77 B noise floor, the
  oracle-mirage signature, and the two numbers that price §5's next step (`groupbin8` +64.20 B, `boundary_d` +5.27 B).
* `perfect_localization_is_worthless_the_address_is_the_tax_20260824.md` ([[m118]]) — the law this extends; ra3's
  per-flip correction carrier is closed by the same arithmetic.
* `ddm_df1` — the *right* half's address floor at 3.15× the prize; hc2 is the left half's.
* `ddm_gs4_campaign_state_after_the_post_submission_wave_20260905.md` §3/§5 — the 5,000 B bar and the closure ledger.
* `[[m88]]`/`[[m96]]` — why every number here is n600 with pair-level folds and never a prefix.

## 8. Custody (ALWAYS KEEP THE PAYLOAD)

Store `/Volumes/APDataStore/pact/ddm_hc2_wrong_half_decomposition/`:

| artifact | bytes | sha256 (16) |
|---|---:|---|
| `components_8conn.npz` — per-component pair/size/seed/cost/seed-cost/bbox/shape-id, all 172,193 | 3,161,338 | `5e2eb7e979605b9e` |
| `features_live.npz` — per-live-position logit, indicator bits, flip, pair, causal patterns, boundary distance, argmax, seed/covered flags (50,009,121 rows) | 950,175,759 | (retained) |
| `seed_gaps.i64.npy` — raster gaps between consecutive component seeds | 1,377,672 | `57d7eca38ed6f3fe` |
| `FEATURES.json` — the §2 decomposition and geometry | 2,829 | `902e3a4107100b46` |
| `CEILING.json` — every ceiling, per seed | — | (retained) |
| `CONTROL.json` — the §1 controls | 2,425 | `95cb0245fb0b676b` |
| `PER_PAIR.json` — per-pair flips, branch bytes, component counts | 99,881 | `2632e3d39f516f72` |
| `launch/` — `launch_manifest.json`, `run.log`, resource status | — | — |
| code | `experiments/ddm_hc2_wrong_half_flip_location_decomposition.py` | — |
| lane | `lane_ddm_hc2_wrong_half_decomposition_20260905` | — |

Reproduce: `.venv/bin/python experiments/ddm_hc2_wrong_half_flip_location_decomposition.py --stage all --pairs 600`
(control + features + ceiling, 631 s wall, peak 6.38 GiB, one CPU process). Launched through
`tools/launch_detached_process.py --done-receipt hc2_ceiling --derive-resource-budgets --measured-peak-rss-gib 8.0
--measured-thread-need 2 --walltime-cap-s 5400 --artifact-budget-gib 6.0`. Declared peak from a 20-pair dry pass with
the accumulators fault-in at full 50 M capacity: **1.093 GiB**; the ceiling stage's float64 working set took the
measured full-run peak to **6.38 GiB**, inside the declaration.

## 9. What I did NOT do (plainly)

* **No coder, no receiver, no archive, no scorer run, no Modal, no Metal.** The charter said closed-form ceiling only.
* **I did not re-run mc1's 20-minute encoder.** I verified the retained stream byte-identical *inside the shipped
  archive* plus all five sha256 and the ideal-length reconciliation to 0.0146 B (§1). That is a stronger claim about
  the same fact, but it is not what the charter's wording said, so it is named here.
* **I did not price 4-connectivity ceilings**, only its geometry. 4-conn has *more* components (192,719 vs 172,193),
  so its address-count ratio is worse and every ceiling would be strictly higher.
* **I did not build a causal partial-boundary feature.** §5 prices it and recommends against it; it is unmeasured.
* **I did not test moves outside the location-set family** — in particular the joint field+model escape. §6 states that
  boundary explicitly rather than letting the FAMILY scope imply more than it covers.
* **The ceilings are generous to the alternative by construction** (raster decode order for the component code, which
  the shipped tile/group order does not provide; acausal band geometry for (b); cross-fitted models charged no model
  bytes). Every one of those inflates the alternative, so the refusal is stronger than the bare numbers; an *admission*
  would have needed a receiver.

## 10. LIVE-HYPOTHESES / DEAD-ENDS / NEXT_IF_RESUMED

**DEAD-ENDS (with numbers):**
* Component / run-length / chain-code representation of the flip set: best 128,683.87 B vs a 76,266.24 B incumbent
  (+52,417.63 B); clustering gain **−23,754.00 B**; ≥16-site components contribute −54.88 B. Ideal-shape floor
  34,849.39 B still loses; free shapes still lose against c-narrow by 12,587.56 B. CLOSED.
* Boundary-offset representation: best 106,555.86 B (+30,289.62 B vs c-narrow, −4,353.17 B vs c-full, under the bar),
  and its band geometry is acausal. CLOSED.
* Causal spatial clustering as a conditioning axis: +23.82 B (`pat4`), −403.70 B at 8,192 cells. DRAINED.

**LIVE-HYPOTHESES (named, priced, not tested here):**
* Within-frame partial boundary distance from already-decoded groups (§5): ceiling ≤ 5,823.45 B, nearest proxies
  +51 to +64 B, decode cost ~27 min. Priced; recommended against.
* The address law's escape — a joint field+model move whose refit *is* the addressing — is untouched by this arm and
  is not closed by it.

**NEXT_IF_RESUMED:** nothing on this door. hc1's wrong half is the largest block in the archive and it is at its
structural floor for anything that has to *name* where the flips are. The demand (−41,817.8 B at held distortion) is
elsewhere; see `ddm_x012`'s door map.

## Equations leg (`tac.canonical_equations`)

Registered as **`flip_location_component_address_floor_v1`**
(`src/tac/canonical_equations/flip_location_component_address_floor_20260905.py`, exported from
`tac.canonical_equations`; re-derivation guards in
`src/tac/tests/test_ddm_hc2_flip_location_component_address_floor.py`, 10 tests). The law: a clustering representation
of a sparse flip set pays only when the set clusters, and its whole structural budget is the **address-count ratio**
`rho = components / sites` — `component_representation_headroom_bytes` returns the optimistic bound
`(sites − components) · incumbent/sites`, `component_representation_can_pay` is the (necessary-only) gate,
`boundary_band_is_already_localised` refuses a band restriction the coder has already performed, and `ceiling_refused`
applies the charter bar. Three empirical anchors, all VERIFIED_VIA_EMPIRICAL_ANCHOR: component geometry (residual
0.19995 = the 20 % premise minus the measured 0.0052 % share), boundary band (residual 0.0481 = the out-of-band
indicator share), coder ceiling (residual 6,646.83 B = the prior law's 11,000 B minus the best measured 4,353.17 B).
Predicted vs empirical: predicted ≥ 11,000 B saved, empirical −4,353.17 B at best against the generous comparator and
+30,289.62 B *lost* against the charter's.

## Frontier line

fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600] — unchanged. Candidate line: none (CEILING-REFUSED;
advisory only, nothing READY-FOR-T4).
