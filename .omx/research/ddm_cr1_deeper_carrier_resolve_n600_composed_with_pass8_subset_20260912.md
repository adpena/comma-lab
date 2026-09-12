# ddm_cr1 — the n600 carrier re-solve moves ONE pair of 600, and 504 of those pairs had already been measured, inside pass 8, hours earlier

Tokens: `[no-triality] [p0-ledger-ok]`

Lane `ddm_cr1_deeper_carrier_resolve_n600_composed_pass8_20260912`. Charter
`.omx/research/ddm_cr1_deeper_carrier_resolve_n600_composed_with_pass8_subset_charter_20260912.md`
(commit `8dbadf13c`, sha `2b697a79a7a1e243…`). Base: **move 49, S 0.13632299781031237 @ 179,153 B**,
archive sha `73e41a6620bd4ea3aaf236eff9de46391857907527358e8eb40ded0925a1c214`. Pose axis
`[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT, n600]`; **row A's bytes are EXACT**
through the shipped container, rows B/C carry pass 8's bit-ledger model. **No score is claimed. The
pointer did NOT move, no candidate was sealed, and this arm fired nothing.** No Modal, no
`authorize_*`, no fire, no packet.

---

## 1. The answer first

**MEASURED, n600, all 600 pairs solved: the re-solve moves ONE pair. Two pairs show any gain at all;
one of those two returns the codes it started from.** The whole population gain is
**4.0261e-07** of a d_pose sum of 2.7261e-03 — **0.01477 %** where the charter's falsifier fires below
**0.15 %**. **Falsifier #1 FIRES by 9.99×.**

| | MEASURED |
|---|---:|
| pairs solved | **600 / 600** |
| pairs with any in-loop gain | **2** |
| pairs whose CODES moved | **1** (pair 391; 2 of 12 coordinates, ±1) |
| pairs stopping in outer round 1 | **598** (`no_improving_step` 574, `lattice_floor` 26) |
| in-loop gain, population sum | **4.0942881e-07** (of which 4.0261162e-07 is pair 391) |
| n600 mean d_pose, batch 8 | **4.543568679770593e-06 → 4.5428915620277605e-06** |
| fraction of the base | **−0.014903 %** |
| **exact archive** | **179,153 B**, sha `3d94e5fd0acc9563…`, **Δ 0 B**, twins agreeing |
| ΔS, like-for-like | **−5.022869e-07 = 0.0251 of the −2e-05 bar** |

The charter's own arithmetic expected **−1.08e-05** from this rung. The measured number is **21.6×
smaller**, and §3 shows exactly why the expectation was wrong: it applied a RATIO estimator to a
48-pair sample whose base mean is **0.554×** the population's.

**And the harder finding, which is about the apparatus rather than the carrier.** Pass 8's own
`refine` stage ran `jg5.refine_pair` from the shipped codes over **all 600 pairs**, 504 of them on
renders its edit set never touched — i.e. on move 49's own renders, which is exactly this rung. This
arm reproduces it on **504 of 504 pairs, final d_pose AND all twelve codes bit-identical**, same two
improvers, same 4.0943e-07. **The "unowned ITEM 4" had already been measured, in the same store, by a
stage that was filed under an admission rather than under a carrier re-solve.** cb1 then re-found a
48-pair slice of it and reported it as an open zero-byte lead; the charter budgeted this arm on that
lead. Nothing was wrong at any step — but the number existed the whole time. Genus:
`[[proactive-recall-consult-own-research-before-concluding]]`, at the layer where a measurement is
orphaned by the NAME of the stage that produced it, not by its absence.

**The composition does not net, and neither does anything it contains.** Composed like-for-like
**−9.9756e-06 = 0.499 bars**; composed at pass 8's own nominal convention **−1.4744e-05 = 0.737
bars**. Per the charter: neither nets, so this arm closes with the table and builds nothing.

---

## 2. Base and controls — five, all PASS, before any claim

| control | what it refuses | result |
|---|---|---|
| **C1** pose base | a base inherited rather than reproduced | my n600 mean **4.543568679770593e-06** is **bit-identical** to pp1's move-49 vector: max abs gap **0.0** over 600 pairs, against pp1's measured band gate 2.586e-08. Ratio to the T4 print 4.55e-06: **0.998587** |
| **C2** twins | a byte delta that is the rebuild's, not the re-solve's | repacking move 49's own carrier through `up3.build_archive` returns `73e41a66…` / 179,153 B exactly, **and the double compile is identical both times** |
| **C3** codes identity | rendering from codes the archive does not store | the instrument's `state.codes` equal the archive's parsed `body.codes`, all 600 × 12 |
| **C4** unchanged pairs | attributing measurement drift to the re-solve | the 599 pairs whose codes did not move reproduce their own base at batch 8 with max abs gap **0.0** |
| **C5** frame-1 section identity | a seg leg carried across a body that moved | hpac, semantic and tail sections byte-identical to move 49; only the carrier section differs, and it is the same **18,450 B** |

C5 is what makes the seg leg legitimate: the carrier renders frame 0 and SegNet reads the last frame
(`upstream/modules.py:108`), so d_seg is unchanged by construction — and then the bytes prove it.

---

## 3. Why the charter expected 21.6× more, and which estimator was wrong

cb1's keep-12 control measured a realized ratio **0.99679** on a seeded-random n48 and reported it as
"0.32 % of d_pose at zero bytes". The charter converted that to **−1.08e-05 S**. Both statements are
faithful to cb1's numbers. The conversion is the problem.

MEASURED, by re-reading cb1's own rows: **all 48 of its pairs improved by exactly zero except one —
pair 391, gain 4.0261e-07, two coordinates moved.** (Its LS-projection warm start also reproduced the
shipped codes on all 48, so cb1's control and this arm's start-from-shipped are the same experiment;
that is why pair 391 reproduces to the last digit.)

| estimator | what it assumes | implied ΔS | against MEASURED n600 |
|---|---|---:|---:|
| **ratio** — population ratio = sample ratio | the sample's base mean represents the population's | **−1.0873e-05** | **21.8× too large** |
| **gain-sum** — per-pair mean gain × 600 | gains are exchangeable across pairs | **−6.2247e-06** | **12.5× too large** |
| **MEASURED n600** | nothing | **−4.9776e-07** | — |

The ratio estimator fails because the n48 sample's base mean is **2.5172e-06 = 0.554×** the
population's 4.5436e-06 (the top-12 pairs carry 52.5 % of the mass and a 48-pair draw mostly misses
them), so dividing a gain by a small base inflates the percentage. The gain-sum estimator fails more
mildly, and for the honest reason: **pair 391 is not one improver among many — it is the only
code-moving improver in the entire population**, so any per-pair average built on it over-counts by
600/48 = 12.5. cb1 flagged the sample-base problem for its refit rows and did not carry the flag into
its control's headline; this arm is the measurement that settles it.

This is `[[binding-instruction-numbers-expire-and-nobody-rederives-them]]`'s sister at the ESTIMATOR
layer: the number was measured correctly and then transported by the wrong arithmetic.

---

## 4. The 504-pair reproduction — pass 8 had already run this

`ddm_sj1_joint_admission.cmd_refine` calls `jg5.refine_pair` from
`inst.state.codes[pair]` — the SHIPPED codes — at `outer_rounds=40`, `max_gn_iterations=400`, with
`dd_threshold = jg5.materiality_dd_threshold(base_mean)`. Those are, to the digit, cb1's `resolve`
defaults and this arm's constants; pass 8's shard receipts record the same
`dd_threshold = 4.71841991887919e-09`. **The solver cb1 called "deeper" is the same solver at the same
budgets** — "deeper" was true only against pass 7's LIGHTER control, and pass 8 had already replaced
that.

Pass 8 sharded its refine over all 600 pairs against its full 96-pair edit overlay. On the **504**
pairs that edit set never touched, its instrument is this arm's instrument:

| check | MEASURED |
|---|---:|
| pairs where pass 8's start d_pose equals mine to 1e-15 | **504** |
| of those, final d_pose AND all 12 codes bit-identical | **504 / 504** |
| improvers among them | **2** (387, 391) |
| gain sum among them | **4.0943e-07** |

Both improvers live in the shared 504. On the 96 pairs pass 8 solved against EDITED renders, this
arm's run against move 49's own renders found **zero** improvements. So the population is fully
covered and the two independent harnesses agree exactly.

**What this arm adds over pass 8's stage, said plainly so it is not oversold:** the 96 fresh pairs
(zero gain), the n600 mean re-measured at batch 8 on the composed code table, the **exact archive
bytes**, the twins, and the acceptance discipline of §5. The *distortion* finding was already on
disk.

---

## 5. Pair 387 — a 6.82e-09 "gain" that returns the codes it started with

Pair 387's row reads `start 4.435060e-07 → final 4.366888e-07`, a gain of **6.817189e-09**, with
`changed_coordinates: 0` and `polish_steps_total: 2`. Its history has two decreases. Reconstructed
from the solver: the ±2 polish moved to a neighbour, then moved BACK, and **the same shipped codes read
6.82e-09 lower the second time.**

The mechanism is the batch: `refine_pair` seeds `best` with `br1.evaluate_codes` on a block of **one**,
then reads candidate blocks of **10–32**. The same codes through the same receiver at a different batch
size are not bit-identical on CPU. The magnitude, **6.82e-09, is 2.64× pp1's measured batch-1 vs
batch-8 band of 2.586e-09** — so pp1's band, measured on 16 pairs at batch 8, UNDER-STATES what a
polish block can move.

Two consequences, both landed rather than noted:

1. **Acceptance requires a moved code, not just a gain.** A pair whose final codes equal its start
   codes cannot have improved; the gate now refuses it and reports it as
   `identical_codes_with_positive_gain`. Without this the arm would have reported "2 pairs accepted"
   over an archive that differs on one.
2. **Every accepted pair is re-read at batch 8 against its own base** before it is believed. Pair 391
   verifies: base `1.7269100e-05` → resolved `1.6862829e-05`, **batch-8 gain 4.0627e-07** against an
   in-loop 4.0261e-07 (**1.009×**). The gain is real; its in-loop magnitude was 0.9 % optimistic.

---

## 6. The three legs — alone, subset, composed

Composed by OBJECT CHANGE, never by adding legs. The two moves are MEASURED **disjoint**: pass 8's
admitted pairs are `[59, 70, 159, 196, 292, 361, 392, 403, 412, 522, 561]` and this re-solve moves
`[391]`, so the composed object's d_pose is its own mean over the same 600 pairs, recomposed from the
two measured per-pair changes.

| row | d_seg (T4) | d_pose | archive B | S | ΔS like-for-like | bars | ΔS nominal | bars |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| null control (move 49, this instrument) | 0.00010287 | 4.5435687e-06 | 179,153 | 0.13631822891280948 | 0 | 0 | −4.7689e-06 | 0.238 |
| **A** re-solve alone | 0.00010287 | 4.5428916e-06 | **179,153 EXACT** | 0.13631772662593317 | **−5.0229e-07** | **0.025** | −5.2712e-06 | 0.264 |
| **B** pass 8 subset alone | 0.000102708828 | 4.5434408e-06 | 179,163.12 model | 0.13630875562574571 | **−9.4733e-06** | **0.474** | −1.4242e-05 | 0.712 |
| **C** composed | 0.000102708828 | 4.5427637e-06 | 179,163.12 model | 0.13630825333180135 | **−9.9756e-06** | **0.499** | −1.4744e-05 | **0.737** |

**Realized fraction of the sum: 1.0000007.** The composition is additive here, and the reason is the
measured disjointness plus a second-order effect the concave pose leg contributes at the 7th digit —
not a general composition law. Falsifier #3 ("the composition realizes < 0.8 of the sum") does NOT
fire; it does not need to, because both inputs are small.

**Neither row clears the bar under either accounting, so no candidate was built.** Building past an
admission that cannot be met is what pass 8 declined to do for the same reason.

### The two accountings, and why the memo reports both

* **like-for-like** uses this instrument's own move-49 null control as the base. It is the only
  difference that transfers to a T4 fire, because both numbers come off the same frozen CPU-torch
  PoseNet.
* **nominal** uses the pointer's banked T4 row, which is the convention pass 8's admission ledger
  used. It carries a **−4.7689e-06 phantom** that is purely the gap between this instrument's base
  (4.5435687e-06) and the T4 **PRINT** (4.55e-06). The phantom is identical in every row, so it
  inflates every candidate equally and cancels in any comparison BETWEEN candidates — it does not
  cancel against the bar.

### The rate leg's honest label

Row **A**'s bytes are EXACT: a real `archive.zip`, built through `up3.build_archive` with parse-back
verification, twinned, with move 49's own identity control passing on the same call. Rows **B** and
**C** carry pass 8's subset byte count, which is a **sum of its MEASURED per-frame RLC1 bit ledgers** —
a ranking, not a charge. sj1 measured that same estimator under-charge a 370-pair subset by **19.6 B
(+0.0108 %)**; at this scale that is 1.3e-05 S of slack, larger than the whole of row A. A composed
row would have needed its own real RLC1 encode before any of it could be believed, and at 0.74 bars it
did not earn one.

---

## 7. The composed carrier — re-verified, and it returns pass 8's own codes

The charter asked for the 11 admitted pairs to be re-rendered and re-solved with the same solver. Done,
against pass 8's own overlay RESTRICTED to those 11 pairs (the composed object carries only the
admitted subset, so the other 85 edited pairs must fall through to move 49's decode — the restriction
is in code and refuses a pair the overlay does not carry). The admitted planes were MEASURED identical
to the full-edit planes on all 11 kept pairs, which is what makes reusing the big overlay legitimate.

**Result: 11 of 11 bit-identical to pass 8's `refine_pass8` rows — final d_pose and all twelve codes.**

| pair | base | start (edited render) | resolved | dc |
|---:|---:|---:|---:|---:|
| 59 | 1.3383e-05 | 7.9244e-04 | 1.310189e-05 | 10 |
| 70 | 1.2065e-04 | 1.2722e-04 | 1.206515e-04 | 1 |
| 159 | 6.4764e-07 | 2.4885e-04 | 2.565094e-07 | 4 |
| 196 | 4.5769e-08 | 3.4333e-05 | 2.379588e-07 | 1 |
| 292 | 6.7632e-08 | 4.7467e-04 | 1.920326e-07 | 8 |
| 361 | 2.8355e-07 | 1.4339e-05 | 4.442734e-08 | 9 |
| 392 | 2.3214e-06 | 8.9634e-05 | 3.146506e-06 | 1 |
| 403 | 7.4179e-08 | 2.8612e-04 | 6.270368e-08 | 4 |
| 412 | 1.7377e-07 | 1.1133e-05 | 8.151689e-08 | 1 |
| 522 | 2.6088e-06 | 6.1829e-05 | 2.450442e-06 | 1 |
| 561 | 5.8824e-08 | 1.5888e-05 | 1.517622e-08 | 3 |

The `start` column is the mechanism the whole family runs on: a single token edit moves frame 1's
render and the SHIPPED carrier codes then read **6× to 7,000×** worse on pose, and the carrier re-solve
gives nearly all of it back. Nine of eleven end BELOW their own base. **The carrier is doing the work
pass 8 credited it with — this arm's contribution is that the credit is now re-verified by an
independent harness rather than inherited.**

---

## 8. Frame-0 repair, inside the admission — NOT re-measured, and why

The charter says "frame-0 repair inside", which is the pose re-solve law's clause
(`[[pose_resolve_is_mandatory_after_every_field_change_20260910]]`): the frame-0 selector is admitted
inside the admission or not at all. `ddm_rp1_frame0`'s own contract is narrower still — *"this module
only ever proposes on pairs the admission is about to drop for pose."*

A carrier re-solve has no such population. `refine_pair` is monotone from the shipped codes, so no pair
is ever made worse and no pair is ever dropped for pose; there is nothing for frame 0 to repair that
the re-solve did not already take.

For the composed row the population is pass 8's, and pass 8 MEASURED it: swept on all 43 pose-bound
dropped pairs, the adopt stage reports 12 pairs and +11 B of selector, standalone −3.980e-05, and
**inside the admission it LOSES by +9.384e-07** with its fixed point BREAKING on 2 of the 12. That is
the third consecutive pass to decline it (pass 6 on a build blocker, pass 7 at +1.09e-06, pass 8 at
+9.38e-07).

**This arm did not re-measure it.** DERIVED, and labelled as such: the re-solve improves frame 0's pose
on pair 391 and leaves the other 599 frame-0 renders bit-identical, so it removes headroom from the
frame the selector acts on and cannot make the selector's case better. That is an argument, not a
measurement, and a successor that wants the frame-0 door re-opened must re-run rp1's sweep rather than
cite this paragraph.

---

## 9. The arithmetic, re-derived at move 49 (binding numbers expire)

* base `d_pose = 4.543568679770593e-06`, pose leg `sqrt(10 d) = 0.006740599884113129`, MEASURED.
* exchange `25 / 37,545,489 = 6.658589531221714e-07` S/B.
* pose sensitivity `dS/d(mean d_pose) = 5 / leg = 741.7737420944483`; one unit of **1e-08** of mean
  d_pose is worth **11.140 B**.
* the whole **−2e-05** bar from pose alone needs a mean drop of **2.6962e-08**, i.e. a population gain
  SUM of **1.6177e-05** — **40.2×** the 4.0261e-07 this rung MEASURED.
* identity control on the score arithmetic:
  `100·0.00010287 + sqrt(10·4.55e-06) + 25·179,153/37,545,489 = 0.13632299781031237` — move 49's banked
  row, reproduced from components to 0.0.

---

## 10. Falsifier verdicts, pre-registered in the charter

| falsifier | threshold | MEASURED | verdict |
|---|---|---|---|
| #1 the n600 re-solve alone gains < 0.15 % of d_pose | 0.15 % | **0.01502 %** (0.01477 % code-moving) | **FIRES, by 9.99×** |
| #2 …or costs > +20 B | +20 B | **0 B exact**, twins agreeing | does not fire |
| #3 the composition realizes < 0.8 of the sum | 0.8 | **1.0000007** | does not fire |
| fire bar: net ΔS < −2e-05 on the RESOLVED pose with exact bytes | −2e-05 | **−9.976e-06** lfl / **−1.474e-05** nominal | **NOT MET** |

Pre-registered band for the composition was −1.5e-05 … −3.5e-05 S. The measured nominal −1.474e-05
lands **just below** the band's floor and the like-for-like −9.98e-06 lands **1.5× below** it. The band
was not unreasonable; it inherited the ratio estimator §3 falsifies.

---

## 11. Verdict scope

**`verdict_scope: INSTANCE → FORMULATION on this object.`**

CLOSED by measurement: **re-solving move 49's shipped carrier COEFFICIENTS with the canonical
`jg5.refine_pair` at the shipped basis, shipped 5-bit precision, shipped 24×32 band-limit and shipped
int12 lattice, from the shipped codes, over all 600 pairs.** The shipped carrier is at that solver's
fixed point on 599 of 600 pairs; the one exception is worth 0.025 of the admit bar at zero bytes. This
is an INSTANCE verdict on move 49's codes and a FORMULATION verdict on "re-run the canonical solver
and expect a row" — it says nothing about a different solver, a different start, or a different body.

**NOT closed by this arm, stated so a successor does not over-read it:**

1. **pc3's continuous ceiling.** `mode=ceiling` (the same search with the lattice projection removed)
   measured, on **move 44**, a mean per-pair gain of 4.360e-07 at n=135 — three orders above what the
   lattice solver finds here. That is a DIFFERENT body and this arm did not re-measure it on move 49;
   what it does establish is that the gap between "continuous optimum" and "what the lattice solver
   reaches from the shipped codes" is the live quantity, not the solve itself.
2. **A different solver or a different start.** Every pair here stopped on `no_improving_step` or
   `lattice_floor` — physics, never a budget — but that is a statement about THIS neighbourhood
   (GN direction on `br1.STEP_LADDER` plus a ±2 polish). A basin-hopping or multi-start search is a
   different object.
3. **Pair 391 itself.** It is banked here as a measured, byte-free, twins-proved +0.025-bar move that a
   successor can carry into any composition for free. It is not worth a row alone.
4. **Anything about d_seg.** Unchanged by construction and then confirmed at the bytes (C5).
5. **The composed row's exact rate.** Rows B and C were never re-encoded through RLC1, because at 0.74
   bars they could not be admitted whatever the encode returned.

---

## 12. What this does NOT claim

0. **No score, no candidate, no seal, no packet, no fire.** The pointer is unmoved. The one archive
   this arm built was built for its BYTE measurement and its twins; it is retained with its sha and is
   not a submission.
1. **No axis transfer.** Every pose number is `[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT]`
   and is only ever differenced against another local number. The T4 print 4.55e-06 is quoted at its
   own ratio (0.998587) and never differenced except in the explicitly-labelled nominal column.
2. **Rows B and C are pass 8's legs, not re-measured here** except the carrier, which §7 re-verifies
   bit-identically. Their seg leg is pass 8's jg1 instrument carried to T4 by its own same-instrument
   ratio; their rate leg is its bit ledger.
3. **§8's frame-0 reasoning is DERIVED, not measured** on this object, and says so.
4. **The 504-pair reproduction is not a claim that pass 8 was wrong.** Pass 8's refine is correct and
   this arm agrees with it to the last digit. The finding is about where the number was FILED.
5. **pc3's move-44 ceiling is CITED, never transferred.**

---

## 13. Custody

Everything under `/Volumes/VertigoDataTier/pact/ddm_cr1/` — **57 files, 842,179 B (0.803 MiB)**, every
one with bytes and sha256 in `RETENTION_MANIFEST.json`, far inside the 8 GiB cap. Second copy at
`/Volumes/APDataStore/pact/ddm_cr1_deeper_carrier_resolve_20260912/custody/`, **57 of 57 sha-verified
ON THE DESTINATION**, not on the label (`[[moved_labels_are_not_custody_vr7_deleted_moved_payloads_behind_live_redirects_20260910]]`).
Vertigo reads 38 GiB free, under the charter's 42 GiB threshold, so the overflow copy was taken even
though 0.8 MiB could never need it.

| what | path |
|---|---|
| n600 pose base, shipped codes, C1–C3 receipts | `base/pose_base_move49.npy`, `base/coefficient_codes_shipped.npy`, `base/BASE.json` |
| n600 re-solve rows + 9 shard receipts | `resolve_n600/rows_*.jsonl`, `resolve_n600/RESOLVE_*.json` |
| composition re-verification, 11 pairs | `resolve_subset11/` |
| gate, batch-8 re-measure, **the exact archive** | `assemble_resolve_only/ASSEMBLE.json`, `candidate_archive.zip`, `coefficient_codes_candidate.npy`, `pose_resolved_n600.npy`, `per_pair_gain.npy` |
| twins (double compile + identity) | `twins_resolve_only/TWINS.json` |
| three-leg table | `VERDICT.json` |
| launch manifests and logs | `stage_*/` |
| retention manifest | `RETENTION_MANIFEST.json` |

Producer: `experiments/ddm_cr1_deeper_carrier_resolve.py` (modes `base` / `resolve` / `assemble` /
`verdict` / `twins`), tests `src/tac/tests/test_ddm_cr1_deeper_carrier_resolve.py`. Provenance pins:
HEAD `8dbadf13c7a6c43a7f51bfe18623ce655a9372f5`; cb1 memo sha `07b6e9ecdc911116…`; pass 8 memo sha
`e43dbcf378eecd5c…`; `ddm_jg5` sha `bcf41c5295b314eb…`; `ddm_cb1` sha `c4f8b17ef9ec478d…`. Cost: 4,575
CPU-seconds over 9 shards, 525 s wall-clock, plus 29 s for the batch-8 n600 re-measure.

---

## 14. Boundaries honoured

No Modal, no `authorize_*`, no `fire_modal_auth_eval.py`, no fire, no packet, no `ScheduleWakeup`.
Nothing under `upstream/`, `submissions/semantic_joint_ctxmix/`, any sealed tree,
`src/tac/candidate_seal.py`, `src/tac/decode_wall_clock.py`, receiver code, the renderer, the basis or
the prior was edited. The sj1 / cb1 / pp1 / ren / dpi1 / rq1 / so1 directories were read only — pass
8's overlay, admission receipts and refine rows were consumed, never written, and cb1's rows were read
to adjudicate its own headline. No storage reserve was lowered. `sys.dont_write_bytecode = True` is set
at module import so no `__pycache__` lands in the read-only pointer tree
(`[[read_only_tree_that_did_not_come_back_untouched]]`, pass 8 §6c). Every number above is labelled
MEASURED, DERIVED or CITED.

<!-- # FORMALIZATION_PENDING: this arm produces a scoped negative and an estimator correction, not a
     law. The durable law candidate is the ratio-vs-gain-sum estimator distinction in section 3, which
     needs a second instance on a different statistic before it can be registered as a canonical
     equation; the score arithmetic used throughout is the registered
     S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489. -->

Own-vehicle frontier (unchanged by this arm):
**S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600]** (move 49).
