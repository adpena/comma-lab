# ddm_pr19 — (A) the risk gate projects from measured T4, not a local ratio; (B) timing chains stay refused

`# FORMALIZATION_PENDING: both adjudications are contract text over a custody/spend gate, not laws of
the S-arithmetic. They produce no equation; the canonical equation they must not disturb is the rate
term (archive.zip bytes), which no risk mode touches. Formalize if a later unit generalizes "an
identity class of measured runs dominates an unmeasured member" across measurement surfaces.`

**Axis: `[exact contract arithmetic; measured receipts, scorer-free]`. No score claim, no promotion
claim, no fire. Pointer unmoved: composition S 0.13654774984742127 @ 179,359 B [contest-CUDA T4 n600]
(move 47).**

## Answer first

**(A) YES, with a correction to the proposed form.** The pre-fire risk gate may stop projecting a local
decode ratio — but it may NOT claim `decode_work_delta: none` / `fraction 0`. The projection becomes the
**MAXIMUM of real T4 decode measurements of the candidate's identity class** (same receiver, same decoded
token plane), and the local ratio is **kept and re-aimed**: it must now prove the candidate cannot reach
the **1,800 s hard timeout** even if the entire observed local delta is real work. The 1,260 s policy
limit is untouched and is applied to a real measured number. New mode
`measured_t4_identity_class_envelope`, definition
`tac.candidate_seal.validate_prefire_risk.identity_class_envelope.v1`.

**(B) REFUSED, with no chain-length bound.** Inheritance may not follow a chain. The decisive reason is
not drift arithmetic: **inheritance saves no dispatch.** Every candidate is fired on T4 to produce its
exact row, so a chain would buy only the convenience of sealing *before* that fire, at the price of an
unmeasured accumulating drift assumption. The cheaper legal path already exists and is the one (A)
unblocks: fire the candidate as a first measurement, and its own `t4_direct` leg restores the anchor for
its immediate successor. The contract's "at most ONE inherited row per measured leg" is therefore right.

MEASURED on the real host today: the move-48 candidate (179,111 B, `d830edd3…`), refused by both
inheritance routes, **clears the new rule at 1,232.418725255 s ≤ 1,260 s** (27.58 s spare) with a
hard-timeout stress of **1,477.61 s ≤ 1,800 s** (322 s spare) — while the legacy local-ratio projection
lands at **1,367.77 s** and refuses it.

## The measured facts, re-derived here (not quoted)

**One receiver, five archives.** `measure_prefire_risk_receiver_digest` and pr18's behavior digest both
return `9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890` over **49 rows** for move 44
(`ddm_rlc5_cure_on_move43`), move 45 (`ddm_pc3_pose_carrier_curve`), move 46 (`ddm_ntb2_proof45`), move
47 (the pointer, `ddm_hpr1/public/retrain_control`) and the move-48 candidate
(`ddm_hpr1/public/retrain_frame_even`). Pairwise, **zero normalized rows differ** between the candidate
and moves 44/46/47. Every legacy digest differs, because `MANIFEST.sha256` restates the archive pins.

**One decoded token plane, on both axes.** `decoded_token_sha256` `a92e7d902a449896…` (full digest in
each report) in the T4 cold reports of moves 44 and 46 **and** in the local cold reports
of move 47 and the move-48 candidate. The token plane is a pure integer decode, so it crosses the device
boundary where the rendered raw does not (T4 raw `a35c969e…`, local raw `2b762eba…`).

**Two real T4 decodes of that class, 8.03 % apart — and the spread is the host, not the payload.**

| | move 44 | move 46 | Δ |
|---|---:|---:|---:|
| archive | `04758c0d…` 180,406 B | `a0de607d…` 180,001 B | −405 B |
| `decoder_bit_position` | 958,049 | 960,913 | **+0.30 %** |
| `archive_setup` | 10.092 s | 8.663 s | −14.2 % |
| `frame0_selector_and_io` | 6.201 s | 6.063 s | −2.2 % |
| `neural_render_and_resize` | 49.859 s | 43.371 s | −13.0 % |
| `token_decode` | 1,144.149 s | 1,064.892 s | −6.9 % |
| **inflate_elapsed_seconds** | **1,232.418725255** | **1,140.8051148040001** | **−7.4 %** |

Move 46 consumed **more** coded bits and took **less** time, and **every stage moved down together** —
including the render stage, which is payload-independent once the token plane is identical. That is the
signature of a faster T4 instance. So "same receiver ⇒ same decode seconds" is false at the 8 % level,
and the cause is the instrument.

**The local instrument is worse than that, by an order of magnitude.** Cold n600 windows of this one
identity class: **2,317.38 s** (ntb2, move-46 bytes), **1,957.60** and **1,619.12 s** (pc3 and ntb2 on
*identical* move-44 bytes), **819.69 s** (move 47), **982.77 s** (the move-48 candidate) — a **2.83×
spread**. The gate's `max(0, ratio − 1)` clamp is one-sided, so that noise can only ever inflate the
projection. This is the dwc1 genus at the level of an instrument: the gate is trying to resolve an effect
near zero with a proxy whose own spread is 183 %.

**But the candidate's local delta is NOT obviously noise, and that is why `fraction 0` is refused.**
Move 47 → move-48 candidate, cold, same host, same day: `archive_setup` −0.3 %, `frame0` −0.2 %,
`render` **+2.8 %**, `token_decode` **+28.4 %**. Three stages flat and one up is *not* a host-factor
signature. I cannot attribute it — contention concentrated in the 72 % of wall-clock the token stage
occupies is as consistent with the data as real work. An honest rule must not declare that delta absent.

## Clause table — defect → rule → code → test → measured proof

| # | Defect / requirement | Rule | Code | Test | Measured proof |
|---|---|---|---|---|---|
| 1 | The projection rests on a proxy whose spread (2.83×) swamps the effect | projection = **max measured `inflate_elapsed_seconds` over the declared `t4_direct` legs of the identity class**; no local number enters it | `candidate_seal._validate_prefire_risk_identity_class` | `test_pr19_identity_class_risk_passes_without_a_local_ratio_projection` | ceiling 1,232.418725255 s from two real T4 legs |
| 2 | "Same receiver" does not pin decode work | class membership also requires the **same `decoded_token_sha256`** on every leg | same, leg loop | `test_pr19_refuses_a_leg_that_decoded_a_different_token_plane` | `a92e7d90…` on 4 archives, both axes |
| 3 | A class member must actually **dominate** the candidate | candidate `decoder_bit_position` ≤ and archive bytes ≤ the **dominating** leg's, and that leg must be declared in the class | same | `test_pr19_refuses_a_candidate_that_consumes_more_coded_bits`, `test_pr19_refuses_a_candidate_archive_larger_than_the_dominating_leg`, `…shape_refusals[dominating leg is not declared]` | candidate 951,228 bits / 179,111 B ≤ move 46's 960,913 / 180,001 |
| 4 | tc4: a receiver change NEEDS its own measured wall clock | the delta manifest must carry **zero differing rows**; any receiver change keeps `completed_t4_receiver_delta` | same, delta block | `test_pr19_refuses_a_real_receiver_change` | 0 differing rows, candidate vs moves 44/46/47 |
| 5 | A forged or foreign report could fake the work facts | the facts are re-derived from the **receiver's own** cold report, pinned by the intent's `raw_identity_n600`, and must name the candidate archive AND the retained candidate raw | `candidate_seal._pf_candidate_cold_report`, `_pf_cold_work_facts`, `decode_wall_clock.t4_direct_cold_report` | `test_pr19_refuses_a_candidate_cold_report_that_names_another_archive`, `test_pr19_t4_direct_cold_report_exposes_the_work_facts` | the staged `RAW_IDENTITY_N600.json` log reports `d830edd3…` / 179,111 B / `2b762eba…` |
| 6 | The 1,260 s limit must not be weakened | the ceiling is still tested `≤ 1,260`, and a `t4_direct` leg above it cannot even be built | `decode_wall_clock.validate_decode_wall_clock`, leg loop guard | `test_pr19_keeps_the_1260_policy_limit_and_adds_an_1800_stress_refusal` | a 1,260.5 s leg refuses at build |
| 7 | The local delta must not simply be thrown away | the ratio is **retained** and becomes a hard-timeout stress test: `ceiling × (1 + observed fraction) ≤ 1,800` — a refusal that does not exist today | same, arithmetic block | `test_pr19_keeps_the_1260_policy_limit_and_adds_an_1800_stress_refusal` (both the refusal and the pass) | 1,232.42 × 1.198951 = **1,477.61 ≤ 1,800** |
| 8 | Diagnostics could come from another receiver or a warm run | each diagnostic names a **tree** whose risk digest equals the candidate's, is cold n600, and names the candidate archive (or, for the base, a declared class archive or the pointer's) | same, diagnostics loop | `test_pr19_refuses_a_warm_or_foreign_local_diagnostic` | base = move 47's tree @ 819.6933833750081 s; candidate = move 48's @ 982.7723198329913 s |
| 9 | (B) chains must stay refused | inheritance is **unchanged**; both routes refuse verbatim on the real trees | `decode_wall_clock.validate_decode_wall_clock` (`:574`, `:578`) — untouched | `test_pr19_chain_inheritance_stays_refused_on_the_real_trees` | both refusals + the control that the same leg still inherits while it IS the pointer |

## (A) — which timing-out failure could this let through, and why the conditions exclude it

The failure guarded is: a paid T4 dispatch whose `inflate.sh` exceeds the **1,800 s** hard timeout,
burning the dollar and producing no row.

- **F1 — more symbols to decode.** Excluded by clause 2: the decoded token plane is the decoder's output;
  identical plane ⇒ identical symbol count, and the per-symbol cost is fixed by clause 4's byte-identical
  receiver.
- **F2 — a longer coded stream.** Excluded by clause 3: `decoder_bit_position ≤` the dominating leg's.
- **F3 — a bigger model/table to materialize.** Bounded, not eliminated, by clause 3's archive-bytes
  domination. **Honest residual:** total bytes could fall while a model section grows. Its size is
  measured: `archive_setup` is **0.0988 %** of a cold local decode (2.289 s of 2,317.38 s). A model
  section large enough to threaten 1,800 s would have to inflate that stage by ~500× while the archive
  shrank. I did not close it further because section-level accounting would mean executing the
  candidate's own `residual_archive.py` inside the validator, which I will not do.
- **F4 — the unattributed +28.4 % token-stage delta (the real one).** NOT excluded — it is *priced*.
  Clause 7 takes it at face value against the worst real measurement of the class: 1,477.61 s, which is
  **18 % under** the hard timeout. If a future candidate's local delta were large enough to reach 1,800 s
  on that arithmetic, the gate refuses.
- **F5 — T4 host variance.** Measured at 8.03 % (move 44 vs move 46) and covered by the 1,260 → 1,800
  reserve: from the ceiling, a real timeout needs the host to be **46 % slower** than the slowest T4 run
  we have ever measured of this receiver.
- **F6 — a fabricated diagnostic or report.** Clause 5 and clause 8: every number is re-derived from a
  hash-pinned receipt that names its own tree and archive; `_cold_public_report` still requires 600 pairs,
  no checkpoint resume, and `token_cache: DISABLED`.

**The honest cost of (A).** The ceiling is the max over *declared* legs, so a producer who declares only
the fastest member of the class gets a lower ceiling than the class truly warrants (here: 1,140.81
instead of 1,232.42). Two things bound that: the dominating leg must be declared, and the base diagnostic
must belong to a declared class archive or to the pointer. It is not fully closed, and closing it would
need a registry of every retained `t4_direct` leg that the contract does not have. At today's numbers the
adversarially-minimal declaration is still 1,140.81 × 1.198951 = 1,367.77 s against 1,800.

## (B) — the refusal, and the cheaper legal path

The contract's two clauses (`decode_wall_clock.py:574` "inheritance must point directly to a measured or
t4_direct leg (never to an inherited one)" and `:578` "source measurement is not the pointer archive")
together admit **at most one inherited row per measured leg**. hpr1 hit both on the move-48 candidate;
ntb2 hit both one move earlier. The physics objection is real — the receiver that was timed is
byte-for-byte the receiver that would run — and it is still not sufficient:

1. **A chain buys no dollars.** The seal's timing leg is not a score. The exact row comes from a T4 fire
   that happens either way. Inheritance only decides whether the pre-fire object is a seal carrying an
   inherited leg or a prefire intent carrying a first-measurement authorization. Trading an unmeasured
   drift assumption for zero saving is a bad trade at any chain length.
2. **No per-step drift bound exists.** The one paired observation of the class (1,232.418725255 →
   1,140.8051148040001) is 8.03 % and is **confounded** with T4 host variance — every stage moved
   together. With n = 1 and a confound I cannot bound the per-step distribution, and a bound is exactly
   what a chain-length licence requires. Compounding the observed one-step magnitude twice already
   exceeds the policy limit: 1,140.81 × 1.0803² = 1,331 s > 1,260.
3. **The system self-heals every other move.** Move 46 measured, move 47 inherited, move 48 measures
   again — and move 49 may then inherit move 48's own `t4_direct` leg directly. The alternating pattern
   costs nothing extra and keeps every timing statement within one archive step of a real measurement.

**Cheaper legal path:** the first-measurement route (prefire intent → authorization → fire), which
(A) is what unblocks. It over-pays procedurally relative to a seal-inherit, and it under-pays in risk.

## Attacking my own conclusion

- *Is the risk gate actually a forever-refusal (dwc1)?* No — a matched-concurrent local pair clamps to
  fraction 0 and pc3 got one. So a door exists, and (A) is not justified by "no producer can pass". It is
  justified by evidence QUALITY: a structural domination argument in the receiver's own reported stages
  beats a 2.83×-noisy proxy of a different device. I say this plainly because the weaker justification
  would have been the more convenient one.
- *Does (A) hide a real slowdown?* It prices it instead of hiding it (clause 7). The candidate's +28.4 %
  token-stage delta is carried into the arithmetic at face value, against the 1,800 s limit where a real
  timeout lives. Today's rule never tests anything against 1,800 at all.
- *Is the new mode strictly weaker than the old one?* No. It is weaker in exactly one place (the local
  ratio no longer gates at 1,260) and stronger in five: identical receiver required (the old mode admits
  a delta), token-plane identity required, bit and byte domination required, diagnostics bound to trees
  rather than to a bare stored digest, and an 1,800 s stress test added.
- *Could a candidate dodge the token-plane check?* Only by decoding a different token plane, which is
  precisely the case the rule sends back to the legacy mode and to its own measurement.
- *Did I weaken any refusal guarding a real receiver change?* No. `decode_wall_clock` inheritance is
  untouched; the new mode refuses any receiver delta outright; both chain refusals are now pinned by a
  test on the real trees.

## Consequences to append (HELD)

Per MAIN's hold rule, the amendment and consumer-fix rows are prepared but **NOT appended** to
`.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json`: appending invalidates every in-flight
prefire intent (pr17 NO-GRACE), and the move-48 candidate's first-measurement intent is in flight. The
rows and their implementation manifest are staged unappended under `.omx/research/ddm_pr19_20260911/`.

## Boundaries honored

No Modal, no fires, no scorer runs, no candidate work. `tools/fire_modal_auth_eval.py` (pinned by the
frozen file) was not edited. Arm directories, the pointer receipts, `upstream/` and the PR tree were
READ ONLY — the five candidate trees were hashed and their logs read, never written. Every JSON wrapper
the tests needed was written into `tmp_path`.
