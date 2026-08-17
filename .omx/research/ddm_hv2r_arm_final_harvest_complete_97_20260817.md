# ddm_hv2 (round 2) — complete harvest of the 97 FINISHED-unharvested arms

`date_utc: 2026-08-17` · `owner: ddm_hv2` · `axis: read-only harvest, $0, no dispatch`
`score_claim: false` · `promotable: false` · `pointer_delta: NONE — this unit moved no score`
`own_vehicle_frontier: hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600] — UNMOVED`

**STORES CONSULTED:** `tools/codex_arm_queue.py` (`status` + `_done_receipt` + `latest_by_name`
re-executed, not quoted) · `.omx/research/arm_final_messages/` (all 97 finals, read in full) ·
`.omx/state/codex_arm_queue.jsonl` (1,196 events) · `.omx/state/codex_arm_queue.next_if_resumed.jsonl` ·
`.omx/state/canonical_frontier_pointer.json` · `.omx/state/main_hot_state.md` ·
the prior `ddm_hv2_arm_final_harvest_20260816.md` · `ddm_ns1` · `ddm_na7` · `ddm_gestalt_two_week_recall` ·
`ddm_cn5` · `ddm_gs1` · `ddm_nx1` · `ddm_qw1` · `ddm_av3` · `/Volumes/APDataStore/pact/ddm_xi2_20260812/`.

Ledger: **`.omx/research/ddm_hv2r_arm_disposition_20260817.jsonl`** — 97 rows, sha256
`b9df90ed028f9dc64bf69ed307782ffc58e6fa387c54fcca71d08592dd53e117`, 72,091 B, **0 UNKNOWN**.

---

## 1. Headline

**No new score lever exists in the 97 arms.** I read every one — 97/97, no sampling — and priced
each against the current vehicle. Zero rows qualify as LIVE-LEVER: none is a measured, still-
applicable candidate on the hv1 object with `|ΔS| ≥ 1e-5` and a defined next action.

The **APPARATUS-ONLY** pre-registered fork fires.

That is not the same as "the pile was empty." The pile is 48 apparatus landings, 30 honest
negatives, 15 superseded candidates, and 4 blocked-empty arms. Its value is a prohibition list and
five unowned apparatus items — not a byte.

**Why the levers are absent is now measured, not guessed:** 70 of 97 arms measured on an ANCESTOR
object (cp135 186,252 B / mc36 186,269 B / lc2 187,226 B), which the pointer has since passed twice
(e480b 183,502 → hv1 182,759). Per `[[m18]]`/`[[L18]]` those numbers do not transfer; their
mechanisms do. Only **18** arms measured on the current object, and every candidate among those was
scored and refused.

---

## 2. Coverage — the denominator, stated

| quantity | value | how |
|---|---:|---|
| FINISHED-unharvested arms | **97** | `codex_arm_queue.py` logic re-executed: `status ∈ {queued,live}` ∧ no live process ∧ `.done` receipt exists |
| read in full by this unit | **97 (100%)** | all `rc=0`; 349,540 B of final messages |
| finals persisted to `.omx/research/` | 97/97 | the np1 cure holds |
| `NEXT_IF_RESUMED` extracted | 95/97 | `gca1`, `pk3` lack an extracted block |
| named by ≥1 prior sealing audit | **77/97** | token match across the 9 audits above |
| named by the prior `hv2` specifically | **19/97** | it sampled ~180 of a 797-arm population and *deliberately did not load* 617 |
| named by **no** prior audit | **20/97** | §5 |

**This round's mechanism delta over the prior `hv2`:** that run reduced SCOPE by sampling. This one
does not. The complete pass is what makes §4's two corrections findable — both sit in arms the
sample happened to touch, but the *contradiction* is only visible once the whole set is priced
against one bar.

---

## 3. The bar, re-derived at the current object

I recomputed this from the pointer rather than inheriting it.

| quantity | value | label |
|---|---:|---|
| frontier `S` / archive | `0.15959729295498598` / `182,759 B` | MEASURED, `canonical_frontier_pointer.json` |
| rate term | `0.1216917164` | DERIVED, `25·182,759/37,545,489` |
| distortion (seg+pose) | `0.0379055765` | DERIVED, `S − rate` |
| `S` per byte | `6.65859e-7` | DERIVED |
| `1e-5 S` | **`15.02 B`** | DERIVED — the byte value of the reporting band |
| **rate-only close** | **save ≥ 14,414 B → archive ≤ 168,345 B** | DERIVED |

**Cross-check:** `wd2` and `rfo2` state the bar as "≥15,157 B → ≤168,345 B" off *e480b*. My
independent derivation off *hv1* gives ≤168,345.6 B. The two agree because seg and pose are
decode-identical between the vehicles. **Quote the invariant `archive ≤ 168,345 B`, never the
delta** — the delta went stale the moment the pointer moved, and §6 item 4 shows three live
fire-orders already carrying a stale one.

---

## 4. What the complete pass corrects (the deliverable)

**4.1 — The prior harvest's rank-3 lever is refuted, and the two documents still disagree.**
`ddm_hv2_arm_final_harvest_20260816.md` §4 ranks the `mp2` keep75-minus-keep87 differential at
`182,734 B`, `−25 B`, `−1.7e-5 S`, "built, unscored." `ddm_ns1` §A **measured it the same day**:
`ΔB −25`, `Δd_pose +4.08e-4`, 3 tensors touched (`blocks_1_film_weight`). Priced at the frontier
`d_pose = 6.88e-6`, the pose term goes `0.0082946 → 0.0644` and the row is **`ΔS ≈ +0.056`** —
refused by three orders. `−25 B` of rate cannot pay it. **Rank 3 is dead.** Anyone reading the
prior harvest without `ns1` beside it would fire an advisory n600 on a measured-refused candidate.

**4.2 — Rank 4 is ancestor-scoped and was not labelled.** The prior harvest lists `qs2` at
`−4.37e-6 S`. That projection is bound to the CP135 object (`186,286 B` candidate, `186,252 B`
parent). `qs5` shipped a compiler guard that makes stale compensation **fail closed** precisely
because compensation is content-bound to its token stream. So the number does not transfer to hv1
by the arms' own mechanism. It is `PREMISE-STALE`, not a ranked lever. (It was also sub-band on its
own object.)

**4.3 — `xi2` fired, and its result had never been read.** No audit names `xi2`. Its
`FULL_SCALE_RESULT.json` has sat on APDataStore since 2026-08-13: xi-warped previous-decoded-
partition context = **116,860 B** against the banked CL1 control **116,716 B**;
`xi_over_control = 1.0012337640083622`; the preregistered `<0.98` falsifier **FIRES**;
`status: FORMULATION_CLOSED_FULL_SCALE`. Learned ξ-context token coding is **closed at full scale**,
by a materialized receipt, with its own falsifier. This retires a family that `eu3` had ranked #5
and priced at a `−0.001554781 S` ceiling. It also independently corroborates `ns1` P5's
low-probability grading of the learned-context rate rung, from a different formulation.

**4.4 — The prior harvest's P0 is resolved.** Its §6.0 says the canonical task ledger "has been
refusing every write since 15:23Z." It now loads: **563 rows served**, 2 task IDs excluded as
unreadable with a warning, writes proceed. The strict loader degraded to warn-and-exclude rather
than refuse. The self-protection it asked for (refuse a non-registration append for an unknown
`task_id` at *write* time) is still owed.

**4.5 — This arm's own charter premise was partly wrong; corrected at source.** The charter cites
"#1006 … 49 unread arms surfaced a −903 B lossless rate lever and a 182,364 B archive." Measured:
#1006 harvested **12 of 53**; the successor `ah2` took the remaining **41**. The **−903 B** came
from `ddm_vp1` (split model streams, `73,065` vs `73,968 B`), was **never banked as an archive**,
and was later absorbed into the PR135 lineage. The **182,364 B** archive came from `ddm_cp2` and is
**FOLDED/DEAD** — `sm4` proved its exact decoded state is the already-refuted advisory row at
`S ≈ 7.4924`. **Do not resurrect cp2.** The harvest law survives the correction; the sample
statistic quoted for it does not.

---

## 5. The 20 arms no prior audit named — adjudicated

Not one carries a score lever. Five carry work worth owning.

| arm | what it holds | disposition |
|---|---|---|
| **`wc2`** | HPAC MPS race apparatus, **built and landed, never executed**. Measured routing: profile wall `2,486.478 s`, conv2d forward `1,282.433 s` (51.6%) + backward `1,008.073 s` (40.5%) = **92.1%** in convolution → a ≥3× GPU port is the named wall-clock lever. It failed only because that sandbox reported MPS unavailable. | **QUEUED-WITH-FIRE-ORDER** — the runtime-lift grant makes the device lock a porting item, not a wall. `ns1` P1 needs long training; this is its throughput. |
| **`hr2`** | Cures HR1's P0: the legacy helper rounds **after** downsampling; the correct order is bicubic camera lift → camera-grid uint8 STE → bilinear scorer downsample. MEASURED: 6,104,016/6,104,016 values match independent references, gradient max rel err `0`; legacy vs camera-uint8 differs on **1,126,626 RGB values and 8,288 RGB-channel argmax pixels**. | **FIRED** (landed `436edf452c`) — but verify the live trainers consume it before `ns1` P1 runs. |
| **`sp2`** | The token surrogate is calibrated: epoch-2 **real** RC64 payload `118,292 B` vs surrogate `118,277 B` = ratio **`1.0001268209`** (+15 B). Every RX2 joint-byte projection rests on this and no audit cites it. | **QUEUED-WITH-FIRE-ORDER** — model bytes stay the open term; IHS1 packing changed the deployed state-dict schema, so no archive price yet. |
| **`pq1`** | Submission packet: strict compliance **78/86 green, 8 RED**, HOLD. Reds include missing exact-byte CPU authority, the Brotli bootstrap vs the no-network-install check, and no hosted manifest. | **QUEUED-WITH-FIRE-ORDER** — the CPU axis is the leaderboard axis and our CPU frontier is a *different, non-submission* archive (`0.18804 @ 176,564 B`). |
| **`sc3`** | 132.880 GiB moved, 76/76 files SHA-equal, Vertigo `127.725 → 234.004 GiB` (**+106.279 GiB**). **Its memo is the one residual untracked artifact**, sha256 `48f115e8…` — byte-identical to the hash the arm declared. | **QUEUED-WITH-FIRE-ORDER** — land it; git is writable now. |

The other fifteen: `js2` and `js1` both measure the same instrument refusal (macOS-CPU control
disagrees with promoted T4 by **15,431 flips / 44.13%**, and the rendered-raw hashes differ — local
CPU is not an admissible solve instrument on this vehicle); `se1` adds the calibration
`δ = 0.0803604126` below which a local gain is indistinguishable from CPU/CUDA disagreement;
`vh3` is drain evidence (ms 49/49, j 45/45, zero remaining); `dio1` folds Dion3 (NS5 is `1.789
GFLOP/epoch` = 0.0615% of an epoch — there is no optimizer speed wall to cure); `dg1` yields one
ADOPT-with-named-consumer row (bending-energy change as a held-out **ranking feature**, never a
global loss); `xi1f` fixes a real bug (double module import → zero `bit_depth` parameters
registered) and demands a fresh rerun; `dr1`/`lh1`/`lh2` are landed apparatus; `na6`, `pr135ps`,
`qs1`, `re1t`, `vd1` are superseded or already-recorded.

---

## 6. Dead-ends this harvest confirms — do not re-open without new preconditions

Each has an exact receipt in the ledger row named.

- **Lossless recoding of the model section is exhausted** — `mz1` raced 8 complete representations;
  the incumbent split-Brotli q10/q11/q11 **won all 8** (best alternative +41 B, worst +13,196 B).
  `exact_model_section_savings = 0`.
- **The 52,566 B "serialization gap" is false attribution** (`mz1`): `70,557 = 13,619` HPAC +
  `34,763` semantic + `22,161` carrier + `14` wrapper. Raw HPAC is 5 B off its estimate, not 52 KB.
- **Exact semantic re-representation is closed** (`mz2`): 38/38 tensors receiver-required, 0/38
  derivable at decode, dense/sparse/row-dictionary/hybrid all **+340 B**.
- **Carrier lossless recoding is closed** (`mp2`): Brotli q0–q11 race found no gain; q11 ties the
  incumbent at 22,161 B.
- **Learned ξ-context token coding is closed at full scale** (`xi2`, §4.3) — new here.
- **Full-pixel ξ/XOR partition transport is dead** (`tf1`): 453,449 B vs 356,636 B intra
  (`1.271461658385581×`); identity persistence also loses at 435,536 B.
- **Post-hoc weight edits are pose-fatal** (`mz2`→`mp2`→`ns1` §A): three structurally different
  edits produced ~4.6–5× pose blow-ups; the sensitivity spread across directions is ~94×.
- **Temporal event coding loses to intra** (`ec1`): 633,441–633,606 B ≈ 1.78× intra.
- **Additive/post-hoc edge and pose probability tables are closed** (`sr1` via `js3`/`js1`): −2 B on
  edge context, +43 B on pose context.
- **Summing singleton benefits is invalid** (`js7`): joint remeasurement rejected 21/65 admissions,
  and a 311 B packet became **+323 B** at the container boundary.
- **`pz4a` precision coarsening** (`−2,232 B` net after the depth-map wire) and **F26 pass-9**
  (`pr135ps`: pass 8 accepted 0/595 and 0/14,277) are both closed.

**What survives as mechanism, not number:** `qs5`'s in-compile exact-object compensation moved
`d_pose` **below base** — the one winning sibling of the entire pose-damage genus, and the
mechanism `ns1` P3 routes forward. `eu4`'s allocation arithmetic also survives in FORM because it
depends only on `d_pose` and byte cost: at `d_pose = 6.88e-6`, halving it buys `−0.0024294252` and
a 1,000 B child nets **`−0.0017635663`**; driving pose to zero buys `−0.0082946`, which is **86% of
the 0.0095973 gap but cannot close it alone**.

---

## 7. Boundaries — what I measured and what I did not

**Re-derived at source by me:** the 97-arm denominator (queue logic re-executed, not quoted); every
line of §3; the `e480b`↔`hv1` bar cross-check; `xi2`'s `FULL_SCALE_RESULT.json` and its falsifier;
the §4.1 pose repricing; the task-ledger health; the tracked/untracked status of all 22 git-blocked
artifacts; the coverage join across 9 audits; the final-message SHA-256 of all 97 rows.

**NOT re-derived — pointers carrying their arm's framing:** every byte figure inside §5 and §6
except `xi2`'s and `mp2`'s. I read them from the arms' own final messages; I did not open
`mz1`/`mz2`/`tf1`/`ec1` receipts. Treat §6 as a **well-sourced prohibition list, not my
measurement**. Any single row is cheap to re-derive if it ever blocks a real candidate.

**Not measured at all:** any scorer, decode, archive build, training, or dispatch. No Modal spend.
`$0`, as chartered.

`verdict_scope: FORMULATION` for §4.3 (ξ-context at full scale, by its own preregistered falsifier).
`verdict_scope: INSTANCE` for the individual candidate refusals. The **absence** of a live lever is
scoped to *this population and this object* — 70 of 97 rows are ancestor-scoped, so a vehicle change
reopens their mechanisms, exactly as `ddm_ns2` is chartered to test.

---

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — land the `sc3` memo. Owner: MAIN. Consumer store: repository HEAD.
  Fire trigger: already satisfied (git writable; file sha256 `48f115e8a2ff2a3e87d53c5aeacf0784cc420b790b853aba5867757238be8e6a` matches the arm's declared hash). Action: serializer-commit that one file.
- **QUEUED-WITH-A-FIRE-ORDER** — correct the prior harvest's ranked table at source. Owner: MAIN.
  Consumer store: `.omx/research/ddm_hv2_arm_final_harvest_20260816.md` (APPEND-ONLY addendum).
  Fire trigger: immediate. Action: mark rank 3 REFUTED-BY-`ns1`-§A and rank 4 ANCESTOR-SCOPED, so no
  successor fires an advisory n600 on either.
- **QUEUED-WITH-A-FIRE-ORDER** — execute the `wc2` MPS parity race. Owner: MAIN Metal executor.
  Consumer store: `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/parity/`.
  Fire trigger: a host reporting `torch.backends.mps.is_available() == True` and a free Metal lane.
  Action: run CPU and MPS parity legs, then the full race only if repeated IHS1 packs match exactly.
- **QUEUED-WITH-A-FIRE-ORDER** — re-quote every live fire-order against `archive ≤ 168,345 B`.
  Owner: MAIN. Consumer store: the `rx1`/`rx2`/`sp2` fire orders (they cite `< 186,269 B`, which is
  **3,510 B above** the frontier) and `mz1` (cites `15,153 B`). Fire trigger: before the next
  candidate is priced against its own bar.
- **FOLDED** — the arm-final harvest as a lever-finding instrument on this vehicle. Two complete
  passes (`ah2` 41/41, this one 97/97) plus `vh2` 121/121 and `vh3` 94/94 now return zero live
  levers. Reopen only after a vehicle change, per `ddm_ns2`.

## LIVE-HYPOTHESES

- A ≥3× MPS port of the HPAC trainer is reachable, because 92.1% of the measured wall is inside
  convolution forward/backward. Kernel coverage and float-accumulation drift remain untested.
- The camera-uint8 ordering cure (`hr2`) changes what a scorer-aware trainer sees — 8,288 argmax
  pixels differ on two frames alone — so it may change `ns1` P1's outcome. Only a frozen-SegNet test
  can establish cell crossings.
- The pose leg remains the largest single named allocation (`eu4`: 86% of the gap at zero rate cost),
  and `qs5` is the only mechanism that has moved `d_pose` below base. Neither is a candidate yet.

## DEAD-ENDS

- Treating this pile as a lever source: measured empty across 97/97 on the current object.
- Firing the `mp2` differential advisory row: refuted by `ns1` §A before it was ever queued.
- Transferring any cp135/mc36/lc2 candidate byte count to hv1: `[[m18]]`, and `qs5`'s own
  fail-closed compensation guard proves the objects are not interchangeable.
- Resurrecting `cp2`'s 182,364 B archive: `sm4` closed it at `S ≈ 7.4924`.
- Re-racing learned ξ-context on the token stream: `xi2`'s own falsifier fired at full scale.
- Citing "#1006 = 49 arms / −903 B / 182,364 B" as a precedent statistic: corrected in §4.5.
