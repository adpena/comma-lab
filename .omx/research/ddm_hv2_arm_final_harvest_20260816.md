# ddm_hv2 — harvest of the finished arm final-messages

`date_utc: 2026-08-16` · `owner: ddm_hv2` · `axis: read-only harvest, no dispatch, $0`
`score_claim: false` · `promotable: false` · `pointer_delta: NONE (this unit moved no score)`

**STORES CONSULTED:** `.omx/tmp/codex_runs/*.last.txt` (797 finished arms, read ~180 in full
via four readers) · `.omx/state/codex_arm_queue.next_if_resumed.jsonl` (248 rows) ·
`.omx/research/arm_final_messages/` (343 persisted finals) · `.omx/state/canonical_task_status.jsonl` ·
`.omx/state/canonical_frontier_pointer.json` · `.omx/state/main_hot_state.md` ·
`tools/codex_arm_queue.py` · `tools/costate_digest.py` · the mz1/mz2/rfo2/mp2/ra2c receipts.
**Deliberately not loaded:** the 617 finished arms not in the four read chunks, the
`.omx/research/*.md` corpus beyond the receipts named below, and `.hypothesis`/build state.

---

## 1. The headline

**The pile contains no fireable rate lever that outranks the currently-live work, and its single
biggest advertised lever is already refuted with a receipt.** Lossless recoding is exhausted on
this vehicle. Two families remain that clear the byte bar; both fail on distortion, not rate.
One of them (semantic width distillation) is live as `wd3`. The other (pose-carrier direct
replacement) has **no live owner** and is this harvest's rank-1 recommendation.

**Read §6.0 first regardless of anything else here:** the canonical task ledger has been refusing
every write since 15:23Z today. That blocks the standing law that a follow-on exits as a task row,
which is the mechanism this entire harvest exists to protect.

---

## 2. The arithmetic every row below is ranked against (DERIVED, re-derived here)

| quantity | value | label |
|---|---:|---|
| frontier `S` | `0.15959729295498598` | MEASURED, `.omx/state/canonical_frontier_pointer.json` |
| frontier archive | `182,759 B`, sha `80d9c8c6…0178e` | MEASURED, same pointer |
| components | seg `0.029611` + pose `0.0082945765` + rate `0.1216917` | MEASURED (sum `0.1595972765`, matches to print precision) |
| `d_seg` / `d_pose` | `2.9611e-4` / `6.88e-6` | DERIVED from the components |
| gap to 0.15 | `0.0095973 S` | DERIVED |
| `S` per byte | `25/37,545,489 = 6.658590e-7` | DERIVED |
| **rate-only close** | **−14,413 B → archive ≤ 168,346 B** | DERIVED |
| seg-only close | `d_seg` `2.9611e-4 → 2.0014e-4` (−32.4%) | DERIVED |
| pose-only close | **IMPOSSIBLE** — `pose→0` buys `0.0082946 < 0.0095973`, leaves `S=0.1513027` | DERIVED |

**Cross-check that validates the whole table:** `wd2` and `rfo2` independently state the bar as
"save ≥15,157 B, reaching archive ≤168,345 B" off the *e480b* baseline (`0.1600920261571558 @
183,502 B`). My independent derivation off the *hv1* baseline gives archive ≤168,345.6 B. The two
agree to one byte because seg and pose are decode-identical between the two vehicles. **The
invariant is the absolute target `archive ≤ 168,345 B at unchanged distortion`** — quote that, not
a delta, because deltas go stale every time the pointer moves.

---

## 3. The archive decomposition every rate claim must land in

MEASURED by `mz1` direct parse, re-derived by me from
`/Volumes/VertigoDataTier/pact/ddm_mz1_model_section_rate_race/FINAL_RESULT.json`
(on the e480b 183,502 B object; hv1 differs only in the token section):

| object | bytes | share | rate `S` | recoding status |
|---|---:|---:|---:|---|
| token payload | 112,749 | 61.4% | 0.0750749 | **CLOSED** same-state |
| HPAC section | 13,619 | 7.4% | 0.0090683 | **CLOSED** fixed coder |
| semantic renderer | 34,763 | 18.9% | 0.0231473 | **CLOSED** exact recoding |
| pose carrier | 22,161 | 12.1% | 0.0147561 | **CLOSED** coarsening + rank |
| header + residual + ZIP | 210 | 0.1% | 0.0001398 | at structural floor (100 B ZIP) |
| **total** | **183,502** | 100% | **0.1221864** | — |

---

## 4. The ranked harvest

ΔS is stated against the **0.0095973** gap. "Cost to falsify" is what it takes to get a verdict.

| # | lever | measured? | plausible ΔS vs gap | cost to falsify | owner |
|---|---|---|---|---|---|
| **1** | **Pose-carrier direct replacement** — ship the six PoseNet targets directly instead of the 22,161 B carrier. `pz2` MEASURED a **1,817 B** direct packet for the six official DALI targets. Full realization saves **20,344 B = 0.013546 S = 141% of the gap**, leaving a **2.18× `d_pose` budget** (may rise `6.88e-6 → 1.50e-5` and still land sub-0.15). | rate MEASURED (`pz2`); realization **NOT** achieved | **−0.0135 S (141%)** if realized; realized-so-far **+2.47 S** | one trained output-conditioned renderer; `pz4r`'s own named cure, never built | **UNOWNED** ← the finding |
| **2** | **Scorer-aware semantic width distillation** — `wd2` MEASURED a 17,372 B saving, which **clears the 14,413 B bar by 2,959 B**. It failed only on distortion: Δ`d_seg` **7.0059×** cap, pose **623.76×** base, net ΔS **+0.984**. | rate MEASURED; distortion MEASURED and failing | −0.0116 S rate (121%) minus an unmeasured distortion cost | already funded: `wd3` build landed (`effd8ff4ef`), launch is MAIN-governed | **LIVE (wd3, #1070)** |
| 3 | `mp2` keep75-minus-keep87 differential, 182,734 B | built, **−25 B**, unscored | −1.7e-5 S (0.17%) | one advisory n600 when the scorer lane frees | MAIN (mp2 queue) |
| 4 | `qs2` compensation rung, +34 B at 0.941 flips/B | projection **−4.37e-6 S** | 0.046% | one sealed T4 row | MAIN scorer router |
| 5 | `g4` free per-pixel causal context, "89,161 B, 18.17%, zero payload" | MEASURED in **cell space only** | UNKNOWN — RGB receiver survival never established | build an RGB receiver | unowned; hypothesis only |
| 6 | `cr1` edge-graph-conditional carrier, −110,538 B (−19.22%) | MEASURED on a **support stream**, not the archive | UNKNOWN — not archive bytes | archive realization | unowned; `hr1` called it a top route |

**Rank 1 is the recommendation.** Its fire trigger has already fired and nobody consumed it:
`pz4r`'s NEXT_IF_RESUMED says *"fire trigger: direct-v6 measures `d_pose > 4e-5` … Train a
counted, resumable output-conditioned renderer."* `pz4r` direct-v6 measured `d_pose = 0.631014`.
The condition is satisfied; the successor was never spawned.

---

## 5. Dead-ends — recorded so nobody re-tries them

**Lossless recoding is EXHAUSTED on this vehicle.** Every coder family has an exact receipt:

- **Model section:** `mz1` raced 8 complete lossless representations. The incumbent split-Brotli
  q10/q11/q11 **won all 8**; best alternative +41 B (per-section Brotli q11), worst +13,196 B
  (adaptive RC64). `exact_model_section_savings = 0`.
- **The 52,566 B "serialization gap" is FALSE ATTRIBUTION.** `mz1` receipt verbatim:
  `NO_SERIALIZATION_GAP: the 52,566-byte difference compares one HPAC estimate with a
  three-object model section`. Raw HPAC is 17,996 B against a 17,991 B estimate — **5 bytes**.
  (Note the layer: `+5 B` is raw, `−4,372 B` is compressed. Both appear in `mz1`'s own text and
  they are not in conflict.)
- **ANS loses to RC64 on this vehicle.** `+6 B` on control, `+9 B` on HP3-step2, full-symbol and
  repeat-archive proof (`lp135`, `cp135`). The famous **−2,120 B ANS win was against PR130's
  *Range* coder** (`rc1`/`ap1`/`dt1`) and **does not transfer**. Do not re-propose it.
- **SMEVR** won 0 of 14 exact section races (`cp135`).
- **PPMd** lost on every unchanged section: tokens +4,618, semantic +2,263, pose +441, HPAC +600 (`rc2r`).
- **LDPC/BP syndrome coding** lost by **+540,909 B** (`rc2r`).
- **Outer ZIP deflate** +60 B; ZIP framing already at its 100 B structural floor (`hp3`, `lc2`).
- **Semantic exact re-representation:** all 38/38 tensors receiver-required; 0/38 derivable;
  dense/sparse/row-dictionary/hybrid all **+340 B** (`mz2`).

**Pose fragility is the binding constraint, not rate.** `mp2` scored all three retained `mz2`
candidates on the **current frontier vehicle** at n600 and rejected every one — the rate savings
were real and the pose cost erased them by two orders:

| candidate | archive | Δ`d_pose` | exact ΔS |
|---|---:|---:|---:|
| mixed q3/q4 | 181,936 B (−823) | +5.8376e-4 | **+0.0466762** |
| FiLM keep87 | 182,629 B (−130) | +5.3643e-4 | **+0.0442739** |
| FiLM keep75 | 182,288 B | +4.9212e-4 | **+0.0413659** |

Two structurally different mechanisms produced the same ~4.6–5× pose blow-up. `mp2`'s law:
**pose fragility is a property of the touched carrier region.** The prize needs pose ≤ +6.9%;
measured candidates are +360–400%.

**Also closed:** carrier rank truncation (`ra2c`, today — rank 4 gives `d_pose` **2,400.65×**,
`d_seg` exactly unchanged, REFUSED); `pz4a` absolute-code coarsening (500 B gross became **−2,232 B**
net after the depth-map wire); `gv2` Road↔Lane token grammar (**0/254** candidates improved);
`js8` gen-1 EC1 singletons (38 flips against 4,314 needed); `hm1` coordinate-5 drop (model −420 B
overpaid by tokens +484 B = **+60 B**); `pz3` exact residual through the frozen carrier (**+3,068 B**).

---

## 6. Apparatus defects found while harvesting

These are cheap and they cause the exact signal loss this harvest exists to repair.

0. **P0 — THE CANONICAL TASK LEDGER IS DEAD AND HAS BEEN SINCE 15:23Z TODAY.**
   `.omx/state/canonical_task_status.jsonl` fails strict load, so
   `tac.canonical_task_status.writer.register_task` **and** `tools/canonical_task_status.py update`
   both raise and **refuse every write**. Cause: line 548 — the last row — is an
   `event_type=completion` / `status=completed` row appended by `ddm_pv1` at
   `2026-08-16T15:23:27.639880Z` for task `1079_pv1_pose_floor_and_admission_bar_audit_20260816`,
   and **no `registered` row for that task exists anywhere in the file**. The ledger has accepted
   **zero writes since**. Every arm that has tried to file a follow-on for the last several hours
   has been silently refused — this is an orphan generator sitting directly on top of the law that
   a follow-on must exit as a task row rather than as prose.

   **I did not repair it, deliberately.** The loader validates in file order and does not sort by
   timestamp, so appending cannot fix an ordering defect; and `VALID_TRANSITIONS`
   (`contract.py:147`) forbids `pending → completed`, so a faithful in-order repair needs **both**
   a `registered` and an `in_progress` event **that never happened**. Manufacturing two lifecycle
   events into an append-only custody ledger is history fabrication. That is MAIN's call, not a
   harvest arm's. Repair options, the self-protection owed (`register_task` should refuse a
   non-registration append for an unknown `task_id` at **write** time, so this cannot enter the
   ledger at all), and my five task rows staged for one-command replay are in
   **`.omx/research/ddm_hv2_task_rows_pending_ledger_unblock_20260816.json`**.

1. **There is no read-receipt for an arm final.** `.done` receipts get a `.consumed.json`
   (42 exist); finals get nothing. So "how many finals are unread" is **structurally
   unanswerable from disk** — I will not invent a number. The `.consumed.json` pattern already
   exists and should be mirrored onto the final.
2. **454 of 797 finished finals exist only in gitignored `.omx/tmp/`.** Only 343 are persisted to
   `.omx/research/arm_final_messages/`, and **that directory is untracked** (`git ls-files` → 0,
   `git check-ignore` → rc=1, i.e. not ignored, just never added). One `git clean` loses them.
3. **The costate digest's NEXT_IF_RESUMED reader never prints.** `tools/costate_digest.py:2130`
   sits inside the `if ddm_live:` branch at line 2029. I verified `ddm_live` is `True` and the
   reader returns a correct 248-row line when called directly, yet the string
   `arm-next-if-resumed` appears **0 times** in the 76-line digest output. The reader is wired and
   silent — the vacuity-equals-pass class.
4. **Stale bars in live fire-orders.** `rx1`, `rx2`, `sp2` cite `archive < 186,269 B`; that is
   **3,510 B ABOVE** the current frontier, so a candidate could pass its own bar while being worse
   than what we ship. `mz1` cites `15,153 B`, which `wd2`/`rfo2` corrected to `15,157 B`. Both go
   away if fire-orders quote the invariant **`archive ≤ 168,345 B`** instead of a delta.
5. **The prior harvest's orphan rows are stranded on dead vehicles.** `mh1_orphan_*` fire
   conditions say "re-measure on TR1 at n600". TR1 is `S≈0.75 @ 357,836 B`; the frontier is
   `0.1596 @ 182,759 B`. As written they are unfireable.

---

## 7. What this unit did NOT do

It moved no pointer. It ran no scorer, no dispatch, no training. Every number above is either
re-derived from a receipt I opened or labelled as a pointer I could not re-derive. Specifically
**not re-derived at source:** `g4`'s 89,161 B, `cr1`'s −110,538 B, and `pz2`'s 1,817 B were read
from the arm finals and their receipts were not opened by me — rank 1 rests on `pz2`'s 1,817 B, so
**that number must be re-derived from
`/Volumes/VertigoDataTier/pact/ddm_pz2_pose_representation_20260810_v3/PZ2_MEASUREMENT_RECEIPT.json`
before any launch is routed on it.** That re-derivation is written into the task row as a gate.

`verdict_scope: FORMULATION` for the lossless-recoding closure (exhausted across every coder
family raced on *this vehicle's* current sections; a changed trained state reopens it — `rc2r` and
`rx1` both say so explicitly). `verdict_scope: INSTANCE` for the individual candidate rejections.
