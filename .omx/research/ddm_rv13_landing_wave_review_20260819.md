# ddm_rv13 — fresh-eyes adversarial review of the 2026-08-19 landing wave

Date: 2026-08-19 · Arm: `ddm_rv13` · Posture: **READ-ONLY** (doc fixes only; no `.py`, no
receipts, no seals) · **Score claim: false** · **Pointer moved: false**

`verdict_scope`: **INSTANCE** — the two pointer moves (`ddm_ck2`, `ddm_to1`), the `ddm_rc1x`
two-role recipe, and the corrections trail landed 2026-08-19. Findings about the mirror
schema and the dispatch-claim auto-closer are **CLASS** and are marked so.

STORES CONSULTED: run-artifacts (`ddm_ck2/{seal,t4_row_r1,t4_row_r2,compile,probe,overlay,
generations}`, `ddm_to1/{seal,t4_row_r1,compile,generations,advisory}`, both
`MODAL_REMOTE_RESULT.json`, both `FIRE_MANIFEST.json`, `CK2_CUSTODY_MANIFEST`,
`TO1_RETENTION_MANIFEST`, `ddm_ck1/t4_row/MODAL_REMOTE_RESULT.json`,
`experiments/results/modal_auth_eval_mirror/`) · memories (bounds-add exact margins;
`m53` negative-existence scope; `m89` task-ledger split; corrections-land-in-bodies;
concavity-has-a-sign; price-the-ceiling) · ledgers (`active_lane_dispatch_claims.md`,
`modal_call_id_ledger.jsonl`, `canonical_frontier_pointer.json`,
`canonical_task_status.jsonl`) · git (`d4c7f56ddd`..`16ee5e599a`) · the round-12 cure
commits `8a059d5932` and `6e976eeafd`.

---

## Verdict in one line

**Both pointer moves are REAL.** Every headline number reproduces exactly from the raw
receipt digits, both identity controls are non-vacuous and verified by hash, and the
shipped runtime trees are pristine. The defects are in the **reporting and ledger
surfaces around** the moves, not in the moves. One of them is CRITICAL because it sits
in an actuator input and is demonstrably false.

---

## Findings

| # | Severity | Verdict | Finding |
|---|---|---|---|
| F1 | **CRITICAL** | CONFIRMED-DEFECT | A sealed fire-order asserts a negative-existence claim I disproved by direct hash — no correction |
| F2 | MEDIUM | CONFIRMED-DEFECT | Round-12 F1 genus **recurs**: ck2 memo divides the delta by ONE row's bound (131.1× vs the correct 65.6×) |
| F3 | MEDIUM | CONFIRMED-DEFECT | to1 seal F4: listed addends sum to **half** the stated total; "unequal per row" is false for this row-pair |
| F4 | MEDIUM (CLASS) | CONFIRMED-DEFECT | The pointer's input mirror drops runtime-tree custody; one archive sha carries two contradictory contest-CUDA scores |
| F5 | MEDIUM | CONFIRMED-DEFECT | Two canonical status surfaces contradict themselves inside one file (stale present-tense pointer) |
| F6 | MEDIUM | CONFIRMED-DEFECT | An always-loaded memory instructs future arms to rebase onto a twice-superseded base |
| F7 | MEDIUM (CLASS) | CONFIRMED-DEFECT | to1 left a phantom ACTIVE dispatch claim; ck2's terminal row states a false reason |
| F8 | MEDIUM | CONFIRMED-DEFECT | ma1 and fx2 memos still carry refuted claims; every correction lives in a different file |
| F9 | LOW | CONFIRMED-DEFECT | to1 re-derived its 8dp bound from the rounded d_pose instead of the harness-published field |
| F10 | LOW | CONFIRMED-DEFECT | Seal `runtime.sha256` and receipt `expected_runtime_tree_sha256` are different definitions, undocumented |
| F11 | LOW | CONFIRMED-DEFECT | `#1129`–`#1131` are cited as authority but absent from the repo task ledger (`m89`) |
| F12 | LOW | CONFIRMED-DEFECT | Pointer CUDA leg's null `archive_bytes` was reproduced by the to1 refresh, not cured |
| F13 | MEDIUM | CONFIRMED-DEFECT | The rc64 role taxonomy is incomplete: **four** distinct bodies wear the name, not two |
| R1 | — | **REFUTED-SUSPICION** | AppleDouble does **not** contaminate the shipped runtime trees |
| R2 | — | **VERIFIED-CLEAN** | All headline arithmetic reproduces exactly from raw digits |
| R3 | — | **VERIFIED-CLEAN** | Both identity controls are non-vacuous and hash-verified |
| R4 | — | **VERIFIED-CLEAN** | ck2 build is deterministic across two independent runs |
| R5 | — | **VERIFIED-CLEAN** | The rc64 two-role bodies exist exactly as claimed |
| R6 | — | **VERIFIED-CLEAN** | to1's F5 decode identity is discharged at the strongest available level |

---

## F1 — CRITICAL. A sealed actuator input carries a false negative-existence claim

`.omx/research/ddm_fx2_t4_sealed_fire_order_20260818.json:19-27`:

> `"detail": "experiments/ddm_pq2_compress_e2e.py's default recipe pins rc64_source_sha256 =`
> `5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6. No file on either SSD`
> `matches it: all 252 .c files under the pact tree were hashed …"`

**I disproved this by direct hash:**

    /Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book/src/cpr1_sub4/entropy/rc64_backend.c
    5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6   12,222 B   dated Aug 10 12:19

The file matches the pin exactly, and it has been on disk since 2026-08-10 — eight days
before the fire order declared it nonexistent. `ddm_rc1x` reached the same conclusion; this
review reproduces it independently from the filesystem rather than from rc1x's prose.

Correction markers in that file: **zero** (grepped `corrigend|CORRECTED|WITHDRAWN|SUPERSED|
retract|rc1x` — no hits).

**Why this outranks everything else in the wave.** The other stale surfaces are prose a
human reads. This is a **sealed fire order** — a machine-shaped actuator input whose whole
job is to gate a fire. It says the byte-close is blocked at input verification and names a
`fix` that is unnecessary. Any arm or tool that parses it will decline work that is
already unblocked. `ddm_rc1x` §7 item 3 logged exactly this as owed; it has not been
actioned. The DAG corrigendum *names* this file as stale but never touched it.

**MAIN action:** append a correction banner to the sealed JSON (append-only per Catalog
#110/#113 — do not mutate the original claim), and re-open `ddm_fx2`'s and `ddm_ma1`'s
"blocked" status. I did not edit it myself: a sealed fire order is an actuator artifact,
not documentation, and it sits outside a reviewer's doc-only remit.

---

## F2 — MEDIUM. The round-12 F1 genus recurs in the very next landing

`.omx/research/ddm_ck2_container_plane2_eleventh_move_20260819.md:27`, in the candidate's
own summary table:

> `| vs ck1's report-8dp error bound (3.336608e-6) | **131.1×** |`

That divides a **delta** by **one row's** bound. Round 12 cured exactly this
(`8a059d5932`, `6e976eeafd`): *"a delta carries BOTH rows' 8dp bounds, which add."*

Recomputed from raw digits:

    net dS                     4.374693e-04
    / one row's bound          3.336608e-06  ->  131.1x   <- what the memo says
    / summed two-row bound     6.673217e-06  ->   65.6x   <- correct

The overstatement is **exactly 2.00×** — the same signature round 12 named. The ck2 **seal**
gets it right (`F4 … 65.6x the SUMMED two-row … bound`), so the seal and the memo disagree
with each other. **The sign is determinate either way** — 65.6× is decisive — so no verdict
changes. Only the stated margin is wrong.

Worth stating plainly: a cure whose entire content is *"divide by the bound"* has to be
applied to the next row that quotes a bound, and it was not. The cure landed in the four
files that carried the old error and was not carried forward into the file written hours
later.

**Fixed by this review** (append-only correction block, matching the campaign's own format).

---

## F3 — MEDIUM. to1's seal F4 lists addends that sum to half its stated total

`/Volumes/APDataStore/pact/ddm_to1/seal/CANDIDATE_SEAL_to1_r1.json`, falsifier F4:

> `10.48x the SUMMED two-row report-8dp error bound 6.672304e-06 (bounds ADD for deltas and`
> `are unequal per row: seg 5.000000e-07, pose 2.836152e-06)`

Two problems in one parenthetical:

1. **The addends do not sum to the total.** `5.000000e-07 + 2.836152e-06 = 3.336152e-06`,
   which is **half** of `6.672304e-06`. The listed pair is the *axis* decomposition of ONE
   row; the total is the TWO-row sum. A reader auditing by addition gets half. Commit
   `6e976eeafd` is titled *"the sa3 summed bound must equal its stated addends"* — this is
   that requirement, unmet, two days later. The canonical format (`keep01`, `sa3`) lists the
   **two rows'** bounds and sums them.
2. **"unequal per row" is false here.** `d_pose` is identical in both rows (7.77e-06), so the
   two rows' bounds are exactly **equal** (3.336152e-06 each). The phrase was correct for
   `keep01`/`sa3`, where `d_pose` genuinely differed between rows; it was carried into a
   row-pair where it does not hold. That is the `cross-regime constant transfer` genus
   applied to language rather than to a number.

**The total and the multiple are correct.** `6.991519e-05 / 6.672304e-06 = 10.478` → 10.48×.
Only the parenthetical is wrong.

The **to1 memo** (`ddm_to1_tail_override_twelfth_move_20260819.md:99-105`) states this
correctly — *"Per row: seg 5.000e-07, pose 2.836e-06, total 3.336e-06; two rows
6.672304e-06"* — so the memo is clean and the seal is not. Seals are immutable custody; I
did not edit it. This is recorded so the next seal writer uses the two-row form.

**Note the bound is conservative, not binding.** to1's F5 proves the decoded output
byte-identical, so the distortion legs cancel *exactly* and the true bound on the
distortion delta is **zero**. The seal says this itself. Quoting a conservative bound
understates the margin, which is the safe direction.

---

## F4 — MEDIUM, CLASS. The pointer's input surface cannot tell two receivers apart

`experiments/results/modal_auth_eval_mirror/` is what `tac.frontier_scan` reads and what
auto-moved the pointer (*"Pointer auto-moved with ZERO manual steps"*). Full census — 4 rows:

| rows | carrying `runtime_tree_sha256` |
|---:|---:|
| 4 | **0** |

And one archive sha appears twice, with contradictory scores:

| file | score | archive sha | bytes | lane |
|---|---:|---|---:|---|
| `contest_auth_eval_ck1_r4_t4_r3_20260818T233350Z.json` | **0.15710198138050818** | `35c318d5…` | 177,182 | `ddm_ck1_composed_r4_t4` |
| `contest_auth_eval_ck1_t4_r1.json` | **79.40216174747616** | `35c318d5…` | 177,182 | `ddm_ck1_composed_rebased_t4` |

Both source receipts are `passed: true`, `validation_errors: []`, `score_claim: true`,
`gpu_t4_match: true`, `n_samples: 600`, `evidence_grade: contest-CUDA`. The **only**
differing input is the runtime tree (`71c75468…` on the 79.40 row). Both rows are honest;
the score is a function of `(archive, runtime tree)`, and the mirror keys on the archive
alone.

**Why it matters.** CLAUDE.md's apples-to-apples discipline requires an *"IDENTICAL
`inflate.sh` runtime tree"* for any score comparison, and the mirror is the one surface on
the pointer's path that drops it. The pointer is **currently correct** — `min()` selection
happens to reject the broken row, because a broken receiver can only *raise* distortion and
therefore *raise* S. So today's safety is accidental, not designed. The real exposure is the
reverse case the min-rule cannot catch: a receiver that scores *low* for reasons the archive
does not license. The runtime-tree sha is precisely the custody that would refuse it.

**MAIN action:** carry `expected_runtime_tree_sha256` into
`modal_auth_eval_anchor_mirror.v1` and key rows on the `(archive, runtime_tree)` pair. This
is a `.py`/schema change, outside my remit.

---

## F5 — MEDIUM. Two canonical status surfaces contradict themselves inside one file

**`reports/latest.md:12`** — present tense, the word "CURRENT", two moves behind:

> `**CURRENT effective_frontier [contest-CUDA T4, n600]: S 0.15710198138050818**`
> `archive sha 35c318d5… (ck1 … tenth pointer move, 2026-08-19 …)`

The same file's generated table at line 95 is already right:
`| **[contest-CUDA T4]** | **0.1565945969** | 50e561454b23 | …`

**`.omx/state/current_focus.md:762`** — same shape, stamped with today's date:

> `**Pointer honesty (updated 2026-08-19):** effective competitive bar is now OUR OWN row`
> `0.15710198138050818 [contest-CUDA T4] (ck1 composed, tenth pointer move …)`

The same file is correct at lines 915-918.

This is the `corrections-land-in-bodies-headlines-keep-the-stale-number` law firing on the
two surfaces most likely to be read first. In both files the **generated** section refreshed
and the **hand-written banner** did not. **Fixed by this review.**

Not flagged, correctly: dated trajectory rows and base declarations (the ck2 memo's
*"Base: the ck1 pointer"*, the ck2 fire-order delta table, the ck1 restage records, the
terminal ck1 dispatch row). Those are history and must stay.

**No surface presents ck2 as the current pointer.** Hot state, the p0 ledger, and
`canonical_frontier_pointer.json` all carry to1 correctly.

---

## F6 — MEDIUM. A memory tells future arms to rebase onto a superseded base

`~/.claude/projects/-Users-adpena-Projects-pact/memory/cpu_to_cuda_seg_transfer_has_no_law_20260819.md:31-33`,
under *"How to apply"*:

> `- Next composed candidates rebase onto the ck1 pointer base`
> `  (sha 35c318d5…, 177,182 B).`

This is a forward-looking instruction, not a dated row, and memories are the always-loaded
recall surface. An arm that follows it rebases two moves behind and burns a byte-close.
The lesson the memory carries (*do not trust a CPU-modelled seg delta*) is sound and
unaffected; only the base pointer is stale. **Fixed by this review** — the lesson preserved,
the base corrected.

---

## F7 — MEDIUM, CLASS. Dispatch-claim hygiene failed on both fires

**to1 is a phantom ACTIVE claim.** `.omx/state/active_lane_dispatch_claims.md:9` is the only
to1 row and reads `active_modal_auth_eval_spawning`. The job finished: the Modal ledger
records `harvested`, `rc: 0`, 1360.66 s at `04:38:13Z`, and the pointer refreshed at
`04:38:58Z`. CLAUDE.md binds: *"append a terminal row … Do not leave completed jobs as
phantom active claims."* **The twelfth pointer move has no terminal dispatch row.**

**ck2's terminal row states a false reason.** Line 10 closes with
`stale_superseded_reconciled_no_live_call` and the note *"reconcile reported an active Modal
claim with ZERO live ledger call_ids (the provable-phantom condition)."* But call
`fc-01M0BVWYZWS9VY23G5Z24EYG0Q` was dispatched under that exact `lane_id` at `01:59:14Z` and
harvested `rc=0` at `02:21:08Z` — **1 h 52 m before** the auto-close fired. The eleventh move
has no `completed_contest_cuda_exact_eval_harvested` row, unlike ck1, which does.

**Root cause, and it will recur.** The `harvested` ledger rows carry `"lane_id": null`. A
lane-keyed reconcile cannot see them, so every completed fire looks like a phantom to the
auto-closer. Per the no-manual-dispatch binding the cure belongs in
`tools/fire_modal_auth_eval.py`, not in a hand-edit — so I did not hand-edit the ledger.

Related, systemic: the last 20 `harvested` rows all carry `score: null`, `score_axis: null`,
`archive_sha256: null`. The ledger records *that* a harvest happened, never *what it scored*.

---

## F8 — MEDIUM. Refuted claims sit uncorrected in the files that made them

Every correction to the rc64 diagnosis lives in a **different file** from the claim:

| file | still asserts | correction present |
|---|---|---|
| `ddm_fx2_t4_sealed_fire_order_20260818.json:19-27` | *"No file on either SSD matches it"* | **none** (F1) |
| `ddm_fx2_model_axis_all_sections_20260818.md:336-341` | same, plus `:461` routes to the "blocker" | **none** |
| `ddm_ma1_model_axis_miss_cost_20260819.md:226-229, 290-292` | *"158 copies … neither matches … Clearing this pin is the first owed step"* | **none** |

The correction lives in `ddm_rc1x_rc64_recipe_fix_20260819.md` and in the DAG corrigendum.
A reader who opens ma1 or fx2 directly sees only the refuted claim. Catalog #110/#113 asks
for an append-only superseding row **in the superseded surface**, not merely elsewhere.

**Unreconciled denominators.** Three surfaces state three counts for what reads as the same
sweep, and none states its scope alongside the number: ma1 *"158 copies"*, fx2 *"252 `.c`
files"*, rc1x *"232 copies"*. Each is a different scope. This is the `m53` cure only
half-applied: rc1x named the genus but the campaign still publishes bare counts.

**The reconciled denominator, with its scope stated** (this review, measured):

    scope: /Volumes/APDataStore/pact + /Volumes/VertigoDataTier/pact + /Users/adpena/Projects/pact
    239 files named rc64_backend.c (AppleDouble excluded), 4 distinct contents

**MAIN action** (append-only; I left these to MAIN because two are actuator-adjacent and the
third is a sibling arm's memo of record).

---

## F13 — MEDIUM. Four bodies wear the name, not two — the genus recurs inside its own cure

`ddm_rc1x` opens: *"Two distinct C bodies wear the name `rc64_backend.c`."* Measured over the
full three-tree scope above, there are **four**:

| sha256 (16) | bytes | copies | role | where |
|---|---:|---:|---|---|
| `05839d1416e68a49` | 5,638 | **235** | shipped receiver, decoder-only | every archive's `runtime/entropy/` |
| `1941923a94e4e0a1` | 14,825 | **2** | **checkpoint-extended encoder — unnamed by the wave** | `VertigoDataTier/pact/ddm_rc64p_20260810/{runtime,runtime_optimized}/` |
| `b249b77bb06a27c8` | 22,179 | 1 | foreign intake (PR138 `opal_v1`) | `APDataStore/pact/ddm_pq2/intake/pr138_src/…` |
| `5c75e2c70b89f148` | 12,222 | **1** | **encoder — the pin** | `VertigoDataTier/pact/pr135_intake_20260810/…` |

rc1x describes the 14,825 B checkpoint build as `rc64_backend_checkpoint.c` — but on disk that
exact content also sits under the **plain** name `rc64_backend.c`, twice. So the
name-without-role hazard rc1x correctly diagnosed is **one body worse than rc1x measured**,
and the extra body is an *encoder-class* one — precisely the class whose absence the false
blocker (F1) turned on. A name-keyed search that stops at the first `rc64_backend.c` it finds
can now land on any of four bodies.

This is the round-12 pattern repeating: a cure that does not fully cover its own genus. It
does not weaken rc1x's verdict — the roles it names are real and its byte-close stands.

**One thing this measurement corroborates.** rc1x's P1 control claims that subtracting the
2,603 B checkpoint/resume block from the retained 14,825 B body yields the pin. Independently:
`14,825 − 2,603 = 12,222` — **exactly** the encoder body's size. P1 verified from the
filesystem, not from rc1x's prose.

**MAIN action:** extend the role table to four bodies and queue the sibling-class sweep rc1x
already opened (*"any other recipe/driver pin whose file is named generically and located
off-tree"*), which remains open.

---

## F9 — LOW. to1 re-derived a bound the harness already publishes

Both receipts publish the field directly:

    "report_8dp_score_worst_case_abs_error_bound":       3.336608391523776e-06
    "report_8dp_pose_score_worst_case_abs_error_bound":  2.836608391523776e-06
    "report_8dp_seg_score_worst_case_abs_error_bound":   5e-07

to1 instead recomputed the pose bound from the **rounded** `d_pose` (7.77e-06), getting
`2.836152e-06` and a two-row total of `6.672304e-06`. The harness-published two-row total is
`6.673217e-06`. The multiple is `10.478` either way and rounds to **10.48×**, so nothing
downstream changes — but the campaign's own receipt field is the authority and a
hand-recomputation that disagrees with it in the 4th significant figure should not be the
number quoted in a seal. ck2's seal used `~6.673e-06`, which **is** consistent with the
published field.

---

## F10 — LOW. Two runtime hashes over one directory, no note that they differ

| surface | ck2 | to1 |
|---|---|---|
| seal `runtime.sha256` | `95389ea3…` | `5c165a07…` |
| receipt `expected_runtime_tree_sha256` | `9ebca4f9…` | `cf49cd75…` |

These are different hash definitions over the same directory, so a reviewer cannot
cross-verify the sealed tree against the scored tree by comparing them, and the mismatch
invites a false alarm. I tried six plausible conventions (with/without `archive.zip` × three
separators); none reproduced either value, so the definitions are internal.

**The custody chain is nonetheless intact**, by a different route: `FIRE_MANIFEST.json` pins
`seal_path` + `seal_sha256` (`10b349c5…`, matching the seal's own field) + `runtime_dir`, and
records `seal_validation.verdict = SEAL_VALID` and `stage3b_seal.verdict = CONSISTENT`. The
gap is documentation, not custody. One sentence in the seal naming the two hash definitions
would close it.

---

## F11 — LOW. Cited task numbers do not exist in the repo ledger

`#1131` is cited as authority in `sub015_DAG…:28917`, `ddm_ck2_fire_order_draft…:105`, and
`operator_p0_ledger.jsonl:536`. Measured on `.omx/state/canonical_task_status.jsonl`:
**144 rows, max `task_id` 1029; 1129, 1130 and 1131 are all absent.**

This is `m89` — the harness TaskList is not the repo ledger, and arms see only the repo. The
`#1128`–`#1131` block is unresolvable to any arm reading this tree. The cure `m89` names —
cite content, never a bare id — is not applied on these three surfaces.

---

## F12 — LOW. The pointer's null `archive_bytes` was reproduced, not cured

`.omx/state/canonical_frontier_pointer.json`: the CUDA leg carries
`"archive_bytes": null`, `"lane_id": null`, `"measured_at_utc": null`, while the CPU leg
populates `archive_bytes`. `ddm_rc1x` §5 measured the consequences —
`tac.candidate_seal.read_frontier_archive_identity()` **refuses**, `ddm_pq2_compress_e2e`'s
default expected-archive resolution **refuses**, and
`test_candidate_seal_pin_consistency.py::test_live_pointer_supplies_a_usable_bar` **fails on
HEAD** independent of that arm. rc1x flagged it against the ck2 refresh; **the to1 refresh
reproduced the same gap.** A known defect that survives one more refresh is a defect the
refresher does not guard.

---

## What is VERIFIED CLEAN — stated as plainly as the defects

**R2 — the arithmetic is exact.** Recomputed from the raw receipt digits, not copied:

| quantity | recomputed | claimed |
|---|---|---|
| ck2 S | `0.1566645120483069` | identical |
| to1 S | `0.15659459685822907` | identical |
| ck2 rate term | `0.1175407516998913` | identical |
| to1 rate term | `0.11747083650981346` | identical |
| ck2 net ΔS | `-4.374693322012624e-04` | `-4.374693e-04` |
| to1 net ΔS | `-6.991519007781832e-05` | `-6.991519e-05` |
| ck2 bar multiple | `124.991` | `125.0` |
| to1 bar multiple | `19.976` | `19.98` |
| to1 bound multiple | `10.478` | `10.48` |

Both moves are **pure rate**: −657 B and −105 B reproduce the ΔS exactly with zero
distortion contribution. Both T4 receipts are `passed: true`, `validation_errors: []`,
`gpu_t4_match: true`, `n_samples: 600`, `evidence_grade: contest-CUDA`.

**R3 — both identity controls are non-vacuous, and I hashed them.**

    ck2  candidate_none          35c318d541d70370   177,182 B  == the ck1 pointer, byte-identical
    ck2  candidate_plane2        e3d5e212bb54ba66   176,569 B
    ck2  candidate_plane2_carrier2  0aa1cada2ca79ad4  176,525 B  == the shipped candidate
    to1  control_override_off    0aa1cada…          176,525 B  == the ck2 pointer, byte-identical
    to1  candidate_tail_override 50e56145…          176,420 B  == the shipped candidate

The ck2 ladder reproduces the memo's decomposition exactly: 177,182 → 176,569 is **−613 B**
(semantic) and 176,569 → 176,525 is **−44 B** (carrier), totalling **−657 B**. Each rung
produces different bytes, so no rung is inert.

**R4 — the ck2 build is deterministic.** `compile/build_r1/` and `compile/verify_r2/` agree
sha-for-sha across all six rungs, from two independent runs. That is the determinism repeat
the always-keep-the-payload rule asks for, and it is present without being claimed.

**Payload retention is exemplary.** ck2 retains the full control ladder plus the section
bodies and both `expected_codes.npy` sets — per-candidate payloads, not just the winner's.

**R6 — to1's F5 is discharged at the strongest available level.** `0.raw` is byte-identical
to the pointer's at 3,662,409,600 B, equal by sha256 **and** by direct `cmp`, while the
shipped `token_stream` differs (`5b09fd78…` → `15054e5d…`) and the decoded token field
matches (`9ba2e52b…`). Different stream, same field: that is what makes the control
non-vacuous, and the arm says so itself.

**R5 — the two rc64 role bodies are exactly as claimed**, verified from the filesystem:
encoder 12,222 B `5c75e2c70b89f148…` on VertigoDataTier dated Aug 10; shipped receiver
5,638 B `05839d1416e68a49…` in ck2's runtime tree.

**rc1x's control ladder is the best instrument work in the wave.** P1–P5 plus A–E, where P5
(a flipped payload bit must break the decode) proves P3 non-vacuous, and P4 (the positive
control) located a fault in the arm's own harness rather than in the bodies. The arm also
retracts its own over-claim in §5. That is the L3 verdict-clearance discipline actually
practised.

### R1 — REFUTED SUSPICION: AppleDouble does not reach the shipped trees

The round-12 concern does **not** recur where it would matter. Measured:

| tree | `._*` files | real files | seal says |
|---|---:|---:|---|
| `ddm_ck2/generations/ck2_plane2_r1` | **0** | 33 | 33 |
| `ddm_to1/generations/to1_tail_override_r1` | **0** | 34 | 34 |

The custody *parents* do carry them (172 in `ddm_ck2`, 52 in `ddm_to1`) — unavoidable
ExFAT sidecars — and neither manifest counts them. The shipped trees are pristine and the
file counts match the seals exactly. **Suspicion refuted.**

---

## Axis E — the assumption challenge (mandatory)

**The shared assumption.** Every move in this class — ck2, to1, and the sz1/fx2 lineage
behind them — holds the **decoded state bit-identical**, so both distortion legs are zero
*by construction* and the move is pure rate. It is a genuinely good assumption: it removes
all measurement risk, it made tonight's two moves un-falsifiable in the strong sense, and
it is why the tenth move's CPU-modelled seg miss cannot recur here.

**What it costs.** Bit-identity buys 100% retention of the rate credit — but only from the
container's entropy slack, and ck2's own memo already records that pool as *exhausted at
−657 B on this base* (`32e632aa9d`). The measured trend is decisive:

| | credit |
|---|---:|
| ck2 (eleventh move) | −657 B |
| to1 (twelfth move) | −105 B |
| **the whole night** | **−762 B**, ΔS `-5.074e-04` |

Against the goal:

    distortion floor (seg+pose), unchanged all night   0.03912376
    bytes needed for S = 0.15 at that floor            166,516 B
    further PURE-RATE credit required from 176,420 B     9,904 B
    gap to 0.15                                       0.00659460  =  13.0x the whole night
    to1-sized moves still required                         ~94

**Pure-rate container work cannot reach the goal.** Not "is slow" — cannot, at any plausible
move count, because the pool it draws from is measured and nearly empty.

**Would violating it unlock anything?** The campaign has already measured the alternative
and, I think, mis-read it. Compensated edits that *do* move the decoded state retained
22.8% (sa3) and 10.5% (keep01) of their rate credit — which reads as a bad exchange rate
against bit-identity's 100%. But the retention is a fraction of a **much larger pool**, and
the campaign's own `concavity-has-a-sign` law says retention **rises with mass**
(10.5% → 12.6%). The `√` term that punishes you for *buying* pose helps you when you *pay*
it, and the effect grows with the size of the move.

So the two axes are not comparable as rates. They are comparable only as
**credit × retention**, and nobody has priced the compensated axis's **ceiling** — which the
campaign's own `price-the-ceiling-first` and `the-counted-byte-is-not-fungible` laws say is
the first thing to measure, not the last. A 20,000 B compensated move retaining 15% beats
94 perfect container moves that do not exist.

**The concrete challenge:** the next unit should measure the *ceiling* of the compensated
axis on the to1 body — the largest distortion-buying edit whose retention still clears the
bar — rather than hunt the container axis for a thirteenth −100 B move. Bit-identity should
be recognised as what it is: a risk-elimination choice with a measured and nearly exhausted
budget, not a free property of a good candidate.

I do not claim the compensated axis wins. I claim its ceiling is **unmeasured**, that the
container axis's ceiling **is** measured and too small, and that the campaign is
concentrating on the axis it has already proved cannot get there.

---

## What needs MAIN action

| # | Action | Why MAIN |
|---|---|---|
| F1 | Append a correction banner to `ddm_fx2_t4_sealed_fire_order_20260818.json`; re-open fx2/ma1 "blocked" status | Sealed actuator artifact, not documentation |
| F4 | Carry `expected_runtime_tree_sha256` into the mirror schema; key on `(archive, runtime_tree)` | `.py`/schema change |
| F7 | Fix `lane_id: null` on harvested rows in `tools/fire_modal_auth_eval.py`; close to1's phantom claim through the tool | No-manual-dispatch binding forbids a hand-edit |
| F8 | Append-only supersession rows inside the ma1 and fx2 memos; reconcile the three copy-counts with stated scopes | Sibling arms' memos of record |
| F9 | Seal writers consume `report_8dp_*_bound` from the receipt instead of recomputing | Convention for the next seal |
| F12 | Populate the pointer's CUDA-leg `archive_bytes` / `lane_id` / `measured_at_utc`; the failing test is on HEAD | `.py` change, one test red |
| F3 | Next seal states the **two rows'** bounds and checks they sum to the stated total | Seals are immutable; convention only |
| F11 | Cite content, not bare `#1129`–`#1131` | Ledger split, `m89` |
| E | Price the ceiling of the compensated axis before the next container move | Direction |

Fixed by this review (documentation only): **F2** (ck2 memo margin), **F5** (both stale
headlines), **F6** (memory rebase base).
